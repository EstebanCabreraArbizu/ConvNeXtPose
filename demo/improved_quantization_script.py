#!/usr/bin/env python3
"""
Enhanced calibration script for quantizing ConvNeXtPose model using SparseML
Based on training code for Human3.6M dataset
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import cv2
from tqdm import tqdm
from  pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.extend([
    str(PROJECT_ROOT / 'main'),
    str(PROJECT_ROOT / 'data'),
    str(PROJECT_ROOT / 'common')
])
from config import cfg
from base import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Quantize ConvNeXtPose model to INT8")
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0', help='GPU IDs to use')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model .pt file')
    parser.add_argument('--output', type=str, default='convnextpose_int8.onnx', help='Output quantized model path')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of calibration samples')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for calibration')
    parser.add_argument('--skip_verification', action='store_true', help='Skip DeepSparse verification')
    args = parser.parse_args()
    
    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

def collect_calibration_data(dataset, num_samples, batch_size=1):
    """Collect calibration data from the dataset with stratified sampling"""
    
    # Create subset of dataset for calibration
    if len(dataset) > num_samples:
        # Use stratified sampling instead of random sampling for better representation
        indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
        dataset = Subset(dataset, indices)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),  # Ensure we don't use more workers than CPU cores
        pin_memory=True
    )
    
    # Collect input data for calibration
    print(f"Collecting {num_samples} calibration samples...")
    calibration_data = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_img = batch[0]  # First item is the input image
            # Move to CPU and convert to numpy
            input_np = input_img.cpu().numpy()
            calibration_data.append(input_np)
    
    # Concatenate all batches
    calibration_data = np.vstack(calibration_data)
    print(f"Collected calibration data shape: {calibration_data.shape}")
    return calibration_data

def export_to_onnx(model, dummy_input, output_path):
    """Export PyTorch model to ONNX format with optimizations"""
    print(f"Exporting model to ONNX: {output_path}")
    
    # Verificar si el modelo es TorchScript
    is_torchscript = isinstance(model, torch.jit.ScriptModule)
    
    if is_torchscript:
        print("Detectado modelo TorchScript, usando método de exportación específico...")
        try:
            # Para modelos TorchScript, usar torch.onnx._export_to_zip
            from torch.onnx import _export_to_zip
            _export_to_zip(
                model,
                dummy_input,
                output_path,
                verbose=True,
                opset_version=13
            )
        except (ImportError, AttributeError):
            # Alternativa: trazar el modelo primero
            print("Método alternativo: trazando modelo antes de exportar...")
            traced_model = torch.jit.trace(model, dummy_input)
            torch.onnx.export(
                traced_model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
    else:
        # Método estándar para modelos PyTorch normales
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False,
            keep_initializers_as_inputs=False
        )
    
    # Verificar el modelo ONNX
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("¡Modelo ONNX verificado exitosamente!")
    except ImportError:
        print("Paquete ONNX no encontrado. Omitiendo verificación del modelo ONNX.")
    except Exception as e:
        print(f"Falló la verificación del modelo ONNX: {e}")
    
    print(f"Modelo exportado a {output_path}")

def quantize_model(model_path, calibration_data, output_path):
    """Quantize ONNX model using SparseML with calibration data"""
    try:
        from sparseml.onnx.utils import quantize_model_post_training
    except ImportError:
        print("SparseML not found. Installing...")
        os.system("pip install sparseml[onnx]")
        from sparseml.onnx.utils import quantize_model_post_training
    
    print(f"Quantizing model to INT8: {output_path}")
    
    # Save calibration data temporarily
    cal_data_path = "temp_calibration_data.npy"
    np.save(cal_data_path, calibration_data)
    
    # Create quantization recipe
    recipe = """
    version: 1.1.0
    modifiers:
      - !QuantizationModifier
        quantize_inputs: true
        quantize_outputs: true
        mode: static
        activation_bits: 8
        weight_bits: 8
        activation_symmetric: false
        weight_symmetric: true
        exclude_nodes: []
        exclude_op_types: ["BatchNormalization", "Softmax"]
        scheme: QDQ
        # Skip layernorm operations as they're sensitive to quantization
        # exclude_op_types: ["LayerNormalization"]
    """
    
    recipe_path = "temp_quantization_recipe.yaml"
    with open(recipe_path, "w") as f:
        f.write(recipe)
    
    # Quantize model
    try:
        quantize_model_post_training(
            model_path=model_path,
            output_path=output_path,
            calibration_data_path=cal_data_path,
            recipe_path=recipe_path,
        )
        print(f"Model quantized successfully: {output_path}")
    except Exception as e:
        print(f"Quantization failed: {e}")
        raise
    finally:
        # Clean up temporary files
        for file_path in [cal_data_path, recipe_path]:
            if os.path.exists(file_path):
                os.remove(file_path)

def verify_with_deepsparse(model_path, sample_input):
    """Verify the quantized model with DeepSparse"""
    try:
        from deepsparse import Engine
        print("\nVerifying with DeepSparse...")
        engine = Engine(model_path=model_path, batch_size=1)
        sample_input = sample_input[0:1]  # Take first sample
        outputs = engine.run([sample_input])
        print(f"DeepSparse inference successful! Output shape: {outputs[0].shape}")
        return True
    except ImportError:
        print("DeepSparse not installed. Install with 'pip install deepsparse'")
        return False
    except Exception as e:
        print(f"DeepSparse verification failed: {e}")
        return False

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Cargar modelo
    print(f"Cargando modelo desde {args.model}")
    try:
        # Detectar si el archivo es un modelo TorchScript
        if args.model.endswith('.pt') or args.model.endswith('.pth'):
            try:
                # Intentar cargar como modelo TorchScript
                model = torch.jit.load(args.model, map_location=device)
                print("Modelo TorchScript cargado correctamente")
            except RuntimeError:
                # Si falla, intentar cargar como state dict normal
                state_dict = torch.load(args.model, map_location=device)
                
                # Verificar si es un checkpoint con la clave 'network'
                if isinstance(state_dict, dict) and 'network' in state_dict:
                    print("Cargando modelo desde checkpoint...")
                    from model import get_pose_net
                    
                    # Configurar trainer para obtener joint_num
                    trainer = Trainer()
                    trainer._make_batch_generator()
                    joint_num = trainer.joint_num
                    
                    # Crear modelo y cargar pesos
                    model = get_pose_net(cfg, False, joint_num)
                    model.load_state_dict(state_dict['network'])
                else:
                    # Intentar cargar como state dict directo
                    from model import get_pose_net
                    trainer = Trainer()
                    trainer._make_batch_generator()
                    joint_num = trainer.joint_num
                    model = get_pose_net(cfg, False, joint_num)
                    model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Formato de archivo de modelo no soportado: {args.model}")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Primero, exportar a ONNX
    onnx_path = args.output.replace("_int8.onnx", ".onnx")
    
    # Obtener forma de entrada desde config
    input_shape = (1, 3, cfg.input_shape[1], cfg.input_shape[0])  # formato BCHW
    
    # Crear entrada dummy
    dummy_input = torch.randn(*input_shape, device=device)
    
    # Exportar a ONNX
    export_to_onnx(model, dummy_input, onnx_path)
    
    # Configurar dataset para calibración
    trainer = Trainer()
    trainer._make_batch_generator()
    dataset = trainer.batch_generator.dataset
    
    # Recopilar datos de calibración
    calibration_data = collect_calibration_data(
        dataset, 
        num_samples=args.num_samples,
        batch_size=args.batch_size
    )
    
    # Cuantizar el modelo
    quantize_model(onnx_path, calibration_data, args.output)
    
    # Opcional: Verificar con DeepSparse
    if not args.skip_verification:
        verify_with_deepsparse(args.output, calibration_data)
    
    print("\n¡Proceso de calibración y cuantización completado!")
    print(f"Modelo cuantizado final: {args.output}")

if __name__ == "__main__":
    main()