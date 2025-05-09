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
import sys
from pathlib import Path
from tqdm import tqdm
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
    parser.add_argument('--val_samples', type=int, default=50, help='Number of validation samples for metrics')
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
    
    torch.onnx.export(
        model,                        # PyTorch model
        dummy_input,                  # Example input
        output_path,                  # Output path
        export_params=True,           # Store trained parameters
        opset_version=13,             # ONNX version
        do_constant_folding=True,     # Optimize: fold constants
        input_names=['input'],        # Input name
        output_names=['output'],      # Output name
        dynamic_axes={
            'input': {0: 'batch_size'},   # Variable batch size
            'output': {0: 'batch_size'}
        },
        verbose=False,
        keep_initializers_as_inputs=False  # More optimized model
    )
    
    # Verify the ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully!")
    except ImportError:
        print("ONNX package not found. Skipping ONNX model verification.")
    except Exception as e:
        print(f"ONNX model verification failed: {e}")
    
    print(f"Model exported to {output_path}")

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

class SparseEngineAdapter:
    def __init__(self, engine):
        self.engine = engine
    def eval(self): pass
    def __call__(self, images):
        # images: torch.Tensor (B,3,H,W)
        np_in = images.cpu().numpy().astype(np.float32)
        # returns (B,J,3) float32 NumPy
        result = self.engine.run([np_in])[0]
        return torch.from_numpy(result).to(images.device)

def verify_with_deepsparse(model_path, batch_size, val_data, device, mpjpe_before):
    """Verify the quantized model with DeepSparse"""
    try:
        from deepsparse import Engine
        print("\nVerifying with DeepSparse (INT8)...")
            
        engine = Engine(model_path=model_path, batch_size=batch_size)
        quantized_engine = SparseEngineAdapter(engine)

        mpjpe_after = compute_mpjpe(quantized_engine, val_data, device)
        print(f"MPJPE after quantization:  {mpjpe_after:.2f} mm")
        print(f"Difference between original and quantized model: {abs(mpjpe_after - mpjpe_before):.2f} mm")

        with open("quantization_results.txt", "w") as f:
            f.write(f"MPJPE before quantization: {mpjpe_before:.2f} mm\n")
            f.write(f"MPJPE after quantization:  {mpjpe_after:.2f} mm\n")
            f.write(f"Diference between original and quantized model: {abs(mpjpe_after - mpjpe_before):.2f} mm ({(mpjpe_after - mpjpe_before) / mpjpe_before * 100:.2f}%)\n")
        return True
    except ImportError:
        print("DeepSparse not installed. Install with 'pip install deepsparse'")
        return False
    except Exception as e:
        print(f"DeepSparse verification failed: {e}")
        return False
    
def compute_mpjpe(model_or_engine, dataloader, device):
    model_or_engine.eval()
    total_error = 0.0
    count = 0

    for batch in dataloader:
        input_img, joint_img, joint_vis, joints_have_depth = batch
        input_img = input_img.to(device)
        joint_img = joint_img.to(device) if joint_vis is not None else None

        with torch.no_grad():
            pred_coords = model_or_engine(input_img)  
            # if DeepSparse engine: pred_coords = engine.run([images.numpy()])[0]

        # 2) Compute per-joint Euclidean distances
        errors = torch.norm(pred_coords - joint_img, dim=2)  # (B,J)

        if joint_vis is not None:
            valid_joints = joint_vis.sum()
            if valid_joints > 0:
                errors = errors * joint_vis
                total_error += errors.sum().item()
                count += valid_joints.item()
            else:
                # No valid joints in this batch
                continue
        else:
            total_error += errors.sum().item()
            count += errors.numel()

    return total_error / count if count > 0 else float('inf')  # mean over all joints & frames

def create_validation_subset(dataloader, num_samples, device):
    """Create a validation subset for consistent evaluation"""
    validation_samples = min(num_samples, len(dataloader))
    print(f"Creating validation subset with {validation_samples} samples")
    
    val_subset = []
    for _ in range(validation_samples):
        try:
            batch = next(iter(dataloader))
            # Normalize data types and ensure on correct device
            input_img, joint_img, joint_vis, joints_have_depth = batch
            input_img = input_img.to(device)
            joint_img = joint_img.to(device)
            if joint_vis is not None:
                joint_vis = joint_vis.to(device)
            val_subset.append((input_img, joint_img, joint_vis, joints_have_depth))
        except StopIteration:
            break
    
    return val_subset

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}")
    try:
        # Try to load as a state dict first
        state_dict = torch.load(args.model, map_location=device)
        # Setup trainer to get joint_num
        trainer = Trainer()
        trainer._make_batch_generator()
        joint_num = trainer.joint_num
        # Check if this is a checkpoint with 'network' key
        if isinstance(state_dict, dict) and 'network' in state_dict:
            print("Loading model from checkpoint...")
            from model import get_pose_net
            
            # Create model and load weights
            model = get_pose_net(cfg, False, joint_num)
            model.load_state_dict(state_dict['network'])
        else:
            # Try to load as a full model
            model = state_dict
            if not isinstance(model, torch.nn.Module):
                raise ValueError("Loaded model is not a valid PyTorch model.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.to(device)
    model.eval()

    # First, export to ONNX
    onnx_path = args.output.replace("_int8.onnx", ".onnx")
    
    # Get input shape from config
    input_shape = (1, 3, cfg.input_shape[1], cfg.input_shape[0])  # BCHW format
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)
    
    # Export to ONNX
    export_to_onnx(model, dummy_input, onnx_path)
    
    # Setup dataset for calibration
    val_loader = trainer.batch_generator
    dataset = trainer.batch_generator.dataset

    val_subset = create_validation_subset(val_loader, args.val_samples, device)
    
    print("\nComputing MPJPE before quantization (FP32)...")
    mpjpe_before = compute_mpjpe(model, val_subset, device)
    print(f"MPJPE before quantization: {mpjpe_before:.2f} mm")
    # Collect calibration data
    calibration_data = collect_calibration_data(
        dataset, 
        num_samples=args.num_samples,
        batch_size=args.batch_size
    )
    
    # Quantize the model
    quantize_model(onnx_path, calibration_data, args.output)
    
    # Optional: Verify with DeepSparse
    if not args.skip_verification:
        verify_with_deepsparse(args.output, args.batch_size, val_subset, device, mpjpe_before)
    
    print("\nCalibration and quantization process complete!")
    print(f"Final quantized model: {args.output}")

if __name__ == "__main__":
    main()