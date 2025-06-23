#!/usr/bin/env python3
"""
test_tflite_conversion.py - Script para probar conversión a TFLite paso a paso
"""

import sys
import os
from pathlib import Path

# Añadir rutas del proyecto
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.extend([
    str(PROJECT_ROOT / 'main'),
    str(PROJECT_ROOT / 'data'),
    str(PROJECT_ROOT / 'common'),
    str(ROOT)
])

import torch
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pytorch_to_onnx():
    """Probar conversión PyTorch a ONNX"""
    try:
        from config import cfg
        from model import get_pose_net
        
        # Configurar modelo
        cfg.input_shape = (256, 256)
        cfg.output_shape = (32, 32) 
        cfg.depth_dim = 32
        cfg.bbox_3d_shape = (2000, 2000, 2000)
        
        model_path = r'D:\Repository-Projects\ConvNeXtPose\exports\model_opt_S.pth'
        onnx_path = model_path.replace('.pth', '_optimized.onnx')
        
        if os.path.exists(onnx_path):
            logger.info(f"✅ ONNX model already exists: {onnx_path}")
            return onnx_path
        
        logger.info("🔄 Converting PyTorch to ONNX...")
        
        # Cargar modelo PyTorch
        model = get_pose_net(cfg, is_train=False, joint_num=18)
        state = torch.load(model_path, map_location='cpu')
        sd = state.get('network', state)
        model.load_state_dict(sd, strict=False)
        model.eval()
        
        # Crear input dummy
        dummy_input = torch.randn(1, 3, cfg.input_shape[0], cfg.input_shape[1])
        
        # Exportar a ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        if os.path.exists(onnx_path):
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)
            logger.info(f"✅ ONNX conversion successful: {file_size:.2f} MB")
            return onnx_path
        else:
            logger.error("❌ ONNX file was not created")
            return None
            
    except Exception as e:
        logger.error(f"❌ PyTorch to ONNX conversion failed: {e}")
        return None

def test_onnx_to_tflite_simple():
    """Probar conversión ONNX a TFLite usando onnx-tf"""
    try:
        import tensorflow as tf
        import onnx
        from onnx_tf.backend import prepare
        
        model_path = r'D:\Repository-Projects\ConvNeXtPose\exports\model_opt_S.pth'
        onnx_path = model_path.replace('.pth', '_optimized.onnx')
        tflite_path = model_path.replace('.pth', '_simple.tflite')
        saved_model_path = model_path.replace('.pth', '_tf_model')
        
        if not os.path.exists(onnx_path):
            logger.error(f"❌ ONNX model not found: {onnx_path}")
            return None
        
        logger.info("🔄 Converting ONNX to TFLite using onnx-tf...")
        
        # Paso 1: Cargar modelo ONNX
        logger.info("📋 Step 1: Loading ONNX model...")
        onnx_model = onnx.load(onnx_path)
        logger.info(f"   ONNX model loaded successfully")
        
        # Paso 2: Convertir ONNX a TensorFlow usando onnx-tf
        logger.info("📋 Step 2: Converting ONNX to TensorFlow...")
        tf_rep = prepare(onnx_model)
        
        # Paso 3: Exportar como SavedModel
        logger.info("📋 Step 3: Exporting TensorFlow SavedModel...")
        tf_rep.export_graph(saved_model_path)
        
        # Paso 4: Convertir SavedModel a TFLite
        logger.info("📋 Step 4: Converting SavedModel to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        
        # Configurar optimizaciones
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Configurar tipos de datos de entrada y salida
        converter.target_spec.supported_types = [tf.float32]
        
        # Permitir ops personalizadas si es necesario
        converter.allow_custom_ops = True
        
        # Convertir
        tflite_model = converter.convert()
        
        # Guardar modelo TFLite
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        if os.path.exists(tflite_path):
            file_size = os.path.getsize(tflite_path) / (1024 * 1024)
            logger.info(f"✅ TFLite conversion successful: {file_size:.2f} MB")
            
            # Limpiar directorio temporal
            import shutil
            if os.path.exists(saved_model_path):
                shutil.rmtree(saved_model_path)
                logger.info("🧹 Cleaned up temporary SavedModel directory")
            
            return tflite_path
        else:
            logger.error("❌ TFLite file was not created")
            return None
        
    except ImportError as e:
        logger.error(f"❌ Missing required library: {e}")
        logger.info("💡 Make sure onnx-tf is installed: pip install onnx-tf")
        return None
    except Exception as e:
        logger.error(f"❌ ONNX to TFLite conversion failed: {e}")
        logger.info(f"Error details: {str(e)}")
        return None

def test_tflite_inference():
    """Probar inferencia TFLite"""
    try:
        import tensorflow as tf
        
        model_path = r'D:\Repository-Projects\ConvNeXtPose\exports\model_opt_S.pth'
        tflite_path = model_path.replace('.pth', '_simple.tflite')
        
        if not os.path.exists(tflite_path):
            logger.error(f"❌ TFLite model not found: {tflite_path}")
            return False
        
        logger.info("🔄 Testing TFLite inference...")
        
        # Cargar modelo TFLite
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Obtener detalles de input/output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info(f"   Input shape: {input_details[0]['shape']}")
        logger.info(f"   Output shape: {output_details[0]['shape']}")
        
        # Crear input dummy
        input_shape = input_details[0]['shape']
        dummy_input = np.random.random(input_shape).astype(np.float32)
        
        # Ejecutar inferencia
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        logger.info(f"✅ TFLite inference successful, output shape: {output.shape}")
        return True
        
    except Exception as e:
        logger.error(f"❌ TFLite inference failed: {e}")
        return False

def main():
    """Función principal de pruebas"""
    logger.info("🚀 TFLite Conversion Test Suite using onnx-tf")
    logger.info("=" * 50)
    
    # Verificar dependencias
    logger.info("🔍 Checking dependencies...")
    try:
        import onnx
        import tensorflow as tf
        from onnx_tf.backend import prepare
        logger.info("✅ All required packages available")
        logger.info(f"   - ONNX: {onnx.__version__}")
        logger.info(f"   - TensorFlow: {tf.__version__}")
        logger.info("   - onnx-tf: Available")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("💡 Install with: pip install onnx-tf")
        return False
    
    # Paso 1: PyTorch a ONNX
    logger.info("\n📋 STEP 1: PyTorch → ONNX")
    onnx_path = test_pytorch_to_onnx()
    if not onnx_path:
        logger.error("❌ Cannot proceed without ONNX model")
        return False
    
    # Paso 2: ONNX a TFLite usando onnx-tf
    logger.info("\n📋 STEP 2: ONNX → TensorFlow → TFLite")
    tflite_path = test_onnx_to_tflite_simple()
    if not tflite_path:
        logger.warning("⚠️ TFLite conversion failed, but ONNX is available")
        logger.info("💡 You can still use --backend onnx")
        return False
    
    # Paso 3: Probar inferencia TFLite
    logger.info("\n📋 STEP 3: TFLite Inference Test")
    inference_ok = test_tflite_inference()
    
    logger.info("\n" + "=" * 50)
    logger.info("📊 CONVERSION RESULTS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"PyTorch → ONNX: {'✅ SUCCESS' if onnx_path else '❌ FAILED'}")
    logger.info(f"ONNX → TFLite: {'✅ SUCCESS' if tflite_path else '❌ FAILED'}")
    logger.info(f"TFLite Inference: {'✅ SUCCESS' if inference_ok else '❌ FAILED'}")
    
    if onnx_path and tflite_path and inference_ok:
        logger.info("\n🎉 FULL TFLITE PIPELINE WORKING!")
        logger.info("You can now use: --backend tflite")
        return True
    elif onnx_path:
        logger.info("\n⚠️ TFLite failed, but ONNX is available")
        logger.info("You can use: --backend onnx")
        return False
    else:
        logger.info("\n❌ All conversions failed")
        logger.info("You can still use: --backend pytorch")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
