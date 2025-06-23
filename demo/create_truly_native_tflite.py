#!/usr/bin/env python3
"""
create_truly_native_tflite.py - Crear modelos TFLite VERDADERAMENTE nativos

Este script corrige el ERROR FUNDAMENTAL del script anterior que habilitaba SELECT_TF_OPS.
Objetivo: Crear modelos TFLite que SOLO usen operaciones nativas para máxima velocidad.
"""

import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_native_tflite_model():
    """Crear modelo TFLite VERDADERAMENTE nativo (sin SELECT_TF_OPS)"""
    try:
        import tensorflow as tf
        import onnx
        import numpy as np
        
        # Paths
        exports_dir = Path("d:/Repository-Projects/ConvNeXtPose/exports")
        onnx_path = exports_dir / "model_opt_S_optimized.onnx"
        tflite_output = exports_dir / "model_opt_S_truly_native.tflite"
        
        logger.info("🎯 Creating TRULY NATIVE TFLite model (NO SELECT_TF_OPS)")
        logger.info("=" * 60)
        
        if not onnx_path.exists():
            logger.error(f"❌ ONNX model not found: {onnx_path}")
            return None
        
        # Load and analyze ONNX model
        onnx_model = onnx.load(str(onnx_path))
        
        # Check for problematic ops
        problematic_ops = []
        for node in onnx_model.graph.node:
            if node.op_type in ['Range', 'NonMaxSuppression', 'TopK', 'ScatterND']:
                problematic_ops.append(node.op_type)
        
        if problematic_ops:
            logger.warning(f"⚠️ Found problematic ops: {set(problematic_ops)}")
            logger.info("🔧 These need to be removed/replaced for native TFLite")
        
        # Strategy 1: Try direct conversion with STRICT native-only settings
        logger.info("🔄 Attempting STRICT native-only conversion...")
        
        try:
            # Convert ONNX to TensorFlow first
            from onnx_tf.backend import prepare
            tf_rep = prepare(onnx_model)
            
            # Create SavedModel
            saved_model_dir = exports_dir / "temp_saved_model"
            if saved_model_dir.exists():
                import shutil
                shutil.rmtree(saved_model_dir)
            
            tf_rep.export_graph(str(saved_model_dir))
            
            # Configure TFLite converter with STRICT native-only settings
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
            
            # 🚀 CRITICAL: ONLY native TFLite ops - NO SELECT_TF_OPS!
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            
            # Optimization settings
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float32]
            
            # STRICT settings - fail if non-native ops are found
            converter.allow_custom_ops = False
            converter.experimental_new_converter = True
            
            logger.info("⚙️ Converting with STRICT native-only ops...")
            
            # Convert
            tflite_model = converter.convert()
            
            # Save
            with open(tflite_output, 'wb') as f:
                f.write(tflite_model)
            
            # Test inference speed
            interpreter = tf.lite.Interpreter(model_path=str(tflite_output))
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            dummy_input = np.random.random(input_details[0]['shape']).astype(np.float32)
            
            # Speed test
            import time
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            
            start_time = time.time()
            interpreter.invoke()
            inference_time = time.time() - start_time
            
            # Clean up
            if saved_model_dir.exists():
                shutil.rmtree(saved_model_dir)
            
            file_size = tflite_output.stat().st_size / (1024 * 1024)
            
            if inference_time < 0.05:  # < 50ms indicates native ops
                logger.info(f"✅ SUCCESS: Truly native TFLite model created!")
                logger.info(f"📊 File size: {file_size:.2f} MB")
                logger.info(f"⚡ Inference time: {inference_time*1000:.1f}ms (FAST - native ops)")
                return str(tflite_output)
            else:
                logger.warning(f"⚠️ Model created but inference time: {inference_time*1000:.1f}ms (still slow)")
                return None
                
        except Exception as e:
            logger.error(f"❌ Native conversion failed: {e}")
            logger.info("💭 This means the ONNX model contains ops incompatible with native TFLite")
            logger.info("💡 Solution: Need to modify the ONNX graph to remove/replace problematic ops")
            return None
            
    except Exception as e:
        logger.error(f"❌ Function failed: {e}")
        return None

def analyze_onnx_ops():
    """Analyze ONNX model to identify all operations"""
    try:
        import onnx
        
        exports_dir = Path("d:/Repository-Projects/ConvNeXtPose/exports")
        onnx_path = exports_dir / "model_opt_S_optimized.onnx"
        
        if not onnx_path.exists():
            logger.error(f"❌ ONNX model not found: {onnx_path}")
            return
        
        logger.info("🔍 Analyzing ONNX model operations...")
        
        onnx_model = onnx.load(str(onnx_path))
        
        # Count all ops
        op_counts = {}
        for node in onnx_model.graph.node:
            op_type = node.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
        
        logger.info("📊 ONNX Operations found:")
        for op_type, count in sorted(op_counts.items()):
            logger.info(f"   {op_type}: {count}")
        
        # Identify problematic ops for TFLite
        tflite_incompatible = ['Range', 'NonMaxSuppression', 'TopK', 'ScatterND', 'Resize', 'Upsample']
        problematic = []
        
        for op_type in op_counts:
            if op_type in tflite_incompatible:
                problematic.append(op_type)
        
        if problematic:
            logger.warning(f"⚠️ Problematic ops for native TFLite: {problematic}")
            logger.info("💡 These ops cause SELECT_TF_OPS fallback (slow performance)")
        else:
            logger.info("✅ No obviously problematic ops detected")
            
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")

def main():
    """Main function"""
    logger.info("🚀 TFLite Native Conversion Tool")
    logger.info("🎯 Goal: Create FAST TFLite models without SELECT_TF_OPS")
    logger.info("=" * 60)
    
    # Step 1: Analyze current ONNX model
    analyze_onnx_ops()
    
    print()
    
    # Step 2: Attempt native conversion
    result = create_native_tflite_model()
    
    print()
    
    if result:
        logger.info("🎉 SUCCESS: Native TFLite model created!")
        logger.info(f"📄 Output: {result}")
        logger.info("💡 This model should be MUCH faster than SELECT_TF_OPS models")
    else:
        logger.error("❌ FAILED: Could not create native TFLite model")
        logger.info("💭 Root cause: ONNX model contains TFLite-incompatible operations")
        logger.info("🔧 Solution needed: Modify ONNX graph to remove/replace problematic ops")
        logger.info("💡 Alternative: Use ONNX runtime (already fast) instead of TFLite")

if __name__ == "__main__":
    main()
