#!/usr/bin/env python3
"""
Prueba directa del convertidor automático ONNX→TFLite
"""

import os
import sys
import time

# Add paths
sys.path.append('/home/fabri/ConvNeXtPose/demo')

def test_conversion():
    """Test direct conversion"""
    print("🚀 Testing Automatic ONNX→TFLite Conversion")
    print("=" * 50)
    
    # Check dependencies
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow not available")
        return False
    
    # Import converter
    try:
        from automatic_onnx_to_tflite_converter import convert_onnx_to_tflite_automatic
        print("✅ Automatic converter imported")
    except Exception as e:
        print(f"❌ Converter import failed: {e}")
        return False
    
    # Setup paths
    onnx_path = "/home/fabri/ConvNeXtPose/exports/model_opt_S_optimized.onnx"
    tflite_path = "/home/fabri/ConvNeXtPose/exports/model_opt_S_auto_test.tflite"
    
    print(f"📥 Input: {onnx_path}")
    print(f"📤 Output: {tflite_path}")
    
    # Check input exists
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX model not found: {onnx_path}")
        return False
    
    input_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"📊 Input size: {input_size:.2f} MB")
    
    # Remove existing output
    if os.path.exists(tflite_path):
        os.remove(tflite_path)
    
    # Perform conversion
    print("\n🔄 Starting conversion...")
    start_time = time.time()
    
    try:
        result = convert_onnx_to_tflite_automatic(
            onnx_path=onnx_path,
            tflite_path=tflite_path,
            optimization="default"
        )
        
        conversion_time = time.time() - start_time
        
        print(f"⏱️ Conversion time: {conversion_time:.2f} seconds")
        
        if result['success']:
            print("✅ Conversion successful!")
            print(f"🎯 Strategy: {result['strategy_used']}")
            print(f"📊 Output size: {result['file_size_mb']:.2f} MB")
            
            # Verify file
            if os.path.exists(tflite_path):
                actual_size = os.path.getsize(tflite_path) / (1024 * 1024)
                print(f"✅ File created: {actual_size:.2f} MB")
                
                # Test TFLite loading
                try:
                    interpreter = tf.lite.Interpreter(model_path=tflite_path)
                    interpreter.allocate_tensors()
                    print("✅ TFLite model validates successfully")
                except Exception as e:
                    print(f"⚠️ TFLite validation error: {e}")
                
                return True
            else:
                print("❌ Output file not created")
                return False
        else:
            print(f"❌ Conversion failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_conversion()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 TEST PASSED - Automatic conversion working!")
    else:
        print("❌ TEST FAILED - Check errors above")
    
    sys.exit(0 if success else 1)
