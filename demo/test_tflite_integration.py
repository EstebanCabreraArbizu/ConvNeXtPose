#!/usr/bin/env python3
"""
Test TFLite Integration with V4
===============================

Este script prueba específicamente el uso de TFLite en V4 Enhanced.
"""

import sys
import os
import cv2
import time
import logging

# Add paths
sys.path.insert(0, '/home/fabri/ConvNeXtPose')
sys.path.insert(0, '/home/fabri/ConvNeXtPose/demo')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tflite_engine():
    """Test TFLite engine directly"""
    try:
        from convnext_realtime_v4_threading_fixed import TFLiteThreadSafeEngine
        
        model_path = "/home/fabri/ConvNeXtPose/exports/model_opt_S.pth"
        
        logger.info("🧪 Testing TFLite engine creation...")
        engine = TFLiteThreadSafeEngine(model_path)
        
        logger.info(f"✅ TFLite engine created successfully!")
        logger.info(f"   Engine type: {engine.engine_type}")
        logger.info(f"   Model path: {engine.model_path}")
        logger.info(f"   Input shape: {engine.input_shape}")
        
        # Test inference with dummy data
        import numpy as np
        dummy_input = np.random.random((256, 192, 3)).astype(np.float32)
        
        logger.info("🔄 Testing TFLite inference...")
        start_time = time.time()
        output = engine.infer(dummy_input)
        inference_time = time.time() - start_time
        
        logger.info(f"✅ TFLite inference successful!")
        logger.info(f"   Inference time: {inference_time*1000:.1f}ms")
        logger.info(f"   Output shape: {output.shape}")
        logger.info(f"   Output dtype: {output.dtype}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ TFLite engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_v4_with_tflite():
    """Test V4 with TFLite enabled"""
    try:
        from convnext_realtime_v4_threading_fixed import ThreadSafeFrameProcessor
        
        logger.info("🧪 Testing V4 with TFLite enabled...")
        
        # Create V4 processor with TFLite
        processor = ThreadSafeFrameProcessor(
            model_path="/home/fabri/ConvNeXtPose/exports/model_opt_S.pth",
            use_tflite=True,  # Enable TFLite
            yolo_model='yolov8n.pt',
            num_workers=1
        )
        
        logger.info(f"✅ V4 processor created with TFLite!")
        logger.info(f"   Engine type: {processor.pose_engine.engine_type}")
        
        # Load test image
        test_image = "/home/fabri/ConvNeXt Pose/demo/input.jpg"
        if not os.path.exists(test_image):
            logger.warning(f"Test image not found: {test_image}")
            logger.info("Creating dummy test image...")
            import numpy as np
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        else:
            dummy_frame = cv2.imread(test_image)
        
        # Test processing
        logger.info("🔄 Testing V4 + TFLite processing...")
        start_time = time.time()
        
        processor.add_frame(dummy_frame)
        
        # Wait for result
        max_wait = 10.0
        wait_start = time.time()
        result = None
        
        while (time.time() - wait_start) < max_wait:
            result = processor.get_result()
            if result:
                break
            time.sleep(0.1)
        
        total_time = time.time() - start_time
        
        if result:
            logger.info(f"✅ V4 + TFLite processing successful!")
            logger.info(f"   Total time: {total_time*1000:.1f}ms")
            logger.info(f"   Poses detected: {len(result.get('poses', []))}")
            logger.info(f"   Bboxes detected: {len(result.get('bboxes', []))}")
            success = True
        else:
            logger.warning(f"⚠️ V4 + TFLite: No result in {total_time*1000:.1f}ms")
            success = False
        
        # Cleanup
        processor.stop()
        
        return success
        
    except Exception as e:
        logger.error(f"❌ V4 + TFLite test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_engines():
    """Compare ONNX vs TFLite performance"""
    logger.info("🔄 Comparing ONNX vs TFLite performance...")
    
    results = {}
    
    # Test ONNX
    try:
        from convnext_realtime_v4_threading_fixed import ThreadSafeFrameProcessor
        
        logger.info("Testing ONNX engine...")
        processor_onnx = ThreadSafeFrameProcessor(
            model_path="/home/fabri/ConvNeXtPose/exports/model_opt_S.pth",
            use_tflite=False,  # Use ONNX
            yolo_model='yolov8n.pt',
            num_workers=1
        )
        
        # Dummy frame
        import numpy as np
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Time ONNX
        start_time = time.time()
        processor_onnx.add_frame(dummy_frame)
        
        result_onnx = None
        wait_start = time.time()
        while (time.time() - wait_start) < 5.0:
            result_onnx = processor_onnx.get_result()
            if result_onnx:
                break
            time.sleep(0.01)
        
        onnx_time = time.time() - start_time
        processor_onnx.stop()
        
        results['ONNX'] = {
            'time_ms': onnx_time * 1000,
            'success': result_onnx is not None,
            'engine': processor_onnx.pose_engine.engine_type
        }
        
    except Exception as e:
        logger.error(f"ONNX test failed: {e}")
        results['ONNX'] = {'error': str(e)}
    
    # Test TFLite
    try:
        logger.info("Testing TFLite engine...")
        processor_tflite = ThreadSafeFrameProcessor(
            model_path="/home/fabri/ConvNeXtPose/exports/model_opt_S.pth",
            use_tflite=True,  # Use TFLite
            yolo_model='yolov8n.pt',
            num_workers=1
        )
        
        # Time TFLite
        start_time = time.time()
        processor_tflite.add_frame(dummy_frame)
        
        result_tflite = None
        wait_start = time.time()
        while (time.time() - wait_start) < 5.0:
            result_tflite = processor_tflite.get_result()
            if result_tflite:
                break
            time.sleep(0.01)
        
        tflite_time = time.time() - start_time
        processor_tflite.stop()
        
        results['TFLite'] = {
            'time_ms': tflite_time * 1000,
            'success': result_tflite is not None,
            'engine': processor_tflite.pose_engine.engine_type
        }
        
    except Exception as e:
        logger.error(f"TFLite test failed: {e}")
        results['TFLite'] = {'error': str(e)}
    
    # Print results
    logger.info("📊 Engine Comparison Results:")
    for engine, result in results.items():
        if 'error' in result:
            logger.error(f"   {engine}: ERROR - {result['error']}")
        else:
            logger.info(f"   {engine}: {result['time_ms']:.1f}ms, Success: {result['success']}, Engine: {result['engine']}")
    
    return results

def main():
    """Main test function"""
    logger.info("🎯 TFLite Integration Test for ConvNeXt V4")
    logger.info("=" * 50)
    
    # Check TensorFlow availability
    try:
        import tensorflow as tf
        logger.info(f"✅ TensorFlow: {tf.__version__}")
    except ImportError:
        logger.error("❌ TensorFlow not available")
        return False
    
    # Check TFLite model
    tflite_path = "/home/fabri/ConvNeXtPose/exports/model_opt_S_optimized.tflite"
    if os.path.exists(tflite_path):
        size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        logger.info(f"✅ TFLite model found: {size_mb:.1f} MB")
    else:
        logger.error(f"❌ TFLite model not found: {tflite_path}")
        return False
    
    # Test 1: TFLite engine directly
    logger.info("\n=== Test 1: TFLite Engine ===")
    engine_success = test_tflite_engine()
    
    # Test 2: V4 with TFLite
    logger.info("\n=== Test 2: V4 + TFLite ===")
    v4_success = test_v4_with_tflite()
    
    # Test 3: Performance comparison
    logger.info("\n=== Test 3: Performance Comparison ===")
    comparison_results = compare_engines()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"TFLite Engine: {'✅ PASS' if engine_success else '❌ FAIL'}")
    logger.info(f"V4 + TFLite: {'✅ PASS' if v4_success else '❌ FAIL'}")
    
    if comparison_results:
        logger.info("Performance Comparison:")
        for engine, result in comparison_results.items():
            if 'error' not in result:
                logger.info(f"  {engine}: {result['time_ms']:.1f}ms")
    
    overall_success = engine_success and v4_success
    
    if overall_success:
        logger.info("\n🎉 TFLite integration is working correctly!")
        logger.info("   You can now use use_tflite=True in production")
    else:
        logger.info("\n⚠️ TFLite integration has issues")
        logger.info("   V4 will fallback to ONNX automatically")
    
    return overall_success

if __name__ == "__main__":
    main()
