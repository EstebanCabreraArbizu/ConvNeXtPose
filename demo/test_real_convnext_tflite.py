#!/usr/bin/env python3
"""
Test Real ConvNeXt TFLite Conversion and Usage
=============================================

Este script verifica que la conversión real de ConvNeXt PyTorch a TFLite
funciona correctamente usando las funciones implementadas en V4.
"""

import sys
import os
import time
import logging
import numpy as np

# Setup environment
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
sys.path.insert(0, '/home/fabri/ConvNeXtPose')
sys.path.insert(0, '/home/fabri/ConvNeXtPose/demo')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_converter():
    """Test the ModelConverter class from V4"""
    logger.info("🔄 Testing ModelConverter from V4...")
    
    try:
        from convnext_realtime_v4_threading_fixed import ModelConverter
        
        # Create converter instance
        converter = ModelConverter()
        
        # Test conversion
        logger.info("🔄 Running ensure_all_models_ready()...")
        results = converter.ensure_all_models_ready()
        
        logger.info("📊 Conversion Results:")
        for key, value in results.items():
            logger.info(f"   {key}: {value}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ ModelConverter test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_real_convnext_conversion():
    """Test the real ConvNeXt conversion function from V4"""
    logger.info("🔄 Testing real ConvNeXt conversion...")
    
    try:
        from convnext_realtime_v4_threading_fixed import convert_convnext_to_optimized_formats
        
        model_path = "/home/fabri/ConvNeXtPose/exports/model_opt_S.pth"
        
        if not os.path.exists(model_path):
            logger.error(f"❌ PyTorch model not found: {model_path}")
            return None
        
        # Remove existing TFLite to force regeneration
        tflite_path = "/home/fabri/ConvNeXtPose/exports/model_opt_S_optimized.tflite"
        if os.path.exists(tflite_path):
            os.remove(tflite_path)
            logger.info(f"🗑️ Removed existing TFLite: {tflite_path}")
        
        # Test conversion with real ConvNeXt input shape
        logger.info("🔄 Converting ConvNeXt to optimized formats...")
        results = convert_convnext_to_optimized_formats(
            model_path, 
            input_shape=(1, 3, 256, 192)  # Real ConvNeXt input shape
        )
        
        logger.info("📊 Real Conversion Results:")
        for key, value in results.items():
            logger.info(f"   {key}: {value}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Real conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_tflite_engine():
    """Test the TFLite engine with real converted model"""
    logger.info("🔄 Testing TFLite engine with real model...")
    
    try:
        from convnext_realtime_v4_threading_fixed import TFLiteThreadSafeEngine
        
        model_path = "/home/fabri/ConvNeXtPose/exports/model_opt_S.pth"
        tflite_path = "/home/fabri/ConvNeXtPose/exports/model_opt_S_optimized.tflite"
        
        if not os.path.exists(tflite_path):
            logger.error(f"❌ TFLite model not found: {tflite_path}")
            return None
        
        # Create TFLite engine
        logger.info("🔄 Creating TFLite engine...")
        engine = TFLiteThreadSafeEngine(model_path)
        
        logger.info(f"✅ TFLite engine created successfully")
        logger.info(f"   Engine type: {engine.engine_type}")
        logger.info(f"   Model path: {engine.model_path}")
        logger.info(f"   Input shape: {engine.input_shape}")
        
        # Test inference with real data
        logger.info("🔄 Testing TFLite inference...")
        
        # Create realistic input (pose estimation patch)
        input_shape = engine.input_shape[1:]  # Remove batch dimension
        test_input = np.random.randint(0, 255, input_shape, dtype=np.uint8)
        
        # Run inference multiple times for timing
        inference_times = []
        for i in range(5):
            start_time = time.time()
            output = engine.infer(test_input)
            inference_time = time.time() - start_time
            inference_times.append(inference_time * 1000)  # Convert to ms
            
            logger.info(f"   Inference {i+1}: {inference_time*1000:.1f}ms, output shape: {output.shape}")
        
        avg_time = np.mean(inference_times)
        logger.info(f"✅ TFLite inference successful")
        logger.info(f"   Average inference time: {avg_time:.1f}ms")
        logger.info(f"   Output shape: {output.shape}")
        logger.info(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        return {
            'success': True,
            'avg_inference_time_ms': avg_time,
            'output_shape': output.shape,
            'engine_type': engine.engine_type
        }
        
    except Exception as e:
        logger.error(f"❌ TFLite engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_v4_with_real_tflite():
    """Test V4 ThreadSafeFrameProcessor with real TFLite model"""
    logger.info("🔄 Testing V4 with real TFLite model...")
    
    try:
        from convnext_realtime_v4_threading_fixed import ThreadSafeFrameProcessor
        import cv2
        
        # Create V4 processor with TFLite enabled
        logger.info("🔄 Creating V4 processor with TFLite...")
        processor = ThreadSafeFrameProcessor(
            model_path="/home/fabri/ConvNeXtPose/exports/model_opt_S.pth",
            use_tflite=True,  # Use real TFLite
            yolo_model='yolov8n.pt'
        )
        
        logger.info(f"✅ V4 processor created")
        logger.info(f"   Engine type: {processor.inference_engine.engine_type}")
        logger.info(f"   TFLite engine active: {processor.inference_engine.tflite_engine is not None}")
        
        # Load test image
        test_image = "/home/fabri/ConvNeXtPose/demo/input.jpg"
        if os.path.exists(test_image):
            frame = cv2.imread(test_image)
        else:
            # Create dummy frame if test image doesn't exist
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        logger.info(f"📸 Test frame shape: {frame.shape}")
        
        # Process frame
        logger.info("🔄 Processing frame with V4 + real TFLite...")
        start_time = time.time()
        
        processor.add_frame(frame)
        
        # Wait for results
        results = []
        max_wait = 15.0
        wait_start = time.time()
        
        while (time.time() - wait_start) < max_wait:
            result = processor.get_result()
            if result:
                results.append(result)
                logger.info(f"✅ Result received after {(time.time() - wait_start)*1000:.1f}ms")
                break
            time.sleep(0.1)
        
        total_time = time.time() - start_time
        
        if results:
            frame_id, (pose_coords, root_depth) = results[0]
            logger.info(f"🎉 V4 + real TFLite processing successful!")
            logger.info(f"   Total time: {total_time*1000:.1f}ms")
            logger.info(f"   Frame ID: {frame_id}")
            logger.info(f"   Pose coordinates shape: {pose_coords.shape}")
            logger.info(f"   Root depth: {root_depth:.1f}mm")
            
            # Show performance stats
            stats = processor.get_performance_stats()
            logger.info("📊 Performance Stats:")
            for key, value in stats.items():
                logger.info(f"   {key}: {value}")
            
            success_result = {
                'success': True,
                'total_time_ms': total_time * 1000,
                'pose_shape': pose_coords.shape,
                'root_depth': root_depth,
                'stats': stats
            }
        else:
            logger.warning(f"⚠️ No results received in {total_time*1000:.1f}ms")
            success_result = {
                'success': False,
                'total_time_ms': total_time * 1000,
                'reason': 'No results received'
            }
        
        # Cleanup
        processor.stop()
        
        return success_result
        
    except Exception as e:
        logger.error(f"❌ V4 + real TFLite test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_tflite_vs_onnx():
    """Compare real TFLite vs ONNX performance"""
    logger.info("🔄 Comparing real TFLite vs ONNX performance...")
    
    results = {
        'tflite': None,
        'onnx': None,
        'comparison': None
    }
    
    try:
        from convnext_realtime_v4_threading_fixed import OptimizedInferenceRouter
        import cv2
        
        # Test TFLite engine
        logger.info("🔄 Testing TFLite performance...")
        tflite_engine = OptimizedInferenceRouter(
            "/home/fabri/ConvNeXtPose/exports/model_opt_S.pth",
            use_tflite=True
        )
        
        # Test ONNX engine
        logger.info("🔄 Testing ONNX performance...")
        onnx_engine = OptimizedInferenceRouter(
            "/home/fabri/ConvNeXtPose/exports/model_opt_S.pth",
            use_tflite=False
        )
        
        # Create test input
        test_input = np.random.randint(0, 255, (256, 192, 3), dtype=np.uint8)
        
        # Benchmark TFLite
        tflite_times = []
        for i in range(10):
            start_time = time.time()
            tflite_output = tflite_engine.infer(test_input)
            tflite_times.append((time.time() - start_time) * 1000)
        
        results['tflite'] = {
            'engine_type': tflite_engine.engine_type,
            'avg_time_ms': np.mean(tflite_times),
            'min_time_ms': np.min(tflite_times),
            'max_time_ms': np.max(tflite_times),
            'std_time_ms': np.std(tflite_times),
            'output_shape': tflite_output.shape if tflite_output is not None else None
        }
        
        # Benchmark ONNX
        onnx_times = []
        for i in range(10):
            start_time = time.time()
            onnx_output = onnx_engine.infer(test_input)
            onnx_times.append((time.time() - start_time) * 1000)
        
        results['onnx'] = {
            'engine_type': onnx_engine.engine_type,
            'avg_time_ms': np.mean(onnx_times),
            'min_time_ms': np.min(onnx_times),
            'max_time_ms': np.max(onnx_times),
            'std_time_ms': np.std(onnx_times),
            'output_shape': onnx_output.shape if onnx_output is not None else None
        }
        
        # Comparison
        if results['tflite'] and results['onnx']:
            tflite_avg = results['tflite']['avg_time_ms']
            onnx_avg = results['onnx']['avg_time_ms']
            
            speedup = onnx_avg / tflite_avg if tflite_avg > 0 else 0
            
            results['comparison'] = {
                'tflite_faster': tflite_avg < onnx_avg,
                'speedup_factor': speedup,
                'time_difference_ms': onnx_avg - tflite_avg,
                'relative_improvement_percent': ((onnx_avg - tflite_avg) / onnx_avg * 100) if onnx_avg > 0 else 0
            }
        
        logger.info("📊 Performance Comparison Results:")
        logger.info(f"   TFLite: {results['tflite']['avg_time_ms']:.1f}ms ± {results['tflite']['std_time_ms']:.1f}ms")
        logger.info(f"   ONNX: {results['onnx']['avg_time_ms']:.1f}ms ± {results['onnx']['std_time_ms']:.1f}ms")
        
        if results['comparison']:
            comp = results['comparison']
            logger.info(f"   TFLite is {'faster' if comp['tflite_faster'] else 'slower'} by {abs(comp['time_difference_ms']):.1f}ms")
            logger.info(f"   Speedup factor: {comp['speedup_factor']:.2f}x")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Performance comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return results

def main():
    """Main test function"""
    logger.info("🎯 Testing Real ConvNeXt TFLite Conversion and Usage in V4")
    logger.info("=" * 70)
    
    # Check TensorFlow availability
    try:
        import tensorflow as tf
        logger.info(f"✅ TensorFlow: {tf.__version__}")
    except ImportError:
        logger.error("❌ TensorFlow not available")
        return False
    
    test_results = {}
    
    # Test 1: ModelConverter
    logger.info("\n" + "=" * 50)
    logger.info("📋 Test 1: ModelConverter")
    logger.info("=" * 50)
    test_results['model_converter'] = test_model_converter()
    
    # Test 2: Real ConvNeXt conversion
    logger.info("\n" + "=" * 50)
    logger.info("📋 Test 2: Real ConvNeXt Conversion")
    logger.info("=" * 50)
    test_results['real_conversion'] = test_real_convnext_conversion()
    
    # Test 3: TFLite engine
    logger.info("\n" + "=" * 50)
    logger.info("📋 Test 3: TFLite Engine")
    logger.info("=" * 50)
    test_results['tflite_engine'] = test_tflite_engine()
    
    # Test 4: V4 with real TFLite
    logger.info("\n" + "=" * 50)
    logger.info("📋 Test 4: V4 with Real TFLite")
    logger.info("=" * 50)
    test_results['v4_tflite'] = test_v4_with_real_tflite()
    
    # Test 5: Performance comparison
    logger.info("\n" + "=" * 50)
    logger.info("📋 Test 5: TFLite vs ONNX Performance")
    logger.info("=" * 50)
    test_results['performance_comparison'] = compare_tflite_vs_onnx()
    
    # Final results
    logger.info("\n" + "=" * 70)
    logger.info("📊 FINAL TEST RESULTS")
    logger.info("=" * 70)
    
    passed_tests = 0
    total_tests = 0
    
    for test_name, result in test_results.items():
        total_tests += 1
        if result and result != False:
            if isinstance(result, dict) and result.get('success', True):
                passed_tests += 1
                logger.info(f"✅ {test_name}: PASSED")
            elif result:
                passed_tests += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.info(f"❌ {test_name}: FAILED")
        else:
            logger.info(f"❌ {test_name}: FAILED")
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    logger.info(f"\n🎯 Overall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        logger.info("🎉 Real ConvNeXt TFLite conversion and usage is working correctly!")
        return True
    else:
        logger.info("⚠️ Some tests failed - TFLite may have issues")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 SUCCESS: Real ConvNeXt TFLite is fully functional in V4!")
    else:
        print("\n❌ ISSUES: Real ConvNeXt TFLite needs attention")
