#!/usr/bin/env python3
"""
Comparación de rendimiento entre v3 y v4
"""

import os
import sys
import cv2
import time
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_v4_onnx():
    """Test v4 with ONNX"""
    logger.info("🔥 Testing v4 with ONNX Runtime...")
    
    # Import v4
    from convnext_realtime_v4_threading_fixed import ThreadSafeFrameProcessor
    
    # Load test image
    frame = cv2.imread('input.jpg')
    if frame is None:
        logger.error("Cannot load test image")
        return None
    
    # Initialize processor
    processor = ThreadSafeFrameProcessor('/home/fabri/ConvNeXtPose/exports/model_opt_S.pth', use_tflite=False)
    
    # Test multiple runs for average
    times = []
    results = []
    
    for i in range(3):
        start_time = time.time()
        
        # Reset frame count for each test
        processor.frame_count = 0
        processor.add_frame(frame)
        
        # Wait for result
        max_wait = 5.0
        wait_start = time.time()
        result = None
        
        while (time.time() - wait_start) < max_wait:
            result = processor.get_result()
            if result:
                break
            time.sleep(0.01)
        
        total_time = (time.time() - start_time) * 1000
        
        if result:
            times.append(total_time)
            results.append(result)
            logger.info(f"  Run {i+1}: {total_time:.1f}ms")
        else:
            logger.warning(f"  Run {i+1}: Failed")
    
    processor.stop()
    
    if times:
        avg_time = sum(times) / len(times)
        logger.info(f"✅ v4 ONNX Average: {avg_time:.1f}ms ({len(results)} successful)")
        return {
            'engine': 'v4_onnx',
            'avg_time': avg_time,
            'times': times,
            'success_rate': len(results) / 3 * 100,
            'sample_result': results[0] if results else None
        }
    
    return None

def test_v4_tflite():
    """Test v4 with TFLite"""
    logger.info("🔥 Testing v4 with TensorFlow Lite...")
    
    # Import v4
    from convnext_realtime_v4_threading_fixed import ThreadSafeFrameProcessor
    
    # Load test image
    frame = cv2.imread('input.jpg')
    if frame is None:
        logger.error("Cannot load test image")
        return None
    
    # Initialize processor
    processor = ThreadSafeFrameProcessor('/home/fabri/ConvNeXtPose/exports/model_opt_S.pth', use_tflite=True)
    
    # Test multiple runs for average
    times = []
    results = []
    
    for i in range(3):
        start_time = time.time()
        
        # Reset frame count for each test
        processor.frame_count = 0
        processor.add_frame(frame)
        
        # Wait for result
        max_wait = 5.0
        wait_start = time.time()
        result = None
        
        while (time.time() - wait_start) < max_wait:
            result = processor.get_result()
            if result:
                break
            time.sleep(0.01)
        
        total_time = (time.time() - start_time) * 1000
        
        if result:
            times.append(total_time)
            results.append(result)
            logger.info(f"  Run {i+1}: {total_time:.1f}ms")
        else:
            logger.warning(f"  Run {i+1}: Failed")
    
    processor.stop()
    
    if times:
        avg_time = sum(times) / len(times)
        logger.info(f"✅ v4 TFLite Average: {avg_time:.1f}ms ({len(results)} successful)")
        return {
            'engine': 'v4_tflite',
            'avg_time': avg_time,
            'times': times,
            'success_rate': len(results) / 3 * 100,
            'sample_result': results[0] if results else None
        }
    
    return None

def test_v3_style():
    """Test using v3 components directly"""
    logger.info("🔥 Testing v3-style processing...")
    
    # Add v3 paths
    sys.path.insert(0, '/home/fabri/ConvNeXtPose/main')
    sys.path.insert(0, '/home/fabri/ConvNeXtPose/common')
    sys.path.insert(0, '/home/fabri/ConvNeXtPose/data')
    
    from config import cfg
    from model import get_pose_net
    from dataset import generate_patch_image
    import utils.pose_utils as pose_utils
    import torch
    import onnxruntime as ort
    
    # Load test image
    frame = cv2.imread('input.jpg')
    if frame is None:
        logger.error("Cannot load test image")
        return None
    
    # Setup ONNX session (like v3)
    onnx_path = '/home/fabri/ConvNeXtPose/exports/model_opt_S_optimized.onnx'
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    
    # Test multiple runs
    times = []
    results = []
    
    for i in range(3):
        start_time = time.time()
        
        try:
            # Process like v3
            h, w = frame.shape[:2]
            bbox = [0, 0, w, h]
            
            # Generate patch
            bbox_xywh = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
            proc_bbox = pose_utils.process_bbox(bbox_xywh, w, h)
            img_patch, img2bb_trans = generate_patch_image(frame, proc_bbox, False, 1.0, 0.0, False)
            
            # Inference
            if img_patch.dtype == np.uint8:
                img_patch = img_patch.astype(np.float32) / 255.0
            input_data = np.expand_dims(img_patch.transpose(2, 0, 1), axis=0).astype(np.float32)
            
            outputs = session.run(None, {input_name: input_data})
            pose_3d = outputs[0].squeeze()
            
            # Post-process
            pose_2d = pose_3d[:, :2]
            pose_2d[:, 0] = pose_2d[:, 0] / cfg.output_shape[1] * cfg.input_shape[1]
            pose_2d[:, 1] = pose_2d[:, 1] / cfg.output_shape[0] * cfg.input_shape[0]
            
            # Transform
            pose_2d_homo = np.column_stack((pose_2d, np.ones(len(pose_2d))))
            img2bb_trans_full = np.vstack((img2bb_trans, [0, 0, 1]))
            final_coords = np.linalg.solve(img2bb_trans_full, pose_2d_homo.T).T[:, :2]
            
            total_time = (time.time() - start_time) * 1000
            times.append(total_time)
            results.append(final_coords)
            logger.info(f"  Run {i+1}: {total_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"  Run {i+1}: Failed - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        logger.info(f"✅ v3-style Average: {avg_time:.1f}ms ({len(results)} successful)")
        return {
            'engine': 'v3_style',
            'avg_time': avg_time,
            'times': times,
            'success_rate': len(results) / 3 * 100,
            'sample_result': results[0] if results else None
        }
    
    return None

def compare_results(results_list):
    """Compare coordinate accuracy between different engines"""
    logger.info("\n📊 COMPARISON RESULTS:")
    logger.info("=" * 60)
    
    for result in results_list:
        if result is None:
            continue
            
        logger.info(f"\n{result['engine'].upper()}:")
        logger.info(f"  Average Time: {result['avg_time']:.1f}ms")
        logger.info(f"  Success Rate: {result['success_rate']:.1f}%")
        logger.info(f"  Time Range: {min(result['times']):.1f}ms - {max(result['times']):.1f}ms")
        
        if result['sample_result'] is not None:
            if result['engine'].startswith('v4'):
                frame_id, (coords, depth) = result['sample_result']
                logger.info(f"  Sample coords shape: {coords.shape}")
                logger.info(f"  Root depth: {depth:.1f}mm")
                logger.info(f"  Sample joint: ({coords[0][0]:.1f}, {coords[0][1]:.1f})")
            else:
                coords = result['sample_result']
                logger.info(f"  Sample coords shape: {coords.shape}")
                logger.info(f"  Sample joint: ({coords[0][0]:.1f}, {coords[0][1]:.1f})")
    
    # Find fastest
    valid_results = [r for r in results_list if r is not None]
    if valid_results:
        fastest = min(valid_results, key=lambda x: x['avg_time'])
        logger.info(f"\n🏆 FASTEST: {fastest['engine'].upper()} - {fastest['avg_time']:.1f}ms")
        
        # Speed comparison
        logger.info(f"\n⚡ SPEED COMPARISON:")
        baseline = fastest['avg_time']
        for result in valid_results:
            speedup = baseline / result['avg_time']
            if speedup < 1:
                logger.info(f"  {result['engine']}: {speedup:.2f}x slower")
            else:
                logger.info(f"  {result['engine']}: {speedup:.2f}x faster (baseline)")

def main():
    logger.info("🚀 Starting v3 vs v4 Performance Comparison")
    logger.info("=" * 60)
    
    # Test all configurations
    results = []
    
    try:
        results.append(test_v3_style())
    except Exception as e:
        logger.error(f"v3-style test failed: {e}")
    
    try:
        results.append(test_v4_onnx())
    except Exception as e:
        logger.error(f"v4 ONNX test failed: {e}")
    
    try:
        results.append(test_v4_tflite())
    except Exception as e:
        logger.error(f"v4 TFLite test failed: {e}")
    
    # Compare results
    compare_results(results)

if __name__ == "__main__":
    main()
