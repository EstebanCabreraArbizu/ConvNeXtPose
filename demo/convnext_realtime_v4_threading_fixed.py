#!/usr/bin/env python3
"""
convnext_realtime_v4_threading_fixed.py - Thread-safe version for TFLite support

Key fixes:
1. ✅ Thread-safe TFLite implementation with interpreter per thread
2. ✅ Proper model memory management in threaded environment
3. ✅ Fixed reference handling for TensorFlow Lite
"""

import sys
import os
import time
import queue
import threading
import concurrent.futures
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from collections import deque
import numpy as np
import cv2

# TensorFlow setup (quiet)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)

# Core imports
import torch
import torchvision.transforms as transforms

# ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# TensorFlow Lite
TFLITE_AVAILABLE = False
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    pass

# Project imports
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.extend([
    str(PROJECT_ROOT / 'main'),
    str(PROJECT_ROOT / 'data'),
    str(PROJECT_ROOT / 'common'),
    str(ROOT)
])

from config import cfg
from model import get_pose_net
from dataset import generate_patch_image
import utils.pose_utils as pose_utils
from root_wrapper import RootNetWrapper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessingTask:
    """Task structure for parallel processing"""
    frame: np.ndarray
    bbox: List[int]
    timestamp: float
    frame_id: int
    cache_key: str

def detect_hardware_capabilities():
    """Detect hardware capabilities (from v3)"""
    capabilities = {
        'has_cuda': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_memory_gb': 0,
        'cpu_cores': os.cpu_count(),
        'recommended_workers': 1,
        'recommended_cache_timeout': 0.12,
        'recommended_frame_skip': 2
    }
    
    if capabilities['has_cuda']:
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            capabilities['cuda_memory_gb'] = gpu_memory
            
            if gpu_memory >= 8:  # RTX 3070+
                capabilities.update({
                    'recommended_workers': 3,  # Más workers para paralelización
                    'recommended_cache_timeout': 0.08,
                    'recommended_frame_skip': 1
                })
            elif gpu_memory >= 4:  # GTX 1660+
                capabilities.update({
                    'recommended_workers': 2,
                    'recommended_cache_timeout': 0.10,
                    'recommended_frame_skip': 2
                })
        except:
            pass
    else:
        capabilities.update({
            'recommended_workers': 2,  # CPU también puede paralelizar
            'recommended_cache_timeout': 0.15,
            'recommended_frame_skip': 3
        })
    
    logger.info(f"🔧 Hardware detected: GPU={'✅' if capabilities['has_cuda'] else '❌'} "
                f"({capabilities['cuda_memory_gb']:.1f}GB), Workers={capabilities['recommended_workers']}")
    
    return capabilities

class TFLiteThreadSafeEngine:
    """Thread-safe TFLite engine that creates a separate interpreter per thread"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path.replace('.pth', '_optimized.tflite')
        self.engine_type = "TensorFlow Lite (Thread-safe)"
        
        # Store each interpreter by thread ID to ensure thread safety
        self.interpreters = {}
        self.interpreter_locks = {}
        
        # Validate model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"TFLite model not found: {self.model_path}")
        
        # Create first interpreter to validate and get shapes
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        
        logger.info(f"✅ Thread-safe TFLite engine initialized")
        
    def _get_interpreter(self):
        """Get interpreter for current thread"""
        thread_id = threading.get_ident()
        
        # Create new interpreter and lock for this thread if needed
        if thread_id not in self.interpreters:
            self.interpreters[thread_id] = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreters[thread_id].allocate_tensors()
            self.interpreter_locks[thread_id] = threading.Lock()
            
        return self.interpreters[thread_id], self.interpreter_locks[thread_id]
    
    def infer(self, img_patch: np.ndarray) -> np.ndarray:
        """Run TFLite inference in a thread-safe manner"""
        interpreter, lock = self._get_interpreter()
        
        # Prepare input tensor
        input_data = self._prepare_input(img_patch)
        
        # Use lock to ensure thread safety
        with lock:
            input_index = interpreter.get_input_details()[0]['index']
            output_index = interpreter.get_output_details()[0]['index']
            
            # Copy input data to avoid reference issues
            interpreter.set_tensor(input_index, input_data.copy())
            interpreter.invoke()
            
            # Get output and copy it to avoid reference issues
            output_data = interpreter.get_tensor(output_index).copy()
        
        return output_data
    
    def _prepare_input(self, img_patch: np.ndarray) -> np.ndarray:
        """Prepare input tensor for TFLite"""
        # Convert to float and normalize
        if img_patch.dtype == np.uint8:
            img_patch = img_patch.astype(np.float32) / 255.0
        
        # Match expected layout
        input_tensor = np.transpose(img_patch, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Ensure shape matches
        expected_shape = tuple(self.input_shape)
        if input_tensor.shape != expected_shape:
            # If just batch size is different
            if input_tensor.shape[1:] == expected_shape[1:]:
                input_tensor = input_tensor.reshape(expected_shape)
            else:
                # Need to resize
                h, w = expected_shape[2], expected_shape[3]
                img_resized = cv2.resize(img_patch, (w, h))
                input_tensor = np.expand_dims(np.transpose(img_resized, (2, 0, 1)), axis=0)
        
        return input_tensor.astype(np.float32)

class OptimizedInferenceRouter:
    """Routes inference requests to the appropriate backend"""
    
    def __init__(self, model_path: str, use_tflite: bool = False):
        self.model_path = model_path
        self.engine_type = "Uninitialized"
        
        # Initialize engines based on availability and preference
        self.tflite_engine = None
        self.onnx_engine = None
        self.pytorch_engine = None
        
        # Setup engines
        if use_tflite and TFLITE_AVAILABLE:
            try:
                self.tflite_engine = TFLiteThreadSafeEngine(model_path)
                self.engine_type = self.tflite_engine.engine_type
                logger.info("✅ TensorFlow Lite (thread-safe) engine configured")
                return
            except Exception as e:
                logger.warning(f"⚠️ TFLite engine setup failed: {e}")
        
        if ONNX_AVAILABLE:
            try:
                self.onnx_engine = self._setup_onnx()
                self.engine_type = "ONNX Runtime"
                logger.info("✅ ONNX Runtime engine configured")
                return
            except Exception as e:
                logger.warning(f"⚠️ ONNX engine setup failed: {e}")
        
        # PyTorch as fallback
        try:
            self.pytorch_engine = self._setup_pytorch()
            self.engine_type = "PyTorch"
            logger.info("✅ PyTorch engine configured")
        except Exception as e:
            logger.error(f"❌ PyTorch engine setup failed: {e}")
            raise RuntimeError("No inference engine available")
    
    def _setup_onnx(self):
        """Setup ONNX Runtime engine"""
        onnx_path = self.model_path.replace('.pth', '_optimized.onnx')
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(onnx_path, providers=providers, sess_options=session_options)
        return {
            'session': session,
            'input_name': session.get_inputs()[0].name
        }
    
    def _setup_pytorch(self):
        """Setup PyTorch engine"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_pose_net(cfg, is_train=False, joint_num=18)
        
        state = torch.load(self.model_path, map_location=device)
        state_dict = state.get('network', state) 
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device).eval()
        
        return {
            'model': model,
            'device': device
        }
    
    def infer(self, img_patch: np.ndarray) -> Optional[np.ndarray]:
        """Run inference with the appropriate engine"""
        try:
            if self.tflite_engine:
                return self.tflite_engine.infer(img_patch)
            elif self.onnx_engine:
                return self._infer_onnx(img_patch)
            elif self.pytorch_engine:
                return self._infer_pytorch(img_patch)
            else:
                raise RuntimeError("No inference engine available")
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _infer_onnx(self, img_patch: np.ndarray) -> np.ndarray:
        """ONNX Runtime inference"""
        if img_patch.dtype == np.uint8:
            img_patch = img_patch.astype(np.float32) / 255.0
            
        input_data = np.expand_dims(img_patch.transpose(2, 0, 1), axis=0).astype(np.float32)
        session = self.onnx_engine['session']
        input_name = self.onnx_engine['input_name']
        
        outputs = session.run(None, {input_name: input_data})
        return outputs[0]
    
    def _infer_pytorch(self, img_patch: np.ndarray) -> np.ndarray:
        """PyTorch inference"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std),
        ])
        
        from PIL import Image
        pil_img = Image.fromarray(cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0).to(self.pytorch_engine['device'])
        
        with torch.no_grad():
            output = self.pytorch_engine['model'](input_tensor)
            return output.cpu().numpy()

class IntelligentCacheManager:
    """Intelligent cache manager (optimized from v3)"""
    
    def __init__(self, cache_timeout: float = 0.1, max_size: int = 50):
        self.cache = {}
        self.cache_timeout = cache_timeout
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.cache_lock = threading.RLock()
    
    def generate_key(self, bbox: List[int], timestamp: float) -> str:
        """Generate intelligent cache key"""
        # Spatial quantization (10-pixel groups)
        x1, y1, x2, y2 = bbox
        spatial_key = f"{x1//10}_{y1//10}_{x2//10}_{y2//10}"
        
        # Temporal quantization (100ms groups)
        temporal_key = int(timestamp * 10)
        
        return f"{spatial_key}_{temporal_key}"
    
    def get(self, key: str, current_time: float) -> Optional[Tuple[np.ndarray, float]]:
        """Get cached result if valid"""
        with self.cache_lock:
            if key in self.cache:
                cached_result, cached_time = self.cache[key]
                if current_time - cached_time < self.cache_timeout:
                    self.hits += 1
                    return cached_result, 0.0  # No depth from cache
            
            self.misses += 1
            return None
    
    def put(self, key: str, result: np.ndarray, timestamp: float):
        """Store result in cache"""
        with self.cache_lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            self.cache[key] = (result.copy(), timestamp)  # Copy to avoid reference issues
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.cache_lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            return {
                'cache_hits': self.hits,
                'cache_misses': self.misses,
                'cache_hit_rate': hit_rate,
                'cache_size': len(self.cache)
            }

class ParallelPoseProcessor:
    """Parallel pose processor using ThreadPoolExecutor (like v3)"""
    
    def __init__(self, inference_engine: OptimizedInferenceRouter, 
                 root_wrapper: Optional[RootNetWrapper], 
                 hardware_caps: Dict[str, Any]):
        self.inference_engine = inference_engine
        self.root_wrapper = root_wrapper
        self.hardware_caps = hardware_caps
        
        # Parallel execution setup (from v3)
        max_workers = hardware_caps['recommended_workers']
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Cache manager
        self.cache_manager = IntelligentCacheManager(
            cache_timeout=hardware_caps['recommended_cache_timeout'],
            max_size=50 if hardware_caps['has_cuda'] else 30
        )
        
        # Thread-safe stats tracking
        self.stats_lock = threading.RLock()
        self.processing_times = deque(maxlen=100)
        self.successful_inferences = 0
        self.total_processed = 0
        self.failed_inferences = 0
        
        logger.info(f"✅ ParallelPoseProcessor initialized with {max_workers} workers")
    
    def process_pose_parallel(self, task: ProcessingTask) -> Optional[Tuple[np.ndarray, float]]:
        """Process a single pose task (runs in thread pool)"""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_result = self.cache_manager.get(task.cache_key, task.timestamp)
            if cached_result is not None:
                return cached_result
            
            # Generate patch
            h, w = task.frame.shape[:2]
            bbox_xywh = [task.bbox[0], task.bbox[1], 
                        task.bbox[2] - task.bbox[0], 
                        task.bbox[3] - task.bbox[1]]
            
            proc_bbox = pose_utils.process_bbox(np.array(bbox_xywh), w, h)
            if proc_bbox is None:
                return None
            img_patch, img2bb_trans = generate_patch_image(task.frame, proc_bbox, False, 1.0, 0.0, False)
            
            # Ensure correct data type
            if img_patch.dtype != np.uint8:
                img_patch = (img_patch * 255).astype(np.uint8)
            
            # Inference
            pose_3d = self.inference_engine.infer(img_patch)
            if pose_3d is None:
                with self.stats_lock:
                    self.failed_inferences += 1
                return None
            
            # Post-process coordinates
            pose_3d_np = pose_3d.squeeze()
            
            # Handle different output formats
            if len(pose_3d_np.shape) == 3:  # Heatmap format [joints, height, width]
                joint_num, output_h, output_w = pose_3d_np.shape
                heatmaps = pose_3d_np.reshape(joint_num, -1)
                max_indices = np.argmax(heatmaps, axis=1)
                pose_y = max_indices // output_w
                pose_x = max_indices % output_w
                pose_2d = np.stack([pose_x, pose_y], axis=1).astype(np.float32)
            else:  # Direct coordinate format
                pose_2d = pose_3d_np[:, :2]
            
            # Scale to input space
            pose_2d[:, 0] = pose_2d[:, 0] / cfg.output_shape[1] * cfg.input_shape[1]
            pose_2d[:, 1] = pose_2d[:, 1] / cfg.output_shape[0] * cfg.input_shape[0]
            
            # Apply inverse transformation
            pose_2d_homo = np.column_stack((pose_2d, np.ones(len(pose_2d))))
            img2bb_trans_full = np.vstack((img2bb_trans, [0, 0, 1]))
            
            try:
                final_coords = np.linalg.solve(img2bb_trans_full, pose_2d_homo.T).T[:, :2]
            except:
                final_coords = pose_2d
            
            # Root depth estimation using correct method
            root_depth = 8000
            if self.root_wrapper is not None:
                try:
                    # Use predict_depth instead of get_root_depth
                    bbox_for_root = [task.bbox[0], task.bbox[1], 
                                   task.bbox[2] - task.bbox[0], 
                                   task.bbox[3] - task.bbox[1]]
                    root_depth = self.root_wrapper.predict_depth(task.frame, bbox_for_root)
                except Exception as e:
                    logger.debug(f"RootNet failed: {e}")
                    pass
            
            # Cache result
            self.cache_manager.put(task.cache_key, final_coords, task.timestamp)
            
            # Update stats
            processing_time = (time.time() - start_time) * 1000
            with self.stats_lock:
                self.processing_times.append(processing_time)
                self.successful_inferences += 1
            
            return final_coords, root_depth
            
        except Exception as e:
            logger.error(f"❌ Pose processing failed: {e}")
            with self.stats_lock:
                self.failed_inferences += 1
            return None
        finally:
            with self.stats_lock:
                self.total_processed += 1

class ThreadSafeFrameProcessor:
    """Thread-safe frame processor with improved architecture"""
    
    def __init__(self, model_path: str, use_tflite: bool = False):
        self.hardware_caps = detect_hardware_capabilities()
        
        # Initialize components separately (like v3)
        self.inference_engine = OptimizedInferenceRouter(model_path, use_tflite)
        
        # Initialize RootNet
        try:
            rootnet_dir = "/home/fabri/3DMPPE_ROOTNET_RELEASE"
            rootnet_model = "/home/fabri/3DMPPE_ROOTNET_RELEASE/demo/snapshot_18.pth.tar"
            self.root_wrapper = RootNetWrapper(rootnet_dir, rootnet_model)
            self.root_wrapper.load_model(use_gpu=torch.cuda.is_available())
            logger.info("✅ RootNet initialized")
        except Exception as e:
            logger.warning(f"⚠️ RootNet initialization failed: {e}")
            self.root_wrapper = None
        
        # Parallel processor (like v3's IntelligentPoseProcessor)
        self.pose_processor = ParallelPoseProcessor(
            self.inference_engine, self.root_wrapper, self.hardware_caps
        )
        
        # Frame management with thread-safe queues
        queue_size = 2 if self.hardware_caps['has_cuda'] else 1
        self.input_queue = queue.Queue(maxsize=queue_size)
        self.output_queue = queue.Queue(maxsize=queue_size)
        
        # Frame skipping
        self.frame_count = 0
        self.skip_every_n_frames = self.hardware_caps['recommended_frame_skip']
        
        # Processing control
        self.processing = True
        self.processor_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processor_thread.start()
        
        logger.info(f"✅ ThreadSafeFrameProcessor initialized with {self.inference_engine.engine_type}")
    
    def _process_frames(self):
        """Main processing loop with parallel execution (improved from v3 architecture)"""
        pending_futures = []
        logger.info("🔄 Processing thread started")
        
        while self.processing:
            try:
                # Process pending futures first
                completed_futures = []
                for future, frame_id in pending_futures:
                    if future.done():
                        try:
                            result = future.result(timeout=0.1)
                            if result is not None:
                                try:
                                    self.output_queue.put_nowait((frame_id, result))
                                    logger.info(f"✅ Result for frame {frame_id} added to output queue")
                                except queue.Full:
                                    logger.warning(f"⚠️ Output queue full, dropping result for frame {frame_id}")
                                    pass  # Skip if output queue is full
                            completed_futures.append((future, frame_id))
                        except Exception as e:
                            logger.error(f"❌ Future failed for frame {frame_id}: {e}")
                            completed_futures.append((future, frame_id))
                
                # Remove completed futures
                for completed in completed_futures:
                    pending_futures.remove(completed)
                
                # Get next task (non-blocking)
                try:
                    task = self.input_queue.get(timeout=0.1)
                    if task is None:  # Shutdown signal
                        logger.info("🛑 Received shutdown signal")
                        break
                        
                    logger.info(f"📋 Processing task for frame {task.frame_id}")
                    
                    # Submit to thread pool for parallel processing
                    future = self.pose_processor.executor.submit(
                        self.pose_processor.process_pose_parallel, task
                    )
                    pending_futures.append((future, task.frame_id))
                    logger.info(f"🚀 Submitted frame {task.frame_id} to thread pool")
                    
                except queue.Empty:
                    continue
                
                # Limit pending futures to prevent memory buildup
                if len(pending_futures) > self.hardware_caps['recommended_workers'] * 2:
                    oldest_future, oldest_frame_id = pending_futures.pop(0)
                    oldest_future.cancel()
                    logger.warning(f"⚠️ Cancelled oldest future for frame {oldest_frame_id}")
                
            except Exception as e:
                logger.error(f"❌ Processing loop error: {e}")
                time.sleep(0.01)  # Small delay to prevent busy waiting
        
        logger.info("🔄 Processing thread ending")
    
    def add_frame(self, frame: np.ndarray, bbox: Optional[List[int]] = None):
        """Add frame for processing (improved with better error handling)"""
        if not self.processing:
            return
            
        self.frame_count += 1
        
        # Simplified frame skipping - always process first frame for testing
        if self.frame_count == 1:
            # Always process first frame
            pass
        else:
            # Apply frame skipping for subsequent frames
            queue_load = self.input_queue.qsize()
            skip_rate = max(self.skip_every_n_frames, queue_load + 1)
            
            if self.frame_count % skip_rate != 0:
                return
        
        timestamp = time.time()
        h, w = frame.shape[:2]
        
        # Use provided bbox or default to full frame
        if bbox is None:
            bbox = [0, 0, w, h]
        
        # Generate cache key
        cache_key = self.pose_processor.cache_manager.generate_key(bbox, timestamp)
        
        task = ProcessingTask(
            frame=frame.copy(),
            bbox=bbox,
            timestamp=timestamp,
            frame_id=self.frame_count,
            cache_key=cache_key
        )
        
        logger.info(f"📝 Adding frame {self.frame_count} to queue (queue size: {self.input_queue.qsize()})")
        
        try:
            # Non-blocking put with fallback
            self.input_queue.put_nowait(task)
            logger.info(f"✅ Frame {self.frame_count} added to queue successfully")
        except queue.Full:
            # Queue is full, try to remove oldest task
            try:
                self.input_queue.get_nowait()  # Remove oldest
                self.input_queue.put_nowait(task)  # Add new
                logger.info(f"🔄 Frame {self.frame_count} replaced older task in queue")
            except:
                logger.warning(f"⚠️ Could not add frame {self.frame_count} - queue full")
                pass  # Skip this frame if still can't add
    
    def get_result(self) -> Optional[Tuple[int, Tuple[np.ndarray, float]]]:
        """Get processing result"""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        cache_stats = self.pose_processor.cache_manager.get_stats()
        
        with self.pose_processor.stats_lock:
            avg_time = 0
            if self.pose_processor.processing_times:
                avg_time = sum(self.pose_processor.processing_times) / len(self.pose_processor.processing_times)
            
            stats = {
                'engine_type': self.inference_engine.engine_type,
                'avg_processing_time_ms': avg_time,
                'successful_inferences': self.pose_processor.successful_inferences,
                'failed_inferences': self.pose_processor.failed_inferences,
                'total_processed': self.pose_processor.total_processed,
                'frame_skip_rate': self.skip_every_n_frames,
                'queue_size': self.input_queue.qsize(),
                'workers': self.hardware_caps['recommended_workers'],
            }
        
        # Combine with cache stats
        stats.update(cache_stats)
        return stats
    
    def stop(self):
        """Stop processing gracefully"""
        logger.info("🛑 Stopping ThreadSafeFrameProcessor...")
        
        # Signal stop
        self.processing = False
        
        # Add shutdown signal to queue
        try:
            self.input_queue.put_nowait(None)
        except queue.Full:
            # Clear queue and add stop signal
            while not self.input_queue.empty():
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    break
            self.input_queue.put_nowait(None)
        
        # Wait for processor thread
        if self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2.0)
            if self.processor_thread.is_alive():
                logger.warning("⚠️ Processor thread did not stop gracefully")
        
        # Shutdown executor
        try:
            self.pose_processor.executor.shutdown(wait=True, timeout=3.0)
        except:
            logger.warning("⚠️ Executor shutdown timeout")
        
        logger.info("✅ ThreadSafeFrameProcessor stopped")

# Alias for compatibility
ImprovedFrameProcessor = ThreadSafeFrameProcessor

def main():
    """Test thread-safe v4"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Thread-Safe ConvNeXt v4')
    parser.add_argument('--model_path', type=str, 
                       default='/home/fabri/ConvNeXtPose/exports/model_opt_S.pth',
                       help='Path to pose model')
    parser.add_argument('--image_path', type=str, 
                       default='/home/fabri/ConvNeXtPose/demo/input.jpg',
                       help='Path to input image')
    parser.add_argument('--use_tflite', action='store_true',
                       help='Use TensorFlow Lite if available')
    
    args = parser.parse_args()
    
    # Create improved processor
    processor = ThreadSafeFrameProcessor(args.model_path, use_tflite=args.use_tflite)
    
    # Load test image
    frame = cv2.imread(args.image_path)
    if frame is None:
        logger.error(f"❌ Cannot load image: {args.image_path}")
        return
    
    logger.info(f"📸 Testing with image: {frame.shape}")
    
    # Process frame with better timing
    start_time = time.time()
    
    # Add frame for processing
    processor.add_frame(frame)
    
    # Wait for result with progressive timeout
    results = []
    max_wait_time = 10.0  # 10 seconds max
    wait_start = time.time()
    check_interval = 0.05  # Check every 50ms
    
    logger.info("⏳ Waiting for processing results...")
    
    while (time.time() - wait_start) < max_wait_time:
        result = processor.get_result()
        if result:
            results.append(result)
            logger.info(f"✅ Result received after {(time.time() - wait_start)*1000:.1f}ms")
            break
        time.sleep(check_interval)
        
        # Log progress every 2 seconds
        elapsed = time.time() - wait_start
        if int(elapsed) % 2 == 0 and elapsed > 2:
            stats = processor.get_performance_stats()
            logger.info(f"⏳ Still waiting... processed: {stats['total_processed']}, queue: {stats['queue_size']}")
    
    total_time = (time.time() - start_time) * 1000
    
    # Show results
    if results:
        frame_id, (pose_coords, root_depth) = results[-1]
        logger.info(f"🎉 SUCCESS: {len(results)} results in {total_time:.1f}ms")
        logger.info(f"   Frame ID: {frame_id}")
        logger.info(f"   Pose shape: {pose_coords.shape}")
        logger.info(f"   Root depth: {root_depth:.1f}mm")
        
        # Show some sample coordinates
        if len(pose_coords) > 0:
            logger.info("   Sample 2D coordinates:")
            for i in range(min(3, len(pose_coords))):
                x, y = pose_coords[i]
                logger.info(f"     Joint {i}: ({x:.1f}, {y:.1f})")
    else:
        logger.error("❌ FAILED: No results received")
    
    # Show comprehensive stats
    stats = processor.get_performance_stats()
    logger.info("📊 Final Performance Stats:")
    for key, value in stats.items():
        logger.info(f"   {key}: {value}")
    
    # Cleanup
    try:
        processor.stop()
    except Exception as e:
        logger.warning(f"⚠️ Cleanup warning: {e}")

if __name__ == "__main__":
    main()