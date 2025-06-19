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

# Fix protobuf compatibility issue for TFLite
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

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

# YOLO Detection
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass

# YOLO for person detection (from v3)
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

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

# Try to import the corrected converter using onnx-tf
CORRECTED_CONVERTER_AVAILABLE = False
try:
    from corrected_onnx_to_tflite_converter import convert_onnx_to_tflite_corrected
    CORRECTED_CONVERTER_AVAILABLE = True
    logger.info("✅ Corrected ONNX→TFLite converter (onnx-tf) available")
except ImportError:
    logger.warning("⚠️ Corrected ONNX→TFLite converter not available, using fallback")
except Exception as e:
    logger.warning(f"⚠️ Error loading corrected converter: {e}")

# Try to import the automatic converter (tf2onnx - incorrect but fallback)
# AUTOMATIC_CONVERTER_AVAILABLE = False
# try:
#     from automatic_onnx_to_tflite_converter import convert_onnx_to_tflite_automatic
#     AUTOMATIC_CONVERTER_AVAILABLE = True
#     logger.info("✅ Automatic ONNX→TFLite converter available (fallback)")
# except ImportError:
#     logger.warning("⚠️ Automatic ONNX→TFLite converter not available")
# except Exception as e:
#     logger.warning(f"⚠️ Error loading automatic converter: {e}")
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

def convert_yolo_to_onnx_optimized(pt_model_path='yolov11n.pt', 
                                   conf_thresh=0.3, iou_thresh=0.45, img_size=640):
    """Convert YOLO to ONNX with optimizations and auto-fallback (improved)"""
    base_name = pt_model_path.replace('.pt', '')
    onnx_path = f"{base_name}_optimized_conf{conf_thresh}_iou{iou_thresh}.onnx"
    
    # Check if optimized ONNX already exists
    if os.path.exists(onnx_path):
        logger.info(f"✅ ONNX optimized model exists: {onnx_path}")
        return onnx_path
    
    # Try to create ONNX model
    logger.info(f"🔄 Converting {pt_model_path} to optimized ONNX...")
    try:
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not available - will use PyTorch fallback")
            
        # Download model if it doesn't exist
        if not os.path.exists(pt_model_path):
            logger.info(f"📥 Downloading YOLO model: {pt_model_path}")
            model = YOLO(pt_model_path)  # This will download automatically
        else:
            model = YOLO(pt_model_path)
        
        # Export with multiple fallback strategies
        export_strategies = [
            # Strategy 1: Full optimization
            {
                'format': 'onnx', 'imgsz': img_size, 'optimize': True, 
                'half': False, 'dynamic': False, 'simplify': True, 
                'opset': 13, 'nms': True, 'conf': conf_thresh, 
                'iou': iou_thresh, 'max_det': 100
            },
            # Strategy 2: Reduced optimization
            {
                'format': 'onnx', 'imgsz': img_size, 'optimize': True, 
                'half': False, 'dynamic': False, 'simplify': False, 
                'opset': 11, 'nms': False
            },
            # Strategy 3: Basic export
            {
                'format': 'onnx', 'imgsz': img_size, 'optimize': False, 
                'half': False, 'dynamic': False
            }
        ]
        
        exported_path = None
        for i, strategy in enumerate(export_strategies):
            try:
                logger.info(f"🔄 Trying export strategy {i+1}/3...")
                exported_path = model.export(**strategy)
                logger.info(f"✅ Export strategy {i+1} successful")
                break
            except Exception as e:
                logger.warning(f"⚠️ Export strategy {i+1} failed: {e}")
                if i == len(export_strategies) - 1:
                    raise
        
        # Rename to expected path
        if exported_path and exported_path != onnx_path:
            if os.path.exists(exported_path):
                os.rename(exported_path, onnx_path)
            else:
                # Sometimes the export path is different, try to find it
                possible_paths = [
                    exported_path,
                    pt_model_path.replace('.pt', '.onnx'),
                    f"{base_name}.onnx"
                ]
                for possible_path in possible_paths:
                    if os.path.exists(possible_path):
                        os.rename(possible_path, onnx_path)
                        break
                else:
                    raise FileNotFoundError(f"Exported ONNX model not found at expected locations")
        
        logger.info(f"✅ ONNX optimized model created: {onnx_path}")
        return onnx_path
        
    except Exception as e:
        logger.warning(f"⚠️ ONNX conversion failed: {e}")
        logger.info("🔄 Will use PyTorch YOLO fallback")
        return None  # Signal to use PyTorch fallback

class AdaptiveYOLODetector:
    """YOLO detector optimized for person detection with auto-fallback (improved)"""
    
    def __init__(self, hardware_caps: Dict[str, Any], yolo_model_path: str = 'yolov11n.pt'):
        self.hardware_caps = hardware_caps
        self.yolo_model_path = yolo_model_path
        self.session = None
        self.model = None
        self.engine_type = "YOLO Failed"
        
        # Try initialization strategies in order of preference
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize detector with fallback strategies"""
        strategies = [
            ("ONNX Runtime", self._try_onnx_initialization),
            ("PyTorch", self._try_pytorch_initialization),
            ("Fallback", self._try_fallback_initialization)
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                logger.info(f"🔄 Trying {strategy_name} YOLO initialization...")
                if strategy_func():
                    self.engine_type = f"YOLO {strategy_name}"
                    logger.info(f"✅ {strategy_name} YOLO detector initialized successfully")
                    return
            except Exception as e:
                logger.warning(f"⚠️ {strategy_name} YOLO initialization failed: {e}")
                continue
        
        # If all strategies fail, create a dummy detector
        logger.error("❌ All YOLO initialization strategies failed - using full frame fallback")
        self.engine_type = "YOLO Disabled"
    
    def _try_onnx_initialization(self) -> bool:
        """Try to initialize ONNX YOLO detector"""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available")
        
        # Try to convert to ONNX
        self.onnx_path = convert_yolo_to_onnx_optimized(self.yolo_model_path)
        if self.onnx_path is None:
            raise RuntimeError("ONNX conversion failed")
        
        # Setup ONNX session
        self._setup_onnx_session()
        return True
    
    def _try_pytorch_initialization(self) -> bool:
        """Try to initialize PyTorch YOLO detector"""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not available")
        
        # Download model if needed and initialize
        self._setup_pytorch_model(self.yolo_model_path)
        return True
    
    def _try_fallback_initialization(self) -> bool:
        """Last resort fallback - try basic PyTorch with alternative models"""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("No YOLO backend available")
        
        # Try alternative model sizes
        fallback_models = ['yolov8n.pt', 'yolov8s.pt', 'yolo11n.pt']
        
        for model_path in fallback_models:
            try:
                logger.info(f"🔄 Trying fallback model: {model_path}")
                self.model = YOLO(model_path)
                self.img_size = 640
                logger.info(f"✅ Fallback model {model_path} loaded successfully")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Fallback model {model_path} failed: {e}")
                continue
        
        raise RuntimeError("All fallback models failed")
    
    def _setup_onnx_session(self):
        """Setup ONNX Runtime session for YOLO"""
        providers = []
        session_options = ort.SessionOptions()
        
        if self.hardware_caps['has_cuda']:
            cuda_provider = ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': int(self.hardware_caps['cuda_memory_gb'] * 0.3 * 1024**3),
                'cudnn_conv_algo_search': 'HEURISTIC',
            })
            providers.append(cuda_provider)
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        else:
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
        providers.append('CPUExecutionProvider')
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(self.onnx_path, providers=providers, sess_options=session_options)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.img_size = self.input_shape[2]
        
        # Warmup
        dummy_input = np.random.rand(*self.input_shape).astype(np.float32)
        for _ in range(3):
            self.session.run(self.output_names, {self.input_name: dummy_input})
    
    def _setup_pytorch_model(self, model_path: str):
        """Setup PyTorch YOLO with auto-download"""
        try:
            self.model = YOLO(model_path)
            self.img_size = 640
            
            # Test the model with a dummy image to ensure it works
            dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            test_results = self.model(dummy_frame, conf=0.5, classes=[0])
            logger.info(f"✅ PyTorch YOLO model {model_path} tested successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ PyTorch YOLO setup failed: {e}")
            raise
    
    def _setup_onnx_session(self):
        """Setup ONNX Runtime session for YOLO"""
        providers = []
        session_options = ort.SessionOptions()
        
        if self.hardware_caps['has_cuda']:
            cuda_provider = ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': int(self.hardware_caps['cuda_memory_gb'] * 0.3 * 1024**3),
                'cudnn_conv_algo_search': 'HEURISTIC',
            })
            providers.append(cuda_provider)
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        else:
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
        providers.append('CPUExecutionProvider')
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(self.onnx_path, providers=providers, sess_options=session_options)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.img_size = self.input_shape[2]
        
        # Warmup
        dummy_input = np.random.rand(*self.input_shape).astype(np.float32)
        for _ in range(3):
            self.session.run(self.output_names, {self.input_name: dummy_input})
    
    def _setup_pytorch_model(self, model_path: str):
        """Fallback to PyTorch YOLO"""
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Neither ONNX nor PyTorch YOLO available")
            
        self.model = YOLO(model_path)
        self.img_size = 640
        
    def detect_persons(self, frame: np.ndarray, conf_threshold: float = 0.3) -> List[List[int]]:
        """Detect persons in frame and return bboxes with intelligent fallback"""
        # If no detector is available, return full frame
        if self.session is None and self.model is None:
            h, w = frame.shape[:2]
            logger.warning("⚠️ No YOLO detector available - using full frame")
            return [[0, 0, w, h]]
        
        try:
            if self.session is not None:
                return self._detect_onnx(frame, conf_threshold)
            elif self.model is not None:
                return self._detect_pytorch(frame, conf_threshold)
            else:
                raise RuntimeError("No detection method available")
                
        except Exception as e:
            logger.warning(f"⚠️ YOLO detection failed: {e} - using full frame fallback")
            h, w = frame.shape[:2]
            return [[0, 0, w, h]]
    
    def _detect_onnx(self, frame: np.ndarray, conf_threshold: float) -> List[List[int]]:
        """ONNX YOLO detection"""
        # Preprocess
        input_tensor, scale, pad_w, pad_h = self._preprocess_frame(frame)
        
        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Post-process
        if len(outputs) == 0 or outputs[0].size == 0:
            return []
            
        detections = outputs[0][0]  # Remove batch dimension
        
        # Filter persons (class 0) with confidence
        person_mask = (detections[:, 5] == 0) & (detections[:, 4] >= conf_threshold)
        person_detections = detections[person_mask]
        
        if len(person_detections) == 0:
            return []
        
        # Convert coordinates back to original frame
        bboxes = []
        h, w = frame.shape[:2]
        
        for det in person_detections:
            x1, y1, x2, y2, conf, cls = det
            
            # Transform coordinates
            x1 = (x1 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            x2 = (x2 - pad_w) / scale
            y2 = (y2 - pad_h) / scale
            
            # Clip to frame bounds
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            
            # Ensure valid bbox
            if x2 > x1 and y2 > y1:
                bboxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        return bboxes
    
    def _detect_pytorch(self, frame: np.ndarray, conf_threshold: float) -> List[List[int]]:
        """PyTorch YOLO detection (fallback)"""
        results = self.model(frame, conf=conf_threshold, classes=[0])  # Only person class
        
        bboxes = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box
                bboxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        return bboxes
    
    def _preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """Preprocess frame for YOLO ONNX"""
        h, w = frame.shape[:2]
        scale = min(self.img_size / w, self.img_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad
        pad_w = (self.img_size - new_w) // 2
        pad_h = (self.img_size - new_h) // 2
        
        padded = cv2.copyMakeBorder(
            resized, pad_h, self.img_size - new_h - pad_h, 
            pad_w, self.img_size - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Normalize and transpose
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
        
        return input_tensor, scale, pad_w, pad_h

class YOLODetector:
    """YOLO detector integrado de v3 para v4"""
    
    def __init__(self, model_path: str = 'yolov11n.pt', conf_thresh: float = 0.3):
        self.conf_thresh = conf_thresh
        self.model = None
        
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                logger.info(f"✅ YOLO detector loaded: {model_path}")
            except Exception as e:
                logger.warning(f"⚠️ YOLO loading failed: {e}")
                self.model = None
        else:
            logger.warning("⚠️ YOLO not available - will use full frame")
    
    def detect_persons(self, frame: np.ndarray) -> List[List[int]]:
        """Detect person bboxes in frame"""
        if self.model is None:
            # Fallback to full frame
            h, w = frame.shape[:2]
            return [[0, 0, w, h]]
        
        try:
            results = self.model(frame, conf=self.conf_thresh, classes=[0])  # class 0 = person
            
            bboxes = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        if box.cls[0] == 0:  # person class
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bboxes.append([int(x1), int(y1), int(x2), int(y2)])
            
            # If no persons detected, use full frame
            if not bboxes:
                h, w = frame.shape[:2]
                bboxes = [[0, 0, w, h]]
                
            return bboxes
            
        except Exception as e:
            logger.warning(f"⚠️ YOLO detection failed: {e}")
            h, w = frame.shape[:2]
            return [[0, 0, w, h]]

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
    """Thread-safe frame processor with improved architecture and YOLO integration"""
    
    def __init__(self, model_path: str, use_tflite: bool = False, yolo_model: str = 'yolov11n.pt'):
        self.hardware_caps = detect_hardware_capabilities()
        
        # Initialize YOLO detector (CRITICAL for performance) - Use AdaptiveYOLODetector for robustness
        self.yolo_detector = AdaptiveYOLODetector(self.hardware_caps, yolo_model)
        logger.info("✅ AdaptiveYOLO detector integrated for bbox optimization")
        
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
        """Add frame for processing with YOLO person detection (improved from v3)"""
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
        
        # Use YOLO to detect person bboxes (CRITICAL OPTIMIZATION from v3)
        if bbox is None:
            person_bboxes = self.yolo_detector.detect_persons(frame)
            logger.info(f"🎯 YOLO detected {len(person_bboxes)} person(s) in frame {self.frame_count}")
        else:
            person_bboxes = [bbox]
        
        # Process each detected person bbox (same as v3 approach)
        for i, bbox in enumerate(person_bboxes):
            # Generate cache key for this specific bbox
            cache_key = self.pose_processor.cache_manager.generate_key(bbox, timestamp)
            
            task = ProcessingTask(
                frame=frame.copy(),
                bbox=bbox,  # Optimized bbox from YOLO
                timestamp=timestamp,
                frame_id=f"{self.frame_count}_{i}",  # Unique ID per person
                cache_key=cache_key
            )
            
            logger.info(f"📝 Adding person {i} from frame {self.frame_count} (bbox: {bbox[:2]}...{bbox[2:]}) to queue")
            
            try:
                # Non-blocking put with fallback
                self.input_queue.put_nowait(task)
                logger.info(f"✅ Person {i} from frame {self.frame_count} added to queue successfully")
            except queue.Full:
                # Queue is full, try to remove oldest task
                try:
                    self.input_queue.get_nowait()  # Remove oldest
                    self.input_queue.put_nowait(task)  # Add new
                    logger.info(f"🔄 Person {i} from frame {self.frame_count} replaced older task in queue")
                except:
                    logger.warning(f"⚠️ Could not add person {i} from frame {self.frame_count} - queue full")
                    pass  # Skip this detection if still can't add
    
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
    """Test thread-safe v4 with YOLO integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Thread-Safe ConvNeXt v4 with YOLO')
    parser.add_argument('--model_path', type=str, 
                       default='/home/fabri/ConvNeXtPose/exports/model_opt_S.pth',
                       help='Path to pose model')
    parser.add_argument('--image_path', type=str, 
                       default='/home/fabri/ConvNeXtPose/demo/input.jpg',
                       help='Path to input image')
    parser.add_argument('--use_tflite', action='store_true',
                       help='Use TensorFlow Lite if available')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt',
                       help='YOLO model for person detection')
    
    args = parser.parse_args()
    
    # Create improved processor with YOLO integration
    processor = ThreadSafeFrameProcessor(args.model_path, 
                                       use_tflite=args.use_tflite, 
                                       yolo_model=args.yolo_model)
    # Esqueleto optimizado
    skeleton = [
        (10, 9), (9, 8), (8, 11), (8, 14),
        (11, 12), (12, 13), (14, 15), (15, 16),
        (11, 4), (14, 1), (0, 4), (0, 1),
        (4, 5), (5, 6), (1, 2), (2, 3)
    ]
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
    max_wait_time = 20.0  # 20 seconds max
    wait_start = time.time()
    check_interval = 0.05  # Check every 50ms
    coords = []
    logger.info("⏳ Waiting for processing results...")
    
    while (time.time() - wait_start) < max_wait_time:
        result = processor.get_result()
        if result:
            frame_id, (coords, root_depth) = result
            results.append(result)
            logger.info(f"✅ Result received after {(time.time() - wait_start)*1000:.1f}ms")
            break
        for pose_coords in coords:
                if pose_coords is not None:
                    pose_coords = np.clip(pose_coords, 0, [frame.shape[1]-1, frame.shape[0]-1])
                    
                    # Dibujar conexiones
                    for i, j in skeleton:
                        if i < len(pose_coords) and j < len(pose_coords):
                            pt1 = tuple(map(int, pose_coords[i]))
                            pt2 = tuple(map(int, pose_coords[j]))
                            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                    
                    # Dibujar articulaciones
                    for point in pose_coords:
                        cv2.circle(frame, tuple(map(int, point)), 3, (0, 255, 0), -1)
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

def convert_convnext_to_optimized_formats(model_path: str, input_shape: tuple = (1, 3, 256, 256)):
    """Convert ConvNeXt model to ONNX and TFLite formats with fallback"""
    base_name = model_path.replace('.pth', '')
    onnx_path = f"{base_name}_optimized.onnx"
    tflite_path = f"{base_name}_optimized.tflite"
    
    results = {
        'onnx_path': None,
        'tflite_path': None,
        'onnx_success': False,
        'tflite_success': False
    }
    
    # Check if models already exist
    if os.path.exists(onnx_path):
        logger.info(f"✅ ONNX model exists: {onnx_path}")
        results['onnx_path'] = onnx_path
        results['onnx_success'] = True
    
    if os.path.exists(tflite_path):
        logger.info(f"✅ TFLite model exists: {tflite_path}")
        results['tflite_path'] = tflite_path
        results['tflite_success'] = True
    
    # If both exist, return early
    if results['onnx_success'] and results['tflite_success']:
        return results
    
    # Load PyTorch model for conversion
    try:
        from config import cfg
        from model import get_pose_net
        
        device = torch.device('cpu')  # Use CPU for conversion
        model = get_pose_net(cfg, is_train=False, joint_num=18)
        
        state = torch.load(model_path, map_location=device)
        state_dict = state.get('network', state)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device).eval()
        
        logger.info(f"✅ PyTorch model loaded for conversion: {model_path}")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=device)
        
        # Convert to ONNX if needed
        if not results['onnx_success']:
            results.update(convert_to_onnx(model, dummy_input, onnx_path))
        
        # Convert to TFLite if needed  
        if not results['tflite_success']:
            results.update(convert_to_tflite(model, dummy_input, tflite_path, onnx_path))
            
    except Exception as e:
        logger.warning(f"⚠️ Model conversion setup failed: {e}")
    
    return results

def convert_to_onnx(model, dummy_input, onnx_path: str) -> dict:
    """Convert PyTorch model to ONNX with multiple strategies"""
    if not ONNX_AVAILABLE:
        logger.warning("⚠️ ONNX not available for conversion")
        return {'onnx_success': False, 'onnx_path': None}
    
    # ONNX export strategies
    strategies = [
        # Strategy 1: Full optimization
        {
            'opset_version': 13,
            'do_constant_folding': True,
            'input_names': ['input'],
            'output_names': ['output'],
            'dynamic_axes': None
        },
        # Strategy 2: Basic export
        {
            'opset_version': 11,
            'do_constant_folding': False,
            'input_names': ['input'],
            'output_names': ['output']
        },
        # Strategy 3: Minimal export
        {
            'opset_version': 10
        }
    ]
    
    for i, strategy in enumerate(strategies):
        try:
            logger.info(f"🔄 Trying ONNX export strategy {i+1}/3...")
            
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                verbose=False,
                **strategy
            )
            
            # Verify the exported model
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            test_output = session.run(None, {'input': dummy_input.numpy()})
            
            logger.info(f"✅ ONNX export strategy {i+1} successful: {onnx_path}")
            return {'onnx_success': True, 'onnx_path': onnx_path}
            
        except Exception as e:
            logger.warning(f"⚠️ ONNX export strategy {i+1} failed: {e}")
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
            continue
    
    logger.warning("⚠️ All ONNX export strategies failed")
    return {'onnx_success': False, 'onnx_path': None}

def convert_to_tflite(model, dummy_input, tflite_path: str, onnx_path: str = None) -> dict:
    """Convert to TFLite with multiple strategies"""
    if not TFLITE_AVAILABLE:
        logger.warning("⚠️ TensorFlow not available for TFLite conversion")
        return {'tflite_success': False, 'tflite_path': None}
    
    strategies = []
    
    # Strategy 1: Convert from ONNX if available
    if onnx_path and os.path.exists(onnx_path):
        strategies.append(('ONNX->TFLite', lambda: convert_onnx_to_tflite(onnx_path, tflite_path)))
    
    # Strategy 2: Convert from PyTorch via ONNX
    # strategies.append(('PyTorch->ONNX->TFLite', lambda: convert_pytorch_to_tflite(model, dummy_input, tflite_path)))
    
    for strategy_name, strategy_func in strategies:
        try:
            logger.info(f"🔄 Trying TFLite conversion: {strategy_name}...")
            if strategy_func():
                logger.info(f"✅ TFLite conversion successful: {tflite_path}")
                return {'tflite_success': True, 'tflite_path': tflite_path}
        except Exception as e:
            logger.warning(f"⚠️ TFLite conversion {strategy_name} failed: {e}")
            if os.path.exists(tflite_path):
                os.remove(tflite_path)
            continue
    
    logger.warning("⚠️ All TFLite conversion strategies failed")
    return {'tflite_success': False, 'tflite_path': None}

def convert_onnx_to_tflite(onnx_path: str, tflite_path: str) -> bool:
    """Convert ONNX to TFLite using corrected onnx-tf method"""
    try:
        # Prioritize corrected converter using onnx-tf
        if CORRECTED_CONVERTER_AVAILABLE:
            logger.info(f"🔄 Converting ONNX to TFLite with corrected onnx-tf converter: {onnx_path} -> {tflite_path}")
            
            # Use the corrected converter with onnx-tf
            result = convert_onnx_to_tflite_corrected(
                onnx_path=onnx_path,
                tflite_path=tflite_path,
                optimization="default"
            )
            
            if result['success']:
                logger.info(f"✅ ONNX->TFLite conversion successful with {result['strategy_used']}")
                logger.info(f"📊 TFLite model size: {result['file_size_mb']:.2f} MB")
                logger.info("🔍 Model weights preserved from original ONNX")
                return True
            else:
                logger.warning(f"⚠️ Corrected conversion failed: {result.get('error', 'Unknown error')}")
                # Fall through to other methods
        
        # Fallback to automatic converter (tf2onnx - conceptually incorrect but may work)
        # elif AUTOMATIC_CONVERTER_AVAILABLE:
        #     logger.warning("⚠️ Using tf2onnx-based converter (conceptually incorrect but available)")
            
        #     result = convert_onnx_to_tflite_automatic(
        #         onnx_path=onnx_path,
        #         tflite_path=tflite_path,
        #         optimization="default"
        #     )
            
        #     if result['success']:
        #         logger.warning(f"⚠️ Conversion completed with tf2onnx fallback: {result['strategy_used']}")
        #         return True
        #     else:
        #         logger.error(f"❌ tf2onnx fallback also failed: {result.get('error', 'Unknown error')}")
        else:
            logger.warning("⚠️ No automatic converters available, using legacy method")
            return False
        # Final fallback to legacy conversion
        # return convert_onnx_to_tflite_legacy(onnx_path, tflite_path)
            
    except Exception as e:
        logger.warning(f"⚠️ Conversion error: {e}")
        #return convert_onnx_to_tflite_legacy(onnx_path, tflite_path)
        return False

# def convert_onnx_to_tflite_legacy(onnx_path: str, tflite_path: str) -> bool:
#     """Legacy ONNX to TFLite conversion (fallback)"""
#     try:
#         import tensorflow as tf
        
#         logger.info(f"🔄 Using legacy ONNX->TFLite conversion: {onnx_path} -> {tflite_path}")
        
#         # Create generic ConvNeXt model as fallback
#         tf_model = create_convnext_tf_equivalent([1, 3, 256, 192])
        
#         # Convert to TFLite
#         converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
#         converter.optimizations = [tf.lite.Optimize.DEFAULT]
#         converter.allow_custom_ops = True
        
#         tflite_model = converter.convert()
        
#         # Save TFLite model
#         with open(tflite_path, 'wb') as f:
#             f.write(tflite_model)
        
#         logger.warning("⚠️ Legacy conversion used - weights may not be preserved")
#         return True
        
#     except Exception as e:
#         logger.error(f"❌ Legacy ONNX->TFLite conversion failed: {e}")
#         return False

# def create_convnext_tf_equivalent(input_shape):
#     """Create TensorFlow equivalent of ConvNeXt architecture"""
#     import tensorflow as tf
    
#     # Create a ConvNeXt-like architecture in TensorFlow
#     inputs = tf.keras.layers.Input(shape=input_shape[1:])  # Remove batch dimension
    
#     x = inputs
    
#     # Stem layer
#     x = tf.keras.layers.Conv2D(96, 4, strides=4, padding='same')(x)
#     x = tf.keras.layers.LayerNormalization()(x)
    
#     # ConvNeXt blocks - Stage 1
#     for _ in range(3):
#         residual = x
#         # Depthwise convolution
#         x = tf.keras.layers.DepthwiseConv2D(7, padding='same')(x)
#         x = tf.keras.layers.LayerNormalization()(x)
#         # MLP layers
#         x = tf.keras.layers.Conv2D(384, 1, activation='gelu')(x)
#         x = tf.keras.layers.Conv2D(96, 1)(x)
#         # Residual connection
#         x = tf.keras.layers.Add()([x, residual])
    
#     # Downsampling to Stage 2
#     x = tf.keras.layers.LayerNormalization()(x)
#     x = tf.keras.layers.Conv2D(192, 2, strides=2, padding='same')(x)
    
#     # ConvNeXt blocks - Stage 2
#     for _ in range(3):
#         residual = x
#         x = tf.keras.layers.DepthwiseConv2D(7, padding='same')(x)
#         x = tf.keras.layers.LayerNormalization()(x)
#         x = tf.keras.layers.Conv2D(768, 1, activation='gelu')(x)
#         x = tf.keras.layers.Conv2D(192, 1)(x)
#         x = tf.keras.layers.Add()([x, residual])
    
#     # Downsampling to Stage 3
#     x = tf.keras.layers.LayerNormalization()(x)
#     x = tf.keras.layers.Conv2D(384, 2, strides=2, padding='same')(x)
    
#     # ConvNeXt blocks - Stage 3
#     for _ in range(9):
#         residual = x
#         x = tf.keras.layers.DepthwiseConv2D(7, padding='same')(x)
#         x = tf.keras.layers.LayerNormalization()(x)
#         x = tf.keras.layers.Conv2D(1536, 1, activation='gelu')(x)
#         x = tf.keras.layers.Conv2D(384, 1)(x)
#         x = tf.keras.layers.Add()([x, residual])
    
#     # Downsampling to Stage 4
#     x = tf.keras.layers.LayerNormalization()(x)
#     x = tf.keras.layers.Conv2D(768, 2, strides=2, padding='same')(x)
    
#     # ConvNeXt blocks - Stage 4
#     for _ in range(3):
#         residual = x
#         x = tf.keras.layers.DepthwiseConv2D(7, padding='same')(x)
#         x = tf.keras.layers.LayerNormalization()(x)
#         x = tf.keras.layers.Conv2D(3072, 1, activation='gelu')(x)
#         x = tf.keras.layers.Conv2D(768, 1)(x)
#         x = tf.keras.layers.Add()([x, residual])
    
#     # Global average pooling and final layers for pose estimation
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
#     x = tf.keras.layers.LayerNormalization()(x)
    
#     # Pose estimation head
#     x = tf.keras.layers.Dense(1024, activation='gelu')(x)
#     x = tf.keras.layers.Dropout(0.1)(x)
#     x = tf.keras.layers.Dense(512, activation='gelu')(x)
#     x = tf.keras.layers.Dense(17 * 3)(x)  # 17 joints * 3 coordinates
    
#     # Reshape for pose output
#     outputs = tf.keras.layers.Reshape((17, 3))(x)
    
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model

# def convert_pytorch_to_tflite(model, dummy_input, tflite_path: str) -> bool:
#     """Convert PyTorch ConvNeXt to TFLite via TensorFlow recreation"""
#     try:
#         import tensorflow as tf
        
#         logger.info(f"🔄 Converting PyTorch to TFLite: {tflite_path}")
        
#         # Get input shape from dummy input
#         input_shape = list(dummy_input.shape)
#         logger.info(f"PyTorch model input shape: {input_shape}")
        
#         # Create equivalent TensorFlow model
#         tf_model = create_convnext_tf_equivalent(input_shape)
        
#         # Compile model (required for some TFLite conversions)
#         tf_model.compile(optimizer='adam', loss='mse')
        
#         logger.info(f"TensorFlow equivalent model created with {tf_model.count_params():,} parameters")
        
#         # Convert to TFLite with optimizations
#         converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
#         converter.optimizations = [tf.lite.Optimize.DEFAULT]
#         converter.target_spec.supported_types = [tf.float16]
        
#         # Additional optimizations for ConvNeXt
#         converter.allow_custom_ops = True
#         converter.experimental_new_converter = True
        
#         # Convert
#         tflite_model = converter.convert()
        
#         # Save TFLite model
#         with open(tflite_path, 'wb') as f:
#             f.write(tflite_model)
        
#         # Validate the converted model
#         interpreter = tf.lite.Interpreter(model_path=tflite_path)
#         interpreter.allocate_tensors()
        
#         input_details = interpreter.get_input_details()
#         output_details = interpreter.get_output_details()
        
#         logger.info(f"✅ PyTorch->TFLite conversion successful")
#         logger.info(f"   Input: {input_details[0]['shape']} {input_details[0]['dtype']}")
#         logger.info(f"   Output: {output_details[0]['shape']} {output_details[0]['dtype']}")
#         logger.info(f"   Model size: {len(tflite_model) / 1024:.1f} KB")
        
#         return True
        
#     except Exception as e:
#         logger.warning(f"⚠️ PyTorch->TFLite conversion failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

class ModelConverter:
    """Model converter that ensures all model formats are available"""
    
    def __init__(self):
        self.base_model_path = "/home/fabri/ConvNeXtPose/exports/model_opt_S.pth"
        self.onnx_path = "/home/fabri/ConvNeXtPose/exports/model_opt_S_optimized.onnx"
        self.tflite_path = "/home/fabri/ConvNeXtPose/exports/model_opt_S_optimized.tflite"
    
    def ensure_all_models_ready(self) -> dict:
        """Ensure all model formats are available with real ConvNeXt conversion"""
        logger.info("🔄 Ensuring all model formats are ready...")
        
        results = {
            'pytorch_available': os.path.exists(self.base_model_path),
            'onnx_available': os.path.exists(self.onnx_path),
            'tflite_available': os.path.exists(self.tflite_path),
            'conversion_attempted': False,
            'conversion_results': {}
        }
        
        if not results['pytorch_available']:
            logger.error(f"❌ Base PyTorch model not found: {self.base_model_path}")
            return results
        
        # Convert missing models
        missing_models = []
        if not results['onnx_available']:
            missing_models.append('ONNX')
        if not results['tflite_available']:
            missing_models.append('TFLite')
        
        if missing_models:
            logger.info(f"🔄 Converting missing models: {missing_models}")
            results['conversion_attempted'] = True
            
            # Use the real conversion function from V4
            conversion_results = convert_convnext_to_optimized_formats(
                self.base_model_path, 
                input_shape=(1, 3, 256, 192)
            )
            
            results['conversion_results'] = conversion_results
            results['onnx_available'] = conversion_results.get('onnx_success', False)
            results['tflite_available'] = conversion_results.get('tflite_success', False)
        
        # Final verification
        results['onnx_available'] = os.path.exists(self.onnx_path)
        results['tflite_available'] = os.path.exists(self.tflite_path)
        
        logger.info("📊 Model availability status:")
        logger.info(f"   PyTorch: {'✅' if results['pytorch_available'] else '❌'}")
        logger.info(f"   ONNX: {'✅' if results['onnx_available'] else '❌'}")
        logger.info(f"   TFLite: {'✅' if results['tflite_available'] else '❌'}")
        
        return results