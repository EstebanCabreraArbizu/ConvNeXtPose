#!/usr/bin/env python3
"""
convnext_realtime_v4_production_optimized.py - Versión híbrida ESTABLE para producción

🎯 COMBINA:
- Estabilidad y post-procesamiento correcto de Final Working
- Optimizaciones de rendimiento de Ultra Optimized
- Threading controlado sin complejidad excesiva

🔧 OPTIMIZACIONES APLICADAS:
- Frame skipping inteligente
- Multi-th        # Fase 2: Fallback a modelos con SELECT_TF_OPS (lentos pero estables)
        logger.warning("⚠️ No native TFLite models worked, trying SELECT_TF_OPS models (slower)...")
        for candidate in flex_candidates:
            if candidate.exists():
                logger.info(f"🔄 Testing TFLite model (SELECT_TF_OPS): {candidate.name}")
                success, is_flex = self._try_load_tflite_model(candidate, is_native_preferred=False)
                if success:
                    logger.warning(f"⚠️ FALLBACK: Using SELECT_TF_OPS model: {candidate.name}")
                    logger.warning("🐌 This model will be SLOWER (~0.2 FPS) due to TensorFlow fallback")
                    logger.info("💡 Consider using --backend onnx for better performance")
                    return Truentrolado
- Cache de detecciones optimizado
- Backends múltiples estables

💡 USO:
python convnext_realtime_v4_production_optimized.py --preset ultra_fast --backend pytorch
"""

import os
import sys
import time
import logging
import argparse
import warnings
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

# Setup
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '4'
warnings.filterwarnings("ignore", category=UserWarning)

# ONNX Runtime
try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# TensorFlow Lite  
TFLITE_AVAILABLE = False
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project imports
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.extend([
    str(PROJECT_ROOT / 'main'),
    str(PROJECT_ROOT / 'data'),
    str(PROJECT_ROOT / 'common'),
    str(ROOT)
])

try:
    from config import cfg
    from model import get_pose_net
    from dataset import generate_patch_image
    from utils.pose_utils import process_bbox  # Import the working function
except ImportError as e:
    logger.error(f"Critical: Cannot import required modules: {e}")
    sys.exit(1)

# PRODUCTION PRESETS - Hybrid approach: stable + optimized
PRODUCTION_PRESETS = {
    'ultra_fast': {
        'target_fps': 15.0,
        'frame_skip': 3,
        'yolo_size': 320,
        'max_persons': 2,
        'detection_freq': 4,
        'thread_count': 2,
        'enable_threading': True,
        'description': 'Ultra rápido - 15+ FPS con estabilidad'
    },
    'speed_balanced': {
        'target_fps': 12.0,
        'frame_skip': 2,
        'yolo_size': 416,
        'max_persons': 3,
        'detection_freq': 3,
        'thread_count': 2,
        'enable_threading': True,
        'description': 'Balance velocidad-calidad - 12+ FPS'
    },
    'quality_focused': {
        'target_fps': 10.0,
        'frame_skip': 1,
        'yolo_size': 512,
        'max_persons': 4,
        'detection_freq': 2,
        'thread_count': 1,
        'enable_threading': False,
        'description': 'Mejor calidad - 10+ FPS sin threading'
    }
}

def detect_optimized_hardware():
    """Detectar hardware y configurar optimizaciones"""
    hardware_info = {
        'has_cuda': torch.cuda.is_available(),
        'cpu_count': os.cpu_count(),
        'torch_threads': torch.get_num_threads()
    }
    
    # Optimize torch threads
    if hardware_info['cpu_count'] >= 8:
        torch.set_num_threads(6)
    elif hardware_info['cpu_count'] >= 4:
        torch.set_num_threads(4)
    else:
        torch.set_num_threads(2)
    
    if hardware_info['has_cuda']:
        return 'gpu_available'
    else:
        return 'cpu_optimized'

class ProductionYOLODetector:
    """Detector YOLO optimizado pero estable"""
    
    def __init__(self, model_path: str = 'yolo11n.pt', input_size: int = 320):
        self.model_path = model_path
        self.input_size = input_size
        self.detector = None
        self.frame_count = 0
        self.last_detections = []
        self.detection_cache = {}  # Cache simple para optimización
        
        self._initialize()
    
    def _initialize(self):
        if not YOLO_AVAILABLE:
            logger.warning("⚠️ YOLO not available")
            return
        
        try:
            self.detector = YOLO(self.model_path)
            # Configure for optimal CPU performance
            self.detector.overrides['imgsz'] = self.input_size
            self.detector.overrides['half'] = False
            self.detector.overrides['device'] = 'cpu'
            
            logger.info("✅ Production YOLO initialized: %s (size: %d)", self.model_path, self.input_size)
        except Exception as e:
            logger.error("❌ YOLO initialization failed: %s", e)
    
    def detect_persons(self, frame: np.ndarray, detection_freq: int = 4, 
                      conf_threshold: float = 0.4) -> List[List[int]]:
        """Detección con cache optimizado"""
        if self.detector is None:
            return []
        
        self.frame_count += 1
        
        # Use cached detections for performance (same as Final Working)
        if self.frame_count % detection_freq != 0:
            return self.last_detections
        
        try:
            # Optimization: resize frame if too large for faster detection
            detection_frame = frame
            scale_factor = 1.0
            
            if frame.shape[0] > 640:
                scale_factor = 640 / frame.shape[0]
                new_height = int(frame.shape[0] * scale_factor)
                new_width = int(frame.shape[1] * scale_factor)
                detection_frame = cv2.resize(frame, (new_width, new_height))
            
            results = self.detector(detection_frame, verbose=False)
            
            persons = []
            for result in results:
                for box in result.boxes:
                    if box.cls == 0 and box.conf >= conf_threshold:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Scale back to original frame size
                        if scale_factor != 1.0:
                            x1, y1, x2, y2 = [coord / scale_factor for coord in [x1, y1, x2, y2]]
                        
                        persons.append([int(x1), int(y1), int(x2), int(y2)])
            
            self.last_detections = persons
            return persons
            
        except Exception as e:
            logger.warning(f"⚠️ Detection failed: {e}")
            return self.last_detections
    
    def cleanup(self):
        pass

class ProductionInferenceEngine:
    """Motor de inferencia híbrido: estable + optimizado"""
    def __init__(self, model_path: str, backend: str = 'pytorch'):
        self.model_path = model_path
        self.backend = backend
        self.active_backend = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use consistent 256x256 for all backends (from Final Working)
        self.input_size = 256
        self.output_size = 32
        
        # Transform (same as demo.py - CRITICAL)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)
        ])
        
        # Models
        self.pytorch_model = None
        self.onnx_session = None
        self.tflite_interpreter = None
        
        self._initialize()
    
    def _initialize(self):
        logger.info("🚀 Initializing production inference engine...")
        logger.info("   Backend: %s", self.backend)
        if self.backend == 'pytorch':
            self._setup_pytorch()
        elif self.backend == 'onnx' and ONNX_AVAILABLE:
            self._setup_onnx()
        elif self.backend == 'tflite' and TFLITE_AVAILABLE:
            self._setup_tflite()
        
        if self.active_backend is None:
            logger.warning("⚠️ Fallback to PyTorch backend")
            self._setup_pytorch()
        
        logger.info("✅ Production inference engine active: %s", self.active_backend)
    
    def _setup_pytorch(self) -> bool:
        try:
            logger.info("🔄 Setting up PyTorch backend...")
            joint_num = 18
            self.pytorch_model = get_pose_net(cfg, False, joint_num)
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle DataParallel models (like in demo.py)
            if 'network' in checkpoint:
                state_dict = checkpoint['network']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            if any(key.startswith('module.') for key in state_dict.keys()):
                logger.info("🔧 Removing DataParallel prefix from state_dict...")
                state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            
            # Load with strict=False to handle minor architecture differences
            missing_keys, unexpected_keys = self.pytorch_model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning("⚠️ Missing keys in model: %s", missing_keys[:5])
            if unexpected_keys:
                logger.warning("⚠️ Unexpected keys in model: %s", unexpected_keys[:5])
            
            self.pytorch_model.to(self.device)
            self.pytorch_model.eval()
            
            self.active_backend = 'pytorch'
            logger.info("✅ PyTorch backend ready")
            return True            
        except Exception as e:
            logger.error("❌ PyTorch setup failed: %s", e)
            return False
    
    def _setup_onnx(self) -> bool:
        try:
            # Use same logic as Final Working but prioritize 256x256 models
            model_dir = PROJECT_ROOT / "exports"
            onnx_candidates = [
                model_dir / "model_opt_S_optimized.onnx",  # 256x256
                model_dir / "model_S.onnx",                # 256x256  
                model_dir / "model_opt_S.onnx"             # 192x192 (fallback)
            ]
            
            onnx_path = None
            for candidate in onnx_candidates:
                if candidate.exists():
                    onnx_path = candidate
                    break
            
            if onnx_path is None:
                logger.warning("⚠️ No ONNX model found")
                return False
            logger.info("🔄 Setting up ONNX backend...")
            
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
            
            self.active_backend = 'onnx'
            logger.info("✅ ONNX backend ready")
            return True
            
        except Exception as e:
            logger.error("❌ ONNX setup failed: %s", e)
            return False
    
    def _setup_tflite(self) -> bool:
        """Setup TFLite with intelligent fallback: prioritize native ops over SELECT_TF_OPS"""
        model_dir = PROJECT_ROOT / "exports"
          # 🎯 ESTRATEGIA DE FALLBACK INTELIGENTE:
        # 1. Primero intentar modelos NATIVOS TFLite (sin SELECT_TF_OPS) - MÁS RÁPIDOS
        # 2. Si fallan, usar modelos con SELECT_TF_OPS como último recurso - MÁS LENTOS
          # Available models (from directory listing):
        # - model_opt_S_fast_native.tflite
        # - model_opt_S_balanced.tflite
        
        native_candidates = [
            model_dir / "model_opt_S_fast_native.tflite",   # Optimized for speed, should be native
            model_dir / "model_opt_S_balanced.tflite",      # Balanced model, might be native
        ]
        
        flex_candidates = [
            model_dir / "model_opt_S_balanced.tflite",      # Fallback if not native
            model_dir / "model_opt_S_fast_native.tflite",   # Fallback if not native
        ]
        
        # Fase 1: Intentar modelos nativos (rápidos)
        logger.info("🔄 Setting up TFLite backend - Phase 1: Trying NATIVE models (fastest)...")
        for candidate in native_candidates:
            if candidate.exists():
                success, is_flex = self._try_load_tflite_model(candidate, is_native_preferred=True)
                if success:
                    logger.info(f"✅ SUCCESS: Native TFLite model loaded: {candidate.name}")
                    if not is_flex:
                        logger.info("🚀 Using NATIVE TFLite ops - MAXIMUM SPEED!")
                    return True
        
        # Fase 2: Fallback a modelos con SELECT_TF_OPS (lentos pero estables)
        logger.warning("⚠️ No native TFLite models worked, trying SELECT_TF_OPS models (slower)...")
        for candidate in flex_candidates:
            if candidate.exists():
                success, is_flex = self._try_load_tflite_model(candidate, is_native_preferred=False)
                if success:
                    logger.warning(f"⚠️ FALLBACK: Using SELECT_TF_OPS model: {candidate.name}")
                    logger.warning("� This model will be SLOWER (~0.2 FPS) due to TensorFlow fallback")
                    logger.info("💡 Consider using --backend onnx for better performance")
                    return True
        
        logger.error("❌ No TFLite models could be loaded")
        return False
    
    def _try_load_tflite_model(self, model_path: Path, is_native_preferred: bool = True) -> Tuple[bool, bool]:
        """Try to load a TFLite model and detect if it uses SELECT_TF_OPS"""
        try:
            logger.info(f"🔄 Testing TFLite model: {model_path.name}")
            
            # 🚀 OPTIMIZACIONES CRÍTICAS para velocidad TFLite
            interpreter = tf.lite.Interpreter(
                model_path=str(model_path),
                num_threads=4,  # Usar múltiples threads CPU
                experimental_preserve_all_tensors=False  # Reducir memoria
            )
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Test inference con datos dummy para verificar que funciona
            input_shape = input_details[0]['shape']
            dummy_input = np.random.random(input_shape).astype(np.float32)
            
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            
            # Medir tiempo de inferencia para detectar SELECT_TF_OPS
            start_time = time.time()
            interpreter.invoke()
            inference_time = time.time() - start_time
            
            output = interpreter.get_tensor(output_details[0]['index'])
            
            # Detectar si usa SELECT_TF_OPS basado en:
            # 1. Tiempo de inferencia (>100ms indica SELECT_TF_OPS)
            # 2. Nombre del archivo
            # 3. Metadatos del modelo si están disponibles
            is_flex_ops = False
            
            if inference_time > 0.1:  # >100ms indica SELECT_TF_OPS
                is_flex_ops = True
                logger.warning(f"⚠️ Slow inference detected ({inference_time*1000:.1f}ms) - likely SELECT_TF_OPS")
            
            if any(keyword in str(model_path).lower() for keyword in ['configurable', 'enhanced', 'fixed', 'optimized']):
                is_flex_ops = True
                logger.warning(f"⚠️ Model name suggests SELECT_TF_OPS: {model_path.name}")
            
            # Si preferimos nativos y este es FLEX, rechazar
            if is_native_preferred and is_flex_ops:
                logger.info(f"❌ Rejecting {model_path.name} - uses SELECT_TF_OPS (slow)")
                return False, is_flex_ops
            
            # Modelo válido - asignar a la instancia
            self.tflite_interpreter = interpreter
            self.tflite_input_details = input_details
            self.tflite_output_details = output_details
            self.active_backend = 'tflite'
            
            logger.info(f"📊 TFLite model input shape: {input_details[0]['shape']}")
            logger.info(f"📊 TFLite model output shape: {output_details[0]['shape']}")
            logger.info(f"📊 Test inference time: {inference_time*1000:.1f}ms")
            
            if is_flex_ops:
                logger.warning("🐌 Model uses SELECT_TF_OPS - expect slower performance")
            else:
                logger.info("🚀 Model uses native TFLite ops - optimal performance")
            
            return True, is_flex_ops
            
        except Exception as e:
            logger.warning(f"❌ Failed to load {model_path.name}: {e}")
            return False, False
    
    def infer(self, img_patch: np.ndarray) -> Optional[np.ndarray]:
        """Inference using exact same logic as Final Working"""
        if self.active_backend == 'pytorch':
            return self._infer_pytorch(img_patch)
        elif self.active_backend == 'onnx':
            return self._infer_onnx(img_patch)
        elif self.active_backend == 'tflite':
            return self._infer_tflite(img_patch)
        return None
    
    def _infer_pytorch(self, img_patch: np.ndarray) -> np.ndarray:
        # Prepare input (same as demo.py)
        inp = self.transform(img_patch).to(self.device)[None, :, :, :]
        
        # Inference
        with torch.no_grad():
            output = self.pytorch_model(inp)
        
        return output.cpu().numpy()
    
    def _infer_onnx(self, img_patch: np.ndarray) -> np.ndarray:
        # Prepare input
        inp = self.transform(img_patch).numpy()[None, :, :, :]
        
        # ONNX inference
        output = self.onnx_session.run(None, {'input': inp})
        
        return output[0]
    
    def _infer_tflite(self, img_patch: np.ndarray) -> np.ndarray:
        """TFLite inference"""
        if self.tflite_interpreter is None:
            return None
        
        # Prepare input
        inp = self.transform(img_patch).numpy()[None, :, :, :].astype(np.float32)
        
        # Set input tensor
        self.tflite_interpreter.set_tensor(self.tflite_input_details[0]['index'], inp)
        
        # Run inference
        self.tflite_interpreter.invoke()
        
        # Get output tensor
        output = self.tflite_interpreter.get_tensor(self.tflite_output_details[0]['index'])
        
        return output

class ProductionV4Processor:
    """Procesador V4 híbrido: Final Working + Ultra optimizations"""
    
    def __init__(self, model_path: str, preset: str = 'ultra_fast', backend: str = 'pytorch'):
        self.model_path = model_path
        self.preset = preset
        self.backend = backend
        self.config = PRODUCTION_PRESETS[preset]
        self.hardware = detect_optimized_hardware()
        
        # Components
        self.yolo_detector = ProductionYOLODetector(input_size=self.config['yolo_size'])
        self.inference_engine = ProductionInferenceEngine(model_path, backend)
        
        # Threading (controlled)
        self.thread_pool = None
        if self.config['enable_threading'] and self.config['thread_count'] > 1:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config['thread_count'])
        
        # Stats and optimization
        self.frame_count = 0
        self.processing_times = deque(maxlen=50)
        self.last_poses = []
        self.skip_count = 0
        
        logger.info("✅ Production V4 Processor initialized")
        logger.info("   Preset: %s (%s)", preset, self.config['description'])
        logger.info("   Target FPS: %.1f", self.config['target_fps'])
        logger.info("   Hardware: %s", self.hardware)
        logger.info("   Backend: %s", self.inference_engine.active_backend)
        logger.info("   Threading: %s (%d threads)", 
                   "enabled" if self.config['enable_threading'] else "disabled",
                   self.config['thread_count'])
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        start_time = time.time()
        self.frame_count += 1
        
        # Intelligent frame skipping (Ultra optimization)
        if self._should_skip_frame():
            return self.last_poses, self._get_stats(start_time, len(self.last_poses), skipped=True)
        
        # Detection
        detection_start = time.time()
        persons = self.yolo_detector.detect_persons(
            frame, 
            self.config['detection_freq'],
            conf_threshold=0.4
        )
        detection_time = time.time() - detection_start
        
        # Pose estimation with controlled threading
        pose_start = time.time()
        if self.thread_pool and len(persons) > 1:
            poses = self._estimate_poses_threaded(frame, persons[:self.config['max_persons']])
        else:
            poses = self._estimate_poses_single(frame, persons[:self.config['max_persons']])
        
        pose_time = time.time() - pose_start
        
        # Update cache
        self.last_poses = poses
        
        # Stats
        stats = self._get_stats(start_time, len(poses), 
                              detection_time=detection_time, 
                              pose_time=pose_time)
        
        return poses, stats
    
    def _should_skip_frame(self) -> bool:
        """Smart frame skipping (Ultra optimization)"""
        if len(self.processing_times) < 10:
            return False
        
        avg_time = np.mean(list(self.processing_times)[-10:])
        target_time = 1.0 / self.config['target_fps']
        
        # Skip if we're running too slow
        if avg_time > target_time * 1.2:
            self.skip_count += 1
            if self.skip_count < self.config['frame_skip']:
                return True
        
        self.skip_count = 0
        return False
    
    def _estimate_poses_single(self, frame: np.ndarray, persons: List[List[int]]) -> List[np.ndarray]:
        """Single-threaded pose estimation (stable)"""
        poses = []
        
        for bbox in persons:
            pose_2d = self._process_single_person(frame, bbox)
            if pose_2d is not None:
                poses.append(pose_2d)
        
        return poses
    
    def _estimate_poses_threaded(self, frame: np.ndarray, persons: List[List[int]]) -> List[np.ndarray]:
        """Multi-threaded pose estimation (controlled)"""
        if not persons:
            return []
        
        # Submit tasks to thread pool
        futures = []
        for bbox in persons:
            future = self.thread_pool.submit(self._process_single_person, frame, bbox)
            futures.append(future)
        
        # Collect results with timeout
        poses = []
        for future in futures:
            try:
                pose_2d = future.result(timeout=0.2)  # 200ms timeout
                if pose_2d is not None:
                    poses.append(pose_2d)
            except Exception as e:
                logger.warning(f"⚠️ Threaded pose estimation failed: {e}")
                continue
        
        return poses
    
    def _process_single_person(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Process single person using EXACT Final Working logic that works"""
        try:
            # Convert bbox format from YOLO [x1, y1, x2, y2] to ConvNeXt [x, y, w, h]
            x1, y1, x2, y2 = bbox
            bbox_array = np.array([x1, y1, x2 - x1, y2 - y1])
            
            # Process bbox using EXACT function from pose_utils.py that works
            processed_bbox = process_bbox(bbox_array, frame.shape[1], frame.shape[0])
            if processed_bbox is None:
                return None
            
            # Generate patch using EXACT method from demo.py (standard 256x256)
            img_patch, img2bb_trans = generate_patch_image(
                frame, processed_bbox, False, 1.0, 0.0, False
            )
            
            # Inference
            pose_output = self.inference_engine.infer(img_patch)
            if pose_output is None:
                return None
            
            # Process output using EXACT demo.py post-processing
            pose_3d = pose_output[0]  # Extract first (and only) result
            
            # EXACT post-processing from demo.py lines 153-159
            pose_3d = pose_3d.copy()
            pose_3d[:, 0] = pose_3d[:, 0] / cfg.output_shape[1] * cfg.input_shape[1]
            pose_3d[:, 1] = pose_3d[:, 1] / cfg.output_shape[0] * cfg.input_shape[0]
            pose_3d_xy1 = np.concatenate((pose_3d[:, :2], np.ones_like(pose_3d[:, :1])), 1)
            img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0, 0, 1]).reshape(1, 3)))
            pose_3d[:, :2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            
            # Return 2D coordinates (same as demo.py)
            return pose_3d[:, :2].copy()
            
        except Exception as e:
            logger.debug(f"⚠️ Pose processing failed: {e}")
            return None
    
    def _get_stats(self, start_time: float, poses_count: int, 
                   detection_time: float = 0, pose_time: float = 0, 
                   skipped: bool = False) -> Dict[str, Any]:
        """Get processing statistics"""
        total_time = time.time() - start_time
        self.processing_times.append(total_time)
        avg_time = np.mean(self.processing_times)
        
        return {
            'frame_count': self.frame_count,
            'avg_fps': 1.0 / avg_time if avg_time > 0 else 0,
            'instant_fps': 1.0 / total_time if total_time > 0 else 0,
            'target_fps': self.config['target_fps'],
            'processing_time_ms': total_time * 1000,
            'detection_time_ms': detection_time * 1000,
            'pose_time_ms': pose_time * 1000,
            'poses_detected': poses_count,
            'active_backend': self.inference_engine.active_backend,
            'preset': self.preset,
            'hardware': self.hardware,
            'skipped': skipped,
            'threading_enabled': self.config['enable_threading']
        }
    
    def cleanup(self):
        self.yolo_detector.cleanup()
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)

def draw_pose(image: np.ndarray, pose_2d: np.ndarray, color: Tuple[int, int, int]):
    """Draw pose using standard skeleton (same as Final Working)"""
    skeleton = [
        (10, 9), (9, 8), (8, 11), (8, 14),
        (11, 12), (12, 13), (14, 15), (15, 16),
        (11, 4), (14, 1), (0, 4), (0, 1),
        (4, 5), (5, 6), (1, 2), (2, 3)
    ]
    
    # Draw joints
    for x, y in pose_2d:
        if x > 0 and y > 0 and x < image.shape[1] and y < image.shape[0]:
            cv2.circle(image, (int(x), int(y)), 3, color, -1)
    
    # Draw skeleton
    for joint1, joint2 in skeleton:
        if (joint1 < len(pose_2d) and joint2 < len(pose_2d) and 
            pose_2d[joint1][0] > 0 and pose_2d[joint1][1] > 0 and
            pose_2d[joint2][0] > 0 and pose_2d[joint2][1] > 0 and
            pose_2d[joint1][0] < image.shape[1] and pose_2d[joint1][1] < image.shape[0] and
            pose_2d[joint2][0] < image.shape[1] and pose_2d[joint2][1] < image.shape[0]):
            
            pt1 = (int(pose_2d[joint1][0]), int(pose_2d[joint1][1]))
            pt2 = (int(pose_2d[joint2][0]), int(pose_2d[joint2][1]))
            cv2.line(image, pt1, pt2, color, 2)

def draw_production_stats(image: np.ndarray, stats: Dict[str, Any]):
    """Draw production statistics on image"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    
    # Background with transparency
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (450, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
    
    # Color by FPS performance
    fps_ratio = stats['avg_fps'] / stats['target_fps'] if stats['target_fps'] > 0 else 0
    if fps_ratio >= 0.9:
        color = (0, 255, 0)    # Green - excellent
    elif fps_ratio >= 0.7:
        color = (0, 255, 255)  # Yellow - good
    else:
        color = (0, 0, 255)    # Red - poor
    
    # Show skipped status
    status = "SKIPPED" if stats.get('skipped', False) else "PROCESSING"
    threading_status = "MT" if stats.get('threading_enabled', False) else "ST"
    
    texts = [
        f"PRODUCTION V4 - {stats['preset'].upper()} ({threading_status})",
        f"FPS: {stats['avg_fps']:.1f} / {stats['target_fps']:.1f} (instant: {stats['instant_fps']:.1f})",
        f"Backend: {stats['active_backend']} | Poses: {stats['poses_detected']} | Status: {status}",
        f"Processing: {stats['processing_time_ms']:.1f}ms | Frame: {stats['frame_count']}",
        f"Detection: {stats.get('detection_time_ms', 0):.1f}ms | Pose: {stats.get('pose_time_ms', 0):.1f}ms"
    ]
    
    for i, text in enumerate(texts):
        y_pos = 30 + i * 20
        cv2.putText(image, text, (15, y_pos), font, font_scale, color, 1)

def main():
    """Main function for production optimized pipeline"""
    parser = argparse.ArgumentParser(description='ConvNeXt V4 Production - Híbrido estable + optimizado')
    
    parser.add_argument('--model_path', type=str,
                       default='ConvNeXtPose_S.tar',
                       help='Path to pose model')
    parser.add_argument('--preset', type=str, default='ultra_fast',
                       choices=['ultra_fast', 'speed_balanced', 'quality_focused'],
                       help='Production preset')
    parser.add_argument('--backend', type=str, default='pytorch',
                       choices=['pytorch', 'onnx', 'tflite'],
                       help='Inference backend')
    parser.add_argument('--input', type=str, default='0',
                       help='Input source (0 for camera, video file)')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting ConvNeXt V4 PRODUCTION (Stable + Optimized)...")
    logger.info("   Preset: %s", args.preset)
    logger.info("   Backend: %s", args.backend)
    logger.info("   🎯 Combining Final Working stability + Ultra optimizations")
    
    # Initialize processor
    try:
        processor = ProductionV4Processor(args.model_path, args.preset, args.backend)
    except Exception as e:
        logger.error("❌ Processor initialization failed: %s", e)
        return False
    
    # Setup input
    if args.input == '0':
        cap = cv2.VideoCapture(0)
    elif args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)
    
    if not cap.isOpened():
        logger.error("❌ Cannot open input source: %s", args.input)
        return False
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    logger.info("✅ Production V4 started. Press 'q' to quit")
    
    # Main loop
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠️ Cannot read frame")
                break
            
            # Process frame
            poses, stats = processor.process_frame(frame)
            
            # Draw results
            for i, pose_2d in enumerate(poses):
                color = colors[i % len(colors)]
                draw_pose(frame, pose_2d, color)
            
            # Draw stats
            draw_production_stats(frame, stats)
            
            # Show frame
            cv2.imshow('ConvNeXt V4 PRODUCTION - Stable + Optimized', frame)
            
            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        logger.info("⏹️ Interrupted by user")
    
    finally:
        processor.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("🎉 Production V4 shutdown complete")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
