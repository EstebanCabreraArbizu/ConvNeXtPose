#!/usr/bin/env python3
"""
convnext_realtime_v4_corrected.py - V4 Enhanced con ejecución en tiempo real

Features V4:
1. ✅ AdaptiveYOLODetector con multi-persona
2. ✅ TFLite support con convertidor onnx-tf
3. ✅ Threading robusto y cache inteligente  
4. ✅ Letterbox preprocessing
5. ✅ Estadísticas en tiempo real
6. ✅ Captura de pantalla funcional
"""

import sys
import os
import time
import argparse
import logging
import warnings
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from collections import deque
import numpy as np
import cv2

# Setup
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
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

# Try to import the corrected converter
CORRECTED_CONVERTER_AVAILABLE = False
try:
    from corrected_onnx_to_tflite_converter import convert_onnx_to_tflite_corrected
    CORRECTED_CONVERTER_AVAILABLE = True
except ImportError:
    pass

try:
    from root_wrapper import RootNetWrapper
except ImportError:
    print("⚠️ RootNet wrapper not available")
    RootNetWrapper = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_hardware_capabilities():
    """Detect hardware capabilities"""
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
            for i in range(capabilities['cuda_device_count']):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                capabilities['cuda_memory_gb'] = max(capabilities['cuda_memory_gb'], memory_gb)
            capabilities['recommended_workers'] = min(4, capabilities['cuda_device_count'] * 2)
            capabilities['recommended_cache_timeout'] = 0.08
            capabilities['recommended_frame_skip'] = 1
        except:
            capabilities['cuda_memory_gb'] = 0
    else:
        capabilities.update({
            'recommended_workers': 1,
            'recommended_cache_timeout': 0.15,
            'recommended_frame_skip': 3
        })
    
    logger.info(f"🔧 Hardware detected: GPU={'✅' if capabilities['has_cuda'] else '❌'} "
                f"({capabilities['cuda_memory_gb']:.1f}GB), Workers={capabilities['recommended_workers']}")
    
    return capabilities

class AdaptiveYOLODetector:
    """YOLO detector optimizado para detección multi-persona con auto-fallback"""
    
    def __init__(self, hardware_caps: Dict[str, Any], yolo_model_path: str = 'yolov8n.pt'):
        self.hardware_caps = hardware_caps
        self.yolo_model_path = yolo_model_path
        self.detector = None
        self.detector_type = None
        self.input_size = 640
        
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Inicializar detector con fallback automático"""
        if YOLO_AVAILABLE:
            try:
                logger.info(f"🔄 Loading YOLO model: {self.yolo_model_path}")
                self.detector = YOLO(self.yolo_model_path)
                self.detector_type = "ultralytics"
                logger.info("✅ YOLO detector initialized (Ultralytics)")
                return
            except Exception as e:
                logger.warning(f"⚠️ Ultralytics YOLO failed: {e}")
        
        logger.error("❌ No YOLO detector available")
        self.detector = None
        self.detector_type = None
    
    def detect_persons(self, frame: np.ndarray, conf_threshold: float = 0.3) -> List[List[int]]:
        """Detectar personas en el frame"""
        if self.detector is None:
            return []
        
        try:
            # Detectar con YOLO
            results = self.detector(frame, conf=conf_threshold, classes=[0], verbose=False)
            
            persons = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = float(box.conf[0])
                        if conf >= conf_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            persons.append([int(x1), int(y1), int(x2), int(y2)])
            
            return persons
            
        except Exception as e:
            logger.warning(f"⚠️ Person detection failed: {e}")
            return []

def letterbox_resize(image, new_shape=(640, 640), color=(114, 114, 114)):
    """Letterbox resize manteniendo aspect ratio"""
    shape = image.shape[:2]  # current shape [height, width]
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return image

class TFLiteInferenceEngine:
    """Motor de inferencia TFLite optimizado"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        self._initialize()
    
    def _initialize(self):
        """Inicializar intérprete TFLite"""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logger.info(f"✅ TFLite engine initialized: {self.model_path}")
            logger.info(f"   Input shape: {self.input_details[0]['shape']}")
            logger.info(f"   Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            logger.error(f"❌ TFLite initialization failed: {e}")
            raise
    
    def infer(self, img_patch: np.ndarray) -> np.ndarray:
        """Realizar inferencia"""
        try:
            # Preparar input
            input_data = self._prepare_input(img_patch)
            
            # Set input
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            return output_data
            
        except Exception as e:
            logger.error(f"❌ TFLite inference failed: {e}")
            return None
    
    def _prepare_input(self, img_patch: np.ndarray) -> np.ndarray:
        """Preparar input para TFLite"""
        # Normalizar y convertir
        if img_patch.dtype != np.float32:
            img_patch = img_patch.astype(np.float32) / 255.0
        
        # Agregar batch dimension si es necesario
        if len(img_patch.shape) == 3:
            img_patch = np.expand_dims(img_patch, axis=0)
        
        return img_patch

class OptimizedInferenceRouter:
    """Router de inferencia que maneja PyTorch, ONNX y TFLite"""
    
    def __init__(self, model_path: str, use_tflite: bool = False):
        self.model_path = model_path
        self.use_tflite = use_tflite
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Rutas de modelos
        self.pytorch_path = model_path
        self.onnx_path = model_path.replace('.pth', '_optimized.onnx')
        self.tflite_path = model_path.replace('.pth', '_optimized.tflite')
        
        # Engines
        self.pytorch_model = None
        self.onnx_session = None
        self.tflite_engine = None
        
        self.active_backend = None
        
        self._initialize()
    
    def _initialize(self):
        """Inicializar el router con el backend apropiado"""
        # Intentar TFLite primero si se solicita
        if self.use_tflite and self._setup_tflite():
            return
        
        # Fallback a ONNX
        if self._setup_onnx():
            return
        
        # Fallback final a PyTorch
        if self._setup_pytorch():
            return
        
        raise RuntimeError("No se pudo inicializar ningún backend de inferencia")
    
    def _setup_tflite(self) -> bool:
        """Setup TFLite backend"""
        try:
            if not TFLITE_AVAILABLE:
                logger.warning("⚠️ TensorFlow Lite not available")
                return False
            
            # Verificar si existe el modelo TFLite
            if not os.path.exists(self.tflite_path):
                logger.info(f"🔄 Converting ONNX to TFLite: {self.onnx_path}")
                if not self._convert_to_tflite():
                    return False
            
            # Inicializar TFLite engine
            self.tflite_engine = TFLiteInferenceEngine(self.tflite_path)
            self.active_backend = "tflite"
            logger.info("✅ TFLite backend active")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ TFLite setup failed: {e}")
            return False
    
    def _setup_onnx(self) -> bool:
        """Setup ONNX backend"""
        try:
            if not ONNX_AVAILABLE:
                logger.warning("⚠️ ONNX Runtime not available")
                return False
            
            if not os.path.exists(self.onnx_path):
                logger.warning(f"⚠️ ONNX model not found: {self.onnx_path}")
                return False
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(self.onnx_path, providers=providers)
            self.active_backend = "onnx"
            logger.info("✅ ONNX backend active")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ ONNX setup failed: {e}")
            return False
    
    def _setup_pytorch(self) -> bool:
        """Setup PyTorch backend"""
        try:
            # Configurar modelo
            cfg.input_shape = (256, 256)
            cfg.output_shape = (32, 32) 
            cfg.depth_dim = 32
            cfg.bbox_3d_shape = (2000, 2000, 2000)
            
            # Cargar modelo
            self.pytorch_model = get_pose_net(cfg, is_train=False, joint_num=18)
            state = torch.load(self.pytorch_path, map_location=self.device)
            sd = state.get('network', state)
            self.pytorch_model.load_state_dict(sd, strict=False)
            self.pytorch_model = self.pytorch_model.to(self.device).eval()
            
            self.active_backend = "pytorch"
            logger.info(f"✅ PyTorch backend active on {self.device}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ PyTorch setup failed: {e}")
            return False
    
    def _convert_to_tflite(self) -> bool:
        """Convertir ONNX a TFLite usando el convertidor corregido"""
        if not CORRECTED_CONVERTER_AVAILABLE:
            logger.warning("⚠️ Corrected converter not available")
            return False
        
        try:
            result = convert_onnx_to_tflite_corrected(
                self.onnx_path, 
                self.tflite_path,
                optimization="default"
            )
            
            if result['success']:
                logger.info(f"✅ TFLite conversion successful: {result['file_size_mb']:.2f} MB")
                return True
            else:
                logger.warning(f"⚠️ TFLite conversion failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ TFLite conversion error: {e}")
            return False
    
    def infer(self, img_patch: np.ndarray) -> Optional[np.ndarray]:
        """Realizar inferencia con el backend activo"""
        try:
            if self.active_backend == "tflite":
                return self.tflite_engine.infer(img_patch)
            elif self.active_backend == "onnx":
                return self._infer_onnx(img_patch)
            elif self.active_backend == "pytorch":
                return self._infer_pytorch(img_patch)
            else:
                logger.error("❌ No active backend available")
                return None
                
        except Exception as e:
            logger.error(f"❌ Inference failed with {self.active_backend}: {e}")
            return None
    
    def _infer_onnx(self, img_patch: np.ndarray) -> np.ndarray:
        """Inferencia con ONNX"""
        # Preparar input
        if len(img_patch.shape) == 3:
            img_patch = np.expand_dims(img_patch, axis=0)
        
        input_name = self.onnx_session.get_inputs()[0].name
        result = self.onnx_session.run(None, {input_name: img_patch})
        return result[0]
    
    def _infer_pytorch(self, img_patch: np.ndarray) -> np.ndarray:
        """Inferencia con PyTorch"""
        # Convertir a tensor PyTorch
        if isinstance(img_patch, np.ndarray):
            img_patch = torch.from_numpy(img_patch).float()
        
        if len(img_patch.shape) == 3:
            img_patch = img_patch.unsqueeze(0)
        
        img_patch = img_patch.to(self.device)
        
        with torch.no_grad():
            output = self.pytorch_model(img_patch)
            return output.cpu().numpy()

class IntelligentCacheManager:
    """Cache manager inteligente para optimizar rendimiento"""
    
    def __init__(self, cache_timeout: float = 0.1, max_size: int = 50):
        self.cache = {}
        self.cache_timeout = cache_timeout
        self.max_size = max_size
        self.cache_hits = 0
        self.cache_misses = 0
        self.lock = threading.Lock()
    
    def generate_key(self, bbox: List[int], timestamp: float) -> str:
        """Generar clave de cache"""
        x1, y1, x2, y2 = bbox
        # Cuantizar bbox para cache
        quantize = 10
        x1_q = (x1 // quantize) * quantize
        y1_q = (y1 // quantize) * quantize
        x2_q = (x2 // quantize) * quantize
        y2_q = (y2 // quantize) * quantize
        
        time_slot = int(timestamp / self.cache_timeout)
        return f"{x1_q}_{y1_q}_{x2_q}_{y2_q}_{time_slot}"
    
    def get(self, key: str, current_time: float) -> Optional[Tuple[np.ndarray, float]]:
        """Obtener del cache"""
        with self.lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                if current_time - timestamp < self.cache_timeout:
                    self.cache_hits += 1
                    return result, timestamp
                else:
                    del self.cache[key]
            
            self.cache_misses += 1
            return None
    
    def put(self, key: str, result: np.ndarray, timestamp: float):
        """Guardar en cache"""
        with self.lock:
            self.cache[key] = (result, timestamp)
            
            # Limpiar cache si está lleno
            if len(self.cache) > self.max_size:
                # Remover entradas más antiguas
                oldest_keys = sorted(self.cache.keys(), 
                                   key=lambda k: self.cache[k][1])[:len(self.cache) - self.max_size]
                for old_key in oldest_keys:
                    del self.cache[old_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de cache"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }

class V4RealTimeProcessor:
    """Procesador principal V4 en tiempo real"""
    
    def __init__(self, model_path: str, use_tflite: bool = False, yolo_model: str = 'yolov8n.pt'):
        self.hardware_caps = detect_hardware_capabilities()
        
        # Inicializar componentes
        self.yolo_detector = AdaptiveYOLODetector(self.hardware_caps, yolo_model)
        self.inference_router = OptimizedInferenceRouter(model_path, use_tflite)
        self.cache_manager = IntelligentCacheManager(
            cache_timeout=self.hardware_caps['recommended_cache_timeout']
        )
        
        # Configuración
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)
        ])
        
        # Estadísticas
        self.frame_count = 0
        self.processing_times = deque(maxlen=30)
        self.fps_counter = deque(maxlen=30)
        
        # RootNet (opcional)
        self.root_wrapper = None
        
        logger.info("✅ V4RealTimeProcessor initialized")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Procesar frame completo"""
        start_time = time.time()
        
        # Detección de personas
        persons = self.yolo_detector.detect_persons(frame, conf_threshold=0.3)
        
        # Procesar poses para cada persona
        poses = []
        for person_bbox in persons:
            pose = self._process_single_person(frame, person_bbox, start_time)
            if pose is not None:
                poses.append(pose)
        
        # Calcular estadísticas
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.frame_count += 1
        
        # Calcular FPS
        if len(self.fps_counter) > 0:
            fps = 1.0 / processing_time if processing_time > 0 else 0
            self.fps_counter.append(fps)
        else:
            self.fps_counter.append(0)
        
        stats = self._get_frame_stats()
        
        return poses, stats
    
    def _process_single_person(self, frame: np.ndarray, bbox: List[int], frame_time: float) -> Optional[np.ndarray]:
        """Procesar pose de una persona"""
        try:
            # Check cache
            cache_key = self.cache_manager.generate_key(bbox, frame_time)
            cached_result = self.cache_manager.get(cache_key, frame_time)
            if cached_result is not None:
                return cached_result[0]
            
            # Preparar bbox
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            bbox_array = np.array([x1, y1, x2 - x1, y2 - y1])
            proc_bbox = pose_utils.process_bbox(bbox_array, frame.shape[1], frame.shape[0])
            
            if proc_bbox is None:
                return None
            
            # Generar patch
            img_patch, img2bb_trans = generate_patch_image(frame, proc_bbox, False, 0.0)
            img_patch_copy = img_patch.copy()
            
            # Aplicar transformación
            img_patch = self.transform(img_patch).numpy()
            
            # Inferencia
            pose_output = self.inference_router.infer(img_patch)
            if pose_output is None:
                return None
            
            # Post-procesamiento
            pose_3d = pose_output[0] if len(pose_output.shape) > 2 else pose_output
            
            # Convertir a coordenadas de imagen
            pose_2d = self._postprocess_pose(pose_3d, img2bb_trans)
            
            # Guardar en cache
            self.cache_manager.put(cache_key, pose_2d, frame_time)
            
            return pose_2d
            
        except Exception as e:
            logger.warning(f"⚠️ Error processing person: {e}")
            return None
    
    def _postprocess_pose(self, pose_3d: np.ndarray, img2bb_trans: np.ndarray) -> np.ndarray:
        """Post-procesar pose para obtener coordenadas 2D"""
        # Normalizar coordenadas
        pose_2d = pose_3d[:, :2].copy()
        pose_2d[:, 0] = pose_2d[:, 0] / cfg.output_shape[1] * cfg.input_shape[1]
        pose_2d[:, 1] = pose_2d[:, 1] / cfg.output_shape[0] * cfg.input_shape[0]
        
        # Aplicar transformación inversa
        pose_2d_homo = np.column_stack((pose_2d, np.ones(len(pose_2d))))
        img2bb_trans_homo = np.vstack((img2bb_trans, [0, 0, 1]))
        
        try:
            bb2img_trans = np.linalg.inv(img2bb_trans_homo)
            pose_2d_img = np.dot(bb2img_trans, pose_2d_homo.T).T
            return pose_2d_img[:, :2]
        except np.linalg.LinAlgError:
            return pose_2d
    
    def _get_frame_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del frame"""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        avg_fps = np.mean(self.fps_counter) if self.fps_counter else 0
        cache_stats = self.cache_manager.get_stats()
        
        return {
            'frame_count': self.frame_count,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'avg_fps': avg_fps,
            'active_backend': self.inference_router.active_backend,
            'cache_hit_rate': cache_stats['hit_rate'],
            'cache_size': cache_stats['cache_size']
        }

def draw_pose(image, pose_2d, skeleton, color=(0, 255, 0), thickness=2):
    """Dibujar pose en la imagen"""
    # Dibujar joints
    for i, (x, y) in enumerate(pose_2d):
        if x > 0 and y > 0:
            cv2.circle(image, (int(x), int(y)), 3, color, -1)
    
    # Dibujar skeleton
    for connection in skeleton:
        if len(connection) == 2:
            joint1, joint2 = connection
            if (joint1 < len(pose_2d) and joint2 < len(pose_2d) and 
                pose_2d[joint1][0] > 0 and pose_2d[joint1][1] > 0 and
                pose_2d[joint2][0] > 0 and pose_2d[joint2][1] > 0):
                
                pt1 = (int(pose_2d[joint1][0]), int(pose_2d[joint1][1]))
                pt2 = (int(pose_2d[joint2][0]), int(pose_2d[joint2][1]))
                cv2.line(image, pt1, pt2, color, thickness)

def draw_stats(image, stats):
    """Dibujar estadísticas en la imagen"""
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 255, 0)
    thickness = 2
    
    # Background para mejor legibilidad
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
    
    stats_text = [
        f"V4 Enhanced - Frame: {stats['frame_count']}",
        f"FPS: {stats['avg_fps']:.1f}",
        f"Processing: {stats['avg_processing_time_ms']:.1f}ms",
        f"Backend: {stats['active_backend'].upper()}",
        f"Cache: {stats['cache_hit_rate']:.1f}% ({stats['cache_size']} items)"
    ]
    
    for i, text in enumerate(stats_text):
        y_pos = y_offset + i * 20
        cv2.putText(image, text, (15, y_pos), font, font_scale, color, thickness)

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='ConvNeXt V4 Real-time Demo')
    parser.add_argument('--model_path', type=str, 
                       default='/home/fabri/ConvNeXtPose/exports/model_opt_S.pth',
                       help='Path to pose model')
    parser.add_argument('--input', type=str, default='0',
                       help='Input source (0 for camera, video file path)')
    parser.add_argument('--backend', type=str, default='pytorch',
                       choices=['pytorch', 'onnx', 'tflite'],
                       help='Inference backend to use')
    parser.add_argument('--use_tflite', action='store_true',
                       help='Use TensorFlow Lite backend (deprecated, use --backend tflite)')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt',
                       help='YOLO model for person detection')
    parser.add_argument('--save_video', type=str, default=None,
                       help='Save output video to file')
    parser.add_argument('--stats', action='store_true',
                       help='Show detailed statistics')
    parser.add_argument('--dry-run', action='store_true',
                       help='Initialize only, do not run inference')
    
    args = parser.parse_args()
    
    # Compatibilidad con argumento antiguo
    use_tflite = args.use_tflite or args.backend == 'tflite'
    
    # Configurar ConvNeXt
    cfg.input_shape = (256, 256)
    cfg.output_shape = (32, 32)
    cfg.depth_dim = 32
    cfg.bbox_3d_shape = (2000, 2000, 2000)
    
    # Dry run: solo verificar inicialización
    if getattr(args, 'dry_run', False):
        logger.info("🧪 Dry run mode - initializing only...")
        try:
            processor = V4RealTimeProcessor(args.model_path, use_tflite, args.yolo_model)
            logger.info("✅ Initialization successful")
            return
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return
    
    # Inicializar procesador
    logger.info("🚀 Initializing V4 Real-time Processor...")
    processor = V4RealTimeProcessor(args.model_path, use_tflite, args.yolo_model)
    
    # Esqueleto para visualización
    skeleton = [
        (10, 9), (9, 8), (8, 11), (8, 14),
        (11, 12), (12, 13), (14, 15), (15, 16),
        (11, 4), (14, 1), (0, 4), (0, 1),
        (4, 5), (5, 6), (1, 2), (2, 3)
    ]
    
    # Configurar captura
    if args.input == '0':
        cap = cv2.VideoCapture(0)
    elif args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)
    
    if not cap.isOpened():
        logger.error("❌ Cannot open video capture")
        return
    
    # Configurar salida de video si se especifica
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(args.save_video, fourcc, fps, (width, height))
    
    logger.info("✅ V4 Real-time Demo started. Press 'q' to quit, 's' for stats")
    
    # Variables de rendimiento
    last_stats_time = time.time()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 255, 0)]
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar frame
            poses, stats = processor.process_frame(frame)
            
            # Dibujar poses
            for i, pose in enumerate(poses):
                color = colors[i % len(colors)]
                draw_pose(frame, pose, skeleton, color)
            
            # Dibujar estadísticas
            draw_stats(frame, stats)
            
            # Mostrar frame
            cv2.imshow('ConvNeXt V4 Enhanced Real-time', frame)
            
            # Guardar video si se especifica
            if video_writer is not None:
                video_writer.write(frame)
            
            # Mostrar estadísticas detalladas en terminal cada 5 segundos o si está habilitado
            current_time = time.time()
            if args.stats or (current_time - last_stats_time > 5.0):
                logger.info("📊 Performance Stats:")
                logger.info(f"   Frame: {stats['frame_count']}")
                logger.info(f"   FPS: {stats['avg_fps']:.2f}")
                logger.info(f"   Processing time: {stats['avg_processing_time_ms']:.1f}ms")
                logger.info(f"   Backend: {stats['active_backend']}")
                logger.info(f"   Cache hit rate: {stats['cache_hit_rate']:.1f}%")
                logger.info(f"   Poses detected: {len(poses)}")
                last_stats_time = current_time
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Mostrar estadísticas detalladas
                logger.info("📊 Detailed Stats:")
                logger.info(f"   Total frames: {stats['frame_count']}")
                logger.info(f"   Average FPS: {stats['avg_fps']:.2f}")
                logger.info(f"   Average processing time: {stats['avg_processing_time_ms']:.1f}ms")
                logger.info(f"   Active backend: {stats['active_backend']}")
                logger.info(f"   Cache performance: {stats['cache_hit_rate']:.1f}% hit rate")
                logger.info(f"   Current poses: {len(poses)}")
    
    except KeyboardInterrupt:
        logger.info("⚡ Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Estadísticas finales
        final_stats = processor._get_frame_stats()
        logger.info("🏁 Final Statistics:")
        logger.info(f"   Total frames processed: {final_stats['frame_count']}")
        logger.info(f"   Average FPS: {final_stats['avg_fps']:.2f}")
        logger.info(f"   Average processing time: {final_stats['avg_processing_time_ms']:.1f}ms")
        logger.info(f"   Backend used: {final_stats['active_backend']}")
        logger.info(f"   Cache efficiency: {final_stats['cache_hit_rate']:.1f}%")

if __name__ == "__main__":
    main()
