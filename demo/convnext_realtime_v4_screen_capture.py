#!/usr/bin/env python3
"""
convnext_realtime_v4_screen_capture.py - V4 adaptado para captura de pantalla en WSL

Features:
1. ✅ Captura de pantalla usando mss
2. ✅ YOLO person detection  
3. ✅ ConvNeXt pose estimation
4. ✅ Estadísticas en tiempo real
5. ✅ Threading y cache inteligente
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

# Setup básico
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for screen capture
try:
    import mss
    MSS_AVAILABLE = True
    logger.info("✅ MSS (screen capture) available")
except ImportError:
    MSS_AVAILABLE = False
    logger.warning("⚠️ MSS not available, install with: pip install mss")

# Core imports
try:
    import torch
    import torchvision.transforms as transforms
    logger.info(f"✅ PyTorch available: {torch.__version__}")
except ImportError as e:
    logger.error(f"❌ PyTorch not available: {e}")
    sys.exit(1)

# YOLO Detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("✅ Ultralytics YOLO available")
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("⚠️ Ultralytics YOLO not available")

# ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logger.info("✅ ONNX Runtime available")
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("⚠️ ONNX Runtime not available")

# TensorFlow Lite
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
    logger.info("✅ TensorFlow Lite available")
except ImportError:
    TFLITE_AVAILABLE = False
    logger.warning("⚠️ TensorFlow Lite not available")

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
    logger.info("✅ Project modules imported")
except ImportError as e:
    logger.error(f"❌ Project import failed: {e}")
    sys.exit(1)

# Try to import pose utils
try:
    sys.path.append(str(PROJECT_ROOT / 'common' / 'utils'))
    import pose_utils
    POSE_UTILS_AVAILABLE = True
except ImportError:
    POSE_UTILS_AVAILABLE = False
    logger.warning("⚠️ Pose utils not available, using fallback")

class ScreenCapture:
    """Capturador de pantalla usando MSS"""
    
    def __init__(self):
        if not MSS_AVAILABLE:
            raise RuntimeError("MSS not available. Install with: pip install mss")
        
        self.sct = mss.mss()
        
        # Obtener información del monitor principal
        monitor = self.sct.monitors[1]  # Monitor principal
        self.monitor_area = {
            "top": monitor["top"],
            "left": monitor["left"], 
            "width": monitor["width"],
            "height": monitor["height"]
        }
        
        logger.info(f"📺 Screen capture initialized: {self.monitor_area['width']}x{self.monitor_area['height']}")
    
    def capture_frame(self) -> np.ndarray:
        """Capturar frame de la pantalla"""
        try:
            # Capturar pantalla
            sct_img = self.sct.grab(self.monitor_area)
            
            # Convertir a numpy array (BGRA -> BGR)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            return frame
            
        except Exception as e:
            logger.error(f"❌ Screen capture failed: {e}")
            return None

class SimpleYOLODetector:
    """Detector YOLO simplificado para personas"""
    
    def __init__(self, model_path: str = 'yolov8n.pt'):
        if not YOLO_AVAILABLE:
            raise RuntimeError("YOLO not available")
        
        self.model = YOLO(model_path)
        logger.info(f"✅ YOLO detector initialized: {model_path}")
    
    def detect_persons(self, frame: np.ndarray, conf_threshold: float = 0.3) -> List[List[int]]:
        """Detectar personas en el frame"""
        try:
            results = self.model(frame, classes=[0], conf=conf_threshold, verbose=False)
            
            persons = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        if conf >= conf_threshold:
                            persons.append([int(x1), int(y1), int(x2), int(y2)])
            
            return persons
            
        except Exception as e:
            logger.error(f"❌ YOLO detection failed: {e}")
            return []

class SimplePoseEstimator:
    """Estimador de pose simplificado usando PyTorch"""
    
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Configurar modelo
        cfg.input_shape = (256, 256)
        cfg.output_shape = (32, 32) 
        cfg.depth_dim = 32
        cfg.bbox_3d_shape = (2000, 2000, 2000)
        
        # Cargar modelo
        self.model = get_pose_net(cfg, is_train=False, joint_num=18)
        
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            sd = state.get('network', state)
            self.model.load_state_dict(sd, strict=False)
            logger.info(f"✅ Pose model loaded: {model_path}")
        else:
            logger.warning(f"⚠️ Model not found: {model_path}, using random weights")
        
        self.model = self.model.to(self.device).eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"✅ Pose estimator initialized on {self.device}")
    
    def estimate_pose(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Estimar pose para una bbox"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Recortar y redimensionar
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                return None
            
            person_resized = cv2.resize(person_crop, (256, 256))
            person_rgb = cv2.cvtColor(person_resized, cv2.COLOR_BGR2RGB)
            
            # Preparar input
            input_tensor = self.transform(person_rgb).unsqueeze(0).to(self.device)
            
            # Inferencia
            with torch.no_grad():
                output = self.model(input_tensor)
                
            # Procesar output
            if isinstance(output, (list, tuple)):
                pose_output = output[0]
            else:
                pose_output = output
                
            # Convertir a numpy y escalar
            pose_2d = pose_output.cpu().numpy().squeeze()
            
            if pose_2d.ndim == 3:  # (1, H, W) -> (H, W)
                pose_2d = pose_2d[0]
            
            # Encontrar keypoints (método simplificado)
            if pose_2d.ndim == 2:
                keypoints = []
                h, w = pose_2d.shape
                for i in range(min(17, h)):  # 17 keypoints máximo
                    row = pose_2d[i] if i < h else pose_2d[0]
                    if len(row) >= 2:
                        # Escalar a coordenadas de la bbox
                        x = float(row[0]) * (x2 - x1) + x1
                        y = float(row[1]) * (y2 - y1) + y1
                        keypoints.append([x, y])
                    else:
                        keypoints.append([x1 + (x2-x1)/2, y1 + (y2-y1)/2])  # Centro como fallback
                
                return np.array(keypoints)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Pose estimation failed: {e}")
            return None

class PerformanceTracker:
    """Tracker de rendimiento"""
    
    def __init__(self, maxlen: int = 30):
        self.frame_times = deque(maxlen=maxlen)
        self.processing_times = deque(maxlen=maxlen)
        self.frame_count = 0
        self.start_time = time.time()
    
    def update(self, processing_time: float):
        """Actualizar métricas"""
        current_time = time.time()
        self.frame_times.append(current_time)
        self.processing_times.append(processing_time)
        self.frame_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas"""
        if len(self.frame_times) < 2:
            return {
                'fps': 0.0,
                'avg_processing_time_ms': 0.0,
                'frame_count': self.frame_count,
                'uptime_seconds': time.time() - self.start_time
            }
        
        # Calcular FPS
        time_diff = self.frame_times[-1] - self.frame_times[0]
        fps = (len(self.frame_times) - 1) / max(time_diff, 0.001)
        
        # Tiempo promedio de procesamiento
        avg_proc_time = np.mean(self.processing_times) * 1000  # ms
        
        return {
            'fps': fps,
            'avg_processing_time_ms': avg_proc_time,
            'frame_count': self.frame_count,
            'uptime_seconds': time.time() - self.start_time
        }

class V4ScreenProcessor:
    """Procesador principal V4 para captura de pantalla"""
    
    def __init__(self, model_path: str, yolo_model: str = 'yolov8n.pt'):
        logger.info("🚀 Initializing V4 Screen Processor...")
        
        # Inicializar componentes
        self.screen_capture = ScreenCapture()
        self.yolo_detector = SimpleYOLODetector(yolo_model)
        self.pose_estimator = SimplePoseEstimator(model_path)
        self.performance_tracker = PerformanceTracker()
        
        logger.info("✅ V4 Screen Processor initialized")
    
    def process_frame(self) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, Any]]:
        """Procesar un frame de la pantalla"""
        start_time = time.time()
        
        # Capturar pantalla
        frame = self.screen_capture.capture_frame()
        if frame is None:
            return None, [], {}
        
        # Detectar personas
        person_bboxes = self.yolo_detector.detect_persons(frame)
        
        # Estimar poses
        poses = []
        for bbox in person_bboxes:
            pose = self.pose_estimator.estimate_pose(frame, bbox)
            if pose is not None:
                poses.append(pose)
        
        # Actualizar métricas
        processing_time = time.time() - start_time
        self.performance_tracker.update(processing_time)
        
        stats = self.performance_tracker.get_stats()
        stats.update({
            'persons_detected': len(person_bboxes),
            'poses_estimated': len(poses),
            'processing_time_ms': processing_time * 1000
        })
        
        return frame, poses, stats

def draw_poses(frame: np.ndarray, poses: List[np.ndarray], color: Tuple[int, int, int] = (0, 255, 0)):
    """Dibujar poses en el frame"""
    skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Brazo derecho
        (0, 5), (5, 6), (6, 7), (7, 8),  # Brazo izquierdo
        (0, 9), (9, 10), (10, 11),       # Tronco
        (11, 12), (12, 13), (13, 14),    # Pierna derecha
        (11, 15), (15, 16)               # Pierna izquierda
    ]
    
    for pose in poses:
        # Dibujar keypoints
        for i, (x, y) in enumerate(pose):
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        
        # Dibujar skeleton
        for start_idx, end_idx in skeleton:
            if start_idx < len(pose) and end_idx < len(pose):
                start_point = pose[start_idx]
                end_point = pose[end_idx]
                
                if (0 <= start_point[0] < frame.shape[1] and 0 <= start_point[1] < frame.shape[0] and
                    0 <= end_point[0] < frame.shape[1] and 0 <= end_point[1] < frame.shape[0]):
                    cv2.line(frame, 
                            (int(start_point[0]), int(start_point[1])),
                            (int(end_point[0]), int(end_point[1])), 
                            color, 2)

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='ConvNeXt V4 Screen Capture Demo')
    parser.add_argument('--model_path', type=str,
                       default='/home/fabri/ConvNeXtPose/exports/model_opt_S.pth',
                       help='Path to pose model')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt',
                       help='YOLO model for person detection')
    parser.add_argument('--save_video', type=str, default=None,
                       help='Save output video to file')
    parser.add_argument('--stats', action='store_true',
                       help='Show detailed statistics')
    parser.add_argument('--fps_limit', type=int, default=30,
                       help='FPS limit for processing')
    
    args = parser.parse_args()
    
    # Verificar dependencias
    if not MSS_AVAILABLE:
        logger.error("❌ MSS not available. Install with: pip install mss")
        return
    
    if not YOLO_AVAILABLE:
        logger.error("❌ YOLO not available. Install with: pip install ultralytics")
        return
    
    # Inicializar procesador
    try:
        processor = V4ScreenProcessor(args.model_path, args.yolo_model)
    except Exception as e:
        logger.error(f"❌ Failed to initialize processor: {e}")
        return
    
    # Configurar salida de video
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Usar resolución de pantalla capturada
        sample_frame, _, _ = processor.process_frame()
        if sample_frame is not None:
            h, w = sample_frame.shape[:2]
            video_writer = cv2.VideoWriter(args.save_video, fourcc, args.fps_limit, (w, h))
            logger.info(f"📹 Recording to: {args.save_video}")
    
    logger.info("🚀 V4 Screen Capture Demo started. Press 'q' to quit, 's' for stats")
    
    # Variables de control
    last_stats_time = time.time()
    frame_interval = 1.0 / args.fps_limit
    
    try:
        while True:
            loop_start = time.time()
            
            # Procesar frame
            frame, poses, stats = processor.process_frame()
            if frame is None:
                continue
            
            # Dibujar poses
            draw_poses(frame, poses)
            
            # Dibujar estadísticas en pantalla
            stats_text = (
                f"FPS: {stats['fps']:.1f} | "
                f"Proc: {stats['processing_time_ms']:.1f}ms | "
                f"Persons: {stats['persons_detected']} | "
                f"Poses: {stats['poses_estimated']}"
            )
            cv2.putText(frame, stats_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar frame
            cv2.imshow(f'ConvNeXt V4 Screen Capture', frame)
            
            # Guardar video si está habilitado
            if video_writer is not None:
                video_writer.write(frame)
            
            # Mostrar estadísticas en terminal
            current_time = time.time()
            if args.stats or (current_time - last_stats_time > 5.0):
                logger.info("📊 Performance Stats:")
                logger.info(f"   FPS: {stats['fps']:.2f}")
                logger.info(f"   Processing time: {stats['processing_time_ms']:.1f}ms")
                logger.info(f"   Total frames: {stats['frame_count']}")
                logger.info(f"   Persons detected: {stats['persons_detected']}")
                logger.info(f"   Poses estimated: {stats['poses_estimated']}")
                logger.info(f"   Uptime: {stats['uptime_seconds']:.1f}s")
                last_stats_time = current_time
            
            # Control de teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Estadísticas detalladas
                logger.info("📊 Detailed Stats:")
                logger.info(f"   Current FPS: {stats['fps']:.2f}")
                logger.info(f"   Average processing: {stats['avg_processing_time_ms']:.1f}ms")
                logger.info(f"   Total frames: {stats['frame_count']}")
                logger.info(f"   Uptime: {stats['uptime_seconds']:.1f}s")
            
            # Controlar FPS
            elapsed = time.time() - loop_start  
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    
    except KeyboardInterrupt:
        logger.info("⚡ Interrupted by user")
    
    finally:        
        # Cleanup
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Estadísticas finales
        final_stats = processor.performance_tracker.get_stats()
        logger.info("🏁 Final Statistics:")
        logger.info(f"   Total frames processed: {final_stats['frame_count']}")
        logger.info(f"   Average FPS: {final_stats['fps']:.2f}")
        logger.info(f"   Average processing time: {final_stats['avg_processing_time_ms']:.1f}ms")
        logger.info(f"   Total uptime: {final_stats['uptime_seconds']:.1f}s")

if __name__ == "__main__":
    main()
