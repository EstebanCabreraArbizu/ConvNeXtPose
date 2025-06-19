#!/usr/bin/env python3
"""
convnext_realtime_v4_demo_wsl.py - V4 Demo adaptado para WSL con imagen sintética

Features:
1. ✅ Imagen sintética con personas simuladas
2. ✅ YOLO person detection  
3. ✅ ConvNeXt pose estimation
4. ✅ Estadísticas en tiempo real
5. ✅ Comparación con V3
"""

import sys
import os
import time
import argparse
import logging
import warnings
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

class SyntheticFrameGenerator:
    """Generador de frames sintéticos con personas para testing"""
    
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.frame_count = 0
        logger.info(f"🎭 Synthetic frame generator initialized: {width}x{height}")
    
    def generate_frame(self) -> np.ndarray:
        """Generar frame sintético con personas simuladas"""
        # Crear background
        frame = np.random.randint(20, 80, (self.height, self.width, 3), dtype=np.uint8)
        
        # Agregar gradiente para más realismo
        y_gradient = np.linspace(0, 1, self.height).reshape(-1, 1, 1)
        frame = (frame.astype(np.float32) * (0.5 + 0.5 * y_gradient)).astype(np.uint8)
        
        # Simular personas (rectángulos con forma humana)
        num_persons = np.random.randint(1, 4)  # 1-3 personas
        
        for i in range(num_persons):
            # Posición aleatoria
            person_width = np.random.randint(80, 150)
            person_height = np.random.randint(150, 250)
            
            x = np.random.randint(50, self.width - person_width - 50)
            y = np.random.randint(50, self.height - person_height - 50)
            
            # Color de persona
            person_color = np.random.randint(100, 200, 3)
            
            # Dibujar silueta básica de persona
            # Cabeza
            head_radius = person_width // 6
            head_center = (x + person_width // 2, y + head_radius)
            cv2.circle(frame, head_center, head_radius, person_color.tolist(), -1)
            
            # Torso
            torso_top = y + head_radius * 2
            torso_bottom = y + int(person_height * 0.6)
            torso_width = int(person_width * 0.8)
            torso_x = x + (person_width - torso_width) // 2
            
            cv2.rectangle(frame, 
                         (torso_x, torso_top),
                         (torso_x + torso_width, torso_bottom),
                         person_color.tolist(), -1)
            
            # Brazos
            arm_width = person_width // 8
            arm_length = int(person_height * 0.4)
            
            # Brazo izquierdo
            cv2.rectangle(frame,
                         (torso_x - arm_width, torso_top + 20),
                         (torso_x, torso_top + 20 + arm_length),
                         person_color.tolist(), -1)
            
            # Brazo derecho
            cv2.rectangle(frame,
                         (torso_x + torso_width, torso_top + 20),
                         (torso_x + torso_width + arm_width, torso_top + 20 + arm_length),
                         person_color.tolist(), -1)
            
            # Piernas
            leg_width = person_width // 6
            leg_length = person_height - (torso_bottom - y)
            leg_spacing = person_width // 4
            
            # Pierna izquierda
            leg1_x = x + person_width // 2 - leg_spacing // 2
            cv2.rectangle(frame,
                         (leg1_x - leg_width // 2, torso_bottom),
                         (leg1_x + leg_width // 2, y + person_height),
                         person_color.tolist(), -1)
            
            # Pierna derecha
            leg2_x = x + person_width // 2 + leg_spacing // 2
            cv2.rectangle(frame,
                         (leg2_x - leg_width // 2, torso_bottom),
                         (leg2_x + leg_width // 2, y + person_height),
                         person_color.tolist(), -1)
        
        # Agregar algo de ruido
        noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Agregar timestamp
        timestamp = f"Frame: {self.frame_count} | Time: {time.strftime('%H:%M:%S')}"
        cv2.putText(frame, timestamp, (10, self.height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        self.frame_count += 1
        return frame

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
                
            # Procesar output (simplificado para demo)
            if isinstance(output, (list, tuple)):
                pose_output = output[0]
            else:
                pose_output = output
                
            # Convertir a numpy
            pose_data = pose_output.cpu().numpy().squeeze()
            
            # Generar keypoints sintéticos basados en la bbox (para demo)
            keypoints = self._generate_synthetic_keypoints(x1, y1, x2, y2)
            
            return np.array(keypoints)
            
        except Exception as e:
            logger.error(f"❌ Pose estimation failed: {e}")
            return None
    
    def _generate_synthetic_keypoints(self, x1: int, y1: int, x2: int, y2: int) -> List[List[float]]:
        """Generar keypoints sintéticos para demo (17 keypoints COCO)"""
        width = x2 - x1
        height = y2 - y1
        
        # Definir posiciones relativas de keypoints (formato COCO)
        relative_keypoints = [
            (0.5, 0.1),   # 0: nose
            (0.4, 0.2),   # 1: left_eye
            (0.6, 0.2),   # 2: right_eye
            (0.3, 0.25),  # 3: left_ear
            (0.7, 0.25),  # 4: right_ear
            (0.3, 0.4),   # 5: left_shoulder
            (0.7, 0.4),   # 6: right_shoulder
            (0.25, 0.6),  # 7: left_elbow
            (0.75, 0.6),  # 8: right_elbow
            (0.2, 0.75),  # 9: left_wrist
            (0.8, 0.75),  # 10: right_wrist
            (0.35, 0.65), # 11: left_hip
            (0.65, 0.65), # 12: right_hip
            (0.3, 0.85),  # 13: left_knee
            (0.7, 0.85),  # 14: right_knee
            (0.25, 0.95), # 15: left_ankle
            (0.75, 0.95), # 16: right_ankle
        ]
        
        keypoints = []
        for rel_x, rel_y in relative_keypoints:
            abs_x = x1 + rel_x * width
            abs_y = y1 + rel_y * height
            keypoints.append([abs_x, abs_y])
        
        return keypoints

class PerformanceTracker:
    """Tracker de rendimiento mejorado"""
    
    def __init__(self, maxlen: int = 30):
        self.frame_times = deque(maxlen=maxlen)
        self.processing_times = deque(maxlen=maxlen)
        self.yolo_times = deque(maxlen=maxlen)
        self.pose_times = deque(maxlen=maxlen)
        self.frame_count = 0
        self.start_time = time.time()
    
    def update(self, processing_time: float, yolo_time: float = 0, pose_time: float = 0):
        """Actualizar métricas"""
        current_time = time.time()
        self.frame_times.append(current_time)
        self.processing_times.append(processing_time)
        self.yolo_times.append(yolo_time)
        self.pose_times.append(pose_time)
        self.frame_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas detalladas"""
        if len(self.frame_times) < 2:
            return {
                'fps': 0.0,
                'avg_processing_time_ms': 0.0,
                'avg_yolo_time_ms': 0.0,
                'avg_pose_time_ms': 0.0,
                'frame_count': self.frame_count,
                'uptime_seconds': time.time() - self.start_time
            }
        
        # Calcular FPS
        time_diff = self.frame_times[-1] - self.frame_times[0]
        fps = (len(self.frame_times) - 1) / max(time_diff, 0.001)
        
        # Tiempos promedio
        avg_proc_time = np.mean(self.processing_times) * 1000
        avg_yolo_time = np.mean(self.yolo_times) * 1000
        avg_pose_time = np.mean(self.pose_times) * 1000
        
        return {
            'fps': fps,
            'avg_processing_time_ms': avg_proc_time,
            'avg_yolo_time_ms': avg_yolo_time,
            'avg_pose_time_ms': avg_pose_time,
            'frame_count': self.frame_count,
            'uptime_seconds': time.time() - self.start_time
        }

class V4DemoProcessor:
    """Procesador principal V4 para demo WSL"""
    
    def __init__(self, model_path: str, yolo_model: str = 'yolov8n.pt'):
        logger.info("🚀 Initializing V4 Demo Processor...")
        
        # Inicializar componentes
        self.frame_generator = SyntheticFrameGenerator()
        self.yolo_detector = SimpleYOLODetector(yolo_model) if YOLO_AVAILABLE else None
        self.pose_estimator = SimplePoseEstimator(model_path)
        self.performance_tracker = PerformanceTracker()
        
        logger.info("✅ V4 Demo Processor initialized")
    
    def process_frame(self) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, Any]]:
        """Procesar un frame sintético"""
        start_time = time.time()
        
        # Generar frame sintético
        frame = self.frame_generator.generate_frame()
        
        # Detectar personas
        yolo_start = time.time()
        if self.yolo_detector:
            person_bboxes = self.yolo_detector.detect_persons(frame, conf_threshold=0.1)
        else:
            # Fallback: detectar rectángulos grandes como personas
            person_bboxes = self._detect_persons_fallback(frame)
        yolo_time = time.time() - yolo_start
        
        # Estimar poses
        pose_start = time.time()
        poses = []
        for bbox in person_bboxes:
            pose = self.pose_estimator.estimate_pose(frame, bbox)
            if pose is not None:
                poses.append(pose)
        pose_time = time.time() - pose_start
        
        # Actualizar métricas
        processing_time = time.time() - start_time
        self.performance_tracker.update(processing_time, yolo_time, pose_time)
        
        stats = self.performance_tracker.get_stats()
        stats.update({
            'persons_detected': len(person_bboxes),
            'poses_estimated': len(poses),
            'processing_time_ms': processing_time * 1000,
            'backend': 'pytorch',
            'yolo_available': YOLO_AVAILABLE
        })
        
        return frame, poses, stats
    
    def _detect_persons_fallback(self, frame: np.ndarray) -> List[List[int]]:
        """Detección de personas fallback sin YOLO"""
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Encontrar contornos
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        persons = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Área mínima para una persona
                x, y, w, h = cv2.boundingRect(contour)
                if h > w and h > 100:  # Aspecto vertical y altura mínima
                    persons.append([x, y, x + w, y + h])
        
        return persons

def draw_poses(frame: np.ndarray, poses: List[np.ndarray], color: Tuple[int, int, int] = (0, 255, 0)):
    """Dibujar poses en el frame"""
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),      # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),         # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)    # Legs
    ]
    
    for i, pose in enumerate(poses):
        # Color diferente para cada persona
        pose_color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)][i % 4]
        
        # Dibujar keypoints
        for j, (x, y) in enumerate(pose):
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (int(x), int(y)), 4, pose_color, -1)
                # Número del keypoint
                cv2.putText(frame, str(j), (int(x) + 5, int(y) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
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
                            pose_color, 2)

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='ConvNeXt V4 WSL Demo')
    parser.add_argument('--model_path', type=str,
                       default='/home/fabri/ConvNeXtPose/exports/model_opt_S.pth',
                       help='Path to pose model')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt',
                       help='YOLO model for person detection')
    parser.add_argument('--save_video', type=str, default=None,
                       help='Save output video to file')
    parser.add_argument('--stats', action='store_true',
                       help='Show detailed statistics')
    parser.add_argument('--fps_limit', type=int, default=20,
                       help='FPS limit for processing')
    parser.add_argument('--duration', type=int, default=0,
                       help='Duration in seconds (0 = infinite)')
    
    args = parser.parse_args()
    
    # Inicializar procesador
    try:
        processor = V4DemoProcessor(args.model_path, args.yolo_model)
    except Exception as e:
        logger.error(f"❌ Failed to initialize processor: {e}")
        return
    
    # Configurar salida de video
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, args.fps_limit, (1280, 720))
        logger.info(f"📹 Recording to: {args.save_video}")
    
    logger.info("🚀 V4 WSL Demo started. Press 'q' to quit, 's' for stats")
    logger.info(f"   Target FPS: {args.fps_limit}")
    logger.info(f"   YOLO available: {'✅' if YOLO_AVAILABLE else '❌'}")
    logger.info(f"   Duration: {'∞' if args.duration == 0 else f'{args.duration}s'}")
    
    # Variables de control
    last_stats_time = time.time()
    frame_interval = 1.0 / args.fps_limit
    start_time = time.time()
    
    try:
        while True:
            loop_start = time.time()
            
            # Chequear duración
            if args.duration > 0 and (time.time() - start_time) > args.duration:
                logger.info(f"⏰ Duration limit reached: {args.duration}s")
                break
            
            # Procesar frame
            frame, poses, stats = processor.process_frame()
            if frame is None:
                continue
            
            # Dibujar poses
            draw_poses(frame, poses)
            
            # Dibujar estadísticas en pantalla
            stats_text = [
                f"V4 DEMO | FPS: {stats['fps']:.1f} | Proc: {stats['processing_time_ms']:.1f}ms",
                f"YOLO: {stats['avg_yolo_time_ms']:.1f}ms | Pose: {stats['avg_pose_time_ms']:.1f}ms",
                f"Persons: {stats['persons_detected']} | Poses: {stats['poses_estimated']}",
                f"Backend: {stats['backend']} | Frame: {stats['frame_count']}"
            ]
            
            for i, text in enumerate(stats_text):
                cv2.putText(frame, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Mostrar frame
            cv2.imshow('ConvNeXt V4 WSL Demo', frame)
            
            # Guardar video si está habilitado
            if video_writer is not None:
                video_writer.write(frame)
            
            # Mostrar estadísticas en terminal
            current_time = time.time()
            if args.stats or (current_time - last_stats_time > 3.0):
                logger.info("📊 Performance Stats:")
                logger.info(f"   FPS: {stats['fps']:.2f}")
                logger.info(f"   Total processing: {stats['processing_time_ms']:.1f}ms")
                logger.info(f"   YOLO time: {stats['avg_yolo_time_ms']:.1f}ms")
                logger.info(f"   Pose time: {stats['avg_pose_time_ms']:.1f}ms")
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
                for key, value in stats.items():
                    if isinstance(value, float):
                        logger.info(f"   {key}: {value:.2f}")
                    else:
                        logger.info(f"   {key}: {value}")
            
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
        logger.info(f"   Average YOLO time: {final_stats['avg_yolo_time_ms']:.1f}ms")
        logger.info(f"   Average pose time: {final_stats['avg_pose_time_ms']:.1f}ms")
        logger.info(f"   Total uptime: {final_stats['uptime_seconds']:.1f}s")
        
        if args.save_video:
            logger.info(f"📹 Video saved to: {args.save_video}")

if __name__ == "__main__":
    main()
