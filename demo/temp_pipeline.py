#!/usr/bin/env python3
"""
ConvNeXt Ultra Performance 3D - FIXED VERSION
============================================
- ONNX/TFLite optimizado para 30+ FPS
- Threading inteligente
- Frame skipping adaptativo
- Cache de detección inteligente
- Basado en convnext_clean pero con optimizaciones de rendimiento
"""

import os
import sys
import time
import logging
import cv2
import numpy as np
import torch
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import argparse

# Configuración de entorno optimizada
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '4'
warnings.filterwarnings("ignore", category=UserWarning)

# Agregar path del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Importar nuestro detector optimizado
from optimized_yolo_detector_fixed import OptimizedYOLODetectorFixed


# Importar componentes existentes
from convnext_pose_production_final_corrected import (
    RobustPoseProcessor,
    MODEL_CONFIGS,
    visualize_pose_2d
)

# Importar RootNet wrapper para 3D
try:
    from demo.root_wrapper import RootNetWrapper
except ImportError:
    # Fallback si no existe
    RootNetWrapper = None
    logger.warning("⚠️ RootNetWrapper no disponible, 3D deshabilitado")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 🎯 PRESETS DE RENDIMIENTO OPTIMIZADOS
ULTRA_PERFORMANCE_PRESETS = {
    'ultra_fast_30fps': {
        'target_fps': 30.0,
        'frame_skip': 2,
        'max_persons': 1,
        'detection_freq': 4,  # Detección cada 4 frames
        'thread_count': 2,
        'enable_threading': True,
        'enable_cache': True,
        'output_3d': False,
        'description': 'Ultra rápido - 30+ FPS con cache (2D por defecto)'
    },
    'ultra_fast_30fps_3d': {
        'target_fps': 30.0,
        'frame_skip': 2,
        'max_persons': 1,
        'detection_freq': 4,
        'thread_count': 2,
        'enable_threading': True,
        'enable_cache': True,
        'output_3d': True,
        'description': 'Ultra rápido - 30+ FPS con cache (3D activado)'
    },
    'balanced_25fps': {
        'target_fps': 25.0,
        'frame_skip': 1,
        'max_persons': 2,
        'detection_freq': 3,
        'thread_count': 2,
        'enable_threading': True,
        'enable_cache': True,
        'output_3d': False,
        'description': 'Balance - 25+ FPS'
    },
    'balanced_25fps_3d': {
        'target_fps': 25.0,
        'frame_skip': 1,
        'max_persons': 2,
        'detection_freq': 3,
        'thread_count': 2,
        'enable_threading': True,
        'enable_cache': True,
        'output_3d': True,
        'description': 'Balance - 25+ FPS (3D activado)'
    },
    'quality_20fps': {
        'target_fps': 20.0,
        'frame_skip': 1,
        'max_persons': 3,
        'detection_freq': 2,
        'thread_count': 1,
        'enable_threading': False,
        'enable_cache': False,
        'output_3d': False,
        'description': 'Calidad - 20+ FPS'
    },
    'quality_20fps_3d': {
        'target_fps': 20.0,
        'frame_skip': 1,
        'max_persons': 3,
        'detection_freq': 2,
        'thread_count': 1,
        'enable_threading': False,
        'enable_cache': False,
        'output_3d': True,
        'description': 'Calidad - 20+ FPS (3D activado)'
    }
}

class UltraOptimizedPipeline:
    """Pipeline ultra optimizado basado en convnext_clean con mejoras de rendimiento y soporte 3D opcional"""
    def __init__(self, 
                 model_type: str = 'XS',
                 backend: str = 'onnx',
                 preset: str = 'ultra_fast_30fps',
                 rootnet_path: str = '/home/user/convnextpose_esteban/3DMPPE_ROOTNET_RELEASE',
                 rootnet_ckpt: str = 'demo/snapshot_18.pth.tar'):
        self.model_type = model_type
        self.backend = backend
        self.preset = preset
        self.config = ULTRA_PERFORMANCE_PRESETS[preset].copy()
        logger.info("🚀 ULTRA OPTIMIZED PIPELINE")
        logger.info("=" * 50)
        logger.info(f"   🎯 Modelo ConvNeXt: {model_type}")
        logger.info(f"   ⚙️ Backend: {backend}")
        logger.info(f"   🎪 Preset: {preset}")
        logger.info(f"   📈 Target FPS: {self.config['target_fps']}")
        logger.info(f"   👤 Máx personas: {self.config['max_persons']}")
        logger.info(f"   🔧 Threading: {self.config['enable_threading']}")
        logger.info(f"   💾 Cache: {self.config['enable_cache']}")
        logger.info(f"   📦 Output 3D: {self.config.get('output_3d', False)}")
        # Inicializar componentes
        self._initialize_components()
        # Threading controlado
        self.thread_pool = None
        if self.config['enable_threading']:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config['thread_count'])
        # Estadísticas y cache
        self.frame_count = 0
        self.successful_poses = 0
        self.processing_times = deque(maxlen=30)
        self.detection_cache = deque(maxlen=10)
        self.last_detection = None
        self.cache_hits = 0
        # Inicializar RootNet solo si se requiere 3D
        self.rootnet = None
        if self.config.get('output_3d', False):
            logger.info("\n🦴 Inicializando RootNet para 3D...")
            self.rootnet = RootNetWrapper(rootnet_path, rootnet_ckpt)
            self.rootnet.load_model(use_gpu=torch.cuda.is_available())
        
    def _initialize_components(self):
        """Inicializar YOLO optimizado + RobustPoseProcessor"""
        
        # 1. YOLO Optimizado FIXED
        logger.info("\\n📥 Inicializando YOLO Ultra Optimizado...")
        try:
            self.yolo_detector = OptimizedYOLODetectorFixed(
                conf_threshold=0.5,
                max_persons=self.config['max_persons']
            )
            logger.info("✅ YOLO Ultra Optimizado iniciado")
        except Exception as e:
            logger.error(f"❌ Error YOLO: {e}")
            raise
        
        # 2. RobustPoseProcessor (confiable y optimizado)
        logger.info("\\n🦴 Inicializando RobustPoseProcessor...")
        try:
            self.pose_processor = RobustPoseProcessor(
                model_type=self.model_type,
                backend=self.backend,
                use_yolo_tflite=False  # Ya tenemos nuestro YOLO optimizado
            )
            logger.info("✅ RobustPoseProcessor iniciado")
        except Exception as e:
            logger.error(f"❌ Error Pose: {e}")
            raise
        
        logger.info("\\n✅ PIPELINE ULTRA OPTIMIZADO COMPLETAMENTE INICIALIZADO")
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Procesa un frame con optimizaciones de cache y threading. Si output_3d está activo, retorna también coordenadas 3D.
        Returns:
            poses: Lista de poses detectadas (2D o 3D)
            stats: Estadísticas de rendimiento
        """
        start_time = time.time()
        self.frame_count += 1
        # Frame skipping inteligente
        if self._should_skip_frame():
            total_time = time.time() - start_time
            stats = self._get_basic_stats(total_time, skipped=True)
            return [], stats
        # Detección con cache inteligente
        detect_start = time.time()
        bbox, confidence = self._detect_person_cached(frame)
        detect_time = time.time() - detect_start
        if bbox is None:
            total_time = time.time() - start_time
            stats = {
                'detection_time': detect_time * 1000,
                'pose_time': 0,
                'total_time': total_time * 1000,
                'confidence': 0,
                'detected': False,
                'persons_detected': 0,
                'poses_detected': 0,
                'instant_fps': 1.0 / total_time if total_time > 0 else 0,
                'backend_used': self.backend,
                'model_type': self.model_type,
                'bbox': None,
                'frame_count': self.frame_count,
                'cache_hits': self.cache_hits,
                'skipped': False
            }
            return [], stats
        # Estimación de pose
        pose_start = time.time()
        try:
            pose_2d = self.pose_processor._process_single_person(frame, bbox)
            poses = []
            pose_3d = None
            if pose_2d is not None:
                if self.config.get('output_3d', False) and self.rootnet is not None:
                    # Calcular profundidad root con RootNet
                    root_depth = self.rootnet.predict_depth(frame, bbox)
                    # Expandir a 3D: (x, y, z) para cada joint (z = root_depth para todos)
                    pose_3d = np.zeros((pose_2d.shape[0], 3), dtype=np.float32)
                    pose_3d[:, :2] = pose_2d
                    pose_3d[:, 2] = root_depth
                    poses = [pose_3d]
                else:
                    poses = [pose_2d]
                self.successful_poses += 1
        except Exception as e:
            logger.debug(f"Error en pose estimation: {e}")
            poses = []
        pose_time = time.time() - pose_start
        total_time = time.time() - start_time
        # Actualizar estadísticas
        self.processing_times.append(total_time)
        # Estadísticas completas
        stats = {
            'detection_time': detect_time * 1000,
            'pose_time': pose_time * 1000,
            'total_time': total_time * 1000,
            'confidence': confidence,
            'detected': len(poses) > 0,
            'persons_detected': 1 if bbox is not None else 0,
            'poses_detected': len(poses),
            'instant_fps': 1.0 / total_time if total_time > 0 else 0,
            'backend_used': self.backend,
            'model_type': self.model_type,
            'bbox': bbox,
            'frame_count': self.frame_count,
            'cache_hits': self.cache_hits,
            'skipped': False,
            'output_3d': self.config.get('output_3d', False)
        }
        return poses, stats
    
    def _should_skip_frame(self) -> bool:
        """Frame skipping inteligente basado en rendimiento"""
        if len(self.processing_times) < 5:
            return False
        
        # Calcular FPS actual de los últimos 5 frames
        recent_times = list(self.processing_times)[-5:]
        avg_time = np.mean(recent_times)
        current_fps = 1.0 / avg_time if avg_time > 0 else 0
        target_fps = self.config['target_fps']
        
        # Skip si estamos muy por debajo del target
        if current_fps < target_fps * 0.8:
            if self.frame_count % self.config['frame_skip'] != 0:
                return True
        
        return False
    
    def _detect_person_cached(self, frame: np.ndarray) -> tuple:
        """Detección con cache inteligente"""
        
        # Usar cache si está habilitado y no es momento de detectar
        if (self.config['enable_cache'] and 
            self.frame_count % self.config['detection_freq'] != 0 and
            self.last_detection is not None):
            
            bbox, confidence = self.last_detection
            self.cache_hits += 1
            return bbox, confidence
        
        # Nueva detección
        bbox, confidence = self.yolo_detector.detect_person(frame)
        
        # Actualizar cache
        if self.config['enable_cache']:
            self.last_detection = (bbox, confidence)
        
        return bbox, confidence
    
    def _get_basic_stats(self, total_time: float, skipped: bool = False) -> dict:
        """Estadísticas básicas para frames skipped"""
        return {
            'detection_time': 0,
            'pose_time': 0,
            'total_time': total_time * 1000,
            'confidence': 0,
            'detected': False,
            'persons_detected': 0,
            'poses_detected': 0,
            'instant_fps': 1.0 / total_time if total_time > 0 else 0,
            'backend_used': self.backend,
            'model_type': self.model_type,
            'bbox': None,
            'frame_count': self.frame_count,
            'cache_hits': self.cache_hits,
            'skipped': skipped
        }
    
    def get_performance_stats(self) -> dict:
        """Obtener estadísticas de rendimiento"""
        if not self.processing_times:
            return {}
        
        avg_time = np.mean(self.processing_times) * 1000
        fps = 1000 / avg_time if avg_time > 0 else 0
        success_rate = (self.successful_poses / self.frame_count) * 100 if self.frame_count > 0 else 0
        cache_rate = (self.cache_hits / self.frame_count) * 100 if self.frame_count > 0 else 0
        
        return {
            'total_frames': self.frame_count,
            'successful_poses': self.successful_poses,
            'success_rate': success_rate,
            'avg_time': avg_time,
            'fps': fps,
            'cache_hits': self.cache_hits,
            'cache_rate': cache_rate,
            'preset': self.preset,
            'backend': self.backend
        }
    
    def cleanup(self):
        """Limpiar recursos"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)

def main():
    """Función principal ultra optimizada"""
    parser = argparse.ArgumentParser(description='ConvNeXt Ultra Performance Fixed')
    parser.add_argument('--preset', choices=list(ULTRA_PERFORMANCE_PRESETS.keys()), 
                       default='ultra_fast_30fps', help='Preset de rendimiento')
    parser.add_argument('--backend', choices=['onnx', 'tflite', 'pytorch'], 
                       default='onnx', help='Backend de inferencia')
    parser.add_argument('--model', choices=['XS', 'S'], 
                       default='XS', help='Modelo ConvNeXt')
    parser.add_argument('--input', type=str, default='0',
                       help='Entrada: webcam (0) o archivo de video')
    parser.add_argument('--output', type=str, default=None,
                       help='Archivo de salida de video')
    parser.add_argument('--output_npz', type=str, default=None,
                       help='Archivo de salida de coordenadas 3D (.npz)')
    parser.add_argument('--output_video', type=str, default=None,
                       help='Archivo de salida de video (.mp4)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Ejecutar benchmark de rendimiento')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duración del benchmark (segundos)')
    parser.add_argument('--rootnet_path', type=str, default='/home/user/convnextpose_esteban/3DMPPE_ROOTNET_RELEASE',
                       help='Ruta al repo RootNet')
    parser.add_argument('--rootnet_ckpt', type=str, default='demo/snapshot_18.pth.tar',
                       help='Checkpoint RootNet')
    
    args = parser.parse_args()
    
    # Detectar hardware
    hardware_info = {
        'has_cuda': torch.cuda.is_available(),
        'cpu_count': os.cpu_count(),
    }
    
    logger.info("💻 DETECCIÓN DE HARDWARE:")
    logger.info(f"   🔧 CPU cores: {hardware_info['cpu_count']}")
    logger.info(f"   🚀 CUDA disponible: {hardware_info['has_cuda']}")
    
    # Seleccionar preset
    preset = args.preset
    
    try:
        # Crear pipeline ultra optimizado
        pipeline = UltraOptimizedPipeline(
            model_type=args.model,
            backend=args.backend,
            preset=preset,
            rootnet_path=args.rootnet_path,
            rootnet_ckpt=args.rootnet_ckpt
