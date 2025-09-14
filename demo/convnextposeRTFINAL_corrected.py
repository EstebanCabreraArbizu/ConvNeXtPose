#!/usr/bin/env python3

"""
Test 3D Isolated - Sin conflictos de imports
============================================
"""

import os
import sys
import cv2
import numpy as np
import time
import logging
import torch
import math
import importlib.util
from pathlib import Path
from contextlib import contextmanager
import torchvision.transforms as transforms
from collections import OrderedDict, deque
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from threading import Lock, Event
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import argparse
import traceback
from datetime import datetime

# TensorFlow Lite import
TFLITE_AVAILABLE = False
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False

# Agregar path del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_torchvision_patch():
    """Aplica un parche a torchvision para compatibilidad con código antiguo"""
    import torchvision
    import torchvision.models.resnet as resnet_module
    
    # Solo aplicar si no existe model_urls
    if not hasattr(resnet_module, 'model_urls'):
        # Recrear model_urls como existía en versiones antiguas
        resnet_module.model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        }
        logger.info("[INFO] Se aplicó parche de compatibilidad para torchvision")

@dataclass
class FrameData:
    """Estructura de datos para frames en pipeline paralelo"""
    frame_id: int
    frame: np.ndarray
    timestamp: float
    bbox: Optional[List[int]] = None
    confidence: Optional[float] = None
    pose_2d: Optional[np.ndarray] = None
    pose_3d: Optional[np.ndarray] = None
    depth: Optional[float] = None
    processing_times: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.processing_times is None:
            self.processing_times = {}

class PerformanceConfig:
    """Configuración de rendimiento basada en preset"""
    
    # Atributos explícitos para type checking
    max_workers: int
    cache_size: int
    cache_timeout: float
    frame_skip: int
    adaptive_skip: bool
    parallel_stages: bool
    target_fps: int
    
    PRESET_CONFIGS = {
        'ultra_fast_30fps': {
            'max_workers': 4,
            'cache_size': 100,
            'cache_timeout': 0.1,
            'frame_skip': 2,
            'adaptive_skip': True,
            'parallel_stages': True,
            'target_fps': 30
        },
        'ultra_fast_30fps_3d': {
            'max_workers': 5,
            'cache_size': 150,
            'cache_timeout': 0.15,
            'frame_skip': 1,
            'adaptive_skip': True,
            'parallel_stages': True,
            'target_fps': 25
        },
        'balanced_25fps': {
            'max_workers': 3,
            'cache_size': 80,
            'cache_timeout': 0.12,
            'frame_skip': 1,
            'adaptive_skip': True,
            'parallel_stages': True,
            'target_fps': 25
        },
        'balanced_25fps_3d': {
            'max_workers': 4,
            'cache_size': 120,
            'cache_timeout': 0.18,
            'frame_skip': 1,
            'adaptive_skip': True,
            'parallel_stages': True,
            'target_fps': 20
        },
        'quality_20fps': {
            'max_workers': 2,
            'cache_size': 50,
            'cache_timeout': 0.08,
            'frame_skip': 0,
            'adaptive_skip': False,
            'parallel_stages': False,
            'target_fps': 20
        },
        'quality_20fps_3d': {
            'max_workers': 3,
            'cache_size': 80,
            'cache_timeout': 0.2,
            'frame_skip': 0,
            'adaptive_skip': False,
            'parallel_stages': True,
            'target_fps': 15
        }
    }
    
    def __init__(self, preset: str = 'balanced_25fps_3d'):
        config = self.PRESET_CONFIGS.get(preset, self.PRESET_CONFIGS['balanced_25fps_3d'])
        for key, value in config.items():
            setattr(self, key, value)
        
        # Ajustes dinámicos según hardware
        cpu_count = mp.cpu_count()
        if cpu_count >= 8:
            self.max_workers = min(self.max_workers + 2, 8)
        elif cpu_count <= 4:
            self.max_workers = max(self.max_workers - 1, 2)
            
        logger.info(f"🔧 Performance Config ({preset}):")
        logger.info(f"   Workers: {self.max_workers}, Cache: {self.cache_size}, Skip: {self.frame_skip}")

class IntelligentCache:
    """Cache inteligente con TTL y gestión de memoria"""
    
    def __init__(self, max_size: int = 100, ttl: float = 0.15):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.lock = Lock()
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_key(self, bbox: List[int], timestamp: float) -> str:
        """Generar clave cuantizada para cache"""
        # Cuantizar bbox para permitir coincidencias aproximadas
        quantized_bbox = [int(x / 10) * 10 for x in bbox]
        time_slot = int(timestamp * 10) / 10  # Cuantizar tiempo a 100ms
        key_str = f"{quantized_bbox}_{time_slot}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def get(self, bbox: List[int], timestamp: float) -> Optional[Tuple[np.ndarray, float]]:
        """Obtener del cache si disponible y válido"""
        key = self._generate_key(bbox, timestamp)
        
        with self.lock:
            if key in self.cache:
                stored_time, result, depth = self.cache[key]
                if timestamp - stored_time <= self.ttl:
                    self.access_times[key] = timestamp
                    self.hit_count += 1
                    return result, depth
                else:
                    # Expirado
                    del self.cache[key]
                    del self.access_times[key]
            
            self.miss_count += 1
            return None
    
    def put(self, bbox: List[int], timestamp: float, result: np.ndarray, depth: float):
        """Guardar en cache con limpieza automática"""
        key = self._generate_key(bbox, timestamp)
        
        with self.lock:
            # Limpiar cache si está lleno
            if len(self.cache) >= self.max_size:
                self._cleanup_old_entries(timestamp)
            
            self.cache[key] = (timestamp, result.copy(), depth)
            self.access_times[key] = timestamp
    
    def _cleanup_old_entries(self, current_time: float):
        """Limpiar entradas antiguas"""
        # Eliminar entradas expiradas
        expired_keys = [
            key for key, (stored_time, _, _) in self.cache.items()
            if current_time - stored_time > self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.access_times[key]
        
        # Si aún está lleno, eliminar LRU
        if len(self.cache) >= self.max_size:
            lru_keys = sorted(self.access_times.items(), key=lambda x: x[1])[:10]
            for key, _ in lru_keys:
                if key in self.cache:
                    del self.cache[key]
                    del self.access_times[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas del cache"""
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'hit_rate': hit_rate,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count
        }

class AdaptiveFrameSkipper:
    """Frame skipping inteligente basado en rendimiento"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.frame_times = deque(maxlen=30)  # Últimos 30 frames
        self.current_skip = config.frame_skip
        self.last_adjustment = time.time()
        self.adjustment_interval = 2.0  # Ajustar cada 2 segundos
        
    def should_process_frame(self, frame_id: int, current_time: float) -> bool:
        """Determinar si procesar este frame"""
        # Si no hay skip adaptativo, usar skip fijo
        if not self.config.adaptive_skip:
            return frame_id % (self.current_skip + 1) == 0
        
        # Ajustar skip basado en rendimiento
        if current_time - self.last_adjustment > self.adjustment_interval:
            self._adjust_skip_rate()
            self.last_adjustment = current_time
        
        return frame_id % (self.current_skip + 1) == 0
    
    def record_frame_time(self, frame_time: float):
        """Registrar tiempo de procesamiento de frame"""
        self.frame_times.append(frame_time)
    
    def _adjust_skip_rate(self):
        """Ajustar tasa de skip basado en rendimiento reciente"""
        if len(self.frame_times) < 10:
            return
        
        avg_time = sum(self.frame_times) / len(self.frame_times)
        target_time = 1.0 / self.config.target_fps
        
        if avg_time > target_time * 1.2:  # 20% más lento que objetivo
            self.current_skip = min(self.current_skip + 1, 4)
            logger.debug(f"📈 Incrementando skip a {self.current_skip} (avg: {avg_time*1000:.1f}ms)")
        elif avg_time < target_time * 0.8:  # 20% más rápido que objetivo
            self.current_skip = max(self.current_skip - 1, 0)
            logger.debug(f"📉 Reduciendo skip a {self.current_skip} (avg: {avg_time*1000:.1f}ms)")

class ParallelPipelineProcessor:
    """Procesador pipeline paralelo para máximo rendimiento"""
    
    def __init__(self, config: PerformanceConfig, yolo_detector, pose_processor, rootnet_wrapper):
        self.config = config
        self.yolo = yolo_detector
        self.pose_processor = pose_processor
        self.rootnet = rootnet_wrapper
        
        # Componentes de optimización
        self.cache = IntelligentCache(config.cache_size, config.cache_timeout)
        self.frame_skipper = AdaptiveFrameSkipper(config)
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.frame_queue = queue.Queue(maxsize=config.max_workers * 2)
        self.result_queue = queue.Queue()
        
        # Estadísticas
        self.stats = {
            'frames_processed': 0,
            'frames_skipped': 0,
            'cache_hits': 0,
            'total_processing_time': 0,
            'stage_times': {'detection': [], 'pose': [], 'depth': []}
        }
        
        # Control de threads
        self.processing = False
        self.workers_started = False
        
    def process_frame_parallel(self, frame: np.ndarray, frame_id: int) -> Optional[Dict[str, Any]]:
        """Procesar frame con pipeline paralelo optimizado"""
        current_time = time.time()
        
        # Frame skipping inteligente
        if not self.frame_skipper.should_process_frame(frame_id, current_time):
            self.stats['frames_skipped'] += 1
            return None
        
        frame_start = time.time()
        
        # Crear estructura de datos
        frame_data = FrameData(
            frame_id=frame_id,
            frame=frame,
            timestamp=current_time
        )
        
        try:
            # Pipeline optimizado
            result = self._process_single_frame_optimized(frame_data)
            
            # Registrar rendimiento
            total_time = time.time() - frame_start
            self.frame_skipper.record_frame_time(total_time)
            self.stats['frames_processed'] += 1
            self.stats['total_processing_time'] += total_time
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error procesando frame {frame_id}: {e}")
            return None
    
    def _process_single_frame_optimized(self, frame_data: FrameData) -> Optional[Dict[str, Any]]:
        """Procesar un frame con todas las optimizaciones"""
        
        # Stage 1: Detection
        det_start = time.time()
        bbox, confidence = self.yolo.detect_person(frame_data.frame)
        self.stats['stage_times']['detection'].append(time.time() - det_start)
        
        if bbox is None:
            return {
                'frame_id': frame_data.frame_id,
                'detected': False,
                'bbox': None,
                'confidence': 0.0
            }
        
        # Verificar cache antes de procesar
        cached = self.cache.get(bbox, frame_data.timestamp)
        if cached:
            pose_3d, depth = cached
            self.stats['cache_hits'] += 1
            return {
                'frame_id': frame_data.frame_id,
                'detected': True,
                'bbox': bbox,
                'confidence': confidence,
                'pose_2d': pose_3d[:, :2],  # Extraer 2D del 3D cacheado
                'pose_3d': pose_3d,
                'depth': depth,
                'cached': True
            }
        
        # Stage 2: Pose estimation
        pose_start = time.time()
        pose_2d = self.pose_processor._process_single_person(frame_data.frame, bbox)
        self.stats['stage_times']['pose'].append(time.time() - pose_start)
        
        if pose_2d is None:
            return {
                'frame_id': frame_data.frame_id,
                'detected': True,
                'bbox': bbox,
                'confidence': confidence,
                'pose_2d': None
            }
        
        # Stage 3: Depth estimation (paralelo si es posible)
        depth_start = time.time()
        
        # Crear crop para RootNet
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_data.frame.shape[1], x2)
        y2 = min(frame_data.frame.shape[0], y2)
        crop = frame_data.frame[y1:y2, x1:x2]
        
        depth = None
        if crop.size > 0:
            depth = self.rootnet.predict_depth(crop, bbox)
        
        self.stats['stage_times']['depth'].append(time.time() - depth_start)
        
        # Crear pose 3D
        pose_3d = None
        if depth is not None:
            pose_3d = np.zeros((pose_2d.shape[0], 3))
            pose_3d[:, :2] = pose_2d[:, :2]
            pose_3d[:, 2] = depth
            
            # Guardar en cache
            self.cache.put(bbox, frame_data.timestamp, pose_3d, depth)
        
        return {
            'frame_id': frame_data.frame_id,
            'detected': True,
            'bbox': bbox,
            'confidence': confidence,
            'pose_2d': pose_2d,
            'pose_3d': pose_3d,
            'depth': depth,
            'cached': False
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas completas de rendimiento"""
        cache_stats = self.cache.get_stats()
        
        avg_times = {}
        for stage, times in self.stats['stage_times'].items():
            if times:
                avg_times[f'{stage}_avg_ms'] = np.mean(times) * 1000
                avg_times[f'{stage}_max_ms'] = np.max(times) * 1000
            else:
                avg_times[f'{stage}_avg_ms'] = 0
                avg_times[f'{stage}_max_ms'] = 0
        
        total_frames = self.stats['frames_processed'] + self.stats['frames_skipped']
        skip_rate = (self.stats['frames_skipped'] / max(1, total_frames)) * 100
        
        return {
            'frames_processed': self.stats['frames_processed'],
            'frames_skipped': self.stats['frames_skipped'],
            'skip_rate_percent': skip_rate,
            'cache_hit_rate': cache_stats['hit_rate'],
            'cache_size': cache_stats['size'],
            'cache_hits': self.stats['cache_hits'],
            'avg_processing_time_ms': (self.stats['total_processing_time'] / 
                                     max(1, self.stats['frames_processed'])) * 1000,
            'current_skip_rate': self.frame_skipper.current_skip,
            'max_workers': self.config.max_workers,
            'parallel_stages': self.config.parallel_stages,
            **avg_times
        }

class SimpleRootNetWrapper:
    """Wrapper híbrido: TFLite optimizado con fallback PyTorch"""
    
    def __init__(self, rootnet_path, checkpoint_path, use_tflite=True, tflite_variant="size"):
        self.rootnet_path = rootnet_path
        self.checkpoint_path = checkpoint_path
        self.use_tflite = use_tflite
        self.tflite_variant = tflite_variant
        self.model = None
        self.cfg = None
        self._original_path = None
        
        # Importar wrapper TFLite optimizado
        self.tflite_wrapper = None
        if use_tflite:
            try:
                from rootnet_tflite_wrapper import RootNetTFLiteWrapper
                self.tflite_wrapper = RootNetTFLiteWrapper(model_variant=tflite_variant)
                if self.tflite_wrapper.backbone_available:
                    logger.info(f"✅ RootNet TFLite '{tflite_variant}' wrapper cargado exitosamente")
                else:
                    logger.warning("⚠️ TFLite wrapper no disponible, usando fallback PyTorch")
                    self.use_tflite = False
            except ImportError as e:
                logger.warning(f"⚠️ No se pudo cargar TFLite wrapper: {e}")
                self.use_tflite = False
        
    @contextmanager
    def _isolated_import(self):
        """Context manager para aislar imports de RootNet."""
        # Guardar estado actual
        self._original_path = sys.path.copy()
        original_modules = set(sys.modules.keys())
        
        try:
            apply_torchvision_patch()
            # Añadir paths de RootNet
            if self.rootnet_path not in sys.path:
                sys.path.insert(0, self.rootnet_path)
                sys.path.insert(0, os.path.join(self.rootnet_path, 'main'))
                sys.path.insert(0, os.path.join(self.rootnet_path, 'data'))
                sys.path.insert(0, os.path.join(self.rootnet_path, 'common'))
            
            yield
            
        finally:
            # Restaurar estado
            sys.path = self._original_path
            # Remover módulos de RootNet para evitar conflictos
            new_modules = set(sys.modules.keys()) - original_modules
            for module in new_modules:
                if any(path in module for path in ['rootnet', 'main', 'data', 'common']):
                    sys.modules.pop(module, None)
    
    def load_model(self, use_gpu=True):
        """Carga RootNet en contexto aislado - Solo si TFLite no está disponible."""
        # Si TFLite está funcionando, no necesitamos cargar PyTorch
        if self.use_tflite and self.tflite_wrapper and self.tflite_wrapper.backbone_available:
            logger.info("✅ Usando TFLite optimizado, saltando carga PyTorch")
            return
            
        with self._isolated_import():
            try:
                # Importar módulos necesarios
                spec = importlib.util.spec_from_file_location(
                    "rootnet_model", 
                    os.path.join(self.rootnet_path, 'main', "model.py")
                )
                if spec is None or spec.loader is None:
                    logger.error("❌ No se pudo cargar módulo model.py")
                    self.model = None
                    return
                    
                rootnet_model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(rootnet_model_module)
                
                spec = importlib.util.spec_from_file_location(
                    "rootnet_config", 
                    os.path.join(self.rootnet_path, 'main', "config.py")
                )
                if spec is None or spec.loader is None:
                    logger.error("❌ No se pudo cargar módulo config.py")
                    self.model = None
                    return
                    
                rootnet_config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(rootnet_config_module)
                
                # Crear modelo siguiendo el mismo patrón que demo.py
                self.cfg = rootnet_config_module.cfg
                model = rootnet_model_module.get_pose_net(self.cfg, is_train=False)
                
                if use_gpu and torch.cuda.is_available():
                    from torch.nn.parallel.data_parallel import DataParallel
                    model = DataParallel(model).cuda()
                
                # Cargar checkpoint
                checkpoint = torch.load(self.checkpoint_path, 
                                       map_location='cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                state_dict = checkpoint.get('network', checkpoint)
                
                if not(use_gpu and torch.cuda.is_available()):
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        new_key = k.replace('module.', '')
                        new_state_dict[new_key] = v
                    state_dict = new_state_dict
                
                model.load_state_dict(state_dict)
                self.model = model.eval()
                logger.info("[INFO] RootNet PyTorch cargado como fallback")
                
            except Exception as e:
                logger.error(f"[ERROR] No se pudo cargar RootNet PyTorch: {e}")
                self.model = None
    
    def predict_depth(self, img_patch, bbox, focal=[1500, 1500]):
        """Predice profundidad usando TFLite optimizado o PyTorch fallback"""
        
        # Usar TFLite wrapper si está disponible (mucho más rápido)
        if self.use_tflite and self.tflite_wrapper and self.tflite_wrapper.backbone_available:
            try:
                return self.tflite_wrapper.predict_depth(img_patch, bbox, focal)
            except Exception as e:
                logger.warning(f"⚠️ Error en TFLite, usando fallback: {e}")
                # Continuar con PyTorch fallback
        
        # Fallback a PyTorch (original)
        if self.model is None or self.cfg is None:
            return self._fallback_depth(bbox)
        
        try:
            with self._isolated_import():
                # Importar módulos para procesamiento
                spec = importlib.util.spec_from_file_location(
                    "rootnet_utils", 
                    os.path.join(self.rootnet_path, 'common', "utils", "pose_utils.py")
                )
                if spec is None or spec.loader is None:
                    return self._fallback_depth(bbox)
                    
                rootnet_utils = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(rootnet_utils)
                
                spec = importlib.util.spec_from_file_location(
                    "rootnet_dataset", 
                    os.path.join(self.rootnet_path, 'data', "dataset.py")
                )
                if spec is None or spec.loader is None:
                    return self._fallback_depth(bbox)
                    
                rootnet_dataset = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(rootnet_dataset)
                
                # Preparar imagen
                transform = transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=self.cfg.pixel_mean, std=self.cfg.pixel_std)
                ])
                
                # Procesar bbox
                processed_bbox = rootnet_utils.process_bbox(np.array(bbox), 
                                                           img_patch.shape[1], img_patch.shape[0])
                if processed_bbox is None:
                    return self._fallback_depth(bbox)
                
                img, img2bb_trans = rootnet_dataset.generate_patch_image(img_patch, 
                                                                        processed_bbox, False, 0.0)
                img = transform(img)
                
                # Calcular k_value
                k_value = np.array([
                    math.sqrt(self.cfg.bbox_real[0] * self.cfg.bbox_real[1] * 
                             focal[0] * focal[1] / (processed_bbox[2] * processed_bbox[3]))
                ]).astype(np.float32)
                
                # Preparar tensores
                if torch.cuda.is_available():
                    img = img.cuda()[None,:,:,:]
                    k_value = torch.FloatTensor([k_value]).cuda()[None,:]
                else:
                    img = img[None,:,:,:]
                    k_value = torch.FloatTensor([k_value])[None,:]
                
                # Ejecutar modelo
                with torch.no_grad():
                    root_3d = self.model(img, k_value)
                root_depth = root_3d[0, 2].cpu().numpy()
                
                return root_depth
                
        except Exception as e:
            logger.warning(f"[WARNING] Error en predicción PyTorch: {e}")
            return self._fallback_depth(bbox)
    
    def _fallback_depth(self, bbox):
        """Fallback mejorado basado en bbox y posición."""
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        
        # Estimación más realista basada en altura del bbox
        if bbox_height > 400:  # Persona muy cerca
            estimated_depth = 900 + np.random.normal(0, 150)
        elif bbox_height > 300:  # Persona cerca
            estimated_depth = 1400 + np.random.normal(0, 200)
        elif bbox_height > 200:  # Persona distancia media
            estimated_depth = 2200 + np.random.normal(0, 350)
        elif bbox_height > 120:  # Persona lejos
            estimated_depth = 3500 + np.random.normal(0, 500)
        else:  # Persona muy lejos
            estimated_depth = 5000 + np.random.normal(0, 700)
        
        # Ajustar por posición horizontal (perspectiva)
        bbox_center_x = (x1 + x2) / 2
        if bbox_center_x < 640:  # Lado izquierdo
            estimated_depth *= 1.1
        elif bbox_center_x > 1280:  # Lado derecho
            estimated_depth *= 1.1
        
        # Variación por área total
        if bbox_area > 40000:  # Muy cerca
            estimated_depth *= 0.8
        elif bbox_area < 8000:  # Muy lejos
            estimated_depth *= 1.3
        
        return max(min(estimated_depth, 6000.0), 700.0)

def pixel2cam(pose_coord, focal, princpt):
    """
    Convierte coordenadas de píxeles a coordenadas métricas de cámara
    Replicando exactamente la función del common/utils/pose_utils.py
    """
    x = (pose_coord[:, 0] - princpt[0]) / focal[0] * pose_coord[:, 2]
    y = (pose_coord[:, 1] - princpt[1]) / focal[1] * pose_coord[:, 2]
    z = pose_coord[:, 2]
    return np.stack((x, y, z), axis=1)

def process_convnext_depth(pose_3d_raw, root_depth, cfg_depth_dim=64, cfg_bbox_3d_shape=[2000, 2000, 2000]):
    """
    Pipeline completo de profundidad siguiendo exactamente el patrón de demo.py
    
    Args:
        pose_3d_raw: Output crudo del modelo ConvNeXtPose (x, y, z_relative)
        root_depth: Profundidad absoluta del centro de masa de RootNet
        cfg_depth_dim: Número de bins de discretización de profundidad (default: 64)
        cfg_bbox_3d_shape: Forma del bounding box 3D en mm (default: [2000, 2000, 2000])
    
    Returns:
        pose_3d_absolute: Coordenadas 3D absolutas con profundidad relativa aplicada
    """
    # Copiar para no modificar el original
    pose_3d = pose_3d_raw.copy()
    
    # PASO 1: Transformación de profundidad relativa (siguiendo demo.py línea por línea)
    # "root-relative discretized depth -> absolute continuous depth"
    pose_3d[:, 2] = (pose_3d[:, 2] / cfg_depth_dim * 2 - 1) * \
                    (cfg_bbox_3d_shape[0]/2) + root_depth
    
    return pose_3d

def visualize_pose_complete(frame, pose_2d, bbox, confidence, depth=None, is_3d=False):
    """Visualización completa de la detección con pose y bbox"""
    output_frame = frame.copy()
    
    # 1. Dibujar bounding box
    if bbox is not None:
        x1, y1, x2, y2 = [int(x) for x in bbox]
        # Bounding box verde para 3D, azul para 2D
        color = (0, 255, 0) if is_3d else (255, 0, 0)
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        
        # Etiqueta del bbox
        label = f"3D Person" if is_3d else "2D Person"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(output_frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(output_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 2. Dibujar esqueleto de pose 2D
    if pose_2d is not None:
        # Definir conexiones del esqueleto humano (COCO format)
        skeleton_connections = [
            # Cabeza y cuello
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Brazos
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            # Torso
            (5, 11), (6, 12), (11, 12),
            # Piernas
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        # Colores para diferentes partes del cuerpo
        joint_colors = [
            (255, 0, 0),    # 0: nariz - rojo
            (255, 85, 0),   # 1: ojo izq - naranja
            (255, 170, 0),  # 2: ojo der - amarillo
            (255, 255, 0),  # 3: oreja izq - amarillo
            (170, 255, 0),  # 4: oreja der - lima
            (85, 255, 0),   # 5: hombro izq - verde claro
            (0, 255, 0),    # 6: hombro der - verde
            (0, 255, 85),   # 7: codo izq - verde azulado
            (0, 255, 170),  # 8: codo der - cian claro
            (0, 255, 255),  # 9: muñeca izq - cian
            (0, 170, 255),  # 10: muñeca der - azul claro
            (0, 85, 255),   # 11: cadera izq - azul
            (0, 0, 255),    # 12: cadera der - azul oscuro
            (85, 0, 255),   # 13: rodilla izq - violeta
            (170, 0, 255),  # 14: rodilla der - magenta
            (255, 0, 255),  # 15: tobillo izq - magenta
            (255, 0, 170)   # 16: tobillo der - rosa
        ]
        
        # Dibujar conexiones (huesos)
        for connection in skeleton_connections:
            joint1_idx, joint2_idx = connection
            if joint1_idx < len(pose_2d) and joint2_idx < len(pose_2d):
                x1, y1 = pose_2d[joint1_idx][:2]
                x2, y2 = pose_2d[joint2_idx][:2]
                
                # Solo dibujar si ambos joints son válidos
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                            (255, 255, 255), 3)  # Líneas blancas gruesas
        
        # Dibujar joints (articulaciones)
        for i, (x, y) in enumerate(pose_2d[:, :2]):
            if x > 0 and y > 0:  # Joint válido
                color = joint_colors[i % len(joint_colors)]
                cv2.circle(output_frame, (int(x), int(y)), 5, color, -1)
                cv2.circle(output_frame, (int(x), int(y)), 5, (255, 255, 255), 2)  # Borde blanco
    
    # 3. Información de estado en la parte superior
    info_y = 30
    
    if is_3d and depth is not None:
        cv2.putText(output_frame, f"3D MODE - Depth: {depth:.1f}mm", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        info_y += 35
    else:
        cv2.putText(output_frame, "2D MODE", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        info_y += 35
    
    if confidence is not None:
        cv2.putText(output_frame, f"Confidence: {confidence:.3f}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        info_y += 30
    
    # 4. Información de joints detectados
    if pose_2d is not None:
        valid_joints = np.sum((pose_2d[:, 0] > 0) & (pose_2d[:, 1] > 0))
        cv2.putText(output_frame, f"Joints: {valid_joints}/{len(pose_2d)}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return output_frame

def test_3d_complete(args):
    """Test completo del pipeline 3D"""
    
    try:
        # Importar componentes directamente
        from optimized_yolo_detector_fixed import OptimizedYOLODetectorFixed
        from convnext_pose_production_final_corrected import RobustPoseProcessor, visualize_pose_2d
        
        logger.info("✅ Imports base exitosos")
        
        # 1. Inicializar YOLO
        logger.info("🔧 Inicializando YOLO...")
        yolo = OptimizedYOLODetectorFixed(conf_threshold=0.7, max_persons=1)
        logger.info("✅ YOLO listo")
        
        # 2. Inicializar ConvNeXt
        logger.info("🦴 Inicializando ConvNeXt...")
        pose_processor = RobustPoseProcessor(model_type='XS', backend=args.backend)
        logger.info("✅ ConvNeXt listo")
        
        # 3. Inicializar RootNet con TFLite optimizado
        logger.info("📦 Inicializando RootNet TFLite optimizado...")
        
        # Determinar configuración de backend
        use_tflite = args.use_tflite
        if args.rootnet_backend == 'pytorch':
            use_tflite = False
        elif args.rootnet_backend == 'tflite':
            use_tflite = True
        # 'auto' usa el valor de args.use_tflite
        
        rootnet = SimpleRootNetWrapper(
            args.rootnet_path,
            args.rootnet_ckpt,
            use_tflite=use_tflite,
            tflite_variant=args.tflite_variant
        )
        rootnet.load_model(use_gpu=False)
        
        # Reportar configuración final
        if rootnet.use_tflite and rootnet.tflite_wrapper and rootnet.tflite_wrapper.backbone_available:
            logger.info(f"✅ RootNet TFLite '{args.tflite_variant}' listo")
        else:
            logger.info("✅ RootNet PyTorch fallback listo")
        
        # 4. Inicializar Pipeline Paralelo Optimizado
        logger.info("🚀 Inicializando Pipeline Paralelo...")
        perf_config = PerformanceConfig(args.preset)
        parallel_processor = ParallelPipelineProcessor(perf_config, yolo, pose_processor, rootnet)
        logger.info(f"✅ Pipeline paralelo listo ({perf_config.max_workers} workers)")
        
        # 5. Procesar video
        logger.info("🎬 Procesando video con 3D...")
        video_path = "barbell biceps curl_12.mp4"
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"❌ No se pudo abrir: {video_path}")
            return
        
        # Propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"📊 Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Configurar writer
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        writer = cv2.VideoWriter('demo/output_3d_complete.mp4', fourcc, fps, (width, height))
        
        frame_count = 0
        poses_3d_data = []
        poses_2d_success = 0
        poses_3d_success = 0
        start_time = time.time()
        
        # Estadísticas de rendimiento del pipeline paralelo
        processing_times = []
        last_stats_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_start = time.time()
            
            # OPCIÓN 1: Pipeline paralelo optimizado (si es compatible)
            try:
                result = parallel_processor.process_frame_parallel(frame, frame_count)
                
                if result and result['detected']:
                    bbox = result['bbox']
                    confidence = result['confidence']
                    pose_2d = result['pose_2d']
                    pose_3d = result['pose_3d']
                    depth = result['depth']
                    
                    # Usar resultados del pipeline paralelo
                    if pose_2d is not None:
                        poses_2d_success += 1
                    if pose_3d is not None:
                        poses_3d_success += 1
                else:
                    bbox, confidence, pose_2d, pose_3d, depth = None, None, None, None, None
                    
            except Exception as e:
                logger.warning(f"⚠️ Pipeline paralelo falló, usando secuencial: {e}")
                # FALLBACK: Pipeline secuencial original
                bbox, confidence = yolo.detect_person(frame)
                pose_2d, pose_3d, depth = None, None, None
                
                if bbox is not None:
                    pose_2d = pose_processor._process_single_person(frame, bbox)
                    if pose_2d is not None:
                        poses_2d_success += 1
                        # Depth estimation simple
                        try:
                            x1, y1, x2, y2 = bbox
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                depth = rootnet.predict_depth(crop, bbox)
                                if depth:
                                    pose_3d = np.zeros((pose_2d.shape[0], 3))
                                    pose_3d[:, :2] = pose_2d[:, :2]
                                    pose_3d[:, 2] = depth
                                    poses_3d_success += 1
                        except Exception as depth_e:
                            logger.warning(f"⚠️ Depth estimation falló: {depth_e}")
            
            # FPS del frame
            frame_time = time.time() - frame_start
            frame_fps = 1.0 / frame_time if frame_time > 0 else 0
            processing_times.append(frame_time)
            
            # Procesar resultados y generar frame de salida
            output_frame = frame.copy()
            
            if bbox is not None and pose_2d is not None:
                # Crear estadísticas para visualización
                stats = {
                    'confidence': confidence if confidence is not None else 0.0,
                    'is_3d': pose_3d is not None,
                    'bbox': bbox,
                    'frame_count': frame_count,
                    'fps': frame_fps,
                    'instant_fps': frame_fps,
                    'persons_detected': 1,
                    'poses_detected': 1,
                    'backend_used': args.backend,
                    'model_type': 'XS',
                    'total_time': frame_time * 1000,
                    'detection_time': 0,
                    'pose_time': 0,
                    'detected': True
                }
                
                # Añadir información de profundidad si está disponible
                if depth is not None:
                    stats['depth'] = depth
                
                # Visualizar resultados
                output_frame = visualize_pose_2d(frame, [pose_2d], stats)
                
                # Añadir información específica según el tipo de detección
                if pose_3d is not None and depth is not None:
                    cv2.putText(output_frame, f"3D MODE - Depth: {depth:.1f}mm", 
                               (10, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(output_frame, f"3D Confidence: {confidence:.3f}", 
                               (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Guardar datos 3D
                    poses_3d_data.append({
                        'frame': frame_count,
                        'pose3d': pose_3d,
                        'bbox': bbox,
                        'depth': depth,
                        'confidence': confidence
                    })
                else:
                    cv2.putText(output_frame, "2D ONLY", 
                               (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # No se detectó persona o pose
                stats = {
                    'confidence': 0,
                    'is_3d': False,
                    'bbox': None,
                    'frame_count': frame_count,
                    'fps': frame_fps,
                    'instant_fps': frame_fps,
                    'persons_detected': 0,
                    'poses_detected': 0,
                    'backend_used': args.backend,
                    'model_type': 'XS',
                    'total_time': frame_time * 1000,
                    'detection_time': 0,
                    'pose_time': 0,
                    'detected': False
                }
                
                output_frame = visualize_pose_2d(frame, [], stats)
                cv2.putText(output_frame, "No person detected", 
                           (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            
            # Agregar contador de frames y FPS
            cv2.putText(output_frame, f"Frame: {frame_count}/{total_frames}", 
                       (width - 200, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(output_frame, f"FPS: {frame_fps:.1f}", 
                       (width - 200, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            writer.write(output_frame)
            
            # MOSTRAR EL VIDEO EN TIEMPO REAL
            cv2.imshow('ConvNeXt 3D Pose Detection', output_frame)
            
            # Presionar 'q' para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("🛑 Saliendo por solicitud del usuario...")
                break
            
            # Progreso cada 25 frames
            if frame_count % 25 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed
                logger.info(f"📈 Frame {frame_count}/{total_frames} - "
                          f"FPS: {avg_fps:.1f} - "
                          f"3D Success: {poses_3d_success}/{poses_2d_success}")
        
        # Cleanup
        cap.release()
        writer.release()
        cv2.destroyAllWindows()  # Cerrar todas las ventanas de OpenCV
        
        # Guardar coordenadas 3D
        if poses_3d_data:
            # Guardar la última pose como ejemplo
            last_pose = poses_3d_data[-1]
            np.savez_compressed('output_3d_coords.npz', 
                              pose3d=last_pose['pose3d'], 
                              bbox=last_pose['bbox'], 
                              frame=last_pose['frame'],
                              depth=last_pose['depth'],
                              confidence=last_pose['confidence'])
            logger.info(f"📦 Coordenadas 3D guardadas: output_3d_coords.npz")
            
            # Guardar todas las poses para análisis
            all_poses = [p['pose3d'] for p in poses_3d_data]
            all_depths = [p['depth'] for p in poses_3d_data]
            np.savez_compressed('all_3d_poses.npz',
                              poses=all_poses,
                              depths=all_depths,
                              frames=[p['frame'] for p in poses_3d_data])
            logger.info(f"📊 Todas las poses 3D guardadas: all_3d_poses.npz")
        
        # Estadísticas finales
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        success_2d_rate = (poses_2d_success / frame_count) * 100
        success_3d_rate = (poses_3d_success / poses_2d_success) * 100 if poses_2d_success > 0 else 0
        
        logger.info("\\n📊 RESULTADOS FINALES 3D:")
        logger.info("=" * 50)
        logger.info(f"🎬 Frames procesados: {frame_count}")
        logger.info(f"✅ Poses 2D exitosas: {poses_2d_success} ({success_2d_rate:.1f}%)")
        logger.info(f"📦 Poses 3D exitosas: {poses_3d_success} ({success_3d_rate:.1f}%)")
        logger.info(f"⚡ FPS promedio: {avg_fps:.1f}")
        logger.info(f"⏱️ Tiempo total: {total_time:.1f}s")
        logger.info(f"💾 Video guardado: demo/output_3d_complete.mp4")
        
        # Estadísticas del wrapper TFLite
        if rootnet.use_tflite and rootnet.tflite_wrapper:
            tflite_stats = rootnet.tflite_wrapper.get_performance_stats()
            if tflite_stats:
                logger.info(f"\\n🚀 RENDIMIENTO TFLITE:")
                logger.info(f"📱 Modelo usado: {tflite_stats['model_variant']}")
                logger.info(f"⚡ Inferencia promedio: {tflite_stats['avg_inference_ms']:.2f} ± {tflite_stats['std_inference_ms']:.2f} ms")
                logger.info(f"🎯 Total inferencias: {tflite_stats['total_inferences']}")
                logger.info(f"📊 Rango: {tflite_stats['min_inference_ms']:.1f} - {tflite_stats['max_inference_ms']:.1f} ms")
        
        if poses_3d_data:
            depths = [p['depth'] for p in poses_3d_data]
            logger.info(f"📏 Profundidad promedio: {np.mean(depths):.1f}mm")
            logger.info(f"📏 Rango profundidad: {np.min(depths):.1f} - {np.max(depths):.1f}mm")
        
        logger.info("\\n🎉 TEST 3D COMPLETADO EXITOSAMENTE!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal con parser de argumentos completo"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ConvNeXt Ultra Performance - Versión Corregida')
    
    # Presets disponibles
    presets = {
        'ultra_fast_30fps': 'Ultra rápido - 30+ FPS (solo 2D)',
        'ultra_fast_30fps_3d': 'Ultra rápido - 30+ FPS con coordenadas 3D',
        'balanced_25fps': 'Balance - 25+ FPS (solo 2D)', 
        'balanced_25fps_3d': 'Balance - 25+ FPS con coordenadas 3D',
        'quality_20fps': 'Calidad - 20+ FPS (solo 2D)',
        'quality_20fps_3d': 'Calidad - 20+ FPS con coordenadas 3D'
    }
    
    parser.add_argument('--preset', choices=list(presets.keys()), 
                       default='ultra_fast_30fps', help='Preset de rendimiento')
    parser.add_argument('--backend', choices=['onnx', 'pytorch', 'tflite'], 
                       default='onnx', help='Backend de inferencia')
    parser.add_argument('--model', choices=['XS', 'S'], 
                       default='XS', help='Modelo ConvNeXt')
    parser.add_argument('--input', type=str, default='demo/Personas caminando en la calle.mp4',
                       help='Entrada: webcam (0) o archivo de video')
    parser.add_argument('--output_video', type=str, default=None,
                       help='Archivo de salida de video (.mp4)')
    parser.add_argument('--output_npz', type=str, default=None,
                       help='Archivo de salida de coordenadas 3D (.npz)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Ejecutar benchmark de rendimiento')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duración del benchmark (segundos)')
    parser.add_argument('--rootnet_path', type=str, default='/home/user/convnextpose_esteban/3DMPPE_ROOTNET_RELEASE',
                       help='Ruta al repo RootNet')
    parser.add_argument('--rootnet_ckpt', type=str, default='demo/snapshot_18.pth.tar',
                       help='Checkpoint RootNet')
    parser.add_argument('--use_tflite', action='store_true', default=True,
                       help='Usar TFLite optimizado para RootNet (default: True)')
    parser.add_argument('--tflite_variant', choices=['default', 'size', 'latency'], 
                       default='size', help='Variante del modelo TFLite RootNet')
    parser.add_argument('--rootnet_backend', choices=['tflite', 'pytorch', 'auto'], 
                       default='auto', help='Backend específico para RootNet')
    
    args = parser.parse_args()
    
    # Información del preset seleccionado
    output_3d = args.preset.endswith('_3d')
    
    logger.info("🚀 CONVNEXT ULTRA PERFORMANCE - VERSIÓN TFLITE OPTIMIZADA")
    logger.info("=" * 60)
    logger.info(f"🎯 Preset: {args.preset}")
    logger.info(f"📝 Descripción: {presets[args.preset]}")
    logger.info(f"🔧 Backend ConvNeXt: {args.backend}")
    logger.info(f"🎪 Modelo: {args.model}")
    logger.info(f"📦 Salida 3D: {'✅ Activada' if output_3d else '❌ Solo 2D'}")
    logger.info(f"📁 Entrada: {args.input}")
    
    # Información específica de RootNet
    if output_3d:
        logger.info(f"🚀 RootNet Backend: {args.rootnet_backend}")
        logger.info(f"⚡ TFLite Variant: {args.tflite_variant}")
        logger.info(f"🎛️ TFLite Enabled: {'✅' if args.use_tflite else '❌'}")
    
    if args.benchmark:
        logger.info(f"🧪 Modo benchmark: {args.duration} segundos")
        test_3d_complete(args)
    else:
        # Ejecutar procesamiento normal
        if output_3d:
            logger.info("🦴 Iniciando procesamiento con 3D...")
            test_3d_complete(args)
        else:
            logger.info("🎬 Iniciando procesamiento solo 2D...")
            test_3d_complete(args)

if __name__ == "__main__":
    main()
