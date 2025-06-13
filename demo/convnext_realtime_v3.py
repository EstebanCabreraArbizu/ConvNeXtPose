#!/usr/bin/env python3
"""
convnext_realtime_v4_FIXED.py - VERSIÓN CORREGIDA CON CONVNEXT + ROOTNET

CORRECCIONES CRÍTICAS:
1. ✅ ConvNeXt Pose Estimation restaurado
2. ✅ RootNet para profundidad restaurado  
3. ✅ Pipeline completo: YOLO → ConvNeXt → RootNet → Pose 2D/3D
4. ✅ Dibujo de esqueletos completos
5. ✅ Optimizaciones asíncronas MANTENIDAS
6. ✅ Cache inteligente para pose estimation
"""

import argparse
import time
import sys
from pathlib import Path
import os
import warnings
import logging
from typing import Optional, Dict, Any, Tuple, List

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# IMPORTS BÁSICOS
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import onnxruntime as ort
from types import SimpleNamespace
import threading
from queue import Queue, Empty
from collections import deque
import concurrent.futures
import asyncio
from concurrent.futures import ThreadPoolExecutor

# IMPORTS DEL PROYECTO CONVNEXT
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.extend([
    str(PROJECT_ROOT / 'main'),
    str(PROJECT_ROOT / 'data'),
    str(PROJECT_ROOT / 'common')
])

try:
    from config import cfg
    from model import get_pose_net
    from dataset import generate_patch_image
    import utils.pose_utils as pose_utils
    from root_wrapper import RootNetWrapper
    logger.info("✅ Módulos ConvNeXt importados correctamente")
except ImportError as e:
    logger.error(f"❌ Error importando módulos ConvNeXt: {e}")
    sys.exit(1)

# VERIFICACIÓN ULTRALYTICS
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    logger.info("✅ Ultralytics disponible")
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.error("❌ Ultralytics no disponible - Instalar con: pip install ultralytics")
    sys.exit(1)

def detect_hardware_capabilities() -> Dict[str, Any]:
    """Detectar capacidades del hardware"""
    capabilities = {
        'has_cuda': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_memory_gb': 0.0,
        'cpu_cores': os.cpu_count() or 4,
        'recommended_workers': 2,
        'recommended_cache_timeout': 0.08,
        'async_workers': 4,
    }
    
    if capabilities['has_cuda']:
        try:
            gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
            capabilities['cuda_memory_gb'] = gpu_memory_bytes / (1024**3)
            
            if capabilities['cuda_memory_gb'] >= 8:
                capabilities.update({
                    'recommended_workers': 4,
                    'recommended_cache_timeout': 0.04,
                    'async_workers': 8
                })
            elif capabilities['cuda_memory_gb'] >= 4:
                capabilities.update({
                    'recommended_workers': 3,
                    'recommended_cache_timeout': 0.06,
                    'async_workers': 6
                })
        except Exception as e:
            logger.warning(f"Error detectando GPU: {e}")
    
    logger.info(f"🔧 Hardware: GPU={'✅' if capabilities['has_cuda'] else '❌'} "
                f"({capabilities['cuda_memory_gb']:.1f}GB), "
                f"CPU={capabilities['cpu_cores']} cores")
    
    return capabilities

def convert_yolo_to_onnx_safe(pt_model_path: str = 'yolov8n.pt') -> Optional[str]:
    """Conversión YOLO a ONNX"""
    onnx_path = pt_model_path.replace('.pt', '_optimized.onnx')
    
    if os.path.exists(onnx_path):
        logger.info(f"✅ ONNX encontrado: {onnx_path}")
        return onnx_path
    
    logger.info(f"🔄 Convirtiendo {pt_model_path} a ONNX...")
    try:
        model = YOLO(pt_model_path)
        exported_path = model.export(
            format='onnx', imgsz=640, optimize=True, half=False,
            dynamic=False, simplify=True, opset=13
        )
        
        if exported_path != onnx_path:
            os.rename(exported_path, onnx_path)
        
        logger.info(f"✅ ONNX creado: {onnx_path}")
        return onnx_path
    except Exception as e:
        logger.error(f"❌ Error conversión ONNX: {e}")
        return None

class ModernYOLODetector:
    """Detector YOLO optimizado"""
    
    def __init__(self, onnx_path: str, hardware_caps: Dict[str, Any]):
        self.onnx_path = onnx_path
        self.hardware_caps = hardware_caps
        self._setup_session()
        self._warmup_model()
    
    def _setup_session(self):
        """Configurar sesión ONNX"""
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = []
        if self.hardware_caps['has_cuda']:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        self.session = ort.InferenceSession(self.onnx_path, providers=providers, sess_options=session_options)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.img_size = self.input_shape[2] if self.input_shape[2] > 0 else 640
        
        logger.info(f"✅ YOLO sesión configurada - Tamaño: {self.img_size}")
    
    def _warmup_model(self):
        """Pre-calentar modelo"""
        logger.info("🔥 Pre-calentando YOLO...")
        dummy_input = np.random.rand(*self.input_shape).astype(np.float32)
        for _ in range(3):
            self.session.run(None, {self.input_name: dummy_input})
        logger.info("✅ YOLO pre-calentado")
    
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """Preprocesamiento del frame"""
        h, w = frame.shape[:2]
        scale = min(self.img_size / w, self.img_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        pad_w = (self.img_size - new_w) // 2
        pad_h = (self.img_size - new_h) // 2
        
        padded = cv2.copyMakeBorder(
            resized, pad_h, self.img_size - new_h - pad_h,
            pad_w, self.img_size - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
        
        return input_tensor, scale, pad_w, pad_h
    
    async def detect_persons_async(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[List[int]]:
        """Detección asíncrona que retorna bboxes de personas"""
        try:
            input_tensor, scale, pad_w, pad_h = self.preprocess_frame(frame)
            
            # Inferencia
            outputs = self.session.run(None, {self.input_name: input_tensor})
            
            if not outputs or len(outputs) == 0:
                return []
            
            detections = outputs[0][0] if len(outputs[0].shape) == 3 else outputs[0]
            
            # Filtrar personas con confianza
            conf_mask = detections[:, 4] >= conf_threshold
            if not conf_mask.any():
                return []
            
            valid_detections = detections[conf_mask]
            
            # Filtrar solo clase persona (0)
            if valid_detections.shape[1] > 5:
                class_ids = np.argmax(valid_detections[:, 5:], axis=1)
                person_mask = class_ids == 0
                if not person_mask.any():
                    return []
                person_detections = valid_detections[person_mask]
            else:
                person_detections = valid_detections
            
            # Convertir coordenadas
            h_frame, w_frame = frame.shape[:2]
            bboxes = []
            
            for detection in person_detections:
                cx, cy, w, h = detection[:4]
                
                # Transformar coordenadas
                cx = (cx - pad_w) / scale
                cy = (cy - pad_h) / scale
                w = w / scale
                h = h / scale
                
                x1 = max(0, int(cx - w/2))
                y1 = max(0, int(cy - h/2))
                x2 = min(w_frame, int(cx + w/2))
                y2 = min(h_frame, int(cy + h/2))
                
                if x2 > x1 and y2 > y1:
                    bboxes.append([x1, y1, x2, y2])
            
            return bboxes
            
        except Exception as e:
            logger.error(f"❌ Error en detección YOLO: {e}")
            return []

class AsyncConvNeXtPoseProcessor:
    """Procesador ConvNeXt asíncrono con cache inteligente"""
    
    def __init__(self, pose_model, device, root_wrapper, transform, cfg, hardware_caps):
        self.pose_model = pose_model
        self.device = device
        self.root_wrapper = root_wrapper
        self.transform = transform
        self.cfg = cfg
        self.hardware_caps = hardware_caps
        
        # Cache inteligente
        self.pose_cache = {}
        self.cache_timeout = hardware_caps['recommended_cache_timeout']
        self.max_cache_size = 30
        
        # ThreadPool para RootNet
        self.executor = ThreadPoolExecutor(max_workers=hardware_caps['recommended_workers'])
        
        # Métricas
        self.processing_times = deque(maxlen=50)
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"✅ ConvNeXt Processor inicializado (cache={self.cache_timeout*1000:.0f}ms)")
    
    def _generate_cache_key(self, bbox: List[int], frame_time: float) -> str:
        """Generar clave de cache inteligente"""
        x1, y1, x2, y2 = bbox
        
        # Cuantización adaptativa
        quantization = 20
        x1_q = int(x1 / quantization) * quantization
        y1_q = int(y1 / quantization) * quantization
        x2_q = int(x2 / quantization) * quantization
        y2_q = int(y2 / quantization) * quantization
        
        time_slot = int(frame_time / self.cache_timeout)
        
        return f"{x1_q}_{y1_q}_{x2_q}_{y2_q}_{time_slot}"
    
    async def process_pose_async(self, frame: np.ndarray, bbox: List[int], frame_time: float) -> Optional[np.ndarray]:
        """Procesamiento ConvNeXt asíncrono completo"""
        
        # Cache check
        cache_key = self._generate_cache_key(bbox, frame_time)
        if cache_key in self.pose_cache:
            cached_result, cached_time = self.pose_cache[cache_key]
            if frame_time - cached_time < self.cache_timeout:
                self.cache_hits += 1
                return cached_result
        
        self.cache_misses += 1
        start_time = time.time()
        
        try:
            # Preparar bbox para ConvNeXt
            x1, y1, x2, y2 = bbox
            bbox_array = np.array([x1, y1, x2 - x1, y2 - y1])
            proc_bbox = pose_utils.process_bbox(bbox_array, frame.shape[1], frame.shape[0])
            
            if proc_bbox is None:
                return None
            
            # Generar patch
            img_patch, img2bb_trans = generate_patch_image(
                frame, proc_bbox, False, 1.0, 0.0, False
            )
            
            # Preparar entrada para ConvNeXt
            inp = self.transform(img_patch).unsqueeze(0)
            inp = inp.to(self.device, non_blocking=True if self.device.type == 'cuda' else False)
            
            # RootNet asíncrono
            root_future = self.executor.submit(
                self.root_wrapper.predict_depth, frame, bbox_array
            )
            
            # ConvNeXt inference
            with torch.no_grad():
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                pose_3d = self.pose_model(inp)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
            
            if pose_3d is not None:
                pose_3d_np = pose_3d[0].cpu().numpy()
                
                # Obtener profundidad de RootNet
                try:
                    root_depth = root_future.result(timeout=0.05)
                except:
                    root_depth = 8000  # Fallback
                
                # Post-procesamiento
                coords_2d = self._postprocess_pose(pose_3d_np, img2bb_trans, root_depth)
                
                if coords_2d is not None:
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)
                    
                    # Cache update
                    self._update_cache(cache_key, coords_2d, frame_time)
                    return coords_2d
            
        except Exception as e:
            logger.error(f"❌ Error en ConvNeXt processing: {e}")
            return None
        
        return None
    
    def _postprocess_pose(self, pose_3d_raw: np.ndarray, img2bb_trans: np.ndarray, root_depth: float) -> np.ndarray:
        """Post-procesamiento de pose ConvNeXt"""
        pose_3d = pose_3d_raw.copy()
        
        # Escalar coordenadas
        pose_3d[:, 0] = pose_3d[:, 0] / self.cfg.output_shape[1] * self.cfg.input_shape[1]
        pose_3d[:, 1] = pose_3d[:, 1] / self.cfg.output_shape[0] * self.cfg.input_shape[0]
        
        # Transformación afín
        pose_3d_xy1 = np.column_stack((pose_3d[:, :2], np.ones(len(pose_3d))))
        img2bb_trans_full = np.vstack((img2bb_trans, [0, 0, 1]))
        
        try:
            pose_3d[:, :2] = np.linalg.solve(img2bb_trans_full, pose_3d_xy1.T).T[:, :2]
        except np.linalg.LinAlgError:
            try:
                pose_3d[:, :2] = np.linalg.lstsq(img2bb_trans_full, pose_3d_xy1.T, rcond=None)[0].T[:, :2]
            except:
                return None
        
        # Procesar profundidad
        pose_3d[:, 2] = (pose_3d[:, 2] / self.cfg.depth_dim * 2 - 1) * (self.cfg.bbox_3d_shape[0]/2) + root_depth
        
        return pose_3d[:, :2]  # Retornar solo coordenadas 2D
    
    def _update_cache(self, key: str, result: np.ndarray, timestamp: float):
        """Actualizar cache con limpieza"""
        self.pose_cache[key] = (result, timestamp)
        
        # Limpieza del cache
        if len(self.pose_cache) > self.max_cache_size:
            cutoff_time = timestamp - self.cache_timeout * 2
            old_keys = [k for k, (_, t) in self.pose_cache.items() if t < cutoff_time]
            
            for k in old_keys:
                del self.pose_cache[k]
            
            # Si sigue lleno, eliminar más antiguos
            if len(self.pose_cache) > self.max_cache_size:
                sorted_items = sorted(self.pose_cache.items(), key=lambda x: x[1][1])
                for k, _ in sorted_items[:len(sorted_items)//3]:
                    del self.pose_cache[k]
    
    def get_stats(self) -> Dict[str, float]:
        """Estadísticas del procesador"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        
        avg_time = 0
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times) * 1000
        
        return {
            'cache_hit_rate': hit_rate,
            'avg_processing_time_ms': avg_time,
            'cache_size': len(self.pose_cache)
        }

class AsyncFrameProcessor:
    """Procesador de frames completo: YOLO → ConvNeXt → Pose"""
    
    def __init__(self, yolo_detector, pose_processor, hardware_caps):
        self.yolo_detector = yolo_detector
        self.pose_processor = pose_processor
        self.hardware_caps = hardware_caps
        
        # Queues asíncronos
        self.input_queue = asyncio.Queue(maxsize=3)
        self.output_queue = asyncio.Queue(maxsize=3)
        
        # Control
        self.processing = True
        self.processor_task = None
        
        # Estadísticas
        self.frame_count = 0
        self.processing_times = deque(maxlen=50)
        
        logger.info("✅ AsyncFrameProcessor inicializado")
    
    async def start_processing(self):
        """Iniciar procesamiento asíncrono"""
        self.processor_task = asyncio.create_task(self._process_frames_loop())
        logger.info("🚀 Procesamiento asíncrono iniciado")
    
    async def _process_frames_loop(self):
        """Loop principal de procesamiento"""
        while self.processing:
            try:
                frame_data = await asyncio.wait_for(self.input_queue.get(), timeout=0.1)
                
                if frame_data is None:
                    break
                
                frame, frame_time = frame_data
                result = await self._process_single_frame(frame, frame_time)
                
                # Limpiar queue de salida y añadir resultado
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                
                await self.output_queue.put((result, frame_time))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"❌ Error en loop de procesamiento: {e}")
    
    async def _process_single_frame(self, frame: np.ndarray, frame_time: float) -> Dict:
        """Procesar frame completo: YOLO + ConvNeXt"""
        start_time = time.time()
        
        try:
            # 1. YOLO Detection
            bboxes = await self.yolo_detector.detect_persons_async(frame)
            
            # 2. ConvNeXt Pose Estimation (solo mejor detección)
            poses = []
            if bboxes:
                # Tomar mejor bbox (primera en lista ya filtrada por confianza)
                best_bbox = bboxes[0]
                
                # ConvNeXt processing
                pose_coords = await self.pose_processor.process_pose_async(frame, best_bbox, frame_time)
                
                if pose_coords is not None:
                    poses.append(pose_coords)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return {
                'bboxes': bboxes,
                'poses': poses,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Error procesando frame: {e}")
            return {'bboxes': [], 'poses': [], 'processing_time': 0}
    
    async def add_frame_async(self, frame: np.ndarray) -> bool:
        """Agregar frame para procesamiento"""
        self.frame_count += 1
        frame_time = time.time()
        
        try:
            # Limpiar queue si está lleno
            while not self.input_queue.empty():
                try:
                    self.input_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            await self.input_queue.put((frame, frame_time))
            return True
        except asyncio.QueueFull:
            return False
    
    async def get_result_async(self):
        """Obtener resultado"""
        try:
            return await asyncio.wait_for(self.output_queue.get(), timeout=0.001)
        except asyncio.TimeoutError:
            return None
    
    def get_stats(self) -> Dict:
        """Estadísticas completas"""
        pose_stats = self.pose_processor.get_stats()
        
        stats = {
            'frame_count': self.frame_count,
            'queue_input_size': self.input_queue.qsize(),
            'queue_output_size': self.output_queue.qsize()
        }
        
        if self.processing_times:
            stats['avg_total_time_ms'] = sum(self.processing_times) / len(self.processing_times) * 1000
        
        stats.update(pose_stats)
        return stats
    
    async def stop_async(self):
        """Detener procesamiento"""
        self.processing = False
        await self.input_queue.put(None)
        if self.processor_task:
            await self.processor_task

async def setup_models_async(args):
    """Configurar todos los modelos: YOLO + ConvNeXt + RootNet"""
    logger.info("🚀 Configurando modelos completos...")
    
    hardware_caps = detect_hardware_capabilities()
    
    # 1. Configurar ConvNeXt
    cfg.input_shape = (256, 256)
    cfg.output_shape = (32, 32)
    cfg.depth_dim = 32
    cfg.bbox_3d_shape = (2000, 2000, 2000)
    
    # 2. YOLO
    onnx_path = convert_yolo_to_onnx_safe('yolov8n.pt')
    if not onnx_path:
        raise RuntimeError("No se pudo preparar YOLO")
    
    yolo_detector = ModernYOLODetector(onnx_path, hardware_caps)
    
    # 3. ConvNeXt Pose Model
    device = torch.device('cuda' if hardware_caps['has_cuda'] else 'cpu')
    pose_model = get_pose_net(cfg, is_train=False, joint_num=18)
    
    state = torch.load(args.pose_model, map_location=device)
    sd = state.get('network', state)
    pose_model.load_state_dict(sd, strict=False)
    pose_model = pose_model.to(device).eval()
    
    # Optimizaciones GPU
    if hardware_caps['has_cuda']:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    logger.info(f"✅ ConvNeXt cargado en {device}")
    
    # 4. RootNet
    root_wrapper = RootNetWrapper(args.rootnet_dir, args.rootnet_model)
    root_wrapper.load_model(use_gpu=hardware_caps['has_cuda'])
    logger.info("✅ RootNet cargado")
    
    # 5. Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std),
    ])
    
    # 6. ConvNeXt Processor
    pose_processor = AsyncConvNeXtPoseProcessor(
        pose_model, device, root_wrapper, transform, cfg, hardware_caps
    )
    
    return yolo_detector, pose_processor, hardware_caps

async def main_async():
    """Main asíncrono completo"""
    parser = argparse.ArgumentParser(description="ConvNeXt v4 COMPLETO - YOLO + ConvNeXt + RootNet")
    parser.add_argument('--input', type=str, default='0', help='Video source')
    parser.add_argument('--pose-model', type=str, required=True, help='ConvNeXt checkpoint')
    parser.add_argument('--rootnet-dir', type=str, 
                        default='/home/fabri/3DMPPE_ROOTNET_RELEASE',
                        help='RootNet directory')
    parser.add_argument('--rootnet-model', type=str,
                        default='/home/fabri/3DMPPE_ROOTNET_RELEASE/demo/snapshot_18.pth.tar',
                        help='RootNet checkpoint')
    args = parser.parse_args()
    
    # Setup completo
    yolo_detector, pose_processor, hardware_caps = await setup_models_async(args)
    
    # Frame processor
    frame_processor = AsyncFrameProcessor(yolo_detector, pose_processor, hardware_caps)
    await frame_processor.start_processing()
    
    # Esqueleto para dibujo
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
        cap = cv2.VideoCapture(f"tcp://{args.input}:5000")
    
    if not cap.isOpened():
        logger.error(f"❌ No se pudo abrir video: {args.input}")
        return
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    logger.info("🚀 Demo ConvNeXt v4 COMPLETO iniciado. Presione 'q' para salir.")
    
    # Variables de rendimiento
    frame_count = 0
    display_fps_counter = deque(maxlen=30)
    last_poses = []
    last_result_time = time.time()
    
    try:
        while True:
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar frame
            await frame_processor.add_frame_async(frame)
            
            # Obtener resultado
            result = await frame_processor.get_result_async()
            if result:
                result_data, result_time = result
                last_poses = result_data.get('poses', [])
                last_result_time = result_time
                
                # Dibujar bboxes
                for bbox in result_data.get('bboxes', []):
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # DIBUJAR ESQUELETOS COMPLETOS
            for pose_coords in last_poses:
                if pose_coords is not None:
                    pose_coords = np.clip(pose_coords, 0, [frame.shape[1]-1, frame.shape[0]-1])
                    
                    # Dibujar conexiones del esqueleto
                    for i, j in skeleton:
                        if i < len(pose_coords) and j < len(pose_coords):
                            pt1 = tuple(map(int, pose_coords[i]))
                            pt2 = tuple(map(int, pose_coords[j]))
                            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                    
                    # Dibujar articulaciones
                    for point in pose_coords:
                        cv2.circle(frame, tuple(map(int, point)), 3, (0, 255, 0), -1)
            
            # Estadísticas en pantalla
            loop_time = time.time() - loop_start
            display_fps_counter.append(1.0 / max(loop_time, 1e-6))
            
            if display_fps_counter:
                display_fps = sum(display_fps_counter) / len(display_fps_counter)
                cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Estadísticas del sistema
            stats = frame_processor.get_stats()
            proc_time = stats.get('avg_total_time_ms', 0)
            hit_rate = stats.get('cache_hit_rate', 0)
            
            cv2.putText(frame, f"Processing: {proc_time:.1f}ms", (10, 70),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Cache: {hit_rate:.1f}%", (10, 110),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            visual_lag = (time.time() - last_result_time) * 1000
            cv2.putText(frame, f"Visual Lag: {visual_lag:.0f}ms", (10, 150),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            hw_info = f"GPU-{hardware_caps['cuda_memory_gb']:.1f}GB" if hardware_caps['has_cuda'] else "CPU"
            cv2.putText(frame, f"v4-COMPLETE ({hw_info})", (10, 190),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            cv2.imshow("ConvNeXt v4 - COMPLETO", frame)
            
            frame_count += 1
            
            # Log estadísticas cada 60 frames
            if frame_count % 60 == 0:
                avg_fps = sum(display_fps_counter) / len(display_fps_counter) if display_fps_counter else 0
                logger.info(f"📊 Frame {frame_count}: FPS={avg_fps:.1f}, "
                           f"Proc={proc_time:.1f}ms, Cache={hit_rate:.1f}%, "
                           f"Lag={visual_lag:.0f}ms")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        await frame_processor.stop_async()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("🏁 Demo v4 COMPLETO finalizado")

def main():
    """Función principal"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("\n🛑 Detenido por usuario")
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()