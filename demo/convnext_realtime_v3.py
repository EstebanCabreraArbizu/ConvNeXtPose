#!/usr/bin/env python3
"""
convnext_realtime_FINAL.py - VERSIÓN DEFINITIVA BASADA EN ANÁLISIS COMPLETO

OPTIMIZACIONES FINALES:
1. Híbrido v3 como base (mejor balance demostrado) ✅
2. Optimizaciones GPU específicas ✅
3. Cache más inteligente ✅
4. Métricas mejoradas ✅
5. Configuración adaptativa según hardware ✅
"""

import argparse
import time
import sys
from pathlib import Path
import os
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

# IMPORTS CORREGIDOS
try:
    from deepsparse.pipeline import Pipeline
    DEEPSPARSE_AVAILABLE = True
except ImportError:
    print("⚠️ DeepSparse no disponible, usando fallback a Ultralytics")
    DEEPSPARSE_AVAILABLE = False

# Importar módulos del proyecto
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.extend([
    str(PROJECT_ROOT / 'main'),
    str(PROJECT_ROOT / 'data'),
    str(PROJECT_ROOT / 'common')
])

from config import cfg
from model import get_pose_net
from dataset import generate_patch_image
import utils.pose_utils as pose_utils
from root_wrapper import RootNetWrapper

def detect_hardware_capabilities():
    """Detectar capacidades del hardware para optimización automática"""
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
            
            # Configuración basada en GPU
            if gpu_memory >= 8:  # RTX 3070+
                capabilities.update({
                    'recommended_workers': 2,
                    'recommended_cache_timeout': 0.08,
                    'recommended_frame_skip': 1
                })
            elif gpu_memory >= 4:  # GTX 1660+
                capabilities.update({
                    'recommended_workers': 1,
                    'recommended_cache_timeout': 0.10,
                    'recommended_frame_skip': 2
                })
        except:
            pass
    else:
        # Configuración CPU conservadora
        capabilities.update({
            'recommended_workers': 1,
            'recommended_cache_timeout': 0.15,
            'recommended_frame_skip': 3
        })
    
    print(f"🔧 CONFIGURACIÓN AUTOMÁTICA DETECTADA:")
    print(f"   GPU: {'✅' if capabilities['has_cuda'] else '❌'} ({capabilities['cuda_memory_gb']:.1f}GB)")
    print(f"   CPU Cores: {capabilities['cpu_cores']}")
    print(f"   Workers: {capabilities['recommended_workers']}")
    print(f"   Cache timeout: {capabilities['recommended_cache_timeout']*1000:.0f}ms")
    print(f"   Frame skip: 1/{capabilities['recommended_frame_skip']}")
    
    return capabilities

def convert_yolo_to_onnx_optimized(pt_model_path='yolov8n.pt', 
                                   conf_thresh=0.3, iou_thresh=0.45, img_size=640):
    """Convierte YOLO a ONNX con optimizaciones específicas"""
    base_name = pt_model_path.replace('.pt', '')
    onnx_path = f"{base_name}_optimized_conf{conf_thresh}_iou{iou_thresh}.onnx"
    
    if os.path.exists(onnx_path):
        print(f"✅ ONNX optimizado existente: {onnx_path}")
        return onnx_path
    
    print(f"🔄 Convirtiendo {pt_model_path} a ONNX optimizado...")
    try:
        from ultralytics import YOLO
        model = YOLO(pt_model_path)
        
        exported_path = model.export(
            format='onnx', 
            imgsz=img_size, 
            optimize=True, 
            half=False,  # Mantener FP32 para compatibilidad
            dynamic=False, 
            simplify=True, 
            opset=13,  # Opset más reciente
            nms=True,
            conf=conf_thresh, 
            iou=iou_thresh, 
            max_det=100  # Reducido para velocidad
        )
        
        if exported_path != onnx_path:
            os.rename(exported_path, onnx_path)
        
        print(f"✅ ONNX optimizado creado: {onnx_path}")
        return onnx_path
        
    except Exception as e:
        print(f"❌ Error en conversión optimizada: {e}")
        raise

class AdaptiveYOLODetector:
    """Detector YOLO adaptativo según hardware"""
    
    def __init__(self, onnx_path, hardware_caps):
        self.onnx_path = onnx_path
        self.hardware_caps = hardware_caps
        self._setup_session()
        self.warmup()
    
    def _setup_session(self):
        """Configurar sesión adaptativa"""
        providers = []
        session_options = ort.SessionOptions()
        
        if self.hardware_caps['has_cuda']:
            # Configuración GPU optimizada
            cuda_provider = ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested' if self.hardware_caps['cuda_memory_gb'] < 6 else 'kNextPowerOfTwo',
                'gpu_mem_limit': int(self.hardware_caps['cuda_memory_gb'] * 0.6 * 1024**3),  # 60% de GPU
                'cudnn_conv_algo_search': 'HEURISTIC',
                'do_copy_in_default_stream': True,
            })
            providers.append(cuda_provider)
            
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            session_options.inter_op_num_threads = min(2, self.hardware_caps['cpu_cores'] // 2)
            session_options.intra_op_num_threads = min(4, self.hardware_caps['cpu_cores'])
            
            print("🚀 YOLO configurado para GPU de alto rendimiento")
        else:
            # Configuración CPU optimizada
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.inter_op_num_threads = 1
            session_options.intra_op_num_threads = min(4, self.hardware_caps['cpu_cores'])
            
            print("🚀 YOLO configurado para CPU optimizado")
        
        providers.append('CPUExecutionProvider')
        
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(self.onnx_path, providers=providers, sess_options=session_options)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.img_size = self.input_shape[2]
    
    def warmup(self):
        """Pre-calentar con número de iteraciones adaptativo"""
        print("🔥 Pre-calentando YOLO adaptativo...")
        dummy_input = np.random.rand(*self.input_shape).astype(np.float32)
        
        warmup_iterations = 5 if self.hardware_caps['has_cuda'] else 3
        for _ in range(warmup_iterations):
            self.session.run(self.output_names, {self.input_name: dummy_input})
        print("✅ YOLO adaptativo pre-calentado")
    
    def preprocess_frame_adaptive(self, frame):
        """Preprocesamiento adaptativo según hardware"""
        h, w = frame.shape[:2]
        scale = min(self.img_size / w, self.img_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Interpolación adaptativa
        interpolation = cv2.INTER_LINEAR if self.hardware_caps['has_cuda'] else cv2.INTER_NEAREST
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
        
        # Padding optimizado
        pad_w = (self.img_size - new_w) // 2
        pad_h = (self.img_size - new_h) // 2
        
        padded = cv2.copyMakeBorder(
            resized, pad_h, self.img_size - new_h - pad_h, 
            pad_w, self.img_size - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Conversión optimizada
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
        
        return input_tensor, scale, pad_w, pad_h
    
    def predict_persons(self, frame, conf_threshold=0.3):
        """Detección adaptativa"""
        try:
            input_tensor, scale, pad_w, pad_h = self.preprocess_frame_adaptive(frame)
            
            # Inferencia
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # Post-procesamiento optimizado
            if len(outputs) == 0 or outputs[0].size == 0:
                return SimpleNamespace(boxes=[[]], scores=[[]], labels=[[]])
            
            detections = outputs[0][0]
            
            # Filtrado vectorizado
            person_mask = (detections[:, 5] == 0) & (detections[:, 4] >= conf_threshold)
            person_detections = detections[person_mask]
            
            if len(person_detections) == 0:
                return SimpleNamespace(boxes=[[]], scores=[[]], labels=[[]])
            
            # Convertir coordenadas
            boxes = person_detections[:, :4].copy()
            scores = person_detections[:, 4]
            
            # Transformación vectorizada
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale
            
            # Clipping
            h, w = frame.shape[:2]
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
            
            return SimpleNamespace(
                boxes=[boxes.astype(int)],
                scores=[scores],
                labels=[np.zeros(len(scores), dtype=int)]
            )
            
        except Exception as e:
            print(f"⚠️ Error en YOLO adaptativo: {e}")
            return SimpleNamespace(boxes=[[]], scores=[[]], labels=[[]])

class IntelligentPoseProcessor:
    """Procesador de pose inteligente con optimizaciones adaptativas"""
    
    def __init__(self, pose_model, device, root_wrapper, transform, cfg, hardware_caps):
        self.pose_model = pose_model
        self.device = device
        self.root_wrapper = root_wrapper
        self.transform = transform
        self.cfg = cfg
        self.hardware_caps = hardware_caps
        
        # Cache adaptativo
        self.bbox_cache = {}
        self.cache_timeout = hardware_caps['recommended_cache_timeout']
        self.max_cache_size = 50 if hardware_caps['has_cuda'] else 30
        
        # ThreadPool adaptativo
        max_workers = hardware_caps['recommended_workers']
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Métricas de rendimiento
        self.processing_times = deque(maxlen=20)
        self.cache_hits = 0
        self.cache_misses = 0
        
        print(f"✅ IntelligentPoseProcessor inicializado (workers={max_workers}, cache={self.cache_timeout*1000:.0f}ms)")
    
    def _intelligent_cache_key(self, bbox, frame_time):
        """Cache key inteligente con cuantización adaptativa"""
        x1, y1, x2, y2 = bbox
        
        # Cuantización adaptativa según rendimiento
        if len(self.processing_times) > 5:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            if avg_time > 0.4:  # Si es lento, cache más agresivo
                quantization = 25
            elif avg_time > 0.2:  # Rendimiento medio
                quantization = 20
            else:  # Rendimiento bueno
                quantization = 15
        else:
            quantization = 20  # Default
        
        x1_q, y1_q = int(x1 / quantization) * quantization, int(y1 / quantization) * quantization
        x2_q, y2_q = int(x2 / quantization) * quantization, int(y2 / quantization) * quantization
        time_slot = int(frame_time / self.cache_timeout)
        
        return f"{x1_q}_{y1_q}_{x2_q}_{y2_q}_{time_slot}"
    
    def process_pose_intelligent(self, frame, bbox, frame_time):
        """Procesamiento inteligente con cache adaptativo"""
        
        # 1. Cache check inteligente
        cache_key = self._intelligent_cache_key(bbox, frame_time)
        if cache_key in self.bbox_cache:
            cached_result, cached_time = self.bbox_cache[cache_key]
            if frame_time - cached_time < self.cache_timeout:
                self.cache_hits += 1
                return cached_result
        
        self.cache_misses += 1
        
        # 2. Preparar bbox
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        bbox_array = np.array([x1, y1, x2 - x1, y2 - y1])
        proc_bbox = pose_utils.process_bbox(bbox_array, frame.shape[1], frame.shape[0])
        
        if proc_bbox is None:
            return None
        
        try:
            start_time = time.time()
            
            # 3. Generar patch
            img_patch, img2bb_trans = generate_patch_image(
                frame, proc_bbox, False, 1.0, 0.0, False
            )
            
            # 4. Preparar entrada con optimizaciones GPU
            inp = self.transform(img_patch).unsqueeze(0)
            if self.device.type == 'cuda':
                inp = inp.to(self.device, non_blocking=True)
            else:
                inp = inp.to(self.device)
            
            # 5. RootNet con timeout adaptativo
            root_timeout = 0.05 if self.hardware_caps['has_cuda'] else 0.03
            root_future = self.executor.submit(
                self.root_wrapper.predict_depth, frame, bbox_array
            )
            
            # 6. Inferencia ConvNeXt optimizada
            with torch.no_grad():
                if self.device.type == 'cuda':
                    # Stream paralelo para GPU
                    torch.cuda.synchronize()
                
                pose_3d = self.pose_model(inp)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
            if pose_3d is not None:
                pose_3d_np = pose_3d[0].cpu().numpy()
                
                # 7. Obtener profundidad con timeout adaptativo
                try:
                    root_depth = root_future.result(timeout=root_timeout)
                except:
                    root_depth = 8000  # Fallback
                
                # 8. Post-procesamiento optimizado
                coords_2d = self._postprocess_pose_optimized(pose_3d_np, img2bb_trans, root_depth)
                
                if coords_2d is not None:
                    # 9. Cache update con limpieza inteligente
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)
                    
                    self._update_cache_intelligent(cache_key, coords_2d, frame_time)
                    return coords_2d
                    
        except Exception as e:
            print(f"⚠️ Error en pose inteligente: {e}")
            return None
        
        return None
    
    def _postprocess_pose_optimized(self, pose_3d_raw, img2bb_trans, root_depth):
        """Post-procesamiento optimizado"""
        pose_3d = pose_3d_raw.copy()
        
        # Transformación optimizada
        pose_3d[:, 0] = pose_3d[:, 0] / self.cfg.output_shape[1] * self.cfg.input_shape[1]
        pose_3d[:, 1] = pose_3d[:, 1] / self.cfg.output_shape[0] * self.cfg.input_shape[0]
        
        # Transformación afín robusta
        pose_3d_xy1 = np.column_stack((pose_3d[:, :2], np.ones(len(pose_3d))))
        img2bb_trans_full = np.vstack((img2bb_trans, [0, 0, 1]))
        
        try:
            pose_3d[:, :2] = np.linalg.solve(img2bb_trans_full, pose_3d_xy1.T).T[:, :2]
        except np.linalg.LinAlgError:
            try:
                pose_3d[:, :2] = np.linalg.lstsq(img2bb_trans_full, pose_3d_xy1.T, rcond=None)[0].T[:, :2]
            except:
                return None
        
        # Profundidad
        pose_3d[:, 2] = (pose_3d[:, 2] / self.cfg.depth_dim * 2 - 1) * (self.cfg.bbox_3d_shape[0]/2) + root_depth
        
        return pose_3d[:, :2]
    
    def _update_cache_intelligent(self, key, result, timestamp):
        """Update de cache inteligente"""
        self.bbox_cache[key] = (result, timestamp)
        
        # Limpieza adaptativa
        if len(self.bbox_cache) > self.max_cache_size:
            # Eliminar entradas más antiguas que 2x timeout
            cutoff_time = timestamp - self.cache_timeout * 2
            old_keys = [k for k, (_, t) in self.bbox_cache.items() if t < cutoff_time]
            
            for k in old_keys:
                del self.bbox_cache[k]
            
            # Si aún está lleno, eliminar las más antiguas
            if len(self.bbox_cache) > self.max_cache_size:
                sorted_items = sorted(self.bbox_cache.items(), key=lambda x: x[1][1])
                for k, _ in sorted_items[:len(sorted_items)//3]:  # Eliminar 1/3
                    del self.bbox_cache[k]
    
    def get_performance_stats(self):
        """Estadísticas de rendimiento avanzadas"""
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        avg_processing_time = 0
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times) * 1000
        
        return {
            'avg_processing_time_ms': avg_processing_time,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.bbox_cache),
            'total_requests': total_requests
        }

class AdaptiveFrameProcessor:
    """Procesador de frames adaptativo"""
    
    def __init__(self, yolo_pipeline, pose_processor, hardware_caps):
        self.yolo_pipeline = yolo_pipeline
        self.pose_processor = pose_processor
        self.hardware_caps = hardware_caps
        
        # Queue adaptativo
        queue_size = 2 if hardware_caps['has_cuda'] else 1
        self.input_queue = Queue(maxsize=queue_size)
        self.output_queue = Queue(maxsize=queue_size)
        
        # Frame skipping adaptativo
        self.frame_skip_counter = 0
        self.skip_every_n_frames = hardware_caps['recommended_frame_skip']
        
        # Control de threading
        self.processing = True
        self.processor_thread = threading.Thread(target=self._process_frames_adaptive)
        self.processor_thread.daemon = True
        self.processor_thread.start()
        
        # Estadísticas avanzadas
        self.processing_times = deque(maxlen=30)
        self.frame_count = 0
        
        print(f"✅ AdaptiveFrameProcessor inicializado (queue={queue_size}, skip=1/{self.skip_every_n_frames})")
    
    def _process_frames_adaptive(self):
        """Loop de procesamiento adaptativo"""
        while self.processing:
            try:
                frame_data = self.input_queue.get(timeout=0.1)
                if frame_data is None:
                    break
                
                frame, frame_time = frame_data
                start_time = time.time()
                
                # Procesar frame
                result = self._process_single_frame_adaptive(frame, frame_time)
                
                # Estadísticas
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # Enviar resultado sin bloqueo
                try:
                    # Limpiar queue de salida si está lleno
                    while True:
                        try:
                            self.output_queue.get_nowait()
                        except Empty:
                            break
                    
                    self.output_queue.put((result, frame_time, processing_time), block=False)
                except:
                    pass
                
                # Adaptación dinámica de frame skipping
                if len(self.processing_times) >= 10:
                    avg_time = sum(list(self.processing_times)[-10:]) / 10
                    
                    if avg_time > 0.5:  # Muy lento
                        self.skip_every_n_frames = min(5, self.skip_every_n_frames + 1)
                    elif avg_time < 0.2:  # Rápido
                        self.skip_every_n_frames = max(1, self.skip_every_n_frames - 1)
                
            except Empty:
                continue
            except Exception as e:
                print(f"❌ Error en thread adaptativo: {e}")
    
    def _process_single_frame_adaptive(self, frame, frame_time):
        """Procesar frame individual con optimizaciones"""
        try:
            # YOLO detection
            detections = self.yolo_pipeline.predict_persons(frame, conf_threshold=0.25)
            
            # Extraer bboxes (tomar solo la mejor)
            bboxes = []
            if hasattr(detections, 'boxes') and len(detections.boxes[0]) > 0:
                boxes = detections.boxes[0]
                scores = detections.scores[0] if hasattr(detections, 'scores') else np.ones(len(boxes))
                
                if len(boxes) > 0:
                    best_idx = np.argmax(scores)
                    bboxes = [boxes[best_idx].tolist()]
            
            # Procesar pose
            coords_list = []
            if bboxes:
                coords = self.pose_processor.process_pose_intelligent(frame, bboxes[0], frame_time)
                if coords is not None:
                    coords_list.append(coords)
            
            return {'bboxes': bboxes, 'poses': coords_list}
            
        except Exception as e:
            print(f"⚠️ Error procesando frame adaptativo: {e}")
            return {'bboxes': [], 'poses': []}
    
    def add_frame(self, frame):
        """Agregar frame con skipping adaptativo"""
        self.frame_count += 1
        
        # Frame skipping dinámico
        if self.frame_count % self.skip_every_n_frames != 0:
            return True
        
        frame_time = time.time()
        try:
            # Limpiar queue de entrada si está lleno
            try:
                self.input_queue.get_nowait()
            except Empty:
                pass
            
            self.input_queue.put((frame, frame_time), block=False)
            return True
        except:
            return False
    
    def get_result(self):
        """Obtener resultado sin bloqueo"""
        try:
            return self.output_queue.get_nowait()
        except Empty:
            return None
    
    def get_comprehensive_stats(self):
        """Estadísticas comprehensivas"""
        basic_stats = self.pose_processor.get_performance_stats()
        
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            basic_stats.update({
                'queue_size': self.input_queue.qsize(),
                'fps_estimate': 1.0 / avg_time if avg_time > 0 else 0,
                'frames_processed': len(self.processing_times),
                'current_skip_rate': self.skip_every_n_frames,
                'total_frames': self.frame_count
            })
        
        return basic_stats
    
    def stop(self):
        """Detener procesamiento"""
        self.processing = False
        try:
            self.input_queue.put(None, timeout=1)
        except:
            pass
        self.processor_thread.join(timeout=3)

def setup_models_adaptive(args):
    """Configurar modelos con optimización adaptativa"""
    print("🚀 Configurando modelos ADAPTATIVOS...")
    
    # Detectar hardware
    hardware_caps = detect_hardware_capabilities()
    
    # Configuración ConvNeXt (mantener original)
    cfg.input_shape = (256, 256)
    cfg.output_shape = (32, 32)
    cfg.depth_dim = 32
    cfg.bbox_3d_shape = (2000, 2000, 2000)
    
    # 1. YOLO adaptativo
    if DEEPSPARSE_AVAILABLE:
        try:
            yolo_pipeline = Pipeline.create(task="yolo", model_path=args.yolo_model)
            print("✅ DeepSparse YOLO cargado")
        except:
            yolo_pipeline = None
    else:
        yolo_pipeline = None
    
    if yolo_pipeline is None:
        onnx_path = convert_yolo_to_onnx_optimized('yolov8n.pt')
        yolo_pipeline = AdaptiveYOLODetector(onnx_path, hardware_caps)
        print("✅ YOLO adaptativo cargado")
    
    # 2. ConvNeXt optimizado
    device = torch.device('cuda' if hardware_caps['has_cuda'] else 'cpu')
    pose_model = get_pose_net(cfg, is_train=False, joint_num=18)
    
    state = torch.load(args.pose_model, map_location=device)
    sd = state.get('network', state)
    pose_model.load_state_dict(sd, strict=False)
    pose_model = pose_model.to(device).eval()
    
    # Optimizaciones específicas de hardware
    if hardware_caps['has_cuda']:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        
        if hardware_caps['cuda_memory_gb'] >= 6:
            # Optimizaciones adicionales para GPUs potentes
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    print(f"✅ ConvNeXt adaptativo en {device}")
    
    # 3. RootNet
    root_wrapper = RootNetWrapper(args.rootnet_dir, args.rootnet_model)
    root_wrapper.load_model(use_gpu=hardware_caps['has_cuda'])
    print("✅ RootNet cargado")
    
    # 4. Transformación
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std),
    ])
    
    return yolo_pipeline, pose_model, device, root_wrapper, transform, hardware_caps
def optimize_for_cpu():
    """Optimizaciones específicas para CPU"""
    import torch
    torch.set_num_threads(6)  # Usar la mitad de tus cores
    torch.set_num_interop_threads(2)
    
    # Configuraciones específicas CPU
    os.environ['OMP_NUM_THREADS'] = '6'
    os.environ['MKL_NUM_THREADS'] = '6'
def main():
    optimize_for_cpu()
    parser = argparse.ArgumentParser(description="ConvNeXt FINAL - Adaptativo y Optimizado")
    parser.add_argument('--input', type=str, default='0', help='Video source')
    parser.add_argument('--yolo-model', type=str, 
                        default='zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none',
                        help='YOLO model path')
    parser.add_argument('--pose-model', type=str, required=True, help='ConvNeXt checkpoint')
    parser.add_argument('--rootnet-dir', type=str, 
                        default='/home/fabri/3DMPPE_ROOTNET_RELEASE',
                        help='RootNet directory')
    parser.add_argument('--rootnet-model', type=str,
                        default='/home/fabri/3DMPPE_ROOTNET_RELEASE/demo/snapshot_18.pth.tar',
                        help='RootNet checkpoint')
    args = parser.parse_args()
    
    # Configurar modelos adaptativos
    yolo_pipeline, pose_model, device, root_wrapper, transform, hardware_caps = setup_models_adaptive(args)
    
    # Procesador inteligente
    pose_processor = IntelligentPoseProcessor(
        pose_model, device, root_wrapper, transform, cfg, hardware_caps
    )
    
    # Frame processor adaptativo
    frame_processor = AdaptiveFrameProcessor(yolo_pipeline, pose_processor, hardware_caps)
    
    # Esqueleto optimizado
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
        if args.input.startswith('tcp://'):
            cap = cv2.VideoCapture(args.input)
        else:
            cap = cv2.VideoCapture(f"tcp://{args.input}:5000")
    
    if not cap.isOpened():
        print(f"❌ No se pudo abrir video: {args.input}")
        return
    
    # Optimizaciones de captura
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if hardware_caps['has_cuda']:
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("🚀 Demo FINAL ADAPTATIVO iniciado. Presione 'q' para salir.")
    
    # Variables de rendimiento
    frame_count = 0
    fps_counter = deque(maxlen=30)
    last_poses = []
    last_result_time = time.time()
    last_stats_time = time.time()
    
    try:
        while True:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Agregar frame para procesamiento
            frame_processor.add_frame(frame)
            
            # Obtener resultado más reciente
            result = frame_processor.get_result()
            if result:
                result_data, result_time, processing_time = result
                last_poses = result_data.get('poses', [])
                last_result_time = result_time
                
                # Dibujar bboxes
                for bbox in result_data.get('bboxes', []):
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Dibujar esqueletos
            for pose_coords in last_poses:
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
            
            # Estadísticas en pantalla
            frame_time = time.time() - frame_start
            fps_counter.append(1.0 / max(frame_time, 1e-6))
            
            if fps_counter:
                fps = sum(fps_counter) / len(fps_counter)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Estadísticas avanzadas
            stats = frame_processor.get_comprehensive_stats()
            proc_time = stats.get('avg_processing_time_ms', 0)
            cache_hit_rate = stats.get('cache_hit_rate', 0)
            
            # Color coding para latencia
            color = (0, 255, 0) if proc_time < 200 else (0, 165, 255) if proc_time < 400 else (0, 0, 255)
            
            cv2.putText(frame, f"Proc: {proc_time:.1f}ms", (10, 70),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Cache: {cache_hit_rate:.1f}%", (10, 110),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Mostrar latencia visual
            visual_latency = (time.time() - last_result_time) * 1000
            cv2.putText(frame, f"Visual Lag: {visual_latency:.0f}ms", (10, 150),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Información del hardware
            hw_info = "GPU" if hardware_caps['has_cuda'] else "CPU"
            cv2.putText(frame, f"FINAL ADAPTIVE ({hw_info})", (10, 190),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            cv2.imshow("ConvNeXt FINAL - Adaptive Optimized", frame)
            
            frame_count += 1
            
            # Estadísticas detalladas cada 60 frames
            if frame_count % 60 == 0:
                current_time = time.time()
                elapsed = current_time - last_stats_time
                
                avg_fps = sum(fps_counter) / len(fps_counter) if fps_counter else 0
                
                print(f"📊 Frame {frame_count}:")
                print(f"   Display FPS: {avg_fps:.1f}")
                print(f"   Processing: {proc_time:.1f}ms")
                print(f"   Cache Hit Rate: {cache_hit_rate:.1f}%")
                print(f"   Visual Lag: {visual_latency:.0f}ms")
                print(f"   Skip Rate: 1/{stats.get('current_skip_rate', 1)}")
                print(f"   Hardware: {hw_info} ({hardware_caps.get('cuda_memory_gb', 0):.1f}GB)")
                
                last_stats_time = current_time
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        frame_processor.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("🏁 Demo final adaptativo finalizado")

if __name__ == "__main__":
    main()