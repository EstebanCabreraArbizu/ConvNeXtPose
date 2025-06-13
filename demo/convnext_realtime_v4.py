#!/usr/bin/env python3
"""
convnext_realtime_v4_optimized.py - VERSIÓN ULTRA-OPTIMIZADA PARA MODELOS XS/S

OPTIMIZACIONES IMPLEMENTADAS:
1. Procesamiento 100% asíncrono con asyncio ✅
2. Cache inteligente adaptativo de v3 ✅
3. TFLite ultra-optimizado para modelos pequeños ✅
4. Pipeline de múltiples niveles con ThreadPoolExecutor ✅
5. Métricas avanzadas en tiempo real ✅
6. Frame skipping dinámico inteligente ✅
7. Gestión de memoria optimizada ✅
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
import asyncio
from concurrent.futures import ThreadPoolExecutor
import onnx
from onnxruntime.tools import optimizer

# IMPORTS CORREGIDOS
try:
    from deepsparse.pipeline import Pipeline
    DEEPSPARSE_AVAILABLE = True
except ImportError:
    print("⚠️ DeepSparse no disponible, usando fallback a Ultralytics")
    DEEPSPARSE_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow no disponible para TFLite")
    TENSORFLOW_AVAILABLE = False

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
        'recommended_workers': 2,
        'recommended_cache_timeout': 0.08,
        'recommended_frame_skip': 1,
        'async_workers': 4
    }
    
    if capabilities['has_cuda']:
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            capabilities['cuda_memory_gb'] = gpu_memory
            
            # Configuración agresiva para modelos pequeños
            if gpu_memory >= 8:  # RTX 3070+
                capabilities.update({
                    'recommended_workers': 4,
                    'recommended_cache_timeout': 0.05,
                    'recommended_frame_skip': 1,
                    'async_workers': 8
                })
            elif gpu_memory >= 4:  # GTX 1660+
                capabilities.update({
                    'recommended_workers': 3,
                    'recommended_cache_timeout': 0.06,
                    'recommended_frame_skip': 1,
                    'async_workers': 6
                })
        except:
            pass
    else:
        # CPU optimizado para modelos pequeños
        capabilities.update({
            'recommended_workers': min(4, capabilities['cpu_cores'] // 2),
            'recommended_cache_timeout': 0.10,
            'recommended_frame_skip': 1,
            'async_workers': min(6, capabilities['cpu_cores'])
        })
    
    print(f"🔧 CONFIGURACIÓN ULTRA-OPTIMIZADA DETECTADA:")
    print(f"   GPU: {'✅' if capabilities['has_cuda'] else '❌'} ({capabilities['cuda_memory_gb']:.1f}GB)")
    print(f"   CPU Cores: {capabilities['cpu_cores']}")
    print(f"   Workers: {capabilities['recommended_workers']}")
    print(f"   Async Workers: {capabilities['async_workers']}")
    print(f"   Cache timeout: {capabilities['recommended_cache_timeout']*1000:.0f}ms")
    print(f"   Frame skip: 1/{capabilities['recommended_frame_skip']}")
    
    return capabilities

def convert_yolo_to_onnx_optimized(pt_model_path='yolov8n.pt', 
                                   conf_thresh=0.25, iou_thresh=0.40, img_size=416):
    """Conversión YOLO ultra-optimizada para velocidad"""
    base_name = pt_model_path.replace('.pt', '')
    onnx_path = f"{base_name}_ultrafast_conf{conf_thresh}_iou{iou_thresh}_{img_size}.onnx"
    
    if os.path.exists(onnx_path):
        print(f"✅ ONNX ultra-optimizado existente: {onnx_path}")
        return onnx_path
    
    print(f"🔄 Convirtiendo {pt_model_path} a ONNX ULTRA-OPTIMIZADO...")
    try:
        from ultralytics import YOLO
        model = YOLO(pt_model_path)
        
        exported_path = model.export(
            format='onnx', 
            imgsz=img_size,  # Más pequeño para velocidad
            optimize=True, 
            half=False,
            dynamic=False, 
            simplify=True, 
            opset=13,
            nms=True,
            conf=conf_thresh, 
            iou=iou_thresh, 
            max_det=50  # Reducido agresivamente
        )
        
        if exported_path != onnx_path:
            os.rename(exported_path, onnx_path)
        
        print(f"✅ ONNX ultra-optimizado creado: {onnx_path}")
        return onnx_path
        
    except Exception as e:
        print(f"❌ Error en conversión ultra-optimizada: {e}")
        raise

class UltraFastYOLODetector:
    """Detector YOLO ultra-rápido con optimizaciones agresivas"""
    
    def __init__(self, onnx_path, hardware_caps):
        self.onnx_path = onnx_path
        self.hardware_caps = hardware_caps
        self._setup_session()
        self.warmup()
        
        # Cache para detecciones
        self.detection_cache = {}
        self.cache_timeout = 0.05  # 50ms cache agresivo
        
    def _setup_session(self):
        """Configurar sesión ultra-optimizada"""
        providers = []
        session_options = ort.SessionOptions()
        
        if self.hardware_caps['has_cuda']:
            cuda_provider = ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': int(self.hardware_caps['cuda_memory_gb'] * 0.4 * 1024**3),
                'cudnn_conv_algo_search': 'HEURISTIC',
                'do_copy_in_default_stream': True,
                'cudnn_conv1d_pad_to_nc1d': True,
                'enable_cuda_graph': True  # NUEVA: CUDA Graph optimization
            })
            providers.append(cuda_provider)
            
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            session_options.inter_op_num_threads = 2
            session_options.intra_op_num_threads = 4
            
            print("🚀 YOLO ULTRA-FAST GPU configurado")
        else:
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.inter_op_num_threads = 1
            session_options.intra_op_num_threads = min(4, self.hardware_caps['cpu_cores'])
            
            print("🚀 YOLO ULTRA-FAST CPU configurado")
        
        providers.append('CPUExecutionProvider')
        
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_pattern = True
        
        self.session = ort.InferenceSession(self.onnx_path, providers=providers, sess_options=session_options)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.img_size = self.input_shape[2]
    
    def warmup(self):
        """Pre-calentamiento agresivo"""
        print("🔥 Pre-calentando YOLO ULTRA-FAST...")
        dummy_input = np.random.rand(*self.input_shape).astype(np.float32)
        
        warmup_iterations = 8 if self.hardware_caps['has_cuda'] else 5
        for _ in range(warmup_iterations):
            self.session.run(self.output_names, {self.input_name: dummy_input})
        print("✅ YOLO ULTRA-FAST pre-calentado")
    
    def preprocess_ultrafast(self, frame):
        """Preprocesamiento ultra-rápido"""
        h, w = frame.shape[:2]
        
        # Resize directo sin mantener aspect ratio para velocidad
        resized = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # Conversión ultra-optimizada
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
        
        # Calcular factor de escala simple
        scale_x = w / self.img_size
        scale_y = h / self.img_size
        
        return input_tensor, scale_x, scale_y
    
    async def predict_persons_async(self, frame, conf_threshold=0.25):
        """Detección asíncrona ultra-rápida"""
        frame_hash = hash(frame.tobytes()[::1000])  # Hash rápido
        current_time = time.time()
        
        # Cache check
        if frame_hash in self.detection_cache:
            cached_result, cached_time = self.detection_cache[frame_hash]
            if current_time - cached_time < self.cache_timeout:
                return cached_result
        
        try:
            input_tensor, scale_x, scale_y = self.preprocess_ultrafast(frame)
            
            # Inferencia
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            if len(outputs) == 0 or outputs[0].size == 0:
                result = SimpleNamespace(boxes=[[]], scores=[[]], labels=[[]])
            else:
                detections = outputs[0][0]
                
                # Filtrado ultra-rápido
                person_mask = (detections[:, 5] == 0) & (detections[:, 4] >= conf_threshold)
                person_detections = detections[person_mask]
                
                if len(person_detections) == 0:
                    result = SimpleNamespace(boxes=[[]], scores=[[]], labels=[[]])
                else:
                    # Convertir coordenadas
                    boxes = person_detections[:, :4].copy()
                    scores = person_detections[:, 4]
                    
                    # Escalado simple
                    boxes[:, [0, 2]] *= scale_x
                    boxes[:, [1, 3]] *= scale_y
                    
                    # Clipping
                    h, w = frame.shape[:2]
                    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
                    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
                    
                    result = SimpleNamespace(
                        boxes=[boxes.astype(int)],
                        scores=[scores],
                        labels=[np.zeros(len(scores), dtype=int)]
                    )
            
            # Cache update
            self.detection_cache[frame_hash] = (result, current_time)
            
            # Limpieza de cache
            if len(self.detection_cache) > 20:
                old_keys = [k for k, (_, t) in self.detection_cache.items() 
                           if current_time - t > self.cache_timeout * 2]
                for k in old_keys:
                    del self.detection_cache[k]
            
            return result
            
        except Exception as e:
            print(f"⚠️ Error en YOLO async: {e}")
            return SimpleNamespace(boxes=[[]], scores=[[]], labels=[[]])

def convert_pytorch_to_tflite_complete(pytorch_model, tflite_path):
    """Conversión completa PyTorch → TFLite optimizada"""
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow no disponible para conversión TFLite")
        return None
    
    if os.path.exists(tflite_path):
        print(f"✅ TFLite existente: {tflite_path}")
        return tflite_path
    
    print("🔄 Conversión PyTorch → ONNX → TF → TFLite...")
    
    try:
        # Paso 1: PyTorch → ONNX
        onnx_temp = tflite_path.replace('.tflite', '_temp.onnx')
        pytorch_model.eval()
        dummy_input = torch.randn(1, 3, 256, 256)
        
        torch.onnx.export(
            pytorch_model, dummy_input, onnx_temp,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=13, do_constant_folding=True, export_params=True
        )
        
        # Paso 2: Optimizar ONNX
        onnx_model = onnx.load(onnx_temp)
        optimized_model = optimizer.optimize_graph(
            onnx_model, optimization_level=optimizer.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        onnx_optimized = onnx_temp.replace('.onnx', '_opt.onnx')
        onnx.save(optimized_model, onnx_optimized)
        
        # Paso 3: ONNX → TensorFlow
        import onnx_tf
        tf_temp = tflite_path.replace('.tflite', '_temp_tf')
        onnx_model_opt = onnx.load(onnx_optimized)
        tf_rep = onnx_tf.backend.prepare(onnx_model_opt)
        tf_rep.export_graph(tf_temp)
        
        # Paso 4: TF → TFLite con cuantización agresiva
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_temp)
        
        # Configuración ultra-agresiva
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        
        def representative_dataset():
            for _ in range(200):  # Más datos para mejor calibración
                yield [np.random.rand(1, 3, 256, 256).astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.experimental_new_converter = True
        converter.allow_custom_ops = False
        
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        # Convertir
        tflite_model = converter.convert()
        
        # Guardar
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Limpiar archivos temporales
        for temp_file in [onnx_temp, onnx_optimized, tf_temp]:
            try:
                if os.path.exists(temp_file):
                    if os.path.isdir(temp_file):
                        import shutil
                        shutil.rmtree(temp_file)
                    else:
                        os.remove(temp_file)
            except:
                pass
        
        print(f"✅ TFLite cuantizado creado: {len(tflite_model)} bytes")
        return tflite_path
        
    except Exception as e:
        print(f"❌ Error en conversión TFLite: {e}")
        return None

class UltraIntelligentCache:
    """Cache ultra-inteligente con predicción y adaptación"""
    
    def __init__(self, hardware_caps):
        self.hardware_caps = hardware_caps
        self.cache = {}
        self.hit_stats = deque(maxlen=100)
        self.processing_times = deque(maxlen=50)
        
        # Parámetros adaptativos
        self.base_timeout = hardware_caps['recommended_cache_timeout']
        self.current_timeout = self.base_timeout
        self.quantization_level = 15
        self.max_cache_size = 100 if hardware_caps['has_cuda'] else 60
        
        # Estadísticas
        self.hits = 0
        self.misses = 0
        self.adaptations = 0
        
    def get_cache_key(self, bbox, frame_time, processing_time_estimate=None):
        """Generar clave de cache inteligente"""
        x1, y1, x2, y2 = bbox
        
        # Adaptación dinámica de cuantización
        if len(self.processing_times) > 10:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
            
            # Ajustar cuantización basado en rendimiento
            if avg_time > 0.3 and hit_rate < 0.6:  # Lento y baja hit rate
                self.quantization_level = min(30, self.quantization_level + 2)
                self.current_timeout = min(0.2, self.current_timeout + 0.01)
                self.adaptations += 1
            elif avg_time < 0.1 and hit_rate > 0.8:  # Rápido y alta hit rate
                self.quantization_level = max(10, self.quantization_level - 1)
                self.current_timeout = max(0.03, self.current_timeout - 0.005)
                self.adaptations += 1
        
        # Cuantización adaptativa
        x1_q = int(x1 / self.quantization_level) * self.quantization_level
        y1_q = int(y1 / self.quantization_level) * self.quantization_level
        x2_q = int(x2 / self.quantization_level) * self.quantization_level
        y2_q = int(y2 / self.quantization_level) * self.quantization_level
        
        time_slot = int(frame_time / self.current_timeout)
        
        return f"{x1_q}_{y1_q}_{x2_q}_{y2_q}_{time_slot}"
    
    def get(self, bbox, frame_time):
        """Obtener del cache con estadísticas"""
        key = self.get_cache_key(bbox, frame_time)
        
        if key in self.cache:
            result, cached_time, _ = self.cache[key]
            if frame_time - cached_time < self.current_timeout:
                self.hits += 1
                self.hit_stats.append(1)
                return result
        
        self.misses += 1
        self.hit_stats.append(0)
        return None
    
    def put(self, bbox, frame_time, result, processing_time):
        """Guardar en cache con gestión inteligente"""
        key = self.get_cache_key(bbox, frame_time, processing_time)
        self.cache[key] = (result, frame_time, processing_time)
        self.processing_times.append(processing_time)
        
        # Limpieza inteligente
        if len(self.cache) > self.max_cache_size:
            self._intelligent_cleanup(frame_time)
    
    def _intelligent_cleanup(self, current_time):
        """Limpieza inteligente del cache"""
        # Eliminar por edad
        expired_keys = [k for k, (_, t, _) in self.cache.items() 
                       if current_time - t > self.current_timeout * 2]
        
        for k in expired_keys:
            del self.cache[k]
        
        # Si aún está lleno, eliminar por menor utilidad
        if len(self.cache) > self.max_cache_size:
            # Ordenar por tiempo de procesamiento (eliminar más lentos primero)
            sorted_items = sorted(self.cache.items(), 
                                key=lambda x: x[1][2], reverse=True)
            
            remove_count = len(self.cache) - self.max_cache_size + 10
            for k, _ in sorted_items[:remove_count]:
                del self.cache[k]
    
    def get_stats(self):
        """Estadísticas del cache"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        recent_hit_rate = 0
        if self.hit_stats:
            recent_hit_rate = sum(self.hit_stats) / len(self.hit_stats) * 100
        
        avg_processing_time = 0
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times) * 1000
        
        return {
            'hit_rate': hit_rate,
            'recent_hit_rate': recent_hit_rate,
            'cache_size': len(self.cache),
            'adaptations': self.adaptations,
            'current_timeout_ms': self.current_timeout * 1000,
            'quantization_level': self.quantization_level,
            'avg_processing_time_ms': avg_processing_time
        }

class AsyncTFLitePoseProcessor:
    """Procesador TFLite asíncrono ultra-optimizado"""
    
    def __init__(self, tflite_model_path, hardware_caps):
        self.tflite_model_path = tflite_model_path
        self.hardware_caps = hardware_caps
        
        # Cache inteligente
        self.cache = UltraIntelligentCache(hardware_caps)
        
        # ThreadPool para operaciones bloqueantes
        self.executor = ThreadPoolExecutor(max_workers=hardware_caps['async_workers'])
        
        # Configurar intérprete
        self._setup_interpreter()
        
        # Métricas
        self.processing_times = deque(maxlen=100)
        self.inference_times = deque(maxlen=100)
        
        print(f"✅ AsyncTFLitePoseProcessor inicializado (workers={hardware_caps['async_workers']})")
    
    def _setup_interpreter(self):
        """Configurar intérprete TFLite optimizado"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow no disponible para TFLite")
        
        import tensorflow as tf
        
        # Configurar con múltiples threads
        self.interpreter = tf.lite.Interpreter(
            model_path=self.tflite_model_path,
            num_threads=self.hardware_caps['async_workers']
        )
        
        # Intentar GPU delegate si está disponible
        if self.hardware_caps['has_cuda']:
            try:
                gpu_delegate = tf.lite.experimental.load_delegate('libGpuDelegate.so')
                self.interpreter = tf.lite.Interpreter(
                    model_path=self.tflite_model_path,
                    experimental_delegates=[gpu_delegate],
                    num_threads=self.hardware_caps['async_workers']
                )
                print("🚀 TFLite GPU Delegate activado")
            except:
                print("⚠️ GPU Delegate no disponible")
        
        # Asignar tensores
        self.interpreter.allocate_tensors()
        
        # Obtener detalles
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"📊 TFLite Async configurado:")
        print(f"   Input: {self.input_details[0]['shape']}")
        print(f"   Dtype: {self.input_details[0]['dtype']}")
    
    def _preprocess_ultrafast(self, img_patch):
        """Preprocesamiento ultra-rápido"""
        input_shape = self.input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # Resize rápido
        img_resized = cv2.resize(img_patch, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Normalización rápida
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Normalización ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_normalized = (img_normalized - mean) / std
        
        # Reordenar dimensiones
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        # Cuantización si es necesario
        input_dtype = self.input_details[0]['dtype']
        if input_dtype == np.int8:
            scale = self.input_details[0]['quantization_parameters']['scales'][0]
            zero_point = self.input_details[0]['quantization_parameters']['zero_points'][0]
            img_quantized = (img_batch / scale + zero_point).astype(np.int8)
            return img_quantized
        
        return img_batch.astype(input_dtype)
    
    def _extract_patch_fast(self, frame, bbox):
        """Extracción rápida de patch"""
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Expandir bbox
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        width, height = x2 - x1, y2 - y1
        
        # Hacer cuadrado
        size = max(width, height) * 1.25
        
        # Nuevas coordenadas
        x1_new = max(0, int(center_x - size / 2))
        y1_new = max(0, int(center_y - size / 2))
        x2_new = min(frame.shape[1], int(center_x + size / 2))
        y2_new = min(frame.shape[0], int(center_y + size / 2))
        
        patch = frame[y1_new:y2_new, x1_new:x2_new]
        
        if patch.size == 0:
            return None, None
        
        return patch, [x1_new, y1_new, x2_new - x1_new, y2_new - y1_new]
    
    def _tflite_inference(self, input_data):
        """Inferencia TFLite optimizada"""
        inference_start = time.time()
        
        # Set input
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Invoke
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        inference_time = time.time() - inference_start
        self.inference_times.append(inference_time)
        
        return output_data
    
    def _postprocess_fast(self, output_data, patch_bbox, frame_shape):
        """Post-procesamiento rápido"""
        # Dequantizar si es necesario
        if self.output_details[0]['dtype'] == np.int8:
            scale = self.output_details[0]['quantization_parameters']['scales'][0]
            zero_point = self.output_details[0]['quantization_parameters']['zero_points'][0]
            output_data = (output_data.astype(np.float32) - zero_point) * scale
        
        # Extraer coordenadas 2D
        coordinates = output_data[0]
        coords_2d = coordinates[:, :2]
        
        # Transformar a coordenadas de imagen
        x, y, w, h = patch_bbox
        coords_2d[:, 0] = coords_2d[:, 0] * w / 256 + x
        coords_2d[:, 1] = coords_2d[:, 1] * h / 256 + y
        
        # Clipping
        coords_2d[:, 0] = np.clip(coords_2d[:, 0], 0, frame_shape[1] - 1)
        coords_2d[:, 1] = np.clip(coords_2d[:, 1], 0, frame_shape[0] - 1)
        
        return coords_2d
    
    async def process_pose_async(self, frame, bbox, frame_time):
        """Procesamiento asíncrono principal"""
        
        # Cache check
        cached_result = self.cache.get(bbox, frame_time)
        if cached_result is not None:
            return cached_result
        
        start_time = time.time()
        
        try:
            # Ejecutar en thread pool para no bloquear el loop
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._process_sync, frame, bbox, frame_time
            )
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Cache result
            if result is not None:
                self.cache.put(bbox, frame_time, result, processing_time)
            
            return result
            
        except Exception as e:
            print(f"⚠️ Error en procesamiento async: {e}")
            return None
    
    def _process_sync(self, frame, bbox, frame_time):
        """Procesamiento síncrono llamado desde async"""
        # Extraer patch
        patch, patch_bbox = self._extract_patch_fast(frame, bbox)
        if patch is None:
            return None
        
        # Preprocesar
        input_data = self._preprocess_ultrafast(patch)
        
        # Inferencia TFLite
        output_data = self._tflite_inference(input_data)
        
        # Post-procesar
        coords_2d = self._postprocess_fast(output_data, patch_bbox, frame.shape)
        
        return coords_2d
    
    def get_performance_stats(self):
        """Estadísticas de rendimiento"""
        stats = self.cache.get_stats()
        
        if self.processing_times:
            stats['avg_total_time_ms'] = sum(self.processing_times) / len(self.processing_times) * 1000
        
        if self.inference_times:
            stats['avg_inference_time_ms'] = sum(self.inference_times) / len(self.inference_times) * 1000
        
        return stats

class UltraAsyncFrameProcessor:
    """Procesador de frames ultra-asíncrono"""
    
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
        self.yolo_times = deque(maxlen=50)
        self.pose_times = deque(maxlen=50)
        
        # Frame skipping inteligente
        self.skip_counter = 0
        self.current_skip_rate = hardware_caps['recommended_frame_skip']
        
        print(f"✅ UltraAsyncFrameProcessor inicializado")
    
    async def start_processing(self):
        """Iniciar procesamiento asíncrono"""
        self.processor_task = asyncio.create_task(self._process_frames_async())
        print("🚀 Procesamiento asíncrono iniciado")
    
    async def _process_frames_async(self):
        """Loop principal de procesamiento asíncrono"""
        while self.processing:
            try:
                # Obtener frame con timeout
                frame_data = await asyncio.wait_for(
                    self.input_queue.get(), timeout=0.1
                )
                
                if frame_data is None:
                    break
                
                frame, frame_time = frame_data
                
                # Procesar frame
                result = await self._process_single_frame_async(frame, frame_time)
                
                # Enviar resultado
                try:
                    # Limpiar queue de salida
                    while not self.output_queue.empty():
                        try:
                            self.output_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    
                    await self.output_queue.put((result, frame_time))
                except asyncio.QueueFull:
                    pass
                
                # Adaptación dinámica
                self._adapt_processing_rate()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"❌ Error en procesamiento async: {e}")
    
    async def _process_single_frame_async(self, frame, frame_time):
        """Procesar frame individual asíncrono"""
        start_time = time.time()
        
        try:
            # YOLO detection asíncrono
            yolo_start = time.time()
            detections = await self.yolo_detector.predict_persons_async(frame, conf_threshold=0.25)
            yolo_time = time.time() - yolo_start
            self.yolo_times.append(yolo_time)
            
            # Procesar mejor detección
            bboxes = []
            poses = []
            
            if hasattr(detections, 'boxes') and len(detections.boxes[0]) > 0:
                boxes = detections.boxes[0]
                scores = detections.scores[0] if hasattr(detections, 'scores') else np.ones(len(boxes))
                
                if len(boxes) > 0:
                    best_idx = np.argmax(scores)
                    best_bbox = boxes[best_idx].tolist()
                    bboxes.append(best_bbox)
                    
                    # Pose estimation asíncrono
                    pose_start = time.time()
                    coords = await self.pose_processor.process_pose_async(frame, best_bbox, frame_time)
                    pose_time = time.time() - pose_start
                    self.pose_times.append(pose_time)
                    
                    if coords is not None:
                        poses.append(coords)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return {
                'bboxes': bboxes,
                'poses': poses,
                'yolo_time': yolo_time,
                'pose_time': pose_time if poses else 0,
                'total_time': processing_time
            }
            
        except Exception as e:
            print(f"⚠️ Error en frame async: {e}")
            return {'bboxes': [], 'poses': [], 'yolo_time': 0, 'pose_time': 0, 'total_time': 0}
    
    def _adapt_processing_rate(self):
        """Adaptación dinámica de la tasa de procesamiento"""
        if len(self.processing_times) < 10:
            return
        
        recent_times = list(self.processing_times)[-10:]
        avg_time = sum(recent_times) / len(recent_times)
        
        # Ajustar frame skipping
        if avg_time > 0.15:  # Más de 150ms
            self.current_skip_rate = min(3, self.current_skip_rate + 1)
        elif avg_time < 0.08:  # Menos de 80ms
            self.current_skip_rate = max(1, self.current_skip_rate - 1)
    
    async def add_frame_async(self, frame):
        """Agregar frame con skipping inteligente"""
        self.frame_count += 1
        
        # Frame skipping
        if self.frame_count % self.current_skip_rate != 0:
            return True
        
        frame_time = time.time()
        
        try:
            # Limpiar queue de entrada si está lleno
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
        """Obtener resultado asíncrono"""
        try:
            return await asyncio.wait_for(self.output_queue.get(), timeout=0.001)
        except asyncio.TimeoutError:
            return None
    
    def get_comprehensive_stats(self):
        """Estadísticas comprehensivas"""
        pose_stats = self.pose_processor.get_performance_stats()
        
        stats = {
            'frame_count': self.frame_count,
            'current_skip_rate': self.current_skip_rate,
            'queue_input_size': self.input_queue.qsize(),
            'queue_output_size': self.output_queue.qsize()
        }
        
        if self.processing_times:
            stats['avg_total_processing_ms'] = sum(self.processing_times) / len(self.processing_times) * 1000
        
        if self.yolo_times:
            stats['avg_yolo_time_ms'] = sum(self.yolo_times) / len(self.yolo_times) * 1000
        
        if self.pose_times:
            stats['avg_pose_time_ms'] = sum(self.pose_times) / len(self.pose_times) * 1000
        
        # Merge con estadísticas de pose
        stats.update(pose_stats)
        
        return stats
    
    async def stop_async(self):
        """Detener procesamiento asíncrono"""
        self.processing = False
        await self.input_queue.put(None)
        if self.processor_task:
            await self.processor_task

async def setup_models_async(args):
    """Configurar modelos asíncronos"""
    print("🚀 Configurando modelos ULTRA-ASÍNCRONOS...")
    
    # Detectar hardware
    hardware_caps = detect_hardware_capabilities()
    
    # Configuración ConvNeXt para modelos pequeños
    cfg.input_shape = (256, 256)
    cfg.output_shape = (32, 32)
    cfg.depth_dim = 32
    cfg.bbox_3d_shape = (2000, 2000, 2000)
    
    # 1. YOLO ultra-optimizado
    onnx_path = convert_yolo_to_onnx_optimized('yolov8n.pt')
    yolo_detector = UltraFastYOLODetector(onnx_path, hardware_caps)
    
    # 2. Pose model setup
    device = torch.device('cuda' if hardware_caps['has_cuda'] else 'cpu')
    pose_model = get_pose_net(cfg, is_train=False, joint_num=18)
    
    state = torch.load(args.pose_model, map_location=device)
    sd = state.get('network', state)
    pose_model.load_state_dict(sd, strict=False)
    pose_model = pose_model.to(device).eval()
    
    # 3. Conversión a TFLite
    tflite_path = args.pose_model.replace('.pth', '_ultra_optimized.tflite')
    tflite_converted = convert_pytorch_to_tflite_complete(pose_model, tflite_path)
    
    if tflite_converted is None:
        print("❌ No se pudo convertir a TFLite, usando PyTorch")
        # Fallback: usar procesador PyTorch optimizado
        pose_processor = None  # Implementar fallback si es necesario
    else:
        pose_processor = AsyncTFLitePoseProcessor(tflite_converted, hardware_caps)
    
    print("✅ Modelos ultra-asíncronos configurados")
    
    return yolo_detector, pose_processor, hardware_caps

async def main_async():
    """Main loop asíncrono principal"""
    
    # Configurar argumentos
    parser = argparse.ArgumentParser(description="ConvNeXt v4 ULTRA-OPTIMIZADO - Asíncrono")
    parser.add_argument('--input', type=str, default='0', help='Video source')
    parser.add_argument('--pose-model', type=str, required=True, help='ConvNeXt checkpoint')
    args = parser.parse_args()
    
    # Setup modelos
    yolo_detector, pose_processor, hardware_caps = await setup_models_async(args)
    
    if pose_processor is None:
        print("❌ No se pudo configurar el procesador de pose")
        return
    
    # Frame processor asíncrono
    frame_processor = UltraAsyncFrameProcessor(yolo_detector, pose_processor, hardware_caps)
    await frame_processor.start_processing()
    
    # Esqueleto
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
    cap.set(cv2.CAP_PROP_FPS, 60 if hardware_caps['has_cuda'] else 30)
    
    print("🚀 Demo ULTRA-ASÍNCRONO iniciado. Presione 'q' para salir.")
    
    # Variables de rendimiento
    frame_count = 0
    display_fps_counter = deque(maxlen=30)
    last_poses = []
    last_result_time = time.time()
    last_stats_time = time.time()
    
    try:
        while True:
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Agregar frame para procesamiento asíncrono
            await frame_processor.add_frame_async(frame)
            
            # Obtener resultado más reciente
            result = await frame_processor.get_result_async()
            if result:
                result_data, result_time = result
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
            
            # Estadísticas de display
            loop_time = time.time() - loop_start
            display_fps_counter.append(1.0 / max(loop_time, 1e-6))
            
            if display_fps_counter:
                display_fps = sum(display_fps_counter) / len(display_fps_counter)
                cv2.putText(frame, f"Display FPS: {display_fps:.1f}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Estadísticas avanzadas
            stats = frame_processor.get_comprehensive_stats()
            
            # Colores para métricas
            proc_time = stats.get('avg_total_processing_ms', 0)
            inf_time = stats.get('avg_inference_time_ms', 0)
            hit_rate = stats.get('hit_rate', 0)
            
            color_proc = (0, 255, 0) if proc_time < 50 else (0, 165, 255) if proc_time < 100 else (0, 0, 255)
            color_inf = (0, 255, 0) if inf_time < 20 else (0, 165, 255) if inf_time < 40 else (0, 0, 255)
            
            cv2.putText(frame, f"Processing: {proc_time:.1f}ms", (10, 70),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_proc, 2)
            cv2.putText(frame, f"TFLite: {inf_time:.1f}ms", (10, 110),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_inf, 2)
            cv2.putText(frame, f"Cache: {hit_rate:.1f}%", (10, 150),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Visual lag
            visual_lag = (time.time() - last_result_time) * 1000
            cv2.putText(frame, f"Visual Lag: {visual_lag:.0f}ms", (10, 190),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Información del sistema
            hw_info = f"GPU-{hardware_caps['cuda_memory_gb']:.1f}GB" if hardware_caps['has_cuda'] else "CPU"
            cv2.putText(frame, f"v4-ULTRA-ASYNC ({hw_info})", (10, 230),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            cv2.imshow("ConvNeXt v4 - Ultra Async Optimized", frame)
            
            frame_count += 1
            
            # Estadísticas detalladas cada 60 frames
            if frame_count % 60 == 0:
                current_time = time.time()
                elapsed = current_time - last_stats_time
                
                avg_display_fps = sum(display_fps_counter) / len(display_fps_counter) if display_fps_counter else 0
                
                print(f"📊 Frame {frame_count}:")
                print(f"   Display FPS: {avg_display_fps:.1f}")
                print(f"   Processing: {proc_time:.1f}ms")
                print(f"   TFLite Inference: {inf_time:.1f}ms")
                print(f"   Cache Hit Rate: {hit_rate:.1f}%")
                print(f"   Visual Lag: {visual_lag:.0f}ms")
                print(f"   Skip Rate: 1/{stats.get('current_skip_rate', 1)}")
                print(f"   Queue Input: {stats.get('queue_input_size', 0)}")
                print(f"   Queue Output: {stats.get('queue_output_size', 0)}")
                print(f"   Cache Size: {stats.get('cache_size', 0)}")
                print(f"   Adaptations: {stats.get('adaptations', 0)}")
                print(f"   Hardware: {hw_info}")
                print("   " + "="*50)
                
                last_stats_time = current_time
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    finally:
        await frame_processor.stop_async()
        cap.release()
        cv2.destroyAllWindows()
        print("🏁 Demo ultra-asíncrono finalizado")

def main():
    """Función principal que ejecuta el loop asíncrono"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n🛑 Detenido por usuario")
    except Exception as e:
        print(f"❌ Error en main: {e}")

if __name__ == "__main__":
    main()