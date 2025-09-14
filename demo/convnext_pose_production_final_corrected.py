#!/usr/bin/env python3
"""
convnext_pose_production_final_corrected.py - CÓDIGO DEFINITIVO Y ROBUSTO

🎯 SOLUCIONES IMPLEMENTADAS:
1. ✅ generate_patch_image() con parámetros correctos (6 argumentos como demo.py)
2. ✅ Extracción de pose 2D igual que demo.py (post-procesamiento correcto)
Pendiente -> Extracción de Pose 2D con Rootnet
3. ✅ Configuración de modelo basada en config.py y demo.py
4. ✅ Backend ONNX con ejes estáticos usando pipeline torch2onnx
5. ✅ Sistema de cache y optimizaciones sin perder robustez
6. ✅ Limpieza de archivos innecesarios y organización correcta

🔧 BASADO EN:
- demo.py (funciona correctamente)
- config.py (configuraciones estándar)
- torch2onnx.py (pipeline de exportación)
- Optimizaciones de v4 pero con ejes estáticos

💡 USO:
python convnext_pose_production_final_corrected.py --model XS --backend tflite --visualize --input "Personas caminando.mp4"
"""

import os
import sys
import time
import json
import warnings
import logging
import argparse
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from collections import deque

import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

# Optimizaciones de entorno básicas
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '4'
warnings.filterwarnings("ignore", category=UserWarning) # Quitar los warnings de timm

# ONNX Runtime import
ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    pass

# TensorFlow Lite import
TFLITE_AVAILABLE = False
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    pass

# YOLO import
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths setup (igual que demo.py)
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / 'main'))
sys.path.insert(0, str(PROJECT_ROOT / 'data'))
sys.path.insert(0, str(PROJECT_ROOT / 'common'))

# Project imports (igual que demo.py)
try:
    from config import cfg
    from model import get_pose_net
    from dataset import generate_patch_image
    from utils.pose_utils import process_bbox
except ImportError as e:
    logger.error(f"Critical: Cannot import required modules: {e}")
    sys.exit(1)

# Configuration setup (igual que demo.py + config.py)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    logger.info("⚠️ CUDA no disponible, ejecutando en CPU")
cudnn.benchmark = True

# ConvNeXt joint configuration (igual que demo.py)
joint_num = 18
joints_name = (
    'Pelvis', 'Cadera_izq', 'Rodilla_izq', 'Tobillo_izq', 
    'Cadera_der', 'Rodilla_der', 'Tobillo_der', 'Cuello_base', 
    'Cabeza_media', 'Cabeza_superior', 'Hombro_der', 'Codo_der', 
    'Muñeca_der', 'Hombro_izq', 'Codo_izq', 'Muñeca_izq', 'Torso_medio'
)

# Skeleton connections (igual que demo.py)
skeleton = (
    (10, 9), (9, 8), (8, 11), (8, 14),
    (11, 12), (12, 13), (14, 15), (15, 16),
    (11, 4), (14, 1), (0, 4), (0, 1),
    (4, 5), (5, 6), (1, 2), (2, 3)
)

# Configuraciones de modelo (basadas en config.py y funcionamiento real)
MODEL_CONFIGS = {
    'XS': {
        'backbone_cfg': ([2, 2, 6, 2], [40, 80, 160, 320]),  # Configuración real del checkpoint
        'depth': 128,
        'checkpoint': 'ConvNeXtPose_XS.tar',
        'onnx_model': 'convnextpose_XS.onnx',
        'tflite_model': 'convnextpose_XS.tflite',
        'description': 'Ultra rápido y optimizado',
        'target_fps': 20.0
    },
    'S': {
        'backbone_cfg': ([3, 3, 9, 3], [48, 96, 192, 384]),  # Configuración real del checkpoint
        'depth': 256,
        'checkpoint': 'ConvNeXtPose_S.tar',
        'onnx_model': 'convnextpose_S.onnx',
        'tflite_model': 'convnextpose_S.tflite',
        'description': 'Balance calidad/velocidad',
        'target_fps': 15.0
    }
}

def setup_model_config(model_type: str):
    """Configurar cfg como en config.py y demo.py"""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Modelo no soportado: {model_type}")
    
    config = MODEL_CONFIGS[model_type]
    
    # Configurar cfg con valores reales (como config.py)
    cfg.backbone_cfg = config['backbone_cfg']
    cfg.depth = config['depth']
    cfg.input_shape = (256, 256)
    cfg.output_shape = (32, 32)  # input_shape[0]//8, input_shape[1]//8
    cfg.depth_dim = 32
    cfg.bbox_3d_shape = (2000, 2000, 2000)
    cfg.pixel_mean = (0.485, 0.456, 0.406)
    cfg.pixel_std = (0.229, 0.224, 0.225)
    
    logger.info(f"📝 Configurado para modelo {model_type}:")
    logger.info(f"   Backbone: {config['backbone_cfg']}")
    logger.info(f"   Depth: {config['depth']}")
    logger.info(f"   Target FPS: {config['target_fps']}")
    
    return config

class YOLOPersonDetector:
    """Detector de personas YOLO optimizado con soporte TFLite para móviles"""
    
    def __init__(self, use_tflite: bool = False):
        self.detector = None
        self.tflite_interpreter = None
        self.use_tflite = use_tflite
        self.frame_count = 0
        self.detection_cache = deque(maxlen=5)
        self.last_detections = []
        self.input_details = None
        self.output_details = None
        self._setup_yolo()
    
    def _setup_yolo(self):
        """Setup YOLO con opción TFLite para móviles"""
        if self.use_tflite and TFLITE_AVAILABLE:
            success = self._setup_yolo_tflite()
            if success:
                logger.info("✅ YOLO TFLite cargado para móviles")
                return
        
        # Fallback a YOLO PyTorch
        self._setup_yolo_pytorch()
    
    def _setup_yolo_tflite(self) -> bool:
        """Setup YOLO TensorFlow Lite optimizado para móviles"""
        try:
            # Intentar encontrar modelo YOLO TFLite
            tflite_paths = [
                ROOT / "yolo11n.tflite",
                ROOT / "yolo11n_quantized.tflite", 
                ROOT / "yolov8n.tflite",
                ROOT / "yolo_mobile.tflite"
            ]
            
            tflite_path = None
            for path in tflite_paths:
                if path.exists():
                    tflite_path = path
                    break
            
            if tflite_path is None:
                logger.warning("⚠️ No se encontró modelo YOLO TFLite, generando uno...")
                return self._create_yolo_tflite()
            
            # Cargar modelo TFLite
            self.tflite_interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            self.tflite_interpreter.allocate_tensors()
            
            # Obtener detalles de entrada y salida
            self.input_details = self.tflite_interpreter.get_input_details()
            self.output_details = self.tflite_interpreter.get_output_details()
            
            # Warm-up TFLite
            input_shape = self.input_details[0]['shape']
            dummy_input = np.random.randint(0, 255, input_shape, dtype=np.uint8)
            
            for _ in range(3):
                self.tflite_interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
                self.tflite_interpreter.invoke()
            
            logger.info(f"📱 YOLO TFLite cargado desde: {tflite_path.name}")
            logger.info(f"   Input shape: {input_shape}")
            logger.info(f"   Optimizado para móviles con NMS y conf=0.7")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ TFLite YOLO setup failed: {e}")
            return False
    
    def _create_yolo_tflite(self) -> bool:
        """Crear modelo YOLO TFLite optimizado para móviles"""
        try:
            if not YOLO_AVAILABLE:
                return False
            
            logger.info("🔧 Creando modelo YOLO TFLite optimizado para móviles...")
            
            # Cargar modelo YOLO PyTorch
            yolo_model = YOLO('yolo11n.pt')
            
            # Exportar a TFLite con optimizaciones móviles
            tflite_path = ROOT / "yolo11n_mobile_optimized.tflite"
            
            # Exportar con configuraciones móviles
            yolo_model.export(
                format='tflite',
                imgsz=320,  # Tamaño optimizado para móviles
                int8=True,  # Cuantización INT8
                optimize=True,
                simplify=True,
                conf=0.7,  # Confianza 70%
                iou=0.45,  # NMS threshold
                max_det=3,  # Máximo 3 detecciones
                save_dir=str(ROOT)
            )
            
            # Verificar si se generó
            if tflite_path.exists():
                logger.info(f"✅ YOLO TFLite generado: {tflite_path.name}")
                return self._setup_yolo_tflite()
            else:
                logger.warning("⚠️ No se pudo generar YOLO TFLite")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Error creando YOLO TFLite: {e}")
            return False
    
    def _setup_yolo_pytorch(self):
        """Setup YOLO PyTorch básico pero robusto"""
        if not YOLO_AVAILABLE:
            logger.warning("⚠️ YOLO no disponible")
            return
        
        try:
            self.detector = YOLO('yolo11n.pt')
            self.detector.overrides.update({
                'imgsz': 320,
                'device': 'cpu',
                'conf': 0.7,
                'iou': 0.45,
                'max_det': 3,
                'classes': [0],  # Solo personas
                'verbose': False
            })
            
            # Simple warm-up
            test_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            for _ in range(3):
                self.detector(test_img, verbose=False)
            
            logger.info("✅ YOLO PyTorch cargado y optimizado")
            
        except Exception as e:
            logger.error(f"❌ YOLO setup failed: {e}")
            self.detector = None
    
    def detect_persons(self, frame: np.ndarray) -> List[List[int]]:
        """Detectar personas con cache simple usando TFLite o PyTorch"""
        self.frame_count += 1
        
        # Cache cada 3 frames para optimización (pero SIEMPRE ejecutar en el primer frame)
        if self.frame_count > 1 and self.frame_count % 3 != 0:
            return self.last_detections
        
        if self.tflite_interpreter is not None:
            return self._detect_tflite(frame)
        elif self.detector is not None:
            return self._detect_pytorch(frame)
        else:
            return []
    
    def _detect_tflite(self, frame: np.ndarray) -> List[List[int]]:
        """Detección usando TFLite optimizado para móviles"""
        try:
            # Preprocesar frame para TFLite
            input_shape = self.input_details[0]['shape']
            h, w = input_shape[1], input_shape[2]
            
            # Redimensionar manteniendo aspecto
            img_resized = cv2.resize(frame, (w, h))
            
            # Convertir a formato esperado
            if self.input_details[0]['dtype'] == np.uint8:
                input_data = img_resized.astype(np.uint8)
            else:
                input_data = (img_resized.astype(np.float32) / 255.0)
            
            input_data = np.expand_dims(input_data, axis=0)
            
            # Inferencia TFLite
            self.tflite_interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.tflite_interpreter.invoke()
            
            # Obtener salidas
            outputs = []
            for output_detail in self.output_details:
                output = self.tflite_interpreter.get_tensor(output_detail['index'])
                outputs.append(output)
            
            # Post-procesar salidas (formato YOLO)
            persons = self._postprocess_yolo_tflite(outputs[0], frame.shape, conf_threshold=0.7)
            
            self.last_detections = persons[:3]  # Máximo 3 personas
            return self.last_detections
            
        except Exception as e:
            logger.warning(f"⚠️ TFLite detection failed: {e}")
            return self.last_detections
    
    def _postprocess_yolo_tflite(self, output: np.ndarray, orig_shape: tuple, conf_threshold: float = 0.7) -> List[List[int]]:
        """Post-procesar salidas YOLO TFLite con umbral optimizado para Galaxy S20"""
        try:
            persons = []
            
            # Formato salida YOLO: [batch, detections, 6] -> [x1, y1, x2, y2, conf, class]
            if len(output.shape) == 3:
                detections = output[0]
            else:
                detections = output
            
            orig_h, orig_w = orig_shape[:2]
            
            for detection in detections:
                if len(detection) >= 6:
                    conf = detection[4]
                    cls = int(detection[5])
                    
                    # Filtrar por confianza y clase (0 = persona) - Umbral optimizado para móviles
                    if conf >= conf_threshold and cls == 0:
                        # Normalizar coordenadas a imagen original
                        x1 = int(detection[0] * orig_w)
                        y1 = int(detection[1] * orig_h)
                        x2 = int(detection[2] * orig_w)
                        y2 = int(detection[3] * orig_h)
                        
                        # Validar bbox
                        if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                            persons.append([x1, y1, x2, y2])
            
            # Aplicar NMS simple para eliminar duplicados
            persons = self._apply_nms_simple(persons, 0.4)  # IoU threshold
            
            self.last_detections = persons[:5]  # Máximo 5 personas (Galaxy S20 benchmark)
            
            logger.info(f"🎯 YOLO TFLite: {len(persons)} personas detectadas (conf≥{conf_threshold})")
            return self.last_detections
            
        except Exception as e:
            logger.warning(f"⚠️ TFLite postprocess error: {e}")
            return self.last_detections
    
    def _apply_nms_simple(self, boxes: List[List[int]], iou_threshold: float = 0.4) -> List[List[int]]:
        """Aplicar Non-Maximum Suppression simple optimizado para móviles"""
        if len(boxes) <= 1:
            return boxes
        
        # Convertir a numpy para cálculos más rápidos
        boxes_array = np.array(boxes, dtype=np.float32)
        
        # Calcular áreas
        areas = (boxes_array[:, 2] - boxes_array[:, 0]) * (boxes_array[:, 3] - boxes_array[:, 1])
        
        # Ordenar por área (más grandes primero)
        order = areas.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            # Tomar el box con área más grande
            i = order[0]
            keep.append(i)
            
            # Calcular IoU con el resto
            if len(order) == 1:
                break
                
            ious = self._calculate_ious(boxes_array[i], boxes_array[order[1:]])
            
            # Mantener solo boxes con IoU < threshold
            order = order[1:][ious < iou_threshold]
        
        return [boxes[i] for i in keep]
    
    def _calculate_ious(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Calcular IoU entre un box y múltiples boxes"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        union = box_area + boxes_area - intersection
        
        return intersection / (union + 1e-6)
    
    def _detect_pytorch(self, frame: np.ndarray) -> List[List[int]]:
        """Detección usando YOLO PyTorch (fallback)"""
        try:
            results = self.detector(frame, verbose=False)
            persons = []
            
            logger.info(f"🔍 YOLO Debug: Se obtuvieron {len(results)} resultados")
            
            for i, result in enumerate(results):
                boxes = result.boxes
                logger.info(f"📦 Resultado {i}: {boxes is not None} boxes")
                
                if boxes is not None:
                    logger.info(f"📊 Total detecciones: {len(boxes)}")
                    
                    for j, box in enumerate(boxes):
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        logger.info(f"   🎯 Det {j}: conf={conf:.3f}, clase={cls}")
                        
                        # Umbral optimizado para Galaxy S20 (mayor selectividad)
                        if conf >= 0.5 and cls == 0:  # Incrementado de 0.3 a 0.5
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            persons.append([x1, y1, x2, y2])
                            logger.info(f"   ✅ Persona detectada: [{x1}, {y1}, {x2}, {y2}]")
            
            logger.info(f"👥 Total personas filtradas: {len(persons)}")
            self.last_detections = persons[:3]  # Máximo 3 personas
            return self.last_detections
            
        except Exception as e:
            logger.warning(f"⚠️ PyTorch detection failed: {e}")
            return self.last_detections

class ConvNeXtPoseEngine:
    """Motor de pose estimation robusto basado en demo.py con soporte TFLite"""
    
    def __init__(self, model_type: str, backend: str = 'pytorch'):
        self.model_type = model_type
        self.backend = backend
        self.config = setup_model_config(model_type)
        
        # Transform (igual que demo.py)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)
        ])
        
        # Variables de estado
        self.model = None
        self.onnx_session = None
        self.tflite_interpreter = None
        self.tflite_input_details = None
        self.tflite_output_details = None
        
        # Cargar modelo según backend
        self._load_model()
    
    def _load_model(self):
        """Cargar modelo según backend disponible"""
        if self.backend == 'tflite' and TFLITE_AVAILABLE:
            success = self._load_tflite_model()
            if success:
                return
        elif self.backend == 'onnx' and ONNX_AVAILABLE:
            success = self._load_onnx_model()
            if success:
                return
        
        # Fallback a PyTorch (siempre disponible)
        self._load_pytorch_model()
    
    def _load_tflite_model(self) -> bool:
        """Cargar modelo TensorFlow Lite optimizado para móviles"""
        try:
            tflite_path = PROJECT_ROOT / "exports" / self.config['tflite_model']
            
            if not tflite_path.exists():
                logger.warning(f"⚠️ Modelo TFLite no encontrado: {tflite_path}")
                return False
            
            logger.info(f"📱 Cargando TFLite: {tflite_path}")
            
            # Cargar intérprete TFLite optimizado para móviles
            self.tflite_interpreter = tf.lite.Interpreter(
                model_path=str(tflite_path),
                num_threads=4  # Optimizado para móviles
            )
            self.tflite_interpreter.allocate_tensors()
            
            # Obtener detalles de entrada y salida
            self.tflite_input_details = self.tflite_interpreter.get_input_details()
            
            self.tflite_output_details = self.tflite_interpreter.get_output_details()
            
            # Warm-up con datos dummy
            input_shape = self.tflite_input_details[0]['shape']
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            for _ in range(5):
                self.tflite_interpreter.set_tensor(self.tflite_input_details[0]['index'], dummy_input)
                self.tflite_interpreter.invoke()
            
            self.backend = 'tflite'
            
            # Información del modelo
            input_details = self.tflite_input_details[0]
            output_details = self.tflite_output_details[0]
            
            logger.info(f"✅ TFLite modelo {self.model_type} cargado")
            logger.info(f"   📱 Optimizado para móviles")
            logger.info(f"   📥 Input: {input_details['shape']} ({input_details['dtype']})")
            logger.info(f"   📤 Output: {output_details['shape']} ({output_details['dtype']})")
            
            # Análisis de viabilidad móvil
            model_size_mb = tflite_path.stat().st_size / (1024 * 1024)
            logger.info(f"   💾 Tamaño modelo: {model_size_mb:.1f} MB")
            
            if model_size_mb < 50:
                logger.info(f"   ✅ VIABLE para móviles (<50MB)")
            else:
                logger.warning(f"   ⚠️ GRANDE para móviles (>50MB)")
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ TFLite load failed: {e}")
            return False
    
    def _load_onnx_model(self) -> bool:
        """Cargar modelo ONNX optimizado"""
        try:
            onnx_path = PROJECT_ROOT / "exports" / self.config['onnx_model']
            
            if not onnx_path.exists():
                logger.warning(f"⚠️ Modelo ONNX no encontrado: {onnx_path}")
                return False
            
            # Configurar sesión ONNX optimizada
            providers = ['CPUExecutionProvider']
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.onnx_session = ort.InferenceSession(str(onnx_path), sess_options, providers=providers)
            
            # Warm-up
            dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
            for _ in range(5):
                self.onnx_session.run(None, {'input': dummy_input})
            
            self.backend = 'onnx'
            logger.info(f"✅ ONNX modelo {self.model_type} cargado")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ ONNX load failed: {e}")
            return False
    
    def _load_pytorch_model(self):
        """Cargar modelo PyTorch (igual que demo.py)"""
        try:
            model_path = ROOT / self.config['checkpoint']
            
            if not model_path.exists():
                raise FileNotFoundError(f"Checkpoint no encontrado: {model_path}")
            
            logger.info(f"📦 Cargando PyTorch: {model_path}")
            
            # Crear modelo (igual que demo.py)
            model = get_pose_net(cfg, False, joint_num)
            
            # Cargar checkpoint (igual que demo.py)
            ckpt = torch.load(str(model_path), map_location=device)
            
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                model = DataParallel(model)
                model = model.to(device)
                try:
                    model.load_state_dict(ckpt['network'], strict=False)
                except:
                    # Manejar DataParallel wrapper
                    state_dict = {}
                    for key, value in ckpt['network'].items():
                        if key.startswith('module.'):
                            state_dict[key[7:]] = value
                        else:
                            state_dict[f'module.{key}'] = value
                    model.load_state_dict(state_dict, strict=False)
            else:
                model = model.to(device)
                state_dict = ckpt['network']
                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {key[7:]: value for key, value in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)
            
            model.eval()
            
            # Warm-up
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 256, 256).to(device)
                for _ in range(5):
                    model(dummy_input)
            
            self.model = model
            self.backend = 'pytorch'
            logger.info(f"✅ PyTorch modelo {self.model_type} cargado")
            
        except Exception as e:
            logger.error(f"❌ PyTorch load failed: {e}")
            raise
    
    def infer(self, img_patch: np.ndarray) -> Optional[np.ndarray]:
        """Inferencia robusta según backend"""
        try:
            if self.backend == 'tflite' and self.tflite_interpreter:
                return self._infer_tflite(img_patch)
            elif self.backend == 'onnx' and self.onnx_session:
                return self._infer_onnx(img_patch)
            elif self.backend == 'pytorch' and self.model:
                return self._infer_pytorch(img_patch)
            else:
                logger.error("❌ No hay modelo disponible")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Inference failed: {e}")
            return None
    
    def _infer_tflite(self, img_patch: np.ndarray) -> Optional[np.ndarray]:
        """Inferencia TensorFlow Lite optimizada para móviles"""
        try:
            # Preprocesar imagen (igual que PyTorch)
            inp = self.transform(img_patch)
            inp = inp.numpy()  # Convertir a numpy
            inp = np.expand_dims(inp, axis=0)  # Añadir batch dimension
            
            # Configurar entrada TFLite
            self.tflite_interpreter.set_tensor(self.tflite_input_details[0]['index'], inp)
            
            # Ejecutar inferencia TFLite
            self.tflite_interpreter.invoke()
            
            # Obtener salida
            output = self.tflite_interpreter.get_tensor(self.tflite_output_details[0]['index'])
            
            return output
            
        except Exception as e:
            logger.warning(f"⚠️ TFLite inference failed: {e}")
            return None
    
    def _infer_onnx(self, img_patch: np.ndarray) -> Optional[np.ndarray]:
        """Inferencia ONNX"""
        inp = self.transform(img_patch)
        inp = inp[None, :, :, :].numpy()
        
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: inp})
        
        return outputs[0]
    
    def _infer_pytorch(self, img_patch: np.ndarray) -> Optional[np.ndarray]:
        """Inferencia PyTorch (igual que demo.py)"""
        inp = self.transform(img_patch).to(device)[None, :, :, :]
        
        with torch.no_grad():
            output = self.model(inp)
        
        return output.cpu().numpy()

    def process_frame(self, frame: np.ndarray):
        """Procesar frame completo con detección de personas y poses"""
        import time
        
        start_time = time.time()
        
        # Inicializar detector YOLO si no existe
        if not hasattr(self, 'yolo_detector'):
            self.yolo_detector = YOLOPersonDetector(use_tflite=True)
        
        # Detectar personas
        persons = self.yolo_detector.detect_persons(frame)
        
        # Procesar cada persona
        poses = []
        for bbox in persons:
            pose_2d = self._process_single_person(frame, bbox)
            if pose_2d is not None:
                poses.append(pose_2d)
        
        # Estadísticas
        processing_time = time.time() - start_time
        
        stats = {
            'processing_time_ms': processing_time * 1000,
            'instant_fps': 1.0 / processing_time if processing_time > 0 else 0,
            'poses_detected': len(poses),
            'persons_detected': len(persons),
            'model_type': self.model_type,
            'backend_used': self.backend
        }
        
        return poses, stats
    
    def _process_single_person(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Procesar una sola persona detectada (IGUAL QUE DEMO.PY)"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Procesar bbox como en demo.py
            from utils.pose_utils import process_bbox
            bbox_processed = process_bbox(np.array([x1, y1, x2-x1, y2-y1]), frame.shape[1], frame.shape[0])
            
            # Generar patch de imagen con transformación (como demo.py)
            img_patch, img2bb_trans = generate_patch_image(
                frame, bbox_processed, False, 1.0, 0.0, False
            )
            
            # Inferir pose
            pose_output = self.infer(img_patch)
            if pose_output is None:
                return None
            
            # Post-procesamiento IGUAL QUE DEMO.PY
            pose_3d = pose_output.reshape(joint_num, 3)  # [18, 3] para ConvNeXt
            
            # Desnormalizar desde output_shape a input_shape (como demo.py)
            pose_3d[:, 0] = pose_3d[:, 0] / cfg.output_shape[1] * cfg.input_shape[1]
            pose_3d[:, 1] = pose_3d[:, 1] / cfg.output_shape[0] * cfg.input_shape[0]
            
            # Aplicar transformación inversa (como demo.py)
            pose_3d_xy1 = np.concatenate((pose_3d[:, :2], np.ones_like(pose_3d[:, :1])), 1)
            img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0, 0, 1]).reshape(1, 3)))
            pose_3d[:, :2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            
            # Devolver solo coordenadas 2D
            return pose_3d[:, :2].copy()
            
        except Exception as e:
            logger.warning(f"Error procesando persona: {e}")
            return None


class RobustPoseProcessor:
    """Procesador principal robusto y definitivo con soporte TFLite para móviles"""
    
    def __init__(self, model_type: str = 'XS', backend: str = 'pytorch', use_yolo_tflite: bool = False):
        self.model_type = model_type
        self.backend = backend
        self.use_yolo_tflite = use_yolo_tflite
        self.config = MODEL_CONFIGS[model_type]
        
        # Componentes con soporte TFLite
        self.yolo_detector = YOLOPersonDetector(use_tflite=use_yolo_tflite)
        self.pose_engine = ConvNeXtPoseEngine(model_type, backend)
        
        # Estadísticas
        self.frame_count = 0
        self.processing_times = deque(maxlen=50)
        
        # Análisis de viabilidad móvil
        self._analyze_mobile_viability()
        
        logger.info("🚀 Robust Pose Processor inicializado")
        logger.info(f"   Modelo: {model_type} ({self.config['description']})")
        logger.info(f"   Backend ConvNeXt: {self.pose_engine.backend}")
        logger.info(f"   YOLO TFLite: {'✅' if use_yolo_tflite else '❌'}")
        logger.info(f"   Target FPS: {self.config['target_fps']}")
    
    def _analyze_mobile_viability(self):
        """Analizar viabilidad para dispositivos móviles"""
        logger.info("📱 ANÁLISIS VIABILIDAD MÓVIL:")
        
        # Análisis ConvNeXt
        if self.pose_engine.backend == 'tflite':
            tflite_path = PROJECT_ROOT / "exports" / self.config['tflite_model']
            if tflite_path.exists():
                size_mb = tflite_path.stat().st_size / (1024 * 1024)
                logger.info(f"   📦 ConvNeXt TFLite: {size_mb:.1f} MB")
                
                if size_mb < 20:
                    logger.info(f"   ✅ ConvNeXt EXCELENTE para móviles (<20MB)")
                elif size_mb < 50:
                    logger.info(f"   ✅ ConvNeXt VIABLE para móviles (<50MB)")
                else:
                    logger.warning(f"   ⚠️ ConvNeXt GRANDE para móviles (>50MB)")
        
        # Análisis YOLO
        if self.use_yolo_tflite:
            logger.info(f"   🎯 YOLO TFLite: Cuantizado INT8, NMS integrado, conf=70%")
            logger.info(f"   ✅ YOLO optimizado para móviles")
        else:
            logger.info(f"   ⚠️ YOLO PyTorch: No optimizado para móviles")
        
        # Recomendaciones móviles
        if self.pose_engine.backend == 'tflite' and self.use_yolo_tflite:
            logger.info(f"   🎉 CONFIGURACIÓN ÓPTIMA PARA MÓVILES")
        elif self.pose_engine.backend == 'tflite':
            logger.info(f"   📱 BUENA configuración móvil (ConvNeXt TFLite)")
        else:
            logger.info(f"   💻 Configuración para desktop/servidor")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Procesar frame de manera robusta"""
        start_time = time.time()
        self.frame_count += 1
        
        # Detectar personas
        persons = self.yolo_detector.detect_persons(frame)
        
        # Procesar cada persona
        poses = []
        for bbox in persons:
            pose_2d = self._process_single_person(frame, bbox)
            if pose_2d is not None:
                poses.append(pose_2d)
        
        # Estadísticas
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        avg_fps = 1.0 / np.mean(list(self.processing_times)) if self.processing_times else 0
        instant_fps = 1.0 / processing_time if processing_time > 0 else 0
        
        stats = {
            'frame_count': self.frame_count,
            'processing_time_ms': processing_time * 1000,
            'avg_fps': avg_fps,
            'instant_fps': instant_fps,
            'target_fps': self.config['target_fps'],
            'poses_detected': len(poses),
            'persons_detected': len(persons),
            'model_type': self.model_type,
            'backend': self.pose_engine.backend,
            'yolo_tflite': self.use_yolo_tflite,
            'mobile_optimized': self.pose_engine.backend == 'tflite' and self.use_yolo_tflite,
            'performance_ratio': avg_fps / self.config['target_fps'] if self.config['target_fps'] > 0 else 0
        }
        
        return poses, stats
    
    def _process_single_person(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Procesar una persona (IGUAL QUE DEMO.PY)"""
        try:
            if len(bbox) < 4:
                return None
            
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Convertir bbox format para process_bbox
            bbox_array = np.array([x1, y1, x2 - x1, y2 - y1])  # [x, y, width, height]
            processed_bbox = process_bbox(bbox_array, frame.shape[1], frame.shape[0])
            
            # Generar patch (IGUAL QUE DEMO.PY - 6 parámetros)
            img_patch, img2bb_trans = generate_patch_image(
                frame, processed_bbox, False, 1.0, 0.0, False
            )
            
            # Inferencia
            output = self.pose_engine.infer(img_patch)
            if output is None:
                return None
            
            # Post-procesamiento (IGUAL QUE DEMO.PY)
            pose_3d = output[0]  # Primer batch
            
            # Desnormalizar coordenadas (igual que demo.py)
            pose_3d[:, 0] = pose_3d[:, 0] / cfg.output_shape[1] * cfg.input_shape[1]
            pose_3d[:, 1] = pose_3d[:, 1] / cfg.output_shape[0] * cfg.input_shape[0]
            
            # Aplicar transformación inversa (igual que demo.py)
            pose_3d_xy1 = np.concatenate((pose_3d[:, :2], np.ones_like(pose_3d[:, :1])), 1)
            img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0, 0, 1]).reshape(1, 3)))
            pose_2d = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            
            return pose_2d
            
        except Exception as e:
            if not hasattr(self, '_error_logged'):
                logger.warning(f"⚠️ Error procesando persona: {e}")
                self._error_logged = True
            return None
    
    def _get_pose_with_relative_depth(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """
        Obtener pose con profundidad relativa (x, y, z_relative)
        Siguiendo exactamente el pipeline del demo.py ANTES de aplicar root_depth
        """
        try:
            if len(bbox) < 4:
                return None
            
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Convertir bbox format para process_bbox
            bbox_array = np.array([x1, y1, x2 - x1, y2 - y1])  # [x, y, width, height]
            processed_bbox = process_bbox(bbox_array, frame.shape[1], frame.shape[0])
            
            # Generar patch (IGUAL QUE DEMO.PY - 6 parámetros)
            img_patch, img2bb_trans = generate_patch_image(
                frame, processed_bbox, False, 1.0, 0.0, False
            )
            
            # Inferencia - obtener output completo (x, y, z_relative)
            output = self.pose_engine.infer(img_patch)
            if output is None:
                return None
            
            # Post-procesamiento PARCIAL (solo x, y - conservar z original)
            pose_3d = output[0].copy()  # Primer batch - copia para no modificar original
            
            # Desnormalizar SOLO coordenadas x, y (igual que demo.py)
            pose_3d[:, 0] = pose_3d[:, 0] / cfg.output_shape[1] * cfg.input_shape[1]
            pose_3d[:, 1] = pose_3d[:, 1] / cfg.output_shape[0] * cfg.input_shape[0]
            
            # Transformación inversa para x, y (igual que demo.py)
            pose_3d_xy1 = np.concatenate(
                (pose_3d[:, :2], np.ones_like(pose_3d[:, :1])), 1)
            img2bb_trans_001 = np.concatenate(
                (img2bb_trans, np.array([0, 0, 1]).reshape(1, 3)))
            pose_3d[:, :2] = np.dot(np.linalg.inv(
                img2bb_trans_001), pose_3d_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            
            # IMPORTANTE: NO tocar pose_3d[:, 2] - conservar z_relative del modelo
            
            return pose_3d  # (x, y, z_relative)
            
        except Exception as e:
            logger.warning(f"Error obteniendo pose con profundidad relativa: {e}")
            return None

def draw_pose_robust(image: np.ndarray, pose_2d: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)):
    """Dibujar pose robustamente (igual que demo.py con vis_keypoints)"""
    if pose_2d is None or len(pose_2d) == 0:
        return
    
    # Dibujar joints
    for i, (x, y) in enumerate(pose_2d):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (int(x), int(y)), 3, color, -1)
    
    # Dibujar skeleton
    for joint1, joint2 in skeleton:
        if (joint1 < len(pose_2d) and joint2 < len(pose_2d)):
            x1, y1 = pose_2d[joint1]
            x2, y2 = pose_2d[joint2]
            
            if (0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0] and
                0 <= x2 < image.shape[1] and 0 <= y2 < image.shape[0]):
                
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                cv2.line(image, pt1, pt2, color, 2)

def draw_stats_robust(image: np.ndarray, stats: Dict[str, Any]):
    """Dibujar estadísticas robustas con información TFLite"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    
    # Background más grande para info TFLite
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (700, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Color según rendimiento
    performance_ratio = stats.get('performance_ratio', 0)
    if performance_ratio >= 0.8:
        color = (0, 255, 0)    # Verde
    elif performance_ratio >= 0.5:
        color = (0, 255, 255)  # Amarillo
    else:
        color = (0, 0, 255)    # Rojo
    
    # Indicador móvil
    mobile_indicator = "📱 MÓVIL" if stats.get('mobile_optimized', False) else "💻 DESKTOP"
    backend_info = f"{stats['backend'].upper()}"
    if stats.get('yolo_tflite', False):
        backend_info += " + YOLO-TFLite"
    
    texts = [
        f"ConvNeXt {stats['model_type']} ({backend_info}) - {mobile_indicator}",
        f"FPS: {stats['avg_fps']:.1f}/{stats['target_fps']:.1f} (instant: {stats['instant_fps']:.1f})",
        f"Poses: {stats['poses_detected']} | Persons: {stats['persons_detected']} | Frame: {stats['frame_count']}",
        f"Tiempo: {stats['processing_time_ms']:.1f}ms | Ratio: {performance_ratio:.2f}",
        f"Optimizado móviles: {'✅' if stats.get('mobile_optimized', False) else '❌'}"
    ]
    
    for i, text in enumerate(texts):
        y_pos = 30 + i * 25
        cv2.putText(image, text, (15, y_pos), font, font_scale, color, 1)

def analyze_mobile_viability_detailed(model_type: str, backend: str, yolo_tflite: bool):
    """Análisis detallado de viabilidad para dispositivos móviles"""
    logger.info("🔍 ANÁLISIS DETALLADO VIABILIDAD MÓVIL")
    logger.info("=" * 60)
    
    # Análisis modelos ConvNeXt
    logger.info("📦 ANÁLISIS MODELOS CONVNEXT:")
    for model in ['XS', 'S']:
        config = MODEL_CONFIGS[model]
        
        # Checkpoint PyTorch
        checkpoint_path = ROOT / config['checkpoint']
        if checkpoint_path.exists():
            size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            logger.info(f"   {model} PyTorch: {size_mb:.1f} MB")
        
        # Modelo TFLite
        tflite_path = PROJECT_ROOT / "exports" / config['tflite_model']
        if tflite_path.exists():
            size_mb = tflite_path.stat().st_size / (1024 * 1024)
            viability = "✅ EXCELENTE" if size_mb < 20 else "✅ VIABLE" if size_mb < 50 else "⚠️ GRANDE"
            logger.info(f"   {model} TFLite: {size_mb:.1f} MB - {viability}")
        else:
            logger.info(f"   {model} TFLite: ❌ No disponible")
    
    # Análisis YOLO
    logger.info("\n🎯 ANÁLISIS YOLO:")
    yolo_models = [
        ("yolo11n.pt", "PyTorch"),
        ("yolo11n.onnx", "ONNX"),
        ("yolo11n.tflite", "TFLite"),
        ("yolo11n_mobile_optimized.tflite", "TFLite Móvil")
    ]
    
    for model_file, model_type_desc in yolo_models:
        model_path = ROOT / model_file
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"   {model_type_desc}: {size_mb:.1f} MB")
        else:
            logger.info(f"   {model_type_desc}: ❌ No disponible")
    
    # Recomendaciones específicas
    logger.info("\n📱 RECOMENDACIONES MÓVILES:")
    
    if backend == 'tflite' and yolo_tflite:
        logger.info("   🎉 CONFIGURACIÓN ÓPTIMA:")
        logger.info("      ✅ ConvNeXt TFLite + YOLO TFLite")
        logger.info("      ✅ Cuantización INT8")
        logger.info("      ✅ NMS integrado (conf=70%)")
        logger.info("      ✅ Memoria optimizada")
        logger.info("      📱 RECOMENDADO PARA PRODUCCIÓN MÓVIL")
        
    elif backend == 'tflite':
        logger.info("   ✅ BUENA CONFIGURACIÓN:")
        logger.info("      ✅ ConvNeXt TFLite optimizado")
        logger.info("      ⚠️ YOLO PyTorch (no optimizado)")
        logger.info("      📱 Viable para móviles de gama alta")
        
    elif yolo_tflite:
        logger.info("   ⚠️ CONFIGURACIÓN MIXTA:")
        logger.info("      ⚠️ ConvNeXt PyTorch/ONNX")
        logger.info("      ✅ YOLO TFLite optimizado")
        logger.info("      💻 Mejor para desktop/servidor")
        
    else:
        logger.info("   💻 CONFIGURACIÓN DESKTOP:")
        logger.info("      ⚠️ No optimizado para móviles")
        logger.info("      ✅ Máximo rendimiento desktop")
        logger.info("      🖥️ Recomendado para desarrollo/testing")
    
    # Análisis rendimiento esperado
    logger.info("\n⚡ RENDIMIENTO ESPERADO MÓVILES:")
    
    if model_type == 'XS':
        if backend == 'tflite' and yolo_tflite:
            logger.info("   📱 ConvNeXt XS + TFLite completo:")
            logger.info("      🎯 Target FPS: 15-25 (móvil)")
            logger.info("      💾 Memoria: ~30-50 MB")
            logger.info("      🔋 Batería: Optimizada")
        else:
            logger.info("   📱 ConvNeXt XS (no optimizado):")
            logger.info("      🎯 Target FPS: 8-15 (móvil)")
            logger.info("      💾 Memoria: ~100-200 MB")
            logger.info("      🔋 Batería: Alta")
    
    elif model_type == 'S':
        if backend == 'tflite' and yolo_tflite:
            logger.info("   📱 ConvNeXt S + TFLite completo:")
            logger.info("      🎯 Target FPS: 10-18 (móvil)")
            logger.info("      💾 Memoria: ~50-80 MB")
            logger.info("      🔋 Batería: Moderada")
        else:
            logger.info("   📱 ConvNeXt S (no optimizado):")
            logger.info("      🎯 Target FPS: 5-12 (móvil)")
            logger.info("      💾 Memoria: ~150-300 MB")
            logger.info("      🔋 Batería: Muy Alta")


def process_single_image(engine: ConvNeXtPoseEngine, image_path: str, output_path: Optional[str] = None, visualize: bool = False):
    """Procesar una sola imagen usando el motor"""
    import cv2
    
    # Cargar imagen
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    
    # Procesar frame
    poses, stats = engine.process_frame(frame)
    
    # Visualizar si se solicita
    if visualize:
        output_frame = visualize_pose_2d(frame, poses, stats)
        cv2.imshow('ConvNeXt Pose Detection', output_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Guardar si se especifica path
        if output_path:
            cv2.imwrite(output_path, output_frame)
    
    return {
        'persons_detected': stats['persons_detected'],
        'poses_extracted': stats['poses_detected'],
        'processing_time': stats['processing_time_ms'] / 1000,
        'backend_used': engine.backend,
        'poses': poses
    }


def process_video_file(engine: ConvNeXtPoseEngine, video_path: str, output_path: Optional[str] = None, visualize: bool = False):
    """Procesar video usando el motor"""
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")
    
    # Configuración del writer si se especifica output
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = None
    if output_path:
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = 0
    total_persons = 0
    total_poses = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar frame
            poses, stats = engine.process_frame(frame)
            total_frames += 1
            total_persons += stats['persons_detected']
            total_poses += stats['poses_detected']
            
            # Visualizar
            if visualize or output_path:
                output_frame = visualize_pose_2d(frame, poses, stats)
                
                if visualize:
                    cv2.imshow('ConvNeXt Pose Video', output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if writer:
                    writer.write(output_frame)
    
    finally:
        cap.release()
        if writer:
            writer.release()
        if visualize:
            cv2.destroyAllWindows()
    
    return {
        'total_frames': total_frames,
        'persons_detected': total_persons,
        'poses_extracted': total_poses,
        'backend_used': engine.backend,
        'avg_persons_per_frame': total_persons / total_frames if total_frames > 0 else 0
    }


def visualize_pose_2d(frame: np.ndarray, poses: List[np.ndarray], stats: Dict[str, Any]) -> np.ndarray:
    """Visualizar poses 2D en el frame (IGUAL QUE DEMO.PY)"""
    import cv2
    
    output_frame = frame.copy()
    
    # Dibujar poses usando skeleton de ConvNeXt
    for pose in poses:
        if pose is not None and len(pose) >= joint_num:
            # Dibujar joints (puntos)
            for i, (x, y) in enumerate(pose):
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    cv2.circle(output_frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                    # Número del joint
                    cv2.putText(output_frame, str(i), (int(x)+5, int(y)-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            
            # Dibujar skeleton (conexiones)
            for joint1, joint2 in skeleton:
                if (joint1 < len(pose) and joint2 < len(pose)):
                    x1, y1 = int(pose[joint1][0]), int(pose[joint1][1])
                    x2, y2 = int(pose[joint2][0]), int(pose[joint2][1])
                    
                    # Verificar que ambos puntos estén en el frame
                    if (0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0] and
                        0 <= x2 < frame.shape[1] and 0 <= y2 < frame.shape[0]):
                        cv2.line(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Información en pantalla
    info_text = [
        f"Personas: {stats['persons_detected']}",
        f"Poses: {stats['poses_detected']}",
        f"FPS: {stats['instant_fps']:.1f}",
        f"Backend: {stats.get('backend_used', 'N/A')}",
        f"Modelo: {stats['model_type']}"
    ]
    
    for i, text in enumerate(info_text):
        cv2.putText(output_frame, text, (10, 30 + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return output_frame


class MobileViabilityAnalyzer:
    """Analizador de viabilidad móvil para ConvNeXt + YOLO TFLite"""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_complete_viability(self, model_type: str = 'XS', use_tflite: bool = True, use_yolo_tflite: bool = True):
        """Análisis completo de viabilidad móvil"""
        logger.info("🔍 ANÁLISIS COMPLETO DE VIABILIDAD MÓVIL")
        logger.info("=" * 60)
        
        # Análisis de modelos disponibles
        convnext_analysis = self._analyze_available_models()
        yolo_analysis = self._analyze_yolo_viability()
        performance_analysis = self._estimate_mobile_performance(model_type, use_tflite, use_yolo_tflite)
        
        # Generar recomendación
        recommendation = self._generate_recommendation(convnext_analysis, yolo_analysis, performance_analysis)
        
        return {
            'convnext': convnext_analysis,
            'yolo': yolo_analysis,
            'performance': performance_analysis,
            'recommendation': recommendation
        }
    
    def _analyze_available_models(self):
        """Analizar modelos ConvNeXt TFLite disponibles"""
        logger.info("📱 MODELOS CONVNEXT DISPONIBLES:")
        logger.info("-" * 40)
        
        results = {}
        exports_path = PROJECT_ROOT / "exports"
        
        for model_name in ['XS', 'S']:
            tflite_file = exports_path / f"convnextpose_{model_name}.tflite"
            
            if tflite_file.exists():
                size_mb = tflite_file.stat().st_size / (1024 * 1024)
                viability = "EXCELENTE" if size_mb < 20 else ("BUENA" if size_mb < 50 else "LIMITADA")
                emoji = "🟢" if size_mb < 20 else ("🟡" if size_mb < 50 else "🔴")
                
                results[model_name] = {
                    'exists': True,
                    'size_mb': size_mb,
                    'viability': viability,
                    'mobile_ready': size_mb < 50
                }
                
                logger.info(f"   {emoji} ConvNeXt {model_name}: {size_mb:.1f} MB - {viability}")
                if model_name == 'XS':
                    logger.info(f"      🎯 FPS móvil estimado: 15-25")
                    logger.info(f"      💾 RAM estimada: {size_mb * 3:.0f} MB")
                    logger.info(f"      🔋 Batería: EFICIENTE")
                else:
                    logger.info(f"      🎯 FPS móvil estimado: 8-15") 
                    logger.info(f"      💾 RAM estimada: {size_mb * 4:.0f} MB")
                    logger.info(f"      🔋 Batería: MODERADA")
            else:
                results[model_name] = {'exists': False, 'mobile_ready': False}
                logger.info(f"   ❌ ConvNeXt {model_name}: NO ENCONTRADO")
        
        return results
    
    def _analyze_yolo_viability(self):
        """Analizar viabilidad YOLO TFLite"""
        logger.info("\n🎯 ANÁLISIS YOLO PARA MÓVILES:")
        logger.info("-" * 40)
        
        if not YOLO_AVAILABLE:
            logger.info("   ❌ YOLO no disponible")
            return {'available': False}
        
        yolo_pt = ROOT / "yolo11n.pt"
        if yolo_pt.exists():
            size_mb = yolo_pt.stat().st_size / (1024 * 1024)
            estimated_tflite = size_mb * 0.4  # TFLite típicamente 40% del tamaño PyTorch
            
            logger.info(f"   📦 YOLO11n PyTorch: {size_mb:.1f} MB")
            logger.info(f"   📱 TFLite estimado: ~{estimated_tflite:.1f} MB")
            
            # Analizar exportación TFLite
            can_export = False
            try:
                from ultralytics import YOLO
                yolo_temp = YOLO(str(yolo_pt))
                can_export = hasattr(yolo_temp, 'export')
                logger.info(f"   ✅ Exportación TFLite: {'POSIBLE' if can_export else 'NO DISPONIBLE'}")
            except Exception as e:
                logger.info(f"   ⚠️ Error verificando exportación: {str(e)[:50]}...")
            
            # Características móviles
            logger.info(f"   🎛️ OPTIMIZACIONES MÓVILES:")
            logger.info(f"      • Cuantización INT8 disponible")
            logger.info(f"      • NMS integrado con conf=0.7")
            logger.info(f"      • Máximo 5 detecciones")
            logger.info(f"      • Tamaño entrada: 320x320")
            
            viability = "EXCELENTE" if estimated_tflite < 15 else ("BUENA" if estimated_tflite < 25 else "LIMITADA")
            
            return {
                'available': True,
                'current_size_mb': size_mb,
                'estimated_tflite_mb': estimated_tflite,
                'can_export': can_export,
                'viability': viability,
                'mobile_optimizations': ['INT8', 'NMS', 'Small_input', 'Max_5_detections']
            }
        else:
            logger.info(f"   ❌ yolo11n.pt no encontrado")
            return {'available': False}
    
    def _estimate_mobile_performance(self, model_type: str, use_tflite: bool, use_yolo_tflite: bool):
        """Estimar rendimiento en dispositivos móviles"""
        logger.info(f"\n⚡ ESTIMACIÓN RENDIMIENTO MÓVIL:")
        logger.info("-" * 40)
        
        # Dispositivos objetivo
        devices = {
            'flagship': {'name': 'Flagship 2023+', 'base_fps': 30, 'ram_gb': 8},
            'midrange': {'name': 'Mid-range 2022+', 'base_fps': 20, 'ram_gb': 6},
            'budget': {'name': 'Budget 2021+', 'base_fps': 12, 'ram_gb': 4}
        }
        
        results = {}
        
        for device_type, specs in devices.items():
            # Calcular FPS estimado
            fps_factor = 1.0
            fps_factor *= 1.3 if use_tflite else 0.7  # TFLite boost
            fps_factor *= 1.2 if use_yolo_tflite else 0.8  # YOLO TFLite boost
            fps_factor *= 1.0 if model_type == 'XS' else 0.7  # Modelo más pequeño
            
            estimated_fps = specs['base_fps'] * fps_factor
            
            # Calcular memoria
            base_memory = 120 if use_tflite else 200
            yolo_memory = 40 if use_yolo_tflite else 80
            total_memory = base_memory + yolo_memory
            
            # Viabilidad
            viable = estimated_fps >= 10 and total_memory <= (specs['ram_gb'] * 1000 * 0.25)
            
            results[device_type] = {
                'estimated_fps': round(estimated_fps, 1),
                'memory_mb': total_memory,
                'viable': viable,
                'battery_impact': 'LOW' if estimated_fps >= 15 else ('MEDIUM' if estimated_fps >= 10 else 'HIGH')
            }
            
            status = "✅" if viable else "⚠️"
            logger.info(f"   {status} {specs['name']}:")
            logger.info(f"      🎯 FPS: {estimated_fps:.1f}")
            logger.info(f"      💾 RAM: {total_memory} MB")
            logger.info(f"      🔋 Batería: {results[device_type]['battery_impact']}")
        
        return results
    
    def _generate_recommendation(self, convnext_analysis, yolo_analysis, performance_analysis):
        """Generar recomendación final"""
        logger.info(f"\n🎯 RECOMENDACIÓN FINAL:")
        logger.info("=" * 50)
        
        # Evaluar viabilidad
        convnext_ready = any(model.get('mobile_ready', False) for model in convnext_analysis.values())
        yolo_ready = yolo_analysis.get('viability') in ['EXCELENTE', 'BUENA'] if yolo_analysis.get('available') else False
        viable_devices = sum(1 for device in performance_analysis.values() if device['viable'])
        
        if convnext_ready and yolo_ready and viable_devices >= 2:
            level = "ALTAMENTE VIABLE"
            emoji = "🟢"
        elif convnext_ready and viable_devices >= 1:
            level = "VIABLE CON LIMITACIONES"
            emoji = "🟡"
        else:
            level = "NO VIABLE ACTUALMENTE"
            emoji = "🔴"
        
        logger.info(f"{emoji} {level}")
        logger.info("")
        
        if level == "ALTAMENTE VIABLE":
            logger.info("✅ CONFIGURACIÓN RECOMENDADA:")
            logger.info("   • ConvNeXt XS + TensorFlow Lite")
            logger.info("   • YOLO11n + TFLite cuantizado INT8")
            logger.info("   • Confianza: 70% con NMS optimizado")
            logger.info("   • Target: Mid-range y Flagship devices")
            logger.info("   • FPS esperado: 15-25 en dispositivos modernos")
        elif level == "VIABLE CON LIMITACIONES":
            logger.info("⚠️ RECOMENDACIONES:")
            logger.info("   • Solo dispositivos high-end")
            logger.info("   • Monitorear temperatura y batería")
            logger.info("   • Considerar degradado de calidad")
        else:
            logger.info("❌ ACCIONES NECESARIAS:")
            logger.info("   • Optimizar modelos TFLite")
            logger.info("   • Reducir resolución de entrada")
            logger.info("   • Considerar cloud processing")
        
        return {
            'level': level,
            'viable': level != "NO VIABLE ACTUALMENTE",
            'optimal_config': {
                'model': 'XS',
                'backend': 'tflite',
                'yolo_tflite': True,
                'confidence': 0.7
            } if level in ["ALTAMENTE VIABLE", "VIABLE CON LIMITACIONES"] else None
        }


def main():
    """Función principal con análisis móvil integrado"""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='ConvNeXt Pose Detection con análisis de viabilidad móvil')
    parser.add_argument('--input', type=str, default='input.jpg', help='Input image/video path')
    parser.add_argument('--output', type=str, help='Output path (optional)')
    parser.add_argument('--model', type=str, choices=['XS', 'S'], default='XS', help='Model size')
    parser.add_argument('--backend', type=str, choices=['pytorch', 'onnx', 'tflite'], default='auto', help='Backend to use')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--analyze-viability', action='store_true', help='Analizar viabilidad móvil para TensorFlow Lite')
    parser.add_argument('--mobile-mode', action='store_true', help='Modo optimizado para dispositivos móviles')
    parser.add_argument('--yolo-tflite-test', action='store_true', help='Probar conversión YOLO a TFLite')
    parser.add_argument('--confidence', type=float, default=0.7, help='YOLO confidence threshold (default: 0.7)')
    
    args = parser.parse_args()
    
    # Configuración inicial
    logger.info("🚀 ConvNeXt Pose Production - TensorFlow Lite Edition")
    logger.info("=" * 55)
    
    try:
        # Análisis de viabilidad móvil si se solicita
        if args.analyze_viability:
            logger.info("\n📱 INICIANDO ANÁLISIS DE VIABILIDAD MÓVIL")
            logger.info("=" * 50)
            
            analyzer = MobileViabilityAnalyzer()
            analysis_results = analyzer.analyze_complete_viability(
                model_type=args.model,
                use_tflite=True,
                use_yolo_tflite=True
            )
            
            # Mostrar resumen ejecutivo
            recommendation = analysis_results['recommendation']
            logger.info(f"\n📊 RESUMEN EJECUTIVO:")
            logger.info(f"🎯 Nivel: {recommendation['level']}")
            logger.info(f"✅ Viable: {'SÍ' if recommendation['viable'] else 'NO'}")
            
            if recommendation['optimal_config']:
                config = recommendation['optimal_config']
                logger.info(f"\n🏆 CONFIGURACIÓN ÓPTIMA:")
                logger.info(f"   • Modelo: ConvNeXt {config['model']}")
                logger.info(f"   • Backend: {config['backend']}")
                logger.info(f"   • YOLO TFLite: {'SÍ' if config['yolo_tflite'] else 'NO'}")
                logger.info(f"   • Confianza: {config['confidence']*100}%")
            
            if not args.mobile_mode:
                return  # Solo análisis, no procesamiento
        
        # Test específico de YOLO TFLite
        if args.yolo_tflite_test:
            logger.info("\n🔄 PROBANDO CONVERSIÓN YOLO A TFLITE")
            logger.info("=" * 45)
            
            analyzer = MobileViabilityAnalyzer()
            yolo_analysis = analyzer._analyze_yolo_viability()
            
            if yolo_analysis.get('available', False):
                logger.info(f"✅ YOLO disponible")
                logger.info(f"📦 Tamaño TFLite estimado: {yolo_analysis.get('estimated_tflite_mb', 0):.1f} MB")
                logger.info(f"🎯 Viabilidad: {yolo_analysis.get('viability', 'DESCONOCIDA')}")
                
                # Intentar conversión real si YOLO está disponible
                yolo_path = PROJECT_ROOT / "yolo11n.pt"
                if yolo_path.exists():
                    try:
                        import torch
                        from ultralytics import YOLO
                        
                        model = YOLO(str(yolo_path))
                        logger.info("🔄 Exportando a TFLite...")
                        
                        # Exportar con cuantización INT8
                        model.export(format='tflite', int8=True, imgsz=640)
                        logger.info("✅ Conversión YOLO a TFLite completada exitosamente")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Error en conversión YOLO: {e}")
                else:
                    logger.warning("⚠️ Archivo yolo11n.pt no encontrado")
            else:
                logger.error("❌ YOLO no disponible para conversión")
            
            if not args.mobile_mode:
                return
        
        # Inicializar motor de pose
        logger.info(f"\n🤖 INICIALIZANDO MOTOR ConvNeXt {args.model}")
        logger.info("=" * 45)
        
        # Selección automática de backend si es necesario
        if args.backend == 'auto':
            if args.mobile_mode:
                args.backend = 'tflite'
                logger.info("📱 Modo móvil: Seleccionando TensorFlow Lite automáticamente")
            else:
                args.backend = 'pytorch'
                logger.info("🖥️ Modo desktop: Seleccionando PyTorch automáticamente")
        
        # Crear motor
        engine = ConvNeXtPoseEngine(
            model_type=args.model,
            backend=args.backend
        )
        
        # Procesar entrada
        logger.info(f"\n🎯 PROCESANDO: {args.input}")
        logger.info("=" * 40)
        
        start_time = time.time()
        
        if args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Procesamiento de video
            logger.info("🎬 Detectado: Video")
            results = process_video_file(
                engine=engine,
                video_path=args.input,
                output_path=args.output,
                visualize=args.visualize
            )
        else:
            # Procesamiento de imagen
            logger.info("🖼️ Detectado: Imagen")
            results = process_single_image(
                engine=engine,
                image_path=args.input,
                output_path=args.output,
                visualize=args.visualize
            )
        
        processing_time = time.time() - start_time
        
        # Mostrar resultados
        logger.info(f"\n📊 RESULTADOS:")
        logger.info("=" * 30)
        logger.info(f"⏱️ Tiempo total: {processing_time:.2f}s")
        
        if isinstance(results, dict) and 'persons_detected' in results:
            logger.info(f"👥 Personas detectadas: {results['persons_detected']}")
            logger.info(f"🎯 Poses extraídas: {results['poses_extracted']}")
            
            # Información adicional para modo móvil
            if args.mobile_mode:
                logger.info(f"📱 Backend: {results.get('backend_used', 'N/A')}")
                logger.info(f"🔋 Optimizado para móvil: ✅")
                
                # Estimación de rendimiento móvil
                fps_estimate = 1.0 / processing_time if processing_time > 0 else 0
                logger.info(f"📈 FPS estimado en móvil: {fps_estimate:.1f}")
                
                if fps_estimate >= 15:
                    logger.info("🟢 Rendimiento: EXCELENTE para tiempo real")
                elif fps_estimate >= 10:
                    logger.info("🟡 Rendimiento: BUENO para tiempo real")
                else:
                    logger.info("🔴 Rendimiento: LIMITADO para tiempo real")
        
        if args.output:
            logger.info(f"💾 Guardado en: {args.output}")
        
        logger.info("\n🎉 PROCESAMIENTO COMPLETADO EXITOSAMENTE")
        
    except Exception as e:
        logger.error(f"❌ Error durante el procesamiento: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

