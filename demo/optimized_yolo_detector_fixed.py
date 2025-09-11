#!/usr/bin/env python3
"""
OptimizedYOLODetector FIXED - Sin TorchScript para máximo rendimiento
====================================================================
Versión corregida que usa PyTorch nativo sin compilation overhead
Configurado para NMS=True, conf=0.7, max 1 persona
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class OptimizedYOLODetectorFixed:
    """
    Detector YOLO optimizado SIN TorchScript para máximo rendimiento
    
    Características:
    - PyTorch nativo (sin compilation overhead)
    - NMS automático integrado
    - Confianza configurable (default 0.7)
    - Máximo 1 persona por detección
    - Optimizado para velocidad
    """
    
    def __init__(self, 
                 conf_threshold: float = 0.7,
                 max_persons: int = 1,
                 nms_threshold: float = 0.4):
        
        self.conf_threshold = conf_threshold
        self.max_persons = max_persons
        self.nms_threshold = nms_threshold
        self.model = None
        self.is_torchscript = False  # Siempre False en esta versión
        
        logger.info("🚀 INICIANDO YOLO OPTIMIZADO FIXED")
        logger.info(f"   🎯 Confianza: {conf_threshold}")
        logger.info(f"   👤 Máx personas: {max_persons}")
        logger.info(f"   🔧 NMS threshold: {nms_threshold}")
        logger.info(f"   ⚡ Modo: PyTorch nativo (sin TorchScript)")
        
        self._initialize_yolo()
    
    def _initialize_yolo(self):
        """Inicializar YOLO con PyTorch nativo"""
        try:
            from ultralytics import YOLO
            
            logger.info("📥 Cargando YOLO PyTorch nativo...")
            
            # Cargar modelo PyTorch nativo (NO TorchScript)
            self.model = YOLO('yolo11n.pt')
            
            # Configurar parámetros de optimización
            self.model.conf = self.conf_threshold  # Threshold de confianza
            self.model.iou = self.nms_threshold    # Threshold NMS
            self.model.max_det = self.max_persons  # Máximo detecciones
            
            logger.info("✅ YOLO PyTorch nativo cargado exitosamente")
            logger.info(f"   📊 Configuración: conf={self.conf_threshold}, iou={self.nms_threshold}")
            logger.info(f"   🎯 Máx detecciones: {self.max_persons}")
            
        except ImportError as e:
            logger.error(f"❌ Ultralytics no disponible: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error inicializando YOLO: {e}")
            raise
    
    def detect_person(self, image: np.ndarray) -> Tuple[Optional[List[int]], float]:
        """
        Detectar exactamente 1 persona con máximo rendimiento
        
        Args:
            image: Imagen de entrada (BGR format)
            
        Returns:
            bbox: [x1, y1, x2, y2] o None si no hay detección
            confidence: Confianza de la detección (0-1)
        """
        if self.model is None:
            logger.warning("⚠️ Modelo YOLO no inicializado")
            return None, 0.0
        
        try:
            # Inferencia directa con configuraciones optimizadas
            start_time = time.time()
            
            # Ejecutar YOLO con configuraciones optimizadas
            results = self.model.predict(
                source=image,
                conf=self.conf_threshold,
                iou=self.nms_threshold,
                max_det=self.max_persons,
                classes=[0],  # Solo personas (class 0)
                verbose=False,
                save=False,
                show=False
            )
            
            inference_time = (time.time() - start_time) * 1000
            
            # Procesar resultados
            if not results or len(results) == 0:
                logger.debug(f"🔍 No hay resultados YOLO - {inference_time:.1f}ms")
                return None, 0.0
            
            result = results[0]  # Primera imagen
            
            if result.boxes is None or len(result.boxes) == 0:
                logger.debug(f"🔍 No hay detecciones - {inference_time:.1f}ms")
                return None, 0.0
            
            # Tomar la primera (y probablemente única) detección
            box = result.boxes[0]
            
            # Extraer datos
            bbox_coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
            confidence = float(box.conf[0].cpu().numpy())
            
            logger.debug(f"✅ Persona detectada: bbox={bbox_coords}, conf={confidence:.3f}, tiempo={inference_time:.1f}ms")
            
            return bbox_coords, confidence
            
        except Exception as e:
            logger.error(f"❌ Error en detección YOLO: {e}")
            return None, 0.0
    
    def detect_multiple_persons(self, image: np.ndarray) -> List[dict]:
        """
        Detectar múltiples personas (para compatibilidad, pero limitado a max_persons)
        
        Returns:
            Lista de detecciones: [{'bbox': [x1,y1,x2,y2], 'confidence': float}, ...]
        """
        bbox, confidence = self.detect_person(image)
        
        if bbox is None:
            return []
        
        return [{
            'bbox': bbox,
            'confidence': confidence,
            'class': 0  # Persona
        }]
    
    def benchmark(self, test_image: np.ndarray, iterations: int = 100) -> dict:
        """Benchmark del detector"""
        logger.info(f"🧪 BENCHMARK YOLO FIXED - {iterations} iteraciones")
        
        times = []
        detections = 0
        
        for i in range(iterations):
            start_time = time.time()
            bbox, conf = self.detect_person(test_image)
            elapsed = time.time() - start_time
            
            times.append(elapsed * 1000)  # ms
            if bbox is not None:
                detections += 1
        
        avg_time = np.mean(times)
        fps = 1000 / avg_time if avg_time > 0 else 0
        detection_rate = (detections / iterations) * 100
        
        results = {
            'avg_time_ms': avg_time,
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'fps': fps,
            'detection_rate': detection_rate,
            'total_detections': detections,
            'iterations': iterations
        }
        
        logger.info("📊 RESULTADOS BENCHMARK YOLO:")
        logger.info(f"   ⏱️ Tiempo promedio: {avg_time:.1f}ms")
        logger.info(f"   📈 FPS: {fps:.1f}")
        logger.info(f"   ✅ Tasa detección: {detection_rate:.1f}%")
        logger.info(f"   🎯 Min/Max: {results['min_time_ms']:.1f}ms / {results['max_time_ms']:.1f}ms")
        
        return results

def main():
    """Test del detector optimizado"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='input.jpg', help='Imagen de prueba')
    parser.add_argument('--benchmark', action='store_true', help='Ejecutar benchmark')
    parser.add_argument('--iterations', type=int, default=50, help='Iteraciones de benchmark')
    parser.add_argument('--conf', type=float, default=0.7, help='Confianza threshold')
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Crear detector
        detector = OptimizedYOLODetectorFixed(
            conf_threshold=args.conf,
            max_persons=1
        )
        
        # Cargar imagen de prueba
        if Path(args.input).exists():
            test_image = cv2.imread(args.input)
            logger.info(f"📸 Imagen cargada: {args.input}")
        else:
            test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            logger.info("📸 Usando imagen sintética")
        
        if args.benchmark:
            # Ejecutar benchmark
            results = detector.benchmark(test_image, args.iterations)
        else:
            # Test único
            logger.info("🧪 Test único...")
            start_time = time.time()
            bbox, conf = detector.detect_person(test_image)
            elapsed = (time.time() - start_time) * 1000
            
            if bbox:
                logger.info(f"✅ Persona detectada: {bbox}, confianza: {conf:.3f}")
                logger.info(f"⏱️ Tiempo: {elapsed:.1f}ms")
            else:
                logger.info("❌ No se detectó persona")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
