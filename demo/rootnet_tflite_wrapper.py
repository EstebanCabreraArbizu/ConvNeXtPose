#!/usr/bin/env python3
"""
RootNet TFLite Integration Wrapper
=================================
Wrapper optimizado que usa el modelo TFLite de RootNet
para máximo rendimiento en ConvNeXtPose.
"""

import os
import sys
import time
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import logging

# Configurar paths
CONVNEXT_ROOT = Path("/home/user/convnextpose_esteban/ConvNeXtPose/")
sys.path.insert(0, str(CONVNEXT_ROOT / "demo"))

logger = logging.getLogger(__name__)

class RootNetTFLiteWrapper:
    """Wrapper optimizado usando TFLite para máximo rendimiento"""
    
    def __init__(self, model_variant="size"):
        """
        Inicializar wrapper con modelo TFLite
        
        Args:
            model_variant: "default" (23MB), "size" (45MB, más rápido), o "latency" (23MB)
        """
        self.model_variant = model_variant
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.backbone_available = False
        
        # Configuración por defecto
        self.default_focal = [1500, 1500]
        
        # Parámetros para estimación heurística
        self.person_height_mm = 1700
        self.camera_height_mm = 1600
        
        # Estadísticas de rendimiento
        self.inference_times = []
        
        # Cargar modelo automáticamente
        self.load_tflite_model()
    
    def load_tflite_model(self):
        """Cargar modelo TFLite especificado"""
        model_paths = {
            "default": CONVNEXT_ROOT / "exports/rootnet_backbone_default.tflite",
            "size": CONVNEXT_ROOT / "exports/rootnet_backbone_size.tflite", 
            "latency": CONVNEXT_ROOT / "exports/rootnet_backbone_latency.tflite"
        }
        
        model_path = model_paths.get(self.model_variant)
        
        if not model_path or not model_path.exists():
            print(f"⚠️ Modelo TFLite '{self.model_variant}' no encontrado")
            print("📋 Modelos disponibles:")
            for variant, path in model_paths.items():
                if path.exists():
                    size_mb = path.stat().st_size / (1024 * 1024)
                    print(f"   ✅ {variant}: {size_mb:.1f} MB")
                else:
                    print(f"   ❌ {variant}: No disponible")
            return False
        
        try:
            # Cargar modelo TFLite
            self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
            self.interpreter.allocate_tensors()
            
            # Obtener detalles de input/output
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.backbone_available = True
            
            # Información del modelo
            input_shape = self.input_details[0]['shape']
            output_shape = self.output_details[0]['shape']
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            
            print(f"✅ RootNet TFLite '{self.model_variant}' cargado:")
            print(f"   📊 Input: {input_shape}")
            print(f"   📊 Output: {output_shape}")
            print(f"   📦 Tamaño: {model_size_mb:.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando TFLite: {e}")
            return False
    
    def extract_visual_features(self, img_patch):
        """Extraer features visuales usando backbone TFLite"""
        if not self.backbone_available:
            return None
        
        try:
            start_time = time.time()
            
            # Preprocesar imagen
            img_resized = cv2.resize(img_patch, (256, 256))
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_array = img_normalized.transpose(2, 0, 1)[np.newaxis, :]
            
            # Inferencia TFLite
            self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
            self.interpreter.invoke()
            features = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # ms
            self.inference_times.append(inference_time)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extrayendo features TFLite: {e}")
            return None
    
    def analyze_person_characteristics(self, img_patch, bbox):
        """Analizar características de la persona usando features TFLite"""
        # Extraer features si está disponible
        features = self.extract_visual_features(img_patch)
        
        analysis = {
            'bbox_area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
            'bbox_height': bbox[3] - bbox[1],
            'bbox_width': bbox[2] - bbox[0],
            'aspect_ratio': (bbox[3] - bbox[1]) / (bbox[2] - bbox[0]),
            'position_x': (bbox[0] + bbox[2]) / 2,
            'features_available': features is not None,
            'model_variant': self.model_variant
        }
        
        if features is not None:
            # Análisis de features profundas
            feature_stats = {
                'mean_activation': float(features.mean()),
                'max_activation': float(features.max()),
                'activation_std': float(features.std()),
                'feature_energy': float(np.sum(features ** 2)),
                'activation_sparsity': float(np.mean(features == 0))
            }
            analysis.update(feature_stats)
            
            # Estimación mejorada basada en features
            depth_hint = self._estimate_depth_from_features(features, bbox)
            analysis['feature_depth_hint'] = depth_hint
        
        return analysis
    
    def _estimate_depth_from_features(self, features, bbox):
        """Estimación de profundidad basada en features del backbone TFLite"""
        # Features shape: [1, 2048, 8, 8]
        
        # Calcular estadísticas de activación
        mean_activation = features.mean()
        max_activation = features.max()
        activation_energy = np.sum(features ** 2)
        sparsity = np.mean(features == 0)
        
        # Heurística mejorada: activaciones más altas sugieren personas más cercanas
        base_depth = 2500  # 2.5m baseline
        
        # Ajuste basado en energía de activación
        energy_factor = activation_energy / 1000000  # Normalizar
        depth_adjustment = (1.0 - np.tanh(energy_factor)) * 1500  # ±1.5m
        
        # Ajuste basado en sparsity (menos sparse = más cerca)
        sparsity_adjustment = sparsity * 800  # Más sparse = más lejos
        
        # Ajuste basado en bbox (confirmación)
        bbox_height = bbox[3] - bbox[1]
        if bbox_height > 400:  # Persona cerca
            depth_adjustment -= 600
        elif bbox_height < 150:  # Persona lejos
            depth_adjustment += 1000
        
        estimated_depth = base_depth + depth_adjustment + sparsity_adjustment
        
        # Limitar rango razonable
        return max(min(estimated_depth, 6000.0), 800.0)
    
    def predict_depth(self, img_patch, bbox, focal=None, use_analysis=True):
        """Predecir profundidad con análisis TFLite híbrido"""
        if focal is None:
            focal = self.default_focal
        
        # Análisis completo de la persona usando TFLite
        if use_analysis and self.backbone_available:
            analysis = self.analyze_person_characteristics(img_patch, bbox)
            
            # Usar estimación basada en features si está disponible
            if 'feature_depth_hint' in analysis:
                return analysis['feature_depth_hint']
        
        # Fallback a estimación heurística pura
        return self._estimate_depth_heuristic(bbox)
    
    def _estimate_depth_heuristic(self, bbox):
        """Estimación heurística de profundidad básica"""
        x1, y1, x2, y2 = bbox
        
        bbox_height = y2 - y1
        bbox_center_x = (x1 + x2) / 2
        
        # Estimación basada en altura del bbox
        if bbox_height > 400:  # Persona cerca
            depth = 1000 + np.random.normal(0, 150)
        elif bbox_height > 250:  # Persona media distancia
            depth = 2000 + np.random.normal(0, 250)
        elif bbox_height > 150:  # Persona lejos
            depth = 3500 + np.random.normal(0, 400)
        else:  # Persona muy lejos
            depth = 5000 + np.random.normal(0, 600)
        
        # Ajuste por posición horizontal
        if bbox_center_x < 400:  # Lado izquierdo
            depth *= 0.95
        elif bbox_center_x > 1200:  # Lado derecho
            depth *= 1.05
        
        return max(min(depth, 6000.0), 800.0)
    
    def get_performance_stats(self):
        """Obtener estadísticas de rendimiento"""
        if not self.inference_times:
            return None
        
        return {
            'avg_inference_ms': np.mean(self.inference_times),
            'std_inference_ms': np.std(self.inference_times),
            'min_inference_ms': np.min(self.inference_times),
            'max_inference_ms': np.max(self.inference_times),
            'total_inferences': len(self.inference_times),
            'model_variant': self.model_variant
        }
    
    def benchmark_performance(self, num_tests=100):
        """Benchmark del rendimiento del wrapper TFLite"""
        print(f"⚡ Benchmarking TFLite '{self.model_variant}' ({num_tests} tests)...")
        
        # Generar datos de prueba
        test_cases = []
        for i in range(num_tests):
            img = np.random.rand(256, 256, 3) * 255
            img = img.astype(np.uint8)
            bbox = [50, 50, 200, 400]  # Bbox típico
            test_cases.append((img, bbox))
        
        # Test con análisis completo TFLite
        start_time = time.time()
        depths = []
        
        for img, bbox in test_cases:
            depth = self.predict_depth(img, bbox, use_analysis=True)
            depths.append(depth)
        
        total_time = time.time() - start_time
        
        # Estadísticas
        stats = self.get_performance_stats()
        
        print("📊 RESULTADOS DEL BENCHMARK:")
        if stats:
            print(f"   Inferencia TFLite: {stats['avg_inference_ms']:.2f} ± {stats['std_inference_ms']:.2f} ms")
            print(f"   Tiempo total: {total_time*1000:.1f} ms para {num_tests} predicciones")
            print(f"   FPS teórico: {num_tests/total_time:.1f}")
        
        depth_variation = np.std(depths)
        print(f"   Variación profundidad: {depth_variation:.1f} mm")
        print(f"   Rango profundidad: [{np.min(depths):.0f}, {np.max(depths):.0f}] mm")

def test_tflite_wrapper():
    """Test completo del wrapper TFLite"""
    print("🧪 Testing RootNet TFLite Wrapper")
    print("="*50)
    
    # Probar diferentes variantes
    variants = ["default", "size", "latency"]
    results = {}
    
    for variant in variants:
        print(f"\n🔬 Testing variant: {variant}")
        
        # Crear wrapper
        wrapper = RootNetTFLiteWrapper(model_variant=variant)
        
        if not wrapper.backbone_available:
            print(f"   ❌ Variant '{variant}' no disponible")
            continue
        
        # Test básico
        test_img = np.random.rand(256, 256, 3) * 255
        test_img = test_img.astype(np.uint8)
        test_bbox = [100, 50, 250, 450]
        
        depth = wrapper.predict_depth(test_img, test_bbox)
        print(f"   Profundidad estimada: {depth:.1f} mm")
        
        # Benchmark rápido
        wrapper.benchmark_performance(20)
        
        results[variant] = wrapper.get_performance_stats()
    
    # Comparación final
    print("\n📊 COMPARACIÓN DE VARIANTES:")
    print("   Variant   │ Inferencia (ms) │ Disponible")
    print("   ──────────│─────────────────│───────────")
    
    for variant in variants:
        if variant in results and results[variant]:
            avg_time = results[variant]['avg_inference_ms']
            print(f"   {variant:9} │ {avg_time:15.2f} │ ✅")
        else:
            print(f"   {variant:9} │ {'N/A':15} │ ❌")
    
    print("\n✅ Test completado")

if __name__ == "__main__":
    test_tflite_wrapper()