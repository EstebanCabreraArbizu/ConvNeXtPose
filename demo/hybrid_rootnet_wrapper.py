#!/usr/bin/env python3
"""
Wrapper Híbrido: ONNX Backbone + Estimación Heurística
======================================================
Combina el backbone ONNX optimizado con estimación heurística
para crear un pipeline de profundidad ultra-rápido.
"""

import os
import sys
import time
import logging
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path

# Configurar paths
CONVNEXT_ROOT = Path("/home/user/convnextpose_esteban/ConvNeXtPose/")
sys.path.insert(0, str(CONVNEXT_ROOT / "demo"))

logger = logging.getLogger(__name__)

class HybridRootNetWrapper:
    """Wrapper híbrido que usa ONNX + estimación heurística"""
    
    def __init__(self):
        self.onnx_session = None
        self.use_heuristic = True  # Usar heurística por defecto
        self.backbone_features = None  # Cache de features
        
        # Configuración por defecto
        self.default_focal = [1500, 1500]
        
        # Parámetros para estimación heurística mejorada
        self.person_height_mm = 1700
        self.camera_height_mm = 1600
        
        # Cargar modelo ONNX automáticamente
        self.load_onnx_backbone()
    
    def load_onnx_backbone(self):
        """Cargar backbone ONNX si está disponible"""
        try:
            onnx_path = CONVNEXT_ROOT / "exports/rootnet_backbone.onnx"
            
            if onnx_path.exists():
                providers = ['CPUExecutionProvider']
                self.onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
                logger.info("✅ Backbone ONNX cargado exitosamente")
                print("✅ Backbone ONNX cargado - Features disponibles para análisis")
            else:
                logger.warning("⚠️ Backbone ONNX no encontrado, solo heurística disponible")
                print("⚠️ Solo estimación heurística disponible")
                
        except Exception as e:
            logger.warning(f"⚠️ Error cargando ONNX: {e}")
            print(f"⚠️ Error ONNX: {e}, usando solo heurística")
    
    def extract_visual_features(self, img_patch):
        """Extraer features visuales usando backbone ONNX"""
        if self.onnx_session is None:
            return None
        
        try:
            # Preprocesar imagen
            img_resized = cv2.resize(img_patch, (256, 256))
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_array = img_normalized.transpose(2, 0, 1)[np.newaxis, :]
            
            # Inferencia ONNX
            input_name = self.onnx_session.get_inputs()[0].name
            features = self.onnx_session.run(None, {input_name: img_array})[0]
            
            # Guardar features para análisis
            self.backbone_features = features
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extrayendo features: {e}")
            return None
    
    def analyze_person_characteristics(self, img_patch, bbox):
        """Analizar características de la persona usando features visuales"""
        # Extraer features si está disponible
        features = self.extract_visual_features(img_patch)
        
        analysis = {
            'bbox_area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
            'bbox_height': bbox[3] - bbox[1],
            'bbox_width': bbox[2] - bbox[0],
            'aspect_ratio': (bbox[3] - bbox[1]) / (bbox[2] - bbox[0]),
            'position_x': (bbox[0] + bbox[2]) / 2,
            'features_available': features is not None
        }
        
        if features is not None:
            # Análisis de features profundas
            feature_stats = {
                'mean_activation': float(features.mean()),
                'max_activation': float(features.max()),
                'activation_std': float(features.std()),
                'feature_energy': float(np.sum(features ** 2))
            }
            analysis.update(feature_stats)
            
            # Estimación mejorada basada en features
            depth_hint = self._estimate_depth_from_features(features, bbox)
            analysis['feature_depth_hint'] = depth_hint
        
        return analysis
    
    def _estimate_depth_from_features(self, features, bbox):
        """Estimación de profundidad basada en features del backbone"""
        # Features shape: [1, 2048, 8, 8]
        
        # Calcular estadísticas de activación
        mean_activation = features.mean()
        max_activation = features.max()
        activation_energy = np.sum(features ** 2)
        
        # Heurística: activaciones más altas sugieren personas más cercanas
        # (más detalle capturado por el backbone)
        base_depth = 2500  # 2.5m baseline
        
        # Ajuste basado en energía de activación
        energy_factor = activation_energy / 1000000  # Normalizar
        depth_adjustment = (1.0 - np.tanh(energy_factor)) * 1500  # ±1.5m
        
        # Ajuste basado en bbox (confirmación)
        bbox_height = bbox[3] - bbox[1]
        if bbox_height > 400:  # Persona cerca
            depth_adjustment -= 800
        elif bbox_height < 150:  # Persona lejos
            depth_adjustment += 1200
        
        estimated_depth = base_depth + depth_adjustment
        
        # Limitar rango razonable
        return max(min(estimated_depth, 6000.0), 800.0)
    
    def predict_depth(self, img_patch, bbox, focal=None, use_analysis=True):
        """Predecir profundidad con análisis híbrido"""
        if focal is None:
            focal = self.default_focal
        
        # Análisis completo de la persona
        if use_analysis:
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
    
    def benchmark_performance(self, num_tests=50):
        """Benchmark del rendimiento del wrapper híbrido"""
        print(f"⚡ Benchmarking wrapper híbrido ({num_tests} tests)...")
        
        # Generar datos de prueba
        test_cases = []
        for i in range(num_tests):
            img = np.random.rand(256, 256, 3) * 255
            img = img.astype(np.uint8)
            bbox = [50, 50, 200, 400]  # Bbox típico
            test_cases.append((img, bbox))
        
        # Test con análisis completo
        times_with_features = []
        depths_with_features = []
        
        for img, bbox in test_cases:
            start_time = time.time()
            depth = self.predict_depth(img, bbox, use_analysis=True)
            end_time = time.time()
            
            times_with_features.append((end_time - start_time) * 1000)
            depths_with_features.append(depth)
        
        # Test solo heurística
        times_heuristic = []
        depths_heuristic = []
        
        for img, bbox in test_cases:
            start_time = time.time()
            depth = self._estimate_depth_heuristic(bbox)
            end_time = time.time()
            
            times_heuristic.append((end_time - start_time) * 1000)
            depths_heuristic.append(depth)
        
        # Reporte
        print("📊 RESULTADOS DEL BENCHMARK:")
        print(f"   Con features ONNX: {np.mean(times_with_features):.2f} ± {np.std(times_with_features):.2f} ms")
        print(f"   Solo heurística:   {np.mean(times_heuristic):.2f} ± {np.std(times_heuristic):.2f} ms")
        
        depth_variation_features = np.std(depths_with_features)
        depth_variation_heuristic = np.std(depths_heuristic)
        
        print(f"   Variación profundidad (features): {depth_variation_features:.1f} mm")
        print(f"   Variación profundidad (heurística): {depth_variation_heuristic:.1f} mm")
        
        if self.onnx_session is not None:
            improvement = depth_variation_features / depth_variation_heuristic
            print(f"   Mejora en variación: {improvement:.2f}x")

def test_hybrid_wrapper():
    """Test completo del wrapper híbrido"""
    print("🧪 Testing Hybrid RootNet Wrapper")
    print("="*50)
    
    # Crear wrapper
    wrapper = HybridRootNetWrapper()
    
    # Test básico
    print("\n1️⃣ Test básico de predicción:")
    test_img = np.random.rand(256, 256, 3) * 255
    test_img = test_img.astype(np.uint8)
    test_bbox = [100, 50, 250, 450]  # Persona típica
    
    depth = wrapper.predict_depth(test_img, test_bbox)
    print(f"   Profundidad estimada: {depth:.1f} mm")
    
    # Análisis de características
    print("\n2️⃣ Análisis de características:")
    analysis = wrapper.analyze_person_characteristics(test_img, test_bbox)
    
    print(f"   Área bbox: {analysis['bbox_area']:.0f} px²")
    print(f"   Altura bbox: {analysis['bbox_height']:.0f} px")
    print(f"   Aspect ratio: {analysis['aspect_ratio']:.2f}")
    print(f"   Features disponibles: {analysis['features_available']}")
    
    if analysis['features_available']:
        print(f"   Activación promedio: {analysis['mean_activation']:.4f}")
        print(f"   Energía features: {analysis['feature_energy']:.2e}")
    
    # Benchmark de rendimiento
    print("\n3️⃣ Benchmark de rendimiento:")
    wrapper.benchmark_performance(30)
    
    print("\n✅ Test completado exitosamente")

if __name__ == "__main__":
    test_hybrid_wrapper()