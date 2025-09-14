#!/usr/bin/env python3
"""
Comparación PyTorch vs ONNX RootNet Backbone
===========================================
Evalúa el rendimiento y consistencia entre las versiones
PyTorch y ONNX del backbone de RootNet usando el wrapper mejorado.
"""

import sys
import os
import time
import numpy as np
import cv2
import torch
import onnxruntime as ort
from pathlib import Path
import matplotlib.pyplot as plt
from collections import OrderedDict

# Configurar paths
ROOTNET_ROOT = Path("/home/user/convnextpose_esteban/3DMPPE_ROOTNET_RELEASE")
CONVNEXT_ROOT = Path("/home/user/convnextpose_esteban/ConvNeXtPose/")

# Agregar paths al sistema
sys.path.insert(0, str(CONVNEXT_ROOT / "demo"))
sys.path.insert(0, str(ROOTNET_ROOT))
sys.path.insert(0, str(ROOTNET_ROOT / "main"))

from root_wrapper_improved import RootNetWrapperImproved

class RootNetBackboneComparator:
    """Comparador entre PyTorch y ONNX para RootNet Backbone"""
    
    def __init__(self):
        self.pytorch_model = None
        self.onnx_session = None
        self.test_images = []
        self.results = {
            'pytorch': {'times': [], 'outputs': []},
            'onnx': {'times': [], 'outputs': []},
            'differences': [],
            'errors': []
        }
        
    def load_pytorch_model(self):
        """Cargar modelo PyTorch del backbone"""
        print("📦 Cargando modelo PyTorch...")
        
        try:
            # Configurar paths
            main_path = str(ROOTNET_ROOT / "main")
            os.chdir(main_path)
            
            # Importar componentes
            from config import cfg
            from model import get_pose_net
            
            # Configurar
            cfg.set_args('0')
            
            # Crear modelo
            model = get_pose_net(cfg, is_train=False)
            model.eval()
            
            # Cargar checkpoint si existe
            checkpoint_path = CONVNEXT_ROOT / "demo/snapshot_18.pth.tar"
            
            if checkpoint_path.exists():
                checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
                state_dict = checkpoint.get('network', checkpoint)
                
                # Limpiar state_dict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_key = k.replace('module.', '')
                    new_state_dict[new_key] = v.cpu()
                
                model.load_state_dict(new_state_dict, strict=False)
                print("✅ Checkpoint PyTorch cargado")
            
            # Solo usar el backbone
            self.pytorch_model = model.backbone
            self.pytorch_model.eval()
            
            print("✅ Modelo PyTorch backbone cargado exitosamente")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando PyTorch: {e}")
            return False
    
    def load_onnx_model(self):
        """Cargar modelo ONNX"""
        print("📦 Cargando modelo ONNX...")
        
        try:
            onnx_path = CONVNEXT_ROOT / "exports/rootnet_backbone.onnx"
            
            if not onnx_path.exists():
                print(f"❌ Archivo ONNX no encontrado: {onnx_path}")
                return False
            
            # Crear sesión ONNX con CPU
            providers = ['CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
            
            # Verificar inputs/outputs
            input_info = self.onnx_session.get_inputs()[0]
            output_info = self.onnx_session.get_outputs()[0]
            
            print(f"✅ ONNX Input: {input_info.name} {input_info.shape}")
            print(f"✅ ONNX Output: {output_info.name} {output_info.shape}")
            print("✅ Modelo ONNX cargado exitosamente")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando ONNX: {e}")
            return False
    
    def generate_test_data(self, num_samples=10):
        """Generar datos de prueba simulando crops de personas"""
        print(f"🎲 Generando {num_samples} muestras de prueba...")
        
        self.test_images = []
        
        for i in range(num_samples):
            # Generar imagen sintética que simula un crop de persona
            img = np.random.rand(256, 256, 3).astype(np.float32)
            
            # Agregar algunas características más realistas
            # Simular gradientes verticales (persona típica)
            for y in range(256):
                intensity = 0.3 + 0.4 * (y / 256)  # Más oscuro arriba, más claro abajo
                img[y, :, :] *= intensity
            
            # Agregar algo de estructura
            center_x, center_y = 128, 128
            for y in range(256):
                for x in range(256):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist < 80:  # Área central más brillante (torso)
                        img[y, x, :] *= 1.2
            
            # Normalizar a [0, 1]
            img = np.clip(img, 0, 1)
            
            self.test_images.append(img)
        
        print(f"✅ {len(self.test_images)} muestras generadas")
    
    def preprocess_for_pytorch(self, img):
        """Preprocesar imagen para PyTorch"""
        # Convertir a tensor [1, 3, 256, 256]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img_tensor
    
    def preprocess_for_onnx(self, img):
        """Preprocesar imagen para ONNX"""
        # Convertir a numpy [1, 3, 256, 256]
        img_array = img.transpose(2, 0, 1)[np.newaxis, :]
        return img_array.astype(np.float32)
    
    def benchmark_pytorch(self):
        """Benchmark del modelo PyTorch"""
        print("⚡ Benchmarking PyTorch...")
        
        if self.pytorch_model is None:
            print("❌ Modelo PyTorch no cargado")
            return
        
        with torch.no_grad():
            for i, img in enumerate(self.test_images):
                # Preprocesar
                input_tensor = self.preprocess_for_pytorch(img)
                
                # Medir tiempo
                start_time = time.time()
                output = self.pytorch_model(input_tensor)
                end_time = time.time()
                
                # Guardar resultados
                inference_time = (end_time - start_time) * 1000  # ms
                self.results['pytorch']['times'].append(inference_time)
                self.results['pytorch']['outputs'].append(output.cpu().numpy())
                
                if i == 0:
                    print(f"✅ PyTorch output shape: {output.shape}")
        
        avg_time = np.mean(self.results['pytorch']['times'])
        print(f"📊 PyTorch promedio: {avg_time:.2f} ms")
    
    def benchmark_onnx(self):
        """Benchmark del modelo ONNX"""
        print("⚡ Benchmarking ONNX...")
        
        if self.onnx_session is None:
            print("❌ Modelo ONNX no cargado")
            return
        
        input_name = self.onnx_session.get_inputs()[0].name
        
        for i, img in enumerate(self.test_images):
            # Preprocesar
            input_array = self.preprocess_for_onnx(img)
            
            # Medir tiempo
            start_time = time.time()
            output = self.onnx_session.run(None, {input_name: input_array})
            end_time = time.time()
            
            # Guardar resultados
            inference_time = (end_time - start_time) * 1000  # ms
            self.results['onnx']['times'].append(inference_time)
            self.results['onnx']['outputs'].append(output[0])
            
            if i == 0:
                print(f"✅ ONNX output shape: {output[0].shape}")
        
        avg_time = np.mean(self.results['onnx']['times'])
        print(f"📊 ONNX promedio: {avg_time:.2f} ms")
    
    def compare_outputs(self):
        """Comparar outputs entre PyTorch y ONNX"""
        print("🔍 Comparando precisión...")
        
        if not self.results['pytorch']['outputs'] or not self.results['onnx']['outputs']:
            print("❌ No hay outputs para comparar")
            return
        
        for i in range(len(self.test_images)):
            pytorch_out = self.results['pytorch']['outputs'][i]
            onnx_out = self.results['onnx']['outputs'][i]
            
            # Calcular diferencias
            diff = np.abs(pytorch_out - onnx_out)
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            self.results['differences'].append({
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'relative_error': max_diff / (np.abs(pytorch_out).max() + 1e-8)
            })
        
        # Estadísticas generales
        max_diffs = [d['max_diff'] for d in self.results['differences']]
        mean_diffs = [d['mean_diff'] for d in self.results['differences']]
        rel_errors = [d['relative_error'] for d in self.results['differences']]
        
        print(f"📊 Diferencia máxima promedio: {np.mean(max_diffs):.2e}")
        print(f"📊 Diferencia media promedio: {np.mean(mean_diffs):.2e}")
        print(f"📊 Error relativo promedio: {np.mean(rel_errors):.2%}")
        
        # Evaluar calidad
        if np.mean(max_diffs) < 1e-5:
            print("✅ EXCELENTE: Diferencias < 1e-5")
        elif np.mean(max_diffs) < 1e-4:
            print("✅ BUENO: Diferencias < 1e-4")
        elif np.mean(max_diffs) < 1e-3:
            print("⚠️ ACEPTABLE: Diferencias < 1e-3")
        else:
            print("❌ PROBLEMÁTICO: Diferencias > 1e-3")
    
    def generate_report(self):
        """Generar reporte completo"""
        print("\n" + "="*80)
        print("📋 REPORTE FINAL DE COMPARACIÓN")
        print("="*80)
        
        # Rendimiento
        pytorch_times = self.results['pytorch']['times']
        onnx_times = self.results['onnx']['times']
        
        if pytorch_times and onnx_times:
            pytorch_avg = np.mean(pytorch_times)
            onnx_avg = np.mean(onnx_times)
            speedup = pytorch_avg / onnx_avg
            
            print(f"⚡ RENDIMIENTO:")
            print(f"   PyTorch: {pytorch_avg:.2f} ± {np.std(pytorch_times):.2f} ms")
            print(f"   ONNX:    {onnx_avg:.2f} ± {np.std(onnx_times):.2f} ms")
            print(f"   Speedup: {speedup:.2f}x {'🚀' if speedup > 1 else '📉'}")
        
        # Precisión
        if self.results['differences']:
            max_diffs = [d['max_diff'] for d in self.results['differences']]
            rel_errors = [d['relative_error'] for d in self.results['differences']]
            
            print(f"🎯 PRECISIÓN:")
            print(f"   Max diff: {np.mean(max_diffs):.2e} ± {np.std(max_diffs):.2e}")
            print(f"   Rel error: {np.mean(rel_errors):.2%} ± {np.std(rel_errors):.2%}")
        
        # Recomendación
        print(f"💡 RECOMENDACIÓN:")
        if onnx_times and pytorch_times:
            if onnx_avg < pytorch_avg and np.mean(max_diffs) < 1e-4:
                print("   ✅ USAR ONNX - Más rápido y suficientemente preciso")
            elif np.mean(max_diffs) > 1e-3:
                print("   ⚠️ USAR PYTORCH - ONNX tiene diferencias significativas")
            else:
                print("   🤔 AMBOS VIABLES - Elegir según requirements específicos")
        
        print("="*80)
    
    def run_comparison(self):
        """Ejecutar comparación completa"""
        print("🔄 Iniciando comparación PyTorch vs ONNX...")
        
        # 1. Cargar modelos
        pytorch_ok = self.load_pytorch_model()
        onnx_ok = self.load_onnx_model()
        
        if not pytorch_ok or not onnx_ok:
            print("❌ No se pudieron cargar ambos modelos")
            return False
        
        # 2. Generar datos de prueba
        self.generate_test_data(20)  # 20 muestras para mejor estadística
        
        # 3. Benchmark
        self.benchmark_pytorch()
        self.benchmark_onnx()
        
        # 4. Comparar
        self.compare_outputs()
        
        # 5. Reporte
        self.generate_report()
        
        return True

def main():
    """Función principal"""
    print("🎯 Comparador RootNet Backbone: PyTorch vs ONNX")
    print("="*60)
    
    # Cambiar al directorio correcto
    os.chdir(CONVNEXT_ROOT / "demo")
    
    # Crear y ejecutar comparador
    comparator = RootNetBackboneComparator()
    success = comparator.run_comparison()
    
    if success:
        print("🎉 Comparación completada exitosamente")
    else:
        print("❌ Error durante la comparación")

if __name__ == "__main__":
    main()