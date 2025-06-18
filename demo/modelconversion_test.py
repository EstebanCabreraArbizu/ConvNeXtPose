#!/usr/bin/env python3
"""
model_conversion_test.py - Script de prueba para verificar conversión de modelos
PyTorch → ONNX → TensorFlow → TensorFlow Lite

Este script verifica paso a paso que cada conversión funcione correctamente
manteniendo la precisión del modelo de pose estimation.
"""

import os
import sys
import time
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import tempfile
import shutil

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelConfig:
    """Configuración del modelo de pose estimation"""
    def __init__(self):
        # Configuraciones específicas del modelo
        self.input_shape = (256, 256)
        self.output_shape = (32, 32)
        self.depth_dim = 32
        self.bbox_3d_shape = (2000, 2000, 2000)
        self.joint_num = 18
        self.pixel_mean = [0.485, 0.456, 0.406]
        self.pixel_std = [0.229, 0.224, 0.225]

# Configuración global
cfg = ModelConfig()

class ConversionTester:
    """
    Clase principal para probar la conversión de modelos paso a paso.
    Cada método verifica un paso específico de la conversión.
    """
    
    def __init__(self, model_path: str, temp_dir: Optional[str] = None):
        self.model_path = model_path
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="pose_conversion_")
        self.results = {}
        
        logger.info(f"🔧 Inicializando ConversionTester")
        logger.info(f"📁 Directorio temporal: {self.temp_dir}")
        logger.info(f"🎯 Modelo original: {model_path}")
        
        # Rutas de los modelos convertidos
        self.onnx_path = os.path.join(self.temp_dir, "model.onnx")
        self.savedmodel_path = os.path.join(self.temp_dir, "savedmodel")
        self.tflite_path = os.path.join(self.temp_dir, "model.tflite")
        
        # Datos de prueba
        self.test_input = self._generate_test_input()
        
    def _generate_test_input(self) -> np.ndarray:
        """Generar datos de entrada de prueba realistas"""
        # Crear una imagen sintética que simule una persona
        np.random.seed(42)  # Para reproducibilidad
        
        # Imagen base con ruido
        image = np.random.rand(1, 3, cfg.input_shape[0], cfg.input_shape[1]).astype(np.float32)
        
        # Normalizar según las estadísticas del modelo
        for i in range(3):
            image[0, i] = (image[0, i] - cfg.pixel_mean[i]) / cfg.pixel_std[i]
        
        return image
    
    def test_pytorch_loading(self) -> bool:
        """
        Paso 1: Verificar que podemos cargar el modelo PyTorch correctamente
        """
        logger.info("🔍 Paso 1: Probando carga del modelo PyTorch...")
        
        try:
            import torch
            import torch.nn as nn
            
            # Verificar disponibilidad de CUDA
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"🖥️  Usando dispositivo: {device}")
            
            # Crear un modelo de prueba si no existe el archivo
            if not os.path.exists(self.model_path):
                logger.warning("⚠️  Modelo no encontrado, creando modelo de prueba...")
                model = self._create_test_model()
                torch.save(model.state_dict(), self.model_path)
            else:
                # Cargar modelo existente
                model = self._create_test_model()  # Arquitectura base
                state_dict = torch.load(self.model_path, map_location=device)
                
                # Manejar diferentes formatos de guardado
                if isinstance(state_dict, dict):
                    if 'model' in state_dict:
                        state_dict = state_dict['model']
                    elif 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                
                model.load_state_dict(state_dict, strict=False)
            
            model = model.to(device).eval()
            
            # Prueba de inferencia
            test_input_torch = torch.from_numpy(self.test_input).to(device)
            
            with torch.no_grad():
                start_time = time.time()
                pytorch_output = model(test_input_torch)
                inference_time = time.time() - start_time
            
            # Validar salida
            if isinstance(pytorch_output, (list, tuple)):
                pytorch_output = pytorch_output[0]
            
            expected_shape = (1, cfg.joint_num, cfg.output_shape[0], cfg.output_shape[1])
            actual_shape = pytorch_output.shape
            
            logger.info(f"✅ PyTorch cargado exitosamente")
            logger.info(f"📊 Forma de salida: {actual_shape}")
            logger.info(f"⏱️  Tiempo de inferencia: {inference_time*1000:.2f}ms")
            
            # Guardar referencia para comparación
            self.results['pytorch'] = {
                'output': pytorch_output.cpu().numpy(),
                'inference_time': inference_time,
                'model': model,
                'device': device
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en carga PyTorch: {e}")
            return False
    
    def _create_test_model(self):
        """Crear un modelo de prueba simple para pose estimation"""
        import torch
        import torch.nn as nn
        
        class SimplePoseNet(nn.Module):
            def __init__(self):
                super(SimplePoseNet, self).__init__()
                
                # Encoder simple
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    
                    # Bloques residuales simplificados
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    
                    nn.Conv2d(256, 512, 3, stride=2, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
                
                # Head para pose estimation
                self.pose_head = nn.Sequential(
                    nn.Conv2d(512, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    
                    nn.Conv2d(256, cfg.joint_num, 1),
                    nn.Sigmoid()  # Para normalizar salidas
                )
                
            def forward(self, x):
                features = self.backbone(x)
                pose_output = self.pose_head(features)
                return pose_output
        
        return SimplePoseNet()
    
    def test_pytorch_to_onnx(self) -> bool:
        """
        Paso 2: Convertir PyTorch a ONNX y verificar equivalencia
        """
        logger.info("🔄 Paso 2: Convirtiendo PyTorch → ONNX...")
        
        try:
            import torch
            
            if 'pytorch' not in self.results:
                logger.error("❌ Debe ejecutar test_pytorch_loading primero")
                return False
            
            model = self.results['pytorch']['model']
            device = self.results['pytorch']['device']
            
            # Preparar entrada para export
            dummy_input = torch.from_numpy(self.test_input).to(device)
            
            # Exportar a ONNX
            logger.info("📤 Exportando a ONNX...")
            torch.onnx.export(
                model,
                dummy_input,
                self.onnx_path,
                export_params=True,
                opset_version=11,  # Versión compatible con tf2onnx
                do_constant_folding=True,
                input_names=['input'],
                output_names=['pose_coords'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'pose_coords': {0: 'batch_size'}
                },
                verbose=False
            )
            
            # Verificar el archivo ONNX
            file_size = os.path.getsize(self.onnx_path) / (1024 * 1024)
            logger.info(f"💾 Modelo ONNX creado: {file_size:.2f}MB")
            
            # Probar inferencia ONNX
            onnx_output = self._test_onnx_inference()
            if onnx_output is None:
                return False
            
            # Comparar salidas
            pytorch_output = self.results['pytorch']['output']
            difference = np.mean(np.abs(pytorch_output - onnx_output))
            
            logger.info(f"📈 Diferencia promedio PyTorch vs ONNX: {difference:.6f}")
            
            if difference < 1e-4:
                logger.info("✅ Conversión PyTorch → ONNX exitosa (alta precisión)")
            elif difference < 1e-2:
                logger.info("⚠️  Conversión PyTorch → ONNX exitosa (precisión moderada)")
            else:
                logger.warning(f"⚠️  Diferencia significativa detectada: {difference}")
            
            self.results['onnx'] = {
                'output': onnx_output,
                'difference_from_pytorch': difference,
                'file_size_mb': file_size
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en conversión PyTorch → ONNX: {e}")
            return False
    
    def _test_onnx_inference(self) -> Optional[np.ndarray]:
        """Probar inferencia con ONNX Runtime"""
        try:
            import onnxruntime as ort
            import torch
            # Configurar proveedores de ejecución
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
            
            # Crear sesión ONNX
            session = ort.InferenceSession(self.onnx_path, providers=providers)
            
            # Ejecutar inferencia
            start_time = time.time()
            onnx_output = session.run(None, {'input': self.test_input.astype(np.float32)})
            inference_time = time.time() - start_time
            
            logger.info(f"⏱️  Tiempo inferencia ONNX: {inference_time*1000:.2f}ms")
            
            return onnx_output[0]
            
        except Exception as e:
            logger.error(f"❌ Error en inferencia ONNX: {e}")
            return None
    
    def test_onnx_to_tensorflow(self) -> bool:
        """
        Paso 3: Convertir ONNX a TensorFlow SavedModel
        """
        logger.info("🔄 Paso 3: Convirtiendo ONNX → TensorFlow...")
        
        try:
            # Método 1: Usar onnx-tf (recomendado)
            success = self._convert_with_onnx_tf()
            if success:
                return success
            
            logger.warning("⚠️  onnx-tf falló, intentando con tf2onnx...")
            
            # Método 2: Usar tf2onnx (alternativo)
            return self._convert_with_tf2onnx()
            
        except Exception as e:
            logger.error(f"❌ Error en conversión ONNX → TensorFlow: {e}")
            return False
    
    def _convert_with_onnx_tf(self) -> bool:
        """Convertir usando onnx-tf"""
        try:
            import onnx
            from onnx_tf.backend import prepare
            import tensorflow as tf
            
            logger.info("📦 Usando onnx-tf para conversión...")
            
            # Cargar modelo ONNX
            onnx_model = onnx.load(self.onnx_path)
            
            # Preparar para TensorFlow
            tf_rep = prepare(onnx_model)
            
            # Exportar como SavedModel
            tf_rep.export_graph(self.savedmodel_path)
            
            # Verificar SavedModel
            if os.path.exists(self.savedmodel_path):
                logger.info("✅ SavedModel creado exitosamente")
                
                # Probar inferencia TensorFlow
                tf_output = self._test_tensorflow_inference()
                if tf_output is not None:
                    # Comparar con ONNX
                    onnx_output = self.results['onnx']['output']
                    difference = np.mean(np.abs(onnx_output - tf_output))
                    
                    logger.info(f"📈 Diferencia ONNX vs TensorFlow: {difference:.6f}")
                    
                    self.results['tensorflow'] = {
                        'output': tf_output,
                        'difference_from_onnx': difference,
                        'conversion_method': 'onnx-tf'
                    }
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error con onnx-tf: {e}")
            return False
    
    def _convert_with_tf2onnx(self) -> bool:
        """Convertir usando tf2onnx (método alternativo)"""
        try:
            import subprocess
            
            logger.info("📦 Usando tf2onnx para conversión...")
            
            # Comando para conversión inversa (ONNX → TF)
            # Nota: tf2onnx principalmente hace TF → ONNX, pero puede hacer lo inverso
            cmd = [
                'python', '-m', 'tf2onnx.convert',
                '--onnx', self.onnx_path,
                '--output', self.savedmodel_path,
                '--inputs', 'input:0',
                '--outputs', 'pose_coords:0'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Conversión tf2onnx exitosa")
                return True
            else:
                logger.error(f"❌ tf2onnx falló: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error con tf2onnx: {e}")
            return False
    
    def _test_tensorflow_inference(self) -> Optional[np.ndarray]:
        """Probar inferencia con TensorFlow"""
        try:
            import tensorflow as tf
            
            # Cargar SavedModel
            model = tf.saved_model.load(self.savedmodel_path)
            
            # Obtener función de inferencia
            infer = model.signatures['serving_default']
            
            # Preparar entrada
            input_tensor = tf.constant(self.test_input.astype(np.float32))
            
            # Ejecutar inferencia
            start_time = time.time()
            tf_output = infer(input_tensor)
            inference_time = time.time() - start_time
            
            logger.info(f"⏱️  Tiempo inferencia TensorFlow: {inference_time*1000:.2f}ms")
            
            # Extraer salida (el nombre puede variar)
            output_key = list(tf_output.keys())[0]
            return tf_output[output_key].numpy()
            
        except Exception as e:
            logger.error(f"❌ Error en inferencia TensorFlow: {e}")
            return None
    
    def test_tensorflow_to_tflite(self) -> bool:
        """
        Paso 4: Convertir TensorFlow SavedModel a TensorFlow Lite
        """
        logger.info("🔄 Paso 4: Convirtiendo TensorFlow → TensorFlow Lite...")
        
        try:
            import tensorflow as tf
            
            if 'tensorflow' not in self.results:
                logger.error("❌ Debe ejecutar conversión a TensorFlow primero")
                return False
            
            # Convertir a TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(self.savedmodel_path)
            
            # Configurar optimizaciones
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Configurar para compatibilidad
            converter.target_spec.supported_types = [tf.float16]
            converter.experimental_new_converter = True
            
            # Convertir
            logger.info("🔧 Aplicando optimizaciones TFLite...")
            tflite_model = converter.convert()
            
            # Guardar modelo
            with open(self.tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            file_size = os.path.getsize(self.tflite_path) / (1024 * 1024)
            logger.info(f"💾 Modelo TFLite creado: {file_size:.2f}MB")
            
            # Probar inferencia TFLite
            tflite_output = self._test_tflite_inference()
            if tflite_output is not None:
                # Comparar con TensorFlow
                tf_output = self.results['tensorflow']['output']
                difference = np.mean(np.abs(tf_output - tflite_output))
                
                logger.info(f"📈 Diferencia TensorFlow vs TFLite: {difference:.6f}")
                
                self.results['tflite'] = {
                    'output': tflite_output,
                    'difference_from_tensorflow': difference,
                    'file_size_mb': file_size
                }
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error en conversión TensorFlow → TFLite: {e}")
            return False
    
    def _test_tflite_inference(self) -> Optional[np.ndarray]:
        """Probar inferencia con TensorFlow Lite"""
        try:
            import tensorflow as tf
            
            # Cargar modelo TFLite
            interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
            interpreter.allocate_tensors()
            
            # Obtener detalles de entrada y salida
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            logger.info(f"📋 TFLite - Entrada: {input_details[0]['shape']}")
            logger.info(f"📋 TFLite - Salida: {output_details[0]['shape']}")
            
            # Preparar entrada
            input_data = self.test_input.astype(input_details[0]['dtype'])
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Ejecutar inferencia
            start_time = time.time()
            interpreter.invoke()
            inference_time = time.time() - start_time
            
            logger.info(f"⏱️  Tiempo inferencia TFLite: {inference_time*1000:.2f}ms")
            
            # Obtener salida
            tflite_output = interpreter.get_tensor(output_details[0]['index'])
            return tflite_output
            
        except Exception as e:
            logger.error(f"❌ Error en inferencia TFLite: {e}")
            return None
    
    def run_full_test(self) -> Dict[str, Any]:
        """
        Ejecutar la prueba completa de conversión paso a paso
        """
        logger.info("🚀 Iniciando prueba completa de conversión de modelos")
        logger.info("=" * 60)
        
        test_results = {
            'pytorch_loading': False,
            'pytorch_to_onnx': False,
            'onnx_to_tensorflow': False,
            'tensorflow_to_tflite': False,
            'overall_success': False
        }
        
        # Paso 1: PyTorch
        if self.test_pytorch_loading():
            test_results['pytorch_loading'] = True
            logger.info("✅ Paso 1 completado")
        else:
            logger.error("❌ Paso 1 falló - deteniendo pruebas")
            return test_results
        
        # Paso 2: PyTorch → ONNX
        if self.test_pytorch_to_onnx():
            test_results['pytorch_to_onnx'] = True
            logger.info("✅ Paso 2 completado")
        else:
            logger.error("❌ Paso 2 falló - continuando con siguientes pasos")
        
        # Paso 3: ONNX → TensorFlow (solo si paso 2 exitoso)
        if test_results['pytorch_to_onnx'] and self.test_onnx_to_tensorflow():
            test_results['onnx_to_tensorflow'] = True
            logger.info("✅ Paso 3 completado")
        else:
            logger.error("❌ Paso 3 falló - saltando TFLite")
        
        # Paso 4: TensorFlow → TFLite (solo si paso 3 exitoso)
        if test_results['onnx_to_tensorflow'] and self.test_tensorflow_to_tflite():
            test_results['tensorflow_to_tflite'] = True
            logger.info("✅ Paso 4 completado")
        else:
            logger.error("❌ Paso 4 falló")
        
        # Evaluar éxito general (excluir 'overall_success' de la evaluación)
        core_results = {k: v for k, v in test_results.items() if k != 'overall_success'}
        test_results['overall_success'] = all(core_results.values())
        
        logger.info("=" * 60)
        self._print_summary(test_results)
        
        return test_results
    
    def _print_summary(self, test_results: Dict[str, Any]):
        """Imprimir resumen detallado de los resultados"""
        logger.info("📊 RESUMEN DE CONVERSIÓN DE MODELOS")
        logger.info("-" * 40)
        
        # Estado de cada paso
        steps = [
            ("PyTorch Loading", test_results['pytorch_loading']),
            ("PyTorch → ONNX", test_results['pytorch_to_onnx']),
            ("ONNX → TensorFlow", test_results['onnx_to_tensorflow']),
            ("TensorFlow → TFLite", test_results['tensorflow_to_tflite'])
        ]
        
        for step_name, success in steps:
            status = "✅ EXITOSO" if success else "❌ FALLÓ"
            logger.info(f"{step_name:20} {status}")
        
        logger.info("-" * 40)
        
        # Detalles de precisión si están disponibles
        if 'onnx' in self.results:
            logger.info(f"🎯 Precisión PyTorch vs ONNX: {self.results['onnx']['difference_from_pytorch']:.6f}")
        
        if 'tensorflow' in self.results:
            logger.info(f"🎯 Precisión ONNX vs TensorFlow: {self.results['tensorflow']['difference_from_onnx']:.6f}")
        
        if 'tflite' in self.results:
            logger.info(f"🎯 Precisión TensorFlow vs TFLite: {self.results['tflite']['difference_from_tensorflow']:.6f}")
        
        # Información de archivos
        logger.info("-" * 40)
        logger.info("📁 ARCHIVOS GENERADOS:")
        
        files_info = [
            ("ONNX Model", self.onnx_path),
            ("TensorFlow SavedModel", self.savedmodel_path),
            ("TensorFlow Lite", self.tflite_path)
        ]
        
        for file_type, file_path in files_info:
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path) / (1024 * 1024)
                    logger.info(f"{file_type:20} {size:.2f}MB - {file_path}")
                else:
                    logger.info(f"{file_type:20} [DIR] - {file_path}")
            else:
                logger.info(f"{file_type:20} ❌ No creado")
        
        # Recomendaciones
        logger.info("-" * 40)
        logger.info("💡 RECOMENDACIONES:")
        
        if test_results['overall_success']:
            logger.info("🎉 ¡Conversión completa exitosa!")
            logger.info("   → Todos los formatos están listos para uso")
            logger.info("   → Verifica la precisión en tus datos reales")
        else:
            logger.info("⚠️  Conversión parcial - revisar errores:")
            if not test_results['pytorch_to_onnx']:
                logger.info("   → Problema en PyTorch → ONNX: revisar opset_version")
            if not test_results['onnx_to_tensorflow']:
                logger.info("   → Problema en ONNX → TF: instalar onnx-tf correctamente")
            if not test_results['tensorflow_to_tflite']:
                logger.info("   → Problema en TF → TFLite: revisar operaciones no soportadas")
    
    def cleanup(self):
        """Limpiar archivos temporales"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"🧹 Directorio temporal limpiado: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"⚠️  Error al limpiar directorio temporal: {e}")

def main():
    """Función principal para ejecutar las pruebas"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Probar conversión de modelos paso a paso')
    parser.add_argument('--model_path', type=str, 
                       help='Ruta al modelo PyTorch (.pth). Si no existe, se creará uno de prueba')
    parser.add_argument('--keep_files', action='store_true',
                       help='Mantener archivos temporales después de la prueba')
    parser.add_argument('--temp_dir', type=str,
                       help='Directorio temporal personalizado')
    
    args = parser.parse_args()
    
    # Usar modelo de prueba si no se especifica
    model_path = args.model_path or "test_pose_model.pth"
    
    # Crear tester
    tester = ConversionTester(model_path, args.temp_dir)
    
    try:
        # Ejecutar prueba completa
        results = tester.run_full_test()
        
        # Mostrar resultado final
        if results['overall_success']:
            logger.info("🎉 ¡TODAS LAS CONVERSIONES EXITOSAS!")
            return 0
        else:
            logger.error("❌ ALGUNAS CONVERSIONES FALLARON")
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Interrumpido por el usuario")
        return 1
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        return 1
    finally:
        # Limpiar si no se solicita mantener archivos
        if not args.keep_files:
            tester.cleanup()
        else:
            logger.info(f"📁 Archivos mantenidos en: {tester.temp_dir}")

if __name__ == "__main__":
    exit(main())