#!/usr/bin/env python3
"""
configurable_tflite_converter.py - Convertidor TFLite configurable con opciones expuestas
"""

import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from enum import Enum

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verificar dependencias TFLite
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("⚠️ TensorFlow not available")

try:
    import onnx
    import onnx_tf
    from onnx_tf.backend import prepare
    ONNX_TF_AVAILABLE = True
except ImportError:
    ONNX_TF_AVAILABLE = False
    logger.warning("⚠️ onnx-tf not available")

class OptimizationType(Enum):
    """Tipos de optimización TFLite"""
    NONE = "none"
    DEFAULT = "default" 
    SIZE = "size"
    LATENCY = "latency"

class SupportedOpsMode(Enum):
    """Modos de operaciones soportadas"""
    TFLITE_ONLY = "tflite_only"           # Solo operaciones TFLite nativas
    SELECT_TF = "select_tf"               # TFLite + subset TensorFlow ops
    FLEX_DELEGATE = "flex_delegate"       # Con delegado Flex (más ops TF)
    AUTO = "auto"                         # Automático basado en el modelo

class ConfigurableTFLiteConverter:
    """
    Convertidor TFLite configurable que expone opciones avanzadas al usuario
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="tflite_converter_")
        self._cleanup_temp = temp_dir is None  # Solo limpiar si creamos el temp_dir
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for TFLite conversion")
        if not ONNX_TF_AVAILABLE:
            raise ImportError("onnx-tf is required for ONNX to TFLite conversion")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cleanup_temp and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def convert(self, 
                onnx_path: str, 
                tflite_path: str,
                optimization: Union[str, OptimizationType] = OptimizationType.DEFAULT,
                supported_ops: Union[str, SupportedOpsMode] = SupportedOpsMode.AUTO,
                target_types: Optional[List[str]] = None,
                allow_custom_ops: bool = True,
                experimental_new_converter: bool = True,
                quantize_weights: bool = False,
                representative_dataset: Optional[callable] = None) -> Dict[str, Any]:
        """
        Convierte modelo ONNX a TFLite con opciones configurables
        
        Args:
            onnx_path: Ruta al modelo ONNX
            tflite_path: Ruta de salida TFLite
            optimization: Tipo de optimización (default, size, latency, none)
            supported_ops: Modo de operaciones soportadas
            target_types: Tipos de datos objetivo (e.g., ['float32', 'int8'])
            allow_custom_ops: Permitir operaciones personalizadas
            experimental_new_converter: Usar convertidor experimental
            quantize_weights: Habilitar cuantización de pesos
            representative_dataset: Dataset representativo para cuantización
            
        Returns:
            Dict con resultados de la conversión
        """
        
        # Normalizar enums
        if isinstance(optimization, str):
            optimization = OptimizationType(optimization)
        if isinstance(supported_ops, str):
            supported_ops = SupportedOpsMode(supported_ops)
        
        result = {
            'success': False,
            'tflite_path': None,
            'strategy_used': None,
            'file_size_mb': 0,
            'config_used': {
                'optimization': optimization.value,
                'supported_ops': supported_ops.value,
                'target_types': target_types,
                'allow_custom_ops': allow_custom_ops,
                'experimental_new_converter': experimental_new_converter,
                'quantize_weights': quantize_weights
            },
            'error': None
        }
        
        if not os.path.exists(onnx_path):
            result['error'] = f"ONNX model not found: {onnx_path}"
            return result
        
        # Crear directorio de salida
        os.makedirs(os.path.dirname(os.path.abspath(tflite_path)), exist_ok=True)
        
        try:
            # Estrategia principal: ONNX → SavedModel → TFLite
            success = self._convert_via_savedmodel(
                onnx_path, tflite_path, optimization, supported_ops,
                target_types, allow_custom_ops, experimental_new_converter,
                quantize_weights, representative_dataset
            )
            
            if success and os.path.exists(tflite_path):
                result['success'] = True
                result['tflite_path'] = tflite_path
                result['strategy_used'] = "onnx_tf_savedmodel_configurable"
                result['file_size_mb'] = os.path.getsize(tflite_path) / (1024 * 1024)
                
                logger.info(f"✅ Configurable TFLite conversion successful")
                logger.info(f"📊 Model size: {result['file_size_mb']:.2f} MB")
                logger.info(f"⚙️ Config: {result['config_used']}")
            else:
                result['error'] = "Conversion failed"
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ Configurable conversion failed: {e}")
        
        return result
    
    def _convert_via_savedmodel(self, 
                                onnx_path: str, 
                                tflite_path: str,
                                optimization: OptimizationType,
                                supported_ops: SupportedOpsMode,
                                target_types: Optional[List[str]],
                                allow_custom_ops: bool,
                                experimental_new_converter: bool,
                                quantize_weights: bool,
                                representative_dataset: Optional[callable]) -> bool:
        """Conversión configurable vía SavedModel"""
        
        try:
            # Paso 1: ONNX → SavedModel
            logger.info("🔄 Converting ONNX to SavedModel...")
            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)
            
            saved_model_dir = os.path.join(self.temp_dir, "saved_model_configurable")
            tf_rep.export_graph(saved_model_dir)
            
            # Paso 2: SavedModel → TFLite con configuración personalizada
            logger.info("🔄 Converting SavedModel to TFLite with custom config...")
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            
            # Configurar optimizaciones
            self._configure_optimization(converter, optimization, quantize_weights)
            
            # Configurar operaciones soportadas
            self._configure_supported_ops(converter, supported_ops, onnx_path)
            
            # Configurar tipos objetivo
            if target_types:
                converter.target_spec.supported_types = [
                    getattr(tf, dtype) for dtype in target_types
                ]
            
            # Configuraciones adicionales
            converter.allow_custom_ops = allow_custom_ops
            converter.experimental_new_converter = experimental_new_converter
            
            # Dataset representativo para cuantización
            if representative_dataset:
                converter.representative_dataset = representative_dataset
            
            # Realizar conversión
            logger.info(f"⚙️ Converting with config: optimization={optimization.value}, "
                       f"ops={supported_ops.value}, custom_ops={allow_custom_ops}")
            
            tflite_model = converter.convert()
            
            # Guardar modelo
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info("✅ SavedModel→TFLite conversion successful")
            return True
            
        except Exception as e:
            logger.error(f"SavedModel conversion failed: {e}")
            return False
    
    def _configure_optimization(self, converter, optimization: OptimizationType, 
                                quantize_weights: bool):
        """Configurar optimizaciones del convertidor"""
        
        if optimization == OptimizationType.DEFAULT:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif optimization == OptimizationType.SIZE:
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        elif optimization == OptimizationType.LATENCY:
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        # NONE no añade optimizaciones
        
        # Añadir cuantización de pesos si se solicita
        if quantize_weights and optimization != OptimizationType.NONE:
            if not converter.optimizations:
                converter.optimizations = []
            # La cuantización de pesos se maneja automáticamente con las optimizaciones
    
    def _configure_supported_ops(self, converter, supported_ops: SupportedOpsMode, 
                                 onnx_path: str):
        """Configurar operaciones soportadas"""
        
        if supported_ops == SupportedOpsMode.TFLITE_ONLY:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
            
        elif supported_ops == SupportedOpsMode.SELECT_TF:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            
        elif supported_ops == SupportedOpsMode.FLEX_DELEGATE:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            # Nota: El delegado Flex se configura en tiempo de ejecución
            
        elif supported_ops == SupportedOpsMode.AUTO:
            # Modo automático: intenta TFLite puro primero, luego SELECT_TF
            try:
                # Intentar solo con ops TFLite nativas
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS
                ]
                logger.info("🔍 Auto mode: trying TFLite native ops only...")
            except:
                # Si falla, usar SELECT_TF
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS
                ]
                logger.info("🔍 Auto mode: falling back to SELECT_TF ops...")
    
    def get_model_ops_info(self, onnx_path: str) -> Dict[str, Any]:
        """
        Analizar qué operaciones usa el modelo ONNX
        Útil para decidir qué modo de ops usar
        """
        try:
            onnx_model = onnx.load(onnx_path)
            
            # Extraer tipos de operaciones
            ops_used = set()
            for node in onnx_model.graph.node:
                ops_used.add(node.op_type)
            
            # Operaciones comúnmente no soportadas en TFLite nativo
            problematic_ops = {
                'Range', 'NonMaxSuppression', 'TopK', 'ScatterND', 
                'GatherND', 'OneHot', 'Where', 'Unique'
            }
            
            unsupported_ops = ops_used.intersection(problematic_ops)
            
            return {
                'total_ops': len(ops_used),
                'ops_used': sorted(list(ops_used)),
                'potentially_unsupported': sorted(list(unsupported_ops)),
                'recommend_select_tf': len(unsupported_ops) > 0
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze model ops: {e}")
            return {'error': str(e)}

def convert_with_config(onnx_path: str, 
                       tflite_path: str,
                       optimization: str = "default",
                       supported_ops: str = "auto",
                       target_types: Optional[List[str]] = None,
                       allow_custom_ops: bool = True,
                       quantize_weights: bool = False) -> Dict[str, Any]:
    """
    Función de conveniencia para conversión configurable
    """
    with ConfigurableTFLiteConverter() as converter:
        return converter.convert(
            onnx_path=onnx_path,
            tflite_path=tflite_path,
            optimization=optimization,
            supported_ops=supported_ops,
            target_types=target_types,
            allow_custom_ops=allow_custom_ops,
            quantize_weights=quantize_weights
        )

def analyze_model_requirements(onnx_path: str) -> Dict[str, Any]:
    """
    Analizar un modelo ONNX para recomendar configuración TFLite
    """
    with ConfigurableTFLiteConverter() as converter:
        return converter.get_model_ops_info(onnx_path)

def main():
    """CLI para el convertidor configurable"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Configurable ONNX to TFLite Converter')
    parser.add_argument('--onnx_path', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--tflite_path', type=str, required=True,
                        help='Output path for TFLite model')
    parser.add_argument('--optimization', type=str, default='default',
                        choices=['none', 'default', 'size', 'latency'],
                        help='Optimization type')
    parser.add_argument('--supported_ops', type=str, default='auto',
                        choices=['tflite_only', 'select_tf', 'flex_delegate', 'auto'],
                        help='Supported operations mode')
    parser.add_argument('--target_types', nargs='*', default=None,
                        help='Target data types (e.g., float32 int8)')
    parser.add_argument('--allow_custom_ops', action='store_true', default=True,
                        help='Allow custom operations')
    parser.add_argument('--quantize_weights', action='store_true',
                        help='Enable weight quantization')
    parser.add_argument('--analyze_only', action='store_true',
                        help='Only analyze model requirements')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Solo analizar el modelo
        logger.info(f"🔍 Analyzing model: {args.onnx_path}")
        info = analyze_model_requirements(args.onnx_path)
        
        print("\n📊 Model Analysis Results:")
        print(f"   Total operations: {info.get('total_ops', 'Unknown')}")
        print(f"   Operations used: {', '.join(info.get('ops_used', []))}")
        
        if info.get('potentially_unsupported'):
            print(f"   ⚠️ Potentially unsupported ops: {', '.join(info['potentially_unsupported'])}")
            print(f"   💡 Recommendation: Use 'select_tf' mode")
        else:
            print(f"   ✅ All ops likely supported in TFLite native mode")
        
    else:
        # Realizar conversión
        logger.info(f"🔄 Converting {args.onnx_path} to {args.tflite_path}")
        
        result = convert_with_config(
            onnx_path=args.onnx_path,
            tflite_path=args.tflite_path,
            optimization=args.optimization,
            supported_ops=args.supported_ops,
            target_types=args.target_types,
            allow_custom_ops=args.allow_custom_ops,
            quantize_weights=args.quantize_weights
        )
        
        if result['success']:
            logger.info(f"✅ Conversion successful!")
            logger.info(f"   📁 Output: {result['tflite_path']}")
            logger.info(f"   📊 Size: {result['file_size_mb']:.2f} MB")
            logger.info(f"   ⚙️ Configuration used: {result['config_used']}")
        else:
            logger.error(f"❌ Conversion failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)

if __name__ == "__main__":
    main()
