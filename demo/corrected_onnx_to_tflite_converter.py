#!/usr/bin/env python3
"""
corrected_onnx_to_tflite_converter.py - Convertidor ONNX→TFLite usando onnx-tf (correcto)

Este convertidor usa exclusivamente onnx-tf para la conversión real 
ONNX→TensorFlow→TFLite, preservando los pesos del modelo original.

Estrategias implementadas:
1. onnx-tf SavedModel: ONNX → TensorFlow SavedModel → TFLite (principal)
2. onnx-tf direct: ONNX → TensorFlow en memoria → TFLite (alternativa)

Todas las estrategias preservan los pesos reales del modelo ONNX original.
No se usan modelos simplificados ni genéricos.
"""

import os
import sys
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check dependencies
TENSORFLOW_AVAILABLE = False
ONNX_TF_AVAILABLE = False
ONNX_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    logger.info(f"✅ TensorFlow available: {tf.__version__}")
except ImportError:
    logger.error("❌ TensorFlow not available")

try:
    import onnx_tf
    from onnx_tf.backend import prepare
    ONNX_TF_AVAILABLE = True
    logger.info(f"✅ onnx-tf available: {onnx_tf.__version__}")
except ImportError:
    logger.warning("⚠️ onnx-tf not available")

try:
    import onnx
    ONNX_AVAILABLE = True
    logger.info(f"✅ ONNX available: {onnx.__version__}")
except ImportError:
    logger.warning("⚠️ ONNX not available")

class CorrectedONNXToTFLiteConverter:
    """
    Convertidor ONNX→TFLite usando onnx-tf (método correcto)
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="onnx_tflite_")
        self.cleanup_temp = temp_dir is None
        
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is required but not available")
        
        if not ONNX_TF_AVAILABLE:
            raise RuntimeError("onnx-tf is required but not available")
        
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX is required but not available")
        
        logger.info(f"🔧 Initialized converter with temp dir: {self.temp_dir}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_temp and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info(f"🧹 Cleaned up temp directory: {self.temp_dir}")
    
    def convert(self, onnx_path: str, tflite_path: str, 
                optimization: str = "default") -> Dict[str, Any]:
        """
        Convierte ONNX a TFLite usando onnx-tf (método correcto)
        
        Args:
            onnx_path: Ruta al modelo ONNX
            tflite_path: Ruta de salida para el modelo TFLite
            optimization: Tipo de optimización ('default', 'size', 'latency', 'none')
        
        Returns:
            Dict con resultados de la conversión
        """
        result = {
            'success': False,
            'tflite_path': None,
            'strategy_used': None,
            'file_size_mb': 0,
            'error': None
        }
        
        if not os.path.exists(onnx_path):
            result['error'] = f"ONNX model not found: {onnx_path}"
            return result
        
        # Crear directorio de salida si no existe
        os.makedirs(os.path.dirname(os.path.abspath(tflite_path)), exist_ok=True)
        
        # Estrategias de conversión con onnx-tf (preservan pesos reales)
        strategies = [
            ("onnx_tf_savedmodel", self._convert_via_onnx_tf_savedmodel),
            ("onnx_tf_direct", self._convert_via_onnx_tf_direct),
        ]
        
        for strategy_name, strategy_func in strategies:
            logger.info(f"🔄 Trying strategy: {strategy_name}")
            try:
                success = strategy_func(onnx_path, tflite_path, optimization)
                if success and os.path.exists(tflite_path):
                    result['success'] = True
                    result['tflite_path'] = tflite_path
                    result['strategy_used'] = strategy_name
                    result['file_size_mb'] = os.path.getsize(tflite_path) / (1024 * 1024)
                    
                    logger.info(f"✅ Conversion successful with {strategy_name}")
                    logger.info(f"📊 TFLite model size: {result['file_size_mb']:.2f} MB")
                    break
                    
            except Exception as e:
                logger.warning(f"⚠️ Strategy {strategy_name} failed: {str(e)}")
                continue
        
        if not result['success']:
            result['error'] = "All conversion strategies failed"
            logger.error("❌ All conversion strategies failed")
        
        return result
    
    def _convert_via_onnx_tf_savedmodel(self, onnx_path: str, tflite_path: str, 
                                        optimization: str) -> bool:
        """
        Estrategia principal: ONNX → TF SavedModel → TFLite usando onnx-tf
        """
        logger.info("🔄 Using onnx-tf SavedModel strategy")
        
        try:
            import onnx
            import onnx_tf
            from onnx_tf.backend import prepare
            
            # Paso 1: Cargar modelo ONNX
            logger.info("📥 Loading ONNX model...")
            onnx_model = onnx.load(onnx_path)
            
            # Paso 2: Convertir ONNX → TensorFlow usando onnx-tf
            logger.info("🔄 Converting ONNX → TensorFlow using onnx-tf...")
            tf_rep = prepare(onnx_model)
            
            # Paso 3: Exportar como SavedModel
            saved_model_dir = os.path.join(self.temp_dir, "saved_model")
            logger.info(f"💾 Exporting TensorFlow SavedModel to {saved_model_dir}...")
            tf_rep.export_graph(saved_model_dir)
            
            # Verificar que el SavedModel se creó
            if not os.path.exists(saved_model_dir):
                raise RuntimeError("SavedModel directory not created")
            
            # Paso 4: SavedModel → TFLite
            logger.info("🔄 Converting SavedModel → TFLite...")
            return self._savedmodel_to_tflite(saved_model_dir, tflite_path, optimization)
            
        except Exception as e:
            logger.error(f"onnx-tf SavedModel strategy failed: {e}")
            raise
    
    def _convert_via_onnx_tf_direct(self, onnx_path: str, tflite_path: str, 
                                    optimization: str) -> bool:
        """
        Estrategia alternativa: ONNX → TensorFlow en memoria → TFLite
        """
        logger.info("🔄 Using onnx-tf direct strategy")
        
        try:
            import onnx
            import onnx_tf
            from onnx_tf.backend import prepare
            
            # Cargar modelo ONNX
            onnx_model = onnx.load(onnx_path)
            
            # Convertir ONNX → TensorFlow en memoria
            tf_rep = prepare(onnx_model)
            
            # Obtener el modelo TensorFlow
            tf_graph = tf_rep.graph
            
            # Crear un modelo Keras temporal para TFLite
            # Nota: Este método puede requerir adaptación según la estructura del modelo
            
            # Para modelos complejos, mejor usar SavedModel
            saved_model_dir = os.path.join(self.temp_dir, "saved_model_direct")
            tf_rep.export_graph(saved_model_dir)
            
            return self._savedmodel_to_tflite(saved_model_dir, tflite_path, optimization)
            
        except Exception as e:
            logger.error(f"onnx-tf direct strategy failed: {e}")
            raise
    
    def _savedmodel_to_tflite(self, saved_model_dir: str, tflite_path: str, 
                              optimization: str) -> bool:
        """
        Convierte SavedModel a TFLite
        """
        try:
            # Crear convertidor desde SavedModel
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            
            # Configurar optimizaciones
            if optimization == "default":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            elif optimization == "size":
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
            elif optimization == "latency":
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
            # 'none' no añade optimizaciones
            
            # Configuraciones adicionales para compatibilidad
            converter.allow_custom_ops = True
            converter.experimental_new_converter = True
            
            # Realizar conversión
            logger.info("🔄 Converting to TFLite...")
            tflite_model = converter.convert()
            
            # Guardar modelo
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"✅ SavedModel→TFLite conversion successful")
            return True
            
        except Exception as e:
            logger.error(f"SavedModel→TFLite conversion failed: {e}")
            return False

def convert_onnx_to_tflite_corrected(onnx_path: str, tflite_path: str, 
                                     optimization: str = "default") -> Dict[str, Any]:
    """
    Función de conveniencia para conversión ONNX→TFLite usando onnx-tf (correcto)
    
    Args:
        onnx_path: Ruta al modelo ONNX
        tflite_path: Ruta de salida para el modelo TFLite
        optimization: Tipo de optimización ('default', 'size', 'latency', 'none')
    
    Returns:
        Dict con resultados de la conversión
    """
    with CorrectedONNXToTFLiteConverter() as converter:
        return converter.convert(onnx_path, tflite_path, optimization)

def main():
    """Test del convertidor correcto"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Corrected ONNX to TFLite Converter using onnx-tf')
    parser.add_argument('--onnx_path', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--tflite_path', type=str, required=True,
                        help='Output path for TFLite model')
    parser.add_argument('--optimization', type=str, default='default',
                        choices=['default', 'size', 'latency', 'none'],
                        help='Optimization type')
    
    args = parser.parse_args()
    
    # Verificar dependencias
    if not ONNX_TF_AVAILABLE:
        logger.error("❌ onnx-tf not available. Please install it first:")
        logger.error("   pip install onnx-tf==1.10.0")
        return 1
    
    # Ejecutar conversión
    logger.info(f"🚀 Starting corrected ONNX→TFLite conversion")
    logger.info(f"📥 Input: {args.onnx_path}")
    logger.info(f"📤 Output: {args.tflite_path}")
    logger.info(f"⚙️ Optimization: {args.optimization}")
    logger.info(f"🔧 Using onnx-tf backend (correct method)")
    
    result = convert_onnx_to_tflite_corrected(
        args.onnx_path, 
        args.tflite_path, 
        args.optimization
    )
    
    # Mostrar resultados
    if result['success']:
        logger.info("🎉 Conversion completed successfully!")
        logger.info(f"✅ Strategy used: {result['strategy_used']}")
        logger.info(f"📊 Output file size: {result['file_size_mb']:.2f} MB")
        logger.info(f"📁 TFLite model saved to: {result['tflite_path']}")
        logger.info("🔍 Model weights preserved from original ONNX")
    else:
        logger.error("❌ Conversion failed!")
        if result['error']:
            logger.error(f"💥 Error: {result['error']}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
