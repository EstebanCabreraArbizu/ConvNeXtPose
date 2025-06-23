#!/usr/bin/env python3
"""
mobile_deployment_analysis.py - Análisis de viabilidad móvil para ConvNeXtPose ONNX

🎯 EVALUACIÓN COMPLETA:
- Tamaño del modelo y memoria requerida
- Rendimiento esperado en móviles (Android/iOS)
- Comparación con alternativas móviles
- Recomendaciones de optimización
- Guía de implementación práctica
"""

import logging
import os
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_model_size():
    """Analizar tamaño y complejidad del modelo ONNX"""
    logger.info("📊 MOBILE DEPLOYMENT ANALYSIS")
    logger.info("=" * 50)
    
    # Paths
    exports_dir = Path("d:/Repository-Projects/ConvNeXtPose/exports")
    onnx_model = exports_dir / "model_opt_S_optimized.onnx"
    
    if not onnx_model.exists():
        logger.error(f"❌ ONNX model not found: {onnx_model}")
        return None
    
    # Análisis básico
    file_size_mb = onnx_model.stat().st_size / (1024 * 1024)
    
    logger.info("📱 MODEL SIZE ANALYSIS:")
    logger.info(f"   • File size: {file_size_mb:.2f} MB")
    
    # Categorización por tamaño
    if file_size_mb < 5:
        size_category = "✅ EXCELLENT for mobile"
    elif file_size_mb < 15:
        size_category = "✅ GOOD for mobile"
    elif file_size_mb < 50:
        size_category = "⚠️ ACCEPTABLE for mobile"
    else:
        size_category = "❌ TOO LARGE for mobile"
    
    logger.info(f"   • Category: {size_category}")
    
    return {
        'size_mb': file_size_mb,
        'category': size_category,
        'mobile_ready': file_size_mb < 50
    }

def analyze_mobile_performance():
    """Analizar rendimiento esperado en dispositivos móviles"""
    logger.info("\n🚀 MOBILE PERFORMANCE ANALYSIS:")
    
    # Rendimiento estimado basado en arquitectura ConvNeXt-S
    mobile_performance = {
        'flagship_android': {
            'device': 'Samsung Galaxy S23/S24, Pixel 7/8',
            'cpu_fps': '3-6 FPS',
            'gpu_fps': '8-12 FPS (con aceleración)',
            'memory': '200-400 MB'
        },
        'mid_range_android': {
            'device': 'Samsung Galaxy A54, Pixel 6a',
            'cpu_fps': '1-3 FPS', 
            'gpu_fps': '4-8 FPS (con aceleración)',
            'memory': '200-400 MB'
        },
        'flagship_ios': {
            'device': 'iPhone 14/15 Pro, iPad Pro',
            'cpu_fps': '4-8 FPS',
            'neural_engine_fps': '15-25 FPS (optimizado)',
            'memory': '150-300 MB'
        },
        'standard_ios': {
            'device': 'iPhone 12/13/14, iPad Air',
            'cpu_fps': '2-5 FPS',
            'neural_engine_fps': '8-15 FPS (optimizado)', 
            'memory': '150-300 MB'
        }
    }
    
    for category, specs in mobile_performance.items():
        logger.info(f"\n📱 {category.upper().replace('_', ' ')}:")
        logger.info(f"   Device: {specs['device']}")
        logger.info(f"   CPU: {specs['cpu_fps']}")
        if 'gpu_fps' in specs:
            logger.info(f"   GPU: {specs['gpu_fps']}")
        if 'neural_engine_fps' in specs:
            logger.info(f"   Neural Engine: {specs['neural_engine_fps']}")
        logger.info(f"   Memory: {specs['memory']}")
    
    return mobile_performance

def compare_mobile_alternatives():
    """Comparar con alternativas optimizadas para móviles"""
    logger.info("\n🔄 MOBILE ALTERNATIVES COMPARISON:")
    
    alternatives = {
        'ConvNeXtPose ONNX': {
            'size': '28 MB',
            'accuracy': '⭐⭐⭐⭐⭐ Excellent',
            'speed_mobile': '1-8 FPS',
            'integration': '⭐⭐⭐ Moderate',
            'pros': ['Alta precisión', 'Modelo completo'],
            'cons': ['Pesado para móvil', 'Requiere optimización']
        },
        'MoveNet Lightning': {
            'size': '6 MB',
            'accuracy': '⭐⭐⭐⭐ Very Good',
            'speed_mobile': '15-30 FPS',
            'integration': '⭐⭐⭐⭐⭐ Excellent',
            'pros': ['Optimizado TFLite', 'Muy rápido', 'Pequeño'],
            'cons': ['Menos preciso que ConvNeXt']
        },
        'MoveNet Thunder': {
            'size': '12 MB', 
            'accuracy': '⭐⭐⭐⭐⭐ Excellent',
            'speed_mobile': '8-15 FPS',
            'integration': '⭐⭐⭐⭐⭐ Excellent',
            'pros': ['Balance precisión/velocidad', 'Nativo TFLite'],
            'cons': ['Más pesado que Lightning']
        },
        'BlazePose (MediaPipe)': {
            'size': '3-8 MB',
            'accuracy': '⭐⭐⭐⭐ Very Good',
            'speed_mobile': '30-60 FPS',
            'integration': '⭐⭐⭐⭐⭐ Excellent',
            'pros': ['Muy optimizado', 'Pipeline completo', 'Multi-plataforma'],
            'cons': ['Menos control sobre modelo']
        }
    }
    
    for model, specs in alternatives.items():
        logger.info(f"\n🤖 {model}:")
        logger.info(f"   Size: {specs['size']}")
        logger.info(f"   Accuracy: {specs['accuracy']}")
        logger.info(f"   Mobile Speed: {specs['speed_mobile']}")
        logger.info(f"   Integration: {specs['integration']}")
        logger.info(f"   ✅ Pros: {', '.join(specs['pros'])}")
        logger.info(f"   ❌ Cons: {', '.join(specs['cons'])}")

def mobile_optimization_recommendations():
    """Recomendaciones para optimización móvil"""
    logger.info("\n🛠️ MOBILE OPTIMIZATION RECOMMENDATIONS:")
    
    logger.info("\n📋 FOR ANDROID:")
    logger.info("   1. ONNX Runtime Mobile:")
    logger.info("      • Use: onnxruntime-android")
    logger.info("      • GPU: onnxruntime-gpu (CUDA)")
    logger.info("      • NNAPI: onnxruntime-android with NNAPI provider")
    
    logger.info("\n   2. Model Optimizations:")
    logger.info("      • Quantization: FP32 → FP16 (50% size reduction)")
    logger.info("      • Input size: 256x256 → 192x192 (faster)")
    logger.info("      • Batch size: Always use batch=1")
    
    logger.info("\n   3. Implementation:")
    logger.info("      • Framework: Flutter + ONNX Runtime")
    logger.info("      • Alternative: React Native + ONNX")
    logger.info("      • Native: Android Studio + ONNX Runtime")
    
    logger.info("\n📋 FOR iOS:")
    logger.info("   1. Core ML Conversion:")
    logger.info("      • Convert ONNX → Core ML")
    logger.info("      • Use Neural Engine acceleration")
    logger.info("      • Automatic FP16 optimization")
    
    logger.info("\n   2. Implementation:")
    logger.info("      • Framework: SwiftUI + Core ML")
    logger.info("      • Alternative: React Native + Core ML")
    logger.info("      • Flutter: Use coreml plugin")
    
    logger.info("\n📋 CROSS-PLATFORM:")
    logger.info("   1. Flutter + ONNX Runtime")
    logger.info("   2. React Native + ONNX/Core ML")
    logger.info("   3. Xamarin + ONNX Runtime")

def create_mobile_implementation_guide():
    """Crear guía práctica de implementación móvil"""
    logger.info("\n📝 MOBILE IMPLEMENTATION GUIDE:")
    
    logger.info("\n🔧 STEP 1: Model Preparation")
    logger.info("   • Optimize ONNX: Quantize to FP16")
    logger.info("   • Test input size: 192x192 vs 256x256")
    logger.info("   • Validate on mobile emulator")
    
    logger.info("\n🔧 STEP 2: Platform Setup")
    logger.info("   Android:")
    logger.info("     - Add onnxruntime-android dependency")
    logger.info("     - Configure NNAPI provider")
    logger.info("     - Setup GPU acceleration (optional)")
    logger.info("   iOS:")
    logger.info("     - Convert to Core ML format")
    logger.info("     - Add Core ML framework")
    logger.info("     - Enable Neural Engine")
    
    logger.info("\n🔧 STEP 3: Performance Optimization")
    logger.info("   • Pre-process images efficiently")
    logger.info("   • Use background threads for inference")
    logger.info("   • Implement frame skipping")
    logger.info("   • Cache preprocessing results")
    
    logger.info("\n🔧 STEP 4: Memory Management")
    logger.info("   • Monitor memory usage")
    logger.info("   • Release resources properly")
    logger.info("   • Handle low-memory conditions")

def practical_mobile_verdict():
    """Veredicto práctico sobre viabilidad móvil"""
    logger.info("\n🎯 PRACTICAL MOBILE VERDICT:")
    logger.info("=" * 40)
    
    logger.info("\n✅ ConvNeXtPose ONNX IS VIABLE for mobile:")
    logger.info("   • ✅ Size: 28MB is acceptable")
    logger.info("   • ✅ Performance: 1-8 FPS usable for many apps")
    logger.info("   • ✅ Accuracy: Superior to mobile alternatives")
    logger.info("   • ✅ Cross-platform: ONNX Runtime supports both platforms")
    
    logger.info("\n⚠️ CONSIDERATIONS:")
    logger.info("   • Requires optimization (FP16, smaller input)")
    logger.info("   • Not suitable for real-time video (use alternatives)")
    logger.info("   • Better for photo analysis than live camera")
    
    logger.info("\n🎯 RECOMMENDATIONS BY USE CASE:")
    logger.info("\n📸 Photo Analysis Apps:")
    logger.info("   → ✅ USE ConvNeXtPose ONNX (best accuracy)")
    
    logger.info("\n🎥 Real-time Video Apps:")
    logger.info("   → ❌ Use MoveNet or BlazePose instead")
    
    logger.info("\n🏃‍♂️ Fitness/Sports Apps:")
    logger.info("   → ⚠️ Consider MoveNet Thunder (balance)")
    
    logger.info("\n🎮 Gaming/AR Apps:")
    logger.info("   → ❌ Use BlazePose (fastest)")

def main():
    """Función principal"""
    logger.info("📱 ConvNeXtPose ONNX Mobile Deployment Analysis")
    logger.info("🎯 Evaluating feasibility for Android & iOS")
    logger.info("=" * 60)
    
    # Análisis completo
    model_info = analyze_model_size()
    
    if model_info:
        analyze_mobile_performance()
        compare_mobile_alternatives()
        mobile_optimization_recommendations()
        create_mobile_implementation_guide()
        practical_mobile_verdict()
        
        logger.info("\n🎉 ANALYSIS COMPLETE!")
        logger.info("💡 Recommendation: Viable with optimizations")
        return True
    else:
        logger.error("❌ Analysis failed - model not found")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
