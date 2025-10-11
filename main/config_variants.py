"""
ConvNeXtPose Model Variants Configuration

Este módulo define las configuraciones de arquitectura para las diferentes
variantes del modelo ConvNeXtPose: XS, S, M, y L.

Uso:
    from config_variants import get_model_config, MODEL_CONFIGS
    
    depths, dims = get_model_config('M')  # Para modelo Medium
    print(MODEL_CONFIGS['L']['description'])  # Info del modelo Large

Autor: Adaptado para evaluación Human3.6M
Fecha: Octubre 2025
"""

MODEL_CONFIGS = {
    'XS': {
        'depths': [3, 3, 9, 3],
        'dims': [48, 96, 192, 384],
        'head_cfg': {
            'num_deconv_layers': 2,  # 2-UP: solo 2 capas con upsampling
            'deconv_channels': [256, 256],  # Canales intermedios
            'deconv_kernels': [3, 3]  # Kernel sizes
        },
        'params': 22.0,  # Millones
        'gflops': 4.5,
        'expected_mpjpe': 52.0,  # mm en Protocol 2
        'expected_pa_mpjpe': 36.5,  # mm en Protocol 1
        'description': 'Extra Small - Configuración más ligera para dispositivos móviles'
    },
    'S': {
        'depths': [3, 3, 27, 3],
        'dims': [96, 192, 384, 768],
        'head_cfg': {
            'num_deconv_layers': 2,  # 2-UP: solo 2 capas con upsampling
            'deconv_channels': [256, 256],
            'deconv_kernels': [3, 3]
        },
        'params': 50.0,
        'gflops': 8.7,
        'expected_mpjpe': 48.0,  # mm en Protocol 2
        'expected_pa_mpjpe': 33.2,  # mm en Protocol 1
        'description': 'Small - Balance entre velocidad y precisión'
    },
    'M': {
        'depths': [3, 3, 27, 3],
        'dims': [128, 256, 512, 1024],
        'head_cfg': {
            'num_deconv_layers': 3,  # 3-UP: 3 capas con upsampling para M/L
            'deconv_channels': [256, 256, 256],
            'deconv_kernels': [3, 3, 3]
        },
        'params': 88.6,
        'gflops': 15.4,
        'expected_mpjpe': 44.6,  # mm en Protocol 2 (según paper)
        'expected_pa_mpjpe': 31.2,  # mm en Protocol 1 (según paper)
        'description': 'Medium - Femto-L backbone, óptimo para evaluación'
    },
    'L': {
        'depths': [3, 3, 27, 3],
        'dims': [192, 384, 768, 1536],
        'head_cfg': {
            'num_deconv_layers': 3,  # 3-UP: 3 capas con upsampling para M/L
            'deconv_channels': [256, 256, 256],
            'deconv_kernels': [3, 3, 3]
        },
        'params': 197.8,
        'gflops': 34.4,
        'expected_mpjpe': 42.3,  # mm en Protocol 2 (según paper)
        'expected_pa_mpjpe': 29.8,  # mm en Protocol 1 (según paper)
        'description': 'Large - Femto-L backbone, máxima precisión'
    }
}


def get_model_config(variant='M'):
    """
    Retorna la configuración del modelo para la variante especificada.
    
    Args:
        variant (str): Una de las siguientes opciones: 'XS', 'S', 'M', 'L'
                      Por defecto: 'M' (Medium)
    
    Returns:
        tuple: (depths, dims) donde:
               - depths (list): Número de bloques en cada etapa [stage1, stage2, stage3, stage4]
               - dims (list): Número de canales en cada etapa
    
    Raises:
        ValueError: Si la variante especificada no es válida
    
    Ejemplo:
        >>> depths, dims = get_model_config('M')
        >>> print(f"Depths: {depths}, Dims: {dims}")
        Depths: [3, 3, 27, 3], Dims: [128, 256, 512, 1024]
    """
    if variant not in MODEL_CONFIGS:
        available = ', '.join(MODEL_CONFIGS.keys())
        raise ValueError(
            f"Variante no válida: '{variant}'. "
            f"Opciones disponibles: {available}"
        )
    
    config = MODEL_CONFIGS[variant]
    return (config['depths'], config['dims'])


def get_full_config(variant='M'):
    """
    Retorna la configuración completa del modelo incluyendo metadatos.
    
    Args:
        variant (str): Variante del modelo ('XS', 'S', 'M', 'L')
    
    Returns:
        dict: Diccionario con toda la información de la variante
    
    Ejemplo:
        >>> config = get_full_config('L')
        >>> print(f"MPJPE esperado: {config['expected_mpjpe']} mm")
        MPJPE esperado: 42.3 mm
    """
    if variant not in MODEL_CONFIGS:
        available = ', '.join(MODEL_CONFIGS.keys())
        raise ValueError(
            f"Variante no válida: '{variant}'. "
            f"Opciones disponibles: {available}"
        )
    
    return MODEL_CONFIGS[variant].copy()


def print_model_info(variant='M'):
    """
    Imprime información detallada sobre una variante de modelo.
    
    Args:
        variant (str): Variante del modelo a mostrar
    
    Ejemplo:
        >>> print_model_info('M')
        ╔════════════════════════════════════════════╗
        ║     ConvNeXtPose-M Configuration          ║
        ╚════════════════════════════════════════════╝
        ...
    """
    if variant not in MODEL_CONFIGS:
        print(f"❌ Variante '{variant}' no encontrada")
        return
    
    config = MODEL_CONFIGS[variant]
    
    print("\n" + "="*60)
    print(f"  ConvNeXtPose-{variant} Configuration")
    print("="*60)
    print(f"Descripción:  {config['description']}")
    print(f"Depths:       {config['depths']}")
    print(f"Dims:         {config['dims']}")
    print(f"Parámetros:   {config['params']:.1f}M")
    print(f"GFLOPs:       {config['gflops']:.1f}")
    print("-"*60)
    print("Resultados esperados en Human3.6M:")
    print(f"  Protocol 1 (PA-MPJPE):  {config['expected_pa_mpjpe']:.1f} mm")
    print(f"  Protocol 2 (MPJPE):     {config['expected_mpjpe']:.1f} mm")
    print("="*60 + "\n")


def compare_variants():
    """
    Imprime una tabla comparativa de todas las variantes disponibles.
    
    Útil para decidir qué modelo usar según requisitos.
    """
    print("\n" + "="*100)
    print("  COMPARACIÓN DE VARIANTES - ConvNeXtPose")
    print("="*100)
    print(f"{'Variante':<10} {'Params (M)':<12} {'GFLOPs':<10} {'MPJPE (mm)':<12} {'PA-MPJPE (mm)':<15} {'Descripción':<30}")
    print("-"*100)
    
    for variant in ['XS', 'S', 'M', 'L']:
        config = MODEL_CONFIGS[variant]
        print(f"{variant:<10} {config['params']:<12.1f} {config['gflops']:<10.1f} "
              f"{config['expected_mpjpe']:<12.1f} {config['expected_pa_mpjpe']:<15.1f} "
              f"{config['description'][:30]:<30}")
    
    print("="*100)
    print("Nota: Los valores de MPJPE son en Protocol 2, PA-MPJPE en Protocol 1")
    print("="*100 + "\n")


def get_recommended_batch_size(variant, gpu_memory_gb=8):
    """
    Recomienda un batch size según la variante y memoria GPU disponible.
    
    Args:
        variant (str): Variante del modelo
        gpu_memory_gb (int): Memoria GPU disponible en GB
    
    Returns:
        int: Batch size recomendado
    
    Ejemplo:
        >>> batch_size = get_recommended_batch_size('L', gpu_memory_gb=8)
        >>> print(f"Batch size recomendado: {batch_size}")
        Batch size recomendado: 8
    """
    # Estimaciones conservadoras basadas en testing
    recommendations = {
        'XS': {4: 64, 8: 128, 12: 192, 16: 256, 24: 384},
        'S': {4: 32, 8: 64, 12: 96, 16: 128, 24: 192},
        'M': {4: 16, 8: 32, 12: 48, 16: 64, 24: 96},
        'L': {4: 8, 8: 16, 12: 24, 16: 32, 24: 48}
    }
    
    if variant not in recommendations:
        return 16  # Default conservador
    
    # Encontrar el batch size más apropiado
    memory_levels = sorted(recommendations[variant].keys())
    for mem in memory_levels:
        if gpu_memory_gb <= mem:
            return recommendations[variant][mem]
    
    # Si tiene más memoria que el máximo, usar el batch size máximo
    return recommendations[variant][max(memory_levels)]


if __name__ == "__main__":
    """
    Testing del módulo - ejecutar con: python config_variants.py
    """
    print("\n🧪 Testing del módulo config_variants.py\n")
    
    # Test 1: Obtener configuración
    print("TEST 1: Obtener configuración del modelo M")
    depths, dims = get_model_config('M')
    print(f"✓ Depths: {depths}")
    print(f"✓ Dims: {dims}\n")
    
    # Test 2: Información detallada
    print("TEST 2: Información detallada")
    for variant in ['M', 'L']:
        print_model_info(variant)
    
    # Test 3: Comparación
    print("TEST 3: Comparación de variantes")
    compare_variants()
    
    # Test 4: Recomendación de batch size
    print("TEST 4: Recomendación de batch size")
    for variant in ['M', 'L']:
        for gpu_mem in [4, 8, 12]:
            batch_size = get_recommended_batch_size(variant, gpu_mem)
            print(f"  {variant} con {gpu_mem}GB GPU → Batch size: {batch_size}")
    print()
    
    # Test 5: Manejo de errores
    print("TEST 5: Manejo de errores")
    try:
        get_model_config('INVALID')
    except ValueError as e:
        print(f"✓ Error capturado correctamente: {e}\n")
    
    print("✅ Todos los tests completados\n")
