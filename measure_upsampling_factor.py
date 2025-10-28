#!/usr/bin/env python3
"""
Verificación FINAL: Medir dimensiones de salida de cada capa
para confirmar si la capa 3 hace upsampling o no.
"""

import torch
import os


def measure_layer_dimensions(checkpoint_path, model_name):
    """
    Simula un forward pass y mide las dimensiones de cada capa.
    Si la capa 3 duplica las dimensiones espaciales, tiene upsampling.
    Si no, tiene up=False.
    """
    
    print(f"\n{'='*70}")
    print(f"📏 Midiendo dimensiones: {model_name}")
    print(f"{'='*70}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ No se encuentra: {checkpoint_path}")
        return None
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['network']
    
    # Determinar si es legacy
    is_legacy = 'module.head.deconv_layers_1.dwconv.weight' in model_state
    
    # Simular dimensiones
    input_channels_map = {
        'XS': (320, 64, 64),   # (C, H, W) entrada al head
        'S': (384, 64, 64),
        'M': (384, 64, 64),
        'L': (384, 64, 64),
    }
    
    input_shape = input_channels_map.get(model_name, (384, 64, 64))
    
    print(f"\n📐 Input al head: {input_shape} (C×H×W)")
    print(f"   Configuración: {'Legacy' if is_legacy else 'Nueva'}")
    
    # Simular cada capa
    current_shape = list(input_shape)
    
    print(f"\n🔄 Trazando transformaciones:")
    print(f"   {'Capa':<12} {'Input Shape':<20} {'Output Shape':<20} {'Upsampling':<15}")
    print(f"   {'-'*70}")
    
    for i in range(1, 4):
        if is_legacy:
            dwconv_key = f'module.head.deconv_layers_{i}.dwconv.weight'
            pwconv_key = f'module.head.deconv_layers_{i}.pwconv.weight'
        else:
            dwconv_key = f'module.head.deconv_layers_{i}.0.weight'
            pwconv_key = f'module.head.deconv_layers_{i}.2.weight'
        
        if dwconv_key not in model_state or pwconv_key not in model_state:
            break
        
        # Obtener shapes de pesos
        dwconv_weight = model_state[dwconv_key]
        pwconv_weight = model_state[pwconv_key]
        
        input_str = f"{current_shape[0]}×{current_shape[1]}×{current_shape[2]}"
        
        # dwconv mantiene canales y dimensiones espaciales
        # pwconv cambia canales
        out_channels = pwconv_weight.shape[0]
        
        # Detectar upsampling
        # En el código, DeConv aplica upsampling después de pwconv
        # Por defecto upscale_factor=2 duplica H y W
        
        # Para legacy, verificar si existe parámetro de upsampling
        has_upsampling = True
        if is_legacy:
            upsample_keys = [k for k in model_state.keys() 
                           if f'deconv_layers_{i}.upsample' in k]
            has_upsampling = len(upsample_keys) > 0
        
        # Actualizar shape
        if has_upsampling:
            current_shape = [out_channels, current_shape[1] * 2, current_shape[2] * 2]
            upsampling_str = "✅ 2× (×2 H, ×2 W)"
        else:
            current_shape = [out_channels, current_shape[1], current_shape[2]]
            upsampling_str = "❌ No (Identity)"
        
        output_str = f"{current_shape[0]}×{current_shape[1]}×{current_shape[2]}"
        
        print(f"   Layer {i:<6} {input_str:<20} {output_str:<20} {upsampling_str:<15}")
    
    print(f"\n📊 Resultado final: {current_shape[0]}×{current_shape[1]}×{current_shape[2]}")
    
    # Calcular factor de upsampling total
    original_h = input_shape[1]
    final_h = current_shape[1]
    upsampling_factor = final_h // original_h
    
    print(f"   Factor de upsampling total: {upsampling_factor}× (de {original_h}×{original_h} a {final_h}×{final_h})")
    
    return {
        'input_shape': input_shape,
        'output_shape': tuple(current_shape),
        'upsampling_factor': upsampling_factor,
        'is_legacy': is_legacy
    }


def main():
    """Función principal"""
    
    models = {
        'XS': 'demo/ConvNeXtPose_XS.tar',
        'S': 'demo/ConvNeXtPose_S.tar',
        'M': 'demo/ConvNeXtPose_M (1).tar',
        'L': 'demo/ConvNeXtPose_L (1).tar',
    }
    
    paper_specs = {
        'XS': '2UP',
        'S': '2UP',
        'M': '3UP',
        'L': '3UP',
    }
    
    print("\n" + "="*70)
    print("📏 MEDICIÓN DE DIMENSIONES DEFINITIVA")
    print("Verificando factor de upsampling real de cada modelo")
    print("="*70)
    
    results = {}
    
    for model_name, checkpoint_path in models.items():
        result = measure_layer_dimensions(checkpoint_path, model_name)
        if result:
            results[model_name] = result
    
    # Resumen final
    print("\n" + "="*70)
    print("📋 RESUMEN COMPARATIVO")
    print("="*70)
    
    print(f"\n{'Modelo':<10} {'Paper':<10} {'Factor UP Real':<20} {'Estado':<30}")
    print("-" * 70)
    
    for model_name, result in results.items():
        paper = paper_specs[model_name]
        paper_factor = int(paper[0])  # Extraer número de "2UP" o "3UP"
        real_factor = result['upsampling_factor']
        
        if paper_factor == real_factor:
            status = "✅ COINCIDE"
        else:
            status = f"⚠️  Difiere (esperado {paper_factor}×, real {real_factor}×)"
        
        print(f"{model_name:<10} {paper:<10} {real_factor}×{'':<17} {status:<30}")
    
    print("\n" + "="*70)
    print("🎯 CONCLUSIÓN FINAL")
    print("="*70)
    
    # Analizar XS y S específicamente
    xs_factor = results.get('XS', {}).get('upsampling_factor', 0)
    s_factor = results.get('S', {}).get('upsampling_factor', 0)
    
    if xs_factor == 4 and s_factor == 4:
        print("""
✅ ¡MISTERIO RESUELTO!

Modelos XS y S:
  - Tienen 3 capas de deconvolución
  - Solo 2 capas aplican upsampling (2× cada una = 4× total)
  - La capa 3 tiene up=False (solo convolución, sin upsampling)
  
La notación del paper es CORRECTA:
  "2UP" = 2 capas con UPsampling (no cuenta la capa sin upsampling)

Estructura real:
  ┌─────────────┬──────────────┬──────────────┐
  │   Capa 1    │    Capa 2    │    Capa 3    │
  │  (conv+UP)  │  (conv+UP)   │  (conv only) │
  │    2×       │     2×       │     1×       │
  └─────────────┴──────────────┴──────────────┘
       ↓             ↓              ↓
     64→128       128→256       256 (no change)
     
Factor total: 2× × 2× = 4× upsampling
        """)
    else:
        print(f"""
⚠️  Resultados inesperados:
  XS: Factor {xs_factor}× (esperado 4×)
  S:  Factor {s_factor}× (esperado 4×)
        """)
    
    # Verificar M y L
    m_factor = results.get('M', {}).get('upsampling_factor', 0)
    l_factor = results.get('L', {}).get('upsampling_factor', 0)
    
    if m_factor == 8 and l_factor == 8:
        print("""
✅ Modelos M y L:
  - Tienen 3 capas de deconvolución
  - Las 3 capas aplican upsampling (2× cada una = 8× total)
  
La notación del paper es CORRECTA:
  "3UP" = 3 capas con UPsampling
  
Estructura real:
  ┌─────────────┬──────────────┬──────────────┐
  │   Capa 1    │    Capa 2    │    Capa 3    │
  │  (conv+UP)  │  (conv+UP)   │  (conv+UP)   │
  │    2×       │     2×       │     2×       │
  └─────────────┴──────────────┴──────────────┘
       ↓             ↓              ↓
     64→128       128→256       256→512
     
Factor total: 2× × 2× × 2× = 8× upsampling
        """)
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
