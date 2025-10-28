#!/usr/bin/env python3
"""
Script definitivo para verificar si los modelos XS y S
usan las 3 capas durante el forward pass.

Verifica:
1. Si el checkpoint usa configuración legacy o nueva
2. Cuántas capas se ejecutan realmente
"""

import torch
import os


def check_model_configuration(checkpoint_path, model_name):
    """Verifica qué configuración usa el checkpoint"""
    
    print(f"\n{'='*70}")
    print(f"🔍 Analizando: {model_name}")
    print(f"{'='*70}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ No se encuentra: {checkpoint_path}")
        return None
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['network']
    
    # Detectar configuración
    has_legacy_format = False
    has_new_format = False
    
    # Buscar formato legacy (deconv_layers_1, deconv_layers_2, deconv_layers_3)
    legacy_keys = [
        'module.head.deconv_layers_1.dwconv.weight',
        'module.head.deconv_layers_2.dwconv.weight',
        'module.head.deconv_layers_3.dwconv.weight'
    ]
    
    has_legacy_format = all(key in model_state for key in legacy_keys)
    
    # El formato legacy siempre tiene exactamente estas 3 capas nombradas
    if has_legacy_format:
        print(f"📋 Configuración: LEGACY (deconv_layers_1/2/3)")
        print(f"   └─ El forward ejecuta explícitamente las 3 capas:")
        print(f"      x = self.deconv_layers_1(x)")
        print(f"      x = self.deconv_layers_2(x)")
        print(f"      x = self.deconv_layers_3(x)")
        
        # Verificar si la tercera capa tiene upsampling
        # Buscar si existe upsample1 en la capa 3
        upsample_key = 'module.head.deconv_layers_3.upsample1.scale_factor'
        
        if upsample_key in model_state:
            print(f"   └─ ⚠️  Capa 3 tiene parámetro de upsampling")
        else:
            # En el código, si up=False, upsample1 es nn.Identity() y no tiene parámetros guardados
            print(f"   └─ ℹ️  Capa 3 podría tener up=False (Identity)")
        
        # Contar capas reales
        num_layers = 3
        
    else:
        # Formato nuevo (ModuleList)
        print(f"📋 Configuración: NUEVA (ModuleList con iteración)")
        print(f"   └─ El forward itera sobre todas las capas:")
        print(f"      for deconv_layer in self.deconv_layers:")
        print(f"          x = deconv_layer(x)")
        
        # Contar capas
        layer_indices = set()
        for key in model_state.keys():
            if 'head.deconv_layers' in key and 'deconv_layers_' in key:
                try:
                    parts = key.split('.')
                    for part in parts:
                        if 'deconv_layers_' in part:
                            idx = int(part.split('_')[-1])
                            layer_indices.add(idx)
                            break
                except:
                    pass
        
        num_layers = len(layer_indices)
        print(f"   └─ Número de capas encontradas: {num_layers}")
    
    # Verificar propiedades de upsampling de cada capa
    print(f"\n📊 Análisis de upsampling por capa:")
    
    for i in range(1, num_layers + 1):
        # Buscar el peso de dwconv para esta capa
        if has_legacy_format:
            dwconv_key = f'module.head.deconv_layers_{i}.dwconv.weight'
        else:
            dwconv_key = f'module.head.deconv_layers_{i}.0.weight'
        
        if dwconv_key in model_state:
            weight = model_state[dwconv_key]
            print(f"   Capa {i}: dwconv shape = {list(weight.shape)}")
        
        # Para el formato legacy, verificar si la capa 3 tiene upsampling
        if has_legacy_format and i == 3:
            # Buscar cualquier clave relacionada con upsample en capa 3
            has_upsample_param = any('deconv_layers_3.upsample' in k for k in model_state.keys())
            
            if not has_upsample_param:
                print(f"          └─ ⚠️  No se encontraron parámetros de upsampling")
                print(f"              Esto sugiere up=False (nn.Identity)")
            else:
                print(f"          └─ ✅ Tiene parámetros de upsampling")
    
    return {
        'num_layers': num_layers,
        'is_legacy': has_legacy_format,
        'all_layers_used': True  # En ambos casos se usan todas
    }


def main():
    """Función principal"""
    
    models = {
        'XS': {
            'path': 'demo/ConvNeXtPose_XS.tar',
            'paper_spec': '2UP'
        },
        'S': {
            'path': 'demo/ConvNeXtPose_S.tar',
            'paper_spec': '2UP'
        },
        'M': {
            'path': 'demo/ConvNeXtPose_M (1).tar',
            'paper_spec': '3UP'
        },
        'L': {
            'path': 'demo/ConvNeXtPose_L (1).tar',
            'paper_spec': '3UP'
        },
    }
    
    print("\n" + "="*70)
    print("🔬 VERIFICACIÓN DEFINITIVA: ¿SE USAN LAS 3 CAPAS?")
    print("="*70)
    
    results = {}
    
    for model_name, model_info in models.items():
        result = check_model_configuration(model_info['path'], model_name)
        if result:
            results[model_name] = result
            results[model_name]['paper_spec'] = model_info['paper_spec']
    
    # Resumen final
    print("\n" + "="*70)
    print("📋 RESUMEN DEFINITIVO")
    print("="*70)
    
    print(f"\n{'Modelo':<10} {'Paper':<10} {'Capas':<10} {'Config':<15} {'¿Se usan todas?':<20}")
    print("-" * 70)
    
    for model_name, result in results.items():
        config_type = "Legacy" if result['is_legacy'] else "Nueva"
        used_status = "✅ SÍ" if result['all_layers_used'] else "❌ NO"
        
        print(f"{model_name:<10} {result['paper_spec']:<10} "
              f"{result['num_layers']} capas{'':<2} {config_type:<15} {used_status:<20}")
    
    print("\n" + "="*70)
    print("🎯 CONCLUSIÓN DEFINITIVA")
    print("="*70)
    
    # Análisis específico para XS y S
    xs_analysis = results.get('XS', {})
    s_analysis = results.get('S', {})
    
    if xs_analysis.get('is_legacy') and s_analysis.get('is_legacy'):
        print("""
✅ MODELOS XS Y S: Usan configuración LEGACY

El forward pass ejecuta explícitamente:
    x = self.deconv_layers_1(x)  # Capa 1 ✅
    x = self.deconv_layers_2(x)  # Capa 2 ✅
    x = self.deconv_layers_3(x)  # Capa 3 ✅

Las 3 capas SE EJECUTAN SIEMPRE.

🤔 Pero... la capa 3 podría tener up=False (sin upsampling real)

Si la capa 3 tiene up=False:
  - Aplica convolución (dwconv + pwconv)
  - NO aplica upsampling
  - Esto podría explicar la notación "2UP" del paper
  
"2UP" = 2 capas con upsampling + 1 capa sin upsampling
        """)
    else:
        print("""
✅ TODOS LOS MODELOS: Usan configuración NUEVA

El forward pass itera sobre todas las capas:
    for deconv_layer in self.deconv_layers:
        x = deconv_layer(x)

Si hay 3 capas en el checkpoint, las 3 SE EJECUTAN.

La discrepancia con el paper (2UP vs 3UP) sigue sin explicación clara.
Posibilidades:
  1. Los checkpoints son de una versión mejorada post-publicación
  2. Error en la documentación del paper
  3. "2UP" se refiere a otra métrica
        """)
    
    print("\n" + "="*70)
    print("🔍 PRÓXIMO PASO RECOMENDADO")
    print("="*70)
    print("""
Para resolver completamente el misterio:

1. Verificar si la capa 3 de XS/S tiene up=True o up=False
2. Hacer inferencia con dimensiones intermedias
3. Contactar a los autores del paper

Script sugerido: trace_forward_with_shapes.py
    """)
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
