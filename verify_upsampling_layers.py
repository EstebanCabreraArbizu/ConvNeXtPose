#!/usr/bin/env python3
"""
Script para verificar el número REAL de capas de upsampling en cada modelo.
Compara con lo reportado en el paper.
"""

import torch
import os

def analyze_upsampling_layers(checkpoint_path, model_name):
    """Analiza las capas de upsampling en un checkpoint"""
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ No se encuentra: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['network']
    
    print(f"\n{'='*60}")
    print(f"🔍 Analizando: {model_name}")
    print(f"{'='*60}")
    
    # Buscar capas de deconvolution
    deconv_layers = {}
    
    for key in sorted(model_state.keys()):
        # Buscar capas del head
        if 'head.deconv_layers' in key:
            # Extraer número de capa
            if 'deconv_layers_' in key:
                # Formato: module.head.deconv_layers_1.dwconv.weight
                parts = key.split('.')
                for i, part in enumerate(parts):
                    if 'deconv_layers_' in part:
                        layer_num = int(part.split('_')[-1])
                        break
            else:
                continue
            
            if layer_num not in deconv_layers:
                deconv_layers[layer_num] = []
            
            deconv_layers[layer_num].append(key)
    
    # Reportar hallazgos
    print(f"\n📊 Capas encontradas: {len(deconv_layers)}")
    
    for layer_num in sorted(deconv_layers.keys()):
        print(f"\n  🔷 Capa {layer_num}:")
        
        # Buscar pesos principales
        for key in deconv_layers[layer_num]:
            if 'weight' in key:
                weight = model_state[key]
                print(f"    - {key.split('module.head.')[1]}")
                print(f"      Shape: {list(weight.shape)}")
                
                # Verificar si todos los pesos son cero (capa deshabilitada)
                if torch.all(weight == 0):
                    print(f"      ⚠️  ADVERTENCIA: Todos los pesos son CERO (capa deshabilitada)")
                else:
                    # Calcular estadísticas
                    non_zero = torch.count_nonzero(weight)
                    total = weight.numel()
                    print(f"      ✅ Pesos activos: {non_zero}/{total} ({100*non_zero/total:.2f}%)")
    
    return len(deconv_layers)


def main():
    """Función principal"""
    
    models = {
        'XS': {
            'path': 'demo/ConvNeXtPose_XS.tar',
            'paper_upsampling': 2,  # 2UP según el paper
        },
        'S': {
            'path': 'demo/ConvNeXtPose_S.tar',
            'paper_upsampling': 2,  # 2UP según el paper
        },
        'M': {
            'path': 'demo/ConvNeXtPose_M (1).tar',
            'paper_upsampling': 3,  # 3UP según el paper
        },
        'L': {
            'path': 'demo/ConvNeXtPose_L (1).tar',
            'paper_upsampling': 3,  # 3UP según el paper
        },
    }
    
    print("\n" + "="*60)
    print("🔬 VERIFICACIÓN DE CAPAS DE UPSAMPLING")
    print("Comparando checkpoints vs paper IEEE Access 2023")
    print("="*60)
    
    results = {}
    
    for model_name, model_info in models.items():
        actual_layers = analyze_upsampling_layers(model_info['path'], model_name)
        
        if actual_layers is not None:
            results[model_name] = {
                'actual': actual_layers,
                'paper': model_info['paper_upsampling']
            }
    
    # Resumen final
    print("\n" + "="*60)
    print("📋 RESUMEN COMPARATIVO")
    print("="*60)
    print(f"\n{'Modelo':<10} {'Paper':<10} {'Checkpoint':<12} {'Estado':<20}")
    print("-" * 60)
    
    for model_name, data in results.items():
        paper = data['paper']
        actual = data['actual']
        
        if paper == actual:
            status = "✅ COINCIDE"
        else:
            status = f"⚠️  DIFIERE ({actual - paper:+d})"
        
        print(f"{model_name:<10} {paper}UP{'':<6} {actual} capas{'':<5} {status:<20}")
    
    print("\n" + "="*60)
    print("🎯 CONCLUSIÓN")
    print("="*60)
    
    # Verificar si hay discrepancias
    discrepancies = [m for m, d in results.items() if d['actual'] != d['paper']]
    
    if discrepancies:
        print("\n⚠️  Se encontraron discrepancias en los modelos:")
        for model in discrepancies:
            print(f"   - {model}: {results[model]['paper']}UP (paper) vs {results[model]['actual']} capas (checkpoint)")
        
        print("\n💡 Posibles explicaciones:")
        print("   1. Las capas extra están presentes pero deshabilitadas (pesos = 0)")
        print("   2. La arquitectura cambió después de la publicación")
        print("   3. Los checkpoints son de una versión diferente")
        print("   4. El paper usa una notación diferente (capas activas vs totales)")
    else:
        print("\n✅ Todos los modelos coinciden con el paper")


if __name__ == '__main__':
    main()
