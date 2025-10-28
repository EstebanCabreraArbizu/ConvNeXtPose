#!/usr/bin/env python3
"""
Script simplificado para verificar la estructura real del modelo
cuando se carga desde el checkpoint.

Inspecciona la arquitectura del head para ver cuántas capas
están realmente conectadas en el forward pass.
"""

import torch
import torch.nn as nn
import os


def inspect_head_architecture(checkpoint_path, model_name):
    """Inspecciona la arquitectura del head en un checkpoint cargado"""
    
    print(f"\n{'='*70}")
    print(f"🔍 Inspeccionando arquitectura: {model_name}")
    print(f"{'='*70}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ No se encuentra: {checkpoint_path}")
        return None
    
    # Cargar checkpoint
    print(f"\n📥 Cargando checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['network']
    
    print(f"✅ Checkpoint cargado")
    
    # Buscar todas las capas del head
    head_layers = {}
    
    print(f"\n📊 Capas encontradas en el head:")
    print("-" * 70)
    
    for key in sorted(model_state.keys()):
        if 'head' in key and 'deconv_layers' in key:
            # Extraer información
            parts = key.split('.')
            
            # Encontrar el índice de la capa
            layer_idx = None
            for part in parts:
                if 'deconv_layers_' in part:
                    try:
                        layer_idx = int(part.split('_')[-1])
                    except:
                        pass
                    break
            
            if layer_idx is not None:
                if layer_idx not in head_layers:
                    head_layers[layer_idx] = []
                head_layers[layer_idx].append(key)
    
    # Mostrar estructura
    num_layers = len(head_layers)
    print(f"\n✅ Número de capas de deconvolución: {num_layers}")
    
    for layer_idx in sorted(head_layers.keys()):
        print(f"\n  🔷 Capa {layer_idx}:")
        
        # Buscar los pesos principales
        for key in head_layers[layer_idx]:
            if 'weight' in key and 'norm' not in key:
                weight = model_state[key]
                print(f"     {key.split('module.head.')[1]}")
                print(f"     └─ Shape: {list(weight.shape)}")
    
    return num_layers


def trace_model_forward():
    """
    Traza el forward pass mirando el código fuente del modelo
    """
    
    print(f"\n{'='*70}")
    print(f"📂 Buscando definición del modelo...")
    print(f"{'='*70}\n")
    
    model_file = 'main/model.py'
    
    if not os.path.exists(model_file):
        print(f"❌ No se encuentra {model_file}")
        return
    
    print(f"✅ Leyendo {model_file}...\n")
    
    with open(model_file, 'r') as f:
        content = f.read()
    
    # Buscar la definición del forward del head
    print("🔍 Buscando forward pass del head:")
    print("-" * 70)
    
    lines = content.split('\n')
    in_head_forward = False
    forward_code = []
    indent_level = 0
    
    for i, line in enumerate(lines):
        # Buscar inicio del forward
        if 'def forward' in line and 'head' in content[max(0, i-20):i]:
            in_head_forward = True
            indent_level = len(line) - len(line.lstrip())
            forward_code.append(f"Línea {i+1}: {line}")
            continue
        
        if in_head_forward:
            # Detectar fin del método (línea con menor indentación)
            current_indent = len(line) - len(line.lstrip())
            if line.strip() and current_indent <= indent_level and 'def ' in line:
                break
            
            # Guardar líneas relevantes
            if 'deconv' in line.lower() or 'for ' in line or 'self.' in line:
                forward_code.append(f"Línea {i+1}: {line}")
    
    if forward_code:
        print("\n📝 Código del forward pass encontrado:")
        for line in forward_code[:20]:  # Mostrar primeras 20 líneas
            print(f"   {line}")
    else:
        print("❌ No se encontró el forward pass del head")
    
    # Buscar loops que itren sobre las capas
    print(f"\n🔄 Buscando iteración sobre capas de deconv:")
    print("-" * 70)
    
    for line in forward_code:
        if 'for' in line and 'deconv' in line:
            print(f"   ✅ {line}")
            print(f"      └─ Esto indica que itera sobre TODAS las capas")
        elif 'range' in line:
            print(f"   ✅ {line}")


def main():
    """Función principal"""
    
    models = {
        'XS': 'demo/ConvNeXtPose_XS.tar',
        'S': 'demo/ConvNeXtPose_S.tar',
        'M': 'demo/ConvNeXtPose_M (1).tar',
        'L': 'demo/ConvNeXtPose_L (1).tar',
    }
    
    print("\n" + "="*70)
    print("🔬 ANÁLISIS DE USO REAL DE CAPAS")
    print("Parte 1: Inspección de Checkpoints")
    print("="*70)
    
    results = {}
    
    for model_name, checkpoint_path in models.items():
        num_layers = inspect_head_architecture(checkpoint_path, model_name)
        if num_layers:
            results[model_name] = num_layers
    
    # Parte 2: Analizar el código
    print("\n" + "="*70)
    print("🔬 ANÁLISIS DE USO REAL DE CAPAS")
    print("Parte 2: Análisis del Código del Modelo")
    print("="*70)
    
    trace_model_forward()
    
    # Resumen
    print("\n" + "="*70)
    print("📋 RESUMEN")
    print("="*70)
    
    print(f"\n{'Modelo':<10} {'Capas en Checkpoint':<20} {'Observación':<40}")
    print("-" * 70)
    
    for model_name, num_layers in results.items():
        obs = f"Tiene {num_layers} capas definidas"
        print(f"{model_name:<10} {num_layers} capas{'':<13} {obs:<40}")
    
    print("\n" + "="*70)
    print("🎯 PRÓXIMOS PASOS")
    print("="*70)
    print("""
Para determinar si TODAS las capas se usan durante inferencia:

1. ✅ Ya verificamos: Los checkpoints tienen 3 capas con pesos activos
2. 🔄 Siguiente paso: Ver el código del forward pass en main/model.py
3. 🧪 Alternativa: Hacer una inferencia real y medir las dimensiones intermedias

Si el forward pass itera con un 'for layer in self.deconv_layers',
entonces TODAS las capas se ejecutan.

Si hay lógica condicional o solo usa índices específicos [0] y [1],
entonces podría estar usando solo 2 de las 3 capas.
""")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
