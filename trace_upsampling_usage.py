#!/usr/bin/env python3
"""
Script para verificar si las 3 capas de upsampling están siendo USADAS
durante el forward pass, o si algunas están presentes pero no se ejecutan.

Traza el flujo de datos a través de cada capa del head.
"""

import torch
import torch.nn as nn
import sys
import os

# Agregar paths necesarios
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'main'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))

from config import cfg
from base import Tester


class TracingWrapper(nn.Module):
    """Wrapper para trazar cuando una capa es ejecutada"""
    
    def __init__(self, module, name):
        super().__init__()
        self.module = module
        self.name = name
        self.execution_count = 0
        self.input_shapes = []
        self.output_shapes = []
    
    def forward(self, x):
        self.execution_count += 1
        self.input_shapes.append(tuple(x.shape))
        output = self.module(x)
        self.output_shapes.append(tuple(output.shape))
        return output


def inject_tracers(model, model_name):
    """Inyecta trazadores en las capas de deconvolución"""
    
    print(f"\n{'='*70}")
    print(f"🔬 Inyectando trazadores en modelo: {model_name}")
    print(f"{'='*70}\n")
    
    tracers = {}
    
    # Buscar las capas de deconvolución
    if hasattr(model.module, 'head'):
        head = model.module.head
        
        # Verificar estructura del head
        if hasattr(head, 'deconv_layers'):
            deconv_layers = head.deconv_layers
            
            print(f"📊 Estructura del head:")
            print(f"   Tipo: {type(deconv_layers)}")
            
            if isinstance(deconv_layers, nn.ModuleList):
                print(f"   Número de capas: {len(deconv_layers)}")
                
                # Inyectar trazadores en cada capa
                for i, layer in enumerate(deconv_layers, 1):
                    layer_name = f"deconv_layer_{i}"
                    tracer = TracingWrapper(layer, layer_name)
                    
                    # Reemplazar la capa con el wrapper
                    deconv_layers[i-1] = tracer
                    tracers[layer_name] = tracer
                    
                    print(f"   ✅ Trazador inyectado en capa {i}")
            
            elif isinstance(deconv_layers, nn.Sequential):
                print(f"   Número de módulos: {len(deconv_layers)}")
                
                # Recorrer el Sequential
                layer_count = 0
                for idx, (name, module) in enumerate(deconv_layers.named_children()):
                    # Identificar si es una capa de deconv completa
                    if isinstance(module, nn.Sequential) or 'deconv' in str(type(module)).lower():
                        layer_count += 1
                        layer_name = f"deconv_layer_{layer_count}"
                        tracer = TracingWrapper(module, layer_name)
                        
                        # Reemplazar
                        deconv_layers[idx] = tracer
                        tracers[layer_name] = tracer
                        
                        print(f"   ✅ Trazador inyectado en capa {layer_count} (índice {idx})")
    
    return tracers


def analyze_model_usage(checkpoint_path, model_name, cfg_variant):
    """Analiza si todas las capas son usadas durante inferencia"""
    
    print(f"\n{'='*70}")
    print(f"🔍 Analizando uso de capas: {model_name}")
    print(f"{'='*70}")
    
    # Verificar que existe el checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"❌ No se encuentra: {checkpoint_path}")
        return None
    
    # Configurar el modelo
    cfg.set_args('0', test_epoch=str(cfg_variant))
    
    # Cargar modelo
    print(f"\n📥 Cargando modelo...")
    tester = Tester(cfg_variant)
    tester._make_model()
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    tester.model.load_state_dict(checkpoint['network'], strict=False)
    tester.model.eval()
    
    print(f"✅ Modelo cargado exitosamente")
    
    # Inyectar trazadores
    tracers = inject_tracers(tester.model, model_name)
    
    if not tracers:
        print("\n❌ No se pudieron inyectar trazadores")
        return None
    
    # Crear entrada de prueba (batch=1, channels=384, height=64, width=64)
    # Esto simula la salida del backbone
    print(f"\n🧪 Ejecutando forward pass de prueba...")
    
    # Determinar canales de entrada según el modelo
    input_channels_map = {
        'XS': 320,  # Salida del backbone Atto
        'S': 384,   # Salida del backbone Femto-L
        'M': 384,   # Salida del backbone Femto-L
        'L': 384,   # Salida del backbone Femto-L
    }
    
    input_channels = input_channels_map.get(model_name, 384)
    
    with torch.no_grad():
        # Simular salida del backbone
        dummy_input = torch.randn(1, input_channels, 64, 64)
        
        # Forward pass completo
        try:
            # Ejecutar solo el head
            if hasattr(tester.model.module, 'head'):
                output = tester.model.module.head(dummy_input)
                print(f"✅ Forward pass completado")
                print(f"   Input shape:  {dummy_input.shape}")
                print(f"   Output shape: {output.shape}")
            else:
                print("❌ No se encontró el módulo head")
                return None
        except Exception as e:
            print(f"❌ Error durante forward pass: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Analizar resultados
    print(f"\n{'='*70}")
    print(f"📊 RESULTADOS DEL ANÁLISIS")
    print(f"{'='*70}\n")
    
    results = {}
    
    for layer_name, tracer in tracers.items():
        results[layer_name] = {
            'executed': tracer.execution_count > 0,
            'execution_count': tracer.execution_count,
            'input_shapes': tracer.input_shapes,
            'output_shapes': tracer.output_shapes
        }
        
        status = "✅ EJECUTADA" if tracer.execution_count > 0 else "❌ NO EJECUTADA"
        print(f"  🔷 {layer_name}:")
        print(f"     Estado: {status}")
        print(f"     Ejecuciones: {tracer.execution_count}")
        
        if tracer.input_shapes:
            print(f"     Input shape:  {tracer.input_shapes[0]}")
            print(f"     Output shape: {tracer.output_shapes[0]}")
        print()
    
    return results


def main():
    """Función principal"""
    
    models = {
        'XS': {
            'path': 'demo/ConvNeXtPose_XS.tar',
            'variant': 'XS',
            'paper_layers': 2,
        },
        'S': {
            'path': 'demo/ConvNeXtPose_S.tar',
            'variant': 'S',
            'paper_layers': 2,
        },
        'M': {
            'path': 'demo/ConvNeXtPose_M (1).tar',
            'variant': 'M',
            'paper_layers': 3,
        },
        'L': {
            'path': 'demo/ConvNeXtPose_L (1).tar',
            'variant': 'L',
            'paper_layers': 3,
        },
    }
    
    print("\n" + "="*70)
    print("🔬 ANÁLISIS DE USO DE CAPAS DE UPSAMPLING")
    print("Verificando si las 3 capas se ejecutan durante inferencia")
    print("="*70)
    
    all_results = {}
    
    for model_name, model_info in models.items():
        results = analyze_model_usage(
            model_info['path'],
            model_name,
            model_info['variant']
        )
        
        if results:
            all_results[model_name] = results
    
    # Resumen final
    print("\n" + "="*70)
    print("📋 RESUMEN FINAL")
    print("="*70)
    print(f"\n{'Modelo':<10} {'Paper':<15} {'Capas en Ckpt':<15} {'Capas Usadas':<15} {'Conclusión':<20}")
    print("-" * 70)
    
    for model_name, results in all_results.items():
        paper_layers = models[model_name]['paper_layers']
        checkpoint_layers = len(results)
        used_layers = sum(1 for r in results.values() if r['executed'])
        
        if used_layers == checkpoint_layers:
            conclusion = "✅ Todas usadas"
        elif used_layers == paper_layers:
            conclusion = f"⚠️  Solo {used_layers}/{checkpoint_layers}"
        else:
            conclusion = f"❓ {used_layers}/{checkpoint_layers} usadas"
        
        print(f"{model_name:<10} {paper_layers}UP{'':<11} {checkpoint_layers} capas{'':<7} "
              f"{used_layers} capas{'':<7} {conclusion:<20}")
    
    print("\n" + "="*70)
    print("🎯 CONCLUSIONES")
    print("="*70)
    
    # Verificar modelos XS y S específicamente
    xs_s_use_all = True
    for model in ['XS', 'S']:
        if model in all_results:
            results = all_results[model]
            used = sum(1 for r in results.values() if r['executed'])
            total = len(results)
            
            if used < total:
                xs_s_use_all = False
                print(f"\n⚠️  Modelo {model}:")
                print(f"   - Tiene {total} capas en el checkpoint")
                print(f"   - Solo usa {used} capas durante inferencia")
                print(f"   - Esto explicaría la notación '{models[model]['paper_layers']}UP' del paper")
    
    if xs_s_use_all:
        print("\n✅ Los modelos XS y S usan TODAS sus 3 capas durante inferencia")
        print("   La discrepancia con el paper (2UP) sigue sin explicación clara.")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
