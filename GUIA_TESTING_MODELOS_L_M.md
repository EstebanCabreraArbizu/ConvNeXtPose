# Guía Completa: Testing de Modelos ConvNeXtPose L y M en Human3.6M

## 📋 Resumen Ejecutivo

Esta guía proporciona los pasos detallados para testear los modelos **ConvNeXtPose L y M** en el dataset **Human3.6M** utilizando el **Protocolo 2 (MPJPE)** para evaluar en los sujetos **S9 y S11**.

### Estado Actual del Repositorio
- **Configuración por defecto**: Modelos XS y S
- **Objetivo**: Adaptar para modelos L y M
- **Protocolo**: 2 (MPJPE - Mean Per Joint Position Error)
- **Sujetos de evaluación**: S9 y S11

---

## 🎯 PASO 1: Preparación del Entorno

### 1.1 Verificar Dependencias

```bash
# Verificar Python version (debe ser 3.8+)
python --version

# Verificar CUDA
nvcc --version

# Verificar PyTorch y CUDA disponibilidad
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### 1.2 Instalar Dependencias Faltantes

```bash
cd /home/user/convnextpose_esteban/ConvNeXtPose

# Instalar dependencias si es necesario
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm
pip install pycocotools
pip install opencv-python
pip install tqdm
pip install numpy
pip install matplotlib
```

---

## 🗂️ PASO 2: Verificar Estructura de Datos

### 2.1 Verificar Dataset Human3.6M

```bash
# Verificar que los datos de Human3.6M existan
ls -la data/Human36M/

# Debe contener:
# - images/
# - annotations/
# - bbox_root/bbox_root_human36m_output.json
```

### 2.2 Verificar Bbox Root para Protocolo 2

El Protocolo 2 evalúa en S9 y S11. Verificar que existe el archivo correcto:

```bash
# Verificar bbox_root para Protocol 2
ls -la "data/Human36M/bbox_root/Subject 9,11 (trained on subject 1,5,6,7,8)/"

# Debe mostrar: bbox_root_human36m_output.json
```

**Importante**: Si usas el bounding box GT, verifica que la configuración `use_gt_info = True` en `config.py`.

---

## 🔧 PASO 3: Descargar Modelos Pre-entrenados

### 3.1 Descargar Modelos L y M

Según el paper, los modelos pre-entrenados están disponibles en Google Drive:
- **Link oficial**: https://drive.google.com/drive/folders/12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI?usp=sharing

```bash
# Crear carpeta para modelos
mkdir -p output/model_dump

# Descargar manualmente desde Google Drive o usar gdown:
pip install gdown

# Descargar modelo M (ajustar ID según el drive)
# gdown <ID_DEL_MODELO_M> -O output/model_dump/model_M_h36m.pth.tar

# Descargar modelo L (ajustar ID según el drive)
# gdown <ID_DEL_MODELO_L> -O output/model_dump/model_L_h36m.pth.tar
```

**Nota**: Los nombres de los archivos deben seguir el formato: `snapshot_<epoch>.pth.tar`

---

## ⚙️ PASO 4: Modificar Configuraciones (ACTUALIZADO - Refactorización Completa)

> **IMPORTANTE**: El código ha sido completamente refactorizado para soportar dinámicamente las variantes XS/S/M/L. Ya NO es necesario editar manualmente los archivos de configuración.

### 4.1 NUEVO: Uso de Argumentos CLI (Método Recomendado)

```bash
# Testing ConvNeXtPose-M
python test.py --gpu 0-3 --epochs 70 --variant M

# Testing ConvNeXtPose-L
python test.py --gpu 0-3 --epochs 70 --variant L
```

El argumento `--variant` automáticamente:
- ✅ Carga la configuración correcta del backbone
- ✅ Configura la arquitectura de HeadNet (2-UP para XS/S, 3-UP para M/L)
- ✅ Aplica remapping de checkpoints si es necesario

### 4.2 Método Alternativo: Edición de config.py

Si prefieres configurar por defecto en el código:

```python
# main/config.py

class Config:
    # ... otras configuraciones ...
    
    ## model variant configuration
    variant = 'M'  # Cambiar a 'M' o 'L' según el modelo a testear
    
    # NO es necesario modificar backbone_cfg manualmente
    # Se carga automáticamente con load_variant_config()
```

Luego ejecutar:
```bash
python test.py --gpu 0-3 --epochs 70  # Usará variant por defecto
```

## ⚙️ PASO 4 (LEGACY): Modificar Configuraciones Manualmente [OBSOLETO]

> **⚠️ ADVERTENCIA**: Esta sección es obsoleta después de la refactorización. Se mantiene por referencia histórica.

### 4.1 LEGACY: Modificar main/config.py (NO RECOMENDADO)

> **⚠️ Este método está OBSOLETO**. La modificación manual de `backbone_cfg` en `config.py` **NO es suficiente** para los modelos M/L porque requieren arquitectura de HeadNet diferente (3-UP en lugar de 2-UP).

**Antes de la refactorización**, el código tenía:
```python
backbone_cfg = ([3, 3, 9, 3], [48, 96, 192, 384])  # XS
```

**Problema identificado**: Cambiar solo `backbone_cfg` **NO funciona** porque:
1. ❌ HeadNet estaba hardcoded con 2 capas de upsampling
2. ❌ Modelos M/L requieren 3 capas de upsampling
3. ❌ Faltaba compatibilidad con formatos de checkpoint del paper

**Solución implementada**: Ver sección 4.1 (NUEVO) arriba.

---

### 4.3 Entender las Diferencias Arquitectónicas

#### Tabla Comparativa de Variantes

| Variante | Depths | Dims | Backbone Out | HeadNet | Params | GFLOPs | MPJPE |
|----------|--------|------|--------------|---------|--------|--------|-------|
| **XS**   | [3,3,9,3] | [48,96,192,384] | 384 ch | 2-UP | 22M | 4.5 | ~52mm |
| **S**    | [3,3,27,3] | [96,192,384,768] | 768 ch | 2-UP | 50M | 8.7 | ~48mm |
| **M**    | [3,3,27,3] | [128,256,512,1024] | 1024 ch | **3-UP** | 88.6M | 15.4 | **~44.6mm** |
| **L**    | [3,3,27,3] | [192,384,768,1536] | 1536 ch | **3-UP** | 197.8M | 34.4 | **~42.3mm** |

#### Arquitecturas de HeadNet

**XS/S (2-UP):**
```
Backbone (384/768 ch) 
  → DeConv+Upsample 2x (256 ch)
  → DeConv+Upsample 2x (256 ch)
  → Conv 1x1 (Final)
  → Heatmaps
```

**M/L (3-UP):**
```
Backbone (1024/1536 ch)
  → DeConv+Upsample 2x (256 ch)
  → DeConv+Upsample 2x (256 ch)
  → DeConv+Upsample 2x (256 ch)  ← Capa adicional
  → Conv 1x1 (Final)
  → Heatmaps
```

**Diferencia clave:** M/L realizan **3 upsamples** en lugar de 2, lo que permite:
- Mejor resolución espacial de los heatmaps
- Predicciones más precisas de posiciones articulares
- Ganancia de ~5-6mm en MPJPE vs modelos S

### 4.2 Modificar Protocol en Human36M.py

Verificar que el protocolo esté configurado correctamente:

```bash
# El archivo data/Human36M/Human36M.py debe tener:
self.protocol = 2  # Para MPJPE (no PA-MPJPE)
```

---

## 📝 PASO 5: Crear Archivos de Configuración

### 5.1 Crear config_variants.py

Vamos a crear un archivo que defina las configuraciones para cada variante:

```python
# main/config_variants.py

MODEL_CONFIGS = {
    'XS': {
        'depths': [3, 3, 9, 3],
        'dims': [48, 96, 192, 384],
        'description': 'Extra Small - Original default'
    },
    'S': {
        'depths': [3, 3, 27, 3],
        'dims': [96, 192, 384, 768],
        'description': 'Small'
    },
    'M': {
        'depths': [3, 3, 27, 3],
        'dims': [128, 256, 512, 1024],
        'description': 'Medium - Femto-L backbone'
    },
    'L': {
        'depths': [3, 3, 27, 3],
        'dims': [192, 384, 768, 1536],
        'description': 'Large - Femto-L backbone'
    }
}

def get_model_config(variant='M'):
    """
    Retorna la configuración del modelo para la variante especificada
    
    Args:
        variant (str): 'XS', 'S', 'M', o 'L'
    
    Returns:
        tuple: (depths, dims)
    """
    if variant not in MODEL_CONFIGS:
        raise ValueError(f"Variante no válida: {variant}. Usa: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[variant]
    return (config['depths'], config['dims'])
```

### 5.2 Modificar config.py

Modificaremos `main/config.py` para agregar soporte de variantes:

```python
# Agregar al final de main/config.py después de la clase Config

import argparse

class ConfigWithVariant(Config):
    """
    Extensión de Config que soporta diferentes variantes de modelo
    """
    def __init__(self, model_variant='S'):
        super().__init__()
        self.model_variant = model_variant
        self.set_model_variant(model_variant)
    
    def set_model_variant(self, variant):
        """Configura el backbone según la variante del modelo"""
        from config_variants import get_model_config, MODEL_CONFIGS
        
        if variant not in MODEL_CONFIGS:
            print(f"⚠️ Variante '{variant}' no válida. Usando 'S' por defecto.")
            variant = 'S'
        
        depths, dims = get_model_config(variant)
        self.backbone_cfg = (depths, dims)
        self.model_variant = variant
        
        print(f"✓ Configuración del modelo: {variant}")
        print(f"  - Depths: {depths}")
        print(f"  - Dims: {dims}")
        print(f"  - Descripción: {MODEL_CONFIGS[variant]['description']}")

def get_config(model_variant='S'):
    """
    Función helper para obtener configuración con variante específica
    
    Args:
        model_variant (str): 'XS', 'S', 'M', 'L'
    
    Returns:
        ConfigWithVariant: Objeto de configuración
    """
    return ConfigWithVariant(model_variant)
```

---

## 🔄 PASO 6: Modificar Script de Testing

### 6.1 Modificar test.py para Soportar Variantes

Necesitamos modificar `main/test.py` para aceptar el argumento de variante:

```python
# Modificar la función parse_args() en main/test.py

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--epochs', type=str, dest='model')
    parser.add_argument('--variant', type=str, default='S', 
                       choices=['XS', 'S', 'M', 'L'],
                       help='Model variant: XS, S, M, or L')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, "Please set proper gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    if '-' in args.model:
        model_epoch = args.model.split('-')
        model_epoch[0] = int(model_epoch[0])
        model_epoch[1] = int(model_epoch[1]) + 1
        args.model_epoch = model_epoch

    return args
```

### 6.2 Modificar función main() en test.py

```python
def main():
    args = parse_args()
    
    # Usar configuración con variante
    from config import get_config
    global cfg
    cfg = get_config(args.variant)
    
    cfg.set_args(args.gpu_ids)
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    tester = Tester()
    tester._make_batch_generator()

    # ... resto del código
```

---

## 🧪 PASO 7: Crear Script de Testing Unificado

### 7.1 Crear test_variants.py

Vamos a crear un script maestro para facilitar el testing:

```python
# main/test_variants.py

"""
Script para testear diferentes variantes de ConvNeXtPose en Human3.6M
Uso:
    python test_variants.py --variant M --gpu 0 --epoch 70
    python test_variants.py --variant L --gpu 0 --epoch 70
"""

import argparse
import os
import sys
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np

# Importar configuración
from config_variants import MODEL_CONFIGS, get_model_config
from config import Config
from base import Tester
from utils.pose_utils import flip

def parse_args():
    parser = argparse.ArgumentParser(description='Test ConvNeXtPose variants on Human3.6M')
    parser.add_argument('--variant', type=str, required=True, 
                       choices=['XS', 'S', 'M', 'L'],
                       help='Model variant to test')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU id to use')
    parser.add_argument('--epoch', type=int, required=True,
                       help='Epoch number to test')
    parser.add_argument('--protocol', type=int, default=2,
                       choices=[1, 2],
                       help='Evaluation protocol (1: PA-MPJPE, 2: MPJPE)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Test batch size')
    parser.add_argument('--flip_test', action='store_true',
                       help='Use flip test augmentation')
    parser.add_argument('--use_gt_bbox', action='store_true',
                       help='Use ground truth bounding box')
    
    return parser.parse_args()

class VariantTester:
    def __init__(self, variant, epoch, gpu_id, protocol=2, 
                 batch_size=16, flip_test=True, use_gt_bbox=True):
        self.variant = variant
        self.epoch = epoch
        self.protocol = protocol
        
        # Configurar modelo
        self.setup_config(variant, gpu_id, protocol, batch_size, flip_test, use_gt_bbox)
        
        # Configurar CUDA
        self.setup_cuda()
        
    def setup_config(self, variant, gpu_id, protocol, batch_size, flip_test, use_gt_bbox):
        """Configura la configuración del modelo"""
        global cfg
        cfg = Config()
        
        # Configurar variante
        depths, dims = get_model_config(variant)
        cfg.backbone_cfg = (depths, dims)
        cfg.model_variant = variant
        
        # Configurar testing
        cfg.test_batch_size = batch_size
        cfg.flip_test = flip_test
        cfg.use_gt_info = use_gt_bbox
        
        # Configurar GPU
        cfg.set_args(gpu_id)
        
        print("\n" + "="*70)
        print(f"🔧 CONFIGURACIÓN DE TESTING")
        print("="*70)
        print(f"Modelo: ConvNeXtPose-{variant}")
        print(f"Época: {epoch}")
        print(f"Protocolo: {protocol} ({'PA-MPJPE' if protocol == 1 else 'MPJPE'})")
        print(f"Batch size: {batch_size}")
        print(f"Flip test: {flip_test}")
        print(f"GT bbox: {use_gt_bbox}")
        print(f"GPU: {gpu_id}")
        print(f"Depths: {depths}")
        print(f"Dims: {dims}")
        print("="*70 + "\n")
        
        return cfg
    
    def setup_cuda(self):
        """Configura optimizaciones CUDA"""
        cudnn.fastest = True
        cudnn.benchmark = True
        cudnn.deterministic = False
        cudnn.enabled = True
        
    def run_test(self):
        """Ejecuta el testing"""
        print(f"📊 Iniciando evaluación del modelo {self.variant}...\n")
        
        # Crear tester
        tester = Tester()
        tester._make_batch_generator()
        tester._make_model(self.epoch)
        
        # Recolectar predicciones
        preds = []
        
        with torch.no_grad():
            for itr, input_img in enumerate(tqdm(tester.batch_generator, 
                                                  desc=f"Testing epoch {self.epoch}")):
                # Forward pass
                coord_out = tester.model(input_img)
                
                # Flip test si está habilitado
                if cfg.flip_test:
                    flipped_input_img = flip(input_img, dims=3)
                    flipped_coord_out = tester.model(flipped_input_img)
                    flipped_coord_out[:, :, 0] = cfg.output_shape[1] - flipped_coord_out[:, :, 0] - 1
                    
                    for pair in tester.flip_pairs:
                        flipped_coord_out[:, pair[0], :], flipped_coord_out[:, pair[1], :] = \
                            flipped_coord_out[:, pair[1], :].clone(), flipped_coord_out[:, pair[0], :].clone()
                    
                    coord_out = (coord_out + flipped_coord_out) / 2.
                
                coord_out = coord_out.cpu().numpy()
                preds.append(coord_out)
        
        # Concatenar predicciones
        preds = np.concatenate(preds, axis=0)
        
        print(f"\n✓ Predicciones completadas: {preds.shape}")
        
        # Evaluar
        print(f"\n📈 Evaluando resultados...\n")
        eval_result = tester._evaluate(preds, cfg.result_dir)
        
        return preds, eval_result

def main():
    args = parse_args()
    
    # Crear tester
    tester = VariantTester(
        variant=args.variant,
        epoch=args.epoch,
        gpu_id=args.gpu,
        protocol=args.protocol,
        batch_size=args.batch_size,
        flip_test=args.flip_test,
        use_gt_bbox=args.use_gt_bbox
    )
    
    # Ejecutar testing
    preds, results = tester.run_test()
    
    print("\n" + "="*70)
    print("✅ TESTING COMPLETADO")
    print("="*70)
    print(f"Variante: {args.variant}")
    print(f"Época: {args.epoch}")
    print(f"Predicciones: {preds.shape}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
```

---

## 🚀 PASO 8: Ejecutar Testing

### 8.1 Comandos para Testing

#### Testear Modelo M (Medium):

```bash
cd /home/user/convnextpose_esteban/ConvNeXtPose/main

# Con el script modificado test.py
python test.py --gpu 0 --epochs 70-71 --variant M

# O con el script unificado
python test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2
```

#### Testear Modelo L (Large):

```bash
# Con el script modificado test.py
python test.py --gpu 0 --epochs 70-71 --variant L

# O con el script unificado
python test_variants.py --variant L --gpu 0 --epoch 70 --protocol 2
```

### 8.2 Opciones Adicionales

```bash
# Testear sin flip augmentation (más rápido)
python test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2

# Testear con flip augmentation (más preciso, más lento)
python test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test

# Testear con bbox predicho (sin GT)
python test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test

# Testear con bbox GT
python test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox
```

---

## 📊 PASO 9: Comparar Resultados

### 9.1 Crear Script de Comparación

```python
# main/compare_variants.py

"""
Script para comparar resultados entre variantes
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

def load_results(result_dir, variant, epoch):
    """Carga resultados de una variante"""
    result_file = os.path.join(result_dir, f'result_{variant}_epoch{epoch}.json')
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            return json.load(f)
    return None

def compare_variants(variants=['S', 'M', 'L'], epoch=70):
    """Compara resultados entre variantes"""
    result_dir = '../output/result'
    
    results = {}
    for variant in variants:
        result = load_results(result_dir, variant, epoch)
        if result:
            results[variant] = result
    
    # Crear tabla comparativa
    print("\n" + "="*80)
    print("📊 COMPARACIÓN DE RESULTADOS - HUMAN3.6M PROTOCOL 2")
    print("="*80)
    print(f"{'Variante':<12} {'MPJPE (mm)':<15} {'Mejora vs S (%)':<20}")
    print("-"*80)
    
    baseline_mpjpe = results['S']['total_mpjpe'] if 'S' in results else None
    
    for variant in variants:
        if variant in results:
            mpjpe = results[variant]['total_mpjpe']
            improvement = ((baseline_mpjpe - mpjpe) / baseline_mpjpe * 100) if baseline_mpjpe else 0
            print(f"{variant:<12} {mpjpe:<15.2f} {improvement:<20.2f}")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    compare_variants(['S', 'M', 'L'], epoch=70)
```

---

## 🔍 PASO 10: Verificación de Resultados

### 10.1 Resultados Esperados (Protocol 2 - MPJPE)

Según el paper de ConvNeXtPose:

| Modelo | MPJPE (mm) | Mejora vs S | GFLOPs | Params (M) |
|--------|------------|-------------|---------|------------|
| **S**  | ~48-50     | Baseline    | ~8.7    | ~50        |
| **M**  | ~44-46     | ~8-10%      | ~15.4   | ~88.6      |
| **L**  | ~42-44     | ~12-15%     | ~34.4   | ~197.8     |

### 10.2 Verificar Outputs

```bash
# Verificar archivos de resultado
ls -la output/result/

# Debe contener:
# - bbox_root_pose_human36m_output.json (predicciones)
# - Logs con métricas por acción
```

### 10.3 Resultados por Acción

Los resultados deben mostrar MPJPE para cada acción:
- Directions
- Discussion
- Eating
- Greeting
- Phoning
- Posing
- Purchases
- Sitting
- SittingDown
- Smoking
- Photo
- Waiting
- Walking
- WalkDog
- WalkTogether

---

## 🐛 PASO 11: Troubleshooting

### 11.1 Error: Out of Memory (OOM)

```python
# Reducir batch size en config.py
cfg.test_batch_size = 8  # En lugar de 16

# O en el comando
python test_variants.py --variant L --gpu 0 --epoch 70 --batch_size 8
```

### 11.2 Error: Modelo no carga

```bash
# Verificar nombre del archivo
ls -la output/model_dump/

# Debe ser: snapshot_<epoch>.pth.tar
# Ejemplo: snapshot_70.pth.tar
```

Si el nombre es diferente, renombrar:

```bash
mv output/model_dump/model_M_final.pth.tar output/model_dump/snapshot_70.pth.tar
```

### 11.3 Error: Dataset no encontrado

```bash
# Verificar estructura de datos
tree -L 3 data/Human36M/

# Verificar que existan:
# - images/
# - annotations/ (con archivos JSON)
# - bbox_root/ (con bbox_root_human36m_output.json)
```

### 11.4 Error: Protocolo incorrecto

Editar `data/Human36M/Human36M.py`:

```python
# Línea ~30
self.protocol = 2  # DEBE SER 2 para evaluar en S9 y S11
```

### 11.5 Error: Dimensiones del modelo no coinciden

Esto indica que el checkpoint fue entrenado con una configuración diferente. Verificar:

```python
# Cargar checkpoint y verificar arquitectura
import torch
checkpoint = torch.load('output/model_dump/snapshot_70.pth.tar', map_location='cpu')

# Ver claves
print(checkpoint.keys())

# Ver forma de algunos pesos
for key in list(checkpoint['network'].keys())[:10]:
    print(f"{key}: {checkpoint['network'][key].shape}")
```

---

## 📈 PASO 12: Análisis Avanzado

### 12.1 Benchmarking de Rendimiento

```python
# main/benchmark_variants.py

import time
import torch
import numpy as np
from config_variants import get_model_config
from model import get_pose_net
from config import Config

def benchmark_variant(variant, num_runs=100):
    """Benchmark de velocidad e inferencia"""
    cfg = Config()
    depths, dims = get_model_config(variant)
    cfg.backbone_cfg = (depths, dims)
    
    # Crear modelo
    model = get_pose_net(cfg, is_train=False, joint_num=18)
    model.eval()
    model.cuda()
    
    # Input dummy
    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / num_runs
    fps = 1.0 / avg_time
    
    print(f"\n{'='*50}")
    print(f"Benchmark: ConvNeXtPose-{variant}")
    print(f"{'='*50}")
    print(f"Tiempo promedio: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"{'='*50}\n")
    
    return avg_time, fps

if __name__ == "__main__":
    for variant in ['S', 'M', 'L']:
        benchmark_variant(variant)
```

### 12.2 Análisis de Memoria

```bash
# Monitorear uso de GPU durante testing
watch -n 1 nvidia-smi

# O instalar gpustat
pip install gpustat
gpustat -i 1
```

---

## 📝 PASO 13: Documentar Resultados

### 13.1 Crear Reporte Final

```python
# main/generate_report.py

"""
Genera reporte final de evaluación
"""

import json
import os
from datetime import datetime

def generate_report(variant, epoch, results):
    """Genera reporte markdown"""
    
    report = f"""# Reporte de Evaluación: ConvNeXtPose-{variant}

## Información General
- **Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Modelo**: ConvNeXtPose-{variant}
- **Época**: {epoch}
- **Dataset**: Human3.6M
- **Protocolo**: 2 (MPJPE)
- **Sujetos**: S9, S11

## Configuración del Modelo
- **Depths**: {results.get('depths', 'N/A')}
- **Dims**: {results.get('dims', 'N/A')}
- **Parámetros**: {results.get('params', 'N/A')} M
- **GFLOPs**: {results.get('gflops', 'N/A')}

## Resultados Principales
- **MPJPE Total**: {results.get('total_mpjpe', 'N/A'):.2f} mm

## Resultados por Acción
{generate_action_table(results.get('action_results', {}))}

## Comparación con Paper
- **Esperado**: {results.get('expected_mpjpe', 'N/A')} mm
- **Obtenido**: {results.get('total_mpjpe', 'N/A'):.2f} mm
- **Diferencia**: {results.get('difference', 'N/A'):.2f} mm

## Notas
{results.get('notes', 'Sin notas adicionales')}
"""
    
    # Guardar reporte
    report_path = f'../output/result/report_{variant}_epoch{epoch}.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Reporte guardado en: {report_path}")

def generate_action_table(action_results):
    """Genera tabla de resultados por acción"""
    if not action_results:
        return "No disponible"
    
    table = "| Acción | MPJPE (mm) |\n|--------|------------|\n"
    for action, mpjpe in action_results.items():
        table += f"| {action} | {mpjpe:.2f} |\n"
    
    return table
```

---

## ✅ CHECKLIST FINAL

Antes de ejecutar el testing, verifica:

- [ ] Python 3.8+ instalado
- [ ] PyTorch con CUDA instalado y funcionando
- [ ] Dataset Human3.6M descargado y en ubicación correcta
- [ ] Bbox root para Protocol 2 (S9, S11) disponible
- [ ] Modelos pre-entrenados L y M descargados
- [ ] Archivos de configuración modificados (`config_variants.py`)
- [ ] Script de testing adaptado para variantes
- [ ] Protocol = 2 configurado en `Human36M.py`
- [ ] GPU con suficiente VRAM (≥8GB para L, ≥4GB para M)
- [ ] Estructura de output creada (`output/model_dump`, `output/result`, etc.)

---

## 🎓 RESUMEN DE COMANDOS CLAVE

```bash
# 1. Testing básico modelo M
python test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2

# 2. Testing básico modelo L
python test_variants.py --variant L --gpu 0 --epoch 70 --protocol 2

# 3. Testing con todas las opciones (recomendado)
python test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox --batch_size 16

# 4. Comparar resultados
python compare_variants.py

# 5. Benchmark de rendimiento
python benchmark_variants.py
```

---

## 📚 Referencias

1. **Paper original**: ConvNeXtPose: A Fast Accurate Method for 3D Human Pose Estimation (IEEE Access 2023)
2. **Repositorio oficial**: https://github.com/medialab-ku/ConvNeXtPose
3. **Human3.6M**: http://vision.imar.ro/human3.6m/
4. **Modelos pre-entrenados**: https://drive.google.com/drive/folders/12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI

---

## 📧 Soporte

Si encuentras problemas durante la implementación:
1. Verificar logs en `output/log/`
2. Verificar errores CUDA con `nvidia-smi`
3. Revisar troubleshooting en Paso 11
4. Comparar configuración con los valores esperados del paper

---

**Última actualización**: Octubre 2025
**Versión**: 1.0
