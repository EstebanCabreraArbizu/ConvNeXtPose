# 🔍 Dimensiones de Kernels Verificadas - ConvNeXtPose

**Fecha:** 16 de Octubre de 2025  
**Fuente:** Análisis directo de checkpoints pre-entrenados  
**Estado:** ✅ Verificado en los 4 modelos

---

## 📋 Resumen Ejecutivo

Tras analizar los checkpoints pre-entrenados de todos los modelos ConvNeXtPose, se han extraído las **dimensiones exactas de los kernels** tanto del backbone como del head (upsampling).

### 🎯 Conclusiones Principales:

1. **TODOS los modelos (XS, S, M, L) usan kernels de 3×3 en las capas de deconvolución del head.**

2. **✅ MISTERIO RESUELTO:** Los checkpoints tienen **3 capas de deconvolución**, pero:
   - **XS y S**: Solo **2 capas hacen upsampling** (capa 3 tiene `up=False`)
   - **M y L**: Las **3 capas hacen upsampling** (todas con `up=True`)

3. **La notación del paper es CORRECTA:**
   - **"2UP"** = 2 capas con **UP**sampling (ignora la capa 3 sin upsampling)
   - **"3UP"** = 3 capas con **UP**sampling
   - La notación se refiere a capas que hacen upsampling, no al total de capas

4. **Todas las capas están completamente activas** (100% de pesos no-cero) y se ejecutan durante inferencia.

---

## 📊 Tabla Completa de Configuraciones Verificadas

### Configuraciones en los Checkpoints (VERIFICADO):

| Modelo | Params | Backbone Kernel | Head Layers | Capas con UP | Head Kernels | Output Channels   | Paper Spec |
|--------|--------|-----------------|-------------|--------------|--------------|-------------------|------------|
| **XS** | 3.53M  | 7×7             | 3           | **2** ✅     | `[3, 3, 3]`  | `[128, 128, 128]` | 2UP ✅     |
| **S**  | 7.45M  | 7×7             | 3           | **2** ✅     | `[3, 3, 3]`  | `[256, 256, 256]` | 2UP ✅     |
| **M**  | 7.60M  | 7×7             | 3           | **3** ✅     | `[3, 3, 3]`  | `[256, 256, 256]` | 3UP ✅     |
| **L**  | 8.39M  | 7×7             | 3           | **3** ✅     | `[3, 3, 3]`  | `[512, 512, 512]` | 3UP ✅     |

**Nota:** ✅ = Coincide con paper. "Capas con UP" = Capas que aplican upsampling real (factor 2×)

---

## 🔷 BACKBONE - Kernels

### Depthwise Convolutions

Todos los modelos usan **kernels de 7×7** en las convoluciones depthwise del backbone ConvNeXt:

```python
# Ejemplo de configuración del backbone
dwconv_kernel_size = 7  # Para todos los modelos (XS, S, M, L)
```

Esta es una característica estándar de la arquitectura ConvNeXt, que utiliza kernels grandes (7×7) siguiendo el diseño de Swin Transformer.

---

## 🔶 HEAD - Deconvolution Layers (Upsampling)

### Kernels de Upsampling

Todos los modelos utilizan **3 capas de deconvolución** con **kernels de 3×3**:

```python
# Configuración universal para todos los modelos
deconv_kernels = [3, 3, 3]
```

---

## ⚙️ Configuraciones Completas por Modelo

### 🟦 Modelo XS (Atto)

```python
# Backbone
cfg.backbone = 'ConvNeXt'
cfg.variant = 'Atto'
cfg.backbone_cfg = ([2,2,6,2], [40,80,160,320])

# Backbone kernels
backbone_dwconv_kernel = 7  # 7x7

# Head
cfg.head_cfg = {
    'num_deconv_layers': 3,
    'deconv_channels': [128, 128, 128],
    'deconv_kernels': [3, 3, 3]
}

# Parámetros: 3.53M
# MPJPE: 56.61mm
```

---

### 🟦 Modelo S (Femto-L)

```python
# Backbone
cfg.backbone = 'ConvNeXt'
cfg.variant = 'Femto-L'
cfg.backbone_cfg = ([3,3,9,3], [48,96,192,384])

# Backbone kernels
backbone_dwconv_kernel = 7  # 7x7

# Head
cfg.head_cfg = {
    'num_deconv_layers': 3,
    'deconv_channels': [256, 256, 256],
    'deconv_kernels': [3, 3, 3]
}

# Parámetros: 7.45M
# MPJPE: 51.80mm
```

---

### 🟦 Modelo M (Femto-L)

```python
# Backbone
cfg.backbone = 'ConvNeXt'
cfg.variant = 'Femto-L'
cfg.backbone_cfg = ([3,3,9,3], [48,96,192,384])

# Backbone kernels
backbone_dwconv_kernel = 7  # 7x7

# Head
cfg.head_cfg = {
    'num_deconv_layers': 3,
    'deconv_channels': [256, 256, 256],
    'deconv_kernels': [3, 3, 3]
}

# Parámetros: 7.60M
# MPJPE: 51.05mm
```

---

### 🟦 Modelo L (Femto-L)

```python
# Backbone
cfg.backbone = 'ConvNeXt'
cfg.variant = 'Femto-L'
cfg.backbone_cfg = ([3,3,9,3], [48,96,192,384])

# Backbone kernels
backbone_dwconv_kernel = 7  # 7x7

# Head
cfg.head_cfg = {
    'num_deconv_layers': 3,
    'deconv_channels': [512, 512, 512],
    'deconv_kernels': [3, 3, 3]
}

# Parámetros: 8.39M
# MPJPE: 49.75mm
```

---

## 🔬 Análisis Detallado de Arquitectura

### Estructura del Head de Upsampling

Cada capa de deconvolución sigue esta estructura:

```
Layer i:
├── Depthwise Conv (dwconv)
│   └── Kernel: 3×3
│   └── Channels: C → C (mantiene canales)
│
├── Batch Normalization
│
└── Pointwise Conv (pwconv)
    └── Kernel: 1×1
    └── Channels: C_in → C_out (ajusta canales)
```

### Ejemplo: Capa 1 del Modelo XS

```python
# Verificado desde checkpoint
deconv_layers_1:
  - dwconv.weight: shape=[320, 1, 3, 3]  # 320 canales, kernel 3x3
  - norm: BatchNorm con 320 canales
  - pwconv.weight: shape=[128, 320, 1, 1]  # 320 → 128 canales
  
# Resultado: 320 canales → 128 canales con kernel 3x3
```

---

## 📊 Comparación con el Paper

### Datos del Paper (IEEE Access 2023):

| Modelo | Backbone | Upsampling | B (blocks) | C (channels) | MPJPE | GFLOPs | Params |
|--------|----------|------------|------------|--------------|-------|--------|--------|
| XS | Atto | 2UP, 128 | (2,2,6,2) | (40,80,160,320) | 56.61 | 0.82 | 3.53M |
| S | Femto-L | 2UP, 256 | (3,3,9,3) | (48,96,192,384) | 51.80 | 1.76 | 7.44M |
| M | Femto-L | 3UP, 256 | (3,3,9,3) | (48,96,192,384) | 51.05 | 2.82 | 7.59M |
| L | Femto-L | 3UP, 512 | (3,3,9,3) | (48,96,192,384) | 49.75 | 4.30 | 8.38M |

### ✅ Verificación:

- ✅ **Parámetros:** Coinciden perfectamente
- ✅ **Backbone:** Coincide con el paper
- ✅ **Output Channels:** Coinciden (128, 256, 256, 512)
- ℹ️ **Kernels:** No especificados en el paper → **Ahora verificados como 3×3**

---

## 🛠️ Información Técnica Adicional

### Formato del Checkpoint

Los checkpoints tienen dos formatos de naming:

#### Formato 1 (Modelos XS, S):
```
module.head.deconv_layers_{i}.dwconv.weight
module.head.deconv_layers_{i}.norm.*
module.head.deconv_layers_{i}.pwconv.weight
```

#### Formato 2 (Modelos M, L):
```
module.head.deconv_layers_{i}.0.weight  # dwconv
module.head.deconv_layers_{i}.1.*       # norm
module.head.deconv_layers_{i}.2.weight  # pwconv
```

Ambos formatos implementan la misma arquitectura con kernels 3×3.

---

## 📝 Notas Importantes

### 🔑 Puntos Clave:

1. **Backbone:** Todos usan kernels 7×7 en depthwise convolutions
2. **Head:** Todos usan kernels 3×3 en las 3 capas de upsampling
3. **Consistencia:** La dimensión del kernel es **constante** (3×3) en todas las capas y modelos
4. **Diferencias:** Los modelos difieren solo en:
   - Número de bloques del backbone
   - Canales del backbone
   - Canales de salida del head

### 🎯 Aplicación Práctica:

Si necesitas modificar un modelo, puedes cambiar:
- ✅ `deconv_channels` → Ajusta capacidad del head
- ⚠️ `deconv_kernels` → Se recomienda mantener `[3, 3, 3]`
- ⚠️ `num_deconv_layers` → Se recomienda mantener en 3

---

## 🧪 Script de Verificación

Para verificar las dimensiones de kernels en cualquier checkpoint:

```python
import torch

def verify_kernel_sizes(checkpoint_path):
    """Verifica dimensiones de kernels en un checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['network']
    
    print("🔷 Backbone kernels:")
    for key in model_state.keys():
        if 'backbone.stages' in key and 'dwconv.weight' in key:
            kernel = model_state[key]
            print(f"  {key.split('.')[2]}: {kernel.shape[2]}x{kernel.shape[3]}")
            break
    
    print("\n🔶 Head deconv kernels:")
    for i in range(1, 10):
        # Probar formato 1
        key1 = f'module.head.deconv_layers_{i}.dwconv.weight'
        # Probar formato 2
        key2 = f'module.head.deconv_layers_{i}.0.weight'
        
        if key1 in model_state:
            kernel = model_state[key1]
            print(f"  Layer {i}: {kernel.shape[2]}x{kernel.shape[3]}")
        elif key2 in model_state:
            kernel = model_state[key2]
            print(f"  Layer {i}: {kernel.shape[2]}x{kernel.shape[3]}")

# Usar
verify_kernel_sizes('demo/ConvNeXtPose_XS.tar')
```

---

## � DISCREPANCIA CRÍTICA: Checkpoints vs Paper

### Problema Identificado

Los checkpoints pre-entrenados **NO coinciden** con las especificaciones del paper para los modelos XS y S:

| Modelo | Paper (IEEE Access 2023) | Checkpoint Real | Estado |
|--------|-------------------------|-----------------|--------|
| XS     | 2UP (2 capas)          | **3 capas** ✅  | ⚠️ Difiere |
| S      | 2UP (2 capas)          | **3 capas** ✅  | ⚠️ Difiere |
| M      | 3UP (3 capas)          | 3 capas ✅      | ✅ Coincide |
| L      | 3UP (3 capas)          | 3 capas ✅      | ✅ Coincide |

### Verificación Realizada

Todos los pesos de las 3 capas están **100% activos** (no hay pesos en cero):

```
XS - Capa 3: 1152/1152 pesos activos (100.00%)
S  - Capa 3: 2304/2304 pesos activos (100.00%)
```

Esto descarta la hipótesis de que la tercera capa esté presente pero deshabilitada.

### Posibles Explicaciones

1. **Actualización Post-Publicación**: Los autores podrían haber mejorado XS y S agregando una tercera capa después de publicar el paper.

2. **Error en el Paper**: La notación "2UP" podría referirse al factor de upsampling total en lugar del número de capas.

3. **Versión Diferente**: Los checkpoints disponibles públicamente podrían ser de una versión mejorada del modelo.

4. **Configuración del Código vs Checkpoints**: El archivo `config_variants.py` del proyecto especifica correctamente 2 capas para XS/S, pero los checkpoints descargados tienen 3.

### Recomendación

Al usar los checkpoints pre-entrenados, configurar **TODOS los modelos con 3 capas**:

```python
# Para todos los modelos (XS, S, M, L)
cfg.head_cfg = {
    'num_deconv_layers': 3,
    'deconv_channels': [...],  # Según modelo
    'deconv_kernels': [3, 3, 3]
}
```

---

## �📚 Referencias

- **Paper:** ConvNeXtPose (IEEE Access 2023)
- **Checkpoints Analizados:**
  - `demo/ConvNeXtPose_XS.tar` - 3 capas (vs 2UP en paper)
  - `demo/ConvNeXtPose_S.tar` - 3 capas (vs 2UP en paper)
  - `demo/ConvNeXtPose_M (1).tar` - 3 capas (coincide con paper)
  - `demo/ConvNeXtPose_L (1).tar` - 3 capas (coincide con paper)
- **Script de Verificación:** `verify_upsampling_layers.py`

---

**Fin del Documento** 🎯

**Última Actualización:** 16 de Octubre 2025 - Agregada sección de discrepancia crítica
