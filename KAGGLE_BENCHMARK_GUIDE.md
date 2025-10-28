# 🚀 Guía Completa: Benchmark de ConvNeXtPose en Kaggle

**Actualizado:** 17 de Octubre 2025  
**Notebook:** `convnextpose (5).ipynb`

---

## 📋 Resumen

Este notebook ejecuta un benchmark automatizado de **todos los modelos ConvNeXtPose** (XS, S, M, L) en Kaggle con:

✅ Configuraciones **verificadas desde checkpoints reales**  
✅ Extracción automática de archivos `.tar` (formato legacy → moderno)  
✅ Mapeo correcto de keys para modelos M y L  
✅ Comparación automática con resultados del paper  
✅ Gráficos y reportes exportables  

---

## 🎯 Configuraciones Verificadas

### Análisis de Checkpoints Reales:

Todos los valores han sido **extraídos directamente de los checkpoints** pre-entrenados:

| Modelo | Params | Backbone | Head Kernels | Head Channels | MPJPE Esperado |
|--------|--------|----------|--------------|---------------|----------------|
| **XS** | 3.53M | Atto (7×7) | `[3,3,3]` | `[128,128,128]` | 56.61 mm |
| **S** | 7.45M | Femto-L (7×7) | `[3,3,3]` | `[256,256,256]` | 51.80 mm |
| **M** | 7.60M | Femto-L (7×7) | `[3,3,3]` | `[256,256,256]` | 51.05 mm |
| **L** | 8.39M | Femto-L (7×7) | `[3,3,3]` | `[512,512,512]` | 49.75 mm |

**Fuente:** Documentación completa en `DIMENSIONES_KERNELS_VERIFICADAS.md`

---

## 🔧 Configuración del Upsampling

### ✅ Todos los Modelos Usan "Legacy Mode"

Los checkpoints pre-entrenados fueron guardados con **Legacy mode**, donde:

```python
cfg.head_cfg = None           # NO especificar head_cfg
cfg.depth = valor_por_modelo  # Controla canales del head
cfg.depth_dim = 64            # Profundidad de predicción 3D
```

### 📊 Valores de `cfg.depth` por Modelo:

```python
# XS
cfg.depth = 128  # → head channels [128, 128, 128]

# S
cfg.depth = 256  # → head channels [256, 256, 256]

# M
cfg.depth = 256  # → head channels [256, 256, 256]

# L
cfg.depth = 512  # → head channels [512, 512, 512]
```

**Nota:** Aunque los modelos M y L tienen los mismos canales (256 y 512), difieren en el número de capas con upsampling real:
- **M:** 3 capas con upsampling
- **L:** 3 capas con upsampling pero configuración diferente internamente

---

## 🗂️ Extracción de Checkpoints

### Formato de Archivos `.tar`

Los checkpoints de ConvNeXtPose tienen un formato especial:

```
ConvNeXtPose_L.tar (archivo ZIP disfrazado)
└── snapshot_83.pth/          ← Directorio con formato legacy
    ├── data.pkl              ← Metadatos del checkpoint
    ├── version               ← Versión de PyTorch
    └── data/                 ← Pesos del modelo
        ├── 0                 ← Storage files (binarios)
        ├── 1
        └── ...
```

### Proceso de Conversión Automática:

El notebook incluye una función `extract_checkpoint()` que:

1. ✅ Detecta formato (ZIP o TAR real)
2. ✅ Extrae contenido a directorio temporal
3. ✅ Carga formato legacy con `LegacyUnpickler` personalizado
4. ✅ Convierte a formato PyTorch moderno (`.pth`)
5. ✅ Guarda en `output/model_dump/snapshot_XX.pth`
6. ✅ Verifica integridad del checkpoint

**Resultado:** Archivos `.pth` estándar compatibles con `torch.load()`

---

## 🔑 Mapeo de Keys Legacy → Moderno

### Problema:

Los modelos M y L usan un formato de naming diferente a XS y S:

```python
# Formato XS, S (moderno):
'module.head.deconv_layers_1.dwconv.weight'
'module.head.deconv_layers_1.norm.bias'
'module.head.deconv_layers_1.pwconv.weight'

# Formato M, L (legacy):
'module.head.deconv_layers_1.0.weight'    # dwconv
'module.head.deconv_layers_1.1.bias'      # norm
'module.head.deconv_layers_1.2.weight'    # pwconv
```

### Solución:

El notebook incluye la función `map_legacy_head_keys()`:

```python
def map_legacy_head_keys(state_dict):
    """Mapea keys del formato legacy (M, L) al formato moderno (XS, S)"""
    new_state = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.head.deconv_layers_'):
            suffix = k.split('.', 3)[-1]
            
            # Mapeo: .0 → .dwconv, .1 → .norm, .2 → .pwconv
            if suffix.startswith('0.'):
                new_k = k.replace('.0.', '.dwconv.')
            elif suffix.startswith('1.'):
                new_k = k.replace('.1.', '.norm.')
            elif suffix.startswith('2.'):
                new_k = k.replace('.2.', '.pwconv.')
            else:
                new_k = k
            
            new_state[new_k] = v
        else:
            new_state[k] = v
    
    return new_state
```

Este mapeo se aplica **automáticamente** al cargar los checkpoints de M y L.

---

## 📝 Uso del Notebook

### 1️⃣ **Setup Inicial**

```python
# Celda 1: Clonar repositorio
!git clone https://github.com/EstebanCabreraArbizu/ConvNeXtPose.git
%cd ConvNeXtPose

# Verificar GPU
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 2️⃣ **Configurar Dataset Human3.6M**

```python
# Celda 2: Enlazar dataset de Kaggle
KAGGLE_DATASET_PATH = '/kaggle/input/tu-dataset-human36m'

!python setup_kaggle_dataset.py --kaggle-input {KAGGLE_DATASET_PATH} \
                                 --project-root /kaggle/working/ConvNeXtPose
```

### 3️⃣ **Descargar y Extraer Checkpoints**

```python
# Celda 3: Descargar desde Google Drive
import gdown

folder_id = "12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI"
gdown.download_folder(id=folder_id, output="models_tar", quiet=False)

# Celda 4: Extraer y convertir (automático)
# La función extract_checkpoint() maneja todo el proceso
```

### 4️⃣ **Ejecutar Benchmark**

```python
# Celda 5: Benchmark automatizado de todos los modelos
# El script test_model() ejecuta secuencialmente:
# - XS (epoch detectado automáticamente)
# - S (epoch detectado automáticamente)
# - M (epoch detectado automáticamente)
# - L (epoch detectado automáticamente)

# Tiempo estimado: 40-80 minutos para los 4 modelos (GPU T4 x2)
```

### 5️⃣ **Analizar Resultados**

```python
# Celda 6: Generar gráficos y análisis
# Crea automáticamente:
# - benchmark_results.json (datos estructurados)
# - BENCHMARK_REPORT.md (reporte completo)
# - benchmark_comparison.png (gráficos)
```

---

## 📊 Resultados Esperados

### Comparación con el Paper:

El benchmark compara automáticamente los resultados obtenidos con los del paper:

| Modelo | MPJPE Paper | Rango Aceptable | Clasificación |
|--------|-------------|-----------------|---------------|
| XS | 56.61 mm | 55.6 - 57.6 mm | ±1mm Excelente |
| S | 51.80 mm | 50.8 - 52.8 mm | ±1mm Excelente |
| M | 51.05 mm | 50.0 - 52.0 mm | ±1mm Excelente |
| L | 49.75 mm | 48.8 - 50.8 mm | ±1mm Excelente |

**Criterios de Evaluación:**
- ✅ **Excelente:** Diferencia < 1mm
- ✓✓ **Muy bueno:** Diferencia < 2mm
- ✓ **Aceptable:** Diferencia < 5mm
- ⚠️ **Revisar:** Diferencia ≥ 5mm

---

## 🛠️ Troubleshooting

### Problema: "Checkpoint no encontrado"

**Causa:** Extracción incompleta o path incorrecto

**Solución:**
```python
# Verificar checkpoints extraídos
!ls -lh output/model_dump/

# Re-ejecutar extracción si es necesario
```

### Problema: "Size mismatch en head.deconv_layers"

**Causa:** Configuración incorrecta de `cfg.depth`

**Solución:**
```python
# Verificar que usas Legacy mode:
cfg.head_cfg = None  # ← IMPORTANTE
cfg.depth = valor_correcto  # Ver tabla arriba
```

### Problema: "Keys no coinciden (modelos M, L)"

**Causa:** Formato legacy no mapeado

**Solución:** El notebook aplica `map_legacy_head_keys()` automáticamente. Si persiste:
```python
# Verificar que el mapeo se está aplicando
state_dict = map_legacy_head_keys(checkpoint['network'])
model.load_state_dict(state_dict)
```

### Problema: "Testing muy lento"

**Causa:** Ejecutando en CPU en lugar de GPU

**Solución:**
1. Settings → Accelerator → **GPU T4 x2**
2. Reiniciar notebook
3. Verificar con `torch.cuda.is_available()`

---

## 📁 Estructura de Archivos Generados

Después de ejecutar el benchmark:

```
/kaggle/working/ConvNeXtPose/
├── output/
│   ├── model_dump/
│   │   ├── snapshot_XX.pth  (XS)
│   │   ├── snapshot_YY.pth  (S)
│   │   ├── snapshot_ZZ.pth  (M)
│   │   └── snapshot_WW.pth  (L)
│   │
│   ├── result/
│   │   ├── benchmark_results.json       ← Resultados estructurados
│   │   ├── BENCHMARK_REPORT.md          ← Reporte completo
│   │   └── benchmark_comparison.png     ← Gráficos comparativos
│   │
│   └── log/
│       ├── test_XS.log
│       ├── test_S.log
│       ├── test_M.log
│       └── test_L.log
```

---

## 💾 Guardar y Exportar Resultados

### Opción 1: Commit del Notebook

Kaggle guarda automáticamente todo en `/kaggle/working/` al hacer commit.

### Opción 2: Descargar Output

```python
# En la última celda del notebook:
!zip -r benchmark_results.zip output/result/
!zip -r benchmark_logs.zip output/log/

# Descargar desde el panel de Output del notebook
```

### Opción 3: Copiar a Dataset

```python
# Crear un nuevo dataset de Kaggle con los resultados
!kaggle datasets init -p output/result/
# Editar dataset-metadata.json
!kaggle datasets create -p output/result/
```

---

## 🔬 Validación de Configuraciones

Para verificar que las configuraciones son correctas:

```python
# Script de verificación rápida
import torch
from main.config import cfg
from main.model import get_pose_net

def verify_model_config(model_name, variant, backbone_cfg, depth):
    """Verifica que la configuración carga correctamente"""
    cfg.head_cfg = None
    cfg.variant = variant
    cfg.backbone_cfg = backbone_cfg
    cfg.depth = depth
    cfg.depth_dim = 64
    
    # Crear modelo
    model = get_pose_net(cfg, is_train=False, joint_num=17)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"{model_name}: {total_params/1e6:.2f}M parámetros")
    
    return model

# Verificar todos los modelos
verify_model_config('XS', 'Atto', ([2,2,6,2],[40,80,160,320]), 128)
verify_model_config('S', 'Femto-L', ([3,3,9,3],[48,96,192,384]), 256)
verify_model_config('M', 'Femto-L', ([3,3,9,3],[48,96,192,384]), 256)
verify_model_config('L', 'Femto-L', ([3,3,9,3],[48,96,192,384]), 512)
```

**Parámetros esperados:**
- XS: ~3.53M
- S: ~7.45M
- M: ~7.60M
- L: ~8.39M

---

## 📚 Documentación Adicional

- **Análisis de Kernels:** `DIMENSIONES_KERNELS_VERIFICADAS.md`
- **Guía Completa:** `GUIA_COMPLETA_ACTUALIZADA.md`
- **Paper Original:** ConvNeXtPose (IEEE Access 2023)
- **Repositorio:** https://github.com/EstebanCabreraArbizu/ConvNeXtPose

---

## ✅ Checklist de Ejecución

Antes de ejecutar el benchmark, verifica:

- [ ] GPU activada (T4 x2 o P100)
- [ ] Dataset Human3.6M enlazado correctamente
- [ ] Checkpoints descargados y extraídos
- [ ] Configuraciones verificadas (Legacy mode)
- [ ] Espacio suficiente en disco (~5GB)

Durante la ejecución:

- [ ] Monitor GPU usage (`nvidia-smi`)
- [ ] Verificar logs en tiempo real
- [ ] Guardar resultados parciales

Después de completar:

- [ ] Revisar BENCHMARK_REPORT.md
- [ ] Comparar gráficos generados
- [ ] Exportar resultados
- [ ] Documentar hallazgos

---

## 🎯 Conclusiones

Este notebook proporciona:

✅ **Configuraciones 100% verificadas** desde checkpoints reales  
✅ **Extracción automática** de archivos `.tar` legacy  
✅ **Mapeo correcto** de keys para todos los modelos  
✅ **Benchmark completo** de los 4 modelos  
✅ **Análisis comparativo** automático con el paper  
✅ **Resultados exportables** en múltiples formatos  

**Tiempo total:** ~1-2 horas (incluyendo descarga y extracción)  
**Costo:** Gratis en Kaggle con GPU T4 x2

---

**¿Preguntas o problemas?**  
Consulta la documentación completa o abre un issue en el repositorio.

**¡Buena suerte con tu benchmark!** 🚀
