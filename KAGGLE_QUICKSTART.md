# 🚀 Guía Rápida: ConvNeXtPose en Kaggle

## Problema Identificado

Tu dataset en Kaggle tiene esta estructura:
```
/kaggle/input/human36m-dataset/
├── S9_ACT2_16/          # Sujeto 9
├── S11_ACT2_16/         # Sujeto 11
└── annotations (1)/     # Anotaciones con nombre diferente
```

Pero el código espera:
```
data/Human36M/
├── images/
│   ├── S9/
│   └── S11/
└── annotations/
```

## ✅ Solución: Script Automático

He creado `setup_kaggle_dataset.py` que **crea enlaces simbólicos** sin copiar archivos, ahorrando espacio.

---

## 📋 Pasos en Kaggle Notebook

### 1. Subir el Proyecto a Kaggle

Opción A: Desde Git
```python
!git clone https://github.com/EstebanCabreraArbizu/ConvNeXtPose.git
%cd ConvNeXtPose
```

Opción B: Subir como Dataset de Kaggle
- Sube todo el proyecto como dataset
- Monta en `/kaggle/input/convnextpose`

### 2. Configurar Dataset Automáticamente

```python
# Ejecutar el script de setup
!python setup_kaggle_dataset.py --kaggle-input /kaggle/input/human36m-dataset

# Esto crea enlaces simbólicos en /kaggle/working/data/Human36M/
# sin copiar archivos (ahorra ~30GB de espacio)
```

### 3. Configurar Variable de Entorno

```python
import os
os.environ['CONVNEXPOSE_DATA_DIR'] = '/kaggle/working/data'
```

### 4. Preparar Checkpoints

```python
import tarfile
import os

# Directorio para modelos
os.makedirs('/kaggle/working/ConvNeXtPose/output/model_dump', exist_ok=True)

# Extraer modelo L (ajusta según tu dataset de modelos)
tar_path = '/kaggle/input/convnextpose-models/models_tar/ConvNeXtPose_L.tar'
with tarfile.open(tar_path, 'r') as tar:
    tar.extractall('/kaggle/working/ConvNeXtPose/output/model_dump')

# Renombrar si es necesario (ej: snapshot_68.pth -> snapshot_70.pth)
# os.rename('output/model_dump/snapshot_68.pth', 'output/model_dump/snapshot_70.pth')
```

### 5. Ejecutar Testing

```python
%cd /kaggle/working/ConvNeXtPose/main

# Testing modelo L
!python test.py --gpu 0 --epochs 68 --variant L

# Testing modelo M (si tienes el checkpoint)
# !python test.py --gpu 0 --epochs 68 --variant M
```

---

## 🔍 Verificación de Estructura

Para verificar que todo esté correctamente configurado:

```python
!python setup_kaggle_dataset.py --verify /kaggle/working/data
```

Salida esperada:
```
  ✓ Human36M directory
  ✓ annotations folder
  ✓ images folder
  ✓ S9 subject
  ✓ S11 subject
  ✓ bbox_root folder (optional)
```

---

## 📊 Estructura Completa del Notebook

```python
# ============================================
# PASO 1: Clonar Repositorio
# ============================================
!git clone https://github.com/EstebanCabreraArbizu/ConvNeXtPose.git
%cd ConvNeXtPose

# ============================================
# PASO 2: Configurar Dataset (enlaces simbólicos)
# ============================================
!python setup_kaggle_dataset.py --kaggle-input /kaggle/input/human36m-dataset

# ============================================
# PASO 3: Variable de Entorno
# ============================================
import os
os.environ['CONVNEXPOSE_DATA_DIR'] = '/kaggle/working/data'

# ============================================
# PASO 4: Preparar Checkpoints
# ============================================
import tarfile
os.makedirs('output/model_dump', exist_ok=True)

# Extraer modelo
tar_path = '/kaggle/input/convnextpose-models/models_tar/ConvNeXtPose_L.tar'
with tarfile.open(tar_path, 'r') as tar:
    tar.extractall('output/model_dump')

# Verificar que se extrajo correctamente
!ls -lh output/model_dump/

# ============================================
# PASO 5: Verificar Estructura
# ============================================
!python setup_kaggle_dataset.py --verify /kaggle/working/data

# ============================================
# PASO 6: Ejecutar Testing
# ============================================
%cd main
!python test.py --gpu 0 --epochs 68 --variant L
```

---

## 🎯 Ventajas de Este Método

✅ **Sin copias**: Enlaces simbólicos ahorran ~30-40GB de espacio  
✅ **Automático**: Detecta automáticamente las variantes de nombres  
✅ **Flexible**: Funciona con diferentes estructuras de Kaggle  
✅ **Verificable**: Comando `--verify` para diagnosticar problemas  
✅ **No destructivo**: No modifica el dataset original  

---

## 🐛 Troubleshooting

### Error: "Cannot find checkpoint: tried snapshot_70.pth.tar and snapshot_70.pth"

**Causa**: El checkpoint extraído tiene otro nombre (ej: `snapshot_68.pth`)

**Solución**:
```python
# Opción 1: Usar el epoch correcto
!python test.py --gpu 0 --epochs 68 --variant L

# Opción 2: Renombrar el checkpoint
import os
os.rename('output/model_dump/snapshot_68.pth', 
          'output/model_dump/snapshot_70.pth')
```

### Error: "No se encontró carpeta annotations"

**Causa**: El script no detectó automáticamente el nombre

**Solución**: Verifica el nombre exacto en tu dataset
```python
!ls /kaggle/input/human36m-dataset/
```

Luego edita manualmente el enlace:
```python
import os
os.symlink('/kaggle/input/human36m-dataset/annotations (1)',
           '/kaggle/working/data/Human36M/annotations')
```

### Error: "OSError: [Errno 30] Read-only file system"

**Causa**: Intentas crear enlaces en `/kaggle/input` (solo lectura)

**Solución**: Usa siempre `/kaggle/working` como destino:
```python
!python setup_kaggle_dataset.py \
    --kaggle-input /kaggle/input/human36m-dataset \
    --output /kaggle/working/data
```

### Error: "RuntimeError: Expected all tensors to be on the same device"

**Causa**: Modelo muy grande para la GPU asignada

**Solución**: Reduce batch size en `config.py`:
```python
# Antes de importar config
import sys
sys.path.insert(0, '/kaggle/working/ConvNeXtPose/main')

from config import cfg
cfg.test_batch_size = 8  # Default es 16, reduce a 8 o 4
```

---

## 📈 Resultados Esperados

### Protocol 2 (MPJPE)

| Modelo | Paper | Rango Aceptable |
|--------|-------|-----------------|
| **L**  | 42.3mm | 41-45mm |
| **M**  | 44.6mm | 43-47mm |
| **S**  | ~48mm | 46-50mm |

Si obtienes valores fuera de este rango, verifica:
1. ✓ Dataset configurado correctamente (`--verify`)
2. ✓ Variante correcta (`--variant L` para modelo L)
3. ✓ Checkpoint correcto (no mezclar checkpoints de diferentes variantes)
4. ✓ Epoch correcto del checkpoint

---

## 🔗 Referencias

- **Repositorio**: https://github.com/EstebanCabreraArbizu/ConvNeXtPose
- **Paper Original**: ConvNeXtPose (ver `README.md`)
- **Documentación Completa**: Ver `ARCHITECTURE_ADAPTATION_COMPLETE.md`
- **Guía de Testing**: Ver `GUIA_TESTING_MODELOS_L_M.md`

---

## 💡 Tips Adicionales

1. **Activar GPU en Kaggle**: Settings → Accelerator → GPU T4 x2
2. **Aumentar tiempo**: Settings → Persistence → 12 hours
3. **Guardar outputs**: Los resultados se guardan en `/kaggle/working/ConvNeXtPose/output/result/`
4. **Commit outputs**: Kaggle guarda automáticamente todo en `/kaggle/working/`

---

**¿Preguntas?** Revisa `ARCHITECTURE_ADAPTATION_COMPLETE.md` para detalles técnicos.
