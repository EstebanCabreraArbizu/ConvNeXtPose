# 🔧 Solución: Carpetas Anidadas en Dataset de Kaggle

## 🎯 Problema Identificado

Tu dataset de Kaggle tiene una estructura anidada:

```
/kaggle/input/human36m-dataset/
└── annotations (1)/          ← Carpeta contenedor
    └── annotations/          ← Carpeta REAL con los archivos JSON
        ├── Human36M_subject1_camera.json
        ├── Human36M_subject1_data.json
        ├── Human36M_subject1_joint_3d.json
        └── ...
```

El script anterior enlazaba `annotations (1)` directamente, pero Human36M.py esperaba encontrar los archivos JSON en el primer nivel de `annotations/`.

## ✅ Solución Implementada

### 1. **Detección Inteligente de Carpetas Anidadas**

El script ahora:
1. Busca en múltiples niveles: `annotations (1)/annotations/`, `annotations (1)/`, `annotations/`
2. Verifica que la carpeta contiene archivos JSON (no es solo un contenedor vacío)
3. Enlaza directamente la carpeta con contenido real

```python
annotations_candidates = [
    kaggle_input / 'annotations (1)' / 'annotations',  # ← Detecta el nivel correcto
    kaggle_input / 'annotations (1)',
    kaggle_input / 'annotations',
]

# Verificar que contiene archivos JSON
if candidate.is_dir():
    json_files = list(candidate.glob('*.json'))
    if json_files:
        annotations_src = candidate  # ← Solo usa carpetas con contenido
```

### 2. **Búsqueda Recursiva para bbox_root**

Similar para estructuras complejas de `bbox_root`:

```python
# Busca recursivamente el archivo bbox_root_human36m_output.json
json_file = list(candidate.rglob('bbox_root_human36m_output.json'))
if json_file:
    bbox_src = json_file[0].parent  # ← Usa el directorio padre del archivo
```

## 🛠️ Herramientas Nuevas

### 1. **diagnose_kaggle_dataset.py**

Script de diagnóstico para inspeccionar tu dataset:

```bash
!python diagnose_kaggle_dataset.py /kaggle/input/your-dataset
```

**Salida esperada:**
```
📁 [1/3] Buscando carpeta annotations...

  📁 annotations (1)
     ⚠️  Sin archivos JSON (puede ser contenedor)
    📁 annotations (1)/annotations
       ✓ 18 archivos JSON
         • Human36M_subject1_camera.json
         • Human36M_subject1_data.json
         • Human36M_subject1_joint_3d.json

✅ Encontradas 2 carpeta(s) con 'annotation' en el nombre
💡 Recomendación: Usar 'annotations (1)/annotations' (18 archivos JSON)
```

### 2. **setup_kaggle_dataset.py (Actualizado)**

Ahora muestra información detallada durante la configuración:

```
📁 [1/3] Configurando annotations...
  ✓ Encontrado: annotations (1)/annotations
    (18 archivos JSON detectados)
  ✓ Creado: annotations -> /kaggle/input/.../annotations (1)/annotations
```

### 3. **verify_kaggle_structure.py**

Verifica que todo esté correctamente enlazado después de la configuración.

## 📝 Uso en Kaggle Notebook

### Flujo Completo:

```python
# 1. Clonar repo
!git clone https://github.com/EstebanCabreraArbizu/ConvNeXtPose.git
%cd ConvNeXtPose

# 2. Definir dataset
KAGGLE_DATASET_PATH = '/kaggle/input/your-dataset-name'

# 3. (OPCIONAL) Diagnosticar para ver la estructura
!python diagnose_kaggle_dataset.py {KAGGLE_DATASET_PATH}

# 4. Configurar - Automáticamente detecta carpetas anidadas
!python setup_kaggle_dataset.py \
    --kaggle-input {KAGGLE_DATASET_PATH} \
    --project-root /kaggle/working/ConvNeXtPose

# 5. Verificar
!python verify_kaggle_structure.py

# 6. Testing
%cd main
!python test.py --gpu 0 --epochs 83 --variant L
```

## 🔍 Casos Soportados

### Caso 1: Annotations anidado
```
✅ annotations (1)/annotations/         ← Detecta archivos JSON aquí
✅ annotations/                         ← Alternativa directa
```

### Caso 2: bbox_root profundamente anidado
```
✅ Bounding box.../Bounding box.../Human3.6M/Subject 9,11.../bbox_root_human36m_output.json
✅ bbox_root/Subject 9,11.../bbox_root_human36m_output.json
✅ bbox_root/bbox_root_human36m_output.json
```

### Caso 3: Sujetos con variantes de nombre
```
✅ S9_ACT2_16/      ← Prioridad 1 (más común)
✅ S9_ACT2_16/      ← Prioridad 2 (alternativa)
✅ S9_ACT2/         ← Prioridad 3
✅ S9/              ← Prioridad 4
```

## ✅ Resultado Final

Después de ejecutar la configuración:

```
ConvNeXtPose/data/
├── dataset.py                           ← Módulo original INTACTO
├── multiple_datasets.py                 ← Módulo original INTACTO
└── Human36M/
    ├── Human36M.py                      ← Módulo original INTACTO
    ├── annotations -> /kaggle/input/.../annotations (1)/annotations  ← Nivel correcto
    ├── images/
    │   ├── S9 -> /kaggle/input/.../S9_ACT2_16
    │   └── S11 -> /kaggle/input/.../S11_ACT2_16
    └── bbox_root -> /kaggle/input/.../Subject 9,11.../
```

**Verificación:**
```python
import os
import json

# Verificar que los archivos JSON son accesibles
annot_path = '/kaggle/working/ConvNeXtPose/data/Human36M/annotations'
json_files = [f for f in os.listdir(annot_path) if f.endswith('.json')]
print(f"✓ {len(json_files)} archivos JSON detectados en annotations/")

# Probar lectura
with open(os.path.join(annot_path, json_files[0])) as f:
    data = json.load(f)
    print(f"✓ Archivo JSON legible: {json_files[0]}")
```

## 🎉 Sin Errores

Ahora el testing debería funcionar sin errores:

```python
from config import cfg
from base import Tester
from dataset import DatasetLoader  # ← Sin ModuleNotFoundError
from Human36M import Human36M       # ← Sin problemas de imports
```

Y Human36M.py puede leer las annotations correctamente:

```python
# En Human36M.py:
with open(osp.join(self.annot_path, 'Human36M_subject1_data.json'),'r') as f:
    annot = json.load(f)  # ← Funciona porque annotations/ apunta al nivel correcto
```
