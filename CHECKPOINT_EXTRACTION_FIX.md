# 🔧 Solución: Error "Is a directory" al cargar checkpoints

## ❌ Error Encontrado

```
IsADirectoryError: [Errno 21] Is a directory: 
'/kaggle/working/ConvNeXtPose/main/../output/model_dump/snapshot_83.pth'
```

## 🔍 Causa Raíz

Los archivos `.tar` de ConvNeXtPose son **archivos ZIP con estructura anidada**, no archivos tar convencionales:

```
ConvNeXtPose_L.tar (archivo zip)
└── snapshot_83.pth/          ← ❌ Es un DIRECTORIO
    ├── data.pkl
    ├── version
    └── data/                 ← Carpeta con tensores
        ├── 0                 ← Archivos binarios individuales
        ├── 1
        ├── 2
        └── ...
```

Cuando extraes con `zipfile.ZipFile()`, crea directorios en lugar de un archivo `.pth` único.

## ✅ Solución

### Opción 1: Script de Extracción Mejorado (Recomendado)

El notebook actualizado incluye una función que:
1. Extrae el `.tar` a un directorio temporal
2. Busca recursivamente el archivo `.pth` real dentro de la estructura
3. Copia solo el archivo `.pth` a `output/model_dump/`
4. Limpia temporales

```python
def extract_checkpoint(tar_path, model_name):
    """Extrae checkpoint desde .tar y organiza la estructura correctamente"""
    temp_dir = f'output/model_dump/temp_{model_name}'
    os.makedirs(temp_dir, exist_ok=True)
    
    with zipfile.ZipFile(tar_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Buscar el archivo .pth dentro
    found_checkpoints = []
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.startswith('snapshot_') and file.endswith('.pth'):
                found_checkpoints.append(os.path.join(root, file))
    
    # Copiar al destino final
    for ckpt_path in found_checkpoints:
        dest_path = os.path.join('output/model_dump', os.path.basename(ckpt_path))
        shutil.copy2(ckpt_path, dest_path)
    
    # Limpiar temporal
    shutil.rmtree(temp_dir)
```

### Opción 2: Comandos Manuales en Kaggle

Si ya extrajiste incorrectamente, puedes corregirlo:

```bash
# 1. Limpiar extracción incorrecta
!rm -rf output/model_dump/snapshot_*

# 2. Crear directorio limpio
!mkdir -p output/model_dump

# 3. Extraer a temporal y buscar .pth
!unzip -q /kaggle/input/.../ConvNeXtPose_L.tar -d /tmp/extract_L

# 4. Buscar y copiar archivo .pth real
!find /tmp/extract_L -name "*.pth" -type f -exec cp {} output/model_dump/ \;

# 5. Verificar
!ls -lh output/model_dump/
```

### Opción 3: Usar torch.load directamente con la estructura

Si no puedes reorganizar, modifica `base.py` para cargar desde la estructura anidada:

```python
# En common/base.py, línea ~205
if os.path.isdir(model_path):
    # Si es directorio, buscar dentro
    actual_pth = os.path.join(model_path, 'data.pkl')
    if os.path.exists(actual_pth):
        ckpt = torch.load(actual_pth)
else:
    ckpt = torch.load(model_path)
```

**⚠️ No recomendado**: Requiere modificar código del proyecto.

## 🎯 Verificación Post-Extracción

Ejecuta esto para confirmar que los checkpoints están correctos:

```python
import os
import glob

checkpoints = [f for f in glob.glob('output/model_dump/snapshot_*.pth') 
               if os.path.isfile(f)]  # Solo archivos

if checkpoints:
    for ckpt in checkpoints:
        size_mb = os.path.getsize(ckpt) / (1024 * 1024)
        print(f"✓ {os.path.basename(ckpt)}: {size_mb:.1f} MB")
        
        # Verificar que es un archivo válido de PyTorch
        import torch
        try:
            state = torch.load(ckpt, map_location='cpu')
            print(f"  ✓ Checkpoint válido ({len(state)} keys)")
        except Exception as e:
            print(f"  ❌ Error: {e}")
else:
    print("❌ No se encontraron archivos .pth válidos")
```

**Salida esperada:**
```
✓ snapshot_83.pth: 1234.5 MB
  ✓ Checkpoint válido (245 keys)
```

## 📝 Estructura Correcta Final

```
output/model_dump/
├── snapshot_83.pth          ← ✅ Archivo (no directorio)
└── snapshot_70.pth          ← ✅ Archivo (si tienes modelo M)

NO debe existir:
❌ output/model_dump/snapshot_83.pth/ (directorio)
```

## 🔄 Si Sigues Teniendo Problemas

1. **Limpiar todo y empezar de nuevo:**
   ```bash
   !rm -rf output/model_dump/*
   ```

2. **Re-ejecutar celda de extracción mejorada** del notebook

3. **Verificar con:**
   ```python
   !ls -lh output/model_dump/
   # Debe mostrar archivos, no directorios (sin "/" al final)
   ```

4. **Probar carga manual:**
   ```python
   import torch
   ckpt = torch.load('output/model_dump/snapshot_83.pth', map_location='cpu')
   print(f"Keys: {list(ckpt.keys())}")
   ```

## 🚀 Testing Después de la Corrección

Una vez que los checkpoints estén correctamente extraídos como **archivos**:

```python
%cd main
!python test.py --gpu 0 --epochs 83 --variant L
```

Debería funcionar sin errores de `IsADirectoryError`.

## 📚 Referencia

- **Notebook actualizado**: `kaggle_testing_notebook.ipynb` (celda 10)
- **Función**: `extract_checkpoint()` con búsqueda recursiva
- **Verificación**: Celda 11 detecta directorios vs archivos
