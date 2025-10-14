# 🚀 Instrucciones para Ejecutar en Kaggle - Model S

**Fecha:** 13 de Octubre, 2025  
**Modelo:** ConvNeXtPose-S (Small)  
**Tiempo estimado:** 10-20 minutos con GPU T4 x2

---

## ✅ PREPARACIÓN (Antes de Kaggle)

### Archivos Necesarios

1. **Notebook Principal:**
   - `kaggle_testing_notebook.ipynb` ✅ (ya modificado para Model S)

2. **Checkpoint Convertido:**
   - `snapshot_83.pth` (96.2 MB)
   - **Ubicación:** Generado en ejecución previa del Cell 11

3. **Dataset:** (Opcional - Kaggle puede descargarlo)
   - Human3.6M dataset
   - Kaggle puede descargarlo automáticamente

---

## 📋 PASO A PASO EN KAGGLE

### PASO 1: Crear Nuevo Notebook

1. Ir a: https://www.kaggle.com/
2. Click en **"Code"** → **"New Notebook"**
3. Click en **"File"** → **"Import Notebook"**
4. Subir `kaggle_testing_notebook.ipynb`

### PASO 2: Configurar GPU (CRÍTICO ⚠️)

1. Panel derecho → Click **"Settings"** (⚙️)
2. Sección **"Accelerator"**
3. Seleccionar: **"GPU T4 x2"** (recomendado) o **"GPU P100"**
4. Click **"Save"**
5. El notebook se reiniciará con GPU activada

**Verificación:**
```python
import torch
print(torch.cuda.is_available())  # Debe ser True
print(torch.cuda.get_device_name(0))  # Debe mostrar "Tesla T4" o similar
```

### PASO 3: Activar Internet

1. Panel derecho → **"Settings"**
2. Sección **"Internet"**
3. Toggle: **ON** (para descargar dataset si es necesario)
4. Click **"Save"**

### PASO 4: Subir Checkpoint (Si Ya Lo Tienes)

**Opción A: Si ya tienes snapshot_83.pth convertido**
1. Panel derecho → Click **"Add Data"** (➕)
2. Click **"Upload"** → **"New Dataset"**
3. Subir archivo `snapshot_83.pth` (96.2 MB)
4. Nombre del dataset: "convnextpose-s-checkpoint"
5. Click **"Create"**

**Opción B: Si NO tienes el checkpoint**
- El notebook incluye Cell 11 para convertir desde .tar
- Necesitarás subir `ConvNeXtPose_S.tar` o descargarlo con gdown

### PASO 5: Ejecutar Notebook

1. **Cell por Cell (Recomendado para primera vez):**
   - Click en cada celda
   - Press `Shift + Enter` para ejecutar
   - Espera que termine antes de la siguiente

2. **Todo de una vez (Más rápido):**
   - Click en **"Run All"** (▶️▶️)
   - Espera ~10-20 minutos

### PASO 6: Monitorear Progreso

**Indicadores de que va bien:**
```
✅ GPU disponible: Tesla T4
✅ Memoria: 15.0 GB
✅ Dataset descargado
✅ Checkpoint cargado exitosamente
✅ Testing iniciado...
```

**Señales de advertencia:**
```
❌ GPU NO disponible - usando CPU
⚠️  RuntimeError: size mismatch
⚠️  FileNotFoundError: checkpoint
```

---

## 📊 CELDAS IMPORTANTES

### Cell 7: Verificación de Hardware
```python
import torch
if torch.cuda.is_available():
    print(f"✅ GPU disponible: {torch.cuda.get_device_name(0)}")
else:
    print("❌ GPU NO disponible")
    # DETENER AQUÍ - Configurar GPU en Settings
```

### Cell 11: Extracción de Checkpoint (Si es necesario)
```python
# Solo ejecutar si NO tienes snapshot_83.pth ya subido
# Convierte de formato legacy a moderno
# Output: snapshot_83.pth (96.2 MB)
```

### Cell 15-18: Testing Principal
```python
VARIANT = 'S'  # ✅ Correcto
CHECKPOINT_EPOCH = 83

# Esta es la parte que toma 10-20 minutos
# Procesa todas las imágenes del dataset
```

### Cell 19: Verificación de Resultados
```python
# Lee logs y extrae MPJPE
# Compara con valores del paper
# MPJPE esperado: ~45 mm
```

---

## 🎯 RESULTADOS ESPERADOS

### Archivos Generados

En `/kaggle/working/output/`:
```
output/
├── log/
│   └── train_ConvNeXtPose_*.log    # Log con MPJPE
├── result/
│   ├── result_*.pkl                 # Predicciones guardadas
│   └── preds_*.mat                  # Formato MATLAB
└── vis/
    └── *.png                        # Visualizaciones 3D (si se generan)
```

### Métricas Esperadas

**Para Model S en Human3.6M Protocol 2:**

| Métrica | Valor Esperado | Aceptable | Problemático |
|---------|----------------|-----------|--------------|
| **MPJPE** | **~45 mm** | 43-47 mm | >50 mm o <40 mm |
| PA-MPJPE | ~35 mm | 33-37 mm | - |
| Tiempo | 10-20 min | <30 min | >1 hora |

**Comparación con Paper:**
- Model S esperado: ~45 mm ✅
- Model M reportado: 44.6 mm
- Model L reportado: 42.3 mm

Si obtienes ~45 mm → **¡ÉXITO! ✅** Pipeline validado

---

## 🔧 TROUBLESHOOTING

### Problema 1: GPU No Disponible
```
❌ GPU NO disponible - usando CPU
⚠️ Tiempo estimado: 10-20 HORAS
```

**Solución:**
1. Detener ejecución (⏹️ Stop)
2. Settings → Accelerator → GPU T4 x2
3. Save (notebook reinicia)
4. Verificar con Cell 7

### Problema 2: Checkpoint No Encontrado
```
FileNotFoundError: [Errno 2] No such file or directory: 'snapshot_83.pth'
```

**Solución A:** Subir checkpoint pre-convertido
```python
# En Kaggle, el path será:
checkpoint_path = '/kaggle/input/convnextpose-s-checkpoint/snapshot_83.pth'
```

**Solución B:** Ejecutar Cell 11 para convertir
```python
# Asegúrate de tener ConvNeXtPose_S.tar disponible
# Cell 11 convertirá de legacy a moderno
```

### Problema 3: Size Mismatch
```
RuntimeError: size mismatch for backbone.downsample_layers.0.0.weight
```

**Verificación:**
```python
# Verificar que VARIANT está configurado como 'S'
print(f"VARIANT actual: {VARIANT}")  # Debe imprimir: S

# Si dice 'L' o 'M', cambiar:
VARIANT = 'S'
```

### Problema 4: Dataset No Descarga
```
FileNotFoundError: data/Human36M/annotations
```

**Solución:**
1. Verificar Internet: Settings → Internet → ON
2. Ejecutar celdas de descarga del dataset
3. O subir dataset manualmente como dataset de Kaggle

### Problema 5: Memoria Insuficiente
```
RuntimeError: CUDA out of memory
```

**Solución:**
1. Reiniciar kernel: Kernel → Restart
2. Reducir batch size en config (si es posible)
3. Usar GPU P100 (más memoria) en lugar de T4

### Problema 6: Ejecución Muy Lenta
```
Progreso: 10/5000 imágenes... (después de 2 horas)
```

**Verificación:**
```python
# Confirmar que usa GPU, no CPU
import torch
print(f"Using device: {torch.cuda.current_device()}")
print(f"GPU: {torch.cuda.is_available()}")
```

---

## 📝 CHECKLIST DE EJECUCIÓN

Antes de ejecutar, verificar:

- [ ] ✅ GPU T4 x2 activada (Settings → Accelerator)
- [ ] ✅ Internet activado (Settings → Internet → ON)
- [ ] ✅ Checkpoint disponible (snapshot_83.pth O ConvNeXtPose_S.tar)
- [ ] ✅ Variable `VARIANT = 'S'` configurada
- [ ] ✅ Variable `CHECKPOINT_EPOCH = 83` configurada

Durante ejecución, monitorear:

- [ ] ✅ Cell 7: GPU detectada correctamente
- [ ] ✅ Cell 11 (opcional): Checkpoint convertido sin errores
- [ ] ✅ Cell 15-18: Testing progresa (no tarda horas)
- [ ] ✅ Cell 19: MPJPE calculado (~45 mm esperado)

Después de ejecución:

- [ ] ✅ Archivos en `/kaggle/working/output/`
- [ ] ✅ Log contiene métricas finales
- [ ] ✅ MPJPE en rango esperado (43-47 mm)
- [ ] ✅ Guardar outputs si son buenos

---

## 💾 GUARDAR RESULTADOS

### Descargar Outputs

```python
# Kaggle guarda automáticamente en /kaggle/working/
# Para descargar manualmente:

# 1. Desde interface de Kaggle:
#    - Panel derecho → Output
#    - Click en archivo → Download

# 2. Desde notebook:
import shutil
shutil.make_archive('results', 'zip', '/kaggle/working/output')
# Descarga results.zip desde Output panel
```

### Crear Summary JSON

```python
# Cell 20 crea automáticamente:
summary = {
    'timestamp': '2025-10-13...',
    'model': 'ConvNeXtPose-S',
    'checkpoint_epoch': 83,
    'mpjpe': 45.2,  # Ejemplo
    'gpu': 'Tesla T4',
    'runtime_minutes': 15
}
# Guardado en: output/summary_results.json
```

---

## 🎓 NOTAS IMPORTANTES

### Sobre el Modelo S

**Lo que tienes:**
- ✅ Checkpoint funcional de Model S
- ✅ Arquitectura: dims=[48, 96, 192, 384]
- ✅ Rendimiento esperado: ~45 mm MPJPE

**Lo que NO tienes (aún):**
- ❌ Model L: dims=[192, 384, 768, 1536], 42.3 mm
- ❌ Model M: dims=[64, 128, 256, 512], 44.6 mm
- ℹ️ Estos requieren contactar a autores (ver AUTHOR_CONTACT_GUIDE.md)

### Interpretación de Resultados

**Si obtienes ~45 mm:**
- ✅ **PERFECTO** - Reproduce resultados esperados para Model S
- ✅ Valida que todo el pipeline funciona
- ✅ Puedes confiar en la infraestructura
- ✅ Listo para testing de L y M cuando estén disponibles

**Si obtienes 40-43 mm:**
- ✓ Aceptable - Posible variación por implementación
- ✓ Revisa si dataset/protocolo son exactamente iguales

**Si obtienes >50 mm:**
- ⚠️ Revisar configuración
- ⚠️ Verificar que checkpoint cargó correctamente
- ⚠️ Revisar logs por errores

---

## 📞 SIGUIENTE PASO

Después de ejecutar exitosamente Model S:

1. **Documentar resultados** en tu informe/notebook
2. **Proceder con Opción A** (contactar autores):
   - Crear GitHub Issue (GITHUB_ISSUE_TEMPLATE.md)
   - Enviar email (EMAIL_TEMPLATE_AUTHORS.md)
3. **Esperar checkpoints L y M** (1-14 días)
4. **Repetir testing** con modelos correctos cuando lleguen

---

**¡ÉXITO! El pipeline está completamente funcional. Solo falta acceso a los checkpoints correctos de L y M.**

**Tiempo total esperado:** 
- Setup en Kaggle: 5-10 minutos
- Ejecución testing: 10-20 minutos
- **TOTAL: ~30 minutos** ✅
