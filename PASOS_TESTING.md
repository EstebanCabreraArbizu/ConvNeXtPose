# 📋 LISTA DE PASOS: Testing ConvNeXtPose L y M en Human3.6M Protocol 2

## Objetivo
Testear modelos **L (Large)** y **M (Medium)** de ConvNeXtPose en Human3.6M usando **Protocol 2 (MPJPE)** para obtener los mismos resultados del paper.

**Resultados esperados**:
- Modelo M: MPJPE ≈ 44.6 mm
- Modelo L: MPJPE ≈ 42.3 mm

---

## PASO 1: Preparación del Entorno

### 1.1 Verificar Python y CUDA
```bash
python --version  # Debe ser 3.8+
python -c "import torch; print(torch.cuda.is_available())"  # Debe ser True
nvidia-smi  # Verificar GPU disponible
```

### 1.2 Instalar Dependencias
```bash
cd /home/user/convnextpose_esteban/ConvNeXtPose
pip install torch torchvision timm pycocotools opencv-python tqdm numpy matplotlib
```

### 1.3 Crear Directorios
```bash
mkdir -p output/model_dump output/result output/log
```

---

## PASO 2: Verificar Dataset Human3.6M

### 2.1 Verificar Estructura
```bash
# Verificar que existen los directorios
ls -la data/Human36M/images/
ls -la data/Human36M/annotations/
ls -la data/Human36M/bbox_root/
```

### 2.2 Verificar Bbox Root para Protocol 2
```bash
# Debe existir este archivo para Protocol 2 (S9, S11)
ls -la "data/Human36M/bbox_root/Subject 9,11 (trained on subject 1,5,6,7,8)/bbox_root_human36m_output.json"
```

### 2.3 Configurar Protocol 2
```bash
# Verificar que usa Protocol 2
grep "self.protocol" data/Human36M/Human36M.py
# Debe mostrar: self.protocol = 2
```

**Si muestra `self.protocol = 1`, editar el archivo**:
```python
# En data/Human36M/Human36M.py, línea ~30
self.protocol = 2  # Cambiar de 1 a 2
```

---

## PASO 3: Descargar Modelos Pre-entrenados

### 3.1 Obtener Checkpoints
1. Ir a: https://drive.google.com/drive/folders/12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI
2. Descargar modelos M y L
3. Guardar en: `output/model_dump/`

### 3.2 Renombrar Checkpoints
```bash
# Los checkpoints deben tener el formato: snapshot_<epoch>.pth.tar
# Ejemplo: snapshot_70.pth.tar

# Si el nombre es diferente, renombrar:
cd output/model_dump/
mv MODEL_M.pth.tar snapshot_70.pth.tar  # Ajustar según nombre real
```

**Nota**: Si tienes modelos M y L en archivos separados, testear uno a la vez, moviendo el checkpoint correspondiente.

---

## PASO 4: Instalar Scripts de Testing

Los siguientes archivos ya deberían estar creados en `main/`:

### 4.1 Verificar Scripts
```bash
ls -lh main/config_variants.py
ls -lh main/test_variants.py
ls -lh main/compare_variants.py
```

Si no existen, fueron creados en los pasos anteriores de esta conversación.

### 4.2 Test de Configuración
```bash
cd main
python config_variants.py
```

**Esperado**: Debe mostrar información de variantes XS, S, M, L sin errores.

---

## PASO 5: Testing Modelo M (Medium)

### 5.1 Testing Básico
```bash
cd main
python test_variants.py \
  --variant M \
  --gpu 0 \
  --epoch 70 \
  --protocol 2
```

### 5.2 Testing con Flip Augmentation (Recomendado)
```bash
python test_variants.py \
  --variant M \
  --gpu 0 \
  --epoch 70 \
  --protocol 2 \
  --flip_test \
  --use_gt_bbox
```

### 5.3 Verificar Resultados
```bash
# Ver archivo de resultados
ls -lh ../output/result/results_M_epoch70.json

# Ver métricas rápidamente
cat ../output/result/results_M_epoch70.json | grep -A5 "evaluation_result"
```

**Resultado esperado**: MPJPE ≈ 44.6 mm (rango aceptable: 43-46 mm)

---

## PASO 6: Testing Modelo L (Large)

### 6.1 Cambiar Checkpoint (si es necesario)
```bash
# Si tienes checkpoint separado para L
cd ../output/model_dump/
mv snapshot_70.pth.tar snapshot_70_M.pth.tar.backup
mv MODEL_L.pth.tar snapshot_70.pth.tar
```

### 6.2 Testing Básico
```bash
cd ../../main
python test_variants.py \
  --variant L \
  --gpu 0 \
  --epoch 70 \
  --protocol 2
```

### 6.3 Testing con Flip Augmentation (Recomendado)
```bash
python test_variants.py \
  --variant L \
  --gpu 0 \
  --epoch 70 \
  --protocol 2 \
  --flip_test \
  --use_gt_bbox
```

### 6.4 Verificar Resultados
```bash
# Ver archivo de resultados
ls -lh ../output/result/results_L_epoch70.json

# Ver métricas rápidamente
cat ../output/result/results_L_epoch70.json | grep -A5 "evaluation_result"
```

**Resultado esperado**: MPJPE ≈ 42.3 mm (rango aceptable: 41-44 mm)

---

## PASO 7: Comparar Resultados

### 7.1 Comparación Básica
```bash
cd main
python compare_variants.py \
  --variants M L \
  --epoch 70 \
  --protocol 2
```

### 7.2 Generar Gráficos
```bash
python compare_variants.py \
  --variants M L \
  --epoch 70 \
  --protocol 2 \
  --plot
```

### 7.3 Generar Reporte Completo
```bash
python compare_variants.py \
  --variants M L \
  --epoch 70 \
  --protocol 2 \
  --plot \
  --save_report
```

### 7.4 Ver Resultados
```bash
# Ver reporte markdown
cat ../output/result/comparison_report.md

# Ver gráfico (copiar a Windows si es WSL)
ls -lh ../output/result/comparison_plot.png
```

---

## PASO 8: Verificación de Éxito

### 8.1 Criterios de Éxito

✅ **Testing Exitoso Si**:
- [x] Modelo M: MPJPE entre 43-46 mm
- [x] Modelo L: MPJPE entre 41-44 mm
- [x] Modelo L es ~2-3mm más preciso que M
- [x] Sin errores durante ejecución
- [x] Archivos de resultado generados correctamente

### 8.2 Verificar Archivos Generados
```bash
# Predicciones
ls -lh output/result/bbox_root_pose_human36m_output.json

# Métricas
ls -lh output/result/results_M_epoch70.json
ls -lh output/result/results_L_epoch70.json

# Comparación
ls -lh output/result/comparison_report.md
ls -lh output/result/comparison_plot.png
```

---

## PASO 9: Análisis de Resultados

### 9.1 Tabla Comparativa Esperada

| Métrica          | Modelo M | Modelo L | Mejora  |
|------------------|----------|----------|---------|
| MPJPE Total (mm) | 44.6     | 42.3     | 2.3 mm  |
| Params (M)       | 88.6     | 197.8    | -       |
| GFLOPs           | 15.4     | 34.4     | -       |

### 9.2 Por Acción (Ejemplos)

| Acción     | Modelo M (mm) | Modelo L (mm) |
|------------|---------------|---------------|
| Directions | 39.8          | 37.9          |
| Discussion | 43.6          | 41.4          |
| Eating     | 41.2          | 39.1          |
| Greeting   | 46.9          | 44.5          |
| Walking    | 38.5          | 36.8          |

---

## PASO 10: Troubleshooting

### 10.1 Error: Out of Memory (OOM)

**Síntoma**: Error CUDA Out of Memory durante testing

**Solución**: Reducir batch size
```bash
python test_variants.py \
  --variant L \
  --gpu 0 \
  --epoch 70 \
  --batch_size 8  # O incluso 4
```

### 10.2 Error: Modelo No Carga

**Síntoma**: Error al cargar checkpoint

**Solución 1**: Verificar nombre del archivo
```bash
ls -la output/model_dump/
# Debe ser: snapshot_70.pth.tar (o el epoch que uses)
```

**Solución 2**: Verificar contenido del checkpoint
```bash
python -c "import torch; ckpt = torch.load('output/model_dump/snapshot_70.pth.tar', map_location='cpu'); print(ckpt.keys())"
```

### 10.3 Error: Dataset No Encontrado

**Síntoma**: Error al cargar imágenes o annotations

**Solución**: Verificar estructura completa
```bash
tree -L 3 data/Human36M/
# O si tree no está instalado:
find data/Human36M/ -maxdepth 3 -type d
```

### 10.4 Error: Dimensiones No Coinciden

**Síntoma**: Error "size mismatch" al cargar modelo

**Causa**: El checkpoint fue entrenado con una configuración diferente

**Solución**: Verificar que el checkpoint corresponde a la variante correcta (M o L)

### 10.5 Error: Protocolo Incorrecto

**Síntoma**: Los resultados no coinciden con el paper

**Solución**: Verificar y cambiar protocolo
```bash
# Verificar
grep "self.protocol" data/Human36M/Human36M.py

# Si es 1, cambiar a 2
sed -i 's/self.protocol = 1/self.protocol = 2/' data/Human36M/Human36M.py
```

---

## PASO 11: Documentación y Backup

### 11.1 Guardar Resultados
```bash
# Crear backup de resultados
mkdir -p backup_results_$(date +%Y%m%d)
cp -r output/result/* backup_results_$(date +%Y%m%d)/
```

### 11.2 Screenshots
- Capturar output del testing
- Guardar gráficos comparativos
- Documentar métricas obtenidas

### 11.3 Notas
Documentar:
- Diferencias con paper (si las hay)
- Configuraciones usadas
- Hardware utilizado
- Tiempo de ejecución

---

## RESUMEN DE COMANDOS

### Setup Completo
```bash
cd /home/user/convnextpose_esteban/ConvNeXtPose
pip install torch torchvision timm pycocotools opencv-python tqdm numpy matplotlib
mkdir -p output/model_dump output/result
```

### Testing M + L + Comparación
```bash
cd main

# Modelo M
python test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox

# Modelo L (cambiar checkpoint si es necesario)
python test_variants.py --variant L --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox

# Comparar
python compare_variants.py --variants M L --epoch 70 --protocol 2 --plot --save_report
```

### Verificación Rápida
```bash
# GPU
nvidia-smi

# Dataset
ls data/Human36M/images/ | wc -l

# Checkpoints
ls output/model_dump/*.pth*

# Resultados
ls output/result/
```

---

## CHECKLIST FINAL

Antes de considerar el testing completo, verifica:

- [ ] Python 3.8+ y PyTorch instalados
- [ ] GPU NVIDIA con CUDA disponible
- [ ] Dataset Human3.6M en lugar correcto
- [ ] Protocol configurado a 2
- [ ] Modelos pre-entrenados descargados
- [ ] Scripts de testing creados (config_variants.py, test_variants.py, compare_variants.py)
- [ ] Testing de Modelo M ejecutado sin errores
- [ ] Testing de Modelo L ejecutado sin errores
- [ ] MPJPE de M en rango 43-46 mm
- [ ] MPJPE de L en rango 41-44 mm
- [ ] Modelo L es más preciso que M
- [ ] Comparación generada con gráficos
- [ ] Reporte markdown creado
- [ ] Resultados documentados y respaldados

---

## PRÓXIMOS PASOS

Después de completar el testing:

1. **Analizar diferencias** entre resultados obtenidos y paper
2. **Documentar hallazgos** en un reporte final
3. **Explorar optimizaciones** adicionales si es necesario
4. **Considerar Protocol 1** (PA-MPJPE) para comparación adicional
5. **Benchmark de velocidad** para análisis de rendimiento

---

## RECURSOS ADICIONALES

- **Guía Completa**: `GUIA_TESTING_MODELOS_L_M.md`
- **Checklist Detallada**: `CHECKLIST_TESTING.md`
- **Resumen Ejecutivo**: `RESUMEN_EJECUTIVO.md`
- **Índice General**: `README_TESTING.md`
- **Script Interactivo**: `bash quick_start.sh`

---

**Última actualización**: Octubre 2025  
**Versión**: 1.0

---

## ¡COMIENZA AHORA!

```bash
# Opción 1: Script interactivo (recomendado)
bash quick_start.sh

# Opción 2: Comando directo
cd main && python test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test
```

**¡Éxito en tu testing!** 🚀
