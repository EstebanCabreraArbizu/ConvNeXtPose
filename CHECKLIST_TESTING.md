# ✅ CHECKLIST: Testing ConvNeXtPose L y M en Human3.6M

## 📋 Resumen Ejecutivo

Esta checklist te guiará paso a paso para testear los modelos **L (Large)** y **M (Medium)** de ConvNeXtPose en el dataset **Human3.6M** usando el **Protocolo 2 (MPJPE)** para evaluar en los sujetos **S9 y S11**.

---

## 🎯 Objetivo

Obtener los mismos resultados del paper ConvNeXtPose:
- **Modelo M**: MPJPE ≈ 44.6 mm
- **Modelo L**: MPJPE ≈ 42.3 mm

---

## ✅ FASE 1: Preparación del Entorno

### 1.1 Verificar Dependencias

- [ ] Python 3.8+ instalado
  ```bash
  python --version
  ```

- [ ] PyTorch con CUDA instalado
  ```bash
  python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
  ```

- [ ] GPU NVIDIA con ≥8GB VRAM para modelo L (≥4GB para M)
  ```bash
  nvidia-smi
  ```

- [ ] Dependencias instaladas
  ```bash
  cd /home/user/convnextpose_esteban/ConvNeXtPose
  pip install torch torchvision timm pycocotools opencv-python tqdm numpy matplotlib
  ```

---

## ✅ FASE 2: Verificación de Datos

### 2.1 Dataset Human3.6M

- [ ] Directorio `data/Human36M/` existe
  ```bash
  ls -la data/Human36M/
  ```

- [ ] Subdirectorio `images/` existe con imágenes
  ```bash
  ls -la data/Human36M/images/ | head
  ```

- [ ] Subdirectorio `annotations/` existe con archivos JSON
  ```bash
  ls -la data/Human36M/annotations/
  ```

- [ ] Archivo bbox_root para Protocol 2 existe
  ```bash
  ls -la "data/Human36M/bbox_root/Subject 9,11 (trained on subject 1,5,6,7,8)/bbox_root_human36m_output.json"
  ```

### 2.2 Configuración del Protocolo

- [ ] Verificar que `data/Human36M/Human36M.py` tiene `self.protocol = 2`
  ```bash
  grep "self.protocol" data/Human36M/Human36M.py
  ```

---

## ✅ FASE 3: Modelos Pre-entrenados

### 3.1 Descargar Modelos

- [ ] Directorio `output/model_dump/` existe
  ```bash
  mkdir -p output/model_dump
  ```

- [ ] Modelo M descargado (snapshot_70.pth.tar o similar)
  ```bash
  # Descargar desde: https://drive.google.com/drive/folders/12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI
  ls -lh output/model_dump/*M*.pth*
  ```

- [ ] Modelo L descargado (snapshot_70.pth.tar o similar)
  ```bash
  ls -lh output/model_dump/*L*.pth*
  ```

- [ ] Checkpoints renombrados correctamente (si es necesario)
  ```bash
  # Formato esperado: snapshot_<epoch>.pth.tar
  # Ejemplo: snapshot_70.pth.tar
  ```

**Nota**: Si tienes modelos separados para M y L, deberás ejecutar el testing uno a la vez, moviendo el checkpoint correspondiente.

---

## ✅ FASE 4: Instalación de Scripts de Testing

### 4.1 Scripts Creados

- [ ] `main/config_variants.py` creado
  ```bash
  ls -lh main/config_variants.py
  ```

- [ ] `main/test_variants.py` creado
  ```bash
  ls -lh main/test_variants.py
  ```

- [ ] `main/compare_variants.py` creado
  ```bash
  ls -lh main/compare_variants.py
  ```

### 4.2 Verificar Funcionalidad

- [ ] Testar módulo de configuración
  ```bash
  cd main && python config_variants.py
  ```
  
  **Esperado**: Debe mostrar información de las variantes XS, S, M, L

---

## ✅ FASE 5: Ejecución del Testing

### 5.1 Testing Modelo M (Medium)

- [ ] Ejecutar testing básico
  ```bash
  cd main
  python test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2
  ```

- [ ] Ejecutar testing con flip augmentation (recomendado)
  ```bash
  python test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test
  ```

- [ ] Ejecutar con GT bbox (más preciso)
  ```bash
  python test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox
  ```

**Resultado esperado**: MPJPE ≈ 44.6 mm (±1mm es aceptable)

### 5.2 Testing Modelo L (Large)

- [ ] Ejecutar testing básico
  ```bash
  python test_variants.py --variant L --gpu 0 --epoch 70 --protocol 2
  ```

- [ ] Ejecutar testing con flip augmentation (recomendado)
  ```bash
  python test_variants.py --variant L --gpu 0 --epoch 70 --protocol 2 --flip_test
  ```

- [ ] Ejecutar con GT bbox (más preciso)
  ```bash
  python test_variants.py --variant L --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox
  ```

**Resultado esperado**: MPJPE ≈ 42.3 mm (±1mm es aceptable)

---

## ✅ FASE 6: Verificación de Resultados

### 6.1 Archivos Generados

- [ ] Archivo de predicciones generado
  ```bash
  ls -lh ../output/result/bbox_root_pose_human36m_output.json
  ```

- [ ] Archivo de resumen JSON generado
  ```bash
  ls -lh ../output/result/results_*_epoch70.json
  ```

### 6.2 Validar Métricas

- [ ] MPJPE total dentro del rango esperado (±1-2mm)

- [ ] Resultados por acción muestran distribución similar al paper

- [ ] No hay errores en los logs

---

## ✅ FASE 7: Comparación de Resultados

### 7.1 Comparar Variantes

- [ ] Ejecutar script de comparación
  ```bash
  cd main
  python compare_variants.py --variants M L --epoch 70 --protocol 2
  ```

- [ ] Generar gráficos comparativos
  ```bash
  python compare_variants.py --variants M L --epoch 70 --plot
  ```

- [ ] Generar reporte markdown
  ```bash
  python compare_variants.py --variants M L --epoch 70 --plot --save_report
  ```

### 7.2 Análisis de Resultados

- [ ] Modelo L es más preciso que M (~2-3mm de mejora)

- [ ] Diferencia con paper es ≤ 1mm (excelente) o ≤ 2mm (aceptable)

- [ ] Resultados consistentes entre ejecuciones

---

## ✅ FASE 8: Troubleshooting

### 8.1 Problemas Comunes

Si encuentras errores, verifica:

- [ ] **Out of Memory**: Reducir batch size
  ```bash
  python test_variants.py --variant L --gpu 0 --epoch 70 --batch_size 8
  ```

- [ ] **Modelo no carga**: Verificar nombre de checkpoint
  ```bash
  ls -la output/model_dump/
  # Debe ser: snapshot_70.pth.tar
  ```

- [ ] **Dataset no encontrado**: Verificar estructura de datos
  ```bash
  tree -L 3 data/Human36M/
  ```

- [ ] **Protocolo incorrecto**: Editar `data/Human36M/Human36M.py`
  ```python
  self.protocol = 2  # Línea ~30
  ```

- [ ] **Dimensiones no coinciden**: Verificar que el checkpoint corresponde a la variante

### 8.2 Logs y Debugging

- [ ] Revisar logs en `output/log/`
  ```bash
  ls -lh output/log/
  ```

- [ ] Verificar uso de GPU durante testing
  ```bash
  watch -n 1 nvidia-smi
  ```

---

## 📊 FASE 9: Resultados Esperados

### 9.1 Protocolo 2 (MPJPE en mm)

| Modelo | MPJPE Total | Rango Aceptable | Estado |
|--------|-------------|-----------------|--------|
| **M**  | 44.6        | 43.6 - 45.6     | ✅     |
| **L**  | 42.3        | 41.3 - 43.3     | ✅     |

### 9.2 Por Acción (ejemplos)

| Acción       | Modelo M | Modelo L |
|--------------|----------|----------|
| Directions   | ~39.8    | ~37.9    |
| Discussion   | ~43.6    | ~41.4    |
| Eating       | ~41.2    | ~39.1    |
| Greeting     | ~46.9    | ~44.5    |
| Walking      | ~38.5    | ~36.8    |

---

## 🎓 FASE 10: Documentación

### 10.1 Guardar Resultados

- [ ] Screenshots de resultados guardados

- [ ] Archivos JSON de métricas respaldados

- [ ] Reporte markdown generado

### 10.2 Comparación con Paper

- [ ] Tabla comparativa creada

- [ ] Gráficos generados

- [ ] Análisis de diferencias documentado

---

## 🚀 Comandos Rápidos de Referencia

### Testing Rápido

```bash
# Modelo M
cd main && python test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox

# Modelo L
cd main && python test_variants.py --variant L --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox
```

### Comparación Completa

```bash
cd main && python compare_variants.py --variants M L --epoch 70 --protocol 2 --plot --save_report
```

### Verificación Rápida

```bash
# Ver info de GPU
nvidia-smi

# Ver checkpoints disponibles
ls -lh output/model_dump/

# Ver resultados
ls -lh output/result/

# Test de configuración
cd main && python config_variants.py
```

---

## 📚 Referencias

- **Guía completa**: `GUIA_TESTING_MODELOS_L_M.md`
- **Paper**: IEEE Access 2023 - ConvNeXtPose
- **Repo oficial**: https://github.com/medialab-ku/ConvNeXtPose
- **Modelos pre-entrenados**: https://drive.google.com/drive/folders/12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI

---

## ✨ Checklist de Éxito

**Has completado exitosamente el testing si**:

- ✅ Ambos modelos (M y L) ejecutaron sin errores
- ✅ MPJPE obtenido está dentro de ±2mm del esperado
- ✅ Modelo L es más preciso que M
- ✅ Resultados son consistentes entre ejecuciones
- ✅ Archivos de resultado generados correctamente
- ✅ Comparación y reportes creados

---

## 🆘 Soporte

Si tienes problemas:

1. Revisa la sección **FASE 8: Troubleshooting** arriba
2. Consulta la guía completa: `GUIA_TESTING_MODELOS_L_M.md`
3. Verifica logs en `output/log/`
4. Ejecuta: `bash quick_start.sh` para diagnóstico automático

---

**Última actualización**: Octubre 2025  
**Versión**: 1.0
