# 📚 Documentación: Testing ConvNeXtPose Modelos L y M

## Índice de Documentación

Esta es la documentación completa para testear los modelos **L (Large)** y **M (Medium)** de ConvNeXtPose en el dataset Human3.6M usando el Protocolo 2.

---

## 📖 Guías Disponibles

### 1. 🚀 RESUMEN_EJECUTIVO.md
**Para**: Comenzar rápidamente  
**Tiempo de lectura**: 5 minutos  
**Contenido**:
- TL;DR con pasos principales
- Comandos de una línea
- Diferencias clave vs configuración actual
- Tips importantes

👉 **Usa esto si**: Necesitas comenzar YA y tienes experiencia previa

---

### 2. ✅ CHECKLIST_TESTING.md
**Para**: Seguimiento paso a paso interactivo  
**Tiempo de lectura**: 10 minutos  
**Contenido**:
- 10 fases con checkboxes
- Comandos específicos para cada paso
- Verificación de éxito
- Troubleshooting rápido

👉 **Usa esto si**: Prefieres una lista de tareas clara y organizada

---

### 3. 📘 GUIA_TESTING_MODELOS_L_M.md
**Para**: Guía completa y detallada  
**Tiempo de lectura**: 30 minutos  
**Contenido**:
- 13 pasos detallados con explicaciones
- Scripts de ejemplo completos
- Troubleshooting extensivo
- Análisis avanzado y benchmarking
- Configuraciones arquitectónicas detalladas

👉 **Usa esto si**: Quieres entender TODO el proceso en profundidad

---

## 🛠️ Scripts Implementados

### 1. main/config_variants.py
**Propósito**: Definir configuraciones de arquitectura  
**Funciones principales**:
- `get_model_config(variant)`: Obtiene (depths, dims) para una variante
- `print_model_info(variant)`: Muestra información detallada
- `compare_variants()`: Tabla comparativa de todas las variantes
- `get_recommended_batch_size()`: Recomienda batch size según GPU

**Ejemplo de uso**:
```python
from config_variants import get_model_config, print_model_info

# Obtener configuración
depths, dims = get_model_config('M')

# Ver info
print_model_info('M')
```

---

### 2. main/test_variants.py
**Propósito**: Testing de modelos con soporte para variantes  
**Argumentos principales**:
- `--variant`: Variante del modelo (XS, S, M, L)
- `--epoch`: Número de epoch del checkpoint
- `--gpu`: ID de GPU a usar
- `--protocol`: Protocolo de evaluación (1 o 2)
- `--flip_test`: Habilitar flip augmentation
- `--use_gt_bbox`: Usar GT bounding box

**Ejemplo de uso**:
```bash
python test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test
```

---

### 3. main/compare_variants.py
**Propósito**: Comparar resultados entre variantes  
**Argumentos principales**:
- `--variants`: Variantes a comparar
- `--epoch`: Epoch a analizar
- `--plot`: Generar gráficos
- `--save_report`: Guardar reporte markdown

**Ejemplo de uso**:
```bash
python compare_variants.py --variants M L --epoch 70 --plot --save_report
```

---

### 4. quick_start.sh
**Propósito**: Script bash interactivo para setup y testing  
**Funciones**:
- Verificación automática de entorno
- Detección de GPU y CUDA
- Verificación de estructura de datos
- Menú interactivo de comandos
- Ejecución guiada

**Ejemplo de uso**:
```bash
bash quick_start.sh
```

---

## 🗺️ Flujo de Trabajo Recomendado

### Para Principiantes
```
1. RESUMEN_EJECUTIVO.md (entender el objetivo)
   ↓
2. bash quick_start.sh (verificar entorno)
   ↓
3. CHECKLIST_TESTING.md (seguir paso a paso)
   ↓
4. Ejecutar testing con test_variants.py
   ↓
5. Comparar con compare_variants.py
```

### Para Usuarios Avanzados
```
1. RESUMEN_EJECUTIVO.md (comandos rápidos)
   ↓
2. test_variants.py directo
   ↓
3. compare_variants.py con --plot --save_report
   ↓
4. GUIA_TESTING_MODELOS_L_M.md (troubleshooting si es necesario)
```

### Para Investigación Profunda
```
1. GUIA_TESTING_MODELOS_L_M.md (leer completa)
   ↓
2. Entender configuraciones en config_variants.py
   ↓
3. Experimentar con diferentes configuraciones
   ↓
4. Benchmark avanzado (Paso 12 de la guía)
   ↓
5. Análisis comparativo detallado
```

---

## 🎯 Comandos Esenciales

### Setup Inicial
```bash
# 1. Navegar al directorio
cd /home/user/convnextpose_esteban/ConvNeXtPose

# 2. Verificar entorno
bash quick_start.sh

# 3. O manual: instalar dependencias
pip install torch torchvision timm pycocotools opencv-python tqdm numpy matplotlib
```

### Testing
```bash
# Modelo M
cd main
python test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox

# Modelo L
python test_variants.py --variant L --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox
```

### Comparación
```bash
cd main
python compare_variants.py --variants M L --epoch 70 --protocol 2 --plot --save_report
```

---

## 📊 Configuraciones de Modelos

### Resumen Rápido

| Variante | Depths        | Dims                    | Params | GFLOPs | MPJPE (mm) |
|----------|---------------|-------------------------|--------|--------|------------|
| XS       | [3,3,9,3]     | [48,96,192,384]        | 22M    | 4.5    | ~52        |
| S        | [3,3,27,3]    | [96,192,384,768]       | 50M    | 8.7    | ~48        |
| **M**    | [3,3,27,3]    | [128,256,512,1024]     | 89M    | 15.4   | **44.6**   |
| **L**    | [3,3,27,3]    | [192,384,768,1536]     | 198M   | 34.4   | **42.3**   |

### Diferencias Clave M vs L

**Modelo M (Medium)**:
- Depths: `[3, 3, 27, 3]`
- Dims: `[128, 256, 512, 1024]`
- Balance óptimo precisión/velocidad
- Recomendado para prototipado

**Modelo L (Large)**:
- Depths: `[3, 3, 27, 3]`
- Dims: `[192, 384, 768, 1536]`
- Máxima precisión
- Recomendado para resultados finales

---

## 🔍 Estructura de Archivos

### Archivos de Documentación
```
ConvNeXtPose/
├── README_TESTING.md              # ← Este archivo (índice)
├── RESUMEN_EJECUTIVO.md           # Vista rápida
├── CHECKLIST_TESTING.md           # Checklist paso a paso
├── GUIA_TESTING_MODELOS_L_M.md    # Guía completa
└── quick_start.sh                 # Script bash interactivo
```

### Scripts de Testing
```
ConvNeXtPose/main/
├── config_variants.py             # Configuraciones de variantes
├── test_variants.py               # Testing adaptado
└── compare_variants.py            # Comparación de resultados
```

### Archivos de Entrada
```
ConvNeXtPose/
├── data/Human36M/                 # Dataset
│   ├── images/
│   ├── annotations/
│   └── bbox_root/
└── output/model_dump/             # Checkpoints
    └── snapshot_70.pth.tar
```

### Archivos de Salida
```
ConvNeXtPose/output/
├── result/
│   ├── bbox_root_pose_human36m_output.json  # Predicciones
│   ├── results_M_epoch70.json               # Métricas M
│   ├── results_L_epoch70.json               # Métricas L
│   ├── comparison_plot.png                  # Gráfico comparativo
│   └── comparison_report.md                 # Reporte
└── log/                                      # Logs de ejecución
```

---

## ⚙️ Configuración del Protocolo 2

### Diferencias Protocol 1 vs 2

**Protocol 1 (PA-MPJPE)**:
- Training: S1, S5, S6, S7, S8, S9
- Testing: **S11**
- Métrica: PA-MPJPE (con alineación Procrustes)
- Más fácil (~2-3mm mejor)

**Protocol 2 (MPJPE)** ← **Usamos este**:
- Training: S1, S5, S6, S7, S8
- Testing: **S9, S11**
- Métrica: MPJPE (sin alineación)
- Más estricto (posición absoluta)

### Verificar Protocolo

```bash
grep "self.protocol" data/Human36M/Human36M.py
# Debe mostrar: self.protocol = 2
```

Si muestra `self.protocol = 1`, cambiar a 2:
```python
# data/Human36M/Human36M.py línea ~30
self.protocol = 2
```

---

## 🎓 Resultados Esperados

### Protocol 2 (MPJPE)

**Modelo M**:
- Total: **44.6 mm**
- Rango aceptable: 43-46 mm
- Mejor acción: Directions (~39.8 mm)
- Peor acción: Greeting (~46.9 mm)

**Modelo L**:
- Total: **42.3 mm**
- Rango aceptable: 41-44 mm
- Mejor acción: Directions (~37.9 mm)
- Peor acción: Greeting (~44.5 mm)

**Mejora L vs M**: ~2.3 mm (~5.2%)

---

## 🐛 Troubleshooting Rápido

### Problema: Out of Memory
**Solución**: Reducir batch size
```bash
python test_variants.py --variant L --gpu 0 --epoch 70 --batch_size 8
```

### Problema: Modelo no carga
**Solución**: Verificar nombre del checkpoint
```bash
# Debe ser: snapshot_70.pth.tar
ls -la output/model_dump/
mv output/model_dump/OLD_NAME.pth output/model_dump/snapshot_70.pth.tar
```

### Problema: Dataset no encontrado
**Solución**: Verificar estructura
```bash
ls data/Human36M/images/ | wc -l  # Debe ser > 0
ls data/Human36M/annotations/*.json
```

### Problema: Resultados muy diferentes al paper
**Causas posibles**:
1. Protocolo incorrecto (verificar que sea 2)
2. Checkpoint no corresponde a la variante
3. Configuración de bbox incorrecta
4. Datos incompletos o corruptos

---

## 📞 Próximos Pasos

### 1. Preparación
- [ ] Leer RESUMEN_EJECUTIVO.md
- [ ] Ejecutar `bash quick_start.sh`
- [ ] Verificar que todo está listo

### 2. Testing
- [ ] Testear Modelo M
- [ ] Testear Modelo L
- [ ] Verificar resultados

### 3. Análisis
- [ ] Comparar M vs L
- [ ] Generar gráficos
- [ ] Crear reporte

### 4. Validación
- [ ] Verificar que MPJPE está en rango esperado
- [ ] Confirmar que L > M en precisión
- [ ] Documentar hallazgos

---

## 🌟 Recursos Adicionales

### Paper Original
- **Título**: ConvNeXtPose: A Fast Accurate Method for 3D Human Pose Estimation
- **Journal**: IEEE Access 2023
- **Link**: https://ieeexplore.ieee.org/document/10288440

### Repositorio Oficial
- **GitHub**: https://github.com/medialab-ku/ConvNeXtPose
- **Modelos**: https://drive.google.com/drive/folders/12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI

### Dataset Human3.6M
- **Website**: http://vision.imar.ro/human3.6m/
- **Descripción**: Dataset de pose 3D más grande
- **Sujetos**: 11 actores, 15 acciones

---

## 📧 Soporte

Si tienes problemas:

1. **Primero**: Consulta la sección de troubleshooting en cada guía
2. **Segundo**: Ejecuta `bash quick_start.sh` para diagnóstico
3. **Tercero**: Revisa logs en `output/log/`
4. **Cuarto**: Verifica que seguiste todos los pasos del checklist

---

## ✅ Checklist de Inicio Rápido

Antes de empezar, verifica:

- [ ] Python 3.8+ instalado
- [ ] PyTorch con CUDA funcionando
- [ ] GPU con ≥8GB VRAM (o ≥4GB para M solo)
- [ ] Dataset Human3.6M descargado
- [ ] Modelos pre-entrenados descargados
- [ ] Scripts de testing creados (config_variants.py, test_variants.py, compare_variants.py)
- [ ] Protocolo configurado a 2

Si todos están ✅, estás listo para comenzar:

```bash
bash quick_start.sh
```

---

**Última actualización**: Octubre 2025  
**Versión**: 1.0  
**Mantenedor**: Testing adaptado para modelos L y M

---

## 🚀 ¡Comienza Ahora!

### Opción 1: Guiado (Recomendado)
```bash
bash quick_start.sh
```

### Opción 2: Manual Rápido
```bash
cd main
python test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test
python test_variants.py --variant L --gpu 0 --epoch 70 --protocol 2 --flip_test
python compare_variants.py --variants M L --epoch 70 --plot --save_report
```

### Opción 3: Explorar Documentación
1. Lee `RESUMEN_EJECUTIVO.md` para vista general
2. Sigue `CHECKLIST_TESTING.md` paso a paso
3. Consulta `GUIA_TESTING_MODELOS_L_M.md` para detalles

---

**¡Éxito en tu testing!** 🎉
