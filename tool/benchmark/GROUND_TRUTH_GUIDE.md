# Ground Truth en ConvNeXtPose: Guía Completa

## 🎯 ¿Qué es Ground Truth?

**Ground Truth (GT)** = Poses 3D reales capturadas con Motion Capture (MoCap)

```
┌─────────────────────────────────────────────────────────────┐
│  Ground Truth: La "verdad absoluta"                         │
│  - Capturado con sistemas MoCap profesionales               │
│  - Precisión submilimétrica                                 │
│  - Sirve como referencia para evaluar modelos               │
└─────────────────────────────────────────────────────────────┘
                              ↓
         pred = model(imagen)
         error = ||pred - GT||  ← MPJPE
```

## 📦 ¿De Dónde Viene el Ground Truth?

### En Human3.6M:

```
data/Human36M/
├── annotations/
│   ├── Human36M_subject9_joint_3d.json   ← GROUND TRUTH (MoCap)
│   ├── Human36M_subject9_camera.json     ← Parámetros de cámara
│   ├── Human36M_subject9_data.json       ← Anotaciones COCO
│   ├── Human36M_subject11_joint_3d.json  ← GROUND TRUTH (MoCap)
│   └── ...
└── images/
    └── s_*.jpg                            ← Fotos del dataset
```

**Formato del JSON:**
```json
{
  "1": {              // action_idx
    "1": {            // subaction_idx
      "12345": [      // frame_idx
        [x, y, z],    // Pelvis (joint 0)
        [x, y, z],    // R_Hip (joint 1)
        ...           // 17 joints total
      ]
    }
  }
}
```

## ✅ El Repo Ya Carga el GT Automáticamente

**NO necesitas crear archivos NPZ manualmente.** El código ya lo hace:

```python
# En data/Human36M/Human36M.py:

class Human36M:
    def load_data(self):
        # 1. Carga GT desde JSON
        with open('Human36M_subject9_joint_3d.json') as f:
            joints[str(subject)] = json.load(f)  # ← GT aquí
        
        # 2. Para cada frame, extrae GT
        joint_world = np.array(joints[str(subject)][action][subaction][frame])
        joint_cam = world2cam(joint_world, R, t)  # Transforma a coords cámara
        
        # 3. Guarda en self.data
        data.append({
            'joint_cam': joint_cam,  # ← GT en (17, 3) mm
            'img_path': ...,
            ...
        })
        
        return data

    def evaluate(self, preds, result_dir):
        for n in range(len(preds)):
            gt_3d_kpt = self.data[n]['joint_cam']  # ← GT cargado automáticamente
            pred_3d_kpt = preds[n]
            
            # Calcular error
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))  # MPJPE
```

## 🔄 Flujo Completo

### Durante ENTRENAMIENTO:
```python
dataset = Human36M(data_split='train')
for sample in dataset:
    image = load_image(sample['img_path'])
    gt_3d = sample['joint_cam']  # ← GT del JSON
    
    pred = model(image)
    loss = mse(pred, gt_3d)  # Entrenar contra GT
    loss.backward()
```

### Durante TESTING:
```python
dataset = Human36M(data_split='test')  # S9, S11, stride=64
preds = []

for sample in dataset:
    image = load_image(sample['img_path'])
    pred = model(image)
    preds.append(pred)

# Evaluar contra GT (ya cargado en dataset.data)
mpjpe = dataset.evaluate(preds, output_dir)
```

### Durante BENCHMARK:
```python
# El benchmark runner usa el mismo flujo:
from tool.benchmark.runner_params import prep_data_from_dataset

# Carga GT automáticamente desde JSON
gt_3d, meta = prep_data_from_dataset("Human3.6M", protocol=2)
# gt_3d: (N, 17, 3) array en mm
# meta: info del dataset

# Compara múltiples modelos contra el mismo GT
for model_name in ['rootnet', 'convnextpose', 'integral_pose']:
    pred = model.infer(images)
    mpjpe = compute_mpjpe(pred, gt_3d)  # ← Comparación contra GT
```

## ❓ Preguntas Frecuentes

### Q: ¿Necesito crear un NPZ con ground truth?
**A:** ❌ No. El GT ya está en los archivos JSON y se carga automáticamente.

### Q: ¿Qué es el archivo `all_3d_poses.npz` en el repo?
**A:** Son **predicciones** del modelo ConvNeXtPose sobre un video. NO es ground truth.

```python
# all_3d_poses.npz contiene:
{
  'poses': (123, 18, 3),  # Predicciones del modelo (no GT)
  'depths': (123,),        # Profundidades predichas
  'frames': (123,)         # Números de frame del video
}
```

### Q: ¿Cuándo necesito ground truth?
**A:** 
- ✅ Entrenar el modelo → necesitas GT
- ✅ Evaluar en dataset → necesitas GT
- ✅ Benchmark multi-modelo → necesitas GT
- ❌ Inferencia en video nuevo → NO necesitas GT
- ❌ Demo visual → NO necesitas GT

### Q: ¿Se borra el GT después de testing?
**A:** ❌ No. El GT es parte permanente del dataset (archivos JSON).

Lo que SÍ se puede borrar:
```
output/result/bbox_root_pose_human36m_output.json  ← Predicciones
output/vis/                                        ← Visualizaciones
output/log/                                        ← Logs
```

### Q: ¿Dónde están los archivos JSON con GT?
**A:** Debes descargar el dataset Human3.6M completo de:
- http://vision.imar.ro/human3.6m/
- Seguir las instrucciones en `README.md` del repo

### Q: ¿Puedo usar solo imágenes sin GT?
**A:** Sí, pero **solo para inferencia** (demo/visualización). No podrás calcular MPJPE.

## 🛠️ Herramientas Incluidas

### 1. Verificar si tienes los archivos JSON:
```bash
python3 tool/benchmark/extract_gt.py --verify
```

Salida esperada si tienes el dataset:
```
✓ Found annotation directory: data/Human36M/annotations
✓ Human36M_subject9_joint_3d.json    (XX.XX MB) - Ground Truth 3D
✓ Human36M_subject11_joint_3d.json   (XX.XX MB) - Ground Truth 3D
```

### 2. Extraer estadísticas del GT (opcional):
```bash
python3 tool/benchmark/extract_gt.py --protocol 2
```

Muestra:
- Número de samples
- Shape del GT: (N, 17, 3)
- Rango de valores (mm)
- Estadísticas por joint

### 3. Guardar GT en NPZ (opcional, no recomendado):
```bash
python3 tool/benchmark/extract_gt.py --protocol 2 --save --out gt_p2.npz
```

**Nota:** Esto es opcional. El benchmark carga GT directamente del JSON.

## 📊 Para Benchmark en Kaggle

### Opción A: Dataset Completo (Recomendado)
```python
# 1. Sube Human3.6M completo como Kaggle Dataset
#    Incluye: images/ y annotations/ (con los JSON de GT)

# 2. En el notebook:
from tool.benchmark.runner_params import prep_data_from_dataset

gt_3d, meta = prep_data_from_dataset("Human3.6M", protocol=2)
# ↑ Carga GT automáticamente desde los JSON
```

### Opción B: Subset Pre-procesado (Más Rápido)
```python
# 1. Local: extrae subset con GT
python3 tool/benchmark/extract_gt.py --protocol 2 --save --out gt_h36m_p2_subset.npz

# 2. Sube solo el NPZ a Kaggle Dataset (mucho más ligero)

# 3. En notebook:
from tool.benchmark.runner_params import prep_data_from_npz

gt_3d, meta = prep_data_from_npz('gt_h36m_p2_subset.npz')
```

## 🔑 Resumen Clave

| Aspecto | Respuesta |
|---------|-----------|
| **¿Qué es GT?** | Poses 3D reales de MoCap (referencia perfecta) |
| **¿Dónde está?** | `data/Human36M/annotations/*_joint_3d.json` |
| **¿Necesito NPZ?** | ❌ No, se carga automáticamente del JSON |
| **¿Cuándo lo necesito?** | Entrenar, evaluar, benchmark (NO para demo) |
| **¿Se borra post-test?** | ❌ No, es parte permanente del dataset |
| **¿all_3d_poses.npz es GT?** | ❌ No, son predicciones de un video |
| **¿Cómo lo usa el repo?** | `dataset.data[i]['joint_cam']` automáticamente |

## 📚 Referencias en el Código

- Carga de GT: `data/Human36M/Human36M.py:86` (línea `json.load()`)
- Uso en evaluación: `data/Human36M/Human36M.py:160` (función `evaluate()`)
- Carga automática en benchmark: `tool/benchmark/runner_params.py:prep_data_from_dataset()`
- Formato MPJPE: `tool/benchmark/metrics.py:mpjpe()`
