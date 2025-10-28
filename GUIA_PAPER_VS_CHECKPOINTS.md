# 📊 Guía Rápida: Paper vs Checkpoints

**Última actualización:** 19 de Octubre, 2025

---

## 🎯 Información Clave del Paper

Según la **Tabla 5** del paper de ConvNeXtPose:

| Modelo | Backbone | Dims Backbone | Upsampling | Head Channels | Params | MPJPE |
|--------|----------|---------------|------------|---------------|--------|-------|
| **XS** | **Atto** | **(2,2,6,2)** <br> **[40,80,160,320]** | **3-UP** <br> **[320,128,128]** | **128** | **3.53M** | **56.61mm** |
| **S** | Femto-L | (3,3,9,3) <br> [48,96,192,384] | 3-UP <br> [384,256,256] | 256 | 7.45M | 51.80mm |
| **M** | Femto-L | (3,3,9,3) <br> [48,96,192,384] | 3-UP <br> [384,256,256] | 256 | 7.60M | 51.05mm |
| **L** | Femto-L | (3,3,9,3) <br> [48,96,192,384] | 3-UP <br> [384,512,512] | 512 | 8.39M | 49.75mm |

---

## ⚠️ Discrepancia Descubierta: Modelo XS

### Lo Que Encontramos

Al analizar el checkpoint `snapshot_67.pth` (que debería ser XS), descubrimos:

```python
# Checkpoint snapshot_67.pth contiene:
Backbone: Femto-L (3,3,9,3)  ← NO es Atto
Dims: [48, 96, 192, 384]     ← NO es [40,80,160,320]
Head: depth=256              ← NO es depth=128
Params: ~7.45M               ← NO es 3.53M
```

### Evidencia del Análisis

```python
# Verificación de state_dict:
checkpoint['network']['module.backbone.stages.0.0.dwconv.weight'].shape
# Resultado: torch.Size([48, 1, 7, 7])
# Paper Atto esperaba: torch.Size([40, 1, 7, 7])

# Verificación de bloques:
checkpoint tiene stages.2.0 hasta stages.2.8  (9 bloques)
# Paper Atto esperaba: stages.2.0 hasta stages.2.5 (6 bloques)
# Femto-L especifica: 9 bloques ✅
```

---

## 🔧 Configuraciones para Código

### Configuración 1: Según Paper (XS = Atto)

```python
# Para modelo XS según paper
cfg.backbone = 'Atto'
cfg.backbone_cfg = ([2, 2, 6, 2], [40, 80, 160, 320])
cfg.depth = 128  # Head channels
cfg.depth_dim = 64
cfg.head_cfg = None  # Modo Legacy

# Esta configuración:
# ✅ Coincide con paper
# ❌ NO funciona con snapshot_67.pth
```

### Configuración 2: Según Checkpoint (XS = Femto-L)

```python
# Para modelo XS según checkpoint real
cfg.backbone = 'Femto-L'
cfg.backbone_cfg = ([3, 3, 9, 3], [48, 96, 192, 384])
cfg.depth = 256  # Head channels
cfg.depth_dim = 64
cfg.head_cfg = None  # Modo Legacy

# Esta configuración:
# ❌ NO coincide con paper
# ✅ Funciona con snapshot_67.pth
```

### Configuraciones S, M, L (Sin Discrepancias)

```python
# Modelo S
cfg.backbone = 'Femto-L'
cfg.backbone_cfg = ([3, 3, 9, 3], [48, 96, 192, 384])
cfg.depth = 256
cfg.depth_dim = 64
cfg.head_cfg = None

# Modelo M
cfg.backbone = 'Femto-L'
cfg.backbone_cfg = ([3, 3, 9, 3], [48, 96, 192, 384])
cfg.depth = 256  # Igual que S
cfg.depth_dim = 64
cfg.head_cfg = None

# Modelo L
cfg.backbone = 'Femto-L'
cfg.backbone_cfg = ([3, 3, 9, 3], [48, 96, 192, 384])
cfg.depth = 512  # ← Diferencia principal con S/M
cfg.depth_dim = 64
cfg.head_cfg = None
```

---

## 📊 Comparación Visual

### Arquitectura XS: Paper vs Checkpoint

```
┌─────────────────────────────────────────────────────────┐
│ MODELO XS - PAPER                                       │
├─────────────────────────────────────────────────────────┤
│ Input (256x256x3)                                       │
│    ↓                                                    │
│ [Atto Backbone]                                         │
│  Stage 0: 2 blocks × 40 channels                       │
│  Stage 1: 2 blocks × 80 channels                       │
│  Stage 2: 6 blocks × 160 channels                      │
│  Stage 3: 2 blocks × 320 channels                      │
│    ↓                                                    │
│ [Head - 3 Deconv Layers]                               │
│  Deconv 1: 320 channels                                │
│  Deconv 2: 128 channels                                │
│  Deconv 3: 128 channels                                │
│    ↓                                                    │
│ Output (64x64 heatmaps)                                │
│                                                         │
│ Total Params: 3.53M                                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ MODELO XS - CHECKPOINT REAL                             │
├─────────────────────────────────────────────────────────┤
│ Input (256x256x3)                                       │
│    ↓                                                    │
│ [Femto-L Backbone]                                      │
│  Stage 0: 3 blocks × 48 channels  ← Más bloques        │
│  Stage 1: 3 blocks × 96 channels  ← Más bloques        │
│  Stage 2: 9 blocks × 192 channels ← Más bloques        │
│  Stage 3: 3 blocks × 384 channels ← Más bloques        │
│    ↓                                                    │
│ [Head - 3 Deconv Layers]                               │
│  Deconv 1: 384 channels   ← Más canales                │
│  Deconv 2: 256 channels   ← Más canales                │
│  Deconv 3: 256 channels   ← Más canales                │
│    ↓                                                    │
│ Output (64x64 heatmaps)                                │
│                                                         │
│ Total Params: ~7.45M  ← Casi el doble                  │
└─────────────────────────────────────────────────────────┘
```

---

## 🤔 ¿Por Qué Esta Discrepancia?

### Teorías Posibles

1. **Error en Nomenclatura**
   - El checkpoint `snapshot_67.pth` podría no ser el modelo XS original
   - Posible confusión al subir los modelos

2. **Optimización Post-Paper**
   - El modelo XS pudo ser mejorado después de la publicación
   - Upgrade de Atto → Femto-L para mejor accuracy
   - El MPJPE se mantiene ~56.61mm

3. **Versión de Desarrollo**
   - Los checkpoints podrían ser de una versión diferente
   - Paper documenta versión final, checkpoints son versión intermedia

4. **Configuración Durante Training**
   - Posible cambio de arquitectura durante fine-tuning
   - Checkpoint refleja configuración modificada

---

## ✅ Qué Hacer en la Práctica

### Estrategia Recomendada

El notebook de benchmark ahora implementa una **estrategia dual**:

```python
# Para modelo XS:
try:
    # Intento 1: Configuración del paper (Atto)
    load_model_with_config_paper()
except RuntimeError:
    # Intento 2: Configuración de checkpoint (Femto-L)
    load_model_with_config_checkpoint()
    # ⚠️ Reportar discrepancia

# Para S, M, L:
# Usar configuración del paper directamente
```

### Pasos a Seguir

1. ✅ **Ejecutar benchmark** con estrategia dual
2. 📝 **Documentar** qué configuración funcionó para XS
3. 📊 **Comparar MPJPE** obtenido vs paper (56.61mm)
4. 📧 **Considerar contactar** autores si discrepancia confirmada

---

## 📈 Resultados Esperados

### Si XS Usa Config del Paper (Atto)

```
✅ Arquitectura coincide con documentación
✅ Params: 3.53M
✅ MPJPE: ~56.61mm
✅ Todo como se esperaba
```

### Si XS Usa Config de Checkpoint (Femto-L)

```
⚠️ Arquitectura difiere del paper
⚠️ Params: ~7.45M (111% más)
✅ MPJPE: ~56.61mm (accuracy preservada)
⚠️ Discrepancia confirmada
```

---

## 🔍 Cómo Verificar Tú Mismo

### Script de Verificación Rápida

```python
import torch

# Cargar checkpoint
checkpoint = torch.load('snapshot_67.pth', map_location='cpu')
sd = checkpoint['network']

# Verificar primera capa del backbone
first_conv = sd['module.backbone.stages.0.0.dwconv.weight']
print(f"Primera capa dims: {first_conv.shape[0]}")
# Si es 40 → Atto ✅
# Si es 48 → Femto-L ⚠️

# Verificar bloques en stage 2
stage2_keys = [k for k in sd.keys() if 'stages.2.' in k]
max_block = max([int(k.split('.')[3]) for k in stage2_keys 
                 if k.split('.')[3].isdigit()])
print(f"Stage 2 tiene {max_block + 1} bloques")
# Si es 6 → Atto ✅
# Si es 9 → Femto-L ⚠️

# Verificar head input channels
final_layer = sd['module.head.final_layer.weight']
print(f"Head input: {final_layer.shape[1]} channels")
# Si es 32 → depth=128 ✅
# Si es 64 → depth=256 ⚠️
```

---

## 📚 Documentos Relacionados

- 📄 **DISCREPANCIA_PAPER_VS_CHECKPOINTS.md** - Análisis completo
- 📓 **convnextpose (5).ipynb** - Benchmark con estrategia dual
- 📋 **GUIA_COMPLETA_ACTUALIZADA.md** - Configuraciones verificadas
- 📊 **RESUMEN_ACTUALIZACION_DOCS.md** - Resumen de arquitecturas

---

## 🎯 Conclusión

**Para el modelo XS:** Existe una discrepancia entre paper y checkpoint que debe ser verificada mediante testing.

**Para S, M, L:** No hay discrepancias, usar configuraciones del paper directamente.

**Siguiente paso:** Ejecutar el benchmark y ver qué configuración funciona para XS.

---

**Creado:** 19 de Octubre, 2025  
**Basado en:** Análisis de checkpoint snapshot_67.pth y Tabla 5 del paper  
**Estado:** Pendiente de validación experimental
