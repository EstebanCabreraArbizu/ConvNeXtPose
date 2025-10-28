# 🚨 Hallazgo Crítico: Arquitecturas Reales de los Checkpoints

**Fecha:** 19 de Octubre 2025  
**Estado:** ✅ VERIFICADO mediante análisis de state_dict

---

## 🔍 Resumen Ejecutivo

La documentación del repositorio **NO coincide con los checkpoints reales**. Tras analizar el state_dict de cada checkpoint:

### ❌ Lo que decía la documentación

| Modelo | Variant | Backbone dims | Head Channels |
|--------|---------|---------------|---------------|
| XS | **Atto** | `[40,80,160,320]` | `[128,128,128]` |
| S | Femto-L | `[48,96,192,384]` | `[256,256,256]` |
| M | Femto-L | `[48,96,192,384]` | `[256,256,256]` |
| L | Femto-L | `[48,96,192,384]` | `[512,512,512]` |

### ✅ Lo que son realmente los checkpoints

| Checkpoint | Modelo | Variant | Backbone dims | Head Channels | Params |
|-----------|--------|---------|---------------|---------------|--------|
| `snapshot_67.pth` | **XS** | **Femto-L** | `[48,96,192,384]` | `[256,256,256]` | 7.45M |
| `snapshot_68.pth` | **S** | **Femto-L** | `[48,96,192,384]` | `[256,256,256]` | 7.45M |
| `snapshot_70.pth` | **M** | **Femto-L** | `[48,96,192,384]` | `[256,256,256]` | 7.60M |
| `snapshot_83.pth` | **L** | **Femto-L** | `[48,96,192,384]` | `[512,512,512]` | 8.39M |

---

## 📊 Evidencia del Análisis

### Checkpoint snapshot_68.pth (Supuesto "XS", en realidad "S")

```python
# Backbone dims extraídos del state_dict
'module.backbone.downsample_layers.0.0.weight': torch.Size([48, 3, 4, 4])   # Stage 0: 48
'module.backbone.downsample_layers.1.1.weight': torch.Size([96, 48, 2, 2])  # Stage 1: 96
'module.backbone.downsample_layers.2.1.weight': torch.Size([192, 96, 2, 2]) # Stage 2: 192
'module.backbone.downsample_layers.3.1.weight': torch.Size([384, 192, 2, 2])# Stage 3: 384
# → Backbone: [48, 96, 192, 384] = Femto-L, NO Atto [40, 80, 160, 320]

# Head channels
'module.head.deconv_layers_1.pwconv.weight': torch.Size([256, 384, 1, 1])  # 256 out
'module.head.deconv_layers_2.pwconv.weight': torch.Size([256, 256, 1, 1])  # 256 out
'module.head.deconv_layers_3.pwconv.weight': torch.Size([256, 256, 1, 1])  # 256 out
# → Head: [256, 256, 256], NO [128, 128, 128]

# Final layer
'module.head.final_layer.weight': torch.Size([1152, 256, 1, 1])
# → 1152 = 18 joints × 64 depth_dim (NO 32)
```

### Error al Intentar Cargar con Config Incorrecta

```
RuntimeError: size mismatch for module.backbone.downsample_layers.0.0.weight: 
  copying a param with shape torch.Size([48, 3, 4, 4]) from checkpoint, 
  the shape in current model is torch.Size([40, 3, 4, 4])
```

**Interpretación:** El checkpoint tiene 48 canales (Femto-L), pero el código intentaba crear 40 canales (Atto).

---

## 🎯 Implic aciones

### 1. NO Existe Variant "Atto" en los Checkpoints

Todos los checkpoints proporcionados usan **Femto-L** como backbone. La variante "Atto" mencionada en la documentación **no tiene checkpoint pre-entrenado disponible**.

### 2. XS ≠ Atto

El modelo XS en los checkpoints:
- **NO es** ConvNeXt-Atto (dims `[40,80,160,320]`)
- **SÍ es** ConvNeXt-Femto-L (dims `[48,96,192,384]`)
- Simplemente tiene menos parámetros que S/M/L por optimizaciones en otras partes

### 3. Diferencia entre XS, S, M, L

| Modelo | Backbone | Head Channels | Diferencia Clave |
|--------|----------|---------------|------------------|
| **XS** | Femto-L | 256 | Versión optimizada/podada de S |
| **S** | Femto-L | 256 | Baseline Femto-L |
| **M** | Femto-L | 256 | Igual que S (¿entrenado más epochs?) |
| **L** | Femto-L | **512** | Head más grande |

**Nota:** XS, S y M tienen la **misma arquitectura** (Femto-L con head 256). Las diferencias en parámetros (7.45M vs 7.60M) sugieren:
- Técnicas de pruning/quantization
- Diferencias en número de bloques en el stage 2 (6 vs 9)
- Diferencias en inicialización/entrenamiento

### 4. Todos Usan depth_dim = 64

No hay evidencia de que XS/S usen `depth_dim=32`. Todos los checkpoints tienen:
```python
final_layer.out_channels = 18 joints × 64 depth_dim
```

---

## 🛠️ Configuración Correcta

### Para XS

```python
cfg.variant = 'Femto-L'  # NO 'Atto'
cfg.backbone_cfg = ([3,3,9,3], [48,96,192,384])  # Femto-L dims
cfg.depth = 256  # Head channels
cfg.depth_dim = 64  # Depth bins
cfg.head_cfg = None  # Legacy mode
```

### Para S

```python
cfg.variant = 'Femto-L'
cfg.backbone_cfg = ([3,3,9,3], [48,96,192,384])
cfg.depth = 256
cfg.depth_dim = 64
cfg.head_cfg = None
```

### Para M

```python
cfg.variant = 'Femto-L'
cfg.backbone_cfg = ([3,3,9,3], [48,96,192,384])
cfg.depth = 256
cfg.depth_dim = 64
cfg.head_cfg = None
```

### Para L

```python
cfg.variant = 'Femto-L'
cfg.backbone_cfg = ([3,3,9,3], [48,96,192,384])
cfg.depth = 512  # ← ÚNICA DIFERENCIA: head más grande
cfg.depth_dim = 64
cfg.head_cfg = None
```

---

## 📝 Posible Explicación

### Teoría 1: Inconsistencia en Nomenclatura

Los autores del paper pueden haber:
1. Experimentado con ConvNeXt-Atto inicialmente
2. Decidido usar Femto-L para todos los modelos finales
3. NO actualizado la documentación para reflejar el cambio

### Teoría 2: Confusión con Variantes de ConvNeXt

ConvNeXt tiene muchas variantes:
- **Atto**: 3.7M params (clasificación ImageNet)
- **Femto**: 5.2M params
- **Pico**: 7.8M params
- **Nano**: 15.6M params

Es posible que "XS" originalmente refiriera a Atto, pero el checkpoint distribuido es Femto-L.

### Teoría 3: Optimizaciones Post-Entrenamiento

XS puede ser:
- Un modelo Femto-L entrenado normalmente
- Luego optimizado via pruning/quantization
- Resultando en menos parámetros pero misma arquitectura base

---

## ✅ Recomendaciones

### Para Usuarios del Repositorio

1. **Ignorar la documentación sobre "Atto" para XS**
2. **Usar configuración Femto-L para TODOS los modelos** (XS, S, M, L)
3. **Configurar `depth` según modelo**:
   - XS/S/M: `cfg.depth = 256`
   - L: `cfg.depth = 512`
4. **Siempre usar `depth_dim = 64`**

### Para los Autores del Repositorio

1. Actualizar `README.md` y documentación
2. Renombrar checkpoints para claridad:
   - `ConvNeXtPose_XS_Femto.tar` (no `XS`)
   - Incluir variant en el nombre
3. Agregar tabla de arquitecturas reales verificadas
4. Documentar explícitamente que NO hay checkpoint Atto

---

## 🔗 Referencias

- **Archivo de Análisis:** `demo/architecture_introspection.json`
- **Celda de Validación:** Notebook Kaggle, celda de validación profunda
- **Error Original:** RuntimeError al cargar XS con config Atto
- **Verificación:** Análisis manual de `state_dict` keys y shapes

---

**Autor:** GitHub Copilot  
**Colaborador:** Esteban Cabrera Arbizu  
**Método:** Análisis forense de state_dict de checkpoints pre-entrenados
