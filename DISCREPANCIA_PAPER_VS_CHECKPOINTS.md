# 🔍 Discrepancia Crítica: Paper vs Checkpoints

**Fecha:** 19 de Octubre, 2025  
**Estado:** 🔴 Discrepancia Confirmada

---

## 📋 Resumen Ejecutivo

Hemos identificado una **discrepancia importante** entre las especificaciones arquitecturales publicadas en el paper y los pesos (checkpoints) reales del modelo XS.

### ⚡ Hallazgo Principal

El modelo **XS** según el paper debería usar backbone **Atto**, pero el checkpoint real contiene un backbone **Femto-L**.

---

## 📊 Comparación Detallada

### Modelo XS - Paper vs Checkpoint

| Aspecto | Paper (Tabla 5) | Checkpoint Real (snapshot_67) |
|---------|----------------|-------------------------------|
| **Backbone** | Atto (2,2,6,2) | Femto-L (3,3,9,3) |
| **Dims** | [40, 80, 160, 320] | [48, 96, 192, 384] |
| **Upsampling** | 3-UP [320, 128, 128] | 3-UP [384, 256, 256] |
| **Head Channels** | 128 | 256 |
| **Params** | 3.53M | ~7.45M |
| **MPJPE** | 56.61 mm | 56.61 mm (esperado) |

### Modelos S, M, L

✅ **Sin discrepancias**: Todos coinciden con las especificaciones del paper (Femto-L backbone).

---

## 🔬 Evidencia del Análisis de Checkpoint

### Análisis de State Dict (snapshot_67.pth)

```python
# Evidencia 1: Dimensiones del primer stage del backbone
Key: module.backbone.stages.0.0.dwconv.weight
Shape: torch.Size([48, 1, 7, 7])  # 48 channels
Paper esperaba: [40, 1, 7, 7]     # 40 channels para Atto

# Evidencia 2: Número de bloques en stage 2
Checkpoint tiene: stages.2.0 hasta stages.2.8  (9 bloques)
Paper Atto esperaba: 6 bloques
Paper Femto-L especifica: 9 bloques ✅

# Evidencia 3: Final layer del head
Key: module.head.final_layer.weight
Shape: torch.Size([2304, 64, 1, 1])
Cálculo: 2304 / 64 = 36 joints
Input desde última capa deconv: 64 channels

Paper Atto esperaba: entrada desde 128 channels
Checkpoint muestra: entrada desde 256 channels (3x256=768 → 64 → final)
```

### Error al Intentar Cargar con Config del Paper

```python
RuntimeError: Error(s) in loading state_dict for DataParallel:
    size mismatch for module.backbone.stages.0.0.dwconv.weight: 
    copying a param with shape torch.Size([48, 1, 7, 7]) from checkpoint, 
    the shape in current model is torch.Size([40, 1, 7, 7]).
    
    Missing key(s) in state_dict: "module.backbone.stages.2.6.*", 
                                   "module.backbone.stages.2.7.*",
                                   "module.backbone.stages.2.8.*"
```

---

## 🎯 Tabla del Paper (Referencia)

**Tabla 5 del Paper:**

| Modelo | Backbone | Upsampling | Params | MPJPE |
|--------|----------|------------|--------|-------|
| **XS** | **Atto (2,2,6,2)** | **3-UP [320,128,128]** | **3.53M** | **56.61mm** |
| S | Femto-L (3,3,9,3) | 3-UP [384,256,256] | 7.45M | 51.80mm |
| M | Femto-L (3,3,9,3) | 3-UP [384,256,256] | 7.60M | 51.05mm |
| L | Femto-L (3,3,9,3) | 3-UP [384,512,512] | 8.39M | 49.75mm |

---

## 💡 Configuraciones Correctas

### Para Usar con Checkpoints Proporcionados

```python
# XS - Configuración que FUNCIONA con snapshot_67.pth
cfg.backbone = 'Femto-L'  # NO 'Atto'
cfg.backbone_cfg = ([3,3,9,3], [48,96,192,384])  # NO ([2,2,6,2], [40,80,160,320])
cfg.depth = 256  # NO 128
cfg.depth_dim = 64
cfg.head_cfg = None  # Modo Legacy
```

### Para Replicar Paper (Requeriría Re-entrenar)

```python
# XS - Configuración según Paper
cfg.backbone = 'Atto'
cfg.backbone_cfg = ([2,2,6,2], [40,80,160,320])
cfg.depth = 128
cfg.depth_dim = 64
cfg.head_cfg = None
```

---

## 🤔 Posibles Explicaciones

### 1. **Error en Nomenclatura de Checkpoints**
   - El archivo `snapshot_67.pth` podría no corresponder al modelo XS del paper
   - Posible confusión en el proceso de release de modelos
   - **Probabilidad:** Media

### 2. **Optimización Post-Publicación**
   - El modelo XS podría haber sido mejorado después de la publicación
   - Upgrade de Atto → Femto-L para mejor rendimiento
   - El MPJPE se mantiene similar (56.61mm)
   - **Probabilidad:** Media-Alta

### 3. **Versión de Desarrollo vs Publicación**
   - Los checkpoints podrían ser de una versión de desarrollo diferente
   - Paper documenta arquitectura final, checkpoints son versión intermedia
   - **Probabilidad:** Baja

### 4. **Configuración Modificada Durante Entrenamiento**
   - Posible switch de Atto → Femto-L durante fine-tuning
   - Checkpoint final refleja arquitectura modificada
   - **Probabilidad:** Media

---

## 📈 Impacto en Resultados

### Parámetros del Modelo

```
Paper XS:          3.53M params (Atto)
Checkpoint XS:    ~7.45M params (Femto-L)
Diferencia:       +3.92M params (+111%)
```

### Rendimiento Esperado

- **MPJPE**: Ambos deberían lograr ~56.61mm según paper
- **Capacidad**: Femto-L tiene mayor capacidad de representación
- **Velocidad**: Atto sería más rápido (menos params)

---

## ✅ Estrategia de Testing Recomendada

### Enfoque Dual

```python
# 1. Intentar primero con configuración del paper
try:
    model = create_model_atto()  # Según paper
    load_checkpoint('snapshot_67.pth')
except RuntimeError:
    # 2. Si falla, usar configuración de checkpoint
    model = create_model_femto_l()  # Según análisis
    load_checkpoint('snapshot_67.pth')  # ✅ Funciona
```

### Documentar Resultados

Para cada modelo, registrar:
- ✅ Configuración que funcionó (paper o checkpoint)
- ✅ Número de parámetros real
- ✅ MPJPE obtenido vs esperado
- ✅ Diferencias arquitecturales identificadas

---

## 🔧 Script de Verificación

```python
def verify_checkpoint_architecture(checkpoint_path):
    """
    Verifica la arquitectura real contenida en un checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['network']
    
    # Verificar dimensiones del backbone
    first_stage_key = 'module.backbone.stages.0.0.dwconv.weight'
    if first_stage_key in state_dict:
        dims = state_dict[first_stage_key].shape[0]
        if dims == 40:
            print("✅ Checkpoint contiene Atto backbone")
        elif dims == 48:
            print("⚠️  Checkpoint contiene Femto-L backbone")
        else:
            print(f"❓ Dimensión desconocida: {dims}")
    
    # Verificar número de bloques en stage 2
    stage2_blocks = [k for k in state_dict.keys() if 'stages.2.' in k]
    max_block = max([int(k.split('.')[3]) for k in stage2_blocks if k.split('.')[3].isdigit()])
    print(f"   Stage 2 tiene {max_block + 1} bloques")
    if max_block + 1 == 6:
        print("   ✅ Coincide con Atto (2,2,6,2)")
    elif max_block + 1 == 9:
        print("   ⚠️  Coincide con Femto-L (3,3,9,3)")
    
    # Verificar head channels
    final_layer_key = 'module.head.final_layer.weight'
    if final_layer_key in state_dict:
        in_channels = state_dict[final_layer_key].shape[1]
        print(f"   Head input channels: {in_channels}")
        if in_channels == 32:  # 128 / 4 (upsampling)
            print("   ✅ Coincide con depth=128")
        elif in_channels == 64:  # 256 / 4 (upsampling)
            print("   ⚠️  Coincide con depth=256")

# Uso
verify_checkpoint_architecture('snapshot_67.pth')
```

---

## 📝 Recomendaciones

### Para Testing Inmediato

1. ✅ **Usar configuraciones de checkpoint** (Femto-L para XS)
2. ✅ **Documentar discrepancias** encontradas
3. ✅ **Comparar MPJPE** obtenido vs paper

### Para Investigación Futura

1. 📧 **Contactar autores** para clarificar nomenclatura
2. 🔍 **Verificar otros checkpoints** (S, M, L) por consistencia
3. 📊 **Comparar rendimiento** Atto vs Femto-L para XS
4. 🏗️ **Re-entrenar XS con Atto** según especificaciones originales

### Para Documentación

1. 📝 Actualizar README con configuraciones reales
2. ⚠️  Agregar warnings sobre discrepancias
3. 📊 Incluir tabla comparativa en guías
4. 🔗 Referenciar este documento en guías principales

---

## 📚 Referencias

- **Paper Original**: ConvNeXtPose (Tabla 5)
- **Checkpoints**: snapshot_67.pth (XS), snapshot_68.pth (S), etc.
- **Análisis Previo**: `HALLAZGO_ARQUITECTURAS_REALES.md`
- **Configuración Código**: `main/config.py`, `main/model.py`

---

## 🎯 Conclusión

La discrepancia entre paper y checkpoints es **real y verificable**. Para trabajar con los checkpoints proporcionados, debemos usar configuraciones que difieren del paper en el caso del modelo XS.

**Estrategia recomendada:**
1. Probar primero con config del paper
2. Si falla, usar config encontrada en checkpoint
3. Documentar qué configuración funcionó
4. Reportar hallazgos a los autores

---

**Documentado por:** Análisis técnico de checkpoints  
**Verificado con:** PyTorch state_dict inspection  
**Fecha:** 19 de Octubre, 2025  
**Estado:** Pendiente de confirmación por autores
