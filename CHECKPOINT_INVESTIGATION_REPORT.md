# 🔍 Reporte de Investigación: Checkpoints Faltantes ConvNeXtPose L y M

**Fecha**: 2025-01-XX  
**Investigador**: Sistema de Análisis Automático  
**Objetivo**: Encontrar checkpoints correctos para modelos ConvNeXtPose-L y ConvNeXtPose-M

---

## 🎯 Problema Identificado

El [Google Drive oficial](https://drive.google.com/drive/folders/12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI) mencionado en el README **NO contiene los modelos L y M anunciados en el paper**.

---

## 📊 Análisis de Checkpoints Encontrados

### Archivo 1: `ConvNeXtPose_XS.tar` (40.58 MB)
```
Primera capa: module.backbone.downsample_layers.0.0.weight
Shape: [40, 3, 4, 4]
Variante: ULTRA-XS (40 canales de salida)
```
**Conclusión**: Modelo aún más pequeño que XS estándar (48 canales)

---

### Archivo 2: `ConvNeXtPose_S.tar` (85.41 MB)
```
Primera capa: module.backbone.downsample_layers.0.0.weight  
Shape: [48, 3, 4, 4]
Variante: XS (48 canales de salida)
Config: dims=[48, 96, 192, 384], depths=[3, 3, 9, 3]
```
**Conclusión**: El archivo llamado "S" en realidad contiene el modelo **XS**  
**Error**: Nombre incorrecto del archivo

---

### Archivo 3: `snapshot_18.pth.tar` (389.39 MB)
```
Primera capa: module.backbone.conv1.weight
Shape: [64, 3, 7, 7]
Arquitectura: ResNet-style (NO ConvNeXtPose)
```
**Conclusión**: Este NO es un modelo ConvNeXtPose. Tiene arquitectura ResNet del código base (PoseNet).

---

## 🚫 Modelos Faltantes

### ConvNeXtPose-M (Medium)
- **Config esperada**: dims=[128, 256, 512, 1024], depths=[3, 3, 27, 3]
- **Primera capa esperada**: [128, 3, 4, 4]
- **Tamaño estimado**: ~340-360 MB
- **MPJPE esperado**: 44.6 mm (Protocol 2)
- **Estado**: ❌ NO ENCONTRADO

### ConvNeXtPose-L (Large)
- **Config esperada**: dims=[192, 384, 768, 1536], depths=[3, 3, 27, 3]
- **Primera capa esperada**: [192, 3, 4, 4]
- **Tamaño estimado**: ~760-790 MB
- **MPJPE esperado**: 42.3 mm (Protocol 2)
- **Estado**: ❌ NO ENCONTRADO

---

## 📋 Tabla Comparativa: Esperado vs Encontrado

| Modelo   | Esperado en Paper | Encontrado en Drive | Estado       |
|----------|-------------------|---------------------|--------------|
| **XS**   | 48 ch, ~22M params | ✅ (como "S")      | Mal nombrado |
| **S**    | 96 ch, ~50M params | ❌                 | Faltante     |
| **M**    | 128 ch, ~88M params | ❌                | Faltante     |
| **L**    | 192 ch, ~197M params | ❌               | Faltante     |

---

## 🔗 Fuentes Consultadas

1. **GitHub Repository Original**: https://github.com/medialab-ku/ConvNeXtPose
   - README apunta al mismo Google Drive problemático
   - No hay Issues abiertos sobre checkpoints faltantes

2. **Fork Personal**: https://github.com/EstebanCabreraArbizu/ConvNeXtPose
   - También referencia el mismo Google Drive
   - Sin checkpoints alternativos

3. **IEEE Paper**: https://ieeexplore.ieee.org/document/10288440
   - Acceso bloqueado (paywall)
   - No pudimos verificar enlaces alternativos

4. **Google Drive Oficial**: `12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI`
   - Única fuente oficial de checkpoints
   - Solo contiene XS y versión más pequeña
   - Archivo `snapshot_18.pth.tar` es ResNet, no ConvNeXtPose

---

## 💡 Hipótesis del Problema

### Teoría 1: Checkpoints No Subidos
Los autores posiblemente entrenaron solo XS para la demo y no subieron L/M al Google Drive.

### Teoría 2: Archivos Movidos/Eliminados
Los checkpoints pudieron haber estado disponibles inicialmente y fueron removidos.

### Teoría 3: Nombres Incorrectos
Basado en que "ConvNeXtPose_S.tar" es realmente XS, es posible que:
- No exista "ConvNeXtPose_L.tar" en el Drive
- O existe con nombre diferente

---

## 🎬 Próximos Pasos Recomendados

### Opción A: Contactar Autores (RECOMENDADO)
1. **Crear GitHub Issue** en https://github.com/medialab-ku/ConvNeXtPose
   - Título: "Missing ConvNeXtPose-L and ConvNeXtPose-M checkpoints in Google Drive"
   - Contenido: Explicar análisis detallado
   - Solicitar: Checkpoints correctos o instrucciones de entrenamiento

2. **Contactar Autores Directamente**
   - Hong Son Nguyen (primer autor)
   - MyoungGon Kim
   - Correos posiblemente disponibles en el paper

### Opción B: Verificar Google Drive Manualmente
1. Acceder directamente al link del Google Drive
2. Verificar si hay archivos adicionales no listados
3. Buscar carpetas ocultas o archivos con nombres diferentes

### Opción C: Entrenar Modelos (ÚLTIMA OPCIÓN)
Si los checkpoints no existen:
1. Usar configuración de `config_variants.py`
2. Entrenar modelo L desde scratch
3. Entrenar modelo M desde scratch
4. **Costo**: Muy alto en tiempo y recursos GPU

### Opción D: Probar con Modelo Disponible
**Solución temporal para validar pipeline**:
1. Cambiar código a variante XS
2. Ejecutar testing end-to-end
3. Verificar que el sistema funciona
4. Esperar checkpoints correctos para L/M

---

## 🛠️ Script de Detección Usado

El análisis se realizó con:
```bash
python3 identify_model_variant.py
```

Este script:
- ✅ Detecta formato PyTorch (moderno/legacy)
- ✅ Extrae dimensiones de primera capa
- ✅ Identifica variante automáticamente
- ✅ Funciona con formatos legacy y modernos

---

## 📝 Conclusiones

1. **El Google Drive oficial contiene checkpoints incorrectos/incompletos**
2. **Los archivos están mal nombrados** (S es realmente XS)
3. **Los modelos L y M NO están disponibles públicamente**
4. **El testing con L/M está BLOQUEADO hasta conseguir checkpoints correctos**

### Impacto en el Proyecto
- ❌ No se puede reproducir el resultado 42.3mm del paper (modelo L)
- ❌ No se puede reproducir el resultado 44.6mm del paper (modelo M)
- ✅ Se puede testear XS (esperado: ~52mm)
- ⚠️ El pipeline de testing está listo, solo faltan los checkpoints

---

## 🔄 Estado Actual

**BLOQUEADO**: Esperando checkpoints correctos de modelos L y M.

**Próxima acción inmediata**: Crear GitHub Issue en repositorio oficial.

---

## 📎 Archivos Relacionados

- `identify_model_variant.py` - Script de detección de variantes
- `list_google_drive_contents.py` - Script de análisis de Drive
- `kaggle_testing_notebook.ipynb` - Notebook con extracción working
- `main/config_variants.py` - Configuraciones correctas de arquitectura

---

**Fecha de último análisis**: 2025-01-XX  
**Investigación completa**: ✅  
**Problema identificado**: ✅  
**Solución encontrada**: ❌ (Checkpoints no existen públicamente)
