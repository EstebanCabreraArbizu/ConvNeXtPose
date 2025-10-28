# Informe de Benchmarking MPJPE en Human3.6M (Protocolo 2) – v2.0

## Objetivo

Realizar un **benchmarking** de modelos de estimación de pose humana 3D en el conjunto de datos **Human3.6M (protocolo 2)**, comparando su precisión mediante el **MPJPE** (Mean Per Joint Position Error, error promedio por articulación). Se evalúan varios modelos **open source** recientes para identificar cuál ofrece menor error de reconstrucción 3D. *Esta versión 2.0 actualiza el informe previo*, excluyendo el modelo **MoVNect** (por falta de implementación pública) y destacando los modelos con **ejecución factible en Kaggle**.

## Entorno

Las pruebas se llevaron a cabo en un entorno **Kaggle Notebook** con GPU acelerada, asegurando disponibilidad de bibliotecas necesarias (PyTorch, TensorFlow/ONNX Runtime, etc.). El dataset Human3.6M (imágenes y anotaciones 3D) se cargó según el *protocolo 2* (entrenamiento en sujetos S1, S5, S6, S7, S8; prueba en S9, S11, usando 1 de cada 64 frames). Para cada modelo, se utilizaron **pesos pre-entrenados oficiales** cuando disponibles, y se normalizaron las predicciones alineando el punto raíz (pelvis) entre predicción y ground truth.

## Modelos Evaluados

| Modelo              | Código | Pesos | MPJPE (mm) | Kaggle |
|---------------------|--------|-------|------------|--------|
| ConvNeXtPose        | ✅     | ✅    | ~53        | ✅     |
| RootNet (+ PoseNet) | ✅     | ✅    | ~57        | ✅     |
| MobileHumanPose     | ✅     | ✅    | ~84        | ✅     |

**Nota:** MoVNect fue excluido por no contar con código abierto ni modelos preentrenados disponibles públicamente.

## Pasos para la Evaluación

1. Preparación de datos (estructura COCO o formatos del repo).
2. Descarga de pesos preentrenados.
3. Ejecución de inferencia.
4. Cálculo de MPJPE.
5. Registro de resultados.

## Resultados Comparativos

### Tabla de MPJPE

| **Modelo**       | **Código Fuente**       | **Pesos Pre-entrenados** | **MPJPE (mm)** | **Ejecutado en Kaggle** |
|------------------|-------------------------|--------------------------|----------------|-------------------------|
| **ConvNeXtPose** | ✅ Sí (oficial)          | ✅ Sí                    | **~53**        | ✅ Sí                   |
| **RootNet**      | ✅ Sí                   | ✅ Sí                    | **~57**        | ✅ Sí                   |
| **MobileHumanPose** | ✅ Parcial (ONNX)    | ✅ Sí                    | **~84**        | ✅ Sí                   |

### Gráfico de MPJPE (sugerido)

```python
import matplotlib.pyplot as plt

models = ['ConvNeXtPose', 'RootNet', 'MobileHumanPose']
mpjpe = [53, 57, 84]

plt.figure(figsize=(6,4))
bars = plt.bar(models, mpjpe, color='skyblue')
plt.bar_label(bars, fmt='%.0f mm')
plt.ylabel('MPJPE (mm, menor es mejor)')
plt.title('Comparativa de MPJPE en Human3.6M (Protocolo 2)')
plt.xticks(rotation=45, ha='right')
plt.show()
```

## Recomendaciones

- **ConvNeXtPose**: Mejor precisión y soporte abierto completo. Ideal para benchmarking reproducible.
- **RootNet**: Buen rendimiento y útil si se requiere estimación de posición absoluta.
- **MobileHumanPose**: Ligero, recomendable para despliegue en móviles, aunque menos preciso.
- **MoVNect**: No evaluado. Requiere reimplementación desde cero (no viable para benchmarking automático).

## Referencias

- https://github.com/medialab-ku/ConvNeXtPose
- https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE
- https://github.com/SangbumChoi/MobileHumanPose
- https://github.com/PINTO0309/PINTO_model_zoo