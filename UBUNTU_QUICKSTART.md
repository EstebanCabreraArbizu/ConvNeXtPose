# 🚀 Ubuntu Quick Start - ConvNeXtPose Testing

## Inicio Rápido en Una Línea

```bash
bash ubuntu_quickstart.sh
```

Este script automáticamente:
- ✅ Verifica Python, CUDA y GPU
- ✅ Instala todas las dependencias
- ✅ Verifica estructura de datos
- ✅ Verifica modelos pre-entrenados
- ✅ Ofrece menú interactivo para testing

---

## Requisitos Previos

### Sistema
- Ubuntu 18.04, 20.04, 22.04, o 24.04
- Python 3.8+
- GPU NVIDIA (opcional, pero recomendado)

### Verificación Rápida
```bash
# Verificar Python
python3 --version

# Verificar GPU
nvidia-smi

# Verificar que estás en el directorio correcto
pwd  # Debe mostrar: .../ConvNeXtPose
```

---

## Instalación Manual (si prefieres paso a paso)

### 1. Instalar Dependencias del Sistema
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-dev build-essential git
```

### 2. Instalar PyTorch (con CUDA)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**O sin GPU (CPU only)**:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Instalar Dependencias de Python
```bash
pip3 install timm pycocotools opencv-python tqdm numpy matplotlib scipy
```

### 4. Crear Directorios
```bash
mkdir -p output/model_dump output/result output/log
```

---

## Testing de Modelos

### Testear Modelo M (Medium)
```bash
cd main
python3 test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox
```

**Resultado esperado**: MPJPE ≈ 44.6 mm

### Testear Modelo L (Large)
```bash
cd main
python3 test_variants.py --variant L --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox
```

**Resultado esperado**: MPJPE ≈ 42.3 mm

### Comparar Resultados
```bash
cd main
python3 compare_variants.py --variants M L --epoch 70 --plot --save_report
```

---

## Comandos Útiles

### Monitorear GPU
```bash
watch -n 1 nvidia-smi
```

### Ver Información de Modelos
```bash
cd main
python3 config_variants.py
```

### Verificar PyTorch
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Ver Resultados
```bash
cat output/result/results_M_epoch70.json
cat output/result/comparison_report.md
```

---

## Troubleshooting

### Error: Out of Memory (OOM)
```bash
# Reducir batch size
cd main
python3 test_variants.py --variant L --gpu 0 --epoch 70 --batch_size 8
```

### Error: No module named 'torch'
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Error: CUDA not available
```bash
# Verificar instalación de NVIDIA drivers
nvidia-smi

# Si no funciona, instalar drivers
sudo ubuntu-drivers autoinstall
sudo reboot
```

### Error: Dataset not found
```bash
# Verificar estructura
ls -la data/Human36M/

# Descargar si es necesario desde:
# https://drive.google.com/drive/folders/1r0B9I3XxIIW_jsXjYinDpL6NFcxTZart
```

---

## Estructura de Directorios

```
ConvNeXtPose/
├── ubuntu_quickstart.sh           # ← Script principal
├── main/
│   ├── config_variants.py         # Configuraciones
│   ├── test_variants.py           # Testing
│   └── compare_variants.py        # Comparación
├── data/
│   └── Human36M/                  # Dataset
│       ├── images/
│       ├── annotations/
│       └── bbox_root/
└── output/
    ├── model_dump/                # Checkpoints aquí
    │   └── snapshot_70.pth.tar
    └── result/                    # Resultados
```

---

## Descargar Modelos Pre-entrenados

1. Ir a: https://drive.google.com/drive/folders/12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI
2. Descargar modelos M y/o L
3. Guardar en: `output/model_dump/snapshot_70.pth.tar`

**Alternativamente con gdown**:
```bash
pip3 install gdown
# Ajustar IDs según los archivos en Drive
# gdown <FILE_ID> -O output/model_dump/snapshot_70.pth.tar
```

---

## Configuración de Protocol 2

Verificar que usa Protocol 2:
```bash
grep "self.protocol" data/Human36M/Human36M.py
```

Debe mostrar: `self.protocol = 2`

Si muestra `self.protocol = 1`, cambiar a 2:
```bash
sed -i 's/self.protocol = 1/self.protocol = 2/' data/Human36M/Human36M.py
```

---

## Testing Completo en Un Comando

```bash
cd main && \
python3 test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox && \
python3 test_variants.py --variant L --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox && \
python3 compare_variants.py --variants M L --epoch 70 --plot --save_report
```

---

## Resultados Esperados

| Modelo | MPJPE (mm) | Params | GFLOPs | GPU Memory |
|--------|------------|--------|--------|------------|
| M      | 44.6       | 88.6M  | 15.4   | ~4GB       |
| L      | 42.3       | 197.8M | 34.4   | ~8GB       |

---

## Verificación del Sistema

### Ubuntu
```bash
lsb_release -a
```

### Python
```bash
python3 --version
pip3 --version
```

### CUDA
```bash
nvcc --version
nvidia-smi
```

### PyTorch
```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Espacio en Disco
```bash
df -h
```

---

## Permisos

Si tienes problemas de permisos:
```bash
# Hacer scripts ejecutables
chmod +x ubuntu_quickstart.sh
chmod +x quick_start.sh

# Dar permisos al directorio de output
chmod -R 755 output/
```

---

## Actualizar Dependencias

```bash
pip3 install --upgrade torch torchvision torchaudio
pip3 install --upgrade timm pycocotools opencv-python tqdm numpy matplotlib
```

---

## Logs y Debugging

### Ver logs
```bash
tail -f output/log/*.log
```

### Debug de GPU
```bash
# Información detallada de GPU
nvidia-smi -q

# Uso de memoria en tiempo real
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'
```

### Debug de Python
```bash
# Activar modo verbose
cd main
python3 -v test_variants.py --variant M --gpu 0 --epoch 70
```

---

## Variables de Entorno Útiles

```bash
# Usar GPU específica
export CUDA_VISIBLE_DEVICES=0

# Modo debug de CUDA
export CUDA_LAUNCH_BLOCKING=1

# Número de workers para DataLoader
export OMP_NUM_THREADS=4
```

---

## Documentación Adicional

- **`PASOS_TESTING.md`** - Lista de pasos numerados (1-11)
- **`CHECKLIST_TESTING.md`** - Checklist interactiva con checkboxes
- **`GUIA_TESTING_MODELOS_L_M.md`** - Guía completa detallada
- **`README_TESTING.md`** - Índice general de documentación
- **`RESUMEN_EJECUTIVO.md`** - Vista rápida TL;DR

---

## Soporte y Ayuda

### Re-ejecutar diagnóstico
```bash
bash ubuntu_quickstart.sh
```

### Ver ayuda de scripts
```bash
cd main
python3 test_variants.py --help
python3 compare_variants.py --help
```

### Verificar instalación
```bash
cd main
python3 -c "from config_variants import print_model_info; print_model_info('M')"
```

---

## Atajos de Teclado Útiles

En terminal:
- `Ctrl + C` - Cancelar ejecución
- `Ctrl + Z` - Suspender proceso
- `bg` - Continuar en background
- `jobs` - Ver procesos en background
- `fg` - Traer proceso al foreground

---

## Tips de Rendimiento

### Para máxima velocidad
```bash
# Usar batch size mayor (si tienes memoria)
python3 test_variants.py --variant M --gpu 0 --epoch 70 --batch_size 32

# Sin flip test (más rápido, menos preciso)
python3 test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2
```

### Para máxima precisión
```bash
# Con flip test y GT bbox
python3 test_variants.py --variant L --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox
```

---

## Comandos de Una Línea

### Setup completo
```bash
sudo apt update && sudo apt install -y python3 python3-pip && pip3 install torch torchvision timm pycocotools opencv-python tqdm numpy matplotlib && mkdir -p output/model_dump
```

### Verificación completa
```bash
python3 --version && nvidia-smi && python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" && ls data/Human36M/ && ls output/model_dump/
```

---

## Checklist Rápido

Antes de ejecutar testing:
- [ ] Ubuntu instalado
- [ ] Python 3.8+ funcionando
- [ ] pip3 instalado
- [ ] PyTorch instalado
- [ ] GPU NVIDIA funcionando (opcional)
- [ ] Dataset Human3.6M descargado
- [ ] Modelos pre-entrenados descargados
- [ ] Scripts de testing verificados
- [ ] Protocol configurado a 2

---

**¡Listo para comenzar!**

```bash
bash ubuntu_quickstart.sh
```

---

**Última actualización**: Octubre 2025  
**Sistema**: Ubuntu 18.04 / 20.04 / 22.04 / 24.04  
**Python**: 3.8+
