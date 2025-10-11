#!/bin/bash

################################################################################
# ConvNeXtPose - Ubuntu Quick Start Script
# Testing de Modelos L y M en Human3.6M Protocol 2
################################################################################

set -e  # Exit on error

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Función para imprimir headers
print_header() {
    echo -e "\n${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}$1${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

################################################################################
# PASO 1: Verificación del Sistema
################################################################################
print_header "PASO 1: Verificación del Sistema"

# Verificar Ubuntu
if [ -f /etc/os-release ]; then
    . /etc/os-release
    print_success "Sistema operativo: $NAME $VERSION"
else
    print_warning "No se pudo detectar la versión de Ubuntu"
fi

# Verificar Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    print_success "Python detectado: $PYTHON_VERSION"
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    print_success "Python detectado: $PYTHON_VERSION"
    PYTHON_CMD="python"
else
    print_error "Python no encontrado. Instalando..."
    sudo apt update
    sudo apt install -y python3 python3-pip python3-dev
    PYTHON_CMD="python3"
fi

# Verificar pip
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    print_warning "pip no encontrado. Instalando..."
    sudo apt install -y python3-pip
    PIP_CMD="pip3"
fi
print_success "pip detectado: $PIP_CMD"

# Verificar CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_success "CUDA detectado: $CUDA_VERSION"
else
    print_warning "CUDA (nvcc) no detectado"
    print_info "Si tienes GPU NVIDIA, instala CUDA desde: https://developer.nvidia.com/cuda-downloads"
fi

# Verificar GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)
    print_success "GPU detectada: $GPU_INFO"
    
    # Mostrar uso actual
    echo -e "\n${BOLD}Estado actual de la GPU:${NC}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | \
    while IFS=, read -r idx name mem_used mem_total util; do
        echo -e "  GPU $idx: $name"
        echo -e "    Memoria: $mem_used / $mem_total"
        echo -e "    Utilización: $util"
    done
else
    print_warning "nvidia-smi no disponible"
    print_info "Testing puede ejecutarse en CPU (será muy lento)"
fi

################################################################################
# PASO 2: Instalación de Dependencias
################################################################################
print_header "PASO 2: Instalación de Dependencias"

print_info "Instalando dependencias del sistema..."
sudo apt update
sudo apt install -y \
    build-essential \
    git \
    wget \
    curl \
    tree \
    vim \
    htop

print_info "Instalando dependencias de Python..."
$PIP_CMD install --upgrade pip

# Instalar PyTorch (detectar si hay CUDA)
if command -v nvidia-smi &> /dev/null; then
    print_info "Instalando PyTorch con soporte CUDA..."
    $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    print_info "Instalando PyTorch CPU (sin GPU)..."
    $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

print_info "Instalando otras dependencias..."
$PIP_CMD install timm pycocotools opencv-python tqdm numpy matplotlib scipy

print_success "Dependencias instaladas correctamente"

# Verificar PyTorch
echo -e "\n${BOLD}Verificando instalación de PyTorch:${NC}"
$PYTHON_CMD -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU actual: {torch.cuda.get_device_name(0)}')
"

################################################################################
# PASO 3: Verificación de Estructura de Datos
################################################################################
print_header "PASO 3: Verificación de Estructura de Datos"

# Dataset Human3.6M
if [ -d "data/Human36M" ]; then
    print_success "Dataset Human3.6M encontrado"
    
    # Verificar subdirectorios
    if [ -d "data/Human36M/images" ]; then
        IMG_COUNT=$(find data/Human36M/images -type f 2>/dev/null | wc -l)
        print_success "  images/ encontrado ($IMG_COUNT archivos)"
    else
        print_error "  images/ NO encontrado"
    fi
    
    if [ -d "data/Human36M/annotations" ]; then
        ANNOT_COUNT=$(find data/Human36M/annotations -name "*.json" 2>/dev/null | wc -l)
        print_success "  annotations/ encontrado ($ANNOT_COUNT archivos JSON)"
    else
        print_error "  annotations/ NO encontrado"
    fi
    
    if [ -d "data/Human36M/bbox_root" ]; then
        print_success "  bbox_root/ encontrado"
        
        # Verificar Protocol 2 bbox
        PROTOCOL2_BBOX="data/Human36M/bbox_root/Subject 9,11 (trained on subject 1,5,6,7,8)/bbox_root_human36m_output.json"
        if [ -f "$PROTOCOL2_BBOX" ]; then
            print_success "    Protocol 2 bbox encontrado ✓"
        else
            print_warning "    Protocol 2 bbox NO encontrado"
            print_info "    Buscando alternativas..."
            find data/Human36M/bbox_root -name "*.json" -type f
        fi
    else
        print_error "  bbox_root/ NO encontrado"
    fi
else
    print_error "Dataset Human3.6M NO encontrado en data/Human36M/"
    print_info "Descarga el dataset desde: https://drive.google.com/drive/folders/1r0B9I3XxIIW_jsXjYinDpL6NFcxTZart"
fi

################################################################################
# PASO 4: Verificación de Modelos Pre-entrenados
################################################################################
print_header "PASO 4: Verificación de Modelos Pre-entrenados"

# Crear directorio si no existe
mkdir -p output/model_dump output/result output/log output/vis

if [ -d "output/model_dump" ]; then
    # Contar checkpoints
    MODEL_COUNT=$(find output/model_dump -name "*.pth*" -type f 2>/dev/null | wc -l)
    
    if [ $MODEL_COUNT -gt 0 ]; then
        print_success "Encontrados $MODEL_COUNT checkpoint(s)"
        echo -e "\n${BOLD}Checkpoints disponibles:${NC}"
        ls -lh output/model_dump/*.pth* 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    else
        print_warning "No se encontraron checkpoints en output/model_dump/"
        echo ""
        print_info "Descarga modelos pre-entrenados desde:"
        echo "  https://drive.google.com/drive/folders/12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI"
        echo ""
        print_info "Guárdalos en: output/model_dump/snapshot_<epoch>.pth.tar"
        echo "  Ejemplo: output/model_dump/snapshot_70.pth.tar"
    fi
else
    mkdir -p output/model_dump
    print_success "Creado directorio output/model_dump/"
fi

################################################################################
# PASO 5: Verificación de Scripts de Testing
################################################################################
print_header "PASO 5: Verificación de Scripts de Testing"

# Verificar scripts
SCRIPTS=(
    "main/config_variants.py"
    "main/test_variants.py"
    "main/compare_variants.py"
)

ALL_SCRIPTS_OK=true
for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        print_success "$script encontrado"
    else
        print_error "$script NO encontrado"
        ALL_SCRIPTS_OK=false
    fi
done

if [ "$ALL_SCRIPTS_OK" = true ]; then
    print_success "Todos los scripts de testing están disponibles"
    
    # Test rápido de config_variants
    echo -e "\n${BOLD}Testeando config_variants.py:${NC}"
    cd main
    $PYTHON_CMD -c "
from config_variants import MODEL_CONFIGS
print('  Variantes disponibles:', list(MODEL_CONFIGS.keys()))
" 2>/dev/null || print_warning "  Error al importar config_variants.py"
    cd ..
else
    print_warning "Algunos scripts no están disponibles"
fi

################################################################################
# PASO 6: Verificación de Protocolo
################################################################################
print_header "PASO 6: Verificación de Protocolo"

if [ -f "data/Human36M/Human36M.py" ]; then
    PROTOCOL=$(grep "self.protocol" data/Human36M/Human36M.py | head -n 1 | awk '{print $3}')
    if [ "$PROTOCOL" = "2" ]; then
        print_success "Protocolo configurado correctamente: Protocol 2"
        print_info "  Sujetos de testing: S9 y S11"
        print_info "  Métrica: MPJPE (sin alineación Procrustes)"
    else
        print_warning "Protocolo actual: Protocol $PROTOCOL"
        print_info "Para Protocol 2, edita: data/Human36M/Human36M.py"
        print_info "  Cambia: self.protocol = $PROTOCOL"
        print_info "  A:      self.protocol = 2"
    fi
else
    print_warning "Archivo Human36M.py no encontrado"
fi

################################################################################
# PASO 7: Menú Interactivo
################################################################################
print_header "PASO 7: Comandos Disponibles"

cat << 'EOF'

╔══════════════════════════════════════════════════════════════╗
║                    COMANDOS DE TESTING                        ║
╚══════════════════════════════════════════════════════════════╝

1. Testear Modelo M (Medium)
   cd main && python3 test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox

2. Testear Modelo L (Large)
   cd main && python3 test_variants.py --variant L --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox

3. Testear con batch size personalizado (si OOM)
   cd main && python3 test_variants.py --variant M --gpu 0 --epoch 70 --batch_size 8

4. Comparar resultados M vs L
   cd main && python3 compare_variants.py --variants M L --epoch 70

5. Generar reporte completo con gráficos
   cd main && python3 compare_variants.py --variants M L --epoch 70 --plot --save_report

6. Ver información de una variante
   cd main && python3 -c "from config_variants import print_model_info; print_model_info('M')"

7. Ver todas las variantes
   cd main && python3 config_variants.py

╔══════════════════════════════════════════════════════════════╗
║                   COMANDOS DE UTILIDAD                        ║
╚══════════════════════════════════════════════════════════════╝

• Monitorear GPU en tiempo real:
  watch -n 1 nvidia-smi

• Ver resultados:
  cat output/result/results_M_epoch70.json
  cat output/result/comparison_report.md

• Verificar logs:
  tail -f output/log/*.log

• Estructura de datos:
  tree -L 3 data/Human36M/

EOF

################################################################################
# PASO 8: Ejecución Interactiva
################################################################################
print_header "PASO 8: ¿Qué deseas hacer?"

echo -e "${BOLD}Selecciona una opción:${NC}"
echo "  1) Testear Modelo M"
echo "  2) Testear Modelo L"
echo "  3) Testear M y L + Comparar (completo)"
echo "  4) Ver info de modelos"
echo "  5) Verificar instalación de PyTorch"
echo "  6) Monitorear GPU"
echo "  7) Solo mostrar comandos (ya mostrado arriba)"
echo "  0) Salir"
echo ""
read -p "Opción [0-7]: " choice

case $choice in
    1)
        print_header "Ejecutando Testing de Modelo M"
        cd main
        $PYTHON_CMD test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox
        ;;
    2)
        print_header "Ejecutando Testing de Modelo L"
        cd main
        $PYTHON_CMD test_variants.py --variant L --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox
        ;;
    3)
        print_header "Ejecutando Testing Completo (M + L + Comparación)"
        cd main
        
        print_info "1/3: Testeando Modelo M..."
        $PYTHON_CMD test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox
        
        print_info "2/3: Testeando Modelo L..."
        $PYTHON_CMD test_variants.py --variant L --gpu 0 --epoch 70 --protocol 2 --flip_test --use_gt_bbox
        
        print_info "3/3: Comparando resultados..."
        $PYTHON_CMD compare_variants.py --variants M L --epoch 70 --plot --save_report
        
        print_success "Testing completo finalizado!"
        print_info "Resultados en: output/result/"
        ;;
    4)
        print_header "Información de Modelos"
        cd main
        $PYTHON_CMD config_variants.py
        ;;
    5)
        print_header "Verificación de PyTorch"
        $PYTHON_CMD -c "
import torch
print('\n' + '='*60)
print('PyTorch Installation Check')
print('='*60)
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'Número de GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'  Memoria: {mem:.1f} GB')
else:
    print('GPU no disponible, usando CPU')
print('='*60 + '\n')

# Test tensor
print('Test rápido de operaciones:')
x = torch.randn(3, 3)
print(f'Tensor creado: {x.shape}')
if torch.cuda.is_available():
    x = x.cuda()
    print(f'Tensor en GPU: {x.device}')
print('✓ PyTorch funcionando correctamente\n')
"
        ;;
    6)
        print_header "Monitor de GPU (Ctrl+C para salir)"
        if command -v nvidia-smi &> /dev/null; then
            watch -n 1 nvidia-smi
        else
            print_error "nvidia-smi no disponible"
        fi
        ;;
    7)
        print_success "Comandos ya mostrados arriba"
        ;;
    0)
        print_info "Saliendo..."
        exit 0
        ;;
    *)
        print_warning "Opción no válida"
        ;;
esac

################################################################################
# RESUMEN FINAL
################################################################################
print_header "Resumen y Próximos Pasos"

cat << 'EOF'

✅ SETUP COMPLETO

Próximos pasos recomendados:

1. Si aún no tienes los modelos pre-entrenados:
   • Descárgalos desde: https://drive.google.com/drive/folders/12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI
   • Guárdalos en: output/model_dump/snapshot_70.pth.tar

2. Para testing rápido:
   cd main && python3 test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test

3. Para documentación completa:
   • PASOS_TESTING.md - Lista de pasos numerados
   • CHECKLIST_TESTING.md - Checklist interactiva
   • GUIA_TESTING_MODELOS_L_M.md - Guía completa

4. Para re-ejecutar este script:
   bash ubuntu_quickstart.sh

5. Para ayuda adicional:
   • Ver README_TESTING.md (índice general)
   • Ejecutar: cd main && python3 test_variants.py --help

EOF

print_success "¡Script completado exitosamente!"
echo ""
