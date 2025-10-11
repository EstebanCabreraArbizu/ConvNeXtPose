#!/bin/bash

# Script de Ayuda Rápida para Testing de ConvNeXtPose L y M
# Uso: bash quick_start.sh

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          ConvNeXtPose - Quick Start Testing Guide             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir con color
print_step() {
    echo -e "${BLUE}[PASO $1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# ============================================================================
# PASO 1: Verificar entorno
# ============================================================================
print_step 1 "Verificando entorno..."
echo ""

# Verificar Python
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_success "Python detectado: $PYTHON_VERSION"
else
    print_error "Python no encontrado"
    exit 1
fi

# Verificar CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_success "CUDA detectado: $CUDA_VERSION"
else
    print_warning "CUDA no detectado (nvcc no encontrado)"
fi

# Verificar GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    print_success "GPU detectada: $GPU_INFO ($GPU_MEM MB)"
else
    print_warning "nvidia-smi no disponible"
fi

echo ""

# ============================================================================
# PASO 2: Verificar estructura de datos
# ============================================================================
print_step 2 "Verificando estructura de datos..."
echo ""

# Verificar dataset Human3.6M
if [ -d "data/Human36M" ]; then
    print_success "Dataset Human3.6M encontrado"
    
    # Verificar subdirectorios
    if [ -d "data/Human36M/images" ]; then
        print_success "  - images/ encontrado"
    else
        print_error "  - images/ NO encontrado"
    fi
    
    if [ -d "data/Human36M/annotations" ]; then
        print_success "  - annotations/ encontrado"
    else
        print_error "  - annotations/ NO encontrado"
    fi
    
    if [ -d "data/Human36M/bbox_root" ]; then
        print_success "  - bbox_root/ encontrado"
    else
        print_error "  - bbox_root/ NO encontrado"
    fi
else
    print_error "Dataset Human3.6M NO encontrado en data/Human36M/"
fi

echo ""

# ============================================================================
# PASO 3: Verificar modelos pre-entrenados
# ============================================================================
print_step 3 "Verificando modelos pre-entrenados..."
echo ""

if [ -d "output/model_dump" ]; then
    # Contar archivos .pth o .pth.tar
    MODEL_COUNT=$(find output/model_dump -name "*.pth*" -type f | wc -l)
    
    if [ $MODEL_COUNT -gt 0 ]; then
        print_success "Encontrados $MODEL_COUNT checkpoints"
        echo ""
        echo "Checkpoints disponibles:"
        ls -lh output/model_dump/*.pth* 2>/dev/null || echo "  (ninguno)"
    else
        print_warning "No se encontraron checkpoints en output/model_dump/"
        echo ""
        echo "Descarga modelos pre-entrenados desde:"
        echo "  https://drive.google.com/drive/folders/12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI"
    fi
else
    print_error "Directorio output/model_dump/ no existe"
    mkdir -p output/model_dump
    print_success "Creado directorio output/model_dump/"
fi

echo ""

# ============================================================================
# PASO 4: Verificar scripts de testing
# ============================================================================
print_step 4 "Verificando scripts de testing..."
echo ""

if [ -f "main/config_variants.py" ]; then
    print_success "config_variants.py encontrado"
else
    print_error "config_variants.py NO encontrado"
fi

if [ -f "main/test_variants.py" ]; then
    print_success "test_variants.py encontrado"
else
    print_error "test_variants.py NO encontrado"
fi

if [ -f "main/compare_variants.py" ]; then
    print_success "compare_variants.py encontrado"
else
    print_error "compare_variants.py NO encontrado"
fi

echo ""

# ============================================================================
# PASO 5: Menú de opciones
# ============================================================================
print_step 5 "Comandos disponibles"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "  TESTING DE MODELOS"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "1. Testear Modelo M (Medium):"
echo "   cd main && python test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test"
echo ""
echo "2. Testear Modelo L (Large):"
echo "   cd main && python test_variants.py --variant L --gpu 0 --epoch 70 --protocol 2 --flip_test"
echo ""
echo "3. Testear con batch size personalizado:"
echo "   cd main && python test_variants.py --variant M --gpu 0 --epoch 70 --batch_size 8"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "  COMPARACIÓN DE RESULTADOS"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "4. Comparar resultados:"
echo "   cd main && python compare_variants.py --variants M L --epoch 70"
echo ""
echo "5. Generar gráficos:"
echo "   cd main && python compare_variants.py --variants M L --plot"
echo ""
echo "6. Generar reporte completo:"
echo "   cd main && python compare_variants.py --variants M L --plot --save_report"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "  TESTING DE CONFIGURACIÓN"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "7. Test del módulo de configuración:"
echo "   cd main && python config_variants.py"
echo ""
echo "8. Ver información de modelo:"
echo "   cd main && python -c \"from config_variants import print_model_info; print_model_info('M')\""
echo ""

# ============================================================================
# PASO 6: Modo interactivo
# ============================================================================
echo ""
read -p "¿Deseas ejecutar un comando ahora? (1-8/N): " choice

case $choice in
    1)
        echo ""
        print_step "Ejecutando" "Testear Modelo M"
        cd main && python test_variants.py --variant M --gpu 0 --epoch 70 --protocol 2 --flip_test
        ;;
    2)
        echo ""
        print_step "Ejecutando" "Testear Modelo L"
        cd main && python test_variants.py --variant L --gpu 0 --epoch 70 --protocol 2 --flip_test
        ;;
    3)
        echo ""
        read -p "Ingresa batch size: " batch_size
        print_step "Ejecutando" "Testear Modelo M con batch size $batch_size"
        cd main && python test_variants.py --variant M --gpu 0 --epoch 70 --batch_size $batch_size
        ;;
    4)
        echo ""
        print_step "Ejecutando" "Comparar resultados"
        cd main && python compare_variants.py --variants M L --epoch 70
        ;;
    5)
        echo ""
        print_step "Ejecutando" "Generar gráficos"
        cd main && python compare_variants.py --variants M L --plot
        ;;
    6)
        echo ""
        print_step "Ejecutando" "Generar reporte completo"
        cd main && python compare_variants.py --variants M L --plot --save_report
        ;;
    7)
        echo ""
        print_step "Ejecutando" "Test del módulo de configuración"
        cd main && python config_variants.py
        ;;
    8)
        echo ""
        read -p "Ingresa variante (M/L): " variant
        print_step "Ejecutando" "Ver información de modelo $variant"
        cd main && python -c "from config_variants import print_model_info; print_model_info('$variant')"
        ;;
    [Nn]*)
        echo ""
        print_success "Usa los comandos arriba cuando estés listo"
        ;;
    *)
        echo ""
        print_warning "Opción no válida"
        ;;
esac

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Para más información, consulta: GUIA_TESTING_MODELOS_L_M.md  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
