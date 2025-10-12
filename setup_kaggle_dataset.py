#!/usr/bin/env python3
"""
Script para configurar automáticamente el dataset Human3.6M en Kaggle
sin necesidad de copiar archivos, usando enlaces simbólicos.

Uso en Kaggle:
    import os
    os.environ['CONVNEXPOSE_DATA_DIR'] = '/kaggle/working/data'
    !python setup_kaggle_dataset.py --kaggle-input /kaggle/input/human36m-dataset
"""

import argparse
import os
import sys
from pathlib import Path
import shutil


def create_symlink_safe(src, dst):
    """Crea un enlace simbólico, manejando casos especiales"""
    dst = Path(dst)
    src = Path(src)
    
    # Si el destino ya existe y apunta al lugar correcto, no hacer nada
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            print(f"  ✓ Enlace ya existe: {dst.name} -> {src}")
            return True
        else:
            print(f"  ⚠️  {dst.name} ya existe, saltando...")
            return False
    
    try:
        dst.symlink_to(src)
        print(f"  ✓ Creado: {dst.name} -> {src}")
        return True
    except Exception as e:
        print(f"  ❌ Error creando enlace {dst}: {e}")
        return False


def setup_kaggle_structure(kaggle_input_path, output_data_dir):
    """
    Configura la estructura de datos esperada por ConvNeXtPose
    usando enlaces simbólicos a los datos de Kaggle.
    
    Args:
        kaggle_input_path: Ruta al dataset montado en Kaggle (ej: /kaggle/input/human36m-dataset)
        output_data_dir: Ruta donde crear la estructura (ej: /kaggle/working/data)
    """
    kaggle_input = Path(kaggle_input_path)
    output_dir = Path(output_data_dir)
    
    print("\n" + "="*70)
    print("  Configuración de Dataset Human3.6M para ConvNeXtPose")
    print("="*70)
    print(f"📂 Entrada Kaggle: {kaggle_input}")
    print(f"📂 Salida:         {output_dir}")
    print()
    
    # Verificar que el input existe
    if not kaggle_input.exists():
        print(f"❌ Error: No se encuentra {kaggle_input}")
        print("   Verifica que el dataset esté montado en Kaggle")
        return False
    
    # Crear estructura base
    h36m_dir = output_dir / 'Human36M'
    h36m_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Creado directorio base: {h36m_dir}")
    
    # ==========================================
    # 1. CONFIGURAR CARPETA ANNOTATIONS
    # ==========================================
    print("\n📁 [1/3] Configurando annotations...")
    
    # Buscar carpeta annotations (puede estar como "annotations (1)" o "annotations")
    annotations_candidates = [
        kaggle_input / 'annotations (1)',
        kaggle_input / 'annotations',
        kaggle_input / 'annotation',
    ]
    
    annotations_src = None
    for candidate in annotations_candidates:
        if candidate.exists():
            annotations_src = candidate
            print(f"  ✓ Encontrado: {candidate.name}")
            break
    
    if not annotations_src:
        print(f"  ❌ No se encontró carpeta annotations en {kaggle_input}")
        print(f"     Contenido de {kaggle_input}:")
        for item in kaggle_input.iterdir():
            print(f"       - {item.name}")
        return False
    
    annotations_dst = h36m_dir / 'annotations'
    create_symlink_safe(annotations_src, annotations_dst)
    
    # ==========================================
    # 2. CONFIGURAR CARPETAS DE SUJETOS
    # ==========================================
    print("\n👥 [2/3] Configurando sujetos S9 y S11...")
    
    # Buscar S9_ACT2_16 o similares
    subject_patterns = {
        'S9': ['S9_ACT2_16', 'S9_ACT2', 'S9'],
        'S11': ['S11_ACT2_16', 'S11_ACT2', 'S11']
    }
    
    images_dir = h36m_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    
    for subject, patterns in subject_patterns.items():
        found = False
        for pattern in patterns:
            candidate = kaggle_input / pattern
            if candidate.exists():
                dst = images_dir / subject
                create_symlink_safe(candidate, dst)
                found = True
                break
        
        if not found:
            print(f"  ⚠️  No se encontró {subject} en ninguno de: {patterns}")
            print(f"     Esto puede causar errores si necesitas evaluar {subject}")
    
    # ==========================================
    # 3. CONFIGURAR BBOX_ROOT (SI EXISTE)
    # ==========================================
    print("\n📦 [3/3] Configurando bbox_root...")
    
    bbox_candidates = [
        kaggle_input / 'bbox_root',
        kaggle_input / 'Bounding box + Root joint coordinate-20230423T040706Z-001',
    ]
    
    bbox_src = None
    for candidate in bbox_candidates:
        if candidate.exists():
            bbox_src = candidate
            print(f"  ✓ Encontrado: {candidate.name}")
            break
    
    if bbox_src:
        bbox_dst = h36m_dir / 'bbox_root'
        create_symlink_safe(bbox_src, bbox_dst)
    else:
        print("  ⚠️  No se encontró bbox_root (opcional)")
    
    # ==========================================
    # RESUMEN
    # ==========================================
    print("\n" + "="*70)
    print("  ✅ Configuración Completada")
    print("="*70)
    print(f"\n📂 Estructura creada en: {h36m_dir}")
    print("\nContenido:")
    
    for item in sorted(h36m_dir.rglob('*')):
        if item.is_symlink():
            target = item.resolve()
            rel_path = item.relative_to(h36m_dir)
            print(f"  🔗 {rel_path} -> {target.name}")
        elif item.is_dir() and item != h36m_dir:
            rel_path = item.relative_to(h36m_dir)
            print(f"  📁 {rel_path}/")
    
    print("\n" + "="*70)
    print("  📝 Siguiente Paso: Configurar Variable de Entorno")
    print("="*70)
    print(f"\nEn tu notebook de Kaggle, ejecuta:")
    print(f"\n  import os")
    print(f"  os.environ['CONVNEXPOSE_DATA_DIR'] = '{output_dir}'")
    print(f"\nLuego puedes ejecutar el testing normalmente:")
    print(f"  !cd ConvNeXtPose/main && python test.py --gpu 0 --epochs 70 --variant L")
    print()
    
    return True


def verify_structure(data_dir):
    """Verifica que la estructura esté correctamente configurada"""
    data_path = Path(data_dir)
    h36m_path = data_path / 'Human36M'
    
    print("\n" + "="*70)
    print("  🔍 Verificación de Estructura")
    print("="*70)
    
    checks = {
        'Human36M directory': h36m_path.exists(),
        'annotations folder': (h36m_path / 'annotations').exists(),
        'images folder': (h36m_path / 'images').exists(),
        'S9 subject': (h36m_path / 'images' / 'S9').exists(),
        'S11 subject': (h36m_path / 'images' / 'S11').exists(),
    }
    
    all_ok = True
    for check_name, result in checks.items():
        status = "✓" if result else "❌"
        print(f"  {status} {check_name}")
        if not result:
            all_ok = False
    
    if (h36m_path / 'bbox_root').exists():
        print(f"  ✓ bbox_root folder (optional)")
    
    print()
    if all_ok:
        print("  ✅ Estructura verificada correctamente")
    else:
        print("  ⚠️  Algunos elementos faltan - revisa la configuración")
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description='Configura el dataset Human3.6M para ConvNeXtPose en Kaggle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # En Kaggle notebook:
  !python setup_kaggle_dataset.py --kaggle-input /kaggle/input/human36m-v1
  
  # Especificar directorio de salida personalizado:
  !python setup_kaggle_dataset.py --kaggle-input /kaggle/input/human36m-v1 \\
                                   --output /kaggle/working/custom_data
  
  # Verificar estructura existente:
  !python setup_kaggle_dataset.py --verify /kaggle/working/data
        """
    )
    
    parser.add_argument(
        '--kaggle-input',
        type=str,
        help='Ruta al dataset montado en Kaggle (ej: /kaggle/input/human36m-dataset)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='/kaggle/working/data',
        help='Directorio donde crear la estructura (default: /kaggle/working/data)'
    )
    
    parser.add_argument(
        '--verify',
        type=str,
        help='Solo verificar una estructura existente sin crear enlaces'
    )
    
    args = parser.parse_args()
    
    # Modo verificación
    if args.verify:
        success = verify_structure(args.verify)
        sys.exit(0 if success else 1)
    
    # Modo setup
    if not args.kaggle_input:
        parser.print_help()
        print("\n❌ Error: Se requiere --kaggle-input o --verify")
        sys.exit(1)
    
    success = setup_kaggle_structure(args.kaggle_input, args.output)
    
    if success:
        print("\n🎉 Setup completado exitosamente!")
        print("\n💡 Tip: Ejecuta con --verify para verificar la estructura:")
        print(f"   !python setup_kaggle_dataset.py --verify {args.output}")
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
