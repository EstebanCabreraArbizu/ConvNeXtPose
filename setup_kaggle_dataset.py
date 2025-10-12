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


def setup_kaggle_structure(kaggle_input_path, convnextpose_root):
    """
    Configura la estructura de datos esperada por ConvNeXtPose
    usando enlaces simbólicos SOLO dentro de data/Human36M/.
    
    ⚠️ IMPORTANTE: NO reemplaza la carpeta data/, solo enlaza contenido en data/Human36M/
    
    Args:
        kaggle_input_path: Ruta al dataset montado en Kaggle (ej: /kaggle/input/human36m-dataset)
        convnextpose_root: Ruta raíz del proyecto ConvNeXtPose (ej: /kaggle/working/ConvNeXtPose)
    """
    kaggle_input = Path(kaggle_input_path)
    project_root = Path(convnextpose_root)
    
    print("\n" + "="*70)
    print("  Configuración de Dataset Human3.6M para ConvNeXtPose")
    print("="*70)
    print(f"📂 Dataset Kaggle:     {kaggle_input}")
    print(f"📂 Proyecto ConvNeXt:  {project_root}")
    print()
    
    # Verificar que el input existe
    if not kaggle_input.exists():
        print(f"❌ Error: No se encuentra {kaggle_input}")
        print("   Verifica que el dataset esté montado en Kaggle")
        return False
    
    # Verificar que existe data/Human36M/ en el proyecto
    h36m_dir = project_root / 'data' / 'Human36M'
    if not h36m_dir.exists():
        print(f"❌ Error: No se encuentra {h36m_dir}")
        print(f"   Verifica que estás en el directorio correcto del proyecto")
        return False
    
    print(f"✓ Directorio del proyecto encontrado: {h36m_dir}")
    print(f"✓ Manteniendo módulos Python originales en {project_root / 'data'}")
    
    # ==========================================
    # 1. CONFIGURAR CARPETA ANNOTATIONS
    # ==========================================
    print("\n📁 [1/3] Configurando annotations...")
    
    # Buscar carpeta annotations (puede estar como "annotations (1)/annotations" o "annotations")
    annotations_candidates = [
        kaggle_input / 'annotations (1)' / 'annotations',  # Caso anidado
        kaggle_input / 'annotations (1)',                   # Caso directo
        kaggle_input / 'annotations',                       # Caso simple
        kaggle_input / 'annotation',                        # Variante
    ]
    
    annotations_src = None
    for candidate in annotations_candidates:
        if candidate.exists():
            # Verificar que contiene archivos JSON (no es solo un contenedor vacío)
            if candidate.is_dir():
                json_files = list(candidate.glob('*.json'))
                if json_files:
                    annotations_src = candidate
                    print(f"  ✓ Encontrado: {candidate.relative_to(kaggle_input)}")
                    print(f"    ({len(json_files)} archivos JSON detectados)")
                    break
                else:
                    print(f"  ⚠️  {candidate.name} existe pero está vacío, probando siguiente...")
            else:
                annotations_src = candidate
                print(f"  ✓ Encontrado: {candidate.name}")
                break
    
    if not annotations_src:
        print(f"  ❌ No se encontró carpeta annotations válida en {kaggle_input}")
        print(f"     Contenido de {kaggle_input}:")
        for item in kaggle_input.iterdir():
            print(f"       - {item.name}")
            if item.is_dir() and 'annotation' in item.name.lower():
                print(f"         Contenido de {item.name}:")
                for subitem in item.iterdir():
                    print(f"           • {subitem.name}")
        return False
    
    annotations_dst = h36m_dir / 'annotations'
    create_symlink_safe(annotations_src, annotations_dst)
    
    # ==========================================
    # 2. CONFIGURAR CARPETAS DE SUJETOS (IMAGES)
    # ==========================================
    print("\n👥 [2/3] Configurando sujetos S9 y S11...")
    
    # Buscar S9_ACT2_i6 o similares variantes
    subject_patterns = {
        'S9': ['S9_ACT2_i6', 'S9_ACT2_16', 'S9_ACT2', 'S9'],
        'S11': ['S11_ACT2_i6', 'S11_ACT2_16', 'S11_ACT2', 'S11']
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
        # Buscar la carpeta con el archivo bbox_root_human36m_output.json
        kaggle_input / 'Bounding box + Root joint coordinate-20230423T040706Z-001' / 'Bounding box + Root joint coordinate' / 'Human3.6M' / 'Subject 9,11 (trained on subject 1,5,6,7,8)',
        kaggle_input / 'bbox_root' / 'Subject 9,11 (trained on subject 1,5,6,7,8)',
        kaggle_input / 'bbox_root',
        kaggle_input / 'Bounding box + Root joint coordinate-20230423T040706Z-001',
    ]
    
    bbox_src = None
    for candidate in bbox_candidates:
        if candidate.exists():
            # Verificar que contiene el archivo JSON esperado
            if (candidate / 'bbox_root_human36m_output.json').exists():
                bbox_src = candidate
                print(f"  ✓ Encontrado: {candidate.relative_to(kaggle_input)}")
                print(f"    (con bbox_root_human36m_output.json)")
                break
            elif candidate.is_dir():
                # Si es un directorio, buscar dentro
                json_file = list(candidate.rglob('bbox_root_human36m_output.json'))
                if json_file:
                    bbox_src = json_file[0].parent
                    print(f"  ✓ Encontrado: {bbox_src.relative_to(kaggle_input)}")
                    print(f"    (bbox_root_human36m_output.json detectado)")
                    break
    
    if bbox_src:
        bbox_dst = h36m_dir / 'bbox_root'
        # Si bbox_root ya existe en el repo, crear subdirectorio
        if bbox_dst.exists() and not bbox_dst.is_symlink():
            bbox_dst = bbox_dst / 'Subject 9,11 (trained on subject 1,5,6,7,8)'
            bbox_dst.parent.mkdir(parents=True, exist_ok=True)
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
    print("  ✅ LISTO - NO necesitas configurar CONVNEXPOSE_DATA_DIR")
    print("="*70)
    print(f"\n✅ Los módulos Python originales están intactos en:")
    print(f"   {project_root / 'data' / 'dataset.py'}")
    print(f"   {project_root / 'data' / 'Human36M' / 'Human36M.py'}")
    print(f"\n✅ El dataset de Kaggle está enlazado en:")
    print(f"   {h36m_dir / 'images'}")
    print(f"   {h36m_dir / 'annotations'}")
    print(f"   {h36m_dir / 'bbox_root'} (si existe)")
    print(f"\n🚀 Puedes ejecutar el testing directamente:")
    print(f"   %cd {project_root / 'main'}")
    print(f"   !python test.py --gpu 0 --epochs 70 --variant L")
    print()
    
    return True


def verify_structure(convnextpose_root):
    """Verifica que la estructura esté correctamente configurada"""
    project_root = Path(convnextpose_root)
    data_path = project_root / 'data'
    h36m_path = data_path / 'Human36M'
    
    print("\n" + "="*70)
    print("  🔍 Verificación de Estructura")
    print("="*70)
    
    checks = {
        'data/dataset.py': (data_path / 'dataset.py').exists(),
        'data/Human36M/Human36M.py': (h36m_path / 'Human36M.py').exists(),
        'data/Human36M/annotations': (h36m_path / 'annotations').exists(),
        'data/Human36M/images': (h36m_path / 'images').exists(),
        'data/Human36M/images/S9': (h36m_path / 'images' / 'S9').exists(),
        'data/Human36M/images/S11': (h36m_path / 'images' / 'S11').exists(),
    }
    
    all_ok = True
    for check_name, result in checks.items():
        status = "✓" if result else "❌"
        print(f"  {status} {check_name}")
        if not result:
            all_ok = False
    
    if (h36m_path / 'bbox_root').exists():
        print(f"  ✓ data/Human36M/bbox_root (optional)")
    
    print()
    if all_ok:
        print("  ✅ Estructura verificada correctamente")
        print("  ✅ Módulos Python intactos en data/")
        print("  ✅ Dataset enlazado en data/Human36M/")
    else:
        print("  ⚠️  Algunos elementos faltan - revisa la configuración")
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description='Configura el dataset Human3.6M para ConvNeXtPose en Kaggle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # En Kaggle notebook (desde /kaggle/working/ConvNeXtPose):
  !python setup_kaggle_dataset.py --kaggle-input /kaggle/input/human36m-dataset
  
  # Especificar directorio del proyecto:
  !python setup_kaggle_dataset.py --kaggle-input /kaggle/input/human36m-dataset \\
                                   --project-root /kaggle/working/ConvNeXtPose
  
  # Verificar estructura existente:
  !python setup_kaggle_dataset.py --verify /kaggle/working/ConvNeXtPose
        """
    )
    
    parser.add_argument(
        '--kaggle-input',
        type=str,
        help='Ruta al dataset montado en Kaggle (ej: /kaggle/input/human36m-dataset)'
    )
    
    parser.add_argument(
        '--project-root',
        type=str,
        default='.',
        help='Ruta raíz del proyecto ConvNeXtPose (default: directorio actual)'
    )
    
    parser.add_argument(
        '--verify',
        type=str,
        help='Solo verificar una estructura existente. Especifica la ruta del proyecto ConvNeXtPose.'
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
    
    success = setup_kaggle_structure(args.kaggle_input, args.project_root)
    
    if success:
        print("\n🎉 Setup completado exitosamente!")
        print("\n💡 Tip: Ejecuta con --verify para verificar la estructura:")
        print(f"   !python setup_kaggle_dataset.py --verify {args.project_root}")
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
