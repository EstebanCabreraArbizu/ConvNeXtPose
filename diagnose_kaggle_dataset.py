#!/usr/bin/env python3
"""
Script de diagnóstico para inspeccionar la estructura del dataset de Kaggle.
Útil para identificar carpetas anidadas y rutas correctas.

Uso:
    !python diagnose_kaggle_dataset.py /kaggle/input/human36m-dataset
"""

import sys
from pathlib import Path


def diagnose_dataset(kaggle_input_path, max_depth=3):
    """Analiza la estructura del dataset de Kaggle"""
    
    kaggle_input = Path(kaggle_input_path)
    
    if not kaggle_input.exists():
        print(f"❌ Error: {kaggle_input} no existe")
        print("\n📂 Datasets disponibles en /kaggle/input/:")
        input_dir = Path('/kaggle/input')
        if input_dir.exists():
            for item in sorted(input_dir.iterdir()):
                print(f"   - {item.name}")
        return False
    
    print("\n" + "="*70)
    print(f"  🔍 DIAGNÓSTICO: {kaggle_input.name}")
    print("="*70 + "\n")
    
    # ==========================================
    # 1. BUSCAR ANNOTATIONS
    # ==========================================
    print("📁 [1/3] Buscando carpeta annotations...\n")
    
    annotations_found = []
    for item in kaggle_input.rglob('*annotation*'):
        if item.is_dir():
            # Contar archivos JSON dentro
            json_files = list(item.glob('*.json'))
            rel_path = item.relative_to(kaggle_input)
            annotations_found.append((rel_path, len(json_files)))
            
            indent = "  " * (len(rel_path.parts) - 1)
            print(f"{indent}📁 {rel_path}")
            if json_files:
                print(f"{indent}   ✓ {len(json_files)} archivos JSON")
                for json_file in json_files[:3]:  # Mostrar primeros 3
                    print(f"{indent}     • {json_file.name}")
                if len(json_files) > 3:
                    print(f"{indent}     ... y {len(json_files) - 3} más")
            else:
                print(f"{indent}   ⚠️  Sin archivos JSON (puede ser contenedor)")
    
    if annotations_found:
        print(f"\n✅ Encontradas {len(annotations_found)} carpeta(s) con 'annotation' en el nombre")
        # Recomendar la mejor opción
        best = max(annotations_found, key=lambda x: x[1])
        if best[1] > 0:
            print(f"💡 Recomendación: Usar '{best[0]}' ({best[1]} archivos JSON)")
    else:
        print("❌ No se encontraron carpetas de annotations")
    
    # ==========================================
    # 2. BUSCAR SUJETOS S9, S11
    # ==========================================
    print("\n" + "-"*70)
    print("\n👥 [2/3] Buscando carpetas de sujetos (S9, S11)...\n")
    
    subjects_found = []
    for pattern in ['*S9*', '*S11*']:
        for item in kaggle_input.rglob(pattern):
            if item.is_dir():
                rel_path = item.relative_to(kaggle_input)
                # Contar imágenes dentro
                img_count = sum(1 for _ in item.rglob('*.jpg'))
                subjects_found.append((rel_path, img_count))
                
                indent = "  " * (len(rel_path.parts) - 1)
                print(f"{indent}📁 {rel_path}")
                if img_count > 0:
                    print(f"{indent}   ✓ {img_count} imágenes")
                else:
                    print(f"{indent}   ⚠️  Sin imágenes (puede ser contenedor)")
    
    if subjects_found:
        print(f"\n✅ Encontradas {len(subjects_found)} carpeta(s) de sujetos")
        for path, count in subjects_found:
            if count > 0:
                print(f"💡 {path}: {count} imágenes")
    else:
        print("❌ No se encontraron carpetas de sujetos")
    
    # ==========================================
    # 3. BUSCAR BBOX_ROOT
    # ==========================================
    print("\n" + "-"*70)
    print("\n📦 [3/3] Buscando bbox_root_human36m_output.json...\n")
    
    bbox_files = list(kaggle_input.rglob('bbox_root_human36m_output.json'))
    
    if bbox_files:
        print(f"✅ Encontrado(s) {len(bbox_files)} archivo(s):\n")
        for bbox_file in bbox_files:
            rel_path = bbox_file.relative_to(kaggle_input)
            print(f"  📄 {rel_path}")
            print(f"     Directorio: {rel_path.parent}")
    else:
        print("⚠️  No se encontró bbox_root_human36m_output.json (opcional)")
    
    # ==========================================
    # ESTRUCTURA COMPLETA (primeros 2 niveles)
    # ==========================================
    print("\n" + "="*70)
    print("  📂 ESTRUCTURA COMPLETA (2 niveles)")
    print("="*70 + "\n")
    
    def print_tree(path, prefix="", max_depth=2, current_depth=0):
        if current_depth >= max_depth:
            return
        
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                next_prefix = "    " if is_last else "│   "
                
                icon = "📁" if item.is_dir() else "📄"
                print(f"{prefix}{current_prefix}{icon} {item.name}")
                
                if item.is_dir() and current_depth < max_depth - 1:
                    print_tree(item, prefix + next_prefix, max_depth, current_depth + 1)
        except PermissionError:
            print(f"{prefix}    [Permiso denegado]")
    
    print(f"📂 {kaggle_input.name}")
    print_tree(kaggle_input, max_depth=2)
    
    print("\n" + "="*70)
    print("  ✅ DIAGNÓSTICO COMPLETADO")
    print("="*70)
    print("\n💡 Usa esta información para ajustar setup_kaggle_dataset.py si es necesario\n")
    
    return True


def main():
    if len(sys.argv) < 2:
        print("Uso: python diagnose_kaggle_dataset.py <ruta_dataset_kaggle>")
        print("\nEjemplo:")
        print("  !python diagnose_kaggle_dataset.py /kaggle/input/human36m-dataset")
        sys.exit(1)
    
    kaggle_input = sys.argv[1]
    diagnose_dataset(kaggle_input)


if __name__ == '__main__':
    main()
