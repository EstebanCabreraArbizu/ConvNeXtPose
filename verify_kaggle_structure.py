#!/usr/bin/env python3
"""
Script de verificación rápida para confirmar que la estructura está correcta.
Ejecuta esto DESPUÉS de configurar el dataset.
"""

import os
import sys

def check_kaggle_structure():
    """Verifica la estructura correcta del proyecto en Kaggle"""
    
    project_root = '/kaggle/working/ConvNeXtPose'
    
    print("\n" + "="*70)
    print("  🔍 VERIFICACIÓN RÁPIDA - Estructura del Proyecto")
    print("="*70 + "\n")
    
    checks = {
        '📄 Módulos Python originales': [
            ('data/dataset.py', True),
            ('data/multiple_datasets.py', True),
            ('data/Human36M/Human36M.py', True),
            ('common/base.py', True),
            ('main/config.py', True),
        ],
        '📂 Dataset de Kaggle (enlaces)': [
            ('data/Human36M/images/S9', True),
            ('data/Human36M/images/S11', True),
            ('data/Human36M/annotations', True),
            ('data/Human36M/bbox_root', False),  # Opcional
        ],
    }
    
    all_critical_ok = True
    
    for category, paths in checks.items():
        print(f"{category}:")
        for rel_path, required in paths:
            full_path = os.path.join(project_root, rel_path)
            exists = os.path.exists(full_path)
            
            if required and not exists:
                all_critical_ok = False
                status = "❌"
            elif exists:
                status = "✓"
            else:
                status = "⚠️  (opcional)"
            
            print(f"  {status} {rel_path}")
        print()
    
    print("="*70)
    if all_critical_ok:
        print("✅ ESTRUCTURA CORRECTA - Listo para testing")
        print("\n🚀 Siguiente paso:")
        print("   %cd /kaggle/working/ConvNeXtPose/main")
        print("   !python test.py --gpu 0 --epochs 83 --variant L")
    else:
        print("❌ ESTRUCTURA INCORRECTA")
        print("\n🔧 Ejecuta:")
        print("   !python setup_kaggle_dataset.py --kaggle-input /kaggle/input/... --project-root /kaggle/working/ConvNeXtPose")
    print("="*70 + "\n")
    
    return all_critical_ok

if __name__ == '__main__':
    success = check_kaggle_structure()
    sys.exit(0 if success else 1)
