#!/usr/bin/env python3
"""
RootNet to ONNX Exporter (CPU Compatible)
========================================
Exportar RootNet a ONNX sin dependencias de GPU
"""

import sys
import os
from pathlib import Path
import torch
import onnx
from onnx import checker
import numpy as np

# Configurar paths
ROOTNET_ROOT = Path("/home/user/convnextpose_esteban/3DMPPE_ROOTNET_RELEASE")
CONVNEXT_ROOT = Path("/home/user/convnextpose_esteban/ConvNeXtPose/")

def patch_cuda_calls():
    """Patch para eliminar llamadas CUDA del código RootNet"""
    import torch
    
    # Guardar funciones originales
    original_cuda = torch.Tensor.cuda
    original_arange = torch.arange
    
    def cpu_arange(*args, **kwargs):
        """Versión CPU de torch.arange"""
        result = original_arange(*args, **kwargs)
        return result.float()  # Sin .cuda()
    
    def cpu_cuda(self, *args, **kwargs):
        """Versión que no hace nada (mantiene en CPU)"""
        return self
    
    # Aplicar patches
    torch.arange = cpu_arange
    torch.Tensor.cuda = cpu_cuda
    
    return original_cuda, original_arange

def unpatch_cuda_calls(original_cuda, original_arange):
    """Restaurar funciones originales"""
    import torch
    torch.Tensor.cuda = original_cuda
    torch.arange = original_arange

def create_cpu_compatible_rootnet():
    """Crear versión CPU-compatible de RootNet"""
    
    # 1. Configurar paths
    rootnet_path = str(ROOTNET_ROOT)
    main_path = str(ROOTNET_ROOT / "main")
    
    if rootnet_path not in sys.path:
        sys.path.insert(0, rootnet_path)
    if main_path not in sys.path:
        sys.path.insert(0, main_path)
    
    # 2. Cambiar al directorio main
    original_cwd = os.getcwd()
    os.chdir(main_path)
    
    try:
        # 3. Aplicar patch para CPU
        original_cuda, original_arange = patch_cuda_calls()
        
        # 4. Importar y crear modelo
        from collections import OrderedDict
        from config import cfg
        from model import get_pose_net  # Usar función original
        
        # 5. Configurar para CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Forzar CPU
        cfg.set_args('0')  # Configurar
        
        # 6. Crear modelo completo
        model = get_pose_net(cfg, is_train=False)
        model.eval()
        
        print("✅ Modelo RootNet creado para CPU")
        
        # 7. Cargar checkpoint si existe
        checkpoint_path = CONVNEXT_ROOT / "demo/snapshot_18.pth.tar"
        
        if checkpoint_path.exists():
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
            state_dict = checkpoint.get('network', checkpoint)
            
            # Limpiar state_dict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_key = k.replace('module.', '')
                new_state_dict[new_key] = v.cpu()  # Forzar CPU
            
            model.load_state_dict(new_state_dict, strict=False)
            print("✅ Checkpoint cargado en CPU")
        
        return model
        
    finally:
        # 8. Restaurar funciones originales
        unpatch_cuda_calls(original_cuda, original_arange)
        os.chdir(original_cwd)

def export_rootnet_simple():
    """Exportar RootNet simplificado"""
    
    try:
        # 1. Crear modelo CPU-compatible
        model = create_cpu_compatible_rootnet()
        
        # 2. Preparar inputs de prueba
        dummy_input = torch.randn(1, 3, 256, 256)
        dummy_k_value = torch.FloatTensor([[1500, 1500]])
        
        print("🧪 Probando modelo...")
        
        # 3. Test simple (sin K-value complicado)
        with torch.no_grad():
            # Solo probar la parte de backbone
            x = model.backbone(dummy_input)
            print(f"✅ Backbone output shape: {x.shape}")
        
        # 4. Crear directorio exports
        exports_dir = CONVNEXT_ROOT / "exports"
        exports_dir.mkdir(exist_ok=True)
        
        # 5. Exportar solo backbone (más simple)
        onnx_path = exports_dir / "rootnet_backbone.onnx"
        
        print("🚀 Exportando backbone a ONNX...")
        
        torch.onnx.export(
            model.backbone,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['person_crop'],
            output_names=['features'],
            verbose=True
        )
        
        # 6. Verificar exportación
        print("🔍 Verificando ONNX...")
        onnx_model = onnx.load(str(onnx_path))
        checker.check_model(onnx_model)
        
        file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"📦 RootNet Backbone ONNX: {onnx_path} ({file_size_mb:.1f} MB)")
        
        return onnx_path
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🎯 Exportando RootNet Backbone a ONNX (CPU)...")
    
    onnx_path = export_rootnet_simple()
    
    if onnx_path:
        print(f"🎉 ¡Exportación exitosa!")
        print(f"📁 Archivo: {onnx_path}")
        print("💡 Nota: Solo backbone exportado. Para pipeline completo, usar estimación heurística.")
    else:
        print("❌ Exportación falló")