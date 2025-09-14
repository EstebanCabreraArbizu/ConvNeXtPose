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

def export_rootnet_to_onnx_fixed():
    """Exportar RootNet con dimensiones FIJAS para compatibilidad TFLite"""
    
    # 1. Configurar paths de RootNet correctamente
    rootnet_path = str(ROOTNET_ROOT)
    main_path = str(ROOTNET_ROOT / "main")
    
    # Agregar ambos paths
    if rootnet_path not in sys.path:
        sys.path.insert(0, rootnet_path)
    if main_path not in sys.path:
        sys.path.insert(0, main_path)
    
    # Cambiar directorio de trabajo temporalmente
    original_cwd = os.getcwd()
    os.chdir(main_path)  # Cambiar al directorio main
    
    try:
        # 2. Importar módulos de RootNet
        from collections import OrderedDict
        from config import cfg  # Import directo desde main
        from model import get_pose_net  # Import correcto desde main
        
        # 3. Configurar RootNet
        cfg.set_args('0')  # GPU 0 por defecto
        
        # 4. Crear modelo
        model = get_pose_net(cfg, is_train=False)  # Usar get_pose_net
        model.eval()
        
        # 5. Cargar checkpoint
        checkpoint_path = CONVNEXT_ROOT / "demo/snapshot_18.pth.tar"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")
        
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
        state_dict = checkpoint.get('network', checkpoint)
        
        # Limpiar state_dict para CPU
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            new_state_dict[new_key] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print("✅ Modelo RootNet cargado exitosamente")
        
    finally:
        # Restaurar directorio original
        os.chdir(original_cwd)
    
    # 6. Preparar input de prueba FIJO
    dummy_input = torch.randn(1, 3, 256, 256)  # FIJO: Batch=1
    dummy_k_value = torch.FloatTensor([[1500, 1500]])  # Focal length fijo
    
    # 7. Crear directorio exports si no existe
    exports_dir = CONVNEXT_ROOT / "exports"
    exports_dir.mkdir(exist_ok=True)
    
    # 8. Exportar a ONNX con dimensiones FIJAS
    onnx_path = exports_dir / "rootnet_fixed.onnx"
    
    print("🚀 Iniciando exportación ONNX...")
    
    with torch.no_grad():
        # Test del modelo antes de exportar
        test_output = model(dummy_input, dummy_k_value)
        print(f"✅ Test output shape: {test_output.shape}")
        
        torch.onnx.export(
            model,
            (dummy_input, dummy_k_value),  # Tuple de inputs
            str(onnx_path),
            export_params=True,
            opset_version=11,  # Compatible con onnx-tf
            do_constant_folding=True,  # Optimización
            input_names=['person_crop', 'k_value'],
            output_names=['pose_3d'],
            verbose=True
        )
    
    # 9. Verificar exportación
    print(f"🔍 Verificando ONNX exportado...")
    onnx_model = onnx.load(str(onnx_path))
    checker.check_model(onnx_model)
    
    # Información del modelo
    for input_info in onnx_model.graph.input:
        print(f"✅ Input '{input_info.name}': {[d.dim_value for d in input_info.type.tensor_type.shape.dim]}")
    
    for output_info in onnx_model.graph.output:
        print(f"✅ Output '{output_info.name}': {[d.dim_value for d in output_info.type.tensor_type.shape.dim]}")
    
    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"📦 RootNet ONNX exportado: {onnx_path} ({file_size_mb:.1f} MB)")
    
    return onnx_path

if __name__ == "__main__":
    try:
        onnx_path = export_rootnet_to_onnx_fixed()
        print(f"🎉 Exportación exitosa: {onnx_path}")
    except Exception as e:
        print(f"❌ Error durante exportación: {e}")
        import traceback
        traceback.print_exc()