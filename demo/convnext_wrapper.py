import sys
import os
import importlib
import importlib.util
import types
import torch
from torch.nn import functional as F
import numpy as np
import cv2
import math
import torchvision.transforms as transforms
from contextlib import contextmanager
from collections import OrderedDict
from torch.nn.parallel.data_parallel import DataParallel


class ConvNextPoseWrapper:
    def __init__(self, convnext_path, checkpoint_path):
        self.convnext_path = convnext_path
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.cfg = None
        self.device = None
        self._original_path = None
        
    @contextmanager
    def _isolated_import(self):
        """Context manager para aislar imports de ConvNextPose."""
        # Guardar estado actual
        self._original_path = sys.path.copy()
        original_modules = set(sys.modules.keys())
        
        try:
            # Añadir paths de ConvNextPose
            if self.convnext_path not in sys.path:
                sys.path.insert(0, self.convnext_path)
                sys.path.insert(0, os.path.join(self.convnext_path, 'main'))
                sys.path.insert(0, os.path.join(self.convnext_path, 'data'))
                sys.path.insert(0, os.path.join(self.convnext_path, 'common'))
            
            yield
            
        finally:
            # Restaurar estado
            sys.path = self._original_path
            # Remover módulos de ConvNeXtPose para evitar conflictos
            new_modules = set(sys.modules.keys()) - original_modules
            for module in new_modules:
                if any(path in module for path in ['convnextpose', 'main', 'data', 'common']):
                    sys.modules.pop(module, None)
    
    def load_model(self, use_gpu=True):
        """Carga ConvNextPose en contexto aislado."""
        with self._isolated_import():
            try:
                joint_num = 18
                # Importar módulos necesarios
                spec = importlib.util.spec_from_file_location(
                    "convnextpose_model", 
                    os.path.join(self.convnext_path, 'main', "model.py")
                )
                ConvNextPose_model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ConvNextPose_model_module)
                
                spec = importlib.util.spec_from_file_location(
                    "convnextpose_config", 
                    os.path.join(self.convnext_path, 'main', "config.py")
                )
                ConvNextPose_config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ConvNextPose_config_module)
                
                # Set device properly
                device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                
                # Cargar checkpoint siguiendo patrón de v4_production_optimized
                checkpoint = torch.load(self.checkpoint_path, map_location=device)
                
                self.cfg = ConvNextPose_config_module.cfg
                model = ConvNextPose_model_module.get_pose_net(self.cfg, is_train=False, joint_num=joint_num)
                
                # Handle DataParallel models (like in v4_production_optimized)
                if 'network' in checkpoint:
                    state_dict = checkpoint['network']
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix if present (from DataParallel)
                if any(key.startswith('module.') for key in state_dict.keys()):
                    print("[INFO] Removing DataParallel prefix from state_dict...")
                    state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
                
                # Load with strict=False to handle minor architecture differences
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f"[WARNING] Missing keys in model: {missing_keys[:5]}")
                if unexpected_keys:
                    print(f"[WARNING] Unexpected keys in model: {unexpected_keys[:5]}")
                
                # Move to device and set eval mode
                model.to(device)
                model.eval()
                
                self.model = model
                self.device = device
                print("[INFO] ConvNextPose cargado exitosamente")
                
            except Exception as e:
                print(f"[ERROR] No se pudo cargar ConvNextPose: {e}")
                import traceback
                traceback.print_exc()
                self.model = None
    
    def predict_pose(self, original_img, bbox, root_depth):
        """Predice poses usando ConvNextPose con la misma lógica que demo.py"""
        if self.model is None or self.cfg is None:
            return self._fallback_depth(bbox)
        
        try:
            with self._isolated_import():
                # Transform (same as v4_production_optimized)
                transform = transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=self.cfg.pixel_mean, std=self.cfg.pixel_std)
                ])
                
                # Camera parameters (same as demo.py)
                focal = [1500, 1500]
                original_img_height, original_img_width = original_img.shape[:2]
                princpt = [original_img_width / 2, original_img_height / 2]  # x-axis, y-axis
                
                # Importar módulos para procesamiento
                spec = importlib.util.spec_from_file_location(
                    "ConvNextPose_utils", 
                    os.path.join(self.convnext_path, 'common', "utils", "pose_utils.py")
                )
                ConvNextPose_utils = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ConvNextPose_utils)
                
                spec = importlib.util.spec_from_file_location(
                    "ConvNextPose_dataset", 
                    os.path.join(self.convnext_path, 'data', "dataset.py")
                )
                ConvNextPose_dataset = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ConvNextPose_dataset)
                
                # Procesar bbox como en demo.py
                processed_bbox = ConvNextPose_utils.process_bbox(
                    np.array(bbox), original_img_width, original_img_height
                )
                if processed_bbox is None:
                    print(f"[WARNING] processed_bbox is None for bbox: {bbox}. Using fallback.")
                    return self._fallback_depth(bbox)
                
                # Generate patch image (same as demo.py)
                img, img2bb_trans = ConvNextPose_dataset.generate_patch_image(
                    original_img, processed_bbox, False, 1.0, 0.0, False
                )
                
                # Prepare input tensor with proper device handling
                img_tensor = transform(img).to(self.device)[None, :, :, :]

                # Inference
                with torch.no_grad():
                    pose_3d = self.model(img_tensor)
                
                # Post-processing (exactly like demo.py)
                pose_3d = pose_3d[0].cpu().numpy()
                pose_3d[:, 0] = pose_3d[:, 0] / self.cfg.output_shape[1] * self.cfg.input_shape[1]
                pose_3d[:, 1] = pose_3d[:, 1] / self.cfg.output_shape[0] * self.cfg.input_shape[0]
                
                # Inverse affine transform (restore the crop and resize)
                pose_3d_xy1 = np.concatenate((pose_3d[:, :2], np.ones_like(pose_3d[:, :1])), 1)
                img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0, 0, 1]).reshape(1, 3)))
                pose_3d[:, :2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
                output_pose_2d = pose_3d[:, :2].copy()
                
                # Root-relative discretized depth -> absolute continuous depth
                pose_3d[:, 2] = (pose_3d[:, 2] / self.cfg.depth_dim * 2 - 1) * \
                    (self.cfg.bbox_3d_shape[0] / 2) + root_depth
                pose_3d = ConvNextPose_utils.pixel2cam(pose_3d, focal, princpt)
                output_pose_3d = pose_3d.copy()
                
                return output_pose_2d, output_pose_3d
                
        except Exception as e:
            print(f"[WARNING] Error en predicción ConvNextPose: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_depth(bbox)
    
    def _fallback_depth(self, bbox):
        """Fallback basado en área del bbox."""
        bbox_area = bbox[2] * bbox[3]
        estimated_depth = 3000.0 / np.sqrt(bbox_area + 1e-6)
        return max(min(estimated_depth, 5000.0), 500.0)
    
    def return_model(self):
        if self.model is not None:
            return self.model
        else:
            raise ValueError("ConvNextPose model is not loaded. Please call load_model() first.")