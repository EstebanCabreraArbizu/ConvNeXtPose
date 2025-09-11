#!/usr/bin/env python3
"""
RootNet Wrapper Mejorado con Estimación Heurística
=================================================
Versión mejorada que funciona tanto con GPU como CPU,
con estimación heurística de profundidad cuando RootNet falla.
"""

import os
import sys
import logging
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import OrderedDict

logger = logging.getLogger(__name__)

class RootNetWrapperImproved:
    """Wrapper mejorado de RootNet con estimación heurística"""
    
    def __init__(self, rootnet_path: str, checkpoint_path: str):
        self.rootnet_path = Path(rootnet_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self.cfg = None
        self.use_heuristic = False  # Flag para usar estimación heurística
        
        # Configuración por defecto
        self.default_focal = [1500, 1500]
        
        # Parámetros para estimación heurística
        self.person_height_mm = 1700  # Altura promedio humano en mm
        self.camera_height_mm = 1600  # Altura estimada cámara
        
    def load_model(self, use_gpu=True):
        """Cargar modelo RootNet o activar modo heurístico"""
        try:
            # Intentar cargar RootNet real
            self._load_rootnet_model(use_gpu)
            logger.info("✅ RootNet modelo cargado exitosamente")
            
        except Exception as e:
            logger.warning(f"⚠️ RootNet falló, activando estimación heurística: {e}")
            self.use_heuristic = True
            logger.info("🧠 Modo heurístico activado para estimación 3D")
    
    def _load_rootnet_model(self, use_gpu):
        """Cargar modelo RootNet real"""
        # Agregar path de RootNet
        sys.path.insert(0, str(self.rootnet_path))
        
        # Importar componentes RootNet
        from main.config import cfg as rootnet_cfg
        from main.model import get_model
        
        self.cfg = rootnet_cfg
        
        # Configurar
        self.cfg.set_args('0')  # GPU 0 por defecto
        
        # Crear modelo
        model = get_model('test')
        
        if use_gpu and torch.cuda.is_available():
            model = model.cuda()
        
        # Cargar checkpoint
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location='cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        )
        
        state_dict = checkpoint.get('network', checkpoint)
        
        # Limpiar state_dict para CPU
        if not (use_gpu and torch.cuda.is_available()):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_key = k.replace('module.', '')
                new_state_dict[new_key] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        self.model = model
        
    def predict_depth(self, img_patch, bbox, focal=None):
        """Predecir profundidad usando RootNet o estimación heurística"""
        if focal is None:
            focal = self.default_focal
            
        if self.use_heuristic:
            return self._estimate_depth_heuristic(bbox)
        else:
            return self._predict_depth_rootnet(img_patch, bbox, focal)
    
    def _predict_depth_rootnet(self, img_patch, bbox, focal):
        """Predicción usando RootNet real"""
        try:
            if self.model is None:
                raise ValueError("Modelo RootNet no cargado")
            
            # Preparar imagen
            input_img = self._preprocess_image(img_patch)
            k_value = torch.FloatTensor([focal]).unsqueeze(0)
            
            if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                input_img = input_img.cuda()
                k_value = k_value.cuda()
            
            # Inferencia
            with torch.no_grad():
                pred = self.model(input_img, k_value)
                depth = pred[0, 2].item()  # Z coordinate
            
            return max(min(depth, 5000.0), 500.0)
            
        except Exception as e:
            logger.warning(f"RootNet falló, usando heurística: {e}")
            self.use_heuristic = True
            return self._estimate_depth_heuristic(bbox)
    
    def _estimate_depth_heuristic(self, bbox):
        """Estimación heurística de profundidad basada en tamaño del bbox"""
        x1, y1, x2, y2 = bbox
        
        # Calcular dimensiones del bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        
        # Estimación basada en altura del bbox
        # Asumiendo que una persona ocupa cierta proporción de la imagen
        if bbox_height > 400:  # Persona cerca
            depth = 1000 + np.random.normal(0, 200)  # 1m ± 20cm
        elif bbox_height > 250:  # Persona media distancia
            depth = 2000 + np.random.normal(0, 300)  # 2m ± 30cm  
        elif bbox_height > 150:  # Persona lejos
            depth = 3500 + np.random.normal(0, 500)  # 3.5m ± 50cm
        else:  # Persona muy lejos
            depth = 5000 + np.random.normal(0, 800)  # 5m ± 80cm
        
        # Agregar variación basada en posición horizontal (perspectiva)
        bbox_center_x = (x1 + x2) / 2
        if bbox_center_x < 400:  # Lado izquierdo
            depth *= 0.95
        elif bbox_center_x > 1200:  # Lado derecho  
            depth *= 1.05
        
        # Limitar rango
        depth = max(min(depth, 6000.0), 800.0)
        
        return depth
    
    def _preprocess_image(self, img_patch):
        """Preprocesar imagen para RootNet"""
        # Redimensionar a tamaño esperado por RootNet
        img_resized = cv2.resize(img_patch, (256, 256))
        
        # Normalizar
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Convertir a tensor
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor
    
    def generate_3d_pose(self, pose_2d, root_depth):
        """Generar pose 3D a partir de 2D y profundidad root"""
        pose_3d = np.zeros((len(pose_2d), 3))
        
        # Copiar coordenadas X, Y
        pose_3d[:, :2] = pose_2d[:, :2]
        
        # Generar coordenadas Z más realistas
        root_joint_idx = 0  # Asumimos que el joint 0 es la raíz
        pose_3d[root_joint_idx, 2] = root_depth
        
        # Generar profundidades relativas para otros joints
        # Basado en anatomía humana típica
        depth_offsets = self._get_anatomical_depth_offsets()
        
        for i, offset in enumerate(depth_offsets):
            if i < len(pose_3d):
                pose_3d[i, 2] = root_depth + offset
        
        return pose_3d
    
    def _get_anatomical_depth_offsets(self):
        """Offsets anatómicos típicos desde el punto raíz (en mm)"""
        # Para 18 joints típicos del esqueleto humano
        return [
            0,      # 0: Raíz (pelvis)
            20,     # 1: Cuello  
            30,     # 2: Cabeza
            -50,    # 3: Hombro derecho
            -80,    # 4: Codo derecho
            -100,   # 5: Muñeca derecha
            50,     # 6: Hombro izquierdo
            80,     # 7: Codo izquierdo
            100,    # 8: Muñeca izquierda
            -30,    # 9: Cadera derecha
            -20,    # 10: Rodilla derecha
            -10,    # 11: Tobillo derecho
            30,     # 12: Cadera izquierda
            20,     # 13: Rodilla izquierda
            10,     # 14: Tobillo izquierdo
            40,     # 15: Ojo derecho
            -40,    # 16: Ojo izquierdo
            0,      # 17: Nariz
        ]

# Alias para compatibilidad
RootNetWrapper = RootNetWrapperImproved
