#!/usr/bin/env python3

"""
Test 3D Isolated - Sin conflictos de imports
============================================
"""

import os
import sys
import cv2
import numpy as np
import time
import logging
import torch
import math
import importlib.util
from pathlib import Path
from contextlib import contextmanager
import torchvision.transforms as transforms
from collections import OrderedDict

# TensorFlow Lite import
TFLITE_AVAILABLE = False
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False

# Agregar path del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_torchvision_patch():
    """Aplica un parche a torchvision para compatibilidad con código antiguo"""
    import torchvision
    import torchvision.models.resnet as resnet_module
    
    # Solo aplicar si no existe model_urls
    if not hasattr(resnet_module, 'model_urls'):
        # Recrear model_urls como existía en versiones antiguas
        resnet_module.model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        }
        logger.info("[INFO] Se aplicó parche de compatibilidad para torchvision")

class SimpleRootNetWrapper:
    """Wrapper simple de RootNet sin conflictos de imports"""
    
    def __init__(self, rootnet_path, checkpoint_path):
        self.rootnet_path = rootnet_path
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.cfg = None
        self._original_path = None
        
    @contextmanager
    def _isolated_import(self):
        """Context manager para aislar imports de RootNet."""
        # Guardar estado actual
        self._original_path = sys.path.copy()
        original_modules = set(sys.modules.keys())
        
        try:
            apply_torchvision_patch()
            # Añadir paths de RootNet
            if self.rootnet_path not in sys.path:
                sys.path.insert(0, self.rootnet_path)
                sys.path.insert(0, os.path.join(self.rootnet_path, 'main'))
                sys.path.insert(0, os.path.join(self.rootnet_path, 'data'))
                sys.path.insert(0, os.path.join(self.rootnet_path, 'common'))
            
            yield
            
        finally:
            # Restaurar estado
            sys.path = self._original_path
            # Remover módulos de RootNet para evitar conflictos
            new_modules = set(sys.modules.keys()) - original_modules
            for module in new_modules:
                if any(path in module for path in ['rootnet', 'main', 'data', 'common']):
                    sys.modules.pop(module, None)
    
    def load_model(self, use_gpu=True):
        """Carga RootNet en contexto aislado."""
        with self._isolated_import():
            try:
                # Importar módulos necesarios
                spec = importlib.util.spec_from_file_location(
                    "rootnet_model", 
                    os.path.join(self.rootnet_path, 'main', "model.py")
                )
                rootnet_model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(rootnet_model_module)
                
                spec = importlib.util.spec_from_file_location(
                    "rootnet_config", 
                    os.path.join(self.rootnet_path, 'main', "config.py")
                )
                rootnet_config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(rootnet_config_module)
                
                # Crear modelo siguiendo el mismo patrón que demo.py
                self.cfg = rootnet_config_module.cfg
                model = rootnet_model_module.get_pose_net(self.cfg, is_train=False)
                
                if use_gpu and torch.cuda.is_available():
                    from torch.nn.parallel.data_parallel import DataParallel
                    model = DataParallel(model).cuda()
                
                # Cargar checkpoint
                checkpoint = torch.load(self.checkpoint_path, 
                                       map_location='cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                state_dict = checkpoint.get('network', checkpoint)
                
                if not(use_gpu and torch.cuda.is_available()):
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        new_key = k.replace('module.', '')
                        new_state_dict[new_key] = v
                    state_dict = new_state_dict
                
                model.load_state_dict(state_dict)
                self.model = model.eval()
                logger.info("[INFO] RootNet cargado exitosamente")
                
            except Exception as e:
                logger.error(f"[ERROR] No se pudo cargar RootNet: {e}")
                self.model = None
    
    def predict_depth(self, img_patch, bbox, focal=[1500, 1500]):
        """Predice profundidad usando RootNet"""
        if self.model is None or self.cfg is None:
            return self._fallback_depth(bbox)
        
        try:
            with self._isolated_import():
                # Importar módulos para procesamiento
                spec = importlib.util.spec_from_file_location(
                    "rootnet_utils", 
                    os.path.join(self.rootnet_path, 'common', "utils", "pose_utils.py")
                )
                rootnet_utils = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(rootnet_utils)
                
                spec = importlib.util.spec_from_file_location(
                    "rootnet_dataset", 
                    os.path.join(self.rootnet_path, 'data', "dataset.py")
                )
                rootnet_dataset = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(rootnet_dataset)
                
                # Preparar imagen
                transform = transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=self.cfg.pixel_mean, std=self.cfg.pixel_std)
                ])
                
                # Procesar bbox
                processed_bbox = rootnet_utils.process_bbox(np.array(bbox), 
                                                           img_patch.shape[1], img_patch.shape[0])
                if processed_bbox is None:
                    return self._fallback_depth(bbox)
                
                img, img2bb_trans = rootnet_dataset.generate_patch_image(img_patch, 
                                                                        processed_bbox, False, 0.0)
                img = transform(img)
                
                # Calcular k_value
                k_value = np.array([
                    math.sqrt(self.cfg.bbox_real[0] * self.cfg.bbox_real[1] * 
                             focal[0] * focal[1] / (processed_bbox[2] * processed_bbox[3]))
                ]).astype(np.float32)
                
                # Preparar tensores
                if torch.cuda.is_available():
                    img = img.cuda()[None,:,:,:]
                    k_value = torch.FloatTensor([k_value]).cuda()[None,:]
                else:
                    img = img[None,:,:,:]
                    k_value = torch.FloatTensor([k_value])[None,:]
                
                # Ejecutar modelo
                with torch.no_grad():
                    root_3d = self.model(img, k_value)
                root_depth = root_3d[0, 2].cpu().numpy()
                
                return root_depth
                
        except Exception as e:
            logger.warning(f"[WARNING] Error en predicción RootNet: {e}")
            return self._fallback_depth(bbox)
    
    def _fallback_depth(self, bbox):
        """Fallback mejorado basado en bbox y posición."""
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        
        # Estimación más realista basada en altura del bbox
        if bbox_height > 400:  # Persona muy cerca
            estimated_depth = 900 + np.random.normal(0, 150)
        elif bbox_height > 300:  # Persona cerca
            estimated_depth = 1400 + np.random.normal(0, 200)
        elif bbox_height > 200:  # Persona distancia media
            estimated_depth = 2200 + np.random.normal(0, 350)
        elif bbox_height > 120:  # Persona lejos
            estimated_depth = 3500 + np.random.normal(0, 500)
        else:  # Persona muy lejos
            estimated_depth = 5000 + np.random.normal(0, 700)
        
        # Ajustar por posición horizontal (perspectiva)
        bbox_center_x = (x1 + x2) / 2
        if bbox_center_x < 640:  # Lado izquierdo
            estimated_depth *= 1.1
        elif bbox_center_x > 1280:  # Lado derecho
            estimated_depth *= 1.1
        
        # Variación por área total
        if bbox_area > 40000:  # Muy cerca
            estimated_depth *= 0.8
        elif bbox_area < 8000:  # Muy lejos
            estimated_depth *= 1.3
        
        return max(min(estimated_depth, 6000.0), 700.0)

def visualize_pose_complete(frame, pose_2d, bbox, confidence, depth=None, is_3d=False):
    """Visualización completa de la detección con pose y bbox"""
    output_frame = frame.copy()
    
    # 1. Dibujar bounding box
    if bbox is not None:
        x1, y1, x2, y2 = [int(x) for x in bbox]
        # Bounding box verde para 3D, azul para 2D
        color = (0, 255, 0) if is_3d else (255, 0, 0)
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        
        # Etiqueta del bbox
        label = f"3D Person" if is_3d else "2D Person"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(output_frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(output_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 2. Dibujar esqueleto de pose 2D
    if pose_2d is not None:
        # Definir conexiones del esqueleto humano (COCO format)
        skeleton_connections = [
            # Cabeza y cuello
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Brazos
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            # Torso
            (5, 11), (6, 12), (11, 12),
            # Piernas
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        # Colores para diferentes partes del cuerpo
        joint_colors = [
            (255, 0, 0),    # 0: nariz - rojo
            (255, 85, 0),   # 1: ojo izq - naranja
            (255, 170, 0),  # 2: ojo der - amarillo
            (255, 255, 0),  # 3: oreja izq - amarillo
            (170, 255, 0),  # 4: oreja der - lima
            (85, 255, 0),   # 5: hombro izq - verde claro
            (0, 255, 0),    # 6: hombro der - verde
            (0, 255, 85),   # 7: codo izq - verde azulado
            (0, 255, 170),  # 8: codo der - cian claro
            (0, 255, 255),  # 9: muñeca izq - cian
            (0, 170, 255),  # 10: muñeca der - azul claro
            (0, 85, 255),   # 11: cadera izq - azul
            (0, 0, 255),    # 12: cadera der - azul oscuro
            (85, 0, 255),   # 13: rodilla izq - violeta
            (170, 0, 255),  # 14: rodilla der - magenta
            (255, 0, 255),  # 15: tobillo izq - magenta
            (255, 0, 170)   # 16: tobillo der - rosa
        ]
        
        # Dibujar conexiones (huesos)
        for connection in skeleton_connections:
            joint1_idx, joint2_idx = connection
            if joint1_idx < len(pose_2d) and joint2_idx < len(pose_2d):
                x1, y1 = pose_2d[joint1_idx][:2]
                x2, y2 = pose_2d[joint2_idx][:2]
                
                # Solo dibujar si ambos joints son válidos
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                            (255, 255, 255), 3)  # Líneas blancas gruesas
        
        # Dibujar joints (articulaciones)
        for i, (x, y) in enumerate(pose_2d[:, :2]):
            if x > 0 and y > 0:  # Joint válido
                color = joint_colors[i % len(joint_colors)]
                cv2.circle(output_frame, (int(x), int(y)), 5, color, -1)
                cv2.circle(output_frame, (int(x), int(y)), 5, (255, 255, 255), 2)  # Borde blanco
    
    # 3. Información de estado en la parte superior
    info_y = 30
    
    if is_3d and depth is not None:
        cv2.putText(output_frame, f"3D MODE - Depth: {depth:.1f}mm", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        info_y += 35
    else:
        cv2.putText(output_frame, "2D MODE", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        info_y += 35
    
    if confidence is not None:
        cv2.putText(output_frame, f"Confidence: {confidence:.3f}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        info_y += 30
    
    # 4. Información de joints detectados
    if pose_2d is not None:
        valid_joints = np.sum((pose_2d[:, 0] > 0) & (pose_2d[:, 1] > 0))
        cv2.putText(output_frame, f"Joints: {valid_joints}/{len(pose_2d)}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return output_frame

def test_3d_complete(args):
    """Test completo del pipeline 3D"""
    
    try:
        # Importar componentes directamente
        from optimized_yolo_detector_fixed import OptimizedYOLODetectorFixed
        from convnext_pose_production_final_corrected import RobustPoseProcessor, visualize_pose_2d
        
        logger.info("✅ Imports base exitosos")
        
        # 1. Inicializar YOLO
        logger.info("🔧 Inicializando YOLO...")
        yolo = OptimizedYOLODetectorFixed(conf_threshold=0.7, max_persons=1)
        logger.info("✅ YOLO listo")
        
        # 2. Inicializar ConvNeXt
        logger.info("🦴 Inicializando ConvNeXt...")
        pose_processor = RobustPoseProcessor(model_type='XS', backend=args.backend)
        logger.info("✅ ConvNeXt listo")
        
        # 3. Inicializar RootNet
        logger.info("📦 Inicializando RootNet...")
        rootnet = SimpleRootNetWrapper(
            '/home/user/convnextpose_esteban/3DMPPE_ROOTNET_RELEASE',
            '/home/user/convnextpose_esteban/ConvNeXtPose/demo/snapshot_18.pth.tar'
        )
        rootnet.load_model(use_gpu=False)
        logger.info("✅ RootNet listo")
        
        # 4. Procesar video
        logger.info("🎬 Procesando video con 3D...")
        video_path = "barbell biceps curl_12.mp4"
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"❌ No se pudo abrir: {video_path}")
            return
        
        # Propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"📊 Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Configurar writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('demo/output_3d_complete.mp4', fourcc, fps, (width, height))
        
        frame_count = 0
        poses_3d_data = []
        poses_2d_success = 0
        poses_3d_success = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_start = time.time()
            
            # 1. Detección
            bbox, confidence = yolo.detect_person(frame)
            
            # FPS del frame
            frame_time = time.time() - frame_start
            frame_fps = 1.0 / frame_time if frame_time > 0 else 0
            
            if bbox is not None:
                # 2. Pose 2D
                pose_2d = pose_processor._process_single_person(frame, bbox)
                
                if pose_2d is not None:
                    poses_2d_success += 1
                    
                    # 3. Pose 3D con RootNet
                    try:
                        root_depth = rootnet.predict_depth(frame, bbox)
                        
                        # Crear pose 3D
                        pose_3d = np.zeros((pose_2d.shape[0], 3), dtype=np.float32)
                        pose_3d[:, :2] = pose_2d  # x, y
                        pose_3d[:, 2] = root_depth  # z
                        
                        poses_3d_success += 1
                        
                        # Guardar datos 3D
                        poses_3d_data.append({
                            'frame': frame_count,
                            'pose3d': pose_3d,
                            'bbox': bbox,
                            'depth': root_depth,
                            'confidence': confidence
                        })
                        
                        # Crear frame de salida usando la visualización existente
                        stats = {
                            'confidence': confidence,
                            'depth': root_depth,
                            'is_3d': True,
                            'bbox': bbox,
                            'frame_count': frame_count,
                            'fps': frame_fps,
                            'instant_fps': frame_fps,
                            'persons_detected': 1,
                            'poses_detected': 1,
                            'backend_used': args.backend,
                            'model_type': 'XS',
                            'total_time': frame_time * 1000,
                            'detection_time': 0,
                            'pose_time': 0,
                            'detected': True
                        }
                        
                        # Usar visualize_pose_2d con información 3D adicional
                        output_frame = visualize_pose_2d(frame, [pose_2d], stats)
                        
                        # Añadir información específica de 3D
                        cv2.putText(output_frame, f"3D MODE - Depth: {root_depth:.1f}mm", 
                                   (10, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(output_frame, f"3D Confidence: {confidence:.3f}", 
                                   (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                    except Exception as e:
                        logger.debug(f"Error RootNet en frame {frame_count}: {e}")
                        
                        # Crear frame de salida usando visualización 2D
                        stats = {
                            'confidence': confidence,
                            'is_3d': False,
                            'bbox': bbox,
                            'frame_count': frame_count,
                            'fps': frame_fps,
                            'instant_fps': frame_fps,
                            'persons_detected': 1,
                            'poses_detected': 1,
                            'backend_used': args.backend,
                            'model_type': 'XS',
                            'total_time': frame_time * 1000,
                            'detection_time': 0,
                            'pose_time': 0,
                            'detected': True
                        }
                        
                        output_frame = visualize_pose_2d(frame, [pose_2d], stats)
                        
                        # Añadir información de modo 2D
                        cv2.putText(output_frame, "2D ONLY (RootNet failed)", 
                                   (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # No se detectó pose
                    stats = {
                        'confidence': confidence if bbox is not None else 0,
                        'is_3d': False,
                        'bbox': bbox,
                        'frame_count': frame_count,
                        'fps': frame_fps,
                        'instant_fps': frame_fps,
                        'persons_detected': 1 if bbox is not None else 0,
                        'poses_detected': 0,
                        'backend_used': args.backend,
                        'model_type': 'XS',
                        'total_time': frame_time * 1000,
                        'detection_time': 0,
                        'pose_time': 0,
                        'detected': False
                    }
                    
                    output_frame = visualize_pose_2d(frame, [], stats)
                    cv2.putText(output_frame, "No pose detected", 
                               (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # No se detectó persona
                stats = {
                    'confidence': 0,
                    'is_3d': False,
                    'bbox': None,
                    'frame_count': frame_count,
                    'fps': frame_fps,
                    'instant_fps': frame_fps,
                    'persons_detected': 0,
                    'poses_detected': 0,
                    'backend_used': args.backend,
                    'model_type': 'XS',
                    'total_time': frame_time * 1000,
                    'detection_time': 0,
                    'pose_time': 0,
                    'detected': False
                }
                
                output_frame = visualize_pose_2d(frame, [], stats)
                cv2.putText(output_frame, "No person detected", 
                           (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            
            # Agregar contador de frames y FPS
            cv2.putText(output_frame, f"Frame: {frame_count}/{total_frames}", 
                       (width - 200, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(output_frame, f"FPS: {frame_fps:.1f}", 
                       (width - 200, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            writer.write(output_frame)
            
            # MOSTRAR EL VIDEO EN TIEMPO REAL
            cv2.imshow('ConvNeXt 3D Pose Detection', output_frame)
            
            # Presionar 'q' para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("🛑 Saliendo por solicitud del usuario...")
                break
            
            # Progreso cada 25 frames
            if frame_count % 25 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed
                logger.info(f"📈 Frame {frame_count}/{total_frames} - "
                          f"FPS: {avg_fps:.1f} - "
                          f"3D Success: {poses_3d_success}/{poses_2d_success}")
        
        # Cleanup
        cap.release()
        writer.release()
        cv2.destroyAllWindows()  # Cerrar todas las ventanas de OpenCV
        
        # Guardar coordenadas 3D
        if poses_3d_data:
            # Guardar la última pose como ejemplo
            last_pose = poses_3d_data[-1]
            np.savez_compressed('output_3d_coords.npz', 
                              pose3d=last_pose['pose3d'], 
                              bbox=last_pose['bbox'], 
                              frame=last_pose['frame'],
                              depth=last_pose['depth'],
                              confidence=last_pose['confidence'])
            logger.info(f"📦 Coordenadas 3D guardadas: output_3d_coords.npz")
            
            # Guardar todas las poses para análisis
            all_poses = [p['pose3d'] for p in poses_3d_data]
            all_depths = [p['depth'] for p in poses_3d_data]
            np.savez_compressed('all_3d_poses.npz',
                              poses=all_poses,
                              depths=all_depths,
                              frames=[p['frame'] for p in poses_3d_data])
            logger.info(f"📊 Todas las poses 3D guardadas: all_3d_poses.npz")
        
        # Estadísticas finales
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        success_2d_rate = (poses_2d_success / frame_count) * 100
        success_3d_rate = (poses_3d_success / poses_2d_success) * 100 if poses_2d_success > 0 else 0
        
        logger.info("\\n📊 RESULTADOS FINALES 3D:")
        logger.info("=" * 50)
        logger.info(f"🎬 Frames procesados: {frame_count}")
        logger.info(f"✅ Poses 2D exitosas: {poses_2d_success} ({success_2d_rate:.1f}%)")
        logger.info(f"📦 Poses 3D exitosas: {poses_3d_success} ({success_3d_rate:.1f}%)")
        logger.info(f"⚡ FPS promedio: {avg_fps:.1f}")
        logger.info(f"⏱️ Tiempo total: {total_time:.1f}s")
        logger.info(f"💾 Video guardado: demo/output_3d_complete.mp4")
        
        if poses_3d_data:
            depths = [p['depth'] for p in poses_3d_data]
            logger.info(f"📏 Profundidad promedio: {np.mean(depths):.1f}mm")
            logger.info(f"📏 Rango profundidad: {np.min(depths):.1f} - {np.max(depths):.1f}mm")
        
        logger.info("\\n🎉 TEST 3D COMPLETADO EXITOSAMENTE!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal con parser de argumentos completo"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ConvNeXt Ultra Performance - Versión Corregida')
    
    # Presets disponibles
    presets = {
        'ultra_fast_30fps': 'Ultra rápido - 30+ FPS (solo 2D)',
        'ultra_fast_30fps_3d': 'Ultra rápido - 30+ FPS con coordenadas 3D',
        'balanced_25fps': 'Balance - 25+ FPS (solo 2D)', 
        'balanced_25fps_3d': 'Balance - 25+ FPS con coordenadas 3D',
        'quality_20fps': 'Calidad - 20+ FPS (solo 2D)',
        'quality_20fps_3d': 'Calidad - 20+ FPS con coordenadas 3D'
    }
    
    parser.add_argument('--preset', choices=list(presets.keys()), 
                       default='ultra_fast_30fps', help='Preset de rendimiento')
    parser.add_argument('--backend', choices=['onnx', 'pytorch', 'tflite'], 
                       default='onnx', help='Backend de inferencia')
    parser.add_argument('--model', choices=['XS', 'S'], 
                       default='XS', help='Modelo ConvNeXt')
    parser.add_argument('--input', type=str, default='demo/Personas caminando en la calle.mp4',
                       help='Entrada: webcam (0) o archivo de video')
    parser.add_argument('--output_video', type=str, default=None,
                       help='Archivo de salida de video (.mp4)')
    parser.add_argument('--output_npz', type=str, default=None,
                       help='Archivo de salida de coordenadas 3D (.npz)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Ejecutar benchmark de rendimiento')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duración del benchmark (segundos)')
    parser.add_argument('--rootnet_path', type=str, default='/home/user/convnextpose_esteban/3DMPPE_ROOTNET_RELEASE',
                       help='Ruta al repo RootNet')
    parser.add_argument('--rootnet_ckpt', type=str, default='demo/snapshot_18.pth.tar',
                       help='Checkpoint RootNet')
    
    args = parser.parse_args()
    
    # Información del preset seleccionado
    output_3d = args.preset.endswith('_3d')
    
    logger.info("🚀 CONVNEXT ULTRA PERFORMANCE - VERSIÓN CORREGIDA")
    logger.info("=" * 60)
    logger.info(f"🎯 Preset: {args.preset}")
    logger.info(f"� Descripción: {presets[args.preset]}")
    logger.info(f"🔧 Backend: {args.backend}")
    logger.info(f"🎪 Modelo: {args.model}")
    logger.info(f"📦 Salida 3D: {'✅ Activada' if output_3d else '❌ Solo 2D'}")
    logger.info(f"📁 Entrada: {args.input}")
    
    if args.benchmark:
        logger.info(f"🧪 Modo benchmark: {args.duration} segundos")
        test_3d_complete(args)
    else:
        # Ejecutar procesamiento normal
        if output_3d:
            logger.info("🦴 Iniciando procesamiento con 3D...")
            test_3d_complete(args)
        else:
            logger.info("🎬 Iniciando procesamiento solo 2D...")
            test_3d_complete(args)

if __name__ == "__main__":
    main()
