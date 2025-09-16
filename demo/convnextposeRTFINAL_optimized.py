#!/usr/bin/env python3

"""
ConvNeXt Ultra Fast - Arquitectura Modular Optimizada
=====================================================
Basado en la arquitectura de main.py pero optimizado para máximo rendimiento
"""

import os
import sys
import cv2
import numpy as np
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# Agregar path del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedProcessor:
    """Procesador optimizado basado en la arquitectura modular exitosa"""
    
    def __init__(self, preset: str = 'ultra_fast', backend: str = 'tflite', model: str = 'XS'):
        # Configuración basada en main.py que funciona
        presets = {
            'ultra_fast': {
                'target_fps': 15.0,
                'frame_skip': 2,
                'detection_freq': 3,
                'thread_count': 2
            },
            'ultra_fast_30fps_3d': {
                'target_fps': 20.0,
                'frame_skip': 1,
                'detection_freq': 2,
                'thread_count': 3
            },
            'balanced': {
                'target_fps': 12.0,
                'frame_skip': 2,
                'detection_freq': 3,
                'thread_count': 2
            }
        }
        
        base_config = presets.get(preset, presets['ultra_fast'])
        self.config = {
            **base_config,
            'max_persons': 2,
            'enable_threading': True,
            'backend': backend,
            'model': model
        }
        
        # Threading controlado (igual que main.py exitoso)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config['thread_count'])
        
        # Stats simples (sin overhead)
        self.frame_count = 0
        self.skip_count = 0
        self.last_poses = []
        self.processing_times = deque(maxlen=30)
        self.last_detection_frame = 0
        
        # Inicializar componentes
        self._initialize_components()
        
        logger.info("✅ Optimized Processor initialized")
        logger.info(f"   Preset: {preset}")
        logger.info(f"   Backend: {self.config['backend']}")
        logger.info(f"   Model: {self.config['model']}")
        logger.info(f"   Target FPS: {self.config['target_fps']}")
        logger.info(f"   Threading: {self.config['enable_threading']} ({self.config['thread_count']} workers)")
    
    def _initialize_components(self):
        """Inicializar componentes igual que el main.py exitoso"""
        try:
            logger.info("🔧 Initializing components...")
            
            # Importar componentes directos
            from optimized_yolo_detector_fixed import OptimizedYOLODetectorFixed
            from convnext_pose_production_final_corrected import RobustPoseProcessor
            
            # YOLO
            logger.info("🎯 Loading YOLO...")
            self.yolo = OptimizedYOLODetectorFixed(conf_threshold=0.7, max_persons=1)
            
            # ConvNeXt
            logger.info("🦴 Loading ConvNeXt...")
            self.pose_processor = RobustPoseProcessor(
                model_type=self.config['model'], 
                backend=self.config['backend']
            )
            
            # RootNet TFLite optimizado
            logger.info("📦 Loading RootNet TFLite...")
            try:
                from rootnet_tflite_wrapper import RootNetTFLiteWrapper
                self.rootnet = RootNetTFLiteWrapper(model_variant='size')  # Usar modelo 'size' (más rápido)
                if self.rootnet.backbone_available:
                    logger.info("✅ RootNet TFLite cargado exitosamente")
                else:
                    logger.warning("⚠️ RootNet TFLite no disponible, usando fallback")
                    self.rootnet = SimpleRootNetWrapper()
            except ImportError:
                logger.warning("⚠️ RootNet TFLite wrapper no encontrado, usando fallback")
                self.rootnet = SimpleRootNetWrapper()
            
            logger.info("✅ All components initialized successfully")
            
        except ImportError as e:
            logger.error(f"❌ Import failed: {e}")
            logger.error("Make sure you're in the correct directory with all required modules")
            raise
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def _should_skip_frame(self) -> bool:
        """Skip inteligente como main.py (NO como tu implementación compleja)"""
        self.skip_count += 1
        should_skip = (self.skip_count % (self.config['frame_skip'] + 1)) != 0
        return should_skip
    
    def _should_detect(self) -> bool:
        """Detección inteligente cada N frames (clave del rendimiento)"""
        frames_since_detection = self.frame_count - self.last_detection_frame
        return frames_since_detection >= self.config['detection_freq']
    
    def _process_single_person(self, frame: np.ndarray, bbox: List[int]) -> Optional[Dict[str, Any]]:
        """Procesar persona individual (método directo como main.py)"""
        try:
            # Pose 2D
            pose_2d = self.pose_processor._process_single_person(frame, bbox)
            if pose_2d is None:
                return None
            
            # Depth estimation usando RootNet TFLite real
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            
            depth = None
            if crop.size > 0:
                # Usar RootNet TFLite si está disponible
                if hasattr(self.rootnet, 'backbone_available') and self.rootnet.backbone_available:
                    depth = self.rootnet.predict_depth(crop, bbox, use_analysis=True)
                else:
                    # Fallback a estimación simple
                    depth = self.rootnet.predict_depth(crop, bbox)
            
            pose_3d = None
            if depth:
                pose_3d = np.zeros((pose_2d.shape[0], 3))
                pose_3d[:, :2] = pose_2d[:, :2]
                pose_3d[:, 2] = depth
            
            return {
                'bbox': bbox,
                'pose_2d': pose_2d,
                'pose_3d': pose_3d,
                'depth': depth
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Single person processing failed: {e}")
            return None
    
    def _estimate_depth_simple(self, bbox: List[int]) -> float:
        """Estimación de profundidad simple basada en bbox"""
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        
        # Estimación basada en altura del bbox
        if bbox_height > 400:
            return 800.0
        elif bbox_height > 300:
            return 1200.0
        elif bbox_height > 200:
            return 1800.0
        elif bbox_height > 120:
            return 2500.0
        else:
            return 3000.0
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Procesar frame con arquitectura modular optimizada"""
        start_time = time.time()
        self.frame_count += 1
        
        # Skip inteligente (como main.py)
        if self._should_skip_frame():
            return self.last_poses, {
                'frame_count': self.frame_count,
                'fps': 1.0 / (time.time() - start_time) if time.time() - start_time > 0 else 0,
                'skipped': True,
                'poses_detected': len(self.last_poses)
            }
        
        results = []
        
        # Detección inteligente (NO en cada frame como tu implementación)
        persons = []
        if self._should_detect():
            bbox, confidence = self.yolo.detect_person(frame)
            if bbox is not None:
                persons = [bbox]
                self.last_detection_frame = self.frame_count
        else:
            # Reusar última detección con tracking simple
            if self.last_poses:
                persons = [result['bbox'] for result in self.last_poses if result['bbox'] is not None]
        
        # Procesamiento de poses
        if persons:
            if self.config['enable_threading'] and len(persons) > 1:
                # Threading para múltiples personas
                futures = []
                for bbox in persons[:self.config['max_persons']]:
                    future = self.thread_pool.submit(self._process_single_person, frame, bbox)
                    futures.append(future)
                
                for future in futures:
                    try:
                        result = future.result(timeout=0.2)
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.warning(f"⚠️ Threading failed: {e}")
            else:
                # Single-threaded para una persona (más eficiente)
                for bbox in persons[:self.config['max_persons']]:
                    result = self._process_single_person(frame, bbox)
                    if result:
                        results.append(result)
        
        # Actualizar cache de poses
        self.last_poses = results
        
        # Stats simples
        frame_time = time.time() - start_time
        self.processing_times.append(frame_time)
        avg_fps = 1.0 / (sum(self.processing_times) / len(self.processing_times))
        
        stats = {
            'frame_count': self.frame_count,
            'fps': avg_fps,
            'instant_fps': 1.0 / frame_time if frame_time > 0 else 0,
            'skipped': False,
            'poses_detected': len(results),
            'processing_time_ms': frame_time * 1000
        }
        
        return results, stats

class SimpleRootNetWrapper:
    """Wrapper simple para RootNet (fallback básico)"""
    
    def __init__(self):
        self.available = False
        self.backbone_available = False  # Compatibilidad con TFLite wrapper
        
    def predict_depth(self, img_patch, bbox, focal=[1500, 1500], use_analysis=False):
        """Fallback simple de profundidad con interfaz compatible"""
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        
        # Estimación basada en altura
        if bbox_height > 400:
            return 800.0
        elif bbox_height > 300:
            return 1200.0
        elif bbox_height > 200:
            return 1800.0
        elif bbox_height > 120:
            return 2500.0
        else:
            return 3000.0

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
        # Definir conexiones del esqueleto humano (ConvNeXtPose format para 18 joints)
        # Basado en main/summary.py del proyecto: skeleton para joint_num == 18
        skeleton_connections = [
            (0, 7), (7, 8), (8, 9), (9, 10),          # head to hands
            (8, 11), (11, 12), (12, 13),              # torso to right leg  
            (8, 14), (14, 15), (15, 16),              # torso to left leg
            (0, 1), (1, 2), (2, 3),                   # head to right arm
            (0, 4), (4, 5), (5, 6)                    # head to left arm
        ]
        
        # Colores para diferentes partes del cuerpo (ConvNeXtPose joints)
        # Orden según main/summary.py: ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 
        # 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 
        # 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe') - usando primeros 18
        joint_colors = [
            (255, 0, 0),    # 0: Head_top
            (255, 85, 0),   # 1: Thorax  
            (255, 170, 0),  # 2: R_Shoulder
            (255, 255, 0),  # 3: R_Elbow
            (170, 255, 0),  # 4: R_Wrist
            (85, 255, 0),   # 5: L_Shoulder
            (0, 255, 0),    # 6: L_Elbow
            (0, 255, 85),   # 7: L_Wrist
            (0, 255, 170),  # 8: R_Hip
            (0, 255, 255),  # 9: R_Knee
            (0, 170, 255),  # 10: R_Ankle
            (0, 85, 255),   # 11: L_Hip
            (0, 0, 255),    # 12: L_Knee
            (85, 0, 255),   # 13: L_Ankle
            (170, 0, 255),  # 14: Pelvis
            (255, 0, 255),  # 15: Spine
            (255, 0, 170),  # 16: Head
            (255, 128, 128) # 17: R_Hand (si se usa 18 joints)
        ]
        
        # Dibujar conexiones (huesos)
        for connection in skeleton_connections:
            if len(connection) == 2:
                pt1_idx, pt2_idx = connection
                if pt1_idx < len(pose_2d) and pt2_idx < len(pose_2d):
                    pt1 = pose_2d[pt1_idx][:2]
                    pt2 = pose_2d[pt2_idx][:2]
                    if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                        cv2.line(output_frame, (int(pt1[0]), int(pt1[1])), 
                                (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)
        
        # Dibujar joints (articulaciones)
        for i, (x, y) in enumerate(pose_2d[:, :2]):
            if x > 0 and y > 0:
                color = joint_colors[i % len(joint_colors)]
                cv2.circle(output_frame, (int(x), int(y)), 4, color, -1)
    
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

def test_optimized_pipeline(args):
    """Test del pipeline optimizado"""
    
    logger.info("🚀 TESTING OPTIMIZED PIPELINE")
    logger.info("=" * 50)
    logger.info(f"🎯 Preset: {args.preset}")
    logger.info(f"🔧 Backend: {args.backend}")
    logger.info(f"🎪 Model: {args.model}")
    logger.info(f"👁️ Show live: {'Yes' if args.show_live else 'No'}")
    
    try:
        # Inicializar procesador
        processor = OptimizedProcessor(
            preset=args.preset,
            backend=args.backend,
            model=args.model
        )
        
        # Procesar video
        video_path = "barbell biceps curl_12.mp4"
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"❌ Cannot open video: {video_path}")
            return
        
        # Propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"📊 Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        writer = cv2.VideoWriter('output_optimized.mp4', fourcc, fps, (width, height))
        
        frame_count = 0
        poses_detected = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Procesar frame
            results, stats = processor.process_frame(frame)
            
            if results:
                poses_detected += len(results)
            
            # Visualizar
            output_frame = frame.copy()
            
            for result in results:
                pose_2d = result['pose_2d']
                bbox = result['bbox']
                depth = result['depth']
                
                # Visualizar usando función completa
                output_frame = visualize_pose_complete(
                    output_frame, pose_2d, bbox, 
                    confidence=0.9, depth=depth, is_3d=True
                )
            
            # Stats en pantalla
            cv2.putText(output_frame, f"FPS: {stats['fps']:.1f}", 
                       (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output_frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Visualización en tiempo real con imshow (opcional)
            if args.show_live:
                # Redimensionar para visualización si es muy grande
                display_frame = output_frame.copy()
                if display_frame.shape[1] > 1280:  # Si es muy ancho
                    scale = 1280 / display_frame.shape[1]
                    new_width = 1280
                    new_height = int(display_frame.shape[0] * scale)
                    display_frame = cv2.resize(display_frame, (new_width, new_height))
                
                cv2.imshow('ConvNeXt Pose - Real Time', display_frame)
                
                # Control de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("⚠️ User requested quit (q pressed)")
                    break
                elif key == ord('p'):
                    logger.info("⏸️ Paused - Press any key to continue")
                    cv2.waitKey(0)
                elif key == ord('s'):
                    # Guardar frame actual
                    save_path = f'frame_capture_{frame_count}.jpg'
                    cv2.imwrite(save_path, output_frame)
                    logger.info(f"📸 Frame saved: {save_path}")
            
            writer.write(output_frame)
            
            # Progress
            if frame_count % 25 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed
                logger.info(f"📈 Frame {frame_count}/{total_frames} - FPS: {avg_fps:.1f}")
        
        # Cleanup
        cap.release()
        writer.release()
        
        # Cerrar ventanas de visualización si se usaron
        if args.show_live:
            cv2.destroyAllWindows()
        
        # Resultados finales
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        
        logger.info("\\n📊 RESULTADOS OPTIMIZADOS:")
        logger.info("=" * 40)
        logger.info(f"🎬 Frames procesados: {frame_count}")
        logger.info(f"✅ Poses detectadas: {poses_detected}")
        logger.info(f"⚡ FPS promedio: {avg_fps:.1f}")
        logger.info(f"⏱️ Tiempo total: {total_time:.1f}s")
        logger.info(f"💾 Video guardado: output_optimized.mp4")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal con parser de argumentos"""
    parser = argparse.ArgumentParser(description='ConvNeXt Optimized Pipeline Test')
    
    parser.add_argument('--preset', choices=['ultra_fast', 'ultra_fast_30fps_3d', 'balanced'], 
                       default='ultra_fast_30fps_3d', help='Performance preset')
    parser.add_argument('--backend', choices=['pytorch', 'onnx', 'tflite'], 
                       default='tflite', help='Inference backend')
    parser.add_argument('--model', choices=['XS', 'S'], 
                       default='XS', help='Model size')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark mode')
    parser.add_argument('--duration', type=int, default=30,
                       help='Benchmark duration (not used in this implementation)')
    parser.add_argument('--show_live', action='store_true',
                       help='Show live visualization with cv2.imshow (press q to quit, p to pause, s to save frame)')
    
    args = parser.parse_args()
    
    logger.info("🚀 CONVNEXT OPTIMIZED PIPELINE")
    logger.info("=" * 40)
    logger.info(f"🎯 Preset: {args.preset}")
    logger.info(f"🔧 Backend: {args.backend}")
    logger.info(f"🎪 Model: {args.model}")
    logger.info(f"🧪 Benchmark: {'Yes' if args.benchmark else 'No'}")
    logger.info(f"👁️ Live View: {'Yes' if args.show_live else 'No'}")
    
    if args.show_live:
        logger.info("🎮 Live View Controls:")
        logger.info("   q: Quit")
        logger.info("   p: Pause/Resume") 
        logger.info("   s: Save current frame")
    
    success = test_optimized_pipeline(args)
    
    if success:
        logger.info("\\n🎉 TEST COMPLETED SUCCESSFULLY!")
    else:
        logger.error("\\n❌ TEST FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()