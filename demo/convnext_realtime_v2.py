#!/usr/bin/env python3
"""
run_pose_mixed.py - VERSIÓN CORREGIDA

– Detección de persona con YOLOv5 INT8-ONNX (DeepSparse)
– Estimación de pose 3D con ConvNeXtPose desde checkpoint PyTorch (.pth)
– Visualización en tiempo real con FPS en pantalla y latencia de ConvNeXtPose cada 30 cuadros
"""

import argparse
import time
import sys
import inspect
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from deepsparse import compile_model
from deepsparse.pipeline import Pipeline
from root_wrapper import RootNetWrapper
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.extend([
    str(PROJECT_ROOT / 'main'),
    str(PROJECT_ROOT / 'data'),
    str(PROJECT_ROOT / 'common')
])

# ------ IMPORTACIÓN DE CONVNEXTPOSE (usar rutas relativas al proyecto) ------
from config import cfg
from model import get_pose_net
from dataset import generate_patch_image
import utils.pose_utils as pose_utils
import utils.vis as vis_utils

# ----------------------------------------------------------------------------

def letterbox(image: np.ndarray, new_size=(640, 640), color=(114, 114, 114)):
    """
    Redimensiona con letterbox manteniendo aspecto: 
    devuelve imagen escalada y valores (scale, pad_left, pad_top) para revertir coordenadas.
    """
    h, w = image.shape[:2]
    new_h, new_w = new_size
    scale = min(new_w / w, new_h / h)
    resized_w, resized_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (new_h - resized_h) // 2
    pad_bottom = new_h - resized_h - pad_top
    pad_left = (new_w - resized_w) // 2
    pad_right = new_w - resized_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=color
    )
    return padded, scale, pad_left, pad_top

def draw_skeleton(img: np.ndarray, joints: np.ndarray, skeleton, offset=(0, 0), color=(0, 255, 0)):
    """
    Dibuja articulaciones y líneas de esqueleto en img.
    - joints: array (J, 2) con coordenadas (x, y) en el recorte original
    - offset: esquina superior izquierda del recorte en la imagen principal
    """
    for (i, j) in skeleton:
        if i < len(joints) and j < len(joints):
            x1, y1 = int(joints[i][0] + offset[0]), int(joints[i][1] + offset[1])
            x2, y2 = int(joints[j][0] + offset[0]), int(joints[j][1] + offset[1])
            cv2.line(img, (x1, y1), (x2, y2), color, 2)
    for (x, y) in joints:
        cv2.circle(img, (int(x + offset[0]), int(y + offset[1])), 3, color, -1)

def load_pose_model(pth_path: str, joint_num=18, use_cuda=False):
    """
    Carga ConvNeXtPose desde archivo .pth (state dict).
    Devuelve el modelo PyTorch en modo eval.
    """
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    # Crear arquitectura
    model = get_pose_net(cfg, is_train=False, joint_num=joint_num)
    # Cargar pesos
    state = torch.load(pth_path, map_location=device)
    sd = state.get('network', state)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    return model, device

def xywh2xyxy(box):
    """Convierte [cx, cy, w, h] a [x1, y1, x2, y2]."""
    cx, cy, w, h = box
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]

def correct_skeleton_orientation(coords):
    """Corrige la orientación del esqueleto según la posición esperada."""
    # Asegurarse de tener suficientes puntos para calcular
    if coords.shape[0] < 15:
        return coords
    
    # Identificar explícitamente puntos clave del esqueleto
    # Columna vertebral: de pelvis/cadera (14) a cuello/torso (1)
    spine_bottom = coords[14]  # Pelvis
    spine_top = coords[1]      # Torso
    
    spine_vector = spine_top - spine_bottom
    
    # Calcular ángulo - aquí está la corrección crucial
    # Un humano de pie tendrá vector vertical (0, -1) en coordenadas de imagen
    # atan2 con estos argumentos da el ángulo correcto respecto a la vertical
    desired_angle = 0  # deseamos que la columna sea vertical
    current_angle = np.arctan2(spine_vector[0], -spine_vector[1])
    rotation_angle = desired_angle - current_angle
    
    # Solo rotar si el ángulo es significativo (más de 5 grados)
    if abs(rotation_angle) > 0.09:  # ~5 grados en radianes
        # Calcular centro para la rotación (centroide del cuerpo)
        # Usar solo torso para mayor estabilidad
        torso_points = coords[[1, 14, 15, 8, 11]]  # torso, pelvis, columna, caderas
        center = np.mean(torso_points, axis=0)
        
        # Matriz de rotación en 2D
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        rot_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        # Aplicar rotación
        centered_coords = coords - center
        rotated_coords = np.dot(centered_coords, rot_matrix.T)
        result = rotated_coords + center
        
        # Debug info
        print(f"Rotando esqueleto {np.degrees(rotation_angle):.1f}° grados")
        
        return result
    
    return coords
def adjust_skeleton_proportions(coords, scale_factor=1.3):
    """
    Versión mejorada del ajuste de proporciones del esqueleto
    """
    if coords.shape[0] < 15:
        return coords
        
    # Usar múltiples puntos de anclaje para mayor estabilidad
    pelvis = coords[14].copy()
    spine = coords[15].copy()  
    
    # Vector que define la dirección vertical ideal del cuerpo
    spine_vector = coords[1] - coords[14]
    spine_length = np.linalg.norm(spine_vector)
    
    # Calcular factor de escala adaptativo basado en las dimensiones del bbox detectado
    # El escalado vertical debe ser ligeramente mayor para compensar la compresión
    vertical_scale = scale_factor * 1.1  # Aumentar escala vertical un 10% adicional
    
    # Crear una matriz de escala no uniforme para estirar más verticalmente
    for i in range(len(coords)):
        # Vector desde pelvis a cada articulación
        vec = coords[i] - pelvis
        
        # Proyección sobre el vector columna
        spine_dir = spine_vector / (spine_length + 1e-6)
        proj_len = np.dot(vec, spine_dir)
        
        # Componente paralela a columna (escalar verticalmente)
        parallel = proj_len * spine_dir
        
        # Componente perpendicular a columna (escalar horizontalmente)
        perp = vec - parallel
        
        # Aplicar escalas diferentes
        scaled_vec = parallel * vertical_scale + perp * scale_factor
        
        # Aplicar la transformación
        coords[i] = pelvis + scaled_vec
    
    # Ajustes adicionales específicos para partes del cuerpo
    # Ensanchar hombros
    shoulder_center = (coords[2] + coords[5]) / 2
    coords[2] = shoulder_center + (coords[2] - shoulder_center) * 1.3
    coords[5] = shoulder_center + (coords[5] - shoulder_center) * 1.3
    
    # Ajustar posición de la cabeza si está muy baja
    neck_to_head = coords[16] - coords[1]
    if np.linalg.norm(neck_to_head) < spine_length * 0.15:
        # Subir la cabeza si está muy cercana al cuello
        coords[16] = coords[1] + neck_to_head * 1.8
        coords[0] = coords[16] + (coords[0] - coords[16]) * 1.2  # Ajustar top de cabeza también
        
    print(f"Proporciones ajustadas con escala vertical={vertical_scale:.2f}, horizontal={scale_factor:.2f}")
    
    return coords

def improved_skeleton_alignment(coords_img, bbox, pose_3d=None):
    """Alineación precisa basada en múltiples puntos de referencia anatómicos"""
    x1, y1, x2, y2 = bbox
    
    # 1. Identificar referencias anatómicas clave
    # Cabeza, cuello, hombros, cadera
    head_idx = 16
    neck_idx = 1
    shoulder_r_idx, shoulder_l_idx = 2, 5
    hip_r_idx, hip_l_idx = 8, 11
    
    # 2. Extraer línea central del cuerpo (más estable)
    central_indices = [16, 1, 15, 14]  # cabeza, cuello, columna, pelvis
    central_points = coords_img[central_indices]
    
    # 3. Calcular proporciones corporales ideales respecto al bbox
    body_height = y2 - y1
    ideal_head_pos_y = y1 + body_height * 0.15  # 15% desde arriba
    ideal_neck_pos_y = y1 + body_height * 0.22  # 22% desde arriba
    ideal_hip_pos_y = y1 + body_height * 0.55   # 55% desde arriba
    
    # 4. Calcular centro horizontal ideal basado en tipo de vista
    hip_width = np.linalg.norm(coords_img[hip_r_idx] - coords_img[hip_l_idx])
    shoulder_width = np.linalg.norm(coords_img[shoulder_r_idx] - coords_img[shoulder_l_idx])
    width_ratio = hip_width / (shoulder_width + 1e-6)
    
    # Determinar si es vista frontal, trasera o lateral
    is_side_view = width_ratio < 0.7 or width_ratio > 1.3
    
    # 5. Aplicar transformaciones específicas según el tipo de vista
    if is_side_view:
        # Vista lateral: anclaje en pelvis + columna
        spine_points = coords_img[[14, 15, 1]]  # pelvis, columna, cuello
        ref_point = np.mean(spine_points, axis=0)
        target_x = (x1 + x2) / 2
    else:
        # Vista frontal: anclaje en centro de hombros y caderas
        torso_center = np.mean(coords_img[[2, 5, 8, 11]], axis=0)
        ref_point = torso_center
        # En vista frontal, alinear con el centro del bbox
        target_x = (x1 + x2) / 2
    
    # 6. Calcular y aplicar offsets con mayor precisión
    # Offset horizontal (más preciso)
    offset_x = target_x - ref_point[0]
    
    # Offset vertical (basado en múltiples puntos)
    head_offset = ideal_head_pos_y - coords_img[head_idx, 1]
    neck_offset = ideal_neck_pos_y - coords_img[neck_idx, 1]
    hip_offset = ideal_hip_pos_y - coords_img[14, 1]  # pelvis
    
    # Usar promedio ponderado con más peso en el cuello (más estable)
    offset_y = (head_offset*0.3 + neck_offset*0.5 + hip_offset*0.2)
    
    # 7. Aplicar transformación con amortiguación para mayor estabilidad
    result = coords_img.copy()
    result[:, 0] += offset_x
    result[:, 1] += offset_y
    
    return result
def estimate_depth_from_bbox(bbox):
    width, height = bbox[2], bbox[3]
    normalized_size = np.sqrt(width * height) / 100.0  # Normalizar al tamaño típico
    estimated_depth = 5000/(normalized_size + 0.1)
    return np.clip(estimated_depth, 3000, 15000)
def main():
    cfg.input_shape = (256, 256)  # Ajustar tamaño de entrada para YOLOv5
    cfg.output_shape = (32, 32)  # Ajustar tamaño de salida para ConvNeXtPose
    cfg.depth_dim = 32
    cfg.bbox_3d_shape = (2000, 2000, 2000)  # Dimensiones del espacio 3D
    parser = argparse.ArgumentParser(
        description="Detección de personas con YOLO INT8-ONNX + DeepSparse "
                    "y estimación de pose 3D con ConvNeXtPose desde checkpoint PyTorch"
    )
    parser.add_argument('--input', type=str, default='0',
                        help='Fuente de video (0=webcam o ruta archivo)')
    parser.add_argument('--yolo-model', type=str, default='zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none',
                        help='Ruta al modelo YOLO ONNX (INT8) para DeepSparse')
    parser.add_argument('--pose-model', type=str, required=True,
                        help='Ruta al checkpoint ConvNeXtPose (.pth)')
    parser.add_argument('--conf-thresh', type=float, default=0.3,
                        help='Umbral de confianza para detección de personas')
    parser.add_argument('--iou-thresh', type=float, default=0.45,
                        help='Umbral de IoU para Non-Maximum Suppression (NMS)')
    args = parser.parse_args()

    # 1) Inicializar motor DeepSparse con modelo YOLOv5 INT8-ONNX
    print(f"[INFO] Cargando YOLO INT8-ONNX con DeepSparse desde: {args.yolo_model}")
    yolo_pipeline = Pipeline.create(task="yolo", model_path=args.yolo_model)
    
    # 2) Cargar ConvNeXtPose desde .pth (PyTorch)
    print(f"[INFO] Cargando ConvNeXtPose desde checkpoint: {args.pose_model}")
    pose_model, device = load_pose_model(args.pose_model, joint_num=18, use_cuda=False)


    # 3) Transformación para ConvNeXtPose
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std),
    ])

    # 4) Definir esqueleto
    skeleton = (
        (0, 16), (16, 1), (1, 15), (15, 14), (14, 8),
        (8, 9), (9, 10), (14, 11), (11, 12), (12, 13),
        (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)
    )

    # 5) Captura de video - CORREGIR FUENTE
    if args.input == '0':
        cap = cv2.VideoCapture(0)  # Webcam
    elif args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        # Para TCP o archivo
        if args.input.startswith('tcp://'):
            cap = cv2.VideoCapture(args.input)
        else:
            cap = cv2.VideoCapture(f"tcp://{args.input}:5000")
    
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir fuente de video: {args.input}")
        return

    frame_count = 0
    pose_latencies = []
    yolo_size = (640, 640)
    every_n = 1  # Procesar cada frame para testing
    last_coords_2d = []
    root_wrapper = RootNetWrapper('/home/fabri/3DMPPE_ROOTNET_RELEASE', '/home/fabri/3DMPPE_ROOTNET_RELEASE/demo/snapshot_18.pth.tar')
    root_wrapper.load_model(use_gpu=False)
    print("[INFO] Iniciando demo en tiempo real. Presione 'q' para salir.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error recibiendo frame o fin del video")
            break
            
        t_frame_start = time.time()
        
        if frame_count % every_n == 0:
            # ===== CORRECCIÓN CRÍTICA: Usar directamente el frame sin conversión RGB =====
            # YOLOv5 espera BGR (formato de OpenCV)
            img_lb, scale, pad_left, pad_top = letterbox(frame, new_size=yolo_size)
            
            # Debug: mostrar frame de entrada
            if frame_count % 30 == 0:
                print(f"Frame shape: {frame.shape}, Letterbox shape: {img_lb.shape}")
                cv2.imshow("Input to YOLO", cv2.resize(img_lb, (320, 320)))

            # CORRECCIÓN: Pasar imagen directamente sin normalización previa
            # DeepSparse YOLO pipeline maneja la normalización internamente
            try:
                outputs = yolo_pipeline(
                    images=img_lb,  # Pasar imagen BGR directamente
                    score_threshold=args.conf_thresh,
                    nms_threshold=args.iou_thresh,
                )
                
                if isinstance(outputs, list) and len(outputs) > 0:
                    outputs = outputs[0]
                
                print(f"outputs: {outputs}")
                
                # Verificar si hay detecciones
                if hasattr(outputs, 'boxes') and len(outputs.boxes) > 0:
                    # Aplanar listas anidadas de DeepSparse
                    boxes_640 = np.array(outputs.boxes[0])  # Shape: (N, 4)
                    scores_out = np.array(outputs.scores[0])  # Shape: (N,)
                    labels_out = np.array(outputs.labels[0])  # Shape: (N,)
                    
                    print(f"Detecciones: {len(boxes_640)} cajas, labels: {labels_out}, scores: {scores_out}")
                    
                    # Filtrar por clase persona (label 0) Y por confianza
                    mask_person = (labels_out == 0) & (scores_out >= args.conf_thresh)
                    
                    if np.any(mask_person):
                        boxes_640_person = boxes_640[mask_person]
                        scores_person = scores_out[mask_person]
                        
                        print(f"Personas detectadas: {len(boxes_640_person)}")
                        
                        # Convertir coordenadas y procesar pose
                        final_boxes = []
                        for box in boxes_640_person:
                            # Las cajas ya vienen en formato xyxy
                            x1_lb, y1_lb, x2_lb, y2_lb = box
                            
                            # Revertir letterbox
                            x1_un = (x1_lb - pad_left) / scale
                            y1_un = (y1_lb - pad_top) / scale
                            x2_un = (x2_lb - pad_left) / scale
                            y2_un = (y2_lb - pad_top) / scale

                            x1o = int(max(x1_un, 0))
                            y1o = int(max(y1_un, 0))
                            x2o = int(min(x2_un, frame.shape[1] - 1))
                            y2o = int(min(y2_un, frame.shape[0] - 1))

                            if (x2o > x1o) and (y2o > y1o):
                                final_boxes.append((x1o, y1o, x2o, y2o))

                        # Procesar pose para cada detección
                        last_coords_2d.clear()
                        for n, (x1o, y1o, x2o, y2o) in enumerate(final_boxes):
                            cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)
                            
                            # Procesar pose (mantener código original)
                            bbox = [x1o, y1o, x2o - x1o, y2o - y1o]
                            proc_bbox = pose_utils.process_bbox(np.array(bbox), frame.shape[1], frame.shape[0])
                            
                            if proc_bbox is not None:
                                img_patch, img2bb_trans = generate_patch_image(frame, proc_bbox, False, 1.0, 0.0, False)
                                inp = pose_transform(img_patch).unsqueeze(0).to(device)
                                root_depth = root_wrapper.predict_depth(frame, bbox)
                                print(f"bbox: {proc_bbox}, root_depth: {root_depth}")
                                
                                t0 = time.time()
                                with torch.no_grad():
                                    outputs = pose_model(inp)
                                    if outputs.dim() > 3:
                                        from model import soft_argmax
                                        pose_3d = soft_argmax(outputs, joint_num = 18, depth_dim = cfg.depth_dim, output_shape = cfg.output_shape)
                                    else:
                                        pose_3d = outputs
                                t1 = time.time()
                                
                                latency_ms = (t1 - t0) * 1000
                                pose_latencies.append(latency_ms)
                                
                                pose_3d = pose_3d[0].cpu().numpy()

                                if np.abs(pose_3d).max() > 1000:
                                    scale_factor = np.abs(pose_3d).max() / 18.0
                                    pose_3d = pose_3d / scale_factor
                                    print(f"⚠️ Valores extremos detectados, aplicando normalización. Factor: {scale_factor}")
                                pose_3d[:,0] = np.abs(pose_3d[:,0]) % cfg.output_shape[1]
                                pose_3d[:,1] = np.abs(pose_3d[:,1]) % cfg.output_shape[0]

                                print(f"pose_3d[0:5] después de corregir: {pose_3d[:5]}")
                                # ===== SOLUCIÓN DEFINITIVA: USAR EXACTAMENTE EL MISMO CÓDIGO QUE EN demo.py =====
                                # 1) Primero escalar al tamaño de entrada
                                pose_3d[:, 0] = pose_3d[:, 0] / cfg.output_shape[1] * cfg.input_shape[1]
                                pose_3d[:, 1] = pose_3d[:, 1] / cfg.output_shape[0] * cfg.input_shape[0]
                                pose_3d[:, 2] = (pose_3d[:, 2] / cfg.depth_dim * 2 -1 ) * (cfg.bbox_3d_shape[0]/2) + root_depth
                                
                                # 2) Crear matriz afín invertible completa
                                pose_3d_xy1 = np.concatenate((pose_3d[:, :2], np.ones_like(pose_3d[:, :1])), 1)
                                img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0, 0, 1]).reshape(1, 3)))
                                
                                # 3) Aplicar transformación inversa para mapear de vuelta a espacio de imagen original
                                pose_3d[:, :2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
                                
                                # 4) El resultado ya son coordenadas en el espacio de imagen original
                                coords_img = pose_3d[:, :2].copy()

                                coords_img = correct_skeleton_orientation(coords_img)
                                coords_img = adjust_skeleton_proportions(coords_img, scale_factor=1.3)
                                coords_img = improved_skeleton_alignment(coords_img, [x1o, y1o, x2o, y2o], pose_3d)
                                # 5) Asegurarse de que estén dentro de los límites
                                coords_img[:, 0] = np.clip(coords_img[:, 0], 0, frame.shape[1]-1)
                                coords_img[:, 1] = np.clip(coords_img[:, 1], 0, frame.shape[0]-1)
                                
                                print(f"coords_img definitivas (primeros 5): {coords_img[:5]}")
                                draw_skeleton(frame, coords_img, skeleton, offset=(0,0), color=(0,255,0))
                                
                                if frame_count % every_n == 0:
                                    last_coords_2d.append(coords_img)
                    else:
                        print("⚠️ No se detectaron personas con suficiente confianza")
                else:
                    print("⚠️ No se detectaron objetos en el frame")
                    
            except Exception as e:
                print(f"Error en YOLO pipeline: {e}")
                import traceback
                traceback.print_exc()

        frame_count += 1
        
        # Dibujar esqueleto
        for joints in last_coords_2d:
            draw_skeleton(frame, joints, skeleton, offset=(0, 0), color=(0, 255, 0))
        
        # Mostrar FPS
        fps = 1.0 / max((time.time() - t_frame_start), 1e-6)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Pose 3D (YOLO INT8 + ConvNeXtPose .pth)", frame)

        # Latencia cada 30 frames
        if frame_count % 30 == 0 and pose_latencies:
            avg_lat = sum(pose_latencies) / len(pose_latencies)
            print(f"[INFO] Frame {frame_count}, Latencia promedio ConvNeXtPose: {avg_lat:.2f} ms")
            pose_latencies.clear()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()