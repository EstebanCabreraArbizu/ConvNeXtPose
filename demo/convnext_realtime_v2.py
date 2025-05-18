#!/usr/bin/env python3
"""
run_pose_mixed.py

– Detección de persona con YOLOv5 INT8-ONNX (DeepSparse)
– Estimación de pose 3D con ConvNeXtPose desde checkpoint PyTorch (.pth)
– Visualización en tiempo real con FPS en pantalla y latencia de ConvNeXtPose cada 30 cuadros
"""

import argparse
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from deepsparse import compile_model


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.extend([
    str(PROJECT_ROOT / 'main'),
    str(PROJECT_ROOT / 'data'),
    str(PROJECT_ROOT / 'common')
])

# ------ IMPORTACIÓN DE CONVNEXTPOSE (usar rutas relativas al proyecto) ------
# Asegúrate de que los módulos `config`, `get_pose_net`, `generate_patch_image`
# y funciones asociadas estén en tu PYTHONPATH o en la misma carpeta que este script.
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

def main():
    parser = argparse.ArgumentParser(
        description="Detección de personas con YOLO INT8-ONNX + DeepSparse "
                    "y estimación de pose 3D con ConvNeXtPose desde checkpoint PyTorch"
    )
    parser.add_argument('--input', type=str, default='0',
                        help='Fuente de video (0=webcam o ruta archivo)')
    parser.add_argument('--yolo-model', type=str, required=True,
                        help='Ruta al modelo YOLO ONNX (INT8) para DeepSparse')
    parser.add_argument('--pose-model', type=str, required=True,
                        help='Ruta al checkpoint ConvNeXtPose (.pth)')
    parser.add_argument('--conf-thresh', type=float, default=0.3,
                        help='Umbral de confianza para detección de personas')
    args = parser.parse_args()

    # 1) Inicializar motor DeepSparse con modelo YOLOv5 INT8-ONNX
    print(f"[INFO] Cargando YOLO INT8-ONNX con DeepSparse desde: {args.yolo_model}")
    yolo_engine = compile_model(args.yolo_model, batch_size=1)

    # 2) Cargar ConvNeXtPose desde .pth (PyTorch)
    print(f"[INFO] Cargando ConvNeXtPose desde checkpoint: {args.pose_model}")
    # Asumimos 18 articulaciones; ajústalo según tu configuración
    pose_model, device = load_pose_model(args.pose_model, joint_num=18, use_cuda=False)

    # 3) Transformación para ConvNeXtPose (entrada 256×192 o según cfg.input_shape)
    #    ConvNeXtPose espera generación de parche con `generate_patch_image`, 
    #    pero igual normalizamos antes de pasar a la red:
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std),
    ])

    # 4) Definir esqueleto (subconjunto de 18 joints, según convnextpose_cfg)
    skeleton = (
        (0, 16), (16, 1), (1, 15), (15, 14), (14, 8),
        (8, 9), (9, 10), (14, 11), (11, 12), (12, 13),
        (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)
    )

    # 5) Captura de video
    cap = cv2.VideoCapture(int(args.input) if args.input.isdigit() else args.input)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir fuente de video: {args.input}")
        return

    frame_count = 0
    pose_latencies = []
    yolo_size = (640, 640)  # tamaño que espera el modelo YOLO

    print("[INFO] Iniciando demo en tiempo real. Presione 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        t_frame_start = time.time()

        # ===== 1) Detección de personas con YOLOv5 INT8 (DeepSparse) =====
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_lb, scale, pad_left, pad_top = letterbox(img_rgb, new_size=yolo_size)

        # Normalizar a [0,1] y reordenar canales para DeepSparse
        img_input = img_lb.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))[None, ...]  # (1,3,H,W)

        # Inferencia con DeepSparse → salida: array (1, N_detecciones, 6) [x1,y1,x2,y2,conf,cls]
        yolo_out = yolo_engine.run([img_input])[0]

        # Filtrar detecciones de clase 'persona' (cls == 0) y conf > umbral
        boxes = []
        for det in yolo_out:
            x1, y1, x2, y2, conf, cls = det
            if conf < args.conf_thresh or int(cls) != 0:
                continue
            # Revertir letterbox: revertir padding y escala
            x1_un = (x1 - pad_left) / scale
            y1_un = (y1 - pad_top) / scale
            x2_un = (x2 - pad_left) / scale
            y2_un = (y2 - pad_top) / scale
            # Clampear a límites de la imagen original
            x1o = int(max(x1_un, 0))
            y1o = int(max(y1_un, 0))
            x2o = int(min(x2_un, frame.shape[1] - 1))
            y2o = int(min(y2_un, frame.shape[0] - 1))
            boxes.append((x1o, y1o, x2o, y2o))

        # ===== 2) Estimación de pose con ConvNeXtPose (PyTorch) =====
        for (x1o, y1o, x2o, y2o) in boxes:
            crop = frame[y1o:y2o, x1o:x2o]
            if crop.size == 0:
                continue

            # Generar parche según ConvNeXtPose: se usa generate_patch_image
            # para obtener parche y matriz de transformación inversa
            bbox = [x1o, y1o, x2o - x1o, y2o - y1o]
            proc_bbox = pose_utils.process_bbox(np.array(bbox), frame.shape[1], frame.shape[0])
            if proc_bbox is None:
                continue

            img_patch, img2bb_trans = generate_patch_image(frame, proc_bbox, False, 1.0, 0.0, False)
            # img_patch tiene tamaño cfg.input_shape (p.ej. 256×192)

            # Normalizar y preparar tensor (1,3,H,W)
            inp = pose_transform(img_patch).unsqueeze(0).to(device)

            # Medición de latencia PyTorch
            t0 = time.time()
            with torch.no_grad():
                heatmaps = pose_model(inp)  # salida: (1, J, 3) coordenadas en patch
            t1 = time.time()
            latency_ms = (t1 - t0) * 1000
            pose_latencies.append(latency_ms)

            # Obtener coordenadas 2D (pixel) desde heatmaps
            # ConvNeXtPose devuelve coordenadas 3D (x,y,z) en sistema de patch
            coords_3d = heatmaps[0].cpu().numpy()  # (J,3)
            coords_2d = coords_3d[:, :2].copy()    # (J,2)

            # Escalar coords del parche al tamaño original de bbox
            h_patch, w_patch = img_patch.shape[:2]
            scale_x = proc_bbox[2] / w_patch
            scale_y = proc_bbox[3] / h_patch
            coords_2d[:, 0] = coords_2d[:, 0] * scale_x + proc_bbox[0]
            coords_2d[:, 1] = coords_2d[:, 1] * scale_y + proc_bbox[1]

            # Dibujar esqueleto
            draw_skeleton(frame, coords_2d, skeleton, offset=(0, 0), color=(0, 255, 0))

        # ===== 3) Mostrar FPS en pantalla =====
        fps = 1.0 / max((time.time() - t_frame_start), 1e-6)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Pose 3D (YOLO INT8 + ConvNeXtPose .pth)", frame)

        # ===== 4) Latencia promedio PyTorch cada 30 cuadros =====
        if frame_count % 30 == 0 and pose_latencies:
            avg_lat = sum(pose_latencies) / len(pose_latencies)
            print(f"[INFO] Latencia promedio ConvNeXtPose (últimos 30): {avg_lat:.2f} ms")
            pose_latencies.clear()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
