import argparse
import time
from pathlib import Path
import sys

import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.nn import DataParallel
from torch.utils.mobile_optimizer import optimize_for_mobile
import importlib
from ultralytics import YOLO
yolo_module = importlib.import_module('torch.hub')

# ──────────────────────────────────────────────────────────────────────────────
# 1. Configurar rutas internas de proyecto con pathlib 
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.extend([
    str(PROJECT_ROOT / 'main'),
    str(PROJECT_ROOT / 'data'),
    str(PROJECT_ROOT / 'common')
])

# 2. Importar módulos del proyecto con alias para evitar conflicto :contentReference[oaicite:8]{index=8}
from config import cfg
from model import get_pose_net
from dataset import generate_patch_image
import utils.pose_utils as pose_utils   # alias claro
import utils.vis as vis_utils           # alias claro
def get_max_preds(heatmaps: np.array):
    if heatmaps.ndim == 3:
        heatmaps = heatmaps[None, ...]
    b,j,h,w = heatmaps.shape
    flat = heatmaps.reshape((b,j,-1))
    idx = flat.argmax(axis=2); val = flat.max(axis=2)
    coords = np.zeros((b,j,2), dtype=float)
    coords[:,:,0] = idx % w; coords[:,:,1] = idx // w
    return coords, val


# ──────────────────────────────────────────────────────────────────────────────
# 3. Clase principal de estimación
class PoseEstimator:
    def __init__(self, model_pth, joint_num=18, use_cuda=True, yolo_name='yolov5s'):
        # Dispositivo CUDA/CPU
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Device: {self.device}")

        # 3.1 Cargar YOLOv5 desde PyTorch Hub :contentReference[oaicite:9]{index=9}
        try:
            self.yolo = YOLO(f"{yolo_name}.pt").to(self.device)
            print(f"[INFO] YOLOv5 ({yolo_name}) loaded via ultralytics") 
        except Exception as e:
            print(f"[WARN] YOLO load failed: {e}")
            self.yolo = None

        # 3.2 Preparar modelo ConvNeXtPose
        base = get_pose_net(cfg, is_train=False, joint_num=joint_num)
        self.model = DataParallel(base).to(self.device) if self.device.type=='cuda' else base.to(self.device)
        self.model.eval()

        # 3.3 Cargar pesos .pth optimizado (sin .tar) :contentReference[oaicite:10]{index=10}
        state = torch.load(model_pth, map_location=self.device)
        sd = state.get('network', state)
        self.model.load_state_dict(sd, strict=False)
        print("[INFO] ConvNeXtPose loaded")

        # 3.4 Transformación de imagen
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std),
        ])

        # Esqueleto completo (18 joints)
        if hasattr(cfg, 'skeleton'):
            self.skeleton = cfg.skeleton
        else:
            self.skeleton = (
                (0,16),(16,1),(1,15),(15,14),(14,8),
                (8,9),(9,10),(14,11),(11,12),(12,13),
                (1,2),(2,3),(3,4),(1,5),(5,6),(6,7)
            )

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Exportación optimizada
    def export(self, out_dir='exports'):
        Path(out_dir).mkdir(exist_ok=True)
        dummy = torch.rand(1, 3, *cfg.input_shape).to(self.device)

        # 4.1 Guardar state_dict
        sd = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save({'network': sd}, Path(out_dir)/'model_opt.pth')
        print("[INFO] State_dict exportado")

        # 4.2 TorchScript (use script para cubrir condicionales dinámicos) 
        scripted = torch.jit.script(self.model.module if isinstance(self.model, DataParallel) else self.model)
        mobile_ts = optimize_for_mobile(scripted)
        mobile_ts.save(Path(out_dir)/'model_mobile.pt')
        print("[INFO] TorchScript (mobile) exportado")

        # 4.3 ONNX
        torch.onnx.export(
            self.model.module if isinstance(self.model, DataParallel) else self.model,
            dummy,
            Path(out_dir)/'model.onnx',
            opset_version=12,
            input_names=['input'],
            output_names=['heatmaps'],
            dynamic_axes={'input': {0:'batch'}, 'heatmaps':{0:'batch'}}
        )
        print("[INFO] ONNX exportado")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Detección de personas
    def detect(self, img, conf=0.25):
        h,w = img.shape[:2]
        if self.yolo is None:
            return [{'bbox':[0,0,w,h], 'conf':1.0}]
        # 5.1 Nueva API: usar results.xyxy :contentReference[oaicite:11]{index=11}
        results = self.yolo.predict(source = img, conf=conf, device=self.device, classes = [0])
        out = []
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = box.xyxy.numpy().astype(int)[0]
                c = float(box.conf.numpy()[0])
                out.append({'bbox':[x1,y1,x2-x1,y2-y1], 'conf':c})
            if not out:
                out = [{'bbox':[0,0,w,h], 'conf':1.0}]
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Estimar pose con FP16 y sin gradientes
    def estimate(self, frame, kp_thresh=0.3, use_fp16=True):
        persons = self.detect(frame)
        results=[]
        # Inference en modo no-Grad y potencial FP16 :contentReference[oaicite:12]{index=12}
        with torch.inference_mode():
            for p in persons:
                bbox = pose_utils.process_bbox(np.array(p['bbox']), frame.shape[1], frame.shape[0])
                patch, trans = generate_patch_image(frame, bbox, False,1,0,False)
                inp = self.transform(patch).unsqueeze(0).to(self.device)
                if use_fp16 and self.device.type=='cuda':
                    inp = inp.half()
                    self.model.half()
                heat = self.model(inp).float().cpu().numpy()
                if heat.ndim == 4:
                    _, _, Hh, Ww = heat.shape
                elif heat.ndim == 3:
                    # convertir a batch=1
                    j, Hh, Ww = heat.shape
                    heat = heat[None, ...]
                else:
                    raise ValueError(f"Heatmap inesperado con ndim={heat.ndim}")
                coords, scores = get_max_preds(heat)  # mantiene completos los índices :contentReference[oaicite:13]{index=13}
                coords, scores = coords[0], scores[0]
                valid = scores>kp_thresh
                # escalar coords al frame original
                _,_,Hh, Ww = heat.shape
                coords[:,0] = coords[:,0]/Ww * patch.shape[1] + bbox[0]
                coords[:,1] = coords[:,1]/Hh * patch.shape[0] + bbox[1]
                results.append({'coords':coords[valid], 'scores':scores[valid]})
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Visualización consistente
    def visualize(self, img, results, kp_thresh=0.3):
        vis = img.copy()
        for r in results:
            n = len(self.skeleton)+1
            full = np.zeros((3, n))
            if r['coords'].size > 0:
                count = r['coords'].shape[0]
                full[0, :count] = r['coords'][:,0]
                full[1, :count] = r['coords'][:,1]
                full[2, :count] = r['scores']
            # Ahora vis_keypoints ignora los zeros bajo el umbral
            vis = vis_utils.vis_keypoints(vis, full, self.skeleton, kp_thresh)
        return vis

# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',  required=True, help='model_opt.pth')
    p.add_argument('--export', action='store_true')
    p.add_argument('--yolo',   default='yolov5s')
    p.add_argument('--cpu',    action='store_true')
    args = p.parse_args()

    est = PoseEstimator(
        model_pth=args.model,
        use_cuda=not args.cpu,
        yolo_name=args.yolo
    )
    if args.export:
        est.export(); return

    cap = cv2.VideoCapture(0)
    prev = time.time()
    while True:
        ret, frame = cap.read()
        if not ret: break
        res = est.estimate(frame)
        vis = est.visualize(frame, res)
        # FPS
        now = time.time(); fps = 1/(now-prev); prev=now
        cv2.putText(vis, f"FPS: {fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Pose', vis)
        if cv2.waitKey(1)&0xFF==27: break

    cap.release(); cv2.destroyAllWindows()

if __name__=='__main__':
    main()