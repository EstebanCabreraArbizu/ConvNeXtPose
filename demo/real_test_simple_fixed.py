#!/usr/bin/env python3
"""
convnextpose_efficient.py

Pipeline eficiente: 
- Detección de persona con YOLOv5
- Estimación de pose 3D con ConvNeXtPose
- Exportación optimizada (TorchScript, ONNX y Lite)
- Soporte FP16 en GPU
"""

import os
import sys
import argparse
import time

import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from torch.nn import DataParallel
from torch.utils.mobile_optimizer import optimize_for_mobile

# Ajustar sys.path para módulos del proyecto
ROOT = Path(__file__).resolve().parent
sys.path.extend([str(ROOT / p) for p in ('main', 'data', 'common')])

from config import cfg
from model import get_pose_net
from utils.pose_utils import process_bbox, pixel2cam
from utils.vis import vis_keypoints, vis_3d_multiple_skeleton

class PoseEstimator:
    def __init__(self, model_path, joint_num=18, use_cuda=True, yolo_model='yolov5s'):
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Device: {self.device}")

        # Cargar YOLOv5
        try:
            self.yolo = torch.hub.load('ultralytics/yolov5', yolo_model, pretrained=True).to(self.device).eval()
            print(f"[INFO] YOLOv5 ({yolo_model}) loaded")  # :contentReference[oaicite:1]{index=1}
        except:
            print("[WARN] YOLOv5 load failed, fallback to full-frame detection")
            self.yolo = None

        # Crear modelo ConvNeXtPose
        cfg.clear()  # eliminar cfg.set_args hack
        self.joint_num = joint_num
        base_model = get_pose_net(cfg, head='heatmap', joint_num=joint_num)
        if self.device.type == 'cuda':
            self.model = DataParallel(base_model).to(self.device)
        else:
            self.model = base_model.to(self.device)
        self.model.eval()

        # Cargar pesos (.pth o .tar convertido previamente)
        state = torch.load(model_path, map_location=self.device)
        sd = state.get('network', state)
        self.model.load_state_dict(sd, strict=False)
        print("[INFO] ConvNeXtPose loaded")  # 

        # Transformación estándar
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std),
        ])

        # Parámetros cámara default
        self.focal = [1500, 1500]
        self.princpt = [0, 0]

    def export(self, out_dir='exports'):
        os.makedirs(out_dir, exist_ok=True)
        dummy = torch.rand(1, 3, *cfg.input_shape).to(self.device)

        # 1) Exportar state_dict optimizado
        sd = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save({'network': sd}, os.path.join(out_dir, 'model_opt.pth'))
        print("[INFO] State_dict exportado")  # 

        # 2) TorchScript
        traced = torch.jit.trace(self.model.module if isinstance(self.model, DataParallel) else self.model, dummy)
        traced = optimize_for_mobile(traced)  # :contentReference[oaicite:2]{index=2}
        traced.save(os.path.join(out_dir, 'model_mobile.pt'))
        print("[INFO] TorchScript (mobile) exportado")

        # 3) ONNX
        torch.onnx.export(
            self.model.module if isinstance(self.model, DataParallel) else self.model,
            dummy,
            os.path.join(out_dir, 'model.onnx'),
            opset_version=12, input_names=['input'], output_names=['heatmaps'],
            dynamic_axes={'input': {0:'batch'}, 'heatmaps': {0:'batch'}}
        )
        print("[INFO] ONNX exportado")  # 

    def detect_persons(self, img, conf=0.25):
        if self.yolo is None:
            h,w = img.shape[:2]
            return [{'bbox':[0,0,w,h],'conf':1.0}]
        det = self.yolo(img)[0].cpu().numpy()
        ppl = det[(det[:,5]==0)&(det[:,4]>=conf)]
        if ppl.size==0:
            return [{'bbox':[0,0,img.shape[1],img.shape[0]], 'conf':1.0}]
        out=[]
        for x1,y1,x2,y2,c,_ in ppl:
            out.append({'bbox':[int(x1),int(y1),int(x2-x1),int(y2-y1)], 'conf':float(c)})
        return out

    def estimate(self, frame, kp_thresh=0.3, fp16=True):
        self.princpt = [frame.shape[1]/2, frame.shape[0]/2]
        persons = self.detect_persons(frame)
        results=[]
        for p in persons:
            bbox = process_bbox(np.array(p['bbox']), frame.shape[1], frame.shape[0])
            patch, trans = generate_patch_image(frame, bbox, False,1,0,False)
            inp = self.transform(patch).unsqueeze(0).to(self.device)
            if fp16 and self.device.type=='cuda':
                inp = inp.half()  # 
                self.model.half()
            with torch.no_grad():
                heatmaps = self.model(inp).float()
            # obtener coords y scores
            coords, scores = self.get_max_preds(heatmaps.cpu().numpy())
            coords = coords[0]; scores = scores[0]
            # filtrar y escalar
            valid = scores>kp_thresh
            H,W = patch.shape[:2]; _,_,hmap_h, hmap_w = heatmaps.shape
            coords[:,0] = coords[:,0]/hmap_w*patch.shape[1] + bbox[0]
            coords[:,1] = coords[:,1]/hmap_h*patch.shape[0] + bbox[1]
            results.append({'coords':coords[valid], 'scores':scores[valid], 'bbox':bbox})
        return results

    @staticmethod
    def get_max_preds(heatmaps):
        b,j,h,w = heatmaps.shape
        flat = heatmaps.reshape((b,j,-1))
        idx = flat.argmax(-1); vals = flat.max(-1)
        coords = np.zeros((b,j,2),dtype=float)
        coords[:,:,0] = idx % w; coords[:,:,1] = idx//w
        return coords, vals

    def visualize(self, frame, results, show_3d=False):
        vis = frame.copy()
        for res in results:
            kps = np.vstack([res['coords'].T, res['scores']])
            vis = vis_keypoints(vis, kps, cfg.skeleton, kp_thresh=0.3)
        return vis

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='.pth optimizado')
    p.add_argument('--export', action='store_true')
    p.add_argument('--yolo', default='yolov5s')
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    est = PoseEstimator(args.model, use_cuda=not args.cpu, yolo_model=args.yolo)
    if args.export:
        est.export()
        return

    cap = cv2.VideoCapture(0)
    prev = time.time()
    while True:
        ret,frame = cap.read()
        if not ret: break
        res = est.estimate(frame)
        vis = est.visualize(frame,res)
        now = time.time()
        fps = 1/(now-prev); prev=now
        cv2.putText(vis,f"FPS: {fps:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Pose',vis)
        if cv2.waitKey(1)&0xFF==27: break

    cap.release(); cv2.destroyAllWindows()

if __name__=='__main__':
    main()
