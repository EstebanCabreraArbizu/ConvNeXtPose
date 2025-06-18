#!/usr/bin/env python3
"""
Versión simplificada de v4 que usa ONNX directo sin threading/queues
para comparación justa con v3
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import time
import torch
import torchvision.transforms as transforms

# Setup paths
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / 'main'))
sys.path.insert(0, str(PROJECT_ROOT / 'data'))
sys.path.insert(0, str(PROJECT_ROOT / 'common'))

from config import cfg
from model import get_pose_net
from dataset import generate_patch_image
import utils.pose_utils as pose_utils

class SimplifiedV4Processor:
    """Versión simplificada de v4 usando ONNX directo sin arquitectura compleja"""
    
    def __init__(self, model_path: str, use_onnx: bool = True):
        self.model_path = model_path
        self.use_onnx = use_onnx
        self.engine_type = "Unknown"
        
        # Setup inference engine
        self._setup_inference_engine()
    
    def _setup_inference_engine(self):
        """Set up inference engine"""
        if self.use_onnx and self._setup_onnx():
            return
        
        # Fallback to PyTorch
        if self._setup_pytorch():
            return
        
        raise RuntimeError("❌ Could not configure any inference engine")
    
    def _setup_onnx(self) -> bool:
        """Set up ONNX Runtime inference engine"""
        try:
            import onnxruntime as ort
            
            onnx_path = self.model_path.replace('.pth', '_optimized.onnx')
            
            if not os.path.exists(onnx_path):
                print(f"❌ ONNX file not found: {onnx_path}")
                return False
            
            # Create ONNX session
            providers = ['CPUExecutionProvider']  # Force CPU for stability
            self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
            
            self.onnx_input_name = self.onnx_session.get_inputs()[0].name
            self.onnx_output_names = [output.name for output in self.onnx_session.get_outputs()]
            
            self.engine_type = "ONNX Runtime (Simplified)"
            print("✅ ONNX Runtime engine configured")
            return True
            
        except Exception as e:
            print(f"⚠️ ONNX setup failed: {e}")
            return False
    
    def _setup_pytorch(self) -> bool:
        """Set up PyTorch inference engine"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Configure model
            cfg.input_shape = (256, 256)
            cfg.output_shape = (32, 32)
            cfg.depth_dim = 32
            cfg.bbox_3d_shape = (2000, 2000, 2000)
            
            # Load model
            model = get_pose_net(cfg, is_train=False, joint_num=18)
            state = torch.load(self.model_path, map_location=device)
            state_dict = state.get('network', state)
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device).eval()
            
            self.pytorch_model = model
            self.pytorch_device = device
            self.pytorch_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std),
            ])
            
            self.engine_type = "PyTorch (Simplified)"
            print("✅ PyTorch engine configured")
            return True
            
        except Exception as e:
            print(f"⚠️ PyTorch setup failed: {e}")
            return False
    
    def process_pose(self, frame: np.ndarray, bbox: list) -> tuple:
        """Process pose directly without threading"""
        start_time = time.time()
        
        try:
            # Convert bbox format (x1, y1, x2, y2) to (x, y, w, h)
            x1, y1, x2, y2 = bbox
            bbox_array = np.array([x1, y1, x2 - x1, y2 - y1])
            
            # Process bbox
            h, w = frame.shape[:2]
            proc_bbox = pose_utils.process_bbox(bbox_array, w, h)
            
            # Generate patch
            img_patch, img2bb_trans = generate_patch_image(frame, proc_bbox, False, 1.0, 0.0, False)
            
            # Ensure correct data type
            if img_patch.dtype != np.uint8:
                img_patch = (img_patch * 255).astype(np.uint8) if img_patch.max() <= 1.0 else img_patch.astype(np.uint8)
            
            # Run inference
            if self.engine_type.startswith("ONNX"):
                pose_coords = self._infer_onnx(img_patch)
            else:
                pose_coords = self._infer_pytorch(img_patch)
            
            if pose_coords is None:
                return None, time.time() - start_time
            
            # Post-process coordinates
            final_coords = self._postprocess_coordinates(pose_coords, img2bb_trans)
            
            processing_time = time.time() - start_time
            return final_coords, processing_time
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            return None, time.time() - start_time
    
    def _infer_onnx(self, img_patch: np.ndarray) -> np.ndarray:
        """Run inference using ONNX Runtime"""
        try:
            # Preprocess for ONNX
            from PIL import Image
            pil_img = Image.fromarray(cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB))
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std),
            ])
            
            input_tensor = transform(pil_img).unsqueeze(0).numpy()
            
            # Run inference
            outputs = self.onnx_session.run(self.onnx_output_names, {self.onnx_input_name: input_tensor})
            
            # Handle output
            pose_3d = outputs[0][0]  # Remove batch dimension
            
            # Convert from heatmaps if needed
            if len(pose_3d.shape) == 3:  # [joints, height, width]
                joint_num, output_h, output_w = pose_3d.shape
                heatmaps = pose_3d.reshape(joint_num, -1)
                max_indices = np.argmax(heatmaps, axis=1)
                pose_y = max_indices // output_w
                pose_x = max_indices % output_w
                pose_coords = np.stack([pose_x, pose_y], axis=1).astype(np.float32)
            else:
                pose_coords = pose_3d[:, :2]
            
            return pose_coords
            
        except Exception as e:
            print(f"❌ ONNX inference failed: {e}")
            return None
    
    def _infer_pytorch(self, img_patch: np.ndarray) -> np.ndarray:
        """Run inference using PyTorch"""
        try:
            from PIL import Image
            pil_img = Image.fromarray(cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB))
            input_tensor = self.pytorch_transform(pil_img).unsqueeze(0).to(self.pytorch_device)
            
            # Run inference
            with torch.no_grad():
                pose_3d = self.pytorch_model(input_tensor)
            
            pose_3d = pose_3d[0].cpu().numpy()
            
            # Convert from heatmaps if needed
            if len(pose_3d.shape) == 3:  # [joints, height, width]
                joint_num, output_h, output_w = pose_3d.shape
                heatmaps = pose_3d.reshape(joint_num, -1)
                max_indices = np.argmax(heatmaps, axis=1)
                pose_y = max_indices // output_w
                pose_x = max_indices % output_w
                pose_coords = np.stack([pose_x, pose_y], axis=1).astype(np.float32)
            else:
                pose_coords = pose_3d[:, :2]
            
            return pose_coords
            
        except Exception as e:
            print(f"❌ PyTorch inference failed: {e}")
            return None
    
    def _postprocess_coordinates(self, pose_coords: np.ndarray, img2bb_trans: np.ndarray) -> np.ndarray:
        """Post-process pose coordinates exactly like v3"""
        try:
            # Scale to input space
            pose_coords[:, 0] = pose_coords[:, 0] / cfg.output_shape[1] * cfg.input_shape[1]
            pose_coords[:, 1] = pose_coords[:, 1] / cfg.output_shape[0] * cfg.input_shape[0]
            
            # Apply inverse transformation
            pose_coords_homo = np.column_stack((pose_coords, np.ones(len(pose_coords))))
            img2bb_trans_full = np.vstack((img2bb_trans, [0, 0, 1]))
            
            try:
                final_coords = np.linalg.solve(img2bb_trans_full, pose_coords_homo.T).T[:, :2]
            except:
                final_coords = pose_coords
            
            return final_coords
            
        except Exception as e:
            print(f"❌ Postprocessing failed: {e}")
            return pose_coords

def test_simplified_comparison():
    """Test directo entre v3 PyTorch y v4 ONNX simplificado"""
    print("🏁 ConvNeXt v3 PyTorch vs v4 ONNX Simplified Comparison")
    print("=" * 60)
    
    model_path = "/home/fabri/ConvNeXtPose/exports/model_opt_S.pth"
    image_path = "/home/fabri/ConvNeXtPose/demo/input.jpg"
    
    # Load image
    frame = cv2.imread(image_path)
    h, w = frame.shape[:2]
    bbox = [w//4, h//4, 3*w//4, 3*h//4]  # x1, y1, x2, y2
    
    results = {}
    
    # Test v3 (PyTorch)
    print("\n🧪 Testing v3 (PyTorch)...")
    try:
        processor_v3 = SimplifiedV4Processor(model_path, use_onnx=False)
        
        # Warmup
        for _ in range(3):
            processor_v3.process_pose(frame, bbox)
        
        # Benchmark
        times = []
        coords_list = []
        for i in range(10):
            coords, proc_time = processor_v3.process_pose(frame, bbox)
            if coords is not None:
                times.append(proc_time * 1000)
                coords_list.append(coords)
        
        if times:
            results['v3_pytorch'] = {
                'engine': processor_v3.engine_type,
                'avg_time_ms': np.mean(times),
                'std_time_ms': np.std(times),
                'coordinates': coords_list[0].tolist() if coords_list else None,
                'success': True
            }
            print(f"✅ v3 PyTorch: {np.mean(times):.1f}±{np.std(times):.1f}ms")
        else:
            results['v3_pytorch'] = {'success': False, 'error': 'No successful runs'}
            print("❌ v3 PyTorch failed")
            
    except Exception as e:
        results['v3_pytorch'] = {'success': False, 'error': str(e)}
        print(f"❌ v3 PyTorch failed: {e}")
    
    # Test v4 (ONNX)
    print("\n🧪 Testing v4 (ONNX Simplified)...")
    try:
        processor_v4 = SimplifiedV4Processor(model_path, use_onnx=True)
        
        # Warmup
        for _ in range(3):
            processor_v4.process_pose(frame, bbox)
        
        # Benchmark
        times = []
        coords_list = []
        for i in range(10):
            coords, proc_time = processor_v4.process_pose(frame, bbox)
            if coords is not None:
                times.append(proc_time * 1000)
                coords_list.append(coords)
        
        if times:
            results['v4_onnx'] = {
                'engine': processor_v4.engine_type,
                'avg_time_ms': np.mean(times),
                'std_time_ms': np.std(times),
                'coordinates': coords_list[0].tolist() if coords_list else None,
                'success': True
            }
            print(f"✅ v4 ONNX: {np.mean(times):.1f}±{np.std(times):.1f}ms")
        else:
            results['v4_onnx'] = {'success': False, 'error': 'No successful runs'}
            print("❌ v4 ONNX failed")
            
    except Exception as e:
        results['v4_onnx'] = {'success': False, 'error': str(e)}
        print(f"❌ v4 ONNX failed: {e}")
    
    # Compare results
    print("\n📊 COMPARISON RESULTS:")
    print("=" * 60)
    
    if results.get('v3_pytorch', {}).get('success') and results.get('v4_onnx', {}).get('success'):
        v3_time = results['v3_pytorch']['avg_time_ms']
        v4_time = results['v4_onnx']['avg_time_ms']
        
        speedup = v3_time / v4_time
        
        print(f"📊 Performance Comparison:")
        print(f"   v3 PyTorch: {v3_time:.1f}±{results['v3_pytorch']['std_time_ms']:.1f}ms")
        print(f"   v4 ONNX:    {v4_time:.1f}±{results['v4_onnx']['std_time_ms']:.1f}ms")
        print(f"   Speedup:    {speedup:.2f}x ({'✅ ONNX faster' if speedup > 1 else '❌ PyTorch faster'})")
        
        # Compare coordinates
        v3_coords = np.array(results['v3_pytorch']['coordinates'])
        v4_coords = np.array(results['v4_onnx']['coordinates'])
        
        if v3_coords.shape == v4_coords.shape:
            coord_diff = np.abs(v3_coords - v4_coords)
            mean_diff = np.mean(coord_diff)
            max_diff = np.max(coord_diff)
            
            print(f"\n📏 Coordinate Accuracy:")
            print(f"   Mean difference: {mean_diff:.2f} pixels")
            print(f"   Max difference:  {max_diff:.2f} pixels")
            print(f"   Similar results: {'✅ YES' if mean_diff < 5.0 else '❌ NO'}")
        
    else:
        print("❌ Cannot compare - one or both tests failed")
        for name, result in results.items():
            if not result.get('success', False):
                print(f"   {name}: {result.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    test_simplified_comparison()
