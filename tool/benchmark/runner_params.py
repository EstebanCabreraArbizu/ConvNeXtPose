#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import numpy as np

from .pipeline import PipelineContract, run_many
from .metrics import EXPECTED_MPJPE
from .download import ensure_checkpoint


# --- Data preparation ---

def prep_data_from_npz(npz_path: Optional[Path]) -> Tuple[np.ndarray, dict]:
    """Load ground truth from NPZ file (optional).
    
    If no NPZ provided, returns dummy data for smoke tests.
    For real benchmarks, use prep_data_from_dataset() instead.
    """
    if npz_path is None:
        print("[note] No GT npz provided. Using tiny dummy GT for smoke run.")
        gt = np.zeros((2, 17, 3), dtype=np.float32)
        return gt, {"dataset": "unknown", "note": "dummy"}
    npz = np.load(npz_path)
    # try common keys
    for key in ("gt", "gt_3d", "poses_3d", "coords_3d", "joint_cam"):
        if key in npz:
            gt = np.asarray(npz[key])
            break
    else:
        raise KeyError(f"No known GT key found in {npz_path}. Available keys: {list(npz.keys())}")
    if gt.ndim == 2 and gt.shape[1] % 3 == 0:
        gt = gt.reshape(gt.shape[0], -1, 3)
    assert gt.ndim == 3 and gt.shape[-1] == 3, f"Invalid GT shape: {gt.shape}"
    return gt.astype(np.float32), {"dataset": "from_npz", "path": str(npz_path)}


def prep_data_from_dataset(dataset_name: str = "Human3.6M", protocol: int = 2) -> Tuple[np.ndarray, dict]:
    """Load ground truth directly from dataset loaders (RECOMMENDED).
    
    This uses the existing dataset classes that automatically load GT from JSON files.
    No need to create separate NPZ files - the GT is already in the JSON annotations.
    
    Args:
        dataset_name: "Human3.6M", "MuPoTS", etc.
        protocol: 1 or 2 (for Human3.6M)
    
    Returns:
        gt_3d: (N, J, 3) array in mm, camera coordinates
        meta: dict with dataset info
    """
    if dataset_name.lower() in ("human3.6m", "h36m", "human36m"):
        # Use existing Human36M class that loads GT from JSON automatically
        import sys
        sys.path.insert(0, '/home/user/convnextpose_esteban/ConvNeXtPose')
        from data.Human36M.Human36M import Human36M
        
        # Load test set (includes GT from Human36M_subject*_joint_3d.json)
        dataset = Human36M(data_split='test')
        dataset.protocol = protocol
        dataset.data = dataset.load_data()  # Loads GT from JSON files
        
        # Extract GT from loaded data
        gt_poses = []
        image_paths = []
        camera_params = []
        
        for sample in dataset.data:
            gt_poses.append(sample['joint_cam'])  # ← GT from JSON (17×3 in mm)
            image_paths.append(sample['img_path'])
            camera_params.append({'f': sample['f'], 'c': sample['c']})
        
        gt_3d = np.array(gt_poses, dtype=np.float32)  # (N, 17, 3) or (N, 18, 3)
        
        # Exclude thorax if present (joint 17)
        if gt_3d.shape[1] == 18:
            gt_3d = gt_3d[:, dataset.eval_joint, :]  # Keep only 17 eval joints
        
        meta = {
            "dataset": f"Human3.6M Protocol {protocol}",
            "num_samples": len(gt_3d),
            "joints": gt_3d.shape[1],
            "subjects": dataset.get_subject(),
            "stride": dataset.get_subsampling_ratio(),
            "image_paths": image_paths,
            "camera_params": camera_params,
            "note": "GT loaded directly from JSON annotations (Human36M_subject*_joint_3d.json)"
        }
        
        print(f"[data] Loaded {len(gt_3d)} samples from {dataset_name} Protocol {protocol}")
        print(f"[data] Subjects: {meta['subjects']}, Stride: {meta['stride']}")
        
        return gt_3d, meta
    
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not yet supported. Add loader in prep_data_from_dataset().")


# --- Download hooks per model ---

def make_dl_rootnet(snapshot: str) -> Callable[[], Optional[str]]:
    def _dl() -> Optional[str]:
        try:
            path = ensure_checkpoint("rootnet", snapshot)
            return str(path)
        except Exception as e:
            print(f"[warn] RootNet checkpoint not ensured: {e}")
            return None
    return _dl


def make_dl_integral(snapshot: str) -> Callable[[], Optional[str]]:
    def _dl() -> Optional[str]:
        try:
            path = ensure_checkpoint("integral_human_pose", snapshot)
            return str(path)
        except Exception as e:
            print(f"[warn] Integral checkpoint not ensured: {e}")
            return None
    return _dl


# --- Placeholder inference ---

def infer_zeros_like(gt_array: np.ndarray) -> Callable[[Optional[str]], np.ndarray]:
    """Create a placeholder inference function that returns zeros matching GT shape."""
    def _infer(_weights: Optional[str]) -> np.ndarray:
        return np.zeros_like(gt_array, dtype=np.float32)
    return _infer


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameterized 3D pose benchmark runner")
    parser.add_argument("--dataset", default="Human3.6M", help="Dataset name (for logging)")
    parser.add_argument("--protocol", type=int, default=2, choices=[1, 2], help="Protocol for H36M")
    parser.add_argument("--gt-npz", type=str, default=None, help="(Optional) Path to NPZ with GT. If not provided, loads from dataset JSON automatically.")
    parser.add_argument("--use-dataset-loader", action="store_true", default=True, help="Load GT directly from dataset JSON (recommended)")
    parser.add_argument("--models", nargs="*", default=["rootnet", "integral_human_pose"], help="Models to run")
    parser.add_argument("--rootnet-snapshot", default="snapshot_19.pth.tar", help="RootNet snapshot filename per manifest")
    parser.add_argument("--integral-snapshot", default="snapshot_16.pth.tar", help="Integral snapshot filename per manifest")
    parser.add_argument("--out", default="output/benchmark", help="Output dir")
    args = parser.parse_args()

    # Load ground truth (prefer dataset loader over NPZ)
    if args.use_dataset_loader and args.gt_npz is None:
        print("[data] Using dataset loader (GT from JSON annotations)")
        gt, meta = prep_data_from_dataset(args.dataset, args.protocol)
    else:
        print("[data] Using NPZ file (if provided) or dummy data")
        gt, meta = prep_data_from_npz(Path(args.gt_npz) if args.gt_npz else None)

    contracts = []
    fns = []

    for m in args.models:
        c = PipelineContract(name=m, dataset_name=args.dataset, joints=gt.shape[1], protocol=args.protocol)
        contracts.append(c)
        # wire DL hook
        if m.lower() == "rootnet":
            dl = make_dl_rootnet(args.rootnet_snapshot)
        elif m.lower() == "integral_human_pose":
            dl = make_dl_integral(args.integral_snapshot)
        else:
            dl = lambda: None
        # placeholder inference matching GT shape
        infer = infer_zeros_like(gt)
        prep = lambda: (gt, meta)
        fns.append((prep, dl, infer))

    results = run_many(contracts, fns, out_dir=Path(args.out))

    print("Results:")
    for r in results:
        exp = EXPECTED_MPJPE.get(r.name)
        print(f"- {r.name}: MPJPE={r.mpjpe_mm:.2f} mm (expected ~{exp} mm)")


if __name__ == "__main__":
    main()
