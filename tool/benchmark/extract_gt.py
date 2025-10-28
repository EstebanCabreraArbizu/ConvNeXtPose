#!/usr/bin/env python3
"""
Helper script to extract and verify ground truth from dataset JSON files.

This is optional - the benchmark runner loads GT automatically from JSON.
Use this only if you want to:
  1. Verify GT data is available
  2. Pre-cache GT in NPZ format for faster loading
  3. Inspect GT statistics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.Human36M.Human36M import Human36M


def extract_human36m_gt(protocol: int = 2, save_npz: bool = False, out_path: str = "gt_h36m_p{protocol}.npz") -> None:
    """Extract ground truth from Human3.6M dataset JSON files.
    
    The GT is stored in:
        data/Human36M/annotations/Human36M_subject*_joint_3d.json
    
    These files contain MoCap (Motion Capture) data - the "real" 3D poses.
    """
    print(f"=" * 60)
    print(f"Extracting Ground Truth from Human3.6M Protocol {protocol}")
    print(f"=" * 60)
    
    # Load dataset (this reads JSON files automatically)
    dataset = Human36M(data_split='test')
    dataset.protocol = protocol
    dataset.data = dataset.load_data()
    
    print(f"\n✓ Loaded dataset from JSON annotations")
    print(f"  Test subjects: {dataset.get_subject()}")
    print(f"  Frame stride: {dataset.get_subsampling_ratio()}")
    print(f"  Total samples: {len(dataset.data)}")
    
    # Extract GT
    gt_poses = []
    image_ids = []
    image_paths = []
    
    for sample in dataset.data:
        joint_cam = sample['joint_cam']  # (18, 3) including Thorax
        
        # Exclude Thorax (joint 17) to match evaluation protocol
        joint_cam_eval = joint_cam[dataset.eval_joint, :]  # (17, 3)
        
        gt_poses.append(joint_cam_eval)
        image_ids.append(sample['img_id'])
        image_paths.append(sample['img_path'])
    
    gt_3d = np.array(gt_poses, dtype=np.float32)  # (N, 17, 3) in mm
    
    # Statistics
    print(f"\n📊 Ground Truth Statistics:")
    print(f"  Shape: {gt_3d.shape}")
    print(f"  Dtype: {gt_3d.dtype}")
    print(f"  Range: [{gt_3d.min():.2f}, {gt_3d.max():.2f}] mm")
    print(f"  Mean: {gt_3d.mean():.2f} mm")
    print(f"  Std: {gt_3d.std():.2f} mm")
    
    # Show sample
    print(f"\n📋 Sample GT (first pose):")
    print(f"  Pelvis (root): {gt_3d[0, 0]}")
    print(f"  Head: {gt_3d[0, 10]}")
    print(f"  R_Wrist: {gt_3d[0, 16]}")
    
    # Per-joint stats
    print(f"\n📍 Per-joint depth range (Z coordinate):")
    for j, name in enumerate(dataset.joints_name[:17]):
        z_vals = gt_3d[:, j, 2]
        print(f"  {name:12s}: [{z_vals.min():7.1f}, {z_vals.max():7.1f}] mm")
    
    if save_npz:
        out_file = Path(out_path.format(protocol=protocol))
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            out_file,
            gt_3d=gt_3d,
            joint_cam=gt_3d,  # alias
            image_ids=np.array(image_ids),
            image_paths=np.array(image_paths),
            protocol=protocol,
            subjects=np.array(dataset.get_subject()),
            stride=dataset.get_subsampling_ratio(),
        )
        
        print(f"\n💾 Saved to: {out_file}")
        print(f"   Size: {out_file.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"\n   Keys: gt_3d, joint_cam, image_ids, image_paths, protocol, subjects, stride")
    
    print(f"\n✅ Done! GT is available in dataset.data[i]['joint_cam']")
    print(f"   No need to use NPZ - the benchmark runner loads JSON directly.")
    
    return gt_3d


def verify_json_files() -> None:
    """Verify that required JSON annotation files exist."""
    print("Checking for Human3.6M annotation files...")
    
    annot_dir = Path("data/Human36M/annotations")
    
    if not annot_dir.exists():
        print(f"❌ Annotation directory not found: {annot_dir}")
        print(f"\n📥 You need to download Human3.6M dataset from:")
        print(f"   http://vision.imar.ro/human3.6m/")
        print(f"\n📁 Expected structure:")
        print(f"   data/Human36M/")
        print(f"   ├── annotations/")
        print(f"   │   ├── Human36M_subject9_joint_3d.json   ← Ground Truth")
        print(f"   │   ├── Human36M_subject11_joint_3d.json  ← Ground Truth")
        print(f"   │   └── ...")
        print(f"   └── images/")
        return
    
    required_files = {
        'Protocol 2': [
            ('Human36M_subject9_data.json', 'COCO annotations'),
            ('Human36M_subject9_camera.json', 'Camera parameters'),
            ('Human36M_subject9_joint_3d.json', 'Ground Truth 3D'),
            ('Human36M_subject11_data.json', 'COCO annotations'),
            ('Human36M_subject11_camera.json', 'Camera parameters'),
            ('Human36M_subject11_joint_3d.json', 'Ground Truth 3D'),
        ]
    }
    
    print(f"\n✓ Found annotation directory: {annot_dir}")
    print(f"\nChecking Protocol 2 files (subjects S9, S11):")
    
    all_ok = True
    for filename, description in required_files['Protocol 2']:
        filepath = annot_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024 / 1024
            print(f"  ✓ {filename:40s} ({size_mb:6.2f} MB) - {description}")
        else:
            print(f"  ❌ {filename:40s} - MISSING - {description}")
            all_ok = False
    
    if all_ok:
        print(f"\n✅ All required files present!")
        print(f"   You can run the benchmark without any NPZ conversion.")
    else:
        print(f"\n❌ Some files are missing. Please download Human3.6M dataset.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and verify ground truth from Human3.6M JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify JSON files exist
  python extract_gt.py --verify
  
  # Extract GT statistics (no save)
  python extract_gt.py --protocol 2
  
  # Extract and save to NPZ (optional)
  python extract_gt.py --protocol 2 --save --out gt_h36m_p2.npz
  
Note: Saving to NPZ is optional. The benchmark runner loads GT from JSON automatically.
        """
    )
    parser.add_argument("--verify", action="store_true", help="Verify JSON files exist")
    parser.add_argument("--protocol", type=int, default=2, choices=[1, 2], help="Protocol (1 or 2)")
    parser.add_argument("--save", action="store_true", help="Save GT to NPZ file")
    parser.add_argument("--out", default="gt_h36m_p{protocol}.npz", help="Output NPZ path")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_json_files()
    else:
        extract_human36m_gt(
            protocol=args.protocol,
            save_npz=args.save,
            out_path=args.out
        )


if __name__ == "__main__":
    main()
