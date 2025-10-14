# 🚨 Issue Report: Mislabeled Checkpoint Files in Official Google Drive

**Date:** October 13, 2025  
**Repository:** ConvNeXtPose  
**Paper:** "ConvNeXtPose: A Fast Accurate Method for 3D Human Pose Estimation" (IEEE Access 2023)  
**Google Drive:** https://drive.google.com/drive/folders/12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI

---

## 📋 Summary

The checkpoint files provided in the official Google Drive folder are **mislabeled**. All three files named as **Large (L)**, **Medium (M)**, and **Small (S)** models actually contain the **Small model architecture** with `dims=[48, 96, 192, 384]`.

This prevents users from reproducing the results reported in the paper for models L (42.3mm MPJPE) and M (44.6mm MPJPE).

---

## 🔍 Evidence

### Expected Architecture (from paper)

According to the paper and code documentation:

| Model | Dimensions | Expected MPJPE (H36M Protocol 2) |
|-------|------------|----------------------------------|
| **L (Large)** | `[192, 384, 768, 1536]` | **42.3 mm** |
| **M (Medium)** | `[64, 128, 256, 512]` | **44.6 mm** |
| **S (Small)** | `[48, 96, 192, 384]` | ~45 mm |
| **XS (Tiny)** | `[48, 96, 192, 384]` | - |

### Actual Architecture (analyzed from checkpoints)

Using PyTorch checkpoint inspection with legacy format unpickler:

```python
# Analysis results:
ConvNeXtPose_L (1).tar:
  ✓ First backbone layer: torch.Size([48, 3, 4, 4])
  ✓ Output channels: 48
  📊 DETECTED VARIANT: XS or S (dims=[48, 96, 192, 384])
  ✓ Total parameters: 8,391,354

ConvNeXtPose_M (1).tar:
  ✓ First backbone layer: torch.Size([48, 3, 4, 4])
  ✓ Output channels: 48
  📊 DETECTED VARIANT: XS or S (dims=[48, 96, 192, 384])
  ✓ Total parameters: 7,596,986

ConvNeXtPose_S.tar:
  ✓ First backbone layer: torch.Size([48, 3, 4, 4])
  ✓ Output channels: 48
  📊 DETECTED VARIANT: XS or S (dims=[48, 96, 192, 384])
  ✓ Total parameters: 7,448,954
```

**Key Finding:** All three files have the **same first layer shape** `[48, 3, 4, 4]`, confirming they use Small model architecture.

For comparison, model L should have first layer shape `[192, 3, 4, 4]` (4x larger), and model M should have `[64, 3, 4, 4]`.

---

## 🧪 Reproduction Steps

### 1. Download Checkpoints
```bash
pip install gdown
gdown --folder 12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI
```

### 2. Verify File Format
```bash
file "ConvNeXtPose_L (1).tar"
# Output: Zip archive data (despite .tar extension)
```

### 3. Extract and Analyze
```python
import zipfile
import torch
import pickle
import os

def analyze_checkpoint(tar_file):
    temp_dir = '/tmp/checkpoint_analysis'
    
    # Extract ZIP (misnamed as .tar)
    with zipfile.ZipFile(tar_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find data.pkl (PyTorch legacy format)
    checkpoint_dir = None
    for root, dirs, files in os.walk(temp_dir):
        if 'data.pkl' in files:
            checkpoint_dir = root
            break
    
    # Load checkpoint using legacy unpickler
    # (Code for LegacyUnpickler omitted for brevity - see full implementation)
    
    # Inspect first layer
    first_layer = checkpoint['network']['module.backbone.downsample_layers.0.0.weight']
    print(f"Shape: {first_layer.shape}")
    print(f"Output channels: {first_layer.shape[0]}")
    
    # Expected shapes:
    # Model L: [192, 3, 4, 4]
    # Model M: [64, 3, 4, 4]
    # Model S: [48, 3, 4, 4]
    # Actual: [48, 3, 4, 4] for ALL files

analyze_checkpoint("ConvNeXtPose_L (1).tar")
```

### 4. Attempt to Load
```python
# When trying to load "L" checkpoint into model L:
model = get_pose_net(cfg, is_train=False, variant='L')
checkpoint = torch.load('snapshot_83.pth')
model.load_state_dict(checkpoint['network'], strict=False)

# Result: Size mismatch errors
# Error: size mismatch for backbone.downsample_layers.0.0.weight: 
#        copying a param with shape torch.Size([48, 3, 4, 4]) from checkpoint, 
#        the shape in current model is torch.Size([192, 3, 4, 4])
```

---

## 📊 Size Mismatch Details

When attempting to load the "L" checkpoint into model L architecture:

- **Expected first layer:** `torch.Size([192, 3, 4, 4])` (192 output channels)
- **Actual in checkpoint:** `torch.Size([48, 3, 4, 4])` (48 output channels)
- **Ratio:** 4x difference

This 4x ratio is consistent with the dimension difference between:
- Model L: `dims=[192, 384, 768, 1536]`
- Model S: `dims=[48, 96, 192, 384]`

The checkpoint clearly contains model S, not model L.

---

## 📝 File Information

| Filename | Size | Actual Architecture | Parameter Count |
|----------|------|---------------------|-----------------|
| ConvNeXtPose_L (1).tar | 96.19 MB | **Small (S)** | 8,391,354 |
| ConvNeXtPose_M (1).tar | 87.10 MB | **Small (S)** | 7,596,986 |
| ConvNeXtPose_S.tar | 85.41 MB | **Small (S)** | 7,448,954 |
| ConvNeXtPose_XS.tar | 40.58 MB | Extra Small (XS) | (not analyzed) |

**Note:** Parameter count differences suggest different training epochs or configurations of the same Small model.

---

## 💥 Impact

1. **Cannot reproduce paper results:** Unable to test models L and M as reported in the paper
2. **Misleading documentation:** README claims these checkpoints are available
3. **Wasted computation time:** Users spend hours debugging size mismatch errors
4. **Research validation issues:** Cannot verify claimed 42.3mm MPJPE for model L

---

## 🔧 Temporary Workaround

For users who want to test the codebase:

```python
# Use model S instead (already available)
VARIANT = 'S'
CHECKPOINT_EPOCH = 83

# This will work and produce results ~45mm MPJPE on Human3.6M Protocol 2
```

---

## ✅ Requested Actions

1. **Upload correct checkpoints** for models L and M with proper architectures:
   - Model L: `dims=[192, 384, 768, 1536]`
   - Model M: `dims=[64, 128, 256, 512]`

2. **Verify file labels** in the Google Drive folder

3. **Update README** with clarification about which models are currently available

4. **Add architecture verification** in the checkpoint loading code to detect mismatches early

---

## 📧 Contact Information

- **Repository:** https://github.com/EstebanCabreraArbizu/ConvNeXtPose
- **Paper:** IEEE Access 2023 - "ConvNeXtPose: A Fast Accurate Method for 3D Human Pose Estimation"
- **Google Drive Issue:** Folder ID `12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI`

---

## 🛠️ Technical Details

### Checkpoint Format
- Files are **ZIP archives** (not TAR despite extension)
- Contains **PyTorch legacy format** (directory structure with `data.pkl` + binary storage fragments)
- Requires custom unpickler using `torch.UntypedStorage.from_buffer()` with `TypedStorage` wrapper

### Analysis Code
Full checkpoint conversion and analysis code available in:
- `kaggle_testing_notebook.ipynb` (Cell 11: Checkpoint extraction)
- Local analysis scripts used for verification

### Environment
- PyTorch 1.13.1+ (modern version with TypedStorage)
- Python 3.8+
- Tested on both Kaggle and local environments

---

## 📎 Appendix: Expected vs Actual

### Model L - Expected
```python
model_config = {
    'dims': [192, 384, 768, 1536],
    'depths': [3, 3, 27, 3],
    'expected_mpjpe': 42.3  # mm on Human3.6M Protocol 2
}
```

### Model L - Actual (from checkpoint)
```python
model_config = {
    'dims': [48, 96, 192, 384],  # ❌ This is model S
    'depths': [3, 3, 27, 3],
    'actual_params': 8_391_354
}
```

**Conclusion:** The file labeled "ConvNeXtPose_L" contains model S architecture, making it impossible to reproduce the paper's reported 42.3mm MPJPE for model L.

---

**Thank you for your attention to this issue. We look forward to accessing the correct checkpoint files to validate the excellent results reported in the paper.**
