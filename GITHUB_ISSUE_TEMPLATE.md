# GitHub Issue Template

---

## Title
🐛 [Bug] Checkpoint Files Mislabeled - Models L and M contain Model S architecture

---

## Labels
- `bug`
- `documentation`
- `help wanted`

---

## Description

### 🔍 Problem
The checkpoint files in the official Google Drive are mislabeled. Files named as Model L and Model M actually contain the Small model architecture, preventing reproduction of the paper's results.

### 📊 Expected Behavior
- **ConvNeXtPose_L.tar** should contain model with `dims=[192, 384, 768, 1536]`
- **ConvNeXtPose_M.tar** should contain model with `dims=[64, 128, 256, 512]`

### 🐛 Actual Behavior
All three files (L, M, S) contain architecture with `dims=[48, 96, 192, 384]` (Small model)

### 📝 Evidence

```python
# Analysis of downloaded checkpoints:

ConvNeXtPose_L (1).tar:
  ✓ First layer: torch.Size([48, 3, 4, 4])  # Should be [192, 3, 4, 4]
  ✓ Architecture: dims=[48, 96, 192, 384]    # ❌ Model S, not L
  ✓ Parameters: 8,391,354

ConvNeXtPose_M (1).tar:
  ✓ First layer: torch.Size([48, 3, 4, 4])  # Should be [64, 3, 4, 4]
  ✓ Architecture: dims=[48, 96, 192, 384]    # ❌ Model S, not M
  ✓ Parameters: 7,596,986
```

### 🔄 Reproduction Steps

1. Download checkpoints from Google Drive (ID: `12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI`)
2. Extract ZIP files (misnamed as .tar)
3. Load checkpoint using PyTorch legacy unpickler
4. Inspect first layer shape: `checkpoint['network']['module.backbone.downsample_layers.0.0.weight'].shape`
5. Result: `[48, 3, 4, 4]` for all three files

### 💥 Impact

- Cannot reproduce paper's reported 42.3mm MPJPE for Model L
- Cannot reproduce paper's reported 44.6mm MPJPE for Model M
- Size mismatch errors when attempting to load:
  ```
  RuntimeError: size mismatch for backbone.downsample_layers.0.0.weight: 
  copying a param with shape torch.Size([48, 3, 4, 4]) from checkpoint, 
  the shape in current model is torch.Size([192, 3, 4, 4])
  ```

### 🔧 Environment

- Python 3.8+
- PyTorch 1.13.1+
- Google Drive folder: https://drive.google.com/drive/folders/12H7zkLvmJtrkCmAUAPkQ6788WAnO60gI
- Tested on: Kaggle (GPU T4 x2) and local Linux

### ✅ Requested Fix

Please upload the correct checkpoint files:
1. **Model L** with architecture `dims=[192, 384, 768, 1536]`
2. **Model M** with architecture `dims=[64, 128, 256, 512]`

### 📎 Additional Resources

For complete technical analysis, see: [CHECKPOINT_MISLABELING_ISSUE.md](./CHECKPOINT_MISLABELING_ISSUE.md)

### 🙏 Note

Thank you for making this code publicly available! The codebase works excellently with Model S. We're eager to test models L and M once the correct checkpoints are available.

---

## Workaround (Temporary)

For users who want to test the codebase now:

```python
# Use Model S (works correctly)
VARIANT = 'S'
CHECKPOINT_EPOCH = 83

# This will produce results ~45mm MPJPE on Human3.6M Protocol 2
```

---

**Would appreciate any update on when the correct Model L and M checkpoints might be available. Thank you!**
