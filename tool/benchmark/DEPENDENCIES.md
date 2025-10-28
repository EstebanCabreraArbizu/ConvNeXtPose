# Multi-Model Benchmark: Dependencies & External Resources

This document lists all external repositories, checkpoints, licenses, and official sources used in the multi-model 3D human pose benchmark pipeline.

## Repository References

### ConvNeXtPose
- **Official Repository:** https://github.com/medialab-ku/ConvNeXtPose
- **Paper:** "ConvNeXtPose: A Fast Accurate Method for 3D Human Pose Estimation and its AR Fitness Application in Mobile Devices" (IEEE Access 2023)
- **License:** Check repository for license terms
- **Checkpoints:** Available via Google Drive (see repository README)
- **Expected MPJPE:** ~53 mm (Protocol 2)
- **Parameters:** 3.53M (XS) to 8.39M (L)

### RootNet (3DMPPE)
- **Official Repository:** https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE
- **Paper:** "Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image" (ICCV 2019)
- **License:** Check repository for license terms
- **Checkpoints:** Available via official repository releases
- **Expected MPJPE:** ~57 mm (Protocol 2)
- **Parameters:** ~25M
- **Special Requirements:** Camera intrinsic parameter `k` for depth scaling (depth = gamma * k)

### MobileHumanPose
- **Official Repository:** https://github.com/SangbumChoi/MobileHumanPose
- **Paper:** "Lightweight 3D Human Pose Estimation Network Training Using Teacher-Student Learning"
- **License:** Check repository for license terms
- **Checkpoints:** 
  - Original: Available via repository Google Drive links
  - ONNX versions: https://github.com/PINTO0309/PINTO_model_zoo
- **Expected MPJPE:** ~84 mm (Protocol 2)
- **Parameters:** ~1-2M (highly efficient)
- **Variants:**
  - LpNetSkiConcat (default, best accuracy)
  - LpNetResConcat (balanced)
  - LpNetWoConcat (lightest)

### Integral Human Pose
- **Official Repository:** https://github.com/JimmySuen/integral-human-pose
- **Paper:** "Integral Human Pose Regression" (ECCV 2018)
- **Authors:** Xiao Sun, Bin Xiao, Fangyin Wei, Shuang Liang, Yichen Wei (MSRA)
- **License:** Check repository for license terms
- **Checkpoints:** Available via official repository
- **Expected MPJPE:** ~57 mm (ResNet-152 + flip test, Protocol 2)
- **Parameters:** 50M+ (ResNet-152)
- **Backbone Options:** ResNet-50, ResNet-101, ResNet-152

## Dataset: Human3.6M

- **Official Website:** http://vision.imar.ro/human3.6m/
- **Protocol 2 (used in benchmark):**
  - Train: S1, S5, S6, S7, S8
  - Test: S9, S11
  - Sampling: 1 frame every 64 frames
- **Annotation Format:** MS COCO format (as adapted by each repository)
- **Citation:**
  ```
  @article{h36m_pami,
    author = {Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu, Cristian},
    title = {Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments},
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year = {2014}
  }
  ```

## Checkpoint Locations (Kaggle/Local Setup)

When running on Kaggle, checkpoints should be uploaded as Kaggle Datasets or placed in:

```
/kaggle/input/
├── convnextpose-weights/
│   ├── snapshot_XS.pth.tar
│   ├── snapshot_S.pth.tar
│   ├── snapshot_M.pth.tar
│   └── snapshot_L.pth.tar
├── rootnet-weights/
│   └── snapshot_18.pth.tar
├── mobilehumanpose-weights/
│   ├── lpnet_ski.pth
│   ├── lpnet_res.pth
│   └── lpnet_wo.pth
└── integral-pose-weights/
    ├── resnet50_best.pth
    ├── resnet101_best.pth
    └── resnet152_best.pth
```

For local execution, place checkpoints in:
```
output/benchmark/checkpoints/
└── (same structure as above)
```

## Dependencies

### Core Libraries
- **PyTorch:** >= 1.10 (with CUDA support recommended)
- **NumPy:** >= 1.19
- **Matplotlib:** >= 3.3 (for visualizations)
- **torchvision:** >= 0.11 (for ResNet backbones)

### Optional (for ONNX support)
- **onnx:** >= 1.10
- **onnxruntime-gpu:** >= 1.10 (for MobileHumanPose ONNX inference)

### Installation
```bash
pip install torch torchvision numpy matplotlib
# Optional for ONNX:
pip install onnx onnxruntime-gpu
```

## Benchmark Outputs

All benchmark results are saved to:
```
output/benchmark/
├── {model_name}_summary.json  # Per-model results
├── summary.json                # Consolidated results
├── summary.md                  # Markdown report
├── mpjpe_bar.png              # MPJPE comparison bar chart
└── acc_vs_params.png          # Accuracy vs parameters scatter plot
```

## License & Attribution

This benchmark pipeline integrates multiple open-source projects. When using this code:

1. **Cite all relevant papers** (see references above)
2. **Respect individual repository licenses**
3. **Acknowledge the original authors** in any publications or derivative works

For questions about specific model licenses, refer to the official repositories listed above.

## Contact & Support

For issues specific to:
- **ConvNeXtPose:** See https://github.com/medialab-ku/ConvNeXtPose/issues
- **RootNet:** See https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/issues
- **MobileHumanPose:** See https://github.com/SangbumChoi/MobileHumanPose/issues
- **Integral Human Pose:** See https://github.com/JimmySuen/integral-human-pose/issues

For benchmark pipeline issues, refer to the main ConvNeXtPose repository.

---

**Last Updated:** October 2025
