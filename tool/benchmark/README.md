# Multi-Model Benchmark for 3D Human Pose Estimation

Unified benchmark pipeline for evaluating multiple 3D human pose estimation models on Human3.6M Protocol 2.

## Quick Start

### 1. Verify Ground Truth Availability

```bash
# Check if Human3.6M JSON annotations exist (includes Ground Truth from MoCap)
python3 tool/benchmark/extract_gt.py --verify

# View GT statistics (optional)
python3 tool/benchmark/extract_gt.py --protocol 2
```

**Important:** Ground truth is loaded automatically from `data/Human36M/annotations/*_joint_3d.json`. No manual NPZ creation needed!

### 2. Run Full Benchmark (All Models)

```bash
cd tool/benchmark
python3 run_benchmark.py --models convnextpose rootnet mobilehumanpose integral_human_pose --out ../../output/benchmark
```

### 3. Run with Dataset Loader (Recommended)

```bash
# Loads GT directly from JSON annotations
python3 -m tool.benchmark.runner_params \
    --dataset Human3.6M \
    --protocol 2 \
    --models rootnet convnextpose \
    --use-dataset-loader \
    --out output/benchmark
```

### 4. Run Subset of Models

```bash
python3 run_benchmark.py --models convnextpose rootnet --out ../../output/benchmark
```

### 5. Smoke Test (Quick Validation)

```bash
# Test metrics module
python3 test_metrics.py

# Test individual model wrapper (example)
python3 -c "
from models import RootNetWrapper, RootNetConfig
cfg = RootNetConfig()
wrapper = RootNetWrapper(cfg)
print('RootNet wrapper initialized successfully')
"
```

## Architecture Overview

```
tool/benchmark/
├── __init__.py           # Package exports
├── metrics.py            # MPJPE/PA-MPJPE with Procrustes alignment + validation
├── pipeline.py           # 5-step contract: prepare→download→infer→metrics→log + protocol splits
├── report.py             # JSON/Markdown/plots generation
├── checkpoints_manifest.json # RootNet/Integral snapshots mapping (URLs, hashes/sizes placeholders)
├── run_benchmark.py      # CLI harness for orchestration
├── test_metrics.py       # Smoke tests for metrics
├── DEPENDENCIES.md       # External repos, checkpoints, licenses
├── IMPLEMENTATION_SUMMARY.md # Implementation details
├── README.md             # This file
└── models/
    ├── __init__.py
    ├── rootnet.py        # XY heatmap + Z regression (camera-aware), manifest-aware
    ├── mobilehumanpose.py # MobileNetV2-based (LpNetSkiConcat)
    └── integral_pose.py  # ResNet + volumetric soft-argmax (mks0601/Jimmy sources)
```

## Pipeline Contract (5 Steps)

All models follow this shared contract:

## Ground Truth Preparation

**IMPORTANT:** Ground truth is **automatically loaded from dataset JSON files**. You do NOT need to create separate NPZ files.

### How it works:

```python
# The GT is already in the dataset JSON annotations:
data/Human36M/annotations/
├── Human36M_subject9_joint_3d.json   # ← Ground truth (MoCap data)
├── Human36M_subject11_joint_3d.json  # ← Ground truth (MoCap data)
└── ...

# The dataset loader reads these automatically:
from data.Human36M.Human36M import Human36M
dataset = Human36M(data_split='test')  # Loads GT from JSON
for sample in dataset.data:
    gt_3d = sample['joint_cam']  # ← (17, 3) in mm, camera coords
```

### For benchmark scripts:

```bash
# Option 1: Use dataset loader (RECOMMENDED - loads GT from JSON automatically)
python -m tool.benchmark.runner_params \
    --dataset Human3.6M \
    --protocol 2 \
    --use-dataset-loader

# Option 2: Use pre-extracted NPZ (optional, if you want to cache GT)
python -m tool.benchmark.runner_params \
    --gt-npz path/to/gt.npz
```

The pipeline expects ground truth in the form:
- `(N, J, 3)` array in millimeters (camera coordinates)
- Root-aligned (pelvis at origin) before computing MPJPE

For Human3.6M Protocol 2:
- Test subjects: S9, S11
- Frame stride: 64
- GT is extracted from `Human36M_subject*_joint_3d.json` (MoCap annotations)
2. **Download Weights:** Fetch or locate official checkpoint
3. **Inference:** Predict 3D poses in mm, shape `(N, J, 3)`
4. **Compute Metrics:** Calculate MPJPE + PA-MPJPE with consistent root alignment
5. **Log Results:** Write JSON, Markdown, and plots to `output/benchmark/`

## Expected Baselines (Sanity Checks)

| Model | Config | MPJPE (mm) | Tolerance | Parameters |
|-------|--------|------------|-----------|------------|
| ConvNeXtPose | H36M P2 | 53 | ±6 mm | 3.53M - 8.39M |
| RootNet | H36M P2 | 57 | ±6 mm | ~25M |
| RootNet | H36M P1 | n/a | — | ~25M |
| RootNet | MuCo+COCO (x-dataset) | higher (degradation) | — | ~25M |
| MobileHumanPose | H36M P2 | 84 | ±6 mm | ~1-2M |
| Integral Human Pose | ResNet-152 + flip, H36M P2 | 56–57 | ±6 mm | 50M+ |

Note: Protocol 1 is not directly comparable with Protocol 2.

Results outside the tolerance window indicate potential issues with:
- Checkpoint loading
- Data preprocessing
- Coordinate system alignment
- Metric calculation

## Model-Specific Notes

### RootNet
- **Special Requirement:** Camera intrinsic parameter `k` (default: 1000.0)
- **Architecture:** ResNet-50 + XY branch (heatmap + spatial integral) + Z branch (GAP + gamma*k)
- **Checkpoint Format:** PyTorch `.pth.tar` with `model` key

### MobileHumanPose
- **Variant:** LpNetSkiConcat (default, best accuracy)
- **Architecture:** MobileNetV2 encoder + DeConv decoder + skip connections + PReLU activations
- **Checkpoint Format:** PyTorch `.pth` or ONNX `.onnx`
- **Width Multiplier:** Must match checkpoint (default: 1.0)

### Integral Human Pose
- **Backbone Options:** ResNet-50/101/152
- **Architecture:** ResNet + 3 deconv layers + volumetric soft-argmax integration
- **Depth Dimension:** 64 (volumetric heatmap resolution)
- **Innovation:** Differentiable integral over probability distributions

## Adding a New Model

1. Create wrapper in `models/new_model.py`:
```python
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class NewModelConfig:
    checkpoint: str = None
    # ... other config

class NewModelWrapper:
    def __init__(self, cfg: NewModelConfig):
        self.cfg = cfg
        self.model = None

    def load_weights(self, path: str, device: str = "cpu") -> None:
        # Load checkpoint
        pass

    def infer(self, images: np.ndarray) -> np.ndarray:
        # Return (N, J, 3) in mm
        pass
```

2. Register in `models/__init__.py`:
```python
from .new_model import NewModelConfig, NewModelWrapper
__all__ = [..., "NewModelConfig", "NewModelWrapper"]
```

3. Wire into `run_benchmark.py`:
```python
# Add to contracts and fns lists
```

## Output Format

### Per-Model JSON (`{model}_summary.json`)
```json
{
  "name": "rootnet",
  "mpjpe_mm": 57.23,
  "pa_mpjpe_mm": 45.67,
  "expected_mpjpe_mm": 57.0,
  "params_million": 25.0,
  "fps": 30.5,
  "num_samples": 12345,
  "joints": 17
}
```

### Consolidated JSON (`summary.json`)
```json
{
  "dataset": "Human3.6M",
  "results": [
    { /* model 1 */ },
    { /* model 2 */ },
    ...
  ]
}
```

### Markdown Report (`summary.md`)
Auto-generated table comparing all models with expected vs observed MPJPE.

### Visualizations
- `mpjpe_bar.png`: Bar chart of MPJPE per model
- `acc_vs_params.png`: Scatter plot of accuracy vs model size

## Troubleshooting

### ImportError: No module named 'torchvision'
```bash
pip install torchvision
```

### CUDA out of memory
- Reduce batch size in data loaders
- Use smaller models (e.g., MobileHumanPose)
- Enable gradient checkpointing if training

### Checkpoint loading fails
- Verify checkpoint format matches wrapper expectations
- Check for `model` vs direct state_dict
- Use `strict=False` for partial loading

### MPJPE way off expected value
1. Check coordinate system (some models use different conventions)
2. Verify root alignment is applied consistently
3. Ensure GT and predictions are in same units (mm)
4. Check for missing joints or incorrect joint ordering

## Testing

### Run All Tests
```bash
python test_metrics.py
```

### Smoke Test Each Model
```bash
# Test imports
python -c "from models import RootNetWrapper, MobileHumanPoseWrapper, IntegralPoseWrapper; print('OK')"

# Test instantiation
python -c "
from models import *
configs = [RootNetConfig(), MobileHumanPoseConfig(), IntegralPoseConfig()]
wrappers = [RootNetWrapper(configs[0]), MobileHumanPoseWrapper(configs[1]), IntegralPoseWrapper(configs[2])]
print(f'All {len(wrappers)} wrappers instantiated successfully')
"
```

## Kaggle Execution

When running on Kaggle:

1. Upload checkpoints as Kaggle Datasets
2. Connect datasets to notebook
3. Use the notebook widgets (Protocol/ Dataset/ Checkpoints) to select configuration (see notebook cells added)
4. Or update paths in notebook cells:
```python
CHECKPOINT_PATHS = {
    'convnextpose': '/kaggle/input/convnextpose-weights/snapshot_L.pth.tar',
    'rootnet': '/kaggle/input/rootnet-weights/snapshot_19.pth.tar',  # P2 primary
    # Optional cross-dataset
    # 'rootnet': '/kaggle/input/rootnet-weights/snapshot_18.pth.tar',
    'mobilehumanpose': '/kaggle/input/mobilehumanpose-weights/lpnet_ski.pth',
    'integral_pose': '/kaggle/input/integral-pose-weights/snapshot_16.pth.tar',
}
```

5. Run benchmark cells sequentially
6. Download results from `output/benchmark/`

## References

See [DEPENDENCIES.md](DEPENDENCIES.md) for:
- Official repository links
- Paper citations
- License information
- Checkpoint download instructions

---

**Questions?** Check the main ConvNeXtPose repository or individual model repos for support.
