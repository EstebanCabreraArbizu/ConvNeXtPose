"""Model wrappers for multi-model benchmark.

Each wrapper exposes:
  - Config dataclass with model-specific parameters
  - load_weights(path, device) to load official checkpoints
  - infer(images) -> (N, J, 3) predictions in mm

Available models:
  - RootNet: XY heatmap + Z regression (camera-aware)
  - MobileHumanPose: Lightweight MobileNetV2-based (LpNetSkiConcat)
  - IntegralPose: ResNet + volumetric soft-argmax
"""

from .rootnet import RootNetConfig, RootNetWrapper
from .mobilehumanpose import MobileHumanPoseConfig, MobileHumanPoseWrapper
from .integral_pose import IntegralPoseConfig, IntegralPoseWrapper

__all__ = [
    "RootNetConfig",
    "RootNetWrapper",
    "MobileHumanPoseConfig",
    "MobileHumanPoseWrapper",
    "IntegralPoseConfig",
    "IntegralPoseWrapper",
]
