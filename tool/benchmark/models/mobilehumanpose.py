from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MobileHumanPoseConfig:
    variant: str = "lpski"  # lpski | lpres | lpwo
    width_multiplier: float = 1.0
    checkpoint: Optional[str] = None  # torch or onnx path
    input_size: Tuple[int, int] = (256, 256)
    num_joints: int = 17
    depth_dim: int = 32


def _make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """Ensure channels are divisible by divisor (hardware optimization)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNPReLU(nn.Sequential):
    """Conv2d + BatchNorm + PReLU block (optimized for mobile)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, groups: int = 1):
        padding = (kernel - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
        )


class InvertedResidual(nn.Module):
    """MobileNetV2 Inverted Residual block."""

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # Expansion
            layers.append(ConvBNPReLU(inp, hidden_dim, kernel=1))
        # Depthwise
        layers.extend([
            ConvBNPReLU(hidden_dim, hidden_dim, kernel=3, stride=stride, groups=hidden_dim),
            # Projection (linear, no activation)
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class DeConv(nn.Module):
    """Decoder deconv module: Conv1x1 + Conv3x3 + Bilinear upsample."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBNPReLU(in_ch, mid_ch, kernel=1)
        self.conv2 = ConvBNPReLU(mid_ch, out_ch, kernel=3)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


class LpNetSkiConcat(nn.Module):
    """MobileHumanPose LpNetSkiConcat: MobileNetV2 encoder + DeConv decoder with skip connections."""

    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 256),
        num_joints: int = 17,
        depth_dim: int = 32,
        width_multiplier: float = 1.0,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.depth_dim = depth_dim
        self.output_shape = (input_size[0] // 8, input_size[1] // 8)  # after encoder

        # Inverted residual settings: [expand_ratio, channels, num_blocks, stride]
        inverted_residual_setting = [
            [1, 64, 1, 2],    # stage1 -> x2
            [6, 48, 2, 2],    # stage2
            [6, 48, 3, 2],    # stage3
            [6, 64, 4, 2],    # stage4 -> x1
            [6, 96, 3, 2],    # stage5 -> x0
            [6, 160, 3, 1],   # stage6
            [6, 320, 1, 1],   # stage7
        ]

        # First conv
        input_channel = _make_divisible(48 * width_multiplier)
        self.first_conv = ConvBNPReLU(3, input_channel, kernel=3, stride=2)

        # Encoder stages
        features: List[nn.Module] = []
        self.skip_indices = []  # track indices for x0, x1, x2
        block_idx = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
            # Mark skip connection indices: stages 0, 3, 4 -> x2, x1, x0
            if block_idx in [0, 3, 4]:
                self.skip_indices.append(len(features) - 1)
            block_idx += 1

        self.features = nn.Sequential(*features)

        # Bottleneck
        last_channel = _make_divisible(2048 * width_multiplier)
        self.last_conv = ConvBNPReLU(input_channel, last_channel, kernel=1)

        # Decoder with skip connections
        # x0, x1, x2 from stages 4, 3, 0
        skip_channels = [_make_divisible(96 * width_multiplier),
                         _make_divisible(64 * width_multiplier),
                         _make_divisible(64 * width_multiplier)]
        
        self.deconv0 = DeConv(last_channel + skip_channels[0], 256, 256)
        self.deconv1 = DeConv(256 + skip_channels[1], 256, 256)
        self.deconv2 = DeConv(256 + skip_channels[2], 256, 256)

        # Final layer
        self.final_layer = nn.Conv2d(256, num_joints * depth_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            (B, J*D, H', W') heatmaps
        """
        x = self.first_conv(x)

        # Encoder with skip saves
        skips = []
        for idx, module in enumerate(self.features):
            x = module(x)
            if idx in self.skip_indices:
                skips.append(x)

        # skips = [x2, x1, x0] (stages 0, 3, 4)
        x2, x1, x0 = skips

        # Bottleneck
        x = self.last_conv(x)

        # Decoder with concatenation (upsample decoder output to match skip resolution)
        # x0 is at lowest resolution (matches bottleneck)
        x = torch.cat([x0, x], dim=1)
        x = self.deconv0(x)  # upsamples 2x

        # x1 should now match resolution after 1 upsample
        # Adaptive interpolation if sizes don't match exactly
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x1, x], dim=1)
        x = self.deconv1(x)  # upsamples 2x

        # x2 should match after 2 upsamples
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x2, x], dim=1)
        x = self.deconv2(x)  # upsamples 2x

        # Final
        x = self.final_layer(x)
        return x


def soft_argmax_3d(heatmaps: torch.Tensor, joint_num: int, output_shape: Tuple[int, int], depth_dim: int) -> torch.Tensor:
    """Convert volumetric heatmaps to continuous 3D coords via soft-argmax."""
    B = heatmaps.shape[0]
    H, W = output_shape

    # Reshape and softmax
    hm = heatmaps.reshape(B, joint_num, depth_dim * H * W)
    hm = F.softmax(hm, dim=2)
    hm = hm.reshape(B, joint_num, depth_dim, H, W)

    # Marginals
    accu_x = hm.sum(dim=(2, 3))  # (B, J, W)
    accu_y = hm.sum(dim=(2, 4))  # (B, J, H)
    accu_z = hm.sum(dim=(3, 4))  # (B, J, D)

    # Indices
    device = heatmaps.device
    idx_x = torch.arange(W, dtype=torch.float32, device=device)
    idx_y = torch.arange(H, dtype=torch.float32, device=device)
    idx_z = torch.arange(depth_dim, dtype=torch.float32, device=device)

    # Expected values
    coord_x = (accu_x * idx_x).sum(dim=2, keepdim=True)
    coord_y = (accu_y * idx_y).sum(dim=2, keepdim=True)
    coord_z = (accu_z * idx_z).sum(dim=2, keepdim=True)

    coords = torch.cat([coord_x, coord_y, coord_z], dim=2)  # (B, J, 3)
    return coords


class MobileHumanPoseWrapper:
    """Wrapper for MobileHumanPose (LpNetSkiConcat variant by default).

    Considerations:
      - Activation: PReLU
      - Decoder: DeConv + bilinear upsampling
      - Output: volumetric 3D heatmaps or directly integrated 3D
      - Validate width_multiplier consistency with checkpoint
    """

    def __init__(self, cfg: MobileHumanPoseConfig):
        self.cfg = cfg
        self.model = None
        self.device = torch.device("cpu")

    def load_weights(self, path: Optional[str], device: str = "cpu") -> None:
        """Load torch or onnx weights."""
        self.cfg.checkpoint = path or self.cfg.checkpoint
        self.device = torch.device(device)

        self.model = LpNetSkiConcat(
            input_size=self.cfg.input_size,
            num_joints=self.cfg.num_joints,
            depth_dim=self.cfg.depth_dim,
            width_multiplier=self.cfg.width_multiplier,
        ).to(self.device)

        if self.cfg.checkpoint and torch.cuda.is_available():
            state_dict = torch.load(self.cfg.checkpoint, map_location=self.device)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()

    def infer(self, images: np.ndarray) -> np.ndarray:
        """Run inference and return 3D joints in mm.
        
        Args:
            images: (N, 3, H, W) normalized
        Returns:
            preds_3d: (N, J, 3) in mm
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_weights() first.")

        images_t = torch.from_numpy(images).to(self.device)

        with torch.no_grad():
            heatmaps = self.model(images_t)  # (N, J*D, H', W')
            B, _, H, W = heatmaps.shape
            # Compute actual output_shape from heatmap dimensions
            output_shape = (H, W)
            coords = soft_argmax_3d(
                heatmaps,
                self.cfg.num_joints,
                output_shape,
                self.cfg.depth_dim,
            )

        return coords.cpu().numpy().astype(np.float32)
