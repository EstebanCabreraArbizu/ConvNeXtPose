from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class IntegralPoseConfig:
    backbone: str = "resnet50"  # resnet50|101|152
    deconv_layers: int = 3
    checkpoint: Optional[str] = None
    num_joints: int = 17
    depth_dim: int = 64  # volumetric depth resolution
    input_size: Tuple[int, int] = (256, 256)
    source: str = "mks0601"  # 'mks0601' (recommended) or 'jimmy'
    normalize_h36m: bool = True  # apply RootNet-like normalization for H36M P2


class DeconvHead(nn.Module):
    """Deconv head: 3 transposed convs + final conv for volumetric heatmaps."""

    def __init__(
        self,
        in_channels: int = 2048,
        num_deconv: int = 3,
        num_joints: int = 17,
        depth_dim: int = 64,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.depth_dim = depth_dim

        # 3 deconv layers (each upsamples 2x)
        deconv_layers = []
        in_ch = in_channels
        for i in range(num_deconv):
            out_ch = 256
            deconv_layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch

        self.deconv_layers = nn.Sequential(*deconv_layers)

        # Final conv: produce joint_num * depth_dim channels
        self.final_conv = nn.Conv2d(256, num_joints * depth_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2048, H/32, W/32) from backbone
        Returns:
            (B, J*D, H', W') volumetric heatmaps
        """
        x = self.deconv_layers(x)  # (B, 256, H/4, W/4) after 3 deconvs
        x = self.final_conv(x)     # (B, J*D, H/4, W/4)
        return x


class IntegralPoseModel(nn.Module):
    """Integral Human Pose: ResNet backbone + deconv head + soft-argmax integration."""

    def __init__(
        self,
        backbone: str = "resnet50",
        num_joints: int = 17,
        depth_dim: int = 64,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.depth_dim = depth_dim

        # Backbone
        from torchvision.models import resnet50, resnet101, resnet152

        if backbone == "resnet50":
            resnet = resnet50(pretrained=False)
        elif backbone == "resnet101":
            resnet = resnet101(pretrained=False)
        elif backbone == "resnet152":
            resnet = resnet152(pretrained=False)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Up to layer4
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.head = DeconvHead(
            in_channels=2048,
            num_deconv=3,
            num_joints=num_joints,
            depth_dim=depth_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) RGB image
        Returns:
            (B, J*D, H', W') volumetric heatmaps
        """
        feat = self.backbone(x)  # (B, 2048, H/32, W/32)
        heatmaps = self.head(feat)  # (B, J*D, H', W')
        return heatmaps


def soft_argmax_3d_integral(
    heatmaps: torch.Tensor,
    joint_num: int,
    depth_dim: int,
) -> torch.Tensor:
    """Convert volumetric heatmaps to continuous 3D coords via integral (soft-argmax).
    
    This is the core innovation of Integral Human Pose: differentiable integration
    over probability distributions instead of discrete argmax.
    
    Args:
        heatmaps: (B, J*D, H, W) volumetric heatmaps
        joint_num: number of joints
        depth_dim: depth dimension
    Returns:
        (B, J, 3) continuous 3D coordinates
    """
    B, _, H, W = heatmaps.shape

    # Reshape to (B, J, D, H, W)
    hm = heatmaps.reshape(B, joint_num, depth_dim, H, W)

    # Flatten spatial-depth and softmax (normalize to probability)
    hm_flat = hm.reshape(B, joint_num, depth_dim * H * W)
    hm_prob = F.softmax(hm_flat, dim=2)
    hm_prob = hm_prob.reshape(B, joint_num, depth_dim, H, W)

    # Marginal distributions
    accu_x = hm_prob.sum(dim=(2, 3))  # (B, J, W)
    accu_y = hm_prob.sum(dim=(2, 4))  # (B, J, H)
    accu_z = hm_prob.sum(dim=(3, 4))  # (B, J, D)

    # Indices
    device = heatmaps.device
    idx_x = torch.arange(W, dtype=torch.float32, device=device)
    idx_y = torch.arange(H, dtype=torch.float32, device=device)
    idx_z = torch.arange(depth_dim, dtype=torch.float32, device=device)

    # Expected values (integral)
    coord_x = (accu_x * idx_x).sum(dim=2, keepdim=True)  # (B, J, 1)
    coord_y = (accu_y * idx_y).sum(dim=2, keepdim=True)  # (B, J, 1)
    coord_z = (accu_z * idx_z).sum(dim=2, keepdim=True)  # (B, J, 1)

    coords = torch.cat([coord_x, coord_y, coord_z], dim=2)  # (B, J, 3)
    return coords


class IntegralPoseWrapper:
    """Wrapper for Integral Human Pose Regression (ResNet + deconvs + soft-argmax)."""

    def __init__(self, cfg: IntegralPoseConfig):
        self.cfg = cfg
        self.model = None
        self.device = torch.device("cpu")

    def load_weights(self, path: Optional[str], device: str = "cpu") -> None:
        """Load official checkpoint."""
        self.cfg.checkpoint = path or self.cfg.checkpoint
        self.device = torch.device(device)

        self.model = IntegralPoseModel(
            backbone=self.cfg.backbone,
            num_joints=self.cfg.num_joints,
            depth_dim=self.cfg.depth_dim,
        ).to(self.device)

        if self.cfg.checkpoint:
            state_dict = torch.load(self.cfg.checkpoint, map_location=self.device)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()

    def infer(self, images: np.ndarray) -> np.ndarray:
        """Run inference: 3D heatmaps -> soft-argmax integration.
        
        Args:
            images: (N, 3, H, W) normalized RGB
        Returns:
            preds_3d: (N, J, 3) in mm
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_weights() first.")

        images_t = torch.from_numpy(images).to(self.device)

        # Optional normalization adjustment for H36M Protocol 2
        if self.cfg.normalize_h36m:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=images_t.dtype, device=images_t.device)[None, :, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], dtype=images_t.dtype, device=images_t.device)[None, :, None, None]
            images_t = (images_t - mean) / std

        with torch.no_grad():
            heatmaps = self.model(images_t)  # (N, J*D, H', W')
            coords = soft_argmax_3d_integral(
                heatmaps,
                self.cfg.num_joints,
                self.cfg.depth_dim,
            )

        return coords.cpu().numpy().astype(np.float32)
