from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RootNetConfig:
    # Camera intrinsic scaling for z (depth = gamma * k)
    intrinsic_k: float = 1000.0
    # Checkpoint path if already downloaded
    checkpoint: Optional[str] = None
    # Backbone type (resnet50, resnet101, resnet152)
    backbone: str = "resnet50"
    # Output shape for XY heatmap
    output_shape: Tuple[int, int] = (64, 64)
    # Named preset to pick known checkpoints from manifest (e.g., 'snapshot_19.pth.tar')
    checkpoint_preset: Optional[str] = None


class RootNetXYBranch(nn.Module):
    """XY branch: 3 deconvs + heatmap + soft-argmax integral for 2D localization."""

    def __init__(self, in_channels: int = 2048, num_joints: int = 17, output_shape: Tuple[int, int] = (64, 64)):
        super().__init__()
        self.output_shape = output_shape
        self.num_joints = num_joints

        # 3 transposed convolutions: 8x8 -> 64x64
        self.deconv1 = nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Final 1x1 conv to produce heatmap per joint
        self.final_conv = nn.Conv2d(256, num_joints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2048, H, W) from backbone
        Returns:
            xy_coords: (B, J, 2) continuous xy coordinates via spatial integral
        """
        # Deconvolutions
        x = F.relu(self.bn1(self.deconv1(x)), inplace=True)
        x = F.relu(self.bn2(self.deconv2(x)), inplace=True)
        x = F.relu(self.bn3(self.deconv3(x)), inplace=True)

        # Heatmap
        heatmap = self.final_conv(x)  # (B, J, H, W)

        # Soft-argmax: spatial integration for continuous xy
        B, J, H, W = heatmap.shape
        heatmap_flat = heatmap.reshape(B, J, H * W)
        heatmap_prob = F.softmax(heatmap_flat, dim=2)  # normalize to prob distribution
        heatmap_prob = heatmap_prob.reshape(B, J, H, W)

        # Marginal distributions
        accu_x = heatmap_prob.sum(dim=2)  # (B, J, W)
        accu_y = heatmap_prob.sum(dim=3)  # (B, J, H)

        # Indices
        device = heatmap.device
        idx_x = torch.arange(W, dtype=torch.float32, device=device)
        idx_y = torch.arange(H, dtype=torch.float32, device=device)

        # Expected values
        coord_x = (accu_x * idx_x).sum(dim=2, keepdim=True)  # (B, J, 1)
        coord_y = (accu_y * idx_y).sum(dim=2, keepdim=True)  # (B, J, 1)

        xy_coords = torch.cat([coord_x, coord_y], dim=2)  # (B, J, 2)
        return xy_coords


class RootNetZBranch(nn.Module):
    """Z branch: GAP + conv1x1 predicts gamma; depth = gamma * k."""

    def __init__(self, in_channels: int = 2048, num_joints: int = 17):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(in_channels, num_joints, kernel_size=1)

    def forward(self, x: torch.Tensor, k: float = 1000.0) -> torch.Tensor:
        """
        Args:
            x: (B, 2048, H, W) from backbone
            k: camera intrinsic parameter
        Returns:
            z_coords: (B, J, 1) depth in mm
        """
        x = self.gap(x)  # (B, 2048, 1, 1)
        gamma = self.fc(x)  # (B, J, 1, 1)
        gamma = gamma.squeeze(-1).squeeze(-1).unsqueeze(-1)  # (B, J, 1)
        z_coords = gamma * k
        return z_coords


class RootNetModel(nn.Module):
    """Full RootNet: ResNet backbone + XY branch + Z branch."""

    def __init__(self, backbone: str = "resnet50", num_joints: int = 17, output_shape: Tuple[int, int] = (64, 64)):
        super().__init__()
        # Lazy import to avoid dependency at module load
        from torchvision.models import resnet50, resnet101, resnet152

        if backbone == "resnet50":
            resnet = resnet50(pretrained=False)
        elif backbone == "resnet101":
            resnet = resnet101(pretrained=False)
        elif backbone == "resnet152":
            resnet = resnet152(pretrained=False)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Backbone (up to layer4, before avgpool)
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

        self.xy_branch = RootNetXYBranch(in_channels=2048, num_joints=num_joints, output_shape=output_shape)
        self.z_branch = RootNetZBranch(in_channels=2048, num_joints=num_joints)

    def forward(self, x: torch.Tensor, k: float = 1000.0) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) RGB image
            k: camera intrinsic
        Returns:
            pred_3d: (B, J, 3) in mm
        """
        feat = self.backbone(x)  # (B, 2048, H/32, W/32)
        xy = self.xy_branch(feat)  # (B, J, 2)
        z = self.z_branch(feat, k=k)  # (B, J, 1)
        pred_3d = torch.cat([xy, z], dim=2)  # (B, J, 3)
        return pred_3d


class RootNetWrapper:
    """Thin wrapper around RootNet model with XY heatmap + Z regression branch.

    Notes per design (see claude_output.md):
      - XY branch: deconvs + heatmap + spatial integral
      - Z branch: GAP + conv1x1 predicts gamma; depth = gamma * k
    """

    def __init__(self, cfg: RootNetConfig):
        self.cfg = cfg
        self.model = None
        self.device = torch.device("cpu")

    def load_weights(self, path: Optional[str], device: str = "cpu") -> None:
        """Load official checkpoint (PyTorch state_dict).

        If `checkpoint_preset` is provided, try to resolve from checkpoints_manifest.json.
        """
        self.device = torch.device(device)

        # Resolve checkpoint from manifest if preset provided
        resolved_path = None
        if self.cfg.checkpoint_preset and not path:
            manifest_path = Path(__file__).resolve().parents[1] / "checkpoints_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                entry = manifest.get("rootnet", {}).get("snapshots", {}).get(self.cfg.checkpoint_preset)
                if entry and entry.get("url"):
                    # Expect user to have downloaded to output/benchmark/checkpoints/rootnet/
                    candidate = Path("output/benchmark/checkpoints/rootnet") / self.cfg.checkpoint_preset
                    if candidate.exists():
                        resolved_path = str(candidate)
        # Fallback to provided path or existing cfg.checkpoint
        self.cfg.checkpoint = path or resolved_path or self.cfg.checkpoint

        # Build model
        self.model = RootNetModel(
            backbone=self.cfg.backbone,
            num_joints=17,
            output_shape=self.cfg.output_shape,
        ).to(self.device)

        if self.cfg.checkpoint:
            state_dict = torch.load(self.cfg.checkpoint, map_location=self.device)
            # Handle different checkpoint formats
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()

    def infer(self, images: np.ndarray, *, k: Optional[float] = None) -> np.ndarray:
        """Run inference and return 3D joints in mm.

        Args:
            images: (N, C, H, W) normalized RGB, float32
            k: intrinsic parameter; overrides cfg.intrinsic_k if provided
        Returns:
            preds_3d: (N, J, 3) in mm
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_weights() first.")

        k_val = float(k if k is not None else self.cfg.intrinsic_k)
        images_t = torch.from_numpy(images).to(self.device)

        with torch.no_grad():
            preds = self.model(images_t, k=k_val)  # (N, J, 3)

        return preds.cpu().numpy().astype(np.float32)
