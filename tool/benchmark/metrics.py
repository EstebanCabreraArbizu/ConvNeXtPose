from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class MetricsResult:
    mpjpe: float
    pa_mpjpe: Optional[float] = None
    num_samples: int = 0
    joints: int = 0

    def to_dict(self) -> Dict:
        return {
            "mpjpe_mm": float(self.mpjpe),
            "pa_mpjpe_mm": None if self.pa_mpjpe is None else float(self.pa_mpjpe),
            "num_samples": int(self.num_samples),
            "joints": int(self.joints),
        }

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def _reshape_inputs(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure inputs are (N, J, 3) float64 for numerical stability.

    Accepts shapes (N,J,3), (J,3), (N,3J) and casts to float64.
    Units: assumed millimeters.
    """
    p = np.asarray(pred)
    g = np.asarray(gt)
    if p.ndim == 2 and p.shape[1] == 3:
        p = p[None, ...]
    if g.ndim == 2 and g.shape[1] == 3:
        g = g[None, ...]
    # Flattened variant
    if p.ndim == 2 and p.shape[1] % 3 == 0:
        p = p.reshape(p.shape[0], -1, 3)
    if g.ndim == 2 and g.shape[1] % 3 == 0:
        g = g.reshape(g.shape[0], -1, 3)
    assert p.shape == g.shape and p.shape[-1] == 3, f"Invalid shapes: pred={p.shape}, gt={g.shape}"
    return p.astype(np.float64), g.astype(np.float64)


def mpjpe(pred: np.ndarray, gt: np.ndarray, *, root_align: bool = True) -> float:
    """Compute Mean Per Joint Position Error (MPJPE) in millimeters.

    Args:
        pred: (N, J, 3)
        gt: (N, J, 3)
        root_align: if True, subtract pelvis/root joint (index 0) before error.

    Returns:
        Scalar MPJPE in mm.
    """
    p, g = _reshape_inputs(pred, gt)
    if root_align:
        p = p - p[:, :1, :]
        g = g - g[:, :1, :]
    err = np.linalg.norm(p - g, axis=-1)  # (N, J)
    return float(err.mean())


def _procrustes_align(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Procrustes alignment (rigid + scale) per-sample using SVD.

    Shapes: p,g (N, J, 3) -> returns aligned p (N, J, 3)
    """
    N, J, _ = p.shape
    p0 = p - p.mean(axis=1, keepdims=True)
    g0 = g - g.mean(axis=1, keepdims=True)
    p_aligned = np.empty_like(p0)
    for i in range(N):
        X = p0[i]
        Y = g0[i]
        # Scale
        sX = np.sqrt((X ** 2).sum()) + 1e-8
        sY = np.sqrt((Y ** 2).sum()) + 1e-8
        Xn = X / sX
        Yn = Y / sY
        # Rotation via SVD
        H = Xn.T @ Yn
        U, S, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
        Z = (Xn @ R) * sY  # match scale of Y
        # Recenter to Y mean
        Z += g0[i].mean(axis=0, keepdims=True)
        p_aligned[i] = Z
    return p_aligned


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Procrustes Aligned MPJPE (a.k.a. P-MPJPE or PA-MPJPE) in mm."""
    p, g = _reshape_inputs(pred, gt)
    p_aligned = _procrustes_align(p, g)
    err = np.linalg.norm(p_aligned - g, axis=-1)
    return float(err.mean())


# Expected MPJPE baselines (mm)
EXPECTED_MPJPE = {
    # Generic
    "convnextpose": 53.0,
    "rootnet": 57.0,
    "mobilehumanpose": 84.0,
    "integral_human_pose": 57.0,  # ResNet-152 + flip test ref ~56.9
    # Configuration-specific keys
    "rootnet_h36m_p2": 57.0,
    "rootnet_h36m_p1": None,  # not comparable, skip validation
    "rootnet_muco_coco": None,  # cross-dataset; expect degradation
    "integral_r152_p2": 57.0,
}


def validate_expected_mpjpe(model_name: str, measured: float, expected: Optional[float], tolerance: float = 5.0) -> None:
    """Print a warning if measured MPJPE deviates from expected by > tolerance (mm).

    Non-fatal; intended for quick sanity alerts in logs.
    """
    if expected is None:
        return
    if abs(measured - expected) > tolerance:
        print(
            f"[WARN] {model_name}: MPJPE {measured:.2f} mm deviates from expected {expected:.2f} mm by > {tolerance:.1f} mm"
        )
