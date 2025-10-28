from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .metrics import MetricsResult, mpjpe, pa_mpjpe, EXPECTED_MPJPE, validate_expected_mpjpe

# Centralized dataset/protocol splits and sampling for Human3.6M
H36M_PROTOCOLS = {
    1: {
        "train_subjects": ["S1", "S5", "S6", "S7", "S8"],
        "test_subjects": ["S11"],
        "frame_stride": 1,
        "note": "Protocol 1 (not directly comparable to Protocol 2)",
    },
    2: {
        "train_subjects": ["S1", "S5", "S6", "S7", "S8"],
        "test_subjects": ["S9", "S11"],
        "frame_stride": 64,
        "note": "Protocol 2 standard (comparable).",
    },
}


def get_protocol_config(dataset: str, protocol: int) -> Dict[str, Any]:
    if dataset.lower() in ("h36m", "human36m", "human3.6m"):
        return H36M_PROTOCOLS[protocol]
    # Default fallback
    return {"train_subjects": None, "test_subjects": None, "frame_stride": None}


@dataclass
class PipelineContract:
    """Explicit contract for a model evaluation pipeline.

    This mirrors the 5-step flow used in the repo and notebook:
      1) prepare_data -> returns GT iterator
      2) download_weights -> returns path/handle to weights
      3) infer -> yields predictions aligned with GT
      4) compute_metrics -> MPJPE/PA-MPJPE
      5) log_results -> JSON/Markdown/artifacts
    """
    name: str
    dataset_name: str
    joints: int
    depth_dim: Optional[int] = None
    intrinsic_k: Optional[float] = None  # RootNet requires camera k for z
    protocol: int = 2  # default Protocol 2


@dataclass
class RunResult:
    name: str
    mpjpe_mm: float
    pa_mpjpe_mm: Optional[float]
    params_million: Optional[float] = None
    fps: Optional[float] = None
    expected_mpjpe_mm: Optional[float] = None
    num_samples: int = 0
    joints: int = 0
    protocol: Optional[int] = None
    dataset_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_metrics_shared(pred_3d: np.ndarray, gt_3d: np.ndarray, *, use_pa: bool = True) -> MetricsResult:
    """Compute shared metrics for 3D pose arrays of shape (N, J, 3)."""
    m = mpjpe(pred_3d, gt_3d, root_align=True)
    pa = pa_mpjpe(pred_3d, gt_3d) if use_pa else None
    N, J, _ = pred_3d.shape
    return MetricsResult(mpjpe=m, pa_mpjpe=pa, num_samples=N, joints=J)


def sanity_check(name: str, mpjpe_mm: float, *, config_key: Optional[str] = None, tolerance: float = 6.0) -> Tuple[bool, Optional[float]]:
    """Check measured MPJPE against expected with a tolerance (mm).

    If config_key is provided, use configuration-specific expected values.
    """
    if config_key:
        exp = EXPECTED_MPJPE.get(config_key)
    else:
        exp = EXPECTED_MPJPE.get(name.lower())
    if exp is None:
        return True, None
    return (abs(mpjpe_mm - exp) <= tolerance), exp


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_pipeline(
    contract: PipelineContract,
    prepare_data_fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
    download_weights_fn: Callable[[], Optional[str]],
    infer_fn: Callable[[Optional[str]], np.ndarray],
    *,
    use_pa: bool = True,
    out_dir: Path = Path("output/benchmark"),
    params_million: Optional[float] = None,
    fps: Optional[float] = None,
    expected_config_key: Optional[str] = None,
) -> RunResult:
    """Run the shared 5-step pipeline and persist a JSON summary.

    Returns a RunResult that includes expected MPJPE to ease comparisons.
    """
    # 1) Data
    gt_3d, meta = prepare_data_fn()  # gt_3d: (N,J,3) in mm; meta may include info

    # 2) Weights
    weights_path = download_weights_fn()

    # 3) Inference
    pred_3d = infer_fn(weights_path)

    # 4) Metrics
    metrics = compute_metrics_shared(pred_3d, gt_3d, use_pa=use_pa)
    ok, expected = sanity_check(contract.name, metrics.mpjpe, config_key=expected_config_key)

    # Emit validation alerts (non-fatal)
    validate_expected_mpjpe(contract.name, metrics.mpjpe, expected, tolerance=5.0)

    result = RunResult(
        name=contract.name,
        mpjpe_mm=metrics.mpjpe,
        pa_mpjpe_mm=metrics.pa_mpjpe,
        params_million=params_million,
        fps=fps,
        expected_mpjpe_mm=expected,
        num_samples=metrics.num_samples,
        joints=metrics.joints,
        protocol=contract.protocol,
        dataset_name=contract.dataset_name,
    )

    # 5) Logging
    write_json(out_dir / f"{contract.name}_summary.json", result.to_dict())

    # Also write a single consolidated file for simple consumption
    consolidated = {
        "dataset": contract.dataset_name,
        "protocol": contract.protocol,
        "results": [result.to_dict()],
    }
    write_json(out_dir / "summary.json", consolidated)

    return result


def run_many(
    contracts: Iterable[PipelineContract],
    fns: Iterable[Tuple[Callable, Callable, Callable]],  # tuples per contract
    *,
    out_dir: Path = Path("output/benchmark"),
    use_pa: bool = True,
    expected_config_keys: Optional[List[Optional[str]]] = None,
) -> List[RunResult]:
    results: List[RunResult] = []
    consolidated = {"results": [], "dataset": None, "protocol": None}

    for idx, (c, (prep, dl, infer)) in enumerate(zip(contracts, fns)):
        key = expected_config_keys[idx] if expected_config_keys else None
        res = run_pipeline(
            c,
            prepare_data_fn=prep,
            download_weights_fn=dl,
            infer_fn=infer,
            use_pa=use_pa,
            out_dir=out_dir,
            expected_config_key=key,
        )
        results.append(res)
        consolidated["results"].append(res.to_dict())
        consolidated["dataset"] = c.dataset_name
        consolidated["protocol"] = c.protocol

    write_json(out_dir / "summary.json", consolidated)
    return results
