#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from .pipeline import PipelineContract, run_many
from .metrics import EXPECTED_MPJPE


# Placeholder prep/download/infer for each model; to be wired with real code

def prep_data() -> Tuple[np.ndarray, dict]:
    # TODO: load Human3.6M gt in mm from existing loaders or npz
    # For now, return tiny dummy set to keep script runnable
    gt = np.zeros((2, 17, 3), dtype=np.float32)
    meta = {"dataset": "Human36M", "note": "dummy placeholder"}
    return gt, meta


def dl_weights_none() -> str | None:
    return None


def infer_zeros(_weights: str | None) -> np.ndarray:
    # Returns zeros predictions matching prep_data gt
    return np.zeros((2, 17, 3), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-model 3D pose benchmark")
    parser.add_argument("--models", nargs="*", default=["convnextpose", "rootnet", "mobilehumanpose", "integral_human_pose"],
                        help="Subset of models to run")
    parser.add_argument("--out", default="output/benchmark", help="Output dir")
    args = parser.parse_args()

    contracts = []
    fns = []

    for m in args.models:
        contracts.append(PipelineContract(name=m, dataset_name="Human3.6M", joints=17))
        fns.append((prep_data, dl_weights_none, infer_zeros))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_many(contracts, fns, out_dir=out_dir)

    print("Results:")
    for r in results:
        exp = EXPECTED_MPJPE.get(r.name)
        print(f"- {r.name}: MPJPE={r.mpjpe_mm:.2f} mm (expected ~{exp} mm)")


if __name__ == "__main__":
    main()
