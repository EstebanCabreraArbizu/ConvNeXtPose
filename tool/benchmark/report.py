from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def render_markdown(results: List[Dict[str, Any]], dataset: str) -> str:
    lines = []
    lines.append(f"# Benchmark resumen – {dataset}\n")
    lines.append("| Modelo | MPJPE (mm) | MPJPE esperado (mm) | PA-MPJPE (mm) | Params (M) | FPS | Samples |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for r in results:
        lines.append(
            f"| {r['name']} | {r['mpjpe_mm']:.2f} | {r.get('expected_mpjpe_mm','-')} | "
            f"{('-' if r.get('pa_mpjpe_mm') is None else f'{r['pa_mpjpe_mm']:.2f}')} | "
            f"{('-' if r.get('params_million') is None else r['params_million'])} | "
            f"{('-' if r.get('fps') is None else r['fps'])} | {r.get('num_samples','-')} |"
        )
    return "\n".join(lines) + "\n"


def plot_mpjpe_bar(results: List[Dict[str, Any]], out_path: Path) -> Optional[Path]:
    try:
        names = [r["name"] for r in results]
        vals = [r["mpjpe_mm"] for r in results]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(names, vals, color="skyblue")
        ax.bar_label(bars, fmt="%.0f mm")
        ax.set_ylabel("MPJPE (mm, menor es mejor)")
        ax.set_title("Comparativa de MPJPE")
        plt.xticks(rotation=30, ha="right")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
    except Exception:
        return None


def plot_accuracy_vs_params(results: List[Dict[str, Any]], out_path: Path) -> Optional[Path]:
    try:
        xs, ys, lbl = [], [], []
        for r in results:
            if r.get("params_million") is not None:
                xs.append(r["params_million"])
                ys.append(r["mpjpe_mm"])
                lbl.append(r["name"])
        if not xs:
            return None
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(xs, ys, c="tab:orange")
        for x, y, t in zip(xs, ys, lbl):
            ax.text(x, y, t)
        ax.set_xlabel("Parámetros (M)")
        ax.set_ylabel("MPJPE (mm, menor es mejor)")
        ax.set_title("Precisión vs Tamaño del modelo")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
    except Exception:
        return None


def consolidate_and_save(summary_path: Path, out_dir: Path) -> Dict[str, Any]:
    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    dataset = data.get("dataset", "unknown")
    results = data.get("results", [])

    md = render_markdown(results, dataset)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")

    plot_mpjpe_bar(results, out_dir / "mpjpe_bar.png")
    plot_accuracy_vs_params(results, out_dir / "acc_vs_params.png")

    return {"dataset": dataset, "results": results}
