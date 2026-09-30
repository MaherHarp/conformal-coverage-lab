"""Shared plotting style, output paths, and markdown helpers."""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from conformal.data import hetero_coverage  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

# Categorical order is fixed: a method keeps its colour and marker in every figure.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 6, "font.size": 10, "legend.frameon": False,
    })


def save(fig, name: str):
    FIG_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt(v, p=3):
    if isinstance(v, float) and not np.isfinite(v):
        return "inf"
    return f"{v:.{p}f}"


def union_coverage(x: float, intervals: np.ndarray) -> float:
    """Exact coverage of a disjoint union of intervals at a single x (heteroscedastic model)."""
    if len(intervals) == 0:
        return 0.0
    xs = np.full(len(intervals), x)
    return float(hetero_coverage(xs, intervals[:, 0], intervals[:, 1]).sum())
