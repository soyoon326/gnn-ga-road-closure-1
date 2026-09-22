from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PT_PER_IN = 72.27
COL_W = 244.0 / PT_PER_IN
FULL_W = 510.0 / PT_PER_IN

NETWORK_COLORS = {
    "A": "#0072B2",
    "B": "#E69F00",
    "C": "#009E73",
    "D": "#CC79A7",
    "J": "#D55E00",
}

MODE_COLORS = {
    "none": "#7F7F7F",
    "init": "#0072B2",
    "mutation": "#E69F00",
    "both": "#009E73",
}

MODE_HATCH = {"none": "", "init": "//", "mutation": "\\\\", "both": "xx"}

MODE_LABEL = {
    "none": "none",
    "init": "init",
    "mutation": "mutation",
    "both": "both",
}
MODE_ORDER = ["none", "init", "mutation", "both"]
NETWORK_ORDER = ["A", "B", "C", "D", "J"]


def setup() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "hatch.linewidth": 0.35,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def newfig(width: float = COL_W, height: float | None = None):
    if height is None:
        height = width * 0.68
    return plt.subplots(figsize=(width, height))


def save(fig, name: str, outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def style_axes(ax, ygrid: bool = True) -> None:
    if ygrid:
        ax.grid(axis="y", color="0.85", linestyle="-", zorder=0)
        ax.set_axisbelow(True)


def zero_line(ax, horizontal: bool = True) -> None:
    if horizontal:
        ax.axhline(0, color="0.2", linewidth=0.7, zorder=3)
    else:
        ax.axvline(0, color="0.2", linewidth=0.7, zorder=3)
