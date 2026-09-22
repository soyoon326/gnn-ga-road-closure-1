from __future__ import annotations
import os
K0_MERGED = os.environ.get('K0_MERGED', './merged_analysis_38_k0')
K0_RUNSFX = os.environ.get('K0_RUNSFX', '_pm01k0')
K0_OUT = os.environ.get('K0_OUT', 'manuscript_numbers_k0')
K0_REEVAL = os.environ.get('K0_REEVAL', 'reeval_39_k0')


import argparse
import csv
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import paper_style as ps

PENALTY = 1e9

INSTANCES = [
    ("A_k4_3od_x1.0", "A  $(|E_{\\mathrm{cand}}|\\,=\\,38)$"),
    ("B_k9_3od_x1.0", "B  $(|E_{\\mathrm{cand}}|\\,=\\,86)$"),
    ("J_k9_6od_x1.0", "J  $(|E_{\\mathrm{cand}}|\\,=\\,1{,}415)$"),
]
METHODS = [
    ("none", "none (GA only)"),
    ("init", "init"),
    ("mutation", "mutation"),
    ("both", "both"),
    ("surrogate", "surrogate fitness"),
    ("heuristic", "structural heuristic"),
    ("sa", "simulated annealing"),
]
ARM_COLOR = {
    "none": ps.MODE_COLORS["none"],
    "init": ps.MODE_COLORS["init"],
    "mutation": ps.MODE_COLORS["mutation"],
    "both": ps.MODE_COLORS["both"],
    "surrogate": "#CC79A7",
    "heuristic": "#56B4E9",
    "sa": "#D55E00",
}


def load_modes(root: Path) -> dict[tuple[str, str], list[float]]:
    out = defaultdict(list)
    path = Path(K0_MERGED) / "seed_best_merged.csv"
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                att = float(r["ATT"])
            except (KeyError, ValueError):
                continue
            if att < PENALTY:
                out[(r["instance_id"], r["mode"])].append(att)
    return out


REPORTED_SURROGATE = {
    "A_k4_3od_x1.0": "reviewer_diagnostics/results/arm_A_sum_run38/arm_results.csv",
    "B_k9_3od_x1.0": "reviewer_diagnostics/results/arm_B_sum_run/arm_results.csv",
    "J_k9_6od_x1.0": "reviewer_diagnostics/results/arm_J_sum.csv",
}


def load_reported_surrogate(root: Path, inst: str) -> list[float]:
    best: dict[int, float] = {}
    with open(root / REPORTED_SURROGATE[inst], encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            att = float(r["att"])
            if att < PENALTY:
                s = int(r["seed"])
                best[s] = min(best.get(s, float("inf")), att)
    return [best[s] for s in sorted(best)]


def load_arms(root: Path) -> dict[tuple[str, str], list[float]]:
    out = defaultdict(list)
    for inst, _ in INSTANCES:
        out[(inst, "surrogate")] = load_reported_surrogate(root, inst)
        base = root / ("runs_reviewer_A38" if inst == "A_k4_3od_x1.0" else "runs_reviewer")
        for arm in ("heuristic", "sa"):
            for d in sorted((base / inst).glob(f"{arm}_seed*")):
                f = d / "best.txt"
                if not f.exists():
                    continue
                m = re.search(r"best_score=([0-9.]+)",
                              f.read_text(encoding="utf-8", errors="ignore"))
                if m and float(m.group(1)) < PENALTY:
                    out[(inst, arm)].append(float(m.group(1)))
    return out


def fig_baselines(root: Path, outdir: Path) -> None:
    vals = {**load_modes(root), **load_arms(root)}
    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL_W, 2.35))
    rng = np.random.default_rng(1)

    for ax, (inst, title) in zip(axes, INSTANCES):
        ax.grid(axis="x", color="0.88", linestyle="-", zorder=0)
        ax.set_axisbelow(True)
        for row, (key, _label) in enumerate(METHODS):
            y = len(METHODS) - 1 - row
            v = vals.get((inst, key), [])
            if not v:
                continue
            ax.scatter(v, y + rng.uniform(-0.16, 0.16, len(v)), s=7,
                       color=ARM_COLOR[key], alpha=0.75, linewidths=0, zorder=3)
            ax.scatter([st.mean(v)], [y], s=34, marker="|",
                       color="black", linewidths=1.1, zorder=4)
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([lab for _, lab in METHODS][::-1])
        ax.set_ylim(-0.6, len(METHODS) - 0.4)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("Best ATT (s)")
        ax.tick_params(axis="x", labelsize=6.5)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axes[0].tick_params(axis="y", labelsize=7)
    fig.subplots_adjust(wspace=0.12)
    ps.save(fig, "fig08_baseline_comparison", outdir)

    for inst, _ in INSTANCES:
        s = {k: round(st.mean(vals[(inst, k)]), 2)
             for k, _ in METHODS if vals.get((inst, k))}
        print(f"  {inst}: {s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    ps.setup()
    root, outdir = Path(args.root), Path(args.outdir)
    print("F2: baseline comparison")
    fig_baselines(root, outdir)


if __name__ == "__main__":
    main()
