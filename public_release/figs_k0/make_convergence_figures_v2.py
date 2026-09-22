from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import paper_style as ps

MODE_LABEL = {"none": "GA-only", "both": "GA+GNN"}
MODE_COLOR = {"none": ps.MODE_COLORS["none"], "both": ps.MODE_COLORS["both"]}
MODES = ["none", "both"]


def load(path: Path):
    per_inst = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            per_inst[(r["mode"], int(r["gen"]))][r["instance"]].append(
                float(r["cumulative_improvement_rate"]))
    return {k: {i: st.mean(v) for i, v in d.items()} for k, d in per_inst.items()}


PENALTY = 1e9


def noclosure_from_aggregate(agg_csv: Path) -> dict[str, float]:
    vals: dict[str, set[float]] = defaultdict(set)
    with open(agg_csv, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                k, att = float(r["k_closed"]), float(r["ATT"])
            except (TypeError, ValueError):
                continue
            if k == 0 and r.get("Reason") == "OK" and att < PENALTY:
                vals[r["instance_id"]].add(round(att, 6))
    for inst, v in vals.items():
        if max(v) - min(v) > 1e-6:
            raise SystemExit(f"{inst}: k=0 evaluations disagree: {sorted(v)}")
    return {inst: min(v) for inst, v in vals.items()}


def load_from_aggregate(agg_csv: Path, baseline_csv: Path | None, max_gen: int = 15):
    if baseline_csv is None:
        baseline = noclosure_from_aggregate(agg_csv)
    else:
        baseline = {}
        with open(baseline_csv, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                baseline[r["name"]] = float(r["mean_duration"])

    best_at_gen: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    with open(agg_csv, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            att = r.get("ATT", "")
            if att in ("", "NA", None):
                continue
            try:
                att = float(att)
            except ValueError:
                continue
            if att >= PENALTY or str(r.get("is_valid", "")).lower() in ("false", "0"):
                continue
            gen = int(r["gen"])
            if gen > max_gen:
                continue
            key = (r["instance_id"], r["mode"], r["seed"])
            cur = best_at_gen[key].get(gen)
            if cur is None or att < cur:
                best_at_gen[key][gen] = att

    gens = list(range(0, max_gen + 1))
    per_inst = defaultdict(lambda: defaultdict(list))
    for (inst, mode, _seed), by_gen in best_at_gen.items():
        base = baseline.get(inst)
        if base is None:
            continue
        run_best = math.inf
        for g in gens:
            if g == 0:
                per_inst[(mode, 0)][inst].append(0.0)
                continue
            v = by_gen.get(g)
            if v is not None:
                run_best = min(run_best, v)
            if run_best is math.inf:
                continue
            per_inst[(mode, g)][inst].append((base - run_best) / base * 100.0)

    return {k: {i: st.mean(v) for i, v in d.items()} for k, d in per_inst.items()}


def curve(data, mode: str, gens: list[int]):
    means, ses = [], []
    for g in gens:
        vals = list(data[(mode, g)].values())
        means.append(st.mean(vals))
        ses.append(st.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0)
    return np.array(means), np.array(ses)


def fig_cumulative(data, gens, outdir: Path) -> None:
    fig, ax = ps.newfig(ps.COL_W, ps.COL_W * 0.68)
    ps.style_axes(ax)
    for mode in MODES:
        m, se = curve(data, mode, gens)
        ax.fill_between(gens, m - se, m + se, color=MODE_COLOR[mode],
                        alpha=0.18, linewidth=0, zorder=2)
        ax.plot(gens, m, "-o", color=MODE_COLOR[mode], markersize=2.2,
                label=MODE_LABEL[mode], zorder=3)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Improvement over\nno-closure baseline (%)")
    ax.set_xticks(range(0, max(gens) + 1, 3))
    ax.legend(loc="lower right")
    ps.save(fig, "fig2_cumulative_improvement_rate", outdir)


def incremental(data, mode, gens):
    m, _ = curve(data, mode, gens)
    return np.diff(m)


def fig_incremental(data, gens, outdir: Path) -> None:
    fig, ax = ps.newfig(ps.COL_W, ps.COL_W * 0.62)
    ps.style_axes(ax)
    g = gens[1:]
    w = 0.38
    for i, mode in enumerate(MODES):
        ax.bar(np.array(g) + (i - 0.5) * w, incremental(data, mode, gens), w,
               color=MODE_COLOR[mode], edgecolor="black",
               hatch=ps.MODE_HATCH[mode], label=MODE_LABEL[mode], zorder=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Additional improvement\n(percentage points)")
    ax.set_xticks(range(1, max(gens) + 1, 2))
    ax.legend(loc="upper right")
    ps.save(fig, "fig3_incremental_improvement_percentage_points", outdir)


def _fit_panel(ax, data, gens, mode: str):
    g = np.array(gens[1:], dtype=float)
    y = incremental(data, mode, gens)

    pos = y > 1e-9
    b0, loga0 = np.polyfit(g[pos], np.log(y[pos]), 1)
    a, b = math.exp(loga0), -b0
    try:
        from scipy.optimize import curve_fit
        (a, b), _ = curve_fit(lambda t, A, B: A * np.exp(-B * t), g, y,
                              p0=(a, b), maxfev=20000)
    except Exception:
        pass

    ps.style_axes(ax)
    ax.bar(g, y, 0.65, color=MODE_COLOR[mode], edgecolor="black",
           hatch=ps.MODE_HATCH[mode], label="observed", zorder=2)
    gg = np.linspace(g.min(), g.max(), 200)
    ax.plot(gg, a * np.exp(-b * gg), "-", color="black", linewidth=1.0,
            label=f"fit: ${a:.2f}\\,\\mathrm{{e}}^{{-{b:.2f}g}}$", zorder=3)
    ax.set_xlabel("Generation")
    ax.set_xticks(range(1, max(gens) + 1, 2))
    ax.legend(loc="upper right", title=MODE_LABEL[mode], title_fontsize=7)
    print(f"    {MODE_LABEL[mode]}: a={a:.3f}, decay b={b:.3f}")
    return a, b


def fig_fits(data, gens, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.1), sharey=True)
    for ax, mode in zip(axes, MODES):
        _fit_panel(ax, data, gens, mode)
    axes[0].set_ylabel("Additional improvement\n(percentage points)")
    fig.subplots_adjust(wspace=0.08)
    ps.save(fig, "fig3_incremental_improvement_fit", outdir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",
                    default="figures_convergence_final/all_metrics_used_data_final.csv",
                    help="legacy 28-instance metrics CSV")
    ap.add_argument("--aggregate", default=None,
                    help="merged_analysis_39/aggregate_merged.csv (39 instances)")
    ap.add_argument("--baseline", default=None,
                    help="legacy: take the no-closure ATT from this CSV "
                         "(baseline_39.csv) instead of the GA's own k=0 "
                         "evaluations in --aggregate")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    ps.setup()
    if args.aggregate and args.baseline:
        print(f"WARNING: --baseline {args.baseline} is a plain SUMO run, not the "
              "f(empty set) the article's Figures 6-8 use; omit --baseline to "
              "reproduce the published figures.")
    if args.aggregate:
        data = load_from_aggregate(Path(args.aggregate),
                                   Path(args.baseline) if args.baseline else None)
    else:
        data = load(Path(args.data))
    gens = sorted({g for _, g in data})
    n_inst = len(data[("none", gens[-1])])
    print(f"instances: {n_inst}, generations: {gens[0]}-{gens[-1]}")

    fig_cumulative(data, gens, Path(args.outdir))
    fig_incremental(data, gens, Path(args.outdir))
    fig_fits(data, gens, Path(args.outdir))

    for mode in MODES:
        m, se = curve(data, mode, gens)
        print(f"  {MODE_LABEL[mode]:8s} final={m[-1]:.2f}% (+/-{se[-1]:.2f} SE), "
              f"gen1={m[1]:.2f}%")


if __name__ == "__main__":
    main()
