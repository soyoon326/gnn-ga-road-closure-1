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

GUIDED = ["init", "mutation", "both"]


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(row: dict, key: str):
    v = row.get(key, "")
    if v in ("", "NA", "nan", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def ci95(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    se = st.stdev(values) / math.sqrt(n)
    try:
        from scipy.stats import t
        crit = float(t.ppf(0.975, n - 1))
    except Exception:
        crit = 1.96
    return crit * se


def short_label(instance_id: str) -> str:
    parts = instance_id.split("_")
    net = parts[0]
    od = next((p for p in parts if p.endswith("od")), "")
    scale = next((p[1:] for p in parts if p.startswith("x")), "")
    return f"{net} {od} {scale}"


def fig02(data: Path, outdir: Path) -> None:
    rows = read_csv(data / "att_summary_by_mode_merged.csv")
    vals = defaultdict(list)
    for r in rows:
        if r["mode"] == "none":
            continue
        v = fnum(r, "mean_reduction_vs_none_%")
        if v is not None:
            vals[(r["instance_id"][0], r["mode"])].append(v)

    fig, ax = ps.newfig(ps.COL_W, ps.COL_W * 0.70)
    ps.style_axes(ax)
    nets = ps.NETWORK_ORDER
    x = np.arange(len(nets))
    w = 0.26
    rng = np.random.default_rng(0)
    for i, mode in enumerate(GUIDED):
        pos = x + (i - 1) * w
        means = [st.mean(vals[(n, mode)]) for n in nets]
        ax.bar(pos, means, w, color=ps.MODE_COLORS[mode], edgecolor="black",
               hatch=ps.MODE_HATCH[mode], label=mode, alpha=0.85, zorder=2)
        for xi, net in zip(pos, nets):
            v = vals[(net, mode)]
            jitter = rng.uniform(-w * 0.26, w * 0.26, size=len(v))
            ax.scatter(xi + jitter, v, s=3.5, color="0.15", alpha=0.75,
                       linewidths=0, zorder=4)
    ps.zero_line(ax)
    ax.set_xticks(x)
    n_tot = sum(len(vals[(m, 'both')]) for m in nets)
    ax.set_xticklabels([f"{n}\n({len(vals[(n,'both')])} of {n_tot})" for n in nets])
    ax.set_xlabel("Network")
    ax.set_ylabel("Mean ATT improvement vs.\nGA-only (\\%)".replace("\\%", "%"))
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              handletextpad=0.3, columnspacing=0.9, borderaxespad=0.2)
    ps.save(fig, "fig02_network_improvement", outdir)


def fig03(data: Path, outdir: Path) -> None:
    rows = read_csv(data / "att_summary_by_mode_merged.csv")
    pts = []
    for r in rows:
        if r["mode"] != "both":
            continue
        v = fnum(r, "mean_reduction_vs_none_%")
        if v is not None:
            pts.append((r["instance_id"], v))
    pts.sort(key=lambda t: t[1], reverse=True)
    n_pos = sum(1 for _, v in pts if v > 0)

    fig, ax = ps.newfig(ps.FULL_W, 2.95)
    ps.style_axes(ax)
    x = np.arange(len(pts))
    colors = [ps.NETWORK_COLORS[i[0]] for i, _ in pts]
    ax.bar(x, [v for _, v in pts], 0.75, color=colors,
           edgecolor="black", linewidth=0.4, zorder=2)
    ps.zero_line(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(i) for i, _ in pts], rotation=90, fontsize=6)
    ax.set_xlim(-0.8, len(pts) - 0.2)
    ax.set_ylabel("Final ATT improvement vs. GA-only (\\%)".replace("\\%", "%"))

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=ps.NETWORK_COLORS[n],
                             edgecolor="black", linewidth=0.4)
               for n in ps.NETWORK_ORDER]
    ax.legend(handles, ps.NETWORK_ORDER, ncol=5, loc="upper right",
              title="Network", title_fontsize=7)
    ax.annotate(f"{n_pos} of {len(pts)} instances above zero",
                xy=(0.5, 0.88), xycoords="axes fraction", ha="center", fontsize=7)
    ps.save(fig, "fig03_instance_both", outdir)


def _mode_panel(ax, values: dict[str, list[float]], ylabel: str,
                title: str | None = None, zero_ref: bool = False,
                xlabel: bool = True) -> None:
    ps.style_axes(ax)
    modes = [m for m in ps.MODE_ORDER if m in values]
    x = np.arange(len(modes))
    means = [st.mean(values[m]) for m in modes]
    errs = [ci95(values[m]) for m in modes]
    ax.bar(x, means, 0.6, yerr=errs, capsize=2.2,
           color=[ps.MODE_COLORS[m] for m in modes], edgecolor="black",
           hatch=[ps.MODE_HATCH[m] for m in modes],
           error_kw={"elinewidth": 0.6, "capthick": 0.6}, zorder=2)
    if zero_ref:
        ps.zero_line(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(modes if xlabel else [])
    if xlabel:
        ax.set_xlabel("Guidance mode")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=7.5)


def fig04(data: Path, outdir: Path):
    rows = read_csv(data / "extra_analysis" / "instance_extra_summary.csv")
    jac, top3 = defaultdict(list), defaultdict(list)
    for r in rows:
        for m in ps.MODE_ORDER:
            v = fnum(r, f"{m}_mean_pairwise_jaccard")
            if v is not None:
                jac[m].append(v)
            v = fnum(r, f"{m}_top3_edge_share")
            if v is not None:
                top3[m].append(v)

    fig, axes = plt.subplots(2, 1, figsize=(ps.COL_W, ps.COL_W * 0.92),
                             sharex=True)
    _mode_panel(axes[0], jac, "Mean pairwise\nJaccard similarity",
                title="(a) seed-best agreement", xlabel=False)
    _mode_panel(axes[1], top3, "Top-3 edge share",
                title="(b) edge concentration", xlabel=True)
    fig.subplots_adjust(hspace=0.30)
    ps.save(fig, "fig04_solution_family", outdir)
    return jac, top3


def fig05(data: Path, outdir: Path):
    conv = read_csv(data / "extra_analysis" / "convergence_summary_by_mode.csv")
    by_inst: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in conv:
        by_inst[r["instance_id"]][r["mode"]] = r

    att = read_csv(data / "att_summary_by_mode_merged.csv")
    final_gain = defaultdict(list)
    for r in att:
        if r["mode"] == "none":
            continue
        v = fnum(r, "mean_reduction_vs_none_%")
        if v is not None:
            final_gain[r["mode"]].append(v)

    gen95, auc = defaultdict(list), defaultdict(list)
    raw_gen95, raw_auc = defaultdict(list), defaultdict(list)
    for inst, modes in by_inst.items():
        base = modes.get("none")
        if base is None:
            continue
        b95 = fnum(base, "mean_first_hit_95pct_own_final_gen")
        bauc = fnum(base, "mean_progress_auc")
        for m in ps.MODE_ORDER:
            r = modes.get(m)
            if r is None:
                continue
            v = fnum(r, "mean_first_hit_95pct_own_final_gen")
            if v is not None:
                raw_gen95[m].append(v)
                if m != "none" and b95 is not None:
                    gen95[m].append(b95 - v)
            v = fnum(r, "mean_progress_auc")
            if v is not None:
                raw_auc[m].append(v)
                if m != "none" and bauc is not None:
                    auc[m].append(v - bauc)

    fig, axes = plt.subplots(3, 1, figsize=(ps.COL_W, ps.COL_W * 1.22),
                             sharex=True)
    _mode_panel(axes[0], final_gain, "Final ATT\nimprovement (%)",
                title="(a) final quality", zero_ref=True, xlabel=False)
    _mode_panel(axes[1], gen95, "Generations saved to\nreach 95% of final gain",
                title="(b) convergence speed", zero_ref=True, xlabel=False)
    _mode_panel(axes[2], auc, "Progress AUC gain",
                title="(c) overall progress", zero_ref=True, xlabel=True)
    fig.subplots_adjust(hspace=0.38)
    ps.save(fig, "fig05_search_dynamics", outdir)
    return raw_gen95, raw_auc


def fig06(data: Path, outdir: Path) -> None:
    rows = read_csv(data / "extra_analysis" / "diversity_vs_gain_scatter_ready.csv")
    corr = {}
    for r in read_csv(data / "extra_analysis" / "diversity_vs_gain_correlations.csv"):
        v = fnum(r, "corr_gain_vs_diversity_drop_spearman")
        if v is not None:
            corr[r["mode"]] = v

    pts = []
    for r in rows:
        if r["mode"] != "both":
            continue
        d = fnum(r, "delta_gen1_unique_ratio_vs_none")
        g = fnum(r, "mean_reduction_vs_none_%")
        if d is None or g is None:
            continue
        pts.append((r["instance_id"], d, g))

    fig, ax = ps.newfig(ps.COL_W, ps.COL_W * 0.78)
    ps.style_axes(ax)
    ax.grid(axis="x", color="0.85", linestyle="-", zorder=0)
    for net in ps.NETWORK_ORDER:
        sel = [(x, y) for i, x, y in pts if i[0] == net]
        if not sel:
            continue
        ax.scatter([p[0] for p in sel], [p[1] for p in sel], s=16,
                   color=ps.NETWORK_COLORS[net], edgecolor="black",
                   linewidth=0.3, label=net, zorder=3)
    ps.zero_line(ax)

    for inst_id, dx, dy in (("C_k9_3od_x0.8", 10, -6), ("J_k9_6od_x0.8", -9, 5)):
        hit = next((p for p in pts if p[0] == inst_id), None)
        if hit is None:
            continue
        ax.annotate(short_label(hit[0]), xy=(hit[1], hit[2]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=6,
                    ha="left" if dx > 0 else "right",
                    arrowprops=dict(arrowstyle="-", linewidth=0.4, color="0.3"))

    ax.set_xlabel("Generation-1 diversity loss vs. none")
    ax.set_ylabel("Final ATT improvement vs. none (%)")
    ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              title=None, handletextpad=0.2, columnspacing=0.7,
              borderaxespad=0.2)
    ax.annotate(f"Spearman $\\rho\\,=\\,{corr['both']:.3f}$",
                xy=(0.97, 0.94), xycoords="axes fraction", ha="right",
                va="top", fontsize=7)
    ps.save(fig, "fig06_diversity_vs_gain", outdir)


JK16_DIAGNOSTIC = (41.3, (335.82 - 334.55) / 335.82 * 100)


def fig07(outdir: Path, density_csv: Path) -> None:
    rows = read_csv(density_csv)
    pts = []
    for r in rows:
        d = fnum(r, "obs_per_edge")
        g = fnum(r, "gain_mutation")
        if d is not None and g is not None:
            pts.append((r["instance_id"], r["network"], d, g))

    low = [(i, n, g) for i, n, d, g in pts if d < 10]
    high = [(i, n, g) for i, n, d, g in pts if d >= 10]
    groups = [
        ("Low density\n($\\approx$3 obs./edge)\n"
         "network J only, %d instances" % len(low), 0, low),
        ("High density\n($\\approx$50 obs./edge)\n"
         "networks A–D, %d instances" % len(high), 1, high),
    ]

    fig, ax = ps.newfig(ps.COL_W, ps.COL_W * 0.66)
    ps.style_axes(ax)
    rng = np.random.default_rng(3)
    seen: set[str] = set()

    for label, x0, members in groups:
        vals = [g for _, _, g in members]
        m, s = st.mean(vals), st.stdev(vals)

        ax.add_patch(plt.Rectangle((x0 - 0.30, m - s), 0.60, 2 * s,
                                   facecolor="0.88", edgecolor="none", zorder=1))
        ax.plot([x0 - 0.30, x0 + 0.30], [m, m], color="0.15", linewidth=1.2,
                zorder=5)
        ax.annotate(f"SD {s:.2f}", xy=(x0 + 0.32, m + s), fontsize=6.5,
                    color="0.3", va="center")
        ax.annotate(f"mean {m:+.2f}\\%".replace("\\%", "%"), xy=(x0 + 0.32, m),
                    fontsize=6.5, va="center")

        for net in ps.NETWORK_ORDER:
            sel = [g for _, n, g in members if n == net]
            if not sel:
                continue
            ax.scatter(x0 + rng.uniform(-0.17, 0.17, len(sel)), sel, s=14,
                       color=ps.NETWORK_COLORS[net], edgecolor="black",
                       linewidth=0.3, zorder=4,
                       label=net if net not in seen else None)
            seen.add(net)

    means = [st.mean([g for _, _, g in m]) for _, _, m in groups]
    ax.plot([0, 1], means, linestyle=(0, (4, 3)), color="0.45", linewidth=0.8,
            zorder=3)
    ps.zero_line(ax)

    for inst, _n, _d, g in pts:
        if inst.startswith("J"):
            ax.annotate(inst.split("_")[-1], xy=(0, g), xytext=(-24, -2),
                        textcoords="offset points", fontsize=6.5, ha="left")

    ax.set_xticks([0, 1])
    ax.set_xticklabels([lab for lab, _, _ in groups], fontsize=6.5)
    ax.set_xlim(-0.62, 1.62)
    ax.set_ylabel("Guided-mutation ATT\nimprovement vs. none (%)")

    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(n) for n in ps.NETWORK_ORDER if n in labels]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              ncol=5, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              handletextpad=0.2, columnspacing=0.6, fontsize=6.5,
              borderaxespad=0.2)
    ps.save(fig, "fig07_density_vs_benefit", outdir)

    lo = [g for _, _, d, g in pts if d < 10]
    hi = [g for _, _, d, g in pts if d >= 10]
    print(f"  low density  n={len(lo)} mean={st.mean(lo):+.2f} "
          f"sd={st.stdev(lo):.2f} range=[{min(lo):+.2f}, {max(lo):+.2f}]")
    print(f"  high density n={len(hi)} mean={st.mean(hi):+.2f} "
          f"sd={st.stdev(hi):.2f} range=[{min(hi):+.2f}, {max(hi):+.2f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="merged_analysis_38_k0")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    data, outdir = Path(args.data), Path(args.outdir)
    ps.setup()
    print(f"data   : {data.resolve()}")
    print(f"outdir : {outdir.resolve()}")

    n_inst = len({r["instance_id"] for r in read_csv(data / "att_summary_by_mode_merged.csv")})
    print(f"instances in data: {n_inst}")
    if n_inst != 38:
        raise SystemExit(f"expected 38 instances, found {n_inst}. Wrong data folder?")

    fig02(data, outdir)
    fig03(data, outdir)
    jac, top3 = fig04(data, outdir)
    raw_gen95, raw_auc = fig05(data, outdir)
    fig06(data, outdir)
    fig07(outdir, Path(__file__).resolve().parent / "signal_density.csv")

    print("\n-- values quoted in the text --")
    print("  mean pairwise Jaccard :",
          {m: round(st.mean(v), 3) for m, v in jac.items()})
    print("  mean top-3 edge share :",
          {m: round(st.mean(v), 3) for m, v in top3.items()})
    print("  mean gen to 95%       :",
          {m: round(st.mean(v), 2) for m, v in raw_gen95.items()})
    print("  mean progress AUC     :",
          {m: round(st.mean(v), 3) for m, v in raw_auc.items()})


if __name__ == "__main__":
    main()
