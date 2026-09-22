from __future__ import annotations
import os
K0_MERGED = os.environ.get('K0_MERGED', './merged_analysis_38_k0')
K0_RUNSFX = os.environ.get('K0_RUNSFX', '_pm01k0')
K0_OUT = os.environ.get('K0_OUT', 'manuscript_numbers_k0')
K0_REEVAL = os.environ.get('K0_REEVAL', 'reeval_39_k0')


import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

ROOT = Path(".")
OUT = ROOT / K0_OUT
MODES = ["init", "mutation", "both"]
DUP = "A_k4_1od_x1.2"


def a12(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(((x[:, None] < y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y)))


def holm(ps):
    order = np.argsort(ps); m = len(ps); out = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, ps[i] * (m - rank)); out[i] = min(1.0, running)
    return out


def load_modes():
    d = {}
    for r in csv.DictReader(open(ROOT / f"reviewer_diagnostics/results/{K0_REEVAL}/reeval39_per_run.csv", encoding="utf-8")):
        d[(r["instance_id"], r["mode"], int(r["seed"]))] = (float(r["att_default"]), float(r["att_multi"]))
    fix = {}
    for r in csv.DictReader(open(ROOT / "runs_a12fix/a12fix_heldout_per_run.csv", encoding="utf-8")):
        if r["arm"] in ("none", "init", "mutation", "both"):
            fix[("A_k4_1od_x1.2fix", r["arm"], int(r["seed"]))] = (float(r["insample"]), float(r["heldout_mean"]))
    return d, fix


def variant(d, fix, name):
    if name == "n39":
        return d
    base = {k: v for k, v in d.items() if k[0] != DUP}
    if name == "n38fix":
        base.update(fix)
    return base


def mode_tables(data, tag, md):
    insts = sorted({k[0] for k in data})
    seeds = sorted({k[2] for k in data})
    rows = []
    per_inst = []
    net_acc = {w: defaultdict(lambda: defaultdict(list)) for w in ("default", "heldout")}
    dem_acc = {w: defaultdict(lambda: defaultdict(list)) for w in ("default", "heldout")}
    best_mode = defaultdict(int)
    for w_i, w in enumerate(("default", "heldout")):
        stats = {}
        for m in MODES:
            deltas, pcts, a12s, curses = [], [], [], []
            for inst in insts:
                none = np.array([data[(inst, "none", s)][w_i] for s in seeds if (inst, "none", s) in data])
                mode = np.array([data[(inst, m, s)][w_i] for s in seeds if (inst, m, s) in data])
                ok_n, ok_m = none[none < 1e11], mode[mode < 1e11]
                delta = ok_m.mean() - ok_n.mean()
                pct = 100 * (ok_n.mean() - ok_m.mean()) / ok_n.mean()
                deltas.append(delta); pcts.append(pct); a12s.append(a12(ok_m, ok_n))
                net = inst[0]; dem = inst.split("_x")[1].replace("fix", "")
                net_acc[w][m][net].append(pct); dem_acc[w][m][dem].append(pct)
                if w == "heldout":
                    d0 = np.array([data[(inst, m, s)][0] for s in seeds if (inst, m, s) in data])
                    curses.append((ok_m.mean() - d0[d0 < 1e11].mean()))
                    per_inst.append({"instance_id": inst, "mode": m, "pct_default": None, "pct_heldout": pct,
                                     "a12_heldout": a12s[-1], "delta_heldout_s": delta})
            deltas, pcts, a12s = np.array(deltas), np.array(pcts), np.array(a12s)
            p = wilcoxon(deltas).pvalue
            stats[m] = dict(mean_delta=deltas.mean(), median_pct=float(np.median(pcts)), mean_pct=pcts.mean(),
                            p=p, mean_a12=a12s.mean(), a12_gt=int((a12s > 0.5).sum()),
                            improved=int((deltas < 0).sum()), n=len(insts),
                            curse=float(np.mean(curses)) if curses else float("nan"), pcts=pcts)
        hp = holm([stats[m]["p"] for m in MODES])
        for i, m in enumerate(MODES):
            s = stats[m]
            rows.append({"variant": tag, "evaluation": w, "mode": m, "n": s["n"], "mean_delta_att_s": round(s["mean_delta"], 3),
                         "median_improvement_pct": round(s["median_pct"], 3), "mean_improvement_pct": round(s["mean_pct"], 3),
                         "wilcoxon_p": s["p"], "holm_p": hp[i], "mean_a12": round(s["mean_a12"], 3),
                         "a12_gt_0.5": s["a12_gt"], "improved_instances": s["improved"],
                         "winners_curse_s": round(s["curse"], 2) if w == "heldout" else ""})
        if w == "default":
            for rec in per_inst:
                pass
    for rec in per_inst:
        inst, m = rec["instance_id"], rec["mode"]
        none = np.array([data[(inst, "none", s)][0] for s in seeds if (inst, "none", s) in data])
        mode = np.array([data[(inst, m, s)][0] for s in seeds if (inst, m, s) in data])
        rec["pct_default"] = 100 * (none.mean() - mode.mean()) / none.mean()
    for inst in insts:
        means = {m: np.mean([data[(inst, m, s)][1] for s in seeds if (inst, m, s) in data]) for m in MODES}
        best_mode[min(means, key=means.get)] += 1

    with open(OUT / f"modes_stats_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(OUT / f"per_instance_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_inst[0].keys())); w.writeheader(); w.writerows(per_inst)
    nets = sorted({k[0][0] for k in data}); dems = sorted({k[0].split("_x")[1].replace("fix", "") for k in data})
    with open(OUT / f"network_means_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["evaluation", "mode"] + nets + ["all"])
        for wv in ("default", "heldout"):
            for m in MODES:
                w.writerow([wv, m] + [round(np.mean(net_acc[wv][m][n]), 3) for n in nets] + [round(np.mean(sum((net_acc[wv][m][n] for n in nets), [])), 3)])
    with open(OUT / f"demand_means_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["evaluation", "mode"] + dems)
        for wv in ("default", "heldout"):
            for m in MODES:
                w.writerow([wv, m] + [round(np.mean(dem_acc[wv][m][d]), 3) for d in dems])

    md.append(f"\n## Guidance modes vs none — variant `{tag}` (n={len(insts)})\n")
    md.append("| eval | mode | mean ΔATT (s) | median % | mean % | Wilcoxon p | Holm p | mean Â12 | Â12>0.5 | improved | curse (s) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['evaluation']} | {r['mode']} | {r['mean_delta_att_s']:+.2f} | {r['median_improvement_pct']:.3f} | {r['mean_improvement_pct']:.3f} | "
                  f"{r['wilcoxon_p']:.2e} | {r['holm_p']:.4f} | {r['mean_a12']:.3f} | {r['a12_gt_0.5']}/{r['n']} | {r['improved_instances']}/{r['n']} | {r['winners_curse_s']} |")
    md.append(f"\nBest guided mode per instance (held-out seed-mean): " + ", ".join(f"{m} {best_mode[m]}" for m in MODES))
    md.append("\nPer-network mean improvement % (held-out / default):\n")
    md.append("| mode | " + " | ".join(nets) + " |"); md.append("|---" * (len(nets) + 1) + "|")
    for m in MODES:
        md.append(f"| {m} | " + " | ".join(f"{np.mean(net_acc['heldout'][m][n]):+.2f} / {np.mean(net_acc['default'][m][n]):+.2f}" for n in nets) + " |")
    md.append("\nPer-demand-scale mean improvement % (held-out / default):\n")
    md.append("| mode | " + " | ".join(dems) + " |"); md.append("|---" * (len(dems) + 1) + "|")
    for m in MODES:
        md.append(f"| {m} | " + " | ".join(f"{np.mean(dem_acc['heldout'][m][d]):+.2f} / {np.mean(dem_acc['default'][m][d]):+.2f}" for d in dems) + " |")
    md.append("\nDefault-seed vs held-out agreement of the per-instance Δ (both/mutation/init): " + ", ".join(
        f"{m}: sign {sum((a['pct_default']>0)==(a['pct_heldout']>0) for a in per_inst if a['mode']==m)}/{len(insts)}, "
        f"ρ={spearmanr([a['pct_default'] for a in per_inst if a['mode']==m],[a['pct_heldout'] for a in per_inst if a['mode']==m]).statistic:+.3f}"
        for m in MODES))
    return per_inst


def named_cases(data, md):
    seeds = sorted({k[2] for k in data})
    md.append("\n## Named cases (held-out / default, % vs none)\n")
    for inst in ["J_k9_6od_x0.8", "J_k9_6od_x1.0", "J_k9_6od_x1.2", "C_k9_3od_x0.8", "A_k4_1od_x1.2fix", "A_k4_1od_x1.2"]:
        if not any(k[0] == inst for k in data):
            continue
        line = f"- **{inst}**: "
        parts = []
        for m in MODES:
            for w_i, w in enumerate(("heldout", "default")):
                pass
            n0 = np.mean([data[(inst, "none", s)][0] for s in seeds if (inst, "none", s) in data]); n1 = np.mean([data[(inst, "none", s)][1] for s in seeds if (inst, "none", s) in data])
            m0 = np.mean([data[(inst, m, s)][0] for s in seeds if (inst, m, s) in data]); m1 = np.mean([data[(inst, m, s)][1] for s in seeds if (inst, m, s) in data])
            parts.append(f"{m} {100*(n1-m1)/n1:+.2f} / {100*(n0-m0)/n0:+.2f}")
        md.append(line + "; ".join(parts) + f"  (none held-out {n1:.2f}, default {n0:.2f})")


def seed_context(md):
    acc = defaultdict(list)
    for r in csv.DictReader(open(ROOT / f"reviewer_diagnostics/results/{K0_REEVAL}/reeval39_long.csv", encoding="utf-8")):
        if float(r["att"]) < 1e11:
            acc[(r["instance_id"], r["mode"], r["seed"])].append(float(r["att"]))
    sds = [np.std(v, ddof=1) for v in acc.values() if len(v) >= 3]
    pen = sum(1 for r in csv.DictReader(open(ROOT / f"reviewer_diagnostics/results/{K0_REEVAL}/reeval39_long.csv", encoding="utf-8")) if float(r["att"]) >= 1e11)
    md.append(f"\n## Seed-noise context\n\n- Held-out SD of a single solution across SUMO seeds 201–205: median {np.median(sds):.2f} s, mean {np.mean(sds):.2f} s ({len(sds)} solutions)")
    md.append(f"- Penalised held-out evaluations: {pen} of {sum(len(v) for v in acc.values()) + pen}")


def comparators(md):
    rows = []
    md.append("\n## Comparator study (3 instances, held-out SUMO seeds 101–110; Â12 = P(both < arm), Holm over the 6 comparisons within an instance)\n")
    for inst in ["A_k4_3od_x1.0", "B_k9_3od_x1.0", "J_k9_6od_x1.0"]:
        p = ROOT / f"reviewer_diagnostics/results/reeval_{inst}_v2/reeval_long.csv"
        if not p.exists():
            p = ROOT / f"reviewer_diagnostics/results/reeval_{inst}/reeval_long.csv"
        acc = defaultdict(list); dflt = {}
        for r in csv.DictReader(open(p, encoding="utf-8")):
            key = (r["arm"], int(r["run_seed"]))
            if float(r["att"]) < 1e11:
                acc[key].append(float(r["att"]))
            if r["att_default"] not in ("", "None"):
                dflt[key] = float(r["att_default"])
        arms = defaultdict(list); arms_def = defaultdict(list); noclosure = None
        for (arm, s), vals in sorted(acc.items()):
            if arm == "no_closure":
                noclosure = float(np.mean(vals)); continue
            arms[arm].append(float(np.mean(vals))); arms_def[arm].append(dflt.get((arm, s), float("nan")))
        both = np.array(arms["both"])
        order = ["none", "init", "mutation", "both", "surrogate_repaired", "surrogate_submitted", "heuristic", "sa"]
        order = [a for a in order if a in arms]
        ps = {a: mannwhitneyu(both, np.array(arms[a])).pvalue for a in order if a != "both"}
        hp = dict(zip(ps, holm(list(ps.values()))))
        md.append(f"\n**{inst}** — no-closure held-out {noclosure:.2f} s\n")
        md.append("| arm | held-out mean | best | worst | SD | default mean | curse | vs no-closure % | Â12(both<arm) | MW p | Holm p |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for a in order:
            v = np.array(arms[a]); vd = np.array(arms_def[a])
            a12v = a12(both, v) if a != "both" else float("nan")
            rows.append({"instance": inst, "arm": a, "heldout_mean": v.mean(), "best": v.min(), "worst": v.max(), "sd": v.std(ddof=1),
                         "default_mean": vd.mean(), "curse": (v - vd).mean(), "reduction_vs_noclosure_pct": 100 * (noclosure - v.mean()) / noclosure,
                         "a12_both_vs_arm": a12v, "mw_p": ps.get(a, float("nan")), "holm_p": hp.get(a, float("nan")), "noclosure_heldout": noclosure})
            md.append(f"| {a} | {v.mean():.2f} | {v.min():.2f} | {v.max():.2f} | {v.std(ddof=1):.2f} | {vd.mean():.2f} | {(v-vd).mean():+.2f} | "
                      f"{100*(noclosure-v.mean())/noclosure:+.2f} | {a12v:.3f} | {ps.get(a, float('nan')):.4f} | {hp.get(a, float('nan')):.4f} |")
    with open(OUT / "comparators_heldout.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def constants(md):
    md.append("\n## Numeric constants to correct\n")
    md.append("| item | published | correct |\n|---|---|---|")
    for a, b, c in [
        ("|E_cand| network J", "1,414", "1,415"),
        ("m = round(p_m n) on J", "141", "142 (m_close 141, m_open 1)"),
        ("repair on J at the budget boundary", "opens 139 of 148", "opens 140 of 149"),
        ("parent retention on J", "≈6%", "5.4%"),
        ("signal density J", "3.2", "3.22"),
        ("never-closed candidate edges on J", "58 (4.1%)", "59 (4.2%)"),
        ("never-closed within visited range", "276", "277"),
        ("effective density J", "1.6", "1.66"),
        ("9,000 / |E_cand|", "6.4", "6.36"),
        ("low/high density SD ratio", "'several times' / 'roughly five'", "4.87, 95% CI [2.4, 30.6]"),
        ("distinct instances", "39", "38 (A_k4_1od_x1.2 duplicates x1.0)"),
        ("Appendix D probe demand", "'demand untouched'", "1,350-veh v3 demand (270/135/270/270/270/135) vs 1,500 in the 39-instance J; re-run on plain demand = runs_J_k16_plain"),
        ("161-edge candidate criterion", "not stated", "union of top-3 length-weighted shortest paths per OD (181) − 12 protected = 169; 8 single-cut edges removed at run time → 161"),
        ("'results are deterministic given a seed' (§Implementation)", "stated for everything", "holds for the GA arms; the surrogate comparator is NOT exactly reproducible: re-running surrogate_arm_v3 on J with the same model and seeds gave per-seed bests 594.3/608.4/635.1/601.7/629.8/602.1/627.5/605.3/531.9/427.4 (mean 586.34) against the published 564.11 (1/10 seeds identical). Cause: non-deterministic GNN scatter reductions on near-tied predictions. Conclusion (surrogate fails on J) unchanged; qualify the statement"),
    ]:
        md.append(f"| {a} | {b} | {c} |")


def main():
    OUT.mkdir(exist_ok=True)
    d, fix = load_modes()
    md = ["# Replacement numbers for the revised manuscript (generated by abl/manuscript_tables.py)\n",
          "Held-out = final solutions re-simulated at SUMO seeds 201–205 (modes) or 101–110 (comparators); default = the seed the search optimised."]
    for tag in ("n39", "n38", "n38fix"):
        data = variant(d, fix, tag)
        mode_tables(data, tag, md)
    named_cases(variant(d, fix, "n38fix"), md)
    named_cases({k: v for k, v in d.items() if k[0] == DUP}, md)
    seed_context(md)
    comparators(md)
    constants(md)
    (OUT / "TABLES.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
