import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(".")
RES = ROOT / "reviewer_diagnostics" / "results"
OUTD = ROOT / "manuscript_numbers_k0"
PEN = 1e9
INSTS = ["A_k4_3od_x1.0", "B_k9_3od_x1.0", "J_k9_6od_x1.0"]
ARMS7 = ["none", "init", "mutation", "surrogate_repaired", "heuristic", "sa", "surrogate_submitted"]
KMAX = {"A_k4_3od_x1.0": 4, "B_k9_3od_x1.0": 9, "J_k9_6od_x1.0": 9}


def a12(x, y):
    return sum(1.0 if a < b else 0.5 if a == b else 0.0 for a in x for b in y) / (len(x) * len(y))


def holm(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i]); adj = [0.0] * m; run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def best_txt(f):
    m = re.search(r"best_score=([0-9.eE+]+)", f.read_text(errors="ignore"))
    return float(m.group(1)) if m else None


def arm_root(inst):
    return ROOT / ("runs_reviewer_A38" if inst.startswith("A_") else "runs_reviewer") / inst


def load(inst):
    vals = {}
    if inst.startswith("A_"):
        sb = pd.read_csv(ROOT / "runs_A_k4_3od_x1.0_pm01k0" / "analysis" / "seed_best.csv")
    else:
        m = pd.read_csv(ROOT / "merged_analysis_38_k0" / "seed_best_merged.csv"); sb = m[m.instance_id == inst]
    for mode, d in sb.groupby("mode"):
        d = d.sort_values("seed"); vals[mode] = d[d.ATT < PEN].ATT.tolist()
    for arm, tag in [("surrogate_submitted", "surrogate"), ("heuristic", "heuristic"), ("sa", "sa"),
                     ("heuristic_rev", "heuristic_rev")]:
        v = []
        for p in sorted(arm_root(inst).glob(f"{tag}_seed*"), key=lambda q: int(q.name.rsplit("seed", 1)[1])):
            if (p / "best.txt").exists():
                x = best_txt(p / "best.txt")
                if x is not None and x < PEN:
                    v.append(x)
        vals[arm] = v
    if inst.startswith("J_"):
        av = pd.read_csv(RES / "arm_J_sum.csv"); vals["surrogate_repaired"] = av.groupby("seed")["att"].min().tolist()
    else:
        rundir = RES / ("arm_A_sum_run38" if inst.startswith("A_") else "arm_B_sum_run")
        demand = 1000 if inst.startswith("A_") else 2000
        per = {}
        for cdir in sorted(rundir.glob("s*_r*")):
            tri = cdir / "tripinfo.xml"
            if not tri.exists():
                continue
            durs = [float(ev.attrib["duration"]) for ev in ET.parse(tri).getroot().iter("tripinfo")
                    if float(ev.attrib.get("duration", -1)) >= 0]
            if len(durs) < demand:
                continue
            s = int(cdir.name.split("_")[0][1:]); per[s] = min(per.get(s, 1e18), sum(durs) / len(durs))
        vals["surrogate_repaired"] = [per[k] for k in sorted(per)]
    return vals


pd.set_option("display.width", 240)
V = {inst: load(inst) for inst in INSTS}

print("=== Table 4 (mean / best / worst / sd / range, n)")
t4 = []
for inst in INSTS:
    for arm in ARMS7 + ["both", "heuristic_rev"]:
        x = V[inst].get(arm, [])
        if x:
            t4.append(dict(inst=inst, arm=arm, n=len(x), mean=np.mean(x), best=min(x), worst=max(x),
                           sd=np.std(x, ddof=1), range=max(x) - min(x)))
t4 = pd.DataFrame(t4); print(t4.to_string(float_format=lambda v: f"{v:.2f}"))
t4.to_csv(OUTD / "cmp_table4_A38.csv", index=False)

print("\n=== A: seeds reaching the best value found on A by any arm")
A = V["A_k4_3od_x1.0"]
bestA = min(min(v) for k, v in A.items() if v and k != "heuristic_rev")
for arm, v in A.items():
    print(f"  {arm:20s} best {min(v):.2f}  seeds at {bestA:.2f}: {sum(abs(x - bestA) < 0.005 for x in v)}  per-seed {[round(x, 2) for x in v]}")

print("\n=== default-seed tests vs both (F7 within instance, G21 over all)")
rows = []
for inst in INSTS:
    ref = V[inst]["both"]
    for arm in ARMS7:
        x = V[inst][arm]
        rows.append(dict(inst=inst, arm=arm, n=len(x), mean=np.mean(x), A12=a12(ref, x),
                         p=float(mannwhitneyu(ref, x, alternative="two-sided").pvalue)))
df = pd.DataFrame(rows); df["F7"] = np.nan
for inst, d in df.groupby("inst"):
    df.loc[d.index, "F7"] = holm(d.p.tolist())
df["G21"] = holm(df.p.tolist())
print(df.to_string(float_format=lambda v: f"{v:.5f}"))
df.to_csv(OUTD / "holm_families_A38.csv", index=False)

print("\n=== surrogate arms and heuristic_rev against none / both / heuristic (raw MW p)")
for inst in INSTS:
    for arm, refs in [("surrogate_repaired", ["none", "both"]), ("surrogate_submitted", ["none"]),
                      ("heuristic_rev", ["heuristic", "none", "both"])]:
        x = V[inst].get(arm, [])
        if not x:
            continue
        for r in refs:
            y = V[inst][r]
            print(f"  {inst} {arm:20s} vs {r:9s}: diff {np.mean(x) - np.mean(y):+8.2f} s  A12(arm better) {a12(x, y):.3f}  "
                  f"p {mannwhitneyu(x, y, alternative='two-sided').pvalue:.4g}  min(arm)-max(ref) {min(x) - max(y):+.2f}")

print("\n=== verified-candidate cardinality")
for inst in INSTS:
    km = KMAX[inst]
    if inst.startswith("A_"):
        rr = pd.read_csv(RES / "arm_A_sum_run38" / "arm_results.csv")
    elif inst.startswith("B_"):
        rr = pd.read_csv(RES / "arm_B_sum_run" / "arm_results.csv")
    else:
        rr = pd.read_csv(RES / "arm_J_sum.csv")
    fs = sorted(arm_root(inst).glob("surrogate_seed*/verify.csv"))
    vk = pd.concat([pd.read_csv(f) for f in fs]) if fs else pd.DataFrame({"k": []})
    print(f"  {inst}: reported arm share k=kmax {np.mean(rr.k == km):.2f} (n={len(rr)}); "
          f"Appendix A arm share k=kmax {np.mean(vk.k == km) if len(vk) else float('nan'):.2f} (n={len(vk)}) {vk.k.value_counts().sort_index().to_dict() if len(vk) else ''}")

HA = RES / "reeval_cmp_201_205_A38"
if (HA / "long.csv").exists():
    def runs_from(rdir, keep):
        sol = pd.read_csv(rdir / "solutions.csv"); lg = pd.read_csv(rdir / "long.csv")
        sol = sol[sol.instance_id.isin(keep)]; lg = lg[lg.instance_id.isin(keep)]
        ok = lg[lg.att < 1e11]
        g = ok.groupby(["instance_id", "sol_id"]).att.agg(["mean", "size"]).rename(
            columns={"mean": "att_multi", "size": "n_ok"}).reset_index()
        pen = lg[lg.att >= 1e11].groupby(["instance_id", "sol_id"]).size().rename("n_pen").reset_index()
        return sol.merge(g, on=["instance_id", "sol_id"], how="left").merge(
            pen, on=["instance_id", "sol_id"], how="left").fillna({"n_pen": 0})
    runs = pd.concat([runs_from(HA, ["A_k4_3od_x1.0"]),
                      runs_from(RES / "reeval_cmp_201_205", ["B_k9_3od_x1.0", "J_k9_6od_x1.0"])], ignore_index=True)
    modes = pd.read_csv(OUTD / "heldout_modes3.csv").rename(columns={"mode": "arm"})
    modes["n_pen"] = 5 - modes["n_ok"]
    cols = ["instance_id", "arm", "seed", "att_multi", "att_default", "n_pen"]
    allr = pd.concat([modes[cols], runs[runs.arm != "no_closure"][cols]], ignore_index=True)
    allr.to_csv(OUTD / "cmp_heldout_per_run_A38.csv", index=False)
    hrows = []
    for inst, d in allr.groupby("instance_id"):
        nc = runs[(runs.instance_id == inst) & (runs.arm == "no_closure")].att_multi.iloc[0]
        both = d[d.arm == "both"].att_multi.to_numpy()
        print(f"\n=== held-out {inst}  no-closure {nc:.2f}")
        for arm in ["both"] + ARMS7:
            x = d[d.arm == arm]
            print(f"  {arm:20s} def {x.att_default.mean():8.2f} ho {x.att_multi.mean():8.2f} best {x.att_multi.min():8.2f} "
                  f"worst {x.att_multi.max():8.2f} curse {x.att_multi.mean() - x.att_default.mean():+7.2f} pen {int(x.n_pen.sum())}")
        for arm in ARMS7:
            x = d[d.arm == arm].att_multi.to_numpy()
            hrows.append(dict(inst=inst, arm=arm, mean=x.mean(), A12=a12(both, x),
                              p=float(mannwhitneyu(both, x, alternative="two-sided").pvalue),
                              both_worst=both.max(), arm_best=x.min(), diff_vs_both=x.mean() - both.mean()))
    h = pd.DataFrame(hrows); h["F7"] = np.nan
    for inst, d in h.groupby("inst"):
        h.loc[d.index, "F7"] = holm(d.p.tolist())
    h["G21"] = holm(h.p.tolist())
    print(); print(h.to_string(float_format=lambda v: f"{v:.4f}"))
    h.to_csv(OUTD / "cmp_heldout_stats_A38.csv", index=False)
    for inst in INSTS:
        d = allr[allr.instance_id == inst]
        s, n = d[d.arm == "surrogate_repaired"].att_multi, d[d.arm == "none"].att_multi
        print(f"  held-out {inst}: surrogate_repaired vs none diff {s.mean() - n.mean():+.2f}, A12(none better) {a12(n.tolist(), s.tolist()):.3f}, "
              f"p {mannwhitneyu(s, n, alternative='two-sided').pvalue:.4g}, min(sur)-max(none) {s.min() - n.max():+.2f}")
else:
    print("\n(held-out A38 not yet available)")
