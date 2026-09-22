from __future__ import annotations
import os
K0_MERGED = os.environ.get('K0_MERGED', './merged_analysis_38_k0')
K0_RUNSFX = os.environ.get('K0_RUNSFX', '_pm01k0')
K0_OUT = os.environ.get('K0_OUT', 'manuscript_numbers_k0')
K0_REEVAL = os.environ.get('K0_REEVAL', 'reeval_39_k0')


import re
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(r".")
PEN = 1e9
INSTS = ["A_k4_3od_x1.0", "B_k9_3od_x1.0", "J_k9_6od_x1.0"]
ARMS = ["none", "init", "mutation", "surrogate_submitted",
        "surrogate_repaired", "heuristic", "sa"]
LABEL = {"none": "none (GA only)", "init": "init", "mutation": "mutation",
         "surrogate_submitted": "surrogate (as submitted)",
         "surrogate_repaired": "surrogate (repaired)",
         "heuristic": "structural heuristic", "sa": "simulated annealing"}
REPAIRED = {"A_k4_3od_x1.0": "arm_A_sum_run", "B_k9_3od_x1.0": "arm_B_sum_run"}


def a12(x, y):
    w = sum(1.0 if xi < yi else 0.5 if xi == yi else 0.0 for xi in x for yi in y)
    return w / (len(x) * len(y))


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def load(inst):
    vals = {}
    if inst == "A_k4_3od_x1.0":
        sb = pd.read_csv(ROOT / ("runs_A_k4_3od_x1.0" + K0_RUNSFX) / "analysis" /
                         "seed_best.csv")
    else:
        m = pd.read_csv(ROOT / "merged_analysis_38_k0" / "seed_best_merged.csv")
        sb = m[m.instance_id == inst]
    for mode, d in sb.groupby("mode"):
        vals[mode] = d[d.ATT < PEN].ATT.tolist()
    for arm, tag in [("surrogate_submitted", "surrogate"),
                     ("heuristic", "heuristic"), ("sa", "sa")]:
        v = []
        for p in sorted((ROOT / "runs_reviewer" / inst).glob(f"{tag}_seed*")):
            f = p / "best.txt"
            if f.exists():
                mt = re.search(r"best_score=([0-9.]+)", f.read_text())
                if mt and float(mt.group(1)) < PEN:
                    v.append(float(mt.group(1)))
        if v:
            vals[arm] = v
    rev = inst[0]
    res = ROOT / "reviewer_diagnostics" / "results"
    if rev == "J":
        av = pd.read_csv(res / "arm_J_sum.csv")
        vals["surrogate_repaired"] = av.groupby("seed")["att"].min().tolist()
    else:
        import xml.etree.ElementTree as ET
        rundir = res / f"arm_{rev}_sum_run"
        demand = {"A": 1000, "B": 2000}[rev]
        per_seed = {}
        for cdir in sorted(rundir.glob("s*_r*")):
            tri = cdir / "tripinfo.xml"
            if not tri.exists():
                continue
            durs = [float(ev.attrib["duration"])
                    for ev in ET.parse(tri).getroot().iter("tripinfo")
                    if float(ev.attrib.get("duration", -1)) >= 0]
            if len(durs) < demand:
                continue
            s = int(cdir.name.split("_")[0][1:])
            att = sum(durs) / len(durs)
            per_seed[s] = min(per_seed.get(s, 1e18), att)
        if per_seed:
            vals["surrogate_repaired"] = [per_seed[k] for k in sorted(per_seed)]
    return vals


rows = []
for inst in INSTS:
    vals = load(inst)
    ref = vals["both"]
    for arm in ARMS:
        v = vals.get(arm)
        if not v:
            continue
        p = float(mannwhitneyu(ref, v, alternative="two-sided").pvalue)
        rows.append(dict(instance=inst, arm=arm, n=len(v), mean=sum(v) / len(v),
                         A12=a12(ref, v), p_raw=p))

df = pd.DataFrame(rows)
df["holm_within"] = 0.0
for inst, d in df.groupby("instance"):
    df.loc[d.index, "holm_within"] = holm(d.p_raw.tolist())
df["holm_all18"] = holm(df.p_raw.tolist())

pd.set_option("display.width", 200)
print(f"total tests performed: {len(df)}")
print()
for inst, d in df.groupby("instance", sort=False):
    print(f"=== {inst}")
    print(f"  {'arm':26s} {'mean':>8s} {'A12':>6s} {'p_raw':>8s} "
          f"{'Holm/6':>8s} {'Holm/18':>8s}  changes?")
    for r in d.itertuples():
        chg = ""
        if (r.holm_within < 0.05) != (r.holm_all18 < 0.05):
            chg = "  <-- significance changes"
        print(f"  {LABEL[r.arm]:26s} {r.mean:8.2f} {r.A12:6.3f} {r.p_raw:8.4f} "
              f"{r.holm_within:8.4f} {r.holm_all18:8.4f}{chg}")
    print()

n6 = int((df.holm_within < 0.05).sum())
n18 = int((df.holm_all18 < 0.05).sum())
print(f"significant at 0.05 -- Holm over 6 within each instance: {n6}")
print(f"significant at 0.05 -- Holm over all 18 tests:           {n18}")
df.to_csv(ROOT / "reviewer_diagnostics" / "results" / "table5_holm18.csv",
          index=False)
