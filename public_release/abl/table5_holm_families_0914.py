import re, xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(r".")
PEN = 1e9
INSTS = ["A_k4_3od_x1.0", "B_k9_3od_x1.0", "J_k9_6od_x1.0"]
ARMS7 = ["none", "init", "mutation", "surrogate_repaired", "heuristic", "sa", "surrogate_submitted"]


def a12(x, y):
    w = sum(1.0 if xi < yi else 0.5 if xi == yi else 0.0 for xi in x for yi in y)
    return w / (len(x) * len(y))


def holm(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i]); adj = [0.0] * m; run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


def load(inst):
    vals = {}
    if inst == "A_k4_3od_x1.0":
        sb = pd.read_csv(ROOT / "runs_A_k4_3od_x1.0_pm01k0" / "analysis" / "seed_best.csv")
    else:
        m = pd.read_csv(ROOT / "merged_analysis_38_k0" / "seed_best_merged.csv"); sb = m[m.instance_id == inst]
    for mode, d in sb.groupby("mode"):
        vals[mode] = d[d.ATT < PEN].ATT.tolist()
    for arm, tag in [("surrogate_submitted", "surrogate"), ("heuristic", "heuristic"), ("sa", "sa")]:
        v = []
        for p in sorted((ROOT / "runs_reviewer" / inst).glob(f"{tag}_seed*")):
            f = p / "best.txt"
            if f.exists():
                mt = re.search(r"best_score=([0-9.]+)", f.read_text())
                if mt and float(mt.group(1)) < PEN:
                    v.append(float(mt.group(1)))
        vals[arm] = v
    rev = inst[0]; res = ROOT / "reviewer_diagnostics" / "results"
    if rev == "J":
        av = pd.read_csv(res / "arm_J_sum.csv"); vals["surrogate_repaired"] = av.groupby("seed")["att"].min().tolist()
    else:
        rundir = res / f"arm_{rev}_sum_run"; demand = {"A": 1000, "B": 2000}[rev]; per_seed = {}
        for cdir in sorted(rundir.glob("s*_r*")):
            tri = cdir / "tripinfo.xml"
            if not tri.exists():
                continue
            durs = [float(ev.attrib["duration"]) for ev in ET.parse(tri).getroot().iter("tripinfo")
                    if float(ev.attrib.get("duration", -1)) >= 0]
            if len(durs) < demand:
                continue
            s = int(cdir.name.split("_")[0][1:]); att = sum(durs) / len(durs)
            per_seed[s] = min(per_seed.get(s, 1e18), att)
        vals["surrogate_repaired"] = [per_seed[k] for k in sorted(per_seed)]
    return vals


rows = []
for inst in INSTS:
    v = load(inst); ref = v["both"]
    for arm in ARMS7:
        x = v[arm]
        p = float(mannwhitneyu(ref, x, alternative="two-sided").pvalue)
        rows.append(dict(inst=inst, arm=arm, n=len(x), mean=sum(x) / len(x), A12=a12(ref, x), p=p))
df = pd.DataFrame(rows)
df["F6"] = float("nan"); df["F7"] = float("nan")
for inst, d in df.groupby("inst"):
    d6 = d[d.arm != "surrogate_submitted"]
    df.loc[d6.index, "F6"] = holm(d6.p.tolist())
    df.loc[d.index, "F7"] = holm(d.p.tolist())
d18 = df[df.arm != "surrogate_submitted"]
df["G18"] = float("nan"); df.loc[d18.index, "G18"] = holm(d18.p.tolist())
df["G21"] = holm(df.p.tolist())
pd.set_option("display.width", 220)
print(df.to_string(float_format=lambda x: f"{x:.5f}"))
df.to_csv(Path("./manuscript_numbers_k0").joinpath("holm_families.csv"), index=False)
