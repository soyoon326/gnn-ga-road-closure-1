import re
from pathlib import Path
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(".")


def best(inst, tag):
    v = []
    for s in range(10):
        f = ROOT / "runs_reviewer" / inst / f"{tag}_seed{s}" / "best.txt"
        if f.exists():
            v.append(float(re.search(r"best_score=([0-9.]+)", f.read_text()).group(1)))
    return v


def modes(inst):
    if inst.startswith("A_k4_3od"):
        sb = pd.read_csv(ROOT / "runs_A_k4_3od_x1.0_pm01k0" / "analysis" / "seed_best.csv")
    else:
        sb = pd.read_csv(ROOT / "merged_analysis_38_k0" / "seed_best_merged.csv"); sb = sb[sb.instance_id == inst]
    return {m: d.ATT.tolist() for m, d in sb.groupby("mode")}


def a12(x, y):
    return sum(1.0 if a < b else 0.5 if a == b else 0.0 for a in x for b in y) / (len(x) * len(y))


for inst in ["A_k4_3od_x1.0", "B_k9_3od_x1.0", "J_k9_6od_x1.0"]:
    rev = best(inst, "heuristic_rev")
    if not rev:
        continue
    fwd = best(inst, "heuristic"); m = modes(inst)
    print(f"=== {inst}: reversed n={len(rev)} mean {sum(rev)/len(rev):.2f} range {min(rev):.2f}-{max(rev):.2f}")
    print(f"    forward mean {sum(fwd)/len(fwd):.2f} range {min(fwd):.2f}-{max(fwd):.2f}")
    for name, ref in [("forward heuristic", fwd), ("none", m["none"]), ("both", m["both"])]:
        p = mannwhitneyu(rev, ref, alternative="two-sided").pvalue
        print(f"    reversed vs {name:18s}: mean diff {sum(rev)/len(rev) - sum(ref)/len(ref):+7.2f} s, "
              f"A12(reversed better) {a12(rev, ref):.3f}, MW p {p:.4g}")
