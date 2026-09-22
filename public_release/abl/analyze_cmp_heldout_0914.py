from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SCR = Path("./manuscript_numbers_k0")
R = Path("./reviewer_diagnostics/results/reeval_cmp_201_205")
PEN = 1e11
ARMS = ["none", "init", "mutation", "surrogate_repaired", "heuristic", "sa", "surrogate_submitted"]


def a12(x, y):
    return sum(1.0 if a < b else 0.5 if a == b else 0.0 for a in x for b in y) / (len(x) * len(y))


def holm(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i]); adj = [0.0] * m; run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


sol = pd.read_csv(R / "solutions.csv")
lg = pd.read_csv(R / "long.csv")
ok = lg[lg.att < PEN]
g = ok.groupby(["instance_id", "sol_id"]).att.agg(["mean", "size"]).rename(columns={"mean": "att_multi", "size": "n_ok"}).reset_index()
pen = lg[lg.att >= PEN].groupby(["instance_id", "sol_id"]).size().rename("n_pen").reset_index()
runs = sol.merge(g, on=["instance_id", "sol_id"], how="left").merge(pen, on=["instance_id", "sol_id"], how="left").fillna({"n_pen": 0})
modes = pd.read_csv(SCR / "heldout_modes3.csv").rename(columns={"mode": "arm"})
modes["n_pen"] = 5 - modes["n_ok"]
cols = ["instance_id", "arm", "seed", "att_multi", "att_default", "n_pen"]
allr = pd.concat([modes[cols], runs[runs.arm != "no_closure"][cols]], ignore_index=True)
allr.to_csv(SCR / "cmp_heldout_per_run.csv", index=False)

rows = []
for inst, d in allr.groupby("instance_id"):
    nc = runs[(runs.instance_id == inst) & (runs.arm == "no_closure")].att_multi.iloc[0]
    both = d[d.arm == "both"].att_multi.to_numpy()
    print(f"\n=== {inst}   no-closure held-out {nc:.2f}")
    print(f"  {'arm':20s} {'def.mean':>8s} {'ho.mean':>8s} {'ho.best':>8s} {'ho.worst':>8s} {'curse':>7s} {'pen':>4s}")
    for arm in ["both"] + ARMS:
        x = d[d.arm == arm]
        print(f"  {arm:20s} {x.att_default.mean():8.2f} {x.att_multi.mean():8.2f} {x.att_multi.min():8.2f} "
              f"{x.att_multi.max():8.2f} {x.att_multi.mean() - x.att_default.mean():+7.2f} {int(x.n_pen.sum()):4d}")
    for arm in ARMS:
        x = d[d.arm == arm].att_multi.to_numpy()
        rows.append(dict(inst=inst, arm=arm, mean=x.mean(), A12=a12(both, x),
                         p=float(mannwhitneyu(both, x, alternative="two-sided").pvalue),
                         both_worst=both.max(), arm_best=x.min()))
df = pd.DataFrame(rows)
df["F6"] = np.nan; df["F7"] = np.nan
for inst, d in df.groupby("inst"):
    d6 = d[d.arm != "surrogate_submitted"]
    df.loc[d6.index, "F6"] = holm(d6.p.tolist())
    df.loc[d.index, "F7"] = holm(d.p.tolist())
df["G21"] = holm(df.p.tolist())
pd.set_option("display.width", 200)
print()
print(df.to_string(float_format=lambda v: f"{v:.4f}"))
df.to_csv(SCR / "cmp_heldout_stats.csv", index=False)
