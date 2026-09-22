from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path(r".\reviewer_diagnostics\results\reeval_39")
PEN = 1e11
MODES = ["init", "mutation", "both"]

out = []


def P(s=""):
    print(s, flush=True)
    out.append(str(s))


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


df = pd.read_csv(R / "reeval39_long.csv")
P(f"rows {len(df)}  instances {df.instance_id.nunique()}  "
  f"SUMO seeds {sorted(df.sumo_seed.unique())}")
pen = df[df.att >= PEN]
if len(pen):
    P(f"penalised re-evaluations: {len(pen)} "
      f"({pen.groupby('mode').size().to_dict()})")
ok = df[df.att < PEN]

per_run = (ok.groupby(["instance_id", "mode", "seed"])
             .agg(att_multi=("att", "mean"),
                  att_default=("att_default", "first")).reset_index())

P("")
P("=" * 76)
P("[1] Winner's curse: default-seed ATT vs expected ATT on held-out seeds")
infl = per_run.assign(d=per_run.att_multi - per_run.att_default)
P(f"  {'mode':10s} {'default':>9s} {'held-out':>9s} {'inflation':>10s}")
for m in ["none"] + MODES:
    d = infl[infl["mode"] == m]
    P(f"  {m:10s} {d.att_default.mean():9.2f} {d.att_multi.mean():9.2f} "
      f"{d.d.mean():+10.2f}")
P(f"  mean inflation over all runs: {infl.d.mean():+.2f} s "
  f"(sd {infl.d.std(ddof=1):.2f})")
_by_mode = infl.groupby("mode")["d"].mean()
P(f"  spread of the inflation across modes: "
  f"{_by_mode.max()-_by_mode.min():.2f} s "
  f"(largest {_by_mode.idxmax()}, smallest {_by_mode.idxmin()})")
P("  Any difference in inflation BETWEEN modes is a bias in the reported")
P("  comparison, not just a shift; the guidance margin has to be read against")
P("  it.")

inst_means_d = per_run.pivot_table(index="instance_id", columns="mode",
                                   values="att_default")
inst_means_m = per_run.pivot_table(index="instance_id", columns="mode",
                                   values="att_multi")

for tag, means in [("DEFAULT SEED (as reported in the paper)", inst_means_d),
                   ("HELD-OUT SEEDS (expected ATT)", inst_means_m)]:
    P("")
    P("=" * 76)
    P(f"[2] Headline 39-instance result -- {tag}")
    P(f"  {'mode':10s} {'improved':>9s} {'meanDelta':>10s} {'median%':>9s} "
      f"{'Wilcoxon p':>11s} {'Holm p':>9s} {'meanA12':>8s}")
    col = "att_default" if "DEFAULT" in tag else "att_multi"
    n_inst = len(means)
    ps, rows = [], []
    for m in MODES:
        d = means[m] - means["none"]
        imp = 100 * (means["none"] - means[m]) / means["none"]
        w = stats.wilcoxon(means[m], means["none"])
        a_vals = []
        for inst in means.index:
            g = per_run[(per_run["mode"] == m) &
                        (per_run.instance_id == inst)][col].values
            n = per_run[(per_run["mode"] == "none") &
                        (per_run.instance_id == inst)][col].values
            if len(g) and len(n):
                a_vals.append(a12(g, n))
        rows.append((m, int((means[m] < means["none"]).sum()), d.mean(),
                     imp.median(), w.pvalue, float(np.mean(a_vals))))
        ps.append(float(w.pvalue))
    adj = holm(ps)
    for (m, ni, dm, md, p, a), pa in zip(rows, adj):
        P(f"  {m:10s} {ni:4d}/{n_inst:<3d} {dm:+10.2f} {md:9.3f} {p:11.2e} "
          f"{pa:9.4f} {a:8.3f}{'  *' if pa < 0.05 else ''}")

P("")
P("=" * 76)
P("[3] Per-network mean improvement (Table 2), both ways")
net = inst_means_m.index.str[0]
for tag, means in [("default", inst_means_d), ("held-out", inst_means_m)]:
    P(f"  --- {tag} ---")
    for m in MODES:
        imp = 100 * (means["none"] - means[m]) / means["none"]
        P(f"    {m:9s} " + str(imp.groupby(net).mean().round(2).to_dict()))

P("")
P("=" * 76)
P("[4] Is the margin distinguishable from seed noise?")
seed_sd = ok.groupby(["instance_id", "mode", "seed"])["att"].std(ddof=1).dropna()
if seed_sd.empty:
    P("  (only one SUMO seed present -- cannot estimate the seed spread)")
    seed_sd = pd.Series([float("nan")])
P(f"  within-solution SUMO-seed sd, median over runs: {seed_sd.median():.2f} s")
P(f"                                mean over runs  : {seed_sd.mean():.2f} s")
for m in MODES:
    d_def = (inst_means_d[m] - inst_means_d["none"])
    d_mul = (inst_means_m[m] - inst_means_m["none"])
    P(f"  {m:9s}: paper margin {d_def.mean():+.2f} s | held-out margin "
      f"{d_mul.mean():+.2f} s | ratio to seed sd "
      f"{abs(d_mul.mean())/seed_sd.mean():.3f}")

P("")
P("=" * 76)
P("[5] Do the two evaluations agree instance by instance?")
for m in MODES:
    dd = (inst_means_d[m] - inst_means_d["none"])
    dm = (inst_means_m[m] - inst_means_m["none"])
    rho = stats.spearmanr(dd, dm).statistic
    agree = int((np.sign(dd) == np.sign(dm)).sum())
    P(f"  {m:9s}: sign agreement {agree}/{len(dd)}   Spearman rho = {rho:+.3f}")

Path(R / "reeval39_analysis.txt").write_text("\n".join(out), encoding="utf-8")
per_run.to_csv(R / "reeval39_per_run.csv", index=False)
print("\nwrote", R / "reeval39_analysis.txt")
