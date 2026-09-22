import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = Path(".")
M = Path(os.environ.get("K0_MERGED", "./merged_analysis_38_k0"))
SD = Path(os.environ.get("K0_SD", "./figs38/signal_density.csv"))
print(f"bundle: {M}")

agg = pd.read_csv(M / "aggregate_merged.csv", encoding="utf-8-sig", usecols=["instance_id", "mode", "seed", "gen", "ATT", "is_valid", "Reason"], low_memory=False)
valid = agg["is_valid"].astype(str).str.lower() == "true"
print(f"[3.3] evaluations simulated {len(agg):,}; penalised {int((~valid).sum())} ({100*(~valid).mean():.3f}%); instances with a penalty {agg[~valid].instance_id.nunique()}; on J {int((~valid & agg.instance_id.str.startswith('J')).sum())}")

sb = pd.read_csv(M / "seed_best_merged.csv", encoding="utf-8-sig")[["instance_id", "mode", "seed", "ATT"]]
piv = sb.groupby(["instance_id", "mode"]).ATT.mean().unstack("mode")
imp = pd.DataFrame({m: 100 * (piv["none"] - piv[m]) / piv["none"] for m in ("init", "mutation", "both")})
delta = pd.DataFrame({m: piv[m] - piv["none"] for m in ("init", "mutation", "both")})
n = len(piv); print(f"instances: {n}")
for m in ("init", "mutation", "both"):
    print(f"[Table2/3] {m:8s} improved {(imp[m] > 0).sum()}/{n}  mean dATT {delta[m].mean():+.2f} s  median % {imp[m].median():.3f}  mean % {imp[m].mean():.3f}  pooled % {100*(-delta[m].mean())/piv['none'].mean():.3f}")
best = imp.idxmax(axis=1).value_counts().to_dict(); print("[4.1] best guided mode counts:", best)
net = imp.index.str[0]
for m in ("init", "mutation", "both"):
    print(f"[Table2 net] {m:8s} " + "  ".join(f"{k}:{imp[m][net == k].mean():+.2f}%({(imp[m][net == k] > 0).sum()}/{(net == k).sum()})" for k in "ABCDJ"))
dem = imp.index.str[-3:]
for m in ("mutation", "both"):
    print(f"[Table2 dem] {m:8s} " + "  ".join(f"x{k}:{imp[m][dem == k].mean():+.3f}" for k in ("0.8", "1.0", "1.2")))

def a12(g, u):
    return float(((g[:, None] < u[None, :]).sum() + 0.5 * (g[:, None] == u[None, :]).sum()) / (len(g) * len(u)))
rows = {}
for m in ("init", "mutation", "both"):
    vals = []
    for i in piv.index:
        g = sb[(sb.instance_id == i) & (sb["mode"] == m)].ATT.values; u = sb[(sb.instance_id == i) & (sb["mode"] == "none")].ATT.values
        vals.append(a12(g, u))
    vals = np.array(vals); rows[m] = vals
    print(f"[A12] {m:8s} mean {vals.mean():.3f}  >0.5: {(vals > 0.5).sum()}/{n}  medium(>=0.64): {(vals >= 0.64).sum()}  large(>=0.71): {(vals >= 0.71).sum()}")

rng = np.random.default_rng(0); nets = sorted(set(net)); B = 20000
for m in ("mutation", "both"):
    groups = {k: imp[m][net == k].values for k in nets}
    pooled, netmean = [], []
    for _ in range(B):
        pick = rng.choice(nets, size=len(nets), replace=True)
        allv = np.concatenate([groups[k] for k in pick]); pooled.append(allv.mean()); netmean.append(np.mean([groups[k].mean() for k in pick]))
    print(f"[3.12 bootstrap] {m:8s} pooled-mean 95% [{np.percentile(pooled, 2.5):+.2f}, {np.percentile(pooled, 97.5):+.2f}]  network-mean 95% [{np.percentile(netmean, 2.5):+.2f}, {np.percentile(netmean, 97.5):+.2f}]  positive on {sum(groups[k].mean() > 0 for k in nets)}/5 networks")

s = imp["mutation"]
print("[5.1 SD by network]", {k: round(float(s[net == k].std(ddof=1)), 2) for k in "ABCDJ"})
sd = pd.read_csv(SD, encoding="utf-8-sig").set_index("instance_id")
dens = sd["obs_per_edge"].reindex(s.index)
lo, hi = s[dens < 10], s[dens >= 10]
print(f"[5.1 groups] low n={len(lo)} mean {lo.mean():+.2f} sd {lo.std(ddof=1):.2f} range [{lo.min():+.2f},{lo.max():+.2f}] | high n={len(hi)} mean {hi.mean():+.2f} sd {hi.std(ddof=1):.2f} range [{hi.min():+.2f},{hi.max():+.2f}]")
r = spearmanr(dens, s); print(f"[5.1 rho density vs mutation benefit] rho {r.statistic:+.2f} p {r.pvalue:.2f}   high-density range {dens[dens>=10].min():.1f}-{dens[dens>=10].max():.1f}")


a = sb[sb.instance_id == "A_k4_3od_x1.0"].groupby("mode").ATT.agg(["mean", "min", "max"]).round(2)
print("[Table4 A]\n" + a.to_string())
print(f"   both best minus 223.89: {a.loc['both','min'] - 223.89:+.2f}; modes reaching 223.89: {[m for m in a.index if abs(a.loc[m,'min'] - 223.89) < 0.005]}")

sc = pd.read_csv(M / "extra_analysis/diversity_vs_gain_scatter_ready.csv", encoding="utf-8-sig"); d = sc[sc["mode"] == "both"]
ps, rs = [], []
for c in ("delta_gen1_unique_ratio_vs_none", "delta_gen1_best_ATT_vs_none", "delta_gen1_mean_ATT_vs_none"):
    r = spearmanr(d[c], d["mean_reduction_vs_none_%"]); rs.append(r.statistic); ps.append(r.pvalue)
order = np.argsort(ps); holm = np.empty(3); run = 0
for rank, i in enumerate(order):
    run = max(run, ps[i] * (3 - rank)); holm[i] = min(1, run)
print(f"[4.3 corr both] rho {[round(x,3) for x in rs]}  p {[round(x,4) for x in ps]}  Holm {[round(x,3) for x in holm]}")
for m in ("init", "mutation"):
    dd = sc[sc["mode"] == m]
    if dd["delta_gen1_unique_ratio_vs_none"].notna().any():
        r = spearmanr(dd["delta_gen1_unique_ratio_vs_none"], dd["mean_reduction_vs_none_%"]); print(f"[4.3 corr {m}] rho {r.statistic:+.3f} p {r.pvalue:.3f}")
ov = pd.read_csv(M / "extra_analysis/solution_overlap_summary.csv", encoding="utf-8-sig")
print("[4.2 overlap] columns:", list(ov.columns)[:8]); print(ov.groupby("mode").mean(numeric_only=True).round(3).to_string() if "mode" in ov.columns else ov.head().to_string())
cv = pd.read_csv(M / "extra_analysis/convergence_summary_by_mode.csv", encoding="utf-8-sig")
print("[4.2 conv] " + cv.groupby("mode")[["mean_first_hit_95pct_own_final_gen", "mean_progress_auc", "mean_final_best_ATT"]].mean().round(3).to_string())
