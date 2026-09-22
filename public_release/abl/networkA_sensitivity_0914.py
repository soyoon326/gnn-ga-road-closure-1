import pandas as pd, numpy as np, re, glob
from math import comb
from scipy.stats import wilcoxon
PEN = 1e11
sb = pd.read_csv("./merged_analysis_38_k0/seed_best_merged.csv", encoding="utf-8-sig", low_memory=False, usecols=["instance_id", "mode", "seed", "ATT"])
sb = sb[sb["ATT"] < PEN]
m = sb.groupby(["instance_id", "mode"])["ATT"].mean().unstack()
net = np.array([i[0] for i in m.index])
modes = ["init", "mutation", "both"]
def a12(x, y):
    x = np.asarray(x)[:, None]; y = np.asarray(y)[None, :]
    return float(((x < y).sum() + 0.5 * (x == y).sum()) / (x.shape[0] * y.shape[1]))
pct = pd.DataFrame({md: 100 * (m["none"] - m[md]) / m["none"] for md in modes})
dsec = pd.DataFrame({md: m["none"] - m[md] for md in modes})
A12 = pd.DataFrame({md: [a12(sb[(sb.instance_id == i) & (sb["mode"] == md)]["ATT"], sb[(sb.instance_id == i) & (sb["mode"] == "none")]["ATT"]) for i in m.index] for md in modes}, index=m.index)
def holm(ps):
    order = np.argsort(ps); out = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); out[i] = run
    return out
def summary(mask, label):
    ps = [wilcoxon(m.loc[mask, md] - m.loc[mask, "none"]).pvalue for md in modes]
    hp = holm(ps)
    for k, md in enumerate(modes):
        s = pct.loc[mask, md]
        pooled = 100 * dsec.loc[mask, md].mean() / m.loc[mask, "none"].mean()
        print(f"{label:7s} {md:8s} n={mask.sum():2d} inst-mean%={s.mean():.3f} median%={s.median():.3f} pooled%={pooled:.3f} meanDs={-dsec.loc[mask, md].mean():+.2f} improved={(s > 0).sum()}/{mask.sum()} A12={A12.loc[mask, md].mean():.3f} Wp={ps[k]:.2e} Holm={hp[k]:.2e}")
summary(np.ones(len(m), bool), "all38")
for n in "ABCDJ":
    summary(net != n, f"no-{n}")
for md in modes:
    print(md, "A share of summed %:", round(pct.loc[net == "A", md].sum() / pct[md].sum() * 100, 1), "| of summed seconds:", round(dsec.loc[net == "A", md].sum() / dsec[md].sum() * 100, 1))
top = pct["both"].sort_values(ascending=False)
print("top10 both:", [(i, round(v, 2)) for i, v in top.head(10).items()])
print("per-network both inst-mean%:", {n: round(pct.loc[net == n, "both"].mean(), 3) for n in "ABCDJ"})
print("--- saturation on network A (GA modes, default seed)")
for inst in [i for i in m.index if i[0] == "A"]:
    g = sb[sb.instance_id == inst]; best = g["ATT"].min()
    row = []
    for md in ["none", "init", "mutation", "both"]:
        v = g[g["mode"] == md]["ATT"].values
        row.append(f"{md} {int(np.isclose(v, best, atol=0.005).sum())}/10 min {v.min():.2f} mean {v.mean():.2f}")
    print(f"{inst} best {best:.2f} | " + " | ".join(row))
print("space sizes:", {n: sum(comb(n, k) for k in range(5)) for n in (38, 39, 40)})
def arm(inst, name):
    out = []
    for d in sorted(glob.glob(f"./runs_reviewer/{inst}/{name}_seed*")):
        t = open(d + "/best.txt", encoding="utf-8", errors="ignore").read()
        out.append(float(re.search(r"best_score=([0-9.]+)", t).group(1)))
    return np.array(out)
sur = pd.read_csv("./reviewer_diagnostics/results/arm_A_sum_run/arm_results.csv")
sur = sur[sur.att < PEN].groupby("seed")["att"].min().values
for name, v in [("heuristic", arm("A_k4_3od_x1.0", "heuristic")), ("sa", arm("A_k4_3od_x1.0", "sa")), ("surrogate", sur)]:
    print(f"A_k4_3od_x1.0 {name}: reach 223.89 on {int(np.isclose(v, 223.893, atol=0.005).sum())}/{len(v)} seeds, mean {v.mean():.2f}")
