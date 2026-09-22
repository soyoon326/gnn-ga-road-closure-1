import pandas as pd, numpy as np, re, glob
from scipy.stats import mannwhitneyu
PEN = 1e11
sb = pd.read_csv("./merged_analysis_38_k0/seed_best_merged.csv", encoding="utf-8-sig", low_memory=False)
sb = sb[sb["ATT"] < PEN]
def a12(x, y):
    x = np.asarray(x)[:, None]; y = np.asarray(y)[None, :]
    return float(((x < y).sum() + 0.5 * (x == y).sum()) / (x.shape[0] * y.shape[1]))
for mode in ["init", "mutation", "both"]:
    win_only, mean_only = [], []
    for inst, g in sb.groupby("instance_id"):
        n = g[g["mode"] == "none"]["ATT"].values
        m = g[g["mode"] == mode]["ATT"].values
        A = a12(m, n); lower = m.mean() < n.mean()
        if A > 0.5 and not lower: win_only.append((inst, round(A, 3), round(m.mean() - n.mean(), 2)))
        if lower and not A > 0.5: mean_only.append((inst, round(A, 3), round(m.mean() - n.mean(), 2)))
    print(mode, "A12>0.5 but mean not lower:", win_only, "| mean lower but A12<=0.5:", mean_only)
def arm(inst, name):
    vals = []
    for d in sorted(glob.glob(f"./runs_reviewer/{inst}/{name}_seed*")):
        t = open(d + "/best.txt", encoding="utf-8", errors="ignore").read()
        s = float(re.search(r"best_score=([0-9.]+)", t).group(1))
        e = re.search(r"closed_edges=(.*)", t); e = e.group(1).strip() if e else ""
        vals.append((d[-1], s, e))
    return vals
for inst in ["A_k4_3od_x1.0", "B_k9_3od_x1.0", "J_k9_6od_x1.0"]:
    none = sb[(sb.instance_id == inst) & (sb["mode"] == "none")]["ATT"].values
    both = sb[(sb.instance_id == inst) & (sb["mode"] == "both")]["ATT"].values
    for name in ["sa", "heuristic"]:
        v = [s for _, s, _ in arm(inst, name) if s < PEN]
        print(inst, name, "vs none: mean", round(np.mean(v), 2), "MW p", round(mannwhitneyu(v, none, alternative="two-sided").pvalue, 4),
              "| vs both MW p", round(mannwhitneyu(v, both, alternative="two-sided").pvalue, 4))
inst = "A_k4_3od_x1.0"
g = sb[(sb.instance_id == inst) & (np.isclose(sb["ATT"], 223.893, atol=0.01))]
print("GA modes at 223.893:", g[["mode", "seed", "closed_edges"]].values.tolist())
for name in ["sa", "heuristic"]:
    print(name, [(sd, s, e) for sd, s, e in arm(inst, name) if abs(s - 223.893) < 0.01])
g2 = sb[(sb.instance_id == inst) & (np.isclose(sb["ATT"], 224.053, atol=0.01))]
print("GA modes at 224.053:", g2[["mode", "seed", "closed_edges"]].values.tolist()[:6])
