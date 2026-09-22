import csv, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "py"))
import ga_edge_closure_gnn_policy as G  # noqa: E402

INST = {
    "A": dict(id="A_k4_3od_x1.0", net="netA/network_v2.net.xml", cfg="netA/sim_3od_x1.0.sumocfg",
              protect="E28 E40", pol="output_policy_A_k4_3od_x1.0"),
    "B": dict(id="B_k9_3od_x1.0", net="netB/netB.net.xml", cfg="netB/sim_3od_x1.0.sumocfg",
              protect="E35 -E41 E38 E44 E20 E12", pol="output_policy_B_k9_3od_x1.0"),
    "J": dict(id="J_k9_6od_x1.0", net="netJ/Netzmodell2.net.xml", cfg="netJ/wildau_6od_x1.0.sumocfg",
              protect=("4935299#0 111677671#1 255274250 -27149243 27149243 -255274250 -37548821#0 "
                       "311298682#1 -111677671#1 37548821#0 876057378#6"), pol="output_policy_J_k9_6od_x1.0"),
}
NOCLOSE = {"A": 315.81, "B": 538.63, "J": 644.46}


def finals(iid, mode):
    if iid == "A_k4_3od_x1.0":
        sb = pd.read_csv(ROOT / "runs_A_k4_3od_x1.0_pm01k0" / "analysis" / "seed_best.csv")
    else:
        sb = pd.read_csv(ROOT / "merged_analysis_38_k0" / "seed_best_merged.csv")
        sb = sb[sb.instance_id == iid]
    out = []
    for _, r in sb[sb["mode"] == mode].iterrows():
        out.append([] if pd.isna(r["closed_edges"]) else str(r["closed_edges"]).split(","))
    return out


def heur_finals(iid, tag="heuristic"):
    out = []
    for p in sorted((ROOT / "runs_reviewer" / iid).glob(f"{tag}_seed*")):
        f = p / "best.txt"
        if f.exists():
            lines = f.read_text().splitlines()
            out.append(lines[1].split("=", 1)[1].split() if len(lines) > 1 else [])
    return out


for key, cfg in INST.items():
    iid = cfg["id"]
    protect = set(cfg["protect"].split()) | set(G._extract_route_endpoints(str(ROOT / cfg["cfg"])) or [])
    cands, upmap, net = G.load_candidates(str(ROOT / cfg["net"]), protect, None)
    cands = list(cands)
    n = len(cands)
    bt = {}
    with open(ROOT / f"scores_betweenness_{key}.csv", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if row and row[0] != "edge_id":
                bt[row[0]] = float(row[1])
    cen = np.array([bt.get(e, np.nan) for e in cands])
    pr = pd.Series(cen).rank(pct=True, method="average").to_numpy()
    prk = dict(zip(cands, pr))

    pol = G.try_load_policy(str(ROOT / cfg["pol"] / "policy_model.pt"), None, net, cands, "auto")
    sc, smode = pol
    risk = np.array([sc.get(e, np.nan) * (1 if smode == "risk" else -1) for e in cands])
    rho_pol = spearmanr(cen, risk, nan_policy="omit")

    samples = torch.load(ROOT / cfg["pol"] / "dataset_policy.pt", map_location="cpu", weights_only=False)
    rc = pd.read_csv(ROOT / "reviewer_diagnostics" / "results" / f"recheck_{key}.csv")
    rc = rc[(rc.att_eval_seed < 1e11) & (rc.k >= 1)]
    idx = {e: i for i, e in enumerate(cands)}
    X = np.zeros((len(rc), n)); y = rc.att_eval_seed.to_numpy(); ks = rc.k.to_numpy()
    for r, si in enumerate(rc.sample_idx.to_numpy()):
        for e in samples[int(si)]["closed_edges"]:
            if e in idx:
                X[r, idx[e]] = 1
    kmean = pd.Series(y).groupby(ks).transform("mean").to_numpy()
    res = y - kmean
    cnt = X.sum(0)
    eff = np.where(cnt > 0, (X * res[:, None]).sum(0) / np.maximum(cnt, 1), np.nan)
    ok = cnt >= (3 if key == "J" else 1)
    rho_eff = spearmanr(cen[ok], eff[ok])
    lam = 1.0
    A = X.T @ X + lam * np.eye(n)
    beta = np.linalg.solve(A, X.T @ res)
    rho_ridge = spearmanr(cen[cnt > 0], beta[cnt > 0])
    rho_pol_eff = spearmanr(risk[ok], eff[ok], nan_policy="omit")

    print(f"=== {iid}: |E_cand|={n}, policy score_mode={smode}, non-penalised k>=1 rollouts={len(rc)}")
    print(f"  Spearman(centrality, policy risk)                 = {rho_pol.statistic:+.3f} (p={rho_pol.pvalue:.3g})")
    print(f"  Spearman(centrality, per-edge closure effect)      = {rho_eff.statistic:+.3f} (p={rho_eff.pvalue:.3g}, n={ok.sum()})"
          "   [effect>0: closing raises ATT]")
    print(f"  Spearman(centrality, ridge closure effect)         = {rho_ridge.statistic:+.3f} (p={rho_ridge.pvalue:.3g})")
    print(f"  Spearman(policy risk, per-edge closure effect)     = {rho_pol_eff.statistic:+.3f} (p={rho_pol_eff.pvalue:.3g})")
    q = pd.qcut(pd.Series(cen[ok]).rank(method="first"), 3, labels=["low", "mid", "high"])
    print("  mean closure effect by centrality tercile (s):", {str(k): round(float(v), 2) for k, v in pd.Series(eff[ok]).groupby(q).mean().items()})
    if key == "J":
        base = NOCLOSE["J"]
        inform = np.abs(y - base) > 0.005
        informative_edge = (X[inform].sum(0) > 0)
        closed_any = cnt > 0
        c_inf = cen[closed_any & informative_edge]; c_inert = cen[closed_any & ~informative_edge]
        mw = mannwhitneyu(c_inf, c_inert)
        print(f"  J: informative rollouts {inform.sum()}/{len(y)}; edges closed>=1: {closed_any.sum()}, "
              f"never in an informative rollout: {(closed_any & ~informative_edge).sum()}")
        print(f"  J: centrality percentile, informative edges {pd.Series(pr[closed_any & informative_edge]).mean():.3f} "
              f"vs inert {pd.Series(pr[closed_any & ~informative_edge]).mean():.3f} (MW p={mw.pvalue:.3g})")
        print(f"  J: share of zero-centrality candidates {np.mean(cen == 0):.3f}")
    for label, sols in [("none", finals(iid, "none")), ("both", finals(iid, "both")),
                        ("heuristic", heur_finals(iid)), ("sa", heur_finals(iid, "sa"))]:
        per = [np.mean([prk[e] for e in s if e in prk]) for s in sols if s]
        if per:
            w = wilcoxon(np.array(per) - 0.5) if len(per) >= 5 else None
            print(f"  final solutions [{label:9s}] mean centrality percentile of closed edges = {np.mean(per):.3f} "
                  f"(seed range {min(per):.2f}-{max(per):.2f}; vs 0.5 Wilcoxon p={w.pvalue if w else float('nan'):.3g})")
