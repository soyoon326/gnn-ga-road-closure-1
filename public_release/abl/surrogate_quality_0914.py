import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "py"))
from policy_gnn_model import PolicyGNN  # noqa: E402
from train_policy_gnn_v2 import load_blob  # noqa: E402

D = {"A": "output_policy_A_k4_3od_x1.0", "B": "output_policy_B_k9_3od_x1.0", "J": "output_policy_J_k9_6od_x1.0"}
RES = ROOT / "reviewer_diagnostics" / "results"


def rho_scipy(a, b):
    return float(spearmanr(a, b).statistic)


def rho_argsort(a, b):
    ra = np.argsort(np.argsort(np.asarray(a))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b))).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


rows = []
for K, d in D.items():
    if not (ROOT / d / "graph_policy.pt").exists():
        print(f"skipping {K}: {d}/graph_policy.pt is not in this release")
        continue
    g = load_blob(ROOT / d / "graph_policy.pt")
    samples = load_blob(ROOT / d / "dataset_policy.pt")
    ei, xn, ea0 = g["edge_index"].long(), g["x_node"].float(), g["edge_attr"].float()
    rc = pd.read_csv(RES / f"recheck_{K}.csv")
    ev = dict(zip(rc.sample_idx.astype(int), rc.att_eval_seed))
    ylab = np.array([float(s["y"]) for s in samples])

    def load(p):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = PolicyGNN(ck["in_node"], ck["in_edge"], hid=ck["hid"], heads=ck["heads"])
        m.load_state_dict(ck["state_dict"]); m.eval()
        return m, ck

    def preds(m, readout, ids):
        out = []
        with torch.no_grad():
            for i in ids:
                mk = samples[int(i)]["mask"].float().view(-1)
                sc = m(xn, ei, torch.cat([ea0, mk.view(-1, 1)], 1)).squeeze(-1)
                s = (sc * mk).sum()
                out.append(float(s / mk.sum().clamp(min=1.0)) if readout == "mean" else float(s))
        return np.array(out)

    keep = np.array(sorted(i for i, a in ev.items() if a < 1e11))
    n = len(keep)
    perm = np.random.RandomState(42).permutation(n)
    n_te = int(0.15 * n); n_va = int(0.15 * n)
    splits = {"test": keep[perm[:n_te]], "val": keep[perm[n_te:n_te + n_va]],
              "train": keep[perm[n_te + n_va:]], "all": keep}
    m, ck = load(ROOT / "reviewer_diagnostics" / "models" / f"{K}_sum_clean.pt")
    for name, ids in splits.items():
        p = preds(m, "sum", ids); y = np.array([ev[int(i)] for i in ids])
        rows.append(dict(inst=K, model=f"{K}_sum_clean (reported arm)", subset=name, n=len(ids),
                         rho_obj=rho_scipy(p, y), rho_label=np.nan))
    for name, ids in splits.items():
        y = np.array([ev[int(i)] for i in ids]); yl = ylab[ids.astype(int)]
        rows.append(dict(inst=K, model="label (per-sample seed)", subset=name, n=len(ids),
                         rho_obj=rho_scipy(yl, y), rho_label=np.nan))

    np.random.seed(42)
    idx = np.random.permutation(len(samples))
    nval = min(max(1, int(0.15 * len(samples))), len(samples) - 1)
    own_val, own_tr = idx[:nval], idx[nval:]
    for fname in ["surrogate_att.pt", "surrogate_att_v2.pt"]:
        p_ = ROOT / d / fname
        if not p_.exists():
            continue
        m2, ck2 = load(p_)
        pv = preds(m2, "mean", own_val)
        rows.append(dict(inst=K, model=f"{fname} (hid {ck2['hid']})", subset="own-val vs TRAINING label", n=len(own_val),
                         rho_obj=np.nan, rho_label=rho_scipy(pv, ylab[own_val]), rho_label_argsort=rho_argsort(pv, ylab[own_val])))
        for name, ids in [("own-val", [i for i in own_val if int(i) in ev and ev[int(i)] < 1e11]),
                          ("own-train", [i for i in own_tr if int(i) in ev and ev[int(i)] < 1e11]),
                          ("all", list(keep)), ("v3 test split", list(splits["test"]))]:
            ids = np.array(ids)
            p = preds(m2, "mean", ids); y = np.array([ev[int(i)] for i in ids])
            rows.append(dict(inst=K, model=f"{fname} (hid {ck2['hid']})", subset=name, n=len(ids),
                             rho_obj=rho_scipy(p, y), rho_label=rho_scipy(p, ylab[ids.astype(int)])))
    print(f"done {K}", flush=True)

df = pd.DataFrame(rows)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
print(df.to_string(float_format=lambda v: f"{v:.3f}"))
df.to_csv(ROOT / "manuscript_numbers_k0" / "surrogate_quality_0914.csv", index=False)
