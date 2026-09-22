from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "py"))
from policy_gnn_model import PolicyGNN  # noqa: E402
from train_policy_gnn_v2 import load_blob  # noqa: E402


class KHead(nn.Module):

    def __init__(self, hid: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        kk = k.view(1, 1) / 10.0
        return self.net(torch.cat([kk, kk ** 2], dim=1)).squeeze()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--readout", default="mean", choices=["mean", "sum", "sumk"])
    ap.add_argument("--labels", default="dataset",
                    help="'dataset' = original random-seed labels, or a path to "
                         "recheck.csv for fixed-eval-seed labels")
    ap.add_argument("--drop-empty", action="store_true",
                    help="drop k=0 samples (constant prediction by construction)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model-out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    t0 = time.perf_counter()

    ddir = Path(args.data)
    graph = load_blob(ddir / "graph_policy.pt")
    edge_index = graph["edge_index"].long()
    x_node = graph["x_node"].float()
    edge_attr = graph["edge_attr"].float()
    samples = load_blob(ddir / "dataset_policy.pt")

    ys = np.array([float(s["y"]) for s in samples], dtype=np.float64)
    keep = np.arange(len(samples))

    if args.labels != "dataset":
        import csv as _csv
        clean = {}
        with open(args.labels, newline="", encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                att = float(r["att_eval_seed"])
                if att < 1e11:
                    clean[int(r["sample_idx"])] = att
        keep = np.array(sorted(clean))
        ys = np.array([clean[int(i)] for i in keep], dtype=np.float64)

    masks = torch.stack([samples[int(i)]["mask"].float().view(-1) for i in keep])
    ks = masks.sum(dim=1).numpy()

    if args.drop_empty:
        sel = ks > 0
        masks, ys, ks, keep = masks[sel], ys[sel], ks[sel], keep[sel]

    n = len(ys)
    y_mean, y_std = float(ys.mean()), float(ys.std() or 1.0)
    yt = torch.tensor((ys - y_mean) / y_std, dtype=torch.float32)

    rs = np.random.RandomState(args.seed)
    idx = rs.permutation(n)
    n_te = int(0.15 * n); n_va = int(0.15 * n)
    te_idx, va_idx, tr_idx = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]

    in_node, in_edge = x_node.size(1), edge_attr.size(1) + 1
    model = PolicyGNN(in_node, in_edge, hid=args.hid, heads=args.heads)
    khead = KHead() if args.readout == "sumk" else None
    params = list(model.parameters()) + (list(khead.parameters()) if khead else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    lossf = nn.SmoothL1Loss()

    def predict(mask_vec: torch.Tensor) -> torch.Tensor:
        ea = torch.cat([edge_attr, mask_vec.view(-1, 1)], dim=1)
        scores = model(x_node, edge_index, ea).squeeze(-1)
        m = mask_vec.view(-1)
        s = (scores * m).sum()
        if args.readout == "mean":
            return s / m.sum().clamp(min=1.0)
        if args.readout == "sum":
            return s
        return s + khead(m.sum())

    def evaluate(ii):
        model.eval()
        if khead:
            khead.eval()
        with torch.no_grad():
            p = np.array([float(predict(masks[int(i)])) for i in ii])
        return p

    lower_is_better = True

    best_rho, best_state, best_ep = -2.0, None, -1
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        if khead:
            khead.train()
        perm = np.random.permutation(tr_idx)
        tot, nb = 0.0, 0
        for b in range(0, len(perm), args.batch):
            bi = perm[b:b + args.batch]
            preds = torch.stack([predict(masks[int(i)]) for i in bi])
            loss = lossf(preds, yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        pv = evaluate(va_idx)
        rho = float(spearmanr(pv, ys[va_idx]).statistic)
        hist.append((ep, tot / max(1, nb), rho))
        if rho > best_rho:
            best_rho, best_ep = rho, ep
            best_state = ({k: v.clone() for k, v in model.state_dict().items()},
                          {k: v.clone() for k, v in khead.state_dict().items()} if khead else None)
        if ep % 10 == 0 or ep == 1:
            print(f"[{ep:03d}] train {tot/max(1,nb):.4f} | val rho {rho:+.3f} "
                  f"(best {best_rho:+.3f} @ep{best_ep})", flush=True)

    model.load_state_dict(best_state[0])
    if khead and best_state[1]:
        khead.load_state_dict(best_state[1])

    pt = evaluate(te_idx)
    yte = ys[te_idx]
    rho_te = float(spearmanr(pt, yte).statistic)
    pick = np.argsort(pt)[:10] if lower_is_better else np.argsort(-pt)[:10]
    regret = float(yte[pick].min() - yte.min())
    rnd = np.array([yte[np.random.choice(len(yte), 10, replace=False)].min()
                    for _ in range(2000)])
    regret_random = float(rnd.mean() - yte.min())
    top_decile = set(np.argsort(yte)[:max(1, len(yte) // 10)].tolist())
    prec = len(set(pick.tolist()) & top_decile) / 10.0

    out = dict(data=str(ddir.name), readout=args.readout, labels=args.labels,
               drop_empty=bool(args.drop_empty), n=int(n),
               n_train=len(tr_idx), n_val=len(va_idx), n_test=len(te_idx),
               best_epoch=best_ep, val_rho=round(best_rho, 4),
               test_rho=round(rho_te, 4),
               test_regret_at10=round(regret, 3),
               test_regret_at10_random=round(regret_random, 3),
               test_top_decile_precision=prec,
               label_sd=round(float(ys.std(ddof=1)), 3),
               minutes=round((time.perf_counter() - t0) / 60, 2))
    print("\nRESULT " + json.dumps(out), flush=True)

    if args.model_out:
        torch.save({"state_dict": model.state_dict(), "in_node": in_node,
                    "in_edge": in_edge, "hid": args.hid, "heads": args.heads,
                    "label_mode": "cost", "higher_is_better": False,
                    "y_mean": y_mean, "y_std": y_std, "surrogate": True,
                    "readout": args.readout,
                    "khead": khead.state_dict() if khead else None},
                   args.model_out)
        print("saved", args.model_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
