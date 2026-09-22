from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_gnn_model import PolicyGNN  # noqa: E402
from train_policy_gnn_v2 import load_blob, infer_label_mode  # noqa: E402


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def main() -> int:
    start = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model-out", default=None, help="default: <data>/surrogate_att.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--label-mode", choices=["auto", "cost", "benefit"], default="auto")
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    outdir = Path(args.data)
    graph = load_blob(outdir / "graph_policy.pt")
    edge_index = graph["edge_index"].long()
    x_node = graph["x_node"].float()
    edge_attr = graph["edge_attr"].float()
    dataset_path = Path(args.dataset) if args.dataset else (outdir / "dataset_policy.pt")
    samples = load_blob(dataset_path)
    label_mode = infer_label_mode(outdir, graph, args.label_mode)
    higher_is_better = label_mode == "benefit"
    print(f"[surrogate] dataset n={len(samples)} label_mode={label_mode}")

    ys = np.array([float(s["y"]) for s in samples], dtype=np.float32)
    masks = torch.stack([s["mask"].float().view(-1) for s in samples])
    y_mean, y_std = float(ys.mean()), float(ys.std() or 1.0)
    yt = torch.tensor((ys - y_mean) / y_std)

    idx = np.random.permutation(len(samples))
    nval = min(max(1, int(args.val_split * len(samples))), len(samples) - 1)
    val_idx, tr_idx = idx[:nval], idx[nval:]

    in_node, in_edge = x_node.size(1), edge_attr.size(1) + 1
    model = PolicyGNN(in_node, in_edge, hid=args.hid, heads=args.heads)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = torch.nn.SmoothL1Loss()

    def predict(mask_vec: torch.Tensor) -> torch.Tensor:
        ea = torch.cat([edge_attr, mask_vec.view(-1, 1)], dim=1)
        scores = model(x_node, edge_index, ea).squeeze(-1)
        m = mask_vec.view(-1)
        return (scores * m).sum() / m.sum().clamp(min=1.0)

    save_path = Path(args.model_out) if args.model_out else (outdir / "surrogate_att.pt")
    best_rho = -2.0
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(tr_idx)
        tot, n = 0.0, 0
        for bstart in range(0, len(perm), args.batch):
            bidx = perm[bstart:bstart + args.batch]
            preds = torch.stack([predict(masks[i]) for i in bidx])
            loss = lossf(preds, yt[bidx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); n += 1
        model.eval()
        with torch.no_grad():
            pv = np.array([float(predict(masks[i])) for i in val_idx])
        rho = spearman(pv, ys[val_idx])
        mae = float(np.mean(np.abs(pv * y_std + y_mean - ys[val_idx])))
        print(f"[{ep:03d}] train {tot/max(1,n):.4f} | val spearman {rho:.3f} MAE {mae:.3f}")
        if rho > best_rho:
            best_rho = rho
            torch.save({
                "state_dict": model.state_dict(),
                "in_node": in_node, "in_edge": in_edge,
                "hid": args.hid, "heads": args.heads,
                "label_mode": label_mode,
                "higher_is_better": higher_is_better,
                "y_mean": y_mean, "y_std": y_std,
                "surrogate": True,
                "baseline_cost": graph.get("baseline_cost"),
            }, save_path)
    print(f"Saved {save_path} (best val spearman {best_rho:.3f})")
    print(f"Elapsed: {time.perf_counter() - start:.3f} sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
