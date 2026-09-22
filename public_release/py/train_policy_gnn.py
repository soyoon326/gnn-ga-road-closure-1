from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from policy_gnn_model import PolicyGNN


def set_score_from_edge_scores(edge_scores: torch.Tensor, mask: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    s = edge_scores.view(-1)
    m = mask.view(-1)
    if reduce == "sum":
        return (s * m).sum()
    denom = m.sum().clamp(min=1.0)
    return (s * m).sum() / denom


def load_blob(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def infer_label_mode(outdir: Path, graph_blob: dict, args_label_mode: str) -> str:
    if args_label_mode != "auto":
        return args_label_mode
    if "label_mode" in graph_blob:
        return str(graph_blob["label_mode"])
    meta_path = outdir / "dataset_meta.json"
    if meta_path.exists():
        try:
            return str(json.loads(meta_path.read_text(encoding="utf-8")).get("label_mode", "cost"))
        except Exception:
            pass
    return "cost"


def split_best_worst(samps: List[dict], label_mode: str, qlo: float = 0.1, qhi: float = 0.9) -> Tuple[List[dict], List[dict]]:
    ys = np.array([float(s["y"]) for s in samps], dtype=np.float64)
    lo, hi = np.quantile(ys, qlo), np.quantile(ys, qhi)
    if label_mode == "benefit":
        best = [s for s, y in zip(samps, ys) if y >= hi]
        worst = [s for s, y in zip(samps, ys) if y <= lo]
    else:
        best = [s for s, y in zip(samps, ys) if y <= lo]
        worst = [s for s, y in zip(samps, ys) if y >= hi]
    return best, worst


def main():
    start = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="output_policy")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--model-out", type=str, default=None)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--label-mode", choices=["auto", "cost", "benefit"], default="auto",
                    help="cost: lower y is better. benefit: higher y is better.")
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--use-layernorm", action="store_true")
    ap.add_argument("--reduce", choices=["mean", "sum"], default="mean")
    ap.add_argument("--qlo", type=float, default=0.1)
    ap.add_argument("--qhi", type=float, default=0.9)
    args = ap.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False

    outdir = Path(args.data)
    graph = load_blob(outdir / "graph_policy.pt")
    edge_index = graph["edge_index"]
    x_node = graph["x_node"]
    edge_attr = graph["edge_attr"]
    samples = load_blob(outdir / "dataset_policy.pt")
    if len(samples) < 4:
        raise SystemExit("Need at least 4 samples for train/validation split.")

    label_mode = infer_label_mode(outdir, graph, args.label_mode)
    higher_is_better = label_mode == "benefit"
    print(f"[train] label_mode={label_mode} | {'higher y is better' if higher_is_better else 'lower y is better'}")

    idx = np.random.permutation(len(samples))
    nval = int(args.val_split * len(samples))
    nval = min(max(1, nval), len(samples) - 1)
    val_idx, train_idx = idx[:nval], idx[nval:]
    samples_tr = [samples[int(i)] for i in train_idx.tolist()]
    samples_va = [samples[int(i)] for i in val_idx.tolist()]

    in_node = x_node.size(1)
    in_edge = edge_attr.size(1)
    model = PolicyGNN(
        in_node,
        in_edge,
        hid=args.hid,
        heads=args.heads,
        dropout=args.dropout,
        use_layernorm=args.use_layernorm,
    )
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("Requested --device cuda but CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    x_node = x_node.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    save_path = Path(args.model_out) if args.model_out else (outdir / "policy_model.pt")
    for ep in range(1, args.epochs + 1):
        model.train()
        best, worst = split_best_worst(samples_tr, label_mode, qlo=args.qlo, qhi=args.qhi)
        if not best or not worst:
            raise RuntimeError("Could not split best/worst samples. Check dataset labels or quantiles.")
        tot = 0.0
        n = 0
        for _ in range(max(64, len(samples_tr) // 2)):
            sa = random.choice(best)
            sb = random.choice(worst)
            scores = model(x_node, edge_index, edge_attr).squeeze(-1)
            pred_best = set_score_from_edge_scores(scores, sa["mask"].to(device).float(), reduce=args.reduce)
            pred_worst = set_score_from_edge_scores(scores, sb["mask"].to(device).float(), reduce=args.reduce)
            if higher_is_better:
                loss = torch.clamp(args.margin - (pred_best - pred_worst), min=0.0)
            else:
                loss = torch.clamp(args.margin - (pred_worst - pred_best), min=0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            n += 1
        tr = tot / max(1, n)

        model.eval()
        with torch.no_grad():
            best, worst = split_best_worst(samples_va, label_mode, qlo=args.qlo, qhi=args.qhi)
            tot = 0.0
            n = 0
            for _ in range(200):
                sa = random.choice(best)
                sb = random.choice(worst)
                scores = model(x_node, edge_index, edge_attr).squeeze(-1)
                pred_best = set_score_from_edge_scores(scores, sa["mask"].to(device).float(), reduce=args.reduce)
                pred_worst = set_score_from_edge_scores(scores, sb["mask"].to(device).float(), reduce=args.reduce)
                if higher_is_better:
                    loss = torch.clamp(args.margin - (pred_best - pred_worst), min=0.0)
                else:
                    loss = torch.clamp(args.margin - (pred_worst - pred_best), min=0.0)
                tot += float(loss.item())
                n += 1
            va = tot / max(1, n)
        print(f"[{ep:03d}] train {tr:.3f} | val {va:.3f}")
        if va < best_val:
            best_val = va
            torch.save({
                "state_dict": model.state_dict(),
                "in_node": in_node,
                "in_edge": in_edge,
                "hid": args.hid,
                "heads": args.heads,
                "dropout": args.dropout,
                "use_layernorm": bool(args.use_layernorm),
                "label_mode": label_mode,
                "score_mode": "benefit" if higher_is_better else "risk",
                "reduce": args.reduce,
                "edge_feature_names": graph.get("edge_feature_names"),
                "node_feature_names": graph.get("node_feature_names"),
                "baseline_cost": graph.get("baseline_cost"),
            }, save_path)
    print("Saved", save_path)
    print(f"Elapsed: {time.perf_counter() - start:.3f} sec")


if __name__ == "__main__":
    main()
