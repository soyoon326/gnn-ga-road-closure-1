from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_gnn_model import PolicyGNN


def load_blob(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def infer_label_mode(outdir: Path, graph_blob: dict, args_label_mode: str) -> str:
    if args_label_mode != "auto":
        return args_label_mode
    if "label_mode" in graph_blob:
        return str(graph_blob["label_mode"])
    for name in ("dataset_meta.json", "dataset_meta_merged.json"):
        p = outdir / name
        if p.exists():
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    if "label_mode" in meta:
                        return str(meta["label_mode"])
                    pm = meta.get("parts_meta")
                    if pm:
                        return str(pm[0].get("label_mode", "cost"))
            except Exception:
                pass
    return "cost"


class PairSampler:

    def __init__(self, ys: np.ndarray, higher_is_better: bool,
                 q_top: float, gap_min: float, qlo: float, qhi: float, rng: random.Random):
        self.y = ys if higher_is_better else -ys
        self.rng = rng
        self.gap_min = gap_min
        n = len(self.y)
        order = np.argsort(-self.y)
        self.top_pool = order[: max(2, int(n * (1.0 - q_top)))]
        lo, hi = np.quantile(self.y, qlo), np.quantile(self.y, qhi)
        self.best_pool = np.where(self.y >= hi)[0]
        self.worst_pool = np.where(self.y <= lo)[0]
        if len(self.best_pool) == 0 or len(self.worst_pool) == 0:
            raise RuntimeError("coarse pool is empty; check labels/quantiles")

    def coarse(self):
        return (int(self.rng.choice(self.best_pool)),
                int(self.rng.choice(self.worst_pool)))

    def fine(self, max_tries: int = 50):
        for _ in range(max_tries):
            a = int(self.rng.choice(self.top_pool))
            b = int(self.rng.choice(self.top_pool))
            if abs(self.y[a] - self.y[b]) >= self.gap_min:
                return (a, b) if self.y[a] > self.y[b] else (b, a)
        return self.coarse()

    def batch(self, n_pairs: int, fine_frac: float):
        pairs, kinds = [], []
        for _ in range(n_pairs):
            if self.rng.random() < fine_frac:
                pairs.append(self.fine()); kinds.append(1)
            else:
                pairs.append(self.coarse()); kinds.append(0)
        return pairs, kinds


def set_scores_batch(scores: torch.Tensor, mask_mat: torch.Tensor, reduce: str) -> torch.Tensor:
    s = scores.view(1, -1)
    if reduce == "sum":
        return (s * mask_mat).sum(dim=1)
    denom = mask_mat.sum(dim=1).clamp(min=1.0)
    return (s * mask_mat).sum(dim=1) / denom


def main():
    start = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", default=None,
                    help="dataset .pt path (default: <data>/dataset_policy.pt)")
    ap.add_argument("--holdout", default=None,
                    help="holdout_idx.json path; these indices are left out of training")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--model-out", type=str, default=None,
                    help="default: <data>/policy_model_v2.pt (policy_model.pt is not overwritten)")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--label-mode", choices=["auto", "cost", "benefit"], default="auto")
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--use-layernorm", action="store_true")
    ap.add_argument("--reduce", choices=["mean", "sum"], default="mean")
    ap.add_argument("--fine-frac", type=float, default=0.7, help="share of fine pairs")
    ap.add_argument("--q-top", type=float, default=0.5, help="top region fine pairs are drawn from (0.5 = top half)")
    ap.add_argument("--gap-min", type=float, default=0.5, help="minimum y difference of a fine pair")
    ap.add_argument("--qlo", type=float, default=0.1)
    ap.add_argument("--qhi", type=float, default=0.9)
    ap.add_argument("--margin-fine", type=float, default=0.3)
    ap.add_argument("--margin-coarse", type=float, default=1.0)
    ap.add_argument("--pairs-per-step", type=int, default=32)
    ap.add_argument("--steps-per-epoch", type=int, default=40)
    ap.add_argument("--fine-frac-start", type=float, default=None)
    ap.add_argument("--fine-frac-end", type=float, default=None)
    ap.add_argument("--marginal-pairs", default=None, help="marginal_pairs.json path")
    ap.add_argument("--marg-weight", type=float, default=0.3)
    ap.add_argument("--marg-pairs-per-step", type=int, default=16)
    ap.add_argument("--marg-gap", type=float, default=1.0, help="minimum delta difference of a pair")
    ap.add_argument("--marg-min-support", type=int, default=3)
    args = ap.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    outdir = Path(args.data)
    graph = load_blob(outdir / "graph_policy.pt")
    edge_index = graph["edge_index"]
    x_node = graph["x_node"]
    edge_attr = graph["edge_attr"]
    dataset_path = Path(args.dataset) if args.dataset else (outdir / "dataset_policy.pt")
    samples = load_blob(dataset_path)
    print(f"[train v2] dataset={dataset_path} n={len(samples)}")
    if args.holdout:
        hold = set(json.loads(Path(args.holdout).read_text())["holdout_idx"])
        samples = [s for i, s in enumerate(samples) if i not in hold]
        print(f"[train v2] holdout excluded: {len(hold)} -> train pool n={len(samples)}")

    label_mode = infer_label_mode(outdir, graph, args.label_mode)
    higher_is_better = label_mode == "benefit"
    print(f"[train v2] label_mode={label_mode} | {'higher y is better' if higher_is_better else 'lower y is better'}")
    print(f"[train v2] fine_frac={args.fine_frac} q_top={args.q_top} gap_min={args.gap_min} "
          f"margin_fine={args.margin_fine} margin_coarse={args.margin_coarse}")

    ys_all = np.array([float(s["y"]) for s in samples], dtype=np.float64)
    mask_all = torch.stack([s["mask"].float().view(-1) for s in samples])

    idx = np.random.permutation(len(samples))
    nval = min(max(1, int(args.val_split * len(samples))), len(samples) - 1)
    val_idx, train_idx = idx[:nval], idx[nval:]

    rng_tr = random.Random(args.seed)
    rng_va = random.Random(args.seed + 1)
    samp_tr = PairSampler(ys_all[train_idx], higher_is_better,
                          args.q_top, args.gap_min, args.qlo, args.qhi, rng_tr)
    samp_va = PairSampler(ys_all[val_idx], higher_is_better,
                          args.q_top, args.gap_min, args.qlo, args.qhi, rng_va)

    in_node, in_edge = x_node.size(1), edge_attr.size(1)
    model = PolicyGNN(in_node, in_edge, hid=args.hid, heads=args.heads,
                      dropout=args.dropout, use_layernorm=args.use_layernorm)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("Requested --device cuda but CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    x_node = x_node.to(device); edge_index = edge_index.to(device); edge_attr = edge_attr.to(device)
    mask_tr = mask_all[train_idx].to(device)
    mask_va = mask_all[val_idx].to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def hinge_batch(scores, mask_mat, pairs, kinds, sign):
        ia = torch.tensor([p[0] for p in pairs], device=device)
        ib = torch.tensor([p[1] for p in pairs], device=device)
        ss = set_scores_batch(scores, mask_mat, args.reduce)
        margins = torch.tensor([args.margin_fine if k == 1 else args.margin_coarse for k in kinds],
                               device=device)
        diff = (ss[ia] - ss[ib]) * sign
        return torch.clamp(margins - diff, min=0.0)

    sign = 1.0 if higher_is_better else -1.0
    save_path = Path(args.model_out) if args.model_out else (outdir / "policy_model_v2.pt")
    best_key = -1.0

    marg_idx = marg_delta = None
    if args.marginal_pairs:
        mp = json.loads(Path(args.marginal_pairs).read_text(encoding="utf-8"))
        rows = [e for e in mp["edges"] if e.get("n", 1) >= args.marg_min_support]
        marg_idx = torch.tensor([e["idx"] for e in rows], device=device, dtype=torch.long)
        marg_delta = np.array([e["delta_mean"] for e in rows])
        print(f"[train v2] marginal edges loaded: {len(rows)} (support>={args.marg_min_support})")

    def marginal_loss(scores):
        if marg_idx is None or len(marg_idx) < 2:
            return None
        ia, ib = [], []
        for _ in range(args.marg_pairs_per_step):
            a, b = np.random.randint(len(marg_idx)), np.random.randint(len(marg_idx))
            if marg_delta[a] < marg_delta[b]:
                a, b = b, a
            if marg_delta[a] - marg_delta[b] < args.marg_gap:
                continue
            ia.append(a); ib.append(b)
        if not ia:
            return None
        sa = scores[marg_idx[torch.tensor(ia, device=device)]]
        sb = scores[marg_idx[torch.tensor(ib, device=device)]]
        return torch.clamp(args.margin_fine - (sa - sb) * sign, min=0.0).mean()

    def fine_frac_at(ep):
        if args.fine_frac_start is None or args.fine_frac_end is None:
            return args.fine_frac
        t = (ep - 1) / max(1, args.epochs - 1)
        return args.fine_frac_start + t * (args.fine_frac_end - args.fine_frac_start)

    for ep in range(1, args.epochs + 1):
        model.train()
        ff = fine_frac_at(ep)
        tot, n = 0.0, 0
        for _ in range(args.steps_per_epoch):
            pairs, kinds = samp_tr.batch(args.pairs_per_step, ff)
            scores = model(x_node, edge_index, edge_attr).squeeze(-1)
            loss = hinge_batch(scores, mask_tr, pairs, kinds, sign).mean()
            ml = marginal_loss(scores)
            if ml is not None:
                loss = loss + args.marg_weight * ml
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); n += 1
        tr = tot / max(1, n)

        model.eval()
        with torch.no_grad():
            scores = model(x_node, edge_index, edge_attr).squeeze(-1)
            ss_va = set_scores_batch(scores, mask_va, args.reduce)
            accs = {0: [], 1: []}
            for _ in range(600):
                for kind, drawer in ((0, samp_va.coarse), (1, samp_va.fine)):
                    a, b = drawer()
                    accs[kind].append(bool((ss_va[a] - ss_va[b]) * sign > 0))
            acc_coarse = float(np.mean(accs[0])); acc_fine = float(np.mean(accs[1]))
        key = acc_fine + 0.2 * acc_coarse
        print(f"[{ep:03d}] train {tr:.3f} | val coarse_acc {acc_coarse:.3f} fine_acc {acc_fine:.3f} | ff {ff:.2f}")
        if key > best_key:
            best_key = key
            torch.save({
                "state_dict": model.state_dict(),
                "in_node": in_node, "in_edge": in_edge,
                "hid": args.hid, "heads": args.heads,
                "dropout": args.dropout, "use_layernorm": bool(args.use_layernorm),
                "label_mode": label_mode,
                "score_mode": "benefit" if higher_is_better else "risk",
                "reduce": args.reduce,
                "edge_feature_names": graph.get("edge_feature_names"),
                "node_feature_names": graph.get("node_feature_names"),
                "baseline_cost": graph.get("baseline_cost"),
                "train_version": "v2",
                "pair_sampling": {"fine_frac": args.fine_frac, "q_top": args.q_top,
                                   "gap_min": args.gap_min, "qlo": args.qlo, "qhi": args.qhi,
                                   "margin_fine": args.margin_fine, "margin_coarse": args.margin_coarse},
            }, save_path)
    print("Saved", save_path)
    print(f"Elapsed: {time.perf_counter() - start:.3f} sec")


if __name__ == "__main__":
    main()
