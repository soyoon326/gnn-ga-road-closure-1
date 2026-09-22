from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ga_edge_closure_gnn_policy as GA  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Surrogate-fitness GA + SUMO final verification")
    ap.add_argument("--data", required=True, help="folder containing graph_policy.pt")
    ap.add_argument("--model", required=True, help="surrogate_att.pt")
    ap.add_argument("--sumocfg", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--sumo-bin", default=None)
    ap.add_argument("--kmin", type=int, default=1)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--pc", type=float, default=0.9)
    ap.add_argument("--pm", type=float, default=0.1)
    ap.add_argument("--verify-top", type=int, default=10,
                    help="number of distinct top-predicted solutions verified in SUMO after the search")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--protect-str", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--begin", type=int, default=0)
    ap.add_argument("--end", type=int, default=86400)
    ap.add_argument("--reroute-period", type=int, default=30)
    ap.add_argument("--time-to-teleport", type=int, default=-1)
    ap.add_argument("--time-to-teleport-highways", type=int, default=-1)
    ap.add_argument("--demand-size", type=int, default=None)
    ap.add_argument("--no-auto-protect-dest", action="store_true")
    args = ap.parse_args()

    start = time.perf_counter()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    import torch
    from policy_gnn_model import PolicyGNN
    from train_policy_gnn_v2 import load_blob

    datadir = Path(args.data)
    graph = load_blob(datadir / "graph_policy.pt")
    edge_index = graph["edge_index"].long()
    x_node = graph["x_node"].float()
    edge_attr = graph["edge_attr"].float()
    edge_map: dict[str, int] = dict(graph["edge_map"])
    E = edge_attr.size(0) if edge_attr.dim() == 2 else len(edge_map)

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    model = PolicyGNN(ck["in_node"], ck["in_edge"], hid=ck["hid"], heads=ck["heads"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    higher_is_better = bool(ck.get("higher_is_better", True))
    y_mean, y_std = float(ck.get("y_mean", 0.0)), float(ck.get("y_std", 1.0))
    print(f"[surrogate-ga] model={args.model} higher_is_better={higher_is_better}")

    sumo_bin = GA.resolve_sumo_bin(args.sumo_bin)
    protect = set(args.protect_str.split()) if args.protect_str else set()
    if not args.no_auto_protect_dest:
        eps = GA._extract_route_endpoints(args.sumocfg)
        if eps:
            protect.update(eps)
    limit = None if args.limit == 0 else args.limit
    candidates, upstream_map, _net = GA.load_candidates(args.net, protect, limit)
    candidates = [c for c in candidates if c in edge_map]
    print(f"candidate edges: {len(candidates)} | protected: {sorted(protect)}")

    cand_rows = torch.tensor([edge_map[c] for c in candidates], dtype=torch.long)

    @torch.no_grad()
    def fitness(bits: list[int]) -> float:
        mask = torch.zeros(E)
        rows = cand_rows[torch.tensor(bits, dtype=torch.bool)]
        if len(rows) == 0:
            return -1e9
        mask[rows] = 1.0
        ea = torch.cat([edge_attr, mask.view(-1, 1)], dim=1)
        scores = model(x_node, edge_index, ea).squeeze(-1)
        pred = float((scores * mask).sum() / mask.sum())
        pred = pred * y_std + y_mean
        return pred if higher_is_better else -pred

    def random_bits() -> list[int]:
        k = rng.randint(args.kmin, args.kmax)
        bits = [0] * len(candidates)
        for i in rng.sample(range(len(candidates)), k):
            bits[i] = 1
        return bits

    def clamp(bits: list[int]) -> list[int]:
        return GA._clamp_bits(list(bits), args.kmin, args.kmax)

    pop = [random_bits() for _ in range(args.pop)]
    fits = [fitness(b) for b in pop]
    seen: dict[tuple, float] = {}
    for b, f in zip(pop, fits):
        seen[tuple(b)] = f

    def tournament() -> list[int]:
        i, j = rng.randrange(args.pop), rng.randrange(args.pop)
        return pop[i] if fits[i] >= fits[j] else pop[j]

    gen_log = Path(args.outdir); gen_log.mkdir(parents=True, exist_ok=True)
    with (gen_log / "search_log.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["gen", "best_pred", "mean_pred"])

    for g in range(1, args.gen + 1):
        new_pop: list[list[int]] = []
        elite = pop[int(np.argmax(fits))]
        new_pop.append(list(elite))
        while len(new_pop) < args.pop:
            p1, p2 = tournament(), tournament()
            if rng.random() < args.pc:
                c1 = [(a if rng.random() < 0.5 else b) for a, b in zip(p1, p2)]
            else:
                c1 = list(p1)
            c1 = [(1 - b if rng.random() < args.pm else b) for b in c1]
            new_pop.append(clamp(c1))
        pop = new_pop
        fits = []
        for b in pop:
            key = tuple(b)
            if key not in seen:
                seen[key] = fitness(b)
            fits.append(seen[key])
        if g % 20 == 0 or g == args.gen:
            print(f"[gen {g:04d}] best_pred={max(fits):.3f} mean_pred={np.mean(fits):.3f} "
                  f"unique={len(seen)}")
        with (gen_log / "search_log.csv").open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([g, f"{max(fits):.6f}", f"{float(np.mean(fits)):.6f}"])

    ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[:args.verify_top]
    print(f"[verify] verifying the top {len(top)} predicted solutions in SUMO")

    outdir = Path(args.outdir)
    results = []
    with (outdir / "verify.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["rank", "pred_benefit", "att", "k", "closed_edges", "run_dir"])
    for rank, (bits_key, pred) in enumerate(top):
        closed = [candidates[i] for i, b in enumerate(bits_key) if b]
        run_dir = outdir / f"verify_{rank:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        add_path = run_dir / "closures.add.xml"
        GA.write_closures_additional(add_path, closed, upstream_map,
                                     begin=args.begin, end=args.end)
        att = GA.run_single_sumo_and_score(
            sumo_bin=sumo_bin, sumocfg=args.sumocfg, additional=add_path,
            outdir=run_dir, reroute_period=args.reroute_period,
            time_to_teleport=args.time_to_teleport,
            time_to_teleport_highways=args.time_to_teleport_highways,
            emit_tripinfo=False, demand_size=args.demand_size)
        results.append((att, pred, closed, str(run_dir)))
        print(f"  rank {rank}: pred={pred:.3f} -> ATT={att:.3f} (k={len(closed)})")
        with (outdir / "verify.csv").open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([rank, f"{pred:.6f}", f"{att:.6f}", len(closed),
                                    ",".join(closed), run_dir])

    results.sort(key=lambda r: r[0])
    best_att, best_pred, best_closed, _ = results[0]
    with (outdir / "best.txt").open("w", encoding="utf-8") as f:
        f.write(f"best_score={best_att:.6f}\n")
        f.write("closed_edges=" + " ".join(best_closed) + "\n")
    print("\n=== Surrogate-GA final result ===")
    print(f"best verified ATT: {best_att:.3f} (predicted benefit {best_pred:.3f})")
    print(f"edges to close: {best_closed}")
    print(f"SUMO calls: {len(top)} (the search uses the surrogate only; {len(seen)} distinct solutions scored)")
    print(f"Elapsed: {time.perf_counter() - start:.3f} sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
