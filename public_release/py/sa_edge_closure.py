from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ga_edge_closure_gnn_policy as GA  # noqa: E402


def neighbor(closed: list[str], candidates: list[str], kmin: int, kmax: int,
             rng: random.Random) -> list[str]:
    closed = list(closed)
    pool = [c for c in candidates if c not in closed]
    ops = []
    if closed and pool:
        ops += ["swap"] * 6
    if pool and len(closed) < kmax:
        ops += ["add"] * 2
    if len(closed) > kmin:
        ops += ["remove"] * 2
    if not ops:
        return sorted(closed)
    op = rng.choice(ops)
    if op == "swap":
        closed[rng.randrange(len(closed))] = rng.choice(pool)
    elif op == "add":
        closed.append(rng.choice(pool))
    else:
        closed.pop(rng.randrange(len(closed)))
    return sorted(set(closed))


def main() -> int:
    ap = argparse.ArgumentParser(description="Simulated Annealing for SUMO edge closure")
    ap.add_argument("--sumocfg", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--sumo-bin", default=None)
    ap.add_argument("--kmin", type=int, default=1)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--budget", type=int, default=360, help="budget of SUMO evaluations")
    ap.add_argument("--t0", type=float, default=5.0, help="initial temperature (ATT units)")
    ap.add_argument("--t1", type=float, default=0.2, help="final temperature")
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
    sumo_bin = GA.resolve_sumo_bin(args.sumo_bin)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    protect = set(args.protect_str.split()) if args.protect_str else set()
    if not args.no_auto_protect_dest:
        eps = GA._extract_route_endpoints(args.sumocfg)
        if eps:
            protect.update(eps)
            print(f"added protected edges (origins/destinations): {sorted(eps)}")
    limit = None if args.limit == 0 else args.limit
    candidates, upstream_map, _net = GA.load_candidates(args.net, protect, limit)
    print(f"candidate edges: {len(candidates)} | protected: {sorted(protect)}")

    summary = outdir / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["eval", "k", "score", "accepted", "best_score",
                                "temp", "closed_edges", "run_dir"])

    cache: dict[str, float] = {}

    def evaluate(closed: list[str], eval_no: int) -> tuple[float, str]:
        key = "|".join(sorted(closed))
        if key in cache:
            return cache[key], "(cache)"
        run_dir = outdir / f"e{eval_no:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        add_path = run_dir / "closures.add.xml"
        GA.write_closures_additional(add_path, closed, upstream_map,
                                     begin=args.begin, end=args.end)
        score = GA.run_single_sumo_and_score(
            sumo_bin=sumo_bin, sumocfg=args.sumocfg, additional=add_path,
            outdir=run_dir, reroute_period=args.reroute_period,
            time_to_teleport=args.time_to_teleport,
            time_to_teleport_highways=args.time_to_teleport_highways,
            emit_tripinfo=False, demand_size=args.demand_size)
        cache[key] = score
        return score, str(run_dir)

    k0 = rng.randint(args.kmin, args.kmax)
    cur = sorted(rng.sample(candidates, k0))
    used = 0
    cur_score, run_dir = evaluate(cur, used)
    used += 1
    best, best_score = list(cur), cur_score
    print(f"[SA] init k={len(cur)} score={cur_score:.3f}")

    alpha = (args.t1 / args.t0) ** (1.0 / max(1, args.budget - 1))
    temp = args.t0
    with summary.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([0, len(cur), f"{cur_score:.6f}", 1,
                                f"{best_score:.6f}", f"{temp:.4f}",
                                ",".join(cur), run_dir])

    stall = 0
    max_stall = max(200, 20 * len(candidates))
    max_total_iters = args.budget * 2000
    total_iters = 0
    while used < args.budget:
        total_iters += 1
        if total_iters > max_total_iters:
            print(f"[SA] safety stop: {total_iters} iterations without reaching "
                  f"budget ({used}/{args.budget}); returning current best")
            break
        if stall >= max_stall:
            k = rng.randint(args.kmin, args.kmax)
            cand = sorted(rng.sample(candidates, k)) if k else []
            stall = 0
        else:
            cand = neighbor(cur, candidates, args.kmin, args.kmax, rng)
        if cand == cur:
            temp *= alpha
            continue
        was_cached = "|".join(sorted(cand)) in cache
        score, run_dir = evaluate(cand, used)
        if was_cached:
            stall += 1
        else:
            used += 1
            stall = 0
        delta = score - cur_score
        accept = delta <= 0 or rng.random() < math.exp(-delta / max(1e-9, temp))
        if accept:
            cur, cur_score = cand, score
            if score < best_score:
                best, best_score = list(cand), score
                print(f"[SA] eval {used}: new best {best_score:.3f} (k={len(best)})")
        if not was_cached:
            with summary.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([used, len(cand), f"{score:.6f}", int(accept),
                                        f"{best_score:.6f}", f"{temp:.4f}",
                                        ",".join(cand), run_dir])
        temp *= alpha

    with (outdir / "best.txt").open("w", encoding="utf-8") as f:
        f.write(f"best_score={best_score:.6f}\n")
        f.write("closed_edges=" + " ".join(best) + "\n")
    print("\n=== SA final result ===")
    print(f"best score: {best_score:.3f}")
    print(f"edges to close: {best}")
    print(f"SUMO evaluations: {used} (cache {len(cache)} distinct)")
    print(f"Elapsed: {time.perf_counter() - start:.3f} sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
