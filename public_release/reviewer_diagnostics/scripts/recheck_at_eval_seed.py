from __future__ import annotations

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(ROOT))
import ga_edge_closure_gnn_policy as GA  # noqa: E402

SUMO = "sumo"

INSTANCES = {
    "A": dict(sumocfg="netA/sim_3od_x1.0.sumocfg", net="netA/network_v2.net.xml",
              demand=1000, protect="E28 E40", data="output_policy_A_k4_3od_x1.0"),
    "B": dict(sumocfg="netB/sim_3od_x1.0.sumocfg", net="netB/netB.net.xml",
              demand=2000, protect="E35 -E41 E38 E44 E20 E12",
              data="output_policy_B_k9_3od_x1.0"),
    "J": dict(sumocfg="netJ/wildau_6od_x1.0.sumocfg", net="netJ/Netzmodell2.net.xml",
              demand=1500,
              protect=("4935299#0 111677671#1 255274250 -27149243 27149243 "
                       "-255274250 -37548821#0 311298682#1 -111677671#1 "
                       "37548821#0 876057378#6"),
              data="output_policy_J_k9_6od_x1.0"),
}

_CTX = {}


def _init(ctx):
    _CTX.update(ctx)


def _work(task):
    idx, closed = task
    run_dir = Path(_CTX["outdir"]) / f"s{idx:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    add_path = run_dir / "closures.add.xml"
    GA.write_closures_additional(add_path, closed, _CTX["upstream_map"],
                                 begin=0, end=86400)
    t0 = time.perf_counter()
    att = GA.run_single_sumo_and_score(
        sumo_bin=_CTX["sumo_bin"], sumocfg=_CTX["sumocfg"], additional=add_path,
        outdir=run_dir, reroute_period=30, time_to_teleport=-1,
        time_to_teleport_highways=-1, emit_tripinfo=False,
        demand_size=_CTX["demand_size"])
    return idx, att, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", required=True, choices=list(INSTANCES))
    ap.add_argument("--select", default="beats",
                    choices=["beats", "random", "head"],
                    help="beats: dataset label better than --threshold; "
                         "random: random subset; head: first N")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inst = INSTANCES[args.inst]
    ddir = ROOT / inst["data"]
    samples = torch.load(ddir / "dataset_policy.pt", map_location="cpu",
                         weights_only=False)
    ys = np.array([float(s["y"]) for s in samples])
    ks = np.array([int(s["mask"].sum()) for s in samples])

    if args.select == "beats":
        sel = np.nonzero((ks > 0) & (ys < args.threshold))[0]
    elif args.select == "random":
        rng = np.random.default_rng(args.seed)
        pool_ = np.nonzero(ks > 0)[0]
        sel = rng.choice(pool_, size=min(args.n, len(pool_)), replace=False)
        sel.sort()
    else:
        sel = np.nonzero(ks > 0)[0][:args.n]
    print(f"[{args.inst}] selected {len(sel)} samples "
          f"(label range {ys[sel].min():.2f}..{ys[sel].max():.2f})", flush=True)

    protect = set(inst["protect"].split())
    eps = GA._extract_route_endpoints(str(ROOT / inst["sumocfg"]))
    if eps:
        protect.update(eps)
    _cands, upstream_map, _net = GA.load_candidates(str(ROOT / inst["net"]),
                                                    protect, None)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    ctx = dict(outdir=str(outdir), upstream_map=upstream_map, sumo_bin=SUMO,
               sumocfg=str(ROOT / inst["sumocfg"]), demand_size=inst["demand"])

    tasks = [(int(i), list(samples[int(i)]["closed_edges"])) for i in sel]
    t0 = time.perf_counter()
    rows = []
    with Pool(processes=args.jobs, initializer=_init, initargs=(ctx,)) as pool:
        for n_done, (idx, att, dt) in enumerate(pool.imap_unordered(_work, tasks), 1):
            rows.append((idx, int(ks[idx]), float(ys[idx]), float(att)))
            if n_done % 10 == 0 or n_done == len(tasks):
                el = time.perf_counter() - t0
                print(f"  {n_done}/{len(tasks)} elapsed {el/60:.1f} min "
                      f"(eta {el/n_done*(len(tasks)-n_done)/60:.1f} min)", flush=True)

    rows.sort()
    with (outdir / "recheck.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "k", "y_dataset_randseed", "att_eval_seed"])
        w.writerows(rows)

    a = np.array([[r[2], r[3]] for r in rows])
    ok = a[:, 1] < 1e11
    print(f"\n[{args.inst}] n={len(a)}  penalised(1e12)={int((~ok).sum())}")
    if ok.sum() > 2:
        from scipy.stats import spearmanr, pearsonr
        print(f"  dataset label : mean {a[ok,0].mean():.2f} sd {a[ok,0].std(ddof=1):.2f}")
        print(f"  eval-seed ATT : mean {a[ok,1].mean():.2f} sd {a[ok,1].std(ddof=1):.2f}")
        print(f"  spearman(dataset, eval) = {spearmanr(a[ok,0], a[ok,1]).statistic:.3f}")
        print(f"  pearson (dataset, eval) = {pearsonr(a[ok,0], a[ok,1])[0]:.3f}")
        print(f"  mean |diff| = {np.abs(a[ok,0]-a[ok,1]).mean():.2f}")
    print(f"  wrote {outdir/'recheck.csv'}  ({(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
