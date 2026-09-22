from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "py"))
import ga_edge_closure_gnn_policy as GA  # noqa: E402
from policy_gnn_model import PolicyGNN  # noqa: E402
from train_policy_gnn_v2 import load_blob  # noqa: E402

SUMO = "sumo"

INSTANCES = {
    "A": dict(net="netA/network_v2.net.xml", sumocfg="netA/sim_3od_x1.0.sumocfg",
              protect="E28 E40", kmax=4, demand=1000,
              data="output_policy_A_k4_3od_x1.0", orig_arm=240.89, best_arm=226.42),
    "B": dict(net="netB/netB.net.xml", sumocfg="netB/sim_3od_x1.0.sumocfg",
              protect="E35 -E41 E38 E44 E20 E12", kmax=9, demand=2000,
              data="output_policy_B_k9_3od_x1.0", orig_arm=502.29, best_arm=475.98),
    "J": dict(net="netJ/Netzmodell2.net.xml", sumocfg="netJ/wildau_6od_x1.0.sumocfg",
              protect=("4935299#0 111677671#1 255274250 -27149243 27149243 "
                       "-255274250 -37548821#0 311298682#1 -111677671#1 "
                       "37548821#0 876057378#6"),
              kmax=9, demand=1500,
              data="output_policy_J_k9_6od_x1.0", orig_arm=584.58, best_arm=345.24),
}

_CTX = {}


class KHead(nn.Module):
    def __init__(self, hid: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, k):
        kk = k.view(1, 1) / 10.0
        return self.net(torch.cat([kk, kk ** 2], dim=1)).squeeze()


def _init(ctx):
    _CTX.update(ctx)


def _verify(task):
    tag, closed = task
    run_dir = Path(_CTX["outdir"]) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    add = run_dir / "closures.add.xml"
    GA.write_closures_additional(add, closed, _CTX["upstream_map"], begin=0, end=86400)
    att = GA.run_single_sumo_and_score(
        sumo_bin=SUMO, sumocfg=_CTX["sumocfg"], additional=add, outdir=run_dir,
        reroute_period=30, time_to_teleport=-1, time_to_teleport_highways=-1,
        emit_tripinfo=False, demand_size=_CTX["demand"])
    return tag, att, len(closed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", required=True, choices=list(INSTANCES))
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--pc", type=float, default=0.9)
    ap.add_argument("--pm", type=float, default=0.1)
    ap.add_argument("--kmin", type=int, default=0)
    ap.add_argument("--verify-top", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    inst = INSTANCES[args.inst]
    ddir = ROOT / inst["data"]
    graph = load_blob(ddir / "graph_policy.pt")
    ei = graph["edge_index"].long()
    xn = graph["x_node"].float()
    ea0 = graph["edge_attr"].float()
    edge_map = dict(graph["edge_map"])
    E = ea0.size(0)

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    model = PolicyGNN(ck["in_node"], ck["in_edge"], hid=ck["hid"], heads=ck["heads"])
    model.load_state_dict(ck["state_dict"]); model.eval()
    readout = ck.get("readout", "mean")
    khead = None
    if ck.get("khead"):
        khead = KHead(); khead.load_state_dict(ck["khead"]); khead.eval()
    y_mean, y_std = float(ck.get("y_mean", 0.0)), float(ck.get("y_std", 1.0))
    higher_is_better = bool(ck.get("higher_is_better", False))

    protect = set(inst["protect"].split())
    eps = GA._extract_route_endpoints(str(ROOT / inst["sumocfg"]))
    if eps:
        protect.update(eps)
    candidates, upstream_map, _n = GA.load_candidates(str(ROOT / inst["net"]), protect, None)
    candidates = [c for c in candidates if c in edge_map]
    cand_rows = torch.tensor([edge_map[c] for c in candidates], dtype=torch.long)
    kmax = inst["kmax"]
    print(f"[{args.inst}] readout={readout} candidates={len(candidates)} kmax={kmax}",
          flush=True)

    @torch.no_grad()
    def fitness(bits):
        rows = cand_rows[torch.tensor(bits, dtype=torch.bool)]
        if len(rows) == 0:
            return -1e9
        mask = torch.zeros(E); mask[rows] = 1.0
        sc = model(xn, ei, torch.cat([ea0, mask.view(-1, 1)], 1)).squeeze(-1)
        s = (sc * mask).sum()
        pred = (s / mask.sum() if readout == "mean"
                else s if readout == "sum" else s + khead(mask.sum()))
        pred = float(pred) * y_std + y_mean
        return pred if higher_is_better else -pred

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ctx = dict(outdir=str(outdir), upstream_map=upstream_map,
               sumocfg=str(ROOT / inst["sumocfg"]), demand=inst["demand"])

    per_seed_best, all_k, rows_out = [], collections.Counter(), []
    for seed in range(args.seeds):
        rng = random.Random(seed)

        def random_bits():
            k = rng.randint(args.kmin, kmax)
            b = [0] * len(candidates)
            for i in rng.sample(range(len(candidates)), k):
                b[i] = 1
            return b

        pop = [random_bits() for _ in range(args.pop)]
        fits = [fitness(b) for b in pop]
        seen = {tuple(b): f for b, f in zip(pop, fits)}

        def tournament():
            i, j = rng.randrange(args.pop), rng.randrange(args.pop)
            return pop[i] if fits[i] >= fits[j] else pop[j]

        for _g in range(args.gen):
            new_pop = [list(pop[int(np.argmax(fits))])]
            while len(new_pop) < args.pop:
                p1, p2 = tournament(), tournament()
                c1 = ([(a if rng.random() < 0.5 else b) for a, b in zip(p1, p2)]
                      if rng.random() < args.pc else list(p1))
                c1 = [(1 - b if rng.random() < args.pm else b) for b in c1]
                new_pop.append(GA._clamp_bits(c1, args.kmin, kmax))
            pop = new_pop
            fits = []
            for b in pop:
                key = tuple(b)
                if key not in seen:
                    seen[key] = fitness(b)
                fits.append(seen[key])

        top = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)[:args.verify_top]
        tasks = [(f"s{seed}_r{r:02d}",
                  [candidates[i] for i, b in enumerate(bits) if b])
                 for r, (bits, _p) in enumerate(top)]
        with Pool(processes=args.jobs, initializer=_init, initargs=(ctx,)) as pool:
            res = list(pool.imap_unordered(_verify, tasks))
        res.sort(key=lambda t: t[1])
        best = res[0][1]
        per_seed_best.append(best)
        all_k.update(k for _t, _a, k in res)
        rows_out += [(seed, t, a, k) for t, a, k in res]
        print(f"  seed{seed}: best ATT {best:.2f}  (k of verified: "
              f"{sorted(set(k for _t,_a,k in res))})", flush=True)

    with (outdir / "arm_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["seed", "tag", "att", "k"]); w.writerows(rows_out)

    v = np.array(per_seed_best)
    ok = v < 1e11
    print("\nRESULT " + json.dumps(dict(
        inst=args.inst, model=Path(args.model).name, readout=readout,
        n_seeds=int(ok.sum()),
        mean=round(float(v[ok].mean()), 2), median=round(float(np.median(v[ok])), 2),
        sd=round(float(v[ok].std(ddof=1)), 2),
        min=round(float(v[ok].min()), 2), max=round(float(v[ok].max()), 2),
        original_surrogate_arm=inst["orig_arm"], best_arm_costeq=inst["best_arm"],
        k_of_verified={str(k): n for k, n in sorted(all_k.items())})), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
