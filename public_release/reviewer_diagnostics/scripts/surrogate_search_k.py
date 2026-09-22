from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "py"))
import ga_edge_closure_gnn_policy as GA  # noqa: E402
from policy_gnn_model import PolicyGNN  # noqa: E402
from train_policy_gnn_v2 import load_blob  # noqa: E402

INSTANCES = {
    "A": dict(net="netA/network_v2.net.xml", sumocfg="netA/sim_3od_x1.0.sumocfg",
              protect="E28 E40", kmax=4, data="output_policy_A_k4_3od_x1.0"),
    "B": dict(net="netB/netB.net.xml", sumocfg="netB/sim_3od_x1.0.sumocfg",
              protect="E35 -E41 E38 E44 E20 E12", kmax=9,
              data="output_policy_B_k9_3od_x1.0"),
    "J": dict(net="netJ/Netzmodell2.net.xml", sumocfg="netJ/wildau_6od_x1.0.sumocfg",
              protect=("4935299#0 111677671#1 255274250 -27149243 27149243 "
                       "-255274250 -37548821#0 311298682#1 -111677671#1 "
                       "37548821#0 876057378#6"),
              kmax=9, data="output_policy_J_k9_6od_x1.0"),
}


class KHead(nn.Module):
    def __init__(self, hid: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, k):
        kk = k.view(1, 1) / 10.0
        return self.net(torch.cat([kk, kk ** 2], dim=1)).squeeze()


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
    args = ap.parse_args()

    inst = INSTANCES[args.inst]
    ddir = ROOT / inst["data"]
    graph = load_blob(ddir / "graph_policy.pt")
    edge_index = graph["edge_index"].long()
    x_node = graph["x_node"].float()
    edge_attr = graph["edge_attr"].float()
    edge_map = dict(graph["edge_map"])
    E = edge_attr.size(0)

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    model = PolicyGNN(ck["in_node"], ck["in_edge"], hid=ck["hid"], heads=ck["heads"])
    model.load_state_dict(ck["state_dict"]); model.eval()
    readout = ck.get("readout", "mean")
    khead = None
    if ck.get("khead"):
        khead = KHead(); khead.load_state_dict(ck["khead"]); khead.eval()
    higher_is_better = bool(ck.get("higher_is_better", True))

    protect = set(inst["protect"].split())
    eps = GA._extract_route_endpoints(str(ROOT / inst["sumocfg"]))
    if eps:
        protect.update(eps)
    candidates, _um, _net = GA.load_candidates(str(ROOT / inst["net"]), protect, None)
    candidates = [c for c in candidates if c in edge_map]
    cand_rows = torch.tensor([edge_map[c] for c in candidates], dtype=torch.long)
    kmax = inst["kmax"]
    print(f"[{args.inst}] readout={readout} candidates={len(candidates)} kmax={kmax}")

    @torch.no_grad()
    def fitness(bits):
        rows = cand_rows[torch.tensor(bits, dtype=torch.bool)]
        if len(rows) == 0:
            return -1e9
        mask = torch.zeros(E); mask[rows] = 1.0
        ea = torch.cat([edge_attr, mask.view(-1, 1)], dim=1)
        scores = model(x_node, edge_index, ea).squeeze(-1)
        s = (scores * mask).sum()
        if readout == "mean":
            pred = s / mask.sum()
        elif readout == "sum":
            pred = s
        else:
            pred = s + khead(mask.sum())
        pred = float(pred) * float(ck.get("y_std", 1.0)) + float(ck.get("y_mean", 0.0))
        return pred if higher_is_better else -pred

    all_top_k = collections.Counter()
    per_seed = []
    for seed in range(args.seeds):
        rng = random.Random(seed)

        def random_bits():
            k = rng.randint(args.kmin, kmax)
            bits = [0] * len(candidates)
            for i in rng.sample(range(len(candidates)), k):
                bits[i] = 1
            return bits

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

        ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[:args.verify_top]
        ks = [sum(bits) for bits, _ in top]
        all_top_k.update(ks)
        per_seed.append(ks)
        print(f"  seed{seed}: top{args.verify_top} k = {ks}  unique_evaluated={len(seen)}",
              flush=True)

    print("\nRESULT " + json.dumps(dict(
        inst=args.inst, readout=readout, model=Path(args.model).name,
        k_distribution={str(k): v for k, v in sorted(all_top_k.items())},
        frac_at_kmax=round(all_top_k[kmax] / sum(all_top_k.values()), 3),
        mean_k=round(float(np.mean([k for ks in per_seed for k in ks])), 2))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
