from __future__ import annotations
import os
K0_MERGED = os.environ.get('K0_MERGED', './merged_analysis_38_k0')
K0_RUNSFX = os.environ.get('K0_RUNSFX', '_pm01k0')
K0_OUT = os.environ.get('K0_OUT', 'manuscript_numbers_k0')
K0_REEVAL = os.environ.get('K0_REEVAL', 'reeval_39_k0')


import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PENALTY = 1e9
SUFFIXES = ["", "_old", "_v2", "_main"]


def policy_path(root: Path, inst: str) -> Path | None:
    for s in SUFFIXES:
        p = root / f"output_policy_{inst}{s}" / "dataset_policy.pt"
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    ap.add_argument("--mode", default="both",
                    help="GA mode whose visited distribution defines deployment")
    ap.add_argument("--out", default="distribution_shift.csv")
    args = ap.parse_args()
    root = Path(args.root)

    visited: dict[str, list[int]] = defaultdict(list)
    agg = Path(K0_MERGED) / "aggregate_merged.csv"
    with open(agg, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["mode"] != args.mode:
                continue
            try:
                att, k = float(r["ATT"]), int(r["k_closed"])
            except (KeyError, ValueError):
                continue
            if att < PENALTY:
                visited[r["instance_id"]].append(k)

    rows = []
    for inst in sorted(visited):
        path = policy_path(root, inst)
        if path is None:
            continue
        samples = torch.load(path, map_location="cpu", weights_only=False)
        mask = torch.stack([s["mask"].float().view(-1) for s in samples])
        k_off = mask.sum(1)

        k_ga = np.asarray(visited[inst])
        lo, hi = int(np.percentile(k_ga, 5)), int(np.percentile(k_ga, 95))
        sel = (k_off >= lo) & (k_off <= hi)

        per_edge_all = mask.sum(0)
        per_edge_sel = mask[sel].sum(0) if int(sel.sum()) else torch.zeros_like(per_edge_all)
        rows.append({
            "instance_id": inst,
            "network": inst[0],
            "offline_mean_k": round(float(k_off.mean()), 2),
            "ga_mean_k": round(float(k_ga.mean()), 2),
            "ga_k_p5": lo,
            "ga_k_p95": hi,
            "offline_in_range": int(sel.sum()),
            "offline_total": len(samples),
            "share_in_range": round(float(sel.sum()) / len(samples), 3),
            "density_nominal": round(float(per_edge_all.mean()), 2),
            "density_effective": round(float(per_edge_sel.mean()), 2),
            "never_closed_nominal": int((per_edge_all == 0).sum()),
            "never_closed_effective": int((per_edge_sel == 0).sum()),
            "edges": mask.shape[1],
        })

    out = Path(__file__).resolve().parent / args.out
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} instances, GA mode = {args.mode})")

    share = [r["share_in_range"] for r in rows]
    ratio = [r["density_effective"] / r["density_nominal"] for r in rows
             if r["density_nominal"]]
    print(f"  offline share inside the GA-visited k range: "
          f"mean {st.mean(share):.3f}, min {min(share):.3f}, max {max(share):.3f}")
    print(f"  effective / nominal density: mean {st.mean(ratio):.3f}")
    print(f"  offline mean k {st.mean([r['offline_mean_k'] for r in rows]):.2f} "
          f"vs GA mean k {st.mean([r['ga_mean_k'] for r in rows]):.2f}")


if __name__ == "__main__":
    main()
