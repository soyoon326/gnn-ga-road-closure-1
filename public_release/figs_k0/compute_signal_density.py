from __future__ import annotations
import os
K0_MERGED = os.environ.get('K0_MERGED', './merged_analysis_38_k0')
K0_RUNSFX = os.environ.get('K0_RUNSFX', '_pm01k0')
K0_OUT = os.environ.get('K0_OUT', 'manuscript_numbers_k0')
K0_REEVAL = os.environ.get('K0_REEVAL', 'reeval_39_k0')


import argparse
import csv
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    ap.add_argument("--out", default="signal_density.csv")
    args = ap.parse_args()
    root = Path(args.root)

    att = list(csv.DictReader(
        open(Path(K0_MERGED) / "att_summary_by_mode_merged.csv",
             encoding="utf-8-sig")))
    instances = sorted({r["instance_id"] for r in att})

    gain: dict[tuple[str, str], float] = {}
    for r in att:
        v = r.get("mean_reduction_vs_none_%", "")
        if r["mode"] != "none" and v not in ("", "NA"):
            gain[(r["instance_id"], r["mode"])] = float(v)

    suffixes = ["", "_old", "_v2", "_main"]

    rows, missing = [], []
    for inst in instances:
        path = next(
            (p for s in suffixes
             if (p := root / f"output_policy_{inst}{s}" / "dataset_policy.pt").exists()),
            None,
        )
        if path is None:
            missing.append(inst)
            continue
        if path.parent.name != f"output_policy_{inst}":
            print(f"  note: {inst} resolved to {path.parent.name}")
        samples = torch.load(path, map_location="cpu", weights_only=False)
        mask = torch.stack([s["mask"].float().view(-1) for s in samples])
        per_edge = mask.sum(0)
        rows.append({
            "instance_id": inst,
            "network": inst[0],
            "n_samples": len(samples),
            "edges": mask.shape[1],
            "mean_k": round(mask.sum(1).mean().item(), 3),
            "obs_per_edge": round(per_edge.mean().item(), 3),
            "never_closed": int((per_edge == 0).sum().item()),
            "gain_init": gain.get((inst, "init"), ""),
            "gain_mutation": gain.get((inst, "mutation"), ""),
            "gain_both": gain.get((inst, "both"), ""),
        })

    out = Path(__file__).resolve().parent / args.out
    if not rows:
        raise SystemExit(
            f"no policy datasets found under {root.resolve()} -- "
            f"refusing to overwrite {out}. Checked {len(instances)} instances; "
            f"expected directories named output_policy_<instance_id>."
        )
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} instances)")
    if missing:
        print(f"  no policy dataset for: {missing}")


if __name__ == "__main__":
    main()
