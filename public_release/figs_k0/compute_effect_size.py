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

PENALTY = 1e9
MODES = ["init", "mutation", "both"]


def a12(guided: list[float], baseline: list[float]) -> float:
    wins = sum(
        1.0 if g < b else 0.5 if g == b else 0.0
        for g in guided for b in baseline
    )
    return wins / (len(guided) * len(baseline))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    args = ap.parse_args()

    path = Path(K0_MERGED) / "seed_best_merged.csv"
    seeds: dict[tuple[str, str], list[float]] = defaultdict(list)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                att = float(r["ATT"])
            except (KeyError, ValueError):
                continue
            if att < PENALTY:
                seeds[(r["instance_id"], r["mode"])].append(att)

    instances = sorted({k[0] for k in seeds})
    print(f"{'mode':10s} {'mean':>7s} {'median':>7s} {'>0.5':>7s} "
          f"{'small':>7s} {'medium':>7s} {'large':>7s}")
    for mode in MODES:
        vals = [a12(seeds[(i, mode)], seeds[(i, "none")])
                for i in instances
                if seeds.get((i, mode)) and seeds.get((i, "none"))]
        n = len(vals)
        print(f"{mode:10s} {st.mean(vals):7.3f} {st.median(vals):7.3f} "
              f"{sum(v > 0.5 for v in vals):4d}/{n} "
              f"{sum(v >= 0.56 for v in vals):4d}/{n} "
              f"{sum(v >= 0.64 for v in vals):4d}/{n} "
              f"{sum(v >= 0.71 for v in vals):4d}/{n}")

    print("\nthresholds: 0.56 small, 0.64 medium, 0.71 large; 0.5 = no effect")


if __name__ == "__main__":
    main()
