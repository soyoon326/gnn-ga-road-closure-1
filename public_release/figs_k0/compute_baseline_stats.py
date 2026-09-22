from __future__ import annotations
import os
K0_MERGED = os.environ.get('K0_MERGED', './merged_analysis_38_k0')
K0_RUNSFX = os.environ.get('K0_RUNSFX', '_pm01k0')
K0_OUT = os.environ.get('K0_OUT', 'manuscript_numbers_k0')
K0_REEVAL = os.environ.get('K0_REEVAL', 'reeval_39_k0')


import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from scipy.stats import mannwhitneyu

PENALTY = 1e9
REFERENCE = "both"
INSTANCES = ["A_k4_3od_x1.0", "B_k9_3od_x1.0", "J_k9_6od_x1.0"]
OTHERS = ["none", "init", "mutation", "surrogate", "heuristic", "sa"]
LABEL = {
    "none": "none (GA only)", "init": "init", "mutation": "mutation",
    "surrogate": "surrogate fitness", "heuristic": "structural heuristic",
    "sa": "simulated annealing",
}


def a12(x: list[float], y: list[float]) -> float:
    w = sum(1.0 if xi < yi else 0.5 if xi == yi else 0.0 for xi in x for yi in y)
    return w / (len(x) * len(y))


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def load(root: Path) -> dict[tuple[str, str], list[float]]:
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    with open(Path(K0_MERGED) / "seed_best_merged.csv",
              encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["instance_id"] not in INSTANCES:
                continue
            try:
                att = float(r["ATT"])
            except (KeyError, ValueError):
                continue
            if att < PENALTY:
                out[(r["instance_id"], r["mode"])].append(att)
    for inst in INSTANCES:
        for arm in ("surrogate", "heuristic", "sa"):
            for d in sorted((root / "runs_reviewer" / inst).glob(f"{arm}_seed*")):
                f = d / "best.txt"
                if not f.exists():
                    continue
                m = re.search(r"best_score=([0-9.]+)",
                              f.read_text(encoding="utf-8", errors="ignore"))
                if m and float(m.group(1)) < PENALTY:
                    out[(inst, arm)].append(float(m.group(1)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    args = ap.parse_args()
    vals = load(Path(args.root))

    print(f"Reference method: {REFERENCE}\n")
    for inst in INSTANCES:
        ref = vals[(inst, REFERENCE)]
        rows, ps = [], []
        for m in OTHERS:
            v = vals.get((inst, m))
            if not v:
                continue
            p = float(mannwhitneyu(ref, v, alternative="two-sided").pvalue)
            rows.append([LABEL[m], len(v), a12(ref, v), p])
            ps.append(p)
        adj = holm(ps)
        print(f"=== {inst}  (n={len(ref)} seeds for {REFERENCE})")
        print(f"  {'method':22s} {'n':>3s} {'A12(both>x)':>11s} {'MWU p':>9s} {'Holm p':>8s}")
        for (lab, n, a, p), pa in zip(rows, adj):
            star = "*" if pa < 0.05 else " "
            print(f"  {lab:22s} {n:3d} {a:11.3f} {p:9.4f} {pa:8.4f} {star}")
        print()
    print("A12 > 0.5 favours the proposed method; * marks Holm p < 0.05.")


if __name__ == "__main__":
    main()
