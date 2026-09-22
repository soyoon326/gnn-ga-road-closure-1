from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ga_edge_closure_gnn_policy as GA  # noqa: E402

_orig_try_load_policy = GA.try_load_policy


def _load_static_or_delegate(policy_model_path, policy_graph_path, net,
                             candidate_ids, score_mode_arg="auto"):
    if not policy_model_path or not str(policy_model_path).lower().endswith(".csv"):
        return _orig_try_load_policy(policy_model_path, policy_graph_path, net,
                                     candidate_ids, score_mode_arg)
    scores: dict[str, float] = {}
    with open(policy_model_path, encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row or row[0].strip() in ("", "edge_id") or row[0].startswith("#"):
                continue
            try:
                scores[row[0].strip()] = float(row[1])
            except (IndexError, ValueError):
                continue
    if not scores:
        print(f"[StaticPolicy] empty score file: {policy_model_path}; policy not used")
        return None
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    scores = {k: (v - lo) / rng for k, v in scores.items()}
    mode = score_mode_arg if score_mode_arg in ("risk", "benefit") else "benefit"
    covered = sum(1 for c in candidate_ids if c in scores)
    print(f"[StaticPolicy] {policy_model_path}: loaded scores for {len(scores)} edges "
          f"(candidate coverage {covered}/{len(candidate_ids)}, score_mode={mode})")
    return scores, mode


GA.try_load_policy = _load_static_or_delegate

if __name__ == "__main__":
    GA.main()
