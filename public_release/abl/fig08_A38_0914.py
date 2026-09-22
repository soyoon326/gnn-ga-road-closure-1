import re
import sys
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "figs_k0"))
import make_new_figures as M  # noqa: E402

M.REPORTED_SURROGATE["A_k4_3od_x1.0"] = "reviewer_diagnostics/results/arm_A_sum_run38/arm_results.csv"
_orig_load_arms = M.load_arms


def load_arms(root):
    out = _orig_load_arms(root)
    inst = "A_k4_3od_x1.0"
    for arm in ("heuristic", "sa"):
        vals = []
        for d in sorted((root / "runs_reviewer_A38" / inst).glob(f"{arm}_seed*")):
            f = d / "best.txt"
            if f.exists():
                m = re.search(r"best_score=([0-9.]+)", f.read_text(encoding="utf-8", errors="ignore"))
                if m and float(m.group(1)) < M.PENALTY:
                    vals.append(float(m.group(1)))
        assert len(vals) == 10, (arm, len(vals))
        out[(inst, arm)] = vals
    return out


M.load_arms = load_arms

if __name__ == "__main__":
    M.ps.setup()
    M.fig_baselines(ROOT, Path(sys.argv[1]))
