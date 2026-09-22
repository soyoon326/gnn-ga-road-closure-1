from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(ROOT / "reviewer_diagnostics" / "scripts"))
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
SUMO = "sumo"
PROTECT = "E28 E40 E18 E12"
INST = dict(id="A_k4_3od_x1.0", sumocfg="netA/sim_3od_x1.0.sumocfg", net="netA/network_v2.net.xml",
            kmin=0, kmax=4, demand=1000, scores_csv="scores_betweenness_A.csv",
            policy_dir="output_policy_A_k4_3od_x1.0")
OUTROOT = ROOT / "runs_reviewer_A38" / INST["id"]


def log(which: str, msg: str) -> None:
    print(msg, flush=True)
    with (ROOT / "runs_reviewer_A38" / f"rerun_A38_{which}.log").open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def preflight() -> None:
    import ga_edge_closure_gnn_policy as GA
    eps = GA._extract_route_endpoints(str(ROOT / INST["sumocfg"]))
    cands, _, _ = GA.load_candidates(str(ROOT / INST["net"]), set(PROTECT.split()) | set(eps or []), None)
    assert len(cands) == 38 and not ({"E12", "E18", "E28", "E40"} & set(cands)), len(cands)


def common(outdir: Path, seed: int) -> list[str]:
    return ["--sumocfg", INST["sumocfg"], "--net", INST["net"], "--sumo-bin", SUMO,
            "--kmin", str(INST["kmin"]), "--kmax", str(INST["kmax"]),
            "--time-to-teleport", "-1", "--time-to-teleport-highways", "-1",
            "--outdir", str(outdir), "--seed", str(seed),
            "--demand-size", str(INST["demand"])]


def ga_cmd(outdir: Path, seed: int, score_mode: str) -> list[str]:
    return ([PY, "py\\ga_static_policy.py"] + common(outdir, seed) +
            ["--gnn-usage", "both", "--mode", "variable", "--pop", "24", "--gen", "15", "--jobs", "8",
             "--pc", "0.9", "--pm", "0.1", "--policy-model", INST["scores_csv"],
             "--policy-score-mode", score_mode, "--protect-str", PROTECT])


def sa_cmd(outdir: Path, seed: int) -> list[str]:
    return [PY, "py\\sa_edge_closure.py"] + common(outdir, seed) + ["--budget", "360", "--protect-str", PROTECT]


def surr_cmd(outdir: Path, seed: int) -> list[str]:
    return ([PY, "py\\surrogate_ga.py", "--data", INST["policy_dir"],
             "--model", str(ROOT / INST["policy_dir"] / "surrogate_att.pt")] + common(outdir, seed) +
            ["--pop", "24", "--gen", "200", "--verify-top", "10", "--protect-str", PROTECT])


def run(which: str, tag: str, seed: int, build) -> None:
    outdir = OUTROOT / f"{tag}_seed{seed}"
    if (outdir / "best.txt").exists():
        log(which, f"[skip] {outdir}")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with (outdir / "run.log").open("a", encoding="utf-8", errors="ignore") as f:
        r = subprocess.run(build(outdir, seed), cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
    best = (outdir / "best.txt").read_text().splitlines()[0] if (outdir / "best.txt").exists() else "no best.txt"
    log(which, f"{tag} seed {seed}: rc={r.returncode} {best} ({(time.time() - t0) / 60:.1f} min)")


def surr_reported() -> None:
    import surrogate_arm_v3 as S
    S.INSTANCES["A"]["protect"] = PROTECT
    out = ROOT / "reviewer_diagnostics" / "results" / "arm_A_sum_run38"
    if (out / "arm_results.csv").exists():
        log("surr", f"[skip] {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    sys.argv = ["surrogate_arm_v3.py", "--inst", "A",
                "--model", str(ROOT / "reviewer_diagnostics" / "models" / "A_sum_clean.pt"), "--outdir", str(out)]
    t0 = time.time()
    with (out / "run.log").open("a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        rc = S.main()
    log("surr", f"reported arm (sum readout): rc={rc} ({(time.time() - t0) / 60:.1f} min) -> {out}")


if __name__ == "__main__":
    which = sys.argv[1]
    (ROOT / "runs_reviewer_A38").mkdir(exist_ok=True)
    preflight()
    if which == "ga":
        for s in range(10):
            run(which, "heuristic", s, lambda o, sd: ga_cmd(o, sd, "benefit"))
        for s in range(10):
            run(which, "heuristic_rev", s, lambda o, sd: ga_cmd(o, sd, "risk"))
    elif which == "sa":
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(lambda s: run(which, "sa", s, sa_cmd), range(10)))
    elif which == "surr":
        surr_reported()
        for s in range(10):
            run(which, "surrogate", s, surr_cmd)
    log(which, f"[DONE {which}]")
