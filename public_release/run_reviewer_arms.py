from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
SUMO = "sumo"

INSTANCES = {
    "A": {
        "id": "A_k4_3od_x1.0",
        "sumocfg": "netA/sim_3od_x1.0.sumocfg",
        "net": "netA/network_v2.net.xml",
        "kmin": 0, "kmax": 4,
        "demand": 1000,
        "protect": "E28 E40",
        "policy_dir": "output_policy_A_k4_3od_x1.0",
        "scores_csv": "scores_betweenness_A.csv",
        "jobs": 8,
    },
    "B": {
        "id": "B_k9_3od_x1.0",
        "sumocfg": "netB/sim_3od_x1.0.sumocfg",
        "net": "netB/netB.net.xml",
        "kmin": 0, "kmax": 9,
        "demand": 2000,
        "protect": "E35 -E41 E38 E44 E20 E12",
        "policy_dir": "output_policy_B_k9_3od_x1.0",
        "scores_csv": "scores_betweenness_B.csv",
        "jobs": 8,
    },
    "J": {
        "id": "J_k9_6od_x1.0",
        "sumocfg": "netJ/wildau_6od_x1.0.sumocfg",
        "net": "netJ/Netzmodell2.net.xml",
        "kmin": 0, "kmax": 9,
        "demand": 1500,
        "protect": ("4935299#0 111677671#1 255274250 -27149243 27149243 "
                    "-255274250 -37548821#0 311298682#1 -111677671#1 "
                    "37548821#0 876057378#6"),
        "policy_dir": "output_policy_J_k9_6od_x1.0",
        "scores_csv": "scores_betweenness_J.csv",
        "jobs": 8,
    },
}


def run(cmd: list[str], log_path: Path, dry: bool) -> int:
    print("[CMD]", " ".join(cmd), flush=True)
    if dry:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="ignore") as f:
        r = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
    return r.returncode


def arm_heuristic(inst: dict, seed: int, outroot: Path, dry: bool) -> int:
    outdir = outroot / f"heuristic_seed{seed}"
    if (outdir / "best.txt").exists():
        print(f"[skip] {outdir} already complete"); return 0
    cmd = [PY, "py\\ga_static_policy.py",
           "--sumocfg", inst["sumocfg"], "--net", inst["net"], "--sumo-bin", SUMO,
           "--gnn-usage", "both", "--mode", "variable",
           "--kmin", str(inst["kmin"]), "--kmax", str(inst["kmax"]),
           "--pop", "24", "--gen", "15", "--jobs", str(inst["jobs"]),
           "--pc", "0.9", "--pm", "0.1",
           "--policy-model", inst["scores_csv"], "--policy-score-mode", "benefit",
           "--time-to-teleport", "-1", "--time-to-teleport-highways", "-1",
           "--outdir", str(outdir), "--seed", str(seed),
           "--demand-size", str(inst["demand"]),
           "--protect-str", inst["protect"]]
    return run(cmd, outdir / "run.log", dry)


def arm_sa(inst: dict, seed: int, outroot: Path, dry: bool) -> int:
    outdir = outroot / f"sa_seed{seed}"
    if (outdir / "best.txt").exists():
        print(f"[skip] {outdir} already complete"); return 0
    cmd = [PY, "py\\sa_edge_closure.py",
           "--sumocfg", inst["sumocfg"], "--net", inst["net"], "--sumo-bin", SUMO,
           "--kmin", str(inst["kmin"]), "--kmax", str(inst["kmax"]),
           "--budget", "360", "--seed", str(seed),
           "--time-to-teleport", "-1", "--time-to-teleport-highways", "-1",
           "--outdir", str(outdir),
           "--demand-size", str(inst["demand"]),
           "--protect-str", inst["protect"]]
    return run(cmd, outdir / "run.log", dry)


def arm_costeq(inst: dict, seed: int, outroot: Path, dry: bool) -> int:
    outdir = outroot / f"costeq_seed{seed}"
    if (outdir / "best.txt").exists():
        print(f"[skip] {outdir} already complete"); return 0
    cmd = [PY, "py\\ga_edge_closure_gnn_policy.py",
           "--sumocfg", inst["sumocfg"], "--net", inst["net"], "--sumo-bin", SUMO,
           "--gnn-usage", "none", "--mode", "variable",
           "--kmin", str(inst["kmin"]), "--kmax", str(inst["kmax"]),
           "--pop", "24", "--gen", "57", "--jobs", str(inst["jobs"]),
           "--pc", "0.9", "--pm", "0.1",
           "--time-to-teleport", "-1", "--time-to-teleport-highways", "-1",
           "--outdir", str(outdir), "--seed", str(seed),
           "--demand-size", str(inst["demand"]),
           "--protect-str", inst["protect"]]
    return run(cmd, outdir / "run.log", dry)


def arm_surrogate(inst: dict, seed: int, outroot: Path, dry: bool) -> int:
    outdir = outroot / f"surrogate_seed{seed}"
    if (outdir / "best.txt").exists():
        print(f"[skip] {outdir} already complete"); return 0
    model = ROOT / inst["policy_dir"] / "surrogate_att.pt"
    if not model.exists():
        print(f"[surrogate] model not found: {model}; skipped")
        return 1
    cmd = [PY, "py\\surrogate_ga.py",
           "--data", inst["policy_dir"], "--model", str(model),
           "--sumocfg", inst["sumocfg"], "--net", inst["net"], "--sumo-bin", SUMO,
           "--kmin", str(inst["kmin"]), "--kmax", str(inst["kmax"]),
           "--pop", "24", "--gen", "200", "--verify-top", "10",
           "--time-to-teleport", "-1", "--time-to-teleport-highways", "-1",
           "--outdir", str(outdir), "--seed", str(seed),
           "--demand-size", str(inst["demand"]),
           "--protect-str", inst["protect"]]
    return run(cmd, outdir / "run.log", dry)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=["A", "B", "J"],
                    choices=list(INSTANCES))
    ap.add_argument("--arms", nargs="+", default=["surrogate", "heuristic", "sa", "costeq"],
                    choices=["heuristic", "sa", "surrogate", "costeq"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--sa-parallel", type=int, default=4,
                    help="number of SA chains run in parallel")
    ap.add_argument("--wait-for", default=None,
                    help="wait until this file exists before starting")
    ap.add_argument("--wait-timeout-h", type=float, default=72.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.wait_for:
        gate = Path(args.wait_for)
        deadline = time.time() + args.wait_timeout_h * 3600
        print(f"[gate] waiting for {gate}...", flush=True)
        while not gate.exists() and time.time() < deadline:
            time.sleep(300)
        print(f"[gate] released (exists={gate.exists()})", flush=True)

    for key in args.instances:
        inst = INSTANCES[key]
        outroot = ROOT / "runs_reviewer" / inst["id"]
        outroot.mkdir(parents=True, exist_ok=True)
        print(f"\n========== {inst['id']} ==========", flush=True)
        for arm in args.arms:
            t0 = time.time()
            print(f"--- arm: {arm} ---", flush=True)
            if arm == "sa":
                with ThreadPoolExecutor(max_workers=max(1, args.sa_parallel)) as ex:
                    futs = {ex.submit(arm_sa, inst, s, outroot, args.dry_run): s
                            for s in args.seeds}
                    for fut in as_completed(futs):
                        s = futs[fut]
                        print(f"[sa seed{s}] exit={fut.result()}", flush=True)
            else:
                fn = {"heuristic": arm_heuristic, "surrogate": arm_surrogate,
                      "costeq": arm_costeq}[arm]
                for s in args.seeds:
                    rc = fn(inst, s, outroot, args.dry_run)
                    print(f"[{arm} seed{s}] exit={rc}", flush=True)
            print(f"--- {arm} done in {(time.time()-t0)/3600:.2f} h ---", flush=True)
    print("[ALL DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
