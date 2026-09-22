from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(".")
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
SUMO = "sumo"

INSTANCES = {
    "A": dict(id="A_k4_3od_x1.0", sumocfg="netA/sim_3od_x1.0.sumocfg", net="netA/network_v2.net.xml",
              kmin=0, kmax=4, demand=1000, protect="E28 E40", scores_csv="scores_betweenness_A.csv"),
    "B": dict(id="B_k9_3od_x1.0", sumocfg="netB/sim_3od_x1.0.sumocfg", net="netB/netB.net.xml",
              kmin=0, kmax=9, demand=2000, protect="E35 -E41 E38 E44 E20 E12", scores_csv="scores_betweenness_B.csv"),
    "J": dict(id="J_k9_6od_x1.0", sumocfg="netJ/wildau_6od_x1.0.sumocfg", net="netJ/Netzmodell2.net.xml",
              kmin=0, kmax=9, demand=1500,
              protect=("4935299#0 111677671#1 255274250 -27149243 27149243 -255274250 -37548821#0 "
                       "311298682#1 -111677671#1 37548821#0 876057378#6"),
              scores_csv="scores_betweenness_J.csv"),
}


def main(keys):
    for key in keys:
        inst = INSTANCES[key]
        for seed in range(10):
            outdir = ROOT / "runs_reviewer" / inst["id"] / f"heuristic_rev_seed{seed}"
            if (outdir / "best.txt").exists():
                print(f"[skip] {outdir}", flush=True)
                continue
            cmd = [PY, "py\\ga_static_policy.py",
                   "--sumocfg", inst["sumocfg"], "--net", inst["net"], "--sumo-bin", SUMO,
                   "--gnn-usage", "both", "--mode", "variable",
                   "--kmin", str(inst["kmin"]), "--kmax", str(inst["kmax"]),
                   "--pop", "24", "--gen", "15", "--jobs", "8",
                   "--pc", "0.9", "--pm", "0.1",
                   "--policy-model", inst["scores_csv"], "--policy-score-mode", "risk",
                   "--time-to-teleport", "-1", "--time-to-teleport-highways", "-1",
                   "--outdir", str(outdir), "--seed", str(seed),
                   "--demand-size", str(inst["demand"]),
                   "--protect-str", inst["protect"]]
            outdir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            with (outdir / "run.log").open("a", encoding="utf-8", errors="ignore") as f:
                r = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
            best = (outdir / "best.txt").read_text().splitlines()[0] if (outdir / "best.txt").exists() else "no best.txt"
            print(f"{inst['id']} seed {seed}: rc={r.returncode} {best} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["A", "B"])
