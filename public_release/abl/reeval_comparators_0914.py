from __future__ import annotations

import csv
import sys
import time
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "py"))
import ga_edge_closure_gnn_policy as GA  # noqa: E402

SUMO = "sumo"
PEN = 1e12
SEEDS = [201, 202, 203, 204, 205]
OUT = ROOT / "reviewer_diagnostics" / "results" / "reeval_cmp_201_205"

INSTANCES = {
    "A_k4_3od_x1.0": dict(sumocfg="netA/sim_3od_x1.0.sumocfg", net="netA/network_v2.net.xml",
                          demand=1000, protect="E28 E40", rev="A"),
    "B_k9_3od_x1.0": dict(sumocfg="netB/sim_3od_x1.0.sumocfg", net="netB/netB.net.xml",
                          demand=2000, protect="E35 -E41 E38 E44 E20 E12", rev="B"),
    "J_k9_6od_x1.0": dict(sumocfg="netJ/wildau_6od_x1.0.sumocfg", net="netJ/Netzmodell2.net.xml",
                          demand=1500,
                          protect=("4935299#0 111677671#1 255274250 -27149243 27149243 "
                                   "-255274250 -37548821#0 311298682#1 -111677671#1 "
                                   "37548821#0 876057378#6"), rev="J"),
}

_CTX = {}


def _init(ctx):
    _CTX.clear()
    _CTX.update(ctx)


def _run(sumocfg, additional, outdir, demand_size, sumo_seed):
    import subprocess
    tripinfo = Path(outdir) / "tripinfo.xml"
    cmd = [SUMO, "-c", sumocfg,
           "--no-step-log", "true",
           "--tripinfo-output", str(tripinfo),
           "--tripinfo-output.write-unfinished", "true",
           "--device.rerouting.probability", "1.0",
           "--device.rerouting.period", "30",
           "--duration-log.disable", "true",
           "--time-to-teleport", "-1",
           "--time-to-teleport.highways", "-1",
           "--seed", str(sumo_seed)]
    if additional is not None:
        cmd += ["-a", str(additional)]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except Exception:
        return PEN
    if not tripinfo.exists():
        return PEN
    durs, n = [], 0
    try:
        for ev in ET.parse(tripinfo).getroot().iter("tripinfo"):
            n += 1
            if ev.attrib.get("vaporized", "false") == "true":
                return PEN
            d = ev.attrib.get("duration")
            if d is not None and float(d) >= 0:
                durs.append(float(d))
    except Exception:
        return PEN
    finally:
        try:
            tripinfo.unlink(missing_ok=True)
        except Exception:
            pass
    if demand_size is not None and n < demand_size:
        return PEN
    return float(np.mean(durs)) if durs else PEN


def _work(task):
    inst, sol_id, closed, sumo_seed = task
    cfg = _CTX["cfgs"][inst]
    run_dir = OUT / "tmp" / inst / f"{sol_id}_s{sumo_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    add_path = None
    if closed:
        add_path = run_dir / "closures.add.xml"
        GA.write_closures_additional(add_path, closed, cfg["upstream_map"], begin=0, end=86400)
    att = _run(cfg["sumocfg"], add_path, run_dir, cfg["demand"], sumo_seed)
    try:
        for p in run_dir.iterdir():
            p.unlink()
        run_dir.rmdir()
    except Exception:
        pass
    return inst, sol_id, sumo_seed, att


def _closures(addf: Path):
    return sorted({ca.attrib["id"] for ca in ET.parse(addf).getroot().iter("closingReroute")})


def _tripinfo_att(tri: Path, demand: int):
    durs = [float(ev.attrib["duration"]) for ev in ET.parse(tri).getroot().iter("tripinfo")
            if float(ev.attrib.get("duration", -1)) >= 0]
    return float(np.mean(durs)) if len(durs) >= demand else None


def collect(inst):
    cfg = INSTANCES[inst]
    rows = []
    for arm, tag in [("heuristic", "heuristic"), ("sa", "sa"), ("surrogate_submitted", "surrogate")]:
        for s in range(10):
            bt = ROOT / "runs_reviewer" / inst / f"{tag}_seed{s}" / "best.txt"
            lines = bt.read_text().splitlines()
            att = float(lines[0].split("=", 1)[1])
            closed = sorted(lines[1].split("=", 1)[1].split()) if len(lines) > 1 else []
            rows.append(dict(arm=arm, seed=s, closed=closed, att_default=att, source=str(bt.parent.name)))
    res = ROOT / "reviewer_diagnostics" / "results"
    if cfg["rev"] == "J":
        ar = pd.read_csv(res / "arm_J_sum_run" / "arm_results.csv")
        for s, d in ar.groupby("seed"):
            r = d.loc[d.att.idxmin()]
            rows.append(dict(arm="surrogate_repaired", seed=int(s),
                             closed=_closures(res / "arm_J_sum_run" / r.tag / "closures.add.xml"),
                             att_default=float(r.att), source=f"arm_J_sum_run/{r.tag} (re-run)"))
    else:
        per = {}
        for cdir in sorted((res / f"arm_{cfg['rev']}_sum_run").glob("s*_r*")):
            tri, addf = cdir / "tripinfo.xml", cdir / "closures.add.xml"
            if not (tri.exists() and addf.exists()):
                continue
            att = _tripinfo_att(tri, cfg["demand"])
            if att is None:
                continue
            s = int(cdir.name.split("_")[0][1:])
            if s not in per or att < per[s][0]:
                per[s] = (att, addf)
        for s, (att, addf) in sorted(per.items()):
            rows.append(dict(arm="surrogate_repaired", seed=s, closed=_closures(addf),
                             att_default=att, source=f"arm_{cfg['rev']}_sum_run/{addf.parent.name}"))
    rows.append(dict(arm="no_closure", seed=0, closed=[], att_default=float("nan"), source="empty set"))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfgs, tasks, sol_rows = {}, [], []
    for inst, cfg in INSTANCES.items():
        protect = set(cfg["protect"].split()) | set(GA._extract_route_endpoints(str(ROOT / cfg["sumocfg"])) or [])
        _c, upmap, _n = GA.load_candidates(str(ROOT / cfg["net"]), protect, None)
        cfgs[inst] = dict(sumocfg=str(ROOT / cfg["sumocfg"]), demand=cfg["demand"], upstream_map=upmap)
        sol_ids = {}
        for r in collect(inst):
            key = tuple(r["closed"])
            if key not in sol_ids:
                sol_ids[key] = f"S{len(sol_ids):03d}"
                tasks += [(inst, sol_ids[key], list(key), h) for h in SEEDS]
            sol_rows.append(dict(instance_id=inst, arm=r["arm"], seed=r["seed"], sol_id=sol_ids[key],
                                 att_default=r["att_default"], k=len(r["closed"]), source=r["source"],
                                 closed=" ".join(r["closed"])))
        print(f"{inst}: {len(sol_ids)} distinct solutions -> {len(sol_ids) * len(SEEDS)} simulations", flush=True)
    pd.DataFrame(sol_rows).to_csv(OUT / "solutions.csv", index=False)
    done = set()
    long_path = OUT / "long.csv"
    if long_path.exists():
        prev = pd.read_csv(long_path)
        done = {(a, b, int(c)) for a, b, c in zip(prev.instance_id, prev.sol_id, prev.sumo_seed)}
    tasks = [t for t in tasks if (t[0], t[1], t[3]) not in done]
    print(f"{len(tasks)} simulations to run ({len(done)} already done)", flush=True)
    t0 = time.perf_counter()
    new = not long_path.exists()
    with long_path.open("a", newline="", encoding="utf-8") as f, \
            Pool(processes=8, initializer=_init, initargs=(dict(cfgs=cfgs),)) as pool:
        w = csv.writer(f)
        if new:
            w.writerow(["instance_id", "sol_id", "sumo_seed", "att"])
        for i, (inst, sol, h, att) in enumerate(pool.imap_unordered(_work, tasks), 1):
            w.writerow([inst, sol, h, att]); f.flush()
            if i % 10 == 0 or i == len(tasks):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(tasks)} {inst} elapsed {el / 60:.1f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
