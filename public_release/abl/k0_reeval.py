from __future__ import annotations
import os
K0_MERGED = os.environ.get('K0_MERGED', './merged_analysis_38_k0')
K0_RUNSFX = os.environ.get('K0_RUNSFX', '_pm01k0')
K0_OUT = os.environ.get('K0_OUT', 'manuscript_numbers_k0')
K0_REEVAL = os.environ.get('K0_REEVAL', 'reeval_39_k0')


import argparse
import csv
import json
import sys
import time
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(ROOT))
import ga_edge_closure_gnn_policy as GA  # noqa: E402

SUMO = "sumo"
PEN = 1e12

PM01 = {"A_k4_3od_x0.8", "A_k4_3od_x1.0", "A_k4_3od_x1.2"}

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
    run_dir = Path(_CTX["tmp"]) / inst / f"{sol_id}_s{sumo_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    add_path = None
    if closed:
        add_path = run_dir / "closures.add.xml"
        GA.write_closures_additional(add_path, closed, cfg["upstream_map"],
                                     begin=0, end=86400)
    att = _run(cfg["sumocfg"], add_path, run_dir, cfg["demand"], sumo_seed)
    try:
        for p in run_dir.iterdir():
            p.unlink()
        run_dir.rmdir()
    except Exception:
        pass
    return inst, sol_id, sumo_seed, att


def load_finals():
    sb = pd.read_csv(ROOT / "merged_analysis_38_k0" / "seed_best_merged.csv")
    sb = sb[~sb.instance_id.isin(PM01)]
    frames = [sb[["instance_id", "mode", "seed", "ATT", "closed_edges"]]]
    for inst in sorted(PM01):
        p = ROOT / f"runs_{inst}{K0_RUNSFX}" / "analysis" / "seed_best.csv"
        d = pd.read_csv(p)
        d["instance_id"] = inst
        frames.append(d[["instance_id", "mode", "seed", "ATT", "closed_edges"]])
    df = pd.concat(frames, ignore_index=True)
    df["closed"] = df.closed_edges.apply(
        lambda v: [] if pd.isna(v) else sorted(str(v).split(",")))
    return df


NET_FILE = {"A": "netA/network_v2.net.xml", "B": "netB/netB.net.xml",
            "C": "netC/netC.net.xml", "D": "netD/netD.net.xml",
            "J": "netJ/Netzmodell2.net.xml"}
EXTRA_PROTECT = {
    "J": ("4935299#0 111677671#1 255274250 -27149243 27149243 -255274250 "
          "-37548821#0 311298682#1 -111677671#1 37548821#0 876057378#6").split(),
}
A_PROTECT = {"1od": ["E28", "E40"], "2od": ["E28", "E40", "E18"],
             "3od": ["E28", "E40", "E18", "E12"]}


def _load_config_index():
    idx = {}
    for fn in ["experiments.json", "experiments_v2.json",
               "experiments_pm_rerun.json"]:
        p = ROOT / fn
        if not p.exists():
            continue
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        lst = e["experiments"] if isinstance(e, dict) and "experiments" in e else e
        for x in lst:
            if not isinstance(x, dict) or "sumocfg" not in x:
                continue
            for k in (x.get("name"), str(x.get("runs_root", "")).replace("runs_", "", 1)):
                if k and k not in idx:
                    idx[k] = x
    return idx


def _demand_from_cfg(sumocfg: Path) -> int:
    root = ET.parse(sumocfg).getroot()
    total = 0
    for rf in root.iter("route-files"):
        for name in str(rf.attrib.get("value", "")).split(","):
            name = name.strip()
            if not name:
                continue
            p = (sumocfg.parent / name)
            if not p.exists():
                continue
            r = ET.parse(p).getroot()
            for fl in r.iter("flow"):
                if "number" in fl.attrib:
                    total += int(float(fl.attrib["number"]))
            total += sum(1 for _ in r.iter("vehicle"))
            total += sum(1 for _ in r.iter("trip"))
    return total


def build_cfgs(instances):
    idx = _load_config_index()
    cfgs = {}
    for inst in instances:
        net_key = inst[0]
        e = idx.get(inst)
        if e is not None:
            sumocfg = ROOT / e["sumocfg"]
            net = ROOT / e["net"]
            prot = e.get("protect")
            prot = set(prot.split()) if isinstance(prot, str) else set(prot or [])
        else:
            parts = inst.split("_")
            sumocfg = ROOT / f"net{net_key}" / f"sim_{parts[2]}_{parts[3]}.sumocfg"
            net = ROOT / NET_FILE[net_key]
            prot = set(EXTRA_PROTECT.get(net_key, []))
            if net_key == "A":
                prot |= set(A_PROTECT.get(parts[2], []))
        if not sumocfg.exists():
            raise SystemExit(f"{inst}: missing sumocfg {sumocfg}")
        demand = _demand_from_cfg(sumocfg)
        full_prot = prot | GA._extract_route_endpoints(str(sumocfg))
        _c, upstream_map, _n = GA.load_candidates(str(net), full_prot, None)
        cfgs[inst] = dict(sumocfg=str(sumocfg), demand=demand,
                          upstream_map=upstream_map)
        print(f"  {inst:16s} cfg={sumocfg.name:24s} demand={demand:5d} "
              f"cand={len(_c)}", flush=True)
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumo-seeds", type=int, nargs="+",
                    default=[201, 202, 203, 204, 205])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--instances", nargs="+", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = load_finals()
    if args.instances:
        df = df[df.instance_id.isin(args.instances)]
    instances = sorted(df.instance_id.unique())
    print(f"instances: {len(instances)}  final solutions: {len(df)}", flush=True)

    outdir = Path(args.out or (ROOT / "reviewer_diagnostics" / "results" /
                               (K0_REEVAL + "_A3od")))
    outdir.mkdir(parents=True, exist_ok=True)

    print("building candidate maps ...", flush=True)
    cfgs = build_cfgs(instances)

    df["key"] = df.closed.apply(tuple)
    sol_ids, tasks = {}, []
    for inst, d in df.groupby("instance_id"):
        for i, key in enumerate(sorted(set(d.key))):
            sol_ids[(inst, key)] = f"S{i:04d}"
    df["sol_id"] = [sol_ids[(r.instance_id, r.key)] for r in df.itertuples()]
    for (inst, key), sid in sol_ids.items():
        for s in args.sumo_seeds:
            tasks.append((inst, sid, list(key), s))
    print(f"distinct closure sets: {len(sol_ids)}  SUMO runs: {len(tasks)}",
          flush=True)

    ctx = dict(cfgs=cfgs, tmp=str(outdir / "tmp"))
    t0 = time.perf_counter()
    res = {}
    with Pool(processes=args.jobs, initializer=_init, initargs=(ctx,)) as pool:
        for n, (inst, sid, seed, att) in enumerate(
                pool.imap_unordered(_work, tasks, chunksize=4), 1):
            res[(inst, sid, seed)] = att
            if n % 200 == 0 or n == len(tasks):
                el = time.perf_counter() - t0
                print(f"  {n}/{len(tasks)}  {el/60:.1f} min "
                      f"(eta {el/n*(len(tasks)-n)/60:.1f} min)", flush=True)

    with (outdir / "reeval39_long.csv").open("w", newline="",
                                             encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "mode", "seed", "sol_id", "att_default",
                    "sumo_seed", "att"])
        for r in df.itertuples():
            for s in args.sumo_seeds:
                w.writerow([r.instance_id, r.mode, r.seed, r.sol_id, r.ATT,
                            s, res[(r.instance_id, r.sol_id, s)]])
    print("wrote", outdir / "reeval39_long.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
