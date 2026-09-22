import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "abl"))
sys.path.insert(0, str(ROOT / "py"))
import reeval_comparators_0914 as RC  # noqa: E402

RC.OUT = ROOT / "reviewer_diagnostics" / "results" / "reeval_cmp_201_205_A38"
RC.INSTANCES = {"A_k4_3od_x1.0": dict(sumocfg="netA/sim_3od_x1.0.sumocfg", net="netA/network_v2.net.xml",
                                      demand=1000, protect="E28 E40 E18 E12", rev="A")}
RR = ROOT / "runs_reviewer_A38" / "A_k4_3od_x1.0"
RES = ROOT / "reviewer_diagnostics" / "results"
INST = "A_k4_3od_x1.0"


def collect():
    rows = []
    for arm, tag in [("heuristic", "heuristic"), ("sa", "sa"), ("surrogate_submitted", "surrogate")]:
        for s in range(10):
            bt = RR / f"{tag}_seed{s}" / "best.txt"
            if not bt.exists():
                continue
            lines = bt.read_text().splitlines()
            att = float(lines[0].split("=", 1)[1])
            closed = sorted(lines[1].split("=", 1)[1].split()) if len(lines) > 1 else []
            rows.append(dict(arm=arm, seed=s, closed=closed, att_default=att,
                             source=f"runs_reviewer_A38/{bt.parent.name}"))
    per = {}
    for cdir in sorted((RES / "arm_A_sum_run38").glob("s*_r*")):
        tri, addf = cdir / "tripinfo.xml", cdir / "closures.add.xml"
        if not (tri.exists() and addf.exists()):
            continue
        att = RC._tripinfo_att(tri, 1000)
        if att is None:
            continue
        s = int(cdir.name.split("_")[0][1:])
        if s not in per or att < per[s][0]:
            per[s] = (att, addf)
    for s, (att, addf) in sorted(per.items()):
        rows.append(dict(arm="surrogate_repaired", seed=s, closed=RC._closures(addf), att_default=att,
                         source=f"arm_A_sum_run38/{addf.parent.name}"))
    rows.append(dict(arm="no_closure", seed=0, closed=[], att_default=float("nan"), source="empty set"))
    return rows


def main():
    out = RC.OUT
    out.mkdir(parents=True, exist_ok=True)
    cfg = RC.INSTANCES[INST]
    protect = set(cfg["protect"].split()) | set(RC.GA._extract_route_endpoints(str(ROOT / cfg["sumocfg"])) or [])
    cands, upmap, _n = RC.GA.load_candidates(str(ROOT / cfg["net"]), protect, None)
    assert len(cands) == 38, len(cands)
    cfgs = {INST: dict(sumocfg=str(ROOT / cfg["sumocfg"]), demand=cfg["demand"], upstream_map=upmap)}
    tasks, sol_rows, seen = [], [], set()
    for r in collect():
        key = tuple(r["closed"])
        sid = "S_" + ("-".join(key) if key else "empty")
        if sid not in seen:
            seen.add(sid)
            tasks += [(INST, sid, list(key), h) for h in RC.SEEDS]
        sol_rows.append(dict(instance_id=INST, arm=r["arm"], seed=r["seed"], sol_id=sid,
                             att_default=r["att_default"], k=len(key), source=r["source"], closed=" ".join(key)))
    pd.DataFrame(sol_rows).to_csv(out / "solutions.csv", index=False)
    long_path = out / "long.csv"
    done = set()
    if long_path.exists():
        prev = pd.read_csv(long_path)
        done = {(a, b, int(c)) for a, b, c in zip(prev.instance_id, prev.sol_id, prev.sumo_seed)}
    tasks = [t for t in tasks if (t[0], t[1], t[3]) not in done]
    counts = pd.DataFrame(sol_rows).groupby("arm").size().to_dict()
    print(f"runs per arm {counts}; {len(seen)} distinct solutions; {len(tasks)} simulations to run ({len(done)} done)", flush=True)
    new = not long_path.exists()
    with long_path.open("a", newline="", encoding="utf-8") as f, \
            Pool(processes=8, initializer=RC._init, initargs=(dict(cfgs=cfgs),)) as pool:
        w = csv.writer(f)
        if new:
            w.writerow(["instance_id", "sol_id", "sumo_seed", "att"])
        for i, (inst, sol, h, att) in enumerate(pool.imap_unordered(RC._work, tasks), 1):
            w.writerow([inst, sol, h, att]); f.flush()
            if i % 25 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
