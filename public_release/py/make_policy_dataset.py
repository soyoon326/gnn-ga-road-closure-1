from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def ensure_sumo_tools_on_path():
    sh = os.environ.get("SUMO_HOME")
    if sh:
        tools = str(Path(sh) / "tools")
        import sys
        if tools not in sys.path:
            sys.path.append(tools)


def import_traci_sumolib():
    ensure_sumo_tools_on_path()
    import traci  # type: ignore
    from sumolib.net import readNet  # type: ignore
    return traci, readNet


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _norm_cols(a: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = a.mean(axis=0, keepdims=True).astype(np.float32)
    sd = a.std(axis=0, keepdims=True).astype(np.float32)
    sd = np.where(sd < eps, 1.0, sd).astype(np.float32)
    return ((a - mu) / sd).astype(np.float32), mu.reshape(-1), sd.reshape(-1)


def build_graph(net_path: str):
    _, readNet = import_traci_sumolib()
    net = readNet(net_path)
    nodes = net.getNodes()
    node_id_map = {n.getID(): i for i, n in enumerate(nodes)}
    N = len(nodes)

    deg_in = np.zeros((N, 1), dtype=np.float32)
    deg_out = np.zeros((N, 1), dtype=np.float32)
    is_tls = np.zeros((N, 1), dtype=np.float32)

    for n in nodes:
        i = node_id_map[n.getID()]
        deg_in[i, 0] = len([e for e in n.getIncoming() if not e.getID().startswith(":")])
        deg_out[i, 0] = len([e for e in n.getOutgoing() if not e.getID().startswith(":")])
        is_tls[i, 0] = 1.0 if n.getType() == "traffic_light" else 0.0

    x_node = np.concatenate([deg_in, deg_out, is_tls], axis=1).astype(np.float32)
    node_feature_names = ["deg_in", "deg_out", "is_tls"]

    edges = [
        e for e in net.getEdges()
        if not e.getID().startswith(":")
        and (e.getFunction() or "") != "internal"
        and len(e.getLanes()) > 0
    ]
    edge_id_map = {e.getID(): i for i, e in enumerate(edges)}
    edge_ids_all = [e.getID() for e in edges]
    E = len(edges)

    src = np.array([node_id_map[e.getFromNode().getID()] for e in edges], dtype=np.int64)
    dst = np.array([node_id_map[e.getToNode().getID()] for e in edges], dtype=np.int64)
    edge_index = np.stack([src, dst], axis=0)

    length = np.array([e.getLength() for e in edges], dtype=np.float32).reshape(E, 1)
    lanes = np.array([len(e.getLanes()) for e in edges], dtype=np.float32).reshape(E, 1)
    speed = np.array([e.getSpeed() for e in edges], dtype=np.float32).reshape(E, 1)
    priority = np.array([(e.getPriority() or 0) for e in edges], dtype=np.float32).reshape(E, 1)
    from_tls = is_tls[src].astype(np.float32)
    to_tls = is_tls[dst].astype(np.float32)

    edge_attr = np.concatenate([length, lanes, speed, priority, from_tls, to_tls], axis=1).astype(np.float32)
    edge_feature_names = ["length", "lanes", "speed_limit", "priority", "from_tls", "to_tls"]
    return edge_index, x_node, edge_attr, node_id_map, edge_id_map, edge_ids_all, node_feature_names, edge_feature_names


def build_upstream_map(net_path: str) -> Dict[str, List[str]]:
    _, readNet = import_traci_sumolib()
    net = readNet(net_path)
    edges = [
        e for e in net.getEdges()
        if not e.getID().startswith(":")
        and (e.getFunction() or "") != "internal"
        and len(e.getLanes()) > 0
    ]
    m: Dict[str, List[str]] = {}
    for e in edges:
        up = []
        for ine in e.getFromNode().getIncoming():
            if ine.getID().startswith(":"):
                continue
            up.append(ine.getID())
        m[e.getID()] = up if up else [e.getID()]
    return m


def _split_route_files(value: str) -> List[str]:
    value = (value or "").strip()
    if not value:
        return []
    parts = re.split(r"[;,]\s*|\s+", value)
    return [p for p in parts if p]


def _route_files(sumocfg_path: str) -> List[Path]:
    sumocfg = Path(sumocfg_path)
    try:
        cfg_root = ET.parse(sumocfg).getroot()
    except Exception:
        return []
    rf = cfg_root.find(".//route-files")
    if rf is None:
        return []
    files = _split_route_files(rf.get("value") or "")
    out = []
    for f in files:
        p = Path(f)
        if not p.is_absolute():
            p = (sumocfg.parent / f).resolve()
        if p.exists():
            out.append(p)
    return out


def extract_route_endpoints(sumocfg_path: str) -> List[str]:
    endpoints: set[str] = set()
    for p in _route_files(sumocfg_path):
        try:
            rroot = ET.parse(p).getroot()
        except Exception:
            continue
        for fl in rroot.iter("flow"):
            frm = fl.get("from"); to = fl.get("to")
            if frm: endpoints.add(frm)
            if to: endpoints.add(to)
        for tr in rroot.iter("trip"):
            frm = tr.get("from"); to = tr.get("to")
            if frm: endpoints.add(frm)
            if to: endpoints.add(to)
        for r in rroot.iter("route"):
            edges = (r.get("edges") or "").split()
            if edges:
                endpoints.add(edges[0]); endpoints.add(edges[-1])
        for v in rroot.iter("vehicle"):
            rr = v.find("route")
            if rr is not None:
                edges = (rr.get("edges") or "").split()
                if edges:
                    endpoints.add(edges[0]); endpoints.add(edges[-1])
    return sorted(endpoints)


def compute_route_edge_counts(sumocfg_path: str, edge_map: Dict[str, int], E: int) -> np.ndarray:
    counts = np.zeros((E, 1), dtype=np.float32)
    for p in _route_files(sumocfg_path):
        try:
            root = ET.parse(p).getroot()
        except Exception:
            continue
        route_defs: Dict[str, List[str]] = {}
        for r in root.iter("route"):
            rid = r.get("id")
            edges = (r.get("edges") or "").split()
            if rid and edges:
                route_defs[rid] = edges
                for eid in edges:
                    if eid in edge_map:
                        counts[edge_map[eid], 0] += 1.0
        for v in root.iter("vehicle"):
            rr = v.find("route")
            edges: List[str] = []
            if rr is not None and rr.get("edges"):
                edges = (rr.get("edges") or "").split()
            elif v.get("route") in route_defs:
                edges = route_defs[v.get("route")]  # type: ignore[index]
            for eid in edges:
                if eid in edge_map:
                    counts[edge_map[eid], 0] += 1.0
    return counts


def write_addfile(path: Path, closed_edges: List[str], upstream_map: Dict[str, List[str]]):
    add = ET.Element("additional")
    if closed_edges:
        notify = set()
        for e in closed_edges:
            notify.update(upstream_map.get(e, [e]))
        if notify:
            rr = ET.SubElement(add, "rerouter", attrib={
                "id": "rr_close",
                "edges": " ".join(sorted(notify)),
                "probability": "1.0",
            })
            iv = ET.SubElement(rr, "interval", attrib={"begin": "0", "end": "86400"})
            for e in closed_edges:
                ET.SubElement(iv, "closingReroute", attrib={"id": e})
    ET.ElementTree(add).write(path, encoding="UTF-8", xml_declaration=True)


def run_sumo_score(
    sumo_bin: str,
    sumocfg: str,
    addfile: Path,
    workdir: Path,
    *,
    sumo_seed: Optional[int] = None,
    reroute_period: int = 30,
    step_limit: int = 200000,
    time_to_teleport: int = -1,
    time_to_teleport_highways: int = -1,
) -> float:
    traci, _ = import_traci_sumolib()
    tripinfo = workdir / "tripinfo.xml"
    if tripinfo.exists():
        try:
            tripinfo.unlink()
        except Exception:
            pass
    cmd = [
        sumo_bin, "-c", sumocfg, "-a", str(addfile),
        "--no-step-log", "true",
        "--tripinfo-output", str(tripinfo),
        "--device.rerouting.probability", "1.0",
        "--device.rerouting.period", str(reroute_period),
        "--duration-log.disable", "true",
        "--time-to-teleport", str(time_to_teleport),
        "--time-to-teleport.highways", str(time_to_teleport_highways),
    ]
    if sumo_seed is not None:
        cmd += ["--seed", str(int(sumo_seed))]
    label = f"policy_ds_{os.getpid()}_{sumo_seed if sumo_seed is not None else 'x'}"
    try:
        traci.start(cmd, label=label)
        conn = traci.getConnection(label)
        step = 0
        while conn.simulation.getMinExpectedNumber() > 0 and step < step_limit:
            conn.simulationStep()
            step += 1
        conn.close()
        if not tripinfo.exists():
            return 1e12
        durs = []
        root = ET.parse(tripinfo).getroot()
        for ev in root.iter("tripinfo"):
            ds = ev.attrib.get("duration")
            if ds is None:
                continue
            d = float(ds)
            if d >= 0:
                durs.append(d)
        if not durs:
            return 1e12
        return float(np.mean(durs))
    except Exception:
        return 1e12
    finally:
        try:
            traci.close(False)
        except Exception:
            pass


def run_baseline_features(
    sumo_bin: str,
    sumocfg: str,
    addfile: Path,
    workdir: Path,
    edge_ids_all: List[str],
    *,
    sumo_seed: Optional[int],
    reroute_period: int,
    time_to_teleport: int,
    time_to_teleport_highways: int,
    step_limit: int = 200000,
) -> Tuple[float, np.ndarray, List[str]]:
    traci, _ = import_traci_sumolib()
    E = len(edge_ids_all)
    sums = {"veh": np.zeros(E, dtype=np.float64), "speed": np.zeros(E, dtype=np.float64),
            "tt": np.zeros(E, dtype=np.float64), "occ": np.zeros(E, dtype=np.float64),
            "halt": np.zeros(E, dtype=np.float64)}
    valid_steps = 0
    tripinfo = workdir / "tripinfo.xml"
    if tripinfo.exists():
        try: tripinfo.unlink()
        except Exception: pass
    cmd = [
        sumo_bin, "-c", sumocfg, "-a", str(addfile),
        "--no-step-log", "true",
        "--tripinfo-output", str(tripinfo),
        "--device.rerouting.probability", "1.0",
        "--device.rerouting.period", str(reroute_period),
        "--duration-log.disable", "true",
        "--time-to-teleport", str(time_to_teleport),
        "--time-to-teleport.highways", str(time_to_teleport_highways),
    ]
    if sumo_seed is not None:
        cmd += ["--seed", str(int(sumo_seed))]
    label = f"baseline_{os.getpid()}_{sumo_seed if sumo_seed is not None else 'x'}"
    try:
        traci.start(cmd, label=label)
        conn = traci.getConnection(label)

        use_sub = False
        try:
            import traci.constants as tc
            sub_vars = [
                tc.LAST_STEP_VEHICLE_NUMBER,
                tc.LAST_STEP_MEAN_SPEED,
                tc.VAR_CURRENT_TRAVELTIME,
                tc.LAST_STEP_OCCUPANCY,
                tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
            ]
            for eid in edge_ids_all:
                conn.edge.subscribe(eid, sub_vars)
            use_sub = True
        except Exception as e:
            print(f"[baseline] edge subscription unavailable ({e}); falling back to per-edge polling")

        step = 0
        while conn.simulation.getMinExpectedNumber() > 0 and step < step_limit:
            conn.simulationStep()
            if use_sub:
                res = conn.edge.getAllSubscriptionResults()
                for i, eid in enumerate(edge_ids_all):
                    r = res.get(eid)
                    if r:
                        sums["veh"][i] += r.get(tc.LAST_STEP_VEHICLE_NUMBER, 0.0)
                        sums["speed"][i] += r.get(tc.LAST_STEP_MEAN_SPEED, 0.0)
                        sums["tt"][i] += r.get(tc.VAR_CURRENT_TRAVELTIME, 0.0)
                        sums["occ"][i] += r.get(tc.LAST_STEP_OCCUPANCY, 0.0)
                        sums["halt"][i] += r.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0.0)
            else:
                for i, eid in enumerate(edge_ids_all):
                    try:
                        sums["veh"][i] += conn.edge.getLastStepVehicleNumber(eid)
                        sums["speed"][i] += conn.edge.getLastStepMeanSpeed(eid)
                        sums["tt"][i] += conn.edge.getTraveltime(eid)
                        sums["occ"][i] += conn.edge.getLastStepOccupancy(eid)
                        sums["halt"][i] += conn.edge.getLastStepHaltingNumber(eid)
                    except Exception:
                        pass
            valid_steps += 1
            step += 1
        conn.close()
    except Exception:
        try: conn.close()
        except Exception: pass
        return 1e12, np.zeros((E, 5), dtype=np.float32), ["base_veh", "base_speed", "base_travel_time", "base_occupancy", "base_halting"]
    finally:
        try: traci.close(False)
        except Exception: pass

    denom = max(1, valid_steps)
    feat = np.stack([sums[k] / denom for k in ["veh", "speed", "tt", "occ", "halt"]], axis=1).astype(np.float32)
    att = 1e12
    if tripinfo.exists():
        try:
            durs = [float(ev.attrib["duration"]) for ev in ET.parse(tripinfo).getroot().iter("tripinfo") if "duration" in ev.attrib]
            if durs:
                att = float(np.mean(durs))
        except Exception:
            pass
    return att, feat, ["base_veh", "base_speed", "base_travel_time", "base_occupancy", "base_halting"]


def _choose_random_closed(rng: np.random.Generator, edge_ids_sorted: List[str], kmax: int, kmin: int = 0) -> List[str]:
    hi = min(int(kmax), len(edge_ids_sorted))
    lo = max(0, min(int(kmin), hi))
    k = int(rng.integers(lo, hi + 1))
    if k <= 0:
        return []
    return sorted(rng.choice(edge_ids_sorted, size=k, replace=False).tolist())


def _neighbor_closed(rng: np.random.Generator, base: List[str], edge_ids_sorted: List[str], kmax: int, kmin: int = 0) -> List[str]:
    s = set(base)
    n_edits = int(rng.integers(1, 3))
    for _ in range(n_edits):
        if s and rng.random() < 0.5:
            s.remove(rng.choice(sorted(list(s))))
        if len(s) < kmax:
            pool = [e for e in edge_ids_sorted if e not in s]
            if pool:
                s.add(str(rng.choice(pool)))
    while len(s) < max(0, int(kmin)):
        pool = [e for e in edge_ids_sorted if e not in s]
        if not pool:
            break
        s.add(str(rng.choice(pool)))
    return sorted(list(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumocfg", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--outdir", default="output_policy")
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--kmax", type=int, required=True)
    ap.add_argument("--kmin", type=int, default=0,
                    help="minimum number of closed edges for random/neighbor samples")
    ap.add_argument("--candidates-file", type=str, default=None,
                    help="optional file with one edge id per line; restricts closure candidates")
    ap.add_argument("--enforce-od-connectivity", action="store_true",
                    help="reject closure samples that disconnect any OD (BFS pre-check); "
                         "single-cut edges are excluded from candidates")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--protect", nargs="*", default=[])
    ap.add_argument("--protect-str", type=str, default="")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--auto-protect-endpoints", dest="auto_protect_endpoints", action="store_true")
    grp.add_argument("--no-auto-protect-endpoints", dest="auto_protect_endpoints", action="store_false")
    ap.set_defaults(auto_protect_endpoints=True)
    ap.add_argument("--sumo-bin", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--part-id", type=int, default=0)
    ap.add_argument("--sumo-seed-base", type=int, default=100000)
    ap.add_argument("--reroute-period", type=int, default=30)
    ap.add_argument("--time-to-teleport", type=int, default=-1)
    ap.add_argument("--time-to-teleport-highways", type=int, default=-1)

    ap.add_argument("--label-mode", choices=["cost", "benefit"], default="benefit",
                    help="cost: y=ATT_closure(lower better), benefit: y=ATT_base-ATT_closure(higher better)")
    ap.add_argument("--add-traffic-features", action="store_true",
                    help="Add no-closure SUMO traffic features to edge_attr.")
    ap.add_argument("--add-route-count-feature", action="store_true",
                    help="Add route-file edge count / OD path count feature when available.")
    ap.add_argument("--demand-scale", type=float, default=1.0,
                    help="Stored as an edge feature if traffic/demand features are enabled.")
    ap.add_argument("--normalize-edge-features", action="store_true",
                    help="Z-normalize edge features after feature construction.")
    ap.add_argument("--sample-strategy", choices=["random", "mixed"], default="mixed")
    ap.add_argument("--single-edge-frac", type=float, default=0.25)
    ap.add_argument("--neighbor-frac", type=float, default=0.25,
                    help="Fraction of samples generated around current best/worst random samples.")
    args = ap.parse_args()

    sumo_bin = args.sumo_bin or shutil.which("sumo")
    if not sumo_bin:
        raise RuntimeError("sumo executable not found; set --sumo-bin")

    shard_seed = int(args.seed) + int(args.part_id) * 10007
    set_all_seeds(shard_seed)
    rng = np.random.default_rng(shard_seed)

    edge_index, x_node, edge_attr, node_map, edge_map, edge_ids_all, node_feature_names, edge_feature_names = build_graph(args.net)
    E = edge_attr.shape[0]

    protect = set(args.protect or [])
    endpoints: List[str] = []
    if bool(getattr(args, "auto_protect_endpoints", True)):
        endpoints = extract_route_endpoints(args.sumocfg)
        endpoints_set = {e for e in endpoints if e in edge_map}
        protect |= endpoints_set
        if endpoints_set:
            print(f"[dataset] auto-protect endpoints: {len(endpoints_set)} -> {sorted(list(endpoints_set))}")

    edge_ids = [eid for eid in edge_map.keys() if eid not in protect]

    if getattr(args, "candidates_file", None):
        cand_set = {ln.strip() for ln in Path(args.candidates_file).read_text(encoding="utf-8").splitlines() if ln.strip()}
        before = len(edge_ids)
        missing = sorted(e for e in cand_set if e not in edge_map)
        edge_ids = [eid for eid in edge_ids if eid in cand_set]
        print(f"[dataset] candidates-file: {before} -> {len(edge_ids)} eligible "
              f"({len(missing)} listed ids not in net{': ' + ','.join(missing[:5]) if missing else ''})")

    feas_checker = None
    if getattr(args, "enforce_od_connectivity", False):
        from od_feasibility import FeasibilityChecker
        feas_checker = FeasibilityChecker(args.net, args.sumocfg)
        cuts = set(feas_checker.single_cut_edges(edge_ids))
        if cuts:
            edge_ids = [e for e in edge_ids if e not in cuts]
            print(f"[dataset] feasibility: removed {len(cuts)} single-cut edge(s) -> {len(edge_ids)} candidates")

    if args.limit and len(edge_ids) > args.limit:
        rng.shuffle(edge_ids)
        edge_ids = edge_ids[:args.limit]
    edge_ids_sorted = sorted(edge_ids)

    def sample_feasible(sampler, max_tries: int = 300):
        closed = sampler()
        if feas_checker is None:
            return closed
        tries = 1
        while closed and not feas_checker.feasible(closed) and tries < max_tries:
            closed = sampler()
            tries += 1
        if closed and not feas_checker.feasible(closed):
            closed = list(closed)
            while closed and not feas_checker.feasible(closed):
                closed.pop(int(rng.integers(0, len(closed))))
        return closed
    upstream_map = build_upstream_map(args.net)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    baseline_dir = outdir / "baseline"
    baseline_dir.mkdir(exist_ok=True)
    baseline_add = baseline_dir / "no_closure.add.xml"
    write_addfile(baseline_add, [], upstream_map)
    baseline_seed = int(args.sumo_seed_base) + int(args.part_id) * 1_000_000 - 1
    baseline_att, traffic_feat, traffic_names = run_baseline_features(
        sumo_bin, args.sumocfg, baseline_add, baseline_dir, edge_ids_all,
        sumo_seed=baseline_seed,
        reroute_period=int(args.reroute_period),
        time_to_teleport=int(args.time_to_teleport),
        time_to_teleport_highways=int(args.time_to_teleport_highways),
    )
    print(f"[dataset] baseline ATT(no closure): {baseline_att:.6f}")

    if args.add_traffic_features:
        edge_attr = np.concatenate([edge_attr, traffic_feat], axis=1).astype(np.float32)
        edge_feature_names += traffic_names
        demand_feat = np.full((E, 1), float(args.demand_scale), dtype=np.float32)
        edge_attr = np.concatenate([edge_attr, demand_feat], axis=1).astype(np.float32)
        edge_feature_names += ["demand_scale"]

    if args.add_route_count_feature:
        route_counts = compute_route_edge_counts(args.sumocfg, edge_map, E)
        edge_attr = np.concatenate([edge_attr, route_counts], axis=1).astype(np.float32)
        edge_feature_names += ["route_edge_count"]

    edge_feature_mean = None
    edge_feature_std = None
    if args.normalize_edge_features:
        edge_attr, edge_feature_mean, edge_feature_std = _norm_cols(edge_attr)

    def evaluate_closed(closed: List[str], sample_idx: int, sample_type: str) -> Dict:
        mask = np.zeros((E,), dtype=np.float32)
        for eid in closed:
            mask[edge_map[eid]] = 1.0
        run_dir = outdir / f"sim_{sample_idx:04d}"
        run_dir.mkdir(exist_ok=True)
        addfile = run_dir / "closures.add.xml"
        write_addfile(addfile, sorted(closed), upstream_map)
        sumo_seed = int(args.sumo_seed_base) + int(args.part_id) * 1_000_000 + sample_idx
        cost = run_sumo_score(
            sumo_bin, args.sumocfg, addfile, run_dir,
            sumo_seed=sumo_seed,
            reroute_period=int(args.reroute_period),
            time_to_teleport=int(args.time_to_teleport),
            time_to_teleport_highways=int(args.time_to_teleport_highways),
        )
        if args.label_mode == "benefit":
            y = float(baseline_att - cost)
        else:
            y = float(cost)
        return {
            "mask": torch.tensor(mask),
            "y": torch.tensor(float(y)),
            "cost": torch.tensor(float(cost)),
            "baseline_cost": torch.tensor(float(baseline_att)),
            "closed_edges": sorted(closed),
            "sample_type": sample_type,
        }

    samples: List[Dict] = []
    total = int(args.samples)
    if args.sample_strategy == "mixed":
        n_single = min(len(edge_ids_sorted), int(round(total * max(0.0, args.single_edge_frac))))
        n_neighbor = int(round(total * max(0.0, args.neighbor_frac)))
        n_random = max(0, total - n_single - n_neighbor)
    else:
        n_single = 0; n_neighbor = 0; n_random = total

    shuffled_edges = edge_ids_sorted[:]
    rng.shuffle(shuffled_edges)
    for eid in shuffled_edges[:n_single]:
        samples.append(evaluate_closed([eid], len(samples), "single_edge"))
        print(f"[dataset] {len(samples)}/{total} single_edge")

    for _ in range(n_random):
        closed = sample_feasible(lambda: _choose_random_closed(rng, edge_ids_sorted, int(args.kmax), int(args.kmin)))
        samples.append(evaluate_closed(closed, len(samples), "random"))
        print(f"[dataset] {len(samples)}/{total} random")

    if n_neighbor > 0 and samples:
        ys = np.array([float(s["y"]) for s in samples], dtype=np.float64)
        if args.label_mode == "benefit":
            order = np.argsort(-ys)
        else:
            order = np.argsort(ys)
        q = max(1, len(order) // 5)
        seeds = [samples[int(i)] for i in order[:q].tolist()] + [samples[int(i)] for i in order[-q:].tolist()]
        for _ in range(n_neighbor):
            base = random.choice(seeds)["closed_edges"]
            closed = sample_feasible(lambda: _neighbor_closed(rng, base, edge_ids_sorted, int(args.kmax), int(args.kmin)))
            samples.append(evaluate_closed(closed, len(samples), "neighbor"))
            print(f"[dataset] {len(samples)}/{total} neighbor")

    while len(samples) < total:
        closed = sample_feasible(lambda: _choose_random_closed(rng, edge_ids_sorted, int(args.kmax), int(args.kmin)))
        samples.append(evaluate_closed(closed, len(samples), "random_fill"))
        print(f"[dataset] {len(samples)}/{total} random_fill")

    graph_blob = {
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "x_node": torch.tensor(x_node, dtype=torch.float32),
        "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
        "edge_ids_all": edge_ids_all,
        "edge_ids": edge_ids_sorted,
        "edge_map": edge_map,
        "node_map": node_map,
        "node_feature_names": node_feature_names,
        "edge_feature_names": edge_feature_names,
        "edge_feature_mean": edge_feature_mean,
        "edge_feature_std": edge_feature_std,
        "baseline_cost": float(baseline_att),
        "label_mode": args.label_mode,
    }
    torch.save(graph_blob, outdir / "graph_policy.pt")
    torch.save(samples, outdir / "dataset_policy.pt")

    meta = {
        "seed": int(args.seed),
        "part_id": int(args.part_id),
        "shard_seed": shard_seed,
        "sumo_seed_base": int(args.sumo_seed_base),
        "samples": len(samples),
        "kmax": int(args.kmax),
        "kmin": int(args.kmin),
        "candidates_file": getattr(args, "candidates_file", None),
        "n_candidates": len(edge_ids_sorted),
        "enforce_od_connectivity": bool(getattr(args, "enforce_od_connectivity", False)),
        "protect": sorted(list(protect)),
        "auto_protect_endpoints": bool(getattr(args, "auto_protect_endpoints", True)),
        "endpoints": endpoints,
        "sumocfg": str(args.sumocfg),
        "net": str(args.net),
        "label_mode": args.label_mode,
        "baseline_cost": float(baseline_att),
        "edge_feature_names": edge_feature_names,
        "add_traffic_features": bool(args.add_traffic_features),
        "add_route_count_feature": bool(args.add_route_count_feature),
        "normalize_edge_features": bool(args.normalize_edge_features),
        "sample_strategy": args.sample_strategy,
        "single_edge_frac": float(args.single_edge_frac),
        "neighbor_frac": float(args.neighbor_frac),
    }
    (outdir / "dataset_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved graph + {len(samples)} samples to: {outdir}")
    print(f"Meta: {outdir/'dataset_meta.json'}")


if __name__ == "__main__":
    main()
