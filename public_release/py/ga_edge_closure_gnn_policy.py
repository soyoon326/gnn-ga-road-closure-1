from __future__ import annotations
import argparse
import csv
import gzip
import os
import random
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, List, Tuple, Dict, Optional, Set

import numpy as np

import time

START_RETRIES = int(os.environ.get("GA_SUMO_START_RETRIES", "4"))


def _as_file_uri_if_windows(path_str: str) -> str:
    return path_str
    if isinstance(path_str, str):
        if path_str.startswith("file:") or "://" in path_str:
            return path_str
        if re.match(r"^[A-Za-z]:[\\/]", path_str):
            return Path(path_str).resolve().as_uri()
    return path_str


def ensure_sumo_tools_on_path():
    sh = os.environ.get("SUMO_HOME")
    if sh:
        tools = str(Path(sh) / "tools")
        if tools not in sys.path:
            sys.path.append(tools)


def resolve_sumo_bin(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    cand = shutil.which("sumo")
    if cand:
        return cand
    sh = os.environ.get("SUMO_HOME")
    if sh:
        exe = Path(sh) / "bin" / ("sumo.exe" if os.name == "nt" else "sumo")
        if exe.exists():
            return str(exe)
    raise RuntimeError("sumo executable not found. Set --sumo-bin or SUMO_HOME.")


def import_traci_sumolib():
    ensure_sumo_tools_on_path()
    try:
        import traci  # type: ignore
        from sumolib.net import readNet  # type: ignore
        return traci, readNet
    except Exception as e:
        raise RuntimeError("could not import traci/sumolib. Check SUMO_HOME/tools or 'pip install eclipse-sumo'.") from e


def load_candidates(net_path: str, protect: Set[str], limit: Optional[int]=None) -> Tuple[List[str], Dict[str, List[str]], "sumolib.net.Net"]:
    _, readNet = import_traci_sumolib()
    net = readNet(_as_file_uri_if_windows(net_path))

    def is_drivable_edge(e) -> bool:
        eid = e.getID()
        if eid.startswith(":"): return False
        fn = e.getFunction()
        if fn and fn.lower() == "internal": return False
        if len(e.getLanes()) == 0: return False
        return True

    edges = [e for e in net.getEdges() if is_drivable_edge(e) and e.getID() not in protect]
    cand_ids = [e.getID() for e in edges]
    if limit and len(cand_ids) > limit:
        random.shuffle(cand_ids)
        cand_ids = cand_ids[:limit]

    upstream_map: Dict[str, List[str]] = {}
    for e in edges:
        up: List[str] = []
        for ine in e.getFromNode().getIncoming():
            if ine.getID().startswith(":"): continue
            up.append(ine.getID())
        upstream_map[e.getID()] = up if up else [e.getID()]

    return cand_ids, upstream_map, net


def write_closures_additional(path: Path, closed: Sequence[str], upstream_map: Dict[str, List[str]], begin: int = 0, end: int = 86400) -> None:
    add = ET.Element("additional")
    if closed:
        notify_edges: set[str] = set()
        for e in closed:
            notify_edges.update(upstream_map.get(e, [e]))
        if notify_edges:
            rr = ET.SubElement(add, "rerouter", attrib={
                "id": "rr_close",
                "edges": " ".join(sorted(notify_edges)),
                "probability": "1.0",
            })
            iv = ET.SubElement(rr, "interval", attrib={"begin": str(begin), "end": str(end)})
            for e in closed:
                ET.SubElement(iv, "closingReroute", attrib={"id": e})
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(add).write(path, encoding="UTF-8", xml_declaration=True)


def _extract_route_endpoints(sumocfg_path: str) -> Set[str]:
    endpoints: Set[str] = set()
    try:
        cfg = ET.parse(sumocfg_path).getroot()
        input_node = cfg.find(".//input")
        if input_node is None:
            return endpoints
        route_files_node = input_node.find("route-files")
        if route_files_node is None:
            return endpoints
        value = route_files_node.get("value", "")
        if not value:
            return endpoints
        base = Path(sumocfg_path).parent
        files = [x.strip() for x in value.split(";") if x.strip()]

        def _iter_routes(root: ET.Element):
            for r in root.iter("route"):
                edges = r.get("edges")
                if not edges:
                    continue
                toks = [t for t in edges.strip().split() if t]
                if toks:
                    endpoints.add(toks[0])
                    endpoints.add(toks[-1])
            for t in root.iter("trip"):
                frm = t.get("from"); to = t.get("to")
                if frm: endpoints.add(frm)
                if to:  endpoints.add(to)

        for f in files:
            rp = base / f
            if not rp.exists():
                rp = Path(f)
                if not rp.exists():
                    continue
            if str(rp).endswith(".gz"):
                with gzip.open(rp, "rb") as g:
                    data = g.read()
                root = ET.fromstring(data)
            else:
                root = ET.parse(rp).getroot()
            _iter_routes(root)
    except Exception:
        pass
    return endpoints


def run_single_sumo_and_score(*,
                       sumo_bin: str,
                       sumocfg: str,
                       additional: Path,
                       outdir: Path,
                       reroute_period: int = 30,
                       step_limit: int = int(os.environ.get("GA_STEP_LIMIT", "200000")),
                       time_to_teleport: Optional[int] = None,
                       time_to_teleport_highways: Optional[int] = None,
                       emit_tripinfo: bool = True,
                       demand_size: Optional[int] = None,
                       sumo_seed: Optional[int] = None) -> float:
    traci, _ = import_traci_sumolib()
    tripinfo = outdir / "tripinfo.xml"
    sumo_cmd = [
        sumo_bin, "-c", sumocfg, "-a", str(additional),
        "--no-step-log", "true",
        "--tripinfo-output", str(tripinfo),
        "--tripinfo-output.write-unfinished", "true",
        "--device.rerouting.probability", "1.0",
        "--device.rerouting.period", str(reroute_period),
        "--duration-log.disable", "true",
    ]
    if time_to_teleport is not None:
        sumo_cmd += ["--time-to-teleport", str(time_to_teleport)]
    if time_to_teleport_highways is not None:
        sumo_cmd += ["--time-to-teleport.highways", str(time_to_teleport_highways)]
    if sumo_seed is not None:
        sumo_cmd += ["--seed", str(int(sumo_seed))]

    conn = None
    for attempt in range(START_RETRIES):
        conn = None
        label = f"ga_run_{os.getpid()}_{attempt}"
        try:
            traci.start(sumo_cmd, label=label)
            conn = traci.getConnection(label)
        except Exception as exc:
            try:
                traci.switch(label); traci.close()
            except Exception:
                pass
            if attempt + 1 < START_RETRIES:
                time.sleep(0.25 + random.random() * 0.75)
                continue
            print(f"[SUMO] start failed after {START_RETRIES} attempts: {exc}")
            return 1e12

        try:
            step = 0
            while conn.simulation.getMinExpectedNumber() > 0 and step < step_limit:
                conn.simulationStep(); step += 1
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            return 1e12
        break

    try:
        conn.close()
    except Exception:
        pass

    if not tripinfo.exists():
        return 1e12

    durations: List[float] = []
    n_vehicles = 0
    try:
        root = ET.parse(tripinfo).getroot()
        for ev in root.iter("tripinfo"):
            n_vehicles += 1

            if ev.attrib.get("vaporized", "false") == "true":
                return 1e12

            ds = ev.attrib.get("duration")
            if ds is None:
                continue
            d = float(ds)
            if d >= 0:
                durations.append(d)
    except Exception:
        return 1e12

    if demand_size is not None and n_vehicles < demand_size:
        return 1e12

    if not durations:
        return 1e12

    return float(np.mean(durations))


def try_load_policy(
    policy_model_path: Optional[str],
    policy_graph_path: Optional[str],
    net,
    candidate_ids: List[str],
    score_mode_arg: str = "auto",
) -> Optional[Tuple[Dict[str, float], str]]:
    if not policy_model_path:
        return None
    try:
        import torch
        from policy_gnn_model import PolicyGNN
        from torch_geometric.nn import GATv2Conv  # noqa: F401
    except Exception as e:
        print("[Policy] could not import torch/torch_geometric/policy_gnn_model; continuing without the policy (random).", e)
        return None

    ck = torch.load(policy_model_path, map_location="cpu")
    score_mode = score_mode_arg
    if score_mode == "auto":
        score_mode = str(ck.get("score_mode", "benefit" if ck.get("label_mode") == "benefit" else "risk"))
    if score_mode not in ("risk", "benefit"):
        print(f"[Policy] invalid score_mode={score_mode}; using risk.")
        score_mode = "risk"

    graph_path: Optional[Path] = Path(policy_graph_path) if policy_graph_path else None
    if graph_path is None and policy_model_path:
        inferred = Path(policy_model_path).parent / "graph_policy.pt"
        if inferred.exists():
            graph_path = inferred

    if graph_path is not None and graph_path.exists():
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        x_node_t = graph["x_node"].float()
        edge_index_t = graph["edge_index"].long()
        edge_attr_t = graph["edge_attr"].float()
        edge_map = graph.get("edge_map", {})
        if int(edge_attr_t.size(1)) != int(ck.get("in_edge", edge_attr_t.size(1))):
            print("[Policy] the number of edge features in graph_policy.pt does not match the checkpoint; policy not used.")
            print(f"         graph in_edge={edge_attr_t.size(1)}, checkpoint in_edge={ck.get('in_edge')}")
            return None
    else:
        nodes = net.getNodes()
        node_id_map = {n.getID(): i for i, n in enumerate(nodes)}
        N = len(nodes)
        deg_in  = np.zeros((N,1), dtype=np.float32)
        deg_out = np.zeros((N,1), dtype=np.float32)
        is_tls  = np.zeros((N,1), dtype=np.float32)
        for n in nodes:
            i = node_id_map[n.getID()]
            deg_in[i,0]  = len([e for e in n.getIncoming() if not e.getID().startswith(":")])
            deg_out[i,0] = len([e for e in n.getOutgoing() if not e.getID().startswith(":")])
            is_tls[i,0]  = 1.0 if n.getType() == "traffic_light" else 0.0
        x_node = np.concatenate([deg_in, deg_out, is_tls], axis=1)
        edges_all = [e for e in net.getEdges() if not e.getID().startswith(":") and (e.getFunction() or "") != "internal" and len(e.getLanes())>0]
        edge_map = {e.getID(): i for i, e in enumerate(edges_all)}
        E = len(edges_all)
        src = np.array([node_id_map[e.getFromNode().getID()] for e in edges_all], dtype=np.int64)
        dst = np.array([node_id_map[e.getToNode().getID()] for e in edges_all], dtype=np.int64)
        edge_index = np.stack([src, dst], axis=0)
        length   = np.array([e.getLength() for e in edges_all], dtype=np.float32).reshape(E,1)
        lanes    = np.array([len(e.getLanes()) for e in edges_all], dtype=np.float32).reshape(E,1)
        speed    = np.array([e.getSpeed() for e in edges_all], dtype=np.float32).reshape(E,1)
        priority = np.array([(e.getPriority() or 0) for e in edges_all], dtype=np.float32).reshape(E,1)
        from_tls = is_tls[src].astype(np.float32)
        to_tls   = is_tls[dst].astype(np.float32)
        edge_attr = np.concatenate([length, lanes, speed, priority, from_tls, to_tls], axis=1).astype(np.float32)
        if int(edge_attr.shape[1]) != int(ck.get("in_edge", edge_attr.shape[1])):
            print("[Policy] the checkpoint appears to be trained with traffic-aware features.")
            print("         --policy-graph <output_policy/graph_policy.pt> must be given; policy not used.")
            return None
        x_node_t = torch.tensor(x_node, dtype=torch.float32)
        edge_index_t = torch.tensor(edge_index, dtype=torch.long)
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

    sd = ck["state_dict"]
    hid = int(ck.get("hid", sd["n_lin.weight"].shape[0]))
    heads = int(ck.get("heads", sd["gat1.att"].shape[1] if "gat1.att" in sd else 4))
    dropout = float(ck.get("dropout", 0.0))
    use_layernorm = bool(ck.get("use_layernorm", False))

    try:
        model = PolicyGNN(ck["in_node"], ck["in_edge"], hid=hid, heads=heads, dropout=dropout, use_layernorm=use_layernorm)
    except TypeError:
        model = PolicyGNN(ck["in_node"], ck["in_edge"], hid=hid, heads=heads)
    model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        scores = model(x_node_t, edge_index_t, edge_attr_t).cpu().numpy()

    scored: Dict[str, float] = {}
    for eid in candidate_ids:
        idx = edge_map.get(eid, None)
        if idx is not None:
            scored[eid] = float(scores[idx])
    if not scored:
        print("[Policy] no scores for the candidate edges; continuing at random.")
        return None

    arr = np.array(list(scored.values()), dtype=np.float32)
    mu, sdv = float(arr.mean()), float(arr.std() + 1e-6)
    for k in scored:
        scored[k] = (scored[k] - mu) / sdv
    print(f"[Policy] loaded {len(scored)} scores | score_mode={score_mode} | graph={graph_path}")
    return scored, score_mode


def load_policy_scores_file(path: str, candidate_ids: List[str]) -> Tuple[Dict[str, float], str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    vals = np.array([float(v) for v in raw.values()], dtype=np.float64)
    mu, sd = float(vals.mean()), float(vals.std() + 1e-6)
    scored = {c: (float(raw[c]) - mu) / sd if c in raw else 0.0 for c in candidate_ids}
    print(f"[Policy] scores file {path}: {sum(c in raw for c in candidate_ids)}/{len(candidate_ids)} "
          f"candidates covered | score_mode=benefit")
    return scored, "benefit"


class SetConditionalPrior:

    def __init__(self, model_path: str, graph_path: str, candidates: List[str],
                 sample: int = 0):
        import torch
        from policy_gnn_model import PolicyGNN
        self.torch = torch
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        if not ck.get("set_conditional"):
            raise RuntimeError(f"{model_path} is not a set-conditional checkpoint")
        g = torch.load(graph_path, map_location="cpu", weights_only=False)
        self.x = g["x_node"].float()
        self.ei = g["edge_index"].long()
        self.ea = g["edge_attr"].float()
        self.N, self.E = int(self.x.size(0)), int(self.ea.size(0))
        if int(ck["in_edge"]) != self.E and int(ck["in_edge"]) != self.ea.size(1) + 1:
            raise RuntimeError("set model / graph feature mismatch")
        edge_map = g["edge_map"]
        self.candidates = list(candidates)
        self.cand_idx = np.array([int(edge_map.get(c, -1)) for c in candidates])
        sd = ck["state_dict"]
        hid = int(ck.get("hid", sd["n_lin.weight"].shape[0]))
        heads = int(ck.get("heads", sd["gat1.att"].shape[1] if "gat1.att" in sd else 4))
        self.model = PolicyGNN(ck["in_node"], ck["in_edge"], hid=hid, heads=heads,
                               dropout=0.0, use_layernorm=bool(ck.get("use_layernorm", False)))
        self.model.load_state_dict(sd)
        self.model.eval()
        self.sign = 1.0 if str(ck.get("score_mode", "risk")) == "benefit" else -1.0
        self.sample = int(sample)
        self.calls = 0
        self.seconds = 0.0
        print(f"[SetPrior] {model_path} | E={self.E} | candidates {int((self.cand_idx >= 0).sum())}"
              f"/{len(candidates)} in graph | sample={self.sample or 'all'}")

    def _set_scores(self, masks):
        torch = self.torch
        B = int(masks.size(0))
        out = []
        chunk = max(1, 250_000 // self.E)
        with torch.no_grad():
            for s in range(0, B, chunk):
                mb = masks[s:s + chunk]
                b = int(mb.size(0))
                x = self.x.repeat(b, 1)
                offs = (torch.arange(b) * self.N).repeat_interleave(self.E)
                ei = self.ei.repeat(1, b) + offs.unsqueeze(0)
                ea = torch.cat([self.ea.repeat(b, 1), mb.reshape(-1, 1)], dim=1)
                sc = self.model(x, ei, ea).view(b, self.E)
                out.append((sc * mb).sum(1) / mb.sum(1).clamp(min=1.0))
        return torch.cat(out)

    def scores_for(self, bits: Sequence[int]) -> Dict[str, float]:
        torch = self.torch
        t0 = time.perf_counter()
        base = torch.zeros(self.E)
        closed_pos = [i for i, b in enumerate(bits) if b == 1 and self.cand_idx[i] >= 0]
        open_pos = [i for i, b in enumerate(bits) if b == 0 and self.cand_idx[i] >= 0]
        for i in closed_pos:
            base[self.cand_idx[i]] = 1.0
        if self.sample and len(open_pos) > self.sample:
            open_pos = random.sample(open_pos, self.sample)
        variants = [base]
        for i in open_pos:
            v = base.clone(); v[self.cand_idx[i]] = 1.0; variants.append(v)
        for i in closed_pos:
            v = base.clone(); v[self.cand_idx[i]] = 0.0; variants.append(v)
        sc = self._set_scores(torch.stack(variants))
        gain = ((sc[1:] - sc[0]) * self.sign).numpy()
        good: Dict[str, float] = {}
        for j, i in enumerate(open_pos):
            good[self.candidates[i]] = float(gain[j])
        for j, i in enumerate(closed_pos):
            good[self.candidates[i]] = -float(gain[len(open_pos) + j])
        vals = np.array(list(good.values()), dtype=np.float64)
        mu, sd = (float(vals.mean()), float(vals.std() + 1e-6)) if len(vals) else (0.0, 1.0)
        out = {c: 0.0 for c in self.candidates}
        for c, v in good.items():
            out[c] = (v - mu) / sd
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        return out


def _normalize_scores(s):
    s = np.asarray(s, dtype=np.float32)
    m, sd = float(s.mean()), float(s.std())
    return (s - m) / (sd if sd > 1e-12 else 1.0)

def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)

def _policy_logits(scores, alpha: float):
    return alpha * _normalize_scores(scores)

def _mix_with_uniform(p, mix: float):
    p = np.asarray(p, dtype=np.float64)
    if p.size == 0:
        return p
    p = p / (p.sum() + 1e-12)
    mix = max(0.0, min(1.0, float(mix)))
    u = np.ones_like(p, dtype=np.float64) / float(len(p))
    q = mix * p + (1.0 - mix) * u
    return q / (q.sum() + 1e-12)

def _closure_good_scores(edges, policy: Optional[Dict[str, float]], score_mode: str):
    raw = np.array([policy.get(e, 0.0) if policy else 0.0 for e in edges], dtype=np.float32)
    return raw if score_mode == "benefit" else -raw


def _policy_set_score(closed, policy: Optional[Dict[str, float]], score_mode: str) -> float:
    if not closed:
        return float("-inf")
    return float(np.mean(_closure_good_scores(list(closed), policy, score_mode)))


def _wsample_without_replacement(indexes, probs, k):
    idx = list(indexes)
    p   = [max(1e-12, float(x)) for x in probs]
    out = []
    k = min(k, len(idx))
    for _ in range(k):
        s = sum(p); r = random.random() * s
        acc = 0.0
        j = 0
        for j, w in enumerate(p):
            acc += w
            if acc >= r:
                break
        out.append(idx[j])
        idx.pop(j); p.pop(j)
    return out

def _clamp_bits(bits: List[int], kmin: int, kmax: int):
    ones = [i for i,b in enumerate(bits) if b==1]
    zeros= [i for i,b in enumerate(bits) if b==0]
    k = len(ones)
    if k > kmax:
        random.shuffle(ones)
        for i in ones[:(k-kmax)]: bits[i]=0
    elif k < kmin:
        random.shuffle(zeros)
        for i in zeros[:(kmin-k)]: bits[i]=1
    return bits


@dataclass(frozen=True)
class GAParams:
    mode: str
    k: int
    kmin: int
    kmax: int
    p_init: float
    pc: float
    pm: float
    pop: int
    gen: int
    jobs: int
    begin: int
    end: int
    reroute_period: int
    seed: int
    gnn_usage: str
    policy_model: Optional[str]
    policy_graph: Optional[str]
    policy_score_mode: str
    policy_mix_init: float
    policy_mix_mut: float
    seed_ratio: float
    policy_alpha_init: float
    policy_alpha_mut: float
    policy_temp: float
    policy_jitter: int
    time_to_teleport: Optional[int]
    time_to_teleport_highways: Optional[int]
    emit_tripinfo: bool
    init_include: Optional[str] = None
    candidates_file: Optional[str] = None
    enforce_od_connectivity: bool = False
    policy_filter_frac: float = 0.0
    policy_temp_end: Optional[float] = None
    init_guided_frac: float = 1.0
    dedup_population: bool = False
    eval_seeds: Optional[List[int]] = None
    prescreen: int = 1
    prescreen_random: bool = False
    evaluator: str = "sumo"
    ue_iters: int = 10
    ue_tail: int = 5
    ue_trips: Optional[str] = None
    policy_set_model: Optional[str] = None
    set_cand_sample: int = 0
    policy_scores_file: Optional[str] = None


def init_population_fixed_guided(cands: List[str], k: int, pop: int,
                                 policy: Optional[Dict[str,float]], seed_ratio: float, jitter: int,
                                 score_mode: str = "risk") -> List[List[str]]:
    popu: List[List[str]] = []
    n_guided = int(pop * max(0.0, min(1.0, seed_ratio)))
    order = list(cands)
    if policy:
        good = {e: _closure_good_scores([e], policy, score_mode)[0] for e in order}
        order.sort(key=lambda e: good.get(e, 0.0), reverse=True)
    else:
        random.shuffle(order)
    base = order[:max(k, 0)]
    for _ in range(n_guided):
        s = set(base[:k])
        for _ in range(max(0, jitter)):
            if not order or len(s) == 0:
                break
            drop = random.choice(tuple(s)); s.remove(drop)
            add_pool = [e for e in order[k:k+20] if e not in s] or [e for e in order if e not in s]
            if add_pool:
                s.add(random.choice(add_pool))
        popu.append(sorted(s))
    while len(popu) < pop:
        popu.append(sorted(random.sample(cands, k)))
    return popu


def init_population_bits_guided(cands: List[str], p_init: float, pop: int,
                                kmin: int, kmax: int, policy: Optional[Dict[str,float]],
                                alpha_init: float, temp: float,
                                score_mode: str = "risk", mix: float = 1.0,
                                guided_frac: float = 1.0,
                                banned: Optional[np.ndarray] = None) -> List[List[int]]:
    m = len(cands)
    popu: List[List[int]] = []
    if policy:
        good_scores = _closure_good_scores(cands, policy, score_mode)
        logits = _policy_logits(good_scores, alpha_init) / max(1e-12, temp)
        p = _mix_with_uniform(_softmax(logits), mix)
        if banned is not None:
            p = np.asarray(p, dtype=np.float64).copy()
            p[np.asarray(banned, dtype=bool)] = 0.0
            s = p.sum()
            p = (p / s) if s > 0 else None
    else:
        p = None

    n_guided = int(round(pop * max(0.0, min(1.0, guided_frac)))) if p is not None else 0

    for i in range(pop):
        if p is not None and i < n_guided:
            k = random.randint(kmin, kmax)
            chosen = np.random.choice(np.arange(m), size=k, replace=False, p=p)
            bits = [0] * m
            for j in chosen:
                bits[int(j)] = 1
        else:
            bits = [1 if random.random() < p_init else 0 for _ in range(m)]
            bits = _clamp_bits(bits, kmin, kmax)
        popu.append(bits)
    return popu


def crossover_fixed(p1: Sequence[str], p2: Sequence[str], all_cands: Sequence[str], k: int, pc: float) -> Tuple[List[str], List[str]]:
    if random.random() > pc: return sorted(p1), sorted(p2)
    s1, s2 = set(p1), set(p2)
    common = list(s1 & s2)
    a_only = list(s1 - s2); random.shuffle(a_only)
    b_only = list(s2 - s1); random.shuffle(b_only)
    def build(first: List[str], second: List[str]) -> List[str]:
        child = list(common)
        for src in (first, second):
            for e in src:
                if len(child) >= k: break
                if e not in child: child.append(e)
        pool = [c for c in all_cands if c not in child]
        while len(child) < k and pool:
            x = random.choice(pool); pool.remove(x); child.append(x)
        child.sort(); return child
    return build(a_only, b_only), build(b_only, a_only)


def mutate_fixed_guided(ind: List[str], all_cands: Sequence[str], k: int, pm: float,
                        policy: Optional[Dict[str,float]], alpha: float, temp: float,
                        score_mode: str = "risk", mix: float = 1.0) -> List[str]:
    if random.random() > pm:
        return ind[:]
    closed = list(ind)
    closed_set = set(closed)
    open_edges = [e for e in all_cands if e not in closed_set]
    if k <= 0 or len(open_edges) == 0 or len(closed) == 0:
        return ind[:]
    m_swap = max(1, int(round(pm * k)))

    good_closed = _closure_good_scores(closed, policy, score_mode) if policy else np.zeros(len(closed), dtype=np.float32)
    good_open = _closure_good_scores(open_edges, policy, score_mode) if policy else np.zeros(len(open_edges), dtype=np.float32)

    p_open = _softmax(_policy_logits(-good_closed, alpha) / max(1e-12, temp))
    p_open = _mix_with_uniform(p_open, mix)
    idx_open = _wsample_without_replacement(range(len(closed)), p_open, m_swap)
    to_open = [closed[i] for i in idx_open]

    p_close = _softmax(_policy_logits(good_open, alpha) / max(1e-12, temp))
    p_close = _mix_with_uniform(p_close, mix)
    idx_close = _wsample_without_replacement(range(len(open_edges)), p_close, len(to_open))
    to_close = [open_edges[i] for i in idx_close]

    new_closed = [e for e in closed if e not in set(to_open)]
    new_closed.extend(to_close)
    if len(new_closed) > k:
        new_closed = new_closed[:k]
    return sorted(new_closed)


def crossover_bits(p1: List[int], p2: List[int], pc: float) -> Tuple[List[int], List[int]]:
    if random.random() > pc: return p1[:], p2[:]
    m = len(p1); c1=[0]*m; c2=[0]*m
    for i in range(m):
        if random.random() < 0.5: c1[i]=p1[i]; c2[i]=p2[i]
        else: c1[i]=p2[i]; c2[i]=p1[i]
    return c1, c2


def mutate_bits_guided(ind: List[int], pm: float, policy: Optional[Dict[str,float]],
                       cands: List[str], kmin: int, kmax: int,
                       alpha: float, temp: float,
                       score_mode: str = "risk", mix: float = 1.0,
                       banned: Optional[np.ndarray] = None) -> List[int]:
    n = len(ind)
    x = np.array(ind, dtype=np.uint8)
    m = max(1, int(round(pm * n)))
    good_scores = _closure_good_scores(cands, policy, score_mode) if policy else np.zeros(n, dtype=np.float32)
    idx_open = np.where(x == 0)[0]
    idx_close = np.where(x == 1)[0]
    m_close = int(round(m * (len(idx_open) / max(1, n))))
    m_open = m - m_close

    sel_close = []
    if len(idx_open) and m_close > 0:
        logits = _policy_logits(good_scores[idx_open], alpha) / max(1e-12, temp)
        p = _mix_with_uniform(_softmax(logits), mix)
        if banned is not None:
            p = np.asarray(p, dtype=np.float64).copy()
            p[np.asarray(banned, dtype=bool)[idx_open]] = 0.0
            s = p.sum()
            if s > 0:
                p = p / s
            else:
                p = _mix_with_uniform(_softmax(logits), mix)
        sel_close = _wsample_without_replacement(idx_open, p, m_close)

    sel_open = []
    if len(idx_close) and m_open > 0:
        logits = _policy_logits(-good_scores[idx_close], alpha) / max(1e-12, temp)
        p = _mix_with_uniform(_softmax(logits), mix)
        sel_open = _wsample_without_replacement(idx_close, p, m_open)

    y = x.copy()
    if sel_close:
        y[np.array(sel_close)] = 1
    if sel_open:
        y[np.array(sel_open)] = 0
    y = _clamp_bits(y.tolist(), kmin, kmax)
    return list(map(int, y))


def dedup_population_inplace(pop_list, mode: str, candidates: Sequence[str]) -> int:
    seen = set()
    n_swapped = 0
    for i, ind in enumerate(pop_list):
        key = tuple(ind)
        tries = 0
        while key in seen and tries < 20:
            ind = list(ind)
            if mode == "fixed":
                pool = [c for c in candidates if c not in ind]
                if not pool:
                    break
                ind[random.randrange(len(ind))] = random.choice(pool)
                ind = sorted(ind)
            else:
                ones = [j for j, b in enumerate(ind) if b]
                zeros = [j for j, b in enumerate(ind) if not b]
                if ones and zeros:
                    ind[random.choice(ones)] = 0
                    ind[random.choice(zeros)] = 1
                elif zeros:
                    ind[random.choice(zeros)] = 1
                else:
                    break
            key = tuple(ind)
            tries += 1
        if tries:
            pop_list[i] = list(ind) if mode != "fixed" else ind
            n_swapped += 1
        seen.add(key)
    return n_swapped


def fingerprint_closed(closed: Sequence[str]) -> str:
    return "|".join(sorted(closed))


def eval_task(task) -> Tuple[int, float, str, List[str]]:
    (idx, closed, context) = task
    run_dir = context["outdir"] / f"g{context['gen']:03d}_i{idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    add_path = run_dir / "closures.add.xml"
    write_closures_additional(add_path, closed, context["upstream_map"], begin=context["begin"], end=context["end"])

    if context.get("evaluator") == "ue":
        from ue_eval import ue_score
        score, _ = ue_score(closed, base_net=Path(context["net_path"]),
                            trips=Path(context["ue_trips"]), workdir=run_dir,
                            iters=int(context["ue_iters"]), tail=int(context["ue_tail"]),
                            demand_size=context.get("demand_size"), seed=None,
                            keep_files=False)
        return idx, float(score), str(run_dir), list(closed)

    def _one(seed_val, out_dir):
        return run_single_sumo_and_score(
            sumo_bin=context["sumo_bin"],
            sumocfg=context["sumocfg"],
            additional=add_path,
            outdir=out_dir,
            reroute_period=context["reroute_period"],
            time_to_teleport=context["time_to_teleport"],
            time_to_teleport_highways=context["time_to_teleport_highways"],
            emit_tripinfo=context["emit_tripinfo"],
            demand_size=context.get("demand_size"),
            sumo_seed=seed_val,
        )

    eval_seeds = context.get("eval_seeds")
    if not eval_seeds:
        score = _one(None, run_dir)
    else:
        scores = []
        for s in eval_seeds:
            sub = run_dir / f"s{int(s)}"
            sub.mkdir(parents=True, exist_ok=True)
            scores.append(_one(int(s), sub))
        score = float(sum(scores) / len(scores))
    return idx, score, str(run_dir), list(closed)


def run_ga(*, sumo_bin: str, sumocfg: str, net_path: str, protect: List[str], limit: Optional[int],
           params: GAParams, outdir: Path, auto_protect_dest: bool = True,
           demand_size: Optional[int] = None):

    random.seed(params.seed)
    np.random.seed(params.seed)
    import_traci_sumolib()

    outdir.mkdir(parents=True, exist_ok=True)
    summary_csv = outdir / "summary.csv"
    if not summary_csv.exists():
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["gen","idx","k","score","closed_edges","run_dir","tripinfo"])

    protect_set = set(protect or [])
    if auto_protect_dest:
        endpoints = _extract_route_endpoints(sumocfg)
        if endpoints:
            protect_set.update(endpoints)
            print(f"added protected edges (origins/destinations): {sorted(endpoints)}")

    candidates, upstream_map, net = load_candidates(net_path, protect_set, limit)

    if params.candidates_file:
        cand_set = {ln.strip() for ln in Path(params.candidates_file).read_text(encoding="utf-8").splitlines() if ln.strip()}
        before = len(candidates)
        candidates = [e for e in candidates if e in cand_set]
        print(f"[Candidates] restricted by file: {before} -> {len(candidates)}")
        if not candidates:
            raise RuntimeError(f"candidates-file leaves no eligible edges: {params.candidates_file}")

    feas_checker = None
    if params.enforce_od_connectivity:
        from od_feasibility import FeasibilityChecker
        feas_checker = FeasibilityChecker(net_path, sumocfg)
        cuts = set(feas_checker.single_cut_edges(candidates))
        if cuts:
            candidates = [e for e in candidates if e not in cuts]
            print(f"[Feasibility] removed {len(cuts)} single-cut edge(s) from candidates "
                  f"-> {len(candidates)} remain: {sorted(cuts)[:8]}{'...' if len(cuts) > 8 else ''}")
        print(f"[Feasibility] OD pairs: {len(feas_checker.od_pairs)}; "
              f"infeasible individuals will be repaired before evaluation")

    if params.mode == "fixed" and len(candidates) < params.k:
        raise RuntimeError(f"too few candidate edges: {len(candidates)} < k={params.k}")

    policy_bundle = try_load_policy(params.policy_model, params.policy_graph, net, candidates, params.policy_score_mode) if params.gnn_usage != 'none' else None
    if policy_bundle is None:
        policy_scores = None
        policy_score_mode = "risk"
    else:
        policy_scores, policy_score_mode = policy_bundle
    if params.policy_scores_file and params.gnn_usage != 'none':
        policy_scores, policy_score_mode = load_policy_scores_file(params.policy_scores_file, candidates)

    set_prior = None
    if params.policy_set_model and params.gnn_usage in ("mutation", "both"):
        gp = params.policy_graph or str(Path(params.policy_set_model).parent / "graph_policy.pt")
        set_prior = SetConditionalPrior(params.policy_set_model, gp, candidates, params.set_cand_sample)

    ue_trips_path = None
    ue_prefeas = None
    if params.evaluator == "ue":
        from ue_eval import strip_rerouting
        if not params.ue_trips:
            raise RuntimeError("--evaluator ue requires --ue-trips <route/trip file>")
        src = Path(params.ue_trips).resolve()
        ue_dir = Path.cwd() / "ue_work"
        ue_trips_path = strip_rerouting(src, ue_dir / f"{src.stem}.noreroute.xml")
        from od_feasibility import FeasibilityChecker
        ue_prefeas = feas_checker or FeasibilityChecker(net_path, sumocfg)
        print(f"[UE] evaluator=duaIterate iters={params.ue_iters} tail={params.ue_tail} "
              f"trips={ue_trips_path} | OD pairs {len(ue_prefeas.od_pairs)}")

    print(f"candidate edges: {len(candidates)} (limit={limit}) | protected edges: {sorted(protect_set)}")
    print(f"output directory: {outdir}")
    policy_used = params.gnn_usage if policy_scores is not None else 'none'
    print(f"mode: {params.mode} | policy used: {policy_used}")

    cache: Dict[str, float] = {}
    cache_hit = 0
    cache_miss = 0
    best_closed: Optional[List[str]] = None
    best_score = float("inf")

    banned_arr: Optional[np.ndarray] = None
    if policy_scores is not None and params.policy_filter_frac > 0:
        gs = _closure_good_scores(candidates, policy_scores, policy_score_mode)
        thr = np.quantile(gs, params.policy_filter_frac)
        banned_arr = np.asarray(gs <= thr, dtype=bool)
        print(f"[Diversity] B1 policy filter: {int(banned_arr.sum())}/{len(candidates)} edges banned from closing")

    if params.mode == "fixed":
        policy_for_init = policy_scores if params.gnn_usage in ("init","both") else None
        population = init_population_fixed_guided(candidates, params.k, params.pop, policy_for_init, params.seed_ratio, params.policy_jitter, policy_score_mode)
    else:
        policy_for_init = policy_scores if params.gnn_usage in ("init","both") else None
        population = init_population_bits_guided(candidates, params.p_init, params.pop, params.kmin, params.kmax,
                                                 policy_for_init, params.policy_alpha_init, params.policy_temp,
                                                 policy_score_mode, params.policy_mix_init,
                                                 params.init_guided_frac, banned_arr)

    if params.init_include:
        try:
            lines = [l.strip() for l in Path(params.init_include).read_text(encoding="utf-8").splitlines() if l.strip()]
            cand_idx = {e: i for i, e in enumerate(candidates)}
            injected = 0
            for line in lines:
                if injected >= params.pop:
                    break
                ids = [e for e in (t.strip() for t in line.split(",")) if e in cand_idx]
                if not ids:
                    continue
                if params.mode == "fixed":
                    if len(ids) < params.k:
                        continue
                    population[injected] = sorted(ids[:params.k])
                else:
                    bits = [0] * len(candidates)
                    for e in ids[:params.kmax]:
                        bits[cand_idx[e]] = 1
                    population[injected] = bits
                injected += 1
            print(f"[InitInclude] injected {injected} individual(s) from {params.init_include}")
        except Exception as e:
            print(f"[InitInclude] failed ({e}); using unmodified init population")

    if params.dedup_population:
        ns = dedup_population_inplace(population, params.mode, candidates)
        if ns:
            print(f"[Diversity] B4 init dedup: {ns} individual(s) perturbed")

    for gen in range(1, params.gen+1):
        if params.policy_temp_end is not None and params.gen > 1:
            _t = (gen - 1) / (params.gen - 1)
            temp_g = params.policy_temp + _t * (params.policy_temp_end - params.policy_temp)
        else:
            temp_g = params.policy_temp
        print(f"\n[generation {gen}/{params.gen}] evaluating... (policy_temp={temp_g:.3f})")
        eval_ctx = {
            "sumo_bin": sumo_bin, "sumocfg": sumocfg, "upstream_map": upstream_map,
            "begin": params.begin, "end": params.end, "reroute_period": params.reroute_period,
            "outdir": outdir, "gen": gen,
            "time_to_teleport": params.time_to_teleport,
            "time_to_teleport_highways": params.time_to_teleport_highways,
            "emit_tripinfo": params.emit_tripinfo,
            "demand_size": demand_size,
            "eval_seeds": params.eval_seeds,
            "evaluator": params.evaluator,
            "net_path": net_path,
            "ue_trips": str(ue_trips_path) if ue_trips_path else None,
            "ue_iters": params.ue_iters,
            "ue_tail": params.ue_tail,
        }

        tasks = []
        closures_for_eval = []
        cached_scores = {}
        if params.mode == "fixed":
            for i, ind in enumerate(population):
                closed = ind
                key = fingerprint_closed(closed)
                closures_for_eval.append(closed)
                if key in cache:
                    cached_scores[i] = cache[key]
                    cache_hit += 1
                else:
                    tasks.append((i, closed, eval_ctx))
                    cache_miss += 1
        else:
            repaired_total = 0
            for i, bits in enumerate(population):
                if feas_checker is not None:
                    repaired_total += feas_checker.repair_bits(bits, candidates, random)
                closed = [eid for eid,b in zip(candidates, bits) if b==1]
                closures_for_eval.append(closed)
                key = fingerprint_closed(closed)
                if key in cache:
                    cached_scores[i] = cache[key]
                elif ue_prefeas is not None and closed and not ue_prefeas.feasible(closed):
                    cache[key] = 1e12
                    cached_scores[i] = 1e12
                    cache_miss += 1
                else:
                    tasks.append((i, closed, eval_ctx))
                    cache_miss += 1
            if feas_checker is not None and repaired_total:
                print(f"[Feasibility] opened {repaired_total} edge bit(s) to restore OD connectivity")

        results = [None] * len(population)  # type: ignore
        for i, sc in cached_scores.items():
            results[i] = (i, sc, str(outdir / f"g{gen:03d}_i{i:03d}"), closures_for_eval[i])

        if tasks:
            if params.jobs > 1:
                from multiprocessing import Pool
                with Pool(processes=params.jobs) as pool:
                    for res in pool.imap_unordered(eval_task, tasks):
                        results[res[0]] = res
            else:
                for t in tasks:
                    res = eval_task(t)
                    results[res[0]] = res
            for r in results:
                i, sc, _, closed = r  # type: ignore
                cache[fingerprint_closed(closed)] = sc  # type: ignore

        with summary_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for idx, sc, run_dir, closed in results:  # type: ignore
                k = len(closed)
                tripinfo = str(Path(run_dir) / "tripinfo.xml")
                w.writerow([gen, idx, k, sc, ";".join(closed), run_dir, tripinfo])
                print(f"  individual {idx:02d}: k={k:02d} -> {sc:.3f} | closed: {closed}")
                if sc < best_score:
                    best_score = sc; best_closed = list(closed)

        gen_best = min(r[1] for r in results)  # type: ignore
        print(f"generation {gen} best: {gen_best:.3f} | overall best: {best_score:.3f} {best_closed}")

        if gen == 1 and gen_best >= 1e11:
            msg = ("[FATAL] all individuals in generation 1 received the penalty score. "
                   "Check --demand-size vs the actual number of vehicles in the route file, "
                   "and teleport/arrival settings.")
            print(msg)
            with open(outdir / "run.log", "a", encoding="utf-8", errors="ignore") as lf:
                lf.write(msg + "\n")
            raise SystemExit(2)

        scores = [r[1] for r in results]  # type: ignore
        with open(outdir / "gen_best.csv", "a", encoding="utf-8") as w:
            w.write(f"{gen},{gen_best:.6f}\n")

        def tournament():
            tsize = max(2, min(5, len(population)//6 or 2))
            idxs = random.sample(range(len(population)), tsize)
            best_idx = min(idxs, key=lambda j: scores[j])
            return population[best_idx][:] if params.mode=="fixed" else population[best_idx][:]

        new_pop = []
        elite_idx = min(range(len(population)), key=lambda j: scores[j])
        new_pop.append(population[elite_idx][:])

        policy_for_mutate = policy_scores if params.gnn_usage in ("mutation","both") else None
        do_prescreen = params.prescreen > 1 and policy_scores is not None

        def _spawn(child, fixed_mode: bool):
            best_ind, best_val = None, None
            pool = []
            for _ in range(params.prescreen if do_prescreen else 1):
                if fixed_mode:
                    cand = mutate_fixed_guided(list(child), candidates, params.k, params.pm,
                                               policy_for_mutate, params.policy_alpha_mut, temp_g,
                                               policy_score_mode, params.policy_mix_mut)
                    closed = cand
                else:
                    pm_policy, pm_mode = policy_for_mutate, policy_score_mode
                    if set_prior is not None:
                        pm_policy, pm_mode = set_prior.scores_for(child), "benefit"
                    cand = mutate_bits_guided(list(child), params.pm, pm_policy, candidates,
                                              params.kmin, params.kmax, params.policy_alpha_mut, temp_g,
                                              pm_mode, params.policy_mix_mut, banned_arr)
                    closed = [eid for eid, b in zip(candidates, cand) if b == 1]
                if not do_prescreen:
                    return cand
                pool.append(cand)
                v = _policy_set_score(closed, policy_scores, policy_score_mode)
                if best_val is None or v > best_val:
                    best_ind, best_val = cand, v
            return random.choice(pool) if params.prescreen_random else best_ind

        while len(new_pop) < params.pop:
            p1, p2 = tournament(), tournament()
            fixed_mode = params.mode == "fixed"
            if fixed_mode:
                c1, c2 = crossover_fixed(p1, p2, candidates, params.k, params.pc)
            else:
                c1, c2 = crossover_bits(p1, p2, params.pc)
            c1 = _spawn(c1, fixed_mode)
            if len(new_pop) < params.pop: new_pop.append(c1)
            c2 = _spawn(c2, fixed_mode)
            if len(new_pop) < params.pop: new_pop.append(c2)

        population = new_pop[:params.pop]
        if params.dedup_population:
            ns = dedup_population_inplace(population, params.mode, candidates)
            if ns:
                print(f"[Diversity] B4 dedup: {ns} duplicate(s) perturbed")

    best_file = outdir / "best.txt"
    with best_file.open("w", encoding="utf-8") as f:
        f.write(f"best_score={best_score:.6f}\n")
        f.write("closed_edges=" + (" ".join(best_closed) if best_closed else "") + "\n")
    print("\n=== final result ===")
    print(f"best score: {best_score:.3f}")
    print(f"edges to close: {best_closed}")
    print(f"output folder: {outdir}")
    total_lookups = cache_hit + cache_miss
    hit_ratio = cache_hit / total_lookups if total_lookups > 0 else 0.0
    print(f"Cache hits: {cache_hit}, misses: {cache_miss}, "
      f"hit ratio: {hit_ratio:.3f}")
    if set_prior is not None:
        print(f"[SetPrior] {set_prior.calls} conditional scorings, {set_prior.seconds:.1f} s total")


def parse_args():
    p = argparse.ArgumentParser(description="GA + Policy-GNN for SUMO edge closure (avg trip time only)")
    p.add_argument("--sumocfg", required=True)
    p.add_argument("--net", required=True)
    p.add_argument("--sumo-bin", default=None)

    p.add_argument("--gnn-usage", choices=["none","init","mutation","both"], default="both", help="where the policy GNN is used")
    p.add_argument("--mode", choices=["fixed","variable"], default="variable")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--kmin", type=int, default=0)
    p.add_argument("--kmax", type=int, default=8)
    p.add_argument("--p-init", dest="p_init", type=float, default=0.05)
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--gen", type=int, default=10)
    p.add_argument("--pc", type=float, default=0.9)
    p.add_argument("--pm", type=float, default=0.1)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--protect", nargs=argparse.REMAINDER, default=[],
                   help="Protected edge IDs. NOTE: keep this option LAST; it will consume all remaining tokens (allows IDs like -E41).")
    p.add_argument("--protect-str", type=str, default="")
    p.add_argument("--no_auto_protect_dest", action="store_true", help="Disable automatic protection of route start/end edges")
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--begin", type=int, default=0)
    p.add_argument("--end", type=int, default=86400)
    p.add_argument("--reroute-period", type=int, default=30)
    p.add_argument("--outdir", default=str(Path.cwd() / "output"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--policy-model", default=None, help="policy_model.pt path (without it the policy is not used)")
    p.add_argument("--policy-graph", default=None, help="graph_policy.pt saved at training time; required for traffic-aware features")
    p.add_argument("--policy-score-mode", choices=["auto", "risk", "benefit"], default="benefit",
                   help="risk: a low score favours closing; benefit: a high score favours closing")
    p.add_argument("--policy-mix-init", type=float, default=1.0,
                   help="weight of the GNN distribution against the uniform one in initialisation. 1 = GNN only, 0 = random")
    p.add_argument("--policy-mix-mut", type=float, default=1.0,
                   help="weight of the GNN distribution against the uniform one in mutation. 1 = GNN only, 0 = random")
    p.add_argument("--seed-ratio", dest="seed_ratio", type=float, default=0.5, help="share of policy-seeded initial individuals (fixed mode)")
    p.add_argument("--policy-alpha-init", type=float, default=+1.0, help="sign and strength of the policy scores in initialisation")
    p.add_argument("--policy-alpha-mut",  type=float, default=+1.0, help="sign and strength of the policy scores in mutation")
    p.add_argument("--policy-temp",       type=float, default=1.0,  help="softmax temperature (larger = closer to uniform)")
    p.add_argument("--policy-jitter",     type=int,   default=2,    help="fixed mode: random swaps per seeded individual")
    p.add_argument("--init-include", default=None,
                   help="file of solutions injected into the initial population (one comma-joined list of edge ids per line)")
    p.add_argument("--candidates-file", default=None,
                   help="restrict the closure candidates to the edges in this file (one edge id per line)")
    p.add_argument("--enforce-od-connectivity", action="store_true",
                   help="repair OD-disconnecting solutions before evaluation with a BFS check (single cut edges are removed from the candidates)")
    p.add_argument("--eval-seeds", type=int, default=0,
                   help="P1: SUMO seeds averaged per evaluation (0 = one run with the SUMO default seed)")
    p.add_argument("--eval-seed-base", type=int, default=200000,
                   help="P1: first SUMO seed of the average")
    p.add_argument("--prescreen", type=int, default=1,
                   help="P2: generate N offspring and evaluate only the one with the best policy score (1 = off)")
    p.add_argument("--prescreen-random", action="store_true",
                   help="P2 control: pick one of the N at random instead of the best by policy score")
    p.add_argument("--evaluator", choices=["sumo", "ue"], default="sumo",
                   help="sumo: one run with reactive routing, as in the article. ue: ATT after duaIterate equilibration of the network with the edges removed")
    p.add_argument("--ue-iters", type=int, default=10, help="duaIterate iterations")
    p.add_argument("--ue-tail", type=int, default=5, help="last iterations averaged into the score")
    p.add_argument("--ue-trips", default=None, help="trip/route file for UE (rerouting parameters are removed)")
    p.add_argument("--policy-set-model", default=None,
                   help="train_policy_set.py checkpoint; replaces the mutation prior with set-conditional marginal gains")
    p.add_argument("--set-cand-sample", type=int, default=0,
                   help="open edges sampled per mutation for set-conditional scoring (0 = all)")
    p.add_argument("--policy-scores-file", default=None,
                   help="json {edge_id: score} (higher favours closing), used instead of the GNN scores (oracle or heuristic)")
    p.add_argument("--policy-filter-frac", type=float, default=0.0,
                   help="B1: exclude the lowest-scored fraction of edges from close sampling (0 = off)")
    p.add_argument("--policy-temp-end", type=float, default=None,
                   help="B2: linear schedule from policy-temp to this value over the generations (unset = constant)")
    p.add_argument("--init-guided-frac", type=float, default=1.0,
                   help="B3: share of the initial population sampled from the policy, the rest uniformly at random")
    p.add_argument("--dedup-population", action="store_true",
                   help="B4: replace duplicate individuals in every generation to keep diversity")

    p.add_argument("--time-to-teleport", type=int, default=-1,
                   help="value passed to SUMO --time-to-teleport; -1 disables teleporting")
    p.add_argument("--time-to-teleport-highways", type=int, default=-1,
                   help="value passed to SUMO --time-to-teleport.highways")
    p.add_argument("--emit-tripinfo", action="store_true",
                   help="extra tripinfo options (tripinfo is always written for the score)")
    p.add_argument(
        "--demand-size", type=int, default=None,
        help="total demand (vehicles). If set, a run with fewer departures or any vehicle not arriving is penalised."
    )
    args = p.parse_args()

    prot: list[str] = []
    if getattr(args, "protect", None):
        prot.extend(list(args.protect))
        if prot and prot[0] == "--":
            prot = prot[1:]
    if getattr(args, "protect_str", ""):
        prot.extend(str(args.protect_str).split())
    args.protect = prot
    return args


def main():
    args = parse_args()
    sumo_bin = resolve_sumo_bin(args.sumo_bin)
    outdir = Path(args.outdir)
    limit = None if args.limit == 0 else args.limit

    params = GAParams(
        gnn_usage=args.gnn_usage,
        mode=args.mode, k=args.k, kmin=args.kmin, kmax=args.kmax, p_init=args.p_init,
        pc=args.pc, pm=args.pm, pop=args.pop, gen=args.gen, jobs=args.jobs,
        begin=args.begin, end=args.end, reroute_period=args.reroute_period, seed=args.seed,
        policy_model=args.policy_model, policy_graph=args.policy_graph,
        policy_score_mode=args.policy_score_mode,
        policy_mix_init=args.policy_mix_init, policy_mix_mut=args.policy_mix_mut,
        seed_ratio=args.seed_ratio,
        policy_alpha_init=args.policy_alpha_init, policy_alpha_mut=args.policy_alpha_mut,
        policy_temp=args.policy_temp, policy_jitter=args.policy_jitter,
        time_to_teleport=args.time_to_teleport,
        time_to_teleport_highways=args.time_to_teleport_highways,
        emit_tripinfo=bool(args.emit_tripinfo),
        init_include=getattr(args, "init_include", None),
        candidates_file=getattr(args, "candidates_file", None),
        enforce_od_connectivity=bool(getattr(args, "enforce_od_connectivity", False)),
        policy_filter_frac=float(getattr(args, "policy_filter_frac", 0.0)),
        policy_temp_end=getattr(args, "policy_temp_end", None),
        init_guided_frac=float(getattr(args, "init_guided_frac", 1.0)),
        dedup_population=bool(getattr(args, "dedup_population", False)),
        eval_seeds=([int(args.eval_seed_base) + i for i in range(int(args.eval_seeds))]
                    if int(getattr(args, "eval_seeds", 0)) > 0 else None),
        prescreen=int(getattr(args, "prescreen", 1)),
        prescreen_random=bool(getattr(args, "prescreen_random", False)),
        evaluator=str(getattr(args, "evaluator", "sumo")),
        ue_iters=int(getattr(args, "ue_iters", 10)),
        ue_tail=int(getattr(args, "ue_tail", 5)),
        ue_trips=getattr(args, "ue_trips", None),
        policy_set_model=getattr(args, "policy_set_model", None),
        set_cand_sample=int(getattr(args, "set_cand_sample", 0)),
        policy_scores_file=getattr(args, "policy_scores_file", None),
    )

    start = time.perf_counter()

    run_ga(
        sumo_bin=sumo_bin, sumocfg=args.sumocfg, net_path=args.net,
        protect=args.protect, limit=limit, params=params, outdir=outdir,
        auto_protect_dest=(not getattr(args, "no_auto_protect_dest", False)),
        demand_size=args.demand_size,
    )

    elapsed = time.perf_counter() - start
    print(f"Elapsed: {elapsed:.3f} sec")


if __name__ == "__main__":
    main()
