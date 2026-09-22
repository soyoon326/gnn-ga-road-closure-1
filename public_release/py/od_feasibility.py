from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def parse_net_edges(net_path: str) -> Dict[str, Tuple[str, str]]:
    edges: Dict[str, Tuple[str, str]] = {}
    cur_id = cur_from = cur_to = cur_func = None
    cur_lanes = 0
    for event, elem in ET.iterparse(str(net_path), events=("start", "end")):
        if elem.tag == "edge":
            if event == "start":
                cur_id, cur_from, cur_to = elem.get("id"), elem.get("from"), elem.get("to")
                cur_func = elem.get("function")
                cur_lanes = 0
            else:
                if (cur_id and not cur_id.startswith(":")
                        and (cur_func or "").lower() != "internal"
                        and cur_from and cur_to and cur_lanes > 0):
                    edges[cur_id] = (cur_from, cur_to)
                elem.clear()
        elif elem.tag == "lane" and event == "end":
            if cur_id is not None:
                cur_lanes += 1
    return edges


def _route_files_from_cfg(sumocfg: str) -> List[Path]:
    base = Path(sumocfg).parent
    try:
        root = ET.parse(str(sumocfg)).getroot()
    except Exception:
        return []
    node = root.find(".//input/route-files")
    if node is None:
        return []
    files: List[Path] = []
    for part in (node.get("value") or "").replace(";", ",").split(","):
        part = part.strip()
        if part:
            files.append((base / part).resolve())
    return files


def extract_od_pairs(sumocfg: str, valid_edges: Set[str]) -> List[Tuple[str, str]]:
    od: Set[Tuple[str, str]] = set()
    for rp in _route_files_from_cfg(sumocfg):
        if not rp.exists():
            continue
        try:
            if str(rp).endswith(".gz"):
                with gzip.open(rp, "rb") as g:
                    root = ET.fromstring(g.read())
            else:
                root = ET.parse(str(rp)).getroot()
        except Exception:
            continue
        for tag in ("trip", "flow"):
            for el in root.iter(tag):
                fe, te = el.get("from"), el.get("to")
                if fe and te and fe in valid_edges and te in valid_edges and fe != te:
                    od.add((fe, te))
        for el in root.iter("route"):
            toks = (el.get("edges") or "").split()
            if toks and toks[0] in valid_edges and toks[-1] in valid_edges and toks[0] != toks[-1]:
                od.add((toks[0], toks[-1]))
    return sorted(od)


def build_edge_adjacency(edges: Dict[str, Tuple[str, str]]) -> Dict[str, List[str]]:
    out_by_node: Dict[str, List[str]] = {}
    for eid, (u, _v) in edges.items():
        out_by_node.setdefault(u, []).append(eid)
    return {eid: out_by_node.get(v, []) for eid, (_u, v) in edges.items()}


class FeasibilityChecker:
    def __init__(self, net_path: str, sumocfg: str):
        self.edges = parse_net_edges(net_path)
        self.adj = build_edge_adjacency(self.edges)
        self.od_pairs = extract_od_pairs(sumocfg, set(self.edges.keys()))
        if not self.od_pairs:
            raise RuntimeError(f"no OD pairs extracted from {sumocfg}")
        if not self.feasible(()):
            raise RuntimeError("baseline network does not connect all OD pairs")

    def _has_path(self, closed: Set[str], src: str, dst: str) -> bool:
        if src in closed or dst in closed:
            return False
        if src == dst:
            return True
        seen = {src}
        q = deque([src])
        while q:
            cur = q.popleft()
            for nxt in self.adj.get(cur, ()):
                if nxt in closed or nxt in seen:
                    continue
                if nxt == dst:
                    return True
                seen.add(nxt)
                q.append(nxt)
        return False

    def feasible(self, closed: Iterable[str]) -> bool:
        cs = set(closed)
        return all(self._has_path(cs, fe, te) for fe, te in self.od_pairs)

    def single_cut_edges(self, eligible: Sequence[str]) -> List[str]:
        return [e for e in eligible if not self.feasible((e,))]

    def repair_bits(self, bits: List[int], candidates: Sequence[str], rng) -> int:
        closed_idx = [i for i, b in enumerate(bits) if b == 1]
        opened = 0
        while closed_idx and not self.feasible(candidates[i] for i in closed_idx):
            j = rng.randrange(len(closed_idx))
            bits[closed_idx[j]] = 0
            closed_idx.pop(j)
            opened += 1
        return opened
