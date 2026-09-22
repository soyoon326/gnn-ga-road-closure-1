from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from itertools import islice
from pathlib import Path

import networkx as nx

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "py"))
import sumolib  # noqa: E402

NET = ROOT / "netJ" / "Netzmodell2.net.xml"
ROU = ROOT / "netJ" / "wildau_6od_x1.0.rou.xml"
CAND = ROOT / "netJ" / "wildau_6od_candidate_edges.txt"
PROT = ROOT / "netJ" / "wildau_6od_protected_edges.txt"


def od_pairs():
    out = []
    for fl in ET.parse(ROU).getroot().iter("flow"):
        out.append((fl.attrib["from"], fl.attrib["to"]))
    return out


def main():
    net = sumolib.net.readNet(str(NET))
    target = set(CAND.read_text().split())
    prot = set(PROT.read_text().split())
    ods = od_pairs()
    print(f"OD pairs: {len(ods)}")
    print(f"target candidate file: {len(target)} edges")
    print(f"protected file: {len(prot)} edges")

    G = nx.DiGraph()
    for e in net.getEdges():
        if e.getFunction() == "internal" or e.getLaneNumber() == 0:
            continue
        if not e.allows("passenger"):
            continue
        G.add_node(e.getID())
    for e in net.getEdges():
        if e.getID() not in G:
            continue
        for nxt in e.getOutgoing():
            if nxt.getID() in G:
                G.add_edge(e.getID(), nxt.getID(),
                           length=nxt.getLength(),
                           fftt=nxt.getLength() / max(1e-6, nxt.getSpeed()))
    print(f"edge-graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} arcs")

    for weight in ("fftt", "length"):
        for k in (3,):
            union = set()
            for o, d in ods:
                if o not in G or d not in G:
                    print(f"  [!] OD ({o},{d}) not in graph")
                    continue
                try:
                    paths = list(islice(
                        nx.shortest_simple_paths(G, o, d, weight=weight), k))
                except nx.NetworkXNoPath:
                    continue
                for p in paths:
                    union.update(p)
            minus = union - prot
            inter = len(minus & target)
            print(f"\n  weight={weight:6s} k={k}: union {len(union)}, "
                  f"minus protected {len(minus)}")
            print(f"    matches target on {inter} of {len(target)} "
                  f"({100*inter/len(target):.1f}%)")
            print(f"    in reconstruction only: {len(minus - target)}; "
                  f"in target only: {len(target - minus)}")

    print("\n" + "=" * 70)
    print("Inputs the reconstruction used: network geometry (length, speed,")
    print("connections, vClass) and the OD endpoints from the route file.")
    print("No tripinfo, no baseline flow, no ATT, no policy score.")


if __name__ == "__main__":
    raise SystemExit(main())
