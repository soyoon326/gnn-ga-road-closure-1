from __future__ import annotations

import argparse
import gzip
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_net(net_path: str):
    opener = gzip.open if str(net_path).endswith(".gz") else open
    edges = {}
    cur = {}
    lane_lengths = []
    with opener(net_path, "rb") as f:
        for event, elem in ET.iterparse(f, events=("start", "end")):
            if elem.tag == "edge":
                if event == "start":
                    cur = {
                        "id": elem.get("id"),
                        "from": elem.get("from"),
                        "to": elem.get("to"),
                        "function": (elem.get("function") or "").lower(),
                    }
                    lane_lengths = []
                else:
                    eid = cur.get("id")
                    if (eid and not eid.startswith(":") and cur["function"] != "internal"
                            and cur.get("from") and cur.get("to") and lane_lengths):
                        edges[eid] = (cur["from"], cur["to"], max(lane_lengths))
                    elem.clear()
            elif elem.tag == "lane" and event == "end":
                try:
                    lane_lengths.append(float(elem.get("length") or 0.0))
                except (TypeError, ValueError):
                    pass
    return edges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weight", choices=["length", "none"], default="length",
                    help="shortest-path weight: length (road length) or none (hop count)")
    ap.add_argument("--normalized", action="store_true", default=True)
    args = ap.parse_args()

    import networkx as nx

    edges = parse_net(args.net)
    G = nx.DiGraph()
    parallel: dict[tuple, list[str]] = {}
    for eid, (u, v, length) in edges.items():
        parallel.setdefault((u, v), []).append(eid)
        w = length if args.weight == "length" else 1.0
        if not G.has_edge(u, v) or G[u][v]["weight"] > w:
            G.add_edge(u, v, weight=w)

    print(f"[betweenness] nodes={G.number_of_nodes()} arcs={G.number_of_edges()} "
          f"(drivable edges={len(edges)})")
    eb = nx.edge_betweenness_centrality(
        G, weight=("weight" if args.weight == "length" else None), normalized=True)

    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="") as f:
        f.write("edge_id,score\n")
        n = 0
        for (u, v), score in eb.items():
            for eid in parallel.get((u, v), []):
                f.write(f"{eid},{score:.10f}\n")
                n += 1
    vals = list(eb.values())
    print(f"[betweenness] wrote {n} edge scores -> {out}")
    print(f"[betweenness] score range: min={min(vals):.6f} max={max(vals):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
