from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Output policy directory that contains part0..partN-1")
    ap.add_argument("--parts", type=int, default=8, help="Number of parts (default: 8)")
    ap.add_argument("--dataset-name", default="dataset_policy.pt", help="Dataset filename (default: dataset_policy.pt)")
    ap.add_argument("--graph-name", default="graph_policy.pt", help="Graph filename (default: graph_policy.pt)")
    args = ap.parse_args()

    root = Path(args.root)
    samples = []
    metas = []

    for i in range(args.parts):
        part_dir = root / f"part{i}"
        part_path = part_dir / args.dataset_name
        if not part_path.exists():
            raise FileNotFoundError(f"Missing part dataset: {part_path}")
        print("loading", part_path)
        ds = torch.load(part_path, map_location="cpu")
        samples.extend(ds)

        meta_path = part_dir / "dataset_meta.json"
        if meta_path.exists():
            try:
                metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception:
                pass

    out_ds = root / args.dataset_name
    torch.save(samples, out_ds)
    print("saved merged dataset to", out_ds)

    src_graph = root / "part0" / args.graph_name
    dst_graph = root / args.graph_name
    if not src_graph.exists():
        raise FileNotFoundError(f"Missing graph file: {src_graph}")
    shutil.copy(src_graph, dst_graph)
    print("copied graph file from", src_graph, "to", dst_graph)

    merged = {"parts": args.parts, "total_samples": len(samples), "parts_meta": metas}
    if metas:
        keys_to_check = ["net", "sumocfg", "kmax", "protect"]
        for k in keys_to_check:
            vals = {json.dumps(m.get(k), sort_keys=True, ensure_ascii=False) for m in metas if k in m}
            if len(vals) > 1:
                merged.setdefault("warnings", []).append(f"Meta mismatch for key '{k}': {sorted(vals)}")
    (root / "dataset_meta_merged.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", root / "dataset_meta_merged.json")

    print("merged", len(samples), "samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
