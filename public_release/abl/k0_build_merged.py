import subprocess, sys
from pathlib import Path
import pandas as pd
ROOT = Path("."); SRC = ROOT / "merged_analysis_39_pm01"
RUNSFX = sys.argv[1] if len(sys.argv) > 1 else "_pm01k0"
OUT39 = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "merged_analysis_39_k0")
OUT38 = ROOT / (sys.argv[3] if len(sys.argv) > 3 else "merged_analysis_38_k0")
A3 = ["A_k4_3od_x0.8", "A_k4_3od_x1.0", "A_k4_3od_x1.2"]; DUP = "A_k4_1od_x1.2"
PAIRS = [("aggregate_merged.csv", "aggregate.csv"), ("seed_best_merged.csv", "seed_best.csv"),
         ("att_summary_by_mode_merged.csv", "tables/att_summary_by_mode.csv"), ("pairwise_vs_none_merged.csv", "tables/pairwise_vs_none.csv"),
         ("reason_counts_merged.csv", "tables/reason_counts.csv"), ("assumption_checks_merged.csv", "tables/assumption_checks.csv"),
         ("seed_best_preview_merged.csv", "tables/seed_best_preview.csv")]
META = ("instance_id", "source_file", "source_path")
for inst in A3:
    for _, rf in PAIRS:
        assert (ROOT / f"runs_{inst}{RUNSFX}" / "analysis" / rf).exists(), (inst, rf)
for out, keep_dup in ((OUT39, True), (OUT38, False)):
    out.mkdir(exist_ok=True)
    for mf, rf in PAIRS:
        m = pd.read_csv(SRC / mf, encoding="utf-8-sig", low_memory=False)
        cols = list(m.columns)
        m = m[~m.instance_id.isin(A3)]
        if not keep_dup: m = m[m.instance_id != DUP]
        frames = [m]
        for inst in A3:
            p = ROOT / f"runs_{inst}{RUNSFX}" / "analysis" / rf
            r = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
            assert list(r.columns) == [c for c in cols if c not in META], (mf, inst, list(r.columns)[:6])
            r.insert(0, "instance_id", inst); r.insert(1, "source_file", Path(rf).name); r.insert(2, "source_path", str(p))
            frames.append(r)
        res = pd.concat(frames, ignore_index=True)[cols]
        res.to_csv(out / mf, index=False, encoding="utf-8-sig")
        print(f"{out.name}/{mf}: {len(res)} rows, {res.instance_id.nunique()} instances")
    idx = pd.read_csv(SRC / "instance_index.csv", encoding="utf-8-sig")
    if not keep_dup: idx = idx[idx.instance_id != DUP]
    for inst in A3:
        for c in ("run_dir", "analysis_dir", "tables_dir"):
            idx.loc[idx.instance_id == inst, c] = idx.loc[idx.instance_id == inst, c].astype(str).str.replace(
                f"runs_{inst}", f"runs_{inst}{RUNSFX}", regex=False)
    idx.to_csv(out / "instance_index.csv", index=False, encoding="utf-8-sig")
    for f in ("policy_model_inventory.csv", "train_log_summary.csv"):
        if (SRC / f).exists():
            d = pd.read_csv(SRC / f, encoding="utf-8-sig")
            if not keep_dup and "instance_id" in d.columns: d = d[d.instance_id != DUP]
            d.to_csv(out / f, index=False, encoding="utf-8-sig")
subprocess.run([sys.executable, str(ROOT / "analyze_guidance_patterns.py"), "--input", str(OUT38), "--output", str(OUT38 / "extra_analysis")], check=True)
c = pd.read_csv(OUT38 / "extra_analysis/diversity_vs_gain_correlations.csv", encoding="utf-8-sig"); print(c.to_string())
print("done")
