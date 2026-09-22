import os
from pathlib import Path
import pandas as pd
ROOT = Path("."); M = Path(os.environ.get("K0_MERGED", "./merged_analysis_38_k0"))
OUT = Path(os.environ.get("K0_SD_OUT", "./figs_k0/signal_density.csv"))
sd = pd.read_csv(ROOT / "figs38/signal_density.csv", encoding="utf-8-sig")
att = pd.read_csv(M / "att_summary_by_mode_merged.csv", encoding="utf-8-sig")
g = att[att["mode"] != "none"].pivot(index="instance_id", columns="mode", values="mean_reduction_vs_none_%")
changed = []
for i, row in sd.iterrows():
    inst = row["instance_id"]
    for m in ("init", "mutation", "both"):
        new = float(g.loc[inst, m]); old = float(row[f"gain_{m}"])
        if abs(new - old) > 1e-9:
            changed.append((inst, m, round(old, 4), round(new, 4))); sd.at[i, f"gain_{m}"] = new
sd.to_csv(OUT, index=False)
print("wrote", OUT, "rows", len(sd)); print("changed:", *changed, sep="\n  ")
