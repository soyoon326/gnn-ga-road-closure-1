from __future__ import annotations
import os
K0_MERGED = os.environ.get('K0_MERGED', './merged_analysis_38_k0')
K0_RUNSFX = os.environ.get('K0_RUNSFX', '_pm01k0')
K0_OUT = os.environ.get('K0_OUT', 'manuscript_numbers_k0')
K0_REEVAL = os.environ.get('K0_REEVAL', 'reeval_39_k0')

import csv
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("."); OUT = ROOT / K0_OUT; OUT.mkdir(exist_ok=True)
MODES = ["none", "init", "mutation", "both"]; GENS = (1, 2, 3, 5, 10, 15)
d = pd.read_csv(ROOT / "merged_analysis_38_k0/aggregate_merged.csv", encoding="utf-8-sig",
                usecols=["instance_id", "source_path", "mode", "seed", "gen", "indiv", "ATT", "Score", "k_closed", "closed_edges", "is_valid"])
d["score"] = np.where(d["is_valid"].astype(str).str.lower() == "true", d["ATT"], 1e12)
d["closed_edges"] = d["closed_edges"].fillna("").astype(str)
rows = []
for inst in ["A_k4_3od_x0.8", "A_k4_3od_x1.0", "A_k4_3od_x1.2"]:
    for rd in sorted((ROOT / f"runs_{inst}{K0_RUNSFX}").glob("*_seed*")):
        m, s = rd.name.rsplit("_seed", 1)
        for r in csv.DictReader(open(rd / "summary.csv", encoding="utf-8")):
            rows.append({"instance_id": inst, "source_path": str(rd), "mode": m, "seed": int(s), "gen": int(r["gen"]), "indiv": int(r["idx"]),
                         "ATT": float(r["score"]), "Score": float(r["score"]), "k_closed": int(r["k"]),
                         "closed_edges": r["closed_edges"].replace(";", ","), "is_valid": float(r["score"]) < 1e11, "score": float(r["score"])})
d = pd.concat([d[~d.instance_id.str.startswith("A_k4_3od")], pd.DataFrame(rows)], ignore_index=True)
d = d[d.instance_id != "A_k4_1od_x1.2"]
d["net"] = d["instance_id"].str[0]
d["full_population"] = d.instance_id.str.startswith("A_k4_3od")
print("rows", len(d), "instances", d.instance_id.nunique(), "modes", sorted(d["mode"].unique()), "gens", d.gen.min(), d.gen.max())
print("A_3od sources:", sorted(set(Path(p).parts[-2] for p in d[d.instance_id.str.startswith("A_k4_3od")].source_path.unique())))
cnt = d.groupby(["instance_id", "mode", "seed"]).size(); print("indiv per run: min", cnt.min(), "max", cnt.max(), "runs", len(cnt))

gmin = d.groupby(["instance_id", "mode", "seed", "gen"])["score"].min().reset_index()
gmin = gmin.sort_values(["instance_id", "mode", "seed", "gen"])
gmin["bsf"] = gmin.groupby(["instance_id", "mode", "seed"])["score"].cummin()
gmin.loc[gmin.bsf > 1e11, "bsf"] = np.nan

def curve_table(sub, label):
    rows = []
    for m in MODES:
        c = sub[sub["mode"] == m].groupby("gen")["bsf"].mean()
        rows.append({"group": label, "mode": m, **{f"g{g}": c.get(g, np.nan) for g in GENS}})
    return rows

def geno(s):
    return tuple(sorted(str(s).split(","))) if isinstance(s, str) and s and s != "nan" else ()
div_rows = []
for (inst, m, s, g), grp in d.groupby(["instance_id", "mode", "seed", "gen"]):
    sets = [geno(x) for x in grp["closed_edges"]]
    div_rows.append({"instance_id": inst, "net": inst[0], "mode": m, "seed": s, "gen": g, "n_indiv": len(sets),
                     "distinct_genotypes": len(set(sets)), "distinct_edges": len(set(e for st in sets for e in st)),
                     "mean_k": np.mean([len(st) for st in sets])})
div = pd.DataFrame(div_rows)
div.to_csv(OUT / "sec52_diversity_per_run_gen.csv", index=False)
gmin.to_csv(OUT / "sec52_bsf_per_run_gen.csv", index=False)

md = ["# Section 5.2 material rebuilt from the MAIN 38-instance runs (replaces the restored-density probe)\n",
      "Source: merged_analysis_38_k0/aggregate_merged.csv (every individual of every generation). Best-so-far uses the run's own default-seed scores (penalised individuals excluded).\n"]

groups = [("J_k9_6od_x1.0", gmin[gmin.instance_id == "J_k9_6od_x1.0"])] + \
         [(f"network {n} (mean over its instances)", gmin[gmin.instance_id.str[0] == n]) for n in "ABCDJ"] + \
         [("all 38 instances", gmin)]
md.append("## 1. Mean best-so-far ATT (s) by generation\n")
allrows = []
for label, sub in groups:
    rows = curve_table(sub, label); allrows += rows
    md.append(f"**{label}**\n\n| mode | " + " | ".join(f"g{g}" for g in GENS) + " |\n|---|" + "---|" * len(GENS))
    for r in rows: md.append(f"| {r['mode']} | " + " | ".join(f"{r[f'g{g}']:.1f}" for g in GENS) + " |")
    md.append("")
pd.DataFrame(allrows).to_csv(OUT / "sec52_curves.csv", index=False)

md.append("## 2. Generation-1 lead of guided initialisation and where it vanishes (per instance, default seed)\n")
md.append("lead_g = mean bsf(none) − mean bsf(init) at generation g (positive = init ahead); same for both. 'gone at' = first generation where the lead ≤ 0 (— if never).\n")
md.append("| instance | lead init g1 | init gone at | init lead g15 | lead both g1 | both gone at | both lead g15 |\n|---|---|---|---|---|---|---|")
lead_rows = []
for inst, sub in gmin.groupby("instance_id"):
    piv = sub.groupby(["mode", "gen"])["bsf"].mean().unstack("gen")
    r = {"instance": inst}
    for m in ("init", "both"):
        lead = piv.loc["none"] - piv.loc[m]
        gone = next((g for g in range(1, 16) if lead.get(g, np.nan) <= 0), None)
        r[f"{m}_lead_g1"] = lead.get(1, np.nan); r[f"{m}_gone_at"] = gone; r[f"{m}_lead_g15"] = lead.get(15, np.nan)
    lead_rows.append(r)
    md.append(f"| {inst} | {r['init_lead_g1']:+.1f} | {r['init_gone_at'] or '—'} | {r['init_lead_g15']:+.1f} | {r['both_lead_g1']:+.1f} | {r['both_gone_at'] or '—'} | {r['both_lead_g15']:+.1f} |")
L = pd.DataFrame(lead_rows); L.to_csv(OUT / "sec52_lead_per_instance.csv", index=False)
for m in ("init", "both"):
    pos1 = (L[f"{m}_lead_g1"] > 0).sum(); pos15 = (L[f"{m}_lead_g15"] > 0).sum(); gone = L[f"{m}_gone_at"].notna().sum()
    md.append(f"\n{m}: ahead of none at g1 in {pos1}/{len(L)} instances (median lead {L[f'{m}_lead_g1'].median():+.1f} s); "
              f"lead vanishes at some generation in {gone}/{len(L)}; still ahead at g15 in {pos15}/{len(L)} (median {L[f'{m}_lead_g15'].median():+.1f} s).")

md.append("\n## 3. Population diversity (mean over seeds; population 24)\n")
md.append("Generation-1 figures are exact for all 38 instances. Later generations: the merged file omits cache hits "
          "(mean 22.9 of 24 individuals listed from generation 2 on), so distinct-edge counts at g15 are lower bounds "
          "and genotype counts at g>=2 are only available for the three A_3od instances read from raw run logs.\n")
def div_table(sub, label):
    md.append(f"**{label}**\n\n| mode | distinct genotypes g1 (of 24) | distinct edges g1 | distinct edges g15 (lower bound) | mean k g1 | mean k g15 |\n|---|---|---|---|---|---|")
    for m in MODES:
        x = sub[sub["mode"] == m]
        f = lambda g, col: x[x.gen == g][col].mean()
        md.append(f"| {m} | {f(1,'distinct_genotypes'):.1f} | {f(1,'distinct_edges'):.1f} | {f(15,'distinct_edges'):.1f} | {f(1,'mean_k'):.2f} | {f(15,'mean_k'):.2f} |")
    md.append("")
div_table(div[div.instance_id == "J_k9_6od_x1.0"], "J_k9_6od_x1.0")
for n in "ABCDJ": div_table(div[div.net == n], f"network {n}")
div_table(div, "all 38 instances")
r1 = div[div.gen == 1].groupby(["instance_id", "mode"])["distinct_genotypes"].mean().unstack("mode")
e1 = div[div.gen == 1].groupby(["instance_id", "mode"])["distinct_edges"].mean().unstack("mode")
ratio = ((r1["init"] + r1["both"]) / 2) / ((r1["none"] + r1["mutation"]) / 2)
eratio = ((e1["init"] + e1["both"]) / 2) / ((e1["none"] + e1["mutation"]) / 2)
md.append(f"Generation-1 diversity of the guided-initialisation modes relative to the unguided modes, per instance: "
          f"genotypes median ratio {ratio.median():.2f} (min {ratio.min():.2f}, max {ratio.max():.2f}; below 0.75 in {(ratio < 0.75).sum()}/{len(ratio)} instances); "
          f"distinct edges median ratio {eratio.median():.2f} (min {eratio.min():.2f}, max {eratio.max():.2f}).")
md.append("\n**Full-population diversity by generation, A_3od pm=0.1 instances (raw logs, all 24 individuals)**\n")
md.append("| mode | genotypes g1 | g2 | g3 | g5 | g10 | g15 | edges g1 | edges g15 |\n|---|---|---|---|---|---|---|---|---|")
full = div[div.instance_id.str.startswith("A_k4_3od")]
for m in MODES:
    x = full[full["mode"] == m]; f = lambda g, col: x[x.gen == g][col].mean()
    md.append(f"| {m} | " + " | ".join(f"{f(g,'distinct_genotypes'):.1f}" for g in (1, 2, 3, 5, 10, 15)) + f" | {f(1,'distinct_edges'):.1f} | {f(15,'distinct_edges'):.1f} |")
print("A_3od source check:", d[d.instance_id.str.startswith("A_k4_3od")].source_path.str.contains("pm01").all(),
      "| listed per gen A_3od:", div[div.instance_id.str.startswith("A_k4_3od")].n_indiv.mean())
(OUT / "SEC52_MAINRUNS.md").write_text("\n".join(md), encoding="utf-8")
print("saved", OUT / "SEC52_MAINRUNS.md")
