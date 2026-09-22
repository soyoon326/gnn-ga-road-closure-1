import sys
from pathlib import Path
import pandas as pd
ROOT = Path("."); R = ROOT / "reviewer_diagnostics/results"
NEW = R / (sys.argv[1] if len(sys.argv) > 1 else "reeval_39_k0_A3od")
OUT = R / (sys.argv[2] if len(sys.argv) > 2 else "reeval_39_k0"); OUT.mkdir(exist_ok=True)
PEN = 1e11; A3 = ["A_k4_3od_x0.8", "A_k4_3od_x1.0", "A_k4_3od_x1.2"]
old = pd.read_csv(R / "reeval_39/reeval39_long.csv"); new = pd.read_csv(NEW / "reeval39_long.csv")
assert sorted(new.instance_id.unique()) == A3, sorted(new.instance_id.unique())
assert list(new.columns) == list(old.columns)
long = pd.concat([old[~old.instance_id.isin(A3)], new], ignore_index=True)
def per_run(df):
    ok = df[df.att < PEN]
    return ok.groupby(["instance_id", "mode", "seed"]).agg(att_multi=("att", "mean"), att_default=("att_default", "first")).reset_index()
pr = per_run(long)
old_pr = pd.read_csv(R / "reeval_39/reeval39_per_run.csv")
chk = pr[~pr.instance_id.isin(A3)].merge(old_pr[~old_pr.instance_id.isin(A3)], on=["instance_id", "mode", "seed"], suffixes=("", "_old"))
assert len(chk) == len(old_pr[~old_pr.instance_id.isin(A3)])
assert (chk.att_multi - chk.att_multi_old).abs().max() < 1e-9 and (chk.att_default - chk.att_default_old).abs().max() < 1e-9
long.to_csv(OUT / "reeval39_long.csv", index=False); pr.to_csv(OUT / "reeval39_per_run.csv", index=False)
print("long rows", len(long), "per_run rows", len(pr), "instances", pr.instance_id.nunique())
print("penalised held-out evaluations:", int((long.att >= PEN).sum()), long[long.att >= PEN].groupby(["instance_id", "mode"]).size().to_dict())
print(pr[pr.instance_id.isin(A3)].groupby(["instance_id", "mode"])[["att_default", "att_multi"]].mean().round(2).to_string())
