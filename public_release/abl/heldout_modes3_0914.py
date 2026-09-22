from pathlib import Path
import pandas as pd

R = Path("./reviewer_diagnostics/results")
PEN = 1e11
out = []
for iid in ["A_k4_3od_x1.0", "B_k9_3od_x1.0", "J_k9_6od_x1.0"]:
    srcs = [R / "reeval_39_k0_A3od" / "reeval39_long.csv"] if iid.startswith("A_k4_3od") else [R / "reeval_39_k0" / "reeval39_long.csv"]
    for src in srcs:
        d = pd.read_csv(src)
        d = d[d.instance_id == iid]
        seeds = sorted(d.sumo_seed.unique())
        pen = int((d.att >= PEN).sum())
        g = d[d.att < PEN].groupby(["mode", "seed"]).agg(att_multi=("att", "mean"), att_default=("att_default", "first"),
                                                          n_ok=("att", "size")).reset_index()
        print(f"== {iid} from {src.parent.name}: sumo seeds {seeds}, penalised evals {pen}")
        print(g.groupby("mode").agg(default_mean=("att_default", "mean"), heldout_mean=("att_multi", "mean"),
                                    heldout_sd=("att_multi", "std"), runs=("seed", "size"), min_ok=("n_ok", "min")).round(2))
        g["instance_id"] = iid
        out.append(g)
pd.concat(out).to_csv(Path("./manuscript_numbers_k0").joinpath("heldout_modes3.csv"), index=False)
