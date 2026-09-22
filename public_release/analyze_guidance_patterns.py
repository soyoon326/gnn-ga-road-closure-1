from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_MODES = ["none", "init", "mutation", "both"]
COMPARE_MODES = ["init", "mutation", "both"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=r".\merged_analysis",
        help="merged_analysis folder or merged_analysis.zip path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r".\merged_analysis\extra_analysis",
        help="output folder for the analysis results",
    )
    return parser.parse_args()


def _read_csv_from_folder(folder: Path, name: str) -> pd.DataFrame:
    path = folder / name
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    return pd.read_csv(path)


def _read_csv_from_zip(zippath: Path, inner_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zippath) as zf:
        with zf.open(inner_name) as f:
            return pd.read_csv(f)


def load_merged_csv(input_path: Path, name: str) -> pd.DataFrame:
    if input_path.is_dir():
        return _read_csv_from_folder(input_path, name)
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        return _read_csv_from_zip(input_path, f"merged_analysis/{name}")
    raise FileNotFoundError(f"unsupported INPUT_PATH: {input_path}")


def parse_closed_edges(value: object) -> frozenset[str]:
    if pd.isna(value) or value is None:
        return frozenset()
    s = str(value).strip()
    if not s:
        return frozenset()
    return frozenset(x.strip() for x in s.split(",") if x.strip())


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def normalized_entropy_from_counts(counts: Iterable[int]) -> float:
    counts = np.asarray(list(counts), dtype=float)
    counts = counts[counts > 0]
    if len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    h = -(p * np.log(p)).sum()
    return float(h / np.log(len(counts)))


def hhi_from_counts(counts: Iterable[int]) -> float:
    counts = np.asarray(list(counts), dtype=float)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return np.nan
    p = counts / counts.sum()
    return float((p ** 2).sum())


def analyze_solution_overlap(seed_best: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = seed_best.copy()
    df = df[df["mode"].isin(TARGET_MODES)].copy()
    df["edge_set"] = df["closed_edges"].apply(parse_closed_edges)

    pair_rows = []
    summary_rows = []
    edge_rows = []

    for (instance_id, mode), g in df.groupby(["instance_id", "mode"], dropna=False):
        g = g.sort_values("seed")
        records = list(g[["seed", "edge_set", "closed_edges"]].itertuples(index=False, name=None))

        sims = []
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                s1, e1, _ = records[i]
                s2, e2, _ = records[j]
                sim = jaccard(e1, e2)
                sims.append(sim)
                pair_rows.append(
                    {
                        "instance_id": instance_id,
                        "mode": mode,
                        "seed_a": s1,
                        "seed_b": s2,
                        "jaccard": sim,
                    }
                )

        edge_counter: dict[str, int] = {}
        for _, edge_set, _ in records:
            for e in edge_set:
                edge_counter[e] = edge_counter.get(e, 0) + 1

        n_solutions = len(records)
        counts_sorted = sorted(edge_counter.values(), reverse=True)
        total_edge_uses = sum(counts_sorted)
        top1_share = counts_sorted[0] / total_edge_uses if total_edge_uses else np.nan
        top3_share = sum(counts_sorted[:3]) / total_edge_uses if total_edge_uses else np.nan

        summary_rows.append(
            {
                "instance_id": instance_id,
                "mode": mode,
                "n_seed_best_solutions": n_solutions,
                "mean_pairwise_jaccard": float(np.mean(sims)) if sims else np.nan,
                "median_pairwise_jaccard": float(np.median(sims)) if sims else np.nan,
                "min_pairwise_jaccard": float(np.min(sims)) if sims else np.nan,
                "max_pairwise_jaccard": float(np.max(sims)) if sims else np.nan,
                "n_unique_edges_in_seed_bests": len(edge_counter),
                "total_edge_uses_in_seed_bests": total_edge_uses,
                "edge_freq_hhi": hhi_from_counts(counts_sorted),
                "edge_freq_entropy_norm": normalized_entropy_from_counts(counts_sorted),
                "top1_edge_share": top1_share,
                "top3_edge_share": top3_share,
            }
        )

        for edge, count in sorted(edge_counter.items(), key=lambda x: (-x[1], x[0])):
            edge_rows.append(
                {
                    "instance_id": instance_id,
                    "mode": mode,
                    "edge": edge,
                    "count_in_seed_bests": count,
                    "share_in_seed_bests": count / n_solutions if n_solutions else np.nan,
                }
            )

    return pd.DataFrame(pair_rows), pd.DataFrame(summary_rows), pd.DataFrame(edge_rows)


def build_best_so_far_curve(aggregate: pd.DataFrame) -> pd.DataFrame:
    df = aggregate.copy()
    df = df[df["mode"].isin(TARGET_MODES)].copy()
    gen_best = (
        df.groupby(["instance_id", "mode", "seed", "gen"], dropna=False)["ATT"]
        .min()
        .reset_index(name="gen_best_ATT")
        .sort_values(["instance_id", "mode", "seed", "gen"])
    )
    gen_best["best_so_far_ATT"] = (
        gen_best.groupby(["instance_id", "mode", "seed"], dropna=False)["gen_best_ATT"].cummin()
    )
    return gen_best


def analyze_convergence(aggregate: pd.DataFrame, seed_best: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve = build_best_so_far_curve(aggregate)
    finals = seed_best[["instance_id", "mode", "seed", "ATT"]].rename(columns={"ATT": "final_best_ATT"})
    curve = curve.merge(finals, on=["instance_id", "mode", "seed"], how="left")

    rows = []
    for (instance_id, mode, seed), g in curve.groupby(["instance_id", "mode", "seed"], dropna=False):
        g = g.sort_values("gen")
        gen1 = float(g.iloc[0]["best_so_far_ATT"])
        final_best = float(g.iloc[-1]["final_best_ATT"])
        denom = gen1 - final_best

        if abs(denom) < 1e-12:
            first_hit_95 = int(g.iloc[0]["gen"])
            progress_auc = 0.0
        else:
            progress = (gen1 - g["best_so_far_ATT"]) / denom
            progress = progress.clip(lower=0.0, upper=1.0)
            hit = g.loc[progress >= 0.95, "gen"]
            first_hit_95 = int(hit.iloc[0]) if len(hit) else int(g.iloc[-1]["gen"])
            progress_auc = float(progress.mean())

        first_improve = g.loc[g["best_so_far_ATT"] < gen1, "gen"]
        first_improve_gen = int(first_improve.iloc[0]) if len(first_improve) else np.nan

        rows.append(
            {
                "instance_id": instance_id,
                "mode": mode,
                "seed": seed,
                "gen1_best_ATT": gen1,
                "final_best_ATT": final_best,
                "total_improvement_from_gen1": gen1 - final_best,
                "first_improve_gen": first_improve_gen,
                "first_hit_95pct_own_final_gen": first_hit_95,
                "progress_auc_mean": progress_auc,
            }
        )

    seed_conv = pd.DataFrame(rows)
    mode_conv = (
        seed_conv.groupby(["instance_id", "mode"], dropna=False)
        .agg(
            n_seeds=("seed", "count"),
            mean_gen1_best_ATT=("gen1_best_ATT", "mean"),
            mean_final_best_ATT=("final_best_ATT", "mean"),
            mean_total_improvement_from_gen1=("total_improvement_from_gen1", "mean"),
            mean_first_improve_gen=("first_improve_gen", "mean"),
            mean_first_hit_95pct_own_final_gen=("first_hit_95pct_own_final_gen", "mean"),
            mean_progress_auc=("progress_auc_mean", "mean"),
        )
        .reset_index()
    )
    return seed_conv, mode_conv


def analyze_diversity_vs_gain(aggregate: pd.DataFrame, att_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = aggregate.copy()
    df = df[df["mode"].isin(TARGET_MODES)].copy()
    gen1 = df[df["gen"] == 1].copy()
    gen1["closed_edges_key"] = gen1["closed_edges"].fillna("<NONE>").astype(str)

    div = (
        gen1.groupby(["instance_id", "mode"], dropna=False)
        .agg(
            gen1_n=("closed_edges_key", "size"),
            gen1_unique_closed_edges=("closed_edges_key", "nunique"),
            gen1_best_ATT=("ATT", "min"),
            gen1_mean_ATT=("ATT", "mean"),
        )
        .reset_index()
    )
    div["gen1_unique_ratio"] = div["gen1_unique_closed_edges"] / div["gen1_n"]

    none_div = div[div["mode"] == "none"].copy().rename(
        columns={
            "gen1_unique_ratio": "none_gen1_unique_ratio",
            "gen1_best_ATT": "none_gen1_best_ATT",
            "gen1_mean_ATT": "none_gen1_mean_ATT",
        }
    )[["instance_id", "none_gen1_unique_ratio", "none_gen1_best_ATT", "none_gen1_mean_ATT"]]

    merged = div.merge(none_div, on="instance_id", how="left")
    merged = merged.merge(
        att_summary[["instance_id", "mode", "mean_ATT", "mean_reduction_vs_none_%"]],
        on=["instance_id", "mode"],
        how="left",
    )

    merged["delta_gen1_unique_ratio_vs_none"] = merged["none_gen1_unique_ratio"] - merged["gen1_unique_ratio"]
    merged["delta_gen1_best_ATT_vs_none"] = merged["none_gen1_best_ATT"] - merged["gen1_best_ATT"]
    merged["delta_gen1_mean_ATT_vs_none"] = merged["none_gen1_mean_ATT"] - merged["gen1_mean_ATT"]

    scatter = merged[merged["mode"].isin(COMPARE_MODES)].copy()

    corr_rows = []
    for mode, g in scatter.groupby("mode"):
        corr_rows.append(
            {
                "mode": mode,
                "n_instances": len(g),
                "corr_gain_vs_diversity_drop_spearman": g["mean_reduction_vs_none_%"].corr(
                    g["delta_gen1_unique_ratio_vs_none"], method="spearman"
                ),
                "corr_gain_vs_gen1_best_gain_spearman": g["mean_reduction_vs_none_%"].corr(
                    g["delta_gen1_best_ATT_vs_none"], method="spearman"
                ),
                "corr_gain_vs_gen1_mean_gain_spearman": g["mean_reduction_vs_none_%"].corr(
                    g["delta_gen1_mean_ATT_vs_none"], method="spearman"
                ),
            }
        )

    return scatter, pd.DataFrame(corr_rows)


def build_extra_summary(
    overlap_summary: pd.DataFrame,
    convergence_summary: pd.DataFrame,
    diversity_scatter: pd.DataFrame,
) -> pd.DataFrame:
    ov = overlap_summary.pivot(index="instance_id", columns="mode")
    ov.columns = [f"{mode}_{metric}" for metric, mode in ov.columns]
    ov = ov.reset_index()

    cv = convergence_summary.pivot(index="instance_id", columns="mode")
    cv.columns = [f"{mode}_{metric}" for metric, mode in cv.columns]
    cv = cv.reset_index()

    dv = diversity_scatter[[
        "instance_id",
        "mode",
        "mean_reduction_vs_none_%",
        "delta_gen1_unique_ratio_vs_none",
        "delta_gen1_best_ATT_vs_none",
        "delta_gen1_mean_ATT_vs_none",
    ]].copy()
    dv = dv.pivot(index="instance_id", columns="mode")
    dv.columns = [f"{mode}_{metric}" for metric, mode in dv.columns]
    dv = dv.reset_index()

    return ov.merge(cv, on="instance_id", how="outer").merge(dv, on="instance_id", how="outer")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = load_merged_csv(input_path, "aggregate_merged.csv")
    seed_best = load_merged_csv(input_path, "seed_best_merged.csv")
    att_summary = load_merged_csv(input_path, "att_summary_by_mode_merged.csv")

    pairwise_jaccard, overlap_summary, edge_frequency = analyze_solution_overlap(seed_best)
    pairwise_jaccard.to_csv(output_dir / "pairwise_jaccard_seed_bests.csv", index=False, encoding="utf-8-sig")
    overlap_summary.to_csv(output_dir / "solution_overlap_summary.csv", index=False, encoding="utf-8-sig")
    edge_frequency.to_csv(output_dir / "top_edge_frequency_in_seed_bests.csv", index=False, encoding="utf-8-sig")

    seed_conv, mode_conv = analyze_convergence(aggregate, seed_best)
    seed_conv.to_csv(output_dir / "convergence_by_seed.csv", index=False, encoding="utf-8-sig")
    mode_conv.to_csv(output_dir / "convergence_summary_by_mode.csv", index=False, encoding="utf-8-sig")

    diversity_scatter, diversity_corr = analyze_diversity_vs_gain(aggregate, att_summary)
    diversity_scatter.to_csv(output_dir / "diversity_vs_gain_scatter_ready.csv", index=False, encoding="utf-8-sig")
    diversity_corr.to_csv(output_dir / "diversity_vs_gain_correlations.csv", index=False, encoding="utf-8-sig")

    extra_summary = build_extra_summary(overlap_summary, mode_conv, diversity_scatter)
    extra_summary.to_csv(output_dir / "instance_extra_summary.csv", index=False, encoding="utf-8-sig")

    print(f"done: {output_dir}")
    for p in sorted(output_dir.glob("*.csv")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
