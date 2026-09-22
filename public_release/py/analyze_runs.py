#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import json

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception as e:  # pragma: no cover
    print("[WARN] scipy not available; statistical tests will be skipped:", e, file=sys.stderr)
    stats = None

try:
    from lxml import etree as ET
except Exception:
    import xml.etree.ElementTree as ET  # type: ignore

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

try:
    import pyarrow  # noqa: F401
    PARQUET_AVAILABLE = True
except Exception:
    try:
        import fastparquet  # noqa: F401
        PARQUET_AVAILABLE = True
    except Exception:
        PARQUET_AVAILABLE = False


def save_parquet_or_csv(df: pd.DataFrame, path: Path, name: str):
    if PARQUET_AVAILABLE:
        df.to_parquet(path, index=False)
    else:
        csv_path = path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"[WARN] parquet engine missing; wrote {name} as CSV -> {csv_path}")

def read_json_if_exists(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


VALID_MODES = {"none", "init", "mutation", "both"}
LEGACY_MODES = {"graph", "model"}

RUN_DIR_RE = re.compile(r"^(?P<mode>none|init|mutation|both|graph|model)_seed(?P<seed>\d+)$", re.IGNORECASE)
INDIV_RE = re.compile(r"^g(?P<gen>\d{3})_i(?P<indiv>\d{3})$", re.IGNORECASE)


@dataclass
class IndivItem:
    mode: str
    seed: int
    gen: int
    indiv: int
    dir: Path


def normalize_mode(mode: str) -> str:
    return mode.lower()


def discover_runs(root: Path, modes: Optional[List[str]] = None) -> List[IndivItem]:
    items: List[IndivItem] = []
    modes_norm = None
    if modes:
        modes_norm = {normalize_mode(m) for m in modes}

    for d in root.iterdir():
        if not d.is_dir():
            continue
        m = RUN_DIR_RE.match(d.name)
        if not m:
            continue
        mode = normalize_mode(m.group("mode"))
        if modes_norm and mode not in modes_norm:
            if mode in LEGACY_MODES:
                print(f"[SKIP legacy mode] {d.name} (use --modes graph model to include)", file=sys.stderr)
            continue
        seed = int(m.group("seed"))
        for sub in d.iterdir():
            if not sub.is_dir():
                continue
            mi = INDIV_RE.match(sub.name)
            if not mi:
                continue
            gen = int(mi.group("gen"))
            indiv = int(mi.group("indiv"))
            items.append(IndivItem(mode=mode, seed=seed, gen=gen, indiv=indiv, dir=sub))
    items.sort(key=lambda x: (x.mode, x.seed, x.gen, x.indiv))
    return items


def parse_tripinfo(path: Path) -> Dict[str, float]:
    nan = float("nan")

    if not path.exists():
        return {
            "Arrivals": 0,
            "TTT": nan,
            "ATT": nan,
            "TimeLossSum": nan,
            "WaitingTimeSum": nan,
            "RouteLengthSum": nan,
            "MeanSpeed": nan,
        }

    arrivals = 0
    ttt_sum = 0.0
    time_loss = 0.0
    waiting = 0.0
    route_len = 0.0
    speed_sum = 0.0
    speed_cnt = 0

    try:
        ctx = ET.iterparse(str(path), events=("end",))
    except Exception:
        return {
            "Arrivals": 0,
            "TTT": nan,
            "ATT": nan,
            "TimeLossSum": nan,
            "WaitingTimeSum": nan,
            "RouteLengthSum": nan,
            "MeanSpeed": nan,
        }

    try:
        for ev, elem in ctx:
            tag = elem.tag.split("}")[-1]
            if tag == "tripinfo":
                arrivals += 1
                d = float(elem.get("duration", "nan"))
                tl = float(elem.get("timeLoss", elem.get("timeLossSum", "nan")))
                wt = float(elem.get("waitingTime", "nan"))
                rl = float(elem.get("routeLength", "nan"))
                sp = elem.get("speed")

                ttt_sum += 0.0 if math.isnan(d) else d
                time_loss += 0.0 if math.isnan(tl) else tl
                waiting += 0.0 if math.isnan(wt) else wt
                route_len += 0.0 if math.isnan(rl) else rl

                if sp is not None:
                    try:
                        speed_sum += float(sp)
                        speed_cnt += 1
                    except Exception:
                        pass

                elem.clear()
    except ET.ParseError as e:
        print(f"[WARN] corrupt tripinfo.xml, treated as NO_TRIPINFO: {path} ({e})")
        return {
            "Arrivals": 0,
            "TTT": nan,
            "ATT": nan,
            "TimeLossSum": nan,
            "WaitingTimeSum": nan,
            "RouteLengthSum": nan,
            "MeanSpeed": nan,
        }

    mean_speed = nan
    if speed_cnt > 0:
        mean_speed = speed_sum / speed_cnt
    elif ttt_sum > 0 and not math.isnan(route_len):
        mean_speed = route_len / ttt_sum

    if arrivals > 0:
        att = ttt_sum / arrivals
        ttt_out = ttt_sum
        tl_out = time_loss
        wt_out = waiting
        rl_out = route_len
    else:
        att = nan
        ttt_out = nan
        tl_out = nan
        wt_out = nan
        rl_out = nan

    return {
        "Arrivals": arrivals,
        "TTT": ttt_out,
        "ATT": att,
        "TimeLossSum": tl_out,
        "WaitingTimeSum": wt_out,
        "RouteLengthSum": rl_out,
        "MeanSpeed": mean_speed,
    }


def count_teleports(log_path: Path, summary_path: Optional[Path] = None) -> Optional[int]:
    if summary_path and summary_path.exists():
        try:
            tree = ET.parse(str(summary_path))
            root = tree.getroot()
            if root is not None:
                t = root.get("teleports")
                if t is not None:
                    return int(float(t))
                steps = list(root)
                if steps:
                    t2 = steps[-1].get("teleports")
                    if t2 is not None:
                        return int(float(t2))
        except Exception:
            pass
    if log_path and log_path.exists():
        try:
            cnt = 0
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "Teleporting vehicle" in line:
                        cnt += 1
            return cnt
        except Exception:
            return None
    return None


def parse_closed_edges(add_path: Path) -> Tuple[int, List[str]]:
    if not add_path.exists():
        return 0, []
    try:
        ctx = ET.iterparse(str(add_path), events=("end",))
    except Exception:
        return 0, []
    edges: List[str] = []
    try:
        for ev, elem in ctx:
            tag = elem.tag.split("}")[-1]
            eid = None
            if tag in {"edge", "closure", "close"}:
                eid = elem.get("id") or elem.get("edge")
            elif tag == "closingReroute":
                eid = elem.get("id") or elem.get("edge")
            if eid:
                edges.append(eid)
            elem.clear()
    except ET.ParseError as e:
        print(f"[WARN] corrupt closures.add.xml, using edges parsed so far: {add_path} ({e})")
    uniq = sorted(set(edges))
    return len(uniq), uniq


def summarize_individual(item: IndivItem) -> Dict[str, object]:
    ti = item.dir / "tripinfo.xml"
    if not ti.exists():
        alt = list(item.dir.glob("*tripinfo*.xml"))
        if alt:
            ti = alt[0]
    sm = item.dir / "summary.xml"
    lg = item.dir / "sumo.log"
    ad = item.dir / "closures.add.xml"
    mj = item.dir / "meta.json"
    rj = item.dir / "result.json"
    meta = read_json_if_exists(mj) or {}
    res  = read_json_if_exists(rj) or {}

    ti_metrics = parse_tripinfo(ti)
    teleports = count_teleports(lg, sm)
    k_closed, closed_edges = parse_closed_edges(ad)

    reason = res.get("reason")
    if not reason:
        reason = "OK" if not math.isnan(float(ti_metrics.get("ATT", float("nan")))) else "NO_TRIPINFO"

    wall_time = meta.get("wall_time_sec", np.nan)
    step = res.get("step", np.nan)
    min_expected_end = res.get("min_expected_end", np.nan)
    vaporized_cnt = res.get("vaporized_cnt", np.nan)
    demand_size = meta.get("demand_size", np.nan)
    score = res.get("score", np.nan)

    arrivals_override = res.get("arrivals")


    rec: Dict[str, object] = {
        "mode": item.mode,
        "seed": item.seed,
        "gen": item.gen,
        "indiv": item.indiv,
        "run_dir": str(item.dir),
        **ti_metrics,
        "Teleports": teleports if teleports is not None else np.nan,
        "k_closed": k_closed,
        "closed_edges": ",".join(closed_edges) if closed_edges else "",
        "Reason": reason,
        "WallTimeSec": wall_time,
        "Step": step,
        "MinExpectedEnd": min_expected_end,
        "VaporizedCnt": vaporized_cnt,
        "Demand": demand_size,
        "Score": score,

    }
    if arrivals_override is not None:
        rec["Arrivals"] = arrivals_override

    return rec


def aggregate(root: Path, modes: Optional[List[str]] = None, jobs: int = 0) -> pd.DataFrame:
    items = discover_runs(root, modes)
    if not items:
        print(f"[WARN] No run directories found under: {root}")
        return pd.DataFrame()

    recs: List[Dict[str, object]] = []
    if jobs and jobs > 1:
        import multiprocessing as mp
        with mp.Pool(processes=jobs) as pool:
            for rec in pool.imap_unordered(summarize_individual, items):
                recs.append(rec)
    else:
        for it in items:
            recs.append(summarize_individual(it))
    df = pd.DataFrame.from_records(recs)
    if not df.empty:
        df["mode"] = df["mode"].str.lower()
        if "Reason" in df.columns:
            df["is_valid"] = df["Reason"].fillna("").eq("OK") & ~df["ATT"].isna()
        else:
            df["is_valid"] = ~df["ATT"].isna()
    return df


def select_representative(df: pd.DataFrame, how: str = "best") -> pd.DataFrame:
    if df.empty:
        return df
    gcols = ["mode", "seed", "gen"]
    df_gen = (df[df["is_valid"]]
              .groupby(gcols, as_index=False)
              .agg(ATT_min=("ATT", "min")))
    df_gen["bsf"] = (df_gen
        .sort_values(["mode", "seed", "gen"])
        .groupby(["mode", "seed"])['ATT_min']
        .cummin())

    if how == "final":
        last_gen = df_gen.groupby(["mode", "seed"], as_index=False)["gen"].max()
        dfm = df.merge(last_gen, on=["mode", "seed", "gen"], how="inner")
        sel = dfm.loc[dfm.groupby(["mode", "seed"])['ATT'].idxmin()]
    elif how == "median-of-gen":
        med = df_gen.groupby(["mode", "seed"], as_index=False)["ATT_min"].median()
        dfm = df.merge(df_gen, on=["mode", "seed", "gen"], how="inner")
        dfm["dist"] = (dfm["ATT"] - dfm["ATT_min"]).abs()
        sel = dfm.loc[dfm.groupby(["mode", "seed"])['dist'].idxmin()]
    else:
        sel = df.loc[df.groupby(["mode", "seed"])['ATT'].idxmin()]

    return sel.reset_index(drop=True)


def wilcoxon_paired(a: np.ndarray, b: np.ndarray):
    if stats is None:
        return np.nan, np.nan, np.nan
    res = stats.wilcoxon(a, b, alternative='two-sided', zero_method='wilcox', correction=False, mode='auto')
    p = res.pvalue
    diff = np.median(a - b)
    z = stats.norm.isf(p / 2.0)
    z = float(z) * (-1.0 if diff < 0 else 1.0)
    r = z / math.sqrt(len(a))
    return p, z, r


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    pos = np.sum(diff > 0)
    neg = np.sum(diff < 0)
    n = len(diff)
    if n == 0:
        return np.nan
    return (pos - neg) / n


def bootstrap_ci(data: np.ndarray, func, n_boot: int = 2000, ci: float = 95.0, seed: int = 123) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(data)
    if n == 0:
        return (float('nan'), float('nan'))
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = func(data[idx])
    alpha = (100 - ci) / 2.0
    lo = np.percentile(vals, alpha)
    hi = np.percentile(vals, 100 - alpha)
    return float(lo), float(hi)


def common_language_effect(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if len(diff) == 0:
        return np.nan
    return float(np.mean(diff < 0))


def pairwise_vs_none(df_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    have = set(df_seed['mode'].unique())
    if 'none' not in have:
        return pd.DataFrame()
    base = df_seed[df_seed['mode'] == 'none'][['seed', 'ATT']].rename(columns={'ATT': 'ATT_none'})
    for mode in [m for m in ['init', 'mutation', 'both'] if m in have]:
        cur = df_seed[df_seed['mode'] == mode][['seed', 'ATT']].rename(columns={'ATT': f'ATT_{mode}'})
        mrg = base.merge(cur, on='seed', how='inner')
        a = mrg[f'ATT_{mode}'].to_numpy(dtype=float)
        b = mrg['ATT_none'].to_numpy(dtype=float)
        n = len(mrg)
        if n == 0:
            continue
        p, z, r = wilcoxon_paired(a, b)
        delta = cliffs_delta(a, b)
        lo, hi = bootstrap_ci(a - b, func=lambda x: cliffs_delta(x, np.zeros_like(x)))
        cles = common_language_effect(a, b)
        dz = float(np.mean(a - b)) / float(np.std(a - b, ddof=1)) if n > 1 else np.nan
        rows.append({
            'comparison': f'{mode} vs none',
            'n_pairs': n,
            'test': 'Wilcoxon',
            'p_raw': p,
            'Z': z,
            'r': r,
            "cliffs_delta": delta,
            "cliffs_delta_CI_low": lo,
            "cliffs_delta_CI_high": hi,
            'CLES': cles,
            'cohen_dz': dz,
        })
    df = pd.DataFrame(rows)
    if not df.empty and stats is not None:
        pvals = df['p_raw'].to_numpy(dtype=float)
        order = np.argsort(pvals)
        p_holm = np.empty_like(pvals)
        m = len(pvals)
        for rank, idx in enumerate(order, start=1):
            p_holm[idx] = min((m - rank + 1) * pvals[idx], 1.0)
        df['p_holm'] = p_holm
    return df


def plot_ttt_box_by_mode(df_seed: pd.DataFrame, outdir: Path):
    if df_seed.empty:
        return

    modes = [m for m in ['none', 'init', 'mutation', 'both']
             if m in set(df_seed['mode'])]

    plt.figure(figsize=(6, 4))

    mean_x = []
    mean_y = []

    for i, mode in enumerate(modes):
        sub = df_seed[df_seed['mode'] == mode]
        y = sub['ATT'].to_numpy(dtype=float)
        if y.size == 0:
            continue

        x = np.full_like(y, i, dtype=float)

        plt.scatter(x, y, alpha=0.7, s=30)

        mean_x.append(i)
        mean_y.append(float(y.mean()))

    if mean_x:
        plt.plot(mean_x, mean_y, marker='o', linewidth=2)

    plt.xticks(range(len(modes)), modes)
    plt.ylabel('Average travel time (s)')
    out = outdir / 'figures' / 'att_box_by_mode.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=600)
    plt.close()


def per_gen_minima(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = (df[df['is_valid']]
         .groupby(['mode', 'seed', 'gen'], as_index=False)
         .agg(ATT_min=('ATT', 'min')))
    g = g.sort_values(['mode', 'seed', 'gen'])
    g['bsf'] = g.groupby(['mode', 'seed'])['ATT_min'].cummin()
    return g


def plot_progress_mean_ci(df: pd.DataFrame, outdir: Path, focus_modes: Tuple[str, str] = ('none', 'both')):
    if df.empty:
        return
    g = per_gen_minima(df)
    modes_present = set(g['mode'])
    focus_modes = tuple([m for m in focus_modes if m in modes_present])
    if len(focus_modes) < 1:
        return

    mode_labels = {
        'none': 'baseline (GA only)',
        'both': 'GNN + GA',
        'init': 'GNN init only',
        'mutation': 'GNN-guided mutation',
    }

    plt.figure(figsize=(7.5, 5))
    max_generations = 0

    from typing import List
    for mode in focus_modes:
        sub = g[g['mode'] == mode]
        gens = sorted(sub['gen'].unique())
        means: List[float] = []
        ci_low: List[float] = []
        ci_high: List[float] = []
        for gen in gens:
            arr = sub[sub['gen'] == gen].groupby('seed')['bsf'].min().to_numpy()
            if len(arr) == 0:
                continue
            means.append(float(np.mean(arr)))
            lo, hi = bootstrap_ci(arr, func=np.mean)
            ci_low.append(lo)
            ci_high.append(hi)
        if not means:
            continue

        max_generations = max(max_generations, len(means))
        x = np.arange(1, len(means) + 1)
        label_mode = mode_labels.get(mode, mode)

        plt.plot(x, means, label=label_mode)
        plt.fill_between(x, ci_low, ci_high, alpha=0.2,
                         label=f"{label_mode} 95% bootstrap CI (mean)")

    if max_generations > 0:
        plt.xticks(np.arange(1, max_generations + 1))

    plt.xlabel('Generation')
    plt.ylabel("Best-so-far average travel time (s)")
    plt.legend()
    out = outdir / 'figures' / 'att_progress_by_gen.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Aggregate GA/GNN runs and compute statistics.')
    ap.add_argument('--root', type=Path, required=True, help='Runs root directory (contains <mode>_seedXX/)')
    ap.add_argument('--outdir', type=Path, default=None, help='Output directory (default: <root>/analysis)')
    ap.add_argument('--modes', nargs='*', default=['none', 'init', 'mutation', 'both'],
                    help='Modes to include (legacy graph/model are ignored unless explicitly listed)')
    ap.add_argument('--progress-modes', nargs='*', default=['none', 'both'],
                    help='Modes to plot in the generation progress figure (default: none both).')
    ap.add_argument('--select', choices=['best', 'final', 'median-of-gen'], default='best',
                    help='Representative selection rule per seed')
    ap.add_argument('--jobs', type=int, default=0, help='Parallel workers for parsing (0/1 = serial)')
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()

    outdir = args.outdir or (args.root / 'analysis')
    outdir.mkdir(parents=True, exist_ok=True)

    print('[1/5] Scanning & aggregating...')
    df = aggregate(args.root, args.modes, jobs=args.jobs)
    if df.empty:
        print('No data found. Exiting.')
        return

    agg_path = outdir / 'aggregate.parquet'
    save_parquet_or_csv(df, agg_path, "aggregate")

    print('[2/5] Selecting per-seed representatives...')
    df_seed = select_representative(df, how=args.select)
    seed_path = outdir / 'seed_best.parquet'
    save_parquet_or_csv(df_seed, seed_path, "seed_best")

    print('[3/5] Building summary tables...')
    tdir = outdir / 'tables'
    tdir.mkdir(exist_ok=True)
    if "Reason" in df.columns:
        (df.groupby(["mode", "Reason"])
           .size()
           .reset_index(name="n")
           .sort_values(["mode", "n"], ascending=[True, False])
           .to_csv(tdir / "reason_counts.csv", index=False))

    def ci_mean(x: np.ndarray) -> Tuple[float, float]:
        return bootstrap_ci(x, np.mean)

    rows = []
    for mode, sub in df_seed.groupby('mode'):
        att = sub['ATT'].to_numpy(dtype=float)
        mean = float(np.mean(att))
        sd = float(np.std(att, ddof=1)) if len(att) > 1 else float('nan')
        med = float(np.median(att))
        iqr = float(np.percentile(att, 75) - np.percentile(att, 25)) if len(att) > 0 else float('nan')
        lo, hi = ci_mean(att)
        rows.append({
            'mode': mode,
            'n_seeds': len(sub),
            'mean_ATT': mean,
            'sd_ATT': sd,
            'median_ATT': med,
            'IQR_ATT': iqr,
            'mean_ATT_95CI_low': lo,
            'mean_ATT_95CI_high': hi,
        })
    df_sum = pd.DataFrame(rows).sort_values('mode')
    if 'none' in set(df_seed['mode']):
        base_mean = df_sum.loc[df_sum['mode'] == 'none', 'mean_ATT'].values[0]
        df_sum['mean_reduction_vs_none_%'] = (base_mean - df_sum['mean_ATT']) / base_mean * 100.0
    df_sum.to_csv(tdir / 'att_summary_by_mode.csv', index=False)


    print('[4/5] Pairwise tests vs none...')
    df_tests = pairwise_vs_none(df_seed)
    if not df_tests.empty:
        cols = ['comparison', 'n_pairs', 'test', 'p_raw', 'p_holm', 'Z', 'r',
                'cliffs_delta', 'cliffs_delta_CI_low', 'cliffs_delta_CI_high',
                'CLES', 'cohen_dz']
        okcols = [c for c in cols if c in df_tests.columns]
        df_tests = df_tests[okcols]
        df_tests.to_csv(tdir / 'pairwise_vs_none.csv', index=False)

    print('[5/5] Assumption checks...')
    rows = []
    if stats is not None:
        for mode, sub in df_seed.groupby('mode'):
            if len(sub) >= 3:
                w, p = stats.shapiro(sub['ATT'].to_numpy(dtype=float))
                rows.append({'group': mode, 'metric': 'ATT',
                             'Shapiro_W': float(w), 'Shapiro_p': float(p)})
        if 'none' in set(df_seed['mode']):
            base = df_seed[df_seed['mode'] == 'none'][['seed', 'ATT']].rename(columns={'ATT': 'ATT_none'})
            for mode in ['init', 'mutation', 'both']:
                cur = df_seed[df_seed['mode'] == mode][['seed', 'ATT']].rename(columns={'ATT': f'ATT_{mode}'})
                mrg = base.merge(cur, on='seed', how='inner')
                if len(mrg) >= 3:
                    diff = (mrg[f'ATT_{mode}'] - mrg['ATT_none']).to_numpy(dtype=float)
                    w, p = stats.shapiro(diff)
                    rows.append({'group': f'paired_diff({mode}-none)', 'metric': 'ATT',
                                 'Shapiro_W': float(w), 'Shapiro_p': float(p)})
    df_assume = pd.DataFrame(rows)
    if not df_assume.empty:
        df_assume.to_csv(tdir / 'assumption_checks.csv', index=False)


    if not args.no_plots:
        plot_ttt_box_by_mode(df_seed, outdir)
        plot_progress_mean_ci(df, outdir, focus_modes=tuple(args.progress_modes))

    prev = df_seed[['seed', 'mode', 'gen', 'indiv',
                    'ATT', 'TTT', 'TimeLossSum',
                    'Arrivals', 'Teleports', 'k_closed', 'run_dir']].sort_values(['seed', 'mode'])
    prev.to_csv(tdir / 'seed_best_preview.csv', index=False)


    print('\nDone. Outputs under:', outdir)


if __name__ == '__main__':
    main()
