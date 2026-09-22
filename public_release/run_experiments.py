from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List


def _q(s: str) -> str:
    s = str(s)
    if " " in s or "\t" in s:
        return f'"{s}"'
    return s


def run_cmd(cmd: List[str], cwd: Path, stdout_path: Path | None, dry_run: bool) -> None:
    printable = " ".join(_q(x) for x in cmd)
    print(f"[CMD] (cwd={cwd}) {printable}")
    if dry_run:
        return

    cwd.mkdir(parents=True, exist_ok=True)
    if stdout_path is None:
        subprocess.run(cmd, cwd=str(cwd), check=True)
        return

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stdout_path, "a", encoding="utf-8", errors="ignore") as f:
        subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, check=True)


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_experiment(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    for exp in cfg.get("experiments", []):
        if exp.get("name") == name:
            return exp
    names = [e.get("name") for e in cfg.get("experiments", [])]
    raise KeyError(f"Experiment '{name}' not found. Available: {names}")


def as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def as_bool(exp: Dict[str, Any], key: str, default: bool = False) -> bool:
    value = exp.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def add_flag(cmd: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def add_opt(cmd: List[str], flag: str, value: Any, *, skip_none: bool = True) -> None:
    if skip_none and value is None:
        return
    cmd += [flag, str(value)]


def step_dataset(exp: Dict[str, Any], root: Path, python_exe: str, dry_run: bool) -> None:
    out_policy = root / exp["out_policy"]
    out_policy.mkdir(parents=True, exist_ok=True)

    parts = int(exp.get("parts", 8))
    parallel_parts = int(exp.get("parallel_parts", 4))
    samples_per_part = int(exp.get("samples_per_part", 125))

    sumo_bin = exp["sumo_bin"]
    sumocfg = root / exp["sumocfg"]
    net = root / exp["net"]
    kmax = int(exp["kmax"])
    protect = as_list(exp.get("protect", []))

    dataset_seed = int(exp.get("dataset_seed", 1234))
    sumo_seed_base = int(exp.get("sumo_seed_base", 100000))

    label_mode = str(exp.get("dataset_label_mode", exp.get("label_mode", "benefit")))
    add_traffic_features = as_bool(exp, "add_traffic_features", True)
    add_route_count_feature = as_bool(exp, "add_route_count_feature", True)
    normalize_edge_features = as_bool(exp, "normalize_edge_features", True)

    sample_strategy = exp.get("sample_strategy", "mixed")
    single_edge_frac = exp.get("single_edge_frac", 0.25)
    neighbor_frac = exp.get("neighbor_frac", 0.25)
    reroute_period = exp.get("reroute_period", None)

    procs = []

    def start_one(part_id: int):
        part_dir = out_policy / f"part{part_id}"
        part_dir.mkdir(parents=True, exist_ok=True)
        log_path = part_dir / "dataset.log"

        cmd = [
            python_exe, "py\\make_policy_dataset.py",
            "--sumo-bin", str(sumo_bin),
            "--sumocfg", str(sumocfg),
            "--net", str(net),
            "--outdir", str(part_dir),
            "--samples", str(samples_per_part),
            "--kmax", str(kmax),
            "--seed", str(dataset_seed),
            "--part-id", str(part_id),
            "--sumo-seed-base", str(sumo_seed_base),
            "--label-mode", str(label_mode),
        ]

        dataset_kmin = exp.get("dataset_kmin", exp.get("kmin", None))
        if dataset_kmin is not None:
            cmd += ["--kmin", str(int(dataset_kmin))]

        if exp.get("candidates_file"):
            cmd += ["--candidates-file", str(root / str(exp["candidates_file"]))]

        add_flag(cmd, "--enforce-od-connectivity", as_bool(exp, "enforce_od_connectivity", False))

        add_flag(cmd, "--add-traffic-features", add_traffic_features)
        add_flag(cmd, "--add-route-count-feature", add_route_count_feature)
        add_flag(cmd, "--normalize-edge-features", normalize_edge_features)

        if sample_strategy is not None:
            add_opt(cmd, "--sample-strategy", sample_strategy)
        if single_edge_frac is not None:
            add_opt(cmd, "--single-edge-frac", single_edge_frac)
        if neighbor_frac is not None:
            add_opt(cmd, "--neighbor-frac", neighbor_frac)
        if reroute_period is not None:
            add_opt(cmd, "--reroute-period", reroute_period)

        if protect:
            cmd += ["--protect-str", " ".join(str(p) for p in protect)]

        if dry_run:
            print("DRY:", " ".join(_q(x) for x in cmd))
            return None, None

        lf = open(log_path, "w", encoding="utf-8", errors="ignore")
        p = subprocess.Popen(cmd, cwd=str(root), stdout=lf, stderr=subprocess.STDOUT)
        return p, lf

    next_part = 0
    while next_part < parts or procs:
        while next_part < parts and len(procs) < parallel_parts:
            p, lf = start_one(next_part)
            if p is not None:
                procs.append((p, lf))
            next_part += 1

        if dry_run:
            break

        still = []
        finished_any = False
        for p, lf in procs:
            ret = p.poll()
            if ret is None:
                still.append((p, lf))
            else:
                finished_any = True
                lf.close()
                if ret != 0:
                    raise RuntimeError(f"make_policy_dataset failed (exit {ret})")
        procs = still

        if not finished_any:
            time.sleep(0.2)


def step_merge(exp: Dict[str, Any], root: Path, python_exe: str, dry_run: bool) -> None:
    out_policy = root / exp["out_policy"]
    parts = int(exp.get("parts", 8))
    cmd = [python_exe, "merge_policy_parts.py", "--root", str(out_policy), "--parts", str(parts)]
    run_cmd(cmd, cwd=root, stdout_path=out_policy / "merge.log", dry_run=dry_run)


def step_train(exp: Dict[str, Any], root: Path, python_exe: str, dry_run: bool) -> None:
    out_policy = root / exp["out_policy"]
    epochs = int(exp.get("epochs", 60))

    train_seed = int(exp.get("train_seed", 42))
    deterministic = as_bool(exp, "train_deterministic", True)
    device = str(exp.get("train_device", "cpu"))

    model_out = root / exp.get("policy_model", str(Path(exp["out_policy"]) / "policy_model.pt"))

    label_mode = str(exp.get("train_label_mode", exp.get("label_mode", "auto")))
    hid = exp.get("hid", exp.get("train_hid", None))
    dropout = exp.get("dropout", exp.get("train_dropout", None))
    use_layernorm = as_bool(exp, "use_layernorm", as_bool(exp, "train_use_layernorm", False))

    trainer = str(exp.get("trainer", "v1")).lower()
    train_script = "py\\train_policy_gnn_v2.py" if trainer == "v2" else "py\\train_policy_gnn.py"

    cmd = [
        python_exe, train_script,
        "--data", str(out_policy),
        "--epochs", str(epochs),
        "--seed", str(train_seed),
        "--device", device,
        "--model-out", str(model_out),
        "--label-mode", str(label_mode),
    ]

    if hid is not None:
        cmd += ["--hid", str(hid)]
    if dropout is not None:
        cmd += ["--dropout", str(dropout)]
    if use_layernorm:
        cmd.append("--use-layernorm")

    if trainer == "v2":
        for cfg_key, cli_key in [
            ("fine_frac", "--fine-frac"),
            ("q_top", "--q-top"),
            ("gap_min", "--gap-min"),
            ("qlo", "--qlo"),
            ("qhi", "--qhi"),
            ("margin_fine", "--margin-fine"),
            ("margin_coarse", "--margin-coarse"),
            ("pairs_per_step", "--pairs-per-step"),
            ("steps_per_epoch", "--steps-per-epoch"),
            ("fine_frac_start", "--fine-frac-start"),
            ("fine_frac_end", "--fine-frac-end"),
            ("holdout", "--holdout"),
            ("val_split", "--val-split"),
            ("lr", "--lr"),
        ]:
            if cfg_key in exp:
                cmd += [cli_key, str(exp[cfg_key])]
    elif deterministic:
        cmd.append("--deterministic")

    run_cmd(cmd, cwd=root, stdout_path=out_policy / "train.log", dry_run=dry_run)


def step_ga(exp: Dict[str, Any], root: Path, python_exe: str, dry_run: bool) -> None:
    sumocfg = str(root / exp["sumocfg"])
    net = str(root / exp["net"])
    sumo_bin = str(exp.get("sumo_bin", "")) or str(root / exp.get("sumo_bin_rel", ""))
    kmax = int(exp["kmax"])
    kmin = int(exp.get("kmin", 0))
    pop = int(exp.get("pop", 24))
    gen = int(exp.get("gen", 15))
    gen_by_mode = exp.get("gen_by_mode", {}) or {}
    jobs = int(exp.get("jobs", 8))
    pc = float(exp.get("pc", 0.9))
    pm = float(exp.get("pm", 0.1))
    demand_size = int(exp.get("demand_size", 1000))
    protect = as_list(exp.get("protect", []))

    out_policy = root / exp["out_policy"]
    policy_model = exp.get("policy_model")
    if not policy_model:
        policy_model = str(out_policy / "policy_model.pt")
    else:
        policy_model = str(root / policy_model) if not Path(policy_model).is_absolute() else str(policy_model)

    policy_graph = exp.get("policy_graph")
    if policy_graph is None:
        policy_graph = str(out_policy / "graph_policy.pt")
    else:
        policy_graph = str(root / policy_graph) if not Path(policy_graph).is_absolute() else str(policy_graph)

    policy_score_mode = str(exp.get("policy_score_mode", "auto"))
    policy_mix_init = exp.get("policy_mix_init", exp.get("policy_mix", 0.5))
    policy_mix_mut = exp.get("policy_mix_mut", exp.get("policy_mix", 0.5))

    runs_root = root / exp["runs_root"]

    seeds = as_list(exp.get("seeds", list(range(10))))
    modes = as_list(exp.get("modes", ["both", "init", "mutation", "none"]))
    parallel_runs = int(exp.get("parallel_runs", 1))

    print(f"[ga] modes={modes} seeds={seeds} parallel_runs={parallel_runs} runs_root={runs_root}")

    def one_run(mode: str, seed: int) -> None:
        outdir = runs_root / f"{mode}_seed{seed}"
        log = outdir / "run.log"
        gen_eff = int(gen_by_mode.get(mode, gen))
        cmd = [
            python_exe, "py\\ga_edge_closure_gnn_policy.py",
            "--sumocfg", sumocfg, "--net", net, "--sumo-bin", sumo_bin,
            "--gnn-usage", mode, "--mode", "variable",
            "--kmin", str(kmin), "--kmax", str(kmax),
            "--pop", str(pop), "--gen", str(gen_eff), "--jobs", str(jobs),
            "--pc", str(pc), "--pm", str(pm),
            "--policy-model", policy_model,
            "--time-to-teleport", "-1", "--time-to-teleport-highways", "-1",
            "--outdir", str(outdir), "--seed", str(seed),
            "--demand-size", str(demand_size),
        ]

        if policy_graph:
            cmd += ["--policy-graph", str(policy_graph)]
        if policy_score_mode:
            cmd += ["--policy-score-mode", str(policy_score_mode)]
        if policy_mix_init is not None:
            cmd += ["--policy-mix-init", str(policy_mix_init)]
        if policy_mix_mut is not None:
            cmd += ["--policy-mix-mut", str(policy_mix_mut)]

        for cfg_key, cli_key in [
            ("policy_alpha_init", "--policy-alpha-init"),
            ("policy_alpha_mut", "--policy-alpha-mut"),
            ("policy_temp", "--policy-temp"),
            ("seed_ratio", "--seed-ratio"),
            ("policy_jitter", "--policy-jitter"),
            ("policy_temp_end", "--policy-temp-end"),
            ("policy_filter_frac", "--policy-filter-frac"),
            ("init_guided_frac", "--init-guided-frac"),
            ("prescreen", "--prescreen"),
            ("eval_seeds", "--eval-seeds"),
            ("eval_seed_base", "--eval-seed-base"),
            ("evaluator", "--evaluator"),
            ("ue_iters", "--ue-iters"),
            ("ue_tail", "--ue-tail"),
            ("ue_trips", "--ue-trips"),
            ("policy_set_model", "--policy-set-model"),
            ("set_cand_sample", "--set-cand-sample"),
            ("policy_scores_file", "--policy-scores-file"),
        ]:
            if cfg_key in exp:
                cmd += [cli_key, str(exp[cfg_key])]

        if as_bool(exp, "dedup_population", False):
            cmd.append("--dedup-population")

        init_include = exp.get("init_include")
        if init_include:
            p = Path(str(init_include))
            cmd += ["--init-include", str(root / p) if not p.is_absolute() else str(p)]

        if exp.get("candidates_file"):
            cmd += ["--candidates-file", str(root / str(exp["candidates_file"]))]

        if as_bool(exp, "enforce_od_connectivity", False):
            cmd.append("--enforce-od-connectivity")

        if protect:
            cmd += ["--protect"] + [str(p) for p in protect]
        run_cmd(cmd, cwd=root, stdout_path=log, dry_run=dry_run)

    with ThreadPoolExecutor(max_workers=max(1, parallel_runs)) as ex:
        futs = [ex.submit(one_run, mode, int(seed)) for mode in modes for seed in seeds]
        for fut in as_completed(futs):
            fut.result()


def step_analyze(exp: Dict[str, Any], root: Path, python_exe: str, dry_run: bool) -> None:
    runs_root = root / exp["runs_root"]
    modes = as_list(exp.get("modes", ["both", "init", "mutation", "none"]))
    jobs = int(exp.get("analyze_jobs", 4))
    cmd = [
        python_exe, "py\\analyze_runs.py",
        "--root", str(runs_root),
        "--modes", *[str(m) for m in modes],
        "--select", "best",
        "--jobs", str(jobs),
    ]
    run_cmd(cmd, cwd=root, stdout_path=runs_root / "analyze.log", dry_run=dry_run)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to experiments.json")
    ap.add_argument("--name", required=True, help="Experiment name (matches experiments[].name)")
    ap.add_argument("--steps", nargs="+", default=["dataset", "merge", "train", "ga", "analyze"],
                    choices=["dataset", "merge", "train", "ga", "analyze"])
    ap.add_argument("--dry-run", action="store_true", help="Print commands only")
    ap.add_argument("--python", default=sys.executable, help="Python executable (default: current)")
    ap.add_argument("--modes", nargs="+", default=None,
                    help="Override experiment modes for this invocation (e.g. --modes mutation "
                         "to add one arm without re-running existing ones)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Override experiment seeds for this invocation (e.g. --seeds 8 9 "
                         "to resume just the missing seeds without re-running existing ones)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    root = Path(cfg.get("root", "."))
    exp = get_experiment(cfg, args.name)
    if args.modes:
        exp = {**exp, "modes": args.modes}
        print(f"[override] modes = {args.modes}")
    if args.seeds:
        exp = {**exp, "seeds": args.seeds}
        print(f"[override] seeds = {args.seeds}")

    if exp.get("root"):
        root = Path(exp["root"])

    for s in args.steps:
        if s == "dataset":
            step_dataset(exp, root, args.python, args.dry_run)
        elif s == "merge":
            step_merge(exp, root, args.python, args.dry_run)
        elif s == "train":
            step_train(exp, root, args.python, args.dry_run)
        elif s == "ga":
            step_ga(exp, root, args.python, args.dry_run)
        elif s == "analyze":
            step_analyze(exp, root, args.python, args.dry_run)

    print("[DONE]", args.name, "steps:", args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
