# GNN-guided evolutionary search for simulation-based road-closure optimisation

Code, SUMO scenarios, offline datasets, trained models and result data for

> *Graph Neural Network-Guided Evolutionary Search for Simulation-Based Road-Closure Optimisation*, International Journal of Computational Science and Engineering.

A policy graph neural network is trained offline on SUMO rollouts and its edge scores bias the initialisation and mutation of a genetic algorithm; SUMO remains the only source of fitness values. The benchmark has 38 instances on five networks (A, B, C, D and the Wildau network J), each run with four guidance modes (`none`, `init`, `mutation`, `both`) and ten seeds.

## Repository layout

| Path | Contents |
|---|---|
| `run_experiments.py` | Pipeline driver: `dataset`, `merge`, `train`, `ga`, `analyze` steps for one configuration entry |
| `merge_policy_parts.py` | Merges the parallel parts of an offline rollout dataset |
| `py/` | Dataset builder, Policy GNN and trainer, the GA with guided initialisation and mutation, run analysis, and the comparators (simulated annealing, centrality heuristic, surrogate-fitness GA) |
| `experiments_benchmark38.json` | One entry per benchmark instance with every GA setting written out |
| `instance_manifest.csv` | For each of the 38 instances: the configuration file and entry, the original entry it was derived from, the changes, the policy model, the offline dataset and the recorded runs |
| `experiments_39.json`, `experiments_pm_rerun_kmin0.json`, `experiments_k16plain.json` | The original configuration files (39 designed instances; the kmin = 0 re-run of the three A_k4_3od instances; the restricted-candidate probe of Section 5.1) |
| `netA/` … `netD/` | SUMO networks, demand files and `.sumocfg` files of the instances on networks A–D (network J is not included; see below) |
| `output_policy_<instance>/` | Offline dataset (`dataset_policy.pt`, 1,000 rollouts), policy graph (`graph_policy.pt`), trained Policy GNN (`policy_model.pt`), training and merge logs; for network J the model and logs only |
| `merged_analysis_38_k0/` | Every individual evaluated in the 1,520 benchmark runs (`aggregate_merged.csv.gz`) and the per-instance summaries derived from it |
| `runs_A_k4_3od_x*_pm01k0/`, `runs_reviewer_A38/`, `runs_reviewer/`, `runs_J_k16_plain/` | Per-run summaries (`summary.csv`, `best.txt`) of the A_k4_3od benchmark runs, the comparators and the Section 5.1 probe |
| `reviewer_diagnostics/` | Surrogate models of the comparator, held-out re-simulations at SUMO seeds 201–205, and the Appendix A and C diagnostics |
| `abl/`, `figs_k0/`, `reviewer_diagnostics/scripts/`, `analyze_guidance_patterns.py` | Scripts that produce the tables, statistics and figures |
| `manuscript_numbers_k0/` | The numbers and tables quoted in the article, as produced by those scripts |
| `figures/` | Figures 2–11 of the article |
| `requirements-pinned.txt` | Pinned package list of the Python environment |
| `decompress.py` | Restores every `*.gz` file in the repository next to its archive |

## Network J

Network J is the SUMO model of Wildau (Brandenburg, Germany), `Netzmodell2.net.xml`, from the Wildau scenario of the DLR-TS SUMO scenarios collection (https://github.com/DLR-TS/sumo-scenarios/tree/main/Wildau), created in a student project at Technische Hochschule Wildau. The network was built from OpenStreetMap data (© OpenStreetMap contributors, ODbL, https://www.openstreetmap.org/copyright). It is not redistributed here, and neither are the files that encode its graph (`graph_policy.pt` and `dataset_policy.pt` of the `output_policy_J_*` folders).

To run the J instances, obtain `Netzmodell2.net.xml` from that repository and place it in `netJ/`. The demand, candidate and protected-edge files of the J instances were written for this study; they are available from the authors on request. Result tables, run summaries and trained models for network J are included, so every value reported for J can be checked. The scripts that read the J graph or dataset (`figs_k0/compute_signal_density.py`, `figs_k0/compute_distribution_shift.py`, `abl/surrogate_quality_0914.py`) skip network J when those files are absent; the released outputs of those scripts still contain the J rows.

## Requirements

* Windows. The experiments were run on Windows 11; several scripts build paths with backslashes.
* Python 3.13 and the packages in `requirements-pinned.txt` (CPU builds of PyTorch are taken from the PyTorch package index listed in the file).
* Eclipse SUMO 1.24.0 with `sumo` on `PATH` and `SUMO_HOME` set; `sumolib` and `traci` are imported from `SUMO_HOME/tools`.

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements-pinned.txt
```

Run every script from the repository root. The comparator drivers start `.venv\Scripts\python.exe`, so the environment should live in `.venv` there.

## Reproducing the benchmark runs

The GA stage of any instance is re-run from `experiments_benchmark38.json`, using the released offline dataset and policy model:

```bat
python run_experiments.py --config experiments_benchmark38.json --name B_k9_3od_x1.0 --steps ga analyze
```

Results are written to `runs/<instance>/`. Every run is deterministic given its seed. Use only the `ga` and `analyze` steps: the `dataset` and `train` steps would overwrite the released dataset and model, and the released dataset builder and trainer do not reproduce them with their defaults (see below).

## Reproducing the comparators

| Comparator | Command |
|---|---|
| Centrality heuristic, sign-reversed heuristic, simulated annealing and surrogate arms on instance A | `python abl/rerun_A38_0914.py ga`, `… sa`, `… surr` |
| Heuristic, simulated annealing and Appendix A surrogate arm on B and J | `python run_reviewer_arms.py --instances B J --arms heuristic sa surrogate` |
| Sign-reversed heuristic on B | `python abl/run_heuristic_rev_0914.py B` |
| Reported surrogate arm (Table 4) | `reviewer_diagnostics/scripts/train_surrogate_v3.py`, `reviewer_diagnostics/scripts/surrogate_arm_v3.py` |
| Held-out re-simulation at SUMO seeds 201–205 | `abl/k0_reeval.py` and `abl/k0_merge_reeval.py` (guidance modes), `abl/reeval_comparators_0914.py` (B, J), `abl/reeval_comparators_A38_0914.py` (A) |

The centrality heuristic reads the edge scores in `scores_betweenness_A.csv`, `_B.csv` and `_J.csv` (length-weighted edge betweenness from `py/compute_structural_scores.py`). The A entries of `run_reviewer_arms.py`, `abl/reeval_comparators_0914.py` and `abl/run_heuristic_rev_0914.py` leave the origin edges E12 and E18 as candidates (40 instead of 38). Every value for instance A in the article comes from `abl/rerun_A38_0914.py`, `runs_reviewer_A38/` and `reviewer_diagnostics/results/reeval_cmp_201_205_A38/`; the A rows of `reviewer_diagnostics/results/reeval_cmp_201_205/` are superseded.

## Reproducing the tables and figures

These steps need no simulation. First restore the compressed files (the evaluation log of the benchmark and the SUMO trip outputs of the surrogate arm's verification runs, which Table 4 is computed from):

```bat
python decompress.py
```

| Output | Script |
|---|---|
| Tables 2 and 3, per-instance and per-network summaries | `abl/k0_manuscript_tables.py`, `figs_k0/compute_effect_size.py` |
| Numbers quoted in Sections 4 and 5 | `abl/k0_numbers.py`, `abl/networkA_sensitivity_0914.py`, `abl/table3_a12_vs_mean_0913.py`, `abl/sign_control_stats_0914.py` |
| Tables 4, 5 and 9 | `abl/a38_numbers_0914.py`, `abl/table5_holm_families_0914.py`, `reviewer_diagnostics/scripts/k0_table5_holm18.py`, `abl/heldout_modes3_0914.py`, `abl/analyze_cmp_heldout_0914.py` |
| Table 6 | `abl/surrogate_quality_0914.py` |
| Table 7 and Figure 11 | `figs_k0/signal_density.csv`, written by `abl/k0_signal_density.py` (see the note below) |
| Table 8 and Section 5.2 | `abl/k0_sec52.py` |
| Table 10 | `figs_k0/compute_distribution_shift.py --root .` |
| Figures 2–5, 9 and 11 | `analyze_guidance_patterns.py --input merged_analysis_38_k0 --output merged_analysis_38_k0/extra_analysis`, then `figs_k0/make_paper_figures_v2.py --data merged_analysis_38_k0 --outdir <dir>` |
| Figures 6–8 | `figs_k0/make_convergence_figures_v2.py --aggregate merged_analysis_38_k0/aggregate_merged.csv --outdir <dir>` |
| Figure 10 | `figs_k0/make_new_figures.py --root . --outdir <dir>` |

`figs_k0/compute_signal_density.py` counts every edge of an offline dataset's mask, including the edges that dataset protects. Those edges are never closed, so its `edges` and `never_closed` columns exceed `figs_k0/signal_density.csv` by the number of protected edges (2 to 12 per instance) and `obs_per_edge` is lower by the same ratio; the other columns agree. The table counts candidate edges only. For network J the article also counts the origin edge `111681905#0` as a candidate, as the benchmark runs do (1,415 candidates, 59 never closed), whereas the offline dataset protects it (1,414 and 58 in the table).

`abl/k0_build_merged.py` rebuilt `merged_analysis_38_k0/` from the raw run folders, which are not released because of their size (tens of gigabytes of SUMO output).

Some scripts also read earlier or superseded data alongside the data the article uses. These files are released so that the scripts run unchanged:

| Path | What it is |
|---|---|
| `runs_reviewer/A_k4_3od_x1.0/`, `reviewer_diagnostics/results/arm_A_sum_run/` | Comparator runs on instance A with 40 candidates, replaced by `runs_reviewer_A38/` and `arm_A_sum_run38/` |
| `reviewer_diagnostics/results/reeval_<instance>/`, `reeval_<instance>_v2/` | An earlier held-out re-simulation at SUMO seeds 101–110 |
| `reviewer_diagnostics/results/reeval_39/` | Held-out re-simulation of the 39-instance bundle; `abl/k0_merge_reeval.py` carries its rows for the instances other than A_k4_3od into `reeval_39_k0/` unchanged |
| `runs_a12fix/` | Held-out values of a corrected re-run of the duplicate design instance, used for a variant table |
| `figs38/signal_density.csv` | Signal-density table before the guidance gains were refreshed by `abl/k0_signal_density.py` |

## Notes on reproducibility

* **Offline datasets and policy models.** They were produced by earlier versions of `py/make_policy_dataset.py` and `py/train_policy_gnn.py` than the released ones. The released versions add options for other labels, features, sampling designs and splits, and their defaults do not reproduce the released models; for example, the trainer now defaults to a hidden width of 128 and a 15% validation split. The released models have the settings stated in Sections 3.5 and 3.6 (hidden width 256, four attention heads, 800/200 split, 60 epochs, seed 42). Re-running the GA stage uses the released models.
* **Uniform mixture.** The three A_k4_3od instances sample from a mixture of the guided softmax (weight 0.25) and the uniform distribution (Section 3.8). The other instances use the softmax alone. `run_experiments.py` applies a mixture weight of 0.5 when an entry does not set one, so `experiments_benchmark38.json` sets the weight for every instance.
* **Instances flagged in `instance_manifest.csv`.** For A_k4_2od_x1.2, B_k9_3od_x0.8 and some seeds of B_k9_2od_x0.8, the recorded first-generation guided populations are flatter than the released policy model gives at the stated settings, and the setting that produced them could not be recovered. Re-running these entries reproduces the method as stated, not those recorded runs.
* **Options present but not used for the 38 instances.** The OD-connectivity feasibility repair (`enforce_od_connectivity`) and the candidate-restriction file (`candidates_file`) are set only in `experiments_k16plain.json`, the restricted-candidate probe of Section 5.1. The GA also accepts options for later work (`--evaluator ue`, `--policy-set-model`) whose modules are not part of this release.
* **Failed simulations.** Six of the 1,520 benchmark runs contain simulations that failed. A failure is not determined by the seed, so a re-run reproduces such a run only if its failures recur.
* **Surrogate comparator.** Its set-level model uses scatter reductions whose floating-point order is not fixed, so its per-seed values are not bitwise reproducible.
* **Configuration files.** The original files contain overlapping entry names, and the entry `A_k4_1od_x1.0` of `experiments_39.json` was later edited for a different run. Use `experiments_benchmark38.json` and `instance_manifest.csv` rather than matching instances by name.
* **Duplicate design instance.** `netA/demand_1od_x1.2.trips.xml` is byte-identical to `netA/demand_1od_x1.0.trips.xml`, so the designed instance A_k4_1od_x1.2 is not distinct and the benchmark counts 38 instances.
* **Recorded paths.** The result tables and the SUMO network headers keep the absolute paths of the machine the experiments ran on. They are records only; no released script opens a file through them.
