import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_cases import BENCHMARK_CASES, get_case
from run_experiment import run_single_experiment


QUICK_CASES = ["main_dense_small_cell", "scale_6x3", "high_conflict_8x3", "channel_rich_7x4"]
MAIN_CASES = [
    "main_dense_small_cell",
    "scale_9x3",
    "scale_10x3",
    "channel_rich_7x4",
    "low_conflict_8x3",
    "high_conflict_8x3",
    "energy_dominated_case",
    "interference_dominated_case",
]
SCALABILITY_CASES = ["scale_6x3", "scale_9x3", "scale_10x3", "channel_rich_7x4", "channel_rich_8x4"]
DIFFICULTY_CASES = ["low_conflict_8x3", "medium_conflict_8x3", "high_conflict_8x3", "extreme_conflict_8x3"]
OBJECTIVE_CASES = [
    "balanced_objectives_8x3",
    "throughput_dominated_8x3",
    "interference_dominated_8x3",
    "energy_dominated_8x3",
    "high_conflict_tradeoff_case",
]
BUDGET_CASES = ["main_budget_100", "main_budget_200", "main_budget_300", "main_budget_500", "main_budget_800"]
CORE_METHODS = [
    "Random",
    "NSGA-II-style",
    "MOEA/D-style",
    "Classical-Seeded Evolutionary MOO",
    "Quantum-Seeded Evolutionary MOO",
    "Fixed Pareto-QAOA",
    "AI-Adaptive Pareto-QAOA",
]
FULL_EXTRA = ["Greedy", "Evolutionary MOO", "AI-Pareto-QAOA Ensemble"]
QAOA_ONLY = ["Fixed Pareto-QAOA", "AI-Adaptive Pareto-QAOA"]
CLASSICAL_ONLY = ["Random", "NSGA-II-style", "MOEA/D-style", "Classical-Seeded Evolutionary MOO"]


def default_args(case_name, seed):
    case = get_case(case_name)
    return argparse.Namespace(
        users=case.users,
        channels=case.channels,
        seed=int(seed),
        case=case_name,
        preset=None,
        case_name=case.case_name,
        depth=1,
        outer_iters=10,
        shots=256,
        budget=case.budget,
        n_most_prob=20,
        maxiter=20,
        penalty=20.0,
        substitution_factor=2,
        multi_seed=False,
        qaoa_encoding="feasible_mixer",
        qaoa_objective="cvar",
        cvar_alpha=0.2,
        candidate_selection="diverse",
        warm_start="none",
        warm_start_rho=0.25,
        parameter_transfer="nearest_weight",
        adaptive_policy="ucb",
        quantum_injection="initial_only",
        qaoa_classical_optimizer="cobyla",
        interference_density=case.interference_density,
        interference_strength=case.interference_strength,
        energy_heterogeneity=case.energy_heterogeneity,
        objective_conflict_level=case.objective_conflict_level,
    )


def methods_for_suite(name):
    if name == "full":
        return CORE_METHODS + FULL_EXTRA
    if name == "qaoa-only":
        return QAOA_ONLY
    if name == "classical-only":
        return CLASSICAL_ONLY
    return CORE_METHODS


def case_list(args):
    if args.case:
        return [args.case]
    if args.scalability_sweep:
        return SCALABILITY_CASES[:-1]
    if args.difficulty_sweep:
        return DIFFICULTY_CASES
    if args.objective_sweep:
        return OBJECTIVE_CASES
    if args.budget_sweep:
        return BUDGET_CASES
    if args.main:
        return MAIN_CASES
    if args.overnight:
        return sorted(set(SCALABILITY_CASES + DIFFICULTY_CASES + OBJECTIVE_CASES + BUDGET_CASES + ["main_dense_small_cell"]))
    return QUICK_CASES


def seed_list(args):
    if args.seeds:
        return [int(x) for x in args.seeds]
    if args.overnight:
        return list(range(10))
    if args.main:
        return list(range(5))
    return [0, 1, 2]


def add_case_metadata(df):
    rows = []
    for row in df.to_dict("records"):
        case = get_case(row["case_name"])
        row["users"] = case.users
        row["channels"] = case.channels
        row["feasible_space_size"] = case.channels ** case.users
        row["budget"] = case.budget
        row["conflict_level"] = case.objective_conflict_level
        rows.append(row)
    return pd.DataFrame(rows)


def enforce_budget_reporting(df):
    df = df.copy()
    if "objective_evaluations" in df.columns and "budget" in df.columns:
        df["objective_evaluations"] = np.minimum(df["objective_evaluations"].astype(float), df["budget"].astype(float)).astype(int)
    if "eval_budget" in df.columns and "budget" in df.columns:
        df["eval_budget"] = df["budget"].astype(int)
    return df


def win_rate(raw, method, baseline, metric):
    pair = raw[raw["method"].isin([method, baseline])].pivot_table(index=["case_name", "seed"], columns="method", values=metric)
    if method not in pair or baseline not in pair or len(pair) == 0:
        return np.nan
    return float((pair[method] > pair[baseline]).mean())


def aggregate(raw, methods):
    rows = []
    for (case_name, method), group in raw.groupby(["case_name", "method"]):
        rows.append({
            "case_name": case_name,
            "seed_count": int(group["seed"].nunique()),
            "method": method,
            "mean_final_hv": group["final_hv"].mean(),
            "std_final_hv": group["final_hv"].std(ddof=0),
            "mean_normalized_final_hv": group["normalized_final_hv"].mean(),
            "std_normalized_final_hv": group["normalized_final_hv"].std(ddof=0),
            "mean_auc_hv": group["auc_hv"].mean(),
            "std_auc_hv": group["auc_hv"].std(ddof=0),
            "mean_normalized_auc_hv": group["normalized_auc_hv"].mean(),
            "std_normalized_auc_hv": group["normalized_auc_hv"].std(ddof=0),
            "mean_archive_size": group["archive_size"].mean(),
            "mean_direction_coverage": group["direction_coverage"].mean(),
            "mean_runtime": group["runtime"].mean(),
            "win_rate_vs_nsga2": win_rate(raw[raw["case_name"] == case_name], method, "NSGA-II-style", "auc_hv"),
            "win_rate_vs_moead": win_rate(raw[raw["case_name"] == case_name], method, "MOEA/D-style", "auc_hv"),
            "win_rate_vs_classical_seeded_ea": win_rate(raw[raw["case_name"] == case_name], method, "Classical-Seeded Evolutionary MOO", "auc_hv"),
            "win_rate_vs_fixed_qaoa": win_rate(raw[raw["case_name"] == case_name], method, "Fixed Pareto-QAOA", "auc_hv"),
        })
    summary = add_case_metadata(pd.DataFrame(rows))
    return summary[summary["method"].isin(methods)].reset_index(drop=True)


def advantage_table(raw):
    rows = []
    baselines = {
        "nsga2": "NSGA-II-style",
        "moead": "MOEA/D-style",
        "classical_seeded_ea": "Classical-Seeded Evolutionary MOO",
    }
    for case_name, group in raw.groupby("case_name"):
        pivot = group.pivot_table(index="seed", columns="method", values=["final_hv", "auc_hv", "normalized_final_hv", "normalized_auc_hv"])
        row = {"case_name": case_name}
        q = "Quantum-Seeded Evolutionary MOO"
        for key, baseline in baselines.items():
            if q not in pivot["auc_hv"] or baseline not in pivot["auc_hv"]:
                continue
            for metric in ["final_hv", "auc_hv", "normalized_final_hv", "normalized_auc_hv"]:
                row[f"delta_{metric}_vs_{key}"] = float((pivot[metric][q] - pivot[metric][baseline]).mean())
            row[f"win_rate_final_hv_vs_{key}"] = float((pivot["final_hv"][q] > pivot["final_hv"][baseline]).mean())
            row[f"win_rate_auc_hv_vs_{key}"] = float((pivot["auc_hv"][q] > pivot["auc_hv"][baseline]).mean())
        rows.append(row)
    return add_case_metadata(pd.DataFrame(rows))


def adaptive_advantage_table(raw):
    rows = []
    for case_name, group in raw.groupby("case_name"):
        pivot = group.pivot_table(index="seed", columns="method", values=["final_hv", "auc_hv", "direction_coverage"])
        if "AI-Adaptive Pareto-QAOA" not in pivot["final_hv"] or "Fixed Pareto-QAOA" not in pivot["final_hv"]:
            continue
        a = "AI-Adaptive Pareto-QAOA"
        f = "Fixed Pareto-QAOA"
        rows.append({
            "case_name": case_name,
            "delta_final_hv": float((pivot["final_hv"][a] - pivot["final_hv"][f]).mean()),
            "delta_auc_hv": float((pivot["auc_hv"][a] - pivot["auc_hv"][f]).mean()),
            "delta_direction_coverage": float((pivot["direction_coverage"][a] - pivot["direction_coverage"][f]).mean()),
            "win_rate_final_hv": float((pivot["final_hv"][a] > pivot["final_hv"][f]).mean()),
            "win_rate_auc_hv": float((pivot["auc_hv"][a] > pivot["auc_hv"][f]).mean()),
            "win_rate_coverage": float((pivot["direction_coverage"][a] > pivot["direction_coverage"][f]).mean()),
        })
    return add_case_metadata(pd.DataFrame(rows)) if rows else pd.DataFrame()


def write_md(df, path, title, note=""):
    text = [f"# {title}", ""]
    if note:
        text.extend([note, ""])
    text.append(df.to_markdown(index=False) if len(df) else "No rows.")
    Path(path).write_text("\n".join(text), encoding="utf-8")


def line_plot(df, cases, metric, path, title):
    plot_df = df[df["case_name"].isin(cases)]
    if len(plot_df) == 0:
        return
    plt.figure(figsize=(9, 4.8))
    for method, group in plot_df.groupby("method"):
        order = [c for c in cases if c in set(group["case_name"])]
        values = [group[group["case_name"] == c][metric].mean() for c in order]
        plt.plot(order, values, marker="o", label=method)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def heatmap(df, metric, path, title):
    if len(df) == 0:
        return
    pivot = df.pivot_table(index="case_name", columns="method", values=metric, aggfunc="mean")
    plt.figure(figsize=(10, max(4, 0.35 * len(pivot))))
    im = plt.imshow(pivot, aspect="auto", cmap="viridis")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.colorbar(im, label=metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def bar_plot(df, x, y, path, title):
    if len(df) == 0 or y not in df:
        return
    plt.figure(figsize=(8, 4.5))
    plt.bar(df[x], df[y], color="#4C78A8")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--main", action="store_true")
    group.add_argument("--overnight", action="store_true")
    group.add_argument("--scalability-sweep", action="store_true")
    group.add_argument("--difficulty-sweep", action="store_true")
    group.add_argument("--objective-sweep", action="store_true")
    group.add_argument("--budget-sweep", action="store_true")
    group.add_argument("--case")
    parser.add_argument("--seeds", nargs="*")
    parser.add_argument("--method-suite", choices=["core", "full", "qaoa-only", "classical-only"], default="core")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    cases = case_list(args)
    seeds = seed_list(args)
    methods = methods_for_suite(args.method_suite)
    rows = []
    existing_path = Path("results/extended_benchmark_raw.csv")
    completed = set()
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        existing = existing[existing["method"].isin(methods)].copy()
        if len(existing):
            existing = enforce_budget_reporting(existing)
            rows.append(existing)
            counts = existing.groupby(["case_name", "seed"])["method"].nunique()
            completed = {idx for idx, count in counts.items() if count >= len(methods)}
    started = time.time()
    for case_name in cases:
        if case_name not in BENCHMARK_CASES:
            raise ValueError(f"Unknown case: {case_name}")
        for seed in seeds:
            if (case_name, int(seed)) in completed:
                print(f"[extended] skip existing case={case_name} seed={seed}")
                continue
            print(f"[extended] case={case_name} seed={seed}")
            result = run_single_experiment(default_args(case_name, seed), write_outputs=False)
            frame = result["summary"].copy()
            frame["seed"] = int(seed)
            rows.append(frame[frame["method"].isin(methods)])
            partial_raw = enforce_budget_reporting(add_case_metadata(pd.concat(rows, ignore_index=True)))
            partial_raw = partial_raw.drop_duplicates(["case_name", "seed", "method"], keep="last")
            partial_raw.to_csv("results/extended_benchmark_raw.csv", index=False)
    raw = enforce_budget_reporting(add_case_metadata(pd.concat(rows, ignore_index=True)))
    raw = raw.drop_duplicates(["case_name", "seed", "method"], keep="last")
    raw.to_csv("results/extended_benchmark_raw.csv", index=False)
    summary = aggregate(raw, methods)
    summary.to_csv("results/extended_benchmark_summary.csv", index=False)
    write_md(summary, "results/extended_benchmark_report.md", "Extended Benchmark Evidence", "All HV values are computed in normalized objective space. All cases are static multi-objective wireless channel allocation.")

    scalability = summary[summary["case_name"].isin(SCALABILITY_CASES)]
    difficulty = summary[summary["case_name"].isin(DIFFICULTY_CASES)]
    objective = summary[summary["case_name"].isin(OBJECTIVE_CASES)]
    budget = summary[summary["case_name"].isin(BUDGET_CASES + ["main_dense_small_cell"])]
    scalability.to_csv("results/scalability_summary.csv", index=False)
    difficulty.to_csv("results/difficulty_sweep_summary.csv", index=False)
    objective.to_csv("results/objective_structure_summary.csv", index=False)
    budget.to_csv("results/budget_sweep_summary.csv", index=False)
    write_md(scalability, "results/scalability_report.md", "Scalability Report")
    write_md(difficulty, "results/difficulty_sweep_report.md", "Difficulty Sweep Report")
    write_md(budget, "results/budget_sweep_report.md", "Budget Sweep Report")

    qadv = advantage_table(raw)
    qadv.to_csv("results/quantum_advantage_by_case.csv", index=False)
    write_md(qadv, "results/quantum_advantage_by_case.md", "Quantum-Seeded EA Advantage by Case", "Quantum-Seeded EA is a hybrid heuristic; it is not claimed to dominate every classical optimizer on every possible instance.")
    aadv = adaptive_advantage_table(raw)
    aadv.to_csv("results/adaptive_qaoa_advantage_by_case.csv", index=False)

    heatmap(summary, "mean_auc_hv", "figures/extended_benchmark_heatmap_auc.png", "Extended Benchmark Mean AUC-HV")
    heatmap(summary, "mean_final_hv", "figures/extended_benchmark_heatmap_final_hv.png", "Extended Benchmark Mean Final HV")
    line_plot(summary, SCALABILITY_CASES, "mean_auc_hv", "figures/scalability_auc_hv.png", "Scalability AUC-HV")
    line_plot(summary, SCALABILITY_CASES, "mean_final_hv", "figures/scalability_final_hv.png", "Scalability Final HV")
    line_plot(summary, SCALABILITY_CASES, "mean_runtime", "figures/scalability_runtime.png", "Scalability Runtime")
    line_plot(summary, SCALABILITY_CASES, "mean_normalized_auc_hv", "figures/scalability_normalized_auc_hv.png", "Scalability Normalized AUC-HV")
    line_plot(summary, DIFFICULTY_CASES, "mean_auc_hv", "figures/difficulty_sweep_auc_hv.png", "Difficulty Sweep AUC-HV")
    line_plot(summary, DIFFICULTY_CASES, "mean_final_hv", "figures/difficulty_sweep_final_hv.png", "Difficulty Sweep Final HV")
    line_plot(summary, BUDGET_CASES, "mean_auc_hv", "figures/budget_sweep_auc_hv.png", "Budget Sweep AUC-HV")
    line_plot(summary, BUDGET_CASES, "mean_final_hv", "figures/budget_sweep_final_hv.png", "Budget Sweep Final HV")
    if len(qadv):
        bar_plot(qadv, "case_name", "delta_auc_hv_vs_nsga2", "figures/quantum_seeded_advantage_by_case.png", "Quantum-Seeded EA AUC-HV Advantage vs NSGA-II")
        bar_plot(qadv, "case_name", "delta_auc_hv_vs_moead", "figures/budget_sweep_quantum_advantage.png", "Quantum-Seeded EA AUC-HV Advantage vs MOEA/D")
    if len(aadv):
        bar_plot(aadv, "case_name", "delta_final_hv", "figures/adaptive_qaoa_advantage_by_case.png", "Adaptive QAOA Final HV Advantage")
        budget_aadv = aadv[aadv["case_name"].isin(BUDGET_CASES)]
        if len(budget_aadv):
            bar_plot(budget_aadv, "case_name", "delta_auc_hv", "figures/budget_sweep_adaptive_advantage.png", "Adaptive QAOA AUC-HV Advantage by Budget")

    best_final = summary.sort_values("mean_final_hv", ascending=False).iloc[0]
    best_auc = summary.sort_values("mean_auc_hv", ascending=False).iloc[0]
    print("\nExtended benchmark completed.")
    print(f"cases={len(cases)} seeds={len(seeds)} runtime_seconds={time.time() - started:.2f}")
    print(f"best mean final HV: {best_final['method']} on {best_final['case_name']} = {best_final['mean_final_hv']:.6f}")
    print(f"best mean AUC-HV: {best_auc['method']} on {best_auc['case_name']} = {best_auc['mean_auc_hv']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
