import argparse
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptive import BASE_WEIGHTS, compute_archive_coverage
from benchmark_cases import get_case
from metrics import auc_hv_on_grid, hv_at_common_budget, normalized_hypervolume_mc
from pareto import non_dominated_filter
from problem import make_problem, objective_table
from run_experiment import run_qaoa_loop, write_markdown_table


MAIN_CASES = [
    "main_dense_small_cell",
    "high_conflict_tradeoff_case",
    "energy_dominated_case",
    "interference_dominated_case",
    "channel_rich_7x4",
]

VARIANTS = [
    ("Fixed Pareto-QAOA", False, False, False, False),
    ("Adaptive weights only", True, False, False, False),
    ("Adaptive CVaR only", False, True, False, False),
    ("Adaptive candidate diversity only", False, False, True, False),
    ("Adaptive resources only", False, False, False, True),
    ("Adaptive weights + CVaR", True, True, False, False),
    ("Adaptive weights + diversity", True, False, True, False),
    ("Full AI-Adaptive Pareto-QAOA", True, True, True, True),
]


def make_args(case_name, seed):
    case = get_case(case_name)
    return argparse.Namespace(
        users=case.users,
        channels=case.channels,
        seed=int(seed),
        case_name=case.case_name,
        depth=1,
        outer_iters=2,
        shots=96,
        budget=case.budget,
        n_most_prob=10,
        maxiter=1,
        penalty=20.0,
        substitution_factor=2,
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
    )


def setup_case(case_name, seed):
    case = get_case(case_name)
    problem = make_problem(
        case.users,
        case.channels,
        seed,
        case_name=case.case_name,
        interference_density=case.interference_density,
        interference_strength=case.interference_strength,
        energy_heterogeneity=case.energy_heterogeneity,
        objective_conflict_level=case.objective_conflict_level,
    )
    assignments, objectives = objective_table(problem)
    reference = objectives.max(axis=0) + np.array([1.0, 0.5, 0.5])
    obj_min, obj_max = objectives.min(axis=0), objectives.max(axis=0)
    exact_front = objectives[non_dominated_filter(objectives)]
    exact_hv = normalized_hypervolume_mc(exact_front, reference, obj_min, obj_max, seed=seed)
    return problem, {"assignments": assignments, "objectives": objectives}, reference, obj_min, obj_max, exact_hv


def run_variant(case_name, seed, variant, setup_bundle=None):
    label, adaptive_weights, adaptive_cvar, adaptive_diversity, adaptive_resources = variant
    args = make_args(case_name, seed)
    args.adaptive_weights_enabled = adaptive_weights
    args.adaptive_cvar_enabled = adaptive_cvar
    args.adaptive_diversity_enabled = adaptive_diversity
    args.adaptive_resources_enabled = adaptive_resources
    args.qaoa_label_override = label
    problem, cache, reference, obj_min, obj_max, exact_hv = setup_bundle or setup_case(case_name, seed)
    adaptive = any([adaptive_weights, adaptive_cvar, adaptive_diversity, adaptive_resources])
    result = run_qaoa_loop(problem, args, reference, adaptive=adaptive, qaoa_cache=cache, obj_min=obj_min, obj_max=obj_max, reference_hv=exact_hv)
    curve = result["curve"].copy()
    if len(curve):
        curve["method"] = label
    final_hv = float(curve["hv"].iloc[-1]) if len(curve) else 0.0
    common_budget = float(curve["cumulative_evaluations"].max()) if len(curve) else 1.0
    hv_parts = hv_at_common_budget(curve, common_budget)
    history = result["history"]
    return {
        "case_name": case_name,
        "seed": int(seed),
        "method": label,
        "final_hv": final_hv,
        "normalized_final_hv": final_hv / max(exact_hv, 1e-12),
        "auc_hv": auc_hv_on_grid(curve, np.linspace(0.0, common_budget, 101)),
        "normalized_auc_hv": auc_hv_on_grid(curve, np.linspace(0.0, common_budget, 101)) / max(exact_hv, 1e-12),
        "hv_25": hv_parts["hv_at_25pct"],
        "hv_50": hv_parts["hv_at_50pct"],
        "hv_75": hv_parts["hv_at_75pct"],
        "archive_size": len(result["objectives"]),
        "direction_coverage": compute_archive_coverage(result["objectives"], BASE_WEIGHTS, obj_min, obj_max),
        "feasibility_rate": result["feasibility_rate"],
        "valid_sample_efficiency": float(history["valid_sample_efficiency"].mean()) if len(history) and "valid_sample_efficiency" in history else 0.0,
        "duplicate_rate_mean": float(history["duplicate_rate"].mean()) if len(history) else 0.0,
        "runtime": result["runtime"],
        "eval_budget": args.budget,
        "curve": curve,
        "history": history.assign(method=label),
    }


def summarize_qaoa_result(case_name, seed, label, result, exact_hv):
    curve = result["curve"].copy()
    if len(curve):
        curve["method"] = label
    final_hv = float(curve["hv"].iloc[-1]) if len(curve) else 0.0
    common_budget = float(curve["cumulative_evaluations"].max()) if len(curve) else 1.0
    hv_parts = hv_at_common_budget(curve, common_budget)
    history = result["history"].copy()
    if len(history):
        history["method"] = label
    return {
        "case_name": case_name,
        "seed": int(seed),
        "method": label,
        "final_hv": final_hv,
        "normalized_final_hv": final_hv / max(exact_hv, 1e-12),
        "auc_hv": auc_hv_on_grid(curve, np.linspace(0.0, common_budget, 101)),
        "normalized_auc_hv": auc_hv_on_grid(curve, np.linspace(0.0, common_budget, 101)) / max(exact_hv, 1e-12),
        "hv_25": hv_parts["hv_at_25pct"],
        "hv_50": hv_parts["hv_at_50pct"],
        "hv_75": hv_parts["hv_at_75pct"],
        "archive_size": len(result["objectives"]),
        "direction_coverage": float(history["coverage_after"].iloc[-1]) if len(history) and "coverage_after" in history else 0.0,
        "feasibility_rate": result["feasibility_rate"],
        "valid_sample_efficiency": float(history["valid_sample_efficiency"].mean()) if len(history) and "valid_sample_efficiency" in history else 0.0,
        "duplicate_rate_mean": float(history["duplicate_rate"].mean()) if len(history) else 0.0,
        "runtime": result["runtime"],
        "eval_budget": make_args(case_name, seed).budget,
        "curve": curve,
        "history": history,
    }


def run_replay_variant(case_name, seed, online_history, setup_bundle=None):
    args = make_args(case_name, seed)
    args.qaoa_label_override = "Replay-Adaptive Schedule"
    args.replay_weight_indices = [int(x) for x in online_history.sort_values("iteration")["selected_weight_index"].tolist()]
    problem, cache, reference, obj_min, obj_max, exact_hv = setup_bundle or setup_case(case_name, seed)
    result = run_qaoa_loop(problem, args, reference, adaptive=False, qaoa_cache=cache, obj_min=obj_min, obj_max=obj_max, reference_hv=exact_hv)
    return summarize_qaoa_result(case_name, seed, "Replay-Adaptive Schedule", result, exact_hv)


def merge_study_rows(existing, new_rows):
    if new_rows:
        new_df = add_wins(pd.DataFrame(new_rows))
    else:
        new_df = pd.DataFrame()
    if len(existing) and len(new_df):
        merged = pd.concat([existing, new_df], ignore_index=True)
    elif len(existing):
        merged = existing.copy()
    else:
        merged = new_df
    if len(merged):
        merged = merged.drop_duplicates(["case_name", "seed", "method"], keep="last")
        merged = add_wins(merged)
    return merged


def merge_history_rows(history_path, new_histories):
    histories_df = pd.concat(new_histories, ignore_index=True) if new_histories else pd.DataFrame()
    if history_path.exists():
        old_history = pd.read_csv(history_path)
        histories_df = pd.concat([old_history, histories_df], ignore_index=True)
    if len(histories_df):
        histories_df = histories_df.drop_duplicates(["case_name", "seed", "method", "iteration"], keep="last")
        histories_df.to_csv(history_path, index=False)
    elif history_path.exists():
        histories_df = pd.read_csv(history_path)
    return histories_df


def add_wins(df):
    rows = []
    fixed = df[df["method"] == "Fixed Pareto-QAOA"].set_index(["case_name", "seed"])
    for row in df.to_dict("records"):
        key = (row["case_name"], row["seed"])
        f = fixed.loc[key] if key in fixed.index else None
        if f is None:
            row["wins_final_hv_against_fixed"] = False
            row["wins_auc_hv_against_fixed"] = False
            row["wins_coverage_against_fixed"] = False
        else:
            row["wins_final_hv_against_fixed"] = bool(row["final_hv"] > f["final_hv"])
            row["wins_auc_hv_against_fixed"] = bool(row["auc_hv"] > f["auc_hv"])
            row["wins_coverage_against_fixed"] = bool(row["direction_coverage"] > f["direction_coverage"])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df):
    grouped = df.groupby(["case_name", "method"])
    rows = []
    for (case_name, method), g in grouped:
        fixed = df[(df["case_name"] == case_name) & (df["method"] == "Fixed Pareto-QAOA")].set_index("seed")
        improvements = []
        for _, row in g.iterrows():
            if row["seed"] in fixed.index:
                improvements.append(row["final_hv"] - fixed.loc[row["seed"], "final_hv"])
        rows.append({
            "case_name": case_name,
            "method": method,
            "mean_final_hv": g["final_hv"].mean(),
            "std_final_hv": g["final_hv"].std(ddof=0),
            "mean_auc_hv": g["auc_hv"].mean(),
            "std_auc_hv": g["auc_hv"].std(ddof=0),
            "mean_normalized_final_hv": g["normalized_final_hv"].mean(),
            "std_normalized_final_hv": g["normalized_final_hv"].std(ddof=0),
            "mean_normalized_auc_hv": g["normalized_auc_hv"].mean(),
            "std_normalized_auc_hv": g["normalized_auc_hv"].std(ddof=0),
            "mean_direction_coverage": g["direction_coverage"].mean(),
            "std_direction_coverage": g["direction_coverage"].std(ddof=0),
            "win_rate_final_hv_vs_fixed": g["wins_final_hv_against_fixed"].mean(),
            "win_rate_auc_hv_vs_fixed": g["wins_auc_hv_against_fixed"].mean(),
            "win_rate_coverage_vs_fixed": g["wins_coverage_against_fixed"].mean(),
            "median_improvement_vs_fixed": float(np.median(improvements)) if improvements else np.nan,
            "best_seed": int(g.sort_values("final_hv", ascending=False).iloc[0]["seed"]),
            "worst_seed": int(g.sort_values("final_hv", ascending=True).iloc[0]["seed"]),
        })
    return pd.DataFrame(rows)


def sign_test(diffs):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.abs(diffs) > 1e-12]
    n = len(diffs)
    if n == 0:
        return 1.0
    wins = int((diffs > 0).sum())
    tail = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * min(tail, 1.0 - tail + math.comb(n, wins) / (2 ** n))))


def bootstrap_ci(diffs, seed=0, reps=1000):
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(reps)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def statistical_tests(df):
    rows = []
    for case_name, g in df.groupby("case_name"):
        pivot = g[g["method"].isin(["Fixed Pareto-QAOA", "Full AI-Adaptive Pareto-QAOA"])].pivot_table(index="seed", columns="method", values=["final_hv", "auc_hv", "direction_coverage", "archive_size"])
        if "Full AI-Adaptive Pareto-QAOA" not in pivot["final_hv"] or "Fixed Pareto-QAOA" not in pivot["final_hv"]:
            continue
        for metric in ["final_hv", "auc_hv", "direction_coverage", "archive_size"]:
            diffs = pivot[metric]["Full AI-Adaptive Pareto-QAOA"] - pivot[metric]["Fixed Pareto-QAOA"]
            lo, hi = bootstrap_ci(diffs)
            rows.append({
                "case_name": case_name,
                "metric": metric,
                "mean_difference": float(diffs.mean()),
                "median_difference": float(diffs.median()),
                "std_difference": float(diffs.std(ddof=0)),
                "win_rate": float((diffs > 0).mean()),
                "paired_sign_test_p": sign_test(diffs),
                "bootstrap_ci95_low": lo,
                "bootstrap_ci95_high": hi,
                "wilcoxon_p": np.nan,
            })
    return pd.DataFrame(rows)

def plot_metric(summary, metric, path, title):
    if len(summary) == 0:
        return
    plt.figure(figsize=(10, 4.8))
    for method, g in summary.groupby("method"):
        plt.plot(g["case_name"], g[metric], marker="o", label=method)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_history(history, path_prefix="figures"):
    if len(history) == 0:
        return
    full = history[history["method"] == "Full AI-Adaptive Pareto-QAOA"]
    if len(full) == 0:
        full = history
    sample = full[(full["case_name"] == full["case_name"].iloc[0]) & (full["seed"] == full["seed"].iloc[0])]
    plt.figure(figsize=(7, 4))
    plt.plot(sample["iteration"], sample["w_throughput"], marker="o", label="throughput")
    plt.plot(sample["iteration"], sample["w_interference"], marker="o", label="interference")
    plt.plot(sample["iteration"], sample["w_energy"], marker="o", label="energy")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(f"{path_prefix}/adaptive_weight_timeline.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(sample["iteration"], sample["hv_before"], label="before")
    plt.plot(sample["iteration"], sample["hv_after"], label="after")
    plt.bar(sample["iteration"], sample["delta_hv"], alpha=0.4, label="delta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_prefix}/adaptive_hv_gain_timeline.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(sample["iteration"], sample["coverage_before"], marker="o", label="before")
    plt.plot(sample["iteration"], sample["coverage_after"], marker="o", label="after")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_prefix}/adaptive_coverage_timeline.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(sample["iteration"], sample["reward"], marker="o", label="reward")
    plt.plot(sample["iteration"], sample["selection_score"].fillna(0), marker="s", label="selection score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_prefix}/adaptive_reward_timeline.png", dpi=180)
    plt.close()

    heat = pd.crosstab(sample["selected_action"], sample["iteration"])
    plt.figure(figsize=(8, max(3, 0.35 * len(heat))))
    plt.imshow(heat, aspect="auto", cmap="Blues")
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xticks(range(len(heat.columns)), heat.columns)
    plt.tight_layout()
    plt.savefig(f"{path_prefix}/adaptive_action_heatmap.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    parser.add_argument("--all-main-cases", action="store_true")
    parser.add_argument("--force", action="store_true", help="rerun requested case/seed pairs even if checkpointed")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    cases = MAIN_CASES if args.all_main_cases else [args.case or "main_dense_small_cell"]
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    rows = []
    curves = []
    histories = []
    existing_path = Path("results/adaptive_qaoa_study.csv")
    history_path = Path("results/adaptive_action_history.csv")
    existing = pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame()
    completed = set()
    if len(existing):
        required_actual = {variant[0] for variant in VARIANTS} | {"Replay-Adaptive Schedule"}
        all_history_methods = {}
        if history_path.exists():
            try:
                hist = pd.read_csv(history_path, usecols=["case_name", "seed", "method"])
                for hidx, hg in hist.groupby(["case_name", "seed"]):
                    all_history_methods[hidx] = set(hg["method"])
            except Exception:
                all_history_methods = {}
        for idx, g in existing.groupby(["case_name", "seed"]):
            methods = set(g["method"])
            history_methods = all_history_methods.get(idx, set())
            if required_actual.issubset(methods) and required_actual.issubset(history_methods):
                completed.add(idx)
    for case_name in cases:
        for seed in args.seeds:
            if not args.force and (case_name, int(seed)) in completed:
                print(f"[adaptive-study] skip existing case={case_name} seed={seed}", flush=True)
                continue
            seed_rows = []
            seed_curves = []
            seed_histories = []
            setup_bundle = setup_case(case_name, seed)
            online_history = None
            for variant in VARIANTS:
                print(f"[adaptive-study] case={case_name} seed={seed} method={variant[0]}", flush=True)
                out = run_variant(case_name, seed, variant, setup_bundle=setup_bundle)
                curve = out.pop("curve")
                curves.append(curve)
                seed_curves.append(curve)
                variant_history = out.pop("history")
                histories.append(variant_history)
                seed_histories.append(variant_history)
                if variant[0] == "Full AI-Adaptive Pareto-QAOA":
                    online_history = variant_history
                rows.append(out)
                seed_rows.append(out)
            if online_history is None:
                raise RuntimeError("Full AI-Adaptive Pareto-QAOA history is required for replay.")
            print(f"[adaptive-study] case={case_name} seed={seed} method=Replay-Adaptive Schedule", flush=True)
            replay_out = run_replay_variant(case_name, seed, online_history, setup_bundle=setup_bundle)
            replay_curve = replay_out.pop("curve")
            curves.append(replay_curve)
            seed_curves.append(replay_curve)
            replay_history = replay_out.pop("history")
            histories.append(replay_history)
            seed_histories.append(replay_history)
            rows.append(replay_out)
            seed_rows.append(replay_out)
            existing = merge_study_rows(existing, seed_rows)
            existing.to_csv(existing_path, index=False)
            merge_history_rows(history_path, seed_histories)
            print(f"[adaptive-study] checkpoint saved case={case_name} seed={seed}", flush=True)
    df = merge_study_rows(existing, rows)
    df.to_csv("results/adaptive_qaoa_study.csv", index=False)
    summary = summarize(df)
    write_markdown_table(summary, "results/adaptive_qaoa_study_summary.md")
    summary.to_csv("results/adaptive_component_ablation.csv", index=False)
    write_markdown_table(summary, "results/adaptive_component_ablation.md")

    stats = statistical_tests(df)
    stats.to_csv("results/adaptive_statistical_tests.csv", index=False)
    write_markdown_table(stats, "results/adaptive_statistical_tests.md")

    histories_df = merge_history_rows(history_path, histories)
    plot_history(histories_df)
    plot_metric(summary, "mean_final_hv", "figures/adaptive_vs_fixed_final_hv.png", "Adaptive Study Final HV")
    plot_metric(summary, "mean_auc_hv", "figures/adaptive_vs_fixed_auc_hv.png", "Adaptive Study AUC-HV")
    plot_metric(summary, "mean_direction_coverage", "figures/adaptive_vs_fixed_coverage.png", "Adaptive Study Coverage")
    plot_metric(summary, "win_rate_final_hv_vs_fixed", "figures/adaptive_win_rate.png", "Win Rate vs Fixed QAOA")
    plot_metric(summary, "mean_final_hv", "figures/adaptive_component_ablation_final_hv.png", "Component Ablation Final HV")
    plot_metric(summary, "mean_auc_hv", "figures/adaptive_component_ablation_auc_hv.png", "Component Ablation AUC-HV")
    plot_metric(summary, "mean_direction_coverage", "figures/adaptive_component_ablation_coverage.png", "Component Ablation Coverage")

    if curves:
        all_curves = pd.concat(curves, ignore_index=True)
    else:
        all_curves = pd.DataFrame({"method": [], "iteration": [], "hv": []})
    qcurves = all_curves[all_curves["method"].isin(["Fixed Pareto-QAOA", "Full AI-Adaptive Pareto-QAOA"])]
    plt.figure(figsize=(7, 4))
    for method, g in qcurves.groupby("method"):
        mean_curve = g.groupby("iteration")["hv"].mean().reset_index()
        plt.plot(mean_curve["iteration"], mean_curve["hv"], marker="o", label=method)
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Normalized HV")
    plt.tight_layout()
    plt.savefig("figures/fixed_vs_adaptive_hv_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    for method, g in df[df["method"].isin(["Fixed Pareto-QAOA", "Full AI-Adaptive Pareto-QAOA"])].groupby("method"):
        plt.bar(method, g["archive_size"].mean())
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Mean archive size")
    plt.tight_layout()
    plt.savefig("figures/fixed_vs_adaptive_archive_growth.png", dpi=180)
    plt.close()

    replay_rows = []
    replay_methods = ["Fixed Pareto-QAOA", "Replay-Adaptive Schedule", "Full AI-Adaptive Pareto-QAOA"]
    for (case_name, seed), g in df[df["method"].isin(replay_methods)].groupby(["case_name", "seed"]):
        for method, label in [
            ("Fixed Pareto-QAOA", "Fixed cyclic schedule"),
            ("Replay-Adaptive Schedule", "Replay-Adaptive Schedule"),
            ("Full AI-Adaptive Pareto-QAOA", "Online AI-Adaptive Pareto-QAOA"),
        ]:
            if method in set(g["method"]):
                row = g[g["method"] == method].iloc[0]
                replay_rows.append({"case_name": case_name, "seed": seed, "method": label, "final_hv": row["final_hv"], "auc_hv": row["auc_hv"]})
    replay = pd.DataFrame(replay_rows)
    replay.to_csv("results/adaptive_counterfactual_replay.csv", index=False)
    write_markdown_table(replay.groupby("method", as_index=False)[["final_hv", "auc_hv"]].mean(), "results/adaptive_counterfactual_replay.md")
    plt.figure(figsize=(7, 4))
    plt.bar(replay.groupby("method")["final_hv"].mean().index, replay.groupby("method")["final_hv"].mean().values)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean normalized final HV")
    plt.tight_layout()
    plt.savefig("figures/adaptive_counterfactual_replay.png", dpi=180)
    plt.close()

    gain = stats[stats["metric"] == "final_hv"].copy()
    gain.to_csv("results/adaptive_qaoa_advantage_by_case.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.bar(gain["case_name"], gain["mean_difference"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Adaptive - Fixed final HV")
    plt.tight_layout()
    plt.savefig("figures/adaptive_gain_vs_conflict.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    cov = summary[summary["method"].isin(["Fixed Pareto-QAOA", "Full AI-Adaptive Pareto-QAOA"])]
    for method, g in cov.groupby("method"):
        plt.plot(g["case_name"], g["mean_direction_coverage"], marker="o", label=method)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Direction coverage")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/adaptive_objective_region_coverage.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    front_proxy = df[df["method"].isin(["Fixed Pareto-QAOA", "Full AI-Adaptive Pareto-QAOA"])]
    for method, g in front_proxy.groupby("method"):
        plt.scatter(g["archive_size"], g["final_hv"], label=method, alpha=0.75)
    plt.xlabel("Archive size")
    plt.ylabel("Normalized final HV")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/fixed_vs_adaptive_pareto_front.png", dpi=180)
    plt.close()

    evidence = [
        "# Evidence for AI-Adaptive Pareto-QAOA",
        "",
        "The AI controller is not an LLM making manual decisions. It is an online archive-aware policy that observes hypervolume improvement, direction coverage, duplicate rate, valid sample efficiency, and budget usage. It then selects scalarization weights, CVaR alpha, candidate diversity, and sampling resources for the next QAOA call.",
        "",
        "Interpretation is intentionally conservative: AI-Adaptive Pareto-QAOA is evaluated against Fixed Pareto-QAOA by final HV, AUC-HV, coverage, real component ablations, real counterfactual replay of recorded adaptive action sequences, and paired seed statistics. In the current short-budget setting, replayed adaptive schedules improve coverage and remain competitive in AUC-HV, while the online feedback loop does not dominate fixed scheduling on every seed.",
        "",
        summary.to_markdown(index=False),
    ]
    Path("results/adaptive_qaoa_evidence.md").write_text("\n".join(evidence), encoding="utf-8")
    Path("results/adaptive_action_explanations.md").write_text(evidence[2], encoding="utf-8")

    main = summary[summary["case_name"] == cases[0]]
    print("\nAdaptive QAOA study completed.")
    print(main[["method", "mean_final_hv", "mean_auc_hv", "win_rate_final_hv_vs_fixed", "win_rate_auc_hv_vs_fixed"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
