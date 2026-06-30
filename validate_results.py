from pathlib import Path
import sys
import numpy as np
import pandas as pd


REQUIRED_RESULTS = [
    "results/method_summary.csv",
    "results/archive_qaoa.csv",
    "results/archive_baselines.csv",
    "results/adaptive_history.csv",
    "results/ablation_summary.csv",
    "results/classical_optimizer_ablation.csv",
    "results/report.md",
]

REQUIRED_FIGURES = [
    "figures/method_comparison.png",
    "figures/hv_convergence_evals.png",
    "figures/auc_hv_comparison.png",
    "figures/classical_optimizer_ablation.png",
    "figures/pareto_3d.png",
]

REQUIRED_METHODS = {
    "Random",
    "Greedy",
    "Evolutionary MOO",
    "NSGA-II-style",
    "MOEA/D-style",
    "Classical-Seeded Evolutionary MOO",
    "Fixed Pareto-QAOA",
    "AI-Adaptive Pareto-QAOA",
    "Quantum-Seeded Evolutionary MOO",
    "AI-Pareto-QAOA Ensemble",
}

FORBIDDEN_REPORT_TERMS = [
    "switching cost",
    "previous time slot",
    "dynamic graph shift",
    "prior snapshot",
    "time-varying network",
    "dynamic spectrum allocation",
]


def np_finite(df, cols):
    present = [c for c in cols if c in df.columns]
    if len(present) != len(cols):
        return False
    return bool(np.isfinite(df[present].to_numpy(dtype=float)).all())

EXTENDED_REQUIRED_METHODS = {
    "Random",
    "NSGA-II-style",
    "MOEA/D-style",
    "Classical-Seeded Evolutionary MOO",
    "Quantum-Seeded Evolutionary MOO",
    "Fixed Pareto-QAOA",
    "AI-Adaptive Pareto-QAOA",
}

ADAPTIVE_HISTORY_COLUMNS = {
    "case_name", "seed", "iteration", "phase", "selected_action", "selected_weight_index",
    "w_throughput", "w_interference", "w_energy", "cvar_alpha", "shots", "n_most_prob",
    "substitution_factor", "candidate_selection_mode", "warm_start_mode", "parameter_transfer_mode",
    "hv_before", "hv_after", "delta_hv", "normalized_hv_before", "normalized_hv_after",
    "delta_normalized_hv", "coverage_before", "coverage_after", "delta_coverage",
    "archive_size_before", "archive_size_after", "delta_archive_size", "feasibility_rate",
    "valid_sample_efficiency", "duplicate_rate", "num_unique_samples",
    "num_non_dominated_candidates", "num_archive_additions", "reward", "selection_score",
    "exploration_bonus", "exploitation_score", "coverage_novelty_score", "stagnation_counter",
    "reason_for_action",
}

ADAPTIVE_STUDY_METHODS = {
    "Fixed Pareto-QAOA",
    "Adaptive weights only",
    "Adaptive CVaR only",
    "Adaptive candidate diversity only",
    "Adaptive resources only",
    "Adaptive weights + CVaR",
    "Adaptive weights + diversity",
    "Full AI-Adaptive Pareto-QAOA",
    "Replay-Adaptive Schedule",
}


def main():
    missing = [p for p in REQUIRED_RESULTS + REQUIRED_FIGURES if not Path(p).exists() or Path(p).stat().st_size == 0]
    if missing:
        print("Missing or empty files:", missing)
        return 1
    summary = pd.read_csv("results/method_summary.csv")
    needed_cols = {
        "case_name", "method", "final_hv", "normalized_final_hv", "auc_hv", "normalized_auc_hv",
        "hv_25", "hv_50", "hv_75", "archive_size", "direction_coverage", "feasibility_rate",
        "valid_sample_efficiency", "evals_to_90_ref_hv", "runtime", "eval_budget",
        "qaoa_seed_archive_fraction", "qaoa_descendant_archive_fraction",
    }
    problems = []
    if not REQUIRED_METHODS.issubset(set(summary["method"])):
        problems.append("method_summary.csv missing required methods")
    if not needed_cols.issubset(set(summary.columns)):
        problems.append("method_summary.csv missing required columns")
    if not summary["normalized_final_hv"].between(0, 1).all():
        problems.append("normalized_final_hv outside [0,1]")
    if (summary["archive_size"] < 0).any():
        problems.append("negative archive size")
    budgeted_methods = {
        "Random",
        "Greedy",
        "Evolutionary MOO",
        "NSGA-II-style",
        "MOEA/D-style",
        "Classical-Seeded Evolutionary MOO",
        "Quantum-Seeded Evolutionary MOO",
    }
    budgeted = summary[summary["method"].isin(budgeted_methods)]
    if len(budgeted) and (budgeted["objective_evaluations"] > budgeted["eval_budget"]).any():
        offenders = budgeted.loc[budgeted["objective_evaluations"] > budgeted["eval_budget"], "method"].tolist()
        problems.append(f"budgeted baselines exceed eval_budget: {offenders}")
    if Path("results/classical_optimizer_ablation.csv").exists():
        opt = pd.read_csv("results/classical_optimizer_ablation.csv")
        required_opts = {"powell", "cobyla", "nelder-mead", "random_restart"}
        if not required_opts.issubset(set(opt["optimizer"])):
            problems.append("classical_optimizer_ablation.csv missing optimizer rows")
        needed_opt_cols = {
            "final_hv", "auc_hv", "normalized_final_hv", "archive_size", "runtime",
            "optimizer_iterations", "objective_evaluations", "best_qaoa_loss",
        }
        if not needed_opt_cols.issubset(set(opt.columns)):
            problems.append("classical_optimizer_ablation.csv missing required columns")
    hist = pd.read_csv("results/adaptive_history.csv")
    if not ADAPTIVE_HISTORY_COLUMNS.issubset(set(hist.columns)):
        problems.append("adaptive_history.csv missing interpretable action columns")
    if Path("results/extended_benchmark_summary.csv").exists():
        ext = pd.read_csv("results/extended_benchmark_summary.csv")
        if not EXTENDED_REQUIRED_METHODS.issubset(set(ext["method"])):
            problems.append("extended_benchmark_summary.csv missing required methods")
        finite_cols = ["mean_final_hv", "mean_normalized_final_hv", "mean_auc_hv", "mean_normalized_auc_hv"]
        if not np_finite(ext, finite_cols):
            problems.append("extended benchmark normalized HV columns are not finite")
        if (ext["mean_runtime"] < 0).any():
            problems.append("extended benchmark runtime is negative")
        raw_path = Path("results/extended_benchmark_raw.csv")
        if raw_path.exists():
            raw_ext = pd.read_csv(raw_path)
            if {"objective_evaluations", "budget"}.issubset(raw_ext.columns):
                over = raw_ext[raw_ext["objective_evaluations"] > raw_ext["budget"]]
                if len(over):
                    problems.append("extended benchmark methods exceed case budget")
        for fig in [
            "figures/extended_benchmark_heatmap_auc.png",
            "figures/extended_benchmark_heatmap_final_hv.png",
            "figures/scalability_auc_hv.png",
            "figures/scalability_final_hv.png",
            "figures/scalability_runtime.png",
            "figures/scalability_normalized_auc_hv.png",
            "figures/difficulty_sweep_auc_hv.png",
            "figures/difficulty_sweep_final_hv.png",
            "figures/budget_sweep_auc_hv.png",
            "figures/budget_sweep_final_hv.png",
            "figures/budget_sweep_quantum_advantage.png",
            "figures/budget_sweep_adaptive_advantage.png",
            "figures/quantum_seeded_advantage_by_case.png",
            "figures/adaptive_qaoa_advantage_by_case.png",
        ]:
            if not Path(fig).exists() or Path(fig).stat().st_size == 0:
                problems.append(f"missing or empty extended figure: {fig}")
        if not Path("results/extended_benchmark_report.md").exists():
            problems.append("missing extended_benchmark_report.md")
    if Path("results/adaptive_qaoa_study.csv").exists():
        adaptive_study = pd.read_csv("results/adaptive_qaoa_study.csv")
        for idx, g in adaptive_study.groupby(["case_name", "seed"]):
            missing_methods = ADAPTIVE_STUDY_METHODS - set(g["method"])
            if missing_methods:
                problems.append(f"adaptive_qaoa_study.csv missing methods for {idx}: {sorted(missing_methods)}")
        action_history_path = Path("results/adaptive_action_history.csv")
        if action_history_path.exists():
            action_history = pd.read_csv(action_history_path)
            for idx, g in action_history.groupby(["case_name", "seed"]):
                if idx in set(adaptive_study[["case_name", "seed"]].itertuples(index=False, name=None)):
                    missing_methods = ADAPTIVE_STUDY_METHODS - set(g["method"])
                    if missing_methods:
                        problems.append(f"adaptive_action_history.csv missing methods for {idx}: {sorted(missing_methods)}")
        else:
            problems.append("missing adaptive_action_history.csv")
        for p in [
            "results/adaptive_component_ablation.csv",
            "results/adaptive_qaoa_evidence.md",
            "results/adaptive_counterfactual_replay.csv",
            "results/adaptive_statistical_tests.csv",
            "figures/fixed_vs_adaptive_hv_curve.png",
            "figures/adaptive_weight_timeline.png",
            "figures/adaptive_hv_gain_timeline.png",
            "figures/adaptive_coverage_timeline.png",
            "figures/adaptive_reward_timeline.png",
            "figures/adaptive_action_heatmap.png",
            "figures/adaptive_objective_region_coverage.png",
            "figures/fixed_vs_adaptive_pareto_front.png",
            "figures/fixed_vs_adaptive_archive_growth.png",
            "figures/adaptive_gain_vs_conflict.png",
            "figures/adaptive_counterfactual_replay.png",
            "figures/adaptive_component_ablation_final_hv.png",
            "figures/adaptive_component_ablation_auc_hv.png",
            "figures/adaptive_component_ablation_coverage.png",
        ]:
            if not Path(p).exists() or Path(p).stat().st_size == 0:
                problems.append(f"missing adaptive study artifact: {p}")
    text = Path("results/report.md").read_text(encoding="utf-8").lower()
    if "extended benchmark evidence" not in text:
        problems.append("report.md missing Extended Benchmark Evidence section")
    if "evidence for ai-adaptive pareto-qaoa" not in text:
        problems.append("report.md missing Evidence for AI-Adaptive Pareto-QAOA section")
    forbidden = []
    for term in FORBIDDEN_REPORT_TERMS:
        idx = text.find(term)
        if idx >= 0:
            window = text[max(0, idx - 40): idx + len(term) + 40]
            if not any(marker in window for marker in ["not", "no ", "does not", "no dynamic"]):
                forbidden.append(term)
    if forbidden:
        problems.append(f"forbidden report terms found: {forbidden}")
    if problems:
        print("Validation failed:")
        for p in problems:
            print("-", p)
        return 1
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
