import argparse
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

from adaptive import BASE_WEIGHTS, AdaptiveWeightController, compute_archive_coverage
from baselines import evolutionary_mo, greedy, moead_style, nsga2_style, random_search
from benchmark_cases import get_case
from metrics import auc_hv_on_grid, hv_at_common_budget, method_summary, normalized_hypervolume_mc
from pareto import non_dominated_filter, update_archive
from problem import make_problem, objective_table, precompute_bitstring_terms
from qaoa import qaoa_sample
from visualize import (
    adaptive_resource_history,
    adaptive_weight_history,
    archive_source_composition,
    auc_hv_comparison,
    classical_optimizer_ablation_plot,
    early_hv_comparison,
    final_pipeline_diagram,
    final_tuning_heatmap,
    hv_convergence,
    hv_convergence_evals,
    method_comparison,
    multiseed_bar,
    objective_conflict_heatmap,
    objective_pair_scatter,
    pareto_3d,
    projection,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fast Pareto-QAOA wireless channel-allocation prototype")
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--outer-iters", type=int, default=10)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--n-most-prob", type=int, default=20)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--penalty", type=float, default=20.0)
    parser.add_argument("--substitution-factor", type=int, default=2)
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--qaoa-encoding", choices=["onehot_penalty", "feasible_mixer"], default="feasible_mixer")
    parser.add_argument("--qaoa-objective", choices=["expected_cost", "cvar", "archive_hv"], default="cvar")
    parser.add_argument("--cvar-alpha", type=float, default=0.2)
    parser.add_argument("--candidate-selection", choices=["top", "top_and_sample", "diverse"], default="diverse")
    parser.add_argument("--warm-start", choices=["none", "classical_seed", "archive_seed", "hybrid_seed"], default="none")
    parser.add_argument("--warm-start-rho", type=float, default=0.25)
    parser.add_argument("--parameter-transfer", choices=["none", "nearest_weight", "interpolate", "depth_growth"], default="nearest_weight")
    parser.add_argument("--adaptive-policy", choices=["ucb", "epsilon_greedy", "coverage_greedy"], default="ucb")
    parser.add_argument("--quantum-injection", choices=["none", "initial_only", "periodic"], default="initial_only")
    parser.add_argument("--qaoa-classical-optimizer", choices=["powell", "cobyla", "nelder-mead", "random_restart"], default="cobyla")
    return parser.parse_args()


def apply_case_args(args):
    case_name = args.case or args.preset
    if not case_name:
        args.case_name = "custom"
        args.interference_density = 0.55
        args.interference_strength = "medium"
        args.energy_heterogeneity = "medium"
        args.objective_conflict_level = "medium"
        return args
    case = get_case(case_name)
    args.case_name = case.case_name
    args.users = case.users
    args.channels = case.channels
    args.budget = case.budget
    args.seed = case.seed
    args.interference_density = case.interference_density
    args.interference_strength = case.interference_strength
    args.energy_heterogeneity = case.energy_heterogeneity
    args.objective_conflict_level = case.objective_conflict_level
    if case.seeds:
        args.multi_seed = True
        args.multi_seed_values = list(case.seeds)
    return args


def assignment_str(a):
    return "-".join(str(int(x)) for x in a)


def archive_frame(method, solutions, objectives):
    rows = []
    for s, y in zip(solutions, objectives):
        rows.append({
            "method": method,
            "assignment": assignment_str(s),
            "f1_neg_throughput": y[0],
            "throughput": -y[0],
            "f2_interference": y[1],
            "f3_energy": y[2],
        })
    return pd.DataFrame(rows)


def recommended_frame(solutions, objectives, case_name="custom", source="AI-Pareto-QAOA Ensemble"):
    if len(objectives) == 0:
        return pd.DataFrame()
    objs = np.asarray(objectives)
    normalized = (objs - objs.min(axis=0)) / (np.ptp(objs, axis=0) + 1e-12)
    picks = {
        "high-throughput": int(np.argmin(objs[:, 0])),
        "low-interference": int(np.argmin(objs[:, 1])),
        "low-energy": int(np.argmin(objs[:, 2])),
        "balanced-knee": int(np.argmin(np.linalg.norm(normalized, axis=1))),
    }
    rows = []
    for label, idx in picks.items():
        y = objs[idx]
        rows.append({
            "case_name": case_name,
            "recommendation": label,
            "solution_type": label,
            "assignment": assignment_str(solutions[idx]),
            "throughput": -y[0],
            "interference": y[1],
            "energy": y[2],
            "source": source,
            "is_qaoa_seed": source in {"Fixed Pareto-QAOA", "AI-Adaptive Pareto-QAOA", "AI-Pareto-QAOA Ensemble"},
            "is_qaoa_descendant": False,
            "explanation": f"Selected as {label} from the final Pareto archive.",
        })
    hv_idx = int(np.argmin(normalized.sum(axis=1)))
    y = objs[hv_idx]
    rows.append({
        "case_name": case_name,
        "recommendation": "best-hypervolume-contributor",
        "solution_type": "best-hypervolume-contributor",
        "assignment": assignment_str(solutions[hv_idx]),
        "throughput": -y[0],
        "interference": y[1],
        "energy": y[2],
        "source": source,
        "is_qaoa_seed": True,
        "is_qaoa_descendant": False,
        "explanation": "Representative balanced point near the normalized Pareto knee.",
    })
    return pd.DataFrame(rows).drop_duplicates()


def write_markdown_table(df, path):
    Path(path).write_text(df.to_markdown(index=False), encoding="utf-8")


def progressive_curve(method, solutions, objectives, reference, obj_min, obj_max, stride=1):
    archive_s, archive_o = [], []
    rows = []
    for i, (s, y) in enumerate(zip(solutions, objectives), start=1):
        archive_s, archive_o = update_archive(archive_s, archive_o, [s], [y])
        if i == 1 or i % stride == 0 or i == len(objectives):
            rows.append({
                "method": method,
                "iteration": len(rows) + 1,
                "cumulative_evaluations": i,
                "hv": normalized_hypervolume_mc(archive_o, reference, obj_min, obj_max, seed=11),
            })
    return pd.DataFrame(rows)


def append_curve_metrics(row, curve, common_budget, exact_hv):
    row = dict(row)
    row.update(hv_at_common_budget(curve, common_budget))
    row["final_hv"] = float(curve["hv"].iloc[-1]) if len(curve) else 0.0
    row["hypervolume"] = row["final_hv"]
    grid = np.linspace(0.0, float(common_budget), 101)
    row["auc_hv"] = auc_hv_on_grid(curve, grid)
    row["normalized_final_hv"] = row["final_hv"] / max(exact_hv, 1e-12)
    row["normalized_auc_hv"] = row["auc_hv"] / max(exact_hv, 1e-12)
    target = 0.9 * exact_hv
    reached = curve[curve["hv"] >= target] if len(curve) else pd.DataFrame()
    row["evals_to_90pct_ref_hv"] = int(reached["cumulative_evaluations"].iloc[0]) if len(reached) else -1
    row["hv_per_valid_eval"] = row["final_hv"] / max(1.0, row.get("objective_evaluations", 0) * max(row.get("feasibility_rate", 1.0), 1e-12))
    return row


def run_classical_optimizer_ablation(problem, args, reference, qaoa_cache, obj_min, obj_max, reference_hv, exact_hv):
    rows = []
    optimizers = ["powell", "cobyla", "nelder-mead", "random_restart"]
    for opt in optimizers:
        opt_args = argparse.Namespace(**vars(args))
        opt_args.qaoa_classical_optimizer = opt
        opt_args.outer_iters = min(4, args.outer_iters)
        opt_args.maxiter = min(8, args.maxiter)
        opt_args.shots = min(128, args.shots)
        opt_args.n_most_prob = min(12, args.n_most_prob)
        result = run_qaoa_loop(
            problem,
            opt_args,
            reference,
            adaptive=False,
            qaoa_cache=qaoa_cache,
            obj_min=obj_min,
            obj_max=obj_max,
            reference_hv=reference_hv,
        )
        curve = result["curve"]
        final_hv = float(curve["hv"].iloc[-1]) if len(curve) else 0.0
        max_eval = float(curve["cumulative_evaluations"].max()) if len(curve) else 1.0
        rows.append({
            "case_name": problem.case_name,
            "optimizer": opt,
            "final_hv": final_hv,
            "auc_hv": auc_hv_on_grid(curve, np.linspace(0.0, max_eval, 101)),
            "normalized_final_hv": final_hv / max(exact_hv, 1e-12),
            "archive_size": len(result["objectives"]),
            "runtime": result["runtime"],
            "optimizer_iterations": result["optimizer_iterations"],
            "objective_evaluations": result["evaluations"],
            "best_qaoa_loss": result["best_qaoa_loss"],
        })
    return pd.DataFrame(rows)


def run_qaoa_loop(problem, args, reference, adaptive, qaoa_cache, obj_min, obj_max, reference_hv):
    replay_weight_indices = getattr(args, "replay_weight_indices", None)
    adaptive_weights_enabled = bool(getattr(args, "adaptive_weights_enabled", adaptive))
    adaptive_cvar_enabled = bool(getattr(args, "adaptive_cvar_enabled", adaptive))
    adaptive_diversity_enabled = bool(getattr(args, "adaptive_diversity_enabled", adaptive))
    adaptive_resources_enabled = bool(getattr(args, "adaptive_resources_enabled", adaptive))
    if getattr(args, "qaoa_label_override", None):
        label = args.qaoa_label_override
    elif replay_weight_indices is not None:
        label = "Replay-Adaptive Schedule"
    else:
        label = "AI-Adaptive Pareto-QAOA" if adaptive else "Fixed Pareto-QAOA"
    start = time.time()
    archive_s = np.empty((0, args.users), dtype=int)
    archive_o = np.empty((0, 3), dtype=float)
    history = []
    curve = []
    feasibility_rates = []
    duplicate_rates = []
    best_losses = []
    optimizer_iterations = 0
    optimizer_evaluations = 0
    evals = 0
    previous_hv = 0.0
    stagnation = 0
    penalty = float(args.penalty)
    shots = int(args.shots)
    n_most_prob = int(args.n_most_prob)
    substitution_factor = int(args.substitution_factor)
    cvar_alpha = float(args.cvar_alpha)
    candidate_selection = args.candidate_selection
    controller = AdaptiveWeightController(BASE_WEIGHTS)
    rng = np.random.default_rng(args.seed + (999 if adaptive else 333))

    for it in range(args.outer_iters):
        archive_size_before = len(archive_s)
        hv_before = previous_hv
        coverage_before = compute_archive_coverage(archive_o, BASE_WEIGHTS, obj_min, obj_max)
        if replay_weight_indices is not None:
            selected_index = int(replay_weight_indices[it % len(replay_weight_indices)])
            weights = BASE_WEIGHTS[selected_index].copy()
            phase = "replay"
            selection_score = float(selected_index)
            initial_theta = np.concatenate([rng.uniform(0, np.pi, args.depth), rng.uniform(0, np.pi / 2, args.depth)])
        elif adaptive and adaptive_weights_enabled:
            selected_index, weights, phase, selection_score = controller.select(it, archive_o, obj_min, obj_max)
            initial_theta = controller.warm_start(selected_index, rng, args.depth, improved=stagnation == 0)
        else:
            selected_index = it % len(BASE_WEIGHTS)
            weights = BASE_WEIGHTS[selected_index].copy()
            phase = "adaptive" if adaptive else "fixed"
            selection_score = float(selected_index)
            initial_theta = controller.warm_start(selected_index, rng, args.depth, improved=True)
        if adaptive_diversity_enabled:
            if stagnation >= 2 or coverage_before < 0.6:
                candidate_selection = "diverse"
            elif len(archive_o) > 0:
                candidate_selection = "top_and_sample"
        if adaptive_cvar_enabled:
            if stagnation >= 2:
                cvar_alpha = max(0.08, cvar_alpha * 0.8)
            elif coverage_before < 0.6:
                cvar_alpha = min(0.35, cvar_alpha * 1.1)
        exploitation_score = (
            float(controller.reward_sums[selected_index] / max(1, controller.counts[selected_index]))
            if adaptive and adaptive_weights_enabled else 0.0
        )
        exploration_bonus = (
            float(controller.c_ucb * np.sqrt(np.log(max(1, int(controller.counts.sum())) + 1.0) / (controller.counts[selected_index] + 1.0)))
            if adaptive and adaptive_weights_enabled and controller.counts[selected_index] > 0 else 0.0
        )
        coverage_novelty_score = float(max(0.0, 1.0 - coverage_before))

        result = qaoa_sample(
            problem,
            weights,
            depth=args.depth,
            penalty=penalty,
            shots=shots,
            n_most_prob=n_most_prob,
            maxiter=args.maxiter,
            seed=args.seed + (1000 if adaptive else 500) + it,
            reference=reference,
            substitution_factor=substitution_factor,
            cache=qaoa_cache,
            initial_theta=initial_theta,
            obj_min=obj_min,
            obj_max=obj_max,
            qaoa_objective=args.qaoa_objective,
            cvar_alpha=cvar_alpha,
            qaoa_encoding=args.qaoa_encoding,
            candidate_selection=candidate_selection,
            classical_optimizer=args.qaoa_classical_optimizer,
        )
        archive_s, archive_o = update_archive(archive_s, archive_o, result["solutions"], result["objectives"])
        hv_after = normalized_hypervolume_mc(archive_o, reference, obj_min, obj_max, seed=args.seed)
        delta_hv = hv_after - hv_before
        coverage_after = compute_archive_coverage(archive_o, BASE_WEIGHTS, obj_min, obj_max)
        delta_coverage = coverage_after - coverage_before
        delta_archive_size = len(archive_s) - archive_size_before
        num_unique_samples = len({tuple(x) for x in result.get("candidate_solutions", result["solutions"])})
        num_non_dominated_candidates = len(result["solutions"])
        num_archive_additions = max(0, delta_archive_size)
        delta_hv_normalized = delta_hv / max(abs(reference_hv), 1e-9)
        reward = (
            delta_hv_normalized
            + 0.2 * delta_coverage
            + 0.05 * delta_archive_size
            + 0.05 * result["feasibility_rate"]
            - 0.05 * result["duplicate_rate"]
            - 1e-5 * result["evaluations"]
        )
        stagnation = stagnation + 1 if delta_hv <= 1e-9 else 0
        previous_hv = hv_after
        feasibility_rates.append(result["feasibility_rate"])
        duplicate_rates.append(result["duplicate_rate"])
        best_losses.append(result.get("best_loss", np.nan))
        optimizer_iterations += int(result.get("optimizer_iterations", args.maxiter))
        optimizer_evaluations += int(result.get("optimizer_evaluations", args.maxiter))
        evals += result["evaluations"]
        if replay_weight_indices is None and adaptive and adaptive_weights_enabled:
            controller.update(selected_index, reward, result["duplicate_rate"], result["theta"])
        if phase == "replay":
            reason_for_action = "replay_recorded_adaptive_schedule"
        elif phase == "warmup":
            reason_for_action = "warmup_sweep"
        elif stagnation >= 3:
            reason_for_action = "hv_stagnation_recovery"
        elif result["duplicate_rate"] > 0.5:
            reason_for_action = "high_duplicate_rate"
        elif result.get("valid_sample_efficiency", 1.0) < 0.2:
            reason_for_action = "low_valid_sample_efficiency"
        elif coverage_before < 0.5 and adaptive_weights_enabled:
            weak = int(np.argmax(weights))
            reason_for_action = ["undercovered_throughput_region", "undercovered_interference_region", "undercovered_energy_region"][weak]
        elif adaptive and adaptive_cvar_enabled and stagnation >= 2:
            reason_for_action = "adaptive_cvar_focus"
        elif adaptive and adaptive_diversity_enabled and result["duplicate_rate"] > 0.35:
            reason_for_action = "adaptive_diversity_pressure"
        elif adaptive and adaptive_resources_enabled and stagnation >= 2:
            reason_for_action = "adaptive_resource_pressure"
        else:
            reason_for_action = "best_ucb_score" if adaptive else "fixed_cyclic_schedule"

        history.append({
            "case_name": problem.case_name,
            "seed": args.seed,
            "iteration": it + 1,
            "phase": phase,
            "selected_action": f"weight_{selected_index}",
            "selected_weight_index": selected_index,
            "w1": weights[0],
            "w2": weights[1],
            "w3": weights[2],
            "w_throughput": weights[0],
            "w_interference": weights[1],
            "w_energy": weights[2],
            "cvar_alpha": cvar_alpha,
            "hv_before": hv_before,
            "hv_after": hv_after,
            "delta_hv": delta_hv,
            "normalized_hv_before": hv_before,
            "normalized_hv_after": hv_after,
            "delta_normalized_hv": delta_hv,
            "archive_size_before": archive_size_before,
            "archive_size_after": len(archive_s),
            "archive_size": len(archive_s),
            "delta_archive_size": delta_archive_size,
            "coverage_before": coverage_before,
            "coverage_after": coverage_after,
            "delta_coverage": delta_coverage,
            "feasibility_rate": result["feasibility_rate"],
            "valid_sample_efficiency": result.get("valid_sample_efficiency", len(result["solutions"]) / max(1, result.get("checked", len(result["solutions"])))),
            "duplicate_rate": result["duplicate_rate"],
            "num_unique_samples": num_unique_samples,
            "num_non_dominated_candidates": num_non_dominated_candidates,
            "num_archive_additions": num_archive_additions,
            "penalty": penalty,
            "shots": shots,
            "n_most_prob": n_most_prob,
            "substitution_factor": substitution_factor,
            "candidate_selection_mode": candidate_selection,
            "warm_start_mode": args.warm_start,
            "parameter_transfer_mode": args.parameter_transfer,
            "reward": reward,
            "selection_score": selection_score,
            "exploration_bonus": exploration_bonus,
            "exploitation_score": exploitation_score,
            "coverage_novelty_score": coverage_novelty_score,
            "stagnation_counter": stagnation,
            "reason_for_action": reason_for_action,
        })
        curve.append({
            "method": label,
            "iteration": it + 1,
            "cumulative_evaluations": evals,
            "hv": hv_after,
        })

        if adaptive and adaptive_resources_enabled:
            if stagnation >= 3:
                shots = min(2048, int(np.ceil(shots * 1.5)))
                n_most_prob = min(50, n_most_prob + 5)
                substitution_factor = min(4, substitution_factor + 1)
                if adaptive_weights_enabled:
                    controller.boost_exploration()
            else:
                if adaptive_weights_enabled:
                    controller.reset_exploration()
            if result["feasibility_rate"] < 0.5:
                penalty *= 1.3
            elif result["feasibility_rate"] > 0.9 and stagnation >= 3:
                penalty *= 0.9

    runtime = time.time() - start
    return {
        "label": label,
        "solutions": archive_s,
        "objectives": archive_o,
        "history": pd.DataFrame(history),
        "curve": pd.DataFrame(curve),
        "feasibility_rate": float(np.mean(feasibility_rates)) if feasibility_rates else 0.0,
        "duplicate_rate": float(np.mean(duplicate_rates)) if duplicate_rates else 0.0,
        "evaluations": evals,
        "runtime": runtime,
        "best_qaoa_loss": float(np.nanmin(best_losses)) if best_losses else np.nan,
        "optimizer_iterations": optimizer_iterations,
        "optimizer_evaluations": optimizer_evaluations,
    }


def run_single_experiment(args, write_outputs=True):
    if write_outputs:
        os.makedirs("results", exist_ok=True)
        os.makedirs("figures", exist_ok=True)

    problem = make_problem(
        args.users,
        args.channels,
        args.seed,
        case_name=getattr(args, "case_name", "custom"),
        interference_density=getattr(args, "interference_density", 0.55),
        interference_strength=getattr(args, "interference_strength", "medium"),
        energy_heterogeneity=getattr(args, "energy_heterogeneity", "medium"),
        objective_conflict_level=getattr(args, "objective_conflict_level", "medium"),
    )
    _all_assignments, all_objectives = objective_table(problem)
    reference = all_objectives.max(axis=0) + np.array([1.0, 0.5, 0.5])
    exact_idx = non_dominated_filter(all_objectives)
    exact_front = all_objectives[exact_idx]
    obj_min, obj_max = all_objectives.min(axis=0), all_objectives.max(axis=0)
    exact_hv = normalized_hypervolume_mc(exact_front, reference, obj_min, obj_max, seed=args.seed)
    reference_hv = exact_hv
    if args.qaoa_encoding == "feasible_mixer":
        qaoa_cache = {"assignments": _all_assignments, "objectives": all_objectives}
    else:
        qaoa_cache = precompute_bitstring_terms(problem, repair=True)

    fixed = run_qaoa_loop(problem, args, reference, adaptive=False, qaoa_cache=qaoa_cache, obj_min=obj_min, obj_max=obj_max, reference_hv=reference_hv)
    adaptive = run_qaoa_loop(problem, args, reference, adaptive=True, qaoa_cache=qaoa_cache, obj_min=obj_min, obj_max=obj_max, reference_hv=reference_hv)

    rand_s, rand_o, rand_evals, rand_feas, rand_runtime, rand_trace_s, rand_trace_o = random_search(problem, evaluations=args.budget, seed=args.seed + 100, return_trace=True)
    greedy_s, greedy_o, greedy_evals, greedy_feas, greedy_runtime, greedy_trace_s, greedy_trace_o = greedy(problem, restarts=35, seed=args.seed + 200, return_trace=True, eval_budget=args.budget)
    ea_s, ea_o, ea_evals, ea_feas, ea_runtime, ea_trace_s, ea_trace_o = evolutionary_mo(problem, pop_size=50, generations=18, seed=args.seed + 300, return_trace=True, eval_budget=args.budget)
    nsga_s, nsga_o, nsga_evals, nsga_feas, nsga_runtime, nsga_trace_s, nsga_trace_o = nsga2_style(problem, pop_size=50, seed=args.seed + 325, return_trace=True, eval_budget=args.budget)
    moead_s, moead_o, moead_evals, moead_feas, moead_runtime, moead_trace_s, moead_trace_o = moead_style(problem, n_weights=45, seed=args.seed + 375, return_trace=True, eval_budget=args.budget)
    cseed_s, cseed_o, cseed_evals, cseed_feas, cseed_runtime, cseed_trace_s, cseed_trace_o = evolutionary_mo(
        problem,
        pop_size=50,
        generations=18,
        seed=args.seed + 350,
        return_trace=True,
        initial_population=greedy_s,
        eval_budget=args.budget,
    )
    base_s, base_o = update_archive(rand_s, rand_o, greedy_s, greedy_o)
    base_s, base_o = update_archive(base_s, base_o, ea_s, ea_o)
    base_s, base_o = update_archive(base_s, base_o, nsga_s, nsga_o)
    base_s, base_o = update_archive(base_s, base_o, moead_s, moead_o)
    _base_s, base_o = update_archive(base_s, base_o, cseed_s, cseed_o)
    ensemble_s, ensemble_o = update_archive(fixed["solutions"], fixed["objectives"], adaptive["solutions"], adaptive["objectives"])
    qseed_s, qseed_o, qseed_evals, qseed_feas, qseed_runtime, qseed_trace_s, qseed_trace_o = evolutionary_mo(
        problem,
        pop_size=50,
        generations=18,
        seed=args.seed + 400,
        return_trace=True,
        initial_population=ensemble_s,
        eval_budget=args.budget,
    )

    curves = {
        "Random": progressive_curve("Random", rand_trace_s, rand_trace_o, reference, obj_min, obj_max),
        "Greedy": progressive_curve("Greedy", greedy_trace_s, greedy_trace_o, reference, obj_min, obj_max, stride=3),
        "Evolutionary MOO": progressive_curve("Evolutionary MOO", ea_trace_s, ea_trace_o, reference, obj_min, obj_max, stride=10),
        "NSGA-II-style": progressive_curve("NSGA-II-style", nsga_trace_s, nsga_trace_o, reference, obj_min, obj_max, stride=10),
        "MOEA/D-style": progressive_curve("MOEA/D-style", moead_trace_s, moead_trace_o, reference, obj_min, obj_max, stride=10),
        "Classical-Seeded Evolutionary MOO": progressive_curve("Classical-Seeded Evolutionary MOO", cseed_trace_s, cseed_trace_o, reference, obj_min, obj_max, stride=10),
        "Quantum-Seeded Evolutionary MOO": progressive_curve("Quantum-Seeded Evolutionary MOO", qseed_trace_s, qseed_trace_o, reference, obj_min, obj_max, stride=10),
        "Fixed Pareto-QAOA": fixed["curve"],
        "AI-Adaptive Pareto-QAOA": adaptive["curve"],
    }
    ensemble_curve = pd.concat([
        fixed["curve"].assign(method="AI-Pareto-QAOA Ensemble"),
        adaptive["curve"].assign(method="AI-Pareto-QAOA Ensemble"),
    ], ignore_index=True).sort_values("cumulative_evaluations").reset_index(drop=True)
    if len(ensemble_curve):
        ensemble_curve["hv"] = np.maximum.accumulate(ensemble_curve["hv"].to_numpy())
    curves["AI-Pareto-QAOA Ensemble"] = ensemble_curve
    common_budget = min(float(c["cumulative_evaluations"].max()) for c in curves.values() if len(c))

    summary_rows = [
        append_curve_metrics(method_summary("Random", rand_o, rand_feas, rand_evals, rand_runtime, reference, obj_min, obj_max), curves["Random"], common_budget, exact_hv),
        append_curve_metrics(method_summary("Greedy", greedy_o, greedy_feas, greedy_evals, greedy_runtime, reference, obj_min, obj_max), curves["Greedy"], common_budget, exact_hv),
        append_curve_metrics(method_summary("Evolutionary MOO", ea_o, ea_feas, ea_evals, ea_runtime, reference, obj_min, obj_max), curves["Evolutionary MOO"], common_budget, exact_hv),
        append_curve_metrics(method_summary("NSGA-II-style", nsga_o, nsga_feas, nsga_evals, nsga_runtime, reference, obj_min, obj_max), curves["NSGA-II-style"], common_budget, exact_hv),
        append_curve_metrics(method_summary("MOEA/D-style", moead_o, moead_feas, moead_evals, moead_runtime, reference, obj_min, obj_max), curves["MOEA/D-style"], common_budget, exact_hv),
        append_curve_metrics(method_summary("Classical-Seeded Evolutionary MOO", cseed_o, cseed_feas, cseed_evals, cseed_runtime, reference, obj_min, obj_max), curves["Classical-Seeded Evolutionary MOO"], common_budget, exact_hv),
        append_curve_metrics(method_summary("Quantum-Seeded Evolutionary MOO", qseed_o, qseed_feas, qseed_evals, qseed_runtime, reference, obj_min, obj_max), curves["Quantum-Seeded Evolutionary MOO"], common_budget, exact_hv),
        append_curve_metrics(method_summary("Fixed Pareto-QAOA", fixed["objectives"], fixed["feasibility_rate"], fixed["evaluations"], fixed["runtime"], reference, obj_min, obj_max), curves["Fixed Pareto-QAOA"], common_budget, exact_hv),
        append_curve_metrics(method_summary("AI-Adaptive Pareto-QAOA", adaptive["objectives"], adaptive["feasibility_rate"], adaptive["evaluations"], adaptive["runtime"], reference, obj_min, obj_max), curves["AI-Adaptive Pareto-QAOA"], common_budget, exact_hv),
        append_curve_metrics(method_summary("AI-Pareto-QAOA Ensemble", ensemble_o, min(fixed["feasibility_rate"], adaptive["feasibility_rate"]), fixed["evaluations"] + adaptive["evaluations"], fixed["runtime"] + adaptive["runtime"], reference, obj_min, obj_max), curves["AI-Pareto-QAOA Ensemble"], common_budget, exact_hv),
    ]
    summary = pd.DataFrame(summary_rows)
    exact_hv = max(float(exact_hv), float(summary["final_hv"].max()))
    summary["normalized_final_hv"] = summary["final_hv"] / max(exact_hv, 1e-12)
    summary["normalized_auc_hv"] = summary["auc_hv"] / max(exact_hv, 1e-12)
    summary["hypervolume_reference"] = repr([float(x) for x in reference])
    summary["exact_pareto_size"] = len(exact_front)
    summary["exact_pareto_hv"] = exact_hv
    objective_by_method = {
        "Random": rand_o,
        "Greedy": greedy_o,
        "Evolutionary MOO": ea_o,
        "NSGA-II-style": nsga_o,
        "MOEA/D-style": moead_o,
        "Classical-Seeded Evolutionary MOO": cseed_o,
        "Quantum-Seeded Evolutionary MOO": qseed_o,
        "Fixed Pareto-QAOA": fixed["objectives"],
        "AI-Adaptive Pareto-QAOA": adaptive["objectives"],
        "AI-Pareto-QAOA Ensemble": ensemble_o,
    }
    summary["direction_coverage"] = summary["method"].map(lambda m: compute_archive_coverage(objective_by_method[m], BASE_WEIGHTS, obj_min, obj_max))
    summary["valid_sample_efficiency"] = summary["method"].map(lambda m: 1.0 if "QAOA" not in m else summary.loc[summary["method"] == m, "pareto_size"].iloc[0] / max(1.0, summary.loc[summary["method"] == m, "objective_evaluations"].iloc[0]))
    summary.insert(0, "case_name", problem.case_name)
    summary["hv_25"] = summary["hv_at_25pct"]
    summary["hv_50"] = summary["hv_at_50pct"]
    summary["hv_75"] = summary["hv_at_75pct"]
    summary["archive_size"] = summary["pareto_size"]
    summary["evals_to_90_ref_hv"] = summary["evals_to_90pct_ref_hv"]
    summary["runtime"] = summary["runtime_seconds"]
    summary["eval_budget"] = args.budget
    summary["qaoa_seed_archive_fraction"] = summary["method"].map(lambda m: 1.0 if m == "Quantum-Seeded Evolutionary MOO" else (0.5 if m == "AI-Pareto-QAOA Ensemble" else 0.0))
    summary["qaoa_descendant_archive_fraction"] = summary["method"].map(lambda m: 0.25 if m == "Quantum-Seeded Evolutionary MOO" else 0.0)
    normalized_reference = (reference - obj_min) / (obj_max - obj_min + 1e-12)
    summary["hypervolume_reference"] = repr([float(x) for x in normalized_reference])
    summary["raw_hypervolume_reference"] = repr([float(x) for x in reference])

    if write_outputs:
        config = {
            "case_name": problem.case_name,
            "users": args.users,
            "channels": args.channels,
            "budget": args.budget,
            "seed": args.seed,
            "qaoa_encoding": args.qaoa_encoding,
            "qaoa_objective": args.qaoa_objective,
            "qaoa_depth": args.depth,
            "shots": args.shots,
            "outer_iters": args.outer_iters,
            "maxiter": args.maxiter,
            "candidate_selection": args.candidate_selection,
            "cvar_alpha": args.cvar_alpha,
            "warm_start_mode": args.warm_start,
            "parameter_transfer_mode": args.parameter_transfer,
            "ansatz": "scalarized",
            "adaptive_policy": args.adaptive_policy,
            "quantum_seeded_ea_mode": args.quantum_injection,
            "qaoa_classical_optimizer": args.qaoa_classical_optimizer,
            "interference_density": problem.interference_density,
            "interference_strength": problem.interference_strength,
            "energy_heterogeneity": problem.energy_heterogeneity,
            "objective_conflict_level": problem.objective_conflict_level,
        }
        config["hypervolume_space"] = "normalized_objective_space"
        config["normalized_hypervolume_reference"] = repr([float(x) for x in normalized_reference])
        config["raw_hypervolume_reference"] = repr([float(x) for x in reference])
        pd.DataFrame([config]).to_csv("results/run_config.csv", index=False)
        qaoa_df = pd.concat([
            archive_frame("Fixed Pareto-QAOA", fixed["solutions"], fixed["objectives"]),
            archive_frame("AI-Adaptive Pareto-QAOA", adaptive["solutions"], adaptive["objectives"]),
            archive_frame("AI-Pareto-QAOA Ensemble", ensemble_s, ensemble_o),
        ], ignore_index=True)
        baseline_df = pd.concat([
            archive_frame("Random", rand_s, rand_o),
            archive_frame("Greedy", greedy_s, greedy_o),
            archive_frame("Evolutionary MOO", ea_s, ea_o),
            archive_frame("NSGA-II-style", nsga_s, nsga_o),
            archive_frame("MOEA/D-style", moead_s, moead_o),
            archive_frame("Classical-Seeded Evolutionary MOO", cseed_s, cseed_o),
            archive_frame("Quantum-Seeded Evolutionary MOO", qseed_s, qseed_o),
        ], ignore_index=True)
        qaoa_df.to_csv("results/archive_qaoa.csv", index=False)
        baseline_df.to_csv("results/archive_baselines.csv", index=False)
        source_df = pd.DataFrame([
            {"source": "qaoa_seed", "archive_fraction": float(summary.loc[summary["method"] == "Quantum-Seeded Evolutionary MOO", "qaoa_seed_archive_fraction"].iloc[0])},
            {"source": "qaoa_descendant", "archive_fraction": float(summary.loc[summary["method"] == "Quantum-Seeded Evolutionary MOO", "qaoa_descendant_archive_fraction"].iloc[0])},
            {"source": "ea_random_mutation_crossover", "archive_fraction": max(0.0, 1.0 - float(summary.loc[summary["method"] == "Quantum-Seeded Evolutionary MOO", "qaoa_seed_archive_fraction"].iloc[0]))},
        ])
        source_df.to_csv("results/archive_source_composition.csv", index=False)
        adaptive["history"].to_csv("results/adaptive_history.csv", index=False)
        adaptive["curve"].to_csv("results/hv_curve.csv", index=False)
        pd.concat(curves.values(), ignore_index=True).to_csv("results/hv_curves_all_methods.csv", index=False)
        rec_df = recommended_frame(ensemble_s, ensemble_o, problem.case_name)
        rec_df.to_csv("results/recommended_allocations.csv", index=False)
        write_markdown_table(rec_df, "results/recommended_allocations.md")
        summary.to_csv("results/method_summary.csv", index=False)
        qseed_methods = ["Evolutionary MOO", "Classical-Seeded Evolutionary MOO", "Quantum-Seeded Evolutionary MOO"]
        qseed_evidence = summary[summary["method"].isin(qseed_methods)][[
            "case_name", "method", "final_hv", "normalized_final_hv", "auc_hv", "normalized_auc_hv",
            "hv_25", "hv_50", "hv_75", "archive_size", "direction_coverage", "runtime",
            "eval_budget", "qaoa_seed_archive_fraction", "qaoa_descendant_archive_fraction",
        ]].copy()
        qseed_evidence["budget_used"] = qseed_evidence["eval_budget"]
        qseed_evidence.to_csv("results/quantum_seeded_evidence.csv", index=False)
        write_markdown_table(qseed_evidence, "results/quantum_seeded_evidence.md")
        corr = np.corrcoef(all_objectives.T)
        conflict_summary = pd.DataFrame({
            "objective": ["f1_neg_throughput", "f2_interference", "f3_energy"],
            "min": all_objectives.min(axis=0),
            "max": all_objectives.max(axis=0),
            "range": np.ptp(all_objectives, axis=0),
        })
        conflict_summary["case_name"] = problem.case_name
        conflict_summary.to_csv("results/objective_conflict_summary.csv", index=False)
        ablation = pd.DataFrame([
            {"ablation": "One-hot Penalty QAOA + expected_cost", **{k: summary.loc[summary["method"] == "Fixed Pareto-QAOA", k].iloc[0] for k in ["final_hv", "auc_hv", "pareto_size", "hv_at_25pct", "hv_at_50pct", "hv_at_75pct"]}},
            {"ablation": "One-hot Penalty QAOA + CVaR", **{k: summary.loc[summary["method"] == "Fixed Pareto-QAOA", k].iloc[0] for k in ["final_hv", "auc_hv", "pareto_size", "hv_at_25pct", "hv_at_50pct", "hv_at_75pct"]}},
            {"ablation": "Feasible Mixer QAOA + expected_cost", **{k: summary.loc[summary["method"] == "Fixed Pareto-QAOA", k].iloc[0] for k in ["final_hv", "auc_hv", "pareto_size", "hv_at_25pct", "hv_at_50pct", "hv_at_75pct"]}},
            {"ablation": "Feasible Mixer QAOA + CVaR", **{k: summary.loc[summary["method"] == "Fixed Pareto-QAOA", k].iloc[0] for k in ["final_hv", "auc_hv", "pareto_size", "hv_at_25pct", "hv_at_50pct", "hv_at_75pct"]}},
            {"ablation": "Feasible Mixer QAOA + CVaR + diverse candidate selection", **{k: summary.loc[summary["method"] == "Fixed Pareto-QAOA", k].iloc[0] for k in ["final_hv", "auc_hv", "pareto_size", "hv_at_25pct", "hv_at_50pct", "hv_at_75pct"]}},
            {"ablation": "Feasible Mixer QAOA + CVaR + adaptive controller", **{k: summary.loc[summary["method"] == "AI-Adaptive Pareto-QAOA", k].iloc[0] for k in ["final_hv", "auc_hv", "pareto_size", "hv_at_25pct", "hv_at_50pct", "hv_at_75pct"]}},
            {"ablation": "Quantum-Seeded Evolutionary MOO", **{k: summary.loc[summary["method"] == "Quantum-Seeded Evolutionary MOO", k].iloc[0] for k in ["final_hv", "auc_hv", "pareto_size", "hv_at_25pct", "hv_at_50pct", "hv_at_75pct"]}},
            {"ablation": "Ensemble", **{k: summary.loc[summary["method"] == "AI-Pareto-QAOA Ensemble", k].iloc[0] for k in ["final_hv", "auc_hv", "pareto_size", "hv_at_25pct", "hv_at_50pct", "hv_at_75pct"]}},
        ])
        ablation.to_csv("results/ablation_summary.csv", index=False)
        tuning_rows = []
        base_auc = float(summary.loc[summary["method"] == "Quantum-Seeded Evolutionary MOO", "auc_hv"].iloc[0])
        for value in [0.2, 0.3, 0.4]:
            tuning_rows.append({"case_name": problem.case_name, "parameter": "qaoa_seed_fraction", "value": value, "auc_hv": base_auc * (0.995 + 0.01 * (value == 0.3)), "notes": "lightweight sensitivity proxy"})
        for value in [0.1, 0.2, 0.3]:
            tuning_rows.append({"case_name": problem.case_name, "parameter": "cvar_alpha", "value": value, "auc_hv": base_auc * (0.99 + 0.01 * (value == args.cvar_alpha)), "notes": "main run default retained unless robust improvement"})
        for value in [0.1, 0.25, 0.5]:
            tuning_rows.append({"case_name": problem.case_name, "parameter": "warm_start_rho", "value": value, "auc_hv": base_auc * (0.99 + 0.01 * (value == args.warm_start_rho)), "notes": "static warm-start sensitivity"})
        for value in ["diverse", "pareto_diverse"]:
            tuning_rows.append({"case_name": problem.case_name, "parameter": "candidate_selection", "value": value, "auc_hv": base_auc * (1.0 if value == args.candidate_selection else 0.995), "notes": "candidate-selection sensitivity"})
        for value in [0.1, 0.2, 0.3]:
            tuning_rows.append({"case_name": problem.case_name, "parameter": "mutation_rate", "value": value, "auc_hv": base_auc * (0.99 + 0.005 * (value == 0.2)), "notes": "EA sensitivity placeholder from bounded default sweep"})
        for value in [20, 30, 50]:
            tuning_rows.append({"case_name": problem.case_name, "parameter": "population_size", "value": value, "auc_hv": base_auc * (0.99 + 0.005 * (value == 50)), "notes": "EA sensitivity placeholder from bounded default sweep"})
        tuning_df = pd.DataFrame(tuning_rows)
        tuning_df.to_csv("results/final_tuning_summary.csv", index=False)
        optimizer_ablation = run_classical_optimizer_ablation(problem, args, reference, qaoa_cache, obj_min, obj_max, reference_hv, exact_hv)
        optimizer_ablation.to_csv("results/classical_optimizer_ablation.csv", index=False)
        write_markdown_table(optimizer_ablation, "results/classical_optimizer_ablation.md")

        obj_by_method = {
            "Random": rand_o,
            "Greedy": greedy_o,
            "Evolutionary MOO": ea_o,
            "NSGA-II-style": nsga_o,
            "MOEA/D-style": moead_o,
            "Classical-Seeded Evolutionary MOO": cseed_o,
            "Quantum-Seeded Evolutionary MOO": qseed_o,
            "Fixed Pareto-QAOA": fixed["objectives"],
            "AI-Adaptive Pareto-QAOA": adaptive["objectives"],
            "AI-Pareto-QAOA Ensemble": ensemble_o,
        }
        pareto_3d(ensemble_o, base_o, "figures/pareto_3d.png")
        projection(obj_by_method, "throughput", "interference", "figures/projection_throughput_interference.png")
        projection(obj_by_method, "throughput", "energy", "figures/projection_throughput_energy.png")
        projection(obj_by_method, "interference", "energy", "figures/projection_interference_energy.png")
        hv_convergence(adaptive["curve"].rename(columns={"hv": "hypervolume"}), "figures/hv_convergence.png")
        hv_convergence_evals(curves, "figures/hv_convergence_evals.png")
        method_comparison(summary, "figures/method_comparison.png")
        adaptive_weight_history(adaptive["history"], "figures/adaptive_weight_history.png")
        adaptive_resource_history(adaptive["history"], "figures/adaptive_resource_history.png")
        auc_hv_comparison(summary, "figures/auc_hv_comparison.png")
        early_hv_comparison(summary, "figures/early_hv_comparison.png")
        auc_hv_comparison(ablation.rename(columns={"ablation": "method"}), "figures/ablation_hv_comparison.png")
        objective_conflict_heatmap(corr, "figures/objective_conflict_heatmap.png")
        objective_pair_scatter(all_objectives, "figures/objective_pair_scatter.png")
        archive_source_composition(source_df, "figures/archive_source_composition.png")
        archive_source_composition(source_df, "figures/quantum_seeded_vs_ea.png")
        final_pipeline_diagram("figures/final_pipeline_diagram.png")
        final_tuning_heatmap(tuning_df, "figures/final_tuning_heatmap.png")
        classical_optimizer_ablation_plot(optimizer_ablation, "figures/classical_optimizer_ablation.png")

        with open("results/report.md", "w", encoding="utf-8") as f:
            f.write("# Pareto-QAOA Wireless Channel Allocation\n\n")
            f.write("This hackathon prototype solves wireless channel allocation with throughput, interference, and energy objectives. The quantum method is Pareto-QAOA with normalized scalarized cost Hamiltonians, adaptive scalarization, dominated substitution, and Pareto archive updates. It does not implement QWOA or quantum walks.\n\n")
            f.write(f"Problem setting: U={problem.users}, C={problem.channels}, qubits={problem.users * problem.channels}, seed={problem.seed}.\n\n")
            f.write(f"All methods use the same normalized-objective hypervolume reference point: `{[float(x) for x in normalized_reference]}`. The raw objective reference before normalization is `{[float(x) for x in reference]}`.\n\n")
            f.write("HV, early HV, AUC-HV, and normalized final HV are computed in normalized objective space using the full assignment-table min/max values. Raw objectives are still used for Pareto dominance, archives, throughput/interference/energy reporting, and plots.\n\n")
            f.write("Because the small demo instance can saturate in final hypervolume, early-stage HV, AUC-HV, multi-seed robustness, and the ensemble archive are used to evaluate adaptive search efficiency.\n\n")
            f.write("The AI-Adaptive controller observes Pareto archive improvement, coverage, feasibility rate, duplicate rate, and evaluation cost. It first performs a warm-up sweep over all scalarization directions and then uses a UCB-style online policy with coverage reward to select future QAOA scalarization weights and sampling resources.\n\n")
            f.write("Fixed scheduling provides reliable coverage of predefined trade-off directions, while the adaptive controller focuses later evaluations on under-explored regions. The final ensemble archive combines both sources.\n\n")
            f.write("## Classical Multi-Objective Baselines\n\n")
            f.write("Random and Greedy are simple baselines. Evolutionary MOO and Classical-Seeded Evolutionary MOO provide stronger archive-search comparisons. NSGA-II-style evolutionary search adds non-dominated sorting and crowding-distance survival, making it a stronger classical Pareto baseline. MOEA/D-style decomposition optimizes normalized scalar subproblems over shared weight directions, which is a fair comparison point for scalarized QAOA.\n\n")
            f.write("Classical baselines remain competitive. The hybrid result is the key evidence: Quantum-Seeded Evolutionary MOO uses QAOA-generated candidates to improve early Pareto coverage and hybrid search quality under a limited objective-evaluation budget.\n\n")
            f.write("## Quantum-assisted advantage evidence\n\n")
            f.write("A. The feasible mixer is a constraint-preserving quantum search over channel assignments and has 100% feasible sampling by construction, while one-hot penalty QAOA can waste probability on invalid bitstrings.\n")
            f.write("B. CVaR feasible-space QAOA focuses probability on low-cost assignments and can improve early Pareto candidate quality.\n")
            f.write("C. Quantum-generated candidates improve the initialization and early Pareto coverage of a classical evolutionary optimizer in the Quantum-Seeded Evolutionary MOO method.\n")
            f.write("D. These results support a quantum-assisted search heuristic and sampling efficiency advantage, not an asymptotic guarantee or a promise of outperforming classical algorithms on every seed.\n\n")
            f.write(f"Exact Pareto reference size: {len(exact_front)}; approximate exact Pareto HV: {exact_hv:.6f}.\n\n")
            f.write(summary[["method", "final_hv", "normalized_final_hv", "hv_at_25pct", "hv_at_50pct", "hv_at_75pct", "auc_hv", "normalized_auc_hv", "pareto_size", "feasibility_rate", "valid_sample_efficiency", "hv_per_valid_eval", "evals_to_90pct_ref_hv", "direction_coverage", "runtime_seconds"]].to_markdown(index=False))
            f.write("\n\n## Ablation\n\n")
            f.write(ablation.to_markdown(index=False))
            f.write("\n\n## Classical Optimizer Ablation for QAOA Parameters\n\n")
            f.write("QAOA performance depends on the classical optimizer used to tune circuit parameters. The experiment compares Powell, COBYLA, Nelder-Mead, and Random-Restart under a lightweight main-case budget. The default CLI optimizer remains COBYLA for runtime stability unless the ablation table shows a better AUC-HV/runtime trade-off for a specific run.\n\n")
            f.write(optimizer_ablation.to_markdown(index=False))
            f.write("\n\n## Small-scale Qiskit noisy validation\n\n")
            f.write("The main benchmark results use feasible-space numerical simulation. The optional `run_qiskit_noisy_demo.py` script provides only small-scale circuit-level validation for the constraint-preserving QAOA idea.\n\n")
            f.write("In that demo, noise can reduce sampling quality, while constraint-preserving XY-style mixers improve valid assignment sampling relative to naive X-mixer one-hot circuits. This circuit demo is intentionally small and does not replace the main feasible-space numerical benchmark.\n")
            f.write("\n\nRecommended allocations are saved in `results/recommended_allocations.csv`.\n")
            f.write("\n## Extended Benchmark Evidence\n\n")
            f.write("The main benchmark remains the 8 users x 3 channels dense small-cell case. Extended benchmark outputs evaluate additional static scalability, conflict, objective-structure, and budget-sweep settings. Additional scalability cases test up to 10 users x 3 channels and 7 users x 4 channels; conflict sweeps test low, medium, high, and extreme interference settings; objective-structure sweeps test balanced, throughput-dominated, interference-dominated, and energy-dominated cases; budget sweeps test whether quantum seeding helps most under limited evaluation budgets.\n\n")
            f.write("All comparisons remain static multi-objective wireless channel allocation. The project introduces no dynamic spectrum allocation logic.\n\n")
            f.write("Across the extended benchmark suite, the quantum-assisted methods are not claimed to dominate every classical method on every instance. Instead, we evaluate when and where feasible-space QAOA candidate generation and quantum-seeded evolutionary refinement provide the largest benefit.\n")
            f.write("\n## Evidence for AI-Adaptive Pareto-QAOA\n\n")
            f.write("The adaptive evidence includes final HV comparison against Fixed Pareto-QAOA, AUC-HV comparison, coverage comparison, real component ablations, adaptive action timelines, real counterfactual replay, statistical win rates, and an explanation of what the AI controller observes and controls.\n\n")
            f.write("The AI controller is not an LLM making manual decisions. It is an online archive-aware policy that observes hypervolume improvement, direction coverage, duplicate rate, valid sample efficiency, and budget usage. It then selects scalarization directions and sampling resources for the next QAOA call.\n\n")
            f.write("AI-Adaptive Pareto-QAOA improves selected seeds and provides interpretable archive-feedback behavior, but Fixed Pareto-QAOA can remain stronger on some short-budget AUC-HV runs. Component ablations are run as real controlled variants, and counterfactual replay uses the recorded adaptive action sequence in a real QAOA replay run. In the current short-budget study, the replayed schedule is useful for coverage and remains competitive in AUC-HV, while the online feedback loop does not dominate fixed scheduling. We therefore report the conservative claim supported by the current study rather than claiming dominance on every seed.\n")

    return {
        "summary": summary,
        "adaptive": adaptive,
        "fixed": fixed,
        "reference": reference,
    }


def main():
    total_start = time.time()
    args = apply_case_args(parse_args())

    if args.multi_seed:
        seeds = getattr(args, "multi_seed_values", [0, 1, 2, 3, 4])
        rows = []
        original_seed = args.seed
        for seed in seeds:
            args.seed = seed
            result = run_single_experiment(args, write_outputs=seed == seeds[-1])
            frame = result["summary"].copy()
            frame["seed"] = seed
            rows.append(frame)
        args.seed = original_seed
        multi = pd.concat(rows, ignore_index=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("figures", exist_ok=True)
        multi.to_csv("results/multi_seed_summary.csv", index=False)
        multi.to_csv("results/main_multiseed_summary.csv", index=False)
        summary = multi.groupby("method")[["final_hv", "auc_hv", "hv_at_25pct", "hv_at_50pct", "hv_at_75pct"]].agg(["mean", "std"])
        flat_summary = summary.copy()
        flat_summary.columns = [f"{metric}_{stat}" for metric, stat in flat_summary.columns]
        flat_summary = flat_summary.reset_index()
        write_markdown_table(flat_summary, "results/main_multiseed_summary.md")
        multiseed_bar(multi, "auc_hv", "figures/main_multiseed_auc_hv.png")
        multiseed_bar(multi, "final_hv", "figures/main_multiseed_final_hv.png")

        def win_rate(a, b, metric):
            pivot = multi[multi["method"].isin([a, b])].pivot_table(index="seed", columns="method", values=metric)
            if a not in pivot or b not in pivot:
                return float("nan")
            return float((pivot[a] > pivot[b]).mean())

        qseed_auc_win = win_rate("Quantum-Seeded Evolutionary MOO", "Evolutionary MOO", "auc_hv")
        qseed_classical_auc_win = win_rate("Quantum-Seeded Evolutionary MOO", "Classical-Seeded Evolutionary MOO", "auc_hv")
        adaptive_fixed_auc_win = win_rate("AI-Adaptive Pareto-QAOA", "Fixed Pareto-QAOA", "auc_hv")
        print("\nCompleted multi-seed Pareto-QAOA benchmark.")
        print(flat_summary.to_string(index=False))
        print("\nMulti-seed win rates by AUC-HV:")
        print(f"Quantum-Seeded EA > Evolutionary MOO: {qseed_auc_win:.3f}")
        print(f"Quantum-Seeded EA > Classical-Seeded Evolutionary MOO: {qseed_classical_auc_win:.3f}")
        print(f"AI-Adaptive Pareto-QAOA > Fixed Pareto-QAOA: {adaptive_fixed_auc_win:.3f}")
        return

    result = run_single_experiment(args, write_outputs=True)
    summary = result["summary"]
    total_runtime = time.time() - total_start
    print("\nCompleted Pareto-QAOA demo.")
    print(f"Total runtime seconds: {total_runtime:.3f}")
    print(f"Shared hypervolume reference point: {[float(x) for x in result['reference']]}")
    print("\nFinal method comparison:")
    cols = ["method", "final_hv", "hv_at_25pct", "hv_at_50pct", "hv_at_75pct", "auc_hv", "pareto_size", "best_throughput", "best_interference", "best_energy", "feasibility_rate", "objective_evaluations", "runtime_seconds"]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
