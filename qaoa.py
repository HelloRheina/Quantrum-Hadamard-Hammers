import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from problem import decode_bitstring, objective_vector, precompute_bitstring_costs
from pareto import dominated_substitution, update_archive
from metrics import normalized_hypervolume_mc


def apply_x_mixer(psi, beta, n_qubits):
    cos_beta = np.cos(beta)
    sin_beta = -1j * np.sin(beta)
    state = psi.reshape([2] * n_qubits)
    for q in range(n_qubits):
        state = np.moveaxis(state, q, 0)
        a0 = state[0].copy()
        a1 = state[1].copy()
        state[0] = cos_beta * a0 + sin_beta * a1
        state[1] = sin_beta * a0 + cos_beta * a1
        state = np.moveaxis(state, 0, q)
    return state.reshape(-1)


def run_statevector(costs, theta, depth):
    n_states = len(costs)
    n_qubits = int(np.log2(n_states))
    gammas = theta[:depth]
    betas = theta[depth:]
    psi = np.ones(n_states, dtype=np.complex128) / np.sqrt(n_states)
    for gamma, beta in zip(gammas, betas):
        psi *= np.exp(-1j * gamma * costs)
        psi = apply_x_mixer(psi, beta, n_qubits)
    return psi


def expected_cost(theta, costs, depth):
    psi = run_statevector(costs, theta, depth)
    return float((np.abs(psi) ** 2) @ costs)


def cvar_cost(theta, costs, depth, alpha=0.2, feasible_shape=None, channels=None):
    psi = run_feasible_statevector(costs, theta, depth, feasible_shape, channels) if feasible_shape is not None else run_statevector(costs, theta, depth)
    probs = np.abs(psi) ** 2
    order = np.argsort(costs)
    remaining = float(alpha)
    total = 0.0
    mass = 0.0
    for idx in order:
        take = min(remaining, float(probs[idx]))
        if take > 0:
            total += take * costs[idx]
            mass += take
            remaining -= take
        if remaining <= 1e-12:
            break
    return float(total / max(mass, 1e-12))


def optimize_theta(costs, depth=1, optimizer="COBYLA", maxiter=50, seed=0, initial_theta=None):
    rng = np.random.default_rng(seed)
    x0 = np.asarray(initial_theta, dtype=float) if initial_theta is not None else np.concatenate([rng.uniform(0, np.pi, depth), rng.uniform(0, np.pi / 2, depth)])
    if optimizer.lower() == "powell":
        res = minimize(lambda x: expected_cost(x, costs, depth), x0, method="Powell", options={"maxiter": maxiter})
    else:
        res = minimize(lambda x: expected_cost(x, costs, depth), x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.6})
    return res.x


def optimize_theta_scalar(costs, depth=1, maxiter=50, seed=0, initial_theta=None, objective="expected_cost", cvar_alpha=0.2, feasible_shape=None, channels=None, optimizer="cobyla", return_info=False):
    rng = np.random.default_rng(seed)
    x0 = np.asarray(initial_theta, dtype=float) if initial_theta is not None else np.concatenate([rng.uniform(0, np.pi, depth), rng.uniform(0, np.pi / 2, depth)])
    if objective == "cvar":
        fn = lambda x: cvar_cost(x, costs, depth, alpha=cvar_alpha, feasible_shape=feasible_shape, channels=channels)
    else:
        if feasible_shape is None:
            fn = lambda x: expected_cost(x, costs, depth)
        else:
            fn = lambda x: float((np.abs(run_feasible_statevector(costs, x, depth, feasible_shape, channels)) ** 2) @ costs)
    if maxiter <= 0:
        if return_info:
            return x0, {
                "best_loss": float(fn(x0)),
                "optimizer_iterations": 0,
                "optimizer_evaluations": 1,
            }
        return x0
    opt = optimizer.lower().replace("_", "-")
    if opt == "powell":
        res = minimize(fn, x0, method="Powell", options={"maxiter": maxiter})
    elif opt == "nelder-mead":
        res = minimize(fn, x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-4})
    elif opt == "random-restart":
        best_res = None
        restarts = max(2, min(5, maxiter // 4))
        per_restart = max(2, maxiter // restarts)
        for r in range(restarts):
            start = x0 if r == 0 else np.concatenate([rng.uniform(0, np.pi, depth), rng.uniform(0, np.pi / 2, depth)])
            cand = minimize(fn, start, method="COBYLA", options={"maxiter": per_restart, "rhobeg": 0.6})
            if best_res is None or float(cand.fun) < float(best_res.fun):
                best_res = cand
        res = best_res
    else:
        res = minimize(fn, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.6})
    if return_info:
        return res.x, {
            "best_loss": float(res.fun) if hasattr(res, "fun") else float(fn(res.x)),
            "optimizer_iterations": int(getattr(res, "nit", getattr(res, "nfev", maxiter))),
            "optimizer_evaluations": int(getattr(res, "nfev", maxiter)),
        }
    return res.x


def apply_feasible_mixer(psi, beta, users, channels):
    adjacency = np.ones((channels, channels), dtype=float) - np.eye(channels)
    unitary = expm(-1j * beta * adjacency)
    state = psi.reshape([channels] * users)
    for axis in range(users):
        state = np.moveaxis(state, axis, 0)
        flat = state.reshape(channels, -1)
        flat = unitary @ flat
        state = flat.reshape([channels] + [channels] * (users - 1))
        state = np.moveaxis(state, 0, axis)
    return state.reshape(-1)


def run_feasible_statevector(costs, theta, depth, users, channels):
    n_states = len(costs)
    psi = np.ones(n_states, dtype=np.complex128) / np.sqrt(n_states)
    for gamma, beta in zip(theta[:depth], theta[depth:]):
        psi *= np.exp(-1j * gamma * costs)
        psi = apply_feasible_mixer(psi, beta, users, channels)
    return psi


def crowding_select(objectives, limit):
    if len(objectives) <= limit:
        return np.arange(len(objectives))
    vals = np.asarray(objectives, dtype=float)
    norm = (vals - vals.min(axis=0)) / (np.ptp(vals, axis=0) + 1e-12)
    selected = [int(np.argmax(np.linalg.norm(norm - norm.mean(axis=0), axis=1)))]
    while len(selected) < limit:
        remaining = [i for i in range(len(vals)) if i not in selected]
        dists = np.array([np.min(np.linalg.norm(norm[i] - norm[selected], axis=1)) for i in remaining])
        selected.append(remaining[int(np.argmax(dists))])
    return np.array(selected, dtype=int)


def qaoa_feasible_sample(problem, weights, depth=1, shots=512, n_most_prob=20, maxiter=50, seed=0, cache=None, initial_theta=None, obj_min=None, obj_max=None, qaoa_objective="cvar", cvar_alpha=0.2, candidate_selection="diverse", classical_optimizer="cobyla"):
    assignments = cache["assignments"]
    objectives = cache["objectives"]
    if obj_min is None:
        obj_min = objectives.min(axis=0)
    if obj_max is None:
        obj_max = objectives.max(axis=0)
    normalized = (objectives - obj_min) / (np.asarray(obj_max) - np.asarray(obj_min) + 1e-12)
    costs = normalized @ np.asarray(weights, dtype=float)
    theta, opt_info = optimize_theta_scalar(
        costs,
        depth=depth,
        maxiter=maxiter,
        seed=seed,
        initial_theta=initial_theta,
        objective=qaoa_objective,
        cvar_alpha=cvar_alpha,
        feasible_shape=problem.users,
        channels=problem.channels,
        optimizer=classical_optimizer,
        return_info=True,
    )
    psi = run_feasible_statevector(costs, theta, depth, problem.users, problem.channels)
    probs = np.abs(psi) ** 2
    rng = np.random.default_rng(seed + 123)
    top_count = min(len(probs), max(n_most_prob, 2 * n_most_prob))
    top = np.argpartition(-probs, top_count - 1)[:top_count]
    sampled = rng.choice(len(probs), size=shots, replace=True, p=probs / probs.sum())
    idx = np.unique(np.concatenate([top, sampled]))
    if candidate_selection == "top":
        idx = top[np.argsort(-probs[top])[:n_most_prob]]
    elif candidate_selection == "diverse":
        cand_obj = objectives[idx]
        nd = []
        try:
            from pareto import non_dominated_filter
            nd = non_dominated_filter(cand_obj)
        except Exception:
            nd = np.arange(len(idx))
        selected_local = nd if len(nd) >= n_most_prob else np.arange(len(idx))
        selected_local = crowding_select(cand_obj[selected_local], min(n_most_prob, len(selected_local)))
        base = nd if len(nd) >= n_most_prob else np.arange(len(idx))
        idx = idx[base[selected_local]]
    else:
        idx = idx[np.argsort(-probs[idx])[:n_most_prob]]
    sol = assignments[idx]
    obj = objectives[idx]
    sol, obj = update_archive([], [], sol, obj)
    duplicate_rate = 1.0 - len({tuple(x) for x in assignments[idx]}) / max(1, len(idx))
    return {
        "solutions": sol,
        "objectives": obj,
        "theta": theta,
        "probabilities": probs,
        "checked": len(idx),
        "evaluations": len(idx),
        "feasibility_rate": 1.0,
        "duplicate_rate": duplicate_rate,
        "valid_sample_efficiency": len(sol) / max(1, len(idx)),
        "candidate_solutions": assignments[idx],
        "candidate_objectives": objectives[idx],
        "best_loss": opt_info["best_loss"],
        "optimizer_iterations": opt_info["optimizer_iterations"],
        "optimizer_evaluations": opt_info["optimizer_evaluations"],
    }


def archive_hv_objective(theta, costs, decoded_objectives, depth, reference, n_most_prob, substitution_factor, seed, obj_min=None, obj_max=None):
    psi = run_statevector(costs, theta, depth)
    probs = np.abs(psi) ** 2
    candidate_limit = min(len(probs), substitution_factor * n_most_prob)
    top = np.argpartition(-probs, candidate_limit - 1)[:candidate_limit]
    ordered = top[np.argsort(-probs[top])]
    archive_o = decoded_objectives[ordered]
    if obj_min is None or obj_max is None:
        obj_min = decoded_objectives.min(axis=0)
        obj_max = decoded_objectives.max(axis=0)
    return -normalized_hypervolume_mc(archive_o, reference, obj_min, obj_max, samples=4000, seed=seed)


def optimize_theta_for_hv(costs, decoded_objectives, depth=1, reference=None, maxiter=50, n_most_prob=20, substitution_factor=2, seed=0, initial_theta=None, obj_min=None, obj_max=None):
    rng = np.random.default_rng(seed)
    x0 = np.asarray(initial_theta, dtype=float) if initial_theta is not None else np.concatenate([rng.uniform(0, np.pi, depth), rng.uniform(0, np.pi / 2, depth)])
    if reference is None:
        return optimize_theta(costs, depth=depth, maxiter=maxiter, seed=seed, initial_theta=x0)
    objective = lambda x: archive_hv_objective(x, costs, decoded_objectives, depth, reference, n_most_prob, substitution_factor, seed, obj_min, obj_max)
    res = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.6})
    return res.x


def qaoa_sample(
    problem,
    weights,
    depth=1,
    penalty=20.0,
    shots=512,
    n_most_prob=20,
    maxiter=50,
    seed=0,
    reference=None,
    substitution_factor=2,
    cache=None,
    initial_theta=None,
    obj_min=None,
    obj_max=None,
    qaoa_objective="expected_cost",
    cvar_alpha=0.2,
    qaoa_encoding="onehot_penalty",
    candidate_selection="diverse",
    classical_optimizer="cobyla",
):
    if qaoa_encoding == "feasible_mixer":
        return qaoa_feasible_sample(problem, weights, depth, shots, n_most_prob, maxiter, seed, cache, initial_theta, obj_min, obj_max, qaoa_objective, cvar_alpha, candidate_selection, classical_optimizer)
    if cache is None:
        bits, decoded, feasible, costs = precompute_bitstring_costs(problem, weights, penalty=penalty, repair=True)
        decoded_objectives = np.array([objective_vector(problem, a) for a in decoded])
    else:
        bits = cache["bits"]
        decoded = cache["decoded"]
        feasible = cache["feasible"]
        decoded_objectives = cache["objectives"]
    if obj_min is None:
        obj_min = decoded_objectives.min(axis=0)
    if obj_max is None:
        obj_max = decoded_objectives.max(axis=0)
    normalized_objectives = (decoded_objectives - obj_min) / (np.asarray(obj_max) - np.asarray(obj_min) + 1e-12)
    costs = normalized_objectives @ np.asarray(weights, dtype=float) + penalty * cache["violations"] if cache is not None else costs
    if qaoa_objective == "archive_hv":
        theta = optimize_theta_for_hv(costs, decoded_objectives, depth=depth, reference=reference, maxiter=maxiter, n_most_prob=n_most_prob, substitution_factor=substitution_factor, seed=seed, initial_theta=initial_theta, obj_min=obj_min, obj_max=obj_max)
    elif qaoa_objective == "cvar":
        theta, opt_info = optimize_theta_scalar(costs, depth=depth, maxiter=maxiter, seed=seed, initial_theta=initial_theta, objective="cvar", cvar_alpha=cvar_alpha, optimizer=classical_optimizer, return_info=True)
    else:
        theta, opt_info = optimize_theta_scalar(costs, depth=depth, maxiter=maxiter, seed=seed, initial_theta=initial_theta, objective="expected_cost", optimizer=classical_optimizer, return_info=True)
    psi = run_statevector(costs, theta, depth)
    probs = np.abs(psi) ** 2
    rng = np.random.default_rng(seed + 123)
    sampled = rng.choice(len(probs), size=shots, replace=True, p=probs / probs.sum())
    candidate_limit = min(len(probs), substitution_factor * n_most_prob)
    top = np.argpartition(-probs, candidate_limit - 1)[:candidate_limit]
    idx = np.unique(np.concatenate([sampled, top]))
    ordered = idx[np.argsort(-probs[idx])]
    candidates = []
    candidate_objectives = {}
    feasible_hits = 0
    unique_assignments = set()
    for i in ordered:
        assignment, ok = decode_bitstring(bits[i], problem, repair=True)
        feasible_hits += int(feasible[i])
        key = tuple(int(x) for x in assignment)
        unique_assignments.add(key)
        candidate_objectives[key] = decoded_objectives[i]
        candidates.append((assignment, float(probs[i])))
    cand_s, cand_o, checked = dominated_substitution(
        candidates,
        lambda a: candidate_objectives.get(tuple(int(x) for x in a), objective_vector(problem, a)),
        n_most_prob=n_most_prob,
        substitution_factor=substitution_factor,
    )
    duplicate_rate = 1.0 - len(unique_assignments) / max(1, len(candidates))
    return {
        "solutions": cand_s,
        "objectives": cand_o,
        "theta": theta,
        "probabilities": probs,
        "checked": checked,
        "evaluations": checked,
        "feasibility_rate": feasible_hits / max(1, len(ordered)),
        "duplicate_rate": duplicate_rate,
        "best_loss": opt_info["best_loss"] if qaoa_objective != "archive_hv" else float("nan"),
        "optimizer_iterations": opt_info["optimizer_iterations"] if qaoa_objective != "archive_hv" else maxiter,
        "optimizer_evaluations": opt_info["optimizer_evaluations"] if qaoa_objective != "archive_hv" else maxiter,
    }
