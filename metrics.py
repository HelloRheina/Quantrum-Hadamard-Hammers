import numpy as np


def hypervolume_mc(points, reference, samples=20000, seed=0, lower=None):
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return 0.0
    reference = np.asarray(reference, dtype=float)
    lower = np.asarray(lower, dtype=float) if lower is not None else np.minimum(points.min(axis=0), reference)
    span = reference - lower
    if np.any(span <= 0):
        return 0.0
    rng = np.random.default_rng(seed)
    cloud = lower + rng.random((samples, points.shape[1])) * span
    dominated = np.zeros(samples, dtype=bool)
    for p in points:
        dominated |= np.all(p <= cloud, axis=1)
    return float(dominated.mean() * np.prod(span))


def normalize_objectives(points, obj_min, obj_max):
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return points.reshape(0, len(obj_min))
    return (points - np.asarray(obj_min, dtype=float)) / (np.asarray(obj_max, dtype=float) - np.asarray(obj_min, dtype=float) + 1e-12)


def normalized_hypervolume_mc(points, reference, obj_min, obj_max, samples=20000, seed=0):
    norm_points = normalize_objectives(points, obj_min, obj_max)
    norm_reference = normalize_objectives(np.asarray(reference, dtype=float).reshape(1, -1), obj_min, obj_max)[0]
    return hypervolume_mc(norm_points, norm_reference, samples=samples, seed=seed, lower=np.zeros_like(norm_reference))


def hypervolume_2d(points, reference):
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return 0.0
    pts = points[np.argsort(points[:, 0])]
    hv = 0.0
    best_y = reference[1]
    for x, y in pts:
        if y < best_y:
            hv += max(0.0, reference[0] - x) * max(0.0, best_y - y)
            best_y = y
    return float(hv)


def method_summary(name, objectives, feasibility_rate, evaluations, runtime, reference, obj_min=None, obj_max=None):
    objectives = np.asarray(objectives, dtype=float)
    hv = normalized_hypervolume_mc(objectives, reference, obj_min, obj_max, seed=11) if obj_min is not None and obj_max is not None else hypervolume_mc(objectives, reference, seed=11)
    return {
        "method": name,
        "hypervolume": hv,
        "pareto_size": int(len(objectives)),
        "best_throughput": float(-objectives[:, 0].min()) if len(objectives) else 0.0,
        "best_interference": float(objectives[:, 1].min()) if len(objectives) else 0.0,
        "best_energy": float(objectives[:, 2].min()) if len(objectives) else 0.0,
        "feasibility_rate": float(feasibility_rate),
        "objective_evaluations": int(evaluations),
        "runtime_seconds": float(runtime),
    }


def hv_at_budget(curve_df, fractions=(0.25, 0.5, 0.75, 1.0)):
    if curve_df is None or len(curve_df) == 0:
        return {f"hv_at_{int(frac * 100)}pct": 0.0 for frac in fractions}
    evals = np.asarray(curve_df["cumulative_evaluations"], dtype=float)
    hvs = np.asarray(curve_df["hv"], dtype=float)
    max_eval = max(1.0, float(evals.max()))
    out = {}
    for frac in fractions:
        target = frac * max_eval
        eligible = np.flatnonzero(evals <= target)
        idx = int(eligible[-1]) if len(eligible) else 0
        out[f"hv_at_{int(frac * 100)}pct"] = float(hvs[idx])
    return out


def auc_hv(curve_df):
    if curve_df is None or len(curve_df) == 0:
        return 0.0
    evals = np.asarray(curve_df["cumulative_evaluations"], dtype=float)
    hvs = np.asarray(curve_df["hv"], dtype=float)
    order = np.argsort(evals)
    evals = evals[order]
    hvs = hvs[order]
    if len(evals) == 1:
        return float(hvs[0])
    area = np.trapezoid(hvs, evals)
    span = max(1.0, float(evals[-1] - evals[0]))
    return float(area / span)


def interpolate_curve(curve_df, grid):
    if curve_df is None or len(curve_df) == 0:
        return np.zeros(len(grid), dtype=float)
    evals = np.asarray(curve_df["cumulative_evaluations"], dtype=float)
    hvs = np.asarray(curve_df["hv"], dtype=float)
    order = np.argsort(evals)
    evals = np.concatenate([[0.0], evals[order]])
    hvs = np.concatenate([[0.0], hvs[order]])
    return np.interp(grid, evals, hvs, left=0.0, right=float(hvs[-1]))


def auc_hv_on_grid(curve_df, grid):
    values = interpolate_curve(curve_df, grid)
    if len(grid) <= 1:
        return float(values[-1]) if len(values) else 0.0
    return float(np.trapezoid(values, grid) / max(1.0, float(grid[-1] - grid[0])))


def hv_at_common_budget(curve_df, common_budget, fractions=(0.25, 0.5, 0.75, 1.0)):
    grid = np.asarray([common_budget * f for f in fractions], dtype=float)
    values = interpolate_curve(curve_df, grid)
    return {f"hv_at_{int(frac * 100)}pct": float(v) for frac, v in zip(fractions, values)}
