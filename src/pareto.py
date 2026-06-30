import numpy as np


def dominates(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b + 1e-12) and np.any(a < b - 1e-12))


def get_pareto_indices(objective_values):
    vals = np.asarray(objective_values, dtype=float)
    keep = []
    for i, v in enumerate(vals):
        if not any(dominates(vals[j], v) for j in range(len(vals)) if j != i):
            keep.append(i)
    return np.asarray(keep, dtype=int)


def build_pareto_archive(records):
    dedup = {}
    for r in records:
        key = tuple(int(x) for x in r["assignment"])
        if key not in dedup or r.get("score", 0.0) < dedup[key].get("score", 0.0):
            dedup[key] = r
    rows = list(dedup.values())
    if not rows:
        return []
    idx = get_pareto_indices([r["objective_vector"] for r in rows])
    return [rows[i] for i in idx]

