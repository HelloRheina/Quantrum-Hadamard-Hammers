import numpy as np


def dominates(y1, y2):
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    return bool(np.all(y1 <= y2 + 1e-12) and np.any(y1 < y2 - 1e-12))


def non_dominated_filter(points):
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.array([], dtype=int)
    keep = []
    for i, p in enumerate(points):
        if not any(dominates(points[j], p) for j in range(len(points)) if j != i):
            keep.append(i)
    return np.array(keep, dtype=int)


def update_archive(archive_solutions, archive_objectives, new_solutions, new_objectives):
    if archive_solutions is None or len(archive_solutions) == 0:
        sol = np.asarray(new_solutions, dtype=int)
        obj = np.asarray(new_objectives, dtype=float)
    elif len(new_solutions) == 0:
        sol = np.asarray(archive_solutions, dtype=int)
        obj = np.asarray(archive_objectives, dtype=float)
    else:
        sol = np.vstack([archive_solutions, new_solutions]).astype(int)
        obj = np.vstack([archive_objectives, new_objectives]).astype(float)

    unique = {}
    for s, o in zip(sol, obj):
        key = tuple(int(x) for x in s)
        if key not in unique:
            unique[key] = (s, o)
    sol = np.array([v[0] for v in unique.values()], dtype=int)
    obj = np.array([v[1] for v in unique.values()], dtype=float)
    idx = non_dominated_filter(obj)
    return sol[idx], obj[idx]


def dominated_substitution(sorted_candidates, objective_fn, n_most_prob=20, substitution_factor=2):
    solutions, objectives = [], []
    checked = 0
    archive_s, archive_o = np.empty((0, 0), dtype=int), np.empty((0, 3), dtype=float)
    for assignment, _prob in sorted_candidates:
        checked += 1
        y = objective_fn(assignment)
        solutions.append(assignment)
        objectives.append(y)
        archive_s, archive_o = update_archive(archive_s, archive_o, [assignment], [y])
        if len(archive_s) >= n_most_prob or checked >= substitution_factor * n_most_prob:
            break
    return archive_s, archive_o, checked
