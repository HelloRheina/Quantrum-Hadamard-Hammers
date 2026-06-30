import time
import numpy as np
from problem import objective_vector
from pareto import update_archive, non_dominated_filter


def random_search(problem, evaluations=300, seed=0, return_trace=False):
    start = time.time()
    rng = np.random.default_rng(seed)
    sols, objs = [], []
    for _ in range(evaluations):
        a = rng.integers(0, problem.channels, size=problem.users)
        sols.append(a)
        objs.append(objective_vector(problem, a))
    sol, obj = update_archive([], [], sols, objs)
    if return_trace:
        return sol, obj, evaluations, 1.0, time.time() - start, np.asarray(sols, dtype=int), np.asarray(objs, dtype=float)
    return sol, obj, evaluations, 1.0, time.time() - start


def greedy(problem, restarts=40, seed=1, return_trace=False, eval_budget=None):
    start = time.time()
    rng = np.random.default_rng(seed)
    sols, objs = [], []
    trace_s, trace_o = [], []
    evals = 0
    weights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    for _ in range(restarts):
        if eval_budget is not None and evals >= eval_budget:
            break
        a = rng.integers(0, problem.channels, size=problem.users)
        improved = True
        while improved:
            if eval_budget is not None and evals >= eval_budget:
                break
            improved = False
            y = objective_vector(problem, a)
            evals += 1
            trace_s.append(a.copy())
            trace_o.append(y)
            sols.append(a.copy())
            objs.append(y)
            base = float(weights @ y)
            for u in range(problem.users):
                for c in range(problem.channels):
                    if eval_budget is not None and evals >= eval_budget:
                        break
                    if c == a[u]:
                        continue
                    b = a.copy()
                    b[u] = c
                    yb = objective_vector(problem, b)
                    score = float(weights @ yb)
                    evals += 1
                    trace_s.append(b.copy())
                    trace_o.append(yb)
                    if score < base:
                        a = b
                        base = score
                        improved = True
                if eval_budget is not None and evals >= eval_budget:
                    break
    sol, obj = update_archive([], [], trace_s, trace_o)
    if return_trace:
        return sol, obj, evals, 1.0, time.time() - start, np.asarray(trace_s, dtype=int), np.asarray(trace_o, dtype=float)
    return sol, obj, evals, 1.0, time.time() - start


def non_dominated_sort(points):
    points = np.asarray(points, dtype=float)
    n = len(points)
    dominates = [[] for _ in range(n)]
    dominated_count = np.zeros(n, dtype=int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            p_dom_q = np.all(points[p] <= points[q]) and np.any(points[p] < points[q])
            q_dom_p = np.all(points[q] <= points[p]) and np.any(points[q] < points[p])
            if p_dom_q:
                dominates[p].append(q)
            elif q_dom_p:
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
    return [np.asarray(front, dtype=int) for front in fronts if len(front)]


def crowding_distance(points):
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n == 0:
        return np.array([])
    dist = np.zeros(n)
    for m in range(points.shape[1]):
        order = np.argsort(points[:, m])
        dist[order[0]] = dist[order[-1]] = np.inf
        span = points[order[-1], m] - points[order[0], m]
        if span <= 1e-12:
            continue
        for k in range(1, n - 1):
            dist[order[k]] += (points[order[k + 1], m] - points[order[k - 1], m]) / span
    return dist


def evolutionary_mo(problem, pop_size=50, generations=20, seed=2, return_trace=False, initial_population=None, eval_budget=None):
    start = time.time()
    rng = np.random.default_rng(seed)
    if initial_population is not None and len(initial_population):
        init = np.asarray(initial_population, dtype=int)
        fill = rng.integers(0, problem.channels, size=(max(0, pop_size - len(init)), problem.users))
        pop = np.vstack([init[:pop_size], fill])[:pop_size]
    else:
        pop = rng.integers(0, problem.channels, size=(pop_size, problem.users))
    archive_s, archive_o = [], []
    trace_s, trace_o = [], []
    evals = 0
    for _ in range(generations):
        objs = np.array([objective_vector(problem, a) for a in pop])
        evals += len(pop)
        trace_s.extend([a.copy() for a in pop])
        trace_o.extend([o.copy() for o in objs])
        archive_s, archive_o = update_archive(archive_s, archive_o, pop, objs)
        if eval_budget is not None and evals >= eval_budget:
            break
        children = []
        while len(children) < pop_size:
            p1, p2 = pop[rng.integers(0, len(pop), size=2)]
            mask = rng.random(problem.users) < 0.5
            child = np.where(mask, p1, p2).copy()
            if rng.random() < 0.8:
                child[rng.integers(0, problem.users)] = rng.integers(0, problem.channels)
            children.append(child)
        combined = np.vstack([pop, np.asarray(children, dtype=int)])
        combined_obj = np.array([objective_vector(problem, a) for a in combined])
        evals += len(combined)
        trace_s.extend([a.copy() for a in combined])
        trace_o.extend([o.copy() for o in combined_obj])
        nd = non_dominated_filter(combined_obj)
        if len(nd) >= pop_size:
            cd = crowding_distance(combined_obj[nd])
            pop = combined[nd[np.argsort(-cd)[:pop_size]]]
        else:
            chosen = list(nd)
            remaining = [i for i in range(len(combined)) if i not in set(chosen)]
            scores = combined_obj[remaining].sum(axis=1)
            chosen += [remaining[i] for i in np.argsort(scores)[:pop_size - len(chosen)]]
            pop = combined[chosen]
        if eval_budget is not None and evals >= eval_budget:
            break
    archive_s, archive_o = update_archive(archive_s, archive_o, trace_s, trace_o)
    if return_trace:
        return archive_s, archive_o, evals, 1.0, time.time() - start, np.asarray(trace_s, dtype=int), np.asarray(trace_o, dtype=float)
    return archive_s, archive_o, evals, 1.0, time.time() - start


def nsga2_style(problem, pop_size=50, seed=3, return_trace=False, eval_budget=300):
    start = time.time()
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, problem.channels, size=(pop_size, problem.users))
    pop_obj = []
    archive_s, archive_o = [], []
    trace_s, trace_o = [], []
    evals = 0

    for a in pop:
        if evals >= eval_budget:
            break
        y = objective_vector(problem, a)
        pop_obj.append(y)
        trace_s.append(a.copy())
        trace_o.append(y.copy())
        evals += 1
    pop = pop[:len(pop_obj)]
    pop_obj = np.asarray(pop_obj, dtype=float)
    archive_s, archive_o = update_archive(archive_s, archive_o, pop, pop_obj)

    while evals < eval_budget and len(pop):
        children = []
        while len(children) < pop_size and evals + len(children) < eval_budget:
            p1, p2 = pop[rng.integers(0, len(pop), size=2)]
            mask = rng.random(problem.users) < 0.5
            child = np.where(mask, p1, p2).copy()
            child[rng.integers(0, problem.users)] = rng.integers(0, problem.channels)
            children.append(child)
        if not children:
            break
        child_obj = np.array([objective_vector(problem, c) for c in children])
        evals += len(children)
        trace_s.extend([c.copy() for c in children])
        trace_o.extend([o.copy() for o in child_obj])
        archive_s, archive_o = update_archive(archive_s, archive_o, children, child_obj)

        combined = np.vstack([pop, np.asarray(children, dtype=int)])
        combined_obj = np.vstack([pop_obj, child_obj])
        chosen = []
        for front in non_dominated_sort(combined_obj):
            if len(chosen) + len(front) <= pop_size:
                chosen.extend(front.tolist())
            else:
                cd = crowding_distance(combined_obj[front])
                slots = pop_size - len(chosen)
                chosen.extend(front[np.argsort(-cd)[:slots]].tolist())
                break
        pop = combined[chosen]
        pop_obj = combined_obj[chosen]

    archive_s, archive_o = update_archive(archive_s, archive_o, trace_s, trace_o)
    if return_trace:
        return archive_s, archive_o, evals, 1.0, time.time() - start, np.asarray(trace_s, dtype=int), np.asarray(trace_o, dtype=float)
    return archive_s, archive_o, evals, 1.0, time.time() - start


def _simplex_weights(count):
    base = [
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),
        (1 / 3, 1 / 3, 1 / 3),
    ]
    weights = [np.array(w, dtype=float) for w in base]
    grid = max(2, int(np.ceil(np.sqrt(count))))
    for i in range(grid + 1):
        for j in range(grid + 1 - i):
            k = grid - i - j
            weights.append(np.array([i, j, k], dtype=float) / grid)
    unique = []
    seen = set()
    for w in weights:
        key = tuple(np.round(w, 6))
        if key not in seen and np.isclose(w.sum(), 1.0):
            unique.append(w)
            seen.add(key)
    while len(unique) < count:
        unique.extend(unique)
    return np.asarray(unique[:count], dtype=float)


def moead_style(problem, n_weights=45, seed=4, return_trace=False, eval_budget=300):
    start = time.time()
    rng = np.random.default_rng(seed)
    weights = _simplex_weights(n_weights)
    pop = rng.integers(0, problem.channels, size=(len(weights), problem.users))
    pop_obj = []
    archive_s, archive_o = [], []
    trace_s, trace_o = [], []
    evals = 0

    def normalized_scores(objs):
        vals = np.asarray(objs, dtype=float)
        if len(trace_o):
            basis = np.vstack([np.asarray(trace_o, dtype=float), vals.reshape(-1, 3)])
        else:
            basis = vals.reshape(-1, 3)
        lo = basis.min(axis=0)
        hi = basis.max(axis=0)
        return (vals - lo) / (hi - lo + 1e-12)

    for a in pop:
        if evals >= eval_budget:
            break
        y = objective_vector(problem, a)
        pop_obj.append(y)
        trace_s.append(a.copy())
        trace_o.append(y.copy())
        evals += 1
    pop = pop[:len(pop_obj)]
    weights = weights[:len(pop_obj)]
    pop_obj = np.asarray(pop_obj, dtype=float)
    archive_s, archive_o = update_archive(archive_s, archive_o, pop, pop_obj)
    if len(pop) == 0:
        if return_trace:
            return archive_s, archive_o, evals, 1.0, time.time() - start, np.asarray(trace_s, dtype=int), np.asarray(trace_o, dtype=float)
        return archive_s, archive_o, evals, 1.0, time.time() - start

    weight_dist = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    neighbors = np.argsort(weight_dist, axis=1)[:, : min(8, len(weights))]

    while evals < eval_budget:
        i = int(rng.integers(0, len(pop)))
        parent = pop[int(rng.choice(neighbors[i]))].copy()
        child = parent.copy()
        child[rng.integers(0, problem.users)] = rng.integers(0, problem.channels)
        if rng.random() < 0.35:
            child[rng.integers(0, problem.users)] = rng.integers(0, problem.channels)
        y = objective_vector(problem, child)
        evals += 1
        trace_s.append(child.copy())
        trace_o.append(y.copy())
        archive_s, archive_o = update_archive(archive_s, archive_o, [child], [y])

        candidate_pool = np.vstack([pop_obj, y.reshape(1, 3)])
        norm_pool = normalized_scores(candidate_pool)
        norm_pop = norm_pool[:-1]
        norm_child = norm_pool[-1]
        for j in neighbors[i]:
            old_score = float(weights[j] @ norm_pop[j])
            new_score = float(weights[j] @ norm_child)
            if new_score <= old_score:
                pop[j] = child
                pop_obj[j] = y

    archive_s, archive_o = update_archive(archive_s, archive_o, trace_s, trace_o)
    if return_trace:
        return archive_s, archive_o, evals, 1.0, time.time() - start, np.asarray(trace_s, dtype=int), np.asarray(trace_o, dtype=float)
    return archive_s, archive_o, evals, 1.0, time.time() - start
