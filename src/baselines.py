import time
import numpy as np
from .encoding import enumerate_valid_assignments
from .objectives import weighted_score, raw_components
from .pareto import build_pareto_archive


def make_record(instance, normalizer, assignment, weights, method, probability=0.0):
    vec = normalizer.vector(instance, assignment)
    return {
        "method": method,
        "assignment": np.asarray(assignment, dtype=int),
        "objective_vector": vec,
        "raw": raw_components(instance, assignment),
        "score": weighted_score(vec, weights),
        "probability": probability,
    }


def random_valid(instance, normalizer, weights, candidate_size=128, seed=0):
    start = time.time()
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(candidate_size):
        a = rng.integers(0, instance.channels, size=instance.users)
        records.append(make_record(instance, normalizer, a, weights, "Random"))
    return build_pareto_archive(records), time.time() - start


def greedy_local_search(instance, normalizer, weights, candidate_size=128, seed=0):
    start = time.time()
    rng = np.random.default_rng(seed)
    records = []
    restarts = max(4, candidate_size // max(1, instance.users * instance.channels))
    for _ in range(restarts):
        a = rng.integers(0, instance.channels, size=instance.users)
        improved = True
        while improved and len(records) < candidate_size:
            improved = False
            base = weighted_score(normalizer.vector(instance, a), weights)
            records.append(make_record(instance, normalizer, a.copy(), weights, "Greedy"))
            for u in range(instance.users):
                for c in range(instance.channels):
                    if c == a[u]:
                        continue
                    b = a.copy()
                    b[u] = c
                    score = weighted_score(normalizer.vector(instance, b), weights)
                    if score + 1e-12 < base:
                        a, base, improved = b, score, True
    return build_pareto_archive(records), time.time() - start


def exhaustive_records(instance, normalizer, weights):
    return [make_record(instance, normalizer, a, weights, "Exact") for a in enumerate_valid_assignments(instance.users, instance.channels)]

