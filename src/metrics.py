import numpy as np
from .pareto import get_pareto_indices
from .objectives import raw_components


def approximate_hypervolume(points, reference=(1.1, 1.1, 1.1), samples=25000, seed=0):
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return 0.0
    ref = np.asarray(reference, dtype=float)
    rng = np.random.default_rng(seed)
    sample = rng.random((samples, 3)) * ref
    dominated = np.zeros(samples, dtype=bool)
    for p in points:
        dominated |= np.all(p <= sample, axis=1)
    return float(dominated.mean() * np.prod(ref))


def igd_plus(approx, reference):
    approx = np.asarray(approx, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(approx) == 0 or len(reference) == 0:
        return float("inf")
    dists = []
    for r in reference:
        diff = np.maximum(approx - r, 0.0)
        dists.append(np.min(np.linalg.norm(diff, axis=1)))
    return float(np.mean(dists))


def exact_pareto_front(instance, normalizer):
    from .encoding import enumerate_valid_assignments
    records = []
    for a in enumerate_valid_assignments(instance.users, instance.channels):
        vec = normalizer.vector(instance, a)
        records.append((a, vec, raw_components(instance, a)))
    idx = get_pareto_indices([r[1] for r in records])
    return [records[i] for i in idx]


def summarize_method(name, archive, exact_vectors, seed=0, valid_ratio=1.0, runtime=0.0):
    vectors = np.asarray([r["objective_vector"] for r in archive], dtype=float) if archive else np.empty((0, 3))
    raws = [r["raw"] for r in archive]
    return {
        "method": name,
        "hypervolume": approximate_hypervolume(vectors, seed=seed),
        "igd_plus": igd_plus(vectors, exact_vectors),
        "pareto_archive_size": len(archive),
        "best_throughput": max((r[0] for r in raws), default=0.0),
        "best_interference": min((r[1] for r in raws), default=0.0),
        "best_energy": min((r[2] for r in raws), default=0.0),
        "valid_ratio_before_repair": valid_ratio,
        "runtime_seconds": runtime,
    }

