import numpy as np
from .encoding import enumerate_valid_assignments


WEIGHT_PRESETS = {
    "balanced": (1 / 3, 1 / 3, 1 / 3),
    "throughput": (1.0, 0.0, 0.0),
    "low_interference": (0.0, 1.0, 0.0),
    "low_energy": (0.0, 0.0, 1.0),
}


def raw_components(instance, assignment):
    assignment = np.asarray(assignment, dtype=int)
    throughput = float(sum(instance.demand[u] * instance.gain[u, c] for u, c in enumerate(assignment)))
    interference = 0.0
    for u, v, w in instance.interference_edges:
        if assignment[u] == assignment[v]:
            interference += w
    energy = float(sum(instance.energy[u, c] for u, c in enumerate(assignment)))
    return throughput, float(interference), energy


class ObjectiveNormalizer:
    """Normalizes valid-assignment objectives to [0, 1]."""

    def __init__(self, instance):
        vals = np.asarray([raw_components(instance, a) for a in enumerate_valid_assignments(instance.users, instance.channels)])
        self.t_min, self.t_max = float(vals[:, 0].min()), float(vals[:, 0].max())
        self.i_min, self.i_max = float(vals[:, 1].min()), float(vals[:, 1].max())
        self.e_min, self.e_max = float(vals[:, 2].min()), float(vals[:, 2].max())

    @staticmethod
    def _scale(x, lo, hi):
        return 0.0 if abs(hi - lo) < 1e-12 else float((x - lo) / (hi - lo))

    def evaluate(self, instance, assignment):
        throughput, interference, energy = raw_components(instance, assignment)
        throughput_norm = self._scale(throughput, self.t_min, self.t_max)
        return {
            "throughput": throughput,
            "interference": interference,
            "energy": energy,
            "throughput_loss": 1.0 - throughput_norm,
            "interference_norm": self._scale(interference, self.i_min, self.i_max),
            "energy_norm": self._scale(energy, self.e_min, self.e_max),
        }

    def vector(self, instance, assignment):
        d = self.evaluate(instance, assignment)
        return np.array([d["throughput_loss"], d["interference_norm"], d["energy_norm"]], dtype=float)


def weighted_score(vector, weights):
    return float(np.dot(np.asarray(weights, dtype=float), np.asarray(vector, dtype=float)))


def resolve_weights(mode, custom=None, rng=None, num_random=8):
    if mode in WEIGHT_PRESETS:
        return [WEIGHT_PRESETS[mode]]
    if mode == "custom":
        if custom is None:
            raise ValueError("custom mode requires --weights")
        w = np.asarray(custom, dtype=float)
        return [tuple((w / w.sum()).tolist())]
    base = [
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        (1 / 3, 1 / 3, 1 / 3), (0.5, 0.25, 0.25), (0.25, 0.5, 0.25), (0.25, 0.25, 0.5),
    ]
    if rng is None:
        rng = np.random.default_rng(0)
    if mode == "pareto_sweep":
        return base + [tuple(w) for w in rng.dirichlet(np.ones(3), size=max(0, num_random))]
    if mode == "random_weights":
        return [tuple(w) for w in rng.dirichlet(np.ones(3), size=num_random)]
    raise ValueError(f"Unknown objective mode: {mode}")

