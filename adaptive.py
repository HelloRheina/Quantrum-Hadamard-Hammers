import numpy as np


BASE_WEIGHTS = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
    [1 / 3, 1 / 3, 1 / 3],
    [0.6, 0.2, 0.2],
    [0.2, 0.6, 0.2],
    [0.2, 0.2, 0.6],
], dtype=float)


def choose_weight(iteration, archive_objectives=None):
    if archive_objectives is None or len(archive_objectives) < 3:
        return BASE_WEIGHTS[iteration % len(BASE_WEIGHTS)]
    spread = np.ptp(np.asarray(archive_objectives, dtype=float), axis=0)
    weak = int(np.argmin(spread))
    w = BASE_WEIGHTS[iteration % len(BASE_WEIGHTS)].copy()
    w[weak] += 0.4
    return w / w.sum()


def normalize_objectives(objectives, obj_min=None, obj_max=None):
    objectives = np.asarray(objectives, dtype=float)
    if len(objectives) == 0:
        return objectives
    if obj_min is None:
        obj_min = objectives.min(axis=0)
    if obj_max is None:
        obj_max = objectives.max(axis=0)
    return (objectives - obj_min) / (np.asarray(obj_max) - np.asarray(obj_min) + 1e-12)


def compute_archive_coverage(archive_objectives, weight_pool, obj_min=None, obj_max=None):
    """Direction coverage: fraction of scalarization directions selecting unique archive points."""
    archive_objectives = np.asarray(archive_objectives, dtype=float)
    if len(archive_objectives) == 0:
        return 0.0
    norm = normalize_objectives(archive_objectives, obj_min, obj_max)
    selected = set()
    for w in np.asarray(weight_pool, dtype=float):
        selected.add(int(np.argmin(norm @ w)))
    return len(selected) / max(1, len(weight_pool))


class AdaptiveWeightController:
    """Warm-up plus UCB and coverage-guided scalarization controller."""

    def __init__(self, weight_pool=None, c_ucb=0.35, coverage_weight=0.3, duplicate_penalty_weight=0.15):
        self.weight_pool = np.asarray(weight_pool if weight_pool is not None else BASE_WEIGHTS, dtype=float)
        self.c_ucb_base = c_ucb
        self.c_ucb = c_ucb
        self.coverage_weight = coverage_weight
        self.duplicate_penalty_weight = duplicate_penalty_weight
        n = len(self.weight_pool)
        self.counts = np.zeros(n, dtype=int)
        self.reward_sums = np.zeros(n, dtype=float)
        self.recent_duplicates = np.zeros(n, dtype=float)
        self.best_thetas = {}
        self.last_improved = True

    def select(self, iteration, archive_objectives, obj_min=None, obj_max=None):
        unused = np.flatnonzero(self.counts == 0)
        if len(unused) > 0:
            idx = int(unused[0])
            return idx, self.weight_pool[idx].copy(), "warmup", float("nan")

        current_coverage = compute_archive_coverage(archive_objectives, self.weight_pool, obj_min, obj_max)
        scores = []
        total = max(1, int(self.counts.sum()))
        for i, w in enumerate(self.weight_pool):
            mean_reward = self.reward_sums[i] / max(1, self.counts[i])
            ucb = self.c_ucb * np.sqrt(np.log(total + 1.0) / (self.counts[i] + 1.0))
            coverage_novelty = max(0.0, 1.0 - current_coverage)
            duplicate_penalty = self.duplicate_penalty_weight * self.recent_duplicates[i]
            scores.append(mean_reward + ucb + self.coverage_weight * coverage_novelty - duplicate_penalty)
        idx = int(np.argmax(scores))
        return idx, self.weight_pool[idx].copy(), "adaptive", float(scores[idx])

    def warm_start(self, selected_index, rng, depth, improved=True):
        if selected_index in self.best_thetas:
            theta = self.best_thetas[selected_index].copy()
        elif self.best_thetas:
            weights = self.weight_pool
            nearest = min(self.best_thetas, key=lambda k: float(np.linalg.norm(weights[k] - weights[selected_index])))
            theta = self.best_thetas[nearest].copy()
        else:
            theta = np.concatenate([rng.uniform(0, np.pi, depth), rng.uniform(0, np.pi / 2, depth)])
        scale = 0.03 if improved else 0.12
        return theta + rng.normal(0.0, scale, size=2 * depth)

    def update(self, selected_index, reward, duplicate_rate, theta):
        self.counts[selected_index] += 1
        self.reward_sums[selected_index] += reward
        self.recent_duplicates[selected_index] = 0.7 * self.recent_duplicates[selected_index] + 0.3 * duplicate_rate
        self.best_thetas[selected_index] = np.asarray(theta, dtype=float).copy()

    def boost_exploration(self):
        self.c_ucb = min(2.0 * self.c_ucb_base, self.c_ucb * 1.5)

    def reset_exploration(self):
        self.c_ucb = self.c_ucb_base
