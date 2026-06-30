from dataclasses import dataclass
import itertools
import numpy as np


@dataclass(frozen=True)
class WirelessProblem:
    users: int
    channels: int
    channel_gain: np.ndarray
    power: np.ndarray
    interference_matrix: np.ndarray
    noise: float
    seed: int
    case_name: str = "custom"
    interference_density: float = 0.55
    interference_strength: str = "medium"
    energy_heterogeneity: str = "medium"
    objective_conflict_level: str = "medium"


def _level_scale(level, low=0.5, medium=1.0, high=1.7, very_high=2.5):
    return {
        "low": low,
        "medium": medium,
        "high": high,
        "very_high": very_high,
    }.get(level, medium)


def make_problem(users=5, channels=3, seed=42, noise=0.1, case_name="custom", interference_density=0.55, interference_strength="medium", energy_heterogeneity="medium", objective_conflict_level="medium"):
    rng = np.random.default_rng(seed)
    conflict = _level_scale(objective_conflict_level, low=0.25, medium=0.55, high=0.85, very_high=1.0)
    energy_scale = _level_scale(energy_heterogeneity, low=0.35, medium=0.8, high=1.3, very_high=2.0)
    gain_base = rng.uniform(0.55, 1.55, size=(users, channels))
    power_base = rng.uniform(0.25, 1.0 + energy_scale, size=(users, channels))
    channel_gain = gain_base + conflict * 0.8 * (power_base - power_base.min(axis=1, keepdims=True)) / (np.ptp(power_base, axis=1, keepdims=True) + 1e-12)
    power = power_base
    mask = rng.random((users, users)) < interference_density
    raw = rng.uniform(0.05, 0.7 * _level_scale(interference_strength), size=(users, users)) * mask
    interference = (raw + raw.T) / 2.0
    np.fill_diagonal(interference, 0.0)
    return WirelessProblem(users, channels, channel_gain, power, interference, noise, seed, case_name, interference_density, interference_strength, energy_heterogeneity, objective_conflict_level)


def enumerate_assignments(problem):
    return np.array(list(itertools.product(range(problem.channels), repeat=problem.users)), dtype=int)


def encode_assignment(assignment, channels):
    bits = np.zeros(len(assignment) * channels, dtype=np.int8)
    for u, c in enumerate(assignment):
        bits[u * channels + int(c)] = 1
    return bits


def decode_bitstring(bits, problem, repair=True):
    bits = np.asarray(bits, dtype=np.int8)
    assignment = np.zeros(problem.users, dtype=int)
    feasible = True
    for u in range(problem.users):
        block = bits[u * problem.channels:(u + 1) * problem.channels]
        selected = np.flatnonzero(block == 1)
        if len(selected) == 1:
            assignment[u] = int(selected[0])
        else:
            feasible = False
            if not repair:
                assignment[u] = -1
            elif block.sum() > 0:
                assignment[u] = int(np.argmax(block))
            else:
                score = problem.channel_gain[u] / (problem.power[u] + 1e-12)
                assignment[u] = int(np.argmax(score))
    return assignment, feasible


def objective_vector(problem, assignment):
    assignment = np.asarray(assignment, dtype=int)
    throughput = 0.0
    for u in range(problem.users):
        cu = assignment[u]
        signal = problem.channel_gain[u, cu] * problem.power[u, cu]
        interference_u = 0.0
        for v in range(problem.users):
            if u != v and assignment[v] == cu:
                interference_u += problem.interference_matrix[u, v] * problem.power[v, assignment[v]]
        throughput += np.log2(1.0 + signal / (problem.noise + interference_u))

    interference = 0.0
    for u in range(problem.users):
        for v in range(u + 1, problem.users):
            if assignment[u] == assignment[v]:
                interference += problem.interference_matrix[u, v]
    energy = float(sum(problem.power[u, assignment[u]] for u in range(problem.users)))
    return np.array([-float(throughput), float(interference), energy], dtype=float)


def objective_table(problem):
    assignments = enumerate_assignments(problem)
    values = np.array([objective_vector(problem, a) for a in assignments], dtype=float)
    return assignments, values


def one_hot_violation(bits, users, channels):
    bits = np.asarray(bits, dtype=np.int8)
    violation = 0.0
    for u in range(users):
        s = bits[u * channels:(u + 1) * channels].sum()
        violation += float((s - 1) ** 2)
    return violation


def all_bitstrings(n_qubits):
    vals = np.arange(2 ** n_qubits, dtype=np.uint32)
    shifts = np.arange(n_qubits, dtype=np.uint32)
    return ((vals[:, None] >> shifts[None, :]) & 1).astype(np.int8)


def precompute_bitstring_costs(problem, weights, penalty=20.0, repair=True):
    bits = all_bitstrings(problem.users * problem.channels)
    costs = np.zeros(len(bits), dtype=float)
    decoded = []
    feasible = np.zeros(len(bits), dtype=bool)
    for i, b in enumerate(bits):
        assignment, ok = decode_bitstring(b, problem, repair=repair)
        feasible[i] = ok
        decoded.append(assignment)
        costs[i] = float(np.dot(weights, objective_vector(problem, assignment))) + penalty * one_hot_violation(b, problem.users, problem.channels)
    return bits, np.array(decoded, dtype=int), feasible, costs


def precompute_bitstring_terms(problem, repair=True):
    bits = all_bitstrings(problem.users * problem.channels)
    decoded = []
    feasible = np.zeros(len(bits), dtype=bool)
    objectives = np.zeros((len(bits), 3), dtype=float)
    violations = np.zeros(len(bits), dtype=float)
    for i, b in enumerate(bits):
        assignment, ok = decode_bitstring(b, problem, repair=repair)
        feasible[i] = ok
        decoded.append(assignment)
        objectives[i] = objective_vector(problem, assignment)
        violations[i] = one_hot_violation(b, problem.users, problem.channels)
    return {
        "bits": bits,
        "decoded": np.array(decoded, dtype=int),
        "feasible": feasible,
        "objectives": objectives,
        "violations": violations,
    }
