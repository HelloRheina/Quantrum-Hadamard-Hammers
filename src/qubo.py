import numpy as np
from .encoding import var_index, bitstrings_to_matrix
from .objectives import ObjectiveNormalizer


def build_qubo(instance, weights, penalty=8.0):
    """Build Q for x^T Q x. Off-diagonal Q entries are counted twice by matrix multiplication."""
    u_count, c_count = instance.users, instance.channels
    n = u_count * c_count
    Q = np.zeros((n, n), dtype=float)
    normalizer = ObjectiveNormalizer(instance)
    t_range = max(normalizer.t_max - normalizer.t_min, 1e-12)
    i_range = max(normalizer.i_max - normalizer.i_min, 1e-12)
    e_range = max(normalizer.e_max - normalizer.e_min, 1e-12)
    wt, wi, we = weights

    for u in range(u_count):
        for c in range(c_count):
            idx = var_index(u, c, c_count)
            Q[idx, idx] += wt * (-(instance.demand[u] * instance.gain[u, c]) / t_range)
            Q[idx, idx] += we * (instance.energy[u, c] / e_range)

    for u, v, w in instance.interference_edges:
        for c in range(c_count):
            i, j = var_index(u, c, c_count), var_index(v, c, c_count)
            Q[i, j] += 0.5 * wi * (w / i_range)
            Q[j, i] += 0.5 * wi * (w / i_range)

    for u in range(u_count):
        block = [var_index(u, c, c_count) for c in range(c_count)]
        for i in block:
            Q[i, i] += -penalty
        for a, i in enumerate(block):
            for j in block[a + 1:]:
                Q[i, j] += penalty
                Q[j, i] += penalty
    return Q


def qubo_energy(Q, bits):
    bits = np.asarray(bits, dtype=float)
    return float(bits @ Q @ bits)


def precompute_energies(Q):
    bits = bitstrings_to_matrix(Q.shape[0]).astype(float)
    return np.einsum("bi,ij,bj->b", bits, Q, bits)

