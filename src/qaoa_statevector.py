import numpy as np
from .qubo import precompute_energies
from .encoding import bitstrings_to_matrix, decode_bitstring, assignment_to_label


def apply_x_mixer(psi, beta, n_qubits):
    cos, sin = np.cos(beta), -1j * np.sin(beta)
    state = psi.reshape([2] * n_qubits)
    for q in range(n_qubits):
        state = np.moveaxis(state, q, 0)
        a0 = state[0].copy()
        a1 = state[1].copy()
        state[0] = cos * a0 + sin * a1
        state[1] = sin * a0 + cos * a1
        state = np.moveaxis(state, 0, q)
    return state.reshape(-1)


def run_statevector(energies, gammas, betas):
    n_states = len(energies)
    n_qubits = int(np.log2(n_states))
    psi = np.ones(n_states, dtype=np.complex128) / np.sqrt(n_states)
    norms = []
    for gamma, beta in zip(gammas, betas):
        psi *= np.exp(-1j * gamma * energies)
        psi = apply_x_mixer(psi, beta, n_qubits)
        norms.append(float(np.vdot(psi, psi).real))
    return psi, norms


def expected_energy(params, energies, depth):
    gammas = params[:depth]
    betas = params[depth:]
    psi, _ = run_statevector(energies, gammas, betas)
    probs = np.abs(psi) ** 2
    return float(probs @ energies)


def sample_candidates(Q, instance, params, depth, candidate_size=128, seed=0, repair=True):
    energies = precompute_energies(Q)
    psi, norms = run_statevector(energies, params[:depth], params[depth:])
    probs = np.abs(psi) ** 2
    rng = np.random.default_rng(seed)
    top_k = min(candidate_size, len(probs))
    top = np.argpartition(-probs, top_k - 1)[:top_k]
    sampled = rng.choice(len(probs), size=top_k, replace=True, p=probs / probs.sum())
    indices = np.unique(np.concatenate([top, sampled]))
    bit_matrix = bitstrings_to_matrix(Q.shape[0])
    rows = []
    valid_count = 0
    seen = set()
    for idx in indices:
        bits = bit_matrix[idx]
        assignment, valid = decode_bitstring(bits, instance.users, instance.channels, instance, repair=repair)
        valid_count += int(valid)
        label = assignment_to_label(assignment)
        if label in seen:
            continue
        seen.add(label)
        rows.append({"assignment": assignment, "bits": bits, "probability": float(probs[idx]), "valid_before_repair": valid})
    valid_ratio = valid_count / max(1, len(indices))
    rows.sort(key=lambda r: r["probability"], reverse=True)
    return rows[:candidate_size], valid_ratio, norms

