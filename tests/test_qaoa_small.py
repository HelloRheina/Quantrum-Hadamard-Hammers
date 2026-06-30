import numpy as np
from src.instance import generate_instance
from src.qubo import build_qubo, precompute_energies
from src.qaoa_statevector import run_statevector


def test_qaoa_state_normalization_after_layer():
    inst = generate_instance(2, 2, seed=3)
    Q = build_qubo(inst, (1 / 3, 1 / 3, 1 / 3), penalty=5.0)
    energies = precompute_energies(Q)
    psi, norms = run_statevector(energies, [0.4], [0.2])
    assert np.isclose(np.vdot(psi, psi).real, 1.0)
    assert all(np.isclose(n, 1.0) for n in norms)

