import itertools
import numpy as np
from src.encoding import encode_assignment
from src.instance import generate_instance
from src.objectives import ObjectiveNormalizer, weighted_score
from src.qubo import build_qubo, qubo_energy


def test_qubo_energy_consistent_ranking_for_valid_assignments():
    inst = generate_instance(3, 2, seed=2)
    weights = (1 / 3, 1 / 3, 1 / 3)
    normalizer = ObjectiveNormalizer(inst)
    Q = build_qubo(inst, weights, penalty=8.0)
    pairs = []
    for a in itertools.product(range(inst.channels), repeat=inst.users):
        vec = normalizer.vector(inst, a)
        direct = weighted_score(vec, weights)
        energy = qubo_energy(Q, encode_assignment(a, inst.channels))
        pairs.append((direct, energy))
    direct_order = np.argsort([p[0] for p in pairs])
    qubo_order = np.argsort([p[1] for p in pairs])
    assert direct_order[0] == qubo_order[0]

