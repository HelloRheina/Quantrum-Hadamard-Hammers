from src.pareto import dominates, get_pareto_indices


def test_pareto_dominance():
    assert dominates([0.1, 0.2, 0.3], [0.1, 0.3, 0.4])
    assert not dominates([0.2, 0.2, 0.3], [0.1, 0.3, 0.4])


def test_get_pareto_indices():
    idx = get_pareto_indices([[0, 1, 1], [1, 0, 1], [0.5, 0.5, 1], [2, 2, 2]])
    assert set(idx.tolist()) == {0, 1, 2}

