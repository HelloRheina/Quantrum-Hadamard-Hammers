import numpy as np
from problem import make_problem, objective_table
from qaoa import qaoa_sample, cvar_cost


def test_feasible_mixer_samples_valid_assignments_only():
    problem = make_problem(3, 2, seed=0)
    assignments, objectives = objective_table(problem)
    cache = {"assignments": assignments, "objectives": objectives}
    result = qaoa_sample(
        problem,
        weights=np.array([1 / 3, 1 / 3, 1 / 3]),
        depth=1,
        shots=32,
        n_most_prob=6,
        maxiter=3,
        seed=0,
        cache=cache,
        obj_min=objectives.min(axis=0),
        obj_max=objectives.max(axis=0),
        qaoa_encoding="feasible_mixer",
        qaoa_objective="cvar",
    )
    assert result["feasibility_rate"] == 1.0
    assert result["solutions"].shape[1] == problem.users
    assert np.all((result["solutions"] >= 0) & (result["solutions"] < problem.channels))


def test_cvar_cost_sanity():
    costs = np.array([0.0, 1.0, 2.0, 3.0])
    theta = np.array([0.1, 0.2])
    value = cvar_cost(theta, costs, depth=1, alpha=0.5)
    assert 0.0 <= value <= 3.0

