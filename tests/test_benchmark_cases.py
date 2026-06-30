from benchmark_cases import BENCHMARK_CASES, get_case
from problem import make_problem, enumerate_assignments


def test_every_benchmark_case_loads():
    for name in BENCHMARK_CASES:
        case = get_case(name)
        assert case.users > 0
        assert case.channels > 1
        assert case.budget > 0


def test_feasible_space_size_tiny_case():
    case = get_case("tiny_exact_debug")
    problem = make_problem(case.users, case.channels, case.seed)
    assert len(enumerate_assignments(problem)) == case.channels ** case.users

