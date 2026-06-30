from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkCase:
    case_name: str
    users: int
    channels: int
    budget: int
    seed: int = 0
    seeds: tuple = ()
    interference_density: float = 0.55
    interference_strength: str = "medium"
    energy_heterogeneity: str = "medium"
    objective_conflict_level: str = "medium"
    reference_mode: str = "auto"
    reference_pool_size: int = 5000
    recommended_methods: tuple = ("Random", "Greedy", "Evolutionary MOO", "Fixed Pareto-QAOA", "AI-Adaptive Pareto-QAOA", "Quantum-Seeded Evolutionary MOO")
    expected_purpose: str = ""
    max_runtime_hint: str = "fast"


BENCHMARK_CASES = {
    "tiny_exact_debug": BenchmarkCase("tiny_exact_debug", 3, 2, 50, seed=0, reference_mode="exact", expected_purpose="correctness and exact Pareto/HV validation"),
    "small_demo": BenchmarkCase("small_demo", 5, 3, 200, seed=0, reference_mode="exact", expected_purpose="fast visual demo"),
    "main_dense_small_cell": BenchmarkCase("main_dense_small_cell", 8, 3, 300, seed=0, interference_density=0.75, interference_strength="high", energy_heterogeneity="medium", objective_conflict_level="high", reference_mode="auto", expected_purpose="main hackathon benchmark"),
    "scale_6x3": BenchmarkCase("scale_6x3", 6, 3, 200, seed=0, interference_density=0.65, interference_strength="medium", energy_heterogeneity="medium", objective_conflict_level="medium", expected_purpose="scalability smoke test"),
    "scale_9x3": BenchmarkCase("scale_9x3", 9, 3, 400, seed=0, interference_density=0.75, interference_strength="high", energy_heterogeneity="medium", objective_conflict_level="high", expected_purpose="larger 3-channel scalability test"),
    "scale_10x3": BenchmarkCase("scale_10x3", 10, 3, 500, seed=0, interference_density=0.8, interference_strength="high", energy_heterogeneity="high", objective_conflict_level="high", expected_purpose="largest default 3-channel scalability test", max_runtime_hint="slow"),
    "channel_rich_7x4": BenchmarkCase("channel_rich_7x4", 7, 4, 500, seed=1, interference_density=0.65, interference_strength="medium", energy_heterogeneity="high", objective_conflict_level="high", expected_purpose="channel-rich scalability test"),
    "channel_rich_8x4": BenchmarkCase("channel_rich_8x4", 8, 4, 600, seed=1, interference_density=0.7, interference_strength="high", energy_heterogeneity="high", objective_conflict_level="high", expected_purpose="optional overnight channel-rich test", max_runtime_hint="slow"),
    "low_conflict_8x3": BenchmarkCase("low_conflict_8x3", 8, 3, 300, seed=2, interference_density=0.25, interference_strength="low", energy_heterogeneity="low", objective_conflict_level="low", expected_purpose="conflict sweep low"),
    "medium_conflict_8x3": BenchmarkCase("medium_conflict_8x3", 8, 3, 300, seed=2, interference_density=0.55, interference_strength="medium", energy_heterogeneity="medium", objective_conflict_level="medium", expected_purpose="conflict sweep medium"),
    "high_conflict_8x3": BenchmarkCase("high_conflict_8x3", 8, 3, 300, seed=2, interference_density=0.8, interference_strength="high", energy_heterogeneity="medium", objective_conflict_level="high", expected_purpose="conflict sweep high"),
    "extreme_conflict_8x3": BenchmarkCase("extreme_conflict_8x3", 8, 3, 500, seed=2, interference_density=0.95, interference_strength="very_high", energy_heterogeneity="high", objective_conflict_level="very_high", expected_purpose="conflict sweep extreme", max_runtime_hint="slow"),
    "balanced_objectives_8x3": BenchmarkCase("balanced_objectives_8x3", 8, 3, 300, seed=6, interference_density=0.55, interference_strength="medium", energy_heterogeneity="medium", objective_conflict_level="medium", expected_purpose="balanced objective structure"),
    "throughput_dominated_8x3": BenchmarkCase("throughput_dominated_8x3", 8, 3, 300, seed=7, interference_density=0.35, interference_strength="low", energy_heterogeneity="low", objective_conflict_level="high", expected_purpose="throughput-dominated objective structure"),
    "interference_dominated_8x3": BenchmarkCase("interference_dominated_8x3", 8, 3, 300, seed=5, interference_density=0.95, interference_strength="very_high", energy_heterogeneity="medium", objective_conflict_level="high", expected_purpose="interference-dominated objective structure"),
    "energy_dominated_8x3": BenchmarkCase("energy_dominated_8x3", 8, 3, 300, seed=4, interference_density=0.55, interference_strength="medium", energy_heterogeneity="very_high", objective_conflict_level="medium", expected_purpose="energy-dominated objective structure"),
    "main_budget_100": BenchmarkCase("main_budget_100", 8, 3, 100, seed=0, interference_density=0.75, interference_strength="high", energy_heterogeneity="medium", objective_conflict_level="high", expected_purpose="main-family budget sweep 100"),
    "main_budget_200": BenchmarkCase("main_budget_200", 8, 3, 200, seed=0, interference_density=0.75, interference_strength="high", energy_heterogeneity="medium", objective_conflict_level="high", expected_purpose="main-family budget sweep 200"),
    "main_budget_300": BenchmarkCase("main_budget_300", 8, 3, 300, seed=0, interference_density=0.75, interference_strength="high", energy_heterogeneity="medium", objective_conflict_level="high", expected_purpose="main-family budget sweep 300"),
    "main_budget_500": BenchmarkCase("main_budget_500", 8, 3, 500, seed=0, interference_density=0.75, interference_strength="high", energy_heterogeneity="medium", objective_conflict_level="high", expected_purpose="main-family budget sweep 500"),
    "main_budget_800": BenchmarkCase("main_budget_800", 8, 3, 800, seed=0, interference_density=0.75, interference_strength="high", energy_heterogeneity="medium", objective_conflict_level="high", expected_purpose="main-family budget sweep 800", max_runtime_hint="slow"),
    "hard_dense_small_cell": BenchmarkCase("hard_dense_small_cell", 10, 3, 500, seed=0, interference_density=0.8, interference_strength="high", energy_heterogeneity="high", objective_conflict_level="high", reference_mode="auto", expected_purpose="stronger scalability benchmark", max_runtime_hint="slow"),
    "channel_rich_case": BenchmarkCase("channel_rich_case", 7, 4, 500, seed=1, interference_density=0.65, interference_strength="medium", energy_heterogeneity="high", objective_conflict_level="high", reference_mode="auto", expected_purpose="test C > 3"),
    "low_conflict_case": BenchmarkCase("low_conflict_case", 8, 3, 300, seed=2, interference_density=0.3, interference_strength="low", energy_heterogeneity="low", objective_conflict_level="low", expected_purpose="easy case and honesty check"),
    "high_conflict_tradeoff_case": BenchmarkCase("high_conflict_tradeoff_case", 8, 3, 400, seed=3, interference_density=0.85, interference_strength="high", energy_heterogeneity="high", objective_conflict_level="very_high", expected_purpose="stress-test Pareto behavior"),
    "energy_dominated_case": BenchmarkCase("energy_dominated_case", 8, 3, 300, seed=4, energy_heterogeneity="very_high", objective_conflict_level="medium", expected_purpose="test objective normalization"),
    "interference_dominated_case": BenchmarkCase("interference_dominated_case", 8, 3, 300, seed=5, interference_density=0.95, interference_strength="very_high", energy_heterogeneity="medium", expected_purpose="test dense interference"),
    "multi_seed_main": BenchmarkCase("multi_seed_main", 8, 3, 300, seed=0, seeds=tuple(range(10)), interference_density=0.75, interference_strength="high", energy_heterogeneity="medium", objective_conflict_level="high", expected_purpose="robustness"),
}


PRESETS = {
    "quick_demo": "small_demo",
    "main_benchmark": "main_dense_small_cell",
    "hard_benchmark": "hard_dense_small_cell",
}


def get_case(name):
    if name in PRESETS:
        name = PRESETS[name]
    if name not in BENCHMARK_CASES:
        raise ValueError(f"Unknown benchmark case: {name}")
    return BENCHMARK_CASES[name]
