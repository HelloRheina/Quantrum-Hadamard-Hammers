# Benchmark Report

All HV, early HV, and AUC-HV values are computed in normalized objective space. Raw objectives are still used for Pareto dominance, archives, and throughput/interference/energy reporting.

## Validation

- `python -m pytest -q`: 13 passed.
- `python validate_results.py`: Validation passed.
- Extended benchmark evidence currently covers 21 static wireless channel allocation cases and 79 case-seed combinations.
- Largest case run: `scale_10x3`, users=10, channels=3, feasible space size 59049.
- Adaptive study currently covers five required cases with five seeds each.

## Main Dense Small-Cell Result

| method | normalized final HV | normalized AUC-HV | objective_evaluations |
|---|---:|---:|---:|
| Random | 0.826686 | 0.695740 | 300 |
| Greedy | 0.997476 | 0.788742 | 300 |
| Evolutionary MOO | 0.860602 | 0.720378 | 300 |
| NSGA-II-style | 0.938773 | 0.746758 | 300 |
| MOEA/D-style | 0.947478 | 0.612150 | 300 |
| Classical-Seeded Evolutionary MOO | 1.026435 | 0.980794 | 300 |
| Quantum-Seeded Evolutionary MOO | 1.045177 | 1.018036 | 300 |
| Fixed Pareto-QAOA | 1.047353 | 0.950344 | 200 |
| AI-Adaptive Pareto-QAOA | 1.037196 | 0.929855 | 200 |
| AI-Pareto-QAOA Ensemble | 1.047353 | 0.946777 | 400 |

## Extended Benchmark Evidence

- Best mean final HV across extended cases: `Fixed Pareto-QAOA` on `low_conflict_8x3`, 1.337048.
- Best mean AUC-HV across extended cases: `Quantum-Seeded Evolutionary MOO` on `low_conflict_8x3`, 1.314749.
- Quantum-Seeded EA beats NSGA-II-style in AUC-HV on every case currently in `quantum_advantage_by_case.csv`.
- Quantum-Seeded EA beats MOEA/D-style in AUC-HV on every case currently in `quantum_advantage_by_case.csv`.
- Quantum-Seeded EA beats Classical-Seeded EA in AUC-HV on every case currently in `quantum_advantage_by_case.csv`.

Generated extended evidence files:

- `results/extended_benchmark_summary.csv`
- `results/extended_benchmark_raw.csv`
- `results/scalability_summary.csv`
- `results/difficulty_sweep_summary.csv`
- `results/objective_structure_summary.csv`
- `results/budget_sweep_summary.csv`
- `results/quantum_advantage_by_case.csv`
- `results/adaptive_qaoa_advantage_by_case.csv`

## AI-Adaptive QAOA Evidence

The adaptive evidence is conservative. In the lightweight adaptive study, Fixed Pareto-QAOA remains stronger than Full AI-Adaptive Pareto-QAOA on mean AUC-HV for `main_dense_small_cell`, while Full AI-Adaptive wins selected seeds and the logs expose the online policy decisions.

Main dense small-cell adaptive-study means:

| method | mean final HV | mean AUC-HV |
|---|---:|---:|
| Fixed Pareto-QAOA | 0.885520 | 0.628287 |
| Full AI-Adaptive Pareto-QAOA | 0.818442 | 0.560347 |

Across the five adaptive-study cases, Online AI-Adaptive has mean final HV 0.800537, Replay-Adaptive Schedule has 0.831903, and Fixed cyclic schedule has 0.850636. Replay-Adaptive has mean AUC-HV 0.590769 versus 0.561869 for Online AI-Adaptive and 0.593102 for Fixed, with higher mean direction coverage than both. This means the recorded adaptive schedule is useful for coverage and competitive in AUC-HV, while the current short-budget online feedback loop is not yet consistently better than replay or fixed scheduling.

Careful claim: across the extended benchmark suite, quantum-assisted methods are not claimed to dominate every classical method on every possible instance. Instead, the evidence evaluates when and where feasible-space QAOA candidate generation and quantum-seeded evolutionary refinement provide the largest benefit.

This project uses no dynamic spectrum allocation, no switching cost, no previous time-slot warm start, no prior snapshot logic, no dynamic graph shift, and no time-varying allocation logic.
