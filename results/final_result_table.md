# Final Result Table

All HV columns are normalized-objective-space HV values.

| method | final_hv | auc_hv | archive_size | feasibility_rate |
|---|---:|---:|---:|---:|
| Random | 0.826686 | 0.695740 | 20 | 1.000 |
| Greedy | 0.997476 | 0.788742 | 17 | 1.000 |
| Evolutionary MOO | 0.860602 | 0.720378 | 11 | 1.000 |
| NSGA-II-style | 0.938773 | 0.746758 | 15 | 1.000 |
| MOEA/D-style | 0.947478 | 0.612150 | 13 | 1.000 |
| Classical-Seeded Evolutionary MOO | 1.026435 | 0.980794 | 19 | 1.000 |
| Quantum-Seeded Evolutionary MOO | 1.045177 | 1.018036 | 28 | 1.000 |
| Fixed Pareto-QAOA | 1.047353 | 0.950344 | 28 | 1.000 |
| AI-Adaptive Pareto-QAOA | 1.037196 | 0.929855 | 24 | 1.000 |
| AI-Pareto-QAOA Ensemble | 1.047353 | 0.946777 | 28 | 1.000 |

Key readout:

- Best final HV: Fixed Pareto-QAOA and AI-Pareto-QAOA Ensemble, 1.047353.
- Best AUC-HV: Quantum-Seeded Evolutionary MOO, 1.018036.
- Full machine-readable tables are in `results/method_summary.csv`, `results/main_multiseed_summary.csv`, `results/quantum_seeded_evidence.csv`, and `results/classical_optimizer_ablation.csv`.

Extended Benchmark Evidence:

- Extended results now cover 21 static wireless channel allocation cases and 79 case-seed combinations.
- The largest completed case is `scale_10x3`, with 10 users, 3 channels, and 59049 feasible assignments.

Evidence for AI-Adaptive Pareto-QAOA:

- The adaptive study covers five required cases and five seeds per case.
- Component ablations are run as real controlled variants.
- Replay-Adaptive Schedule reaches mean AUC-HV 0.590769 and mean coverage 0.520, while Full Online AI-Adaptive reaches mean AUC-HV 0.561869 and coverage 0.476. Fixed remains the strongest mean final-HV comparator in the short-budget study.
