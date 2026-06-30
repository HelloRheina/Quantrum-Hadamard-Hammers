# Final Technical Route

Static wireless instance -> feasible assignment space -> objective evaluation -> objective normalization -> feasible CVaR Pareto-QAOA -> diverse candidate selection -> Pareto archive -> AI adaptive update -> quantum-seeded evolutionary refinement -> final Pareto front.

Project scope:

- Static multi-objective wireless channel allocation only.
- Objectives: f1 = negative throughput, f2 = interference, f3 = energy.
- Plots display throughput as `-f1`.
- No QWOA is included.
- No temporal channel handoff model is included.
- The optional Qiskit noisy demo is only a small circuit-level validation and does not replace the main feasible-space numerical benchmark.

Why this route is defensible:

- Feasible-space QAOA keeps assignment sampling valid by construction in the main numerical benchmark.
- CVaR scoring emphasizes high-quality sampled assignments.
- Adaptive scalarization improves search control, while fixed scalarization provides a stable comparator under the same reference point.
- Quantum-Seeded Evolutionary MOO tests whether QAOA candidates are useful to a strong classical optimizer under a comparable evaluation budget.
- Reporting uses final HV, AUC-HV, early HV, archive size, feasibility, valid sample efficiency, and multi-seed win rates.
- Extended Benchmark Evidence: current outputs cover 21 static benchmark cases, including scalability, conflict, objective-structure, budget-sweep, and channel-rich settings. The largest completed case is `scale_10x3`, with 59049 feasible assignments.
- Evidence for AI-Adaptive Pareto-QAOA: the adaptive study covers five required cases and five seeds per case, with real component ablations, action logging, real counterfactual replay, and lightweight paired statistics. The current conclusion is conservative: replayed adaptive schedules improve coverage and remain competitive in AUC-HV, but online adaptation does not dominate fixed scheduling on every seed.
