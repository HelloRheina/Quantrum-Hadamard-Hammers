# Small-scale Qiskit noisy validation

Main benchmark results use feasible-space numerical simulation. This optional Qiskit/Aer demo is only circuit-level validation for small static wireless channel allocation cases.

The naive X-mixer circuit explores the full one-hot bitstring space. The XY mixer starts in the one-hot feasible subspace and uses pairwise exchange-style gates, so it better preserves valid assignment sampling.

Final HV is computed in normalized objective space using the full small-case assignment-table min/max values.

Noise generally reduces sampling quality. The demo is not a hardware-scale benchmark and does not replace the main feasible-space Pareto-QAOA results.

Circuit diagrams for project presentation:

- `figures/qiskit_naive_x_circuit.png`
- `figures/qiskit_xy_constraint_preserving_circuit.png`
- `figures/qiskit_circuit_comparison_schematic.png`

| status   | variant                  | backend   |   users |   channels |   shots |   feasibility_rate |   valid_sample_efficiency |   final_hv |   archive_size |
|:---------|:-------------------------|:----------|--------:|-----------:|--------:|-------------------:|--------------------------:|-----------:|---------------:|
| ok       | naive_x                  | ideal     |       3 |          2 |    1024 |           0.152588 |                 0.0078125 |    1.76565 |              4 |
| ok       | naive_x                  | noisy     |       3 |          2 |    1024 |           0.150879 |                 0.0078125 |    1.76565 |              4 |
| ok       | xy_constraint_preserving | ideal     |       3 |          2 |    1024 |           1        |                 0.0078125 |    1.76565 |              4 |
| ok       | xy_constraint_preserving | noisy     |       3 |          2 |    1024 |           0.833008 |                 0.0078125 |    1.76565 |              4 |