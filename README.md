# QUANTRUM: Pareto-QAOA for Wireless Channel Allocation

> *A QAOA-based multi-objective optimization framework for communication resource scheduling*

---

## 📡 Problem

Wireless networks need to assign limited channels to multiple users. A good allocation should achieve **high throughput**, **low interference**, and **low energy consumption**. These objectives are often **conflicting**—improving one may worsen another.

## Problem

For `U` users and `C` channels, an allocation is:

```text
a = [c_0, c_1, ..., c_{U-1}],  c_u in {0, ..., C-1}
```

The main hackathon case is:

```text
users = 8
channels = 3
feasible assignments = 3^8 = 6561
logical one-hot qubits = 8 * 3 = 24
```

## Objectives

The Pareto archive minimizes:

```text
f1 = -throughput
f2 = interference
f3 = energy
```

Throughput uses a simple SINR-style model:

```text
T(a) = sum_u log2(1 + signal_u / (noise + interference_u))
```

The stored objective vector is:

```text
f(a) = [-T(a), I(a), E(a)]
```

Raw objectives are used for Pareto dominance, archive contents, and displayed trade-offs. Scalarized QAOA costs and hypervolume are computed in normalized objective space.

## Encoding

Logically, the project uses one-hot channel encoding:

```text
channel 0 -> [1, 0, 0]
channel 1 -> [0, 1, 0]
channel 2 -> [0, 0, 1]
```

The default benchmark uses a faster feasible-space encoding instead of simulating the full `2^(U*C)` bitstring space. It directly simulates the valid assignment space of size `C^U`, so every state corresponds to one legal user-channel allocation.

## Pareto-QAOA

For each scalarization weight vector `w`, QAOA optimizes:

```text
C_w(a) = w1 * norm(f1(a)) + w2 * norm(f2(a)) + w3 * norm(f3(a))
```

The default weight pool covers throughput-focused, interference-focused, energy-focused, pairwise, balanced, and biased trade-off directions.

Fixed Pareto-QAOA cycles through this weight pool. AI-Adaptive Pareto-QAOA uses the same pool but selects the next direction online.

## AI-Adaptive Controller

The AI-Adaptive controller observes:

- hypervolume improvement
- Pareto archive growth
- direction coverage
- feasibility rate
- duplicate rate
- evaluation cost

It then selects the next scalarization weight using a UCB-style policy, transfers QAOA parameters between nearby weight directions, and can adjust candidate diversity, CVaR focus, and sampling resources when search stagnates.

The controller is intentionally lightweight and interpretable. It is not a neural network.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main benchmark:

```bash
python run_experiment.py --case main_dense_small_cell
```

Validate generated results:

```bash
python validate_results.py
```

Run tests:

```bash
python -m pytest -q
```

The project is also runnable with:

```bash
python run_experiment.py
```

## Optional Studies

Adaptive QAOA study:

```bash
python run_adaptive_qaoa_study.py --all-main-cases --seeds 0 1 2 3 4
```

Classical optimizer and component ablations:

```bash
python run_ablation_suite.py --case main_dense_small_cell
```

Extended benchmark:

```bash
python run_extended_benchmark_suite.py
```

## Optional Qiskit Demo

Qiskit is optional and is not required for the main benchmark.

```bash
python run_qiskit_circuit_gallery.py --users 2 --channels 2
python run_qiskit_noisy_demo.py --users 3 --channels 2 --shots 1024 --noise depolarizing
```

The Qiskit demo is intentionally small-scale. It validates the circuit-level idea that a constraint-preserving XY mixer keeps more probability inside the one-hot feasible space than a naive X mixer.

For presentation, the repository also includes an 8-user, 3-channel logical Qiskit schematic that maps the main benchmark to 24 one-hot qubits without drawing an unreadable full 24-qubit diagonal cost circuit.

## 🤝 Team

| Role | Responsibility |


---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

---

## 📧 Contact

For questions or collaboration, please reach out via GitHub Issues or email the team.

---

**Built with ❤️ for the ISIT26 Quantum Hackathon**
