# QUANTRUM: Pareto-QAOA for Wireless Channel Allocation

> *A QAOA-based multi-objective optimization framework for communication resource scheduling*

---

## 📡 Problem

Wireless networks need to assign limited channels to multiple users. A good allocation should achieve **high throughput**, **low interference**, and **low energy consumption**. These objectives are often **conflicting**—improving one may worsen another.

---

## 🧠 Model

Channel assignments under a specific trade-off preference. We formulate wireless channel allocation as a **constrained combinatorial optimization problem**. Each candidate solution represents a complete user-channel assignment. Constraint penalties are used to avoid invalid or high-conflict assignments.

---

## ⚛️ Solution

We use **QAOA** to solve scalarized versions of the problem under different objective weights. Each QAOA run searches for good channel assignments under a specific trade-off preference.

**Key Innovation:** Instead of finding a single "optimal" solution, we explore the **Pareto frontier**—the set of trade-off solutions where no objective can be improved without sacrificing another.

---

## 🛠️ Tech Ingredients

- **QUBO Formulation** → Maps the problem to quantum-compatible form
- **QAOA Circuit** → Quantum Approximate Optimization Algorithm
- **Multiple Weight Settings** → Scalarize different trade-off preferences
- **Pareto Archive** → Non-dominated solution selection

---

## 🎯 Expected Demo

- **Network:** 5 users × 3 channels
- **Comparison:** Classical algorithms vs. Pareto-QAOA results
- **Visualization:** Pareto front among **Throughput**, **Interference**, and **Energy**

---

## 🌐 Quantum Multi-Objective Optimization in Communication

> *From one optimal channel assignment to a Pareto set of communication trade-offs.*

---

## 📊 Demo Showcase

### Live Demo Features

1. **Interactive Parameters:** Users, channels, QAOA layers (p)
2. **Before/After Visualization:** Conflict graph comparison
3. **Real-time Metrics:** Throughput gain, fairness index, conflicts resolved
4. **Convergence Plot:** QAOA energy vs. iterations
5. **Pareto Frontier:** Multi-objective trade-off visualization

### Expected Results

| Metric | Before (Random) | After (Pareto-Q) | Improvement |
|:---|:---:|:---:|:---:|
| **Conflicts** | 6 | 2 | **-67%** |
| **Throughput** | 85% | 92% | **+8%** |
| **Fairness Index** | 0.78 | 0.94 | **+21%** |
| **Feasible Solutions** | 68% | **100%** | **+32%** |

---

## 🗓️ Development Timeline

| Day | Focus | Deliverable |
|:---|:---|:---|
| **Day 1** | Formulation + Setup | QUBO formulation, Qiskit environment, Streamlit skeleton |
| **Day 2** | Implementation | QAOA circuit with Dicke+XY, COBYLA optimizer, network graph |
| **Day 3** | Polish + Present | Metrics dashboard, convergence plots, live demo rehearsal |

---

## 📖 Related Work

| Work | Algorithm | Problem | Platform | Relevance |
|:---|:---|:---|:---|:---|
| Min et al. (2026) | Subspace-Confined QAOA | CBRS multi-channel allocation | Qiskit | **Core innovation** |
| Choi et al. (2020) | QAOA (QAOS) | Wireless scheduling | Cirq | QAOA for NP-hard wireless |
| Tafur Monroy et al. (2025) | QAOA | MO-RSA optimization | IBM Qiskit | Multi-objective validation |
| Survey (2024) | QAOA, QA, QGA | Channel assignment | IBM Qiskit, D-Wave | Comprehensive benchmark |

---

## 🎓 References


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
