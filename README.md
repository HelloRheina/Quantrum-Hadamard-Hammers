
---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|:---|:---|:---|
| **Quantum Core** | [Qiskit](https://qiskit.org/) 1.0+ | QAOA circuit, Dicke States, Aer simulator |
| **Optimization** | `scipy.optimize` (COBYLA) | Classical outer-loop parameter optimization |
| **UI & Visualization** | [Streamlit](https://streamlit.io/) | Interactive demo, live metrics, before/after graphs |
| **Graph Visualization** | `networkx` + `matplotlib` | Network topology, conflict mapping, Pareto front |
| **Environment** | Python 3.9+ + Jupyter | Development and experimentation |

### Key Qiskit Features Used

- `QAOAAnsatz` for circuit construction
- `DickeState` for constraint-preserving initialization
- `SparsePauliOp` for custom XY-mixer Hamiltonians
- `AerSimulator` with noise models for NISQ emulation

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

**Primary Reference (Our Innovation):**

> Min, G., Seo, Y., & Heo, J. (2026). *Subspace-Confined QAOA with Generalized Dicke States for Multi-Channel Allocation in 5G CBRS Networks.* arXiv preprint arXiv:2601.16396.

**QAOA for Wireless Optimization:**

> Choi, J., Oh, S., & Kim, J. (2020). *Quantum Approximation for Wireless Scheduling.* arXiv:2004.11229.

**Quantum Computing for Channel Assignment:**

> *Summary of QC works on channel assignment in wireless networks* (2024 Survey). TABLE VI in arXiv:2406.02240.

**QUBO & QAOA Implementation:**

> *Multi-objective quantum routing optimization* (2025). Chapter 4 in arXiv:2506.16524.

---

## 🤝 Team

| Role | Responsibility |
|:---|:---|
| **Person A** | Mathematical formulation, QUBO, multi-objective weighting |
| **Person B** | QAOA circuit, Qiskit implementation, optimizer integration |
| **Person C** | Visualization, Streamlit UI, metrics dashboard |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Qiskit team for the quantum computing framework
- Min et al. (2026) for the subspace-confined QAOA innovation
- ISIT26 Quantum Hackathon organizers

---

## 📧 Contact

For questions or collaboration, please reach out via GitHub Issues or email the team.

---

**Built with ❤️ for the ISIT26 Quantum Hackathon**
