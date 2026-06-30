import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import normalized_hypervolume_mc
from pareto import update_archive
from problem import decode_bitstring, encode_assignment, make_problem, objective_table, objective_vector


WEIGHTS = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1 / 3, 1 / 3, 1 / 3),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Optional small-scale Qiskit noisy QAOA validation")
    parser.add_argument("--users", type=int, default=3)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--noise", choices=["none", "depolarizing"], default="depolarizing")
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.35)
    return parser.parse_args()


def try_import_qiskit():
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit import Gate
        from qiskit.circuit.library import DiagonalGate
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
        return {
            "QuantumCircuit": QuantumCircuit,
            "Gate": Gate,
            "transpile": transpile,
            "DiagonalGate": DiagonalGate,
            "AerSimulator": AerSimulator,
            "NoiseModel": NoiseModel,
            "depolarizing_error": depolarizing_error,
            "ReadoutError": ReadoutError,
        }
    except Exception as exc:
        return {"error": exc}


def one_hot_state_vector(channels):
    vec = np.zeros(2 ** channels, dtype=complex)
    for c in range(channels):
        vec[1 << c] = 1 / np.sqrt(channels)
    return vec


def bitstring_costs(problem, weights, gamma):
    n = problem.users * problem.channels
    assignments, objectives = objective_table(problem)
    obj_min, obj_max = objectives.min(axis=0), objectives.max(axis=0)
    phases = []
    for z in range(2 ** n):
        bits = np.array([(z >> i) & 1 for i in range(n)], dtype=np.int8)
        assignment, feasible = decode_bitstring(bits, problem, repair=True)
        raw = objective_vector(problem, assignment)
        normalized = (raw - obj_min) / (obj_max - obj_min + 1e-12)
        violation = 0.0 if feasible else sum((bits[u * problem.channels:(u + 1) * problem.channels].sum() - 1) ** 2 for u in range(problem.users))
        cost = float(np.dot(weights, normalized)) + 4.0 * violation
        phases.append(np.exp(-1j * gamma * cost))
    return phases


def build_circuit(q, problem, weights, variant, gamma, beta):
    QuantumCircuit = q["QuantumCircuit"]
    DiagonalGate = q["DiagonalGate"]
    n = problem.users * problem.channels
    qc = QuantumCircuit(n, n)

    if variant == "naive_x":
        qc.h(range(n))
    else:
        block_state = one_hot_state_vector(problem.channels)
        for u in range(problem.users):
            block = list(range(u * problem.channels, (u + 1) * problem.channels))
            qc.initialize(block_state, block)

    qc.append(DiagonalGate(bitstring_costs(problem, weights, gamma)), list(range(n)))

    if variant == "naive_x":
        for qubit in range(n):
            qc.rx(2 * beta, qubit)
    else:
        for u in range(problem.users):
            block = list(range(u * problem.channels, (u + 1) * problem.channels))
            for i, qi in enumerate(block):
                for qj in block[i + 1:]:
                    qc.rxx(2 * beta, qi, qj)
                    qc.ryy(2 * beta, qi, qj)

    qc.measure(range(n), range(n))
    return qc


def build_display_circuit(q, problem, weights, variant, gamma, beta):
    QuantumCircuit = q["QuantumCircuit"]
    Gate = q["Gate"]
    n = problem.users * problem.channels
    qc = QuantumCircuit(n, n)

    if variant == "naive_x":
        qc.h(range(n))
    else:
        for u in range(problem.users):
            block = list(range(u * problem.channels, (u + 1) * problem.channels))
            qc.append(Gate("1-hot init", len(block), []), block)

    qc.append(Gate("Cost phase", n, []), list(range(n)))

    if variant == "naive_x":
        for qubit in range(n):
            qc.rx(2 * beta, qubit)
    else:
        for u in range(problem.users):
            block = list(range(u * problem.channels, (u + 1) * problem.channels))
            for i, qi in enumerate(block):
                for qj in block[i + 1:]:
                    qc.rxx(2 * beta, qi, qj)
                    qc.ryy(2 * beta, qi, qj)

    qc.measure(range(n), range(n))
    return qc


def save_circuit_diagram(qc, path, title):
    Path("figures").mkdir(exist_ok=True)
    try:
        fig = qc.draw(output="mpl", fold=80, idle_wires=False)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return
    except Exception:
        text = str(qc.draw(output="text", fold=100))
        lines = text.splitlines()
        width = max(10, min(22, max(len(line) for line in lines) * 0.095))
        height = max(4, min(14, len(lines) * 0.28 + 1.0))
        fig, ax = plt.subplots(figsize=(width, height))
        ax.axis("off")
        ax.set_title(title, fontsize=12, loc="left")
        ax.text(0.0, 0.98, text, family="monospace", fontsize=8, va="top", ha="left")
        fig.tight_layout()
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)


def draw_circuit_diagrams(q, problem, args):
    balanced = (1 / 3, 1 / 3, 1 / 3)
    outputs = []
    for variant, title, filename in [
        (
            "naive_x",
            "Naive one-hot QAOA with X mixer",
            "figures/qiskit_naive_x_circuit.png",
        ),
        (
            "xy_constraint_preserving",
            "Constraint-preserving one-hot QAOA with XY mixer",
            "figures/qiskit_xy_constraint_preserving_circuit.png",
        ),
    ]:
        qc = build_display_circuit(q, problem, balanced, variant, args.gamma, args.beta)
        save_circuit_diagram(qc, filename, title)
        outputs.append(filename)
    save_circuit_schematic(problem, "figures/qiskit_circuit_comparison_schematic.png")
    outputs.append("figures/qiskit_circuit_comparison_schematic.png")
    return outputs


def save_circuit_schematic(problem, path):
    Path("figures").mkdir(exist_ok=True)
    n = problem.users * problem.channels
    labels = [f"u{u}c{c}" for u in range(problem.users) for c in range(problem.channels)]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9.0), sharex=True)
    variants = [
        ("Naive QAOA (X mixer)", "H on all qubits", "RX on every qubit", "#4C78A8"),
        ("Constraint-preserving QAOA (XY mixer)", "1-hot init per user", "RXX/RYY exchanges inside each user block", "#54A24B"),
    ]
    for ax, (title, init_label, mixer_label, color) in zip(axes, variants):
        ax.set_xlim(-0.3, 4.8)
        ax.set_ylim(-0.8, n + 1.15)
        ax.axis("off")
        ax.text(-0.3, n + 0.88, title, ha="left", va="center", fontsize=13, weight="bold")
        for i, label in enumerate(labels):
            y = n - 1 - i
            ax.plot([0, 4.5], [y, y], color="#B8B8B8", lw=1.1)
            ax.text(-0.08, y, label, va="center", ha="right", fontsize=8)

        stages = [
            (0.45, init_label, color),
            (1.65, "Cost phase\nw=[1/3,1/3,1/3]", "#F58518"),
            (2.9, mixer_label, color),
            (4.05, "Measure", "#777777"),
        ]
        for x, label, stage_color in stages:
            ax.add_patch(plt.Rectangle((x - 0.32, -0.35), 0.64, n - 0.3, facecolor=stage_color, alpha=0.14, edgecolor=stage_color, lw=1.4))
            ax.text(x, n + 0.20, label, ha="center", va="bottom", fontsize=9, color="#222222")

        if "XY" in title:
            for u in range(problem.users):
                y0 = n - 1 - (u * problem.channels)
                y1 = n - 1 - (u * problem.channels + problem.channels - 1)
                ax.plot([2.9, 2.9], [y0, y1], color="#2F7D32", lw=2.0)
                ax.scatter([2.9, 2.9], [y0, y1], s=24, color="#2F7D32", zorder=3)
        else:
            for i in range(n):
                y = n - 1 - i
                ax.scatter([2.9], [y], s=22, color="#2F5F9E", zorder=3)

    fig.suptitle("Small-scale Qiskit circuit-level validation: naive X mixer vs constraint-preserving XY mixer", fontsize=14, y=0.985)
    fig.subplots_adjust(hspace=0.55, top=0.90, bottom=0.05, left=0.08, right=0.98)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_noise_model(q):
    NoiseModel = q["NoiseModel"]
    depolarizing_error = q["depolarizing_error"]
    ReadoutError = q["ReadoutError"]
    model = NoiseModel()
    one = depolarizing_error(0.003, 1)
    two = depolarizing_error(0.015, 2)
    ro = ReadoutError([[0.985, 0.015], [0.02, 0.98]])
    for gate in ["rx", "ry", "rz", "h", "x"]:
        model.add_all_qubit_quantum_error(one, [gate])
    for gate in ["rxx", "ryy"]:
        model.add_all_qubit_quantum_error(two, [gate])
    if ro is not None:
        model.add_all_qubit_readout_error(ro)
    return model


def counts_to_archive(problem, counts):
    total = sum(counts.values())
    records = {}
    valid = 0
    for bitstr, count in counts.items():
        # Qiskit reports classical bits big-endian; reverse to match variable index order.
        bits = np.array([int(x) for x in bitstr[::-1]], dtype=np.int8)
        assignment, feasible = decode_bitstring(bits, problem, repair=False)
        if feasible:
            valid += count
            key = tuple(int(x) for x in assignment)
            records[key] = records.get(key, 0) + count
    if not records:
        return np.empty((0, problem.users), dtype=int), np.empty((0, 3)), 0.0, 0.0
    sol = np.array(list(records.keys()), dtype=int)
    obj = np.array([objective_vector(problem, s) for s in sol])
    archive_s, archive_o = update_archive([], [], sol, obj)
    feasibility = valid / max(1, total)
    valid_eff = len(records) / max(1, total)
    return archive_s, archive_o, feasibility, valid_eff


def run_backend(q, circuit, shots, noise):
    AerSimulator = q["AerSimulator"]
    transpile = q["transpile"]
    simulator = AerSimulator() if noise == "none" else AerSimulator(noise_model=make_noise_model(q))
    tqc = transpile(circuit, simulator, optimization_level=1)
    result = simulator.run(tqc, shots=shots, seed_simulator=123).result()
    return result.get_counts()


def write_skip(reason):
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    pd.DataFrame([{"status": "skipped", "reason": reason}]).to_csv("results/qiskit_noisy_summary.csv", index=False)
    Path("results/qiskit_noisy_report.md").write_text(
        "# Small-scale Qiskit noisy validation\n\n"
        f"Skipped: {reason}\n\n"
        "Install hint: `pip install qiskit qiskit-aer` in the project environment.\n",
        encoding="utf-8",
    )
    print(f"Qiskit noisy demo skipped: {reason}")


def plot_outputs(df):
    Path("figures").mkdir(exist_ok=True)
    completed = df[df["status"] == "ok"]
    if len(completed) == 0:
        return
    labels = completed["variant"] + "/" + completed["backend"]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, completed["feasibility_rate"], color="#4C78A8")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Feasibility rate")
    plt.tight_layout()
    plt.savefig("figures/qiskit_noise_feasibility.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(labels, completed["final_hv"], color="#54A24B")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Final normalized HV")
    plt.tight_layout()
    plt.savefig("figures/qiskit_noise_hv_comparison.png", dpi=180)
    plt.close()


def main():
    args = parse_args()
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    q = try_import_qiskit()
    if "error" in q:
        write_skip(str(q["error"]))
        return 0

    if args.users * args.channels > 9:
        write_skip("demo intentionally limited to <=9 qubits")
        return 0

    problem = make_problem(args.users, args.channels, args.seed, case_name="qiskit_noisy_demo")
    circuit_diagrams = draw_circuit_diagrams(q, problem, args)
    _assignments, all_obj = objective_table(problem)
    reference = all_obj.max(axis=0) + np.array([1.0, 0.5, 0.5])
    obj_min, obj_max = all_obj.min(axis=0), all_obj.max(axis=0)

    rows = []
    for variant in ["naive_x", "xy_constraint_preserving"]:
        for backend in ["ideal", "noisy"]:
            if backend == "noisy" and args.noise == "none":
                continue
            merged_s, merged_o = [], []
            feasibilities = []
            efficiencies = []
            for weights in WEIGHTS:
                try:
                    circuit = build_circuit(q, problem, weights, variant, args.gamma, args.beta)
                    counts = run_backend(q, circuit, args.shots, "none" if backend == "ideal" else args.noise)
                    archive_s, archive_o, feasibility, valid_eff = counts_to_archive(problem, counts)
                    merged_s, merged_o = update_archive(merged_s, merged_o, archive_s, archive_o)
                    feasibilities.append(feasibility)
                    efficiencies.append(valid_eff)
                except Exception as exc:
                    rows.append({"status": "failed", "variant": variant, "backend": backend, "error": str(exc)})
                    break
            else:
                hv = normalized_hypervolume_mc(merged_o, reference, obj_min, obj_max, seed=args.seed)
                rows.append({
                    "status": "ok",
                    "variant": variant,
                    "backend": backend,
                    "users": args.users,
                    "channels": args.channels,
                    "shots": args.shots,
                    "feasibility_rate": float(np.mean(feasibilities)),
                    "valid_sample_efficiency": float(np.mean(efficiencies)),
                    "final_hv": hv,
                    "archive_size": len(merged_o),
                })

    df = pd.DataFrame(rows)
    df.to_csv("results/qiskit_noisy_summary.csv", index=False)
    plot_outputs(df)
    report = [
        "# Small-scale Qiskit noisy validation",
        "",
        "Main benchmark results use feasible-space numerical simulation. This optional Qiskit/Aer demo is only circuit-level validation for small static wireless channel allocation cases.",
        "",
        "The naive X-mixer circuit explores the full one-hot bitstring space. The XY mixer starts in the one-hot feasible subspace and uses pairwise exchange-style gates, so it better preserves valid assignment sampling.",
        "",
        "Final HV is computed in normalized objective space using the full small-case assignment-table min/max values.",
        "",
        "Noise generally reduces sampling quality. The demo is not a hardware-scale benchmark and does not replace the main feasible-space Pareto-QAOA results.",
        "",
        "Circuit diagrams for project presentation:",
        "",
        "- `figures/qiskit_naive_x_circuit.png`",
        "- `figures/qiskit_xy_constraint_preserving_circuit.png`",
        "- `figures/qiskit_circuit_comparison_schematic.png`",
        "",
        df.to_markdown(index=False),
    ]
    Path("results/qiskit_noisy_report.md").write_text("\n".join(report), encoding="utf-8")
    print("Qiskit noisy demo completed.")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
