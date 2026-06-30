from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt

from problem import make_problem
from run_qiskit_noisy_demo import (
    build_display_circuit,
    save_circuit_diagram,
    save_circuit_schematic,
    try_import_qiskit,
)


def parse_args():
    parser = ArgumentParser(description="Draw small Qiskit circuit diagrams for hackathon presentation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.35)
    parser.add_argument("--users", type=int, default=2)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--include-3x2", action="store_true", default=True)
    return parser.parse_args()


def save_noise_schematic(path):
    Path("figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.set_title("Noisy Qiskit validation model", fontsize=14, weight="bold", loc="left")

    boxes = [
        (0.5, "Prepared\none-hot state", "#54A24B"),
        (2.6, "Cost phase\nmulti-objective", "#F58518"),
        (4.7, "Mixer\nX or XY", "#4C78A8"),
        (6.8, "Noise model\ndepolarizing + readout", "#E45756"),
        (8.8, "Measured\nbitstrings", "#777777"),
    ]
    for x, label, color in boxes:
        ax.add_patch(plt.Rectangle((x - 0.65, 0.9), 1.3, 1.0, facecolor=color, alpha=0.16, edgecolor=color, lw=1.8))
        ax.text(x, 1.4, label, ha="center", va="center", fontsize=10)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + 0.75
        x1 = boxes[i + 1][0] - 0.75
        ax.annotate("", xy=(x1, 1.4), xytext=(x0, 1.4), arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#444444"})
    ax.text(
        0.5,
        0.35,
        "Presentation point: noise reduces valid assignment sampling, while the XY mixer keeps more probability in the feasible one-hot subspace.",
        fontsize=10,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_case(q, users, channels, seed, gamma, beta):
    problem = make_problem(users, channels, seed, case_name=f"qiskit_gallery_{users}x{channels}")
    balanced = (1 / 3, 1 / 3, 1 / 3)
    suffix = f"{users}x{channels}"
    outputs = []
    for variant, title, filename in [
        ("naive_x", f"{suffix}: naive one-hot QAOA with X mixer", f"figures/qiskit_gallery_{suffix}_naive_x.png"),
        (
            "xy_constraint_preserving",
            f"{suffix}: constraint-preserving one-hot QAOA with XY mixer",
            f"figures/qiskit_gallery_{suffix}_xy_mixer.png",
        ),
    ]:
        qc = build_display_circuit(q, problem, balanced, variant, gamma, beta)
        save_circuit_diagram(qc, filename, title)
        outputs.append(filename)
    schematic = f"figures/qiskit_gallery_{suffix}_comparison_schematic.png"
    save_circuit_schematic(problem, schematic)
    outputs.append(schematic)
    return outputs


def main():
    args = parse_args()
    q = try_import_qiskit()
    Path("figures").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    if "error" in q:
        reason = str(q["error"])
        Path("results/qiskit_circuit_gallery.md").write_text(
            "# Qiskit Circuit Gallery\n\n"
            f"Skipped: {reason}\n\n"
            "Install hint: `pip install qiskit qiskit-aer`.\n",
            encoding="utf-8",
        )
        print(f"Qiskit circuit gallery skipped: {reason}")
        return 0

    outputs = []
    outputs.extend(draw_case(q, args.users, args.channels, args.seed, args.gamma, args.beta))
    if args.include_3x2 and not (args.users == 3 and args.channels == 2):
        outputs.extend(draw_case(q, 3, 2, args.seed, args.gamma, args.beta))
    noise_path = "figures/qiskit_gallery_noise_model_schematic.png"
    save_noise_schematic(noise_path)
    outputs.append(noise_path)

    report = [
        "# Qiskit Circuit Gallery",
        "",
        "These figures are small-scale presentation diagrams. They reduce the circuit size so the one-hot encoding, cost phase, and mixer differences are visible on slides.",
        "",
        "Recommended slide order:",
        "",
        "1. `figures/qiskit_gallery_2x2_comparison_schematic.png`",
        "2. `figures/qiskit_gallery_2x2_naive_x.png`",
        "3. `figures/qiskit_gallery_2x2_xy_mixer.png`",
        "4. `figures/qiskit_gallery_noise_model_schematic.png`",
        "5. `figures/qiskit_noise_feasibility.png`",
        "",
        "Generated files:",
        "",
    ]
    report.extend(f"- `{path}`" for path in outputs)
    Path("results/qiskit_circuit_gallery.md").write_text("\n".join(report), encoding="utf-8")
    print("Qiskit circuit gallery generated.")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
