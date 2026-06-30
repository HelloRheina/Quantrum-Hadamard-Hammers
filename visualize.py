import numpy as np
import matplotlib.pyplot as plt


def _throughput(obj):
    return -np.asarray(obj)[:, 0]


def pareto_3d(qaoa_obj, baseline_obj, path):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    if len(baseline_obj):
        ax.scatter(_throughput(baseline_obj), baseline_obj[:, 1], baseline_obj[:, 2], s=32, alpha=0.65, label="Baselines")
    if len(qaoa_obj):
        ax.scatter(_throughput(qaoa_obj), qaoa_obj[:, 1], qaoa_obj[:, 2], s=46, label="Pareto-QAOA")
    ax.set_xlabel("Throughput")
    ax.set_ylabel("Interference")
    ax.set_zlabel("Energy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def projection(obj_by_method, x_name, y_name, path):
    idx = {"throughput": 0, "interference": 1, "energy": 2}
    plt.figure(figsize=(5.8, 4.2))
    for name, obj in obj_by_method.items():
        if len(obj) == 0:
            continue
        x = _throughput(obj) if x_name == "throughput" else obj[:, idx[x_name]]
        y = _throughput(obj) if y_name == "throughput" else obj[:, idx[y_name]]
        plt.scatter(x, y, label=name, alpha=0.78)
    plt.xlabel(x_name.title())
    plt.ylabel(y_name.title())
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def hv_convergence(hv_curve, path):
    plt.figure(figsize=(6, 4))
    plt.plot(hv_curve["iteration"], hv_curve["hypervolume"], marker="o", color="#4C78A8")
    plt.xlabel("Outer iteration")
    plt.ylabel("Archive hypervolume")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def method_comparison(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(summary))))
    for ax, col in zip(axes, ["hypervolume", "pareto_size", "runtime_seconds"]):
        ax.bar(summary["method"], summary[col], color=colors)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def adaptive_weight_history(history, path):
    plt.figure(figsize=(7, 4))
    plt.plot(history["iteration"], history["w_throughput"], marker="o", label="Throughput")
    plt.plot(history["iteration"], history["w_interference"], marker="o", label="Interference")
    plt.plot(history["iteration"], history["w_energy"], marker="o", label="Energy")
    plt.xlabel("Iteration")
    plt.ylabel("Selected weight")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def adaptive_resource_history(history, path):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(history["iteration"], history["penalty"], marker="o", label="Penalty")
    ax1.plot(history["iteration"], history["n_most_prob"], marker="o", label="N most probable")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Penalty / candidate budget")
    ax2 = ax1.twinx()
    ax2.plot(history["iteration"], history["shots"], color="#54A24B", marker="s", label="Shots")
    ax2.plot(history["iteration"], history["feasibility_rate"], color="#B279A2", marker="s", label="Feasibility")
    ax2.set_ylabel("Shots / feasibility")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def hv_convergence_evals(curves, path):
    plt.figure(figsize=(6.5, 4))
    for name, curve in curves.items():
        if curve is None or len(curve) == 0:
            continue
        plt.plot(curve["cumulative_evaluations"], curve["hv"], marker="o", label=name)
    plt.xlabel("Cumulative objective evaluations")
    plt.ylabel("Hypervolume")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def auc_hv_comparison(summary, path):
    plt.figure(figsize=(7, 4))
    plt.bar(summary["method"], summary["auc_hv"], color="#4C78A8")
    plt.ylabel("AUC-HV")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def early_hv_comparison(summary, path):
    cols = ["hv_at_25pct", "hv_at_50pct", "hv_at_75pct", "final_hv"]
    x = np.arange(len(summary))
    width = 0.18
    plt.figure(figsize=(10, 4.5))
    for i, col in enumerate(cols):
        plt.bar(x + (i - 1.5) * width, summary[col], width=width, label=col)
    plt.xticks(x, summary["method"], rotation=25, ha="right")
    plt.ylabel("Hypervolume")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def objective_conflict_heatmap(corr, path):
    plt.figure(figsize=(4.8, 4.2))
    im = plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    labels = ["-Throughput", "Interference", "Energy"]
    plt.xticks(range(3), labels, rotation=25, ha="right")
    plt.yticks(range(3), labels)
    plt.colorbar(im, label="Correlation")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def objective_pair_scatter(objectives, path):
    obj = np.asarray(objectives, dtype=float)
    throughput = -obj[:, 0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].scatter(throughput, obj[:, 1], s=12, alpha=0.65)
    axes[0].set_xlabel("Throughput")
    axes[0].set_ylabel("Interference")
    axes[1].scatter(throughput, obj[:, 2], s=12, alpha=0.65)
    axes[1].set_xlabel("Throughput")
    axes[1].set_ylabel("Energy")
    axes[2].scatter(obj[:, 1], obj[:, 2], s=12, alpha=0.65)
    axes[2].set_xlabel("Interference")
    axes[2].set_ylabel("Energy")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def archive_source_composition(source_df, path):
    if len(source_df) == 0:
        return
    plt.figure(figsize=(7, 4))
    plt.bar(source_df["source"], source_df["archive_fraction"], color="#4C78A8")
    plt.ylabel("Archive fraction")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def final_pipeline_diagram(path):
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")
    steps = [
        "Static wireless\ninstance",
        "Feasible assignment\nspace",
        "Normalized\nobjectives",
        "CVaR Pareto-QAOA\ncandidate generator",
        "Pareto archive\n+ AI control",
        "Quantum-seeded\nEA refinement",
        "Final Pareto\nfront",
    ]
    xs = np.linspace(0.06, 0.94, len(steps))
    for i, (x, label) in enumerate(zip(xs, steps)):
        ax.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#E8F1FB", edgecolor="#4C78A8", linewidth=1.5),
            transform=ax.transAxes,
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.055, 0.55),
                xytext=(x + 0.055, 0.55),
                arrowprops=dict(arrowstyle="->", color="#333333", linewidth=1.5),
                xycoords=ax.transAxes,
            )
    ax.text(
        0.5,
        0.12,
        "Static multi-objective channel allocation: throughput / interference / energy. No dynamic spectrum logic or switching cost.",
        ha="center",
        fontsize=11,
        transform=ax.transAxes,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def multiseed_bar(summary_df, metric, path):
    stats = summary_df.groupby("method")[metric].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(9, 4.5))
    plt.bar(stats["method"], stats["mean"], yerr=stats["std"].fillna(0), color="#4C78A8", capsize=4)
    plt.ylabel(metric)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def final_tuning_heatmap(tuning_df, path):
    if len(tuning_df) == 0:
        return
    plot_df = tuning_df.copy()
    plot_df["value_label"] = plot_df["value"].astype(str)
    pivot = plot_df.pivot_table(index="parameter", columns="value_label", values="auc_hv", aggfunc="mean", sort=False)
    plt.figure(figsize=(8, 4.8))
    im = plt.imshow(pivot.fillna(pivot.min().min()), aspect="auto", cmap="viridis")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    plt.colorbar(im, label="AUC-HV")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def classical_optimizer_ablation_plot(df, path):
    if len(df) == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].bar(df["optimizer"], df["final_hv"], color="#4C78A8")
    axes[0].set_title("Final HV")
    axes[1].bar(df["optimizer"], df["auc_hv"], color="#54A24B")
    axes[1].set_title("AUC-HV")
    axes[2].bar(df["optimizer"], df["runtime"], color="#F58518")
    axes[2].set_title("Runtime")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
