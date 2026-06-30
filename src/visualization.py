import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_interference_graph(instance, path):
    g = nx.Graph()
    g.add_nodes_from(range(instance.users))
    for u, v, w in instance.interference_edges:
        g.add_edge(u, v, weight=w)
    pos = nx.spring_layout(g, seed=instance.seed)
    plt.figure(figsize=(5, 4))
    nx.draw_networkx_nodes(g, pos, node_color="#4C78A8", node_size=700)
    nx.draw_networkx_labels(g, pos, font_color="white")
    widths = [1 + 3 * g[u][v]["weight"] for u, v in g.edges]
    nx.draw_networkx_edges(g, pos, width=widths, edge_color="#F58518", alpha=0.8)
    nx.draw_networkx_edge_labels(g, pos, edge_labels={(u, v): f"{g[u][v]['weight']:.2f}" for u, v in g.edges}, font_size=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_assignment_comparison(instance, before, after, path):
    g = nx.Graph()
    g.add_nodes_from(range(instance.users))
    g.add_edges_from((u, v) for u, v, _ in instance.interference_edges)
    pos = nx.circular_layout(g)
    cmap = plt.get_cmap("Set2", instance.channels)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, title, assignment in zip(axes, ["Random initial", "Optimized selected"], [before, after]):
        nx.draw(g, pos, ax=ax, with_labels=True, node_color=assignment, cmap=cmap, vmin=0, vmax=instance.channels - 1, node_size=650)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_pareto_2d(exact, qaoa, path):
    plt.figure(figsize=(6, 4))
    if exact:
        raw = np.asarray([r[2] for r in exact])
        plt.scatter(raw[:, 1], raw[:, 0], c=raw[:, 2], cmap="viridis", marker="x", label="Exact Pareto")
    if qaoa:
        raw = np.asarray([r["raw"] for r in qaoa])
        sc = plt.scatter(raw[:, 1], raw[:, 0], c=raw[:, 2], cmap="plasma", label="QAOA archive")
        plt.colorbar(sc, label="Energy")
    plt.xlabel("Interference")
    plt.ylabel("Throughput")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_pareto_3d(exact, qaoa, path):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    if exact:
        raw = np.asarray([r[2] for r in exact])
        ax.scatter(raw[:, 0], raw[:, 1], raw[:, 2], marker="x", label="Exact Pareto")
    if qaoa:
        raw = np.asarray([r["raw"] for r in qaoa])
        ax.scatter(raw[:, 0], raw[:, 1], raw[:, 2], label="QAOA archive")
    ax.set_xlabel("Throughput")
    ax.set_ylabel("Interference")
    ax.set_zlabel("Energy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_metric_comparison(summary_df, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, col in zip(axes, ["hypervolume", "igd_plus", "pareto_archive_size"]):
        ax.bar(summary_df["method"], summary_df[col], color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][:len(summary_df)])
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_top_qaoa_samples(samples, path):
    top = samples[:10]
    labels = ["-".join(str(int(x)) for x in s["assignment"]) for s in top]
    probs = [s.get("probability", 0.0) for s in top]
    plt.figure(figsize=(7, 4))
    plt.barh(labels[::-1], probs[::-1], color="#4C78A8")
    plt.xlabel("Probability")
    plt.ylabel("Decoded assignment")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

