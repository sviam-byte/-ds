"""Visualization helpers for connectivity matrices."""

from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_heatmap(
    matrix: np.ndarray,
    title: str,
    labels: list[str] | None = None,
    legend_text: str = "",
    annotate: bool = False,
    vmin=None,
    vmax=None,
) -> BytesIO:
    """Generate a heatmap image and return it as an in-memory PNG buffer."""
    fig, ax = plt.subplots(figsize=(4, 3.2))

    if matrix is None or not isinstance(matrix, np.ndarray) or matrix.size == 0:
        ax.text(0.5, 0.5, "Error\n(No Data)", ha="center", va="center", color="red", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        cax = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        fig.colorbar(cax, ax=ax)
        ax.set_title(title, fontsize=10)

        if labels:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)

        if annotate and matrix.shape[0] < 10:
            min_val = vmin if vmin is not None else np.nanmin(matrix)
            max_val = vmax if vmax is not None else np.nanmax(matrix)
            threshold = min_val + (max_val - min_val) / 2.0 if np.isfinite(min_val) and np.isfinite(max_val) and max_val > min_val else 0.5
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = matrix[i, j]
                    display_val = "NaN" if np.isnan(val) else f"{val:.2f}"
                    color = "red" if np.isnan(val) else ("white" if val < threshold else "black")
                    ax.text(j, i, display_val, ha="center", va="center", color=color, fontsize=8)

    if legend_text:
        ax.text(
            0.05,
            0.95,
            legend_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def plot_connectome(
    matrix: np.ndarray,
    method_name: str,
    threshold: float = 0.2,
    directed: bool = False,
    invert_threshold: bool = False,
    legend_text: str = "",
) -> BytesIO:
    """Generate a connectome graph for a connectivity matrix and return PNG buffer."""
    n = matrix.shape[0]
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(range(n))

    if directed:
        for src in range(n):
            for tgt in range(n):
                if src == tgt:
                    continue
                weight = matrix[src, tgt]
                if weight is None or np.isnan(weight):
                    continue
                if (invert_threshold and weight < threshold) or (not invert_threshold and abs(weight) > threshold):
                    graph.add_edge(src, tgt, weight=float(weight))
    else:
        for i in range(n):
            for j in range(i + 1, n):
                weight = matrix[i, j]
                if weight is None or np.isnan(weight):
                    continue
                if abs(weight) > threshold:
                    graph.add_edge(i, j, weight=float(weight))

    pos = nx.circular_layout(graph)
    fig, ax = plt.subplots(figsize=(4, 4))
    if directed:
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="lightblue", node_size=500)
        nx.draw_networkx_labels(graph, pos, ax=ax)
        nx.draw_networkx_edges(graph, pos, ax=ax, arrowstyle="->", arrowsize=10)
    else:
        nx.draw_networkx(graph, pos, ax=ax, node_color="lightblue", node_size=500)

    ax.set_title(f"Connectome: {method_name}")
    if legend_text:
        ax.text(
            0.05,
            0.05,
            legend_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            bbox=dict(facecolor="white", alpha=0.5),
        )
    ax.axis("off")

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf
