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


def plot_timeseries_panel(
    df,
    title: str,
    *,
    max_points: int = 2000,
    y_domain: tuple[float, float] | None = None,
) -> BytesIO:
    """Рисует несколько рядов в отдельных осях, но с общим диапазоном Y."""
    import pandas as pd

    if df is None or len(df) == 0:
        buf = BytesIO()
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf

    data = df.copy() if hasattr(df, "copy") else pd.DataFrame(df)
    cols = list(data.columns)
    n = len(data)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).round().astype(int)
        data = data.iloc[idx]

    if y_domain is None:
        vals = data.to_numpy(dtype=float)
        vmin = float(np.nanmin(vals)) if np.isfinite(vals).any() else -1.0
        vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
        y_domain = (vmin, vmax)

    rows = len(cols)
    fig_h = max(2.0, 1.2 * rows)
    fig, axes = plt.subplots(rows, 1, figsize=(8, fig_h), sharex=True)
    if rows == 1:
        axes = [axes]

    for ax, c in zip(axes, cols):
        y = np.asarray(data[c], dtype=float)
        ax.plot(y)
        ax.set_ylabel(str(c), fontsize=8)
        ax.set_ylim(y_domain[0], y_domain[1])
        ax.grid(True, alpha=0.2)

    axes[0].set_title(title, fontsize=10)
    axes[-1].set_xlabel("t (index)")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def plot_fft_spectrum(
    series,
    title: str,
    *,
    fs: float = 1.0,
    max_points: int = 4096,
) -> BytesIO:
    """Быстрый спектр (FFT амплитуды по частоте) для ряда."""
    import pandas as pd

    s = pd.to_numeric(series, errors="coerce")
    arr = np.asarray(s.to_numpy(), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 8:
        buf = BytesIO()
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Too short", ha="center", va="center")
        ax.axis("off")
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf

    if n > max_points:
        arr = arr[-max_points:]
        n = int(arr.size)

    fs = float(fs) if fs and np.isfinite(fs) and fs > 0 else 1.0
    dt = 1.0 / fs
    freqs = np.fft.fftfreq(n, d=dt)
    yf = np.fft.fft(arr - float(np.mean(arr)))
    amp = np.abs(yf)
    mask = freqs > 0
    freqs = freqs[mask]
    amp = amp[mask]

    buf = BytesIO()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(freqs, amp)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("frequency")
    ax.set_ylabel("|FFT|")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def plot_window_cube_3d(points: list[dict], title: str) -> BytesIO:
    """3D scatter: window_size × lag × start_pos, цвет/размер по metric."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    buf = BytesIO()
    fig = plt.figure(figsize=(7.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")

    if not points:
        ax.text2D(0.5, 0.5, "No window×lag×position data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf

    ws = np.array([p.get("window_size", np.nan) for p in points], dtype=float)
    lg = np.array([p.get("lag", np.nan) for p in points], dtype=float)
    st = np.array([p.get("start", np.nan) for p in points], dtype=float)
    mt = np.array([p.get("metric", np.nan) for p in points], dtype=float)

    finite = np.isfinite(ws) & np.isfinite(lg) & np.isfinite(st) & np.isfinite(mt)
    ws, lg, st, mt = ws[finite], lg[finite], st[finite], mt[finite]

    if mt.size == 0:
        ax.text2D(0.5, 0.5, "No finite metrics", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf

    m_min, m_max = float(np.min(mt)), float(np.max(mt))
    denom = (m_max - m_min) if m_max > m_min else 1.0
    sizes = 18.0 + 60.0 * (mt - m_min) / denom

    sc = ax.scatter(ws, lg, st, c=mt, s=sizes, cmap="viridis", alpha=0.85)
    fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.08, label="metric")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("window_size")
    ax.set_ylabel("lag")
    ax.set_zlabel("start_pos")
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf
