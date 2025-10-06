from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _compute_decision_boundary_line(weights: np.ndarray, x_limits: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Given weights [w1, w2, b], compute y for a line over x_limits for w1*x + w2*y + b = 0."""
    if weights is None or weights.shape[0] != 3:
        x = np.array(x_limits)
        return x, np.full_like(x, np.nan, dtype=float)
    w1, w2, b = float(weights[0]), float(weights[1]), float(weights[2])
    x_vals = np.linspace(x_limits[0], x_limits[1], num=100)
    if abs(w2) < 1e-12:
        y_vals = np.full_like(x_vals, np.nan)
    else:
        y_vals = -(w1 * x_vals + b) / w2
    return x_vals, y_vals


essential_nan = np.nan  # to avoid import-unused lint if needed


def plot_data_and_boundary(fig: Figure, points: Dict[str, List[Tuple[float, float]]], weights: np.ndarray | None, train_indices: np.ndarray, test_indices: np.ndarray) -> None:
    fig.clf()
    ax: Axes = fig.add_subplot(111)

    pts1 = points.get("Class 1", [])
    pts2 = points.get("Class 2", [])
    flat: List[Tuple[float, float]] = list(pts1) + list(pts2)

    if len(flat) > 0:
        X = np.array(flat, dtype=float)
        x_min, x_max = float(np.min(X[:, 0])) - 1.0, float(np.max(X[:, 0])) + 1.0
        y_min, y_max = float(np.min(X[:, 1])) - 1.0, float(np.max(X[:, 1])) + 1.0
    else:
        x_min, x_max, y_min, y_max = -5.0, 5.0, -5.0, 5.0

    if len(flat) > 0:
        X = np.array(flat, dtype=float)
        labels = np.array([-1] * len(pts1) + [1] * len(pts2), dtype=int)

        def scatter_subset(subset_idx: np.ndarray, edge: str, face: str, label_prefix: str) -> None:
            if subset_idx.size == 0:
                return
            subset = X[subset_idx]
            subset_labels = labels[subset_idx]
            class1 = subset[subset_labels == -1]
            class2 = subset[subset_labels == 1]
            if class1.size:
                ax.scatter(class1[:, 0], class1[:, 1], c=face, edgecolors=edge, marker="o", label=f"{label_prefix} Class 1")
            if class2.size:
                ax.scatter(class2[:, 0], class2[:, 1], c=face, edgecolors=edge, marker="s", label=f"{label_prefix} Class 2")

        scatter_subset(train_indices.astype(int), edge="black", face="#1f77b4", label_prefix="Train")
        scatter_subset(test_indices.astype(int), edge="#ff7f0e", face="none", label_prefix="Test")

    x_vals, y_vals = _compute_decision_boundary_line(weights if weights is not None else np.array([np.nan, np.nan, np.nan]), (x_min, x_max))
    ax.plot(x_vals, y_vals, "k-", linewidth=2, label="Decision boundary")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Data and Decision Boundary")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", linewidth=0.5)
