from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import streamlit as st


def initialize_state() -> None:
    """Initialize session state containers for storing class-labeled points.

    points structure:
        {
            'Class 1': List[Tuple[float, float]],
            'Class 2': List[Tuple[float, float]],
        }
    """
    if "points" not in st.session_state:
        st.session_state.points = {"Class 1": [], "Class 2": []}


def add_point(klass: str, coords: Tuple[float, float]) -> None:
    if "points" not in st.session_state:
        initialize_state()
    if klass not in st.session_state.points:
        st.session_state.points[klass] = []
    st.session_state.points[klass].append((float(coords[0]), float(coords[1])))


def clear_points() -> None:
    st.session_state.points = {"Class 1": [], "Class 2": []}


def _collect_all_points() -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]], np.ndarray]:
    """Return X, y, flat_points, class_ids for all points in session state.

    Labels are {-1, +1} corresponding to Class 1 -> -1, Class 2 -> +1.
    class_ids holds -1 or +1 per point in the same order as flat_points.
    """
    pts1 = st.session_state.points.get("Class 1", [])
    pts2 = st.session_state.points.get("Class 2", [])
    flat_points: List[Tuple[float, float]] = list(pts1) + list(pts2)
    if len(flat_points) == 0:
        return np.empty((0, 2)), np.empty((0,)), flat_points, np.empty((0,))
    X = np.array(flat_points, dtype=float)
    y = np.array([-1] * len(pts1) + [1] * len(pts2), dtype=float)
    return X, y, flat_points, y.copy()


def prepare_data(test_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare shuffled train/test splits and return indices for visualization.

    Returns
    -------
    X_train, y_train, X_test, y_test, train_indices, test_indices
    Indices refer to positions in the flat list of all points.
    """
    X, y, _flat, _class_ids = _collect_all_points()
    n_samples = X.shape[0]
    if n_samples == 0:
        empty = np.empty((0, 2))
        return empty, np.empty((0,)), empty, np.empty((0,)), np.empty((0,), dtype=int), np.empty((0,), dtype=int)

    rng = np.random.default_rng(seed=0)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_test = max(0, min(n_samples, int(round(test_size * n_samples))))
    n_train = n_samples - n_test
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test, train_idx, test_idx
