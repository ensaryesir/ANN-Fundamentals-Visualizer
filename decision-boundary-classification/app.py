from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import algorithms
import data_manager
import ui_components
import visualizer


def _compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean((y_true == y_pred).astype(float)))


def main() -> None:
    data_manager.initialize_state()

    st.title("Decision Boundary Classification Visualizer")

    settings = ui_components.setup_sidebar()

    st.info(
        "Add points by entering coordinates below. Choose the class in the sidebar, then click 'Add Point'."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        x_val = st.number_input("x1", value=0.0, step=0.1, format="%.3f")
    with col2:
        y_val = st.number_input("x2", value=0.0, step=0.1, format="%.3f")
    with col3:
        if st.button("Add Point", use_container_width=True):
            data_manager.add_point(settings.add_class, (x_val, y_val))
            st.rerun()

    st.subheader("Current Points")
    st.write(st.session_state.points)

    st.subheader("Training")
    placeholder = st.empty()

    if st.button("Train Model", use_container_width=True):
        X_train, y_train, X_test, y_test, train_idx, test_idx = data_manager.prepare_data(settings.test_size)

        if X_train.shape[0] == 0:
            st.warning("Not enough data to train. Please add points.")
            return

        if settings.algorithm == "Perceptron":
            model = algorithms.Perceptron(learning_rate=settings.learning_rate, max_epochs=settings.max_epochs)
        else:
            model = algorithms.DeltaRule(learning_rate=settings.learning_rate, max_epochs=settings.max_epochs)

        fig, _ax = plt.subplots(figsize=(6, 6))

        for current_weights in model.fit(X_train, y_train):
            visualizer.plot_data_and_boundary(fig, st.session_state.points, current_weights, train_idx, test_idx)
            placeholder.pyplot(fig)
            time.sleep(0.1)

        try:
            y_pred_test = model.predict(X_test) if X_test.shape[0] > 0 else np.array([])
            acc = _compute_accuracy(y_test, y_pred_test)
        except Exception:
            acc = 0.0
        st.metric(label="Test accuracy", value=f"{acc * 100:.1f}%")


if __name__ == "__main__":
    main()
