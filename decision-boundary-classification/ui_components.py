from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import streamlit as st

import data_manager as data_manager


@dataclass
class UserSettings:
    algorithm: Literal["Perceptron", "DeltaRule"]
    test_size: float
    learning_rate: float
    max_epochs: int
    add_class: Literal["Class 1", "Class 2"]


def setup_sidebar() -> UserSettings:
    with st.sidebar:
        st.header("Controls")
        algorithm = st.selectbox("Algorithm", ["Perceptron", "DeltaRule"], index=0)
        test_size = st.slider("Test size", min_value=0.0, max_value=0.9, value=0.2, step=0.05)
        learning_rate = st.number_input("Learning rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01)
        max_epochs = st.number_input("Max epochs", min_value=1, max_value=1000, value=100, step=10)
        add_class = st.selectbox("Add point to", ["Class 1", "Class 2"], index=0)

        if st.button("Clear Points", use_container_width=True):
            data_manager.clear_points()
            st.rerun()

    return UserSettings(
        algorithm=algorithm,
        test_size=float(test_size),
        learning_rate=float(learning_rate),
        max_epochs=int(max_epochs),
        add_class=add_class,
    )
