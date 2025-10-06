from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


class BaseClassifier(ABC):
    """Abstract base class for simple binary classifiers.

    All subclasses must implement fit and predict. Labels are expected to be in {-1, +1}.
    Inputs X are numpy arrays of shape (n_samples, n_features).
    We use augmented vectors (bias term appended as 1) internally.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Iterable[np.ndarray]:
        """Train the model. Should yield the weight vector after each epoch.

        Parameters
        ----------
        X : np.ndarray
            Training inputs of shape (n_samples, n_features)
        y : np.ndarray
            Labels in {-1, +1} with shape (n_samples,)
        Yields
        ------
        np.ndarray
            Current weight vector of shape (n_features + 1,) including bias as last element.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted labels in {-1, +1} for inputs X."""


def _augment_inputs(X: np.ndarray) -> np.ndarray:
    """Append a column of ones to X to incorporate bias as an extra weight."""
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    return np.hstack([X, ones])


class Perceptron(BaseClassifier):
    def __init__(self, learning_rate: float = 0.1, max_epochs: int = 100):
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.weights: np.ndarray | None = None  # shape (n_features + 1,)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Iterable[np.ndarray]:
        X_aug = _augment_inputs(X)
        n_features_aug = X_aug.shape[1]
        if self.weights is None:
            self.weights = np.zeros(n_features_aug, dtype=float)

        for _ in range(self.max_epochs):
            errors = 0
            for xi, target in zip(X_aug, y):
                activation = np.dot(xi, self.weights)
                prediction = 1 if activation >= 0 else -1
                update = self.learning_rate * (target - prediction)
                if update != 0.0:
                    self.weights += update * xi
                    errors += 1
            # yield weights for animation after epoch
            yield self.weights.copy()
            # Early stopping if perfectly classified
            if errors == 0:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not trained yet.")
        X_aug = _augment_inputs(X)
        activations = X_aug @ self.weights
        return np.where(activations >= 0.0, 1, -1)


def bipolar_sigmoid(net: np.ndarray) -> np.ndarray:
    """Bipolar sigmoid: f(net) = (2 / (1 + exp(-net))) - 1."""
    return 2.0 / (1.0 + np.exp(-net)) - 1.0


def bipolar_sigmoid_derivative_from_output(output: np.ndarray) -> np.ndarray:
    """Derivative using output: f'(net) = 0.5 * (1 - output^2)."""
    return 0.5 * (1.0 - np.square(output))


class DeltaRule(BaseClassifier):
    """Single-layer neuron trained with delta rule using bipolar sigmoid activation."""

    def __init__(self, learning_rate: float = 0.1, max_epochs: int = 200):
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.weights: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Iterable[np.ndarray]:
        X_aug = _augment_inputs(X)
        n_features_aug = X_aug.shape[1]
        if self.weights is None:
            # small random init helps avoid symmetry; scale down for stability
            rng = np.random.default_rng(seed=42)
            self.weights = rng.normal(loc=0.0, scale=0.1, size=n_features_aug)

        for _ in range(self.max_epochs):
            # online updates for interactivity
            for xi, target in zip(X_aug, y):
                net = float(np.dot(xi, self.weights))
                out = float(bipolar_sigmoid(net))
                # error in bipolar target space {-1, +1}
                error = target - out
                derivative = 0.5 * (1.0 - out * out)
                delta = self.learning_rate * error * derivative
                self.weights += delta * xi
            yield self.weights.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not trained yet.")
        X_aug = _augment_inputs(X)
        net = X_aug @ self.weights
        out = bipolar_sigmoid(net)
        return np.where(out >= 0.0, 1, -1)
