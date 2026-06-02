"""Universal Approximation Theorem demonstrations.

This script consolidates the notebook examples into a single runnable module
that visualizes activation functions, approximates simple functions with MLPs,
and compares shallow and deep networks.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor


LOGGER = logging.getLogger(__name__)
DEFAULT_SEED = 42


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@dataclass(frozen=True)
class SingleHiddenLayerNN:
    input_size: int
    hidden_size: int
    output_size: int
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        object.__setattr__(self, "W1", rng.random((self.input_size, self.hidden_size)) * 0.01)
        object.__setattr__(self, "b1", np.zeros((1, self.hidden_size)))
        object.__setattr__(self, "W2", rng.random((self.hidden_size, self.output_size)) * 0.01)
        object.__setattr__(self, "b2", np.zeros((1, self.output_size)))

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        hidden_linear = np.dot(inputs, self.W1) + self.b1
        hidden_activation = self.sigmoid(hidden_linear)
        output_linear = np.dot(hidden_activation, self.W2) + self.b2
        return self.sigmoid(output_linear)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def save_activation_plot(output_path: Path) -> None:
    x = np.linspace(-5, 5, 100)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(x, sigmoid(x))
    axes[0].set_title("Sigmoid")

    axes[1].plot(x, tanh(x))
    axes[1].set_title("Tanh")

    axes[2].plot(x, relu(x))
    axes[2].set_title("ReLU")

    fig.suptitle("Common activation functions")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_sine_approximation(output_path: Path, seed: int) -> None:
    x = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(x).ravel()

    model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=seed)
    model.fit(x, y)
    y_pred = model.predict(x)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.scatter(x, y, color="steelblue", label="True function", s=14)
    ax.plot(x, y_pred, color="crimson", label="NN approximation")
    ax.set_title("Neural network approximation of sin(x)")
    ax.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_learning_curves(output_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, 1000).reshape(-1, 1)
    y = np.sin(x).ravel() + 0.1 * rng.standard_normal(1000)

    shallow = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=seed)
    deep = MLPRegressor(hidden_layer_sizes=(10, 10, 10), max_iter=1000, random_state=seed)

    train_sizes, train_scores_shallow, test_scores_shallow = learning_curve(shallow, x, y, cv=5, n_jobs=None)
    _, train_scores_deep, test_scores_deep = learning_curve(deep, x, y, cv=5, n_jobs=None)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(train_sizes, np.mean(train_scores_shallow, axis=1), label="Shallow NN train", marker="o")
    ax.plot(train_sizes, np.mean(test_scores_shallow, axis=1), label="Shallow NN test", marker="o")
    ax.plot(train_sizes, np.mean(train_scores_deep, axis=1), label="Deep NN train", marker="o")
    ax.plot(train_sizes, np.mean(test_scores_deep, axis=1), label="Deep NN test", marker="o")
    ax.set_title("Learning curves: shallow vs deep neural network")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_regularization_comparison(output_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(x).ravel() + 0.3 * rng.standard_normal(100)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    no_reg = MLPRegressor(hidden_layer_sizes=(100,), alpha=0.0, max_iter=5000, random_state=seed)
    l2_reg = MLPRegressor(hidden_layer_sizes=(100,), alpha=0.01, max_iter=5000, random_state=seed)

    no_reg.fit(x_train, y_train)
    l2_reg.fit(x_train, y_train)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.scatter(x, y, color="steelblue", label="Data", s=14)
    ax.plot(x, no_reg.predict(x), color="crimson", label="No regularization")
    ax.plot(x, l2_reg.predict(x), color="seagreen", label="L2 regularization")
    ax.set_title("Effect of regularization on a neural network")
    ax.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

def save_grid_search_summary(output_path: Path, seed: int) -> None:
    digits = load_digits()
    x, y = digits.data, digits.target

    param_grid = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["constant", "adaptive"],
    }

    model = MLPClassifier(max_iter=1000, random_state=seed)
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        grid_search.fit(x, y)

    output_path.write_text(
        f"Best parameters: {grid_search.best_params_}\nBest score: {grid_search.best_score_:.4f}\n",
        encoding="utf-8",
    )


def run_demo(output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    nn = SingleHiddenLayerNN(input_size=2, hidden_size=5, output_size=1, seed=seed)
    sample_output = nn.forward(np.array([[1.0, 2.0]]))
    LOGGER.info("Single hidden layer demo output: %s", np.array2string(sample_output, precision=6))

    save_activation_plot(output_dir / "activation_functions.png")
    save_sine_approximation(output_dir / "sine_approximation.png", seed)
    save_learning_curves(output_dir / "learning_curves.png", seed)
    save_regularization_comparison(output_dir / "regularization_comparison.png", seed)
    save_grid_search_summary(output_dir / "grid_search_summary.txt", seed)

    LOGGER.info("Saved Universal Approximation theorem outputs to %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Universal Approximation Theorem demonstrations.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    np.random.seed(args.seed)

    try:
        run_demo(args.output_dir, args.seed)
    except Exception:
        LOGGER.exception("Universal Approximation demo failed")
        raise


if __name__ == "__main__":
    main()