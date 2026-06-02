"""CPU/GPU placement demo for Keras.

This script converts the notebook into a standalone program that benchmarks a
small CNN on CPU and, if available, GPU. It also demonstrates using multiple
logical GPUs together with a CPU reduction step.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D


LOGGER = logging.getLogger(__name__)
DEFAULT_SEED = 42


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def list_devices() -> tuple[list[str], list[str]]:
    physical_gpus = [device.name for device in tf.config.list_physical_devices("GPU")]
    logical_gpus = [device.name for device in tf.config.list_logical_devices("GPU")]
    LOGGER.info("Physical GPUs: %s", physical_gpus if physical_gpus else "none")
    LOGGER.info("Logical GPUs: %s", logical_gpus if logical_gpus else "none")
    return physical_gpus, logical_gpus


def load_mnist_subset(train_size: int, test_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train[:train_size].astype("float32") / 255.0
    y_train = y_train[:train_size]
    x_test = x_test[:test_size].astype("float32") / 255.0
    y_test = y_test[:test_size]

    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    return x_train, y_train, x_test, y_test


def build_cnn() -> Sequential:
    model = Sequential(
        [
            Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model


def train_on_device(
    device_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
) -> tuple[float, dict[str, float]]:
    keras.backend.clear_session()
    model = build_cnn()

    start_time = time.perf_counter()
    with tf.device(device_name):
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, verbose=0)
    duration = time.perf_counter() - start_time

    metrics = {key: float(values[-1]) for key, values in history.history.items()}
    LOGGER.info("Finished training on %s in %.2f seconds", device_name, duration)
    LOGGER.info("Final metrics on %s: %s", device_name, metrics)
    return duration, metrics


def run_joint_device_demo(logical_gpus: list[str]) -> np.ndarray | None:
    if not logical_gpus:
        LOGGER.info("Skipping joint CPU/GPU demo because no GPUs are available.")
        return None

    tf.debugging.set_log_device_placement(True)
    matrices = []
    for gpu_name in logical_gpus:
        with tf.device(gpu_name):
            matrix_a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            matrix_b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            matrices.append(tf.matmul(matrix_a, matrix_b))

    with tf.device("/CPU:0"):
        matmul_sum = tf.add_n(matrices)

    LOGGER.info("Joint CPU/GPU matmul result:\n%s", matmul_sum.numpy())
    return matmul_sum.numpy()


def save_timing_plot(output_path: Path, cpu_time: float, gpu_time: float | None) -> None:
    labels = ["CPU"]
    durations = [cpu_time]
    if gpu_time is not None:
        labels.append("GPU")
        durations.append(gpu_time)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.bar(labels, durations, color=["#6c757d", "#1f77b4"][: len(labels)])
    ax.set_ylabel("Seconds")
    ax.set_title("CNN training time by device")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_experiment(output_dir: Path, train_size: int, test_size: int, epochs: int, mode: str, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    physical_gpus, logical_gpus = list_devices()
    x_train, y_train, x_test, y_test = load_mnist_subset(train_size, test_size)

    cpu_time, cpu_metrics = train_on_device("/CPU:0", x_train, y_train, x_test, y_test, epochs)
    gpu_time = None
    gpu_metrics = None

    if mode in {"gpu", "all", "joint"} and physical_gpus:
        gpu_time, gpu_metrics = train_on_device(logical_gpus[0], x_train, y_train, x_test, y_test, epochs)
    elif mode in {"gpu", "all", "joint"}:
        LOGGER.warning("No GPU available, skipping GPU benchmark")

    if mode in {"joint", "all"}:
        run_joint_device_demo(logical_gpus)

    save_timing_plot(output_dir / "cpu_gpu_timings.png", cpu_time, gpu_time)

    summary_path = output_dir / "timings.txt"
    summary_lines = [
        f"CPU: {cpu_time:.4f} seconds, metrics={cpu_metrics}",
    ]
    if gpu_time is not None and gpu_metrics is not None:
        summary_lines.append(f"GPU: {gpu_time:.4f} seconds, metrics={gpu_metrics}")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    LOGGER.info("Saved timing summary to %s", summary_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a small CNN on CPU and GPU.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--train-size", type=int, default=5_000)
    parser.add_argument("--test-size", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--mode", choices=["cpu", "gpu", "joint", "all"], default="all")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    configure_logging()
    args = build_parser().parse_args()
    set_seed(args.seed)

    try:
        run_experiment(args.output_dir, args.train_size, args.test_size, args.epochs, args.mode, args.seed)
    except Exception:
        LOGGER.exception("GPU/CPU demo failed")
        raise


if __name__ == "__main__":
    main()