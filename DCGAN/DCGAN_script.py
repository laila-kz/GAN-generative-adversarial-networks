"""Deep Convolutional GAN demo.

This script refactors the notebook into a standalone workflow with optional
dataset download, brief training, pretrained model loading, and saved output
figures for latent-space exploration.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


try:
    import skillsnetwork
except ImportError:  # pragma: no cover - optional dependency
    skillsnetwork = None


LOGGER = logging.getLogger(__name__)
DEFAULT_SEED = 42
DEFAULT_LATENT_DIM = 100
DEFAULT_IMAGE_SIZE = (64, 64)
DEFAULT_BATCH_SIZE = 128
DEFAULT_TRAIN_EPOCHS = 1
DATASET_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module6/cartoon_20000.zip"
PRETRAINED_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module6/generator.tar.gz"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def prepare_archive(url: str, overwrite: bool = False) -> None:
    if skillsnetwork is None:
        raise RuntimeError("skillsnetwork is not installed, cannot download remote assets")
    LOGGER.info("Downloading %s", url)
    skillsnetwork.prepare(url, overwrite=overwrite)


def plot_image_batch(images: np.ndarray | tf.Tensor, output_path: Path, title: str | None = None) -> None:
    array = images.numpy() if hasattr(images, "numpy") else np.asarray(images)
    count = min(5, len(array))

    fig, axes = plt.subplots(1, count, figsize=(4 * count, 4), constrained_layout=True)
    if count == 1:
        axes = [axes]

    for axis, image in zip(axes, array[:count]):
        image = np.asarray(image)
        min_value = image.min()
        max_value = image.max()
        if max_value > min_value:
            image = np.uint8(255 * (image - min_value) / (max_value - min_value))
        else:
            image = np.uint8(np.clip(image, 0, 255))
        axis.imshow(image)
        axis.axis("off")

    if title:
        fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_dataset(data_dir: Path, batch_size: int, image_size: tuple[int, int]) -> tf.data.Dataset:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=str(data_dir),
        image_size=image_size,
        batch_size=batch_size,
        label_mode=None,
    )

    normalization_layer = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)
    return dataset.map(lambda x: normalization_layer(x), num_parallel_calls=tf.data.AUTOTUNE)


def make_generator(latent_dim: int = DEFAULT_LATENT_DIM) -> tf.keras.Model:
    model = tf.keras.Sequential(name="generator")
    model.add(layers.Input(shape=(1, 1, latent_dim), name="input_layer"))
    model.add(
        layers.Conv2DTranspose(
            64 * 8,
            kernel_size=4,
            strides=4,
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            use_bias=False,
            name="conv_transpose_1",
        )
    )
    model.add(layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=True, scale=True, name="bn_1"))
    model.add(layers.ReLU(name="relu_1"))

    model.add(
        layers.Conv2DTranspose(
            64 * 4,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            use_bias=False,
            name="conv_transpose_2",
        )
    )
    model.add(layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=True, scale=True, name="bn_2"))
    model.add(layers.ReLU(name="relu_2"))

    model.add(
        layers.Conv2DTranspose(
            64 * 2,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            use_bias=False,
            name="conv_transpose_3",
        )
    )
    model.add(layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=True, scale=True, name="bn_3"))
    model.add(layers.ReLU(name="relu_3"))

    model.add(
        layers.Conv2DTranspose(
            64,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            use_bias=False,
            name="conv_transpose_4",
        )
    )
    model.add(layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=True, scale=True, name="bn_4"))
    model.add(layers.ReLU(name="relu_4"))

    model.add(
        layers.Conv2DTranspose(
            3,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            use_bias=False,
            activation="tanh",
            name="conv_transpose_5",
        )
    )
    return model


def make_discriminator() -> tf.keras.Model:
    model = tf.keras.Sequential(name="discriminator")
    model.add(layers.Input(shape=(64, 64, 3), name="input_layer"))
    model.add(
        layers.Conv2D(
            64,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            use_bias=False,
            name="conv_1",
        )
    )
    model.add(layers.LeakyReLU(negative_slope=0.2, name="leaky_relu_1"))

    model.add(
        layers.Conv2D(
            64 * 2,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            use_bias=False,
            name="conv_2",
        )
    )
    model.add(layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=True, scale=True, name="bn_1"))
    model.add(layers.LeakyReLU(negative_slope=0.2, name="leaky_relu_2"))

    model.add(
        layers.Conv2D(
            64 * 4,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            use_bias=False,
            name="conv_3",
        )
    )
    model.add(layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=True, scale=True, name="bn_2"))
    model.add(layers.LeakyReLU(negative_slope=0.2, name="leaky_relu_3"))

    model.add(
        layers.Conv2D(
            64 * 8,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            use_bias=False,
            name="conv_4",
        )
    )
    model.add(layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=True, scale=True, name="bn_3"))
    model.add(layers.LeakyReLU(negative_slope=0.2, name="leaky_relu_4"))

    model.add(
        layers.Conv2D(
            1,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            use_bias=False,
            activation="sigmoid",
            name="conv_5",
        )
    )
    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return 0.5 * (real_loss + fake_loss)


def train_step(
    real_images: tf.Tensor,
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    generator_optimizer: tf.keras.optimizers.Optimizer,
    discriminator_optimizer: tf.keras.optimizers.Optimizer,
    latent_dim: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([batch_size, 1, 1, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train_gan(
    dataset: tf.data.Dataset,
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    epochs: int,
    latent_dim: int,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)

    for epoch in range(epochs):
        gen_losses: list[float] = []
        disc_losses: list[float] = []
        for batch in dataset:
            gen_loss, disc_loss = train_step(
                batch,
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
                latent_dim,
            )
            gen_losses.append(float(gen_loss.numpy()))
            disc_losses.append(float(disc_loss.numpy()))

        LOGGER.info(
            "Epoch %s/%s - generator loss %.4f - discriminator loss %.4f",
            epoch + 1,
            epochs,
            float(np.mean(gen_losses)),
            float(np.mean(disc_losses)),
        )

    return generator, discriminator


def load_pretrained_generator(model_dir: Path) -> tf.keras.Model | None:
    if model_dir.exists():
        LOGGER.info("Loading pretrained generator from %s", model_dir)
        return tf.keras.models.load_model(model_dir)
    return None


def save_generator_samples(generator: tf.keras.Model, output_dir: Path, latent_dim: int, prefix: str) -> None:
    noise = tf.random.normal([200, 1, 1, latent_dim])
    generated = generator(noise, training=False)
    plot_image_batch(generated, output_dir / f"{prefix}_generated_samples.png", title=f"{prefix} samples")


def explore_latent_variables(generator: tf.keras.Model, latent_dim: int, output_dir: Path) -> None:
    for value in [1.0, 0.8, 0.6, 0.4]:
        generated = generator(value * tf.ones([1, 1, 1, latent_dim]), training=False)
        plot_image_batch(generated, output_dir / f"latent_constant_{value:.1f}.png", title=f"z = {value:.1f}")

    z = np.ones((1, 1, 1, latent_dim), dtype=np.float32)
    for n in range(5):
        z[0, 0, 0, : 20 * n] = -0.5 * n
        generated = generator(z, training=False)
        plot_image_batch(generated, output_dir / f"latent_segment_{n}.png", title=f"z[0:20*{n}] = {-0.5 * n:.1f}")

    for n in range(10):
        z = np.random.normal(0, 1, (1, 1, 1, latent_dim)).astype(np.float32)
        z[0, 0, 0, :35] = -1.0
        generated = generator(z, training=False)
        plot_image_batch(generated, output_dir / f"latent_prefix_{n}.png", title="z[0:35] = -1")


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = make_generator(args.latent_dim)
    discriminator = make_discriminator()

    data_dir = args.data_dir
    if args.download_data:
        try:
            prepare_archive(DATASET_URL, overwrite=args.overwrite_download)
        except Exception:
            LOGGER.exception("Dataset download failed; continuing with existing files if available")

    dataset = None
    if data_dir.exists():
        dataset = build_dataset(data_dir, args.batch_size, DEFAULT_IMAGE_SIZE)
        first_batch = next(iter(dataset.take(1)))
        plot_image_batch(first_batch, output_dir / "training_batch.png", title="Training batch")
    else:
        LOGGER.warning("Dataset directory %s not found; skipping GAN training", data_dir)

    if dataset is not None and args.train_epochs > 0:
        generator, discriminator = train_gan(dataset, generator, discriminator, args.train_epochs, args.latent_dim)

    pretrained_model = None
    if args.download_pretrained:
        try:
            prepare_archive(PRETRAINED_URL, overwrite=args.overwrite_download)
        except Exception:
            LOGGER.exception("Pretrained generator download failed")

    pretrained_model = load_pretrained_generator(args.pretrained_model_dir)
    if pretrained_model is not None:
        generator = pretrained_model

    save_generator_samples(generator, output_dir, args.latent_dim, prefix="generator")
    explore_latent_variables(generator, args.latent_dim, output_dir)

    if args.save_models:
        generator.save(output_dir / "generator_model")
        discriminator.save(output_dir / "discriminator_model")

    LOGGER.info("GAN outputs saved to %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DCGAN demo: train or load a generator and save sample outputs.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "images")
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--train-epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--download-pretrained", action="store_true")
    parser.add_argument("--pretrained-model-dir", type=Path, default=Path(__file__).resolve().parent / "pretrained")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--overwrite-download", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    set_seed(args.seed)

    try:
        run_experiment(args)
    except Exception:
        LOGGER.exception("DCGAN demo failed")
        raise


if __name__ == "__main__":
    main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DCGAN notebook as a script.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "cartoon_20000")
    parser.add_argument("--pretrained-model-dir", type=Path, default=Path(__file__).resolve().parent / "generator")
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--train-epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--download-pretrained", action="store_true")
    parser.add_argument("--overwrite-download", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    return parser


def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    configure_logging()
    args = build_parser().parse_args()
    set_seed(args.seed)

    try:
        run_experiment(args)
    except Exception:
        LOGGER.exception("DCGAN demo failed")
        raise


if __name__ == "__main__":
    main()