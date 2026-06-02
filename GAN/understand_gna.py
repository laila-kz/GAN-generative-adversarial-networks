# =========================
# GAN Lab - Python Script
# =========================

import warnings
warnings.simplefilter('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm
import argparse
from pathlib import Path
import logging

tf.compat.v1.enable_eager_execution()

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# =========================
# Helper function
# =========================
def plot_distribution(real_data, generated_data, discriminator=None, density=True):
    plt.hist(real_data.numpy(), 100, density=density, alpha=0.75, label='real data')
    plt.hist(generated_data.numpy(), 100, density=density, alpha=0.75, label='generated data')

    if discriminator:
        max_ = int(max(real_data.numpy().max(), generated_data.numpy().max()))
        min_ = int(min(real_data.numpy().min(), generated_data.numpy().min()))
        x = np.linspace(min_, max_, 1000).reshape(-1, 1)

        plt.plot(x, tf.math.sigmoid(discriminator(x, training=False).numpy()), label='discriminator')
        plt.plot(x, 0.5 * np.ones(x.shape), label='0.5')

    plt.legend()
    plt.show()


# =========================
# Data generation, models, training and orchestration
# =========================


def _run_script() -> None:
    # data
    X = tf.random.normal((5000, 1), mean=10, stddev=1.0)
    Z = tf.random.normal((5000, 1), mean=0, stddev=2)

    plot_distribution(X, Z)

    # models
    def make_generator_model():
        model = tf.keras.Sequential()
        model.add(layers.Dense(1))
        return model

    def make_discriminator_model():
        model = tf.keras.Sequential()
        model.add(layers.Dense(1))
        return model

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    Xhat = generator(Z)
    plot_distribution(X, Xhat, discriminator)

    # accuracy and losses
    def get_accuracy(X, Xhat):
        py_x = tf.math.sigmoid(discriminator(X))
        acc_real = np.mean(py_x)

        py_x = tf.math.sigmoid(discriminator(Xhat))
        acc_fake = np.mean(py_x)

        return (acc_real + acc_fake) / 2

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        return 0.5 * (real_loss + fake_loss)

    # optimizers and parameters
    generator_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    discriminator_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)

    epochs = 20
    BATCH_SIZE = 5000
    noise_dim = 1
    epsilon = 100

    tf.random.set_seed(0)
    discriminator = make_discriminator_model()
    generator = make_generator_model()

    tf.config.run_functions_eagerly(True)

    gen_loss_epoch = []
    disc_loss_epoch = []

    # initial plot
    plot_distribution(real_data=X, generated_data=Xhat, discriminator=discriminator)
    print("epoch", 0)

    for epoch in tqdm(range(epochs)):
        idx = np.random.randint(0, X.shape[0], BATCH_SIZE)
        x = tf.gather(X, idx)

        z = tf.random.normal([BATCH_SIZE, noise_dim], mean=0, stddev=1)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            xhat = generator(z, training=True)
            real_output = discriminator(x, training=True)
            fake_output = discriminator(xhat, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        gen_loss_epoch.append(gen_loss.numpy())
        disc_loss_epoch.append(disc_loss.numpy())

        current_acc = get_accuracy(x, xhat)

        if abs(0.5 - current_acc) < epsilon:
            epsilon = abs(0.5 - current_acc)

            # safe save into local folder
            Path("outputs").mkdir(parents=True, exist_ok=True)
            generator.save(str(Path("outputs") / "generator"))
            discriminator.save(str(Path("outputs") / "discriminator"))

            print("Accuracy:", current_acc)
            plot_distribution(real_data=X, generated_data=xhat, discriminator=discriminator)
            print("epoch", epoch)

    # load best model if present
    try:
        generator = models.load_model(str(Path("outputs") / "generator"))
        discriminator = models.load_model(str(Path("outputs") / "discriminator"))
    except Exception:
        LOGGER.warning("No saved models found in outputs/")

    z = tf.random.normal((5000, 1))
    xhat = generator(z)
    plot_distribution(X, xhat, discriminator)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small illustrative GAN demo (1D toy example)")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    tf.random.set_seed(args.seed)
    _run_script()


if __name__ == "__main__":
    main()