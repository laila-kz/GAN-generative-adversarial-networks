import sys
from pathlib import Path

# ensure repo root is on path for test discovery
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DCGAN.DCGAN_script import make_generator, make_discriminator
import tensorflow as tf


def test_generator_discriminator_forward():
    gen = make_generator(100)
    disc = make_discriminator()

    noise = tf.random.normal([4, 1, 1, 100])
    generated = gen(noise, training=False)
    assert generated.shape[0] == 4
    assert generated.shape[-1] == 3

    scores = disc(generated, training=False)
    assert scores.shape[0] == 4


if __name__ == "__main__":
    test_generator_discriminator_forward()
    print("smoke test passed")
