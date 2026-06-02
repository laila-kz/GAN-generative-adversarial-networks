# CPU vs GPU Summary

## Overview

**CPU (Central Processing Unit)** and **GPU (Graphics Processing Unit)** are both essential processors in a computer, but they excel at different types of tasks.

## Key Differences

| Feature | CPU | GPU |
|---------|-----|-----|
| **Design** | Few powerful cores (2-32) | Thousands of smaller, efficient cores |
| **Strengths** | Sequential processing, complex logic | Parallel processing, simple math operations |
| **Best for** | General-purpose computing, OS tasks, sequential code | Matrix operations, image processing, deep learning |
| **Deep Learning** | Slower for training large models | Significantly faster for training and inference |

## Why GPU for Deep Learning?

- **Parallel Computing**: Neural networks involve many independent matrix multiplications that can run simultaneously
- **Efficiency**: Tasks like convolution operations are naturally parallelizable
- **Time Savings**: Training epochs complete much faster, especially with large datasets

## TensorFlow/Keras Control

| Action | Code |
|--------|------|
| Force CPU | `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'` |
| Run specific code on CPU | `with tf.device('/CPU:0'):` |
| Run specific code on GPU | `with tf.device('/GPU:0'):` |
| Check GPU availability | `tf.config.list_physical_devices('GPU')` |

## Best Practices

- Use **GPU** for model training, especially with CNNs and large datasets
- Use **CPU** for data preprocessing, small models, or debugging
- Combine both: preprocessing on CPU, training on GPU