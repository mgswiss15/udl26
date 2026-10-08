"""Datasets shared by several labs."""
import math

import numpy as np
import torch


def make_dataset(name: str, n: int = 2000, seed: int = 0) -> torch.Tensor:
    """Toy 2D dataset, standardised to zero mean and unit variance per coordinate.

    name: 'moons', 'spirals' or 'ring' (8 Gaussian blobs on a circle).
    Returns a float32 tensor of shape (n, 2) with rows in random order.
    """
    rng = np.random.default_rng(seed)
    if name == "moons":
        n1, n2 = n // 2, n - n // 2
        t1, t2 = rng.uniform(0, math.pi, n1), rng.uniform(0, math.pi, n2)
        upper = np.stack([np.cos(t1), np.sin(t1)], axis=1)
        lower = np.stack([1 - np.cos(t2), 0.5 - np.sin(t2)], axis=1)
        x = np.concatenate([upper, lower]) + rng.normal(0, 0.08, (n, 2))
    elif name == "spirals":
        n1, n2 = n // 2, n - n // 2

        def arm(m, sign):
            t = np.sqrt(rng.uniform(0.02, 1.0, m)) * 3 * math.pi
            r = t / (3 * math.pi)
            return sign * np.stack([r * np.cos(t), r * np.sin(t)], axis=1)

        x = np.concatenate([arm(n1, 1.0), arm(n2, -1.0)]) + rng.normal(0, 0.03, (n, 2))
    elif name == "ring":
        k = rng.integers(0, 8, n)
        ang = 2 * math.pi * k / 8
        centers = 2.0 * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        x = centers + rng.normal(0, 0.2, (n, 2))
    else:
        raise ValueError(f"unknown dataset {name!r}")
    x = (x - x.mean(0)) / x.std(0)
    rng.shuffle(x)  # shuffles rows in place
    return torch.tensor(x, dtype=torch.float32)


def train_val_split(x: torch.Tensor, n_train: int):
    """First n_train rows for training, the rest for validation (rows are already shuffled)."""
    return x[:n_train], x[n_train:]


def load_mnist():
    """MNIST digits as float64 tensors in [0, 1], shapes (60000, 28, 28) and (10000, 28, 28).
    Downloads once into ./mnist_data (needs torchvision)."""
    from torchvision.datasets import MNIST

    train, test = MNIST("mnist_data", train=True, download=True), MNIST("mnist_data", train=False, download=True)
    return train.data.double() / 255.0, test.data.double() / 255.0
