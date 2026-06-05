"""Backward-compatible MNIST imports.

New dataset loaders live in :mod:`sm9rrsfl.datasets`.
"""

from __future__ import annotations

from .datasets import (
    ImageDataset,
    MNIST_FILES,
    MNIST_URL,
    download_mnist,
    load_mnist,
    make_synthetic_mnist_like,
    partition_clients,
)

__all__ = [
    "ImageDataset",
    "MNIST_FILES",
    "MNIST_URL",
    "download_mnist",
    "load_mnist",
    "make_synthetic_mnist_like",
    "partition_clients",
]
