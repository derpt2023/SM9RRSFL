"""MNIST loading and client partition helpers."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
import struct
from typing import Iterable
from urllib.request import urlretrieve

import numpy as np


MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


@dataclass(frozen=True)
class ImageDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def download_mnist(data_dir: str | Path, base_url: str = MNIST_URL) -> None:
    """Download MNIST gzip IDX files if they are missing."""

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    for filename in MNIST_FILES.values():
        target = data_path / filename
        if target.exists():
            continue
        urlretrieve(f"{base_url}/{filename}", target)


def load_mnist(
    data_dir: str | Path,
    *,
    download: bool = False,
    train_limit: int | None = None,
    test_limit: int | None = None,
    seed: int = 0,
) -> ImageDataset:
    """Load MNIST from IDX gzip files and return flattened float32 images."""

    data_path = Path(data_dir)
    if download:
        download_mnist(data_path)

    paths = {name: data_path / filename for name, filename in MNIST_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        files = ", ".join(missing)
        raise FileNotFoundError(
            "MNIST files are missing. Re-run with --download or place IDX gzip "
            f"files in {data_path}. Missing: {files}"
        )

    x_train = _read_idx_images(paths["train_images"])
    y_train = _read_idx_labels(paths["train_labels"])
    x_test = _read_idx_images(paths["test_images"])
    y_test = _read_idx_labels(paths["test_labels"])

    rng = np.random.default_rng(seed)
    if train_limit is not None and train_limit < len(y_train):
        idx = rng.choice(len(y_train), size=train_limit, replace=False)
        x_train = x_train[idx]
        y_train = y_train[idx]
    if test_limit is not None and test_limit < len(y_test):
        idx = rng.choice(len(y_test), size=test_limit, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]

    return ImageDataset(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def make_synthetic_mnist_like(
    *,
    train_samples: int = 1000,
    test_samples: int = 300,
    seed: int = 0,
) -> ImageDataset:
    """Create a small MNIST-shaped dataset for smoke tests without downloads."""

    rng = np.random.default_rng(seed)
    prototypes = rng.uniform(0.0, 1.0, size=(10, 28 * 28)).astype(np.float32)
    masks = rng.uniform(0.0, 1.0, size=(10, 28 * 28)).astype(np.float32)
    prototypes = (0.75 * prototypes + 0.25 * (masks > 0.82)).astype(np.float32)

    def sample(count: int) -> tuple[np.ndarray, np.ndarray]:
        labels = rng.integers(0, 10, size=count, dtype=np.int64)
        noise = rng.normal(0.0, 0.18, size=(count, 28 * 28)).astype(np.float32)
        x = np.clip(prototypes[labels] + noise, 0.0, 1.0).astype(np.float32)
        return x, labels

    x_train, y_train = sample(train_samples)
    x_test, y_test = sample(test_samples)
    return ImageDataset(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def partition_clients(
    labels: np.ndarray,
    num_clients: int,
    *,
    strategy: str = "iid",
    dirichlet_alpha: float = 0.5,
    seed: int = 0,
) -> list[np.ndarray]:
    """Partition sample indices among clients using IID or label Dirichlet splits."""

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(labels), dtype=np.int64)

    if strategy == "iid":
        shuffled = rng.permutation(all_indices)
        return [chunk.astype(np.int64) for chunk in np.array_split(shuffled, num_clients)]

    if strategy != "dirichlet":
        raise ValueError("strategy must be 'iid' or 'dirichlet'")
    if dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive")

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for label in np.unique(labels):
        label_indices = all_indices[labels == label]
        rng.shuffle(label_indices)
        proportions = rng.dirichlet(np.full(num_clients, dirichlet_alpha))
        cuts = (np.cumsum(proportions)[:-1] * len(label_indices)).astype(int)
        for client_id, split in enumerate(np.split(label_indices, cuts)):
            client_indices[client_id].extend(int(i) for i in split)

    _repair_empty_clients(client_indices, rng)
    result = []
    for indices in client_indices:
        arr = np.array(indices, dtype=np.int64)
        rng.shuffle(arr)
        result.append(arr)
    return result


def _repair_empty_clients(client_indices: list[list[int]], rng: np.random.Generator) -> None:
    empty = [idx for idx, values in enumerate(client_indices) if not values]
    for empty_idx in empty:
        donor_idx = max(range(len(client_indices)), key=lambda idx: len(client_indices[idx]))
        if len(client_indices[donor_idx]) <= 1:
            break
        take_pos = int(rng.integers(0, len(client_indices[donor_idx])))
        client_indices[empty_idx].append(client_indices[donor_idx].pop(take_pos))


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    images = data.reshape(count, rows * cols).astype(np.float32) / 255.0
    return images


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file")
        labels = np.frombuffer(handle.read(), dtype=np.uint8)
    if len(labels) != count:
        raise ValueError(f"{path} label count mismatch")
    return labels.astype(np.int64)
