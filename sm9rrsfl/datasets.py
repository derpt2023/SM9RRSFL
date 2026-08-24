"""Image dataset loading and client partition helpers."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
import pickle
import struct
import tarfile
from urllib.request import urlretrieve

import numpy as np


MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR10_DIR = "cifar-10-batches-py"
CIFAR10_CHANNEL_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
CIFAR10_CHANNEL_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)


@dataclass(frozen=True)
class ImageDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    name: str = "image"
    input_shape: tuple[int, int, int] | None = None
    num_classes: int = 10
    # Training-only auxiliary samples used by the targeted attacker.  Keeping
    # them separate from ``x_test`` prevents the attack optimizer from seeing
    # the official evaluation split.  Legacy callers may omit both fields.
    x_attack: np.ndarray | None = None
    y_attack: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.x_train.ndim != 4 or self.x_test.ndim != 4:
            raise ValueError("image tensors must have shape [samples, channels, height, width]")
        inferred = tuple(int(dim) for dim in self.x_train.shape[1:])
        if self.input_shape is None:
            object.__setattr__(self, "input_shape", inferred)
        elif tuple(self.input_shape) != inferred:
            raise ValueError(f"input_shape {self.input_shape} does not match x_train shape {inferred}")
        if (self.x_attack is None) != (self.y_attack is None):
            raise ValueError("x_attack and y_attack must either both be set or both be omitted")
        if self.x_attack is not None:
            if self.x_attack.ndim != 4:
                raise ValueError(
                    "attack auxiliary images must have shape "
                    "[samples, channels, height, width]"
                )
            if tuple(int(dim) for dim in self.x_attack.shape[1:]) != inferred:
                raise ValueError("x_attack shape must match x_train shape")
            if len(self.x_attack) != len(self.y_attack):
                raise ValueError("x_attack and y_attack must have the same length")


@dataclass(frozen=True)
class TrainingThreeWaySplit:
    """Deterministic training/calibration/attack split and its source indices."""

    main_dataset: ImageDataset
    calibration_dataset: ImageDataset
    train_indices: np.ndarray
    calibration_indices: np.ndarray
    attack_indices: np.ndarray


def stratified_training_three_way_split(
    dataset: ImageDataset,
    *,
    seed: int,
    train_fraction: float = 0.90,
    calibration_fraction: float = 0.05,
    attack_fraction: float = 0.05,
) -> TrainingThreeWaySplit:
    """Split only the original training set into disjoint stratified subsets.

    The returned main run keeps the official test split for evaluation.  The
    calibration run instead evaluates on its training-derived holdout.  Both
    use the same disjoint, training-derived attack auxiliary subset.
    """

    fractions = (train_fraction, calibration_fraction, attack_fraction)
    if any(not np.isfinite(value) or value <= 0.0 for value in fractions):
        raise ValueError("three-way split fractions must be finite and positive")
    if not np.isclose(sum(fractions), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("three-way split fractions must sum to 1")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    calibration_parts: list[np.ndarray] = []
    attack_parts: list[np.ndarray] = []
    labels = np.asarray(dataset.y_train)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 3:
            raise ValueError(
                f"class {int(label)} needs at least three training samples "
                "for train/calibration/attack splitting"
            )
        shuffled = np.asarray(rng.permutation(indices), dtype=np.int64)
        calibration_count = max(
            1,
            int(round(len(shuffled) * calibration_fraction)),
        )
        attack_count = max(1, int(round(len(shuffled) * attack_fraction)))
        # Always retain at least one federated-training sample for the class.
        while calibration_count + attack_count >= len(shuffled):
            if calibration_count >= attack_count and calibration_count > 1:
                calibration_count -= 1
            elif attack_count > 1:
                attack_count -= 1
            else:
                raise ValueError(
                    f"class {int(label)} is too small for the requested split"
                )
        calibration_parts.append(shuffled[:calibration_count])
        attack_parts.append(
            shuffled[calibration_count : calibration_count + attack_count]
        )
        train_parts.append(shuffled[calibration_count + attack_count :])

    train_indices = np.asarray(
        rng.permutation(np.concatenate(train_parts)),
        dtype=np.int64,
    )
    calibration_indices = np.asarray(
        rng.permutation(np.concatenate(calibration_parts)),
        dtype=np.int64,
    )
    attack_indices = np.asarray(
        rng.permutation(np.concatenate(attack_parts)),
        dtype=np.int64,
    )
    train_x = dataset.x_train[train_indices].copy()
    train_y = dataset.y_train[train_indices].copy()
    calibration_x = dataset.x_train[calibration_indices].copy()
    calibration_y = dataset.y_train[calibration_indices].copy()
    attack_x = dataset.x_train[attack_indices].copy()
    attack_y = dataset.y_train[attack_indices].copy()

    main_dataset = ImageDataset(
        x_train=train_x,
        y_train=train_y,
        x_test=dataset.x_test,
        y_test=dataset.y_test,
        name=dataset.name,
        input_shape=dataset.input_shape,
        num_classes=dataset.num_classes,
        x_attack=attack_x,
        y_attack=attack_y,
    )
    calibration_dataset = ImageDataset(
        x_train=train_x,
        y_train=train_y,
        x_test=calibration_x,
        y_test=calibration_y,
        name=dataset.name,
        input_shape=dataset.input_shape,
        num_classes=dataset.num_classes,
        x_attack=attack_x,
        y_attack=attack_y,
    )
    return TrainingThreeWaySplit(
        main_dataset=main_dataset,
        calibration_dataset=calibration_dataset,
        train_indices=train_indices,
        calibration_indices=calibration_indices,
        attack_indices=attack_indices,
    )


def default_data_dir(dataset: str) -> Path:
    if dataset == "cifar10":
        return Path("data/cifar10")
    return Path(f"data/{dataset}")


def load_image_dataset(
    dataset: str,
    data_dir: str | Path | None = None,
    *,
    download: bool = False,
    train_limit: int | None = None,
    test_limit: int | None = None,
    seed: int = 0,
) -> ImageDataset:
    """Load a named image dataset for experiments."""

    if dataset == "synthetic":
        return make_synthetic_mnist_like(
            train_samples=train_limit or 1000,
            test_samples=test_limit or 300,
            seed=seed,
        )
    path = Path(data_dir) if data_dir is not None else default_data_dir(dataset)
    if dataset == "mnist":
        return load_mnist(
            path,
            download=download,
            train_limit=train_limit,
            test_limit=test_limit,
            seed=seed,
        )
    if dataset == "cifar10":
        return load_cifar10(
            path,
            download=download,
            train_limit=train_limit,
            test_limit=test_limit,
            seed=seed,
        )
    raise ValueError("dataset must be one of: mnist, cifar10, synthetic")


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
    """Load MNIST from IDX gzip files and return NCHW float32 images."""

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

    x_train, y_train = _limit_samples(x_train, y_train, train_limit, seed)
    x_test, y_test = _limit_samples(x_test, y_test, test_limit, seed + 1)

    return ImageDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        name="mnist",
        input_shape=(1, 28, 28),
        num_classes=10,
    )


def download_cifar10(data_dir: str | Path, url: str = CIFAR10_URL) -> None:
    """Download and extract CIFAR-10 python batches if they are missing."""

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    extracted = data_path / CIFAR10_DIR
    if extracted.exists():
        return

    archive = data_path / CIFAR10_ARCHIVE
    if not archive.exists():
        urlretrieve(url, archive)

    with tarfile.open(archive, "r:gz") as handle:
        _safe_extract(handle, data_path)


def load_cifar10(
    data_dir: str | Path,
    *,
    download: bool = False,
    train_limit: int | None = None,
    test_limit: int | None = None,
    seed: int = 0,
) -> ImageDataset:
    """Load CIFAR-10 python batches and return NCHW float32 images."""

    data_path = Path(data_dir)
    if download:
        download_cifar10(data_path)

    batch_dir = data_path / CIFAR10_DIR
    train_paths = [batch_dir / f"data_batch_{idx}" for idx in range(1, 6)]
    test_path = batch_dir / "test_batch"
    missing = [str(path) for path in [*train_paths, test_path] if not path.exists()]
    if missing:
        files = ", ".join(missing)
        raise FileNotFoundError(
            "CIFAR-10 files are missing. Re-run with --download or place the "
            f"{CIFAR10_DIR} directory in {data_path}. Missing: {files}"
        )

    train_batches = [_read_cifar10_batch(path) for path in train_paths]
    x_train = np.concatenate([batch[0] for batch in train_batches], axis=0)
    y_train = np.concatenate([batch[1] for batch in train_batches], axis=0)
    x_test, y_test = _read_cifar10_batch(test_path)
    x_train = _normalize_cifar10(x_train)
    x_test = _normalize_cifar10(x_test)

    x_train, y_train = _limit_samples(x_train, y_train, train_limit, seed)
    x_test, y_test = _limit_samples(x_test, y_test, test_limit, seed + 1)

    return ImageDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        name="cifar10",
        input_shape=(3, 32, 32),
        num_classes=10,
    )


def make_synthetic_mnist_like(
    *,
    train_samples: int = 1000,
    test_samples: int = 300,
    seed: int = 0,
) -> ImageDataset:
    """Create a small MNIST-shaped dataset for smoke tests without downloads."""

    rng = np.random.default_rng(seed)
    prototypes = rng.uniform(0.0, 1.0, size=(10, 1, 28, 28)).astype(np.float32)
    masks = rng.uniform(0.0, 1.0, size=(10, 1, 28, 28)).astype(np.float32)
    prototypes = (0.75 * prototypes + 0.25 * (masks > 0.82)).astype(np.float32)

    def sample(count: int) -> tuple[np.ndarray, np.ndarray]:
        labels = rng.integers(0, 10, size=count, dtype=np.int64)
        noise = rng.normal(0.0, 0.18, size=(count, 1, 28, 28)).astype(np.float32)
        x = np.clip(prototypes[labels] + noise, 0.0, 1.0).astype(np.float32)
        return x, labels

    x_train, y_train = sample(train_samples)
    x_test, y_test = sample(test_samples)
    return ImageDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        name="synthetic",
        input_shape=(1, 28, 28),
        num_classes=10,
    )


def partition_clients(
    labels: np.ndarray,
    num_clients: int,
    *,
    strategy: str = "iid",
    dirichlet_alpha: float = 0.5,
    seed: int = 0,
) -> list[np.ndarray]:
    """按 IID 或标签 Dirichlet 分布划分客户端样本索引。"""

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
    # 每个标签独立抽取 Dirichlet 比例，alpha 越小，客户端标签偏斜越明显。
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


def _limit_samples(
    x: np.ndarray,
    y: np.ndarray,
    limit: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if limit is None or limit >= len(y):
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=limit, replace=False)
    return x[idx], y[idx]


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
    images = data.reshape(count, 1, rows, cols).astype(np.float32) / 255.0
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


def _read_cifar10_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    data = np.asarray(batch["data"], dtype=np.uint8)
    images = data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.asarray(batch["labels"], dtype=np.int64)
    return images, labels


def _normalize_cifar10(images: np.ndarray) -> np.ndarray:
    return ((images.astype(np.float32) - CIFAR10_CHANNEL_MEAN) / CIFAR10_CHANNEL_STD).astype(np.float32)


def _safe_extract(handle: tarfile.TarFile, destination: Path) -> None:
    base = destination.resolve()
    for member in handle.getmembers():
        target = (destination / member.name).resolve()
        if base != target and base not in target.parents:
            raise ValueError(f"refusing to extract archive member outside {destination}: {member.name}")
    handle.extractall(destination)
