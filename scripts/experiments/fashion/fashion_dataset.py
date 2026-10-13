"""Verified Fashion-MNIST input for the independent Fashion experiment.

This module deliberately does not change the legacy MNIST/CIFAR loaders.  The
four archive names are identical to MNIST, so official content checksums are
mandatory even for a populated local cache.  Only training images enter the
deterministic train/validation/attacker split.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import os
from pathlib import Path
import stat
import struct
import tempfile
import time
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np

from sm9rrsfl.datasets import ImageDataset, stratified_training_three_way_split
from sm9rrsfl.ours_calibration import split_metadata, validate_targeted_split


DATASET_NAME = "fashion_mnist"
TRAIN_SAMPLES = 60000
TEST_SAMPLES = 10000
SOURCE_README = "https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md"
SOURCE_PAPER = "https://arxiv.org/abs/1708.07747"
SOURCE_BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
# MD5s published in the authors' README; they identify compressed archives.
FASHION_FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
}
LABELS = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
          "Shirt", "Sneaker", "Bag", "Ankle boot")
PREPROCESSING = {
    "version": "fashion-mnist-native-nchw-unit-v1",
    "input_shape": [1, 28, 28],
    "image_dtype": "float32",
    "label_dtype": "int64",
    "pixel_transform": "uint8 / 255.0",
    "pixel_range": [0.0, 1.0],
    "resize": False,
    "augmentation": False,
    "estimated_normalization_statistics": False,
}


def _regular_file(path):
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        raise FileNotFoundError(f"Fashion-MNIST archive is missing: {path}") from None
    if not stat.S_ISREG(mode):
        raise ValueError(f"Fashion-MNIST cache entry must be a regular file, not a symlink: {path}")


@contextmanager
def _cache_lock(directory, timeout=300.0):
    """Keep the lock inode persistent; unlinking a live flock breaks exclusion."""
    lock_path = directory / ".fashion_mnist.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"Fashion-MNIST lock is not a regular file: {lock_path}")
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Fashion-MNIST cache is locked: {directory}") from None
                time.sleep(.05)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _source_checksum(path, expected_md5):
    _regular_file(path)
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
            size += len(block)
    if md5.hexdigest() != expected_md5:
        raise ValueError(
            f"Fashion-MNIST checksum mismatch: {path.name}; expected official MD5 "
            f"{expected_md5}, observed {md5.hexdigest()}. The cache may contain "
            "MNIST or a damaged file; the existing file was not overwritten."
        )
    return {"official_md5": expected_md5, "md5": md5.hexdigest(),
            "sha256": sha256.hexdigest(), "size_bytes": size}


def _read_idx(path, kind, count):
    """Bound decompression and check the entire IDX payload, including EOF."""
    images = kind.endswith("images")
    header_size = 16 if images else 8
    payload_size = count * 28 * 28 if images else count
    try:
        with gzip.open(path, "rb") as stream:
            header = stream.read(header_size)
            if len(header) != header_size:
                raise ValueError(f"Fashion-MNIST truncated IDX header: {path.name}")
            values = struct.unpack(">IIII" if images else ">II", header)
            expected = (2051, count, 28, 28) if images else (2049, count)
            if values != expected:
                raise ValueError(
                    f"Fashion-MNIST unexpected IDX header/count/shape: {path.name}; "
                    f"expected {expected}, got {values}"
                )
            payload = stream.read(payload_size + 1)
            if len(payload) != payload_size:
                raise ValueError(f"Fashion-MNIST incorrect IDX payload length: {path.name}")
    except (OSError, EOFError) as error:
        raise ValueError(f"Fashion-MNIST invalid gzip archive: {path.name}: {error}") from error
    array = np.frombuffer(payload, dtype=np.uint8)
    if images:
        return array.reshape(count, 1, 28, 28).astype(np.float32) / np.float32(255.0)
    if np.any(array >= len(LABELS)):
        raise ValueError(f"Fashion-MNIST labels outside [0, 9]: {path.name}")
    frequencies = np.bincount(array, minlength=len(LABELS))
    if count % len(LABELS) or not np.all(frequencies == count // len(LABELS)):
        raise ValueError(f"Fashion-MNIST unexpected class counts: {path.name}: {frequencies.tolist()}")
    return array.astype(np.int64)


def _download_verified(directory, kind, filename, expected_md5, count):
    """Publish a complete valid archive; preserve unrelated and invalid caches."""
    target = directory / filename
    temporary_fd, temporary_name = tempfile.mkstemp(prefix=f".{filename}.", suffix=".partial", dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(temporary_fd, "wb") as destination:
            with urlopen(f"{SOURCE_BASE}/{filename}", timeout=60) as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(block)
            destination.flush()
            os.fsync(destination.fileno())
        _source_checksum(temporary, expected_md5)
        _read_idx(temporary, kind, count)
        # Under our cache lock, another cooperating downloader cannot appear.
        # Also reject a file created outside the protocol rather than replacing it.
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Fashion-MNIST cache entry appeared during download: {target}")
        os.replace(temporary, target)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _load_verified(data_dir, *, download):
    directory = Path(data_dir)
    if directory.is_symlink():
        raise ValueError("Fashion-MNIST data directory must not be a symlink")
    if download:
        directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise FileNotFoundError(f"Fashion-MNIST data directory is missing: {directory}")
    arrays, checksums = {}, {}
    with _cache_lock(directory):
        # Validate existing archives first, so a wrong MNIST cache triggers no
        # downloads and never gets partially converted to a Fashion cache.
        for kind, (filename, expected_md5) in FASHION_FILES.items():
            path = directory / filename
            if path.exists() or path.is_symlink():
                _source_checksum(path, expected_md5)
        for kind, (filename, expected_md5) in FASHION_FILES.items():
            path = directory / filename
            count = TRAIN_SAMPLES if kind.startswith("train_") else TEST_SAMPLES
            if not path.exists() and not path.is_symlink() and download:
                _download_verified(directory, kind, filename, expected_md5, count)
            checksums[kind] = {"filename": filename, "url": f"{SOURCE_BASE}/{filename}",
                               **_source_checksum(path, expected_md5)}
            arrays[kind] = _read_idx(path, kind, count)
    return ImageDataset(x_train=arrays["train_images"], y_train=arrays["train_labels"],
                        x_test=arrays["test_images"], y_test=arrays["test_labels"],
                        name=DATASET_NAME, input_shape=(1, 28, 28), num_classes=10), checksums


def load_fashion_mnist(data_dir="data/fashion_mnist", *, download=False):
    """Read all 60,000 training and 10,000 test examples after source checks."""
    return _load_verified(data_dir, download=download)[0]


def _array_sha256(array):
    view = memoryview(np.ascontiguousarray(array)).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(view), 8 * 1024 * 1024):
        digest.update(view[start:start + 8 * 1024 * 1024])
    return digest.hexdigest()


def load_split(spec, data_dir=None):
    """Return the legacy split object and a portable Fashion source contract."""
    data = spec["dataset"]
    if (data.get("name") != DATASET_NAME or data.get("train_samples") != TRAIN_SAMPLES
            or data.get("test_samples") != TEST_SAMPLES or data.get("validation_fraction") != .05):
        raise ValueError("Fashion-MNIST requires the complete official dataset and a 5% validation split")
    for field, expected in (("train_fraction", .9), ("attack_fraction", .05)):
        if field in data and data[field] != expected:
            raise ValueError("Fashion-MNIST requires the fixed 90%/5%/5% training split")
    seed = data["split_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Fashion-MNIST split_seed must be a nonnegative integer")
    parameters = spec["shared_parameters"]
    source, target = parameters["attack_source_label"], parameters["attack_target_label"]
    count = parameters["attack_target_count"]
    if (any(isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < 10
            for label in (source, target)) or source == target):
        raise ValueError("Fashion-MNIST source and target labels must be distinct integers in [0, 9]")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("Fashion-MNIST attack_target_count must be a positive integer")
    original, checksums = _load_verified(data_dir or data["data_dir"], download=data["download"])
    split = stratified_training_three_way_split(original, seed=seed, train_fraction=.9,
                                                calibration_fraction=.05, attack_fraction=.05)
    validate_targeted_split(split, SimpleNamespace(attack=parameters["attack"],
                            attack_source_label=source, attack_target_count=count))
    contract = {**split_metadata(split, seed), "dataset_name": DATASET_NAME,
                "source_readme": SOURCE_README, "source_paper": SOURCE_PAPER,
                "source_checksums": checksums,
                "official_train_samples": TRAIN_SAMPLES, "official_test_samples": TEST_SAMPLES,
                "requested_split_fractions": {"train": .9, "validation": .05, "attack_auxiliary": .05},
                "label_mapping": {str(index): label for index, label in enumerate(LABELS)},
                "preprocessing": dict(PREPROCESSING),
                "targeted_attack": {"source_label": source, "source_name": LABELS[source],
                    "target_label": target, "target_name": LABELS[target], "evaluation_count": count,
                    "available_source_samples": {
                        "attack_auxiliary": int(np.count_nonzero(split.main_dataset.y_attack == source)),
                        "validation": int(np.count_nonzero(split.calibration_dataset.y_test == source)),
                        "official_evaluation": int(np.count_nonzero(split.main_dataset.y_test == source))}}}
    fields = ("x_train", "y_train", "x_test", "y_test", "x_attack", "y_attack")
    for phase, dataset in (("validation", split.calibration_dataset), ("final", split.main_dataset)):
        contract[phase + "_arrays"] = {field: _array_sha256(getattr(dataset, field)) for field in fields}
        contract[phase + "_array_layout"] = {
            field: {"shape": list(getattr(dataset, field).shape), "dtype": str(getattr(dataset, field).dtype)}
            for field in fields}
    return split, contract
