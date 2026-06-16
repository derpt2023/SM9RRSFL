"""Reproduction of the trajectory anomaly detector from literature [13]."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2

import numpy as np

from .model import DEFAULT_SPEC, NUM_CLASSES


@dataclass(frozen=True)
class Ding13RoundResult:
    weights: dict[str, float]
    outliers: set[str]
    newly_removed: set[str]
    true_positive_removed: int
    false_positive_removed: int


class Ding13TrajectoryDetector:
    """SVD-difference + Isolation Forest detector described by Ding et al. [13]."""

    def __init__(
        self,
        client_ids: list[str],
        *,
        contamination: float,
        penalty_factor: float = 0.5,
        remove_after: int = 3,
        n_trees: int = 100,
        sample_size: int = 256,
        max_depth: int = 10,
        seed: int = 0,
        input_dim: int | None = None,
        num_classes: int = NUM_CLASSES,
        matrix_offset: int | None = None,
        matrix_shape: tuple[int, int] | None = None,
        compute_backend: str = "numpy",
        device: str = "auto",
    ) -> None:
        if not client_ids:
            raise ValueError("client_ids must not be empty")
        if not 0.0 <= contamination < 1.0:
            raise ValueError("contamination must be in [0, 1)")
        self.client_ids = tuple(client_ids)
        self.contamination = contamination
        self.penalty_factor = penalty_factor
        self.remove_after = remove_after
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.max_depth = max_depth
        self.seed = seed
        if input_dim is not None:
            self.matrix_offset = 0
            self.matrix_shape = (input_dim, num_classes)
        else:
            self.matrix_offset = DEFAULT_SPEC.svd_matrix_offset if matrix_offset is None else matrix_offset
            self.matrix_shape = DEFAULT_SPEC.svd_matrix_shape if matrix_shape is None else matrix_shape
        self.compute_backend = compute_backend
        self.device = device
        self.weights = {identity: 1.0 / len(client_ids) for identity in client_ids}
        self.previous_singulars: dict[str, np.ndarray] = {}
        self.consecutive_outliers = {identity: 0 for identity in client_ids}
        self.removed: set[str] = set()

    def evaluate_round(
        self,
        updates: dict[str, np.ndarray],
        malicious_clients: set[str],
        *,
        round_id: int,
    ) -> Ding13RoundResult:
        active_ids = [identity for identity in self.client_ids if identity in updates]
        current = {identity: self._singular_values(updates[identity]) for identity in active_ids}

        diff_features: dict[str, np.ndarray] = {}
        for identity in active_ids:
            previous = self.previous_singulars.get(identity)
            if previous is not None:
                diff_features[identity] = current[identity] - previous
            self.previous_singulars[identity] = current[identity]

        outliers: set[str] = set()
        if len(diff_features) >= 3:
            ordered_ids = list(diff_features)
            features = np.asarray([diff_features[identity] for identity in ordered_ids], dtype=np.float32)
            scores = IsolationForestLite(
                n_trees=self.n_trees,
                sample_size=self.sample_size,
                max_depth=self.max_depth,
                seed=self.seed + round_id * 7919,
            ).score_samples(features)
            count = int(round(len(ordered_ids) * self.contamination))
            count = min(max(count, 0), len(ordered_ids) - 1)
            if count > 0:
                selected = np.argsort(scores)[-count:]
                outliers = {ordered_ids[int(idx)] for idx in selected}

        newly_removed: set[str] = set()
        for identity in active_ids:
            if identity in outliers:
                self.weights[identity] *= self.penalty_factor
                self.consecutive_outliers[identity] += 1
            else:
                if self.weights[identity] < 1.0 / len(self.client_ids):
                    self.weights[identity] = min(
                        1.0 / len(self.client_ids),
                        self.weights[identity] * 2.0,
                    )
                self.consecutive_outliers[identity] = 0

            if self.consecutive_outliers[identity] >= self.remove_after:
                self.weights[identity] = 0.0
                if identity not in self.removed:
                    newly_removed.add(identity)
                self.removed.add(identity)

        self._renormalize_weights()
        true_positive = sum(1 for identity in newly_removed if identity in malicious_clients)
        false_positive = len(newly_removed) - true_positive
        return Ding13RoundResult(
            weights=dict(self.weights),
            outliers=outliers,
            newly_removed=newly_removed,
            true_positive_removed=true_positive,
            false_positive_removed=false_positive,
        )

    def _renormalize_weights(self) -> None:
        active = [identity for identity in self.client_ids if identity not in self.removed]
        total = sum(self.weights[identity] for identity in active)
        if total <= 0.0:
            reset = 1.0 / max(1, len(active))
            for identity in active:
                self.weights[identity] = reset
        else:
            for identity in active:
                self.weights[identity] /= total
        for identity in self.removed:
            self.weights[identity] = 0.0

    def _singular_values(self, update: np.ndarray) -> np.ndarray:
        rows, cols = self.matrix_shape
        matrix_size = rows * cols
        end = self.matrix_offset + matrix_size
        if update.shape[0] < end:
            raise ValueError(f"update length {update.shape[0]} is too short for SVD matrix ending at {end}")
        matrix = np.asarray(update[self.matrix_offset:end], dtype=np.float32).reshape(rows, cols)
        if _should_use_torch(self.compute_backend, self.device):
            from .torch_backend import torch_singular_values_from_gram

            return torch_singular_values_from_gram(matrix, device=self.device)
        gram = matrix.T @ matrix
        return np.linalg.svd(gram, compute_uv=False).astype(np.float32)


class IsolationForestLite:
    """Small NumPy Isolation Forest sufficient for the experiment sizes here."""

    def __init__(
        self,
        *,
        n_trees: int = 100,
        sample_size: int = 256,
        max_depth: int = 10,
        seed: int = 0,
    ) -> None:
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)

    def score_samples(self, samples: np.ndarray) -> np.ndarray:
        data = np.asarray(samples, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("samples must have shape [n_samples, n_features]")
        n_samples = data.shape[0]
        if n_samples <= 1:
            return np.zeros(n_samples, dtype=np.float32)
        tree_sample_size = min(self.sample_size, n_samples)
        paths = np.zeros(n_samples, dtype=np.float64)
        for _ in range(self.n_trees):
            sample_indices = self.rng.choice(n_samples, size=tree_sample_size, replace=False)
            tree = _IsolationTree.build(
                data[sample_indices],
                depth=0,
                max_depth=min(self.max_depth, max(1, ceil(log2(tree_sample_size)))),
                rng=self.rng,
            )
            for idx, row in enumerate(data):
                paths[idx] += tree.path_length(row)
        avg_path = paths / max(1, self.n_trees)
        normalizer = _average_path_length(tree_sample_size)
        if normalizer <= 0.0:
            return np.zeros(n_samples, dtype=np.float32)
        return np.power(2.0, -avg_path / normalizer).astype(np.float32)


@dataclass
class _IsolationTree:
    size: int
    split_feature: int | None = None
    split_value: float | None = None
    left: "_IsolationTree | None" = None
    right: "_IsolationTree | None" = None

    @classmethod
    def build(
        cls,
        samples: np.ndarray,
        *,
        depth: int,
        max_depth: int,
        rng: np.random.Generator,
    ) -> "_IsolationTree":
        if depth >= max_depth or len(samples) <= 1:
            return cls(size=len(samples))
        mins = np.min(samples, axis=0)
        maxs = np.max(samples, axis=0)
        valid = np.flatnonzero(maxs > mins)
        if len(valid) == 0:
            return cls(size=len(samples))
        feature = int(rng.choice(valid))
        split = float(rng.uniform(mins[feature], maxs[feature]))
        left_mask = samples[:, feature] < split
        if not np.any(left_mask) or np.all(left_mask):
            return cls(size=len(samples))
        return cls(
            size=len(samples),
            split_feature=feature,
            split_value=split,
            left=cls.build(samples[left_mask], depth=depth + 1, max_depth=max_depth, rng=rng),
            right=cls.build(samples[~left_mask], depth=depth + 1, max_depth=max_depth, rng=rng),
        )

    def path_length(self, sample: np.ndarray, depth: int = 0) -> float:
        if self.split_feature is None or self.left is None or self.right is None:
            return depth + _average_path_length(self.size)
        if sample[self.split_feature] < self.split_value:
            return self.left.path_length(sample, depth + 1)
        return self.right.path_length(sample, depth + 1)


def _average_path_length(size: int) -> float:
    if size <= 1:
        return 0.0
    if size == 2:
        return 1.0
    harmonic = np.log(size - 1) + 0.5772156649
    return float(2.0 * harmonic - 2.0 * (size - 1) / size)


def _should_use_torch(compute_backend: str, device: str) -> bool:
    try:
        from .torch_backend import should_use_torch
    except RuntimeError:
        return False
    return should_use_torch(compute_backend, device)
