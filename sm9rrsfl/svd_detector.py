"""Longitudinal SVD poisoning detector described by the project scheme."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

from .model import NUM_CLASSES


@dataclass(frozen=True)
class DetectionResult:
    accepted: bool
    reason: str
    count_increment: bool = False
    z_sigma: float = 0.0
    z_direction: float = 0.0
    sigma_delta: float = 0.0
    cosine_similarity: float = 0.0


@dataclass
class _Feature:
    sigma: float
    u0: np.ndarray


@dataclass
class _TagState:
    last_observed: _Feature | None = None
    observed_count: int = 0
    normal_history: Deque[tuple[float, float]] = field(default_factory=deque)


class LongitudinalSVDDetector:
    """Per-task-tag SVD trajectory detector with sliding-window Z-scores."""

    def __init__(
        self,
        *,
        window_size: int = 3,
        z_threshold: float = 3.0,
        input_dim: int | None = None,
        num_classes: int = NUM_CLASSES,
        matrix_offset: int | None = None,
        matrix_shape: tuple[int, int] | None = None,
        expected_update_size: int | None = None,
        compute_backend: str = "numpy",
        device: str = "auto",
        eps: float = 1e-8,
    ) -> None:
        if isinstance(window_size, bool) or not isinstance(window_size, int):
            raise TypeError("window_size must be an integer")
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if not np.isfinite(z_threshold) or z_threshold <= 0.0:
            raise ValueError("z_threshold must be finite and positive")
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise TypeError("num_classes must be an integer")
        if num_classes < 1:
            raise ValueError("num_classes must be at least 1")
        if input_dim is not None and (
            isinstance(input_dim, bool) or not isinstance(input_dim, int)
        ):
            raise TypeError("input_dim must be an integer")
        if input_dim is not None and input_dim < 1:
            raise ValueError("input_dim must be positive")
        if expected_update_size is not None:
            if isinstance(expected_update_size, bool) or not isinstance(
                expected_update_size,
                int,
            ):
                raise TypeError("expected_update_size must be an integer")
            if expected_update_size < 1:
                raise ValueError("expected_update_size must be positive")
        if not np.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be finite and positive")
        if input_dim is not None and (
            matrix_offset is not None or matrix_shape is not None
        ):
            raise ValueError("input_dim cannot be combined with matrix_offset or matrix_shape")
        self.window_size = window_size
        self.z_threshold = z_threshold
        if input_dim is not None:
            self.matrix_offset = 0
            self.matrix_shape = (input_dim, num_classes)
        elif matrix_offset is not None or matrix_shape is not None:
            if matrix_shape is None:
                raise ValueError("matrix_shape is required when matrix_offset is provided")
            self.matrix_offset = 0 if matrix_offset is None else matrix_offset
            self.matrix_shape = matrix_shape
        else:
            # Word 4.3.3 defines G_pi^(r) from the complete one-dimensional
            # model update.  Its concrete row count depends on the received
            # update length and is therefore resolved in _extract().
            self.matrix_offset = None
            self.matrix_shape = None
        if self.matrix_offset is not None:
            if isinstance(self.matrix_offset, bool) or not isinstance(
                self.matrix_offset,
                int,
            ):
                raise TypeError("matrix_offset must be an integer")
            if self.matrix_offset < 0:
                raise ValueError("matrix_offset must be non-negative")
        if self.matrix_shape is not None:
            if len(self.matrix_shape) != 2 or any(
                isinstance(size, bool)
                or not isinstance(size, int)
                or size < 1
                for size in self.matrix_shape
            ):
                raise ValueError("matrix_shape must contain two positive dimensions")
        self.num_classes = num_classes
        self.expected_update_size = expected_update_size
        self.eps = eps
        self.compute_backend = compute_backend
        self.device = device
        self._states: dict[str, _TagState] = {}

    def evaluate(self, tag: str, update: np.ndarray) -> DetectionResult:
        # Tag_pi 对应稳定的任务内匿名轨迹；窗口只吸收通过检测的正常特征。
        feature = self._extract(update)
        state = self._states.setdefault(tag, _TagState())
        if state.last_observed is None:
            state.last_observed = feature
            state.observed_count = 1
            return DetectionResult(accepted=True, reason="initial_observation")

        # Word 4.3.3: Delta lambda is signed and rho is the original cosine,
        # not |Delta lambda| or 1-|cosine|.
        sigma_delta = feature.sigma - state.last_observed.sigma
        cosine = _cosine(feature.u0, state.last_observed.u0, self.eps)

        # The first K observations are the baseline period, so round K+1 is
        # scored for the first time.  There are K-1 adjacent Delta/rho pairs
        # at that point because the first observation has no predecessor.
        should_score = state.observed_count >= self.window_size
        if should_score and state.normal_history:
            history = np.asarray(state.normal_history, dtype=np.float64)
            mu = np.mean(history, axis=0)
            std = np.std(history, axis=0)
            z_sigma = abs(sigma_delta - mu[0]) / max(float(std[0]), self.eps)
            z_direction = abs(cosine - mu[1]) / max(float(std[1]), self.eps)
            sigma_exceeded = z_sigma > self.z_threshold
            direction_exceeded = z_direction > self.z_threshold
            if sigma_exceeded or direction_exceeded:
                # Delta/rho are defined against the immediately preceding
                # communication round.  An anomalous feature is excluded from
                # the normal-history window, but it is still the r-1 endpoint
                # for the next round's adjacent-trajectory comparison.
                state.last_observed = feature
                state.observed_count += 1
                return DetectionResult(
                    accepted=False,
                    reason="z_score_threshold",
                    # A single exceeded indicator is still suspicious and is
                    # downweighted, but Count_pi advances only when both
                    # indicators exceed theta in the same round.
                    count_increment=sigma_exceeded and direction_exceeded,
                    z_sigma=float(z_sigma),
                    z_direction=float(z_direction),
                    sigma_delta=float(sigma_delta),
                    cosine_similarity=float(cosine),
                )
        else:
            z_sigma = 0.0
            z_direction = 0.0

        state.normal_history.append((float(sigma_delta), float(cosine)))
        while len(state.normal_history) > self.window_size:
            state.normal_history.popleft()
        state.last_observed = feature
        state.observed_count += 1
        reason = "accepted" if should_score else "baseline_warmup"
        return DetectionResult(
            accepted=True,
            reason=reason,
            z_sigma=float(z_sigma),
            z_direction=float(z_direction),
            sigma_delta=float(sigma_delta),
            cosine_similarity=float(cosine),
        )

    def _extract(self, update: np.ndarray) -> _Feature:
        vector = np.asarray(update, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("SVD update must be a one-dimensional vector")
        if (
            self.expected_update_size is not None
            and vector.size != self.expected_update_size
        ):
            raise ValueError(
                "SVD update length does not match the registered model"
            )
        if self.matrix_shape is None:
            if vector.size == 0:
                raise ValueError("SVD update must not be empty")
            rows = (vector.size + self.num_classes - 1) // self.num_classes
            padded = np.zeros(rows * self.num_classes, dtype=np.float32)
            padded[: vector.size] = vector
            matrix = padded.reshape(rows, self.num_classes)
        else:
            rows, cols = self.matrix_shape
            matrix_size = rows * cols
            assert self.matrix_offset is not None
            end = self.matrix_offset + matrix_size
            if vector.size < end:
                raise ValueError(
                    f"update length {vector.size} is too short for SVD matrix ending at {end}"
                )
            matrix = vector[self.matrix_offset:end].reshape(rows, cols)
        if _should_use_torch(self.compute_backend, self.device):
            from .torch_backend import torch_top_singular_feature

            sigma, u0 = torch_top_singular_feature(matrix, device=self.device)
            return _Feature(sigma=sigma, u0=_canonical_direction(u0))
        from .torch_backend import numpy_top_singular_feature

        sigma, u0 = numpy_top_singular_feature(matrix)
        return _Feature(sigma=sigma, u0=_canonical_direction(u0))


def _cosine(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), eps)
    return float(np.dot(a, b) / denom)


def _canonical_direction(direction: np.ndarray) -> np.ndarray:
    """Choose one deterministic representative of the SVD sign ambiguity."""

    vector = np.asarray(direction, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("first singular vector must be non-empty and one-dimensional")
    pivot = int(np.argmax(np.abs(vector)))
    return -vector if vector[pivot] < 0.0 else vector


def _should_use_torch(compute_backend: str, device: str) -> bool:
    try:
        from .torch_backend import should_use_torch
    except RuntimeError:
        return False
    return should_use_torch(compute_backend, device)
