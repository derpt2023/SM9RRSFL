"""Longitudinal SVD poisoning detector described by the project scheme."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

from .model import INPUT_DIM, NUM_CLASSES


@dataclass(frozen=True)
class DetectionResult:
    accepted: bool
    reason: str
    z_sigma: float = 0.0
    z_direction: float = 0.0
    sigma_delta: float = 0.0
    direction_shift: float = 0.0


@dataclass
class _Feature:
    sigma: float
    u0: np.ndarray


@dataclass
class _TagState:
    previous: _Feature | None = None
    history: Deque[tuple[float, float]] = field(default_factory=deque)


class LongitudinalSVDDetector:
    """Per-link-tag SVD trajectory detector with sliding-window Z-scores."""

    def __init__(
        self,
        *,
        window_size: int = 3,
        z_threshold: float = 3.0,
        input_dim: int = INPUT_DIM,
        num_classes: int = NUM_CLASSES,
        eps: float = 1e-8,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.eps = eps
        self._states: dict[str, _TagState] = {}

    def evaluate(self, tag: str, update: np.ndarray) -> DetectionResult:
        feature = self._extract(update)
        state = self._states.setdefault(tag, _TagState())
        if state.previous is None:
            state.previous = feature
            return DetectionResult(accepted=True, reason="initial_observation")

        sigma_delta = abs(feature.sigma - state.previous.sigma)
        cosine = _abs_cosine(feature.u0, state.previous.u0, self.eps)
        direction_shift = 1.0 - cosine

        if len(state.history) >= self.window_size:
            history = np.asarray(state.history, dtype=np.float64)
            mu = np.mean(history, axis=0)
            std = np.std(history, axis=0)
            sigma_scale = max(std[0], abs(mu[0]) * 0.25, 1e-6)
            direction_scale = max(std[1], 0.05)
            z_sigma = (sigma_delta - mu[0]) / sigma_scale
            z_direction = (direction_shift - mu[1]) / direction_scale
            sigma_alert = z_sigma > self.z_threshold and sigma_delta > max(mu[0] * 3.0, 1e-4)
            direction_alert = z_direction > self.z_threshold and direction_shift > 0.35
            if sigma_alert or direction_alert:
                return DetectionResult(
                    accepted=False,
                    reason="z_score_threshold",
                    z_sigma=float(z_sigma),
                    z_direction=float(z_direction),
                    sigma_delta=float(sigma_delta),
                    direction_shift=float(direction_shift),
                )
        else:
            z_sigma = 0.0
            z_direction = 0.0

        state.history.append((float(sigma_delta), float(direction_shift)))
        while len(state.history) > self.window_size:
            state.history.popleft()
        state.previous = feature
        reason = "baseline_warmup" if len(state.history) < self.window_size else "accepted"
        return DetectionResult(
            accepted=True,
            reason=reason,
            z_sigma=float(z_sigma),
            z_direction=float(z_direction),
            sigma_delta=float(sigma_delta),
            direction_shift=float(direction_shift),
        )

    def _extract(self, update: np.ndarray) -> _Feature:
        weight_count = self.input_dim * self.num_classes
        matrix = np.asarray(update[:weight_count], dtype=np.float32).reshape(
            self.input_dim, self.num_classes
        )
        u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
        return _Feature(sigma=float(singular_values[0]), u0=u[:, 0].astype(np.float32))


def _abs_cosine(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), eps)
    return abs(float(np.dot(a, b) / denom))
