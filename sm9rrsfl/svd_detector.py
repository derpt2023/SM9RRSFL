"""Trusted-anchor, dual-reference longitudinal SVD poisoning detector."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

from .model import NUM_CLASSES


_DISTANCE_KEYS = (
    "spectrum_adjacent",
    "subspace_adjacent",
    "spectrum_anchor",
    "subspace_anchor",
)


@dataclass(frozen=True)
class DetectionResult:
    accepted: bool
    reason: str
    count_increment: bool = False
    # Compatibility summaries retained for older analysis code. In v3 these
    # are the largest spectral Z and reliability-weighted subspace Z across
    # the two references, respectively.
    z_sigma: float = 0.0
    z_direction: float = 0.0
    # Compatibility summaries: adjacent log-spectrum distance and normalized
    # projector overlap (1 means the same q-dimensional subspace).
    sigma_delta: float = 0.0
    cosine_similarity: float = 1.0
    spectrum_adjacent_distance: float = 0.0
    subspace_adjacent_distance: float = 0.0
    spectrum_anchor_distance: float = 0.0
    subspace_anchor_distance: float = 0.0
    z_spectrum_adjacent: float = 0.0
    z_subspace_adjacent: float = 0.0
    z_spectrum_anchor: float = 0.0
    z_subspace_anchor: float = 0.0
    adjacent_score: float = 0.0
    anchor_score: float = 0.0
    cumulative_drift: float = 0.0
    spectral_gap: float = 0.0
    direction_reliability: float = 0.0
    adjacent_exceeded: bool = False
    anchor_exceeded: bool = False
    drift_exceeded: bool = False
    trusted_history_size: int = 0


@dataclass
class _Feature:
    log_spectrum: np.ndarray
    basis: np.ndarray
    singular_values: np.ndarray
    spectral_gap: float

    @property
    def lambda1(self) -> float:
        return float(self.singular_values[0])


@dataclass
class _TimedFeature:
    round_id: int
    feature: _Feature


@dataclass
class _TagState:
    last_observed: _TimedFeature | None = None
    observed_count: int = 0
    trusted_history: Deque[_TimedFeature] = field(default_factory=deque)
    normal_distances: dict[str, Deque[float]] = field(default_factory=dict)
    cumulative_drift: float = 0.0
    trusted_gram: np.ndarray | None = None


class LongitudinalSVDDetector:
    """Per-tag trusted-anchor detector from the third scheme revision.

    ``decision_rule="any"`` implements the Section 4.3.3 formula: an adjacent
    jump, trusted-anchor deviation, or accumulated drift may trigger
    independently. No AND variant is supported because it does not implement
    the formula's logical-OR operator.
    """

    def __init__(
        self,
        *,
        window_size: int = 3,
        z_threshold: float = 3.0,
        subspace_dim: int = 2,
        gap_threshold: float = 0.1,
        adjacent_threshold: float | None = None,
        anchor_threshold: float | None = None,
        drift_memory: float = 0.9,
        drift_allowance: float = 1.0,
        drift_threshold: float = 5.0,
        decision_rule: str = "any",
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
        if isinstance(subspace_dim, bool) or not isinstance(subspace_dim, int):
            raise TypeError("subspace_dim must be an integer")
        if subspace_dim < 1:
            raise ValueError("subspace_dim must be at least 1")
        if not np.isfinite(gap_threshold) or gap_threshold <= 0.0:
            raise ValueError("gap_threshold must be finite and positive")
        adjacent_threshold = (
            z_threshold if adjacent_threshold is None else adjacent_threshold
        )
        anchor_threshold = z_threshold if anchor_threshold is None else anchor_threshold
        if not np.isfinite(adjacent_threshold) or adjacent_threshold <= 0.0:
            raise ValueError("adjacent_threshold must be finite and positive")
        if not np.isfinite(anchor_threshold) or anchor_threshold <= 0.0:
            raise ValueError("anchor_threshold must be finite and positive")
        if not np.isfinite(drift_memory) or not 0.0 < drift_memory <= 1.0:
            raise ValueError("drift_memory must be in (0, 1]")
        if not np.isfinite(drift_allowance) or drift_allowance <= 0.0:
            raise ValueError("drift_allowance must be finite and positive")
        if not np.isfinite(drift_threshold) or drift_threshold <= 0.0:
            raise ValueError("drift_threshold must be finite and positive")
        if decision_rule != "any":
            raise ValueError("decision_rule must be 'any' (the v3 OR formula)")
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
        self.z_threshold = float(z_threshold)
        self.subspace_dim = subspace_dim
        self.gap_threshold = float(gap_threshold)
        self.adjacent_threshold = float(adjacent_threshold)
        self.anchor_threshold = float(anchor_threshold)
        self.drift_memory = float(drift_memory)
        self.drift_allowance = float(drift_allowance)
        self.drift_threshold = float(drift_threshold)
        self.decision_rule = decision_rule
        if input_dim is not None:
            self.matrix_offset = 0
            self.matrix_shape = (input_dim, num_classes)
        elif matrix_offset is not None or matrix_shape is not None:
            if matrix_shape is None:
                raise ValueError("matrix_shape is required when matrix_offset is provided")
            self.matrix_offset = 0 if matrix_offset is None else matrix_offset
            self.matrix_shape = matrix_shape
        else:
            self.matrix_offset = None
            self.matrix_shape = None
        if self.matrix_shape is None and self.subspace_dim >= num_classes:
            raise ValueError(
                "subspace_dim must leave room for the q+1 singular value"
            )
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
            if self.subspace_dim >= min(self.matrix_shape):
                raise ValueError(
                    "subspace_dim must leave room for the q+1 singular value"
                )
        self.num_classes = num_classes
        self.expected_update_size = expected_update_size
        self.eps = float(eps)
        self.compute_backend = compute_backend
        self.device = device
        self._states: dict[str, _TagState] = {}

    def evaluate(
        self,
        tag: str,
        update: np.ndarray,
        *,
        round_id: int | None = None,
    ) -> DetectionResult:
        feature = self._extract(update)
        state = self._states.setdefault(tag, self._new_state())
        current_round = self._resolve_round_id(state, round_id)
        timed = _TimedFeature(current_round, feature)

        if state.last_observed is None:
            state.last_observed = timed
            state.observed_count = 1
            self._append_trusted(state, timed)
            return DetectionResult(
                accepted=True,
                reason="initial_observation",
                spectral_gap=feature.spectral_gap,
                trusted_history_size=1,
            )

        predicted_spectrum, trusted_lambda1 = self._trusted_reference(
            state,
            current_round,
        )
        previous = state.last_observed.feature
        distances = {
            "spectrum_adjacent": float(
                np.linalg.norm(feature.log_spectrum - previous.log_spectrum)
            ),
            "subspace_adjacent": _projector_distance(
                feature.basis,
                previous.basis,
                self.subspace_dim,
            ),
            "spectrum_anchor": float(
                np.linalg.norm(feature.log_spectrum - predicted_spectrum)
            ),
            "subspace_anchor": _anchor_projector_distance(
                feature.basis,
                state,
                self.subspace_dim,
            ),
        }
        reliability = min(1.0, feature.spectral_gap / self.gap_threshold) * min(
            1.0,
            feature.lambda1 / (trusted_lambda1 + self.eps),
        )
        should_score = state.observed_count >= self.window_size
        z_scores = {key: 0.0 for key in _DISTANCE_KEYS}
        adjacent_score = 0.0
        anchor_score = 0.0
        adjacent_exceeded = False
        anchor_exceeded = False
        drift_exceeded = False
        anomalous = False
        if should_score:
            z_scores = {
                key: _robust_one_sided_z(
                    distances[key],
                    state.normal_distances[key],
                    self.eps,
                )
                for key in _DISTANCE_KEYS
            }
            # The PDF typesets chi and Z_U with a comma, but its prose defines
            # chi as a reliability weight. Multiplication is the only reading
            # that actually suppresses an unreliable direction.
            adjacent_score = max(
                z_scores["spectrum_adjacent"],
                reliability * z_scores["subspace_adjacent"],
            )
            anchor_score = max(
                z_scores["spectrum_anchor"],
                reliability * z_scores["subspace_anchor"],
            )
            state.cumulative_drift = max(
                0.0,
                self.drift_memory * state.cumulative_drift
                + anchor_score
                - self.drift_allowance,
            )
            adjacent_exceeded = adjacent_score > self.adjacent_threshold
            anchor_exceeded = anchor_score > self.anchor_threshold
            drift_exceeded = state.cumulative_drift > self.drift_threshold
            evidence = (adjacent_exceeded, anchor_exceeded, drift_exceeded)
            anomalous = any(evidence)

        if not anomalous:
            for key in _DISTANCE_KEYS:
                state.normal_distances[key].append(distances[key])
            self._append_trusted(state, timed)

        # An anomaly still becomes the recent observation for the adjacent
        # reference; it never contaminates the trusted anchor or distance data.
        state.last_observed = timed
        state.observed_count += 1
        reason = (
            f"composite_threshold_{self.decision_rule}"
            if anomalous
            else ("accepted" if should_score else "baseline_warmup")
        )
        adjacent_subspace_similarity = max(
            0.0,
            1.0 - distances["subspace_adjacent"] ** 2,
        )
        return DetectionResult(
            accepted=not anomalous,
            reason=reason,
            count_increment=anomalous,
            z_sigma=max(
                z_scores["spectrum_adjacent"],
                z_scores["spectrum_anchor"],
            ),
            z_direction=max(
                reliability * z_scores["subspace_adjacent"],
                reliability * z_scores["subspace_anchor"],
            ),
            sigma_delta=distances["spectrum_adjacent"],
            cosine_similarity=adjacent_subspace_similarity,
            spectrum_adjacent_distance=distances["spectrum_adjacent"],
            subspace_adjacent_distance=distances["subspace_adjacent"],
            spectrum_anchor_distance=distances["spectrum_anchor"],
            subspace_anchor_distance=distances["subspace_anchor"],
            z_spectrum_adjacent=z_scores["spectrum_adjacent"],
            z_subspace_adjacent=z_scores["subspace_adjacent"],
            z_spectrum_anchor=z_scores["spectrum_anchor"],
            z_subspace_anchor=z_scores["subspace_anchor"],
            adjacent_score=float(adjacent_score),
            anchor_score=float(anchor_score),
            cumulative_drift=float(state.cumulative_drift),
            spectral_gap=feature.spectral_gap,
            direction_reliability=float(reliability),
            adjacent_exceeded=adjacent_exceeded,
            anchor_exceeded=anchor_exceeded,
            drift_exceeded=drift_exceeded,
            trusted_history_size=len(state.trusted_history),
        )

    def _new_state(self) -> _TagState:
        return _TagState(
            trusted_history=deque(maxlen=self.window_size),
            normal_distances={
                key: deque(maxlen=self.window_size) for key in _DISTANCE_KEYS
            },
        )

    def forget(self, tag: str) -> bool:
        """Release trajectory state after a certified permanent revocation."""

        return self._states.pop(tag, None) is not None

    def estimated_state_bytes(self) -> int:
        """Return unique NumPy-array bytes retained by all tag trajectories."""

        seen: set[int] = set()
        total = 0
        for state in self._states.values():
            timed_features = list(state.trusted_history)
            if state.last_observed is not None:
                timed_features.append(state.last_observed)
            for timed in timed_features:
                for array in (
                    timed.feature.log_spectrum,
                    timed.feature.basis,
                    timed.feature.singular_values,
                ):
                    identity = id(array)
                    if identity not in seen:
                        seen.add(identity)
                        total += int(array.nbytes)
            if state.trusted_gram is not None and id(state.trusted_gram) not in seen:
                seen.add(id(state.trusted_gram))
                total += int(state.trusted_gram.nbytes)
        return total

    def _resolve_round_id(self, state: _TagState, round_id: int | None) -> int:
        if round_id is None:
            return 1 if state.last_observed is None else state.last_observed.round_id + 1
        if isinstance(round_id, bool) or not isinstance(round_id, int):
            raise TypeError("round_id must be an integer")
        if round_id < 1:
            raise ValueError("round_id must be positive")
        if state.last_observed is not None and round_id <= state.last_observed.round_id:
            raise ValueError("round_id must increase for each task tag")
        return round_id

    def _trusted_reference(
        self,
        state: _TagState,
        current_round: int,
    ) -> tuple[np.ndarray, float]:
        trusted = tuple(state.trusted_history)
        if not trusted:
            raise RuntimeError("trusted history is unexpectedly empty")
        latest = trusted[-1]
        if len(trusted) < 2:
            velocity = np.zeros_like(latest.feature.log_spectrum)
        else:
            slopes = []
            for previous, current in zip(trusted, trusted[1:]):
                elapsed = current.round_id - previous.round_id
                if elapsed <= 0:
                    raise RuntimeError("trusted history round ids must increase")
                slopes.append(
                    (current.feature.log_spectrum - previous.feature.log_spectrum)
                    / elapsed
                )
            velocity = np.median(np.stack(slopes, axis=0), axis=0)
        predicted = latest.feature.log_spectrum + (
            current_round - latest.round_id
        ) * velocity
        trusted_lambda1 = float(
            np.median([item.feature.lambda1 for item in trusted])
        )
        return predicted, trusted_lambda1

    def _append_trusted(self, state: _TagState, timed: _TimedFeature) -> None:
        """Append one accepted feature and update the compact anchor Gram."""

        existing = list(state.trusted_history)
        gram = state.trusted_gram
        if len(existing) == self.window_size:
            existing = existing[1:]
            if gram is not None:
                gram = gram[self.subspace_dim :, self.subspace_dim :]
        new_basis = np.asarray(timed.feature.basis, dtype=np.float64)
        self_block = new_basis.T @ new_basis
        if not existing:
            updated = self_block
        else:
            if gram is None:
                old_columns = np.concatenate(
                    [
                        np.asarray(item.feature.basis, dtype=np.float64)
                        for item in existing
                    ],
                    axis=1,
                )
                gram = old_columns.T @ old_columns
            cross = np.concatenate(
                [
                    np.asarray(item.feature.basis, dtype=np.float64).T @ new_basis
                    for item in existing
                ],
                axis=0,
            )
            updated = np.block([[gram, cross], [cross.T, self_block]])
        state.trusted_history.append(timed)
        state.trusted_gram = np.asarray(updated, dtype=np.float64)

    def _extract(self, update: np.ndarray) -> _Feature:
        vector = np.asarray(update, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("SVD update must be a one-dimensional vector")
        if not np.isfinite(vector).all():
            raise ValueError("SVD update contains NaN or infinity")
        if (
            self.expected_update_size is not None
            and vector.size != self.expected_update_size
        ):
            raise ValueError("SVD update length does not match the registered model")
        if self.matrix_shape is None:
            if vector.size == 0:
                raise ValueError("SVD update must not be empty")
            rows = (vector.size + self.num_classes - 1) // self.num_classes
            if self.subspace_dim >= min(rows, self.num_classes):
                raise ValueError(
                    "subspace_dim must leave room for the q+1 singular value"
                )
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
            from .torch_backend import torch_top_singular_subspace

            singular_values, basis = torch_top_singular_subspace(
                matrix,
                rank=self.subspace_dim,
                device=self.device,
            )
        else:
            from .torch_backend import numpy_top_singular_subspace

            singular_values, basis = numpy_top_singular_subspace(
                matrix,
                rank=self.subspace_dim,
            )
        singular_values = np.asarray(singular_values, dtype=np.float64)
        basis = np.asarray(basis, dtype=np.float32)
        log_spectrum = np.log(singular_values[: self.subspace_dim] + self.eps)
        lambda_q = float(singular_values[self.subspace_dim - 1])
        lambda_next = float(singular_values[self.subspace_dim])
        gap = max(0.0, lambda_q - lambda_next) / max(lambda_q, self.eps)
        return _Feature(
            log_spectrum=log_spectrum,
            basis=basis,
            singular_values=singular_values,
            spectral_gap=float(gap),
        )


def _robust_one_sided_z(
    value: float,
    history: Deque[float],
    eps: float,
) -> float:
    if not history:
        return 0.0
    values = np.asarray(history, dtype=np.float64)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = max(1.4826 * mad, eps)
    return max(0.0, (float(value) - center) / scale)


def _projector_distance(a: np.ndarray, b: np.ndarray, rank: int) -> float:
    """Compute ||UU^T-VV^T||_F/sqrt(2q) without an O(m^2) projector."""

    overlap = np.asarray(a, dtype=np.float64).T @ np.asarray(b, dtype=np.float64)
    squared = 1.0 - float(np.sum(overlap * overlap)) / rank
    return float(np.sqrt(max(0.0, min(1.0, squared))))


def _anchor_projector_distance(
    current: np.ndarray,
    state: _TagState,
    rank: int,
) -> float:
    """Distance to Eig_q(sum U_jU_j^T) using only a small cached Gram."""

    trusted = tuple(state.trusted_history)
    if len(trusted) == 1:
        return _projector_distance(current, trusted[0].feature.basis, rank)
    gram = state.trusted_gram
    if gram is None:
        raise RuntimeError("trusted anchor Gram is unexpectedly missing")
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:rank]
    values = np.maximum(eigenvalues[order], 0.0)
    if np.any(values <= np.finfo(np.float64).eps):
        raise np.linalg.LinAlgError("trusted anchor subspace is rank deficient")
    current64 = np.asarray(current, dtype=np.float64)
    cross = np.concatenate(
        [
            current64.T @ np.asarray(item.feature.basis, dtype=np.float64)
            for item in trusted
        ],
        axis=1,
    )
    overlap = cross @ eigenvectors[:, order]
    overlap /= np.sqrt(values)[None, :]
    squared = 1.0 - float(np.sum(overlap * overlap)) / rank
    return float(np.sqrt(max(0.0, min(1.0, squared))))


def _should_use_torch(compute_backend: str, device: str) -> bool:
    try:
        from .torch_backend import should_use_torch
    except RuntimeError:
        return False
    return should_use_torch(compute_backend, device)
