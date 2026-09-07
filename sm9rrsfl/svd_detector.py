"""Per-client SVD normal-state clustering, with two-phase trusted admission.

K-means fits *normal modes*, never a malicious/benign partition. All decision
inputs are server-observable; experiment labels and malicious ratios are absent.
The clean first K rounds are an explicit assumption, not inferred from labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import numpy as np

from .ours_policy import OursParameters


@dataclass(frozen=True)
class DetectionResult:
    accepted: bool
    reason: str
    would_flag: bool = False
    count_increment: bool = False
    novelty_score: float = 0.0
    anchor_score: float = 0.0
    signed_score: float = 0.0
    class_score: float = 0.0
    cumulative_drift: float = 0.0
    clip_factor: float = 1.0
    history_eligible: bool = False
    trusted_history_size: int = 0
    normal_cluster_count: int = 0
    immediate_revocation: bool = False
    recovery_eligible: bool = False
    norm_score: float = 0.0


@dataclass
class _NormalModel:
    location: np.ndarray
    scale: np.ndarray
    centers: np.ndarray
    radii: np.ndarray


@dataclass
class _TagState:
    history: deque = field(default_factory=deque)
    anchor: _NormalModel | None = None
    normal: _NormalModel | None = None
    norm_limit: float = 0.0
    drift: float = 0.0
    clean_streak: int = 0
    recovery_streak: int = 0
    last_round: int = 0
    pending: tuple | None = None


def _fit_normal(features, clusters, groups=None, reference=None, previous=None):
    """Deterministic farthest-point K-means, <=2 modes for short histories.

    At least three observations per mode; a singleton cannot legitimize a
    novel state. Radius floors regularize small K; they are not a claimed
    statistical false-positive guarantee.
    """
    x = np.asarray(features, dtype=np.float64)
    location = np.median(x, axis=0) if reference is None else reference.location.copy()
    scale = (np.maximum(1.4826 * np.median(np.abs(x - location), axis=0), 0.15)
             if reference is None else reference.scale.copy())
    groups = groups or (slice(0, x.shape[1]),)
    z = (x - location) / scale
    count = min(clusters, max(1, len(x) // 3))
    centers = [z[np.argmin(np.sum(z * z, axis=1))]]
    for _ in range(1, count):
        distances = np.min(np.sum((z[:, None] - np.asarray(centers)) ** 2, axis=2), axis=1)
        if distances.max() < 1e-10:
            break
        centers.append(z[int(np.argmax(distances))])
    centers = np.asarray(centers)
    for _ in range(20):
        labels = np.argmin(np.sum((z[:, None] - centers) ** 2, axis=2), axis=1)
        if any(np.count_nonzero(labels == j) < 3 for j in range(len(centers))):
            centers = z.mean(axis=0, keepdims=True)
            break
        updated = np.asarray([z[labels == j].mean(axis=0) for j in range(len(centers))])
        if np.allclose(updated, centers, rtol=0, atol=1e-6):
            centers = updated
            break
        centers = updated
    if previous is not None:
        # Fixed clean coordinates prevent a rolling window from collapsing its
        # scales. Limit each admitted observation's movement to 0.25 clean
        # block radii; the immutable anchor separately bounds total drift.
        for j, center in enumerate(centers):
            old = previous.centers[np.argmin(np.mean((previous.centers - center) ** 2, axis=1))]
            movement = max(np.sqrt(np.mean((center[g] - old[g]) ** 2)) /
                           max(1., float(reference.radii[:, k].max()))
                           for k, g in enumerate(groups))
            centers[j] = old + (center - old) * min(1., .25 / max(movement, 1e-12))
    distances = np.stack([
        np.sqrt(np.mean((z[:, None, g] - centers[None, :, g]) ** 2, axis=2))
        for g in groups], axis=2)
    labels = np.argmin(distances.max(axis=2), axis=1)
    radii = np.asarray([np.maximum(1., distances[labels == j, j].max(axis=0, initial=0.))
                        for j in range(len(centers))])
    if reference is not None:
        radii = np.maximum(radii, reference.radii.max(axis=0))
    return _NormalModel(location, scale, centers, radii)


class LongitudinalSVDDetector:
    """Compact CPU detector; expensive local training remains on assigned GPU."""

    def __init__(self, *, window_size=7, policy=None, num_classes=10,
                 expected_update_size=None, matrix_offset=None, matrix_shape=None):
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 3:
            raise ValueError("window_size must be an integer >= 3")
        self.policy = policy or OursParameters()
        self.policy.validate()
        if self.policy.detector_subspace_dim >= num_classes:
            raise ValueError("q must be smaller than num_classes")
        self.window_size = window_size
        self.num_classes = num_classes
        self.expected_update_size = expected_update_size
        self.matrix_offset = matrix_offset
        self.matrix_shape = matrix_shape
        if (matrix_offset is None) != (matrix_shape is None):
            raise ValueError("classifier offset and shape must be specified together")
        self._states = {}
        self._sketch_signs = None
        self._groups = None

    def _extract(self, update, learning_rate):
        vector = np.asarray(update, dtype=np.float64)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError("update must be a finite non-empty vector")
        if self.expected_update_size is not None and vector.size != self.expected_update_size:
            raise ValueError("unexpected update size")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        vector = vector / learning_rate
        if not np.isfinite(vector).all():
            raise ValueError("learning-rate-normalized update is not finite")
        # Stable scaling before the small Gram eigendecomposition. Spectral
        # evidence covers the WHOLE model, not only its final classifier.
        peak = max(float(np.max(np.abs(vector))), 1e-30)
        scaled = vector / peak
        norm = float(np.linalg.norm(scaled)) * peak
        if not np.isfinite(norm):
            raise ValueError("update norm is not finite")
        unit = scaled / max(float(np.linalg.norm(scaled)), 1e-30)
        padded = np.pad(unit, (0, (-len(unit)) % self.num_classes))
        matrix = padded.reshape(-1, self.num_classes)
        eig, basis = np.linalg.eigh(matrix.T @ matrix)
        order = np.argsort(eig)[::-1]
        q = self.policy.detector_subspace_dim
        spectrum = np.sqrt(np.maximum(eig[order[:q]], 0.0))
        # Compact right projector avoids storing P x q left bases per client.
        right = basis[:, order[:q]]
        projector = right @ right.T
        if self._sketch_signs is None:
            self._sketch_signs = np.random.default_rng(73129).choice(
                [-1.0, 1.0], size=len(unit))
        if len(unit) != len(self._sketch_signs):
            raise ValueError("update size changed during a task")
        sketch = np.pad(unit * self._sketch_signs, (0, (-len(unit)) % 32))
        sketch = sketch.reshape(-1, 32).sum(axis=0)
        sketch /= max(float(np.linalg.norm(sketch)), 1e-12)
        sketch *= np.sqrt(len(sketch))
        classes = np.zeros(self.num_classes * 2)
        if self.matrix_shape is not None:
            start = self.matrix_offset
            size = int(np.prod(self.matrix_shape))
            if start < 0 or start + size + self.num_classes > len(vector):
                raise ValueError("classifier slice exceeds update size")
            classifier = vector[start:start + size].reshape(self.matrix_shape)
            # Actual model layout, class-sensitive signed weight and bias
            # changes; no knowledge of the attack's source/target labels.
            class_weights = classifier.mean(axis=0)
            bias = vector[start + size:start + size + self.num_classes]
            classes = np.concatenate([class_weights, bias])
            classes /= max(float(np.linalg.norm(classes)), 1e-12)
            classes *= np.sqrt(len(classes))
        spectral = np.concatenate([[np.log(max(norm, 1e-30))], spectrum, projector.ravel()])
        feature = np.concatenate([spectral, sketch, classes])
        a, b = len(spectral), len(spectral) + len(sketch)
        # Norm gets an independent block: one amplitude coordinate must not
        # be diluted by a hundred projector entries.
        self._groups = (slice(1, a), slice(a, b), slice(b, len(feature)), slice(0, 1))
        return feature, norm

    def _scores(self, model, feature, *, norm_score=None):
        z = (feature - model.location) / model.scale
        differences = z[None] - model.centers
        group_distances = np.column_stack([
            np.sqrt(np.mean(differences[:, group] ** 2, axis=1)) / model.radii[:, k]
            for k, group in enumerate(self._groups)
        ])
        # Decreasing gradient norms during convergence are not norm attacks.
        group_distances[:, -1] = (np.maximum(differences[:, 0], 0.) / model.radii[:, -1]
                                 if norm_score is None else norm_score)
        # Use one normal mode for ALL evidence blocks, not a different mode
        # per block (which could manufacture an unobserved composite state).
        selected = int(np.argmin(group_distances.max(axis=1)))
        return group_distances[selected]

    def evaluate(self, tag, update, *, round_id=None, learning_rate=1.0):
        state = self._states.setdefault(tag, _TagState())
        if state.pending is not None:
            raise RuntimeError("commit the previous decision before evaluating again")
        current = state.last_round + 1 if round_id is None else round_id
        if isinstance(current, bool) or not isinstance(current, int) or current <= state.last_round:
            raise ValueError("round_id must increase")
        feature, norm = self._extract(update, learning_rate)
        p = self.policy
        if current <= self.window_size:
            # No late-joining tag can obtain a fresh trusted warm-up after K.
            decision = DetectionResult(True, "clean_warmup", history_eligible=True,
                                       trusted_history_size=len(state.history))
        elif state.anchor is None:
            decision = DetectionResult(False, "insufficient_clean_history", True)
        else:
            # A live center tracks convergence toward small gradients. A
            # later benign rebound must not become a spurious amplitude
            # attack; only growth above the immutable clean maximum counts.
            norm_scale = state.anchor.scale[0] * float(state.anchor.radii[:, -1].max())
            norm_score = max(0., np.log(max(norm, 1e-30))
                             - np.log(max(state.norm_limit, 1e-30))) / norm_scale
            live = self._scores(state.normal, feature, norm_score=norm_score)
            anchor = self._scores(state.anchor, feature, norm_score=norm_score)
            score = max(float(live.max()), float(anchor.max()) / p.detector_reference_budget)
            # Accumulate excess over a calibrated normal envelope, not the
            # raw distance from an early-training center.
            state.drift = max(0.0, p.detector_drift_memory * state.drift
                              + min(score, p.detector_reject_threshold)
                              - p.detector_drift_allowance)
            flagged = score > p.detector_distance_threshold or state.drift > p.detector_drift_threshold
            strong = score > p.detector_reject_threshold
            # Aggressive policy: every suspicious observation counts; a
            # severe deviation requests same-round certificate-gated removal,
            # without waiting for C_tol or a separate corroboration condition.
            clip = min(1.0, state.norm_limit * p.detector_clip_factor / max(norm, 1e-30))
            safe = (not flagged and clip == 1.0 and score <= p.detector_history_threshold
                    and float(anchor.max()) <= p.detector_reference_budget
                    and state.drift <= p.detector_drift_allowance)
            state.clean_streak = state.clean_streak + 1 if safe else 0
            eligible = safe and state.clean_streak >= p.detector_history_confirm
            recoverable = (not flagged and clip == 1.0
                           and score <= p.detector_drift_allowance
                           and state.drift <= p.detector_drift_allowance)
            state.recovery_streak = state.recovery_streak + 1 if recoverable else 0
            decision = DetectionResult(
                accepted=not flagged,
                reason="strong_novelty" if strong else "suspicious" if flagged else "normal",
                would_flag=flagged, count_increment=flagged,
                novelty_score=score, anchor_score=float(anchor.max()),
                signed_score=float(live[1]), class_score=float(live[2]),
                cumulative_drift=state.drift, clip_factor=clip,
                history_eligible=eligible, trusted_history_size=len(state.history),
                normal_cluster_count=len(state.normal.centers),
                immediate_revocation=strong,
                recovery_eligible=state.recovery_streak >= p.detector_recovery_confirm,
                norm_score=float(live[3]))
        state.pending = (current, feature, norm, decision)
        return decision

    def commit(self, tag, *, admit_history):
        """Called AFTER aggregation/revocation guards; no rejected data refit."""
        state = self._states[tag]
        if state.pending is None:
            raise RuntimeError("no pending observation")
        current, feature, norm, decision = state.pending
        admitted = bool(admit_history and decision.history_eligible and decision.accepted)
        if admitted:
            state.history.append(feature)
            while len(state.history) > self.window_size:
                state.history.popleft()
            if state.anchor is None:
                state.norm_limit = max(state.norm_limit, norm)
                if current == self.window_size and len(state.history) >= 3:
                    state.anchor = _fit_normal(state.history, self.policy.detector_normal_clusters, self._groups)
                    state.normal = state.anchor
            elif len(state.history) >= 3:
                state.normal = _fit_normal(state.history, self.policy.detector_normal_clusters,
                                          self._groups, state.anchor, state.normal)
            # Anchor, scaling budget and norm limit NEVER learn from attack-era
            # observations, even if those observations are allowed to aggregate.
        elif not admit_history:
            state.clean_streak = 0
        state.last_round = current
        state.pending = None
        return admitted

    def forget(self, tag):
        self._states.pop(tag, None)

    def memory_bytes(self):
        arrays = {}
        for state in self._states.values():
            arrays.update({id(x): x for x in state.history})
            for model in (state.anchor, state.normal):
                if model is not None:
                    arrays.update({id(x): x for x in
                                   (model.location, model.scale, model.centers, model.radii)})
        return sum(x.nbytes for x in arrays.values()) + (
            0 if self._sketch_signs is None else self._sketch_signs.nbytes)
