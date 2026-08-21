"""Task-tag keyed dynamic weights from Word Section 4.3.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class WeightUpdateResult:
    weights: dict[str, float]
    pre_normalization_weights: dict[str, float]
    suspicious_tags: set[str]
    count_increment_tags: set[str]
    trace_requested_tags: set[str]


class SuspicionWeightManager:
    """Maintain ``w_pi`` and ``Count_pi`` by opaque ``Tag_pi``.

    Reaching ``C_tol`` creates a trace request and assigns zero weight to the
    trigger-round update.  Permanent removal happens only after AS validates a
    D-KGC threshold trace certificate via :meth:`confirm_revocation`.
    """

    def __init__(
        self,
        tag_ids: Iterable[str] = (),
        *,
        participant_count: int | None = None,
        penalty_factor: float = 0.5,
        recovery_factor: float = 2.0,
        remove_after: int = 3,
        max_count: int | None = None,
    ) -> None:
        initial = tuple(dict.fromkeys(str(tag) for tag in tag_ids))
        if participant_count is None:
            participant_count = len(initial)
        if participant_count < 1:
            raise ValueError("participant_count must be positive")
        if not 0.0 < penalty_factor < 1.0:
            raise ValueError("penalty_factor must be in (0, 1)")
        if recovery_factor <= 1.0:
            raise ValueError("recovery_factor must be greater than 1")
        if remove_after < 1:
            raise ValueError("remove_after must be positive")
        if max_count is None:
            max_count = remove_after
        if max_count < remove_after:
            raise ValueError("max_count must be at least remove_after")
        self.participant_count = participant_count
        self.penalty_factor = penalty_factor
        self.recovery_factor = recovery_factor
        self.remove_after = remove_after
        self.max_count = max_count
        initial_weight = 1.0 / participant_count
        self.weights = {tag: initial_weight for tag in initial}
        # Keep the historical attribute name for checkpoint compatibility.
        # Values are non-negative integer anomaly-evidence scores: a composite
        # R=1 round adds one up to C_max, while R=0 floor-halves the evidence.
        self.consecutive_suspicions = {tag: 0 for tag in initial}
        self.pending_trace: set[str] = set()
        self.revoked: set[str] = set()

    def update(
        self,
        active_tags: list[str],
        suspicious_tags: set[str],
        count_increment_tags: set[str],
    ) -> WeightUpdateResult:
        """Apply the paper's penalty/recovery/count equations for one round.

        The third scheme revision has one composite decision ``R_pi``.
        Consequently ``suspicious_tags`` and ``count_increment_tags`` must be
        identical: R=1 both downweights and increments ``Count_pi``; R=0 both
        recovers the weight and replaces Count with ``floor(Count / 2)``.
        """

        active = list(dict.fromkeys(active_tags))
        if not active:
            return WeightUpdateResult(
                dict(self.weights),
                dict(self.weights),
                set(),
                set(),
                set(),
            )
        if count_increment_tags != suspicious_tags:
            raise ValueError(
                "count_increment_tags must equal suspicious_tags for composite R"
            )
        uniform_weight = 1.0 / len(active)
        trace_requested: set[str] = set()

        for tag in active:
            self.weights.setdefault(tag, uniform_weight)
            self.consecutive_suspicions.setdefault(tag, 0)
            if tag in self.revoked:
                self.weights[tag] = 0.0
                continue
            if tag in suspicious_tags:
                self.weights[tag] *= self.penalty_factor
            else:
                if self.weights[tag] < uniform_weight:
                    self.weights[tag] = min(
                        self.recovery_factor * self.weights[tag],
                        uniform_weight,
                    )

            if tag in count_increment_tags:
                self.consecutive_suspicions[tag] = min(
                    self.max_count,
                    self.consecutive_suspicions[tag] + 1,
                )
            else:
                # A normal round weakens rather than erases historical anomaly
                # evidence, preventing one benign-looking update from clearing
                # a persistent attack trajectory.
                self.consecutive_suspicions[tag] //= 2

            if self.consecutive_suspicions[tag] >= self.remove_after:
                if tag not in self.pending_trace:
                    self.pending_trace.add(tag)
                    trace_requested.add(tag)
                # The trigger-round update is rejected while D-KGC tracing is
                # pending, but this is not yet a permanent revocation.
                self.weights[tag] = 0.0

        pre_normalization_weights = {
            tag: self.weights.get(tag, 0.0) for tag in active
        }
        self._renormalize(active)
        return WeightUpdateResult(
            weights=dict(self.weights),
            pre_normalization_weights=pre_normalization_weights,
            suspicious_tags=set(suspicious_tags),
            count_increment_tags=set(count_increment_tags),
            trace_requested_tags=trace_requested,
        )

    def confirm_revocation(self, tag: str) -> None:
        """Permanently retire a tag only after Equation (7) was accepted."""

        if tag not in self.pending_trace:
            raise ValueError("tag has no pending trace request")
        self.pending_trace.remove(tag)
        self.revoked.add(tag)
        self.weights[tag] = 0.0

    def _renormalize(self, active_tags: list[str]) -> None:
        eligible = [tag for tag in active_tags if tag not in self.revoked]
        total = sum(self.weights.get(tag, 0.0) for tag in eligible)
        if total > 0.0:
            for tag in eligible:
                self.weights[tag] /= total
        for tag in self.revoked:
            self.weights[tag] = 0.0
