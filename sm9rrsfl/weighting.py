"""Task-tag keyed dynamic weights from Word Section 4.3.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class WeightUpdateResult:
    weights: dict[str, float]
    suspicious_tags: set[str]
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
        self.participant_count = participant_count
        self.penalty_factor = penalty_factor
        self.recovery_factor = recovery_factor
        self.remove_after = remove_after
        initial_weight = 1.0 / participant_count
        self.weights = {tag: initial_weight for tag in initial}
        self.consecutive_suspicions = {tag: 0 for tag in initial}
        self.pending_trace: set[str] = set()
        self.revoked: set[str] = set()

    def update(
        self,
        active_tags: list[str],
        suspicious_tags: set[str],
    ) -> WeightUpdateResult:
        """Apply the paper's penalty/recovery/count equations for one round."""

        active = list(dict.fromkeys(active_tags))
        if not active:
            return WeightUpdateResult(dict(self.weights), set(), set())
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
                self.consecutive_suspicions[tag] += 1
            else:
                if self.weights[tag] < uniform_weight:
                    self.weights[tag] = min(
                        self.recovery_factor * self.weights[tag],
                        uniform_weight,
                    )
                self.consecutive_suspicions[tag] = 0

            if self.consecutive_suspicions[tag] >= self.remove_after:
                if tag not in self.pending_trace:
                    self.pending_trace.add(tag)
                    trace_requested.add(tag)
                # The trigger-round update is rejected while D-KGC tracing is
                # pending, but this is not yet a permanent revocation.
                self.weights[tag] = 0.0

        self._renormalize(active)
        return WeightUpdateResult(
            weights=dict(self.weights),
            suspicious_tags=set(suspicious_tags),
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
