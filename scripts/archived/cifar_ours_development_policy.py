"""Isolated Ours development policy; importing this module changes no defaults.

The extra gate excludes a weak warning for the current round. It does not turn
that warning into permanent reliability loss or revocation evidence. Original
003 novelty/drift alarms, severe removal, and global shock handling remain in
the underlying implementation. Use only through the development runner.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import math

from sm9rrsfl import fl
from sm9rrsfl.svd_detector import LongitudinalSVDDetector
from sm9rrsfl.weighting import SuspicionWeightManager


@dataclass
class _PolicyState:
    threshold: float
    weak_tags: set = field(default_factory=set)


_active_policy = None


def _policy_for_new_instance():
    if _active_policy is None:
        raise RuntimeError("development classes require weak_quarantine_policy()")
    return _active_policy


class WeakQuarantineDetector(LongitudinalSVDDetector):
    """Keep 003 scores/calibration; quarantine only otherwise-normal updates."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.development_policy = _policy_for_new_instance()
        threshold = self.development_policy.threshold
        if not (max(self.policy.detector_history_threshold,
                    self.policy.detector_drift_allowance) <= threshold
                < self.policy.detector_distance_threshold):
            raise ValueError("weak threshold must separate trusted recovery from strong warnings")

    def evaluate(self, tag, update, *, round_id=None, learning_rate=1.0):
        decision = super().evaluate(tag, update, round_id=round_id,
                                    learning_rate=learning_rate)
        weak = (decision.reason == "normal" and decision.accepted
                and not decision.would_flag and not decision.count_increment
                and not decision.immediate_revocation
                and decision.novelty_score > self.development_policy.threshold)
        if weak:
            decision = replace(
                decision, accepted=False, would_flag=True,
                reason="development_weak_quarantine", count_increment=False,
                history_eligible=False, recovery_eligible=False)
            state = self._states[tag]
            # commit() reads the saved decision, not just the returned value.
            current, feature, norm, _ = state.pending
            state.pending = (current, feature, norm, decision)
            state.clean_streak = 0
            state.recovery_streak = 0
            self.development_policy.weak_tags.add(tag)
        else:
            self.development_policy.weak_tags.discard(tag)
        return decision

    def forget(self, tag):
        self.development_policy.weak_tags.discard(tag)
        super().forget(tag)


class WeakQuarantineWeightManager(SuspicionWeightManager):
    def __init__(self, tag_ids=(), **kwargs):
        super().__init__(tag_ids, **kwargs)
        # Shared with the detector, including after checkpoint unpickling.
        self.development_policy = _policy_for_new_instance()

    def update(self, active_tags, suspicious_tags, count_increment_tags, *,
               immediate_revocation_tags=None, recovery_tags=None):
        suspicious = set(suspicious_tags)
        weak = suspicious & self.development_policy.weak_tags
        counted = set(count_increment_tags)
        immediate = set(immediate_revocation_tags or ())
        if weak & (counted | immediate):
            raise RuntimeError("weak quarantine cannot override strong revocation evidence")
        # Removing only explicitly weak tags preserves uncalibrated rejection
        # handling. The base manager naturally halves their old evidence and
        # leaves reliability unchanged. The base 003 shock rule sees the same
        # original alarms; weak warnings cannot freeze other clients' history.
        result = super().update(
            active_tags, suspicious - weak, counted,
            immediate_revocation_tags=immediate,
            recovery_tags=set(recovery_tags or ()) - weak)
        self.development_policy.weak_tags.clear()
        return replace(result, suspicious_tags=suspicious)


@contextmanager
def weak_quarantine_policy(threshold=1.25):
    """Patch only the two Ours constructors for one sequential worker process.

    Classes are module-level so ordinary production round checkpoints can
    pickle them. A checkpoint contains both instances and their shared state;
    resuming therefore does not depend on an old process's global variables.
    """
    global _active_policy
    if isinstance(threshold, bool) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    if _active_policy is not None:
        raise RuntimeError("nested development policy contexts are unsupported")
    if (fl.LongitudinalSVDDetector is not LongitudinalSVDDetector
            or fl.SuspicionWeightManager is not SuspicionWeightManager):
        raise RuntimeError("Ours constructors are already overridden")
    _active_policy = _PolicyState(float(threshold))
    fl.LongitudinalSVDDetector = WeakQuarantineDetector
    fl.SuspicionWeightManager = WeakQuarantineWeightManager
    try:
        yield
    finally:
        fl.LongitudinalSVDDetector = LongitudinalSVDDetector
        fl.SuspicionWeightManager = SuspicionWeightManager
        _active_policy = None
