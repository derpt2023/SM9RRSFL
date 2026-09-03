"""Bounded reliability and aggressive, certificate-gated revocation."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WeightUpdateResult:
    weights: dict
    reliability_after_update: dict
    suspicious_tags: set
    count_increment_tags: set
    trace_requested_tags: set
    history_frozen: bool


class SuspicionWeightManager:
    """Reliability stays in [0,1]; this class NEVER redistributes lost mass."""

    def __init__(self, tag_ids=(), *,
                 penalty_factor=0.1, recovery_factor=1.25, remove_after=3):
        initial = tuple(dict.fromkeys(str(tag) for tag in tag_ids))
        if not math.isfinite(penalty_factor) or not 0 < penalty_factor < 1:
            raise ValueError("penalty_factor must be in (0,1)")
        if not math.isfinite(recovery_factor) or recovery_factor <= 1:
            raise ValueError("recovery_factor must be greater than 1")
        if isinstance(remove_after, bool) or not isinstance(remove_after, int) or remove_after < 1:
            raise ValueError("remove_after must be a positive integer")
        self.penalty_factor = penalty_factor
        self.recovery_factor = recovery_factor
        self.remove_after = remove_after
        self.weights = dict.fromkeys(initial, 1.0)
        self.evidence_counts = dict.fromkeys(initial, 0.0)
        self.pending_trace = set()
        self.revoked = set()
        self.previous_suspicious = set()

    def update(self, active_tags, suspicious_tags, count_increment_tags, *,
               immediate_revocation_tags=None, recovery_tags=None):
        active = list(dict.fromkeys(active_tags))
        immediate = set() if immediate_revocation_tags is None else set(immediate_revocation_tags)
        if not immediate <= count_increment_tags <= suspicious_tags <= set(active):
            raise ValueError("immediate revocation must be a subset of counted suspicious active tags")
        recovery_tags = set() if recovery_tags is None else set(recovery_tags)
        if recovery_tags & suspicious_tags or not recovery_tags <= set(active):
            raise ValueError("recovery requires a normal active tag")
        eligible = [tag for tag in active if tag not in self.revoked and tag not in self.pending_trace]
        newly_suspicious = (suspicious_tags & set(eligible)) - self.previous_suspicious
        # A distribution shock freezes trusted history and reliability
        # recovery only. It must NEVER defer either revocation path.
        shock = bool(eligible and len(newly_suspicious) > 0.5 * len(eligible))
        self.previous_suspicious = set(suspicious_tags)
        for tag in eligible:
            self.weights.setdefault(tag, 1.0)
            self.evidence_counts.setdefault(tag, 0.0)
            if tag in suspicious_tags:
                self.weights[tag] = max(1e-12, self.weights[tag] * self.penalty_factor)
            elif tag in recovery_tags and not shock:
                self.weights[tag] = min(1.0, self.weights[tag] * self.recovery_factor)
            if tag in count_increment_tags:
                self.evidence_counts[tag] = min(
                    float(self.remove_after), self.evidence_counts[tag] + 1.0)
            elif tag not in suspicious_tags:
                # Each observed normal round halves the suspicion value,
                # independently of the stricter trusted-history/recovery gate.
                # Missing observations or uncalibrated rejections are NOT normal.
                self.evidence_counts[tag] *= 0.5
        requested = {
            tag for tag in eligible
            if tag in immediate or (
                tag in count_increment_tags and self.evidence_counts[tag] >= self.remove_after)
        }
        # No per-round quota or minimum-survivor exception: severe evidence
        # bypasses C_tol; an update that crosses C_tol is also excluded now.
        self.pending_trace.update(requested)
        effective = {
            tag: (0.0 if tag in self.pending_trace or tag in self.revoked
                  else self.weights.get(tag, 1.0)) for tag in active
        }
        return WeightUpdateResult(effective, dict(self.weights), set(suspicious_tags),
                                  set(count_increment_tags), requested, shock)

    def confirm_revocation(self, tag):
        if tag not in self.pending_trace:
            raise ValueError("tag has no pending trace request")
        self.pending_trace.remove(tag)
        self.revoked.add(tag)
        self.weights[tag] = 0.0


def bounded_aggregation_coefficients(tags, nominal_weights, reliability,
                                     decisions, weight_cap):
    """Sample-proportional, capped, clipped coefficients, without renormalizing.

    Missing mass means a smaller server step, not an invitation to amplify the
    few survivors. All-zero input yields a zero step (never uniform fallback).
    """
    raw = {tag: nominal_weights[tag] * reliability.get(tag, 0.0)
           if decisions[tag].accepted else 0.0 for tag in tags}
    total = sum(raw.values())
    if total <= 0:
        return dict.fromkeys(tags, 0.0)
    return {tag: min(raw[tag] / total, weight_cap * raw[tag])
            * decisions[tag].clip_factor for tag in tags}
