"""One source of truth for the normal-state detector's frozen policy."""

from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class OursParameters:
    detector_subspace_dim: int = 2
    detector_normal_clusters: int = 2
    detector_distance_threshold: float = 2.5
    detector_reject_threshold: float = 5.0
    detector_drift_memory: float = 0.8
    detector_drift_allowance: float = 2.0
    detector_drift_threshold: float = 6.0
    detector_history_confirm: int = 2
    detector_history_threshold: float = 1.75
    detector_recovery_confirm: int = 2
    detector_reference_budget: float = 3.0
    detector_clip_factor: float = 2.0
    detector_weight_cap: float = 2.0
    suspicion_penalty_factor: float = 0.5
    suspicion_recovery_factor: float = 1.25
    suspicion_remove_after: int = 3

    @classmethod
    def from_object(cls, obj):
        return cls(**{f.name: getattr(obj, f.name) for f in fields(cls)})

    def validate(self):
        for item in fields(OursParameters):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{item.name} must be numeric")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{item.name} must be finite and positive")
            if isinstance(getattr(OursParameters(), item.name), int) and not isinstance(value, int):
                raise ValueError(f"{item.name} must be an integer")
        if not 1 <= self.detector_subspace_dim < 10:
            raise ValueError("detector_subspace_dim must be in [1, 9]")
        if not 1 <= self.detector_normal_clusters <= 4:
            raise ValueError("detector_normal_clusters must be in [1, 4]")
        if self.detector_reject_threshold <= self.detector_distance_threshold:
            raise ValueError("detector_reject_threshold must exceed detector_distance_threshold")
        if self.detector_history_threshold >= self.detector_distance_threshold:
            raise ValueError("history threshold must be below warning threshold")
        if self.detector_drift_allowance > self.detector_distance_threshold:
            raise ValueError("drift allowance must not exceed warning threshold")
        if not 0 < self.detector_drift_memory < 1:
            raise ValueError("detector_drift_memory must be in (0, 1)")
        if not 0 < self.suspicion_penalty_factor < 1:
            raise ValueError("suspicion_penalty_factor must be in (0, 1)")
        if self.suspicion_recovery_factor <= 1:
            raise ValueError("suspicion_recovery_factor must exceed 1")
        if min(self.detector_reference_budget, self.detector_clip_factor, self.detector_weight_cap) < 1:
            raise ValueError("reference budget, clipping factor and weight cap must be >= 1")


OURS_PARAMETER_NAMES = tuple(f.name for f in fields(OursParameters))


def bounded_candidates(budget: int) -> tuple[dict, ...]:
    """Public, deterministic, bounded designs; no labels or test outcomes.

    The first four designs hold the measured balanced detector fixed while
    varying C_tol=2/3/1/5. The rest retain broad q/threshold/drift coverage.
    This space is informed by development holdout experiments, never formal
    test outcomes. Normal centres,
    feature scales and radii are learned afresh from each run's clean prefix.
    Hyperparameters are selected by the shared validation budget, not these
    defaults. Twelve designs are deliberately not an exhaustive grid.
    """
    if not 1 <= budget <= 36:
        raise ValueError("calibration_candidate_budget must be in [1, 36]")
    candidates = []
    for i in range(budget):
        p = OursParameters(
            detector_subspace_dim=1 + i % 3,
            detector_normal_clusters=1 + (i // 3) % 2,
            detector_distance_threshold=(2.0, 2.5, 3.0, 3.5)[(i // 3) % 4],
            detector_reject_threshold=(4.0, 5.0, 6.0, 7.0)[(i // 3) % 4],
            detector_drift_memory=(0.8, 0.9)[i % 2],
            detector_drift_allowance=(1.75, 2.0, 2.5, 3.0)[(i // 3) % 4],
            detector_drift_threshold=(4.0, 6.0, 8.0)[(i // 2) % 3],
            detector_history_confirm=(2, 3)[i % 2],
            detector_history_threshold=(1.5, 1.75, 2.0, 2.5)[(i // 3) % 4],
            detector_recovery_confirm=(2, 3)[i % 2],
            detector_reference_budget=(2.5, 3.0, 3.5)[i % 3],
            suspicion_remove_after=(1, 3, 5)[(i + i // 4) % 3],
            suspicion_penalty_factor=(0.25, 0.5, 0.75)[(i // 3 + i) % 3],
            suspicion_recovery_factor=(1.1, 1.25, 1.5)[(i // 2 + i) % 3],
            detector_clip_factor=(1.5, 2.0, 3.0)[(i // 4) % 3],
            detector_weight_cap=(1.5, 2.0, 2.5)[(i // 4) % 3],
        )
        if i < 4:
            p = OursParameters(
                detector_distance_threshold=3.0,
                detector_reject_threshold=6.0,
                detector_drift_allowance=2.5,
                detector_history_threshold=2.0,
                detector_reference_budget=3.5,
                suspicion_remove_after=(2, 3, 1, 5)[i],
            )
        p.validate()
        candidates.append(asdict(p))
    return tuple(candidates)
