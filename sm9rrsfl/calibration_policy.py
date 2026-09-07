"""Shared, auditable policy helpers for offline defense calibration.

The helpers in this module deliberately contain no model-training code.  They
define the public ratio protocol, hard-constraint defaults, and deterministic
objective-weight search used by both the Ours calibrator and the unified fair
tuner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import math
from typing import Iterable, Mapping, Sequence


DEFAULT_MIN_ROUND_COMPLETION_RATE = 1.0
DEFAULT_MAX_NONFINITE_UPDATES = 0
DEFAULT_OBJECTIVE_WEIGHT_FLOOR = 0.05
DEFAULT_OBJECTIVE_WEIGHT_STEP = 0.05
TRAINING_HEALTH_CONSTRAINTS = {
    "max_clean_false_revocation_rate": 0.10,
    "reject_any_all_clients_revoked": True,
    "reject_any_all_honest_revoked": True,
    "ours_starvation_honest_weight_loss": 0.99,
    "ours_starvation_consecutive_rounds": 5,
}

OBJECTIVE_WEIGHT_NAMES = (
    "clean_accuracy_weight",
    "robust_accuracy_weight",
    "attack_success_weight",
    "honest_weight_loss_weight",
)


@dataclass(frozen=True)
class RatioSchedule:
    """Formal ratios and disjoint attacked calibration midpoints."""

    minimum: float
    maximum: float
    count: int
    formal_ratios: tuple[float, ...]
    calibration_ratios: tuple[float, ...]
    policy: str = "formal_endpoints_and_calibration_midpoints"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationHardConstraints:
    """Method-neutral execution-integrity envelope for calibration runs."""

    min_round_completion_rate: float = DEFAULT_MIN_ROUND_COMPLETION_RATE
    max_nonfinite_updates: int = DEFAULT_MAX_NONFINITE_UPDATES

    def validate(self) -> "CalibrationHardConstraints":
        if (
            not math.isfinite(self.min_round_completion_rate)
            or not 0.0 <= self.min_round_completion_rate <= 1.0
        ):
            raise ValueError(
                "min_round_completion_rate must be finite and in [0, 1]"
            )
        if (
            isinstance(self.max_nonfinite_updates, bool)
            or not isinstance(self.max_nonfinite_updates, int)
            or self.max_nonfinite_updates < 0
        ):
            raise ValueError("max_nonfinite_updates must be a non-negative integer")
        return self

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_ratio_schedule(values: Sequence[float]) -> RatioSchedule:
    """Build the confirmed ``[minimum, maximum, count]`` ratio protocol.

    Formal ratios include both endpoints.  Calibration keeps the mandatory
    0% clean control and places every attacked calibration ratio at a midpoint
    between neighboring formal ratios.  Thus ``[0, .8, 5]`` becomes formal
    ``[0, .2, .4, .6, .8]`` and calibration ``[0, .1, .3, .5, .7]``.
    """

    if len(values) != 3:
        raise ValueError("ratio_range must contain [minimum, maximum, count]")
    minimum = float(values[0])
    maximum = float(values[1])
    raw_count = float(values[2])
    if not all(math.isfinite(value) for value in (minimum, maximum, raw_count)):
        raise ValueError("ratio_range values must be finite")
    count = int(raw_count)
    if raw_count != count or count < 2:
        raise ValueError("ratio_range count must be an integer of at least 2")
    if abs(minimum) > 1.0e-12:
        raise ValueError("ratio_range minimum must be 0 for the clean control")
    if not 0.0 < maximum < 1.0:
        raise ValueError("ratio_range maximum must be in (0, 1)")
    step = (maximum - minimum) / (count - 1)
    formal = tuple(_stable_ratio(minimum + index * step) for index in range(count))
    midpoints = tuple(
        _stable_ratio((left + right) / 2.0)
        for left, right in zip(formal, formal[1:])
    )
    calibration = (0.0, *midpoints)
    if len(calibration) != count:
        raise AssertionError("ratio schedule cardinality mismatch")
    attacked_overlap = {
        value for value in formal if value > 0.0
    } & {value for value in calibration if value > 0.0}
    if attacked_overlap:
        raise AssertionError("attacked formal and calibration ratios must be disjoint")
    return RatioSchedule(
        minimum=minimum,
        maximum=maximum,
        count=count,
        formal_ratios=formal,
        calibration_ratios=calibration,
    )


def objective_weight_grid(
    *,
    floor: float = DEFAULT_OBJECTIVE_WEIGHT_FLOOR,
    step: float = DEFAULT_OBJECTIVE_WEIGHT_STEP,
) -> tuple[dict[str, float], ...]:
    """Enumerate a deterministic positive simplex for learned Score weights."""

    if not math.isfinite(floor) or not 0.0 <= floor < 0.25:
        raise ValueError("objective weight floor must be in [0, 0.25)")
    if not math.isfinite(step) or not 0.0 < step <= 1.0:
        raise ValueError("objective weight step must be in (0, 1]")
    units = round(1.0 / step)
    floor_units = math.ceil((floor - 1.0e-12) / step)
    if not math.isclose(units * step, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("objective weight step must divide 1 exactly")
    if 4 * floor_units > units:
        raise ValueError("objective weight floor is incompatible with the step")
    candidates: list[dict[str, float]] = []
    admissible = range(floor_units, units + 1)
    for first, second, third in product(admissible, repeat=3):
        fourth = units - first - second - third
        if fourth < floor_units:
            continue
        values = (first, second, third, fourth)
        candidates.append(
            {
                name: float(value * step)
                for name, value in zip(OBJECTIVE_WEIGHT_NAMES, values)
            }
        )
    if not candidates:
        raise ValueError("objective weight grid is empty")
    return tuple(candidates)


def weighted_score(
    metrics: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """Apply the four normalized benefit terms used by offline calibration."""

    return (
        float(weights["clean_accuracy_weight"])
        * float(metrics["clean_accuracy"])
        + float(weights["robust_accuracy_weight"])
        * float(metrics["robust_accuracy"])
        + float(weights["attack_success_weight"])
        * (1.0 - float(metrics["attack_success_rate"]))
        + float(weights["honest_weight_loss_weight"])
        * (1.0 - float(metrics["honest_weight_loss"]))
    )


def mean_metric(rows: Iterable[Mapping[str, float]], name: str) -> float:
    values = [float(row[name]) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _stable_ratio(value: float) -> float:
    return float(round(value, 12))


__all__ = [
    "CalibrationHardConstraints",
    "DEFAULT_MAX_NONFINITE_UPDATES",
    "DEFAULT_MIN_ROUND_COMPLETION_RATE",
    "DEFAULT_OBJECTIVE_WEIGHT_FLOOR",
    "DEFAULT_OBJECTIVE_WEIGHT_STEP",
    "OBJECTIVE_WEIGHT_NAMES",
    "RatioSchedule",
    "build_ratio_schedule",
    "mean_metric",
    "objective_weight_grid",
    "weighted_score",
]
