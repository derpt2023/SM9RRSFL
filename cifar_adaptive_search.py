"""Versioned, replayable product-KDE TPE for the independent CIFAR study.

This is a tree-structured Parzen-estimator search, not a Gaussian process and
not an exhaustive grid. Each conditional method has its own sampler bound to
one public configuration. After one declared default and random startup,
completed losses split observations into good/bad densities; proposals maximize
log l(x) - log g(x). Continuous coordinates use bounded Gaussian KDEs plus a
uniform prior; finite choices use smoothed categorical frequencies.

The runner owns persistence, scheduling, health decisions, exact rational
performance gates and the wall-clock budget. No old experiment is imported.
Algorithm reference: https://proceedings.neurips.cc/paper_files/paper/2011/file/
86e8f7ab32cfd12577bc2619bc635690-Paper.pdf (Bergstra et al., 2011).
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any

import numpy as np

from sm9rrsfl.ours_policy import OursParameters


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "conditional-product-kde-tpe-v1"
METHODS = ("sm9rrs", "vert", "alignins", "krum", "ding13", "fedavg")
FIXED_METHODS = ("krum", "ding13", "fedavg")


def _real(lo, hi, *, log=False):
    return {"kind": "real", "low": float(lo), "high": float(hi), "log": log}


def _choice(*values):
    return {"kind": "choice", "values": list(values)}


SPACES = {
    "public": {
        "detector_window": _choice(*range(6, 21)),
        "lr": _real(.01, .1, log=True),
        "batch_size": _choice(32, 50, 64),
        "local_epochs": _choice(1, 2, 3),
        "attack_boost": _real(2, 15, log=True),
        "attack_epochs": _choice(1, 2, 3),
    },
    "sm9rrs": {
        "detector_subspace_dim": _choice(1, 2, 3, 4),
        "detector_normal_clusters": _choice(1, 2, 3),
        "detector_distance_threshold": _real(1.05, 4.5),
        "reject_margin": _real(.5, 5.5),
        "detector_drift_memory": _real(.7, .95),
        "allowance_fraction": _real(.3, 1.),
        "detector_drift_threshold": _real(.5, 9., log=True),
        "detector_history_confirm": _choice(2, 3, 4),
        "history_fraction": _real(.3, .9),
        "detector_recovery_confirm": _choice(2, 3, 4),
        "detector_reference_budget": _real(1.5, 4.5),
        "detector_clip_factor": _real(1., 3.5),
        "detector_weight_cap": _real(1., 3.),
        "suspicion_penalty_factor": _real(.25, .85),
        "suspicion_recovery_factor": _real(1.05, 1.5),
        "suspicion_remove_after": _choice(3, 4, 5, 6, 7, 8, 9),
    },
    "vert": {
        "vert_history_window": _choice(5, 7, 10, 15, 20),
        "vert_predict_epochs": _choice(5, 10, 20, 30),
        "vert_predict_lr": _real(.0001, .003, log=True),
    },
    "alignins": {
        "alignins_sparsity": _real(.1, .5),
        "alignins_tda_radius": _real(.5, 1.5),
        "alignins_mpsa_radius": _real(.5, 1.5),
    },
    **{method: {} for method in FIXED_METHODS},
}

# Only these first points are declared. Startup point two is seeded random,
# and is labelled as such. Historical validation scores never enter the fit.
DEFAULT_COORDINATES = {
    "public": dict(detector_window=10, lr=.05, batch_size=50,
                   local_epochs=1, attack_boost=5., attack_epochs=1),
    "sm9rrs": dict(detector_subspace_dim=2, detector_normal_clusters=2,
                   detector_distance_threshold=1.25, reject_margin=4.75,
                   detector_drift_memory=.8, allowance_fraction=1.,
                   detector_drift_threshold=6., detector_history_confirm=2,
                   history_fraction=.8, detector_recovery_confirm=2,
                   detector_reference_budget=3.5, detector_clip_factor=2.,
                   detector_weight_cap=2., suspicion_penalty_factor=.5,
                   suspicion_recovery_factor=1.25, suspicion_remove_after=5),
    "vert": dict(vert_history_window=10, vert_predict_epochs=20, vert_predict_lr=.0005),
    "alignins": dict(alignins_sparsity=.3, alignins_tda_radius=.5, alignins_mpsa_radius=1.),
    **{method: {} for method in FIXED_METHODS},
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _validate_coordinates(space, values):
    if not isinstance(values, dict) or set(values) != set(SPACES[space]):
        raise ValueError("coordinate fields do not match the declared search space")
    for key, dimension in SPACES[space].items():
        value = values[key]
        if not _finite_number(value):
            raise ValueError(f"{key} must be a finite number, not a boolean")
        if dimension["kind"] == "choice":
            if not any(type(value) is type(option) and value == option for option in dimension["values"]):
                raise ValueError(f"{key} is not a declared finite choice")
        elif not dimension["low"] <= value <= dimension["high"]:
            raise ValueError(f"{key} is outside its declared bounds")


def decode_parameters(space, coordinates):
    """Decode conditional Ours ratios without proposing invalid inequalities."""
    if space not in SPACES:
        raise ValueError("unknown sampler space")
    _validate_coordinates(space, coordinates)
    parameters = deepcopy(coordinates)
    if space == "sm9rrs":
        warning = parameters["detector_distance_threshold"]
        parameters["detector_reject_threshold"] = warning + parameters.pop("reject_margin")
        parameters["detector_drift_allowance"] = warning * parameters.pop("allowance_fraction")
        parameters["detector_history_threshold"] = warning * parameters.pop("history_fraction")
        OursParameters(**parameters).validate()
    elif space == "vert":
        parameters.update(vert_projection_dim=128, vert_top_k=0, vert_use_ratio_prior=False)
    return parameters


def space_contract(space):
    """JSON-safe schema, default, conditional transforms and fixed policy."""
    if space not in SPACES:
        raise ValueError("unknown sampler space")
    return {
        "space": space, "dimensions": deepcopy(SPACES[space]),
        "default_coordinates": deepcopy(DEFAULT_COORDINATES[space]),
        "default_parameters": decode_parameters(space, DEFAULT_COORDINATES[space]),
        "conditional_transforms": (
            {"detector_reject_threshold": "warning + reject_margin",
             "detector_drift_allowance": "warning * allowance_fraction",
             "detector_history_threshold": "warning * history_fraction"}
            if space == "sm9rrs" else {}),
        "public_fixed": ({"lr_decay": .99, "attack_stealth_steps": 1,
                          "attack_distance_weight": .0001,
                          "attack_start_round": "detector_window + 2"}
                         if space == "public" else {}),
    }


def _bounds(dimension):
    lo, hi = dimension["low"], dimension["high"]
    return (math.log(lo), math.log(hi)) if dimension["log"] else (lo, hi)


def _transform(dimension, value):
    return math.log(value) if dimension["log"] else float(value)


def _cdf(value):
    return .5 * (1. + math.erf(value / math.sqrt(2.)))


class _Density:
    def __init__(self, dimension, observed):
        self.dimension = dimension
        self.observed = observed
        if dimension["kind"] == "choice":
            choices = dimension["values"]
            # One total uniform prior count; every category keeps support.
            self.probabilities = np.asarray([
                sum(value == option for value in observed) + 1. / len(choices)
                for option in choices], dtype=np.float64)
            self.probabilities /= self.probabilities.sum()
        else:
            self.lo, self.hi = _bounds(dimension)
            self.centers = np.asarray([_transform(dimension, v) for v in observed])
            width = self.hi - self.lo
            # Broad kernels during small-sample search, narrowing as n grows.
            self.bandwidth = max(width * .03, width / max(4., math.sqrt(len(observed) + 1)))
            self.normalizers = np.asarray([
                _cdf((self.hi - center) / self.bandwidth) -
                _cdf((self.lo - center) / self.bandwidth) for center in self.centers])

    def draw(self, rng):
        dimension = self.dimension
        if dimension["kind"] == "choice":
            index = int(rng.choice(len(dimension["values"]), p=self.probabilities))
            return dimension["values"][index]
        component = int(rng.integers(len(self.centers) + 1))
        if component == len(self.centers):
            value = float(rng.uniform(self.lo, self.hi))
        else:
            # Rejection samples the bounded normal whose normalization is used
            # in log_density. Clipping would introduce unmodelled point masses.
            while True:
                value = float(rng.normal(self.centers[component], self.bandwidth))
                if self.lo <= value <= self.hi:
                    break
        return math.exp(value) if dimension["log"] else value

    def log_density(self, value):
        dimension = self.dimension
        if dimension["kind"] == "choice":
            return math.log(float(self.probabilities[dimension["values"].index(value)]))
        x = _transform(dimension, value)
        gaussian = np.exp(-.5 * ((x - self.centers) / self.bandwidth) ** 2)
        gaussian /= self.bandwidth * math.sqrt(2. * math.pi) * self.normalizers
        density = (1. / (self.hi - self.lo) + float(gaussian.sum())) / (len(self.centers) + 1)
        # Log-coordinate Jacobians cancel in l/g for each identical proposal.
        return math.log(density)


class AdaptiveTPESampler:
    """Ask/tell state machine. Save state after ask before launching work.

    complete requires a finite scalar loss (smaller is better). failed is an
    observed algorithm failure and can only contribute to the bad density.
    incomplete is retained but excluded from fitting; use it for infrastructure
    failures or tasks interrupted by the wall-clock budget. These statuses do
    not replace the runner's complete-matrix and health qualification checks.
    """
    def __init__(self, space, seed, *, context_id="", startup_trials=2,
                 gamma=.25, candidate_pool_size=96):
        if space not in SPACES:
            raise ValueError("unknown sampler space")
        if type(seed) is not int or not 0 <= seed < 2 ** 63:
            raise ValueError("sampler seed must be a nonnegative 63-bit integer")
        if not isinstance(context_id, str) or (space != "public" and not context_id):
            raise ValueError("defense samplers require a public-context identity")
        if type(startup_trials) is not int or startup_trials < 2:
            raise ValueError("startup_trials must be an integer >= 2")
        if not _finite_number(gamma) or not 0 < gamma < 1:
            raise ValueError("gamma must be strictly between zero and one")
        if type(candidate_pool_size) is not int or candidate_pool_size < 2:
            raise ValueError("candidate_pool_size must be at least two")
        self.space, self.seed, self.context_id = space, seed, context_id
        self.startup_trials, self.gamma = startup_trials, float(gamma)
        self.candidate_pool_size = candidate_pool_size
        self.trials: list[dict[str, Any]] = []

    def _random(self):
        seed_material = {"seed": self.seed, "space": self.space,
                         "context_id": self.context_id, "ask_index": len(self.trials)}
        return np.random.default_rng(int(_digest(seed_material)[:16], 16))

    def _groups(self):
        completed = sorted((t for t in self.trials if t["status"] == "complete"),
                           key=lambda t: (t["loss"], t["trial_id"]))
        failed = [t for t in self.trials if t["status"] == "failed"]
        if len(completed) < 2 or len(completed) + len(failed) < self.startup_trials:
            return [], []
        # Failure observations never enter the good set, even if all trials fail.
        good_count = min(len(completed) - 1, max(1, math.ceil(self.gamma * len(completed))))
        return completed[:good_count], completed[good_count:] + failed

    def ask(self):
        if self.space in FIXED_METHODS and self.trials:
            raise StopIteration("a fixed method has exactly one genuine configuration")
        index, rng = len(self.trials), self._random()
        good, bad = self._groups()
        audit = {"algorithm": ALGORITHM_VERSION,
                 "good_trial_ids": [t["trial_id"] for t in good],
                 "bad_trial_ids": [t["trial_id"] for t in bad],
                 "fitted_observations": len(good) + len(bad),
                 "log_density_ratio": None}
        if index == 0:
            coordinates = deepcopy(DEFAULT_COORDINATES[self.space])
            strategy = "fixed" if self.space in FIXED_METHODS else "declared_default"
        else:
            use_tpe = bool(good and bad and index >= self.startup_trials)
            strategy = "tpe" if use_tpe else "random_startup"
            good_density = {key: _Density(d, [t["coordinates"][key] for t in good] if use_tpe else [])
                            for key, d in SPACES[self.space].items()}
            bad_density = {key: _Density(d, [t["coordinates"][key] for t in bad])
                           for key, d in SPACES[self.space].items()} if use_tpe else {}
            seen = {_canonical(t["parameters"]) for t in self.trials}
            proposals = []
            for _ in range(self.candidate_pool_size if use_tpe else 1):
                for _retry in range(1024):
                    point = {key: density.draw(rng) for key, density in good_density.items()}
                    if _canonical(decode_parameters(self.space, point)) not in seen:
                        break
                else:
                    raise RuntimeError("could not propose an unseen configuration")
                ratio = sum(good_density[k].log_density(v) - bad_density[k].log_density(v)
                            for k, v in point.items()) if use_tpe else 0.
                proposals.append((ratio, point))
            ratio, coordinates = max(proposals, key=lambda item: item[0])
            if use_tpe:
                audit["log_density_ratio"] = float(ratio)
                audit["candidate_pool_size"] = len(proposals)
                audit["good_losses"] = [t["loss"] for t in good]
                audit["bad_losses"] = [t["loss"] for t in bad]
        trial = {"trial_id": f"{self.space}-{index:04d}", "parameters": decode_parameters(self.space, coordinates),
                 "coordinates": coordinates, "strategy": strategy, "fit_audit": audit,
                 "status": "pending", "loss": None, "metrics": {}}
        self.trials.append(trial)
        return deepcopy(trial)

    def tell(self, trial_id, loss=None, *, status="complete", metrics=None):
        if status not in ("complete", "failed", "incomplete"):
            raise ValueError("observation status must be complete, failed or incomplete")
        if status == "complete" and not _finite_number(loss):
            raise ValueError("a complete observation requires a finite loss")
        if status != "complete" and loss is not None:
            raise ValueError("failed/incomplete observations must not invent a scalar loss")
        if metrics is not None and not isinstance(metrics, dict):
            raise ValueError("metrics must be a JSON object")
        metrics = deepcopy(metrics or {})
        _canonical(metrics)
        matches = [t for t in self.trials if t["trial_id"] == trial_id]
        if len(matches) != 1:
            raise ValueError("unknown trial_id")
        trial = matches[0]
        update = {"status": status, "loss": float(loss) if loss is not None else None, "metrics": metrics}
        if trial["status"] != "pending":
            if all(trial[k] == v for k, v in update.items()):
                return  # Safe replay after the result was saved before runner state.
            raise ValueError("a recorded observation is immutable")
        trial.update(update)

    @property
    def pending_trials(self):
        return deepcopy([t for t in self.trials if t["status"] == "pending"])

    def state_dict(self):
        return deepcopy({"schema_version": SCHEMA_VERSION, "algorithm_version": ALGORITHM_VERSION,
                         "space": self.space, "space_fingerprint": _digest(space_contract(self.space)),
                         "seed": self.seed, "context_id": self.context_id,
                         "startup_trials": self.startup_trials, "gamma": self.gamma,
                         "candidate_pool_size": self.candidate_pool_size, "trials": self.trials})

    @classmethod
    def from_state(cls, state):
        if not isinstance(state, dict):
            raise ValueError("sampler state must be a JSON object")
        expected = {"schema_version", "algorithm_version", "space", "space_fingerprint", "seed",
                    "context_id", "startup_trials", "gamma", "candidate_pool_size", "trials"}
        if set(state) != expected or state["schema_version"] != SCHEMA_VERSION or state["algorithm_version"] != ALGORITHM_VERSION:
            raise ValueError("unsupported sampler state schema/algorithm")
        _canonical(state)
        sampler = cls(state["space"], state["seed"], context_id=state["context_id"],
                      startup_trials=state["startup_trials"], gamma=state["gamma"],
                      candidate_pool_size=state["candidate_pool_size"])
        if state["space_fingerprint"] != _digest(space_contract(sampler.space)):
            raise ValueError("sampler search-space identity changed")
        if not isinstance(state["trials"], list):
            raise ValueError("trials must be an array")
        seen = set()
        for index, trial in enumerate(state["trials"]):
            fields = {"trial_id", "parameters", "coordinates", "strategy", "fit_audit", "status", "loss", "metrics"}
            if not isinstance(trial, dict) or set(trial) != fields or trial["trial_id"] != f"{sampler.space}-{index:04d}":
                raise ValueError("invalid trial identity/schema")
            if trial["parameters"] != decode_parameters(sampler.space, trial["coordinates"]):
                raise ValueError("saved parameters do not match conditional coordinates")
            signature = _canonical(trial["parameters"])
            if signature in seen:
                raise ValueError("duplicate saved configuration")
            seen.add(signature)
            if trial["status"] not in ("pending", "complete", "failed", "incomplete"):
                raise ValueError("invalid saved observation status")
            if trial["status"] == "complete":
                if not _finite_number(trial["loss"]):
                    raise ValueError("saved complete loss must be finite")
            elif trial["loss"] is not None:
                raise ValueError("non-complete observation must have null loss")
            if not isinstance(trial["metrics"], dict) or not isinstance(trial["fit_audit"], dict):
                raise ValueError("saved audit/metrics must be objects")
            if trial["strategy"] not in ("fixed", "declared_default", "random_startup", "tpe"):
                raise ValueError("invalid saved strategy")
            sampler.trials.append(deepcopy(trial))
        if sampler.space in FIXED_METHODS and len(sampler.trials) > 1:
            raise ValueError("fixed method has more than one trial")
        return sampler


def compute_relative_objective(ours_rows, baseline_rows, *, target=.02, original_score=None):
    """Return a transparent search loss, never a promotion or final-test gate.

    Rows contain scenario (JSON identity including seed), accuracy, asr,
    attacked, complete and healthy. All Ours rows must be complete and finite;
    invalid Ours returns loss=None for runner failure handling. Complete,
    finite baseline rows participate even when healthy=False, as in v7.
    Missing baseline rows are disclosed and penalized, not imputed to zero.
    The separate exact-rational gate decides whether a two-point boundary
    passes; floating-point comparisons here only guide the next proposal.
    """
    if not _finite_number(target) or target <= 0:
        raise ValueError("target must be positive and finite")
    if not isinstance(baseline_rows, dict) or set(baseline_rows) != set(METHODS) - {"sm9rrs"}:
        raise ValueError("baseline_rows must name all five fixed-selected methods")
    if not isinstance(ours_rows, list) or not ours_rows:
        raise ValueError("ours_rows must be a nonempty list")

    def index_rows(rows):
        indexed = {}
        for row in rows:
            if not isinstance(row, dict) or "scenario" not in row:
                raise ValueError("every row needs a scenario identity")
            key = _canonical(row["scenario"])
            if key in indexed:
                raise ValueError("duplicate scenario row")
            if type(row.get("attacked")) is not bool or type(row.get("complete")) is not bool or type(row.get("healthy")) is not bool:
                raise ValueError("attacked, complete and healthy must be explicit booleans")
            indexed[key] = row
        return indexed

    ours = index_rows(ours_rows)
    peers = {method: index_rows(rows) for method, rows in baseline_rows.items()}

    def valid_rate(value):
        return _finite_number(value) and 0 <= value <= 1

    invalid = [row["scenario"] for row in ours.values()
               if not row["complete"] or not valid_rate(row.get("accuracy")) or
               (row["attacked"] and not valid_rate(row.get("asr")))]
    if invalid:
        return {"loss": None, "status": "invalid_ours_evidence", "invalid_scenarios": invalid,
                "comparison_complete": False, "promotion_assessed": False}
    diagnostics, missing, gaps = [], [], []
    for key, row in ours.items():
        available = {"sm9rrs": row}
        for method, rows in peers.items():
            peer = rows.get(key)
            if peer is not None and peer["attacked"] != row["attacked"]:
                raise ValueError("paired scenarios disagree on attack status")
            if peer is None or not peer["complete"] or not valid_rate(peer.get("accuracy")) or (row["attacked"] and not valid_rate(peer.get("asr"))):
                missing.append({"method": method, "scenario": row["scenario"]})
            else:
                available[method] = peer
        best_acc = max(peer["accuracy"] for peer in available.values())
        accuracy_gap = best_acc - row["accuracy"]
        asr_gap = row["asr"] - min(peer["asr"] for peer in available.values()) if row["attacked"] else None
        gaps.append(max(0., accuracy_gap - target) / target)
        if asr_gap is not None:
            gaps.append(max(0., asr_gap - target) / target)
        diagnostics.append({"scenario": row["scenario"], "accuracy_gap": accuracy_gap,
                            "asr_gap": asr_gap, "available_methods": sorted(available),
                            "scorable_unhealthy_baselines": sorted(method for method, peer in available.items()
                                                                   if method != "sm9rrs" and not peer["healthy"])})
    if original_score is not None and not valid_rate(original_score):
        raise ValueError("original_score must be a finite value in [0,1] or null")
    unhealthy = [row["scenario"] for row in ours.values() if not row["healthy"]]
    maximum, mean = max(gaps), float(np.mean(gaps))
    missing_fraction = len(missing) / (5 * len(ours))
    # Known gap bounds (rates in [0,1]) make health dominate performance; an
    # unhealthy run must never look better because all its updates disappeared.
    health_penalty = (3. / target) if unhealthy else 0.
    missing_penalty = missing_fraction
    score_penalty = .01 * (1. - original_score) if original_score is not None else 0.
    loss = health_penalty + maximum + mean + missing_penalty + score_penalty
    return {"loss": loss, "status": "observed", "promotion_assessed": False,
            "comparison_complete": not missing, "missing_baselines": missing,
            "unhealthy_ours_scenarios": unhealthy, "scenarios": diagnostics,
            "components": {"health_penalty": health_penalty,
                           "max_normalized_excess": maximum, "mean_normalized_excess": mean,
                           "missing_baseline_penalty": missing_penalty,
                           "original_score_penalty": score_penalty},
            "loss_formula": "health_penalty + max_excess/target + mean_excess/target + missing_fraction + .01*(1-Score)"}
