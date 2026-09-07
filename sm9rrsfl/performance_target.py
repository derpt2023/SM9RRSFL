"""Validation-only selection towards an explicitly declared VERT target.

This is a target-aware Ours selection stage, not a method-neutral Score and
not a claim of test performance. VERT's independently selected policy stays
fixed. An unmet target remains visible and never erases failed experiments.
"""

from dataclasses import asdict, dataclass
import math
from statistics import fmean


@dataclass(frozen=True)
class PerformanceTarget:
    accuracy_gap: float = 0.02
    asr_gap: float = 0.01
    max_asr: float = 0.05
    max_peak_asr: float = 0.20
    tail_rounds: int = 10

    @classmethod
    def parse(cls, payload):
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("performance_target must be a JSON object or null")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown performance_target field: {sorted(unknown)[0]}")
        target = cls(**payload)
        for name in ("accuracy_gap", "asr_gap", "max_asr", "max_peak_asr"):
            value = getattr(target, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"performance_target.{name} must be finite and in [0, 1]")
        if isinstance(target.tail_rounds, bool) or not isinstance(target.tail_rounds, int) or target.tail_rounds < 1:
            raise ValueError("performance_target.tail_rounds must be a positive integer")
        if target.max_peak_asr < target.max_asr:
            raise ValueError("max_peak_asr must be at least max_asr")
        return target


def scenario_key(result):
    c = result.config
    return (c.partition, c.dirichlet_alpha, c.num_clients, c.malicious_ratio, c.seed)


def evaluate_target(ours, vert, policy, *, expected_scenarios=None):
    """Check paired runs individually: neither seed means nor tails hide failure."""
    failures, rows, violations = [], [], []
    groups = {}
    for method, results in (("sm9rrs", ours), ("vert", vert)):
        groups[method] = {scenario_key(r): r for r in results}
        if len(groups[method]) != len(results):
            failures.append(f"{method}:duplicate_scenario")
    a, v = groups["sm9rrs"], groups["vert"]
    expected = set(expected_scenarios) if expected_scenarios is not None else set(a) | set(v)
    if not expected or not any(k[3] == 0 for k in expected) or not any(k[3] > 0 for k in expected):
        failures.append("missing_clean_or_attacked_scenarios")
    for method, group in groups.items():
        if set(group) != expected:
            failures.append(f"{method}:incomplete_or_unexpected_scenarios")

    def check(row, name, excess, scale):
        if excess > 1e-12:
            row["failures"].append(name)
            violations.append(excess / max(scale, .01))

    for key in sorted(set(a) & set(v)):
        ra, rv = a[key], v[key]
        row = {"scenario": list(key), "failures": [], "windows": {}}
        shared_fields = ("rounds", "local_epochs", "batch_size", "lr", "lr_decay",
                         "attack", "attack_boost", "attack_epochs", "attack_stealth_steps",
                         "attack_distance_weight", "attack_source_label", "attack_target_label",
                         "attack_target_count")
        if any(getattr(ra.config, name) != getattr(rv.config, name) for name in shared_fields):
            row["failures"].append("mismatched_training_or_attack_protocol")
        for method, run in (("sm9rrs", ra), ("vert", rv)):
            if run.stopped_round != run.config.rounds or not run.records or run.records[-1].round != run.config.rounds or run.nonfinite_updates:
                row["failures"].append(f"{method}:incomplete_or_nonfinite_run")
        if key[3] == 0:
            windows = {"clean_final": ([ra.records[-1]], [rv.records[-1]])} if ra.records and rv.records else {}
        else:
            attack = [[r for r in run.records if r.round >= (run.config.attack_start_round or run.config.detector_window + 2)] for run in (ra, rv)]
            if not all(attack) or [r.round for r in attack[0]] != [r.round for r in attack[1]]:
                row["failures"].append("missing_or_mismatched_attack_rounds")
                windows = {}
            else:
                windows = {"attack_mean": attack, "tail_mean": [r[-policy.tail_rounds:] for r in attack], "final": [r[-1:] for r in attack]}
                rates = [r.attack_target_success_rate for r in attack[0]]
                if all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in rates):
                    row["ours_peak_asr"] = max(rates)
                    check(row, "peak_asr", max(rates) - policy.max_peak_asr, policy.max_peak_asr)
        for name, (ar, vr) in windows.items():
            values = {}
            for method, records in (("ours", ar), ("vert", vr)):
                for metric, attr in (("accuracy", "accuracy"), ("asr", "attack_target_success_rate")):
                    if key[3] == 0 and metric == "asr":
                        continue
                    entries = [getattr(r, attr) for r in records]
                    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in entries):
                        row["failures"].append(f"{name}:{method}:{metric}_missing")
                    else:
                        values[method + "_" + metric] = fmean(entries)
            row["windows"][name] = values
            if all(k in values for k in ("ours_accuracy", "vert_accuracy")):
                check(row, name + ":accuracy_gap", values["vert_accuracy"] - values["ours_accuracy"] - policy.accuracy_gap, policy.accuracy_gap)
            if all(k in values for k in ("ours_asr", "vert_asr")):
                check(row, name + ":asr_gap", values["ours_asr"] - values["vert_asr"] - policy.asr_gap, policy.asr_gap)
                check(row, name + ":absolute_asr", values["ours_asr"] - policy.max_asr, policy.max_asr)
        rows.append(row)
    structural = bool(failures) or any(any("missing" in f or "incomplete" in f or "mismatched" in f
                                         for f in r["failures"]) for r in rows)
    return {
        "status": "passed" if not failures and not any(r["failures"] for r in rows) else "unmet",
        "policy": asdict(policy), "failures": failures, "scenarios": rows,
        "structurally_complete": not structural,
        "worst_normalized_excess": max(violations, default=0.),
        "mean_normalized_excess": fmean(violations) if violations else 0.,
    }


def select_towards_target(trials, selected, results_by_candidate, policy, *, expected_scenarios=None):
    """Freeze VERT, then choose a healthy Ours policy meeting/nearest the goal."""
    if policy is None:
        return selected, {"status": "disabled"}
    vert = selected["vert"]
    audits, choices = {}, []
    for trial in trials:
        if trial.method != "sm9rrs":
            continue
        audit = evaluate_target(results_by_candidate.get(trial.candidate_id, []),
                                results_by_candidate[vert.candidate_id], policy,
                                expected_scenarios=expected_scenarios)
        audit["health_valid"] = trial.valid
        audit["health_failures"] = list(trial.invalid_reasons)
        if not trial.valid:
            audit["status"] = "ineligible"
        audits[trial.candidate_id] = audit
        if trial.valid and audit["structurally_complete"]:
            # First meet the declared goal, then minimize ASR. If no candidate
            # meets it, keep the closest healthy result visible as an unmet
            # exploratory fallback; do not suppress formal comparison data.
            rank = (audit["status"] == "passed", -audit["worst_normalized_excess"],
                    -audit["mean_normalized_excess"], -trial.attack_success_rate,
                    trial.robust_accuracy, trial.score, trial.candidate_id)
            choices.append((rank, trial))
    winner = max(choices, key=lambda pair: pair[0])[1] if choices else selected["sm9rrs"]
    report = {"status": "passed" if audits[winner.candidate_id]["status"] == "passed" else "unmet",
              "selection_data": "training_holdout_only", "policy": asdict(policy),
              "reference_candidate": vert.candidate_id,
              "independent_ours_candidate": selected["sm9rrs"].candidate_id,
              "selected_ours_candidate": winner.candidate_id, "candidates": audits,
              "fallback": "nearest_healthy_candidate_without_claiming_target_met"}
    return {**selected, "sm9rrs": winner}, report
