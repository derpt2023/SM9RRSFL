"""Read-only, integrity-gated reporting for the fixed C7 replication panel."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


PROFILE = {
    "id": "C7", "optimizer": "adam", "lr": 0.001,
    "boost": 20.0, "stealth_steps": 10, "scope": "weights",
}
METHODS = ("fedavg", "sm9rrs", "vert")
WINDOW = tuple(range(25, 31))
METRICS = (
    "accuracy", "training_target_success", "training_target_confidence",
    "legacy_asr", "legacy_confidence", "source_transfer_asr",
    "source_transfer_confidence", "honest_weight_loss", "malicious_weight_mass",
)
COUNTS = {"training_target_count": 1, "legacy_target_count": 1,
          "source_transfer_count": 750}
TARGET_FIELDS = {
    "training_target_indices": "training_target_count",
    "legacy_validation_indices": "legacy_target_count",
    "transfer_validation_indices": "source_transfer_count",
}
TASKS = {
    "clean_sm9rrs_s301": ("sm9rrs", 0.0, 301),
    "clean_vert_s301": ("vert", 0.0, 301),
    **{f"C7_{method}_dir{ratio}_s401": (method, ratio / 100, 401)
       for ratio in (60, 80) for method in METHODS},
}
REFERENCES = {
    "clean_dirichlet": ("fedavg", 0.0, 301),
    **{f"C7_{method}_dir{ratio}": (method, ratio / 100, 301)
       for ratio in (60, 80) for method in METHODS},
}


def _load(path):
    def reject(value):
        raise ValueError(f"non-finite JSON constant {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _number(value, field, *, lo=0.0, hi=None, integer=False, tolerance=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    if not math.isfinite(value) or value < lo - tolerance:
        raise ValueError(f"{field} is non-finite or below its range")
    if hi is not None and value > hi + tolerance:
        raise ValueError(f"{field} exceeds its range")
    if integer and int(value) != value:
        raise ValueError(f"{field} must be an integer")
    return value


def _finite_tree(value, prefix):
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{prefix}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{prefix} contains a non-finite value")


def _check_identity(identity, spec):
    method, ratio, seed = spec
    if identity.get("protocol") != "cifar_attack_stability_task_v1":
        raise ValueError("unexpected worker protocol")
    fingerprint = identity.get("fingerprint")
    body = {key: value for key, value in identity.items() if key != "fingerprint"}
    if not fingerprint or fingerprint != _digest(body):
        raise ValueError("task identity fingerprint mismatch")
    config = identity["config"]
    fixed = {
        "method": method, "malicious_ratio": ratio, "seed": seed,
        "partition": "dirichlet", "dirichlet_alpha": .5,
        "num_clients": 100, "rounds": 30, "local_epochs": 1, "batch_size": 50,
        "lr": .01, "lr_decay": .99, "detector_window": 20,
        "attack": "alternating_minimization", "attack_start_round": 25,
        "attack_source_label": 5, "attack_target_label": 7, "attack_target_count": 1,
        "attack_epochs": 1, "attack_distance_weight": .0001,
        "eval_interval": 1, "early_stop": False, "compute_backend": "torch",
    }
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ValueError(f"task config {key} differs from fixed protocol")
    if identity.get("profile") != (PROFILE if ratio else None):
        raise ValueError("task attack profile differs from fixed C7/clean protocol")
    data = identity["data_config"]
    fixed_data = {
        "dataset": "cifar10", "seed": 42, "split_seed": 20261013,
        "train_samples": 50000, "test_samples": 10000, "validation_fraction": .15,
    }
    for key, expected in fixed_data.items():
        if data.get(key) != expected:
            raise ValueError(f"task data {key} differs from fixed protocol")
    return fingerprint


def _check_targets(targets):
    if targets.get("official_test_used_for_selection") is not False:
        raise ValueError("target source is not explicitly training-derived")
    if targets.get("training_target_is_optimizer_input") is not True:
        raise ValueError("training target exposure missing")
    if targets.get("transfer_target_is_optimizer_input") is not False:
        raise ValueError("transfer targets must not enter the optimizer")
    for field, count in COUNTS.items():
        if _number(targets.get(field), f"target manifest {field}", integer=True) != count:
            raise ValueError(f"target manifest {field} differs from {count}")
    for field, count_field in TARGET_FIELDS.items():
        indices = targets.get(field)
        if not isinstance(indices, list) or len(indices) != COUNTS[count_field]:
            raise ValueError(f"target manifest {field} has the wrong denominator")
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices):
            raise ValueError(f"target manifest {field} has invalid indices")
        if len(set(indices)) != len(indices):
            raise ValueError(f"target manifest {field} duplicates indices")
    if not set(targets["legacy_validation_indices"]).issubset(
            targets["transfer_validation_indices"]):
        raise ValueError("legacy target is not among source-class validation targets")


def _check_rows(rows, spec, completed_round):
    method, ratio, _seed = spec
    if not isinstance(rows, list) or [row.get("round") for row in rows] != list(
            range(completed_round + 1)):
        raise ValueError("metrics do not contain the exact completed-round prefix")
    cumulative = 0
    previous_fp = 0
    honest_count = 100 - round(100 * ratio)
    for row in rows:
        rid = _number(row["round"], "round", hi=30, integer=True)
        if row.get("method") != method or row.get("malicious_ratio") != ratio:
            raise ValueError(f"round {rid} method/ratio differs from task")
        if row.get("attack_active") is not (ratio > 0 and rid >= 25):
            raise ValueError(f"round {rid} attack timing differs from protocol")
        for metric in METRICS:
            _number(row.get(metric), f"round {rid} {metric}", hi=1.0,
                    tolerance=1e-9 if metric == "malicious_weight_mass" else 0.0)
        for field, expected in COUNTS.items():
            if _number(row.get(field), f"round {rid} {field}", integer=True) != expected:
                raise ValueError(f"round {rid} {field} differs from {expected}")
        for metric, denominator in (
                ("training_target_success", 1), ("legacy_asr", 1),
                ("source_transfer_asr", 750)):
            count = row[metric] * denominator
            if abs(count - round(count)) > 1e-7:
                raise ValueError(f"round {rid} {metric} contradicts its denominator")
        nf = _number(row.get("nonfinite_updates"), f"round {rid} nonfinite_updates",
                     integer=True)
        cumulative += nf
        observed = _number(row.get("cumulative_nonfinite_updates"),
                           f"round {rid} cumulative_nonfinite_updates", integer=True)
        if observed != cumulative:
            raise ValueError(f"round {rid} nonfinite counters disagree")
        fp = _number(row.get("false_positive_revocations"),
                     f"round {rid} false_positive_revocations",
                     hi=honest_count, integer=True)
        if fp < previous_fp:
            raise ValueError("cumulative false revocations decreased")
        previous_fp = fp


def _mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def _summarize(directory, task_id, spec):
    """Incomplete or malformed evidence never receives complete-run metrics."""
    directory = Path(directory) / "tasks" / task_id
    report = {
        "task_id": task_id, "method": spec[0], "malicious_ratio": spec[1],
        "seed": spec[2], "status": "not_started", "valid_complete": False,
        "numerically_finite": None, "integrity_errors": [],
    }
    private = {}
    if not (directory / "status.json").exists():
        return report, private
    try:
        status = _load(directory / "status.json")
        report["status"] = status["status"]
        for field in ("error", "error_type"):
            if field in status:
                report[field] = status[field]
        if (directory / "first_failure.json").exists():
            report["first_failure"] = _load(directory / "first_failure.json")
        report["completed_round"] = status["completed_round"]
        report["elapsed_seconds"] = _number(status["elapsed_seconds"], "elapsed_seconds")
        report["timing_has_unclean_resume"] = bool(status.get("timing_has_unclean_resume"))
        identity = _load(directory / "task.json")
        fingerprint = _check_identity(identity, spec)
        if status.get("task_id") != task_id or status.get("fingerprint") != fingerprint:
            raise ValueError("status identity differs from task")
        if status.get("requested_rounds") != 30:
            raise ValueError("status requested rounds differs from protocol")
        completed = _number(status["completed_round"], "completed_round",
                            lo=-1, hi=30, integer=True)
        private["identity"] = identity
        if completed < 0:
            if status["status"] == "complete":
                raise ValueError("complete status has no completed rounds")
            return report, private
        metrics = _load(directory / "metrics.json")
        if metrics.get("fingerprint") != fingerprint:
            raise ValueError("metric identity differs from task")
        if metrics.get("official_test_used_for_selection") is not False:
            raise ValueError("metric source is not explicitly training-derived")
        targets = _load(directory / "targets.json")
        _check_targets(targets)
        rows = metrics["rounds"]
        _finite_tree(status, "status")
        _finite_tree(metrics, "metrics")
        _check_rows(rows, spec, completed)
        report["observed_metric_points"] = len(rows)
        report["observed_nonfinite_updates"] = rows[-1]["cumulative_nonfinite_updates"]
        private.update(rows=rows, targets=targets)
        if status["status"] != "complete":
            return report, private
        if completed != 30 or len(rows) != 31:
            raise ValueError("complete status requires rounds 0 through 30")
        if _number(status.get("nonfinite_updates"), "status nonfinite_updates",
                   integer=True) != rows[-1]["cumulative_nonfinite_updates"]:
            raise ValueError("complete status nonfinite counter disagrees")
        if status.get("final_metrics") != rows[-1]:
            raise ValueError("complete status final metrics disagree")
        window = [row for row in rows if row["round"] in WINDOW]
        report.update(
            valid_complete=True, numerically_finite=status["nonfinite_updates"] == 0,
            nonfinite_updates=status["nonfinite_updates"],
            final_false_positive_revocations=rows[-1]["false_positive_revocations"],
            false_revocation_rate=rows[-1]["false_positive_revocations"] /
            (100 - round(100 * spec[1])),
            attack_window_rounds=list(WINDOW), metric_denominators=dict(COUNTS),
        )
        for key in METRICS:
            report[f"final_{key}"] = rows[-1][key]
            report[f"attack_window_mean_{key}"] = _mean(window, key)
        for key in ("honest_weight_loss", "malicious_weight_mass"):
            report[f"attack_window_max_{key}"] = max(row[key] for row in window)
        report["target_indices"] = {
            field: targets[field] for field in ("training_target_indices",
                                               "legacy_validation_indices")}
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        report["integrity_errors"].append(str(error))
        report["valid_complete"] = False
        report["numerically_finite"] = None
    return report, private


def _same_conditions(left, right, *, allowed_config_changes):
    left_config = dict(left["identity"]["config"])
    right_config = dict(right["identity"]["config"])
    for key in allowed_config_changes:
        left_config.pop(key, None)
        right_config.pop(key, None)
    if left_config != right_config:
        return "scientific task configuration differs beyond declared comparison"
    if left["identity"]["data_config"] != right["identity"]["data_config"]:
        return "dataset configuration differs"
    return None


def _clean_pair(clean, attacked, clean_private, attacked_private):
    value = {"clean_task_id": clean["task_id"], "attack_task_id": attacked["task_id"],
             "method": attacked["method"], "malicious_ratio": attacked["malicious_ratio"],
             "available": False}
    if not clean["valid_complete"] or not attacked["valid_complete"]:
        value["reason"] = "complete valid clean and attacked tasks are required"
        return value
    error = _same_conditions(clean_private, attacked_private,
                             allowed_config_changes=("malicious_ratio",))
    if error or clean_private["targets"] != attacked_private["targets"]:
        value["reason"] = error or "paired target identities differ"
        return value
    clean_rows = clean_private["rows"][25:31]
    attack_rows = attacked_private["rows"][25:31]
    value.update(
        available=True,
        final_accuracy_loss=clean["final_accuracy"] - attacked["final_accuracy"],
        attack_window_accuracy_loss=clean["attack_window_mean_accuracy"] -
        attacked["attack_window_mean_accuracy"],
        clean_final_false_positive_revocations=clean["final_false_positive_revocations"],
        clean_false_revocation_rate=clean["false_revocation_rate"],
        clean_window_target_hit_rate=clean["attack_window_mean_training_target_success"],
        attacked_window_target_hit_rate=attacked["attack_window_mean_training_target_success"],
        target_confidence_gain=attacked["attack_window_mean_training_target_confidence"] -
        clean["attack_window_mean_training_target_confidence"],
        target_flips_to_attack_label=sum(
            c["training_target_success"] == 0 and a["training_target_success"] == 1
            for c, a in zip(clean_rows, attack_rows)),
        target_flips_away_from_attack_label=sum(
            c["training_target_success"] == 1 and a["training_target_success"] == 0
            for c, a in zip(clean_rows, attack_rows)),
        paired_round_count=6,
        short_clean_numeric_and_revocation_gates_pass=(
            clean["numerically_finite"] and clean["false_revocation_rate"] <= .1),
    )
    return value


def _seed_pair(new, old, new_private, old_private):
    value = {"seed401_task_id": new["task_id"], "seed301_task_id": old["task_id"],
             "method": new["method"], "malicious_ratio": new["malicious_ratio"],
             "available": False}
    if not new["valid_complete"] or not old["valid_complete"]:
        value["reason"] = "complete valid tasks from both seeds are required"
        return value
    error = _same_conditions(new_private, old_private, allowed_config_changes=("seed",))
    if error or new_private["identity"]["profile"] != old_private["identity"]["profile"]:
        value["reason"] = error or "attack profiles differ"
        return value
    value.update(available=True, delta_direction="seed401 minus seed301",
                 both_numerically_finite=new["numerically_finite"] and old["numerically_finite"],
                 target_selection_changed={
                     field: new_private["targets"][field] != old_private["targets"][field]
                     for field in TARGET_FIELDS})
    for key in METRICS:
        for prefix in ("final_", "attack_window_mean_"):
            value[prefix + key + "_delta"] = new[prefix + key] - old[prefix + key]
    value["final_false_positive_revocations_delta"] = (
        new["final_false_positive_revocations"] - old["final_false_positive_revocations"])
    return value


def _rankings(tasks, references):
    output = []
    fields = ("final_accuracy", "attack_window_mean_training_target_success",
              "attack_window_mean_source_transfer_asr")
    for ratio in (60, 80):
        current = [tasks[f"C7_{method}_dir{ratio}_s401"] for method in METHODS]
        prior = [references[f"C7_{method}_dir{ratio}"] for method in METHODS]
        for field in fields:
            row = {"malicious_ratio": ratio / 100, "metric": field, "available": False}
            if all(task["valid_complete"] for task in current + prior):
                descending = field == "final_accuracy"
                orders = {}
                for seed, group in ((301, prior), (401, current)):
                    orders[str(seed)] = [
                        {"method": task["method"], "value": task[field]}
                        for task in sorted(group, key=lambda task: (
                            -task[field] if descending else task[field], task["method"]))]
                directions = []
                for i in range(len(METHODS)):
                    for j in range(i + 1, len(METHODS)):
                        a, b = METHODS[i], METHODS[j]
                        def sign(group):
                            by_method = {task["method"]: task for task in group}
                            difference = by_method[a][field] - by_method[b][field]
                            return 0 if abs(difference) <= 1e-12 else (1 if difference > 0 else -1)
                        first, second = sign(prior), sign(current)
                        directions.append({"method_a": a, "method_b": b,
                                           "seed301_sign": first, "seed401_sign": second,
                                           "same_direction": first == second})
                row.update(available=True, preferred_direction="higher" if descending else "lower",
                           rankings=orders, pairwise_directions=directions,
                           all_pairwise_directions_same=all(
                               item["same_direction"] for item in directions))
            output.append(row)
    return output


def build_report(output_dir, reference_dir):
    """Return a JSON-compatible report without writing or loading pickle files."""
    tasks, private, references, old_private = {}, {}, {}, {}
    for task_id, spec in TASKS.items():
        tasks[task_id], private[task_id] = _summarize(output_dir, task_id, spec)
    for task_id, spec in REFERENCES.items():
        references[task_id], old_private[task_id] = _summarize(reference_dir, task_id, spec)
    # Method comparisons must share the actual sampled images, not just counts.
    combined = [(task, private[task_id]) for task_id, task in tasks.items()]
    combined += [(task, old_private[task_id]) for task_id, task in references.items()]
    for seed in (301, 401):
        observed = [(task, details) for task, details in combined
                    if task["seed"] == seed and task["valid_complete"]]
        if observed and any(details["targets"] != observed[0][1]["targets"]
                            for _task, details in observed[1:]):
            for task, _details in observed:
                task["integrity_errors"].append("same-seed tasks use different target identities")
                task["valid_complete"] = False
                task["numerically_finite"] = None
    clean_pairs, seed_pairs = [], []
    for method in METHODS:
        clean_id = "clean_dirichlet" if method == "fedavg" else f"clean_{method}_s301"
        clean, details = ((references[clean_id], old_private[clean_id])
                          if method == "fedavg" else (tasks[clean_id], private[clean_id]))
        for ratio in (60, 80):
            old_id, new_id = f"C7_{method}_dir{ratio}", f"C7_{method}_dir{ratio}_s401"
            clean_pairs.append(_clean_pair(
                clean, references[old_id], details, old_private[old_id]))
            seed_pairs.append(_seed_pair(
                tasks[new_id], references[old_id], private[new_id], old_private[old_id]))
    valid = sum(task["valid_complete"] for task in tasks.values())
    return {
        "protocol": "cifar10_c7_replication_v1", "profile": dict(PROFILE),
        "planned_tasks": len(TASKS),
        "completed_tasks": sum(task["status"] == "complete" for task in tasks.values()),
        "valid_complete_tasks": valid, "all_complete": valid == len(TASKS),
        "reference_integrity_valid": all(task["valid_complete"] for task in references.values()),
        "all_comparisons_available": all(pair["available"] for pair in clean_pairs + seed_pairs),
        "all_numerically_finite": (valid == len(TASKS) and
                                   all(task["numerically_finite"] for task in tasks.values())),
        "task_runtime_seconds": sum(task.get("elapsed_seconds", 0.) for task in tasks.values()),
        "official_test_used_for_selection": False,
        "tasks": tasks, "reference_tasks": references,
        "seed301_clean_comparisons": clean_pairs, "seed_comparisons": seed_pairs,
        "rankings": _rankings(tasks, references),
        "seed401_matched_clean_available": False,
        "seed401_attack_effectiveness_qualification": "not_evaluated_without_matched_clean",
        "limitations": [
            "Only 30 rounds and six attack rounds (25 through 30); no 150-round qualification.",
            "Two seeds describe repeatability and do not establish statistical significance.",
            "Seed401 has no matched clean reference; confidence-gain attack qualification is not evaluated.",
            "The seed changes client partition, malicious identities and target selection; target indices are reported.",
            "Clean controls use seed301 and only establish the observed short-horizon clean behavior.",
            "Training-target success uses the image seen by the attacker; transfer ASR uses 750 independent validation images.",
            "Source-code and dataset-content lineage checks are performed by the execution controller.",
        ],
    }
