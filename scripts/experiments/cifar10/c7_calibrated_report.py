"""Read-only evidence checks for clean calibration and the C7 60-round panel.

Completion, numerical stability and observed training health are separate
properties. Failed health never silently removes a completed baseline from a
comparison, and incomplete tasks never receive completed-run window metrics.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from .cifar_c7_replication_report import (
    COUNTS, METRICS, PROFILE, _check_targets, _digest, _finite_tree, _load, _number,
)

ROOT = Path(__file__).resolve().parents[3]
METHODS = ("fedavg", "sm9rrs", "vert", "alignins", "krum", "ding13")
RANK_FIELDS = {
    "final_accuracy": "higher",
    "last10_mean_accuracy": "higher",
    "attack_window_mean_training_target_success": "lower",
    "attack_window_mean_source_transfer_asr": "lower",
    "last10_mean_source_transfer_asr": "lower",
}
ARRAYS = ("x_train", "y_train", "x_test", "y_test", "x_attack", "y_attack")
ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError)


def _invalidate(task, message):
    task["integrity_errors"].append(str(message))
    task["complete"] = task["valid_complete"] = False
    task["numerically_finite"] = None
    if "health" in task:
        task["health"]["complete_observed_health_pass"] = False


def _configuration(spec):
    config = {k: v for k, v in spec["config"].items() if k != "device"}
    fixed = {
        "partition": "dirichlet", "dirichlet_alpha": .5, "num_clients": 100,
        "local_epochs": 1, "batch_size": 50, "lr": .01, "lr_decay": .99,
        "detector_window": 20, "attack": "alternating_minimization",
        "attack_start_round": 25, "attack_source_label": 5, "attack_target_label": 7,
        "attack_target_count": 1, "attack_epochs": 1,
        "attack_distance_weight": .0001, "eval_interval": 1,
        "early_stop": False, "compute_backend": "torch",
    }
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ValueError(f"declared config {key} differs from the C7 protocol")
    if config["rounds"] not in (30, 60) or config["method"] not in METHODS:
        raise ValueError("unsupported horizon or method")
    _number(config["seed"], "seed", integer=True)
    _number(config["malicious_ratio"], "malicious_ratio", hi=.8)
    if config["malicious_ratio"] not in (0., .6, .8):
        raise ValueError("unsupported malicious ratio")
    if spec.get("profile") != (PROFILE if config["malicious_ratio"] else None):
        raise ValueError("declared attack profile differs from C7/clean")
    return config


def _dataset(identity, dataset):
    config = identity["data_config"]
    expected = {
        "dataset": "cifar10", "seed": 42, "split_seed": 20261013,
        "train_samples": 50000, "test_samples": 10000, "validation_fraction": .15,
    }
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("dataset configuration differs from the frozen split")
    arrays = dataset["arrays"]
    if set(arrays) != set(ARRAYS) or any(
            not isinstance(arrays[k], str) or not re.fullmatch("[a-f0-9]{64}", arrays[k])
            for k in ARRAYS):
        raise ValueError("dataset array identity is incomplete")
    contract = dataset["contract"]
    for key, value in {
        "train_samples": 40000, "calibration_samples": 7500,
        "attack_auxiliary_samples": 2500, "split_seed": 20261013,
        "official_test_used_for_selection": False,
    }.items():
        if contract.get(key) != value:
            raise ValueError(f"dataset contract differs on {key}")


def _rows(rows, config, completed):
    if [r.get("round") for r in rows] != list(range(completed + 1)):
        raise ValueError("metrics lack the exact completed-round prefix")
    nf_total = fp_previous = tp_previous = blacklist_previous = 0
    malicious = round(config["num_clients"] * config["malicious_ratio"])
    honest = config["num_clients"] - malicious
    for row in rows:
        r = _number(row["round"], "round", hi=config["rounds"], integer=True)
        if (row.get("method") != config["method"]
                or row.get("malicious_ratio") != config["malicious_ratio"]
                or row.get("attack_active") is not (
                    config["malicious_ratio"] > 0 and r >= config["attack_start_round"])):
            raise ValueError(f"round {r} method, ratio or attack timing differs")
        for field in METRICS:
            _number(row[field], f"round {r} {field}", hi=1.,
                    tolerance=1e-9 if field == "malicious_weight_mass" else 0.)
        for field, denominator in COUNTS.items():
            if _number(row[field], field, integer=True) != denominator:
                raise ValueError(f"round {r} {field} denominator differs")
        for field, denominator in (
            ("training_target_success", 1), ("legacy_asr", 1),
            ("source_transfer_asr", 750), ("accuracy", 7500),
        ):
            count = row[field] * denominator
            if abs(count - round(count)) > 1e-7:
                raise ValueError(f"round {r} {field} contradicts its denominator")
        nf_total += _number(row["nonfinite_updates"], "nonfinite_updates", integer=True)
        if _number(row["cumulative_nonfinite_updates"], "cumulative_nonfinite_updates",
                   integer=True) != nf_total:
            raise ValueError("nonfinite counters disagree")
        fp = _number(row["false_positive_revocations"], "false_positive_revocations",
                     hi=honest, integer=True)
        tp = _number(row["true_positive_revocations"], "true_positive_revocations",
                     hi=malicious, integer=True)
        blacklist = _number(row["blacklisted_clients"], "blacklisted_clients",
                            hi=config["num_clients"], integer=True)
        if fp < fp_previous or tp < tp_previous or blacklist < blacklist_previous:
            raise ValueError("cumulative revocation counts decreased")
        if blacklist != fp + tp:
            raise ValueError("blacklist and revocation counts disagree")
        for field in ("accepted_updates", "rejected_updates"):
            _number(row[field], field, hi=config["num_clients"], integer=True)
        fp_previous, tp_previous, blacklist_previous = fp, tp, blacklist


def _trend(rows, start, end, field):
    selected = [r for r in rows if start <= r["round"] <= end]
    if [r["round"] for r in selected] != list(range(start, end + 1)):
        return {"available": False}
    xs, ys = [r["round"] for r in selected], [r[field] for r in selected]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    denominator = sum((x - xm) ** 2 for x in xs)
    return {
        "available": True, "rounds": xs, "values": ys,
        "end_minus_start": ys[-1] - ys[0], "range": max(ys) - min(ys),
        "ols_slope_per_round": (sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
                                / denominator if denominator else 0.),
        "statistical_significance": "not_estimated; dependent observations from one run",
    }


def _health(task, rows, config):
    actual = [r for r in rows if r["round"] > 0]
    honest = config["num_clients"] - round(config["num_clients"] * config["malicious_ratio"])
    all_clients = [r["round"] for r in actual if r["blacklisted_clients"] == 100]
    all_honest = [r["round"] for r in actual if r["false_positive_revocations"] == honest]
    empty = [r["round"] for r in actual if r["accepted_updates"] == 0]
    consecutive, starvation = 0, []
    for row in actual:
        consecutive = consecutive + 1 if row["honest_weight_loss"] >= .99 else 0
        if config["method"] == "sm9rrs" and consecutive >= 5:
            starvation.append(row["round"])
    false_rate = rows[-1]["false_positive_revocations"] / honest if rows else None
    reasons = []
    for reason, evidence in (
        ("all_clients_revoked", all_clients), ("all_honest_revoked", all_honest),
        ("empty_aggregation", empty), ("honest_weight_starvation", starvation),
        ("nonfinite_updates", task.get("observed_nonfinite_updates", 0)
         or task["status"] == "numerical_failure"),
        ("clean_false_revocation_rate", config["malicious_ratio"] == 0
         and false_rate is not None and false_rate > .1),
    ):
        if evidence:
            reasons.append(reason)
    return {
        "observed_failure_reasons": reasons,
        "all_clients_revoked_rounds": all_clients, "all_honest_revoked_rounds": all_honest,
        "empty_aggregation_rounds": empty, "ours_starvation_rounds": starvation,
        "false_revocation_rate": false_rate,
        "pre_attack_false_positive_revocations": max(
            (r["false_positive_revocations"] for r in rows if r["round"] < 25), default=0),
        "pre_attack_all_honest_revoked": any(r < 25 for r in all_honest),
        "complete_observed_health_pass": bool(
            task["complete"] and task["numerically_finite"] and not reasons),
        "formal_150_round_qualification": False,
    }


def _exposure(path, task, config, malicious_clients):
    fields = ("call_attempts", "completed_calls", "normal_steps",
              "target_steps", "target_skipped_zero_loss")
    output = {"available": False, "integrity_errors": [],
              "note": "Counts are observed attempts; resumed client-rounds may be duplicated."}
    if not config["malicious_ratio"]:
        return {**output, "available": True, "reason": "clean_task",
                "totals": dict.fromkeys(fields, 0), "by_round": {},
                "unique_completed_client_rounds": 0, "duplicate_client_round_pairs": []}
    if not path.exists():
        return {**output, "reason": "audit absent; zero actual calls are not inferred"}
    try:
        counts = {r: dict.fromkeys(fields, 0) for r in range(25, config["rounds"] + 1)}
        seen, completed, duplicates = set(), set(), set()
        uncommitted = failures = 0
        for line in path.read_text().splitlines():
            record = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(
                ValueError(f"non-finite audit value {x}")))
            if record.get("event") == "first_numerical_failure":
                failures += 1
                continue
            _finite_tree(record, "audit")
            rid = _number(record["round"], "audit round", lo=25,
                          hi=config["rounds"], integer=True)
            expected = {"task_id": task["task_id"], "profile_id": "C7",
                        "optimizer": "adam", "scope": "weights", "lr": .001,
                        "boost": 20., "stealth_steps": 10}
            if any(record.get(k) != v for k, v in expected.items()):
                raise ValueError("attack audit identity/profile differs from C7")
            client = record["client_id"]
            if malicious_clients is not None and client not in malicious_clients:
                raise ValueError("attack audit contains a nonmalicious client")
            pair = (rid, client)
            if pair in seen:
                duplicates.add(pair)
            seen.add(pair)
            if rid > task.get("completed_round", -1):
                uncommitted += 1
            current = counts[rid]
            current["call_attempts"] += 1
            current["completed_calls"] += record["status"] == "complete"
            if record["status"] == "complete":
                completed.add(pair)
            for field in fields[2:]:
                current[field] += _number(record[field], f"audit {field}", integer=True)
        output.update(
            available=True, by_round={str(k): v for k, v in counts.items()},
            totals={k: sum(r[k] for r in counts.values()) for k in fields},
            unique_client_rounds=len(seen), unique_completed_client_rounds=len(completed),
            duplicate_client_round_pairs=[{"round": r, "client_id": c}
                                          for r, c in sorted(duplicates)],
            attempts_after_last_completed_round=uncommitted,
            numerical_failure_events=failures,
            completed_run_counts_without_duplicate_attempts=bool(
                task["complete"] and not duplicates and not uncommitted
                and len(completed) == len(seen)),
        )
    except ERRORS as error:
        output["integrity_errors"].append(str(error))
    return output


def _task_details(directory, spec, root_fingerprint):
    task = {
        "task_id": spec["task_id"], "status": "not_started", "complete": False,
        "valid_complete": False, "numerically_finite": None, "integrity_errors": [],
        "method": spec["config"]["method"], "seed": spec["config"]["seed"],
        "malicious_ratio": spec["config"]["malicious_ratio"],
        "requested_rounds": spec["config"]["rounds"],
    }
    for field in ("stage", "candidate_id", "method_variant"):
        if field in spec:
            task[field] = spec[field]
    private, rows, config, rows_valid = {}, [], spec["config"], False
    directory = Path(directory) / "tasks" / spec["task_id"]
    try:
        config = _configuration(spec)
        if not (directory / "status.json").exists():
            task["health"] = _health(task, [], config)
            return task, private
        status = _load(directory / "status.json")
        task.update(status=status["status"], completed_round=status["completed_round"],
                    elapsed_seconds=_number(status["elapsed_seconds"], "elapsed_seconds"),
                    timing_has_unclean_resume=bool(status.get("timing_has_unclean_resume")))
        for field in ("error", "error_type"):
            if field in status:
                task[field] = status[field]
        if (directory / "first_failure.json").exists():
            task["first_failure"] = _load(directory / "first_failure.json")
        identity = _load(directory / "task.json")
        fingerprint = identity["fingerprint"]
        if fingerprint != _digest({k: v for k, v in identity.items() if k != "fingerprint"}):
            raise ValueError("task identity fingerprint mismatch")
        if (identity.get("protocol") != "cifar_attack_stability_task_v1"
                or identity.get("root_fingerprint") != root_fingerprint
                or identity.get("config") != config
                or identity.get("profile") != spec.get("profile")):
            raise ValueError("task differs from the declared manifest specification")
        if (status.get("task_id") != task["task_id"]
                or status.get("fingerprint") != fingerprint
                or status.get("requested_rounds") != config["rounds"]):
            raise ValueError("status identity/horizon differs from task")
        completed = _number(status["completed_round"], "completed_round",
                            lo=-1, hi=config["rounds"], integer=True)
        private["identity"] = identity
        if completed < 0:
            if status["status"] == "complete":
                raise ValueError("complete task has no completed rounds")
            task["health"] = _health(task, [], config)
            return task, private
        dataset = _load(directory / "dataset_identity.json")
        _dataset(identity, dataset)
        targets = _load(directory / "targets.json")
        _check_targets(targets)
        metrics = _load(directory / "metrics.json")
        if (metrics.get("fingerprint") != fingerprint
                or metrics.get("official_test_used_for_selection") is not False):
            raise ValueError("metric identity/source differs from task")
        rows = metrics["rounds"]
        if status["status"] != "complete":
            # Atomic metrics are published before the checkpoint/status.
            task["uncommitted_metric_rows"] = sum(r["round"] > completed for r in rows)
            rows = [r for r in rows if r["round"] <= completed]
        _finite_tree(status, "status")
        _finite_tree(rows, "metrics")
        _rows(rows, config, completed)
        rows_valid = True
        task.update(observed_metric_points=len(rows),
                    observed_nonfinite_updates=rows[-1]["cumulative_nonfinite_updates"])
        private.update(rows=rows, dataset=dataset, targets=targets)
        if status["status"] == "complete":
            if completed != config["rounds"]:
                raise ValueError("complete task lacks its requested horizon")
            if (status.get("final_metrics") != rows[-1]
                    or status.get("nonfinite_updates") != rows[-1]["cumulative_nonfinite_updates"]):
                raise ValueError("complete status and final metrics disagree")
            result = _load(directory / "result.json")
            _finite_tree(result, "result")
            for field in ("status", "task_id", "fingerprint", "completed_round",
                          "requested_rounds", "final_metrics", "nonfinite_updates"):
                if result.get(field) != status.get(field):
                    raise ValueError(f"result/status disagree on {field}")
            if result.get("profile") != spec.get("profile") or result.get("metric_scope") != targets:
                raise ValueError("result profile/metric scope differs")
            clients = result["summary"]["malicious_clients"]
            clients = clients.split(",") if clients else []
            if (len(clients) != round(100 * config["malicious_ratio"])
                    or len(set(clients)) != len(clients)
                    or any(not re.fullmatch(r"client-(?:[0-9]|[1-9][0-9])", c) for c in clients)):
                raise ValueError("malicious client list contradicts the configured ratio")
            private["malicious_clients"] = sorted(clients)
            task.update(complete=True, valid_complete=True,
                        numerically_finite=status["nonfinite_updates"] == 0,
                        nonfinite_updates=status["nonfinite_updates"],
                        metric_denominators=dict(COUNTS),
                        final_false_positive_revocations=rows[-1]["false_positive_revocations"],
                        false_revocation_rate=rows[-1]["false_positive_revocations"] /
                        (100 - round(100 * config["malicious_ratio"])))
            windows = {
                "attack_window": [r for r in rows if 25 <= r["round"] <= completed],
                "last10": [r for r in rows if completed - 9 <= r["round"] <= completed],
            }
            for name, selected in windows.items():
                task[f"{name}_rounds"] = [r["round"] for r in selected]
                for metric in METRICS:
                    task[f"{name}_mean_{metric}"] = math.fsum(r[metric] for r in selected) / len(selected)
            for metric in METRICS:
                task[f"final_{metric}"] = rows[-1][metric]
            task["late_round_trends"] = {
                f"{start}_{completed}": {m: _trend(rows, start, completed, m) for m in METRICS}
                for start in (25, completed - 9)
            }
            task["trajectory"] = [
                {k: r[k] for k in ("round", *METRICS, "accepted_updates", "rejected_updates",
                                   "blacklisted_clients", "true_positive_revocations",
                                   "false_positive_revocations", "nonfinite_updates")}
                for r in rows
            ]
        task["actual_attack_exposure"] = _exposure(
            directory / "audit.jsonl", task, config, private.get("malicious_clients"))
        for error in task["actual_attack_exposure"]["integrity_errors"]:
            _invalidate(task, error)
    except ERRORS as error:
        _invalidate(task, error)
    task["health"] = _health(task, rows, config) if rows_valid or not task["integrity_errors"] else {
        "observed_failure_reasons": [], "complete_observed_health_pass": False,
        "formal_150_round_qualification": False,
    }
    return task, private


def task_summary(directory, spec, root_fingerprint):
    """Read ``directory/tasks/<task_id>`` using the exact declared task spec."""
    return _task_details(directory, spec, root_fingerprint)[0]


def _clean_pair(clean, attacked, left, right):
    pair = {"clean_task_id": clean["task_id"], "attack_task_id": attacked["task_id"],
            "method": attacked["method"], "seed": attacked["seed"],
            "malicious_ratio": attacked["malicious_ratio"], "available": False}
    if not clean["complete"] or not attacked["complete"]:
        pair["reason"] = "both exact completed tasks are required"
        return pair
    a, b = dict(left["identity"]["config"]), dict(right["identity"]["config"])
    a.pop("malicious_ratio")
    b.pop("malicious_ratio")
    if (a != b or left["identity"]["data_config"] != right["identity"]["data_config"]
            or left["dataset"] != right["dataset"] or left["targets"] != right["targets"]):
        pair["reason"] = "scientific configuration/data/targets differ beyond malicious ratio"
        return pair
    pair.update(available=True, final_accuracy_loss=clean["final_accuracy"] - attacked["final_accuracy"],
                clean_observed_healthy=clean["health"]["complete_observed_health_pass"],
                attacked_observed_healthy=attacked["health"]["complete_observed_health_pass"])
    for window in ("attack_window", "last10"):
        prefix = window + "_mean_"
        pair[window + "_accuracy_loss"] = clean[prefix + "accuracy"] - attacked[prefix + "accuracy"]
        gain = attacked[prefix + "training_target_confidence"] - clean[prefix + "training_target_confidence"]
        hit = attacked[prefix + "training_target_success"]
        high_clean = (clean[prefix + "training_target_success"] >= .8
                      and clean[prefix + "training_target_confidence"] >= .8)
        pair[window + "_target_confidence_gain"] = gain
        pair[window + "_attack_effectiveness"] = {
            "effective": bool(clean["numerically_finite"] and attacked["numerically_finite"]
                              and hit >= .8 - 1e-12 and gain >= .2 - 1e-12 and not high_clean),
            "target_hit_rate": hit, "confidence_gain": gain,
            "clean_already_high_confidence_target_error": high_clean,
            "interpretation": "Paired local evidence; effectiveness is qualified using FedAvg only.",
        }
    return pair


def build_report(directory):
    """Validate a frozen panel and report all observed methods, including failures."""
    directory = Path(directory)
    errors, tasks, private, specs, manifest = [], {}, {}, [], {}
    try:
        manifest = _load(directory / "manifest.json")
        if manifest["fingerprint"] != _digest({k: v for k, v in manifest.items() if k != "fingerprint"}):
            raise ValueError("manifest fingerprint mismatch")
        specs = manifest["task_specs"]
        if not specs or len({s["task_id"] for s in specs}) != len(specs):
            raise ValueError("manifest has missing or duplicate task IDs")
        for field in ("source_files", "reference_files"):
            for path, expected in manifest[field].items():
                if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                    raise ValueError(f"immutable {field} changed: {path}")
    except ERRORS as error:
        errors.append(str(error))
    for spec in specs:
        task, detail = _task_details(directory, spec, manifest.get("fingerprint"))
        tasks[spec["task_id"]], private[spec["task_id"]] = task, detail
        if (detail.get("identity")
                and detail["identity"]["data_config"] != manifest.get("data_config")):
            _invalidate(task, "task dataset configuration differs from manifest")
        if errors:
            _invalidate(task, "manifest or frozen reference integrity failed")
    valid = [t for t in tasks.values() if t["complete"]]
    # The seed controls every single-image selector, while the training split is fixed.
    for seed in {t["seed"] for t in valid}:
        group = [t for t in valid if t["seed"] == seed]
        first = private[group[0]["task_id"]]
        if any(private[t["task_id"]][field] != first[field]
               for t in group for field in ("dataset", "targets")):
            for task in group:
                _invalidate(task, "same-seed dataset/target identities differ")
        for ratio in {t["malicious_ratio"] for t in group if t["malicious_ratio"]}:
            subset = [t for t in group if t["malicious_ratio"] == ratio]
            client_list = private[subset[0]["task_id"]]["malicious_clients"]
            if any(private[t["task_id"]]["malicious_clients"] != client_list for t in subset):
                for task in subset:
                    _invalidate(task, "same-seed/ratio malicious client lists differ")
    comparisons, rankings, utility = [], [], []
    for task in tasks.values():
        if task["malicious_ratio"]:
            clean = [c for c in tasks.values()
                     if (c["method"], c["seed"], c["requested_rounds"], c["malicious_ratio"])
                     == (task["method"], task["seed"], task["requested_rounds"], 0)]
            if len(clean) == 1:
                comparisons.append(_clean_pair(clean[0], task, private[clean[0]["task_id"]],
                                               private[task["task_id"]]))
            else:
                comparisons.append({"attack_task_id": task["task_id"], "available": False,
                                    "reason": "exactly one matched clean task is required"})
        elif task["complete"]:
            controls = [c for c in tasks.values() if c["complete"] and c["method"] == "fedavg"
                        and c["malicious_ratio"] == 0 and c["seed"] == task["seed"]
                        and c["requested_rounds"] == task["requested_rounds"]]
            if len(controls) == 1:
                baseline = controls[0]
                loss = baseline["final_accuracy"] - task["final_accuracy"]
                utility.append({"task_id": task["task_id"], "fedavg_task_id": baseline["task_id"],
                                "final_accuracy_loss": loss,
                                "last10_accuracy_loss": baseline["last10_mean_accuracy"]
                                - task["last10_mean_accuracy"],
                                "clean_accuracy_drop_gate_pass": loss <= .05 + 1e-12,
                                "observed_clean_health_pass": task["health"]["complete_observed_health_pass"]})
    groups = {(t["seed"], t["requested_rounds"], t["malicious_ratio"])
              for t in tasks.values() if t["malicious_ratio"]}
    for seed, horizon, ratio in sorted(groups):
        group = [t for t in tasks.values()
                 if (t["seed"], t["requested_rounds"], t["malicious_ratio"]) == (seed, horizon, ratio)]
        available = len(group) == 6 and {t["method"] for t in group} == set(METHODS) and all(
            t["complete"] for t in group)
        for field, direction in RANK_FIELDS.items():
            entry = {"seed": seed, "rounds": horizon, "malicious_ratio": ratio,
                     "metric": field, "preferred_direction": direction, "available": available}
            if available:
                entry["ordering"] = [
                    {"method": t["method"], "value": t[field],
                     "observed_health_pass": t["health"]["complete_observed_health_pass"]}
                    for t in sorted(group, key=lambda t: (
                        -t[field] if direction == "higher" else t[field], t["method"]))]
            rankings.append(entry)
    all_complete = bool(tasks) and not errors and all(t["complete"] for t in tasks.values())
    return {
        "protocol": manifest.get("protocol"), "fingerprint": manifest.get("fingerprint"),
        "planned_tasks": len(specs), "planned_rounds": sum(s["config"]["rounds"] for s in specs),
        "completed_tasks": sum(t["complete"] for t in tasks.values()), "all_complete": all_complete,
        "all_numerically_finite": all_complete and all(t["numerically_finite"] for t in tasks.values()),
        "all_observed_health_pass": all_complete and all(
            t["health"]["complete_observed_health_pass"] for t in tasks.values()),
        "reference_integrity_valid": not errors, "report_integrity_errors": errors,
        "task_runtime_seconds": math.fsum(t.get("elapsed_seconds", 0.) for t in tasks.values()),
        "official_test_used_for_selection": False, "tasks": tasks,
        "clean_comparisons": comparisons, "clean_utility_comparisons": utility,
        "six_method_rankings": rankings,
        "all_six_method_comparisons_available": bool(rankings) and all(r["available"] for r in rankings),
        "interpretation_limits": [
            "A single new seed and 60 rounds do not establish 150-round or multi-seed qualification.",
            "Actual malicious calls can stop after permanent revocation; nominal attack-active rounds differ.",
            "Clean calibration does not demonstrate optimal defense performance under attack.",
            "Krum at 60%/80% is outside its original 2*f+2<n guarantee.",
            "Completed health-failed baselines remain visible and are not automatically retried.",
        ],
    }
