"""Read-only reporting for the fixed C7 AI Station comparison.

Execution, numerical validity, original v14 health gates and additional health
review are distinct. Failed methods remain in coverage and comparison tables.
No result from this module selects parameters or decides whether final runs start.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
METHODS = ("fedavg", "sm9rrs", "vert", "alignins", "krum", "ding13")
PROFILE = {"id": "C7", "optimizer": "adam", "lr": .001,
           "boost": 20., "stealth_steps": 10, "scope": "weights"}
METRICS = ("accuracy", "legacy_asr", "legacy_confidence", "training_target_success",
           "training_target_confidence", "source_transfer_asr", "source_transfer_confidence",
           "honest_weight_loss", "malicious_weight_mass")
SCOPES = {"validation": "training_validation", "final": "official_test"}
ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError)


def _load(path):
    def reject(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value, name, *, lo=0., hi=None, integer=False, tolerance=0.):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < lo - tolerance or (
            hi is not None and value > hi + tolerance):
        raise ValueError(f"{name} is non-finite or outside its range")
    if integer and int(value) != value:
        raise ValueError(f"{name} must be an integer")
    return value


def _finite_tree(value, name):
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{name}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} is non-finite")


def _semantic(config):
    return {key: value for key, value in config.items() if key not in ("device", "sm9_workers")}


def _task_path(root, spec):
    relative = Path(spec["task_dir"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("task_dir must stay inside its experiment output")
    expected = Path("c7_runtime") / spec["phase"] / _digest(_semantic(spec["config"]))
    if relative != expected:
        raise ValueError("task_dir differs from stage/semantic configuration key")
    return Path(root) / relative


def _check_targets(targets, identity, phase):
    if (targets.get("stage") != phase or targets.get("evaluation_scope") != SCOPES[phase]
            or targets.get("official_test_used_for_selection") is not False
            or targets.get("training_target_is_optimizer_input") is not True
            or targets.get("transfer_target_is_optimizer_input") is not False):
        raise ValueError("metric target scope differs from the declared stage")
    data = identity["data_identity"]
    arrays = data["arrays"]
    if set(arrays) != {"x_train", "y_train", "x_test", "y_test", "x_attack", "y_attack"}:
        raise ValueError("dataset array identity is incomplete")
    for name, descriptor in arrays.items():
        digest = descriptor["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"dataset hash is invalid: {name}")
        if not descriptor.get("shape") or not descriptor.get("dtype"):
            raise ValueError(f"dataset shape/dtype is missing: {name}")
    evaluation = _number(targets["evaluation_count"], "evaluation_count", lo=1, integer=True)
    transfer = _number(targets["source_transfer_count"], "source_transfer_count", lo=1, integer=True)
    auxiliary = _number(data["attack_auxiliary_count"], "attack_auxiliary_count", lo=1, integer=True)
    if (evaluation != data["evaluation_count"] or evaluation != arrays["y_test"]["shape"][0]
            or transfer != data["source_transfer_count"] or transfer > evaluation
            or auxiliary != arrays["y_attack"]["shape"][0]):
        raise ValueError("metric denominator contradicts dataset identity")
    counts = {"evaluation_count": evaluation, "source_transfer_count": transfer}
    for field, count_field, bound in (
            ("training_target_indices", "training_target_count", auxiliary),
            ("legacy_evaluation_indices", "legacy_target_count", evaluation),
            ("source_transfer_indices", "source_transfer_count", evaluation)):
        indices = targets[field]
        count = _number(targets[count_field], count_field, lo=1, integer=True)
        if (not isinstance(indices, list) or len(indices) != count or len(set(indices)) != count):
            raise ValueError(f"{field} does not match its denominator")
        for index in indices:
            _number(index, field, hi=bound - 1, integer=True)
        counts[count_field] = count
    if counts["training_target_count"] != 1 or counts["legacy_target_count"] != 1:
        raise ValueError("C7 retains exactly one auxiliary and one legacy evaluation target")
    return counts


def _check_rows(rows, config, completed, counts):
    if [row.get("round") for row in rows] != list(range(completed + 1)):
        raise ValueError("metrics lack the exact completed-round prefix")
    total = config["num_clients"]
    malicious = round(total * config["malicious_ratio"])
    honest = total - malicious
    nf = fp_previous = tp_previous = blacklist_previous = 0
    for row in rows:
        rid = _number(row["round"], "round", hi=config["rounds"], integer=True)
        if (row.get("method") != config["method"]
                or row.get("malicious_ratio") != config["malicious_ratio"]
                or row.get("attack_active") is not (
                    config["malicious_ratio"] > 0 and rid >= config["attack_start_round"])):
            raise ValueError(f"round {rid} method/ratio/attack timing differs")
        for metric in METRICS:
            _number(row[metric], f"round {rid} {metric}", hi=1.,
                    tolerance=1e-9 if metric == "malicious_weight_mass" else 0.)
        for count_field in ("training_target_count", "legacy_target_count", "source_transfer_count"):
            if _number(row[count_field], count_field, integer=True) != counts[count_field]:
                raise ValueError(f"round {rid} {count_field} denominator differs")
        for field, denominator in (("accuracy", counts["evaluation_count"]),
                                   ("training_target_success", counts["training_target_count"]),
                                   ("legacy_asr", counts["legacy_target_count"]),
                                   ("source_transfer_asr", counts["source_transfer_count"])):
            value = row[field] * denominator
            if abs(value - round(value)) > 1e-7:
                raise ValueError(f"round {rid} {field} contradicts its denominator")
        for old, new in (("attack_target_success_rate", "legacy_asr"),
                         ("attack_target_confidence", "legacy_confidence")):
            if old not in row or row[old] != row[new]:
                raise ValueError(f"round {rid} legacy metric differs from the original record")
        nf += _number(row["nonfinite_updates"], "nonfinite_updates", integer=True)
        if _number(row["cumulative_nonfinite_updates"], "cumulative_nonfinite_updates", integer=True) != nf:
            raise ValueError("nonfinite counters disagree")
        fp = _number(row["false_positive_revocations"], "false_positive_revocations", hi=honest, integer=True)
        tp = _number(row["true_positive_revocations"], "true_positive_revocations", hi=malicious, integer=True)
        blacklist = _number(row["blacklisted_clients"], "blacklisted_clients", hi=total, integer=True)
        if fp < fp_previous or tp < tp_previous or blacklist < blacklist_previous or blacklist != fp + tp:
            raise ValueError("cumulative revocation/blacklist counts disagree")
        for field in ("accepted_updates", "rejected_updates"):
            _number(row[field], field, hi=total, integer=True)
        calls = _number(row["attack_calls_this_round"], "attack_calls_this_round", hi=malicious, integer=True)
        if not row["attack_active"] and calls != 0:
            raise ValueError("attack calls occurred outside the active attack window")
        fp_previous, tp_previous, blacklist_previous = fp, tp, blacklist


def _health(task, rows, config):
    actual = [row for row in rows if row["round"] > 0]
    honest = config["num_clients"] - round(config["num_clients"] * config["malicious_ratio"])
    all_clients = [r["round"] for r in actual if r["blacklisted_clients"] == config["num_clients"]]
    all_honest = [r["round"] for r in actual if honest and r["false_positive_revocations"] == honest]
    high_fp = [r["round"] for r in actual if honest and r["false_positive_revocations"] > .1 * honest]
    empty = [r["round"] for r in actual if r["accepted_updates"] == 0]
    starvation, consecutive = [], 0
    for row in actual:
        consecutive = consecutive + 1 if row["honest_weight_loss"] >= .99 else 0
        if config["method"] == "sm9rrs" and consecutive >= 5:
            starvation.append(row["round"])
    original = [name for name, evidence in (
        ("all_clients_revoked", all_clients), ("all_honest_revoked", all_honest),
        ("clean_false_revocation_rate", high_fp if config["malicious_ratio"] == 0 else []),
        ("honest_weight_starvation", starvation)) if evidence]
    additional = [name for name, evidence in (
        ("high_honest_false_revocation_rate_under_attack", high_fp if config["malicious_ratio"] else []),
        ("empty_aggregation", empty)) if evidence]
    finite_complete = task["valid_complete"] and task["numerically_finite"] is True
    return {
        "v14_failure_reasons": original, "additional_review_reasons": additional,
        "all_clients_revoked_rounds": all_clients, "all_honest_revoked_rounds": all_honest,
        "high_false_revocation_rounds": high_fp, "empty_aggregation_rounds": empty,
        "ours_starvation_rounds": starvation,
        "honest_clients": honest,
        "final_false_positive_revocations": rows[-1]["false_positive_revocations"] if rows else None,
        "false_revocation_rate": rows[-1]["false_positive_revocations"] / honest if rows and honest else None,
        "pre_attack_false_positive_revocations": max(
            (r["false_positive_revocations"] for r in rows if r["round"] < config["attack_start_round"]), default=None),
        "original_v14_health_pass": bool(finite_complete and not original),
        "complete_observed_health_pass": bool(finite_complete and not original and not additional),
        "review_policy": "Original v14 gates retained; >10% honest false revocations under attack and empty aggregation are separately flagged.",
    }


def _windows(rows, config):
    horizon, start = config["rounds"], config["attack_start_round"]
    result = {}
    for name, first, last in (("final", horizon, horizon),
                              ("all_attack", start, horizon), ("tail10", max(1, horizon - 9), horizon)):
        selected = [row for row in rows if first <= row["round"] <= last]
        expected = list(range(first, last + 1))
        if not expected or [row["round"] for row in selected] != expected:
            result[name] = {"available": False, "expected_rounds": expected}
            continue
        result[name] = {"available": True, "rounds": expected, "n_rounds": len(selected),
                        "mean": {field: math.fsum(row[field] for row in selected) / len(selected)
                                 for field in METRICS}}
    return result


def _exposure(directory, rows, config, identity, include_attempt_audit=False):
    observed = [r for r in rows if r["round"] >= config["attack_start_round"]]
    result = {
        "committed_round_counts_available": bool(rows),
        "committed_attack_calls": sum(r["attack_calls_this_round"] for r in observed) if rows else None,
        "committed_calls_by_round": {str(r["round"]): r["attack_calls_this_round"] for r in observed},
        "audit_available": False,
        "note": "Committed metric counters describe effective execution; audit attempts can include repeated or uncommitted client-rounds.",
    }
    path = Path(directory) / "attack_audit.jsonl"
    if not path.exists():
        result["audit_note"] = "No per-call log; attempt counts are unknown, not zero."
        return result
    result.update(audit_available=True, audit_bytes=path.stat().st_size,
                  audit_attempt_counts_scanned=bool(include_attempt_audit))
    if not include_attempt_audit:
        result["audit_note"] = "Per-call log retained; default progress reports use committed counters without rescanning all attempts."
        return result
    seen, completed, repeated = set(), set(), set()
    normal = target = failures = attempts = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        _finite_tree(record, "attack audit")
        if (record.get("fingerprint") != identity["fingerprint"]
                or record.get("task_key") != identity["task_key"]
                or record.get("stage") != identity["stage"]):
            raise ValueError("per-call audit task identity differs")
        if record.get("event") == "first_numerical_failure":
            failures += 1
            continue
        for field, expected in (("profile_id", "C7"), ("optimizer", "adam"), ("lr", .001),
                                ("boost", 20.), ("stealth_steps", 10), ("scope", "weights")):
            if record.get(field) != expected:
                raise ValueError(f"per-call audit {field} differs from fixed C7")
        rid = _number(record["round"], "audit round", lo=config["attack_start_round"], hi=config["rounds"], integer=True)
        client = _number(record["client_idx"], "audit client", hi=config["num_clients"] - 1, integer=True)
        pair = (rid, client)
        if pair in seen:
            repeated.add(pair)
        seen.add(pair)
        attempts += 1
        if record.get("status") == "complete":
            completed.add(pair)
        normal += _number(record["normal_steps"], "audit normal_steps", integer=True)
        target += _number(record["target_steps"], "audit target_steps", integer=True)
    result.update(audit_available=True, audit_attempt_records=attempts,
                  audit_unique_client_rounds=len(seen), audit_unique_completed_client_rounds=len(completed),
                  audit_repeated_client_rounds=[list(pair) for pair in sorted(repeated)],
                  audit_attempt_normal_steps=normal, audit_attempt_target_steps=target,
                  audit_numerical_failure_events=failures)
    observed_counts = Counter(rid for rid, _client in completed)
    if any(observed_counts[row["round"]] < row["attack_calls_this_round"] for row in observed):
        raise ValueError("per-call audit lacks coverage for committed attack calls")
    return result


def _task_details(directory, spec, run_fingerprint=None, include_attempt_audit=False):
    config, phase = spec["config"], spec["phase"]
    task = {"task_dir": spec["task_dir"], "stage": phase, "candidate_id": spec["candidate_id"],
            "method": config["method"], "partition": config["partition"], "seed": config["seed"],
            "malicious_ratio": config["malicious_ratio"], "requested_rounds": config["rounds"],
            "status": "not_started", "complete": False, "valid_complete": False,
            "numerically_finite": None, "integrity_errors": [], "windows": {},
            "evaluation_scope": SCOPES.get(phase)}
    private, rows, valid_rows = {}, [], False
    try:
        directory = _task_path(directory, spec)
        if phase not in SCOPES or config["method"] not in METHODS:
            raise ValueError("unsupported stage/method")
        if spec.get("profile") != (PROFILE if config["malicious_ratio"] else None):
            raise ValueError("manifest attack profile differs from fixed C7")
        if not (directory / "status.json").exists():
            task["health"] = _health(task, [], config)
            return task, private
        status = _load(directory / "status.json")
        task["status"] = status["status"]
        task["completed_round"] = _number(status["completed_round"], "completed_round", lo=-1,
                                            hi=config["rounds"], integer=True)
        task["elapsed_seconds"] = _number(status["elapsed_seconds"], "elapsed_seconds")
        for key in ("error", "error_type", "timing_has_unclean_resume", "attempt_id"):
            if key in status:
                task[key] = status[key]
        failure_path = directory / "first_failure.json"
        if failure_path.exists():
            task["first_failure"] = _load(failure_path)
        identity = _load(directory / "task.json")
        fingerprint = identity["fingerprint"]
        if fingerprint != _digest({k: v for k, v in identity.items() if k not in ("fingerprint", "config")}):
            raise ValueError("task identity fingerprint mismatch")
        semantic = _semantic(config)
        if (identity.get("semantic_config") != semantic or _semantic(identity["config"]) != semantic
                or identity.get("task_key") != _digest(semantic)
                or identity.get("stage") != phase or identity.get("evaluation_scope") != SCOPES[phase]
                or identity.get("profile") != PROFILE
                or identity.get("attack_enabled") is not (config["malicious_ratio"] > 0)
                or (run_fingerprint is not None and identity.get("run_fingerprint") != run_fingerprint)):
            raise ValueError("task identity differs from manifest configuration/profile/stage")
        if (status.get("fingerprint") != fingerprint or status.get("requested_rounds") != config["rounds"]
                or status.get("task_id") != identity["task_id"]
                or status.get("task_key") != identity["task_key"]
                or status.get("stage") != phase
                or status.get("run_fingerprint") != identity["run_fingerprint"]):
            raise ValueError("status identity/horizon mismatch")
        private["identity"] = identity
        completed = task["completed_round"]
        if completed < 0:
            if status["status"] == "complete":
                raise ValueError("complete task has no completed rounds")
            task["numerically_finite"] = False if status["status"] == "numerical_failure" else None
            task["health"] = _health(task, [], config)
            return task, private
        targets = _load(directory / "targets.json")
        counts = _check_targets(targets, identity, phase)
        metrics = _load(directory / "metrics.json")
        if (metrics.get("fingerprint") != fingerprint or metrics.get("stage") != phase
                or metrics.get("evaluation_scope") != SCOPES[phase]
                or metrics.get("official_test_used_for_selection") is not False):
            raise ValueError("metric identity/evaluation scope differs")
        rows = metrics["rounds"]
        if status["status"] != "complete":
            task["uncommitted_metric_points"] = sum(row["round"] > completed for row in rows)
            rows = [row for row in rows if row["round"] <= completed]
        _finite_tree(rows, "metrics")
        _check_rows(rows, config, completed, counts)
        valid_rows = True
        private.update(rows=rows, targets=targets)
        task.update(metric_denominators=counts, observed_metric_points=len(rows),
                    observed_nonfinite_updates=rows[-1]["cumulative_nonfinite_updates"],
                    observed_last_metrics=rows[-1])
        task["actual_attack_exposure"] = _exposure(directory, rows, config, identity, include_attempt_audit)
        if status["status"] == "complete":
            if (completed != config["rounds"] or status.get("final_metrics") != rows[-1]
                    or status.get("nonfinite_updates") != rows[-1]["cumulative_nonfinite_updates"]):
                raise ValueError("complete status disagrees with the requested horizon/final metrics")
            result = _load(directory / "result.json")
            _finite_tree(result, "result")
            for key in ("status", "task_id", "fingerprint", "task_key", "stage", "run_fingerprint", "completed_round", "requested_rounds",
                        "final_metrics", "nonfinite_updates"):
                if result.get(key) != status.get(key):
                    raise ValueError(f"result/status disagree on {key}")
            if result.get("metric_scope") != targets:
                raise ValueError("result metric scope differs from actual targets")
            malicious = result["malicious_clients"]
            blacklist = result["blacklisted_clients"]
            all_clients = {f"client-{i}" for i in range(config["num_clients"])}
            if (len(malicious) != round(config["num_clients"] * config["malicious_ratio"])
                    or len(set(malicious)) != len(malicious) or not set(malicious) <= all_clients
                    or len(set(blacklist)) != len(blacklist) or not set(blacklist) <= all_clients
                    or len(blacklist) != rows[-1]["blacklisted_clients"]
                    or len(set(blacklist) - set(malicious)) != rows[-1]["false_positive_revocations"]):
                raise ValueError("actual malicious/blacklisted identities contradict round metrics")
            summary = result["summary"]
            for old, new in (("final_accuracy", "accuracy"),
                             ("final_attack_target_success_rate", "legacy_asr"),
                             ("final_attack_target_confidence", "legacy_confidence")):
                if summary.get(old) != rows[-1][new]:
                    raise ValueError(f"original summary {old} differs from final round")
            private["malicious_clients"] = sorted(malicious)
            task.update(complete=True, valid_complete=True,
                        numerically_finite=status["nonfinite_updates"] == 0,
                        nonfinite_updates=status["nonfinite_updates"],
                        windows=_windows(rows, config), final_metrics=rows[-1],
                        summary=summary, attack_seconds=_number(result["attack_seconds"], "attack_seconds"))
        elif status["status"] == "numerical_failure" or rows[-1]["cumulative_nonfinite_updates"]:
            task["numerically_finite"] = False
    except ERRORS as error:
        task["integrity_errors"].append(str(error))
        task["complete"] = task["valid_complete"] = False
        task["windows"] = {}
        task["numerically_finite"] = False if (
            "non-finite" in str(error) or task["status"] == "numerical_failure") else None
    task["health"] = _health(task, rows if valid_rows else [], config)
    return task, private


def task_summary(output_dir, spec, run_fingerprint=None, *, include_attempt_audit=False):
    """Read the manifest-declared task directory; never execute or modify a task."""
    return _task_details(output_dir, spec, run_fingerprint, include_attempt_audit)[0]


def _pair(clean, attacked, left, right):
    pair = {"clean_task_dir": clean["task_dir"], "attack_task_dir": attacked["task_dir"],
            "stage": attacked["stage"], "method": attacked["method"], "partition": attacked["partition"],
            "seed": attacked["seed"], "malicious_ratio": attacked["malicious_ratio"], "available": False}
    if not clean["valid_complete"] or not attacked["valid_complete"]:
        pair["reason"] = "Both exact completed tasks are required; failed/partial tasks remain reported."
        return pair
    a, b = dict(left["identity"]["semantic_config"]), dict(right["identity"]["semantic_config"])
    a.pop("malicious_ratio")
    b.pop("malicious_ratio")
    if (a != b or left["targets"] != right["targets"]
            or left["identity"]["data_identity"] != right["identity"]["data_identity"]):
        pair["reason"] = "Scientific config/data/targets differ beyond malicious ratio."
        return pair
    pair["available"] = True
    pair["both_numerically_finite"] = clean["numerically_finite"] and attacked["numerically_finite"]
    pair["clean_health_pass"] = clean["health"]["complete_observed_health_pass"]
    pair["attack_health_pass"] = attacked["health"]["complete_observed_health_pass"]
    pair["windows"] = {}
    for window in ("final", "all_attack", "tail10"):
        c, a = clean["windows"][window], attacked["windows"][window]
        if not c["available"] or not a["available"]:
            pair["windows"][window] = {"available": False}
            continue
        cm, am = c["mean"], a["mean"]
        gain = am["training_target_confidence"] - cm["training_target_confidence"]
        high_clean = cm["training_target_success"] >= .8 and cm["training_target_confidence"] >= .8
        pair["windows"][window] = {
            "available": True, "accuracy_loss": cm["accuracy"] - am["accuracy"],
            "training_target_confidence_gain": gain,
            "training_target_hit_rate": am["training_target_success"],
            "clean_already_high_confidence_target_error": high_clean,
            "fedavg_attack_effectiveness_confirmed": bool(
                attacked["method"] == "fedavg" and pair["both_numerically_finite"]
                and am["training_target_success"] >= .8 - 1e-12 and gain >= .2 - 1e-12 and not high_clean),
        }
    return pair


def _coverage(tasks):
    complete = [task for task in tasks if task["valid_complete"]]
    return {
        "planned_tasks": len(tasks), "completed_tasks": len(complete),
        "planned_rounds": sum(task["requested_rounds"] for task in tasks),
        "observed_completed_rounds": sum(max(0, task.get("completed_round", 0)) for task in tasks),
        "status_counts": dict(Counter(task["status"] for task in tasks)),
        "all_complete": bool(tasks) and len(complete) == len(tasks),
        "all_numerically_finite": bool(tasks) and len(complete) == len(tasks)
            and all(task["numerically_finite"] is True for task in tasks),
        "all_original_v14_health_pass": bool(tasks) and all(
            task["health"]["original_v14_health_pass"] for task in tasks),
        "all_observed_health_pass": bool(tasks) and all(
            task["health"]["complete_observed_health_pass"] for task in tasks),
        "failed_or_invalid_task_dirs": [task["task_dir"] for task in tasks if (
            task["integrity_errors"] or task["status"] in ("failed", "numerical_failure"))],
    }


def _phase_bindings(root, manifest):
    bindings, errors = {}, {phase: [] for phase in SCOPES}
    path = root / "controller_status.json"
    controller = _load(path) if path.exists() else {}
    if controller and controller.get("fingerprint") != manifest["fingerprint"]:
        raise ValueError("controller identity differs from protocol manifest")
    for phase in SCOPES:
        phase_state = controller.get("phases", {}).get(phase, {})
        fingerprint = phase_state.get("fingerprint")
        if not fingerprint:
            if any((root / spec["task_dir"] / "status.json").exists()
                   for spec in manifest["task_specs"] if spec["phase"] == phase):
                errors[phase].append("started tasks lack controller phase fingerprint")
            continue
        try:
            if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(
                    char not in "0123456789abcdef" for char in fingerprint):
                raise ValueError("invalid phase fingerprint")
            path = root / ".tuning_state" / phase / fingerprint / "run_manifest.json"
            run = _load(path)
            if (run.get("fingerprint") != fingerprint or run.get("tuning_phase") != phase
                    or fingerprint != _digest({k: v for k, v in run.items() if k != "fingerprint"})):
                raise ValueError("phase run manifest fingerprint/phase mismatch")
            context = run["tuning_context"]
            if (context.get("protocol_fingerprint") != manifest["fingerprint"]
                    or context.get("c7_runtime_protocol") != "cifar_c7_station_runtime_v1"
                    or context.get("c7_attack_profile") != PROFILE
                    or context.get("evaluation_scope") != SCOPES[phase]
                    or context.get("parameter_selection") != "fixed_before_validation"):
                raise ValueError("phase run manifest does not bind the fixed C7 protocol/scope")
            actual = [_digest(_semantic(item["config"])) for item in run["candidates"]]
            expected = [_digest(_semantic(spec["config"])) for spec in manifest["task_specs"]
                        if spec["phase"] == phase]
            if sorted(actual) != sorted(expected):
                raise ValueError("phase manifest candidate coverage differs from protocol")
            bindings[phase] = fingerprint
        except ERRORS as error:
            errors[phase].append(str(error))
    return bindings, errors


def build_report(output_dir, *, include_attempt_audit=False):
    """Return JSON-safe observations for protocol_manifest.json and its runtime files."""
    root = Path(output_dir)
    errors, specs, manifest, tasks, private = [], [], {}, {}, {}
    try:
        manifest = _load(root / "protocol_manifest.json")
        if manifest["fingerprint"] != _digest({k: v for k, v in manifest.items() if k != "fingerprint"}):
            raise ValueError("protocol manifest fingerprint mismatch")
        specs = manifest["task_specs"]
        if not specs or len({spec["task_dir"] for spec in specs}) != len(specs):
            raise ValueError("manifest task coverage is empty or duplicated")
        for path, expected in manifest["source_files"].items():
            if _sha(ROOT / path) != expected:
                raise ValueError(f"frozen source changed: {path}")
    except ERRORS as error:
        errors.append(str(error))
    phase_fingerprints, phase_errors = {}, {phase: [] for phase in SCOPES}
    if specs and not errors:
        try:
            phase_fingerprints, phase_errors = _phase_bindings(root, manifest)
        except ERRORS as error:
            errors.append(str(error))
    for spec in specs:
        task, detail = _task_details(root, spec, phase_fingerprints.get(spec["phase"]), include_attempt_audit)
        if errors or phase_errors.get(spec["phase"]):
            task.update(complete=False, valid_complete=False, windows={})
            task["integrity_errors"].append("protocol/source integrity failed")
            task["integrity_errors"].extend(phase_errors.get(spec["phase"], []))
            task["health"]["original_v14_health_pass"] = False
            task["health"]["complete_observed_health_pass"] = False
        tasks[spec["task_dir"]], private[spec["task_dir"]] = task, detail
    # Same-stage seed controls evaluation/auxiliary targets; final and validation
    # remain separate because their evaluation data and image indices differ.
    valid = [task for task in tasks.values() if task["valid_complete"]]
    for key in {(task["stage"], task["seed"]) for task in valid}:
        group = [task for task in valid if (task["stage"], task["seed"]) == key]
        first = private[group[0]["task_dir"]]
        if any(private[t["task_dir"]]["targets"] != first["targets"] or
               private[t["task_dir"]]["identity"]["data_identity"] != first["identity"]["data_identity"]
               for t in group):
            for task in group:
                task.update(complete=False, valid_complete=False, windows={})
                task["integrity_errors"].append("same-stage/seed data or targets differ across methods")
                task["health"]["original_v14_health_pass"] = False
                task["health"]["complete_observed_health_pass"] = False
        for ratio in {task["malicious_ratio"] for task in group}:
            subgroup = [t for t in group if t["malicious_ratio"] == ratio]
            malicious = private[subgroup[0]["task_dir"]]["malicious_clients"]
            if any(private[t["task_dir"]]["malicious_clients"] != malicious for t in subgroup):
                for task in subgroup:
                    task.update(complete=False, valid_complete=False, windows={})
                    task["integrity_errors"].append("same-stage/seed/ratio malicious identities differ")
                    task["health"]["original_v14_health_pass"] = False
                    task["health"]["complete_observed_health_pass"] = False
    pairs, utility = [], []
    for task in tasks.values():
        matched = [t for t in tasks.values() if (t["stage"], t["partition"], t["seed"], t["requested_rounds"])
                   == (task["stage"], task["partition"], task["seed"], task["requested_rounds"])]
        if task["malicious_ratio"]:
            clean = [t for t in matched if t["method"] == task["method"] and t["malicious_ratio"] == 0]
            if len(clean) == 1:
                pairs.append(_pair(clean[0], task, private[clean[0]["task_dir"]], private[task["task_dir"]]))
            else:
                pairs.append({"attack_task_dir": task["task_dir"], "available": False,
                              "reason": "Exactly one matching same-method clean task is required."})
        else:
            controls = [t for t in matched if t["method"] == "fedavg" and t["malicious_ratio"] == 0]
            entry = {"task_dir": task["task_dir"], "available": False}
            if len(controls) == 1 and task["valid_complete"] and controls[0]["valid_complete"]:
                base = controls[0]
                losses = {w: base["windows"][w]["mean"]["accuracy"] - task["windows"][w]["mean"]["accuracy"]
                          for w in ("final", "all_attack", "tail10") if task["windows"][w]["available"]}
                entry.update(available=True, fedavg_task_dir=base["task_dir"], accuracy_loss=losses,
                             final_clean_accuracy_drop_pass=losses["final"] <= .05 + 1e-12,
                             tail10_clean_accuracy_drop_pass=losses["tail10"] <= .05 + 1e-12,
                             observed_clean_health_pass=task["health"]["complete_observed_health_pass"])
            utility.append(entry)
    aggregates = []
    groups = {(t["stage"], t["partition"], t["malicious_ratio"], t["method"]) for t in tasks.values()}
    for stage, partition, ratio, method in sorted(groups):
        group = [t for t in tasks.values() if (t["stage"], t["partition"], t["malicious_ratio"], t["method"])
                 == (stage, partition, ratio, method)]
        included = [t for t in group if t["valid_complete"]]
        entry = {"stage": stage, "partition": partition, "malicious_ratio": ratio, "method": method,
                 "planned_n": len(group), "observed_n": len(included),
                 "complete_coverage": len(included) == len(group),
                 "included_seeds": [t["seed"] for t in included],
                 "missing_or_failed_seeds": [t["seed"] for t in group if not t["valid_complete"]],
                 "numerically_failed_seeds_included": [t["seed"] for t in included if not t["numerically_finite"]],
                 "health_failed_seeds_included": [t["seed"] for t in included
                                                  if not t["health"]["complete_observed_health_pass"]],
                 "health_filter_applied": False,
                 "windows": {}}
        if included:
            for window in ("final", "all_attack", "tail10"):
                available = [t for t in included if t["windows"][window]["available"]]
                entry["windows"][window] = {"observed_n": len(available), "mean": {
                    metric: math.fsum(t["windows"][window]["mean"][metric] for t in available) / len(available)
                    for metric in METRICS} if available else {}}
        aggregates.append(entry)
    coverage = _coverage(list(tasks.values()))
    return {
        "protocol": manifest.get("protocol"), "fingerprint": manifest.get("fingerprint"),
        "parameter_selection": "fixed configurations; no best-parameter selection",
        "official_test_used_for_selection": False, "final_evaluation_uses_official_test": True,
        "reference_integrity_valid": not errors and not any(phase_errors.values()),
        "report_integrity_errors": errors + [f"{phase}: {message}" for phase, messages in phase_errors.items() for message in messages]
            + [f"{task['task_dir']}: {message}" for task in tasks.values() for message in task["integrity_errors"]],
        "phase_fingerprints": phase_fingerprints,
        **coverage, "tasks": tasks,
        "stages": {phase: _coverage([t for t in tasks.values() if t["stage"] == phase]) for phase in SCOPES},
        "clean_comparisons": pairs, "clean_utility_comparisons": utility, "seed_aggregates": aggregates,
        "interpretation_limits": [
            "Every declared task remains reported; health-failed complete methods are not dropped from means.",
            "Partial/failed tasks have no final/all-attack/tail10 completed-run statistics; observed n is explicit.",
            "Validation evaluates training-validation data; final evaluates the official test set with separate denominators.",
            "Auxiliary-target ASR uses an optimizer input image; transfer ASR uses independent source-class evaluation images.",
            "Original v14 gates and additional high-FP/empty-aggregation review remain distinct; 95% honest revocation is not healthy.",
            "Nominal attack-active rounds do not guarantee live malicious calls; committed calls and all audit attempts differ.",
            "Krum outside 2*f+2<n and Ding13's true-ratio prior require separate interpretation.",
            "No validation health outcome in this report selects parameters or suppresses final-stage observations.",
        ],
    }
