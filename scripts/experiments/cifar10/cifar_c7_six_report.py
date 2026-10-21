"""Integrity-gated, read-only six-method C7 comparison; no outcome-based selection."""
from __future__ import annotations

import json
from pathlib import Path

from .cifar_c7_replication_report import (
    PROFILE, METRICS, _digest, _load, _number, _summarize,
    build_report as replication_report,
)

NEW_METHODS = ("alignins", "krum", "ding13")
REFERENCE_METHODS = ("fedavg", "sm9rrs", "vert")
METHODS = REFERENCE_METHODS + NEW_METHODS
TASKS = {f"C7_{method}_dir{ratio}_s401": (method, ratio / 100, 401)
         for ratio in (60, 80) for method in NEW_METHODS}
REFERENCES = {f"C7_{method}_dir{ratio}_s401": (method, ratio / 100, 401)
              for ratio in (60, 80) for method in REFERENCE_METHODS}
RANKING_FIELDS = {
    "final_accuracy": "higher",
    "attack_window_mean_accuracy": "higher",
    "attack_window_mean_training_target_success": "lower",
    "attack_window_mean_source_transfer_asr": "lower",
    "attack_window_mean_legacy_asr": "lower",
}


def _invalidate(task, message):
    task["integrity_errors"].append(message)
    task["valid_complete"] = False
    task["numerically_finite"] = None


def _manifest(directory):
    value = _load(Path(directory) / "manifest.json")
    if value.get("fingerprint") != _digest({k: v for k, v in value.items() if k != "fingerprint"}):
        raise ValueError("manifest fingerprint mismatch")
    return value


def _bind_task(directory, task, details, manifest):
    """Tie metrics to the exact declared candidate, public data and result."""
    if not details.get("identity"):
        return
    try:
        identity = details["identity"]
        specs = {spec["task_id"]: spec for spec in manifest["task_specs"]}
        spec = specs[task["task_id"]]
        expected_config = {k: v for k, v in spec["config"].items() if k != "device"}
        if (identity["root_fingerprint"] != manifest["fingerprint"]
                or identity["config"] != expected_config
                or identity["profile"] != spec["profile"]
                or identity["data_config"] != manifest["data_config"]):
            raise ValueError("task differs from its declared manifest specification")
        if task["valid_complete"]:
            path = Path(directory) / "tasks" / task["task_id"]
            details["dataset"] = _load(path / "dataset_identity.json")
            if not details["dataset"].get("arrays") or not details["dataset"].get("contract"):
                raise ValueError("dataset identity is incomplete")
            result, status = _load(path / "result.json"), _load(path / "status.json")
            for key in ("status", "fingerprint", "completed_round", "final_metrics", "nonfinite_updates"):
                if result.get(key) != status.get(key):
                    raise ValueError(f"result/status disagree on {key}")
            clients = result["summary"]["malicious_clients"]
            clients = clients.split(",") if clients else []
            if len(clients) != round(100 * task["malicious_ratio"]) or len(set(clients)) != len(clients):
                raise ValueError("malicious client evidence differs from configured ratio")
            details["malicious_clients"] = sorted(clients)
    except (OSError, KeyError, TypeError, ValueError) as error:
        _invalidate(task, str(error))


def _health(task, details):
    rows = [row for row in details.get("rows", []) if row["round"] > 0]
    honest = 100 - round(100 * task["malicious_ratio"])
    all_clients, all_honest, starvation = [], [], []
    consecutive = 0
    try:
        for row in rows:
            blacklisted = _number(row["blacklisted_clients"], "blacklisted_clients", hi=100, integer=True)
            if blacklisted >= 100:
                all_clients.append(row["round"])
            if row["false_positive_revocations"] >= honest:
                all_honest.append(row["round"])
            if task["method"] == "sm9rrs":
                consecutive = consecutive + 1 if row["honest_weight_loss"] >= .99 else 0
                if consecutive >= 5:
                    starvation.append(row["round"])
    except (KeyError, ValueError, TypeError) as error:
        _invalidate(task, str(error))
    reasons = (["all_clients_revoked"] if all_clients else []) + (
        ["all_honest_revoked"] if all_honest else []) + (
        ["honest_weight_starvation"] if starvation else [])
    if task.get("observed_nonfinite_updates", 0) or task["status"] == "numerical_failure":
        reasons.append("nonfinite_updates")
    return {
        "observed_failure_reasons": reasons,
        "all_clients_revoked_rounds": all_clients, "all_honest_revoked_rounds": all_honest,
        "ours_starvation_consecutive_round_gate_rounds": starvation,
        "complete_observed_health_pass": bool(task["valid_complete"]
                                              and task["numerically_finite"] and not reasons),
        "attacked_false_revocation_rate_exceeds_0_10": (
            task["false_revocation_rate"] > .1 if task["valid_complete"] else None),
        "formal_clean_health_qualified": False,
        "note": "Observed short-run health only; no matched seed401 clean qualification.",
    }


def _attack_exposure(directory, task):
    """Distinguish actual attack calls from nominal attack-active metric rows."""
    path = Path(directory) / "tasks" / task["task_id"] / "audit.jsonl"
    output = {
        "available": False, "integrity_errors": [],
        "note": "Audit counts observed call attempts; interrupted rounds can repeat after resume.",
    }
    if not path.exists():
        output["reason"] = "audit file absent; no zero-call claim is made"
        return output
    try:
        counts = {r: {"call_attempts": 0, "completed_calls": 0,
                      "normal_steps": 0, "target_steps": 0,
                      "target_skipped_zero_loss": 0} for r in range(25, 31)}
        seen, duplicates = set(), set()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rid = _number(row["round"], "audit round", lo=25, hi=30, integer=True)
            if (row.get("task_id") != task["task_id"] or row.get("profile_id") != "C7"
                    or row.get("optimizer") != "adam" or row.get("scope") != "weights"
                    or row.get("lr") != .001 or row.get("boost") != 20
                    or row.get("stealth_steps") != 10):
                raise ValueError("attack audit identity/profile differs from C7")
            pair = (rid, row["client_id"])
            if pair in seen:
                duplicates.add(pair)
            seen.add(pair)
            current = counts[rid]
            current["call_attempts"] += 1
            current["completed_calls"] += row["status"] == "complete"
            for key in ("normal_steps", "target_steps", "target_skipped_zero_loss"):
                current[key] += _number(row[key], f"audit {key}", integer=True)
        output.update(
            available=True, by_round={str(r): value for r, value in counts.items()},
            totals={key: sum(row[key] for row in counts.values())
                    for key in ("call_attempts", "completed_calls", "normal_steps",
                                "target_steps", "target_skipped_zero_loss")},
            duplicate_client_round_pairs=[{"round": r, "client_id": client}
                                          for r, client in sorted(duplicates)],
            unique_client_rounds=len(seen),
            completed_run_counts_without_duplicate_attempts=(
                task["valid_complete"] and not duplicates
                and all(r["call_attempts"] == r["completed_calls"] for r in counts.values())),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        output["integrity_errors"].append(str(error))
    return output


def _trend(rows, start, stop, metric):
    selected = [row for row in rows if start <= row["round"] <= stop]
    if [row["round"] for row in selected] != list(range(start, stop + 1)):
        return {"available": False}
    xs = [row["round"] for row in selected]
    ys = [row[metric] for row in selected]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)
    return {
        "available": True, "rounds": xs, "values": ys,
        "end_minus_start": ys[-1] - ys[0],
        "range": max(ys) - min(ys), "ols_slope_per_round": slope,
        "statistical_significance": "not_estimated; dependent points from one training run",
    }


def _rankings(tasks):
    results = []
    for ratio in (60, 80):
        group = [tasks[f"C7_{method}_dir{ratio}_s401"] for method in METHODS]
        for field, direction in RANKING_FIELDS.items():
            row = {"malicious_ratio": ratio / 100, "metric": field,
                   "preferred_direction": direction, "available": False}
            if all(task["valid_complete"] for task in group):
                order = sorted(group, key=lambda task: (
                    -task[field] if direction == "higher" else task[field], task["method"]))
                row.update(available=True, ordering=[
                    {"method": task["method"], "value": task[field],
                     "observed_health_pass": task["health"]["complete_observed_health_pass"]}
                    for task in order])
            results.append(row)
    return results


def build_report(output_dir, reference_dir):
    """Summarize all twelve attacked tasks without modifying either experiment."""
    output_dir, reference_dir = Path(output_dir), Path(reference_dir)
    tasks, details, references, reference_details = {}, {}, {}, {}
    errors = []
    try:
        manifest, reference = _manifest(output_dir), _manifest(reference_dir)
        if manifest["protocol"]["reference_fingerprint"] != reference["fingerprint"]:
            raise ValueError("supplement names a different replication")
        # The reference's configured prior directory is repository-relative.
        root = Path(__file__).resolve().parents[3]
        old = replication_report(reference_dir, root / reference["protocol"]["reference_dir"])
        reference_valid = (
            _load(reference_dir / "controller_status.json")["status"] == "complete"
            and old["all_complete"] and old["all_numerically_finite"]
            and old["reference_integrity_valid"] and old["all_comparisons_available"])
    except (OSError, KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
        manifest = reference = None
        reference_valid = False
    for directory, specifications, reports, private, declared in (
            (output_dir, TASKS, tasks, details, manifest),
            (reference_dir, REFERENCES, references, reference_details, reference)):
        for task_id, spec in specifications.items():
            reports[task_id], private[task_id] = _summarize(directory, task_id, spec)
            if declared is not None:
                _bind_task(directory, reports[task_id], private[task_id], declared)
            elif reports[task_id]["valid_complete"]:
                _invalidate(reports[task_id], "missing or invalid manifest")
    combined = {**references, **tasks}
    private = {**reference_details, **details}
    observed = [task for task in combined.values() if task["valid_complete"]]
    if observed:
        first = private[observed[0]["task_id"]]
        for field in ("dataset", "targets"):
            if any(private[task["task_id"]].get(field) != first.get(field) for task in observed[1:]):
                for task in observed:
                    _invalidate(task, f"same-seed cross-method {field} identities differ")
    for ratio in (.6, .8):
        group = [task for task in observed if task["malicious_ratio"] == ratio]
        if group and any(private[task["task_id"]].get("malicious_clients")
                         != private[group[0]["task_id"]].get("malicious_clients")
                         for task in group[1:]):
            for task in group:
                _invalidate(task, "same-ratio cross-method malicious client identities differ")
    for task_id, task in combined.items():
        task["actual_attack_exposure"] = _attack_exposure(
            output_dir if task_id in tasks else reference_dir, task)
        for error in task["actual_attack_exposure"]["integrity_errors"]:
            _invalidate(task, error)
        task["health"] = _health(task, private[task_id])
        if task["valid_complete"]:
            rows = private[task_id]["rows"]
            task["late_round_trends"] = {
                f"{start}_{stop}": {metric: _trend(rows, start, stop, metric)
                                   for metric in METRICS}
                for start, stop in ((25, 30), (27, 30))}
            task["trajectory_24_30"] = [
                {key: row[key] for key in ("round", *METRICS, "blacklisted_clients",
                                           "false_positive_revocations", "nonfinite_updates")}
                for row in rows if 24 <= row["round"] <= 30]
        if task["method"] == "krum":
            f = round(100 * task["malicious_ratio"])
            task["method_context"] = {
                "configured_byzantine_count": f, "original_condition": "2*f+2 < n",
                "original_condition_holds": 2 * f + 2 < 100,
                "interpretation": "Stress test outside Krum's original Byzantine-count guarantee.",
            }
        elif task["method"] == "ding13":
            task["method_context"] = {
                "uses_actual_malicious_ratio_prior": True,
                "interpretation": "Original implementation may revoke clients before attack round 25.",
            }
        elif task["method"] in ("vert", "alignins"):
            task["method_context"] = {"candidate": f"{task['method']}-001",
                                      "optimized_for_this_protocol": False}
    reference_valid = reference_valid and all(t["valid_complete"] for t in references.values())
    rankings = _rankings(combined)
    comparisons = []
    for ratio in (60, 80):
        ours = combined[f"C7_sm9rrs_dir{ratio}_s401"]
        for method in METHODS:
            if method == "sm9rrs":
                continue
            other = combined[f"C7_{method}_dir{ratio}_s401"]
            comparison = {"malicious_ratio": ratio / 100, "other_method": method,
                          "available": ours["valid_complete"] and other["valid_complete"]}
            if comparison["available"]:
                comparison["ours_minus_other"] = {field: ours[field] - other[field]
                                                   for field in RANKING_FIELDS}
                comparison["both_observed_healthy"] = (
                    ours["health"]["complete_observed_health_pass"]
                    and other["health"]["complete_observed_health_pass"])
            comparisons.append(comparison)
    valid = sum(task["valid_complete"] for task in tasks.values())
    finite = valid == len(TASKS) and all(task["numerically_finite"] for task in tasks.values())
    return {
        "protocol": "cifar_c7_six_methods_v1", "profile": dict(PROFILE),
        "planned_tasks": len(TASKS), "planned_rounds": 180,
        "completed_tasks": sum(task["status"] == "complete" for task in tasks.values()),
        "valid_complete_tasks": valid, "all_complete": valid == len(TASKS),
        "all_numerically_finite": finite, "reference_integrity_valid": reference_valid,
        "all_six_method_comparisons_available": (
            reference_valid and all(row["available"] for row in rankings)),
        "report_integrity_errors": errors,
        "task_runtime_seconds": sum(task.get("elapsed_seconds", 0.) for task in tasks.values()),
        "official_test_used_for_selection": False, "tasks": tasks, "reference_tasks": references,
        "six_method_rankings": rankings, "ours_pairwise_comparisons": comparisons,
        "all_observed_health_pass": all(task["health"]["complete_observed_health_pass"]
                                        for task in combined.values()),
        "seed401_matched_clean_available": False,
        "limitations": [
            "Fixed first-grid candidates; this does not establish superiority over a tuned SOTA method.",
            "Only seed401 is available for all six methods; no multi-seed six-method significance.",
            "Thirty rounds include only six attack rounds; neither convergence nor lasting protection is established.",
            "Late-round slopes are descriptive, not independent observations or a convergence test.",
            "Seed401 has no matched clean control; confidence-gain attack qualification is unavailable.",
            "Krum at 60%/80% is outside 2*f+2<n; its result cannot validate its original guarantee.",
            "Ding13 uses the actual malicious-ratio prior; failed revocation health is reported beside low ASR.",
            "Actual attack calls/steps are separately reported; nominal attack_active is not proof that a revoked client attacked.",
            "No outcome or expected ranking changes any method, attack, seed or task inclusion.",
        ],
    }
