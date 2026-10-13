"""Read-only comparison of four C3 runs with the frozen stage-1 R3/C0 evidence."""
import json
from pathlib import Path
from statistics import fmean

import cifar_diagnostic_report as original

runtime, base = original.runtime, original.base
finite, observations = original.finite, original.observations
PAIRS = {(partition, seed) for partition in ("iid", "dirichlet")
         for seed in (2026093001, 2026093002)}


def collect(output, tasks):
    """Read snapshots and provenance without repairing or rewriting any evidence."""
    rows = []
    for task in tasks:
        folder = output / "tasks" / task["task_id"]
        row = {"task_id": task["task_id"], "setting": task["candidate"]["candidate_id"],
               "partition": task["config"]["partition"], "seed": task["config"]["seed"],
               "status": "pending", "healthy": None}
        try:
            if folder.exists() and any(folder.iterdir()):
                if runtime.read_json(folder / "task.json") != task:
                    raise ValueError("task identity differs from the declared matched CNN task")
            result = base.checked_completed(output, task)
            if result is None:
                failure = runtime.terminal_failure(output, task)
                if failure:
                    row.update(status=failure["kind"], error=failure.get("message"),
                               exception=failure.get("exception"))
                progress = folder / "progress.json"
                if progress.exists():
                    row["last_completed_round"] = runtime.read_json(progress).get("last_completed_round")
                log = folder / "worker.log"
                if log.exists():
                    with log.open("rb") as handle:
                        handle.seek(max(0, log.stat().st_size - 2000))
                        row["log_tail"] = handle.read().decode("utf-8", errors="replace")[-1200:]
            else:
                metric, obs = base.metrics(result), observations(folder, task)
                if ([r.round for r in result.records] != list(range(151))
                        or any(not finite(r.accuracy) or not 0 <= r.accuracy <= 1 for r in result.records)):
                    raise ValueError("incomplete or invalid recorded accuracy")
                records = {r.round: r for r in result.records}
                if result.final_accuracy != records[150].accuracy:
                    raise ValueError("inconsistent final accuracy")
                background = records[150].attack_target_success_rate
                if not finite(background) or not 0 <= background <= 1:
                    raise ValueError("invalid clean source-to-target background confusion")
                attempts = [runtime.read_json(p) for p in sorted((folder / "attempts").glob("*.json"))]
                if any(a.get("task_fingerprint") != task["fingerprint"] for a in attempts):
                    raise ValueError("worker cost provenance has a mismatched task fingerprint")
                if any(a.get("status") not in ("running", "failed", "complete") for a in attempts):
                    raise ValueError("invalid worker attempt status")
                if any(a.get("status") != "running" and
                       (not finite(a.get("wall_seconds")) or a["wall_seconds"] < 0) for a in attempts):
                    raise ValueError("invalid worker elapsed time")
                if any("wall_seconds" in a and
                       (not finite(a["wall_seconds"]) or a["wall_seconds"] < 0) for a in attempts):
                    raise ValueError("invalid worker elapsed time")
                unfinished = sum(a.get("status") == "running" for a in attempts)
                row.update(status="complete", healthy=metric["healthy"], reasons=metric["reasons"],
                    accuracy50=records[50].accuracy, accuracy100=records[100].accuracy,
                    accuracy150=result.final_accuracy, background_5_to_7=background,
                    local_train_loss150=obs[-1]["local_train_loss"], calibration_loss150=obs[-1]["calibration_loss"],
                    losses50_100=[[obs[r]["local_train_loss"], obs[r]["calibration_loss"]] for r in (50, 100)],
                    worker_wall_seconds=sum(a.get("wall_seconds", 0) for a in attempts) if attempts else None,
                    unfinished_attempts=unfinished if attempts else None,
                    cost_provenance="complete" if attempts and not unfinished else "unfinished_attempts" if attempts else "missing",
                    measured_runtime_seconds=result.runtime_seconds,
                    cuda_peak_allocated_mib=max((r["cuda_peak_allocated_mib"] for r in obs + attempts
                        if finite(r.get("cuda_peak_allocated_mib"))), default=None),
                    nonfinite_updates=result.nonfinite_updates)
        except Exception as exc:
            row.update(status="invalid_evidence", healthy=None, error=str(exc))
        rows.append(row)
    return rows


def _group(rows, setting):
    group = [r for r in rows if r.get("setting") == setting]
    mapped = {(r["partition"], r["seed"]): r for r in group}
    if len(group) != 4 or set(mapped) != PAIRS:
        raise ValueError("missing or duplicate paired rows for " + setting)
    return mapped


def _cost_exact(row):
    return (finite(row.get("worker_wall_seconds")) and row["worker_wall_seconds"] > 0
            and row.get("unfinished_attempts") == 0
            and row.get("cost_provenance", "complete") == "complete")


def decision(rows, reference_rows, *, reference_verified, environment_compatible, source_matches):
    result = {"next_stage_started": False, "formal_qualification_assessed": False,
              "architecture_engineering_line_passed": None,
              "engineering_line": "mean R3 minus C3 >= 2 pp and every pair >= -1 pp, inclusive",
              "cost_acceptability": "requires_review", "paired_comparison_available": False}
    if not reference_verified:
        return {**result, "action": "resolve_changed_or_invalid_reference"}
    if not source_matches:
        return {**result, "action": "resolve_changed_source_identity"}
    if not environment_compatible:
        return {**result, "action": "resolve_missing_or_incompatible_execution_environment"}
    try:
        c3, r3, c0 = (_group(rows, "C3"), _group(reference_rows, "R3"), _group(reference_rows, "C0"))
    except (ValueError, KeyError) as exc:
        return {**result, "action": "resolve_incomplete_or_invalid_evidence", "error": str(exc)}
    all_rows = list(c3.values()) + list(r3.values()) + list(c0.values())
    if any(r.get("status") != "complete" or not finite(r.get("accuracy150"))
           or not 0 <= r["accuracy150"] <= 1 for r in all_rows):
        return {**result, "action": "resolve_incomplete_or_invalid_evidence"}
    if any(r.get("healthy") is not True for r in all_rows):
        return {**result, "action": "review_unhealthy_completed_runs_before_model_decision"}
    pairs = []
    for key in sorted(PAIRS, key=lambda k: (k[1], k[0])):
        cnn, resnet, old_cnn = c3[key], r3[key], c0[key]
        cost_exact = _cost_exact(cnn) and _cost_exact(resnet)
        pairs.append({"partition": key[0], "seed": key[1],
            "R3_accuracy150": resnet["accuracy150"], "C3_accuracy150": cnn["accuracy150"],
            "C0_accuracy150": old_cnn["accuracy150"],
            "R3_minus_C3_pp": 100 * (resnet["accuracy150"] - cnn["accuracy150"]),
            "C3_minus_C0_pp": 100 * (cnn["accuracy150"] - old_cnn["accuracy150"]),
            "R3_over_C3_worker_time_ratio": resnet["worker_wall_seconds"] / cnn["worker_wall_seconds"] if cost_exact else None})
    mean_gap = fmean(p["R3_minus_C3_pp"] for p in pairs)
    passed = mean_gap >= 2. - 1e-10 and min(p["R3_minus_C3_pp"] for p in pairs) >= -1. - 1e-10
    cost_exact = all(_cost_exact(r) for r in list(c3.values()) + list(r3.values()))
    result.update(paired_comparison_available=True, paired_results=pairs,
        paired_mean_R3_minus_C3_pp=mean_gap,
        paired_min_R3_minus_C3_pp=min(p["R3_minus_C3_pp"] for p in pairs),
        paired_mean_C3_minus_C0_pp=fmean(p["C3_minus_C0_pp"] for p in pairs),
        mean_accuracy_R3=fmean(r["accuracy150"] for r in r3.values()),
        mean_accuracy_C3=fmean(r["accuracy150"] for r in c3.values()),
        mean_accuracy_C0=fmean(r["accuracy150"] for r in c0.values()),
        architecture_engineering_line_passed=passed,
        worker_cost_exact=cost_exact,
        R3_over_C3_total_worker_time_ratio=(sum(r["worker_wall_seconds"] for r in r3.values()) /
            sum(r["worker_wall_seconds"] for r in c3.values())) if cost_exact else None,
        mean_worker_hours_R3=fmean(r["worker_wall_seconds"] / 3600 for r in r3.values()) if cost_exact else None,
        mean_worker_hours_C3=fmean(r["worker_wall_seconds"] / 3600 for r in c3.values()) if cost_exact else None,
        cost_note="worker wall time includes recorded failed/retried attempts; missing or unfinished attempts prevent an exact comparison",
        interpretation="development engineering review only; R3 was selected using these development seeds; no independent validation or formal qualification")
    result["action"] = ("review_missing_cost_evidence_before_model_decision" if not cost_exact else
                        "review_resnet_cost_before_retaining" if passed else
                        "prefer_cnn_for_next_development_review_learning_curves")
    return result


def summarize(output, reference_output):
    from run_cifar_matched_cnn import read_study, source_hashes, audit_reference
    output, reference_output = Path(output), Path(reference_output)
    try:
        manifest, tasks = read_study(output, current_sources=False)
    except Exception as exc:
        return {"status": "unavailable_or_invalid_study", "output": str(output),
                "error": str(exc), "training_started_by_summary": False}
    reference = manifest["reference"]
    reference_verified, reference_error = False, None
    try:
        current_reference = audit_reference(reference_output,
            expected_fingerprint=manifest["spec"]["reference_manifest_fingerprint"])
        if current_reference != reference:
            raise ValueError("reference evidence differs from the immutable matched-study reference")
        reference_verified = True
    except Exception as exc:
        reference_error = str(exc)
    rows = collect(output, tasks)
    complete = sum(r["status"] == "complete" for r in rows)
    healthy = sum(r.get("healthy") is True for r in rows)
    environment, environment_error = None, None
    try:
        environment = runtime.read_json(output / "execution_environment.json")
    except Exception as exc:
        environment_error = str(exc)
    environment_compatible = (environment is not None and environment == reference["execution_environment"])
    source_matches, source_error = False, None
    try:
        source_matches = source_hashes() == manifest["source_sha256"]
    except Exception as exc:
        source_error = str(exc)
    evidence_ready = reference_verified and environment_compatible and source_matches
    status = ("invalid_comparison_evidence" if not evidence_ready else
              "complete" if complete == 4 else
              "resolved_with_numerical_failures" if all(r["status"] in ("complete", "algorithm_numerical") for r in rows) else
              "incomplete")
    result = {"status": status, "output": str(output), "reference_output": str(reference_output),
        "protocol": manifest["spec"]["protocol"], "manifest_fingerprint": manifest["fingerprint"],
        "reference_manifest_fingerprint": reference["manifest_fingerprint"],
        "source_matches_current": source_matches, "reference_verified": reference_verified,
        "execution_environment_compatible": environment_compatible,
        "execution_hardware": environment.get("actual_compute_device") if isinstance(environment, dict) else None,
        "complete_tasks": complete, "expected_tasks": 4, "healthy_tasks": healthy,
        "reference_complete_tasks": reference["complete_tasks"], "reference_healthy_tasks": reference["healthy_tasks"],
        "evaluation_split": "2500 calibration samples; official test unused for selection",
        "background_metric": "clean source 5 -> target 7 confusion, not attack ASR",
        "loss_metric": "local sample-weighted client minibatch loss; calibration global-model CE",
        "training_started_by_summary": False, "rows": rows,
        "reference_rows": reference["rows"] if reference_verified else [],
        "decision": decision(rows, reference["rows"], reference_verified=reference_verified,
                             environment_compatible=environment_compatible, source_matches=source_matches)}
    for key, value in (("reference_error", reference_error), ("environment_error", environment_error),
                       ("source_error", source_error)):
        if value is not None:
            result[key] = value
    return result


def print_summary(report):
    print("=== CIFAR_CNN_MATCH_BEGIN ===", flush=True)
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "reference_rows", "decision")},
                     indent=2, ensure_ascii=False, allow_nan=False))
    for row in report.get("reference_rows", []):
        print("REFERENCE_TASK " + json.dumps(row, ensure_ascii=False, allow_nan=False))
    for row in report.get("rows", []):
        print("TASK " + json.dumps(row, ensure_ascii=False, allow_nan=False))
    if "decision" in report:
        print("DECISION " + json.dumps(report["decision"], ensure_ascii=False, allow_nan=False))
    print("=== CIFAR_CNN_MATCH_END ===", flush=True)
