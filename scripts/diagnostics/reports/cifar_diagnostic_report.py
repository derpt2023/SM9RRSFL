"""Read-only, missing-aware stage-1 summary. Never launches the next stage."""
import json
import math
from statistics import fmean

import cifar_adaptive_runtime as runtime

base = runtime.base


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def observations(folder, task):
    data = runtime.read_json(folder / "observations.json")
    if data.get("task_fingerprint") != task["fingerprint"]:
        raise ValueError("loss observations have the wrong task fingerprint")
    rows = data["rounds"]
    if [r["round"] for r in rows] != list(range(task["config"]["rounds"] + 1)):
        raise ValueError("loss observation rounds are incomplete")
    for row in rows:
        if not finite(row["calibration_loss"]) or row["calibration_loss"] < 0 or row["calibration_samples"] != 2500:
            raise ValueError("invalid calibration loss/sample count")
        if row["round"] and (not finite(row["local_train_loss"]) or row["local_train_loss"] < 0
                             or row["local_train_samples"] != 45000):
            raise ValueError("invalid local training loss/sample count")
    return rows


def decision(rows, settings):
    candidates = []
    for setting in settings:
        group = [r for r in rows if r["setting"] == setting["id"]]
        completed = [r for r in group if r["status"] == "complete"]
        candidates.append({"setting": setting["id"], "model": setting["model"],
            "complete": len(completed), "healthy": sum(r.get("healthy") is True for r in completed),
            "mean_accuracy": fmean(r["accuracy150"] for r in completed) if completed else None,
            "mean_background_5_to_7": fmean(r["background_5_to_7"] for r in completed) if completed else None,
            "mean_worker_hours": fmean(r["worker_wall_seconds"] / 3600 for r in completed)
                if completed and all(r["unfinished_attempts"] == 0 for r in completed) else None,
            "selection_eligible": len(completed) == 4 and all(r.get("healthy") is True for r in completed)})
    result = {"candidates": candidates, "next_stage_started": False,
              "formal_qualification_assessed": False, "cost_acceptability": "requires_review"}
    # Never crown a winner while one of the declared settings is unresolved.
    if len(rows) != 24 or any(r["status"] not in ("complete", "algorithm_numerical") for r in rows):
        return {**result, "action": "resolve_incomplete_or_invalid_evidence"}
    resnets = [c for c in candidates if c["model"] == "resnet18_gn2" and c["selection_eligible"]]
    if not resnets:
        return {**result, "action": "no_healthy_resnet_setting_review_cnn_and_failures"}
    winner = sorted(resnets, key=lambda c: (-c["mean_accuracy"], c["setting"]))[0]
    result["best_healthy_resnet"] = winner["setting"]
    if winner["setting"] != "R0":
        settings_by_id = {s["id"]: s for s in settings}
        return {**result, "action": "needs_four_matched_cnn_runs_before_architecture_decision",
                "cnn_followup_parameters": {k: settings_by_id[winner["setting"]][k]
                                             for k in ("lr", "local_epochs", "lr_decay")}}
    cnn = next(c for c in candidates if c["setting"] == "C0")
    if not cnn["selection_eligible"]:
        return {**result, "action": "cnn_reference_unhealthy_review_before_architecture_decision"}
    paired = {(r["partition"], r["seed"]): r for r in rows if r["setting"] == "C0"}
    gaps = [r["accuracy150"] - paired[(r["partition"], r["seed"])]["accuracy150"]
            for r in rows if r["setting"] == "R0"]
    passed = fmean(gaps) >= .02 - 1e-12 and min(gaps) >= -.01 - 1e-12
    return {**result, "paired_R0_minus_C0_pp": [100 * v for v in gaps],
        "paired_mean_gain_pp": 100 * fmean(gaps), "architecture_engineering_line_passed": passed,
        "action": "review_resnet_cost_before_retaining" if passed else "prefer_cnn_for_next_development_review_learning_curves"}


def summarize(output):
    from run_cifar_diagnostic import read_study, source_hashes
    try:
        manifest, tasks = read_study(output)
    except Exception as exc:
        return {"status": "unavailable_or_invalid_study", "output": str(output),
                "error": str(exc), "training_started_by_summary": False}
    rows = []
    for task in tasks:
        folder = output / "tasks" / task["task_id"]
        row = {"task_id": task["task_id"], "setting": task["candidate"]["candidate_id"],
            "partition": task["config"]["partition"], "seed": task["config"]["seed"],
            "status": "pending", "healthy": None}
        try:
            if folder.exists() and any(folder.iterdir()):
                if runtime.read_json(folder / "task.json") != task:
                    raise ValueError("task identity differs from the declared diagnostic task")
            result = base.checked_completed(output, task)
            if result is None:
                failure = runtime.terminal_failure(output, task)
                if failure:
                    row.update(status=failure["kind"], error=failure.get("message"),
                               exception=failure.get("exception"))
                progress_path = folder / "progress.json"
                if progress_path.exists():
                    row["last_completed_round"] = runtime.read_json(progress_path).get("last_completed_round")
                log = folder / "worker.log"
                if log.exists():
                    with log.open("rb") as handle:
                        handle.seek(max(0, log.stat().st_size - 2000))
                        row["log_tail"] = handle.read().decode("utf-8", errors="replace")[-1200:]
            else:
                metric = base.metrics(result)
                obs = observations(folder, task)
                records = {r.round: r for r in result.records}
                if (any(not finite(r.accuracy) or not 0 <= r.accuracy <= 1 for r in result.records)
                        or result.final_accuracy != records[150].accuracy):
                    raise ValueError("invalid or inconsistent recorded accuracy")
                background = records[150].attack_target_success_rate
                if not finite(background) or not 0 <= background <= 1:
                    raise ValueError("invalid clean source-to-target background confusion")
                attempts = [runtime.read_json(p) for p in sorted((folder / "attempts").glob("*.json"))]
                if not attempts or any(a.get("task_fingerprint") != task["fingerprint"] for a in attempts):
                    raise ValueError("worker cost provenance missing or mismatched")
                if any(a.get("status") != "running" and (not finite(a.get("wall_seconds")) or a["wall_seconds"] < 0)
                       for a in attempts):
                    raise ValueError("invalid worker elapsed time")
                row.update(status="complete", healthy=metric["healthy"], reasons=metric["reasons"],
                    accuracy50=records[50].accuracy, accuracy100=records[100].accuracy,
                    accuracy150=result.final_accuracy, background_5_to_7=background,
                    local_train_loss150=obs[-1]["local_train_loss"], calibration_loss150=obs[-1]["calibration_loss"],
                    losses50_100=[[obs[r]["local_train_loss"], obs[r]["calibration_loss"]] for r in (50, 100)],
                    worker_wall_seconds=sum(a.get("wall_seconds", 0) for a in attempts),
                    unfinished_attempts=sum(a.get("status") == "running" for a in attempts),
                    measured_runtime_seconds=result.runtime_seconds,
                    cuda_peak_allocated_mib=max((r["cuda_peak_allocated_mib"] for r in obs + attempts
                                                if finite(r.get("cuda_peak_allocated_mib"))), default=None),
                    nonfinite_updates=result.nonfinite_updates)
        except Exception as exc:
            row.update(status="invalid_evidence", error=str(exc))
        rows.append(row)
    complete = sum(r["status"] == "complete" for r in rows)
    resolved = all(r["status"] in ("complete", "algorithm_numerical") for r in rows)
    environment_path = output / "execution_environment.json"
    try:
        hardware = (runtime.read_json(environment_path).get("actual_compute_device")
                    if environment_path.exists() else None)
    except Exception as exc:
        hardware = {"read_error": str(exc)}
    return {"status": "complete" if complete == 24 else "resolved_with_numerical_failures" if resolved else "incomplete",
        "output": str(output), "protocol": manifest["protocol"],
        "execution_hardware": hardware,
        "manifest_fingerprint": manifest["fingerprint"], "source_matches_current": source_hashes() == manifest["source_sha256"],
        "complete_tasks": complete, "expected_tasks": 24, "healthy_tasks": sum(r.get("healthy") is True for r in rows),
        "evaluation_split": "2500 calibration samples; official test unused for selection",
        "background_metric": "clean source 5 -> target 7 confusion, not attack ASR",
        "loss_metric": "local sample-weighted client minibatch loss; calibration global-model CE",
        "training_started_by_summary": False, "rows": rows, "decision": decision(rows, manifest["spec"]["settings"])}


def print_summary(report):
    print("=== CIFAR_CLEAN_DIAGNOSTIC_BEGIN ===", flush=True)
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "decision")},
                     indent=2, ensure_ascii=False, allow_nan=False))
    for row in report.get("rows", []):
        print("TASK " + json.dumps(row, ensure_ascii=False, allow_nan=False))
    if "decision" in report:
        print("DECISION " + json.dumps(report["decision"], ensure_ascii=False, allow_nan=False))
    print("=== CIFAR_CLEAN_DIAGNOSTIC_END ===", flush=True)
