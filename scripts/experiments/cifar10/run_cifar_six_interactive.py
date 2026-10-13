#!/usr/bin/env python3
"""Interactive v7 promotion without changing frozen training identities.

The original v7 module remains the worker, protocol and metric implementation.
Only a user-confirmed continuation may proceed after unmet performance targets.
Every invocation asks again; a previous choice fixes parameters, not consent.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import run_cifar_six_relative_best as original

REPO = Path(__file__).resolve().parent
DECISION_FILE = "continuation_decision.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validation_digest(report):
    # Task status paths may differ when evidence is copied between hosts.
    return original.digest(original.json_safe({k: v for k, v in report.items() if k != "tasks"}))


def controller_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def best_healthy_selection(report):
    if (report.get("status") != "needs_ours_target_development"
            or report["final_metric_gate"].get("status") != "unmet"
            or report["final_metric_gate"].get("qualified_ours_candidates")):
        raise ValueError("continuation requires unmet targets and a healthy Ours candidate")
    cid = report.get("best_healthy_score_candidate")
    row = report["final_metric_gate"]["candidate_rows"].get(cid, {})
    if (row.get("method") != "sm9rrs" or not row.get("eligible")
            or not row.get("health_qualified") or not row.get("scorable")
            or not isinstance(row.get("raw_score"), (int, float))
            or not math.isfinite(row["raw_score"])):
        raise ValueError("the best healthy Ours candidate is missing or invalid")
    selected = {**report["selected"], "sm9rrs": cid}
    if set(selected) != set(original.ALL_METHODS):
        raise ValueError("continuation requires all five fixed baseline selections")
    return selected, row["raw_score"]


def decision_basis(manifest, report):
    selected, score = best_healthy_selection(report)
    return {
        "schema_version": 1, "decision": "Y",
        "mode": "user_selected_after_unmet_validation",
        "ours_candidate": selected["sm9rrs"], "selection_raw_score": score,
        "selected": selected, "manifest_fingerprint": manifest["fingerprint"],
        "validation_report_digest": validation_digest(report),
        "source_validation_status": report["status"], "validation_target_passed": False,
        "selection_basis": "best_healthy_score_candidate",
    }


def check_decision(decision, manifest, report):
    expected = decision_basis(manifest, report)
    if ({k: decision.get(k) for k in expected} != expected
            or set(decision) != set(expected) | {"controller_source_sha256", "created_at_utc"}
            or not re.fullmatch(r"[0-9a-f]{64}", str(decision.get("controller_source_sha256", "")))
            or not isinstance(decision.get("created_at_utc"), str)):
        raise ValueError("saved continuation differs from the immutable validation/selection")
    return decision["selected"]


def ask_continuation(candidate, score, *, resuming=False, input_stream=None):
    stream = sys.stdin if input_stream is None else input_stream
    action = "继续已冻结的正式实验（保留已完成任务和检查点）" if resuming else "直接进入正式实验"
    message = ("所有Ours候选均未通过原双指标≤2个百分点要求。"
               f"最佳健康候选为 {candidate}（Score={score:.8f}）。\n"
               f"是否采用该候选及原定五个基线{action}？[Y/N] "
               "Y=继续；N=停止并保留结果。原验证仍记为未达标。")
    while True:
        # Newline-delimited event allows the progress wrapper to display and
        # relay terminal input without hiding a buffered input() prompt.
        print("CONTINUATION_PROMPT " + json.dumps({"candidate": candidate,
              "raw_score": score, "message": message}, ensure_ascii=False), flush=True)
        answer = stream.readline()
        if not answer:
            print("CONTINUATION_INPUT_EOF 未收到确认，停止并保留结果。", flush=True)
            return False
        answer = answer.strip().upper()
        if answer in {"Y", "N"}:
            return answer == "Y"
        print("CONTINUATION_INVALID_INPUT 请输入Y或N，不会自动选择。", flush=True)


def record_response(output, approved, manifest, report, *, resuming):
    now = datetime.now(timezone.utc)
    folder = output / "continuation_responses"
    folder.mkdir(exist_ok=True)
    original.write_json(folder / f"{now.strftime('%Y%m%dT%H%M%S_%f')}_{os.getpid()}.json", {
        "decision": "Y" if approved else "N", "resuming": resuming,
        "created_at_utc": now.isoformat(), "manifest_fingerprint": manifest["fingerprint"],
        "validation_report_digest": validation_digest(report),
        "ours_candidate": report["best_healthy_score_candidate"],
        "controller_source_sha256": controller_hash(),
    })


def check_data(spec, data_dir, manifest):
    split, contract = original.load_split(spec, data_dir)
    del split
    if contract != manifest["data_contract"]:
        raise ValueError("CIFAR data digests or split differ from the immutable study")


def write_final(output, spec, selected, report, tasks, results, statuses, decision):
    details = deepcopy(report["methods"])
    if decision is not None:
        details["sm9rrs"].update(selected_candidate=selected["sm9rrs"],
            selection_status="user_selected_after_unmet_validation", health_qualified=True,
            scorable=True, raw_score=decision["selection_raw_score"],
            selection_raw_score=decision["selection_raw_score"], pass_route=None)
    final = original.summarize_final(spec, selected, results, statuses, tasks, details)
    final["validation_final_metric_gate"] = {k: v for k, v in report["final_metric_gate"].items()
                                            if k not in ("candidate_rows", "ours_candidate_targets")}
    final["validation_ours_target"] = report["ours_target"]
    final["formal_results_used_for_selection"] = False
    if decision is not None:
        final["continuation_decision"] = decision
        final["methods"]["sm9rrs"].update(validation_target_passed=False,
            selected_without_target_qualification=True, validation_health_qualified=True)
        final["validation_selected_ours_target"] = report["ours_candidate_targets"][selected["sm9rrs"]]
    original.write_json(output / "final_summary.json", final)
    completed = [run for runs in results.values() for run in runs]
    folder = output / "final_results"
    folder.mkdir(exist_ok=True)
    if completed:
        original.experiments.write_result_files(folder, completed)
    aggregate = original.final_aggregate(tasks, results)
    with (folder / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    print("FINAL_STATUS " + final["status"], flush=True)
    print("FINAL_EXECUTION_SUMMARY " + json.dumps({
        "all_scheduled_tasks_attempted": final["all_scheduled_tasks_attempted"],
        "full_execution_completed": final["full_execution_completed"],
        "ours_independent_health_passed": final["ours_independent_health_passed"],
        "health_failure_tasks": len(final["health_failures"]),
        "report": str(output / "final_summary.json"),
    }), flush=True)


def run_parent(args):
    spec = original.load_spec(args.config)
    output = (args.output or REPO / spec["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with original.file_lock(output / ".runner.lock", nonblocking=True):
        path = output / "manifest.json"
        data_checked = False
        if path.exists():
            manifest = read_json(path)
            if manifest != original.build_manifest(spec, manifest["data_contract"], REPO):
                raise ValueError("immutable experiment identity changed; preserve this output")
        else:
            if any(p.name != ".runner.lock" for p in output.iterdir()):
                raise ValueError("refusing a nonempty output without its immutable manifest")
            split, contract = original.load_split(spec, args.data_dir)
            del split
            manifest = original.build_manifest(spec, contract, REPO)
            original.immutable_json(path, manifest)
            data_checked = True
        validation = original.attach_fingerprints(original.build_tasks(spec, "validation"), manifest)
        original.immutable_json(output / "validation_plan.json", {
            "manifest_fingerprint": manifest["fingerprint"], "tasks": validation})
        frozen = read_json(output / "final_plan.json") if (output / "final_plan.json").exists() else None
        decision = read_json(output / DECISION_FILE) if (output / DECISION_FILE).exists() else None
        if frozen is not None and frozen != original.final_plan(spec, manifest, frozen.get("selected", {})):
            raise ValueError("frozen final plan differs from its manifest or declared candidates")
        print(f"SIX_METHOD_OUTPUT {output}\nVALIDATION_RUNS {len(validation)} rounds=100 devices={','.join(args.devices)}", flush=True)
        print("VALIDATION_REUSE auditing saved validation before scheduling any missing work", flush=True)
        results, statuses = original.collect_results(output, validation)
        if (args.phase != "final" and frozen is None and decision is None
                and any(row["status"] != "complete" for row in statuses)):
            if not data_checked:
                check_data(spec, args.data_dir, manifest)
                data_checked = True
            original.execute_phase(args, validation, output)
            results, statuses = original.collect_results(output, validation)
        blockers = original.validation_blockers(output, validation, statuses)
        if blockers:
            # Do not overwrite an established scientific verdict on a damaged
            # resume; keep the operational problem in a separate audit file.
            original.write_json(output / "validation_resume_audit.json", {
                "status": "validation_evidence_incomplete", "blockers": blockers})
            print("VALIDATION_STATUS validation_evidence_incomplete", flush=True)
            print("FINAL_NOT_STARTED: validation evidence is incomplete or damaged; no user override is offered.", flush=True)
            return 2
        report = original.select_validation(spec, results, validation)
        report["tasks"] = statuses
        report = original.json_safe(report)
        del results
        if decision is not None:
            selected = check_decision(decision, manifest, report)
        elif report["status"] == "qualified_for_final":
            selected = report["selected"]
        elif frozen is not None:
            raise ValueError("unmet validation has a frozen final plan without its user decision")
        else:
            selected = None
        if frozen is not None and frozen["selected"] != selected:
            raise ValueError("validation selection differs from the frozen final plan")
        saved = output / "validation_summary.json"
        if not saved.exists() or read_json(saved) != report:
            original.write_json(saved, report)
        print("VALIDATION_STATUS " + report["status"], flush=True)
        print("VALIDATION_SELECTION " + json.dumps({m: {
            "candidate": info.get("selected_candidate"),
            "selection_status": info.get("selection_status"),
            "health_qualified": info.get("health_qualified"),
            "has_healthy_candidate": info.get("status") == "valid",
            "raw_score": info.get("selection_raw_score")}
            for m, info in report["methods"].items()}), flush=True)
        print("OURS_FINAL_METRIC_GATE " + json.dumps({
            k: report["final_metric_gate"].get(k) for k in ("status", "qualified_ours_candidates",
                "eligible_ours_candidates", "asr_target", "accuracy_target", "require_mean_dual_best")}), flush=True)
        if args.phase == "validation":
            return 0
        if report["status"] != "qualified_for_final":
            if report["status"] != "needs_ours_target_development":
                print("FINAL_NOT_STARTED: 所有Ours均未达标，且没有健康候选；不能进入正式实验。", flush=True)
                return 0
            proposed, score = best_healthy_selection(report)
            approved = ask_continuation(proposed["sm9rrs"], score, resuming=decision is not None)
            record_response(output, approved, manifest, report, resuming=decision is not None)
            if not approved:
                print("FINAL_NOT_STARTED: 用户未确认继续，已停止；结果与检查点保留。", flush=True)
                return 0
            if decision is None:
                decision = {**decision_basis(manifest, report),
                    "controller_source_sha256": controller_hash(),
                    "created_at_utc": datetime.now(timezone.utc).isoformat()}
                original.immutable_json(output / DECISION_FILE, decision)
            selected = check_decision(decision, manifest, report)
            print("CONTINUATION_CONFIRMED 原验证目标未通过；按已确认的最佳健康候选继续。", flush=True)
        planned = original.final_plan(spec, manifest, selected)
        original.immutable_json(output / "final_plan.json", planned)
        final_tasks = planned["tasks"]
        print(f"FINAL_RUNS {len(final_tasks)} parameters_frozen=true", flush=True)
        # Completed snapshots need no data reload. The original scheduler also
        # repairs artifacts if a process died after saving its result pickle.
        results, statuses = original.collect_results(output, final_tasks)
        if any(row["status"] != "complete" for row in statuses):
            if not data_checked:
                check_data(spec, args.data_dir, manifest)
        original.execute_phase(args, final_tasks, output)
        results, statuses = original.collect_results(output, final_tasks)
        write_final(output, spec, selected, report, final_tasks, results, statuses, decision)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=original.DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--phase", choices=("all", "validation", "final"), default="all")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.plan_only:
        return original.main(["--config", str(args.config), "--plan-only",
                              *(["--output", str(args.output)] if args.output else [])])
    if (not args.devices or len(set(args.devices)) != len(args.devices)
            or any(not re.fullmatch(r"cuda:\d+", d) for d in args.devices)):
        parser.error("use explicit CUDA ordinals, or the progress wrapper with --devices auto")
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
