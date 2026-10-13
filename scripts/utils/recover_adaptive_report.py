#!/usr/bin/env python3
"""Audit/rebuild an existing schema-8 formal report without resuming training.

Scientific modules remain frozen. This separate entry verifies their hashes,
the original selection evidence and all 60 completed snapshots before applying
the report-only aggregation-weight tolerance already used by health scoring.
Default mode is read-only; --write publishes derived reports with backups.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import uuid

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import run_cifar_six_from_scratch as base
from adaptive_report_validation import (AGGREGATION_WEIGHT_TOLERANCE,
                                        aggregation_weight_tolerance)

PROTOCOLS = {
    "cifar-resnet18-gn-tpe-v1": ("run_cifar_adaptive", "cifar_adaptive_reporting"),
    "fashion-mnist-resnet18-gn-tpe-v1": ("run_fashion_adaptive", "fashion_adaptive_reporting"),
}
METADATA = ("manifest.json", "search_state.json", "continuation_decision.json",
            "final_plan.json", "task_plans/formal.json", "validation_summary.json",
            "search_summary.json", "best_parameters.json")


def _sha(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


class Inputs:
    """Record immutable inputs, including absent optional files, before reading."""
    def __init__(self, output):
        self.output = output
        self.observed = {}

    def track(self, relative):
        path = self.output / relative
        if path.is_symlink():
            raise ValueError(f"input symlink requires inspection: {relative}")
        value = _sha(path) if path.is_file() else None
        if relative in self.observed and self.observed[relative] != value:
            raise ValueError(f"input changed during recovery: {relative}")
        self.observed[relative] = value
        return path

    def read(self, relative):
        return json.loads(self.track(relative).read_text(encoding="utf-8"))

    def verify(self):
        for relative in list(self.observed):
            self.track(relative)

    def hashes(self):
        return {name: checksum for name, checksum in self.observed.items() if checksum is not None}


@contextmanager
def _locked(output):
    # Existing studies already have this lock. Do not create even a lock file
    # during the default, entirely read-only audit.
    path = output / ".runner.lock"
    if path.is_symlink() or not path.is_file():
        raise ValueError("existing .runner.lock is required; verify the original output directory")
    with path.open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("study is in use; wait for the original controller to exit") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _collector(output, runtime, inputs):
    def collect(where, tasks):
        if Path(where) != output:
            raise ValueError("unexpected evidence directory")
        groups, statuses = {}, []
        for task in tasks:
            tid = task["task_id"]
            if Path(tid).name != tid or tid in (".", ".."):
                raise ValueError("invalid task path")
            folder = output / "tasks" / tid
            if folder.is_symlink():
                raise ValueError(f"task symlink requires inspection: {tid}")
            for name in ("task.json", base.experiments.COMPLETED_RESULTS_SNAPSHOT,
                         "metrics.json", "rounds.csv", "summary.json", "summary.csv", "failure.json"):
                inputs.track(f"tasks/{tid}/{name}")
            try:
                result = base.checked_completed(output, task)
                row = {"task_id": tid, "method": task["method"],
                       "candidate_id": task["candidate"]["candidate_id"]}
                if result is not None:
                    groups.setdefault(row["candidate_id"], []).append(result)
                    row.update(status="complete", healthy=base.metrics(result)["healthy"])
                else:
                    failure = runtime.terminal_failure(output, task)
                    row.update(status="failed" if failure else "pending", failure=failure)
            except Exception as exc:
                raise ValueError(f"invalid evidence for {tid}: {exc}") from exc
            statuses.append(row)
        return groups, statuses
    return collect


def _expected_search_summary(state, choice):
    """Reconstruct the frozen controller's descriptive search summary."""
    return {
        "status": state["status"], "elapsed_search_seconds": state["elapsed_search_seconds"],
        "completed_public_blocks": sum(b["status"] == "complete" for b in state["blocks"]),
        "actual_search_budget": [{
            "block_id": b["id"], "status": b["status"],
            "fully_attempted_waves": sum(w["status"] == "complete" for w in b["waves"]),
            "public_strategy": b["proposal"]["strategy"],
            "defense_tpe_proposals": {m: sum(t["strategy"] == "tpe" for t in s["trials"])
                                      for m, s in b["method_samplers"].items()},
            "budget_prefix_selected_for_comparison": "budget_selection_snapshot" in b,
        } for b in state["blocks"]],
        "public_tpe_proposals": sum(t["strategy"] == "tpe" for t in state["public_sampler"]["trials"]),
        "cross_public_condition_search_budgets_may_differ": True,
        "choice": choice, "formal_results_used_for_selection": False,
        "search_budget_is_not_a_global_optimality_guarantee": True,
    }


def _evidence(output, inputs):
    manifest = inputs.read("manifest.json")
    spec = manifest["spec"]
    if manifest.get("schema_version") != 8 or spec.get("protocol") not in PROTOCOLS:
        raise ValueError("only the independent CIFAR/Fashion schema-8 protocols are supported")
    controller_name, reporter_name = PROTOCOLS[spec["protocol"]]
    runner = importlib.import_module(controller_name)
    reporting = importlib.import_module(reporter_name)
    runtime = runner.runtime
    if manifest != runtime.build_manifest(spec, manifest["data_contract"]):
        raise ValueError("manifest/config/model/scientific source identity changed; refusing recovery")
    for name in METADATA:
        inputs.track(name)
    state = inputs.read("search_state.json")
    if (state.get("manifest_fingerprint") != manifest["fingerprint"] or
            state.get("status") not in ("budget_exhausted", "trial_limit_reached")):
        raise ValueError("search evidence is not a matching terminal search state")
    runner.AdaptiveTPESampler.from_state(state["public_sampler"])
    for block in state["blocks"]:
        for sampler in block["method_samplers"].values():
            runner.AdaptiveTPESampler.from_state(sampler)

    collect = _collector(output, runtime, inputs)
    original_collect = runtime.collect
    try:
        # The controller's audit is read-only once its collector is replaced:
        # unlike runtime.collect, this collector NEVER calls repair_completed.
        runtime.collect = collect
        audited_state = deepcopy(state)
        runner.prepare_selection(output, spec, manifest, audited_state)
    finally:
        runtime.collect = original_collect
    if audited_state != state:
        raise ValueError("selection has not been frozen; report recovery cannot change search state")
    choice = runner.choose_block(state)
    if choice is None:
        raise ValueError("no original healthy selection exists")
    if ((output / "search_summary.json").exists() and
            inputs.read("search_summary.json") != _expected_search_summary(state, choice)):
        raise ValueError("search_summary.json differs from the verified search state/selection")
    basis = {"manifest_fingerprint": manifest["fingerprint"], "choice": choice,
             "validation_target_passed": choice["target_qualified"],
             "source_search_digest": base.digest(state)}
    decision = inputs.read("continuation_decision.json")
    if decision != basis:
        raise ValueError("frozen selection no longer matches its validation evidence")
    if not choice["target_qualified"]:
        approvals = []
        for path in sorted((output / "continuation_responses").glob("*.json")):
            record = inputs.read(str(path.relative_to(output)))
            if (record.get("approved") is True and record.get("response") == "Y" and
                    record.get("manifest_fingerprint") == manifest["fingerprint"] and
                    record.get("choice") == choice):
                approvals.append(str(path.relative_to(output)))
        if not approvals:
            raise ValueError("no matching historical Y approval for the frozen formal selection")
    else:
        approvals = []
    block = runner.selection_view(next(b for b in state["blocks"] if b["id"] == choice["block_id"]))
    current = runner.block_spec(spec, block)
    tasks = runtime.attach_tasks(current, "final", manifest, choice["selected"])
    if len(tasks) != 60:
        raise ValueError("expected six methods x ten scenarios x one formal seed = 60 tasks")
    expected_plan = {"manifest_fingerprint": manifest["fingerprint"],
                     "selected": choice["selected"],
                     "public_parameters": block["proposal"]["parameters"], "tasks": tasks}
    if inputs.read("final_plan.json") != expected_plan:
        raise ValueError("final plan differs from the original frozen selection")
    if inputs.read("task_plans/formal.json") != {
            "manifest_fingerprint": manifest["fingerprint"], "tasks": tasks}:
        raise ValueError("formal worker plan differs from the original frozen selection")
    results, statuses = collect(output, tasks)
    missing = [row["task_id"] for row in statuses if row["status"] != "complete"]
    if missing:
        raise ValueError(f"report-only recovery requires all 60 completed snapshots; missing: {missing}")
    reporting._check_inputs(current, choice["selected"], tasks, statuses, results)
    tasks_by_config = {base.digest(base.semantic_config(t["config"])): t for t in tasks}
    with aggregation_weight_tolerance(reporting) as observations:
        for runs in results.values():
            for run in runs:
                tid = tasks_by_config[base.digest(base.semantic_config(run.config))]["task_id"]
                try:
                    complete, reason, _ = reporting._inspect_result(run, current["shared_parameters"]["rounds"])
                    if not complete:
                        raise ValueError(reason)
                except Exception as exc:
                    raise ValueError(f"invalid formal result {tid}: {exc}") from exc
    roundoff = []
    for row in observations:
        task = tasks_by_config[row["semantic_config_sha256"]]
        roundoff.append({**{k: v for k, v in row.items() if k != "config"},
                         "task_id": task["task_id"], "candidate_id": task["candidate"]["candidate_id"]})
    audit = {"status": "ready", "protocol": spec["protocol"], "output": str(output),
             "manifest_fingerprint": manifest["fingerprint"],
             "expected_tasks": len(tasks), "completed_tasks": len(tasks),
             "healthy_tasks": sum(s["healthy"] for s in statuses),
             "training_started": False, "parameters_reselected": False,
             "validation_target_passed": choice["target_qualified"], "selected": choice["selected"],
             "historical_approval_files": approvals, "new_training_approval_granted": False,
             "aggregation_weight_upper_tolerance": AGGREGATION_WEIGHT_TOLERANCE,
             "roundoff_accepted": roundoff, "raw_values_preserved": True,
             "accuracy_asr_health_and_promotion_rules_unchanged": True,
             "source_sha256": inputs.hashes(),
             "recovery_source_sha256": {name: _sha(Path(__file__).with_name(name)) for name in
                                        ("recover_adaptive_report.py", "adaptive_report_validation.py")}}
    inputs.verify()
    return audit, runtime, manifest, reporting, current, choice, tasks, results, statuses, block, decision


def _backup_path(output):
    return output / "report_recovery_backups" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8])


def _publish(output, stage, backup):
    names = ("final_results", "final_summary.json")
    moved, published = [], []
    try:
        for name in names:
            old = output / name
            if old.exists() or old.is_symlink():
                backup.mkdir(parents=True, exist_ok=True)
                old.rename(backup / name)
                moved.append(name)
        for name in names:
            (stage / name).rename(output / name)
            published.append(name)
    except BaseException:
        for name in reversed(published):
            (output / name).rename(stage / name)
        for name in reversed(moved):
            (backup / name).rename(output / name)
        raise
    return str(backup) if moved else None


def recover(output: Path, *, write=False) -> dict:
    output = Path(output).resolve(strict=True)
    with _locked(output):
        inputs = Inputs(output)
        (audit, runtime, manifest, reporting, spec, choice, tasks, results,
         statuses, block, decision) = _evidence(output, inputs)
        if not write:
            return audit
        with tempfile.TemporaryDirectory(prefix=".adaptive-report-recovery-", dir=output) as temporary:
            stage = Path(temporary)
            (stage / "tasks").symlink_to(output / "tasks", target_is_directory=True)
            for name in METADATA:
                source = output / name
                if source.is_file():
                    target = stage / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(source)
            with aggregation_weight_tolerance(reporting):
                summary = reporting.write_final_report(
                    stage, spec, choice["selected"], results, tasks, statuses, block["report"], decision)
            audit["status"] = "report_recovered"
            audit["created_utc"] = datetime.now(timezone.utc).isoformat()
            backup = _backup_path(output)
            audit["backup"] = str(backup) if any(
                (output / name).exists() or (output / name).is_symlink()
                for name in ("final_results", "final_summary.json")) else None
            audit["report"] = str(output / "final_results" / "visualizations.html")
            policy = {key: audit[key] for key in (
                "aggregation_weight_upper_tolerance", "raw_values_preserved",
                "training_started", "parameters_reselected", "recovery_source_sha256")}
            policy["accepted_roundoff_observations"] = len(audit["roundoff_accepted"])
            summary["report_recovery"] = policy
            base.write_json(stage / "final_summary.json", summary)
            destination = stage / "final_results"
            data_audit = json.loads((destination / "data_audit.json").read_text(encoding="utf-8"))
            data_audit["source_sha256"].update(inputs.hashes())
            data_audit["report_source_sha256"].update(audit["recovery_source_sha256"])
            data_audit["report_recovery"] = policy
            base.write_json(destination / "data_audit.json", data_audit)
            base.write_json(destination / "report_recovery.json", audit)
            html_path = destination / "visualizations.html"
            html = html_path.read_text(encoding="utf-8")
            notice = ('<p class="notice">报告恢复：仅聚合权重诊断采用训练既有的1e-9上界容差；'
                      f'共接受{len(audit["roundoff_accepted"])}项微小越界，原始数值保留。'
                      'Acc、ASR、健康和准入规则未改变，没有训练或重新选参。'
                      '<a href="report_recovery.json">恢复审计</a></p>')
            html_path.write_text(html.replace("</h1>", "</h1>" + notice, 1), encoding="utf-8")
            inputs.verify()
            if runtime.build_manifest(manifest["spec"], manifest["data_contract"]) != manifest:
                raise ValueError("scientific sources changed while generating report")
            _publish(output, stage, backup)
        return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="original schema-8 study directory")
    parser.add_argument("--write", action="store_true", help="publish derived reports; back up previous reports")
    args = parser.parse_args(argv)
    try:
        audit = recover(args.output, write=args.write)
    except Exception as exc:
        print(f"REPORT_RECOVERY_REFUSED {type(exc).__name__}: {exc}", flush=True)
        return 2
    compact = {key: value for key, value in audit.items() if key not in
               ("roundoff_accepted", "source_sha256", "recovery_source_sha256")}
    compact["accepted_roundoff_observations"] = len(audit["roundoff_accepted"])
    compact["roundoff_examples"] = audit["roundoff_accepted"][:5]
    print("REPORT_RECOVERY " + json.dumps(compact, ensure_ascii=False), flush=True)
    if not args.write:
        print("AUDIT_ONLY no files written; rerun with --write to generate the report.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
