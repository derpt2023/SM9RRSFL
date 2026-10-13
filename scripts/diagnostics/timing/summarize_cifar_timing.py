#!/usr/bin/env python3
"""Read-only compact JSONL view of the frozen 26-task timing report.

This adapter is deliberately outside the study's scientific source manifest.
It never trains, probes a GPU, downloads data, repairs files, or writes reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import cifar_timing_protocol as protocol
import cifar_timing_report as original

BEGIN = "=== CIFAR_TIMING_BRIEF_BEGIN ==="
END = "=== CIFAR_TIMING_BRIEF_END ==="

# Arrays replace repetitive labels only. Values are never rounded or clamped.
GROUP_FIELDS = ("original_clients", "remaining_client_rounds", "observed_verified_finite_updates",
                "unobserved_remaining_client_rounds", "aggregation_accepted", "history_admitted",
                "prior_revoked_client_rounds", "revoked_observed_events")
LEGEND = {
    "type": "legend", "schema": "cifar-timing-brief-v1",
    "task": {"p": "partition", "m": "method", "ratio": "malicious_ratio", "K": "detector_window",
        "start": "attack_start_round", "acc": "[accuracy50,accuracy100,accuracy150]",
        "asr": "attack_asr150; null for clean", "bg": "clean background 5->7; null for attacked",
        "clients": "[original_malicious,original_honest]",
        "rev": "[final_FP,final_TP,pre_attack_FP,pre_attack_designated_TP]",
        "nf": "nonfinite_updates", "last": "[last_round,last_accuracy,last_source_5_to_7]",
        "cost": "[worker_seconds,exact,unfinished_attempts,measured_runtime_seconds,CUDA_peak_allocated_MiB]",
        "clean": "[available,C0_accuracy150,accuracy_drop_pp,within_3pp]; diagnostic only"},
    "window": {"w1/w5/wa": "first scheduled round / first five / full attack period; clean wa is not applicable",
        "r": "[requested_start,requested_end,requested_n,observed_n,last_observed]",
        "complete": "complete_round_records", "weights": "[malicious_n,mean,max,honest_loss_n,mean,max]",
        "M/H": "malicious/honest " + json.dumps(GROUP_FIELDS, separators=(",", ":")),
        "gs": "[malicious_status,honest_status]", "cd": "client_diagnostics status",
        "new_rev": "[new_honest_revocations,new_designated_malicious_revocations]"},
    "semantics": [
        "accepted/observed and admitted/observed use verified finite online updates; denominator zero means not computable",
        "unobserved remaining clients are not assumed rejected; prior revoked clients are distinct",
        "pre_attack designated malicious identities had not yet attacked; their revocation is not detection of an active attack",
        "malicious mass is the raw sum of aggregation coefficients, not a renormalized share; small tolerated overflow retained",
        "FP uses original honest population; counts are cumulative, not summed over rounds",
        "A->B changes onset and exposure139->126; B->C changes K only; one development seed",
        "TASK values retain every pair for recomputation; no winner or new gate is generated; null never means zero"
    ]}
PAIR_FIELDS = ("accuracy", "error", "attack_target_success_rate", "attack_target_confidence", "false_positive_revocations",
               "true_positive_revocations", "malicious_weight_mass", "honest_weight_loss",
               "accepted_updates", "rejected_updates", "blacklisted_clients", "nonfinite_updates")


def short_error(value, limit=320):
    """Bound diagnostic text, never scientific measurements or health reasons."""
    encoded = str(value).encode("utf-8")
    if len(encoded) <= limit:
        return str(value)
    return encoded[:limit].decode("utf-8", errors="ignore") + " [truncated]"


def select(row, names):
    return {name: row[name] for name in names if name in row}


def audit_context(output):
    """Read identities and exact clean A/B evidence; do not infer a cause."""
    try:
        manifest, tasks = protocol.read_study(output, current_sources=False)
        reference = manifest["reference"]
        metadata = {"audit_manifest_fingerprint": manifest["fingerprint"],
            "clean_manifest_fingerprint": reference.get("clean_manifest_fingerprint"),
            "matched_manifest_fingerprint": reference.get("matched_manifest_fingerprint"),
            "recorded_source_map_sha256": protocol.base.digest(manifest["source_sha256"]),
            "recorded_source_count": len(manifest["source_sha256"])}
    except Exception as exc:
        return {"status": "unavailable", "error": short_error(exc), "audits": []}
    audits = []
    for partition in ("iid", "dirichlet"):
        audit = {"type": "clean_pair_audit", "partition": partition, "seed": original.DEV_SEED,
                 "compared_fields": list(PAIR_FIELDS)}
        try:
            pair = {}
            for arm in ("A", "B"):
                matches = [t for t in tasks if t["arm"] == arm and t["config"]["method"] == "sm9rrs"
                           and t["config"]["partition"] == partition and t["config"]["malicious_ratio"] == 0
                           and t["config"]["seed"] == original.DEV_SEED]
                if len(matches) != 1:
                    raise ValueError("expected one clean task for each A/B partition/seed")
                pair[arm] = matches[0]
            audit["task_ids"] = {arm: pair[arm]["task_id"] for arm in pair}
            a, b = pair["A"]["config"], pair["B"]["config"]
            changes = {key: {"A": a.get(key), "B": b.get(key)} for key in sorted(set(a) | set(b))
                       if key not in a or key not in b or a[key] != b[key]}
            audit["config_differences"] = changes
            expected_change = changes == {"attack_start_round": {"A": 12, "B": 25}}
            audit["only_expected_attack_start_difference"] = expected_change
            root_environment = protocol.runtime.read_json(output / "execution_environment.json")
            environments, records, devices = {}, {}, {}
            for arm, task in pair.items():
                folder = output / "tasks" / task["task_id"]
                value = protocol.runtime.read_json(folder / "environment.json")
                environments[arm] = protocol.matched.normalized_environment(value)
                devices[arm] = select(value, ("requested_device", "actual_compute_device"))
                result = protocol.base.checked_completed(output, task)
                if result is None or result.stopped_round != 150:
                    raise ValueError("complete 150-round clean snapshot unavailable: " + task["task_id"])
                if [r.round for r in result.records] != list(range(151)):
                    raise ValueError("clean snapshot round records are incomplete: " + task["task_id"])
                records[arm] = result.records
            env_checks = {"A_matches_root": environments["A"] == root_environment,
                "B_matches_root": environments["B"] == root_environment,
                "A_equals_B": environments["A"] == environments["B"],
                "root_matches_reference": root_environment == reference["execution_environment"]}
            audit["environment_checks"] = env_checks
            audit["recorded_devices"] = devices
            audit["round0"] = {arm: {field: getattr(rows[0], field) for field in PAIR_FIELDS}
                               for arm, rows in records.items()}
            differing = []
            for left, right in zip(records["A"], records["B"]):
                differences = {field: {"A": getattr(left, field), "B": getattr(right, field)}
                               for field in PAIR_FIELDS if getattr(left, field) != getattr(right, field)}
                if differences:
                    differing.append({"round": left.round, "fields": differences})
            audit.update(status="audited" if expected_change and all(env_checks.values()) else "evidence_mismatch",
                compared_rounds=151, differing_rounds=len(differing),
                first_differing_round=differing[0] if differing else None)
        except Exception as exc:
            audit.update(status="unavailable", error=short_error(exc))
        audits.append(audit)
    return {"status": "audited" if all(row["status"] == "audited" for row in audits) else "unavailable_or_mismatched",
            **metadata, "audits": audits}


def compact_window(window):
    if window is None:
        return None
    if "rounds_requested" in window:
        requested, observed = window["rounds_requested"], window.get("rounds_observed", [])
        rounds = [requested[0] if requested else None, requested[-1] if requested else None,
                  len(requested), len(observed), observed[-1] if observed else None]
    elif "expected_rounds" in window:
        rounds = [window.get(name) for name in
                  ("start_round", "end_round", "expected_rounds", "observed_rounds", "last_observed_round")]
    else:
        return select(window, ("status",))
    mass, loss = window.get("malicious_weight_mass", {}), window.get("honest_weight_loss", {})
    clients = window.get("client_diagnostics", {})
    output = {"r": rounds, "complete": window.get("complete_round_records"),
        "weights": [mass.get("n"), mass.get("mean"), mass.get("max"),
                    loss.get("n"), loss.get("mean"), loss.get("max")],
        "cd": clients.get("status")}
    for key, group in (("M", "malicious"), ("H", "honest")):
        item = clients.get(group)
        output[key] = [item.get(field) for field in GROUP_FIELDS] if item is not None else None
    output["gs"] = [clients.get(group, {}).get("status") if clients.get(group) is not None else None
                    for group in ("malicious", "honest")]
    if "new_honest_revocations" in window or "new_designated_malicious_revocations" in window:
        output["new_rev"] = [window.get("new_honest_revocations"), window.get("new_designated_malicious_revocations")]
    return output


def compact_task(row):
    output = {"type": "task", "id": row["task_id"], "arm": row.get("arm"),
        "m": row.get("method"), "p": row.get("partition"), "seed": row.get("seed"),
        "ratio": row.get("malicious_ratio"), "K": row.get("detector_window"),
        "start": row.get("attack_start_round"), "exposure": row.get("scheduled_attack_rounds"),
        "status": row.get("status"), "healthy": row.get("healthy"),
        "reasons": row.get("reasons"), "round": row.get("stopped_round", row.get("last_completed_round")),
        "acc": [row.get("accuracy50"), row.get("accuracy100"), row.get("accuracy150")],
        "asr": row.get("attack_asr150"), "bg": row.get("background_5_to_7"),
        "clients": [row.get("original_malicious_clients"), row.get("original_honest_clients")],
        "rev": [row.get("false_positive_revocations"), row.get("true_positive_revocations"),
                row.get("pre_attack_honest_revoked"), row.get("pre_attack_designated_malicious_revoked")],
        "nf": row.get("nonfinite_updates"),
        "cost": [row.get("worker_wall_seconds"), row.get("worker_cost_exact"), row.get("unfinished_attempts"),
                 row.get("measured_runtime_seconds"), row.get("cuda_peak_allocated_mib")]}
    if "last_observed_accuracy" in row or "last_observed_source_5_to_7" in row:
        output["last"] = [output["round"], row.get("last_observed_accuracy"), row.get("last_observed_source_5_to_7")]
    if "clean_utility_vs_C0" in row:
        value = row["clean_utility_vs_C0"]
        output["clean"] = [value.get(name) for name in ("available", "C0_accuracy150", "accuracy_drop_pp", "within_3pp")]
    for dest, source in (("w1", "first_round"), ("w5", "first_five_rounds")):
        if source in row.get("mechanism_windows", {}):
            output[dest] = compact_window(row["mechanism_windows"][source])
    if "full_attack_period" in row:
        output["wa"] = compact_window(row["full_attack_period"])
    for name in ("error", "exception"):
        if name in row:
            output[name] = short_error(row[name])
    return output


def compact_records(report, context=None):
    header = {"type": "header", "schema": "cifar-timing-brief-v1", **select(report, (
        "status", "output", "protocol", "manifest_fingerprint", "reference_verified", "source_matches_current",
        "execution_environment_compatible", "execution_hardware", "complete_tasks", "expected_tasks",
        "healthy_tasks", "snapshot_tasks", "base_health_rule_unchanged", "evaluation_split")),
        "task_lines": len(report.get("rows", [])), "reference_lines": len(report.get("reference_rows", [])),
        "training_started_by_summary": False}
    for name in ("error", "reference_error", "environment_error", "source_error", "brief_error"):
        if name in report:
            header[name] = short_error(report[name])
    if context is not None:
        header.update(select(context, ("audit_manifest_fingerprint", "clean_manifest_fingerprint", "matched_manifest_fingerprint",
                                      "recorded_source_map_sha256", "recorded_source_count")))
        header["clean_pair_audit_status"] = context["status"]
        if "error" in context:
            header["clean_pair_audit_error"] = context["error"]
    records = [header, LEGEND]
    for row in report.get("reference_rows", []):
        records.append({"type": "reference", **select(row, ("setting", "partition", "seed", "status", "healthy")),
            "acc": [row.get("accuracy50"), row.get("accuracy100"), row.get("accuracy150")],
            "bg": row.get("background_5_to_7"), "worker_seconds": row.get("worker_wall_seconds")})
    records.extend(compact_task(row) for row in report.get("rows", []))
    if context is not None:
        records.extend(context.get("audits", []))
    if "decision" in report:
        records.append({"type": "decision", **select(report["decision"], (
            "action", "next_stage_started", "formal_qualification_assessed", "selected_arm", "performance_gate_changed"))})
    return records


def render(report, context=None):
    # Serialize all lines before emitting BEGIN so a malformed value cannot
    # leave a seemingly valid but truncated report on stdout.
    lines = [json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
             for row in compact_records(report, context)]
    return "\n".join([BEGIN, *lines, END]) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=protocol.DEFAULT_OUTPUT)
    parser.add_argument("--clean-output", type=Path, default=protocol.DEFAULT_CLEAN)
    parser.add_argument("--matched-output", type=Path, default=protocol.DEFAULT_MATCHED)
    args = parser.parse_args(argv)
    context = None
    try:
        report = original.summarize(args.output.resolve(), args.clean_output.resolve(), args.matched_output.resolve())
        context = audit_context(args.output.resolve())
        if (report.get("manifest_fingerprint") and context.get("audit_manifest_fingerprint")
                and report["manifest_fingerprint"] != context["audit_manifest_fingerprint"]):
            report = {**report, "status": "invalid_comparison_evidence",
                      "brief_error": "timing manifest changed between summary and clean-pair audit; evidence not combined"}
            context = {"status": "manifest_changed_during_read",
                       "audit_manifest_fingerprint": context["audit_manifest_fingerprint"],
                       "error": "second-read reference/source metadata omitted because the manifest changed",
                       "audits": []}
        text = render(report, context)
    except Exception as exc:
        report = {"status": "brief_unavailable_or_invalid", "output": str(args.output.resolve()),
                  "error": f"{type(exc).__name__}: {exc}"}
        text = render(report)
    print(text, end="", flush=True)
    valid = all(report.get(key) is True for key in
                ("reference_verified", "source_matches_current", "execution_environment_compatible"))
    return 0 if report.get("status") == "complete" and valid and context and context.get("status") == "audited" else 2


if __name__ == "__main__":
    raise SystemExit(main())
