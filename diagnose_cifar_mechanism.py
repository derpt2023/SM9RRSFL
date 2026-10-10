#!/usr/bin/env python3
"""Read existing paired clean artifacts for round-26 scores and coefficients.

No experiment is started or resumed. Scientific sources and all input evidence
are checked before and after reading; redirect stdout to transport the result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import cifar_mechanism_protocol as protocol
import cifar_mechanism_report as report
import cifar_mechanism_details as details
import cifar_prefix_probe_protocol as old_protocol
import cifar_prefix_probe_report as old_report

EXPECTED_MANIFEST = "5704197af4da9705aa1690290780f54c6b6fd99da3bda9c32c70e9ffbdf39709"
SCHEMA = "cifar-clean-mechanism-details-v1"
REPO = Path(__file__).resolve().parent
READER_FILES = ("diagnose_cifar_mechanism.py", "cifar_mechanism_details.py")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def reader_hashes():
    return {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in READER_FILES}


def reference_evidence_hashes(reference):
    """Use each study's existing coverage, including old public NPZ evidence."""
    prefix = protocol.prefix
    values = {
        "deterministic_prefix": prefix.evidence_hashes(Path(reference["deterministic_prefix_output"])),
        "tail": prefix.tail.evidence_hashes(Path(reference["tail_output"])),
        "step": prefix.tail.step.evidence_hashes(Path(reference["step_output"])),
        "prefix": old_report.evidence_hashes(Path(reference["prefix_output"])),
    }
    values.update({"old_" + name: old_protocol.evidence_hashes(Path(path))
                   for name, path in reference["prefix_manifest"]["reference"]["paths"].items()})
    return values


def snapshot(output, reference):
    return {"readers": reader_hashes(), "science": protocol.source_hashes(),
            "mechanism": protocol.evidence_hashes(output),
            "references": reference_evidence_hashes(reference)}


def _reviewed_summary(summary, manifest, tasks):
    _require(summary.get("status") == "complete" and summary.get("manifest_fingerprint") == manifest["fingerprint"]
             and summary.get("source_reference_and_evidence_verified") is True
             and summary.get("complete_tasks") == summary.get("thirty_round_tasks") == 4
             and summary.get("available_pairs") == 4, "complete verified four-task thirty-round evidence required")
    _require(summary.get("source_count") == len(manifest["source_sha256"])
             and summary.get("source_map_sha256") == protocol.base.digest(manifest["source_sha256"])
             and summary.get("same_gpu_uuid") == manifest["same_gpu_uuid"],
             "summary source map or physical GPU differs from manifest")
    decision = summary.get("decision", {})
    _require(decision.get("conclusion") == "paired_clean_mechanism_evidence_ready_for_review"
             and all(decision.get(field) is True for field in (
                 "full_thirty_round_coverage", "within_arm_all_observations_equal",
                 "paired_preintervention_conditions_equal", "three_round_scoped_history_reproduced",
                 "round26_local_training_equal_before_first_affected_detection")),
             "repeatability or paired common conditions do not support this mechanism drilldown")
    _require([(t["arm"], t["repeat"]) for t in tasks] ==
             [(a, r) for r in (1, 2) for a in ("H0", "H1")], "four interleaved arm/repeat tasks required")
    rows = summary.get("rows", [])
    _require([r.get("task_id") for r in rows] == [t["task_id"] for t in tasks]
             and all(r.get("status") == "complete" and r.get("completed_training_rounds") == 30 for r in rows),
             "task coverage differs from verified summary")
    expected = {(kind, earlier, later, er, lr) for kind, earlier, later, er, lr in (
        ("within_arm", "H0", "H0", 1, 2), ("within_arm", "H1", "H1", 1, 2),
        ("cross_arm", "H0", "H1", 1, 1), ("cross_arm", "H0", "H1", 2, 2))}
    pairs = summary.get("pairs", [])
    _require(len(pairs) == 4 and {(p.get("kind"), p.get("earlier_arm"), p.get("later_arm"),
              p.get("earlier_repeat"), p.get("later_repeat")) for p in pairs} == expected,
             "paired comparison identities differ")
    for pair in pairs:
        _require(pair.get("available") is True and pair.get("full_thirty_round_coverage") is True,
                 "paired comparison incomplete")
        if pair["kind"] == "within_arm":
            _require(pair.get("equal") is True, "within-arm repeat observations differ")
        else:
            _require(pair.get("common_through_round24_and_round25_precommit_conditions_equal") is True
                     and pair.get("round26_local_training_equal_before_first_affected_detection") is True,
                     "cross-arm common conditions differ")


def diagnose(output):
    output = Path(output).resolve()
    original = protocol.runtime.read_json(output / "manifest.json")
    before = snapshot(output, original["reference"])
    manifest, tasks = protocol.read_study(output, current_sources=True)
    _require(manifest == original and manifest["fingerprint"] == EXPECTED_MANIFEST,
             "this reader requires the reviewed clean mechanism manifest")
    summary = report.summarize(output)
    _reviewed_summary(summary, manifest, tasks)
    payloads, task_rows = {}, {row["task_id"]: row for row in summary["rows"]}
    for task in tasks:
        artifact = protocol.load_completed(output, task)
        _require(artifact is not None and artifact.get("artifact_fingerprint") ==
                 task_rows[task["task_id"]]["artifact_fingerprint"], "artifact changed or disappeared after summary")
        payloads[(task["arm"], task["repeat"])] = report.validate_observations(artifact["observations"], task)
    paired = []
    for repeat in (1, 2):
        h0, h1 = [next(t for t in tasks if (t["arm"], t["repeat"]) == (arm, repeat)) for arm in ("H0", "H1")]
        _require(h0["config"] == h1["config"], "paired scientific configurations differ")
        analysis = details.analyze_pair(payloads[("H1", repeat)], payloads[("H0", repeat)], h0["config"])
        paired.append({"type": "paired_details", "repeat": repeat,
                       "h0_task_id": h0["task_id"], "h1_task_id": h1["task_id"], **analysis})
    # Compare content rather than task identities or wall times.
    repeated = {k: v for k, v in paired[0].items() if k not in ("repeat", "h0_task_id", "h1_task_id")} == {
        k: v for k, v in paired[1].items() if k not in ("repeat", "h0_task_id", "h1_task_id")}
    _require(repeated, "detailed paired results differ between repeated runs")
    protocol.verify_reference(manifest["reference"])
    _require(before == snapshot(output, manifest["reference"]), "evidence, reader or scientific sources changed during read")
    header = {"type": "header", "schema": SCHEMA, "status": "complete",
        "manifest_fingerprint": manifest["fingerprint"], "same_gpu_uuid": manifest["same_gpu_uuid"],
        "source_count": len(manifest["source_sha256"]), "source_map_sha256": protocol.base.digest(manifest["source_sha256"]),
        "reader_sha256": before["readers"], "evidence_map_sha256": protocol.base.digest({
            "mechanism": before["mechanism"], "references": before["references"]}),
        "complete_tasks": 4, "available_pairs": 4, "reference_studies_guarded": len(before["references"]),
        "source_reference_and_evidence_verified": True, "input_evidence_unchanged": True,
        "within_arm_all_observations_equal": True, "round26_local_training_equal": True,
        "detailed_pair_repeats_equal": repeated, "training_started": False, "experiment_files_written": False,
        "checkpoints_opened": False, "gpu_queried": False}
    legend = {"type": "legend", "direction": "H1 minus H0, paired at the same repeat number",
        "thresholds": "raw strict > for warning, drift and strong; equality does not trigger; margins are raw score minus threshold",
        "coefficients": "actual final coefficients; global normalization, reliability, cap and clipping can change coefficients even with unchanged client decisions",
        "history_frozen": "original weight manager guard, not the experimental H1 admission suppression",
        "quantiles": "linear interpolation over the stated matched public client observations",
        "scope": "r26 local inputs and updates match; r27..30 may have affected inputs and are reported separately; hashes do not measure tensor distance; repeats are the same seed; full detail is emitted once after exact repeat confirmation"}
    common_fields = {k: v for k, v in paired[0].items() if k not in ("repeat", "h0_task_id", "h1_task_id")}
    confirmation = {"type": "repeat_confirmation", "repeat": 2,
        "h0_task_id": paired[1]["h0_task_id"], "h1_task_id": paired[1]["h1_task_id"],
        "detailed_pair_equal_to_repeat1": True, "paired_detail_sha256": protocol.base.digest(common_fields),
        "artifact_fingerprints": {task["arm"]: task_rows[task["task_id"]]["artifact_fingerprint"]
                                  for task in tasks if task["repeat"] == 2}}
    paired[0]["paired_detail_sha256"] = confirmation["paired_detail_sha256"]
    decision = {"type": "decision", "action": "review_scores_and_coefficients_before_selecting_next_bounded_stage",
        "adopt_frozen_history": False, "automatic_next_stage": False, "formal_qualification_assessed": False,
        "full_protocol_health_assessed": False, "performance_improvement_assessed": False,
        "limitation": "one thirty-round clean Dirichlet configuration; no late-failure assessment, attacks, independent-seed effect estimate or operator identity"}
    records = [header, legend, paired[0], confirmation, decision]
    json.dumps(records, allow_nan=False)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=protocol.DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        records = diagnose(args.output)
        lines = [json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) for row in records]
        code = 0
    except Exception as exc:
        lines = [json.dumps({"type": "header", "schema": SCHEMA, "status": "unavailable_or_invalid_evidence",
            "error": str(exc)[:600], "training_started": False, "experiment_files_written": False,
            "checkpoints_opened": False, "gpu_queried": False}, ensure_ascii=False, allow_nan=False)]
        code = 2
    print("=== CIFAR_MECHANISM_DETAILS_BEGIN ===", flush=True)
    for line in lines:
        print(line)
    print("=== CIFAR_MECHANISM_DETAILS_END ===", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
