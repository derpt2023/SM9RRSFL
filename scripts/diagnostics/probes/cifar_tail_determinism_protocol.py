"""Frozen identities for an interleaved two-tail backward-policy diagnostic."""
from copy import deepcopy
import hashlib
from pathlib import Path

import cifar_client_step_probe_protocol as step
import cifar_client_step_probe_report as step_report

base, runtime = step.base, step.runtime
prefix, prefix_report = step.prefix, step.prefix_report
REPO = Path(__file__).resolve().parent
PROTOCOL = "cifar-tail-determinism-v1"
DEFAULT_OUTPUT = REPO / "outputs/cifar_v8_diagnostic_v1/tail_determinism_v1"
DEFAULT_STEP = step.DEFAULT_OUTPUT
TARGET_CLIENTS = [19, 84]
STOP_AFTER_CLIENT = 84
POLICIES = ("original", "tail_cudnn_deterministic")
NEW_SOURCES = ("cifar_tail_determinism_protocol.py", "cifar_tail_determinism_runtime.py",
               "cifar_tail_determinism_report.py", "run_cifar_tail_determinism.py")
PARAMETER_NAMES = ("conv1_w", "conv1_b", "conv2_w", "conv2_b", "dense1_w", "dense1_b",
                   "dense2_w", "dense2_b", "logits_w", "logits_b")
TAIL_SCOPE = [{"round": 1, "client_id": "client-19", "batch": 14, "samples": 1},
              {"round": 1, "client_id": "client-84", "batch": 6, "samples": 1}]


def source_hashes():
    values = step.source_hashes()
    values.update({name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in NEW_SOURCES})
    return values


def separate_outputs(output, step_output, old_paths=()):
    output = Path(output).resolve()
    for other in [step_output, *old_paths]:
        other = Path(other).resolve()
        if output == other or output in other.parents or other in output.parents:
            raise ValueError("tail diagnostic output must be separate and non-nested with every reference study")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _reviewed_pair(later, earlier, later_arrays, earlier_arrays):
    """Check the reviewed pattern using validated observations and actual NPZ arrays."""
    pair = step_report.compare_observations(later, earlier, later_arrays, earlier_arrays)
    first = pair.get("first_divergence") or {}
    _require({key: first.get(key) for key in ("stage", "client_id", "batch")} ==
             {"stage": "gradients", "client_id": "client-19", "batch": 14},
             "reference first divergence is not the reviewed client19 tail gradient")
    _require(pair.get("different_boundaries_by_stage") ==
             {"gradients": 2, "post_parameters": 2, "final_delta": 2},
             "reference differs outside the reviewed two-tail backward boundaries")
    for new, old, scope in zip(later["targets"], earlier["targets"], TAIL_SCOPE):
        _require(new["client_id"] == old["client_id"] == scope["client_id"], "reference target order differs")
        for index, (a, b) in enumerate(zip(new["batches"], old["batches"])):
            for stage in step_report.STAGES:
                allowed = index == scope["batch"] and stage in ("gradients", "post_parameters")
                _require(allowed or a[stage] == b[stage], "reference has a changed input, forward or non-tail boundary")
    for tail, scope in zip(pair["tail_comparisons"], TAIL_SCOPE):
        _require((tail["client_id"], tail["last_batch"], tail["samples"]) ==
                 (scope["client_id"], scope["batch"], 1), "reference tail scope differs")
        _require([p["name"] for p in tail["per_parameter"]] == list(PARAMETER_NAMES),
                 "reference does not use the reviewed ten-parameter CNN")
        for stage, metrics in tail["metrics"].items():
            changed = stage in ("gradients", "post_parameters", "final_delta")
            _require((metrics["max_abs"] > 0 and metrics["numerically_unequal_fraction"] > 0) if changed else
                     (metrics["max_abs"] == 0 and metrics["numerically_unequal_fraction"] == 0),
                     "reference retained arrays differ from the reviewed tail pattern")
        for parameter in tail["per_parameter"][2:]:
            _require(all(parameter[stage]["max_abs"] == 0 and
                         parameter[stage]["numerically_unequal_fraction"] == 0
                         for stage in step_report.PARAMETER_STAGES),
                     "reference differences extend beyond conv1 parameters")
    return pair


def audit_reference(step_output):
    output = Path(step_output).resolve()
    before = {"evidence": step.evidence_hashes(output), "sources": step.source_hashes()}
    review = step_report.summarize(output)
    _require(review.get("status") == "complete" and review.get("expected_tasks") == 3
             and review.get("complete_tasks") == 3 and review.get("all_historical_contexts_reproduced") is True
             and review.get("source_reference_and_evidence_verified") is True,
             "complete reviewed three-step diagnosis required: " + str(review.get("error", review.get("status"))))
    manifest, tasks = step.read_study(output, current_sources=True)
    _require([task["repeat"] for task in tasks] == [1, 2, 3] and len({t["task_id"] for t in tasks}) == 3,
             "three ordered original step anchors are required")
    anchors, arrays = [], []
    for task in tasks:
        artifact = step.load_completed(output, task)
        _require(artifact is not None, "missing original step artifact")
        _require(artifact.get("training_round_completed") is False, "original step unexpectedly completed a round")
        _require(step_report.matched.normalized_environment(artifact["environment"]) ==
                 manifest["reference"]["execution_environment"], "reference step numerical environment differs")
        _require(step_report._finite(artifact.get("wall_seconds")) and artifact["wall_seconds"] >= 0,
                 "invalid reference step elapsed time")
        values = step.load_arrays(output, task, artifact)
        observations = step_report.validate_observations(artifact["observations"], task, values)
        _require(step_report.historical_context(observations, manifest["reference"]["anchors"])
                 ["historical_context_reproduced"], "reference step historical context differs")
        _require([(t["client_id"], t["samples"], len(t["batches"]), t["batches"][-1]["samples"])
                  for t in observations["targets"]] == [("client-19", 701, 15, 1), ("client-84", 301, 7, 1)],
                 "reference target sample and minibatch coverage differs")
        anchors.append({"task": task, "observations": observations,
                        "artifact_fingerprint": artifact["artifact_fingerprint"]})
        arrays.append(values)
    for earlier, later in step_report.PAIRS:
        _reviewed_pair(anchors[later - 1]["observations"], anchors[earlier - 1]["observations"],
                       arrays[later - 1], arrays[earlier - 1])
    step.verify_reference(manifest["reference"])
    _require(before == {"evidence": step.evidence_hashes(output), "sources": step.source_hashes()},
             "step evidence or sources changed during the reference audit")
    reference = deepcopy(manifest["reference"])
    reference.update(step_output=str(output), step_manifest=manifest,
                     step_evidence_sha256=before["evidence"], step_anchors=anchors)
    return reference


def verify_reference(reference):
    output = Path(reference["step_output"])
    _require(step.evidence_hashes(output) == reference["step_evidence_sha256"], "frozen three-step evidence changed")
    _require(audit_reference(output) == reference, "frozen step reference identity or observations changed")


def build_manifest(reference):
    old = reference["step_manifest"]
    order = [{"policy": policy, "repeat": repeat} for repeat in (1, 2, 3) for policy in POLICIES]
    value = {"protocol": PROTOCOL, "reference": deepcopy(reference), "source_sha256": source_hashes(),
        "same_gpu_uuid": old["same_gpu_uuid"], "data_contract": deepcopy(old["data_contract"]),
        "spec": deepcopy(old["spec"]), "target_clients": list(TARGET_CLIENTS),
        "stop_after_client": STOP_AFTER_CLIENT, "stop_boundary": "before_round1_client85_local_training",
        "policies": list(POLICIES), "repeats": 3, "execution_order": order,
        "policy_scope": {"tail_cudnn_deterministic": {"boundaries": deepcopy(TAIL_SCOPE),
            "operation": "only original loss.backward calls", "cudnn_deterministic": True,
            "restore_after_each_backward": True, "other_numerical_flags_modified": False},
            "original": {"numerical_flags_modified": False}},
        "original_numerical_policy": False, "policy_change_explicit": True,
        "original_training_function": True, "fresh_process_per_task": True,
        "fresh_process_per_repeat": True, "serial_same_physical_gpu": True,
        "aggregation_executed": False, "training_round_completed": False, "checkpoints_reused": False,
        "health_assessed": False, "formal_qualification_assessed": False, "next_stage_automatic": False,
        "observation_limit": "host copies and scoped backend policy can alter execution; this is a localization experiment, not performance qualification"}
    return {**value, "fingerprint": base.digest(value)}


def build_tasks(manifest):
    old = manifest["reference"]["step_anchors"][0]["task"]
    expected_old = step.build_tasks(manifest["reference"]["step_manifest"])
    _require([anchor["task"] for anchor in manifest["reference"]["step_anchors"]] == expected_old,
             "original step task identities differ")
    config = old["config"]
    tasks = [{"task_id": "tail_" + policy + "_repeat" + str(repeat), "policy": policy,
        "repeat": repeat, "partition": "dirichlet", "config": deepcopy(config),
        "target_clients": list(TARGET_CLIENTS), "stop_after_client": STOP_AFTER_CLIENT,
        "same_gpu_uuid": manifest["same_gpu_uuid"],
        "original_prefix_task_fingerprint": old["original_prefix_task_fingerprint"],
        "original_step_task_fingerprint": old["fingerprint"],
        "purpose": "scoped_tail_backward_determinism_diagnostic"}
        for repeat in (1, 2, 3) for policy in POLICIES]
    return base.attach_fingerprints(tasks, manifest)


def read_study(output, *, current_sources=False, verify_reference=True):
    output = Path(output).resolve()
    manifest = runtime.read_json(output / "manifest.json")
    _require(manifest.get("protocol") == PROTOCOL and
             base.digest({k: v for k, v in manifest.items() if k != "fingerprint"}) == manifest.get("fingerprint"),
             "tail determinism manifest identity mismatch")
    expected = build_manifest(manifest["reference"])
    if not current_sources:
        expected["source_sha256"] = manifest["source_sha256"]
        expected["fingerprint"] = base.digest({k: v for k, v in expected.items() if k != "fingerprint"})
    _require(expected == manifest, "tail determinism source/configuration identity changed")
    reference = manifest["reference"]
    separate_outputs(output, reference["step_output"], [reference["prefix_output"],
        *reference["prefix_manifest"]["reference"]["paths"].values()])
    if verify_reference:
        globals()["verify_reference"](reference)
    tasks = build_tasks(manifest)
    _require(runtime.read_json(output / "task_plans/steps.json") ==
             {"manifest_fingerprint": manifest["fingerprint"], "tasks": tasks}, "tail determinism task plan differs")
    return manifest, tasks


# Keep the frozen path/SHA/zip guards shared with the old bounded probe.
seal_artifact = step.seal_artifact
load_completed = step.load_completed
load_arrays = step.load_arrays
evidence_hashes = step.evidence_hashes
