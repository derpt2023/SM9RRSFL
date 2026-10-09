"""Six fresh full prefixes with a declared singleton-backward numerical policy."""
from copy import deepcopy
import hashlib
from pathlib import Path

import cifar_tail_determinism_protocol as tail
import cifar_tail_determinism_report as tail_report

prefix, prefix_report = tail.prefix, tail.prefix_report
base, runtime = tail.base, tail.runtime
REPO = Path(__file__).resolve().parent
PROTOCOL = "cifar-deterministic-prefix-v1"
DEFAULT_OUTPUT = REPO / "outputs/cifar_v8_diagnostic_v1/deterministic_prefix_v1"
DEFAULT_TAIL = tail.DEFAULT_OUTPUT
POLICIES = ("original", "singleton_backward_cudnn_deterministic")
NEW_SOURCES = ("cifar_deterministic_prefix_protocol.py", "cifar_deterministic_prefix_runtime.py",
               "cifar_deterministic_prefix_report.py", "run_cifar_deterministic_prefix.py")
SUPPORTED_CONCLUSION = "supports_suppression_of_observed_variability_in_this_scope"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def source_hashes():
    values = tail.source_hashes()
    values.update({name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in NEW_SOURCES})
    return values


def separate_outputs(output, tail_output, old_paths=()):
    output = Path(output).resolve()
    for other in [tail_output, *old_paths]:
        other = Path(other).resolve()
        if output == other or output in other.parents or other in output.parents:
            raise ValueError("deterministic prefix output must be separate and non-nested with every reference study")


def _supported(decision):
    clients = decision.get("clients", [])
    return ([item.get("client_id") for item in clients] == ["client-19", "client-84"]
        and all(item.get("conclusion") == SUPPORTED_CONCLUSION
            and item.get("original_three_pairs_reproduce_tail_gradient_first_difference") is True
            and item.get("deterministic_three_pairs_all_target_boundaries_equal") is True
            and item.get("context_review_reasons") == [] for item in clients))


def audit_reference(tail_output):
    """Revalidate archived tensors and policy observations, then recompute all pairs."""
    output = Path(tail_output).resolve()
    before = {"evidence": tail.evidence_hashes(output), "sources": tail.source_hashes()}
    review = tail_report.summarize(output)
    _require(review.get("status") == "complete" and review.get("expected_tasks") == 6
        and review.get("complete_tasks") == 6 and review.get("source_reference_and_evidence_verified") is True
        and _supported(review.get("decision", {})), "complete supported six-tail diagnosis required: "
        + str(review.get("error", review.get("status"))))
    manifest, tasks = tail.read_study(output, current_sources=True)
    _require([(t["policy"], t["repeat"]) for t in tasks] ==
        [(p, r) for r in (1, 2, 3) for p in tail.POLICIES], "six interleaved tail anchors are required")
    anchors, arrays_by_id, payloads, rows = [], {}, {}, []
    for task in tasks:
        artifact = tail.load_completed(output, task)
        _require(artifact is not None, "missing frozen tail artifact")
        _require(artifact.get("training_round_completed") is False, "tail reference unexpectedly completed a round")
        _require(tail_report.step._finite(artifact.get("wall_seconds")) and artifact["wall_seconds"] >= 0,
                 "invalid tail reference elapsed time")
        _require(tail_report.step.matched.normalized_environment(artifact["environment"]) ==
                 manifest["reference"]["execution_environment"], "tail reference numerical environment differs")
        arrays = tail.load_arrays(output, task, artifact)
        observations = tail_report.validate_observations(artifact["observations"], task, arrays)
        tail_report.validate_environment_policy(observations, artifact["environment"])
        history = tail_report.historical_context(observations, manifest["reference"])
        _require(history["prefix_context_reproduced"] and history["step_context_reproduced"]
            and all(c["historical_pre_backward_equal"] for c in history["clients"]),
            "tail reference historical context or pre-backward boundary differs")
        partitions = observations["prefix"]["partition"]["clients"]
        singleton_ids = [p["client_id"] for p in partitions if p["samples"] % task["config"]["batch_size"] == 1]
        _require(singleton_ids == ["client-19", "client-84"] and
            [(p["client_id"], p["samples"]) for p in partitions if p["client_id"] in singleton_ids] ==
            [("client-19", 701), ("client-84", 301)], "reviewed partition singleton client/sample coverage differs")
        anchors.append({"task": task, "observations": observations,
                        "artifact_fingerprint": artifact["artifact_fingerprint"]})
        payloads[task["task_id"]], arrays_by_id[task["task_id"]] = observations, arrays
        rows.append({"task_id": task["task_id"], "historical_context": history})
    lookup = {(t["policy"], t["repeat"]): t["task_id"] for t in tasks}
    specifications = [("within_policy", (p, a), (p, b)) for p in tail.POLICIES for a, b in tail_report.step.PAIRS]
    specifications += [("cross_policy", (tail.POLICIES[0], r), (tail.POLICIES[1], r)) for r in (1, 2, 3)]
    pairs = []
    for kind, earlier, later in specifications:
        a, b = lookup[later], lookup[earlier]
        pair = {"pair_id": b + "__" + a, "kind": kind, "earlier_policy": earlier[0],
            "earlier_repeat": earlier[1], "later_policy": later[0], "later_repeat": later[1],
            **tail_report.compare_observations(payloads[a], payloads[b], arrays_by_id[a], arrays_by_id[b])}
        _require(pair["available"] and not pair["context_mismatches"] and
            all(c["pre_backward_equal"] for c in pair["clients"]), "tail pair context or pre-backward inputs differ")
        pairs.append(pair)
    _require(len(pairs) == 9 and _supported(tail_report.decision(rows, pairs, True, ["client-19", "client-84"])),
             "actual archived tail arrays do not reproduce the supported policy result")
    tail.verify_reference(manifest["reference"])
    _require(before == {"evidence": tail.evidence_hashes(output), "sources": tail.source_hashes()},
             "tail evidence or sources changed during the reference audit")
    reference = deepcopy(manifest["reference"])
    reference.update(tail_output=str(output), tail_manifest=manifest,
        tail_evidence_sha256=before["evidence"], tail_anchors=anchors)
    return reference


def verify_reference(reference):
    output = Path(reference["tail_output"])
    _require(tail.evidence_hashes(output) == reference["tail_evidence_sha256"], "frozen six-tail evidence changed")
    _require(audit_reference(output) == reference, "frozen tail reference identity or observations changed")


def build_manifest(reference):
    old = reference["tail_manifest"]
    order = [{"policy": p, "repeat": r} for r in (1, 2, 3) for p in POLICIES]
    value = {"protocol": PROTOCOL, "reference": deepcopy(reference), "source_sha256": source_hashes(),
        "same_gpu_uuid": old["same_gpu_uuid"], "data_contract": deepcopy(old["data_contract"]),
        "spec": deepcopy(old["spec"]), "partition": "dirichlet", "rounds": 3, "num_clients": 100,
        "policies": list(POLICIES), "repeats": 3, "execution_order": order,
        "policy_scope": {"singleton_backward_cudnn_deterministic": {
            "selection": "actual single-sample minibatch in every client local-training call; all such batches are final in this frozen partition",
            "operation": "only original loss.backward calls", "cudnn_deterministic": True,
            "restore_after_each_backward": True, "other_numerical_flags_modified": False,
            "expected_events_per_task": 6,
            "expected_clients_from_frozen_partition": ["client-19", "client-84"], "rounds": [1, 2, 3]},
            "original": {"numerical_flags_modified": False}},
        "original_numerical_policy": False, "policy_change_explicit": True,
        "original_training_function": True, "fresh_process_per_task": True,
        "serial_same_physical_gpu": True, "aggregation_executed": True,
        "completed_training_rounds": 3, "checkpoints_reused": False,
        "health_assessed": False, "formal_qualification_assessed": False, "next_stage_automatic": False,
        "observation_limit": "three clean instrumented rounds test repeatability including aggregation; they do not establish full-training determinism or performance improvement"}
    return {**value, "fingerprint": base.digest(value)}


def build_tasks(manifest):
    reference = manifest["reference"]
    expected = tail.build_tasks(reference["tail_manifest"])
    _require([a["task"] for a in reference["tail_anchors"]] == expected, "original six-tail task identities differ")
    old = reference["tail_anchors"][0]["task"]
    config = old["config"]
    _require((config["rounds"], config["num_clients"], config["malicious_ratio"], config["partition"],
        config["seed"], config["local_epochs"], config["batch_size"]) ==
        (3, 100, 0., "dirichlet", 2026093001, 1, 50), "original full-prefix configuration differs")
    tasks = [{"task_id": "deterministic_prefix_" + p + "_repeat" + str(r), "policy": p, "repeat": r,
        "partition": "dirichlet", "config": deepcopy(config), "same_gpu_uuid": manifest["same_gpu_uuid"],
        "original_prefix_task_fingerprint": old["original_prefix_task_fingerprint"],
        "original_tail_task_fingerprint": old["fingerprint"],
        "purpose": "three_round_singleton_backward_repeatability"} for r in (1, 2, 3) for p in POLICIES]
    return base.attach_fingerprints(tasks, manifest)


def read_study(output, *, current_sources=True, verify_reference=True):
    output = Path(output).resolve()
    manifest = runtime.read_json(output / "manifest.json")
    _require(manifest.get("protocol") == PROTOCOL and
        base.digest({k: v for k, v in manifest.items() if k != "fingerprint"}) == manifest.get("fingerprint"),
        "deterministic prefix manifest identity mismatch")
    expected = build_manifest(manifest["reference"])
    if not current_sources:
        expected["source_sha256"] = manifest["source_sha256"]
        expected["fingerprint"] = base.digest({k: v for k, v in expected.items() if k != "fingerprint"})
    _require(expected == manifest, "deterministic prefix source/configuration identity changed")
    reference = manifest["reference"]
    separate_outputs(output, reference["tail_output"], [reference["step_output"], reference["prefix_output"],
        *reference["prefix_manifest"]["reference"]["paths"].values()])
    if verify_reference:
        globals()["verify_reference"](reference)
    tasks = build_tasks(manifest)
    _require(runtime.read_json(output / "task_plans/prefix.json") ==
        {"manifest_fingerprint": manifest["fingerprint"], "tasks": tasks}, "deterministic prefix task plan differs")
    return manifest, tasks


seal_artifact = prefix.seal_artifact
load_completed = prefix.load_completed


def evidence_hashes(output):
    return prefix_report.evidence_hashes(Path(output))
