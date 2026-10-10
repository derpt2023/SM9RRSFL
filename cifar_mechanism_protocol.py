"""Independent, short clean history-mechanism panel after the reviewed prefix.

All four fresh tasks use the same declared numerical policy. Only H1 changes
history admission. Old experiments and their 86 scientific sources stay frozen.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import cifar_deterministic_prefix_protocol as prefix
import cifar_deterministic_prefix_report as deterministic_report

base, runtime = prefix.base, prefix.runtime
prefix_report = prefix.prefix_report
REPO = Path(__file__).resolve().parent
PROTOCOL = "cifar-clean-history-mechanism-v1"
DEFAULT_OUTPUT = REPO / "outputs/cifar_v8_diagnostic_v1/mechanism_clean_v1"
DEFAULT_PREFIX = prefix.DEFAULT_OUTPUT
POLICY = "singleton_backward_cudnn_deterministic"
ARMS = (("H0", "original", None), ("H1", "Ours-FrozenHistory-v1", 25))
ROUNDS = 30
NEW_SOURCES = ("cifar_mechanism_protocol.py", "cifar_mechanism_observer.py",
               "cifar_mechanism_runtime.py", "cifar_mechanism_report.py", "run_cifar_mechanism_panel.py")
SUPPORTED_CONCLUSION = "supports_observed_three_round_reproducibility_for_scoped_policy"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def source_hashes():
    values = prefix.source_hashes()
    values.update({name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in NEW_SOURCES})
    return values


def separate_outputs(output, prefix_output, old_paths=()):
    output = Path(output).resolve()
    for path in (prefix_output, *old_paths):
        other = Path(path).resolve()
        _require(output != other and output not in other.parents and other not in output.parents,
                 "mechanism output must be separate and non-nested with every reference study")


def reference_paths(reference):
    return [reference["deterministic_prefix_output"], reference["tail_output"], reference["step_output"],
            reference["prefix_output"], *reference["prefix_manifest"]["reference"]["paths"].values()]


def _supported(decision):
    return (decision.get("conclusion") == SUPPORTED_CONCLUSION
            and decision.get("original_pairs_different") == 3
            and decision.get("scoped_policy_pairs_equal") == 3
            and decision.get("scoped_policy_all_three_round_observations_equal") is True
            and decision.get("context_review_reasons") == []
            and decision.get("historical_deterministic_effect_requires_review") is False)


def audit_reference(prefix_output):
    """Recompute the nine comparisons from six sealed artifacts and all ancestors."""
    output = Path(prefix_output).resolve()
    before = {"evidence": prefix.evidence_hashes(output), "sources": prefix.source_hashes()}
    manifest, tasks = prefix.read_study(output, current_sources=True)
    _require([(t["policy"], t["repeat"]) for t in tasks] ==
             [(p, r) for r in (1, 2, 3) for p in prefix.POLICIES], "six interleaved prefix references required")
    anchors, payloads, rows = [], {}, []
    for task in tasks:
        artifact = prefix.load_completed(output, task)
        _require(artifact is not None and artifact.get("fresh_start") is True
                 and artifact.get("checkpoints_used") is False, "missing or non-fresh prefix reference")
        _require(type(artifact.get("completed_training_rounds")) is int
                 and artifact["completed_training_rounds"] == 3, "reference must complete three rounds")
        _require(prefix_report.timing.finite(artifact.get("wall_seconds")) and artifact["wall_seconds"] >= 0,
                 "invalid reference elapsed time")
        _require(prefix_report.matched.normalized_environment(artifact["environment"]) ==
                 manifest["reference"]["execution_environment"], "prefix reference numerical environment differs")
        payload = deterministic_report.validate_observations(artifact["observations"], task)
        deterministic_report.validate_environment_policy(payload, artifact["environment"])
        history = deterministic_report.historical_context(payload, manifest["reference"])
        _require(history["common_context_reproduced"] and history["round1_singleton_pre_reproduced"],
                 "prefix reference historical common/pre-backward context differs")
        if task["policy"] == POLICY:
            _require(history["deterministic_effect_reproduced"] is True,
                     "prefix deterministic effect no longer reproduces the old tail evidence")
        anchors.append({"task": task, "observations": payload,
                        "artifact_fingerprint": artifact["artifact_fingerprint"]})
        payloads[task["task_id"]] = payload
        rows.append({"task_id": task["task_id"], "historical_context": history})
    lookup = {(t["policy"], t["repeat"]): t["task_id"] for t in tasks}
    specs = [("within_policy", (p, a), (p, b)) for p in prefix.POLICIES
             for a, b in deterministic_report.PAIRS]
    specs += [("cross_policy", (prefix.POLICIES[0], r), (POLICY, r)) for r in (1, 2, 3)]
    pairs = []
    for kind, earlier, later in specs:
        a, b = lookup[later], lookup[earlier]
        pairs.append({"pair_id": b + "__" + a, "kind": kind,
            "earlier_policy": earlier[0], "later_policy": later[0],
            "earlier_repeat": earlier[1], "later_repeat": later[1],
            **deterministic_report.compare_observations(payloads[a], payloads[b])})
    decision = deterministic_report.decision(rows, pairs, True)
    _require(len(pairs) == 9 and _supported(decision),
             "actual archived prefix evidence does not support the reviewed three-round result")
    prefix.verify_reference(manifest["reference"])
    _require(before == {"evidence": prefix.evidence_hashes(output), "sources": prefix.source_hashes()},
             "prefix evidence or sources changed during reference audit")
    reference = deepcopy(manifest["reference"])
    reference.update(deterministic_prefix_output=str(output), deterministic_prefix_manifest=manifest,
        deterministic_prefix_evidence_sha256=before["evidence"], deterministic_prefix_anchors=anchors,
        deterministic_prefix_decision=decision)
    return reference


def verify_reference(reference):
    output = Path(reference["deterministic_prefix_output"])
    _require(prefix.evidence_hashes(output) == reference["deterministic_prefix_evidence_sha256"],
             "frozen deterministic prefix evidence changed")
    _require(audit_reference(output) == reference, "frozen prefix reference identity or observations changed")


def build_manifest(reference):
    old = reference["deterministic_prefix_manifest"]
    _require(_supported(reference["deterministic_prefix_decision"]), "supported prefix result required")
    value = {"protocol": PROTOCOL, "reference": deepcopy(reference), "source_sha256": source_hashes(),
        "same_gpu_uuid": old["same_gpu_uuid"], "spec": deepcopy(old["spec"]),
        "data_contract": deepcopy(old["data_contract"]), "rounds": ROUNDS,
        "partition": "dirichlet", "malicious_ratio": 0., "num_clients": 100,
        "execution_order": [{"arm": a, "repeat": r} for r in (1, 2) for a, _, _ in ARMS],
        "numerical_policy": POLICY, "numerical_scope": "actual singleton ordinary local-training backward only; no attack tasks",
        "history_intervention": {"H0": "original", "H1": "suppress history admission from round25 commit; retain evaluate, weighting and revocation"},
        "comparison_boundary": "rounds0..24 all observations; round25 through decisions/coefficients plus aggregate/model/evaluation must agree; commit state/admission may differ; round26 local training must still agree before its first affected detection; subsequent divergence is observed, not automatically proof of propagation",
        "fresh_process_per_task": True, "serial_same_physical_gpu": True, "checkpoints_reused": False,
        "maximum_training_rounds": 120, "maximum_client_training_calls": 12000,
        "diagnostic_health_only": True, "formal_qualification_assessed": False,
        "automatic_next_stage": False,
        "limitation": "two observed repeats per arm, 30 clean rounds on one GPU; late false revocation and attack defense are not assessed"}
    return {**value, "fingerprint": base.digest(value)}


def build_tasks(manifest):
    reference = manifest["reference"]
    expected = prefix.build_tasks(reference["deterministic_prefix_manifest"])
    _require([a["task"] for a in reference["deterministic_prefix_anchors"]] == expected,
             "original six-prefix task identities differ")
    anchor = next(t for t in expected if t["policy"] == POLICY and t["repeat"] == 1)
    config = deepcopy(anchor["config"])
    _require((config["rounds"], config["num_clients"], config["malicious_ratio"], config["partition"],
              config["seed"], config["local_epochs"], config["batch_size"], config["detector_window"],
              config["attack_start_round"]) == (3, 100, 0., "dirichlet", 2026093001, 1, 50, 20, 25),
             "original reviewed prefix configuration differs")
    config["rounds"] = ROUNDS
    tasks = [{"task_id": f"mechanism_{arm}_dirichlet_clean_repeat{repeat}",
        "arm": arm, "repeat": repeat, "partition": "dirichlet", "policy": POLICY,
        "candidate": {"candidate_id": arm, "variant": variant},
        "history_freeze_start_round": freeze, "config": deepcopy(config),
        "same_gpu_uuid": manifest["same_gpu_uuid"], "original_prefix_task_fingerprint": anchor["fingerprint"],
        "purpose": "short_clean_history_mechanism", "ours_algorithm_modified": arm == "H1"}
        for repeat in (1, 2) for arm, variant, freeze in ARMS]
    return base.attach_fingerprints(tasks, manifest)


def read_study(output, *, current_sources=True, verify_reference=True):
    output = Path(output).resolve()
    manifest = runtime.read_json(output / "manifest.json")
    _require(manifest.get("protocol") == PROTOCOL and
             base.digest({k: v for k, v in manifest.items() if k != "fingerprint"}) == manifest.get("fingerprint"),
             "mechanism manifest identity mismatch")
    expected = build_manifest(manifest["reference"])
    if not current_sources:
        expected["source_sha256"] = manifest["source_sha256"]
        expected["fingerprint"] = base.digest({k: v for k, v in expected.items() if k != "fingerprint"})
    _require(expected == manifest, "mechanism source/configuration identity changed")
    paths = reference_paths(manifest["reference"])
    separate_outputs(output, paths[0], paths[1:])
    if verify_reference:
        globals()["verify_reference"](manifest["reference"])
    tasks = build_tasks(manifest)
    _require(runtime.read_json(output / "task_plans/mechanism.json") ==
             {"manifest_fingerprint": manifest["fingerprint"], "tasks": tasks}, "mechanism task plan differs")
    return manifest, tasks


seal_artifact = prefix.seal_artifact
load_completed = prefix.load_completed


def evidence_hashes(output):
    output = Path(output)
    paths = {output / "manifest.json", output / "task_plans/mechanism.json"}
    for pattern in ("tasks/*/task.json", "tasks/*/completed.json"):
        paths.update(output.glob(pattern))
    return {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file() else None for path in sorted(paths)}
