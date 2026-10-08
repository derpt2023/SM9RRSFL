"""Frozen CNN E1 development panel separating K from attack onset.

No old controller, optimizer, defense, or report source is modified. The two
completed clean studies are audited read-only and frozen as external evidence.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import run_cifar_matched_cnn as matched
import cifar_matched_cnn_report as matched_report

clean, base, runtime = matched.clean, matched.base, matched.runtime
REPO = Path(__file__).resolve().parent
PROTOCOL = "cifar-cnn-timing-diagnostic-v1"
DEFAULT_CONFIG = REPO / "configs/cifar10_cnn_timing_v1.json"
DEFAULT_CLEAN = matched.DEFAULT_REFERENCE
DEFAULT_MATCHED = matched.DEFAULT_OUTPUT
DEFAULT_OUTPUT = REPO / "outputs/cifar_v8_diagnostic_v1/timing_cnn_e1"
CLEAN_FINGERPRINT = matched.REFERENCE_FINGERPRINT
MATCHED_FINGERPRINT = "2eff6875f5b571171f8fcb9a2b4b952ab6ba92fbfdb873687fd4f71264798c22"
OURS_014 = {
    "detector_subspace_dim": 2, "detector_normal_clusters": 2,
    "detector_distance_threshold": 1.25, "detector_reject_threshold": 6.,
    "detector_drift_memory": .8, "detector_drift_allowance": 1.25,
    "detector_drift_threshold": 6., "detector_history_confirm": 2,
    "detector_history_threshold": 1., "detector_recovery_confirm": 2,
    "detector_reference_budget": 3.5, "detector_clip_factor": 2.,
    "detector_weight_cap": 2., "suspicion_penalty_factor": .5,
    "suspicion_recovery_factor": 1.25, "suspicion_remove_after": 5}
ARMS = [{"id": "A", "detector_window": 10, "attack_start_round": 12},
        {"id": "B", "detector_window": 10, "attack_start_round": 25},
        {"id": "C", "detector_window": 20, "attack_start_round": 25}]
PUBLIC = {"model": "v7_cnn", "lr": .05, "local_epochs": 1, "lr_decay": .99,
          "rounds": 150, "num_clients": 100, "batch_size": 50, "dirichlet_alpha": .5}
ATTACK = {"attack": "alternating_minimization", "attack_boost": 5., "attack_epochs": 1,
          "attack_stealth_steps": 1, "attack_distance_weight": .0001,
          "attack_source_label": 5, "attack_target_label": 7, "attack_target_count": 200}
NEW_SOURCES = ("cifar_timing_protocol.py", "run_cifar_timing_diagnostic.py", "cifar_timing_report.py")


def validate_spec(spec):
    expected = {"schema_version": 1, "protocol": PROTOCOL,
        "clean_manifest_fingerprint": CLEAN_FINGERPRINT, "matched_manifest_fingerprint": MATCHED_FINGERPRINT,
        "chosen_setting": "C0", "public": PUBLIC, "attack": ATTACK, "ours_parameters": OURS_014,
        "seed": 2026093001, "partitions": ["iid", "dirichlet"],
        "ours_ratios": [0., .1, .7], "fedavg_ratios": [.1, .7], "arms": ARMS}
    if {k: v for k, v in spec.items() if k != "output_dir"} != expected or "output_dir" not in spec:
        raise ValueError("stage 2 fixes CNN E1, original Ours014, and the 26-task K/onset panel")
    return spec


def source_hashes():
    hashes = matched.source_hashes()
    for name in NEW_SOURCES:
        hashes[name] = hashlib.sha256((REPO / name).read_bytes()).hexdigest()
    return hashes


def matched_evidence_hashes(output, tasks):
    paths = [output / n for n in ("manifest.json", "task_plans/matched_cnn.json", "execution_environment.json")]
    for task in tasks:
        folder = output / "tasks" / task["task_id"]
        paths.extend(folder / n for n in ("task.json", "environment.json", "observations.json",
                                          base.experiments.COMPLETED_RESULTS_SNAPSHOT))
        paths.extend(sorted((folder / "attempts").glob("*.json")))
    return {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def audit_reference(clean_output, matched_output):
    clean_output, matched_output = Path(clean_output).resolve(), Path(matched_output).resolve()
    clean_manifest, clean_tasks = clean.read_study(clean_output, current_sources=True)
    match_manifest, match_tasks = matched.read_study(matched_output, current_sources=True)
    if (clean_manifest["fingerprint"] != CLEAN_FINGERPRINT or match_manifest["fingerprint"] != MATCHED_FINGERPRINT):
        raise ValueError("stage 2 requires the two reviewed clean study identities")
    before = matched_evidence_hashes(matched_output, match_tasks)
    review = matched_report.summarize(matched_output, clean_output)
    if (review.get("status") != "complete" or review.get("healthy_tasks") != 4
            or not review.get("reference_verified") or not review.get("source_matches_current")
            or not review.get("execution_environment_compatible")
            or review.get("decision", {}).get("action") != "prefer_cnn_for_next_development_review_learning_curves"):
        raise ValueError("paired clean evidence must remain complete/healthy and support the reviewed CNN choice")
    environment = match_manifest["reference"]["execution_environment"]
    for task in match_tasks:
        metadata = runtime.read_json(matched_output / "tasks" / task["task_id"] / "environment.json")
        if matched.normalized_environment(metadata) != environment:
            raise ValueError("matched CNN task numerical environment differs from its reference")
    if before != matched_evidence_hashes(matched_output, match_tasks):
        raise ValueError("matched reference changed while being audited")
    # summarize already recomputed and compared all original 24 source hashes
    # with this immutable reference (including per-task environment records).
    original = match_manifest["reference"]
    c0_tasks = [t for t in clean_tasks if t["candidate"]["candidate_id"] == "C0"]
    clean_rows = [r for r in review["reference_rows"] if r["setting"] == "C0"]
    # Explicit audit of the reviewed accuracy evidence; this is not a new gate.
    means = review["decision"]
    for key, value in (("mean_accuracy_C0", .6164), ("mean_accuracy_C3", .6167), ("mean_accuracy_R3", .6160)):
        if abs(means[key] - value) > 1e-12:
            raise ValueError("reference results differ from the reviewed model-choice evidence: " + key)
    return {"clean_manifest_fingerprint": clean_manifest["fingerprint"],
        "matched_manifest_fingerprint": match_manifest["fingerprint"],
        "clean_evidence_sha256": original["evidence_sha256"], "matched_evidence_sha256": before,
        "data_contract": clean_manifest["data_contract"], "dataset": clean_manifest["spec"]["dataset"],
        "execution_environment": environment, "c0_tasks": c0_tasks, "clean_rows": clean_rows,
        "matched_rows": review["rows"], "model_review": review["decision"],
        "chosen_model": "v7_cnn", "chosen_setting": "C0",
        "choice_reason": "CNN E1 retains nearly the same final accuracy as E2 with lower calibration CE and cost",
        "choice_is_development_decision_not_formal_qualification": True}


def validate_reference_identity(spec, reference):
    if (reference["clean_manifest_fingerprint"] != spec["clean_manifest_fingerprint"]
            or reference["matched_manifest_fingerprint"] != spec["matched_manifest_fingerprint"]
            or reference["chosen_model"] != "v7_cnn" or reference["chosen_setting"] != "C0"):
        raise ValueError("model choice/reference identity differs from the fixed stage-2 protocol")


def build_manifest(spec, reference):
    validate_spec(spec)
    validate_reference_identity(spec, reference)
    payload = {"protocol": PROTOCOL, "spec": {**spec, "dataset": reference["dataset"]},
        "reference": reference, "data_contract": reference["data_contract"],
        "source_sha256": source_hashes(), "purpose": "development_timing_panel",
        "evaluation_split": "calibration_dataset", "official_test_used_for_selection": False,
        "next_stage_automatic": False, "baseline_algorithms_modified": False,
        "ours_algorithm_modified": False, "loss_observer_enabled": False}
    return {**payload, "fingerprint": base.digest(payload)}


def build_tasks(manifest):
    spec = manifest["spec"]
    validate_spec({k: v for k, v in spec.items() if k != "dataset"})
    originals = manifest["reference"]["c0_tasks"]
    if len(originals) != 4 or {(t["config"]["partition"], t["config"]["seed"]) for t in originals} != {
            (p, s) for p in ("iid", "dirichlet") for s in (2026093001, 2026093002)}:
        raise ValueError("the four original C0 clean controls must be retained")
    for task in originals:
        raw = {k: v for k, v in task.items() if k != "fingerprint"}
        if base.attach_fingerprints([raw], {"fingerprint": spec["clean_manifest_fingerprint"]})[0] != task:
            raise ValueError("C0 reference task fingerprint mismatch")
        if (task["model"] != "v7_cnn" or task["candidate"]["candidate_id"] != "C0"
                or task["config"]["method"] != "fedavg" or task["config"]["malicious_ratio"] != 0
                or any(task["config"][k] != v for k, v in PUBLIC.items() if k != "model")
                or any(task["config"][k] != v for k, v in ATTACK.items())):
            raise ValueError("C0 reference no longer matches the chosen public training/attack template")
    tasks = []
    conditions = [("sm9rrs", a["id"], a["detector_window"], a["attack_start_round"], spec["ours_ratios"])
                  for a in ARMS]
    conditions += [("fedavg", "FA12", 10, 12, spec["fedavg_ratios"]),
                   ("fedavg", "FA25", 10, 25, spec["fedavg_ratios"])]
    for method, arm, window, onset, ratios in conditions:
        for partition in spec["partitions"]:
            reference = next(t for t in originals if t["config"]["partition"] == partition and t["config"]["seed"] == spec["seed"])
            for ratio in ratios:
                config = deepcopy(reference["config"])
                if method == "sm9rrs":
                    config.update(OURS_014)
                config.update(method=method, malicious_ratio=ratio, detector_window=window, attack_start_round=onset)
                base.fl.ExperimentConfig(**config).validate()
                parameters = {**(OURS_014 if method == "sm9rrs" else {}),
                              "detector_window": window, "attack_start_round": onset}
                tasks.append({"task_id": f"timing_{arm}_{method}_{partition}_ratio{ratio:g}_seed{spec['seed']}",
                    "phase": "validation", "purpose": "development_timing_panel", "arm": arm, "method": method,
                    "model": "v7_cnn", "candidate": {"candidate_id": "ours014-" + arm if method == "sm9rrs" else arm,
                        "variant": "original", "parameters": parameters}, "config": config,
                    "clean_reference_task_id": reference["task_id"], "clean_reference_fingerprint": reference["fingerprint"]})
    return base.attach_fingerprints(tasks, manifest)


def read_study(output, *, current_sources=False):
    output = Path(output)
    manifest = runtime.read_json(output / "manifest.json")
    if (manifest.get("protocol") != PROTOCOL
            or base.digest({k: v for k, v in manifest.items() if k != "fingerprint"}) != manifest["fingerprint"]):
        raise ValueError("timing manifest fingerprint/protocol mismatch")
    spec = {k: v for k, v in manifest["spec"].items() if k != "dataset"}
    validate_spec(spec)
    validate_reference_identity(spec, manifest["reference"])
    if current_sources and manifest != build_manifest(spec, manifest["reference"]):
        raise ValueError("timing source/reference/data identity changed")
    if (manifest["data_contract"] != manifest["reference"]["data_contract"]
            or manifest["spec"]["dataset"] != manifest["reference"]["dataset"]):
        raise ValueError("timing dataset identity mismatch")
    tasks = build_tasks(manifest)
    plan = runtime.read_json(output / "task_plans/timing.json")
    if plan != {"manifest_fingerprint": manifest["fingerprint"], "tasks": tasks}:
        raise ValueError("timing task plan mismatch")
    return manifest, tasks
