"""Two-arm history-freeze ablation with original P0 configuration and fresh runs.

Only H1 changes the history-update policy, from the publicly fixed round 25.
All 78 preceding tasks remain immutable, read-only development evidence.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import cifar_cnn_threshold_protocol as threshold
import cifar_cnn_threshold_report as threshold_report

base, runtime = threshold.base, threshold.runtime
REPO = Path(__file__).resolve().parent
PROTOCOL = "cifar-cnn-history-ablation-v1"
DEFAULT_CONFIG = REPO / "configs/cifar10_cnn_history_panel_v1.json"
DEFAULT_THRESHOLD = threshold.DEFAULT_OUTPUT
DEFAULT_TIMING, DEFAULT_CLEAN, DEFAULT_MATCHED = threshold.DEFAULT_TIMING, threshold.DEFAULT_CLEAN, threshold.DEFAULT_MATCHED
DEFAULT_OUTPUT = REPO / "outputs/cifar_v8_diagnostic_v1/history_cnn_e1_k20"
THRESHOLD_FINGERPRINT = "53b480049d8f18a81fb45f518dfbbe8a88cded441cf66f9256e8dda2f522e0ff"
ARMS = [{"id": "H0", "variant": "original", "history_freeze_start_round": None},
        {"id": "H1", "variant": "Ours-FrozenHistory-v1", "history_freeze_start_round": 25}]
NEW_SOURCES = ("cifar_cnn_history_protocol.py", "run_cifar_cnn_history_panel.py",
               "cifar_cnn_history_report.py", "cifar_cnn_history_runtime.py")
REVIEWED_P0 = {
    ("iid", 0.): (.6332, None, 0), ("iid", .1): (.6256, .095, 0),
    ("iid", .7): (.5952, .155, 2), ("dirichlet", 0.): (.6056, None, 3),
    ("dirichlet", .1): (.6056, .08, 4), ("dirichlet", .7): (.4768, .505, 9),
}


def validate_spec(spec):
    expected = {"schema_version": 1, "protocol": PROTOCOL,
        "threshold_manifest_fingerprint": THRESHOLD_FINGERPRINT, "chosen_candidate": "P0",
        "public": threshold.timing.PUBLIC, "attack": threshold.timing.ATTACK,
        "ours_parameters": threshold.timing.OURS_014, "detector_window": 20, "attack_start_round": 25,
        "seed": 2026093001, "partitions": ["iid", "dirichlet"], "malicious_ratios": [0., .1, .7],
        "arms": ARMS, "both_arms_fresh": True, "freeze_applies_to_clean": True,
        "automatic_tpe": False, "next_stage_automatic": False}
    if {k: v for k, v in spec.items() if k != "output_dir"} != expected or "output_dir" not in spec:
        raise ValueError("history panel fixes original H0 and round-25 frozen-history H1, with 12 fresh P0-condition tasks")
    return spec


def source_hashes():
    values = threshold.source_hashes()
    for name in NEW_SOURCES:
        values[name] = hashlib.sha256((REPO / name).read_bytes()).hexdigest()
    return values


def threshold_evidence_hashes(output, tasks):
    output = Path(output)
    paths = [output / name for name in ("manifest.json", "task_plans/threshold.json", "execution_environment.json")]
    for task in tasks:
        folder = output / "tasks" / task["task_id"]
        paths.extend(folder / name for name in ("task.json", "environment.json", base.experiments.COMPLETED_RESULTS_SNAPSHOT))
        paths.extend(sorted((folder / "attempts").glob("*.json")))
    return {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def audit_reference(threshold_output, timing_output, clean_output, matched_output):
    """Read raw 24+4+26+24 evidence, retaining completed unhealthy P2 results."""
    output = Path(threshold_output).resolve()
    manifest, tasks = threshold.read_study(output, current_sources=True)
    if manifest["fingerprint"] != THRESHOLD_FINGERPRINT:
        raise ValueError("history panel requires the reviewed 24-task threshold study")
    before = threshold_evidence_hashes(output, tasks)
    report = threshold_report.summarize(output, Path(timing_output), Path(clean_output), Path(matched_output))
    if (report.get("status") != "complete" or report.get("complete_tasks") != 24
            or any(report.get(key) is not True for key in
                   ("reference_verified", "source_matches_current", "execution_environment_compatible"))):
        raise ValueError("all 24 threshold snapshots and their 54-task source/reference/environment chain must be verifiable")
    environment = manifest["reference"]["execution_environment"]
    for task in tasks:
        metadata = runtime.read_json(output / "tasks" / task["task_id"] / "environment.json")
        if threshold.timing.matched.normalized_environment(metadata) != environment:
            raise ValueError("threshold task numerical environment differs from the frozen reference")
    rows = [row for row in report["rows"] if row["arm"] == "P0"]
    if (len(rows) != 6 or any(row["status"] != "complete" or row["healthy"] is not True for row in rows)
            or {(r["partition"], r["malicious_ratio"]) for r in rows} != set(REVIEWED_P0)):
        raise ValueError("all six original P0 tasks must be complete and healthy")
    for row in rows:
        accuracy, asr, fp = REVIEWED_P0[(row["partition"], row["malicious_ratio"])]
        if (abs(row["accuracy150"] - accuracy) > 1e-12 or row["false_positive_revocations"] != fp
                or (asr is not None and abs(row["attack_asr150"] - asr) > 1e-7)):
            raise ValueError("P0 results differ from the reviewed history-ablation anchor")
        if any(row["mechanism_windows"][window]["client_diagnostics"]["status"] != "available"
               for window in ("first_round", "first_five_rounds")):
            raise ValueError("P0 mechanism observations are unavailable")
    if before != threshold_evidence_hashes(output, tasks):
        raise ValueError("threshold evidence changed while being audited")
    old = manifest["reference"]
    return {"threshold_manifest_fingerprint": manifest["fingerprint"],
        "timing_manifest_fingerprint": old["timing_manifest_fingerprint"],
        "clean_manifest_fingerprint": old["clean_manifest_fingerprint"],
        "matched_manifest_fingerprint": old["matched_manifest_fingerprint"],
        "threshold_evidence_sha256": before, "upstream_reference": old,
        "data_contract": manifest["data_contract"], "dataset": manifest["spec"]["dataset"],
        "execution_environment": environment, "chosen_candidate": "P0", "chosen_model": "v7_cnn",
        "p0_tasks": [task for task in tasks if task["arm"] == "P0"], "p0_rows": rows,
        "clean_rows": old["clean_rows"], "threshold_health_count": report["healthy_tasks"],
        "failure_rows": [{k: row[k] for k in ("task_id", "arm", "healthy", "reasons", "nonfinite_updates")}
                         for row in report["rows"] if row["healthy"] is not True],
        "selection_role": "reviewed original P0 anchor; no formal qualification or new parameter selection",
        "repeatability_cause": "unresolved; both ablation arms start from fresh training"}


def validate_reference(spec, reference):
    if (reference["threshold_manifest_fingerprint"] != spec["threshold_manifest_fingerprint"]
            or reference["chosen_candidate"] != "P0" or reference["chosen_model"] != "v7_cnn"):
        raise ValueError("history panel reference/anchor identity mismatch")


def build_manifest(spec, reference):
    validate_spec(spec)
    validate_reference(spec, reference)
    payload = {"protocol": PROTOCOL, "spec": {**spec, "dataset": reference["dataset"]},
        "reference": reference, "data_contract": reference["data_contract"], "source_sha256": source_hashes(),
        "purpose": "development_history_ablation", "evaluation_split": "calibration_dataset",
        "official_test_used_for_selection": False, "both_arms_fresh": True, "next_stage_automatic": False,
        "automatic_tpe": False, "baseline_algorithms_modified": False, "ours_algorithm_modified": True,
        "modified_variant": "Ours-FrozenHistory-v1", "unmodified_control": "H0",
        "numerical_policy_modified": False, "formal_qualification_assessed": False}
    return {**payload, "fingerprint": base.digest(payload)}


def build_tasks(manifest):
    spec, reference = manifest["spec"], manifest["reference"]
    validate_spec({k: v for k, v in spec.items() if k != "dataset"})
    validate_reference(spec, reference)
    originals = reference["p0_tasks"]
    if len(originals) != 6 or {(t["config"]["partition"], t["config"]["malicious_ratio"])
                              for t in originals} != set(REVIEWED_P0):
        raise ValueError("six original P0 scenario identities must be retained")
    for task in originals:
        raw = {k: v for k, v in task.items() if k != "fingerprint"}
        if base.attach_fingerprints([raw], {"fingerprint": reference["threshold_manifest_fingerprint"]})[0] != task:
            raise ValueError("original P0 task fingerprint mismatch")
        expected = {**threshold.timing.OURS_014,
            **{k: v for k, v in threshold.timing.PUBLIC.items() if k != "model"}, **threshold.timing.ATTACK,
            "seed": spec["seed"], "detector_window": 20, "attack_start_round": 25, "method": "sm9rrs"}
        if (task["arm"] != "P0" or task["model"] != "v7_cnn" or task["candidate"]["variant"] != "original"
                or any(task["config"][k] != v for k, v in expected.items())):
            raise ValueError("original P0 is not the fixed original-014 public development condition")
    tasks = []
    for arm in ARMS:
        for old in originals:
            config = deepcopy(old["config"])
            base.fl.ExperimentConfig(**config).validate()
            ident = arm["id"]
            tasks.append({"task_id": f"history_{ident}_sm9rrs_{config['partition']}_ratio{config['malicious_ratio']:g}_seed{config['seed']}",
                "phase": "validation", "purpose": "development_history_ablation", "arm": ident,
                "method": "sm9rrs", "model": "v7_cnn", "candidate": {"candidate_id": ident,
                    "variant": arm["variant"], "parameters": deepcopy(old["candidate"]["parameters"])},
                "history_freeze_start_round": arm["history_freeze_start_round"],
                "ours_algorithm_modified": ident == "H1", "config": config,
                "original_p0_task_id": old["task_id"], "original_p0_fingerprint": old["fingerprint"],
                "clean_reference_task_id": old["clean_reference_task_id"],
                "clean_reference_fingerprint": old["clean_reference_fingerprint"]})
    return base.attach_fingerprints(tasks, manifest)


def read_study(output, *, current_sources=False):
    output = Path(output)
    manifest = runtime.read_json(output / "manifest.json")
    if (manifest.get("protocol") != PROTOCOL
            or base.digest({k: v for k, v in manifest.items() if k != "fingerprint"}) != manifest["fingerprint"]):
        raise ValueError("history panel manifest fingerprint/protocol mismatch")
    spec = {k: v for k, v in manifest["spec"].items() if k != "dataset"}
    validate_spec(spec)
    validate_reference(spec, manifest["reference"])
    if current_sources and manifest != build_manifest(spec, manifest["reference"]):
        raise ValueError("history panel source/reference/data identity changed")
    if (manifest["data_contract"] != manifest["reference"]["data_contract"]
            or manifest["spec"]["dataset"] != manifest["reference"]["dataset"]):
        raise ValueError("history panel dataset identity mismatch")
    tasks = build_tasks(manifest)
    if runtime.read_json(output / "task_plans/history.json") != {"manifest_fingerprint": manifest["fingerprint"], "tasks": tasks}:
        raise ValueError("history panel task plan mismatch")
    return manifest, tasks
