"""Four predeclared CNN threshold points, with a freshly trained P0 repeat.

This is a development panel, not TPE or the original six-method qualification.
All previous studies and their scientific sources remain immutable references.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import cifar_timing_protocol as timing
import cifar_timing_report as timing_report
import summarize_cifar_timing as brief

base, runtime = timing.base, timing.runtime
REPO = Path(__file__).resolve().parent
PROTOCOL = "cifar-cnn-fixed-threshold-panel-v1"
DEFAULT_CONFIG = REPO / "configs/cifar10_cnn_threshold_panel_v1.json"
DEFAULT_TIMING = timing.DEFAULT_OUTPUT
DEFAULT_CLEAN, DEFAULT_MATCHED = timing.DEFAULT_CLEAN, timing.DEFAULT_MATCHED
DEFAULT_OUTPUT = REPO / "outputs/cifar_v8_diagnostic_v1/threshold_cnn_e1_k20"
TIMING_FINGERPRINT = "20837b56e9a8d35ed0922a0e2ce41c821a1fcbebf8f609f21e04a47bfc479a60"
CANDIDATES = [
    {"id": "P0", "warning": 1.25, "kappa": 1.25, "h": 6.},
    {"id": "P1", "warning": 1.50, "kappa": 1.25, "h": 6.},
    {"id": "P2", "warning": 1.50, "kappa": .85, "h": 1.5},
    {"id": "P3", "warning": 1.75, "kappa": 1., "h": 2.},
]
PARAMETER_FIELDS = {"warning": "detector_distance_threshold", "kappa": "detector_drift_allowance",
                    "h": "detector_drift_threshold"}
NEW_SOURCES = ("summarize_cifar_timing.py", "cifar_cnn_threshold_protocol.py",
               "run_cifar_cnn_threshold_panel.py", "cifar_cnn_threshold_report.py")
REVIEWED_C = {
    ("iid", 0.): (.6332, None, 0), ("iid", .1): (.6256, .095, 0),
    ("iid", .7): (.5952, .155, 2), ("dirichlet", 0.): (.6096, None, 2),
    ("dirichlet", .1): (.6024, .09, 4), ("dirichlet", .7): (.478, .53, 12),
}


def validate_spec(spec):
    expected = {"schema_version": 1, "protocol": PROTOCOL,
        "timing_manifest_fingerprint": TIMING_FINGERPRINT, "chosen_arm": "C",
        "public": timing.PUBLIC, "attack": timing.ATTACK, "ours_parameters": timing.OURS_014,
        "detector_window": 20, "attack_start_round": 25, "seed": 2026093001,
        "partitions": ["iid", "dirichlet"], "malicious_ratios": [0., .1, .7],
        "candidates": CANDIDATES, "p0_fresh_repeat": True, "automatic_tpe": False}
    if {k: v for k, v in spec.items() if k != "output_dir"} != expected or "output_dir" not in spec:
        raise ValueError("fixed panel requires exactly P0-P3, six C scenarios, fresh P0, and no automatic TPE")
    return spec


def source_hashes():
    values = timing.source_hashes()
    for name in NEW_SOURCES:
        values[name] = hashlib.sha256((REPO / name).read_bytes()).hexdigest()
    return values


def timing_evidence_hashes(output, tasks):
    output = Path(output)
    paths = [output / name for name in ("manifest.json", "task_plans/timing.json", "execution_environment.json")]
    for task in tasks:
        folder = output / "tasks" / task["task_id"]
        paths.extend(folder / name for name in ("task.json", "environment.json", base.experiments.COMPLETED_RESULTS_SNAPSHOT))
        paths.extend(sorted((folder / "attempts").glob("*.json")))
    return {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def audit_reference(timing_output, clean_output, matched_output):
    """Read raw 24+4+26 evidence; retain old failures instead of hiding them."""
    output = Path(timing_output).resolve()
    manifest, tasks = timing.read_study(output, current_sources=True)
    if manifest["fingerprint"] != TIMING_FINGERPRINT:
        raise ValueError("fixed panel requires the reviewed 26-task timing study")
    before = timing_evidence_hashes(output, tasks)
    report = timing_report.summarize(output, Path(clean_output), Path(matched_output))
    if (report.get("status") != "complete" or report.get("complete_tasks") != 26
            or any(report.get(key) is not True for key in
                   ("reference_verified", "source_matches_current", "execution_environment_compatible"))):
        raise ValueError("all 26 timing snapshots and their sources/references/environment must be verifiable")
    context = brief.audit_context(output)
    if context.get("status") != "audited" or context.get("audit_manifest_fingerprint") != manifest["fingerprint"]:
        raise ValueError("clean A/B configuration and recorded environment audit is incomplete or mismatched")
    environment = manifest["reference"]["execution_environment"]
    for task in tasks:
        metadata = runtime.read_json(output / "tasks" / task["task_id"] / "environment.json")
        if timing.matched.normalized_environment(metadata) != environment:
            raise ValueError("timing task numerical environment differs from the frozen reference")
    c_rows = [row for row in report["rows"] if row["arm"] == "C"]
    if (len(c_rows) != 6 or any(row["status"] != "complete" or row["healthy"] is not True for row in c_rows)
            or {(r["partition"], r["malicious_ratio"]) for r in c_rows} != set(REVIEWED_C)):
        raise ValueError("all six original C tasks must be complete and healthy")
    for row in c_rows:
        accuracy, asr, fp = REVIEWED_C[(row["partition"], row["malicious_ratio"])]
        if (abs(row["accuracy150"] - accuracy) > 1e-12 or row["false_positive_revocations"] != fp
                or (asr is not None and abs(row["attack_asr150"] - asr) > 1e-7)):
            raise ValueError("C results differ from the reviewed development anchor")
        if any(row["mechanism_windows"][window]["client_diagnostics"]["status"] != "available"
               for window in ("first_round", "first_five_rounds")):
            raise ValueError("C mechanism observations are unavailable")
    if before != timing_evidence_hashes(output, tasks):
        raise ValueError("timing evidence changed while being audited")
    old = manifest["reference"]
    return {"timing_manifest_fingerprint": manifest["fingerprint"],
        "clean_manifest_fingerprint": old["clean_manifest_fingerprint"],
        "matched_manifest_fingerprint": old["matched_manifest_fingerprint"],
        "timing_evidence_sha256": before, "clean_evidence_sha256": old["clean_evidence_sha256"],
        "matched_evidence_sha256": old["matched_evidence_sha256"],
        "data_contract": manifest["data_contract"], "dataset": manifest["spec"]["dataset"],
        "execution_environment": environment, "chosen_arm": "C", "chosen_model": "v7_cnn",
        "c_tasks": [task for task in tasks if task["arm"] == "C"], "c_rows": c_rows,
        "clean_rows": old["clean_rows"], "clean_pair_audits": context["audits"],
        "timing_health_count": report["healthy_tasks"],
        "failure_rows": [{k: row[k] for k in ("task_id", "arm", "healthy", "reasons", "nonfinite_updates")}
                         for row in report["rows"] if row["healthy"] is not True],
        "selection_role": "healthy development anchor; not best defense or formal qualification",
        "repeatability_cause": "unresolved; recorded environment agreement is not bitwise reproducibility"}


def validate_reference(spec, reference):
    if (reference["timing_manifest_fingerprint"] != spec["timing_manifest_fingerprint"]
            or reference["chosen_arm"] != "C" or reference["chosen_model"] != "v7_cnn"):
        raise ValueError("fixed panel reference/anchor identity mismatch")


def build_manifest(spec, reference):
    validate_spec(spec)
    validate_reference(spec, reference)
    payload = {"protocol": PROTOCOL, "spec": {**spec, "dataset": reference["dataset"]},
        "reference": reference, "data_contract": reference["data_contract"], "source_sha256": source_hashes(),
        "purpose": "development_fixed_threshold_panel", "evaluation_split": "calibration_dataset",
        "official_test_used_for_selection": False, "p0_fresh_repeat": True, "next_stage_automatic": False,
        "automatic_tpe": False, "baseline_algorithms_modified": False, "ours_algorithm_modified": False,
        "numerical_policy_modified": False, "formal_qualification_assessed": False}
    return {**payload, "fingerprint": base.digest(payload)}


def build_tasks(manifest):
    spec, reference = manifest["spec"], manifest["reference"]
    validate_spec({k: v for k, v in spec.items() if k != "dataset"})
    validate_reference(spec, reference)
    originals = reference["c_tasks"]
    if len(originals) != 6 or {(t["config"]["partition"], t["config"]["malicious_ratio"])
                              for t in originals} != set(REVIEWED_C):
        raise ValueError("six original C scenario identities must be retained")
    for task in originals:
        raw = {k: v for k, v in task.items() if k != "fingerprint"}
        if base.attach_fingerprints([raw], {"fingerprint": reference["timing_manifest_fingerprint"]})[0] != task:
            raise ValueError("original C task fingerprint mismatch")
        config = task["config"]
        expected = {**timing.OURS_014, **{k: v for k, v in timing.PUBLIC.items() if k != "model"},
                    **timing.ATTACK, "seed": spec["seed"], "detector_window": 20, "attack_start_round": 25,
                    "method": "sm9rrs"}
        if task["arm"] != "C" or task["model"] != "v7_cnn" or any(config[k] != v for k, v in expected.items()):
            raise ValueError("original C is not the fixed public/014 development condition")
    tasks = []
    for candidate in CANDIDATES:
        parameters = {PARAMETER_FIELDS[key]: candidate[key] for key in PARAMETER_FIELDS}
        for old in originals:
            config = {**deepcopy(old["config"]), **parameters}
            base.fl.ExperimentConfig(**config).validate()
            ident = candidate["id"]
            tasks.append({"task_id": f"threshold_{ident}_sm9rrs_{config['partition']}_ratio{config['malicious_ratio']:g}_seed{config['seed']}",
                "phase": "validation", "purpose": "development_fixed_threshold_panel", "arm": ident,
                "method": "sm9rrs", "model": "v7_cnn", "candidate": {"candidate_id": ident,
                    "variant": "original", "parameters": {**timing.OURS_014, **parameters,
                        "detector_window": 20, "attack_start_round": 25}}, "config": config,
                "original_c_task_id": old["task_id"], "original_c_fingerprint": old["fingerprint"],
                "clean_reference_task_id": old["clean_reference_task_id"],
                "clean_reference_fingerprint": old["clean_reference_fingerprint"]})
    return base.attach_fingerprints(tasks, manifest)


def read_study(output, *, current_sources=False):
    output = Path(output)
    manifest = runtime.read_json(output / "manifest.json")
    if (manifest.get("protocol") != PROTOCOL
            or base.digest({k: v for k, v in manifest.items() if k != "fingerprint"}) != manifest["fingerprint"]):
        raise ValueError("threshold panel manifest fingerprint/protocol mismatch")
    spec = {k: v for k, v in manifest["spec"].items() if k != "dataset"}
    validate_spec(spec)
    validate_reference(spec, manifest["reference"])
    if current_sources and manifest != build_manifest(spec, manifest["reference"]):
        raise ValueError("threshold panel source/reference/data identity changed")
    if (manifest["data_contract"] != manifest["reference"]["data_contract"]
            or manifest["spec"]["dataset"] != manifest["reference"]["dataset"]):
        raise ValueError("threshold panel dataset identity mismatch")
    tasks = build_tasks(manifest)
    if runtime.read_json(output / "task_plans/threshold.json") != {"manifest_fingerprint": manifest["fingerprint"], "tasks": tasks}:
        raise ValueError("threshold panel task plan mismatch")
    return manifest, tasks
