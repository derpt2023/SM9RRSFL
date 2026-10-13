"""Independent, three-round original-Ours repeatability probe identities.

The reviewed 90-task evidence is read-only. This is not a new training setting,
health qualification, or defense variant. Only the execution horizon is three.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import diagnose_cifar_history as forensic

base, runtime, history = forensic.base, forensic.runtime, forensic.protocol
REPO = Path(__file__).resolve().parent
PROTOCOL = "cifar-prefix-probe-v1"
DEFAULT_OUTPUT = REPO / "outputs/cifar_v8_diagnostic_v1/prefix_probe_v1"
ROUNDS, REPEATS = 3, 3
REFERENCE_NAMES = ("history", "threshold", "timing", "clean", "matched")
NEW_SOURCES = ("diagnose_cifar_history.py", "cifar_history_forensics.py",
               "cifar_prefix_probe_protocol.py", "cifar_prefix_probe_runtime.py",
               "cifar_prefix_probe_report.py", "run_cifar_prefix_probe.py")


def source_hashes():
    values = history.source_hashes()
    values.update({name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in NEW_SOURCES})
    return values


def separate_outputs(output, paths):
    values = [Path(output).resolve(), *(Path(paths[k]).resolve() for k in REFERENCE_NAMES)]
    for i, left in enumerate(values):
        for right in values[i + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("probe and all five reference directories must be separate, non-nested studies")


def evidence_hashes(root):
    """Hash scientific evidence, excluding logs, reports and all checkpoints."""
    root = Path(root)
    files = {root / "manifest.json", root / "execution_environment.json"}
    files.update((root / "task_plans").glob("*.json"))
    for pattern in ("tasks/*/task.json", "tasks/*/environment.json", "tasks/*/observations.json",
                    "tasks/*/" + base.experiments.COMPLETED_RESULTS_SNAPSHOT, "tasks/*/attempts/*.json"):
        files.update(root.glob(pattern))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def audit_reference(paths):
    paths = {k: Path(paths[k]).resolve() for k in REFERENCE_NAMES}
    before = {k: evidence_hashes(p) for k, p in paths.items()}
    rows = forensic.diagnose(*(paths[k] for k in REFERENCE_NAMES))
    header = rows[0]
    if header.get("status") != "complete" or rows[-1].get("failures"):
        raise ValueError("complete history forensics are required before the prefix probe")
    manifest, tasks = history.read_study(paths["history"], current_sources=True)
    h0 = [t for t in tasks if t["arm"] == "H0" and t["config"]["malicious_ratio"] == 0.]
    if len(h0) != 2 or {t["config"]["partition"] for t in h0} != {"iid", "dirichlet"}:
        raise ValueError("two clean original-H0 references are required")
    historical_prefix = {}
    for task in h0:
        result = base.checked_completed(paths["history"], task)
        records = {r.round: r for r in result.records}
        historical_prefix[task["config"]["partition"]] = [
            {"round": r, **{k: getattr(records[r], k) for k in forensic.RECORD_FIELDS}}
            for r in range(ROUNDS + 1)]
    if before != {k: evidence_hashes(p) for k, p in paths.items()}:
        raise ValueError("reference evidence changed during probe audit")
    return {"history_manifest_fingerprint": manifest["fingerprint"],
        "paths": {k: str(p) for k, p in paths.items()}, "evidence_sha256": before,
        "data_contract": manifest["data_contract"], "dataset": manifest["spec"]["dataset"],
        "execution_environment": manifest["reference"]["execution_environment"],
        "h0_tasks": h0, "historical_prefix": historical_prefix,
        "forensics_header": header, "role": "reviewed evidence, not new selection or formal qualification"}


def verify_reference(reference):
    if reference["history_manifest_fingerprint"] != forensic.EXPECTED_MANIFEST:
        raise ValueError("probe reference is not the reviewed history study")
    current = {k: evidence_hashes(reference["paths"][k]) for k in REFERENCE_NAMES}
    if current != reference["evidence_sha256"]:
        raise ValueError("frozen reference evidence changed; preserve all studies")


def build_manifest(reference, gpu_uuid):
    if reference["history_manifest_fingerprint"] != forensic.EXPECTED_MANIFEST:
        raise ValueError("unexpected history reference")
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise ValueError("same physical GPU must be pinned by NVIDIA UUID")
    payload = {"protocol": PROTOCOL, "reference": reference,
        "source_sha256": source_hashes(), "same_gpu_uuid": gpu_uuid,
        "spec": {"dataset": reference["dataset"], "rounds": ROUNDS, "repeats": REPEATS,
                 "partitions": ["iid", "dirichlet"], "seed": 2026093001, "malicious_ratio": 0.},
        "data_contract": reference["data_contract"], "purpose": "short_prefix_repeatability",
        "original_numerical_policy": True, "original_ours_algorithm": True,
        "fresh_process_per_repeat": True, "serial_same_physical_gpu": True,
        "evaluation_split": "calibration_dataset", "official_test_used_for_selection": False,
        "health_assessed": False, "formal_qualification_assessed": False,
        "next_stage_automatic": False, "checkpoints_reused": False,
        "observation_limit": "hash copies synchronize device work; equality does not explain historical multi-GPU runs"}
    return {**payload, "fingerprint": base.digest(payload)}


def build_tasks(manifest):
    originals = manifest["reference"]["h0_tasks"]
    tasks = []
    for partition in ("iid", "dirichlet"):
        matches = [t for t in originals if t["config"]["partition"] == partition]
        if len(matches) != 1:
            raise ValueError("missing unique H0 partition")
        old = matches[0]
        expected = {**history.threshold.timing.OURS_014,
                    **{k: v for k, v in history.threshold.timing.PUBLIC.items() if k != "model"},
                    **history.threshold.timing.ATTACK, "seed": 2026093001,
                    "malicious_ratio": 0., "method": "sm9rrs", "detector_window": 20,
                    "attack_start_round": 25, "eval_interval": 1, "early_stop": False,
                    "compute_backend": "torch", "crypto_mode": "sm9"}
        if old["arm"] != "H0" or old["candidate"]["variant"] != "original" or any(
                old["config"].get(k) != v for k, v in expected.items()):
            raise ValueError("H0 scientific settings differ from the reviewed probe anchor")
        if base.attach_fingerprints([{k: v for k, v in old.items() if k != "fingerprint"}],
                {"fingerprint": manifest["reference"]["history_manifest_fingerprint"]})[0] != old:
            raise ValueError("H0 task fingerprint mismatch")
        for repeat in range(1, REPEATS + 1):
            config = {**deepcopy(old["config"]), "rounds": ROUNDS}
            base.fl.ExperimentConfig(**config).validate()
            tasks.append({"task_id": f"prefix_{partition}_repeat{repeat}", "partition": partition,
                "repeat": repeat, "phase": "diagnostic_prefix", "model": "v7_cnn",
                "purpose": "short_prefix_repeatability", "config": config,
                "original_h0_config": deepcopy(old["config"]), "original_h0_task_id": old["task_id"],
                "original_h0_fingerprint": old["fingerprint"], "same_gpu_uuid": manifest["same_gpu_uuid"]})
    return base.attach_fingerprints(tasks, manifest)


def read_study(output, *, current_sources=False, verify_reference=True):
    output = Path(output)
    manifest = runtime.read_json(output / "manifest.json")
    if (manifest.get("protocol") != PROTOCOL or base.digest({k: v for k, v in manifest.items()
            if k != "fingerprint"}) != manifest.get("fingerprint")):
        raise ValueError("probe manifest identity mismatch")
    # Rebuild the full contract, including declared horizon and execution policy.
    expected = build_manifest(manifest["reference"], manifest["same_gpu_uuid"])
    if not current_sources:
        expected["source_sha256"] = manifest["source_sha256"]
        expected["fingerprint"] = base.digest({k: v for k, v in expected.items() if k != "fingerprint"})
    if expected != manifest:
        raise ValueError("probe source/configuration identity changed")
    separate_outputs(output, manifest["reference"]["paths"])
    if verify_reference:
        globals()["verify_reference"](manifest["reference"])
    tasks = build_tasks(manifest)
    if runtime.read_json(output / "task_plans/prefix.json") != {"manifest_fingerprint": manifest["fingerprint"], "tasks": tasks}:
        raise ValueError("probe task plan differs from manifest")
    return manifest, tasks


def seal_artifact(value):
    return {**value, "artifact_fingerprint": base.digest(value)}


def validate_gpu_environment(metadata, gpu_uuid):
    """Verify the exact UUID mask plus the worker's recorded device inventory.

Older Torch builds expose no device UUID. In that case CUDA's single, full-UUID
visibility mask is the binding evidence; do not invent an observed Torch UUID.
"""
    if metadata.get("environment", {}).get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise ValueError("probe did not pin the declared physical GPU")
    actual = metadata.get("actual_compute_device", {})
    if actual.get("logical_device") != "cuda:0":
        raise ValueError("pinned probe must compute on its sole cuda:0")
    observed = actual.get("uuid")
    if observed is not None and str(observed).removeprefix("GPU-") != gpu_uuid.removeprefix("GPU-"):
        raise ValueError("observed Torch UUID disagrees with the pinned GPU")
    physical = [r for r in metadata.get("nvidia", {}).get("gpus", []) if r.get("uuid") == gpu_uuid]
    logical = metadata.get("torch", {}).get("logical_cuda_devices", [])
    if (len(physical) != 1 or physical[0].get("name") != actual.get("name")
            or len(logical) != 1 or logical[0].get("logical_index") != 0
            or logical[0].get("name") != actual.get("name")):
        raise ValueError("recorded physical/logical GPU inventory does not establish the UUID binding")
    logical_uuid = logical[0].get("uuid")
    if logical_uuid is not None and str(logical_uuid).removeprefix("GPU-") != gpu_uuid.removeprefix("GPU-"):
        raise ValueError("logical CUDA inventory UUID differs from the pinned GPU")


def load_completed(output, task):
    folder = Path(output) / "tasks" / task["task_id"]
    if runtime.read_json(folder / "task.json") != task:
        raise ValueError("prefix task identity mismatch")
    path = folder / "completed.json"
    if not path.exists():
        return None
    result = runtime.read_json(path)
    if (result.get("status") != "complete" or result.get("task_fingerprint") != task["fingerprint"]
            or result.get("gpu_uuid") != task["same_gpu_uuid"]
            or base.digest({k: v for k, v in result.items() if k != "artifact_fingerprint"}) != result.get("artifact_fingerprint")):
        raise ValueError("prefix completion evidence identity mismatch")
    validate_gpu_environment(result["environment"], task["same_gpu_uuid"])
    return result
