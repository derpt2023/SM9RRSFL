"""Immutable identities for a bounded, original-numerics local-step diagnostic."""
from copy import deepcopy
import hashlib
from pathlib import Path
import zipfile

import cifar_prefix_probe_protocol as prefix
import cifar_prefix_probe_report as prefix_report
import diagnose_cifar_prefix_clients as clients

base, runtime = prefix.base, prefix.runtime
REPO = Path(__file__).resolve().parent
PROTOCOL = "cifar-client-step-probe-v1"
DEFAULT_OUTPUT = REPO / "outputs/cifar_v8_diagnostic_v1/client_step_probe_v1"
DEFAULT_PREFIX = prefix.DEFAULT_OUTPUT
TARGET_CLIENTS = [19, 84]
STOP_AFTER_CLIENT = 84
NEW_SOURCES = ("diagnose_cifar_prefix_clients.py", "resume_cifar_prefix_probe.py",
    "cifar_client_step_probe_protocol.py", "cifar_client_step_probe_runtime.py",
    "cifar_client_step_probe_report.py", "run_cifar_client_step_probe.py")


def source_hashes():
    values = prefix.source_hashes()
    values.update({name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in NEW_SOURCES})
    return values


def separate_outputs(output, prefix_output, old_paths=()):
    output = Path(output).resolve()
    for other in [prefix_output, *old_paths]:
        other = Path(other).resolve()
        if output == other or output in other.parents or other in output.parents:
            raise ValueError("step output must be separate and non-nested with every reference study")


def audit_reference(prefix_output):
    prefix_output = Path(prefix_output).resolve()
    before = prefix_report.evidence_hashes(prefix_output)
    review = clients.summarize(prefix_output)
    if review.get("status") != "complete":
        raise ValueError("complete client diagnosis required: " + str(review))
    manifest, tasks = prefix.read_study(prefix_output, current_sources=True)
    for partition in review["partitions"]:
        expected = [] if partition["partition"] == "iid" else ["client-19", "client-84"]
        if not partition["common"]["all_client_inputs_equal"] or any(
                pair["update_differing_ids"] != expected or pair["loss_differing_ids"]
                or pair["input_differing_ids"] for pair in partition["pairs"]):
            raise ValueError("reference is not the reviewed two-client first-round pattern")
    anchors = []
    for task in tasks:
        if task["partition"] == "dirichlet":
            artifact = prefix.load_completed(prefix_output, task)
            observations = prefix_report.validate_observations(artifact["observations"], task)
            anchors.append({"task": task, "observations": observations,
                            "artifact_fingerprint": artifact["artifact_fingerprint"]})
    anchors.sort(key=lambda anchor: anchor["task"]["repeat"])
    if [a["task"]["repeat"] for a in anchors] != [1, 2, 3]:
        raise ValueError("three original Dirichlet anchors are required")
    for anchor in anchors:
        for index, samples in ((19, 701), (84, 301)):
            row = anchor["observations"]["rounds"][0]["clients"][index]
            if row["samples"] != samples or row["batch_size"] != 50 or row["epochs"] != 1:
                raise ValueError("target sample/batch configuration differs from reviewed evidence")
    if before != prefix_report.evidence_hashes(prefix_output):
        raise ValueError("prefix evidence changed during the reference audit")
    return {"prefix_output": str(prefix_output), "prefix_manifest": manifest,
        "prefix_evidence_sha256": before, "anchors": anchors,
        "execution_environment": manifest["reference"]["execution_environment"],
        "client_reader_sha256": review["reader_sha256"],
        "role": "development localization only; not a new selection or performance qualification"}


def verify_reference(reference):
    output = Path(reference["prefix_output"])
    if prefix_report.evidence_hashes(output) != reference["prefix_evidence_sha256"]:
        raise ValueError("frozen six-prefix evidence changed")
    manifest, tasks = prefix.read_study(output, current_sources=True)
    if manifest != reference["prefix_manifest"]:
        raise ValueError("frozen prefix manifest changed")
    if reference["execution_environment"] != manifest["reference"]["execution_environment"]:
        raise ValueError("reference numerical environment differs")
    if reference["client_reader_sha256"] != hashlib.sha256((REPO / "diagnose_cifar_prefix_clients.py").read_bytes()).hexdigest():
        raise ValueError("reviewed client reader changed")
    wanted = [t for t in tasks if t["partition"] == "dirichlet"]
    if len(reference["anchors"]) != 3 or [a["task"] for a in reference["anchors"]] != wanted:
        raise ValueError("reference anchor task identities differ")
    for anchor in reference["anchors"]:
        artifact = prefix.load_completed(output, anchor["task"])
        if artifact is None or artifact["observations"] != anchor["observations"] or artifact["artifact_fingerprint"] != anchor["artifact_fingerprint"]:
            raise ValueError("reference anchor observations changed")


def build_manifest(reference):
    old = reference["prefix_manifest"]
    value = {"protocol": PROTOCOL, "reference": deepcopy(reference), "source_sha256": source_hashes(),
        "same_gpu_uuid": old["same_gpu_uuid"], "data_contract": old["data_contract"],
        "spec": {"dataset": old["spec"]["dataset"]},
        "target_clients": list(TARGET_CLIENTS), "stop_after_client": STOP_AFTER_CLIENT,
        "stop_boundary": "before_round1_client85_local_training",
        "repeats": 3, "original_numerical_policy": True, "original_training_function": True,
        "fresh_process_per_repeat": True, "serial_same_physical_gpu": True,
        "aggregation_executed": False, "training_round_completed": False,
        "checkpoints_reused": False, "health_assessed": False, "formal_qualification_assessed": False,
        "next_stage_automatic": False,
        "observation_limit": "added host copies can change timing; exact agreement does not certify unobserved execution"}
    return {**value, "fingerprint": base.digest(value)}


def build_tasks(manifest):
    old = manifest["reference"]["anchors"][0]["task"]
    config = old["config"]
    if (config["rounds"], config["num_clients"], config["malicious_ratio"], config["partition"],
            config["seed"], config["local_epochs"], config["batch_size"]) != (3, 100, 0., "dirichlet", 2026093001, 1, 50):
        raise ValueError("local-step anchor configuration differs")
    tasks = [{"task_id": "client_step_repeat" + str(repeat), "repeat": repeat, "partition": "dirichlet",
        "config": deepcopy(config), "target_clients": list(TARGET_CLIENTS), "stop_after_client": STOP_AFTER_CLIENT,
        "same_gpu_uuid": manifest["same_gpu_uuid"], "original_prefix_task_fingerprint": old["fingerprint"],
        "purpose": "local_minibatch_boundary_localization"} for repeat in (1, 2, 3)]
    return base.attach_fingerprints(tasks, manifest)


def read_study(output, *, current_sources=False, verify_reference=True):
    output = Path(output).resolve()
    manifest = runtime.read_json(output / "manifest.json")
    if manifest.get("protocol") != PROTOCOL or base.digest({k: v for k, v in manifest.items() if k != "fingerprint"}) != manifest.get("fingerprint"):
        raise ValueError("step manifest identity mismatch")
    expected = build_manifest(manifest["reference"])
    if not current_sources:
        expected["source_sha256"] = manifest["source_sha256"]
        expected["fingerprint"] = base.digest({k: v for k, v in expected.items() if k != "fingerprint"})
    if expected != manifest:
        raise ValueError("step source/configuration identity changed")
    reference = manifest["reference"]
    separate_outputs(output, reference["prefix_output"], reference["prefix_manifest"]["reference"]["paths"].values())
    if verify_reference:
        globals()["verify_reference"](reference)
    tasks = build_tasks(manifest)
    if runtime.read_json(output / "task_plans/steps.json") != {"manifest_fingerprint": manifest["fingerprint"], "tasks": tasks}:
        raise ValueError("step task plan differs")
    return manifest, tasks


def seal_artifact(value):
    return {**value, "artifact_fingerprint": base.digest(value)}


def load_completed(output, task):
    folder = Path(output) / "tasks" / task["task_id"]
    if runtime.read_json(folder / "task.json") != task:
        raise ValueError("step task identity mismatch")
    path = folder / "completed.json"
    if not path.exists():
        return None
    value = runtime.read_json(path)
    if (value.get("status") != "complete" or value.get("task_fingerprint") != task["fingerprint"]
            or value.get("gpu_uuid") != task["same_gpu_uuid"]
            or base.digest({k: v for k, v in value.items() if k != "artifact_fingerprint"}) != value.get("artifact_fingerprint")
            or value.get("fresh_start") is not True or value.get("checkpoints_used") is not False):
        raise ValueError("step completion identity mismatch")
    prefix.validate_gpu_environment(value["environment"], task["same_gpu_uuid"])
    _snapshot_path(output, task, value)
    return value


def _snapshot_path(output, task, artifact):
    root = Path(output).resolve()
    folder = root / "tasks" / task["task_id"]
    if folder.is_symlink() or folder.parent.is_symlink() or root not in folder.resolve().parents:
        raise ValueError("snapshot task directory escapes its study or uses a symlink")
    folder = folder.resolve()
    entry = artifact["snapshot_file"]
    relative = Path(entry["path"])
    if relative.is_absolute() or len(relative.parts) != 3 or relative.parts[0] != "attempts" or relative.parts[-1] != "snapshots.npz" or ".." in relative.parts:
        raise ValueError("invalid snapshot relative path")
    path = folder / relative
    if path.is_symlink() or path.parent.is_symlink() or path.parent.parent.is_symlink() or folder not in path.resolve().parents:
        raise ValueError("snapshot path escapes its task or uses a symlink")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("snapshot file SHA mismatch")
    return path


def load_arrays(output, task, artifact):
    import numpy as np
    path = _snapshot_path(output, task, artifact)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 64 or sum(m.file_size for m in members) > 256 * 1024 * 1024:
            raise ValueError("snapshot archive exceeds the bounded diagnostic size")
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("duplicate snapshot array names")
        return {key: archive[key] for key in archive.files}


def evidence_hashes(output):
    output = Path(output)
    paths = {output / "manifest.json", output / "task_plans/steps.json"}
    for pattern in ("tasks/*/task.json", "tasks/*/completed.json", "tasks/*/attempts/*/snapshots.npz"):
        paths.update(output.glob(pattern))
    return {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None for path in sorted(paths)}
