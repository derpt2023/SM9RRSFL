"""Independent eighty-round clean history comparison after the reviewed panel.

The original 93 sources and all 115 prior tasks remain frozen. Both new arms
use the already observed singleton numerical policy; only history admission
differs. No checkpoint or previous training state is reused.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import cifar_mechanism_protocol as mechanism
import diagnose_cifar_mechanism as reader

base, runtime = mechanism.base, mechanism.runtime
prefix_report = mechanism.prefix_report
REPO = Path(__file__).resolve().parent
PROTOCOL = "cifar-clean-history-horizon80-v1"
DEFAULT_OUTPUT = REPO / "outputs/cifar_v8_diagnostic_v1/clean_horizon80_v1"
DEFAULT_MECHANISM = mechanism.DEFAULT_OUTPUT
ROUNDS = 80
POLICY = mechanism.POLICY
ARMS = mechanism.ARMS
NEW_SOURCES = ("cifar_clean_horizon_protocol.py", "cifar_clean_horizon_report.py", "run_cifar_clean_horizon_panel.py")
EXPECTED_MECHANISM = reader.EXPECTED_MANIFEST
EXPECTED_DETAIL = "5762830456c3b1868302c32c621cb4499c4c5a83bf3d8ff84067cbf250a88a5e"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def source_hashes():
    values = mechanism.source_hashes()
    values.update(reader.reader_hashes())
    values.update({name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in NEW_SOURCES})
    return values


def separate_outputs(output, mechanism_output, old_paths=()):
    mechanism.separate_outputs(output, mechanism_output, old_paths)


def reference_paths(reference):
    return [reference["mechanism_output"], *reference["upstream_paths"]]


def _snapshot(output, reference):
    return {"sources": mechanism.source_hashes(), "readers": reader.reader_hashes(),
            "mechanism": mechanism.evidence_hashes(output),
            "upstream": reader.reference_evidence_hashes(reference)}


def audit_reference(mechanism_output):
    """Recompute the four actual artifacts and the complete 111-task ancestry."""
    output = Path(mechanism_output).resolve()
    original = runtime.read_json(output / "manifest.json")
    before = _snapshot(output, original["reference"])
    records = reader.diagnose(output)
    _require([r.get("type") for r in records] ==
             ["header", "legend", "paired_details", "repeat_confirmation", "decision"],
             "complete reviewed mechanism details required")
    header, paired, confirmation = records[0], records[2], records[3]
    _require(header.get("status") == "complete" and header.get("manifest_fingerprint") == EXPECTED_MECHANISM
             and all(header.get(k) is True for k in ("source_reference_and_evidence_verified", "input_evidence_unchanged",
                 "within_arm_all_observations_equal", "round26_local_training_equal", "detailed_pair_repeats_equal"))
             and header.get("complete_tasks") == header.get("available_pairs") == 4,
             "reviewed paired clean evidence is incomplete or changed")
    _require(paired.get("paired_detail_sha256") == confirmation.get("paired_detail_sha256") == EXPECTED_DETAIL
             and confirmation.get("detailed_pair_equal_to_repeat1") is True,
             "mechanism details differ from the reviewed paired result")
    content = {k: v for k, v in paired.items() if k not in
               ("repeat", "h0_task_id", "h1_task_id", "paired_detail_sha256")}
    _require(base.digest(content) == EXPECTED_DETAIL, "mechanism detail digest does not match actual content")
    manifest, tasks = mechanism.read_study(output, current_sources=True)
    _require(manifest == original and manifest["fingerprint"] == EXPECTED_MECHANISM,
             "reviewed mechanism identity differs")
    _require(header["source_count"] == len(manifest["source_sha256"])
             and header["source_map_sha256"] == base.digest(manifest["source_sha256"])
             and header["reader_sha256"] == before["readers"]
             and header["same_gpu_uuid"] == manifest["same_gpu_uuid"], "reference source/readers/GPU identity differs")
    fingerprints = {}
    for task in tasks:
        artifact = mechanism.load_completed(output, task)
        _require(artifact is not None and artifact.get("fresh_start") is True and artifact.get("checkpoints_used") is False
                 and artifact.get("completed_training_rounds") == 30, "missing or nonfresh full mechanism anchor")
        fingerprints[task["task_id"]] = artifact["artifact_fingerprint"]
    _require(before == _snapshot(output, manifest["reference"]), "reference evidence or sources changed during horizon audit")
    return {"mechanism_output": str(output), "mechanism_manifest_fingerprint": manifest["fingerprint"],
        "mechanism_tasks": deepcopy(tasks), "mechanism_artifact_fingerprints": fingerprints,
        "mechanism_evidence_sha256": before["mechanism"], "upstream_evidence_sha256": before["upstream"],
        "upstream_paths": mechanism.reference_paths(manifest["reference"]), "frozen_source_sha256": {
            **before["sources"], **before["readers"]}, "reviewed_detail_sha256": EXPECTED_DETAIL,
        "same_gpu_uuid": manifest["same_gpu_uuid"], "execution_environment": deepcopy(manifest["reference"]["execution_environment"]),
        "spec": deepcopy(manifest["spec"]), "data_contract": deepcopy(manifest["data_contract"]),
        "reference_task_count": 115, "role": "reviewed short clean evidence; not a new selection or full qualification"}


def verify_reference(reference):
    _require(audit_reference(reference["mechanism_output"]) == reference,
             "frozen mechanism or upstream reference changed")


def load_mechanism_anchors(reference):
    """Read four old public observation artifacts for eight prefix comparisons."""
    output = Path(reference["mechanism_output"])
    anchors = []
    for task in reference["mechanism_tasks"]:
        artifact = mechanism.load_completed(output, task)
        _require(artifact is not None and artifact["artifact_fingerprint"] ==
                 reference["mechanism_artifact_fingerprints"][task["task_id"]], "mechanism anchor changed or disappeared")
        anchors.append({"task": deepcopy(task), "observations": artifact["observations"],
                        "artifact_fingerprint": artifact["artifact_fingerprint"]})
    return anchors


def build_manifest(reference):
    _require(reference["mechanism_manifest_fingerprint"] == EXPECTED_MECHANISM
             and reference["reviewed_detail_sha256"] == EXPECTED_DETAIL and reference["reference_task_count"] == 115,
             "requires the reviewed thirty-round mechanism evidence")
    current = source_hashes()
    _require(all(current.get(name) == sha for name, sha in reference["frozen_source_sha256"].items())
             and set(reference["frozen_source_sha256"]) == set(mechanism.source_hashes()) | set(reader.reader_hashes()),
             "frozen 93 source identity changed")
    value = {"protocol": PROTOCOL, "reference": deepcopy(reference), "source_sha256": current,
        "same_gpu_uuid": reference["same_gpu_uuid"], "spec": deepcopy(reference["spec"]),
        "data_contract": deepcopy(reference["data_contract"]), "rounds": ROUNDS,
        "partition": "dirichlet", "malicious_ratio": 0., "num_clients": 100,
        "execution_order": [{"arm": arm, "repeat": repeat} for repeat in (1, 2) for arm, _, _ in ARMS],
        "numerical_policy": POLICY, "numerical_scope": "actual singleton ordinary local-training backward only; no attacks",
        "history_intervention": {"H0": "original", "H1": "suppress history admission from round25 commit; original evaluation/weighting/revocation"},
        "fresh_process_per_task": True, "serial_same_physical_gpu": True, "checkpoints_reused": False,
        "maximum_training_rounds": 320, "maximum_client_training_calls": 32000,
        "prefix_validation": "each new arm must reproduce both same-arm old runs through round30; eight prefix comparisons",
        "late_windows": [[31, 51], [52, 80]], "diagnostic_health_only": True,
        "formal_qualification_assessed": False, "automatic_next_stage": False,
        "limitation": "one seed and one physical GPU; eighty clean rounds cover a late interval but do not qualify the original150-round protocol or assess attacks"}
    return {**value, "fingerprint": base.digest(value)}


def build_tasks(manifest):
    old = manifest["reference"]["mechanism_tasks"]
    _require([(t["arm"], t["repeat"]) for t in old] == [(a, r) for r in (1, 2) for a, _, _ in ARMS],
             "four interleaved original mechanism tasks required")
    tasks = []
    for anchor in old:
        config = deepcopy(anchor["config"])
        _require((config["rounds"], config["num_clients"], config["malicious_ratio"], config["partition"], config["seed"],
                  config["local_epochs"], config["batch_size"], config["detector_window"], config["attack_start_round"]) ==
                 (30, 100, 0., "dirichlet", 2026093001, 1, 50, 20, 25), "reviewed mechanism configuration differs")
        _require(anchor["same_gpu_uuid"] == manifest["same_gpu_uuid"] and anchor["policy"] == POLICY,
                 "reviewed mechanism GPU or numerical policy differs")
        config["rounds"] = ROUNDS
        arm, repeat = anchor["arm"], anchor["repeat"]
        expected = next((variant, freeze) for a, variant, freeze in ARMS if a == arm)
        _require(anchor["candidate"]["variant"] == expected[0] and anchor["history_freeze_start_round"] == expected[1],
                 "reviewed history intervention differs")
        tasks.append({"task_id": f"clean_horizon80_{arm}_dirichlet_clean_repeat{repeat}", "arm": arm, "repeat": repeat,
            "partition": "dirichlet", "policy": POLICY, "candidate": deepcopy(anchor["candidate"]),
            "history_freeze_start_round": expected[1], "config": config, "same_gpu_uuid": manifest["same_gpu_uuid"],
            "original_mechanism_task_fingerprint": anchor["fingerprint"],
            "purpose": "bounded_late_clean_history_diagnostic", "ours_algorithm_modified": arm == "H1"})
    return base.attach_fingerprints(tasks, manifest)


def read_study(output, *, current_sources=True, verify_reference=True):
    output = Path(output).resolve()
    manifest = runtime.read_json(output / "manifest.json")
    _require(manifest.get("protocol") == PROTOCOL and
             base.digest({k: v for k, v in manifest.items() if k != "fingerprint"}) == manifest.get("fingerprint"),
             "clean horizon manifest identity mismatch")
    expected = build_manifest(manifest["reference"])
    if not current_sources:
        expected["source_sha256"] = manifest["source_sha256"]
        expected["fingerprint"] = base.digest({k: v for k, v in expected.items() if k != "fingerprint"})
    _require(expected == manifest, "clean horizon source/configuration identity changed")
    paths = reference_paths(manifest["reference"])
    separate_outputs(output, paths[0], paths[1:])
    if verify_reference:
        globals()["verify_reference"](manifest["reference"])
    tasks = build_tasks(manifest)
    _require(runtime.read_json(output / "task_plans/clean_horizon.json") ==
             {"manifest_fingerprint": manifest["fingerprint"], "tasks": tasks}, "clean horizon task plan differs")
    return manifest, tasks


seal_artifact = mechanism.seal_artifact
load_completed = mechanism.load_completed


def evidence_hashes(output):
    output = Path(output)
    paths = {output / "manifest.json", output / "task_plans/clean_horizon.json"}
    for pattern in ("tasks/*/task.json", "tasks/*/completed.json"):
        paths.update(output.glob(pattern))
    return {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file() else None for path in sorted(paths)}
