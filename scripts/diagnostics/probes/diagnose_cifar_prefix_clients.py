#!/usr/bin/env python3
"""Read all recorded first-round client boundaries without running a new probe.

The existing six completion artifacts and their frozen references are the only
experimental inputs. No dataset, GPU, checkpoint or experiment output is opened
for mutation. Redirect stdout to save this independent reader's JSONL report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import cifar_prefix_probe_protocol as protocol
import cifar_prefix_probe_report as report

SCHEMA = "cifar-prefix-client-diagnosis-v1"
PAIRS = ((1, 2), (1, 3), (2, 3))
INPUT_FIELDS = ("model_input", "partition_indices", "epoch_indices", "training_seed",
                "learning_rate", "epochs", "batch_size", "samples", "minibatch_sizes")
CLIENT_COLUMNS = ("client_id", "samples", "batch_count", "remainder", "last_batch_size",
                  "minibatch_sizes", "update_sha256", "loss", "inputs_equal", "different_input_fields")
LEGEND = {
    "repeat_order": [1, 2, 3], "pair_order": [[1, 2], [1, 3], [2, 3]],
    "client_values": CLIENT_COLUMNS,
    "input_comparison_fields": INPUT_FIELDS,
    "global_inputs": ["data", "model_spec", "initial_model", "partition_metadata"],
    "input_equality": "every declared client and global input is compared exactly across all three repeats; differing global values appear once in the partition common record, local values only on the affected client",
    "samples_and_batches": "batch_count and minibatch_sizes describe one epoch; samples/batches use repeat1 when equal, otherwise all repeat values appear in input_differences",
    "remainder": "samples % batch_size; zero is not an empty final batch: a divisible nonempty client has last_batch_size == batch_size",
    "update_sha256": "full recorded update hashes in repeat_order; a mismatch does not measure numerical distance",
    "loss": "original mean local training loss in repeat_order, with no rounding; equal mean loss does not establish equal gradients",
    "epoch_indices": "reconstructed permutations, not an observation of executed minibatch indices",
    "tail_groups": "all clients with matching batch shapes across repeats are denominators, including identical updates; input equality is reported separately; mixed batch shapes are never assigned to repeat1's group",
    "propagation": "round2/3 are later-round observations; when round1 already differs they may reflect propagation, not independent root-cause evidence",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(values):
    return all(value == values[0] for value in values[1:])


def _global_inputs(payload):
    return {"data": payload["data"], "model_spec": payload["model_spec"],
            "initial_model": payload["initial_model"],
            "partition_metadata": {key: payload["partition"][key] for key in ("strategy", "seed", "alpha")}}


def _client_inputs(client, partition_client):
    return {"model_input": client["model_input"], "partition_indices": partition_client["indices"],
            "epoch_indices": client["epoch_indices"], "training_seed": client["training_seed"],
            "learning_rate": client["learning_rate"], "epochs": client["epochs"],
            "batch_size": client["batch_size"], "samples": client["samples"],
            "minibatch_sizes": client["minibatch_sizes_per_epoch"]}


def _round_analysis(payloads, rd, global_inputs):
    rounds = [payload["rounds"][rd - 1] for payload in payloads]
    ids = [client["client_id"] for client in rounds[0]["clients"]]
    _require(len(ids) == len(set(ids)) and all([c["client_id"] for c in row["clients"]] == ids
        for row in rounds), "incomplete or reordered client coverage")
    partitions = [{client["client_id"]: client for client in payload["partition"]["clients"]} for payload in payloads]
    rows, internal = [], []
    for index, cid in enumerate(ids):
        clients = [row["clients"][index] for row in rounds]
        inputs = [_client_inputs(client, part[cid]) for client, part in zip(clients, partitions)]
        differences = {field: [values[field] for values in inputs] for field in INPUT_FIELDS
                       if not _same([values[field] for values in inputs])}
        different_fields = list(differences) + ["global." + field for field in global_inputs[0]
            if not _same([item[field] for item in global_inputs])]
        first = clients[0]
        sizes = first["minibatch_sizes_per_epoch"]
        row = {"client_id": cid, "samples": first["samples"], "batch_count": len(sizes),
            "remainder": first["samples"] % first["batch_size"],
            "last_batch_size": sizes[-1] if sizes else 0, "minibatch_sizes": sizes,
            "update_sha256": [client["update"]["sha256"] for client in clients],
            "loss": [client["stats"]["loss"] for client in clients],
            "inputs_equal": not different_fields, "different_input_fields": different_fields}
        if differences:
            row["input_differences"] = differences
        rows.append(row)
        internal.append({"client_id": cid, "clients": clients, "inputs": inputs,
                         "shape_equal": all(_same([v[field] for v in inputs])
                                            for field in ("samples", "batch_size", "minibatch_sizes"))})
    pairs = []
    for earlier, later in PAIRS:
        a, b = earlier - 1, later - 1
        global_equal = global_inputs[a] == global_inputs[b]
        pairs.append({"earlier_repeat": earlier, "later_repeat": later, "clients": len(ids),
            "update_differing_ids": [item["client_id"] for item in internal
                if item["clients"][a]["update"] != item["clients"][b]["update"]],
            "loss_differing_ids": [item["client_id"] for item in internal
                if item["clients"][a]["stats"]["loss"] != item["clients"][b]["stats"]["loss"]],
            "input_differing_ids": [item["client_id"] for item in internal
                if not global_equal or item["inputs"][a] != item["inputs"][b]]})
    return rows, pairs, internal


def analyze_partition(partition, payloads):
    """Compare three already-validated observations; no scientific input is changed."""
    _require(len(payloads) == 3, "exactly three repeats per partition are required")
    _require(all(p["configuration"]["partition"] == partition for p in payloads), "partition identity differs")
    globals_ = [_global_inputs(payload) for payload in payloads]
    global_differences = {field: [v[field] for v in globals_] for field in globals_[0]
                          if not _same([v[field] for v in globals_])}
    clients, pairs, internal = _round_analysis(payloads, 1, globals_)
    expected = payloads[0]["configuration"]["num_clients"]
    _require(len(clients) == expected and all(p["configuration"]["num_clients"] == expected for p in payloads),
             "client denominator differs from declared population")
    groups, unmatched = {}, []
    differing = [set(pair["update_differing_ids"]) for pair in pairs]
    for row, evidence in zip(clients, internal):
        cid = row["client_id"]
        if not evidence["shape_equal"]:
            unmatched.append(cid)
            continue
        key = (row["remainder"], row["last_batch_size"])
        group = groups.setdefault(key, {"remainder": key[0], "last_batch_size": key[1], "clients": 0,
            "any_update_different": 0, "pairwise_update_different": [0, 0, 0]})
        group["clients"] += 1
        group["any_update_different"] += any(cid in values for values in differing)
        for index, values in enumerate(differing):
            group["pairwise_update_different"][index] += cid in values
    propagation = []
    for rd in (2, 3):
        _, later_pairs, _ = _round_analysis(payloads, rd, globals_)
        rounds = [payload["rounds"][rd - 1] for payload in payloads]
        propagation.append({"round": rd, "role": "later_propagation_observations", "pairs": later_pairs,
            "aggregate_sha256": [row["aggregate"]["sha256"] for row in rounds],
            "post_model_sha256": [row["post_model"]["sha256"] for row in rounds],
            "accuracy": [row["record"]["accuracy"] for row in rounds]})
    config = payloads[0]["configuration"]
    common = {"round": 1, "repeat_order": [1, 2, 3], "clients": len(clients),
        "seed": config["seed"], "round1_training_seed": "seed + 1009 + numeric client index",
        "learning_rate": config["lr"], "epochs": config["local_epochs"], "batch_size": config["batch_size"],
        "global_input_equal": not global_differences, "global_different_input_fields": list(global_differences),
        "global_input_differences": global_differences,
        "all_client_inputs_equal": all(row["inputs_equal"] for row in clients),
        "any_update_different_clients": len(set().union(*differing)),
        "tail_group_denominator": sum(group["clients"] for group in groups.values())}
    return {"partition": partition, "common": common, "clients": clients, "pairs": pairs,
            "tail_groups": [groups[key] for key in sorted(groups)], "unmatched_batch_shape_ids": unmatched,
            "propagation": propagation}


def _reader_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _snapshot(output):
    return {"probe_evidence": report.evidence_hashes(output), "sources": protocol.source_hashes(),
            "reader_sha256": _reader_sha256()}


def summarize(output):
    """Return one complete analysis or a copyable failure with no partial details."""
    output = Path(output).resolve()
    reviewed = None
    try:
        before = _snapshot(output)
        reviewed = report.summarize(output)
        _require(reviewed.get("status") == "complete" and reviewed.get("complete_tasks") == 6
            and reviewed.get("available_pairs") == 4 and reviewed.get("source_and_reference_verified") is True
            and reviewed.get("input_evidence_unchanged") is True, "all six valid prefix artifacts and their complete original audit are required")
        manifest, tasks = protocol.read_study(output, current_sources=True)
        _require(manifest["fingerprint"] == reviewed.get("manifest_fingerprint"), "manifest changed since original audit")
        _require(before["sources"] == manifest["source_sha256"], "source identity changed before artifact read")
        _require(reviewed.get("recorded_source_map_sha256") == protocol.base.digest(manifest["source_sha256"]),
                 "source map differs from original audit")
        task_map = {(t["partition"], t["repeat"]): t for t in tasks}
        _require(len(tasks) == len(task_map) == 6 and set(task_map) == {(p, r) for p in ("iid", "dirichlet") for r in (1, 2, 3)},
                 "probe task matrix is incomplete or duplicated")
        reviewed_rows = {row["task_id"]: row for row in reviewed["rows"]}
        _require(len(reviewed_rows) == 6, "original audit task identities are incomplete")
        payloads, identities = {}, []
        for task in tasks:
            artifact = protocol.load_completed(output, task)
            _require(artifact is not None, "missing completed artifact: " + task["task_id"])
            old = reviewed_rows.get(task["task_id"], {})
            _require(old.get("status") == "complete" and old.get("task_fingerprint") == task["fingerprint"]
                and old.get("artifact_fingerprint") == artifact["artifact_fingerprint"], "completed artifact changed since original audit: " + task["task_id"])
            payloads[(task["partition"], task["repeat"])] = report.validate_observations(artifact["observations"], task)
            identities.append({"task_id": task["task_id"], "task_fingerprint": task["fingerprint"],
                               "artifact_fingerprint": artifact["artifact_fingerprint"]})
        partitions = [analyze_partition(p, [payloads[(p, repeat)] for repeat in (1, 2, 3)]) for p in ("iid", "dirichlet")]
        # Reference verification hashes the exact immutable old 90-task evidence
        # map; it does not load datasets, checkpoints or train any model.
        protocol.verify_reference(manifest["reference"])
        _require(before == _snapshot(output), "source, reader or probe evidence changed during this read")
        value = {"status": "complete", "schema": SCHEMA, "output": str(output),
            "manifest_fingerprint": manifest["fingerprint"], "source_count": len(manifest["source_sha256"]),
            "source_map_sha256": protocol.base.digest(manifest["source_sha256"]),
            "reader_sha256": before["reader_sha256"], "reader_in_frozen_source_map": False,
            "reference_evidence_map_sha256": protocol.base.digest(manifest["reference"]["evidence_sha256"]),
            "probe_evidence_map_sha256": protocol.base.digest(before["probe_evidence"]),
            "same_gpu_uuid": manifest["same_gpu_uuid"], "complete_tasks": 6, "tasks": identities,
            "source_reference_and_input_evidence_verified": True, "input_evidence_unchanged": True,
            "protection_scope": "probe manifest/plan/task/completion files, frozen sources and this reader hashed before the original summary and after extraction; old90 reference evidence verified by the original summary, read_study and final verify_reference",
            "numerical_environment": reviewed.get("numerical_environment"), "partitions": partitions,
            "decision": {"action": "review_round1_client_boundaries_before_any_new_probe",
                "training_started": False, "health_assessed": False, "formal_qualification_assessed": False,
                "automatic_next_stage": False, "cuda_cause_identified": False, "root_cause_identified": False,
                "limitation": "exact recorded hash and scalar comparisons only; batch-shape association is descriptive, not causal; no numerical update-distance or actual executed sample-order evidence is created"}}
        json.dumps(value, allow_nan=False)
        return value
    except Exception as exc:
        invalid = {"status": "invalid", "schema": SCHEMA, "output": str(output), "error": str(exc),
                   "partial_details_emitted": False, "training_started": False, "automatic_next_stage": False}
        if isinstance(reviewed, dict):
            invalid["original_audit"] = {key: reviewed[key] for key in
                ("status", "error", "complete_tasks", "expected_tasks") if key in reviewed}
            invalid["original_audit"]["problem_tasks"] = [
                {key: row[key] for key in ("task_id", "status", "error") if key in row}
                for row in reviewed.get("rows", []) if row.get("status") != "complete"]
        return invalid


def print_summary(value):
    records = [{"type": "header", **{k: v for k, v in value.items() if k not in ("partitions", "decision")}}]
    if value["status"] == "complete":
        records.append({"type": "legend", **LEGEND})
        for partition in value["partitions"]:
            name = partition["partition"]
            records.append({"type": "partition", "partition": name, **partition["common"]})
            for row in partition["clients"]:
                record = {"type": "client", "partition": name, "values": [row[key] for key in CLIENT_COLUMNS]}
                if row.get("input_differences"):
                    record["input_differences"] = row["input_differences"]
                records.append(record)
            records.extend({"type": "pair", "partition": name, "round": 1, **pair} for pair in partition["pairs"])
            records.append({"type": "tail_groups", "partition": name, "groups": partition["tail_groups"],
                            "unmatched_batch_shape_ids": partition["unmatched_batch_shape_ids"]})
            records.extend({"type": "propagation", "partition": name, **row} for row in partition["propagation"])
        records.append({"type": "decision", **value["decision"]})
    try:
        lines = [json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) for row in records]
    except Exception as exc:
        lines = [json.dumps({"type": "header", "status": "invalid", "error": str(exc),
                             "partial_details_emitted": False, "training_started": False})]
    print("=== CIFAR_PREFIX_CLIENTS_BEGIN ===", flush=True)
    for line in lines:
        print(line)
    print("=== CIFAR_PREFIX_CLIENTS_END ===", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=protocol.DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = summarize(args.output)
    print_summary(result)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
