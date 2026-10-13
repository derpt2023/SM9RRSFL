"""Read-only local-step comparisons; saved arrays alone support magnitudes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import cifar_prefix_probe_report as prefix_report
from cifar_prefix_probe_runtime import tensor_fingerprint
import run_cifar_matched_cnn as matched

SCHEMA = "cifar-client-step-probe-observation-v1"
PAIRS = ((1, 2), (1, 3), (2, 3))
STAGES = ("local_indices", "dataset_indices", "features", "pre_parameters",
          "logits", "labels", "loss", "gradients", "post_parameters")
SNAPSHOT_STAGES = STAGES + ("final_delta",)
PARAMETER_STAGES = ("pre_parameters", "gradients", "post_parameters", "final_delta")
INPUT_FIELDS = ("samples", "model_input", "training_seed", "learning_rate", "epochs",
                "batch_size", "epoch_indices", "minibatch_sizes_per_epoch")
METRIC_FIELDS = ("max_abs", "relative_l2", "numerically_unequal_fraction", "reference_l2", "elements")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fp(value, name):
    prefix_report._tensor(value, name)


def _finite(value):
    return prefix_report.timing.finite(value)


def _batch_fp(batch, stage):
    return batch["loss"]["tensor"] if stage == "loss" else batch[stage]


def validate_observations(payload, task, arrays):
    """Verify complete scheduled local observations without inventing a full round."""
    _require(isinstance(payload, dict) and payload.get("schema") == SCHEMA, "wrong local-step observation schema")
    _require(payload.get("task_id") == task["task_id"] and payload.get("task_fingerprint") == task["fingerprint"],
             "step observation task identity differs")
    _require(prefix_report._finite_tree(payload), "nonfinite local-step observation")
    config = task["config"]
    observed_config = payload.get("configuration", {})
    _require({k: v for k, v in observed_config.items() if k != "device"} ==
             {k: v for k, v in config.items() if k != "device"}, "observed configuration differs")
    target_indices, last = task["target_clients"], task["stop_after_client"]
    _require(payload.get("target_client_indices") == target_indices and payload.get("stop_after_client") == last,
             "target or stopping identity differs")
    stop = payload.get("stop", {})
    _require(stop.get("round") == 1 and stop.get("after_client") == "client-" + str(last)
        and stop.get("before_aggregation") is True and stop.get("completed_rounds") == 0
        and stop.get("crypto_finalized") is False and stop.get("next_client_not_trained") == "client-" + str(last + 1),
        "scheduled pre-aggregation stop is incomplete")
    prefix = payload.get("prefix", {})
    _require(prefix.get("task_id") == task["task_id"] and prefix.get("task_fingerprint") == task["fingerprint"]
        and prefix.get("configuration") == observed_config, "partial prefix identity or configuration differs")
    for field in prefix_report.DATA_FIELDS:
        value = prefix.get("data", {}).get(field)
        if field not in ("x_attack", "y_attack") or value is not None:
            _fp(value, "prefix.data." + field)
    _require((prefix["data"].get("x_attack") is None) == (prefix["data"].get("y_attack") is None), "unpaired attack data")
    _fp(prefix.get("initial_model"), "prefix.initial_model")
    spec = prefix.get("model_spec", {})
    _require(isinstance(spec, dict) and isinstance(spec.get("num_classes"), int), "missing model specification")
    partition = prefix.get("partition", {})
    _require(partition.get("strategy") == config["partition"] and partition.get("seed") == config["seed"]
        and partition.get("alpha") == config["dirichlet_alpha"], "partition configuration differs")
    partitions = partition.get("clients", [])
    _require([p.get("client_id") for p in partitions] == ["client-" + str(i) for i in range(config["num_clients"])],
             "partition client coverage differs")
    for part in partitions:
        _fp(part.get("indices"), "partition.indices")
        _require(type(part.get("samples")) is int and part["samples"] >= 0
                 and part["indices"]["shape"] == [part["samples"]], "partition sample coverage differs")
    rounds = prefix.get("rounds", [])
    _require(len(rounds) == 1 and rounds[0].get("round") == 1, "exactly one partial round is required")
    clients = rounds[0].get("clients", [])
    ids = ["client-" + str(i) for i in range(last + 1)]
    _require([c.get("client_id") for c in clients] == ids, "missing or reordered observed prefix clients")
    _require(not any(k in rounds[0] for k in ("aggregate", "post_model", "coefficients", "record")),
             "partial prefix must not contain an aggregated round")
    for index, client in enumerate(clients):
        for key in ("model_input", "update"):
            _fp(client.get(key), "prefix.client." + key)
        _require(client["model_input"] == prefix["initial_model"], "client input differs from round-zero model")
        _require(client["update"]["shape"] == prefix["initial_model"]["shape"]
            and client["update"]["dtype"] == prefix["initial_model"]["dtype"], "client update layout differs")
        _require(client["samples"] == partitions[index]["samples"] and client["stats"]["samples"] == client["samples"]
            and _finite(client["stats"].get("loss")), "invalid client training statistics")
        _require(client.get("training_seed") == config["seed"] + 1009 + index
            and client.get("learning_rate") == config["lr"] and client.get("epochs") == config["local_epochs"]
            and client.get("batch_size") == config["batch_size"], "local training schedule differs")
        expected_sizes = [min(config["batch_size"], client["samples"] - n)
                          for n in range(0, client["samples"], config["batch_size"])]
        _require(client.get("minibatch_sizes_per_epoch") == expected_sizes, "prefix minibatch coverage differs")
        _require(len(client.get("epoch_indices", [])) == config["local_epochs"], "missing reconstructed epoch order")
        for value in client["epoch_indices"]:
            _fp(value, "prefix.epoch_indices")
            _require(value["shape"] == [client["samples"]], "reconstructed order shape differs")
    checkpoints = prefix.get("checkpoints", [])
    _require(len(checkpoints) == 1 and checkpoints[0].get("round") == 0
        and checkpoints[0].get("model") == prefix["initial_model"], "round-zero checkpoint observation is missing")
    prefix_report._record(checkpoints[0].get("record"), 0)
    _require(checkpoints[0].get("diagnostic_count") == 0 and all(checkpoints[0]["record"][field] == 0 for field in
        ("false_positive_revocations", "true_positive_revocations", "accepted_updates", "rejected_updates", "blacklisted_clients", "nonfinite_updates")),
        "round-zero observation contains unexpected client events")
    evaluations = prefix.get("evaluations", [])
    _require(len(evaluations) == 2 and {(e.get("round"), e.get("kind")) for e in evaluations}
             == {(0, "accuracy"), (0, "target")}, "round-zero evaluations are missing")
    for evaluation in evaluations:
        _require(evaluation.get("model_input") == prefix["initial_model"], "round-zero evaluation model differs")
        record = checkpoints[0]["record"]
        expected = record["accuracy"] if evaluation["kind"] == "accuracy" else [record[k] for k in
            ("attack_target_success_rate", "attack_target_confidence")]
        _require(evaluation.get("value") == expected, "round-zero evaluation values disagree")
        batches = evaluation.get("batches", [])
        _require(batches and [b.get("batch") for b in batches] == list(range(len(batches))), "missing round-zero evaluation batches")
        for batch in batches:
            _fp(batch.get("logits"), "round-zero logits")
            _fp(batch.get("predictions"), "round-zero predictions")
            shape = batch["logits"]["shape"]
            _require(len(shape) == 2 and shape[0] > 0 and shape[1] == spec["num_classes"]
                and batch["predictions"]["shape"] == [shape[0]], "round-zero evaluation shapes differ")
        expected_samples = prefix["data"]["y_test"]["shape"][0] if evaluation["kind"] == "accuracy" else config["attack_target_count"]
        _require(sum(b["logits"]["shape"][0] for b in batches) == expected_samples, "round-zero evaluation sample coverage differs")
    targets = payload.get("targets", [])
    _require([t.get("client_id") for t in targets] == ["client-" + str(i) for i in target_indices], "target coverage differs")
    snapshot_refs = {}
    for index, target in zip(target_indices, targets):
        client = clients[index]
        for field in ("samples", "model_input", "training_seed", "learning_rate", "epochs", "batch_size", "stats"):
            _require(target.get(field) == client.get(field), "target/prefix mismatch: " + field)
        _require(target.get("final_delta") == client["update"], "target final delta differs from original candidate CPU copy")
        layout = target.get("parameter_layout", [])
        offset, names = 0, set()
        for parameter in layout:
            shape = parameter.get("shape")
            _require(isinstance(shape, list) and all(type(n) is int and n > 0 for n in shape), "invalid parameter shape")
            size = int(np.prod(shape, dtype=np.int64))
            _require(isinstance(parameter.get("name"), str) and parameter["name"] not in names
                and parameter.get("offset") == offset and parameter.get("size") == size, "invalid parameter layout")
            offset += size
            names.add(parameter["name"])
        _require(layout and prefix["initial_model"]["shape"] == [offset], "parameter layout does not cover the model")
        sizes = client["minibatch_sizes_per_epoch"]
        batches = target.get("batches", [])
        _require(sizes and len(batches) == len(sizes) * target["epochs"], "missing target minibatches")
        previous = target["model_input"]
        for batch_index, batch in enumerate(batches):
            n = sizes[batch_index % len(sizes)]
            _require(batch.get("batch") == batch_index and batch.get("epoch") == batch_index // len(sizes)
                and batch.get("batch_in_epoch") == batch_index % len(sizes) and batch.get("samples") == n,
                "target minibatch position or coverage differs")
            for stage in STAGES:
                _fp(_batch_fp(batch, stage), "batch." + stage)
            for stage in ("local_indices", "dataset_indices", "labels"):
                _require(batch[stage]["shape"] == [n], "actual batch index/label shape differs")
            _require(batch["features"]["shape"][0] == n and batch["logits"]["shape"] == [n, spec["num_classes"]],
                     "actual batch feature/logit shape differs")
            _require(batch["loss"]["tensor"]["shape"] == [] and _finite(batch["loss"].get("value")), "invalid actual batch loss")
            for stage in ("pre_parameters", "gradients", "post_parameters"):
                _require(batch[stage]["shape"] == [offset], "parameter/gradient vector shape differs")
            _require(batch["pre_parameters"] == previous, "observed SGD parameter chain is broken")
            previous = batch["post_parameters"]
        _require(set(target.get("tail_snapshots", {})) == set(SNAPSHOT_STAGES), "tail snapshots are incomplete")
        for stage, key in target["tail_snapshots"].items():
            _require(isinstance(key, str) and key not in snapshot_refs, "duplicate or invalid snapshot key")
            expected = target["final_delta"] if stage == "final_delta" else _batch_fp(batches[-1], stage)
            snapshot_refs[key] = expected
    _require(isinstance(arrays, dict) and set(arrays) == set(payload.get("snapshots", {})) == set(snapshot_refs),
             "snapshot keyspace differs from actual observations")
    for key, array in arrays.items():
        _require(isinstance(array, np.ndarray) and array.dtype.kind in "fiub" and np.isfinite(array).all(),
                 "invalid or nonfinite snapshot array: " + key)
        actual = tensor_fingerprint(array)
        _require(actual == payload["snapshots"][key] == snapshot_refs[key], "snapshot content fingerprint differs: " + key)
    for target in targets:
        loss = arrays[target["tail_snapshots"]["loss"]]
        _require(float(loss) == target["batches"][-1]["loss"]["value"], "tail loss value disagrees with saved scalar")
    return payload


def magnitude(later, earlier):
    """Numerical distances on retained original arrays, never on hashes."""
    _require(later.shape == earlier.shape and later.dtype == earlier.dtype, "snapshot array layouts differ")
    a, b = later.astype(np.float64), earlier.astype(np.float64)
    delta = a - b
    distance = float(np.linalg.norm(delta.reshape(-1)))
    reference = float(np.linalg.norm(b.reshape(-1)))
    return {"max_abs": float(np.max(np.abs(delta))) if delta.size else 0.,
        "relative_l2": distance / reference if reference else (0. if not distance else None),
        "numerically_unequal_fraction": float(np.count_nonzero(a != b) / delta.size) if delta.size else 0.,
        "reference_l2": reference, "elements": int(delta.size)}


def historical_context(payload, anchors):
    prefix = payload["prefix"]
    mismatches, targets = [], {t["client_id"] for t in payload["targets"]}
    _require(len(anchors) == 3, "three historical Dirichlet observations are required")
    old = [a["observations"] for a in anchors]
    for field in ("data", "model_spec", "initial_model", "partition"):
        if any(prefix[field] != value[field] for value in old):
            mismatches.append(field)
    if any(prefix["evaluations"] != [e for e in value["evaluations"] if e["round"] == 0] for value in old):
        mismatches.append("round0_evaluations")
    if any(prefix["checkpoints"][0]["record"] != value["checkpoints"][0]["record"] for value in old):
        mismatches.append("round0_record")
    for index, client in enumerate(prefix["rounds"][0]["clients"]):
        references = [value["rounds"][0]["clients"][index] for value in old]
        if any(any(client[field] != ref[field] for field in INPUT_FIELDS) for ref in references):
            mismatches.append(client["client_id"] + ".inputs")
        if client["client_id"] not in targets and any(client["update"] != ref["update"] for ref in references):
            mismatches.append(client["client_id"] + ".non_target_update")
    matches = {target["client_id"]: [target["final_delta"] == value["rounds"][0]["clients"][int(target["client_id"].split("-")[-1])]["update"]
        for value in old] for target in payload["targets"]}
    return {"historical_context_reproduced": not mismatches, "context_mismatches": mismatches,
            "target_delta_matches_historical_repeats": matches,
            "scope": "original data/partition/model, round0 evaluation, recorded client inputs and stable non-target updates; target updates are allowed to differ"}


def compare_observations(later, earlier, later_arrays, earlier_arrays):
    differences = []
    def compare(stage, a, b, *, client=None, batch=None, target=None):
        if a == b:
            return
        item = {"stage": stage, "client_id": client, "batch": batch,
                "first_field": next(prefix_report._leaf_differences(a, b))}
        if target is not None and (stage == "final_delta" or batch == len(target["batches"]) - 1):
            key = target["tail_snapshots"][stage]
            old_target = next(t for t in earlier["targets"] if t["client_id"] == client)
            item["magnitude"] = magnitude(later_arrays[key], earlier_arrays[old_target["tail_snapshots"][stage]])
            item["magnitude_scope"] = "saved final delta" if stage == "final_delta" else "saved last minibatch"
        else:
            item["magnitude"] = None
            item["magnitude_scope"] = "no full snapshot at this observed boundary"
        differences.append(item)
    left, right = later["prefix"], earlier["prefix"]
    for field in ("data", "partition", "model_spec", "initial_model", "evaluations"):
        compare("prefix." + field, left[field], right[field])
    target_map = {t["client_id"]: t for t in later["targets"]}
    earlier_targets = {t["client_id"]: t for t in earlier["targets"]}
    for a, b in zip(left["rounds"][0]["clients"], right["rounds"][0]["clients"]):
        cid = a["client_id"]
        compare("client_inputs", {k: a[k] for k in INPUT_FIELDS}, {k: b[k] for k in INPUT_FIELDS}, client=cid)
        if cid in target_map:
            target, old_target = target_map[cid], earlier_targets[cid]
            _require(target["parameter_layout"] == old_target["parameter_layout"], "paired parameter layouts differ")
            _require(len(target["batches"]) == len(old_target["batches"]), "paired minibatch counts differ")
            for batch_index, (new, old) in enumerate(zip(target["batches"], old_target["batches"])):
                for stage in STAGES:
                    compare(stage, new[stage], old[stage], client=cid, batch=batch_index, target=target)
            compare("final_delta", target["final_delta"], old_target["final_delta"], client=cid, target=target)
        else:
            compare("non_target_update", a["update"], b["update"], client=cid)
        compare("local_training_statistics", a["stats"], b["stats"], client=cid)
    tails = []
    for target in later["targets"]:
        cid = target["client_id"]
        old_target = earlier_targets[cid]
        metrics = {stage: magnitude(later_arrays[target["tail_snapshots"][stage]],
                                   earlier_arrays[old_target["tail_snapshots"][stage]]) for stage in SNAPSHOT_STAGES}
        per_parameter = []
        for param in target["parameter_layout"]:
            start, end = param["offset"], param["offset"] + param["size"]
            per_parameter.append({"name": param["name"], **{stage: magnitude(
                later_arrays[target["tail_snapshots"][stage]][start:end],
                earlier_arrays[old_target["tail_snapshots"][stage]][start:end]) for stage in PARAMETER_STAGES}})
        tails.append({"client_id": cid, "last_batch": len(target["batches"]) - 1,
                      "samples": target["batches"][-1]["samples"], "metrics": metrics, "per_parameter": per_parameter})
    counts = {}
    for difference in differences:
        counts[difference["stage"]] = counts.get(difference["stage"], 0) + 1
    return {"available": True, "all_recorded_equal": not differences,
            "first_divergence": differences[0] if differences else None,
            "different_boundaries_by_stage": counts, "tail_comparisons": tails}


def summarize(output):
    import cifar_client_step_probe_protocol as protocol
    output = Path(output).resolve()
    def snapshot():
        return {"evidence": protocol.evidence_hashes(output), "sources": protocol.source_hashes()}
    try:
        before = snapshot()
        manifest, tasks = protocol.read_study(output, current_sources=True)
        rows, observations, arrays_by_id = [], {}, {}
        for task in tasks:
            row = {"task_id": task["task_id"], "repeat": task["repeat"], "task_fingerprint": task["fingerprint"], "status": "missing"}
            try:
                artifact = protocol.load_completed(output, task)
                if artifact is not None:
                    _require(artifact.get("status") == "complete" and artifact.get("task_fingerprint") == task["fingerprint"], "artifact identity differs")
                    _require(artifact.get("fresh_start") is True and artifact.get("checkpoints_used") is False, "artifact is not a fresh local probe")
                    _require(artifact.get("gpu_uuid") == manifest["same_gpu_uuid"], "physical GPU identity differs")
                    _require(_finite(artifact.get("wall_seconds")) and artifact["wall_seconds"] >= 0, "invalid observed worker wall time")
                    _require(matched.normalized_environment(artifact["environment"]) == manifest["reference"]["execution_environment"], "numerical environment differs")
                    arrays = protocol.load_arrays(output, task, artifact)
                    payload = validate_observations(artifact["observations"], task, arrays)
                    context = historical_context(payload, manifest["reference"]["anchors"])
                    row.update(status="complete", artifact_fingerprint=artifact["artifact_fingerprint"], wall_seconds=artifact["wall_seconds"], **context,
                        observed_clients=len(payload["prefix"]["rounds"][0]["clients"]), completed_rounds=0,
                        targets=[{"client_id": t["client_id"], "samples": t["samples"], "batches": len(t["batches"]),
                            "last_batch_size": t["batches"][-1]["samples"], "mean_loss": t["stats"]["loss"],
                            "final_delta_sha256": t["final_delta"]["sha256"]} for t in payload["targets"]])
                    observations[task["task_id"]], arrays_by_id[task["task_id"]] = payload, arrays
            except Exception as exc:
                row.update(status="invalid_evidence", error=str(exc))
            rows.append(row)
        _require(len(tasks) == 3 and {t["repeat"] for t in tasks} == {1, 2, 3} and len({t["task_id"] for t in tasks}) == 3, "repeat matrix differs")
        by_repeat = {t["repeat"]: t for t in tasks}
        pairs = []
        for earlier, later in PAIRS:
            a, b = by_repeat[later]["task_id"], by_repeat[earlier]["task_id"]
            pair = {"later_repeat": later, "earlier_repeat": earlier, "available": False,
                    "all_recorded_equal": None, "first_divergence": None}
            if a in observations and b in observations:
                try:
                    pair.update(compare_observations(observations[a], observations[b], arrays_by_id[a], arrays_by_id[b]))
                except Exception as exc:
                    pair["error"] = str(exc)
            pairs.append(pair)
        protocol.verify_reference(manifest["reference"])
        _require(before == snapshot(), "step evidence or scientific sources changed during report read")
        complete = sum(row["status"] == "complete" for row in rows)
        ready = complete == 3 and all(pair["available"] for pair in pairs)
        return {"status": "complete" if ready else "incomplete_or_invalid_evidence",
            "manifest_fingerprint": manifest["fingerprint"], "same_gpu_uuid": manifest["same_gpu_uuid"],
            "source_count": len(manifest["source_sha256"]), "source_map_sha256": prefix_report.timing.base.digest(manifest["source_sha256"]),
            "source_reference_and_evidence_verified": True, "expected_tasks": 3, "complete_tasks": complete,
            "rows": rows, "pairs": pairs, "all_pairs_equal": all(p["all_recorded_equal"] for p in pairs) if ready else None,
            "all_historical_contexts_reproduced": all(row.get("historical_context_reproduced") is True for row in rows) if ready else None,
            "decision": {"action": "review_local_step_boundaries" if ready else "resolve_incomplete_or_invalid_step_evidence",
                "health_assessed": False, "formal_qualification_assessed": False, "automatic_next_stage": False,
                "cuda_cause_identified": False, "root_cause_identified": False, "completed_training_rounds": 0,
                "limitation": "first observed difference is not an identified operator cause; matching instrumented replays does not explain old differences; magnitude exists only for retained last-minibatch/final-delta arrays"}}
    except Exception as exc:
        return {"status": "invalid", "error": str(exc), "training_started": False, "all_pairs_equal": None}


def print_summary(value):
    records = [{"type": "header", **{k: v for k, v in value.items() if k not in ("rows", "pairs", "decision")}},
        {"type": "legend", "metric_order": METRIC_FIELDS, "parameter_stage_order": PARAMETER_STAGES,
         "direction": "later repeat minus earlier repeat; relative L2 uses the earlier norm, null if that norm is zero and distance is nonzero",
         "scope": "last-minibatch arrays and final deltas only; actual in-call indices; hashes are full dtype/shape/byte identities; numerical unequal fraction can be zero for a bitwise-only difference"}]
    records.extend({"type": "task", **row} for row in value.get("rows", []))
    for pair in value.get("pairs", []):
        compact = {k: v for k, v in pair.items() if k != "tail_comparisons"}
        compact["tail_comparisons"] = [{**{k: v for k, v in tail.items() if k not in ("metrics", "per_parameter")},
            "metrics": {stage: [metrics[k] for k in METRIC_FIELDS] for stage, metrics in tail["metrics"].items()},
            "per_parameter": [[param["name"], *[[param[stage][k] for k in METRIC_FIELDS] for stage in PARAMETER_STAGES]]
                for param in tail["per_parameter"]]} for tail in pair.get("tail_comparisons", [])]
        records.append({"type": "pair", **compact})
    if "decision" in value:
        records.append({"type": "decision", **value["decision"]})
    lines = [json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) for row in records]
    print("=== CIFAR_CLIENT_STEP_PROBE_BEGIN ===", flush=True)
    for line in lines:
        print(line)
    print("=== CIFAR_CLIENT_STEP_PROBE_END ===", flush=True)
