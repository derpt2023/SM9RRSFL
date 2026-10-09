"""Read-only comparisons of six fresh three-round reproducibility probes."""
import json
from dataclasses import fields
import hashlib
from pathlib import Path

import cifar_timing_report as timing
import run_cifar_matched_cnn as matched
import summarize_cifar_timing as brief

SCHEMA = "cifar-prefix-probe-observation-v1"
DATA_FIELDS = ("x_train", "y_train", "x_test", "y_test", "x_attack", "y_attack")
PAIR_FIELDS = brief.PAIR_FIELDS


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return type(value) is int and value >= 0


def _tensor(value, name):
    _require(isinstance(value, dict), "missing tensor fingerprint: " + name)
    _require(set(value) == {"sha256", "dtype", "shape"}, "incomplete tensor fingerprint: " + name)
    sha = value["sha256"]
    _require(isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "invalid tensor SHA: " + name)
    _require(isinstance(value["dtype"], str) and bool(value["dtype"]), "invalid tensor dtype: " + name)
    _require(isinstance(value["shape"], list) and all(_integer(n) for n in value["shape"]), "invalid tensor shape: " + name)


def _finite_tree(value):
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(v) for v in value)
    return not isinstance(value, (int, float)) or isinstance(value, bool) or timing.finite(value)


def _record(record, rd):
    _require(isinstance(record, dict) and record.get("round") == rd, "missing or wrong round record")
    _require(all(field in record for field in PAIR_FIELDS), "incomplete scientific round record")
    _require(_finite_tree(record), "nonfinite scientific round record")
    for field in ("accuracy", "error", "attack_target_success_rate", "attack_target_confidence"):
        _require(timing.finite(record[field]) and 0 <= record[field] <= 1, "missing or invalid round metric: " + field)
    for field in ("false_positive_revocations", "true_positive_revocations", "accepted_updates", "rejected_updates", "blacklisted_clients", "nonfinite_updates"):
        _require(_integer(record[field]), "missing or invalid round counter: " + field)


def validate_observations(payload, task):
    """Reject absent stages before comparison; actual hashes are never synthesized."""
    _require(isinstance(payload, dict) and payload.get("schema") == SCHEMA, "wrong observation schema")
    _require(payload.get("task_id") == task["task_id"] and payload.get("task_fingerprint") == task["fingerprint"], "observation task identity differs")
    _require(_finite_tree(payload), "nonfinite probe observation")
    _require(isinstance(payload.get("observation_contract"), (dict, str)), "missing observation contract")
    data = payload.get("data", {})
    _require(all(key in data for key in DATA_FIELDS), "incomplete data fingerprints")
    for key in DATA_FIELDS:
        if key not in ("x_attack", "y_attack") or data[key] is not None:
            _tensor(data[key], "data." + key)
    _require((data["x_attack"] is None) == (data["y_attack"] is None), "attack data fingerprints must be both present or both absent")
    _require(isinstance(payload.get("model_spec"), dict) and bool(payload["model_spec"]), "missing model specification")
    _tensor(payload.get("initial_model"), "initial_model")
    config = task["config"]
    _require(isinstance(payload.get("configuration"), dict) and
        {k: v for k, v in payload["configuration"].items() if k != "device"} ==
        {k: v for k, v in config.items() if k != "device"}, "observed scientific configuration differs")
    partition = payload.get("partition", {})
    _require(partition.get("strategy") == config["partition"] and partition.get("seed") == config["seed"]
        and partition.get("alpha") == config["dirichlet_alpha"], "partition scientific identity differs")
    clients = partition.get("clients", [])
    ids = ["client-" + str(i) for i in range(config["num_clients"])]
    _require([c.get("client_id") for c in clients] == ids, "partition client coverage/order differs")
    for c in clients:
        _require(_integer(c.get("samples")), "invalid partition sample count")
        _tensor(c.get("indices"), "partition.indices")
        _require(c["indices"]["shape"] == [c["samples"]], "partition index length differs from sample count")
    _require(config["rounds"] == 3 and config["malicious_ratio"] == 0, "probe must be exactly three clean rounds")
    rounds = payload.get("rounds", [])
    _require([r.get("round") for r in rounds] == [1, 2, 3], "missing or duplicated training rounds")
    for r in rounds:
        rd = r["round"]
        _require([c.get("client_id") for c in r.get("clients", [])] == ids, "missing or reordered client training observations")
        for index, c in enumerate(r["clients"]):
            for key in ("model_input", "update"):
                _tensor(c.get(key), "client." + key)
            _require(c.get("samples") == clients[index]["samples"], "client sample count changed")
            stats = c.get("stats", {})
            _require(timing.finite(stats.get("loss")) and stats.get("samples") == c["samples"], "missing or invalid local training statistics")
            _require(c.get("training_seed") == config["seed"] + rd * 1009 + index, "training seed schedule differs")
            _require(c.get("epochs") == config["local_epochs"] and c.get("batch_size") == config["batch_size"], "local training parameters differ")
            _require(c.get("learning_rate") == config["lr"] * config["lr_decay"] ** (rd - 1), "learning rate schedule differs")
            epochs = c.get("epoch_indices", [])
            _require(len(epochs) == config["local_epochs"], "missing reconstructed sample-order fingerprints")
            for epoch in epochs:
                _tensor(epoch, "client.epoch_indices")
                _require(epoch["shape"] == [c["samples"]], "sample-order fingerprint has wrong shape")
            sizes = [min(config["batch_size"], c["samples"] - start) for start in range(0, c["samples"], config["batch_size"])]
            _require(c.get("minibatch_sizes_per_epoch") == sizes, "missing or invalid reconstructed minibatch sizes")
        for name in ("candidate_order", "verified_order", "aggregate_order"):
            _require(isinstance(r.get(name), list) and len(r[name]) == len(ids) and set(r[name]) == set(ids), "missing or invalid " + name)
        coefficients = r.get("coefficients", {})
        order = coefficients.get("order", [])
        _require(len(order) == len(ids) and set(order) == set(ids) and set(coefficients.get("by_client", {})) == set(ids), "coefficient client identity/coverage differs")
        _require(all(timing.finite(v) and 0 <= v <= 1 + 1e-9 for v in coefficients["by_client"].values()), "invalid aggregation coefficient")
        _tensor(r.get("aggregate"), "aggregate")
        _tensor(r.get("post_model"), "post_model")
        _record(r.get("record"), rd)
        diagnostics = r.get("diagnostics", [])
        _require([d.get("client_id") for d in diagnostics] == ids, "missing or reordered client diagnostics")
        _require(all(d.get("round") == rd and "task_tag" not in d for d in diagnostics), "diagnostic round differs or contains task tag")
        required = {f.name for f in fields(timing.base.fl.ClientDiagnosticRecord)} - {"task_tag"}
        flags = ("is_malicious", "suspicious", "count_increment", "trace_requested", "trace_pending", "revoked",
                 "aggregation_accepted", "history_eligible", "history_admitted", "history_frozen", "immediate_revocation", "attack_active", "recovery_eligible")
        for d in diagnostics:
            _require(required <= set(d), "incomplete scientific client diagnostic")
            _require(all(type(d[field]) is bool for field in flags), "invalid diagnostic boolean")
            _require(d["aggregation_weight"] == coefficients["by_client"][d["client_id"]], "diagnostic coefficient disagrees with actual aggregation")
            _require(d["decision_reason"] == "clean_warmup" and not any(d[field] for field in
                ("is_malicious", "attack_active", "suspicious", "trace_requested", "trace_pending", "revoked")), "unexpected non-warmup event in clean prefix")
    evaluations = payload.get("evaluations", [])
    keys = [(e.get("round"), e.get("kind")) for e in evaluations]
    _require(len(keys) == 8 and set(keys) == {(rd, kind) for rd in range(4) for kind in ("accuracy", "target")}, "missing or duplicated evaluations")
    for e in evaluations:
        _tensor(e.get("model_input"), "evaluation.model_input")
        batches = e.get("batches", [])
        _require(bool(batches) and [b.get("batch") for b in batches] == list(range(len(batches))), "missing or reordered evaluation batches")
        for b in batches:
            _tensor(b.get("logits"), "evaluation.logits")
            _tensor(b.get("predictions"), "evaluation.predictions")
            shape = b["logits"]["shape"]
            _require(len(shape) == 2 and shape[1] == payload["model_spec"].get("num_classes")
                and shape[0] > 0 and b["predictions"]["shape"] == [shape[0]], "evaluation logits/predictions shapes disagree")
        expected_n = data["y_test"]["shape"][0] if e["kind"] == "accuracy" else config["attack_target_count"]
        _require(sum(b["logits"]["shape"][0] for b in batches) == expected_n, "evaluation batches do not cover the declared sample count")
        values = [e.get("value")] if e["kind"] == "accuracy" else e.get("value")
        _require(isinstance(values, list) and len(values) == (1 if e["kind"] == "accuracy" else 2)
            and all(timing.finite(v) and 0 <= v <= 1 for v in values), "invalid evaluation values")
    checkpoints = payload.get("checkpoints", [])
    _require([c.get("round") for c in checkpoints] == [0, 1, 2, 3], "missing or duplicated observed checkpoints")
    for c in checkpoints:
        _tensor(c.get("model"), "checkpoint.model")
        _record(c.get("record"), c["round"])
        _require(c.get("diagnostic_count") == (config["num_clients"] if c["round"] else 0), "checkpoint diagnostic coverage differs")
        model = payload["initial_model"] if c["round"] == 0 else rounds[c["round"] - 1]["post_model"]
        _require(c["model"] == model, "duplicated model observations disagree")
        if c["round"]:
            _require(c["record"] == rounds[c["round"] - 1]["record"], "duplicated round record observations disagree")
        _require(all(c["record"][key] == 0 for key in ("false_positive_revocations", "true_positive_revocations", "blacklisted_clients", "nonfinite_updates")), "unexpected removal or nonfinite counter in clean prefix")
        if c["round"]:
            _require(c["record"]["accepted_updates"] == config["num_clients"] and c["record"]["rejected_updates"] == 0,
                     "clean warmup round does not contain every configured update")
            r = rounds[c["round"] - 1]
            for client in r["clients"]:
                _require(client["model_input"] == checkpoints[c["round"] - 1]["model"], "client model input disagrees with preceding global model")
                _require(client["update"]["shape"] == payload["initial_model"]["shape"], "client update shape differs from global model")
            for name in ("aggregate", "post_model"):
                _require(r[name]["shape"] == payload["initial_model"]["shape"], "aggregation/global model shape mismatch")
    for e in evaluations:
        checkpoint = checkpoints[e["round"]]
        _require(e["model_input"] == checkpoint["model"], "evaluation model input disagrees with corresponding global model")
        expected = checkpoint["record"]["accuracy"] if e["kind"] == "accuracy" else [checkpoint["record"][k]
            for k in ("attack_target_success_rate", "attack_target_confidence")]
        _require(e["value"] == expected, "evaluation values disagree with original round record")
    return payload


def _leaf_differences(later, earlier, path=""):
    if isinstance(later, dict) and isinstance(earlier, dict):
        for key in sorted(set(later) | set(earlier)):
            location = path + "." + key if path else key
            if key not in later or key not in earlier:
                yield {"field": location, "later_present": key in later, "earlier_present": key in earlier}
            else:
                yield from _leaf_differences(later[key], earlier[key], location)
    elif isinstance(later, list) and isinstance(earlier, list):
        if len(later) != len(earlier):
            yield {"field": path + ".length", "later": len(later), "earlier": len(earlier)}
        for index, (a, b) in enumerate(zip(later, earlier)):
            yield from _leaf_differences(a, b, path + "[" + str(index) + "]")
    elif later != earlier or isinstance(later, bool) != isinstance(earlier, bool):
        yield {"field": path, "later": later, "earlier": earlier}


def compare_observations(later, earlier):
    """Locate the earliest differing observed boundary, never infer a CUDA cause."""
    stages = []
    def compare(stage, a, b, rd=None, client=None):
        differences = list(_leaf_differences(a, b))
        if differences:
            stages.append({"stage": stage, "round": rd, "client_id": client,
                           "different_fields": len(differences), "first": differences[0]})
    compare("data", later["data"], earlier["data"])
    compare("partition", later["partition"], earlier["partition"])
    compare("model_spec", later["model_spec"], earlier["model_spec"])
    compare("initial_model", later["initial_model"], earlier["initial_model"])
    for rd in range(4):
        if rd:
            a, b = later["rounds"][rd - 1], earlier["rounds"][rd - 1]
            for left, right in zip(a["clients"], b["clients"]):
                cid = left["client_id"]
                compare("client_inputs_and_reconstructed_order", {k: v for k, v in left.items() if k not in ("update", "stats")},
                        {k: v for k, v in right.items() if k not in ("update", "stats")}, rd, cid)
                compare("client_update", left["update"], right["update"], rd, cid)
                compare("local_training_statistics", left["stats"], right["stats"], rd, cid)
            compare("coefficients", a["coefficients"], b["coefficients"], rd)
            compare("candidate_and_aggregation_order", {k: a[k] for k in ("candidate_order", "verified_order", "aggregate_order")},
                    {k: b[k] for k in ("candidate_order", "verified_order", "aggregate_order")}, rd)
            compare("aggregate", a["aggregate"], b["aggregate"], rd)
            compare("post_model", a["post_model"], b["post_model"], rd)
        for kind in ("accuracy", "target"):
            a = next(e for e in later["evaluations"] if e["round"] == rd and e["kind"] == kind)
            b = next(e for e in earlier["evaluations"] if e["round"] == rd and e["kind"] == kind)
            compare("evaluation_" + kind, a, b, rd)
        compare("round_record", later["checkpoints"][rd]["record"], earlier["checkpoints"][rd]["record"], rd)
        compare("checkpoint_model", later["checkpoints"][rd]["model"], earlier["checkpoints"][rd]["model"], rd)
        if rd:
            compare("client_diagnostics", later["rounds"][rd - 1]["diagnostics"], earlier["rounds"][rd - 1]["diagnostics"], rd)
    counts = {}
    for item in stages:
        counts[item["stage"]] = counts.get(item["stage"], 0) + item["different_fields"]
    return {"available": True, "equal": not stages, "first_divergence": stages[0] if stages else None,
            "differing_boundaries": len(stages), "different_fields_by_stage": counts}


def _historical(payload, historical):
    if not isinstance(historical, list) or [r.get("round") for r in historical] != [0, 1, 2, 3]:
        return {"available": False, "equal": None, "error": "old H0 first four round records unavailable"}
    differences = []
    for rd, old in enumerate(historical):
        if any(field not in old for field in PAIR_FIELDS):
            return {"available": False, "equal": None, "error": "old H0 scientific record fields incomplete"}
        actual = payload["checkpoints"][rd]["record"]
        for field in PAIR_FIELDS:
            if actual[field] != old[field]:
                differences.append({"round": rd, "field": field, "probe": actual[field], "old_H0": old[field]})
    return {"available": True, "equal": not differences, "different_fields": len(differences),
            "first_divergence": differences[0] if differences else None,
            "scope": "old scalar records only; no old model or update hashes exist in this comparison"}


def evidence_hashes(output):
    """Only this probe's identity/completion files; never checkpoints or model files."""
    paths = {output / "manifest.json", output / "task_plans/prefix.json"}
    for pattern in ("tasks/*/task.json", "tasks/*/completed.json"):
        paths.update(output.glob(pattern))
    return {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
            for p in sorted(paths)}


def summarize(output):
    import cifar_prefix_probe_protocol as protocol
    output = Path(output)
    try:
        before = evidence_hashes(output)
        sources_before = protocol.source_hashes()
        manifest, tasks = protocol.read_study(output, current_sources=True)
    except Exception as exc:
        return {"status": "unavailable_or_invalid_study", "output": str(output), "error": str(exc),
                "training_started_by_summary": False}
    rows, payloads = [], {}
    for task in tasks:
        row = {"task_id": task["task_id"], "partition": task["partition"], "repeat": task["repeat"],
               "task_fingerprint": task["fingerprint"], "status": "missing"}
        try:
            artifact = protocol.load_completed(output, task)
            if artifact is not None:
                _require(artifact.get("status") == "complete" and artifact.get("task_fingerprint") == task["fingerprint"], "artifact task identity/status differs")
                _require(artifact.get("fresh_start") is True and artifact.get("checkpoints_used") is False, "probe artifact does not establish a fresh run")
                _require(artifact.get("gpu_uuid") == manifest["same_gpu_uuid"], "probe did not record the same frozen physical GPU")
                _require(isinstance(artifact.get("environment"), dict), "missing recorded execution environment")
                _require(matched.normalized_environment(artifact["environment"]) == manifest["reference"]["execution_environment"],
                         "recorded numerical environment differs from frozen H0 reference")
                _require(timing.finite(artifact.get("wall_seconds")) and artifact["wall_seconds"] >= 0, "invalid recorded elapsed time")
                _require(isinstance(artifact.get("artifact_fingerprint"), str) and len(artifact["artifact_fingerprint"]) == 64, "missing artifact fingerprint")
                payload = validate_observations(artifact.get("observations"), task)
                payloads[task["task_id"]] = payload
                row.update(status="complete", gpu_uuid=artifact["gpu_uuid"], wall_seconds=artifact["wall_seconds"],
                    artifact_fingerprint=artifact["artifact_fingerprint"], observed_rounds=3,
                    observed_clients=sum(len(r["clients"]) for r in payload["rounds"]),
                    old_H0_comparison=_historical(payload, manifest["reference"].get("historical_prefix", {}).get(task["partition"])))
        except Exception as exc:
            row.update(status="invalid_evidence", error=str(exc))
        rows.append(row)
    pairs = []
    matrix = {(p, repeat): [t for t in tasks if (t["partition"], t["repeat"]) == (p, repeat)]
              for p in ("iid", "dirichlet") for repeat in (1, 2, 3)}
    for partition in ("iid", "dirichlet"):
        for repeat in (2, 3):
            pair = {"partition": partition, "later_repeat": repeat, "earlier_repeat": 1,
                    "available": False, "equal": None, "first_divergence": None}
            later, earlier = matrix[(partition, repeat)], matrix[(partition, 1)]
            if len(later) == len(earlier) == 1:
                pair.update(later_task_id=later[0]["task_id"], earlier_task_id=earlier[0]["task_id"])
                if all(t["task_id"] in payloads for t in (later[0], earlier[0])):
                    pair.update(compare_observations(payloads[later[0]["task_id"]], payloads[earlier[0]["task_id"]]))
            pairs.append(pair)
    exact = len(tasks) == 6 and len({t["task_id"] for t in tasks}) == 6 and all(len(items) == 1 for items in matrix.values())
    complete = sum(r["status"] == "complete" for r in rows)
    ready = exact and complete == 6 and all(p["available"] for p in pairs)
    equal = ready and all(p["equal"] for p in pairs)
    try:
        unchanged = before == evidence_hashes(output) and sources_before == protocol.source_hashes()
    except Exception:
        unchanged = False
    if not unchanged:
        ready, equal = False, False
        for pair in pairs:
            pair.update(available=False, equal=None, first_divergence=None, error="probe evidence or sources changed during summary")
            pair.pop("different_fields_by_stage", None)
    profile = manifest["reference"]["execution_environment"].get("torch", {})
    return {"status": "changed_during_read" if not unchanged else "complete" if ready else "incomplete_or_invalid_evidence", "output": str(output),
        "manifest_fingerprint": manifest["fingerprint"], "protocol": manifest.get("protocol"),
        "same_gpu_uuid": manifest["same_gpu_uuid"], "recorded_source_count": len(manifest["source_sha256"]),
        "recorded_source_map_sha256": timing.base.digest(manifest["source_sha256"]),
        "source_and_reference_verified": True, "expected_tasks": 6, "complete_tasks": complete,
        "input_evidence_unchanged": unchanged,
        "unchanged_check_scope": "probe manifest/plan/task/completion files and current source hashes throughout this read; old90 reference files verified at read_study entry",
        "numerical_environment": {key: profile.get(key) for key in ("version", "cuda_version", "cudnn_version", "deterministic_algorithms",
            "deterministic_warn_only", "cudnn_deterministic", "cudnn_benchmark", "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "float32_matmul_precision")},
        "expected_pairs": 4, "available_pairs": sum(p["available"] for p in pairs),
        "all_pairs_equal": equal if ready else None, "training_started_by_summary": False,
        "rows": rows, "pairs": pairs,
        "decision": {"action": "review_observed_prefix_boundaries" if ready else "resolve_incomplete_or_invalid_probe_evidence",
            "formal_qualification_assessed": False, "three_round_health_assessed": False, "automatic_next_stage": False,
            "cuda_cause_identified": False,
            "limitation": "agreement describes these same-GPU fresh three-round observations only and does not explain the old multi-GPU divergence; differing hashes locate an observed boundary, not its cause; reconstructed epoch indices do not certify the actual executed sample order"}}


def print_summary(report):
    records = [{"type": "header", **{k: v for k, v in report.items() if k not in ("rows", "pairs", "decision")}},
        {"type": "legend", "schema": "cifar-prefix-probe-summary-v1", "comparison_direction": "repeat2/3 minus repeat1; exact fingerprints/scalars",
         "stages": "data, partition, model spec, initialization, then each round's client inputs/reconstructed sample order, returned updates, coefficients, aggregate, post model, evaluations, round record and diagnostic observations",
         "first_divergence": "first differing recorded boundary in the declared traversal; hashes include dtype/shape/bytes, no update magnitude is inferred",
         "historic_scope": "four old H0 scalar records, separate from fresh-repeat hashes; no health, performance gate or causal attribution"}]
    records.extend({"type": "task", **row} for row in report.get("rows", []))
    records.extend({"type": "pair", **row} for row in report.get("pairs", []))
    if "decision" in report:
        records.append({"type": "decision", **report["decision"]})
    lines = [json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) for row in records]
    print("=== CIFAR_PREFIX_PROBE_BEGIN ===", flush=True)
    for line in lines:
        print(line)
    print("=== CIFAR_PREFIX_PROBE_END ===", flush=True)
