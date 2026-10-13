"""Read-only evidence for the two explicitly scoped backward policies."""
from __future__ import annotations

import json
from pathlib import Path

import cifar_client_step_probe_report as step

POLICIES = ("original", "tail_cudnn_deterministic")
PRE_STAGES = step.STAGES[:step.STAGES.index("gradients")]
ENVIRONMENT_FLAGS = ("cudnn_deterministic", "cudnn_benchmark", "cudnn_allow_tf32", "cuda_matmul_allow_tf32",
                     "deterministic_algorithms", "deterministic_warn_only", "float32_matmul_precision")


def validate_observations(payload, task, arrays):
    """The old observation contract and the new flag contract are both required."""
    from cifar_tail_determinism_runtime import validate_policy_observation
    step.validate_observations(payload, task, arrays)
    validate_policy_observation(payload, task)
    return payload


def validate_environment_policy(payload, environment):
    """Cross-check every flag recorded by the frozen environment collector."""
    baseline = payload["numerical_policy"]["baseline"]
    profile = environment.get("torch", {})
    for field in ENVIRONMENT_FLAGS:
        step._require(field in profile and type(profile[field]) is type(baseline[field]) and
            profile[field] == baseline[field], "policy/environment baseline differs or is missing: " + field)
    return {"verified_fields": list(ENVIRONMENT_FLAGS),
            "not_in_frozen_environment_collector": ["cudnn_enabled", "cudnn_benchmark_limit"]}


def _implementation_evidence(value):
    """Optional installed-source observations, never an executed-binary claim."""
    if not isinstance(value, dict):
        return None
    result = {key: value.get(key) for key in ("torch_version", "torch_git_version")}
    files = value.get("files", {})
    result["files"] = {name: {key: files.get(name, {}).get(key) for key in ("status", "sha256", "pattern_checks")}
        for name in ("tools/autograd/derivatives.yaml", "aten/src/ATen/native/Convolution.cpp")}
    result.update(binary_equivalence_verified=False, executed_algorithm_identified=False)
    return result


def _target(payload, client_id):
    return next(t for t in payload["targets"] if t["client_id"] == client_id)


def context_differences(later, earlier):
    """Compare the common context, allowing both designated target updates to vary."""
    a, b = later["prefix"], earlier["prefix"]
    differences = [field for field in ("data", "model_spec", "initial_model", "partition", "evaluations")
                   if a[field] != b[field]]
    if a["checkpoints"][0]["record"] != b["checkpoints"][0]["record"]:
        differences.append("round0_record")
    targets = {t["client_id"] for t in later["targets"]}
    if "numerical_policy" in later and "numerical_policy" in earlier and later["numerical_policy"]["baseline"] != earlier["numerical_policy"]["baseline"]:
        differences.append("numerical_policy.baseline")
    ac, bc = a["rounds"][0]["clients"], b["rounds"][0]["clients"]
    if [c["client_id"] for c in ac] != [c["client_id"] for c in bc]:
        return differences + ["client_coverage"]
    for new, old in zip(ac, bc):
        cid = new["client_id"]
        if cid not in targets:
            differences.extend(cid + "." + key for key in step.INPUT_FIELDS if new[key] != old[key])
            if new["update"] != old["update"]:
                differences.append(cid + ".non_target_update")
            if new["stats"] != old["stats"]:
                differences.append(cid + ".non_target_statistics")
    return differences


def pre_backward_differences(later, earlier, client_id):
    """Every earlier target batch, and actual tail inputs through scalar loss."""
    a, b = _target(later, client_id), _target(earlier, client_id)
    differences = [key for key in ("samples", "model_input", "training_seed", "learning_rate", "epochs", "batch_size", "parameter_layout")
                   if a[key] != b[key]]
    ac = next(c for c in later["prefix"]["rounds"][0]["clients"] if c["client_id"] == client_id)
    bc = next(c for c in earlier["prefix"]["rounds"][0]["clients"] if c["client_id"] == client_id)
    differences.extend("prefix." + key for key in step.INPUT_FIELDS if ac[key] != bc[key])
    if len(a["batches"]) != len(b["batches"]):
        return differences + ["minibatch_coverage"]
    for index, (new, old) in enumerate(zip(a["batches"], b["batches"])):
        fields = step.STAGES if index < len(a["batches"]) - 1 else PRE_STAGES
        differences.extend("batch" + str(index) + "." + field for field in fields if new[field] != old[field])
    return differences


def _client_first_difference(later, earlier, client_id, later_arrays, earlier_arrays):
    a, b = _target(later, client_id), _target(earlier, client_id)
    for batch, (new, old) in enumerate(zip(a["batches"], b["batches"])):
        for stage in step.STAGES:
            if new[stage] != old[stage]:
                saved = batch == len(a["batches"]) - 1
                return {"batch": batch, "stage": stage, "first_field": next(step.prefix_report._leaf_differences(new[stage], old[stage])),
                        "magnitude": step.magnitude(later_arrays[a["tail_snapshots"][stage]], earlier_arrays[b["tail_snapshots"][stage]]) if saved else None,
                        "magnitude_scope": "saved last minibatch" if saved else "no full snapshot at this boundary"}
    if a["final_delta"] != b["final_delta"]:
        return {"batch": None, "stage": "final_delta", "magnitude": step.magnitude(
            later_arrays[a["tail_snapshots"]["final_delta"]], earlier_arrays[b["tail_snapshots"]["final_delta"]]),
            "magnitude_scope": "saved final delta"}
    if a["stats"] != b["stats"]:
        return {"batch": None, "stage": "local_training_statistics", "magnitude": None}
    return None


def compare_observations(later, earlier, later_arrays, earlier_arrays):
    value = step.compare_observations(later, earlier, later_arrays, earlier_arrays)
    value["context_mismatches"] = context_differences(later, earlier)
    value["clients"] = []
    for target in later["targets"]:
        cid = target["client_id"]
        old = _target(earlier, cid)
        mismatches = pre_backward_differences(later, earlier, cid)
        first = _client_first_difference(later, earlier, cid, later_arrays, earlier_arrays)
        value["clients"].append({"client_id": cid, "pre_backward_equal": not mismatches,
            "last_batch": len(target["batches"]) - 1,
            "pre_backward_mismatches": mismatches, "first_divergence": first,
            "all_target_boundaries_equal": first is None,
            "tail_gradients_equal": target["batches"][-1]["gradients"] == old["batches"][-1]["gradients"],
            "final_delta_equal": target["final_delta"] == old["final_delta"]})
    return value


def historical_context(payload, reference):
    original = step.historical_context(payload, reference["anchors"])
    anchors = reference["step_anchors"]
    step._require(len(anchors) == 3, "three original local-step anchors are required")
    shared = []
    per_client = {t["client_id"]: [] for t in payload["targets"]}
    prefix_common = list(original["context_mismatches"])
    for cid in per_client:
        if cid + ".inputs" in prefix_common:
            prefix_common.remove(cid + ".inputs")
            per_client[cid].append("old_prefix.inputs")
    for number, anchor in enumerate(anchors, 1):
        old = anchor["observations"]
        shared.extend("step" + str(number) + "." + field for field in context_differences(payload, old))
        for cid in per_client:
            per_client[cid].extend("step" + str(number) + "." + field for field in pre_backward_differences(payload, old, cid))
    return {"prefix_context_reproduced": not prefix_common,
        "prefix_context_mismatches": prefix_common,
        "step_context_reproduced": not shared, "step_context_mismatches": shared,
        "clients": [{"client_id": cid, "historical_pre_backward_equal": not fields,
                     "historical_pre_backward_mismatches": fields} for cid, fields in per_client.items()]}


def decision(rows, pairs, ready, client_ids):
    clients = []
    for cid in client_ids:
        reasons = []
        if not ready:
            conclusion = "incomplete_or_invalid_evidence"
            original_reproduced = deterministic_equal = None
        else:
            for row in rows:
                history = row["historical_context"]
                if not history["prefix_context_reproduced"] or not history["step_context_reproduced"]:
                    reasons.append(row["task_id"] + ":historical_common_context")
                if not next(c for c in history["clients"] if c["client_id"] == cid)["historical_pre_backward_equal"]:
                    reasons.append(row["task_id"] + ":historical_pre_backward")
            groups = {policy: [] for policy in POLICIES}
            for pair in pairs:
                client = next(c for c in pair["clients"] if c["client_id"] == cid)
                if pair["context_mismatches"] or not client["pre_backward_equal"]:
                    reasons.append(pair["pair_id"] + ":fresh_context_or_pre_backward")
                if pair["kind"] == "within_policy":
                    groups[pair["earlier_policy"]].append(client)
            # The prior observation was a tail-gradient first difference in all three pairs.
            original_reproduced = all(c["first_divergence"] is not None and
                c["first_divergence"]["stage"] == "gradients" and not c["tail_gradients_equal"]
                and c["first_divergence"]["batch"] == c["last_batch"]
                for c in groups["original"])
            original_equal = all(c["all_target_boundaries_equal"] for c in groups["original"])
            deterministic_equal = all(c["all_target_boundaries_equal"] for c in groups["tail_cudnn_deterministic"])
            if reasons:
                conclusion = "context_mismatch_no_causal_interpretation"
            elif original_equal:
                conclusion = "original_variability_not_reproduced_in_this_panel"
            elif not deterministic_equal:
                conclusion = "scoped_policy_insufficient_for_observed_repeatability"
            elif original_reproduced:
                conclusion = "supports_suppression_of_observed_variability_in_this_scope"
            else:
                conclusion = "original_difference_pattern_changed_requires_review"
        clients.append({"client_id": cid, "conclusion": conclusion,
            "original_three_pairs_reproduce_tail_gradient_first_difference": original_reproduced,
            "deterministic_three_pairs_all_target_boundaries_equal": deterministic_equal,
            "context_review_reasons": reasons})
    return {"action": "review_scoped_backward_policy" if ready else "resolve_incomplete_or_invalid_evidence",
        "clients": clients, "health_assessed": False, "formal_qualification_assessed": False,
        "performance_improvement_assessed": False, "automatic_next_stage": False,
        "completed_training_rounds": 0, "operator_cause_identified": False,
        "limitation": "three instrumented fresh repeats per policy test only the two scoped backward calls; leaf gradients do not locate a convolution operator, and internal repeatability does not establish full-training determinism or accuracy improvement"}


def summarize(output):
    import cifar_tail_determinism_protocol as protocol
    output = Path(output).resolve()
    def snapshot():
        return {"evidence": protocol.evidence_hashes(output), "sources": protocol.source_hashes()}
    try:
        before = snapshot()
        manifest, tasks = protocol.read_study(output, current_sources=True)
        step._require(len(tasks) == 6 and {(t["policy"], t["repeat"]) for t in tasks} ==
            {(policy, repeat) for policy in POLICIES for repeat in (1, 2, 3)} and
            len({t["task_id"] for t in tasks}) == 6, "six-task policy/repeat matrix differs")
        rows, payloads, arrays_by_id, implementation = [], {}, {}, {}
        for task in tasks:
            row = {"task_id": task["task_id"], "policy": task["policy"], "repeat": task["repeat"],
                   "task_fingerprint": task["fingerprint"], "status": "missing"}
            try:
                artifact = protocol.load_completed(output, task)
                if artifact is not None:
                    step._require(artifact.get("status") == "complete" and artifact.get("task_fingerprint") == task["fingerprint"], "artifact identity differs")
                    step._require(artifact.get("fresh_start") is True and artifact.get("checkpoints_used") is False, "artifact is not a fresh probe")
                    step._require(artifact.get("training_round_completed") is False, "artifact does not certify the scheduled pre-aggregation stop")
                    step._require(artifact.get("gpu_uuid") == manifest["same_gpu_uuid"], "physical GPU identity differs")
                    step._require(step._finite(artifact.get("wall_seconds")) and artifact["wall_seconds"] >= 0, "invalid observed worker wall time")
                    step._require(step.matched.normalized_environment(artifact["environment"]) == manifest["reference"]["execution_environment"], "numerical environment differs")
                    arrays = protocol.load_arrays(output, task, artifact)
                    payload = validate_observations(artifact["observations"], task, arrays)
                    policy_environment = validate_environment_policy(payload, artifact["environment"])
                    row.update(status="complete", artifact_fingerprint=artifact["artifact_fingerprint"],
                        wall_seconds=artifact["wall_seconds"], observed_clients=len(payload["prefix"]["rounds"][0]["clients"]),
                        completed_rounds=0, historical_context=historical_context(payload, manifest["reference"]),
                        numerical_policy=payload["numerical_policy"],
                        policy_environment=policy_environment,
                        targets=[{"client_id": t["client_id"], "samples": t["samples"], "batches": len(t["batches"]),
                            "last_batch_size": t["batches"][-1]["samples"], "mean_loss": t["stats"]["loss"],
                            "final_delta_sha256": t["final_delta"]["sha256"]} for t in payload["targets"]])
                    payloads[task["task_id"]], arrays_by_id[task["task_id"]] = payload, arrays
                    source_info = _implementation_evidence(artifact.get("implementation_evidence"))
                    if source_info is not None:
                        source_id = step.prefix_report.timing.base.digest(source_info)
                        implementation[source_id] = source_info
                        row["implementation_evidence_id"] = source_id
                    else:
                        row["implementation_evidence_status"] = "not_recorded"
            except Exception as exc:
                row.update(status="invalid_evidence", error=str(exc))
            rows.append(row)
        lookup = {(t["policy"], t["repeat"]): t["task_id"] for t in tasks}
        specifications = [("within_policy", (policy, a), (policy, b)) for policy in POLICIES for a, b in step.PAIRS]
        specifications += [("cross_policy", (POLICIES[0], repeat), (POLICIES[1], repeat)) for repeat in (1, 2, 3)]
        pairs = []
        for kind, earlier, later in specifications:
            a, b = lookup[later], lookup[earlier]
            pair = {"pair_id": b + "__" + a, "kind": kind, "earlier_policy": earlier[0], "earlier_repeat": earlier[1],
                    "later_policy": later[0], "later_repeat": later[1], "available": False,
                    "all_recorded_equal": None, "first_divergence": None}
            if a in payloads and b in payloads:
                try:
                    pair.update(compare_observations(payloads[a], payloads[b], arrays_by_id[a], arrays_by_id[b]))
                except Exception as exc:
                    pair["error"] = str(exc)
            pairs.append(pair)
        protocol.verify_reference(manifest["reference"])
        step._require(before == snapshot(), "policy evidence or scientific sources changed during report read")
        complete = sum(row["status"] == "complete" for row in rows)
        ready = complete == 6 and all(p["available"] for p in pairs)
        client_ids = ["client-" + str(i) for i in tasks[0]["target_clients"]]
        value = {"status": "complete" if ready else "incomplete_or_invalid_evidence",
            "manifest_fingerprint": manifest["fingerprint"], "same_gpu_uuid": manifest["same_gpu_uuid"],
            "source_count": len(manifest["source_sha256"]),
            "source_map_sha256": step.prefix_report.timing.base.digest(manifest["source_sha256"]),
            "source_reference_and_evidence_verified": True, "expected_tasks": 6, "complete_tasks": complete,
            "implementation_evidence": implementation,
            "numerical_environment": {key: manifest["reference"]["execution_environment"].get("torch", {}).get(key)
                for key in ("version", "cuda_version", "cudnn_version", *ENVIRONMENT_FLAGS)},
            "rows": rows, "pairs": pairs, "decision": decision(rows, pairs, ready, client_ids)}
        json.dumps(value, allow_nan=False)
        return value
    except Exception as exc:
        return {"status": "invalid", "error": str(exc), "training_started": False}


def _compact_pair(pair):
    value = {k: v for k, v in pair.items() if k != "tail_comparisons"}
    def first_difference(first):
        if first is None:
            return None
        result = dict(first)
        metrics = result.get("magnitude")
        if metrics is not None:
            result["magnitude"] = [metrics[key] for key in step.METRIC_FIELDS]
        return result
    value["first_divergence"] = first_difference(value.get("first_divergence"))
    if "clients" in value:
        value["clients"] = [{**client, "first_divergence": first_difference(client["first_divergence"])} for client in value["clients"]]
    value["tail_comparisons"] = []
    for tail in pair.get("tail_comparisons", []):
        stages = {stage: [metrics[k] for k in step.METRIC_FIELDS] for stage, metrics in tail["metrics"].items()
                  if metrics["max_abs"] != 0 or metrics["numerically_unequal_fraction"] != 0}
        parameters = []
        for param in tail["per_parameter"]:
            changed = {stage: [param[stage][k] for k in step.METRIC_FIELDS] for stage in step.PARAMETER_STAGES
                       if param[stage]["max_abs"] != 0 or param[stage]["numerically_unequal_fraction"] != 0}
            if changed:
                parameters.append({"name": param["name"], "stages": changed})
        value["tail_comparisons"].append({"client_id": tail["client_id"], "last_batch": tail["last_batch"],
            "samples": tail["samples"], "numerically_changed_stages": stages, "numerically_changed_parameters": parameters})
    return value


def _compact_policy(value, baseline_index):
    return {"policy": value["policy"], "baseline_profile_index": baseline_index,
        "events": [[event[k] for k in ("client_id", "batch", "epoch", "batch_in_epoch", "samples")] +
            [event[phase]["cudnn_deterministic"] for phase in ("before", "effective", "restored")] +
            [event["backward_completed"]] for event in value["events"]],
        "checks": value["checks"], "scoped_changes": value["scoped_changes"],
        "all_other_flags_verified_unchanged": value["other_flags_changed"] is False,
        "restored_on_exit": value["restored_on_exit"]}


def print_summary(value):
    profiles = []
    for row in value.get("rows", []):
        profile = row.get("numerical_policy", {}).get("baseline")
        if profile is not None and profile not in profiles:
            profiles.append(profile)
    records = [{"type": "header", **{k: v for k, v in value.items() if k not in ("rows", "pairs", "decision")}, "flag_profiles": profiles},
        {"type": "legend", "metric_order": step.METRIC_FIELDS, "policies": POLICIES,
         "direction": "later minus earlier; cross-policy uses original as the norm reference; null relative L2 means a zero reference with nonzero difference",
         "policy_event_order": ["client_id", "batch", "epoch", "batch_in_epoch", "samples", "before_cudnn_deterministic", "effective_cudnn_deterministic", "restored_cudnn_deterministic", "backward_completed"],
         "baseline_profile_index": "zero-based header.flag_profiles index; all nine flags retained for every unique observed baseline",
         "policy_environment_verified_fields": ENVIRONMENT_FLAGS,
         "policy_environment_not_in_old_collector": ["cudnn_enabled", "cudnn_benchmark_limit"],
         "implementation_evidence": "optional installed-source text checks deduplicated by reported-content SHA; absent source is allowed and present source does not certify the loaded binary or executed algorithm",
         "pre_backward": "all earlier target minibatches, then actual last-minibatch inputs/parameters/logits/labels/loss; common context and stable non-target updates checked separately",
         "omitted_magnitudes": "omitted tail stages/parameter blocks have zero numerical distance, not necessarily identical bytes; first-divergence and equality fields use full byte fingerprints",
         "scope": "only last-minibatch and final-delta arrays support numerical distances; no complete training round or performance/health inference"}]
    for row in value.get("rows", []):
        compact = dict(row)
        if "numerical_policy" in compact:
            compact["numerical_policy"] = _compact_policy(compact["numerical_policy"], profiles.index(row["numerical_policy"]["baseline"]))
        if "policy_environment" in compact:
            compact.pop("policy_environment")
            compact["policy_environment_verified"] = True
        records.append({"type": "task", **compact})
    records.extend({"type": "pair", **_compact_pair(pair)} for pair in value.get("pairs", []))
    if "decision" in value:
        records.append({"type": "decision", **value["decision"]})
    lines = [json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) for record in records]
    print("=== CIFAR_TAIL_DETERMINISM_BEGIN ===", flush=True)
    for line in lines:
        print(line)
    print("=== CIFAR_TAIL_DETERMINISM_END ===", flush=True)
