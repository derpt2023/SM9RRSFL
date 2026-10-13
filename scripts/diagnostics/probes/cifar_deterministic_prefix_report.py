"""Read-only, three-round comparisons of declared singleton backward policies."""
from __future__ import annotations

import json
from pathlib import Path

import cifar_prefix_probe_report as prefix
import cifar_tail_determinism_report as tail

POLICIES = ("original", "singleton_backward_cudnn_deterministic")
STAGES = tail.step.STAGES
PRE_STAGES = tail.PRE_STAGES
PAIRS = tail.step.PAIRS
validate_environment_policy = tail.validate_environment_policy


def validate_observations(payload, task):
    from cifar_deterministic_prefix_runtime import validate_policy_observation
    prefix.validate_observations(payload, task)
    validate_policy_observation(payload, task)
    layout = None
    model = payload["initial_model"]
    for batch in payload["singleton_batches"]:
        prefix._require(batch["logits"]["shape"] == [1, payload["model_spec"]["num_classes"]],
                        "singleton logit class coverage differs")
        prefix._require(all(batch[field]["shape"] == model["shape"] and batch[field]["dtype"] == model["dtype"]
            for field in ("pre_parameters", "gradients", "post_parameters")), "singleton model/gradient layout differs from global model")
        prefix._require(layout is None or batch["parameter_layout"] == layout, "singleton parameter layout changed across client rounds")
        layout = batch["parameter_layout"]
        if batch["batch"] == 0:
            client = next(c for c in payload["rounds"][batch["round"] - 1]["clients"] if c["client_id"] == batch["client_id"])
            prefix._require(batch["pre_parameters"] == client["model_input"], "first minibatch parameters differ from actual client input")
    return payload


def _singleton_key(batch):
    return batch["round"], batch["client_id"], batch["batch"]


def _common_context(later, earlier):
    fields = [field for field in ("data", "partition", "model_spec", "initial_model") if later[field] != earlier[field]]
    for kind in ("accuracy", "target"):
        a = next(e for e in later["evaluations"] if e["round"] == 0 and e["kind"] == kind)
        b = next(e for e in earlier["evaluations"] if e["round"] == 0 and e["kind"] == kind)
        if a != b:
            fields.append("round0.evaluation_" + kind)
    if later["checkpoints"][0]["record"] != earlier["checkpoints"][0]["record"]:
        fields.append("round0.record")
    if later["numerical_policy"]["baseline"] != earlier["numerical_policy"]["baseline"]:
        fields.append("numerical_policy.baseline")
    return fields


def _boundary_order(item):
    stage, rd = item["stage"], item.get("round")
    initialization = {"numerical_policy.baseline": -5, "data": -4, "partition": -3, "model_spec": -2, "initial_model": -1}
    if stage in initialization:
        return (-1, initialization[stage], 0, 0)
    rd = 0 if rd is None else rd
    client = item.get("client_id")
    if client is not None and stage not in ("client_diagnostics",):
        index = int(client.rsplit("-", 1)[-1])
        if stage == "client_inputs_and_reconstructed_order":
            position = -2
        elif stage == "actual_batch_coverage":
            position = -1
        elif stage.startswith("singleton."):
            position = item["batch"] * len(STAGES) + STAGES.index(stage.split(".", 1)[1])
        else:
            position = 10**9 + (stage == "local_training_statistics")
        return (rd, index, position, 0)
    order = {"candidate_and_aggregation_order": 0, "coefficients": 1, "aggregate": 2, "post_model": 3,
             "evaluation_accuracy": 4, "evaluation_target": 5, "round_record": 6, "checkpoint_model": 7,
             "client_diagnostics": 8}
    return (rd, 10**9, order.get(stage, 9), 0)


def compare_observations(later, earlier):
    """Interleave actual singleton boundaries before the resulting client update."""
    pipeline = prefix.compare_observations(later, earlier)
    differences = []
    def compare(stage, a, b, *, rd=None, cid=None, batch=None):
        leaves = list(prefix._leaf_differences(a, b))
        if leaves:
            differences.append({"stage": stage, "round": rd, "client_id": cid, "batch": batch,
                "different_fields": len(leaves), "first": leaves[0], "magnitude": None})
    compare("numerical_policy.baseline", later["numerical_policy"]["baseline"], earlier["numerical_policy"]["baseline"])
    for a, b in zip(later["actual_batches"], earlier["actual_batches"]):
        compare("actual_batch_coverage", a, b, rd=a["round"], cid=a["client_id"])
    left = {_singleton_key(b): b for b in later["singleton_batches"]}
    right = {_singleton_key(b): b for b in earlier["singleton_batches"]}
    prefix._require(set(left) == set(right), "paired actual singleton event coverage differs")
    singleton_equal_by_round = []
    for rd in (1, 2, 3):
        before = len(differences)
        for key in sorted(left, key=lambda key: (key[0], int(key[1].rsplit("-", 1)[-1]), key[2])):
            if key[0] != rd:
                continue
            a, b = left[key], right[key]
            prefix._require(a["parameter_layout"] == b["parameter_layout"], "paired singleton parameter layouts differ")
            for stage in STAGES:
                compare("singleton." + stage, a[stage], b[stage], rd=rd, cid=key[1], batch=key[2])
        singleton_equal_by_round.append({"round": rd, "equal": len(differences) == before})
    firsts = [item for item in (pipeline["first_divergence"], *differences) if item is not None]
    first = min(firsts, key=_boundary_order) if firsts else None
    counts = {}
    for difference in differences:
        counts[difference["stage"]] = counts.get(difference["stage"], 0) + difference["different_fields"]
    r1_pre = []
    for key in left:
        if key[0] == 1:
            fields = [field for field in PRE_STAGES if left[key][field] != right[key][field]]
            r1_pre.extend(key[1] + "." + field for field in fields)
    return {"available": True, "equal": pipeline["equal"] and not differences,
        "first_divergence": first, "pipeline": pipeline,
        "singleton_different_fields_by_stage": counts,
        "singleton_equal_by_round": singleton_equal_by_round,
        "common_initial_context_mismatches": _common_context(later, earlier),
        "round1_singleton_pre_mismatches": r1_pre,
        "later_round_input_differences_are_propagation": True,
        "numerical_magnitudes_available": False}


def historical_context(payload, reference):
    """Compare only the shared and actually observed old first-round boundaries."""
    anchors = reference["tail_anchors"]
    prefix._require(len(anchors) == 6, "six historical scoped-tail observations are required")
    policy_counts = {policy: sum(a["task"]["policy"] == policy for a in anchors)
                     for policy in ("original", "tail_cudnn_deterministic")}
    prefix._require(set(policy_counts.values()) == {3}, "historical scoped-tail policy coverage differs")
    common, pre, effect = [], [], []
    r1 = {b["client_id"]: b for b in payload["singleton_batches"] if b["round"] == 1}
    new_clients = {c["client_id"]: c for c in payload["rounds"][0]["clients"]}
    for anchor in anchors:
        name = anchor["task"]["task_id"]
        old = anchor["observations"]
        old_prefix = old["prefix"]
        for field in ("data", "partition", "model_spec", "initial_model"):
            if payload[field] != old_prefix[field]:
                common.append(name + "." + field)
        for kind in ("accuracy", "target"):
            a = next(e for e in payload["evaluations"] if e["round"] == 0 and e["kind"] == kind)
            b = next(e for e in old_prefix["evaluations"] if e["round"] == 0 and e["kind"] == kind)
            if a != b:
                common.append(name + ".round0.evaluation_" + kind)
        if payload["checkpoints"][0]["record"] != old_prefix["checkpoints"][0]["record"]:
            common.append(name + ".round0.record")
        if payload["numerical_policy"]["baseline"] != old["numerical_policy"]["baseline"]:
            common.append(name + ".numerical_policy.baseline")
        targets = {target["client_id"] for target in old["targets"]}
        prefix._require(set(r1) == targets, "first-round singleton identity differs from historical targets")
        for client in old_prefix["rounds"][0]["clients"]:
            actual = new_clients[client["client_id"]]
            common.extend(name + "." + client["client_id"] + "." + field
                for field in tail.step.INPUT_FIELDS if actual[field] != client[field])
            if client["client_id"] not in targets and actual["update"] != client["update"]:
                common.append(name + "." + client["client_id"] + ".non_target_update")
        for target in old["targets"]:
            cid = target["client_id"]
            actual, batch = r1[cid], target["batches"][-1]
            for field in ("batch", "epoch", "batch_in_epoch", "samples", *PRE_STAGES):
                if actual[field] != batch[field]:
                    pre.append(name + "." + cid + "." + field)
            if actual["parameter_layout"] != target["parameter_layout"]:
                pre.append(name + "." + cid + ".parameter_layout")
            if payload["numerical_policy"]["policy"] == POLICIES[1] and anchor["task"]["policy"] == "tail_cudnn_deterministic":
                effect.append({"reference_task_id": name, "client_id": cid,
                    "gradients_equal": actual["gradients"] == batch["gradients"],
                    "post_parameters_equal": actual["post_parameters"] == batch["post_parameters"],
                    "final_delta_equal": new_clients[cid]["update"] == target["final_delta"]})
    return {"common_context_reproduced": not common, "common_context_mismatches": common,
        "round1_singleton_pre_reproduced": not pre, "round1_singleton_pre_mismatches": pre,
        "deterministic_effect_comparisons": effect,
        "deterministic_effect_reproduced": all(row[field] for row in effect for field in
            ("gradients_equal", "post_parameters_equal", "final_delta_equal")) if effect else None,
        "scope": "old tail data/init/round0, recorded client inputs and non-target updates through client84, and the two actual singleton pre-backward boundaries; earlier target minibatches were not observed by this new probe"}


def decision(rows, pairs, ready):
    if not ready:
        return {"action": "resolve_incomplete_or_invalid_evidence", "conclusion": "incomplete_or_invalid_evidence",
                **_limits(), "original_pairs_different": None, "scoped_policy_pairs_equal": None}
    groups = {policy: [p for p in pairs if p["kind"] == "within_policy" and p["earlier_policy"] == policy] for policy in POLICIES}
    original_different = sum(not p["equal"] for p in groups[POLICIES[0]])
    stable = all(p["equal"] for p in groups[POLICIES[1]])
    reasons = [r["task_id"] for r in rows if not r["historical_context"]["common_context_reproduced"] or
               not r["historical_context"]["round1_singleton_pre_reproduced"]]
    reasons += [p["pair_id"] for p in pairs if p["common_initial_context_mismatches"] or p["round1_singleton_pre_mismatches"]]
    if reasons:
        conclusion, action = "context_mismatch_no_causal_interpretation", "resolve_context_mismatch_before_interpretation"
    elif not stable:
        conclusion, action = "scoped_policy_did_not_reproduce_three_round_trajectory", "review_first_divergence_before_any_extension"
    elif original_different:
        conclusion, action = "supports_observed_three_round_reproducibility_for_scoped_policy", "review_three_round_reproducibility_before_mechanism_study"
    else:
        conclusion, action = "scoped_policy_three_round_equal_control_variability_not_reproduced", "review_three_round_reproducibility_before_mechanism_study"
    return {"action": action, "conclusion": conclusion, "original_pairs_different": original_different,
        "scoped_policy_pairs_equal": sum(p["equal"] for p in groups[POLICIES[1]]),
        "scoped_policy_all_three_round_observations_equal": stable,
        "context_review_reasons": reasons,
        "historical_deterministic_effect_requires_review": any(r["historical_context"]["deterministic_effect_reproduced"] is False for r in rows),
        **_limits()}


def _limits():
    return {"health_assessed": False, "performance_improvement_assessed": False,
        "formal_qualification_assessed": False, "automatic_next_stage": False, "operator_cause_identified": False,
        "limitation": "three rounds of clean warmup on the recorded physical GPU; bitwise equality is observed repeatability, not a guarantee for later rounds or platforms; round2/3 cross-policy differences are expected propagation after round1 and do not identify an operator or establish performance improvement"}


def summarize(output):
    import cifar_deterministic_prefix_protocol as protocol
    output = Path(output).resolve()
    def snapshot():
        return {"evidence": protocol.evidence_hashes(output), "sources": protocol.source_hashes()}
    try:
        before = snapshot()
        manifest, tasks = protocol.read_study(output, current_sources=True)
        prefix._require(len(tasks) == 6 and {(t["policy"], t["repeat"]) for t in tasks} ==
            {(policy, repeat) for policy in POLICIES for repeat in (1, 2, 3)} and len({t["task_id"] for t in tasks}) == 6,
            "six-task policy/repeat matrix differs")
        rows, payloads, implementation = [], {}, {}
        for task in tasks:
            row = {"task_id": task["task_id"], "task_fingerprint": task["fingerprint"], "policy": task["policy"],
                   "repeat": task["repeat"], "status": "missing"}
            try:
                artifact = protocol.load_completed(output, task)
                if artifact is not None:
                    prefix._require(artifact.get("status") == "complete" and artifact.get("task_fingerprint") == task["fingerprint"], "artifact identity differs")
                    prefix._require(artifact.get("fresh_start") is True and artifact.get("checkpoints_used") is False, "artifact is not a fresh probe")
                    prefix._require(type(artifact.get("completed_training_rounds")) is int and artifact["completed_training_rounds"] == 3,
                                    "artifact does not certify three completed training rounds")
                    prefix._require(artifact.get("gpu_uuid") == manifest["same_gpu_uuid"], "physical GPU identity differs")
                    prefix._require(prefix.timing.finite(artifact.get("wall_seconds")) and artifact["wall_seconds"] >= 0, "invalid observed worker wall time")
                    prefix._require(prefix.matched.normalized_environment(artifact["environment"]) == manifest["reference"]["execution_environment"], "numerical environment differs")
                    payload = validate_observations(artifact["observations"], task)
                    validate_environment_policy(payload, artifact["environment"])
                    history = historical_context(payload, manifest["reference"])
                    row.update(status="complete", artifact_fingerprint=artifact["artifact_fingerprint"], wall_seconds=artifact["wall_seconds"],
                        completed_training_rounds=3, observed_client_rounds=sum(len(r["clients"]) for r in payload["rounds"]),
                        actual_singleton_events=len(payload["singleton_batches"]), historical_context=history,
                        numerical_policy=payload["numerical_policy"],
                        trajectory=[[c["round"], c["record"]["accuracy"], c["record"]["attack_target_success_rate"],
                                     c["record"]["attack_target_confidence"], c["model"]["sha256"]] for c in payload["checkpoints"]])
                    source_info = tail._implementation_evidence(artifact.get("implementation_evidence"))
                    if source_info is not None:
                        source_id = prefix.timing.base.digest(source_info)
                        implementation[source_id] = source_info
                        row["implementation_evidence_id"] = source_id
                    else:
                        row["implementation_evidence_status"] = "not_recorded"
                    payloads[task["task_id"]] = payload
            except Exception as exc:
                row.update(status="invalid_evidence", error=str(exc))
            rows.append(row)
        lookup = {(t["policy"], t["repeat"]): t["task_id"] for t in tasks}
        specifications = [("within_policy", (policy, a), (policy, b)) for policy in POLICIES for a, b in PAIRS]
        specifications += [("cross_policy", (POLICIES[0], r), (POLICIES[1], r)) for r in (1, 2, 3)]
        pairs = []
        for kind, earlier, later in specifications:
            a, b = lookup[later], lookup[earlier]
            pair = {"pair_id": b + "__" + a, "kind": kind, "earlier_policy": earlier[0], "earlier_repeat": earlier[1],
                    "later_policy": later[0], "later_repeat": later[1], "available": False, "equal": None, "first_divergence": None}
            if a in payloads and b in payloads:
                try:
                    pair.update(compare_observations(payloads[a], payloads[b]))
                except Exception as exc:
                    pair["error"] = str(exc)
            pairs.append(pair)
        protocol.verify_reference(manifest["reference"])
        prefix._require(before == snapshot(), "probe evidence or scientific sources changed during report read")
        complete = sum(row["status"] == "complete" for row in rows)
        ready = complete == 6 and all(pair["available"] for pair in pairs)
        value = {"status": "complete" if ready else "incomplete_or_invalid_evidence", "manifest_fingerprint": manifest["fingerprint"],
            "same_gpu_uuid": manifest["same_gpu_uuid"], "source_count": len(manifest["source_sha256"]),
            "source_map_sha256": prefix.timing.base.digest(manifest["source_sha256"]),
            "source_reference_and_evidence_verified": True, "training_started_by_summary": False,
            "expected_tasks": 6, "complete_tasks": complete, "expected_pairs": 9, "available_pairs": sum(p["available"] for p in pairs),
            "numerical_environment": {key: manifest["reference"]["execution_environment"].get("torch", {}).get(key)
                for key in ("version", "cuda_version", "cudnn_version", *tail.ENVIRONMENT_FLAGS)},
            "implementation_evidence": implementation, "rows": rows, "pairs": pairs, "decision": decision(rows, pairs, ready)}
        json.dumps(value, allow_nan=False)
        return value
    except Exception as exc:
        return {"status": "invalid", "error": str(exc), "training_started_by_summary": False}


def print_summary(value):
    profiles = []
    for row in value.get("rows", []):
        profile = row.get("numerical_policy", {}).get("baseline")
        if profile is not None and profile not in profiles:
            profiles.append(profile)
    records = [{"type": "header", **{k: v for k, v in value.items() if k not in ("rows", "pairs", "decision")}, "flag_profiles": profiles},
        {"type": "legend", "policies": POLICIES,
         "trajectory_order": ["round", "accuracy", "clean_source_to_target_background_rate", "target_confidence", "global_model_sha256"],
         "policy_event_order": ["round", "client_id", "batch", "epoch", "batch_in_epoch", "samples", "before_deterministic", "effective_deterministic", "restored_deterministic", "backward_completed"],
         "baseline_profile_index": "zero-based header.flag_profiles index; all nine baseline flags retained",
         "environment_cross_check": {"verified_fields": tail.ENVIRONMENT_FLAGS, "not_in_old_collector": ["cudnn_enabled", "cudnn_benchmark_limit"]},
         "direction": "later compared with earlier; cross-policy means singleton policy versus original at the same repeat number",
         "scope": "actual singleton boundaries are interleaved before each resulting client update; only first-round common/pre-backward conditions are required across policies; round2/3 model differences are propagation; fingerprints alone cannot give numerical distance",
         "implementation_evidence": "optional installed-source text, not proof of the loaded binary or executed kernel"}]
    for row in value.get("rows", []):
        compact = dict(row)
        if "numerical_policy" in row:
            policy = row["numerical_policy"]
            compact["numerical_policy"] = {"policy": policy["policy"], "baseline_profile_index": profiles.index(policy["baseline"]),
                "events": [[event[k] for k in ("round", "client_id", "batch", "epoch", "batch_in_epoch", "samples")] +
                    [event[phase]["cudnn_deterministic"] for phase in ("before", "effective", "restored")] + [event["backward_completed"]]
                    for event in policy["events"]], "checks": policy["checks"], "scoped_changes": policy["scoped_changes"],
                "all_other_flags_verified_unchanged": policy["other_flags_changed"] is False, "restored_on_exit": policy["restored_on_exit"]}
        records.append({"type": "task", **compact})
    records.extend({"type": "pair", **pair} for pair in value.get("pairs", []))
    if "decision" in value:
        records.append({"type": "decision", **value["decision"]})
    lines = [json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) for row in records]
    print("=== CIFAR_DETERMINISTIC_PREFIX_BEGIN ===", flush=True)
    for line in lines:
        print(line)
    print("=== CIFAR_DETERMINISTIC_PREFIX_END ===", flush=True)
