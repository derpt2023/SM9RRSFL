"""Read-only, paired clean history-mechanism evidence and short-run diagnostics."""
from collections import Counter
from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace

import cifar_prefix_probe_report as prefix
import cifar_history_forensics as forensic
import cifar_tail_determinism_report as tail

ARMS = ("H0", "H1")
ROUNDS = 30
FREEZE_START = 25
SCHEMA = "cifar-mechanism-observation-v1"
PHASES = (("warmup_1_20", 1, 20), ("detection_21_24", 21, 24),
          ("first_freeze_25", 25, 25), ("post_freeze_26_30", 26, 30))
validate_environment_policy = tail.validate_environment_policy
_require, _tensor = prefix._require, prefix._tensor
FLAGS = ("is_malicious", "suspicious", "count_increment", "trace_requested", "trace_pending", "revoked",
         "aggregation_accepted", "history_eligible", "history_admitted", "history_frozen", "immediate_revocation",
         "attack_active", "recovery_eligible")


def _result(payload, task):
    terminal = payload["terminal"]
    return SimpleNamespace(config=SimpleNamespace(**task["config"]),
        records=[SimpleNamespace(**c["record"]) for c in payload["checkpoints"]],
        diagnostics=[SimpleNamespace(**d) for r in payload["rounds"] for d in r["diagnostics"]],
        malicious_clients=terminal["malicious_clients"], blacklisted_clients=terminal["blacklisted_clients"],
        stopped_round=terminal["stopped_round"], nonfinite_updates=terminal["nonfinite_updates"],
        final_accuracy=payload["checkpoints"][-1]["record"]["accuracy"], runtime_seconds=0.)


def validate_observations(payload, task):
    """Validate all recorded rounds, including a legitimate early exhausted run."""
    from cifar_mechanism_runtime import validate_policy_observation
    from cifar_mechanism_observer import validate_history_observation
    _require(isinstance(payload, dict) and payload.get("schema") == SCHEMA, "wrong mechanism observation schema")
    _require(payload.get("task_id") == task["task_id"] and payload.get("task_fingerprint") == task["fingerprint"], "observation task identity differs")
    _require(prefix._finite_tree(payload), "nonfinite mechanism observation")
    config = task["config"]
    _require(config["rounds"] == ROUNDS and config["malicious_ratio"] == 0 and config["method"] == "sm9rrs", "requires declared thirty clean Ours rounds")
    _require({k: v for k, v in payload.get("configuration", {}).items() if k != "device"} ==
             {k: v for k, v in config.items() if k != "device"}, "scientific configuration differs")
    terminal = payload.get("terminal", {})
    stopped = terminal.get("stopped_round")
    _require(type(stopped) is int and 1 <= stopped <= ROUNDS and terminal.get("requested_rounds") == ROUNDS,
             "invalid terminal round coverage")
    ids = ["client-" + str(i) for i in range(config["num_clients"])]
    _require(terminal.get("malicious_clients") == [] and prefix._integer(terminal.get("nonfinite_updates")), "invalid clean terminal labels/counters")
    blacklist = terminal.get("blacklisted_clients", [])
    _require(isinstance(blacklist, list) and len(set(blacklist)) == len(blacklist) and set(blacklist) <= set(ids), "invalid terminal blacklist")
    _require(terminal.get("stop_reason") == ("all_honest_revoked" if set(blacklist) == set(ids) else None), "terminal stop reason contradicts actual blacklist")
    _require(stopped == ROUNDS or (set(blacklist) == set(ids) and terminal.get("stop_reason") == "all_honest_revoked"),
             "incomplete run has no legitimate all-honest-revoked termination")
    data = payload.get("data", {})
    _require(all(key in data for key in prefix.DATA_FIELDS), "incomplete data fingerprints")
    for key in prefix.DATA_FIELDS:
        if key not in ("x_attack", "y_attack") or data[key] is not None:
            _tensor(data[key], "data." + key)
    _require((data["x_attack"] is None) == (data["y_attack"] is None), "incomplete paired attack data hashes")
    _require(isinstance(payload.get("model_spec"), dict) and bool(payload["model_spec"]), "missing model specification")
    _tensor(payload.get("initial_model"), "initial_model")
    model = payload["initial_model"]
    partition = payload.get("partition", {})
    _require(partition.get("strategy") == config["partition"] and partition.get("seed") == config["seed"] and
             partition.get("alpha") == config["dirichlet_alpha"], "partition identity differs")
    clients = partition.get("clients", [])
    _require([c.get("client_id") for c in clients] == ids, "partition client coverage/order differs")
    for c in clients:
        _require(prefix._integer(c.get("samples")), "invalid partition samples")
        _tensor(c.get("indices"), "partition.indices")
        _require(c["indices"]["shape"] == [c["samples"]], "partition index length differs")
    samples = {c["client_id"]: c["samples"] for c in clients}
    rounds, checkpoints = payload.get("rounds", []), payload.get("checkpoints", [])
    _require([r.get("round") for r in rounds] == list(range(1, stopped + 1)), "missing or duplicated training rounds")
    _require([r.get("round") for r in checkpoints] == list(range(stopped + 1)), "missing or duplicated checkpoint observations")
    required = {f.name for f in fields(prefix.timing.base.fl.ClientDiagnosticRecord)} - {"task_tag"}
    revoked = set()
    for row in rounds:
        rd = row["round"]
        active = [cid for cid in ids if cid not in revoked]
        _require(row.get("blacklisted_before") == sorted(revoked), "round starting blacklist differs from prior revocations")
        _require([c.get("client_id") for c in row.get("clients", [])] == active, "actual active client coverage/order differs")
        for c in row["clients"]:
            cid = c["client_id"]
            for field in ("model_input", "raw_update"):
                _tensor(c.get(field), "client." + field)
                _require(c[field]["shape"] == model["shape"] and c[field]["dtype"] == model["dtype"], "client vector shape/dtype differs")
            _require(c["model_input"] == checkpoints[rd - 1]["model"], "client input differs from prior model")
            _require(c.get("samples") == samples[cid] and c.get("training_seed") == config["seed"] + rd * 1009 + int(cid.split("-")[-1]), "client samples/seed differs")
            _require(c.get("epochs") == config["local_epochs"] and c.get("batch_size") == config["batch_size"] and
                     c.get("learning_rate") == config["lr"] * config["lr_decay"] ** (rd - 1), "client training schedule differs")
            stats = c.get("stats", {})
            _require(prefix.timing.finite(stats.get("loss")) and stats.get("samples") == c["samples"], "invalid local training statistics")
            epochs = c.get("epoch_indices", [])
            _require(len(epochs) == c["epochs"], "missing reconstructed epoch coverage")
            for epoch in epochs:
                _tensor(epoch, "client.epoch_indices")
                _require(epoch["shape"] == [c["samples"]], "epoch sample shape differs")
            _require(c.get("minibatch_sizes_per_epoch") == [min(c["batch_size"], c["samples"] - start)
                for start in range(0, c["samples"], c["batch_size"])], "reconstructed minibatch sizes differ")
        finite_ids = [c["client_id"] for c in row["clients"] if c["update_finite"]]
        _require(row.get("aggregation_executed") is bool(finite_ids), "aggregation execution differs from actual finite updates")
        for c in row["clients"]:
            _require(type(c.get("update_finite")) is bool, "missing actual update finiteness")
            if c["update_finite"]:
                _require(c.get("update") == c["raw_update"], "candidate raw update differs")
            else:
                _require("update" not in c, "nonfinite update incorrectly became candidate")
        for name in ("candidate_order", "verified_order", "aggregate_order"):
            values = row.get(name)
            expected_ids = finite_ids if name != "aggregate_order" or row["aggregation_executed"] else []
            _require(isinstance(values, list) and len(values) == len(expected_ids) and set(values) == set(expected_ids), "missing/duplicate client coverage in " + name)
        coefficients = row.get("coefficients", {})
        order = coefficients.get("order", [])
        values = coefficients.get("by_client", {})
        _require(len(order) == len(finite_ids) and set(order) == set(values) == set(finite_ids), "coefficient coverage differs")
        _require(all(prefix.timing.finite(v) and 0 <= v <= 1 + 1e-9 for v in values.values()), "invalid coefficient")
        _require(type(row.get("aggregation_executed")) is bool, "missing actual aggregation flag")
        if row["aggregation_executed"]:
            _tensor(row.get("aggregate"), "aggregate")
            _require(row["aggregate"]["shape"] == model["shape"] and row["aggregate"]["dtype"] == model["dtype"], "aggregate shape/dtype differs")
        else:
            _require(row.get("aggregate") is None, "unexecuted aggregation has an invented hash")
        _tensor(row.get("post_model"), "post_model")
        _require(row["post_model"] == checkpoints[rd]["model"], "checkpoint/postmodel disagreement")
        if not row["aggregation_executed"]:
            _require(row["post_model"] == checkpoints[rd - 1]["model"], "unaggregated model changed")
        prefix._record(row.get("record"), rd)
        _require(row["record"]["nonfinite_updates"] == sum(not c["update_finite"] for c in row["clients"]),
                 "round nonfinite count differs from observed actual updates")
        diagnostics = row.get("diagnostics", [])
        _require([d.get("client_id") for d in diagnostics] == row["verified_order"], "diagnostic verified coverage/order differs")
        for d in diagnostics:
            _require(required <= set(d) and "task_tag" not in d and d.get("round") == rd, "incomplete/public diagnostic identity")
            _require(all(type(d[field]) is bool for field in FLAGS), "invalid diagnostic boolean")
            _require(not d["is_malicious"] and not d["attack_active"], "attack evidence in clean panel")
            _require(d["aggregation_weight"] == values[d["client_id"]], "diagnostic coefficient differs")
            if d["revoked"]:
                revoked.add(d["client_id"])
        _require(row.get("blacklisted_after") == sorted(revoked) == checkpoints[rd].get("blacklisted"), "public blacklist differs from observed revocations")
        _require(row["record"]["false_positive_revocations"] == len(revoked) and row["record"]["true_positive_revocations"] == 0
                 and row["record"]["blacklisted_clients"] == len(revoked), "revocation counters differ")
        _require(row["record"]["accepted_updates"] == sum(d["aggregation_accepted"] for d in diagnostics)
                 and row["record"]["rejected_updates"] == len(active) - row["record"]["accepted_updates"], "accepted/rejected counts differ")
    _require(set(blacklist) == revoked and checkpoints[0].get("blacklisted") == [], "terminal/initial blacklist differs")
    _require(checkpoints[0]["record"]["nonfinite_updates"] == 0, "round0 has impossible training nonfinite counter")
    for c in checkpoints:
        rd = c["round"]
        _tensor(c.get("model"), "checkpoint.model")
        prefix._record(c.get("record"), rd)
        _require(c.get("diagnostic_count") == (len(rounds[rd - 1]["diagnostics"]) if rd else 0), "checkpoint diagnostic coverage differs")
        _require(c["record"] == rounds[rd - 1]["record"] if rd else c["model"] == model, "duplicated scientific observations differ")
    evaluations = payload.get("evaluations", [])
    _require([(e.get("round"), e.get("kind")) for e in evaluations] ==
             [(rd, kind) for rd in range(stopped + 1) for kind in ("accuracy", "target")], "missing/reordered evaluations")
    for e in evaluations:
        cp = checkpoints[e["round"]]
        _require(e.get("model_input") == cp["model"], "evaluation input differs from model")
        batches = e.get("batches", [])
        _require(bool(batches) and [b.get("batch") for b in batches] == list(range(len(batches))), "evaluation batch coverage differs")
        for b in batches:
            _tensor(b.get("logits"), "evaluation.logits")
            _tensor(b.get("predictions"), "evaluation.predictions")
            shape = b["logits"]["shape"]
            _require(len(shape) == 2 and shape[0] > 0 and shape[1] == payload["model_spec"]["num_classes"]
                and b["predictions"]["shape"] == [shape[0]], "evaluation batch shapes differ")
        count = data["y_test"]["shape"][0] if e["kind"] == "accuracy" else config["attack_target_count"]
        _require(sum(b["logits"]["shape"][0] for b in batches) == count, "evaluation sample coverage differs")
        expected = cp["record"]["accuracy"] if e["kind"] == "accuracy" else [cp["record"][f]
            for f in ("attack_target_success_rate", "attack_target_confidence")]
        _require(e.get("value") == expected, "evaluation metric disagrees with record")
    result = _result(payload, task)
    records = prefix.timing._records(result)
    prefix.timing._diagnostics(result, records)
    forensic._validate_details(result)
    _require(terminal["nonfinite_updates"] == sum(r["record"]["nonfinite_updates"] for r in rounds), "terminal nonfinite count differs")
    validate_policy_observation(payload, task)
    validate_history_observation(payload, task)
    diagnostic_map = {(r["round"], d["client_id"]): d for r in rounds for d in r["diagnostics"]}
    from sm9rrsfl.svd_detector import DetectionResult
    decision_fields = {f.name for f in fields(DetectionResult)}
    decision_flags = ("accepted", "would_flag", "count_increment", "history_eligible", "immediate_revocation", "recovery_eligible")
    mapping = {"reason": "decision_reason", "would_flag": "suspicious"}
    for event in payload["history_events"]:
        d = diagnostic_map[(event["round"], event["client_id"])]
        _require(set(event["decision"]) == decision_fields, "incomplete detector decision schema")
        _require(all(type(event["decision"][f]) is bool for f in decision_flags), "invalid detector decision boolean")
        for field, value in event["decision"].items():
            if field != "accepted":
                _require(d[mapping.get(field, field)] == value, "detector decision disagrees with client diagnostic")
        if event["round"] <= config["detector_window"]:
            _require(d["decision_reason"] == "clean_warmup" and not any(d[f] for f in
                ("suspicious", "revoked", "trace_requested", "trace_pending", "immediate_revocation")), "unexpected warmup detector outcome")
    return payload


def compare_observations(later, earlier, *, cross_arm=False):
    """Separate history effects from the still-common round-26 local training."""
    differences = []
    def compare(stage, a, b, rd=None, cid=None, batch=None, intended=False):
        leaves = list(prefix._leaf_differences(a, b))
        if leaves:
            local_chain = stage in ("active_client_order", "client_inputs_and_reconstructed_order", "actual_batch_coverage",
                "client_update", "local_training_statistics", "blacklisted_before") or stage.startswith("singleton.")
            role = "intended_history_intervention" if cross_arm and rd == FREEZE_START and intended else (
                "round26_local_common_condition" if cross_arm and rd == FREEZE_START + 1 and local_chain else
                "post_intervention_divergence" if cross_arm and rd is not None and rd > FREEZE_START else
                "common_precondition" if cross_arm else "repeatability")
            differences.append({"stage": stage, "round": rd, "client_id": cid, "batch": batch,
                "different_fields": len(leaves), "first": leaves[0], "scope": role, "magnitude": None})
    for field in ("data", "partition", "model_spec", "initial_model"):
        compare(field, later[field], earlier[field])
    compare("numerical_policy.baseline", later["numerical_policy"]["baseline"], earlier["numerical_policy"]["baseline"])
    stopped = min(later["terminal"]["stopped_round"], earlier["terminal"]["stopped_round"])
    def per_round(payload, name, rd):
        return [row for row in payload[name] if row["round"] == rd]
    def indexed(rows):
        return {row["client_id"]: row for row in rows}
    for rd in range(stopped + 1):
        if rd:
            a, b = later["rounds"][rd - 1], earlier["rounds"][rd - 1]
            compare("active_client_order", [c["client_id"] for c in a["clients"]], [c["client_id"] for c in b["clients"]], rd)
            left, right = indexed(a["clients"]), indexed(b["clients"])
            for cid in sorted(set(left) & set(right), key=lambda v: int(v.split("-")[-1])):
                x, y = left[cid], right[cid]
                compare("client_inputs_and_reconstructed_order", {k: v for k, v in x.items() if k not in ("stats", "update", "raw_update", "update_finite")},
                        {k: v for k, v in y.items() if k not in ("stats", "update", "raw_update", "update_finite")}, rd, cid)
                actual_a = [v for v in per_round(later, "actual_batches", rd) if v["client_id"] == cid]
                actual_b = [v for v in per_round(earlier, "actual_batches", rd) if v["client_id"] == cid]
                compare("actual_batch_coverage", actual_a, actual_b, rd, cid)
                singles_a = {v["batch"]: v for v in per_round(later, "singleton_batches", rd) if v["client_id"] == cid}
                singles_b = {v["batch"]: v for v in per_round(earlier, "singleton_batches", rd) if v["client_id"] == cid}
                compare("singleton.coverage", sorted(singles_a), sorted(singles_b), rd, cid)
                for batch in sorted(set(singles_a) & set(singles_b)):
                    for stage in (*tail.step.STAGES, "parameter_layout"):
                        compare("singleton." + stage, singles_a[batch][stage], singles_b[batch][stage], rd, cid, batch)
                compare("client_update", {k: x.get(k) for k in ("raw_update", "update_finite", "update")},
                        {k: y.get(k) for k in ("raw_update", "update_finite", "update")}, rd, cid)
                compare("local_training_statistics", x["stats"], y["stats"], rd, cid)
            for stage in ("candidate_order", "verified_order", "aggregate_order"):
                compare(stage, a[stage], b[stage], rd)
            events_a, events_b = per_round(later, "history_events", rd), per_round(earlier, "history_events", rd)
            compare("history_event_order", [e["client_id"] for e in events_a], [e["client_id"] for e in events_b], rd)
            left, right = indexed(events_a), indexed(events_b)
            for cid in sorted(set(left) & set(right), key=lambda v: int(v.split("-")[-1])):
                x, y = left[cid], right[cid]
                for field in ("before_evaluate", "after_evaluate", "decision"):
                    compare("detector." + field, x[field], y[field], rd, cid)
                compare("detector.forget", x["forget"], y["forget"], rd, cid)
                ca, cb = x["commit"], y["commit"]
                if isinstance(ca, dict) and isinstance(cb, dict):
                    for field in ("requested_admission", "before"):
                        compare("commit." + field, ca[field], cb[field], rd, cid)
                    compare("commit.admitted", ca["admitted"], cb["admitted"], rd, cid, intended=True)
                    for field in sorted(set(ca["after"]) | set(cb["after"])):
                        compare("commit.after." + field, ca["after"].get(field), cb["after"].get(field), rd, cid,
                                intended=field in ("history", "history_size", "normal"))
                else:
                    compare("commit.coverage", ca, cb, rd, cid)
            diagnostics_a, diagnostics_b = indexed(a["diagnostics"]), indexed(b["diagnostics"])
            for cid in sorted(set(diagnostics_a) & set(diagnostics_b), key=lambda v: int(v.split("-")[-1])):
                x, y = diagnostics_a[cid], diagnostics_b[cid]
                compare("detector_diagnostics", {k: v for k, v in x.items() if k != "history_admitted"},
                        {k: v for k, v in y.items() if k != "history_admitted"}, rd, cid)
                compare("history_admitted", x["history_admitted"], y["history_admitted"], rd, cid, intended=True)
            compare("coefficients", a["coefficients"], b["coefficients"], rd)
            for field in ("aggregation_executed", "aggregate", "post_model", "blacklisted_before", "blacklisted_after"):
                compare(field, a[field], b[field], rd)
        for kind in ("accuracy", "target"):
            x = next(e for e in later["evaluations"] if e["round"] == rd and e["kind"] == kind)
            y = next(e for e in earlier["evaluations"] if e["round"] == rd and e["kind"] == kind)
            compare("evaluation_" + kind, x, y, rd)
        for field in ("record", "model", "diagnostic_count", "blacklisted"):
            compare("checkpoint." + field, later["checkpoints"][rd][field], earlier["checkpoints"][rd][field], rd)
    compare("terminal", later["terminal"], earlier["terminal"], stopped + 1)
    # Original execution trains every client, evaluates every detector, then
    # revokes/weights, calculates coefficients, commits history, and aggregates.
    def chronology(d):
        stage, rd, cid = d["stage"], d["round"], d["client_id"]
        ci = int(cid.split("-")[-1]) if cid else -1
        if rd is None:
            return (-1, ("data", "partition", "model_spec", "initial_model", "numerical_policy.baseline").index(stage), 0, 0)
        if stage in ("active_client_order", "client_inputs_and_reconstructed_order", "actual_batch_coverage", "client_update", "local_training_statistics") or stage.startswith("singleton."):
            positions = {"active_client_order": -3, "client_inputs_and_reconstructed_order": -2, "actual_batch_coverage": -1,
                         "singleton.coverage": 0, "client_update": 100000, "local_training_statistics": 100001}
            pos = positions.get(stage)
            if pos is None:
                pos = 1 + (d["batch"] or 0) * 20 + (*tail.step.STAGES, "parameter_layout").index(stage.split(".", 1)[1])
            return (rd, 0, ci, pos)
        if stage == "blacklisted_before":
            return (rd, -1, 0, 0)
        if stage in ("candidate_order", "history_event_order"):
            return (rd, 1, 0, 0)
        if stage == "verified_order":
            return (rd, 2, 10**9, 0)
        if stage.startswith("detector.") and stage != "detector.forget":
            return (rd, 2, ci, ("before_evaluate", "after_evaluate", "decision").index(stage.split(".", 1)[1]))
        if stage == "detector.forget":
            return (rd, 3, ci, 0)
        if stage == "blacklisted_after":
            return (rd, 3, 10**9, 0)
        if stage == "coefficients":
            return (rd, 4, 0, 0)
        if stage.startswith("commit.") or stage in ("detector_diagnostics", "history_admitted"):
            return (rd, 5, ci, 0 if stage == "commit.before" else 1)
        if stage == "aggregate_order":
            return (rd, 6, -1, "")
        if stage.startswith("evaluation_"):
            return (rd, 7, 0, stage)
        if stage.startswith("checkpoint.") or stage == "terminal":
            return (rd, 8, 0, stage)
        return (rd, 6, 0, stage)
    differences.sort(key=chronology)
    pre = [d for d in differences if d["scope"] == "common_precondition"]
    intended = [d for d in differences if d["scope"] == "intended_history_intervention"]
    round26_local = [d for d in differences if d["scope"] == "round26_local_common_condition"]
    post_intervention = [d for d in differences if d["scope"] == "post_intervention_divergence"]
    counts = Counter()
    for d in differences:
        counts[d["stage"]] += d["different_fields"]
    return {"available": True, "equal": not differences, "first_divergence": differences[0] if differences else None,
        "differing_boundaries": len(differences), "different_fields_by_stage": dict(counts), "compared_through_round": stopped,
        "full_thirty_round_coverage": stopped == ROUNDS,
        "common_through_round24_and_round25_precommit_conditions_equal": not pre if cross_arm and stopped >= FREEZE_START else None,
        "first_common_precondition_mismatch": pre[0] if pre else None,
        "round26_local_training_equal_before_first_affected_detection": not round26_local if cross_arm and stopped >= FREEZE_START + 1 else None,
        "first_round26_local_training_mismatch": round26_local[0] if round26_local else None,
        "round25_intended_history_differing_boundaries": len(intended),
        "post_intervention_differing_boundaries": len(post_intervention),
        "first_post_intervention_divergence": post_intervention[0] if post_intervention else None,
        "paired_round25_h0_admissions": sum(e["commit"] is not None and e["commit"]["admitted"]
            for e in earlier["history_events"] if e["round"] == FREEZE_START) if cross_arm else None,
        "paired_round25_h1_admissions": sum(e["commit"] is not None and e["commit"]["admitted"]
            for e in later["history_events"] if e["round"] == FREEZE_START) if cross_arm else None,
        "numerical_magnitudes_available": False}


def mechanism_diagnostics(payload, task):
    """Use the existing raw-threshold trigger and permanent-revocation taxonomy."""
    result = _result(payload, task)
    diagnostics = result.diagnostics
    by_client = forensic._validate_details(result)
    records = {r.round: r for r in result.records}
    revocations, consistency = forensic._revocations(result, records, by_client)
    health = prefix.timing.base.metrics(result)
    health.pop("runtime_seconds", None)
    health.update(scope="observed clean diagnostic only; thirty rounds do not certify full-protocol health",
                  original_honest=task["config"]["num_clients"], FP=records[result.stopped_round].false_positive_revocations)
    def window(name, start, end):
        available = [row for row in payload["rounds"] if start <= row["round"] <= end]
        ds = [d for d in diagnostics if start <= d.round <= end]
        events = [e for e in payload["history_events"] if start <= e["round"] <= end]
        commits = [e["commit"] for e in events if e["commit"] is not None]
        rev = [d for d in ds if d.revoked]
        paths = dict.fromkeys(forensic.PATHS, 0)
        for d in rev:
            immediate = d.immediate_revocation
            counted = d.count_increment and d.count_after >= result.config.suspicion_remove_after
            paths["both" if immediate and counted else "immediate_only" if immediate else "count_only" if counted else "neither"] += 1
        active = sum(task["config"]["num_clients"] - len(r["blacklisted_before"]) for r in available)
        count = len(ds)
        return {"name": name, "start": start, "end": end, "requested_rounds": end - start + 1,
            "available_rounds": len(available), "complete": len(available) == end - start + 1,
            "active_client_rounds": active, "trained_client_rounds": sum(len(r["clients"]) for r in available),
            "observed_verified_finite_updates": count, "unobserved_active_client_rounds": active - count,
            "accepted": sum(d.aggregation_accepted for d in ds), "history_eligible": sum(d.history_eligible for d in ds),
            "history_admitted": sum(d.history_admitted for d in ds),
            "history_admission_rate": sum(d.history_admitted for d in ds) / count if count else None,
            "suspicious": sum(d.suspicious for d in ds), "count_increment": sum(d.count_increment for d in ds),
            "revoked": len(rev), "triggers": forensic._triggers(ds, result.config), "revocation_routes": paths,
            "decision_reasons": dict(sorted(Counter(d.decision_reason for d in ds).items())),
            "actual_commit_calls": len(commits), "requested_admission_calls": sum(c["requested_admission"] for c in commits),
            "admitted_commit_calls": sum(c["admitted"] for c in commits),
            "eligible_requested_not_admitted": sum(e["decision"]["history_eligible"] and e["commit"]["requested_admission"]
                and not e["commit"]["admitted"] for e in events if e["commit"] is not None),
            "unchanged_history_and_live_normal_commits": sum(all(c["before"][f] == c["after"][f]
                for f in ("history", "history_size", "normal")) for c in commits),
            "revoked_without_commit": sum(e["forget"] is not None and e["commit"] is None for e in events)}
    return {"health": health, "per_round": [window("round_" + str(rd), rd, rd) for rd in range(1, result.stopped_round + 1)],
        "phases": [window(*phase) for phase in PHASES], "revocations": revocations["honest"],
        "revocation_consistency": consistency, "history_snapshots_validated": True,
        "history_frozen_field_is_original_weight_guard": True}


def _limits():
    return {"short_clean_health_assessed": True, "full_protocol_health_assessed": False,
        "performance_improvement_assessed": False, "formal_qualification_assessed": False,
        "automatic_next_stage": False, "operator_cause_identified": False,
        "limitation": "one clean Dirichlet configuration on one physical GPU, two fresh repeats per arm and at most thirty rounds; round26 local training remains a required common boundary before the first affected detector evaluation; later divergence is compatible with propagation but is not itself proof of a history effect. No attacks, full-run performance, operator identity or formal qualification is established"}


def decision(rows, pairs, ready):
    full = ready and all(row["completed_training_rounds"] == ROUNDS for row in rows)
    within = [p for p in pairs if p["kind"] == "within_arm"]
    cross = [p for p in pairs if p["kind"] == "cross_arm"]
    stable = ready and all(p["equal"] for p in within)
    pre25 = ready and all(p["common_through_round24_and_round25_precommit_conditions_equal"] is True for p in cross)
    local26 = ready and all(p["round26_local_training_equal_before_first_affected_detection"] is True for p in cross)
    pre = pre25 and local26
    historical = ready and all(row["historical_context"]["equal"] for row in rows)
    if not ready:
        conclusion, action = "incomplete_or_invalid_evidence", "resolve_missing_or_invalid_evidence"
    elif not full:
        conclusion, action = "completed_early_health_failure_without_full_thirty_round_comparison", "review_short_clean_health_failure"
    elif not historical or not pre:
        conclusion, action = "common_conditions_differ_no_paired_causal_interpretation", "review_first_common_condition_mismatch"
    elif not stable:
        conclusion, action = "within_arm_repeat_divergence", "review_first_repeat_divergence"
    elif not any(p["paired_round25_h0_admissions"] for p in cross):
        conclusion, action = "paired_common_conditions_equal_without_round25_admission_opportunity", "review_unexercised_first_freeze_boundary"
    else:
        conclusion, action = "paired_clean_mechanism_evidence_ready_for_review", "review_clean_history_effect_before_authorizing_next_stage"
    return {"conclusion": conclusion, "action": action, "full_thirty_round_coverage": full,
        "within_arm_all_observations_equal": stable if ready else None,
        "paired_preintervention_conditions_equal": pre if ready else None,
        "round26_local_training_equal_before_first_affected_detection": local26 if ready else None,
        "three_round_scoped_history_reproduced": historical if ready else None,
        "short_clean_healthy_tasks": sum(row.get("diagnostics", {}).get("health", {}).get("healthy") is True for row in rows),
        **_limits(), "short_clean_health_assessed": any("diagnostics" in row for row in rows)}


def historical_context(payload, reference):
    from copy import deepcopy
    import cifar_deterministic_prefix_report as deterministic
    if payload["terminal"]["stopped_round"] < 3:
        return {"available": False, "equal": False, "comparisons": [], "reason": "fewer_than_three_observed_rounds"}
    clipped = deepcopy(payload)
    for name in ("rounds", "evaluations", "checkpoints", "actual_batches", "singleton_batches"):
        clipped[name] = [row for row in clipped[name] if row["round"] <= 3]
    for row in clipped["rounds"]:
        for client in row["clients"]:
            client.pop("raw_update", None)
            client.pop("update_finite", None)
    anchors = [a for a in reference["deterministic_prefix_anchors"]
               if a["task"]["policy"] == "singleton_backward_cudnn_deterministic"]
    _require(len(anchors) == 3, "three historical deterministic policy anchors required")
    comparisons = []
    for anchor in anchors:
        try:
            value = deterministic.compare_observations(clipped, anchor["observations"])
            comparison = {"equal": value["equal"], "first_divergence": value["first_divergence"]}
        except (ValueError, KeyError, StopIteration) as exc:
            comparison = {"equal": False, "first_divergence": None, "comparison_error": str(exc)}
        comparisons.append({"reference_task_id": anchor["task"]["task_id"], **comparison})
    return {"available": True, "equal": all(c["equal"] for c in comparisons), "comparisons": comparisons,
        "scope": "original observed initialization and rounds0..3; new history-state snapshots were absent from the old prefix"}


def summarize(output):
    import cifar_mechanism_protocol as protocol
    output = Path(output).resolve()
    def snapshot():
        return {"sources": protocol.source_hashes(), "evidence": protocol.evidence_hashes(output)}
    try:
        before = snapshot()
        manifest, tasks = protocol.read_study(output, current_sources=True)
        _require([(t["arm"], t["repeat"]) for t in tasks] == [(arm, repeat) for repeat in (1, 2) for arm in ARMS]
                 and len({t["task_id"] for t in tasks}) == 4, "four interleaved arm/repeat tasks required")
        rows, payloads = [], {}
        for task in tasks:
            row = {"task_id": task["task_id"], "task_fingerprint": task["fingerprint"], "arm": task["arm"],
                "repeat": task["repeat"], "policy": task["policy"], "status": "missing"}
            try:
                artifact = protocol.load_completed(output, task)
                if artifact is not None:
                    _require(artifact.get("status") == "complete" and artifact.get("task_fingerprint") == task["fingerprint"], "artifact identity differs")
                    _require(artifact.get("fresh_start") is True and artifact.get("checkpoints_used") is False, "fresh execution evidence missing")
                    _require(artifact.get("gpu_uuid") == manifest["same_gpu_uuid"], "physical GPU differs")
                    _require(prefix.timing.finite(artifact.get("wall_seconds")) and artifact["wall_seconds"] >= 0, "invalid worker walltime")
                    _require(prefix.matched.normalized_environment(artifact["environment"]) == manifest["reference"]["execution_environment"], "numerical environment differs")
                    payload = validate_observations(artifact["observations"], task)
                    validate_environment_policy(payload, artifact["environment"])
                    stopped = payload["terminal"]["stopped_round"]
                    _require(type(artifact.get("completed_training_rounds")) is int and artifact["completed_training_rounds"] == stopped,
                             "artifact round count differs from observed terminal execution")
                    _require(type(artifact.get("requested_training_rounds")) is int and artifact["requested_training_rounds"] == ROUNDS, "artifact requested horizon differs")
                    row.update(status="complete", completed_training_rounds=stopped, requested_training_rounds=ROUNDS,
                        artifact_fingerprint=artifact["artifact_fingerprint"], wall_seconds=artifact["wall_seconds"],
                        observed_client_rounds=sum(len(r["clients"]) for r in payload["rounds"]),
                        actual_singleton_events=len(payload["singleton_batches"]), numerical_policy=payload["numerical_policy"],
                        terminal=payload["terminal"], historical_context=historical_context(payload, manifest["reference"]),
                        diagnostics=mechanism_diagnostics(payload, task),
                        trajectory=[[c["round"], c["record"]["accuracy"], c["record"]["attack_target_success_rate"],
                            c["record"]["attack_target_confidence"], c["model"]["sha256"]] for c in payload["checkpoints"]])
                    payloads[task["task_id"]] = payload
            except Exception as exc:
                row.update(status="invalid_evidence", error=str(exc))
            rows.append(row)
        lookup = {(t["arm"], t["repeat"]): t["task_id"] for t in tasks}
        specifications = [("within_arm", (arm, 1), (arm, 2)) for arm in ARMS]
        specifications += [("cross_arm", ("H0", repeat), ("H1", repeat)) for repeat in (1, 2)]
        pairs = []
        for kind, earlier, later in specifications:
            a, b = lookup[later], lookup[earlier]
            pair = {"pair_id": b + "__" + a, "kind": kind, "earlier_arm": earlier[0], "earlier_repeat": earlier[1],
                "later_arm": later[0], "later_repeat": later[1], "available": False, "equal": None, "first_divergence": None}
            if a in payloads and b in payloads:
                try:
                    pair.update(compare_observations(payloads[a], payloads[b], cross_arm=kind == "cross_arm"))
                except Exception as exc:
                    pair["error"] = str(exc)
            pairs.append(pair)
        protocol.verify_reference(manifest["reference"])
        _require(before == snapshot(), "mechanism evidence/scientific sources changed during read")
        complete = sum(row["status"] == "complete" for row in rows)
        ready = complete == 4 and all(p["available"] for p in pairs)
        value = {"status": "complete" if ready else "incomplete_or_invalid_evidence", "manifest_fingerprint": manifest["fingerprint"],
            "same_gpu_uuid": manifest["same_gpu_uuid"], "source_count": len(manifest["source_sha256"]),
            "source_map_sha256": prefix.timing.base.digest(manifest["source_sha256"]),
            "source_reference_and_evidence_verified": True, "training_started_by_summary": False,
            "expected_tasks": 4, "complete_tasks": complete, "thirty_round_tasks": sum(r.get("completed_training_rounds") == ROUNDS for r in rows),
            "expected_pairs": 4, "available_pairs": sum(p["available"] for p in pairs),
            "numerical_environment": {key: manifest["reference"]["execution_environment"].get("torch", {}).get(key)
                for key in ("version", "cuda_version", "cudnn_version", *tail.ENVIRONMENT_FLAGS)},
            "rows": rows, "pairs": pairs, "decision": decision(rows, pairs, ready)}
        json.dumps(value, allow_nan=False)
        return value
    except Exception as exc:
        return {"status": "invalid", "error": str(exc), "training_started_by_summary": False}


ROUND_FIELDS = ("start", "active_client_rounds", "trained_client_rounds", "observed_verified_finite_updates",
    "unobserved_active_client_rounds", "accepted", "history_eligible", "history_admitted", "suspicious",
    "count_increment", "revoked", "actual_commit_calls", "requested_admission_calls", "admitted_commit_calls",
    "eligible_requested_not_admitted", "unchanged_history_and_live_normal_commits", "revoked_without_commit")


def print_summary(value):
    from copy import deepcopy
    profiles = []
    rows = []
    for original in value.get("rows", []):
        row = deepcopy(original)
        if row.get("status") == "complete":
            policy = row["numerical_policy"]
            if policy["baseline"] not in profiles:
                profiles.append(policy["baseline"])
            row["numerical_policy"] = {"policy": policy["policy"], "baseline_profile_index": profiles.index(policy["baseline"]),
                "actual_singleton_events": len(policy["events"]), "scoped_changes": policy["scoped_changes"],
                "checks": policy["checks"], "all_other_flags_verified_unchanged": not policy["other_flags_changed"],
                "restored_on_exit": policy["restored_on_exit"]}
            diag = row["diagnostics"]
            diag["per_round"] = [{"c": [r[f] for f in ROUND_FIELDS],
                "tr": [r["triggers"][f] for f in forensic.TRIGGERS],
                "rv": [r["revocation_routes"][f] for f in forensic.PATHS], "reasons": r["decision_reasons"]}
                for r in diag["per_round"]]
            examples = diag["revocations"].pop("examples", [])
            diag["revocations"]["earliest_honest_case"] = examples[0] if examples else None
        rows.append(row)
    records = [{"type": "header", **{k: v for k, v in value.items() if k not in ("rows", "pairs", "decision")}, "flag_profiles": profiles},
        {"type": "legend", "schema": "cifar-clean-history-mechanism-summary-v1", "round_counts": list(ROUND_FIELDS),
         "triggers": list(forensic.TRIGGERS), "revocation_routes": list(forensic.PATHS),
         "trigger_semantics": "strict raw novelty>warning and drift>h; first four partition observations; strong is overlapping novelty>reject",
         "trajectory_order": ["round", "accuracy", "clean_source_to_target_background_rate", "target_confidence", "global_model_sha256"],
         "comparison_scope": "all common rounds0..24 and round25 training/detection/aggregation/model/evaluation must match; only round25 admitted/history/live-normal commit changes are intended; history_frozen retains its original weight-guard meaning; round26 local inputs/batches/gradients/updates/statistics must still match before detector26; later differences are post-intervention observations, not proof of propagation",
         "denominators": "actual available rounds and nonrevoked starting population; absent rounds and zero-observation rates are never filled with zero",
         "numerical_distance": "hash inequality locates observed boundaries without a distance or kernel attribution"}]
    records.extend({"type": "task", **row} for row in rows)
    records.extend({"type": "pair", **pair} for pair in value.get("pairs", []))
    if "decision" in value:
        records.append({"type": "decision", **value["decision"]})
    lines = [json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) for row in records]
    print("=== CIFAR_MECHANISM_BEGIN ===", flush=True)
    for line in lines:
        print(line)
    print("=== CIFAR_MECHANISM_END ===", flush=True)
