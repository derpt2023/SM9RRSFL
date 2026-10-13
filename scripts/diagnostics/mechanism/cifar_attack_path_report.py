"""Read-only full-chain evidence for a bounded alternating-attack path probe.

The clean singleton policy is applied only to ordinary local training. Actual
attacker benign-reference, stealth and target backwards retain baseline flags.
Hashes describe observed boundaries; they do not identify an executed kernel
or provide numerical distances. Summary never imports Torch or trains.
"""
from collections import Counter
from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace

import cifar_prefix_probe_report as prefix
import cifar_history_forensics as forensic
import cifar_tail_determinism_report as tail

ROUNDS = 27
ATTACK_START = 25
POLICY = "clean_singleton"
SCHEMA = "cifar-attack-path-observation-v1"
ATTACK_STAGES = ("benign_reference", "stealth_benign", "target_adversarial")
BATCH_STAGES = ("local_indices", "dataset_indices", "features", "labels", "pre_parameters",
                "logits", "loss", "gradients", "post_parameters")
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
    from cifar_attack_path_runtime import validate_policy_observation
    from cifar_mechanism_observer import validate_history_observation
    _require(isinstance(payload, dict) and payload.get("schema") == SCHEMA, "wrong attack-path observation schema")
    _require(payload.get("task_id") == task["task_id"] and payload.get("task_fingerprint") == task["fingerprint"], "observation task identity differs")
    _require(prefix._finite_tree(payload), "nonfinite attack-path observation")
    config = task["config"]
    _require(config["rounds"] == ROUNDS and 0 < config["malicious_ratio"] < 1 and config["method"] == "sm9rrs"
             and config["attack_start_round"] == ATTACK_START and config["attack"] == "alternating_minimization",
             "requires declared twenty-seven-round alternating-attack Ours probe")
    _require({k: v for k, v in payload.get("configuration", {}).items() if k != "device"} ==
             {k: v for k, v in config.items() if k != "device"}, "scientific configuration differs")
    terminal = payload.get("terminal", {})
    stopped = terminal.get("stopped_round")
    _require(type(stopped) is int and 1 <= stopped <= ROUNDS and terminal.get("requested_rounds") == ROUNDS,
             "invalid terminal round coverage")
    ids = ["client-" + str(i) for i in range(config["num_clients"])]
    malicious = set(prefix.timing.base.fl._choose_malicious(ids, config["malicious_ratio"], config["seed"]))
    _require(bool(malicious) and terminal.get("malicious_clients") == sorted(malicious)
             and prefix._integer(terminal.get("nonfinite_updates")), "invalid attack terminal labels/counters")
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
            attack_active = cid in malicious and rd >= ATTACK_START
            expected_epochs = config["attack_epochs"] if attack_active else config["local_epochs"]
            for field in ("model_input", "raw_update"):
                _tensor(c.get(field), "client." + field)
                _require(c[field]["shape"] == model["shape"] and c[field]["dtype"] == model["dtype"], "client vector shape/dtype differs")
            _require(c["model_input"] == checkpoints[rd - 1]["model"], "client input differs from prior model")
            _require(c.get("samples") == samples[cid] and c.get("training_seed") == config["seed"] + rd * 1009 + int(cid.split("-")[-1]), "client samples/seed differs")
            _require(c.get("epochs") == expected_epochs and c.get("batch_size") == config["batch_size"] and
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
            _require(d["is_malicious"] is (d["client_id"] in malicious)
                     and d["attack_active"] is (d["client_id"] in malicious and rd >= ATTACK_START),
                     "diagnostic malicious or attack-active identity differs")
            _require(d["aggregation_weight"] == values[d["client_id"]], "diagnostic coefficient differs")
            if d["revoked"]:
                revoked.add(d["client_id"])
        _require(row.get("blacklisted_after") == sorted(revoked) == checkpoints[rd].get("blacklisted"), "public blacklist differs from observed revocations")
        _require(row["record"]["false_positive_revocations"] == len(revoked - malicious)
                 and row["record"]["true_positive_revocations"] == len(revoked & malicious)
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


def compare_observations(later, earlier):
    """Compare every actual boundary and retain the original temporal order."""
    differences = []

    def compare(stage, left, right, rd=None, cid=None, batch=None, position=None):
        leaves = list(prefix._leaf_differences(left, right))
        if leaves:
            differences.append({"stage": stage, "round": rd, "client_id": cid, "batch": batch,
                "different_fields": len(leaves), "first": leaves[0], "magnitude": None,
                "scope": "common_initial_context" if rd is None else
                    "preattack_repeatability" if rd < ATTACK_START else "attack_period_repeatability",
                "_position": position})

    context_fields = ("data", "partition", "model_spec", "initial_model")
    for index, field in enumerate(context_fields):
        compare(field, later[field], earlier[field], position=(-1, index, 0, 0))
    compare("numerical_policy.baseline", later["numerical_policy"]["baseline"], earlier["numerical_policy"]["baseline"],
            position=(-1, len(context_fields), 0, 0))
    compare("malicious_client_identity", later["terminal"]["malicious_clients"], earlier["terminal"]["malicious_clients"],
            position=(-1, len(context_fields) + 1, 0, 0))
    stopped = min(later["terminal"]["stopped_round"], earlier["terminal"]["stopped_round"])
    caches = {}

    def per_round(payload, name, rd):
        key = (id(payload), name)
        if key not in caches:
            values = {}
            for row in payload.get(name, []):
                values.setdefault(row["round"], []).append(row)
            caches[key] = values
        return caches[key].get(rd, [])

    def index(rows):
        return {row["client_id"]: row for row in rows}

    for rd in range(stopped + 1):
        if rd:
            a, b = later["rounds"][rd - 1], earlier["rounds"][rd - 1]
            compare("blacklisted_before", a["blacklisted_before"], b["blacklisted_before"], rd, position=(rd, -1, 0, 0))
            compare("active_client_order", [c["client_id"] for c in a["clients"]], [c["client_id"] for c in b["clients"]],
                    rd, position=(rd, 0, -1, 0))
            left, right = index(a["clients"]), index(b["clients"])
            for cid in sorted(set(left) & set(right), key=lambda value: int(value.split("-")[-1])):
                ci = int(cid.split("-")[-1])
                x, y = left[cid], right[cid]
                excluded = ("stats", "update", "raw_update", "update_finite")
                compare("client_inputs_and_reconstructed_order", {k: v for k, v in x.items() if k not in excluded},
                        {k: v for k, v in y.items() if k not in excluded}, rd, cid, position=(rd, 0, ci, 0))
                actual_a = [row for row in per_round(later, "actual_batches", rd) if row["client_id"] == cid]
                actual_b = [row for row in per_round(earlier, "actual_batches", rd) if row["client_id"] == cid]
                compare("ordinary_batch_coverage", actual_a, actual_b, rd, cid, position=(rd, 0, ci, 1))
                calls_a = [row for row in per_round(later, "attack_calls", rd) if row["client_id"] == cid]
                calls_b = [row for row in per_round(earlier, "attack_calls", rd) if row["client_id"] == cid]
                call_excluded = ("update", "stats", "backward_counts", "completed")
                compare("attack.call_inputs", [{k: v for k, v in row.items() if k not in call_excluded} for row in calls_a],
                        [{k: v for k, v in row.items() if k not in call_excluded} for row in calls_b], rd, cid,
                        position=(rd, 0, ci, 2))
                for name, label in (("singleton_batches", "singleton"), ("attack_batches", "attack")):
                    rows_a = [row for row in per_round(later, name, rd) if row["client_id"] == cid]
                    rows_b = [row for row in per_round(earlier, name, rd) if row["client_id"] == cid]
                    identity = lambda row: (row.get("stage"), row["batch"])
                    compare(label + ".coverage", [identity(row) for row in rows_a], [identity(row) for row in rows_b],
                            rd, cid, position=(rd, 0, ci, 3))
                    map_b = {identity(row): row for row in rows_b}
                    for ordinal, row in enumerate(rows_a):
                        if identity(row) not in map_b:
                            continue
                        old = map_b[identity(row)]
                        stages = (*BATCH_STAGES, "parameter_layout") if name == "singleton_batches" else BATCH_STAGES
                        stage_label = label + ("." + row["stage"] if label == "attack" else "")
                        for field_index, field in enumerate(stages):
                            compare(stage_label + "." + field, row.get(field), old.get(field), rd, cid, row["batch"],
                                    position=(rd, 0, ci, 4 + ordinal * 20 + field_index))
                compare("attack.call_result", [{k: row.get(k) for k in call_excluded} for row in calls_a],
                        [{k: row.get(k) for k in call_excluded} for row in calls_b], rd, cid,
                        position=(rd, 0, ci, 10**8))
                compare("client_update", {k: x.get(k) for k in ("raw_update", "update_finite", "update")},
                        {k: y.get(k) for k in ("raw_update", "update_finite", "update")}, rd, cid,
                        position=(rd, 0, ci, 10**8 + 1))
                compare("local_training_statistics", x["stats"], y["stats"], rd, cid, position=(rd, 0, ci, 10**8 + 2))
            compare("candidate_order", a["candidate_order"], b["candidate_order"], rd, position=(rd, 1, 0, 0))
            events_a, events_b = per_round(later, "history_events", rd), per_round(earlier, "history_events", rd)
            compare("history_event_order", [e["client_id"] for e in events_a], [e["client_id"] for e in events_b],
                    rd, position=(rd, 1, 0, 1))
            left, right = index(events_a), index(events_b)
            for cid in sorted(set(left) & set(right), key=lambda value: int(value.split("-")[-1])):
                ci, x, y = int(cid.split("-")[-1]), left[cid], right[cid]
                for field_index, field in enumerate(("before_evaluate", "after_evaluate", "decision")):
                    compare("detector." + field, x[field], y[field], rd, cid, position=(rd, 2, ci, field_index))
                compare("detector.forget", x["forget"], y["forget"], rd, cid, position=(rd, 3, ci, 0))
                compare("history_commit", x["commit"], y["commit"], rd, cid, position=(rd, 5, ci, 0))
            compare("verified_order", a["verified_order"], b["verified_order"], rd, position=(rd, 2, 10**9, 0))
            left, right = index(a["diagnostics"]), index(b["diagnostics"])
            for cid in sorted(set(left) & set(right), key=lambda value: int(value.split("-")[-1])):
                compare("detector_diagnostics", left[cid], right[cid], rd, cid,
                        position=(rd, 3, int(cid.split("-")[-1]), 1))
            compare("blacklisted_after", a["blacklisted_after"], b["blacklisted_after"], rd, position=(rd, 3, 10**9, 0))
            compare("coefficients", a["coefficients"], b["coefficients"], rd, position=(rd, 4, 0, 0))
            for field_index, field in enumerate(("aggregate_order", "aggregation_executed", "aggregate", "post_model")):
                compare(field, a[field], b[field], rd, position=(rd, 6, 0, field_index))
        for kind_index, kind in enumerate(("accuracy", "target")):
            x = next(e for e in later["evaluations"] if e["round"] == rd and e["kind"] == kind)
            y = next(e for e in earlier["evaluations"] if e["round"] == rd and e["kind"] == kind)
            compare("evaluation_" + kind, x, y, rd, position=(rd, 7, 0, kind_index))
        for field_index, field in enumerate(("record", "model", "diagnostic_count", "blacklisted")):
            compare("checkpoint." + field, later["checkpoints"][rd][field], earlier["checkpoints"][rd][field],
                    rd, position=(rd, 8, 0, field_index))
    compare("terminal", later["terminal"], earlier["terminal"], stopped + 1, position=(stopped + 1, 9, 0, 0))
    differences.sort(key=lambda value: value.pop("_position"))
    warmup = [d for d in differences if d["round"] is None or d["round"] < ATTACK_START]
    attacks = [d for d in differences if d["round"] is not None and ATTACK_START <= d["round"] <= stopped]
    return {"available": True, "equal": not differences,
        "first_divergence": differences[0] if differences else None,
        "compared_through_round": stopped, "differing_boundaries": len(differences),
        "different_fields_by_stage": dict(Counter({stage: sum(d["different_fields"] for d in differences if d["stage"] == stage)
            for stage in sorted({d["stage"] for d in differences})})),
        "preattack_rounds0_24_equal": not warmup and stopped >= ATTACK_START - 1,
        "first_preattack_divergence": warmup[0] if warmup else None,
        "attack_rounds25_27_equal": not attacks and stopped == ROUNDS,
        "first_attack_period_divergence": attacks[0] if attacks else None,
        "attack_period_differing_boundaries": len(attacks),
        "numerical_magnitudes_available": False,
        "later_round_input_differences_may_be_propagation": True}


def path_diagnostics(payload, task):
    """Retain actual populations, calls and backwards separately for every round."""
    config, terminal = task["config"], payload["terminal"]
    malicious = set(terminal["malicious_clients"])
    ordinary = Counter(row["round"] for row in payload["actual_batches"])
    singletons = Counter(row["round"] for row in payload["singleton_batches"])
    calls = {}
    batches = {}
    for row in payload["attack_calls"]:
        calls.setdefault(row["round"], []).append(row)
    for row in payload["attack_batches"]:
        batches.setdefault(row["round"], []).append(row)
    rounds = []
    for rd in range(1, ROUNDS + 1):
        actual = payload["rounds"][rd - 1] if rd <= terminal["stopped_round"] else None
        if actual is None:
            rounds.append({"round": rd, "available": False,
                "nominal_client_rounds": config["num_clients"], "active_client_rounds": None,
                "permanently_excluded_client_rounds": None, "observed_finite_updates": None,
                "nonfinite_updates": None, "original_malicious_clients": len(malicious),
                "active_malicious_clients": None, "actual_attack_calls": None,
                "attack_backward_counts": None, "ordinary_local_calls": None, "ordinary_singleton_backwards": None})
            continue
        active = actual["clients"]
        attacks = calls.get(rd, [])
        ds = actual["diagnostics"]
        rounds.append({"round": rd, "available": True, "nominal_client_rounds": config["num_clients"],
            "active_client_rounds": len(active),
            "permanently_excluded_client_rounds": len(actual["blacklisted_before"]),
            "observed_finite_updates": sum(c["update_finite"] for c in active),
            "nonfinite_updates": sum(not c["update_finite"] for c in active),
            "original_malicious_clients": len(malicious),
            "active_malicious_clients": sum(c["client_id"] in malicious for c in active),
            "actual_attack_calls": len(attacks), "attack_clients": [c["client_id"] for c in attacks],
            "attack_backward_counts": {stage: sum(b["stage"] == stage for b in batches.get(rd, [])) for stage in ATTACK_STAGES},
            "ordinary_local_calls": ordinary[rd], "ordinary_singleton_backwards": singletons[rd],
            "accepted_finite_updates": sum(d["aggregation_accepted"] for d in ds),
            "malicious_accepted": sum(d["is_malicious"] and d["aggregation_accepted"] for d in ds),
            "malicious_history_admitted": sum(d["is_malicious"] and d["history_admitted"] for d in ds),
            "honest_accepted": sum(not d["is_malicious"] and d["aggregation_accepted"] for d in ds),
            "new_honest_revocations": sum(not d["is_malicious"] and d["revoked"] for d in ds),
            "new_malicious_revocations": sum(d["is_malicious"] and d["revoked"] for d in ds),
            "actual_attack_call_details": [{k: c[k] for k in ("client_id", "samples", "target_samples", "epochs",
                "batch_size", "stealth_steps", "backward_counts")} for c in attacks]})
    return {"per_round": rounds, "actual_attack_calls": len(payload["attack_calls"]),
        "actual_attack_backwards": len(payload["attack_batches"]),
        "stage_backward_counts": {stage: sum(b["stage"] == stage for b in payload["attack_batches"]) for stage in ATTACK_STAGES},
        "nonfinite_updates": terminal["nonfinite_updates"],
        "nonfinite_loss_batches": sum(b.get("loss", {}).get("nonfinite", False) for b in payload["attack_batches"]),
        "final_honest_revocations": len(set(terminal["blacklisted_clients"]) - malicious),
        "final_malicious_revocations": len(set(terminal["blacklisted_clients"]) & malicious),
        "denominator": "actual active client calls and their original complete backwards; revoked clients are excluded only in later rounds; missing requested rounds remain unavailable"}


def decision(rows, pairs, ready):
    full = ready and all(row.get("completed_training_rounds") == ROUNDS for row in rows)
    pair = pairs[0] if pairs else {}
    warmup = ready and pair.get("preattack_rounds0_24_equal") is True
    stable = ready and pair.get("equal") is True
    attacked = ready and all(row.get("path_diagnostics", {}).get("actual_attack_calls", 0) > 0 for row in rows)
    nonfinite = ready and any(row["terminal"]["nonfinite_updates"] > 0 for row in rows)
    if not ready:
        conclusion, action = "incomplete_or_invalid_evidence", "resolve_missing_or_invalid_evidence"
    elif not full:
        conclusion, action = "early_exhaustion_without_full_attack_horizon", "review_actual_terminal_execution"
    elif nonfinite:
        conclusion, action = "observed_nonfinite_updates", "review_first_nonfinite_training_boundary"
    elif not warmup:
        conclusion, action = "preattack_repeat_divergence", "review_first_preattack_divergence"
    elif not attacked:
        conclusion, action = "no_actual_attack_calls", "review_unexercised_attack_path"
    elif not stable:
        conclusion, action = "attack_period_repeat_divergence", "review_first_actual_attack_or_pipeline_divergence"
    else:
        conclusion, action = "supports_observed_twenty_seven_round_repeatability", "review_short_attack_path_evidence_before_next_stage"
    return {"conclusion": conclusion, "action": action, "full_horizon_coverage": full,
        "preattack_rounds0_24_equal": warmup if ready else None,
        "all_observed_rounds0_27_equal": stable if ready else None,
        "actual_attack_paths_exercised": attacked if ready else None,
        "requested_training_rounds": ROUNDS, "attack_start_round": ATTACK_START,
        "health_assessed": False, "performance_improvement_assessed": False,
        "formal_qualification_assessed": False, "automatic_next_stage": False,
        "operator_cause_identified": False, "attack_backward_deterministic_policy_applied": False,
        "limitation": "two fresh H0 runs of one Dirichlet alternating-attack configuration on one physical GPU, only rounds25..27 contain attacks; clean singleton policy does not change attacker backward flags. Equality supports observed short-run repeatability, without a later-round/platform guarantee, health qualification, performance improvement, kernel attribution or automatic main/TPE/NoPermanent progression"}


def summarize(output):
    import cifar_attack_path_protocol as protocol
    output = Path(output).resolve()

    def snapshot():
        return {"sources": protocol.source_hashes(), "evidence": protocol.evidence_hashes(output)}

    try:
        before = snapshot()
        manifest, tasks = protocol.read_study(output, current_sources=True, verify_reference=True)
        _require(len(tasks) == 2 and [t["repeat"] for t in tasks] == [1, 2]
                 and all(t["policy"] == POLICY and t["arm"] == "H0" for t in tasks)
                 and len({t["task_id"] for t in tasks}) == 2, "two same-policy fresh H0 tasks required")
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
                    _require(type(artifact.get("requested_training_rounds")) is int and artifact["requested_training_rounds"] == ROUNDS,
                             "artifact requested horizon differs")
                    row.update(status="complete", completed_training_rounds=stopped, requested_training_rounds=ROUNDS,
                        artifact_fingerprint=artifact["artifact_fingerprint"], wall_seconds=artifact["wall_seconds"],
                        observed_client_rounds=sum(len(r["clients"]) for r in payload["rounds"]),
                        numerical_policy=payload["numerical_policy"], terminal=payload["terminal"],
                        path_diagnostics=path_diagnostics(payload, task),
                        trajectory=[[c["round"], c["record"]["accuracy"], c["record"]["attack_target_success_rate"],
                            c["record"]["attack_target_confidence"], c["model"]["sha256"]] for c in payload["checkpoints"]])
                    payloads[task["task_id"]] = payload
            except Exception as exc:
                row.update(status="invalid_evidence", error=str(exc))
            rows.append(row)
        first, second = tasks
        pair = {"pair_id": first["task_id"] + "__" + second["task_id"], "kind": "within_policy",
            "earlier_repeat": first["repeat"], "later_repeat": second["repeat"], "policy": POLICY,
            "available": False, "equal": None, "first_divergence": None}
        if all(task["task_id"] in payloads for task in tasks):
            pair.update(compare_observations(payloads[second["task_id"]], payloads[first["task_id"]]))
        protocol.verify_reference(manifest["reference"])
        _require(before == snapshot(), "attack evidence/scientific sources changed during read")
        complete = sum(row["status"] == "complete" for row in rows)
        ready = complete == 2 and pair["available"]
        value = {"status": "complete" if ready else "incomplete_or_invalid_evidence",
            "manifest_fingerprint": manifest["fingerprint"], "same_gpu_uuid": manifest["same_gpu_uuid"],
            "source_count": len(manifest["source_sha256"]), "source_map_sha256": prefix.timing.base.digest(manifest["source_sha256"]),
            "source_reference_and_evidence_verified": True, "training_started_by_summary": False,
            "expected_tasks": 2, "complete_tasks": complete, "requested_training_rounds": ROUNDS,
            "full_horizon_tasks": sum(r.get("completed_training_rounds") == ROUNDS for r in rows),
            "expected_pairs": 1, "available_pairs": int(pair["available"]),
            "numerical_environment": {key: manifest["reference"]["execution_environment"].get("torch", {}).get(key)
                for key in ("version", "cuda_version", "cudnn_version", *tail.ENVIRONMENT_FLAGS)},
            "rows": rows, "pairs": [pair], "decision": decision(rows, [pair], ready)}
        json.dumps(value, allow_nan=False)
        return value
    except Exception as exc:
        return {"status": "invalid", "error": str(exc), "training_started_by_summary": False}


def print_summary(value):
    """Emit six bounded JSON records for a complete two-task/one-pair result."""
    from copy import deepcopy
    profiles, rows = [], []
    for original in value.get("rows", []):
        row = deepcopy(original)
        if row.get("status") == "complete":
            policy = row["numerical_policy"]
            if policy["baseline"] not in profiles:
                profiles.append(policy["baseline"])
            row["numerical_policy"] = {"policy": policy["policy"], "baseline_profile_index": profiles.index(policy["baseline"]),
                "ordinary_singleton_events": len(policy["events"]), "attack_events": len(policy["attack_events"]),
                "scoped_changes": policy["scoped_changes"], "checks": policy["checks"],
                "all_other_flags_verified_unchanged": policy["other_flags_changed"] is False,
                "restored_on_exit": policy["restored_on_exit"], "attacker_flags_at_baseline": True}
        rows.append(row)
    records = [{"type": "header", **{k: v for k, v in value.items() if k not in ("rows", "pairs", "decision")},
                "flag_profiles": profiles},
        {"type": "legend", "schema": "cifar-attack-path-summary-v1", "attack_stages": list(ATTACK_STAGES),
         "trajectory_order": ["round", "accuracy", "source_to_target_rate", "target_confidence", "global_model_sha256"],
         "policy_scope": "ordinary actual singleton local backwards only; every attacker benign-reference, stealth-benign and target-adversarial backward retains all baseline flags",
         "comparison_scope": "complete chronological pipeline, warmup0..24 separate from attack25..27; actual attacker pre/logits/loss/gradients/post precede resulting client updates",
         "denominators": "actual active calls, actual stage backwards and finite detector observations; revoked clients reduce later active counts; missing requested rounds and rates are never replaced with zero",
         "numerical_distance": "hash inequality locates observed boundaries, without a numerical distance or kernel attribution"}]
    records.extend({"type": "task", **row} for row in rows)
    records.extend({"type": "pair", **pair} for pair in value.get("pairs", []))
    if "decision" in value:
        records.append({"type": "decision", **value["decision"]})
    print("=== CIFAR_ATTACK_PATH_BEGIN ===", flush=True)
    for record in records:
        print(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    print("=== CIFAR_ATTACK_PATH_END ===", flush=True)
