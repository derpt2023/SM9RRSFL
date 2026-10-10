"""Public fingerprints of clean local training and history-detector boundaries.

No numerical policy, RNG state, update, detector, or aggregation rule is changed.
Hashing resident tensors adds device-to-host synchronization and can change
execution timing; an observed repeat is not proof of an unobserved execution.
Only compact public scientific observations leave this context, never keys,
cryptographic state, task tags, checkpoints, or complete model/update vectors.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import re

import numpy as np

from sm9rrsfl import fl
from sm9rrsfl import torch_backend as backend
from sm9rrsfl.svd_detector import LongitudinalSVDDetector


SCHEMA = "cifar-mechanism-observation-v1"
_active = False


def tensor_fingerprint(value):
    """Hash dtype, logical shape, and exact contiguous bytes without rounding."""
    if type(value).__module__.startswith("torch"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("object arrays are not scientific tensor evidence")
    descriptor = {"dtype": array.dtype.str, "shape": list(array.shape)}
    digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode())
    data = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
    for start in range(0, data.size, 8 * 1024 * 1024):
        digest.update(memoryview(data[start:start + 8 * 1024 * 1024]))
    return {"sha256": digest.hexdigest(), **descriptor}


def _scientific_diagnostic(diagnostic):
    return {key: value for key, value in asdict(diagnostic).items() if key != "task_tag"}


def _state(state):
    """Only hashes of public detector arrays and scientific scalars leave the worker."""
    if state is None:
        return None
    def normal(model):
        return None if model is None else {k: tensor_fingerprint(getattr(model, k))
            for k in ("location", "scale", "centers", "radii")}
    pending = state.pending
    return {"history": tensor_fingerprint(np.stack(state.history)) if state.history else None,
        "history_size": len(state.history), "normal": normal(state.normal), "anchor": normal(state.anchor),
        "norm_limit": float(state.norm_limit), "drift": float(state.drift),
        "clean_streak": state.clean_streak, "recovery_streak": state.recovery_streak,
        "last_round": state.last_round,
        "pending": None if pending is None else {"round": pending[0],
            "feature": tensor_fingerprint(pending[1]), "norm": float(pending[2]), "decision": asdict(pending[3])}}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_state(value):
    if value is None:
        return
    _require(isinstance(value, dict) and set(value) == {"history", "history_size", "normal", "anchor",
        "norm_limit", "drift", "clean_streak", "recovery_streak", "last_round", "pending"}, "invalid detector state")
    def fingerprint(fp):
        _require(isinstance(fp, dict) and set(fp) == {"sha256", "dtype", "shape"}
            and isinstance(fp["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", fp["sha256"]) is not None
            and isinstance(fp["dtype"], str) and isinstance(fp["shape"], list)
            and all(type(n) is int and n >= 0 for n in fp["shape"]), "invalid detector array fingerprint")
    for key in ("history_size", "clean_streak", "recovery_streak", "last_round"):
        _require(type(value[key]) is int and value[key] >= 0, "invalid detector state counter")
    for key in ("norm_limit", "drift"):
        _require(type(value[key]) in (int, float) and math.isfinite(value[key]) and value[key] >= 0,
                 "invalid detector state scalar")
    if value["history_size"]:
        fingerprint(value["history"])
        _require(len(value["history"]["shape"]) == 2 and value["history"]["shape"][0] == value["history_size"],
                 "history array count differs")
    else:
        _require(value["history"] is None, "empty history carries an array")
    for key in ("normal", "anchor"):
        if value[key] is not None:
            _require(isinstance(value[key], dict) and set(value[key]) == {"location", "scale", "centers", "radii"},
                     "invalid normal model")
            for fp in value[key].values():
                fingerprint(fp)
    pending = value["pending"]
    if pending is not None:
        _require(isinstance(pending, dict) and set(pending) == {"round", "feature", "norm", "decision"}
            and type(pending["round"]) is int and pending["round"] > value["last_round"]
            and type(pending["norm"]) in (int, float) and math.isfinite(pending["norm"])
            and pending["norm"] >= 0 and isinstance(pending["decision"], dict), "invalid pending observation")
        fingerprint(pending["feature"])


def validate_history_observation(payload, task):
    """Verify observation continuity, actual commit outcomes and frozen array state."""
    rows = payload.get("rounds", [])
    expected = [(row["round"], d["client_id"]) for row in rows for d in row["diagnostics"]]
    events = payload.get("history_events")
    _require(isinstance(events, list) and [(e.get("round"), e.get("client_id")) for e in events] == expected,
             "history event coverage differs from verified client diagnostics")
    diagnostics = {(row["round"], d["client_id"]): d for row in rows for d in row["diagnostics"]}
    _require(len(diagnostics) == len(expected), "duplicate diagnostic identities")
    previous = {}
    freeze = task.get("history_freeze_start_round")
    _require((task.get("arm") == "H0" and freeze is None and task["candidate"]["variant"] == "original")
        or (task.get("arm") == "H1" and type(freeze) is int
            and task["config"]["detector_window"] < freeze <= task["config"]["rounds"]
            and task["candidate"]["variant"] == "Ours-FrozenHistory-v1"), "invalid declared history variant")
    stable = ("history", "history_size", "normal", "anchor", "norm_limit")
    for event in events:
        rd, cid = event["round"], event["client_id"]
        d = diagnostics[(rd, cid)]
        before, evaluated = event["before_evaluate"], event["after_evaluate"]
        _validate_state(before)
        _validate_state(evaluated)
        _require(before == previous.get(cid), "detector state changed between observed rounds")
        _require(evaluated is not None and evaluated["pending"] is not None
            and evaluated["pending"]["round"] == rd and evaluated["pending"]["decision"] == event["decision"],
            "detector decision differs from pending state")
        if before is not None:
            _require(all(before[k] == evaluated[k] for k in stable)
                and before["last_round"] == evaluated["last_round"], "evaluate changed trusted history or model")
        commit, forgotten = event["commit"], event["forget"]
        _require((commit is None) != (forgotten is None), "evaluation must be committed or explicitly forgotten")
        if forgotten is not None:
            _require(d["revoked"] is True and forgotten == {"before": evaluated, "after": None}
                and d["history_admitted"] is False, "invalid revoked detector-state deletion")
            previous[cid] = None
            continue
        _require(isinstance(commit, dict) and set(commit) == {"requested_admission", "admitted", "before", "after"}
            and type(commit["requested_admission"]) is bool and type(commit["admitted"]) is bool
            and commit["before"] == evaluated and d["revoked"] is False, "invalid history commit boundary")
        after = commit["after"]
        _validate_state(after)
        _require(after is not None and after["pending"] is None and after["last_round"] == rd,
                 "history commit failed to finalize pending state")
        requested = d["aggregation_weight"] > 0 and not d["history_frozen"] and not d["trace_pending"]
        decision = event["decision"]
        original_admission = requested and decision["history_eligible"] and decision["accepted"]
        is_frozen = freeze is not None and rd >= freeze
        expected_admission = False if is_frozen else original_admission
        _require(commit["requested_admission"] == requested
            and commit["admitted"] == d["history_admitted"] == expected_admission, "actual history admission violates variant")
        if not expected_admission:
            _require(all(after[k] == evaluated[k] for k in stable), "non-admitted commit changed trusted history/model")
        if before is not None and before["anchor"] is not None:
            _require(after["anchor"] == before["anchor"] and after["norm_limit"] == before["norm_limit"],
                     "immutable anchor or norm limit changed")
        _require(after["drift"] == evaluated["drift"] and after["recovery_streak"] == evaluated["recovery_streak"]
            and after["clean_streak"] == (evaluated["clean_streak"] if requested else 0),
            "commit changed original drift/streak semantics")
        previous[cid] = after
    return events


class MechanismObserver:
    def __init__(self, task):
        self.task = task
        self.payload = {"schema": SCHEMA, "task_id": task["task_id"],
            "task_fingerprint": task["fingerprint"], "data": {}, "model_spec": None,
            "partition": None, "initial_model": None, "rounds": [], "evaluations": [],
            "checkpoints": [], "history_events": [], "observation_contract": {
                "actual_values": "model inputs, returned client updates, coefficients, aggregates, models, and original evaluation logits",
                "epoch_indices": "reconstructed global sample-index permutations from the original per-client seed using a separate local NumPy Generator",
                "predictions": "CPU argmax of the captured original logits; no second model forward is executed",
                "synchronization": "fingerprinting CUDA tensors adds host synchronization and may alter timing or scheduling",
                "policy": "no numerical flags, random seeds, model values, or scientific rules are set or changed",
                "scope": "fresh clean history-mechanism diagnostic; no formal qualification or algorithm selection"}}
        self._round = 0
        self._rounds = {}
        self._checkpoints = {}
        self._tags = {}
        self._evaluation = None
        self._started = False
        self._finished = False
        self._candidate_ids = {}
        self._event_by_tag = {}
        self._pending_update = None

    def _round_row(self, round_id=None):
        rd = self._round if round_id is None else round_id
        if rd not in self._rounds:
            self._rounds[rd] = {"round": rd, "clients": []}
        return self._rounds[rd]

    def checkpoint(self, state):
        """Read only the public round state; never serialize crypto/checkpoint state."""
        rd = int(state["completed_round"])
        records = state.get("records", [])
        if not records or records[-1].round != rd:
            raise ValueError("checkpoint has no matching scientific round record")
        row = {"round": rd, "model": tensor_fingerprint(state["params"]),
            "record": asdict(records[-1]),
            "diagnostic_count": sum(d.round == rd for d in state.get("diagnostics", [])),
            "blacklisted": sorted(state.get("blacklisted", ())) }
        previous = self._checkpoints.get(rd)
        if previous is not None:
            if {k: v for k, v in previous.items() if k != "callback_count"} != row:
                raise ValueError("repeated checkpoint changed the observed scientific round")
            previous["callback_count"] += 1
        else:
            self._checkpoints[rd] = {**row, "callback_count": 1}

    def finish(self, result):
        """Retain actual execution, including a proved terminal health failure."""
        if self._finished or not self._started:
            raise ValueError("observer requires one fresh completed execution")
        stop, horizon = result.stopped_round, self.task["config"]["rounds"]
        all_revoked = len(result.blacklisted_clients) == result.config.num_clients
        if (not 1 <= stop <= horizon or result.config.rounds != horizon
                or (stop < horizon and not all_revoked)):
            raise ValueError("unexplained incomplete mechanism execution")
        if set(self._rounds) != set(range(1, stop + 1)) or set(self._checkpoints) != set(range(stop + 1)):
            raise ValueError("missing actual rounds or checkpoint observations")
        if [(r["round"], r["kind"]) for r in self.payload["evaluations"]] != [
                (rd, kind) for rd in range(stop + 1) for kind in ("accuracy", "target")]:
            raise ValueError("missing or repeated original evaluation calls")
        records = {record.round: record for record in result.records}
        if len(records) != len(result.records) or set(records) != set(range(stop + 1)):
            raise ValueError("result records do not cover actual execution")
        identities = [c["client_id"] for c in self.payload["partition"]["clients"]]
        for rd in range(1, stop + 1):
            row = self._rounds[rd]
            previous, current = self._checkpoints[rd - 1], self._checkpoints[rd]
            active = [cid for cid in identities if cid not in previous["blacklisted"]]
            if [c["client_id"] for c in row["clients"]] != active:
                raise ValueError("missing or reordered active-client training observations")
            row["blacklisted_before"], row["blacklisted_after"] = previous["blacklisted"], current["blacklisted"]
            row["aggregation_executed"] = "post_model" in row
            if row["aggregation_executed"]:
                if "aggregate" not in row or row["post_model"] != current["model"]:
                    raise ValueError("post-aggregation model disagrees with checkpoint")
            else:
                if current["model"] != previous["model"] or "aggregate" in row:
                    raise ValueError("unobserved aggregation changed the model")
                row["aggregate"], row["post_model"] = None, current["model"]
            row.setdefault("candidate_order", [])
            row.setdefault("verified_order", [])
            row.setdefault("coefficients", {"order": [], "by_client": {}})
            row.setdefault("aggregate_order", [])
            row["record"] = asdict(records[rd])
            row["diagnostics"] = [_scientific_diagnostic(d) for d in result.diagnostics if d.round == rd]
        self.payload["rounds"] = [self._rounds[rd] for rd in range(1, stop + 1)]
        self.payload["checkpoints"] = [self._checkpoints[rd] for rd in range(stop + 1)]
        self.payload["configuration"] = asdict(result.config)
        self.payload["terminal"] = {"stopped_round": stop, "requested_rounds": horizon,
            "stop_reason": "all_honest_revoked" if all_revoked else None,
            "nonfinite_updates": result.nonfinite_updates,
            "blacklisted_clients": list(result.blacklisted_clients),
            "malicious_clients": list(result.malicious_clients)}
        validate_history_observation(self.payload, self.task)
        json.dumps(self.payload, allow_nan=False)
        self._finished = True
        return self.payload


@contextmanager
def observe(task):
    """Install one process-local passive observer and restore every hook on exit.

Caller supplies ``observer.checkpoint`` to the original run and calls
``observer.finish(result)`` after success. Incomplete runs remain failures.
"""
    global _active
    if _active:
        raise RuntimeError("prefix observation contexts cannot be nested")
    observer = MechanismObserver(task)
    restorations = []

    def install(obj, name, replacement):
        restorations.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    original_run = fl.run_experiment
    original_partition = fl.partition_clients
    original_init = fl.init_params
    original_local = fl._local_train_client_delta
    original_process = fl._process_sm9_candidates
    original_coefficients = fl.bounded_aggregation_coefficients
    original_aggregate = fl.aggregate_with_coefficients
    original_add = backend.TorchTrainingContext.add_update
    original_accuracy = fl._evaluate_accuracy
    original_target = fl._evaluate_attack_target_metrics
    original_forward = backend._torch_forward
    original_finite = fl._update_is_finite
    original_evaluate = LongitudinalSVDDetector.evaluate
    original_commit = LongitudinalSVDDetector.commit
    original_forget = LongitudinalSVDDetector.forget

    def run(dataset, config, *args, **kwargs):
        if observer._started or kwargs.get("resume_state") is not None:
            raise ValueError("prefix probes require one fresh run per process")
        if (type(config.rounds) is not int or config.rounds <= config.detector_window
                or config.malicious_ratio != 0 or config.method != "sm9rrs"
                or config.early_stop or config.eval_interval != 1):
            raise ValueError("mechanism observation requires fresh clean Ours with complete evaluation")
        actual, expected = asdict(config), dict(task["config"])
        actual.pop("device", None)
        expected.pop("device", None)
        if actual != expected:
            raise ValueError("actual probe configuration disagrees with task identity")
        observer._started = True
        observer.payload["data"] = {key: None if getattr(dataset, key) is None else tensor_fingerprint(getattr(dataset, key))
            for key in ("x_train", "y_train", "x_test", "y_test", "x_attack", "y_attack")}
        return original_run(dataset, config, *args, **kwargs)

    def partition(labels, num_clients, **kwargs):
        indices = original_partition(labels, num_clients, **kwargs)
        observer.payload["partition"] = {"strategy": kwargs.get("strategy", "iid"),
            "seed": kwargs.get("seed", 0), "alpha": kwargs.get("dirichlet_alpha", .5),
            "clients": [{"client_id": "client-" + str(i), "samples": len(values),
                         "indices": tensor_fingerprint(values)} for i, values in enumerate(indices)]}
        return indices

    def initialized(*args, **kwargs):
        params = original_init(*args, **kwargs)
        observer.payload["initial_model"] = tensor_fingerprint(params)
        observer.payload["model_spec"] = asdict(kwargs["spec"])
        return params

    def local(params, dataset, indices, *, client_idx, round_id, model_spec, config, torch_context):
        observer._round = round_id
        seed = config.seed + round_id * 1009 + client_idx
        rng = np.random.default_rng(seed)
        row = {"client_id": "client-" + str(client_idx), "samples": len(indices),
            "model_input": tensor_fingerprint(params), "training_seed": seed,
            "learning_rate": config.lr * config.lr_decay ** (round_id - 1),
            "epochs": config.local_epochs, "batch_size": config.batch_size,
            "epoch_indices": [tensor_fingerprint(indices[rng.permutation(len(indices))])
                              for _ in range(config.local_epochs)],
            "minibatch_sizes_per_epoch": [min(config.batch_size, len(indices) - start)
                                         for start in range(0, len(indices), config.batch_size)]}
        value = original_local(params, dataset, indices, client_idx=client_idx, round_id=round_id,
            model_spec=model_spec, config=config, torch_context=torch_context)
        row["stats"] = asdict(value[1])
        if observer._pending_update is not None:
            raise ValueError("previous local update missed its original finiteness check")
        observer._pending_update = (value[0], row)
        observer._round_row()["clients"].append(row)
        return value

    def finite(update):
        pending = observer._pending_update
        if pending is None or pending[0] is not update:
            raise ValueError("finiteness check is outside the original local update")
        value = original_finite(update)
        pending[1]["update_finite"] = bool(value)
        if not value:
            pending[1]["raw_update"] = tensor_fingerprint(update)
        observer._pending_update = None
        return value

    def processed(candidates, *args, **kwargs):
        row = observer._round_row()
        clients = {client["client_id"]: client for client in row["clients"]}
        row["candidate_order"] = [candidate.identity for candidate in candidates]
        for candidate in candidates:
            client = clients[candidate.identity]
            if client["stats"]["samples"] != candidate.samples:
                raise ValueError("candidate samples disagree with original local training")
            # Reuse SM9's existing, mandatory CPU copy of the actual update.
            client["update"] = tensor_fingerprint(candidate.cpu_delta)
            client["raw_update"] = client["update"]
        observer._candidate_ids = {id(candidate.cpu_delta): candidate.identity for candidate in candidates}
        try:
            result = original_process(candidates, *args, **kwargs)
        finally:
            observer._candidate_ids = {}
        observer._tags = dict(result.client_ids_by_tag)
        row["verified_order"] = [observer._tags[tag] for tag in result.tags]
        return result

    def coefficients(tags, *args, **kwargs):
        values = original_coefficients(tags, *args, **kwargs)
        observer._round_row()["coefficients"] = {
            "order": [observer._tags[tag] for tag in tags],
            "by_client": {observer._tags[tag]: value for tag, value in values.items()}}
        return values

    def aggregate(updates, coefficients):
        value = original_aggregate(updates, coefficients)
        row = observer._round_row()
        row["aggregate"] = tensor_fingerprint(value)
        row["aggregate_order"] = [observer._tags[tag] for tag in updates]
        return value

    def added(context, params, update):
        value = original_add(context, params, update)
        observer._round_row()["post_model"] = tensor_fingerprint(value)
        return value

    def evaluation(kind, original, params, *args, **kwargs):
        if observer._evaluation is not None:
            raise RuntimeError("unexpected nested model evaluation")
        row = {"round": observer._round, "kind": kind,
               "model_input": tensor_fingerprint(params), "batches": []}
        observer._evaluation = row
        try:
            value = original(params, *args, **kwargs)
            row["value"] = list(value) if isinstance(value, tuple) else value
            observer.payload["evaluations"].append(row)
            return value
        finally:
            observer._evaluation = None

    def accuracy(params, *args, **kwargs):
        return evaluation("accuracy", original_accuracy, params, *args, **kwargs)

    def target(params, *args, **kwargs):
        return evaluation("target", original_target, params, *args, **kwargs)

    def forward(torch, params, features, spec):
        logits = original_forward(torch, params, features, spec)
        if observer._evaluation is not None:
            # Do not rerun inference or replace the tensors used by accuracy.
            array = logits.detach().cpu().numpy()
            batches = observer._evaluation["batches"]
            batches.append({"batch": len(batches), "logits": tensor_fingerprint(array),
                "predictions": tensor_fingerprint(np.argmax(array, axis=1).astype(np.int64, copy=False))})
        return logits

    def evaluated(detector, tag, update, *, round_id=None, learning_rate=1.0):
        client_id = observer._candidate_ids.get(id(update))
        if client_id is None or round_id != observer._round:
            raise ValueError("detector evaluation is outside captured candidates")
        event = {"round": round_id, "client_id": client_id,
            "before_evaluate": _state(detector._states.get(tag)),
            "after_evaluate": None, "decision": None, "commit": None, "forget": None}
        decision = original_evaluate(detector, tag, update, round_id=round_id, learning_rate=learning_rate)
        event.update(after_evaluate=_state(detector._states[tag]), decision=asdict(decision))
        observer.payload["history_events"].append(event)
        observer._event_by_tag[tag] = event
        return decision

    def committed(detector, tag, *, admit_history):
        event = observer._event_by_tag.get(tag)
        if event is None or event["round"] != observer._round or event["commit"] is not None or event["forget"] is not None:
            raise ValueError("history commit has no unique observed evaluation")
        before = _state(detector._states[tag])
        admitted = original_commit(detector, tag, admit_history=admit_history)
        event["commit"] = {"requested_admission": bool(admit_history), "admitted": admitted,
            "before": before, "after": _state(detector._states[tag])}
        return admitted

    def forgotten(detector, tag):
        event = observer._event_by_tag.get(tag)
        if event is None or event["round"] != observer._round or event["commit"] is not None or event["forget"] is not None:
            raise ValueError("detector forget has no unique observed evaluation")
        before = _state(detector._states.get(tag))
        value = original_forget(detector, tag)
        event["forget"] = {"before": before, "after": _state(detector._states.get(tag))}
        return value

    _active = True
    try:
        for obj, name, replacement in (
            (fl, "run_experiment", run), (fl, "partition_clients", partition),
            (fl, "init_params", initialized), (fl, "_local_train_client_delta", local),
            (fl, "_process_sm9_candidates", processed),
            (fl, "_update_is_finite", finite),
            (fl, "bounded_aggregation_coefficients", coefficients),
            (fl, "aggregate_with_coefficients", aggregate),
            (backend.TorchTrainingContext, "add_update", added),
            (fl, "_evaluate_accuracy", accuracy), (fl, "_evaluate_attack_target_metrics", target),
            (backend, "_torch_forward", forward),
            (LongitudinalSVDDetector, "evaluate", evaluated),
            (LongitudinalSVDDetector, "commit", committed),
            (LongitudinalSVDDetector, "forget", forgotten),
        ):
            install(obj, name, replacement)
        yield observer
    finally:
        for obj, name, original in reversed(restorations):
            setattr(obj, name, original)
        _active = False
