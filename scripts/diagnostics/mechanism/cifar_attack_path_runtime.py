"""Observe the original alternating attack, preserving all attack backward flags.

The already studied singleton policy applies only to ordinary local training.
Every benign-reference, stealth and target attack backward retains the baseline.
All observations are fingerprints of actual tensors; no algorithm is rewritten.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import json
import math

import numpy as np

import cifar_attack_path_observer as pipeline
import cifar_tail_determinism_runtime as flags
from sm9rrsfl import fl
from sm9rrsfl import torch_backend as backend

POLICY_SCHEMA = "cifar-attack-path-policy-v1"
STAGES = ("benign_reference", "stealth_benign", "target_adversarial")
SCOPE = "ordinary_single_sample_local_backward_only_attack_stages_baseline"
_profile = flags._profile
_require = flags._require
tensor_fingerprint = pipeline.tensor_fingerprint
_active = False


def _flat(parameters, gradients=False):
    values = [p.grad if gradients else p for p in parameters]
    _require(all(value is not None for value in values), "missing actual parameter gradient")
    return np.concatenate([value.detach().cpu().numpy().reshape(-1) for value in values])


def _fp(value, label):
    import re
    _require(isinstance(value, dict) and set(value) == {"sha256", "shape", "dtype"}
        and isinstance(value["sha256"], str) and re.fullmatch("[0-9a-f]{64}", value["sha256"]) is not None
        and isinstance(value["shape"], list) and all(type(n) is int and n >= 0 for n in value["shape"])
        and isinstance(value["dtype"], str), "invalid tensor fingerprint: " + label)


def _valid_task(task):
    c = task["config"]
    _require(task.get("policy") == "clean_singleton" and task.get("arm", "H0") == "H0"
        and task.get("candidate", {}).get("variant", "original") == "original"
        and task.get("history_freeze_start_round") is None, "only original H0 clean-singleton policy is supported")
    _require(c["compute_backend"] == "torch" and c["method"] == "sm9rrs"
        and 0 < c["malicious_ratio"] < 1 and c["attack"] in ("alternating", "alternating_minimization")
        and c["detector_window"] < (c["attack_start_round"] or c["detector_window"] + 2) <= c["rounds"]
        and not c["early_stop"] and c["eval_interval"] == 1, "invalid fresh attack-path configuration")
    _require(all(type(c[k]) is int and c[k] > 0 for k in
        ("rounds", "local_epochs", "batch_size", "attack_epochs", "attack_stealth_steps", "attack_target_count")),
        "invalid attack batch configuration")


def _batch_validate(batch, shape, classes, *, attack):
    for name in ("dataset_indices", "features", "labels", "pre_parameters", "logits", "gradients", "post_parameters"):
        _fp(batch.get(name), name)
    n = batch["samples"]
    _require(type(n) is int and n > 0 and batch["dataset_indices"]["shape"] == batch["labels"]["shape"] == [n]
        and batch["features"]["shape"] and batch["features"]["shape"][0] == n
        and batch["logits"]["shape"] == [n, classes], "batch input shapes differ")
    if attack and batch["stage"] == "target_adversarial":
        _require(batch.get("local_indices") is None, "target batch has local training indices")
    else:
        _fp(batch.get("local_indices"), "local_indices")
        _require(batch["local_indices"]["shape"] == [n], "batch local indices differ")
    for name in ("pre_parameters", "gradients", "post_parameters"):
        _require(batch[name]["shape"] == shape, "batch parameter shape differs")
    loss = batch.get("loss", {})
    _fp(loss.get("tensor"), "loss")
    _require(loss["tensor"]["shape"] == [] and (
        type(loss.get("value")) in (int, float) and math.isfinite(loss["value"])
        or loss.get("value") is None and loss.get("nonfinite") in ("nan", "+inf", "-inf")),
        "invalid original scalar loss")


def validate_policy_observation(payload, task):
    """Pure JSON coverage validator. Never loads Torch or queries a GPU."""
    _valid_task(task)
    _require(payload.get("schema") == pipeline.SCHEMA and payload.get("task_id") == task["task_id"]
        and payload.get("task_fingerprint") == task["fingerprint"], "observation identity differs")
    policy = payload.get("numerical_policy", {})
    baseline = policy.get("baseline")
    flags._validate_baseline(baseline)
    _require(policy.get("schema") == POLICY_SCHEMA and policy.get("policy") == "clean_singleton"
        and policy.get("expected_scope") == SCOPE and policy.get("other_flags_changed") is False
        and policy.get("restored_on_exit") is True and policy.get("exit_profile") == baseline,
        "numerical scope or exit restoration differs")
    rows = payload.get("rounds", [])
    stop = payload.get("terminal", {}).get("stopped_round")
    _require(type(stop) is int and 1 <= stop <= task["config"]["rounds"]
        and [r.get("round") for r in rows] == list(range(1, stop + 1)), "missing actual rounds")
    malicious = set(payload["terminal"]["malicious_clients"])
    start = task["config"]["attack_start_round"] or task["config"]["detector_window"] + 2
    ordinary, attacks = [], []
    for row in rows:
        for client in row["clients"]:
            kind = "attack" if row["round"] >= start and client["client_id"] in malicious else "ordinary"
            _require(client.get("training_kind") == kind, "actual client training path differs")
            (attacks if kind == "attack" else ordinary).append((row["round"], client))
    actuals = payload.get("actual_batches", [])
    _require(len(actuals) == len(ordinary), "ordinary active-client batch coverage differs")
    expected_singletons, ordinary_training = [], 0
    for (rd, client), actual in zip(ordinary, actuals):
        _require(actual.get("round") == rd and actual.get("client_id") == client["client_id"], "ordinary client order differs")
        _require(all(actual.get(k) == client[k] for k in ("samples", "epochs", "batch_size")), "ordinary metadata differs")
        sizes = client["minibatch_sizes_per_epoch"] * client["epochs"]
        _require(actual.get("batch_sizes") == sizes and actual.get("forward_count") == actual.get("backward_count") == len(sizes),
            "ordinary actual backward denominator differs")
        ordinary_training += len(sizes)
        expected_singletons.extend((rd, client["client_id"], i) for i, n in enumerate(sizes) if n == 1)
    singleton_batches, singleton_events = payload.get("singleton_batches", []), policy.get("events", [])
    key = lambda e: (e.get("round"), e.get("client_id"), e.get("batch"))
    _require([key(e) for e in singleton_batches] == [key(e) for e in singleton_events] == expected_singletons,
        "ordinary singleton event coverage differs")
    _fp(payload.get("initial_model"), "initial_model")
    shape, classes = payload["initial_model"]["shape"], payload["model_spec"]["num_classes"]
    for batch, event in zip(singleton_batches, singleton_events):
        _require(batch["samples"] == event.get("samples") == 1, "ordinary singleton sample count differs")
        _batch_validate(batch, shape, classes, attack=False)
        _require(event.get("before") == event.get("restored") == baseline
            and event.get("effective") == {**baseline, "cudnn_deterministic": True}
            and event.get("backward_completed") is True, "ordinary singleton flags differ")
    calls = payload.get("attack_calls", [])
    _require([key(e)[:2] for e in calls] == [(rd, c["client_id"]) for rd, c in attacks],
        "actual active attack-client coverage differs")
    batches, events = payload.get("attack_batches", []), policy.get("attack_events", [])
    _require(len(events) == len(batches), "attack flag event coverage differs")
    expected_keys = []
    expected_sizes = []
    expected_call = []
    config = task["config"]
    for (rd, client), call in zip(attacks, calls):
        n = client["samples"]
        sizes = client["minibatch_sizes_per_epoch"] * config["attack_epochs"]
        counts = {"benign_reference": len(sizes), "stealth_benign": len(sizes),
            "target_adversarial": math.ceil(len(sizes) / config["attack_stealth_steps"])}
        _require(call.get("samples") == n and call.get("epochs") == config["attack_epochs"]
            and call.get("batch_size") == config["batch_size"] and call.get("stealth_steps") == config["attack_stealth_steps"]
            and call.get("backward_counts") == counts and call.get("completed") is True
            and type(call.get("target_samples")) is int and 0 < call["target_samples"] <= config["attack_target_count"],
            "actual attack denominator or completed call differs")
        _fp(call.get("target_indices"), "target_indices")
        _require(call["target_indices"]["shape"] == [call["target_samples"]], "attack target denominator differs")
        _require(call.get("model_input") == client["model_input"] and call.get("stats") == client["stats"]
            and call.get("update") == client.get("raw_update"), "attack call differs from original client result")
        for i, size in enumerate(sizes):
            expected_keys.append((rd, client["client_id"], "benign_reference", i)); expected_sizes.append(size)
        target_batch = 0
        for begin in range(0, len(sizes), config["attack_stealth_steps"]):
            for i in range(begin, min(len(sizes), begin + config["attack_stealth_steps"])):
                expected_keys.append((rd, client["client_id"], "stealth_benign", i)); expected_sizes.append(sizes[i])
            expected_keys.append((rd, client["client_id"], "target_adversarial", target_batch))
            expected_sizes.append(call["target_samples"]); target_batch += 1
        expected_call.append(counts)
    attack_key = lambda e: (e.get("round"), e.get("client_id"), e.get("stage"), e.get("batch"))
    _require([attack_key(e) for e in batches] == [attack_key(e) for e in events] == expected_keys,
        "actual attack stage order or coverage differs")
    for batch, event, size in zip(batches, events, expected_sizes):
        _require(batch.get("samples") == event.get("samples") == size, "attack batch denominator differs")
        _batch_validate(batch, shape, classes, attack=True)
        _require(event.get("before") == event.get("effective") == event.get("restored") == baseline
            and event.get("backward_completed") is True, "attack backward changed baseline flags")
    evaluation = sum(len(row["batches"]) for row in payload["evaluations"])
    expected_checks = {"forward_before": ordinary_training + len(batches) + evaluation,
        "forward_after": ordinary_training + len(batches) + evaluation,
        "backward_before": ordinary_training + len(batches), "backward_after": ordinary_training + len(batches),
        "exit": 1}
    _require(policy.get("checks") == expected_checks and policy.get("scoped_changes") == len(singleton_events),
        "full numerical flag coverage differs")
    return policy


class AttackPathObserver:
    def __init__(self, task, original, torch):
        self.task, self.pipeline, self.torch = task, original, torch
        self.baseline = _profile(torch)
        flags._validate_baseline(self.baseline)
        self.actual_batches, self.singleton_batches, self.attack_calls, self.attack_batches = [], [], [], []
        self.events, self.attack_events = [], []
        self.checks = dict.fromkeys(("forward_before", "forward_after", "backward_before", "backward_after", "exit"), 0)
        self.current = self.call = None
        self.local_round = None
        self.exit_profile = None
        self.closed = self.finished = False

    def checkpoint(self, state):
        return self.pipeline.checkpoint(state)

    def check(self, boundary):
        value = _profile(self.torch)
        _require(value == self.baseline, "numerical flags changed at " + boundary)
        self.checks[boundary] += 1
        return value

    def post(self, current):
        row = current.get("batch")
        if row is not None and row.get("capture") and "post_parameters" not in row:
            _require("gradients" in row, "post-SGD observation preceded actual backward")
            row["post_parameters"] = tensor_fingerprint(_flat(current["params"]))

    def finish(self, result):
        _require(self.closed and not self.finished, "observation context must restore before finalization")
        payload = self.pipeline.finish(result)
        for rows in (self.singleton_batches, self.attack_batches):
            for row in rows:
                row.pop("capture", None)
        payload.update(actual_batches=deepcopy(self.actual_batches), singleton_batches=deepcopy(self.singleton_batches),
            attack_calls=deepcopy(self.attack_calls), attack_batches=deepcopy(self.attack_batches))
        payload["numerical_policy"] = {"schema": POLICY_SCHEMA, "policy": "clean_singleton", "baseline": self.baseline,
            "events": self.events, "attack_events": self.attack_events, "checks": self.checks,
            "exit_profile": self.exit_profile, "scoped_changes": len(self.events), "expected_scope": SCOPE,
            "other_flags_changed": False, "restored_on_exit": True}
        payload["observation_contract"].update(policy=SCOPE,
            attack_batches="Actual original loss/backward and post-SGD tensors; fingerprints only; all attack stages retain baseline flags",
            synchronization="Hashing CUDA tensors synchronizes the host and may alter timing; repeatability applies to observed execution")
        validate_policy_observation(payload, self.task)
        json.dumps(payload, allow_nan=False)
        self.finished = True
        return payload


@contextmanager
def observe(task):
    """Install hooks around original functions and restore every hook on all exits."""
    global _active
    _require(not _active, "attack observation contexts cannot be nested")
    _valid_task(task)
    _active = True
    restored, observer = [], None

    def install(obj, name, replacement):
        restored.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    try:
        with pipeline.observe(task) as original:
            torch = backend._torch_module()
            observer = AttackPathObserver(task, original, torch)
            local0, attack0 = fl._local_train_client_delta, fl._alternating_minimization_client_delta
            resident0 = backend.TorchTrainingContext.local_train_delta_resident
            alternating0 = backend.TorchTrainingContext.alternating_minimization_delta_resident
            params0, flat0, forward0 = backend._torch_params_from_tensor, backend._torch_flat_vector_from_params, backend._torch_forward
            select0, ce0, backward0 = torch.Tensor.index_select, torch.nn.functional.cross_entropy, torch.Tensor.backward

            def local(*args, **kwargs):
                _require(observer.local_round is None, "nested ordinary client call")
                observer.local_round = kwargs["round_id"]
                try:
                    return local0(*args, **kwargs)
                finally:
                    observer.local_round = None

            def attack(params, dataset, indices, **kwargs):
                _require(observer.call is None and observer.current is None, "nested attack client call")
                config = kwargs["config"]
                call = {"round": kwargs["round_id"], "client_id": "client-" + str(kwargs["client_idx"]),
                    "samples": len(indices), "target_samples": len(kwargs["attack_target_indices"]),
                    "epochs": config.attack_epochs, "batch_size": config.batch_size,
                    "stealth_steps": config.attack_stealth_steps, "backward_counts": dict.fromkeys(STAGES, 0),
                    "completed": False, "model_input": tensor_fingerprint(params),
                    "target_indices": tensor_fingerprint(kwargs["attack_target_indices"]),
                    "training_seed": config.seed + kwargs["round_id"] * 1009 + kwargs["client_idx"]}
                observer.attack_calls.append(call)
                observer.call = call
                try:
                    value = attack0(params, dataset, indices, **kwargs)
                    call.update(completed=True, update=tensor_fingerprint(value[0]), stats=asdict(value[1]))
                    return value
                finally:
                    observer.call = None

            def new_current(context, client_idx, actual=None, stage=None):
                return {"context": context, "client_idx": client_idx, "actual": actual, "stage": stage,
                    "params": None, "pending": None, "batch": None, "loss": None, "logits": None}

            def resident(context, global_vector, *, client_idx, lr, epochs, batch_size, seed):
                previous = observer.current
                if observer.call is None:
                    actual = {"round": observer.local_round, "client_id": "client-" + str(client_idx),
                        "samples": int(context.client_indices[client_idx].numel()), "epochs": epochs,
                        "batch_size": batch_size, "batch_sizes": [], "forward_count": 0, "backward_count": 0}
                    observer.actual_batches.append(actual)
                    current = new_current(context, client_idx, actual=actual)
                else:
                    current = new_current(context, client_idx, stage="benign_reference")
                observer.current = current
                try:
                    return resident0(context, global_vector, client_idx=client_idx, lr=lr,
                        epochs=epochs, batch_size=batch_size, seed=seed)
                finally:
                    observer.current = previous

            def alternating(context, global_vector, **kwargs):
                _require(observer.call is not None and observer.current is None, "attack backend outside original call")
                current = new_current(context, kwargs["client_idx"], stage="stealth_benign")
                observer.current = current
                try:
                    return alternating0(context, global_vector, **kwargs)
                finally:
                    observer.current = None

            def parameters(torch_module, vector, spec, *, requires_grad, clone):
                values = params0(torch_module, vector, spec, requires_grad=requires_grad, clone=clone)
                current = observer.current
                if current is not None and requires_grad:
                    _require(clone and current["params"] is None, "unexpected trainable parameter construction")
                    current["params"] = values
                return values

            def select(tensor, dim, index, *args, **kwargs):
                value = select0(tensor, dim, index, *args, **kwargs)
                current = observer.current
                if current is None:
                    return value
                context = current["context"]
                if tensor is context.client_indices[current["client_idx"]]:
                    _require(dim == 0 and current["pending"] is None, "unexpected local index selection")
                    current["pending"] = {"local_indices": tensor_fingerprint(index),
                        "dataset_indices": tensor_fingerprint(value), "samples": int(value.numel())}
                    current["selected_index"] = value
                elif tensor is context.x_train:
                    _require(current["pending"] is not None and index is current["selected_index"], "actual train features differ")
                    current["features"] = value
                    current["pending"]["features"] = tensor_fingerprint(value)
                elif tensor is context.x_attack:
                    current["target_features"] = value
                    current["target_indices"] = tensor_fingerprint(index)
                return value

            def forward(torch_module, parameters, features, spec):
                observer.check("forward_before")
                current = observer.current
                if current is not None:
                    _require(current["params"] is not None and all(a is b for a, b in zip(parameters, current["params"])),
                        "original forward parameter identity differs")
                    observer.post(current)
                    call, actual = observer.call, current["actual"]
                    if features is current.get("target_features"):
                        stage = "target_adversarial"
                        row = {"local_indices": None, "dataset_indices": current["target_indices"],
                            "features": tensor_fingerprint(features), "samples": int(features.shape[0])}
                    else:
                        row = current["pending"]
                        _require(row is not None and features is current.get("features"), "original forward input differs")
                        current["pending"] = None
                        stage = current["stage"]
                    if call is None:
                        row.update(round=actual["round"], client_id=actual["client_id"],
                            batch=actual["forward_count"], capture=row["samples"] == 1)
                        actual["batch_sizes"].append(row["samples"]); actual["forward_count"] += 1
                        if row["capture"]:
                            observer.singleton_batches.append(row)
                    else:
                        row.update(round=call["round"], client_id=call["client_id"], stage=stage,
                            batch=call["backward_counts"][stage], capture=True)
                        observer.attack_batches.append(row)
                    current["batch"] = row
                    if row["capture"]:
                        row["pre_parameters"] = tensor_fingerprint(_flat(parameters))
                value = forward0(torch_module, parameters, features, spec)
                observer.check("forward_after")
                if current is not None:
                    current["logits"] = value
                    if current["batch"]["capture"]:
                        current["batch"]["logits"] = tensor_fingerprint(value)
                return value

            def ce(logits, labels, *args, **kwargs):
                current = observer.current
                _require(current is not None and logits is current["logits"] and current["loss"] is None,
                    "loss outside actual original training forward")
                value = ce0(logits, labels, *args, **kwargs)
                current["loss"] = value
                row = current["batch"]
                if row["capture"]:
                    scalar = float(value.detach().cpu().item())
                    row["labels"] = tensor_fingerprint(labels)
                    row["loss"] = {"value": scalar if math.isfinite(scalar) else None, "tensor": tensor_fingerprint(value)}
                    if not math.isfinite(scalar):
                        row["loss"]["nonfinite"] = "nan" if math.isnan(scalar) else "+inf" if scalar > 0 else "-inf"
                return value

            def backward(tensor, *args, **kwargs):
                current = observer.current
                _require(current is not None and tensor is current["loss"], "backward outside actual original loss")
                row = current["batch"]
                before = observer.check("backward_before")
                event = None
                if row["capture"]:
                    event = {k: row[k] for k in ("round", "client_id", "batch", "samples")}
                    if observer.call is not None:
                        event["stage"] = row["stage"]
                    event.update(before=before, effective=None, restored=None, backward_completed=False)
                    (observer.attack_events if observer.call is not None else observer.events).append(event)
                change = observer.call is None and row["samples"] == 1
                try:
                    if change:
                        torch.backends.cudnn.deterministic = True
                    if event is not None:
                        event["effective"] = _profile(torch)
                    value = backward0(tensor, *args, **kwargs)
                    if event is not None:
                        event["backward_completed"] = True
                    _require(_profile(torch) == ({**observer.baseline, "cudnn_deterministic": True} if change else observer.baseline),
                        "flags changed inside original backward")
                finally:
                    if change:
                        torch.backends.cudnn.deterministic = before["cudnn_deterministic"]
                    if event is not None:
                        event["restored"] = _profile(torch)
                    _require(_profile(torch) == observer.baseline, "flags failed to restore after backward")
                observer.check("backward_after")
                if row["capture"]:
                    row["gradients"] = tensor_fingerprint(_flat(current["params"], gradients=True))
                if observer.call is not None:
                    observer.call["backward_counts"][row["stage"]] += 1
                else:
                    current["actual"]["backward_count"] += 1
                current["loss"] = current["logits"] = None
                return value

            def flatten(torch_module, parameters):
                value = flat0(torch_module, parameters)
                current = observer.current
                if current is not None and current["params"] is not None and all(a is b for a, b in zip(parameters, current["params"])):
                    observer.post(current)
                return value

            for obj, name, fn in ((fl, "_local_train_client_delta", local),
                (fl, "_alternating_minimization_client_delta", attack),
                (backend.TorchTrainingContext, "local_train_delta_resident", resident),
                (backend.TorchTrainingContext, "alternating_minimization_delta_resident", alternating),
                (backend, "_torch_params_from_tensor", parameters), (backend, "_torch_flat_vector_from_params", flatten),
                (backend, "_torch_forward", forward), (torch.Tensor, "index_select", select),
                (torch.nn.functional, "cross_entropy", ce), (torch.Tensor, "backward", backward)):
                install(obj, name, fn)
            try:
                yield observer
                observer.exit_profile = observer.check("exit")
                observer.closed = True
            finally:
                for obj, name, fn in reversed(restored):
                    setattr(obj, name, fn)
                restored.clear()
    finally:
        for obj, name, fn in reversed(restored):
            setattr(obj, name, fn)
        _active = False
