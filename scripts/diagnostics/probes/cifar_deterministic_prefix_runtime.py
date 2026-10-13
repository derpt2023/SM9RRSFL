"""Observe complete clean prefixes with one declared singleton backward policy.

This composes the frozen prefix observer and calls every original numerical
operation. Only arm B temporarily sets cudnn.deterministic during an actual
single-sample local loss.backward. Host fingerprint reads synchronize device
work and can affect scheduling; repeatability applies to this observed run.
No arrays, checkpoint state, crypto material, or files are saved by this module.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import re

import cifar_prefix_probe_runtime as prefix
import cifar_client_step_probe_runtime as step
import cifar_tail_determinism_runtime as tail
from sm9rrsfl import fl
from sm9rrsfl import torch_backend as backend

POLICY_SCHEMA = "cifar-deterministic-prefix-policy-observation-v1"
POLICIES = ("original", "singleton_backward_cudnn_deterministic")
SCOPE = "every_actual_single_sample_local_training_backward"
CHECK_FIELDS = tail.CHECK_FIELDS
PROFILE_FIELDS = tail.PROFILE_FIELDS
STAGES = ("local_indices", "dataset_indices", "features", "labels", "pre_parameters",
          "logits", "gradients", "post_parameters")
IDENTITY = ("round", "client_id", "batch", "epoch", "batch_in_epoch", "samples")
tensor_fingerprint = prefix.tensor_fingerprint
source_evidence = tail.source_evidence
_profile = tail._profile
_require = tail._require
_active = False


def _valid_task(task):
    config = task["config"]
    _require(task.get("policy") in POLICIES, "unknown singleton numerical policy")
    _require(config["rounds"] == 3 and config["malicious_ratio"] == 0
             and config["method"] == "sm9rrs" and config["compute_backend"] == "torch",
             "singleton prefix requires exactly three clean original Ours Torch rounds")
    _require(type(config["local_epochs"]) is int and config["local_epochs"] > 0
             and type(config["batch_size"]) is int and config["batch_size"] > 0,
             "positive original batch size and epochs are required")


def _fingerprint(value, name):
    _require(isinstance(value, dict) and set(value) == {"sha256", "shape", "dtype"}
             and isinstance(value["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None
             and isinstance(value["shape"], list) and all(type(n) is int and n >= 0 for n in value["shape"])
             and isinstance(value["dtype"], str), "invalid singleton fingerprint: " + name)


def validate_policy_observation(payload, task):
    """Validate JSON evidence without importing Torch or querying a device.

Call in addition to the frozen prefix report's full scientific validator.
Coverage follows the actual complete client calls, never fixed client IDs.
"""
    _valid_task(task)
    value = payload.get("numerical_policy", {})
    _require(value.get("schema") == POLICY_SCHEMA and value.get("policy") == task["policy"],
             "singleton policy identity differs")
    baseline = value.get("baseline")
    tail._validate_baseline(baseline)
    tail._validate_profile(value.get("exit_profile"))
    _require(value.get("expected_scope") == SCOPE and value.get("other_flags_changed") is False
             and value.get("restored_on_exit") is True and value.get("exit_profile") == baseline,
             "singleton policy scope or restoration differs")
    rows = payload.get("rounds", [])
    _require([row.get("round") for row in rows] == [1, 2, 3], "three complete rounds are required")
    originals = [(row["round"], client) for row in rows for client in row["clients"]]
    observed = payload.get("actual_batches", [])
    _require(isinstance(observed, list) and len(observed) == len(originals), "actual client batch coverage differs")
    expected_events = []
    training = 0
    for (rd, client), actual in zip(originals, observed):
        _require(actual.get("round") == rd and actual.get("client_id") == client["client_id"],
                 "actual client batch identity/order differs")
        for key in ("samples", "epochs", "batch_size"):
            _require(type(actual.get(key)) is int and actual[key] == client[key], "actual client batch metadata differs")
        samples, size, epochs = actual["samples"], actual["batch_size"], actual["epochs"]
        per_epoch = [min(size, samples - start) for start in range(0, samples, size)]
        expected_sizes = per_epoch * epochs
        _require(client["minibatch_sizes_per_epoch"] == per_epoch
                 and actual.get("batch_sizes") == expected_sizes
                 and all(type(n) is int for n in actual["batch_sizes"]), "actual batch sizes differ from original client data")
        count = len(expected_sizes)
        _require(type(actual.get("forward_count")) is int and type(actual.get("backward_count")) is int
                 and actual["forward_count"] == actual["backward_count"] == count,
                 "actual forward/backward batch coverage differs")
        training += count
        expected_events.extend({"round": rd, "client_id": client["client_id"], "batch": i,
            "epoch": i // len(per_epoch), "batch_in_epoch": i % len(per_epoch), "samples": 1}
            for i, n in enumerate(expected_sizes) if n == 1)
    singletons, events = payload.get("singleton_batches", []), value.get("events", [])
    _require(isinstance(singletons, list) and isinstance(events, list)
             and len(singletons) == len(events) == len(expected_events), "singleton event coverage differs")
    effective = {**baseline, "cudnn_deterministic": task["policy"] != "original"}
    initial = payload.get("initial_model")
    _fingerprint(initial, "initial_model")
    num_classes = payload.get("model_spec", {}).get("num_classes")
    _require(type(num_classes) is int and num_classes > 0, "invalid model class count")
    for expected, batch, event in zip(expected_events, singletons, events):
        for entry in (batch, event):
            _require(all(entry.get(key) == expected[key] and (key == "client_id" or type(entry[key]) is int)
                         for key in IDENTITY), "singleton event does not match actual batch")
        for field in STAGES:
            _fingerprint(batch.get(field), field)
        loss = batch.get("loss", {})
        _fingerprint(loss.get("tensor"), "loss")
        _require(type(loss.get("value")) in (int, float) and math.isfinite(loss["value"])
                 and loss["tensor"]["shape"] == [], "invalid original singleton loss")
        _require(batch["local_indices"]["shape"] == batch["dataset_indices"]["shape"] == batch["labels"]["shape"] == [1]
                 and bool(batch["features"]["shape"]) and batch["features"]["shape"][0] == 1
                 and batch["logits"]["shape"] == [1, num_classes],
                 "singleton fingerprints do not describe actual single-sample tensors")
        layout, offset, names = batch.get("parameter_layout", []), 0, set()
        _require(isinstance(layout, list) and bool(layout), "missing singleton parameter layout")
        for parameter in layout:
            shape = parameter.get("shape", [])
            _require(isinstance(parameter.get("name"), str) and parameter["name"] not in names
                     and isinstance(shape, list) and all(type(n) is int and n > 0 for n in shape)
                     and parameter.get("offset") == offset and parameter.get("size") == math.prod(shape),
                     "invalid singleton parameter layout")
            names.add(parameter["name"])
            offset += parameter["size"]
        _require(initial["shape"] == [offset] and all(
            batch[field]["shape"] == initial["shape"] and batch[field]["dtype"] == initial["dtype"]
            for field in ("pre_parameters", "gradients", "post_parameters")),
            "singleton parameter vector shape/dtype differs from the original model")
        for field in ("before", "effective", "restored"):
            tail._validate_profile(event.get(field))
        _require(event["before"] == event["restored"] == baseline and event["effective"] == effective
                 and event.get("backward_completed") is True, "singleton effective policy differs")
    evaluation = sum(len(row["batches"]) for row in payload["evaluations"])
    checks = {"forward_before": training + evaluation, "forward_after": training + evaluation,
              "non_target_backward_before": training - len(events), "non_target_backward_after": training - len(events),
              "post_flat": sum(client["samples"] > 0 for _, client in originals), "exit": 1}
    _require(value.get("checks") == checks and all(type(n) is int for n in value["checks"].values()),
             "numerical flag coverage differs")
    expected_changes = len(events) if task["policy"] != "original" else 0
    _require(type(value.get("scoped_changes")) is int and value["scoped_changes"] == expected_changes,
             "singleton policy change count differs")
    return value


class DeterministicPrefixObserver:
    def __init__(self, task, old, torch):
        self.task, self.prefix, self.torch = task, old, torch
        self.baseline = _profile(torch)
        tail._validate_baseline(self.baseline)
        self.actual_batches, self.singleton_batches, self.events = [], [], []
        self.checks = dict.fromkeys(CHECK_FIELDS, 0)
        self.scoped_changes = 0
        self.exit_profile = None
        self._current = None
        self._closed = False
        self._finished = False

    def checkpoint(self, state):
        return self.prefix.checkpoint(state)

    def check(self, boundary):
        profile = _profile(self.torch)
        _require(profile == self.baseline, "numerical flags changed at " + boundary)
        self.checks[boundary] += 1
        return profile

    def post(self, params, flat=None):
        current = self._current
        batch = current["batch"]
        if batch is not None and batch["samples"] == 1 and "post_parameters" not in batch:
            _require("gradients" in batch, "post-SGD capture preceded backward")
            batch["post_parameters"] = tensor_fingerprint(step._flat(params) if flat is None else flat)

    def finish(self, result):
        _require(self._closed and not self._finished, "policy context must finish successfully before finalization")
        payload = self.prefix.finish(result)
        payload.update(actual_batches=deepcopy(self.actual_batches), singleton_batches=deepcopy(self.singleton_batches))
        payload["numerical_policy"] = {"schema": POLICY_SCHEMA, "policy": self.task["policy"],
            "baseline": deepcopy(self.baseline), "events": deepcopy(self.events), "checks": dict(self.checks),
            "exit_profile": deepcopy(self.exit_profile), "scoped_changes": self.scoped_changes,
            "expected_scope": SCOPE, "other_flags_changed": False, "restored_on_exit": True}
        payload["observation_contract"].update(
            policy="Original numerical flags throughout" if self.task["policy"] == "original" else
                "Only cudnn.deterministic becomes true during every actual single-sample local loss.backward, restored before original SGD; all other flags and numerical operations are unchanged",
            actual_batches="Original index_select, forward, loss and backward identities establish every batch size; singleton stage hashes come from actual original tensors",
            singleton_stages="Post-SGD parameters read at next original forward entry or final flatten; hashes only, no numerical-distance arrays",
            scope="Fresh clean original Ours three-round diagnostic only; attack paths, other methods and longer histories are not tested")
        validate_policy_observation(payload, self.task)
        json.dumps(payload, allow_nan=False)
        self._finished = True
        return payload


@contextmanager
def observe(task):
    """Compose the original complete prefix observer; finalize after context exit."""
    global _active
    _require(not _active, "deterministic prefix contexts cannot be nested")
    _valid_task(task)
    restorations = []
    observer = None
    success = False

    def install(obj, name, replacement):
        restorations.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    _active = True
    try:
        with prefix.observe(task) as old:
            torch = backend._torch_module()
            observer = DeterministicPrefixObserver(task, old, torch)
            original_local = fl._local_train_client_delta
            original_resident = backend.TorchTrainingContext.local_train_delta_resident
            original_params = backend._torch_params_from_tensor
            original_flat = backend._torch_flat_vector_from_params
            original_forward = backend._torch_forward
            original_select = torch.Tensor.index_select
            original_ce = torch.nn.functional.cross_entropy
            original_backward = torch.Tensor.backward
            local_round = None

            def local(*args, **kwargs):
                nonlocal local_round
                _require(local_round is None, "nested local training")
                local_round = kwargs["round_id"]
                try:
                    return original_local(*args, **kwargs)
                finally:
                    local_round = None

            def resident(context, global_vector, *, client_idx, lr, epochs, batch_size, seed):
                _require(observer._current is None and local_round in (1, 2, 3), "unexpected resident training scope")
                samples = int(context.client_indices[client_idx].numel())
                actual = {"round": local_round, "client_id": "client-" + str(client_idx), "samples": samples,
                          "epochs": epochs, "batch_size": batch_size, "batch_sizes": [],
                          "forward_count": 0, "backward_count": 0}
                observer.actual_batches.append(actual)
                current = {"context": context, "index": client_idx, "actual": actual, "params": None,
                           "layout": None, "batch": None, "pending": None, "loss": None, "logits": None}
                observer._current = current
                try:
                    value = original_resident(context, global_vector, client_idx=client_idx, lr=lr,
                                              epochs=epochs, batch_size=batch_size, seed=seed)
                    count = ((samples + batch_size - 1) // batch_size) * epochs
                    _require(actual["forward_count"] == actual["backward_count"] == len(actual["batch_sizes"]) == count
                             and current["pending"] is None and current["loss"] is None,
                             "incomplete actual minibatch observations")
                    if current["batch"] is not None and current["batch"]["samples"] == 1:
                        _require("post_parameters" in current["batch"], "missing final original SGD capture")
                    return value
                finally:
                    observer._current = None

            def params(torch_module, vector, spec, *, requires_grad, clone):
                values = original_params(torch_module, vector, spec, requires_grad=requires_grad, clone=clone)
                current = observer._current
                if current is not None:
                    _require(requires_grad and clone and current["params"] is None, "unexpected local parameter construction")
                    current["params"] = values
                    current["layout"] = step._layout(values, spec)
                return values

            def select(tensor, dim, index, *args, **kwargs):
                value = original_select(tensor, dim, index, *args, **kwargs)
                current = observer._current
                if current is None:
                    return value
                context, actual = current["context"], current["actual"]
                if tensor is context.client_indices[current["index"]]:
                    _require(dim == 0 and current["pending"] is None, "unexpected actual client index selection")
                    n, batch = int(value.numel()), len(actual["batch_sizes"])
                    per_epoch = (actual["samples"] + actual["batch_size"] - 1) // actual["batch_size"]
                    row = {"round": actual["round"], "client_id": actual["client_id"], "batch": batch,
                           "epoch": batch // per_epoch, "batch_in_epoch": batch % per_epoch, "samples": n}
                    if n == 1:
                        row.update(local_indices=tensor_fingerprint(index), dataset_indices=tensor_fingerprint(value),
                                   parameter_layout=deepcopy(current["layout"]))
                    current["pending"] = row
                    current["selected_index"] = value
                    actual["batch_sizes"].append(n)
                elif tensor is context.x_train:
                    row = current["pending"]
                    _require(row is not None and dim == 0 and index is current["selected_index"]
                             and int(value.shape[0]) == row["samples"], "actual feature selection differs")
                    current["features"] = value
                    if row["samples"] == 1:
                        row["features"] = tensor_fingerprint(value)
                elif tensor is context.y_train:
                    row = current["batch"]
                    _require(row is not None and dim == 0 and index is current["selected_index"]
                             and int(value.shape[0]) == row["samples"], "actual label selection differs")
                    current["labels"] = value
                    if row["samples"] == 1:
                        row["labels"] = tensor_fingerprint(value)
                return value

            def forward(torch_module, parameters, features, spec):
                observer.check("forward_before")
                current = observer._current
                if current is not None:
                    _require(len(parameters) == len(current["params"]) and all(a is b for a, b in zip(parameters, current["params"])),
                             "actual forward parameter identity differs")
                    observer.post(parameters)
                    row = current["pending"]
                    _require(row is not None and features is current.get("features")
                             and int(features.shape[0]) == row["samples"], "actual forward batch differs")
                    current["pending"] = None
                    current["batch"] = row
                    current["actual"]["forward_count"] += 1
                    if row["samples"] == 1:
                        row["pre_parameters"] = tensor_fingerprint(step._flat(parameters))
                        observer.singleton_batches.append(row)
                value = original_forward(torch_module, parameters, features, spec)
                observer.check("forward_after")
                if current is not None:
                    current["logits"] = value
                    if current["batch"]["samples"] == 1:
                        current["batch"]["logits"] = tensor_fingerprint(value)
                return value

            def cross_entropy(logits, labels, *args, **kwargs):
                current = observer._current
                _require(current is not None and logits is current["logits"] and labels is current.get("labels")
                         and current["loss"] is None, "loss outside the captured original local forward")
                value = original_ce(logits, labels, *args, **kwargs)
                current["loss"] = value
                if current["batch"]["samples"] == 1:
                    array = step._array(value)
                    current["batch"]["loss"] = {"value": float(array.item()), "tensor": tensor_fingerprint(array)}
                return value

            def backward(tensor, *args, **kwargs):
                current = observer._current
                _require(current is not None and tensor is current["loss"], "backward outside captured original local loss")
                row = current["batch"]
                if row["samples"] != 1:
                    observer.check("non_target_backward_before")
                    value = original_backward(tensor, *args, **kwargs)
                    observer.check("non_target_backward_after")
                else:
                    before = _profile(torch)
                    _require(before == observer.baseline, "flags changed before singleton backward")
                    event = {**{k: row[k] for k in IDENTITY}, "before": before,
                             "effective": None, "restored": None, "backward_completed": False}
                    observer.events.append(event)
                    try:
                        if task["policy"] != "original":
                            torch.backends.cudnn.deterministic = True
                            observer.scoped_changes += 1
                        event["effective"] = _profile(torch)
                        _require(event["effective"] == {**before, "cudnn_deterministic": task["policy"] != "original"},
                                 "unexpected effective singleton flags")
                        value = original_backward(tensor, *args, **kwargs)
                        event["backward_completed"] = True
                        _require(_profile(torch) == event["effective"], "flags changed inside singleton backward")
                    finally:
                        if task["policy"] != "original":
                            torch.backends.cudnn.deterministic = before["cudnn_deterministic"]
                        event["restored"] = _profile(torch)
                        _require(event["restored"] == observer.baseline, "singleton flags failed to restore")
                    row["gradients"] = tensor_fingerprint(step._flat(current["params"], gradients=True))
                current["actual"]["backward_count"] += 1
                current["loss"] = None
                current["logits"] = None
                return value

            def flat(torch_module, parameters):
                value = original_flat(torch_module, parameters)
                current = observer._current
                _require(current is not None and len(parameters) == len(current["params"])
                         and all(a is b for a, b in zip(parameters, current["params"])), "unexpected local final flatten")
                observer.post(parameters, flat=value)
                observer.check("post_flat")
                return value

            for obj, name, replacement in (
                (fl, "_local_train_client_delta", local),
                (backend.TorchTrainingContext, "local_train_delta_resident", resident),
                (backend, "_torch_params_from_tensor", params), (backend, "_torch_flat_vector_from_params", flat),
                (backend, "_torch_forward", forward), (torch.Tensor, "index_select", select),
                (torch.nn.functional, "cross_entropy", cross_entropy), (torch.Tensor, "backward", backward),
            ):
                install(obj, name, replacement)
            try:
                yield observer
                observer.exit_profile = observer.check("exit")
                success = True
            finally:
                for obj, name, original in reversed(restorations):
                    setattr(obj, name, original)
                restorations.clear()
    finally:
        for obj, name, original in reversed(restorations):
            setattr(obj, name, original)
        if observer is not None:
            observer._closed = success
        _active = False
