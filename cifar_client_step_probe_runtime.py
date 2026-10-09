"""Observe original local SGD without replacing its numerical operations.

The old prefix observer remains installed. The original first-round client loop
executes through the selected stop client, including its ordinary CPU candidate
copy. A private sentinel stops at the next client's entry, before any aggregation.
CPU reads synchronize device work and can alter scheduling: this is an observed
diagnostic execution, not a claim that observation is invisible to CUDA timing.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np

import cifar_prefix_probe_runtime as prefix
from sm9rrsfl import fl
from sm9rrsfl import torch_backend as backend

SCHEMA = "cifar-client-step-probe-observation-v1"
tensor_fingerprint = prefix.tensor_fingerprint
_active = False


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value):
    if type(value).__module__.startswith("torch"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    _require(not result.dtype.hasobject, "object arrays cannot be probe evidence")
    if np.issubdtype(result.dtype, np.inexact):
        _require(bool(np.isfinite(result).all()), "nonfinite step observation")
    return result


def _flat(params, *, gradients=False):
    values = [param.grad if gradients else param for param in params]
    _require(all(value is not None for value in values), "missing parameter gradient")
    # CPU concatenation only; no replacement device operation or SGD arithmetic.
    return np.concatenate([_array(value).reshape(-1) for value in values])


def _layout(params, spec):
    names = ("conv1_w", "conv1_b", "conv2_w", "conv2_b", "dense1_w", "dense1_b",
             "dense2_w", "dense2_b", "logits_w", "logits_b") if spec.architecture == "cifar10" else (
                 "conv_w", "conv_b", "logits_w", "logits_b")
    _require(len(names) == len(params), "unknown model parameter layout")
    rows, offset = [], 0
    for name, value in zip(names, params):
        size = int(value.numel())
        rows.append({"name": name, "shape": list(value.shape), "offset": offset, "size": size})
        offset += size
    return rows


class _StopAtNextClient(BaseException):
    def __init__(self, observer):
        self.observer = observer


class ClientStepObserver:
    def __init__(self, task, prefix_observer):
        self.task = task
        self.prefix = prefix_observer
        self.target_indices = list(task["target_clients"])
        self.stop_after = task["stop_after_client"]
        self.snapshots = {}
        self._snapshot_fingerprints = {}
        self._targets = {}
        self._current = None
        self._configuration = None
        self._candidate_ids = []
        self._stopped = False
        self._finished = False

    def checkpoint(self, state):
        _require(state.get("completed_round") == 0, "step probe must stop before first aggregation")
        self.prefix.checkpoint(state)

    def _save(self, target, field, array):
        key = target["client_id"].replace("-", "_") + "__" + field
        _require(key not in self.snapshots, "duplicate saved step tensor")
        value = _array(array).copy()
        self.snapshots[key] = value
        self._snapshot_fingerprints[key] = tensor_fingerprint(value)
        target["tail_snapshots"][field] = key

    def _record(self, field, value, *, row=None):
        current = self._current
        _require(current is not None, "step tensor outside target client")
        row = current["batch"] if row is None else row
        _require(field not in row, "duplicate batch observation: " + field)
        array = _array(value)
        row[field] = tensor_fingerprint(array)
        if row["batch"] == current["expected_batches"] - 1:
            self._save(current["target"], field, array)
        return array

    def _post_parameters(self, params, *, flat_value=None):
        current = self._current
        row = current["batch"]
        if row is None or "post_parameters" in row:
            return
        _require("gradients" in row, "SGD observation before original backward")
        self._record("post_parameters", _flat(params) if flat_value is None else flat_value)

    def finish(self):
        """Finalize only after the context has consumed its own successful stop.

The returned JSON contains exact fingerprints and compact scalar values.
``snapshots`` separately holds public numerical arrays for a caller-owned NPZ;
this module never writes files or serializes cryptographic/checkpoint state.
"""
        _require(not self._finished, "step observer can only be finalized once")
        _require(self._stopped and self._current is None, "controlled client-prefix stop was not reached")
        wanted = ["client-" + str(i) for i in range(self.stop_after + 1)]
        _require(self._candidate_ids == wanted, "original candidate prefix is incomplete or reordered")
        _require(set(self.prefix._rounds) == {1} and set(self.prefix._checkpoints) == {0},
                 "unexpected round or checkpoint coverage")
        row = self.prefix._rounds[1]
        _require([item["client_id"] for item in row["clients"]] == wanted,
                 "original local-training prefix is incomplete")
        _require(not any(key in row for key in ("aggregate", "post_model", "coefficients")),
                 "aggregation unexpectedly ran")
        _require([(item["round"], item["kind"]) for item in self.prefix.payload["evaluations"]] ==
                 [(0, "accuracy"), (0, "target")], "round-zero evaluation coverage differs")
        _require(set(self._targets) == set(self.target_indices), "target client coverage differs")
        required = {"local_indices", "dataset_indices", "features", "labels", "pre_parameters",
                    "logits", "loss", "gradients", "post_parameters"}
        targets = [self._targets[index] for index in self.target_indices]
        for target in targets:
            count = (target["samples"] + target["batch_size"] - 1) // target["batch_size"]
            _require(len(target["batches"]) == count * target["epochs"], "target minibatch coverage differs")
            _require(all(required <= set(batch) for batch in target["batches"]), "incomplete target minibatch")
            _require(set(target["tail_snapshots"]) == required | {"final_delta"}, "tail snapshot coverage differs")
            _require("final_delta" in target and "stats" in target, "target return was not observed")
        partial = {**deepcopy(self.prefix.payload), "rounds": [deepcopy(row)],
                   "checkpoints": [deepcopy(self.prefix._checkpoints[0])],
                   "configuration": self._configuration,
                   "capture_scope": "round-zero evaluation and first-round clients through controlled stop; not a completed round or experiment"}
        result = {"schema": SCHEMA, "task_id": self.task["task_id"],
            "task_fingerprint": self.task["fingerprint"], "configuration": self._configuration,
            "target_client_indices": self.target_indices, "stop_after_client": self.stop_after,
            "stop": {"round": 1, "after_client": "client-" + str(self.stop_after),
                     "next_client_not_trained": "client-" + str(self.stop_after + 1),
                     "before_aggregation": True, "completed_rounds": 0, "crypto_finalized": False},
            "prefix": partial, "targets": targets, "snapshots": self._snapshot_fingerprints,
            "observation_contract": {
                "actual_batch_indices": "original Tensor.index_select arguments and results, not reconstructed sample permutations",
                "parameter_stages": "original pre-forward parameters, original forward/logits and loss, leaf gradients after original backward, parameters after original SGD",
                "post_sgd_read": "read at the following original forward entry or original final parameter flattening; no SGD replacement",
                "tail_arrays": "only final minibatch tensors plus returned final delta; other minibatches have exact hashes but no numerical distance evidence",
                "prefix": "all original client calls and candidate CPU copies through stop client retained; next client and aggregation never executed",
                "synchronization": "CPU copies/hash reads synchronize device work and may alter timing or scheduling",
                "policy": "no numerical flag, random state, optimizer, seed, or original training source changed",
                "scope": "controlled diagnostic stop, not completed training, health qualification or crypto finalization"}}
        json.dumps(result, allow_nan=False)
        self._finished = True
        return result


@contextmanager
def observe(task):
    """Use around original ``fl.run_experiment``; call ``finish`` after exit.

Only this observer's private stop is suppressed. Any numerical, infrastructure,
keyboard or other exception escapes after every installed hook is restored.
"""
    global _active
    _require(not _active, "client step observation contexts cannot be nested")
    config = task["config"]
    targets, stop = task.get("target_clients"), task.get("stop_after_client")
    _require(isinstance(targets, list) and bool(targets) and all(type(i) is int for i in targets)
             and targets == sorted(set(targets)), "target_clients must be sorted distinct integers")
    _require(type(stop) is int and 0 <= targets[0] <= targets[-1] <= stop < config["num_clients"] - 1,
             "stop requires a next client entry and must cover every target")
    _require(config["compute_backend"] == "torch" and config["local_epochs"] >= 1
             and config["batch_size"] > 0, "step observations require resident Torch local training")
    restorations = []
    def install(obj, name, replacement):
        restorations.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    _active = True
    try:
        with prefix.observe(task) as prefix_observer:
            observer = ClientStepObserver(task, prefix_observer)
            torch = backend._torch_module()
            original_run = fl.run_experiment
            original_local = fl._local_train_client_delta
            original_candidate = fl._ClientUpdateCandidate
            original_resident = backend.TorchTrainingContext.local_train_delta_resident
            original_params = backend._torch_params_from_tensor
            original_flat = backend._torch_flat_vector_from_params
            original_forward = backend._torch_forward
            original_index_select = torch.Tensor.index_select
            original_cross_entropy = torch.nn.functional.cross_entropy
            original_backward = torch.Tensor.backward

            def run(dataset, actual_config, *args, **kwargs):
                observer._configuration = asdict(actual_config)
                return original_run(dataset, actual_config, *args, **kwargs)

            def local(*args, **kwargs):
                index, rd = kwargs["client_idx"], kwargs["round_id"]
                _require(rd == 1, "step probe reached a later training round")
                if index == stop + 1:
                    _require(observer._candidate_ids == ["client-" + str(i) for i in range(stop + 1)],
                             "cannot stop successfully with missing original candidates")
                    raise _StopAtNextClient(observer)
                _require(index <= stop, "step probe passed its stop boundary")
                return original_local(*args, **kwargs)

            def resident(context, global_vector, *, client_idx, lr, epochs, batch_size, seed):
                if client_idx not in targets:
                    return original_resident(context, global_vector, client_idx=client_idx,
                        lr=lr, epochs=epochs, batch_size=batch_size, seed=seed)
                _require(observer._current is None and client_idx not in observer._targets,
                         "repeated or nested target local training")
                samples = int(context.client_indices[client_idx].numel())
                _require(samples > 0, "target client has no samples")
                target = {"client_id": "client-" + str(client_idx), "samples": samples,
                    "model_input": tensor_fingerprint(global_vector), "training_seed": seed,
                    "learning_rate": lr, "epochs": epochs, "batch_size": batch_size,
                    "parameter_layout": None, "batches": [], "tail_snapshots": {}}
                observer._targets[client_idx] = target
                per_epoch = (samples + batch_size - 1) // batch_size
                current = {"context": context, "index": client_idx, "target": target,
                    "per_epoch": per_epoch, "expected_batches": per_epoch * epochs,
                    "params": None, "batch": None, "pending": None, "logits": None, "loss": None}
                observer._current = current
                try:
                    value = original_resident(context, global_vector, client_idx=client_idx,
                        lr=lr, epochs=epochs, batch_size=batch_size, seed=seed)
                    _require(len(target["batches"]) == current["expected_batches"]
                             and "post_parameters" in current["batch"], "original SGD did not finish every target minibatch")
                    target["stats"] = asdict(value[1])
                    return value
                finally:
                    observer._current = None

            def params(torch_module, flat_vector, spec, *, requires_grad, clone):
                value = original_params(torch_module, flat_vector, spec, requires_grad=requires_grad, clone=clone)
                current = observer._current
                if current is not None:
                    _require(requires_grad and clone and current["params"] is None,
                             "unexpected target parameter construction")
                    current["params"] = value
                    current["target"]["parameter_layout"] = _layout(value, spec)
                return value

            def index_select(tensor, dim, index, *args, **kwargs):
                value = original_index_select(tensor, dim, index, *args, **kwargs)
                current = observer._current
                if current is None:
                    return value
                context = current["context"]
                if tensor is context.client_indices[current["index"]]:
                    _require(dim == 0, "unexpected client index dimension")
                    batch = len(current["target"]["batches"])
                    _require(batch < current["expected_batches"], "too many actual minibatches")
                    row = {"batch": batch, "epoch": batch // current["per_epoch"],
                           "batch_in_epoch": batch % current["per_epoch"], "samples": int(value.numel())}
                    observer._record("local_indices", index, row=row)
                    observer._record("dataset_indices", value, row=row)
                    current["pending"] = row
                elif tensor is context.x_train:
                    row = current["pending"]
                    _require(row is not None and dim == 0 and tensor_fingerprint(index) == row["dataset_indices"],
                             "actual feature indices differ from selected client batch")
                    observer._record("features", value, row=row)
                elif tensor is context.y_train:
                    row = current["batch"]
                    _require(row is not None and dim == 0 and tensor_fingerprint(index) == row["dataset_indices"],
                             "actual label indices differ from selected client batch")
                    observer._record("labels", value)
                return value

            def forward(torch_module, parameters, features, spec):
                current = observer._current
                if current is None:
                    return original_forward(torch_module, parameters, features, spec)
                _require(all(a is b for a, b in zip(parameters, current["params"]))
                         and len(parameters) == len(current["params"]), "forward parameter identity differs")
                observer._post_parameters(parameters)
                row = current["pending"]
                _require(row is not None and tensor_fingerprint(features) == row.get("features"),
                         "forward does not use the recorded actual features")
                current["pending"] = None
                current["batch"] = row
                current["target"]["batches"].append(row)
                observer._record("pre_parameters", _flat(parameters))
                logits = original_forward(torch_module, parameters, features, spec)
                observer._record("logits", logits)
                current["logits"] = logits
                return logits

            def cross_entropy(logits, labels, *args, **kwargs):
                value = original_cross_entropy(logits, labels, *args, **kwargs)
                current = observer._current
                if current is not None:
                    _require(logits is current["logits"] and tensor_fingerprint(labels) == current["batch"].get("labels"),
                             "loss does not use the recorded logits and actual labels")
                    array = _array(value)
                    current["batch"]["loss"] = {"value": float(array.item()), "tensor": tensor_fingerprint(array)}
                    if current["batch"]["batch"] == current["expected_batches"] - 1:
                        observer._save(current["target"], "loss", array)
                    current["loss"] = value
                return value

            def backward(tensor, *args, **kwargs):
                current = observer._current
                if current is not None:
                    _require(tensor is current["loss"], "unexpected backward tensor in target local training")
                value = original_backward(tensor, *args, **kwargs)
                if current is not None:
                    observer._record("gradients", _flat(current["params"], gradients=True))
                    current["loss"] = None
                    current["logits"] = None
                return value

            def flat(torch_module, parameters):
                value = original_flat(torch_module, parameters)
                current = observer._current
                if current is not None:
                    _require(len(parameters) == len(current["params"]) and all(
                        a is b for a, b in zip(parameters, current["params"])), "final parameter identity differs")
                    observer._post_parameters(parameters, flat_value=value)
                return value

            def candidate(*args, **kwargs):
                value = original_candidate(*args, **kwargs)
                wanted = "client-" + str(len(observer._candidate_ids))
                _require(value.identity == wanted, "original candidate order differs")
                rows = prefix_observer._rounds[1]["clients"]
                _require(rows[-1]["client_id"] == value.identity, "candidate and local-training identities differ")
                array = _array(value.cpu_delta)
                rows[-1]["update"] = tensor_fingerprint(array)
                observer._candidate_ids.append(value.identity)
                index = len(observer._candidate_ids) - 1
                if index in targets:
                    target = observer._targets[index]
                    target["final_delta"] = tensor_fingerprint(array)
                    observer._save(target, "final_delta", array)
                return value

            try:
                for obj, name, replacement in (
                    (fl, "run_experiment", run), (fl, "_local_train_client_delta", local),
                    (fl, "_ClientUpdateCandidate", candidate),
                    (backend.TorchTrainingContext, "local_train_delta_resident", resident),
                    (backend, "_torch_params_from_tensor", params),
                    (backend, "_torch_flat_vector_from_params", flat),
                    (backend, "_torch_forward", forward), (torch.Tensor, "index_select", index_select),
                    (torch.nn.functional, "cross_entropy", cross_entropy), (torch.Tensor, "backward", backward)):
                    install(obj, name, replacement)
                try:
                    yield observer
                except _StopAtNextClient as stopped:
                    if stopped.observer is not observer:
                        raise
                    observer._stopped = True
            finally:
                for obj, name, original in reversed(restorations):
                    setattr(obj, name, original)
    finally:
        _active = False
