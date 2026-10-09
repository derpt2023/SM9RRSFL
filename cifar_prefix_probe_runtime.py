"""Passive fingerprints around the original three-round clean training prefix.

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

import numpy as np

from sm9rrsfl import fl
from sm9rrsfl import torch_backend as backend


SCHEMA = "cifar-prefix-probe-observation-v1"
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


class PrefixObserver:
    def __init__(self, task):
        self.task = task
        self.payload = {"schema": SCHEMA, "task_id": task["task_id"],
            "task_fingerprint": task["fingerprint"], "data": {}, "model_spec": None,
            "partition": None, "initial_model": None, "rounds": [], "evaluations": [],
            "checkpoints": [], "observation_contract": {
                "actual_values": "model inputs, returned client updates, coefficients, aggregates, models, and original evaluation logits",
                "epoch_indices": "reconstructed global sample-index permutations from the original per-client seed using a separate local NumPy Generator",
                "predictions": "CPU argmax of the captured original logits; no second model forward is executed",
                "synchronization": "fingerprinting CUDA tensors adds host synchronization and may alter timing or scheduling",
                "policy": "no numerical flags, random seeds, model values, or scientific rules are set or changed",
                "scope": "fresh clean three-round diagnostic only; no accuracy qualification or algorithm selection"}}
        self._round = 0
        self._rounds = {}
        self._checkpoints = {}
        self._tags = {}
        self._evaluation = None
        self._started = False
        self._finished = False

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
            "diagnostic_count": sum(d.round == rd for d in state.get("diagnostics", []))}
        previous = self._checkpoints.get(rd)
        if previous is not None:
            if {k: v for k, v in previous.items() if k != "callback_count"} != row:
                raise ValueError("repeated checkpoint changed the observed scientific round")
            previous["callback_count"] += 1
        else:
            self._checkpoints[rd] = {**row, "callback_count": 1}

    def finish(self, result):
        """Return JSON-safe complete evidence; missing observations are never invented."""
        if self._finished:
            raise ValueError("prefix observer can only be finalized once")
        if not self._started or result.stopped_round != 3 or result.config.rounds != 3:
            raise ValueError("a fresh complete three-round result is required")
        if set(self._rounds) != {1, 2, 3} or set(self._checkpoints) != {0, 1, 2, 3}:
            raise ValueError("missing training rounds or checkpoint observations")
        if [(r["round"], r["kind"]) for r in self.payload["evaluations"]] != [
                (rd, kind) for rd in range(4) for kind in ("accuracy", "target")]:
            raise ValueError("missing or repeated original evaluation calls")
        records = {record.round: record for record in result.records}
        if set(records) != {0, 1, 2, 3}:
            raise ValueError("result records do not cover the complete prefix")
        identities = [client["client_id"] for client in self.payload["partition"]["clients"]]
        for rd in range(1, 4):
            row = self._rounds[rd]
            if [client["client_id"] for client in row["clients"]] != identities:
                raise ValueError("missing or reordered local training observations")
            for client in row["clients"]:
                if "update" not in client or not math.isfinite(client["stats"]["loss"]):
                    raise ValueError("missing update or nonfinite local statistics")
            if any(key not in row for key in ("coefficients", "aggregate", "post_model")):
                raise ValueError("missing aggregation observations")
            if row["post_model"] != self._checkpoints[rd]["model"]:
                raise ValueError("post-aggregation model disagrees with checkpoint observation")
            row["record"] = asdict(records[rd])
            row["diagnostics"] = [_scientific_diagnostic(d) for d in result.diagnostics if d.round == rd]
        self.payload["rounds"] = [self._rounds[rd] for rd in range(1, 4)]
        self.payload["checkpoints"] = [self._checkpoints[rd] for rd in range(4)]
        self.payload["configuration"] = asdict(result.config)
        # Validate that no NaN, tensor, secret object or non-JSON type escaped.
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
    observer = PrefixObserver(task)
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

    def run(dataset, config, *args, **kwargs):
        if observer._started or kwargs.get("resume_state") is not None:
            raise ValueError("prefix probes require one fresh run per process")
        if config.rounds != 3 or config.malicious_ratio != 0 or config.method != "sm9rrs":
            raise ValueError("prefix observation requires three clean original Ours rounds")
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
        observer._round_row()["clients"].append(row)
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
        result = original_process(candidates, *args, **kwargs)
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

    _active = True
    try:
        for obj, name, replacement in (
            (fl, "run_experiment", run), (fl, "partition_clients", partition),
            (fl, "init_params", initialized), (fl, "_local_train_client_delta", local),
            (fl, "_process_sm9_candidates", processed),
            (fl, "bounded_aggregation_coefficients", coefficients),
            (fl, "aggregate_with_coefficients", aggregate),
            (backend.TorchTrainingContext, "add_update", added),
            (fl, "_evaluate_accuracy", accuracy), (fl, "_evaluate_attack_target_metrics", target),
            (backend, "_torch_forward", forward),
        ):
            install(obj, name, replacement)
        yield observer
    finally:
        for obj, name, original in reversed(restorations):
            setattr(obj, name, original)
        _active = False
