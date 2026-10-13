"""Passive loss/cost observations for the independent clean diagnostic study.

Local loss is the sample-weighted mean of each client's reported minibatch
training loss, NOT the loss of the final global model on all training samples.
Calibration CE reuses the unchanged accuracy forward passes. No extra training,
random draws, gradients, selection on test data, or optimizer changes occur.
"""
from contextlib import contextmanager
import math
import time

import run_cifar_six_from_scratch as base
from sm9rrsfl import torch_backend as backend


@contextmanager
def observe(task, folder):
    original_run = base.experiments.run_experiment
    original_local = base.fl._local_train_client_delta
    original_accuracy = base.fl._evaluate_accuracy
    fp = task["fingerprint"]

    def run(*args, **kwargs):
        resume = kwargs.get("resume_state") or {}
        saved = resume.get("clean_diagnostic_observations", {})
        if saved and saved.get("task_fingerprint") != fp:
            raise ValueError("diagnostic observations belong to another task")
        rows = list(saved.get("rounds", []))
        if resume and not rows:
            raise ValueError("diagnostic checkpoint is missing its loss observations")
        local_losses, calibration = [], {}
        previous_callback = kwargs.get("checkpoint_callback")
        last_tick = time.monotonic()

        def train(*a, **kw):
            delta, stats = original_local(*a, **kw)
            local_losses.append((float(stats.loss), int(stats.samples)))
            return delta, stats

        def accuracy(params, dataset, spec, config, context):
            if context is None:
                raise ValueError("clean diagnostic loss requires the torch backend")
            forward, offset, losses = backend._torch_forward, 0, []

            def capture(torch, parts, x, model_spec):
                nonlocal offset
                logits = forward(torch, parts, x, model_spec)
                labels = context.y_test[offset:offset + len(x)]
                with torch.no_grad():
                    losses.append(float(torch.nn.functional.cross_entropy(
                        logits, labels, reduction="sum").detach().cpu().item()))
                offset += len(x)
                return logits

            backend._torch_forward = capture
            try:
                value = original_accuracy(params, dataset, spec, config, context)
            finally:
                backend._torch_forward = forward
            if offset != len(dataset.y_test) or not offset:
                raise ValueError("calibration loss did not cover exactly the calibration set")
            calibration["ce"] = math.fsum(losses) / offset
            calibration["n"] = offset
            return value

        def checkpoint(state):
            nonlocal last_tick
            rd = int(state["completed_round"])
            # The runner can invoke a final callback for the same round again.
            if not rows or rd > rows[-1]["round"]:
                samples = sum(n for _, n in local_losses)
                import torch
                rows.append({"round": rd,
                    "local_train_loss": (math.fsum(v * n for v, n in local_losses) / samples
                                         if samples else None),
                    "local_train_samples": samples,
                    "calibration_loss": calibration.get("ce"),
                    "calibration_samples": calibration.get("n"),
                    "round_wall_seconds": time.monotonic() - last_tick if rd else None,
                    "cuda_peak_allocated_mib": (torch.cuda.max_memory_allocated() / 2**20
                                                if torch.cuda.is_available() else None)})
                local_losses.clear()
            payload = {"task_fingerprint": fp, "rounds": rows}
            # Loss observations are checkpointed with exactly the same model round.
            state["clean_diagnostic_observations"] = payload
            if previous_callback:
                previous_callback(state)
            base.write_json(folder / "observations.json", payload)
            last_tick = time.monotonic()

        base.fl._local_train_client_delta = train
        base.fl._evaluate_accuracy = accuracy
        kwargs["checkpoint_callback"] = checkpoint
        try:
            result = original_run(*args, **kwargs)
            base.write_json(folder / "observations.json", {"task_fingerprint": fp, "rounds": rows})
            return result
        finally:
            base.fl._local_train_client_delta = original_local
            base.fl._evaluate_accuracy = original_accuracy

    base.experiments.run_experiment = run
    try:
        yield
    finally:
        base.experiments.run_experiment = original_run
