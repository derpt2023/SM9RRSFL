"""A declared cuDNN policy change scoped to two original tail backward calls.

No forward, gradient implementation, optimizer, data order, or old source is
replaced. The original step observer supplies scientific values and owns the
controlled client-prefix stop. Flag observations are public provenance only.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cifar_client_step_probe_runtime as step

POLICY_SCHEMA = "cifar-tail-determinism-policy-observation-v1"
POLICIES = ("original", "tail_cudnn_deterministic")
PROFILE_FIELDS = ("cudnn_enabled", "cudnn_deterministic", "cudnn_benchmark",
    "cudnn_allow_tf32", "cuda_matmul_allow_tf32", "deterministic_algorithms",
    "deterministic_warn_only", "float32_matmul_precision", "cudnn_benchmark_limit")
CHECK_FIELDS = ("forward_before", "forward_after", "non_target_backward_before",
                "non_target_backward_after", "post_flat", "exit")
SCOPE = "target_clients_final_minibatch_original_backward"
_active = False


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _profile(torch):
    """Read current flags without changing them or probing/initializing CUDA."""
    return {"cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark_limit": getattr(torch.backends.cudnn, "benchmark_limit", None)}


def _validate_profile(value):
    _require(isinstance(value, dict) and set(value) == set(PROFILE_FIELDS), "incomplete numerical flag profile")
    _require(all(type(value[key]) is bool for key in PROFILE_FIELDS[:7]), "invalid numerical flag boolean")
    _require(value["float32_matmul_precision"] in ("highest", "high", "medium"), "invalid matmul precision")
    limit = value["cudnn_benchmark_limit"]
    _require(limit is None or type(limit) is int and limit >= 0, "invalid cuDNN benchmark limit")


def _validate_baseline(value):
    _validate_profile(value)
    _require(value["cudnn_enabled"] and not value["cudnn_deterministic"]
             and not value["deterministic_algorithms"], "baseline does not permit the declared single-factor intervention")


def validate_policy_observation(payload, task):
    """Pure JSON validator; no Torch import, GPU query, or environment probe.

Call after the original step report's scientific and NPZ validation. Return the
validated ``numerical_policy`` object; never silently repair missing evidence.
"""
    value = payload.get("numerical_policy", {})
    policy = task.get("policy")
    _require(policy in POLICIES and value.get("policy") == policy
             and value.get("schema") == POLICY_SCHEMA, "numerical policy identity differs")
    baseline = value.get("baseline")
    _validate_baseline(baseline)
    _require(value.get("expected_scope") == SCOPE and value.get("other_flags_changed") is False
             and value.get("restored_on_exit") is True, "numerical policy scope/restoration is incomplete")
    _validate_profile(value.get("exit_profile"))
    _require(value["exit_profile"] == baseline, "numerical flags were not restored on exit")
    targets = payload.get("targets", [])
    _require(len(targets) == len(task["target_clients"]) == 2, "exactly two target tail calls are required")
    events = value.get("events", [])
    _require(isinstance(events, list) and len(events) == 2, "exactly two observed tail backward events are required")
    effective = {**baseline, "cudnn_deterministic": policy == "tail_cudnn_deterministic"}
    for index, target, event in zip(task["target_clients"], targets, events):
        _require(target["client_id"] == "client-" + str(index), "policy target order differs")
        batch = target["batches"][-1]
        _require(event.get("client_id") == target["client_id"] and batch["samples"] == 1
                 and all(type(event.get(field)) is int and event[field] == batch[field]
                         for field in ("batch", "epoch", "batch_in_epoch", "samples")),
                 "policy event does not match the actual final single-sample batch")
        for field in ("before", "effective", "restored"):
            _validate_profile(event.get(field))
        _require(event["before"] == event["restored"] == baseline and event["effective"] == effective
                 and event.get("backward_completed") is True, "effective backward flags differ from declared policy")
    rows = payload["prefix"]["rounds"][0]["clients"]
    training = sum(len(client["minibatch_sizes_per_epoch"]) * client["epochs"] for client in rows)
    evaluation = sum(len(item["batches"]) for item in payload["prefix"]["evaluations"])
    expected = {"forward_before": training + evaluation, "forward_after": training + evaluation,
        "non_target_backward_before": training - 2, "non_target_backward_after": training - 2,
        "post_flat": len(rows), "exit": 1}
    checks = value.get("checks", {})
    _require(set(checks) == set(CHECK_FIELDS) and all(type(checks[key]) is int for key in checks)
             and checks == expected, "flag observation coverage differs from actual training/evaluation calls")
    changes = 2 if policy == "tail_cudnn_deterministic" else 0
    _require(type(value.get("scoped_changes")) is int and value["scoped_changes"] == changes,
             "number of scoped flag changes differs")
    return value


def source_evidence(root="/opt/pytorch"):
    """Optionally read installed-source provenance; absence is not a failure.

These text checks do not certify the loaded binary or identify an executed
cuDNN engine. Importing Torch here obtains version metadata only, never CUDA.
"""
    result = {"source_root": str(root), "files": {}, "binary_equivalence_verified": False,
              "executed_algorithm_identified": False,
              "scope": "optional read-only source text and version metadata, not binary or operator proof"}
    try:
        torch = step.backend._torch_module()
        result.update(torch_version=str(torch.__version__), torch_git_version=getattr(torch.version, "git_version", None))
    except Exception as exc:
        result.update(torch_version=None, torch_git_version=None, version_error=str(exc))
    for relative in ("tools/autograd/derivatives.yaml", "aten/src/ATen/native/Convolution.cpp"):
        path = Path(root) / relative
        entry = {"path": str(path), "status": "unavailable"}
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
            if relative.endswith(".yaml"):
                checks = {"backward_queries_global_flags_comment": "flags are queried from the global context" in text
                    and "convolution_backward instead of being passed along from the forward pass" in text}
            else:
                start = text.find("std::tuple<Tensor, Tensor, Tensor> convolution_backward(")
                section = text[start:] if start >= 0 else ""
                checks = {"convolution_backward_found": start >= 0,
                    "backward_reads_current_deterministic_flags": "params.deterministic = ctx.deterministicCuDNN() || ctx.deterministicAlgorithms();" in section,
                    "backward_has_cudnn_dispatch": "cudnn_convolution_backward_stub(" in section}
            entry.update(status="read", sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), pattern_checks=checks)
        except (OSError, UnicodeError) as exc:
            entry["error"] = type(exc).__name__ + ": " + str(exc)
        result["files"][relative] = entry
    return result


class TailPolicyObserver:
    def __init__(self, task, observer, torch):
        self.task, self.step, self.torch = task, observer, torch
        self.baseline = _profile(torch)
        _validate_baseline(self.baseline)
        self.events = []
        self.checks = dict.fromkeys(CHECK_FIELDS, 0)
        self.scoped_changes = 0
        self.exit_profile = None
        self._closed = False

    @property
    def snapshots(self):
        return self.step.snapshots

    def checkpoint(self, state):
        return self.step.checkpoint(state)

    def check(self, boundary):
        profile = _profile(self.torch)
        _require(profile == self.baseline, "numerical policy leaked or changed at " + boundary)
        self.checks[boundary] += 1
        return profile

    def finish(self):
        _require(self._closed, "policy context must finish before evidence is finalized")
        payload = self.step.finish()
        payload["numerical_policy"] = {"schema": POLICY_SCHEMA, "policy": self.task["policy"],
            "baseline": deepcopy(self.baseline), "events": deepcopy(self.events), "checks": dict(self.checks),
            "exit_profile": deepcopy(self.exit_profile), "scoped_changes": self.scoped_changes,
            "expected_scope": SCOPE, "other_flags_changed": False, "restored_on_exit": True}
        description = ("Original numerical flags throughout; the same scope and flag observations are installed in both arms."
            if self.task["policy"] == "original" else
            "Only cudnn.deterministic becomes true during the two declared target final single-sample backward calls; it is restored before original SGD. All other flags, RNG, forward calls and optimizer arithmetic are unchanged.")
        payload["observation_contract"]["policy"] = description
        payload["prefix"]["observation_contract"]["policy"] = description
        validate_policy_observation(payload, self.task)
        json.dumps(payload, allow_nan=False)
        return payload


@contextmanager
def observe(task):
    """Same caller API and controlled stop as the frozen local-step observer."""
    global _active
    _require(not _active, "tail determinism contexts cannot be nested")
    _require(task.get("policy") in POLICIES, "unknown tail numerical policy")
    _require(len(task.get("target_clients", [])) == 2, "exactly two declared targets are required")
    _active = True
    try:
        with step.observe(task) as old:
            torch = step.backend._torch_module()
            observer = TailPolicyObserver(task, old, torch)
            original_backward = torch.Tensor.backward
            original_forward = step.backend._torch_forward
            original_flat = step.backend._torch_flat_vector_from_params

            def forward(*args, **kwargs):
                observer.check("forward_before")
                value = original_forward(*args, **kwargs)
                observer.check("forward_after")
                return value

            def flat(*args, **kwargs):
                value = original_flat(*args, **kwargs)
                observer.check("post_flat")
                return value

            def backward(tensor, *args, **kwargs):
                current = old._current
                selected = (current is not None and current["index"] in task["target_clients"]
                    and current["batch"] is not None
                    and current["batch"]["batch"] == current["expected_batches"] - 1)
                if not selected:
                    observer.check("non_target_backward_before")
                    value = original_backward(tensor, *args, **kwargs)
                    observer.check("non_target_backward_after")
                    return value
                row = current["batch"]
                _require(tensor is current["loss"] and row["samples"] == 1,
                         "declared intervention is not the actual final single-sample loss")
                before = _profile(torch)
                _require(before == observer.baseline, "numerical flags changed before target backward")
                _require(len(observer.events) < 2 and current["index"] == task["target_clients"][len(observer.events)],
                         "target intervention calls are repeated or reordered")
                event = {"client_id": current["target"]["client_id"],
                    **{key: row[key] for key in ("batch", "epoch", "batch_in_epoch", "samples")},
                    "before": before, "effective": None, "restored": None, "backward_completed": False}
                observer.events.append(event)
                try:
                    if task["policy"] == "tail_cudnn_deterministic":
                        # Do NOT use cudnn.flags(deterministic=True): its defaults
                        # also change enabled/benchmark/TF32 in supported Torch.
                        torch.backends.cudnn.deterministic = True
                        observer.scoped_changes += 1
                    event["effective"] = _profile(torch)
                    _require(event["effective"] == {**before,
                        "cudnn_deterministic": task["policy"] == "tail_cudnn_deterministic"},
                        "effective target flags differ from the single declared change")
                    value = original_backward(tensor, *args, **kwargs)
                    event["backward_completed"] = True
                    _require(_profile(torch) == event["effective"], "numerical flags changed inside target backward")
                    return value
                finally:
                    if task["policy"] == "tail_cudnn_deterministic":
                        torch.backends.cudnn.deterministic = before["cudnn_deterministic"]
                    event["restored"] = _profile(torch)
                    _require(event["restored"] == before, "numerical flags were not restored after target backward")

            torch.Tensor.backward = backward
            step.backend._torch_forward = forward
            step.backend._torch_flat_vector_from_params = flat
            try:
                yield observer
            finally:
                torch.Tensor.backward = original_backward
                step.backend._torch_forward = original_forward
                step.backend._torch_flat_vector_from_params = original_flat
                observer.exit_profile = observer.check("exit")
        observer._closed = True
    finally:
        _active = False
