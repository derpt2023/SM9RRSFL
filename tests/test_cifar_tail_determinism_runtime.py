"""CPU checks for policy scope; they do not assert CUDA reproducibility."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import cifar_tail_determinism_runtime as runtime
import cifar_client_step_probe_runtime as step
import cifar_client_step_probe_report as report
from sm9rrsfl import fl
from sm9rrsfl import torch_backend as backend
from sm9rrsfl.model import ModelSpec
from tests import test_cifar_client_step_probe_runtime as original_tests


class TailDeterminismRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = original_tests.ClientStepProbeRuntimeTests
        cls.reference.setUpClass()
        cls.data, cls.config = cls.reference.data, cls.reference.config
        cls.tasks, cls.payloads, cls.arrays, cls.backward_flags, cls.forward_flags = {}, {}, {}, {}, {}
        cls.rng_before = original_tests.rng_state()
        cls.baseline = runtime._profile(torch)
        raw_backward, raw_forward = torch.Tensor.backward, backend._torch_forward
        for policy in runtime.POLICIES:
            task = {**deepcopy(cls.reference.task), "policy": policy,
                    "task_id": "tail-cpu-" + policy, "fingerprint": "tail-id-" + policy}
            cls.tasks[policy] = task
            actual_backward, actual_forward = [], []
            def backward(tensor, *args, **kwargs):
                actual_backward.append(runtime._profile(torch))
                return raw_backward(tensor, *args, **kwargs)
            def forward(*args, **kwargs):
                actual_forward.append(runtime._profile(torch))
                return raw_forward(*args, **kwargs)
            with mock.patch.object(torch.Tensor, "backward", new=backward):
                with mock.patch.object(backend, "_torch_forward", new=forward):
                    with runtime.observe(task) as observer:
                        fl.run_experiment(cls.data, cls.config, checkpoint_callback=observer.checkpoint)
            cls.payloads[policy], cls.arrays[policy] = observer.finish(), observer.snapshots
            cls.backward_flags[policy], cls.forward_flags[policy] = actual_backward, actual_forward
        cls.rng_after = original_tests.rng_state()

    def test_original_arm_matches_frozen_step_science_arrays_and_rng_bitwise(self):
        payload = self.payloads["original"]
        for key in ("targets", "snapshots", "stop", "configuration"):
            self.assertEqual(payload[key], self.reference.payload[key])
        for key, array in self.arrays["original"].items():
            np.testing.assert_array_equal(array, self.reference.snapshots[key])
        self.assertEqual(self.rng_before, self.rng_after)
        self.assertEqual(self.baseline, runtime._profile(torch))

    def test_only_two_actual_backward_calls_observe_true_and_every_other_flag_is_unchanged(self):
        self.assertEqual(len(self.backward_flags["original"]), 4)
        self.assertEqual(self.backward_flags["original"], [self.baseline] * 4)
        observed = self.backward_flags["tail_cudnn_deterministic"]
        self.assertEqual([row["cudnn_deterministic"] for row in observed], [False, True, False, True])
        for profile in observed:
            self.assertEqual({k: v for k, v in profile.items() if k != "cudnn_deterministic"},
                             {k: v for k, v in self.baseline.items() if k != "cudnn_deterministic"})
        for policy in runtime.POLICIES:
            evidence = self.payloads[policy]["numerical_policy"]
            self.assertEqual([event["client_id"] for event in evidence["events"]], ["client-0", "client-1"])
            self.assertEqual(evidence["scoped_changes"], 2 if policy != "original" else 0)
            for event in evidence["events"]:
                self.assertEqual(event["samples"], 1)
                self.assertEqual(event["before"], self.baseline)
                self.assertEqual(event["restored"], self.baseline)
                self.assertTrue(event["backward_completed"])

    def test_every_forward_is_original_flags_and_complete_step_validator_accepts_both_arms(self):
        for policy in runtime.POLICIES:
            self.assertEqual(self.forward_flags[policy], [self.baseline] * 6)
            payload, task = self.payloads[policy], self.tasks[policy]
            report.validate_observations(payload, task, self.arrays[policy])
            self.assertIs(runtime.validate_policy_observation(payload, task), payload["numerical_policy"])
            self.assertEqual(payload["numerical_policy"]["checks"], {"forward_before": 6,
                "forward_after": 6, "non_target_backward_before": 2,
                "non_target_backward_after": 2, "post_flat": 2, "exit": 1})
        for before, after in zip(self.payloads["original"]["targets"], self.payloads["tail_cudnn_deterministic"]["targets"]):
            # CPU proof of no replacement forward/SGD; not a prediction for CUDA.
            self.assertEqual(before, after)

    def test_cifar_ten_parameter_two_convolution_path_preserves_original_and_scopes_both_arms(self):
        spec = ModelSpec(input_shape=(1, 8, 8), num_classes=10, architecture="cifar10",
                         cifar_conv_filters=(2, 3), cifar_hidden_dims=(7, 5))
        initial_rng = original_tests.rng_state()
        raw_backward, raw_forward = torch.Tensor.backward, backend._torch_forward
        with mock.patch.object(fl, "model_spec_for_dataset", return_value=spec):
            with step.observe(self.tasks["original"]) as old:
                fl.run_experiment(self.data, self.config, checkpoint_callback=old.checkpoint)
            old_payload = old.finish()
            for policy in runtime.POLICIES:
                backward_flags, forward_flags = [], []
                def backward(tensor, *args, **kwargs):
                    backward_flags.append(runtime._profile(torch))
                    return raw_backward(tensor, *args, **kwargs)
                def forward(*args, **kwargs):
                    forward_flags.append(runtime._profile(torch))
                    return raw_forward(*args, **kwargs)
                with mock.patch.object(torch.Tensor, "backward", new=backward):
                    with mock.patch.object(backend, "_torch_forward", new=forward):
                        with runtime.observe(self.tasks[policy]) as observer:
                            fl.run_experiment(self.data, self.config, checkpoint_callback=observer.checkpoint)
                payload = observer.finish()
                report.validate_observations(payload, self.tasks[policy], observer.snapshots)
                runtime.validate_policy_observation(payload, self.tasks[policy])
                self.assertEqual([value["cudnn_deterministic"] for value in backward_flags],
                                 [False, True, False, True] if policy != "original" else [False] * 4)
                self.assertEqual(forward_flags, [self.baseline] * 6)
                self.assertTrue(all(len(target["parameter_layout"]) == 10 for target in payload["targets"]))
                self.assertEqual(payload["numerical_policy"]["checks"], self.payloads[policy]["numerical_policy"]["checks"])
                if policy == "original":
                    self.assertEqual(payload["targets"], old_payload["targets"])
                    for key, array in observer.snapshots.items():
                        np.testing.assert_array_equal(array, old.snapshots[key])
                self.assertEqual(runtime._profile(torch), self.baseline)
        self.assertEqual(initial_rng, original_tests.rng_state())

    def test_failure_inside_true_scope_restores_flag_and_all_hooks(self):
        before = original_tests.hooks()
        raw_backward = torch.Tensor.backward
        seen = []
        def failing(tensor, *args, **kwargs):
            if torch.backends.cudnn.deterministic:
                seen.append(runtime._profile(torch))
                raise RuntimeError("failure inside selected backward")
            return raw_backward(tensor, *args, **kwargs)
        with mock.patch.object(torch.Tensor, "backward", new=failing):
            with self.assertRaisesRegex(RuntimeError, "inside selected"):
                with runtime.observe(self.tasks["tail_cudnn_deterministic"]) as observer:
                    fl.run_experiment(self.data, self.config, checkpoint_callback=observer.checkpoint)
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0]["cudnn_deterministic"])
        self.assertEqual(runtime._profile(torch), self.baseline)
        self.assertEqual(observer.events[0]["restored"], self.baseline)
        self.assertFalse(observer.events[0]["backward_completed"])
        with self.assertRaisesRegex(ValueError, "context must finish"):
            observer.finish()
        for obj, name, original in before:
            self.assertIs(getattr(obj, name), original)
        self.assertFalse(runtime._active)
        self.assertFalse(step._active)

    def test_nested_and_unrelated_exception_restore_context_without_rng_use(self):
        before, rng = original_tests.hooks(), original_tests.rng_state()
        with self.assertRaises(KeyboardInterrupt):
            with runtime.observe(self.tasks["original"]):
                with self.assertRaisesRegex(ValueError, "nested"):
                    with runtime.observe(self.tasks["original"]):
                        pass
                raise KeyboardInterrupt
        self.assertEqual(rng, original_tests.rng_state())
        self.assertEqual(self.baseline, runtime._profile(torch))
        for obj, name, original in before:
            self.assertIs(getattr(obj, name), original)

    def test_validator_rejects_missing_events_other_flag_changes_and_bad_coverage_without_torch(self):
        valid = self.payloads["tail_cudnn_deterministic"]
        cases = []
        value = deepcopy(valid)
        value["numerical_policy"]["events"].pop()
        cases.append(value)
        value = deepcopy(valid)
        value["numerical_policy"]["events"][0]["effective"]["cudnn_enabled"] = False
        cases.append(value)
        value = deepcopy(valid)
        value["numerical_policy"]["checks"]["forward_before"] -= 1
        cases.append(value)
        value = deepcopy(valid)
        value["numerical_policy"]["events"][1]["batch"] = 0
        cases.append(value)
        value = deepcopy(valid)
        value["numerical_policy"]["exit_profile"]["cudnn_deterministic"] = True
        cases.append(value)
        with mock.patch.object(backend, "_torch_module", side_effect=AssertionError("Torch must not be accessed by validation")):
            runtime.validate_policy_observation(valid, self.tasks["tail_cudnn_deterministic"])
            for value in cases:
                with self.assertRaises(ValueError):
                    runtime.validate_policy_observation(value, self.tasks["tail_cudnn_deterministic"])

    def test_disabled_cudnn_or_already_deterministic_baseline_is_rejected_without_mutation(self):
        before = original_tests.hooks()
        for field, value in (("cudnn_enabled", False), ("cudnn_deterministic", True), ("deterministic_algorithms", True)):
            profile = {**self.baseline, field: value}
            with mock.patch.object(runtime, "_profile", return_value=profile):
                with self.assertRaisesRegex(ValueError, "baseline"):
                    with runtime.observe(self.tasks["tail_cudnn_deterministic"]):
                        pass
        for obj, name, original in before:
            self.assertIs(getattr(obj, name), original)
        self.assertEqual(runtime._profile(torch), self.baseline)

    def test_wrong_actual_tail_size_fails_before_applying_policy(self):
        task = deepcopy(self.tasks["tail_cudnn_deterministic"])
        task["config"]["batch_size"] = 4
        config = fl.ExperimentConfig(**task["config"])
        with self.assertRaisesRegex(ValueError, "single-sample"):
            with runtime.observe(task) as observer:
                fl.run_experiment(self.data, config, checkpoint_callback=observer.checkpoint)
        self.assertEqual(observer.scoped_changes, 0)
        self.assertEqual(runtime._profile(torch), self.baseline)

    def test_optional_source_evidence_is_read_only_nonblocking_and_never_claims_binary_proof(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            missing = runtime.source_evidence(root)
            self.assertTrue(all(row["status"] == "unavailable" for row in missing["files"].values()))
            sources = {
                "tools/autograd/derivatives.yaml": "flags are queried from the global context\nconvolution_backward instead of being passed along from the forward pass",
                "aten/src/ATen/native/Convolution.cpp": "std::tuple<Tensor, Tensor, Tensor> convolution_backward(\nparams.deterministic = ctx.deterministicCuDNN() || ctx.deterministicAlgorithms();\ncudnn_convolution_backward_stub("}
            for relative, text in sources.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            before = {relative: (root / relative).read_bytes() for relative in sources}
            with mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("no GPU probe")):
                with mock.patch.object(torch.cuda, "device_count", side_effect=AssertionError("no GPU probe")):
                    evidence = runtime.source_evidence(root)
            self.assertFalse(evidence["binary_equivalence_verified"])
            self.assertFalse(evidence["executed_algorithm_identified"])
            self.assertEqual(evidence["torch_git_version"], torch.version.git_version)
            for relative, contents in before.items():
                self.assertEqual((root / relative).read_bytes(), contents)
                self.assertEqual(evidence["files"][relative]["sha256"], hashlib.sha256(contents).hexdigest())
                self.assertTrue(all(evidence["files"][relative]["pattern_checks"].values()))
            json.dumps(evidence, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
