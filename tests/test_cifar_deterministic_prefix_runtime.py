"""Small real CPU SGD checks; no assertion of CUDA determinism is made here."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import unittest
from unittest import mock

import numpy as np
import torch

import cifar_deterministic_prefix_runtime as runtime
import cifar_prefix_probe_runtime as prefix
import cifar_prefix_probe_report as report
from sm9rrsfl import fl
from sm9rrsfl import torch_backend as backend
from sm9rrsfl.model import ModelSpec
from tests import test_cifar_prefix_probe_runtime as prior
from tests import test_cifar_client_step_probe_runtime as step_tests


class DeterministicPrefixRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = prior.synthetic_dataset()
        cls.config = fl.ExperimentConfig(method="sm9rrs", malicious_ratio=0., num_clients=3,
            rounds=3, partition="dirichlet", seed=2201, compute_backend="torch", device="cpu",
            crypto_mode="simulated", detector_window=20, attack_start_round=25, attack_target_count=2,
            checkpoint_interval=1, early_stop=False, batch_size=4, local_epochs=1, sm9_workers=1)
        cls.task = {"task_id": "cpu-deterministic-prefix", "fingerprint": "cpu-fingerprint",
                    "config": asdict(cls.config), "policy": "original"}
        cls.baseline = runtime._profile(torch)
        cls.rng_before = prior.rng_state()
        with prefix.observe(cls.task) as observer:
            cls.original = fl.run_experiment(cls.data, cls.config, checkpoint_callback=observer.checkpoint)
        cls.original_payload = observer.finish(cls.original)
        cls.payloads, cls.results, cls.backward_flags, cls.forward_flags = {}, {}, {}, {}
        for policy in runtime.POLICIES:
            task = {**deepcopy(cls.task), "policy": policy}
            result, payload, backward, forward = cls.run_observed(task)
            cls.payloads[policy], cls.results[policy] = payload, result
            cls.backward_flags[policy], cls.forward_flags[policy] = backward, forward
        cls.rng_after = prior.rng_state()

    @classmethod
    def run_observed(cls, task):
        backflags, forwardflags = [], []
        raw_backward, raw_forward = torch.Tensor.backward, backend._torch_forward
        def backward(tensor, *args, **kwargs):
            backflags.append(runtime._profile(torch))
            return raw_backward(tensor, *args, **kwargs)
        def forward(*args, **kwargs):
            forwardflags.append(runtime._profile(torch))
            return raw_forward(*args, **kwargs)
        with mock.patch.object(torch.Tensor, "backward", new=backward):
            with mock.patch.object(backend, "_torch_forward", new=forward):
                with runtime.observe(task) as observer:
                    result = fl.run_experiment(cls.data, fl.ExperimentConfig(**task["config"]),
                        checkpoint_callback=observer.checkpoint)
                payload = observer.finish(result)
        return result, payload, backflags, forwardflags

    def test_original_arm_matches_every_frozen_prefix_scientific_field_and_rng_bitwise(self):
        observed = self.payloads["original"]
        for key, value in self.original_payload.items():
            if key != "observation_contract":
                self.assertEqual(observed[key], value, key)
        self.assertEqual(self.original.records, self.results["original"].records)
        self.assertEqual(self.original.diagnostics, self.results["original"].diagnostics)
        self.assertEqual(self.rng_before, self.rng_after)
        self.assertEqual(runtime._profile(torch), self.baseline)

    def test_actual_all_client_batches_cover_three_rounds_and_six_singletons_without_fixed_ids(self):
        payload = self.payloads["original"]
        self.assertEqual(len(payload["actual_batches"]), 9)
        self.assertEqual([(r["round"], r["client_id"]) for r in payload["singleton_batches"]],
                         [(rd, cid) for rd in (1, 2, 3) for cid in ("client-0", "client-2")])
        expected = [[4, 4, 1], [4, 4], [4, 4, 4, 1]] * 3
        self.assertEqual([row["batch_sizes"] for row in payload["actual_batches"]], expected)
        for row in payload["actual_batches"]:
            self.assertEqual(row["forward_count"], len(row["batch_sizes"]))
            self.assertEqual(row["backward_count"], len(row["batch_sizes"]))
        for row in payload["singleton_batches"]:
            self.assertEqual(row["samples"], 1)
            self.assertEqual(row["features"]["shape"], [1, 1, 8, 8])
            self.assertEqual(row["labels"]["shape"], [1])
            self.assertEqual(len(row["parameter_layout"]), 4)

    def test_only_six_raw_backward_calls_switch_and_all_original_forward_flags_are_preserved(self):
        for policy in runtime.POLICIES:
            payload, backward, forward = self.payloads[policy], self.backward_flags[policy], self.forward_flags[policy]
            self.assertEqual(len(backward), 27)
            expected = [False, False, True, False, False, False, False, False, True] * 3
            self.assertEqual([row["cudnn_deterministic"] for row in backward],
                             expected if policy != "original" else [False] * 27)
            for row in backward:
                self.assertEqual({k: v for k, v in row.items() if k != "cudnn_deterministic"},
                                 {k: v for k, v in self.baseline.items() if k != "cudnn_deterministic"})
            self.assertEqual(forward, [self.baseline] * 35)
            self.assertEqual(payload["numerical_policy"]["scoped_changes"], 6 if policy != "original" else 0)
            self.assertEqual(payload["numerical_policy"]["checks"], {"forward_before": 35, "forward_after": 35,
                "non_target_backward_before": 21, "non_target_backward_after": 21, "post_flat": 9, "exit": 1})

    def test_both_strict_validators_accept_and_payload_contains_only_json_no_arrays_or_npz(self):
        for policy in runtime.POLICIES:
            task, payload = {**self.task, "policy": policy}, self.payloads[policy]
            report.validate_observations(payload, task)
            with mock.patch.object(backend, "_torch_module", side_effect=AssertionError("no Torch access")):
                self.assertIs(runtime.validate_policy_observation(payload, task), payload["numerical_policy"])
            encoded = json.dumps(payload, allow_nan=False)
            for forbidden in ("snapshot_file", '"snapshots"', '"task_tag"', '"crypto_state"'):
                self.assertNotIn(forbidden, encoded)
        self.assertEqual(self.payloads["original"]["singleton_batches"],
                         self.payloads["singleton_backward_cudnn_deterministic"]["singleton_batches"])

    def test_real_cifar_ten_parameter_branch_preserves_original_and_exactly_six_scopes(self):
        spec = ModelSpec(input_shape=(1, 8, 8), num_classes=10, architecture="cifar10",
                         cifar_conv_filters=(2, 3), cifar_hidden_dims=(7, 5))
        before = prior.rng_state()
        with mock.patch.object(fl, "model_spec_for_dataset", return_value=spec):
            with prefix.observe(self.task) as old:
                result = fl.run_experiment(self.data, self.config, checkpoint_callback=old.checkpoint)
            old_payload = old.finish(result)
            for policy in runtime.POLICIES:
                task = {**self.task, "policy": policy}
                _, payload, backwards, forwards = self.run_observed(task)
                report.validate_observations(payload, task)
                runtime.validate_policy_observation(payload, task)
                self.assertTrue(all(len(row["parameter_layout"]) == 10 for row in payload["singleton_batches"]))
                self.assertEqual(sum(row["cudnn_deterministic"] for row in backwards), 6 if policy != "original" else 0)
                self.assertEqual(forwards, [self.baseline] * 35)
                if policy == "original":
                    for key in old_payload:
                        if key != "observation_contract":
                            self.assertEqual(payload[key], old_payload[key], key)
        self.assertEqual(prior.rng_state(), before)

    def test_actual_singleton_indices_are_selected_original_permutation_positions(self):
        partitions = fl.partition_clients(self.data.y_train, 3, strategy="dirichlet", seed=2201, dirichlet_alpha=.5)
        for row in self.payloads["original"]["singleton_batches"]:
            client = int(row["client_id"].split("-")[1])
            order = np.random.default_rng(self.config.seed + row["round"] * 1009 + client).permutation(len(partitions[client]))
            local = order[row["batch"] * 4: row["batch"] * 4 + 1]
            global_indices = partitions[client][local]
            self.assertEqual(row["local_indices"], prefix.tensor_fingerprint(local))
            self.assertEqual(row["dataset_indices"], prefix.tensor_fingerprint(global_indices))
            self.assertEqual(row["features"], prefix.tensor_fingerprint(self.data.x_train[global_indices]))
            self.assertEqual(row["labels"], prefix.tensor_fingerprint(self.data.y_train[global_indices]))

    def test_failure_inside_singleton_backward_restores_flag_hooks_and_rejects_finalization(self):
        hooks, original = step_tests.hooks(), torch.Tensor.backward
        task = {**self.task, "policy": runtime.POLICIES[1]}
        def fail(tensor, *args, **kwargs):
            if torch.backends.cudnn.deterministic:
                raise RuntimeError("selected backward failure")
            return original(tensor, *args, **kwargs)
        with mock.patch.object(torch.Tensor, "backward", new=fail):
            with self.assertRaisesRegex(RuntimeError, "selected backward"):
                with runtime.observe(task) as observer:
                    fl.run_experiment(self.data, self.config, checkpoint_callback=observer.checkpoint)
        self.assertEqual(observer.events[0]["restored"], self.baseline)
        self.assertFalse(observer.events[0]["backward_completed"])
        with self.assertRaisesRegex(ValueError, "successfully"):
            observer.finish(self.original)
        for obj, name, original in hooks:
            self.assertIs(getattr(obj, name), original)
        self.assertEqual(runtime._profile(torch), self.baseline)
        self.assertFalse(runtime._active)
        self.assertFalse(prefix._active)

    def test_nested_keyboard_interrupt_and_wrong_config_restore_without_rng_changes(self):
        hooks, rng = step_tests.hooks(), prior.rng_state()
        with self.assertRaises(KeyboardInterrupt):
            with runtime.observe(self.task):
                with self.assertRaisesRegex(ValueError, "nested"):
                    with runtime.observe(self.task):
                        pass
                raise KeyboardInterrupt
        for obj, name, original in hooks:
            self.assertIs(getattr(obj, name), original)
        for updates in ({"rounds": 25}, {"malicious_ratio": .1}, {"method": "fedavg"}, {"compute_backend": "numpy"}):
            task = deepcopy(self.task)
            task["config"].update(updates)
            with self.assertRaisesRegex(ValueError, "three clean"):
                with runtime.observe(task):
                    pass
        with runtime.observe(self.task) as observer:
            with self.assertRaisesRegex(ValueError, "fresh"):
                fl.run_experiment(self.data, self.config, resume_state={"completed_round": 1})
            with self.assertRaisesRegex(ValueError, "identity"):
                fl.run_experiment(self.data, replace(self.config, seed=1))
        self.assertEqual(prior.rng_state(), rng)
        self.assertEqual(runtime._profile(torch), self.baseline)

    def test_validator_rejects_missing_batch_event_and_singleton_tensor_or_flag_tampering(self):
        valid = self.payloads[runtime.POLICIES[1]]
        mutations = [lambda p: p["actual_batches"].pop(),
            lambda p: p["actual_batches"][0]["batch_sizes"].__setitem__(0, 1),
            lambda p: p["actual_batches"][0].__setitem__("backward_count", 2),
            lambda p: p["singleton_batches"].pop(),
            lambda p: p["singleton_batches"][0]["features"].__setitem__("shape", [2, 1, 8, 8]),
            lambda p: p["numerical_policy"]["events"][0].__setitem__("round", 2),
            lambda p: p["numerical_policy"]["events"][0]["effective"].__setitem__("cudnn_allow_tf32", False),
            lambda p: p["numerical_policy"]["checks"].__setitem__("post_flat", 8),
            lambda p: p["singleton_batches"][0]["loss"].__setitem__("value", float("nan"))]
        for mutation in mutations:
            payload = deepcopy(valid)
            mutation(payload)
            with self.assertRaises(ValueError):
                runtime.validate_policy_observation(payload, {**self.task, "policy": runtime.POLICIES[1]})

    def test_scope_depends_on_actual_size_including_non_tail_singletons_and_zero_singleton_runs(self):
        for size in (1, 5):
            task = deepcopy(self.task)
            task["config"]["batch_size"] = size
            task["policy"] = runtime.POLICIES[1]
            _, payload, backwards, _ = self.run_observed(task)
            expected = 90 if size == 1 else 0
            self.assertEqual(len(payload["singleton_batches"]), expected)
            self.assertEqual(sum(row["cudnn_deterministic"] for row in backwards), expected)
            runtime.validate_policy_observation(payload, task)
        self.assertEqual(runtime._profile(torch), self.baseline)

    def test_validator_binds_singleton_shapes_dtypes_and_exit_booleans_to_original_model(self):
        valid = self.payloads[runtime.POLICIES[1]]
        cases = []
        value = deepcopy(valid)
        value["singleton_batches"][0]["logits"]["shape"] = [1, 9]
        cases.append(value)
        value = deepcopy(valid)
        value["singleton_batches"][0]["gradients"]["dtype"] = "<f8"
        cases.append(value)
        value = deepcopy(valid)
        batch = value["singleton_batches"][0]
        batch["parameter_layout"] = batch["parameter_layout"][:1]
        for field in ("pre_parameters", "gradients", "post_parameters"):
            batch[field]["shape"] = [batch["parameter_layout"][0]["size"]]
        cases.append(value)
        value = deepcopy(valid)
        value["numerical_policy"]["exit_profile"]["cudnn_deterministic"] = 0
        cases.append(value)
        for value in cases:
            with self.assertRaises(ValueError):
                runtime.validate_policy_observation(value, {**self.task, "policy": runtime.POLICIES[1]})


if __name__ == "__main__":
    unittest.main()
