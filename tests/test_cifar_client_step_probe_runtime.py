"""Small CPU experiments verify actual-batch observation and controlled stop."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import pickle
import random
import unittest
from unittest import mock

import numpy as np
import torch

import cifar_client_step_probe_runtime as runtime
import cifar_prefix_probe_runtime as prefix
from sm9rrsfl import fl
from sm9rrsfl import torch_backend as backend
from sm9rrsfl.datasets import ImageDataset
from sm9rrsfl.model import ModelSpec


def dataset():
    rng = np.random.default_rng(2817)
    return ImageDataset(rng.normal(0, .1, (18, 1, 8, 8)).astype(np.float32),
        np.arange(18, dtype=np.int64) % 10,
        rng.normal(0, .1, (20, 1, 8, 8)).astype(np.float32),
        np.arange(20, dtype=np.int64) % 10, name="step-synthetic", num_classes=10)


def rng_state():
    return pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()), torch.get_rng_state().numpy().tobytes()


def hooks():
    return [(obj, name, getattr(obj, name)) for obj, name in (
        (fl, "run_experiment"), (fl, "partition_clients"), (fl, "init_params"),
        (fl, "_local_train_client_delta"), (fl, "_ClientUpdateCandidate"),
        (fl, "_process_sm9_candidates"), (fl, "bounded_aggregation_coefficients"),
        (fl, "aggregate_with_coefficients"), (fl, "_evaluate_accuracy"),
        (fl, "_evaluate_attack_target_metrics"), (backend.TorchTrainingContext, "add_update"),
        (backend.TorchTrainingContext, "local_train_delta_resident"),
        (backend, "_torch_params_from_tensor"), (backend, "_torch_flat_vector_from_params"),
        (backend, "_torch_forward"), (torch.Tensor, "index_select"),
        (torch.Tensor, "backward"), (torch.nn.functional, "cross_entropy"))]


class ClientStepProbeRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = dataset()
        cls.config = fl.ExperimentConfig(method="sm9rrs", malicious_ratio=0., num_clients=3,
            rounds=3, partition="iid", seed=2201, compute_backend="torch", device="cpu",
            crypto_mode="simulated", detector_window=20, attack_start_round=25, attack_target_count=2,
            checkpoint_interval=1, early_stop=False, batch_size=5, local_epochs=1, sm9_workers=1)
        cls.task = {"task_id": "client-step-cpu", "fingerprint": "synthetic-step-id",
                    "config": asdict(cls.config), "target_clients": [0, 1], "stop_after_client": 1}
        cls.plain_candidates = []
        original_candidate = fl._ClientUpdateCandidate
        def collect(*args, **kwargs):
            value = original_candidate(*args, **kwargs)
            cls.plain_candidates.append((value.identity, value.cpu_delta.copy()))
            return value
        cls.plain_states = []
        cls.rng_before = rng_state()
        with mock.patch.object(fl, "_ClientUpdateCandidate", side_effect=collect):
            cls.plain_result = fl.run_experiment(cls.data, cls.config, checkpoint_callback=cls.plain_states.append)
        cls.rng_after_plain = rng_state()
        cls.forward_calls = 0
        original_forward = backend._torch_forward
        def forward(*args, **kwargs):
            cls.forward_calls += 1
            return original_forward(*args, **kwargs)
        with mock.patch.object(backend, "_torch_forward", side_effect=forward):
            with mock.patch.object(fl, "_process_sm9_candidates", side_effect=AssertionError("aggregation reached")):
                with runtime.observe(cls.task) as observer:
                    fl.run_experiment(cls.data, cls.config, checkpoint_callback=observer.checkpoint)
        cls.payload = observer.finish()
        cls.snapshots = observer.snapshots
        cls.rng_after_observed = rng_state()

    def test_original_cpu_outputs_initialization_and_rng_are_preserved(self):
        self.assertEqual(self.rng_before, self.rng_after_plain)
        self.assertEqual(self.rng_before, self.rng_after_observed)
        self.assertEqual(self.payload["prefix"]["initial_model"], runtime.tensor_fingerprint(self.plain_states[0]["params"]))
        self.assertEqual(self.payload["prefix"]["checkpoints"][0]["record"], asdict(self.plain_result.records[0]))
        for target, (cid, delta) in zip(self.payload["targets"], self.plain_candidates):
            self.assertEqual(target["client_id"], cid)
            np.testing.assert_array_equal(self.snapshots[target["tail_snapshots"]["final_delta"]], delta)

    def test_original_candidate_prefix_completes_and_next_client_and_aggregation_do_not_run(self):
        row = self.payload["prefix"]["rounds"][0]
        self.assertEqual([item["client_id"] for item in row["clients"]], ["client-0", "client-1"])
        self.assertEqual(self.payload["stop"]["next_client_not_trained"], "client-2")
        self.assertEqual(self.payload["stop"]["completed_rounds"], 0)
        self.assertFalse(self.payload["stop"]["crypto_finalized"])
        self.assertNotIn("aggregate", row)
        self.assertEqual(self.forward_calls, 6)  # Two original evaluations + four real minibatches.
        self.assertEqual([row["round"] for row in self.payload["prefix"]["checkpoints"]], [0])

    def test_every_actual_batch_boundary_and_sgd_chain_is_recorded(self):
        clients = fl.partition_clients(self.data.y_train, 3, strategy="iid", seed=self.config.seed)
        for index, target in enumerate(self.payload["targets"]):
            actual_order = np.random.default_rng(self.config.seed + 1009 + index).permutation(6)
            self.assertEqual([b["samples"] for b in target["batches"]], [5, 1])
            self.assertEqual(target["batches"][0]["pre_parameters"], target["model_input"])
            self.assertEqual(target["batches"][0]["post_parameters"], target["batches"][1]["pre_parameters"])
            for batch_index, batch in enumerate(target["batches"]):
                local = actual_order[batch_index * 5:(batch_index + 1) * 5]
                global_indices = clients[index][local]
                self.assertEqual(batch["local_indices"], runtime.tensor_fingerprint(local))
                self.assertEqual(batch["dataset_indices"], runtime.tensor_fingerprint(global_indices))
                self.assertEqual(batch["features"], runtime.tensor_fingerprint(self.data.x_train[global_indices]))
                self.assertEqual(batch["labels"], runtime.tensor_fingerprint(self.data.y_train[global_indices]))
                self.assertEqual(batch["logits"]["shape"], [len(local), 10])
                self.assertEqual(batch["gradients"]["shape"], target["model_input"]["shape"])
                self.assertTrue(np.isfinite(batch["loss"]["value"]))
            layout = target["parameter_layout"]
            self.assertEqual(layout[0]["offset"], 0)
            self.assertEqual(sum(part["size"] for part in layout), target["model_input"]["shape"][0])
            for a, b in zip(layout, layout[1:]):
                self.assertEqual(a["offset"] + a["size"], b["offset"])

    def test_tail_arrays_are_exact_fingerprinted_copies_with_no_secret_state(self):
        self.assertEqual(len(self.snapshots), 20)
        self.assertEqual(set(self.snapshots), set(self.payload["snapshots"]))
        for target in self.payload["targets"]:
            tail = target["batches"][-1]
            for stage, key in target["tail_snapshots"].items():
                array = self.snapshots[key]
                self.assertEqual(runtime.tensor_fingerprint(array), self.payload["snapshots"][key])
                expected = target[stage] if stage == "final_delta" else tail[stage]
                if stage == "loss":
                    expected = expected["tensor"]
                self.assertEqual(runtime.tensor_fingerprint(array), expected)
            delta = self.snapshots[target["tail_snapshots"]["final_delta"]]
            post = self.snapshots[target["tail_snapshots"]["post_parameters"]]
            np.testing.assert_array_equal(delta, post - self.plain_states[0]["params"])
        text = json.dumps(self.payload, allow_nan=False)
        for forbidden in ("crypto_state", "task_tag", "private_key", "pending_evidence"):
            self.assertNotIn('"' + forbidden + '"', text)

    def test_original_cifar_cnn_ten_parameter_path_is_unchanged_on_small_cpu_data(self):
        spec = ModelSpec(input_shape=(1, 8, 8), num_classes=10, architecture="cifar10",
                         cifar_conv_filters=(2, 3), cifar_hidden_dims=(7, 5))
        plain = []
        original_candidate = fl._ClientUpdateCandidate
        def collect(*args, **kwargs):
            value = original_candidate(*args, **kwargs)
            plain.append(value.cpu_delta.copy())
            return value
        with mock.patch.object(fl, "model_spec_for_dataset", return_value=spec):
            with mock.patch.object(fl, "_ClientUpdateCandidate", side_effect=collect):
                fl.run_experiment(self.data, self.config)
            with runtime.observe(self.task) as observer:
                fl.run_experiment(self.data, self.config, checkpoint_callback=observer.checkpoint)
        payload = observer.finish()
        for index, target in enumerate(payload["targets"]):
            self.assertEqual(len(target["parameter_layout"]), 10)
            self.assertEqual(target["parameter_layout"][-1]["name"], "logits_b")
            np.testing.assert_array_equal(observer.snapshots[target["tail_snapshots"]["final_delta"]], plain[index])

    def test_batch_indices_are_observed_actual_results_not_reconstructed(self):
        original = torch.Tensor.index_select
        def reordered(tensor, dim, index, *args, **kwargs):
            value = original(tensor, dim, index, *args, **kwargs)
            # Change only original client-index gather result, before observation.
            if tensor.dtype == torch.int64 and tensor.ndim == 1 and tensor.numel() == 6:
                return value.flip(0)
            return value
        with mock.patch.object(torch.Tensor, "index_select", new=reordered):
            with runtime.observe(self.task) as observer:
                fl.run_experiment(self.data, self.config, checkpoint_callback=observer.checkpoint)
        payload = observer.finish()
        indices = fl.partition_clients(self.data.y_train, 3, strategy="iid", seed=self.config.seed)[0]
        order = np.random.default_rng(self.config.seed + 1009).permutation(6)
        actual = indices[order[:5]][::-1]
        batch = payload["targets"][0]["batches"][0]
        self.assertEqual(batch["dataset_indices"], runtime.tensor_fingerprint(actual))
        self.assertNotEqual(batch["dataset_indices"], runtime.tensor_fingerprint(indices[order[:5]]))
        self.assertEqual(batch["features"], runtime.tensor_fingerprint(self.data.x_train[actual]))

    def test_all_hooks_numerical_flags_and_rng_restore_on_unrelated_exception(self):
        before, rng = hooks(), rng_state()
        flags = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
                 torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32,
                 torch.backends.cudnn.allow_tf32)
        with self.assertRaisesRegex(RuntimeError, "unrelated failure"):
            with runtime.observe(self.task):
                with self.assertRaisesRegex(ValueError, "nested"):
                    with runtime.observe(self.task):
                        pass
                raise RuntimeError("unrelated failure")
        for obj, name, original in before:
            self.assertIs(getattr(obj, name), original)
        self.assertFalse(runtime._active)
        self.assertFalse(prefix._active)
        self.assertEqual(rng, rng_state())
        self.assertEqual(flags, (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))

    def test_original_backward_failure_escapes_and_restores_every_hook(self):
        before = hooks()
        with mock.patch.object(torch.Tensor, "backward", side_effect=RuntimeError("backward failed")):
            with self.assertRaisesRegex(RuntimeError, "backward failed"):
                with runtime.observe(self.task) as observer:
                    fl.run_experiment(self.data, self.config, checkpoint_callback=observer.checkpoint)
        with self.assertRaisesRegex(ValueError, "not reached"):
            observer.finish()
        for obj, name, original in before:
            self.assertIs(getattr(obj, name), original)

    def test_missing_stop_invalid_identity_resume_and_bad_targets_are_rejected(self):
        with runtime.observe(self.task) as observer:
            with self.assertRaisesRegex(ValueError, "identity"):
                fl.run_experiment(self.data, replace(self.config, seed=1))
            with self.assertRaisesRegex(ValueError, "fresh"):
                fl.run_experiment(self.data, self.config, resume_state={"completed_round": 0})
        with self.assertRaisesRegex(ValueError, "not reached"):
            observer.finish()
        changed = deepcopy(self.task)
        changed["stop_after_client"] = 2
        with self.assertRaisesRegex(ValueError, "next client"):
            with runtime.observe(changed):
                pass


if __name__ == "__main__":
    unittest.main()
