"""Small synthetic CPU runs prove the observer preserves original computation."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import pickle
import random
import unittest
from unittest import mock

import numpy as np
import torch

import cifar_prefix_probe_runtime as runtime
from sm9rrsfl import fl
from sm9rrsfl import torch_backend as backend
from sm9rrsfl.datasets import ImageDataset


def synthetic_dataset():
    rng = np.random.default_rng(912)
    return ImageDataset(rng.normal(0, .1, (30, 1, 8, 8)).astype(np.float32),
        np.arange(30, dtype=np.int64) % 10,
        rng.normal(0, .1, (20, 1, 8, 8)).astype(np.float32), np.arange(20, dtype=np.int64) % 10,
        name="prefix-synthetic", num_classes=10)


def rng_state():
    return (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()),
            torch.get_rng_state().numpy().tobytes())


def hooks():
    return [(obj, name, getattr(obj, name)) for obj, name in (
        (fl, "run_experiment"), (fl, "partition_clients"), (fl, "init_params"),
        (fl, "_local_train_client_delta"), (fl, "_process_sm9_candidates"),
        (fl, "bounded_aggregation_coefficients"), (fl, "aggregate_with_coefficients"),
        (backend.TorchTrainingContext, "add_update"), (fl, "_evaluate_accuracy"),
        (fl, "_evaluate_attack_target_metrics"), (backend, "_torch_forward"))]


class PrefixProbeRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = synthetic_dataset()
        cls.config = fl.ExperimentConfig(method="sm9rrs", malicious_ratio=0., num_clients=3,
            rounds=3, partition="dirichlet", seed=2201, compute_backend="torch", device="cpu",
            crypto_mode="simulated", detector_window=20, attack_start_round=25, attack_target_count=2,
            checkpoint_interval=1, early_stop=False, batch_size=7, local_epochs=1, sm9_workers=1)
        cls.task = {"task_id": "prefix-dirichlet-repeat0", "fingerprint": "synthetic-probe-id",
                    "config": asdict(cls.config)}
        plain_states, observed_states = [], []
        cls.rng_before_plain = rng_state()
        cls.plain = fl.run_experiment(cls.dataset, cls.config, checkpoint_callback=plain_states.append)
        cls.rng_after_plain = rng_state()
        cls.forward_calls = []
        original_forward = backend._torch_forward
        def counted(*args, **kwargs):
            cls.forward_calls.append(1)
            return original_forward(*args, **kwargs)
        cls.rng_before_observed = rng_state()
        with mock.patch.object(backend, "_torch_forward", side_effect=counted):
            with runtime.observe(cls.task) as observer:
                def checkpoint(state):
                    observed_states.append(state)
                    observer.checkpoint(state)
                cls.observed = fl.run_experiment(cls.dataset, cls.config, checkpoint_callback=checkpoint)
                cls.payload = observer.finish(cls.observed)
        cls.rng_after_observed = rng_state()
        cls.plain_states, cls.observed_states = plain_states, observed_states

    def test_true_cpu_observation_preserves_models_records_diagnostics_and_rng(self):
        self.assertEqual(self.rng_before_plain, self.rng_after_plain)
        self.assertEqual(self.rng_before_observed, self.rng_after_observed)
        self.assertEqual(self.plain.records, self.observed.records)
        self.assertEqual(self.plain.diagnostics, self.observed.diagnostics)
        self.assertEqual(len(self.plain_states), len(self.observed_states))
        for left, right in zip(self.plain_states, self.observed_states):
            self.assertEqual(left["completed_round"], right["completed_round"])
            np.testing.assert_array_equal(left["params"], right["params"])

    def test_fingerprints_cover_actual_data_initialization_every_client_and_aggregation(self):
        payload = self.payload
        self.assertEqual(payload["schema"], runtime.SCHEMA)
        self.assertEqual(payload["data"]["x_train"], runtime.tensor_fingerprint(self.dataset.x_train))
        self.assertIsNone(payload["data"]["x_attack"])
        self.assertEqual(payload["initial_model"], runtime.tensor_fingerprint(self.plain_states[0]["params"]))
        self.assertEqual([r["round"] for r in payload["rounds"]], [1, 2, 3])
        expected_ids = ["client-" + str(i) for i in range(3)]
        for row in payload["rounds"]:
            rd = row["round"]
            for key in ("candidate_order", "verified_order", "aggregate_order"):
                self.assertEqual(row[key], expected_ids)
            self.assertEqual(row["coefficients"]["order"], expected_ids)
            self.assertEqual(set(row["coefficients"]["by_client"]), set(expected_ids))
            self.assertEqual(row["post_model"], payload["checkpoints"][rd]["model"])
            self.assertEqual(row["record"], asdict(self.observed.records[rd]))
            for client in row["clients"]:
                self.assertEqual(client["model_input"], payload["checkpoints"][rd - 1]["model"])
                self.assertEqual(client["update"]["dtype"], "<f4")
                self.assertEqual(client["samples"], client["stats"]["samples"])
                self.assertEqual(sum(client["minibatch_sizes_per_epoch"]), client["samples"])

    def test_reconstructed_order_matches_original_local_seed_without_global_rng_use(self):
        indices = fl.partition_clients(self.dataset.y_train, self.config.num_clients,
            strategy=self.config.partition, dirichlet_alpha=self.config.dirichlet_alpha, seed=self.config.seed)
        for row in self.payload["rounds"]:
            for index, client in enumerate(row["clients"]):
                seed = self.config.seed + row["round"] * 1009 + index
                expected = indices[index][np.random.default_rng(seed).permutation(len(indices[index]))]
                self.assertEqual(client["training_seed"], seed)
                self.assertEqual(client["epoch_indices"], [runtime.tensor_fingerprint(expected)])

    def test_actual_evaluation_logits_are_captured_without_repeated_forward(self):
        evaluations = self.payload["evaluations"]
        self.assertEqual([(e["round"], e["kind"]) for e in evaluations],
                         [(rd, kind) for rd in range(4) for kind in ("accuracy", "target")])
        training_calls = sum(len(c["minibatch_sizes_per_epoch"]) * c["epochs"]
            for row in self.payload["rounds"] for c in row["clients"])
        self.assertEqual(len(self.forward_calls), training_calls + 8)
        for evaluation in evaluations:
            self.assertEqual(len(evaluation["batches"]), 1)
            batch = evaluation["batches"][0]
            self.assertEqual(batch["logits"]["shape"], [20 if evaluation["kind"] == "accuracy" else 2, 10])
            self.assertEqual(batch["predictions"]["shape"], [batch["logits"]["shape"][0]])
            self.assertEqual(batch["predictions"]["dtype"], "<i8")
            record = self.observed.records[evaluation["round"]]
            expected = record.accuracy if evaluation["kind"] == "accuracy" else [record.attack_target_success_rate, record.attack_target_confidence]
            self.assertEqual(evaluation["value"], expected)

    def test_checkpoint_deduplicates_final_crypto_callback_without_leaking_crypto_or_vectors(self):
        self.assertEqual([c["round"] for c in self.payload["checkpoints"]], [0, 1, 2, 3])
        self.assertEqual(self.payload["checkpoints"][-1]["callback_count"], 2)
        encoded = json.dumps(self.payload, allow_nan=False)
        for forbidden in ("crypto_state", "task_tag", "private_key", "pending_evidence"):
            self.assertNotIn('"' + forbidden + '"', encoded)
        self.assertTrue(all("sha256" in r["post_model"] for r in self.payload["rounds"]))

    def test_all_hooks_and_numerical_flags_restore_on_exception_and_nested_context_refused(self):
        before = hooks()
        flags = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
                 torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32)
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with runtime.observe(self.task):
                with self.assertRaisesRegex(RuntimeError, "nested"):
                    with runtime.observe(self.task):
                        pass
                raise RuntimeError("sentinel")
        self.assertFalse(runtime._active)
        for obj, name, original in before:
            self.assertIs(getattr(obj, name), original)
        self.assertEqual(flags, (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
                 torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32))

    def test_changed_config_resume_and_incomplete_observation_are_rejected(self):
        with runtime.observe(self.task) as observer:
            with self.assertRaisesRegex(ValueError, "fresh"):
                fl.run_experiment(self.dataset, self.config, resume_state={"completed_round": 1})
            with self.assertRaisesRegex(ValueError, "identity"):
                fl.run_experiment(self.dataset, replace(self.config, seed=1))
            with self.assertRaisesRegex(ValueError, "fresh complete"):
                observer.finish(self.observed)

    def test_fingerprint_preserves_one_bit_difference_shape_dtype_and_input_storage(self):
        value = np.array([0., 1.], dtype=np.float32)
        before = value.copy()
        baseline = runtime.tensor_fingerprint(value)
        tiny = value.copy()
        tiny[1] = np.nextafter(tiny[1], np.float32(2.))
        self.assertNotEqual(baseline, runtime.tensor_fingerprint(tiny))
        self.assertNotEqual(baseline, runtime.tensor_fingerprint(value.reshape(1, 2)))
        self.assertNotEqual(baseline, runtime.tensor_fingerprint(value.astype(np.float64)))
        self.assertEqual(baseline, runtime.tensor_fingerprint(torch.from_numpy(value)))
        np.testing.assert_array_equal(value, before)


if __name__ == "__main__":
    unittest.main()
