"""Real small CPU runs verify numerical equivalence and history-boundary evidence."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import unittest
from unittest import mock

import numpy as np
import torch

import cifar_mechanism_observer as observer_module
import cifar_mechanism_runtime as runtime
from cifar_cnn_history_runtime import history_runtime
from sm9rrsfl import fl
from sm9rrsfl import torch_backend as backend
from sm9rrsfl.model import ModelSpec
from sm9rrsfl.svd_detector import LongitudinalSVDDetector
from tests.test_cifar_prefix_probe_runtime import synthetic_dataset, rng_state, hooks


def task_for(arm="H0", **updates):
    config = fl.ExperimentConfig(method="sm9rrs", malicious_ratio=0., num_clients=3,
        rounds=30, partition="dirichlet", seed=2201, compute_backend="torch", device="cpu",
        crypto_mode="simulated", detector_window=20, attack_start_round=25, attack_target_count=2,
        checkpoint_interval=1, early_stop=False, batch_size=4, local_epochs=1, sm9_workers=1)
    config = replace(config, **updates)
    return {"task_id": "cpu-mechanism-" + arm, "fingerprint": "cpu-mechanism-test",
        "arm": arm, "repeat": 1, "config": asdict(config),
        "candidate": {"variant": "original" if arm == "H0" else "Ours-FrozenHistory-v1"},
        "history_freeze_start_round": None if arm == "H0" else 25,
        "policy": "singleton_backward_cudnn_deterministic"}


def execute(task, *, observed=True):
    config = fl.ExperimentConfig(**task["config"])
    states = []
    with history_runtime(task["candidate"]["variant"], freeze_start_round=task["history_freeze_start_round"]):
        if observed:
            with runtime.observe(task) as observer:
                def checkpoint(state):
                    states.append(np.array(state["params"], copy=True))
                    observer.checkpoint(state)
                result = fl.run_experiment(synthetic_dataset(), config, checkpoint_callback=checkpoint)
        else:
            result = fl.run_experiment(synthetic_dataset(), config,
                checkpoint_callback=lambda state: states.append(np.array(state["params"], copy=True)))
    return result, observer.finish(result) if observed else None, states


class MechanismRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = {arm: task_for(arm) for arm in ("H0", "H1")}
        cls.results, cls.payloads = {}, {}
        cls.before_rng = rng_state()
        cls.baseline = runtime._profile(torch)
        cls.states, cls.plain = {}, {}
        for arm, task in cls.tasks.items():
            cls.plain[arm] = execute(task, observed=False)
            cls.results[arm], cls.payloads[arm], cls.states[arm] = execute(task)
        cls.after_rng = rng_state()

    def test_actual_30_round_h0_and_h1_match_unobserved_cpu_scientific_results_and_rng(self):
        self.assertEqual(self.before_rng, self.after_rng)
        for arm in self.tasks:
            plain, _, states = self.plain[arm]
            self.assertEqual(plain.records, self.results[arm].records)
            self.assertEqual(plain.diagnostics, self.results[arm].diagnostics)
            self.assertEqual(len(states), len(self.states[arm]))
            for a, b in zip(states, self.states[arm]):
                np.testing.assert_array_equal(a, b)
        self.assertEqual(runtime._profile(torch), self.baseline)

    def test_exact_real_singleton_scope_and_all_30_round_client_coverage(self):
        for arm, payload in self.payloads.items():
            runtime.validate_policy_observation(payload, self.tasks[arm])
            self.assertEqual(len(payload["actual_batches"]), 90)
            self.assertEqual(len(payload["singleton_batches"]), 60)
            self.assertEqual(payload["numerical_policy"]["scoped_changes"], 60)
            self.assertEqual([(e["round"], e["client_id"]) for e in payload["numerical_policy"]["events"]],
                [(rd, cid) for rd in range(1, 31) for cid in ("client-0", "client-2")])
            for event in payload["numerical_policy"]["events"]:
                self.assertEqual(event["before"], self.baseline)
                self.assertEqual(event["restored"], self.baseline)
                self.assertEqual(event["effective"], {**self.baseline, "cudnn_deterministic": True})
            self.assertEqual(len(payload["evaluations"]), 62)
            self.assertTrue(all(row["aggregation_executed"] for row in payload["rounds"]))

    def test_cifar_ten_parameter_layout_preserves_original_two_convolution_branch(self):
        spec = ModelSpec(input_shape=(1, 8, 8), num_classes=10, architecture="cifar10",
            cifar_conv_filters=(2, 3), cifar_hidden_dims=(7, 5))
        with mock.patch.object(fl, "model_spec_for_dataset", return_value=spec):
            plain, _, states = execute(self.tasks["H0"], observed=False)
            result, payload, observed_states = execute(self.tasks["H0"])
        self.assertEqual(plain.records, result.records)
        for a, b in zip(states, observed_states):
            np.testing.assert_array_equal(a, b)
        self.assertTrue(all(len(row["parameter_layout"]) == 10 for row in payload["singleton_batches"]))
        runtime.validate_policy_observation(payload, self.tasks["H0"])

    def test_history_freeze_retains_end24_arrays_but_finalizes_every_commit(self):
        payload = self.payloads["H1"]
        observer_module.validate_history_observation(payload, self.tasks["H1"])
        end24 = {e["client_id"]: e["commit"]["after"] for e in payload["history_events"] if e["round"] == 24}
        admitted0 = 0
        for event in payload["history_events"]:
            if event["round"] < 25:
                continue
            commit, anchor = event["commit"], end24[event["client_id"]]
            self.assertFalse(commit["admitted"])
            self.assertIsNone(commit["after"]["pending"])
            self.assertEqual(commit["after"]["last_round"], event["round"])
            for field in ("history", "normal", "anchor", "norm_limit"):
                self.assertEqual(commit["after"][field], anchor[field])
        admitted0 = sum(e["commit"]["admitted"] for e in self.payloads["H0"]["history_events"] if e["round"] >= 25)
        self.assertGreater(admitted0, 0, "fixture must exercise the mechanism contrast")
        for a, b in zip(self.payloads["H0"]["rounds"][:24], payload["rounds"][:24]):
            self.assertEqual(a, b)
        for field in ("aggregate", "post_model", "coefficients"):
            self.assertEqual(self.payloads["H0"]["rounds"][24][field], payload["rounds"][24][field])

    def test_public_json_excludes_tags_crypto_arrays_and_duplicate_final_callback(self):
        payload = self.payloads["H1"]
        encoded = json.dumps(payload, allow_nan=False)
        for forbidden in ("task_tag", "crypto_state", "private_key", "pending_evidence"):
            self.assertNotIn('"' + forbidden + '"', encoded)
        self.assertEqual(payload["checkpoints"][-1]["callback_count"], 2)
        self.assertEqual(payload["terminal"], {"stopped_round": 30, "requested_rounds": 30,
            "stop_reason": None, "nonfinite_updates": 0, "blacklisted_clients": [], "malicious_clients": []})

    @staticmethod
    def force_revocations(all_clients):
        original = LongitudinalSVDDetector.evaluate
        first_tag = []
        def evaluate(detector, tag, update, *, round_id=None, learning_rate=1.0):
            decision = original(detector, tag, update, round_id=round_id, learning_rate=learning_rate)
            if round_id == 22 and (all_clients or not first_tag):
                first_tag.append(tag)
                decision = replace(decision, accepted=False, reason="strong_novelty", would_flag=True,
                    count_increment=True, history_eligible=False, immediate_revocation=True)
                state = detector._states[tag]
                state.pending = (*state.pending[:3], decision)
            return decision
        return mock.patch.object(LongitudinalSVDDetector, "evaluate", new=evaluate)

    def test_certificate_revocation_removes_only_that_client_from_later_training_and_history(self):
        with self.force_revocations(False):
            result, payload, _ = execute(task_for())
        self.assertEqual(result.blacklisted_clients, ("client-0",))
        self.assertEqual(len(payload["rounds"][21]["clients"]), 3)
        self.assertEqual([c["client_id"] for c in payload["rounds"][22]["clients"]], ["client-1", "client-2"])
        forgotten = [e for e in payload["history_events"] if e["forget"] is not None]
        self.assertEqual([(e["round"], e["client_id"]) for e in forgotten], [(22, "client-0")])
        self.assertIsNone(forgotten[0]["forget"]["after"])
        runtime.validate_policy_observation(payload, task_for())

    def test_all_honest_revoked_is_retained_as_early_complete_execution(self):
        with self.force_revocations(True):
            result, payload, _ = execute(task_for())
        self.assertEqual(result.stopped_round, 22)
        self.assertEqual(payload["terminal"]["stop_reason"], "all_honest_revoked")
        self.assertEqual(len(payload["rounds"]), 22)
        self.assertEqual(len(payload["terminal"]["blacklisted_clients"]), 3)
        runtime.validate_policy_observation(payload, task_for())

    def test_nonfinite_updates_keep_actual_denominators_without_invented_aggregation(self):
        original = backend.TorchTrainingContext.local_train_delta_resident
        def nonfinite(context, *args, **kwargs):
            delta, stats = original(context, *args, **kwargs)
            delta[0] = float("nan")
            return delta, stats
        with mock.patch.object(backend.TorchTrainingContext, "local_train_delta_resident", new=nonfinite):
            result, payload, _ = execute(task_for())
        self.assertEqual(result.nonfinite_updates, 90)
        self.assertEqual(payload["terminal"]["nonfinite_updates"], 90)
        self.assertEqual(payload["history_events"], [])
        self.assertEqual(len(payload["actual_batches"]), 90)
        for row in payload["rounds"]:
            self.assertFalse(row["aggregation_executed"])
            self.assertIsNone(row["aggregate"])
            self.assertEqual(row["post_model"], payload["initial_model"])
            self.assertEqual(row["diagnostics"], [])
            self.assertTrue(all(c["update_finite"] is False and "raw_update" in c and "update" not in c
                                for c in row["clients"]))
        runtime.validate_policy_observation(payload, task_for())

    def test_policy_rejects_missing_events_bad_scope_shapes_flags_and_dtype(self):
        mutations = [lambda p: p["actual_batches"].pop(), lambda p: p["singleton_batches"].pop(),
            lambda p: p["numerical_policy"]["events"][0]["effective"].__setitem__("cudnn_allow_tf32", False),
            lambda p: p["singleton_batches"][0]["gradients"].__setitem__("dtype", "<f8"),
            lambda p: p["singleton_batches"][0]["logits"].__setitem__("shape", [1, 9]),
            lambda p: p["numerical_policy"]["checks"].__setitem__("post_flat", 91)]
        for mutate in mutations:
            payload = deepcopy(self.payloads["H1"])
            mutate(payload)
            with self.assertRaises(ValueError):
                runtime.validate_policy_observation(payload, self.tasks["H1"])

    def test_history_validator_rejects_fabricated_freeze_and_state_changes(self):
        mutations = [lambda p: p["history_events"].pop(),
            lambda p: p["history_events"][75]["commit"].__setitem__("admitted", True),
            lambda p: p["history_events"][75]["commit"]["after"].__setitem__("drift", 123.),
            lambda p: p["history_events"][75]["commit"]["after"]["history"].__setitem__("sha256", "0" * 64),
            lambda p: p["history_events"][75]["before_evaluate"].__setitem__("clean_streak", 10000)]
        for mutate in mutations:
            payload = deepcopy(self.payloads["H1"])
            mutate(payload)
            with self.assertRaises(ValueError):
                observer_module.validate_history_observation(payload, self.tasks["H1"])

    def test_hooks_restore_on_singleton_backward_failure_and_cannot_finalize_failed_observation(self):
        before = hooks() + [(LongitudinalSVDDetector, k, getattr(LongitudinalSVDDetector, k))
                            for k in ("evaluate", "commit", "forget")]
        original = torch.Tensor.backward
        def backward(tensor, *args, **kwargs):
            if torch.backends.cudnn.deterministic:
                raise RuntimeError("injected singleton failure")
            return original(tensor, *args, **kwargs)
        with mock.patch.object(torch.Tensor, "backward", new=backward):
            with self.assertRaisesRegex(RuntimeError, "singleton failure"):
                with runtime.observe(self.tasks["H0"]) as observer:
                    fl.run_experiment(synthetic_dataset(), fl.ExperimentConfig(**self.tasks["H0"]["config"]),
                        checkpoint_callback=observer.checkpoint)
        with self.assertRaisesRegex(ValueError, "successfully"):
            observer.finish(self.results["H0"])
        for obj, name, saved in before:
            self.assertIs(getattr(obj, name), saved)
        self.assertEqual(runtime._profile(torch), self.baseline)
        self.assertFalse(runtime._active)
        self.assertFalse(observer_module._active)

    def test_nested_interrupt_wrong_arm_and_attack_restore(self):
        with self.assertRaises(KeyboardInterrupt):
            with runtime.observe(self.tasks["H0"]):
                with self.assertRaisesRegex(ValueError, "nested"):
                    with runtime.observe(self.tasks["H0"]):
                        pass
                raise KeyboardInterrupt
        for field, value in (("malicious_ratio", .1), ("method", "fedavg"), ("compute_backend", "numpy")):
            task = deepcopy(self.tasks["H0"])
            task["config"][field] = value
            with self.assertRaises(ValueError):
                with runtime.observe(task):
                    pass
        self.assertEqual(runtime._profile(torch), self.baseline)
        self.assertFalse(runtime._active)
        self.assertFalse(observer_module._active)

    def test_empty_clients_and_no_singleton_batches_are_observed_without_false_events(self):
        with mock.patch.object(fl, "partition_clients", return_value=[np.arange(15), np.arange(15, 30), np.array([], dtype=np.int64)]):
            _, payload, _ = execute(task_for(batch_size=5))
        self.assertEqual(payload["numerical_policy"]["scoped_changes"], 0)
        self.assertEqual(payload["singleton_batches"], [])
        self.assertEqual(payload["numerical_policy"]["checks"]["post_flat"], 60)
        self.assertEqual(len(payload["actual_batches"]), 90)

    def test_batch_size_one_targets_every_actual_backward_even_nonfinal_batches(self):
        task = task_for(batch_size=1)
        _, payload, _ = execute(task)
        self.assertEqual(len(payload["singleton_batches"]), 900)
        self.assertEqual(payload["numerical_policy"]["checks"]["non_target_backward_before"], 0)
        self.assertEqual(payload["numerical_policy"]["scoped_changes"], 900)
        runtime.validate_policy_observation(payload, task)


if __name__ == "__main__":
    unittest.main()
