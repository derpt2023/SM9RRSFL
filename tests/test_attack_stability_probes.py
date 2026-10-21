from dataclasses import replace
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from scripts.experiments.cifar10 import attack_stability_probes as probes
from sm9rrsfl.fl import ExperimentConfig, ExperimentResult, RoundRecord


class RepresentativeTests(unittest.TestCase):
    def test_extremes_quantiles_and_source_groups_are_deterministic(self):
        parts, labels = [], []
        for client in range(40):
            parts.append(np.arange(len(labels), len(labels) + client + 1))
            labels.extend([5 if client % 3 else 2] * (client + 1))
        candidates = list(range(5, 40))
        actual = probes.select_representatives(np.array(labels), parts, candidates)
        repeated = probes.select_representatives(np.array(labels), parts, reversed(candidates))
        self.assertEqual(actual, repeated)
        self.assertEqual(len(actual), 16)
        self.assertEqual(len(set(actual)), 16)
        self.assertTrue(set(actual) <= set(candidates))
        self.assertIn(5, actual)
        self.assertIn(39, actual)
        self.assertTrue(any(i % 3 == 0 for i in actual))
        self.assertTrue(any(i % 3 != 0 for i in actual))
        self.assertTrue(any(20 <= i <= 24 for i in actual))

    def test_count_capped_and_ties_stable(self):
        parts = [np.array([i]) for i in range(4)]
        self.assertEqual(probes.select_representatives(
            np.array([1, 5, 1, 5]), parts, [3, 1, 2]), [1, 2, 3])

    def test_empty_and_invalid_clients_rejected(self):
        for clients in ([], [-1], [2]):
            with self.assertRaises(ValueError):
                probes.select_representatives(np.array([5]), [np.array([0])], clients)


class CoverageTests(unittest.TestCase):
    def fixture(self):
        return {
            "status": "complete", "all_finite": True, "requested_clients": 2,
            "actual_results": [
                {"status": "complete", "all_finite": True},
                {"status": "complete", "all_finite": True},
            ],
        }

    def test_complete_finite_coverage_required(self):
        self.assertTrue(probes.probe_rankable(self.fixture()))
        for change in ({"status": "partial"}, {"status": "numerical_failure"},
                       {"all_finite": False}, {"requested_clients": 16}):
            value = self.fixture()
            value.update(change)
            self.assertFalse(probes.probe_rankable(value))

    def test_no_empty_or_failed_rows(self):
        value = self.fixture()
        value["actual_results"][0]["all_finite"] = False
        self.assertFalse(probes.probe_rankable(value))
        self.assertFalse(probes.probe_rankable(
            {"status": "complete", "all_finite": True, "requested_clients": 0,
             "actual_results": []}))

    def test_json_nonfinite_evidence_not_zero(self):
        self.assertEqual(probes._safe({"failure": [float("nan"), float("inf")]}),
                         {"failure": ["NaN", "Infinity"]})


class BenchmarkSignatureTests(unittest.TestCase):
    def fixture(self):
        record = RoundRecord(
            method="sm9rrs", malicious_ratio=0., round=1, accuracy=.4, error=.6,
            accepted_updates=4, rejected_updates=0, blacklisted_clients=0,
            true_positive_revocations=0, false_positive_revocations=0,
            krum_selected_client="",
        )
        return ExperimentResult(
            config=ExperimentConfig(), records=[record], final_accuracy=.4,
            final_error=.6, stopped_round=1, malicious_clients=(),
            blacklisted_clients=(), runtime_seconds=10.,
        )

    def test_timing_and_worker_count_ignored(self):
        original = self.fixture()
        faster = replace(original, runtime_seconds=1.,
                         config=replace(original.config, sm9_workers=8))
        self.assertEqual(probes.benchmark_signature(original, [(1, "same")]),
                         probes.benchmark_signature(faster, [(1, "same")]))

    def test_scientific_and_model_differences_retained(self):
        original = self.fixture()
        changed = replace(original, records=[
            replace(original.records[0], malicious_weight_mass=.1)])
        self.assertNotEqual(probes.benchmark_signature(original, [(1, "same")]),
                            probes.benchmark_signature(changed, [(1, "same")]))
        self.assertNotEqual(probes.benchmark_signature(original, [(1, "same")]),
                            probes.benchmark_signature(original, [(1, "different")]))


class RunProbeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        np.savez(self.path / "anchor.npz", params=np.ones(3, dtype=np.float32))
        self.request = {
            "task_id": "C1-iid-initial", "profile": {"id": "C1"},
            "fingerprint": "root-test-protocol-fingerprint",
            "config": {"num_clients": 2, "seed": 301, "lr": .01, "lr_decay": .99},
            "data_config": {}, "device": "cpu", "outdir": str(self.path),
            "anchor_path": str(self.path / "anchor.npz"), "clients": [0, 1],
            "deadline_epoch": time.time() + 100,
        }
        dataset = SimpleNamespace(y_train=np.array([5, 1]))
        finite = SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True))
        context = SimpleNamespace(spec=SimpleNamespace(parameter_size=3),
                                  torch=SimpleNamespace(isfinite=lambda _: finite))
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(probes, "_INITIALIZED", True))
        stack.enter_context(mock.patch.object(
            probes, "_context", return_value=(dataset, context,
                                             [np.array([0]), np.array([1])])))
        stack.enter_context(mock.patch.object(
            probes.fl, "_select_attack_target_indices", return_value=np.array([1])))

    def valid_attack(self):
        return np.zeros(3), SimpleNamespace(loss=1.), {
            "target_success": 1., "target_confidence": .9,
            "max_parameter_growth_ratio": 1e12, "max_weight_growth_ratio": 2.,
            "normal_steps": 3, "target_steps": 1, "target_delta_l2_total": .2,
        }

    def test_complete_metrics_and_exact_client_seeds(self):
        with mock.patch(
            "scripts.experiments.cifar10.attack_stability_attack.run_profile_attack",
            return_value=self.valid_attack(),
        ) as attack:
            result = probes.run_probe(self.request)
        self.assertTrue(result["rankable"])
        self.assertEqual(result["mean_target_success"], 1.)
        self.assertEqual(result["worst_growth_ratio"], 2.)
        self.assertEqual(result["worst_client_idx"], 0)
        self.assertEqual([call.kwargs["seed"] for call in attack.call_args_list],
                         [301 + 25 * 1009, 301 + 25 * 1009 + 1])
        self.assertEqual([call.kwargs["lr"] for call in attack.call_args_list],
                         [.01 * .99 ** 24] * 2)
        self.assertTrue((self.path / self.request["task_id"] / "client_001.json").exists())

    def test_first_failure_saved_and_remaining_clients_not_run(self):
        from scripts.experiments.cifar10.attack_stability_attack import NumericalFailure
        failure = NumericalFailure({
            "first_failure": {"stage": "stealth.gradients", "block": 6},
            "status": "nonfinite",
        })
        with mock.patch(
            "scripts.experiments.cifar10.attack_stability_attack.run_profile_attack",
            side_effect=failure,
        ) as attack:
            result = probes.run_probe(self.request)
        self.assertEqual(attack.call_count, 1)
        self.assertEqual(result["status"], "numerical_failure")
        self.assertFalse(result["rankable"])
        self.assertEqual(result["completed_clients"], 0)
        saved = json.loads((self.path / self.request["task_id"] / "client_000.json").read_text())
        self.assertEqual(saved["audit"]["first_failure"]["block"], 6)
        self.assertFalse((self.path / self.request["task_id"] / "client_001.json").exists())

    def test_budget_exhaustion_cannot_claim_coverage(self):
        self.request["deadline_epoch"] = time.time() - 1
        result = probes.run_probe(self.request)
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["rankable"])
        self.assertEqual(result["actual_results"], [])

    def run_valid(self):
        with mock.patch(
            "scripts.experiments.cifar10.attack_stability_attack.run_profile_attack",
            return_value=self.valid_attack(),
        ) as attack:
            result = probes.run_probe(self.request)
        return result, attack

    def test_terminal_reuse_does_not_overwrite_request_or_results(self):
        original, _ = self.run_valid()
        folder = self.path / self.request["task_id"]
        request_bytes = (folder / "request.json").read_bytes()
        result_bytes = (folder / "result.json").read_bytes()
        self.request["deadline_epoch"] += 100
        with mock.patch(
            "scripts.experiments.cifar10.attack_stability_attack.run_profile_attack",
            side_effect=AssertionError("terminal result must not rerun"),
        ):
            reused = probes.run_probe(self.request)
        self.assertEqual(original, reused)
        self.assertEqual(request_bytes, (folder / "request.json").read_bytes())
        self.assertEqual(result_bytes, (folder / "result.json").read_bytes())

    def test_changed_identity_fields_rejected(self):
        self.run_valid()
        changes = (
            ("fingerprint", "new-root"), ("profile", {"id": "C2"}),
            ("config", {**self.request["config"], "seed": 302}),
            ("clients", [1, 0]), ("data_config", {"seed": 43}),
            ("device", "mps"),
        )
        for field, value in changes:
            with self.subTest(field=field):
                request = deepcopy(self.request)
                request[field] = value
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    probes.run_probe(request)
        np.savez(self.path / "anchor.npz", params=np.ones(3, dtype=np.float32) * 2)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            probes.run_probe(self.request)

    def test_partial_retries_only_incomplete_client_and_archives_attempt(self):
        with mock.patch(
            "scripts.experiments.cifar10.attack_stability_attack.run_profile_attack",
            side_effect=[self.valid_attack(), TimeoutError("budget reached")],
        ):
            partial = probes.run_probe(self.request)
        self.assertEqual(partial["status"], "partial")
        folder = self.path / self.request["task_id"]
        completed_bytes = (folder / "client_000.json").read_bytes()
        result, attack = self.run_valid()
        self.assertEqual(attack.call_count, 1)
        self.assertEqual(attack.call_args.kwargs["client_idx"], 1)
        self.assertTrue(result["rankable"])
        self.assertEqual(completed_bytes, (folder / "client_000.json").read_bytes())
        self.assertEqual(len(list((folder / "partial_attempts").glob("*.json"))), 1)

    def test_runtime_failure_durable_even_before_result_commit(self):
        with mock.patch(
            "scripts.experiments.cifar10.attack_stability_attack.run_profile_attack",
            side_effect=RuntimeError("device failed"),
        ):
            result = probes.run_probe(self.request)
        self.assertEqual(result["status"], "failed")
        self.assertIn("RuntimeError: device failed",
                      result["actual_results"][0]["traceback"])
        folder = self.path / self.request["task_id"]
        evidence = (folder / "client_000.json").read_bytes()
        (folder / "result.json").unlink()
        with mock.patch(
            "scripts.experiments.cifar10.attack_stability_attack.run_profile_attack",
            side_effect=AssertionError("saved failure must not rerun"),
        ):
            recovered = probes.run_probe(self.request)
        self.assertEqual(recovered["status"], "failed")
        self.assertEqual(evidence, (folder / "client_000.json").read_bytes())

    def test_keyboard_interrupt_propagates(self):
        with mock.patch(
            "scripts.experiments.cifar10.attack_stability_attack.run_profile_attack",
            side_effect=KeyboardInterrupt,
        ):
            with self.assertRaises(KeyboardInterrupt):
                probes.run_probe(self.request)

    def test_setup_runtime_failure_is_durable(self):
        with mock.patch.object(probes, "_context", side_effect=RuntimeError("MPS unavailable")):
            result = probes.run_probe(self.request)
        self.assertEqual(result["status"], "failed")
        self.assertIn("MPS unavailable", result["failure"]["traceback"])
        folder = self.path / self.request["task_id"]
        evidence = (folder / "setup_failure.json").read_bytes()
        (folder / "result.json").unlink()
        with mock.patch.object(probes, "_context", side_effect=AssertionError("no retry")):
            recovered = probes.run_probe(self.request)
        self.assertEqual(recovered["status"], "failed")
        self.assertEqual(evidence, (folder / "setup_failure.json").read_bytes())

    def test_root_fingerprint_required(self):
        self.request.pop("fingerprint")
        with self.assertRaisesRegex(ValueError, "root protocol fingerprint"):
            probes.run_probe(self.request)

    def test_same_task_lock_rejects_parallel_execution(self):
        import fcntl
        folder = self.path / self.request["task_id"]
        folder.mkdir()
        with (folder / "probe.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                probes.run_probe(self.request)


if __name__ == "__main__":
    unittest.main()
