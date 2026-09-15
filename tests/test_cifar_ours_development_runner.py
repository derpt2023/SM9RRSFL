"""Small CPU fixtures for development matrix, resume identity, and reporting."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from sm9rrsfl import fl, experiments
import run_cifar_ours_development as runner


def sources():
    return {(partition, ratio): fl.ExperimentConfig(
        method="sm9rrs", partition=partition, malicious_ratio=ratio, seed=401,
        rounds=100, early_stop=False, compute_backend="auto", device="cuda:3",
        detector_distance_threshold=1.75, suspicion_remove_after=3,
        detector_window=20, attack_start_round=25, num_clients=100)
        for partition, ratio in runner.SCENARIOS}


def make_result(config, *, accuracy=.6, asr=.05, false_revoked=0, nonfinite=0):
    records = [fl.RoundRecord(
        method="sm9rrs", malicious_ratio=config.malicious_ratio, round=rd,
        accuracy=accuracy, error=1-accuracy, accepted_updates=100 if rd else 0,
        rejected_updates=0, blacklisted_clients=false_revoked if rd >= 97 else 0,
        true_positive_revocations=0, false_positive_revocations=false_revoked if rd >= 97 else 0,
        krum_selected_client="", attack_target_success_rate=asr,
        attack_active=bool(config.malicious_ratio and rd >= 25),
        nonfinite_updates=nonfinite if rd == 25 else 0)
        for rd in range(101)]
    return fl.ExperimentResult(config=config, records=records, final_accuracy=accuracy,
        final_error=1-accuracy, stopped_round=100, malicious_clients=tuple(
            f"client-{i}" for i in range(round(100*config.malicious_ratio))),
        blacklisted_clients=(), nonfinite_updates=nonfinite)


class DevelopmentRunnerTest(unittest.TestCase):
    def test_matrix_preserves_training_and_uses_fresh_seed(self):
        recorded = sources()
        tasks = runner.build_tasks(recorded)
        self.assertEqual(len(tasks), 8)
        self.assertEqual(len({task["task_id"] for task in tasks}), 8)
        for task in tasks:
            config = task["config"]
            original = asdict(recorded[(config["partition"], config["malicious_ratio"])])
            differences = {key for key in config if config[key] != original[key]}
            self.assertEqual(differences, {"device", "seed", "checkpoint_interval"})
            self.assertEqual(config["seed"], 601)
            self.assertEqual(config["rounds"], 100)
            self.assertEqual(config["detector_distance_threshold"], 1.75)
        for reserved_seed in (401, 402, 403, 501, 502, 503):
            with self.assertRaises(ValueError):
                runner.build_tasks(recorded, seed=reserved_seed)

    def test_output_guard_and_manifest_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            for output in (source, source / "new", root):
                with self.assertRaises(ValueError):
                    runner.validate_paths(source, output)
            output = root / "separate"
            output.mkdir()
            runner.ensure_manifest(output, {"fingerprint": "first"})
            runner.ensure_manifest(output, {"fingerprint": "first"})
            with self.assertRaises(ValueError):
                runner.ensure_manifest(output, {"fingerprint": "changed"})

    def test_completed_result_allows_only_runtime_remapping(self):
        task = runner.build_tasks(sources())[0]
        config = replace(fl.ExperimentConfig(**task["config"]), device="cuda:5", sm9_workers=4)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = make_result(config)
            runner.ensure_task_identity(output, task)
            experiments.write_result_files(output, [result])
            self.assertEqual(runner.checked_completed(output, task).config.device, "cuda:5")
            task["config"]["lr"] *= 2
            with self.assertRaises(ValueError):
                runner.checked_completed(output, task)
            (output / experiments.COMPLETED_RESULTS_SNAPSHOT).write_bytes(b"broken")
            with self.assertRaises(ValueError):
                runner.checked_completed(output, task)

    def test_late_clean_revocation_and_nonfinite_fail_health(self):
        config = fl.ExperimentConfig(**runner.build_tasks(sources())[0]["config"])
        self.assertTrue(runner.result_metrics(make_result(config))["health_pass"])
        metrics = runner.result_metrics(make_result(config, false_revoked=11))
        self.assertIn("clean_false_revocation_rate", metrics["health_reasons"])
        self.assertAlmostEqual(metrics["max_false_revocation_rate"], .11)
        self.assertIn("nonfinite_updates", runner.result_metrics(make_result(config, nonfinite=1))["health_reasons"])
        early = replace(make_result(config), stopped_round=25)
        self.assertFalse(runner.result_metrics(early)["completed"])

    def test_identical_configs_do_not_allow_cross_variant_result_reuse(self):
        control, candidate = runner.build_tasks(sources())[:2]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            runner.ensure_task_identity(output, control)
            experiments.write_result_files(output, [make_result(fl.ExperimentConfig(**control["config"]))])
            with self.assertRaises(ValueError):
                runner.checked_completed(output, candidate)
            with self.assertRaises(ValueError):
                runner.ensure_task_identity(output, candidate)
            (output / "task.json").unlink()
            with self.assertRaises(ValueError):
                runner.checked_completed(output, control)
            with self.assertRaises(ValueError):
                runner.ensure_task_identity(output, control)

    def test_summary_uses_all_attack_rounds_and_paired_clean_reference(self):
        tasks = runner.build_tasks(sources())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for task in tasks:
                config = fl.ExperimentConfig(**task["config"])
                result = make_result(config, accuracy=.56 if task["variant"] == "weak_quarantine" and config.malicious_ratio == 0 else .6)
                if task["variant"] == "weak_quarantine" and config.malicious_ratio == .7:
                    result.records[25] = replace(result.records[25], attack_target_success_rate=.9)
                task_dir = output / "tasks" / task["task_id"]
                task_dir.mkdir(parents=True)
                runner.ensure_task_identity(task_dir, task)
                experiments.write_result_files(task_dir, [result])
            report = runner.summarize(output, {"tasks": tasks, "fingerprint": "test"})
            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["candidate_health_pass"])
            self.assertFalse(report["candidate_control_relative_clean_retention_pass"])
            self.assertFalse(report["candidate_attack_targets_met_in_development"])
            self.assertFalse(report["formal_feasibility_assessed"])
            attack = next(row["metrics"] for row in report["tasks"] if row["variant"] == "weak_quarantine" and row["malicious_ratio"] == .7)
            self.assertAlmostEqual(attack["attack_mean_asr"], (.9+75*.05)/76)
            self.assertEqual(attack["attack_peak_asr"], .9)
            self.assertAlmostEqual(attack["attack_tail10_asr"], .05)

    def test_strict_environment_rejects_tf32_drift_but_allows_gpu_remap(self):
        metadata = {"device": "cuda:0", "gpu": "same GPU", "cudnn_tf32": True,
                    "source_sha256": {}, "environment": {"CUDA_VISIBLE_DEVICES": "7", "OMP_NUM_THREADS": "1"}}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            runner.check_execution_environment(output, metadata)
            remapped = {**metadata, "device": "cuda:1", "environment": {**metadata["environment"], "CUDA_VISIBLE_DEVICES": "6,7"}}
            runner.check_execution_environment(output, remapped)
            with self.assertRaises(ValueError):
                runner.check_execution_environment(output, {**remapped, "cudnn_tf32": False})

    def test_progress_wrapper_preserves_checkpoint_callback_and_restores(self):
        callback = mock.Mock()
        fake = mock.Mock(side_effect=lambda *args, **kwargs: kwargs["checkpoint_callback"]({"completed_round": 0, "records": []}))
        with mock.patch.object(experiments, "run_experiment", fake):
            with runner.round_progress("test"):
                experiments.run_experiment(None, None, checkpoint_callback=callback)
            self.assertIs(experiments.run_experiment, fake)
        callback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
