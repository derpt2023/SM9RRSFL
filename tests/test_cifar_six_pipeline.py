"""Bounded protocol tests; no CIFAR download, CUDA run or old result needed."""
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import run_cifar_six_from_scratch as runner
from sm9rrsfl import experiments, fl


CONFIG = Path(__file__).resolve().parents[1] / "configs/cifar10_six_stable_v1.json"


def synthetic_run(config, accuracy=.8, asr=.05, nonfinite=0):
    records = [fl.RoundRecord(config.method, config.malicious_ratio, rd, accuracy,
                1 - accuracy, 100, 0, 0, 0, 0, "", attack_target_success_rate=asr,
                nonfinite_updates=nonfinite if rd == config.attack_start_round else 0,
                attack_active=bool(config.malicious_ratio and rd >= config.attack_start_round))
               for rd in range(config.rounds + 1)]
    malicious = tuple(str(i) for i in range(round(config.num_clients * config.malicious_ratio)))
    return fl.ExperimentResult(config, records, accuracy, 1 - accuracy, config.rounds,
                               malicious, (), nonfinite_updates=nonfinite)


def result_matrix(spec, tasks):
    groups = {}
    for task in tasks:
        config = fl.ExperimentConfig(**task["config"])
        attacked_fedavg = config.method == "fedavg" and config.malicious_ratio > 0
        result = synthetic_run(config, accuracy=.4 if attacked_fedavg else .8,
                               asr=.8 if attacked_fedavg else .05)
        groups.setdefault(task["candidate"]["candidate_id"], []).append(result)
    return groups


class SixMethodPipelineTests(unittest.TestCase):
    def setUp(self):
        self.spec = runner.load_spec(CONFIG)
        self.tasks = runner.build_tasks(self.spec, "validation")
        self.results = result_matrix(self.spec, self.tasks)

    def test_standalone_task_matrix_and_frozen_formal_parameters(self):
        self.assertEqual(len(self.tasks), 84)
        self.assertEqual(len({t["task_id"] for t in self.tasks}), 84)
        self.assertEqual({t["config"]["rounds"] for t in self.tasks}, {100})
        selected = {m: cs[0]["candidate_id"] for m, cs in self.spec["candidates"].items()}
        final = runner.build_tasks(self.spec, "final", selected)
        self.assertEqual(len(final), 180)
        self.assertEqual({t["config"]["seed"] for t in final}, {801, 802, 803})
        self.assertEqual({t["config"]["attack_boost"] for t in final}, {5.})
        self.assertTrue(all("-v8-" in t["candidate"]["candidate_id"] for t in final))
        self.assertEqual({t["candidate"]["candidate_id"] for t in final}, set(selected.values()))
        self.assertNotIn("source", self.spec)

    def test_changed_shared_attack_or_seed_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for mutate in (
                lambda s: s["shared_parameters"].update(attack_boost=1.),
                lambda s: s["final"].update(seeds=[701]),
                lambda s: s["candidates"]["vert"][0]["parameters"].update(lr=.01),
                lambda s: s["gates"].update(max_nonfinite_updates=1),
            ):
                with self.subTest(mutate=mutate):
                    spec = deepcopy(self.spec)
                    mutate(spec)
                    path.write_text(json.dumps(spec))
                    with self.assertRaises(ValueError):
                        runner.load_spec(path)

    def test_healthy_near_best_all_five_and_effective_attack_qualify(self):
        report = runner.select_validation(self.spec, self.results, self.tasks)
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertTrue(report["all_six_methods_healthy"])
        self.assertTrue(report["attack_effectiveness"]["passed"])
        self.assertEqual(set(report["ours_target"]["comparisons"]), set(runner.ALL_METHODS[1:]))
        self.assertFalse(report["official_test_used_for_selection"])
        for name, audit in report["ours_target"]["comparisons"].items():
            self.assertEqual(audit["reference_method"], name)
            self.assertEqual(audit["status"], "passed")

    def test_nonfinite_vert_cannot_veto_other_selections_or_qualify(self):
        for candidate in self.spec["candidates"]["vert"]:
            cid = candidate["candidate_id"]
            self.results[cid][2] = synthetic_run(self.results[cid][2].config, nonfinite=1)
        report = runner.select_validation(self.spec, self.results, self.tasks)
        self.assertEqual(report["status"], "needs_development")
        self.assertNotIn("vert", report["selected"])
        self.assertEqual(len(report["selected"]), 5)
        self.assertEqual(report["methods"]["vert"]["status"], "no_eligible_candidate")
        self.assertEqual(report["ours_target"]["status"], "incomplete")
        self.assertTrue(all("nonfinite_updates" in reason for reason in report["methods"]["vert"]["candidate_failures"].values()))

    def test_missing_vert_results_remain_explicit_without_exception(self):
        for candidate in self.spec["candidates"]["vert"]:
            self.results.pop(candidate["candidate_id"])
        report = runner.select_validation(self.spec, self.results, self.tasks)
        self.assertEqual(report["status"], "needs_development")
        self.assertEqual(len(report["selected"]), 5)
        self.assertEqual(report["ours_target"]["status"], "incomplete")

    def test_bad_clean_fedavg_reference_is_a_reported_failure(self):
        cid = self.spec["candidates"]["fedavg"][0]["candidate_id"]
        self.results[cid][0] = synthetic_run(self.results[cid][0].config, accuracy=float("nan"))
        report = runner.select_validation(self.spec, self.results, self.tasks)
        self.assertEqual(report["status"], "needs_development")
        self.assertIn("invalid accuracy", report["clean_reference_error"])
        self.assertFalse(report["all_six_methods_healthy"])

    def test_best_krum_prevents_claim_based_only_on_vert(self):
        cid = self.spec["candidates"]["krum"][0]["candidate_id"]
        self.results[cid] = [synthetic_run(r.config, accuracy=.9) for r in self.results[cid]]
        report = runner.select_validation(self.spec, self.results, self.tasks)
        self.assertEqual(report["status"], "needs_development")
        self.assertEqual(report["ours_target"]["comparisons"]["vert"]["status"], "passed")
        self.assertEqual(report["ours_target"]["comparisons"]["krum"]["status"], "unmet")

    def test_ineffective_attack_cannot_make_ours_qualified(self):
        cid = self.spec["candidates"]["fedavg"][0]["candidate_id"]
        self.results[cid] = [synthetic_run(r.config) for r in self.results[cid]]
        report = runner.select_validation(self.spec, self.results, self.tasks)
        self.assertEqual(report["ours_target"]["status"], "passed")
        self.assertFalse(report["attack_effectiveness"]["passed"])
        self.assertEqual(report["status"], "needs_development")

    def test_target_selection_changes_only_ours(self):
        good = self.spec["candidates"]["sm9rrs"][2]["candidate_id"]
        for candidate in self.spec["candidates"]["sm9rrs"]:
            cid = candidate["candidate_id"]
            if cid != good:
                self.results[cid] = [synthetic_run(r.config, asr=.4) for r in self.results[cid]]
        report = runner.select_validation(self.spec, self.results, self.tasks)
        self.assertEqual(report["selected"]["sm9rrs"], good)
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertEqual(report["selected"]["vert"], "vert-v8-006")

    def test_global_nonfinite_model_rejected_before_checkpoint_write(self):
        callback = mock.Mock()
        def fake_run(*args, **kwargs):
            kwargs["checkpoint_callback"]({"completed_round": 4, "params": np.array([float("nan")]), "records": []})
        with mock.patch.object(experiments, "run_experiment", fake_run):
            with self.assertRaises(FloatingPointError):
                with runner.round_observer("test"):
                    experiments.run_experiment(checkpoint_callback=callback)
            self.assertIs(experiments.run_experiment, fake_run)
        callback.assert_not_called()

    def test_nonfinite_forward_metric_rejected_before_checkpoint(self):
        config = fl.ExperimentConfig(**self.tasks[0]["config"])
        record = replace(synthetic_run(config).records[-1], attack_target_confidence=float("nan"))
        callback = mock.Mock()
        def fake_run(*args, **kwargs):
            kwargs["checkpoint_callback"]({"completed_round": 100, "params": np.array([1.]), "records": [record]})
        with mock.patch.object(experiments, "run_experiment", fake_run):
            with self.assertRaises(FloatingPointError):
                with runner.round_observer("test"):
                    experiments.run_experiment(checkpoint_callback=callback)
        callback.assert_not_called()

    def test_partial_final_aggregate_has_all_sixty_rows_and_missing_seeds(self):
        selected = {m: cs[0]["candidate_id"] for m, cs in self.spec["candidates"].items()}
        final = runner.build_tasks(self.spec, "final", selected)
        first = final[0]
        results = {first["candidate"]["candidate_id"]: [synthetic_run(fl.ExperimentConfig(**first["config"]))]}
        rows = runner.final_aggregate(final, results)
        self.assertEqual(len(rows), 60)
        ours = next(r for r in rows if r["method"] == "sm9rrs" and r["partition"] == "iid" and r["malicious_ratio"] == 0.)
        self.assertEqual(ours["expected_runs"], 3)
        self.assertEqual(ours["completed_runs"], 1)
        self.assertEqual(ours["status"], "incomplete")
        self.assertEqual(json.loads(ours["missing_seeds"]), [802, 803])
        self.assertTrue(all(r["completed_runs"] == 0 and r["status"] == "incomplete" for r in rows if r["method"] == "vert"))

    def test_round_observer_records_resume_round_and_persists_finite_state(self):
        contexts, context, callback = [], {}, mock.Mock()
        def fake_run(*args, **kwargs):
            contexts.append(dict(context))
            kwargs["checkpoint_callback"]({"completed_round": 5, "params": np.array([1.]), "records": []})
        with mock.patch.object(experiments, "run_experiment", fake_run):
            with runner.round_observer("test", context):
                experiments.run_experiment(checkpoint_callback=callback, resume_state={"completed_round": 4})
        self.assertEqual(contexts[0]["round"], 5)
        self.assertEqual(context["round"], 6)
        callback.assert_called_once()

    def test_identity_change_and_orphan_snapshot_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / "identity.json"
            runner.immutable_json(path, {"variant": "original"})
            with self.assertRaises(ValueError):
                runner.immutable_json(path, {"variant": "weak_quarantine"})
            task = self.tasks[0]
            folder = output / "tasks" / task["task_id"]
            folder.mkdir(parents=True)
            (folder / experiments.COMPLETED_RESULTS_SNAPSHOT).write_bytes(b"orphan")
            with self.assertRaises(ValueError):
                runner.checked_completed(output, task)

    def test_environment_allows_same_gpu_model_remap_but_rejects_different_model(self):
        metadata = {"requested_device": "cuda:0", "environment": {"CUDA_VISIBLE_DEVICES": "7"},
                    "actual_compute_device": {"logical_device": "cuda:0", "uuid": "A", "name": "4090D",
                                               "compute_capability": [8, 9], "total_memory_bytes": 24000},
                    "torch": {"logical_cuda_devices": [{"name": "4090D", "uuid": "A"}]},
                    "nvidia": {"gpus": [{"driver_version": "550"}]}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            runner.check_environment(output, metadata)
            remapped = deepcopy(metadata)
            remapped["actual_compute_device"].update(logical_device="cuda:1", uuid="B")
            remapped["requested_device"] = "cuda:1"
            remapped["environment"]["CUDA_VISIBLE_DEVICES"] = "6,7"
            runner.check_environment(output, remapped)
            remapped["actual_compute_device"]["name"] = "A100"
            with self.assertRaises(ValueError):
                runner.check_environment(output, remapped)

    def test_completed_pickle_repairs_missing_derived_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            task = {**self.tasks[0], "fingerprint": "test"}
            folder = output / "tasks" / task["task_id"]
            folder.mkdir(parents=True)
            result = synthetic_run(fl.ExperimentConfig(**task["config"]))
            with mock.patch.object(experiments, "finalize_config_checkpoint"):
                runner.repair_completed(output, task, result)
            self.assertTrue((folder / "metrics.json").exists())
            self.assertTrue((folder / "summary.csv").exists())
            self.assertTrue((folder / "rounds.csv").exists())

    def test_resume_keeps_first_attempt_numeric_stats_and_failure(self):
        from sm9rrsfl import stable_training
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            task = {**self.tasks[0], "fingerprint": "test"}
            manifest = {"spec": self.spec, "source_sha256": {}, "data_contract": {}}
            (output / "manifest.json").write_text(json.dumps(manifest))
            (output / "validation_plan.json").write_text(json.dumps({"tasks": [task]}))
            args = SimpleNamespace(output=output, phase="validation", worker=task["task_id"], devices=["cuda:0"], data_dir=None)
            stats = SimpleNamespace(summary=lambda: {"totals": {"backtracked_steps": 2, "retries": 3}})
            split = SimpleNamespace(calibration_dataset=object())
            result = synthetic_run(fl.ExperimentConfig(**task["config"]))
            with mock.patch.object(runner, "source_hashes", return_value={}), mock.patch.object(runner, "configure_strict_numerics"), mock.patch.object(runner, "worker_environment", return_value={"environment": {}}), mock.patch.object(runner, "load_split", return_value=(split, {})), mock.patch.object(stable_training, "stable_client_training", side_effect=lambda **kwargs: nullcontext(stats)), mock.patch.object(experiments, "run_measured_experiment", side_effect=[RuntimeError("simulated stop"), result]), mock.patch.object(experiments, "finalize_config_checkpoint"), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "simulated stop"):
                    runner.run_worker(args)
                runner.run_worker(args)
            folder = output / "tasks" / task["task_id"]
            report = json.loads((folder / "numerical_stats.json").read_text())
            self.assertEqual(len(report["attempts"]), 2)
            self.assertEqual(report["backtracked_steps_all_attempts"], 4)
            self.assertEqual(report["retries_all_attempts"], 6)
            self.assertTrue((folder / "failure.json").exists())

    def test_worker_failure_does_not_cancel_unrelated_task(self):
        launched = []
        class FakeProcess:
            def __init__(self, command, **kwargs):
                launched.append(command)
                self.stdout = io.StringIO("")
                self.code = 1 if len(launched) == 1 else 0
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def wait(self):
                return self.code
            def terminate(self):
                return None
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = SimpleNamespace(devices=["cuda:0"], data_dir=None)
            with mock.patch.object(runner.subprocess, "Popen", FakeProcess), mock.patch.object(runner, "checked_completed", return_value=None), redirect_stdout(io.StringIO()):
                failures = runner.execute_phase(args, self.tasks[:2], output)
            self.assertEqual(failures, 1)
            self.assertEqual(len(launched), 2)
            for command in launched:
                self.assertIn("--worker", command)

    def test_qualified_parent_runs_both_phases_and_creates_final_outputs(self):
        selected = runner.select_validation(self.spec, self.results, self.tasks)["selected"]
        final = runner.build_tasks(self.spec, "final", selected)
        final_results = result_matrix(self.spec, final)
        def statuses(tasks, groups):
            by_config = {runner.digest(runner.semantic_config(r.config)): r for runs in groups.values() for r in runs}
            return [{"task_id": t["task_id"], "method": t["method"], "status": "complete",
                     "metrics": runner.metrics(by_config[runner.digest(runner.semantic_config(t["config"]))])} for t in tasks]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = SimpleNamespace(config=CONFIG, output=output, devices=["cuda:0"], data_dir=None, phase="all")
            with mock.patch.object(runner, "configure_strict_numerics"), mock.patch.object(runner, "source_hashes", return_value={}), mock.patch.object(runner, "load_split", return_value=(object(), {})), mock.patch.object(runner, "execute_phase", return_value=0) as execution, mock.patch.object(runner, "collect_results", side_effect=[(self.results, statuses(self.tasks, self.results)), (final_results, statuses(final, final_results))]), redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run_parent(args), 0)
            self.assertEqual(execution.call_count, 2)
            self.assertTrue((output / "final_results" / "aggregate.csv").exists())
            self.assertTrue((output / "final_results" / "summary.csv").exists())
            report = json.loads((output / "final_summary.json").read_text())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(set(report["methods"]), set(runner.ALL_METHODS))
            self.assertFalse(report["parameters_reselected"])


if __name__ == "__main__":
    unittest.main()
