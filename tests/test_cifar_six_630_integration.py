"""Exercise the 630-to-180 lifecycle without downloading CIFAR or using CUDA."""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

import run_cifar_six_630 as runner
from sm9rrsfl import fl, torch_backend
from sm9rrsfl.datasets import ImageDataset
from tests.test_cifar_six_pipeline import result_matrix, synthetic_run


CONFIG = Path(__file__).resolve().parents[1] / "configs/cifar10_six_630_mean_v3.json"


def task_statuses(tasks, groups):
    by_key = {
        (cid, run.config.partition, run.config.malicious_ratio, run.config.seed): run
        for cid, runs in groups.items() for run in runs
    }
    statuses = []
    for task in tasks:
        cid, config = task["candidate"]["candidate_id"], task["config"]
        run = by_key.get((cid, config["partition"], config["malicious_ratio"], config["seed"]))
        statuses.append({
            "task_id": task["task_id"], "candidate_id": cid, "method": task["method"],
            "partition": config["partition"], "ratio": config["malicious_ratio"],
            "seed": config["seed"], "status": "complete" if run else "pending",
            "metrics": runner.base.metrics(run) if run else None,
            "error": None, "failure_record": None,
        })
    return statuses


class Six630LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.spec = runner.load_spec(CONFIG)
        self.tasks = runner.build_tasks(self.spec, "validation")
        self.groups = result_matrix(self.spec, self.tasks)

    def test_cli_rejects_duplicate_lanes_and_multi_device_worker_before_execution(self):
        for arguments in (["--devices", "cuda:0", "cuda:0"],
                          ["--worker", "a", "--phase", "validation", "--output", "unused",
                           "--devices", "cuda:0", "cuda:1"]):
            with self.subTest(arguments=arguments), \
                    mock.patch("sys.stderr", new_callable=io.StringIO), \
                    mock.patch.object(runner, "run_parent") as parent, \
                    mock.patch.object(runner, "run_worker") as worker:
                with self.assertRaises(SystemExit) as stopped:
                    runner.main(arguments)
                self.assertEqual(stopped.exception.code, 2)
                parent.assert_not_called()
                worker.assert_not_called()

    def invoke(self, output, *, phase="all", groups=None, status_transform=None,
               final_unhealthy=False):
        groups = self.groups if groups is None else groups
        launched = []
        collected = []

        def execute(args, tasks, destination):
            launched.append((tasks[0]["phase"], list(tasks)))
            return 0

        def collect(destination, tasks):
            collected.append(tasks[0]["phase"])
            if tasks[0]["phase"] == "validation":
                rows = task_statuses(tasks, groups)
                if status_transform:
                    status_transform(rows, tasks)
                return groups, rows
            results = result_matrix(self.spec, tasks)
            if final_unhealthy:
                cid = next(t["candidate"]["candidate_id"] for t in tasks if t["method"] == "sm9rrs")
                runs = results[cid]
                index = next(i for i, run in enumerate(runs) if run.config.malicious_ratio)
                runs[index] = synthetic_run(runs[index].config, nonfinite=1)
            return results, task_statuses(tasks, results)

        args = SimpleNamespace(config=CONFIG, output=output, data_dir=None,
                               devices=["cuda:0", "cuda:1"], phase=phase)
        with mock.patch.object(runner, "load_split", return_value=(object(), {})), \
                mock.patch.object(runner, "execute_phase", side_effect=execute), \
                mock.patch.object(runner, "collect_results", side_effect=collect), \
                mock.patch.object(runner.experiments, "write_result_files"), \
                redirect_stdout(io.StringIO()) as stdout:
            result = runner.run_parent(args)
        return result, launched, collected, stdout.getvalue()

    def test_restored_630_grid_and_fresh_180_final_grid(self):
        self.assertEqual(len(self.tasks), 630)
        self.assertEqual(len({task["task_id"] for task in self.tasks}), 630)
        self.assertEqual({task["config"]["seed"] for task in self.tasks}, {401, 402, 403})
        for candidates in self.spec["candidates"].values():
            for candidate in candidates:
                self.assertEqual(sum(task["candidate"]["candidate_id"] == candidate["candidate_id"]
                                     for task in self.tasks), 30)
        self.assertEqual({candidate["variant"] for candidate in self.spec["candidates"]["sm9rrs"]},
                         {"original"})
        self.assertEqual({(candidate["parameters"]["detector_distance_threshold"],
                           candidate["parameters"]["suspicion_remove_after"])
                          for candidate in self.spec["candidates"]["sm9rrs"]},
                         {(warning, remove) for warning in (1.25, 1.75, 2.5) for remove in (3, 5)})
        selected = {method: candidates[0]["candidate_id"]
                    for method, candidates in self.spec["candidates"].items()}
        final = runner.build_tasks(self.spec, "final", selected)
        self.assertEqual(len(final), 180)
        self.assertEqual({task["config"]["seed"] for task in final}, {901, 902, 903})
        self.assertEqual({task["method"] for task in final}, set(runner.ALL_METHODS))
        self.assertTrue(all(sum(task["method"] == method for task in final) == 30
                            for method in runner.ALL_METHODS))

    def test_mean_gate_qualified_parent_runs_all_six_and_retains_final_health_failures(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            code, launched, collected, text = self.invoke(output, final_unhealthy=True)
            self.assertEqual(code, 0)
            self.assertEqual([(phase, len(tasks)) for phase, tasks in launched],
                             [("validation", 630), ("final", 180)])
            self.assertEqual(collected, ["validation", "final"])
            validation = json.loads((output / "validation_summary.json").read_text())
            final = json.loads((output / "final_summary.json").read_text())
            self.assertEqual(validation["status"], "qualified_for_final")
            self.assertEqual(len(validation["selected"]), 6)
            self.assertTrue(final["full_execution_completed"])
            self.assertFalse(final["ours_independent_health_passed"])
            self.assertEqual(final["status"], "completed_with_health_failures")
            self.assertTrue(all(info["completed_runs"] == 30 for info in final["methods"].values()))
            self.assertIn("FINAL_RUNS 180", text)

    def test_unhealthy_baselines_do_not_veto_six_method_execution(self):
        for cid, runs in self.groups.items():
            if runs[0].config.method != "sm9rrs":
                self.groups[cid] = [synthetic_run(run.config, nonfinite=1) for run in runs]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            code, launched, _, _ = self.invoke(output)
            report = json.loads((output / "validation_summary.json").read_text())
            self.assertEqual(code, 0)
            self.assertEqual([(phase, len(tasks)) for phase, tasks in launched],
                             [("validation", 630), ("final", 180)])
            self.assertEqual(report["status"], "qualified_for_final")
            self.assertFalse(report["six_method_comparison_available"])
            self.assertFalse(report["mean_dual_gate"]["six_method_dual_optimality_assessed"])
            self.assertTrue(report["mean_dual_gate"]["six_method_observed_means_compared"])
            self.assertTrue(report["mean_dual_gate"]["comparison_includes_unqualified_references"])
            self.assertEqual(set(report["mean_dual_gate"]["available_baselines"]),
                             set(runner.ALL_METHODS[1:]))
            self.assertEqual(report["mean_dual_gate"]["unavailable_baselines"], {})
            for method in runner.ALL_METHODS[1:]:
                self.assertEqual(report["methods"][method]["selection_status"],
                                 "best_scored_unqualified")
                self.assertFalse(report["mean_dual_gate"]["available_baselines"][method]["health_qualified"])
                self.assertTrue(report["methods"][method]["selected_without_valid_validation"])

    def test_near_vert_promotes_all_six_even_when_another_baseline_is_better(self):
        for cid, runs in self.groups.items():
            method = runs[0].config.method
            if method == "sm9rrs":
                self.groups[cid] = [synthetic_run(run.config, accuracy=.796, asr=.059) for run in runs]
            elif method == "alignins":
                self.groups[cid] = [synthetic_run(run.config, accuracy=.85, asr=.01) for run in runs]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            code, launched, _, _ = self.invoke(output)
            report = json.loads((output / "validation_summary.json").read_text())
            gate = report["mean_dual_gate"]
            ours = report["selected"]["sm9rrs"]
            self.assertEqual(code, 0)
            self.assertEqual([(phase, len(tasks)) for phase, tasks in launched],
                             [("validation", 630), ("final", 180)])
            self.assertFalse(gate["ours_candidate_targets"][ours]["joint_best_passed"])
            self.assertTrue(gate["ours_candidate_targets"][ours]["near_vert_passed"])
            self.assertEqual(gate["pass_route"], "near_vert")

    def test_best_scored_failed_vert_is_near_reference_and_final_selection_source(self):
        vert_ids = [candidate["candidate_id"] for candidate in self.spec["candidates"]["vert"]]
        best_vert = vert_ids[-1]
        for cid, runs in self.groups.items():
            method = runs[0].config.method
            if method == "sm9rrs":
                self.groups[cid] = [synthetic_run(run.config, accuracy=.796, asr=.059) for run in runs]
            elif method == "vert":
                accuracy, asr = (.8, .05) if cid == best_vert else (.76, .1)
                self.groups[cid] = [synthetic_run(run.config, accuracy=accuracy, asr=asr, nonfinite=1)
                                    for run in runs]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            code, launched, _, _ = self.invoke(output)
            report = json.loads((output / "validation_summary.json").read_text())
            final = json.loads((output / "final_summary.json").read_text())
            self.assertEqual(code, 0)
            self.assertEqual([(phase, len(tasks)) for phase, tasks in launched],
                             [("validation", 630), ("final", 180)])
            self.assertEqual(report["selected"]["vert"], best_vert)
            self.assertEqual(report["mean_dual_gate"]["pass_route"], "near_vert")
            reference = report["mean_dual_gate"]["available_baselines"]["vert"]
            self.assertEqual(reference["candidate_id"], best_vert)
            self.assertFalse(reference["health_qualified"])
            self.assertEqual(report["methods"]["vert"]["selection_status"], "best_scored_unqualified")
            self.assertTrue(final["methods"]["vert"]["selected_without_valid_validation"])
            self.assertEqual(final["methods"]["vert"]["validation_selection_status"],
                             "best_scored_unqualified")
            final_vert = [task for task in launched[-1][1] if task["method"] == "vert"]
            self.assertEqual(len(final_vert), 30)
            self.assertTrue(all(task["candidate"]["candidate_id"] == best_vert for task in final_vert))

    def test_all_unscorable_vert_uses_fixed_fallback_without_inventing_comparison(self):
        for candidate in self.spec["candidates"]["vert"]:
            self.groups[candidate["candidate_id"]] = []
        with TemporaryDirectory() as directory:
            output = Path(directory)

            def record_failures(rows, tasks):
                by_id = {task["task_id"]: task for task in tasks}
                for row in rows:
                    if row["method"] != "vert":
                        continue
                    task = by_id[row["task_id"]]
                    folder = output / "tasks" / task["task_id"]
                    folder.mkdir(parents=True, exist_ok=True)
                    runner.write_json(folder / "task.json", task)
                    runner.write_json(folder / "failure.json", {
                        "task_id": task["task_id"], "exception": "FloatingPointError",
                        "message": "nonfinite VERT prediction",
                        "execution_context": {
                            "task_id": task["task_id"], "candidate_id": task["candidate"]["candidate_id"],
                            "study_phase": "validation",
                        },
                    })
                    row.update(status="failed", failure_record=str(folder / "failure.json"))

            code, launched, _, _ = self.invoke(output, status_transform=record_failures)
            report = json.loads((output / "validation_summary.json").read_text())
            self.assertEqual(code, 0)
            self.assertEqual([(phase, len(tasks)) for phase, tasks in launched],
                             [("validation", 630), ("final", 180)])
            self.assertEqual(report["selected"]["vert"], self.spec["fallback_candidates"]["vert"])
            self.assertEqual(report["methods"]["vert"]["selection_status"], "fixed_fallback_unqualified")
            self.assertNotIn("vert", report["mean_dual_gate"]["available_baselines"])
            self.assertIn("vert", report["mean_dual_gate"]["unavailable_baselines"])
            self.assertEqual(report["mean_dual_gate"]["pass_route"], "joint_best")
            self.assertFalse(report["mean_dual_gate"]["ours_candidate_targets"]
                             [report["selected"]["sm9rrs"]]["near_vert_passed"])

    def test_unmet_mean_target_stops_before_creating_final_plan(self):
        for cid, runs in self.groups.items():
            if runs[0].config.method == "sm9rrs":
                self.groups[cid] = [synthetic_run(run.config, accuracy=.8, asr=.4) for run in runs]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            code, launched, _, text = self.invoke(output)
            self.assertEqual(code, 0)
            self.assertEqual([phase for phase, _ in launched], ["validation"])
            self.assertFalse((output / "final_plan.json").exists())
            self.assertIn("FINAL_NOT_STARTED", text)

    def test_unhealthy_ours_cannot_promote_even_with_superior_metrics(self):
        for cid, runs in self.groups.items():
            if runs[0].config.method == "sm9rrs":
                self.groups[cid] = [synthetic_run(run.config, accuracy=.85, asr=.01, nonfinite=1)
                                    for run in runs]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            code, launched, _, text = self.invoke(output)
            report = json.loads((output / "validation_summary.json").read_text())
            self.assertEqual(code, 0)
            self.assertEqual([phase for phase, _ in launched], ["validation"])
            self.assertFalse(report["ours_health_passed"])
            self.assertNotIn("sm9rrs", report["selected"])
            self.assertFalse((output / "final_plan.json").exists())
            self.assertIn("FINAL_NOT_STARTED", text)

    def test_pending_and_corrupt_validation_cannot_be_fallback_evidence(self):
        for field, value in (("status", "pending"), ("error", "damaged completed snapshot")):
            with self.subTest(field=field), TemporaryDirectory() as directory:
                output = Path(directory)
                def invalid(rows, tasks):
                    row = next(row for row in rows if row["method"] == "vert")
                    row[field] = value
                code, launched, _, _ = self.invoke(output, phase="final", status_transform=invalid)
                self.assertEqual(code, 2)
                self.assertEqual(launched, [])
                report = json.loads((output / "validation_summary.json").read_text())
                self.assertEqual(report["status"], "validation_evidence_incomplete")
                self.assertFalse((output / "final_plan.json").exists())

    def test_identity_verified_algorithm_failure_allowed_but_resource_failure_blocks(self):
        for exception, message, context_valid, expected in (
            ("FloatingPointError", "nonfinite VERT prediction", True, 0),
            ("OutOfMemoryError", "CUDA out of memory", True, 2),
            ("KeyboardInterrupt", "", True, 2),
            ("ModuleNotFoundError", "No module named 'torch'", True, 2),
            ("RuntimeError", "CUDA error: invalid device ordinal", True, 2),
            ("FileNotFoundError", "missing CIFAR batch file", True, 2),
            ("NameError", "name 'metrics' is not defined", True, 2),
            ("FloatingPointError", "nonfinite VERT prediction", False, 2),
        ):
            with self.subTest(exception=exception, context_valid=context_valid), TemporaryDirectory() as directory:
                output = Path(directory)
                missing = next(task for task in self.tasks if task["method"] == "vert")
                cid, cfg = missing["candidate"]["candidate_id"], missing["config"]
                groups = dict(self.groups)
                groups[cid] = [run for run in self.groups[cid]
                               if (run.config.partition, run.config.malicious_ratio, run.config.seed) !=
                               (cfg["partition"], cfg["malicious_ratio"], cfg["seed"])]
                def failed(rows, tasks):
                    task = next(task for task in tasks if task["task_id"] == missing["task_id"])
                    folder = output / "tasks" / task["task_id"]
                    folder.mkdir(parents=True, exist_ok=True)
                    runner.write_json(folder / "task.json", task)
                    runner.write_json(folder / "failure.json", {
                        "task_id": task["task_id"], "exception": exception, "message": message,
                        "execution_context": {
                            "task_id": task["task_id"], "candidate_id": task["candidate"]["candidate_id"],
                            "study_phase": "validation",
                        } if context_valid else None,
                    })
                    row = next(row for row in rows if row["task_id"] == task["task_id"])
                    row.update(status="failed", failure_record=str(folder / "failure.json"))
                code, launched, _, _ = self.invoke(output, phase="final", groups=groups,
                                                   status_transform=failed)
                self.assertEqual(code, expected)
                self.assertEqual([phase for phase, _ in launched], ["final"] if expected == 0 else [])

    def test_frozen_final_plan_resumes_without_reexecuting_validation(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            self.invoke(output)
            frozen = (output / "final_plan.json").read_bytes()
            code, launched, collected, _ = self.invoke(output)
            self.assertEqual(code, 0)
            self.assertEqual([phase for phase, _ in launched], ["final"])
            self.assertEqual(collected, ["validation", "final"])
            self.assertEqual((output / "final_plan.json").read_bytes(), frozen)

    def test_frozen_plan_does_not_bypass_validation_audit(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            self.invoke(output)
            frozen = (output / "final_plan.json").read_bytes()
            def invalid(rows, tasks):
                rows[0]["error"] = "changed validation snapshot"
            code, launched, collected, _ = self.invoke(output, status_transform=invalid)
            self.assertEqual(code, 2)
            self.assertEqual(launched, [])
            self.assertEqual(collected, ["validation"])
            self.assertEqual((output / "final_plan.json").read_bytes(), frozen)

    def test_changed_selection_cannot_replace_an_existing_final_plan(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            self.invoke(output)
            frozen = (output / "final_plan.json").read_bytes()
            old_selected = json.loads(frozen)["selected"]["sm9rrs"]
            replacement = next(candidate["candidate_id"]
                               for candidate in self.spec["candidates"]["sm9rrs"]
                               if candidate["candidate_id"] != old_selected)
            changed = dict(self.groups)
            changed[replacement] = [synthetic_run(run.config, accuracy=.82, asr=.01)
                                    for run in changed[replacement]]
            with self.assertRaisesRegex(ValueError, "selection differs"):
                self.invoke(output, groups=changed)
            self.assertEqual((output / "final_plan.json").read_bytes(), frozen)

    def test_corrupt_frozen_plan_rejected_without_mutation_or_training(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            self.invoke(output)
            plan_path = output / "final_plan.json"
            plan = json.loads(plan_path.read_text())
            plan["tasks"][0]["config"]["lr"] = .123
            runner.write_json(plan_path, plan)
            frozen = plan_path.read_bytes()
            with self.assertRaises(ValueError):
                self.invoke(output)
            self.assertEqual(plan_path.read_bytes(), frozen)

    def test_manifest_includes_new_policy_and_runner_with_original_sources(self):
        manifest = runner.build_manifest(self.spec, {}, CONFIG.parent.parent)
        hashes = manifest["source_sha256"]
        self.assertIn("run_cifar_six_630.py", hashes)
        self.assertIn("cifar_mean_dual_gate.py", hashes)
        for name, digest in runner.base.source_hashes(CONFIG.parent.parent).items():
            self.assertEqual(hashes[name], digest)
        self.assertFalse(manifest["baseline_algorithms_modified_by_runner"])
        self.assertFalse(manifest["shared_numerical_optimizer_changed"])
        self.assertFalse(manifest["formal_weights_inherited_from_validation"])

    def test_scheduler_launches_new_worker_entry_point(self):
        process = mock.MagicMock()
        process.stdout = []
        process.wait.return_value = 0
        process.__enter__.return_value = process
        args = SimpleNamespace(devices=["cuda:0"], data_dir=None)
        with TemporaryDirectory() as directory, \
                mock.patch.object(runner.subprocess, "Popen", return_value=process) as launch, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.execute_phase(args, self.tasks[:1], Path(directory)), 0)
        command = launch.call_args.args[0]
        self.assertEqual(Path(command[2]), Path(runner.__file__).resolve())
        self.assertIn("--worker", command)
        self.assertEqual(command[command.index("--worker") + 1], self.tasks[0]["task_id"])

    def test_worker_rejects_a_manifest_missing_new_policy_identity_before_training(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            task = self.tasks[0]
            runner.write_json(output / "manifest.json", {
                "spec": self.spec, "data_contract": {},
                "source_sha256": runner.base.source_hashes(CONFIG.parent.parent),
            })
            runner.write_json(output / "validation_plan.json", {"tasks": [task]})
            args = SimpleNamespace(output=output, phase="validation", worker=task["task_id"],
                                   devices=["cuda:0"], data_dir=None)
            with mock.patch.object(runner, "worker_environment") as environment, \
                    mock.patch.object(runner.experiments, "run_measured_experiment") as train:
                with self.assertRaisesRegex(ValueError, "source code changed"):
                    runner.run_worker(args)
            environment.assert_not_called()
            train.assert_not_called()


class Six630RealWorkerTests(unittest.TestCase):
    def test_worker_original_training_and_completed_cache_reuse(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        rng = np.random.default_rng(631)
        dataset = ImageDataset(
            rng.normal(0, .1, (20, 3, 32, 32)).astype(np.float32), np.arange(20) % 10,
            rng.normal(0, .1, (10, 3, 32, 32)).astype(np.float32), np.arange(10),
            name="cifar10", x_attack=rng.normal(0, .1, (20, 3, 32, 32)).astype(np.float32),
            y_attack=np.arange(20) % 10)
        config = fl.ExperimentConfig(
            num_clients=4, rounds=2, detector_window=3, attack_start_round=4,
            attack_target_count=2, malicious_ratio=.25, compute_backend="torch",
            device="cpu", crypto_mode="simulated", early_stop=False,
            lr=.005, batch_size=5, local_epochs=1, attack_epochs=1,
            attack_stealth_steps=1, attack_boost=5., seed=631)
        task = {"task_id": "integration_630_worker", "phase": "validation", "method": "sm9rrs",
                "candidate": {"candidate_id": "sm9rrs-v9-001", "variant": "original"},
                "config": asdict(config), "fingerprint": "integration630"}
        with TemporaryDirectory() as directory:
            output = Path(directory)
            runner.write_json(output / "manifest.json", {"spec": {}, "data_contract": {},
                "source_sha256": runner.source_hashes(CONFIG.parent.parent)})
            runner.write_json(output / "validation_plan.json", {"tasks": [task]})
            args = SimpleNamespace(output=output, phase="validation", worker=task["task_id"],
                                   devices=["cpu"], data_dir=None)
            with mock.patch.object(runner, "worker_environment", return_value={"environment": {}}), \
                    mock.patch.object(runner, "load_split", return_value=(
                        SimpleNamespace(calibration_dataset=dataset), {})), \
                    redirect_stdout(io.StringIO()):
                original_optimizer = torch_backend.TorchTrainingContext
                runner.run_worker(args)
                self.assertIs(torch_backend.TorchTrainingContext, original_optimizer)
                result = runner.checked_completed(output, task)
                self.assertEqual(result.stopped_round, config.rounds)
                self.assertEqual(result.nonfinite_updates, 0)
                with mock.patch.object(runner.experiments, "run_measured_experiment",
                                       side_effect=AssertionError("completed worker cannot retrain")):
                    runner.run_worker(args)
                self.assertFalse((output / "tasks" / task["task_id"] / "numerical_stats.json").exists())


if __name__ == "__main__":
    unittest.main()
