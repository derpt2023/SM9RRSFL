"""Exercise the expanded MNIST-target lifecycle without CIFAR data or CUDA."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import run_cifar_six_mnist_gate as runner
from tests.test_cifar_six_630_integration import task_statuses
from tests.test_cifar_six_pipeline import result_matrix, synthetic_run


CONFIG = Path(__file__).resolve().parents[1] / "configs/cifar10_six_mnist_gate_v4.json"


class MnistGateLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.spec = runner.load_spec(CONFIG)
        # Candidate counts are dynamic; retain the complete protocol matrix.
        for method, candidates in self.spec["candidates"].items():
            self.spec["candidates"][method] = candidates[:2 if method in ("sm9rrs", "vert") else 1]
        self.spec["run_budget"] = runner.run_budget(self.spec)
        self.config_path = self.directory / "small_v4.json"
        runner.write_json(self.config_path, self.spec)
        self.spec = runner.load_spec(self.config_path)
        self.tasks = runner.build_tasks(self.spec, "validation")
        self.groups = result_matrix(self.spec, self.tasks)
        self.assertEqual(len(self.tasks), 240)

    def invoke(self, output, *, phase="all", groups=None, status_transform=None,
               final_unhealthy=False):
        groups = self.groups if groups is None else groups
        launched, collected = [], []

        def execute(args, tasks, destination):
            launched.append((tasks[0]["phase"], list(tasks)))
            return 0

        def collect(destination, tasks):
            collected.append(tasks[0]["phase"])
            if tasks[0]["phase"] == "validation":
                statuses = task_statuses(tasks, groups)
                if status_transform:
                    status_transform(statuses, tasks)
                return groups, statuses
            results = result_matrix(self.spec, tasks)
            if final_unhealthy:
                cid = next(task["candidate"]["candidate_id"] for task in tasks if task["method"] == "sm9rrs")
                index = next(i for i, run in enumerate(results[cid]) if run.config.malicious_ratio)
                results[cid][index] = synthetic_run(results[cid][index].config, nonfinite=1)
            return results, task_statuses(tasks, results)

        args = SimpleNamespace(config=self.config_path, output=output, data_dir=None,
                               devices=["cuda:0", "cuda:1"], phase=phase)
        with mock.patch.object(runner, "load_split", return_value=(object(), {})), \
                mock.patch.object(runner, "execute_phase", side_effect=execute), \
                mock.patch.object(runner, "collect_results", side_effect=collect), \
                mock.patch.object(runner.experiments, "write_result_files"), \
                redirect_stdout(io.StringIO()) as stdout:
            code = runner.run_parent(args)
        return code, launched, collected, stdout.getvalue()

    def failure_transform(self, output, *, methods=("vert",), exception="FloatingPointError",
                          message="nonfinite VERT prediction", valid_identity=True,
                          valid_context=True):
        def transform(statuses, tasks):
            by_id = {task["task_id"]: task for task in tasks}
            for row in statuses:
                if row["method"] not in methods or row["status"] != "pending":
                    continue
                task = by_id[row["task_id"]]
                folder = output / "tasks" / task["task_id"]
                folder.mkdir(parents=True, exist_ok=True)
                identity = deepcopy(task)
                if not valid_identity:
                    identity["config"]["seed"] = 999999
                runner.write_json(folder / "task.json", identity)
                runner.write_json(folder / "failure.json", {
                    "task_id": task["task_id"], "exception": exception, "message": message,
                    "execution_context": {"task_id": task["task_id"],
                                          "candidate_id": task["candidate"]["candidate_id"],
                                          "study_phase": "validation"} if valid_context else None,
                })
                row.update(status="failed", failure_record=str(folder / "failure.json"))
        return transform

    def without_vert_results(self):
        return {cid: runs for cid, runs in self.groups.items() if runs[0].config.method != "vert"}

    def test_qualified_ours_and_failed_scorable_baselines_still_run_all_180(self):
        for method in runner.ALL_METHODS[1:]:
            for index, candidate in enumerate(self.spec["candidates"][method]):
                cid = candidate["candidate_id"]
                self.groups[cid] = [synthetic_run(run.config, accuracy=.79 + .01 * index,
                                                   asr=.05, nonfinite=1) for run in self.groups[cid]]
        output = self.directory / "study"
        code, launched, collected, stdout = self.invoke(output, final_unhealthy=True)
        self.assertEqual(code, 0)
        self.assertEqual([(phase, len(tasks)) for phase, tasks in launched],
                         [("validation", 240), ("final", 180)])
        self.assertEqual(collected, ["validation", "final"])
        validation = json.loads((output / "validation_summary.json").read_text())
        final = json.loads((output / "final_summary.json").read_text())
        self.assertEqual(validation["status"], "qualified_for_final")
        self.assertTrue(validation["ours_mnist_target_passed"])
        self.assertFalse(validation["ours_target"]["reference_health_qualified"])
        self.assertTrue(validation["ours_target"]["reference_scorable"])
        self.assertFalse(validation["six_method_comparison_available"])
        for method in runner.ALL_METHODS[1:]:
            expected = self.spec["candidates"][method][-1]["candidate_id"]
            self.assertEqual(validation["selected"][method], expected)
            info = validation["methods"][method]
            self.assertEqual(info["selection_status"], "best_scored_unqualified")
            self.assertFalse(info["health_qualified"])
            self.assertIn("nonfinite_updates", info["validation_selected_candidate_failures"])
            self.assertIsNotNone(info["selection_raw_score"])
            self.assertTrue(final["methods"][method]["selected_without_valid_validation"])
            self.assertEqual(final["methods"][method]["validation_selection_status"], "best_scored_unqualified")
            self.assertIn("nonfinite_updates", final["methods"][method]["validation_selected_candidate_failures"])
        self.assertTrue(final["full_execution_completed"])
        self.assertFalse(final["ours_independent_health_passed"])
        self.assertEqual(final["status"], "completed_with_health_failures")
        self.assertFalse(final["formal_results_used_for_selection"])
        self.assertTrue(all(info["completed_runs"] == 30 for info in final["methods"].values()))
        self.assertIn("FINAL_RUNS 180", stdout)

    def test_healthy_ours_with_unmet_target_stops_before_formal_plan(self):
        for cid, runs in self.groups.items():
            if runs[0].config.method == "sm9rrs":
                self.groups[cid] = [synthetic_run(run.config, accuracy=.8, asr=.051) for run in runs]
        output = self.directory / "unmet"
        code, launched, _, stdout = self.invoke(output)
        report = json.loads((output / "validation_summary.json").read_text())
        self.assertEqual(code, 0)
        self.assertEqual([phase for phase, _ in launched], ["validation"])
        self.assertEqual(report["status"], "needs_ours_target_development")
        self.assertTrue(report["ours_health_passed"])
        self.assertFalse(report["ours_mnist_target_passed"])
        self.assertNotIn("sm9rrs", report["selected"])
        self.assertFalse((output / "final_plan.json").exists())
        self.assertIn("FINAL_NOT_STARTED", stdout)

    def test_precomputed_baseline_fallback_cannot_start_formal_when_ours_target_fails(self):
        for method, candidates in self.spec["candidates"].items():
            for index, candidate in enumerate(candidates):
                cid = candidate["candidate_id"]
                ours = method == "sm9rrs"
                self.groups[cid] = [synthetic_run(run.config,
                    accuracy=.8 if ours else .79 + .001 * index,
                    asr=.051 if ours else .05, nonfinite=0 if ours else 1)
                    for run in self.groups[cid]]
        output = self.directory / "fallback_with_unmet_ours"
        code, launched, collected, stdout = self.invoke(output)
        report = json.loads((output / "validation_summary.json").read_text())
        self.assertEqual(code, 0)
        self.assertEqual([phase for phase, _ in launched], ["validation"])
        self.assertEqual(collected, ["validation"])
        self.assertEqual(report["status"], "needs_ours_target_development")
        self.assertTrue(report["ours_health_passed"])
        self.assertFalse(report["ours_mnist_target_passed"])
        self.assertEqual(set(report["selected"]), set(runner.ALL_METHODS[1:]))
        for method in runner.ALL_METHODS[1:]:
            self.assertEqual(report["selected"][method], self.spec["candidates"][method][-1]["candidate_id"])
            self.assertEqual(report["methods"][method]["selection_status"], "best_scored_unqualified")
            self.assertFalse(report["methods"][method]["health_qualified"])
        self.assertFalse((output / "final_plan.json").exists())
        self.assertFalse((output / "final_summary.json").exists())
        self.assertFalse((output / "final_results").exists())
        self.assertIn("FINAL_NOT_STARTED", stdout)

    def test_missing_or_damaged_snapshot_is_not_algorithm_failure_evidence(self):
        for missing_kind in ("pending", "damaged", "absent_status"):
            with self.subTest(missing_kind=missing_kind):
                output = self.directory / missing_kind

                def invalid(statuses, tasks):
                    index = next(i for i, row in enumerate(statuses) if row["method"] == "vert")
                    if missing_kind == "absent_status":
                        del statuses[index]
                    elif missing_kind == "damaged":
                        statuses[index]["error"] = "damaged completed snapshot"
                    else:
                        statuses[index]["status"] = "pending"

                code, launched, _, _ = self.invoke(output, phase="final", status_transform=invalid)
                report = json.loads((output / "validation_summary.json").read_text())
                self.assertEqual(code, 2)
                self.assertEqual(launched, [])
                self.assertEqual(report["status"], "validation_evidence_incomplete")
                self.assertTrue(report["blockers"])
                self.assertFalse((output / "final_plan.json").exists())

    def test_resource_environment_and_unidentified_failures_block_without_algorithm_verdict(self):
        cases = (
            ("oom", "OutOfMemoryError", "CUDA out of memory", True, True),
            ("missing_data", "FileNotFoundError", "missing CIFAR batch file", True, True),
            ("missing_module", "ModuleNotFoundError", "No module named torch", True, True),
            ("bad_identity", "FloatingPointError", "nonfinite VERT prediction", False, True),
            ("no_training_context", "FloatingPointError", "nonfinite VERT prediction", True, False),
        )
        for name, exception, message, identity, context in cases:
            with self.subTest(name=name):
                output = self.directory / name
                transform = self.failure_transform(output, exception=exception, message=message,
                                                   valid_identity=identity, valid_context=context)
                code, launched, _, _ = self.invoke(output, phase="final", groups=self.without_vert_results(),
                                                   status_transform=transform)
                report = json.loads((output / "validation_summary.json").read_text())
                self.assertEqual(code, 2)
                self.assertEqual(launched, [])
                self.assertEqual(report["status"], "validation_evidence_incomplete")
                self.assertTrue(report["blockers"])
                self.assertFalse((output / "final_plan.json").exists())

    def test_identity_verified_numerical_vert_failures_allow_explicit_partial_target_route(self):
        output = self.directory / "numerical_failures"
        code, launched, _, _ = self.invoke(output, phase="final", groups=self.without_vert_results(),
                                           status_transform=self.failure_transform(output))
        report = json.loads((output / "validation_summary.json").read_text())
        final = json.loads((output / "final_summary.json").read_text())
        self.assertEqual(code, 0)
        self.assertEqual([(phase, len(tasks)) for phase, tasks in launched], [("final", 180)])
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertEqual(report["methods"]["vert"]["selection_status"], "fixed_fallback_unqualified")
        self.assertEqual(report["selected"]["vert"], self.spec["fallback_candidates"]["vert"])
        self.assertFalse(report["methods"]["vert"]["health_qualified"])
        self.assertFalse(report["methods"]["vert"]["scorable"])
        self.assertEqual(report["ours_target"]["status"], "partially_assessed")
        self.assertEqual(report["ours_target"]["relative_target_status"], "unassessed")
        self.assertFalse(report["ours_target"]["full_target_passed"])
        self.assertTrue(report["ours_absolute_target_passed"])
        self.assertFalse(report["ours_mnist_target_passed"])
        self.assertEqual(report["pass_route"], "mnist_absolute_target_without_scorable_vert")
        self.assertEqual(final["validation_ours_target"]["relative_target_status"], "unassessed")
        self.assertFalse(final["validation_mnist_target_gate"]["full_target_passed"])

    def test_frozen_final_resumes_without_retraining_validation(self):
        output = self.directory / "resume"
        self.invoke(output)
        frozen = (output / "final_plan.json").read_bytes()
        code, launched, collected, stdout = self.invoke(output)
        self.assertEqual(code, 0)
        self.assertEqual([phase for phase, _ in launched], ["final"])
        self.assertEqual(collected, ["validation", "final"])
        self.assertEqual((output / "final_plan.json").read_bytes(), frozen)
        self.assertIn("VALIDATION_REUSE", stdout)

    def test_frozen_plan_still_requires_valid_unchanged_validation(self):
        output = self.directory / "frozen_audit"
        self.invoke(output)
        frozen = (output / "final_plan.json").read_bytes()

        def corrupt(statuses, tasks):
            statuses[0]["error"] = "changed validation snapshot"

        code, launched, collected, _ = self.invoke(output, status_transform=corrupt)
        self.assertEqual(code, 2)
        self.assertEqual(launched, [])
        self.assertEqual(collected, ["validation"])
        self.assertEqual((output / "final_plan.json").read_bytes(), frozen)

    def test_changed_selection_cannot_replace_frozen_final_plan(self):
        output = self.directory / "changed_selection"
        self.invoke(output)
        frozen = (output / "final_plan.json").read_bytes()
        old = json.loads(frozen)["selected"]["sm9rrs"]
        replacement = next(c["candidate_id"] for c in self.spec["candidates"]["sm9rrs"] if c["candidate_id"] != old)
        changed = dict(self.groups)
        changed[replacement] = [synthetic_run(run.config, accuracy=.81, asr=.04)
                                for run in changed[replacement]]
        with self.assertRaisesRegex(ValueError, "selection differs"):
            self.invoke(output, groups=changed)
        self.assertEqual((output / "final_plan.json").read_bytes(), frozen)

    def test_manifest_freezes_new_policy_generator_and_entrypoint(self):
        manifest = runner.build_manifest(self.spec, {}, CONFIG.parent.parent)
        self.assertEqual(manifest["schema_version"], 4)
        self.assertEqual(manifest["validation_run_count"], 240)
        self.assertEqual(manifest["final_run_count"], 180)
        self.assertEqual(manifest["promotion_requires"], "ours_health_and_mnist_scenario_targets")
        for name in ("run_cifar_six_mnist_gate.py", "cifar_mnist_target_gate.py", "cifar_expanded_candidates.py"):
            self.assertIn(name, manifest["source_sha256"])

        for name, digest in runner.base.source_hashes(CONFIG.parent.parent).items():
            self.assertEqual(manifest["source_sha256"][name], digest)
        self.assertFalse(manifest["baseline_algorithms_modified_by_runner"])
        self.assertFalse(manifest["shared_numerical_optimizer_changed"])
        self.assertFalse(manifest["formal_weights_inherited_from_validation"])

    def test_changed_candidates_require_correct_declared_budget(self):
        self.spec["run_budget"]["validation_runs"] = 2610
        runner.write_json(self.config_path, self.spec)
        with self.assertRaisesRegex(ValueError, "declared run budget differs"):
            runner.load_spec(self.config_path)

    def test_scheduler_uses_new_worker_script(self):
        process = mock.MagicMock()
        process.stdout = []
        process.wait.return_value = 0
        process.__enter__.return_value = process
        args = SimpleNamespace(devices=["cuda:0"], data_dir=None)
        output = self.directory / "scheduler"
        output.mkdir()
        with mock.patch.object(runner.subprocess, "Popen", return_value=process) as launch, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.execute_phase(args, self.tasks[:1], output), 0)
        command = launch.call_args.args[0]
        self.assertEqual(Path(command[2]), Path(runner.__file__).resolve())
        self.assertEqual(command[command.index("--worker") + 1], self.tasks[0]["task_id"])

    def test_worker_rejects_missing_new_source_identity_before_training(self):
        output = self.directory / "source_identity"
        output.mkdir()
        task = self.tasks[0]
        runner.write_json(output / "manifest.json", {"spec": self.spec, "data_contract": {},
                          "source_sha256": runner.base.source_hashes(CONFIG.parent.parent)})
        runner.write_json(output / "validation_plan.json", {"tasks": [task]})
        args = SimpleNamespace(output=output, phase="validation", worker=task["task_id"],
                               devices=["cuda:0"], data_dir=None)
        with mock.patch.object(runner, "worker_environment") as environment, \
                mock.patch.object(runner.experiments, "run_measured_experiment") as train:
            with self.assertRaisesRegex(ValueError, "source code changed"):
                runner.run_worker(args)
        environment.assert_not_called()
        train.assert_not_called()

    def test_plan_only_requires_no_data_gpu_or_output_write(self):
        output = self.directory / "must_not_exist"
        with mock.patch.object(runner, "load_split", side_effect=AssertionError("no data access")), \
                mock.patch.object(runner, "worker_environment", side_effect=AssertionError("no GPU access")), \
                mock.patch.object(runner, "run_parent", side_effect=AssertionError("no execution")), \
                mock.patch.object(runner, "write_json", side_effect=AssertionError("no output writes")), \
                mock.patch("torch.cuda.is_available", side_effect=AssertionError("no CUDA access")), \
                redirect_stdout(io.StringIO()) as stdout:
            code = runner.main(["--config", str(self.config_path), "--output", str(output),
                                "--plan-only", "--devices", "no_gpu_required"])
        self.assertEqual(code, 0)
        self.assertFalse(output.exists())
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["output_dir"], str(output.resolve()))
        self.assertEqual(summary["validation_runs"], 240)
        self.assertEqual(summary["final_runs"], 180)
        self.assertEqual(summary["performance_target"], self.spec["performance_target"])
        self.assertEqual(summary["candidate_counts"], {m: len(cs) for m, cs in self.spec["candidates"].items()})


if __name__ == "__main__":
    unittest.main()
