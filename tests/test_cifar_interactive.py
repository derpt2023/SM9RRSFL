"""Real v7 selection and saved snapshots; no CUDA or image downloads."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_cifar_six_interactive as runner
from tests.test_cifar_six_pipeline import synthetic_run
from tests.test_cifar_six_630_integration import task_statuses

original = runner.original
OURS = "sm9rrs-v10-014"
BASELINES = {"vert": "vert-v10-015", "alignins": "alignins-v10-008",
             "krum": "krum-v10-001", "ding13": "ding13-v10-001", "fedavg": "fedavg-v10-001"}


class InteractiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = original.load_spec(original.DEFAULT_CONFIG)
        for method, candidates in cls.spec["candidates"].items():
            wanted = {OURS, "sm9rrs-v10-015"} if method == "sm9rrs" else {BASELINES[method]}
            cls.spec["candidates"][method] = [c for c in candidates if c["candidate_id"] in wanted]
        cls.spec["fallback_candidates"] = BASELINES
        cls.spec["run_budget"] = original.run_budget(cls.spec)
        cls.contract = {"synthetic_test_contract": "no dataset access"}
        cls.manifest = original.build_manifest(cls.spec, cls.contract, original.REPO)
        cls.tasks = original.attach_fingerprints(original.build_tasks(cls.spec, "validation"), cls.manifest)
        cls.groups = {}
        for task in cls.tasks:
            cid = task["candidate"]["candidate_id"]
            config = original.fl.ExperimentConfig(**task["config"])
            asr = .20 if cid == OURS else .30 if config.method == "sm9rrs" else .05
            if config.method == "fedavg" and config.malicious_ratio:
                asr = .80
            result = synthetic_run(config, accuracy=.8, asr=asr,
                                   nonfinite=1 if config.method == "vert" else 0)
            cls.groups.setdefault(cid, []).append(result)
        cls.report = original.json_safe(original.select_validation(cls.spec, cls.groups, cls.tasks))
        cls.report["tasks"] = task_statuses(cls.tasks, cls.groups)
        assert cls.report["status"] == "needs_ours_target_development"
        assert cls.report["best_healthy_score_candidate"] == OURS

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.output = root / "study"
        self.output.mkdir()
        self.config = root / "config.json"
        self.config.write_text(json.dumps(self.spec))
        original.write_json(self.output / "manifest.json", self.manifest)
        original.write_json(self.output / "validation_plan.json", {
            "manifest_fingerprint": self.manifest["fingerprint"], "tasks": self.tasks})
        original.write_json(self.output / "validation_summary.json", self.report)
        self.args = SimpleNamespace(config=self.config, output=self.output, data_dir=None,
                                    devices=["cuda:0"], phase="all")
        self.written = []

    def invoke(self, answer="N\n", *, groups=None, transform=None, execute_override=None):
        groups = self.groups if groups is None else groups
        original_collect = original.collect_results
        calls = []

        def collect(output, tasks):
            if tasks[0]["phase"] == "validation":
                statuses = task_statuses(tasks, groups)
                if transform:
                    transform(statuses)
                return groups, statuses
            return original_collect(output, tasks)

        def execute(args, tasks, output):
            calls.append(tasks)
            self.assertEqual(tasks[0]["phase"], "final", "completed validation must never retrain")
            if execute_override:
                return execute_override(tasks)
            for task in tasks:
                if original.checked_completed(output, task) is not None:
                    continue
                folder = output / "tasks" / task["task_id"]
                folder.mkdir(parents=True, exist_ok=True)
                original.write_json(folder / "task.json", task)
                config = original.fl.ExperimentConfig(**task["config"])
                original.experiments._write_completed_results_snapshot(folder, [synthetic_run(config)])
                self.written.append(task["task_id"])
            return 0

        with mock.patch.object(original, "collect_results", side_effect=collect), \
                mock.patch.object(original, "load_split", return_value=(object(), self.contract)) as data, \
                mock.patch.object(original, "execute_phase", side_effect=execute), \
                mock.patch("sys.stdin", io.StringIO(answer)), redirect_stdout(io.StringIO()) as text:
            code = runner.run_parent(self.args)
        return code, calls, text.getvalue(), data.call_count

    def test_n_eof_and_validation_only_do_not_start_final_or_load_data(self):
        before = {p.name: p.read_bytes() for p in self.output.glob("*.json")}
        for answer in ("N\n", ""):
            code, calls, text, data_calls = self.invoke(answer)
            self.assertEqual(code, 0)
            self.assertEqual(calls, [])
            self.assertEqual(data_calls, 0)
            self.assertIn("CONTINUATION_PROMPT", text)
            self.assertFalse((self.output / "final_plan.json").exists())
            self.assertFalse((self.output / runner.DECISION_FILE).exists())
        self.args.phase = "validation"
        self.assertNotIn("CONTINUATION_PROMPT", self.invoke("Y\n")[2])
        for name, content in before.items():
            self.assertEqual((self.output / name).read_bytes(), content)

    def test_y_runs_180_and_every_resume_asks_without_retraining(self):
        before = {p.name: p.read_bytes() for p in self.output.glob("*.json")}
        code, calls, text, data_calls = self.invoke("what\ny\n")
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 180)
        self.assertEqual(data_calls, 1)
        self.assertEqual(text.count("CONTINUATION_PROMPT"), 2)
        self.assertEqual(len(self.written), 180)
        plan = runner.read_json(self.output / "final_plan.json")
        self.assertEqual(plan, original.final_plan(self.spec, self.manifest, {**BASELINES, "sm9rrs": OURS}))
        self.assertEqual({t["config"]["seed"] for t in plan["tasks"]}, {1101, 1102, 1103})
        self.assertEqual({t["config"]["malicious_ratio"] for t in plan["tasks"]}, {0., .2, .4, .6, .8})
        self.assertTrue(all(t["config"]["checkpoint_interval"] == 1 for t in plan["tasks"]))
        decision = (self.output / runner.DECISION_FILE).read_bytes()
        summary = (self.output / "final_summary.json").read_bytes()
        for answer in ("N\n", "Y\n"):
            code, calls, text, data_calls = self.invoke(answer)
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 1 if answer == "Y\n" else 0)
            self.assertEqual(len(self.written), 180, "no completed task may be trained again")
            self.assertEqual(data_calls, 0)
            self.assertIn("CONTINUATION_PROMPT", text)
            self.assertEqual((self.output / runner.DECISION_FILE).read_bytes(), decision)
            self.assertEqual((self.output / "final_summary.json").read_bytes(), summary)
        final = runner.read_json(self.output / "final_summary.json")
        self.assertEqual(final["validation_final_metric_gate"]["status"], "unmet")
        self.assertFalse(final["formal_results_used_for_selection"])
        self.assertEqual(final["continuation_decision"]["selected"], plan["selected"])
        self.assertTrue(final["methods"]["sm9rrs"]["validation_health_qualified"])
        self.assertFalse(final["methods"]["sm9rrs"]["validation_target_passed"])
        self.assertEqual(len(list((self.output / "continuation_responses").glob("*.json"))), 3)
        for name, content in before.items():
            self.assertEqual((self.output / name).read_bytes(), content)

    def test_interrupted_after_decision_recovers_but_still_requires_y(self):
        real_immutable = original.immutable_json

        def fail_plan(path, payload):
            if Path(path).name == "final_plan.json":
                raise KeyboardInterrupt()
            return real_immutable(path, payload)

        with mock.patch.object(original, "immutable_json", side_effect=fail_plan):
            with self.assertRaises(KeyboardInterrupt):
                self.invoke("Y\n")
        self.assertTrue((self.output / runner.DECISION_FILE).exists())
        self.assertFalse((self.output / "final_plan.json").exists())
        self.invoke("N\n")
        self.assertFalse((self.output / "final_plan.json").exists())
        self.invoke("Y\n")
        self.assertEqual(len(self.written), 180)

    def test_partial_final_resumes_same_plan_and_preserves_checkpoint_directory(self):
        def interrupt(tasks):
            folder = self.output / "tasks" / tasks[0]["task_id"] / "checkpoints"
            folder.mkdir(parents=True)
            (folder / "sentinel").write_text("saved round")
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.invoke("Y\n", execute_override=interrupt)
        plan = (self.output / "final_plan.json").read_bytes()
        self.invoke("N\n")
        # The existing checkpoint directory must survive the fake worker too.
        self.invoke("Y\n")
        self.assertEqual((self.output / "final_plan.json").read_bytes(), plan)
        self.assertEqual(next((self.output / "tasks").rglob("sentinel")).read_text(), "saved round")

    def test_qualified_validation_auto_promotes_without_prompt(self):
        groups = deepcopy(self.groups)
        for cid in (OURS, "sm9rrs-v10-015"):
            groups[cid] = [synthetic_run(run.config, accuracy=.8, asr=.05) for run in groups[cid]]
        code, calls, text, _ = self.invoke(groups=groups)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls[0]), 180)
        self.assertNotIn("CONTINUATION_PROMPT", text)
        self.assertFalse((self.output / runner.DECISION_FILE).exists())

    def test_fresh_output_keeps_validation_then_automatic_final_protocol(self):
        self.args.output = self.output.parent / "fresh"
        groups = deepcopy(self.groups)
        for cid in (OURS, "sm9rrs-v10-015"):
            groups[cid] = [synthetic_run(run.config, accuracy=.8, asr=.05) for run in groups[cid]]
        executed = []

        def execute(args, tasks, destination):
            executed.append(tasks[0]["phase"])
            return 0

        def collect(destination, tasks):
            phase = tasks[0]["phase"]
            if phase not in executed:
                return {}, task_statuses(tasks, {})
            if phase == "validation":
                return groups, task_statuses(tasks, groups)
            final_groups = {}
            for task in tasks:
                config = original.fl.ExperimentConfig(**task["config"])
                final_groups.setdefault(task["candidate"]["candidate_id"], []).append(synthetic_run(config))
            return final_groups, task_statuses(tasks, final_groups)

        with mock.patch.object(original, "load_split", return_value=(object(), self.contract)), \
                mock.patch.object(original, "execute_phase", side_effect=execute), \
                mock.patch.object(original, "collect_results", side_effect=collect), \
                mock.patch.object(original.experiments, "write_result_files"), \
                mock.patch.object(runner, "ask_continuation", side_effect=AssertionError("no prompt when qualified")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_parent(self.args), 0)
        self.assertEqual(executed, ["validation", "final"])
        self.assertEqual(runner.read_json(self.args.output / "manifest.json"), self.manifest)
        final = runner.read_json(self.args.output / "final_plan.json")
        self.assertEqual(len(final["tasks"]), 180)
        self.assertTrue(runner.read_json(self.args.output / "final_summary.json")["full_execution_completed"])
        self.assertFalse((self.args.output / runner.DECISION_FILE).exists())

    def test_resume_repairs_artifacts_after_completed_snapshot_before_csv_commit(self):
        self.invoke("Y\n")
        plan = runner.read_json(self.output / "final_plan.json")
        first_folder = self.output / "tasks" / plan["tasks"][0]["task_id"]
        self.assertTrue((first_folder / original.experiments.COMPLETED_RESULTS_SNAPSHOT).exists())
        self.assertFalse((first_folder / "rounds.csv").exists())
        execute = original.execute_phase
        self.invoke("Y\n", execute_override=lambda tasks: execute(self.args, tasks, self.output))
        self.assertTrue((first_folder / "rounds.csv").exists())
        self.assertTrue((first_folder / "metrics.json").exists())
        self.assertEqual(len(self.written), 180)

    def test_no_healthy_ours_cannot_be_overridden(self):
        groups = deepcopy(self.groups)
        for cid in (OURS, "sm9rrs-v10-015"):
            groups[cid] = [synthetic_run(run.config, nonfinite=1) for run in groups[cid]]
        code, calls, text, _ = self.invoke("Y\n", groups=groups)
        self.assertEqual(code, 0)
        self.assertFalse(calls)
        self.assertNotIn("CONTINUATION_PROMPT", text)
        self.assertFalse((self.output / "final_plan.json").exists())

    def test_incomplete_final_only_evidence_is_not_a_performance_failure(self):
        self.args.phase = "final"
        before = (self.output / "validation_summary.json").read_bytes()
        def damage(rows):
            rows[0].update(status="pending", metrics=None)
        code, calls, text, _ = self.invoke("Y\n", transform=damage)
        self.assertEqual(code, 2)
        self.assertFalse(calls)
        self.assertNotIn("CONTINUATION_PROMPT", text)
        self.assertEqual((self.output / "validation_summary.json").read_bytes(), before)

    def test_plan_without_decision_and_changed_decision_or_source_are_refused(self):
        original.write_json(self.output / "final_plan.json", original.final_plan(
            self.spec, self.manifest, {**BASELINES, "sm9rrs": OURS}))
        with self.assertRaisesRegex(ValueError, "without its user decision"):
            self.invoke("Y\n")
        (self.output / "final_plan.json").unlink()
        self.invoke("Y\n")
        decision = runner.read_json(self.output / runner.DECISION_FILE)
        decision["ours_candidate"] = "sm9rrs-v10-015"
        original.write_json(self.output / runner.DECISION_FILE, decision)
        with self.assertRaisesRegex(ValueError, "continuation differs"):
            self.invoke("Y\n")
        manifest = deepcopy(self.manifest)
        manifest["source_sha256"]["run_cifar_six_relative_best.py"] = "changed"
        original.write_json(self.output / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "immutable experiment identity changed"):
            self.invoke("Y\n")


if __name__ == "__main__":
    unittest.main()
