"""Read-only evidence audit and report-only recovery from synthetic v8 studies."""
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import csv
import fcntl
import hashlib
import importlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_cifar_six_from_scratch as base
from tests.test_cifar_adaptive_runner import SyntheticStudy
from tests.test_cifar_six_pipeline import synthetic_run


def tree_identity(root):
    return {str(path.relative_to(root)): (hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns) for path in root.rglob("*") if path.is_file()}


def make_study(output, flavor, *, completed_waves=4):
    runner = importlib.import_module("run_" + flavor + "_adaptive")
    runtime = runner.runtime
    reporting = importlib.import_module(flavor + "_adaptive_reporting")
    output.mkdir()
    (output / ".runner.lock").touch()
    spec = runner.load_spec(runner.DEFAULT_CONFIG)
    spec["search"]["max_public_trials"] = 1
    spec["search"]["max_gpus"] = 2
    study = SyntheticStudy(output, spec, unqualified=flavor == "cifar")
    if completed_waves < 4:
        study.pause_after = completed_waves
    study.manifest = runtime.build_manifest(spec, {"synthetic": "report recovery fixture"})
    base.write_json(output / "manifest.json", study.manifest)
    state = runner.new_state(spec, study.manifest)
    with mock.patch.object(runtime, "execute", side_effect=study.execute), \
            mock.patch.object(runtime, "collect", side_effect=study.collect), \
            mock.patch.object(reporting, "write_final_report"), \
            mock.patch("sys.stdin", io.StringIO("Y\n")), redirect_stdout(io.StringIO()):
        runner.run_search(study.args, spec, study.manifest, state)
        if completed_waves < 4:
            assert state["status"] == "execution_blocked"
            state["status"] = "budget_exhausted"
        runner.prepare_selection(output, spec, study.manifest, state)
        base.write_json(output / "search_state.json", state)
        choice = runner.choose_block(state)
        study.pause_after = None
        runner.run_final(study.args, spec, study.manifest, state, choice)
    current = runner.block_spec(spec, runner.selection_view(state["blocks"][0]))
    validation_tasks = runtime.attach_tasks(current, "validation", study.manifest)
    formal_tasks = runtime.read_json(output / "final_plan.json")["tasks"]
    assert len(validation_tasks) == (3 * completed_waves + 3) * 20 and len(formal_tasks) == 60
    for task in validation_tasks + formal_tasks:
        folder = runtime.ensure_identity(output, task)
        base.experiments._write_completed_results_snapshot(folder, [study.results[task["task_id"]]])
        # Derived per-task CSV files intentionally do not exist. Recovery must
        # read the authoritative snapshot without calling repair_completed.
    if completed_waves < 4:
        # A partially completed later wave contains tempting perfect Ours
        # evidence. Its unmatched VERT/AlignIns budget must exclude it.
        pending_id = next(item["candidate"]["candidate_id"]
                          for item in state["blocks"][0]["waves"][-1]["candidates"]
                          if item["method"] == "sm9rrs")
        all_tasks = runtime.attach_tasks(runner.block_spec(spec, state["blocks"][0]),
                                         "validation", study.manifest)
        for task in all_tasks:
            if task["candidate"]["candidate_id"] == pending_id:
                folder = runtime.ensure_identity(output, task)
                perfect = synthetic_run(base.fl.ExperimentConfig(**task["config"]), accuracy=1., asr=0.)
                base.experiments._write_completed_results_snapshot(folder, [perfect])
    first = formal_tasks[0]
    folder = output / "tasks" / first["task_id"]
    config = base.fl.ExperimentConfig(**first["config"])
    base.experiments._write_round_checkpoint(
        base.experiments._checkpoint_path(folder / "checkpoints", config),
        config, first["fingerprint"],
        {"completed_round": 150, "terminal_result": study.results[first["task_id"]]},
        runtime_seconds=1., peak_memory_mb=1.)
    base.write_json(folder / "progress.json", {"last_completed_round": 150})
    return runner, runtime, reporting


class AdaptiveReportRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.templates = Path(cls.temporary.name).resolve()
        cls.modules = {flavor: make_study(cls.templates / flavor, flavor)
                       for flavor in ("cifar", "fashion")}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.scratch = Path(temporary.name).resolve()
        self.select("cifar")

    def select(self, flavor):
        self.output = self.scratch / flavor
        shutil.copytree(self.templates / flavor, self.output)
        self.runner, self.runtime, self.reporting = self.modules[flavor]
        self.tasks = self.runtime.read_json(self.output / "final_plan.json")["tasks"]

    def recover(self, *, write=False, plots=False):
        recovery = importlib.import_module("recover_adaptive_report")
        with ExitStack() as stack:
            for flavor in ("cifar", "fashion"):
                runner, runtime, reporting = self.modules[flavor]
                stack.enter_context(mock.patch.object(runtime, "execute", side_effect=AssertionError("no execution")))
                stack.enter_context(mock.patch.object(runtime, "worker", side_effect=AssertionError("no worker")))
                stack.enter_context(mock.patch.object(runtime, "collect", side_effect=AssertionError("mutating collect forbidden")))
                stack.enter_context(mock.patch.object(runner, "ask_override", side_effect=AssertionError("no new approval")))
                if flavor == "fashion":
                    stack.enter_context(mock.patch.object(runner, "load_split", side_effect=AssertionError("no data load")))
                if not plots:
                    stack.enter_context(mock.patch.object(reporting, "_plots", return_value=[]))
            stack.enter_context(mock.patch.object(base, "load_split", side_effect=AssertionError("no data load")))
            stack.enter_context(mock.patch.object(base, "worker_environment", side_effect=AssertionError("no GPU")))
            stack.enter_context(mock.patch.object(base, "repair_completed", side_effect=AssertionError("no repair")))
            stack.enter_context(mock.patch.object(base.experiments, "run_measured_experiment", side_effect=AssertionError("no training")))
            stack.enter_context(mock.patch("builtins.input", side_effect=AssertionError("no input")))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return recovery.recover(self.output, write=write)

    def assert_refused_without_writes(self):
        before = tree_identity(self.output)
        with self.assertRaises((ValueError, RuntimeError)):
            self.recover(write=True)
        self.assertEqual(tree_identity(self.output), before)

    def update_json(self, relative, change):
        path = self.output / relative
        payload = json.loads(path.read_text())
        change(payload)
        base.write_json(path, payload)

    def change_result(self, task, change):
        folder = self.output / "tasks" / task["task_id"]
        result = base.experiments.load_completed_results_snapshot(folder)[0]
        base.experiments._write_completed_results_snapshot(folder, [change(result)])

    def test_default_cifar_and_fashion_audits_are_entirely_read_only(self):
        for flavor in ("cifar", "fashion"):
            with self.subTest(flavor=flavor):
                if flavor == "fashion":
                    self.select(flavor)
                before = tree_identity(self.output)
                directories = {str(p.relative_to(self.output)) for p in self.output.rglob("*") if p.is_dir()}
                result = self.recover()
                self.assertIsInstance(result, dict)
                self.assertEqual(result["status"], "ready")
                self.assertEqual(result["expected_tasks"], 60)
                self.assertEqual(result["completed_tasks"], 60)
                self.assertFalse(result["training_started"])
                self.assertEqual(tree_identity(self.output), before)
                self.assertEqual({str(p.relative_to(self.output)) for p in self.output.rglob("*") if p.is_dir()}, directories)
                self.assertFalse((self.output / "final_results").exists())

    def test_write_makes_real_figures_and_preserves_every_source_file(self):
        before = tree_identity(self.output)
        result = self.recover(write=True, plots=True)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "report_recovered")
        self.assertFalse(result["training_started"])
        after = tree_identity(self.output)
        self.assertEqual({name: after[name] for name in before}, before)
        created = set(after) - set(before)
        self.assertTrue(created)
        self.assertTrue(all(name == "final_summary.json" or name.startswith("final_results/") for name in created))
        destination = self.output / "final_results"
        self.assertEqual(len(list(destination.glob("*.svg"))), 4)
        self.assertEqual(len(list(destination.glob("*.png"))), 4)
        self.assertTrue((destination / "visualizations.html").is_file())
        summary = self.runtime.read_json(self.output / "final_summary.json")
        self.assertEqual(summary["completed_tasks"], 60)
        self.assertTrue(summary["full_execution_completed"])
        self.assertFalse(summary["parameters_reselected"])
        self.assertEqual(summary["formal_seeds"], [2026093011])
        self.assertEqual(len(list(self.output.glob("tasks/*/checkpoints/*.pickle"))), 1)
        self.assertFalse(any(self.output.glob("tasks/*/rounds.csv")))

    def test_fashion_write_keeps_fashion_metadata(self):
        self.select("fashion")
        before = tree_identity(self.output)
        self.recover(write=True)
        summary = self.runtime.read_json(self.output / "final_summary.json")
        self.assertEqual(summary["model"]["input_shape"], [1, 28, 28])
        self.assertEqual(summary["formal_seeds"], [2026100111])
        html = (self.output / "final_results/visualizations.html").read_text()
        self.assertIn("Fashion-MNIST", html)
        self.assertNotIn("CIFAR-10", html)
        after = tree_identity(self.output)
        self.assertEqual({name: after[name] for name in before}, before)

    def test_existing_reports_are_backed_up_after_success_without_touching_evidence(self):
        old_reports = self.output / "final_results"
        old_reports.mkdir()
        (old_reports / "previous.txt").write_text("preserve old derived report")
        (self.output / "final_summary.json").write_text('{"previous_report": true}\n')
        before = tree_identity(self.output)
        self.recover(write=True)
        backups = list((self.output / "report_recovery_backups").iterdir())
        self.assertEqual(len(backups), 1)
        saved_audit = self.runtime.read_json(self.output / "final_results/report_recovery.json")
        self.assertEqual(Path(saved_audit["backup"]), backups[0])
        self.assertEqual(Path(saved_audit["report"]), self.output / "final_results/visualizations.html")
        self.assertEqual((backups[0] / "final_results/previous.txt").read_text(), "preserve old derived report")
        self.assertEqual((backups[0] / "final_summary.json").read_text(), '{"previous_report": true}\n')
        after = tree_identity(self.output)
        evidence = {name: value for name, value in before.items()
                    if name != "final_summary.json" and not name.startswith("final_results/")}
        self.assertEqual({name: after[name] for name in evidence}, evidence)
        self.assertEqual(self.runtime.read_json(self.output / "final_summary.json")["completed_tasks"], 60)

    def test_failed_report_generation_keeps_existing_reports_and_evidence(self):
        (self.output / "final_results").mkdir()
        (self.output / "final_results/previous.txt").write_text("original derived content")
        (self.output / "final_summary.json").write_text('{"previous_report": true}\n')
        before = tree_identity(self.output)
        with mock.patch.object(self.reporting, "write_final_report", side_effect=ValueError("synthetic report failure")):
            with self.assertRaises((ValueError, RuntimeError)):
                self.recover(write=True)
        self.assertEqual(tree_identity(self.output), before)

    def test_publish_failure_rolls_back_previous_reports(self):
        (self.output / "final_results").mkdir()
        (self.output / "final_results/previous.txt").write_text("original report")
        (self.output / "final_summary.json").write_text('{"previous_report": true}\n')
        before = tree_identity(self.output)
        original_rename = Path.rename
        def fail_summary_publish(source, target):
            if (source.name == "final_summary.json"
                    and source.parent.name.startswith(".adaptive-report-recovery-")
                    and Path(target) == self.output / "final_summary.json"):
                raise OSError("synthetic publish failure")
            return original_rename(source, target)
        with mock.patch.object(Path, "rename", fail_summary_publish):
            with self.assertRaisesRegex(OSError, "synthetic publish failure"):
                self.recover(write=True)
        self.assertEqual(tree_identity(self.output), before)

    def test_complete_health_failures_are_preserved_in_formal_statistics(self):
        task = next(t for t in self.tasks if t["method"] == "vert")
        self.change_result(task, lambda r: replace(r, nonfinite_updates=1))
        before = tree_identity(self.output)
        audit = self.recover(write=True)
        self.assertEqual(audit["completed_tasks"], 60)
        self.assertEqual(audit["healthy_tasks"], 59)
        summary = self.runtime.read_json(self.output / "final_summary.json")
        self.assertTrue(summary["full_execution_completed"])
        self.assertEqual(summary["status"], "completed_with_health_failures")
        self.assertIsNotNone(summary["methods"]["vert"]["overall"])
        self.assertEqual(summary["methods"]["vert"]["healthy_runs"], 9)
        after = tree_identity(self.output)
        self.assertEqual({name: after[name] for name in before}, before)

    def test_budget_prefix_recovers_without_completing_or_selecting_pending_wave(self):
        self.output = self.scratch / "budget-prefix"
        self.runner, self.runtime, self.reporting = make_study(self.output, "cifar", completed_waves=2)
        state = self.runtime.read_json(self.output / "search_state.json")
        block = state["blocks"][0]
        self.assertEqual(state["status"], "budget_exhausted")
        self.assertEqual(block["budget_selection_snapshot"]["completed_waves"], 2)
        self.assertEqual(block["waves"][2]["status"], "pending")
        pending_ids = {item["candidate"]["candidate_id"] for item in block["waves"][2]["candidates"]}
        self.assertEqual(len(list(self.output.glob("tasks/validation_*d002*/.completed_results.pickle"))), 20)
        before = tree_identity(self.output)
        audit = self.recover(write=True)
        self.assertEqual(audit["completed_tasks"], 60)
        self.assertFalse(audit["validation_target_passed"])
        self.assertTrue(set(audit["selected"].values()).isdisjoint(pending_ids))
        self.assertTrue(audit["selected"]["sm9rrs"].endswith("-d001"))
        after = tree_identity(self.output)
        self.assertEqual({name: after[name] for name in before}, before)
        summary = self.runtime.read_json(self.output / "final_summary.json")
        self.assertEqual(summary["completed_tasks"], 60)
        self.assertTrue(all(row["candidate_id"] not in pending_ids for row in summary["tasks"]))

    def test_active_runner_lock_refuses_recovery_without_writing(self):
        with (self.output / ".runner.lock").open("rb") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                self.assert_refused_without_writes()
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def test_missing_formal_snapshot_refuses_write(self):
        (self.output / "tasks" / self.tasks[0]["task_id"] / base.experiments.COMPLETED_RESULTS_SNAPSHOT).unlink()
        self.assert_refused_without_writes()

    def test_corrupt_formal_snapshot_refuses_write(self):
        (self.output / "tasks" / self.tasks[0]["task_id"] / base.experiments.COMPLETED_RESULTS_SNAPSHOT).write_bytes(b"broken pickle")
        self.assert_refused_without_writes()

    def test_incomplete_final_round_never_substitutes_previous_round(self):
        self.change_result(self.tasks[0], lambda r: replace(r, stopped_round=149, records=r.records[:-1]))
        self.assert_refused_without_writes()

    def test_task_identity_mismatch_refuses_write(self):
        task = self.tasks[0]
        self.update_json("tasks/" + task["task_id"] + "/task.json", lambda p: p["config"].update(seed=99))
        self.assert_refused_without_writes()

    def test_manifest_source_mismatch_refuses_write(self):
        self.update_json("manifest.json", lambda p: p["source_sha256"].update({"cifar_adaptive_gate.py": "0" * 64}))
        self.assert_refused_without_writes()

    def test_search_fingerprint_mismatch_refuses_write(self):
        self.update_json("search_state.json", lambda p: p.update(manifest_fingerprint="foreign-study"))
        self.assert_refused_without_writes()

    def test_original_controller_summary_is_accepted_but_changed_budget_or_choice_is_not(self):
        # Generate the summary with the frozen original controller, without
        # running its final phase or repairing any per-task CSV/checkpoint.
        def read_only_collect(where, tasks):
            groups, statuses = {}, []
            for task in tasks:
                run = base.checked_completed(where, task)
                self.assertIsNotNone(run)
                cid = task["candidate"]["candidate_id"]
                groups.setdefault(cid, []).append(run)
                statuses.append({"task_id": task["task_id"], "method": task["method"],
                                 "candidate_id": cid, "status": "complete",
                                 "healthy": base.metrics(run)["healthy"]})
            return groups, statuses
        spec = self.runtime.read_json(self.output / "manifest.json")["spec"]
        args = SimpleNamespace(output=self.output, data_dir=None, devices=["cuda:0"], phase="all")
        with mock.patch.object(self.runtime, "collect", side_effect=read_only_collect), \
                mock.patch.object(self.runner, "run_final", return_value=0), \
                mock.patch.object(base, "load_split", side_effect=AssertionError("no data")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(self.runner.run_parent(args, spec), 0)
        self.assertEqual(self.recover()["status"], "ready")
        path = self.output / "search_summary.json"
        original = path.read_bytes()
        for field in ("elapsed_search_seconds", "choice", "public_tpe_proposals"):
            with self.subTest(field=field):
                path.write_bytes(original)
                summary = json.loads(original)
                if field == "choice":
                    summary["choice"]["ours_candidate"] = "other-candidate"
                else:
                    summary[field] += 1
                base.write_json(path, summary)
                self.assert_refused_without_writes()

    def test_frozen_choice_or_search_digest_mismatch_refuses_write(self):
        original = (self.output / "continuation_decision.json").read_bytes()
        for field in ("choice", "source_search_digest"):
            with self.subTest(field=field):
                (self.output / "continuation_decision.json").write_bytes(original)
                if field == "choice":
                    self.update_json("continuation_decision.json", lambda p: p["choice"].update(raw_score=42.))
                else:
                    self.update_json("continuation_decision.json", lambda p: p.update(source_search_digest="0" * 64))
                self.assert_refused_without_writes()

    def test_both_formal_plans_must_exactly_match_reconstructed_tasks(self):
        for relative in ("final_plan.json", "task_plans/formal.json"):
            with self.subTest(relative=relative):
                original = (self.output / relative).read_bytes()
                self.update_json(relative, lambda p: p["tasks"][0]["config"].update(lr=.001))
                self.assert_refused_without_writes()
                (self.output / relative).write_bytes(original)

    def test_changed_validation_result_cannot_reuse_saved_selection_report(self):
        state = self.runtime.read_json(self.output / "search_state.json")
        manifest = self.runtime.read_json(self.output / "manifest.json")
        current = self.runner.block_spec(manifest["spec"], self.runner.selection_view(state["blocks"][0]))
        task = self.runtime.attach_tasks(current, "validation", manifest)[0]
        self.change_result(task, lambda r: replace(r, final_accuracy=.2,
            records=[replace(row, accuracy=.2) if row.round == 150 else row for row in r.records]))
        self.assert_refused_without_writes()

    def test_manual_selection_requires_matching_historical_yes(self):
        response = next((self.output / "continuation_responses").glob("*.json"))
        original = response.read_bytes()
        for mismatch in ("manifest", "choice", "answer", "missing"):
            with self.subTest(mismatch=mismatch):
                response.write_bytes(original)
                if mismatch == "missing":
                    response.unlink()
                else:
                    payload = json.loads(original)
                    if mismatch == "manifest":
                        payload["manifest_fingerprint"] = "foreign-study"
                    elif mismatch == "choice":
                        payload["choice"]["ours_candidate"] = "other-candidate"
                    else:
                        payload.update(response="N", approved=False)
                    base.write_json(response, payload)
                self.assert_refused_without_writes()

    def test_later_no_does_not_erase_past_yes_for_already_completed_results(self):
        response = next((self.output / "continuation_responses").glob("*.json"))
        declined = json.loads(response.read_text())
        declined.update(response="N", approved=False)
        base.write_json(response.with_name("99999999T999999_999999.json"), declined)
        before = tree_identity(self.output)
        self.recover()
        self.assertEqual(tree_identity(self.output), before)

    def test_small_aggregation_roundoff_keeps_raw_values_and_restores_validator(self):
        task = next(t for t in self.tasks if t["method"] == "fedavg" and t["config"]["malicious_ratio"] > 0)
        value = 1.0000000000000002
        self.change_result(task, lambda r: replace(r, records=[replace(row,
            honest_weight_loss=value, malicious_weight_mass=value) if row.round == 150 else row for row in r.records]))
        before = tree_identity(self.output)
        original_inspect = self.reporting._inspect_result
        audit = self.recover(write=True)
        self.assertEqual(audit["aggregation_weight_upper_tolerance"], 1e-9)
        self.assertTrue(audit["roundoff_accepted"])
        self.assertIs(self.reporting._inspect_result, original_inspect)
        after = tree_identity(self.output)
        self.assertEqual({name: after[name] for name in before}, before)
        with (self.output / "final_results/rounds.csv").open(newline="") as handle:
            row = next(row for row in csv.DictReader(handle)
                       if row["task_id"] == task["task_id"] and row["round"] == "150")
        self.assertEqual(row["honest_weight_loss"], repr(value))
        self.assertEqual(row["malicious_weight_mass"], repr(value))

    def test_probability_errors_are_not_hidden_by_aggregation_tolerance(self):
        task = next(t for t in self.tasks if t["config"]["malicious_ratio"] > 0)
        path = self.output / "tasks" / task["task_id"] / base.experiments.COMPLETED_RESULTS_SNAPSHOT
        original = path.read_bytes()
        for field, value in (("honest_weight_loss", 1.000000002),
                             ("malicious_weight_mass", -.0000000001),
                             ("accuracy", 1.0000000001),
                             ("attack_target_success_rate", 1.0000000001),
                             ("attack_target_confidence", float("nan"))):
            with self.subTest(field=field):
                path.write_bytes(original)
                self.change_result(task, lambda r: replace(r, records=[replace(row, **{field: value})
                    if row.round == 149 else row for row in r.records]))
                self.assert_refused_without_writes()


if __name__ == "__main__":
    unittest.main()
