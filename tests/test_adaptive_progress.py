"""Small metadata fixtures: no model, CUDA, pickle reads, or real training."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

from adaptive_progress import AdaptiveProgress


class AdaptiveProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.spec = {"schema_version": 8, "protocol": "fashion-mnist-resnet18-gn-tpe-v1"}
        self.manifest = {"schema_version": 8, "spec": self.spec, "fingerprint": "fashion-study"}
        self.write("manifest.json", self.manifest)
        self.clock = [0.]
        self.view = AdaptiveProgress(self.output, self.spec, now=lambda: self.clock[0])

    def write(self, name, value):
        path = self.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def task(self, index=0, phase="final", wave=0):
        cid = f"fedavg-b000-d{wave:03d}"
        tid = f"{phase}_{cid}_iid_ratio0_seed{index}"
        return {"task_id": tid, "fingerprint": "fingerprint-" + tid, "phase": phase,
                "method": "fedavg", "candidate": {"candidate_id": cid},
                "config": {"rounds": 150, "seed": index, "method": "fedavg"}}

    def plan(self, tasks, name="final_plan.json"):
        self.write(name, {"manifest_fingerprint": self.manifest["fingerprint"], "tasks": tasks})
        for task in tasks:
            self.write(f"tasks/{task['task_id']}/task.json", task)
        self.view.refresh()

    def progress(self, task, rd, **overrides):
        value = {"task_id": task["task_id"], "study_phase": task["phase"],
                 "candidate_id": task["candidate"]["candidate_id"], "round": rd,
                 "last_completed_round": rd, **overrides}
        self.write(f"tasks/{task['task_id']}/progress.json", value)

    def snapshot(self, task, *, healthy=True):
        folder = self.output / "tasks" / task["task_id"]
        (folder / ".completed_results.pickle").write_bytes(b"intentionally not pickle")
        self.write(f"tasks/{task['task_id']}/metrics.json", {"stopped_round": 150,
            "final_accuracy": .8, "healthy": healthy,
            "reasons": [] if healthy else ["clean_false_revocations"]})

    def failure(self, task, kind="algorithm_numerical", **overrides):
        self.write(f"tasks/{task['task_id']}/failure.json", {"task_id": task["task_id"],
            "task_fingerprint": task["fingerprint"], "kind": kind, **overrides})

    def test_formal_cache_rounds_and_health_are_separate(self):
        tasks = [self.task(i) for i in range(60)]
        self.plan(tasks)
        for task in tasks:
            self.snapshot(task, healthy=False)
        with mock.patch.object(pickle, "load", side_effect=AssertionError("display must never unpickle")):
            self.view.refresh()
            self.assertEqual(len(self.view.completed), 0)
            self.assertIn("unverified=60", " ".join(self.view.lines()))
            for task in tasks:
                self.view.consume("REUSE " + task["task_id"])
            self.view.finish(0)
        self.assertEqual(len(self.view.completed), 60)
        self.assertIn("saved_rounds=9000/9000", self.view.lines()[0])
        self.assertIn("done=60/60", self.view.lines()[1])
        self.assertFalse(any(t["healthy"] for t in self.view.tasks.values()))
        self.assertIn("not a health verdict", self.view.status)

    def test_progress_identity_bounds_non_decreasing_and_bad_round(self):
        task = self.task()
        self.plan([task])
        for rd, changes in ((9, {"task_id": "foreign"}), (15, {"candidate_id": "wrong"}),
                            (29, {"study_phase": "validation"}), (151, {}), (-1, {}), (True, {})):
            self.progress(task, rd, **changes)
            self.view.refresh()
            self.assertEqual(self.view.tasks[task["task_id"]]["round"], 0)
        self.progress(task, 29, round=30)
        self.failure(task)
        self.view.refresh()
        self.assertEqual(self.view.tasks[task["task_id"]]["round"], 29)
        self.assertEqual(len(self.view.failed), 1)
        self.progress(task, 25)
        self.view.refresh()
        self.assertEqual(self.view.tasks[task["task_id"]]["round"], 29)

    def test_start_retry_clear_historical_failure_and_success_precedes_it(self):
        task = self.task()
        self.plan([task])
        self.failure(task, kind="infrastructure_or_execution")
        self.view.refresh()
        self.assertFalse(self.view.failed)  # the controller will retry it
        self.failure(task)
        self.view.refresh()
        self.assertEqual(len(self.view.failed), 1)
        self.view.consume(f"START {task['task_id']} device=cuda:0")
        self.assertFalse(self.view.failed)
        self.assertEqual(self.view.tasks[task["task_id"]]["state"], "running")
        self.view.consume(f"RETRY_SAME_CONFIG {task['task_id']}")
        self.view.refresh()
        self.assertEqual(self.view.tasks[task["task_id"]]["state"], "queued")
        self.snapshot(task)
        self.view.consume(f"WORKER_EXIT {task['task_id']} code=0 kind=algorithm_numerical")
        self.assertEqual(self.view.completed, {task["task_id"]})
        self.assertFalse(self.view.failed)
        self.assertFalse(self.view.lanes)

    def test_exit_zero_requires_metadata_identity_and_snapshot(self):
        task = self.task()
        self.plan([task])
        self.failure(task)
        self.view.consume(f"WORKER_EXIT {task['task_id']} code=0 kind=algorithm_numerical")
        self.assertFalse(self.view.completed)
        self.assertIn("unverified=1", " ".join(self.view.lines()))
        self.snapshot(task)
        wrong = deepcopy(task)
        wrong["fingerprint"] = "foreign"
        self.write(f"tasks/{task['task_id']}/task.json", wrong)
        self.view.refresh()
        self.assertFalse(self.view.completed)
        self.write(f"tasks/{task['task_id']}/task.json", task)
        self.view.refresh()
        self.assertEqual(self.view.completed, {task["task_id"]})

    def test_paused_failed_queued_are_not_success_or_settled(self):
        tasks = [self.task(i) for i in range(3)]
        self.plan(tasks)
        self.view.consume(f"WORKER_EXIT {tasks[0]['task_id']} code=75 kind=budget_or_interrupt")
        self.view.consume(f"WORKER_EXIT {tasks[1]['task_id']} code=1 kind=algorithm_numerical")
        self.assertFalse(self.view.consume("PROGRESS settled=3/3 active=0 queued=0 budget_expired=True"))
        self.assertIn("done=0/3 failed=1 paused=1 active=0 queued=1", self.view.lines()[1])

    def test_search_dynamic_wave_and_contiguous_evaluated_prefix(self):
        waves = [[self.task(i, "validation", wave) for i in range(3 if wave == 0 else 2)]
                 for wave in range(3)]
        for index, tasks in enumerate(waves):
            self.plan(tasks, f"task_plans/b000-wave{index:03d}.json")
        state = {"manifest_fingerprint": self.manifest["fingerprint"], "status": "searching",
            "blocks": [{"id": "b000", "waves": [
                {"index": 0, "status": "complete"}, {"index": 1, "status": "pending"},
                {"index": 2, "status": "complete"}]}]}
        self.write("search_state.json", state)
        self.view.refresh()
        self.assertEqual(self.view.phase, "validation")
        self.assertEqual(self.view.total, 2)
        self.assertEqual(self.view.wave_index, 1)
        self.assertEqual(len(self.view.evaluated_ids), 3)
        self.assertIn("future search size is adaptive", " ".join(self.view.lines()))
        state["blocks"][0]["waves"][1]["status"] = "complete"
        self.write("search_state.json", state)
        self.view.refresh()
        self.assertEqual(self.view.wave_index, 2)
        self.assertEqual(len(self.view.evaluated_ids), 7)

    def test_eta_excludes_checkpoint_baseline_and_prompt_wait(self):
        task = self.task()
        self.plan([task])
        self.progress(task, 100)
        self.view.refresh()
        self.view.consume("CONTINUATION_PROMPT choice [Y/N] ")
        self.clock[0] = 60.
        self.view.consume(f"START {task['task_id']} device=cuda:0")
        self.assertFalse(self.view.waiting_for_choice)
        self.assertEqual(self.view.new_rounds, 0)
        self.clock[0] = 62.
        self.progress(task, 101)
        self.view.refresh()
        self.assertEqual(self.view.new_rounds, 0)  # first fresh observation is a baseline
        self.clock[0] = 64.
        self.progress(task, 102)
        self.view.refresh()
        self.assertEqual(self.view.new_rounds, 1)
        self.assertEqual(self.view.eta(), 48 * 4)
        self.view.refresh()
        self.assertEqual(self.view.new_rounds, 1)

    def test_devices_prompt_decline_and_completed_report(self):
        task = self.task()
        self.plan([task])
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "6,7"}):
            self.view.consume('DEVICES {"selected": ["cuda:0", "cuda:1"]}')
        self.assertEqual(self.view.mapping["cuda:1"], "visible GPU 7")
        self.view.consume("CONTINUATION_PROMPT choice [Y/N]")
        self.view.refresh()
        self.assertIn("awaiting Y/N", self.view.status)
        self.view.consume("FINAL_NOT_STARTED 用户未确认")
        self.view.refresh()
        self.view.finish(0)
        self.assertIn("FINAL_NOT_STARTED", self.view.status)
        self.assertFalse(self.view.waiting_for_choice)
        view = AdaptiveProgress(self.output, self.spec)
        self.snapshot(task, healthy=False)
        self.write("final_summary.json", {"report_status": "completed", "status": "completed_with_health_failures",
            "tasks": [{"task_id": task["task_id"], "fingerprint": task["fingerprint"],
                "execution_status": "complete", "final_round_available": True, "required_round": 150}]})
        view.consume("FINAL_REPORT report.html")
        self.assertEqual(view.completed, {task["task_id"]})
        self.assertIn("training=completed_with_health_failures", view.status)
        view.consume("ValueError: scientific source changed")
        view.finish(1)
        self.assertIn("exited code=1", view.status)
        self.assertIn("saved report=completed", view.status)
        self.assertIn("did not finish successfully", view.status)

    def test_metadata_cache_avoids_reparsing_unchanged_files_and_writes_nothing(self):
        task = self.task()
        self.plan([task])
        self.progress(task, 10)
        self.view.refresh()
        before = {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        with mock.patch.object(Path, "read_text", side_effect=AssertionError("unchanged JSON should be cached")):
            self.view.refresh()
            self.view.lines()
        after = {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_cifar_schema8_and_cross_manifest_plan(self):
        spec = {"schema_version": 8, "protocol": "cifar-resnet18-gn-tpe-v1"}
        self.manifest["spec"] = spec
        self.write("manifest.json", self.manifest)
        task = self.task()
        self.plan([task])
        other = AdaptiveProgress(self.output, spec)
        other.refresh()
        self.assertEqual(other.phase, "final")
        self.write("final_plan.json", {"manifest_fingerprint": "other-study", "tasks": [task]})
        other = AdaptiveProgress(self.output, spec)
        other.refresh()
        self.assertEqual(other.total, 0)

    def test_malformed_metadata_and_spec_mismatch_do_not_validate_tasks(self):
        task = self.task()
        self.plan([task])
        self.write("search_state.json", {"manifest_fingerprint": "fashion-study", "blocks": [None, {"waves": 4}]})
        self.write("final_plan.json", {"manifest_fingerprint": "fashion-study",
            "tasks": [None, {"task_id": "bad", "config": [], "candidate": []}]})
        self.write("final_summary.json", {"tasks": [None, "invalid"]})
        view = AdaptiveProgress(self.output, self.spec)
        view.refresh()
        self.assertEqual(view.total, 0)
        self.manifest["spec"] = {**self.spec, "changed": True}
        self.write("manifest.json", self.manifest)
        self.view.refresh()
        self.assertEqual(self.view.total, 0)
        self.assertIn("mismatch", self.view.status)


if __name__ == "__main__":
    unittest.main()
