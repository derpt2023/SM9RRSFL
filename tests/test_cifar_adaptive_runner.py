"""Whole synthetic search/resume/final flows; no data, GPU, or real training."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_cifar_adaptive as runner
from cifar_adaptive_search import AdaptiveTPESampler, METHODS, space_contract
from tests.test_cifar_six_pipeline import synthetic_run


def minimal_spec():
    old = json.loads((runner.REPO / "configs/cifar10_six_relative_best_five_day_v7.json").read_text())
    shared = old["shared_parameters"]
    shared.update(space_contract("public")["default_parameters"])
    shared.update(rounds=150, attack_start_round=shared["detector_window"] + 2)
    return {"schema_version": 8, "protocol": "cifar-resnet18-gn-tpe-v1",
            "model": {"architecture": "cifar_resnet18_gn", "group_norm_groups": 2},
            "dataset": old["dataset"], "shared_parameters": shared,
            "search_spaces": {m: space_contract(m) for m in ("public", *METHODS)},
            "validation": {**old["validation"], "seeds": [2026093001, 2026093002]},
            "final": {**old["final"], "seeds": [2026093011]},
            "objective": dict(runner.gate.OBJECTIVE),
            "performance_target": dict(runner.gate.PERFORMANCE_TARGET),
            "gates": {"max_clean_accuracy_drop": .03, "max_nonfinite_updates": 0,
                      "min_round_completion_rate": 1.},
            "promotion": {"require_mean_dual_best": False, "manual_healthy_override": True,
                          "ask_on_every_resume": True},
            "search": {"max_public_trials": 2, "defense_trials_per_method": 4,
                       "startup_trials": 2, "max_gpus": 2, "seed": 2026093000,
                       "budget_hours": 48.},
            "output_dir": "outputs/synthetic-not-used"}


class SyntheticStudy:
    def __init__(self, output, spec, *, unqualified=False):
        self.output, self.spec, self.unqualified = output, spec, unqualified
        self.manifest = {"fingerprint": "synthetic-adaptive-study", "data_contract": {"synthetic": True}}
        self.results = {}
        self.dispatched = []
        self.worker_count = 0
        self.pause_after = None
        self.execute_calls = 0
        self.args = SimpleNamespace(output=output, data_dir=None, devices=["cuda:0", "cuda:1"], phase="all")

    def statuses(self, tasks):
        return [{"task_id": t["task_id"], "candidate_id": t["candidate"]["candidate_id"],
                 "method": t["method"], "status": "complete" if t["task_id"] in self.results else "pending"}
                for t in tasks]

    def execute(self, args, tasks, plan, *, deadline=None, heartbeat=None):
        self.execute_calls += 1
        self.dispatched.append(deepcopy(tasks))
        if self.pause_after is not None and self.execute_calls > self.pause_after:
            return self.statuses(tasks)
        for task in tasks:
            if task["task_id"] in self.results:
                continue
            config = runner.base.fl.ExperimentConfig(**task["config"])
            cid = task["candidate"]["candidate_id"]
            block = int(cid.split("-b")[1].split("-")[0])
            wave = int(cid.rsplit("-d", 1)[1])
            accuracy = .8 + (.008 * block + .0004 * wave if config.method == "sm9rrs" else 0.)
            asr = .2 if self.unqualified and config.method == "sm9rrs" else .05
            self.results[task["task_id"]] = synthetic_run(config, accuracy=accuracy, asr=asr)
            self.worker_count += 1
        if heartbeat is not None:
            heartbeat()
        return self.statuses(tasks)

    def collect(self, output, tasks):
        groups = {}
        for task in tasks:
            if task["task_id"] in self.results:
                groups.setdefault(task["candidate"]["candidate_id"], []).append(self.results[task["task_id"]])
        return groups, self.statuses(tasks)

    def search(self, state):
        with mock.patch.object(runner.runtime, "execute", side_effect=self.execute), \
                mock.patch.object(runner.runtime, "collect", side_effect=self.collect), redirect_stdout(io.StringIO()):
            runner.run_search(self.args, self.spec, self.manifest, state)


class AdaptiveRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name).resolve()
        self.spec = minimal_spec()
        self.study = SyntheticStudy(self.output, self.spec)
        self.state = runner.new_state(self.spec, self.study.manifest)

    def write_spec(self, spec=None):
        path = self.output / "config.json"
        path.write_text(json.dumps(self.spec if spec is None else spec))
        return path

    def final(self, state=None, *, answer="Y\n"):
        state = self.state if state is None else state
        choice = runner.choose_block(state)
        with mock.patch.object(runner.runtime, "execute", side_effect=self.study.execute), \
                mock.patch.object(runner.runtime, "collect", side_effect=self.study.collect), \
                mock.patch("cifar_adaptive_reporting.write_final_report") as report, \
                mock.patch("sys.stdin", io.StringIO(answer)), redirect_stdout(io.StringIO()) as text:
            code = runner.run_final(self.study.args, self.spec, self.study.manifest, state, choice)
        return code, text.getvalue(), report

    def test_real_new_config_schema_loads_and_rejects_seed_leakage(self):
        self.assertEqual(runner.load_spec(self.write_spec()), self.spec)
        bad = deepcopy(self.spec)
        bad["final"]["seeds"] = bad["validation"]["seeds"][:1]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            runner.load_spec(self.write_spec(bad))
        bad = deepcopy(self.spec)
        bad["shared_parameters"]["rounds"] = 100
        with self.assertRaisesRegex(ValueError, "150-round"):
            runner.load_spec(self.write_spec(bad))
        bad = deepcopy(self.spec)
        bad["search"]["budget_hours"] = 49
        with self.assertRaisesRegex(ValueError, "48 hours"):
            runner.load_spec(self.write_spec(bad))

    def test_two_complete_blocks_each_retune_three_methods_with_equal_budget(self):
        self.study.search(self.state)
        self.assertEqual(self.state["status"], "trial_limit_reached")
        self.assertEqual(len(self.state["blocks"]), 2)
        self.assertEqual(self.study.worker_count, 600)
        self.assertEqual(len(self.study.dispatched), 8)
        for block in self.state["blocks"]:
            self.assertEqual(block["status"], "complete")
            self.assertEqual(len(block["waves"]), 4)
            current = runner.block_spec(self.spec, block)
            self.assertEqual({m: len(cs) for m, cs in current["candidates"].items()},
                             {"sm9rrs": 4, "vert": 4, "alignins": 4, "krum": 1, "ding13": 1, "fedavg": 1})
            tasks = runner.runtime.attach_tasks(current, "validation", self.study.manifest)
            self.assertEqual(len(tasks), 300)
            for task in tasks:
                config = task["config"]
                self.assertIn(config["seed"], self.spec["validation"]["seeds"])
                self.assertEqual(config["rounds"], 150)
                for key, value in block["proposal"]["parameters"].items():
                    self.assertEqual(config[key], value)
                self.assertEqual(config["attack_start_round"], config["detector_window"] + 2)
            for method in ("sm9rrs", "vert", "alignins"):
                trials = block["method_samplers"][method]["trials"]
                self.assertEqual([t["strategy"] for t in trials],
                                 ["declared_default", "random_startup", "tpe", "tpe"])
                self.assertTrue(all(t["status"] == "complete" for t in trials))
            self.assertNotEqual(block["method_samplers"]["vert"]["context_id"], "")
        first, second = self.state["blocks"]
        self.assertNotEqual(first["context_id"], second["context_id"])
        self.assertNotEqual(first["method_samplers"]["vert"]["trials"][1]["parameters"],
                            second["method_samplers"]["vert"]["trials"][1]["parameters"])
        public = AdaptiveTPESampler.from_state(json.loads(json.dumps(self.state["public_sampler"])))
        proposal = public.ask()
        self.assertEqual(proposal["strategy"], "tpe")
        self.assertEqual(proposal["fit_audit"]["fitted_observations"], 2)

    def test_one_formal_seed_has_60_fresh_tasks_and_never_updates_search(self):
        self.study.search(self.state)
        frozen_state = json.dumps(self.state, sort_keys=True)
        before = self.study.worker_count
        code, output, report = self.final()
        self.assertEqual(code, 0)
        self.assertEqual(self.study.worker_count - before, 60)
        report.assert_called_once()
        formal = runner.runtime.read_json(self.output / "final_plan.json")
        self.assertEqual(len(formal["tasks"]), 60)
        self.assertEqual({t["config"]["seed"] for t in formal["tasks"]}, set(self.spec["final"]["seeds"]))
        self.assertEqual({t["phase"] for t in formal["tasks"]}, {"final"})
        self.assertEqual(json.dumps(self.state, sort_keys=True), frozen_state)
        count = self.study.worker_count
        self.final()
        self.assertEqual(self.study.worker_count, count)
        self.assertEqual(json.dumps(self.state, sort_keys=True), frozen_state)

    def test_incomplete_wave_blocks_selection_then_resume_reuses_same_proposal(self):
        self.study.pause_after = 1
        self.study.search(self.state)
        self.assertEqual(self.state["status"], "execution_blocked")
        self.assertIsNone(runner.choose_block(self.state))
        initial = deepcopy(self.state["blocks"][0]["waves"])
        self.assertEqual(self.study.worker_count, 120)
        restored = runner.runtime.read_json(self.output / "search_state.json")
        self.assertEqual(restored["blocks"][0]["waves"], initial)
        self.study.pause_after = None
        restored["status"] = "searching"
        self.study.search(restored)
        self.assertEqual(restored["status"], "trial_limit_reached")
        self.assertEqual(self.study.worker_count, 600)
        self.assertEqual(restored["blocks"][0]["waves"][0], initial[0])
        self.assertEqual(restored["blocks"][0]["waves"][1]["candidates"], initial[1]["candidates"])

    def test_budget_exhaustion_preserves_partial_block_without_qualification(self):
        block = runner.add_block(self.state, self.spec)
        runner.add_wave(block)
        self.state["elapsed_search_seconds"] = self.spec["search"]["budget_hours"] * 3600
        with mock.patch.object(runner.runtime, "execute") as execute, redirect_stdout(io.StringIO()):
            runner.run_search(self.study.args, self.spec, self.study.manifest, self.state)
        execute.assert_not_called()
        self.assertEqual(self.state["status"], "budget_exhausted")
        self.assertIsNone(runner.choose_block(self.state))
        restored = runner.runtime.read_json(self.output / "search_state.json")
        self.assertEqual(restored["blocks"][0]["waves"], block["waves"])
        self.assertEqual(restored["public_sampler"]["trials"][0]["status"], "pending")

    def test_budget_uses_only_equal_complete_prefix_and_excludes_better_partial_candidate(self):
        self.study.pause_after = 1
        self.study.search(self.state)
        self.state["status"] = "budget_exhausted"
        block = self.state["blocks"][0]
        pending_wave = block["waves"][1]
        all_tasks = runner.runtime.attach_tasks(runner.block_spec(self.spec, block), "validation", self.study.manifest)
        pending_ours = next(item["candidate"]["candidate_id"] for item in pending_wave["candidates"] if item["method"] == "sm9rrs")
        for task in all_tasks:
            if task["candidate"]["candidate_id"] == pending_ours:
                self.study.results[task["task_id"]] = synthetic_run(
                    runner.base.fl.ExperimentConfig(**task["config"]), accuracy=.99, asr=0.)
        before_sampler = deepcopy(block["method_samplers"])
        with mock.patch.object(runner.runtime, "collect", side_effect=self.study.collect):
            runner.prepare_selection(self.output, self.spec, self.study.manifest, self.state)
        snapshot = block["budget_selection_snapshot"]
        self.assertEqual(snapshot["completed_waves"], 1)
        self.assertIn(pending_ours, snapshot["excluded_candidates"])
        self.assertEqual(len(snapshot["task_fingerprints"]), 120)
        self.assertTrue(snapshot["cross_public_condition_search_budgets_may_differ"])
        self.assertEqual(block["method_samplers"], before_sampler)
        self.assertEqual(self.state["public_sampler"]["trials"][0]["status"], "pending")
        choice = runner.choose_block(self.state)
        self.assertEqual(choice["completed_waves"], 1)
        self.assertNotEqual(choice["ours_candidate"], pending_ours)
        self.assertTrue(all(cid.endswith("-d000") for cid in choice["selected"].values()))
        self.assertEqual(sum(trial["strategy"] == "tpe" for value in block["method_samplers"].values()
                             for trial in value["trials"]), 0)
        self.study.pause_after = None
        code, _, report = self.final()
        self.assertEqual(code, 0)
        report.assert_called_once()
        formal = runner.runtime.read_json(self.output / "final_plan.json")
        self.assertEqual(len(formal["tasks"]), 60)
        self.assertTrue(all(task["candidate"]["candidate_id"].endswith("-d000") for task in formal["tasks"]))
        self.assertEqual(block["method_samplers"], before_sampler)

    def test_selection_reaudits_raw_evidence_and_rejects_a_changed_result(self):
        self.study.search(self.state)
        with mock.patch.object(runner.runtime, "collect", side_effect=self.study.collect):
            runner.prepare_selection(self.output, self.spec, self.study.manifest, self.state)
            key = next(iter(self.study.results))
            config = self.study.results[key].config
            self.study.results[key] = synthetic_run(config, accuracy=.7, asr=.05)
            with self.assertRaisesRegex(ValueError, "raw validation evidence"):
                runner.prepare_selection(self.output, self.spec, self.study.manifest, self.state)

    def test_candidate_clean_utility_failure_penalizes_public_search_loss(self):
        self.study.pause_after = 1
        self.study.search(self.state)
        original = self.state["blocks"][0]
        view = deepcopy(original)
        view["waves"] = view["waves"][:1]
        for key, run in list(self.study.results.items()):
            if run.config.method == "sm9rrs":
                self.study.results[key] = synthetic_run(run.config, accuracy=.7, asr=.05)
        with mock.patch.object(runner.runtime, "collect", side_effect=self.study.collect):
            report, results, _ = runner.audit_block(self.output, self.spec, self.study.manifest, view)
        cid = view["waves"][0]["candidates"][0]["candidate"]["candidate_id"]
        row = report["final_metric_gate"]["candidate_rows"][cid]
        self.assertTrue(row["scorable"])
        self.assertFalse(row["eligible"])
        self.assertIn("clean_accuracy_drop", row["invalid_reasons"])
        self.assertTrue(all(runner.gate.run_audit(run)["healthy"] for run in results[cid]))
        runner.observe_block(self.state, view, report, results)
        self.assertGreater(view["objectives"][cid]["components"]["health_penalty"], 100.)
        self.assertIn("clean_accuracy_drop", view["objectives"][cid]["candidate_health_reasons"])

    def test_last_wave_saved_before_block_observation_recovers_without_training(self):
        self.study.search(self.state)
        block = self.state["blocks"][-1]
        # Simulate the valid atomic snapshot immediately after observe_wave
        # saved wave 3, before observe_block completed the outer observation.
        block["status"] = "running"
        block.pop("report")
        block.pop("objectives")
        pending_public = self.state["public_sampler"]["trials"][-1]
        pending_public.update(status="pending", loss=None, metrics={})
        self.state["status"] = "searching"
        before = self.study.worker_count
        # A bounded fake clock prevents a regression from spinning indefinitely.
        ticks = iter([0.] * 30 + [200000.] * 100)
        with mock.patch.object(runner.time, "time", side_effect=lambda: next(ticks, 200000.)):
            self.study.search(self.state)
        self.assertEqual(self.study.worker_count, before)
        self.assertEqual(block["status"], "complete")
        self.assertEqual(self.state["public_sampler"]["trials"][-1]["status"], "complete")
        self.assertEqual(self.state["status"], "trial_limit_reached")

    def test_manual_override_reasks_on_every_resume_and_n_stops(self):
        self.study.unqualified = True
        self.study.search(self.state)
        choice = runner.choose_block(self.state)
        self.assertIsNotNone(choice)
        self.assertFalse(choice["target_qualified"])
        count = self.study.worker_count
        code, text, report = self.final(answer="N\n")
        self.assertEqual(code, 0)
        self.assertIn("CONTINUATION_PROMPT", text)
        report.assert_not_called()
        self.assertFalse((self.output / "continuation_decision.json").exists())
        self.assertEqual(self.study.worker_count, count)
        self.final(answer="Y\n")
        self.assertEqual(self.study.worker_count, count + 60)
        frozen = (self.output / "continuation_decision.json").read_bytes()
        code, text, report = self.final(answer="N\n")
        self.assertEqual(code, 0)
        self.assertIn("CONTINUATION_PROMPT", text)
        report.assert_not_called()
        self.assertEqual((self.output / "continuation_decision.json").read_bytes(), frozen)
        self.assertEqual(self.study.worker_count, count + 60)
        code, text, report = self.final(answer="Y\n")
        self.assertEqual(code, 0)
        self.assertIn("CONTINUATION_PROMPT", text)
        report.assert_called_once()
        self.assertEqual(self.study.worker_count, count + 60)
        self.assertEqual(len(list((self.output / "continuation_responses").glob("*.json"))), 4)

    def test_final_choice_change_is_rejected_before_any_new_execution(self):
        self.study.search(self.state)
        self.final()
        choice = runner.choose_block(self.state)
        choice["raw_score"] += .001
        with mock.patch.object(runner.runtime, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "frozen selection"):
                runner.run_final(self.study.args, self.spec, self.study.manifest, self.state, choice)
        execute.assert_not_called()

    def test_parent_resume_uses_frozen_search_without_reloading_or_retuning(self):
        output = self.output / "parent-study"
        study = SyntheticStudy(output, self.spec)
        with mock.patch.object(runner.base, "load_split", return_value=(object(), study.manifest["data_contract"])) as data, \
                mock.patch.object(runner.runtime, "build_manifest", return_value=study.manifest), \
                mock.patch.object(runner.runtime, "execute", side_effect=study.execute), \
                mock.patch.object(runner.runtime, "collect", side_effect=study.collect), \
                mock.patch("cifar_adaptive_reporting.write_final_report"), redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_parent(study.args, self.spec), 0)
            self.assertEqual(study.worker_count, 660)
            state = (output / "search_state.json").read_bytes()
            decision = (output / "continuation_decision.json").read_bytes()
            study.args.phase = "final"
            self.assertEqual(runner.run_parent(study.args, self.spec), 0)
        data.assert_called_once()
        self.assertEqual(study.worker_count, 660)
        self.assertEqual((output / "search_state.json").read_bytes(), state)
        self.assertEqual((output / "continuation_decision.json").read_bytes(), decision)

    def test_plan_only_never_discovers_gpu_or_loads_data(self):
        import run_cifar_six_with_progress
        path = self.write_spec()
        with mock.patch.object(run_cifar_six_with_progress, "discover_gpus") as gpu, \
                mock.patch.object(runner.base, "load_split") as data, redirect_stdout(io.StringIO()) as text:
            self.assertEqual(runner.main(["--config", str(path), "--plan-only"]), 0)
        gpu.assert_not_called()
        data.assert_not_called()
        info = json.loads(text.getvalue())
        self.assertEqual(info["maximum_validation_tasks"], 600)
        self.assertEqual(info["formal_tasks"], 60)
        self.assertFalse(info["execution_started"])


if __name__ == "__main__":
    unittest.main()
