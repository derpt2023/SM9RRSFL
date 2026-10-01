"""Independent Fashion controller/worker/report integration; no real training."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import cifar_adaptive_runtime as cifar_runtime
import fashion_adaptive_reporting as reporting
import fashion_adaptive_runtime as runtime
import fashion_resnet_gn
import run_cifar_adaptive as cifar_runner
import run_fashion_adaptive as runner
from cifar_adaptive_search import AdaptiveTPESampler, METHODS
from cifar_relative_best_gate import evaluation_count
from tests.test_cifar_six_pipeline import synthetic_run


def small_spec(blocks=1):
    spec = runner.load_spec(runner.DEFAULT_CONFIG)
    spec["search"]["max_public_trials"] = blocks
    spec["search"]["max_gpus"] = 2
    return spec


class SyntheticFashionStudy:
    def __init__(self, output, spec, *, unqualified=False):
        self.output, self.spec, self.unqualified = output, spec, unqualified
        self.manifest = runtime.build_manifest(spec, {"synthetic": "Fashion integration"})
        self.results, self.dispatched = {}, []
        self.worker_count = self.execute_calls = 0
        self.pause_after = None
        self.args = SimpleNamespace(output=output, data_dir=None,
                                    devices=["cuda:0", "cuda:1"], phase="all")

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
            accuracy = (2400 + 15 * block + wave) / 3000 if config.method == "sm9rrs" else .8
            asr = .2 if self.unqualified and config.method == "sm9rrs" else .05
            self.results[task["task_id"]] = synthetic_run(config, accuracy=accuracy, asr=asr)
            self.worker_count += 1
        if heartbeat:
            heartbeat()
        return self.statuses(tasks)

    def collect(self, output, tasks):
        groups = {}
        for task in tasks:
            if task["task_id"] in self.results:
                groups.setdefault(task["candidate"]["candidate_id"], []).append(self.results[task["task_id"]])
        return groups, self.statuses(tasks)

    def search(self, state):
        with mock.patch.object(runtime, "execute", side_effect=self.execute), \
                mock.patch.object(runtime, "collect", side_effect=self.collect), redirect_stdout(io.StringIO()):
            runner.run_search(self.args, self.spec, self.manifest, state)


class FashionIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name).resolve()
        self.spec = small_spec()
        self.study = SyntheticFashionStudy(self.output, self.spec)
        self.state = runner.new_state(self.spec, self.study.manifest)

    def initial_block(self):
        block = runner.add_block(self.state, self.spec)
        runner.add_wave(block)
        return runner.block_spec(self.spec, block)

    def final(self, answer="Y\n"):
        choice = runner.choose_block(self.state)
        with mock.patch.object(runtime, "execute", side_effect=self.study.execute), \
                mock.patch.object(runtime, "collect", side_effect=self.study.collect), \
                mock.patch.object(reporting, "write_final_report") as report, \
                mock.patch("sys.stdin", io.StringIO(answer)), redirect_stdout(io.StringIO()) as output:
            code = runner.run_final(self.study.args, self.spec, self.study.manifest, self.state, choice)
        return code, output.getvalue(), report

    def test_explicit_fashion_contract_and_cross_protocol_rejection(self):
        self.assertEqual(self.spec["dataset"]["name"], "fashion_mnist")
        self.assertEqual(self.spec["dataset"]["train_samples"], 60000)
        self.assertEqual(self.spec["validation"]["seeds"], [2026100101, 2026100102])
        self.assertEqual(self.spec["final"]["seeds"], [2026100111])
        self.assertEqual(evaluation_count(self.spec, "validation"), 3000)
        self.assertEqual(evaluation_count(self.spec, "final"), 10000)
        old = cifar_runner.load_spec(cifar_runner.DEFAULT_CONFIG)
        self.assertNotEqual(old["output_dir"], self.spec["output_dir"])
        self.assertNotEqual(old["search"]["seed"], self.spec["search"]["seed"])
        with self.assertRaisesRegex(ValueError, "independent"):
            runner.load_spec(cifar_runner.DEFAULT_CONFIG)
        with self.assertRaises(ValueError):
            cifar_runner.load_spec(runner.DEFAULT_CONFIG)
        for mutate in (lambda s: s["dataset"].update(name="mnist"),
                       lambda s: s["dataset"].update(train_samples=50000),
                       lambda s: s["final"].update(seeds=s["validation"]["seeds"][:1])):
            bad = deepcopy(self.spec)
            mutate(bad)
            path = self.output / "bad.json"
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                runner.load_spec(path)

    def test_old_49_source_bytes_and_cifar_config_remain_frozen(self):
        # Frozen at pre-Fashion HEAD 47a2720. New root modules cannot enter the
        # package glob or change identities of already running CIFAR studies.
        hashes = cifar_runtime.source_hashes()
        self.assertEqual(len(hashes), 49)
        digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(digest, "3d31ceec50f108a617b062fde7aaaecdac33a76fd2189c91ca2a2e19c89b7619")
        self.assertEqual(hashlib.sha256(cifar_runner.DEFAULT_CONFIG.read_bytes()).hexdigest(),
                         "077f90670fe6f848e1d6f081191efb3947805dda9b33902f3a419a311bbc078e")
        fashion_hashes = runtime.source_hashes()
        self.assertEqual(len(fashion_hashes), 54)
        self.assertEqual({key: fashion_hashes[key] for key in hashes}, hashes)
        self.assertEqual(set(fashion_hashes) - set(hashes), set(runtime.NEW_SOURCES))

    def test_manifest_rejects_cifar_cache_before_loading_data(self):
        old = cifar_runner.load_spec(cifar_runner.DEFAULT_CONFIG)
        manifest = cifar_runtime.build_manifest(old, {"synthetic": True})
        runner.base.write_json(self.output / "manifest.json", manifest)
        self.assertNotEqual(manifest["fingerprint"], self.study.manifest["fingerprint"])
        with mock.patch.object(runner, "load_split") as load, \
                mock.patch.object(runtime, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "identity changed"):
                runner.run_parent(self.study.args, self.spec)
        load.assert_not_called()
        execute.assert_not_called()
        self.assertEqual(runtime.read_json(self.output / "manifest.json"), manifest)

    def test_three_public_blocks_fit_tpe_and_run_60_independent_formal_tasks(self):
        self.spec["search"]["max_public_trials"] = 3
        self.study = SyntheticFashionStudy(self.output, self.spec)
        self.state = runner.new_state(self.spec, self.study.manifest)
        self.study.search(self.state)
        self.assertEqual(self.state["status"], "trial_limit_reached")
        self.assertEqual(self.study.worker_count, 900)
        for block in self.state["blocks"]:
            current = runner.block_spec(self.spec, block)
            self.assertEqual({m: len(cs) for m, cs in current["candidates"].items()},
                             {"sm9rrs": 4, "vert": 4, "alignins": 4, "krum": 1, "ding13": 1, "fedavg": 1})
            for method in ("sm9rrs", "vert", "alignins"):
                self.assertEqual([t["strategy"] for t in block["method_samplers"][method]["trials"]],
                                 ["declared_default", "random_startup", "tpe", "tpe"])
            for task in runtime.attach_tasks(current, "validation", self.study.manifest):
                self.assertEqual(task["config"]["rounds"], 150)
                self.assertIn(task["config"]["seed"], self.spec["validation"]["seeds"])
                for name, value in block["proposal"]["parameters"].items():
                    self.assertEqual(task["config"][name], value)
                self.assertEqual(task["config"]["attack_start_round"], task["config"]["detector_window"] + 2)
        public = self.state["public_sampler"]["trials"]
        self.assertEqual(public[2]["strategy"], "tpe")
        self.assertEqual(public[2]["fit_audit"]["fitted_observations"], 2)
        restored = AdaptiveTPESampler.from_state(json.loads(json.dumps(self.state["public_sampler"])))
        self.assertEqual(restored.state_dict(), self.state["public_sampler"])
        frozen = json.dumps(self.state, sort_keys=True)
        self.assertEqual(self.final()[0], 0)
        self.assertEqual(self.study.worker_count, 960)
        plan = runtime.read_json(self.output / "final_plan.json")
        self.assertEqual(len(plan["tasks"]), 60)
        self.assertEqual({t["config"]["seed"] for t in plan["tasks"]}, {2026100111})
        self.assertTrue(all(t["phase"] == "final" for t in plan["tasks"]))
        self.assertEqual(json.dumps(self.state, sort_keys=True), frozen)
        self.assertEqual(self.final()[0], 0)
        self.assertEqual(self.study.worker_count, 960)
        self.assertEqual(json.dumps(self.state, sort_keys=True), frozen)

    def test_incomplete_search_resume_preserves_proposal_and_cached_tasks(self):
        self.study.pause_after = 1
        self.study.search(self.state)
        self.assertEqual(self.state["status"], "execution_blocked")
        self.assertIsNone(runner.choose_block(self.state))
        count = self.study.worker_count
        self.assertEqual(count, 120)
        pending = deepcopy(self.state["blocks"][0]["waves"][-1]["candidates"])
        self.state = json.loads(json.dumps(self.state))
        self.state["status"] = "searching"
        self.study.pause_after = None
        self.study.search(self.state)
        self.assertEqual(self.study.worker_count, 300)
        self.assertEqual(self.state["blocks"][0]["waves"][1]["candidates"], pending)

    def test_budget_selection_uses_only_complete_fair_prefix(self):
        self.study.pause_after = 1
        self.study.search(self.state)
        self.state["status"] = "budget_exhausted"
        block = self.state["blocks"][0]
        self.assertIsNone(runner.choose_block(self.state))
        with mock.patch.object(runtime, "collect", side_effect=self.study.collect):
            runner.prepare_selection(self.output, self.spec, self.study.manifest, self.state)
        snapshot = block["budget_selection_snapshot"]
        self.assertEqual(snapshot["completed_waves"], 1)
        pending_ids = {item["candidate"]["candidate_id"] for item in block["waves"][1]["candidates"]}
        self.assertEqual(set(snapshot["excluded_candidates"]), pending_ids)
        self.assertEqual(len(block["waves"]), 2)
        self.assertEqual(block["waves"][1]["status"], "pending")
        self.assertEqual(len(runner.selection_view(block)["waves"]), 1)
        choice = runner.choose_block(self.state)
        self.assertTrue(set(choice["selected"].values()).isdisjoint(pending_ids))
        self.study.pause_after = None
        self.assertEqual(self.final()[0], 0)
        tasks = runtime.read_json(self.output / "final_plan.json")["tasks"]
        self.assertEqual(len(tasks), 60)
        self.assertTrue(all(t["candidate"]["candidate_id"].endswith("-d000") for t in tasks))

    def test_budget_deadline_checkpoints_before_pause_and_dispatches_no_new_worker(self):
        task = runtime.attach_tasks(self.initial_block(), "validation", self.study.manifest)[0]
        folder = runtime.ensure_identity(self.output, task)
        record = SimpleNamespace(accuracy=.8, attack_target_success_rate=.1,
                                 attack_target_confidence=.2, honest_weight_loss=0., malicious_weight_mass=0.)
        checkpoint = {"completed_round": 1, "params": [0., 1.], "records": [record]}
        saved = []
        def original(*args, **kwargs):
            kwargs["checkpoint_callback"](checkpoint)
        def save(value):
            self.assertFalse((folder / "progress.json").exists())
            saved.append(value["completed_round"])
        with mock.patch.object(runtime.base.experiments, "run_experiment", side_effect=original), \
                mock.patch.object(runtime.time, "time", return_value=100.), redirect_stdout(io.StringIO()):
            with runtime.round_monitor(task, folder, deadline=99.) as event:
                with self.assertRaises(runtime.BudgetPause) as paused:
                    runtime.base.experiments.run_experiment(checkpoint_callback=save)
        self.assertEqual(saved, [1])
        self.assertEqual(runtime.read_json(folder / "progress.json")["last_completed_round"], 1)
        self.assertEqual(runtime.failure_class(paused.exception, event), "budget_or_interrupt")
        self.assertEqual(runtime.failure_class(RuntimeError("CUDA out of memory"), event), "infrastructure_or_execution")
        self.assertEqual(runtime.failure_class(FloatingPointError("non-finite"), event), "algorithm_numerical")
        plan = runtime.save_plan(self.output, "expired", [task], self.study.manifest)
        with mock.patch.object(runtime.time, "time", return_value=100.), \
                mock.patch.object(runtime.subprocess, "Popen") as start, redirect_stdout(io.StringIO()):
            statuses = runtime.execute(self.study.args, [task], plan, deadline=99.)
        start.assert_not_called()
        self.assertEqual(statuses[0]["status"], "pending")

    def test_manual_y_n_is_required_on_every_resume(self):
        self.study.unqualified = True
        self.study.search(self.state)
        self.assertFalse(runner.choose_block(self.state)["target_qualified"])
        count = self.study.worker_count
        code, output, report = self.final("N\n")
        self.assertEqual(code, 0)
        self.assertIn("CONTINUATION_PROMPT", output)
        report.assert_not_called()
        self.assertEqual(self.study.worker_count, count)
        self.assertFalse((self.output / "continuation_decision.json").exists())
        self.final("Y\n")
        self.assertEqual(self.study.worker_count, count + 60)
        frozen = (self.output / "continuation_decision.json").read_bytes()
        for answer in ("N\n", "Y\n"):
            code, output, report = self.final(answer)
            self.assertEqual(code, 0)
            self.assertIn("CONTINUATION_PROMPT", output)
            self.assertEqual(self.study.worker_count, count + 60)
            self.assertEqual((self.output / "continuation_decision.json").read_bytes(), frozen)
        self.assertEqual(len(list((self.output / "continuation_responses").glob("*.json"))), 4)

    def test_fashion_3000_accuracy_samples_preserve_exact_two_pp_boundary(self):
        spec = self.initial_block()
        tasks = runtime.attach_tasks(spec, "validation", self.study.manifest)
        results = {}
        for task in tasks:
            ours = task["method"] == "sm9rrs"
            results.setdefault(task["candidate"]["candidate_id"], []).append(synthetic_run(
                runner.base.fl.ExperimentConfig(**task["config"]), accuracy=2340 / 3000 if ours else .8,
                asr=.12 if ours else .1))
        self.assertEqual(runner.gate.select_validation(spec, results, tasks)["status"], "qualified_for_final")
        cid = spec["candidates"]["sm9rrs"][0]["candidate_id"]
        results[cid] = [synthetic_run(r.config, accuracy=2339 / 3000, asr=.12) for r in results[cid]]
        self.assertEqual(runner.gate.select_validation(spec, results, tasks)["status"], "needs_ours_target_development")

    def test_plan_only_does_not_load_any_dataset_or_gpu(self):
        import run_cifar_six_with_progress
        with mock.patch.object(runner, "load_split") as fashion_load, \
                mock.patch.object(runner.base, "load_split") as cifar_load, \
                mock.patch.object(run_cifar_six_with_progress, "discover_gpus") as gpu, \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(runner.main(["--plan-only"]), 0)
        fashion_load.assert_not_called()
        cifar_load.assert_not_called()
        gpu.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["formal_tasks"], 60)
        self.assertEqual(plan["maximum_validation_tasks"], 4800)
        self.assertFalse(plan["execution_started"])

    def test_worker_dispatch_uses_fashion_entry_and_never_cifar_entry(self):
        task = runtime.attach_tasks(self.initial_block(), "validation", self.study.manifest)[0]
        plan = runtime.save_plan(self.output, "worker-entry", [task], self.study.manifest)
        command = []
        def launch(arguments, **kwargs):
            command.extend(arguments)
            folder = self.output / "tasks" / task["task_id"]
            result = synthetic_run(runner.base.fl.ExperimentConfig(**task["config"]))
            runner.base.experiments._write_completed_results_snapshot(folder, [result])
            return SimpleNamespace(pid=41001, poll=lambda: 0)
        with mock.patch.object(runtime.subprocess, "Popen", side_effect=launch), redirect_stdout(io.StringIO()):
            statuses = runtime.execute(self.study.args, [task], plan)
        self.assertEqual(Path(command[2]).name, "run_fashion_adaptive.py")
        self.assertNotIn(str(runner.REPO / "run_cifar_adaptive.py"), command)
        self.assertEqual(statuses[0]["status"], "complete")
        with mock.patch.object(runtime.subprocess, "Popen") as start, redirect_stdout(io.StringIO()):
            runtime.execute(self.study.args, [task], plan)
        start.assert_not_called()

    def test_worker_loads_fashion_split_and_installs_only_fashion_model(self):
        spec = self.initial_block()
        selected = {m: cs[0]["candidate_id"] for m, cs in spec["candidates"].items()}
        split = SimpleNamespace(calibration_dataset=object(), main_dataset=object())
        runner.base.write_json(self.output / "manifest.json", self.study.manifest)
        import cifar_resnet_gn
        for phase, expected in (("validation", split.calibration_dataset), ("final", split.main_dataset)):
            task = runtime.attach_tasks(spec, phase, self.study.manifest, selected if phase == "final" else None)[0]
            runtime.ensure_identity(self.output, task)
            plan = runtime.save_plan(self.output, "worker-" + phase, [task], self.study.manifest)
            args = SimpleNamespace(**vars(self.study.args), worker=task["task_id"], plan=plan, stop_at=None)
            result = synthetic_run(runner.base.fl.ExperimentConfig(**task["config"]))
            with mock.patch.object(runtime, "load_split", return_value=(split, self.study.manifest["data_contract"])) as load, \
                    mock.patch.object(runtime.base, "load_split", side_effect=AssertionError("CIFAR loader called")), \
                    mock.patch.object(fashion_resnet_gn, "install_runtime") as install, \
                    mock.patch.object(cifar_resnet_gn, "install_runtime") as wrong_install, \
                    mock.patch.object(runtime.base, "worker_environment", return_value={}), \
                    mock.patch.object(runtime.base, "check_environment"), \
                    mock.patch.object(runtime.base.experiments, "run_measured_experiment", return_value=result) as run, \
                    mock.patch.object(runtime.base.experiments, "write_result_files"), \
                    mock.patch.object(runtime.base.experiments, "finalize_config_checkpoint"), redirect_stdout(io.StringIO()):
                self.assertEqual(runtime.worker(args), 0)
            load.assert_called_once_with(self.spec, None)
            install.assert_called_once()
            wrong_install.assert_not_called()
            self.assertIs(run.call_args.args[0], expected)
            self.assertEqual(run.call_args.kwargs["run_fingerprint"], task["fingerprint"])

    def test_report_labels_grayscale_model_and_single_seed_without_reselection(self):
        spec = self.initial_block()
        selected = {m: cs[0]["candidate_id"] for m, cs in spec["candidates"].items()}
        tasks = runtime.attach_tasks(spec, "final", self.study.manifest, selected)
        results, statuses = {}, []
        for task in tasks:
            runtime.ensure_identity(self.output, task)
            cid = task["candidate"]["candidate_id"]
            results.setdefault(cid, []).append(synthetic_run(runner.base.fl.ExperimentConfig(**task["config"])))
            statuses.append({"task_id": task["task_id"], "candidate_id": cid,
                             "method": task["method"], "status": "complete"})
        summary = reporting.write_final_report(self.output, spec, selected, results, tasks, statuses,
                                               {"status": "qualified_for_final"})
        self.assertEqual(summary["completed_tasks"], 60)
        self.assertEqual(summary["model"]["input_shape"], [1, 28, 28])
        self.assertEqual(summary["formal_seeds"], [2026100111])
        self.assertFalse(summary["parameters_reselected"])
        self.assertFalse(summary["official_test_used_for_selection"])
        html = (self.output / "final_results/visualizations.html").read_text()
        self.assertIn("Fashion-MNIST", html)
        self.assertNotIn("CIFAR-10", html)
        for path in (self.output / "final_results").glob("*.svg"):
            self.assertIn("Fashion-MNIST", path.read_text())
            self.assertNotIn("CIFAR-10", path.read_text())
        audit = runtime.read_json(self.output / "final_results/data_audit.json")
        self.assertIn("fashion_resnet_gn.py", audit["report_source_sha256"])
        self.assertIn("fashion_adaptive_reporting.py", audit["report_source_sha256"])


if __name__ == "__main__":
    unittest.main()
