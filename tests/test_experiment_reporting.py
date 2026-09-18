"""Audit synthetic CSV/JSON studies without datasets, GPUs or plot rendering."""
import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import statistics
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.parse import urlsplit

import experiment_reporting as reporting


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_study(root):
    """Two methods, clean/attacked scenarios, three seeds and rounds 0..3."""
    seeds, methods = [901, 902, 903], ["sm9rrs", "vert"]
    shared = dict(partition="iid", dirichlet_alpha=.5, num_clients=10,
                  rounds=3, attack_start_round=2, attack_source_label=5,
                  attack_target_label=7, attack_target_count=20,
                  lr=.05, device="cuda:0", sm9_workers=1)
    scenarios = [{"partition": "iid", "malicious_ratio": ratio} for ratio in (0., .4)]
    selected = {method: method + "-fixed" for method in methods}
    manifest = {"fingerprint": "synthetic-final-study", "final_run_count": 12,
                "spec": {"dataset": {"name": "cifar10"}, "shared_parameters": shared,
                         "candidates": {m: [{"candidate_id": selected[m]}] for m in methods},
                         "final": {"seeds": seeds, "scenarios": scenarios}}}
    plan = {"manifest_fingerprint": manifest["fingerprint"], "selected": selected, "tasks": []}
    final = {"full_execution_completed": True, "all_scheduled_tasks_attempted": True,
             "tasks": [], "health_failures": [], "methods": {}, "selected": selected}
    summaries, rounds = [], []
    for method in methods:
        for scenario in scenarios:
            for seed_index, seed in enumerate(seeds):
                config = dict(shared, **scenario, method=method, seed=seed)
                task_id = f"final_{method}_{config['malicious_ratio']}_{seed}"
                task = {"task_id": task_id, "phase": "final", "method": method,
                        "candidate": {"candidate_id": selected[method]}, "config": config,
                        "fingerprint": "fingerprint-" + task_id}
                plan["tasks"].append(task)
                attacked = config["malicious_ratio"] > 0
                # The failed seed has the largest finite observation. Excluding it
                # changes the expected mean and therefore makes the test fail.
                failed = method == "vert" and attacked and seed == 903
                final_accuracy = .4 + .2 * seed_index
                final_asr = .1 + .2 * seed_index
                observations = []
                for round_id in range(4):
                    observations.append({
                        **{name: config[name] for name in ("partition", "dirichlet_alpha",
                                                          "num_clients", "method", "malicious_ratio", "seed")},
                        "round": round_id, "accuracy": final_accuracy - .05 * (3 - round_id),
                        "attack_target_success_rate": final_asr - .01 * (3 - round_id),
                        "attack_active": attacked and round_id >= 2,
                        "nonfinite_updates": int(failed and round_id == 3),
                    })
                rounds.extend(observations)
                reasons = ["nonfinite_updates"] if failed else []
                metrics = {"healthy": not failed, "reasons": reasons,
                           "final_accuracy": final_accuracy, "stopped_round": 3,
                           "attack_mean_accuracy": statistics.mean(x["accuracy"] for x in observations[2:]) if attacked else None,
                           "attack_mean_asr": statistics.mean(x["attack_target_success_rate"] for x in observations[2:]) if attacked else None,
                           "nonfinite_updates": int(failed), "runtime_seconds": 10. + seed_index}
                summary = {**config, "final_accuracy": final_accuracy,
                           "final_attack_target_success_rate": final_asr,
                           "stopped_round": 3, "effective_attack_start_round": 2,
                           "runtime_seconds": metrics["runtime_seconds"],
                           "crypto_wall_seconds": 2., "runtime_without_crypto_seconds": 8. + seed_index,
                           "peak_memory_mb": 100. + seed_index, "nonfinite_updates": int(failed)}
                summaries.append(summary)
                final["tasks"].append({"task_id": task_id, "status": "complete", "metrics": metrics,
                                       "error": None, "failure_record": None})
                if failed:
                    final["health_failures"].append({"task_id": task_id, "reasons": reasons})
                task_dir = root / "tasks" / task_id
                write_json(task_dir / "task.json", task)
                write_json(task_dir / "metrics.json", metrics)
        final["methods"][method] = {"expected_runs": 6, "completed_runs": 6,
                                    "healthy_runs": 5 if method == "vert" else 6,
                                    "selected_candidate": selected[method],
                                    "validation_selection_status": "eligible_score_selection"}
    write_json(root / "manifest.json", manifest)
    write_json(root / "final_plan.json", plan)
    write_json(root / "final_summary.json", final)
    write_csv(root / "final_results/summary.csv", summaries)
    write_csv(root / "final_results/rounds.csv", rounds)


class ResourceLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for name, value in attrs if name in ("src", "href") and value)


class ExperimentReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        make_study(self.root)

    def load(self):
        return reporting.load_completed_study(self.root)

    def test_failed_observations_are_retained_with_sample_sd(self):
        study = self.load()
        self.assertEqual(len(study["runs"]), 12)
        self.assertEqual(len(study["raw_curves"]), 48)
        self.assertEqual(study["health_failed_runs"], 1)
        data = reporting.aggregate_study(study)
        attacked = next(x for x in data["scenarios"] if x["method"] == "vert" and x["malicious_ratio"] == .4)
        self.assertEqual((attacked["n"], attacked["failed_runs"]), (3, 1))
        self.assertEqual(attacked["failure_reasons"], ["nonfinite_updates"])
        self.assertAlmostEqual(attacked["final_accuracy_mean"], .6)
        self.assertAlmostEqual(attacked["final_accuracy_sd"], .2)
        self.assertAlmostEqual(attacked["attack_asr_mean"], .295)
        self.assertAlmostEqual(attacked["attack_asr_sd"], .2)
        self.assertNotAlmostEqual(attacked["final_accuracy_sd"], statistics.pstdev([.4, .6, .8]))
        curve = next(x for x in data["curves"] if x["method"] == "vert" and x["malicious_ratio"] == .4 and x["round"] == 2)
        self.assertAlmostEqual(curve["accuracy_mean"], .55)
        self.assertAlmostEqual(curve["accuracy_sd"], .2)
        clean = next(x for x in data["scenarios"] if x["malicious_ratio"] == 0)
        self.assertIsNone(clean["attack_asr_mean"])
        self.assertIsNone(clean["attack_asr_sd"])

    def test_single_seed_has_no_sd_and_keeps_its_failure(self):
        self.assertEqual(reporting._mean_sd([.75]), (.75, None))
        data = reporting.aggregate_study(self.load(), [903])
        self.assertEqual(len(data["runs"]), 4)
        self.assertEqual(data["health_failed_runs"], 1)
        for row in data["scenarios"] + data["curves"]:
            self.assertEqual(row["n"], 1)
            self.assertTrue(all(value is None for name, value in row.items() if name.endswith("_sd")))
        self.assertAlmostEqual(data["scenarios"][0]["final_accuracy_mean"], .8)

    def test_rejects_missing_round(self):
        path = self.root / "final_results/rounds.csv"
        write_csv(path, read_csv(path)[1:])
        with self.assertRaisesRegex(reporting.ReportIncompleteError, "Round CSV is incomplete"):
            self.load()

    def test_rejects_seed_absent_from_merged_summary(self):
        path = self.root / "final_results/summary.csv"
        write_csv(path, [row for row in read_csv(path) if row["seed"] != "903"])
        with self.assertRaisesRegex(reporting.ReportIncompleteError, "missing or extra runs"):
            self.load()

    def test_rejects_seed_dropped_from_plan_and_observations(self):
        plan = read_json(self.root / "final_plan.json")
        plan["tasks"] = [t for t in plan["tasks"] if t["config"]["seed"] != 903]
        write_json(self.root / "final_plan.json", plan)
        final = read_json(self.root / "final_summary.json")
        final["tasks"] = [t for t in final["tasks"] if not t["task_id"].endswith("_903")]
        write_json(self.root / "final_summary.json", final)
        for name in ("summary.csv", "rounds.csv"):
            path = self.root / "final_results" / name
            write_csv(path, [row for row in read_csv(path) if row["seed"] != "903"])
        with self.assertRaisesRegex(reporting.ReportIncompleteError, "full seed/scenario matrix"):
            self.load()

    def test_rejects_duplicate_observation(self):
        for filename, message in (("summary.csv", "Duplicate summary row"), ("rounds.csv", "Duplicate round observation")):
            with self.subTest(filename=filename):
                make_study(self.root)
                path = self.root / "final_results" / filename
                rows = read_csv(path)
                write_csv(path, rows + [rows[0]])
                with self.assertRaisesRegex(reporting.ReportIncompleteError, message):
                    self.load()

    def test_rejects_nan_or_infinity_in_metrics(self):
        for filename, field, value in (("rounds.csv", "accuracy", "nan"),
                                       ("rounds.csv", "attack_target_success_rate", "inf"),
                                       ("summary.csv", "runtime_seconds", "nan")):
            with self.subTest(filename=filename, field=field):
                make_study(self.root)
                path = self.root / "final_results" / filename
                rows = read_csv(path)
                rows[0][field] = value
                write_csv(path, rows)
                with self.assertRaisesRegex(reporting.ReportIncompleteError, "Non-finite observation"):
                    self.load()

    def test_rejects_task_identity_mismatch(self):
        task = read_json(self.root / "final_plan.json")["tasks"][0]
        path = self.root / "tasks" / task["task_id"] / "task.json"
        task["candidate"]["candidate_id"] = "a-different-candidate"
        write_json(path, task)
        with self.assertRaisesRegex(reporting.ReportIncompleteError, "Task identity differs"):
            self.load()

    def test_rejects_observed_training_config_mismatch(self):
        path = self.root / "final_results/summary.csv"
        rows = read_csv(path)
        rows[0]["lr"] = ".1"
        write_csv(path, rows)
        with self.assertRaisesRegex(reporting.ReportIncompleteError, "Observation/plan mismatch.*lr"):
            self.load()

    def test_rejects_incomplete_execution_even_when_all_csv_rows_exist(self):
        path = self.root / "final_summary.json"
        final = read_json(path)
        final["full_execution_completed"] = False
        write_json(path, final)
        with self.assertRaisesRegex(reporting.ReportIncompleteError, "Formal execution is incomplete"):
            self.load()

    def test_rejects_failed_task_despite_global_completed_flag(self):
        path = self.root / "final_summary.json"
        final = read_json(path)
        final["tasks"][0]["status"] = "failed"
        final["tasks"][0]["error"] = "worker interrupted"
        write_json(path, final)
        with self.assertRaisesRegex(reporting.ReportIncompleteError, "Task not completed"):
            self.load()

    def test_rejects_missing_failure_disclosure(self):
        path = self.root / "final_summary.json"
        final = read_json(path)
        final["health_failures"] = []
        write_json(path, final)
        with self.assertRaisesRegex(reporting.ReportIncompleteError, "Health failure list differs"):
            self.load()

    def test_html_is_offline_and_discloses_failed_observations(self):
        data = reporting.aggregate_study(self.load())
        figures = [{"name": "accuracy", "title": "Accuracy <script>unsafe</script>",
                    "svg": "accuracy.svg", "png": "accuracy.png"}]
        html = reporting._html(data, figures, mean_page=True)
        parser = ResourceLinks()
        parser.feed(html)
        self.assertTrue(parser.links)
        self.assertTrue(all(not urlsplit(link).scheme and not urlsplit(link).netloc for link in parser.links))
        self.assertNotIn("<script", html)
        self.assertNotIn("@import", html)
        self.assertIn("&lt;script&gt;unsafe&lt;/script&gt;", html)
        self.assertIn("ddof=1", html)
        self.assertIn("1 组未通过健康检查", html)
        self.assertIn("final_vert_0.4_903", html)
        self.assertIn("进程峰值常驻内存", html)
        self.assertIn("并非关闭密码模块后重跑", html)

    def test_v4_html_exposes_effective_gate_and_unassessed_reference(self):
        manifest = read_json(self.root / "manifest.json")
        manifest["schema_version"] = 4
        manifest["spec"]["performance_target"] = {
            "accuracy_gap": .02, "asr_gap": .01, "max_asr": .05,
            "max_peak_asr": .2, "tail_rounds": 10}
        write_json(self.root / "manifest.json", manifest)
        final = read_json(self.root / "final_summary.json")
        final["validation_mnist_target_gate"] = {
            "status": "passed", "pass_route": "mnist_absolute_target_without_scorable_vert",
            "comparison_scope": "ours_and_available_healthy_selected_baselines"}
        final["validation_ours_target"] = {
            "relative_target_status": "unassessed", "mean_dual_best_passed": True,
            "full_target_passed": False}
        write_json(self.root / "final_summary.json", final)
        html = reporting._html(reporting.aggregate_study(self.load()), [], mean_page=True)
        for text in ("MNIST 逐场景目标", "ASR ≤ 5%", "峰值 ≤ 20%", "末尾 10 轮",
                     "准确率落后 ≤ 2 个百分点", "ASR 高出 ≤ 1 个百分点",
                     "mnist_absolute_target_without_scorable_vert", "unassessed",
                     "不是正式测试最优性的保证", "不能宣称完整相对目标已通过"):
            self.assertIn(text, html)

    def test_staging_failure_preserves_existing_report_and_raw_data(self):
        destination = self.root / "final_results"
        old_files = {destination / "visualizations.html": b"old complete index",
                     destination / "mean_plots/accuracy.svg": b"old complete figure",
                     destination / "mean_plots/mean_figures.pdf": b"old complete pdf"}
        for path, content in old_files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        raw_files = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file() and path not in old_files}
        calls = []

        def failing_renderer(data, directory, **kwargs):
            calls.append(tuple(data["seeds"]))
            (directory / "accuracy.svg").write_text("partial replacement", encoding="utf-8")
            if len(calls) == 2:
                raise RuntimeError("synthetic seed-render failure")
            return [{"name": "accuracy", "title": "Accuracy", "svg": "accuracy.svg", "png": "accuracy.png"}]

        fake_plots = SimpleNamespace(render_figures=failing_renderer)
        with mock.patch.object(reporting, "check_dependencies"), mock.patch.dict("sys.modules", {"experiment_report_plots": fake_plots}):
            with self.assertRaisesRegex(RuntimeError, "synthetic seed-render failure"):
                reporting.generate_report(self.root)
        self.assertEqual(calls, [(901, 902, 903), (901,)])
        for path, content in {**old_files, **raw_files}.items():
            self.assertEqual(path.read_bytes(), content, str(path))
        self.assertEqual(list(destination.glob(".report-*")), [])
        self.assertFalse((destination / "seed_901").exists())

    def test_rejects_entire_method_missing_even_without_declared_run_count(self):
        manifest_path = self.root / "manifest.json"
        manifest = read_json(manifest_path)
        manifest.pop("final_run_count")
        write_json(manifest_path, manifest)
        plan_path = self.root / "final_plan.json"
        plan = read_json(plan_path)
        plan["tasks"] = [t for t in plan["tasks"] if t["method"] == "sm9rrs"]
        write_json(plan_path, plan)
        task_ids = {t["task_id"] for t in plan["tasks"]}
        final_path = self.root / "final_summary.json"
        final = read_json(final_path)
        final["tasks"] = [t for t in final["tasks"] if t["task_id"] in task_ids]
        write_json(final_path, final)
        for name in ("summary.csv", "rounds.csv"):
            path = self.root / "final_results" / name
            write_csv(path, [r for r in read_csv(path) if r["method"] == "sm9rrs"])
        with self.assertRaisesRegex(reporting.ReportIncompleteError, "missing a declared method"):
            self.load()

    def test_source_change_during_render_does_not_publish_new_report(self):
        index = self.root / "final_results/visualizations.html"
        index.write_text("old complete report", encoding="utf-8")

        def changing_renderer(data, directory, **kwargs):
            path = self.root / "final_summary.json"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            return []

        fake_plots = SimpleNamespace(render_figures=changing_renderer)
        with mock.patch.object(reporting, "check_dependencies"), mock.patch.dict("sys.modules", {"experiment_report_plots": fake_plots}):
            with self.assertRaisesRegex(reporting.ReportIncompleteError, "Source results changed"):
                reporting.generate_report(self.root)
        self.assertEqual(index.read_text(encoding="utf-8"), "old complete report")
        self.assertFalse((self.root / "final_results/mean_plots").exists())


if __name__ == "__main__":
    unittest.main()
