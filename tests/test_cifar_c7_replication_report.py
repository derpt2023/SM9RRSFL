"""Integrity gates and paired comparisons use public JSON fixtures only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.experiments.cifar10 import cifar_c7_replication_report as report


def config(method, ratio, seed):
    return {
        "method": method, "malicious_ratio": ratio, "seed": seed,
        "partition": "dirichlet", "dirichlet_alpha": .5,
        "num_clients": 100, "rounds": 30, "local_epochs": 1, "batch_size": 50,
        "lr": .01, "lr_decay": .99, "detector_window": 20,
        "attack": "alternating_minimization", "attack_start_round": 25,
        "attack_source_label": 5, "attack_target_label": 7, "attack_target_count": 1,
        "attack_epochs": 1, "attack_distance_weight": .0001,
        "eval_interval": 1, "early_stop": False, "compute_backend": "torch",
        "sm9_workers": 8,
    }


def write_task(root, task_id, spec):
    method, ratio, seed = spec
    directory = root / "tasks" / task_id
    directory.mkdir(parents=True)
    identity = {
        "protocol": "cifar_attack_stability_task_v1", "root_fingerprint": "source-v1",
        "config": config(*spec), "profile": copy.deepcopy(report.PROFILE) if ratio else None,
        "data_config": {
            "dataset": "cifar10", "data_dir": "/fixed/data", "seed": 42,
            "split_seed": 20261013, "train_samples": 50000,
            "test_samples": 10000, "validation_fraction": .15,
        },
    }
    identity["fingerprint"] = report._digest(identity)
    fingerprint = identity["fingerprint"]
    target = 1 if seed == 301 else 2
    targets = {
        **report.COUNTS, "training_target_indices": [target + 900],
        "legacy_validation_indices": [target],
        "transfer_validation_indices": list(range(750)),
        "official_test_used_for_selection": False,
        "training_target_is_optimizer_input": True,
        "transfer_target_is_optimizer_input": False,
    }
    rows = []
    for r in range(31):
        attacked = ratio > 0 and r >= 25
        accuracy = .5 - (.1 if attacked else 0) + (.01 if seed == 401 else 0)
        rows.append({
            "round": r, "method": method, "malicious_ratio": ratio,
            "attack_active": attacked,
            "accuracy": accuracy, "training_target_success": float(attacked),
            "training_target_confidence": .9 if attacked else .1,
            "legacy_asr": float(attacked), "legacy_confidence": .8 if attacked else .1,
            "source_transfer_asr": .8 if attacked else .1,
            "source_transfer_confidence": .8 if attacked else .1,
            "honest_weight_loss": 0., "malicious_weight_mass": 0.,
            "nonfinite_updates": 0, "cumulative_nonfinite_updates": 0,
            "false_positive_revocations": 0, **report.COUNTS,
        })
    status = {
        "task_id": task_id, "status": "complete", "completed_round": 30,
        "requested_rounds": 30, "fingerprint": fingerprint,
        "elapsed_seconds": 100., "nonfinite_updates": 0,
        "final_metrics": rows[-1],
    }
    contents = {"task.json": identity, "targets.json": targets,
                "status.json": status,
                "metrics.json": {"fingerprint": fingerprint,
                                 "official_test_used_for_selection": False, "rounds": rows}}
    for name, content in contents.items():
        (directory / name).write_text(json.dumps(content))
    return directory


class ReplicationReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "new"
        self.reference = Path(self.temp.name) / "old"
        for key, spec in report.TASKS.items():
            write_task(self.root, key, spec)
        for key, spec in report.REFERENCES.items():
            write_task(self.reference, key, spec)

    def build(self):
        return report.build_report(self.root, self.reference)

    def mutate(self, task_id, filename, change, *, old=False):
        root = self.reference if old else self.root
        path = root / "tasks" / task_id / filename
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))

    def test_complete_panel_has_qualified_metrics_and_explicit_limits(self):
        result = self.build()
        self.assertEqual(result["valid_complete_tasks"], 8)
        self.assertTrue(result["all_complete"])
        self.assertTrue(result["all_numerically_finite"])
        self.assertTrue(result["reference_integrity_valid"])
        self.assertTrue(result["all_comparisons_available"])
        self.assertEqual(result["task_runtime_seconds"], 800.)
        self.assertFalse(result["seed401_matched_clean_available"])
        self.assertIn("not_evaluated", result["seed401_attack_effectiveness_qualification"])
        self.assertEqual(len(result["seed301_clean_comparisons"]), 6)
        self.assertEqual(len(result["seed_comparisons"]), 6)
        for pair in result["seed301_clean_comparisons"]:
            self.assertTrue(pair["available"])
            self.assertAlmostEqual(pair["final_accuracy_loss"], .1)
            self.assertEqual(pair["target_flips_to_attack_label"], 6)
            self.assertEqual(pair["target_flips_away_from_attack_label"], 0)
        for pair in result["seed_comparisons"]:
            self.assertAlmostEqual(pair["final_accuracy_delta"], .01)
            self.assertTrue(pair["target_selection_changed"]["training_target_indices"])
            self.assertFalse(pair["target_selection_changed"]["transfer_validation_indices"])

    def test_partial_never_receives_final_or_window_metrics(self):
        key = "C7_sm9rrs_dir80_s401"
        self.mutate(key, "status.json",
                    lambda value: value.update(status="partial", completed_round=26))
        self.mutate(key, "metrics.json",
                    lambda value: value.update(rounds=value["rounds"][:27]))
        result = self.build()
        task = result["tasks"][key]
        self.assertFalse(result["all_complete"])
        self.assertFalse(task["valid_complete"])
        self.assertEqual(task["integrity_errors"], [])
        self.assertNotIn("final_accuracy", task)
        self.assertNotIn("attack_window_mean_accuracy", task)
        pair = next(p for p in result["seed_comparisons"] if p["seed401_task_id"] == key)
        self.assertFalse(pair["available"])

    def test_complete_missing_or_duplicate_round_is_invalid(self):
        key = "C7_fedavg_dir60_s401"
        for modification in (
                lambda value: value["rounds"].pop(20),
                lambda value: value["rounds"].insert(20, copy.deepcopy(value["rounds"][20]))):
            with self.subTest(modification=modification):
                path = self.root / "tasks" / key / "metrics.json"
                original = path.read_text()
                self.mutate(key, "metrics.json", modification)
                task = self.build()["tasks"][key]
                self.assertFalse(task["valid_complete"])
                self.assertTrue(task["integrity_errors"])
                path.write_text(original)

    def test_nonfinite_metric_cannot_be_reported_as_success(self):
        key = "C7_vert_dir60_s401"
        self.mutate(key, "metrics.json",
                    lambda value: value["rounds"][27].update(accuracy=float("nan")))
        result = self.build()
        self.assertFalse(result["all_numerically_finite"])
        self.assertIn("non-finite", result["tasks"][key]["integrity_errors"][0])

    def test_wrong_denominator_and_impossible_rate_are_invalid(self):
        key = "C7_fedavg_dir80_s401"
        for change in (
                {"source_transfer_count": 749},
                {"training_target_success": .5},
                {"source_transfer_asr": .12345},
                {"accuracy": 1.01},
        ):
            with self.subTest(change=change):
                path = self.root / "tasks" / key / "metrics.json"
                original = path.read_text()
                self.mutate(key, "metrics.json",
                            lambda value: value["rounds"][28].update(change))
                self.assertFalse(self.build()["tasks"][key]["valid_complete"])
                path.write_text(original)

    def test_profile_tampering_is_rejected_even_with_recomputed_identity(self):
        key = "C7_fedavg_dir80_s401"
        def tamper(value):
            value["profile"]["boost"] = 10.
            value["fingerprint"] = report._digest(
                {k: v for k, v in value.items() if k != "fingerprint"})
        self.mutate(key, "task.json", tamper)
        task = self.build()["tasks"][key]
        self.assertFalse(task["valid_complete"])
        self.assertIn("profile", task["integrity_errors"][0])

    def test_same_seed_methods_must_use_identical_target_images(self):
        key = "C7_fedavg_dir80_s401"
        self.mutate(key, "targets.json",
                    lambda value: value.update(training_target_indices=[999]))
        result = self.build()
        for task in result["tasks"].values():
            if task["seed"] == 401:
                self.assertFalse(task["valid_complete"])
                self.assertIn("same-seed", task["integrity_errors"][0])

    def test_positive_nonfinite_counter_is_not_numerical_stability(self):
        key = "C7_sm9rrs_dir60_s401"
        def counters(value):
            for row in value["rounds"][25:]:
                row["cumulative_nonfinite_updates"] = 1
            value["rounds"][25]["nonfinite_updates"] = 1
        self.mutate(key, "metrics.json", counters)
        metrics = json.loads((self.root / "tasks" / key / "metrics.json").read_text())
        self.mutate(key, "status.json", lambda value: value.update(
            nonfinite_updates=1, final_metrics=metrics["rounds"][-1]))
        result = self.build()
        self.assertTrue(result["tasks"][key]["valid_complete"])
        self.assertFalse(result["tasks"][key]["numerically_finite"])
        self.assertFalse(result["all_numerically_finite"])

    def test_configuration_change_blocks_seed_comparison(self):
        key = "C7_vert_dir80_s401"
        def update(value):
            value["config"]["sm9_workers"] = 4
            value["fingerprint"] = report._digest(
                {k: v for k, v in value.items() if k != "fingerprint"})
        self.mutate(key, "task.json", update)
        identity = json.loads((self.root / "tasks" / key / "task.json").read_text())
        for file in ("metrics.json", "status.json"):
            self.mutate(key, file, lambda value: value.update(fingerprint=identity["fingerprint"]))
        result = self.build()
        self.assertTrue(result["tasks"][key]["valid_complete"])
        pair = next(p for p in result["seed_comparisons"] if p["seed401_task_id"] == key)
        self.assertFalse(pair["available"])
        self.assertIn("configuration differs", pair["reason"])

    def test_attack_timing_and_boolean_denominator_are_invalid(self):
        key = "C7_fedavg_dir80_s401"
        for change in ({"attack_active": False}, {"training_target_count": True}):
            with self.subTest(change=change):
                path = self.root / "tasks" / key / "metrics.json"
                original = path.read_text()
                self.mutate(key, "metrics.json",
                            lambda value: value["rounds"][28].update(change))
                self.assertFalse(self.build()["tasks"][key]["valid_complete"])
                path.write_text(original)

    def test_complete_status_and_final_row_must_agree(self):
        key = "C7_sm9rrs_dir60_s401"
        self.mutate(key, "status.json",
                    lambda value: value["final_metrics"].update(accuracy=.99))
        task = self.build()["tasks"][key]
        self.assertFalse(task["valid_complete"])
        self.assertIn("final metrics disagree", task["integrity_errors"][0])

    def test_ranking_direction_change_is_reported_without_statistical_claim(self):
        key = "C7_sm9rrs_dir60_s401"
        def improve(value):
            value["rounds"][-1]["accuracy"] = .9
        self.mutate(key, "metrics.json", improve)
        metrics = json.loads((self.root / "tasks" / key / "metrics.json").read_text())
        self.mutate(key, "status.json", lambda value: value.update(final_metrics=metrics["rounds"][-1]))
        result = self.build()
        ranking = next(item for item in result["rankings"]
                       if item["malicious_ratio"] == .6 and item["metric"] == "final_accuracy")
        self.assertFalse(ranking["all_pairwise_directions_same"])
        self.assertEqual(ranking["rankings"]["401"][0]["method"], "sm9rrs")

    def test_missing_task_and_summary_read_do_not_create_files(self):
        empty = Path(self.temp.name) / "empty"
        result = report.build_report(empty, self.reference)
        self.assertEqual(result["completed_tasks"], 0)
        self.assertFalse(empty.exists())


if __name__ == "__main__":
    unittest.main()
