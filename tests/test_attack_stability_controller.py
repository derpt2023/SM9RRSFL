"""Controller policy tests without launching training or child processes."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_cifar_attack_stability as runner


def metric_rows(*, hit=1., confidence=.9, accuracy=.6):
    return [{
        "round": r, "training_target_success": hit,
        "training_target_confidence": confidence, "accuracy": accuracy,
        "source_transfer_asr": .3, "legacy_asr": 0.,
        "nonfinite_updates": 0, "cumulative_nonfinite_updates": 0,
        "training_target_count": 1, "source_transfer_count": 750,
        "legacy_target_count": 1,
    } for r in range(25, 31)]


class EffectivenessTests(unittest.TestCase):
    def setUp(self):
        self.policy = runner.load_protocol()[0]["effectiveness"]
        self.attacked = metric_rows()
        self.clean = metric_rows(hit=0., confidence=.1)

    def evaluate(self):
        return runner.effectiveness(self.attacked, self.clean, self.policy)

    def test_effective_targeted_attack_uses_training_target_not_transfer_or_legacy(self):
        for row in self.attacked:
            row["source_transfer_asr"] = 0.
            row["legacy_asr"] = 0.
        result = self.evaluate()
        self.assertTrue(result["effective"])
        self.assertEqual(result["rounds"], list(range(25, 31)))
        self.assertEqual(result["training_target_counts"], [1])
        self.assertEqual(result["source_transfer_counts"], [750])
        self.assertEqual(result["legacy_target_counts"], [1])

    def test_missing_duplicate_and_reordered_rounds_fail(self):
        for rows in (self.attacked[:-1], self.attacked + [self.attacked[-1]],
                     list(reversed(self.attacked))):
            result = runner.effectiveness(rows, self.clean, self.policy)
            self.assertFalse(result["effective"])
            self.assertEqual(result["reason"], "incomplete_paired_attack_window")

    def test_nonfinite_metric_or_update_count_fails(self):
        for field, value in (("accuracy", float("nan")),
                             ("training_target_confidence", float("inf")),
                             ("nonfinite_updates", 1)):
            with self.subTest(field=field):
                attacked = deepcopy(self.attacked)
                attacked[2][field] = value
                self.assertFalse(runner.effectiveness(
                    attacked, self.clean, self.policy)["effective"])

    def test_preexisting_clean_target_error_does_not_count_as_success(self):
        self.clean = metric_rows(hit=1., confidence=.85)
        result = self.evaluate()
        self.assertFalse(result["effective"])
        self.assertEqual(result["reason"], "clean_already_high_confidence_target_error")

    def test_confidence_gain_and_hit_rate_both_required(self):
        for hit, confidence in ((.5, .95), (1., .2)):
            result = runner.effectiveness(
                metric_rows(hit=hit, confidence=confidence), self.clean, self.policy)
            self.assertFalse(result["effective"])
            self.assertEqual(result["reason"], "attack_effect_unproven")

    def test_cumulative_failure_and_invalid_metric_denominators_fail(self):
        for field, value in (
                ("cumulative_nonfinite_updates", 1),
                ("training_target_count", 0),
                ("source_transfer_count", 0),
                ("source_transfer_count", 749),
                ("legacy_target_count", 2),
                ("training_target_success", 1.1),
                ("training_target_confidence", -.01)):
            attacked = deepcopy(self.attacked)
            attacked[2][field] = value
            with self.subTest(field=field, value=value):
                self.assertFalse(runner.effectiveness(
                    attacked, self.clean, self.policy)["effective"])

    def test_exact_policy_boundaries_are_not_changed_by_roundoff(self):
        self.attacked = metric_rows(hit=1., confidence=.3)
        self.assertTrue(self.evaluate()["effective"])
        self.clean = metric_rows(hit=1., confidence=.8)
        self.attacked = metric_rows(hit=1., confidence=1.)
        result = self.evaluate()
        self.assertFalse(result["effective"])
        self.assertEqual(result["reason"], "clean_already_high_confidence_target_error")


class ProtocolAndRankingTests(unittest.TestCase):
    def test_original_shared_training_and_defense_candidates_are_preserved(self):
        protocol, configs, data = runner.load_protocol()
        source = json.loads((runner.ROOT / protocol["source_config"]).read_text())
        shared = source["shared_parameters"]
        common = (
            "local_epochs", "batch_size", "lr", "lr_decay", "attack",
            "attack_epochs", "attack_distance_weight", "attack_source_label",
            "attack_target_label", "attack_start_round", "crypto_mode",
            "dkg_nodes", "dkg_threshold", "dirichlet_alpha",
        )
        for method, config in configs.items():
            for field in common:
                self.assertEqual(getattr(config, field), shared[field], (method, field))
            self.assertEqual((config.num_clients, config.rounds, config.seed), (100, 30, 301))
            self.assertEqual(config.detector_window, shared["K"])
            self.assertEqual(config.attack_target_count, 1)
            self.assertEqual(config.device, "mps")
            self.assertEqual(config.eval_interval, 1)
            self.assertFalse(config.early_stop)
        self.assertEqual(configs["sm9rrs"].detector_distance_threshold, 2.)
        self.assertEqual((configs["vert"].vert_history_window,
                          configs["vert"].vert_projection_dim,
                          configs["vert"].vert_predict_epochs,
                          configs["vert"].vert_top_k), (5, 20, 5, 5))
        self.assertEqual(data["validation_fraction"], .15)
        self.assertEqual(data["split_seed"], 20261013)
        self.assertEqual((data["train_samples"], data["test_samples"]), (50000, 10000))

    def test_profile_order_and_unapproved_budget_are_rejected(self):
        protocol = runner.load_protocol()[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for change in (
                    {"budget_seconds": 7201},
                    {"profiles": list(reversed(protocol["profiles"]))},
                    {"num_clients": 50},
                    {"candidate_ids": {**protocol["candidate_ids"], "vert": "vert-002"}}):
                path.write_text(json.dumps({**protocol, **change}))
                with self.subTest(change=list(change)), self.assertRaises(ValueError):
                    runner.load_protocol(path)

    def probe_results(self):
        return [{
            "task_id": f"C1_{partition}_r{anchor}_cpu", "profile_id": "C1",
            "partition": partition, "anchor_id": f"r{anchor}",
            "status": "complete", "all_finite": True, "requested_clients": 16,
            "actual_results": [{"status": "complete", "all_finite": True,
                                "client_idx": i} for i in range(16)],
            "mean_target_success": 1., "mean_target_confidence": .9,
            "runtime_seconds": 10.,
        } for anchor in (0, 24) for partition in ("dirichlet", "iid")]

    def test_incomplete_or_failed_probes_never_rank(self):
        good = self.probe_results()
        self.assertEqual(runner.micro_rank(good)["target_hit_rate"], 1.)
        self.assertIsNone(runner.micro_rank(good[:-1]))
        for change in ({"status": "partial"}, {"status": "numerical_failure"},
                       {"all_finite": False}, {"actual_results": []}):
            rows = deepcopy(good)
            rows[1].update(change)
            self.assertIsNone(runner.micro_rank(rows))

    def test_duplicate_probe_cells_or_mixed_profiles_cannot_rank(self):
        good = self.probe_results()
        self.assertIsNone(runner.micro_rank([good[0]] * 4))
        mixed = deepcopy(good)
        mixed[-1]["profile_id"] = "C2"
        self.assertIsNone(runner.micro_rank(mixed))

    def test_failed_first_probe_stops_candidate_without_claiming_other_anchors(self):
        from scripts.experiments.cifar10 import attack_stability_probes
        failure = {"status": "numerical_failure", "all_finite": False}
        with mock.patch.object(attack_stability_probes, "run_probe",
                               return_value=failure) as run:
            results = runner.candidate_probe_job([{"id": i} for i in range(4)])
        self.assertEqual(results, [failure])
        self.assertEqual(run.call_count, 1)
        self.assertIsNone(runner.micro_rank(results))

    def test_source_fingerprint_changes_with_scientific_source_or_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            entry = root / "run_cifar_attack_stability.py"
            entry.write_text("# entry\n")
            (root / "sm9rrsfl").mkdir()
            scientific = root / "sm9rrsfl" / "model.py"
            scientific.write_text("VERSION = 1\n")
            (root / "configs").mkdir()
            (root / "configs/source.json").write_text("{}")
            protocol = {"source_config": "configs/source.json", "profiles": ["C0"]}
            with mock.patch.object(runner, "ROOT", root), \
                    mock.patch.object(runner, "__file__", str(entry)):
                first = runner.source_manifest(protocol)
                second = runner.source_manifest(protocol)
                self.assertEqual(first, second)
                self.assertFalse(first["official_test_used_for_selection"])
                scientific.write_text("VERSION = 2\n")
                self.assertNotEqual(first["fingerprint"],
                                    runner.source_manifest(protocol)["fingerprint"])
                changed = {**protocol, "profiles": ["C1"]}
                self.assertNotEqual(runner.source_manifest(protocol)["fingerprint"],
                                    runner.source_manifest(changed)["fingerprint"])


class RecoveryBudgetTests(unittest.TestCase):
    def create_controller(self, previous=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        protocol = {"output_dir": "screen", "budget_seconds": 7200}
        if previous is not None:
            runner.write_json(root / "screen/controller_status.json", previous)
        with mock.patch.object(runner, "ROOT", root), \
                mock.patch.object(runner, "source_manifest", return_value={"fingerprint": "stable"}), \
                mock.patch.object(runner.time, "time", return_value=1000.):
            controller = runner.Controller(protocol, {}, {}, mode="log")
        self.addCleanup(controller.lock.close)
        return controller

    def test_clean_resume_preserves_consumed_budget_and_excludes_pause_time(self):
        controller = self.create_controller(
            {"status": "interrupted", "elapsed_seconds": 123., "updated_epoch": 800.})
        self.assertEqual(controller.elapsed_before, 123.)
        self.assertEqual(controller.deadline, 1000. + 7200. - 123.)

    def test_unclean_running_exit_does_not_reset_elapsed_time(self):
        controller = self.create_controller(
            {"status": "running", "elapsed_seconds": 123., "updated_epoch": 800.})
        self.assertEqual(controller.elapsed_before, 323.)
        self.assertEqual(controller.deadline, 1000. + 7200. - 323.)

    def test_exhausted_budget_cannot_start_another_fl_task(self):
        controller = self.create_controller(
            {"status": "interrupted", "elapsed_seconds": 7300., "updated_epoch": 800.})
        with mock.patch.object(runner.time, "time", return_value=1000.), \
                mock.patch.object(controller, "subprocess_task") as launch:
            result = controller.run_fl("new", "fedavg", "iid", .8)
        launch.assert_not_called()
        self.assertEqual(result["status"], "not_started")
        self.assertEqual(result["reason"], "insufficient_budget")

    def test_finished_task_is_reused_without_spending_training_budget(self):
        controller = self.create_controller()
        existing = {"status": "complete", "task_id": "old", "completed_round": 30}
        runner.write_json(controller.out / "tasks/old/status.json", existing)
        with mock.patch.object(controller, "subprocess_task") as launch:
            result = controller.run_fl("old", "fedavg", "iid", 0.)
        self.assertEqual(result, existing)
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
