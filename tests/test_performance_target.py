import unittest
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sm9rrsfl.fl import ExperimentConfig, ExperimentResult, RoundRecord
from sm9rrsfl.performance_target import PerformanceTarget, evaluate_target, scenario_key, select_towards_target
from sm9rrsfl.ours_policy import bounded_candidates


def run(ratio=.4, accuracy=.9, asr=0., seed=1, rates=None):
    rates = rates if rates is not None else [asr] * 40
    config = ExperimentConfig(num_clients=20, rounds=len(rates), attack_start_round=1,
                              malicious_ratio=ratio, seed=seed)
    records = [RoundRecord("sm9rrs", ratio, i + 1, accuracy, 1 - accuracy, 20, 0, 0, 0, 0, "",
                           attack_target_success_rate=value) for i, value in enumerate(rates)]
    return ExperimentResult(config, records, accuracy, 1 - accuracy, len(rates), (), ())


def trial(name, score=.8, asr=0., valid=True, method="sm9rrs"):
    return SimpleNamespace(candidate_id=name, method=method, score=score, attack_success_rate=asr,
                           robust_accuracy=.9, valid=valid, invalid_reasons=() if valid else ("health",))


class PerformanceTargetTest(unittest.TestCase):
    def setUp(self):
        self.policy = PerformanceTarget()
        self.reference = [run(0.), run()]

    def test_near_accuracy_and_equal_zero_asr_pass(self):
        result = evaluate_target([run(0.), run(accuracy=.885)], self.reference, self.policy)
        self.assertEqual(result["status"], "passed")

    def test_relative_win_with_high_asr_is_not_protection(self):
        result = evaluate_target([run(0.), run(asr=.3)], [run(0.), run(asr=.8)], self.policy)
        self.assertEqual(result["status"], "unmet")
        self.assertIn("attack_mean:absolute_asr", result["scenarios"][1]["failures"])

    def test_peak_damage_cannot_hide_in_mean_or_recovered_tail(self):
        result = evaluate_target([run(0.), run(rates=[1.] + [0.] * 39)], self.reference, self.policy)
        attacked = result["scenarios"][1]
        self.assertLess(attacked["windows"]["attack_mean"]["ours_asr"], self.policy.max_asr)
        self.assertIn("peak_asr", attacked["failures"])

    def test_late_failure_cannot_hide_in_full_run_mean(self):
        result = evaluate_target([run(0.), run(rates=[0.] * 39 + [.1])], self.reference, self.policy)
        self.assertIn("final:absolute_asr", result["scenarios"][1]["failures"])

    def test_one_bad_seed_is_not_hidden_by_an_average(self):
        ours = [run(0.), run(accuracy=.93), run(0., seed=2), run(accuracy=.86, seed=2)]
        ref = self.reference + [run(0., seed=2), run(seed=2)]
        self.assertEqual(evaluate_target(ours, ref, self.policy)["status"], "unmet")

    def test_missing_both_methods_seed_is_detected_from_expected_matrix(self):
        expected = {scenario_key(r) for r in self.reference + [run(0., seed=2), run(seed=2)]}
        result = evaluate_target(self.reference, self.reference, self.policy, expected_scenarios=expected)
        self.assertFalse(result["structurally_complete"])

    def test_duplicate_or_missing_metric_cannot_pass(self):
        self.assertEqual(evaluate_target(self.reference + [run()], self.reference, self.policy)["status"], "unmet")
        self.assertEqual(evaluate_target([run(0.), run(asr=None)], self.reference, self.policy)["status"], "unmet")

    def test_different_attack_strength_is_not_a_paired_comparison(self):
        attacked = run()
        changed = replace(attacked, config=replace(attacked.config, attack_boost=99.))
        audit = evaluate_target([run(0.), changed], self.reference, self.policy)
        self.assertFalse(audit["structurally_complete"])
        self.assertEqual(audit["status"], "unmet")

    def test_target_selects_healthy_protection_over_high_score_and_freezes_vert(self):
        bad, good, unhealthy, vert = trial("bad", .99, .6), trial("good", .8), trial("unhealthy", 1., valid=False), trial("vert", method="vert")
        selected = {"sm9rrs": bad, "vert": vert}
        results = {"bad": [run(0.), run(asr=.6)], "good": self.reference,
                   "unhealthy": self.reference, "vert": self.reference}
        chosen, report = select_towards_target([bad, good, unhealthy, vert], selected, results, self.policy)
        self.assertIs(chosen["sm9rrs"], good)
        self.assertIs(chosen["vert"], vert)
        self.assertIs(selected["sm9rrs"], bad)
        self.assertEqual(report["status"], "passed")

    def test_no_goal_match_keeps_closest_and_reports_unmet(self):
        bad, closer, vert = trial("bad", .99), trial("closer", .7), trial("vert", method="vert")
        chosen, report = select_towards_target([bad, closer, vert], {"sm9rrs": bad, "vert": vert},
            {"bad": [run(0.), run(asr=.9)], "closer": [run(0.), run(asr=.2)], "vert": self.reference}, self.policy)
        self.assertIs(chosen["sm9rrs"], closer)
        self.assertEqual(report["status"], "unmet")

    def test_final_audit_is_read_only_and_does_not_change_validation_selection(self):
        chosen = {"sm9rrs": trial("frozen"), "vert": trial("vert", method="vert")}
        before = chosen.copy()
        report = evaluate_target([run(0.), run(asr=1.)], self.reference, self.policy)
        self.assertEqual(chosen, before)
        self.assertEqual(report["status"], "unmet")

    def test_config_rejects_invalid_or_misspelled_limits(self):
        for payload in ({"accuracy_gap": -1}, {"max_asr": True}, {"max_asr": float("nan")},
                        {"tail_rounds": 0}, {"tail_rounds": 1.5}, {"max_asr": .5}, {"asr": .1}):
            with self.assertRaises(ValueError):
                PerformanceTarget.parse(payload)

    def test_budget_retains_measured_c2_and_unique_broad_candidates(self):
        candidates = bounded_candidates(12)
        self.assertEqual(candidates[0]["suspicion_remove_after"], 2)
        self.assertEqual(candidates[0]["detector_distance_threshold"], 3.)
        self.assertEqual(candidates[0]["detector_subspace_dim"], 2)
        self.assertEqual(len({tuple(sorted(c.items())) for c in bounded_candidates(36)}), 36)

    def test_launcher_persists_validation_goal_and_audits_final_without_reselection(self):
        from sm9rrsfl.datasets import make_synthetic_mnist_like
        from sm9rrsfl.fair_tuning import load_fair_tuning_config, run_fair_tuning
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "configs/fair_tuning.example.json").read_text())
        payload["shared_parameters"].update(client_counts=[20], partitions=["iid"],
            ratio_range=[0, .4, 2], attack_target_count=2, crypto_mode="simulated",
            visualizations=False, progress=False, calibration_candidate_budget=1)
        payload["tuning"].update(trials_per_tunable_method=1, validation_seeds=[1], final_seeds=[2])
        for method in ("vert", "alignins"):
            payload["tuning"]["method_spaces"][method] = {
                k: [v[0]] for k, v in payload["tuning"]["method_spaces"][method].items()}
        calls = []
        def execute(dataset, tasks, args, **kwargs):
            calls.append(tasks)
            executions = []
            for task in tasks:
                c = task.config
                asr = .4 if task.phase == "final" and c.method == "sm9rrs" and c.malicious_ratio else 0.
                records = [RoundRecord(c.method, c.malicious_ratio, i, .9, .1, 20, 0, 0, 0, 0, "",
                    attack_target_success_rate=asr if i >= c.attack_start_round else 0.,
                    honest_weight_loss=0., malicious_weight_mass=0.) for i in range(1, c.rounds + 1)]
                executions.append((task, ExperimentResult(c, records, .9, .1, c.rounds, (), ())))
            return executions, tasks[0].phase + "-fixture"
        def prepare(dataset, tasks, args, **kwargs):
            return tasks, 1, "numpy", 1
        with tempfile.TemporaryDirectory() as folder:
            payload["shared_parameters"]["output_dir"] = folder
            config = Path(folder) / "config.json"
            config.write_text(json.dumps(payload))
            spec = load_fair_tuning_config(config)
            dataset = make_synthetic_mnist_like(train_samples=1000, test_samples=100)
            with mock.patch("sm9rrsfl.fair_tuning.load_image_dataset", return_value=dataset), \
                 mock.patch("sm9rrsfl.fair_tuning.prepare_tuning_tasks", side_effect=prepare), \
                 mock.patch("sm9rrsfl.fair_tuning.execute_resumable_tuning_phase", side_effect=execute), \
                 mock.patch("sm9rrsfl.fair_tuning.print_resource_plan"), mock.patch("builtins.print"):
                chosen = run_fair_tuning(spec)
            validation = json.loads((Path(folder) / "performance_target_validation.json").read_text())
            final = json.loads((Path(folder) / "final_evaluation/performance_target.json").read_text())
            self.assertEqual(validation["status"], "passed")
            self.assertEqual(final["status"], "unmet")
            self.assertFalse(final["parameters_reselected"])
            self.assertFalse(final["selection_used_final_data"])
            self.assertEqual(len(calls), 2)
            for task in calls[-1]:
                self.assertEqual(task.candidate_id, chosen[task.method].candidate_id)


if __name__ == "__main__":
    unittest.main()
