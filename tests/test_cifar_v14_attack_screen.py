import unittest
from dataclasses import replace

from run_cifar_v14_attack_screen import DEFAULT_CONFIG, load_screen, screen_report
from sm9rrsfl import experiments, fair_tuning
from sm9rrsfl.fl import ExperimentResult, RoundRecord


class AttackScreenTest(unittest.TestCase):
    def setUp(self):
        self.payload, self.spec, self.args, self.tasks, self.work = load_screen(DEFAULT_CONFIG)

    def executions(self):
        executions = []
        for task in self.tasks:
            c = task.config
            rows = [RoundRecord(
                method=c.method, malicious_ratio=c.malicious_ratio, round=i,
                accuracy=.7, error=.3, accepted_updates=50, rejected_updates=0,
                blacklisted_clients=0, true_positive_revocations=0,
                false_positive_revocations=0, krum_selected_client="",
                attack_target_success_rate=.04, attack_active=i >= 25,
            ) for i in range(51)]
            result = ExperimentResult(
                config=c, records=rows, final_accuracy=.7, final_error=.3,
                stopped_round=50, malicious_clients=tuple(
                    f"client-{i}" for i in range(round(50 * c.malicious_ratio))),
                blacklisted_clients=(),
            )
            executions.append((task, result))
        return executions

    def report(self, runs):
        return screen_report(self.tasks, runs, self.spec.performance_target)

    def test_tasks_match_original_candidates_with_only_requested_training_changes(self):
        original = fair_tuning.load_fair_tuning_config(
            DEFAULT_CONFIG.parent / "cifar10_v14_fair_grid.json")
        from sm9rrsfl.config_runner import parameters_to_argv
        args = experiments.parse_args(parameters_to_argv(original.shared_parameters))
        all_tasks = fair_tuning.build_validation_tasks(original, experiments.build_experiment_configs(args))
        expected = {replace(t.config, rounds=50, num_clients=50) for t in all_tasks
                    if t.candidate_id in self.payload["candidate_ids"].values()
                    and t.config.seed == 301 and t.config.malicious_ratio in (.6, .8)}
        self.assertEqual({t.config for t in self.tasks}, expected)
        self.assertEqual(len(self.tasks), 12)
        self.assertEqual(self.work["validation_rounds"], 600)
        self.assertEqual(self.work["final_tasks"], 0)
        self.assertEqual((self.tasks[0].method, self.tasks[0].config.partition,
                          self.tasks[0].config.malicious_ratio), ("sm9rrs", "dirichlet", .8))

    def test_missing_baseline_never_marks_screen_complete(self):
        self.assertEqual(self.report([])["attack_screen_status"], "pending")
        self.assertEqual(self.report(self.executions()[:-1])["attack_screen_status"], "pending")

    def test_100_client_screen_changes_only_client_count_and_output(self):
        payload, spec, args, tasks, work = load_screen(
            DEFAULT_CONFIG.parent / "cifar10_v14_attack_screen_100clients.json")
        changed = {k for k in self.payload if self.payload[k] != payload[k]}
        self.assertEqual(changed, {"num_clients", "output_dir"})
        self.assertEqual([t.config for t in tasks],
                         [replace(t.config, num_clients=100) for t in self.tasks])
        self.assertEqual(work["total_tasks"], 12)
        self.assertEqual(work["validation_rounds"], 600)
        report = screen_report(tasks, [], spec.performance_target)
        self.assertEqual(report["run_dimensions"],
                         {"num_clients": [100], "rounds": [50], "seed": [301]})
        self.assertNotIn("50-client", report["interpretation"])

    def test_numerical_audit_preserves_first_round_and_cumulative_count(self):
        runs = self.executions()
        task, result = runs[0]
        rows = list(result.records)
        rows[25] = replace(rows[25], nonfinite_updates=3)
        rows[26] = replace(rows[26], nonfinite_updates=2)
        runs[0] = task, replace(result, records=rows, nonfinite_updates=5)
        report = self.report(runs)
        audit = report["numerical_audit"][0]
        self.assertEqual(audit["nonfinite_updates"], 5)
        self.assertEqual(audit["first_nonfinite_round"], 25)
        self.assertEqual(audit["max_nonfinite_updates_per_round"], 3)
        self.assertEqual(report["attack_screen_status"], "unmet")

    def test_attack_pass_never_claims_main_experiment_or_clean_qualification(self):
        report = self.report(self.executions())
        self.assertEqual(report["attack_screen_status"], "passed")
        self.assertEqual(report["main_experiment_target_status"], "not_evaluated")
        self.assertEqual(report["original_target_audit"]["status"], "unmet")
        self.assertIn("missing_clean_or_attacked_scenarios",
                      report["original_target_audit"]["failures"])

    def test_early_attack_peak_is_not_hidden_by_good_tail(self):
        runs = self.executions()
        task, result = runs[0]
        rows = list(result.records)
        rows[25] = replace(rows[25], attack_target_success_rate=.9)
        runs[0] = task, replace(result, records=rows)
        report = self.report(runs)
        self.assertEqual(report["attack_screen_status"], "unmet")
        self.assertTrue(any("peak_asr" in row["failures"]
                            for row in report["original_target_audit"]["scenarios"]))

    def test_missing_round_or_nonfinite_baseline_rejects_pass(self):
        runs = self.executions()
        task, result = runs[-1]
        runs[-1] = task, replace(result, records=result.records[:25] + result.records[26:])
        self.assertEqual(self.report(runs)["attack_screen_status"], "unmet")
        runs[-1] = task, replace(result, nonfinite_updates=1)
        self.assertEqual(self.report(runs)["attack_screen_status"], "unmet")

    def test_health_failure_rejects_numerically_good_attack_metrics(self):
        runs = self.executions()
        task, result = runs[0]
        rows = list(result.records)
        rows[-1] = replace(rows[-1], false_positive_revocations=10)
        runs[0] = task, replace(result, records=rows)
        report = self.report(runs)
        self.assertEqual(report["attack_screen_status"], "unmet")
        self.assertIn("all_honest_revoked", report["health_failures"][0]["reasons"])


if __name__ == "__main__":
    unittest.main()
