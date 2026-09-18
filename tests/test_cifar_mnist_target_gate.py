"""Synthetic selection tests: no download, model training, or official results."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import unittest
from unittest import mock

import cifar_mnist_target_gate as gate
import run_cifar_six_from_scratch as base
from sm9rrsfl.calibration_policy import weighted_score
from sm9rrsfl.performance_target import PerformanceTarget
from tests.test_cifar_six_pipeline import result_matrix, synthetic_run


class MnistTargetGateTests(unittest.TestCase):
    def setUp(self):
        self.spec = base.load_spec(Path(__file__).resolve().parents[1] /
                                   "configs/cifar10_six_original_v2.json")
        self.spec["validation"] = {
            "seeds": [1001, 1002, 1003],
            "scenarios": [{"partition": p, "malicious_ratio": r}
                          for p in ("iid", "dirichlet") for r in (0., .1, .3, .5, .7)]}
        self.spec["performance_target"] = asdict(PerformanceTarget())
        self.spec["promotion"]["performance_target_is_gate"] = True
        self.spec["promotion"]["require_mean_dual_best"] = False
        self.tasks = base.build_tasks(self.spec, "validation")
        self.results = result_matrix(self.spec, self.tasks)
        self.ours = [c["candidate_id"] for c in self.spec["candidates"]["sm9rrs"]]

    def change(self, cid, accuracy=.8, asr=.05, nonfinite=0):
        self.results[cid] = [synthetic_run(run.config, accuracy=accuracy, asr=asr, nonfinite=nonfinite)
                             for run in self.results[cid]]

    def report(self):
        return gate.select_validation(self.spec, self.results, self.tasks)

    def distort_one_attacked_run_for_all_ours(self, transform):
        for cid in self.ours:
            index = next(i for i, r in enumerate(self.results[cid]) if r.config.malicious_ratio)
            old = self.results[cid][index]
            records = [transform(r) for r in old.records]
            self.results[cid][index] = replace(old, records=records, final_accuracy=records[-1].accuracy)

    def test_complete_mnist_targets_promote_all_six(self):
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertEqual(len(report["selected"]), 6)
        self.assertTrue(report["ours_mnist_target_passed"])
        self.assertEqual(report["pass_route"], "mnist_target_with_paired_vert")
        self.assertEqual(report["ours_target"]["policy"], asdict(PerformanceTarget()))
        self.assertEqual(len(report["ours_target"]["scenarios"]), 30)
        self.assertEqual(report["ours_target"]["relative_target_status"], "passed")
        self.assertTrue(report["mnist_target_gate"]["six_method_dual_optimality_assessed"])
        self.assertFalse(report["official_test_used_for_selection"])

    def test_one_seed_scenario_failure_cannot_be_hidden_in_means(self):
        self.distort_one_attacked_run_for_all_ours(
            lambda r: replace(r, attack_target_success_rate=.0501) if r.round >= 25 else r)
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_target_development")
        self.assertTrue(report["ours_health_passed"])
        self.assertNotIn("sm9rrs", report["selected"])
        target = report["ours_candidate_targets"][self.ours[0]]
        self.assertFalse(target["absolute_target_passed"])
        self.assertLess(target["mean_asr"], .05001)
        self.assertEqual(len(target["absolute_failures"]), 3)

    def test_peak_final_and_tail_are_independently_required(self):
        original = deepcopy(self.results)
        cases = {
            "peak_asr": lambda r: replace(r, attack_target_success_rate=.201 if r.round == 30 else .0),
            "final:absolute_asr": lambda r: replace(r, attack_target_success_rate=.06 if r.round == 100 else .0),
            "tail_mean:absolute_asr": lambda r: replace(r, attack_target_success_rate=.06 if 91 <= r.round < 100 else .0),
        }
        for reason, mutate in cases.items():
            with self.subTest(reason=reason):
                self.results = deepcopy(original)
                self.distort_one_attacked_run_for_all_ours(mutate)
                report = self.report()
                self.assertEqual(report["status"], "needs_ours_target_development")
                target = report["ours_candidate_targets"][self.ours[0]]
                self.assertIn(reason, [r["reason"] for r in target["absolute_failures"]])

    def test_accuracy_and_asr_gaps_use_each_paired_window(self):
        original = deepcopy(self.results)
        cases = {
            "attack_mean:accuracy_gap": lambda r: replace(r, accuracy=.775) if 25 <= r.round < 91 else r,
            "tail_mean:accuracy_gap": lambda r: replace(r, accuracy=.775) if 91 <= r.round < 100 else r,
            "final:accuracy_gap": lambda r: replace(r, accuracy=.779) if r.round == 100 else r,
        }
        for reason, mutate in cases.items():
            with self.subTest(reason=reason):
                self.results = deepcopy(original)
                self.distort_one_attacked_run_for_all_ours(mutate)
                report = self.report()
                target = report["ours_candidate_targets"][self.ours[0]]
                self.assertTrue(target["absolute_target_passed"])
                self.assertEqual(target["relative_target_status"], "unmet")
                self.assertIn(reason, [r["reason"] for r in target["relative_failures"]])
                self.assertNotIn("sm9rrs", report["selected"])
        self.results = deepcopy(original)
        for candidate in self.spec["candidates"]["vert"]:
            self.change(candidate["candidate_id"], asr=.039)
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_target_development")
        self.assertTrue(all(t["absolute_target_passed"] for t in report["ours_candidate_targets"].values()))

    def test_clean_accuracy_gap_to_vert_is_enforced(self):
        for cid in self.ours:
            index = next(i for i, run in enumerate(self.results[cid]) if not run.config.malicious_ratio)
            old = self.results[cid][index]
            self.results[cid][index] = synthetic_run(old.config, accuracy=.775)
        report = self.report()
        self.assertTrue(report["ours_health_passed"])
        self.assertEqual(report["status"], "needs_ours_target_development")
        self.assertIn("clean_final:accuracy_gap", [f["reason"] for f in
                      report["ours_candidate_targets"][self.ours[0]]["relative_failures"]])

    def test_all_failed_scorable_baselines_choose_highest_raw_score_without_veto(self):
        for method in base.ALL_METHODS[1:]:
            for index, candidate in enumerate(self.spec["candidates"][method]):
                self.change(candidate["candidate_id"], accuracy=.79 + .001 * index, nonfinite=1)
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertFalse(report["ours_target"]["reference_health_qualified"])
        self.assertTrue(report["ours_target"]["full_target_passed"])
        self.assertFalse(report["all_six_methods_healthy"])
        trials = {t["candidate_id"]: t for t in report["trials"]}
        for method in base.ALL_METHODS[1:]:
            expected = self.spec["candidates"][method][-1]["candidate_id"]
            self.assertEqual(report["selected"][method], expected)
            info, trial = report["methods"][method], trials[expected]
            self.assertEqual(info["selection_status"], "best_scored_unqualified")
            self.assertFalse(info["health_qualified"])
            self.assertFalse(trial["valid"])
            self.assertIsNone(trial["score"])
            self.assertIn("nonfinite_updates", trial["invalid_reasons"])
            self.assertAlmostEqual(trial["raw_score"], weighted_score(trial, self.spec["objective"]))

    def test_failed_scorable_vert_still_requires_valid_relative_comparison(self):
        for candidate in self.spec["candidates"]["vert"]:
            self.change(candidate["candidate_id"], accuracy=.83, asr=.05, nonfinite=1)
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_target_development")
        target = report["ours_candidate_targets"][self.ours[0]]
        self.assertFalse(target["reference_health_qualified"])
        self.assertTrue(target["reference_scorable"])
        self.assertEqual(target["relative_target_status"], "unmet")

    def test_only_one_failed_baseline_uses_fallback_without_changing_healthy_baselines(self):
        before = self.report()["selected"]
        for index, candidate in enumerate(self.spec["candidates"]["vert"]):
            self.change(candidate["candidate_id"], accuracy=.79 + .001 * index,
                        asr=.05, nonfinite=1)
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertTrue(report["ours_mnist_target_passed"])
        self.assertEqual(report["selected"]["vert"], self.spec["candidates"]["vert"][-1]["candidate_id"])
        self.assertEqual(report["methods"]["vert"]["selection_status"], "best_scored_unqualified")
        self.assertFalse(report["methods"]["vert"]["health_qualified"])
        for method in ("alignins", "krum", "ding13", "fedavg"):
            self.assertEqual(report["selected"][method], before[method])
            self.assertEqual(report["methods"][method]["selection_status"], "eligible_score_selection")
            self.assertTrue(report["methods"][method]["health_qualified"])
            self.assertFalse(report["methods"][method]["selected_without_valid_validation"])
        self.assertEqual(report["mnist_target_gate"]["unqualified_reference_methods"], ["vert"])

    def test_healthy_baseline_preferred_over_higher_scoring_failure(self):
        for candidate in self.spec["candidates"]["vert"]:
            self.change(candidate["candidate_id"], accuracy=.95, asr=.0, nonfinite=1)
        healthy = self.spec["candidates"]["vert"][0]["candidate_id"]
        self.change(healthy)
        report = self.report()
        self.assertEqual(report["selected"]["vert"], healthy)
        self.assertEqual(report["methods"]["vert"]["selection_status"], "eligible_score_selection")
        self.assertTrue(report["ours_target"]["reference_health_qualified"])

    def test_missing_baselines_are_unassessed_not_passed_and_do_not_veto_absolute_target(self):
        for method in base.ALL_METHODS[1:]:
            for candidate in self.spec["candidates"][method]:
                self.results.pop(candidate["candidate_id"])
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertEqual(len(report["selected"]), 6)
        self.assertEqual(report["ours_target"]["status"], "partially_assessed")
        self.assertEqual(report["ours_target"]["relative_target_status"], "unassessed")
        self.assertFalse(report["ours_mnist_target_passed"])
        self.assertFalse(report["ours_relative_performance_passed"])
        self.assertTrue(report["ours_absolute_target_passed"])
        self.assertEqual(report["pass_route"], "mnist_absolute_target_without_scorable_vert")
        self.assertTrue(report["missing_healthy_clean_references"])
        for method in base.ALL_METHODS[1:]:
            self.assertEqual(report["selected"][method], self.spec["fallback_candidates"][method])
            self.assertFalse(report["methods"][method]["health_qualified"])
            self.assertFalse(report["methods"][method]["scorable"])
            self.assertEqual(report["mnist_target_gate"]["unavailable_baselines"][method]["status"], "unassessed")

    def test_missing_vert_does_not_bypass_absolute_target(self):
        for candidate in self.spec["candidates"]["vert"]:
            self.results.pop(candidate["candidate_id"])
        for cid in self.ours:
            self.change(cid, asr=.051)
        self.assertEqual(self.report()["status"], "needs_ours_target_development")

    def test_incomplete_duplicate_foreign_and_changed_candidate_are_unscorable(self):
        cid = self.ours[0]
        original = self.results[cid]
        cases = {
            "missing": original[:-1],
            "duplicate": original[:-1] + original[:1],
            "formal_seed": [synthetic_run(replace(original[0].config, seed=801))] + original[1:],
            "changed_parameter": [synthetic_run(replace(r.config, detector_distance_threshold=9.)) for r in original],
            "other_candidate": self.results[self.ours[-1]],
        }
        for name, runs in cases.items():
            with self.subTest(name=name):
                self.results[cid] = runs
                report = self.report()
                row = report["mnist_target_gate"]["candidate_rows"][cid]
                self.assertFalse(row["scorable"])
                self.assertFalse(row["health_qualified"])
                self.assertIsNone(row["raw_score"])
                self.assertEqual(row["invalid_reasons"], "incomplete_or_mismatched_validation_results")
                self.assertNotEqual(report["selected"]["sm9rrs"], cid)
        self.results[cid] = original
        self.results["old_or_formal_study"] = object()
        self.assertEqual(self.report()["status"], "qualified_for_final")

    def test_missing_or_invalid_metric_never_receives_a_score(self):
        cid = self.spec["candidates"]["vert"][0]["candidate_id"]
        original = self.results[cid]
        cases = {
            "nan_asr": [synthetic_run(r.config, asr=float("nan")) for r in original],
            "short_rounds": [replace(original[0], records=original[0].records[:-1])] + original[1:],
            "invalid_final": [replace(original[0], final_accuracy=.99)] + original[1:],
            "invalid_weights": [replace(original[0], records=[replace(r, honest_weight_loss=None) for r in original[0].records])] + original[1:],
            "incomplete_candidate": original[:-1],
        }
        for name, runs in cases.items():
            with self.subTest(name=name):
                self.results[cid] = runs
                report = self.report()
                row = report["mnist_target_gate"]["candidate_rows"][cid]
                self.assertFalse(row["scorable"])
                self.assertIsNone(row["raw_score"])
                self.assertNotEqual(report["selected"]["vert"], cid)

    def test_complete_healthy_fedavg_controls_survive_missing_attack_result(self):
        fedavg = self.spec["candidates"]["fedavg"][0]["candidate_id"]
        index = next(i for i, run in enumerate(self.results[fedavg]) if run.config.malicious_ratio)
        del self.results[fedavg][index]
        for cid in self.ours:
            self.results[cid] = [synthetic_run(run.config, accuracy=.7 if not run.config.malicious_ratio else .8)
                                 for run in self.results[cid]]
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_development")
        self.assertFalse(report["ours_health_passed"])
        for cid in self.ours:
            row = report["mnist_target_gate"]["candidate_rows"][cid]
            self.assertIn("clean_accuracy_drop", row["invalid_reasons"])

    def test_nonfinite_ours_never_qualifies_even_with_perfect_metrics(self):
        for cid in self.ours:
            self.change(cid, accuracy=1., asr=0., nonfinite=1)
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_development")
        self.assertNotIn("sm9rrs", report["selected"])

    def test_mean_best_is_soft_preference_unless_explicitly_required(self):
        krum = self.spec["candidates"]["krum"][0]["candidate_id"]
        self.change(krum, accuracy=.81, asr=0.)
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertTrue(report["ours_mnist_target_passed"])
        self.assertFalse(report["ours_mean_dual_passed"])
        self.spec["promotion"]["require_mean_dual_best"] = True
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_target_development")
        self.assertTrue(all(t["full_target_passed"] for t in report["ours_candidate_targets"].values()))
        self.assertNotIn("sm9rrs", report["selected"])

    def test_target_qualification_precedes_mean_best(self):
        unqualified = self.ours[0]
        # Its best final means cannot hide one failed ASR scenario.
        self.change(unqualified, accuracy=.81, asr=.0)
        attacked = next(i for i, run in enumerate(self.results[unqualified]) if run.config.malicious_ratio)
        old = self.results[unqualified][attacked]
        self.results[unqualified][attacked] = synthetic_run(old.config, accuracy=.81, asr=.06)
        report = self.report()
        self.assertTrue(report["ours_candidate_targets"][unqualified]["mean_dual_best_passed"])
        self.assertFalse(report["ours_candidate_targets"][unqualified]["absolute_target_passed"])
        self.assertNotEqual(report["selected"]["sm9rrs"], unqualified)
        self.assertEqual(report["status"], "qualified_for_final")

    def test_mean_dual_preference_precedes_common_score_among_target_passes(self):
        preferred = self.ours[0]
        self.change(preferred, accuracy=.805, asr=.045)
        self.results[preferred] = [replace(run, records=[replace(r, accuracy=.785) if 25 <= r.round < 100 and run.config.malicious_ratio else r
                                                        for r in run.records]) for run in self.results[preferred]]
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertEqual(report["selected"]["sm9rrs"], preferred)
        self.assertNotEqual(report["best_healthy_score_candidate"], preferred)

    def test_expanded_candidate_plan_is_not_fixed_to_630_tasks(self):
        candidate = deepcopy(self.spec["candidates"]["sm9rrs"][0])
        candidate["candidate_id"] = "sm9rrs-v10-999"
        candidate["parameters"]["detector_distance_threshold"] = 3.5
        self.spec["candidates"]["sm9rrs"].append(candidate)
        self.tasks = base.build_tasks(self.spec, "validation")
        self.results = result_matrix(self.spec, self.tasks)
        report = self.report()
        self.assertEqual(len(self.tasks), 660)
        self.assertEqual(len(report["mnist_target_gate"]["candidate_rows"]), 22)
        self.assertEqual(report["status"], "qualified_for_final")

    def test_bad_spec_and_incomplete_or_foreign_task_plan_are_rejected(self):
        mutations = (
            lambda s: s["performance_target"].update(max_asr=.1),
            lambda s: s["promotion"].update(require_mean_dual_best="true"),
            lambda s: s["objective"].update(clean_accuracy_weight=float("nan")),
            lambda s: s["candidates"]["sm9rrs"][0]["parameters"].update(lr=.001),
            lambda s: s["validation"].update(seeds=[801, 1002, 1003]),
            lambda s: s["fallback_candidates"].update(vert="unknown"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                spec = deepcopy(self.spec)
                mutate(spec)
                with self.assertRaises(ValueError):
                    gate.select_validation(spec, self.results, self.tasks)
        with self.assertRaisesRegex(ValueError, "entire declared"):
            gate.select_validation(self.spec, self.results, self.tasks[:-1])
        tasks = deepcopy(self.tasks)
        tasks[0]["phase"] = "final"
        with self.assertRaisesRegex(ValueError, "validation-only"):
            gate.select_validation(self.spec, self.results, tasks)

    def test_selection_is_read_only_and_starts_no_training(self):
        spec, tasks = deepcopy(self.spec), deepcopy(self.tasks)
        before = {cid: [(run.config, run.final_accuracy, tuple(run.records)) for run in runs]
                  for cid, runs in self.results.items()}
        with mock.patch.object(base, "load_split", side_effect=AssertionError("no dataset access")), \
             mock.patch.object(base, "execute_phase", side_effect=AssertionError("no training")):
            self.report()
        self.assertEqual(self.spec, spec)
        self.assertEqual(self.tasks, tasks)
        self.assertEqual(before, {cid: [(run.config, run.final_accuracy, tuple(run.records)) for run in runs]
                                  for cid, runs in self.results.items()})


if __name__ == "__main__":
    unittest.main()
