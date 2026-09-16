"""Synthetic validation-only tests; no dataset download or model training."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from statistics import fmean
import unittest
from unittest import mock

import cifar_mean_dual_gate as gate
import run_cifar_six_from_scratch as base
from sm9rrsfl.calibration_policy import weighted_score
from tests.test_cifar_six_pipeline import result_matrix, synthetic_run


class MeanDualGateTests(unittest.TestCase):
    def setUp(self):
        self.spec = base.load_spec(Path(__file__).resolve().parents[1] /
                                   "configs/cifar10_six_original_v2.json")
        self.spec["validation"] = {
            "seeds": [401, 402, 403],
            "scenarios": [{"partition": p, "malicious_ratio": r}
                          for p in ("iid", "dirichlet") for r in (0., .1, .3, .5, .7)]}
        self.tasks = base.build_tasks(self.spec, "validation")
        self.results = result_matrix(self.spec, self.tasks)
        self.ours = [c["candidate_id"] for c in self.spec["candidates"]["sm9rrs"]]

    def change(self, cid, accuracy=.8, asr=.05, nonfinite=0):
        self.results[cid] = [synthetic_run(r.config, accuracy=accuracy, asr=asr, nonfinite=nonfinite)
                             for r in self.results[cid]]

    def report(self):
        return gate.select_validation(self.spec, self.results, self.tasks)

    def test_complete_ties_promote_six_methods_with_original_score_tiebreak(self):
        original = base.select_validation(self.spec, self.results, self.tasks)
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertEqual(report["selected"], original["selected"])
        self.assertEqual(report["ours_target"], original["ours_candidate_targets"][report["selected"]["sm9rrs"]])
        audit = report["mean_dual_gate"]
        self.assertEqual(audit["comparison_scope"], "all_six_methods")
        self.assertEqual(len(audit["candidate_rows"]), 21)
        self.assertEqual(len(audit["qualified_ours_candidates"]), 6)
        self.assertTrue(audit["six_method_dual_optimality_assessed"])
        self.assertEqual(audit["definition"]["numeric_tolerance"], 1e-12)
        self.assertFalse(report["official_test_used_for_selection"])

    def test_all_baselines_missing_preserve_fallback_and_do_not_veto(self):
        for method in base.ALL_METHODS[1:]:
            for candidate in self.spec["candidates"][method]:
                self.results.pop(candidate["candidate_id"])
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertEqual(len(report["selected"]), 6)
        audit = report["mean_dual_gate"]
        self.assertFalse(audit["six_method_dual_optimality_assessed"])
        self.assertEqual(audit["comparison_scope"], "ours_candidates_only_no_qualified_baselines")
        self.assertEqual(audit["available_baselines"], {})
        self.assertEqual(len(audit["unavailable_baselines"]), 5)
        for method, info in audit["unavailable_baselines"].items():
            self.assertEqual(info["status"], "unassessed")
            self.assertFalse(info["blocks_promotion"])
            self.assertFalse(info["fallback_qualified"])
            self.assertEqual(report["selected"][method], self.spec["fallback_candidates"][method])
        self.assertEqual(report["ours_target"]["status"], "incomplete")

    def test_failed_scorable_vert_is_reference_without_health_qualification(self):
        for candidate in self.spec["candidates"]["vert"]:
            self.change(candidate["candidate_id"], accuracy=1., asr=0., nonfinite=1)
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_dual_development")
        audit = report["mean_dual_gate"]
        self.assertNotIn("vert", audit["unavailable_baselines"])
        self.assertFalse(audit["available_baselines"]["vert"]["health_qualified"])
        self.assertFalse(report["near_vert_gate"]["reference_health_qualified"])
        self.assertTrue(audit["comparison_includes_unqualified_references"])
        self.assertEqual(report["methods"]["vert"]["selection_status"], "best_scored_unqualified")

    def test_other_baseline_better_does_not_veto_near_vert_route(self):
        cid = self.spec["candidates"]["krum"][0]["candidate_id"]
        for accuracy, asr in ((.801, .05), (.8, .049)):
            with self.subTest(accuracy=accuracy, asr=asr):
                self.change(cid, accuracy=accuracy, asr=asr)
                report = self.report()
                self.assertEqual(report["status"], "qualified_for_final")
                self.assertTrue(report["ours_health_passed"])
                self.assertTrue(report["ours_mean_dual_passed"])
                self.assertFalse(report["ours_joint_mean_dual_passed"])
                self.assertEqual(report["pass_route"], "near_vert")

    def test_split_mean_winners_may_pass_only_through_near_vert(self):
        self.change(self.ours[0], accuracy=.82, asr=.05)
        self.change(self.ours[1], accuracy=.8, asr=.01)
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        audit = report["mean_dual_gate"]
        self.assertEqual(audit["joint_qualified_ours_candidates"], [])
        self.assertEqual(report["pass_route"], "near_vert")
        self.assertAlmostEqual(audit["best_mean_accuracy"], .82)
        self.assertAlmostEqual(audit["best_mean_asr"], .01)

    def test_joint_optimal_candidate_selected_even_with_lower_common_score(self):
        winning = self.ours[0]
        runs = []
        for old in self.results[winning]:
            run = synthetic_run(old.config, accuracy=.82, asr=.01)
            if old.config.malicious_ratio > 0:
                run = replace(run, records=[replace(r, accuracy=.4) if 25 <= r.round < 100 else r for r in run.records])
            runs.append(run)
        self.results[winning] = runs
        original = base.select_validation(self.spec, self.results, self.tasks)
        self.assertNotEqual(original["selected"]["sm9rrs"], winning)
        report = self.report()
        self.assertEqual(report["selected"]["sm9rrs"], winning)
        self.assertEqual(report["best_healthy_score_candidate"], original["selected"]["sm9rrs"])
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertEqual(report["ours_target"], original["ours_candidate_targets"][winning])

    def test_baseline_reference_still_uses_original_score_independently(self):
        cid = self.spec["candidates"]["vert"][0]["candidate_id"]
        values = []
        for old in self.results[cid]:
            run = synthetic_run(old.config, accuracy=.9, asr=.01)
            if old.config.malicious_ratio > 0:
                run = replace(run, records=[replace(r, accuracy=.1) if 25 <= r.round < 100 else r for r in run.records])
            values.append(run)
        self.results[cid] = values
        original = base.select_validation(self.spec, self.results, self.tasks)
        self.assertNotEqual(original["selected"]["vert"], cid)
        report = self.report()
        self.assertEqual(report["selected"]["vert"], original["selected"]["vert"])
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertNotIn(cid, report["mean_dual_gate"]["comparator_candidates"])

    def test_clean_asr_is_excluded_and_task_seed_scenario_weights_are_equal(self):
        cid = self.ours[0]
        values = []
        for index, old in enumerate(self.results[cid]):
            accuracy = .8 + index / 1000
            asr = .02 + (index % 4) / 1000 if old.config.malicious_ratio > 0 else 1.
            run = synthetic_run(old.config, accuracy=accuracy, asr=asr)
            # Metrics before the attack window must not enter mean ASR.
            run = replace(run, records=[replace(r, attack_target_success_rate=1.) if r.round < 25 else r for r in run.records])
            values.append(run)
        self.results[cid] = values
        report = self.report()
        row = report["mean_dual_gate"]["candidate_rows"][cid]
        self.assertEqual(row["matched_validation_tasks"], 30)
        self.assertEqual(row["attacked_tasks"], 24)
        self.assertAlmostEqual(row["mean_accuracy"], fmean(r.final_accuracy for r in values))
        attacked = [r for r in values if r.config.malicious_ratio > 0]
        self.assertAlmostEqual(row["mean_asr"], fmean(fmean(x.attack_target_success_rate for x in r.records[25:]) for r in attacked))
        self.assertEqual(report["selected"]["sm9rrs"], cid)
        self.assertTrue(all(t["attack_mean_asr"] is None for t in row["tasks"] if t["malicious_ratio"] == 0))

    def test_missing_or_duplicate_validation_task_disqualifies_candidate(self):
        for mutation in (lambda runs: runs[:-1], lambda runs: runs[:-1] + runs[:1]):
            with self.subTest(mutation=mutation):
                before = self.results[self.ours[0]]
                self.results[self.ours[0]] = mutation(before)
                report = self.report()
                row = report["mean_dual_gate"]["candidate_rows"][self.ours[0]]
                self.assertFalse(row["eligible"])
                self.assertIsNone(row["mean_accuracy"])
                self.assertNotIn(self.ours[0], report["mean_dual_gate"]["comparator_candidates"])
                self.results[self.ours[0]] = before

    def test_nonfinite_ours_cannot_win_and_all_bad_ours_block(self):
        self.change(self.ours[0], accuracy=.99, asr=0., nonfinite=1)
        report = self.report()
        self.assertNotEqual(report["selected"]["sm9rrs"], self.ours[0])
        for cid in self.ours[1:]:
            self.change(cid, nonfinite=1)
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_development")
        self.assertFalse(report["ours_health_passed"])
        self.assertNotIn("sm9rrs", report["selected"])

    def test_invalid_attack_metrics_never_become_a_zero_asr_reference(self):
        cid = self.ours[0]
        self.change(cid, accuracy=.99, asr=float("nan"))
        report = self.report()
        row = report["mean_dual_gate"]["candidate_rows"][cid]
        self.assertFalse(row["eligible"])
        self.assertIsNone(row["mean_asr"])
        self.assertIn("invalid_attack_metrics", row["invalid_reasons"])
        self.assertNotIn(cid, report["mean_dual_gate"]["comparator_candidates"])

    def test_formal_substitution_and_changed_config_cannot_enter_gate(self):
        cid = self.ours[0]
        for config in (replace(self.results[cid][0].config, seed=801),
                       replace(self.results[cid][0].config, lr=.001)):
            with self.subTest(config=config):
                old = self.results[cid][0]
                self.results[cid][0] = synthetic_run(config, accuracy=1., asr=0.)
                report = self.report()
                row = report["mean_dual_gate"]["candidate_rows"][cid]
                self.assertFalse(row["eligible"])
                self.assertEqual(row["invalid_reasons"], "incomplete_or_mismatched_validation_results")
                self.results[cid][0] = old
        self.results["unknown_formal_results"] = object()
        self.assertEqual(self.report()["status"], "qualified_for_final")

    def test_short_or_formal_task_plans_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "630"):
            gate.select_validation(self.spec, self.results, self.tasks[:-1])
        tasks = deepcopy(self.tasks)
        tasks[0]["phase"] = "final"
        with self.assertRaisesRegex(ValueError, "validation-only"):
            gate.select_validation(self.spec, self.results, tasks)
        spec = deepcopy(self.spec)
        spec["validation"]["seeds"] = [401]
        with self.assertRaisesRegex(ValueError, "3 independent"):
            gate.select_validation(spec, self.results, self.tasks)

    def test_joint_route_keeps_only_numerical_tolerance(self):
        cid = self.spec["candidates"]["krum"][0]["candidate_id"]
        self.change(cid, accuracy=.8 + 5e-13)
        self.assertEqual(self.report()["pass_route"], "joint_best")
        self.change(cid, accuracy=.8 + 1e-9)
        self.assertEqual(self.report()["pass_route"], "near_vert")

    def test_all_failed_baselines_choose_highest_original_raw_score(self):
        for method in base.ALL_METHODS[1:]:
            candidates = self.spec["candidates"][method]
            for index, candidate in enumerate(candidates):
                self.change(candidate["candidate_id"], accuracy=.79 + .001 * index,
                            asr=.05, nonfinite=1)
        report = self.report()
        trials = {t["candidate_id"]: t for t in report["trials"]}
        self.assertEqual(report["status"], "qualified_for_final")
        for method in base.ALL_METHODS[1:]:
            expected = self.spec["candidates"][method][-1]["candidate_id"]
            self.assertEqual(report["selected"][method], expected)
            info = report["methods"][method]
            self.assertEqual(info["selection_status"], "best_scored_unqualified")
            self.assertTrue(info["selected_without_valid_validation"])
            self.assertFalse(info["comparison_available"])
            trial = trials[expected]
            self.assertFalse(trial["valid"])
            self.assertFalse(trial["health_qualified"])
            self.assertIsNone(trial["score"])
            self.assertIn("nonfinite_updates", trial["invalid_reasons"])
            self.assertAlmostEqual(trial["raw_score"], weighted_score(trial, self.spec["objective"]))
            self.assertEqual(info["selection_raw_score"], trial["raw_score"])

    def test_healthy_baseline_is_preferred_over_better_scored_failure(self):
        candidates = self.spec["candidates"]["vert"]
        for candidate in candidates:
            self.change(candidate["candidate_id"], accuracy=.95, asr=0., nonfinite=1)
        healthy = candidates[0]["candidate_id"]
        self.change(healthy, accuracy=.8, asr=.05)
        report = self.report()
        self.assertEqual(report["selected"]["vert"], healthy)
        self.assertTrue(report["near_vert_gate"]["reference_health_qualified"])
        self.assertEqual(report["methods"]["vert"]["selection_status"], "eligible_score_selection")

    def test_failed_score_retains_all_health_reasons(self):
        for candidate in self.spec["candidates"]["vert"]:
            cid = candidate["candidate_id"]
            self.change(cid, nonfinite=1)
            self.results[cid] = [replace(run, records=[replace(r, blacklisted_clients=100,
                                                               false_positive_revocations=100)
                                                      if r.round > 0 else r for r in run.records])
                                 for run in self.results[cid]]
        report = self.report()
        selected = report["selected"]["vert"]
        row = next(r for r in report["trials"] if r["candidate_id"] == selected)
        self.assertTrue(row["scorable"])
        self.assertFalse(row["valid"])
        for reason in ("nonfinite_updates", "all_clients_revoked", "all_honest_revoked",
                       "clean_false_revocation_rate"):
            self.assertIn(reason, row["invalid_reasons"])
            self.assertIn(reason, report["methods"]["vert"]["candidate_failures"][selected])

    def test_failed_vert_near_boundary_requires_both_limits(self):
        for candidate in self.spec["candidates"]["vert"]:
            self.change(candidate["candidate_id"], accuracy=.805, asr=.04, nonfinite=1)
        report = self.report()
        self.assertEqual(report["pass_route"], "near_vert")
        self.assertFalse(report["near_vert_gate"]["reference_health_qualified"])
        self.assertEqual(report["near_vert_gate"]["status"], "passed")
        for accuracy, asr in ((.805000001, .04), (.805, .039999999)):
            with self.subTest(accuracy=accuracy, asr=asr):
                for candidate in self.spec["candidates"]["vert"]:
                    self.change(candidate["candidate_id"], accuracy=accuracy, asr=asr, nonfinite=1)
                report = self.report()
                self.assertEqual(report["status"], "needs_ours_dual_development")
                self.assertTrue(report["ours_health_passed"])
                self.assertIsNone(report["pass_route"])
                self.assertNotIn("sm9rrs", report["selected"])

    def test_lower_asr_cannot_compensate_accuracy_outside_vert_limit(self):
        for candidate in self.spec["candidates"]["vert"]:
            self.change(candidate["candidate_id"], accuracy=.806, asr=.2)
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_dual_development")
        self.assertEqual(report["near_vert_gate"]["status"], "unmet")

    def test_joint_best_tier_precedes_higher_score_near_candidate(self):
        winning = self.ours[0]
        values = []
        for old in self.results[winning]:
            run = synthetic_run(old.config, accuracy=.805, asr=.045)
            if old.config.malicious_ratio:
                run = replace(run, records=[replace(r, accuracy=.2) if 25 <= r.round < 100 else r
                                            for r in run.records])
            values.append(run)
        self.results[winning] = values
        original = base.select_validation(self.spec, self.results, self.tasks)
        self.assertNotEqual(original["selected"]["sm9rrs"], winning)
        report = self.report()
        self.assertEqual(report["selected"]["sm9rrs"], winning)
        self.assertEqual(report["pass_route"], "joint_best")
        self.assertEqual(len(report["mean_dual_gate"]["near_vert_qualified_ours_candidates"]), 6)

    def test_incomplete_or_unmeasurable_baseline_never_gets_raw_score(self):
        cid = self.spec["candidates"]["vert"][0]["candidate_id"]
        original = self.results[cid]
        cases = {
            "one_good_seed": lambda runs: runs[:10],
            "missing_round": lambda runs: [replace(runs[0], records=runs[0].records[1:])] + runs[1:],
            "missing_asr": lambda runs: [replace(r, records=[replace(x, attack_target_success_rate=None)
                                                            for x in r.records]) for r in runs],
            "invalid_weight": lambda runs: [replace(runs[0], records=[replace(x, honest_weight_loss=None)
                                                                      for x in runs[0].records])] + runs[1:],
            "invalid_final_accuracy": lambda runs: [replace(runs[0], final_accuracy=float("nan"))] + runs[1:],
            "final_accuracy_disagrees": lambda runs: [replace(runs[0], final_accuracy=.99)] + runs[1:],
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                self.results[cid] = mutate(original)
                report = self.report()
                row = report["mean_dual_gate"]["candidate_rows"][cid]
                self.assertFalse(row["scorable"])
                self.assertIsNone(row["raw_score"])
                self.assertIsNone(row["mean_accuracy"])
                self.assertNotEqual(report["selected"]["vert"], cid)
        self.results[cid] = original

    def test_missing_vert_cannot_fabricate_near_pass(self):
        for candidate in self.spec["candidates"]["vert"]:
            self.results.pop(candidate["candidate_id"])
        self.change(self.ours[0], accuracy=.82, asr=.05)
        self.change(self.ours[1], accuracy=.8, asr=.01)
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_dual_development")
        self.assertEqual(report["near_vert_gate"]["status"], "unassessed")
        self.assertIsNone(report["near_vert_gate"]["reference_candidate"])
        self.assertFalse(report["near_vert_gate"]["reference_scorable"])
        self.assertTrue(all(not t["near_vert_passed"] for t in
                            report["mean_dual_gate"]["ours_candidate_targets"].values()))

    def test_missing_healthy_fedavg_reference_does_not_destroy_complete_raw_score(self):
        fedavg = self.spec["candidates"]["fedavg"][0]["candidate_id"]
        self.change(fedavg, nonfinite=1)
        vert = self.spec["candidates"]["vert"][0]["candidate_id"]
        self.change(vert, nonfinite=1)
        report = self.report()
        row = next(t for t in report["trials"] if t["candidate_id"] == vert)
        self.assertEqual(row["clean_utility"]["status"], "unassessed")
        self.assertTrue(row["scorable"])
        self.assertIsNotNone(row["raw_score"])

    def test_invalid_or_missing_fedavg_attack_does_not_erase_healthy_clean_controls(self):
        fedavg = self.spec["candidates"]["fedavg"][0]["candidate_id"]
        original = self.results[fedavg]
        for cid in self.ours:
            self.results[cid] = [synthetic_run(r.config, accuracy=.7 if not r.config.malicious_ratio else .9)
                                 for r in self.results[cid]]
        attacked_index = next(i for i, run in enumerate(original) if run.config.malicious_ratio)
        bad = replace(original[attacked_index], records=[replace(r, accuracy=float("nan"))
                                                       if r.round == 25 else r
                                                       for r in original[attacked_index].records])
        cases = {
            "invalid_attack": original[:attacked_index] + [bad] + original[attacked_index + 1:],
            "missing_attack": original[:attacked_index] + original[attacked_index + 1:],
            "foreign_attack": original[:attacked_index] + [synthetic_run(replace(
                original[attacked_index].config, seed=801))] + original[attacked_index + 1:],
        }
        for name, runs in cases.items():
            with self.subTest(name=name):
                self.results[fedavg] = runs
                report = self.report()
                self.assertEqual(report["status"], "needs_ours_development")
                self.assertFalse(report["ours_health_passed"])
                for row in report["trials"]:
                    if row["method"] == "sm9rrs":
                        self.assertEqual(row["clean_utility"]["status"], "failed")
                        self.assertIn("clean_accuracy_drop", row["invalid_reasons"])
                        self.assertFalse(row["health_qualified"])
                self.assertFalse(report["methods"]["fedavg"]["scorable"])

    def test_weight_roundoff_preserves_original_runtime_health_tolerance(self):
        cid = self.spec["candidates"]["vert"][0]["candidate_id"]
        original = self.results[cid]
        for mass, expected in ((1.0000000000000002, True), (1.0001, False), (float("nan"), False)):
            with self.subTest(mass=mass):
                self.results[cid] = [replace(r, records=[replace(x, malicious_weight_mass=mass)
                                                       for x in r.records]) for r in original]
                row = self.report()["mean_dual_gate"]["candidate_rows"][cid]
                self.assertEqual(row["scorable"], expected)
                self.assertEqual(row["health_qualified"], expected)

    def test_selection_preserves_inputs_and_never_starts_training(self):
        spec = deepcopy(self.spec)
        tasks = deepcopy(self.tasks)
        before = {cid: [(run.config, run.final_accuracy, tuple(run.records)) for run in runs]
                  for cid, runs in self.results.items()}
        with mock.patch.object(base, "load_split", side_effect=AssertionError("no data reload")), \
             mock.patch.object(base, "execute_phase", side_effect=AssertionError("no training")):
            self.report()
        self.assertEqual(self.spec, spec)
        self.assertEqual(self.tasks, tasks)
        self.assertEqual(before, {cid: [(run.config, run.final_accuracy, tuple(run.records)) for run in runs]
                                  for cid, runs in self.results.items()})


if __name__ == "__main__":
    unittest.main()
