from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest

import cifar_adaptive_gate as gate
import run_cifar_six_from_scratch as base
from tests.test_cifar_six_pipeline import synthetic_run


class AdaptiveGateTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(Path("configs/cifar10_six_relative_best_five_day_v7.json").read_text())
        self.spec["schema_version"] = 8
        self.spec["shared_parameters"].update(rounds=150, detector_window=10, attack_start_round=12)
        self.spec["selection_metrics"] = gate.selection_metrics(150)
        self.spec["validation"]["seeds"] = [3101, 3102]
        self.spec["final"]["seeds"] = [4101]
        self.spec["objective"] = dict(gate.OBJECTIVE)
        self.spec["gates"]["max_clean_accuracy_drop"] = .03
        self.spec["candidates"] = {m: [dict(c, candidate_id=f"{m}-b000-d{i:03d}")
                                           for i, c in enumerate(cs[:2])]
                                   for m, cs in self.spec["candidates"].items()}
        self.tasks = base.build_tasks(self.spec, "validation")
        self.results = {}
        for task in self.tasks:
            self.results.setdefault(task["candidate"]["candidate_id"], []).append(
                synthetic_run(base.fl.ExperimentConfig(**task["config"]), accuracy=.8, asr=.1))
        self.ours = [c["candidate_id"] for c in self.spec["candidates"]["sm9rrs"]]

    def report(self):
        return gate.select_validation(self.spec, self.results, self.tasks)

    def change(self, cid, accuracy=.8, asr=.1, nonfinite=0):
        self.results[cid] = [synthetic_run(r.config, accuracy=accuracy, asr=asr, nonfinite=nonfinite)
                             for r in self.results[cid]]

    def test_150_round_two_seed_matrix_qualifies(self):
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertEqual(report["selection_metrics"]["round"], 150)
        self.assertEqual(report["ours_target"]["accuracy_task_count"], 20)
        self.assertEqual(report["ours_target"]["asr_task_count"], 16)
        self.assertEqual(report["ours_target"]["scenarios"][0]["selection_round"], 150)

    def test_real_v8_block_config_with_three_gates_integrates(self):
        import run_cifar_adaptive as runner
        master = runner.load_spec(Path("configs/cifar10_resnet18_gn_tpe_v8.json"))
        state = runner.new_state(master, {"fingerprint": "synthetic-test-only"})
        block = runner.add_block(state, master)
        runner.add_wave(block)
        spec = runner.block_spec(master, block)
        self.assertNotIn("fedavg_min_attack_mean_asr", spec["gates"])
        tasks = base.build_tasks(spec, "validation")
        results = {}
        for task in tasks:
            results.setdefault(task["candidate"]["candidate_id"], []).append(
                synthetic_run(base.fl.ExperimentConfig(**task["config"]), accuracy=.8, asr=.1))
        report = gate.select_validation(spec, results, tasks)
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertFalse(report["attack_effectiveness"]["passed"])
        self.assertEqual(report["attack_effectiveness"]["minimum_mean_asr"], .5)
        self.assertEqual(report["attack_effectiveness"]["role"], "diagnostic_only_not_selection_or_promotion")
        self.assertNotIn("fedavg_min_attack_mean_asr", spec["gates"])

    def test_exact_two_pp_inclusive_then_one_sample_fails(self):
        for cid in self.ours:
            self.change(cid, accuracy=.78, asr=.12)
        self.assertEqual(self.report()["status"], "qualified_for_final")
        for cid in self.ours:
            self.change(cid, accuracy=.7796, asr=.12)
        self.assertEqual(self.report()["status"], "needs_ours_target_development")

    def test_process_asr_diagnostic_only(self):
        for cid in self.ours:
            self.results[cid] = [replace(run, records=[replace(r, attack_target_success_rate=.99)
                if run.config.attack_start_round <= r.round < 150 else r for r in run.records])
                if run.config.malicious_ratio else run for run in self.results[cid]]
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        trial = next(t for t in report["trials"] if t["candidate_id"] == self.ours[0])
        self.assertAlmostEqual(trial["attack_success_rate"], .1)
        self.assertGreater(trial["process_diagnostics"]["attack_success_rate"], .9)

    def test_health_failure_not_qualified_even_perfect_final(self):
        for cid in self.ours:
            self.change(cid, accuracy=.99, asr=0, nonfinite=1)
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_development")
        self.assertIsNone(report["best_healthy_ours_candidate"])

    def test_unhealthy_complete_baseline_still_reference(self):
        cid = self.spec["candidates"]["ding13"][0]["candidate_id"]
        self.change(cid, accuracy=.83, nonfinite=1)
        report = self.report()
        self.assertEqual(report["methods"]["ding13"]["selection_status"], "best_scored_unqualified")
        self.assertEqual(report["status"], "needs_ours_target_development")

    def test_healthy_baseline_preferred_over_high_failed_score(self):
        first, second = [c["candidate_id"] for c in self.spec["candidates"]["vert"]]
        self.change(second, accuracy=.99, asr=0, nonfinite=1)
        self.assertEqual(self.report()["selected"]["vert"], first)

    def test_missing_baseline_single_task_does_not_erase_other_tasks(self):
        cid = self.spec["candidates"]["ding13"][0]["candidate_id"]
        self.results[cid].pop()
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertFalse(report["ours_target"]["full_target_passed"])
        self.assertEqual(report["methods"]["ding13"]["selection_status"], "fixed_fallback_unqualified")
        self.change(cid, accuracy=.83)
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_target_development")
        self.assertEqual(report["ours_candidate_targets"][self.ours[0]]["accuracy_passed_task_count"], 1)

    def test_missing_ours_task_cannot_qualify(self):
        for cid in self.ours:
            self.results[cid].pop()
        self.assertEqual(self.report()["status"], "needs_ours_development")

    def test_wrong_result_config_duplicate_or_foreign_seed_rejected(self):
        original = deepcopy(self.results)
        for mode in ("duplicate", "foreign"):
            self.results = deepcopy(original)
            run = self.results[self.ours[0]][0]
            if mode == "duplicate":
                self.results[self.ours[0]].append(run)
            else:
                self.results[self.ours[0]][0] = replace(run, config=replace(run.config, seed=999))
            with self.assertRaisesRegex(ValueError, "duplicate or foreign"):
                self.report()

    def test_missing_plan_and_tampered_task_rejected(self):
        tasks = self.tasks
        self.tasks = tasks[:-1]
        with self.assertRaisesRegex(ValueError, "entire"):
            self.report()
        self.tasks = deepcopy(tasks)
        self.tasks[0]["config"]["lr"] *= 2
        with self.assertRaisesRegex(ValueError, "identity"):
            self.report()

    def test_fingerprint_and_public_trial_metadata_ignored_for_semantics(self):
        self.tasks = [dict(task, fingerprint="opaque", public_trial_id="b000") for task in self.tasks]
        self.assertEqual(self.report()["status"], "qualified_for_final")

    def test_100_round_endpoint_cannot_qualify(self):
        for cid in self.ours:
            self.results[cid] = [replace(run, stopped_round=100, records=run.records[:101])
                                 for run in self.results[cid]]
        self.assertEqual(self.report()["status"], "needs_ours_development")

    def test_final_accuracy_summary_must_match_record(self):
        for cid in self.ours:
            self.results[cid] = [replace(run, final_accuracy=.9) for run in self.results[cid]]
        self.assertEqual(self.report()["status"], "needs_ours_development")

    def test_score_formula_uses_endpoint_and_original_weight_loss(self):
        report = self.report()
        trial = next(t for t in report["trials"] if t["candidate_id"] == self.ours[0])
        self.assertAlmostEqual(trial["raw_score"], .25 * .8 + .5 * .8 + .2 * .9
                               + .05 * (1 - trial["honest_weight_loss"]))

    def test_float32_rate_boundary_passes(self):
        import numpy as np
        for cid in self.ours:
            self.change(cid, accuracy=float(np.float32(.78)), asr=float(np.float32(.12)))
        self.assertEqual(self.report()["status"], "qualified_for_final")

    def test_available_clean_control_still_vetoes_ours_when_fedavg_attacked_missing(self):
        fedavg = self.spec["candidates"]["fedavg"][0]["candidate_id"]
        self.results[fedavg].pop()
        for cid in self.ours:
            self.results[cid] = [synthetic_run(run.config, accuracy=.76, asr=.1)
                                if not run.config.malicious_ratio else run
                                for run in self.results[cid]]
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_development")
        self.assertIn("clean_accuracy_drop", report["final_metric_gate"]["candidate_rows"][self.ours[0]]["invalid_reasons"])

    def test_endpoint_metadata_cannot_claim_round100(self):
        self.spec["selection_metrics"]["round"] = 100
        with self.assertRaisesRegex(ValueError, "selection_metrics"):
            self.report()

    def test_formal_description_single_seed_missing_ours_not_pass(self):
        selected = {m: cs[0]["candidate_id"] for m, cs in self.spec["candidates"].items()}
        tasks = base.build_tasks(self.spec, "final", selected)
        results = {}
        for task in tasks:
            results.setdefault(task["candidate"]["candidate_id"], []).append(
                synthetic_run(base.fl.ExperimentConfig(**task["config"]), accuracy=.8, asr=.1))
        audit = gate.describe_final_targets(self.spec, selected, results, tasks)
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["scenarios"][0]["accuracy_maximum"]["sample_count"], 10000)
        results[selected["sm9rrs"]].pop()
        audit = gate.describe_final_targets(self.spec, selected, results, tasks)
        self.assertEqual(audit["status"], "incomplete")
        self.assertFalse(audit["full_target_passed"])


if __name__ == "__main__":
    unittest.main()
