"""Algorithm failure must remain visible without discarding completed work."""

import csv
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from sm9rrsfl import fair_tuning as tuning
from sm9rrsfl.datasets import ImageDataset
from sm9rrsfl.fl import ExperimentResult, RoundRecord
from tests.test_fair_tuning import OBJECTIVE, _minimal_spec, _result


ROOT = Path(__file__).resolve().parents[1]


def trial(method, *, reasons=(), candidate_id=None, **changes):
    result = tuning.score_trial(method, candidate_id or f"{method}-001", {}, [
        _result(0., .9, 0, method), _result(.7, .8, 0, method),
    ], objective=OBJECTIVE)
    if reasons:
        result = replace(result, valid=False, score=float("-inf"), invalid_reasons=reasons)
    return replace(result, **changes)


def fixture_result(_dataset, config, **_kwargs):
    """Complete trajectories; only attacked TAD runs revoke all honest clients."""
    malicious = tuple(f"client_{i}" for i in range(round(config.num_clients * config.malicious_ratio)))
    failed = config.method == "ding13" and bool(malicious)
    false_revoked = config.num_clients - len(malicious) if failed else 0
    accuracy = .1 if failed else .9
    records = [RoundRecord(
        config.method, config.malicious_ratio, i, accuracy, 1 - accuracy,
        config.num_clients - false_revoked, false_revoked, false_revoked,
        0, false_revoked, "", attack_target_success_rate=.9 if failed else 0.,
        honest_weight_loss=1. if failed else 0., malicious_weight_mass=0.,
    ) for i in range(1, config.rounds + 1)]
    return ExperimentResult(config, records, accuracy, 1 - accuracy,
                            config.rounds, malicious, ())


def integration_payload(dataset_name, folder):
    filename = "fair_tuning.cifar10.json" if dataset_name == "cifar10" else "fair_tuning.example.json"
    payload = json.loads((ROOT / "configs" / filename).read_text())
    payload["shared_parameters"].update(
        output_dir=str(folder), client_counts=[20], partitions=["iid"],
        ratio_range=[0., .4, 3], train_samples=1000, test_samples=100,
        attack_target_count=2, crypto_mode="simulated", visualizations=False,
        progress=False, calibration_candidate_budget=1, compute_backend="numpy",
        device="cpu", jobs=1, resume=True,
    )
    payload["tuning"].update(trials_per_tunable_method=1, validation_seeds=[1],
                            final_seeds=[2], final_jobs=1)
    for method in tuning.TUNABLE_METHODS:
        if not isinstance(payload["tuning"]["method_spaces"][method], dict):
            continue  # MNIST keeps its existing automatic Ours candidate pool.
        payload["tuning"]["method_spaces"][method] = {
            key: [values[0]] for key, values in payload["tuning"]["method_spaces"][method].items()
        }
    return payload


def fixture_dataset(dataset_name):
    shape = (3, 32, 32) if dataset_name == "cifar10" else (1, 28, 28)
    return ImageDataset(np.zeros((1000, *shape), dtype=np.float32),
                        np.tile(np.arange(10), 100),
                        np.ones((100, *shape), dtype=np.float32),
                        np.tile(np.arange(10), 10), name=dataset_name)


class ComparisonSelectionTest(unittest.TestCase):
    def test_failed_tad_is_retained_without_becoming_valid(self):
        trials = [trial(method) for method in tuning.ALL_METHODS]
        trials[4] = trial("ding13", reasons=("all_honest_revoked",))
        chosen = tuning.select_best_trials(trials)
        self.assertEqual(set(chosen), set(tuning.ALL_METHODS))
        self.assertIs(chosen["ding13"], trials[4])
        self.assertFalse(chosen["ding13"].valid)
        self.assertEqual(chosen["ding13"].score, float("-inf"))
        self.assertEqual(chosen["ding13"].invalid_reasons, ("all_honest_revoked",))

    def test_required_defenses_still_need_healthy_candidates(self):
        for method in tuning.REQUIRED_VALID_METHODS:
            with self.subTest(method=method):
                trials = [trial(m, reasons=("all_honest_revoked",) if m == method else ())
                          for m in tuning.ALL_METHODS]
                with self.assertRaisesRegex(tuning.FairTuningError, f"no valid candidate remains for {method}"):
                    tuning.select_best_trials(trials)

    def test_comparison_fallback_never_waives_missing_or_corrupt_data(self):
        for reason in ("round_completion_rate", "nonfinite_updates", "record_continuity",
                       "accuracy_metrics", "attack_metrics", "honest_weight_metrics",
                       "early_malicious_weight_metrics", "matched_fedavg_clean_reference",
                       "unknown_future_gate"):
            with self.subTest(reason=reason):
                trials = [trial(m) for m in tuning.ALL_METHODS]
                trials[4] = trial("ding13", reasons=("all_honest_revoked", reason))
                with self.assertRaisesRegex(tuning.FairTuningError, "no valid candidate remains for ding13"):
                    tuning.select_best_trials(trials)
        for change in ({"all_runs_completed": False}, {"nonfinite_updates": 1},
                       {"attack_success_rate": float("nan")}):
            with self.subTest(change=change):
                self.assertFalse(tuning._comparison_trial_usable(
                    trial("ding13", reasons=("all_honest_revoked",), **change)))

    def test_all_optional_comparators_can_fail_but_healthy_candidates_take_priority(self):
        trials = [trial(m, reasons=() if m in tuning.REQUIRED_VALID_METHODS else
                        ("all_honest_revoked",)) for m in tuning.ALL_METHODS]
        chosen = tuning.select_best_trials(trials)
        self.assertEqual(sum(t.valid for t in chosen.values()), 2)
        healthy = trial("alignins", candidate_id="alignins-002", score=.01)
        trials.append(healthy)
        self.assertIs(tuning.select_best_trials(trials)["alignins"], healthy)

    def test_failed_comparisons_use_same_objective_and_stable_tie_break(self):
        trials = [trial(m) for m in tuning.ALL_METHODS if m != "alignins"]
        accurate = trial("alignins", reasons=("clean_accuracy_drop",),
                         robust_accuracy=.99, attack_success_rate=.9)
        protected = trial("alignins", candidate_id="alignins-002", reasons=("clean_accuracy_drop",),
                          robust_accuracy=.1, attack_success_rate=0.)
        trials.extend([accurate, protected])
        accuracy_weights = {k: .05 for k in OBJECTIVE}
        accuracy_weights["robust_accuracy_weight"] = .85
        asr_weights = {k: .05 for k in OBJECTIVE}
        asr_weights["attack_success_weight"] = .85
        self.assertIs(tuning.select_best_trials(trials, objective=accuracy_weights)["alignins"], accurate)
        self.assertIs(tuning.select_best_trials(trials, objective=asr_weights)["alignins"], protected)
        trials.append(replace(protected, candidate_id="alignins-003"))
        self.assertEqual(tuning.select_best_trials(trials, objective=asr_weights)["alignins"].candidate_id,
                         "alignins-003")


class ObjectiveFallbackTest(unittest.TestCase):
    def results(self, failed_ratio=None):
        return {f"{m}-001": [_result(r, .9, 0, m,
                    false_positive_revocations=10 if m == "alignins" and r == failed_ratio else 0)
                            for r in (0., .1, .3)] for m in tuning.ALL_METHODS}

    def test_joint_weight_learning_is_unchanged_when_all_tunable_methods_work(self):
        spec = _minimal_spec({m: ({},) for m in tuning.ALL_METHODS})
        results = self.results()
        with mock.patch.object(tuning, "objective_weight_grid", return_value=(OBJECTIVE,)):
            actual = tuning._learn_unified_objective_weights(spec, results)
            original = tuning._fit_unified_objective_weights(spec, results, methods=tuning.TUNABLE_METHODS)
        self.assertEqual(actual, original)

    def test_optional_clean_or_attacked_failure_does_not_block_weight_learning(self):
        spec = _minimal_spec({m: ({},) for m in tuning.ALL_METHODS})
        for ratio in (0., .1):
            with self.subTest(ratio=ratio), \
                 mock.patch.object(tuning, "objective_weight_grid", return_value=(OBJECTIVE,)):
                weights, report = tuning._learn_unified_objective_weights(spec, self.results(ratio))
                self.assertEqual(weights, OBJECTIVE)
                self.assertEqual(report["status"], "learned")
                self.assertEqual(report["scope"], ["sm9rrs", "vert"])
                self.assertEqual(report["excluded_methods"], ["alignins"])
                self.assertTrue(report["failed_attempts"])

    def test_infeasible_cross_ratio_fit_uses_predeclared_weights_but_does_not_pass_bad_ours(self):
        spec = _minimal_spec({m: ({},) for m in tuning.ALL_METHODS})
        results = self.results()
        results["sm9rrs-001"][1] = _result(.1, .9, 0, "sm9rrs", false_positive_revocations=10)
        with mock.patch.object(tuning, "objective_weight_grid", return_value=(OBJECTIVE,)):
            weights, report = tuning._learn_unified_objective_weights(spec, results)
        self.assertEqual(weights, tuning.OBJECTIVE_DEFAULTS)
        self.assertEqual(report["status"], "fallback_infeasible_cross_ratio_fit")
        trials = [tuning.score_trial(m, f"{m}-001", {}, results[f"{m}-001"], objective=weights)
                  for m in tuning.ALL_METHODS]
        with self.assertRaisesRegex(tuning.FairTuningError, "no valid candidate remains for sm9rrs"):
            tuning.select_best_trials(trials, objective=weights)

    def test_cross_ratio_fallback_can_retain_a_candidate_healthy_on_all_scenarios(self):
        candidates = {m: ({},) for m in tuning.ALL_METHODS}
        candidates["sm9rrs"] = ({}, {"detector_subspace_dim": 2})
        spec = _minimal_spec(candidates)
        results = self.results()
        # Candidate 1 wins the fit on ratio .3, but fails health on held-out .1.
        # Candidate 2 remains eligible on the full validation matrix.
        results["sm9rrs-001"] = [
            _result(0., .9, 0, "sm9rrs"),
            _result(.1, .99, 0, "sm9rrs", false_positive_revocations=10),
            _result(.3, .99, 0, "sm9rrs"),
        ]
        results["sm9rrs-002"] = [_result(r, .9, 0, "sm9rrs") for r in (0., .1, .3)]
        with mock.patch.object(tuning, "objective_weight_grid", return_value=(OBJECTIVE,)):
            weights, report = tuning._learn_unified_objective_weights(spec, results)
        self.assertEqual(report["status"], "fallback_infeasible_cross_ratio_fit")
        trials = [tuning.score_trial(m, f"{m}-{i:03d}", p, results[f"{m}-{i:03d}"], objective=weights)
                  for m in tuning.ALL_METHODS for i, p in enumerate(candidates[m], 1)]
        chosen = tuning.select_best_trials(trials, objective=weights)
        self.assertEqual(chosen["sm9rrs"].candidate_id, "sm9rrs-002")
        self.assertTrue(chosen["sm9rrs"].valid and chosen["vert"].valid)


class ResumeComparisonFailureTest(unittest.TestCase):
    def test_mnist_and_cifar_resume_completed_validation_and_keep_six_formal_methods(self):
        for dataset_name in ("mnist", "cifar10"):
            with self.subTest(dataset=dataset_name), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                config = directory / "config.json"
                config.write_text(json.dumps(integration_payload(dataset_name, directory)))
                spec = tuning.load_fair_tuning_config(config)
                data = fixture_dataset(dataset_name)
                output = io.StringIO()
                from contextlib import redirect_stdout
                with mock.patch.object(tuning, "load_image_dataset", return_value=data), \
                     mock.patch.object(tuning, "prepare_tuning_tasks", side_effect=lambda data, tasks, args, **kw:
                                       (tasks, 1, "numpy", 1)), \
                     mock.patch.object(tuning, "print_resource_plan"), \
                     mock.patch.object(tuning, "objective_weight_grid", return_value=(OBJECTIVE,)), \
                     redirect_stdout(output):
                    # Mimic the old post-training gate. The actual phase runner
                    # writes the same on-disk manifests and pickled run results.
                    with mock.patch.object(tuning, "select_best_trials", side_effect=tuning.FairTuningError(
                            "no valid candidate remains for ding13")), \
                         mock.patch.object(tuning, "run_measured_experiment", side_effect=fixture_result) as first:
                        with self.assertRaisesRegex(tuning.FairTuningError, "ding13"):
                            tuning.run_fair_tuning(spec)
                    self.assertEqual(first.call_count, 18)
                    state = directory / ".tuning_state" / "validation"
                    fingerprints_before = sorted(p.name for p in state.iterdir())
                    self.assertFalse((directory / "best_parameters.json").exists())
                    with mock.patch.object(tuning, "run_measured_experiment", side_effect=fixture_result) as resumed:
                        selected = tuning.run_fair_tuning(spec)
                    # Only 6 methods x 3 ratios x 1 FORMAL seed may train.
                    self.assertEqual(resumed.call_count, 18)
                    self.assertEqual({call.args[1].seed for call in resumed.call_args_list}, {2})
                    self.assertEqual({call.args[1].method for call in resumed.call_args_list}, set(tuning.ALL_METHODS))
                    self.assertEqual(sorted(p.name for p in state.iterdir()), fingerprints_before)
                    with mock.patch.object(tuning, "run_measured_experiment",
                                           side_effect=AssertionError("all phases should resume")):
                        tuning.run_fair_tuning(spec)
                self.assertIn("resumed_completed_configurations=18", output.getvalue())
                best = json.loads((directory / "best_parameters.json").read_text())
                self.assertEqual(best["dataset"], dataset_name)
                self.assertEqual(best["selection_policy"]["required_valid_methods"], ["sm9rrs", "vert"])
                tad = best["selected"]["ding13"]
                self.assertEqual(tad["selection_status"], "comparison_only_failed")
                self.assertFalse(tad["validation_valid"])
                self.assertIsNone(tad["validation_score"])
                self.assertIn("all_honest_revoked", tad["invalid_reasons"])
                self.assertFalse(selected["ding13"].valid)
                self.assertEqual(best["performance_target_selection"]["status"], "passed")
                final = directory / "final_evaluation"
                target = json.loads((final / "performance_target.json").read_text())
                self.assertEqual(target["status"], "passed")
                self.assertFalse(target["parameters_reselected"])
                with (final / "aggregate.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                failed = [r for r in rows if r["method"] == "ding13" and float(r["malicious_ratio"]) > 0]
                self.assertEqual(len(failed), 2)
                self.assertTrue(all(r["training_health_failed_runs"] == "1" for r in failed))
                with (final / "scenario_audit.csv").open() as handle:
                    audit = list(csv.DictReader(handle))
                self.assertEqual(len(audit), 18)
                self.assertTrue(any("all_honest_revoked" in r["training_health_reasons"] for r in audit))


if __name__ == "__main__":
    unittest.main()
