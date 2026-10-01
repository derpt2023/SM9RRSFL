"""Adaptive search causality, replay, constraints and objective boundaries."""
from copy import deepcopy
import json
import math
import unittest

import numpy as np

from cifar_adaptive_search import (
    AdaptiveTPESampler, DEFAULT_COORDINATES, FIXED_METHODS, METHODS, SPACES,
    _Density, compute_relative_objective, decode_parameters, space_contract,
)
from sm9rrsfl.ours_policy import OursParameters


class SamplerTests(unittest.TestCase):
    def test_observed_loss_changes_actual_learned_proposal(self):
        first = AdaptiveTPESampler("public", 91)
        second = AdaptiveTPESampler("public", 91)
        a, b = first.ask(), first.ask()
        self.assertEqual(a, second.ask())
        self.assertEqual(b, second.ask())
        self.assertEqual(a["strategy"], "declared_default")
        self.assertEqual(b["strategy"], "random_startup")
        first.tell(a["trial_id"], 0.)
        first.tell(b["trial_id"], 10.)
        second.tell(a["trial_id"], 10.)
        second.tell(b["trial_id"], 0.)
        good_a, good_b = first.ask(), second.ask()
        self.assertEqual(good_a["strategy"], "tpe")
        self.assertEqual(good_b["strategy"], "tpe")
        self.assertNotEqual(good_a["parameters"], good_b["parameters"])
        self.assertEqual(good_a["fit_audit"]["good_trial_ids"], [a["trial_id"]])
        self.assertEqual(good_b["fit_audit"]["good_trial_ids"], [b["trial_id"]])
        self.assertGreater(good_a["fit_audit"]["log_density_ratio"], 0.)
        self.assertEqual(good_a["fit_audit"]["candidate_pool_size"], 96)

    def test_json_roundtrip_replays_pending_and_next_ask_exactly(self):
        sampler = AdaptiveTPESampler("sm9rrs", 823, context_id="public-0004")
        one = sampler.ask()
        sampler.tell(one["trial_id"], 2., metrics={"health": True})
        pending = sampler.ask()
        saved = json.loads(json.dumps(sampler.state_dict(), allow_nan=False))
        restored = AdaptiveTPESampler.from_state(saved)
        self.assertEqual(restored.pending_trials, [pending])
        for model in (sampler, restored):
            model.tell(pending["trial_id"], 1.)
        self.assertEqual(sampler.ask(), restored.ask())
        self.assertEqual(sampler.state_dict(), restored.state_dict())

    def test_all_search_spaces_respect_constraints_and_use_continuous_values(self):
        for name in SPACES:
            sampler = AdaptiveTPESampler(name, 1892, context_id="public-context")
            draws = 1 if name in FIXED_METHODS else 9
            seen = set()
            for i in range(draws):
                trial = sampler.ask()
                params = trial["parameters"]
                self.assertNotIn(json.dumps(params, sort_keys=True), seen)
                seen.add(json.dumps(params, sort_keys=True))
                if name == "sm9rrs":
                    OursParameters(**params).validate()
                    self.assertLess(params["detector_history_threshold"], params["detector_distance_threshold"])
                    self.assertLessEqual(params["detector_drift_allowance"], params["detector_distance_threshold"])
                if name == "vert":
                    self.assertEqual(params["vert_top_k"], 0)
                    self.assertFalse(params["vert_use_ratio_prior"])
                sampler.tell(trial["trial_id"], float((i - 3) ** 2))
            self.assertEqual(sampler.state_dict(), AdaptiveTPESampler.from_state(sampler.state_dict()).state_dict())
            if name in FIXED_METHODS:
                self.assertEqual(trial["parameters"], {})
                with self.assertRaises(StopIteration):
                    sampler.ask()
            elif name == "public":
                rates = [t["parameters"]["lr"] for t in sampler.trials[1:]]
                self.assertTrue(any(abs(x * 1000 - round(x * 1000)) > 1e-6 for x in rates))

    def test_failed_never_good_incomplete_never_fitted_and_nan_rejected(self):
        sampler = AdaptiveTPESampler("public", 43)
        fail = sampler.ask()
        sampler.tell(fail["trial_id"], status="failed", metrics={"reason": "nonfinite_update"})
        skipped = sampler.ask()
        sampler.tell(skipped["trial_id"], status="incomplete", metrics={"reason": "resource_failure"})
        for loss in (1., 2.):
            trial = sampler.ask()
            sampler.tell(trial["trial_id"], loss)
        trial = sampler.ask()
        self.assertEqual(trial["strategy"], "tpe")
        audit = trial["fit_audit"]
        self.assertIn(fail["trial_id"], audit["bad_trial_ids"])
        self.assertNotIn(fail["trial_id"], audit["good_trial_ids"])
        self.assertNotIn(skipped["trial_id"], audit["good_trial_ids"] + audit["bad_trial_ids"])
        for invalid in (float("nan"), float("inf"), None, True):
            with self.assertRaises(ValueError):
                sampler.tell(trial["trial_id"], invalid)
        with self.assertRaises(ValueError):
            sampler.tell(trial["trial_id"], -1000., status="failed")
        self.assertEqual(sampler.trials[-1]["status"], "pending")

    def test_all_failed_observations_do_not_become_successful_model_fit(self):
        sampler = AdaptiveTPESampler("public", 40)
        for _ in range(5):
            trial = sampler.ask()
            self.assertNotEqual(trial["strategy"], "tpe")
            sampler.tell(trial["trial_id"], status="failed")

    def test_repeated_tell_is_idempotent_but_changed_observation_rejected(self):
        sampler = AdaptiveTPESampler("public", 90)
        trial = sampler.ask()
        sampler.tell(trial["trial_id"], 5., metrics={"a": 2})
        sampler.tell(trial["trial_id"], 5., metrics={"a": 2})
        with self.assertRaises(ValueError):
            sampler.tell(trial["trial_id"], 4.)
        with self.assertRaises(ValueError):
            sampler.tell("no-such-id", 4.)

    def test_conditional_context_identity_affects_random_startup(self):
        a = AdaptiveTPESampler("sm9rrs", 7, context_id="public-A")
        b = AdaptiveTPESampler("sm9rrs", 7, context_id="public-B")
        a.ask()
        b.ask()
        self.assertNotEqual(a.ask()["parameters"], b.ask()["parameters"])
        with self.assertRaises(ValueError):
            AdaptiveTPESampler("sm9rrs", 7)

    def test_state_identity_nonfinite_and_parameter_tampering_rejected(self):
        sampler = AdaptiveTPESampler("public", 67)
        trial = sampler.ask()
        sampler.tell(trial["trial_id"], 1.)
        for mutate in (
            lambda s: s.update(space_fingerprint="wrong"),
            lambda s: s.update(algorithm_version="wrong"),
            lambda s: s["trials"][0].update(loss=float("nan")),
            lambda s: s["trials"][0]["parameters"].update(detector_window=11),
            lambda s: s["trials"][0].update(trial_id="public-0008"),
        ):
            state = sampler.state_dict()
            mutate(state)
            with self.assertRaises(ValueError):
                AdaptiveTPESampler.from_state(state)

    def test_density_normalizes_and_retains_unobserved_support(self):
        continuous = _Density({"kind": "real", "low": 0., "high": 1., "log": False}, [.001, .9, .99])
        x = np.linspace(0., 1., 10001)
        density = np.asarray([math.exp(continuous.log_density(v)) for v in x])
        area = float(np.sum((density[:-1] + density[1:]) * np.diff(x) * .5))
        self.assertAlmostEqual(area, 1., places=6)
        discrete = _Density({"kind": "choice", "values": [1, 2, 3]}, [1, 1, 1])
        values = [math.exp(discrete.log_density(v)) for v in (1, 2, 3)]
        self.assertAlmostEqual(sum(values), 1.)
        self.assertGreater(values[1], 0.)
        self.assertGreater(values[0], values[1])

    def test_declared_contract_default_and_learning_enabled_drift_space(self):
        default = space_contract("sm9rrs")["default_parameters"]
        self.assertEqual(default["detector_distance_threshold"], 1.25)
        self.assertEqual(default["detector_drift_allowance"], 1.25)
        coordinates = deepcopy(DEFAULT_COORDINATES["sm9rrs"])
        coordinates.update(allowance_fraction=.5, detector_drift_threshold=.5)
        adapted = decode_parameters("sm9rrs", coordinates)
        warning = adapted["detector_distance_threshold"]
        allowance = adapted["detector_drift_allowance"]
        memory = adapted["detector_drift_memory"]
        self.assertGreater((warning - allowance) / (1 - memory), adapted["detector_drift_threshold"])
        self.assertEqual(space_contract("public")["public_fixed"]["attack_start_round"], "detector_window + 2")


class ObjectiveTests(unittest.TestCase):
    def fixture(self):
        ours = [dict(scenario=["iid", 0., 901], accuracy=.8, asr=None,
                     attacked=False, complete=True, healthy=True),
                dict(scenario=["iid", .5, 901], accuracy=.75, asr=.1,
                     attacked=True, complete=True, healthy=True)]
        peers = {method: deepcopy(ours) for method in METHODS if method != "sm9rrs"}
        return ours, peers

    def test_complete_equal_methods_have_only_declared_score_penalty(self):
        ours, peers = self.fixture()
        audit = compute_relative_objective(ours, peers, original_score=.8)
        self.assertAlmostEqual(audit["loss"], .002)
        self.assertTrue(audit["comparison_complete"])
        self.assertFalse(audit["promotion_assessed"])

    def test_healthy_failure_baseline_still_sets_best_reference(self):
        ours, peers = self.fixture()
        peers["vert"][1].update(accuracy=.9, asr=.01, healthy=False)
        audit = compute_relative_objective(ours, peers)
        attacked = audit["scenarios"][1]
        self.assertAlmostEqual(attacked["accuracy_gap"], .15)
        self.assertAlmostEqual(attacked["asr_gap"], .09)
        self.assertEqual(attacked["scorable_unhealthy_baselines"], ["vert"])
        self.assertGreater(audit["loss"], 0.)

    def test_missing_does_not_impute_zero_or_silently_disappear(self):
        ours, peers = self.fixture()
        peers["vert"] = []
        audit = compute_relative_objective(ours, peers)
        self.assertFalse(audit["comparison_complete"])
        self.assertEqual(len(audit["missing_baselines"]), 2)
        self.assertAlmostEqual(audit["loss"], .2)
        self.assertEqual(audit["scenarios"][1]["asr_gap"], 0.)
        self.assertNotIn("vert", audit["scenarios"][1]["available_methods"])

    def test_invalid_incomplete_ours_never_gets_favorable_loss(self):
        for changes in (dict(accuracy=float("nan")), dict(asr=None), dict(complete=False)):
            ours, peers = self.fixture()
            ours[1].update(changes)
            audit = compute_relative_objective(ours, peers)
            self.assertIsNone(audit["loss"])
            self.assertEqual(audit["status"], "invalid_ours_evidence")

    def test_unhealthy_loss_dominates_all_healthy_performance_range(self):
        ours, peers = self.fixture()
        ours[0]["healthy"] = False
        audit = compute_relative_objective(ours, peers)
        self.assertGreater(audit["loss"], 2 / .02 + 1.01)
        self.assertEqual(audit["unhealthy_ours_scenarios"], [ours[0]["scenario"]])

    def test_duplicate_mismatched_scenario_and_nonfinite_score_rejected(self):
        ours, peers = self.fixture()
        with self.assertRaises(ValueError):
            compute_relative_objective(ours + [ours[0]], peers)
        with self.assertRaises(ValueError):
            compute_relative_objective(ours, peers, original_score=float("nan"))
        peers["vert"][1]["attacked"] = False
        with self.assertRaises(ValueError):
            compute_relative_objective(ours, peers)


if __name__ == "__main__":
    unittest.main()
