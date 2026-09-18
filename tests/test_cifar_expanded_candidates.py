"""Validate the search boundary without data downloads or model training."""
from copy import deepcopy
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import unittest
from unittest import mock

import cifar_expanded_candidates as search
from sm9rrsfl.fair_tuning import METHOD_TUNABLE_PARAMETERS
from sm9rrsfl.fl import ExperimentConfig
from sm9rrsfl.ours_policy import bounded_candidates
from sm9rrsfl.performance_target import PerformanceTarget


REPO = Path(__file__).resolve().parents[1]


def parameters_key(parameters):
    return json.dumps(parameters, sort_keys=True)


class ExpandedCandidateTests(unittest.TestCase):
    def setUp(self):
        self.previous = json.loads((REPO / "configs/cifar10_six_630_mean_v3.json").read_text())
        self.spec = search.build_spec(self.previous)

    def test_frozen_json_matches_generator_and_has_explicit_budget(self):
        frozen = json.loads(search.DEFAULT_CONFIG.read_text())
        self.assertEqual(frozen, self.spec)
        self.assertEqual(self.spec["run_budget"], {
            "candidate_counts": {"sm9rrs": 42, "vert": 24, "alignins": 18,
                                 "krum": 1, "ding13": 1, "fedavg": 1},
            "total_candidates": 87, "validation_runs_per_candidate": 30,
            "validation_runs": 2610, "formal_runs_if_ours_qualifies": 180,
            "maximum_total_runs": 2790,
        })

    def test_old_protocol_not_mutated_and_shared_training_and_score_stay_frozen(self):
        original = deepcopy(self.previous)
        search.build_spec(self.previous)
        self.assertEqual(self.previous, original)
        self.assertEqual(self.spec["shared_parameters"], self.previous["shared_parameters"])
        self.assertEqual(self.spec["dataset"], self.previous["dataset"])
        self.assertEqual(self.spec["objective"], self.previous["objective"])
        self.assertEqual(self.spec["gates"], self.previous["gates"])
        self.assertEqual(self.spec["numerics"], self.previous["numerics"])

    def test_every_candidate_uses_only_its_method_interface_and_is_valid(self):
        shared = ExperimentConfig(**self.spec["shared_parameters"])
        identities = []
        for method, candidates in self.spec["candidates"].items():
            effective_keys = []
            for candidate in candidates:
                with self.subTest(candidate=candidate["candidate_id"]):
                    self.assertLessEqual(set(candidate["parameters"]), METHOD_TUNABLE_PARAMETERS[method])
                    self.assertNotIn("detector_window", candidate["parameters"])
                    self.assertEqual(candidate["variant"], "original")
                    config = replace(shared, method=method, **candidate["parameters"])
                    config.validate()
                    identities.append(candidate["candidate_id"])
                    effective_keys.append(parameters_key({
                        name: getattr(config, name) for name in METHOD_TUNABLE_PARAMETERS[method]
                    }))
            self.assertEqual(len(effective_keys), len(set(effective_keys)), method)
        self.assertEqual(len(identities), len(set(identities)))

    def test_all_twelve_mnist_policies_and_all_six_cifar_policies_are_retained(self):
        ours = self.spec["candidates"]["sm9rrs"]
        keys = {parameters_key(candidate["parameters"]) for candidate in ours}
        self.assertLessEqual({parameters_key(p) for p in bounded_candidates(12)}, keys)
        for old in self.previous["candidates"]["sm9rrs"]:
            complete = {name: self.previous["shared_parameters"][name]
                        for name in search.OURS_POLICY_FIELDS}
            complete.update(old["parameters"])
            self.assertIn(parameters_key(complete), keys)

    def test_bridge_probes_each_change_exactly_one_legal_field(self):
        probes = [c for c in self.spec["candidates"]["sm9rrs"]
                  if c["origin"].startswith("cifar_anchor_one_factor:")]
        self.assertEqual(len(probes), 24)
        for probe in probes:
            changes = [name for name, value in probe["parameters"].items()
                       if value != search.CIFAR_OURS_ANCHOR[name]]
            self.assertEqual(changes, [probe["origin"].split(":")[1]])

    def test_baseline_grids_preserve_prior_policies_without_extra_ratio_knowledge(self):
        for method in ("vert", "alignins"):
            for old in self.previous["candidates"][method]:
                self.assertTrue(any(all(candidate["parameters"][k] == v
                                        for k, v in old["parameters"].items())
                                    for candidate in self.spec["candidates"][method]))
        for history, epochs, rate in product((5, 7, 10), (5, 20), (.0005, .001)):
            self.assertTrue(any(c["parameters"]["vert_history_window"] == history
                                and c["parameters"]["vert_predict_epochs"] == epochs
                                and c["parameters"]["vert_predict_lr"] == rate
                                for c in self.spec["candidates"]["vert"]))
        for mpsa, sparsity, tda in product((.5, 1.), (.1, .3, .5), (.5, 1.)):
            self.assertTrue(any(c["parameters"] == {
                "alignins_mpsa_radius": mpsa, "alignins_sparsity": sparsity,
                "alignins_tda_radius": tda,
            } for c in self.spec["candidates"]["alignins"]))
        for candidate in self.spec["candidates"]["vert"]:
            self.assertEqual(candidate["parameters"]["vert_top_k"], 0)
            self.assertEqual(candidate["parameters"]["vert_projection_dim"], 128)
            self.assertIs(candidate["parameters"]["vert_use_ratio_prior"], False)

    def test_parameterless_baselines_do_not_get_duplicate_fake_choices(self):
        for method in ("krum", "ding13", "fedavg"):
            self.assertEqual(len(self.spec["candidates"][method]), 1)
            self.assertEqual(self.spec["candidates"][method][0]["parameters"], {})
        for method, cid in self.spec["fallback_candidates"].items():
            self.assertEqual(cid, self.spec["candidates"][method][0]["candidate_id"])

    def test_gate_uses_mnist_defaults_and_independent_new_experiment_identity(self):
        self.assertEqual(PerformanceTarget.parse(self.spec["performance_target"]), PerformanceTarget())
        self.assertEqual(self.spec["schema_version"], 4)
        self.assertEqual(self.spec["validation"]["seeds"], [1001, 1002, 1003])
        self.assertEqual(self.spec["final"]["seeds"], [1101, 1102, 1103])
        prior_seeds = set(self.previous["validation"]["seeds"] + self.previous["final"]["seeds"])
        new_seeds = self.spec["validation"]["seeds"] + self.spec["final"]["seeds"]
        self.assertEqual(len(new_seeds), len(set(new_seeds)))
        self.assertFalse(prior_seeds.intersection(new_seeds))
        self.assertNotEqual(self.spec["output_dir"], self.previous["output_dir"])
        self.assertNotIn("mean_dual_gate", self.spec)
        self.assertEqual(self.spec["promotion"]["required_healthy_methods"], ["sm9rrs"])

    def test_candidates_do_not_read_outputs_or_other_files_at_runtime(self):
        with mock.patch.object(Path, "read_text", side_effect=AssertionError("No file input permitted")):
            one = search.build_candidates()
            two = search.build_candidates()
        one["sm9rrs"][0]["parameters"]["detector_subspace_dim"] = 9
        self.assertNotEqual(one, two)
        self.assertEqual(two["sm9rrs"][0]["parameters"]["detector_subspace_dim"], 2)


if __name__ == "__main__":
    unittest.main()
