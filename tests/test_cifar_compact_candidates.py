"""Budget reduction preserves full validation, promotion and method identities."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest import mock

import cifar_compact_candidates as compact
import cifar_expanded_candidates as expanded
import run_cifar_six_mnist_gate as runner
from tests.test_cifar_six_pipeline import result_matrix


class CompactCandidateTests(unittest.TestCase):
    def setUp(self):
        self.spec = compact.build_spec()
        self.full = expanded.build_spec()

    def test_checked_in_config_is_reproducible_and_actual_budget_is_630_plus_180(self):
        self.assertEqual(json.loads(compact.DEFAULT_CONFIG.read_text()), self.spec)
        self.assertEqual(runner.load_spec(compact.DEFAULT_CONFIG), self.spec)
        self.assertEqual(self.spec["run_budget"], {
            "candidate_counts": {"sm9rrs": 10, "vert": 4, "alignins": 4,
                                 "krum": 1, "ding13": 1, "fedavg": 1},
            "total_candidates": 21, "validation_runs_per_candidate": 30,
            "validation_runs": 630, "formal_runs_if_ours_qualifies": 180,
            "maximum_total_runs": 810,
        })
        self.assertNotEqual(self.spec["output_dir"], self.full["output_dir"])

    def test_only_search_space_and_its_metadata_change(self):
        unchanged = {"schema_version", "dataset", "shared_parameters", "gates", "numerics",
                     "objective", "performance_target", "promotion", "validation", "final"}
        for field in unchanged:
            with self.subTest(field=field):
                self.assertEqual(self.spec[field], self.full[field])
        self.assertEqual(self.spec["validation"]["seeds"], [1001, 1002, 1003])
        self.assertFalse(set(self.spec["validation"]["seeds"]) & set(self.spec["final"]["seeds"]))

    def test_shortlist_retains_full_policies_without_duplicates_or_mutating_parent(self):
        parent = deepcopy(self.full)
        base = runner.fl.ExperimentConfig(**self.spec["shared_parameters"])
        for method, candidates in self.spec["candidates"].items():
            full_candidates = {c["candidate_id"]: c for c in parent["candidates"][method]}
            identities = []
            for candidate in candidates:
                original = {k: v for k, v in candidate.items() if k != "shortlist_reason"}
                self.assertEqual(original, full_candidates[candidate["candidate_id"]])
                self.assertTrue(candidate["shortlist_reason"])
                config = replace(base, method=method, **candidate["parameters"])
                config.validate()
                identities.append(runner.digest(runner.semantic_config(config)))
            self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(expanded.build_spec(), parent)

    def test_all_retained_candidates_still_receive_three_seeds_and_ten_full_scenarios(self):
        tasks = runner.build_tasks(self.spec, "validation")
        self.assertEqual(len(tasks), 630)
        self.assertEqual(len({t["task_id"] for t in tasks}), 630)
        for candidates in self.spec["candidates"].values():
            for candidate in candidates:
                subset = [t for t in tasks if t["candidate"]["candidate_id"] == candidate["candidate_id"]]
                self.assertEqual(len(subset), 30)
                self.assertEqual({t["config"]["seed"] for t in subset}, {1001, 1002, 1003})
                self.assertTrue(all(t["config"]["rounds"] == 100 for t in subset))
        report = runner.select_validation(self.spec, result_matrix(self.spec, tasks), tasks)
        self.assertEqual(report["status"], "qualified_for_final")
        final = runner.build_tasks(self.spec, "final", report["selected"])
        self.assertEqual(len(final), 180)
        self.assertEqual({t["method"] for t in final}, set(runner.ALL_METHODS))
        self.assertEqual({t["config"]["seed"] for t in final}, {1101, 1102, 1103})

    def test_prior_validation_is_provenance_and_not_required_on_training_station(self):
        original_read = Path.read_text

        def reject_outputs(path, *args, **kwargs):
            if "outputs" in path.parts:
                raise AssertionError("fresh configuration generation must not require historical results")
            return original_read(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", reject_outputs):
            generated = compact.build_spec()
        self.assertEqual(generated, self.spec)
        self.assertTrue(generated["search_design"]["historical_validation_used_for_search_design"])
        self.assertFalse(generated["search_design"]["historical_scores_reused_for_new_qualification"])
        self.assertFalse(generated["search_design"]["official_test_used_for_selection"])

    def test_fixed_fallback_is_declared_for_this_ordered_shortlist(self):
        self.assertEqual(self.spec["fallback_candidates"], {
            method: candidates[0]["candidate_id"] for method, candidates in self.spec["candidates"].items()
            if method != "sm9rrs"
        })
        self.assertEqual(self.spec["fallback_candidates"]["vert"], "vert-v10-014")
        self.assertEqual(self.spec["fallback_candidates"]["alignins"], "alignins-v10-008")
        self.assertNotIn("sm9rrs", self.spec["fallback_candidates"])


if __name__ == "__main__":
    unittest.main()
