"""Small Ours expansion must not silently change the accepted protocol."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest import mock

import cifar_compact_candidates as compact
import cifar_compact_plus_candidates as plus
import cifar_expanded_candidates as expanded
import run_cifar_six_mnist_gate as runner
from tests.test_cifar_six_pipeline import result_matrix, synthetic_run


class CompactPlusCandidateTests(unittest.TestCase):
    def setUp(self):
        self.parent = compact.build_spec()
        self.spec = plus.build_spec()

    def test_frozen_config_loads_with_750_validation_and_180_formal_runs(self):
        self.assertEqual(json.loads(plus.DEFAULT_CONFIG.read_text()), self.spec)
        self.assertEqual(runner.load_spec(plus.DEFAULT_CONFIG), self.spec)
        self.assertEqual(self.spec["run_budget"], {
            "candidate_counts": {"sm9rrs": 14, "vert": 4, "alignins": 4,
                                 "krum": 1, "ding13": 1, "fedavg": 1},
            "total_candidates": 25, "validation_runs_per_candidate": 30,
            "validation_runs": 750, "formal_runs_if_ours_qualifies": 180,
            "maximum_total_runs": 930,
        })
        tasks = runner.build_tasks(self.spec, "validation")
        self.assertEqual(len(tasks), 750)
        self.assertEqual(len({t["task_id"] for t in tasks}), 750)
        for method, candidates in self.spec["candidates"].items():
            for candidate in candidates:
                rows = [t for t in tasks if t["candidate"]["candidate_id"] == candidate["candidate_id"]]
                self.assertEqual(len(rows), 30)
                self.assertEqual({r["config"]["seed"] for r in rows}, {1001, 1002, 1003})
                self.assertTrue(all(r["config"]["rounds"] == 100 for r in rows))

    def test_every_rule_and_original_candidate_is_preserved_exactly(self):
        mutable = {"name", "candidates", "output_dir", "run_budget", "search_design", "protocol_note"}
        self.assertEqual(set(self.spec), set(self.parent))
        for field in set(self.parent) - mutable:
            self.assertEqual(self.spec[field], self.parent[field], field)
        for method, candidates in self.parent["candidates"].items():
            self.assertEqual(self.spec["candidates"][method][:len(candidates)], candidates)
            if method != "sm9rrs":
                self.assertEqual(self.spec["candidates"][method], candidates)
        self.assertFalse(self.spec["promotion"]["require_mean_dual_best"])
        self.assertEqual(self.spec["fallback_candidates"], self.parent["fallback_candidates"])
        self.assertNotEqual(self.spec["output_dir"], self.parent["output_dir"])
        self.assertEqual(compact.build_spec(), self.parent)

    def test_added_policies_are_complete_legal_unique_and_cover_mnist_top_three(self):
        full = {c["candidate_id"]: c for c in expanded.build_candidates()["sm9rrs"]}
        additions = self.spec["candidates"]["sm9rrs"][10:]
        self.assertEqual([c["candidate_id"] for c in additions],
                         [f"sm9rrs-v10-{n:03d}" for n in (10, 11, 16, 39)])
        for candidate in additions:
            self.assertEqual({k: v for k, v in candidate.items() if k != "shortlist_reason"},
                             full[candidate["candidate_id"]])
        ids, configurations = set(), set()
        shared = runner.fl.ExperimentConfig(**self.spec["shared_parameters"])
        for method, candidates in self.spec["candidates"].items():
            for candidate in candidates:
                config = replace(shared, method=method, **candidate["parameters"])
                config.validate()
                identity = runner.digest(runner.semantic_config(config))
                self.assertNotIn(candidate["candidate_id"], ids)
                self.assertNotIn(identity, configurations)
                ids.add(candidate["candidate_id"])
                configurations.add(identity)
        self.assertLessEqual({f"sm9rrs-v10-{n:03d}" for n in (10, 11, 12)}, ids)
        anchor = full["sm9rrs-v10-015"]["parameters"]
        for suffix, field, value in ((16, "suspicion_remove_after", 5),
                                     (39, "detector_history_confirm", 3)):
            params = full[f"sm9rrs-v10-{suffix:03d}"]["parameters"]
            self.assertEqual({k: v for k, v in params.items() if anchor[k] != v}, {field: value})

    def test_new_candidate_must_qualify_and_can_be_selected_for_all_six_formal_arms(self):
        tasks = runner.build_tasks(self.spec, "validation")
        groups = result_matrix(self.spec, tasks)
        for cid, runs in groups.items():
            if runs[0].config.method == "sm9rrs":
                groups[cid] = [synthetic_run(run.config, accuracy=.8, asr=.06) for run in runs]
        failed = runner.select_validation(self.spec, groups, tasks)
        self.assertEqual(failed["status"], "needs_ours_target_development")
        self.assertNotIn("sm9rrs", failed["selected"])
        selected = "sm9rrs-v10-011"
        groups[selected] = [synthetic_run(run.config, accuracy=.81, asr=.04) for run in groups[selected]]
        qualified = runner.select_validation(self.spec, groups, tasks)
        self.assertEqual(qualified["status"], "qualified_for_final")
        self.assertEqual(qualified["selected"]["sm9rrs"], selected)
        self.assertTrue(qualified["ours_mnist_target_passed"])
        final = runner.build_tasks(self.spec, "final", qualified["selected"])
        self.assertEqual(len(final), 180)
        self.assertEqual({t["method"] for t in final}, set(runner.ALL_METHODS))
        self.assertEqual({t["config"]["seed"] for t in final}, {1101, 1102, 1103})
        self.assertEqual({t["candidate"]["candidate_id"] for t in final if t["method"] == "sm9rrs"}, {selected})

    def test_generation_does_not_depend_on_local_historical_outputs(self):
        read = Path.read_text

        def reject_outputs(path, *args, **kwargs):
            if "outputs" in path.parts:
                raise AssertionError("past results cannot be required on a clean training station")
            return read(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", reject_outputs):
            self.assertEqual(plus.build_spec(), self.spec)
        self.assertFalse(self.spec["search_design"]["historical_scores_reused_for_new_qualification"])
        self.assertFalse(self.spec["search_design"]["official_test_used_for_selection"])


if __name__ == "__main__":
    unittest.main()
