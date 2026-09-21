"""Costed search and unchanged scientific/checkpoint protocol; no deadline."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cifar_budgeted_search as search
import run_cifar_six_relative_best as runner
from cifar_relative_best_gate import ACCURACY_TARGET, ASR_TARGET, PERFORMANCE_TARGET
import run_cifar_six_with_progress as progress


class SearchTests(unittest.TestCase):
    def test_regeneration_production_validation_and_frozen_protocol(self):
        spec = search.build_spec()
        self.assertEqual(spec, json.loads(search.OUTPUT.read_text()))
        self.assertEqual(spec, runner.load_spec(search.OUTPUT))
        parent = json.loads(search.BASE.read_text())
        for key in ("dataset", "shared_parameters", "validation", "final", "objective", "gates",
                    "numerics", "promotion", "fallback_candidates", "selection_metrics"):
            self.assertEqual(spec[key], parent[key], key)
        self.assertEqual(spec["accuracy_target"], ACCURACY_TARGET)
        self.assertEqual(spec["asr_target"], ASR_TARGET)
        self.assertEqual(spec["performance_target"], PERFORMANCE_TARGET)
        expanded_v6 = json.loads((search.REPO / "configs/cifar10_six_relative_asr_five_day_v6.json").read_text())
        self.assertEqual(search.build_spec_v6(), expanded_v6)
        self.assertEqual(spec["candidates"], expanded_v6["candidates"])
        for method, old in parent["candidates"].items():
            self.assertEqual(spec["candidates"][method][:len(old)], old)
        self.assertNotEqual(spec["output_dir"], parent["output_dir"])
        self.assertEqual(spec["run_budget"]["candidate_counts"],
                         dict(sm9rrs=28, vert=6, alignins=6, krum=1, ding13=1, fedavg=1))
        self.assertEqual(len(runner.build_tasks(spec, "validation")), 1290)
        self.assertEqual(spec["run_budget"]["maximum_total_runs"], 1470)

    def test_recommended_combinations_are_real_distinct_original_candidates(self):
        spec = search.build_spec()
        rows = {c["candidate_id"]: c for c in spec["candidates"]["sm9rrs"]}
        anchor = rows["sm9rrs-v10-014"]["parameters"]
        differences = {}
        for i in range(101, 115):
            row = rows[f"sm9rrs-v10-{i}"]
            self.assertEqual(row["variant"], "original")
            differences[i] = {k: v for k, v in row["parameters"].items() if v != anchor[k]}
        self.assertEqual(differences[101], dict(detector_drift_allowance=.85, detector_drift_threshold=1.))
        self.assertEqual(differences[102], dict(detector_history_threshold=.8, detector_history_confirm=3))
        self.assertEqual(differences[103], {**differences[101], **differences[102]})
        self.assertEqual(differences[104], {**differences[103], "detector_distance_threshold": 1.10})

    def test_estimate_includes_slowest_new_cost_formal_and_margin(self):
        spec = search.build_spec()
        estimate = spec["runtime_estimate"]
        self.assertGreater(estimate["formal_reserve_hours"], 5)
        self.assertAlmostEqual(estimate["planned_total_hours_with_margin"],
                               estimate["training_hours"] * 1.2 + 1.)
        self.assertLess(estimate["planned_total_hours_with_margin"], 120)
        self.assertGreater(estimate["planned_total_hours_with_margin"], 110)
        timing = json.loads(search.TIMING.read_text())
        cost = estimate["new_candidate_seconds_by_method"]["sm9rrs"]
        self.assertEqual(cost, max(v["mean_seconds"] for c, v in timing["candidates"].items()
                                   if c.startswith("sm9rrs-")))
        spec["candidates"]["sm9rrs"].append({"candidate_id": "sm9rrs-v10-999"})
        higher = search.estimate_cost(spec, timing)
        self.assertGreater(higher["training_hours"], estimate["training_hours"])
        self.assertEqual(higher["formal_reserve_hours"], estimate["formal_reserve_hours"])


class LauncherTests(unittest.TestCase):
    def test_no_deadline_and_round_checkpoints_preserved(self):
        spec = search.build_spec()
        self.assertNotIn("runtime_budget", spec)
        self.assertEqual(spec["shared_parameters"]["checkpoint_interval"], 1)
        self.assertFalse(spec["shared_parameters"]["early_stop"])
        self.assertEqual(spec["runtime_estimate"]["reference_gpu_lanes"], 7)

    def test_plan_only_prints_estimate_without_gpu_or_training(self):
        with mock.patch.object(progress, "discover_gpus") as probe, \
                mock.patch.object(progress, "monitor") as monitor, \
                mock.patch.object(progress.subprocess, "call", return_value=0), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(progress.main(["--config", str(search.OUTPUT), "--plan-only"]), 0)
        probe.assert_not_called()
        monitor.assert_not_called()
        self.assertIn("RUNTIME_ESTIMATE", out.getvalue())

    def test_actual_search_can_start_with_two_cards_and_render_normally(self):
        with tempfile.TemporaryDirectory() as directory:
            cards = [{"logical_device": f"cuda:{i}", "name": "A100",
                      "compute_capability": [8, 0], "total_memory_bytes": 40 * 1024 ** 3,
                      "initialization_ok": True, "free_memory_bytes": 40 * 1024 ** 3} for i in range(2)]
            def monitor(command, devices, **kwargs):
                self.assertEqual(devices, ["cuda:0", "cuda:1"])
                self.assertNotIn("budget", kwargs)
                kwargs["phase_state"]["phase"] = "final"
                return 0
            with mock.patch.object(progress, "discover_gpus", return_value=cards), \
                    mock.patch.object(progress, "monitor", side_effect=monitor), \
                    mock.patch.object(progress, "render_final_report", return_value=True) as render, \
                    mock.patch("sys.stdout", new_callable=io.StringIO):
                output = Path(directory) / "study"
                self.assertEqual(progress.main(["--config", str(search.OUTPUT), "--output", str(output)]), 0)
                render.assert_called_once_with(output.resolve(), training_exit_code=0)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
