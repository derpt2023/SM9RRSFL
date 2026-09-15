"""The probe must stop cleanly, preserve state, and leave training unchanged."""
import contextlib
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import torch

import diagnose_cifar_nonfinite as probe
from sm9rrsfl.datasets import make_synthetic_mnist_like, stratified_training_three_way_split
from sm9rrsfl import fl
from sm9rrsfl.experiments import write_summary, write_rounds


class NonfiniteProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.dataset = stratified_training_three_way_split(
            make_synthetic_mnist_like(train_samples=400, test_samples=40, seed=18),
            seed=19, train_fraction=.8, calibration_fraction=.1,
            attack_fraction=.1).calibration_dataset
        cls.config = fl.ExperimentConfig(
            method="vert", num_clients=4, rounds=4, malicious_ratio=.5,
            local_epochs=1, batch_size=16, lr=.01, attack_start_round=3,
            detector_window=3, vert_history_window=2, vert_predict_epochs=1,
            attack_target_count=1, attack_boost=1., compute_backend="torch",
            device="cpu", early_stop=False, seed=20)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_bounded_prefix_and_checkpoint_match_uninstrumented_run(self):
        baseline = fl.run_experiment(self.dataset, self.config)
        reference = {r.round: {k: str(v) for k, v in asdict(r).items()}
                     for r in baseline.records}
        original = fl._alternating_minimization_client_delta
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temp) / "probe"
            result = probe.run_probe(self.dataset, self.config, output, 3, reference)
            self.assertEqual(result["status"], "no_nonfinite_client_update_within_limit")
            self.assertEqual(result["config"]["rounds"], 4)
            self.assertEqual(result["last_completed_round"], 3)
            self.assertEqual(result["prefix_metric_mismatches"], [])
            with (output / "checkpoint_round_002.pickle").open("rb") as handle:
                state = pickle.load(handle)
            self.assertEqual(state["completed_round"], 2)
            resumed = fl.run_experiment(self.dataset, self.config, resume_state=state)
            self.assertEqual([asdict(r) for r in resumed.records],
                             [asdict(r) for r in baseline.records])
        self.assertIs(fl._alternating_minimization_client_delta, original)

    def test_finite_input_overflow_captures_client_and_replay_phase(self):
        # Deliberately extreme attack on synthetic CPU data, never production parameters.
        config = replace(self.config, attack_boost=1e30)
        original = fl._alternating_minimization_client_delta
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temp) / "probe"
            report = probe.run_probe(self.dataset, config, output, 3, {})
            self.assertEqual(report["status"], "nonfinite_client_update_captured")
            self.assertEqual(report["fault"]["round"], 3)
            self.assertTrue(report["fault"]["is_malicious"])
            self.assertTrue(report["fault"]["input_params"]["finite"])
            self.assertFalse(report["fault"]["delta"]["finite"])
            self.assertEqual(report["client_replay"]["status"], "first_nonfinite_observed")
            self.assertIn(report["client_replay"]["phase"], ("attack_target", "attack_stealth"))
            self.assertTrue((output / "checkpoint_round_002.pickle").exists())
            self.assertTrue((output / "fault_client_input.npz").exists())
        self.assertIs(fl._alternating_minimization_client_delta, original)

    def test_ours_prefix_unchanged_and_mismatch_values_are_reported(self):
        config = replace(self.config, method="sm9rrs", crypto_mode="simulated",
                         attack_start_round=4)
        baseline = fl.run_experiment(self.dataset, config)
        reference = {r.round: {k: str(v) for k, v in asdict(r).items()}
                     for r in baseline.records}
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temp) / "ours"
            result = probe.run_probe(self.dataset, config, output, 4, reference)
            self.assertEqual(result["prefix_metric_mismatches"], [])
            self.assertEqual(json.loads((output / "probe_rounds.json").read_text()),
                             [asdict(r) for r in baseline.records])
            # Change only the comparison fixture; diagnosis must report the
            # original/probe values without influencing the actual trajectory.
            reference[1]["accuracy"] = "-1"
            result = probe.run_probe(self.dataset, config, Path(temp) / "mismatch", 1, reference)
            self.assertEqual(result["prefix_metric_mismatches"], [{
                "round": 1, "fields": ["accuracy"],
                "values": {"accuracy": {"original": "-1", "probe": baseline.records[1].accuracy}},
            }])

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            path = output / "environment.json"
            path.write_text("original")
            with self.assertRaises(FileExistsError):
                probe.run_probe(self.dataset, self.config, output, 3, {}, environment={"new": True})
            self.assertEqual(path.read_text(), "original")

    def test_recorded_candidates_with_same_scenario_are_not_mixed(self):
        baseline = fl.run_experiment(self.dataset, self.config)
        first = replace(baseline, config=replace(self.config, device="cuda:7"))
        second = replace(baseline, config=replace(self.config, device="cuda:7", vert_predict_epochs=5))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = root / ".tuning_state/validation/fixture"
            state.mkdir(parents=True)
            entries = [{"candidate_id": f"vert-{i:03d}",
                        "config": asdict(replace(r.config, device="cuda:3"))}
                       for i, r in enumerate((first, second), 1)]
            probe.write_json(state / "run_manifest.json",
                             {"fingerprint": "fixture", "candidates": entries})
            probe.write_json(root / "tuning_progress.json", {"phases": {"validation": {
                "fingerprint": "fixture", "status": "complete", "total": 2, "completed": 2}}})
            write_summary(state / "summary.csv", [first, second])
            write_rounds(state / "rounds.csv", [first, second])
            rows = probe.read_csv(state / "summary.csv")
            for i, row in enumerate(rows, 1):
                row["candidate_id"] = f"vert-{i:03d}"
            import csv
            with (root / "validation_results.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            chosen, _, records = probe.select_recorded_run(root, "vert-002", "iid", .5, 20)
            self.assertEqual(chosen.vert_predict_epochs, 5)
            self.assertEqual(chosen.device, "cuda:3")
            self.assertEqual(len(records), 5)

    def test_data_digest_mismatch_stops_before_training(self):
        manifest = {"dataset": {"name": "cifar10", "data_dir": "unused", "seed": 18,
                                 "train_content_digest": "wrong"},
                    "tuning_context": {"validation_fraction": .1, "split_seed": 19}}
        with mock.patch.object(probe, "load_image_dataset", return_value=self.dataset):
            with self.assertRaisesRegex(ValueError, "train_content_digest differs"):
                probe.load_verified_data(manifest)

    def test_nonfinite_diagnostic_values_are_explicit_json_strings(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            probe.write_json(path, {"loss": float("nan"), "gradient": float("inf")})
            self.assertEqual(json.loads(path.read_text()), {"loss": "nan", "gradient": "inf"})


if __name__ == "__main__":
    unittest.main()
