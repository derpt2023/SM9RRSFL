import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sm9rrsfl.config_runner import (
    ConfigError,
    config_file_to_argv,
    load_experiment_config,
    main,
    parameters_to_argv,
)
from sm9rrsfl.experiments import parse_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigRunnerTest(unittest.TestCase):
    def test_example_configuration_maps_to_existing_cli(self):
        argv = config_file_to_argv(PROJECT_ROOT / "configs" / "experiment.json")
        args = parse_args(argv)

        self.assertIn(args.dataset, {"mnist", "cifar10", "synthetic"})
        self.assertTrue(args.methods)
        self.assertTrue(args.ratios)
        self.assertTrue(args.client_counts or [args.num_clients])
        self.assertTrue(args.partitions or [args.partition])
        self.assertEqual(args.ours_parameter_mode, "auto")
        self.assertIn("alignins", args.methods)
        self.assertNotIn("fedredefense", args.methods)
        self.assertAlmostEqual(args.alignins_sparsity, 0.3)
        self.assertAlmostEqual(args.alignins_tda_radius, 1.0)
        self.assertAlmostEqual(args.alignins_mpsa_radius, 1.0)
        self.assertEqual(args.ratios, [0.0, 0.2, 0.4, 0.6, 0.8])
        self.assertEqual(args.calibration_ratios, [0.0, 0.1, 0.3, 0.5, 0.7])
        self.assertEqual(args.calibration_candidate_budget, 12)

    def test_ratio_range_separates_calibration_and_formal_attacked_ratios(self):
        args = parse_args(["--ratio-range", "0", "0.8", "5"])

        self.assertEqual(args.ratios, [0.0, 0.2, 0.4, 0.6, 0.8])
        self.assertEqual(args.calibration_ratios, [0.0, 0.1, 0.3, 0.5, 0.7])
        self.assertFalse(
            set(args.ratios[1:]) & set(args.calibration_ratios[1:])
        )

        for argv in (
            ["--ratio-range", "0", "1", "5"],
            ["--ratio-range", "0", "0.8", "5", "--ratios", "0", "0.4"],
        ):
            with self.subTest(argv=argv), mock.patch(
                "sys.stderr", new=io.StringIO()
            ), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_removed_attack_hard_constraint_interfaces_are_rejected(self):
        for name in (
            "ASR",
            "attack_FP",
            "attack_recall",
            "calibration_max_asr",
            "calibration_max_attack_false_positive_rate",
            "calibration_min_three_round_recall",
            "fedre_threshold",
            "fedre_teacher_lr",
        ):
            with self.subTest(name=name), self.assertRaisesRegex(
                ConfigError,
                "unknown experiment parameter",
            ):
                parameters_to_argv({name: 0.5})

        args = parse_args([])
        self.assertEqual(args.calibration_min_round_completion_rate, 1.0)
        self.assertEqual(args.calibration_max_nonfinite_updates, 0)

    def test_boolean_aliases_and_store_false_option_are_supported(self):
        argv = parameters_to_argv(
            {
                "dataset": "synthetic",
                "methods": ["vert", "fedavg"],
                "ratios": [0.0, 0.6],
                "early_stop": False,
                "visualizations": False,
                "progress": False,
                "resume": False,
                "download": True,
                "vert_use_ratio_prior": True,
            }
        )
        args = parse_args(argv)

        self.assertTrue(args.no_early_stop)
        self.assertTrue(args.no_visualizations)
        self.assertTrue(args.no_progress)
        self.assertFalse(args.resume)
        self.assertTrue(args.download)
        self.assertTrue(args.vert_use_ratio_prior)

    def test_paper_parameter_aliases_map_to_their_cli_destinations(self):
        argv = parameters_to_argv(
            {
                "K": 10,
                "q": 2,
                "beta": 0.8,
                "kappa": 0.75,
                "h": 4.5,
                "C_tol": 4,
            }
        )
        args = parse_args(argv)

        self.assertEqual(args.detector_window, 10)
        self.assertEqual(args.detector_subspace_dim, 2)
        self.assertAlmostEqual(args.detector_drift_memory, 0.8)
        self.assertAlmostEqual(args.detector_drift_allowance, 0.75)
        self.assertAlmostEqual(args.detector_drift_threshold, 4.5)
        self.assertEqual(args.suspicion_remove_after, 4)

    def test_paper_alias_and_canonical_key_cannot_conflict(self):
        conflicts = (
            {"K": 10, "detector_window": 5},
            {"q": 2, "detector_subspace_dim": 3},
            {"beta": 0.9, "detector_drift_memory": 0.8},
            {"kappa": 1.0, "detector_drift_allowance": 0.5},
            {"h": 5.0, "detector_drift_threshold": 4.0},
        )
        for parameters in conflicts:
            with self.subTest(parameters=parameters):
                with self.assertRaisesRegex(ConfigError, "configured more than once"):
                    parameters_to_argv(parameters)

    def test_unknown_or_duplicate_parameter_is_rejected(self):
        with self.assertRaisesRegex(ConfigError, "unknown experiment parameter"):
            parameters_to_argv({"not_a_parameter": 1})
        with self.assertRaisesRegex(ConfigError, "configured more than once"):
            parameters_to_argv({"early_stop": True, "no_early_stop": False})
        with self.assertRaisesRegex(ConfigError, "non-empty JSON array"):
            parameters_to_argv({"methods": []})

    def test_invalid_json_wrapper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrong_schema = Path(tmp) / "wrong-schema.json"
            wrong_schema.write_text(
                json.dumps({"schema_version": 1, "parameters": {}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "schema_version must be 3"):
                load_experiment_config(wrong_schema)

            unknown_root = Path(tmp) / "unknown-root.json"
            unknown_root.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "parameters": {},
                        "unexpected": True,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "unknown configuration root key"):
                load_experiment_config(unknown_root)

    def test_dry_run_prints_source_command_and_resolved_parameters(self):
        config = PROJECT_ROOT / "configs" / "experiment.json"
        with mock.patch("sys.stdout", new=io.StringIO()) as output:
            main(["--config", str(config), "--dry-run"])

        text = output.getvalue()
        self.assertIn(f"config_file={config.resolve()}", text)
        self.assertIn("experiment_command=", text)
        self.assertIn("config_execution_request=", text)
        self.assertIn("progress=live", text)
        self.assertIn("resolved_parameters=", text)
        self.assertIn("effective_detector_parameters=", text)
        self.assertIn('"effective_attack_start_round": 12', text)
        self.assertIn('"status": "pending_global_offline_calibration"', text)
        self.assertIn('"detector_subspace_dim": "<pending_global_offline_calibration>"', text)
        self.assertIn('"dataset":', text)

    def test_auto_mode_rejects_every_manual_ours_parameter_but_allows_k(self):
        from dataclasses import asdict
        from sm9rrsfl.ours_policy import OursParameters
        base = {"methods": ["sm9rrs"], "ours_parameter_mode": "auto", "early_stop": False, "K": 7}
        self.assertEqual(parse_args(parameters_to_argv(base)).detector_window, 7)
        for name, value in asdict(OursParameters()).items():
            with self.subTest(name=name), mock.patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(parameters_to_argv({**base, name: value}))
        for name in ("g0", "theta_adj", "theta_anc", "C_max"):
            with self.assertRaises(ConfigError):
                parameters_to_argv({name: 3})

    def test_non_dry_run_dispatches_to_authoritative_experiment_entry(self):
        config = PROJECT_ROOT / "configs" / "experiment.json"
        with (
            mock.patch("sys.stdout", new=io.StringIO()),
            mock.patch("sm9rrsfl.config_runner.run_experiments") as run,
        ):
            main(["--config", str(config)])

        run.assert_called_once()
        experiment_argv = run.call_args.args[0]
        resolved = parse_args(experiment_argv)
        expected = parse_args(config_file_to_argv(config))
        self.assertEqual(vars(resolved), vars(expected))


if __name__ == "__main__":
    unittest.main()
