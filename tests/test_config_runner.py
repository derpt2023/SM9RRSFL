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
                json.dumps({"schema_version": 2, "parameters": {}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "schema_version must be 1"):
                load_experiment_config(wrong_schema)

            unknown_root = Path(tmp) / "unknown-root.json"
            unknown_root.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
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
        self.assertIn("resolved_parameters=", text)
        self.assertIn('"dataset":', text)

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
