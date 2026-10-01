"""Compatibility entry delegates schema 8 before touching legacy progress."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

import run_cifar_six_with_progress as legacy


class AdaptiveDispatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.config = Path(temporary.name) / "config.json"
        self.module = ModuleType("run_adaptive_with_progress")
        self.module.main = mock.Mock(return_value=17)

    def write_spec(self, protocol, schema=8):
        self.config.write_text(json.dumps({"schema_version": schema, "protocol": protocol}))

    def dispatch(self, arguments):
        # Any old preflight, training subprocess, monitor or report attempt is
        # a regression: these responsibilities belong to the adaptive wrapper.
        with mock.patch.dict(sys.modules, {self.module.__name__: self.module}), \
                mock.patch.object(legacy, "discover_gpus", side_effect=AssertionError("legacy GPU probe")), \
                mock.patch.object(legacy, "monitor", side_effect=AssertionError("legacy monitor")), \
                mock.patch.object(legacy, "render_final_report", side_effect=AssertionError("legacy report")), \
                mock.patch.object(legacy.subprocess, "call", side_effect=AssertionError("legacy subprocess")):
            return legacy.main(arguments)

    def test_known_protocols_preserve_all_arguments_before_legacy_phase_parser(self):
        for protocol in ("cifar-resnet18-gn-tpe-v1", "fashion-mnist-resnet18-gn-tpe-v1"):
            with self.subTest(protocol=protocol):
                self.write_spec(protocol)
                argv = ["--phase", "search", "--config", str(self.config),
                        "--devices", "cuda:0", "cuda:2", "--progress-mode", "live",
                        "--progress-interval", ".5", "--progress-log-interval", "30",
                        "--data-dir", "data/custom", "--output", "outputs/independent"]
                self.assertEqual(self.dispatch(argv), 17)
                self.module.main.assert_called_with(argv)

    def test_help_plan_only_and_adaptive_invalid_options_are_delegated_unchanged(self):
        self.write_spec("cifar-resnet18-gn-tpe-v1")
        for options in (["--help"], ["--plan-only"], ["--report-only"],
                        ["--phase=validation"], ["--progress-refresh", ".2"]):
            with self.subTest(options=options):
                argv = [f"--config={self.config}", *options]
                self.assertEqual(self.dispatch(argv), 17)
                self.module.main.assert_called_with(argv)

    def test_none_argv_uses_current_command_line_without_losing_flags(self):
        self.write_spec("fashion-mnist-resnet18-gn-tpe-v1")
        argv = ["--config", str(self.config), "--phase", "final", "--devices=auto"]
        with mock.patch.object(sys, "argv", ["run_cifar_six_with_progress.py", *argv]):
            self.assertEqual(self.dispatch(None), 17)
        self.module.main.assert_called_once_with(argv)

    def test_unknown_schema8_protocol_is_rejected_without_v2_fallback(self):
        for protocol in ("unknown-study", None):
            with self.subTest(protocol=protocol):
                self.write_spec(protocol)
                errors = io.StringIO()
                with redirect_stderr(errors), self.assertRaises(SystemExit) as stopped:
                    self.dispatch(["--config", str(self.config), "--phase", "search"])
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn("unsupported schema-8 protocol", errors.getvalue())
        self.module.main.assert_not_called()

    def test_real_schema8_configs_have_explicit_compatible_routes(self):
        repo = Path(legacy.__file__).resolve().parent
        for name in ("cifar10_resnet18_gn_tpe_v8.json", "fashion_mnist_resnet18_gn_tpe_v8.json"):
            with self.subTest(name=name):
                argv = ["--config", str(repo / "configs" / name), "--plan-only"]
                self.assertEqual(self.dispatch(argv), 17)
                self.module.main.assert_called_with(argv)


if __name__ == "__main__":
    unittest.main()
