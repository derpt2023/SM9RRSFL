"""Resource admission and bounded retry, without allocating CUDA memory."""
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import run_cifar_six_with_progress as progress


def card(index, *, name="4090D", free_mib=12000, initialized=True):
    return {"logical_device": f"cuda:{index}", "name": name,
            "compute_capability": [8, 9], "total_memory_bytes": 24 * 1024 ** 3,
            "initialization_ok": initialized, "free_memory_bytes": free_mib * 1024 ** 2}


def result(payload, returncode=0):
    return subprocess.CompletedProcess([], returncode,
                                       "GPU_PROBE_JSON " + json.dumps(payload), "")


class PreflightTests(unittest.TestCase):
    def test_failed_card_does_not_hide_later_cards(self):
        replies = [result(3), result(card(0)),
                   subprocess.CompletedProcess([], 1, "", "CUDA error: out of memory"),
                   result(card(2))]
        with mock.patch.object(progress.subprocess, "run", side_effect=replies) as probe:
            found = progress.discover_gpus(Path("."))
        self.assertEqual(probe.call_count, 4)
        self.assertEqual([row["logical_device"] for row in found], ["cuda:0", "cuda:1", "cuda:2"])
        self.assertFalse(found[1]["initialization_ok"])
        self.assertIn("out of memory", found[1]["preflight_error"])
        self.assertNotIn("name", found[1])
        self.assertTrue(found[2]["initialization_ok"])
        chosen, skipped = progress.select_devices(found, ["auto"])
        self.assertEqual(chosen, ["cuda:0", "cuda:2"])
        self.assertIn("initialization failed", skipped[0]["reason"])
        self.assertIsNone(skipped[0]["free_memory_bytes"])

    def test_timeout_and_malformed_card_do_not_hide_later_cards(self):
        replies = [result(3), subprocess.TimeoutExpired("probe", 60),
                   result({"logical_device": "cuda:99"}), result(card(2))]
        with mock.patch.object(progress.subprocess, "run", side_effect=replies):
            found = progress.discover_gpus(Path("."))
        self.assertEqual(progress.select_devices(found, ["auto"])[0], ["cuda:2"])
        self.assertFalse(found[0]["initialization_ok"])
        self.assertFalse(found[1]["initialization_ok"])

    def test_auto_uses_all_healthy_compatible_cards(self):
        cards = [card(0, free_mib=4096), card(1, free_mib=4095), card(2), card(3, name="A100")]
        chosen, skipped = progress.select_devices(cards, ["auto"])
        self.assertEqual(chosen, ["cuda:0", "cuda:2"])
        self.assertIn("insufficient free", skipped[0]["reason"])
        self.assertEqual(skipped[0]["free_memory_bytes"], 4095 * 1024 ** 2)
        self.assertIn("incompatible", skipped[1]["reason"])

    def test_only_healthy_cards_contribute_to_largest_group(self):
        cards = [card(0, name="A100"), card(1, free_mib=10), card(2, free_mib=0)]
        self.assertEqual(progress.select_devices(cards, ["auto"])[0], ["cuda:0"])
        with self.assertRaisesRegex(ValueError, "matches the saved experiment"):
            progress.select_devices(cards, ["auto"], card(7))

    def test_explicit_low_memory_is_error_without_partial_admission(self):
        with self.assertRaisesRegex(ValueError, "no devices were silently removed"):
            progress.select_devices([card(0), card(1, free_mib=20)], ["cuda:0", "cuda:1"])
        self.assertEqual(progress.select_devices([card(0), card(1, free_mib=20)], ["cuda:0"])[0], ["cuda:0"])

    def test_memory_unavailable_is_not_treated_as_free_even_with_zero_threshold(self):
        missing = card(0)
        del missing["free_memory_bytes"]
        with self.assertRaisesRegex(ValueError, "free GPU memory unavailable"):
            progress.select_devices([missing], ["auto"], min_free_memory_mib=0)
        self.assertEqual(progress.select_devices([card(0, free_mib=0)], ["auto"], min_free_memory_mib=0)[0], ["cuda:0"])
        for invalid in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                progress.select_devices([card(0)], ["auto"], min_free_memory_mib=invalid)

    def test_cli_rejects_invalid_threshold_before_probing(self):
        with mock.patch.object(progress, "discover_gpus") as probe, mock.patch("sys.stderr", new_callable=io.StringIO):
            for value in ("-1", "inf", "nan"):
                with self.assertRaises(SystemExit):
                    progress.main(["--min-free-gpu-memory-mib", value])
        probe.assert_not_called()

    def test_cli_reports_actual_exclusions_and_consumes_wrapper_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = Path(directory) / "config.json"
            spec.write_text(json.dumps({"output_dir": directory}))
            out = io.StringIO()
            with mock.patch.object(progress, "discover_gpus", return_value=[card(0), card(1, free_mib=7000)]), \
                    mock.patch.object(progress, "monitor", return_value=0) as monitor, mock.patch("sys.stdout", out):
                progress.main(["--config", str(spec), "--min-free-gpu-memory-mib", "8000"])
            command, chosen = monitor.call_args.args
            self.assertEqual(chosen, ["cuda:0"])
            self.assertNotIn("--min-free-gpu-memory-mib", command)
            payload = json.loads(out.getvalue().splitlines()[0].split(" ", 1)[1])
            self.assertEqual(payload["excluded"][0]["logical_device"], "cuda:1")
            self.assertEqual(payload["minimum_free_memory_mib"], 8000)
            self.assertIn("does not reserve memory", out.getvalue())


class ResourceRetryTests(unittest.TestCase):
    def invoke(self, discoveries, responses, *, devices=None, phases=None):
        with tempfile.TemporaryDirectory() as directory:
            spec = Path(directory) / "config.json"
            spec.write_text(json.dumps({"output_dir": directory}))
            commands = []
            def monitor(command, chosen, **kwargs):
                commands.append((command, list(chosen)))
                code, failed_devices = responses[len(commands) - 1]
                kwargs["oom_devices"].update(failed_devices)
                if phases:
                    kwargs["phase_state"]["phase"] = phases[len(commands) - 1]
                return code
            out = io.StringIO()
            with mock.patch.object(progress, "discover_gpus", side_effect=discoveries) as probe, \
                    mock.patch.object(progress, "monitor", side_effect=monitor), mock.patch("sys.stdout", out):
                args = ["--config", str(spec)]
                if devices:
                    args.extend(["--devices", *devices])
                code = progress.main(args)
            return code, commands, out.getvalue(), probe.call_count

    def test_oom_retries_same_output_on_remaining_cards_and_rechecks_memory(self):
        initial = [card(0), card(1), card(2)]
        refreshed = [card(0), card(1), card(2, free_mib=4)]
        code, commands, out, probes = self.invoke([initial, refreshed], [(75, {"cuda:0"}), (0, set())])
        self.assertEqual(code, 0)
        self.assertEqual(probes, 2)
        self.assertEqual([chosen for _, chosen in commands], [["cuda:0", "cuda:1", "cuda:2"], ["cuda:1"]])
        self.assertEqual(commands[0][0][:-4], commands[1][0][:-2])
        self.assertIn('"failure_records_and_checkpoints": "preserved"', out)
        self.assertIn('"reason": "insufficient free GPU memory', out)

    def test_explicit_selection_has_visible_resource_retry(self):
        code, commands, out, _ = self.invoke([[card(0), card(1)], [card(0), card(1)]],
                                             [(75, {"cuda:1"}), (0, set())], devices=["cuda:0", "cuda:1"])
        self.assertEqual(code, 0)
        self.assertEqual(commands[1][1], ["cuda:0"])
        self.assertIn("GPU_RESOURCE_RETRY", out)
        self.assertIn('"excluded_after_oom": ["cuda:1"]', out)

    def test_formal_oom_retry_does_not_rerun_validation_or_reselect_parameters(self):
        cards = [card(0), card(1)]
        code, commands, out, _ = self.invoke([cards, cards], [(75, {"cuda:0"}), (0, set())],
                                             phases=["final", "final"])
        self.assertEqual(code, 0)
        first, second = commands[0][0], commands[1][0]
        self.assertNotIn("--phase", first)
        self.assertEqual(second[second.index("--phase") + 1], "final")
        self.assertIn('"phase_at_oom": "final"', out)
        for args in (["--phase=all"], ["--phase", "all"], ["--phase", "final"]):
            self.assertEqual(progress.forwarded_phase(args, "final"), ["--phase", "final"])

    def test_every_card_can_be_excluded_only_once(self):
        cards = [card(0), card(1)]
        code, commands, out, probes = self.invoke([cards, cards], [(75, {"cuda:0"}), (75, {"cuda:1"})])
        self.assertEqual(code, 75)
        self.assertEqual(len(commands), 2)
        self.assertEqual(probes, 2)
        self.assertIn("all initially admitted GPUs", out)

    def test_interrupt_or_non_resource_exit_is_not_retried(self):
        for exit_code, failed in ((130, {"cuda:0"}), (7, {"cuda:0"}), (75, set())):
            with self.subTest(code=exit_code):
                code, commands, _, probes = self.invoke([[card(0), card(1)]], [(exit_code, failed)])
                self.assertEqual(code, exit_code)
                self.assertEqual(len(commands), 1)
                self.assertEqual(probes, 1)

    def test_retry_never_changes_hardware_group_or_adds_unselected_cards(self):
        initial = [card(0), card(1), card(2, name="A100")]
        refreshed = [card(0), card(1, free_mib=0), card(2, name="A100")]
        code, commands, out, _ = self.invoke([initial, refreshed], [(75, {"cuda:0"})])
        self.assertEqual(code, 75)
        self.assertEqual(len(commands), 1)
        self.assertIn("remaining-card preflight failed", out)

    def test_unmapped_oom_stops_without_loop(self):
        code, commands, out, probes = self.invoke([[card(0)]], [(75, {"cuda:99"})])
        self.assertEqual(code, 75)
        self.assertEqual(len(commands), 1)
        self.assertEqual(probes, 1)
        self.assertIn("did not identify a new active GPU", out)


if __name__ == "__main__":
    unittest.main()
