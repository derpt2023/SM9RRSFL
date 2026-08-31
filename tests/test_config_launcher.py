import errno
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import run_experiments_from_config as launcher
import run_fair_tuning_from_config as fair_launcher


class ConfigLauncherTest(unittest.TestCase):
    def _candidate(self, root: Path) -> Path:
        candidate = root / ".venv" / "bin" / "python"
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(b"not a portable Python executable")
        candidate.chmod(0o755)
        return candidate

    def test_unusable_cross_platform_virtualenv_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate(root)
            error = OSError(errno.ENOEXEC, "Exec format error", str(candidate))
            stderr = io.StringIO()
            with (
                mock.patch.object(launcher.os, "execv", side_effect=error) as execv,
                mock.patch.object(sys, "stderr", stderr),
            ):
                reexecuted = launcher._try_project_virtualenv(root)

        self.assertFalse(reexecuted)
        execv.assert_called_once()
        self.assertIn("ignoring unusable project virtualenv Python", stderr.getvalue())
        self.assertIn(str(sys.executable), stderr.getvalue())

    def test_non_executable_virtualenv_falls_back_without_exec(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate(root)
            candidate.chmod(0o644)
            stderr = io.StringIO()
            with (
                mock.patch.object(launcher.os, "execv") as execv,
                mock.patch.object(sys, "stderr", stderr),
            ):
                reexecuted = launcher._try_project_virtualenv(root)

        self.assertFalse(reexecuted)
        execv.assert_not_called()
        self.assertIn("ignoring non-executable", stderr.getvalue())

    def test_unexpected_exec_error_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._candidate(root)
            error = OSError(errno.EIO, "I/O error")
            with mock.patch.object(launcher.os, "execv", side_effect=error):
                with self.assertRaises(OSError) as raised:
                    launcher._try_project_virtualenv(root)

        self.assertEqual(raised.exception.errno, errno.EIO)

    def test_fair_tuning_launcher_reexecutes_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate(root)
            with mock.patch.object(launcher.os, "execv") as execv:
                reexecuted = fair_launcher._try_project_virtualenv(root)

        self.assertTrue(reexecuted)
        execv.assert_called_once_with(
            str(candidate),
            [
                str(candidate),
                str(Path(fair_launcher.__file__).resolve()),
                *sys.argv[1:],
            ],
        )

    def test_fair_tuning_launcher_ignores_cross_platform_virtualenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate(root)
            error = OSError(errno.ENOEXEC, "Exec format error", str(candidate))
            stderr = io.StringIO()
            with (
                mock.patch.object(launcher.os, "execv", side_effect=error),
                mock.patch.object(sys, "stderr", stderr),
            ):
                reexecuted = fair_launcher._try_project_virtualenv(root)

        self.assertFalse(reexecuted)
        self.assertIn("ignoring unusable project virtualenv Python", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
