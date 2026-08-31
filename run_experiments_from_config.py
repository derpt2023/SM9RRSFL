#!/usr/bin/env python3
"""Start the experiment declared in configs/experiment.json."""

from __future__ import annotations

from pathlib import Path
import errno
import os
import sys


_RECOVERABLE_EXEC_ERRNOS = {errno.EACCES, errno.ENOENT, errno.ENOEXEC}


def _try_project_virtualenv(
    project_root: Path,
    *,
    launcher_path: Path | None = None,
) -> bool:
    """Re-exec a usable project venv, or keep the current interpreter.

    Virtual environments are not portable across operating systems.  In
    particular, copying a macOS ``.venv`` into a Linux AI Station must not
    prevent the launcher from using the container's current Python.
    """

    virtualenv_candidates = (
        project_root / ".venv" / "bin" / "python",
        project_root / ".venv" / "Scripts" / "python.exe",
    )
    virtualenv_python = next(
        (candidate for candidate in virtualenv_candidates if candidate.is_file()),
        None,
    )
    if virtualenv_python is None:
        return False
    if Path(sys.executable).resolve() == virtualenv_python.resolve():
        return False
    if not os.access(virtualenv_python, os.X_OK):
        print(
            f"warning: ignoring non-executable project virtualenv Python: "
            f"{virtualenv_python}; using {sys.executable}",
            file=sys.stderr,
        )
        return False

    target_launcher = (
        Path(__file__).resolve()
        if launcher_path is None
        else launcher_path.resolve()
    )
    try:
        os.execv(
            str(virtualenv_python),
            [str(virtualenv_python), str(target_launcher), *sys.argv[1:]],
        )
    except OSError as exc:
        if exc.errno not in _RECOVERABLE_EXEC_ERRNOS:
            raise
        print(
            f"warning: ignoring unusable project virtualenv Python "
            f"{virtualenv_python} ({exc.strerror}); using {sys.executable}",
            file=sys.stderr,
        )
        return False
    return True


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    _try_project_virtualenv(project_root)

    os.chdir(project_root)
    from sm9rrsfl.config_runner import main

    main(default_config=project_root / "configs" / "experiment.json")
