#!/usr/bin/env python3
"""Start the experiment declared in configs/experiment.json."""

from pathlib import Path
import os
import sys


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    virtualenv_candidates = (
        project_root / ".venv" / "bin" / "python",
        project_root / ".venv" / "Scripts" / "python.exe",
    )
    virtualenv_python = next(
        (candidate for candidate in virtualenv_candidates if candidate.is_file()),
        None,
    )
    if (
        virtualenv_python is not None
        and Path(sys.executable).resolve() != virtualenv_python.resolve()
    ):
        os.execv(
            str(virtualenv_python),
            [str(virtualenv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        )

    os.chdir(project_root)
    from sm9rrsfl.config_runner import main

    main(default_config=project_root / "configs" / "experiment.json")
