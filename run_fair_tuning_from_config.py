#!/usr/bin/env python3
"""Run the unified fair tuning protocol from a versioned JSON file."""

import os
from pathlib import Path
import sys

from run_experiments_from_config import _try_project_virtualenv as _try_virtualenv

def _try_project_virtualenv(project_root: Path) -> bool:
    """Use the shared safety checks while re-executing this exact launcher."""

    return _try_virtualenv(
        project_root,
        launcher_path=Path(__file__),
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    _try_project_virtualenv(project_root)

    os.chdir(project_root)
    from sm9rrsfl.fair_tuning import main

    # Use enhanced GPU display for better per-device visibility
    try:
        from fair_tuning_gpu_display import gpu_progress_display
        with gpu_progress_display():
            main(sys.argv[1:])
    except ImportError:
        # Fallback to basic display if enhanced version unavailable
        from fair_tuning_progress_display import progress_display
        with progress_display():
            main(sys.argv[1:])
