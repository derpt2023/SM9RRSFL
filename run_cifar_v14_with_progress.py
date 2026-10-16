#!/usr/bin/env python3
"""Launch the independent CIFAR-10 v14 grid with a round/GPU dashboard."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def main(argv=None):
    from sm9rrsfl import fair_tuning
    from sm9rrsfl.config_runner import ConfigError
    from scripts.experiments.cifar10.fair_tuning_dashboard import dashboard, plan
    from scripts.experiments.cifar10.cifar_fair_gpu_pool import check_gpus, gpu_pool

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/cifar10_v14_fair_grid.json"))
    parser.add_argument("--dry-run", action="store_true", help="Validate and count work without training or downloading data")
    parser.add_argument("--display-mode", choices=("auto", "live", "log"), default="auto")
    parser.add_argument("--check-gpus", action="store_true", help="Inspect visible CUDA GPUs without loading data or training")
    options = parser.parse_args(argv)
    try:
        spec = fair_tuning.load_fair_tuning_config(options.config)
        _, _, work = plan(spec)
    except (ConfigError, fair_tuning.FairTuningError) as exc:
        parser.error(str(exc))
    # Reuse the tuner's validation summary and retain all original health/selection gates.
    fair_tuning.main(["--config", str(spec.source), "--dry-run"])
    work["total_rounds"] = work["validation_rounds"] + work["final_rounds"]
    print("planned_work=" + json.dumps(work, sort_keys=True), flush=True)
    print("experiment_output=" + str((PROJECT_ROOT / spec.shared_parameters["output_dir"]).resolve()), flush=True)
    if options.check_gpus:
        try:
            check_gpus()
        except RuntimeError as exc:
            parser.error(str(exc))
        return
    if options.dry_run:
        return
    try:
        check_gpus()
        with dashboard(spec, mode=options.display_mode) as progress, gpu_pool(on_plan=progress.gpu_plan):
            selected = fair_tuning.run_fair_tuning(spec)
            print("selected_parameters=" + json.dumps({method: row.parameters for method, row in selected.items()}, sort_keys=True), flush=True)
    except (ConfigError, fair_tuning.FairTuningError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    # Respect the caller's activated Python environment on AI Station.
    os.chdir(PROJECT_ROOT)
    main()
