#!/usr/bin/env python3
"""Quick validation run for CIFAR-10 with improved configuration.

This script uses the correct config_runner approach with schema_version: 3.

Expected outcome:
- ASR should drop from ~83.5% to <10% at ratio=0.2
- ASR should drop from ~44% to <10% at ratio=0.4
"""

from pathlib import Path
import subprocess
import sys

project_root = Path(__file__).parent


def run_quick_validation():
    """Run quick 50-round validation for ratio=0.2 and 0.4."""

    print("=" * 80)
    print("CIFAR-10 Improved Configuration - Quick Validation")
    print("=" * 80)
    print()
    print("Key changes from cifar10_resnet18_gn_tpe_v8:")
    print("  1. attack_boost: 5.0 → 15.0 (aligned with MNIST)")
    print("  2. detector_subspace_dim: 1 → 3 (more information retained)")
    print("  3. detector_distance_threshold: 4.39 → 2.5 (MNIST configuration)")
    print("  4. detector_reject_threshold: 5.56 → 5.0 (MNIST configuration)")
    print("  5. rounds: 150 → 50 (quick validation)")
    print("  6. crypto_mode: sm9 → simulated (faster execution)")
    print()
    print("Hypothesis:")
    print("  Higher attack strength makes malicious updates more detectable,")
    print("  leading to significantly lower ASR while maintaining accuracy.")
    print()
    print("=" * 80)
    print()

    experiments = [
        {
            "name": "ratio=0.2",
            "config": "configs/cifar10_improved_v9_ratio02.json",
        },
        {
            "name": "ratio=0.4",
            "config": "configs/cifar10_improved_v9_ratio04.json",
        },
    ]

    for exp in experiments:
        print(f"\n{'=' * 80}")
        print(f"Running experiment: {exp['name']}")
        print(f"Config: {exp['config']}")
        print(f"{'=' * 80}\n")

        try:
            result = subprocess.run(
                [sys.executable, "run_experiments_from_config.py", "--config", exp["config"]],
                cwd=project_root,
                check=True,
            )
            print(f"\n✓ Experiment {exp['name']} completed successfully\n")

        except subprocess.CalledProcessError as e:
            print(f"\n✗ Experiment {exp['name']} failed with exit code {e.returncode}\n")
            print("Check the error messages above for details.")
            continue

    print("\n" + "=" * 80)
    print("Quick validation completed!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Check the output directory for results:")
    print("     outputs/cifar10_improved_v9_quick_validation/")
    print("  2. Compare ASR values with baseline (v8):")
    print("     - Baseline ratio=0.2: ASR=83.5%")
    print("     - Baseline ratio=0.4: ASR=44.0%")
    print("     - Target: ASR < 10% for both ratios")
    print("  3. If successful, run full 150-round evaluation with all ratios")
    print()


if __name__ == "__main__":
    run_quick_validation()
