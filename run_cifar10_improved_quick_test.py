#!/usr/bin/env python3
"""Quick validation run for CIFAR-10 with improved configuration.

This script tests the hypothesis that attack_boost=15.0 (aligned with MNIST)
combined with MNIST-proven detector parameters will significantly reduce ASR
on CIFAR-10 dataset.

Expected outcome:
- ASR should drop from ~83.5% to <10% at ratio=0.2
- ASR should drop from ~44% to <10% at ratio=0.4
"""

from pathlib import Path
import sys
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sm9rrsfl.experiments import main as run_experiment


def run_quick_validation():
    """Run quick 50-round validation for ratio=0.2 and 0.4."""

    config_path = project_root / "configs" / "cifar10_improved_v9.json"

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

    # Load and display configuration
    with open(config_path) as f:
        config = json.load(f)

    print(f"Configuration: {config_path.name}")
    print(f"Output directory: {config['output_dir']}")
    print()

    # Prepare arguments for experiment runner
    args = [
        "--dataset", "cifar10",
        "--data-dir", "data/cifar10",
        "--download",
        "--methods", "sm9rrs",
        "--partition", "dirichlet",
        "--dirichlet-alpha", "0.5",
        "--num-clients", "100",
        "--rounds", "50",
        "--local-epochs", "1",
        "--batch-size", "50",
        "--lr", "0.05",
        "--lr-decay", "0.99",
        "--attack", "alternating_minimization",
        "--attack-boost", "15.0",
        "--attack-epochs", "1",
        "--attack-scale", "5.0",
        "--attack-source-label", "5",
        "--attack-target-label", "7",
        "--attack-target-count", "200",
        "--attack-start-round", "12",
        "--detector-window", "10",
        "--detector-subspace-dim", "3",
        "--detector-normal-clusters", "2",
        "--detector-distance-threshold", "2.5",
        "--detector-reject-threshold", "5.0",
        "--detector-drift-memory", "0.8",
        "--detector-drift-allowance", "2.0",
        "--detector-drift-threshold", "6.0",
        "--detector-history-confirm", "2",
        "--detector-history-threshold", "1.75",
        "--detector-recovery-confirm", "2",
        "--detector-reference-budget", "3.0",
        "--detector-clip-factor", "2.0",
        "--detector-weight-cap", "2.0",
        "--suspicion-penalty-factor", "0.5",
        "--suspicion-recovery-factor", "1.25",
        "--suspicion-remove-after", "3",
        "--crypto-mode", "simulated",
        "--compute-backend", "torch",
        "--device", "cuda:0",
        "--output-dir", config['output_dir'],
        "--no-early-stop",
        "--eval-interval", "1",
    ]

    # Run experiments for both ratios
    for exp in config['experiments']:
        ratio = exp['malicious_ratio']
        seed = exp['seed']

        print(f"\n{'=' * 80}")
        print(f"Running experiment: {exp['name']}")
        print(f"  Malicious ratio: {ratio}")
        print(f"  Seed: {seed}")
        print(f"{'=' * 80}\n")

        exp_args = args + [
            "--malicious-ratio", str(ratio),
            "--seed", str(seed),
        ]

        try:
            run_experiment(exp_args)
            print(f"\n✓ Experiment {exp['name']} completed successfully\n")
        except Exception as e:
            print(f"\n✗ Experiment {exp['name']} failed: {e}\n")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("Quick validation completed!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Check the output directory for results:")
    print(f"     {config['output_dir']}")
    print("  2. Compare ASR values with baseline (v8):")
    print("     - Baseline ratio=0.2: ASR=83.5%")
    print("     - Baseline ratio=0.4: ASR=44.0%")
    print("     - Target: ASR < 10% for both ratios")
    print("  3. If successful, run full 150-round evaluation with all ratios")
    print()


if __name__ == "__main__":
    run_quick_validation()
