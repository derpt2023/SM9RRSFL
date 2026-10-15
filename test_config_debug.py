#!/usr/bin/env python3
"""Debug configuration loading."""

import sys
import traceback
from sm9rrsfl.fair_tuning import load_tuning_config

try:
    config = load_tuning_config('configs/cifar10_v13_100clients.json')
    print("✓ Config loaded successfully")
    print(f"  partitions: {config.shared.partitions}")
    print(f"  checkpoint_interval: {config.shared.checkpoint_interval}")
    print(f"  num_clients: {config.shared.num_clients}")
    print(f"  ratios: {config.shared.ratios}")
    print(f"  Total candidates: {sum(len(c.candidates) for c in config.methods)}")
except Exception as e:
    print(f"✗ Error loading config: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
