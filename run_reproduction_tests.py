"""Tests for the fixed reproduction entrypoint and its scientific mechanisms.

Historical search-controller tests retain their original evidence/path contracts;
they are deliberately not presented as a fresh-checkout test suite.
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
MODULES = (
    'reproduction', 'native_sm9_backend', 'sm9_rrs_flow', 'accumulator',
    'aggregation', 'alignins', 'attacks', 'ding13_detector', 'fl', 'model',
    'numerics', 'quarantine_detector', 'svd_detector', 'threshold', 'weighting',
    'vert', 'calibration_policy', 'fair_tuning', 'ours_calibration',
    'performance_target', 'tuning_comparison_failures', 'normal_state_workflow',
    'experiments', 'execution', 'cuda_tuning_resume', 'mnist_partition',
    'crypto_overhead', 'visualization', 'config_runner', 'config_launcher',
    'cifar_tad_no_prior', 'attack_stability_worker', 'attack_stability_attack',
    'attack_stability_probes', 'r4_formal', 'vert_paper_validation',
    'cifar_run_observer',
)

if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.loadTestsFromNames(
        ['tests.test_' + name for name in MODULES])
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
