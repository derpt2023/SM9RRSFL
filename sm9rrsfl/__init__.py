"""Executable SM9-RRS-FL experiment components."""

# Set before NumPy/Torch import in command-line entry points. Respect explicit
# operator overrides. Eight concurrent GPU jobs should not each create eight
# extra BLAS/SGD CPU threads on an eight-core container.
import os

for _thread_option in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_option, "1")

__all__ = [
    "aggregation",
    "alignins",
    "attacks",
    "benchmarks",
    "crypto",
    "datasets",
    "ding13_detector",
    "fl",
    "execution",
    "ours_policy",
    "ours_calibration",
    "fair_tuning",
    "mnist",
    "model",
    "sm9_backend",
    "svd_detector",
    "threshold",
    "vert",
    "weighting",
]
