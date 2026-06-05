import unittest

import numpy as np

from sm9rrsfl.model import INPUT_DIM, NUM_CLASSES, parameter_size
from sm9rrsfl.svd_detector import LongitudinalSVDDetector


class SVDDetectorTest(unittest.TestCase):
    def test_rejects_large_longitudinal_shift(self):
        detector = LongitudinalSVDDetector(window_size=2, z_threshold=2.0)
        base = np.zeros(parameter_size(), dtype=np.float32)
        weight_count = INPUT_DIM * NUM_CLASSES

        for step in range(4):
            update = base.copy()
            update[:weight_count] = (0.01 + step * 0.001)
            result = detector.evaluate("tag-1", update)
            self.assertTrue(result.accepted)

        attack = base.copy()
        attack[:weight_count] = 10.0
        result = detector.evaluate("tag-1", attack)
        self.assertFalse(result.accepted)


if __name__ == "__main__":
    unittest.main()
