import unittest
from unittest.mock import patch

import numpy as np

from sm9rrsfl.svd_detector import LongitudinalSVDDetector


def _rank_one_update(scale: float) -> np.ndarray:
    return np.asarray([scale, 0.0, 0.0, 0.0], dtype=np.float32)


class SVDDetectorTest(unittest.TestCase):
    def test_k_three_scores_round_four_and_keeps_only_passed_history(self):
        detector = LongitudinalSVDDetector(
            window_size=3,
            z_threshold=3.0,
            matrix_offset=0,
            matrix_shape=(2, 2),
        )

        first = detector.evaluate("tag-1", _rank_one_update(1.0))
        second = detector.evaluate("tag-1", _rank_one_update(2.0))
        third = detector.evaluate("tag-1", _rank_one_update(3.0))
        self.assertEqual(first.reason, "initial_observation")
        self.assertEqual(second.reason, "baseline_warmup")
        self.assertEqual(third.reason, "baseline_warmup")

        state = detector._states["tag-1"]
        normal_history = tuple(state.normal_history)
        self.assertEqual(len(normal_history), 2)

        # Word 4.3.3 says r > K, so K=3 makes the fourth observation the
        # first one scored rather than one more unconditionally accepted round.
        fourth = detector.evaluate("tag-1", _rank_one_update(12.0))
        self.assertFalse(fourth.accepted)
        self.assertEqual(fourth.reason, "z_score_threshold")
        self.assertFalse(fourth.count_increment)
        self.assertEqual(tuple(state.normal_history), normal_history)

        # The rejected feature is still the immediately preceding observation.
        # Comparing round five with round three would produce another large
        # jump; comparing it with round four produces the normal delta 1.
        fifth = detector.evaluate("tag-1", _rank_one_update(13.0))
        self.assertTrue(fifth.accepted)
        self.assertEqual(fifth.reason, "accepted")
        self.assertAlmostEqual(fifth.sigma_delta, 1.0, places=6)

    def test_count_advances_only_when_both_z_scores_exceed_threshold(self):
        detector = LongitudinalSVDDetector(
            window_size=2,
            z_threshold=3.0,
            matrix_offset=0,
            matrix_shape=(2, 2),
        )

        detector.evaluate("tag-both", _rank_one_update(1.0))
        detector.evaluate("tag-both", _rank_one_update(2.0))
        both = detector.evaluate(
            "tag-both",
            np.asarray([0.0, 0.0, 12.0, 0.0], dtype=np.float32),
        )
        self.assertFalse(both.accepted)
        self.assertGreater(both.z_sigma, detector.z_threshold)
        self.assertGreater(both.z_direction, detector.z_threshold)
        self.assertTrue(both.count_increment)

        detector.evaluate("tag-direction-only", _rank_one_update(1.0))
        detector.evaluate("tag-direction-only", _rank_one_update(2.0))
        direction_only = detector.evaluate(
            "tag-direction-only",
            np.asarray([0.0, 0.0, 3.0, 0.0], dtype=np.float32),
        )
        self.assertFalse(direction_only.accepted)
        self.assertLessEqual(direction_only.z_sigma, detector.z_threshold)
        self.assertGreater(direction_only.z_direction, detector.z_threshold)
        self.assertFalse(direction_only.count_increment)

    def test_default_matrix_uses_the_complete_update_and_zero_padding(self):
        detector = LongitudinalSVDDetector(num_classes=3)
        update = np.arange(1, 9, dtype=np.float64)

        with patch(
            "sm9rrsfl.torch_backend.numpy_top_singular_feature",
            return_value=(1.0, np.ones(3, dtype=np.float32)),
        ) as top_feature:
            detector._extract(update)

        matrix = top_feature.call_args.args[0]
        self.assertEqual(matrix.dtype, np.float32)
        self.assertEqual(matrix.shape, (3, 3))
        np.testing.assert_array_equal(
            matrix.reshape(-1)[:8],
            update.astype(np.float32),
        )
        self.assertEqual(float(matrix[-1, -1]), 0.0)

    def test_explicit_matrix_slice_remains_available_for_compatibility(self):
        detector = LongitudinalSVDDetector(
            num_classes=4,
            matrix_offset=2,
            matrix_shape=(2, 3),
        )
        update = np.arange(12, dtype=np.float32)

        with patch(
            "sm9rrsfl.torch_backend.numpy_top_singular_feature",
            return_value=(1.0, np.ones(2, dtype=np.float32)),
        ) as top_feature:
            detector._extract(update)

        matrix = top_feature.call_args.args[0]
        np.testing.assert_array_equal(matrix, update[2:8].reshape(2, 3))

    def test_default_matrix_requires_a_nonempty_one_dimensional_update(self):
        detector = LongitudinalSVDDetector()
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            detector.evaluate("tag-1", np.zeros((2, 2), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            detector.evaluate("tag-1", np.asarray([], dtype=np.float32))

    def test_registered_model_size_is_enforced(self):
        detector = LongitudinalSVDDetector(expected_update_size=4)
        with self.assertRaisesRegex(ValueError, "registered model"):
            detector.evaluate("tag-1", np.zeros(5, dtype=np.float32))

    def test_svd_direction_sign_is_canonicalized_before_cosine(self):
        detector = LongitudinalSVDDetector(
            window_size=2,
            matrix_offset=0,
            matrix_shape=(2, 2),
        )
        directions = iter(
            (
                (1.0, np.asarray([-1.0, 0.0], dtype=np.float32)),
                (2.0, np.asarray([1.0, 0.0], dtype=np.float32)),
            )
        )
        with patch(
            "sm9rrsfl.torch_backend.numpy_top_singular_feature",
            side_effect=lambda _matrix: next(directions),
        ):
            detector.evaluate("tag-1", np.ones(4, dtype=np.float32))
            result = detector.evaluate("tag-1", np.ones(4, dtype=np.float32))
        self.assertAlmostEqual(result.cosine_similarity, 1.0, places=6)

    def test_detector_rejects_degenerate_configuration(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            LongitudinalSVDDetector(window_size=1)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            LongitudinalSVDDetector(z_threshold=float("nan"))
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            LongitudinalSVDDetector(eps=0.0)


if __name__ == "__main__":
    unittest.main()
