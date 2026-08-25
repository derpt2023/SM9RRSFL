import unittest

from sm9rrsfl.calibration_policy import (
    OBJECTIVE_WEIGHT_NAMES,
    build_ratio_schedule,
    objective_weight_grid,
)
from sm9rrsfl.ours_calibration import _automatic_candidate_specs


class CalibrationPolicyTest(unittest.TestCase):
    def test_confirmed_ratio_schedule_uses_disjoint_attacked_midpoints(self):
        schedule = build_ratio_schedule([0, 0.8, 5])

        self.assertEqual(schedule.formal_ratios, (0.0, 0.2, 0.4, 0.6, 0.8))
        self.assertEqual(
            schedule.calibration_ratios,
            (0.0, 0.1, 0.3, 0.5, 0.7),
        )
        self.assertFalse(
            set(schedule.formal_ratios[1:])
            & set(schedule.calibration_ratios[1:])
        )

    def test_learnable_objective_grid_is_positive_normalized_and_bounded(self):
        grid = objective_weight_grid()

        self.assertGreater(len(grid), 1)
        for weights in grid:
            self.assertEqual(set(weights), set(OBJECTIVE_WEIGHT_NAMES))
            self.assertAlmostEqual(sum(weights.values()), 1.0)
            self.assertGreaterEqual(min(weights.values()), 0.05)

    def test_automatic_ours_space_expands_every_requested_axis_but_is_bounded(self):
        candidates = _automatic_candidate_specs(
            detector_window=7,
            num_classes=10,
            max_clients=100,
            budget=12,
        )

        self.assertEqual(len(candidates), 12)
        self.assertEqual({item["q"] for item in candidates}, {1, 2, 3})
        self.assertEqual({item["C_tol"] for item in candidates}, {1, 2, 3, 5})
        self.assertEqual(len({item["beta"] for item in candidates}), 3)
        self.assertEqual(
            len(
                {
                    (item["penalty_factor"], item["recovery_factor"])
                    for item in candidates
                }
            ),
            3,
        )

        complete = _automatic_candidate_specs(
            detector_window=7,
            num_classes=10,
            max_clients=100,
            budget=10_000,
        )
        self.assertEqual(len(complete), 108)


if __name__ == "__main__":
    unittest.main()
