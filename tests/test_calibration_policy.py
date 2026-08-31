import unittest

from sm9rrsfl.calibration_policy import (
    CalibrationHardConstraints,
    OBJECTIVE_WEIGHT_NAMES,
    build_ratio_schedule,
    objective_weight_grid,
    weighted_score,
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

    def test_score_uses_four_benefits_and_no_legacy_false_positive_field(self):
        weights = {
            "clean_accuracy_weight": 0.25,
            "robust_accuracy_weight": 0.25,
            "attack_success_weight": 0.25,
            "honest_weight_loss_weight": 0.25,
        }
        score = weighted_score(
            {
                "clean_accuracy": 0.8,
                "robust_accuracy": 0.6,
                "attack_success_rate": 0.2,
                "honest_weight_loss": 0.1,
            },
            weights,
        )

        self.assertAlmostEqual(score, (0.8 + 0.6 + 0.8 + 0.9) / 4.0)
        self.assertNotIn("false_positive_weight", OBJECTIVE_WEIGHT_NAMES)

    def test_public_hard_constraints_only_cover_execution_integrity(self):
        constraints = CalibrationHardConstraints().to_dict()

        self.assertEqual(
            set(constraints),
            {"min_round_completion_rate", "max_nonfinite_updates"},
        )

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
