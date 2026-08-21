import pickle
import unittest
from unittest.mock import patch

import numpy as np

from sm9rrsfl.svd_detector import LongitudinalSVDDetector, _projector_distance


def _normal_update(round_id: int) -> np.ndarray:
    """A stable top-two subspace with a linear per-round log-spectrum drift."""

    matrix = np.diag(
        [
            np.exp(4.0 - 0.1 * round_id),
            np.exp(3.0 - 0.1 * round_id),
            np.exp(1.0 - 0.02 * round_id),
        ]
    )
    return matrix.astype(np.float32).reshape(-1)


def _attack_update(scale: float = 1.0) -> np.ndarray:
    """A large spectrum shift whose top-two subspace is span(e2, e3)."""

    attack_basis = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    matrix = attack_basis @ np.diag([np.exp(6.0), np.exp(5.0), np.exp(1.0)])
    return (scale * matrix).astype(np.float32).reshape(-1)


def _detector(*, decision_rule: str = "any", window_size: int = 5):
    return LongitudinalSVDDetector(
        window_size=window_size,
        z_threshold=3.0,
        subspace_dim=2,
        gap_threshold=0.1,
        adjacent_threshold=3.0,
        anchor_threshold=3.0,
        drift_memory=1.0,
        drift_allowance=0.1,
        drift_threshold=3.0,
        decision_rule=decision_rule,
        matrix_offset=0,
        matrix_shape=(3, 3),
    )


def _warm_with_clean_updates(detector, *, through_round: int = 6) -> None:
    for round_id in range(1, through_round + 1):
        result = detector.evaluate(
            "tag-1",
            _normal_update(round_id),
            round_id=round_id,
        )
        if not result.accepted:
            raise AssertionError(f"normal round {round_id} was rejected: {result}")


def _state_snapshot(detector):
    state = detector._states["tag-1"]
    trusted = [
        (
            item.round_id,
            item.feature.log_spectrum.copy(),
            item.feature.basis.copy(),
            item.feature.singular_values.copy(),
            item.feature.spectral_gap,
        )
        for item in state.trusted_history
    ]
    distances = {key: tuple(values) for key, values in state.normal_distances.items()}
    gram = None if state.trusted_gram is None else state.trusted_gram.copy()
    return trusted, distances, gram


class SVDDetectorTest(unittest.TestCase):
    def test_default_decision_rule_matches_v3_or_formula(self):
        detector = LongitudinalSVDDetector(
            matrix_offset=0,
            matrix_shape=(3, 3),
        )
        self.assertEqual(detector.decision_rule, "any")

    def assertTrustedSnapshotEqual(self, before, after):
        before_trusted, before_distances, before_gram = before
        after_trusted, after_distances, after_gram = after
        self.assertEqual(len(before_trusted), len(after_trusted))
        for expected, actual in zip(before_trusted, after_trusted):
            self.assertEqual(expected[0], actual[0])
            np.testing.assert_array_equal(expected[1], actual[1])
            np.testing.assert_array_equal(expected[2], actual[2])
            np.testing.assert_array_equal(expected[3], actual[3])
            self.assertEqual(expected[4], actual[4])
        self.assertEqual(before_distances, after_distances)
        if before_gram is None or after_gram is None:
            self.assertIs(before_gram, after_gram)
        else:
            np.testing.assert_array_equal(before_gram, after_gram)

    def test_linear_normal_drift_is_accepted_and_refreshes_trusted_window(self):
        detector = _detector()
        results = []
        for round_id in range(1, 9):
            results.append(
                detector.evaluate(
                    "tag-1",
                    _normal_update(round_id),
                    round_id=round_id,
                )
            )

        self.assertEqual(results[0].reason, "initial_observation")
        self.assertTrue(all(item.accepted for item in results))
        self.assertTrue(all(np.isfinite(item.adjacent_score) for item in results))
        self.assertTrue(all(np.isfinite(item.anchor_score) for item in results))
        self.assertTrue(all(not item.count_increment for item in results))
        self.assertLess(results[-1].cumulative_drift, detector.drift_threshold)

        state = detector._states["tag-1"]
        self.assertEqual(
            [item.round_id for item in state.trusted_history],
            [4, 5, 6, 7, 8],
        )
        self.assertEqual(results[-1].trusted_history_size, detector.window_size)

    def test_first_attack_is_rejected_without_polluting_trusted_history(self):
        detector = _detector(decision_rule="any")
        _warm_with_clean_updates(detector)
        before = _state_snapshot(detector)

        result = detector.evaluate("tag-1", _attack_update(), round_id=7)

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "composite_threshold_any")
        self.assertTrue(result.count_increment)
        self.assertTrue(result.adjacent_exceeded)
        self.assertTrue(result.anchor_exceeded)
        self.assertTrue(result.drift_exceeded)
        self.assertTrustedSnapshotEqual(before, _state_snapshot(detector))
        state = detector._states["tag-1"]
        self.assertEqual(state.last_observed.round_id, 7)
        np.testing.assert_allclose(
            state.last_observed.feature.log_spectrum,
            np.asarray([6.0, 5.0]),
            atol=1e-6,
        )

    def test_any_rule_rejects_similar_attack_to_attack_via_trusted_anchor(self):
        detector = _detector(decision_rule="any")
        _warm_with_clean_updates(detector)
        trusted_before = _state_snapshot(detector)

        first_attack = detector.evaluate("tag-1", _attack_update(), round_id=7)
        second_attack = detector.evaluate(
            "tag-1",
            _attack_update(scale=1.001),
            round_id=8,
        )

        self.assertFalse(first_attack.accepted)
        self.assertFalse(second_attack.accepted)
        self.assertFalse(second_attack.adjacent_exceeded)
        self.assertAlmostEqual(second_attack.adjacent_score, 0.0, places=7)
        self.assertTrue(second_attack.anchor_exceeded)
        self.assertTrue(second_attack.drift_exceeded)
        self.assertGreater(second_attack.anchor_score, detector.anchor_threshold)
        self.assertTrue(second_attack.count_increment)
        self.assertTrustedSnapshotEqual(trusted_before, _state_snapshot(detector))
        self.assertEqual(detector._states["tag-1"].last_observed.round_id, 8)

    def test_and_rule_is_not_supported(self):
        with self.assertRaisesRegex(ValueError, "v3 OR formula"):
            _detector(decision_rule="all")

    def test_trusted_trend_uses_real_round_gaps(self):
        detector = _detector(window_size=3)
        for round_id in (1, 2, 5):
            result = detector.evaluate(
                "tag-1",
                _normal_update(round_id),
                round_id=round_id,
            )
            self.assertTrue(result.accepted)

        predicted, _ = detector._trusted_reference(
            detector._states["tag-1"],
            8,
        )
        np.testing.assert_allclose(predicted, [3.2, 2.2], atol=2e-6)

        result = detector.evaluate("tag-1", _normal_update(8), round_id=8)
        self.assertTrue(result.accepted)
        self.assertLess(result.spectrum_anchor_distance, 2e-6)

    def test_pickle_round_trip_preserves_dual_reference_state(self):
        detector = _detector(decision_rule="any")
        _warm_with_clean_updates(detector)
        detector.evaluate("tag-1", _attack_update(), round_id=7)
        restored = pickle.loads(pickle.dumps(detector))

        expected = detector.evaluate(
            "tag-1",
            _attack_update(scale=1.001),
            round_id=8,
        )
        actual = restored.evaluate(
            "tag-1",
            _attack_update(scale=1.001),
            round_id=8,
        )

        self.assertEqual(expected, actual)
        self.assertTrustedSnapshotEqual(
            _state_snapshot(detector),
            _state_snapshot(restored),
        )
        expected_state = detector._states["tag-1"]
        actual_state = restored._states["tag-1"]
        self.assertEqual(expected_state.last_observed.round_id, actual_state.last_observed.round_id)
        self.assertEqual(expected_state.observed_count, actual_state.observed_count)
        self.assertEqual(expected_state.cumulative_drift, actual_state.cumulative_drift)

    def test_projector_distance_ignores_sign_and_top_two_basis_rotation(self):
        basis = np.eye(3, dtype=np.float64)[:, :2]
        sign_flipped = basis @ np.diag([-1.0, 1.0])
        angle = np.pi / 4.0
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float64,
        )
        rotated = basis @ rotation

        self.assertAlmostEqual(_projector_distance(basis, sign_flipped, 2), 0.0)
        self.assertAlmostEqual(_projector_distance(basis, rotated, 2), 0.0, places=6)

    def test_small_q_q_plus_one_gap_downweights_direction_evidence(self):
        permutation = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )

        def scored_transition(values):
            detector = _detector(window_size=2)
            baseline = np.diag(values).astype(np.float32).reshape(-1)
            changed = (permutation @ np.diag(values)).astype(np.float32).reshape(-1)
            detector.evaluate("tag-1", baseline, round_id=1)
            detector.evaluate("tag-1", baseline, round_id=2)
            return detector.evaluate("tag-1", changed, round_id=3)

        reliable = scored_transition([5.0, 3.0, 1.0])
        near_degenerate = scored_transition([5.0, 3.0, 2.999])

        self.assertAlmostEqual(reliable.direction_reliability, 1.0)
        self.assertLess(near_degenerate.direction_reliability, 0.01)
        self.assertAlmostEqual(
            near_degenerate.z_subspace_adjacent,
            reliable.z_subspace_adjacent,
            places=4,
        )
        self.assertLess(
            near_degenerate.adjacent_score,
            reliable.adjacent_score * 0.01,
        )

    def test_default_matrix_uses_complete_update_and_zero_padding(self):
        detector = LongitudinalSVDDetector(num_classes=3, subspace_dim=2)
        update = np.arange(1, 9, dtype=np.float64)
        fake_values = np.asarray([3.0, 2.0, 1.0], dtype=np.float64)
        fake_basis = np.eye(3, dtype=np.float32)[:, :2]

        with patch(
            "sm9rrsfl.torch_backend.numpy_top_singular_subspace",
            return_value=(fake_values, fake_basis),
        ) as top_feature:
            detector._extract(update)

        matrix = top_feature.call_args.args[0]
        self.assertEqual(top_feature.call_args.kwargs["rank"], 2)
        self.assertEqual(matrix.dtype, np.float32)
        self.assertEqual(matrix.shape, (3, 3))
        np.testing.assert_array_equal(matrix.reshape(-1)[:8], update.astype(np.float32))
        self.assertEqual(float(matrix[-1, -1]), 0.0)

    def test_explicit_matrix_slice_remains_available_for_compatibility(self):
        detector = LongitudinalSVDDetector(
            num_classes=4,
            subspace_dim=2,
            matrix_offset=2,
            matrix_shape=(3, 3),
        )
        update = np.arange(12, dtype=np.float32)
        fake_values = np.asarray([3.0, 2.0, 1.0], dtype=np.float64)
        fake_basis = np.eye(3, dtype=np.float32)[:, :2]

        with patch(
            "sm9rrsfl.torch_backend.numpy_top_singular_subspace",
            return_value=(fake_values, fake_basis),
        ) as top_feature:
            detector._extract(update)

        matrix = top_feature.call_args.args[0]
        np.testing.assert_array_equal(matrix, update[2:11].reshape(3, 3))

    def test_default_matrix_requires_finite_nonempty_one_dimensional_update(self):
        detector = LongitudinalSVDDetector()
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            detector.evaluate("tag-1", np.zeros((2, 2), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            detector.evaluate("tag-1", np.asarray([], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            detector.evaluate("tag-1", np.full(20, np.nan, dtype=np.float32))

    def test_registered_model_size_is_enforced(self):
        detector = LongitudinalSVDDetector(expected_update_size=20)
        with self.assertRaisesRegex(ValueError, "registered model"):
            detector.evaluate("tag-1", np.zeros(30, dtype=np.float32))

    def test_round_id_must_increase_for_each_tag(self):
        detector = _detector()
        detector.evaluate("tag-1", _normal_update(1), round_id=1)
        with self.assertRaisesRegex(ValueError, "must increase"):
            detector.evaluate("tag-1", _normal_update(1), round_id=1)

    def test_forget_releases_revoked_tag_state(self):
        detector = _detector()
        detector.evaluate("tag-1", _normal_update(1), round_id=1)

        self.assertGreater(detector.estimated_state_bytes(), 0)
        self.assertTrue(detector.forget("tag-1"))
        self.assertEqual(detector.estimated_state_bytes(), 0)
        self.assertNotIn("tag-1", detector._states)
        self.assertFalse(detector.forget("tag-1"))

    def test_detector_rejects_degenerate_configuration(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            LongitudinalSVDDetector(window_size=1)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            LongitudinalSVDDetector(z_threshold=float("nan"))
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            LongitudinalSVDDetector(eps=0.0)
        with self.assertRaisesRegex(ValueError, "v3 OR formula"):
            LongitudinalSVDDetector(decision_rule="unknown")
        with self.assertRaisesRegex(ValueError, r"q\+1"):
            LongitudinalSVDDetector(subspace_dim=3, matrix_shape=(2, 3))
        with self.assertRaisesRegex(ValueError, r"q\+1"):
            LongitudinalSVDDetector(subspace_dim=3, num_classes=3)
        LongitudinalSVDDetector(
            subspace_dim=3,
            num_classes=3,
            matrix_shape=(4, 4),
        )


if __name__ == "__main__":
    unittest.main()
