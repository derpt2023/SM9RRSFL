import pickle
import unittest
from unittest import mock

import numpy as np

from sm9rrsfl.fedredefense import FedREDefense
from sm9rrsfl.model import DEFAULT_SPEC, init_params


class FedREDefenseTest(unittest.TestCase):
    def _defense(self, **overrides):
        kwargs = {
            "model_spec": DEFAULT_SPEC,
            "initial_iterations": 1,
            "max_iterations": 1,
            "synthetic_steps": 1,
            "device": "cpu",
            "seed": 9,
        }
        kwargs.update(overrides)
        return FedREDefense(["client-0", "client-1"], **kwargs)

    def test_filters_updates_above_reconstruction_threshold(self):
        defense = self._defense()
        with mock.patch.object(
            defense,
            "_reconstruction_error",
            side_effect=lambda client_id, *_args, **_kwargs: (
                0.2 if client_id == "client-0" else 0.9
            ),
        ):
            result = defense.evaluate_round(
                np.zeros(DEFAULT_SPEC.parameter_size, dtype=np.float32),
                {
                    "client-0": np.ones(
                        DEFAULT_SPEC.parameter_size,
                        dtype=np.float32,
                    ),
                    "client-1": np.ones(
                        DEFAULT_SPEC.parameter_size,
                        dtype=np.float32,
                    ),
                },
                round_id=1,
            )
        self.assertEqual(result.accepted_clients, ("client-0",))
        self.assertEqual(result.rejected_clients, ("client-1",))

    def test_differentiable_reconstruction_path_and_checkpoint_pickle(self):
        defense = self._defense(threshold=100.0)
        params = init_params(seed=3, spec=DEFAULT_SPEC)
        update = np.full_like(params, 1e-4)
        result = defense.evaluate_round(
            params,
            {"client-0": update},
            round_id=1,
        )
        self.assertEqual(result.accepted_clients, ("client-0",))
        self.assertTrue(np.isfinite(result.reconstruction_errors["client-0"]))

        restored = pickle.loads(pickle.dumps(defense))
        self.assertIn("client-0", restored._states)
        self.assertEqual(str(restored.device), "cpu")


if __name__ == "__main__":
    unittest.main()
