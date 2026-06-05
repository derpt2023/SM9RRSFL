import unittest

import numpy as np

from sm9rrsfl.crypto import SM9RRSContext


class SM9RRSFlowTest(unittest.TestCase):
    def test_simulated_packet_verify_and_revoke(self):
        ctx = SM9RRSContext(
            ["client-0", "client-1", "client-2"],
            crypto_mode="simulated",
            ring_size=3,
            seed=3,
        )
        update = np.ones(20, dtype=np.float32)
        packet = ctx.create_packet("client-1", update, round_id=1)
        self.assertEqual(packet.ring, tuple())
        self.assertEqual(packet.ring_size, 3)
        self.assertTrue(ctx.verify_packet(packet, update))
        self.assertEqual(ctx.revoke(packet), "client-1")

    def test_sm9_packet_verify_and_revoke(self):
        ctx = SM9RRSContext(
            ["client-0", "client-1", "client-2"],
            crypto_mode="sm9",
            ring_size=3,
            seed=3,
        )
        update = np.ones(20, dtype=np.float32)
        packet = ctx.create_packet("client-1", update, round_id=1)
        self.assertEqual(packet.ring, tuple())
        self.assertEqual(packet.ring_size, 3)
        self.assertEqual(packet._signer_identity_hint, "")
        self.assertTrue(ctx.verify_packet(packet, update))
        self.assertEqual(ctx.revoke(packet), "client-1")

    def test_dynamic_packet_does_not_carry_linear_ring_list(self):
        update = np.ones(20, dtype=np.float32)
        small = SM9RRSContext(
            [f"client-{idx}" for idx in range(5)],
            crypto_mode="simulated",
            accumulator_mode="dynamic",
            seed=5,
        ).create_packet("client-1", update, round_id=1)
        large = SM9RRSContext(
            [f"client-{idx}" for idx in range(50)],
            crypto_mode="simulated",
            accumulator_mode="dynamic",
            seed=5,
        ).create_packet("client-1", update, round_id=1)

        self.assertEqual(small.ring, tuple())
        self.assertEqual(large.ring, tuple())
        self.assertEqual(len(small.signature), len(large.signature))
        self.assertEqual(len(small.ring_accumulator), len(large.ring_accumulator))
        self.assertEqual(small.ring_size, 5)
        self.assertEqual(large.ring_size, 50)

    def test_legacy_mode_still_carries_sampled_ring(self):
        ctx = SM9RRSContext(
            ["client-0", "client-1", "client-2", "client-3"],
            crypto_mode="simulated",
            accumulator_mode="none",
            ring_size=3,
            seed=3,
        )
        update = np.ones(20, dtype=np.float32)
        packet = ctx.create_packet("client-1", update, round_id=1)

        self.assertEqual(len(packet.ring), 3)
        self.assertIn("client-1", packet.ring)
        self.assertTrue(ctx.verify_packet(packet, update))


if __name__ == "__main__":
    unittest.main()
