import unittest
import hashlib
from dataclasses import replace

import numpy as np

from sm9rrsfl.crypto import SM9RRSContext, sm3_hex_bytes


class SM9RRSFlowTest(unittest.TestCase):
    def test_native_compatible_sm3_matches_standard_vector(self):
        self.assertEqual(
            sm3_hex_bytes(b"abc"),
            "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0",
        )

    def test_simulated_update_digest_uses_fast_sha256(self):
        ctx = SM9RRSContext(["client-0"], crypto_mode="simulated", seed=3)
        update = np.arange(16, dtype=np.float32)

        expected = hashlib.sha256(update.tobytes()).hexdigest()
        self.assertEqual(ctx.digest_update(update), expected)

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

        modified = update.copy()
        modified[0] += 1.0
        self.assertFalse(ctx.verify_packet(packet, modified))

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

        if ctx.rrs_backend == "gmssl-native":
            self.assertIsInstance(packet.signature, bytes)
            self.assertEqual(len(packet.signature), 355)
            self.assertIsInstance(packet.trapdoor, bytes)
            damaged = bytearray(packet.trapdoor)
            damaged[-1] ^= 1
            with self.assertRaises(ValueError):
                ctx.revoke(replace(packet, trapdoor=bytes(damaged)))

    def test_native_context_satisfies_accumulator_and_key_relations(self):
        try:
            from sm9rrsfl import _native_rrs
        except ImportError:
            self.skipTest("optional GmSSL native extension is not built")

        context = _native_rrs.create_context(
            ["client-0", "client-1", "client-2"],
            "auditor",
        )
        self.assertTrue(_native_rrs.validate_context(context))
        signature = _native_rrs.sign(context, "client-1", "ring", "message")
        self.assertTrue(_native_rrs.verify(context, "ring", "message", signature))
        self.assertFalse(_native_rrs.verify(context, "ring", "changed", signature))

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
