import unittest

from sm9rrsfl.accumulator import SM9DynamicAccumulator, ring_digest
from sm9rrsfl import sm9_backend
from sm9rrsfl.crypto import ring_identifier


def sign_public():
    p1 = sm9_backend.g1_generator()
    p2 = sm9_backend.g2_generator()
    master_public = sm9_backend.g2_mul(p2, 7)
    return p1, p2, master_public, sm9_backend.pair(p1, master_public)


def public_accumulator(*, max_size: int, trace_secret: int) -> SM9DynamicAccumulator:
    public = sign_public()
    p1, p2, _, _ = public
    return SM9DynamicAccumulator(
        public,
        trace_public=sm9_backend.g2_mul(p2, trace_secret),
        public_basis=tuple(
            sm9_backend.g1_mul(
                p1,
                pow(trace_secret, exponent, sm9_backend.SM9_ORDER),
            )
            for exponent in range(max_size + 1)
        ),
    )


@unittest.skipUnless(sm9_backend.available(), "GmSSL SM9 group bridge is not built")
class DynamicAccumulatorTest(unittest.TestCase):
    def test_dynamic_add_delete_matches_full_accumulation(self):
        accumulator = public_accumulator(max_size=4, trace_secret=9)

        base_ring = ["client-0", "client-1"]
        full_ring = ["client-0", "client-1", "client-2"]
        base = accumulator.accumulate(base_ring)
        added = accumulator.add(base_ring, "client-2")
        full = accumulator.accumulate(full_ring)
        self.assertEqual(added, full)

        deleted = accumulator.delete(full_ring, "client-2")
        self.assertEqual(deleted, base)
        self.assertFalse(hasattr(accumulator, "_trace_secret"))

    def test_witness_verifies_membership(self):
        accumulator = public_accumulator(max_size=3, trace_secret=10)
        ring = ["client-0", "client-1", "client-2"]

        value = accumulator.accumulate(ring)
        witness = accumulator.witness(ring, "client-1")

        self.assertTrue(accumulator.verify_witness(value, witness, "client-1"))
        self.assertEqual(ring_digest(ring), ring_identifier(ring))


if __name__ == "__main__":
    unittest.main()
