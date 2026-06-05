import unittest

from gmssl import sm9
from gmssl import optimized_curve as ec

from sm9rrsfl.accumulator import SM9DynamicAccumulator


class DynamicAccumulatorTest(unittest.TestCase):
    def test_dynamic_add_delete_matches_full_accumulation(self):
        sign_public, _ = sm9.setup("sign")
        accumulator = SM9DynamicAccumulator(sign_public, max_size=4, seed=9)

        base = accumulator.accumulate(["client-0", "client-1"])
        added = accumulator.add(base, "client-2")
        full = accumulator.accumulate(["client-0", "client-1", "client-2"])
        self.assertTrue(ec.eq(added, full))

        deleted = accumulator.delete(added, "client-2")
        self.assertTrue(ec.eq(deleted, base))

    def test_witness_verifies_membership(self):
        sign_public, _ = sm9.setup("sign")
        accumulator = SM9DynamicAccumulator(sign_public, max_size=3, seed=10)
        ring = ["client-0", "client-1", "client-2"]

        value = accumulator.accumulate(ring)
        witness = accumulator.witness(ring, "client-1")

        self.assertTrue(accumulator.verify_witness(value, witness, "client-1"))


if __name__ == "__main__":
    unittest.main()
