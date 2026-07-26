import unittest

from sm9rrsfl import sm9_backend


@unittest.skipUnless(sm9_backend.available(), "GmSSL SM9 group bridge is not built")
class NativeSM9BackendTest(unittest.TestCase):
    def test_standard_order_and_generators(self):
        self.assertEqual(sm9_backend.backend_name(), "gmssl-sm9-native-v2")
        self.assertEqual(
            sm9_backend.SM9_ORDER,
            int(
                "B640000002A3A6F1D603AB4FF58EC744"
                "49F2934B18EA8BEEE56EE19CD69ECF25",
                16,
            ),
        )
        p1 = sm9_backend.g1_generator()
        p2 = sm9_backend.g2_generator()
        self.assertEqual(len(p1), 65)
        self.assertEqual(len(p2), 129)
        self.assertTrue(p1.hex().startswith("0493de051d62bf718f"))
        self.assertTrue(p2.hex().startswith("0485aef3d078640c"))
        self.assertTrue(sm9_backend.g1_validate(p1))
        self.assertTrue(sm9_backend.g2_validate(p2))

    def test_standard_h1_expansion_uses_the_direct_transcript(self):
        self.assertEqual(
            sm9_backend.hash_to_scalar(1, b"Alice\x01"),
            int(
                "2ACC468C3926B0BDB2767E99FF26E084"
                "DE9CED8DBC7D5FBF418027B667862FAB",
                16,
            ),
        )
        with self.assertRaisesRegex(ValueError, "prefix must be 1 or 2"):
            sm9_backend.hash_to_scalar(3, b"Alice\x01")
        with self.assertRaisesRegex(ValueError, "prefix must be 1 or 2"):
            sm9_backend._native.hash_to_scalar(3, b"Alice\x01")

    def test_pairing_is_bilinear_in_word_group_direction(self):
        p1 = sm9_backend.g1_generator()
        p2 = sm9_backend.g2_generator()
        base = sm9_backend.pair(p1, p2)
        left = sm9_backend.pair(
            sm9_backend.g1_mul(p1, 2),
            sm9_backend.g2_mul(p2, 3),
        )
        self.assertTrue(sm9_backend.gt_equal(left, sm9_backend.gt_pow(base, 6)))

    def test_noncanonical_and_identity_inputs_are_rejected(self):
        self.assertFalse(sm9_backend.g1_validate(bytes(65)))
        self.assertFalse(sm9_backend.g2_validate(bytes(129)))
        self.assertFalse(sm9_backend.gt_validate(bytes(384)))
        with self.assertRaises(ValueError):
            sm9_backend.g1_mul(sm9_backend.g1_generator(), 0)


if __name__ == "__main__":
    unittest.main()
