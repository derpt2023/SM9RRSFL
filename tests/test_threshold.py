from dataclasses import replace
import unittest

from sm9rrsfl.threshold import (
    SCALAR_MODULUS,
    DistributedKGC,
    TraceApproval,
    TraceAuthorization,
    TraceAuthorizationRequest,
    ThresholdCertificate,
    ThresholdNotMetError,
)


class ThresholdCertificateTest(unittest.TestCase):
    @staticmethod
    def _encode_fields(*fields: bytes) -> bytes:
        encoded = bytearray()
        for field in fields:
            encoded.extend(len(field).to_bytes(8, "big"))
            encoded.extend(field)
        return bytes(encoded)

    def _trace_request(self, *, session_byte: int = 3) -> TraceAuthorizationRequest:
        return TraceAuthorizationRequest(
            task_id="task-a",
            rid="01" * 32,
            evidence_digest="02" * 32,
            session_id=f"{session_byte:02x}" * 32,
        )

    def test_trace_requires_distinct_message_bound_node_approvals(self):
        dkg = DistributedKGC(
            threshold=2,
            node_count=3,
            crypto_mode="simulated",
            max_accumulator_size=3,
            seed=20,
        )
        request = self._trace_request()
        dkg.build_task_ring(request.rid, {"a": 5, "b": 7})
        task_point = 11
        first = dkg.trace_approval_issuer("KGC-1").approve(request)
        third = dkg.trace_approval_issuer("KGC-3").approve(request)
        self.assertTrue(dkg.verify_trace_approval(request, first))
        self.assertTrue(dkg.verify_trace_approval(request, third))

        with self.assertRaises(ThresholdNotMetError):
            dkg.begin_trace(
                TraceAuthorization(request=request, approvals=(first, first)),
                expected_task_id=request.task_id,
                expected_rid=request.rid,
                expected_evidence_digest=request.evidence_digest,
                expected_task_point=task_point,
                expected_signature_a=1,
                expected_signature_b=2,
                expected_task_tag=3,
            )

        # Renaming an approval to a second public node does not create a quorum:
        # that node's Feldman-derived public share verifies a different key.
        renamed = replace(first, node_identity="KGC-2")
        self.assertFalse(dkg.verify_trace_approval(request, renamed))
        with self.assertRaises(ThresholdNotMetError):
            dkg.begin_trace(
                TraceAuthorization(request=request, approvals=(first, renamed)),
                expected_task_id=request.task_id,
                expected_rid=request.rid,
                expected_evidence_digest=request.evidence_digest,
                expected_task_point=task_point,
                expected_signature_a=1,
                expected_signature_b=2,
                expected_task_tag=3,
            )

        tampered_request = replace(request, evidence_digest="04" * 32)
        self.assertFalse(dkg.verify_trace_approval(tampered_request, first))
        with self.assertRaises(ThresholdNotMetError):
            dkg.begin_trace(
                TraceAuthorization(
                    request=tampered_request,
                    approvals=(first, third),
                ),
                expected_task_id=tampered_request.task_id,
                expected_rid=tampered_request.rid,
                expected_evidence_digest=tampered_request.evidence_digest,
                expected_task_point=task_point,
                expected_signature_a=1,
                expected_signature_b=2,
                expected_task_tag=3,
            )

        session = dkg.begin_trace(
            TraceAuthorization(request=request, approvals=(third, first)),
            expected_task_id=request.task_id,
            expected_rid=request.rid,
            expected_evidence_digest=request.evidence_digest,
            expected_task_point=task_point,
            expected_signature_a=1,
            expected_signature_b=2,
            expected_task_tag=3,
        )
        dkg.end_trace(session)
        with self.assertRaisesRegex(ValueError, "not active"):
            dkg.finalize_trace_certificate(13, session)

        # A valid signed request is one-shot; replaying its session is rejected.
        with self.assertRaisesRegex(ValueError, "already been consumed"):
            dkg.begin_trace(
                TraceAuthorization(request=request, approvals=(first, third)),
                expected_task_id=request.task_id,
                expected_rid=request.rid,
                expected_evidence_digest=request.evidence_digest,
                expected_task_point=task_point,
                expected_signature_a=1,
                expected_signature_b=2,
                expected_task_tag=3,
            )

    def test_authorized_trace_is_a_bound_ordered_state_machine(self):
        dkg = DistributedKGC(
            threshold=2,
            node_count=3,
            crypto_mode="simulated",
            max_accumulator_size=3,
            seed=25,
        )
        request = self._trace_request(session_byte=6)
        identity_scalars = {"a": 5, "b": 7}
        _accumulator, witnesses = dkg.build_task_ring(
            request.rid,
            identity_scalars,
        )
        private_key = dkg.extract_signing_key(identity_scalars["b"])
        response_scalar = 13
        signature_a = witnesses["b"] * response_scalar % SCALAR_MODULUS
        signature_b = private_key * response_scalar % SCALAR_MODULUS
        task_point = 11
        expected_task_tag = private_key * task_point % SCALAR_MODULUS
        approvals = tuple(
            dkg.trace_approval_issuer(identity).approve(request)
            for identity in ("KGC-1", "KGC-3")
        )
        session = dkg.begin_trace(
            TraceAuthorization(request=request, approvals=approvals),
            expected_task_id=request.task_id,
            expected_rid=request.rid,
            expected_evidence_digest=request.evidence_digest,
            expected_task_point=task_point,
            expected_signature_a=signature_a,
            expected_signature_b=signature_b,
            expected_task_tag=expected_task_tag,
        )

        with self.assertRaisesRegex(ValueError, "trace_match"):
            dkg.reconstruct_candidate_tag(
                identity_scalars["b"],
                task_point,
                session,
            )
        with self.assertRaisesRegex(ValueError, "match and tag reconstruction"):
            dkg.finalize_trace_certificate(1, session)
        with self.assertRaisesRegex(ValueError, "authorized evidence"):
            dkg.trace_match(
                rid=request.rid,
                identities=tuple(identity_scalars),
                identity_scalars=identity_scalars,
                signature_a=(signature_a + 1) % SCALAR_MODULUS,
                signature_b=signature_b,
                session=session,
            )
        with self.assertRaisesRegex(ValueError, "registered ring material"):
            dkg.trace_match(
                rid=request.rid,
                identities=tuple(identity_scalars),
                identity_scalars={"a": 5, "b": 8},
                signature_a=signature_a,
                signature_b=signature_b,
                session=session,
            )

        matched = dkg.trace_match(
            rid=request.rid,
            identities=tuple(identity_scalars),
            identity_scalars=identity_scalars,
            signature_a=signature_a,
            signature_b=signature_b,
            session=session,
        )
        self.assertEqual(matched, "b")
        with self.assertRaisesRegex(ValueError, "first trace operation"):
            dkg.trace_match(
                rid=request.rid,
                identities=tuple(identity_scalars),
                identity_scalars=identity_scalars,
                signature_a=signature_a,
                signature_b=signature_b,
                session=session,
            )
        with self.assertRaisesRegex(ValueError, "different ring identity"):
            dkg.reconstruct_candidate_tag(
                identity_scalars["a"],
                task_point,
                session,
            )
        with self.assertRaisesRegex(ValueError, "different task point"):
            dkg.reconstruct_candidate_tag(
                identity_scalars["b"],
                task_point + 1,
                session,
            )

        candidate_tag = dkg.reconstruct_candidate_tag(
            identity_scalars["b"],
            task_point,
            session,
        )
        with self.assertRaisesRegex(ValueError, "successful trace_match"):
            dkg.reconstruct_candidate_tag(
                identity_scalars["b"],
                task_point,
                session,
            )
        with self.assertRaisesRegex(ValueError, "differs from Equation"):
            dkg.finalize_trace_certificate(
                (candidate_tag + 1) % SCALAR_MODULUS,
                session,
            )
        with self.assertRaisesRegex(ValueError, "not session-bound"):
            dkg.threshold_sign_authorized(
                b"arbitrary signing-oracle request",
                session,
                task_tag=candidate_tag,
            )

        certificate = dkg.finalize_trace_certificate(candidate_tag, session)
        expected_message = self._encode_fields(
            bytes.fromhex(request.evidence_digest),
            b"b",
            candidate_tag.to_bytes(32, "big"),
            bytes.fromhex(request.rid),
            b"trace",
        )
        self.assertTrue(dkg.threshold_verify(expected_message, certificate))
        self.assertFalse(dkg.threshold_verify(b"another message", certificate))
        with self.assertRaisesRegex(ValueError, "match and tag reconstruction"):
            dkg.finalize_trace_certificate(candidate_tag, session)
        dkg.end_trace(session)

        wrong_tag_request = replace(request, session_id="07" * 32)
        wrong_tag_approvals = tuple(
            dkg.trace_approval_issuer(identity).approve(wrong_tag_request)
            for identity in ("KGC-1", "KGC-3")
        )
        wrong_tag_session = dkg.begin_trace(
            TraceAuthorization(
                request=wrong_tag_request,
                approvals=wrong_tag_approvals,
            ),
            expected_task_id=wrong_tag_request.task_id,
            expected_rid=wrong_tag_request.rid,
            expected_evidence_digest=wrong_tag_request.evidence_digest,
            expected_task_point=task_point,
            expected_signature_a=signature_a,
            expected_signature_b=signature_b,
            expected_task_tag=(expected_task_tag + 1) % SCALAR_MODULUS,
        )
        dkg.trace_match(
            rid=request.rid,
            identities=tuple(identity_scalars),
            identity_scalars=identity_scalars,
            signature_a=signature_a,
            signature_b=signature_b,
            session=wrong_tag_session,
        )
        with self.assertRaisesRegex(ValueError, "authorized evidence"):
            dkg.reconstruct_candidate_tag(
                identity_scalars["b"],
                task_point,
                wrong_tag_session,
            )
        dkg.end_trace(wrong_tag_session)

    def test_integer_node_names_are_not_trace_authorization(self):
        dkg = DistributedKGC(
            threshold=2,
            node_count=3,
            crypto_mode="simulated",
            max_accumulator_size=3,
            seed=19,
        )
        with self.assertRaises(TypeError):
            dkg.trace_approval_issuer(1)  # type: ignore[arg-type]
        request = self._trace_request(session_byte=5)
        dkg.build_task_ring(request.rid, {"a": 5})
        with self.assertRaises(ThresholdNotMetError):
            dkg.begin_trace(
                TraceAuthorization(
                    request=request,
                    approvals=(
                        TraceApproval(
                            node_identity="KGC-1",
                            request_digest="00" * 32,
                            commitment=1,
                            response=1,
                        ),
                        TraceApproval(
                            node_identity="KGC-2",
                            request_digest="00" * 32,
                            commitment=1,
                            response=1,
                        ),
                    ),
                ),
                expected_task_id=request.task_id,
                expected_rid=request.rid,
                expected_evidence_digest=request.evidence_digest,
                expected_task_point=11,
                expected_signature_a=1,
                expected_signature_b=2,
                expected_task_tag=3,
            )

    def test_certificate_is_message_bound_and_requires_a_quorum(self):
        dkg = DistributedKGC(
            threshold=2,
            node_count=3,
            crypto_mode="simulated",
            max_accumulator_size=3,
            seed=21,
        )
        with self.assertRaises(ThresholdNotMetError):
            dkg.threshold_sign(b"trace-message", (1,))

        certificate = dkg.threshold_sign(b"trace-message", (1, 3))
        self.assertIsInstance(certificate, ThresholdCertificate)
        self.assertTrue(dkg.threshold_verify(b"trace-message", certificate))
        self.assertFalse(dkg.threshold_verify(b"different-message", certificate))
        self.assertFalse(
            dkg.threshold_verify(
                b"trace-message",
                replace(certificate, response=certificate.response + 1),
            )
        )
        self.assertFalse(
            dkg.threshold_verify(
                b"trace-message",
                replace(
                    certificate,
                    commitment=(certificate.commitment + 1) % SCALAR_MODULUS,
                ),
            )
        )

        # A linear one-group-element certificate could be forged by scaling.
        # Schnorr's challenge includes R, so scaling both (R,z) must fail.
        scaled = replace(
            certificate,
            commitment=certificate.commitment * 2 % SCALAR_MODULUS,
            response=certificate.response * 2 % SCALAR_MODULUS,
        )
        self.assertFalse(dkg.threshold_verify(b"trace-message", scaled))

    def test_share_domain_mpc_keeps_secret_values_shared(self):
        dkg = DistributedKGC(
            threshold=2,
            node_count=3,
            crypto_mode="simulated",
            max_accumulator_size=3,
            seed=23,
        )
        attributes = vars(dkg)
        self.assertNotIn("_master_secret", attributes)
        self.assertNotIn("_trace_secret", attributes)
        self.assertNotIn("_certificate_secret", attributes)
        self.assertFalse(hasattr(dkg, "_ideal_mpc_reconstruct"))
        self.assertEqual(len(dkg.export_state().master_shares), 3)
        self.assertEqual(len(dkg.export_state().trace_shares), 3)
        self.assertEqual(len(dkg.export_state().certificate_shares), 3)
        self.assertEqual(dkg.export_state().node_identities, ("KGC-1", "KGC-2", "KGC-3"))
        self.assertNotEqual(dkg.export_state().node_coordinates, (1, 2, 3))
        self.assertTrue(dkg.validate_public_relations())

    def test_task_ring_cannot_exceed_published_l_xi_capacity(self):
        dkg = DistributedKGC(
            threshold=2,
            node_count=3,
            crypto_mode="simulated",
            max_accumulator_size=2,
            seed=24,
        )
        with self.assertRaisesRegex(ValueError, "basis capacity"):
            dkg.build_task_ring(
                "rid",
                {"a": 1, "b": 2, "c": 3},
            )

    def test_checkpoint_preserves_public_keys_and_certificate_verification(self):
        original = DistributedKGC(
            threshold=2,
            node_count=3,
            crypto_mode="simulated",
            max_accumulator_size=3,
            seed=22,
        )
        certificate = original.threshold_sign(b"message", (1, 2))
        restored = DistributedKGC(
            threshold=2,
            node_count=3,
            crypto_mode="simulated",
            max_accumulator_size=3,
            state=original.export_state(),
        )

        self.assertEqual(restored.master_public, original.master_public)
        self.assertEqual(restored.trace_public, original.trace_public)
        self.assertEqual(restored.certificate_public, original.certificate_public)
        self.assertTrue(restored.threshold_verify(b"message", certificate))

        state = original.export_state()
        first = state.master_shares[0]
        corrupted = replace(
            state,
            master_shares=(
                replace(first, value=first.value + 1),
                *state.master_shares[1:],
            ),
        )
        with self.assertRaisesRegex(ValueError, "Feldman"):
            DistributedKGC(
                threshold=2,
                node_count=3,
                crypto_mode="simulated",
                max_accumulator_size=3,
                state=corrupted,
            )


if __name__ == "__main__":
    unittest.main()
