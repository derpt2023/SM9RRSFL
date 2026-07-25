import unittest
import sys
import types
from dataclasses import replace

import numpy as np

from sm9rrsfl import crypto as crypto_module
from sm9rrsfl import sm9_backend
from sm9rrsfl.crypto import (
    RingSignature,
    SM9RRSContext,
    ThresholdNotMetError,
    encode_group_element,
    rrs_backend_name,
    sm3_hex_bytes,
)
from sm9rrsfl.threshold import (
    DistributedKGC,
    TraceAuthorization,
    TraceAuthorizationRequest,
)


class SM9RRSFlowTest(unittest.TestCase):
    def test_v2_selects_only_the_current_group_bridge(self):
        self.assertEqual(rrs_backend_name(), sm9_backend.backend_name())
        self.assertNotIn(rrs_backend_name(), {"gmssl-native", "sm9-python-reference-v2"})
        self.assertNotIn("sm9rrsfl._native_rrs", sys.modules)

    def test_native_compatible_sm3_matches_standard_vector(self):
        self.assertEqual(
            sm3_hex_bytes(b"abc"),
            "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0",
        )

    def test_simulated_packet_verifies_and_threshold_trace_is_authorized(self):
        ctx = SM9RRSContext(
            ["client-0", "client-1", "client-2"],
            crypto_mode="simulated",
            dkg_threshold=2,
            dkg_nodes=3,
            seed=3,
        )
        update = np.ones(20, dtype=np.float32)
        packet = ctx.create_packet("client-1", update, round_id=1, task_id="task-a")

        self.assertTrue(ctx.verify_packet(packet, update))
        self.assertEqual(packet.signature.__class__, RingSignature)
        for forbidden in (
            "trapdoor",
            "event_tag",
            "_signer_identity_hint",
            "task_point",
            "task_salt",
            "ring",
        ):
            self.assertFalse(hasattr(packet, forbidden))

        evidence = ctx.build_trace_evidence(packet, update)
        with self.assertRaises(ThresholdNotMetError):
            ctx.auditor_service(("KGC-1",)).trace(evidence)
        with self.assertRaises(TypeError):
            ctx.auditor_service((1, 3))  # type: ignore[arg-type]
        result = ctx.auditor_service(("KGC-1", "KGC-3")).trace(evidence)
        self.assertEqual(result.identity, "client-1")
        self.assertTrue(ctx.verify_trace_result(evidence, result))

    def test_same_task_tag_is_stable_and_cross_task_tag_changes(self):
        ctx = SM9RRSContext(["a", "b", "c"], crypto_mode="simulated", seed=8)
        update = np.arange(16, dtype=np.float32)
        first = ctx.create_packet("b", update, round_id=1, task_id="task-a")
        second = ctx.create_packet("b", update, round_id=2, task_id="task-a")
        other_task = ctx.create_packet("b", update, round_id=1, task_id="task-b")

        self.assertEqual(first.task_tag, second.task_tag)
        self.assertNotEqual(first.task_tag, other_task.task_tag)
        self.assertNotEqual(first.tag_commitment, second.tag_commitment)

    def test_role_capabilities_expose_only_their_protocol_actions(self):
        ctx = SM9RRSContext(["a", "b", "c"], crypto_mode="simulated", seed=81)
        ctx.register_task("task")
        client = ctx.client_signer("b")
        server = ctx.as_verifier(expected_update_shape=(8,))
        auditor = ctx.auditor_service()
        update = np.arange(8, dtype=np.float32)

        packet = client.create_packet(update, round_id=1, task_id="task")
        self.assertTrue(
            server.verify_packet(
                packet,
                update,
                expected_task_id="task",
                expected_round_id=1,
            )
        )
        self.assertFalse(
            server.verify_packet(
                packet,
                np.ones(9, dtype=np.float32),
                expected_task_id="task",
                expected_round_id=1,
            )
        )
        evidence = server.build_trace_evidence(packet, update)
        self.assertTrue(auditor.verify_evidence(evidence))
        result = auditor.trace(evidence)
        self.assertTrue(server.verify_trace_result(evidence, result))
        self.assertTrue(server.archive_trace_result(evidence, result))

        self.assertFalse(hasattr(server, "sign_packet"))
        self.assertFalse(hasattr(server, "trace"))
        self.assertFalse(hasattr(server, "_ASVerifier__context"))
        self.assertFalse(hasattr(client, "_ClientSigner__context"))
        self.assertFalse(hasattr(auditor, "_AuditorService__context"))
        self.assertFalse(hasattr(client, "verify_packet"))
        self.assertFalse(hasattr(auditor, "sign_packet"))

    def test_as_and_client_object_graphs_cannot_recover_privileged_roles(self):
        ctx = SM9RRSContext(
            ["a", "b", "c"],
            crypto_mode="simulated",
            dkg_threshold=2,
            dkg_nodes=3,
            seed=810,
        )
        ctx.register_task("task")
        client_a = ctx.client_signer("a")
        client_b = ctx.client_signer("b")
        server = ctx.as_verifier(expected_update_shape=(8,))

        def reachable_instances(root):
            seen = set()
            pending = [root]
            while pending:
                value = pending.pop()
                marker = id(value)
                if marker in seen:
                    continue
                seen.add(marker)
                yield value
                if isinstance(value, dict):
                    pending.extend(value.keys())
                    pending.extend(value.values())
                elif isinstance(value, (tuple, list, set, frozenset)):
                    pending.extend(value)
                else:
                    for cls in type(value).__mro__:
                        for descriptor in vars(cls).values():
                            if not isinstance(descriptor, types.MemberDescriptorType):
                                continue
                            try:
                                pending.append(descriptor.__get__(value, type(value)))
                            except AttributeError:
                                pass
                    try:
                        pending.extend(vars(value).values())
                    except TypeError:
                        pass

        for role in (client_a, server):
            reachable = tuple(reachable_instances(role))
            self.assertFalse(any(isinstance(item, SM9RRSContext) for item in reachable))
            self.assertFalse(any(isinstance(item, DistributedKGC) for item in reachable))
            self.assertFalse(any(item.__class__.__name__ == "_TraceGateway" for item in reachable))
            for item in reachable:
                self.assertNotIn("_client_sign_keys", getattr(item, "__dict__", {}))
                for cls in type(item).__mro__:
                    slots = getattr(cls, "__slots__", ())
                    if isinstance(slots, str):
                        slots = (slots,)
                    self.assertNotIn("_client_sign_keys", slots)
        self.assertFalse(any(
            item.__class__.__name__ == "ClientSigner"
            for item in reachable_instances(server)
        ))

        for removed_dispatch_api in (
            "_CAPABILITY_BINDINGS",
            "_register_capability",
            "_resolve_capability",
        ):
            self.assertFalse(hasattr(crypto_module, removed_dispatch_api))

        with self.assertRaises(AttributeError):
            client_a.identity = "b"  # type: ignore[misc]
        update = np.arange(8, dtype=np.float32)
        unsigned_b = client_b.build_unsigned_packet(
            update,
            round_id=1,
            task_id="task",
        )
        # The packet contains no identity selector. Client A can only apply its
        # own key/witness, so tracing the result identifies A rather than B.
        packet = client_a.sign_packet(unsigned_b)
        self.assertTrue(
            server.verify_packet(
                packet,
                update,
                expected_task_id="task",
                expected_round_id=1,
            )
        )
        evidence = server.build_trace_evidence(packet, update)
        result = ctx.auditor_service().trace(evidence)
        self.assertEqual(result.identity, "a")
        with self.assertRaises(TypeError):
            ctx.auditor_service((1, 2))  # type: ignore[arg-type]

    def test_trace_path_rejects_forged_and_replayed_node_authorization(self):
        ctx = SM9RRSContext(
            ["a", "b", "c"],
            crypto_mode="simulated",
            dkg_threshold=2,
            dkg_nodes=3,
            seed=82,
        )
        update = np.arange(8, dtype=np.float32)
        packet = ctx.create_packet("b", update, round_id=1, task_id="task")
        evidence = ctx.build_trace_evidence(packet, update)
        digest = ctx.evidence_digest(evidence)
        request = TraceAuthorizationRequest(
            task_id=packet.task_id,
            rid=packet.rid,
            evidence_digest=digest,
            session_id="77" * 32,
        )
        first = ctx._dkg.trace_approval_issuer("KGC-1").approve(request)
        renamed = replace(first, node_identity="KGC-2")
        with self.assertRaises(ThresholdNotMetError):
            ctx._trace_with_authorization(
                evidence,
                TraceAuthorization(request=request, approvals=(first, renamed)),
            )

        third = ctx._dkg.trace_approval_issuer("KGC-3").approve(request)
        result = ctx._trace_with_authorization(
            evidence,
            TraceAuthorization(request=request, approvals=(first, third)),
        )
        self.assertEqual(result.identity, "b")
        with self.assertRaisesRegex(ValueError, "already been consumed"):
            ctx._trace_with_authorization(
                evidence,
                TraceAuthorization(request=request, approvals=(first, third)),
            )

    def test_every_authenticated_packet_field_is_bound(self):
        ctx = SM9RRSContext(["a", "b", "c"], crypto_mode="simulated", seed=9)
        update = np.arange(12, dtype=np.float32)
        packet = ctx.create_packet("b", update, round_id=1, task_id="task")
        signature = packet.signature
        assert signature is not None

        changed_update = update.copy()
        changed_update[0] += 1
        self.assertFalse(ctx.verify_packet(packet, changed_update))
        self.assertFalse(ctx.verify_packet(replace(packet, round_id=2), update))
        self.assertFalse(
            ctx.verify_packet(packet, update, expected_task_id="another-task")
        )
        self.assertFalse(ctx.verify_packet(packet, update, expected_round_id=2))
        self.assertFalse(ctx.verify_packet(replace(packet, task_id="unknown"), update))
        self.assertFalse(ctx.verify_packet(replace(packet, rid="00" * 32), update))
        self.assertFalse(ctx.verify_packet(replace(packet, accumulator=packet.accumulator + 1), update))
        self.assertFalse(ctx.verify_packet(replace(packet, task_tag=packet.task_tag + 1), update))
        self.assertFalse(
            ctx.verify_packet(
                replace(packet, tag_commitment=packet.tag_commitment + 1),
                update,
            )
        )
        for field in ("c", "A", "B", "C"):
            changed = replace(signature, **{field: getattr(signature, field) + 1})
            self.assertFalse(ctx.verify_packet(replace(packet, signature=changed), update))

    def test_tag_from_another_private_key_cannot_be_attached(self):
        ctx = SM9RRSContext(["a", "b", "c"], crypto_mode="simulated", seed=10)
        update = np.ones(8, dtype=np.float32)
        packet_a = ctx.create_packet("a", update, round_id=1, task_id="task")
        packet_b = ctx.create_packet("b", update, round_id=1, task_id="task")

        transplanted = replace(packet_a, task_tag=packet_b.task_tag)
        self.assertFalse(ctx.verify_packet(transplanted, update))

    def test_trace_certificate_binds_evidence_identity_tag_and_rid(self):
        ctx = SM9RRSContext(["a", "b", "c"], crypto_mode="simulated", seed=11)
        update = np.arange(10, dtype=np.float32)
        packet = ctx.create_packet("c", update, round_id=1, task_id="task")
        evidence = ctx.build_trace_evidence(packet, update)
        result = ctx.trace(evidence)

        self.assertTrue(ctx.verify_trace_result(evidence, result))
        self.assertFalse(ctx.verify_trace_result(evidence, replace(result, identity="a")))
        self.assertFalse(ctx.verify_trace_result(evidence, replace(result, rid="11" * 32)))
        self.assertFalse(ctx.verify_trace_result(evidence, replace(result, task_tag=result.task_tag + 1)))
        self.assertFalse(
            ctx.verify_trace_result(
                evidence,
                replace(
                    result,
                    certificate=replace(
                        result.certificate,
                        response=result.certificate.response + 1,
                    ),
                ),
            )
        )
        changed_update = update.copy()
        changed_update[-1] += 1
        changed_evidence = replace(evidence, model_update=changed_update)
        self.assertFalse(ctx.verify_trace_result(changed_evidence, result))

    def test_auditor_equation_does_not_use_task_point(self):
        ctx = SM9RRSContext(["a", "b"], crypto_mode="simulated", seed=12)
        update = np.ones(6, dtype=np.float32)
        packet = ctx.create_packet("a", update, round_id=1, task_id="task")
        server = ctx.as_verifier()
        evidence = server.build_trace_evidence(packet, update)
        task = getattr(server, "_ASVerifier__tasks")["task"]
        original = task.task_point
        task.task_point = original + 1
        try:
            self.assertTrue(ctx.auditor_service().verify_evidence(evidence))
            self.assertFalse(server._verify_packet(packet, update))
        finally:
            task.task_point = original

    def test_auditor_requires_as_authorization_for_exact_pending_evidence(self):
        ctx = SM9RRSContext(["a", "b"], crypto_mode="simulated", seed=120)
        update = np.ones(6, dtype=np.float32)
        packet = ctx.create_packet("a", update, round_id=1, task_id="task")
        auditor = ctx.auditor_service()

        unsigned_submission = crypto_module.TraceEvidence(
            packet=packet,
            model_update=update,
        )
        self.assertFalse(auditor.verify_evidence(unsigned_submission))
        with self.assertRaisesRegex(ValueError, "Equation \(1\)|authorization"):
            auditor.trace(unsigned_submission)

        evidence = ctx.build_trace_evidence(packet, update)
        self.assertTrue(auditor.verify_evidence(evidence))
        assert evidence.audit_authorization is not None
        forged = replace(
            evidence,
            audit_authorization=replace(
                evidence.audit_authorization,
                response=(evidence.audit_authorization.response + 1)
                % crypto_module.SCALAR_MODULUS,
            ),
        )
        self.assertFalse(auditor.verify_evidence(forged))

    def test_ring_change_rotates_rid_and_rejects_old_packet_for_new_upload(self):
        ctx = SM9RRSContext(["a", "b", "c"], crypto_mode="simulated", seed=13)
        update = np.ones(6, dtype=np.float32)
        old_packet = ctx.create_packet("b", update, round_id=1, task_id="task")
        old_rid = old_packet.rid

        new_rid = ctx.update_task_ring("task", ["a", "b"])
        self.assertNotEqual(old_rid, new_rid)
        self.assertFalse(ctx.verify_packet(old_packet, update))
        self.assertTrue(ctx.verify_ring_equation(old_packet, update))
        new_packet = ctx.create_packet("b", update, round_id=2, task_id="task")
        self.assertTrue(ctx.verify_packet(new_packet, update))

    def test_trace_after_ring_shrink_uses_only_the_selected_rid_members(self):
        ctx = SM9RRSContext(["a", "b", "c"], crypto_mode="simulated", seed=130)
        update = np.ones(6, dtype=np.float32)

        first_packet = ctx.create_packet("c", update, round_id=1, task_id="task")
        first_evidence = ctx.build_trace_evidence(first_packet, update)
        first_result = ctx.trace(first_evidence)
        self.assertEqual(first_result.identity, "c")
        self.assertTrue(ctx.archive_trace_result(first_evidence, first_result))

        ctx.update_task_ring("task", ["a", "b"])
        second_packet = ctx.create_packet("b", update, round_id=2, task_id="task")
        second_evidence = ctx.build_trace_evidence(second_packet, update)
        second_result = ctx.trace(second_evidence)

        self.assertEqual(second_result.identity, "b")
        self.assertTrue(ctx.verify_trace_result(second_evidence, second_result))

    def test_checkpoint_preserves_archived_ring_audit_and_trace_material(self):
        clients = ["a", "b", "c"]
        ctx = SM9RRSContext(clients, crypto_mode="simulated", seed=131)
        update = np.ones(6, dtype=np.float32)
        old_packet = ctx.create_packet("c", update, round_id=1, task_id="task")
        old_evidence = ctx.build_trace_evidence(old_packet, update)
        ctx.update_task_ring("task", ["a", "b"])

        restored = SM9RRSContext(
            clients,
            crypto_mode="simulated",
            seed=131,
            state=ctx.export_state(),
        )
        restored.register_task("task", ["a", "b"])

        self.assertTrue(restored.verify_ring_equation(old_packet, update))
        trace_result = restored.trace(old_evidence)
        self.assertEqual(trace_result.identity, "c")
        self.assertTrue(restored.verify_trace_result(old_evidence, trace_result))

    def test_finalize_uses_internal_pending_audit_state(self):
        ctx = SM9RRSContext(["a", "b", "c"], crypto_mode="simulated", seed=132)
        update = np.ones(6, dtype=np.float32)
        packet = ctx.create_packet("b", update, round_id=1, task_id="task")
        evidence = ctx.build_trace_evidence(packet, update)
        result = ctx.trace(evidence)

        self.assertEqual(
            ctx.pending_audit_digests("task"),
            (ctx.evidence_digest(evidence),),
        )
        # Verification alone is not an archive operation.
        self.assertTrue(ctx.verify_trace_result(evidence, result))
        with self.assertRaises(TypeError):
            ctx.finalize_task("task", pending_audits=False)
        with self.assertRaisesRegex(RuntimeError, "pending audit"):
            ctx.finalize_task("task")
        self.assertFalse(
            ctx.archive_trace_result(evidence, replace(result, identity="a"))
        )
        with self.assertRaisesRegex(RuntimeError, "pending audit"):
            ctx.finalize_task("task")

        self.assertFalse(ctx.is_task_finalized("task"))
        self.assertEqual(ctx.current_ring_id("task"), ctx.register_task("task"))
        self.assertTrue(ctx.archive_trace_result(evidence, result))
        self.assertEqual(ctx.pending_audit_digests("task"), ())
        ctx.finalize_task("task")
        self.assertTrue(ctx.is_task_finalized("task"))

    def test_trace_evidence_owns_an_immutable_update_copy(self):
        ctx = SM9RRSContext(["a", "b", "c"], crypto_mode="simulated", seed=1320)
        update = np.arange(6, dtype=np.float32)
        packet = ctx.create_packet("b", update, round_id=1, task_id="task")
        evidence = ctx.build_trace_evidence(packet, update)
        original_digest = ctx.evidence_digest(evidence)

        update[0] = 99.0
        self.assertEqual(ctx.evidence_digest(evidence), original_digest)
        self.assertFalse(evidence.model_update.flags.writeable)
        with self.assertRaises(ValueError):
            evidence.model_update[0] = 1.0

    def test_h5_evidence_uses_the_word_field_order(self):
        ctx = SM9RRSContext(["a", "b"], crypto_mode="simulated", seed=13201)
        update = np.arange(6, dtype=np.float32)
        packet = ctx.create_packet("b", update, round_id=7, task_id="task")
        evidence = ctx.build_trace_evidence(packet, update)
        assert packet.signature is not None
        assert packet.task_tag is not None and packet.tag_commitment is not None

        expected_payload = crypto_module.encode_fields(
            packet.task_id.encode("utf-8"),
            crypto_module.encode_scalar(packet.round_id),
            crypto_module.encode_update(evidence.model_update),
            bytes.fromhex(packet.rid),
            crypto_module.encode_group_element(
                packet.accumulator,
                crypto_mode=packet.crypto_mode,
            ),
            crypto_module.encode_group_element(
                packet.task_tag,
                crypto_mode=packet.crypto_mode,
            ),
            crypto_module.encode_group_element(
                packet.tag_commitment,
                crypto_mode=packet.crypto_mode,
            ),
            crypto_module.encode_scalar(packet.signature.c),
            crypto_module.encode_group_element(
                packet.signature.A,
                crypto_mode=packet.crypto_mode,
            ),
            crypto_module.encode_group_element(
                packet.signature.B,
                crypto_mode=packet.crypto_mode,
            ),
            crypto_module.encode_group_element(
                packet.signature.C,
                crypto_mode=packet.crypto_mode,
            ),
        )
        self.assertEqual(
            ctx.evidence_digest(evidence),
            crypto_module.domain_hash_hex(
                b"SM9-RRS-FL/H5/evidence/v2",
                expected_payload,
                algorithm="sha256",
            ),
        )

    def test_pending_audit_survives_checkpoint_restore(self):
        clients = ["a", "b", "c"]
        ctx = SM9RRSContext(clients, crypto_mode="simulated", seed=1321)
        update = np.ones(6, dtype=np.float32)
        packet = ctx.create_packet("b", update, round_id=1, task_id="task")
        evidence = ctx.build_trace_evidence(packet, update)

        checkpoint = ctx.export_state()
        self.assertEqual(len(checkpoint.pending_audits), 1)
        self.assertFalse(
            checkpoint.pending_audits[0].evidence.model_update.flags.writeable
        )
        restored = SM9RRSContext(
            clients,
            crypto_mode="simulated",
            seed=1321,
            state=checkpoint,
        )
        restored.register_task("task")
        with self.assertRaisesRegex(RuntimeError, "pending audit"):
            restored.finalize_task("task")
        # The old process-local variable is not needed: the checkpoint carries
        # the complete evidence required for a retry.
        retry_evidence = restored.pending_audit_evidence("task")
        self.assertEqual(len(retry_evidence), 1)
        self.assertEqual(
            restored.evidence_digest(retry_evidence[0]),
            restored.evidence_digest(evidence),
        )
        result = restored.trace(retry_evidence[0])
        self.assertTrue(restored.archive_trace_result(retry_evidence[0], result))
        restored.finalize_task("task")
        self.assertTrue(restored.is_task_finalized("task"))

    def test_finalize_destroys_task_secrets_and_prevents_reactivation(self):
        clients = ["a", "b", "c"]
        ctx = SM9RRSContext(clients, crypto_mode="simulated", seed=133)
        update = np.ones(6, dtype=np.float32)
        unsigned = ctx.build_unsigned_packet(
            "b",
            update,
            round_id=1,
            task_id="task",
        )
        packet = ctx.create_packet("b", update, round_id=1, task_id="task")
        task = ctx._tasks["task"]
        signer = ctx.client_signer("b")
        ctx.precompute_task_material("task")
        signer_task = getattr(signer, "_ClientSigner__tasks")["task"]
        self.assertIsNotNone(signer_task.task_tag)

        ctx.finalize_task("task")

        self.assertTrue(ctx.is_task_finalized("task"))
        self.assertEqual(task.task_salt, bytearray(b"\x00" * 32))
        self.assertIsNone(task.task_point)
        self.assertEqual(task.rings, {})
        self.assertNotIn("task", getattr(signer, "_ClientSigner__tasks"))
        self.assertNotIn(packet.rid, ctx._dkg._ring_delta_shares)
        self.assertNotIn(packet.rid, ctx._dkg._ring_members)
        self.assertFalse(ctx.verify_packet(packet, update))
        self.assertFalse(ctx.verify_ring_equation(packet, update))
        with self.assertRaisesRegex(ValueError, "finalized"):
            ctx.create_packet("b", update, round_id=2, task_id="task")
        with self.assertRaisesRegex(ValueError, "not registered"):
            ctx.sign_packet("b", unsigned)
        with self.assertRaisesRegex(ValueError, "finalized"):
            ctx.register_task("task")

        checkpoint = ctx.export_state()
        self.assertNotIn("task", {item.task_id for item in checkpoint.tasks})
        self.assertIn("task", checkpoint.finalized_task_ids)
        restored = SM9RRSContext(
            clients,
            crypto_mode="simulated",
            seed=133,
            state=checkpoint,
        )
        self.assertTrue(restored.is_task_finalized("task"))
        with self.assertRaisesRegex(ValueError, "finalized"):
            restored.register_task("task")

    def test_verified_result_must_be_explicitly_archived_before_finalization(self):
        ctx = SM9RRSContext(["a", "b", "c"], crypto_mode="simulated", seed=134)
        update = np.ones(6, dtype=np.float32)
        packet = ctx.create_packet("c", update, round_id=1, task_id="task")
        evidence = ctx.build_trace_evidence(packet, update)
        trace_result = ctx.trace(evidence)
        self.assertTrue(ctx.verify_trace_result(evidence, trace_result))
        with self.assertRaisesRegex(RuntimeError, "pending audit"):
            ctx.finalize_task("task")

        self.assertTrue(ctx.archive_trace_result(evidence, trace_result))
        archived = ctx.archived_audit_records("task")
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].result, trace_result)
        self.assertEqual(
            ctx.evidence_digest(archived[0].evidence),
            trace_result.evidence_digest,
        )
        self.assertFalse(archived[0].evidence.model_update.flags.writeable)
        # Returned archive views are detached: even if a caller deliberately
        # re-enables writes on its NumPy copy, the retained audit is unchanged.
        archived[0].evidence.model_update.setflags(write=True)
        archived[0].evidence.model_update[0] = 99.0
        fresh_archive = ctx.archived_audit_records("task")
        self.assertEqual(fresh_archive[0].evidence.model_update[0], 1.0)
        self.assertEqual(
            ctx.evidence_digest(fresh_archive[0].evidence),
            trace_result.evidence_digest,
        )
        ctx.finalize_task("task")

        self.assertTrue(ctx.is_task_finalized("task"))
        # The archive keeps the full evidence/result pair. This finalized live
        # context intentionally refuses further signature verification.
        self.assertFalse(ctx.verify_trace_result(evidence, trace_result))
        checkpoint = ctx.export_state()
        self.assertNotIn(
            "task",
            {item.task_id for item in checkpoint.tasks},
        )
        self.assertEqual(len(checkpoint.archived_audits), 1)
        restored = SM9RRSContext(
            ["a", "b", "c"],
            crypto_mode="simulated",
            seed=134,
            state=checkpoint,
        )
        self.assertEqual(
            restored.archived_audit_records("task")[0].result,
            trace_result,
        )

    def test_finalizing_one_task_keeps_shared_ring_material_for_another_task(self):
        ctx = SM9RRSContext(["a", "b", "c"], crypto_mode="simulated", seed=135)
        update = np.ones(6, dtype=np.float32)
        first = ctx.create_packet("a", update, round_id=1, task_id="task-a")
        second = ctx.create_packet("a", update, round_id=1, task_id="task-b")
        self.assertEqual(first.rid, second.rid)

        ctx.finalize_task("task-a")

        self.assertIn(second.rid, ctx._dkg._ring_delta_shares)
        self.assertTrue(ctx.verify_packet(second, update))

    def test_signature_metadata_size_is_constant_in_ring_size(self):
        update = np.ones(4, dtype=np.float32)
        packets = []
        for count in (3, 20):
            ctx = SM9RRSContext(
                [f"client-{index}" for index in range(count)],
                crypto_mode="simulated",
                seed=count,
            )
            packets.append(ctx.create_packet("client-1", update, round_id=1, task_id="task"))

        def metadata_size(packet):
            signature = packet.signature
            assert signature is not None
            values = (
                packet.accumulator,
                packet.task_tag,
                packet.tag_commitment,
                signature.A,
                signature.B,
                signature.C,
            )
            return 32 + 32 + sum(
                len(encode_group_element(value, crypto_mode=packet.crypto_mode))
                for value in values
            )

        self.assertEqual(metadata_size(packets[0]), metadata_size(packets[1]))

    @unittest.skipUnless(sm9_backend.available(), "GmSSL SM9 v2 bridge is not built")
    def test_real_sm9_equations_and_trace(self):
        ctx = SM9RRSContext(["a", "b", "c"], crypto_mode="sm9", seed=14)
        update = np.arange(8, dtype=np.float32)
        ctx.register_task("task")
        ctx.precompute_task_material("task")
        packet = ctx.create_packet("b", update, round_id=1, task_id="task")

        self.assertTrue(ctx.verify_packet(packet, update))
        assert packet.signature is not None
        self.assertEqual(len(packet.accumulator), 65)
        self.assertEqual(len(packet.signature.A), 65)
        self.assertEqual(len(packet.signature.B), 65)
        self.assertEqual(len(packet.signature.C), 129)
        self.assertEqual(len(packet.task_tag), 384)
        self.assertEqual(len(packet.tag_commitment), 384)
        evidence = ctx.build_trace_evidence(packet, update)
        result = ctx.trace(evidence)
        self.assertEqual(result.identity, "b")
        self.assertTrue(ctx.verify_trace_result(evidence, result))


if __name__ == "__main__":
    unittest.main()
