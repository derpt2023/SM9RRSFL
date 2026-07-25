"""SM9-RRS-FL protocol implementation matching Sections 4.3.1--4.3.4.

The online signature is the constant-size tuple ``(c, A, B, C)``.  A stable
task tag is bound to that same signature by the second verification equation;
there is no encrypted identity trapdoor and no independent NIZK object.

``SM9RRSContext`` is a trusted experiment orchestrator.  It distributes
role-specific data transfer objects in one direction: each client contains only
its own private key and witness; AS contains verification material, ``h_t``, an
independent audit-submission credential and its pending ledger; only the Auditor
can reach the narrow signed-authorization D-KGC gateway.  Python object privacy
is not OS isolation;
production deployments must put these roles in separate processes or hosts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import os
import random
from random import SystemRandom
from typing import Any, Iterable, Sequence

import numpy as np

from gmssl import sm3

from . import sm9_backend
from .threshold import (
    DistributedKGC,
    DistributedKGCState,
    SCALAR_MODULUS,
    TraceApprovalIssuer,
    TraceAuthorization,
    TraceAuthorizationRequest,
    ThresholdCertificateVerifier,
    ThresholdNotMetError,
)

try:
    from ._native_sm3 import sm3_hexdigest as _native_sm3_hexdigest
except ImportError:
    _native_sm3_hexdigest = None


PROTOCOL_VERSION = 2
HID_SIGN = 0x01


@dataclass(frozen=True)
class RingSignature:
    """The paper's constant-size signature ``sigma=(c,A,B,C)``."""

    c: int
    A: Any
    B: Any
    C: Any


@dataclass(frozen=True)
class RRSPacket:
    """Authentication metadata attached to one transmitted model update.

    ``task_id`` and ``round_id`` are the authenticated transport/session
    context required to reconstruct ``M``.  The Word packet lists the model
    update itself; this code passes that large array separately to avoid keeping
    a duplicate copy inside every metadata object.
    """

    task_id: str
    round_id: int
    rid: str
    accumulator: Any
    update_digest: str
    task_tag: Any | None
    tag_commitment: Any | None
    signature: RingSignature | None
    crypto_mode: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class AuditAuthorization:
    """Implementation-layer AS authorization for submitting one ``E_pi``.

    This authenticated-channel ticket is not part of the paper's evidence or
    H5 transcript.  It prevents an Auditor capability from turning any valid
    Equation-(1) packet into a trace request unless AS first registered that
    exact evidence after the C_tol policy decision.
    """

    commitment: Any
    response: int


@dataclass(frozen=True)
class TraceEvidence:
    """Auditable evidence ``E_pi`` for the update that reached ``C_tol``."""

    packet: RRSPacket
    model_update: np.ndarray
    audit_authorization: AuditAuthorization | None = None


@dataclass(frozen=True)
class TraceResult:
    """Auditor response ``(ID_j,Tag_pi,RID,tau_trace)``."""

    identity: str
    task_tag: Any
    rid: str
    certificate: Any
    evidence_digest: str


@dataclass(frozen=True)
class ArchivedAuditRecord:
    """Durable evidence/result pair containing no destroyed task-key material."""

    task_id: str
    evidence: TraceEvidence
    result: TraceResult


@dataclass(frozen=True)
class TaskStateSnapshot:
    """Public ring history plus the protected task salt for one task."""

    task_id: str
    task_salt: bytes
    current_rid: str
    rings: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class PendingAuditSnapshot:
    """Durable trace evidence whose audit lifecycle is not yet archived.

    Keeping the complete immutable evidence, rather than only its digest,
    makes an interrupted audit retryable after checkpoint restore.  The digest
    is retained explicitly so checkpoint corruption is detected before the
    task is reactivated.
    """

    task_id: str
    evidence_digest: str
    evidence: TraceEvidence


@dataclass(frozen=True)
class SM9RRSState:
    """Trusted checkpoint needed to resume cryptographic and audit state."""

    dkg_state: DistributedKGCState
    tasks: tuple[TaskStateSnapshot, ...]
    # Protected AS control-plane key for authenticating C_tol submissions to
    # Auditor/D-KGC endpoints.  It is not an SM9 client or trace secret.
    audit_authorization_secret: int
    # These tombstones contain no task cryptographic material. They prevent a
    # destroyed task from being silently recreated after a process restart.
    finalized_task_ids: tuple[str, ...] = ()
    # Pending evidence must survive a restart; otherwise a restored process
    # could neither retry the trace nor safely destroy h_t.
    pending_audits: tuple[PendingAuditSnapshot, ...] = ()
    # Completed evidence and its threshold-certified result remain auditable
    # after h_t and kappa_t are destroyed; storage still needs access control.
    archived_audits: tuple[ArchivedAuditRecord, ...] = ()


@dataclass
class _RingRecord:
    ring: tuple[str, ...]
    rid: str
    accumulator: Any
    witnesses: dict[str, Any]
    g1: Any


@dataclass
class _TaskState:
    task_id: str
    # Mutable so finalization can overwrite the live copy of kappa_t.
    task_salt: bytearray
    task_point: Any
    current_rid: str
    rings: dict[str, _RingRecord]


@dataclass(frozen=True)
class _PublicRingRecord:
    """The exact public ring state needed by AS/Auditor verification."""

    ring: tuple[str, ...]
    rid: str
    accumulator: Any
    g1: Any


@dataclass(frozen=True)
class _ClientRingMaterial:
    """One client's own current witness and reusable signing values."""

    rid: str
    accumulator: Any
    witness: Any
    g1: Any
    g2: Any


@dataclass
class _ClientTaskMaterial:
    task_point: Any
    ring: _ClientRingMaterial
    task_tag: Any | None = None


@dataclass
class _VerificationTaskState:
    task_point: Any
    current_rid: str
    rings: dict[str, _PublicRingRecord]


class ClientSigner:
    """A signer containing one and only one client's private key.

    The orchestrator pushes this client's current witness and task point into
    the object.  There is deliberately no callback, token broker, context, DKG,
    other-client key map, or tracing operation reachable from this instance.
    """

    __slots__ = (
        "__crypto_mode",
        "__identity",
        "__identity_scalar",
        "__p2",
        "__private_key",
        "__rng",
        "__tasks",
    )

    def __init__(
        self,
        *,
        identity: str,
        identity_scalar: int,
        private_key: Any,
        p2: Any,
        crypto_mode: str,
        seed: int,
    ) -> None:
        self.__identity = identity
        self.__identity_scalar = identity_scalar
        self.__private_key = private_key
        self.__p2 = p2
        self.__crypto_mode = crypto_mode
        self.__rng: random.Random | SystemRandom = (
            random.Random(seed) if crypto_mode == "simulated" else SystemRandom()
        )
        self.__tasks: dict[str, _ClientTaskMaterial] = {}

    @property
    def identity(self) -> str:
        return self.__identity

    def _install_task_material(
        self,
        task_id: str,
        *,
        task_point: Any,
        rid: str,
        accumulator: Any,
        witness: Any,
        g1: Any,
    ) -> None:
        g2 = _pair(
            _g1_add(witness, self.__private_key, self.__crypto_mode),
            self.__p2,
            self.__crypto_mode,
        )
        previous = self.__tasks.get(task_id)
        task_tag = (
            previous.task_tag
            if previous is not None
            and _group_equal(previous.task_point, task_point, self.__crypto_mode)
            else None
        )
        self.__tasks[task_id] = _ClientTaskMaterial(
            task_point=task_point,
            ring=_ClientRingMaterial(
                rid=rid,
                accumulator=accumulator,
                witness=witness,
                g1=g1,
                g2=g2,
            ),
            task_tag=task_tag,
        )

    def _drop_task_material(self, task_id: str) -> None:
        self.__tasks.pop(task_id, None)

    def _precompute_task_material(self, task_id: str) -> None:
        self.__task_tag(self.__require_task(task_id))

    def digest_update(self, update: np.ndarray) -> str:
        return digest_update(update, algorithm=_digest_algorithm(self.__crypto_mode))

    def build_unsigned_packet(
        self,
        update: np.ndarray,
        *,
        round_id: int,
        task_id: str,
        update_digest: str | None = None,
    ) -> RRSPacket:
        if round_id < 1:
            raise ValueError("round_id must be positive")
        task = self.__require_task(task_id)
        if update_digest is None:
            update_digest = self.digest_update(update)
        return RRSPacket(
            task_id=task_id,
            round_id=round_id,
            rid=task.ring.rid,
            accumulator=task.ring.accumulator,
            update_digest=update_digest,
            task_tag=None,
            tag_commitment=None,
            signature=None,
            crypto_mode=self.__crypto_mode,
        )

    def sign_packet(self, unsigned: RRSPacket) -> RRSPacket:
        if unsigned.protocol_version != PROTOCOL_VERSION:
            raise ValueError("unsupported RRS protocol version")
        if unsigned.crypto_mode != self.__crypto_mode:
            raise ValueError("packet crypto mode does not match signer")
        if unsigned.signature is not None or unsigned.task_tag is not None:
            raise ValueError("packet is already signed")
        task = self.__require_task(unsigned.task_id)
        ring = task.ring
        if unsigned.rid != ring.rid:
            raise ValueError("signing requires this client's current task ring")
        if not _group_equal(
            unsigned.accumulator,
            ring.accumulator,
            self.__crypto_mode,
        ):
            raise ValueError("packet accumulator does not match RID")

        task_tag = self.__task_tag(task)
        message = _message_bytes(unsigned, self.__crypto_mode)
        while True:
            r1 = self.__random_scalar()
            r2 = self.__random_scalar()
            omega = _gt_multiply(
                _gt_power(ring.g1, r1, self.__crypto_mode),
                _gt_power(ring.g2, r2, self.__crypto_mode),
                self.__crypto_mode,
            )
            tag_commitment = _gt_power(task_tag, r1, self.__crypto_mode)
            c = challenge_scalar(
                unsigned.rid,
                message,
                task_tag,
                tag_commitment,
                omega,
                crypto_mode=self.__crypto_mode,
            )
            response = (r1 - c) % SCALAR_MODULUS
            if response == 0:
                continue
            c_scalar = (
                r2 * scalar_inverse(response) + self.__identity_scalar
            ) % SCALAR_MODULUS
            if c_scalar != 0:
                break
        signature = RingSignature(
            c=c,
            A=_g1_multiply(ring.witness, response, self.__crypto_mode),
            B=_g1_multiply(self.__private_key, response, self.__crypto_mode),
            C=_g2_multiply(self.__p2, c_scalar, self.__crypto_mode),
        )
        return replace(
            unsigned,
            task_tag=task_tag,
            tag_commitment=tag_commitment,
            signature=signature,
        )

    def create_packet(
        self,
        update: np.ndarray,
        *,
        round_id: int,
        task_id: str,
    ) -> RRSPacket:
        unsigned = self.build_unsigned_packet(
            update,
            round_id=round_id,
            task_id=task_id,
        )
        return self.sign_packet(unsigned)

    def __require_task(self, task_id: str) -> _ClientTaskMaterial:
        task = self.__tasks.get(task_id)
        if task is None:
            raise ValueError(f"client has no active material for task: {task_id}")
        return task

    def __task_tag(self, task: _ClientTaskMaterial) -> Any:
        if task.task_tag is None:
            task.task_tag = _pair(
                self.__private_key,
                task.task_point,
                self.__crypto_mode,
            )
        return task.task_tag

    def __random_scalar(self) -> int:
        return self.__rng.randrange(1, SCALAR_MODULUS)


class ASVerifier:
    """AS verification state with no client key, D-KGC, or trace gateway.

    Its independent control-plane key only authenticates evidence submissions
    over the modeled AS-to-Auditor/D-KGC channel; it cannot sign an RRS packet
    or issue a threshold trace certificate.
    """

    __slots__ = (
        "__audit_authorization_public",
        "__audit_authorization_secret",
        "__certificate_verifier",
        "__archived_audits",
        "__crypto_mode",
        "__expected_update_shape",
        "__master_public",
        "__p2",
        "__pending_audits",
        "__tasks",
        "__trace_public",
    )

    def __init__(
        self,
        *,
        p2: Any,
        master_public: Any,
        trace_public: Any,
        certificate_public: Any,
        audit_authorization_secret: int,
        crypto_mode: str,
        pending_audits: dict[str, dict[str, TraceEvidence]] | None = None,
        archived_audits: Sequence[ArchivedAuditRecord] = (),
    ) -> None:
        self.__p2 = p2
        self.__master_public = master_public
        self.__trace_public = trace_public
        self.__crypto_mode = crypto_mode
        if not 1 <= audit_authorization_secret < SCALAR_MODULUS:
            raise ValueError("AS audit-authorization secret must be in Z_N*")
        self.__audit_authorization_secret = audit_authorization_secret
        self.__audit_authorization_public = _g2_multiply(
            p2,
            audit_authorization_secret,
            crypto_mode,
        )
        self.__certificate_verifier = ThresholdCertificateVerifier(
            certificate_public=certificate_public,
            p2=p2,
            crypto_mode=crypto_mode,
        )
        self.__expected_update_shape: tuple[int, ...] | None = None
        self.__pending_audits: dict[str, dict[str, TraceEvidence]] = {}
        for task_id, records in (pending_audits or {}).items():
            copied: dict[str, TraceEvidence] = {}
            for digest, evidence in records.items():
                owned = _owned_trace_evidence(evidence)
                if (
                    owned.packet.task_id != task_id
                    or _evidence_digest(owned, crypto_mode) != digest
                    or not self.__verify_audit_authorization(owned)
                ):
                    raise ValueError(
                        "checkpoint pending evidence has invalid AS authorization"
                    )
                copied[digest] = owned
            self.__pending_audits[task_id] = copied
        self.__archived_audits: dict[
            str,
            dict[str, ArchivedAuditRecord],
        ] = {}
        for record in archived_audits:
            self.__restore_archived_record(record)
        self.__tasks: dict[str, _VerificationTaskState] = {}

    def _set_expected_update_shape(
        self,
        expected_update_shape: tuple[int, ...] | None,
    ) -> None:
        self.__expected_update_shape = expected_update_shape

    def _install_task_ring(
        self,
        task_id: str,
        task_point: Any,
        record: _PublicRingRecord,
        *,
        make_current: bool,
    ) -> None:
        task = self.__tasks.get(task_id)
        if task is None:
            task = _VerificationTaskState(
                task_point=task_point,
                current_rid=record.rid if make_current else "",
                rings={},
            )
            self.__tasks[task_id] = task
        elif not _group_equal(task.task_point, task_point, self.__crypto_mode):
            raise RuntimeError("AS received inconsistent h_t for one task")
        existing = task.rings.get(record.rid)
        if existing is not None and existing != record:
            raise RuntimeError("AS received conflicting public ring material")
        task.rings[record.rid] = record
        if make_current:
            task.current_rid = record.rid

    def _drop_task(self, task_id: str) -> None:
        self.__tasks.pop(task_id, None)
        self.__pending_audits.pop(task_id, None)

    def _pending_snapshots(self) -> tuple[PendingAuditSnapshot, ...]:
        return tuple(
            PendingAuditSnapshot(
                task_id=task_id,
                evidence_digest=digest,
                evidence=_owned_trace_evidence(
                    self.__pending_audits[task_id][digest]
                ),
            )
            for task_id in sorted(self.__pending_audits)
            for digest in sorted(self.__pending_audits[task_id])
        )

    def _archived_snapshots(self) -> tuple[ArchivedAuditRecord, ...]:
        return tuple(
            _copy_archived_record(self.__archived_audits[task_id][digest])
            for task_id in sorted(self.__archived_audits)
            for digest in sorted(self.__archived_audits[task_id])
        )

    def pending_audit_digests(self, task_id: str) -> tuple[str, ...]:
        return tuple(sorted(self.__pending_audits.get(task_id, ())))

    def pending_audit_evidence(self, task_id: str) -> tuple[TraceEvidence, ...]:
        records = self.__pending_audits.get(task_id, {})
        return tuple(
            _owned_trace_evidence(records[digest]) for digest in sorted(records)
        )

    def archived_audit_records(
        self,
        task_id: str,
    ) -> tuple[ArchivedAuditRecord, ...]:
        records = self.__archived_audits.get(task_id, {})
        return tuple(
            _copy_archived_record(records[digest]) for digest in sorted(records)
        )

    def verify_packet(
        self,
        packet: RRSPacket,
        update: np.ndarray,
        *,
        expected_task_id: str,
        expected_round_id: int,
    ) -> bool:
        if (
            self.__expected_update_shape is not None
            and np.asarray(update).shape != self.__expected_update_shape
        ):
            return False
        return self._verify_packet(
            packet,
            update,
            expected_task_id=expected_task_id,
            expected_round_id=expected_round_id,
        )

    def _verify_packet(
        self,
        packet: RRSPacket,
        update: np.ndarray,
        *,
        expected_task_id: str | None = None,
        expected_round_id: int | None = None,
    ) -> bool:
        if expected_task_id is not None and packet.task_id != expected_task_id:
            return False
        if expected_round_id is not None and packet.round_id != expected_round_id:
            return False
        if not self.verify_ring_equation(
            packet,
            update,
            allow_historical_ring=False,
        ):
            return False
        return self.__verify_tag_equation(packet)

    def verify_ring_equation(
        self,
        packet: RRSPacket,
        update: np.ndarray,
        *,
        allow_historical_ring: bool = True,
    ) -> bool:
        task = self.__tasks.get(packet.task_id)
        if task is None:
            return False
        return _verify_ring_equation(
            packet,
            update,
            task=task,
            p2=self.__p2,
            master_public=self.__master_public,
            trace_public=self.__trace_public,
            crypto_mode=self.__crypto_mode,
            allow_historical_ring=allow_historical_ring,
        )

    def build_trace_evidence(
        self,
        packet: RRSPacket,
        update: np.ndarray,
    ) -> TraceEvidence:
        if (
            self.__expected_update_shape is not None
            and np.asarray(update).shape != self.__expected_update_shape
        ):
            raise ValueError("model update shape does not match the registered task")
        if not self._verify_packet(packet, update):
            raise ValueError("only a fully verified packet can become trace evidence")
        retained_update = np.array(update, dtype=np.float32, order="C", copy=True)
        retained_update.setflags(write=False)
        unsigned_evidence = TraceEvidence(
            packet=packet,
            model_update=retained_update,
        )
        digest = self.evidence_digest(unsigned_evidence)
        evidence = replace(
            unsigned_evidence,
            audit_authorization=_sign_audit_authorization(
                secret=self.__audit_authorization_secret,
                p2=self.__p2,
                task_id=packet.task_id,
                rid=packet.rid,
                evidence_digest=digest,
                crypto_mode=self.__crypto_mode,
            ),
        )
        task_pending = self.__pending_audits.setdefault(packet.task_id, {})
        existing = task_pending.get(digest)
        if existing is None:
            task_pending[digest] = _owned_trace_evidence(evidence)
        return _owned_trace_evidence(task_pending[digest])

    def verify_trace_result(
        self,
        evidence: TraceEvidence,
        result: TraceResult,
    ) -> bool:
        try:
            packet = evidence.packet
            if not self.__verify_audit_authorization(evidence):
                return False
            if not self.verify_ring_equation(
                packet,
                evidence.model_update,
                allow_historical_ring=True,
            ):
                return False
            if not self.__verify_tag_equation(packet):
                return False
            evidence_digest = self.evidence_digest(evidence)
            if result.evidence_digest != evidence_digest or result.rid != packet.rid:
                return False
            if packet.task_tag is None or not _gt_equal(
                result.task_tag,
                packet.task_tag,
                self.__crypto_mode,
            ):
                return False
            task = self.__tasks.get(packet.task_id)
            if task is None:
                return False
            record = task.rings.get(result.rid)
            if record is None or result.identity not in record.ring:
                return False
            return self.__certificate_verifier.verify(
                _trace_message(
                    evidence_digest,
                    result.identity,
                    result.task_tag,
                    result.rid,
                    self.__crypto_mode,
                ),
                result.certificate,
            )
        except (AssertionError, KeyError, OverflowError, TypeError, ValueError):
            return False

    def archive_trace_result(
        self,
        evidence: TraceEvidence,
        result: TraceResult,
    ) -> bool:
        """Explicitly archive a verified result and close its pending audit."""

        try:
            digest_before = self.evidence_digest(evidence)
            task_pending = self.__pending_audits.get(evidence.packet.task_id)
            if task_pending is None or digest_before not in task_pending:
                return False
            if not self.verify_trace_result(evidence, result):
                return False
            if self.evidence_digest(evidence) != digest_before:
                return False
            archived = ArchivedAuditRecord(
                task_id=evidence.packet.task_id,
                evidence=_owned_trace_evidence(evidence),
                result=result,
            )
            self.__archived_audits.setdefault(
                evidence.packet.task_id,
                {},
            )[digest_before] = archived
            task_pending.pop(digest_before, None)
            if not task_pending:
                self.__pending_audits.pop(evidence.packet.task_id, None)
            return True
        except (AssertionError, KeyError, OverflowError, TypeError, ValueError):
            return False

    def __restore_archived_record(self, record: ArchivedAuditRecord) -> None:
        if not isinstance(record, ArchivedAuditRecord):
            raise TypeError("checkpoint archived audit has an invalid type")
        evidence = _owned_trace_evidence(record.evidence)
        if not record.task_id or record.task_id != evidence.packet.task_id:
            raise ValueError("archived audit task does not match its evidence")
        digest = _evidence_digest(evidence, self.__crypto_mode)
        result = record.result
        if (
            not self.__verify_audit_authorization(evidence)
            or result.evidence_digest != digest
            or result.rid != evidence.packet.rid
            or evidence.packet.task_tag is None
            or not _gt_equal(
                result.task_tag,
                evidence.packet.task_tag,
                self.__crypto_mode,
            )
            or not result.identity
            or not self.__certificate_verifier.verify(
                _trace_message(
                    digest,
                    result.identity,
                    result.task_tag,
                    result.rid,
                    self.__crypto_mode,
                ),
                result.certificate,
            )
        ):
            raise ValueError("checkpoint archived audit failed certificate binding")
        records = self.__archived_audits.setdefault(record.task_id, {})
        if digest in records:
            raise ValueError("checkpoint contains a duplicate archived audit")
        records[digest] = ArchivedAuditRecord(
            task_id=record.task_id,
            evidence=evidence,
            result=result,
        )

    def tag_key(self, packet: RRSPacket) -> str:
        if packet.task_tag is None:
            return ""
        return element_digest(
            packet.task_tag,
            crypto_mode=self.__crypto_mode,
        )

    def evidence_digest(self, evidence: TraceEvidence) -> str:
        return _evidence_digest(evidence, self.__crypto_mode)

    def __verify_audit_authorization(self, evidence: TraceEvidence) -> bool:
        try:
            return _verify_audit_authorization(
                evidence.audit_authorization,
                public=self.__audit_authorization_public,
                p2=self.__p2,
                task_id=evidence.packet.task_id,
                rid=evidence.packet.rid,
                evidence_digest=_evidence_digest(evidence, self.__crypto_mode),
                crypto_mode=self.__crypto_mode,
            )
        except (OverflowError, TypeError, ValueError):
            return False

    def __verify_tag_equation(self, packet: RRSPacket) -> bool:
        task = self.__tasks.get(packet.task_id)
        if task is None:
            return False
        return _verify_tag_equation(packet, task.task_point, self.__crypto_mode)


class _TraceGateway:
    """Narrow signed-authorization gateway around D-KGC tracing operations.

    It intentionally has no reference to :class:`SM9RRSContext` or any client
    signer.  The gateway is reachable only from an Auditor object, never from
    AS.  A production deployment replaces this in-process object with mutually
    authenticated RPC endpoints at independently administered D-KGC nodes.
    """

    __slots__ = (
        "__audit_authorization_public",
        "__crypto_mode",
        "__dkg",
        "__identity_scalars",
        "__master_public",
        "__p2",
        "__tasks",
        "__trace_public",
    )

    def __init__(
        self,
        *,
        dkg: DistributedKGC,
        identity_scalars: dict[str, int],
        p2: Any,
        master_public: Any,
        trace_public: Any,
        audit_authorization_public: Any,
        crypto_mode: str,
    ) -> None:
        self.__dkg = dkg
        self.__identity_scalars = dict(identity_scalars)
        self.__p2 = p2
        self.__master_public = master_public
        self.__trace_public = trace_public
        self.__audit_authorization_public = audit_authorization_public
        self.__crypto_mode = crypto_mode
        self.__tasks: dict[str, _VerificationTaskState] = {}

    def install_task_ring(
        self,
        task_id: str,
        task_point: Any,
        record: _PublicRingRecord,
        *,
        make_current: bool,
    ) -> None:
        task = self.__tasks.get(task_id)
        if task is None:
            task = _VerificationTaskState(
                task_point=task_point,
                current_rid=record.rid if make_current else "",
                rings={},
            )
            self.__tasks[task_id] = task
        elif not _group_equal(task.task_point, task_point, self.__crypto_mode):
            raise RuntimeError("trace gateway received inconsistent h_t")
        existing = task.rings.get(record.rid)
        if existing is not None and existing != record:
            raise RuntimeError("trace gateway received conflicting ring material")
        task.rings[record.rid] = record
        if make_current:
            task.current_rid = record.rid

    def drop_task(self, task_id: str) -> None:
        self.__tasks.pop(task_id, None)

    def verify_evidence(self, evidence: TraceEvidence) -> bool:
        task = self.__tasks.get(evidence.packet.task_id)
        if task is None:
            return False
        try:
            evidence_digest = _evidence_digest(evidence, self.__crypto_mode)
        except (OverflowError, TypeError, ValueError):
            return False
        if not _verify_audit_authorization(
            evidence.audit_authorization,
            public=self.__audit_authorization_public,
            p2=self.__p2,
            task_id=evidence.packet.task_id,
            rid=evidence.packet.rid,
            evidence_digest=evidence_digest,
            crypto_mode=self.__crypto_mode,
        ):
            return False
        return _verify_ring_equation(
            evidence.packet,
            evidence.model_update,
            task=task,
            p2=self.__p2,
            master_public=self.__master_public,
            trace_public=self.__trace_public,
            crypto_mode=self.__crypto_mode,
            allow_historical_ring=True,
        )

    def trace(
        self,
        evidence: TraceEvidence,
        authorization: TraceAuthorization,
    ) -> TraceResult:
        if not self.verify_evidence(evidence):
            raise ValueError("Auditor rejected Equation (1) or incomplete evidence")
        packet = evidence.packet
        task = self.__tasks[packet.task_id]
        record = task.rings[packet.rid]
        signature = packet.signature
        assert signature is not None and packet.task_tag is not None
        evidence_digest = _evidence_digest(evidence, self.__crypto_mode)
        ring_identity_scalars = {
            identity: self.__identity_scalars[identity]
            for identity in record.ring
        }
        session = self.__dkg.begin_trace(
            authorization,
            expected_task_id=packet.task_id,
            expected_rid=packet.rid,
            expected_evidence_digest=evidence_digest,
            expected_task_point=task.task_point,
            expected_signature_a=signature.A,
            expected_signature_b=signature.B,
            expected_task_tag=packet.task_tag,
        )
        try:
            identity = self.__dkg.trace_match(
                rid=packet.rid,
                identities=record.ring,
                identity_scalars=ring_identity_scalars,
                signature_a=signature.A,
                signature_b=signature.B,
                session=session,
            )
            reconstructed_tag = self.__dkg.reconstruct_candidate_tag(
                ring_identity_scalars[identity],
                task.task_point,
                session,
            )
            if not _gt_equal(
                packet.task_tag,
                reconstructed_tag,
                self.__crypto_mode,
            ):
                raise ValueError("Equation (6) task-tag consistency check failed")
            certificate = self.__dkg.threshold_sign_authorized(
                _trace_message(
                    evidence_digest,
                    identity,
                    packet.task_tag,
                    packet.rid,
                    self.__crypto_mode,
                ),
                session,
                task_tag=packet.task_tag,
            )
            return TraceResult(
                identity=identity,
                task_tag=packet.task_tag,
                rid=packet.rid,
                certificate=certificate,
                evidence_digest=evidence_digest,
            )
        finally:
            self.__dkg.end_trace(session)


class _TraceApprovalEndpoint:
    """One logical D-KGC endpoint that verifies AS submission authorization."""

    __slots__ = (
        "__audit_authorization_public",
        "__crypto_mode",
        "__issuer",
        "__p2",
    )

    def __init__(
        self,
        *,
        issuer: TraceApprovalIssuer,
        audit_authorization_public: Any,
        p2: Any,
        crypto_mode: str,
    ) -> None:
        self.__issuer = issuer
        self.__audit_authorization_public = audit_authorization_public
        self.__p2 = p2
        self.__crypto_mode = crypto_mode

    def approve(
        self,
        request: TraceAuthorizationRequest,
        evidence: TraceEvidence,
    ):
        evidence_digest = _evidence_digest(evidence, self.__crypto_mode)
        if (
            request.task_id != evidence.packet.task_id
            or request.rid != evidence.packet.rid
            or request.evidence_digest != evidence_digest
            or not _verify_audit_authorization(
                evidence.audit_authorization,
                public=self.__audit_authorization_public,
                p2=self.__p2,
                task_id=request.task_id,
                rid=request.rid,
                evidence_digest=request.evidence_digest,
                crypto_mode=self.__crypto_mode,
            )
        ):
            raise ValueError("D-KGC node rejected missing or forged AS authorization")
        return self.__issuer.approve(request)


class AuditorService:
    """Auditor public verification plus signed D-KGC approval orchestration."""

    __slots__ = ("__gateway", "__trace_issuers")

    def __init__(
        self,
        *,
        gateway: _TraceGateway,
        trace_issuers: tuple[_TraceApprovalEndpoint, ...],
    ) -> None:
        self.__gateway = gateway
        self.__trace_issuers = trace_issuers

    def verify_evidence(self, evidence: TraceEvidence) -> bool:
        return self.__gateway.verify_evidence(evidence)

    def trace(self, evidence: TraceEvidence) -> TraceResult:
        if not self.verify_evidence(evidence):
            raise ValueError("Auditor rejected Equation (1) or incomplete evidence")
        packet = evidence.packet
        # The digest is mode-bound inside the gateway; packet.crypto_mode is
        # already checked by Equation (1), so it is safe to use here.
        evidence_digest = _evidence_digest(evidence, packet.crypto_mode)
        request = TraceAuthorizationRequest.create(
            task_id=packet.task_id,
            rid=packet.rid,
            evidence_digest=evidence_digest,
        )
        authorization = TraceAuthorization(
            request=request,
            approvals=tuple(
                issuer.approve(request, evidence) for issuer in self.__trace_issuers
            ),
        )
        return self.__gateway.trace(evidence, authorization)


class SM9RRSContext:
    """Trusted experiment harness issuing role-limited protocol capabilities."""

    def __init__(
        self,
        client_ids: list[str],
        *,
        crypto_mode: str = "sm9",
        dkg_threshold: int = 2,
        dkg_nodes: int = 3,
        seed: int = 0,
        state: SM9RRSState | None = None,
    ) -> None:
        if crypto_mode not in {"sm9", "simulated"}:
            raise ValueError("crypto_mode must be 'sm9' or 'simulated'")
        if not client_ids:
            raise ValueError("client_ids must not be empty")
        canonical_clients = tuple(str(identity) for identity in client_ids)
        if len(set(canonical_clients)) != len(canonical_clients):
            raise ValueError("client identities must be unique")

        self.client_ids = canonical_clients
        self.crypto_mode = crypto_mode
        self.dkg_threshold = dkg_threshold
        self.dkg_nodes = dkg_nodes
        self._seed = seed
        self._dkg = DistributedKGC(
            threshold=dkg_threshold,
            node_count=dkg_nodes,
            crypto_mode=crypto_mode,
            max_accumulator_size=len(canonical_clients),
            seed=seed,
            state=state.dkg_state if state is not None else None,
        )
        if not self._dkg.validate_public_relations():
            raise RuntimeError("D-KGC public parameter validation failed")

        self.p1 = self._dkg.p1
        self.p2 = self._dkg.p2
        if state is not None:
            audit_authorization_secret = int(state.audit_authorization_secret)
        else:
            audit_rng: random.Random | SystemRandom = (
                random.Random(seed ^ 0x41532D4155444954)
                if crypto_mode == "simulated"
                else SystemRandom()
            )
            audit_authorization_secret = audit_rng.randrange(1, SCALAR_MODULUS)
        if not 1 <= audit_authorization_secret < SCALAR_MODULUS:
            raise ValueError("checkpoint AS authorization key is invalid")
        self._audit_authorization_secret = audit_authorization_secret
        self._audit_authorization_public = _g2_multiply(
            self.p2,
            audit_authorization_secret,
            crypto_mode,
        )
        self.master_public = self._dkg.master_public
        self.trace_public = self._dkg.trace_public
        self.trace_basis = self._dkg.trace_basis
        self.trace_certificate_public = self._dkg.certificate_public
        # Compatibility names used by accumulator tests and earlier callers.
        self.sign_public = (
            self.p1,
            self.p2,
            self.master_public,
            self._dkg.master_pairing,
        )
        self.rrs_backend = (
            sm9_backend.backend_name()
            if crypto_mode == "sm9"
            else "simulated-word-v2"
        )

        self._identity_scalars = {
            identity: identity_scalar(identity, crypto_mode=self.crypto_mode)
            for identity in self.client_ids
        }
        if len(set(self._identity_scalars.values())) != len(self._identity_scalars):
            raise RuntimeError("H1 collision between task-ring client identities")
        self._client_signers = {
            identity: ClientSigner(
                identity=identity,
                identity_scalar=scalar,
                private_key=self._dkg.extract_signing_key(scalar),
                p2=self.p2,
                crypto_mode=self.crypto_mode,
                seed=seed + 0x525253 + index,
            )
            for index, (identity, scalar) in enumerate(
                self._identity_scalars.items()
            )
        }
        self._finalized_tasks = (
            set(state.finalized_task_ids) if state is not None else set()
        )
        self._task_states_to_restore = (
            {snapshot.task_id: snapshot for snapshot in state.tasks}
            if state is not None
            else {}
        )
        inconsistent = self._finalized_tasks & self._task_states_to_restore.keys()
        if inconsistent:
            raise ValueError(
                "checkpoint contains both active and finalized task state: "
                f"{sorted(inconsistent)}"
            )
        pending_audits: dict[str, dict[str, TraceEvidence]] = {}
        if state is not None:
            restorable_task_ids = set(self._task_states_to_restore)
            for pending in state.pending_audits:
                if not pending.task_id or pending.task_id not in restorable_task_ids:
                    raise ValueError(
                        "pending audit references a task absent from the checkpoint"
                    )
                if pending.task_id in self._finalized_tasks:
                    raise ValueError("finalized task cannot contain a pending audit")
                try:
                    digest_bytes = bytes.fromhex(pending.evidence_digest)
                except ValueError as exc:
                    raise ValueError("pending audit digest is not hexadecimal") from exc
                if len(digest_bytes) != 32:
                    raise ValueError("pending audit digest must contain 32 bytes")
                evidence = _owned_trace_evidence(pending.evidence)
                if evidence.packet.task_id != pending.task_id:
                    raise ValueError(
                        "pending audit task does not match its retained evidence"
                    )
                if _evidence_digest(evidence, self.crypto_mode) != pending.evidence_digest:
                    raise ValueError(
                        "pending audit digest does not match its retained evidence"
                    )
                task_pending = pending_audits.setdefault(
                    pending.task_id,
                    {},
                )
                if pending.evidence_digest in task_pending:
                    raise ValueError("checkpoint contains a duplicate pending audit")
                task_pending[pending.evidence_digest] = evidence
        self._tasks: dict[str, _TaskState] = {}
        self._as_verifier = ASVerifier(
            p2=self.p2,
            master_public=self.master_public,
            trace_public=self.trace_public,
            certificate_public=self.trace_certificate_public,
            audit_authorization_secret=self._audit_authorization_secret,
            crypto_mode=self.crypto_mode,
            pending_audits=pending_audits,
            archived_audits=(
                state.archived_audits if state is not None else ()
            ),
        )
        self._trace_gateway = _TraceGateway(
            dkg=self._dkg,
            identity_scalars=self._identity_scalars,
            p2=self.p2,
            master_public=self.master_public,
            trace_public=self.trace_public,
            audit_authorization_public=self._audit_authorization_public,
            crypto_mode=self.crypto_mode,
        )

    def export_state(self) -> SM9RRSState:
        """Export D-KGC shares, task secrets, and read-only RID history."""

        snapshots = dict(self._task_states_to_restore)
        snapshots.update(
            {
                task_id: TaskStateSnapshot(
                    task_id=task_id,
                    task_salt=bytes(task.task_salt),
                    current_rid=task.current_rid,
                    rings=tuple(record.ring for record in task.rings.values()),
                )
                for task_id, task in self._tasks.items()
            }
        )
        return SM9RRSState(
            dkg_state=self._dkg.export_state(),
            tasks=tuple(snapshots[key] for key in sorted(snapshots)),
            audit_authorization_secret=self._audit_authorization_secret,
            finalized_task_ids=tuple(sorted(self._finalized_tasks)),
            pending_audits=self._as_verifier._pending_snapshots(),
            archived_audits=self._as_verifier._archived_snapshots(),
        )

    @property
    def _default_trace_node_identities(self) -> tuple[str, ...]:
        return self._dkg.node_identities[: self.dkg_threshold]

    def client_signer(self, identity: str) -> ClientSigner:
        """Issue a capability that can sign only for one registered client."""

        if identity not in self._client_signers:
            raise ValueError(f"unknown client identity: {identity}")
        return self._client_signers[identity]

    def as_verifier(
        self,
        *,
        expected_update_shape: Sequence[int] | None = None,
    ) -> ASVerifier:
        """Issue the AS verification capability, which has no signing API."""

        shape = (
            None
            if expected_update_shape is None
            else tuple(int(dimension) for dimension in expected_update_shape)
        )
        if shape is not None and (
            not shape or any(dimension < 1 for dimension in shape)
        ):
            raise ValueError("expected_update_shape must contain positive dimensions")
        self._as_verifier._set_expected_update_shape(shape)
        return self._as_verifier

    def auditor_service(
        self,
        node_identities: Sequence[str] | None = None,
    ) -> AuditorService:
        """Issue an Auditor connected to named D-KGC approval endpoints.

        Node names select real one-share issuer capabilities during trusted
        setup.  They are not accepted by the trace operation itself and cannot
        substitute for the per-request Schnorr approvals those issuers create.
        """

        selected = (
            self._default_trace_node_identities
            if node_identities is None
            else tuple(dict.fromkeys(node_identities))
        )
        issuers = tuple(
            _TraceApprovalEndpoint(
                issuer=self._dkg.trace_approval_issuer(identity),
                audit_authorization_public=self._audit_authorization_public,
                p2=self.p2,
                crypto_mode=self.crypto_mode,
            )
            for identity in selected
        )
        return AuditorService(
            gateway=self._trace_gateway,
            trace_issuers=issuers,
        )

    def register_task(
        self,
        task_id: str,
        ring: Sequence[str] | None = None,
    ) -> str:
        """Create ``h_t`` and register/rebuild the fixed task ring.

        Re-registering an unchanged ring is a no-op.  A changed member set gets
        a new RID, accumulator, witnesses, g1/g2 values, and private Delta
        shares while historical ring records remain available for an audit.
        """

        if not task_id:
            raise ValueError("task_id must not be empty")
        if task_id in self._finalized_tasks:
            raise ValueError(f"task is finalized and cannot be reactivated: {task_id}")
        ordered_ring = tuple(self.client_ids if ring is None else map(str, ring))
        if not ordered_ring:
            raise ValueError("task ring must not be empty")
        if len(set(ordered_ring)) != len(ordered_ring):
            raise ValueError("task ring identities must be unique")
        unknown = set(ordered_ring) - set(self.client_ids)
        if unknown:
            raise ValueError(f"unknown task ring identities: {sorted(unknown)}")

        task = self._tasks.get(task_id)
        if task is None:
            snapshot = self._task_states_to_restore.pop(task_id, None)
            task_salt = bytearray(
                snapshot.task_salt
                if snapshot is not None
                else self._task_salt(task_id)
            )
            task = _TaskState(
                task_id=task_id,
                task_salt=task_salt,
                task_point=self._hash_to_task_point(task_id, task_salt),
                current_rid="",
                rings={},
            )
            self._tasks[task_id] = task
            if snapshot is not None:
                for historical_ring in snapshot.rings:
                    self._install_task_ring(task, historical_ring)
                if snapshot.current_rid not in task.rings:
                    raise ValueError("checkpoint current RID is absent from ring history")
                task.current_rid = snapshot.current_rid
                self._activate_task_ring(task, task.rings[snapshot.current_rid])

        rid = ring_identifier(ordered_ring, algorithm=self._digest_algorithm)
        existing = task.rings.get(rid)
        if existing is not None:
            if existing.ring != ordered_ring:
                raise RuntimeError("RID collision between distinct task rings")
            self._activate_task_ring(task, existing)
            return rid

        return self._install_task_ring(task, ordered_ring)

    def _install_task_ring(
        self,
        task: _TaskState,
        ordered_ring: tuple[str, ...],
    ) -> str:
        """Construct one current or archived ring record from D-KGC material."""

        rid = ring_identifier(ordered_ring, algorithm=self._digest_algorithm)
        existing = task.rings.get(rid)
        if existing is not None:
            if existing.ring != ordered_ring:
                raise RuntimeError("RID collision between distinct task rings")
            return rid

        scalars = {identity: self._identity_scalars[identity] for identity in ordered_ring}
        accumulator, witnesses = self._dkg.build_task_ring(rid, scalars)
        g1 = _gt_multiply(
            self._dkg.master_pairing,
            _pair(accumulator, self.p2, self.crypto_mode),
            self.crypto_mode,
        )

        record = _RingRecord(
            ring=ordered_ring,
            rid=rid,
            accumulator=accumulator,
            witnesses=witnesses,
            g1=g1,
        )
        self._validate_ring_material(record)
        task.rings[rid] = record
        self._activate_task_ring(task, record)
        return rid

    def _activate_task_ring(self, task: _TaskState, record: _RingRecord) -> None:
        """Push role-specific copies without giving any role a root back-reference."""

        public_record = _PublicRingRecord(
            ring=record.ring,
            rid=record.rid,
            accumulator=record.accumulator,
            g1=record.g1,
        )
        # Provision the remaining clients first and make AS accept the new RID
        # only after all signing and tracing material is ready.
        for identity, signer in self._client_signers.items():
            if identity in record.ring:
                signer._install_task_material(
                    task.task_id,
                    task_point=task.task_point,
                    rid=record.rid,
                    accumulator=record.accumulator,
                    witness=record.witnesses[identity],
                    g1=record.g1,
                )
            else:
                signer._drop_task_material(task.task_id)
        self._trace_gateway.install_task_ring(
            task.task_id,
            task.task_point,
            public_record,
            make_current=True,
        )
        self._as_verifier._install_task_ring(
            task.task_id,
            task.task_point,
            public_record,
            make_current=True,
        )
        task.current_rid = record.rid

    def update_task_ring(self, task_id: str, ring: Sequence[str]) -> str:
        return self.register_task(task_id, ring)

    def finalize_task(self, task_id: str) -> None:
        """Erase task-local linking material after its audit lifecycle ends.

        Finalization is fail-closed: every evidence item created by
        :meth:`build_trace_evidence` is tracked internally and blocks
        destruction until :meth:`archive_trace_result` verifies and explicitly
        archives the matching threshold-certified result. Caller-supplied
        booleans are deliberately not accepted as proof of audit completion.

        This overwrites the live mutable copy of ``kappa_t``, drops ``h_t`` and
        all cached ``Tag_pi`` values, removes every task ring record, and erases
        D-KGC Delta shares no longer referenced by another task. Python cannot
        revoke immutable copies in checkpoints exported before this call; their
        storage owner must erase those obsolete checkpoints separately.
        """

        if not task_id:
            raise ValueError("task_id must not be empty")
        if task_id in self._finalized_tasks:
            return
        pending = self._as_verifier.pending_audit_digests(task_id)
        if pending:
            raise RuntimeError(
                f"task has {len(pending)} pending audit(s) and cannot be finalized"
            )

        task = self._tasks.pop(task_id, None)
        snapshot = self._task_states_to_restore.pop(task_id, None)
        if task is None and snapshot is None:
            raise ValueError(f"task is not registered: {task_id}")

        if task is not None:
            released_rids = set(task.rings)
            retained_rids = {
                rid
                for other_task in self._tasks.values()
                for rid in other_task.rings
            }
            for rid in released_rids - retained_rids:
                self._dkg.discard_task_ring(rid)
            task.task_salt[:] = b"\x00" * len(task.task_salt)
            task.task_point = None
            task.rings.clear()
            task.current_rid = ""

        for signer in self._client_signers.values():
            signer._drop_task_material(task_id)
        self._as_verifier._drop_task(task_id)
        self._trace_gateway.drop_task(task_id)
        self._finalized_tasks.add(task_id)

    def pending_audit_digests(self, task_id: str) -> tuple[str, ...]:
        """Return the internally tracked, not-yet-archived evidence digests."""

        return self._as_verifier.pending_audit_digests(task_id)

    def pending_audit_evidence(self, task_id: str) -> tuple[TraceEvidence, ...]:
        """Return detached evidence copies that can be retried after restart."""

        return self._as_verifier.pending_audit_evidence(task_id)

    def archived_audit_records(
        self,
        task_id: str,
    ) -> tuple[ArchivedAuditRecord, ...]:
        """Return retained evidence and certified results for external audit."""

        return self._as_verifier.archived_audit_records(task_id)

    def is_task_finalized(self, task_id: str) -> bool:
        """Return whether task-local cryptographic material was destroyed."""

        return task_id in self._finalized_tasks

    def current_ring_id(self, task_id: str) -> str:
        return self._require_task(task_id).current_rid

    def registered_ring(self, task_id: str, rid: str | None = None) -> tuple[str, ...]:
        task = self._require_task(task_id)
        record = task.rings.get(task.current_rid if rid is None else rid)
        if record is None:
            raise ValueError("RID is not registered for this task")
        return record.ring

    def create_packet(
        self,
        identity: str,
        update: np.ndarray,
        *,
        round_id: int,
        task_id: str = "mnist",
        update_digest: str | None = None,
    ) -> RRSPacket:
        if task_id not in self._tasks:
            self.register_task(task_id)
        signer = self.client_signer(identity)
        unsigned = signer.build_unsigned_packet(
            update,
            round_id=round_id,
            task_id=task_id,
            update_digest=update_digest,
        )
        return signer.sign_packet(unsigned)

    def build_unsigned_packet(
        self,
        identity: str,
        update: np.ndarray,
        *,
        round_id: int,
        task_id: str = "mnist",
        update_digest: str | None = None,
    ) -> RRSPacket:
        """Compatibility facade over the isolated client capability."""

        if task_id not in self._tasks:
            self.register_task(task_id)
        return self.client_signer(identity).build_unsigned_packet(
            update,
            round_id=round_id,
            task_id=task_id,
            update_digest=update_digest,
        )

    def precompute_task_material(
        self,
        task_id: str,
        identities: Iterable[str] | None = None,
    ) -> None:
        """Precompute the paper's reusable ``g2,pi`` and ``Tag_pi`` values."""

        if task_id not in self._tasks:
            self.register_task(task_id)
        task = self._require_task(task_id)
        record = task.rings[task.current_rid]
        selected = record.ring if identities is None else tuple(map(str, identities))
        for identity in selected:
            if identity not in record.ring:
                raise ValueError("identity is not in the current task ring")
            self.client_signer(identity)._precompute_task_material(task_id)

    def sign_packet(self, identity: str, unsigned: RRSPacket) -> RRSPacket:
        """Compatibility facade over the isolated client capability."""

        self._require_task(unsigned.task_id)
        return self.client_signer(identity).sign_packet(unsigned)

    def verify_packet(
        self,
        packet: RRSPacket,
        update: np.ndarray,
        *,
        expected_task_id: str | None = None,
        expected_round_id: int | None = None,
    ) -> bool:
        """AS verification: both Equation (1) and Equation (2) must hold."""

        return self._as_verifier._verify_packet(
            packet,
            update,
            expected_task_id=expected_task_id,
            expected_round_id=expected_round_id,
        )

    def verify_ring_equation(
        self,
        packet: RRSPacket,
        update: np.ndarray,
        *,
        allow_historical_ring: bool = True,
    ) -> bool:
        """Auditor verification of Equation (1), which does not require ``h_t``."""

        return self._as_verifier.verify_ring_equation(
            packet,
            update,
            allow_historical_ring=allow_historical_ring,
        )

    def build_trace_evidence(
        self,
        packet: RRSPacket,
        update: np.ndarray,
    ) -> TraceEvidence:
        return self._as_verifier.build_trace_evidence(packet, update)

    def trace(
        self,
        evidence: TraceEvidence,
    ) -> TraceResult:
        """Run the default Auditor capability over a complete trace request."""

        return self.auditor_service().trace(evidence)

    def _trace_with_authorization(
        self,
        evidence: TraceEvidence,
        authorization: TraceAuthorization,
    ) -> TraceResult:
        return self._trace_gateway.trace(evidence, authorization)

    def verify_trace_result(
        self,
        evidence: TraceEvidence,
        result: TraceResult,
    ) -> bool:
        return self._as_verifier.verify_trace_result(evidence, result)

    def archive_trace_result(
        self,
        evidence: TraceEvidence,
        result: TraceResult,
    ) -> bool:
        return self._as_verifier.archive_trace_result(evidence, result)

    def _verify_tag_equation(self, packet: RRSPacket) -> bool:
        """Verify Equation (2), including for an archived task-ring RID."""

        task = self._tasks.get(packet.task_id)
        return task is not None and _verify_tag_equation(
            packet,
            task.task_point,
            self.crypto_mode,
        )

    def evidence_digest(self, evidence: TraceEvidence) -> str:
        return _evidence_digest(evidence, self.crypto_mode)

    def digest_update(self, update: np.ndarray) -> str:
        return digest_update(update, algorithm=self._digest_algorithm)

    @property
    def _digest_algorithm(self) -> str:
        return "sha256" if self.crypto_mode == "simulated" else "sm3"

    def _validate_ring_material(self, record: _RingRecord) -> None:
        for identity in record.ring:
            identity_value = self._identity_scalars[identity]
            left = _pair(
                record.witnesses[identity],
                _g2_add(
                    _g2_multiply(self.p2, identity_value, self.crypto_mode),
                    self.trace_public,
                    self.crypto_mode,
                ),
                self.crypto_mode,
            )
            right = _pair(record.accumulator, self.p2, self.crypto_mode)
            if not _gt_equal(left, right, self.crypto_mode):
                raise RuntimeError("task-ring membership witness validation failed")

    def _hash_to_task_point(
        self,
        task_id: str,
        task_salt: bytes | bytearray,
    ) -> Any:
        transcript = encode_fields(
            b"SM9-RRS-FL/HG2/v2",
            task_id.encode("utf-8"),
            task_salt,
        )
        scalar = protocol_hash_to_scalar(
            1,
            transcript,
            crypto_mode=self.crypto_mode,
        )
        if self.crypto_mode == "simulated":
            return scalar
        return sm9_backend.g2_mul(self.p2, scalar)

    def _task_salt(self, task_id: str) -> bytes:
        if self.crypto_mode == "sm9":
            return os.urandom(32)
        # Deterministic only in the explicitly non-secure simulation mode.
        return hashlib.sha256(f"task-salt:{self._seed}:{task_id}".encode()).digest()

    def _require_task(self, task_id: str) -> _TaskState:
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"task is not registered: {task_id}")
        return task

def _digest_algorithm(crypto_mode: str) -> str:
    if crypto_mode == "simulated":
        return "sha256"
    if crypto_mode == "sm9":
        return "sm3"
    raise ValueError("unsupported crypto mode")


def _message_bytes(packet: RRSPacket, crypto_mode: str) -> bytes:
    return encode_fields(
        packet.task_id.encode("utf-8"),
        encode_scalar(packet.round_id),
        bytes.fromhex(packet.update_digest),
        bytes.fromhex(packet.rid),
        encode_group_element(packet.accumulator, crypto_mode=crypto_mode),
    )


def _trace_message(
    evidence_digest: str,
    identity: str,
    task_tag: Any,
    rid: str,
    crypto_mode: str,
) -> bytes:
    return encode_fields(
        bytes.fromhex(evidence_digest),
        identity.encode("utf-8"),
        encode_group_element(task_tag, crypto_mode=crypto_mode),
        bytes.fromhex(rid),
        b"trace",
    )


def _evidence_digest(evidence: TraceEvidence, crypto_mode: str) -> str:
    packet = evidence.packet
    signature = packet.signature
    if signature is None or packet.task_tag is None or packet.tag_commitment is None:
        raise ValueError("trace evidence packet is incomplete")
    payload = encode_fields(
        packet.task_id.encode("utf-8"),
        encode_scalar(packet.round_id),
        encode_update(evidence.model_update),
        bytes.fromhex(packet.rid),
        encode_group_element(packet.accumulator, crypto_mode=crypto_mode),
        encode_group_element(packet.task_tag, crypto_mode=crypto_mode),
        encode_group_element(packet.tag_commitment, crypto_mode=crypto_mode),
        encode_scalar(signature.c),
        encode_group_element(signature.A, crypto_mode=crypto_mode),
        encode_group_element(signature.B, crypto_mode=crypto_mode),
        encode_group_element(signature.C, crypto_mode=crypto_mode),
    )
    return domain_hash_hex(
        b"SM9-RRS-FL/H5/evidence/v2",
        payload,
        algorithm=_digest_algorithm(crypto_mode),
    )


def _owned_trace_evidence(evidence: TraceEvidence) -> TraceEvidence:
    """Copy an evidence update and make the archived representation immutable."""

    if not isinstance(evidence, TraceEvidence):
        raise TypeError("archived audit evidence has an invalid type")
    retained_update = np.array(
        evidence.model_update,
        dtype=np.float32,
        order="C",
        copy=True,
    )
    retained_update.setflags(write=False)
    return TraceEvidence(
        packet=evidence.packet,
        model_update=retained_update,
        audit_authorization=evidence.audit_authorization,
    )


def _copy_archived_record(record: ArchivedAuditRecord) -> ArchivedAuditRecord:
    """Return a detached archive view so callers cannot mutate stored arrays."""

    if not isinstance(record, ArchivedAuditRecord):
        raise TypeError("archived audit record has an invalid type")
    return ArchivedAuditRecord(
        task_id=record.task_id,
        evidence=_owned_trace_evidence(record.evidence),
        result=record.result,
    )


def _verify_ring_equation(
    packet: RRSPacket,
    update: np.ndarray,
    *,
    task: _VerificationTaskState,
    p2: Any,
    master_public: Any,
    trace_public: Any,
    crypto_mode: str,
    allow_historical_ring: bool,
) -> bool:
    try:
        if packet.protocol_version != PROTOCOL_VERSION:
            return False
        if packet.crypto_mode != crypto_mode:
            return False
        if not allow_historical_ring and packet.rid != task.current_rid:
            return False
        record = task.rings.get(packet.rid)
        if record is None:
            return False
        if not _group_equal(packet.accumulator, record.accumulator, crypto_mode):
            return False
        if packet.update_digest != digest_update(
            update,
            algorithm=_digest_algorithm(crypto_mode),
        ):
            return False
        if not _valid_signature_elements(packet, crypto_mode):
            return False
        signature = packet.signature
        assert signature is not None
        assert packet.task_tag is not None and packet.tag_commitment is not None
        omega_hat = _gt_multiply(
            _gt_multiply(
                _pair(
                    signature.A,
                    _g2_add(trace_public, signature.C, crypto_mode),
                    crypto_mode,
                ),
                _pair(
                    signature.B,
                    _g2_add(master_public, signature.C, crypto_mode),
                    crypto_mode,
                ),
                crypto_mode,
            ),
            _gt_power(record.g1, signature.c, crypto_mode),
            crypto_mode,
        )
        return signature.c == challenge_scalar(
            packet.rid,
            _message_bytes(packet, crypto_mode),
            packet.task_tag,
            packet.tag_commitment,
            omega_hat,
            crypto_mode=crypto_mode,
        )
    except (AssertionError, KeyError, OverflowError, TypeError, ValueError):
        return False


def _verify_tag_equation(
    packet: RRSPacket,
    task_point: Any,
    crypto_mode: str,
) -> bool:
    try:
        signature = packet.signature
        assert signature is not None
        assert packet.task_tag is not None and packet.tag_commitment is not None
        right = _gt_multiply(
            _pair(signature.B, task_point, crypto_mode),
            _gt_power(packet.task_tag, signature.c, crypto_mode),
            crypto_mode,
        )
        return _gt_equal(packet.tag_commitment, right, crypto_mode)
    except (AssertionError, KeyError, OverflowError, TypeError, ValueError):
        return False


def _valid_signature_elements(packet: RRSPacket, crypto_mode: str) -> bool:
    signature = packet.signature
    if signature is None or packet.task_tag is None or packet.tag_commitment is None:
        return False
    if not 1 <= signature.c < SCALAR_MODULUS:
        return False
    if crypto_mode == "simulated":
        return all(
            isinstance(value, int) and 0 < value < SCALAR_MODULUS
            for value in (
                signature.A,
                signature.B,
                signature.C,
                packet.task_tag,
                packet.tag_commitment,
            )
        )
    return (
        crypto_mode == "sm9"
        and sm9_backend.g1_validate(signature.A)
        and sm9_backend.g1_validate(signature.B)
        and sm9_backend.g2_validate(signature.C)
        and sm9_backend.gt_validate(packet.task_tag)
        and sm9_backend.gt_validate(packet.tag_commitment)
        and not sm9_backend.gt_equal(packet.task_tag, sm9_backend.gt_one())
        and not sm9_backend.gt_equal(packet.tag_commitment, sm9_backend.gt_one())
    )


def _pair(paper_g1: Any, paper_g2: Any, crypto_mode: str) -> Any:
    if crypto_mode == "simulated":
        return int(paper_g1) * int(paper_g2) % SCALAR_MODULUS
    return sm9_backend.pair(paper_g1, paper_g2)


def _g1_multiply(value: Any, scalar: int, crypto_mode: str) -> Any:
    if crypto_mode == "simulated":
        return int(value) * scalar % SCALAR_MODULUS
    return sm9_backend.g1_mul(value, scalar % SCALAR_MODULUS)


def _g1_add(left: Any, right: Any, crypto_mode: str) -> Any:
    if crypto_mode == "simulated":
        return (int(left) + int(right)) % SCALAR_MODULUS
    return sm9_backend.g1_add(left, right)


def _g2_multiply(value: Any, scalar: int, crypto_mode: str) -> Any:
    if crypto_mode == "simulated":
        return int(value) * scalar % SCALAR_MODULUS
    return sm9_backend.g2_mul(value, scalar % SCALAR_MODULUS)


def _g2_add(left: Any, right: Any, crypto_mode: str) -> Any:
    if crypto_mode == "simulated":
        return (int(left) + int(right)) % SCALAR_MODULUS
    return sm9_backend.g2_add(left, right)


def _gt_power(value: Any, scalar: int, crypto_mode: str) -> Any:
    if crypto_mode == "simulated":
        return int(value) * scalar % SCALAR_MODULUS
    return sm9_backend.gt_pow(value, scalar % SCALAR_MODULUS)


def _gt_multiply(left: Any, right: Any, crypto_mode: str) -> Any:
    if crypto_mode == "simulated":
        return (int(left) + int(right)) % SCALAR_MODULUS
    return sm9_backend.gt_mul(left, right)


def _gt_equal(left: Any, right: Any, crypto_mode: str) -> bool:
    try:
        if crypto_mode == "simulated":
            return int(left) % SCALAR_MODULUS == int(right) % SCALAR_MODULUS
        return sm9_backend.gt_equal(left, right)
    except (AssertionError, TypeError, ValueError):
        return False


def _group_equal(left: Any, right: Any, crypto_mode: str) -> bool:
    try:
        if crypto_mode == "simulated":
            return int(left) % SCALAR_MODULUS == int(right) % SCALAR_MODULUS
        return bytes(left) == bytes(right)
    except (AssertionError, TypeError, ValueError):
        return False


def identity_scalar(
    identity: str,
    *,
    hid: int = HID_SIGN,
    crypto_mode: str = "sm9",
) -> int:
    """SM9 ``H1(ID||hid,N)`` used by both KeyExtract and the accumulator."""

    if not 0 <= hid <= 255:
        raise ValueError("SM9 hid must fit in one octet")
    return protocol_hash_to_scalar(
        1,
        identity.encode("utf-8") + bytes((hid,)),
        crypto_mode=crypto_mode,
    )


def ring_identifier(identities: Sequence[str], *, algorithm: str = "sm3") -> str:
    encoded_ring = encode_fields(*(identity.encode("utf-8") for identity in identities))
    return domain_hash_hex(
        b"SM9-RRS-FL/H3/RID/v2",
        encoded_ring,
        algorithm=algorithm,
    )


def challenge_scalar(
    rid: str,
    message: bytes,
    task_tag: Any,
    tag_commitment: Any,
    omega: Any,
    *,
    crypto_mode: str,
) -> int:
    """``H2(RID||M||Tag||Rtag||Omega,N)`` with unambiguous encoding."""

    transcript = encode_fields(
        bytes.fromhex(rid),
        message,
        encode_group_element(task_tag, crypto_mode=crypto_mode),
        encode_group_element(tag_commitment, crypto_mode=crypto_mode),
        encode_group_element(omega, crypto_mode=crypto_mode),
    )
    if crypto_mode == "simulated":
        return int.from_bytes(hashlib.sha256(transcript).digest(), "big") % (
            SCALAR_MODULUS - 1
        ) + 1
    return sm9_backend.hash_to_scalar(2, transcript)


def _audit_authorization_message(
    task_id: str,
    rid: str,
    evidence_digest: str,
) -> bytes:
    rid_bytes = bytes.fromhex(rid)
    digest_bytes = bytes.fromhex(evidence_digest)
    if not task_id or len(rid_bytes) != 32 or len(digest_bytes) != 32:
        raise ValueError("invalid AS audit-authorization transcript")
    return encode_fields(
        b"SM9-RRS-FL/AS-audit-authorization/v2",
        task_id.encode("utf-8"),
        rid_bytes,
        digest_bytes,
    )


def _audit_authorization_challenge(
    *,
    commitment: Any,
    task_id: str,
    rid: str,
    evidence_digest: str,
    crypto_mode: str,
) -> int:
    transcript = encode_fields(
        _audit_authorization_message(task_id, rid, evidence_digest),
        encode_group_element(commitment, crypto_mode=crypto_mode),
    )
    return protocol_hash_to_scalar(2, transcript, crypto_mode=crypto_mode)


def _sign_audit_authorization(
    *,
    secret: int,
    p2: Any,
    task_id: str,
    rid: str,
    evidence_digest: str,
    crypto_mode: str,
) -> AuditAuthorization:
    if not 1 <= secret < SCALAR_MODULUS:
        raise ValueError("invalid AS audit-authorization signing key")
    rng = SystemRandom()
    while True:
        nonce = rng.randrange(1, SCALAR_MODULUS)
        commitment = _g2_multiply(p2, nonce, crypto_mode)
        challenge = _audit_authorization_challenge(
            commitment=commitment,
            task_id=task_id,
            rid=rid,
            evidence_digest=evidence_digest,
            crypto_mode=crypto_mode,
        )
        response = (nonce + challenge * secret) % SCALAR_MODULUS
        if response != 0:
            return AuditAuthorization(
                commitment=commitment,
                response=response,
            )


def _verify_audit_authorization(
    authorization: AuditAuthorization | None,
    *,
    public: Any,
    p2: Any,
    task_id: str,
    rid: str,
    evidence_digest: str,
    crypto_mode: str,
) -> bool:
    try:
        if not isinstance(authorization, AuditAuthorization):
            return False
        if not 1 <= authorization.response < SCALAR_MODULUS:
            return False
        challenge = _audit_authorization_challenge(
            commitment=authorization.commitment,
            task_id=task_id,
            rid=rid,
            evidence_digest=evidence_digest,
            crypto_mode=crypto_mode,
        )
        if crypto_mode == "simulated":
            if not all(
                isinstance(value, int) and 0 < value < SCALAR_MODULUS
                for value in (authorization.commitment, public, p2)
            ):
                return False
            return authorization.response == (
                int(authorization.commitment) + challenge * int(public)
            ) % SCALAR_MODULUS
        if crypto_mode != "sm9":
            return False
        if not (
            sm9_backend.g2_validate(p2)
            and sm9_backend.g2_validate(public)
            and sm9_backend.g2_validate(authorization.commitment)
        ):
            return False
        left = sm9_backend.g2_mul(p2, authorization.response)
        right = sm9_backend.g2_add(
            authorization.commitment,
            sm9_backend.g2_mul(public, challenge),
        )
        return left == right
    except (OverflowError, TypeError, ValueError):
        return False


def protocol_hash_to_scalar(
    prefix: int,
    transcript: bytes,
    *,
    crypto_mode: str,
) -> int:
    """Apply direct H_v to a canonical transcript without pre-hashing it."""

    if crypto_mode == "sm9":
        return sm9_backend.hash_to_scalar(prefix, transcript)
    if crypto_mode != "simulated":
        raise ValueError("unsupported crypto mode")
    domain = bytes((prefix,)) + transcript
    return int.from_bytes(hashlib.sha256(domain).digest(), "big") % (
        SCALAR_MODULUS - 1
    ) + 1


def encode_fields(*fields: bytes) -> bytes:
    """Length-prefix protocol fields so ``||`` has one canonical encoding."""

    encoded = bytearray()
    for field in fields:
        encoded.extend(len(field).to_bytes(8, "big"))
        encoded.extend(field)
    return bytes(encoded)


def encode_scalar(value: int) -> bytes:
    if value < 0:
        raise ValueError("protocol integers must be non-negative")
    return int(value).to_bytes(32, "big")


def scalar_inverse(value: int) -> int:
    value %= SCALAR_MODULUS
    if value == 0:
        raise ZeroDivisionError("zero has no inverse in Z_N")
    return pow(value, SCALAR_MODULUS - 2, SCALAR_MODULUS)


def encode_group_element(value: Any, *, crypto_mode: str) -> bytes:
    if crypto_mode == "simulated":
        return encode_scalar(int(value) % SCALAR_MODULUS)
    if crypto_mode != "sm9" or not isinstance(value, bytes):
        raise TypeError("unsupported group element")
    if len(value) not in {
        sm9_backend.G1_BYTES,
        sm9_backend.G2_BYTES,
        sm9_backend.GT_BYTES,
    }:
        raise ValueError("invalid canonical SM9 group-element length")
    return value


def element_digest(value: Any, *, crypto_mode: str) -> str:
    algorithm = "sha256" if crypto_mode == "simulated" else "sm3"
    return domain_hash_hex(
        b"SM9-RRS-FL/task-tag-key/v2",
        encode_group_element(value, crypto_mode=crypto_mode),
        algorithm=algorithm,
    )


def encode_update(update: np.ndarray) -> bytes:
    """Canonical shape-bound little-endian float32 representation of ``G``."""

    array = np.asarray(update)
    if array.dtype.kind != "f" or array.dtype.itemsize != 4:
        raise TypeError("model update wire format must be float32")
    if not array.flags.c_contiguous:
        raise ValueError("model update wire format must be C-contiguous")
    array = array.astype("<f4", copy=False)
    shape = encode_fields(
        encode_scalar(array.ndim),
        *(encode_scalar(dimension) for dimension in array.shape),
    )
    return encode_fields(shape, memoryview(array).cast("B").tobytes())


def digest_update(update: np.ndarray, *, algorithm: str = "sm3") -> str:
    return domain_hash_hex(
        b"SM9-RRS-FL/H3/update/v2",
        encode_update(update),
        algorithm=algorithm,
    )


def domain_hash_hex(domain: bytes, data: bytes, *, algorithm: str) -> str:
    return hash_hex(encode_fields(domain, data), algorithm=algorithm)


def hash_hex(data: bytes | memoryview, *, algorithm: str) -> str:
    if algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    if algorithm != "sm3":
        raise ValueError("algorithm must be 'sm3' or 'sha256'")
    return sm3_hex_bytes(data)


def sm3_hex_text(text: str) -> str:
    return sm3_hex_bytes(text.encode("utf-8"))


def sm3_hex_bytes(data: bytes | memoryview) -> str:
    if _native_sm3_hexdigest is not None:
        return _native_sm3_hexdigest(data)
    if "sm3" in hashlib.algorithms_available:
        return hashlib.new("sm3", data).hexdigest()
    return sm3.sm3_hash(list(data))


def sm3_backend_name() -> str:
    if _native_sm3_hexdigest is not None:
        return "native-extension"
    if "sm3" in hashlib.algorithms_available:
        return "hashlib-openssl"
    return "python-fallback"


def rrs_backend_name() -> str:
    """Return the active standard-SM9 group backend status."""

    return sm9_backend.backend_name()


__all__ = [
    "ArchivedAuditRecord",
    "AuditAuthorization",
    "ASVerifier",
    "AuditorService",
    "ClientSigner",
    "PendingAuditSnapshot",
    "PROTOCOL_VERSION",
    "RRSPacket",
    "RingSignature",
    "SM9RRSContext",
    "SM9RRSState",
    "TaskStateSnapshot",
    "ThresholdNotMetError",
    "TraceEvidence",
    "TraceResult",
    "digest_update",
    "rrs_backend_name",
    "sm3_backend_name",
    "sm3_hex_bytes",
]
