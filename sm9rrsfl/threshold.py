"""D-KGC share-domain model for the SM9-RRS-FL v2 protocol.

The module keeps the SM9 master key, the independent accumulator tracing key,
and the trace-certificate key as ``(t,n)`` Shamir shares.  Public keys and
partial group results are combined with Lagrange coefficients; full secrets,
client private-key scalars, and tracing coefficients are never returned.

The paper's Paillier cross terms are represented by fresh Shamir resharing in
one experiment process.  Multiplication and blinded inversion stay in the
share domain: only the uniformly masked product allowed by the Word protocol
is opened.  Logical D-KGC nodes Schnorr-sign every task/RID/evidence/session
trace request, so integer node labels alone never authorize tracing.  This
models the protocol and authenticated-node algebra, but Python object privacy
is not a process-isolation boundary and the local resharing is not a replacement
for Paillier channels between separately administered D-KGC hosts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import secrets
from random import SystemRandom
from typing import Any, Iterable, Sequence

from gmssl import sm3

from . import sm9_backend


SCALAR_MODULUS = sm9_backend.SM9_ORDER


class ThresholdNotMetError(ValueError):
    """Raised when fewer than ``t`` distinct D-KGC nodes authorize an action."""


@dataclass(frozen=True)
class ScalarShare:
    """One Shamir share held by ``node_id`` at its hashed KGC coordinate."""

    node_id: int
    value: int


@dataclass(frozen=True)
class ThresholdCertificate:
    """Threshold Schnorr certificate ``(R,z)`` under the independent TPK."""

    commitment: Any
    response: int


class ThresholdCertificateVerifier:
    """Public-only verifier for the final threshold trace certificate."""

    __slots__ = ("certificate_public", "crypto_mode", "p2")

    def __init__(self, *, certificate_public: Any, p2: Any, crypto_mode: str) -> None:
        if crypto_mode not in {"sm9", "simulated"}:
            raise ValueError("invalid threshold-certificate crypto mode")
        self.certificate_public = certificate_public
        self.p2 = p2
        self.crypto_mode = crypto_mode

    def verify(self, message: bytes, certificate: ThresholdCertificate) -> bool:
        try:
            if not isinstance(certificate, ThresholdCertificate):
                return False
            if not 1 <= certificate.response < SCALAR_MODULUS:
                return False
            if self.crypto_mode == "simulated":
                if not (
                    isinstance(certificate.commitment, int)
                    and 0 < certificate.commitment < SCALAR_MODULUS
                ):
                    return False
            elif not sm9_backend.g2_validate(certificate.commitment):
                return False
            challenge = _certificate_challenge(
                message,
                certificate.commitment,
                self.crypto_mode,
            )
            if self.crypto_mode == "simulated":
                expected = (
                    int(certificate.commitment)
                    + challenge * int(self.certificate_public)
                ) % SCALAR_MODULUS
                return certificate.response == expected
            left = sm9_backend.g2_mul(self.p2, certificate.response)
            right = sm9_backend.g2_add(
                certificate.commitment,
                sm9_backend.g2_mul(self.certificate_public, challenge),
            )
            return left == right
        except (OverflowError, TypeError, ValueError):
            return False


@dataclass(frozen=True)
class TraceAuthorizationRequest:
    """One replay-protected D-KGC trace-authorization transcript.

    The evidence digest already commits to the complete trigger-round packet
    and model update.  ``task_id`` and ``rid`` are repeated here deliberately:
    an approving node can enforce the routing context without decoding the
    evidence, while ``session_id`` makes an otherwise identical request fresh.
    """

    task_id: str
    rid: str
    evidence_digest: str
    session_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("trace authorization task_id must not be empty")
        if len(self.task_id.encode("utf-8")) > 1024:
            raise ValueError("trace authorization task_id is too long")
        _require_canonical_hex(self.rid, 32, "RID")
        _require_canonical_hex(self.evidence_digest, 32, "evidence digest")
        _require_canonical_hex(self.session_id, 32, "trace session identifier")

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        rid: str,
        evidence_digest: str,
    ) -> "TraceAuthorizationRequest":
        return cls(
            task_id=task_id,
            rid=rid,
            evidence_digest=evidence_digest,
            session_id=secrets.token_hex(32),
        )

    def canonical_bytes(self) -> bytes:
        return _encode_fields(
            b"SM9-RRS-FL/trace-authorization-request/v2",
            self.task_id.encode("utf-8"),
            bytes.fromhex(self.rid),
            bytes.fromhex(self.evidence_digest),
            bytes.fromhex(self.session_id),
        )


@dataclass(frozen=True)
class TraceApproval:
    """One node's Schnorr approval, signed with its certificate-key share."""

    node_identity: str
    request_digest: str
    commitment: Any
    response: int


@dataclass(frozen=True)
class TraceAuthorization:
    """A request plus independently verifiable approvals from D-KGC nodes."""

    request: TraceAuthorizationRequest
    approvals: tuple[TraceApproval, ...]


@dataclass
class _TraceSessionRecord:
    request_digest: str
    task_id: str
    rid: str
    evidence_digest: str
    quorum: tuple[int, ...]
    task_point_encoding: bytes
    signature_a_encoding: bytes
    signature_b_encoding: bytes
    evidence_task_tag_encoding: bytes
    phase: str = "authorized"
    matched_identity: str | None = None
    matched_identity_scalar: int | None = None
    reconstructed_tag_encoding: bytes | None = None


class _AuthorizedTraceSession:
    """Opaque, short-lived handle created only after approval verification."""

    __slots__ = ("__token",)

    def __init__(self, token: bytes) -> None:
        self.__token = token


class TraceApprovalIssuer:
    """Capability held by one logical D-KGC node.

    It contains exactly one certificate-key share and no coordinator reference,
    SM9 master shares, trace shares, ring deltas, or client material.  In a
    deployment this object corresponds to an authenticated node endpoint.
    """

    __slots__ = (
        "__crypto_mode",
        "__node_identity",
        "__p2",
        "__rng",
        "__share",
    )

    def __init__(
        self,
        *,
        node_identity: str,
        share: ScalarShare,
        p2: Any,
        crypto_mode: str,
        simulated_seed: int | None = None,
    ) -> None:
        self.__node_identity = node_identity
        self.__share = share
        self.__p2 = p2
        self.__crypto_mode = crypto_mode
        self.__rng: random.Random | SystemRandom = (
            random.Random(simulated_seed)
            if crypto_mode == "simulated"
            else SystemRandom()
        )

    @property
    def node_identity(self) -> str:
        return self.__node_identity

    def approve(self, request: TraceAuthorizationRequest) -> TraceApproval:
        """Sign this exact task/RID/evidence/session request as one node."""

        if not isinstance(request, TraceAuthorizationRequest):
            raise TypeError("trace approval requires TraceAuthorizationRequest")
        request_digest = _trace_request_digest(request, self.__crypto_mode)
        while True:
            nonce = self.__rng.randrange(1, SCALAR_MODULUS)
            commitment = (
                nonce
                if self.__crypto_mode == "simulated"
                else sm9_backend.g2_mul(self.__p2, nonce)
            )
            challenge = _trace_approval_challenge(
                request,
                self.__node_identity,
                commitment,
                self.__crypto_mode,
            )
            response = (
                nonce + challenge * self.__share.value
            ) % SCALAR_MODULUS
            if response != 0:
                return TraceApproval(
                    node_identity=self.__node_identity,
                    request_digest=request_digest,
                    commitment=commitment,
                    response=response,
                )


@dataclass(frozen=True)
class DistributedKGCState:
    """Trusted experiment checkpoint containing the per-node secret shares."""

    threshold: int
    node_count: int
    max_accumulator_size: int
    node_identities: tuple[str, ...]
    node_coordinates: tuple[int, ...]
    master_shares: tuple[ScalarShare, ...]
    trace_shares: tuple[ScalarShare, ...]
    certificate_shares: tuple[ScalarShare, ...]
    master_commitments: tuple[Any, ...]
    trace_commitments: tuple[Any, ...]
    certificate_commitments: tuple[Any, ...]
    # A durable coordinator must persist this replay ledger atomically.  The
    # local experiment checkpoint carries it so a normal restart cannot reuse
    # an already accepted authorization transcript.
    consumed_trace_authorizations: tuple[str, ...] = ()


class DistributedKGC:
    """Single-process coordinator for the paper's D-KGC share-domain protocol.

    ``crypto_mode="sm9"`` uses the GmSSL national-standard SM9 groups.
    ``"simulated"`` keeps the same equations while representing each group
    element by its generator scalar/exponent; it is only a fast test mode.
    """

    def __init__(
        self,
        *,
        threshold: int,
        node_count: int,
        crypto_mode: str,
        max_accumulator_size: int = 0,
        node_identities: Sequence[str] | None = None,
        seed: int = 0,
        state: DistributedKGCState | None = None,
    ) -> None:
        if not 1 <= threshold <= node_count:
            raise ValueError("threshold must satisfy 1 <= threshold <= node_count")
        if crypto_mode not in {"sm9", "simulated"}:
            raise ValueError("crypto_mode must be 'sm9' or 'simulated'")
        if max_accumulator_size < 0:
            raise ValueError("max_accumulator_size must not be negative")
        if crypto_mode == "sm9":
            sm9_backend.require_available()
        self.threshold = threshold
        self.node_count = node_count
        self.max_accumulator_size = max_accumulator_size
        self.crypto_mode = crypto_mode
        self._rng: random.Random | SystemRandom = (
            random.Random(seed) if crypto_mode == "simulated" else SystemRandom()
        )
        self.p1 = sm9_backend.g1_generator() if crypto_mode == "sm9" else 1
        self.p2 = sm9_backend.g2_generator() if crypto_mode == "sm9" else 1
        identities = tuple(
            f"KGC-{node_id}" for node_id in range(1, node_count + 1)
        ) if node_identities is None else tuple(map(str, node_identities))
        if len(identities) != node_count or len(set(identities)) != node_count:
            raise ValueError("D-KGC identities must be unique and match node_count")
        self.node_identities = identities
        self.node_coordinates = tuple(
            self._hash_node_coordinate(identity) for identity in identities
        )
        if len(set(self.node_coordinates)) != node_count:
            raise RuntimeError("H1 collision between D-KGC identity coordinates")
        self._coordinate_by_id = dict(zip(self.node_ids, self.node_coordinates))
        self._node_id_by_identity = {
            identity: node_id
            for node_id, identity in zip(self.node_ids, self.node_identities)
        }

        if state is None:
            self._master_shares, self._master_commitments = (
                self._generate_distributed_secret()
            )
            self._trace_shares, self._trace_commitments = (
                self._generate_distributed_secret()
            )
            self._certificate_shares, self._certificate_commitments = (
                self._generate_distributed_secret()
            )
            self._consumed_trace_authorizations: set[str] = set()
        else:
            if (
                state.threshold != threshold
                or state.node_count != node_count
                or state.max_accumulator_size != max_accumulator_size
                or state.node_identities != self.node_identities
                or state.node_coordinates != self.node_coordinates
            ):
                raise ValueError("D-KGC checkpoint parameters do not match the context")
            self._master_shares, self._master_commitments = (
                self._validate_shared_secret(
                    state.master_shares,
                    state.master_commitments,
                )
            )
            self._trace_shares, self._trace_commitments = (
                self._validate_shared_secret(
                    state.trace_shares,
                    state.trace_commitments,
                )
            )
            self._certificate_shares, self._certificate_commitments = (
                self._validate_shared_secret(
                    state.certificate_shares,
                    state.certificate_commitments,
                )
            )
            consumed = tuple(state.consumed_trace_authorizations)
            if len(set(consumed)) != len(consumed):
                raise ValueError(
                    "D-KGC checkpoint contains duplicate trace authorizations"
                )
            for digest in consumed:
                _require_canonical_hex(
                    digest,
                    32,
                    "consumed trace authorization digest",
                )
            self._consumed_trace_authorizations = set(consumed)

        self.master_public = self._master_commitments[0]
        self.trace_public = self._trace_commitments[0]
        self.certificate_public = self._certificate_commitments[0]
        if self._g2_equal(self.master_public, self.trace_public):
            raise RuntimeError("trace secret must be independent from the SM9 master key")
        if self._g2_equal(self.master_public, self.certificate_public) or self._g2_equal(
            self.trace_public,
            self.certificate_public,
        ):
            raise RuntimeError("threshold certificate key must be independent")
        self.master_pairing = self._pair(self.p1, self.master_public)
        self.trace_basis = self._build_trace_basis()

        # Delta_j remains as per-node shares inside the D-KGC boundary.  The
        # AS-facing ring record never receives the scalar or its shares.
        self._ring_delta_shares: dict[
            str,
            dict[str, tuple[ScalarShare, ...]],
        ] = {}
        self._ring_members: dict[str, tuple[str, ...]] = {}
        self._ring_identity_scalars: dict[str, dict[str, int]] = {}
        self._active_trace_sessions: dict[bytes, _TraceSessionRecord] = {}

    @property
    def node_ids(self) -> tuple[int, ...]:
        return tuple(range(1, self.node_count + 1))

    def export_state(self) -> DistributedKGCState:
        """Export all node shares only for the trusted local checkpoint owner."""

        return DistributedKGCState(
            threshold=self.threshold,
            node_count=self.node_count,
            max_accumulator_size=self.max_accumulator_size,
            node_identities=self.node_identities,
            node_coordinates=self.node_coordinates,
            master_shares=self._master_shares,
            trace_shares=self._trace_shares,
            certificate_shares=self._certificate_shares,
            master_commitments=self._master_commitments,
            trace_commitments=self._trace_commitments,
            certificate_commitments=self._certificate_commitments,
            consumed_trace_authorizations=tuple(
                sorted(self._consumed_trace_authorizations)
            ),
        )

    def validate_public_relations(self) -> bool:
        """Validate every public key through an independent threshold quorum."""

        participants = self.node_ids[-self.threshold :]
        return (
            self._g2_equal(
                self._combine_public_key(self._master_shares, participants),
                self.master_public,
            )
            and self._g2_equal(
                self._combine_public_key(self._trace_shares, participants),
                self.trace_public,
            )
            and self._g2_equal(
                self._combine_public_key(self._certificate_shares, participants),
                self.certificate_public,
            )
            and not self._g2_equal(self.trace_public, self.master_public)
            and self._verify_all_share_relations(
                self._master_shares,
                self._master_commitments,
            )
            and self._verify_all_share_relations(
                self._trace_shares,
                self._trace_commitments,
            )
            and self._verify_all_share_relations(
                self._certificate_shares,
                self._certificate_commitments,
            )
            and self._validate_trace_basis()
        )

    def trace_approval_issuer(self, node_identity: str) -> TraceApprovalIssuer:
        """Provision one logical node endpoint to the trusted harness.

        Knowing the public node name or its old integer index is insufficient:
        the returned endpoint owns the corresponding certificate-key share and
        must Schnorr-sign each fresh authorization request.
        """

        if not isinstance(node_identity, str):
            raise TypeError("D-KGC node identity must be a string")
        try:
            node_id = self._node_id_by_identity[node_identity]
        except KeyError as exc:
            raise ValueError("unknown D-KGC node identity") from exc
        share = self._certificate_shares[node_id - 1]
        return TraceApprovalIssuer(
            node_identity=node_identity,
            share=share,
            p2=self.p2,
            crypto_mode=self.crypto_mode,
            simulated_seed=(
                self._random_nonzero()
                if self.crypto_mode == "simulated"
                else None
            ),
        )

    def verify_trace_approval(
        self,
        request: TraceAuthorizationRequest,
        approval: TraceApproval,
    ) -> bool:
        """Verify one node approval against its Feldman-derived public share."""

        try:
            if not isinstance(request, TraceAuthorizationRequest):
                return False
            if not isinstance(approval, TraceApproval):
                return False
            node_id = self._node_id_by_identity[approval.node_identity]
            if approval.request_digest != _trace_request_digest(
                request,
                self.crypto_mode,
            ):
                return False
            if not 1 <= approval.response < SCALAR_MODULUS:
                return False
            if not self._valid_certificate_commitment(approval.commitment):
                return False
            public_share = self._public_share_at(
                self._certificate_commitments,
                node_id,
            )
            challenge = _trace_approval_challenge(
                request,
                approval.node_identity,
                approval.commitment,
                self.crypto_mode,
            )
            if self.crypto_mode == "simulated":
                expected = (
                    int(approval.commitment)
                    + challenge * int(public_share)
                ) % SCALAR_MODULUS
                return approval.response == expected
            left = sm9_backend.g2_mul(self.p2, approval.response)
            right = sm9_backend.g2_add(
                approval.commitment,
                sm9_backend.g2_mul(public_share, challenge),
            )
            return left == right
        except (KeyError, OverflowError, TypeError, ValueError):
            return False

    def begin_trace(
        self,
        authorization: TraceAuthorization,
        *,
        expected_task_id: str,
        expected_rid: str,
        expected_evidence_digest: str,
        expected_task_point: Any,
        expected_signature_a: Any,
        expected_signature_b: Any,
        expected_task_tag: Any,
    ) -> _AuthorizedTraceSession:
        """Verify approvals and bind a one-shot session to task evidence/h_t."""

        if not isinstance(authorization, TraceAuthorization):
            raise TypeError("trace requires a signed TraceAuthorization")
        request = authorization.request
        if (
            request.task_id != expected_task_id
            or request.rid != expected_rid
            or request.evidence_digest != expected_evidence_digest
        ):
            raise ValueError("trace authorization is bound to different evidence")
        if expected_rid not in self._ring_members:
            raise ValueError("D-KGC has no trace material for the authorized RID")
        task_point_encoding = self._encode_task_point(expected_task_point)
        signature_a_encoding = self._encode_signature_g1(expected_signature_a)
        signature_b_encoding = self._encode_signature_g1(expected_signature_b)
        evidence_task_tag_encoding = self._encode_task_tag(expected_task_tag)
        request_digest = _trace_request_digest(request, self.crypto_mode)
        if request_digest in self._consumed_trace_authorizations:
            raise ValueError("trace authorization session has already been consumed")

        valid_nodes: dict[int, TraceApproval] = {}
        for approval in authorization.approvals:
            if not isinstance(approval, TraceApproval):
                continue
            node_id = self._node_id_by_identity.get(approval.node_identity)
            if node_id is None or node_id in valid_nodes:
                continue
            if self.verify_trace_approval(request, approval):
                valid_nodes[node_id] = approval
        if len(valid_nodes) < self.threshold:
            raise ThresholdNotMetError(
                f"at least {self.threshold} distinct signed D-KGC approvals are required"
            )

        # A degree-(t-1) Shamir secret needs exactly t shares.  Choosing a
        # canonical subset keeps the trace transcript independent of approval
        # arrival order while still proving that at least t nodes approved it.
        quorum = tuple(sorted(valid_nodes)[: self.threshold])
        token = secrets.token_bytes(32)
        while token in self._active_trace_sessions:  # pragma: no cover
            token = secrets.token_bytes(32)
        self._consumed_trace_authorizations.add(request_digest)
        self._active_trace_sessions[token] = _TraceSessionRecord(
            request_digest=request_digest,
            task_id=request.task_id,
            rid=request.rid,
            evidence_digest=request.evidence_digest,
            quorum=quorum,
            task_point_encoding=task_point_encoding,
            signature_a_encoding=signature_a_encoding,
            signature_b_encoding=signature_b_encoding,
            evidence_task_tag_encoding=evidence_task_tag_encoding,
        )
        return _AuthorizedTraceSession(token)

    def end_trace(self, session: _AuthorizedTraceSession) -> None:
        """Close an authorized trace session and erase its active capability."""

        token = self._trace_session_token(session)
        if self._active_trace_sessions.pop(token, None) is None:
            raise ValueError("trace authorization session is not active")

    def extract_signing_key(self, identity_scalar: int) -> Any:
        """Execute the blinded threshold private-key delivery in Section 4.3.1."""

        if not 1 <= identity_scalar < SCALAR_MODULUS:
            raise ValueError("identity scalar must be in Z_N*")
        participants = self.node_ids[: self.threshold]
        shares = self._select(self._master_shares, participants)
        lambdas = self._lagrange(participants)
        inv_t = _mod_inverse(self.threshold)

        # k'_i = lambda_i*k_i + v/t; their sum is msk+v, but that sum is
        # never materialized below.  The double sum models the paper's
        # Paillier-secured cross terms for delta=(sum k'_i)(sum gamma_i).
        adjusted = tuple(
            (
                lambdas[share.node_id] * share.value
                + identity_scalar * inv_t
            )
            % SCALAR_MODULUS
            for share in shares
        )
        for _ in range(16):
            gammas = tuple(self._random_nonzero() for _ in participants)
            denominator = sum(
                adjusted_i * gamma_j
                for adjusted_i in adjusted
                for gamma_j in gammas
            ) % SCALAR_MODULUS
            if denominator != 0:
                break
        else:
            raise RuntimeError("msk + identity scalar is zero; regenerate D-KGC keys")

        denominator_inverse = _mod_inverse(denominator)
        partial_coefficients = tuple(
            denominator_inverse * gamma * identity_scalar % SCALAR_MODULUS
            for gamma in gammas
        )
        if self.crypto_mode == "simulated":
            private_key = (1 - sum(partial_coefficients)) % SCALAR_MODULUS
        else:
            delivered_sum = _g1_sum(
                sm9_backend.g1_mul(self.p1, coefficient)
                for coefficient in partial_coefficients
            )
            private_key = sm9_backend.g1_add(
                self.p1,
                sm9_backend.g1_mul(delivered_sum, SCALAR_MODULUS - 1),
            )

        identity_public = self._g2_add(
            self._g2_multiply(self.p2, identity_scalar),
            self.master_public,
        )
        if not self._gt_equal(
            self._pair(private_key, identity_public),
            self.master_pairing,
        ):
            raise RuntimeError("distributed SM9 private-key delivery relation failed")
        return private_key

    def build_task_ring(
        self,
        rid: str,
        identity_scalars: dict[str, int],
    ) -> tuple[Any, dict[str, Any]]:
        """Build ACC/W and retain only secret-shared Delta values for tracing."""

        ring = tuple(identity_scalars)
        if len(ring) > self.max_accumulator_size:
            raise ValueError(
                "task ring exceeds the published accumulator basis capacity"
            )
        if not ring or any(not isinstance(identity, str) or not identity for identity in ring):
            raise ValueError("task ring identities must be non-empty strings")
        if any(
            not isinstance(scalar, int) or not 1 <= scalar < SCALAR_MODULUS
            for scalar in identity_scalars.values()
        ):
            raise ValueError("task ring identity scalars must be in Z_N*")
        existing = self._ring_members.get(rid)
        if existing is not None and existing != ring:
            raise RuntimeError("RID collision inside D-KGC ring storage")
        if existing is not None and self._ring_identity_scalars.get(rid) != identity_scalars:
            raise RuntimeError("RID reused with different identity scalars")

        # Public ACC/W are evaluated from L_xi.  In parallel, the same public
        # polynomials are evaluated on xi shares for the one masked inversion
        # prescribed by Section 4.3.4.  Neither xi nor P_r is reconstructed.
        polynomial = _product_polynomial(identity_scalars.values())
        accumulator = self._evaluate_trace_basis(polynomial)
        witness_polynomials = {
            identity: _divide_by_linear_factor(polynomial, scalar)
            for identity, scalar in identity_scalars.items()
        }
        witnesses = {
            identity: self._evaluate_trace_basis(coefficients)
            for identity, coefficients in witness_polynomials.items()
        }

        product_shares = self._secure_polynomial_evaluate(
            polynomial,
            self._trace_shares,
        )
        inverse_product_shares = self._secure_inverse(product_shares)
        self._ring_delta_shares[rid] = {
            identity: self._secure_multiply(
                self._add_public(self._trace_shares, scalar),
                inverse_product_shares,
            )
            for identity, scalar in identity_scalars.items()
        }
        self._ring_members[rid] = ring
        self._ring_identity_scalars[rid] = dict(identity_scalars)
        return accumulator, witnesses

    def discard_task_ring(self, rid: str) -> None:
        """Erase task-ring trace shares after no active task references RID."""

        self._ring_delta_shares.pop(rid, None)
        self._ring_members.pop(rid, None)
        self._ring_identity_scalars.pop(rid, None)

    def trace_match(
        self,
        *,
        rid: str,
        identities: Sequence[str],
        identity_scalars: dict[str, int],
        signature_a: Any,
        signature_b: Any,
        session: _AuthorizedTraceSession,
    ) -> str:
        """Evaluate (3)--(5) once, then bind the session to the unique member."""

        trace_session = self._trace_session_record(session, expected_rid=rid)
        if trace_session.phase != "authorized":
            raise ValueError("trace_match is only allowed as the first trace operation")
        if (
            self._encode_signature_g1(signature_a)
            != trace_session.signature_a_encoding
            or self._encode_signature_g1(signature_b)
            != trace_session.signature_b_encoding
        ):
            raise ValueError("trace_match signature differs from authorized evidence")
        quorum = trace_session.quorum
        expected_ring = self._ring_members.get(rid)
        if expected_ring is None or tuple(identities) != expected_ring:
            raise ValueError("D-KGC has no matching task-ring trace material")
        expected_scalars = self._ring_identity_scalars[rid]
        supplied_scalars = {
            str(identity): int(scalar) % SCALAR_MODULUS
            for identity, scalar in identity_scalars.items()
        }
        if supplied_scalars != expected_scalars:
            raise ValueError("identity scalars differ from registered ring material")
        delta_by_identity = self._ring_delta_shares[rid]
        matches: list[str] = []
        for identity in identities:
            candidate_public = self._g2_add(
                self._g2_multiply(self.p2, identity_scalars[identity]),
                self.master_public,
            )
            x_value = self._pair(signature_b, candidate_public)
            delta_a = self._combine_base_share_products(
                signature_a,
                delta_by_identity[identity],
                quorum,
            )
            y_value = self._pair(delta_a, self.master_public)
            if self._gt_equal(x_value, y_value):
                matches.append(identity)
        if len(matches) != 1:
            raise ValueError("Equation (5) did not identify one unique ring member")
        identity = matches[0]
        trace_session.matched_identity = identity
        trace_session.matched_identity_scalar = expected_scalars[identity]
        trace_session.phase = "matched"
        return identity

    def reconstruct_candidate_tag(
        self,
        identity_scalar: int,
        task_point: Any,
        session: _AuthorizedTraceSession,
    ) -> Any:
        """Run Equation (6) only for the session-bound member and task point."""

        trace_session = self._trace_session_record(session)
        if trace_session.phase != "matched":
            raise ValueError(
                "candidate-tag reconstruction requires one successful trace_match"
            )
        if identity_scalar != trace_session.matched_identity_scalar:
            raise ValueError("candidate tag requested for a different ring identity")
        if self._encode_task_point(task_point) != trace_session.task_point_encoding:
            raise ValueError("candidate tag requested for a different task point")
        quorum = trace_session.quorum
        denominator_shares = self._add_public(
            self._master_shares,
            identity_scalar,
        )
        beta_shares = self._secure_multiply(
            self._master_shares,
            self._secure_inverse(denominator_shares),
        )
        shares = self._select(beta_shares, quorum)
        lambdas = self._lagrange(quorum)
        if self.crypto_mode == "simulated":
            candidate_tag = sum(
                lambdas[share.node_id] * share.value * int(task_point)
                for share in shares
            ) % SCALAR_MODULUS
        else:
            candidate_tag = sm9_backend.gt_one()
            for share in shares:
                # A Shamir share may legitimately be zero even when the shared
                # secret is non-zero.  The bridge rejects infinity encodings,
                # so omit zero contributions instead of computing 0*P1.
                if share.value == 0:
                    continue
                partial_point = sm9_backend.g1_mul(self.p1, share.value)
                partial_tag = sm9_backend.pair(partial_point, task_point)
                weighted = sm9_backend.gt_pow(
                    partial_tag,
                    lambdas[share.node_id],
                )
                candidate_tag = sm9_backend.gt_mul(candidate_tag, weighted)
        encoded_candidate = self._encode_task_tag(candidate_tag)
        if encoded_candidate != trace_session.evidence_task_tag_encoding:
            raise ValueError("Equation (6) task tag differs from authorized evidence")
        trace_session.reconstructed_tag_encoding = encoded_candidate
        trace_session.phase = "tag_reconstructed"
        return candidate_tag

    def threshold_sign(
        self,
        message: bytes,
        participants: Sequence[int],
    ) -> ThresholdCertificate:
        """Generate a threshold Schnorr certificate for trusted setup/tests.

        The trace protocol never calls this integer-selection helper; it uses
        :meth:`threshold_sign_authorized`, whose quorum comes from signed node
        approvals verified by :meth:`begin_trace`.
        """

        quorum = self._quorum(participants)
        return self._threshold_sign_with_quorum(message, quorum)

    def threshold_sign_authorized(
        self,
        message: bytes,
        session: _AuthorizedTraceSession,
        *,
        task_tag: Any,
    ) -> ThresholdCertificate:
        """Sign only the final trace message fixed by this session's state."""

        trace_session = self._trace_session_record(session)
        expected_message = self._authorized_trace_message(trace_session, task_tag)
        if bytes(message) != expected_message:
            raise ValueError("threshold certificate message is not session-bound")
        certificate = self._threshold_sign_with_quorum(
            expected_message,
            trace_session.quorum,
        )
        trace_session.phase = "certified"
        return certificate

    def finalize_trace_certificate(
        self,
        task_tag: Any,
        session: _AuthorizedTraceSession,
    ) -> ThresholdCertificate:
        """Construct and sign the only final trace message allowed by a session."""

        trace_session = self._trace_session_record(session)
        message = self._authorized_trace_message(trace_session, task_tag)
        certificate = self._threshold_sign_with_quorum(
            message,
            trace_session.quorum,
        )
        trace_session.phase = "certified"
        return certificate

    def _threshold_sign_with_quorum(
        self,
        message: bytes,
        quorum: tuple[int, ...],
    ) -> ThresholdCertificate:
        shares = self._select(self._certificate_shares, quorum)
        lambdas = self._lagrange(quorum)
        for _ in range(16):
            nonces = {share.node_id: self._random_nonzero() for share in shares}
            if self.crypto_mode == "simulated":
                commitment = sum(nonces.values()) % SCALAR_MODULUS
            else:
                commitment = _g2_sum(
                    sm9_backend.g2_mul(self.p2, nonces[share.node_id])
                    for share in shares
                )
            if self._valid_certificate_commitment(commitment):
                break
        else:  # pragma: no cover - negligible probability
            raise RuntimeError("failed to generate a non-zero threshold nonce")

        challenge = self._certificate_challenge(message, commitment)
        response = sum(
            nonces[share.node_id]
            + challenge * lambdas[share.node_id] * share.value
            for share in shares
        ) % SCALAR_MODULUS
        if response == 0:  # negligible; retry avoids an infinity left side
            return self._threshold_sign_with_quorum(message, quorum)
        return ThresholdCertificate(commitment=commitment, response=response)

    def threshold_verify(
        self,
        message: bytes,
        certificate: ThresholdCertificate,
    ) -> bool:
        """Verify ``TVrfy(TPK,M_trace,tau_trace)=1``."""

        return ThresholdCertificateVerifier(
            certificate_public=self.certificate_public,
            p2=self.p2,
            crypto_mode=self.crypto_mode,
        ).verify(message, certificate)

    def _certificate_challenge(self, message: bytes, commitment: Any) -> int:
        return _certificate_challenge(message, commitment, self.crypto_mode)

    def _valid_certificate_commitment(self, commitment: Any) -> bool:
        if self.crypto_mode == "simulated":
            return isinstance(commitment, int) and 0 < commitment < SCALAR_MODULUS
        return sm9_backend.g2_validate(commitment)

    def _public_share_at(
        self,
        commitments: Sequence[Any],
        node_id: int,
    ) -> Any:
        """Evaluate Feldman commitments at one node's hashed coordinate."""

        coordinate = self._coordinate_by_id[node_id]
        if self.crypto_mode == "simulated":
            return sum(
                int(commitment)
                * pow(coordinate, exponent, SCALAR_MODULUS)
                for exponent, commitment in enumerate(commitments)
            ) % SCALAR_MODULUS
        return _g2_sum(
            sm9_backend.g2_mul(
                commitment,
                pow(coordinate, exponent, SCALAR_MODULUS),
            )
            for exponent, commitment in enumerate(commitments)
        )

    @staticmethod
    def _trace_session_token(session: _AuthorizedTraceSession) -> bytes:
        if not isinstance(session, _AuthorizedTraceSession):
            raise TypeError("trace operation requires an authorized session")
        token = getattr(session, "_AuthorizedTraceSession__token", None)
        if not isinstance(token, bytes) or len(token) != 32:
            raise ValueError("invalid trace authorization session")
        return token

    def _trace_session_record(
        self,
        session: _AuthorizedTraceSession,
        *,
        expected_rid: str | None = None,
    ) -> _TraceSessionRecord:
        token = self._trace_session_token(session)
        record = self._active_trace_sessions.get(token)
        if record is None:
            raise ValueError("trace authorization session is not active")
        if expected_rid is not None and record.rid != expected_rid:
            raise ValueError("trace authorization session is bound to another RID")
        return record

    def _encode_task_point(self, task_point: Any) -> bytes:
        if self.crypto_mode == "simulated":
            if not isinstance(task_point, int) or not 1 <= task_point < SCALAR_MODULUS:
                raise ValueError("task point must be a non-zero simulated G2 element")
            return int(task_point).to_bytes(32, "big")
        if not sm9_backend.g2_validate(task_point):
            raise ValueError("task point is not a canonical SM9 G2 element")
        return bytes(task_point)

    def _encode_signature_g1(self, point: Any) -> bytes:
        if self.crypto_mode == "simulated":
            if not isinstance(point, int) or not 1 <= point < SCALAR_MODULUS:
                raise ValueError("signature point must be a non-zero simulated G1 element")
            return int(point).to_bytes(32, "big")
        if not sm9_backend.g1_validate(point):
            raise ValueError("signature point is not a canonical SM9 G1 element")
        return bytes(point)

    def _encode_task_tag(self, task_tag: Any) -> bytes:
        if self.crypto_mode == "simulated":
            if not isinstance(task_tag, int) or not 1 <= task_tag < SCALAR_MODULUS:
                raise ValueError("task tag must be a non-zero simulated GT element")
            return int(task_tag).to_bytes(32, "big")
        if not sm9_backend.gt_validate(task_tag) or sm9_backend.gt_equal(
            task_tag,
            sm9_backend.gt_one(),
        ):
            raise ValueError("task tag is not a non-identity canonical SM9 GT element")
        return bytes(task_tag)

    def _authorized_trace_message(
        self,
        trace_session: _TraceSessionRecord,
        task_tag: Any,
    ) -> bytes:
        if trace_session.phase != "tag_reconstructed":
            raise ValueError(
                "trace certificate requires successful match and tag reconstruction"
            )
        if trace_session.matched_identity is None:
            raise RuntimeError("trace session lost its matched identity")
        encoded_tag = self._encode_task_tag(task_tag)
        if encoded_tag != trace_session.reconstructed_tag_encoding:
            raise ValueError("trace certificate task tag differs from Equation (6)")
        return _encode_fields(
            bytes.fromhex(trace_session.evidence_digest),
            trace_session.matched_identity.encode("utf-8"),
            encoded_tag,
            bytes.fromhex(trace_session.rid),
            b"trace",
        )

    def _combine_base_share_products(
        self,
        base: Any,
        shares: Sequence[ScalarShare],
        participants: Sequence[int],
    ) -> Any:
        selected = self._select(shares, participants)
        lambdas = self._lagrange(participants)
        if self.crypto_mode == "simulated":
            return sum(
                int(base) * share.value * lambdas[share.node_id]
                for share in selected
            ) % SCALAR_MODULUS
        coefficients = tuple(
            share.value * lambdas[share.node_id] % SCALAR_MODULUS
            for share in selected
        )
        return _g1_sum(
            sm9_backend.g1_mul(base, coefficient)
            for coefficient in coefficients
            if coefficient != 0
        )

    def _combine_public_key(
        self,
        shares: Sequence[ScalarShare],
        participants: Sequence[int] | None = None,
    ) -> Any:
        quorum = (
            self.node_ids[: self.threshold]
            if participants is None
            else self._quorum(participants)
        )
        selected = self._select(shares, quorum)
        lambdas = self._lagrange(quorum)
        if self.crypto_mode == "simulated":
            return sum(
                share.value * lambdas[share.node_id] for share in selected
            ) % SCALAR_MODULUS
        return _g2_sum(
            sm9_backend.g2_mul(
                self.p2,
                share.value * lambdas[share.node_id] % SCALAR_MODULUS,
            )
            for share in selected
        )

    def _quorum(self, participants: Sequence[int]) -> tuple[int, ...]:
        unique = tuple(dict.fromkeys(int(node_id) for node_id in participants))
        if len(unique) < self.threshold:
            raise ThresholdNotMetError(
                f"at least {self.threshold} distinct D-KGC nodes are required"
            )
        if any(node_id < 1 or node_id > self.node_count for node_id in unique):
            raise ValueError("unknown D-KGC node identifier")
        return unique[: self.threshold]

    def _lagrange(self, participants: Iterable[int]) -> dict[int, int]:
        ids = tuple(int(node_id) for node_id in participants)
        return lagrange_coefficients(ids, coordinates=self._coordinate_by_id)

    def _hash_node_coordinate(self, identity: str) -> int:
        encoded = identity.encode("utf-8")
        if self.crypto_mode == "sm9":
            return sm9_backend.hash_to_scalar(1, encoded)
        return int.from_bytes(hashlib.sha256(b"\x01" + encoded).digest(), "big") % (
            SCALAR_MODULUS - 1
        ) + 1

    def _constant_shares(self, value: int) -> tuple[ScalarShare, ...]:
        scalar = value % SCALAR_MODULUS
        return tuple(
            ScalarShare(node_id=node_id, value=scalar)
            for node_id in self.node_ids
        )

    def _add_public(
        self,
        shares: Sequence[ScalarShare],
        value: int,
    ) -> tuple[ScalarShare, ...]:
        scalar = value % SCALAR_MODULUS
        return tuple(
            ScalarShare(
                node_id=share.node_id,
                value=(share.value + scalar) % SCALAR_MODULUS,
            )
            for share in shares
        )

    def _secure_multiply(
        self,
        left: Sequence[ScalarShare],
        right: Sequence[ScalarShare],
    ) -> tuple[ScalarShare, ...]:
        """Model Paillier cross terms and refresh the product as Shamir shares.

        For a quorum S, a(0)b(0) equals the double sum of
        lambda_i*a_i*lambda_j*b_j.  Each cross term is freshly reshared before
        aggregation, so this method never assembles either input secret or the
        product scalar.
        """

        participants = self.node_ids[: self.threshold]
        selected_left = self._select(left, participants)
        selected_right = self._select(right, participants)
        lambdas = self._lagrange(participants)
        aggregate = [0] * self.node_count
        for left_share in selected_left:
            weighted_left = (
                lambdas[left_share.node_id] * left_share.value
            ) % SCALAR_MODULUS
            for right_share in selected_right:
                cross_term = (
                    weighted_left
                    * lambdas[right_share.node_id]
                    * right_share.value
                ) % SCALAR_MODULUS
                refreshed = split_secret(
                    cross_term,
                    threshold=self.threshold,
                    node_count=self.node_count,
                    rng=self._rng,
                    coordinates=self._coordinate_by_id,
                )
                for share in refreshed:
                    aggregate[share.node_id - 1] = (
                        aggregate[share.node_id - 1] + share.value
                    ) % SCALAR_MODULUS
        return tuple(
            ScalarShare(node_id=node_id, value=aggregate[node_id - 1])
            for node_id in self.node_ids
        )

    def _fresh_random_shares(self) -> tuple[ScalarShare, ...]:
        """Aggregate independent dealer contributions without opening rho."""

        aggregate = [0] * self.node_count
        for _dealer in self.node_ids:
            contribution = split_secret(
                self._random_nonzero(),
                threshold=self.threshold,
                node_count=self.node_count,
                rng=self._rng,
                coordinates=self._coordinate_by_id,
            )
            for share in contribution:
                aggregate[share.node_id - 1] = (
                    aggregate[share.node_id - 1] + share.value
                ) % SCALAR_MODULUS
        return tuple(
            ScalarShare(node_id=node_id, value=aggregate[node_id - 1])
            for node_id in self.node_ids
        )

    def _secure_inverse(
        self,
        shares: Sequence[ScalarShare],
    ) -> tuple[ScalarShare, ...]:
        """Return shares of x^-1 after opening only z=rho*x as in the Word."""

        for _ in range(16):
            random_shares = self._fresh_random_shares()
            masked_product = self._secure_multiply(random_shares, shares)
            opened_mask = self._open_masked(masked_product)
            if opened_mask == 0:
                continue
            inverse_mask = _mod_inverse(opened_mask)
            return tuple(
                ScalarShare(
                    node_id=share.node_id,
                    value=share.value * inverse_mask % SCALAR_MODULUS,
                )
                for share in random_shares
            )
        raise RuntimeError("cannot invert a zero shared value")

    def _open_masked(self, shares: Sequence[ScalarShare]) -> int:
        """Open only a uniformly blinded value explicitly allowed by MPC."""

        participants = self.node_ids[: self.threshold]
        selected = self._select(shares, participants)
        lambdas = self._lagrange(participants)
        return sum(
            lambdas[share.node_id] * share.value for share in selected
        ) % SCALAR_MODULUS

    def _secure_polynomial_evaluate(
        self,
        coefficients: Sequence[int],
        secret_shares: Sequence[ScalarShare],
    ) -> tuple[ScalarShare, ...]:
        if not coefficients:
            raise ValueError("polynomial must have at least one coefficient")
        result = self._constant_shares(coefficients[-1])
        for coefficient in reversed(coefficients[:-1]):
            result = self._add_public(
                self._secure_multiply(result, secret_shares),
                coefficient,
            )
        return result

    def _build_trace_basis(self) -> tuple[Any, ...]:
        participants = self.node_ids[: self.threshold]
        power_shares = self._constant_shares(1)
        basis: list[Any] = []
        for exponent in range(self.max_accumulator_size + 1):
            basis.append(
                self._combine_base_share_products(
                    self.p1,
                    power_shares,
                    participants,
                )
            )
            if exponent != self.max_accumulator_size:
                power_shares = self._secure_multiply(
                    power_shares,
                    self._trace_shares,
                )
        return tuple(basis)

    def _evaluate_trace_basis(self, coefficients: Sequence[int]) -> Any:
        if not coefficients or len(coefficients) > len(self.trace_basis):
            raise ValueError("polynomial exceeds the published L_xi basis")
        reduced = tuple(
            int(coefficient) % SCALAR_MODULUS for coefficient in coefficients
        )
        if self.crypto_mode == "simulated":
            return sum(
                coefficient * int(point)
                for coefficient, point in zip(reduced, self.trace_basis)
            ) % SCALAR_MODULUS
        return _g1_sum(
            sm9_backend.g1_mul(point, coefficient)
            for coefficient, point in zip(reduced, self.trace_basis)
            if coefficient != 0
        )

    def _generate_distributed_secret(
        self,
    ) -> tuple[tuple[ScalarShare, ...], tuple[Any, ...]]:
        """Aggregate n dealer polynomials and their Feldman commitments."""

        while True:
            aggregate_shares = [0] * self.node_count
            aggregate_commitments: list[Any | None] = [None] * self.threshold
            for _dealer in self.node_ids:
                polynomial = [
                    self._random_nonzero() for _ in range(self.threshold)
                ]
                for exponent, coefficient in enumerate(polynomial):
                    contribution = (
                        coefficient
                        if self.crypto_mode == "simulated"
                        else sm9_backend.g2_mul(self.p2, coefficient)
                    )
                    current = aggregate_commitments[exponent]
                    aggregate_commitments[exponent] = (
                        contribution
                        if current is None
                        else (
                            (int(current) + int(contribution)) % SCALAR_MODULUS
                            if self.crypto_mode == "simulated"
                            else sm9_backend.g2_add(current, contribution)
                        )
                    )
                for node_id in self.node_ids:
                    coordinate = self._coordinate_by_id[node_id]
                    aggregate_shares[node_id - 1] = (
                        aggregate_shares[node_id - 1]
                        + evaluate_polynomial(polynomial, coordinate)
                    ) % SCALAR_MODULUS
            # Zero aggregate commitments/shares occur with negligible
            # probability. The bridge rejects infinity, so restart the DKG.
            if all(aggregate_commitments) and all(aggregate_shares):
                shares = tuple(
                    ScalarShare(
                        node_id=node_id,
                        value=aggregate_shares[node_id - 1],
                    )
                    for node_id in self.node_ids
                )
                commitments = tuple(aggregate_commitments)
                if self._verify_all_share_relations(shares, commitments):
                    return shares, commitments

    def _validate_shared_secret(
        self,
        shares: Sequence[ScalarShare],
        commitments: Sequence[Any],
    ) -> tuple[tuple[ScalarShare, ...], tuple[Any, ...]]:
        result = tuple(shares)
        if tuple(share.node_id for share in result) != self.node_ids:
            raise ValueError("D-KGC checkpoint has invalid node share identifiers")
        if any(not 1 <= share.value < SCALAR_MODULUS for share in result):
            raise ValueError("D-KGC checkpoint contains an invalid scalar share")
        public = tuple(commitments)
        if len(public) != self.threshold:
            raise ValueError("D-KGC checkpoint has invalid Feldman commitments")
        if self.crypto_mode == "simulated":
            if any(
                not isinstance(commitment, int)
                or not 1 <= commitment < SCALAR_MODULUS
                for commitment in public
            ):
                raise ValueError("D-KGC checkpoint has invalid commitments")
        elif not all(sm9_backend.g2_validate(commitment) for commitment in public):
            raise ValueError("D-KGC checkpoint has invalid SM9 commitments")
        if not self._verify_all_share_relations(result, public):
            raise ValueError("D-KGC checkpoint share failed Feldman verification")
        return result, public

    def _verify_all_share_relations(
        self,
        shares: Sequence[ScalarShare],
        commitments: Sequence[Any],
    ) -> bool:
        for share in shares:
            coordinate = self._coordinate_by_id[share.node_id]
            if self.crypto_mode == "simulated":
                expected = sum(
                    int(commitment)
                    * pow(coordinate, exponent, SCALAR_MODULUS)
                    for exponent, commitment in enumerate(commitments)
                ) % SCALAR_MODULUS
                if share.value != expected:
                    return False
                continue
            left = sm9_backend.g2_mul(self.p2, share.value)
            right = _g2_sum(
                sm9_backend.g2_mul(
                    commitment,
                    pow(coordinate, exponent, SCALAR_MODULUS),
                )
                for exponent, commitment in enumerate(commitments)
            )
            if left != right:
                return False
        return True

    def _validate_trace_basis(self) -> bool:
        if len(self.trace_basis) != self.max_accumulator_size + 1:
            return False
        if self.trace_basis[0] != self.p1:
            return False
        for exponent in range(self.max_accumulator_size):
            if self.crypto_mode == "simulated":
                if (
                    int(self.trace_basis[exponent]) * int(self.trace_public)
                    - int(self.trace_basis[exponent + 1])
                ) % SCALAR_MODULUS != 0:
                    return False
            elif not sm9_backend.gt_equal(
                sm9_backend.pair(self.trace_basis[exponent], self.trace_public),
                sm9_backend.pair(self.trace_basis[exponent + 1], self.p2),
            ):
                return False
        return True

    def _select(
        self,
        shares: Sequence[ScalarShare],
        participants: Iterable[int],
    ) -> tuple[ScalarShare, ...]:
        by_id = {share.node_id: share for share in shares}
        try:
            return tuple(by_id[int(node_id)] for node_id in participants)
        except KeyError as exc:
            raise ValueError("unknown D-KGC node identifier") from exc

    def _pair(self, paper_g1: Any, paper_g2: Any) -> Any:
        if self.crypto_mode == "simulated":
            return int(paper_g1) * int(paper_g2) % SCALAR_MODULUS
        return sm9_backend.pair(paper_g1, paper_g2)

    def _g2_multiply(self, value: Any, scalar: int) -> Any:
        if self.crypto_mode == "simulated":
            return int(value) * scalar % SCALAR_MODULUS
        return sm9_backend.g2_mul(value, scalar % SCALAR_MODULUS)

    def _g2_add(self, left: Any, right: Any) -> Any:
        if self.crypto_mode == "simulated":
            return (int(left) + int(right)) % SCALAR_MODULUS
        return sm9_backend.g2_add(left, right)

    def _g2_equal(self, left: Any, right: Any) -> bool:
        if self.crypto_mode == "simulated":
            return int(left) % SCALAR_MODULUS == int(right) % SCALAR_MODULUS
        return bytes(left) == bytes(right)

    def _gt_equal(self, left: Any, right: Any) -> bool:
        if self.crypto_mode == "simulated":
            return int(left) % SCALAR_MODULUS == int(right) % SCALAR_MODULUS
        return sm9_backend.gt_equal(left, right)

    def _random_nonzero(self) -> int:
        return self._rng.randrange(1, SCALAR_MODULUS)


def split_secret(
    secret: int,
    *,
    threshold: int,
    node_count: int,
    rng: random.Random | SystemRandom,
    coordinates: dict[int, int] | None = None,
) -> tuple[ScalarShare, ...]:
    """Split a scalar with Shamir sharing over the SM9 group order."""

    polynomial = [secret % SCALAR_MODULUS] + [
        rng.randrange(SCALAR_MODULUS) for _ in range(threshold - 1)
    ]
    points = (
        {node_id: node_id for node_id in range(1, node_count + 1)}
        if coordinates is None
        else coordinates
    )
    if set(points) != set(range(1, node_count + 1)):
        raise ValueError("Shamir coordinates must cover every D-KGC node")
    if any(not 1 <= int(point) < SCALAR_MODULUS for point in points.values()):
        raise ValueError("Shamir coordinates must be non-zero elements of Z_N")
    if len(set(map(int, points.values()))) != node_count:
        raise ValueError("Shamir coordinates must be distinct")
    return tuple(
        ScalarShare(
            node_id=node_id,
            value=evaluate_polynomial(polynomial, int(points[node_id])),
        )
        for node_id in range(1, node_count + 1)
    )


def evaluate_polynomial(coefficients: Sequence[int], x: int) -> int:
    result = 0
    for coefficient in reversed(coefficients):
        result = (result * x + coefficient) % SCALAR_MODULUS
    return result


def lagrange_coefficients(
    node_ids: Iterable[int],
    *,
    coordinates: dict[int, int] | None = None,
) -> dict[int, int]:
    """Return interpolation-at-zero coefficients for D-KGC coordinates."""

    ids = tuple(int(node_id) for node_id in node_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("node identifiers must be non-empty and distinct")
    points = (
        {node_id: node_id for node_id in ids}
        if coordinates is None
        else coordinates
    )
    try:
        selected_points = {node_id: int(points[node_id]) for node_id in ids}
    except KeyError as exc:
        raise ValueError("missing Shamir coordinate for D-KGC node") from exc
    if any(not 1 <= point < SCALAR_MODULUS for point in selected_points.values()):
        raise ValueError("Shamir coordinates must be non-zero elements of Z_N")
    if len(set(selected_points.values())) != len(ids):
        raise ValueError("Shamir coordinates must be distinct")
    coefficients: dict[int, int] = {}
    for current_id in ids:
        current = selected_points[current_id]
        numerator = 1
        denominator = 1
        for other_id in ids:
            if other_id == current_id:
                continue
            other = selected_points[other_id]
            numerator = numerator * (-other) % SCALAR_MODULUS
            denominator = denominator * (current - other) % SCALAR_MODULUS
        coefficients[current_id] = (
            numerator * _mod_inverse(denominator % SCALAR_MODULUS)
        ) % SCALAR_MODULUS
    return coefficients


def _product_polynomial(values: Iterable[int]) -> tuple[int, ...]:
    """Return ascending coefficients of product(x + value)."""

    coefficients = [1]
    for raw_value in values:
        value = int(raw_value) % SCALAR_MODULUS
        updated = [0] * (len(coefficients) + 1)
        for exponent, coefficient in enumerate(coefficients):
            updated[exponent] = (
                updated[exponent] + value * coefficient
            ) % SCALAR_MODULUS
            updated[exponent + 1] = (
                updated[exponent + 1] + coefficient
            ) % SCALAR_MODULUS
        coefficients = updated
    return tuple(coefficients)


def _divide_by_linear_factor(
    coefficients: Sequence[int],
    value: int,
) -> tuple[int, ...]:
    """Exactly divide ascending ``P(x)`` by ``x + value``."""

    if len(coefficients) < 2:
        raise ValueError("polynomial must contain a linear factor")
    factor = int(value) % SCALAR_MODULUS
    quotient = [0] * (len(coefficients) - 1)
    quotient[-1] = int(coefficients[-1]) % SCALAR_MODULUS
    for exponent in range(len(quotient) - 2, -1, -1):
        quotient[exponent] = (
            int(coefficients[exponent + 1])
            - factor * quotient[exponent + 1]
        ) % SCALAR_MODULUS
    remainder = (
        int(coefficients[0]) - factor * quotient[0]
    ) % SCALAR_MODULUS
    if remainder != 0:
        raise ValueError("polynomial does not contain the requested factor")
    return tuple(quotient)


def _require_canonical_hex(value: str, byte_length: int, label: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a hexadecimal string")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not hexadecimal") from exc
    if len(decoded) != byte_length or decoded.hex() != value:
        raise ValueError(
            f"{label} must be canonical lowercase {byte_length}-byte hexadecimal"
        )


def _trace_request_digest(
    request: TraceAuthorizationRequest,
    crypto_mode: str,
) -> str:
    transcript = request.canonical_bytes()
    if crypto_mode == "simulated":
        return hashlib.sha256(transcript).hexdigest()
    if crypto_mode != "sm9":
        raise ValueError("invalid trace-approval crypto mode")
    return sm3.sm3_hash(list(transcript))


def _certificate_challenge(
    message: bytes,
    commitment: Any,
    crypto_mode: str,
) -> int:
    commitment_bytes = (
        int(commitment).to_bytes(32, "big")
        if crypto_mode == "simulated"
        else bytes(commitment)
    )
    transcript = _encode_fields(
        b"SM9-RRS-FL/threshold-schnorr/v2",
        bytes(message),
        commitment_bytes,
    )
    if crypto_mode == "simulated":
        return int.from_bytes(hashlib.sha256(transcript).digest(), "big") % (
            SCALAR_MODULUS - 1
        ) + 1
    if crypto_mode != "sm9":
        raise ValueError("invalid threshold-certificate crypto mode")
    return sm9_backend.hash_to_scalar(2, transcript)


def _trace_approval_challenge(
    request: TraceAuthorizationRequest,
    node_identity: str,
    commitment: Any,
    crypto_mode: str,
) -> int:
    if not isinstance(node_identity, str) or not node_identity:
        raise ValueError("trace approval node identity must not be empty")
    commitment_bytes = (
        int(commitment).to_bytes(32, "big")
        if crypto_mode == "simulated"
        else bytes(commitment)
    )
    transcript = _encode_fields(
        b"SM9-RRS-FL/trace-node-approval-schnorr/v2",
        request.canonical_bytes(),
        node_identity.encode("utf-8"),
        commitment_bytes,
    )
    if crypto_mode == "simulated":
        return int.from_bytes(hashlib.sha256(transcript).digest(), "big") % (
            SCALAR_MODULUS - 1
        ) + 1
    if crypto_mode != "sm9":
        raise ValueError("invalid trace-approval crypto mode")
    return sm9_backend.hash_to_scalar(2, transcript)


def _mod_inverse(value: int) -> int:
    value %= SCALAR_MODULUS
    if value == 0:
        raise ZeroDivisionError("zero has no inverse in Z_N")
    return pow(value, SCALAR_MODULUS - 2, SCALAR_MODULUS)


def _encode_fields(*fields: bytes) -> bytes:
    encoded = bytearray()
    for field in fields:
        encoded.extend(len(field).to_bytes(8, "big"))
        encoded.extend(field)
    return bytes(encoded)


def _g1_sum(points: Iterable[bytes]) -> bytes:
    iterator = iter(points)
    try:
        total = next(iterator)
    except StopIteration as exc:
        raise ValueError("cannot add an empty G1 sequence") from exc
    for point in iterator:
        total = sm9_backend.g1_add(total, point)
    return total


def _g2_sum(points: Iterable[bytes]) -> bytes:
    iterator = iter(points)
    try:
        total = next(iterator)
    except StopIteration as exc:
        raise ValueError("cannot add an empty G2 sequence") from exc
    for point in iterator:
        total = sm9_backend.g2_add(total, point)
    return total


__all__ = [
    "DistributedKGC",
    "DistributedKGCState",
    "SCALAR_MODULUS",
    "ScalarShare",
    "ThresholdCertificateVerifier",
    "ThresholdCertificate",
    "ThresholdNotMetError",
    "TraceApproval",
    "TraceApprovalIssuer",
    "TraceAuthorization",
    "TraceAuthorizationRequest",
    "evaluate_polynomial",
    "lagrange_coefficients",
    "split_secret",
]
