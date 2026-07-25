"""Low-level standard-SM9 bilinear accumulator relation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

from gmssl import sm3

from . import sm9_backend


N = sm9_backend.SM9_ORDER


@dataclass
class AccumulatorRingMaterial:
    ring: tuple[str, ...]
    ring_digest: str
    value: bytes
    value_digest: str
    g1: bytes
    witnesses: dict[str, bytes]
    _g2_cache: dict[str, bytes] = field(default_factory=dict)


class SM9DynamicAccumulator:
    """Nguyen-style accumulator with the Word scheme's G1/G2 direction."""

    def __init__(
        self,
        sign_public: tuple[bytes, bytes, bytes, bytes],
        *,
        trace_public: bytes,
        public_basis: tuple[bytes, ...] | list[bytes],
        hid: int = 0x01,
    ) -> None:
        sm9_backend.require_available()
        self.sign_public = sign_public
        self.p1, self.p2, self.sign_master_public, _ = sign_public
        if not sm9_backend.g1_validate(self.p1):
            raise ValueError("sign_public P1 is not a valid SM9 G1 point")
        if not sm9_backend.g2_validate(self.p2) or not sm9_backend.g2_validate(
            self.sign_master_public
        ):
            raise ValueError("sign_public contains an invalid SM9 G2 point")
        if not 0 <= hid <= 255:
            raise ValueError("hid must fit in one octet")
        self.hid = hid
        if not sm9_backend.g2_validate(trace_public):
            raise ValueError("trace_public is not a valid SM9 G2 point")
        basis = tuple(public_basis)
        if len(basis) < 2 or not all(
            sm9_backend.g1_validate(point) for point in basis
        ):
            raise ValueError("public_basis must contain P1 through [xi^q]P1")
        if basis[0] != self.p1:
            raise ValueError("public_basis[0] must equal P1")
        for exponent in range(len(basis) - 1):
            if not sm9_backend.gt_equal(
                sm9_backend.pair(basis[exponent], trace_public),
                sm9_backend.pair(basis[exponent + 1], self.p2),
            ):
                raise ValueError("public_basis failed the L_xi recurrence")
        self.public_s = trace_public
        self.public_basis = basis
        self.max_size = len(basis) - 1

    def identity_scalar(self, identity: str) -> int:
        """Return the standard direct ``H1(ID||hid,N)`` scalar."""

        return sm9_backend.hash_to_scalar(
            1,
            identity.encode("utf-8") + bytes((self.hid,)),
        )

    def accumulate(self, identities: tuple[str, ...] | list[str]) -> bytes:
        ring = _canonical_ring(identities)
        if len(ring) > self.max_size:
            raise ValueError("ring size exceeds accumulator max_size")
        polynomial = _product_polynomial(
            self.identity_scalar(identity) for identity in ring
        )
        return self._evaluate_basis(polynomial)

    def witness(
        self,
        identities: tuple[str, ...] | list[str],
        identity: str,
    ) -> bytes:
        ring = _canonical_ring(identities)
        if identity not in ring:
            raise ValueError("identity is not in the accumulated ring")
        polynomial = _product_polynomial(
            self.identity_scalar(member) for member in ring
        )
        return self._evaluate_basis(
            _divide_by_linear_factor(
                polynomial,
                self.identity_scalar(identity),
            )
        )

    def add(self, identities: tuple[str, ...] | list[str], identity: str) -> bytes:
        """Rebuild ACC after a member addition, as required by ring rotation."""

        ring = _canonical_ring(identities)
        if identity in ring:
            raise ValueError("identity is already in the accumulated ring")
        return self.accumulate((*ring, str(identity)))

    def delete(self, identities: tuple[str, ...] | list[str], identity: str) -> bytes:
        """Rebuild ACC after revocation without exposing ``xi``."""

        ring = _canonical_ring(identities)
        if identity not in ring:
            raise ValueError("identity is not in the accumulated ring")
        remaining = tuple(member for member in ring if member != identity)
        if not remaining:
            raise ValueError("accumulator ring must not become empty")
        return self.accumulate(remaining)

    def verify_witness(
        self,
        accumulator_value: bytes,
        witness: bytes,
        identity: str,
    ) -> bool:
        identity_public = sm9_backend.g2_add(
            sm9_backend.g2_mul(self.p2, self.identity_scalar(identity)),
            self.public_s,
        )
        return sm9_backend.gt_equal(
            sm9_backend.pair(witness, identity_public),
            sm9_backend.pair(accumulator_value, self.p2),
        )

    def materialize_ring(
        self,
        identities: tuple[str, ...] | list[str],
    ) -> AccumulatorRingMaterial:
        ring = _canonical_ring(identities)
        if len(ring) > self.max_size:
            raise ValueError("ring size exceeds accumulator max_size")
        value = self.accumulate(ring)
        witnesses = {
            identity: self.witness(ring, identity) for identity in ring
        }
        g1 = sm9_backend.gt_mul(
            sm9_backend.pair(self.p1, self.sign_master_public),
            sm9_backend.pair(value, self.p2),
        )
        return AccumulatorRingMaterial(
            ring=ring,
            ring_digest=ring_digest(ring),
            value=value,
            value_digest=point_digest(value),
            g1=g1,
            witnesses=witnesses,
        )

    def g2_for_identity(
        self,
        material: AccumulatorRingMaterial,
        identity: str,
        signing_private_key: bytes,
    ) -> bytes:
        if identity not in material._g2_cache:
            material._g2_cache[identity] = sm9_backend.pair(
                sm9_backend.g1_add(
                    material.witnesses[identity],
                    signing_private_key,
                ),
                self.p2,
            )
        return material._g2_cache[identity]

    def _evaluate_basis(self, coefficients: tuple[int, ...]) -> bytes:
        if len(coefficients) > len(self.public_basis):
            raise ValueError("polynomial exceeds accumulator public_basis")
        points = (
            sm9_backend.g1_mul(point, coefficient % N)
            for coefficient, point in zip(coefficients, self.public_basis)
            if coefficient % N != 0
        )
        return _g1_sum(points)


def point_digest(point: bytes) -> str:
    return _sm3_hex(point)


def ring_digest(identities: tuple[str, ...] | list[str]) -> str:
    ring = _canonical_ring(identities)
    encoded_ring = _encode_fields(
        *(identity.encode("utf-8") for identity in ring)
    )
    return _sm3_hex(
        _encode_fields(
            b"SM9-RRS-FL/H3/RID/v2",
            encoded_ring,
        )
    )


def _canonical_ring(identities: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    ring = tuple(str(identity) for identity in identities)
    if not ring or len(set(ring)) != len(ring):
        raise ValueError("ring identities must be non-empty and unique")
    return ring


def _product_polynomial(values) -> tuple[int, ...]:
    coefficients = [1]
    for raw_value in values:
        value = int(raw_value) % N
        updated = [0] * (len(coefficients) + 1)
        for exponent, coefficient in enumerate(coefficients):
            updated[exponent] = (updated[exponent] + value * coefficient) % N
            updated[exponent + 1] = (updated[exponent + 1] + coefficient) % N
        coefficients = updated
    return tuple(coefficients)


def _divide_by_linear_factor(
    coefficients: tuple[int, ...],
    value: int,
) -> tuple[int, ...]:
    factor = int(value) % N
    quotient = [0] * (len(coefficients) - 1)
    quotient[-1] = coefficients[-1] % N
    for exponent in range(len(quotient) - 2, -1, -1):
        quotient[exponent] = (
            coefficients[exponent + 1] - factor * quotient[exponent + 1]
        ) % N
    if (coefficients[0] - factor * quotient[0]) % N != 0:
        raise ValueError("polynomial does not contain the requested factor")
    return tuple(quotient)


def _g1_sum(points) -> bytes:
    iterator = iter(points)
    try:
        total = next(iterator)
    except StopIteration as exc:
        raise ValueError("cannot evaluate an all-zero accumulator polynomial") from exc
    for point in iterator:
        total = sm9_backend.g1_add(total, point)
    return total


def _encode_fields(*fields: bytes) -> bytes:
    encoded = bytearray()
    for value in fields:
        encoded.extend(len(value).to_bytes(8, "big"))
        encoded.extend(value)
    return bytes(encoded)


def _sm3_hex(data: bytes) -> str:
    if "sm3" in hashlib.algorithms_available:
        return hashlib.new("sm3", data).hexdigest()
    return sm3.sm3_hash(list(data))


__all__ = [
    "AccumulatorRingMaterial",
    "SM9DynamicAccumulator",
    "point_digest",
    "ring_digest",
]
