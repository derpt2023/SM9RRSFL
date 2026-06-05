"""Dynamic accumulator helpers for the SM9 traceable ring-signature facade.

The construction follows the accumulator component used in Xie et al. (2025):
for a ring U={ID_i}, v_i = H1(ID_i || hid, N), V=[prod_i(v_i+s)]P1, and the
signer's witness is W_i=[prod_{j!=i}(v_j+s)]P1.  The gmssl implementation names
the pairing arguments as e(G2, G1), so the paper's P1 lives in gmssl's G2 and
the paper's P2 lives in gmssl's G1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import SystemRandom
from typing import Any

from gmssl import sm9
from gmssl import optimized_curve as ec
from gmssl import optimized_field_elements as fq
from gmssl import optimized_pairing as ate


@dataclass
class AccumulatorRingMaterial:
    ring: tuple[str, ...]
    ring_digest: str
    value: Any
    value_digest: str
    g1: Any
    witnesses: dict[str, Any]
    _g2_cache: dict[str, Any] = field(default_factory=dict)


class SM9DynamicAccumulator:
    """Nguyen-style bilinear dynamic accumulator mapped onto gmssl SM9 groups."""

    def __init__(
        self,
        sign_public: tuple[Any, Any, Any, Any],
        *,
        max_size: int,
        hid: str = "01",
        seed: int | None = None,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.sign_public = sign_public
        self.p1 = sign_public[0]
        self.p2 = sign_public[1]
        self.sign_master_public = sign_public[2]
        self.hid = hid
        self.max_size = max_size
        self.secret = _seeded_scalar(seed, "accumulator-secret") if seed is not None else _random_scalar()
        self.public_s = ec.multiply(self.p2, self.secret)
        self.public_basis = tuple(
            ec.multiply(self.p1, pow(self.secret, exponent, ec.curve_order))
            for exponent in range(max_size + 1)
        )

    def identity_scalar(self, identity: str) -> int:
        """Return v_i = H1(ID_i || hid, N), matching SM9 signing KeyGen."""

        user_id = sm9.sm3_hash(sm9.str2hexbytes(identity))
        return sm9.h2rf(1, (user_id + self.hid).encode("utf-8"), ec.curve_order)

    def accumulate(self, identities: tuple[str, ...] | list[str]) -> Any:
        """Compute V=[prod_i(v_i+s)]P1 for a ring."""

        ring = _canonical_ring(identities)
        if len(ring) > self.max_size:
            raise ValueError("ring size exceeds accumulator max_size")
        return ec.multiply(self.p1, self._ring_factor(ring))

    def witness(self, identities: tuple[str, ...] | list[str], identity: str) -> Any:
        """Compute W_i=[prod_{j!=i}(v_j+s)]P1 for one ring member."""

        ring = _canonical_ring(identities)
        if identity not in ring:
            raise ValueError("identity is not in the accumulated ring")
        return ec.multiply(self.p1, self._ring_factor(ring, skip=identity))

    def add(self, accumulator_value: Any, identity: str) -> Any:
        """Dynamically add an identity to an existing accumulator value."""

        return ec.multiply(accumulator_value, self._identity_factor(identity))

    def delete(self, accumulator_value: Any, identity: str) -> Any:
        """Dynamically delete an identity from an accumulator value."""

        return ec.multiply(
            accumulator_value,
            fq.prime_field_inv(self._identity_factor(identity), ec.curve_order),
        )

    def verify_witness(self, accumulator_value: Any, witness: Any, identity: str) -> bool:
        """Verify e(W_i, [v_i]P2 + S_pub) == e(V, P2)."""

        identity_public = ec.add(ec.multiply(self.p2, self.identity_scalar(identity)), self.public_s)
        return ate.pairing(witness, identity_public) == ate.pairing(accumulator_value, self.p2)

    def materialize_ring(self, identities: tuple[str, ...] | list[str]) -> AccumulatorRingMaterial:
        """Precompute V, all W_i, and g1 for a public ring."""

        ring = _canonical_ring(identities)
        if len(ring) > self.max_size:
            raise ValueError("ring size exceeds accumulator max_size")
        factors = {identity: self._identity_factor(identity) for identity in ring}
        total = 1
        for factor in factors.values():
            total = (total * factor) % ec.curve_order
        value = ec.multiply(self.p1, total)
        witnesses = {
            identity: ec.multiply(
                self.p1,
                (total * fq.prime_field_inv(factor, ec.curve_order)) % ec.curve_order,
            )
            for identity, factor in factors.items()
        }
        g1 = ate.pairing(self.p1, self.sign_master_public) * ate.pairing(value, self.p2)
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
        signing_private_key: Any,
    ) -> Any:
        """Return g2=e(W_i+ds_i, P2), computed lazily and cached per identity."""

        if identity not in material._g2_cache:
            material._g2_cache[identity] = ate.pairing(
                ec.add(material.witnesses[identity], signing_private_key),
                self.p2,
            )
        return material._g2_cache[identity]

    def _ring_factor(self, ring: tuple[str, ...], *, skip: str | None = None) -> int:
        factor = 1
        for identity in ring:
            if identity == skip:
                continue
            factor = (factor * self._identity_factor(identity)) % ec.curve_order
        return factor

    def _identity_factor(self, identity: str) -> int:
        return (self.identity_scalar(identity) + self.secret) % ec.curve_order


def h2_scalar(ring_digest_value: str, message: str, omega: Any) -> int:
    msg_hash = sm9.sm3_hash(sm9.str2hexbytes(f"{ring_digest_value}|{message}"))
    return sm9.h2rf(2, (msg_hash + sm9.fe2sp(omega)).encode("utf-8"), ec.curve_order)


def point_digest(point: Any) -> str:
    normalized = ec.normalize(point) if not ec.is_inf(point) else point
    return sm9.sm3_hash(sm9.str2hexbytes(sm9.ec2sp(normalized)))


def ring_digest(identities: tuple[str, ...] | list[str]) -> str:
    return sm9.sm3_hash(sm9.str2hexbytes("|".join(_canonical_ring(identities))))


def _canonical_ring(identities: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(sorted(str(identity) for identity in identities))


def _random_scalar() -> int:
    return SystemRandom().randrange(1, ec.curve_order)


def _seeded_scalar(seed: int, label: str) -> int:
    digest = sm9.sm3_hash(sm9.str2hexbytes(f"{label}:{seed}"))
    return (int(digest, 16) % (ec.curve_order - 1)) + 1
