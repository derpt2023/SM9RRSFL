"""Thin, typed wrapper around the GmSSL SM9 group-operation bridge.

The protocol uses the standard SM9 group direction: paper ``G1/P1`` is the
65-byte base-curve group and paper ``G2/P2`` is the 129-byte twist group.
GmSSL's low-level pairing function accepts those arguments in reverse order;
the native bridge hides that implementation detail and exposes ``pair(g1,g2)``.
"""

from __future__ import annotations

from typing import Final


# GM/T 0044 SM9 subgroup order.  Keeping the public constant here lets the
# simulated mode use exactly the same scalar field even when the optional
# native extension is not installed.
SM9_ORDER: Final[int] = int(
    "B640000002A3A6F1D603AB4FF58EC74449F2934B18EA8BEEE56EE19CD69ECF25",
    16,
)
G1_BYTES: Final[int] = 65
G2_BYTES: Final[int] = 129
GT_BYTES: Final[int] = 384
_NATIVE_ABI_VERSION: Final[int] = 1
_EXPECTED_P1: Final[bytes] = bytes.fromhex(
    "0493de051d62bf718ff5ed0704487d01d6e1e4086909dc3280e8c4e4817c66dddd2"
    "1fe8dda4f21e607631065125c395bbc1c1c00cbfa6024350c464cd70a3ea616"
)
_EXPECTED_P2: Final[bytes] = bytes.fromhex(
    "0485aef3d078640c98597b6027b441a01ff1dd2c190f5e93c454806c11d88061413"
    "722755292130b08d2aab97fd34ec120ee265948d19c17abf9b7213baf82d65b175"
    "09b092e845c1266ba0d262cbee6ed0736a96fa347c8bd856dc76b84ebeb96a7cf"
    "28d519be3da65f3170153d278ff247efba98a71a08116215bba5c999a7c7"
)
_EXPECTED_ALICE_H1: Final[bytes] = bytes.fromhex(
    "2acc468c3926b0bdb2767e99ff26e084de9ced8dbc7d5fbf418027b667862fab"
)

try:
    from . import _native_sm9 as _native
except ImportError as exc:  # pragma: no cover - exercised on systems without GmSSL
    _native = None
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None


def _native_compatibility_error() -> Exception | None:
    if _native is None:
        return _IMPORT_ERROR
    try:
        dimensions = (
            int(_native.SCALAR_SIZE),
            int(_native.G1_SIZE),
            int(_native.G2_SIZE),
            int(_native.GT_SIZE),
        )
        if int(_native.ABI_VERSION) != _NATIVE_ABI_VERSION:
            return RuntimeError("native SM9 bridge ABI version mismatch")
        if dimensions != (32, G1_BYTES, G2_BYTES, GT_BYTES):
            return RuntimeError("native SM9 bridge element-size mismatch")
        if int.from_bytes(_native.order(), "big") != SM9_ORDER:
            return RuntimeError("native SM9 order does not match GM/T 0044")
        if _native.g1_generator() != _EXPECTED_P1:
            return RuntimeError("native SM9 P1 does not match GM/T 0044")
        if _native.g2_generator() != _EXPECTED_P2:
            return RuntimeError("native SM9 P2 does not match GM/T 0044")
        if _native.hash_to_scalar(1, b"Alice\x01") != _EXPECTED_ALICE_H1:
            return RuntimeError("native SM9 H1 known-answer test failed")
    except (AttributeError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
        error = RuntimeError("native SM9 bridge is incompatible")
        error.__cause__ = exc
        return error
    return None


_COMPATIBILITY_ERROR = _native_compatibility_error()


def available() -> bool:
    return _native is not None and _COMPATIBILITY_ERROR is None


def require_available() -> None:
    if not available():
        raise RuntimeError(
            "crypto_mode='sm9' requires the GmSSL v2 group bridge; "
            "run `python setup.py build_ext --inplace` after making "
            "GmSSL-master available"
        ) from _COMPATIBILITY_ERROR


def scalar_bytes(value: int) -> bytes:
    if not 0 <= int(value) < SM9_ORDER:
        raise ValueError("SM9 scalar is outside Z_N")
    return int(value).to_bytes(32, "big")


def hash_to_scalar(prefix: int, transcript: bytes) -> int:
    require_available()
    if prefix not in (1, 2):
        raise ValueError("SM9 H_v prefix must be 1 or 2")
    return int.from_bytes(_native.hash_to_scalar(prefix, transcript), "big")


def g1_generator() -> bytes:
    require_available()
    return _native.g1_generator()


def g1_mul(point: bytes, scalar: int) -> bytes:
    require_available()
    return _native.g1_mul(point, scalar_bytes(scalar))


def g1_add(left: bytes, right: bytes) -> bytes:
    require_available()
    return _native.g1_add(left, right)


def g1_validate(point: object) -> bool:
    return bool(
        available()
        and isinstance(point, bytes)
        and len(point) == G1_BYTES
        and _native.g1_validate(point)
    )


def g2_generator() -> bytes:
    require_available()
    return _native.g2_generator()


def g2_mul(point: bytes, scalar: int) -> bytes:
    require_available()
    return _native.g2_mul(point, scalar_bytes(scalar))


def g2_add(left: bytes, right: bytes) -> bytes:
    require_available()
    return _native.g2_add(left, right)


def g2_validate(point: object) -> bool:
    return bool(
        available()
        and isinstance(point, bytes)
        and len(point) == G2_BYTES
        and _native.g2_validate(point)
    )


def pair(g1: bytes, g2: bytes) -> bytes:
    require_available()
    return _native.pairing(g1, g2)


def gt_one() -> bytes:
    require_available()
    return _native.gt_one()


def gt_mul(left: bytes, right: bytes) -> bytes:
    require_available()
    return _native.gt_mul(left, right)


def gt_pow(value: bytes, scalar: int) -> bytes:
    require_available()
    return _native.gt_pow(value, scalar_bytes(scalar))


def gt_equal(left: object, right: object) -> bool:
    return bool(
        available()
        and isinstance(left, bytes)
        and isinstance(right, bytes)
        and len(left) == GT_BYTES
        and len(right) == GT_BYTES
        and _native.gt_equal(left, right)
    )


def gt_validate(value: object) -> bool:
    return bool(
        available()
        and isinstance(value, bytes)
        and len(value) == GT_BYTES
        and _native.gt_validate(value)
    )


def backend_name() -> str:
    return "gmssl-sm9-native-v2" if available() else "gmssl-sm9-native-v2-unavailable"


__all__ = [
    "G1_BYTES",
    "G2_BYTES",
    "GT_BYTES",
    "SM9_ORDER",
    "available",
    "backend_name",
    "g1_add",
    "g1_generator",
    "g1_mul",
    "g1_validate",
    "g2_add",
    "g2_generator",
    "g2_mul",
    "g2_validate",
    "gt_equal",
    "gt_mul",
    "gt_one",
    "gt_pow",
    "gt_validate",
    "hash_to_scalar",
    "pair",
    "require_available",
    "scalar_bytes",
]
