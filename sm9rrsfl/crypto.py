"""SM9-backed revocable ring-signature facade for experiments.

The dynamic accumulator mode follows the SM9 traceable ring-signature structure
from Xie et al. (2025): the ring is represented by a bilinear accumulator value
and signer witness, while the transmitted signature stays at constant size.  The
auditor trapdoor is retained from this project's revocation flow so existing FL
experiments can still identify confirmed malicious clients.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from random import SystemRandom
from typing import Any

import numpy as np

from gmssl import optimized_curve as ec
from gmssl import optimized_field_elements as fq
from gmssl import optimized_pairing as ate
from gmssl import sm3, sm9

from .accumulator import SM9DynamicAccumulator, h2_scalar, ring_digest

try:
    from ._native_sm3 import sm3_hexdigest as _native_sm3_hexdigest
except ImportError:
    _native_sm3_hexdigest = None

try:
    from . import _native_rrs
except ImportError:
    _native_rrs = None


@dataclass(frozen=True)
class RRSPacket:
    task_id: str
    round_id: int
    ring: tuple[str, ...]
    ring_accumulator: str
    ring_digest: str
    ring_size: int
    link_tag: str
    event_tag: str
    update_digest: str
    trapdoor: Any
    signature: Any
    crypto_mode: str
    accumulator_mode: str
    _signer_identity_hint: str


class SM9RRSContext:
    """Create, verify, and revoke experiment RRS packets.

    中文说明：真实 SM9 模式保持 SM3 语义；simulated 模式只用于快速验证
    联邦学习流程，因此使用由 OpenSSL/系统库加速的 SHA-256，避免纯 Python
    SM3 对大模型更新逐字节处理造成数小时级开销。
    """

    def __init__(
        self,
        client_ids: list[str],
        *,
        auditor_id: str = "auditor",
        ring_size: int = 5,
        crypto_mode: str = "sm9",
        accumulator_mode: str = "dynamic",
        strict_ring_verify: bool = False,
        seed: int = 0,
    ) -> None:
        if crypto_mode not in {"sm9", "simulated"}:
            raise ValueError("crypto_mode must be 'sm9' or 'simulated'")
        if accumulator_mode not in {"dynamic", "none"}:
            raise ValueError("accumulator_mode must be 'dynamic' or 'none'")
        if ring_size < 1:
            raise ValueError("ring_size must be positive")
        self.client_ids = tuple(client_ids)
        self.auditor_id = auditor_id
        self.ring_size = min(ring_size, len(client_ids))
        self.crypto_mode = crypto_mode
        self.accumulator_mode = accumulator_mode
        self.strict_ring_verify = strict_ring_verify
        self.rng = np.random.default_rng(seed)
        self._simulation_secret = sha256_hex_text(f"sim-secret:{seed}")
        self._scalar_rng = SystemRandom()
        # 真实动态累加器优先走 GmSSL 的标准 SM9 曲线和配对实现。扩展未构建时
        # 仍回退到原有 Python 实现，保证源码包在普通环境中也能运行。
        self._native_rrs_context = None
        self.rrs_backend = "simulated" if crypto_mode == "simulated" else "python-fallback"
        if crypto_mode == "sm9" and accumulator_mode == "dynamic" and _native_rrs is not None:
            self._native_rrs_context = _native_rrs.create_context(
                list(self.client_ids),
                auditor_id,
            )
            self.rrs_backend = "gmssl-native"
        # 环成员在一个实验配置内不变，提前缓存可避免每个客户端反复排序和哈希。
        if crypto_mode == "sm9":
            self._global_ring_digest_value = ring_digest(self.client_ids)
        else:
            self._global_ring_digest_value = sha256_hex_text(
                "|".join(sorted(self.client_ids))
            )
        self._simulated_ring_accumulator = sha256_hex_text(
            f"sim-acc:{self._global_ring_digest_value}"
        )
        self._link_tag_cache: dict[tuple[str, str], str] = {}

        if crypto_mode == "sm9":
            if self._native_rrs_context is None:
                self.sign_public, self.sign_master_secret = sm9.setup("sign")
                self.client_sign_keys = {
                    identity: sm9.private_key_extract(
                        "sign", self.sign_public, self.sign_master_secret, identity
                    )
                    for identity in self.client_ids
                }
            else:
                self.sign_public = None
                self.sign_master_secret = None
                self.client_sign_keys = {}
            if self._native_rrs_context is None:
                self.encrypt_public, self.encrypt_master_secret = sm9.setup("encrypt")
                self.auditor_decrypt_key = sm9.private_key_extract(
                    "encrypt", self.encrypt_public, self.encrypt_master_secret, auditor_id
                )
            else:
                # 原生上下文同时保存标准 SM9 IBE 主公钥和审计者私钥。
                self.encrypt_public = None
                self.encrypt_master_secret = None
                self.auditor_decrypt_key = None
        else:
            self.sign_public = None
            self.sign_master_secret = None
            self.encrypt_public = None
            self.encrypt_master_secret = None
            self.client_sign_keys = {}
            self.auditor_decrypt_key = None

        self.accumulator = None
        self.accumulator_material = None
        if (
            accumulator_mode == "dynamic"
            and crypto_mode == "sm9"
            and self._native_rrs_context is None
        ):
            self.accumulator = SM9DynamicAccumulator(
                self.sign_public,
                max_size=len(self.client_ids),
                seed=seed,
            )
            self.accumulator_material = self.accumulator.materialize_ring(self.client_ids)

    def create_packet(
        self,
        identity: str,
        update: np.ndarray,
        *,
        round_id: int,
        task_id: str = "mnist",
        update_digest: str | None = None,
    ) -> RRSPacket:
        """构造并签名一个数据包；细粒度性能统计可分别调用下面两个方法。"""

        unsigned = self.build_unsigned_packet(
            identity,
            update,
            round_id=round_id,
            task_id=task_id,
            update_digest=update_digest,
        )
        return self.sign_packet(identity, unsigned)

    def build_unsigned_packet(
        self,
        identity: str,
        update: np.ndarray,
        *,
        round_id: int,
        task_id: str = "mnist",
        update_digest: str | None = None,
    ) -> RRSPacket:
        """完成摘要之外的封包工作，但暂不执行签名。"""

        if identity not in self.client_ids:
            raise ValueError(f"unknown client identity: {identity}")
        ring = tuple() if self.accumulator_mode == "dynamic" else self._sample_ring(identity)
        if update_digest is None:
            update_digest = self.digest_update(update)
        ring_accumulator, current_ring_digest, current_ring_size = self._ring_commitment(ring)
        link_tag = self.link_tag(task_id, identity)
        event_tag = self._mode_hash_text(f"event:{task_id}:{round_id}:{update_digest}")[:32]
        trapdoor_plain = json.dumps(
            {"id": identity, "event": event_tag, "tag": link_tag},
            sort_keys=True,
            separators=(",", ":"),
        )
        trapdoor = self._encrypt_trapdoor(trapdoor_plain)

        return RRSPacket(
            task_id=task_id,
            round_id=round_id,
            ring=ring,
            ring_accumulator=ring_accumulator,
            ring_digest=current_ring_digest,
            ring_size=current_ring_size,
            link_tag=link_tag,
            event_tag=event_tag,
            update_digest=update_digest,
            trapdoor=trapdoor,
            signature=None,
            crypto_mode=self.crypto_mode,
            accumulator_mode=self.accumulator_mode,
            _signer_identity_hint=identity,
        )

    def sign_packet(self, identity: str, unsigned: RRSPacket) -> RRSPacket:
        """对已构造的数据包签名，便于独立测量签名阶段耗时。"""

        signature = self._sign(identity, self._message_for(unsigned))
        return replace(
            unsigned,
            signature=signature,
            _signer_identity_hint=(
                identity
                if self.crypto_mode == "simulated" or self.accumulator_mode == "none"
                else ""
            ),
        )

    def verify_packet(
        self,
        packet: RRSPacket,
        update: np.ndarray,
        *,
        update_digest: str | None = None,
    ) -> bool:
        if packet.crypto_mode != self.crypto_mode:
            return False
        if packet.accumulator_mode != self.accumulator_mode:
            return False
        if update_digest is None:
            update_digest = self.digest_update(update)
        if packet.update_digest != update_digest:
            return False
        message = self._message_for(packet)
        if self.accumulator_mode == "dynamic":
            return self._verify_accumulator_signature(packet, message)
        if packet.ring_accumulator != self._mode_hash_text("|".join(sorted(packet.ring))):
            return False
        if packet._signer_identity_hint not in packet.ring:
            return False
        if self.strict_ring_verify:
            return any(self._verify(candidate, message, packet.signature) for candidate in packet.ring)
        return self._verify(packet._signer_identity_hint, message, packet.signature)

    def revoke(self, packet: RRSPacket) -> str:
        plaintext = self._decrypt_trapdoor(packet.trapdoor)
        data = json.loads(plaintext)
        if data.get("event") != packet.event_tag or data.get("tag") != packet.link_tag:
            raise ValueError("trapdoor does not match packet")
        identity = str(data["id"])
        ring = self._effective_ring(packet)
        if identity not in ring:
            raise ValueError("revoked identity is not in the packet ring")
        return identity

    def _sample_ring(self, identity: str) -> tuple[str, ...]:
        others = [client for client in self.client_ids if client != identity]
        need = max(0, self.ring_size - 1)
        if need:
            sampled = list(self.rng.choice(others, size=min(need, len(others)), replace=False))
        else:
            sampled = []
        ring = sampled + [identity]
        self.rng.shuffle(ring)
        return tuple(str(item) for item in ring)

    def _encrypt_trapdoor(self, plaintext: str) -> Any:
        if self.crypto_mode == "sm9":
            if self._native_rrs_context is not None:
                assert _native_rrs is not None
                return _native_rrs.encrypt_trapdoor(
                    self._native_rrs_context,
                    plaintext.encode("utf-8"),
                )
            return sm9.kem_dem_enc(self.encrypt_public, self.auditor_id, plaintext, 32)
        digest = sha256_hex_text(f"{self._simulation_secret}:{plaintext}")
        return {"plain": plaintext, "digest": digest}

    def _decrypt_trapdoor(self, trapdoor: Any) -> str:
        if self.crypto_mode == "sm9":
            if self._native_rrs_context is not None:
                assert _native_rrs is not None
                return str(
                    _native_rrs.decrypt_trapdoor(
                        self._native_rrs_context,
                        trapdoor,
                    )
                )
            plaintext = sm9.kem_dem_dec(
                self.encrypt_public, self.auditor_id, self.auditor_decrypt_key, trapdoor, 32
            )
            if plaintext is False:
                raise ValueError("SM9 trapdoor decryption failed")
            return str(plaintext)
        plaintext = str(trapdoor["plain"])
        expected = sha256_hex_text(f"{self._simulation_secret}:{plaintext}")
        if trapdoor.get("digest") != expected:
            raise ValueError("simulated trapdoor integrity check failed")
        return plaintext

    def _sign(self, identity: str, message: str) -> Any:
        if self.accumulator_mode == "dynamic":
            if self.crypto_mode == "sm9":
                return self._sign_accumulator(identity, message)
            return sha256_hex_text(
                f"acc-sig:{self._simulation_secret}:{identity}:{self._global_ring_digest()}:{message}"
            )
        if self.crypto_mode == "sm9":
            return sm9.sign(self.sign_public, self.client_sign_keys[identity], message)
        return sha256_hex_text(f"sig:{self._simulation_secret}:{identity}:{message}")

    def _verify(self, identity: str, message: str, signature: Any) -> bool:
        if self.crypto_mode == "sm9":
            return bool(sm9.verify(self.sign_public, identity, message, signature))
        expected = sha256_hex_text(f"sig:{self._simulation_secret}:{identity}:{message}")
        return signature == expected

    def _sign_accumulator(self, identity: str, message: str) -> Any:
        if self._native_rrs_context is not None:
            assert _native_rrs is not None
            return _native_rrs.sign(
                self._native_rrs_context,
                identity,
                self._global_ring_digest(),
                message,
            )
        assert self.accumulator is not None and self.accumulator_material is not None
        material = self.accumulator_material
        witness = material.witnesses[identity]
        signing_private_key = self.client_sign_keys[identity]
        g1 = material.g1
        g2 = self.accumulator.g2_for_identity(material, identity, signing_private_key)
        while True:
            r1 = self._random_scalar()
            r2 = self._random_scalar()
            omega = (g1 ** r1) * (g2 ** r2)
            h = h2_scalar(material.ring_digest, message, omega)
            denominator = (r1 - h) % ec.curve_order
            if denominator != 0:
                break
        signer_scalar = self.accumulator.identity_scalar(identity)
        r_component = ec.multiply(witness, denominator)
        s_component = ec.multiply(signing_private_key, denominator)
        t_scalar = (
            r2 * fq.prime_field_inv(denominator, ec.curve_order) + signer_scalar
        ) % ec.curve_order
        t_component = ec.multiply(self.sign_public[1], t_scalar)
        return h, r_component, s_component, t_component

    def _verify_accumulator_signature(self, packet: RRSPacket, message: str) -> bool:
        if packet.ring_digest != self._global_ring_digest():
            return False
        if packet.ring_size != len(self.client_ids):
            return False
        if self.crypto_mode == "simulated":
            if packet.ring_accumulator != self._simulated_ring_accumulator:
                return False
            expected = sha256_hex_text(
                "acc-sig:"
                f"{self._simulation_secret}:{packet._signer_identity_hint}:"
                f"{self._global_ring_digest()}:{message}"
            )
            return packet.signature == expected
        if self._native_rrs_context is not None:
            assert _native_rrs is not None
            if packet.ring_accumulator != _native_rrs.accumulator_digest(
                self._native_rrs_context
            ):
                return False
            try:
                return bool(
                    _native_rrs.verify(
                        self._native_rrs_context,
                        packet.ring_digest,
                        message,
                        packet.signature,
                    )
                )
            except (TypeError, ValueError):
                return False
        assert self.accumulator is not None and self.accumulator_material is not None
        material = self.accumulator_material
        if packet.ring_accumulator != material.value_digest:
            return False
        try:
            h, r_component, s_component, t_component = packet.signature
        except (TypeError, ValueError):
            return False
        if not 1 <= h < ec.curve_order:
            return False
        if not ec.is_on_curve(r_component, ec.b2):
            return False
        if not ec.is_on_curve(s_component, ec.b2):
            return False
        if not ec.is_on_curve(t_component, ec.b):
            return False
        omega = (
            ate.pairing(r_component, ec.add(self.accumulator.public_s, t_component))
            * ate.pairing(s_component, ec.add(self.sign_public[2], t_component))
            * (material.g1 ** h)
        )
        return h2_scalar(material.ring_digest, message, omega) == h

    def _random_scalar(self) -> int:
        return self._scalar_rng.randrange(1, ec.curve_order)

    def _ring_commitment(self, ring: tuple[str, ...]) -> tuple[str, str, int]:
        if self.accumulator_mode == "dynamic":
            ring_digest_value = self._global_ring_digest()
            if self.crypto_mode == "sm9":
                if self._native_rrs_context is not None:
                    assert _native_rrs is not None
                    return (
                        _native_rrs.accumulator_digest(self._native_rrs_context),
                        ring_digest_value,
                        len(self.client_ids),
                    )
                assert self.accumulator_material is not None
                return (
                    self.accumulator_material.value_digest,
                    ring_digest_value,
                    len(self.client_ids),
                )
            return self._simulated_ring_accumulator, ring_digest_value, len(self.client_ids)
        ring_text = "|".join(sorted(ring))
        return self._mode_hash_text(ring_text), self._mode_hash_text(ring_text), len(ring)

    def _effective_ring(self, packet: RRSPacket) -> tuple[str, ...]:
        if packet.accumulator_mode == "dynamic":
            return self.client_ids
        return packet.ring

    def _global_ring_digest(self) -> str:
        return self._global_ring_digest_value

    def digest_update(self, update: np.ndarray) -> str:
        """按当前密码模式计算更新摘要，避免调用方误用慢速后端。"""

        algorithm = "sha256" if self.crypto_mode == "simulated" else "sm3"
        return digest_update(update, algorithm=algorithm)

    def link_tag(self, task_id: str, identity: str) -> str:
        """缓存与轮次无关的客户端链接标签。"""

        key = (task_id, identity)
        cached = self._link_tag_cache.get(key)
        if cached is None:
            cached = self._mode_hash_text(f"link:{task_id}:{identity}")[:32]
            self._link_tag_cache[key] = cached
        return cached

    def _mode_hash_text(self, text: str) -> str:
        if self.crypto_mode == "simulated":
            return sha256_hex_text(text)
        return sm3_hex_text(text)

    def _message_for(self, packet: RRSPacket) -> str:
        trapdoor_digest = self._mode_hash_text(repr(packet.trapdoor))
        ring_descriptor = packet.ring_digest if packet.accumulator_mode == "dynamic" else ",".join(packet.ring)
        parts = [
            packet.task_id,
            str(packet.round_id),
            ring_descriptor,
            str(packet.ring_size),
            packet.ring_accumulator,
            packet.link_tag,
            packet.event_tag,
            packet.update_digest,
            trapdoor_digest,
        ]
        return "|".join(parts)


def digest_update(update: np.ndarray, *, algorithm: str = "sm3") -> str:
    """计算模型更新摘要。

    ``sm3`` 用于真实密码流程；``sha256`` 用于 simulated 快速流程。二者都
    先固定为连续 float32 字节序列，保证相同更新得到稳定摘要。
    """

    # memoryview 直接暴露连续 float32 缓冲区，避免为 7 MB 级更新再复制一份 bytes。
    array = np.ascontiguousarray(update, dtype=np.float32)
    data = memoryview(array).cast("B")
    if algorithm == "sha256":
        return sha256_hex_bytes(data)
    if algorithm != "sm3":
        raise ValueError("algorithm must be 'sm3' or 'sha256'")
    return sm3_hex_bytes(data)


def sm3_hex_text(text: str) -> str:
    return sm3_hex_bytes(text.encode("utf-8"))


def sm3_hex_bytes(data: bytes | memoryview) -> str:
    # 优先使用本项目 C 扩展；其次尝试 OpenSSL，最后才走纯 Python 兼容实现。
    if _native_sm3_hexdigest is not None:
        return _native_sm3_hexdigest(data)
    if "sm3" in hashlib.algorithms_available:
        return hashlib.new("sm3", data).hexdigest()
    return sm3.sm3_hash(list(data))


def sm3_backend_name() -> str:
    """返回当前实际使用的 SM3 后端，便于实验日志核对。"""

    if _native_sm3_hexdigest is not None:
        return "native-extension"
    if "sm3" in hashlib.algorithms_available:
        return "hashlib-openssl"
    return "python-fallback"


def rrs_backend_name() -> str:
    """返回真实动态环签名将使用的后端。"""

    return "gmssl-native" if _native_rrs is not None else "python-fallback"


def sha256_hex_text(text: str) -> str:
    return sha256_hex_bytes(text.encode("utf-8"))


def sha256_hex_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()
