"""Canonical execution-security profiles for proof-v3 hard execution audits.

Proof-v2 manifests authenticate static weights and a local trace format.  A
proof-v3 hard profile additionally signs the complete execution-relation ABI:
ordered graph, tensor ABI, cache/chunk semantics, static parameter bindings,
and minimum audit policy.  A profile is only an authenticated statement; a
qualified adapter must still verify that statement.
"""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.relation import (
    ExecutionRelationSpecV3,
    MAX_RELATION_SPEC_BYTES,
)


PROTOCOL_VERSION = 3
PROFILE_FORMAT_VERSION = 8
MAX_IDENTIFIER_BYTES = 128
_PROFILE_MAGIC = b"V3PR"
_PROFILE_DIGEST_DOMAIN = b"VERATHOS/PROOF_V3/EXECUTION_PROFILE/SHA256"
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,127}$")

# This name deliberately has one exact spelling. A verifier must not infer
# hard-proof capability from a broadly similar adapter label.
GLOBAL_FOLDED_EXECUTION_PROOF_SYSTEM_V3 = "global_folded_execution_v3"


def _fixed(value: bytes, name: str, *, nonzero: bool = False) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    if nonzero and value == bytes(32):
        raise ProofV3Error(f"{name} must not be the zero digest")
    return value


def _u32(value: int, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProofV3Error(f"{name} must be an unsigned 32-bit integer")
    if value < (1 if positive else 0) or value >= 1 << 32:
        qualifier = "positive " if positive else ""
        raise ProofV3Error(f"{name} must be a {qualifier}unsigned 32-bit integer")
    return value


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise ProofV3Error(f"{name} must be a string")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ProofV3Error(f"{name} must be ASCII") from exc
    if len(encoded) > MAX_IDENTIFIER_BYTES or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ProofV3Error(
            f"{name} must be a lowercase canonical identifier no longer than "
            f"{MAX_IDENTIFIER_BYTES} bytes"
        )
    return value


def _encode_identifier(value: str, name: str) -> bytes:
    encoded = _identifier(value, name).encode("ascii")
    return struct.pack("<B", len(encoded)) + encoded


class _Reader:
    def __init__(self, encoded: bytes) -> None:
        if not isinstance(encoded, bytes):
            raise ProofV3Error("profile payload must be bytes")
        self._encoded = encoded
        self._offset = 0

    def read(self, size: int) -> bytes:
        if size < 0 or self._offset + size > len(self._encoded):
            raise ProofV3Error("profile payload is truncated")
        result = self._encoded[self._offset : self._offset + size]
        self._offset += size
        return result

    def unpack(self, format_string: str) -> tuple[object, ...]:
        return struct.unpack(format_string, self.read(struct.calcsize(format_string)))

    def identifier(self, name: str) -> str:
        size = self.unpack("<B")[0]
        if not isinstance(size, int) or size == 0 or size > MAX_IDENTIFIER_BYTES:
            raise ProofV3Error(f"{name} length is out of range")
        try:
            value = self.read(size).decode("ascii")
        except UnicodeDecodeError as exc:
            raise ProofV3Error(f"{name} is not ASCII") from exc
        return _identifier(value, name)

    def bounded_bytes(self, name: str, maximum: int) -> bytes:
        size = self.unpack("<I")[0]
        if not isinstance(size, int) or size == 0 or size > maximum:
            raise ProofV3Error(f"{name} length is out of range")
        return self.read(size)

    def finish(self) -> None:
        if self._offset != len(self._encoded):
            raise ProofV3Error("profile payload has trailing bytes")


@dataclass(frozen=True, slots=True)
class ExecutionSecurityProfileV3:
    """One authority-signed, model/quantization-specific hard proof statement.

    Versions one through four are intentionally unsupported: broad capability
    flags, an unstratified graph, and a single paged-K/V-only cache ABI cannot
    express a complete model-agnostic global execution relation. Version eight
    requires a canonical :class:`ExecutionRelationSpecV3` with exact per-layer
    audit plans and per-family attention/GDN cache semantics, plus exact
    canonical sidecar and model-quantization artifacts for the dynamic proof
    backend. It also requires a signed postcommit hard/light tier-selection
    ABI and binds a constraint-system digest whose qualified artifact loader
    requires a parsed content-addressed AIR program bundle with exact
    program/layout coverage. This prevents a caller from labeling a request
    hard before its envelope is frozen or treating an opaque program label as
    a native circuit. It fails closed for every older profile.
    """

    static_manifest_digest: bytes
    static_artifact_digest: bytes
    adapter_id: str
    adapter_version: str
    proof_system_id: str
    verifier_key_digest: bytes
    quantization_semantics_id: str
    max_verified_context_tokens: int
    max_verified_decode_tokens: int
    relation_spec: ExecutionRelationSpecV3
    profile_version: int = PROFILE_FORMAT_VERSION

    def __post_init__(self) -> None:
        _fixed(
            self.static_manifest_digest,
            "static_manifest_digest",
            nonzero=True,
        )
        _fixed(
            self.static_artifact_digest,
            "static_artifact_digest",
            nonzero=True,
        )
        _identifier(self.adapter_id, "adapter_id")
        _identifier(self.adapter_version, "adapter_version")
        _identifier(self.proof_system_id, "proof_system_id")
        _fixed(self.verifier_key_digest, "verifier_key_digest")
        _identifier(self.quantization_semantics_id, "quantization_semantics_id")
        max_context = _u32(
            self.max_verified_context_tokens,
            "max_verified_context_tokens",
            positive=True,
        )
        max_decode = _u32(
            self.max_verified_decode_tokens,
            "max_verified_decode_tokens",
            positive=True,
        )
        if max_context + max_decode >= 1 << 32:
            raise ProofV3Error(
                "combined context and decode limits exceed the logical coordinate space"
            )
        _u32(self.profile_version, "profile_version", positive=True)
        if self.profile_version != PROFILE_FORMAT_VERSION:
            raise ProofV3Error("execution profile version is not supported")
        if not isinstance(self.relation_spec, ExecutionRelationSpecV3):
            raise ProofV3Error("relation_spec has an unexpected type")
        self.relation_spec.validate_for_limits(
            max_context_tokens=max_context,
            max_decode_tokens=max_decode,
        )

    @property
    def hard_execution_capable(self) -> bool:
        """Whether a complete global-relation statement is signed.

        This does not mean an adapter is registered or a proof can pass. It
        merely removes the old capability-flag ambiguity before dispatch.
        """

        return (
            self.proof_system_id == GLOBAL_FOLDED_EXECUTION_PROOF_SYSTEM_V3
            and self.relation_spec.global_execution_qualified
        )

    def require_hard_execution_capability(self) -> None:
        """Reject incomplete/economic-shell profiles before parsing proof bytes."""

        if not self.hard_execution_capable:
            raise ProofV3VerificationError(
                "execution profile is not qualified for proof-v3 hard auditing"
            )

    def canonical_bytes(self) -> bytes:
        relation_bytes = self.relation_spec.canonical_json_bytes()
        return (
            struct.pack(
                "<4sHHII",
                _PROFILE_MAGIC,
                PROTOCOL_VERSION,
                self.profile_version,
                self.max_verified_context_tokens,
                self.max_verified_decode_tokens,
            )
            + self.static_manifest_digest
            + self.static_artifact_digest
            + self.verifier_key_digest
            + _encode_identifier(self.adapter_id, "adapter_id")
            + _encode_identifier(self.adapter_version, "adapter_version")
            + _encode_identifier(self.proof_system_id, "proof_system_id")
            + _encode_identifier(
                self.quantization_semantics_id,
                "quantization_semantics_id",
            )
            + struct.pack("<I", len(relation_bytes))
            + relation_bytes
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_PROFILE_DIGEST_DOMAIN + self.canonical_bytes()).digest()

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "ExecutionSecurityProfileV3":
        reader = _Reader(encoded)
        (
            magic,
            protocol_version,
            profile_version,
            max_context,
            max_decode,
        ) = reader.unpack("<4sHHII")
        if magic != _PROFILE_MAGIC or protocol_version != PROTOCOL_VERSION:
            raise ProofV3Error("execution profile header is not supported")
        if profile_version != PROFILE_FORMAT_VERSION:
            raise ProofV3Error("execution profile version is not supported")
        result = cls(
            static_manifest_digest=reader.read(32),
            static_artifact_digest=reader.read(32),
            verifier_key_digest=reader.read(32),
            adapter_id=reader.identifier("adapter_id"),
            adapter_version=reader.identifier("adapter_version"),
            proof_system_id=reader.identifier("proof_system_id"),
            quantization_semantics_id=reader.identifier("quantization_semantics_id"),
            max_verified_context_tokens=max_context,
            max_verified_decode_tokens=max_decode,
            relation_spec=ExecutionRelationSpecV3.from_canonical_json_bytes(
                reader.bounded_bytes("relation_spec", MAX_RELATION_SPEC_BYTES)
            ),
            profile_version=profile_version,
        )
        reader.finish()
        if result.canonical_bytes() != encoded:
            raise ProofV3Error("execution profile is not canonical")
        return result


__all__ = [
    "GLOBAL_FOLDED_EXECUTION_PROOF_SYSTEM_V3",
    "PROFILE_FORMAT_VERSION",
    "PROTOCOL_VERSION",
    "ExecutionSecurityProfileV3",
]
