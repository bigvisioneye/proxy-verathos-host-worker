"""Strict wire envelopes for the future proof-v3 hard execution audit.

The objects here deliberately carry opaque proof components.  Parsing or
constructing them never verifies an execution proof; :mod:`verifier` rejects
them until a qualified adapter implements the global relations they describe.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error


PROTOCOL_VERSION = 3
# ``PROTOCOL_VERSION`` identifies the proof-v3 family. The commitment envelope
# has its own revision because it is emitted and parsed independently from the
# final folded proof. Revision two adds ``cache_table_root``; an older envelope
# must never be interpreted under the new cache-table ABI.
COMMITMENT_ENVELOPE_FORMAT_VERSION = 2
MAX_FOLDED_COMPONENT_BYTES = 16 << 20
MAX_FOLDED_PROOF_BYTES = 32 << 20
_ENVELOPE_MAGIC = b"V3CE"
_PROOF_MAGIC = b"V3FP"
_ENVELOPE_DIGEST_DOMAIN = b"VERATHOS/PROOF_V3/COMMITMENT_ENVELOPE/SHA256"


def _fixed(value: bytes, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    return value


def _u32(value: int, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProofV3Error(f"{name} must be an unsigned 32-bit integer")
    if value < (1 if positive else 0) or value >= 1 << 32:
        qualifier = "positive " if positive else ""
        raise ProofV3Error(f"{name} must be a {qualifier}unsigned 32-bit integer")
    return value


def _u64(value: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < 1 << 64
    ):
        raise ProofV3Error(f"{name} must be an unsigned 64-bit integer")
    return value


def _bounded(value: bytes, name: str) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise ProofV3Error(f"{name} must be non-empty bytes")
    if len(value) > MAX_FOLDED_COMPONENT_BYTES:
        raise ProofV3Error(f"{name} exceeds the component limit")
    return value


class _Reader:
    def __init__(self, encoded: bytes, maximum: int, name: str) -> None:
        if not isinstance(encoded, bytes) or len(encoded) > maximum:
            raise ProofV3Error(f"{name} byte length is out of range")
        self._encoded = encoded
        self._offset = 0
        self._name = name

    def read(self, size: int) -> bytes:
        if size < 0 or self._offset + size > len(self._encoded):
            raise ProofV3Error(f"{self._name} is truncated")
        value = self._encoded[self._offset : self._offset + size]
        self._offset += size
        return value

    def unpack(self, format_string: str) -> tuple[object, ...]:
        return struct.unpack(format_string, self.read(struct.calcsize(format_string)))

    def finish(self) -> None:
        if self._offset != len(self._encoded):
            raise ProofV3Error(f"{self._name} has trailing bytes")


@dataclass(frozen=True, slots=True)
class ProofV3CommitmentEnvelope:
    """All pre-nonce roots and request context consumed by a v3 hard proof."""

    static_manifest_digest: bytes
    execution_profile_digest: bytes
    precommit_context_digest: bytes
    prompt_token_root: bytes
    sampler_config_digest: bytes
    cache_lease_digest: bytes
    request_digest: bytes
    output_token_root: bytes
    output_stream_digest: bytes
    execution_root: bytes
    prefill_root: bytes
    cache_table_root: bytes
    context_token_count: int
    decode_token_count: int
    cache_epoch: int

    def __post_init__(self) -> None:
        _fixed(self.static_manifest_digest, "static_manifest_digest")
        _fixed(self.execution_profile_digest, "execution_profile_digest")
        _fixed(self.precommit_context_digest, "precommit_context_digest")
        _fixed(self.prompt_token_root, "prompt_token_root")
        _fixed(self.sampler_config_digest, "sampler_config_digest")
        _fixed(self.cache_lease_digest, "cache_lease_digest")
        if self.cache_lease_digest == bytes(32):
            raise ProofV3Error("cache_lease_digest must not be the zero digest")
        _fixed(self.request_digest, "request_digest")
        _fixed(self.output_token_root, "output_token_root")
        _fixed(self.output_stream_digest, "output_stream_digest")
        _fixed(self.execution_root, "execution_root")
        _fixed(self.prefill_root, "prefill_root")
        _fixed(self.cache_table_root, "cache_table_root")
        if self.cache_table_root == bytes(32):
            raise ProofV3Error("cache_table_root must not be the zero digest")
        _u32(self.context_token_count, "context_token_count", positive=True)
        _u32(self.decode_token_count, "decode_token_count")
        _u64(self.cache_epoch, "cache_epoch")

    def canonical_bytes(self) -> bytes:
        return (
            struct.pack(
                "<4sHIIQ",
                _ENVELOPE_MAGIC,
                COMMITMENT_ENVELOPE_FORMAT_VERSION,
                self.context_token_count,
                self.decode_token_count,
                self.cache_epoch,
            )
            + self.static_manifest_digest
            + self.execution_profile_digest
            + self.precommit_context_digest
            + self.prompt_token_root
            + self.sampler_config_digest
            + self.cache_lease_digest
            + self.request_digest
            + self.output_token_root
            + self.output_stream_digest
            + self.execution_root
            + self.prefill_root
            + self.cache_table_root
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_ENVELOPE_DIGEST_DOMAIN + self.canonical_bytes()).digest()

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "ProofV3CommitmentEnvelope":
        expected_size = struct.calcsize("<4sHIIQ") + 12 * 32
        reader = _Reader(encoded, expected_size, "commitment envelope")
        (
            magic,
            version,
            context_count,
            decode_count,
            cache_epoch,
        ) = reader.unpack("<4sHIIQ")
        if (
            magic != _ENVELOPE_MAGIC
            or version != COMMITMENT_ENVELOPE_FORMAT_VERSION
        ):
            raise ProofV3Error("commitment envelope header is not supported")
        result = cls(
            static_manifest_digest=reader.read(32),
            execution_profile_digest=reader.read(32),
            precommit_context_digest=reader.read(32),
            prompt_token_root=reader.read(32),
            sampler_config_digest=reader.read(32),
            cache_lease_digest=reader.read(32),
            request_digest=reader.read(32),
            output_token_root=reader.read(32),
            output_stream_digest=reader.read(32),
            execution_root=reader.read(32),
            prefill_root=reader.read(32),
            cache_table_root=reader.read(32),
            context_token_count=context_count,
            decode_token_count=decode_count,
            cache_epoch=cache_epoch,
        )
        reader.finish()
        if result.canonical_bytes() != encoded:
            raise ProofV3Error("commitment envelope is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class FoldedExecutionProofV3:
    """Opaque components whose semantics a qualified v3 adapter must verify."""

    commitment_envelope_digest: bytes
    execution_profile_digest: bytes
    linear_fold_proof: bytes
    transition_fold_proof: bytes
    final_token_proof: bytes

    def __post_init__(self) -> None:
        _fixed(self.commitment_envelope_digest, "commitment_envelope_digest")
        _fixed(self.execution_profile_digest, "execution_profile_digest")
        _bounded(self.linear_fold_proof, "linear_fold_proof")
        _bounded(self.transition_fold_proof, "transition_fold_proof")
        _bounded(self.final_token_proof, "final_token_proof")
        if (
            len(self.linear_fold_proof)
            + len(self.transition_fold_proof)
            + len(self.final_token_proof)
            > MAX_FOLDED_PROOF_BYTES
        ):
            raise ProofV3Error("folded execution proof exceeds the payload limit")

    def canonical_bytes(self) -> bytes:
        return (
            struct.pack(
                "<4sHIII",
                _PROOF_MAGIC,
                PROTOCOL_VERSION,
                len(self.linear_fold_proof),
                len(self.transition_fold_proof),
                len(self.final_token_proof),
            )
            + self.commitment_envelope_digest
            + self.execution_profile_digest
            + self.linear_fold_proof
            + self.transition_fold_proof
            + self.final_token_proof
        )

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "FoldedExecutionProofV3":
        reader = _Reader(
            encoded,
            MAX_FOLDED_PROOF_BYTES + 128,
            "folded execution proof",
        )
        magic, version, linear_length, transition_length, final_length = reader.unpack(
            "<4sHIII"
        )
        if magic != _PROOF_MAGIC or version != PROTOCOL_VERSION:
            raise ProofV3Error("folded execution proof header is not supported")
        lengths = (linear_length, transition_length, final_length)
        if any(
            not isinstance(length, int)
            or length == 0
            or length > MAX_FOLDED_COMPONENT_BYTES
            for length in lengths
        ):
            raise ProofV3Error(
                "folded execution proof component length is out of range"
            )
        if sum(lengths) > MAX_FOLDED_PROOF_BYTES:
            raise ProofV3Error("folded execution proof exceeds the payload limit")
        result = cls(
            commitment_envelope_digest=reader.read(32),
            execution_profile_digest=reader.read(32),
            linear_fold_proof=reader.read(linear_length),
            transition_fold_proof=reader.read(transition_length),
            final_token_proof=reader.read(final_length),
        )
        reader.finish()
        if result.canonical_bytes() != encoded:
            raise ProofV3Error("folded execution proof is not canonical")
        return result


def commitment_envelope_from_bytes(encoded: bytes) -> ProofV3CommitmentEnvelope:
    """Parse a canonical v3 commitment envelope."""

    return ProofV3CommitmentEnvelope.from_canonical_bytes(encoded)


def folded_execution_proof_from_bytes(encoded: bytes) -> FoldedExecutionProofV3:
    """Parse a canonical opaque v3 proof payload."""

    return FoldedExecutionProofV3.from_canonical_bytes(encoded)


__all__ = [
    "COMMITMENT_ENVELOPE_FORMAT_VERSION",
    "MAX_FOLDED_COMPONENT_BYTES",
    "MAX_FOLDED_PROOF_BYTES",
    "PROTOCOL_VERSION",
    "FoldedExecutionProofV3",
    "ProofV3CommitmentEnvelope",
    "commitment_envelope_from_bytes",
    "folded_execution_proof_from_bytes",
]
