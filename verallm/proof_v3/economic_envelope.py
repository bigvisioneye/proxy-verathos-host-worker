"""Canonical pre-nonce envelope construction for economic proof-v3.

The generic proof-v3 envelope contains prefill and cache-table roots even
though the economic adapter authenticates those runtime values through its
capture chain, oracle inventory, and execution anchors.  This module gives
those two fields one deterministic, verifier-replayable meaning instead of
allowing callers to insert arbitrary placeholders.
"""

from __future__ import annotations

import hashlib
import struct

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.payload import ProofV3CommitmentEnvelope
from verallm.proof_v3.profile import ExecutionSecurityProfileV3
from verallm.proof_v3.request import (
    ExecutionOutputBindingV3,
    PreExecutionRequestContextV3,
)

_PREFILL_ROOT_DOMAIN = (
    b"VERATHOS/PROOF_V3/ECONOMIC_ENVELOPE/V1/PREFILL_ROOT/SHA256"
)
_CACHE_TABLE_ROOT_DOMAIN = (
    b"VERATHOS/PROOF_V3/ECONOMIC_ENVELOPE/V1/CACHE_TABLE_ROOT/SHA256"
)

__all__ = [
    "build_economic_commitment_envelope_v3",
    "economic_cache_table_root_v3",
    "economic_prefill_root_v3",
    "validate_economic_commitment_envelope_roots_v3",
]


def _fixed32(value: bytes, name: str, *, nonzero: bool = False) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    if nonzero and value == bytes(32):
        raise ProofV3Error(f"{name} must not be the zero digest")
    return value


def _u32(value: int, name: str, *, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < (1 if positive else 0)
        or value >= 1 << 32
    ):
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


def economic_prefill_root_v3(
    *,
    precommit_context_digest: bytes,
    capture_chain_digest: bytes,
    execution_root: bytes,
    context_token_count: int,
) -> bytes:
    """Bind the economic capture chain to the validator's prompt phase."""

    return hashlib.sha256(
        _PREFILL_ROOT_DOMAIN
        + _fixed32(precommit_context_digest, "precommit_context_digest")
        + _fixed32(capture_chain_digest, "capture_chain_digest", nonzero=True)
        + _fixed32(execution_root, "execution_root", nonzero=True)
        + struct.pack(
            "<I",
            _u32(context_token_count, "context_token_count", positive=True),
        )
    ).digest()


def economic_cache_table_root_v3(
    *,
    cache_lease_digest: bytes,
    capture_chain_digest: bytes,
    execution_root: bytes,
    context_token_count: int,
    decode_token_count: int,
    cache_epoch: int,
) -> bytes:
    """Bind economic cache witnesses to one validator-issued cache lease."""

    return hashlib.sha256(
        _CACHE_TABLE_ROOT_DOMAIN
        + _fixed32(cache_lease_digest, "cache_lease_digest", nonzero=True)
        + _fixed32(capture_chain_digest, "capture_chain_digest", nonzero=True)
        + _fixed32(execution_root, "execution_root", nonzero=True)
        + struct.pack(
            "<IIQ",
            _u32(context_token_count, "context_token_count", positive=True),
            _u32(decode_token_count, "decode_token_count"),
            _u64(cache_epoch, "cache_epoch"),
        )
    ).digest()


def build_economic_commitment_envelope_v3(
    *,
    profile: ExecutionSecurityProfileV3,
    precommit_context: PreExecutionRequestContextV3,
    output_binding: ExecutionOutputBindingV3,
    execution_root: bytes,
    capture_chain_digest: bytes,
    cache_epoch: int = 0,
    allow_unverified_context: bool = False,
) -> ProofV3CommitmentEnvelope:
    """Seal one real economic precommit without caller-supplied root fields."""

    if not isinstance(profile, ExecutionSecurityProfileV3):
        raise ProofV3Error("economic envelope profile has an unexpected type")
    if not isinstance(precommit_context, PreExecutionRequestContextV3):
        raise ProofV3Error(
            "economic envelope precommit context has an unexpected type"
        )
    if not isinstance(output_binding, ExecutionOutputBindingV3):
        raise ProofV3Error("economic envelope output binding has an unexpected type")
    profile_digest = profile.digest()
    if (
        precommit_context.static_manifest_digest
        != profile.static_manifest_digest
        or precommit_context.execution_profile_digest != profile_digest
    ):
        raise ProofV3Error(
            "economic envelope context does not match the execution profile"
        )
    if (
        precommit_context.context_token_count
        > profile.max_verified_context_tokens
        and not allow_unverified_context
    ):
        raise ProofV3Error("economic envelope context exceeds the signed profile")
    if output_binding.decode_token_count > profile.max_verified_decode_tokens:
        raise ProofV3Error("economic envelope decode exceeds the signed profile")
    execution = _fixed32(execution_root, "execution_root", nonzero=True)
    capture = _fixed32(
        capture_chain_digest,
        "capture_chain_digest",
        nonzero=True,
    )
    epoch = _u64(cache_epoch, "cache_epoch")
    request_binding = precommit_context.bind_output(output_binding)
    precommit_digest = precommit_context.digest()
    return ProofV3CommitmentEnvelope(
        static_manifest_digest=profile.static_manifest_digest,
        execution_profile_digest=profile_digest,
        precommit_context_digest=precommit_digest,
        prompt_token_root=precommit_context.prompt_token_root,
        sampler_config_digest=precommit_context.sampler_config_digest,
        cache_lease_digest=precommit_context.cache_lease_digest,
        request_digest=request_binding.request_digest,
        output_token_root=output_binding.output_token_root,
        output_stream_digest=output_binding.output_stream_digest,
        execution_root=execution,
        prefill_root=economic_prefill_root_v3(
            precommit_context_digest=precommit_digest,
            capture_chain_digest=capture,
            execution_root=execution,
            context_token_count=precommit_context.context_token_count,
        ),
        cache_table_root=economic_cache_table_root_v3(
            cache_lease_digest=precommit_context.cache_lease_digest,
            capture_chain_digest=capture,
            execution_root=execution,
            context_token_count=precommit_context.context_token_count,
            decode_token_count=output_binding.decode_token_count,
            cache_epoch=epoch,
        ),
        context_token_count=precommit_context.context_token_count,
        decode_token_count=output_binding.decode_token_count,
        cache_epoch=epoch,
    )


def validate_economic_commitment_envelope_roots_v3(
    *,
    envelope: ProofV3CommitmentEnvelope,
    capture_chain_digest: bytes,
) -> None:
    """Require the economic prefill/cache roots derived before nonce reveal."""

    if not isinstance(envelope, ProofV3CommitmentEnvelope):
        raise ProofV3VerificationError(
            "economic commitment envelope has an unexpected type"
        )
    try:
        expected_prefill = economic_prefill_root_v3(
            precommit_context_digest=envelope.precommit_context_digest,
            capture_chain_digest=capture_chain_digest,
            execution_root=envelope.execution_root,
            context_token_count=envelope.context_token_count,
        )
        expected_cache = economic_cache_table_root_v3(
            cache_lease_digest=envelope.cache_lease_digest,
            capture_chain_digest=capture_chain_digest,
            execution_root=envelope.execution_root,
            context_token_count=envelope.context_token_count,
            decode_token_count=envelope.decode_token_count,
            cache_epoch=envelope.cache_epoch,
        )
    except ProofV3Error as exc:
        raise ProofV3VerificationError(
            f"economic envelope roots are malformed: {exc}"
        ) from exc
    if envelope.prefill_root != expected_prefill:
        raise ProofV3VerificationError(
            "economic envelope has an unexpected prefill root"
        )
    if envelope.cache_table_root != expected_cache:
        raise ProofV3VerificationError(
            "economic envelope has an unexpected cache-table root"
        )
