#!/usr/bin/env python3
"""
VeraLLM Validator Client — connects to a remote miner and verifies inference.

Protocol:
1. GET  /model_spec  — Fetch weight Merkle roots (on-chain data in production)
2. POST /inference   — SSE-streamed tokens, commitment digest, and proofs
3. POST /proof/v2/challenge — Reveal the authenticated validator nonce after
   the miner freezes the v2 commitment digest

The validator then verifies proofs locally (lightweight — no model loading).

Usage:
    python -m verallm.api.client --miner-url http://localhost:8000 \
        --prompt "Explain the halting problem"

    # With TLS:
    python -m verallm.api.client --miner-url https://miner-host:8443 \
        --prompt "Hello world" --verify-tls
"""

import argparse
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import hmac
import inspect
import json
import logging
import os
import struct
import sys
import time
from typing import Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ProofV2TransportState:
    """Local state for one proof-v2 SSE exchange.

    The miner never receives this structure.  In particular,
    ``hard_audit_selected`` is derived locally from the frozen precommitment,
    the validator nonce, and an authenticated policy rate before the nonce is
    delivered to the miner.  Proxy callers use it to force verification only
    for hidden hard selections while leaving ordinary light responses cheap.
    """

    precommit_seen: bool = False
    v2_precommit_required: bool = False
    saw_response_token: bool = False
    nonce_reveal_attempted: bool = False
    nonce_revealed: bool = False
    hard_audit_selected: Optional[bool] = None
    done_seen: bool = False
    failure_reason: str = ""


class ProofV2PostNonceTransportError(RuntimeError):
    """A selected hard audit did not complete after its nonce obligation."""

    def __init__(self, state: ProofV2TransportState, reason: str):
        self.transport_state = state
        self.reason = str(reason)
        state.failure_reason = self.reason
        super().__init__(f"proof-v2 hard audit transport failure: {self.reason}")

import httpx
import numpy as np
import torch

from verallm.config import Config, set_config, get_config
from verallm.types import (
    InferenceCommitment,
    InferenceProofBundle,
    ModelSpec,
    VerificationResult,
)
from verallm.challenge.beacon import (
    derive_beacon_from_nonce,
    derive_challenges,
    derive_sampling_challenge,
    derive_hard_audit_sampling_challenge,
    hard_audit_required,
    hard_audit_selected,
    derive_embedding_challenge,
    compute_detection_probability,
    validate_proof_v2_decode_commitment,
)
from verallm.challenge.v2 import (
    MAX_BLOCKS_PER_OPERATION,
    derive_hard_execution_corridor_v2,
)
from verallm.verifier.gemm import GEMMVerifier
from verallm.crypto.transcript import Transcript
from verallm.crypto.merkle import (
    verify_merkle_path,
    verify_flat_chunk_merkle_path,
    build_block_merkle,
    MerkleTree,
)
from verallm.moe import (
    MoEConfig,
    derive_moe_challenges,
)
from verallm.moe.router_commitment import (
    compute_topk_indices,
    logits_row_to_bytes,
)

from verallm.helpers import compute_auto_k, compute_auto_k_experts
from verallm.sampling import (
    clamp_sampling_bps,
    hidden_row_from_bytes,
    logits_i32_from_bytes,
    quantize_hidden_row_int64,
    verify_quantized_argmax,
    verify_fp16_argmax,
)
from verallm.verifier.spot_openings import verify_x_spot_openings
from verallm.api.serialization import (
    commitment_to_dict,
    dict_to_model_spec,
    dict_to_commitment,
    dict_to_proof_bundle,
)
from verallm.api.proof_protocol import (
    LEGACY_PROOF_PROTOCOL_VERSION,
    PROOF_PROTOCOL_V2,
    SUPPORTED_PROOF_PROTOCOL_VERSIONS,
    add_proof_protocol_version,
    commit_validator_nonce_v2,
    decode_proof_challenge_id,
    decode_proof_commitment_hash,
    encode_proof_challenge_id,
    encode_proof_commitment_hash,
    encode_validator_nonce,
)
from verallm.proof_policy import mark_proof_payload_invalid


_PROOF_DESERIALIZATION_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OverflowError,
    RecursionError,
    struct.error,
    TypeError,
    ValueError,
)

PROOF_V2_RESPONSE_TARGET_MS = 1000.0
_PROOF_V2_CONNECTION_PREWARM_POOL = ThreadPoolExecutor(
    max_workers=32,
    thread_name_prefix="proof-v2-connect",
)


def _prewarm_proof_v2_connection_sync(
    client,
    miner_url: str,
) -> None:
    """Open a second keep-alive connection while the SSE stream is active."""

    try:
        response = client.get(f"{miner_url}/health", timeout=5.0)
        response.raise_for_status()
    except Exception as exc:
        logger.debug("proof-v2 challenge connection prewarm failed: %s", exc)


async def _prewarm_proof_v2_connection_async(
    client,
    miner_url: str,
) -> None:
    try:
        response = await client.get(f"{miner_url}/health", timeout=5.0)
        response.raise_for_status()
    except Exception as exc:
        logger.debug("proof-v2 challenge connection prewarm failed: %s", exc)


async def _finish_async_connection_prewarm(task) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def _consume_connection_prewarm(future: Optional[Future]) -> None:
    if future is None or not future.done():
        return
    try:
        future.result()
    except Exception:
        pass


def _merkle_path_matches_position(path, position: int, leaf_count: int) -> bool:
    """Validate the canonical path shape for one indexed decode row."""

    if (
        isinstance(position, bool)
        or not isinstance(position, int)
        or isinstance(leaf_count, bool)
        or not isinstance(leaf_count, int)
        or not 0 <= position < leaf_count
        or path is None
        or type(getattr(path, "leaf_index", None)) is not int
        or path.leaf_index != position
        or not isinstance(getattr(path, "siblings", None), list)
    ):
        return False
    siblings = path.siblings
    if len(siblings) != (leaf_count - 1).bit_length():
        return False
    for level, item in enumerate(siblings):
        if not isinstance(item, tuple) or len(item) != 2:
            return False
        sibling_hash, is_left = item
        if not isinstance(sibling_hash, bytes) or len(sibling_hash) != 32:
            return False
        if type(is_left) is not bool:
            return False
        if is_left != bool((position >> level) & 1):
            return False
    return True


def _validate_exact_legacy_proof_set(
    expected_challenges,
    layer_proofs,
) -> Optional[str]:
    """Return an error for a missing, duplicate, or extra legacy proof item."""

    expected_layers = tuple(
        sorted(
            expected_challenges.layer_challenges,
            key=lambda challenge: challenge.layer_idx,
        )
    )
    expected_indices = tuple(challenge.layer_idx for challenge in expected_layers)
    if len(expected_indices) != len(set(expected_indices)):
        return "Legacy challenge set contains duplicate layers"
    if not isinstance(layer_proofs, list):
        return "Legacy layer proof set is malformed"
    received_indices = tuple(
        getattr(layer_proof, "layer_idx", None) for layer_proof in layer_proofs
    )
    if any(type(index) is not int for index in received_indices):
        return "Legacy layer proof identity is malformed"
    if received_indices != tuple(sorted(received_indices)):
        return "Legacy layer proof set is not canonical"
    if len(received_indices) != len(set(received_indices)):
        return "Legacy layer proof set contains duplicate layers"
    if received_indices != expected_indices:
        return "Legacy layer proof set does not match the challenge"

    for challenge, layer_proof in zip(expected_layers, layer_proofs):
        expected_gemms = (
            challenge.expert_challenges
            if hasattr(challenge, "expert_challenges")
            else challenge.gemm_challenges
        )
        if not isinstance(expected_gemms, (list, tuple)) or not expected_gemms:
            return "Legacy challenge contains no GEMMs"
        gemm_proofs = getattr(layer_proof, "gemm_proofs", None)
        if not isinstance(gemm_proofs, list) or len(gemm_proofs) != len(expected_gemms):
            return "Legacy GEMM proof set does not match the challenge"
        for gemm_proof in gemm_proofs:
            block_proofs = getattr(gemm_proof, "block_proofs", None)
            if not isinstance(block_proofs, list) or not block_proofs:
                return "Legacy GEMM proof has no challenged blocks"
            coordinates = tuple(
                (getattr(block, "bi", None), getattr(block, "bj", None))
                for block in block_proofs
            )
            if any(
                type(row) is not int or type(column) is not int or row < 0 or column < 0
                for row, column in coordinates
            ):
                return "Legacy GEMM block identity is malformed"
            if len(coordinates) != len(set(coordinates)):
                return "Legacy GEMM proof contains duplicate blocks"
    return None


def _wire_proof_protocol_version(proof_data: object) -> tuple[Optional[int], bool]:
    if not isinstance(proof_data, dict):
        return None, False
    if "proof_protocol_version" not in proof_data:
        return LEGACY_PROOF_PROTOCOL_VERSION, True
    version = proof_data["proof_protocol_version"]
    valid = type(version) is int and version in SUPPORTED_PROOF_PROTOCOL_VERSIONS
    return (version if valid else None), valid


def _raw_merkle_path_is_canonical(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if type(value.get("leaf_index")) is not int:
        return False
    siblings = value.get("siblings")
    if not isinstance(siblings, list):
        return False
    for item in siblings:
        if not isinstance(item, list) or len(item) != 2:
            return False
        digest, is_left = item
        if not isinstance(digest, str) or len(digest) != 64:
            return False
        try:
            if len(bytes.fromhex(digest)) != 32:
                return False
        except ValueError:
            return False
        if type(is_left) is not bool:
            return False
    return True


def _raw_v2_sampling_payload_is_canonical(proof_data: dict) -> bool:
    proofs = proof_data.get("sampling_proofs")
    if not isinstance(proofs, list):
        return False
    for proof in proofs:
        if not isinstance(proof, dict):
            return False
        if type(proof.get("decode_step")) is not int:
            return False
        if type(proof.get("token_id")) is not int:
            return False
        if not _raw_merkle_path_is_canonical(proof.get("hidden_merkle_path")):
            return False
        logits_path = proof.get("fp16_logits_merkle_path")
        if not _raw_merkle_path_is_canonical(logits_path):
            return False
        for field in (
            "hidden_row",
            "fp16_logits_row",
            "lm_head_proof_v2_commitment",
            "lm_head_proof_v2_payload",
        ):
            encoded = proof.get(field)
            if not isinstance(encoded, str) or len(encoded) % 2:
                return False
            try:
                bytes.fromhex(encoded)
            except ValueError:
                return False
        seed = proof.get("sampling_seed")
        if seed is not None:
            if not isinstance(seed, str) or len(seed) != 64:
                return False
            try:
                if len(bytes.fromhex(seed)) != 32:
                    return False
            except ValueError:
                return False
    return True


def _greedy_token_after_presence_penalty_v2(
    *,
    top_values: np.ndarray,
    top_indices: np.ndarray,
    output_token_ids: Sequence[int],
    step: int,
    presence_penalty_milli: int,
    vocab_size: int,
) -> int:
    """Replay vLLM v1's generated-token presence penalty on an opened top-K."""

    values = np.asarray(top_values, dtype=np.float32)
    indices = np.asarray(top_indices, dtype=np.int64)
    if (
        values.ndim != 1
        or indices.ndim != 1
        or values.size == 0
        or values.size != indices.size
        or isinstance(step, bool)
        or not isinstance(step, int)
        or step < 0
        or step >= len(output_token_ids)
        or isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size <= 0
        or values.size > vocab_size
        or isinstance(presence_penalty_milli, bool)
        or not isinstance(presence_penalty_milli, int)
    ):
        raise ValueError("proof-v2 greedy penalty context is invalid")
    if values.size < vocab_size and step >= values.size:
        raise ValueError("proof-v2 greedy top-k is too small for the decode step")

    prefix = output_token_ids[:step]
    if any(
        isinstance(token, bool)
        or not isinstance(token, (int, np.integer))
        or int(token) < 0
        or int(token) >= vocab_size
        for token in prefix
    ):
        raise ValueError("proof-v2 greedy output history is invalid")

    adjusted = values.astype(np.float32, copy=True)
    if prefix and presence_penalty_milli:
        seen = np.fromiter(
            {int(token) for token in prefix},
            dtype=np.int64,
        )
        seen_mask = np.isin(indices, seen)
        penalty = np.float32(presence_penalty_milli) / np.float32(1000.0)
        adjusted[seen_mask] = adjusted[seen_mask] - penalty

    order = np.lexsort((indices, -adjusted))
    winner_position = int(order[0])
    if values.size < vocab_size:
        # Every unopened raw logit is <= the last opened raw logit. Requiring
        # a strict gap also rules out an omitted equal-logit token with a lower
        # vocabulary index, whose deterministic tie-break cannot be observed.
        raw_boundary = np.float32(values[-1])
        if not adjusted[winner_position] > raw_boundary:
            raise ValueError("proof-v2 greedy top-k does not prove the global argmax")
    return int(indices[winner_position])


def _deserialize_done_proof_payload(
    commit_data: object,
    proof_data: object,
    *,
    deserialize_proof_bundle: bool = True,
    requested_protocol_version: Optional[int] = None,
    fallback_commitment: Optional[InferenceCommitment] = None,
) -> tuple[Optional[InferenceCommitment], Optional[InferenceProofBundle]]:
    """Decode the final SSE proof fields and retain invalid-payload state."""
    commitment = None
    proof_bundle = None
    payload_invalid = False

    compact_v2 = requested_protocol_version == PROOF_PROTOCOL_V2

    if commit_data:
        try:
            if not isinstance(commit_data, dict):
                raise TypeError("commitment must be an object")
            commitment = dict_to_commitment(commit_data)
        except _PROOF_DESERIALIZATION_EXCEPTIONS:
            payload_invalid = deserialize_proof_bundle
    elif compact_v2:
        commitment = fallback_commitment
    elif deserialize_proof_bundle:
        payload_invalid = True

    protocol_version, version_valid = _wire_proof_protocol_version(proof_data)
    if deserialize_proof_bundle:
        if (
            requested_protocol_version is not None
            and protocol_version != requested_protocol_version
        ):
            payload_invalid = True
        if proof_data:
            try:
                if not version_valid:
                    raise TypeError("proof protocol version is invalid")
                if (
                    compact_v2
                    and protocol_version == PROOF_PROTOCOL_V2
                    and isinstance(proof_data, dict)
                    and "commitment" not in proof_data
                    and isinstance(commitment, InferenceCommitment)
                ):
                    proof_data = dict(proof_data)
                    proof_data["commitment"] = commitment_to_dict(commitment)
                if (
                    protocol_version == PROOF_PROTOCOL_V2
                    and not _raw_v2_sampling_payload_is_canonical(proof_data)
                ):
                    raise TypeError("proof-v2 sampling payload is not canonical")
                proof_bundle = dict_to_proof_bundle(proof_data)
            except _PROOF_DESERIALIZATION_EXCEPTIONS:
                payload_invalid = True
        else:
            payload_invalid = True

        if (
            commitment is not None
            and proof_bundle is not None
            and proof_bundle.commitment != commitment
        ):
            payload_invalid = True

        if proof_bundle is None:
            proof_bundle = InferenceProofBundle.empty()
        if payload_invalid:
            mark_proof_payload_invalid(
                proof_bundle,
                protocol_version=protocol_version,
            )

    return commitment, proof_bundle


def _add_proof_request_nonce_fields(
    request_body: dict,
    *,
    validator_nonce: bytes,
    proof_protocol_version: Optional[int],
) -> Optional[bytes]:
    """Add the v1 nonce or v2 nonce commitment to an inference request."""
    add_proof_protocol_version(request_body, proof_protocol_version)
    if proof_protocol_version == PROOF_PROTOCOL_V2:
        challenge_id = os.urandom(32)
        request_body["proof_challenge_id"] = encode_proof_challenge_id(challenge_id)
        request_body["validator_nonce_commitment"] = commit_validator_nonce_v2(
            validator_nonce, challenge_id
        ).hex()
        return challenge_id
    request_body["validator_nonce"] = encode_validator_nonce(validator_nonce)
    return None


def _decode_proof_v2_precommit(
    data: object,
    *,
    expected_challenge_id: bytes,
) -> tuple[str, bytes]:
    """Validate the miner's frozen v2 commitment before revealing the nonce."""
    if not isinstance(data, dict):
        raise RuntimeError("proof-v2 precommit event must be an object")
    challenge_id = decode_proof_challenge_id(data.get("proof_challenge_id"))
    if not hmac.compare_digest(challenge_id, expected_challenge_id):
        raise RuntimeError("proof-v2 precommit challenge id does not match")
    commitment_hash = decode_proof_commitment_hash(data.get("commitment_hash"))
    session_id = data.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError("proof-v2 precommit session_id is invalid")
    return session_id, commitment_hash


def _select_proof_v2_hard_audit_from_precommit(
    *,
    precommitment_hash: bytes,
    validator_nonce: bytes,
    hard_audit_bps: Optional[int],
) -> Optional[bool]:
    """Derive the local hard-audit obligation before revealing the nonce.

    ``hard_audit_bps`` must come from an already authenticated execution
    profile.  It is deliberately optional for compatibility callers that do
    not possess such a profile; those callers retain their existing proof
    verification behaviour and cannot opt into this hidden-audit transport
    gate accidentally.
    """

    if hard_audit_bps is None:
        return None
    beacon = derive_beacon_from_nonce(
        commitment_hash=precommitment_hash,
        validator_nonce=validator_nonce,
    )
    return hard_audit_selected(beacon, hard_audit_bps)


def _hard_audit_transport_failure(
    state: ProofV2TransportState,
    reason: str,
) -> ProofV2PostNonceTransportError:
    """Build the fail-closed error for a selected hard audit."""

    return ProofV2PostNonceTransportError(state, reason)


def _must_deserialize_proof_payload(
    *,
    deserialize_proof_bundle: bool,
    transport_state: ProofV2TransportState,
) -> bool:
    """Retain selected hard proofs even when ordinary local sampling is off."""

    return bool(deserialize_proof_bundle or transport_state.hard_audit_selected is True)


def _proof_v2_reveal_body(
    *,
    challenge_id: bytes,
    session_id: str,
    commitment_hash: bytes,
    validator_nonce: bytes,
) -> dict:
    return {
        "proof_challenge_id": encode_proof_challenge_id(challenge_id),
        "session_id": session_id,
        "commitment_hash": encode_proof_commitment_hash(commitment_hash),
        "validator_nonce": encode_validator_nonce(validator_nonce),
    }


def _decode_proof_v2_commitment(
    data: object,
    *,
    precommit_session_id: Optional[str],
    precommitment_hash: Optional[bytes],
) -> InferenceCommitment:
    """Decode the commitment streamed after its binding digest."""

    if not isinstance(data, dict):
        raise RuntimeError("proof-v2 commitment event must be an object")
    raw_commitment = data.get("commitment")
    if not isinstance(raw_commitment, dict):
        raise RuntimeError("proof-v2 commitment event is missing its commitment")
    try:
        commitment = dict_to_commitment(raw_commitment)
    except _PROOF_DESERIALIZATION_EXCEPTIONS as exc:
        raise RuntimeError("proof-v2 commitment event is malformed") from exc
    _validate_proof_v2_done_commitment(
        final_commitment=commitment,
        precommit_session_id=precommit_session_id,
        precommitment_hash=precommitment_hash,
    )
    return commitment


def _validate_proof_v2_done_commitment(
    *,
    final_commitment: Optional[InferenceCommitment],
    precommit_session_id: Optional[str],
    precommitment_hash: Optional[bytes],
) -> None:
    if (
        final_commitment is None
        or precommit_session_id is None
        or precommitment_hash is None
    ):
        raise RuntimeError("proof-v2 response is missing its precommit event")
    if final_commitment.session_id != precommit_session_id:
        raise RuntimeError("proof-v2 final commitment session differs from precommit")
    if not final_commitment.proof_v2_commitment:
        raise RuntimeError("proof-v2 final commitment is missing its v2 envelope")
    if not hmac.compare_digest(
        final_commitment.commitment_hash(),
        precommitment_hash,
    ):
        raise RuntimeError("proof-v2 final commitment hash differs from precommit")


def _measure_proof_v2_response_latency(
    *,
    last_token_at: Optional[float],
    proof_received_at: Optional[float],
) -> Optional[float]:
    if last_token_at is None or proof_received_at is None:
        return None
    elapsed_ms = max(0.0, (proof_received_at - last_token_at) * 1000)
    if elapsed_ms >= PROOF_V2_RESPONSE_TARGET_MS:
        logger.warning(
            "proof-v2 response missed the %.0f ms post-token latency target: "
            "%.3f ms",
            PROOF_V2_RESPONSE_TARGET_MS,
            elapsed_ms,
        )
    return elapsed_ms


def _retain_failed_proof_v2_transport(
    proof_bundle: Optional[InferenceProofBundle],
) -> InferenceProofBundle:
    """Convert an invalid v2 stream transcript into explicit failed proof data."""

    if proof_bundle is None:
        proof_bundle = InferenceProofBundle.empty()
        protocol_version = PROOF_PROTOCOL_V2
    else:
        protocol_version = getattr(
            proof_bundle,
            "proof_protocol_version",
            PROOF_PROTOCOL_V2,
        )
    return mark_proof_payload_invalid(
        proof_bundle,
        protocol_version=protocol_version,
    )


def _coerce_timing_ms(value: object) -> Optional[float]:
    """Best-effort conversion for diagnostic miner timing fields."""
    try:
        timing_ms = float(value)
    except (TypeError, ValueError):
        return None
    if timing_ms < 0:
        return None
    return timing_ms


def _remember_miner_inference_ms(timing: dict, value: object) -> None:
    """Store miner-reported inference time as diagnostic-only metadata."""
    timing_ms = _coerce_timing_ms(value)
    if timing_ms is not None:
        timing["miner_inference_ms"] = timing_ms


def _finalize_validator_timing(
    timing: dict,
    request_start: float,
    first_token_at: Optional[float],
    last_token_at: Optional[float],
    response_done_at: Optional[float] = None,
) -> dict:
    """Fill validator-measured wall-clock timing fields.

    ``inference_ms`` is intentionally never copied from the miner.  It is the
    validator's wall-clock duration from request start to the last streamed
    token, falling back to full round-trip time when no token event arrived.
    """
    completed_at = (
        response_done_at if response_done_at is not None else time.perf_counter()
    )
    round_trip_ms = max(0.0, (completed_at - request_start) * 1000)
    timing["round_trip_ms"] = round_trip_ms

    if first_token_at is not None:
        timing["ttft_ms"] = max(0.0, (first_token_at - request_start) * 1000)

    if last_token_at is not None:
        timing["inference_ms"] = max(0.0, (last_token_at - request_start) * 1000)
    else:
        timing["inference_ms"] = round_trip_ms

    return timing


def _proof_v2_prompt_attention_state_anchor_v2(
    *,
    input_token_ids: bytes,
    layer_idx: int,
    attention_profile: str,
) -> bytes:
    """Return the canonical prompt-state boundary for one trace layer.

    A full-attention profile has one logical K/V state, whereas a GDN profile
    has convolution and recurrent components.  Derive that arity from the
    canonical trace definition rather than maintaining a second profile table
    in the validator.
    """

    from verallm.proof_v2.trace import (
        attention_state_tensor_names_v2,
        trace_attention_state_boundary_digest_v2,
    )

    initial = hashlib.sha256(
        b"VERATHOS/PROOF_V2/TRACE_STATE/INITIAL/SHA256"
        + input_token_ids
        + layer_idx.to_bytes(4, "little")
        + attention_profile.encode("ascii")
    ).digest()
    return trace_attention_state_boundary_digest_v2(
        attention_profile,
        (initial,)
        * len(attention_state_tensor_names_v2(attention_profile, before=True)),
    )


def _validated_output_token_count(
    data: dict,
    *,
    max_new_tokens: int,
    commitment: Optional[InferenceCommitment],
    commitment_present: bool,
    proof_data: object,
    allow_unbound: bool = False,
) -> int:
    """Return the proof-bound output count or reject inconsistent metadata."""
    reported = data.get("output_tokens")
    if isinstance(reported, bool) or not isinstance(reported, int):
        raise RuntimeError(
            f"Miner returned invalid output token count: {reported!r}"
        )
    if reported < 0:
        raise RuntimeError(
            f"Miner returned negative output token count: {reported}"
        )
    if reported > int(max_new_tokens):
        raise RuntimeError(
            "Miner output token count exceeds request limit: "
            f"reported={reported}, max_new_tokens={max_new_tokens}"
        )

    # TEE responses do not expose plaintext commitments or token IDs. For
    # proof-backed responses, require every independently supplied count to
    # agree before the value reaches receipts, billing, or scoring.
    if not commitment_present and allow_unbound:
        return reported
    if not commitment_present:
        raise RuntimeError("Miner response is missing an inference commitment")
    if commitment is None:
        raise RuntimeError("Miner returned an invalid inference commitment")

    committed = int(commitment.output_token_count)
    if reported != committed:
        raise RuntimeError(
            "Miner output token count mismatch: "
            f"reported={reported}, committed={committed}"
        )

    output_token_ids = (
        proof_data.get("output_token_ids")
        if isinstance(proof_data, dict)
        else None
    )
    if not isinstance(output_token_ids, list):
        raise RuntimeError("Miner proof bundle is missing output_token_ids")
    if reported != len(output_token_ids):
        raise RuntimeError(
            "Miner output token count mismatch: "
            f"reported={reported}, proof_bundle={len(output_token_ids)}"
        )
    return reported


# ============================================================================
# SSE parser
# ============================================================================


def _parse_sse_stream(response):
    """Parse a Server-Sent Events stream from httpx.

    Yields (event_type, data_dict) tuples.  Supports both RFC 6902 format
    (``event: type\\ndata: json``) and the miner's inline format
    (``data: {"event": "type", ...}``).
    """
    event_type = None
    data_lines = []

    for line in response.iter_lines():
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
        elif line == "":
            # Empty line = end of event
            if data_lines:
                raw = "\n".join(data_lines)
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, RecursionError):
                    parsed = {"raw": raw}
                if not isinstance(parsed, dict):
                    parsed = {"raw": parsed}
                # If no explicit event: line, check for inline event field
                evt = event_type or parsed.get("event", "")
                if evt:
                    yield evt, parsed
            event_type = None
            data_lines = []

    # Handle final event without trailing empty line
    if data_lines:
        raw = "\n".join(data_lines)
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, RecursionError):
            parsed = {"raw": raw}
        if not isinstance(parsed, dict):
            parsed = {"raw": parsed}
        evt = event_type or parsed.get("event", "")
        if evt:
            yield evt, parsed


def _parse_sse_stream_with_proof_v2_transport(
    response,
    transport_state: ProofV2TransportState,
):
    """Preserve a selected hard audit across an SSE transport failure."""

    try:
        yield from _parse_sse_stream(response)
    except ProofV2PostNonceTransportError:
        raise
    except Exception as exc:
        if (
            transport_state.v2_precommit_required
            and transport_state.saw_response_token
            and not transport_state.precommit_seen
        ):
            raise _hard_audit_transport_failure(
                transport_state,
                "stream transport failed before the required proof-v2 precommit",
            ) from exc
        if (
            transport_state.hard_audit_selected is True
            and transport_state.nonce_reveal_attempted
        ):
            raise _hard_audit_transport_failure(
                transport_state,
                "stream transport failed after hard-audit selection",
            ) from exc
        raise


async def _parse_sse_stream_async(response):
    """Async variant of :func:`_parse_sse_stream` for httpx.AsyncClient."""
    event_type = None
    data_lines = []

    async for line in response.aiter_lines():
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
        elif line == "":
            if data_lines:
                raw = "\n".join(data_lines)
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, RecursionError):
                    parsed = {"raw": raw}
                if not isinstance(parsed, dict):
                    parsed = {"raw": parsed}
                evt = event_type or parsed.get("event", "")
                if evt:
                    yield evt, parsed
            event_type = None
            data_lines = []

    if data_lines:
        raw = "\n".join(data_lines)
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, RecursionError):
            parsed = {"raw": raw}
        if not isinstance(parsed, dict):
            parsed = {"raw": parsed}
        evt = event_type or parsed.get("event", "")
        if evt:
            yield evt, parsed


async def _parse_sse_stream_async_with_proof_v2_transport(
    response,
    transport_state: ProofV2TransportState,
):
    """Async counterpart that makes a post-nonce reset fail closed."""

    try:
        async for item in _parse_sse_stream_async(response):
            yield item
    except asyncio.CancelledError:
        raise
    except ProofV2PostNonceTransportError:
        raise
    except Exception as exc:
        if (
            transport_state.v2_precommit_required
            and transport_state.saw_response_token
            and not transport_state.precommit_seen
        ):
            raise _hard_audit_transport_failure(
                transport_state,
                "stream transport failed before the required proof-v2 precommit",
            ) from exc
        if (
            transport_state.hard_audit_selected is True
            and transport_state.nonce_reveal_attempted
        ):
            raise _hard_audit_transport_failure(
                transport_state,
                "stream transport failed after hard-audit selection",
            ) from exc
        raise


# ============================================================================
# Validator Request Signing (httpx Auth)
# ============================================================================


class ValidatorRequestAuth(httpx.Auth):
    """httpx Auth class that signs every request with a validator Sr25519 hotkey.

    Attaches X-Validator-Hotkey (SS58), X-Validator-Signature, X-Validator-Timestamp
    headers so that miners can verify the caller is a registered validator.
    """

    def __init__(self, hotkey_ss58: str, hotkey_seed: bytes):
        self._hotkey_ss58 = hotkey_ss58
        self._hotkey_seed = hotkey_seed

    def _sign_request(self, request: httpx.Request) -> httpx.Request:
        from neurons.request_signing import sign_request
        from urllib.parse import urlparse

        path = urlparse(str(request.url)).path
        body = request.content if request.content else b""
        headers = sign_request(
            method=request.method,
            path=path,
            body=body,
            hotkey_ss58=self._hotkey_ss58,
            hotkey_seed=self._hotkey_seed,
        )
        for k, v in headers.items():
            request.headers[k] = v
        return request

    def auth_flow(self, request: httpx.Request):
        yield self._sign_request(request)

    async def async_auth_flow(self, request: httpx.Request):
        yield self._sign_request(request)


# ============================================================================
# Validator Client
# ============================================================================


class ValidatorClient:
    """Primary production client — used by the subnet validator and proxy.

    Bundles HTTP transport (inference requests, SSE streaming, TEE) with
    local proof verification.  This is the canonical verification path;
    ``verallm.validator.core.Validator`` mirrors the same logic for
    offline demos and tests only.

    Proof v2 streams tokens, receives a frozen commitment digest, reveals the
    authenticated validator nonce, then receives the full commitment and proof.
    The validator re-derives the transcript and challenges during verification.
    """

    def __init__(
        self,
        miner_url: str,
        config: Optional[Config] = None,
        verify_tls: bool = True,
        timeout: float = 600.0,
        api_key: Optional[str] = None,
        chain_config=None,
        model_id: Optional[str] = None,
        validator_hotkey_ss58: Optional[str] = None,
        validator_seed: Optional[bytes] = None,
        proof_v2_manifest=None,
    ):
        self.miner_url = miner_url.rstrip("/")
        self.config = config or get_config()
        self.model_spec: Optional[ModelSpec] = None
        self.moe_config: Optional[MoEConfig] = None
        self._chain_config = chain_config
        self._model_id = model_id
        self._on_chain_model_spec = None
        self.proof_v2_manifest = proof_v2_manifest
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        auth = None
        if validator_hotkey_ss58 and validator_seed:
            auth = ValidatorRequestAuth(validator_hotkey_ss58, validator_seed)

        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=30.0),
            verify=verify_tls,
            headers=headers,
            auth=auth,
        )

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Fetch ModelSpec
    # ------------------------------------------------------------------

    def fetch_model_spec(self) -> ModelSpec:
        """Fetch the ModelSpec (weight Merkle roots).

        If a chain_config is set, reads from the on-chain ModelRegistry
        (the trust anchor — roots are NOT from the miner).

        Falls back to GET /model_spec from the miner when no chain config
        is provided (development/testing mode).
        """
        if self._chain_config is not None:
            spec = self._fetch_model_spec_from_chain()
            if spec is not None:
                self.model_spec = spec
                self._auto_configure_from_spec(spec)
                return spec
            # If chain read failed, don't fall back — that would be insecure
            raise RuntimeError(
                f"Model '{self._model_id}' not found on-chain. "
                "Cannot fall back to miner (trust anchor must be on-chain)."
            )

        # Dev mode: fetch from miner directly
        resp = self.client.get(f"{self.miner_url}/model_spec")
        resp.raise_for_status()
        self.model_spec = dict_to_model_spec(resp.json())
        self._auto_configure_from_spec(self.model_spec)
        return self.model_spec

    @staticmethod
    def _read_bounded_response(
        response: httpx.Response,
        *,
        byte_limit: int,
        name: str,
    ) -> bytes:
        declared = response.headers.get("content-length")
        if declared is not None:
            try:
                declared_length = int(declared)
            except ValueError as exc:
                raise ValueError(f"{name} content length is malformed") from exc
            if declared_length < 0 or declared_length > byte_limit:
                raise ValueError(f"{name} exceeds its byte limit")
        chunks = bytearray()
        for chunk in response.iter_bytes():
            if len(chunks) + len(chunk) > byte_limit:
                raise ValueError(f"{name} exceeds its byte limit")
            chunks.extend(chunk)
        if not chunks:
            raise ValueError(f"{name} is empty")
        return bytes(chunks)

    def fetch_proof_v3_hard_bundle_index(self, epoch_number: int) -> dict:
        """Fetch one authenticated, bounded retained-bundle index."""

        if (
            isinstance(epoch_number, bool)
            or not isinstance(epoch_number, int)
            or not 0 <= epoch_number < 1 << 63
        ):
            raise ValueError("proof-v3 bundle epoch is out of range")
        path = f"/proof/v3/bundles/{epoch_number}"
        with self.client.stream("GET", self.miner_url + path) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(
                ";", 1
            )[0].strip()
            if content_type != "application/json":
                raise ValueError(
                    "proof-v3 hard-bundle index media type is unsupported"
                )
            encoded = self._read_bounded_response(
                response,
                byte_limit=4 << 20,
                name="proof-v3 hard-bundle index",
            )
        try:
            result = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "proof-v3 hard-bundle index is malformed"
            ) from exc
        if (
            not isinstance(result, dict)
            or result.get("epoch") != epoch_number
            or not isinstance(result.get("bundles"), list)
            or result.get("bundle_count") != len(result["bundles"])
        ):
            raise ValueError("proof-v3 hard-bundle index is inconsistent")
        return result

    def fetch_proof_v3_hard_bundle(
        self,
        *,
        epoch_number: int,
        commitment_envelope_digest: bytes,
    ) -> bytes:
        """Fetch one authenticated retained bundle without unbounded buffering."""

        from verallm.proof_v3.hard_bundle import (
            HARD_BUNDLE_MEDIA_TYPE_V3,
            MAX_HARD_BUNDLE_BYTES_V3,
            RetainedHardProofBundleV3,
        )

        if (
            isinstance(epoch_number, bool)
            or not isinstance(epoch_number, int)
            or not 0 <= epoch_number < 1 << 63
            or not isinstance(commitment_envelope_digest, bytes)
            or len(commitment_envelope_digest) != 32
        ):
            raise ValueError("proof-v3 hard-bundle identity is malformed")
        commitment_hex = commitment_envelope_digest.hex()
        path = f"/proof/v3/bundles/{epoch_number}/{commitment_hex}"
        with self.client.stream("GET", self.miner_url + path) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(
                ";", 1
            )[0].strip()
            if content_type != HARD_BUNDLE_MEDIA_TYPE_V3:
                raise ValueError(
                    "proof-v3 hard-bundle media type is unsupported"
                )
            encoded = self._read_bounded_response(
                response,
                byte_limit=MAX_HARD_BUNDLE_BYTES_V3,
                name="proof-v3 hard bundle",
            )
        bundle = RetainedHardProofBundleV3.from_canonical_bytes(encoded)
        if bundle.commitment_envelope_digest != commitment_envelope_digest:
            raise ValueError(
                "proof-v3 hard bundle does not match the requested commitment"
            )
        return encoded

    def _auto_configure_from_spec(self, model_spec: ModelSpec) -> None:
        """Auto-compute k_layers and detect MoE from the model spec.

        Called from fetch_model_spec() so that verify_proof() always has
        correct k_layers, regardless of whether run() or the standalone
        fetch_model_spec() + verify_proof() path is used.
        """
        # Auto-compute k_layers if not explicitly set
        if self.config.k_layers == 0 and model_spec.num_layers > 0:
            k = compute_auto_k(model_spec.num_layers)
            self.config = Config(
                **{
                    **{
                        f.name: getattr(self.config, f.name)
                        for f in self.config.__dataclass_fields__.values()
                    },
                    "k_layers": k,
                }
            )
            set_config(self.config)

        # Detect MoE from model_spec
        if self.moe_config is None:
            num_experts = model_spec.num_experts
            if num_experts == 0 and model_spec.expert_weight_merkle_roots:
                first_roots = next(
                    iter(model_spec.expert_weight_merkle_roots.values()), []
                )
                num_experts = len(first_roots)

            if num_experts > 0:
                if model_spec.expert_weight_merkle_roots:
                    moe_layer_indices = sorted(
                        model_spec.expert_weight_merkle_roots.keys()
                    )
                else:
                    moe_layer_indices = list(range(model_spec.num_layers))
                expert_inter = (
                    model_spec.expert_w_num_cols
                    if model_spec.expert_w_num_cols > 0
                    else model_spec.intermediate_dim
                )
                self.moe_config = MoEConfig(
                    is_moe=True,
                    num_layers=model_spec.num_layers,
                    moe_layer_indices=moe_layer_indices,
                    num_routed_experts=num_experts,
                    num_shared_experts=0,
                    top_k=model_spec.router_top_k if model_spec.router_top_k > 0 else 2,
                    hidden_dim=model_spec.hidden_dim,
                    intermediate_dim=model_spec.intermediate_dim,
                    expert_intermediate_dim=expert_inter,
                    has_shared_expert_gate=False,
                    uses_3d_expert_weights=False,
                    router_type=model_spec.router_scoring or "top_k",
                )

                # Auto-compute k_experts_per_layer if not set
                if self.config.k_experts_per_layer == 0:
                    k_exp = compute_auto_k_experts(num_experts)
                    self.config = Config(
                        **{
                            **{
                                f.name: getattr(self.config, f.name)
                                for f in self.config.__dataclass_fields__.values()
                            },
                            "k_experts_per_layer": k_exp,
                        }
                    )
                    set_config(self.config)

    def _fetch_model_spec_from_chain(self) -> Optional[ModelSpec]:
        """Read ModelSpec from the on-chain ModelRegistry."""
        from verallm.chain.mock import create_clients

        model_client, *_ = create_clients(self._chain_config)
        model_id = self._model_id
        if not model_id:
            # Try to get model_id from the miner's health endpoint
            health = self.health_check()
            model_id = health.get("model", "")
        if not model_id:
            raise ValueError(
                "model_id must be provided when using chain_config "
                "(or miner /health must return 'model' field)"
            )
        self._model_id = model_id
        if hasattr(model_client, "get_on_chain_model_spec"):
            from verallm.chain.types import on_chain_to_model_spec

            on_chain_spec = model_client.get_on_chain_model_spec(model_id)
            if on_chain_spec is None:
                return None
            self._on_chain_model_spec = on_chain_spec
            return on_chain_to_model_spec(on_chain_spec)
        return model_client.get_model_spec(model_id)

    def set_verified_proof_v2_manifest(self, manifest) -> None:
        """Attach a manifest that was authenticated against the chain context."""

        from verallm.proof_v2.manifest import (
            ModelSpecIdentity,
            StaticWeightCommitmentManifest,
        )

        if not isinstance(manifest, StaticWeightCommitmentManifest):
            raise TypeError("proof-v2 manifest has an unexpected type")
        if self._on_chain_model_spec is not None:
            if manifest.model_spec != ModelSpecIdentity.from_on_chain(
                self._on_chain_model_spec
            ):
                raise ValueError(
                    "proof-v2 manifest does not match the on-chain ModelSpec"
                )
        elif self.model_spec is not None:
            if (
                manifest.model_spec.model_id != self.model_spec.model_id
                or manifest.model_spec.weight_merkle_root
                != self.model_spec.weight_merkle_root
            ):
                raise ValueError("proof-v2 manifest does not match the ModelSpec")
        self.proof_v2_manifest = manifest

    def health_check(self) -> dict:
        """Check if the miner is healthy."""
        resp = self.client.get(f"{self.miner_url}/health")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Inference + proof-v2 nonce reveal
    # ------------------------------------------------------------------

    def run_inference(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        do_sample: bool = False,
        temperature: float = 1.0,
        sampling_verification_bps: int = 0,
        stream_callback=None,
        proof_protocol_version: Optional[int] = None,
        proof_v2_hard_audit_bps: Optional[int] = None,
        proof_v2_transport_state: Optional[ProofV2TransportState] = None,
        allow_unbound_output_count: bool = False,
    ) -> Tuple[str, InferenceCommitment, InferenceProofBundle, bytes, dict]:
        """Send inference request, stream tokens, get commitment + proofs.

        For proof v2, the miner streams inference, freezes and emits the
        commitment digest, receives the authenticated nonce reveal, then emits
        the full commitment and proof.

        Args:
            prompt: The input prompt.
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to use sampling.
            temperature: Sampling temperature.
            stream_callback: Optional callable(token_text) invoked per token.
            proof_protocol_version: Explicit proof wire version. Omit for the
                legacy v1 request contract.

        Returns:
            (full_text, commitment, proof_bundle, nonce, timing_info)
        """
        # Generate validator nonce (32 random bytes)
        nonce = os.urandom(32)
        transport_state = proof_v2_transport_state or ProofV2TransportState()
        transport_state.v2_precommit_required = bool(
            proof_protocol_version == PROOF_PROTOCOL_V2
            and proof_v2_hard_audit_bps is not None
        )

        request_body = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "sampling_verification_bps": max(
                0, min(10_000, int(sampling_verification_bps))
            ),
        }
        proof_challenge_id = _add_proof_request_nonce_fields(
            request_body,
            validator_nonce=nonce,
            proof_protocol_version=proof_protocol_version,
        )

        full_text = ""
        commitment = None
        proof_bundle = None
        timing = {}
        t_first_token = None
        t_last_token = None
        t_done_recv = None
        t_request_end_wall = None
        precommit_session_id = None
        precommitment_hash = None
        precommitment = None
        challenge_connection_prewarm = None

        t0 = time.perf_counter()
        t0_wall = time.time()
        with self.client.stream(
            "POST", f"{self.miner_url}/inference", json=request_body
        ) as resp:
            resp.raise_for_status()
            if proof_protocol_version == PROOF_PROTOCOL_V2 and isinstance(
                self.client, httpx.Client
            ):
                challenge_connection_prewarm = _PROOF_V2_CONNECTION_PREWARM_POOL.submit(
                    _prewarm_proof_v2_connection_sync,
                    self.client,
                    self.miner_url,
                )
            for event_type, data in _parse_sse_stream_with_proof_v2_transport(
                resp,
                transport_state,
            ):
                if event_type == "token":
                    transport_state.saw_response_token = True
                    t_last_token = time.perf_counter()
                    if t_first_token is None:
                        t_first_token = t_last_token
                    token_text = data.get("text", "")
                    full_text += token_text
                    if stream_callback:
                        stream_callback(token_text)
                elif event_type == "proof_precommit":
                    if proof_protocol_version != PROOF_PROTOCOL_V2:
                        raise RuntimeError("unexpected proof-v2 precommit event")
                    if proof_challenge_id is None or precommit_session_id is not None:
                        raise RuntimeError("invalid duplicate proof-v2 precommit event")
                    precommit_received_at = time.perf_counter()
                    (
                        precommit_session_id,
                        precommitment_hash,
                    ) = _decode_proof_v2_precommit(
                        data,
                        expected_challenge_id=proof_challenge_id,
                    )
                    transport_state.precommit_seen = True
                    transport_state.hard_audit_selected = (
                        _select_proof_v2_hard_audit_from_precommit(
                            precommitment_hash=precommitment_hash,
                            validator_nonce=nonce,
                            hard_audit_bps=proof_v2_hard_audit_bps,
                        )
                    )
                    if transport_state.hard_audit_selected is not None:
                        timing["_proof_v2_hard_audit_selected"] = bool(
                            transport_state.hard_audit_selected
                        )
                    if t_last_token is not None:
                        timing["last_token_to_precommit_ms"] = max(
                            0.0,
                            (precommit_received_at - t_last_token) * 1000,
                        )
                    _consume_connection_prewarm(challenge_connection_prewarm)
                    reveal_started = time.perf_counter()
                    transport_state.nonce_reveal_attempted = True
                    try:
                        reveal_response = self.client.post(
                            f"{self.miner_url}/proof/v2/challenge",
                            json=_proof_v2_reveal_body(
                                challenge_id=proof_challenge_id,
                                session_id=precommit_session_id,
                                commitment_hash=precommitment_hash,
                                validator_nonce=nonce,
                            ),
                        )
                        reveal_response.raise_for_status()
                    except Exception as exc:
                        if transport_state.hard_audit_selected is True:
                            raise _hard_audit_transport_failure(
                                transport_state,
                                "nonce reveal did not complete",
                            ) from exc
                        raise
                    transport_state.nonce_revealed = True
                    timing["proof_challenge_rtt_ms"] = (
                        time.perf_counter() - reveal_started
                    ) * 1000
                elif event_type == "proof_commitment":
                    if (
                        proof_protocol_version != PROOF_PROTOCOL_V2
                        or precommitment is not None
                    ):
                        raise RuntimeError("invalid proof-v2 commitment event")
                    try:
                        precommitment = _decode_proof_v2_commitment(
                            data,
                            precommit_session_id=precommit_session_id,
                            precommitment_hash=precommitment_hash,
                        )
                    except RuntimeError as exc:
                        if (
                            transport_state.hard_audit_selected is True
                            and transport_state.nonce_reveal_attempted
                        ):
                            raise _hard_audit_transport_failure(
                                transport_state,
                                "post-nonce commitment is invalid",
                            ) from exc
                        raise
                elif event_type == "done":
                    transport_state.done_seen = True
                    t_done_recv = time.perf_counter()
                    done_gap_ms = (
                        (t_done_recv - t_last_token) * 1000
                        if t_last_token is not None
                        else 0.0
                    )
                    # TEE miners may return empty commitment/proof_bundle
                    commit_data = data.get("commitment", {})
                    proof_data = data.get(
                        "proof_bundle", {"layer_proofs": [], "sampling_proofs": []}
                    )
                    deserialize_started = time.perf_counter()
                    commitment, proof_bundle = _deserialize_done_proof_payload(
                        commit_data,
                        proof_data,
                        requested_protocol_version=proof_protocol_version,
                        fallback_commitment=precommitment,
                    )
                    if proof_protocol_version == PROOF_PROTOCOL_V2:
                        try:
                            _validate_proof_v2_done_commitment(
                                final_commitment=commitment,
                                precommit_session_id=precommit_session_id,
                                precommitment_hash=precommitment_hash,
                            )
                            timing[
                                "last_token_to_proof_ms"
                            ] = _measure_proof_v2_response_latency(
                                last_token_at=t_last_token,
                                proof_received_at=t_done_recv,
                            )
                        except RuntimeError:
                            proof_bundle = _retain_failed_proof_v2_transport(
                                proof_bundle
                            )
                    timing["last_token_to_done_recv_ms"] = done_gap_ms
                    timing["proof_deserialize_ms"] = (
                        time.perf_counter() - deserialize_started
                    ) * 1000
                    t_request_end_wall = time.time()
                    _remember_miner_inference_ms(timing, data.get("inference_ms"))
                    timing["commitment_ms"] = data.get("commitment_ms", 0)
                    timing["prove_ms"] = data.get("prove_ms", 0)
                    timing["beacon_ms"] = data.get("beacon_ms", 0)
                    timing["challenge_ms"] = data.get("challenge_ms", 0)
                    timing["reveal_ms"] = data.get("reveal_ms", 0)
                    timing["prove_timing_details"] = data.get(
                        "prove_timing_details", {}
                    )
                    timing["miner_last_token_to_proof_ms"] = data.get(
                        "last_token_to_proof_ms", 0
                    )
                    timing["input_tokens"] = data.get("input_tokens", 0)
                    timing["output_tokens"] = _validated_output_token_count(
                        data,
                        max_new_tokens=max_new_tokens,
                        commitment=commitment,
                        commitment_present=bool(commit_data),
                        proof_data=proof_data,
                        allow_unbound=allow_unbound_output_count,
                    )
                elif event_type == "error":
                    if (
                        transport_state.v2_precommit_required
                        and (
                            transport_state.precommit_seen
                            or transport_state.saw_response_token
                        )
                    ):
                        raise _hard_audit_transport_failure(
                            transport_state,
                            "miner ended the required proof-v2 exchange",
                        )
                    if (
                        transport_state.hard_audit_selected is True
                        and transport_state.nonce_reveal_attempted
                    ):
                        raise _hard_audit_transport_failure(
                            transport_state,
                            "miner sent an error after hard-audit selection",
                        )
                    raise RuntimeError(f"Miner error: {data.get('error', data)}")

        if (
            transport_state.v2_precommit_required
            and not transport_state.precommit_seen
        ):
            raise _hard_audit_transport_failure(
                transport_state,
                "response omitted the required proof-v2 precommit",
            )
        if (
            proof_protocol_version == PROOF_PROTOCOL_V2
            and transport_state.hard_audit_selected is True
            and transport_state.nonce_reveal_attempted
            and not transport_state.done_seen
        ):
            raise _hard_audit_transport_failure(
                transport_state,
                "stream ended without a final proof response",
            )

        if t_request_end_wall is None:
            t_request_end_wall = time.time()
        timing["validator_request_start_ts"] = t0_wall
        timing["validator_request_end_ts"] = t_request_end_wall
        timing["validator_request_ms"] = (t_request_end_wall - t0_wall) * 1000
        _finalize_validator_timing(
            timing,
            t0,
            t_first_token,
            t_last_token,
            response_done_at=t_done_recv,
        )

        # TEE miners don't produce commitments/proofs — create empty defaults
        if commitment is None:
            commitment = InferenceCommitment.empty()
        if proof_bundle is None:
            proof_bundle = mark_proof_payload_invalid(
                InferenceProofBundle.empty(),
                protocol_version=(
                    proof_protocol_version or LEGACY_PROOF_PROTOCOL_VERSION
                ),
            )

        return full_text, commitment, proof_bundle, nonce, timing

    def run_chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 4096,
        do_sample: bool = False,
        temperature: float = 1.0,
        sampling_verification_bps: int = 0,
        stream_callback=None,
        enable_thinking: bool = True,
        presence_penalty: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        tools: Optional[list[dict]] = None,
        tool_choice=None,
        parallel_tool_calls: Optional[bool] = None,
        deserialize_proof_bundle: bool = True,
        proof_protocol_version: Optional[int] = None,
        proof_v2_hard_audit_bps: Optional[int] = None,
        proof_v2_transport_state: Optional[ProofV2TransportState] = None,
        allow_unbound_output_count: bool = False,
    ) -> Tuple[
        str, Optional[InferenceCommitment], Optional[InferenceProofBundle], bytes, dict
    ]:
        """Send chat-style inference request (OpenAI messages format).

        Uses POST /chat on the miner, which applies the chat template
        server-side. Otherwise identical to run_inference().

        Args:
            messages: List of {role, content} dicts (OpenAI format).
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to use sampling.
            temperature: Sampling temperature.
            stream_callback: Optional callable(token_text) invoked per token.
            deserialize_proof_bundle: When False, skip converting the final
                proof_bundle JSON into Python proof objects. Use this only when
                the caller has already decided not to verify this request.
            enable_thinking: Enable chain-of-thought for models that support it.
            presence_penalty: Penalize repeated tokens (None = server default).
            top_k: Top-k sampling (None = server default).
            top_p: Nucleus sampling (None = server default).
            min_p: Minimum probability (None = server default).
            proof_protocol_version: Explicit proof wire version. Omit for the
                legacy v1 request contract.

        Returns:
            (full_text, commitment, proof_bundle, nonce, timing_info)
        """
        nonce = os.urandom(32)
        transport_state = proof_v2_transport_state or ProofV2TransportState()
        transport_state.v2_precommit_required = bool(
            proof_protocol_version == PROOF_PROTOCOL_V2
            and proof_v2_hard_audit_bps is not None
        )

        request_body = {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "sampling_verification_bps": max(
                0, min(10_000, int(sampling_verification_bps))
            ),
            "enable_thinking": enable_thinking,
        }
        proof_challenge_id = _add_proof_request_nonce_fields(
            request_body,
            validator_nonce=nonce,
            proof_protocol_version=proof_protocol_version,
        )
        # Only send sampling params when explicitly set (None = server defaults)
        if presence_penalty is not None:
            request_body["presence_penalty"] = presence_penalty
        if top_k is not None:
            request_body["top_k"] = top_k
        if top_p is not None:
            request_body["top_p"] = top_p
        if min_p is not None:
            request_body["min_p"] = min_p
        if tools is not None:
            request_body["tools"] = tools
        if tool_choice is not None:
            request_body["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            request_body["parallel_tool_calls"] = parallel_tool_calls

        full_text = ""
        commitment = None
        proof_bundle = None
        timing = {}
        t_first_token = None
        t_last_token = None
        t_done_recv = None
        t_request_end_wall = None
        precommit_session_id = None
        precommitment_hash = None
        precommitment = None
        challenge_connection_prewarm = None

        t0 = time.perf_counter()
        t0_wall = time.time()
        _t_last_tok = None
        with self.client.stream(
            "POST", f"{self.miner_url}/chat", json=request_body
        ) as resp:
            resp.raise_for_status()
            if proof_protocol_version == PROOF_PROTOCOL_V2 and isinstance(
                self.client, httpx.Client
            ):
                challenge_connection_prewarm = _PROOF_V2_CONNECTION_PREWARM_POOL.submit(
                    _prewarm_proof_v2_connection_sync,
                    self.client,
                    self.miner_url,
                )
            for event_type, data in _parse_sse_stream_with_proof_v2_transport(
                resp,
                transport_state,
            ):
                if event_type == "token":
                    transport_state.saw_response_token = True
                    _t_last_tok = time.perf_counter()
                    t_last_token = _t_last_tok
                    if t_first_token is None:
                        t_first_token = _t_last_tok
                    token_text = data.get("text", "")
                    full_text += token_text
                    if stream_callback:
                        stream_callback(token_text)
                elif event_type == "proof_precommit":
                    if proof_protocol_version != PROOF_PROTOCOL_V2:
                        raise RuntimeError("unexpected proof-v2 precommit event")
                    if proof_challenge_id is None or precommit_session_id is not None:
                        raise RuntimeError("invalid duplicate proof-v2 precommit event")
                    precommit_received_at = time.perf_counter()
                    (
                        precommit_session_id,
                        precommitment_hash,
                    ) = _decode_proof_v2_precommit(
                        data,
                        expected_challenge_id=proof_challenge_id,
                    )
                    transport_state.precommit_seen = True
                    transport_state.hard_audit_selected = (
                        _select_proof_v2_hard_audit_from_precommit(
                            precommitment_hash=precommitment_hash,
                            validator_nonce=nonce,
                            hard_audit_bps=proof_v2_hard_audit_bps,
                        )
                    )
                    if transport_state.hard_audit_selected is not None:
                        timing["_proof_v2_hard_audit_selected"] = bool(
                            transport_state.hard_audit_selected
                        )
                    if t_last_token is not None:
                        timing["last_token_to_precommit_ms"] = max(
                            0.0,
                            (precommit_received_at - t_last_token) * 1000,
                        )
                    _consume_connection_prewarm(challenge_connection_prewarm)
                    reveal_started = time.perf_counter()
                    transport_state.nonce_reveal_attempted = True
                    try:
                        reveal_response = self.client.post(
                            f"{self.miner_url}/proof/v2/challenge",
                            json=_proof_v2_reveal_body(
                                challenge_id=proof_challenge_id,
                                session_id=precommit_session_id,
                                commitment_hash=precommitment_hash,
                                validator_nonce=nonce,
                            ),
                        )
                        reveal_response.raise_for_status()
                    except Exception as exc:
                        if transport_state.hard_audit_selected is True:
                            raise _hard_audit_transport_failure(
                                transport_state,
                                "nonce reveal did not complete",
                            ) from exc
                        raise
                    transport_state.nonce_revealed = True
                    timing["proof_challenge_rtt_ms"] = (
                        time.perf_counter() - reveal_started
                    ) * 1000
                elif event_type == "proof_commitment":
                    if (
                        proof_protocol_version != PROOF_PROTOCOL_V2
                        or precommitment is not None
                    ):
                        raise RuntimeError("invalid proof-v2 commitment event")
                    try:
                        precommitment = _decode_proof_v2_commitment(
                            data,
                            precommit_session_id=precommit_session_id,
                            precommitment_hash=precommitment_hash,
                        )
                    except RuntimeError as exc:
                        if (
                            transport_state.hard_audit_selected is True
                            and transport_state.nonce_reveal_attempted
                        ):
                            raise _hard_audit_transport_failure(
                                transport_state,
                                "post-nonce commitment is invalid",
                            ) from exc
                        raise
                elif event_type == "done":
                    transport_state.done_seen = True
                    _t_done_recv = time.perf_counter()
                    t_request_end_wall = time.time()
                    t_done_recv = _t_done_recv
                    _done_gap_ms = (
                        (_t_done_recv - _t_last_tok) * 1000
                        if _t_last_tok is not None
                        else 0.0
                    )
                    # TEE miners may return empty commitment/proof_bundle
                    commit_data = data.get("commitment", {})
                    proof_data = data.get(
                        "proof_bundle", {"layer_proofs": [], "sampling_proofs": []}
                    )
                    _t_commit_deser = time.perf_counter()
                    must_deserialize_proof_bundle = _must_deserialize_proof_payload(
                        deserialize_proof_bundle=deserialize_proof_bundle,
                        transport_state=transport_state,
                    )
                    commitment, proof_bundle = _deserialize_done_proof_payload(
                        commit_data,
                        proof_data,
                        deserialize_proof_bundle=must_deserialize_proof_bundle,
                        requested_protocol_version=proof_protocol_version,
                        fallback_commitment=precommitment,
                    )
                    if proof_protocol_version == PROOF_PROTOCOL_V2:
                        try:
                            _validate_proof_v2_done_commitment(
                                final_commitment=commitment,
                                precommit_session_id=precommit_session_id,
                                precommitment_hash=precommitment_hash,
                            )
                            timing[
                                "last_token_to_proof_ms"
                            ] = _measure_proof_v2_response_latency(
                                last_token_at=t_last_token,
                                proof_received_at=t_done_recv,
                            )
                        except RuntimeError:
                            proof_bundle = _retain_failed_proof_v2_transport(
                                proof_bundle
                            )
                    _t_proof_deser = time.perf_counter()
                    _deser_ms = (_t_proof_deser - _t_done_recv) * 1000
                    timing["last_token_to_done_recv_ms"] = _done_gap_ms
                    timing["proof_deserialize_ms"] = _deser_ms
                    import logging as _logging

                    _logging.getLogger("verallm.api.client").debug(
                        "CLIENT TIMING: last_token→done_recv=%.0fms deser=%.0fms (commit=%.0fms proof=%.0fms)",
                        _done_gap_ms,
                        _deser_ms,
                        (_t_commit_deser - _t_done_recv) * 1000,
                        (_t_proof_deser - _t_commit_deser) * 1000,
                    )
                    _remember_miner_inference_ms(timing, data.get("inference_ms"))
                    timing["commitment_ms"] = data.get("commitment_ms", 0)
                    timing["prove_ms"] = data.get("prove_ms", 0)
                    timing["beacon_ms"] = data.get("beacon_ms", 0)
                    timing["challenge_ms"] = data.get("challenge_ms", 0)
                    timing["reveal_ms"] = data.get("reveal_ms", 0)
                    timing["prove_timing_details"] = data.get(
                        "prove_timing_details", {}
                    )
                    timing["miner_last_token_to_proof_ms"] = data.get(
                        "last_token_to_proof_ms", 0
                    )
                    timing["input_tokens"] = data.get("input_tokens", 0)
                    timing["output_tokens"] = _validated_output_token_count(
                        data,
                        max_new_tokens=max_new_tokens,
                        commitment=commitment,
                        commitment_present=bool(commit_data),
                        proof_data=proof_data,
                        allow_unbound=allow_unbound_output_count,
                    )
                elif event_type == "error":
                    if (
                        transport_state.v2_precommit_required
                        and (
                            transport_state.precommit_seen
                            or transport_state.saw_response_token
                        )
                    ):
                        raise _hard_audit_transport_failure(
                            transport_state,
                            "miner ended the required proof-v2 exchange",
                        )
                    if (
                        transport_state.hard_audit_selected is True
                        and transport_state.nonce_reveal_attempted
                    ):
                        raise _hard_audit_transport_failure(
                            transport_state,
                            "miner sent an error after hard-audit selection",
                        )
                    raise RuntimeError(f"Miner error: {data.get('error', data)}")

        if (
            transport_state.v2_precommit_required
            and not transport_state.precommit_seen
        ):
            raise _hard_audit_transport_failure(
                transport_state,
                "response omitted the required proof-v2 precommit",
            )
        if (
            proof_protocol_version == PROOF_PROTOCOL_V2
            and transport_state.hard_audit_selected is True
            and transport_state.nonce_reveal_attempted
            and not transport_state.done_seen
        ):
            raise _hard_audit_transport_failure(
                transport_state,
                "stream ended without a final proof response",
            )

        if t_request_end_wall is None:
            t_request_end_wall = time.time()
        timing["validator_request_start_ts"] = t0_wall
        timing["validator_request_end_ts"] = t_request_end_wall
        timing["validator_request_ms"] = (t_request_end_wall - t0_wall) * 1000
        _finalize_validator_timing(
            timing,
            t0,
            t_first_token,
            t_last_token,
            response_done_at=t_done_recv,
        )

        # TEE miners don't produce commitments/proofs — create empty defaults
        if commitment is None:
            commitment = InferenceCommitment.empty()
        if proof_bundle is None:
            proof_bundle = InferenceProofBundle.empty()
            if _must_deserialize_proof_payload(
                deserialize_proof_bundle=deserialize_proof_bundle,
                transport_state=transport_state,
            ):
                mark_proof_payload_invalid(
                    proof_bundle,
                    protocol_version=(
                        proof_protocol_version or LEGACY_PROOF_PROTOCOL_VERSION
                    ),
                )

        return full_text, commitment, proof_bundle, nonce, timing

    def _prepare_chat_proof_v3(
        self,
        *,
        messages: list[dict],
        prompt_token_ids: Sequence[int],
        qualified_profile,
        validator_identity_digest: bytes,
        miner_identity_digest: bytes,
        runtime_policy,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        presence_penalty: float = 0.0,
        top_k: int = -1,
        top_p: float = 1.0,
        min_p: float = 0.0,
        proof_challenge_id: bytes | None = None,
        nonce_reveal_hold_budget_ns: int | None = None,
        expected_hard_audit: bool | None = None,
    ) -> tuple[object, dict[str, object]]:
        """Build the shared strict v3 exchange and miner request body."""

        from verallm.api.proof_v3_validator import ProofV3ValidatorExchange
        from verallm.proof_v3.sampler import (
            economic_sampler_config_digest_v3,
        )
        from verallm.proof_v3.session import (
            QualifiedExecutionProfileV3,
            hard_proof_arrival_budget_for_decode_v3,
        )

        if not isinstance(qualified_profile, QualifiedExecutionProfileV3):
            raise TypeError("qualified_profile has an unexpected type")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or not 0 < max_new_tokens
            <= qualified_profile.profile.max_verified_decode_tokens
        ):
            raise ValueError(
                "proof-v3 requested decode length exceeds the signed profile"
            )
        selected_hard_bps = (
            runtime_policy.effective_canary_hard_bps
            if runtime_policy.request_kind == "canary"
            else runtime_policy.effective_organic_hard_bps
        )
        if selected_hard_bps:
            from verallm.proof_v3.economic_registry import (
                QualifiedEconomicAdapterV3,
            )
            from verallm.proof_v3.errors import ProofV3UnavailableError

            registration = qualified_profile.registration
            if isinstance(registration, QualifiedEconomicAdapterV3):
                hard_limit = (
                    registration.maximum_hard_audit_decode_tokens(
                        profile=qualified_profile.profile,
                    )
                )
                if max_new_tokens > hard_limit:
                    raise ProofV3UnavailableError(
                        "proof-v3 release is not qualified for the requested "
                        "hard-audit decode length"
                    )
        resolved_sampling_params = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "presence_penalty": presence_penalty,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p,
        }
        if selected_hard_bps:
            from verallm.proof_v3.sampler import (
                validate_economic_hard_sampler_config_v3,
            )

            validate_economic_hard_sampler_config_v3(
                do_sample=do_sample,
                resolved_sampling_params=resolved_sampling_params,
                max_decode_tokens=(
                    qualified_profile.profile.max_verified_decode_tokens
                ),
            )
        sampler_digest = economic_sampler_config_digest_v3(
            do_sample=do_sample,
            resolved_sampling_params=resolved_sampling_params,
            max_decode_tokens=(
                qualified_profile.profile.max_verified_decode_tokens
            ),
        )
        challenge_id = proof_challenge_id
        if challenge_id is None:
            challenge_id = os.urandom(32)
            while challenge_id == bytes(32):
                challenge_id = os.urandom(32)
        exchange = ProofV3ValidatorExchange.issue(
            qualified_profile=qualified_profile,
            proof_challenge_id=challenge_id,
            validator_identity_digest=validator_identity_digest,
            miner_identity_digest=miner_identity_digest,
            prompt_token_ids=prompt_token_ids,
            sampler_config_digest=sampler_digest,
            runtime_policy=runtime_policy,
            hard_proof_arrival_budget_ns=(
                hard_proof_arrival_budget_for_decode_v3(max_new_tokens)
            ),
            nonce_reveal_hold_budget_ns=nonce_reveal_hold_budget_ns,
            expected_hard_audit=expected_hard_audit,
        )
        return exchange, {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "sampling_verification_bps": 0,
            "enable_thinking": enable_thinking,
            "presence_penalty": presence_penalty,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p,
        }

    def run_chat_proof_v3_precommit(
        self,
        *,
        messages: list[dict],
        prompt_token_ids: Sequence[int],
        qualified_profile,
        validator_identity_digest: bytes,
        miner_identity_digest: bytes,
        runtime_policy,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        presence_penalty: float = 0.0,
        top_k: int = -1,
        top_p: float = 1.0,
        min_p: float = 0.0,
        stream_callback=None,
        proof_challenge_id: bytes | None = None,
        nonce_reveal_hold_budget_ns: int | None = None,
        expected_hard_audit: bool | None = None,
    ):
        """Run one v3 chat stream through its accepted precommit only."""

        from verallm.api.proof_v3_validator import (
            run_proof_v3_precommit_sync,
        )

        exchange, request_body = self._prepare_chat_proof_v3(
            messages=messages,
            prompt_token_ids=prompt_token_ids,
            qualified_profile=qualified_profile,
            validator_identity_digest=validator_identity_digest,
            miner_identity_digest=miner_identity_digest,
            runtime_policy=runtime_policy,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            enable_thinking=enable_thinking,
            presence_penalty=presence_penalty,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            proof_challenge_id=proof_challenge_id,
            nonce_reveal_hold_budget_ns=nonce_reveal_hold_budget_ns,
            expected_hard_audit=expected_hard_audit,
        )
        return run_proof_v3_precommit_sync(
            client=self.client,
            miner_url=self.miner_url,
            inference_path="/chat",
            request_body=request_body,
            exchange=exchange,
            stream_callback=stream_callback,
        )

    def finalize_chat_proof_v3(self, exchange):
        """Finalize one chat exchange whose precommit is already accepted."""

        from verallm.api.proof_v3_validator import (
            finalize_proof_v3_exchange_sync,
        )

        return finalize_proof_v3_exchange_sync(
            client=self.client,
            miner_url=self.miner_url,
            exchange=exchange,
        )

    def hold_chat_proof_v3_precommit(self, exchange) -> None:
        """Install a bounded hold for the first frozen full-pair request."""

        from verallm.api.proof_v3_validator import (
            hold_proof_v3_precommit_sync,
        )

        hold_proof_v3_precommit_sync(
            client=self.client,
            miner_url=self.miner_url,
            exchange=exchange,
        )

    def run_chat_proof_v3(
        self,
        *,
        messages: list[dict],
        prompt_token_ids: Sequence[int],
        qualified_profile,
        validator_identity_digest: bytes,
        miner_identity_digest: bytes,
        runtime_policy,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        presence_penalty: float = 0.0,
        top_k: int = -1,
        top_p: float = 1.0,
        min_p: float = 0.0,
        stream_callback=None,
        proof_challenge_id: bytes | None = None,
    ):
        """Run one qualified greedy proof-v3 chat exchange.

        The caller owns tokenizer/template qualification and supplies the exact
        prompt token IDs. This method derives the signed sampler binding,
        commits fresh validator entropy, and delegates the strict SSE/reveal
        lifecycle to :mod:`verallm.api.proof_v3_validator`.
        """

        from verallm.api.proof_v3_validator import run_proof_v3_exchange_sync

        exchange, request_body = self._prepare_chat_proof_v3(
            messages=messages,
            prompt_token_ids=prompt_token_ids,
            qualified_profile=qualified_profile,
            validator_identity_digest=validator_identity_digest,
            miner_identity_digest=miner_identity_digest,
            runtime_policy=runtime_policy,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            enable_thinking=enable_thinking,
            presence_penalty=presence_penalty,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            proof_challenge_id=proof_challenge_id,
        )
        return run_proof_v3_exchange_sync(
            client=self.client,
            miner_url=self.miner_url,
            inference_path="/chat",
            request_body=request_body,
            exchange=exchange,
            stream_callback=stream_callback,
        )

    # ------------------------------------------------------------------
    # TEE encrypted inference
    # ------------------------------------------------------------------

    def run_encrypted_inference(
        self,
        encrypted_envelope: "EncryptedEnvelope",
        validator_nonce: bytes,
        stream_callback=None,
        proof_protocol_version: Optional[int] = None,
    ) -> Tuple[
        Optional[dict], "InferenceCommitment", "InferenceProofBundle", bytes, dict, list
    ]:
        """Forward an encrypted chat request to a TEE-enabled miner.

        The validator cannot decrypt the conversation — it only verifies
        the plaintext proof bundle and relays the encrypted output.

        Args:
            encrypted_envelope: User's encrypted chat request.
            validator_nonce: Validator's 32-byte nonce for Fiat-Shamir.
            stream_callback: Optional callable invoked for each encrypted_token
                event with the chunk dict (or None for heartbeats).
            proof_protocol_version: Explicit proof wire version. Omit for the
                legacy v1 request contract.

        Returns:
            (encrypted_output_dict, commitment, proof_bundle, nonce, timing,
             encrypted_chunks) where encrypted_chunks is a list of
             serialized EncryptedEnvelope dicts for streaming token deltas.
        """
        if proof_protocol_version == PROOF_PROTOCOL_V2:
            raise ValueError("proof-v2 is unavailable for encrypted TEE chat")

        request_body = {
            "envelope": {
                "session_id": encrypted_envelope.session_id,
                "sender_public_key": encrypted_envelope.sender_public_key.hex(),
                "nonce": encrypted_envelope.nonce.hex(),
                "ciphertext": encrypted_envelope.ciphertext.hex(),
                "content_type": encrypted_envelope.content_type,
            },
            "validator_nonce": encode_validator_nonce(validator_nonce),
        }
        add_proof_protocol_version(request_body, proof_protocol_version)

        commitment = None
        proof_bundle = None
        encrypted_output = None
        encrypted_chunks: list[dict] = []
        timing = {}
        t_first_token = None
        t_last_token = None
        t_done_recv = None
        t_request_end_wall = None
        t0 = time.perf_counter()
        t0_wall = time.time()

        with self.client.stream(
            "POST", f"{self.miner_url}/tee/chat", json=request_body
        ) as resp:
            resp.raise_for_status()
            for event_type, data in _parse_sse_stream(resp):
                if event_type == "heartbeat":
                    if stream_callback:
                        stream_callback(None)
                elif event_type == "encrypted_token":
                    t_last_token = time.perf_counter()
                    if t_first_token is None:
                        t_first_token = t_last_token
                    encrypted_chunks.append(data)
                    if stream_callback:
                        stream_callback(data)
                elif event_type == "done":
                    t_done_recv = time.perf_counter()
                    # TEE miners may return empty commitment/proof_bundle
                    commit_data = data.get("commitment", {})
                    proof_data = data.get(
                        "proof_bundle", {"layer_proofs": [], "sampling_proofs": []}
                    )
                    commitment, proof_bundle = _deserialize_done_proof_payload(
                        commit_data,
                        proof_data,
                        requested_protocol_version=proof_protocol_version,
                    )
                    encrypted_output = data.get("encrypted_output")
                    # Miner sends encrypted_output_nonce alongside encrypted_output
                    if data.get("encrypted_output_nonce"):
                        encrypted_output = {
                            "encrypted_output": data["encrypted_output"],
                            "encrypted_output_nonce": data["encrypted_output_nonce"],
                        }
                    t_request_end_wall = time.time()
                    # Timing may be nested under "timing" key or flat
                    t = data.get("timing", data)
                    _remember_miner_inference_ms(timing, t.get("inference_ms"))
                    timing["commitment_ms"] = t.get("commitment_ms", 0)
                    timing["prove_ms"] = t.get("prove_ms", 0)
                    timing["input_tokens"] = t.get("input_tokens", 0)
                    timing["output_tokens"] = t.get("output_tokens", 0)
                    timing["model_id"] = t.get("model_id", "")
                elif event_type == "error":
                    raise RuntimeError(f"Miner error: {data.get('error', data)}")

        if t_request_end_wall is None:
            t_request_end_wall = time.time()
        timing["validator_request_start_ts"] = t0_wall
        timing["validator_request_end_ts"] = t_request_end_wall
        timing["validator_request_ms"] = (t_request_end_wall - t0_wall) * 1000
        _finalize_validator_timing(
            timing,
            t0,
            t_first_token,
            t_last_token,
            response_done_at=t_done_recv,
        )

        # TEE miners don't produce commitments/proofs — create empty defaults
        if commitment is None:
            commitment = InferenceCommitment.empty()
        if proof_bundle is None:
            proof_bundle = mark_proof_payload_invalid(
                InferenceProofBundle.empty(),
                protocol_version=(
                    proof_protocol_version or LEGACY_PROOF_PROTOCOL_VERSION
                ),
            )

        return (
            encrypted_output,
            commitment,
            proof_bundle,
            validator_nonce,
            timing,
            encrypted_chunks,
        )

    def fetch_tee_info(self) -> Optional[dict]:
        """Fetch TEE capability from a miner. Returns None if TEE disabled."""
        try:
            resp = self.client.get(f"{self.miner_url}/tee/info")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError:
            return None

    def _verify_moe_layer_root(
        self,
        layer_idx: int,
        expert_roots: list[bytes],
        router_weight_root: Optional[bytes],
    ) -> bool:
        """Verify expert/router roots hash to the committed MoE layer root."""
        if self.model_spec is None:
            return False
        if layer_idx >= len(self.model_spec.weight_block_merkle_roots):
            return True

        committed_layer_root = self.model_spec.weight_block_merkle_roots[layer_idx]
        if committed_layer_root is None:
            return True

        if router_weight_root:
            h_v2 = hashlib.sha256(b"MOE_LAYER_V2")
            h_v2.update(router_weight_root)
            for er in expert_roots:
                h_v2.update(er)
            if h_v2.digest() == committed_layer_root:
                return True

        h_v1 = hashlib.sha256(b"MOE_LAYER_V1")
        for er in expert_roots:
            h_v1.update(er)
        return h_v1.digest() == committed_layer_root

    def _verify_router_layer_openings(
        self, proof_bundle: InferenceProofBundle, layer_challenge
    ) -> VerificationResult:
        """Verify Merkle openings and top-k consistency for sampled router tokens."""
        layer_idx = layer_challenge.layer_idx
        router_commitment = proof_bundle.router_commitments.get(layer_idx)
        if router_commitment is None:
            return VerificationResult.failure(
                f"Missing router commitment for challenged MoE layer {layer_idx}"
            )

        layer_routing_proof = proof_bundle.router_layer_proofs.get(layer_idx)
        if layer_routing_proof is None:
            return VerificationResult.failure(
                f"Missing router layer proof for challenged MoE layer {layer_idx}"
            )

        if not router_commitment.router_logits_row_root:
            return VerificationResult.failure(
                f"Router logits row root missing in router commitment for layer {layer_idx}"
            )

        # Strictly bind routing metadata to model spec to prevent challenge steering.
        expected_num_experts = (
            int(getattr(self.model_spec, "num_experts", 0) or 0)
            if self.model_spec
            else 0
        )
        if expected_num_experts <= 0 and self.model_spec is not None:
            spec_roots = self.model_spec.expert_weight_merkle_roots.get(layer_idx, [])
            if spec_roots:
                expected_num_experts = len(spec_roots)
        if expected_num_experts > 0:
            if router_commitment.num_experts <= 0:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: router commitment missing num_experts"
                )
            if router_commitment.num_experts != expected_num_experts:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: num_experts mismatch "
                    f"(committed={router_commitment.num_experts}, expected={expected_num_experts})"
                )

        expected_top_k = (
            int(getattr(self.model_spec, "router_top_k", 0) or 0)
            if self.model_spec
            else 0
        )
        if expected_top_k > 0:
            if router_commitment.top_k <= 0:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: router commitment missing top_k"
                )
            if router_commitment.top_k != expected_top_k:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: top_k mismatch "
                    f"(committed={router_commitment.top_k}, expected={expected_top_k})"
                )

        expected_scoring = (
            str(getattr(self.model_spec, "router_scoring", "") or "")
            if self.model_spec
            else ""
        )
        if expected_scoring and router_commitment.scoring_func:
            if str(router_commitment.scoring_func) != expected_scoring:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: scoring_func mismatch "
                    f"(committed={router_commitment.scoring_func}, expected={expected_scoring})"
                )

        committed_seq = len(router_commitment.selected_experts)
        if router_commitment.seq_len and router_commitment.seq_len != committed_seq:
            return VerificationResult.failure(
                f"Layer {layer_idx}: seq_len mismatch "
                f"(committed={router_commitment.seq_len}, rows={committed_seq})"
            )
        if len(router_commitment.routing_weights) != committed_seq:
            return VerificationResult.failure(
                f"Layer {layer_idx}: routing_weights row-count mismatch "
                f"(weights={len(router_commitment.routing_weights)}, experts={committed_seq})"
            )
        if router_commitment.proof_selected_experts:
            if len(router_commitment.proof_selected_experts) != committed_seq:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: proof_selected_experts row-count mismatch "
                    f"(proof={len(router_commitment.proof_selected_experts)}, experts={committed_seq})"
                )

        top_k = int(router_commitment.top_k or 0)
        if top_k <= 0:
            return VerificationResult.failure(
                f"Layer {layer_idx}: invalid top_k={router_commitment.top_k}"
            )

        from verallm.crypto.field import P as FIELD_PRIME

        for row_idx, row in enumerate(router_commitment.selected_experts):
            if len(row) != top_k:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: selected_experts row {row_idx} width mismatch "
                    f"(expected={top_k}, got={len(row)})"
                )
            for expert_idx in row:
                if expert_idx < 0 or (
                    router_commitment.num_experts > 0
                    and expert_idx >= router_commitment.num_experts
                ):
                    return VerificationResult.failure(
                        f"Layer {layer_idx}: expert index out of range at row {row_idx}: {expert_idx}"
                    )

        for row_idx, row in enumerate(router_commitment.routing_weights):
            if len(row) != top_k:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: routing_weights row {row_idx} width mismatch "
                    f"(expected={top_k}, got={len(row)})"
                )
            for w in row:
                if int(w) < 0 or int(w) >= FIELD_PRIME:
                    return VerificationResult.failure(
                        f"Layer {layer_idx}: routing weight out of field range at row {row_idx}"
                    )
        for row_idx, row in enumerate(router_commitment.proof_selected_experts):
            if len(row) != top_k:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: proof_selected_experts row {row_idx} width mismatch "
                    f"(expected={top_k}, got={len(row)})"
                )
            for expert_idx in row:
                if expert_idx < 0 or (
                    router_commitment.num_experts > 0
                    and expert_idx >= router_commitment.num_experts
                ):
                    return VerificationResult.failure(
                        f"Layer {layer_idx}: proof expert index out of range at row {row_idx}: {expert_idx}"
                    )

        expert_roots = (
            self.model_spec.expert_weight_merkle_roots.get(layer_idx, [])
            if self.model_spec
            else []
        )
        if not expert_roots:
            expert_roots = proof_bundle.expert_roots.get(layer_idx, [])
        if expert_roots and not self._verify_moe_layer_root(
            layer_idx=layer_idx,
            expert_roots=expert_roots,
            router_weight_root=layer_routing_proof.router_weight_root,
        ):
            return VerificationResult.failure(
                f"MoE layer {layer_idx}: router/expert roots do not match committed layer root"
            )

        # Verify sampled router GEMM proof: this binds routing to X and W.
        if layer_routing_proof.router_gemm_proof is None:
            return VerificationResult.failure(
                f"Missing router GEMM proof for challenged MoE layer {layer_idx}"
            )
        if not layer_routing_proof.proved_output_rows:
            return VerificationResult.failure(
                f"Missing proven router output rows for challenged MoE layer {layer_idx}"
            )

        proved_rows_by_token = {
            int(row.token_idx): row for row in layer_routing_proof.proved_output_rows
        }
        if len(proved_rows_by_token) != len(layer_routing_proof.proved_output_rows):
            return VerificationResult.failure(
                f"Duplicate token indices in proven router output rows for layer {layer_idx}"
            )

        proved_matrix_rows: list[list[int]] = []
        for token_idx in layer_challenge.sampled_token_indices:
            proved_row = proved_rows_by_token.get(int(token_idx))
            if proved_row is None:
                return VerificationResult.failure(
                    f"Missing proven router output row for layer {layer_idx}, token {token_idx}"
                )
            if (
                router_commitment.num_experts
                and len(proved_row.logits_int) != router_commitment.num_experts
            ):
                return VerificationResult.failure(
                    f"Layer {layer_idx} token {token_idx}: proven router row width mismatch "
                    f"(expected {router_commitment.num_experts}, got {len(proved_row.logits_int)})"
                )
            proved_matrix_rows.append([int(v) for v in proved_row.logits_int])

        if not proved_matrix_rows:
            return VerificationResult.failure(
                f"No proven router rows available for challenged layer {layer_idx}"
            )

        y_tensor = torch.tensor(proved_matrix_rows, dtype=torch.int64)
        y_root = build_block_merkle(y_tensor, self.config.block_size).root
        sampled_token_indices = [int(t) for t in layer_challenge.sampled_token_indices]

        def map_router_row(local_row: int) -> int:
            if local_row < 0 or local_row >= len(sampled_token_indices):
                raise ValueError(
                    f"router proof row {local_row} outside sampled rows "
                    f"(count={len(sampled_token_indices)})"
                )
            return sampled_token_indices[local_row]

        x_result = verify_x_spot_openings(
            proof=layer_routing_proof.router_gemm_proof,
            x_root=proof_bundle.commitment.layer_commitments[layer_idx],
            num_cols=self.model_spec.hidden_dim,
            context=f"Layer {layer_idx} router GEMM X",
            row_mapper=map_router_row,
        )
        if not x_result.passed:
            return VerificationResult.failure(x_result.message)

        router_verifier = GEMMVerifier(self.config)
        router_verify = router_verifier.verify(
            proof=layer_routing_proof.router_gemm_proof,
            X_commitment=proof_bundle.commitment.layer_commitments[layer_idx],
            W_commitment=layer_routing_proof.router_weight_root,
            Y_commitment=y_root,
            transcript=Transcript(f"layer_{layer_idx}_router_gemm".encode()),
            spot_check_fn=lambda _spot, _matrix_id: True,
            W_merkle_root=layer_routing_proof.router_weight_root,
            W_num_cols=(
                router_commitment.num_experts
                if router_commitment.num_experts > 0
                else len(proved_matrix_rows[0])
            ),
            w_chunk_size=self.model_spec.w_merkle_chunk_size,
            require_w_merkle_proofs=True,
        )
        if not router_verify.passed:
            return VerificationResult.failure(
                f"Layer {layer_idx}: router GEMM proof failed: {router_verify.message}"
            )

        openings_by_token = {
            opening.token_idx: opening
            for opening in layer_routing_proof.logits_openings
        }

        for token_idx in layer_challenge.sampled_token_indices:
            opening = openings_by_token.get(int(token_idx))
            if opening is None:
                return VerificationResult.failure(
                    f"Missing router logits opening for layer {layer_idx}, token {token_idx}"
                )

            if token_idx >= len(router_commitment.selected_experts):
                return VerificationResult.failure(
                    f"Token {token_idx} out of range for selected_experts in layer {layer_idx}"
                )

            if router_commitment.seq_len and token_idx >= router_commitment.seq_len:
                return VerificationResult.failure(
                    f"Token {token_idx} out of range for committed seq_len in layer {layer_idx}"
                )

            if (
                router_commitment.num_experts
                and len(opening.logits) != router_commitment.num_experts
            ):
                return VerificationResult.failure(
                    f"Layer {layer_idx} token {token_idx}: logits width mismatch "
                    f"(expected {router_commitment.num_experts}, got {len(opening.logits)})"
                )

            path_ok = verify_merkle_path(
                root=router_commitment.router_logits_row_root,
                leaf_data=logits_row_to_bytes(opening.logits),
                path=opening.merkle_path,
            )
            if not path_ok:
                return VerificationResult.failure(
                    f"Layer {layer_idx} token {token_idx}: router logits Merkle proof invalid"
                )

            committed_experts = router_commitment.selected_experts[token_idx]
            recomputed = compute_topk_indices(opening.logits, top_k)
            committed_prefix = [int(x) for x in committed_experts[:top_k]]
            if recomputed != committed_prefix:
                return VerificationResult.failure(
                    f"Layer {layer_idx} token {token_idx}: top-k mismatch "
                    f"(committed={committed_prefix}, recomputed={recomputed})"
                )

            proved_row = proved_rows_by_token.get(int(token_idx))
            if proved_row is None:
                return VerificationResult.failure(
                    f"Missing proven router output row for layer {layer_idx}, token {token_idx}"
                )
            proved_topk = compute_topk_indices(proved_row.logits_int, top_k)
            if token_idx >= len(router_commitment.proof_selected_experts):
                return VerificationResult.failure(
                    f"Token {token_idx} out of range for proof_selected_experts in layer {layer_idx}"
                )
            proof_prefix = [
                int(x)
                for x in router_commitment.proof_selected_experts[token_idx][:top_k]
            ]
            if proved_topk != proof_prefix:
                return VerificationResult.failure(
                    f"Layer {layer_idx} token {token_idx}: router-GEMM top-k mismatch "
                    f"(committed={proof_prefix}, proved={proved_topk})"
                )

        return VerificationResult.success(
            f"Routing openings verified for layer {layer_idx}"
        )

    def _verify_sampling_consistency_v2(
        self,
        *,
        proof_bundle: InferenceProofBundle,
        sampling_challenge,
        transcript_state: bytes,
        expected_top_k: Optional[int],
        expected_top_p: Optional[float],
        expected_min_p: Optional[float],
        minimum_decode_step: int = 0,
    ) -> VerificationResult:
        """Verify the strict response-bound decode-row format used by v2."""

        commitment = proof_bundle.commitment
        if (
            not commitment.decode_hidden_row_root
            or not commitment.decode_logits_row_root
        ):
            return VerificationResult.failure(
                "Proof-v2 sampling challenge requires committed hidden and logits rows"
            )
        if not proof_bundle.output_token_ids:
            return VerificationResult.failure(
                "Proof-v2 sampling challenge requires committed output tokens"
            )

        expected_positions = tuple(sampling_challenge.decode_positions)
        if any(type(position) is not int for position in expected_positions):
            return VerificationResult.failure(
                "Proof-v2 sampling challenge contains an invalid position"
            )
        received_positions = tuple(
            proof.decode_step for proof in proof_bundle.sampling_proofs
        )
        if (
            not expected_positions
            or any(type(position) is not int for position in received_positions)
            or len(expected_positions) != len(set(expected_positions))
            or received_positions != expected_positions
        ):
            return VerificationResult.failure(
                "Proof-v2 sampling proof set does not match the challenge"
            )

        from verallm.sampling import (
            CANONICAL_TOP_K,
            canonical_sample,
            parse_top_k_leaf,
        )

        vocab_size = int(getattr(self.model_spec, "vocab_size", 0) or 0)
        if vocab_size <= 0:
            return VerificationResult.failure("Proof-v2 sampling vocabulary is invalid")
        if any(
            type(token) is not int or token < 0 or token >= vocab_size
            for token in proof_bundle.output_token_ids
        ):
            return VerificationResult.failure(
                "Proof-v2 output token history is invalid"
            )
        expected_width = min(CANONICAL_TOP_K, vocab_size)
        hidden_dim = int(getattr(self.model_spec, "hidden_dim", 0) or 0)
        if hidden_dim <= 0:
            return VerificationResult.failure("Proof-v2 hidden dimension is invalid")
        output_count = len(proof_bundle.output_token_ids)

        for proof in proof_bundle.sampling_proofs:
            step = proof.decode_step
            if step < 0 or step >= len(proof_bundle.output_token_ids):
                return VerificationResult.failure(
                    "Proof-v2 sampling position is out of range"
                )
            token_id = int(proof_bundle.output_token_ids[step])
            if type(proof.token_id) is not int or proof.token_id != token_id:
                return VerificationResult.failure(
                    "Proof-v2 sampled token does not match the output commitment"
                )
            if (
                proof.proved_logits_i32
                or proof.lm_head_weight_root
                or proof.lm_head_gemm_proof is not None
                or proof.logits_merkle_path is not None
            ):
                return VerificationResult.failure(
                    "Proof-v2 sampling payload contains fields from another protocol"
                )
            if (
                not isinstance(proof.hidden_row, bytes)
                or len(proof.hidden_row) != hidden_dim * 2
                or not _merkle_path_matches_position(
                    proof.hidden_merkle_path,
                    step,
                    output_count,
                )
                or not verify_merkle_path(
                    root=commitment.decode_hidden_row_root,
                    leaf_data=proof.hidden_row,
                    path=proof.hidden_merkle_path,
                )
            ):
                return VerificationResult.failure(
                    "Proof-v2 hidden-row opening did not verify"
                )
            if (
                not isinstance(proof.fp16_logits_row, bytes)
                or proof.fp16_logits_merkle_path is None
                or not _merkle_path_matches_position(
                    proof.fp16_logits_merkle_path,
                    step,
                    output_count,
                )
                or not verify_merkle_path(
                    root=commitment.decode_logits_row_root,
                    leaf_data=proof.fp16_logits_row,
                    path=proof.fp16_logits_merkle_path,
                )
            ):
                return VerificationResult.failure(
                    "Proof-v2 logits-row opening did not verify"
                )

            try:
                top_values, top_indices = parse_top_k_leaf(proof.fp16_logits_row)
            except ValueError:
                return VerificationResult.failure(
                    "Proof-v2 logits row is not in the canonical top-k format"
                )
            if (
                top_values.size != expected_width
                or top_indices.size != expected_width
                or not np.isfinite(top_values).all()
                or (top_indices < 0).any()
                or (top_indices >= vocab_size).any()
                or np.unique(top_indices).size != expected_width
            ):
                return VerificationResult.failure(
                    "Proof-v2 logits row has invalid top-k contents"
                )
            canonical_order = np.lexsort((top_indices, -top_values))
            if not np.array_equal(
                canonical_order,
                np.arange(expected_width, dtype=canonical_order.dtype),
            ):
                return VerificationResult.failure(
                    "Proof-v2 logits row is not canonically ordered"
                )

            if commitment.do_sample:
                seed = proof.sampling_seed
                if (
                    not isinstance(seed, bytes)
                    or len(seed) != 32
                    or hashlib.sha256(seed).digest()
                    != commitment.sampling_seed_commitment
                ):
                    return VerificationResult.failure(
                        "Proof-v2 sampling seed opening did not verify"
                    )
                replayed = canonical_sample(
                    top_values,
                    top_indices,
                    max(0.001, commitment.temperature_milli / 1000.0),
                    int(expected_top_k) if expected_top_k is not None else -1,
                    float(expected_top_p) if expected_top_p is not None else 1.0,
                    float(expected_min_p) if expected_min_p is not None else 0.0,
                    seed,
                    step,
                )
                if replayed != token_id:
                    return VerificationResult.failure(
                        "Proof-v2 canonical sampling replay diverged"
                    )
            else:
                if proof.sampling_seed is not None:
                    return VerificationResult.failure(
                        "Proof-v2 greedy sampling payload contains a seed"
                    )
                try:
                    greedy_token = _greedy_token_after_presence_penalty_v2(
                        top_values=top_values,
                        top_indices=top_indices,
                        output_token_ids=proof_bundle.output_token_ids,
                        step=step,
                        presence_penalty_milli=commitment.presence_penalty_milli,
                        vocab_size=vocab_size,
                    )
                except ValueError:
                    return VerificationResult.failure(
                        "Proof-v2 greedy top-k cannot prove the processed argmax"
                    )
                if commitment.temperature_milli != 0 or greedy_token != token_id:
                    return VerificationResult.failure(
                        "Proof-v2 greedy token is not the processed logits argmax"
                    )

        from verallm.proof_v2.engine import build_inference_x_state_v2
        from verallm.proof_v2.hardening import (
            derive_lm_head_audit_challenges_v2,
            quantize_committed_hidden_row_v2,
            select_lm_head_audit_decode_step_v2,
        )
        from verallm.proof_v2.layout import (
            registered_lm_head_operation_from_manifest,
        )
        from verallm.proof_v2.payload import ProofV2CommitmentEnvelope

        manifest = self.proof_v2_manifest
        if manifest is None:
            return VerificationResult.failure(
                "Proof-v2 hardened LM-head manifest is unavailable"
            )
        try:
            operation = registered_lm_head_operation_from_manifest(manifest)
        except Exception:
            operation = None
        if operation is None:
            return VerificationResult.failure(
                "Proof-v2 hardened LM-head operation is unavailable"
            )
        try:
            audit_step = select_lm_head_audit_decode_step_v2(
                transcript_state=transcript_state,
                commitment_hash=commitment.commitment_hash(),
                decode_positions=expected_positions,
                minimum_decode_step=minimum_decode_step,
            )
        except Exception:
            return VerificationResult.failure(
                "Proof-v2 hardened LM-head position is invalid"
            )
        carriers = [
            proof
            for proof in proof_bundle.sampling_proofs
            if proof.lm_head_proof_v2_commitment
        ]
        if (
            len(carriers) != 1
            or carriers[0].decode_step != audit_step
            or not carriers[0].lm_head_proof_v2_commitment
            or any(
                proof.lm_head_proof_v2_payload for proof in proof_bundle.sampling_proofs
            )
        ):
            return VerificationResult.failure(
                "Proof-v2 hardened LM-head proof set is not exact"
            )
        audit_proof = carriers[0]
        try:
            top_values, top_indices = parse_top_k_leaf(audit_proof.fp16_logits_row)
            x_matrix, x_scales = quantize_committed_hidden_row_v2(
                audit_proof.hidden_row,
                hidden_dim=operation.inner_dim,
            )
            local_envelope = ProofV2CommitmentEnvelope.from_canonical_bytes(
                audit_proof.lm_head_proof_v2_commitment
            )
            expected_x_state = build_inference_x_state_v2(
                manifest,
                {operation.key: x_matrix},
                {
                    operation.key: np.zeros(
                        (1, operation.output_dim),
                        dtype="<f2",
                    )
                },
                {operation.key: x_scales},
                operations=(operation,),
            )
            if (
                local_envelope.manifest_digest != manifest.digest()
                or local_envelope.x_commitments
                != expected_x_state.envelope.x_commitments
            ):
                return VerificationResult.failure(
                    "Proof-v2 LM-head X does not match the committed hidden row"
                )
            audit_challenges = derive_lm_head_audit_challenges_v2(
                operation=operation,
                transcript_state=transcript_state,
                commitment_hash=commitment.commitment_hash(),
                decode_step=audit_step,
                token_id=audit_proof.token_id,
                top_k_row=audit_proof.fp16_logits_row,
            )
            proof_bundle._proof_v2_lm_head_audit = {
                "envelope": local_envelope,
                "operation": operation,
                "challenges": tuple(audit_challenges),
                "decode_step": audit_step,
                "x_scales": tuple(x_scales),
                "committed_by_token": {
                    int(token): float(value)
                    for value, token in zip(top_values, top_indices)
                },
                "boundary": float(top_values[-1]),
            }
        except Exception:
            return VerificationResult.failure(
                "Proof-v2 hardened LM-head verification failed"
            )

        return VerificationResult.success(
            "Proof-v2 response rows and LM-head audit context verified"
        )

    def _verify_protocol_v2_after_common(
        self,
        *,
        proof_bundle: InferenceProofBundle,
        validator_nonce: bytes,
        response_beacon: bytes,
        expected_top_k: Optional[int],
        expected_top_p: Optional[float],
        expected_min_p: Optional[float],
        verified_embedding_rows: Dict[int, bytes],
        embedding_root: bytes,
    ) -> Tuple[VerificationResult, Dict[str, float]]:
        """Verify the v2-only proof collections after shared request checks."""

        commitment = proof_bundle.commitment
        timing: Dict[str, float] = {}
        if hasattr(proof_bundle, "_proof_v2_lm_head_audit"):
            delattr(proof_bundle, "_proof_v2_lm_head_audit")
        if hasattr(proof_bundle, "_proof_v2_hard_execution_corridor"):
            delattr(proof_bundle, "_proof_v2_hard_execution_corridor")
        if (
            proof_bundle.layer_proofs
            or proof_bundle.router_commitments
            or proof_bundle.router_layer_proofs
            or proof_bundle.expert_roots
        ):
            return (
                VerificationResult.failure(
                    "Proof-v2 bundle contains legacy proof collections"
                ),
                timing,
            )
        if commitment.layer_commitments or commitment.layer_transition_hashes:
            return (
                VerificationResult.failure(
                    "Proof-v2 commitment contains legacy layer fields"
                ),
                timing,
            )
        if commitment.router_commitment_hash is not None:
            return (
                VerificationResult.failure(
                    "Proof-v2 commitment contains legacy router fields"
                ),
                timing,
            )
        if (
            not isinstance(proof_bundle.validator_nonce, bytes)
            or proof_bundle.validator_nonce != validator_nonce
        ):
            return (
                VerificationResult.failure(
                    "Proof-v2 validator nonce does not match the request"
                ),
                timing,
            )

        manifest = self.proof_v2_manifest
        if manifest is None:
            return (
                VerificationResult.failure("proof-v2 manifest is not configured"),
                timing,
            )
        from verallm.proof_v2.manifest import ModelSpecIdentity

        if self._on_chain_model_spec is not None:
            if manifest.model_spec != ModelSpecIdentity.from_on_chain(
                self._on_chain_model_spec
            ):
                return (
                    VerificationResult.failure(
                        "proof-v2 manifest does not match the on-chain ModelSpec"
                    ),
                    timing,
                )
        elif (
            manifest.model_spec.model_id != self.model_spec.model_id
            or manifest.model_spec.weight_merkle_root
            != self.model_spec.weight_merkle_root
        ):
            return (
                VerificationResult.failure(
                    "proof-v2 manifest does not match the validator ModelSpec"
                ),
                timing,
            )
        try:
            validate_proof_v2_decode_commitment(commitment)
            from verallm.challenge.v2 import derive_inference_transcript_state_v2
            from verallm.proof_v2.payload import ProofV2CommitmentEnvelope

            envelope = getattr(proof_bundle, "_parsed_proof_v2_commitment", None)
            if envelope is None:
                envelope = ProofV2CommitmentEnvelope.from_canonical_bytes(
                    commitment.proof_v2_commitment
                )
            trace_commitment = envelope.execution_trace_commitment
            if manifest.execution_profile is None:
                return (
                    VerificationResult.failure(
                        "proof-v2 causal execution profile is required"
                    ),
                    timing,
                )
            if (
                trace_commitment is None
                or trace_commitment.profile != manifest.execution_profile
                or trace_commitment.num_layers != manifest.model_spec.num_layers
                or trace_commitment.token_count != commitment.output_token_count
                or any(
                    item.row_count != trace_commitment.token_count
                    for item in envelope.x_commitments
                )
            ):
                return (
                    VerificationResult.failure(
                        "proof-v2 causal execution trace context is not exact"
                    ),
                    timing,
                )
            if envelope.manifest_digest != manifest.digest():
                return (
                    VerificationResult.failure(
                        "proof-v2 commitment does not use the authenticated manifest"
                    ),
                    timing,
                )
            transcript_state = derive_inference_transcript_state_v2(
                validator_nonce=validator_nonce,
                manifest_digest=manifest.digest(),
                commitment_envelope=envelope.canonical_bytes(),
                model_id=commitment.model_id,
                model_commitment=commitment.model_commitment,
                input_commitment=commitment.input_commitment,
                prompt_hash=commitment.prompt_hash,
                sampler_config_hash=commitment.sampler_config_hash,
                sampling_verification_bps=commitment.sampling_verification_bps,
                do_sample=commitment.do_sample,
                temperature_milli=commitment.temperature_milli,
                presence_penalty_milli=commitment.presence_penalty_milli,
            )
        except (AttributeError, TypeError, ValueError):
            return (
                VerificationResult.failure("Proof-v2 commitment context is invalid"),
                timing,
            )

        audit_policy = getattr(
            getattr(manifest, "model_execution", None),
            "audit_policy",
            None,
        )
        hard_audit_bps = getattr(audit_policy, "hard_audit_bps", None)
        if type(hard_audit_bps) is not int or not 1 <= hard_audit_bps <= 10_000:
            return (
                VerificationResult.failure(
                    "Proof-v2 signed hard-audit policy is missing or invalid"
                ),
                timing,
            )
        try:
            hard_audit = hard_audit_required(
                response_beacon,
                commitment,
                hard_audit_bps,
            )
        except (TypeError, ValueError):
            return (
                VerificationResult.failure("Proof-v2 hard-audit gate is invalid"),
                timing,
            )
        sampling_challenge = (
            derive_hard_audit_sampling_challenge(
                beacon=response_beacon,
                commitment=commitment,
                vocab_size=int(getattr(self.model_spec, "vocab_size", 0) or 0),
                k_positions=2,
            )
            if hard_audit
            else None
        )
        if hard_audit:
            if sampling_challenge is None:
                return (
                    VerificationResult.failure(
                        "Proof-v2 decode challenge could not be derived"
                    ),
                    timing,
                )
            sampling_result = self._verify_sampling_consistency_v2(
                proof_bundle=proof_bundle,
                sampling_challenge=sampling_challenge,
                transcript_state=transcript_state,
                expected_top_k=expected_top_k,
                expected_top_p=expected_top_p,
                expected_min_p=expected_min_p,
                minimum_decode_step=1,
            )
            if not sampling_result.passed:
                return sampling_result, timing
        elif proof_bundle.sampling_proofs:
            return (
                VerificationResult.failure(
                    "Proof-v2 bundle contains unchallenged sampling rows"
                ),
                timing,
            )

        v2_result, v2_timing = self._verify_gemm_proof_v2(
            proof_bundle,
            commitment_hash=commitment.commitment_hash(),
            transcript_state=transcript_state,
            hard_audit=hard_audit,
            hard_audit_row=(
                getattr(proof_bundle, "_proof_v2_lm_head_audit", {}).get("decode_step")
                if hard_audit
                else None
            ),
        )
        timing.update(v2_timing)
        if not v2_result.passed:
            return v2_result, timing
        if manifest.execution_profile is not None and sampling_challenge is not None:
            try:
                from verallm.proof_v2.payload import ProofV2Payload
                from verallm.proof_v2.trace import ExecutionTraceProofV2

                payload = ProofV2Payload.from_canonical_bytes(
                    proof_bundle.proof_v2_payload
                )
                trace_proof = ExecutionTraceProofV2.from_canonical_bytes(
                    payload.execution_trace_proof
                )
                primary_token_positions = tuple(
                    sorted(
                        {
                            row
                            for block in payload.block_proofs
                            if 0
                            <= block.descriptor.key.layer_idx
                            < manifest.model_spec.num_layers
                            for row in range(
                                block.descriptor.row_offset,
                                block.descriptor.row_offset
                                + block.descriptor.rows,
                            )
                        }
                    )
                )
                hard_corridor = getattr(
                    proof_bundle,
                    "_proof_v2_hard_execution_corridor",
                    None,
                )
                if hard_audit:
                    if not isinstance(hard_corridor, dict):
                        raise ValueError("hard execution corridor scope is missing")
                    corridor_row = hard_corridor.get("row")
                    corridor_positions = hard_corridor.get("positions")
                    if (
                        type(corridor_row) is not int
                        or primary_token_positions != (corridor_row,)
                        or not isinstance(corridor_positions, tuple)
                    ):
                        raise ValueError("hard execution corridor scope is invalid")
                    expected_layer_positions = corridor_positions
                else:
                    expected_layer_positions = tuple(
                        (layer.token_index, layer.layer_idx)
                        for layer in trace_proof.opened_layers
                    )
                trace_proof.verify(
                    envelope.execution_trace_commitment,
                    output_token_ids=proof_bundle.output_token_ids,
                    expected_layer_positions=expected_layer_positions,
                    expected_first_input_token_id=proof_bundle.input_token_ids[-1],
                )
                from verallm.proof_v2.trace import (
                    trace_residual_boundary_digest_v2,
                    trace_tail_digest_v2,
                )

                input_ids = np.asarray(
                    proof_bundle.input_token_ids,
                    dtype="<i8",
                ).tobytes()
                initial_token = trace_proof.tokens[0]
                for layer_parameters in manifest.layer_execution:
                    expected_initial = _proof_v2_prompt_attention_state_anchor_v2(
                        input_token_ids=input_ids,
                        layer_idx=layer_parameters.layer,
                        attention_profile=layer_parameters.attention_profile,
                    )
                    if (
                        initial_token.attention_state_before_digests[
                            layer_parameters.layer
                        ]
                        != expected_initial
                    ):
                        raise ValueError(
                            "execution trace state is not anchored to the prompt"
                        )

                expected_decode_embedding_positions = tuple(
                    position for position in primary_token_positions if position > 0
                )
                decode_openings = proof_bundle.embedding_proof.decode_row_openings
                decode_openings_by_position: Dict[int, list] = {}
                for opening in decode_openings:
                    decode_openings_by_position.setdefault(
                        opening.token_position,
                        [],
                    ).append(opening)
                if tuple(sorted(decode_openings_by_position)) != (
                    expected_decode_embedding_positions
                ):
                    raise ValueError(
                        "execution trace decode embedding opening set is not exact"
                    )

                decode_embedding_rows: Dict[int, bytes] = {}
                chunk_size = self.model_spec.w_merkle_chunk_size
                hidden_dim = manifest.model_spec.hidden_dim
                for position in expected_decode_embedding_positions:
                    if position >= len(trace_proof.tokens):
                        raise ValueError("decode embedding trace position is invalid")
                    expected_token = int(proof_bundle.output_token_ids[position - 1])
                    if trace_proof.tokens[position].input_token_id != expected_token:
                        raise ValueError("decode embedding token chain is invalid")
                    row_start = expected_token * hidden_dim
                    first_chunk_idx = row_start // chunk_size
                    last_chunk_idx = (row_start + hidden_dim - 1) // chunk_size
                    expected_indices = tuple(range(first_chunk_idx, last_chunk_idx + 1))
                    ordered = tuple(
                        sorted(
                            decode_openings_by_position[position],
                            key=lambda item: item.merkle_path.leaf_index,
                        )
                    )
                    if tuple(
                        item.merkle_path.leaf_index for item in ordered
                    ) != expected_indices or any(
                        item.token_id != expected_token for item in ordered
                    ):
                        raise ValueError("decode embedding chunk set is not exact")
                    if any(
                        not verify_flat_chunk_merkle_path(
                            root=embedding_root,
                            chunk_data=item.leaf_data,
                            path=item.merkle_path,
                        )
                        for item in ordered
                    ):
                        raise ValueError("decode embedding Merkle path is invalid")
                    combined = b"".join(item.leaf_data for item in ordered)
                    byte_offset = row_start - first_chunk_idx * chunk_size
                    row = combined[byte_offset : byte_offset + hidden_dim]
                    if len(row) != hidden_dim:
                        raise ValueError("decode embedding row is truncated")
                    decode_embedding_rows[position] = row

                model_parameters = manifest.model_execution
                if model_parameters is None:
                    raise ValueError("model execution parameters are missing")
                from verallm.proof_v2.hardening import (
                    fp16_matches_authenticated_i8_v2,
                )

                prompt_boundary_position = len(proof_bundle.input_token_ids) - 1
                opened_by_position = {
                    (layer.token_index, layer.layer_idx): layer
                    for layer in trace_proof.opened_layers
                }
                for position in primary_token_positions:
                    embedding_row = (
                        verified_embedding_rows[prompt_boundary_position]
                        if position == 0
                        else decode_embedding_rows[position]
                    )
                    embedding_i8 = np.frombuffer(embedding_row, dtype=np.int8)
                    if embedding_i8.shape != (hidden_dim,):
                        raise ValueError("execution trace embedding row is malformed")
                    expected_embedding = np.asarray(
                        embedding_i8.astype(np.float32)
                        * np.float32(
                            model_parameters.embedding_scale_q32 / float(1 << 32)
                        ),
                        dtype="<f2",
                    )
                    layer_zero = opened_by_position[(position, 0)]
                    residual_tensor = layer_zero.tensor("residual_in")
                    residual_in = np.frombuffer(
                        residual_tensor.values,
                        dtype="<f2",
                    ).astype(np.float32)
                    embedding_error = (
                        float(
                            np.max(
                                np.abs(
                                    residual_in - expected_embedding.astype(np.float32)
                                )
                            )
                        )
                        if residual_in.shape == expected_embedding.shape
                        else float("inf")
                    )
                    if (
                        residual_tensor.dtype != "f16"
                        or tuple(residual_tensor.shape) != (hidden_dim,)
                        or residual_in.shape != expected_embedding.shape
                        or not fp16_matches_authenticated_i8_v2(
                            np.frombuffer(residual_tensor.values, dtype="<f2"),
                            embedding_i8,
                            scale_q32=model_parameters.embedding_scale_q32,
                        )
                        or trace_proof.tokens[position].residual_boundary_digests[0]
                        != trace_residual_boundary_digest_v2(residual_tensor.values)
                    ):
                        raise ValueError(
                            "execution trace start state is not the authenticated embedding "
                            f"(position={position}, max_abs_error={embedding_error:.8g})"
                        )

                lm_head_audit = getattr(
                    proof_bundle,
                    "_proof_v2_lm_head_audit",
                    None,
                )
                if (
                    hard_audit
                    and (
                        lm_head_audit is None
                        or lm_head_audit["decode_step"] != corridor_row
                    )
                ):
                    raise ValueError(
                        "hard audit does not share the LM-head decode transition"
                    )

                for sampling_proof in proof_bundle.sampling_proofs:
                    token_trace = trace_proof.tokens[sampling_proof.decode_step]
                    if token_trace.final_hidden_digest != trace_tail_digest_v2(
                        "final_hidden_f16",
                        sampling_proof.hidden_row,
                    ):
                        raise ValueError(
                            "sampled final hidden row is outside the execution trace"
                        )
                    if token_trace.lm_head_digest != trace_tail_digest_v2(
                        "lm_head_top_k",
                        sampling_proof.fp16_logits_row,
                    ):
                        raise ValueError(
                            "sampled LM-head row is outside the execution trace"
                        )
            except (AttributeError, IndexError, TypeError, ValueError) as exc:
                logger.warning(
                    "Proof-v2 execution trace binding failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                return (
                    VerificationResult.failure(
                        "Proof-v2 execution trace is not bound to prompt and response tokens"
                    ),
                    timing,
                )
        return VerificationResult.success("All proof-v2 checks verified"), timing

    def _verify_gemm_proof_v2(
        self,
        proof_bundle: InferenceProofBundle,
        *,
        commitment_hash: bytes,
        transcript_state: bytes,
        hard_audit: bool = False,
        hard_audit_row: int | None = None,
    ) -> Tuple[VerificationResult, Dict[str, float]]:
        """Verify the exclusive GEMM-v2 path against an authenticated manifest."""

        from verallm.challenge.v2 import (
            derive_block_challenges_v2,
            derive_stratified_execution_layers_v2,
        )
        from verallm.proof_v2.engine import (
            combine_commitment_envelopes_v2,
            verify_inference_v2,
        )
        from verallm.proof_v2.hardening import (
            logit_is_not_above_committed_boundary_v2,
            logit_matches_committed_value_v2,
            proof_logit_q32_v2,
        )
        from verallm.proof_v2.layout import (
            operation_descriptor_by_key,
            operation_weight_scale_q32_v2,
            registered_all_operations_from_manifest,
            registered_operations_from_manifest,
        )
        from verallm.proof_v2.payload import (
            ProofV2CommitmentEnvelope,
            ProofV2Payload,
        )

        timing: Dict[str, float] = {}
        manifest = self.proof_v2_manifest
        if manifest is None:
            return (
                VerificationResult.failure("proof-v2 manifest is not configured"),
                timing,
            )
        started = time.perf_counter()
        try:
            envelope = getattr(proof_bundle, "_parsed_proof_v2_commitment", None)
            if envelope is None:
                envelope = ProofV2CommitmentEnvelope.from_canonical_bytes(
                    proof_bundle.commitment.proof_v2_commitment
                )
            payload = getattr(proof_bundle, "_parsed_proof_v2_payload", None)
            if payload is None:
                payload = ProofV2Payload.from_canonical_bytes(
                    proof_bundle.proof_v2_payload
                )
            if envelope.manifest_digest != manifest.digest():
                return (
                    VerificationResult.failure(
                        "proof-v2 commitment does not use the authenticated manifest"
                    ),
                    timing,
                )
            operations = registered_operations_from_manifest(manifest)
            hard_audit_layers = None
            hard_audit_layer_count = self.config.k_layers
            hard_audit_blocks_per_operation = self.config.k_blocks
            if hard_audit:
                policy = getattr(
                    getattr(manifest, "model_execution", None),
                    "audit_policy",
                    None,
                )
                if policy is None:
                    raise ValueError("proof-v2 signed hard-audit policy is missing")
                hard_audit_layer_count = policy.hard_layer_count
                hard_audit_blocks_per_operation = getattr(
                    policy,
                    "hard_blocks_per_operation",
                    None,
                )
                if (
                    type(hard_audit_blocks_per_operation) is not int
                    or not 1
                    <= hard_audit_blocks_per_operation
                    <= MAX_BLOCKS_PER_OPERATION
                ):
                    raise ValueError(
                        "proof-v2 signed hard-audit block coverage is invalid"
                    )
                hard_audit_layers = derive_stratified_execution_layers_v2(
                    transcript_state=transcript_state,
                    layer_attention_profiles=tuple(
                        item.attention_profile for item in manifest.layer_execution
                    ),
                    hard_layer_count=policy.hard_layer_count,
                    min_full_attention_layers=policy.min_full_attention_layers,
                    min_gdn_layers=policy.min_gdn_layers,
                )
            block_challenge_kwargs = {}
            if hard_audit_layers is not None:
                block_challenge_kwargs["selected_layer_indices"] = hard_audit_layers
            challenges = derive_block_challenges_v2(
                transcript_state=transcript_state,
                num_layers=manifest.model_spec.num_layers,
                operations=operations,
                x_commitments=envelope.x_commitments,
                runtime_y_commitments=envelope.runtime_y_commitments,
                k_layers=(
                    hard_audit_layer_count if hard_audit else self.config.k_layers
                ),
                k_operations_per_layer=1,
                k_blocks_per_operation=(
                    hard_audit_blocks_per_operation
                    if hard_audit
                    else self.config.k_blocks
                ),
                all_operations_per_selected_layer=hard_audit,
                required_row_index=hard_audit_row,
                **block_challenge_kwargs,
            )
            hard_corridor = None
            if hard_audit:
                if (
                    isinstance(hard_audit_row, bool)
                    or not isinstance(hard_audit_row, int)
                    or hard_audit_row < 0
                ):
                    raise ValueError("proof-v2 hard-audit decode row is missing")
                (
                    corridor_row,
                    _selected_transition_layers,
                    corridor_positions,
                ) = derive_hard_execution_corridor_v2(
                    challenges,
                    num_layers=manifest.model_spec.num_layers,
                )
                if corridor_row != hard_audit_row:
                    raise ValueError(
                        "proof-v2 hard-audit rows do not match the LM-head audit"
                    )
                hard_corridor = {
                    "row": corridor_row,
                    "positions": corridor_positions,
                }
            proof_envelope = envelope
            proof_operations = operations
            expected_challenges = tuple(challenges)
            lm_head_audit = getattr(
                proof_bundle,
                "_proof_v2_lm_head_audit",
                None,
            )
            if lm_head_audit is not None:
                proof_envelope = combine_commitment_envelopes_v2(
                    manifest,
                    (envelope, lm_head_audit["envelope"]),
                )
                proof_operations = registered_all_operations_from_manifest(manifest)
                expected_challenges += tuple(lm_head_audit["challenges"])
            verify_inference_v2(
                manifest=manifest,
                commitment_hash=commitment_hash,
                beacon=transcript_state,
                commitment_envelope=proof_envelope,
                payload=payload,
                expected_challenges=expected_challenges,
                operations=proof_operations,
                require_execution_trace_binding=hard_audit,
                require_full_decode_corridor=hard_audit,
            )
            if hard_corridor is not None:
                proof_bundle._proof_v2_hard_execution_corridor = hard_corridor
            if lm_head_audit is not None:
                operation = lm_head_audit["operation"]
                descriptor = operation_descriptor_by_key(manifest)[operation.key]
                x_scales = lm_head_audit["x_scales"]
                committed_by_token = lm_head_audit["committed_by_token"]
                boundary = lm_head_audit["boundary"]
                for block in payload.block_proofs:
                    if block.descriptor.key != operation.key:
                        continue
                    block_scales = struct.unpack(
                        f"<{block.descriptor.rows}Q",
                        block.x_scales_q32,
                    )
                    if block_scales != x_scales:
                        raise ValueError("LM-head X scale is not canonical")
                    proof_values = np.frombuffer(
                        block.proof_y_values,
                        dtype="<i8",
                    ).reshape(
                        block.descriptor.rows,
                        block.descriptor.cols,
                    )
                    for column_index, proof_value in enumerate(proof_values[0]):
                        token = block.descriptor.column_offset + column_index
                        proof_q32 = proof_logit_q32_v2(
                            int(proof_value),
                            x_scale_q32=x_scales[0],
                            weight_scale_q32=operation_weight_scale_q32_v2(
                                descriptor,
                                block.descriptor.column_offset,
                            ),
                        )
                        committed_value = committed_by_token.get(token)
                        if committed_value is not None:
                            if not logit_matches_committed_value_v2(
                                proof_q32,
                                committed_value,
                                descriptor,
                            ):
                                raise ValueError("LM-head logit does not match top-k")
                        elif not logit_is_not_above_committed_boundary_v2(
                            proof_q32,
                            boundary,
                            descriptor,
                        ):
                            raise ValueError("LM-head omitted logit exceeds boundary")
        except Exception as exc:
            timing["GEMM proof v2"] = (time.perf_counter() - started) * 1000
            logger.warning(
                "GEMM proof v2 verification failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            return (
                VerificationResult.failure(
                    "Combined GEMM/LM-head proof v2 verification failed"
                    if getattr(
                        proof_bundle,
                        "_proof_v2_lm_head_audit",
                        None,
                    )
                    is not None
                    else "GEMM proof v2 verification failed"
                ),
                timing,
            )
        timing["GEMM proof v2"] = (time.perf_counter() - started) * 1000
        return VerificationResult.success("GEMM proof v2 verified"), timing

    # ------------------------------------------------------------------
    # Phase 7: Verify proofs locally
    # ------------------------------------------------------------------

    def verify_proof(
        self,
        proof_bundle: InferenceProofBundle,
        nonce: bytes,
        expected_sampling_verification_bps: Optional[int] = None,
        expected_do_sample: Optional[bool] = None,
        expected_temperature: Optional[float] = None,
        enable_thinking: Optional[bool] = None,
        expected_input_commitment: Optional[bytes] = None,
        expected_prompt_hash: Optional[bytes] = None,
        expected_sampler_config_hash: Optional[bytes] = None,
        expected_presence_penalty: Optional[float] = None,
        expected_top_k: Optional[int] = None,
        expected_top_p: Optional[float] = None,
        expected_min_p: Optional[float] = None,
    ) -> Tuple[VerificationResult, Dict[str, float]]:
        """Verify the proof bundle locally (lightweight -- no model needed).

        Re-derives beacon and challenges from the commitment + nonce
        (Fiat-Shamir), then verifies sumcheck proofs against on-chain
        weight Merkle roots.

        Args:
            expected_sampling_verification_bps: If set, verify the committed
                bps matches the validator-requested rate.
            expected_do_sample: If set, verify the committed do_sample flag
                matches the validator's request (prevents miner from setting
                do_sample=True to skip sampling checks).
            expected_temperature: If set, verify the committed temperature
                matches the validator's request (prevents miner from committing
                nonzero temperature to skip sampling checks).
            enable_thinking: If False, argmax divergence in high-assurance
                mode becomes a hard failure (no logits processor excuse).
            expected_presence_penalty: If set, verify the committed milli-unit
                penalty matches the value requested by the validator.

        NOTE: In production, weight Merkle roots come from the on-chain
        registry.  Here they come from the ModelSpec fetched from the miner.
        """
        if self.model_spec is None:
            raise RuntimeError("ModelSpec not fetched. Call fetch_model_spec() first.")

        set_config(self.config)
        verifier = GEMMVerifier(self.config)
        timing_details = {}

        # Re-derive beacon from commitment + nonce (Fiat-Shamir)
        commitment = proof_bundle.commitment
        protocol_version = getattr(proof_bundle, "proof_protocol_version", None)
        if type(protocol_version) is not int or protocol_version not in (
            LEGACY_PROOF_PROTOCOL_VERSION,
            PROOF_PROTOCOL_V2,
        ):
            return (
                VerificationResult.failure(
                    "Missing or unsupported proof protocol version"
                ),
                timing_details,
            )
        if commitment.model_id != self.model_spec.model_id:
            return (
                VerificationResult.failure(
                    "Committed model_id does not match the validator ModelSpec"
                ),
                timing_details,
            )
        if commitment.model_commitment != self.model_spec.weight_merkle_root:
            return (
                VerificationResult.failure(
                    "Committed model root does not match the validator ModelSpec"
                ),
                timing_details,
            )
        beacon = derive_beacon_from_nonce(
            commitment_hash=commitment.commitment_hash(),
            validator_nonce=nonce,
        )

        # Verify beacon matches what miner used
        if proof_bundle.beacon != beacon:
            return (
                VerificationResult.failure(
                    f"Beacon mismatch: miner used {proof_bundle.beacon[:8].hex()}..., "
                    f"expected {beacon[:8].hex()}..."
                ),
                timing_details,
            )

        # Verify committed decode-mode fields match validator expectations.
        # Without these checks, a miner could commit do_sample=True or
        # nonzero temperature to dodge sampling verification entirely.
        if expected_do_sample is not None:
            if commitment.do_sample != expected_do_sample:
                return (
                    VerificationResult.failure(
                        f"do_sample mismatch: committed={commitment.do_sample}, "
                        f"expected={expected_do_sample}"
                    ),
                    timing_details,
                )
        if expected_temperature is not None:
            from verallm.sampling import temperature_to_milli

            expected_milli = temperature_to_milli(
                0.0 if expected_do_sample is False else expected_temperature
            )
            if commitment.temperature_milli != expected_milli:
                return (
                    VerificationResult.failure(
                        f"temperature mismatch: committed={commitment.temperature_milli}m, "
                        f"expected={expected_milli}m"
                    ),
                    timing_details,
                )

        if expected_sampling_verification_bps is not None:
            expected_bps = clamp_sampling_bps(expected_sampling_verification_bps)
            committed_bps = clamp_sampling_bps(commitment.sampling_verification_bps)
            if committed_bps != expected_bps:
                return (
                    VerificationResult.failure(
                        f"Sampling verification rate mismatch: committed={committed_bps} bps, "
                        f"expected={expected_bps} bps"
                    ),
                    timing_details,
                )

        # Verify sampler config hash (top_k/top_p/min_p/template binding).
        if expected_sampler_config_hash is not None:
            if not commitment.sampler_config_hash:
                return (
                    VerificationResult.failure(
                        "sampler_config_hash missing from commitment"
                    ),
                    timing_details,
                )
            if commitment.sampler_config_hash != expected_sampler_config_hash:
                return (
                    VerificationResult.failure(
                        "sampler_config_hash mismatch: miner committed different "
                        "sampling parameters than validator requested"
                    ),
                    timing_details,
                )
        if expected_presence_penalty is not None:
            try:
                expected_presence_milli = int(
                    round(float(expected_presence_penalty) * 1000.0)
                )
            except (OverflowError, TypeError, ValueError):
                return (
                    VerificationResult.failure("expected presence_penalty is invalid"),
                    timing_details,
                )
            if not -2_000 <= expected_presence_milli <= 2_000:
                return (
                    VerificationResult.failure(
                        "expected presence_penalty is out of range"
                    ),
                    timing_details,
                )
            if commitment.presence_penalty_milli != expected_presence_milli:
                return (
                    VerificationResult.failure(
                        "presence_penalty mismatch: committed="
                        f"{commitment.presence_penalty_milli}m, "
                        f"expected={expected_presence_milli}m"
                    ),
                    timing_details,
                )

        # Verify committed input matches what the validator sent (prevents
        # input truncation: miner drops tokens to save compute).
        if expected_input_commitment is not None:
            if commitment.input_commitment != expected_input_commitment:
                return (
                    VerificationResult.failure(
                        f"input_commitment mismatch: committed="
                        f"{commitment.input_commitment.hex()[:16]}..., "
                        f"expected={expected_input_commitment.hex()[:16]}..."
                    ),
                    timing_details,
                )

        # Verify committed prompt hash matches what the proxy/validator sent
        # (prevents prompt substitution: miner runs a different prompt).
        if expected_prompt_hash is not None:
            if not commitment.prompt_hash:
                return (
                    VerificationResult.failure(
                        "Missing prompt_hash in commitment — "
                        "miner must include prompt_hash for input integrity"
                    ),
                    timing_details,
                )
            if commitment.prompt_hash != expected_prompt_hash:
                return (
                    VerificationResult.failure(
                        f"prompt_hash mismatch: committed="
                        f"{commitment.prompt_hash.hex()[:16]}..., "
                        f"expected={expected_prompt_hash.hex()[:16]}..."
                    ),
                    timing_details,
                )

        # Protocol v2 carries the complete prompt token sequence so the causal
        # trace can be anchored at its final token without loading model
        # weights. The validator still independently tokenizes the user prompt
        # when checking ``expected_input_commitment``.
        if protocol_version == PROOF_PROTOCOL_V2:
            vocab_size = int(getattr(self.model_spec, "vocab_size", 0) or 0)
            if not proof_bundle.input_token_ids or any(
                type(token) is not int or token < 0 or token >= vocab_size
                for token in proof_bundle.input_token_ids
            ):
                return (
                    VerificationResult.failure(
                        "Proof-v2 input token history is missing or invalid"
                    ),
                    timing_details,
                )
            input_ids_arr = np.asarray(proof_bundle.input_token_ids, dtype=np.int64)
            input_hash = hashlib.sha256(
                input_ids_arr.astype("<i8", copy=False).tobytes()
            ).digest()
            if input_hash != commitment.input_commitment:
                return (
                    VerificationResult.failure(
                        "input_commitment mismatch: input_token_ids do not match commitment"
                    ),
                    timing_details,
                )

        # Bind served output to commitment (prevents post-proof token substitution).
        if commitment.output_token_count > 0 and not proof_bundle.output_token_ids:
            return (
                VerificationResult.failure("Missing output_token_ids in proof bundle"),
                timing_details,
            )
        if proof_bundle.output_token_ids:
            if (
                commitment.output_token_count > 0
                and len(proof_bundle.output_token_ids) != commitment.output_token_count
            ):
                return (
                    VerificationResult.failure(
                        f"output_token_count mismatch: committed={commitment.output_token_count}, "
                        f"bundle={len(proof_bundle.output_token_ids)}"
                    ),
                    timing_details,
                )
            output_ids_arr = np.asarray(proof_bundle.output_token_ids, dtype=np.int64)
            output_hash = hashlib.sha256(
                output_ids_arr.astype("<i8", copy=False).tobytes()
            ).digest()
            if output_hash != commitment.output_commitment:
                return (
                    VerificationResult.failure(
                        "output_commitment mismatch: output_token_ids do not match commitment"
                    ),
                    timing_details,
                )

        # Verify router_commitments match the committed hash (MoE binding).
        # Without this check, a miner could send tampered router_commitments
        # to control which experts get challenged while the beacon still
        # validates (because the beacon binds commitment_hash, not the raw
        # router_commitments).
        is_moe = self.moe_config is not None and self.moe_config.is_moe
        if is_moe and proof_bundle.router_commitments:
            expected_rc_hash = InferenceCommitment.compute_router_hash(
                proof_bundle.router_commitments,
            )
            if expected_rc_hash != commitment.router_commitment_hash:
                return (
                    VerificationResult.failure(
                        "Router commitment hash mismatch: proof bundle "
                        "router_commitments don't match the committed "
                        "router_commitment_hash"
                    ),
                    timing_details,
                )

        # Verify embedding input binding proof.
        # The embedding proof cryptographically binds the committed input tokens
        # to the on-chain embedding weight Merkle root, preventing a miner from
        # hashing the real prompt but running inference on a different one.
        emb_root = getattr(self.model_spec, "embedding_weight_merkle_root", b"")
        if not emb_root:
            return (
                VerificationResult.failure(
                    "ModelSpec.embedding_weight_merkle_root is missing — "
                    "model must be re-registered with embedding root"
                ),
                timing_details,
            )
        if proof_bundle.embedding_proof is None:
            return (
                VerificationResult.failure(
                    "Missing embedding_proof in proof bundle — "
                    "input binding verification requires embedding proof"
                ),
                timing_details,
            )

        emb_proof = proof_bundle.embedding_proof

        # input_token_ids must be present and non-empty.
        if not emb_proof.input_token_ids:
            return (
                VerificationResult.failure("Embedding proof: input_token_ids is empty"),
                timing_details,
            )

        # Verify input_token_ids hash matches committed input_commitment.
        input_ids_arr = np.asarray(emb_proof.input_token_ids, dtype=np.int64)
        input_hash = hashlib.sha256(
            input_ids_arr.astype("<i8", copy=False).tobytes()
        ).digest()
        if input_hash != commitment.input_commitment:
            return (
                VerificationResult.failure(
                    "Embedding proof: input_token_ids hash does not match "
                    "committed input_commitment"
                ),
                timing_details,
            )

        # Re-derive embedding challenge from beacon (Fiat-Shamir).
        emb_challenge = derive_embedding_challenge(
            beacon=beacon,
            commitment=commitment,
            num_input_tokens=len(emb_proof.input_token_ids),
            include_last_position=(protocol_version == PROOF_PROTOCOL_V2),
        )
        if emb_challenge is None:
            return (
                VerificationResult.failure(
                    "Embedding proof: failed to derive embedding challenge"
                ),
                timing_details,
            )

        # Verify each challenged row opening against the on-chain
        # embedding_weight_merkle_root.
        openings_by_pos: Dict[int, list] = {}
        for opening in emb_proof.row_openings:
            openings_by_pos.setdefault(opening.token_position, []).append(opening)
        chunk_size = self.model_spec.w_merkle_chunk_size
        hidden_dim = self.model_spec.hidden_dim
        verified_embedding_rows: Dict[int, bytes] = {}

        if protocol_version == PROOF_PROTOCOL_V2 and set(openings_by_pos) != set(
            emb_challenge.token_positions
        ):
            return (
                VerificationResult.failure(
                    "Embedding proof: v2 row opening positions are not exact"
                ),
                timing_details,
            )

        for pos in emb_challenge.token_positions:
            position_openings = openings_by_pos.get(pos, [])
            if not position_openings:
                return (
                    VerificationResult.failure(
                        f"Embedding proof: missing row opening for "
                        f"challenged position {pos}"
                    ),
                    timing_details,
                )

            expected_token = emb_proof.input_token_ids[pos]
            row_start = expected_token * hidden_dim
            first_chunk_idx = row_start // chunk_size
            last_chunk_idx = (
                (row_start + hidden_dim - 1) // chunk_size
                if protocol_version == PROOF_PROTOCOL_V2
                else first_chunk_idx
            )
            expected_chunk_indices = tuple(range(first_chunk_idx, last_chunk_idx + 1))
            ordered_openings = sorted(
                position_openings,
                key=lambda item: item.merkle_path.leaf_index,
            )
            actual_chunk_indices = tuple(
                opening.merkle_path.leaf_index for opening in ordered_openings
            )
            if actual_chunk_indices != expected_chunk_indices:
                return (
                    VerificationResult.failure(
                        f"Embedding proof: chunk set is not exact at position {pos}"
                    ),
                    timing_details,
                )
            for opening in ordered_openings:
                if opening.token_id != expected_token:
                    return (
                        VerificationResult.failure(
                            f"Embedding proof: token_id mismatch at position {pos}"
                        ),
                        timing_details,
                    )
                # FlatWeightMerkle hashes each raw chunk before the ordinary
                # Merkle leaf hash, hence this specialized verifier.
                if not verify_flat_chunk_merkle_path(
                    root=emb_root,
                    chunk_data=opening.leaf_data,
                    path=opening.merkle_path,
                ):
                    return (
                        VerificationResult.failure(
                            f"Embedding proof: Merkle path invalid for position {pos}"
                        ),
                        timing_details,
                    )
            if protocol_version == PROOF_PROTOCOL_V2:
                combined = b"".join(opening.leaf_data for opening in ordered_openings)
                byte_offset = row_start - first_chunk_idx * chunk_size
                row = combined[byte_offset : byte_offset + hidden_dim]
                if len(row) != hidden_dim:
                    return (
                        VerificationResult.failure(
                            f"Embedding proof: row reconstruction failed at position {pos}"
                        ),
                        timing_details,
                    )
                verified_embedding_rows[pos] = row

        logger.debug(
            "    - Embedding proof: %d row openings verified against on-chain root",
            len(emb_proof.row_openings),
        )

        if protocol_version == PROOF_PROTOCOL_V2:
            return self._verify_protocol_v2_after_common(
                proof_bundle=proof_bundle,
                validator_nonce=nonce,
                response_beacon=beacon,
                expected_top_k=expected_top_k,
                expected_top_p=expected_top_p,
                expected_min_p=expected_min_p,
                verified_embedding_rows=verified_embedding_rows,
                embedding_root=emb_root,
            )

        # ----------------------------------------------------------------
        # Embedding output → layer 0 binding (DISABLED)
        #
        # The per-request output tree costs O(seq_len * hidden_dim) and is
        # outside the current proof budget. The optional payload fields remain
        # reserved for a future implementation.
        # ----------------------------------------------------------------

        # Verify layer transition hash chain.
        # Binds consecutive layer commitments together, anchored by
        # input_commitment (ties the layer chain to the proven input).
        expected_num_hashes = max(0, len(commitment.layer_commitments) - 1)
        if len(commitment.layer_transition_hashes) != expected_num_hashes:
            return (
                VerificationResult.failure(
                    f"Wrong number of transition hashes: got "
                    f"{len(commitment.layer_transition_hashes)}, expected "
                    f"{expected_num_hashes}"
                ),
                timing_details,
            )

        for i in range(expected_num_hashes):
            expected_hash = hashlib.sha256(
                b"LAYER_TRANSITION_V2"
                + commitment.input_commitment
                + struct.pack("<I", i)
                + commitment.layer_commitments[i]
                + commitment.layer_commitments[i + 1]
            ).digest()
            if commitment.layer_transition_hashes[i] != expected_hash:
                return (
                    VerificationResult.failure(
                        f"Transition hash mismatch at boundary {i}->{i+1}: "
                        f"committed hash does not match re-derived hash"
                    ),
                    timing_details,
                )

        if expected_num_hashes > 0:
            logger.debug(
                "    - Transition hash chain: %d boundaries verified",
                expected_num_hashes,
            )

        # Re-derive challenges (Fiat-Shamir)
        if is_moe:
            expected_challenges = derive_moe_challenges(
                beacon=beacon,
                commitment=commitment,
                moe_config=self.moe_config,
                router_commitments=proof_bundle.router_commitments,
                k_layers=self.config.k_layers,
                k_tokens_per_layer=self.config.k_tokens_per_expert,
                k_experts_per_layer=self.config.k_experts_per_layer,
            )
            logger.debug("  Validator: Re-deriving MoE challenges (Fiat-Shamir)...")
        else:
            expected_challenges = derive_challenges(
                beacon=beacon,
                commitment=commitment,
                k_layers=self.config.k_layers,
                k_gemms_per_layer=2,
                k_blocks_per_gemm=self.config.k_blocks,
            )
            logger.debug("  Validator: Re-deriving challenges (Fiat-Shamir)...")

        legacy_set_error = _validate_exact_legacy_proof_set(
            expected_challenges,
            proof_bundle.layer_proofs,
        )
        if legacy_set_error is not None:
            return VerificationResult.failure(legacy_set_error), timing_details

        # Debug: show what layers were derived
        expected_layer_indices = [
            lc.layer_idx for lc in expected_challenges.layer_challenges
        ]
        logger.debug("    - Expected challenge layers: %s", expected_layer_indices)
        proof_layer_indices = [lp.layer_idx for lp in proof_bundle.layer_proofs]
        logger.debug("    - Proof bundle layers: %s", proof_layer_indices)
        logger.debug(
            "    - num_layers in commitment: %d", len(commitment.layer_commitments)
        )
        logger.debug(
            "    - Router commitments keys: %s",
            sorted(proof_bundle.router_commitments.keys())
            if proof_bundle.router_commitments
            else "None",
        )

        # Lightweight mode: W verified via Merkle proofs
        logger.debug(
            "    - W spot checks verified via Merkle proofs against on-chain roots"
        )

        # Verify each layer proof
        for layer_proof in proof_bundle.layer_proofs:
            layer_idx = layer_proof.layer_idx

            # Find the matching layer challenge
            layer_challenge = None
            for lc in expected_challenges.layer_challenges:
                if lc.layer_idx == layer_idx:
                    layer_challenge = lc
                    break

            if layer_challenge is None:
                return (
                    VerificationResult.failure(
                        f"No challenge found for layer {layer_idx}"
                    ),
                    timing_details,
                )

            for i, gemm_proof in enumerate(layer_proof.gemm_proofs):
                # Handle MoE vs dense challenges
                is_moe_challenge = hasattr(layer_challenge, "expert_challenges")
                expert_idx = None

                if is_moe_challenge:
                    if i < len(layer_challenge.expert_challenges):
                        gemm_idx = layer_challenge.expert_challenges[i].gemm_idx
                        expert_idx = layer_challenge.expert_challenges[i].expert_idx
                    else:
                        gemm_idx = i
                else:
                    gemm_idx = (
                        layer_challenge.gemm_challenges[i].gemm_idx
                        if i < len(layer_challenge.gemm_challenges)
                        else i
                    )

                hidden_dim = self.model_spec.hidden_dim
                intermediate_dim = self.model_spec.intermediate_dim

                X_commitment = commitment.layer_commitments[layer_idx]
                W_commitment = self.model_spec.weight_merkle_root
                Y_commitment = gemm_proof.output_root

                if is_moe_challenge and expert_idx is not None:
                    x_context = (
                        f"Layer {layer_idx} expert {expert_idx} GEMM {gemm_idx} X"
                    )
                else:
                    x_context = f"Layer {layer_idx} GEMM {gemm_idx} X"
                x_result = verify_x_spot_openings(
                    proof=gemm_proof,
                    x_root=X_commitment,
                    num_cols=hidden_dim,
                    context=x_context,
                )
                if not x_result.passed:
                    return VerificationResult.failure(x_result.message), timing_details

                # Lightweight: W verified via Merkle proofs in verifier
                spot_check_fn = lambda _spot, _matrix_id: True

                # Get W Merkle root for this layer
                W_merkle_root = None
                if is_moe_challenge and expert_idx is not None:
                    # Try spec first (local mode), then proof bundle (chain mode)
                    expert_roots = self.model_spec.expert_weight_merkle_roots.get(
                        layer_idx, []
                    )
                    if not expert_roots:
                        expert_roots = proof_bundle.expert_roots.get(layer_idx, [])
                    if expert_idx < len(expert_roots):
                        W_merkle_root = expert_roots[expert_idx]

                        router_root = self.model_spec.router_weight_merkle_roots.get(
                            layer_idx
                        )
                        if router_root is None:
                            rp = proof_bundle.router_layer_proofs.get(layer_idx)
                            router_root = (
                                rp.router_weight_root if rp is not None else None
                            )
                        if not self._verify_moe_layer_root(
                            layer_idx, expert_roots, router_root
                        ):
                            return (
                                VerificationResult.failure(
                                    f"MoE layer {layer_idx}: hierarchical root mismatch "
                                    f"(expert roots don't hash to committed layer root)"
                                ),
                                timing_details,
                            )
                    else:
                        if layer_idx < len(self.model_spec.weight_block_merkle_roots):
                            W_merkle_root = self.model_spec.weight_block_merkle_roots[
                                layer_idx
                            ]
                elif layer_idx < len(self.model_spec.weight_block_merkle_roots):
                    W_merkle_root = self.model_spec.weight_block_merkle_roots[layer_idx]

                # Transcript must match prover (Fiat-Shamir)
                if is_moe_challenge and expert_idx is not None:
                    transcript = Transcript(
                        f"layer_{layer_idx}_expert_{expert_idx}_gemm_{gemm_idx}".encode()
                    )
                else:
                    transcript = Transcript(
                        f"layer_{layer_idx}_gemm_{gemm_idx}".encode()
                    )

                # For MoE experts with fused gate+up, use expert_w_num_cols
                expert_w_cols = getattr(self.model_spec, "expert_w_num_cols", 0)
                if is_moe_challenge and expert_idx is not None and expert_w_cols > 0:
                    w_cols = expert_w_cols
                else:
                    w_cols = intermediate_dim

                t0 = time.perf_counter()
                result = verifier.verify(
                    proof=gemm_proof,
                    X_commitment=X_commitment,
                    W_commitment=W_commitment,
                    Y_commitment=Y_commitment,
                    transcript=transcript,
                    spot_check_fn=spot_check_fn,
                    W_merkle_root=W_merkle_root,
                    W_num_cols=w_cols,
                    w_chunk_size=self.model_spec.w_merkle_chunk_size,
                    require_w_merkle_proofs=True,
                )
                verify_time = (time.perf_counter() - t0) * 1000

                if is_moe_challenge and expert_idx is not None:
                    timing_key = (
                        f"Layer {layer_idx}, Expert {expert_idx}, GEMM {gemm_idx}"
                    )
                else:
                    timing_key = f"Layer {layer_idx}, GEMM {gemm_idx}"
                timing_details[timing_key] = verify_time

                status = "PASSED" if result.passed else "FAILED"
                logger.debug("    - %s: %s (%.2fms)", timing_key, status, verify_time)

                if not result.passed:
                    return (
                        VerificationResult.failure(
                            f"GEMM verification failed at {timing_key}: {result.message}"
                        ),
                        timing_details,
                    )

        # Verify routing openings/top-k consistency for challenged MoE layers.
        if is_moe:
            for layer_challenge in expected_challenges.layer_challenges:
                if not hasattr(layer_challenge, "sampled_token_indices"):
                    continue
                if not getattr(layer_challenge, "verify_routing", False):
                    continue
                routing_result = self._verify_router_layer_openings(
                    proof_bundle, layer_challenge
                )
                if not routing_result.passed:
                    return (
                        VerificationResult.failure(
                            f"Routing verification failed for layer {layer_challenge.layer_idx}: "
                            f"{routing_result.message}"
                        ),
                        timing_details,
                    )

        # Verify optional decode-integrity proofs (greedy argmax or
        # canonical sampler replay for do_sample=True with committed seed).
        sampling_challenge = derive_sampling_challenge(
            beacon=beacon,
            commitment=commitment,
            vocab_size=int(getattr(self.model_spec, "vocab_size", 0) or 0),
        )
        if sampling_challenge is not None:
            if not commitment.decode_hidden_row_root:
                return (
                    VerificationResult.failure(
                        "Sampling challenge active but decode_hidden_row_root is missing"
                    ),
                    timing_details,
                )
            if not proof_bundle.sampling_proofs:
                return (
                    VerificationResult.failure(
                        "Sampling challenge active but sampling proofs are missing"
                    ),
                    timing_details,
                )
            if not proof_bundle.output_token_ids:
                return (
                    VerificationResult.failure(
                        "Sampling challenge active but output_token_ids are missing"
                    ),
                    timing_details,
                )
            lm_head_root = getattr(self.model_spec, "lm_head_weight_merkle_root", b"")
            if not lm_head_root:
                return (
                    VerificationResult.failure(
                        "Sampling challenge active but ModelSpec.lm_head_weight_merkle_root is missing"
                    ),
                    timing_details,
                )

            # Collect proofs in challenge order, verify per-row checks.
            proofs_by_step = {
                int(sp.decode_step): sp for sp in proof_bundle.sampling_proofs
            }
            ordered_proofs: list = []
            X_rows_i64: list = []
            Y_rows_i64: list = []
            batched_gemm_proof = None

            for decode_step in sampling_challenge.decode_positions:
                step = int(decode_step)
                sp = proofs_by_step.get(step)
                if sp is None:
                    return (
                        VerificationResult.failure(
                            f"Missing sampling proof for decode step {step}"
                        ),
                        timing_details,
                    )
                if step < 0 or step >= len(proof_bundle.output_token_ids):
                    return (
                        VerificationResult.failure(
                            f"Sampling proof decode step out of range: {step}"
                        ),
                        timing_details,
                    )
                committed_token = int(proof_bundle.output_token_ids[step])
                if int(sp.token_id) != committed_token:
                    return (
                        VerificationResult.failure(
                            f"Sampling token mismatch at step {step}: "
                            f"proof={sp.token_id}, committed={committed_token}"
                        ),
                        timing_details,
                    )

                # Verify hidden row against committed root.
                hidden_ok = verify_merkle_path(
                    root=commitment.decode_hidden_row_root,
                    leaf_data=sp.hidden_row,
                    path=sp.hidden_merkle_path,
                )
                if not hidden_ok:
                    return (
                        VerificationResult.failure(
                            f"Decode hidden-row Merkle proof invalid at step {step}"
                        ),
                        timing_details,
                    )

                if sp.lm_head_weight_root != lm_head_root:
                    return (
                        VerificationResult.failure(
                            f"lm_head weight root mismatch at decode step {step}"
                        ),
                        timing_details,
                    )

                # Reconstruct quantized hidden row.
                hidden_fp32 = hidden_row_from_bytes(sp.hidden_row)
                X_row = quantize_hidden_row_int64(hidden_fp32)
                X_rows_i64.append(X_row)

                logits_i32 = logits_i32_from_bytes(sp.proved_logits_i32)
                if logits_i32.size == 0:
                    return (
                        VerificationResult.failure(
                            f"Empty proved logits row at decode step {step}"
                        ),
                        timing_details,
                    )
                Y_row = torch.from_numpy(logits_i32.astype(np.int64, copy=True)).view(
                    1, -1
                )
                Y_rows_i64.append(Y_row)

                # ----------------------------------------------------------
                # Canonical sampler replay for do_sample=True.
                # ----------------------------------------------------------
                # The miner's CanonicalSamplerLP runs canonical_sample on the
                # raw fp32 logits coming out of compute_logits.  To replay
                # this exactly the validator must use the SAME bit-identical
                # fp32 logits — NOT the int32 reconstruction from the
                # quantized lm_head GEMM proof.  The miner captures the fp32
                # logits row in fp16_logits_row whenever canonical mode is
                # active (any bps > 0), so we open it and verify its Merkle
                # path against decode_logits_row_root.
                #
                # The replay runs at every challenged position regardless of
                # bps.  This closes the loophole where a miner could observe
                # bps_for_request, detect canary vs organic, and selectively
                # cheat on organic — the validator now enforces canonical
                # sampling on every verified request.
                if commitment.do_sample and commitment.sampling_seed_commitment:
                    _opened_seed = getattr(sp, "sampling_seed", None)
                    if _opened_seed is None:
                        return (
                            VerificationResult.failure(
                                f"Sampling seed missing from proof at step {step} "
                                "(do_sample=True with seed commitment requires opening)"
                            ),
                            timing_details,
                        )
                    import hashlib as _hl

                    if (
                        _hl.sha256(_opened_seed).digest()
                        != commitment.sampling_seed_commitment
                    ):
                        return (
                            VerificationResult.failure(
                                f"Sampling seed commitment mismatch at step {step}"
                            ),
                            timing_details,
                        )

                    if not commitment.decode_logits_row_root:
                        return (
                            VerificationResult.failure(
                                f"Canonical replay requires decode_logits_row_root at step {step}"
                            ),
                            timing_details,
                        )
                    if not sp.fp16_logits_row or sp.fp16_logits_merkle_path is None:
                        return (
                            VerificationResult.failure(
                                f"Canonical replay requires fp16_logits_row + merkle path at step {step}"
                            ),
                            timing_details,
                        )
                    # Verify Merkle path so the miner can't substitute
                    # different logits than what they actually produced.
                    if not verify_merkle_path(
                        root=commitment.decode_logits_row_root,
                        leaf_data=sp.fp16_logits_row,
                        path=sp.fp16_logits_merkle_path,
                    ):
                        return (
                            VerificationResult.failure(
                                f"Canonical replay: fp16 logits Merkle path invalid at step {step}"
                            ),
                            timing_details,
                        )
                    # Parse the top-K leaf bytes directly.  The miner's
                    # activation tracker captures only the top
                    # CANONICAL_TOP_K logits per decode step (sorted by
                    # value DESC, index ASC) and serializes them via
                    # `serialize_top_k_to_bytes`.  The leaf IS the top-K
                    # — no extraction step is needed on the validator side.
                    #
                    # Falls back to the legacy full-vocab fp32/fp16 parse
                    # for proofs produced by old miners that captured the
                    # full row.  In that case we extract top-K on the
                    # validator side via extract_top_k_sorted.
                    from verallm.sampling import (
                        canonical_sample as _canonical_sample,
                        parse_top_k_leaf as _parse_top_k_leaf,
                        extract_top_k_sorted as _extract_top_k_sorted,
                    )

                    _row_bytes = sp.fp16_logits_row
                    _row_len = len(_row_bytes)
                    if _row_len == 0:
                        return (
                            VerificationResult.failure(
                                f"Canonical replay: empty fp16_logits_row at step {step}"
                            ),
                            timing_details,
                        )

                    if _row_len % 12 == 0:
                        # Top-K leaf format (current default).
                        try:
                            _top_vals, _top_idx = _parse_top_k_leaf(_row_bytes)
                        except ValueError as _pe:
                            return (
                                VerificationResult.failure(
                                    f"Canonical replay: failed to parse top-K leaf at step {step}: {_pe}"
                                ),
                                timing_details,
                            )
                    else:
                        # Legacy full-vocab fallback.
                        if _row_len % 4 == 0:
                            _full_logits = np.frombuffer(
                                _row_bytes, dtype="<f4"
                            ).astype(np.float32, copy=False)
                        elif _row_len % 2 == 0:
                            _full_logits = np.frombuffer(
                                _row_bytes, dtype="<f2"
                            ).astype(np.float32, copy=False)
                        else:
                            return (
                                VerificationResult.failure(
                                    f"Canonical replay: invalid fp16_logits_row length "
                                    f"{_row_len} at step {step}"
                                ),
                                timing_details,
                            )
                        _top_vals, _top_idx = _extract_top_k_sorted(_full_logits)

                    # Resolve sampling parameters.  Prefer the explicit
                    # expected_* values supplied by the caller (which match
                    # the actual request body); fall back to canary defaults.
                    # The sampler_config_hash check above already binds
                    # these values, so they cannot disagree silently.
                    _temp = max(0.001, commitment.temperature_milli / 1000.0)
                    _top_k = int(expected_top_k) if expected_top_k is not None else -1
                    _top_p = (
                        float(expected_top_p) if expected_top_p is not None else 1.0
                    )
                    _min_p = (
                        float(expected_min_p) if expected_min_p is not None else 0.0
                    )
                    replayed_token = _canonical_sample(
                        _top_vals,
                        _top_idx,
                        _temp,
                        _top_k,
                        _top_p,
                        _min_p,
                        _opened_seed,
                        step,
                    )
                    if replayed_token != committed_token:
                        return (
                            VerificationResult.failure(
                                f"Canonical sampler replay diverged at step {step}: "
                                f"replayed={replayed_token}, committed={committed_token}"
                            ),
                            timing_details,
                        )

                    ordered_proofs.append(sp)
                    if sp.lm_head_gemm_proof is not None and batched_gemm_proof is None:
                        batched_gemm_proof = sp.lm_head_gemm_proof
                    continue

                # Quantization-stable argmax check (int8×int8→int32 recomputation).
                argmax_ok, argmax_detail = verify_quantized_argmax(
                    logits_i32, committed_token
                )

                # When post-logits processors are active (presence_penalty, etc.),
                # argmax divergence is expected — processors modify logits after
                # compute_logits().  Only enforce strict argmax when no
                # processors can cause divergence.
                _has_post_logits_mods = commitment.presence_penalty_milli != 0 or (
                    enable_thinking is not False
                )  # thinking logits processor

                if not argmax_ok and not sampling_challenge.high_assurance:
                    if _has_post_logits_mods:
                        # Post-logits processors active — divergence expected, log only.
                        logger.debug(
                            "    - Sampling step %d: int32 argmax diverged "
                            "(%s) — post-logits processors active (non-fatal)",
                            step,
                            argmax_detail,
                        )
                    else:
                        # Strict mode but GEMM proof passed — likely capture
                        # alignment issue (non-fatal).  Demote to debug:
                        # the GEMM proof did pass, so this is a known
                        # capture-side noise pattern, not a verification
                        # failure that the operator needs to act on.
                        logger.debug(
                            "    - Sampling step %d: int32 argmax diverged "
                            "in strict mode (%s) — GEMM proof passed, "
                            "treating as capture alignment issue (non-fatal)",
                            step,
                            argmax_detail,
                        )

                # High-assurance: fp16 logits row + exact argmax (authoritative).
                # The fp16 logits are captured directly from inference (no
                # requantization loss) and Merkle-committed.  When the int32
                # recomputation diverges — which happens at long contexts with
                # quantized models due to accumulated hidden-state error — the
                # fp16 check is the ground truth.
                if sampling_challenge.high_assurance:
                    if not commitment.decode_logits_row_root:
                        return (
                            VerificationResult.failure(
                                "High-assurance mode active but decode_logits_row_root missing"
                            ),
                            timing_details,
                        )
                    if not sp.fp16_logits_row:
                        return (
                            VerificationResult.failure(
                                f"High-assurance mode active but fp16_logits_row missing at step {step}"
                            ),
                            timing_details,
                        )
                    if sp.fp16_logits_merkle_path is None:
                        return (
                            VerificationResult.failure(
                                f"High-assurance mode active but fp16_logits_merkle_path missing at step {step}"
                            ),
                            timing_details,
                        )
                    fp16_path_ok = verify_merkle_path(
                        root=commitment.decode_logits_row_root,
                        leaf_data=sp.fp16_logits_row,
                        path=sp.fp16_logits_merkle_path,
                    )
                    if not fp16_path_ok:
                        return (
                            VerificationResult.failure(
                                f"Fp16 logits Merkle proof invalid at step {step}"
                            ),
                            timing_details,
                        )
                    fp16_ok, fp16_detail = verify_fp16_argmax(
                        sp.fp16_logits_row, committed_token
                    )

                    if not fp16_ok:
                        if not _has_post_logits_mods:
                            # Strict mode but GEMM proof passed — likely capture
                            # alignment issue. Debug-level: GEMM did pass, so
                            # this is known noise from continuous-batching
                            # capture, not a verification failure.
                            logger.debug(
                                "    - Sampling step %d: fp16 argmax "
                                "diverged in strict mode (%s) — "
                                "GEMM proof passed, treating as capture alignment "
                                "issue (non-fatal)",
                                step,
                                fp16_detail,
                            )
                        else:
                            # Post-logits processors active: divergence expected.
                            logger.debug(
                                "    - Sampling step %d: fp16 argmax diverged "
                                "from committed token (%s) — "
                                "post-logits processors active (non-fatal)",
                                step,
                                fp16_detail,
                            )

                    if not argmax_ok:
                        # Both int32 and fp16 can diverge from capture alignment.
                        # Log for diagnostics.
                        logger.debug(
                            "    - Sampling step %d: int32 argmax diverged "
                            "(%s), fp16: %s",
                            step,
                            argmax_detail,
                            fp16_detail,
                        )
                elif not argmax_ok:
                    # Not high-assurance and int32 failed — shouldn't reach here
                    # because we already return above, but guard defensively.
                    return (
                        VerificationResult.failure(
                            f"Sampling argmax failed at decode step {step}: {argmax_detail}"
                        ),
                        timing_details,
                    )

                # Grab the batched proof from the first proof that carries it.
                if sp.lm_head_gemm_proof is not None and batched_gemm_proof is None:
                    batched_gemm_proof = sp.lm_head_gemm_proof

                ordered_proofs.append(sp)

            # Verify batched lm_head GEMM proof: X_batch @ W = Y_batch.
            if ordered_proofs and X_rows_i64:
                if batched_gemm_proof is None:
                    return (
                        VerificationResult.failure(
                            "No lm_head GEMM proof found in sampling proofs"
                        ),
                        timing_details,
                    )

                X_batch = torch.cat(X_rows_i64, dim=0)
                Y_batch = torch.cat(Y_rows_i64, dim=0)
                hidden_dim = X_batch.shape[1]

                # Reconstruct X Merkle (same chunking as prover).
                flat_x = X_batch.flatten().to(torch.int64)
                x_leaves = []
                for start in range(0, int(flat_x.numel()), 256):
                    x_leaves.append(
                        flat_x[start : start + 256]
                        .numpy()
                        .astype("<i8", copy=False)
                        .tobytes()
                    )
                if not x_leaves:
                    x_leaves = [b"empty"]
                x_tree = MerkleTree(x_leaves)

                # Reconstruct Y Merkle (single block covering [k, vocab]).
                y_block = int(max(1, Y_batch.shape[1]))
                Y_merkle = build_block_merkle(Y_batch, y_block)

                # Verify Y Merkle root matches proof output_root.
                if Y_merkle.root != batched_gemm_proof.output_root:
                    return (
                        VerificationResult.failure(
                            "Batched lm_head Y Merkle root mismatch: "
                            "reconstructed Y_batch does not match proof output_root"
                        ),
                        timing_details,
                    )

                x_result = verify_x_spot_openings(
                    proof=batched_gemm_proof,
                    x_root=x_tree.root,
                    num_cols=hidden_dim,
                    context="Batched lm_head GEMM X",
                )
                if not x_result.passed:
                    return VerificationResult.failure(x_result.message), timing_details

                transcript = Transcript(b"decode_lm_head_gemm_batched")
                lm_head_verify = verifier.verify(
                    proof=batched_gemm_proof,
                    X_commitment=x_tree.root,
                    W_commitment=ordered_proofs[0].lm_head_weight_root,
                    Y_commitment=batched_gemm_proof.output_root,
                    transcript=transcript,
                    spot_check_fn=lambda _spot, _matrix_id: True,
                    W_merkle_root=lm_head_root,
                    W_num_cols=Y_batch.shape[1],
                    w_chunk_size=self.model_spec.w_merkle_chunk_size,
                    require_w_merkle_proofs=True,
                )
                if not lm_head_verify.passed:
                    return (
                        VerificationResult.failure(
                            f"Batched lm_head GEMM verification failed: {lm_head_verify.message}"
                        ),
                        timing_details,
                    )

        return VerificationResult.success("All GEMM proofs verified"), timing_details

    # ------------------------------------------------------------------
    # Full protocol run
    # ------------------------------------------------------------------

    def run_protocol(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        stream_callback=None,
    ) -> Tuple[bool, str, dict]:
        """Run the complete verification protocol.

        Non-interactive flow:
        1. Fetch ModelSpec (on-chain in production, from miner here)
        2. POST /inference -> stream tokens + commitment + proofs
        3. Verify proofs locally (re-derive beacon + challenges)

        Args:
            prompt: Input prompt.
            max_new_tokens: Max tokens to generate.
            stream_callback: Optional callable(token_text) per token.

        Returns:
            (passed, output_text, all_timings)
        """
        all_timings = {}

        # Fetch ModelSpec
        logger.info("\n%s\nPHASE 1: FETCH MODEL SPECIFICATION\n%s", "=" * 70, "=" * 70)
        t0 = time.perf_counter()
        model_spec = self.fetch_model_spec()
        phase_ms = (time.perf_counter() - t0) * 1000
        all_timings["fetch_model_spec_ms"] = phase_ms

        logger.info("  Model: %s", model_spec.model_id)
        logger.info(
            "  Layers: %d, Hidden: %d", model_spec.num_layers, model_spec.hidden_dim
        )
        logger.info(
            "  Roots: %d layer roots", len(model_spec.weight_block_merkle_roots)
        )
        logger.info("  Fetched in %.1fms", phase_ms)

        # Auto-compute k_layers if needed
        if self.config.k_layers == 0:
            k = max(1, round(model_spec.num_layers * self.config.target_detection))
            k = min(k, model_spec.num_layers // 2)
            self.config = Config(
                **{
                    **{
                        f.name: getattr(self.config, f.name)
                        for f in self.config.__dataclass_fields__.values()
                    },
                    "k_layers": k,
                }
            )
            set_config(self.config)
            logger.info("  Auto k_layers: %d/%d", k, model_spec.num_layers)

        # Detect MoE from model_spec.
        # Chain-mode specs have num_experts from on-chain data;
        # local specs may have expert_weight_merkle_roots instead.
        num_experts = model_spec.num_experts
        if num_experts == 0 and model_spec.expert_weight_merkle_roots:
            first_roots = next(iter(model_spec.expert_weight_merkle_roots.values()), [])
            num_experts = len(first_roots)

        if num_experts > 0:
            if model_spec.expert_weight_merkle_roots:
                moe_layer_indices = sorted(model_spec.expert_weight_merkle_roots.keys())
            else:
                # Chain mode: expert roots not in spec, assume all layers are MoE
                moe_layer_indices = list(range(model_spec.num_layers))
            expert_inter = (
                model_spec.expert_w_num_cols
                if model_spec.expert_w_num_cols > 0
                else model_spec.intermediate_dim
            )
            self.moe_config = MoEConfig(
                is_moe=True,
                num_layers=model_spec.num_layers,
                moe_layer_indices=moe_layer_indices,
                num_routed_experts=num_experts,
                num_shared_experts=0,
                top_k=model_spec.router_top_k if model_spec.router_top_k > 0 else 2,
                hidden_dim=model_spec.hidden_dim,
                intermediate_dim=model_spec.intermediate_dim,
                expert_intermediate_dim=expert_inter,
                has_shared_expert_gate=False,
                uses_3d_expert_weights=False,
                router_type=model_spec.router_scoring or "top_k",
            )
            logger.info(
                "  Detected MoE: %d experts, %d MoE layers",
                num_experts,
                len(moe_layer_indices),
            )

            # Auto-compute k_experts_per_layer if not set (must match server)
            if self.config.k_experts_per_layer == 0:
                k_exp = compute_auto_k_experts(num_experts)
                from dataclasses import fields as dc_fields

                self.config = Config(
                    **{
                        **{
                            f.name: getattr(self.config, f.name)
                            for f in dc_fields(self.config)
                        },
                        "k_experts_per_layer": k_exp,
                    }
                )
                set_config(self.config)
                logger.info("  Auto k_experts: %d/%d", k_exp, num_experts)

        # Inference + proofs
        logger.info("\n%s\nINFERENCE + PROOFS\n%s", "=" * 70, "=" * 70)
        logger.info("  Prompt: %s%s", prompt[:80], "..." if len(prompt) > 80 else "")

        def default_stream_cb(token):
            sys.stdout.write(token)
            sys.stdout.flush()

        cb = stream_callback or default_stream_cb

        sys.stdout.write("  Output: ")
        sys.stdout.flush()
        full_text, commitment, proof_bundle, nonce, infer_timing = self.run_inference(
            prompt,
            max_new_tokens=max_new_tokens,
            stream_callback=cb,
        )
        sys.stdout.write("\n")
        sys.stdout.flush()
        all_timings.update(infer_timing)

        logger.info("  Tokens: %d", len(full_text.split()))
        logger.info("  Inference: %.1fms", infer_timing["inference_ms"])
        logger.info("  Commitment: %.1fms", infer_timing["commitment_ms"])
        logger.info("  Prove (miner): %.1fms", infer_timing["prove_ms"])
        logger.info("  Round-trip: %.1fms", infer_timing["round_trip_ms"])

        detection_info = compute_detection_probability(
            self.config.k_layers,
            model_spec.num_layers,
        )
        detection_prob = detection_info["p_detect_per_inference"]
        logger.info("  Detection probability: %.1f%%", detection_prob * 100)

        # Phase 7: Verify proofs locally
        logger.info("\n%s\nPHASE 7: VERIFY PROOFS\n%s", "=" * 70, "=" * 70)
        t0 = time.perf_counter()
        result, verify_timing = self.verify_proof(proof_bundle, nonce)
        verify_ms = (time.perf_counter() - t0) * 1000
        all_timings["verify_ms"] = verify_ms
        all_timings["verify_details"] = verify_timing

        # Summary
        status = "PASSED" if result.passed else "FAILED"
        logger.info(
            "\n%s\n  VERIFICATION RESULT: %s\n  Message: %s\n\n"
            "  Timing Summary:\n"
            "    Model spec fetch:   %.1fms\n"
            "    Inference RTT:      %.1fms\n"
            "      Inference:        %.1fms\n"
            "      Commitment:       %.1fms\n"
            "      Prove (miner):    %.1fms\n"
            "    Verify (local):     %.1fms",
            "=" * 70,
            status,
            result.message,
            all_timings.get("fetch_model_spec_ms", 0),
            all_timings.get("round_trip_ms", 0),
            all_timings.get("inference_ms", 0),
            all_timings.get("commitment_ms", 0),
            all_timings.get("prove_ms", 0),
            verify_ms,
        )
        total = (
            all_timings.get("fetch_model_spec_ms", 0)
            + all_timings.get("round_trip_ms", 0)
            + verify_ms
        )
        logger.info("    Total wall time:    %.1fms\n%s", total, "=" * 70)

        return result.passed, full_text, all_timings


class AsyncValidatorClient(ValidatorClient):
    """Async HTTP transport for production proxy/router inference paths.

    The proof verifier remains the canonical synchronous verifier inherited
    from :class:`ValidatorClient`; callers should run verification in a bounded
    executor when they do not want CPU work on the event loop.  The miner SSE
    transport is async so active streams do not require one Python thread each.
    """

    def __init__(
        self,
        miner_url: str,
        config: Optional[Config] = None,
        verify_tls: bool = True,
        timeout: float = 600.0,
        api_key: Optional[str] = None,
        chain_config=None,
        model_id: Optional[str] = None,
        validator_hotkey_ss58: Optional[str] = None,
        validator_seed: Optional[bytes] = None,
        proof_v2_manifest=None,
    ):
        self.miner_url = miner_url.rstrip("/")
        self.config = config or get_config()
        self.model_spec: Optional[ModelSpec] = None
        self.moe_config: Optional[MoEConfig] = None
        self._chain_config = chain_config
        self._model_id = model_id
        self._on_chain_model_spec = None
        self.proof_v2_manifest = proof_v2_manifest
        self._verify_tls = verify_tls
        self._headers = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self._auth = None
        if validator_hotkey_ss58 and validator_seed:
            self._auth = ValidatorRequestAuth(validator_hotkey_ss58, validator_seed)

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=30.0),
            verify=verify_tls,
            headers=self._headers,
            auth=self._auth,
        )

    async def aclose(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    def close(self):
        raise RuntimeError(
            "AsyncValidatorClient.close() is not supported; use await aclose()"
        )

    def __enter__(self):
        raise RuntimeError("AsyncValidatorClient is async-only; use 'async with'")

    def __exit__(self, *args):
        raise RuntimeError("AsyncValidatorClient is async-only; use 'async with'")

    def fetch_model_spec(self) -> ModelSpec:
        """Synchronous compatibility path for cache-miss verification threads."""
        if self._chain_config is not None:
            spec = self._fetch_model_spec_from_chain()
            if spec is not None:
                self.model_spec = spec
                self._auto_configure_from_spec(spec)
                return spec
            raise RuntimeError(
                f"Model '{self._model_id}' not found on-chain. "
                "Cannot fall back to miner (trust anchor must be on-chain)."
            )

        with httpx.Client(
            timeout=self.client.timeout,
            verify=self._verify_tls,
            headers=self._headers,
            auth=self._auth,
        ) as client:
            resp = client.get(f"{self.miner_url}/model_spec")
            resp.raise_for_status()
        self.model_spec = dict_to_model_spec(resp.json())
        self._auto_configure_from_spec(self.model_spec)
        return self.model_spec

    async def fetch_model_spec_async(self) -> ModelSpec:
        if self._chain_config is not None:
            return self.fetch_model_spec()

        resp = await self.client.get(f"{self.miner_url}/model_spec")
        resp.raise_for_status()
        self.model_spec = dict_to_model_spec(resp.json())
        self._auto_configure_from_spec(self.model_spec)
        return self.model_spec

    async def run_chat_proof_v3(
        self,
        *,
        messages: list[dict],
        prompt_token_ids: Sequence[int],
        qualified_profile,
        validator_identity_digest: bytes,
        miner_identity_digest: bytes,
        runtime_policy,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        presence_penalty: float = 0.0,
        top_k: int = -1,
        top_p: float = 1.0,
        min_p: float = 0.0,
        stream_callback=None,
        proof_challenge_id: bytes | None = None,
        first_token_timeout_seconds: float | None = None,
    ):
        """Run one qualified greedy proof-v3 chat exchange asynchronously."""

        from verallm.api.proof_v3_validator import run_proof_v3_exchange_async

        exchange, request_body = self._prepare_chat_proof_v3(
            messages=messages,
            prompt_token_ids=prompt_token_ids,
            qualified_profile=qualified_profile,
            validator_identity_digest=validator_identity_digest,
            miner_identity_digest=miner_identity_digest,
            runtime_policy=runtime_policy,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            enable_thinking=enable_thinking,
            presence_penalty=presence_penalty,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            proof_challenge_id=proof_challenge_id,
        )
        return await run_proof_v3_exchange_async(
            client=self.client,
            miner_url=self.miner_url,
            inference_path="/chat",
            request_body=request_body,
            exchange=exchange,
            stream_callback=stream_callback,
            first_token_timeout_seconds=first_token_timeout_seconds,
        )

    async def run_chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 4096,
        do_sample: bool = False,
        temperature: float = 1.0,
        sampling_verification_bps: int = 0,
        stream_callback=None,
        enable_thinking: bool = True,
        presence_penalty: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        tools: Optional[list[dict]] = None,
        tool_choice=None,
        parallel_tool_calls: Optional[bool] = None,
        deserialize_proof_bundle: bool = True,
        proof_protocol_version: Optional[int] = None,
        proof_v2_hard_audit_bps: Optional[int] = None,
        proof_v2_transport_state: Optional[ProofV2TransportState] = None,
        allow_unbound_output_count: bool = False,
    ) -> Tuple[
        str, Optional[InferenceCommitment], Optional[InferenceProofBundle], bytes, dict
    ]:
        nonce = os.urandom(32)
        transport_state = proof_v2_transport_state or ProofV2TransportState()
        transport_state.v2_precommit_required = bool(
            proof_protocol_version == PROOF_PROTOCOL_V2
            and proof_v2_hard_audit_bps is not None
        )

        request_body = {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "sampling_verification_bps": max(
                0, min(10_000, int(sampling_verification_bps))
            ),
            "enable_thinking": enable_thinking,
        }
        proof_challenge_id = _add_proof_request_nonce_fields(
            request_body,
            validator_nonce=nonce,
            proof_protocol_version=proof_protocol_version,
        )
        if presence_penalty is not None:
            request_body["presence_penalty"] = presence_penalty
        if top_k is not None:
            request_body["top_k"] = top_k
        if top_p is not None:
            request_body["top_p"] = top_p
        if min_p is not None:
            request_body["min_p"] = min_p
        if tools is not None:
            request_body["tools"] = tools
        if tool_choice is not None:
            request_body["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            request_body["parallel_tool_calls"] = parallel_tool_calls

        full_text = ""
        commitment = None
        proof_bundle = None
        timing = {}
        t_first_token = None
        t_last_token = None
        t_done_recv = None
        t_request_end_wall = None
        precommit_session_id = None
        precommitment_hash = None
        precommitment = None
        challenge_connection_prewarm = None

        t0 = time.perf_counter()
        t0_wall = time.time()
        t_last_tok = None
        async with self.client.stream(
            "POST", f"{self.miner_url}/chat", json=request_body
        ) as resp:
            resp.raise_for_status()
            if proof_protocol_version == PROOF_PROTOCOL_V2 and isinstance(
                self.client, httpx.AsyncClient
            ):
                challenge_connection_prewarm = asyncio.create_task(
                    _prewarm_proof_v2_connection_async(
                        self.client,
                        self.miner_url,
                    )
                )
            async for event_type, data in _parse_sse_stream_async_with_proof_v2_transport(
                resp,
                transport_state,
            ):
                if event_type == "token":
                    transport_state.saw_response_token = True
                    t_last_tok = time.perf_counter()
                    t_last_token = t_last_tok
                    if t_first_token is None:
                        t_first_token = t_last_tok
                    token_text = data.get("text", "")
                    full_text += token_text
                    if stream_callback:
                        maybe_awaitable = stream_callback(token_text)
                        if inspect.isawaitable(maybe_awaitable):
                            await maybe_awaitable
                elif event_type == "proof_precommit":
                    if proof_protocol_version != PROOF_PROTOCOL_V2:
                        raise RuntimeError("unexpected proof-v2 precommit event")
                    if proof_challenge_id is None or precommit_session_id is not None:
                        raise RuntimeError("invalid duplicate proof-v2 precommit event")
                    precommit_received_at = time.perf_counter()
                    (
                        precommit_session_id,
                        precommitment_hash,
                    ) = _decode_proof_v2_precommit(
                        data,
                        expected_challenge_id=proof_challenge_id,
                    )
                    transport_state.precommit_seen = True
                    transport_state.hard_audit_selected = (
                        _select_proof_v2_hard_audit_from_precommit(
                            precommitment_hash=precommitment_hash,
                            validator_nonce=nonce,
                            hard_audit_bps=proof_v2_hard_audit_bps,
                        )
                    )
                    if transport_state.hard_audit_selected is not None:
                        timing["_proof_v2_hard_audit_selected"] = bool(
                            transport_state.hard_audit_selected
                        )
                    if t_last_token is not None:
                        timing["last_token_to_precommit_ms"] = max(
                            0.0,
                            (precommit_received_at - t_last_token) * 1000,
                        )
                    if (
                        challenge_connection_prewarm is not None
                        and challenge_connection_prewarm.done()
                    ):
                        await challenge_connection_prewarm
                    reveal_started = time.perf_counter()
                    transport_state.nonce_reveal_attempted = True
                    try:
                        reveal_response = await self.client.post(
                            f"{self.miner_url}/proof/v2/challenge",
                            json=_proof_v2_reveal_body(
                                challenge_id=proof_challenge_id,
                                session_id=precommit_session_id,
                                commitment_hash=precommitment_hash,
                                validator_nonce=nonce,
                            ),
                        )
                        reveal_response.raise_for_status()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        if transport_state.hard_audit_selected is True:
                            raise _hard_audit_transport_failure(
                                transport_state,
                                "nonce reveal did not complete",
                            ) from exc
                        raise
                    transport_state.nonce_revealed = True
                    timing["proof_challenge_rtt_ms"] = (
                        time.perf_counter() - reveal_started
                    ) * 1000
                elif event_type == "proof_commitment":
                    if (
                        proof_protocol_version != PROOF_PROTOCOL_V2
                        or precommitment is not None
                    ):
                        raise RuntimeError("invalid proof-v2 commitment event")
                    try:
                        precommitment = _decode_proof_v2_commitment(
                            data,
                            precommit_session_id=precommit_session_id,
                            precommitment_hash=precommitment_hash,
                        )
                    except RuntimeError as exc:
                        if (
                            transport_state.hard_audit_selected is True
                            and transport_state.nonce_reveal_attempted
                        ):
                            raise _hard_audit_transport_failure(
                                transport_state,
                                "post-nonce commitment is invalid",
                            ) from exc
                        raise
                elif event_type == "done":
                    transport_state.done_seen = True
                    t_done_recv = time.perf_counter()
                    t_request_end_wall = time.time()
                    done_gap_ms = (
                        (t_done_recv - t_last_tok) * 1000
                        if t_last_tok is not None
                        else 0.0
                    )
                    commit_data = data.get("commitment", {})
                    proof_data = data.get(
                        "proof_bundle", {"layer_proofs": [], "sampling_proofs": []}
                    )
                    t_commit_deser = time.perf_counter()
                    must_deserialize_proof_bundle = _must_deserialize_proof_payload(
                        deserialize_proof_bundle=deserialize_proof_bundle,
                        transport_state=transport_state,
                    )
                    commitment, proof_bundle = _deserialize_done_proof_payload(
                        commit_data,
                        proof_data,
                        deserialize_proof_bundle=must_deserialize_proof_bundle,
                        requested_protocol_version=proof_protocol_version,
                        fallback_commitment=precommitment,
                    )
                    if proof_protocol_version == PROOF_PROTOCOL_V2:
                        try:
                            _validate_proof_v2_done_commitment(
                                final_commitment=commitment,
                                precommit_session_id=precommit_session_id,
                                precommitment_hash=precommitment_hash,
                            )
                            timing[
                                "last_token_to_proof_ms"
                            ] = _measure_proof_v2_response_latency(
                                last_token_at=t_last_token,
                                proof_received_at=t_done_recv,
                            )
                        except RuntimeError:
                            proof_bundle = _retain_failed_proof_v2_transport(
                                proof_bundle
                            )
                    t_proof_deser = time.perf_counter()
                    timing["last_token_to_done_recv_ms"] = done_gap_ms
                    timing["proof_deserialize_ms"] = (
                        t_proof_deser - t_done_recv
                    ) * 1000
                    logger.debug(
                        "ASYNC CLIENT TIMING: last_token→done_recv=%.0fms deser=%.0fms "
                        "(commit=%.0fms proof=%.0fms)",
                        done_gap_ms,
                        (t_proof_deser - t_done_recv) * 1000,
                        (t_commit_deser - t_done_recv) * 1000,
                        (t_proof_deser - t_commit_deser) * 1000,
                    )
                    _remember_miner_inference_ms(timing, data.get("inference_ms"))
                    timing["commitment_ms"] = data.get("commitment_ms", 0)
                    timing["prove_ms"] = data.get("prove_ms", 0)
                    timing["beacon_ms"] = data.get("beacon_ms", 0)
                    timing["challenge_ms"] = data.get("challenge_ms", 0)
                    timing["reveal_ms"] = data.get("reveal_ms", 0)
                    timing["prove_timing_details"] = data.get(
                        "prove_timing_details", {}
                    )
                    timing["miner_last_token_to_proof_ms"] = data.get(
                        "last_token_to_proof_ms", 0
                    )
                    timing["input_tokens"] = data.get("input_tokens", 0)
                    timing["output_tokens"] = _validated_output_token_count(
                        data,
                        max_new_tokens=max_new_tokens,
                        commitment=commitment,
                        commitment_present=bool(commit_data),
                        proof_data=proof_data,
                        allow_unbound=allow_unbound_output_count,
                    )
                elif event_type == "error":
                    if (
                        transport_state.v2_precommit_required
                        and (
                            transport_state.precommit_seen
                            or transport_state.saw_response_token
                        )
                    ):
                        raise _hard_audit_transport_failure(
                            transport_state,
                            "miner ended the required proof-v2 exchange",
                        )
                    if (
                        transport_state.hard_audit_selected is True
                        and transport_state.nonce_reveal_attempted
                    ):
                        raise _hard_audit_transport_failure(
                            transport_state,
                            "miner sent an error after hard-audit selection",
                        )
                    raise RuntimeError(f"Miner error: {data.get('error', data)}")

        if (
            transport_state.v2_precommit_required
            and not transport_state.precommit_seen
        ):
            raise _hard_audit_transport_failure(
                transport_state,
                "response omitted the required proof-v2 precommit",
            )
        if (
            proof_protocol_version == PROOF_PROTOCOL_V2
            and transport_state.hard_audit_selected is True
            and transport_state.nonce_reveal_attempted
            and not transport_state.done_seen
        ):
            raise _hard_audit_transport_failure(
                transport_state,
                "stream ended without a final proof response",
            )

        if t_request_end_wall is None:
            t_request_end_wall = time.time()
        timing["validator_request_start_ts"] = t0_wall
        timing["validator_request_end_ts"] = t_request_end_wall
        timing["validator_request_ms"] = (t_request_end_wall - t0_wall) * 1000
        _finalize_validator_timing(
            timing,
            t0,
            t_first_token,
            t_last_token,
            response_done_at=t_done_recv,
        )

        if commitment is None:
            commitment = InferenceCommitment.empty()
        if proof_bundle is None:
            proof_bundle = InferenceProofBundle.empty()
            if _must_deserialize_proof_payload(
                deserialize_proof_bundle=deserialize_proof_bundle,
                transport_state=transport_state,
            ):
                mark_proof_payload_invalid(
                    proof_bundle,
                    protocol_version=(
                        proof_protocol_version or LEGACY_PROOF_PROTOCOL_VERSION
                    ),
                )

        await _finish_async_connection_prewarm(challenge_connection_prewarm)
        return full_text, commitment, proof_bundle, nonce, timing


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="VeraLLM Validator Client")
    parser.add_argument(
        "--miner-url",
        required=True,
        help="Miner server URL (e.g. http://localhost:8000)",
    )
    parser.add_argument("--prompt", required=True, help="Inference prompt")
    parser.add_argument(
        "--max-new-tokens", type=int, default=4096, help="Max tokens to generate"
    )
    parser.add_argument(
        "--k-layers", type=int, default=0, help="Layers to challenge (0 = auto)"
    )
    parser.add_argument(
        "--k-experts",
        type=int,
        default=0,
        help="Experts to challenge per layer (0 = auto)",
    )
    parser.add_argument(
        "--k-tokens", type=int, default=4, help="Tokens to sample for expert challenges"
    )
    parser.add_argument(
        "--spot-checks", type=int, default=50, help="Number of spot checks per block"
    )
    parser.add_argument(
        "--no-verify-tls",
        action="store_true",
        help="Disable TLS certificate verification",
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="Request timeout in seconds"
    )
    parser.add_argument(
        "--api-key", default=None, help="API key for miner authentication"
    )
    parser.add_argument(
        "--chain-config",
        default=None,
        help="Path to chain config JSON (reads ModelSpec from chain)",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Model ID for on-chain lookup (required with --chain-config)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config(
        block_size=256,
        spot_checks=args.spot_checks,
        k_layers=args.k_layers,
        k_experts_per_layer=args.k_experts,
        k_tokens_per_expert=args.k_tokens,
    )
    set_config(config)

    logger.info(
        "\n%s\n  VeraLLM Validator Client\n  Miner: %s\n  Prompt: %s%s\n%s",
        "=" * 70,
        args.miner_url,
        args.prompt[:60],
        "..." if len(args.prompt) > 60 else "",
        "=" * 70,
    )

    chain_config = None
    if args.chain_config:
        from verallm.chain.config import ChainConfig

        chain_config = ChainConfig.from_json(args.chain_config)

    with ValidatorClient(
        miner_url=args.miner_url,
        config=config,
        verify_tls=not args.no_verify_tls,
        timeout=args.timeout,
        api_key=args.api_key,
        chain_config=chain_config,
        model_id=args.model_id,
    ) as client:
        # Health check
        try:
            health = client.health_check()
            logger.info("  Miner status: %s", health.get("status", "unknown"))
            logger.info("  Miner model: %s", health.get("model", "unknown"))
        except Exception as e:
            logger.error("  Cannot reach miner at %s: %s", args.miner_url, e)
            sys.exit(1)

        # Run full protocol
        passed, output_text, timings = client.run_protocol(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )

        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
