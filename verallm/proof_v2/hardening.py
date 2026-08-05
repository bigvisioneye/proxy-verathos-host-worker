"""Response-bound LM-head audits built on the proof-v2 PCS engine."""

from __future__ import annotations

import hashlib
import struct
from typing import Mapping, Sequence

import numpy as np

from verallm.challenge.v2 import (
    BlockChallengeV2,
    MAX_BLOCKS_PER_OPERATION,
    RegisteredOperationV2,
)
from verallm.proof_v2.layout import (
    column_segments,
    operation_weight_scale_q32_v2,
    padded_inner_dimension,
)
from verallm.proof_v2.manifest import (
    MAX_RUNTIME_ABS_TOLERANCE_Q32,
    MAX_RUNTIME_REL_TOLERANCE_BPS,
    OperationDescriptor,
)
from verallm.sampling import parse_top_k_leaf
from zkllm.crypto.pcs_v2 import MAX_COMBINE_TERMS


LM_HEAD_AUDIT_BLOCKS = 4
MIN_X_NONZERO_BPS = 5_000
MIN_X_POSITIVE_BPS = 500
MIN_X_NEGATIVE_BPS = 500
_AUDIT_DOMAIN = b"VERATHOS/PROOF_V2/LM_HEAD_AUDIT/SHA256"


class ProofV2HardeningError(ValueError):
    """The hardened decode audit is malformed or outside protocol limits."""


def fp16_matches_authenticated_i8_v2(
    actual: np.ndarray,
    quantized: np.ndarray,
    *,
    scale_q32: int,
) -> bool:
    """Check FP16 runtime values against an authenticated global-int8 row.

    The only ambiguity admitted is the signed quantizer's half step plus the
    exact binary16 and Q32 rounding envelope.  This avoids an arbitrary
    model-wide tolerance and does not grow with the magnitude of a coordinate.
    """

    actual_f16 = np.asarray(actual, dtype="<f2")
    quantized_i8 = np.asarray(quantized)
    if (
        actual_f16.shape != quantized_i8.shape
        or actual_f16.size == 0
        or quantized_i8.dtype != np.int8
        or isinstance(scale_q32, bool)
        or not isinstance(scale_q32, int)
        or not 0 < scale_q32 < 1 << 64
        or not np.isfinite(actual_f16).all()
    ):
        return False
    scale = np.float32(scale_q32 / float(1 << 32))
    expected_f16 = np.asarray(
        quantized_i8.astype(np.float32) * scale,
        dtype="<f2",
    )
    difference = np.abs(
        actual_f16.astype(np.float32) - expected_f16.astype(np.float32)
    )
    fp16_rounding = np.abs(np.spacing(expected_f16)).astype(np.float32)
    q32_rounding = np.float32(127.0 / float(1 << 33))
    allowed = np.float32(scale * np.float32(0.5)) + fp16_rounding + q32_rounding
    return bool(np.all(difference <= allowed))


def quantize_canonical_fp16_rows_v2(
    rows: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Quantize canonical FP16 trace rows with one deterministic Q32 scale.

    Both prover and verifier call this routine.  Converting to binary16 before
    deriving the scale makes the arithmetic witness identical to the bytes in
    the pre-challenge execution trace, while explicit float32 operations avoid
    backend-dependent Torch/Python promotion differences.
    """

    canonical = np.asarray(rows, dtype="<f2")
    if canonical.ndim != 2 or not canonical.shape[0] or not canonical.shape[1]:
        raise ProofV2HardeningError("canonical FP16 rows must be a nonempty matrix")
    values = canonical.astype(np.float32)
    if not np.isfinite(values).all():
        raise ProofV2HardeningError("canonical FP16 rows contain a non-finite value")
    absmax = np.max(np.abs(values), axis=1).astype(np.float32, copy=False)
    absmax = np.maximum(absmax, np.float32(1e-8))
    scales = (absmax / np.float32(127.0)).astype(np.float32, copy=False)
    quantized = np.rint(values / scales[:, None]).clip(-128, 127).astype(np.int8)
    scales_q32 = tuple(
        max(1, int(round(float(scale) * float(1 << 32))))
        for scale in scales
    )
    return np.ascontiguousarray(quantized), scales_q32


def quantized_x_profile_is_nondegenerate_v2(x_matrix: np.ndarray) -> bool:
    """Reject zero/sparse shortcut traces while retaining broad model tolerance.

    The check is intentionally model-agnostic and conservative. Because each
    selected proof opens the complete quantized input row, it prevents the
    all-zero relation from satisfying every registered operation at negligible
    cost. It is economic trace hardening, not a proof of nonlinear transitions.
    """

    matrix = np.asarray(x_matrix)
    if (
        matrix.ndim != 2
        or matrix.shape[0] <= 0
        or matrix.shape[1] <= 0
        or not np.issubdtype(matrix.dtype, np.signedinteger)
    ):
        return False
    width = int(matrix.shape[1])
    minimum_nonzero = max(1, (width * MIN_X_NONZERO_BPS + 9_999) // 10_000)
    minimum_positive = (
        max(1, (width * MIN_X_POSITIVE_BPS + 9_999) // 10_000)
        if width >= 64
        else 0
    )
    minimum_negative = (
        max(1, (width * MIN_X_NEGATIVE_BPS + 9_999) // 10_000)
        if width >= 64
        else 0
    )
    minimum_distinct = min(16, max(2, width // 8))
    for row in matrix:
        if (
            int(np.count_nonzero(row)) < minimum_nonzero
            or int(np.count_nonzero(row > 0)) < minimum_positive
            or int(np.count_nonzero(row < 0)) < minimum_negative
            or int(np.unique(row).size) < minimum_distinct
        ):
            return False
    return True


def select_lm_head_audit_decode_step_v2(
    *,
    transcript_state: bytes,
    commitment_hash: bytes,
    decode_positions: Sequence[int],
    minimum_decode_step: int = 0,
) -> int:
    """Choose one challenged decode position uniformly for the PCS audit."""

    if not isinstance(transcript_state, bytes) or len(transcript_state) != 32:
        raise ProofV2HardeningError("LM-head transcript state must be 32 bytes")
    if not isinstance(commitment_hash, bytes) or len(commitment_hash) != 32:
        raise ProofV2HardeningError("LM-head commitment hash must be 32 bytes")
    positions = tuple(decode_positions)
    if (
        not positions
        or any(
            isinstance(position, bool)
            or not isinstance(position, int)
            or position < 0
            for position in positions
        )
        or tuple(sorted(set(positions))) != positions
    ):
        raise ProofV2HardeningError("LM-head decode position set is not canonical")
    if (
        isinstance(minimum_decode_step, bool)
        or not isinstance(minimum_decode_step, int)
        or minimum_decode_step < 0
    ):
        raise ProofV2HardeningError("LM-head minimum decode step is invalid")
    eligible = tuple(
        position for position in positions if position >= minimum_decode_step
    )
    if not eligible:
        raise ProofV2HardeningError(
            "LM-head audit has no eligible decode position"
        )
    digest = hashlib.sha256(
        _AUDIT_DOMAIN
        + b"/POSITION"
        + transcript_state
        + commitment_hash
        + struct.pack("<I", minimum_decode_step)
        + b"".join(struct.pack("<I", position) for position in positions)
    ).digest()
    return eligible[int.from_bytes(digest, "little") % len(eligible)]


def quantize_committed_hidden_row_v2(
    hidden_row: bytes,
    *,
    hidden_dim: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Quantize an authenticated fp16 hidden row deterministically."""

    if (
        not isinstance(hidden_row, bytes)
        or isinstance(hidden_dim, bool)
        or not isinstance(hidden_dim, int)
        or hidden_dim <= 0
        or len(hidden_row) != hidden_dim * 2
    ):
        raise ProofV2HardeningError("committed hidden row dimensions are invalid")
    return quantize_canonical_fp16_rows_v2(
        np.frombuffer(hidden_row, dtype="<f2").reshape(1, hidden_dim)
    )


def _draw_index(seed: bytes, counter: int, bound: int) -> tuple[int, int]:
    if bound <= 0:
        raise ProofV2HardeningError("audit sampling bound must be positive")
    limit = (1 << 256) - ((1 << 256) % bound)
    while True:
        digest = hashlib.sha256(
            _AUDIT_DOMAIN + seed + struct.pack("<Q", counter)
        ).digest()
        counter += 1
        candidate = int.from_bytes(digest, "little")
        if candidate < limit:
            return candidate % bound, counter


def derive_lm_head_audit_challenges_v2(
    *,
    operation: RegisteredOperationV2,
    transcript_state: bytes,
    commitment_hash: bytes,
    decode_step: int,
    token_id: int,
    top_k_row: bytes,
    block_count: int = LM_HEAD_AUDIT_BLOCKS,
) -> tuple[BlockChallengeV2, ...]:
    """Select mandatory and response-random vocabulary blocks after commitment."""

    if not isinstance(operation, RegisteredOperationV2):
        raise ProofV2HardeningError("LM-head operation has an unexpected type")
    if not isinstance(transcript_state, bytes) or len(transcript_state) != 32:
        raise ProofV2HardeningError("LM-head transcript state must be 32 bytes")
    if not isinstance(commitment_hash, bytes) or len(commitment_hash) != 32:
        raise ProofV2HardeningError("LM-head commitment hash must be 32 bytes")
    if (
        isinstance(decode_step, bool)
        or not isinstance(decode_step, int)
        or decode_step < 0
    ):
        raise ProofV2HardeningError("LM-head decode step is invalid")
    if (
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or not 0 <= token_id < operation.output_dim
    ):
        raise ProofV2HardeningError("LM-head token is outside the vocabulary")
    if (
        isinstance(block_count, bool)
        or not isinstance(block_count, int)
        or not 1 <= block_count <= MAX_BLOCKS_PER_OPERATION
    ):
        raise ProofV2HardeningError("LM-head audit block count is invalid")
    try:
        top_values, top_indices = parse_top_k_leaf(top_k_row)
    except ValueError as exc:
        raise ProofV2HardeningError("LM-head top-k row is malformed") from exc
    if (
        top_values.size == 0
        or top_indices.size != top_values.size
        or not np.isfinite(top_values).all()
        or (top_indices < 0).any()
        or (top_indices >= operation.output_dim).any()
        or np.unique(top_indices).size != top_indices.size
    ):
        raise ProofV2HardeningError("LM-head top-k row contents are invalid")

    segments = column_segments(operation)
    segment_by_token = {}
    mandatory_tokens = {token_id, int(top_indices[0])}
    for block_col, (offset, columns) in enumerate(segments):
        for mandatory_token in mandatory_tokens:
            if offset <= mandatory_token < offset + columns:
                segment_by_token[mandatory_token] = block_col
    if len(segment_by_token) != len(mandatory_tokens):
        raise ProofV2HardeningError("LM-head mandatory token block is missing")

    selected = set(segment_by_token.values())
    term_count = sum(1 + segments[index][1] for index in selected)
    if term_count > MAX_COMBINE_TERMS:
        raise ProofV2HardeningError("LM-head mandatory blocks exceed the PCS term budget")

    seed = hashlib.sha256(
        _AUDIT_DOMAIN
        + transcript_state
        + commitment_hash
        + operation.canonical_bytes()
        + struct.pack("<II", decode_step, token_id)
        + hashlib.sha256(top_k_row).digest()
    ).digest()
    top_k_blocks = {
        int(token) // operation.block_cols
        for token in top_indices
    }
    top_k_remaining = sorted(top_k_blocks - selected)
    outside_remaining = [
        index
        for index in range(len(segments))
        if index not in selected and index not in top_k_blocks
    ]
    available = {
        index for index in range(len(segments)) if index not in selected
    }
    counter = 0
    target = min(block_count, len(segments))
    pool_order = (top_k_remaining, outside_remaining)
    pool_index = 0
    while len(selected) < target and available:
        pool = pool_order[pool_index % len(pool_order)]
        pool_index += 1
        pool[:] = [index for index in pool if index in available]
        if not pool:
            pool = sorted(available)
        position, counter = _draw_index(seed, counter, len(pool))
        block_col = pool[position]
        added_terms = 1 + segments[block_col][1]
        if term_count + added_terms > MAX_COMBINE_TERMS:
            available.remove(block_col)
            continue
        selected.add(block_col)
        available.remove(block_col)
        term_count += added_terms

    padded_inner = padded_inner_dimension(operation.inner_dim)
    challenges = []
    for block_col in sorted(selected):
        column_offset, columns = segments[block_col]
        challenges.append(
            BlockChallengeV2(
                key=operation.key,
                block_row=0,
                block_col=block_col,
                row_offset=0,
                column_offset=column_offset,
                rows=1,
                inner_dim=operation.inner_dim,
                padded_inner_dim=padded_inner,
                cols=columns,
                row_rounds=0,
                inner_rounds=padded_inner.bit_length() - 1,
                col_rounds=columns.bit_length() - 1,
            )
        )
    if not challenges:
        raise ProofV2HardeningError("LM-head audit selected no blocks")
    return tuple(challenges)


def build_lm_head_runtime_y_v2(
    *,
    operation: RegisteredOperationV2,
    challenges: Sequence[BlockChallengeV2],
    x_matrix: np.ndarray,
    weight_matrix: np.ndarray | None,
    x_scale_q32: int,
    descriptor: OperationDescriptor,
    weight_blocks: Mapping[tuple[int, int], np.ndarray] | None = None,
) -> np.ndarray:
    """Build the selected proof-domain logits in fp16 engine form."""

    x = np.asarray(x_matrix, dtype=np.int8)
    weight = None if weight_matrix is None else np.asarray(weight_matrix, dtype=np.int8)
    if x.shape != (1, operation.inner_dim):
        raise ProofV2HardeningError("LM-head X dimensions are invalid")
    if weight_blocks is None:
        if weight is None or weight.shape != (
            operation.inner_dim,
            operation.output_dim,
        ):
            raise ProofV2HardeningError("LM-head W dimensions are invalid")
    else:
        expected_blocks = {
            (challenge.column_offset, challenge.cols) for challenge in challenges
        }
        if set(weight_blocks) != expected_blocks:
            raise ProofV2HardeningError("LM-head selected W block set is not exact")
        for (_column_offset, columns), block in weight_blocks.items():
            if (
                not isinstance(block, np.ndarray)
                or block.dtype != np.dtype(np.int8)
                or block.shape != (operation.inner_dim, columns)
                or not block.flags.c_contiguous
            ):
                raise ProofV2HardeningError(
                    "LM-head selected W block is not canonical"
                )
    if x_scale_q32 <= 0:
        raise ProofV2HardeningError("LM-head quantization scales must be positive")
    runtime_y = np.zeros((1, operation.output_dim), dtype="<f2")
    for challenge in challenges:
        if challenge.key != operation.key or challenge.rows != 1:
            raise ProofV2HardeningError("LM-head challenge set is not exact")
        start = challenge.column_offset
        stop = start + challenge.cols
        weight_scale_q32 = operation_weight_scale_q32_v2(descriptor, start)
        weight_block = (
            weight[:, start:stop]
            if weight_blocks is None
            else weight_blocks[(start, challenge.cols)]
        )
        y_values = (
            x.astype(np.int64)
            @ weight_block.astype(np.int64)
        ).reshape(-1)
        dequantized = np.empty(challenge.cols, dtype=np.float64)
        for index, value in enumerate(y_values):
            q32 = (
                int(value)
                * int(x_scale_q32)
                * int(weight_scale_q32)
            ) >> 32
            dequantized[index] = q32 / float(1 << 32)
        encoded = dequantized.astype("<f2")
        if not np.isfinite(encoded).all():
            raise ProofV2HardeningError("LM-head selected logits exceed fp16")
        runtime_y[0, start:stop] = encoded
    return runtime_y


def proof_logit_q32_v2(
    proof_y: int,
    *,
    x_scale_q32: int,
    weight_scale_q32: int,
) -> int:
    """Convert one exact int8 GEMM output into signed Q32 logit units."""

    return (
        int(proof_y)
        * int(x_scale_q32)
        * int(weight_scale_q32)
    ) >> 32


def logit_matches_committed_value_v2(
    proof_logit_q32: int,
    committed_logit: float,
    descriptor: OperationDescriptor,
) -> bool:
    """Compare an audited logit with the pre-challenge captured fp32 value."""

    if not np.isfinite(committed_logit):
        return False
    observed_q32 = int(round(float(committed_logit) * float(1 << 32)))
    absolute = min(
        int(descriptor.runtime_abs_tolerance_q32),
        MAX_RUNTIME_ABS_TOLERANCE_Q32,
    )
    relative_bps = min(
        int(descriptor.runtime_rel_tolerance_bps),
        MAX_RUNTIME_REL_TOLERANCE_BPS,
    )
    reference = max(abs(int(proof_logit_q32)), abs(observed_q32))
    relative = (relative_bps * reference + 10_000 - 1) // 10_000
    return abs(int(proof_logit_q32) - observed_q32) <= absolute + relative


def logit_is_not_above_committed_boundary_v2(
    proof_logit_q32: int,
    committed_boundary: float,
    descriptor: OperationDescriptor,
) -> bool:
    """Check an audited non-top-k logit against the committed top-k boundary."""

    if not np.isfinite(committed_boundary):
        return False
    boundary_q32 = int(round(float(committed_boundary) * float(1 << 32)))
    absolute = min(
        int(descriptor.runtime_abs_tolerance_q32),
        MAX_RUNTIME_ABS_TOLERANCE_Q32,
    )
    relative_bps = min(
        int(descriptor.runtime_rel_tolerance_bps),
        MAX_RUNTIME_REL_TOLERANCE_BPS,
    )
    reference = max(abs(int(proof_logit_q32)), abs(boundary_q32))
    relative = (relative_bps * reference + 10_000 - 1) // 10_000
    return int(proof_logit_q32) <= boundary_q32 + absolute + relative


__all__ = [
    "LM_HEAD_AUDIT_BLOCKS",
    "ProofV2HardeningError",
    "build_lm_head_runtime_y_v2",
    "derive_lm_head_audit_challenges_v2",
    "fp16_matches_authenticated_i8_v2",
    "logit_is_not_above_committed_boundary_v2",
    "logit_matches_committed_value_v2",
    "proof_logit_q32_v2",
    "quantize_canonical_fp16_rows_v2",
    "quantized_x_profile_is_nondegenerate_v2",
    "quantize_committed_hidden_row_v2",
    "select_lm_head_audit_decode_step_v2",
]
