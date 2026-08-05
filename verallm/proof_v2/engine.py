"""Sound disclosed-X, batched-W, and independent-Y engine for GEMM proof v2."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from verallm.challenge.v2 import (
    BlockChallengeV2,
    OperationKeyV2,
    ProofBlockDescriptorV2,
    RegisteredOperationV2,
    RuntimeYCommitmentV2,
    XCommitmentV2,
    derive_hard_execution_corridor_v2,
    validate_exact_block_proof_set_v2,
)
from verallm.proof_v2.hardening import (
    quantize_canonical_fp16_rows_v2,
    quantized_x_profile_is_nondegenerate_v2,
)
from verallm.proof_v2.layout import (
    RUNTIME_Y_COMMITMENT_BLOCK_COLS,
    RUNTIME_Y_COMMITMENT_BLOCK_ROWS,
    execution_trace_fields_for_operation,
    layout_digest,
    operation_descriptor_by_key,
    operation_weight_scale_q32_v2,
    registered_all_operations_from_manifest,
    registered_operations_from_manifest,
    row_segments,
    runtime_y_column_segments,
)
from verallm.proof_v2.manifest import (
    MAX_RUNTIME_ABS_TOLERANCE_Q32,
    MAX_RUNTIME_REL_TOLERANCE_BPS,
    StaticWeightCommitmentManifest,
)
from verallm.proof_v2.payload import (
    GemmBlockProofV2,
    ProofV2CommitmentEnvelope,
    ProofV2Payload,
)
from verallm.proof_v2.pcs_batch import (
    derive_same_point_batch_opening_context_v2,
)
from verallm.proof_v2.trace import (
    ExecutionTraceProofV2,
    FullAttentionHeadWitnessV2,
    FullAttentionHeadStateOpeningV2,
    GDN_DECODE_SUFFIX_TOKEN_START_V1,
    GDNTransitionWitnessV2,
    ProofV2TraceError,
    TRACE_ATTENTION_FULL_TRANSITION_V1,
    TRACE_ATTENTION_GDN_TRANSITION_V1,
    build_full_attention_witness_root_v2,
    full_attention_head_state_leaf_v2,
    gdn_state_digest_v2,
    trace_attention_state_boundary_digest_v2,
    validate_layer_trace_set_v2,
)
from verallm.proof_v2.transition import (
    FULL_ATTENTION_TRANSITION_PROFILE_V1,
    FullAttentionTransitionParametersV2,
    GDN_TRANSITION_PROFILE_V1,
    GDNTransitionParametersV2,
    ProofV2TransitionError,
    gdn_state_numpy_dtype_v2,
    derive_transition_challenges_v2,
    replay_qwen_full_attention_head_v2,
    replay_qwen_gdn_block_v2,
)
from verallm.crypto.merkle import verify_merkle_path
from zkllm.types import MerklePath
from zkllm.crypto.gemm_v2_batch import (
    GemmV2BatchProof,
    GemmV2BatchStatement,
    verify_gemm_v2_batch_sumcheck,
)
from zkllm.crypto.gemm_v2_reference import (
    PALLAS_SCALAR_MODULUS,
    GemmV2Statement,
    scalar_from_bytes,
)
from zkllm.crypto.pcs_merkle_v2 import (
    PCS_ENCODING_HASHED_FP16_BLOCK,
    PCS_ENCODING_HASHED_QUANTIZED_I8_ROW,
    PCS_ENCODING_SIGNED_I8,
    PCS_MERKLE_DOMAIN_PRECHALLENGE_RUNTIME_Y,
    PCS_MERKLE_DOMAIN_PRECHALLENGE_X,
    PCS_MERKLE_DOMAIN_STATIC_WEIGHT,
    PcsMerkleV2Context,
    PcsMerkleV2Leaf,
    PcsMerkleV2MultiProof,
    PcsMerkleV2Tree,
    hash_fp16_block_v2,
    hash_fp16_blocks_v2,
    hash_quantized_i8_row_v2,
    verify_pcs_merkle_v2_multiproof,
)

_BLOCK_PARENT_DOMAIN = b"VERATHOS/PROOF_V2/GEMM_BLOCK_PARENT/SHA256"
_SELECTED_COMMITMENTS_DOMAIN = b"VERATHOS/PROOF_V2/SELECTED_COMMITMENTS"
_OPENING_TERM_BINDING_DOMAIN = b"VERATHOS/PROOF_V2/SAME_POINT_OPENING_TERM/SHA256"


class ProofV2EngineError(ValueError):
    """Proof-v2 generation or verification failed."""


def _resolve_operations(
    manifest: StaticWeightCommitmentManifest,
    operations: Sequence[RegisteredOperationV2] | None,
) -> tuple[RegisteredOperationV2, ...]:
    selected = tuple(
        registered_operations_from_manifest(manifest)
        if operations is None
        else operations
    )
    if not selected:
        raise ProofV2EngineError("proof-v2 operation set must not be empty")
    if tuple(sorted(selected, key=lambda item: item.key)) != selected:
        raise ProofV2EngineError("proof-v2 operation set is not canonical")
    if len({operation.key for operation in selected}) != len(selected):
        raise ProofV2EngineError("proof-v2 operation set contains duplicates")
    manifest_operations = {
        operation.key: operation
        for operation in registered_all_operations_from_manifest(manifest)
    }
    if any(
        manifest_operations.get(operation.key) != operation for operation in selected
    ):
        raise ProofV2EngineError(
            "proof-v2 operation set does not match the authenticated manifest"
        )
    return selected


def _ceil_sqrt(value: int) -> int:
    """Return ceil(sqrt(value)) using deterministic integer arithmetic."""

    if value < 0:
        raise ProofV2EngineError("square-root input must be nonnegative")
    root = math.isqrt(value)
    return root if root * root == value else root + 1


def _runtime_y_block_within_tolerance(
    expected_q32: Sequence[int],
    observed_q32: Sequence[int],
    *,
    absolute_tolerance_q32: int,
    relative_tolerance_bps: int,
) -> bool:
    """Apply the authenticated runtime-Y tolerance to one challenged block.

    The absolute component is a per-coordinate Q32 allowance converted to an
    L2 budget.  The relative component is applied to the larger expected or
    observed block norm.  Integer square roots and upward rounding keep the
    result deterministic and conservative across validator platforms.
    """

    expected = tuple(int(value) for value in expected_q32)
    observed = tuple(int(value) for value in observed_q32)
    if not expected or len(expected) != len(observed):
        raise ProofV2EngineError("runtime Y tolerance requires equal nonempty blocks")
    if absolute_tolerance_q32 < 0 or not 0 <= relative_tolerance_bps <= 10_000:
        raise ProofV2EngineError("runtime Y tolerance parameters are invalid")

    error_squared = sum(
        (expected_value - observed_value) ** 2
        for expected_value, observed_value in zip(expected, observed)
    )
    expected_squared = sum(value * value for value in expected)
    observed_squared = sum(value * value for value in observed)
    absolute_budget = _ceil_sqrt(len(expected)) * absolute_tolerance_q32
    reference_norm = max(
        _ceil_sqrt(expected_squared),
        _ceil_sqrt(observed_squared),
    )
    relative_budget = (relative_tolerance_bps * reference_norm + 10_000 - 1) // 10_000
    allowed = absolute_budget + relative_budget
    return error_squared <= allowed * allowed


def _effective_runtime_y_tolerance(
    absolute_tolerance_q32: int,
    relative_tolerance_bps: int,
) -> tuple[int, int]:
    """Clamp signed manifest requests to non-vacuous protocol ceilings."""

    return (
        min(absolute_tolerance_q32, MAX_RUNTIME_ABS_TOLERANCE_Q32),
        min(relative_tolerance_bps, MAX_RUNTIME_REL_TOLERANCE_BPS),
    )


def _execution_operation_row_counts_match_v2(
    commitments: Sequence[XCommitmentV2],
    *,
    token_count: int,
    num_layers: int,
) -> bool:
    """Match causal layer rows while allowing one-row model-level audits."""

    return all(
        item.row_count == token_count
        for item in commitments
        if 0 <= item.key.layer_idx < num_layers
    )


@dataclass(frozen=True)
class MatrixCommitmentStateV2:
    """Miner-side matrix values and their row- or column-commitment tree."""

    operation: RegisteredOperationV2
    matrix: np.ndarray | None
    tree: PcsMerkleV2Tree
    scales_q32: tuple[int, ...] = ()
    # Static W witnesses may be reconstructed after the challenge as only the
    # exact authenticated output blocks that the prover must open.  This keeps
    # canonical checkpoint reconstruction proportional to the proof instead of
    # materializing a second full-model int8 copy on every miner.
    selected_blocks: Mapping[tuple[int, int], np.ndarray] | None = None


@dataclass(frozen=True)
class RuntimeYCommitmentStateV2:
    """Miner-side captured runtime output and its pre-challenge block tree."""

    operation: RegisteredOperationV2
    matrix: np.ndarray
    tree: PcsMerkleV2Tree


@dataclass(frozen=True)
class InferenceXStateV2:
    """Complete pre-challenge quantized-X and runtime-Y trace state."""

    envelope: ProofV2CommitmentEnvelope
    operations: Mapping[OperationKeyV2, MatrixCommitmentStateV2]
    runtime_y_operations: Mapping[OperationKeyV2, RuntimeYCommitmentStateV2]


def combine_commitment_envelopes_v2(
    manifest: StaticWeightCommitmentManifest,
    envelopes: Sequence[ProofV2CommitmentEnvelope],
) -> ProofV2CommitmentEnvelope:
    """Combine disjoint authenticated operation envelopes for one batch proof."""

    if not isinstance(manifest, StaticWeightCommitmentManifest):
        raise ProofV2EngineError("combined envelope manifest is invalid")
    items = tuple(envelopes)
    if not items or any(
        not isinstance(item, ProofV2CommitmentEnvelope) for item in items
    ):
        raise ProofV2EngineError(
            "combined envelope set must contain canonical envelopes"
        )
    manifest_digest = manifest.digest()
    x_commitments = []
    runtime_y_commitments = []
    trace_commitments = []
    transition_commitments = []
    seen = set()
    for envelope in items:
        if envelope.manifest_digest != manifest_digest:
            raise ProofV2EngineError("combined envelope uses another manifest digest")
        keys = {item.key for item in envelope.x_commitments}
        if keys & seen:
            raise ProofV2EngineError(
                "combined envelope operation sets must be disjoint"
            )
        seen.update(keys)
        x_commitments.extend(envelope.x_commitments)
        runtime_y_commitments.extend(envelope.runtime_y_commitments)
        if envelope.execution_trace_commitment is not None:
            trace_commitments.append(envelope.execution_trace_commitment)
        if envelope.transition_history_commitment is not None:
            transition_commitments.append(envelope.transition_history_commitment)
    trace_encodings = {item.canonical_bytes() for item in trace_commitments}
    if len(trace_encodings) > 1:
        raise ProofV2EngineError(
            "combined envelopes use different execution trace commitments"
        )
    transition_encodings = {item.canonical_bytes() for item in transition_commitments}
    if len(transition_encodings) > 1:
        raise ProofV2EngineError(
            "combined envelopes use different transition history commitments"
        )
    return ProofV2CommitmentEnvelope(
        manifest_digest,
        tuple(sorted(x_commitments, key=lambda item: item.key)),
        tuple(sorted(runtime_y_commitments, key=lambda item: item.key)),
        trace_commitments[0] if trace_commitments else None,
        transition_commitments[0] if transition_commitments else None,
    )


def combine_inference_x_states_v2(
    manifest: StaticWeightCommitmentManifest,
    states: Sequence[InferenceXStateV2],
) -> InferenceXStateV2:
    """Combine disjoint prover states while retaining their exact openings."""

    items = tuple(states)
    if not items or any(not isinstance(item, InferenceXStateV2) for item in items):
        raise ProofV2EngineError("combined X state set must contain canonical states")
    envelope = combine_commitment_envelopes_v2(
        manifest,
        tuple(item.envelope for item in items),
    )
    operations = {}
    runtime_y_operations = {}
    for state in items:
        envelope_keys = {item.key for item in state.envelope.x_commitments}
        if (
            set(state.operations) != envelope_keys
            or set(state.runtime_y_operations) != envelope_keys
        ):
            raise ProofV2EngineError(
                "combined X state maps do not match their envelope"
            )
        operations.update(state.operations)
        runtime_y_operations.update(state.runtime_y_operations)
    if set(operations) != {item.key for item in envelope.x_commitments}:
        raise ProofV2EngineError("combined X state operation set is not exact")
    return InferenceXStateV2(
        envelope,
        operations,
        runtime_y_operations,
    )


@dataclass(frozen=True)
class _PreparedProverBlockV2:
    descriptor: ProofBlockDescriptorV2
    operation: RegisteredOperationV2
    x_membership_proof: bytes
    w_membership_proof: bytes
    runtime_y_membership_proof: bytes
    x_row_digests: tuple[bytes, ...]
    w_commitments: tuple[bytes, ...]
    x_block: np.ndarray
    w_block: np.ndarray
    y_block: np.ndarray
    x_rows: bytes
    x_scales_q32: bytes
    runtime_y_values: bytes
    proof_y_values: bytes
    proof_y_commitment: bytes
    statement: GemmV2Statement


def _int8_matrix(
    value: object,
    *,
    expected_rows: int | None,
    expected_columns: int,
    name: str,
) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    array = np.asarray(value)
    if array.ndim != 2:
        raise ProofV2EngineError(f"{name} must be a two-dimensional matrix")
    if expected_rows is not None and array.shape[0] != expected_rows:
        raise ProofV2EngineError(f"{name} row count does not match the operation")
    if array.shape[0] <= 0 or array.shape[1] != expected_columns:
        raise ProofV2EngineError(f"{name} dimensions do not match the operation")
    if array.dtype == np.bool_:
        raise ProofV2EngineError(f"{name} must contain signed integer values")
    if not np.issubdtype(array.dtype, np.integer):
        raise ProofV2EngineError(f"{name} must contain signed integer values")
    if int(array.min()) < -128 or int(array.max()) > 127:
        raise ProofV2EngineError(f"{name} contains a value outside signed int8")
    return np.ascontiguousarray(array, dtype=np.int8)


def _runtime_fp16_matrix(
    value: object,
    *,
    expected_rows: int,
    expected_columns: int,
) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().float().numpy()
    except ImportError:
        pass
    array = np.asarray(value)
    if array.ndim != 2 or array.shape != (expected_rows, expected_columns):
        raise ProofV2EngineError("runtime Y dimensions do not match the operation")
    if not np.issubdtype(array.dtype, np.floating):
        raise ProofV2EngineError("runtime Y must contain floating-point values")
    if not np.isfinite(array).all():
        raise ProofV2EngineError("runtime Y contains a non-finite value")
    encoded = np.ascontiguousarray(array, dtype="<f2")
    if not np.isfinite(encoded).all():
        raise ProofV2EngineError("runtime Y is outside the finite fp16 range")
    return encoded


def _tree_context(
    operation: RegisteredOperationV2,
    *,
    domain: bytes,
    leaf_count: int,
) -> PcsMerkleV2Context:
    return PcsMerkleV2Context(
        tree_domain=domain,
        operation_identity_digest=layout_digest(operation),
        logical_leaf_count=leaf_count,
    )


def _commit_i8_vectors(vectors: Sequence[np.ndarray]) -> tuple[bytes, ...]:
    from zkllm.crypto.pcs_v2 import commit

    return tuple(
        commit(np.ascontiguousarray(vector, dtype=np.int8).tobytes(), encoding=1)
        for vector in vectors
    )


def _hash_quantized_i8_rows(
    vectors: Sequence[np.ndarray],
    scales_q32: Sequence[int],
) -> tuple[bytes, ...]:
    if len(vectors) != len(scales_q32):
        raise ProofV2EngineError("X row scales do not match the row set")
    return tuple(
        hash_quantized_i8_row_v2(
            np.ascontiguousarray(vector, dtype=np.int8).tobytes(),
            int(vector.size),
            int(scale_q32),
        )
        for vector, scale_q32 in zip(vectors, scales_q32)
    )


def build_weight_commitment_state_v2(
    operation: RegisteredOperationV2,
    weight_matrix: object,
    *,
    expected_root: bytes | None = None,
) -> MatrixCommitmentStateV2:
    """Build the one-time per-column W commitment tree for an operation."""

    matrix = _int8_matrix(
        weight_matrix,
        expected_rows=operation.inner_dim,
        expected_columns=operation.output_dim,
        name="W",
    )
    commitments = _commit_i8_vectors(
        tuple(matrix[:, index] for index in range(matrix.shape[1]))
    )
    tree = build_weight_commitment_tree_v2(
        operation,
        commitments,
        expected_root=expected_root,
    )
    return MatrixCommitmentStateV2(operation, matrix, tree)


def build_weight_commitment_root_v2(
    operation: RegisteredOperationV2,
    weight_matrix: object,
) -> bytes:
    """Compute a manifest root before the descriptor is signed."""

    return build_weight_commitment_state_v2(
        operation,
        weight_matrix,
        expected_root=b"",
    ).tree.root


def load_weight_commitment_state_v2(
    operation: RegisteredOperationV2,
    weight_matrix: object,
    column_commitments: Sequence[bytes],
) -> MatrixCommitmentStateV2:
    """Load a precomputed W catalog and verify it against the manifest root."""

    matrix = _int8_matrix(
        weight_matrix,
        expected_rows=operation.inner_dim,
        expected_columns=operation.output_dim,
        name="W",
    )
    tree = build_weight_commitment_catalog_tree_v2(operation, column_commitments)
    return MatrixCommitmentStateV2(operation, matrix, tree)


def build_weight_commitment_catalog_tree_v2(
    operation: RegisteredOperationV2,
    column_commitments: Sequence[bytes],
) -> PcsMerkleV2Tree:
    """Validate one static catalog operation directly against its manifest root."""

    return build_weight_commitment_tree_v2(operation, column_commitments)


def build_weight_commitment_tree_v2(
    operation: RegisteredOperationV2,
    column_commitments: Sequence[bytes],
    *,
    expected_root: bytes | None = None,
) -> PcsMerkleV2Tree:
    """Build the canonical static W tree, optionally before its root is known."""

    commitments = tuple(column_commitments)
    if len(commitments) != operation.output_dim:
        raise ProofV2EngineError("W commitment catalog column count is incorrect")
    leaves = tuple(
        PcsMerkleV2Leaf(bytes(commitment), operation.inner_dim, PCS_ENCODING_SIGNED_I8)
        for commitment in commitments
    )
    tree = PcsMerkleV2Tree(
        _tree_context(
            operation,
            domain=PCS_MERKLE_DOMAIN_STATIC_WEIGHT,
            leaf_count=operation.output_dim,
        ),
        leaves,
    )
    required_root = (
        operation.weight_commitment_root if expected_root is None else expected_root
    )
    if required_root and tree.root != required_root:
        raise ProofV2EngineError(
            "W commitment catalog does not match the manifest root"
        )
    return tree


def build_inference_x_state_v2(
    manifest: StaticWeightCommitmentManifest,
    x_matrices: Mapping[OperationKeyV2, object],
    runtime_y_matrices: Mapping[OperationKeyV2, object],
    x_scales_q32: Mapping[OperationKeyV2, Sequence[int]],
    *,
    operations: Sequence[RegisteredOperationV2] | None = None,
    execution_trace_commitment=None,
    transition_history_commitment=None,
) -> InferenceXStateV2:
    """Commit every quantized X row and captured runtime Y before selection."""

    operations = _resolve_operations(manifest, operations)
    if set(x_matrices) != {operation.key for operation in operations}:
        raise ProofV2EngineError(
            "X matrix set does not match the manifest operation set"
        )
    expected_keys = {operation.key for operation in operations}
    if set(runtime_y_matrices) != expected_keys:
        raise ProofV2EngineError(
            "runtime Y matrix set does not match the manifest operation set"
        )
    if set(x_scales_q32) != expected_keys:
        raise ProofV2EngineError(
            "X scale set does not match the manifest operation set"
        )
    states = {}
    runtime_y_states = {}
    metadata = []
    runtime_y_metadata = []
    prepared_runtime_y = {}
    runtime_y_hash_groups = {}
    for operation in operations:
        matrix = _int8_matrix(
            x_matrices[operation.key],
            expected_rows=None,
            expected_columns=operation.inner_dim,
            name="X",
        )
        scales = tuple(int(value) for value in x_scales_q32[operation.key])
        if len(scales) != matrix.shape[0] or any(
            value <= 0 or value >= (1 << 64) for value in scales
        ):
            raise ProofV2EngineError("X row scales are not canonical positive Q32")
        commitments = _hash_quantized_i8_rows(
            tuple(matrix[index, :] for index in range(matrix.shape[0])),
            scales,
        )
        leaves = tuple(
            PcsMerkleV2Leaf(
                commitment,
                operation.inner_dim,
                PCS_ENCODING_HASHED_QUANTIZED_I8_ROW,
            )
            for commitment in commitments
        )
        tree = PcsMerkleV2Tree(
            _tree_context(
                operation,
                domain=PCS_MERKLE_DOMAIN_PRECHALLENGE_X,
                leaf_count=matrix.shape[0],
            ),
            leaves,
        )
        states[operation.key] = MatrixCommitmentStateV2(
            operation,
            matrix,
            tree,
            scales,
        )
        metadata.append(
            XCommitmentV2(
                operation.key,
                int(matrix.shape[0]),
                operation.inner_dim,
                tree.root,
            )
        )

        runtime_y = _runtime_fp16_matrix(
            runtime_y_matrices[operation.key],
            expected_rows=matrix.shape[0],
            expected_columns=operation.output_dim,
        )
        y_blocks = []
        column_blocks = runtime_y_column_segments(operation)
        for _block_row, (row_offset, rows) in enumerate(
            row_segments(operation, matrix.shape[0])
        ):
            padded = np.zeros(
                (rows, len(column_blocks) * RUNTIME_Y_COMMITMENT_BLOCK_COLS),
                dtype="<f2",
            )
            padded[:, : operation.output_dim] = runtime_y[
                row_offset : row_offset + rows,
                :,
            ]
            vector_length = int(rows * RUNTIME_Y_COMMITMENT_BLOCK_COLS)
            for block_col in range(len(column_blocks)):
                column_offset = block_col * RUNTIME_Y_COMMITMENT_BLOCK_COLS
                block = padded[
                    :,
                    column_offset : column_offset + RUNTIME_Y_COMMITMENT_BLOCK_COLS,
                ].tobytes()
                block_index = len(y_blocks)
                y_blocks.append((block, vector_length))
                runtime_y_hash_groups.setdefault(vector_length, []).append(
                    (operation.key, block_index, block)
                )
        prepared_runtime_y[operation.key] = (
            operation,
            runtime_y,
            tuple(y_blocks),
        )

    runtime_y_commitments = {
        operation.key: [b""] * len(prepared_runtime_y[operation.key][2])
        for operation in operations
    }
    for vector_length, indexed_blocks in runtime_y_hash_groups.items():
        digests = hash_fp16_blocks_v2(
            b"".join(block for _, _, block in indexed_blocks),
            vector_length,
        )
        for (key, block_index, _), digest in zip(indexed_blocks, digests):
            runtime_y_commitments[key][block_index] = digest

    for operation in operations:
        _, runtime_y, y_blocks = prepared_runtime_y[operation.key]
        y_commitments = runtime_y_commitments[operation.key]
        y_logical_lengths = tuple(vector_length for _, vector_length in y_blocks)
        y_tree = PcsMerkleV2Tree._from_trusted_packed_components(
            _tree_context(
                operation,
                domain=PCS_MERKLE_DOMAIN_PRECHALLENGE_RUNTIME_Y,
                leaf_count=len(y_commitments),
            ),
            b"".join(y_commitments),
            y_logical_lengths,
            bytes([PCS_ENCODING_HASHED_FP16_BLOCK]) * len(y_commitments),
        )
        runtime_y_states[operation.key] = RuntimeYCommitmentStateV2(
            operation,
            runtime_y,
            y_tree,
        )
        runtime_y_metadata.append(
            RuntimeYCommitmentV2(
                operation.key,
                int(runtime_y.shape[0]),
                operation.output_dim,
                RUNTIME_Y_COMMITMENT_BLOCK_ROWS,
                RUNTIME_Y_COMMITMENT_BLOCK_COLS,
                y_tree.root,
            )
        )
    envelope = ProofV2CommitmentEnvelope(
        manifest.digest(),
        tuple(metadata),
        tuple(runtime_y_metadata),
        execution_trace_commitment,
        transition_history_commitment,
    )
    return InferenceXStateV2(envelope, states, runtime_y_states)


def _descriptor(challenge: BlockChallengeV2) -> ProofBlockDescriptorV2:
    return ProofBlockDescriptorV2(
        key=challenge.key,
        block_row=challenge.block_row,
        block_col=challenge.block_col,
        row_offset=challenge.row_offset,
        column_offset=challenge.column_offset,
        rows=challenge.rows,
        inner_dim=challenge.inner_dim,
        padded_inner_dim=challenge.padded_inner_dim,
        cols=challenge.cols,
        row_rounds=challenge.row_rounds,
        inner_rounds=challenge.inner_rounds,
        col_rounds=challenge.col_rounds,
    )


def _block_parent_digest(
    *,
    commitment_hash: bytes,
    beacon: bytes,
    manifest_digest: bytes,
    descriptor: ProofBlockDescriptorV2,
) -> bytes:
    values = (
        bytes(commitment_hash),
        bytes(beacon),
        bytes(manifest_digest),
        descriptor.as_challenge().canonical_bytes(),
    )
    if any(len(value) != 32 for value in values[:3]):
        raise ProofV2EngineError("proof-v2 parent context digests must be 32 bytes")
    encoded = b"".join(struct.pack("<I", len(value)) + value for value in values)
    return hashlib.sha256(_BLOCK_PARENT_DOMAIN + encoded).digest()


def _selected_commitments_bytes(label: bytes, commitments: Sequence[bytes]) -> bytes:
    points = tuple(bytes(commitment) for commitment in commitments)
    if not points or len(points) > 256 or any(len(point) != 32 for point in points):
        raise ProofV2EngineError("selected PCS commitment set is invalid")
    return (
        _SELECTED_COMMITMENTS_DOMAIN
        + struct.pack("<H", len(label))
        + label
        + struct.pack("<H", len(points))
        + b"".join(points)
    )


def _mle_coefficients(point: Sequence[int]) -> tuple[int, ...]:
    point_tuple = tuple(int(value) for value in point)
    modulus = PALLAS_SCALAR_MODULUS
    if any(not 0 <= challenge < modulus for challenge in point_tuple):
        raise ProofV2EngineError("MLE challenge is outside the Pallas field")

    # Expand the equality polynomial in lexicographic bit order.  This is
    # equivalent to the direct definition but needs one modular multiply per
    # coefficient per round instead of re-evaluating every bit for every
    # final coefficient.  The inner MLE has 13 rounds for current 8192-wide
    # operations, so this keeps validator work proportional to its 8192
    # output coefficients rather than 13 * 8192.
    coefficients = [1]
    for challenge in point_tuple:
        zero = (1 - challenge) % modulus
        expanded = []
        for coefficient in coefficients:
            expanded.append(coefficient * zero % modulus)
            expanded.append(coefficient * challenge % modulus)
        coefficients = expanded
    return tuple(coefficients)


def _opening_term_bindings(
    batch_statement: GemmV2BatchStatement,
    components: Sequence[bytes],
) -> tuple[bytes, ...]:
    """Derive ordered claim identities from the already-authenticated batch."""

    bindings = []
    for statement_index, statement in enumerate(batch_statement.statements):
        for component in components:
            if component not in (b"x", b"w", b"y"):
                raise ProofV2EngineError("opening component is not canonical")
            bindings.append(
                hashlib.sha256(
                    _OPENING_TERM_BINDING_DOMAIN
                    + struct.pack("<I", statement_index)
                    + struct.pack("<B", len(component))
                    + component
                    + statement.digest()
                ).digest()
            )
    return tuple(bindings)


def _statement(
    *,
    parent_digest: bytes,
    operation: RegisteredOperationV2,
    descriptor: ProofBlockDescriptorV2,
    x_commitments: Sequence[bytes],
    w_commitments: Sequence[bytes],
    y_commitment: bytes,
) -> GemmV2Statement:
    return GemmV2Statement(
        outer_digest=parent_digest,
        operation_identity=layout_digest(operation),
        row_block=descriptor.block_row,
        column_block=descriptor.block_col,
        rows=descriptor.rows,
        inner=descriptor.padded_inner_dim,
        valid_inner=descriptor.inner_dim,
        columns=descriptor.cols,
        x_commitment=_selected_commitments_bytes(b"x", x_commitments),
        w_commitment=_selected_commitments_bytes(b"w", w_commitments),
        y_commitment=y_commitment,
    )


def prove_inference_v2(
    *,
    manifest: StaticWeightCommitmentManifest,
    commitment_hash: bytes,
    beacon: bytes,
    x_state: InferenceXStateV2,
    weight_states: Mapping[OperationKeyV2, MatrixCommitmentStateV2],
    challenges: Sequence[BlockChallengeV2],
    operations: Sequence[RegisteredOperationV2] | None = None,
    execution_trace_proof: bytes = b"",
) -> ProofV2Payload:
    """Produce one exact, transcript-bound batch for every selected block."""

    manifest_digest = manifest.digest()
    operations = _resolve_operations(manifest, operations)
    operations_by_key = {operation.key: operation for operation in operations}
    if x_state.envelope.manifest_digest != manifest_digest:
        raise ProofV2EngineError("X state manifest digest does not match")
    if set(x_state.operations) != set(operations_by_key):
        raise ProofV2EngineError("X state operation set does not match the manifest")
    if set(x_state.runtime_y_operations) != set(operations_by_key):
        raise ProofV2EngineError(
            "runtime Y state operation set does not match the manifest"
        )
    challenged_keys = {challenge.key for challenge in challenges}
    if set(weight_states) != challenged_keys:
        raise ProofV2EngineError(
            "W state operation set does not exactly match the challenged operations"
        )
    challenge_set = tuple(challenges)
    if not challenge_set:
        raise ProofV2EngineError("proof-v2 challenge set must not be empty")
    challenge_descriptors = tuple(_descriptor(item) for item in challenge_set)
    batch_rows = max(item.rows for item in challenge_descriptors)
    batch_columns = max(item.cols for item in challenge_descriptors)

    from zkllm.crypto.pcs_v2 import (
        commit,
        prove_gemm_batch_sumcheck_i8,
        prove_i8_linear_combination,
    )

    expected_weight_blocks: dict[OperationKeyV2, set[tuple[int, int]]] = {
        key: set() for key in challenged_keys
    }
    for descriptor in challenge_descriptors:
        expected_weight_blocks[descriptor.key].add(
            (descriptor.column_offset, descriptor.cols)
        )
    for key, weight_state in weight_states.items():
        selected = weight_state.selected_blocks
        if selected is None:
            matrix = weight_state.matrix
            operation = operations_by_key[key]
            if (
                not isinstance(matrix, np.ndarray)
                or matrix.dtype != np.dtype(np.int8)
                or matrix.shape != (operation.inner_dim, operation.output_dim)
            ):
                raise ProofV2EngineError("full W witness matrix is not canonical")
            continue
        if weight_state.matrix is not None:
            raise ProofV2EngineError(
                "W state must not mix full and selected-block witnesses"
            )
        if set(selected) != expected_weight_blocks[key]:
            raise ProofV2EngineError(
                "selected W witness block set does not match the challenges"
            )
        operation = operations_by_key[key]
        for (column_offset, columns), matrix in selected.items():
            if (
                not isinstance(matrix, np.ndarray)
                or matrix.dtype != np.dtype(np.int8)
                or matrix.shape != (operation.inner_dim, columns)
                or not matrix.flags.c_contiguous
                or column_offset < 0
                or column_offset + columns > operation.output_dim
            ):
                raise ProofV2EngineError("selected W witness block is not canonical")

    prepared: list[_PreparedProverBlockV2] = []
    for challenge in challenge_set:
        operation = operations_by_key.get(challenge.key)
        if operation is None:
            raise ProofV2EngineError("challenge references an unknown operation")
        x_operation = x_state.operations[challenge.key]
        runtime_y_operation = x_state.runtime_y_operations[challenge.key]
        w_operation = weight_states[challenge.key]
        if w_operation.tree.root != operation.weight_commitment_root:
            raise ProofV2EngineError("W state does not match the manifest root")
        descriptor = _descriptor(challenge)
        x_indices = tuple(
            range(descriptor.row_offset, descriptor.row_offset + descriptor.rows)
        )
        w_indices = tuple(
            range(descriptor.column_offset, descriptor.column_offset + descriptor.cols)
        )
        x_multiproof = x_operation.tree.multiproof(x_indices)
        w_multiproof = w_operation.tree.multiproof(w_indices)
        runtime_y_segments = runtime_y_column_segments(operation)
        runtime_y_block_col = (
            descriptor.column_offset // RUNTIME_Y_COMMITMENT_BLOCK_COLS
        )
        runtime_y_leaf_index = (
            descriptor.block_row * len(runtime_y_segments) + runtime_y_block_col
        )
        runtime_y_multiproof = runtime_y_operation.tree.multiproof(
            (runtime_y_leaf_index,)
        )
        x_row_digests = tuple(
            item.leaf.commitment for item in x_multiproof.opened_leaves
        )
        w_commitments = tuple(
            item.leaf.commitment for item in w_multiproof.opened_leaves
        )
        x_block = np.ascontiguousarray(
            x_operation.matrix[np.asarray(x_indices), :], dtype=np.int8
        )
        if w_operation.selected_blocks is None:
            w_block = np.ascontiguousarray(
                w_operation.matrix[:, np.asarray(w_indices)], dtype=np.int8
            )
        else:
            try:
                w_block = w_operation.selected_blocks[
                    (descriptor.column_offset, descriptor.cols)
                ]
            except KeyError as exc:
                raise ProofV2EngineError("selected W witness block is missing") from exc
        y_block = x_block.astype(np.int64) @ w_block.astype(np.int64)
        proof_y_values = np.ascontiguousarray(y_block, dtype="<i8").tobytes()
        y_padded = np.zeros((batch_rows, batch_columns), dtype="<i8")
        y_padded[: descriptor.rows, : descriptor.cols] = y_block
        y_flat = tuple(int(value) for value in y_padded.reshape(-1))
        y_commitment = commit(y_flat, encoding=2)
        x_scales = tuple(x_operation.scales_q32[index] for index in x_indices)
        x_scales_bytes = struct.pack(f"<{len(x_scales)}Q", *x_scales)

        runtime_y_column_offset, runtime_y_columns = runtime_y_segments[
            runtime_y_block_col
        ]
        if not (
            runtime_y_column_offset <= descriptor.column_offset
            and descriptor.column_offset + descriptor.cols
            <= runtime_y_column_offset + runtime_y_columns
        ):
            raise ProofV2EngineError(
                "challenged runtime Y slice crosses its commitment segment"
            )
        runtime_y_block = np.zeros(
            (descriptor.rows, RUNTIME_Y_COMMITMENT_BLOCK_COLS), dtype="<f2"
        )
        runtime_y_block[:, :runtime_y_columns] = runtime_y_operation.matrix[
            descriptor.row_offset : descriptor.row_offset + descriptor.rows,
            runtime_y_column_offset : runtime_y_column_offset + runtime_y_columns,
        ]
        statement = _statement(
            parent_digest=_block_parent_digest(
                commitment_hash=commitment_hash,
                beacon=beacon,
                manifest_digest=manifest_digest,
                descriptor=descriptor,
            ),
            operation=operation,
            descriptor=descriptor,
            x_commitments=x_row_digests,
            w_commitments=w_commitments,
            y_commitment=y_commitment,
        )
        prepared.append(
            _PreparedProverBlockV2(
                descriptor,
                operation,
                x_multiproof.canonical_bytes(),
                w_multiproof.canonical_bytes(),
                runtime_y_multiproof.canonical_bytes(),
                x_row_digests,
                w_commitments,
                x_block,
                w_block,
                np.ascontiguousarray(y_block, dtype="<i8"),
                x_block.tobytes(),
                x_scales_bytes,
                runtime_y_block.tobytes(),
                proof_y_values,
                y_commitment,
                statement,
            )
        )

    batch_statement = GemmV2BatchStatement(tuple(item.statement for item in prepared))
    rows = batch_statement.rows
    padded_inner = batch_statement.inner
    columns = batch_statement.columns
    x_padded = np.zeros((len(prepared), rows, padded_inner), dtype=np.int8)
    w_padded = np.zeros((len(prepared), padded_inner, columns), dtype=np.int8)
    y_values = np.zeros((len(prepared), rows, columns), dtype="<i8")
    valid_inner = []
    for index, item in enumerate(prepared):
        valid = item.operation.inner_dim
        x_padded[index, : item.descriptor.rows, :valid] = item.x_block
        w_padded[index, :valid, : item.descriptor.cols] = item.w_block
        y_values[index, : item.descriptor.rows, : item.descriptor.cols] = item.y_block
        valid_inner.append(valid)
    sumcheck_bytes = prove_gemm_batch_sumcheck_i8(
        batch_digest=batch_statement.digest(),
        x_values=x_padded.tobytes(),
        w_values=w_padded.tobytes(),
        y_values_i64_le=y_values.tobytes(),
        valid_inner=tuple(valid_inner),
        block_count=len(prepared),
        rows=rows,
        inner=padded_inner,
        columns=columns,
    )
    sumcheck = GemmV2BatchProof.from_canonical_bytes(
        sumcheck_bytes,
        expected_blocks=len(prepared),
        expected_rounds=batch_statement.inner_bits,
    )
    claims = verify_gemm_v2_batch_sumcheck(batch_statement, sumcheck)
    column_coefficients = _mle_coefficients(claims.column_challenges)

    # Every selected X row is already fully disclosed and authenticated by its
    # pre-challenge Merkle commitment. The verifier evaluates those rows
    # directly at the sumcheck point, so the expensive IPA needs to open only
    # the authority-authenticated static W columns.
    base_vectors: list[bytes] = []
    base_coefficient_groups: list[tuple[int, ...]] = []
    for item in prepared:
        w_coefficients = column_coefficients[: item.descriptor.cols]
        w_vectors = []
        for column in range(item.w_block.shape[1]):
            padded = np.zeros(padded_inner, dtype=np.int8)
            padded[: item.operation.inner_dim] = item.w_block[:, column]
            w_vectors.append(padded.tobytes())
        base_vectors.extend(w_vectors)
        base_coefficient_groups.append(tuple(w_coefficients))

    w_context = derive_same_point_batch_opening_context_v2(
        label=b"w",
        statement_digest=claims.batch_statement_digest,
        term_bindings=_opening_term_bindings(batch_statement, (b"w",)),
        evaluations=claims.w_values,
    )
    base_coefficients: list[int] = []
    for block_index, w_coefficients in enumerate(base_coefficient_groups):
        w_batch = w_context.coefficients[block_index]
        base_coefficients.extend(
            w_batch * coefficient % PALLAS_SCALAR_MODULUS
            for coefficient in w_coefficients
        )
    w_opening = prove_i8_linear_combination(
        b"".join(base_vectors),
        vector_length=padded_inner,
        coefficients=tuple(base_coefficients),
        point=claims.inner_challenges,
        outer_digest=w_context.opening_outer_digest,
    )
    if scalar_from_bytes(w_opening.evaluation) != w_context.combined_evaluation:
        raise ProofV2EngineError(
            "batched W opening evaluation does not match the sumcheck claims"
        )

    block_proofs = tuple(
        GemmBlockProofV2(
            descriptor=item.descriptor,
            x_membership_proof=item.x_membership_proof,
            x_rows=item.x_rows,
            x_scales_q32=item.x_scales_q32,
            runtime_y_membership_proof=item.runtime_y_membership_proof,
            runtime_y_values=item.runtime_y_values,
            proof_y_values=item.proof_y_values,
            w_membership_proof=item.w_membership_proof,
            proof_y_commitment=item.proof_y_commitment,
        )
        for item in prepared
    )
    return ProofV2Payload(
        manifest_digest,
        block_proofs,
        sumcheck_bytes,
        w_opening,
        execution_trace_proof,
    )


def _trace_tensor_equal_v2(left, right) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.values == right.values
    )


def _trace_f16_vector_v2(layer, name: str) -> np.ndarray:
    tensor = layer.tensor(name)
    if tensor.dtype != "f16" or len(tensor.shape) != 1:
        raise ProofV2EngineError(
            f"causal trace tensor {name} is not a canonical fp16 vector"
        )
    values = np.frombuffer(tensor.values, dtype="<f2")
    if values.shape != tensor.shape or not np.isfinite(values).all():
        raise ProofV2EngineError(
            f"causal trace tensor {name} is malformed or non-finite"
        )
    return values


def _trace_fp16_matches_v2(actual: np.ndarray, expected: np.ndarray) -> bool:
    if actual.shape != expected.shape:
        return False
    return bool(
        np.allclose(
            actual.astype(np.float32),
            np.asarray(expected, dtype="<f2").astype(np.float32),
            rtol=5e-3,
            atol=2e-3,
        )
    )


def _gemma_rmsnorm_v2(
    values: np.ndarray,
    weight_f16: bytes,
    epsilon_q32: int,
) -> np.ndarray:
    """Replay the signed Qwen/Gemma RMSNorm semantics deterministically."""

    weights = np.frombuffer(weight_f16, dtype="<f2")
    if (
        weights.shape != values.shape
        or not np.isfinite(weights).all()
        or epsilon_q32 <= 0
    ):
        raise ProofV2EngineError("signed RMSNorm parameters are malformed")
    source = values.astype(np.float32)
    variance = np.mean(source * source, dtype=np.float32)
    epsilon = np.float32(epsilon_q32 / float(1 << 32))
    normalized = source / np.sqrt(variance + epsilon, dtype=np.float32)
    return np.asarray(
        normalized * (np.float32(1.0) + weights.astype(np.float32)),
        dtype="<f2",
    )


def _verify_weightless_layer_bridge_v2(layer, parameters) -> None:
    """Replay signed normalization, residual, and SiLU bridge transitions."""

    residual_in = _trace_f16_vector_v2(layer, "residual_in")
    x_attn = _trace_f16_vector_v2(layer, "x_attn")
    if not _trace_fp16_matches_v2(
        x_attn,
        _gemma_rmsnorm_v2(
            residual_in,
            parameters.input_norm_weight_f16,
            parameters.norm_epsilon_q32,
        ),
    ):
        raise ProofV2EngineError("causal trace input RMSNorm transition is invalid")
    attention_out = _trace_f16_vector_v2(layer, "attention_out_proj")
    residual_after_attention = _trace_f16_vector_v2(
        layer,
        "residual_after_attention",
    )
    if not _trace_fp16_matches_v2(
        residual_after_attention,
        residual_in.astype(np.float32) + attention_out.astype(np.float32),
    ):
        raise ProofV2EngineError(
            "causal trace attention residual transition is invalid"
        )

    x_ffn = _trace_f16_vector_v2(layer, "x_ffn")
    if not _trace_fp16_matches_v2(
        x_ffn,
        _gemma_rmsnorm_v2(
            residual_after_attention,
            parameters.post_attention_norm_weight_f16,
            parameters.norm_epsilon_q32,
        ),
    ):
        raise ProofV2EngineError(
            "causal trace post-attention RMSNorm transition is invalid"
        )

    gate_up = _trace_f16_vector_v2(layer, "mlp_gate_up")
    if gate_up.size % 2:
        raise ProofV2EngineError("causal trace fused MLP width is invalid")
    gate, up = np.split(gate_up.astype(np.float32), 2)
    # Stable exact SiLU replay. Clipping the activation itself would change
    # valid large positive values (SiLU(127) is approximately 127, not 80).
    sigmoid = np.empty_like(gate, dtype=np.float32)
    positive = gate >= 0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-gate[positive]))
    exp_gate = np.exp(gate[~positive])
    sigmoid[~positive] = exp_gate / (1.0 + exp_gate)
    expected_hidden = (gate * sigmoid) * up
    mlp_hidden = _trace_f16_vector_v2(layer, "mlp_hidden")
    if not _trace_fp16_matches_v2(mlp_hidden, expected_hidden):
        raise ProofV2EngineError("causal trace SiLU bridge transition is invalid")

    mlp_down = _trace_f16_vector_v2(layer, "mlp_down")
    residual_out = _trace_f16_vector_v2(layer, "residual_out")
    if not _trace_fp16_matches_v2(
        residual_out,
        residual_after_attention.astype(np.float32) + mlp_down.astype(np.float32),
    ):
        raise ProofV2EngineError("causal trace MLP residual transition is invalid")


def _verify_gdn_decode_suffix_transition_v2(
    *,
    witnesses: Sequence[GDNTransitionWitnessV2],
    tokens,
    parameters,
    opening,
) -> None:
    """Replay one selected GDN layer from the committed prompt boundary.

    ``witnesses`` must contain every generated row for the selected layer.
    The opening is the real cache state left by prefill; after that point each
    compact witness is recomputed and must reproduce both its state digest and
    registered out-projection input.
    """

    if parameters.attention_profile != TRACE_ATTENTION_GDN_TRANSITION_V1:
        raise ProofV2EngineError("GDN transition profile does not match the manifest")
    if parameters.transition_profile != GDN_TRANSITION_PROFILE_V1:
        raise ProofV2EngineError("GDN transition parameters are not supported")
    try:
        signed = GDNTransitionParametersV2.from_canonical_bytes(
            parameters.transition_parameters
        )
        replay_parameters = signed.replay_parameters()
    except ProofV2TransitionError as exc:
        raise ProofV2EngineError("GDN transition parameters are malformed") from exc
    if opening.token_start != GDN_DECODE_SUFFIX_TOKEN_START_V1:
        raise ProofV2EngineError("GDN state opening has an unsupported boundary")
    if not witnesses or tuple(item.token_index for item in witnesses) != tuple(
        range(opening.token_start, opening.token_start + len(witnesses))
    ):
        raise ProofV2EngineError("GDN transition trace rows are not exact")
    if any(
        item.layer_idx != parameters.layer for item in witnesses
    ):
        raise ProofV2EngineError("GDN transition trace profile is inconsistent")
    if opening.layer_idx != parameters.layer:
        raise ProofV2EngineError("GDN state opening layer is inconsistent")
    conv_dtype = gdn_state_numpy_dtype_v2(signed.runtime_dtype)
    recurrent_dtype = gdn_state_numpy_dtype_v2(signed.recurrent_state_dtype)
    expected_conv_bytes = (
        (signed.conv_kernel_size - 1)
        * (
            2 * signed.num_key_heads * signed.key_head_dim
            + signed.num_value_heads * signed.value_head_dim
        )
        * conv_dtype.itemsize
    )
    expected_recurrent_bytes = (
        signed.num_value_heads
        * signed.value_head_dim
        * signed.key_head_dim
        * recurrent_dtype.itemsize
    )
    if (
        opening.conv_state_dtype != signed.runtime_dtype
        or opening.recurrent_state_dtype != signed.recurrent_state_dtype
        or len(opening.conv_state) != expected_conv_bytes
        or len(opening.recurrent_state) != expected_recurrent_bytes
    ):
        raise ProofV2EngineError("GDN state opening dimensions are not canonical")
    conv_state = np.frombuffer(opening.conv_state, dtype=conv_dtype).reshape(
        signed.conv_kernel_size - 1,
        2 * signed.num_key_heads * signed.key_head_dim
        + signed.num_value_heads * signed.value_head_dim,
    )
    recurrent_state = np.frombuffer(
        opening.recurrent_state,
        dtype=recurrent_dtype,
    ).reshape(
        signed.num_value_heads,
        signed.value_head_dim,
        signed.key_head_dim,
    )
    if not np.isfinite(conv_state).all() or not np.isfinite(recurrent_state).all():
        raise ProofV2EngineError("GDN state opening contains non-finite values")

    for expected_token_index, witness in enumerate(witnesses):
        token_index = opening.token_start + expected_token_index
        try:
            token = tokens[token_index]
        except (IndexError, TypeError) as exc:
            raise ProofV2EngineError(
                "GDN transition token witness is out of range"
            ) from exc
        if (
            gdn_state_digest_v2(
                "conv",
                np.asarray(conv_state, dtype=conv_dtype).tobytes(),
                dtype=signed.runtime_dtype,
            )
            != witness.conv_before_digest
            or gdn_state_digest_v2(
                "recurrent",
                np.asarray(recurrent_state, dtype=recurrent_dtype).tobytes(),
                dtype=signed.recurrent_state_dtype,
            )
            != witness.recurrent_before_digest
            or trace_attention_state_boundary_digest_v2(
                TRACE_ATTENTION_GDN_TRANSITION_V1,
                (witness.conv_before_digest, witness.recurrent_before_digest),
            )
            != token.attention_state_before_digests[parameters.layer]
        ):
            raise ProofV2EngineError(
                "GDN state opening does not match the committed decode boundary"
            )
        try:
            replay = replay_qwen_gdn_block_v2(
                mixed_qkvz=np.frombuffer(witness.qkvz_f16, dtype="<f2").reshape(
                    1, -1
                ),
                mixed_ba=np.frombuffer(witness.ba_f16, dtype="<f2").reshape(1, -1),
                conv_state_before=conv_state,
                recurrent_state_before=recurrent_state,
                parameters=replay_parameters,
            )
        except ProofV2TransitionError as exc:
            raise ProofV2EngineError("GDN transition replay is invalid") from exc
        captured_output = np.frombuffer(witness.core_output_f16, dtype="<f2")
        if not np.isfinite(captured_output).all():
            raise ProofV2EngineError("GDN transition witness is non-finite")
        if not _trace_fp16_matches_v2(
            captured_output,
            replay.out_projection_input[0],
        ):
            raise ProofV2EngineError(
                "GDN transition does not produce the committed out-projection input"
            )
        conv_state = replay.conv_state_after
        recurrent_state = replay.recurrent_state_after
        if (
            gdn_state_digest_v2(
                "conv",
                np.asarray(conv_state, dtype=conv_dtype).tobytes(),
                dtype=signed.runtime_dtype,
            )
            != witness.conv_after_digest
            or gdn_state_digest_v2(
                "recurrent",
                np.asarray(recurrent_state, dtype=recurrent_dtype).tobytes(),
                dtype=signed.recurrent_state_dtype,
            )
            != witness.recurrent_after_digest
            or trace_attention_state_boundary_digest_v2(
                TRACE_ATTENTION_GDN_TRANSITION_V1,
                (witness.conv_after_digest, witness.recurrent_after_digest),
            )
            != token.attention_state_after_digests[parameters.layer]
        ):
            raise ProofV2EngineError(
                "GDN replay post-state does not match the committed trace"
            )


def _verify_full_attention_decode_suffix_transition_v2(
    *,
    witnesses: Sequence[FullAttentionHeadWitnessV2],
    tokens,
    parameters,
    opening: FullAttentionHeadStateOpeningV2,
    query_head: int,
) -> None:
    """Replay one nonce-selected logical full-attention head.

    The opening is a logical (rather than paged-cache) K/V prefix.  Every
    replayed row proves membership in the pre-committed per-head cache roots,
    so a miner cannot replace cache state after learning the challenge.  The
    selected Q/K/V values and the corresponding core-output head are opened
    from a Merkle root frozen in each token trace before nonce selection.
    """

    if parameters.attention_profile != TRACE_ATTENTION_FULL_TRANSITION_V1:
        raise ProofV2EngineError(
            "full-attention transition profile does not match the manifest"
        )
    if parameters.transition_profile != FULL_ATTENTION_TRANSITION_PROFILE_V1:
        raise ProofV2EngineError(
            "full-attention transition parameters are not supported"
        )
    try:
        signed = FullAttentionTransitionParametersV2.from_canonical_bytes(
            parameters.transition_parameters
        )
    except ProofV2TransitionError as exc:
        raise ProofV2EngineError(
            "full-attention transition parameters are malformed"
        ) from exc
    if opening.layer_idx != parameters.layer:
        raise ProofV2EngineError("full-attention state opening layer is inconsistent")
    if opening.head_dim != signed.head_dim:
        raise ProofV2EngineError(
            "full-attention state opening head dimension is invalid"
        )
    expected_kv_head = query_head // (
        signed.num_query_heads // signed.num_key_value_heads
    )
    if opening.kv_head_idx != expected_kv_head:
        raise ProofV2EngineError("full-attention state opening head is inconsistent")
    if (
        not witnesses
        or tuple(item.token_index for item in witnesses)
        != tuple(
            range(
                GDN_DECODE_SUFFIX_TOKEN_START_V1,
                GDN_DECODE_SUFFIX_TOKEN_START_V1 + len(witnesses),
            )
        )
        or opening.trace_row_count != len(witnesses)
        or opening.position_start != opening.prefix_token_count
    ):
        raise ProofV2EngineError("full-attention transition trace rows are not exact")
    if any(
        item.layer_idx != parameters.layer
        or item.query_head_idx != query_head
        or item.head_dim != signed.head_dim
        for item in witnesses
    ):
        raise ProofV2EngineError(
            "full-attention transition trace profile is inconsistent"
        )

    prefix_shape = (
        opening.prefix_token_count,
        signed.num_key_value_heads,
        signed.head_dim,
    )
    initial_keys = np.zeros(prefix_shape, dtype="<f2")
    initial_values = np.zeros(prefix_shape, dtype="<f2")
    initial_keys[:, opening.kv_head_idx, :] = np.frombuffer(
        opening.prefix_keys_f16, dtype="<f2"
    ).reshape(opening.prefix_token_count, signed.head_dim)
    initial_values[:, opening.kv_head_idx, :] = np.frombuffer(
        opening.prefix_values_f16, dtype="<f2"
    ).reshape(opening.prefix_token_count, signed.head_dim)
    qkv_rows = np.zeros((len(witnesses), signed.qkv_width), dtype="<f2")
    kv_head = expected_kv_head
    for row_index, witness in enumerate(witnesses):
        try:
            token = tokens[witness.token_index]
        except (IndexError, TypeError) as exc:
            raise ProofV2EngineError(
                "full-attention transition token witness is out of range"
            ) from exc
        if (
            trace_attention_state_boundary_digest_v2(
                TRACE_ATTENTION_FULL_TRANSITION_V1,
                (witness.kv_before_root,),
            )
            != token.attention_state_before_digests[parameters.layer]
            or trace_attention_state_boundary_digest_v2(
                TRACE_ATTENTION_FULL_TRANSITION_V1,
                (witness.kv_after_root,),
            )
            != token.attention_state_after_digests[parameters.layer]
        ):
            raise ProofV2EngineError(
                "full-attention witness state roots do not match the committed trace"
            )
        qkv_rows[row_index, : signed.q_width].reshape(
            signed.num_query_heads,
            signed.head_dim,
        )[query_head] = np.frombuffer(witness.query_f16, dtype="<f2")
        qkv_rows[
            row_index,
            signed.q_width : signed.q_width + signed.kv_width,
        ].reshape(signed.num_key_value_heads, signed.head_dim)[kv_head] = np.frombuffer(
            witness.key_f16,
            dtype="<f2",
        )
        qkv_rows[
            row_index,
            signed.q_width + signed.kv_width :,
        ].reshape(signed.num_key_value_heads, signed.head_dim)[kv_head] = np.frombuffer(
            witness.value_f16,
            dtype="<f2",
        )
    if not np.isfinite(qkv_rows).all():
        raise ProofV2EngineError("full-attention witness is non-finite")
    try:
        replay = replay_qwen_full_attention_head_v2(
            qkv_rows=qkv_rows,
            initial_keys=initial_keys,
            initial_values=initial_values,
            query_head=query_head,
            position_start=opening.position_start,
            parameters=signed,
        )
    except (ProofV2TransitionError, ValueError) as exc:
        raise ProofV2EngineError("full-attention transition replay is invalid") from exc

    key_history = (
        np.frombuffer(opening.prefix_keys_f16, dtype="<f2")
        .reshape(opening.prefix_token_count, signed.head_dim)
        .copy()
    )
    value_history = (
        np.frombuffer(opening.prefix_values_f16, dtype="<f2")
        .reshape(opening.prefix_token_count, signed.head_dim)
        .copy()
    )
    for row_index, witness in enumerate(witnesses):
        before_leaf = full_attention_head_state_leaf_v2(
            kv_head_idx=opening.kv_head_idx,
            head_dim=signed.head_dim,
            key_values_f16=np.asarray(key_history, dtype="<f2").tobytes(),
            value_values_f16=np.asarray(value_history, dtype="<f2").tobytes(),
        )
        if not verify_merkle_path(
            witness.kv_before_root,
            before_leaf,
            opening.before_paths[row_index],
        ):
            raise ProofV2EngineError(
                "full-attention state opening does not match the committed pre-state"
            )
        key_history = np.concatenate(
            (
                key_history,
                np.asarray(replay.rotated_keys[row_index], dtype="<f2").reshape(1, -1),
            ),
            axis=0,
        )
        value_history = np.concatenate(
            (
                value_history,
                np.asarray(replay.values[row_index], dtype="<f2").reshape(1, -1),
            ),
            axis=0,
        )
        after_leaf = full_attention_head_state_leaf_v2(
            kv_head_idx=opening.kv_head_idx,
            head_dim=signed.head_dim,
            key_values_f16=np.asarray(key_history, dtype="<f2").tobytes(),
            value_values_f16=np.asarray(value_history, dtype="<f2").tobytes(),
        )
        if not verify_merkle_path(
            witness.kv_after_root,
            after_leaf,
            opening.after_paths[row_index],
        ):
            raise ProofV2EngineError(
                "full-attention state opening does not match the committed post-state"
            )
        captured_core = np.frombuffer(witness.core_output_f16, dtype="<f2")
        if not np.isfinite(captured_core).all():
            raise ProofV2EngineError("full-attention witness is non-finite")
        expected_core = replay.core_output[row_index]
        if not _trace_fp16_matches_v2(
            captured_core,
            expected_core,
        ):
            raise ProofV2EngineError(
                "full-attention transition does not produce the committed core-output head"
            )


def _opened_transition_witness_root_v2(layer, parameters) -> bytes:
    """Rebuild a selected raw layer's compact transition witness root."""

    if layer.attention_profile == TRACE_ATTENTION_GDN_TRANSITION_V1:
        return GDNTransitionWitnessV2(
            token_index=layer.token_index,
            layer_idx=layer.layer_idx,
            qkvz_f16=layer.tensor("gdn_qkvz").values,
            ba_f16=layer.tensor("gdn_ba").values,
            core_output_f16=layer.tensor("attention_core_out").values,
            conv_before_digest=layer.tensor("gdn_conv_before_digest").values,
            conv_after_digest=layer.tensor("gdn_conv_after_digest").values,
            recurrent_before_digest=layer.tensor(
                "gdn_recurrent_before_digest"
            ).values,
            recurrent_after_digest=layer.tensor(
                "gdn_recurrent_after_digest"
            ).values,
        ).digest()
    if layer.attention_profile != TRACE_ATTENTION_FULL_TRANSITION_V1:
        raise ProofV2EngineError("opened transition profile is unsupported")
    try:
        signed = FullAttentionTransitionParametersV2.from_canonical_bytes(
            parameters.transition_parameters
        )
    except ProofV2TransitionError as exc:
        raise ProofV2EngineError(
            "full-attention transition parameters are malformed"
        ) from exc
    qkv = _trace_f16_vector_v2(layer, "attention_qkv")
    core = _trace_f16_vector_v2(layer, "attention_core_out")
    if qkv.shape != (signed.qkv_width,) or core.shape != (signed.q_width,):
        raise ProofV2EngineError("full-attention trace dimensions are invalid")
    query_rows = qkv[: signed.q_width].reshape(
        signed.num_query_heads,
        signed.head_dim,
    )
    key_rows = qkv[
        signed.q_width : signed.q_width + signed.kv_width
    ].reshape(signed.num_key_value_heads, signed.head_dim)
    value_rows = qkv[signed.q_width + signed.kv_width :].reshape(
        signed.num_key_value_heads,
        signed.head_dim,
    )
    core_rows = core.reshape(signed.num_query_heads, signed.head_dim)
    before_root = layer.tensor("kv_before_digest").values
    after_root = layer.tensor("kv_after_digest").values
    provisional = tuple(
        FullAttentionHeadWitnessV2(
            token_index=layer.token_index,
            layer_idx=layer.layer_idx,
            query_head_idx=query_head,
            head_dim=signed.head_dim,
            query_f16=np.asarray(query_rows[query_head], dtype="<f2").tobytes(),
            key_f16=np.asarray(
                key_rows[
                    query_head
                    // (signed.num_query_heads // signed.num_key_value_heads)
                ],
                dtype="<f2",
            ).tobytes(),
            value_f16=np.asarray(
                value_rows[
                    query_head
                    // (signed.num_query_heads // signed.num_key_value_heads)
                ],
                dtype="<f2",
            ).tobytes(),
            core_output_f16=np.asarray(core_rows[query_head], dtype="<f2").tobytes(),
            kv_before_root=before_root,
            kv_after_root=after_root,
            merkle_path=MerklePath(query_head, ()),
        )
        for query_head in range(signed.num_query_heads)
    )
    root, _tree = build_full_attention_witness_root_v2(provisional)
    return root


def _verify_attention_state_digest_update_v2(
    layer,
    *,
    require_independent_attention_transition: bool,
) -> None:
    """Reject state hashes that have no independently verified transition.

    The old ``*_audit_only`` profiles merely hashed prover-provided projection
    outputs into a new digest.  A prover can create that hash chain for an
    arbitrary prompt or substitute execution, so it must never discharge the
    hard execution-binding requirement.  A supported profile must instead
    provide retained-state openings and a verifier-side transition replay.
    """

    if layer.attention_profile in (
        TRACE_ATTENTION_GDN_TRANSITION_V1,
        TRACE_ATTENTION_FULL_TRANSITION_V1,
    ):
        return

    if require_independent_attention_transition:
        raise ProofV2EngineError(
            "causal execution trace uses an audit-only attention state without "
            "an independently verifiable transition"
        )

    # This compatibility branch exists only so isolated arithmetic/trace
    # plumbing tests can exercise the historical wire format.  Production
    # callers use the default above and cannot obtain a hard execution verdict
    # from this hash relation.
    if layer.attention_profile == "full_attention_audit_only":
        before_names = ("kv_before_digest",)
        after_names = ("kv_after_digest",)
        payload_names = ("attention_qkv",)
    elif layer.attention_profile == "gdn_attention_audit_only":
        before_names = (
            "gdn_conv_before_digest",
            "gdn_recurrent_before_digest",
        )
        after_names = (
            "gdn_conv_after_digest",
            "gdn_recurrent_after_digest",
        )
        payload_names = ("gdn_qkvz", "gdn_ba")
    else:
        raise ProofV2EngineError("causal trace attention profile is unsupported")
    before_values = tuple(layer.tensor(name).values for name in before_names)
    if len(set(before_values)) != 1:
        raise ProofV2EngineError("causal trace state-before digests disagree")
    expected = hashlib.sha256(
        b"VERATHOS/PROOF_V2/TRACE_STATE/STEP/SHA256"
        + layer.attention_profile.encode("ascii")
        + before_values[0]
        + b"".join(layer.tensor(name).values for name in payload_names)
    ).digest()
    if any(layer.tensor(name).values != expected for name in after_names):
        raise ProofV2EngineError("causal trace state digest update is invalid")


def _verify_execution_trace_binding_v2(
    *,
    manifest: StaticWeightCommitmentManifest,
    trace_commitment,
    payload: ProofV2Payload,
    transcript_state: bytes,
    expected_challenges: Sequence[BlockChallengeV2] | None = None,
    require_independent_attention_transition: bool,
    require_full_decode_corridor: bool = False,
) -> None:
    """Bind selected arithmetic rows to the pre-challenge causal trace."""

    if not payload.execution_trace_proof:
        raise ProofV2EngineError("causal execution trace proof is missing")
    try:
        trace_proof = ExecutionTraceProofV2.from_canonical_bytes(
            payload.execution_trace_proof
        )
        primary_positions = set()
        primary_descriptors = tuple(block.descriptor for block in payload.block_proofs)
        if require_full_decode_corridor:
            if expected_challenges is None:
                raise ProofV2EngineError(
                    "hard execution corridor requires transcript-derived challenges"
                )
            primary_descriptors = tuple(expected_challenges)
        for descriptor in primary_descriptors:
            if not 0 <= descriptor.key.layer_idx < manifest.model_spec.num_layers:
                continue
            for token_index in range(
                descriptor.row_offset,
                descriptor.row_offset + descriptor.rows,
            ):
                primary_positions.add((token_index, descriptor.key.layer_idx))
        hard_corridor_row = None
        transition_positions = set(primary_positions)
        if require_full_decode_corridor:
            (
                hard_corridor_row,
                _selected_transition_layers,
                corridor_positions,
            ) = derive_hard_execution_corridor_v2(
                primary_descriptors,
                num_layers=manifest.model_spec.num_layers,
            )
            if hard_corridor_row < GDN_DECODE_SUFFIX_TOKEN_START_V1:
                raise ProofV2EngineError(
                    "hard execution corridor selected a pre-boundary trace row"
                )
            expected_positions = set(corridor_positions)
        else:
            expected_positions = set(primary_positions)
            for token_index, layer_idx in tuple(primary_positions):
                expected_positions.add((token_index, 0))
                expected_positions.add(
                    (token_index, manifest.model_spec.num_layers - 1)
                )
        profiles = {
            descriptor.layer: descriptor for descriptor in manifest.layer_execution
        }
        gdn_transition_layers = tuple(
            sorted(
                {
                    layer_idx
                    for _token_index, layer_idx in transition_positions
                    if profiles.get(layer_idx) is not None
                    and profiles[layer_idx].attention_profile
                    == TRACE_ATTENTION_GDN_TRANSITION_V1
                }
            )
        )
        full_transition_layers = tuple(
            sorted(
                {
                    layer_idx
                    for _token_index, layer_idx in transition_positions
                    if profiles.get(layer_idx) is not None
                    and profiles[layer_idx].attention_profile
                    == TRACE_ATTENTION_FULL_TRANSITION_V1
                }
            )
        )
        # A vLLM cache boundary follows trace row 0 (the final prompt-token
        # forward pass).  Treating it as state-before row 0 would let a miner
        # attach an unrelated post-prefill state to that row.  The hard
        # transition ABI therefore needs at least one generated suffix row and
        # never accepts a selected transition row at index 0.
        if (
            gdn_transition_layers or full_transition_layers
        ) and trace_commitment.token_count <= (GDN_DECODE_SUFFIX_TOKEN_START_V1):
            raise ProofV2EngineError(
                "GDN transition audit requires a generated decode suffix"
            )
        if any(
            token_index < GDN_DECODE_SUFFIX_TOKEN_START_V1
            and layer_idx in (gdn_transition_layers + full_transition_layers)
            for token_index, layer_idx in transition_positions
        ):
            raise ProofV2EngineError(
                "GDN transition audit selected a pre-boundary trace row"
            )
        trace_proof.verify(
            trace_commitment,
            output_token_ids=tuple(
                token.output_token_id for token in trace_proof.tokens
            ),
            expected_layer_positions=expected_positions,
            expected_first_input_token_id=trace_proof.tokens[0].input_token_id,
        )
    except (ProofV2TraceError, IndexError, TypeError, ValueError) as exc:
        raise ProofV2EngineError(
            "causal execution trace proof does not match its commitment"
        ) from exc

    opened = {
        (layer.token_index, layer.layer_idx): layer
        for layer in trace_proof.opened_layers
    }
    if require_full_decode_corridor:
        if hard_corridor_row is None:
            raise ProofV2EngineError("hard execution corridor row is missing")
        try:
            validate_layer_trace_set_v2(
                trace_proof.tokens[hard_corridor_row],
                tuple(
                    opened[(hard_corridor_row, layer_idx)]
                    for layer_idx in range(manifest.model_spec.num_layers)
                ),
            )
        except (KeyError, ProofV2TraceError) as exc:
            raise ProofV2EngineError(
                "hard execution trace corridor is incomplete or discontinuous"
            ) from exc
    operations_by_key = {
        operation.key: operation
        for operation in registered_operations_from_manifest(manifest)
    }
    transition_layers = gdn_transition_layers + full_transition_layers
    if transition_layers and any(
        not token.transition_witness_roots
        or len(token.transition_witness_roots) != manifest.model_spec.num_layers
        for token in trace_proof.tokens[
            GDN_DECODE_SUFFIX_TOKEN_START_V1 :
        ]
    ):
        raise ProofV2EngineError(
            "causal execution trace lacks compact transition witness roots"
        )
    for (token_index, layer_idx), layer in opened.items():
        parameters = profiles.get(layer_idx)
        if (
            parameters is None
            or layer.attention_profile != parameters.attention_profile
        ):
            raise ProofV2EngineError(
                "opened trace layer profile does not match the manifest"
            )
        _verify_weightless_layer_bridge_v2(layer, parameters)
        _verify_attention_state_digest_update_v2(
            layer,
            require_independent_attention_transition=(
                require_independent_attention_transition
            ),
        )
        if layer.attention_profile in (
            TRACE_ATTENTION_GDN_TRANSITION_V1,
            TRACE_ATTENTION_FULL_TRANSITION_V1,
        ):
            token = trace_proof.tokens[token_index]
            if (
                not token.transition_witness_roots
                or _opened_transition_witness_root_v2(layer, parameters)
                != token.transition_witness_roots[layer_idx]
            ):
                raise ProofV2EngineError(
                    "opened layer does not match its compact transition witness"
                )

    if require_independent_attention_transition:
        openings_by_layer = {
            item.layer_idx: item for item in trace_proof.gdn_initial_state_openings
        }
        if tuple(sorted(openings_by_layer)) != gdn_transition_layers:
            raise ProofV2EngineError("GDN transition state opening set is not exact")
        gdn_witnesses_by_position = {
            (item.token_index, item.layer_idx): item
            for item in trace_proof.gdn_transition_witnesses
        }
        expected_gdn_positions = tuple(
            (token_index, layer_idx)
            for token_index in range(
                GDN_DECODE_SUFFIX_TOKEN_START_V1,
                trace_commitment.token_count,
            )
            for layer_idx in gdn_transition_layers
        )
        if tuple(sorted(gdn_witnesses_by_position)) != expected_gdn_positions:
            raise ProofV2EngineError("GDN transition witness set is not exact")
        for layer_idx in gdn_transition_layers:
            _verify_gdn_decode_suffix_transition_v2(
                witnesses=tuple(
                    gdn_witnesses_by_position[(token_index, layer_idx)]
                    for token_index in range(
                        GDN_DECODE_SUFFIX_TOKEN_START_V1,
                        trace_commitment.token_count,
                    )
                ),
                tokens=trace_proof.tokens,
                parameters=profiles[layer_idx],
                opening=openings_by_layer[layer_idx],
            )
        try:
            audit_policy = getattr(
                getattr(manifest, "model_execution", None),
                "audit_policy",
                None,
            )
            if audit_policy is None:
                raise ProofV2EngineError(
                    "full-attention transition profile lacks a signed hard-audit policy"
                )
            signed_full = {
                layer_idx: FullAttentionTransitionParametersV2.from_canonical_bytes(
                    profiles[layer_idx].transition_parameters
                )
                for layer_idx in full_transition_layers
            }
            transition_challenges = (
                derive_transition_challenges_v2(
                    transcript_state=transcript_state,
                    trace_commitment_digest=trace_commitment.digest(),
                    execution_row_count=trace_commitment.token_count,
                    layer_head_counts=tuple(
                        signed_full.get(layer_idx).num_query_heads
                        if layer_idx in signed_full
                        else 1
                        for layer_idx in range(manifest.model_spec.num_layers)
                    ),
                    selected_layers=full_transition_layers,
                    heads_per_layer=audit_policy.full_attention_heads_per_layer,
                )
                if full_transition_layers
                else ()
            )
        except ProofV2TransitionError as exc:
            raise ProofV2EngineError(
                "full-attention transition challenges are invalid"
            ) from exc
        expected_full_openings = tuple(
            sorted(
                (
                    challenge.layer_idx,
                    head
                    // (
                        signed_full[challenge.layer_idx].num_query_heads
                        // signed_full[challenge.layer_idx].num_key_value_heads
                    ),
                )
                for challenge in transition_challenges
                for head in challenge.heads
            )
        )
        expected_full_openings = tuple(
            item
            for index, item in enumerate(expected_full_openings)
            if not index or item != expected_full_openings[index - 1]
        )
        received_full_openings = tuple(
            (item.layer_idx, item.kv_head_idx)
            for item in trace_proof.full_attention_state_openings
        )
        if received_full_openings != expected_full_openings:
            raise ProofV2EngineError("full-attention state opening set is not exact")
        full_openings_by_key = {
            (item.layer_idx, item.kv_head_idx): item
            for item in trace_proof.full_attention_state_openings
        }
        full_witnesses_by_position = {
            (item.token_index, item.layer_idx, item.query_head_idx): item
            for item in trace_proof.full_attention_head_witnesses
        }
        expected_full_witness_positions = tuple(
            (token_index, challenge.layer_idx, query_head)
            for token_index in range(
                GDN_DECODE_SUFFIX_TOKEN_START_V1,
                trace_commitment.token_count,
            )
            for challenge in transition_challenges
            for query_head in challenge.heads
        )
        if tuple(sorted(full_witnesses_by_position)) != expected_full_witness_positions:
            raise ProofV2EngineError(
                "full-attention transition witness set is not exact"
            )
        challenges_by_layer = {item.layer_idx: item for item in transition_challenges}
        for layer_idx, challenge in challenges_by_layer.items():
            for query_head in challenge.heads:
                kv_head = query_head // (
                    signed_full[layer_idx].num_query_heads
                    // signed_full[layer_idx].num_key_value_heads
                )
                _verify_full_attention_decode_suffix_transition_v2(
                    witnesses=tuple(
                        full_witnesses_by_position[
                            (token_index, layer_idx, query_head)
                        ]
                        for token_index in range(
                            GDN_DECODE_SUFFIX_TOKEN_START_V1,
                            trace_commitment.token_count,
                        )
                    ),
                    tokens=trace_proof.tokens,
                    parameters=profiles[layer_idx],
                    opening=full_openings_by_key[(layer_idx, kv_head)],
                    query_head=query_head,
                )

    model_parameters = manifest.model_execution
    if model_parameters is None:
        raise ProofV2EngineError("causal trace model-boundary parameters are missing")
    last_layer_idx = manifest.model_spec.num_layers - 1
    for (token_index, layer_idx), layer in opened.items():
        if layer_idx != last_layer_idx:
            continue
        final_hidden = _gemma_rmsnorm_v2(
            _trace_f16_vector_v2(layer, "residual_out"),
            model_parameters.final_norm_weight_f16,
            model_parameters.final_norm_epsilon_q32,
        )
        token = trace_proof.tokens[token_index]
        if len(token.final_hidden_f16) != manifest.model_spec.hidden_dim * 2:
            raise ProofV2EngineError(
                "causal trace final hidden row has the wrong dimension"
            )
        captured_final_hidden = np.frombuffer(
            token.final_hidden_f16,
            dtype="<f2",
        )
        if not np.isfinite(captured_final_hidden).all() or not _trace_fp16_matches_v2(
            captured_final_hidden,
            final_hidden,
        ):
            raise ProofV2EngineError(
                "causal trace final RMSNorm output is not bound to its token leaf"
            )
    for block in payload.block_proofs:
        descriptor = block.descriptor
        layer_idx = descriptor.key.layer_idx
        if not 0 <= layer_idx < manifest.model_spec.num_layers:
            continue
        try:
            input_name, output_name = execution_trace_fields_for_operation(
                descriptor.key.operation_id
            )
        except Exception as exc:
            raise ProofV2EngineError(
                "challenged operation is not mapped into the causal trace"
            ) from exc
        x_rows = np.frombuffer(block.x_rows, dtype=np.int8).reshape(
            descriptor.rows,
            descriptor.inner_dim,
        )
        scales = struct.unpack(f"<{descriptor.rows}Q", block.x_scales_q32)
        runtime_rows = np.frombuffer(block.runtime_y_values, dtype="<f2").reshape(
            descriptor.rows,
            RUNTIME_Y_COMMITMENT_BLOCK_COLS,
        )
        segment_offset = (
            descriptor.column_offset // RUNTIME_Y_COMMITMENT_BLOCK_COLS
        ) * RUNTIME_Y_COMMITMENT_BLOCK_COLS
        operation = operations_by_key.get(descriptor.key)
        if operation is None:
            raise ProofV2EngineError(
                "trace operation is not registered in the manifest"
            )
        segment_length = min(
            RUNTIME_Y_COMMITMENT_BLOCK_COLS,
            operation.output_dim - segment_offset,
        )
        if segment_length <= 0:
            raise ProofV2EngineError("trace output segment is out of range")

        for row_index in range(descriptor.rows):
            token_index = descriptor.row_offset + row_index
            layer = opened[(token_index, layer_idx)]
            input_tensor = layer.tensor(input_name)
            output_tensor = layer.tensor(output_name)
            if input_tensor.dtype != "f16" or input_tensor.shape != (
                descriptor.inner_dim,
            ):
                raise ProofV2EngineError(
                    "trace operation input shape or dtype is not canonical"
                )
            try:
                quantized, scale_q32 = quantize_canonical_fp16_rows_v2(
                    np.frombuffer(input_tensor.values, dtype="<f2").reshape(1, -1)
                )
            except Exception as exc:
                raise ProofV2EngineError("trace operation input is non-finite") from exc
            if not np.array_equal(quantized[0], x_rows[row_index]) or (
                scale_q32[0] != scales[row_index]
            ):
                raise ProofV2EngineError(
                    "arithmetic X row is not the committed execution input"
                )
            if output_tensor.dtype != "f16" or output_tensor.shape != (
                operation.output_dim,
            ):
                raise ProofV2EngineError(
                    "trace operation output shape or dtype is not canonical"
                )
            output_values = np.frombuffer(output_tensor.values, dtype="<f2")
            expected_segment = np.zeros(
                RUNTIME_Y_COMMITMENT_BLOCK_COLS,
                dtype="<f2",
            )
            expected_segment[:segment_length] = output_values[
                segment_offset : segment_offset + segment_length
            ]
            if not np.array_equal(expected_segment, runtime_rows[row_index]):
                raise ProofV2EngineError(
                    "runtime Y row is not the committed execution output"
                )


def verify_inference_v2(
    *,
    manifest: StaticWeightCommitmentManifest,
    commitment_hash: bytes,
    beacon: bytes,
    commitment_envelope: ProofV2CommitmentEnvelope,
    payload: ProofV2Payload,
    expected_challenges: Sequence[BlockChallengeV2],
    operations: Sequence[RegisteredOperationV2] | None = None,
    require_execution_trace_binding: bool = True,
    require_independent_attention_transition: bool = True,
    require_full_decode_corridor: bool = False,
) -> None:
    """Verify exact proof set, disclosed X/Y, sumchecks, and batched W opening."""

    if type(require_full_decode_corridor) is not bool:
        raise ProofV2EngineError("hard execution corridor requirement is invalid")
    if require_full_decode_corridor and not require_execution_trace_binding:
        raise ProofV2EngineError(
            "hard execution corridor requires causal trace verification"
        )

    manifest_digest = manifest.digest()
    if commitment_envelope.manifest_digest != manifest_digest:
        raise ProofV2EngineError("commitment envelope manifest digest does not match")
    if payload.manifest_digest != manifest_digest:
        raise ProofV2EngineError("final payload manifest digest does not match")
    trace_commitment = commitment_envelope.execution_trace_commitment
    if manifest.execution_profile is not None:
        if trace_commitment is None:
            raise ProofV2EngineError("causal execution trace commitment is missing")
        if (
            trace_commitment.profile != manifest.execution_profile
            or trace_commitment.num_layers != manifest.model_spec.num_layers
        ):
            raise ProofV2EngineError(
                "causal execution trace commitment does not match the manifest"
            )
    operations = _resolve_operations(manifest, operations)
    operations_by_key = {operation.key: operation for operation in operations}
    manifest_descriptors = operation_descriptor_by_key(manifest)
    x_by_key = {item.key: item for item in commitment_envelope.x_commitments}
    runtime_y_by_key = {
        item.key: item for item in commitment_envelope.runtime_y_commitments
    }
    if set(x_by_key) != set(operations_by_key):
        raise ProofV2EngineError("commitment envelope operation set is not exact")
    if set(runtime_y_by_key) != set(operations_by_key):
        raise ProofV2EngineError("runtime Y commitment operation set is not exact")
    if trace_commitment is not None and not _execution_operation_row_counts_match_v2(
        commitment_envelope.x_commitments,
        token_count=trace_commitment.token_count,
        num_layers=manifest.model_spec.num_layers,
    ):
        raise ProofV2EngineError(
            "execution operation row counts do not match the committed token trace"
        )
    for key, operation in operations_by_key.items():
        if x_by_key[key].inner_dim != operation.inner_dim:
            raise ProofV2EngineError(
                "commitment envelope inner dimension does not match the manifest"
            )
        runtime_y = runtime_y_by_key[key]
        if (
            runtime_y.row_count != x_by_key[key].row_count
            or runtime_y.output_dim != operation.output_dim
            or runtime_y.block_rows != RUNTIME_Y_COMMITMENT_BLOCK_ROWS
            or runtime_y.block_cols != RUNTIME_Y_COMMITMENT_BLOCK_COLS
        ):
            raise ProofV2EngineError(
                "runtime Y commitment layout does not match the manifest"
            )
    descriptors = tuple(block.descriptor for block in payload.block_proofs)
    validate_exact_block_proof_set_v2(tuple(expected_challenges), descriptors)
    if trace_commitment is not None and require_execution_trace_binding:
        _verify_execution_trace_binding_v2(
            manifest=manifest,
            trace_commitment=trace_commitment,
            payload=payload,
            transcript_state=beacon,
            expected_challenges=expected_challenges,
            require_independent_attention_transition=(
                require_independent_attention_transition
            ),
            require_full_decode_corridor=require_full_decode_corridor,
        )
    elif payload.execution_trace_proof:
        raise ProofV2EngineError(
            "unchallenged or uncommitted execution trace proof is present"
        )
    batch_rows = max(item.rows for item in descriptors)
    batch_columns = max(item.cols for item in descriptors)

    from zkllm.crypto.pcs_v2 import (
        combine_commitments,
        commit,
        verify,
    )

    batch_statements: list[GemmV2Statement] = []
    verified_x_matrices: list[np.ndarray] = []
    verified_w_commitments: list[tuple[bytes, ...]] = []
    verified_y_matrices: list[np.ndarray] = []

    for block in payload.block_proofs:
        descriptor = block.descriptor
        operation = operations_by_key[descriptor.key]
        manifest_descriptor = manifest_descriptors[descriptor.key]
        x_metadata = x_by_key[descriptor.key]
        runtime_y_metadata = runtime_y_by_key[descriptor.key]
        x_indices = tuple(
            range(descriptor.row_offset, descriptor.row_offset + descriptor.rows)
        )
        w_indices = tuple(
            range(descriptor.column_offset, descriptor.column_offset + descriptor.cols)
        )
        x_multiproof = PcsMerkleV2MultiProof.from_canonical_bytes(
            block.x_membership_proof
        )
        runtime_y_multiproof = PcsMerkleV2MultiProof.from_canonical_bytes(
            block.runtime_y_membership_proof
        )
        w_multiproof = PcsMerkleV2MultiProof.from_canonical_bytes(
            block.w_membership_proof
        )
        x_opened = verify_pcs_merkle_v2_multiproof(
            _tree_context(
                operation,
                domain=PCS_MERKLE_DOMAIN_PRECHALLENGE_X,
                leaf_count=x_metadata.row_count,
            ),
            x_metadata.row_commitment_root,
            x_multiproof,
            expected_indices=x_indices,
        )
        runtime_y_segments = runtime_y_column_segments(operation)
        runtime_y_leaf_count = len(row_segments(operation, x_metadata.row_count)) * len(
            runtime_y_segments
        )
        runtime_y_block_col = (
            descriptor.column_offset // RUNTIME_Y_COMMITMENT_BLOCK_COLS
        )
        runtime_y_leaf_index = (
            descriptor.block_row * len(runtime_y_segments) + runtime_y_block_col
        )
        runtime_y_opened = verify_pcs_merkle_v2_multiproof(
            _tree_context(
                operation,
                domain=PCS_MERKLE_DOMAIN_PRECHALLENGE_RUNTIME_Y,
                leaf_count=runtime_y_leaf_count,
            ),
            runtime_y_metadata.block_commitment_root,
            runtime_y_multiproof,
            expected_indices=(runtime_y_leaf_index,),
        )
        w_opened = verify_pcs_merkle_v2_multiproof(
            _tree_context(
                operation,
                domain=PCS_MERKLE_DOMAIN_STATIC_WEIGHT,
                leaf_count=operation.output_dim,
            ),
            operation.weight_commitment_root,
            w_multiproof,
            expected_indices=w_indices,
        )
        x_row_digests = tuple(item.leaf.commitment for item in x_opened)
        w_commitments = tuple(item.leaf.commitment for item in w_opened)
        x_matrix = np.frombuffer(block.x_rows, dtype=np.int8).reshape(
            descriptor.rows,
            descriptor.inner_dim,
        )
        if not quantized_x_profile_is_nondegenerate_v2(x_matrix):
            raise ProofV2EngineError(
                "proof-v2 X row is outside the nondegenerate trace profile"
            )
        x_scales = struct.unpack(f"<{descriptor.rows}Q", block.x_scales_q32)
        for row_index, opened in enumerate(x_opened):
            row = np.ascontiguousarray(x_matrix[row_index], dtype=np.int8).tobytes()
            if opened.leaf.encoding_id != PCS_ENCODING_HASHED_QUANTIZED_I8_ROW or (
                opened.leaf.commitment
                != hash_quantized_i8_row_v2(
                    row,
                    descriptor.inner_dim,
                    x_scales[row_index],
                )
            ):
                raise ProofV2EngineError(
                    "X row data does not match its authenticated digest"
                )
        runtime_y_segment = np.frombuffer(block.runtime_y_values, dtype="<f2").reshape(
            descriptor.rows,
            RUNTIME_Y_COMMITMENT_BLOCK_COLS,
        )
        runtime_y_bytes = np.ascontiguousarray(
            runtime_y_segment,
            dtype="<f2",
        ).tobytes()
        if (
            len(runtime_y_opened) != 1
            or runtime_y_opened[0].leaf.encoding_id != PCS_ENCODING_HASHED_FP16_BLOCK
            or runtime_y_opened[0].leaf.logical_vector_length
            != descriptor.rows * RUNTIME_Y_COMMITMENT_BLOCK_COLS
            or runtime_y_opened[0].leaf.commitment
            != hash_fp16_block_v2(
                runtime_y_bytes,
                descriptor.rows * RUNTIME_Y_COMMITMENT_BLOCK_COLS,
            )
        ):
            raise ProofV2EngineError(
                "runtime Y data does not match its pre-challenge commitment"
            )
        runtime_y_column_offset, runtime_y_columns = runtime_y_segments[
            runtime_y_block_col
        ]
        if not (
            runtime_y_column_offset <= descriptor.column_offset
            and descriptor.column_offset + descriptor.cols
            <= runtime_y_column_offset + runtime_y_columns
        ):
            raise ProofV2EngineError(
                "challenged runtime Y slice crosses its commitment segment"
            )
        if runtime_y_columns < RUNTIME_Y_COMMITMENT_BLOCK_COLS and np.any(
            runtime_y_segment[:, runtime_y_columns:] != np.float16(0)
        ):
            raise ProofV2EngineError(
                "runtime Y final commitment segment has nonzero padding"
            )
        local_column_offset = descriptor.column_offset - runtime_y_column_offset
        runtime_y_values = runtime_y_segment[
            :,
            local_column_offset : local_column_offset + descriptor.cols,
        ]
        proof_y_matrix = np.frombuffer(block.proof_y_values, dtype="<i8").reshape(
            descriptor.rows,
            descriptor.cols,
        )
        statement = _statement(
            parent_digest=_block_parent_digest(
                commitment_hash=commitment_hash,
                beacon=beacon,
                manifest_digest=manifest_digest,
                descriptor=descriptor,
            ),
            operation=operation,
            descriptor=descriptor,
            x_commitments=x_row_digests,
            w_commitments=w_commitments,
            y_commitment=block.proof_y_commitment,
        )
        proof_y_padded = np.zeros((batch_rows, batch_columns), dtype="<i8")
        proof_y_padded[: descriptor.rows, : descriptor.cols] = proof_y_matrix
        proof_y_flat = tuple(int(value) for value in proof_y_padded.reshape(-1))
        if commit(proof_y_flat, encoding=2) != block.proof_y_commitment:
            raise ProofV2EngineError(
                "proof Y values do not match the authenticated PCS commitment"
            )

        # Bind the exact signed-int8 equation to the output captured before the
        # validator revealed its nonce.  All arithmetic is deterministic Q32;
        # binary16 values multiplied by 2**32 are exact integers in float64.
        runtime_y_q32 = np.rint(
            runtime_y_values.astype(np.float64) * float(1 << 32)
        ).astype(object)
        weight_scale_q32 = operation_weight_scale_q32_v2(
            manifest_descriptor,
            descriptor.column_offset,
        )
        expected_runtime_y_q32 = []
        observed_runtime_y_q32 = []
        for row_index in range(descriptor.rows):
            x_scale_q32 = x_scales[row_index]
            for column_index in range(descriptor.cols):
                expected_q32 = (
                    int(proof_y_matrix[row_index, column_index])
                    * x_scale_q32
                    * weight_scale_q32
                ) >> 32
                observed_q32 = int(runtime_y_q32[row_index, column_index])
                expected_runtime_y_q32.append(expected_q32)
                observed_runtime_y_q32.append(observed_q32)
        absolute_tolerance_q32, relative_tolerance_bps = _effective_runtime_y_tolerance(
            manifest_descriptor.runtime_abs_tolerance_q32,
            manifest_descriptor.runtime_rel_tolerance_bps,
        )
        if not _runtime_y_block_within_tolerance(
            expected_runtime_y_q32,
            observed_runtime_y_q32,
            absolute_tolerance_q32=absolute_tolerance_q32,
            relative_tolerance_bps=relative_tolerance_bps,
        ):
            raise ProofV2EngineError(
                "runtime Y is outside the authenticated block-norm tolerance"
            )

        batch_statements.append(statement)
        verified_x_matrices.append(x_matrix)
        verified_w_commitments.append(w_commitments)
        verified_y_matrices.append(proof_y_padded)

    batch_statement = GemmV2BatchStatement(tuple(batch_statements))
    sumcheck = GemmV2BatchProof.from_canonical_bytes(
        payload.batch_sumcheck_proof,
        expected_blocks=len(payload.block_proofs),
        expected_rounds=batch_statement.inner_bits,
    )
    claims = verify_gemm_v2_batch_sumcheck(batch_statement, sumcheck)
    row_coefficients = _mle_coefficients(claims.row_challenges)
    column_coefficients = _mle_coefficients(claims.column_challenges)
    for claimed_y, proof_y_matrix in zip(claims.y_values, verified_y_matrices):
        disclosed_evaluation = 0
        for row_index, row_coefficient in enumerate(row_coefficients):
            for column_index, column_coefficient in enumerate(column_coefficients):
                disclosed_evaluation += (
                    int(proof_y_matrix[row_index, column_index])
                    * row_coefficient
                    * column_coefficient
                )
        if disclosed_evaluation % PALLAS_SCALAR_MODULUS != claimed_y:
            raise ProofV2EngineError(
                "sumcheck Y evaluation does not match the disclosed Y block"
            )

    inner_coefficients = _mle_coefficients(claims.inner_challenges)
    for claimed_x, x_matrix in zip(claims.x_values, verified_x_matrices):
        disclosed_evaluation = 0
        for row_index, row_coefficient in enumerate(
            row_coefficients[: x_matrix.shape[0]]
        ):
            for inner_index, value in enumerate(x_matrix[row_index]):
                disclosed_evaluation += (
                    int(value) * row_coefficient * inner_coefficients[inner_index]
                )
        if disclosed_evaluation % PALLAS_SCALAR_MODULUS != claimed_x:
            raise ProofV2EngineError(
                "sumcheck X evaluation does not match the disclosed X rows"
            )

    w_context = derive_same_point_batch_opening_context_v2(
        label=b"w",
        statement_digest=claims.batch_statement_digest,
        term_bindings=_opening_term_bindings(batch_statement, (b"w",)),
        evaluations=claims.w_values,
    )
    authenticated_commitments: list[bytes] = []
    authenticated_coefficients: list[int] = []
    for block_index, w_commitments in enumerate(verified_w_commitments):
        w_batch = w_context.coefficients[block_index]
        authenticated_commitments.extend(w_commitments)
        authenticated_coefficients.extend(
            w_batch * coefficient % PALLAS_SCALAR_MODULUS
            for coefficient in column_coefficients[: len(w_commitments)]
        )
    if payload.w_opening.commitment != combine_commitments(
        tuple(authenticated_commitments), tuple(authenticated_coefficients)
    ):
        raise ProofV2EngineError(
            "batched W opening commitment is not the authenticated fold"
        )
    if scalar_from_bytes(payload.w_opening.evaluation) != w_context.combined_evaluation:
        raise ProofV2EngineError(
            "batched W PCS evaluation does not match the sumcheck claims"
        )
    if not verify(
        payload.w_opening,
        claims.inner_challenges,
        w_context.opening_outer_digest,
    ):
        raise ProofV2EngineError("batched W PCS opening did not verify")


__all__ = [
    "InferenceXStateV2",
    "MatrixCommitmentStateV2",
    "RuntimeYCommitmentStateV2",
    "ProofV2EngineError",
    "build_inference_x_state_v2",
    "build_weight_commitment_catalog_tree_v2",
    "build_weight_commitment_root_v2",
    "build_weight_commitment_state_v2",
    "combine_commitment_envelopes_v2",
    "combine_inference_x_states_v2",
    "load_weight_commitment_state_v2",
    "prove_inference_v2",
    "verify_inference_v2",
]
