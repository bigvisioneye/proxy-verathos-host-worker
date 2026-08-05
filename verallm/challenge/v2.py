"""Canonical operation and block challenges for proof protocol v2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import struct
from typing import Iterable, Sequence


PROOF_PROTOCOL_V2 = 2
MAX_OPERATION_ID_BYTES = 64
MAX_REGISTERED_OPERATIONS = 1_000_000
MAX_BLOCKS_PER_OPERATION = 64
MAX_X_ROWS_PER_OPERATION = 4096
RUNTIME_Y_COMMITMENT_BLOCK_COLS = 256
# Inference traces commit every response row, then the validator nonce selects
# one shared row across all challenged operations.  Keeping runtime-Y leaves
# row-granular prevents a short response from accidentally opening every token.
RUNTIME_Y_COMMITMENT_BLOCK_ROWS = 1
MODEL_OPERATION_LAYER_IDX = (1 << 32) - 1
MODEL_LM_HEAD_OPERATION_ID = "model.lm_head"

_DOMAIN = b"VERATHOS_GEMM_CHALLENGE_V2"
_INFERENCE_TRANSCRIPT_DOMAIN = b"VERATHOS_GEMM_INFERENCE_TRANSCRIPT_V2"
_OPERATION_ID = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,63}$")


def _u32(value: int, name: str) -> bytes:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < (1 << 32)
    ):
        raise ValueError(f"{name} must be an unsigned 32-bit integer")
    return struct.pack("<I", value)


def _fixed32(value: bytes, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ValueError(f"{name} must be exactly 32 bytes")
    return value


def _text(value: str, name: str) -> bytes:
    if not isinstance(value, str) or not _OPERATION_ID.fullmatch(value):
        raise ValueError(f"{name} is not a canonical operation identifier")
    raw = value.encode("ascii")
    if len(raw) > MAX_OPERATION_ID_BYTES:
        raise ValueError(f"{name} is too long")
    return struct.pack("<H", len(raw)) + raw


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _ceil_log2(value: int) -> int:
    if value <= 0:
        raise ValueError("dimension must be positive")
    return (value - 1).bit_length()


def derive_inference_transcript_state_v2(
    *,
    validator_nonce: bytes,
    manifest_digest: bytes,
    commitment_envelope: bytes,
    model_id: str,
    model_commitment: bytes,
    input_commitment: bytes,
    prompt_hash: bytes,
    sampler_config_hash: bytes,
    sampling_verification_bps: int,
    do_sample: bool,
    temperature_milli: int,
    presence_penalty_milli: int,
) -> bytes:
    """Bind v2 operation selection to the committed X set and request policy.

    The final block statements separately retain the full response commitment
    hash. This state is limited to stable request context and the canonical X
    envelope so unrelated legacy response fields do not alter operation or
    block selection.
    """

    nonce = _fixed32(validator_nonce, "validator_nonce")
    manifest = _fixed32(manifest_digest, "manifest_digest")
    model_root = _fixed32(model_commitment, "model_commitment")
    input_root = _fixed32(input_commitment, "input_commitment")
    prompt = _fixed32(prompt_hash, "prompt_hash")
    sampler = _fixed32(sampler_config_hash, "sampler_config_hash")
    if not isinstance(commitment_envelope, bytes):
        raise ValueError("commitment_envelope must be bytes")
    envelope = commitment_envelope
    if not envelope or len(envelope) >= (1 << 32):
        raise ValueError("commitment_envelope length is invalid")
    if not isinstance(model_id, str):
        raise ValueError("model_id must be text")
    model = model_id.encode("utf-8")
    if not model or len(model) > 4096:
        raise ValueError("model_id length is invalid")
    if (
        isinstance(sampling_verification_bps, bool)
        or not isinstance(sampling_verification_bps, int)
        or not 0 <= sampling_verification_bps <= 10_000
    ):
        raise ValueError("sampling_verification_bps is invalid")
    if type(do_sample) is not bool:
        raise ValueError("do_sample must be boolean")
    if (
        isinstance(temperature_milli, bool)
        or not isinstance(temperature_milli, int)
        or not 0 <= temperature_milli <= 65_535
    ):
        raise ValueError("temperature_milli is invalid")
    if (
        isinstance(presence_penalty_milli, bool)
        or not isinstance(presence_penalty_milli, int)
        or not -2_000 <= presence_penalty_milli <= 2_000
    ):
        raise ValueError("presence_penalty_milli is invalid")

    encoded = bytearray(_INFERENCE_TRANSCRIPT_DOMAIN)
    encoded.extend(_u32(PROOF_PROTOCOL_V2, "proof version"))
    for value in (nonce, manifest, model_root, input_root, prompt, sampler):
        encoded.extend(struct.pack("<I", len(value)))
        encoded.extend(value)
    encoded.extend(struct.pack("<I", len(model)))
    encoded.extend(model)
    encoded.extend(struct.pack("<I", len(envelope)))
    encoded.extend(envelope)
    encoded.extend(
        struct.pack(
            "<I?Ii",
            sampling_verification_bps,
            do_sample,
            temperature_milli,
            presence_penalty_milli,
        )
    )
    return hashlib.sha256(encoded).digest()


@dataclass(frozen=True, order=True)
class OperationKeyV2:
    """Stable identity of one registered matrix operation."""

    layer_idx: int
    operation_id: str
    expert_idx: int = -1

    def canonical_bytes(self) -> bytes:
        if isinstance(self.expert_idx, bool) or not isinstance(self.expert_idx, int):
            raise ValueError("expert_idx must be an integer")
        if not -1 <= self.expert_idx < (1 << 31):
            raise ValueError("expert_idx is out of range")
        return (
            _u32(self.layer_idx, "layer_idx")
            + struct.pack("<i", self.expert_idx)
            + _text(self.operation_id, "operation_id")
        )


@dataclass(frozen=True)
class RegisteredOperationV2:
    """Verifier-owned dimensions and blocking for one operation."""

    key: OperationKeyV2
    inner_dim: int
    output_dim: int
    block_rows: int
    block_cols: int
    weight_commitment_root: bytes

    def canonical_bytes(self) -> bytes:
        return (
            self.key.canonical_bytes()
            + _u32(self.inner_dim, "inner_dim")
            + _u32(self.output_dim, "output_dim")
            + _u32(self.block_rows, "block_rows")
            + _u32(self.block_cols, "block_cols")
            + _fixed32(self.weight_commitment_root, "weight_commitment_root")
        )

    def validate(self, num_layers: int) -> None:
        self.canonical_bytes()
        if self.key.layer_idx == MODEL_OPERATION_LAYER_IDX:
            if (
                self.key.operation_id != MODEL_LM_HEAD_OPERATION_ID
                or self.key.expert_idx != -1
            ):
                raise ValueError("registered model-level operation is unsupported")
        elif self.key.layer_idx >= num_layers:
            raise ValueError("registered operation layer is outside the ModelSpec")
        if self.inner_dim <= 0 or self.output_dim <= 0:
            raise ValueError("registered operation dimensions must be positive")
        if self.block_rows <= 0 or self.block_cols <= 0:
            raise ValueError("registered operation block dimensions must be positive")
        if not self.block_rows & (self.block_rows - 1) == 0:
            raise ValueError("block_rows must be a power of two")
        if not self.block_cols & (self.block_cols - 1) == 0:
            raise ValueError("block_cols must be a power of two")


@dataclass(frozen=True)
class XCommitmentV2:
    """Pre-challenge commitment metadata for one quantized operation input."""

    key: OperationKeyV2
    row_count: int
    inner_dim: int
    row_commitment_root: bytes

    def canonical_bytes(self) -> bytes:
        return (
            self.key.canonical_bytes()
            + _u32(self.row_count, "row_count")
            + _u32(self.inner_dim, "inner_dim")
            + _fixed32(self.row_commitment_root, "row_commitment_root")
        )


@dataclass(frozen=True)
class RuntimeYCommitmentV2:
    """Pre-challenge block commitment for one captured runtime output.

    Runtime outputs are committed as canonical fp16 segments. The segment
    width is verifier-owned and may be wider than the sampled GEMM block; an
    opened segment authenticates the exact challenged slice it contains.
    """

    key: OperationKeyV2
    row_count: int
    output_dim: int
    block_rows: int
    block_cols: int
    block_commitment_root: bytes

    def canonical_bytes(self) -> bytes:
        return (
            self.key.canonical_bytes()
            + _u32(self.row_count, "row_count")
            + _u32(self.output_dim, "output_dim")
            + _u32(self.block_rows, "block_rows")
            + _u32(self.block_cols, "block_cols")
            + _fixed32(self.block_commitment_root, "block_commitment_root")
        )


@dataclass(frozen=True, order=True)
class BlockChallengeV2:
    """One exact selected block statement."""

    key: OperationKeyV2
    block_row: int
    block_col: int
    row_offset: int
    column_offset: int
    rows: int
    inner_dim: int
    padded_inner_dim: int
    cols: int
    row_rounds: int
    inner_rounds: int
    col_rounds: int

    def canonical_bytes(self) -> bytes:
        return (
            self.key.canonical_bytes()
            + _u32(self.block_row, "block_row")
            + _u32(self.block_col, "block_col")
            + _u32(self.row_offset, "row_offset")
            + _u32(self.column_offset, "column_offset")
            + _u32(self.rows, "rows")
            + _u32(self.inner_dim, "inner_dim")
            + _u32(self.padded_inner_dim, "padded_inner_dim")
            + _u32(self.cols, "cols")
            + _u32(self.row_rounds, "row_rounds")
            + _u32(self.inner_rounds, "inner_rounds")
            + _u32(self.col_rounds, "col_rounds")
        )


@dataclass(frozen=True)
class ProofBlockDescriptorV2:
    """Statement metadata carried by one serialized block proof."""

    key: OperationKeyV2
    block_row: int
    block_col: int
    row_offset: int
    column_offset: int
    rows: int
    inner_dim: int
    padded_inner_dim: int
    cols: int
    row_rounds: int
    inner_rounds: int
    col_rounds: int

    def as_challenge(self) -> BlockChallengeV2:
        return BlockChallengeV2(
            key=self.key,
            block_row=self.block_row,
            block_col=self.block_col,
            row_offset=self.row_offset,
            column_offset=self.column_offset,
            rows=self.rows,
            inner_dim=self.inner_dim,
            padded_inner_dim=self.padded_inner_dim,
            cols=self.cols,
            row_rounds=self.row_rounds,
            inner_rounds=self.inner_rounds,
            col_rounds=self.col_rounds,
        )


class _Sampler:
    def __init__(self, seed: bytes):
        self._seed = _fixed32(seed, "challenge seed")
        self._counter = 0

    def _word(self) -> int:
        digest = hashlib.sha256(
            _DOMAIN + self._seed + struct.pack("<Q", self._counter)
        ).digest()
        self._counter += 1
        return int.from_bytes(digest, "little")

    def below(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("sampling bound must be positive")
        limit = (1 << 256) - ((1 << 256) % bound)
        while True:
            candidate = self._word()
            if candidate < limit:
                return candidate % bound

    def choose(self, population: Sequence, count: int) -> list:
        if count < 0 or count > len(population):
            raise ValueError("sample count is out of range")
        remaining = list(population)
        selected = []
        for _ in range(count):
            selected.append(remaining.pop(self.below(len(remaining))))
        return selected

    def choose_weighted(
        self, population: Sequence, weights: Sequence[int], count: int
    ) -> list:
        if len(population) != len(weights):
            raise ValueError("weighted population length mismatch")
        if count < 0 or count > len(population):
            raise ValueError("sample count is out of range")
        remaining = list(zip(population, weights))
        if any(isinstance(weight, bool) or weight <= 0 for _, weight in remaining):
            raise ValueError("sampling weights must be positive integers")
        selected = []
        for _ in range(count):
            target = self.below(sum(weight for _, weight in remaining))
            cumulative = 0
            for index, (item, weight) in enumerate(remaining):
                cumulative += weight
                if target < cumulative:
                    selected.append(item)
                    remaining.pop(index)
                    break
        return selected


def derive_stratified_execution_layers_v2(
    *,
    transcript_state: bytes,
    layer_attention_profiles: Sequence[str],
    hard_layer_count: int,
    min_full_attention_layers: int,
    min_gdn_layers: int,
) -> tuple[int, ...]:
    """Derive an exact signed-policy hard-audit layer set.

    The pre-challenge trace is already folded into ``transcript_state``.  This
    helper only consumes the authority-authenticated layer profile and coverage
    policy, so validators cannot quietly lower hard-audit transition coverage
    through a local configuration change.
    """

    state = _fixed32(transcript_state, "transcript_state")
    try:
        profiles = tuple(layer_attention_profiles)
    except TypeError as exc:
        raise ValueError("layer attention profiles must be a sequence") from exc
    if not profiles:
        raise ValueError("layer attention profiles must not be empty")
    if (
        isinstance(hard_layer_count, bool)
        or not isinstance(hard_layer_count, int)
        or hard_layer_count <= 0
    ):
        raise ValueError("hard_layer_count must be a positive integer")
    for name, value in (
        ("min_full_attention_layers", min_full_attention_layers),
        ("min_gdn_layers", min_gdn_layers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if hard_layer_count < min_full_attention_layers + min_gdn_layers:
        raise ValueError("hard layer count is smaller than transition coverage")

    full = tuple(
        index
        for index, profile in enumerate(profiles)
        if profile == "full_attention_transition_v1"
    )
    gdn = tuple(
        index
        for index, profile in enumerate(profiles)
        if profile == "gdn_attention_transition_v1"
    )
    target = min(hard_layer_count, len(profiles))
    required_full = min(min_full_attention_layers, len(full), target)
    required_gdn = min(min_gdn_layers, len(gdn), target - required_full)

    encoded = bytearray(b"VERATHOS/PROOF_V2/STRATIFIED_EXECUTION_LAYERS/SHA256")
    encoded.extend(state)
    encoded.extend(_u32(target, "hard_layer_count"))
    encoded.extend(_u32(min_full_attention_layers, "min_full_attention_layers"))
    encoded.extend(_u32(min_gdn_layers, "min_gdn_layers"))
    encoded.extend(_u32(len(profiles), "execution layer count"))
    for profile in profiles:
        if not isinstance(profile, str):
            raise ValueError("layer attention profile must be text")
        raw = profile.encode("ascii")
        if not raw or len(raw) > 255:
            raise ValueError("layer attention profile is invalid")
        encoded.extend(struct.pack("<B", len(raw)))
        encoded.extend(raw)
    sampler = _Sampler(hashlib.sha256(encoded).digest())
    selected = sampler.choose(full, required_full)
    selected.extend(sampler.choose(gdn, required_gdn))
    remaining = tuple(index for index in range(len(profiles)) if index not in selected)
    selected.extend(sampler.choose(remaining, target - len(selected)))
    return tuple(sorted(selected))


def canonical_axis_segments_v2(
    length: int,
    maximum: int,
) -> tuple[tuple[int, int], ...]:
    """Partition an exact axis into aligned power-of-two segments."""

    if length <= 0 or maximum <= 0 or maximum & (maximum - 1):
        raise ValueError("axis partition parameters are invalid")
    segments = []
    offset = 0
    remaining = length
    while remaining:
        size = min(maximum, 1 << (remaining.bit_length() - 1))
        while offset % size:
            size >>= 1
        segments.append((offset, size))
        offset += size
        remaining -= size
    return tuple(segments)


def _canonical_registered_operations(
    operations: Iterable[RegisteredOperationV2], num_layers: int
) -> tuple[RegisteredOperationV2, ...]:
    canonical = tuple(sorted(operations, key=lambda item: item.key))
    if not canonical or len(canonical) > MAX_REGISTERED_OPERATIONS:
        raise ValueError("registered operation count is out of range")
    keys = [item.key for item in canonical]
    if len(keys) != len(set(keys)):
        raise ValueError("registered operations contain duplicate identities")
    for item in canonical:
        item.validate(num_layers)
    return canonical


def _canonical_x_commitments(
    commitments: Iterable[XCommitmentV2],
) -> tuple[XCommitmentV2, ...]:
    canonical = tuple(sorted(commitments, key=lambda item: item.key))
    keys = [item.key for item in canonical]
    if len(keys) != len(set(keys)):
        raise ValueError("X commitments contain duplicate identities")
    for item in canonical:
        item.canonical_bytes()
        if item.row_count <= 0 or item.inner_dim <= 0:
            raise ValueError("X commitment dimensions must be positive")
        if item.row_count > MAX_X_ROWS_PER_OPERATION:
            raise ValueError("X commitment row count exceeds the protocol limit")
    return canonical


def _canonical_runtime_y_commitments(
    commitments: Iterable[RuntimeYCommitmentV2],
) -> tuple[RuntimeYCommitmentV2, ...]:
    canonical = tuple(sorted(commitments, key=lambda item: item.key))
    keys = [item.key for item in canonical]
    if len(keys) != len(set(keys)):
        raise ValueError("runtime Y commitments contain duplicate identities")
    for item in canonical:
        item.canonical_bytes()
        if item.row_count <= 0 or item.output_dim <= 0:
            raise ValueError("runtime Y commitment dimensions must be positive")
        if item.row_count > MAX_X_ROWS_PER_OPERATION:
            raise ValueError(
                "runtime Y commitment row count exceeds the protocol limit"
            )
        if (
            item.block_rows <= 0
            or item.block_cols <= 0
            or item.block_rows & (item.block_rows - 1)
            or item.block_cols & (item.block_cols - 1)
        ):
            raise ValueError("runtime Y block dimensions must be powers of two")
    return canonical


def derive_block_challenges_v2(
    *,
    transcript_state: bytes,
    num_layers: int,
    operations: Iterable[RegisteredOperationV2],
    x_commitments: Iterable[XCommitmentV2],
    runtime_y_commitments: Iterable[RuntimeYCommitmentV2],
    k_layers: int,
    k_operations_per_layer: int,
    k_blocks_per_operation: int,
    all_operations_per_selected_layer: bool = False,
    required_row_index: int | None = None,
    selected_layer_indices: Sequence[int] | None = None,
) -> tuple[BlockChallengeV2, ...]:
    """Derive the complete expected block set from authenticated metadata."""

    state = _fixed32(transcript_state, "transcript_state")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if k_layers <= 0 or k_operations_per_layer <= 0:
        raise ValueError("layer and operation sample counts must be positive")
    if not 0 < k_blocks_per_operation <= MAX_BLOCKS_PER_OPERATION:
        raise ValueError("block sample count is out of range")
    if type(all_operations_per_selected_layer) is not bool:
        raise ValueError("all-operations audit flag must be boolean")
    if required_row_index is not None:
        if (
            isinstance(required_row_index, bool)
            or not isinstance(required_row_index, int)
            or required_row_index < 0
        ):
            raise ValueError("required transition row index is invalid")
        if not all_operations_per_selected_layer:
            raise ValueError(
                "a required transition row is valid only for an all-operations audit"
            )
    if selected_layer_indices is not None:
        try:
            selected_layer_indices = tuple(selected_layer_indices)
        except TypeError as exc:
            raise ValueError("selected layer indices must be a sequence") from exc
        if (
            not selected_layer_indices
            or selected_layer_indices != tuple(sorted(set(selected_layer_indices)))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < num_layers
                for value in selected_layer_indices
            )
        ):
            raise ValueError("selected layer indices are not canonical")

    registered = _canonical_registered_operations(operations, num_layers)
    committed = _canonical_x_commitments(x_commitments)
    committed_y = _canonical_runtime_y_commitments(runtime_y_commitments)
    registered_by_key = {item.key: item for item in registered}
    committed_by_key = {item.key: item for item in committed}
    committed_y_by_key = {item.key: item for item in committed_y}
    if set(registered_by_key) != set(committed_by_key):
        raise ValueError("X commitment set does not match the registered operation set")
    if set(registered_by_key) != set(committed_y_by_key):
        raise ValueError(
            "runtime Y commitment set does not match the registered operation set"
        )
    for key, operation in registered_by_key.items():
        if committed_by_key[key].inner_dim != operation.inner_dim:
            raise ValueError(
                "X commitment inner dimension does not match its operation"
            )
        y_item = committed_y_by_key[key]
        if (
            y_item.row_count != committed_by_key[key].row_count
            or y_item.output_dim != operation.output_dim
            or y_item.block_rows != RUNTIME_Y_COMMITMENT_BLOCK_ROWS
            or y_item.block_cols != RUNTIME_Y_COMMITMENT_BLOCK_COLS
        ):
            raise ValueError(
                "runtime Y commitment dimensions do not match its operation"
            )

    h = hashlib.sha256()
    h.update(_DOMAIN)
    h.update(state)
    h.update(_u32(PROOF_PROTOCOL_V2, "proof version"))
    h.update(_u32(num_layers, "num_layers"))
    h.update(_u32(len(registered), "registered operation count"))
    for operation in registered:
        encoded = operation.canonical_bytes()
        h.update(struct.pack("<I", len(encoded)))
        h.update(encoded)
    h.update(_u32(len(committed), "X commitment count"))
    for commitment in committed:
        encoded = commitment.canonical_bytes()
        h.update(struct.pack("<I", len(encoded)))
        h.update(encoded)
    h.update(_u32(len(committed_y), "runtime Y commitment count"))
    for commitment in committed_y:
        encoded = commitment.canonical_bytes()
        h.update(struct.pack("<I", len(encoded)))
        h.update(encoded)
    h.update(struct.pack("<?", all_operations_per_selected_layer))
    h.update(struct.pack("<?", required_row_index is not None))
    if required_row_index is not None:
        h.update(_u32(required_row_index, "required transition row index"))
    h.update(struct.pack("<?", selected_layer_indices is not None))
    if selected_layer_indices is not None:
        h.update(_u32(len(selected_layer_indices), "selected layer count"))
        for layer_idx in selected_layer_indices:
            h.update(_u32(layer_idx, "selected layer index"))
    seed = h.digest()
    sampler = _Sampler(seed)

    by_layer: dict[int, list[RegisteredOperationV2]] = {}
    for operation in registered:
        by_layer.setdefault(operation.key.layer_idx, []).append(operation)
    layer_indices = sorted(by_layer)
    if selected_layer_indices is None:
        selected_layers = sampler.choose(
            layer_indices,
            min(k_layers, len(layer_indices)),
        )
    else:
        if len(selected_layer_indices) != min(k_layers, len(layer_indices)):
            raise ValueError("selected layer count does not match the audit policy")
        if any(layer_idx not in by_layer for layer_idx in selected_layer_indices):
            raise ValueError("selected layer is missing registered operations")
        selected_layers = list(selected_layer_indices)
    row_counts = {
        committed_by_key[operation.key].row_count
        for layer_idx in selected_layers
        for operation in by_layer[layer_idx]
    }
    if len(row_counts) != 1:
        raise ValueError("selected operation row counts are not consistent")
    row_count = row_counts.pop()
    if required_row_index is not None:
        if required_row_index >= row_count:
            raise ValueError("required transition row is out of range")
        shared_audit_row = required_row_index
    else:
        shared_audit_row = sampler.choose(range(row_count), 1)[0]

    challenges: list[BlockChallengeV2] = []
    for layer_idx in selected_layers:
        layer_operations = by_layer[layer_idx]
        selected_operations = (
            list(layer_operations)
            if all_operations_per_selected_layer
            else sampler.choose(
                layer_operations,
                min(k_operations_per_layer, len(layer_operations)),
            )
        )
        for operation in selected_operations:
            x_meta = committed_by_key[operation.key]
            row_segments = canonical_axis_segments_v2(
                x_meta.row_count, RUNTIME_Y_COMMITMENT_BLOCK_ROWS
            )
            column_segments = canonical_axis_segments_v2(
                operation.output_dim, operation.block_cols
            )
            blocks = [
                (block_row, block_col, row_offset, column_offset, rows, cols)
                for block_row, (row_offset, rows) in enumerate(row_segments)
                for block_col, (column_offset, cols) in enumerate(column_segments)
                if row_offset <= shared_audit_row < row_offset + rows
            ]
            selected_blocks = sampler.choose_weighted(
                blocks,
                [rows * cols for _, _, _, _, rows, cols in blocks],
                min(k_blocks_per_operation, len(blocks)),
            )
            for (
                block_row,
                block_col,
                row_offset,
                column_offset,
                rows,
                cols,
            ) in selected_blocks:
                padded_inner = 1 << _ceil_log2(operation.inner_dim)
                challenges.append(
                    BlockChallengeV2(
                        key=operation.key,
                        block_row=block_row,
                        block_col=block_col,
                        row_offset=row_offset,
                        column_offset=column_offset,
                        rows=rows,
                        inner_dim=operation.inner_dim,
                        padded_inner_dim=padded_inner,
                        cols=cols,
                        row_rounds=_ceil_log2(rows),
                        inner_rounds=_ceil_log2(padded_inner),
                        col_rounds=_ceil_log2(cols),
                    )
                )
    return tuple(sorted(challenges))


def validate_exact_block_proof_set_v2(
    expected: Sequence[BlockChallengeV2],
    received: Sequence[ProofBlockDescriptorV2],
) -> None:
    """Require one canonical proof descriptor for every expected block."""

    expected_tuple = tuple(expected)
    received_tuple = tuple(item.as_challenge() for item in received)
    if tuple(sorted(expected_tuple)) != expected_tuple:
        raise ValueError("expected block challenge set is not in canonical order")
    if tuple(sorted(received_tuple)) != received_tuple:
        raise ValueError("received block proof set is not in canonical order")
    if len(received_tuple) != len(set(received_tuple)):
        raise ValueError("received block proof set contains duplicates")
    if received_tuple != expected_tuple:
        raise ValueError(
            "received block proof set does not exactly match the challenge set"
        )


def derive_hard_execution_corridor_v2(
    challenges: Iterable[object],
    *,
    num_layers: int,
) -> tuple[int, tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Derive the one-row hard-trace corridor from exact layer challenges.

    A hard audit proves sampled arithmetic only for its selected layer set, but
    its execution trace must open one complete residual corridor.  Both the
    miner and verifier derive that corridor from the already canonical block
    challenge set.  Model-level operations (such as the LM head) deliberately
    do not choose the decode row or transition-layer set.
    """

    if (
        isinstance(num_layers, bool)
        or not isinstance(num_layers, int)
        or num_layers <= 0
    ):
        raise ValueError("hard execution corridor layer count is invalid")
    try:
        canonical = tuple(challenges)
    except TypeError as exc:
        raise ValueError("hard execution challenges must be a sequence") from exc

    selected_layers: set[int] = set()
    selected_rows: set[int] = set()
    for challenge in canonical:
        key = getattr(challenge, "key", None)
        layer_idx = getattr(key, "layer_idx", None)
        if layer_idx == MODEL_OPERATION_LAYER_IDX:
            continue
        if (
            isinstance(layer_idx, bool)
            or not isinstance(layer_idx, int)
            or not 0 <= layer_idx < num_layers
        ):
            raise ValueError("hard execution challenge layer is out of range")
        row_offset = getattr(challenge, "row_offset", None)
        rows = getattr(challenge, "rows", None)
        if (
            isinstance(row_offset, bool)
            or not isinstance(row_offset, int)
            or row_offset < 0
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows != RUNTIME_Y_COMMITMENT_BLOCK_ROWS
        ):
            raise ValueError("hard execution challenge row is not canonical")
        selected_layers.add(layer_idx)
        selected_rows.add(row_offset)

    if not selected_layers:
        raise ValueError("hard execution challenge set has no model layer")
    if len(selected_rows) != 1:
        raise ValueError("hard execution challenge set spans multiple decode rows")
    row = selected_rows.pop()
    layers = tuple(sorted(selected_layers))
    return row, layers, tuple((row, layer_idx) for layer_idx in range(num_layers))
