"""Canonical sampled attention-state transitions for proof protocol v2.

The projection PCS authenticates registered linear operations.  This module
defines the complementary model-state check used by the hard audit: replay a
nonce-selected attention block from a committed state boundary and compare the
result with the committed residual trace.  The first qualified runtime profile
is dense Qwen3.5/Qwen3.6 hybrid attention.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from verallm.crypto.merkle import MerkleTree, verify_merkle_path
from zkllm.types import MerklePath

TRANSITION_DOMAIN_V2 = b"VERATHOS/PROOF_V2/TRANSITION_CHALLENGE/SHA256"
GDN_TRANSITION_PROFILE_V1 = "qwen_gdn_delta_rule_v1"
FULL_ATTENTION_TRANSITION_PROFILE_V1 = "qwen_full_attention_v1"
DEFAULT_TRANSITION_BLOCK_ROWS = 64
MAX_TRANSITION_BLOCK_ROWS = 256
TRANSITION_HISTORY_PROFILE_V1 = "qwen_residual_history_v1"
TRANSITION_STREAM_RESIDUAL_IN = 0
TRANSITION_STREAM_AFTER_ATTENTION = 1
TRANSITION_STREAM_FINAL_RESIDUAL = 2
MAX_TRANSITION_HISTORY_ROWS = 16_384
MAX_TRANSITION_HISTORY_LAYERS = 16_384
MAX_TRANSITION_STREAM_BYTES = 256 << 20
_HISTORY_COMMITMENT_DOMAIN = b"VERATHOS/PROOF_V2/TRANSITION_HISTORY_COMMITMENT/SHA256"
_GDN_PARAMETER_MAGIC = b"GDN2"
_FULL_ATTENTION_PARAMETER_MAGIC = b"FAT2"
_GDN_STATE_DTYPE_CODES = {"f16": 1, "f32": 2, "bf16": 3}


def gdn_state_numpy_dtype_v2(dtype: str) -> np.dtype:
    """Return the canonical little-endian cache dtype for a signed profile."""

    if dtype == "f16":
        return np.dtype("<f2")
    if dtype == "f32":
        return np.dtype("<f4")
    # NumPy has no portable built-in BF16 dtype. Proof-v3 decodes BF16 state
    # through its canonical uint16 representation instead of silently
    # treating the words as integers here.
    raise ProofV2TransitionError("GDN state dtype is not supported")


def _decode_bf16_f32(encoded: bytes) -> np.ndarray:
    words = np.frombuffer(encoded, dtype="<u2").astype(np.uint32)
    return (words << np.uint32(16)).view("<f4")


def _round_bf16_f32(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    words = values.view(np.uint32)
    rounded = words + np.uint32(0x7FFF) + ((words >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _decode_gdn_16bit_parameter(
    encoded: bytes,
    runtime_dtype: str,
) -> np.ndarray:
    if runtime_dtype == "f16":
        return np.frombuffer(encoded, dtype="<f2").astype(np.float32)
    if runtime_dtype == "bf16":
        return _decode_bf16_f32(encoded)
    raise ProofV2TransitionError(
        "GDN transition parameter dtype is not supported"
    )


def gdn_state_dtype_code_v2(dtype: str) -> int:
    try:
        return _GDN_STATE_DTYPE_CODES[dtype]
    except KeyError as exc:
        raise ProofV2TransitionError("GDN state dtype is not supported") from exc


def gdn_state_dtype_from_code_v2(code: int) -> str:
    for dtype, known_code in _GDN_STATE_DTYPE_CODES.items():
        if code == known_code:
            return dtype
    raise ProofV2TransitionError("GDN state dtype code is not supported")


class ProofV2TransitionError(ValueError):
    """A transition challenge or canonical replay input is invalid."""


def _fixed32(value: bytes, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV2TransitionError(f"{name} must be exactly 32 bytes")
    return value


def _u32(value: int, name: str) -> bytes:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < (1 << 32)
    ):
        raise ProofV2TransitionError(f"{name} must be an unsigned 32-bit integer")
    return struct.pack("<I", value)


class _ChallengeStream:
    def __init__(self, seed: bytes):
        self.seed = _fixed32(seed, "transition challenge seed")
        self.counter = 0

    def _word(self) -> int:
        encoded = hashlib.sha256(
            self.seed + self.counter.to_bytes(8, "little")
        ).digest()
        self.counter += 1
        return int.from_bytes(encoded[:8], "little")

    def below(self, limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ProofV2TransitionError("transition sampling limit is invalid")
        ceiling = (1 << 64) - ((1 << 64) % limit)
        while True:
            candidate = self._word()
            if candidate < ceiling:
                return candidate % limit

    def choose(self, values: Sequence[int], count: int) -> tuple[int, ...]:
        remaining = list(values)
        if not 0 <= count <= len(remaining):
            raise ProofV2TransitionError("transition sample count is invalid")
        selected = []
        for _ in range(count):
            selected.append(remaining.pop(self.below(len(remaining))))
        return tuple(sorted(selected))


@dataclass(frozen=True, order=True)
class TransitionChallengeV2:
    """One verifier-owned state block and head selection."""

    layer_idx: int
    block_index: int
    row_offset: int
    rows: int
    heads: tuple[int, ...]

    def __post_init__(self) -> None:
        _u32(self.layer_idx, "transition layer")
        _u32(self.block_index, "transition block")
        _u32(self.row_offset, "transition row offset")
        _u32(self.rows, "transition row count")
        if not 0 < self.rows <= MAX_TRANSITION_BLOCK_ROWS:
            raise ProofV2TransitionError("transition row count is out of range")
        try:
            heads = tuple(self.heads)
        except TypeError as exc:
            raise ProofV2TransitionError("transition heads must be a sequence") from exc
        if not heads or heads != tuple(sorted(set(heads))):
            raise ProofV2TransitionError(
                "transition heads must be non-empty, sorted, and unique"
            )
        for head in heads:
            _u32(head, "transition head")
        object.__setattr__(self, "heads", heads)


def derive_transition_challenges_v2(
    *,
    transcript_state: bytes,
    trace_commitment_digest: bytes,
    execution_row_count: int,
    layer_head_counts: Sequence[int],
    selected_layers: Sequence[int],
    block_rows: int = DEFAULT_TRANSITION_BLOCK_ROWS,
    heads_per_layer: int = 1,
) -> tuple[TransitionChallengeV2, ...]:
    """Derive exact block/head openings after the trace has been frozen."""

    transcript = _fixed32(transcript_state, "transition transcript state")
    trace = _fixed32(trace_commitment_digest, "trace commitment digest")
    _u32(execution_row_count, "execution row count")
    if execution_row_count == 0:
        raise ProofV2TransitionError("execution row count must be positive")
    if (
        isinstance(block_rows, bool)
        or not isinstance(block_rows, int)
        or not 0 < block_rows <= MAX_TRANSITION_BLOCK_ROWS
        or block_rows & (block_rows - 1)
    ):
        raise ProofV2TransitionError(
            "transition block row count must be a bounded power of two"
        )
    if (
        isinstance(heads_per_layer, bool)
        or not isinstance(heads_per_layer, int)
        or heads_per_layer <= 0
    ):
        raise ProofV2TransitionError("heads per layer must be positive")
    try:
        head_counts = tuple(layer_head_counts)
        layers = tuple(selected_layers)
    except TypeError as exc:
        raise ProofV2TransitionError(
            "transition layer metadata must be sequences"
        ) from exc
    if not head_counts or layers != tuple(sorted(set(layers))) or not layers:
        raise ProofV2TransitionError("selected transition layers are not canonical")
    for count in head_counts:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ProofV2TransitionError("transition head count is invalid")
    for layer in layers:
        if not 0 <= layer < len(head_counts):
            raise ProofV2TransitionError("transition layer is out of range")

    encoded = bytearray(TRANSITION_DOMAIN_V2)
    encoded.extend(transcript)
    encoded.extend(trace)
    encoded.extend(_u32(execution_row_count, "execution row count"))
    encoded.extend(_u32(block_rows, "transition block rows"))
    encoded.extend(_u32(heads_per_layer, "heads per layer"))
    encoded.extend(_u32(len(head_counts), "layer count"))
    for count in head_counts:
        encoded.extend(_u32(count, "transition head count"))
    encoded.extend(_u32(len(layers), "selected layer count"))
    for layer in layers:
        encoded.extend(_u32(layer, "selected transition layer"))
    stream = _ChallengeStream(hashlib.sha256(encoded).digest())

    block_count = (execution_row_count + block_rows - 1) // block_rows
    result = []
    for layer in layers:
        block_index = stream.below(block_count)
        row_offset = block_index * block_rows
        rows = min(block_rows, execution_row_count - row_offset)
        count = min(heads_per_layer, head_counts[layer])
        heads = stream.choose(tuple(range(head_counts[layer])), count)
        result.append(
            TransitionChallengeV2(
                layer_idx=layer,
                block_index=block_index,
                row_offset=row_offset,
                rows=rows,
                heads=heads,
            )
        )
    return tuple(result)


@dataclass(frozen=True, order=True)
class TransitionHistoryStreamV2:
    """One pre-challenge fp16 residual-history stream leaf."""

    layer_idx: int
    stream_kind: int
    row_count: int
    hidden_dim: int
    row_root: bytes

    def __post_init__(self) -> None:
        _u32(self.layer_idx, "transition history layer")
        _u32(self.row_count, "transition history row count")
        _u32(self.hidden_dim, "transition history hidden dimension")
        if (
            self.stream_kind
            not in (
                TRANSITION_STREAM_RESIDUAL_IN,
                TRANSITION_STREAM_AFTER_ATTENTION,
                TRANSITION_STREAM_FINAL_RESIDUAL,
            )
            or self.row_count == 0
            or self.row_count > MAX_TRANSITION_HISTORY_ROWS
            or self.hidden_dim == 0
        ):
            raise ProofV2TransitionError(
                "transition residual stream metadata is invalid"
            )
        _fixed32(self.row_root, "transition residual row root")

    def canonical_bytes(self) -> bytes:
        return struct.pack(
            "<IBII",
            self.layer_idx,
            self.stream_kind,
            self.row_count,
            self.hidden_dim,
        ) + _fixed32(self.row_root, "transition residual row root")


@dataclass(frozen=True)
class TransitionHistoryCommitmentV2:
    """Exact residual histories frozen before validator challenge entropy."""

    profile: str
    row_count: int
    num_layers: int
    hidden_dim: int
    stream_root: bytes

    def __post_init__(self) -> None:
        if self.profile != TRANSITION_HISTORY_PROFILE_V1:
            raise ProofV2TransitionError("transition history profile is not supported")
        _u32(self.row_count, "transition history row count")
        _u32(self.num_layers, "transition history layer count")
        _u32(self.hidden_dim, "transition history hidden dimension")
        if (
            self.row_count == 0
            or self.row_count > MAX_TRANSITION_HISTORY_ROWS
            or self.num_layers == 0
            or self.num_layers > MAX_TRANSITION_HISTORY_LAYERS
            or self.hidden_dim == 0
        ):
            raise ProofV2TransitionError(
                "transition history commitment metadata is invalid"
            )
        _fixed32(self.stream_root, "transition history stream root")

    def canonical_bytes(self) -> bytes:
        encoded_profile = self.profile.encode("ascii")
        return (
            struct.pack(
                "<BIII",
                len(encoded_profile),
                self.row_count,
                self.num_layers,
                self.hidden_dim,
            )
            + encoded_profile
            + self.stream_root
        )

    def digest(self) -> bytes:
        return hashlib.sha256(
            _HISTORY_COMMITMENT_DOMAIN + self.canonical_bytes()
        ).digest()

    @classmethod
    def from_canonical_bytes(
        cls,
        encoded: bytes,
    ) -> "TransitionHistoryCommitmentV2":
        if not isinstance(encoded, bytes) or len(encoded) < 1 + 12 + 32:
            raise ProofV2TransitionError(
                "transition history commitment encoding is malformed"
            )
        profile_length, row_count, num_layers, hidden_dim = struct.unpack_from(
            "<BIII",
            encoded,
        )
        expected = 1 + 12 + profile_length + 32
        if profile_length == 0 or len(encoded) != expected:
            raise ProofV2TransitionError(
                "transition history commitment encoding is malformed"
            )
        try:
            profile = encoded[13 : 13 + profile_length].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ProofV2TransitionError(
                "transition history profile is not ASCII"
            ) from exc
        result = cls(
            profile,
            row_count,
            num_layers,
            hidden_dim,
            encoded[-32:],
        )
        if result.canonical_bytes() != encoded:
            raise ProofV2TransitionError(
                "transition history commitment is not canonical"
            )
        return result


@dataclass(frozen=True)
class TransitionHistoryOpeningV2:
    """Exact residual rows with inner and request-level membership paths."""

    stream: TransitionHistoryStreamV2
    row_indices: tuple[int, ...]
    row_values_f16: bytes
    row_paths: tuple[MerklePath, ...]
    stream_path: MerklePath

    def __post_init__(self) -> None:
        if not isinstance(self.stream, TransitionHistoryStreamV2):
            raise ProofV2TransitionError("transition history opening stream is invalid")
        try:
            indices = tuple(self.row_indices)
            paths = tuple(self.row_paths)
        except TypeError as exc:
            raise ProofV2TransitionError(
                "transition history row opening set is invalid"
            ) from exc
        if (
            not indices
            or indices != tuple(sorted(set(indices)))
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < self.stream.row_count
                for index in indices
            )
            or len(paths) != len(indices)
            or not all(isinstance(path, MerklePath) for path in paths)
        ):
            raise ProofV2TransitionError(
                "transition history row opening set is invalid"
            )
        expected = len(indices) * self.stream.hidden_dim * 2
        if (
            not isinstance(self.row_values_f16, bytes)
            or len(self.row_values_f16) != expected
            or len(self.row_values_f16) > MAX_TRANSITION_STREAM_BYTES
        ):
            raise ProofV2TransitionError("transition history row values are malformed")
        if not isinstance(self.stream_path, MerklePath):
            raise ProofV2TransitionError("transition history opening path is invalid")
        object.__setattr__(self, "row_indices", indices)
        object.__setattr__(self, "row_paths", paths)

    def verify(self, commitment: TransitionHistoryCommitmentV2) -> bool:
        if not isinstance(commitment, TransitionHistoryCommitmentV2):
            raise ProofV2TransitionError("transition history commitment is invalid")
        if (
            self.stream.row_count != commitment.row_count
            or self.stream.hidden_dim != commitment.hidden_dim
            or self.stream.layer_idx > commitment.num_layers
        ):
            return False
        expected_index = transition_history_stream_index_v2(
            num_layers=commitment.num_layers,
            layer_idx=self.stream.layer_idx,
            stream_kind=self.stream.stream_kind,
        )
        if self.stream_path.leaf_index != expected_index:
            return False
        if not verify_merkle_path(
            commitment.stream_root,
            self.stream.canonical_bytes(),
            self.stream_path,
        ):
            return False
        row_width = self.stream.hidden_dim * 2
        for item_index, (row_index, path) in enumerate(
            zip(self.row_indices, self.row_paths)
        ):
            if path.leaf_index != row_index:
                return False
            start = item_index * row_width
            if not verify_merkle_path(
                self.stream.row_root,
                self.row_values_f16[start : start + row_width],
                path,
            ):
                return False
        return True

    def rows(self) -> np.ndarray:
        return np.frombuffer(self.row_values_f16, dtype="<f2").reshape(
            len(self.row_indices),
            self.stream.hidden_dim,
        )


@dataclass(frozen=True)
class TransitionHistoryStateV2:
    commitment: TransitionHistoryCommitmentV2
    streams: tuple[TransitionHistoryStreamV2, ...]
    values_f16: tuple[bytes, ...]
    tree: MerkleTree
    row_trees: tuple[MerkleTree, ...]

    def opening(
        self,
        *,
        layer_idx: int,
        stream_kind: int,
        row_indices: Sequence[int] | None = None,
    ) -> TransitionHistoryOpeningV2:
        index = transition_history_stream_index_v2(
            num_layers=self.commitment.num_layers,
            layer_idx=layer_idx,
            stream_kind=stream_kind,
        )
        stream = self.streams[index]
        if row_indices is None:
            rows = tuple(range(stream.row_count))
        else:
            try:
                rows = tuple(row_indices)
            except TypeError as exc:
                raise ProofV2TransitionError(
                    "transition history row opening set is invalid"
                ) from exc
        row_width = stream.hidden_dim * 2
        values = self.values_f16[index]
        return TransitionHistoryOpeningV2(
            stream,
            rows,
            b"".join(values[row * row_width : (row + 1) * row_width] for row in rows),
            tuple(self.row_trees[index].get_path(row) for row in rows),
            self.tree.get_path(index),
        )


def transition_history_stream_index_v2(
    *,
    num_layers: int,
    layer_idx: int,
    stream_kind: int,
) -> int:
    _u32(num_layers, "transition history layer count")
    _u32(layer_idx, "transition history layer")
    if num_layers == 0 or num_layers > MAX_TRANSITION_HISTORY_LAYERS:
        raise ProofV2TransitionError("transition history layer count is invalid")
    if stream_kind == TRANSITION_STREAM_RESIDUAL_IN and layer_idx < num_layers:
        return 2 * layer_idx
    if stream_kind == TRANSITION_STREAM_AFTER_ATTENTION and layer_idx < num_layers:
        return 2 * layer_idx + 1
    if stream_kind == TRANSITION_STREAM_FINAL_RESIDUAL and layer_idx == num_layers:
        return 2 * num_layers
    raise ProofV2TransitionError("transition history stream position is not canonical")


def _canonical_residual_history_values_v2(
    value: np.ndarray,
    *,
    row_count: int | None,
    hidden_dim: int | None,
) -> tuple[bytes, int, int]:
    array = np.asarray(value)
    if (
        array.ndim != 2
        or array.dtype != np.dtype("<f2")
        or array.shape[0] == 0
        or array.shape[0] > MAX_TRANSITION_HISTORY_ROWS
        or array.shape[1] == 0
        or not np.isfinite(array).all()
    ):
        raise ProofV2TransitionError(
            "transition residual history must be a finite fp16 matrix"
        )
    rows, width = map(int, array.shape)
    if (row_count is not None and rows != row_count) or (
        hidden_dim is not None and width != hidden_dim
    ):
        raise ProofV2TransitionError(
            "transition residual history dimensions are inconsistent"
        )
    values = np.ascontiguousarray(array, dtype="<f2").tobytes()
    if len(values) > MAX_TRANSITION_STREAM_BYTES:
        raise ProofV2TransitionError(
            "transition residual history exceeds the protocol limit"
        )
    return values, rows, width


def build_transition_history_commitment_v2(
    *,
    residual_inputs: Sequence[np.ndarray],
    residuals_after_attention: Sequence[np.ndarray],
    final_residual: np.ndarray,
) -> TransitionHistoryStateV2:
    """Commit every layer boundary before nonce-selected transition audits."""

    try:
        inputs = tuple(residual_inputs)
        attention = tuple(residuals_after_attention)
    except TypeError as exc:
        raise ProofV2TransitionError(
            "transition residual histories must be sequences"
        ) from exc
    if (
        not inputs
        or len(inputs) != len(attention)
        or len(inputs) > MAX_TRANSITION_HISTORY_LAYERS
    ):
        raise ProofV2TransitionError(
            "transition residual history layer set is not exact"
        )
    rows = None
    width = None
    streams = []
    values = []
    row_trees = []
    for layer_idx, (residual_in, after_attention) in enumerate(zip(inputs, attention)):
        for stream_kind, matrix in (
            (TRANSITION_STREAM_RESIDUAL_IN, residual_in),
            (TRANSITION_STREAM_AFTER_ATTENTION, after_attention),
        ):
            encoded, rows, width = _canonical_residual_history_values_v2(
                matrix,
                row_count=rows,
                hidden_dim=width,
            )
            values.append(encoded)
            row_width = width * 2
            row_tree = MerkleTree(
                [
                    encoded[row * row_width : (row + 1) * row_width]
                    for row in range(rows)
                ]
            )
            row_trees.append(row_tree)
            streams.append(
                TransitionHistoryStreamV2(
                    layer_idx,
                    stream_kind,
                    rows,
                    width,
                    row_tree.root,
                )
            )
    final_values, rows, width = _canonical_residual_history_values_v2(
        final_residual,
        row_count=rows,
        hidden_dim=width,
    )
    values.append(final_values)
    final_row_width = width * 2
    final_row_tree = MerkleTree(
        [
            final_values[row * final_row_width : (row + 1) * final_row_width]
            for row in range(rows)
        ]
    )
    row_trees.append(final_row_tree)
    streams.append(
        TransitionHistoryStreamV2(
            len(inputs),
            TRANSITION_STREAM_FINAL_RESIDUAL,
            rows,
            width,
            final_row_tree.root,
        )
    )
    tree = MerkleTree([item.canonical_bytes() for item in streams])
    commitment = TransitionHistoryCommitmentV2(
        TRANSITION_HISTORY_PROFILE_V1,
        rows,
        len(inputs),
        width,
        tree.root,
    )
    return TransitionHistoryStateV2(
        commitment,
        tuple(streams),
        tuple(values),
        tree,
        tuple(row_trees),
    )


@dataclass(frozen=True)
class GDNTransitionParametersV2:
    """Canonical signed non-projection parameters for one Qwen GDN layer."""

    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_size: int
    rms_epsilon_q32: int
    conv_weight_f16: bytes
    a_log_f32: bytes
    dt_bias_f16: bytes
    norm_weight_f16: bytes
    runtime_dtype: str = "f16"
    recurrent_state_dtype: str = "f16"

    def __post_init__(self) -> None:
        dimensions = (
            self.num_key_heads,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
            self.conv_kernel_size,
        )
        for value in dimensions:
            encoded = _u32(value, "GDN transition dimension")
            if encoded == b"\x00" * 4:
                raise ProofV2TransitionError(
                    "GDN transition dimensions must be positive"
                )
        if self.num_value_heads % self.num_key_heads:
            raise ProofV2TransitionError(
                "GDN value-head count must be divisible by key-head count"
            )
        if (
            isinstance(self.rms_epsilon_q32, bool)
            or not isinstance(self.rms_epsilon_q32, int)
            or not 0 < self.rms_epsilon_q32 < (1 << 64)
            or self.runtime_dtype not in {"f16", "bf16"}
            or self.recurrent_state_dtype not in _GDN_STATE_DTYPE_CODES
        ):
            raise ProofV2TransitionError("GDN transition numeric profile is invalid")
        conv_dim = (
            2 * self.num_key_heads * self.key_head_dim
            + self.num_value_heads * self.value_head_dim
        )
        expected_lengths = (
            (self.conv_weight_f16, conv_dim * self.conv_kernel_size * 2),
            (self.a_log_f32, self.num_value_heads * 4),
            (self.dt_bias_f16, self.num_value_heads * 2),
            (self.norm_weight_f16, self.value_head_dim * 2),
        )
        for value, length in expected_lengths:
            if not isinstance(value, bytes) or len(value) != length:
                raise ProofV2TransitionError(
                    "GDN transition parameter length is not canonical"
                )
        arrays = (
            _decode_gdn_16bit_parameter(
                self.conv_weight_f16,
                self.runtime_dtype,
            ),
            np.frombuffer(self.a_log_f32, dtype="<f4"),
            _decode_gdn_16bit_parameter(
                self.dt_bias_f16,
                self.runtime_dtype,
            ),
            _decode_gdn_16bit_parameter(
                self.norm_weight_f16,
                self.runtime_dtype,
            ),
        )
        if not all(np.isfinite(value).all() for value in arrays):
            raise ProofV2TransitionError(
                "GDN transition parameters contain non-finite values"
            )

    def canonical_bytes(self) -> bytes:
        return (
            struct.pack(
                "<4s5IQBB",
                _GDN_PARAMETER_MAGIC,
                self.num_key_heads,
                self.num_value_heads,
                self.key_head_dim,
                self.value_head_dim,
                self.conv_kernel_size,
                self.rms_epsilon_q32,
                gdn_state_dtype_code_v2(self.runtime_dtype),
                gdn_state_dtype_code_v2(self.recurrent_state_dtype),
            )
            + self.conv_weight_f16
            + self.a_log_f32
            + self.dt_bias_f16
            + self.norm_weight_f16
        )

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "GDNTransitionParametersV2":
        header_size = struct.calcsize("<4s5IQBB")
        if not isinstance(encoded, bytes) or len(encoded) < header_size:
            raise ProofV2TransitionError(
                "GDN transition parameter encoding is malformed"
            )
        magic = encoded[:4]
        if magic == _GDN_PARAMETER_MAGIC:
            (
                _magic,
                num_key_heads,
                num_value_heads,
                key_head_dim,
                value_head_dim,
                conv_kernel_size,
                rms_epsilon_q32,
                runtime_dtype_code,
                recurrent_state_dtype_code,
            ) = struct.unpack_from("<4s5IQBB", encoded)
            parameter_offset = header_size
        else:
            raise ProofV2TransitionError(
                "GDN transition parameter header is unsupported"
            )
        try:
            runtime_dtype = gdn_state_dtype_from_code_v2(runtime_dtype_code)
            recurrent_state_dtype = gdn_state_dtype_from_code_v2(
                recurrent_state_dtype_code
            )
        except ProofV2TransitionError as exc:
            raise ProofV2TransitionError(
                "GDN transition parameter header is unsupported"
            ) from exc
        if runtime_dtype not in {"f16", "bf16"}:
            raise ProofV2TransitionError(
                "GDN transition parameter header is unsupported"
            )
        conv_dim = 2 * num_key_heads * key_head_dim + num_value_heads * value_head_dim
        lengths = (
            conv_dim * conv_kernel_size * 2,
            num_value_heads * 4,
            num_value_heads * 2,
            value_head_dim * 2,
        )
        if len(encoded) != parameter_offset + sum(lengths):
            raise ProofV2TransitionError(
                "GDN transition parameter encoding length is invalid"
            )
        offset = parameter_offset
        values = []
        for length in lengths:
            values.append(encoded[offset : offset + length])
            offset += length
        result = cls(
            num_key_heads,
            num_value_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_size,
            rms_epsilon_q32,
            *values,
            runtime_dtype,
            recurrent_state_dtype,
        )
        if result.canonical_bytes() != encoded:
            raise ProofV2TransitionError("GDN transition parameters are not canonical")
        return result

    def replay_parameters(self) -> "GDNReplayParametersV2":
        conv_dim = (
            2 * self.num_key_heads * self.key_head_dim
            + self.num_value_heads * self.value_head_dim
        )
        return GDNReplayParametersV2(
            self.num_key_heads,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
            self.conv_kernel_size,
            self.rms_epsilon_q32 / float(1 << 32),
            _decode_gdn_16bit_parameter(
                self.conv_weight_f16,
                self.runtime_dtype,
            )
            .reshape(conv_dim, self.conv_kernel_size),
            np.frombuffer(self.a_log_f32, dtype="<f4").copy(),
            _decode_gdn_16bit_parameter(
                self.dt_bias_f16,
                self.runtime_dtype,
            ),
            _decode_gdn_16bit_parameter(
                self.norm_weight_f16,
                self.runtime_dtype,
            ),
            self.runtime_dtype,
            self.recurrent_state_dtype,
        )


@dataclass(frozen=True)
class GDNReplayParametersV2:
    """Signed Qwen GDN dimensions and non-projection parameters."""

    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_size: int
    rms_epsilon: float
    conv_weight: np.ndarray
    a_log: np.ndarray
    dt_bias: np.ndarray
    norm_weight: np.ndarray
    runtime_dtype: str = "f16"
    recurrent_state_dtype: str = "f16"

    def __post_init__(self) -> None:
        integers = (
            self.num_key_heads,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
            self.conv_kernel_size,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integers
        ):
            raise ProofV2TransitionError("GDN replay dimensions are invalid")
        if self.num_value_heads % self.num_key_heads:
            raise ProofV2TransitionError(
                "GDN value-head count must be divisible by key-head count"
            )
        if not math.isfinite(self.rms_epsilon) or self.rms_epsilon <= 0:
            raise ProofV2TransitionError("GDN RMS epsilon is invalid")
        if (
            self.runtime_dtype not in {"f16", "bf16"}
            or self.recurrent_state_dtype not in _GDN_STATE_DTYPE_CODES
        ):
            raise ProofV2TransitionError("GDN replay runtime dtype is not supported")
        conv_dim = (
            2 * self.num_key_heads * self.key_head_dim
            + self.num_value_heads * self.value_head_dim
        )
        arrays = {
            "conv_weight": (self.conv_weight, (conv_dim, self.conv_kernel_size)),
            "a_log": (self.a_log, (self.num_value_heads,)),
            "dt_bias": (self.dt_bias, (self.num_value_heads,)),
            "norm_weight": (self.norm_weight, (self.value_head_dim,)),
        }
        for name, (value, shape) in arrays.items():
            array = np.asarray(value)
            if array.shape != shape or not np.isfinite(array).all():
                raise ProofV2TransitionError(f"GDN {name} shape or values are invalid")


@dataclass(frozen=True)
class GDNReplayResultV2:
    conv_state_after: np.ndarray
    recurrent_state_after: np.ndarray
    core_output: np.ndarray
    out_projection_input: np.ndarray


@dataclass(frozen=True)
class FullAttentionTransitionParametersV2:
    """Signed logical full-attention ABI for one Qwen decoder layer.

    The witness is deliberately expressed in logical Q/K/V-head order rather
    than in a backend-specific paged-cache layout.  The miner adapter is
    responsible for deriving that order from vLLM metadata; the verifier only
    needs this small signed description and the nonce-selected head witness.
    ``q_norm_weight_f16``/``k_norm_weight_f16`` are optional for checkpoints
    without Q/K RMSNorm and, when present, are exact per-head vectors.
    """

    num_query_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta_q32: int
    rms_epsilon_q32: int
    q_norm_weight_f16: bytes = b""
    k_norm_weight_f16: bytes = b""
    runtime_dtype: str = "f16"

    def __post_init__(self) -> None:
        dimensions = (
            self.num_query_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.rotary_dim,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions
        ):
            raise ProofV2TransitionError("full-attention dimensions are invalid")
        if (
            self.num_query_heads % self.num_key_value_heads
            or self.rotary_dim > self.head_dim
            or self.rotary_dim % 2
            or self.runtime_dtype != "f16"
            or isinstance(self.rope_theta_q32, bool)
            or not isinstance(self.rope_theta_q32, int)
            or self.rope_theta_q32 <= 0
            or isinstance(self.rms_epsilon_q32, bool)
            or not isinstance(self.rms_epsilon_q32, int)
            or self.rms_epsilon_q32 <= 0
        ):
            raise ProofV2TransitionError("full-attention numeric profile is invalid")
        for value, name in (
            (self.q_norm_weight_f16, "full-attention Q norm weight"),
            (self.k_norm_weight_f16, "full-attention K norm weight"),
        ):
            if value and (
                not isinstance(value, bytes)
                or len(value) != self.head_dim * 2
                or not np.isfinite(np.frombuffer(value, dtype="<f2")).all()
            ):
                raise ProofV2TransitionError(f"{name} is not canonical")
        if bool(self.q_norm_weight_f16) != bool(self.k_norm_weight_f16):
            raise ProofV2TransitionError(
                "full-attention Q/K norm weights must be present together"
            )

    def canonical_bytes(self) -> bytes:
        q_weight = bytes(self.q_norm_weight_f16)
        k_weight = bytes(self.k_norm_weight_f16)
        return (
            struct.pack(
                "<4s4I2QB",
                _FULL_ATTENTION_PARAMETER_MAGIC,
                self.num_query_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.rotary_dim,
                self.rope_theta_q32,
                self.rms_epsilon_q32,
                gdn_state_dtype_code_v2(self.runtime_dtype),
            )
            + struct.pack("<I", len(q_weight))
            + q_weight
            + struct.pack("<I", len(k_weight))
            + k_weight
        )

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "FullAttentionTransitionParametersV2":
        header_size = struct.calcsize("<4s4I2QB")
        if not isinstance(encoded, bytes) or len(encoded) < header_size + 8:
            raise ProofV2TransitionError(
                "full-attention transition parameter encoding is malformed"
            )
        (
            magic,
            num_query_heads,
            num_key_value_heads,
            head_dim,
            rotary_dim,
            rope_theta_q32,
            rms_epsilon_q32,
            runtime_dtype_code,
        ) = struct.unpack_from("<4s4I2QB", encoded)
        if magic != _FULL_ATTENTION_PARAMETER_MAGIC:
            raise ProofV2TransitionError(
                "full-attention transition parameter header is unsupported"
            )
        try:
            runtime_dtype = gdn_state_dtype_from_code_v2(runtime_dtype_code)
        except ProofV2TransitionError as exc:
            raise ProofV2TransitionError(
                "full-attention transition parameter header is unsupported"
            ) from exc
        offset = header_size
        q_length = struct.unpack_from("<I", encoded, offset)[0]
        offset += 4
        if q_length > (1 << 20) or offset + q_length + 4 > len(encoded):
            raise ProofV2TransitionError(
                "full-attention transition parameter encoding is malformed"
            )
        q_weight = encoded[offset : offset + q_length]
        offset += q_length
        k_length = struct.unpack_from("<I", encoded, offset)[0]
        offset += 4
        if k_length > (1 << 20) or offset + k_length != len(encoded):
            raise ProofV2TransitionError(
                "full-attention transition parameter encoding is malformed"
            )
        result = cls(
            num_query_heads,
            num_key_value_heads,
            head_dim,
            rotary_dim,
            rope_theta_q32,
            rms_epsilon_q32,
            q_weight,
            encoded[offset : offset + k_length],
            runtime_dtype,
        )
        if result.canonical_bytes() != encoded:
            raise ProofV2TransitionError(
                "full-attention transition parameters are not canonical"
            )
        return result

    @property
    def q_width(self) -> int:
        return self.num_query_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def qkv_width(self) -> int:
        return self.q_width + 2 * self.kv_width


def _full_attention_rmsnorm_v2(
    values: np.ndarray,
    weight_bytes: bytes,
    epsilon_q32: int,
) -> np.ndarray:
    """Replay the Qwen per-head Q/K RMSNorm in float32."""

    values = np.asarray(values, dtype=np.float32)
    if not weight_bytes:
        return values
    weight = np.frombuffer(weight_bytes, dtype="<f2").astype(np.float32)
    if values.shape[-1] != weight.shape[0]:
        raise ProofV2TransitionError("full-attention RMSNorm width is invalid")
    variance = np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
    return values * np.reciprocal(
        np.sqrt(variance + np.float32(epsilon_q32 / float(1 << 32))),
        dtype=np.float32,
    ) * weight


def _qwen_rope_v2(
    values: np.ndarray,
    *,
    position: int,
    rotary_dim: int,
    rope_theta_q32: int,
) -> np.ndarray:
    """Apply the canonical non-interleaved Qwen RoPE to one head vector."""

    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise ProofV2TransitionError("full-attention position is invalid")
    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 1 or source.size < rotary_dim:
        raise ProofV2TransitionError("full-attention RoPE input is invalid")
    result = source.copy()
    half = rotary_dim // 2
    exponent = np.arange(half, dtype=np.float32) / np.float32(half)
    theta = np.float32(rope_theta_q32 / float(1 << 32))
    inv_frequency = np.power(theta, -exponent, dtype=np.float32)
    angles = np.float32(position) * inv_frequency
    cosine = np.cos(angles, dtype=np.float32)
    sine = np.sin(angles, dtype=np.float32)
    first = source[:half]
    second = source[half:rotary_dim]
    result[:half] = first * cosine - second * sine
    result[half:rotary_dim] = second * cosine + first * sine
    return result


def project_qwen_full_attention_kv_rows_v2(
    *,
    qkv_rows: np.ndarray,
    position_start: int,
    parameters: FullAttentionTransitionParametersV2,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive canonical rotated K and raw V rows for every logical KV head.

    This is the miner-side counterpart of the verifier replay. It performs
    only the cache-visible K/V transform, allowing trace construction to
    commit every full-attention cache root before nonce-selected head replay.
    """

    if not isinstance(parameters, FullAttentionTransitionParametersV2):
        raise ProofV2TransitionError("full-attention parameters are invalid")
    rows = np.asarray(qkv_rows, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[0] == 0
        or rows.shape[1] != parameters.qkv_width
        or not np.isfinite(rows).all()
        or isinstance(position_start, bool)
        or not isinstance(position_start, int)
        or position_start < 0
    ):
        raise ProofV2TransitionError("full-attention QKV cache rows are invalid")
    keys = rows[
        :,
        parameters.q_width : parameters.q_width + parameters.kv_width,
    ].reshape(
        rows.shape[0],
        parameters.num_key_value_heads,
        parameters.head_dim,
    )
    values = rows[:, parameters.q_width + parameters.kv_width :].reshape(
        rows.shape[0],
        parameters.num_key_value_heads,
        parameters.head_dim,
    )
    rotated = np.empty_like(keys, dtype=np.float32)
    for row_index in range(rows.shape[0]):
        for head_index in range(parameters.num_key_value_heads):
            key = _full_attention_rmsnorm_v2(
                keys[row_index, head_index],
                parameters.k_norm_weight_f16,
                parameters.rms_epsilon_q32,
            )
            rotated[row_index, head_index] = _qwen_rope_v2(
                key,
                position=position_start + row_index,
                rotary_dim=parameters.rotary_dim,
                rope_theta_q32=parameters.rope_theta_q32,
            )
    return np.asarray(rotated, dtype=np.float32), np.asarray(values, dtype=np.float32)


@dataclass(frozen=True)
class FullAttentionReplayResultV2:
    """Selected logical-head outputs for a contiguous decode suffix."""

    rotated_queries: np.ndarray
    rotated_keys: np.ndarray
    values: np.ndarray
    core_output: np.ndarray


def replay_qwen_full_attention_head_v2(
    *,
    qkv_rows: np.ndarray,
    initial_keys: np.ndarray,
    initial_values: np.ndarray,
    query_head: int,
    position_start: int,
    parameters: FullAttentionTransitionParametersV2,
) -> FullAttentionReplayResultV2:
    """Replay one nonce-selected causal GQA head from logical QKV rows.

    ``initial_keys``/``initial_values`` are the committed logical cache prefix
    before trace row zero.  Each subsequent row appends its K/V values before
    evaluating its causal attention output.  This is intentionally backend
    independent: paged-cache slot mapping is resolved by the capture adapter
    before constructing this witness.
    """

    if not isinstance(parameters, FullAttentionTransitionParametersV2):
        raise ProofV2TransitionError("full-attention parameters are invalid")
    if (
        isinstance(query_head, bool)
        or not isinstance(query_head, int)
        or not 0 <= query_head < parameters.num_query_heads
    ):
        raise ProofV2TransitionError("full-attention query head is invalid")
    rows = np.asarray(qkv_rows, dtype=np.float32)
    keys = np.asarray(initial_keys, dtype=np.float32)
    values = np.asarray(initial_values, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[0] == 0
        or rows.shape[1] != parameters.qkv_width
        or keys.ndim != 3
        or values.ndim != 3
        or keys.shape != values.shape
        or keys.shape[1:] != (
            parameters.num_key_value_heads,
            parameters.head_dim,
        )
        or not np.isfinite(rows).all()
        or not np.isfinite(keys).all()
        or not np.isfinite(values).all()
    ):
        raise ProofV2TransitionError("full-attention witness dimensions are invalid")
    if isinstance(position_start, bool) or not isinstance(position_start, int):
        raise ProofV2TransitionError("full-attention position start is invalid")
    kv_head = query_head // (
        parameters.num_query_heads // parameters.num_key_value_heads
    )
    q_width = parameters.q_width
    kv_width = parameters.kv_width
    queries = rows[:, :q_width].reshape(
        rows.shape[0], parameters.num_query_heads, parameters.head_dim
    )
    row_keys = rows[:, q_width : q_width + kv_width].reshape(
        rows.shape[0], parameters.num_key_value_heads, parameters.head_dim
    )
    row_values = rows[:, q_width + kv_width :].reshape(
        rows.shape[0], parameters.num_key_value_heads, parameters.head_dim
    )
    prefix_keys = [keys[index, kv_head].copy() for index in range(keys.shape[0])]
    prefix_values = [values[index, kv_head].copy() for index in range(values.shape[0])]
    selected_queries = []
    selected_keys = []
    selected_values = []
    core = []
    scale = np.float32(parameters.head_dim ** -0.5)
    for row_index in range(rows.shape[0]):
        position = position_start + row_index
        query = _full_attention_rmsnorm_v2(
            queries[row_index, query_head],
            parameters.q_norm_weight_f16,
            parameters.rms_epsilon_q32,
        )
        key = _full_attention_rmsnorm_v2(
            row_keys[row_index, kv_head],
            parameters.k_norm_weight_f16,
            parameters.rms_epsilon_q32,
        )
        query = _qwen_rope_v2(
            query,
            position=position,
            rotary_dim=parameters.rotary_dim,
            rope_theta_q32=parameters.rope_theta_q32,
        )
        key = _qwen_rope_v2(
            key,
            position=position,
            rotary_dim=parameters.rotary_dim,
            rope_theta_q32=parameters.rope_theta_q32,
        )
        value = row_values[row_index, kv_head]
        prefix_keys.append(key)
        prefix_values.append(value)
        key_matrix = np.asarray(prefix_keys, dtype=np.float32)
        value_matrix = np.asarray(prefix_values, dtype=np.float32)
        logits = np.asarray(key_matrix @ query * scale, dtype=np.float32)
        logits -= np.max(logits)
        probabilities = np.exp(logits, dtype=np.float32)
        probabilities /= np.sum(probabilities, dtype=np.float32)
        selected_queries.append(query)
        selected_keys.append(key)
        selected_values.append(value)
        core.append(np.asarray(probabilities @ value_matrix, dtype=np.float32))
    return FullAttentionReplayResultV2(
        np.asarray(selected_queries, dtype=np.float32),
        np.asarray(selected_keys, dtype=np.float32),
        np.asarray(selected_values, dtype=np.float32),
        np.asarray(core, dtype=np.float32),
    )


def _sigmoid_f32(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = np.float32(1.0) / (
        np.float32(1.0) + np.exp(-values[positive], dtype=np.float32)
    )
    exponential = np.exp(values[~positive], dtype=np.float32)
    result[~positive] = exponential / (np.float32(1.0) + exponential)
    return result


def _softplus_f32(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.maximum(values, np.float32(0.0)) + np.log1p(
        np.exp(-np.abs(values), dtype=np.float32),
        dtype=np.float32,
    )


def _round_runtime_f32(values: np.ndarray, runtime_dtype: str) -> np.ndarray:
    """Reproduce the vLLM kernel's model-dtype store/load boundary."""

    if runtime_dtype == "bf16":
        return _round_bf16_f32(values)
    dtype = gdn_state_numpy_dtype_v2(runtime_dtype)
    return np.asarray(values, dtype=dtype).astype(np.float32)


def replay_qwen_gdn_block_v2(
    *,
    mixed_qkvz: np.ndarray,
    mixed_ba: np.ndarray,
    conv_state_before: np.ndarray,
    recurrent_state_before: np.ndarray,
    parameters: GDNReplayParametersV2,
) -> GDNReplayResultV2:
    """Replay one contiguous Qwen GatedDeltaNet block in float32.

    Inputs are the exact captured projection outputs before the depthwise
    convolution.  The returned output-projection input includes the Qwen GDN
    RMSNorm+SiLU gate and can therefore be cross-checked against the registered
    ``gdn.out_proj`` X commitment.
    """

    qkvz = np.asarray(mixed_qkvz)
    ba = np.asarray(mixed_ba)
    if qkvz.ndim != 2 or qkvz.shape[0] == 0:
        raise ProofV2TransitionError("GDN QKVZ block shape is invalid")
    rows = qkvz.shape[0]
    if ba.ndim != 2 or ba.shape[0] != rows:
        raise ProofV2TransitionError("GDN BA block shape is invalid")
    nk = parameters.num_key_heads
    nv = parameters.num_value_heads
    dk = parameters.key_head_dim
    dv = parameters.value_head_dim
    key_width = nk * dk
    value_width = nv * dv
    conv_width = 2 * key_width + value_width
    if qkvz.shape[1] != conv_width + value_width or ba.shape[1] != 2 * nv:
        raise ProofV2TransitionError("GDN projection widths are invalid")
    conv_state = _round_runtime_f32(
        np.asarray(conv_state_before, dtype=np.float32),
        parameters.runtime_dtype,
    )
    recurrent_state = _round_runtime_f32(
        np.asarray(recurrent_state_before, dtype=np.float32),
        parameters.recurrent_state_dtype,
    ).copy()
    if conv_state.shape != (parameters.conv_kernel_size - 1, conv_width):
        raise ProofV2TransitionError("GDN convolution state shape is invalid")
    if recurrent_state.shape != (nv, dv, dk):
        raise ProofV2TransitionError("GDN recurrent state shape is invalid")
    if not (
        np.isfinite(qkvz).all()
        and np.isfinite(ba).all()
        and np.isfinite(conv_state).all()
        and np.isfinite(recurrent_state).all()
    ):
        raise ProofV2TransitionError("GDN replay inputs contain non-finite values")

    qkvz_f32 = qkvz.astype(np.float32)
    ba_f32 = ba.astype(np.float32)
    conv_weight = np.asarray(parameters.conv_weight, dtype=np.float32)
    a_log = np.asarray(parameters.a_log, dtype=np.float32)
    dt_bias = np.asarray(parameters.dt_bias, dtype=np.float32)
    norm_weight = np.asarray(parameters.norm_weight, dtype=np.float32)
    group_size = nv // nk
    scale = np.float32(dk**-0.5)
    core_rows = []
    projection_rows = []

    for row_index in range(rows):
        current = qkvz_f32[row_index, :conv_width]
        window = np.concatenate((conv_state, current.reshape(1, -1)), axis=0)
        convolved = np.sum(
            window.T * conv_weight,
            axis=1,
            dtype=np.float32,
        )
        convolved = convolved * _sigmoid_f32(convolved)
        convolved = _round_runtime_f32(
            convolved,
            parameters.runtime_dtype,
        )
        conv_state = window[1:].copy()

        q = convolved[:key_width].reshape(nk, dk)
        k = convolved[key_width : 2 * key_width].reshape(nk, dk)
        v = convolved[2 * key_width :].reshape(nv, dv)
        z = qkvz_f32[row_index, conv_width:].reshape(nv, dv)
        b = ba_f32[row_index, :nv]
        a = ba_f32[row_index, nv:]

        q *= np.reciprocal(
            np.sqrt(
                np.sum(q * q, axis=1, keepdims=True, dtype=np.float32)
                + np.float32(1e-6),
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        k *= np.reciprocal(
            np.sqrt(
                np.sum(k * k, axis=1, keepdims=True, dtype=np.float32)
                + np.float32(1e-6),
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        decay = np.exp(
            -np.exp(a_log, dtype=np.float32) * _softplus_f32(a + dt_bias),
            dtype=np.float32,
        )
        beta = _round_runtime_f32(
            _sigmoid_f32(b),
            parameters.runtime_dtype,
        )
        output = np.empty((nv, dv), dtype=np.float32)
        for value_head in range(nv):
            key_head = value_head // group_size
            state = recurrent_state[value_head].copy()
            state *= decay[value_head]
            delta = v[value_head] - state @ k[key_head]
            state += (
                beta[value_head] * delta.reshape(-1, 1) * k[key_head].reshape(1, -1)
            )
            state = _round_runtime_f32(
                state,
                parameters.recurrent_state_dtype,
            )
            recurrent_state[value_head] = state
            output[value_head] = state @ (q[key_head] * scale)

        runtime_output = _round_runtime_f32(
            output,
            parameters.runtime_dtype,
        )
        variance = np.mean(
            runtime_output * runtime_output,
            axis=1,
            keepdims=True,
            dtype=np.float32,
        )
        normalized = runtime_output * np.reciprocal(
            np.sqrt(variance + np.float32(parameters.rms_epsilon), dtype=np.float32),
            dtype=np.float32,
        )
        normalized *= norm_weight.reshape(1, -1)
        gated = _round_runtime_f32(
            normalized * (z * _sigmoid_f32(z)),
            parameters.runtime_dtype,
        )
        core_rows.append(runtime_output.copy())
        projection_rows.append(gated.reshape(-1).copy())

    return GDNReplayResultV2(
        conv_state_after=np.ascontiguousarray(conv_state, dtype=np.float32),
        recurrent_state_after=np.ascontiguousarray(
            recurrent_state,
            dtype=np.float32,
        ),
        core_output=np.ascontiguousarray(np.stack(core_rows), dtype=np.float32),
        out_projection_input=np.ascontiguousarray(
            np.stack(projection_rows),
            dtype=np.float32,
        ),
    )


__all__ = [
    "DEFAULT_TRANSITION_BLOCK_ROWS",
    "FULL_ATTENTION_TRANSITION_PROFILE_V1",
    "FullAttentionReplayResultV2",
    "FullAttentionTransitionParametersV2",
    "GDNReplayParametersV2",
    "GDNReplayResultV2",
    "GDNTransitionParametersV2",
    "GDN_TRANSITION_PROFILE_V1",
    "MAX_TRANSITION_BLOCK_ROWS",
    "MAX_TRANSITION_HISTORY_LAYERS",
    "MAX_TRANSITION_HISTORY_ROWS",
    "ProofV2TransitionError",
    "TRANSITION_HISTORY_PROFILE_V1",
    "TRANSITION_STREAM_AFTER_ATTENTION",
    "TRANSITION_STREAM_FINAL_RESIDUAL",
    "TRANSITION_STREAM_RESIDUAL_IN",
    "TransitionChallengeV2",
    "TransitionHistoryCommitmentV2",
    "TransitionHistoryOpeningV2",
    "TransitionHistoryStateV2",
    "TransitionHistoryStreamV2",
    "build_transition_history_commitment_v2",
    "derive_transition_challenges_v2",
    "gdn_state_dtype_code_v2",
    "gdn_state_dtype_from_code_v2",
    "gdn_state_numpy_dtype_v2",
    "replay_qwen_gdn_block_v2",
    "replay_qwen_full_attention_head_v2",
    "project_qwen_full_attention_kv_rows_v2",
    "transition_history_stream_index_v2",
]
