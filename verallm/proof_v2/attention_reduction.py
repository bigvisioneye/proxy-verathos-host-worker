"""Reference authenticated attention-reduction commitments for proof v2.

This module is deliberately a *reference* primitive, not an enabled runtime
adapter.  A full-attention hard audit cannot disclose a whole K/V prefix when
the context is hundreds of thousands of tokens long.  Instead, a qualified
runtime adapter will commit an associative softmax-reduction tree before the
validator nonce.  The nonce can then select one K/V tile; the prover supplies
that tile plus logarithmically many reduction siblings.

The code below fixes the commitment/witness shape and validates the critical
binding properties on a deterministic NumPy reference implementation.  It
does *not* prove that every reduction sibling came from authenticated model
execution: a self-consistent miner-authored cache/tree can satisfy one sampled
opening.  It must therefore never discharge the hard full-attention transition
gate.  A production profile would need authenticated cache provenance and an
aggregate/reduction verification argument in addition to a GPU/kernel
arithmetic adapter and cross-backend qualification.  This reference profile is
intentionally not accepted by the manifest loader.
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


REFERENCE_ATTENTION_REDUCTION_PROFILE_V1 = (
    "reference_f32_attention_reduction_v1"
)
DEFAULT_ATTENTION_REDUCTION_TILE_ROWS_V1 = 128
MAX_ATTENTION_REDUCTION_TILE_ROWS = 65_536
MAX_ATTENTION_REDUCTION_HEAD_DIM = 65_536
MAX_ATTENTION_REDUCTION_TILE_BYTES = 16 << 20

_COMMITMENT_MAGIC = b"ARC1"
_KV_TILE_MAGIC = b"AKV1"
_SUMMARY_MAGIC = b"ARS1"
_NODE_DOMAIN = b"VERATHOS/PROOF_V2/ATTENTION_REDUCTION_NODE/SHA256"
_KV_DIGEST_DOMAIN = b"VERATHOS/PROOF_V2/ATTENTION_REDUCTION_KV/SHA256"
_COMMITMENT_DOMAIN = b"VERATHOS/PROOF_V2/ATTENTION_REDUCTION/SHA256"
_CHALLENGE_DOMAIN = b"VERATHOS/PROOF_V2/ATTENTION_REDUCTION_CHALLENGE/SHA256"


class AttentionReductionError(ValueError):
    """An authenticated attention-reduction object is malformed."""


def _u32(value: int, name: str) -> bytes:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < (1 << 32)
    ):
        raise AttentionReductionError(f"{name} must be an unsigned 32-bit integer")
    return struct.pack("<I", value)


def _fixed32(value: bytes, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise AttentionReductionError(f"{name} must be exactly 32 bytes")
    return value


def _profile_bytes(profile: str) -> bytes:
    if profile != REFERENCE_ATTENTION_REDUCTION_PROFILE_V1:
        raise AttentionReductionError("attention-reduction profile is unsupported")
    encoded = profile.encode("ascii")
    return struct.pack("<B", len(encoded)) + encoded


def _f32_scalar(value: float) -> np.float32:
    return np.asarray(value, dtype=np.float32).reshape(()).item()


def _f32_bytes(values: np.ndarray, *, expected: int, name: str) -> bytes:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (expected,) or not np.isfinite(array).all():
        raise AttentionReductionError(f"{name} is not a finite float32 vector")
    return np.ascontiguousarray(array, dtype="<f4").tobytes()


def _f16_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float16)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise AttentionReductionError(f"{name} must be a non-empty rank-two matrix")
    if not np.isfinite(array).all():
        raise AttentionReductionError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array, dtype="<f2")


def _f16_vector(value: np.ndarray, *, expected: int, name: str) -> bytes:
    array = np.asarray(value, dtype=np.float16)
    if array.shape != (expected,) or not np.isfinite(array).all():
        raise AttentionReductionError(f"{name} is not a finite fp16 vector")
    return np.ascontiguousarray(array, dtype="<f2").tobytes()


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise AttentionReductionError("attention-reduction tile count is invalid")
    return 1 << (value - 1).bit_length()


def _reduction_depth(tile_count: int) -> int:
    return _next_power_of_two(tile_count).bit_length() - 1


def _sample_below(seed: bytes, limit: int) -> int:
    """Draw an unbiased index from a transcript-bound SHA-256 stream."""

    if not isinstance(seed, bytes) or len(seed) != 32:
        raise AttentionReductionError("attention-reduction challenge seed is invalid")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise AttentionReductionError("attention-reduction challenge limit is invalid")
    ceiling = (1 << 64) - ((1 << 64) % limit)
    counter = 0
    while True:
        candidate = int.from_bytes(
            hashlib.sha256(seed + counter.to_bytes(8, "little")).digest()[:8],
            "little",
        )
        if candidate < ceiling:
            return candidate % limit
        counter += 1


@dataclass(frozen=True, order=True)
class AttentionReductionTileChallengeV2:
    """One post-commitment logical attention tile selection."""

    token_index: int
    layer_idx: int
    query_head_idx: int
    tile_index: int

    def __post_init__(self) -> None:
        _u32(self.token_index, "attention-reduction token index")
        _u32(self.layer_idx, "attention-reduction layer index")
        _u32(self.query_head_idx, "attention-reduction query head index")
        _u32(self.tile_index, "attention-reduction tile index")


@dataclass(frozen=True)
class AttentionReductionSummaryV2:
    """One stable-softmax segment summary.

    ``weighted_values_f32 / normalizer_f32`` is the segment's attention
    output.  An empty padded segment is represented canonically by
    ``(-inf, 0, zeros)`` and is the identity for ``combine``.
    """

    head_dim: int
    maximum_f32: bytes
    normalizer_f32: bytes
    weighted_values_f32: bytes

    def __post_init__(self) -> None:
        _u32(self.head_dim, "attention-reduction head dimension")
        if not 0 < self.head_dim <= MAX_ATTENTION_REDUCTION_HEAD_DIM:
            raise AttentionReductionError("attention-reduction head dimension is invalid")
        if not isinstance(self.maximum_f32, bytes) or len(self.maximum_f32) != 4:
            raise AttentionReductionError("attention-reduction maximum is invalid")
        if not isinstance(self.normalizer_f32, bytes) or len(self.normalizer_f32) != 4:
            raise AttentionReductionError("attention-reduction normalizer is invalid")
        if (
            not isinstance(self.weighted_values_f32, bytes)
            or len(self.weighted_values_f32) != self.head_dim * 4
        ):
            raise AttentionReductionError("attention-reduction weighted values are invalid")
        maximum = self.maximum
        normalizer = self.normalizer
        weighted = self.weighted_values
        if np.isnan(maximum) or np.isnan(normalizer) or not np.isfinite(weighted).all():
            raise AttentionReductionError("attention-reduction summary is non-finite")
        if normalizer == np.float32(0.0):
            if maximum != np.float32(-np.inf) or np.any(weighted != np.float32(0.0)):
                raise AttentionReductionError("empty attention-reduction summary is not canonical")
        elif not np.isfinite(maximum) or not np.isfinite(normalizer) or normalizer <= 0:
            raise AttentionReductionError("attention-reduction summary is invalid")

    @property
    def maximum(self) -> np.float32:
        return np.frombuffer(self.maximum_f32, dtype="<f4", count=1)[0]

    @property
    def normalizer(self) -> np.float32:
        return np.frombuffer(self.normalizer_f32, dtype="<f4", count=1)[0]

    @property
    def weighted_values(self) -> np.ndarray:
        return np.frombuffer(self.weighted_values_f32, dtype="<f4").copy()

    @property
    def is_empty(self) -> bool:
        return self.normalizer == np.float32(0.0)

    def output_f32(self) -> bytes:
        if self.is_empty:
            raise AttentionReductionError("empty attention-reduction root has no output")
        return _f32_bytes(
            np.asarray(self.weighted_values / self.normalizer, dtype=np.float32),
            expected=self.head_dim,
            name="attention-reduction output",
        )

    def canonical_bytes(self) -> bytes:
        return (
            _SUMMARY_MAGIC
            + _u32(self.head_dim, "attention-reduction head dimension")
            + self.maximum_f32
            + self.normalizer_f32
            + self.weighted_values_f32
        )

    @classmethod
    def empty(cls, head_dim: int) -> "AttentionReductionSummaryV2":
        _u32(head_dim, "attention-reduction head dimension")
        return cls(
            head_dim,
            np.asarray([-np.inf], dtype="<f4").tobytes(),
            np.asarray([0.0], dtype="<f4").tobytes(),
            np.zeros(head_dim, dtype="<f4").tobytes(),
        )

    @classmethod
    def from_values(
        cls,
        *,
        maximum: float,
        normalizer: float,
        weighted_values: np.ndarray,
        head_dim: int,
    ) -> "AttentionReductionSummaryV2":
        _u32(head_dim, "attention-reduction head dimension")
        maximum_f32 = _f32_scalar(maximum)
        normalizer_f32 = _f32_scalar(normalizer)
        if normalizer_f32 == np.float32(0.0):
            return cls.empty(head_dim)
        return cls(
            head_dim,
            np.asarray([maximum_f32], dtype="<f4").tobytes(),
            np.asarray([normalizer_f32], dtype="<f4").tobytes(),
            _f32_bytes(
                weighted_values,
                expected=head_dim,
                name="attention-reduction weighted values",
            ),
        )


def combine_attention_reduction_summaries_v2(
    left: AttentionReductionSummaryV2,
    right: AttentionReductionSummaryV2,
) -> AttentionReductionSummaryV2:
    """Combine two stable-softmax summaries in a canonical tree order."""

    if not isinstance(left, AttentionReductionSummaryV2) or not isinstance(
        right, AttentionReductionSummaryV2
    ):
        raise AttentionReductionError("attention-reduction summary is invalid")
    if left.head_dim != right.head_dim:
        raise AttentionReductionError("attention-reduction summary dimensions differ")
    if left.is_empty:
        return right
    if right.is_empty:
        return left
    maximum = np.maximum(left.maximum, right.maximum).astype(np.float32)
    left_scale = np.exp(
        np.asarray(left.maximum - maximum, dtype=np.float32), dtype=np.float32
    )
    right_scale = np.exp(
        np.asarray(right.maximum - maximum, dtype=np.float32), dtype=np.float32
    )
    normalizer = np.asarray(
        left_scale * left.normalizer + right_scale * right.normalizer,
        dtype=np.float32,
    )
    weighted = np.asarray(
        left_scale * left.weighted_values + right_scale * right.weighted_values,
        dtype=np.float32,
    )
    return AttentionReductionSummaryV2.from_values(
        maximum=maximum,
        normalizer=normalizer,
        weighted_values=weighted,
        head_dim=left.head_dim,
    )


def reference_attention_tile_summary_v2(
    *,
    query_f16: bytes,
    keys_f16: np.ndarray,
    values_f16: np.ndarray,
) -> AttentionReductionSummaryV2:
    """Return a reference f32 stable-softmax summary for one causal K/V tile.

    This numerical path is only a conformance reference.  It is not asserted
    to match a production FlashAttention kernel until a signed adapter has
    measured and qualified that equality.
    """

    keys = _f16_matrix(keys_f16, name="attention-reduction keys")
    values = _f16_matrix(values_f16, name="attention-reduction values")
    if values.shape != keys.shape:
        raise AttentionReductionError("attention-reduction K/V dimensions differ")
    head_dim = int(keys.shape[1])
    if not isinstance(query_f16, bytes) or len(query_f16) != head_dim * 2:
        raise AttentionReductionError("attention-reduction query is invalid")
    query = np.frombuffer(query_f16, dtype="<f2").astype(np.float32)
    if not np.isfinite(query).all():
        raise AttentionReductionError("attention-reduction query is non-finite")
    keys_f32 = keys.astype(np.float32)
    values_f32 = values.astype(np.float32)
    scores = np.asarray(
        np.sum(keys_f32 * query[None, :], axis=1, dtype=np.float32)
        * np.float32(head_dim ** -0.5),
        dtype=np.float32,
    )
    maximum = np.max(scores).astype(np.float32)
    exponentials = np.exp(
        np.asarray(scores - maximum, dtype=np.float32), dtype=np.float32
    )
    normalizer = np.sum(exponentials, dtype=np.float32)
    weighted = np.sum(
        exponentials[:, None] * values_f32,
        axis=0,
        dtype=np.float32,
    )
    return AttentionReductionSummaryV2.from_values(
        maximum=maximum,
        normalizer=normalizer,
        weighted_values=weighted,
        head_dim=head_dim,
    )


def attention_kv_tile_leaf_v2(
    *,
    tile_index: int,
    keys_f16: np.ndarray,
    values_f16: np.ndarray,
) -> bytes:
    """Canonical raw leaf for one logical paged K/V tile."""

    _u32(tile_index, "attention-reduction tile index")
    keys = _f16_matrix(keys_f16, name="attention-reduction keys")
    values = _f16_matrix(values_f16, name="attention-reduction values")
    if values.shape != keys.shape:
        raise AttentionReductionError("attention-reduction K/V dimensions differ")
    rows, head_dim = map(int, keys.shape)
    encoded_values = keys.tobytes() + values.tobytes()
    if len(encoded_values) > MAX_ATTENTION_REDUCTION_TILE_BYTES:
        raise AttentionReductionError("attention-reduction K/V tile is too large")
    return (
        _KV_TILE_MAGIC
        + _u32(tile_index, "attention-reduction tile index")
        + _u32(rows, "attention-reduction tile rows")
        + _u32(head_dim, "attention-reduction head dimension")
        + encoded_values
    )


def parse_attention_kv_tile_leaf_v2(encoded: bytes) -> tuple[int, np.ndarray, np.ndarray]:
    """Decode a canonical logical K/V tile leaf."""

    if not isinstance(encoded, bytes) or len(encoded) < 16:
        raise AttentionReductionError("attention-reduction K/V tile is malformed")
    if encoded[:4] != _KV_TILE_MAGIC:
        raise AttentionReductionError("attention-reduction K/V tile header is invalid")
    tile_index, rows, head_dim = struct.unpack_from("<III", encoded, 4)
    if (
        rows == 0
        or rows > MAX_ATTENTION_REDUCTION_TILE_ROWS
        or head_dim == 0
        or head_dim > MAX_ATTENTION_REDUCTION_HEAD_DIM
    ):
        raise AttentionReductionError("attention-reduction K/V tile dimensions are invalid")
    payload = encoded[16:]
    expected = rows * head_dim * 4
    if len(payload) != expected or len(payload) > MAX_ATTENTION_REDUCTION_TILE_BYTES:
        raise AttentionReductionError("attention-reduction K/V tile length is invalid")
    split = rows * head_dim * 2
    keys = np.frombuffer(payload[:split], dtype="<f2").reshape(rows, head_dim)
    values = np.frombuffer(payload[split:], dtype="<f2").reshape(rows, head_dim)
    if not np.isfinite(keys).all() or not np.isfinite(values).all():
        raise AttentionReductionError("attention-reduction K/V tile is non-finite")
    return tile_index, keys.copy(), values.copy()


def _kv_digest(encoded: bytes) -> bytes:
    return hashlib.sha256(_KV_DIGEST_DOMAIN + encoded).digest()


@dataclass(frozen=True)
class _ReductionNode:
    summary: AttentionReductionSummaryV2
    digest: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.summary, AttentionReductionSummaryV2):
            raise AttentionReductionError("attention-reduction node summary is invalid")
        _fixed32(self.digest, "attention-reduction node digest")


def _leaf_node(
    *,
    tile_index: int,
    summary: AttentionReductionSummaryV2,
    kv_leaf: bytes | None,
) -> _ReductionNode:
    _u32(tile_index, "attention-reduction tile index")
    if kv_leaf is None:
        if not summary.is_empty:
            raise AttentionReductionError("padded attention-reduction leaf is not empty")
        payload = b"E" + _u32(tile_index, "attention-reduction tile index")
    else:
        payload = (
            b"L"
            + _u32(tile_index, "attention-reduction tile index")
            + _kv_digest(kv_leaf)
        )
    return _ReductionNode(
        summary,
        hashlib.sha256(_NODE_DOMAIN + payload + summary.canonical_bytes()).digest(),
    )


def _parent_node(left: _ReductionNode, right: _ReductionNode) -> _ReductionNode:
    summary = combine_attention_reduction_summaries_v2(left.summary, right.summary)
    return _ReductionNode(
        summary,
        hashlib.sha256(
            _NODE_DOMAIN
            + b"N"
            + summary.canonical_bytes()
            + left.digest
            + right.digest
        ).digest(),
    )


@dataclass(frozen=True)
class AttentionReductionCommitmentV2:
    """Pre-nonce roots for one query head's full causal K/V range."""

    profile: str
    sequence_length: int
    head_dim: int
    tile_rows: int
    kv_root: bytes
    reduction_root: bytes
    reduction_summary: AttentionReductionSummaryV2

    def __post_init__(self) -> None:
        _profile_bytes(self.profile)
        _u32(self.sequence_length, "attention-reduction sequence length")
        _u32(self.head_dim, "attention-reduction head dimension")
        _u32(self.tile_rows, "attention-reduction tile rows")
        if (
            self.sequence_length == 0
            or not 0 < self.head_dim <= MAX_ATTENTION_REDUCTION_HEAD_DIM
            or not 0 < self.tile_rows <= MAX_ATTENTION_REDUCTION_TILE_ROWS
        ):
            raise AttentionReductionError("attention-reduction commitment dimensions are invalid")
        _fixed32(self.kv_root, "attention-reduction K/V root")
        _fixed32(self.reduction_root, "attention-reduction root")
        if (
            not isinstance(self.reduction_summary, AttentionReductionSummaryV2)
            or self.reduction_summary.head_dim != self.head_dim
            or self.reduction_summary.is_empty
        ):
            raise AttentionReductionError("attention-reduction root summary is invalid")

    @property
    def tile_count(self) -> int:
        return (self.sequence_length + self.tile_rows - 1) // self.tile_rows

    def canonical_bytes(self) -> bytes:
        return (
            _COMMITMENT_MAGIC
            + _profile_bytes(self.profile)
            + _u32(self.sequence_length, "attention-reduction sequence length")
            + _u32(self.head_dim, "attention-reduction head dimension")
            + _u32(self.tile_rows, "attention-reduction tile rows")
            + self.kv_root
            + self.reduction_root
            + self.reduction_summary.canonical_bytes()
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_COMMITMENT_DOMAIN + self.canonical_bytes()).digest()


def derive_attention_reduction_tile_challenge_v2(
    *,
    transcript_state: bytes,
    commitment: AttentionReductionCommitmentV2,
    token_index: int,
    layer_idx: int,
    query_head_idx: int,
) -> AttentionReductionTileChallengeV2:
    """Derive a verifier-owned tile after its reduction root is frozen.

    The selected tile is bound to the particular token/layer/query-head root,
    not merely to a request-wide random stream.  Reusing a favorable opening
    from another head or decode position therefore changes the challenge.
    """

    if not isinstance(transcript_state, bytes) or len(transcript_state) != 32:
        raise AttentionReductionError("attention-reduction transcript is invalid")
    if not isinstance(commitment, AttentionReductionCommitmentV2):
        raise AttentionReductionError("attention-reduction commitment is invalid")
    context = (
        _CHALLENGE_DOMAIN
        + transcript_state
        + commitment.digest()
        + _u32(token_index, "attention-reduction token index")
        + _u32(layer_idx, "attention-reduction layer index")
        + _u32(query_head_idx, "attention-reduction query head index")
    )
    return AttentionReductionTileChallengeV2(
        token_index,
        layer_idx,
        query_head_idx,
        _sample_below(hashlib.sha256(context).digest(), commitment.tile_count),
    )


@dataclass(frozen=True)
class AttentionReductionSiblingV2:
    """One authenticated sibling summary on a selected tile's tree path."""

    summary: AttentionReductionSummaryV2
    digest: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.summary, AttentionReductionSummaryV2):
            raise AttentionReductionError("attention-reduction sibling summary is invalid")
        _fixed32(self.digest, "attention-reduction sibling digest")


@dataclass(frozen=True)
class AttentionReductionOpeningV2:
    """A nonce-selected K/V tile plus its reduction and cache paths."""

    tile_index: int
    kv_tile: bytes
    kv_path: MerklePath
    reduction_siblings: tuple[AttentionReductionSiblingV2, ...]

    def __post_init__(self) -> None:
        _u32(self.tile_index, "attention-reduction tile index")
        if not isinstance(self.kv_tile, bytes):
            raise AttentionReductionError("attention-reduction K/V tile is invalid")
        if not isinstance(self.kv_path, MerklePath):
            raise AttentionReductionError("attention-reduction K/V path is invalid")
        try:
            siblings = tuple(self.reduction_siblings)
        except TypeError as exc:
            raise AttentionReductionError("attention-reduction sibling path is invalid") from exc
        if not all(isinstance(item, AttentionReductionSiblingV2) for item in siblings):
            raise AttentionReductionError("attention-reduction sibling path is invalid")
        object.__setattr__(self, "reduction_siblings", siblings)


@dataclass(frozen=True)
class AttentionReductionReferenceStateV2:
    """Reference-only retained state used to create selected openings.

    Production serving must not retain this Python tree for every query.  It
    must emit the committed root during regular GPU attention and recompute the
    selected path from the paged cache after the nonce.
    """

    commitment: AttentionReductionCommitmentV2
    query_f16: bytes
    kv_tree: MerkleTree
    kv_tiles: tuple[bytes, ...]
    reduction_levels: tuple[tuple[_ReductionNode, ...], ...]

    def opening(self, tile_index: int) -> AttentionReductionOpeningV2:
        if not 0 <= tile_index < self.commitment.tile_count:
            raise AttentionReductionError("attention-reduction tile index is invalid")
        index = tile_index
        siblings = []
        for level in self.reduction_levels[:-1]:
            sibling = level[index ^ 1]
            siblings.append(AttentionReductionSiblingV2(sibling.summary, sibling.digest))
            index //= 2
        return AttentionReductionOpeningV2(
            tile_index,
            self.kv_tiles[tile_index],
            self.kv_tree.get_path(tile_index),
            tuple(siblings),
        )


def build_attention_reduction_reference_v2(
    *,
    query_f16: np.ndarray,
    keys_f16: np.ndarray,
    values_f16: np.ndarray,
    tile_rows: int = DEFAULT_ATTENTION_REDUCTION_TILE_ROWS_V1,
) -> AttentionReductionReferenceStateV2:
    """Build a reference commitment over a full causal K/V history.

    The logical sequence length is unrestricted by the proof format (other
    than its canonical uint32 encoding); only a selected tile is ever opened.
    """

    keys = _f16_matrix(keys_f16, name="attention-reduction keys")
    values = _f16_matrix(values_f16, name="attention-reduction values")
    if values.shape != keys.shape:
        raise AttentionReductionError("attention-reduction K/V dimensions differ")
    sequence_length, head_dim = map(int, keys.shape)
    if not 0 < tile_rows <= MAX_ATTENTION_REDUCTION_TILE_ROWS:
        raise AttentionReductionError("attention-reduction tile rows are invalid")
    query = _f16_vector(
        query_f16,
        expected=head_dim,
        name="attention-reduction query",
    )
    leaves = []
    nodes = []
    for tile_index, start in enumerate(range(0, sequence_length, tile_rows)):
        stop = min(start + tile_rows, sequence_length)
        leaf = attention_kv_tile_leaf_v2(
            tile_index=tile_index,
            keys_f16=keys[start:stop],
            values_f16=values[start:stop],
        )
        leaves.append(leaf)
        nodes.append(
            _leaf_node(
                tile_index=tile_index,
                summary=reference_attention_tile_summary_v2(
                    query_f16=query,
                    keys_f16=keys[start:stop],
                    values_f16=values[start:stop],
                ),
                kv_leaf=leaf,
            )
        )
    kv_tree = MerkleTree(leaves)
    padded_count = _next_power_of_two(len(nodes))
    while len(nodes) < padded_count:
        tile_index = len(nodes)
        nodes.append(
            _leaf_node(
                tile_index=tile_index,
                summary=AttentionReductionSummaryV2.empty(head_dim),
                kv_leaf=None,
            )
        )
    levels = [tuple(nodes)]
    current = nodes
    while len(current) > 1:
        current = [
            _parent_node(current[index], current[index + 1])
            for index in range(0, len(current), 2)
        ]
        levels.append(tuple(current))
    root = levels[-1][0]
    commitment = AttentionReductionCommitmentV2(
        REFERENCE_ATTENTION_REDUCTION_PROFILE_V1,
        sequence_length,
        head_dim,
        tile_rows,
        kv_tree.root,
        root.digest,
        root.summary,
    )
    return AttentionReductionReferenceStateV2(
        commitment,
        query,
        kv_tree,
        tuple(leaves),
        tuple(levels),
    )


def verify_attention_reduction_opening_v2(
    *,
    commitment: AttentionReductionCommitmentV2,
    query_f16: bytes,
    core_output_f32: bytes,
    opening: AttentionReductionOpeningV2,
) -> bool:
    """Verify one selected reference K/V tile against a frozen root.

    This deliberately returns ``False`` for malformed untrusted witness data.
    The caller is responsible for separately binding ``query_f16`` and the
    claimed core-output slice to the authenticated QKV/output-projection trace.
    """

    try:
        if not isinstance(commitment, AttentionReductionCommitmentV2) or not isinstance(
            opening, AttentionReductionOpeningV2
        ):
            return False
        if not isinstance(query_f16, bytes) or len(query_f16) != commitment.head_dim * 2:
            return False
        if not isinstance(core_output_f32, bytes) or len(core_output_f32) != commitment.head_dim * 4:
            return False
        claimed_output = np.frombuffer(core_output_f32, dtype="<f4")
        if not np.isfinite(claimed_output).all():
            return False
        if (
            opening.tile_index >= commitment.tile_count
            or opening.kv_path.leaf_index != opening.tile_index
            or len(opening.reduction_siblings) != _reduction_depth(commitment.tile_count)
        ):
            return False
        tile_index, keys, values = parse_attention_kv_tile_leaf_v2(opening.kv_tile)
        expected_rows = min(
            commitment.tile_rows,
            commitment.sequence_length - opening.tile_index * commitment.tile_rows,
        )
        if (
            tile_index != opening.tile_index
            or keys.shape != (expected_rows, commitment.head_dim)
            or values.shape != keys.shape
            or not verify_merkle_path(
                commitment.kv_root,
                opening.kv_tile,
                opening.kv_path,
            )
        ):
            return False
        node = _leaf_node(
            tile_index=opening.tile_index,
            summary=reference_attention_tile_summary_v2(
                query_f16=query_f16,
                keys_f16=keys,
                values_f16=values,
            ),
            kv_leaf=opening.kv_tile,
        )
        index = opening.tile_index
        for sibling in opening.reduction_siblings:
            if sibling.summary.head_dim != commitment.head_dim:
                return False
            node = (
                _parent_node(sibling, node)
                if index & 1
                else _parent_node(node, sibling)
            )
            index //= 2
        if (
            node.digest != commitment.reduction_root
            or node.summary != commitment.reduction_summary
            or node.summary.output_f32() != core_output_f32
        ):
            return False
        return True
    except (AttentionReductionError, ValueError, TypeError, struct.error):
        return False


__all__ = [
    "AttentionReductionCommitmentV2",
    "AttentionReductionError",
    "AttentionReductionOpeningV2",
    "AttentionReductionReferenceStateV2",
    "AttentionReductionSiblingV2",
    "AttentionReductionSummaryV2",
    "AttentionReductionTileChallengeV2",
    "DEFAULT_ATTENTION_REDUCTION_TILE_ROWS_V1",
    "REFERENCE_ATTENTION_REDUCTION_PROFILE_V1",
    "attention_kv_tile_leaf_v2",
    "build_attention_reduction_reference_v2",
    "combine_attention_reduction_summaries_v2",
    "derive_attention_reduction_tile_challenge_v2",
    "parse_attention_kv_tile_leaf_v2",
    "reference_attention_tile_summary_v2",
    "verify_attention_reduction_opening_v2",
]
