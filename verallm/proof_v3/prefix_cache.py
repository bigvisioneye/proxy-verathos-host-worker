"""Canonical pre-nonce prefix-cache commitments for compact proof v3.

The cache lane commits logical cache blocks, never miner-local physical block
numbers.  A hard nonce selects block records only after the inventory root is
frozen.  Runtime adapters subsequently bind each selected record's state root
to paged K/V or GDN boundary values and to the registered-model replay.
"""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from typing import Sequence

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.execution_anchor import (
    ExecutionAnchorCommitmentV3,
    ExecutionAnchorLaneOpeningV3,
    execution_anchor_lane_bytes_v3,
    verify_execution_anchor_lane_v3,
)
from zkllm.crypto.merkle import MerkleTree, verify_merkle_path
from zkllm.types import MerklePath

PREFIX_CACHE_BLOCK_COMMITMENT_ABI_V3 = "prefix_cache.logical_blocks.v1"
PREFIX_CACHE_POSTNONCE_REPLAY_ABI_V3 = "prefix_cache.postnonce_replay.v1"
MAX_PREFIX_CACHE_BLOCKS_V3 = 1 << 16
MAX_PREFIX_CACHE_BLOCK_SAMPLES_V3 = 1024
MAX_PREFIX_CACHE_BLOCK_TOKENS_V3 = 1 << 16
MAX_PREFIX_CACHE_STATE_STAGES_V3 = 1 << 12
PREFIX_CACHE_BLOCK_SAMPLE_COUNT_V3 = 4
PREFIX_CACHE_ROWS_PER_STAGE_V3 = 2
PREFIX_CACHE_LANES_PER_ROW_V3 = 1

_CONTENT_DOMAIN = b"VERATHOS/PROOF_V3/PREFIX_CACHE/BLOCK_CONTENT/V1"
_COMMITMENT_DOMAIN = b"VERATHOS/PROOF_V3/PREFIX_CACHE/COMMITMENT/V1"
_CHALLENGE_DOMAIN = b"VERATHOS/PROOF_V3/PREFIX_CACHE/CHALLENGE/V1"
_FOLD_DOMAIN = b"VERATHOS/PROOF_V3/PREFIX_CACHE/CAPTURE_FOLD/V1"
_SUFFIX_DOMAIN = b"VERATHOS/PROOF_V3/PREFIX_CACHE/EXECUTED_SUFFIX/V1"
_LANE_CHALLENGE_DOMAIN = b"VERATHOS/PROOF_V3/PREFIX_CACHE/LANE_CHALLENGE/V1"
_STAGE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,95}$")


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


def _canonical_required_lane_keys_v3(
    lane_keys: Sequence[tuple[int, str, int, int]],
    *,
    verification: bool = False,
) -> tuple[tuple[int, str, int, int], ...]:
    """Validate verifier-derived cache lanes before indexing their fields."""

    error_type = ProofV3VerificationError if verification else ProofV3Error
    try:
        keys = tuple(lane_keys)
    except TypeError as exc:
        raise error_type(
            "prefix-cache required lane plan is malformed"
        ) from exc
    if len(keys) > MAX_PREFIX_CACHE_BLOCK_SAMPLES_V3:
        raise error_type(
            "prefix-cache required lane plan exceeds its bound"
        )
    canonical: list[tuple[int, str, int, int]] = []
    for key in keys:
        if not isinstance(key, tuple) or len(key) != 4:
            raise error_type(
                "prefix-cache required lane plan is malformed"
            )
        block_index, stage_id, row_index, lane_index = key
        if (
            type(block_index) is not int
            or not 0 <= block_index < MAX_PREFIX_CACHE_BLOCKS_V3
            or not isinstance(stage_id, str)
            or _STAGE_ID_RE.fullmatch(stage_id) is None
            or type(row_index) is not int
            or row_index < 0
            or type(lane_index) is not int
            or lane_index < 0
        ):
            raise error_type(
                "prefix-cache required lane plan is malformed"
            )
        canonical.append(key)
    result = tuple(canonical)
    if result != tuple(sorted(set(result))):
        raise error_type(
            "prefix-cache required lane plan is noncanonical"
        )
    return result


def derive_prefix_cache_content_digests_v3(
    *,
    execution_profile_digest: bytes,
    cache_salt_digest: bytes,
    token_ids: Sequence[int],
    block_token_count: int,
) -> tuple[bytes, ...]:
    """Derive content-addressed block digests in canonical token order."""

    profile = _fixed32(
        execution_profile_digest,
        "execution_profile_digest",
        nonzero=True,
    )
    salt = _fixed32(cache_salt_digest, "cache_salt_digest")
    width = _u32(block_token_count, "block_token_count", positive=True)
    if width > MAX_PREFIX_CACHE_BLOCK_TOKENS_V3:
        raise ProofV3Error("prefix-cache block token count exceeds its bound")
    tokens = tuple(token_ids)
    if len(tokens) >= 1 << 32:
        raise ProofV3Error("prefix-cache token sequence is too large")
    for token in tokens:
        _u32(token, "prefix-cache token id")

    parent = bytes(32)
    result: list[bytes] = []
    for start in range(0, len(tokens), width):
        block = tokens[start : start + width]
        count = len(block)
        parent = hashlib.sha256(
            _CONTENT_DOMAIN
            + profile
            + salt
            + parent
            + struct.pack("<II", start, count)
            + struct.pack(f"<{count}I", *block)
        ).digest()
        result.append(parent)
        if len(result) > MAX_PREFIX_CACHE_BLOCKS_V3:
            raise ProofV3Error("prefix-cache block inventory is too large")
    return tuple(result)


def derive_prefix_cache_residual_lane_keys_v3(
    *,
    selected_layer_indices: Sequence[int],
    positions_by_layer: Sequence[tuple[int, Sequence[int]]],
    cached_token_count: int,
    block_token_count: int,
    hidden_row_width: int,
) -> tuple[tuple[int, str, int, int], ...]:
    """Bind cached projection rows to their pre-nonce residual checkpoints."""

    selected = tuple(int(layer) for layer in selected_layer_indices)
    if selected != tuple(sorted(set(selected))) or any(
        layer < 0 for layer in selected
    ):
        raise ProofV3Error(
            "prefix-cache residual layer inventory is noncanonical"
        )
    cached = _u32(cached_token_count, "cached_token_count", positive=True)
    width = _u32(block_token_count, "block_token_count", positive=True)
    row_width = _u32(hidden_row_width, "hidden_row_width", positive=True)
    try:
        positions = {
            int(layer): tuple(int(position) for position in values)
            for layer, values in positions_by_layer
        }
    except (TypeError, ValueError) as exc:
        raise ProofV3Error(
            "prefix-cache residual position inventory is malformed"
        ) from exc
    if tuple(positions) != tuple(sorted(positions)) or set(positions) != set(
        selected
    ):
        raise ProofV3Error(
            "prefix-cache residual position inventory is noncanonical"
        )
    keys = set()
    for layer in selected:
        layer_positions = positions[layer]
        if (
            layer_positions != tuple(sorted(set(layer_positions)))
            or any(position < 0 for position in layer_positions)
        ):
            raise ProofV3Error(
                "prefix-cache residual positions are noncanonical"
            )
        stages = [f"l{layer}.residual_out"]
        if layer > 0:
            stages.append(f"l{layer - 1}.residual_out")
        for position in layer_positions:
            if position >= cached:
                continue
            block_index, row_index = divmod(position, width)
            for stage_id in stages:
                lane_bytes = execution_anchor_lane_bytes_v3(stage_id)
                for lane_index in range(
                    (row_width + lane_bytes - 1) // lane_bytes
                ):
                    keys.add((
                        block_index,
                        stage_id,
                        row_index,
                        lane_index,
                    ))
    result = tuple(sorted(keys))
    if len(result) > MAX_PREFIX_CACHE_BLOCK_SAMPLES_V3:
        raise ProofV3Error(
            "prefix-cache residual lane inventory exceeds its bound"
        )
    return result


@dataclass(frozen=True, slots=True)
class PrefixCacheBlockRecordV3:
    """One logical cache block and its committed runtime-state root."""

    block_index: int
    token_start: int
    token_count: int
    content_digest: bytes
    state_root: bytes

    def __post_init__(self) -> None:
        index = _u32(self.block_index, "prefix-cache block index")
        start = _u32(self.token_start, "prefix-cache token start")
        count = _u32(self.token_count, "prefix-cache token count", positive=True)
        if count > MAX_PREFIX_CACHE_BLOCK_TOKENS_V3:
            raise ProofV3Error("prefix-cache block token count exceeds its bound")
        if start >= 1 << 32:
            raise ProofV3Error("prefix-cache block coordinates are noncanonical")
        _fixed32(self.content_digest, "prefix-cache content digest", nonzero=True)
        _fixed32(self.state_root, "prefix-cache state root", nonzero=True)

    def canonical_bytes(self) -> bytes:
        return (
            struct.pack("<III", self.block_index, self.token_start, self.token_count)
            + self.content_digest
            + self.state_root
        )


@dataclass(frozen=True, slots=True)
class PrefixCacheStateRecordV3:
    """One signed runtime stage committed beneath one logical cache block."""

    block_index: int
    stage_id: str
    row_count: int
    row_width: int
    value_root: bytes

    def __post_init__(self) -> None:
        _u32(self.block_index, "prefix-cache state block index")
        if (
            not isinstance(self.stage_id, str)
            or _STAGE_ID_RE.fullmatch(self.stage_id) is None
        ):
            raise ProofV3Error("prefix-cache state stage id is malformed")
        _u32(self.row_count, "prefix-cache state row count", positive=True)
        _u32(self.row_width, "prefix-cache state row width", positive=True)
        _fixed32(self.value_root, "prefix-cache state value root", nonzero=True)

    def canonical_bytes(self) -> bytes:
        stage = self.stage_id.encode("ascii")
        return (
            struct.pack("<IH", self.block_index, len(stage))
            + stage
            + struct.pack("<II", self.row_count, self.row_width)
            + self.value_root
        )


@dataclass(frozen=True, slots=True)
class PrefixCacheStateOpeningV3:
    records: tuple[PrefixCacheStateRecordV3, ...]
    paths: tuple[MerklePath, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.records, tuple)
            or not self.records
            or len(self.records) > MAX_PREFIX_CACHE_STATE_STAGES_V3
            or not all(
                isinstance(item, PrefixCacheStateRecordV3)
                for item in self.records
            )
            or not isinstance(self.paths, tuple)
            or len(self.paths) != len(self.records)
            or not all(isinstance(item, MerklePath) for item in self.paths)
        ):
            raise ProofV3Error("prefix-cache state opening is malformed")
        stage_ids = tuple(item.stage_id for item in self.records)
        if (
            stage_ids != tuple(sorted(stage_ids))
            or len(set(stage_ids)) != len(stage_ids)
            or len({item.block_index for item in self.records}) != 1
        ):
            raise ProofV3Error("prefix-cache state opening is noncanonical")
        indices = tuple(path.leaf_index for path in self.paths)
        if indices != tuple(sorted(indices)) or len(set(indices)) != len(indices):
            raise ProofV3Error("prefix-cache state paths are noncanonical")
        for path in self.paths:
            if any(
                is_left != bool((path.leaf_index >> level) & 1)
                for level, (_sibling, is_left) in enumerate(path.siblings)
            ):
                raise ProofV3Error(
                    "prefix-cache state path directions are noncanonical"
                )


@dataclass(frozen=True, slots=True)
class PrefixCacheRetainedMaterialV3:
    """Miner-retained trees needed to answer a later hard nonce."""

    commitment: PrefixCacheCommitmentV3
    block_records: tuple[PrefixCacheBlockRecordV3, ...]
    block_tree: MerkleTree
    state_records: tuple[tuple[PrefixCacheStateRecordV3, ...], ...]
    state_trees: tuple[MerkleTree, ...]
    base_capture_digest: bytes
    # Internal scheduler-owned read addresses for the temporary page lease.
    # They are never serialized, hashed, signed or accepted by a verifier.
    runtime_block_ids_by_layer: tuple[
        tuple[int, tuple[tuple[int, ...], ...]], ...
    ] = ()
    # Internal content-addressed residual row sources. They are committed
    # beneath ``state_trees`` but are never serialized as retained metadata.
    provenance_sources: tuple[
        tuple[int, str, ExecutionAnchorCommitmentV3, MerkleTree], ...
    ] = ()

    def __post_init__(self) -> None:
        _fixed32(
            self.base_capture_digest,
            "prefix-cache base capture digest",
            nonzero=True,
        )
        if (
            not isinstance(self.commitment, PrefixCacheCommitmentV3)
            or not isinstance(self.block_records, tuple)
            or len(self.block_records) != self.commitment.block_count
            or not isinstance(self.block_tree, MerkleTree)
            or self.block_tree.num_leaves != len(self.block_records)
            or self.block_tree.root != self.commitment.block_inventory_root
            or not isinstance(self.state_records, tuple)
            or not isinstance(self.state_trees, tuple)
            or len(self.state_records) != len(self.block_records)
            or len(self.state_trees) != len(self.block_records)
        ):
            raise ProofV3Error("prefix-cache retained material is malformed")
        for block_index, (block, records, tree) in enumerate(
            zip(
                self.block_records,
                self.state_records,
                self.state_trees,
                strict=True,
            )
        ):
            if (
                block.block_index != block_index
                or not isinstance(records, tuple)
                or not records
                or any(item.block_index != block_index for item in records)
                or not isinstance(tree, MerkleTree)
                or tree.num_leaves != len(records)
                or tree.root != block.state_root
            ):
                raise ProofV3Error(
                    "prefix-cache retained block material is inconsistent"
                )
        runtime = tuple(self.runtime_block_ids_by_layer)
        if runtime:
            layers = tuple(layer for layer, _blocks in runtime)
            if layers != tuple(sorted(set(layers))):
                raise ProofV3Error(
                    "prefix-cache runtime block inventory is noncanonical"
                )
            for layer, pages in runtime:
                flat = tuple(block for page in pages for block in page)
                if (
                    isinstance(layer, bool)
                    or not isinstance(layer, int)
                    or layer < 0
                    or not isinstance(pages, tuple)
                    or len(pages) != len(self.block_records)
                    or any(not isinstance(page, tuple) or not page for page in pages)
                    or len(set(flat)) != len(flat)
                    or any(
                        isinstance(block, bool)
                        or not isinstance(block, int)
                        or block < 0
                        for block in flat
                    )
                ):
                    raise ProofV3Error(
                        "prefix-cache runtime block mapping is malformed"
                    )
        sources = tuple(self.provenance_sources)
        source_keys = tuple(
            (block, stage)
            for block, stage, _commitment, _tree in sources
        )
        if source_keys != tuple(sorted(set(source_keys))):
            raise ProofV3Error(
                "prefix-cache provenance inventory is noncanonical"
            )
        records_by_key = {
            (record.block_index, record.stage_id): record
            for records in self.state_records
            for record in records
        }
        for block, stage, commitment, tree in sources:
            record = records_by_key.get((block, stage))
            if (
                type(block) is not int
                or not 0 <= block < len(self.block_records)
                or not isinstance(stage, str)
                or not isinstance(commitment, ExecutionAnchorCommitmentV3)
                or not isinstance(tree, MerkleTree)
                or record is None
                or commitment.stage_id != stage
                or commitment.row_count != record.row_count
                or commitment.row_width != record.row_width
                or commitment.root != record.value_root
                or tree.num_leaves != record.row_count
                or tree.root != record.value_root
            ):
                raise ProofV3Error(
                    "prefix-cache provenance material is inconsistent"
                )

    @property
    def retained_bytes(self) -> int:
        return (
            len(self.block_tree.tree_data_compact)
            + sum(len(tree.tree_data_compact) for tree in self.state_trees)
            + sum(len(item.canonical_bytes()) for item in self.block_records)
            + sum(
                len(item.canonical_bytes())
                for records in self.state_records
                for item in records
            )
            + sum(
                len(tree.tree_data_compact)
                for _block, _stage, _commitment, tree
                in self.provenance_sources
            )
        )


@dataclass(frozen=True, slots=True)
class PrefixCacheCommitmentV3:
    """Bounded request-local commitment to the cached prompt prefix."""

    execution_profile_digest: bytes
    prompt_token_root: bytes
    cache_salt_digest: bytes
    executed_suffix_digest: bytes
    block_inventory_root: bytes
    gdn_boundary_root: bytes | None
    context_token_count: int
    cached_token_count: int
    block_token_count: int
    block_count: int

    def __post_init__(self) -> None:
        _fixed32(
            self.execution_profile_digest,
            "execution_profile_digest",
            nonzero=True,
        )
        _fixed32(self.prompt_token_root, "prompt_token_root", nonzero=True)
        _fixed32(self.cache_salt_digest, "cache_salt_digest")
        _fixed32(
            self.executed_suffix_digest,
            "executed_suffix_digest",
            nonzero=True,
        )
        _fixed32(
            self.block_inventory_root,
            "block_inventory_root",
            nonzero=True,
        )
        if self.gdn_boundary_root is not None:
            _fixed32(self.gdn_boundary_root, "gdn_boundary_root", nonzero=True)
        context = _u32(
            self.context_token_count,
            "context_token_count",
            positive=True,
        )
        cached = _u32(
            self.cached_token_count,
            "cached_token_count",
            positive=True,
        )
        width = _u32(
            self.block_token_count,
            "block_token_count",
            positive=True,
        )
        count = _u32(self.block_count, "block_count", positive=True)
        if count > MAX_PREFIX_CACHE_BLOCKS_V3:
            raise ProofV3Error("prefix-cache block inventory is too large")
        if (
            cached > context
            or cached > count * width
            or cached <= (count - 1) * width
        ):
            raise ProofV3Error("prefix-cache cached range is inconsistent")

    def canonical_bytes(self) -> bytes:
        gdn = bytes(32) if self.gdn_boundary_root is None else self.gdn_boundary_root
        return (
            PREFIX_CACHE_BLOCK_COMMITMENT_ABI_V3.encode("ascii")
            + self.execution_profile_digest
            + self.prompt_token_root
            + self.cache_salt_digest
            + self.executed_suffix_digest
            + self.block_inventory_root
            + gdn
            + struct.pack(
                "<IIII",
                self.context_token_count,
                self.cached_token_count,
                self.block_token_count,
                self.block_count,
            )
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_COMMITMENT_DOMAIN + self.canonical_bytes()).digest()


@dataclass(frozen=True, slots=True)
class PrefixCacheBlockOpeningV3:
    records: tuple[PrefixCacheBlockRecordV3, ...]
    paths: tuple[MerklePath, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.records, tuple)
            or not self.records
            or len(self.records) > MAX_PREFIX_CACHE_BLOCK_SAMPLES_V3
            or not all(isinstance(item, PrefixCacheBlockRecordV3) for item in self.records)
            or not isinstance(self.paths, tuple)
            or len(self.paths) != len(self.records)
            or not all(isinstance(item, MerklePath) for item in self.paths)
        ):
            raise ProofV3Error("prefix-cache block opening is malformed")
        indices = tuple(item.block_index for item in self.records)
        if indices != tuple(sorted(indices)) or len(set(indices)) != len(indices):
            raise ProofV3Error("prefix-cache block opening is noncanonical")
        if any(path.leaf_index != index for path, index in zip(self.paths, indices)):
            raise ProofV3Error("prefix-cache block path index is inconsistent")
        for path in self.paths:
            if any(
                is_left != bool((path.leaf_index >> level) & 1)
                for level, (_sibling, is_left) in enumerate(path.siblings)
            ):
                raise ProofV3Error(
                    "prefix-cache block path directions are noncanonical"
                )


@dataclass(frozen=True, slots=True)
class PrefixCacheLaneRevealV3:
    """One replay-derived lane opened against a frozen cache-stage root."""

    block_index: int
    stage_id: str
    opening: ExecutionAnchorLaneOpeningV3

    def __post_init__(self) -> None:
        _u32(self.block_index, "prefix-cache lane block index")
        if (
            not isinstance(self.stage_id, str)
            or _STAGE_ID_RE.fullmatch(self.stage_id) is None
            or not isinstance(self.opening, ExecutionAnchorLaneOpeningV3)
        ):
            raise ProofV3Error("prefix-cache lane reveal is malformed")


@dataclass(frozen=True, slots=True)
class PrefixCachePostnonceProofV3:
    """Canonical cache-hit section appended only to cache-aware hard proofs."""

    commitment: PrefixCacheCommitmentV3
    base_capture_digest: bytes
    block_opening: PrefixCacheBlockOpeningV3
    state_openings: tuple[PrefixCacheStateOpeningV3, ...]
    lane_reveals: tuple[PrefixCacheLaneRevealV3, ...]

    def __post_init__(self) -> None:
        _fixed32(
            self.base_capture_digest,
            "prefix-cache base capture digest",
            nonzero=True,
        )
        if (
            not isinstance(self.commitment, PrefixCacheCommitmentV3)
            or not isinstance(self.block_opening, PrefixCacheBlockOpeningV3)
            or not isinstance(self.state_openings, tuple)
            or len(self.state_openings) != len(self.block_opening.records)
            or not all(
                isinstance(item, PrefixCacheStateOpeningV3)
                for item in self.state_openings
            )
            or not isinstance(self.lane_reveals, tuple)
            or not self.lane_reveals
            or not all(
                isinstance(item, PrefixCacheLaneRevealV3)
                for item in self.lane_reveals
            )
        ):
            raise ProofV3Error("prefix-cache post-nonce proof is malformed")
        block_indices = tuple(
            opening.records[0].block_index for opening in self.state_openings
        )
        if block_indices != tuple(
            record.block_index for record in self.block_opening.records
        ):
            raise ProofV3Error(
                "prefix-cache state openings do not match the opened blocks"
            )
        keys = tuple(
            (
                item.block_index,
                item.stage_id,
                item.opening.row_index,
                item.opening.lane_index,
            )
            for item in self.lane_reveals
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ProofV3Error("prefix-cache lane reveals are noncanonical")


def build_prefix_cache_commitment_v3(
    *,
    records: Sequence[PrefixCacheBlockRecordV3],
    execution_profile_digest: bytes,
    prompt_token_root: bytes,
    cache_salt_digest: bytes,
    executed_suffix_digest: bytes,
    context_token_count: int,
    gdn_boundary_root: bytes | None = None,
    block_token_count: int | None = None,
) -> tuple[PrefixCacheCommitmentV3, MerkleTree]:
    """Build the pre-nonce block inventory and its request commitment."""

    ordered = tuple(records)
    if not ordered or len(ordered) > MAX_PREFIX_CACHE_BLOCKS_V3:
        raise ProofV3Error("prefix-cache block inventory is empty or too large")
    width = (
        max(item.token_count for item in ordered)
        if block_token_count is None
        else _u32(block_token_count, "block_token_count", positive=True)
    )
    if width > MAX_PREFIX_CACHE_BLOCK_TOKENS_V3:
        raise ProofV3Error("prefix-cache block token count exceeds its bound")
    cursor = 0
    for index, item in enumerate(ordered):
        if (
            item.block_index != index
            or item.token_start != cursor
            or item.token_count > width
            or (index + 1 < len(ordered) and item.token_count != width)
        ):
            raise ProofV3Error("prefix-cache block inventory is not contiguous")
        cursor += item.token_count
    indices = tuple(item.block_index for item in ordered)
    if indices != tuple(range(len(ordered))):
        raise ProofV3Error("prefix-cache block inventory is not contiguous")
    tree = MerkleTree([item.canonical_bytes() for item in ordered])
    commitment = PrefixCacheCommitmentV3(
        execution_profile_digest=execution_profile_digest,
        prompt_token_root=prompt_token_root,
        cache_salt_digest=cache_salt_digest,
        executed_suffix_digest=executed_suffix_digest,
        block_inventory_root=tree.root,
        gdn_boundary_root=gdn_boundary_root,
        context_token_count=context_token_count,
        cached_token_count=cursor,
        block_token_count=width,
        block_count=len(ordered),
    )
    return commitment, tree


def build_prefix_cache_state_tree_v3(
    records: Sequence[PrefixCacheStateRecordV3],
) -> tuple[tuple[PrefixCacheStateRecordV3, ...], MerkleTree]:
    """Build one block's exact signed stage inventory."""

    ordered = tuple(records)
    if not ordered or len(ordered) > MAX_PREFIX_CACHE_STATE_STAGES_V3:
        raise ProofV3Error("prefix-cache state inventory is empty or too large")
    if len({item.block_index for item in ordered}) != 1:
        raise ProofV3Error("prefix-cache state inventory crosses logical blocks")
    stage_ids = tuple(item.stage_id for item in ordered)
    if (
        stage_ids != tuple(sorted(stage_ids))
        or len(stage_ids) != len(set(stage_ids))
    ):
        raise ProofV3Error("prefix-cache state inventory is noncanonical")
    return ordered, MerkleTree([item.canonical_bytes() for item in ordered])


def build_prefix_cache_request_material_v3(
    *,
    execution_profile_digest: bytes,
    base_capture_digest: bytes,
    prompt_token_root: bytes,
    cache_salt_digest: bytes,
    executed_suffix_digest: bytes,
    token_ids: Sequence[int],
    block_token_count: int,
    cached_token_count: int,
    state_records: Sequence[Sequence[PrefixCacheStateRecordV3]],
    state_trees: Sequence[MerkleTree],
    provenance_sources: Sequence[
        tuple[int, str, ExecutionAnchorCommitmentV3, MerkleTree]
    ] = (),
    runtime_block_ids_by_layer: Sequence[
        tuple[int, Sequence[Sequence[int]]]
    ] = (),
) -> PrefixCacheRetainedMaterialV3:
    """Assemble the complete pre-nonce cache lane from captured state roots."""

    records_by_block = tuple(tuple(items) for items in state_records)
    trees = tuple(state_trees)
    if (
        not records_by_block
        or len(records_by_block) != len(trees)
        or len(records_by_block) > MAX_PREFIX_CACHE_BLOCKS_V3
    ):
        raise ProofV3Error("prefix-cache captured state inventory is malformed")
    tokens = tuple(token_ids)
    cached = _u32(cached_token_count, "cached_token_count", positive=True)
    width = _u32(block_token_count, "block_token_count", positive=True)
    if (
        cached > len(tokens)
        or cached > len(records_by_block) * width
        or cached <= (len(records_by_block) - 1) * width
    ):
        raise ProofV3Error("prefix-cache captured token range is inconsistent")
    content = derive_prefix_cache_content_digests_v3(
        execution_profile_digest=execution_profile_digest,
        cache_salt_digest=cache_salt_digest,
        token_ids=tokens[:cached],
        block_token_count=width,
    )
    if len(content) < len(records_by_block):
        raise ProofV3Error("prefix-cache prompt does not cover captured blocks")
    block_records = []
    for block_index, (items, tree) in enumerate(
        zip(records_by_block, trees, strict=True)
    ):
        if (
            not items
            or any(item.block_index != block_index for item in items)
            or not isinstance(tree, MerkleTree)
            or tree.num_leaves != len(items)
            or tree.root == bytes(32)
        ):
            raise ProofV3Error(
                "prefix-cache captured state block is inconsistent"
            )
        block_records.append(
            PrefixCacheBlockRecordV3(
                block_index=block_index,
                token_start=block_index * width,
                token_count=min(width, cached - block_index * width),
                content_digest=content[block_index],
                state_root=tree.root,
            )
        )
    commitment, block_tree = build_prefix_cache_commitment_v3(
        records=tuple(block_records),
        execution_profile_digest=execution_profile_digest,
        prompt_token_root=prompt_token_root,
        cache_salt_digest=cache_salt_digest,
        executed_suffix_digest=executed_suffix_digest,
        context_token_count=len(tokens),
        block_token_count=width,
    )
    return PrefixCacheRetainedMaterialV3(
        commitment=commitment,
        block_records=tuple(block_records),
        block_tree=block_tree,
        state_records=records_by_block,
        state_trees=trees,
        base_capture_digest=base_capture_digest,
        runtime_block_ids_by_layer=tuple(
            (layer, tuple(tuple(page) for page in pages))
            for layer, pages in runtime_block_ids_by_layer
        ),
        provenance_sources=tuple(provenance_sources),
    )


def derive_prefix_cache_block_indices_v3(
    *,
    validator_nonce: bytes,
    commitment: PrefixCacheCommitmentV3,
    count: int,
) -> tuple[int, ...]:
    """Select distinct cache blocks after the request commitment is frozen."""

    nonce = _fixed32(validator_nonce, "validator_nonce", nonzero=True)
    if not isinstance(commitment, PrefixCacheCommitmentV3):
        raise ProofV3Error("prefix-cache commitment has an unexpected type")
    samples = _u32(count, "prefix-cache sample count", positive=True)
    if samples > MAX_PREFIX_CACHE_BLOCK_SAMPLES_V3:
        raise ProofV3Error("prefix-cache sample count exceeds its bound")
    target = min(samples, commitment.block_count)
    seed = hashlib.sha256(
        _CHALLENGE_DOMAIN + nonce + commitment.digest()
    ).digest()
    selected: set[int] = set()
    counter = 0
    while len(selected) < target:
        block = hashlib.sha256(seed + struct.pack("<I", counter)).digest()
        for offset in range(0, 32, 8):
            selected.add(
                int.from_bytes(block[offset : offset + 8], "little")
                % commitment.block_count
            )
            if len(selected) == target:
                break
        counter += 1
    return tuple(sorted(selected))


def derive_prefix_cache_lane_keys_v3(
    *,
    validator_nonce: bytes,
    commitment: PrefixCacheCommitmentV3,
    state_records: Sequence[PrefixCacheStateRecordV3],
    rows_per_stage: int = PREFIX_CACHE_ROWS_PER_STAGE_V3,
    lanes_per_row: int = PREFIX_CACHE_LANES_PER_ROW_V3,
) -> tuple[tuple[int, str, int, int], ...]:
    """Derive exact cache row/lane coordinates after the nonce is known."""

    nonce = _fixed32(validator_nonce, "validator_nonce", nonzero=True)
    if not isinstance(commitment, PrefixCacheCommitmentV3):
        raise ProofV3Error("prefix-cache commitment has an unexpected type")
    row_samples = _u32(
        rows_per_stage,
        "prefix-cache rows per stage",
        positive=True,
    )
    lane_samples = _u32(
        lanes_per_row,
        "prefix-cache lanes per row",
        positive=True,
    )
    if row_samples > 16 or lane_samples > 16:
        raise ProofV3Error("prefix-cache lane sample count exceeds its bound")
    records = tuple(state_records)
    if (
        not records
        or len(records) > MAX_PREFIX_CACHE_STATE_STAGES_V3
        or tuple(
            (item.block_index, item.stage_id) for item in records
        )
        != tuple(sorted(
            (item.block_index, item.stage_id) for item in records
        ))
        or len({(item.block_index, item.stage_id) for item in records})
        != len(records)
    ):
        raise ProofV3Error("prefix-cache challenged state inventory is noncanonical")

    result: list[tuple[int, str, int, int]] = []
    for record in records:
        lane_width = execution_anchor_lane_bytes_v3(record.stage_id)
        lane_count = (record.row_width + lane_width - 1) // lane_width
        seed = hashlib.sha256(
            _LANE_CHALLENGE_DOMAIN
            + nonce
            + commitment.digest()
            + record.canonical_bytes()
        ).digest()

        def _draw_distinct(*, count: int, modulus: int, domain: bytes):
            target = min(count, modulus)
            chosen: set[int] = set()
            counter = 0
            while len(chosen) < target:
                digest = hashlib.sha256(
                    seed + domain + struct.pack("<I", counter)
                ).digest()
                for offset in range(0, 32, 8):
                    chosen.add(
                        int.from_bytes(
                            digest[offset:offset + 8], "little"
                        ) % modulus
                    )
                    if len(chosen) == target:
                        break
                counter += 1
            return tuple(sorted(chosen))

        rows = _draw_distinct(
            count=row_samples,
            modulus=record.row_count,
            domain=b"/ROW",
        )
        for row_index in rows:
            lanes = _draw_distinct(
                count=lane_samples,
                modulus=lane_count,
                domain=b"/LANE" + struct.pack("<I", row_index),
            )
            result.extend(
                (record.block_index, record.stage_id, row_index, lane_index)
                for lane_index in lanes
            )
    return tuple(sorted(result))


def prefix_cache_stage_ids_for_layers_v3(
    *,
    layer_caches: Sequence[object],
    selected_layer_indices: Sequence[int],
) -> tuple[str, ...]:
    """Map signed cache families to physical cache stages opened post-nonce.

    Attention K/V pages are persistent paged values and are opened here.  GDN
    cache state is deliberately absent: hybrid vLLM retains only the final
    recurrent boundary, while compact-v9 binds that boundary through the
    request's pre-nonce prompt-boundary anchor and salted full-prompt replay.
    """

    selected = tuple(sorted(int(layer) for layer in selected_layer_indices))
    if not selected or len(selected) != len(set(selected)):
        raise ProofV3Error("prefix-cache selected layer inventory is malformed")
    by_layer = {}
    for family in tuple(layer_caches):
        try:
            layer = int(family.layer_index)
            kind = str(family.cache_kind)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ProofV3Error(
                "prefix-cache signed family inventory is malformed"
            ) from exc
        if layer in by_layer:
            raise ProofV3Error(
                "prefix-cache signed family inventory contains duplicates"
            )
        by_layer[layer] = kind
    result = []
    for layer in selected:
        kind = by_layer.get(layer)
        if kind == "attention_kv":
            result.extend((
                f"l{layer}.attention_k_cache",
                f"l{layer}.attention_v_cache",
            ))
        elif kind == "gdn_state":
            continue
        else:
            raise ProofV3Error(
                "prefix-cache selected layer lacks a signed cache family"
            )
    return tuple(sorted(result))


def derive_prefix_cache_projection_lane_keys_v3(
    *,
    challenge,
    positions_by_layer,
    kv_dims_by_layer,
    cached_token_count: int,
    block_token_count: int,
) -> tuple[tuple[int, str, int, int], ...]:
    """Derive cache lanes used by the registered-weight K/V corridor."""

    cached = int(cached_token_count)
    block_tokens = int(block_token_count)
    if cached <= 0 or block_tokens <= 0:
        raise ProofV3Error("prefix-cache projection geometry is malformed")
    positions = {
        int(layer): tuple(int(position) for position in layer_positions)
        for layer, layer_positions in tuple(positions_by_layer)
    }
    dimensions = {
        int(layer): int(width)
        for layer, width in tuple(kv_dims_by_layer)
    }
    if (
        not positions
        or set(positions) != set(dimensions)
        or any(
            layer_positions != tuple(sorted(set(layer_positions)))
            or any(position < 0 for position in layer_positions)
            for layer_positions in positions.values()
        )
        or any(width <= 0 for width in dimensions.values())
    ):
        raise ProofV3Error("prefix-cache projection inventory is malformed")
    from verallm.proof_v3.execution_anchor import (
        execution_anchor_lane_bytes_v3,
    )

    result = set()
    for layer, layer_positions in sorted(positions.items()):
        width = dimensions[layer]
        columns = tuple(
            int(column)
            for column in challenge.kv_cols_for(
                layer_index=layer,
                kv_dim=width,
            )
        )
        if (
            not columns
            or columns != tuple(sorted(set(columns)))
            or columns[-1] >= width
        ):
            raise ProofV3Error(
                "prefix-cache projection columns are malformed"
            )
        for position in layer_positions:
            if position >= cached:
                continue
            block_index, row_index = divmod(position, block_tokens)
            for tag in ("k", "v"):
                stage_id = f"l{layer}.attention_{tag}_cache"
                lane_bytes = execution_anchor_lane_bytes_v3(stage_id)
                for column in columns:
                    result.add((
                        block_index,
                        stage_id,
                        row_index,
                        (column * 2) // lane_bytes,
                    ))
    return tuple(sorted(result))


def derive_prefix_cache_projection_head_lane_keys_v3(
    *,
    projection_heads,
    head_dim: int,
    cached_token_count: int,
    block_token_count: int,
) -> tuple[tuple[int, str, int, int], ...]:
    """Expand nonce-selected projection heads to exact cache-page lanes."""

    try:
        heads = tuple(
            (int(layer), int(head), int(position))
            for layer, head, position in tuple(projection_heads)
        )
    except (TypeError, ValueError) as exc:
        raise ProofV3Error(
            "prefix-cache projection-head lane plan is malformed"
        ) from exc
    cached = int(cached_token_count)
    block_tokens = int(block_token_count)
    width = int(head_dim)
    if (
        not heads
        or heads != tuple(sorted(set(heads)))
        or width <= 0
        or cached <= 0
        or block_tokens <= 0
        or any(
            layer < 0
            or head < 0
            or position < 0
            or position >= cached
            for layer, head, position in heads
        )
    ):
        raise ProofV3Error(
            "prefix-cache projection-head lane plan is malformed"
        )
    from verallm.proof_v3.attention_anchor_binding import (
        required_execution_anchor_lanes_v3,
    )
    from verallm.proof_v3.execution_anchor import (
        execution_anchor_lane_bytes_v3,
    )

    result = set()
    for layer, head, position in heads:
        block, row = divmod(position, block_tokens)
        for tag in ("k", "v"):
            stage_id = f"l{layer}.attention_{tag}_cache"
            lane_bytes = execution_anchor_lane_bytes_v3(stage_id)
            result.update(
                (block, stage_id, row, lane)
                for lane in required_execution_anchor_lanes_v3(
                    byte_start=head * width * 2,
                    byte_length=width * 2,
                    lane_bytes=lane_bytes,
                )
            )
    return tuple(sorted(result))


def fold_prefix_cache_commitment_v3(
    *,
    base_capture_digest: bytes,
    commitment: PrefixCacheCommitmentV3,
) -> bytes:
    """Fold the cache lane into the authenticated pre-nonce capture chain."""

    base = _fixed32(base_capture_digest, "base_capture_digest", nonzero=True)
    if not isinstance(commitment, PrefixCacheCommitmentV3):
        raise ProofV3Error("prefix-cache commitment has an unexpected type")
    return hashlib.sha256(_FOLD_DOMAIN + base + commitment.digest()).digest()


def prefix_cache_attention_anchor_root_v3(
    *, commitment: PrefixCacheCommitmentV3, layer_index: int
) -> bytes:
    """Derive the layer-separated root used by attention equality Fiat-Shamir."""

    if not isinstance(commitment, PrefixCacheCommitmentV3):
        raise ProofV3Error("prefix-cache commitment has an unexpected type")
    layer = _u32(layer_index, "prefix-cache attention layer")
    return hashlib.sha256(
        b"VERATHOS/PROOF_V3/PREFIX_CACHE/ATTENTION_ANCHOR/V1"
        + commitment.digest()
        + struct.pack("<I", layer)
    ).digest()


def derive_prefix_cache_executed_suffix_digest_v3(
    execution_anchors: Sequence[object],
) -> bytes:
    """Bind the exact ordered suffix-anchor inventory into the cache lane."""

    anchors = tuple(execution_anchors)
    if not anchors or len(anchors) >= 1 << 16:
        raise ProofV3Error("prefix-cache executed suffix inventory is malformed")
    encoded = []
    for item in anchors:
        canonical = getattr(item, "canonical_bytes", None)
        if not callable(canonical):
            raise ProofV3Error(
                "prefix-cache executed suffix anchor is malformed"
            )
        value = canonical()
        if not isinstance(value, bytes) or not value:
            raise ProofV3Error(
                "prefix-cache executed suffix anchor is malformed"
            )
        encoded.append(struct.pack("<I", len(value)) + value)
    return hashlib.sha256(
        _SUFFIX_DOMAIN + struct.pack("<H", len(encoded)) + b"".join(encoded)
    ).digest()


def open_prefix_cache_blocks_v3(
    *,
    records: Sequence[PrefixCacheBlockRecordV3],
    tree: MerkleTree,
    indices: Sequence[int],
) -> PrefixCacheBlockOpeningV3:
    ordered = tuple(records)
    selected = tuple(indices)
    if (
        selected != tuple(sorted(selected))
        or len(set(selected)) != len(selected)
        or not selected
        or any(index < 0 or index >= len(ordered) for index in selected)
        or tree.num_leaves != len(ordered)
    ):
        raise ProofV3Error("prefix-cache opening indices are invalid")
    return PrefixCacheBlockOpeningV3(
        records=tuple(ordered[index] for index in selected),
        paths=tuple(tree.get_path(index) for index in selected),
    )


def open_prefix_cache_state_v3(
    *,
    records: Sequence[PrefixCacheStateRecordV3],
    tree: MerkleTree,
    stage_ids: Sequence[str],
) -> PrefixCacheStateOpeningV3:
    ordered = tuple(records)
    selected = tuple(stage_ids)
    inventory = tuple(item.stage_id for item in ordered)
    if (
        selected != tuple(sorted(selected))
        or len(set(selected)) != len(selected)
        or not selected
        or tree.num_leaves != len(ordered)
        or any(stage_id not in inventory for stage_id in selected)
    ):
        raise ProofV3Error("prefix-cache state opening stages are invalid")
    indices = tuple(inventory.index(stage_id) for stage_id in selected)
    return PrefixCacheStateOpeningV3(
        records=tuple(ordered[index] for index in indices),
        paths=tuple(tree.get_path(index) for index in indices),
    )


def build_prefix_cache_postnonce_proof_v3(
    *,
    retained: PrefixCacheRetainedMaterialV3,
    selection_seed: bytes,
    layer_caches: Sequence[object],
    selected_layer_indices: Sequence[int],
    runtime_rows: Sequence[tuple[int, str, Sequence[bytes]]],
    provenance_rows: Sequence[tuple[int, str, int, bytes]] = (),
    required_lane_keys: Sequence[tuple[int, str, int, int]] = (),
) -> PrefixCachePostnonceProofV3:
    """Open nonce-selected leased cache rows against their frozen roots.

    ``runtime_rows`` is miner-local material read from scheduler-leased pages.
    Physical page identifiers never enter this function or its result. Every
    reconstructed row tree must equal the corresponding pre-nonce stage root
    before an opening can be emitted.
    """

    if not isinstance(retained, PrefixCacheRetainedMaterialV3):
        raise ProofV3Error(
            "prefix-cache retained material has an unexpected type"
        )
    base_indices = derive_prefix_cache_block_indices_v3(
        validator_nonce=selection_seed,
        commitment=retained.commitment,
        count=PREFIX_CACHE_BLOCK_SAMPLE_COUNT_V3,
    )
    stage_ids = prefix_cache_stage_ids_for_layers_v3(
        layer_caches=layer_caches,
        selected_layer_indices=selected_layer_indices,
    )
    records_by_all_keys = {
        (record.block_index, record.stage_id): record
        for records in retained.state_records
        for record in records
    }
    requested = _canonical_required_lane_keys_v3(required_lane_keys)
    if (
        any(
            key[:2] not in records_by_all_keys
            or key[2] < 0
            or key[2] >= records_by_all_keys[key[:2]].row_count
            or key[3] < 0
            for key in requested
        )
    ):
        raise ProofV3Error("prefix-cache required lane plan is malformed")
    indices = tuple(sorted(
        set(base_indices) | {key[0] for key in requested}
    ))
    block_opening = open_prefix_cache_blocks_v3(
        records=retained.block_records,
        tree=retained.block_tree,
        indices=indices,
    )
    opened_stage_ids = tuple(sorted(
        set(stage_ids) | {key[1] for key in requested}
    ))
    state_openings = tuple(
        open_prefix_cache_state_v3(
            records=retained.state_records[index],
            tree=retained.state_trees[index],
            stage_ids=opened_stage_ids,
        )
        for index in indices
    )
    records_by_key = {
        (record.block_index, record.stage_id): record
        for opening in state_openings
        for record in opening.records
    }
    base_lane_keys = derive_prefix_cache_lane_keys_v3(
        validator_nonce=selection_seed,
        commitment=retained.commitment,
        state_records=tuple(
            record
            for opening in state_openings
            if opening.records[0].block_index in base_indices
            for record in opening.records
            if record.stage_id in stage_ids
        ),
    )
    lane_keys = tuple(sorted(set(base_lane_keys) | set(requested)))
    provenance_sources = {
        (block, stage): (commitment, tree)
        for block, stage, commitment, tree
        in retained.provenance_sources
    }
    expected_keys = tuple(sorted(
        {key[:2] for key in lane_keys}
        - set(provenance_sources)
    ))
    supplied = tuple(runtime_rows)
    keys = tuple((int(block), str(stage)) for block, stage, _rows in supplied)
    if keys != expected_keys or len(keys) != len(set(keys)):
        raise ProofV3Error(
            "prefix-cache runtime row inventory does not match the challenge"
        )
    rows_by_key: dict[tuple[int, str], tuple[bytes, ...]] = {}
    sources = dict(provenance_sources)
    from verallm.proof_v3.execution_anchor import (
        build_execution_anchor_lane_opening_v3,
        build_execution_anchor_tree_v3,
    )

    for block_index, stage_id, rows in supplied:
        key = (int(block_index), str(stage_id))
        record = records_by_key[key]
        canonical_rows = tuple(rows)
        if (
            len(canonical_rows) != record.row_count
            or any(
                not isinstance(row, bytes) or len(row) != record.row_width
                for row in canonical_rows
            )
        ):
            raise ProofV3Error(
                "prefix-cache runtime row geometry changed after capture"
            )
        commitment, tree = build_execution_anchor_tree_v3(
            stage_id=stage_id,
            rows=canonical_rows,
        )
        if (
            commitment.row_count != record.row_count
            or commitment.row_width != record.row_width
            or commitment.root != record.value_root
        ):
            raise ProofV3Error(
                "prefix-cache leased page changed after pre-nonce capture"
            )
        rows_by_key[key] = canonical_rows
        sources[key] = (commitment, tree)

    provenance_values = tuple(provenance_rows)
    provenance_keys = tuple(
        (int(block), str(stage), int(row))
        for block, stage, row, _value in provenance_values
    )
    expected_provenance_keys = tuple(sorted({
        key[:3]
        for key in lane_keys
        if key[:2] in provenance_sources
    }))
    if (
        provenance_keys != expected_provenance_keys
        or len(provenance_keys) != len(set(provenance_keys))
    ):
        raise ProofV3Error(
            "prefix-cache provenance row inventory does not match the challenge"
        )
    provenance_by_key = {
        (int(block), str(stage), int(row)): bytes(value)
        for block, stage, row, value in provenance_values
    }

    reveals = []
    for block_index, stage_id, row_index, lane_index in lane_keys:
        key = (block_index, stage_id)
        commitment, tree = sources[key]
        if key in provenance_sources:
            row_bytes = provenance_by_key[
                (block_index, stage_id, row_index)
            ]
        else:
            row_bytes = rows_by_key[key][row_index]
        opening = build_execution_anchor_lane_opening_v3(
            commitment=commitment,
            row_index=row_index,
            row_bytes=row_bytes,
            row_tree=tree,
            lane_index=lane_index,
        )
        try:
            verify_execution_anchor_lane_v3(
                commitment=commitment,
                opening=opening,
            )
        except ProofV3VerificationError as exc:
            raise ProofV3Error(
                "prefix-cache provenance row disagrees with its pre-nonce root"
            ) from exc
        reveals.append(PrefixCacheLaneRevealV3(
            block_index=block_index,
            stage_id=stage_id,
            opening=opening,
        ))
    return PrefixCachePostnonceProofV3(
        commitment=retained.commitment,
        base_capture_digest=retained.base_capture_digest,
        block_opening=block_opening,
        state_openings=state_openings,
        lane_reveals=tuple(reveals),
    )


def verify_prefix_cache_block_opening_v3(
    *,
    commitment: PrefixCacheCommitmentV3,
    opening: PrefixCacheBlockOpeningV3,
    expected_indices: Sequence[int],
    expected_content_digests: Sequence[bytes],
) -> None:
    """Verify selected logical records without trusting physical cache IDs."""

    if not isinstance(commitment, PrefixCacheCommitmentV3):
        raise ProofV3VerificationError(
            "prefix-cache commitment has an unexpected type"
        )
    if not isinstance(opening, PrefixCacheBlockOpeningV3):
        raise ProofV3VerificationError("prefix-cache opening has an unexpected type")
    indices = tuple(expected_indices)
    if indices != tuple(record.block_index for record in opening.records):
        raise ProofV3VerificationError(
            "prefix-cache opening does not match the nonce-derived blocks"
        )
    content = tuple(expected_content_digests)
    if len(content) != len(indices):
        raise ProofV3VerificationError(
            "prefix-cache expected content inventory is malformed"
        )
    for record, path, expected_digest in zip(
        opening.records,
        opening.paths,
        content,
        strict=True,
    ):
        expected_count = min(
            commitment.block_token_count,
            commitment.cached_token_count
            - record.block_index * commitment.block_token_count,
        )
        if (
            record.token_start
            != record.block_index * commitment.block_token_count
            or record.token_count != expected_count
        ):
            raise ProofV3VerificationError("prefix-cache block width is inconsistent")
        if record.content_digest != expected_digest:
            raise ProofV3VerificationError(
                "prefix-cache block does not match the prompt prefix"
            )
        if not verify_merkle_path(
            commitment.block_inventory_root,
            record.canonical_bytes(),
            path,
        ):
            raise ProofV3VerificationError(
                "prefix-cache block opening does not match the commitment"
            )


def verify_prefix_cache_state_opening_v3(
    *,
    block_record: PrefixCacheBlockRecordV3,
    opening: PrefixCacheStateOpeningV3,
    expected_stage_ids: Sequence[str],
) -> None:
    """Authenticate nonce-selected cache stages beneath one opened block."""

    if not isinstance(block_record, PrefixCacheBlockRecordV3):
        raise ProofV3VerificationError(
            "prefix-cache block record has an unexpected type"
        )
    if not isinstance(opening, PrefixCacheStateOpeningV3):
        raise ProofV3VerificationError(
            "prefix-cache state opening has an unexpected type"
        )
    expected = tuple(expected_stage_ids)
    actual = tuple(item.stage_id for item in opening.records)
    if actual != expected:
        raise ProofV3VerificationError(
            "prefix-cache state opening does not match the signed challenge"
        )
    for record, path in zip(opening.records, opening.paths, strict=True):
        if record.block_index != block_record.block_index:
            raise ProofV3VerificationError(
                "prefix-cache state opening crosses logical blocks"
            )
        if not verify_merkle_path(
            block_record.state_root,
            record.canonical_bytes(),
            path,
        ):
            raise ProofV3VerificationError(
                "prefix-cache state opening does not match the block root"
            )


def verify_prefix_cache_lane_reveals_v3(
    *,
    commitment: PrefixCacheCommitmentV3,
    state_openings: Sequence[PrefixCacheStateOpeningV3],
    lane_reveals: Sequence[PrefixCacheLaneRevealV3],
    validator_nonce: bytes,
    base_stage_ids: Sequence[str] | None = None,
    required_lane_keys: Sequence[tuple[int, str, int, int]] = (),
) -> tuple[tuple[tuple[int, str, int, int], bytes], ...]:
    """Verify replay lanes against the exact frozen cache-stage roots."""

    records = tuple(
        record for opening in state_openings for record in opening.records
    )
    allowed_base_stages = (
        None if base_stage_ids is None else set(base_stage_ids)
    )
    base_expected = derive_prefix_cache_lane_keys_v3(
        validator_nonce=validator_nonce,
        commitment=commitment,
        state_records=tuple(
            record
            for opening in state_openings
            if opening.records[0].block_index in derive_prefix_cache_block_indices_v3(
                validator_nonce=validator_nonce,
                commitment=commitment,
                count=PREFIX_CACHE_BLOCK_SAMPLE_COUNT_V3,
            )
            for record in opening.records
            if (
                allowed_base_stages is None
                or record.stage_id in allowed_base_stages
            )
        ),
    )
    required = _canonical_required_lane_keys_v3(
        required_lane_keys,
        verification=True,
    )
    if any(key[0] >= commitment.block_count for key in required):
        raise ProofV3VerificationError(
            "prefix-cache required lane plan is malformed"
        )
    expected = tuple(sorted(set(base_expected) | set(required)))
    reveals = tuple(lane_reveals)
    actual = tuple(
        (
            item.block_index,
            item.stage_id,
            item.opening.row_index,
            item.opening.lane_index,
        )
        for item in reveals
    )
    if actual != expected:
        raise ProofV3VerificationError(
            "prefix-cache lane reveals do not match the nonce challenge"
        )
    by_stage = {
        (record.block_index, record.stage_id): record for record in records
    }
    verified = []
    for key, reveal in zip(actual, reveals, strict=True):
        record = by_stage.get((reveal.block_index, reveal.stage_id))
        if record is None:
            raise ProofV3VerificationError(
                "prefix-cache lane reveal references an unopened stage"
            )
        value = verify_execution_anchor_lane_v3(
            commitment=ExecutionAnchorCommitmentV3(
                stage_id=record.stage_id,
                row_count=record.row_count,
                row_width=record.row_width,
                root=record.value_root,
            ),
            opening=reveal.opening,
        )
        verified.append((key, value))
    return tuple(verified)


def verify_prefix_cache_postnonce_v3(
    *,
    section: PrefixCachePostnonceProofV3,
    capture_chain_digest: bytes,
    selection_seed: bytes,
    execution_profile_digest: bytes,
    prompt_token_root: bytes,
    prompt_token_ids: Sequence[int],
    executed_suffix_digest: bytes,
    signed_block_token_count: int,
    layer_caches: Sequence[object],
    selected_layer_indices: Sequence[int],
    required_lane_keys: Sequence[tuple[int, str, int, int]] = (),
) -> tuple[tuple[tuple[int, str, int, int], bytes], ...]:
    """Verify the complete structural cache-hit section and opened lanes.

    The returned lane map is intentionally consumed by the economic adapter's
    registered-model relation. Authenticating a cache lane to its pre-nonce
    root alone is not an execution proof.
    """

    if not isinstance(section, PrefixCachePostnonceProofV3):
        raise ProofV3VerificationError(
            "prefix-cache post-nonce section has an unexpected type"
        )
    commitment = section.commitment
    expected_capture_digest = _fixed32(
        capture_chain_digest,
        "capture_chain_digest",
        nonzero=True,
    )
    if fold_prefix_cache_commitment_v3(
        base_capture_digest=section.base_capture_digest,
        commitment=commitment,
    ) != expected_capture_digest:
        raise ProofV3VerificationError(
            "prefix-cache section is detached from the pre-nonce capture"
        )
    tokens = tuple(int(token) for token in prompt_token_ids)
    if (
        commitment.execution_profile_digest != execution_profile_digest
        or commitment.prompt_token_root != prompt_token_root
        or commitment.context_token_count != len(tokens)
        or commitment.block_token_count != signed_block_token_count
        or commitment.executed_suffix_digest != executed_suffix_digest
    ):
        raise ProofV3VerificationError(
            "prefix-cache commitment disagrees with the authenticated request"
        )
    required = _canonical_required_lane_keys_v3(
        required_lane_keys,
        verification=True,
    )
    if any(key[0] >= commitment.block_count for key in required):
        raise ProofV3VerificationError(
            "prefix-cache required lane plan is malformed"
        )
    indices = tuple(sorted(
        set(derive_prefix_cache_block_indices_v3(
        validator_nonce=selection_seed,
        commitment=commitment,
        count=PREFIX_CACHE_BLOCK_SAMPLE_COUNT_V3,
        )) | {key[0] for key in required}
    ))
    content = derive_prefix_cache_content_digests_v3(
        execution_profile_digest=execution_profile_digest,
        cache_salt_digest=commitment.cache_salt_digest,
        token_ids=tokens[:commitment.cached_token_count],
        block_token_count=commitment.block_token_count,
    )
    verify_prefix_cache_block_opening_v3(
        commitment=commitment,
        opening=section.block_opening,
        expected_indices=indices,
        expected_content_digests=tuple(content[index] for index in indices),
    )
    stage_ids = prefix_cache_stage_ids_for_layers_v3(
        layer_caches=layer_caches,
        selected_layer_indices=selected_layer_indices,
    )
    opened_stage_ids = tuple(sorted(
        set(stage_ids) | {key[1] for key in required}
    ))
    for block_record, state_opening in zip(
        section.block_opening.records,
        section.state_openings,
        strict=True,
    ):
        verify_prefix_cache_state_opening_v3(
            block_record=block_record,
            opening=state_opening,
            expected_stage_ids=opened_stage_ids,
        )
    return verify_prefix_cache_lane_reveals_v3(
        commitment=commitment,
        state_openings=section.state_openings,
        lane_reveals=section.lane_reveals,
        validator_nonce=selection_seed,
        base_stage_ids=stage_ids,
        required_lane_keys=required,
    )


__all__ = [
    "MAX_PREFIX_CACHE_BLOCKS_V3",
    "MAX_PREFIX_CACHE_BLOCK_SAMPLES_V3",
    "MAX_PREFIX_CACHE_BLOCK_TOKENS_V3",
    "MAX_PREFIX_CACHE_STATE_STAGES_V3",
    "PREFIX_CACHE_BLOCK_SAMPLE_COUNT_V3",
    "PREFIX_CACHE_LANES_PER_ROW_V3",
    "PREFIX_CACHE_ROWS_PER_STAGE_V3",
    "PREFIX_CACHE_BLOCK_COMMITMENT_ABI_V3",
    "PREFIX_CACHE_POSTNONCE_REPLAY_ABI_V3",
    "PrefixCacheBlockOpeningV3",
    "PrefixCacheBlockRecordV3",
    "PrefixCacheCommitmentV3",
    "PrefixCacheLaneRevealV3",
    "PrefixCachePostnonceProofV3",
    "PrefixCacheStateOpeningV3",
    "PrefixCacheStateRecordV3",
    "PrefixCacheRetainedMaterialV3",
    "build_prefix_cache_commitment_v3",
    "build_prefix_cache_postnonce_proof_v3",
    "build_prefix_cache_state_tree_v3",
    "build_prefix_cache_request_material_v3",
    "derive_prefix_cache_block_indices_v3",
    "derive_prefix_cache_content_digests_v3",
    "derive_prefix_cache_residual_lane_keys_v3",
    "derive_prefix_cache_lane_keys_v3",
    "derive_prefix_cache_executed_suffix_digest_v3",
    "fold_prefix_cache_commitment_v3",
    "prefix_cache_attention_anchor_root_v3",
    "open_prefix_cache_blocks_v3",
    "open_prefix_cache_state_v3",
    "prefix_cache_stage_ids_for_layers_v3",
    "verify_prefix_cache_block_opening_v3",
    "verify_prefix_cache_lane_reveals_v3",
    "verify_prefix_cache_postnonce_v3",
    "verify_prefix_cache_state_opening_v3",
]
