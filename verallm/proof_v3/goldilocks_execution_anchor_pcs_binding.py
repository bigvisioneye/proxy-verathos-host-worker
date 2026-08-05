"""Bind a selected Goldilocks column to a pre-nonce execution anchor.

Streaming proof-v3 freezes raw FP16/BF16 runtime rows in
``ExecutionAnchorCommitmentV3`` before the validator nonce.  Post-nonce
relations use quantized Goldilocks columns.  This module joins those two
planes at transcript-derived *lanes*:

* the execution-anchor opening authenticates one canonical raw lane;
* every active value in that lane is decoded and quantized under the
  validator-owned scale; and
* the identical values are deferred against the selected PCS member.

The proof carries no selection.  Row/lane coordinates are derived only after
the post-nonce PCS root is frozen, from that root, the pre-nonce anchor root,
request nonce, signed geometry, and ordered row map.  A prover therefore
cannot learn the equality sample and then choose a different PCS commitment.
Callers may additionally require the lanes containing relation-selected
cells.  Opening a whole 2-KiB lane amortizes one Merkle path over up to 1,024
runtime values and keeps the bridge bounded without pretending that a
post-nonce economic-oracle root is a pre-nonce commitment.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.attention_anchor_binding import (
    decode_runtime_values_v3,
)
from verallm.proof_v3.economic_wire import bits_to_scale_v3
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.execution_anchor import (
    ExecutionAnchorCommitmentV3,
    ExecutionAnchorLaneOpeningV3,
    build_execution_anchor_lane_opening_v3,
    execution_anchor_lane_bytes_v3,
    verify_execution_anchor_lane_v3,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS


GOLDILOCKS_EXECUTION_ANCHOR_PCS_BINDING_ABI_V3: Final = (
    "execution_anchor_to_goldilocks_pcs.sampled_lanes.v3"
)
EXECUTION_ANCHOR_PCS_LANE_SAMPLE_COUNT_V3: Final = 4

_SAMPLE_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/EXECUTION_ANCHOR_TO_GOLDILOCKS_PCS/"
    b"SAMPLED_LANES/V3"
)

__all__ = [
    "EXECUTION_ANCHOR_PCS_LANE_SAMPLE_COUNT_V3",
    "GOLDILOCKS_EXECUTION_ANCHOR_PCS_BINDING_ABI_V3",
    "GoldilocksExecutionAnchorPcsBindingProofV3",
    "build_goldilocks_execution_anchor_lane_openings_v3",
    "derive_goldilocks_execution_anchor_pcs_lanes_v3",
    "prove_goldilocks_execution_anchor_pcs_binding_v3",
    "verify_goldilocks_execution_anchor_pcs_binding_v3",
]


def _fixed32(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    return value


def _encoded_text(value: object, name: str) -> bytes:
    if not isinstance(value, str):
        raise ProofV3Error(f"{name} is malformed")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ProofV3Error(f"{name} is malformed") from exc
    if not encoded or len(encoded) > 255:
        raise ProofV3Error(f"{name} is malformed")
    return struct.pack("<B", len(encoded)) + encoded


def _pow2(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


def _pcs_geometry(
    *,
    pcs_column,
    anchor: ExecutionAnchorCommitmentV3,
    anchor_rows: tuple[int, ...],
    source_column_offset: int,
    active_columns: int,
) -> tuple[int, int, int]:
    try:
        variable_count = int(pcs_column.pcs_statement.variable_count)
        cell_count = 1 << variable_count
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProofV3Error("execution-anchor PCS column is malformed") from exc
    if (
        not isinstance(anchor, ExecutionAnchorCommitmentV3)
        or not 1 <= variable_count <= 30
        or not anchor_rows
        or len(anchor_rows) != len(set(anchor_rows))
        or any(
            isinstance(row, bool)
            or not isinstance(row, int)
            or row < 0
            or row >= anchor.row_count
            for row in anchor_rows
        )
        or isinstance(source_column_offset, bool)
        or not isinstance(source_column_offset, int)
        or source_column_offset < 0
        or isinstance(active_columns, bool)
        or not isinstance(active_columns, int)
        or active_columns <= 0
    ):
        raise ProofV3Error(
            "execution-anchor PCS selected geometry is malformed"
        )
    if anchor.row_width % 2:
        raise ProofV3Error(
            "execution-anchor PCS source is not a 16-bit runtime row"
        )
    source_width = anchor.row_width // 2
    if source_column_offset + active_columns > source_width:
        raise ProofV3Error(
            "execution-anchor PCS source slice exceeds the runtime row"
        )
    row_pad = _pow2(len(anchor_rows))
    col_pad = _pow2(active_columns)
    if cell_count != row_pad * col_pad:
        raise ProofV3Error(
            "execution-anchor PCS member has the wrong selected-row geometry"
        )
    return cell_count, row_pad, col_pad


def _column_identity(pcs_column) -> tuple[bytes, bytes, str, bytes]:
    try:
        statement_digest = pcs_column.pcs_statement.digest()
        pcs_root = pcs_column.tree.commitment
        group_tag = pcs_column.group_tag or pcs_column.tag
        block_point = tuple(int(bit) for bit in pcs_column.block_point)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProofV3Error("execution-anchor PCS column is malformed") from exc
    if (
        not isinstance(statement_digest, bytes)
        or len(statement_digest) != 32
        or not isinstance(pcs_root, bytes)
        or len(pcs_root) != 32
        or any(bit not in (0, 1) for bit in block_point)
    ):
        raise ProofV3Error(
            "execution-anchor PCS member identity is malformed"
        )
    return statement_digest, pcs_root, group_tag, bytes(block_point)


def _active_lane_indices(
    *,
    anchor: ExecutionAnchorCommitmentV3,
    source_column_offset: int,
    active_columns: int,
) -> tuple[int, ...]:
    lane_bytes = execution_anchor_lane_bytes_v3(anchor.stage_id)
    byte_start = source_column_offset * 2
    byte_stop = (source_column_offset + active_columns) * 2
    first = byte_start // lane_bytes
    last = (byte_stop - 1) // lane_bytes
    return tuple(range(first, last + 1))


def derive_goldilocks_execution_anchor_pcs_lanes_v3(
    *,
    tile_digest: bytes,
    validator_nonce: bytes,
    tag: str,
    anchor: ExecutionAnchorCommitmentV3,
    anchor_rows,
    pcs_column,
    source_column_offset: int,
    active_columns: int,
    scale_bits: int,
    encoding_id: str,
    required_member_columns=(),
) -> tuple[tuple[int, int], ...]:
    """Derive canonical ``(anchor_row, lane_index)`` equality openings."""

    try:
        rows = tuple(anchor_rows)
    except TypeError as exc:
        raise ProofV3Error(
            "execution-anchor PCS row map is malformed"
        ) from exc
    cell_count, _row_pad, col_pad = _pcs_geometry(
        pcs_column=pcs_column,
        anchor=anchor,
        anchor_rows=rows,
        source_column_offset=source_column_offset,
        active_columns=active_columns,
    )
    if encoding_id not in {"fp16.v1", "bf16.v1"}:
        raise ProofV3Error(
            "execution-anchor PCS runtime encoding is unsupported"
        )
    # Decode and validate the exact signed scale now; its canonical IEEE bits
    # are transcript material below.
    bits_to_scale_v3(scale_bits)
    statement_digest, pcs_root, group_tag, block_point = _column_identity(
        pcs_column
    )
    if tag != pcs_column.tag:
        raise ProofV3Error(
            "execution-anchor PCS member tag is inconsistent"
        )
    lanes = _active_lane_indices(
        anchor=anchor,
        source_column_offset=source_column_offset,
        active_columns=active_columns,
    )
    try:
        required = tuple(required_member_columns)
    except TypeError as exc:
        raise ProofV3Error(
            "execution-anchor PCS required columns are malformed"
        ) from exc
    if (
        required != tuple(sorted(set(required)))
        or any(
            isinstance(column, bool)
            or not isinstance(column, int)
            or column < 0
            or column >= active_columns
            for column in required
        )
    ):
        raise ProofV3Error(
            "execution-anchor PCS required columns are malformed"
        )
    candidates = tuple(
        (row, lane)
        for row in rows
        for lane in lanes
    )
    seed = hashlib.sha256(
        _SAMPLE_DOMAIN
        + GOLDILOCKS_EXECUTION_ANCHOR_PCS_BINDING_ABI_V3.encode("ascii")
        + _fixed32(tile_digest, "execution-anchor PCS tile digest")
        + _fixed32(validator_nonce, "execution-anchor PCS validator nonce")
        + _encoded_text(tag, "execution-anchor PCS member tag")
        + _encoded_text(group_tag, "execution-anchor PCS group tag")
        + _encoded_text(anchor.stage_id, "execution-anchor stage id")
        + _encoded_text(encoding_id, "execution-anchor encoding id")
        + _fixed32(anchor.root, "execution-anchor root")
        + statement_digest
        + pcs_root
        + struct.pack(
            "<IIIIIIQII",
            anchor.row_count,
            anchor.row_width,
            source_column_offset,
            active_columns,
            cell_count,
            col_pad,
            scale_bits,
            EXECUTION_ANCHOR_PCS_LANE_SAMPLE_COUNT_V3,
            len(required),
        )
        + struct.pack("<H", len(block_point))
        + block_point
        + b"".join(struct.pack("<I", row) for row in rows)
        + b"".join(struct.pack("<I", column) for column in required)
    ).digest()
    count = min(
        EXECUTION_ANCHOR_PCS_LANE_SAMPLE_COUNT_V3,
        len(candidates),
    )
    selected: set[int] = set()
    counter = 0
    limit = (1 << 64) - ((1 << 64) % len(candidates))
    while len(selected) < count:
        block = hashlib.sha256(
            seed + struct.pack("<Q", counter)
        ).digest()
        counter += 1
        for offset in range(0, 32, 8):
            word = int.from_bytes(block[offset : offset + 8], "little")
            if word < limit:
                selected.add(word % len(candidates))
            if len(selected) == count:
                break
    lane_elements = execution_anchor_lane_bytes_v3(anchor.stage_id) // 2
    mandatory = {
        (
            row,
            (source_column_offset + column) // lane_elements,
        )
        for row in rows
        for column in required
    }
    return tuple(
        sorted(
            mandatory
            | {candidates[index] for index in selected}
        )
    )


def build_goldilocks_execution_anchor_lane_openings_v3(
    *,
    commitment: ExecutionAnchorCommitmentV3,
    row_bytes_by_index,
    row_tree,
    lane_keys,
) -> tuple[ExecutionAnchorLaneOpeningV3, ...]:
    """Build the exact derived lanes from bounded replay material."""

    try:
        rows = dict(row_bytes_by_index)
        keys = tuple(lane_keys)
    except (TypeError, ValueError) as exc:
        raise ProofV3Error(
            "execution-anchor PCS lane sources are malformed"
        ) from exc
    if (
        keys != tuple(sorted(set(keys)))
        or any(row not in rows for row, _lane in keys)
    ):
        raise ProofV3Error(
            "execution-anchor PCS lane sources are incomplete"
        )
    return tuple(
        build_execution_anchor_lane_opening_v3(
            commitment=commitment,
            row_index=row,
            row_bytes=rows[row],
            row_tree=row_tree,
            lane_index=lane,
        )
        for row, lane in keys
    )


def _boolean_point(index: int, variable_count: int) -> tuple[int, ...]:
    return tuple((index >> bit) & 1 for bit in range(variable_count))


def _column_value(pcs_column, index: int) -> int:
    source = pcs_column.values
    if source is None:
        source = pcs_column.device_values
    if source is None:
        source = pcs_column.device_values_host
    if source is None:
        raise ProofV3Error(
            "execution-anchor PCS column values are unavailable"
        )
    raw = source[index]
    if hasattr(raw, "item"):
        raw = raw.item()
    value = int(raw)
    if value < 0:
        value += 1 << 64
    if not 0 <= value < GOLDILOCKS_MODULUS:
        raise ProofV3Error(
            "execution-anchor PCS column value is not canonical"
        )
    return value


def _quantized_lane_values(
    *,
    opening: ExecutionAnchorLaneOpeningV3,
    anchor: ExecutionAnchorCommitmentV3,
    encoding_id: str,
    scale_bits: int,
    source_column_offset: int,
    active_columns: int,
    row_slot: int,
    col_pad: int,
) -> tuple[tuple[int, int], ...]:
    import numpy as np

    raw = verify_execution_anchor_lane_v3(
        commitment=anchor,
        opening=opening,
    )
    decoded = decode_runtime_values_v3(raw, encoding_id)
    scale = bits_to_scale_v3(scale_bits)
    quantized = np.clip(
        np.rint(decoded / scale),
        -128,
        127,
    ).astype(np.int64)
    lane_elements = (
        execution_anchor_lane_bytes_v3(anchor.stage_id) // 2
    )
    lane_start = opening.lane_index * lane_elements
    active_start = source_column_offset
    active_stop = source_column_offset + active_columns
    start = max(lane_start, active_start)
    stop = min(lane_start + len(quantized), active_stop)
    if start >= stop:
        raise ProofV3VerificationError(
            "execution-anchor PCS lane does not intersect the source slice"
        )
    result = []
    for source_column in range(start, stop):
        member_column = source_column - source_column_offset
        value = int(quantized[source_column - lane_start])
        result.append(
            (
                row_slot * col_pad + member_column,
                value % GOLDILOCKS_MODULUS,
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class GoldilocksExecutionAnchorPcsBindingProofV3:
    lane_openings: tuple[ExecutionAnchorLaneOpeningV3, ...]

    def __post_init__(self) -> None:
        openings = tuple(self.lane_openings)
        if not openings or not all(
            isinstance(item, ExecutionAnchorLaneOpeningV3)
            for item in openings
        ):
            raise ProofV3Error(
                "execution-anchor PCS binding proof is malformed"
            )
        object.__setattr__(self, "lane_openings", openings)


def prove_goldilocks_execution_anchor_pcs_binding_v3(
    *,
    tile_digest: bytes,
    validator_nonce: bytes,
    tag: str,
    anchor: ExecutionAnchorCommitmentV3,
    anchor_rows,
    pcs_column,
    source_column_offset: int,
    active_columns: int,
    scale_bits: int,
    encoding_id: str,
    lane_openings,
    collector,
    required_member_columns=(),
) -> GoldilocksExecutionAnchorPcsBindingProofV3:
    """Verify raw lanes locally and defer their quantized PCS cells."""

    rows = tuple(anchor_rows)
    keys = derive_goldilocks_execution_anchor_pcs_lanes_v3(
        tile_digest=tile_digest,
        validator_nonce=validator_nonce,
        tag=tag,
        anchor=anchor,
        anchor_rows=rows,
        pcs_column=pcs_column,
        source_column_offset=source_column_offset,
        active_columns=active_columns,
        scale_bits=scale_bits,
        encoding_id=encoding_id,
        required_member_columns=required_member_columns,
    )
    openings = tuple(lane_openings)
    if tuple(
        (opening.row_index, opening.lane_index)
        for opening in openings
    ) != keys:
        raise ProofV3Error(
            "execution-anchor PCS proof carries the wrong derived lanes"
        )
    _cells, _row_pad, col_pad = _pcs_geometry(
        pcs_column=pcs_column,
        anchor=anchor,
        anchor_rows=rows,
        source_column_offset=source_column_offset,
        active_columns=active_columns,
    )
    slots = {row: index for index, row in enumerate(rows)}
    for opening in openings:
        for member_cell, value in _quantized_lane_values(
            opening=opening,
            anchor=anchor,
            encoding_id=encoding_id,
            scale_bits=scale_bits,
            source_column_offset=source_column_offset,
            active_columns=active_columns,
            row_slot=slots[opening.row_index],
            col_pad=col_pad,
        ):
            if _column_value(pcs_column, member_cell) != value:
                raise ProofV3Error(
                    "execution-anchor lane and PCS column disagree"
                )
            collector.defer(
                tag,
                _boolean_point(
                    member_cell,
                    pcs_column.pcs_statement.variable_count,
                ),
                value,
            )
    return GoldilocksExecutionAnchorPcsBindingProofV3(openings)


def verify_goldilocks_execution_anchor_pcs_binding_v3(
    proof: object,
    *,
    tile_digest: bytes,
    validator_nonce: bytes,
    tag: str,
    anchor: ExecutionAnchorCommitmentV3,
    anchor_rows,
    pcs_column,
    source_column_offset: int,
    active_columns: int,
    scale_bits: int,
    encoding_id: str,
    checker,
    required_member_columns=(),
) -> tuple[tuple[int, int], ...]:
    """Authenticate the derived raw lanes and register PCS expectations."""

    try:
        if not isinstance(
            proof,
            GoldilocksExecutionAnchorPcsBindingProofV3,
        ):
            raise ProofV3VerificationError(
                "execution-anchor PCS binding proof has a wrong type"
            )
        rows = tuple(anchor_rows)
        keys = derive_goldilocks_execution_anchor_pcs_lanes_v3(
            tile_digest=tile_digest,
            validator_nonce=validator_nonce,
            tag=tag,
            anchor=anchor,
            anchor_rows=rows,
            pcs_column=pcs_column,
            source_column_offset=source_column_offset,
            active_columns=active_columns,
            scale_bits=scale_bits,
            encoding_id=encoding_id,
            required_member_columns=required_member_columns,
        )
        if tuple(
            (opening.row_index, opening.lane_index)
            for opening in proof.lane_openings
        ) != keys:
            raise ProofV3VerificationError(
                "execution-anchor PCS proof carries the wrong derived lanes"
            )
        _cells, _row_pad, col_pad = _pcs_geometry(
            pcs_column=pcs_column,
            anchor=anchor,
            anchor_rows=rows,
            source_column_offset=source_column_offset,
            active_columns=active_columns,
        )
        slots = {row: index for index, row in enumerate(rows)}
        bound_values: dict[int, int] = {}
        for opening in proof.lane_openings:
            for member_cell, value in _quantized_lane_values(
                opening=opening,
                anchor=anchor,
                encoding_id=encoding_id,
                scale_bits=scale_bits,
                source_column_offset=source_column_offset,
                active_columns=active_columns,
                row_slot=slots[opening.row_index],
                col_pad=col_pad,
            ):
                checker.expect(
                    tag,
                    _boolean_point(
                        member_cell,
                        pcs_column.pcs_statement.variable_count,
                    ),
                    value,
                )
                signed = (
                    value
                    if value <= 127
                    else value - GOLDILOCKS_MODULUS
                )
                previous = bound_values.setdefault(member_cell, signed)
                if previous != signed:
                    raise ProofV3VerificationError(
                        "execution-anchor PCS lanes disagree on a cell"
                    )
        return tuple(sorted(bound_values.items()))
    except ProofV3VerificationError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "execution-anchor PCS binding proof is malformed"
        ) from exc
