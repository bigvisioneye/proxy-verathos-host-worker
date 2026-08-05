"""Bind selected Goldilocks PCS columns to pre-nonce capture roots.

The serving path commits runtime tensors with the inexpensive economic
capture tree before the validator nonce.  Succinct execution relations use a
Goldilocks multilinear PCS after the nonce.  This module ties the two
commitment planes together at transcript-derived cells:

* the capture opening authenticates the served value;
* the same value is deferred against the selected PCS member; and
* both the capture root and the packed PCS root are fixed before coordinates
  are derived.

Indices are verifier-derived and never carried by the proof.  Packed
heterogeneous columns work through the existing collector/checker alias,
which appends the member's authenticated subcube prefix to each PCS point.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.economic_commitment import (
    EconomicCommittedOracleV3,
    oracle_leaf_index_v3,
    oracle_leaf_width_v3,
    signed_to_field_v3,
    verify_economic_oracle_opening_v3,
)
from verallm.proof_v3.economic_wire import (
    EconomicMerkleOpeningV3,
    EconomicOracleCommitmentV3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS


GOLDILOCKS_CAPTURE_PCS_BINDING_ABI_V3: Final = (
    "capture_to_goldilocks_pcs.sampled_chunks.v2"
)
DYNAMIC_CAPTURE_EQUALITY_CHUNK_COUNT_V3: Final = 4

_SAMPLE_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/CAPTURE_TO_GOLDILOCKS_PCS/SAMPLED_CHUNKS/V2"
)

__all__ = [
    "DYNAMIC_CAPTURE_EQUALITY_CHUNK_COUNT_V3",
    "GOLDILOCKS_CAPTURE_PCS_BINDING_ABI_V3",
    "GoldilocksCapturePcsBindingProofV3",
    "derive_goldilocks_capture_pcs_samples_v3",
    "prove_goldilocks_capture_pcs_binding_v3",
    "verify_goldilocks_capture_pcs_binding_v3",
]


def _fixed32(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    return value


def _pow2(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


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


def _oracle_record(oracle: EconomicOracleCommitmentV3) -> bytes:
    if not isinstance(oracle, EconomicOracleCommitmentV3):
        raise ProofV3Error("capture oracle record is malformed")
    return (
        _encoded_text(oracle.oracle_id, "capture oracle id")
        + _encoded_text(oracle.phase, "capture oracle phase")
        + _encoded_text(oracle.operation, "capture oracle operation")
        + struct.pack(
            "<IIIQ",
            oracle.layer_index,
            oracle.row_count,
            oracle.col_count,
            oracle.scale_bits,
        )
        + _fixed32(oracle.root, "capture oracle root")
    )


def _column_geometry(
    *,
    pcs_column,
    selected_rows: tuple[int, ...],
    oracle: EconomicOracleCommitmentV3,
) -> tuple[int, int, int]:
    try:
        variable_count = int(pcs_column.pcs_statement.variable_count)
        cell_count = 1 << variable_count
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProofV3Error("capture PCS column is malformed") from exc
    if not 1 <= variable_count <= 30:
        raise ProofV3Error("capture PCS column arity is unsupported")
    if (
        not selected_rows
        or len(selected_rows) != len(set(selected_rows))
        or any(
            isinstance(row, bool)
            or not isinstance(row, int)
            or row < 0
            or row >= oracle.row_count
            for row in selected_rows
        )
    ):
        raise ProofV3Error("capture PCS selected rows are malformed")
    row_pad = _pow2(len(selected_rows))
    col_pad = _pow2(oracle.col_count)
    if cell_count != row_pad * col_pad:
        raise ProofV3Error(
            "capture PCS column does not match the selected-row geometry")
    return cell_count, row_pad, col_pad


def _sample_seed(
    *,
    tile_digest: bytes,
    capture_base_binding_digest: bytes,
    validator_nonce: bytes,
    tag: str,
    oracle: EconomicOracleCommitmentV3,
    pcs_column,
    selected_rows: tuple[int, ...],
    cell_count: int,
    col_pad: int,
    leaf_width: int,
    active_chunks: int,
) -> bytes:
    try:
        statement_digest = pcs_column.pcs_statement.digest()
        pcs_root = pcs_column.tree.commitment
        group_tag = pcs_column.group_tag or pcs_column.tag
        block_point = tuple(int(bit) for bit in pcs_column.block_point)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProofV3Error("capture PCS column is malformed") from exc
    if (
        tag != pcs_column.tag
        or any(bit not in (0, 1) for bit in block_point)
        or not isinstance(statement_digest, bytes)
        or len(statement_digest) != 32
    ):
        raise ProofV3Error("capture PCS member identity is malformed")
    encoded_block = bytes(block_point)
    return hashlib.sha256(
        _SAMPLE_DOMAIN
        + GOLDILOCKS_CAPTURE_PCS_BINDING_ABI_V3.encode("ascii")
        + _fixed32(tile_digest, "capture PCS tile digest")
        + _fixed32(
            capture_base_binding_digest,
            "capture PCS base binding",
        )
        + _fixed32(validator_nonce, "capture PCS validator nonce")
        + _encoded_text(tag, "capture PCS member tag")
        + _encoded_text(group_tag, "capture PCS group tag")
        + _oracle_record(oracle)
        + statement_digest
        + _fixed32(pcs_root, "capture PCS root")
        + struct.pack(
            "<IIIIIII",
            cell_count,
            col_pad,
            len(selected_rows),
            oracle.col_count,
            leaf_width,
            active_chunks,
            DYNAMIC_CAPTURE_EQUALITY_CHUNK_COUNT_V3,
        )
        + struct.pack("<H", len(encoded_block))
        + encoded_block
        + b"".join(struct.pack("<I", row) for row in selected_rows)
    ).digest()


def derive_goldilocks_capture_pcs_samples_v3(
    *,
    tile_digest: bytes,
    capture_base_binding_digest: bytes,
    validator_nonce: bytes,
    tag: str,
    oracle: EconomicOracleCommitmentV3,
    pcs_column,
    selected_rows,
) -> tuple[tuple[int, int], ...]:
    """Return ``(member_cell, capture_cell)`` pairs in canonical order."""

    try:
        rows = tuple(selected_rows)
    except TypeError as exc:
        raise ProofV3Error("capture PCS selected rows are malformed") from exc
    cell_count, _row_pad, col_pad = _column_geometry(
        pcs_column=pcs_column,
        selected_rows=rows,
        oracle=oracle,
    )
    leaf_width = oracle_leaf_width_v3(oracle.col_count)
    chunks_per_row = (
        oracle.col_count + leaf_width - 1
    ) // leaf_width
    active_chunks = len(rows) * chunks_per_row
    seed = _sample_seed(
        tile_digest=tile_digest,
        capture_base_binding_digest=capture_base_binding_digest,
        validator_nonce=validator_nonce,
        tag=tag,
        oracle=oracle,
        pcs_column=pcs_column,
        selected_rows=rows,
        cell_count=cell_count,
        col_pad=col_pad,
        leaf_width=leaf_width,
        active_chunks=active_chunks,
    )
    count = min(DYNAMIC_CAPTURE_EQUALITY_CHUNK_COUNT_V3, active_chunks)
    limit = (1 << 64) - ((1 << 64) % active_chunks)
    selected: set[int] = set()
    counter = 0
    while len(selected) < count:
        block = hashlib.sha256(
            seed + struct.pack("<Q", counter)
        ).digest()
        counter += 1
        for offset in range(0, 32, 8):
            word = int.from_bytes(block[offset : offset + 8], "little")
            if word < limit:
                selected.add(word % active_chunks)
            if len(selected) == count:
                break
    pairs = []
    for chunk in sorted(selected):
        row_slot, chunk_in_row = divmod(chunk, chunks_per_row)
        start = chunk_in_row * leaf_width
        stop = min(start + leaf_width, oracle.col_count)
        for column in range(start, stop):
            member_cell = row_slot * col_pad + column
            capture_cell = oracle_leaf_index_v3(
                rows[row_slot],
                column,
                oracle.col_count,
            )
            pairs.append((member_cell, capture_cell))
    return tuple(pairs)


def _boolean_point(index: int, variable_count: int) -> tuple[int, ...]:
    return tuple((index >> bit) & 1 for bit in range(variable_count))


def _column_value(pcs_column, index: int) -> int:
    source = pcs_column.values
    if source is None:
        source = pcs_column.device_values
    if source is None:
        source = pcs_column.device_values_host
    if source is None:
        raise ProofV3Error("capture PCS column values are unavailable")
    raw = source[index]
    if hasattr(raw, "item"):
        raw = raw.item()
    value = int(raw)
    if value < 0:
        value += 1 << 64
    if not 0 <= value < GOLDILOCKS_MODULUS:
        raise ProofV3Error("capture PCS column value is not canonical")
    return value


@dataclass(frozen=True, slots=True)
class GoldilocksCapturePcsBindingProofV3:
    """Capture-side multiproof; PCS answers share the global batch opening."""

    capture_opening: EconomicMerkleOpeningV3

    def __post_init__(self) -> None:
        if not isinstance(self.capture_opening, EconomicMerkleOpeningV3):
            raise ProofV3Error("capture PCS binding opening is malformed")


def prove_goldilocks_capture_pcs_binding_v3(
    *,
    tile_digest: bytes,
    capture_base_binding_digest: bytes,
    validator_nonce: bytes,
    tag: str,
    committed_oracle: EconomicCommittedOracleV3,
    pcs_column,
    selected_rows,
    collector,
    value_mode: int = 1,
    bounded_width: int | None = None,
) -> GoldilocksCapturePcsBindingProofV3:
    """Open sampled capture cells and defer their identical PCS values."""

    if not isinstance(committed_oracle, EconomicCommittedOracleV3):
        raise ProofV3Error("capture PCS committed oracle is malformed")
    pairs = derive_goldilocks_capture_pcs_samples_v3(
        tile_digest=tile_digest,
        capture_base_binding_digest=capture_base_binding_digest,
        validator_nonce=validator_nonce,
        tag=tag,
        oracle=committed_oracle.commitment,
        pcs_column=pcs_column,
        selected_rows=selected_rows,
    )
    capture_cells = tuple(capture for _member, capture in pairs)
    for member_cell, capture_cell in pairs:
        row = capture_cell // committed_oracle.col_pad
        column = capture_cell % committed_oracle.col_pad
        if column >= committed_oracle.commitment.col_count:
            raise ProofV3Error("capture PCS sample points into padding")
        capture_value = signed_to_field_v3(
            committed_oracle.signed_value(row, column))
        if _column_value(pcs_column, member_cell) != capture_value:
            raise ProofV3Error(
                "capture and PCS columns disagree at a derived sample")
        collector.defer(
            tag,
            _boolean_point(
                member_cell,
                pcs_column.pcs_statement.variable_count,
            ),
            capture_value,
        )
    _leaves, opening = committed_oracle.open_cells(
        (
            (
                capture_cell // committed_oracle.col_pad,
                capture_cell % committed_oracle.col_pad,
            )
            for capture_cell in capture_cells
        ),
        value_mode=value_mode,
        bounded_width=bounded_width,
    )
    return GoldilocksCapturePcsBindingProofV3(opening)


def verify_goldilocks_capture_pcs_binding_v3(
    proof: object,
    *,
    tile_digest: bytes,
    capture_base_binding_digest: bytes,
    validator_nonce: bytes,
    tag: str,
    oracle: EconomicOracleCommitmentV3,
    pcs_column,
    selected_rows,
    checker,
    expected_mode: int = 1,
    expected_bounded_width: int | None = None,
) -> None:
    """Authenticate capture samples and register the same PCS expectations."""

    try:
        if not isinstance(proof, GoldilocksCapturePcsBindingProofV3):
            raise ProofV3VerificationError(
                "capture PCS binding proof has a wrong type")
        pairs = derive_goldilocks_capture_pcs_samples_v3(
            tile_digest=tile_digest,
            capture_base_binding_digest=capture_base_binding_digest,
            validator_nonce=validator_nonce,
            tag=tag,
            oracle=oracle,
            pcs_column=pcs_column,
            selected_rows=selected_rows,
        )
        capture_cells = tuple(capture for _member, capture in pairs)
        opened = verify_economic_oracle_opening_v3(
            oracle=oracle,
            base_binding=capture_base_binding_digest,
            expected_indices=capture_cells,
            opening=proof.capture_opening,
            expected_mode=expected_mode,
            expected_bounded_width=expected_bounded_width,
        )
        for member_cell, capture_cell in pairs:
            checker.expect(
                tag,
                _boolean_point(
                    member_cell,
                    pcs_column.pcs_statement.variable_count,
                ),
                signed_to_field_v3(opened[capture_cell]),
            )
    except ProofV3VerificationError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "capture PCS binding proof is malformed") from exc
