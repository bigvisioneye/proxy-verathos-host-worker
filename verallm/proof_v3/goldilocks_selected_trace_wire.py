"""Canonical bounded wire for the complete Goldilocks selected trace.

The selected trace is an internal graph of immutable proof dataclasses.  This
codec serializes only an explicit type registry and a small set of primitive
values; it never imports a type named by the wire and never uses pickle.
Validator-owned statements and contexts are deliberately absent.
"""

from __future__ import annotations

import dataclasses
import struct
from typing import Final

from verallm.proof_v3.economic_wire import (
    EconomicExecutionAnchorLaneRevealV3,
    EconomicMerkleOpeningV3,
    EconomicMerkleSiblingV3,
    EconomicWeightRowRevealV3,
    _Reader,
    _Writer,
)
from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.execution_anchor import ExecutionAnchorLaneOpeningV3
from verallm.proof_v3.goldilocks_batched_pcs_opening import (
    GoldilocksBatchedComponentOpeningV3,
    GoldilocksBatchedOpeningProofV3,
)
from verallm.proof_v3.goldilocks_bottom_anchor import (
    GoldilocksBottomAnchorProofV3,
)
from verallm.proof_v3.goldilocks_capture_pcs_binding import (
    GoldilocksCapturePcsBindingProofV3,
)
from verallm.proof_v3.goldilocks_execution_anchor_pcs_binding import (
    GoldilocksExecutionAnchorPcsBindingProofV3,
)
from verallm.proof_v3.goldilocks_final_rmsnorm import (
    GoldilocksFinalRmsnormProofV3,
)
from verallm.proof_v3.goldilocks_gdn_composition import (
    GoldilocksGdnCaptureProofV3,
    GoldilocksGdnCompositionProofV3,
)
from verallm.proof_v3.goldilocks_merkle_reference import (
    GoldilocksMerkleMultiOpeningReference,
    GoldilocksMerkleSiblingReference,
)
from verallm.proof_v3.goldilocks_mlp_composition import (
    GoldilocksMlpCompositionProofV3,
)
from verallm.proof_v3.goldilocks_projection_composition import (
    GoldilocksProjectionCaptureProofV3,
    GoldilocksProjectionCompositionProofV3,
    GoldilocksProjectionGroupCommitmentV3,
    GoldilocksProjectionRelationProofV3,
)
from verallm.proof_v3.goldilocks_residual_composition import (
    GoldilocksResidualCaptureProofV3,
    GoldilocksResidualCompositionProofV3,
    GoldilocksResidualGroupCommitmentV3,
)
from verallm.proof_v3.goldilocks_rmsnorm_composition import (
    GoldilocksRmsnormCompositionProofV3,
)
from verallm.proof_v3.goldilocks_selected_trace import (
    GoldilocksSelectedTraceProofV3,
)
from verallm.proof_v3.goldilocks_static_catalog_bridge import (
    GoldilocksStaticCatalogBridgeProofV3,
    GoldilocksStaticCatalogOperationProofV3,
    GoldilocksStaticCatalogWidthProofV3,
)
from verallm.proof_v3.goldilocks_succinct_batch_opening import (
    GoldilocksDeferredOpeningV3,
)
from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (
    GoldilocksSuccinctLogupProofV3,
    GoldilocksSuccinctLogupSubProofV3,
)
from verallm.proof_v3.goldilocks_succinct_product_argument_reference import (
    GoldilocksSuccinctProductProofV3,
)
from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
    SuccinctEqFoldProofV3,
)
from verallm.proof_v3.goldilocks_terminal_argmax import (
    GoldilocksTerminalArgmaxProofV3,
)
from verallm.proof_v3.goldilocks_terminal_lm_head import (
    GoldilocksTerminalLmHeadProofV3,
)
from verallm.proof_v3.goldilocks_terminal_path import (
    GoldilocksTerminalPathProofV3,
)
from verallm.proof_v3.succinct_attention_wire import (
    CaptureKvLayerSectionWireV3,
    _r_multiopen,
    _w_multiopen,
)
from zkllm.crypto.pcs_v2 import PCSOpeningV2


GOLDILOCKS_SELECTED_TRACE_WIRE_ABI_V3: Final = (
    "execution.selected_trace.canonical_wire.v7"
)
MAX_SELECTED_TRACE_WIRE_BYTES: Final = 2 << 20

_MAGIC: Final = b"VST3"
_VERSION: Final = 7
_MAX_CONTAINER_ITEMS: Final = 1 << 18
_MAX_MAP_ITEMS: Final = 1 << 12
_MAX_TEXT_BYTES: Final = 1 << 10

_NONE: Final = 0
_FALSE: Final = 1
_TRUE: Final = 2
_INT: Final = 3
_BYTES: Final = 4
_TEXT: Final = 5
_TUPLE: Final = 6
_U64_TUPLE: Final = 7
_I64_TUPLE: Final = 8
_BYTES_TUPLE: Final = 9
_MAP: Final = 10
_RECORD: Final = 11

# Type ids are part of the wire ABI. Append only; never reorder.
_RECORD_TYPES: Final = (
    EconomicExecutionAnchorLaneRevealV3,
    EconomicMerkleOpeningV3,
    EconomicMerkleSiblingV3,
    EconomicWeightRowRevealV3,
    ExecutionAnchorLaneOpeningV3,
    GoldilocksBatchedComponentOpeningV3,
    GoldilocksBatchedOpeningProofV3,
    GoldilocksBottomAnchorProofV3,
    GoldilocksExecutionAnchorPcsBindingProofV3,
    GoldilocksFinalRmsnormProofV3,
    GoldilocksGdnCaptureProofV3,
    GoldilocksGdnCompositionProofV3,
    GoldilocksMerkleMultiOpeningReference,
    GoldilocksMerkleSiblingReference,
    GoldilocksMlpCompositionProofV3,
    GoldilocksProjectionCaptureProofV3,
    GoldilocksProjectionCompositionProofV3,
    GoldilocksProjectionGroupCommitmentV3,
    GoldilocksProjectionRelationProofV3,
    GoldilocksResidualCaptureProofV3,
    GoldilocksResidualCompositionProofV3,
    GoldilocksResidualGroupCommitmentV3,
    GoldilocksRmsnormCompositionProofV3,
    GoldilocksSelectedTraceProofV3,
    GoldilocksStaticCatalogBridgeProofV3,
    GoldilocksStaticCatalogOperationProofV3,
    GoldilocksStaticCatalogWidthProofV3,
    GoldilocksDeferredOpeningV3,
    GoldilocksSuccinctLogupProofV3,
    GoldilocksSuccinctLogupSubProofV3,
    GoldilocksSuccinctProductProofV3,
    SuccinctEqFoldProofV3,
    GoldilocksTerminalArgmaxProofV3,
    GoldilocksTerminalLmHeadProofV3,
    GoldilocksTerminalPathProofV3,
    CaptureKvLayerSectionWireV3,
    PCSOpeningV2,
    GoldilocksCapturePcsBindingProofV3,
)
_TYPE_TO_ID: Final = {
    record_type: index + 1
    for index, record_type in enumerate(_RECORD_TYPES)
}


def _write_text(writer: _Writer, value: str) -> None:
    try:
        encoded = value.encode("ascii")
    except (AttributeError, UnicodeEncodeError) as exc:
        raise ProofV3Error("selected-trace wire text is not ASCII") from exc
    if len(encoded) > _MAX_TEXT_BYTES:
        raise ProofV3Error("selected-trace wire text exceeds the cap")
    writer.pack("<H", len(encoded))
    writer.raw(encoded)


def _read_text(reader: _Reader) -> str:
    (size,) = reader.unpack("<H")
    if size > _MAX_TEXT_BYTES:
        raise ProofV3Error("selected-trace wire text exceeds the cap")
    try:
        return reader.read(size).decode("ascii")
    except UnicodeDecodeError as exc:
        raise ProofV3Error("selected-trace wire text is not ASCII") from exc


def _write_lane_opening(
    writer: _Writer,
    opening: ExecutionAnchorLaneOpeningV3,
) -> None:
    writer.pack(
        "<IIH",
        opening.row_index,
        opening.lane_index,
        len(opening.lane_bytes),
    )
    writer.raw(opening.lane_bytes)
    writer.pack("<B", len(opening.lane_sibling_hashes))
    for sibling in opening.lane_sibling_hashes:
        writer.raw(sibling)
    writer.pack("<B", len(opening.row_sibling_hashes))
    for sibling in opening.row_sibling_hashes:
        writer.raw(sibling)


def _read_lane_opening(reader: _Reader) -> ExecutionAnchorLaneOpeningV3:
    row_index, lane_index, lane_size = reader.unpack("<IIH")
    if lane_size not in (256, 2048):
        raise ProofV3Error(
            "selected-trace execution-anchor lane size is unsupported"
        )
    lane_bytes = reader.read(lane_size)
    (lane_count,) = reader.unpack("<B")
    if lane_count > 32:
        raise ProofV3Error(
            "selected-trace execution-anchor lane path is too deep"
        )
    lane_siblings = tuple(reader.read(32) for _ in range(lane_count))
    (row_count,) = reader.unpack("<B")
    if row_count > 32:
        raise ProofV3Error(
            "selected-trace execution-anchor row path is too deep"
        )
    return ExecutionAnchorLaneOpeningV3(
        row_index=row_index,
        lane_index=lane_index,
        lane_bytes=lane_bytes,
        lane_sibling_hashes=lane_siblings,
        row_sibling_hashes=tuple(
            reader.read(32) for _ in range(row_count)
        ),
    )


def _write_record(writer: _Writer, value: object) -> None:
    record_type = type(value)
    type_id = _TYPE_TO_ID.get(record_type)
    if type_id is None:
        raise ProofV3Error(
            "selected-trace wire contains an unsupported proof type"
        )
    writer.pack("<H", type_id)
    if record_type is EconomicExecutionAnchorLaneRevealV3:
        value.encode(writer)
        return
    if record_type is EconomicMerkleOpeningV3:
        value.encode(writer)
        return
    if record_type is EconomicWeightRowRevealV3:
        value.encode(writer)
        return
    if record_type is ExecutionAnchorLaneOpeningV3:
        _write_lane_opening(writer, value)
        return
    if record_type is GoldilocksMerkleMultiOpeningReference:
        _w_multiopen(writer, value)
        return
    for field in dataclasses.fields(value):
        _write_value(writer, getattr(value, field.name))


def _read_record(reader: _Reader) -> object:
    (type_id,) = reader.unpack("<H")
    if not 1 <= type_id <= len(_RECORD_TYPES):
        raise ProofV3Error("selected-trace wire record type is unsupported")
    record_type = _RECORD_TYPES[type_id - 1]
    if record_type is EconomicExecutionAnchorLaneRevealV3:
        return record_type.decode(reader)
    if record_type is EconomicMerkleOpeningV3:
        return record_type.decode(reader)
    if record_type is EconomicWeightRowRevealV3:
        return record_type.decode(reader)
    if record_type is ExecutionAnchorLaneOpeningV3:
        return _read_lane_opening(reader)
    if record_type is GoldilocksMerkleMultiOpeningReference:
        return _r_multiopen(reader)
    values = tuple(
        _read_value(reader)
        for _field in dataclasses.fields(record_type)
    )
    try:
        return record_type(*values)
    except (TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3Error(
            "selected-trace wire record is malformed"
        ) from exc


def _write_value(writer: _Writer, value: object) -> None:
    if value is None:
        writer.pack("<B", _NONE)
        return
    if value is False:
        writer.pack("<B", _FALSE)
        return
    if value is True:
        writer.pack("<B", _TRUE)
        return
    if type(value) is int:
        magnitude = abs(value)
        if magnitude >= 1 << 64:
            raise ProofV3Error("selected-trace integer exceeds the wire cap")
        writer.pack("<BBQ", _INT, int(value < 0), magnitude)
        return
    if isinstance(value, bytes):
        if len(value) > MAX_SELECTED_TRACE_WIRE_BYTES:
            raise ProofV3Error("selected-trace byte string exceeds the cap")
        writer.pack("<BI", _BYTES, len(value))
        writer.raw(value)
        return
    if isinstance(value, str):
        writer.pack("<B", _TEXT)
        _write_text(writer, value)
        return
    if isinstance(value, tuple):
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise ProofV3Error("selected-trace tuple exceeds the item cap")
        if value and all(type(item) is int for item in value):
            if all(0 <= item < 1 << 64 for item in value):
                writer.pack("<BI", _U64_TUPLE, len(value))
                writer.raw(struct.pack(f"<{len(value)}Q", *value))
                return
            if all(-(1 << 63) <= item < 1 << 63 for item in value):
                writer.pack("<BI", _I64_TUPLE, len(value))
                writer.raw(struct.pack(f"<{len(value)}q", *value))
                return
        if value and all(isinstance(item, bytes) for item in value):
            writer.pack("<BI", _BYTES_TUPLE, len(value))
            for item in value:
                if len(item) > MAX_SELECTED_TRACE_WIRE_BYTES:
                    raise ProofV3Error(
                        "selected-trace byte string exceeds the cap"
                    )
                writer.pack("<I", len(item))
                writer.raw(item)
            return
        writer.pack("<BI", _TUPLE, len(value))
        for item in value:
            _write_value(writer, item)
        return
    if isinstance(value, dict):
        if len(value) > _MAX_MAP_ITEMS or any(
            not isinstance(key, str) for key in value
        ):
            raise ProofV3Error("selected-trace map is malformed")
        keys = tuple(sorted(value))
        writer.pack("<BI", _MAP, len(keys))
        for key in keys:
            _write_text(writer, key)
            _write_value(writer, value[key])
        return
    if dataclasses.is_dataclass(value):
        writer.pack("<B", _RECORD)
        _write_record(writer, value)
        return
    raise ProofV3Error(
        "selected-trace wire contains an unsupported value type"
    )


def _read_count(reader: _Reader, maximum: int, name: str) -> int:
    (count,) = reader.unpack("<I")
    if count > maximum:
        raise ProofV3Error(f"selected-trace {name} exceeds the item cap")
    return count


def _read_value(reader: _Reader) -> object:
    (kind,) = reader.unpack("<B")
    if kind == _NONE:
        return None
    if kind == _FALSE:
        return False
    if kind == _TRUE:
        return True
    if kind == _INT:
        negative, magnitude = reader.unpack("<BQ")
        if negative not in (0, 1) or (negative and magnitude == 0):
            raise ProofV3Error("selected-trace integer is not canonical")
        return -magnitude if negative else magnitude
    if kind == _BYTES:
        size = _read_count(
            reader,
            MAX_SELECTED_TRACE_WIRE_BYTES,
            "byte string",
        )
        return reader.read(size)
    if kind == _TEXT:
        return _read_text(reader)
    if kind == _U64_TUPLE:
        count = _read_count(reader, _MAX_CONTAINER_ITEMS, "tuple")
        return tuple(reader.unpack(f"<{count}Q"))
    if kind == _I64_TUPLE:
        count = _read_count(reader, _MAX_CONTAINER_ITEMS, "tuple")
        return tuple(reader.unpack(f"<{count}q"))
    if kind == _BYTES_TUPLE:
        count = _read_count(reader, _MAX_CONTAINER_ITEMS, "tuple")
        values = []
        for _ in range(count):
            size = _read_count(
                reader,
                MAX_SELECTED_TRACE_WIRE_BYTES,
                "byte string",
            )
            values.append(reader.read(size))
        return tuple(values)
    if kind == _TUPLE:
        count = _read_count(reader, _MAX_CONTAINER_ITEMS, "tuple")
        return tuple(_read_value(reader) for _ in range(count))
    if kind == _MAP:
        count = _read_count(reader, _MAX_MAP_ITEMS, "map")
        result = {}
        previous = None
        for _ in range(count):
            key = _read_text(reader)
            if previous is not None and key <= previous:
                raise ProofV3Error(
                    "selected-trace map keys are not canonical"
                )
            previous = key
            result[key] = _read_value(reader)
        return result
    if kind == _RECORD:
        return _read_record(reader)
    raise ProofV3Error("selected-trace wire value kind is unsupported")


def encode_goldilocks_selected_trace_v3(
    proof: GoldilocksSelectedTraceProofV3,
) -> bytes:
    """Encode one complete selected trace in the bounded canonical ABI."""

    if not isinstance(proof, GoldilocksSelectedTraceProofV3):
        raise ProofV3Error("selected-trace wire proof has a wrong type")
    writer = _Writer()
    writer.raw(_MAGIC)
    writer.pack("<B", _VERSION)
    _write_value(writer, proof)
    encoded = writer.finish()
    if len(encoded) > MAX_SELECTED_TRACE_WIRE_BYTES:
        raise ProofV3Error(
            "selected-trace canonical wire exceeds the byte budget"
        )
    return encoded


def decode_goldilocks_selected_trace_v3(
    encoded: bytes,
) -> GoldilocksSelectedTraceProofV3:
    """Decode one complete trace; unknown, oversized or trailing data fails."""

    if (
        not isinstance(encoded, bytes)
        or len(encoded) > MAX_SELECTED_TRACE_WIRE_BYTES
    ):
        raise ProofV3Error(
            "selected-trace canonical wire exceeds the byte budget"
        )
    reader = _Reader(encoded, "selected-trace canonical wire")
    if reader.read(len(_MAGIC)) != _MAGIC:
        raise ProofV3Error("selected-trace wire magic is unsupported")
    (version,) = reader.unpack("<B")
    if version != _VERSION:
        raise ProofV3Error("selected-trace wire version is unsupported")
    proof = _read_value(reader)
    reader.finish()
    if not isinstance(proof, GoldilocksSelectedTraceProofV3):
        raise ProofV3Error("selected-trace wire root has a wrong type")
    return proof


__all__ = [
    "GOLDILOCKS_SELECTED_TRACE_WIRE_ABI_V3",
    "MAX_SELECTED_TRACE_WIRE_BYTES",
    "decode_goldilocks_selected_trace_v3",
    "encode_goldilocks_selected_trace_v3",
]
