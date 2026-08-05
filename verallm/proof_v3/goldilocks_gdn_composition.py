"""Bounded GDN replay joined to the authenticated projection composition.

GDN is a short decode-suffix recurrence in the qualified v3 profile. Its
QKVZ/BA outputs and out-projection input are exact cells in the shared
projection PCS, so the compact path does not invent pre-nonce anchors for
those post-nonce intermediates. It only opens the real prompt-boundary
convolution/recurrent-state lanes, replays the signed recurrence, and compares
the result with the exact PCS-authenticated ``gdn_o`` input cells.

The projection composition must be verified before this module.  Production
admission therefore exposes the pair only through the selected-trace
coordinator, never as an independently accepted GDN proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from verallm.proof_v2.transition import ProofV2TransitionError
from verallm.proof_v3.attention_anchor_binding import (
    extract_execution_anchor_range_v3,
)
from verallm.proof_v3.economic_gdn_replay import (
    _decode,
    _replay,
    economic_gdn_runtime_columns_v3,
)
from verallm.proof_v3.economic_wire import (
    bits_to_scale_v3,
    scale_to_bits_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.execution_anchor import (
    ExecutionAnchorCommitmentV3,
    ExecutionAnchorLaneOpeningV3,
    build_execution_anchor_lane_opening_v3,
    execution_anchor_lane_bytes_v3,
    verify_execution_anchor_lane_v3,
)
from verallm.proof_v3.gdn_runtime_semantics import (
    GdnRuntimeSemanticsV3,
)
from verallm.proof_v3.goldilocks_projection_composition import (
    GoldilocksProjectionCompositionProofV3,
    GoldilocksProjectionClaimV3,
    goldilocks_projection_input_cells_v3,
    goldilocks_projection_output_cells_v3,
    goldilocks_projection_runtime_binding_v3,
    goldilocks_projection_runtime_cells_v3,
)
from verallm.proof_v3.lean_projection_fold import (
    lean_projection_operation_key_v3,
)
from zkllm.crypto.merkle import MerkleTree


GOLDILOCKS_GDN_COMPOSITION_ABI_V3: Final = (
    "gdn.bounded_replay.projection_links.v3"
)
MAX_GDN_COMPOSITION_LAYERS_V3: Final = 4

__all__ = [
    "GOLDILOCKS_GDN_COMPOSITION_ABI_V3",
    "GoldilocksGdnBoundaryWitnessV3",
    "GoldilocksGdnCaptureProofV3",
    "GoldilocksGdnClaimV3",
    "GoldilocksGdnCompositionProofV3",
    "GoldilocksGdnWitnessV3",
    "prove_goldilocks_gdn_composition_v3",
    "verify_goldilocks_gdn_composition_v3",
]


@dataclass(frozen=True, slots=True)
class GoldilocksGdnClaimV3:
    layer_index: int
    selected_value_heads: tuple[int, ...]
    row_map: tuple[tuple[int, int], ...]
    conv_state_anchor: ExecutionAnchorCommitmentV3
    recurrent_state_anchor: ExecutionAnchorCommitmentV3
    qkvz_projection_index: int
    ba_projection_index: int
    gdn_o_projection_index: int
    start_state_row: int = 0
    end_state_row: int | None = None

    def __post_init__(self) -> None:
        heads = tuple(self.selected_value_heads)
        row_map = tuple(self.row_map)
        positions = tuple(position for position, _row in row_map)
        rows = tuple(row for _position, row in row_map)
        indices = (
            self.qkvz_projection_index,
            self.ba_projection_index,
            self.gdn_o_projection_index,
        )
        checkpointed = self.end_state_row is not None
        expected_conv_stage = (
            f"l{self.layer_index}.gdn_conv_decode_checkpoints"
            if checkpointed
            else f"l{self.layer_index}.gdn_conv_prompt_boundary"
        )
        expected_recurrent_stage = (
            f"l{self.layer_index}.gdn_recurrent_decode_checkpoints"
            if checkpointed
            else f"l{self.layer_index}.gdn_recurrent_prompt_boundary"
        )
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or not 0 <= self.layer_index < 1 << 32
            or not heads
            or heads != tuple(sorted(set(heads)))
            or any(
                isinstance(head, bool)
                or not isinstance(head, int)
                or head < 0
                for head in heads
            )
            or not row_map
            or positions != tuple(sorted(set(positions)))
            or len(set(rows)) != len(rows)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for pair in row_map
                for value in pair
            )
            or not isinstance(
                self.conv_state_anchor,
                ExecutionAnchorCommitmentV3,
            )
            or not isinstance(
                self.recurrent_state_anchor,
                ExecutionAnchorCommitmentV3,
            )
            or self.conv_state_anchor.stage_id != expected_conv_stage
            or self.recurrent_state_anchor.stage_id
            != expected_recurrent_stage
            or (
                not checkpointed
                and (
                    self.start_state_row != 0
                    or self.conv_state_anchor.row_count != 1
                    or self.recurrent_state_anchor.row_count != 1
                )
            )
            or (
                checkpointed
                and (
                    isinstance(self.start_state_row, bool)
                    or not isinstance(self.start_state_row, int)
                    or isinstance(self.end_state_row, bool)
                    or not isinstance(self.end_state_row, int)
                    or not 0
                    <= self.start_state_row
                    < self.end_state_row
                    < self.conv_state_anchor.row_count
                    or self.recurrent_state_anchor.row_count
                    != self.conv_state_anchor.row_count
                )
            )
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                for index in indices
            )
            or len(set(indices)) != len(indices)
        ):
            raise ProofV3Error("GDN composition claim is malformed")
        object.__setattr__(self, "selected_value_heads", heads)
        object.__setattr__(self, "row_map", row_map)


@dataclass(frozen=True, slots=True)
class GoldilocksGdnBoundaryWitnessV3:
    row_bytes: bytes
    row_tree: MerkleTree
    row_index: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.row_bytes, bytes)
            or not isinstance(self.row_tree, MerkleTree)
            or isinstance(self.row_index, bool)
            or not isinstance(self.row_index, int)
            or not 0 <= self.row_index < self.row_tree.num_leaves
        ):
            raise ProofV3Error("GDN boundary witness is malformed")


@dataclass(frozen=True, slots=True)
class GoldilocksGdnWitnessV3:
    claim: GoldilocksGdnClaimV3
    conv_state: GoldilocksGdnBoundaryWitnessV3
    recurrent_state: GoldilocksGdnBoundaryWitnessV3
    end_conv_state: GoldilocksGdnBoundaryWitnessV3 | None = None
    end_recurrent_state: GoldilocksGdnBoundaryWitnessV3 | None = None
    runtime_rows: tuple[tuple[int, bytes, bytes, bytes], ...] = ()

    def __post_init__(self) -> None:
        checkpointed = self.claim.end_state_row is not None
        runtime_rows = tuple(self.runtime_rows)
        if (
            not isinstance(self.claim, GoldilocksGdnClaimV3)
            or not isinstance(
                self.conv_state,
                GoldilocksGdnBoundaryWitnessV3,
            )
            or not isinstance(
                self.recurrent_state,
                GoldilocksGdnBoundaryWitnessV3,
            )
            or len(self.conv_state.row_bytes)
            != self.claim.conv_state_anchor.row_width
            or len(self.recurrent_state.row_bytes)
            != self.claim.recurrent_state_anchor.row_width
            or self.conv_state.row_tree.root
            != self.claim.conv_state_anchor.root
            or self.recurrent_state.row_tree.root
            != self.claim.recurrent_state_anchor.root
            or self.conv_state.row_index != self.claim.start_state_row
            or self.recurrent_state.row_index != self.claim.start_state_row
            or checkpointed
            != (
                self.end_conv_state is not None
                and self.end_recurrent_state is not None
            )
            or (
                checkpointed
                and (
                    not isinstance(
                        self.end_conv_state,
                        GoldilocksGdnBoundaryWitnessV3,
                    )
                    or not isinstance(
                        self.end_recurrent_state,
                        GoldilocksGdnBoundaryWitnessV3,
                    )
                    or self.end_conv_state.row_index
                    != self.claim.end_state_row
                    or self.end_recurrent_state.row_index
                    != self.claim.end_state_row
                    or len(self.end_conv_state.row_bytes)
                    != self.claim.conv_state_anchor.row_width
                    or len(self.end_recurrent_state.row_bytes)
                    != self.claim.recurrent_state_anchor.row_width
                    or self.end_conv_state.row_tree.root
                    != self.claim.conv_state_anchor.root
                    or self.end_recurrent_state.row_tree.root
                    != self.claim.recurrent_state_anchor.root
                )
            )
            or (
                runtime_rows
                and (
                    tuple(record[0] for record in runtime_rows)
                    != tuple(
                        position
                        for position, _row in self.claim.row_map
                    )
                    or any(
                        not isinstance(record[1], bytes)
                        or not isinstance(record[2], bytes)
                        or not isinstance(record[3], bytes)
                        for record in runtime_rows
                    )
                )
            )
            or (checkpointed and not runtime_rows)
        ):
            raise ProofV3Error("GDN composition witness is inconsistent")
        object.__setattr__(self, "runtime_rows", runtime_rows)


@dataclass(frozen=True, slots=True)
class GoldilocksGdnCaptureProofV3:
    conv_state_openings: tuple[ExecutionAnchorLaneOpeningV3, ...]
    recurrent_state_openings: tuple[ExecutionAnchorLaneOpeningV3, ...]
    end_conv_state_openings: tuple[ExecutionAnchorLaneOpeningV3, ...] = ()
    end_recurrent_state_openings: tuple[
        ExecutionAnchorLaneOpeningV3, ...
    ] = ()
    runtime_rows: tuple[tuple[int, bytes, bytes, bytes], ...] = ()

    def __post_init__(self) -> None:
        conv = tuple(self.conv_state_openings)
        recurrent = tuple(self.recurrent_state_openings)
        end_conv = tuple(self.end_conv_state_openings)
        end_recurrent = tuple(self.end_recurrent_state_openings)
        runtime_rows = tuple(self.runtime_rows)
        if (
            not conv
            or not recurrent
            or not all(
                isinstance(item, ExecutionAnchorLaneOpeningV3)
                for item in (*conv, *recurrent, *end_conv, *end_recurrent)
            )
            or bool(end_conv) != bool(end_recurrent)
            or any(
                not isinstance(record, tuple)
                or len(record) != 4
                or isinstance(record[0], bool)
                or not isinstance(record[0], int)
                or record[0] < 0
                or not all(
                    isinstance(value, bytes) and value
                    for value in record[1:]
                )
                for record in runtime_rows
            )
            or tuple(record[0] for record in runtime_rows)
            != tuple(sorted({record[0] for record in runtime_rows}))
        ):
            raise ProofV3Error("GDN state proof is malformed")
        object.__setattr__(self, "conv_state_openings", conv)
        object.__setattr__(self, "recurrent_state_openings", recurrent)
        object.__setattr__(self, "end_conv_state_openings", end_conv)
        object.__setattr__(
            self,
            "end_recurrent_state_openings",
            end_recurrent,
        )
        object.__setattr__(self, "runtime_rows", runtime_rows)


@dataclass(frozen=True, slots=True)
class GoldilocksGdnCompositionProofV3:
    projection_binding_digest: bytes
    captures: tuple[GoldilocksGdnCaptureProofV3, ...]

    def __post_init__(self) -> None:
        captures = tuple(self.captures)
        if (
            not isinstance(self.projection_binding_digest, bytes)
            or len(self.projection_binding_digest) != 32
            or not captures
            or len(captures) > MAX_GDN_COMPOSITION_LAYERS_V3
            or not all(
                isinstance(item, GoldilocksGdnCaptureProofV3)
                for item in captures
            )
        ):
            raise ProofV3Error("GDN composition proof is malformed")
        object.__setattr__(self, "captures", captures)


def _element_bytes(encoding_id: str) -> int:
    try:
        return {
            "fp16.v1": 2,
            "bf16.v1": 2,
            "fp32.v1": 4,
        }[encoding_id]
    except KeyError as exc:
        raise ProofV3VerificationError(
            "GDN state encoding is unsupported"
        ) from exc


def _runs(columns: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    if not columns:
        return ()
    result = []
    start = previous = columns[0]
    for column in columns[1:]:
        if column != previous + 1:
            result.append((start, previous - start + 1))
            start = column
        previous = column
    result.append((start, previous - start + 1))
    return tuple(result)


def _lane_indices(
    commitment: ExecutionAnchorCommitmentV3,
    ranges,
) -> tuple[int, ...]:
    lane_bytes = execution_anchor_lane_bytes_v3(commitment.stage_id)
    indices = set()
    for byte_start, byte_length in ranges:
        if (
            byte_start < 0
            or byte_length <= 0
            or byte_start + byte_length > commitment.row_width
        ):
            raise ProofV3VerificationError(
                "GDN state range exceeds its execution anchor"
            )
        indices.update(
            range(
                byte_start // lane_bytes,
                (byte_start + byte_length - 1) // lane_bytes + 1,
            )
        )
    return tuple(sorted(indices))


def _state_plan(claim: GoldilocksGdnClaimV3, signed):
    parameters = signed.parameters().replay_parameters()
    nk = parameters.num_key_heads
    nv = parameters.num_value_heads
    dk = parameters.key_head_dim
    dv = parameters.value_head_dim
    if (
        any(head >= nv for head in claim.selected_value_heads)
        or nv % nk
        or claim.conv_state_anchor.row_width != signed.conv_state_bytes
        or claim.recurrent_state_anchor.row_width
        != signed.recurrent_state_bytes
    ):
        raise ProofV3VerificationError(
            "GDN signed state geometry is inconsistent"
        )
    qkvz_columns, ba_columns, output_columns = (
        economic_gdn_runtime_columns_v3(
            parameters=parameters,
            selected_value_heads=claim.selected_value_heads,
        )
    )
    key_width = nk * dk
    value_width = nv * dv
    conv_width = 2 * key_width + value_width
    conv_columns = tuple(
        column for column in qkvz_columns if column < conv_width
    )
    conv_bytes = _element_bytes(signed.conv_state_encoding_id)
    recurrent_bytes = _element_bytes(
        signed.recurrent_state_encoding_id
    )
    conv_ranges = tuple(
        (
            (step * conv_width + start) * conv_bytes,
            width * conv_bytes,
        )
        for step in range(parameters.conv_kernel_size - 1)
        for start, width in _runs(conv_columns)
    )
    head_bytes = dv * dk * recurrent_bytes
    recurrent_ranges = tuple(
        (head * head_bytes, head_bytes)
        for head in claim.selected_value_heads
    )
    return (
        parameters,
        qkvz_columns,
        ba_columns,
        output_columns,
        conv_columns,
        conv_ranges,
        recurrent_ranges,
    )


def _build_openings(
    *,
    commitment: ExecutionAnchorCommitmentV3,
    witness: GoldilocksGdnBoundaryWitnessV3,
    lane_indices: tuple[int, ...],
) -> tuple[ExecutionAnchorLaneOpeningV3, ...]:
    return tuple(
        build_execution_anchor_lane_opening_v3(
            commitment=commitment,
            row_index=witness.row_index,
            row_bytes=witness.row_bytes,
            row_tree=witness.row_tree,
            lane_index=lane,
        )
        for lane in lane_indices
    )


def _opening_map(
    *,
    commitment: ExecutionAnchorCommitmentV3,
    openings,
    expected_lanes: tuple[int, ...],
    row_index: int,
) -> dict[tuple[int, int], ExecutionAnchorLaneOpeningV3]:
    openings_t = tuple(openings)
    if tuple(
        (opening.row_index, opening.lane_index)
        for opening in openings_t
    ) != tuple((row_index, lane) for lane in expected_lanes):
        raise ProofV3VerificationError(
            "GDN state openings do not match the derived lanes"
        )
    result = {}
    for opening in openings_t:
        verify_execution_anchor_lane_v3(
            commitment=commitment,
            opening=opening,
        )
        result[(row_index, opening.lane_index)] = opening
    return result


def _extract_runs(
    *,
    commitment: ExecutionAnchorCommitmentV3,
    row_index: int,
    columns: tuple[int, ...],
    element_bytes: int,
    encoding_id: str,
    openings,
) -> np.ndarray:
    values = []
    for start, width in _runs(columns):
        raw = extract_execution_anchor_range_v3(
            commitment=commitment,
            row_index=row_index,
            byte_start=start * element_bytes,
            byte_length=width * element_bytes,
            openings=openings,
        )
        values.extend(_decode(raw, encoding_id).tolist())
    return np.asarray(values, dtype=np.float32)


def _projection_claims(
    *,
    claim: GoldilocksGdnClaimV3,
    projection_claims: tuple[GoldilocksProjectionClaimV3, ...],
    qkvz_columns: tuple[int, ...],
    ba_columns: tuple[int, ...],
    output_columns: tuple[int, ...],
):
    indices = (
        (claim.qkvz_projection_index, "gdn_qkvz"),
        (claim.ba_projection_index, "gdn_ba"),
        (claim.gdn_o_projection_index, "gdn_o"),
    )
    chronological_rows = tuple(row for _position, row in claim.row_map)
    selected_rows = tuple(sorted(chronological_rows))
    position_by_row = {
        row: position for position, row in claim.row_map
    }
    projection_positions = tuple(
        position_by_row[row] for row in selected_rows
    )
    result = []
    for index, role in indices:
        if index >= len(projection_claims):
            raise ProofV3VerificationError(
                "GDN projection reference is outside the inventory"
            )
        projection = projection_claims[index]
        runtime = projection.runtime
        if (
            projection.operation.operation_key
            != lean_projection_operation_key_v3(
                layer_index=claim.layer_index,
                projection=role,
            )
            or projection.selected_rows != selected_rows
        ):
            raise ProofV3VerificationError(
                "GDN projection reference is inconsistent"
            )
        expected_columns = {
            "gdn_qkvz": qkvz_columns,
            "gdn_ba": ba_columns,
            "gdn_o": output_columns,
        }[role]
        required_cells = {
            (row, column)
            for row in selected_rows
            for column in expected_columns
        }
        if role == "gdn_o":
            if projection.x_anchor is not None:
                if (
                    projection.x_anchor.commitment.stage_id
                    != f"l{claim.layer_index}.gdn_o_input"
                    or projection.x_anchor.anchor_rows
                    != projection_positions
                    or projection.x_anchor.source_column_offset != 0
                ):
                    raise ProofV3VerificationError(
                        "GDN output-projection input anchor is inconsistent"
                    )
            elif not required_cells.issubset(
                set(projection.consumer_input_cells)
            ):
                raise ProofV3VerificationError(
                    "derived GDN output-projection input is incomplete"
                )
            if (
                runtime is not None
                and runtime.input_columns
                and runtime.input_columns != expected_columns
            ):
                raise ProofV3VerificationError(
                    "GDN output-projection input inventory is inconsistent"
                )
        elif runtime is not None:
            if (
                runtime.output_columns != expected_columns
                or (
                    runtime.y_anchor is not None
                    and (
                        runtime.y_anchor.commitment.stage_id
                        != f"l{claim.layer_index}.{role}_output"
                        or runtime.y_anchor.anchor_rows
                        != projection_positions
                        or runtime.y_anchor.source_column_offset != 0
                    )
                )
            ):
                raise ProofV3VerificationError(
                    "GDN projection runtime inventory is inconsistent"
                )
        elif (
            not projection.weight_scale_bits
            or not required_cells.issubset(
                set(projection.consumer_output_cells)
            )
        ):
            raise ProofV3VerificationError(
                "derived GDN projection output is incomplete"
            )
        result.append(projection)
    return tuple(result)


def _projection_rows(
    *,
    claim: GoldilocksGdnClaimV3,
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims: tuple[GoldilocksProjectionClaimV3, ...],
    projections,
    qkvz_columns: tuple[int, ...],
    ba_columns: tuple[int, ...],
    output_columns: tuple[int, ...],
):
    qkvz_projection, ba_projection, gdn_o_projection = projections
    specifications = (
        (
            claim.qkvz_projection_index,
            qkvz_columns,
            qkvz_projection,
            "output",
        ),
        (
            claim.ba_projection_index,
            ba_columns,
            ba_projection,
            "output",
        ),
        (
            claim.gdn_o_projection_index,
            output_columns,
            gdn_o_projection,
            "input",
        ),
    )
    all_rows = []
    for (
        projection_index,
        columns,
        projection,
        kind,
    ) in specifications:
        if kind == "output":
            runtime = projection.runtime
            if runtime is not None:
                cells = {
                    (row, column): value
                    for row, column, _surrogate, value in (
                        goldilocks_projection_runtime_cells_v3(
                            projection_proof,
                            projection_claims,
                            claim_index=projection_index,
                        )
                    )
                }
                scale = bits_to_scale_v3(
                    runtime.y_oracle.scale_bits
                )
            else:
                cells = {
                    (row, column): value
                    for row, column, value in (
                        goldilocks_projection_output_cells_v3(
                            projection_proof,
                            projection_claims,
                            claim_index=projection_index,
                        )
                    )
                }
                scale = (
                    bits_to_scale_v3(projection.x_oracle.scale_bits)
                    * bits_to_scale_v3(projection.weight_scale_bits)
                )
        else:
            cells = {
                (row, column): value
                for row, column, value in (
                    goldilocks_projection_input_cells_v3(
                        projection_proof,
                        projection_claims,
                        claim_index=projection_index,
                    )
                )
            }
            scale = bits_to_scale_v3(
                projection.x_oracle.scale_bits
            )
        rows = []
        for _position, projection_row in claim.row_map:
            try:
                row = tuple(
                    cells[(projection_row, column)] * scale
                    for column in columns
                )
            except KeyError as exc:
                raise ProofV3VerificationError(
                    "GDN projection selected-cell inventory is incomplete"
                ) from exc
            rows.append(row)
        all_rows.append(np.asarray(rows, dtype=np.float32))
    return tuple(all_rows)


def _verify_capture(
    *,
    claim: GoldilocksGdnClaimV3,
    capture: GoldilocksGdnCaptureProofV3,
    signed,
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims: tuple[GoldilocksProjectionClaimV3, ...],
) -> None:
    (
        parameters,
        qkvz_columns,
        ba_columns,
        output_columns,
        conv_columns,
        conv_ranges,
        recurrent_ranges,
    ) = _state_plan(claim, signed)
    if len(claim.row_map) > signed.max_decode_replay_rows:
        raise ProofV3VerificationError(
            "GDN replay exceeds the signed decode-row bound"
        )
    projections = _projection_claims(
        claim=claim,
        projection_claims=projection_claims,
        qkvz_columns=qkvz_columns,
        ba_columns=ba_columns,
        output_columns=output_columns,
    )
    qkvz_compact, ba_compact, output_compact = _projection_rows(
        claim=claim,
        projection_proof=projection_proof,
        projection_claims=projection_claims,
        projections=projections,
        qkvz_columns=qkvz_columns,
        ba_columns=ba_columns,
        output_columns=output_columns,
    )
    if capture.runtime_rows:
        positions = tuple(position for position, _row in claim.row_map)
        records = tuple(capture.runtime_rows)
        runtime_element_bytes = _element_bytes(
            signed.runtime_encoding_id
        )
        if (
            tuple(record[0] for record in records) != positions
            or any(
                len(record[1])
                != len(qkvz_columns) * runtime_element_bytes
                or len(record[2])
                != len(ba_columns) * runtime_element_bytes
                or len(record[3])
                != len(output_columns) * runtime_element_bytes
                for record in records
            )
        ):
            raise ProofV3VerificationError(
                "GDN native runtime rows are not canonical"
            )
        native_qkvz = np.stack(
            tuple(
                _decode(record[1], signed.runtime_encoding_id)
                for record in records
            )
        )
        native_ba = np.stack(
            tuple(
                _decode(record[2], signed.runtime_encoding_id)
                for record in records
            )
        )
        native_output = np.stack(
            tuple(
                _decode(record[3], signed.runtime_encoding_id)
                for record in records
            )
        )
        qkvz_projection, ba_projection, gdn_o_projection = projections
        if (
            qkvz_projection.runtime is None
            or ba_projection.runtime is None
        ):
            raise ProofV3VerificationError(
                "GDN native runtime rows require runtime projection bindings"
            )
        runtime_scales = (
            bits_to_scale_v3(
                qkvz_projection.runtime.y_oracle.scale_bits
            ),
            bits_to_scale_v3(
                ba_projection.runtime.y_oracle.scale_bits
            ),
            bits_to_scale_v3(gdn_o_projection.x_oracle.scale_bits),
        )
        for name, native, quantized, scale, selected_scale in (
            ("qkvz", native_qkvz, qkvz_compact, runtime_scales[0], True),
            ("ba", native_ba, ba_compact, runtime_scales[1], True),
            (
                "gdn_o",
                native_output,
                output_compact,
                runtime_scales[2],
                False,
            ),
        ):
            if selected_scale:
                canonical_scale = max(
                    float(np.max(np.abs(native), initial=0.0)),
                    1e-8,
                ) / 127.0
                if (
                    scale_to_bits_v3(canonical_scale)
                    != scale_to_bits_v3(scale)
                ):
                    raise ProofV3VerificationError(
                        f"GDN native {name} scale is not derived from the "
                        "nonce-selected runtime cells"
                    )
            error = np.abs(native - quantized)
            allowance = (
                0.5001 * scale
                + np.finfo(np.float32).eps
                * np.maximum(1.0, np.abs(native))
            )
            if native.shape != quantized.shape or bool(
                np.any(error > allowance)
            ):
                raise ProofV3VerificationError(
                    f"GDN native {name} rows disagree with the "
                    "projection-bound quantization"
                )
        qkvz_compact = native_qkvz
        ba_compact = native_ba
        output_compact = native_output
    elif claim.end_state_row is not None:
        raise ProofV3VerificationError(
            "checkpointed GDN replay lacks native runtime rows"
        )

    conv_lanes = _lane_indices(
        claim.conv_state_anchor,
        conv_ranges,
    )
    recurrent_lanes = _lane_indices(
        claim.recurrent_state_anchor,
        recurrent_ranges,
    )
    conv_openings = _opening_map(
        commitment=claim.conv_state_anchor,
        openings=capture.conv_state_openings,
        expected_lanes=conv_lanes,
        row_index=claim.start_state_row,
    )
    recurrent_openings = _opening_map(
        commitment=claim.recurrent_state_anchor,
        openings=capture.recurrent_state_openings,
        expected_lanes=recurrent_lanes,
        row_index=claim.start_state_row,
    )
    end_conv_openings = None
    end_recurrent_openings = None
    if claim.end_state_row is not None:
        end_conv_openings = _opening_map(
            commitment=claim.conv_state_anchor,
            openings=capture.end_conv_state_openings,
            expected_lanes=conv_lanes,
            row_index=claim.end_state_row,
        )
        end_recurrent_openings = _opening_map(
            commitment=claim.recurrent_state_anchor,
            openings=capture.end_recurrent_state_openings,
            expected_lanes=recurrent_lanes,
            row_index=claim.end_state_row,
        )
    elif (
        capture.end_conv_state_openings
        or capture.end_recurrent_state_openings
    ):
        raise ProofV3VerificationError(
            "legacy GDN replay supplied checkpoint-end openings"
        )

    nk = parameters.num_key_heads
    nv = parameters.num_value_heads
    dk = parameters.key_head_dim
    dv = parameters.value_head_dim
    key_width = nk * dk
    value_width = nv * dv
    conv_width = 2 * key_width + value_width
    conv = np.zeros(
        (parameters.conv_kernel_size - 1, conv_width),
        dtype=np.float32,
    )
    conv_element_bytes = _element_bytes(
        signed.conv_state_encoding_id
    )
    for step in range(parameters.conv_kernel_size - 1):
        for start, width in _runs(conv_columns):
            raw = extract_execution_anchor_range_v3(
                commitment=claim.conv_state_anchor,
                row_index=claim.start_state_row,
                byte_start=(
                    (step * conv_width + start) * conv_element_bytes
                ),
                byte_length=width * conv_element_bytes,
                openings=conv_openings,
            )
            conv[step, start:start + width] = _decode(
                raw,
                signed.conv_state_encoding_id,
            )

    recurrent_element_bytes = _element_bytes(
        signed.recurrent_state_encoding_id
    )
    recurrent_head_bytes = dv * dk * recurrent_element_bytes
    recurrent = np.stack(
        [
            _decode(
                extract_execution_anchor_range_v3(
                    commitment=claim.recurrent_state_anchor,
                    row_index=claim.start_state_row,
                    byte_start=head * recurrent_head_bytes,
                    byte_length=recurrent_head_bytes,
                    openings=recurrent_openings,
                ),
                signed.recurrent_state_encoding_id,
            ).reshape(dv, dk)
            for head in claim.selected_value_heads
        ]
    )

    qkvz = np.zeros(
        (len(claim.row_map), conv_width + value_width),
        dtype=np.float32,
    )
    ba = np.zeros(
        (len(claim.row_map), 2 * nv),
        dtype=np.float32,
    )
    qkvz[:, qkvz_columns] = qkvz_compact
    ba[:, ba_columns] = ba_compact
    try:
        replay = _replay(
            qkvz=qkvz,
            ba=ba,
            conv_state=conv,
            recurrent_state=recurrent,
            parameters=parameters,
            runtime_encoding_id=signed.runtime_encoding_id,
            conv_state_encoding_id=signed.conv_state_encoding_id,
            recurrent_state_encoding_id=(
                signed.recurrent_state_encoding_id
            ),
            selected_value_heads=claim.selected_value_heads,
        )
    except (
        ProofV2TransitionError,
        ValueError,
        FloatingPointError,
    ) as exc:
        raise ProofV3VerificationError(
            "GDN bounded recurrence replay failed"
        ) from exc
    atol = signed.output_atol_q24 / float(1 << 24)
    rtol = signed.output_rtol_q24 / float(1 << 24)
    if replay.out_projection_input.shape != output_compact.shape:
        raise ProofV3VerificationError(
            f"GDN layer {claim.layer_index} replay does not match the "
            "authenticated out-projection input: "
            f"replay_shape={replay.out_projection_input.shape}, "
            f"capture_shape={output_compact.shape}"
        )
    absolute_error = np.abs(
        replay.out_projection_input - output_compact
    )
    allowed_error = atol + rtol * np.abs(
        replay.out_projection_input
    )
    violation = absolute_error - allowed_error
    if bool(np.any(violation > 0.0)):
        row, column = np.unravel_index(
            int(np.argmax(violation)),
            violation.shape,
        )
        raise ProofV3VerificationError(
            f"GDN layer {claim.layer_index} replay does not match the "
            "authenticated out-projection input: "
            f"row={row}, column={column}, "
            f"replay={float(replay.out_projection_input[row, column]):.9g}, "
            f"capture={float(output_compact[row, column]):.9g}, "
            f"absolute_error={float(absolute_error[row, column]):.9g}, "
            f"allowed_error={float(allowed_error[row, column]):.9g}"
        )
    if claim.end_state_row is not None:
        end_conv = np.zeros_like(replay.conv_state_after)
        for step in range(parameters.conv_kernel_size - 1):
            for start, width in _runs(conv_columns):
                raw = extract_execution_anchor_range_v3(
                    commitment=claim.conv_state_anchor,
                    row_index=claim.end_state_row,
                    byte_start=(
                        (step * conv_width + start)
                        * conv_element_bytes
                    ),
                    byte_length=width * conv_element_bytes,
                    openings=end_conv_openings,
                )
                end_conv[step, start:start + width] = _decode(
                    raw,
                    signed.conv_state_encoding_id,
                )
        end_recurrent = np.stack(
            [
                _decode(
                    extract_execution_anchor_range_v3(
                        commitment=claim.recurrent_state_anchor,
                        row_index=claim.end_state_row,
                        byte_start=head * recurrent_head_bytes,
                        byte_length=recurrent_head_bytes,
                        openings=end_recurrent_openings,
                    ),
                    signed.recurrent_state_encoding_id,
                ).reshape(dv, dk)
                for head in claim.selected_value_heads
            ]
        )
        conv_error = np.abs(
            replay.conv_state_after[:, conv_columns]
            - end_conv[:, conv_columns]
        )
        recurrent_error = np.abs(
            replay.recurrent_state_after - end_recurrent
        )
        conv_allowed = signed.conv_state_atol_q24 / float(1 << 24)
        recurrent_allowed = (
            signed.recurrent_state_atol_q24 / float(1 << 24)
        )
        if bool(
            np.any(conv_error > conv_allowed)
            or np.any(recurrent_error > recurrent_allowed)
        ):
            raise ProofV3VerificationError(
                f"GDN layer {claim.layer_index} replay does not reach the "
                "authenticated end checkpoint within its signed corridor"
            )


def prove_goldilocks_gdn_composition_v3(
    *,
    witnesses,
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims,
    semantics: GdnRuntimeSemanticsV3,
) -> GoldilocksGdnCompositionProofV3:
    """Build the bounded state openings and validate the linked replay."""

    witnesses_t = tuple(witnesses)
    projection_claims_t = tuple(projection_claims)
    if (
        not witnesses_t
        or len(witnesses_t) > MAX_GDN_COMPOSITION_LAYERS_V3
        or not isinstance(semantics, GdnRuntimeSemanticsV3)
    ):
        raise ProofV3Error("GDN composition witness inventory is malformed")
    captures = []
    for witness in witnesses_t:
        signed = semantics.layer_for(witness.claim.layer_index)
        (
            _parameters,
            _qkvz,
            _ba,
            _output,
            _conv_columns,
            conv_ranges,
            recurrent_ranges,
        ) = _state_plan(witness.claim, signed)
        capture = GoldilocksGdnCaptureProofV3(
            conv_state_openings=_build_openings(
                commitment=witness.claim.conv_state_anchor,
                witness=witness.conv_state,
                lane_indices=_lane_indices(
                    witness.claim.conv_state_anchor,
                    conv_ranges,
                ),
            ),
            recurrent_state_openings=_build_openings(
                commitment=witness.claim.recurrent_state_anchor,
                witness=witness.recurrent_state,
                lane_indices=_lane_indices(
                    witness.claim.recurrent_state_anchor,
                    recurrent_ranges,
                ),
            ),
            end_conv_state_openings=(
                ()
                if witness.end_conv_state is None
                else _build_openings(
                    commitment=witness.claim.conv_state_anchor,
                    witness=witness.end_conv_state,
                    lane_indices=_lane_indices(
                        witness.claim.conv_state_anchor,
                        conv_ranges,
                    ),
                )
            ),
            end_recurrent_state_openings=(
                ()
                if witness.end_recurrent_state is None
                else _build_openings(
                    commitment=witness.claim.recurrent_state_anchor,
                    witness=witness.end_recurrent_state,
                    lane_indices=_lane_indices(
                        witness.claim.recurrent_state_anchor,
                        recurrent_ranges,
                    ),
                )
            ),
            runtime_rows=witness.runtime_rows,
        )
        _verify_capture(
            claim=witness.claim,
            capture=capture,
            signed=signed,
            projection_proof=projection_proof,
            projection_claims=projection_claims_t,
        )
        captures.append(capture)
    return GoldilocksGdnCompositionProofV3(
        projection_binding_digest=(
            goldilocks_projection_runtime_binding_v3(
                projection_proof,
                projection_claims_t,
            )
        ),
        captures=tuple(captures),
    )


def verify_goldilocks_gdn_composition_v3(
    proof: object,
    *,
    claims,
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims,
    semantics: GdnRuntimeSemanticsV3,
) -> None:
    """Verify GDN replay after the referenced projection proof is verified."""

    try:
        if not isinstance(proof, GoldilocksGdnCompositionProofV3):
            raise ProofV3VerificationError(
                "GDN composition proof has a wrong type"
            )
        claims_t = tuple(claims)
        projection_claims_t = tuple(projection_claims)
        if (
            not isinstance(semantics, GdnRuntimeSemanticsV3)
            or len(claims_t) != len(proof.captures)
            or tuple(claim.layer_index for claim in claims_t)
            != tuple(
                sorted({claim.layer_index for claim in claims_t})
            )
            or proof.projection_binding_digest
            != goldilocks_projection_runtime_binding_v3(
                projection_proof,
                projection_claims_t,
            )
        ):
            raise ProofV3VerificationError(
                "GDN composition inventory is inconsistent"
            )
        for claim, capture in zip(
            claims_t,
            proof.captures,
            strict=True,
        ):
            _verify_capture(
                claim=claim,
                capture=capture,
                signed=semantics.layer_for(claim.layer_index),
                projection_proof=projection_proof,
                projection_claims=projection_claims_t,
            )
    except ProofV3VerificationError:
        raise
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        ProofV3Error,
    ) as exc:
        raise ProofV3VerificationError(
            "GDN composition proof is malformed"
        ) from exc
