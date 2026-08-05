"""Succinct selected-row projection composition for proof-v3.

The transcript is deliberately two phase:

1. selected captured X/S rows are PCS-committed;
2. four output-fold coefficient rows are derived;
3. folded static vectors Z = W*c are PCS-committed; and
4. row/fold aggregation plus static-catalog bridge challenges are derived.

For every registered operation the proof checks

    sum[f,r,i] beta[f] alpha[r] X[r,i] Z[f,i]
      == sum[f,r,j] beta[f] alpha[r] S[r,j] c[f,j].

Production-bound claims sample X directly against the pre-nonce execution
anchor. S is a derived relation value and therefore needs no second raw
capture opening. The legacy X/S helper-root path remains explicit for
reference tests only. Z is bound to the validator-owned signed Pallas catalog
by the cross-field static bridge. All Goldilocks terminal claims share the
packed-group batched opening.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, replace
from typing import Final

from verallm.proof_v3.economic_challenge import (
    CORRIDOR_REL_COEFF_DEN_V3,
    CORRIDOR_REL_COEFF_NUM_V3,
)
from verallm.proof_v3.economic_commitment import (
    EconomicCommittedOracleV3,
    field_to_signed_v3,
    signed_to_field_v3,
)
from verallm.proof_v3.economic_wire import (
    VALUE_MODE_BOUNDED,
    VALUE_MODE_INT8,
    EconomicOracleCommitmentV3,
    bits_to_scale_v3,
    bounded_byte_width_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.execution_anchor import ExecutionAnchorCommitmentV3
from verallm.proof_v3.goldilocks_capture_pcs_binding import (
    GoldilocksCapturePcsBindingProofV3,
    prove_goldilocks_capture_pcs_binding_v3,
    verify_goldilocks_capture_pcs_binding_v3,
)
from verallm.proof_v3.goldilocks_execution_anchor_pcs_binding import (
    GoldilocksExecutionAnchorPcsBindingProofV3,
    build_goldilocks_execution_anchor_lane_openings_v3,
    derive_goldilocks_execution_anchor_pcs_lanes_v3,
    prove_goldilocks_execution_anchor_pcs_binding_v3,
    verify_goldilocks_execution_anchor_pcs_binding_v3,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.goldilocks_static_catalog_bridge import (
    GoldilocksStaticCatalogBridgeProofV3,
    prove_goldilocks_static_catalog_bridge_v3,
    verify_goldilocks_static_catalog_bridge_v3,
)
from verallm.proof_v3.goldilocks_succinct_batch_opening import (
    BatchClaimCheckerV3,
    BatchOpeningCollectorV3,
)
from verallm.proof_v3.goldilocks_succinct_product_argument_reference import (
    GoldilocksSuccinctProductProofV3,
    GoldilocksSuccinctProductStatementV3,
    prove_goldilocks_succinct_product_v3,
    verify_goldilocks_succinct_product_v3,
)
from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
    SuccinctEqFoldProofV3,
    VariableColumnGroupPlanV3,
    _mle_eval_msb_local,
    column_pcs_statement_v3,
    commit_succinct_variable_column_groups_v3,
    pcs_coset_profile_v3,
    plan_succinct_variable_column_groups_v3,
    prove_succinct_public_fold_v3,
    verify_succinct_public_fold_v3,
)
from verallm.proof_v3.lean_projection_fold import (
    LEAN_PROJECTION_FOLD_COUNT_V3,
    LeanProjectionCatalogOperationV3,
)
from zkllm.crypto.merkle import MerkleTree


GOLDILOCKS_PROJECTION_COMPOSITION_ABI_V3: Final = (
    "projection.selected_rows.goldilocks_fri.pallas_catalog.v8"
)
PROJECTION_HELPER_ROOT_BINDING_MODE_V3: Final = "helper_roots.reference.v1"
PROJECTION_EXECUTION_ANCHOR_BINDING_MODE_V3: Final = (
    "execution_anchor_x.derived_s.v1"
)
MAX_PROJECTION_COMPOSITION_OPERATIONS_V3: Final = 64
MAX_PROJECTION_GROUP_CELLS_V3: Final = 1 << 24

_TRANSCRIPT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_PROJECTION_COMPOSITION/V7"
)
_U31_MAX: Final = (1 << 31) - 1

__all__ = [
    "GOLDILOCKS_PROJECTION_COMPOSITION_ABI_V3",
    "GoldilocksProjectionCaptureProofV3",
    "GoldilocksProjectionAnchorClaimV3",
    "GoldilocksProjectionAnchorWitnessV3",
    "GoldilocksProjectionClaimV3",
    "GoldilocksProjectionCompositionProofV3",
    "GoldilocksProjectionGroupCommitmentV3",
    "GoldilocksProjectionRelationProofV3",
    "GoldilocksProjectionRuntimeClaimV3",
    "GoldilocksProjectionWitnessV3",
    "PROJECTION_EXECUTION_ANCHOR_BINDING_MODE_V3",
    "PROJECTION_HELPER_ROOT_BINDING_MODE_V3",
    "derive_goldilocks_projection_coefficients_v3",
    "goldilocks_projection_input_cells_v3",
    "goldilocks_projection_output_cells_v3",
    "goldilocks_projection_runtime_binding_v3",
    "goldilocks_projection_runtime_cells_v3",
    "goldilocks_projection_x_row_squares_v3",
    "prove_goldilocks_projection_composition_v3",
    "verify_goldilocks_projection_composition_v3",
]


def _fixed32(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    return value


def _pow2(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


def _oracle_record(oracle: EconomicOracleCommitmentV3) -> bytes:
    if not isinstance(oracle, EconomicOracleCommitmentV3):
        raise ProofV3Error("projection capture oracle is malformed")
    fields = []
    for value in (oracle.oracle_id, oracle.phase, oracle.operation):
        try:
            encoded = value.encode("ascii")
        except (AttributeError, UnicodeEncodeError) as exc:
            raise ProofV3Error(
                "projection capture oracle identity is malformed") from exc
        if not encoded or len(encoded) > 255:
            raise ProofV3Error(
                "projection capture oracle identity is malformed")
        fields.append(struct.pack("<B", len(encoded)) + encoded)
    return (
        b"".join(fields)
        + struct.pack(
            "<IIIQ",
            oracle.layer_index,
            oracle.row_count,
            oracle.col_count,
            oracle.scale_bits,
        )
        + _fixed32(oracle.root, "projection capture root")
    )


@dataclass(frozen=True, slots=True)
class GoldilocksProjectionAnchorClaimV3:
    """Validator-owned pre-nonce source for one projection input column."""

    commitment: ExecutionAnchorCommitmentV3
    anchor_rows: tuple[int, ...]
    source_column_offset: int
    encoding_id: str

    def __post_init__(self) -> None:
        rows = tuple(self.anchor_rows)
        if (
            not isinstance(self.commitment, ExecutionAnchorCommitmentV3)
            or not rows
            or len(set(rows)) != len(rows)
            or any(
                isinstance(row, bool)
                or not isinstance(row, int)
                or row < 0
                or row >= self.commitment.row_count
                for row in rows
            )
            or isinstance(self.source_column_offset, bool)
            or not isinstance(self.source_column_offset, int)
            or self.source_column_offset < 0
            or self.encoding_id not in {"fp16.v1", "bf16.v1"}
        ):
            raise ProofV3Error(
                "projection execution-anchor claim is malformed"
            )
        object.__setattr__(self, "anchor_rows", rows)


@dataclass(frozen=True, slots=True)
class GoldilocksProjectionAnchorWitnessV3:
    """Bounded replay material used only by the prover."""

    row_bytes_by_index: tuple[tuple[int, bytes], ...]
    row_tree: MerkleTree

    def __post_init__(self) -> None:
        rows = tuple(self.row_bytes_by_index)
        indices = tuple(index for index, _row in rows)
        if (
            not rows
            or len(set(indices)) != len(indices)
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or not isinstance(row, bytes)
                for index, row in rows
            )
            or not isinstance(self.row_tree, MerkleTree)
        ):
            raise ProofV3Error(
                "projection execution-anchor witness is malformed"
            )
        object.__setattr__(self, "row_bytes_by_index", rows)


@dataclass(frozen=True, slots=True)
class GoldilocksProjectionRuntimeClaimV3:
    """Signed runtime-output corridor attached to one projection."""

    y_oracle: EconomicOracleCommitmentV3
    y_anchor: GoldilocksProjectionAnchorClaimV3 | None
    output_columns: tuple[int, ...]
    weight_scale_bits: int
    weight_row_squares: tuple[tuple[int, int], ...]
    corridor_sigma_bits: int
    corridor_chi2_bits: int
    corridor_kind: str
    bias_values: tuple[tuple[int, int], ...] = ()
    bias_scale_bits: int = 0
    input_columns: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        columns = tuple(self.output_columns)
        input_columns = tuple(self.input_columns)
        weight_squares = tuple(self.weight_row_squares)
        biases = tuple(self.bias_values)
        try:
            encoded_kind = self.corridor_kind.encode("ascii")
            weight_scale = bits_to_scale_v3(self.weight_scale_bits)
            sigma = bits_to_scale_v3(self.corridor_sigma_bits)
            chi2 = bits_to_scale_v3(self.corridor_chi2_bits)
            bias_scale = (
                bits_to_scale_v3(self.bias_scale_bits)
                if self.bias_scale_bits
                else 0.0
            )
        except (AttributeError, UnicodeEncodeError, ProofV3Error) as exc:
            raise ProofV3Error(
                "projection runtime corridor claim is malformed"
            ) from exc
        if (
            not isinstance(self.y_oracle, EconomicOracleCommitmentV3)
            or (
                self.y_anchor is not None
                and not isinstance(
                    self.y_anchor,
                    GoldilocksProjectionAnchorClaimV3,
                )
            )
            or not columns
            or columns != tuple(sorted(set(columns)))
            or any(
                isinstance(column, bool)
                or not isinstance(column, int)
                or column < 0
                or column >= self.y_oracle.col_count
                for column in columns
            )
            or tuple(column for column, _value in weight_squares) != columns
            or input_columns != tuple(sorted(set(input_columns)))
            or any(
                isinstance(column, bool)
                or not isinstance(column, int)
                or column < 0
                for column in input_columns
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < 1 << 48
                for _column, value in weight_squares
            )
            or (biases and tuple(column for column, _value in biases) != columns)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not -128 <= value <= 127
                for _column, value in biases
            )
            or bool(biases) != bool(self.bias_scale_bits)
            or not encoded_kind
            or len(encoded_kind) > 95
            or weight_scale <= 0.0
            or sigma <= 0.0
            or chi2 <= 0.0
            or bias_scale < 0.0
        ):
            raise ProofV3Error(
                "projection runtime corridor claim is malformed"
            )
        object.__setattr__(self, "output_columns", columns)
        object.__setattr__(self, "input_columns", input_columns)
        object.__setattr__(self, "weight_row_squares", weight_squares)
        object.__setattr__(self, "bias_values", biases)


@dataclass(frozen=True, slots=True)
class GoldilocksProjectionClaimV3:
    operation: LeanProjectionCatalogOperationV3
    x_oracle: EconomicOracleCommitmentV3
    s_oracle: EconomicOracleCommitmentV3
    selected_rows: tuple[int, ...]
    x_anchor: GoldilocksProjectionAnchorClaimV3 | None = None
    runtime: GoldilocksProjectionRuntimeClaimV3 | None = None
    weight_scale_bits: int = 0
    consumer_input_cells: tuple[tuple[int, int], ...] = ()
    consumer_output_cells: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.operation, LeanProjectionCatalogOperationV3):
            raise ProofV3Error("projection composition operation is malformed")
        if (
            not isinstance(self.x_oracle, EconomicOracleCommitmentV3)
            or not isinstance(self.s_oracle, EconomicOracleCommitmentV3)
        ):
            raise ProofV3Error("projection composition oracle is malformed")
        rows = tuple(self.selected_rows)
        if (
            not rows
            or rows != tuple(sorted(set(rows)))
            or any(
                isinstance(row, bool)
                or not isinstance(row, int)
                or row < 0
                or row >= self.x_oracle.row_count
                for row in rows
            )
        ):
            raise ProofV3Error(
                "projection composition selected rows are malformed")
        if (
            self.s_oracle.row_count != self.x_oracle.row_count
            or self.x_oracle.col_count != self.operation.input_dim
            or self.s_oracle.col_count != self.operation.output_dim
            or not self.x_oracle.operation.endswith("_x")
            or not self.s_oracle.operation.endswith("_s")
            or self.x_oracle.layer_index != self.s_oracle.layer_index
        ):
            raise ProofV3Error(
                "projection composition oracle geometry is inconsistent")
        if (
            self.x_anchor is not None
            and (
                not isinstance(
                    self.x_anchor,
                    GoldilocksProjectionAnchorClaimV3,
                )
                or len(self.x_anchor.anchor_rows) != len(rows)
                or (
                    self.x_anchor.source_column_offset
                    + self.operation.input_dim
                    > self.x_anchor.commitment.row_width // 2
                )
            )
        ):
            raise ProofV3Error(
                "projection execution-anchor geometry is inconsistent"
            )
        consumer_inputs = tuple(self.consumer_input_cells)
        consumer_outputs = tuple(self.consumer_output_cells)
        if (
            isinstance(self.weight_scale_bits, bool)
            or not isinstance(self.weight_scale_bits, int)
            or not 0 <= self.weight_scale_bits < 1 << 64
        ):
            raise ProofV3Error(
                "projection weight scale is malformed"
            )
        if self.weight_scale_bits:
            try:
                weight_scale = bits_to_scale_v3(
                    self.weight_scale_bits
                )
            except ProofV3Error as exc:
                raise ProofV3Error(
                    "projection weight scale is malformed"
                ) from exc
            if weight_scale <= 0.0:
                raise ProofV3Error(
                    "projection weight scale is malformed"
                )
        for cells, width, name in (
            (
                consumer_inputs,
                self.operation.input_dim,
                "input",
            ),
            (
                consumer_outputs,
                self.operation.output_dim,
                "output",
            ),
        ):
            if (
                cells != tuple(sorted(set(cells)))
                or any(
                    not isinstance(cell, tuple)
                    or len(cell) != 2
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        for value in cell
                    )
                    or cell[0] not in rows
                    or cell[1] < 0
                    or cell[1] >= width
                    for cell in cells
                )
            ):
                raise ProofV3Error(
                    f"projection consumer-{name} inventory is malformed"
                )
        if self.runtime is not None:
            runtime = self.runtime
            if (
                not isinstance(runtime, GoldilocksProjectionRuntimeClaimV3)
                or runtime.y_oracle.row_count != self.x_oracle.row_count
                or runtime.y_oracle.col_count != self.operation.output_dim
                or runtime.y_oracle.layer_index
                != self.x_oracle.layer_index
                or any(
                    column >= self.operation.input_dim
                    for column in runtime.input_columns
                )
                or (
                    runtime.input_columns
                    and self.x_anchor is None
                )
                or (
                    runtime.y_anchor is not None
                    and (
                        len(runtime.y_anchor.anchor_rows) != len(rows)
                        or (
                            runtime.y_anchor.source_column_offset
                            + self.operation.output_dim
                            > runtime.y_anchor.commitment.row_width // 2
                        )
                    )
                )
                or (
                    self.weight_scale_bits
                    and self.weight_scale_bits
                    != runtime.weight_scale_bits
                )
            ):
                raise ProofV3Error(
                    "projection runtime corridor geometry is inconsistent"
                )
        object.__setattr__(self, "selected_rows", rows)
        object.__setattr__(
            self,
            "consumer_input_cells",
            consumer_inputs,
        )
        object.__setattr__(
            self,
            "consumer_output_cells",
            consumer_outputs,
        )


@dataclass(frozen=True, slots=True)
class GoldilocksProjectionWitnessV3:
    claim: GoldilocksProjectionClaimV3
    committed_x: EconomicCommittedOracleV3
    committed_s: EconomicCommittedOracleV3
    x_anchor_witness: GoldilocksProjectionAnchorWitnessV3 | None = None
    committed_y: EconomicCommittedOracleV3 | None = None
    y_anchor_witness: GoldilocksProjectionAnchorWitnessV3 | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.claim, GoldilocksProjectionClaimV3)
            or not isinstance(self.committed_x, EconomicCommittedOracleV3)
            or not isinstance(self.committed_s, EconomicCommittedOracleV3)
            or self.committed_x.commitment != self.claim.x_oracle
            or self.committed_s.commitment != self.claim.s_oracle
            or (
                (self.claim.x_anchor is None)
                != (self.x_anchor_witness is None)
            )
            or (
                self.claim.x_anchor is not None
                and self.x_anchor_witness is not None
                and (
                    self.x_anchor_witness.row_tree.root
                    != self.claim.x_anchor.commitment.root
                    or self.x_anchor_witness.row_tree.num_leaves
                    != self.claim.x_anchor.commitment.row_count
                    or tuple(
                        index
                        for index, _row
                        in self.x_anchor_witness.row_bytes_by_index
                    )
                    != self.claim.x_anchor.anchor_rows
                    or any(
                        len(row)
                        != self.claim.x_anchor.commitment.row_width
                        for _index, row
                        in self.x_anchor_witness.row_bytes_by_index
                    )
                )
            )
            or (
                (self.claim.runtime is None)
                != (
                    self.committed_y is None
                    and self.y_anchor_witness is None
                )
            )
            or (
                self.claim.runtime is not None
                and (
                    not isinstance(
                        self.committed_y,
                        EconomicCommittedOracleV3,
                    )
                    or self.committed_y.commitment
                    != self.claim.runtime.y_oracle
                    or (
                        self.claim.runtime.y_anchor is None
                        and self.y_anchor_witness is not None
                    )
                    or (
                        self.claim.runtime.y_anchor is not None
                        and (
                            not isinstance(
                                self.y_anchor_witness,
                                GoldilocksProjectionAnchorWitnessV3,
                            )
                            or self.y_anchor_witness.row_tree.root
                            != self.claim.runtime.y_anchor.commitment.root
                            or self.y_anchor_witness.row_tree.num_leaves
                            != self.claim.runtime.y_anchor.commitment.row_count
                            or tuple(
                                index
                                for index, _row
                                in self.y_anchor_witness.row_bytes_by_index
                            )
                            != self.claim.runtime.y_anchor.anchor_rows
                            or any(
                                len(row)
                                != self.claim.runtime.y_anchor.commitment.row_width
                                for _index, row
                                in self.y_anchor_witness.row_bytes_by_index
                            )
                        )
                    )
                )
            )
        ):
            raise ProofV3Error(
                "projection composition witness is inconsistent")


@dataclass(frozen=True, slots=True)
class GoldilocksProjectionGroupCommitmentV3:
    group_tag: str
    commitment: bytes

    def __post_init__(self) -> None:
        try:
            encoded = self.group_tag.encode("ascii")
        except (AttributeError, UnicodeEncodeError) as exc:
            raise ProofV3Error(
                "projection group tag is malformed") from exc
        if not encoded or len(encoded) > 255:
            raise ProofV3Error("projection group tag is malformed")
        _fixed32(self.commitment, "projection group commitment")


@dataclass(frozen=True, slots=True)
class GoldilocksProjectionCaptureProofV3:
    binding_mode: str
    x_binding: (
        GoldilocksCapturePcsBindingProofV3
        | GoldilocksExecutionAnchorPcsBindingProofV3
    )
    s_binding: GoldilocksCapturePcsBindingProofV3 | None
    y_binding: GoldilocksExecutionAnchorPcsBindingProofV3 | None
    x_row_squares: tuple[int, ...]
    input_cells: tuple[int, ...]
    runtime_cells: tuple[tuple[int, int], ...]
    output_cells: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        helper_mode = (
            self.binding_mode == PROJECTION_HELPER_ROOT_BINDING_MODE_V3
            and isinstance(
                self.x_binding,
                GoldilocksCapturePcsBindingProofV3,
            )
            and isinstance(
                self.s_binding,
                GoldilocksCapturePcsBindingProofV3,
            )
        )
        anchor_mode = (
            self.binding_mode
            == PROJECTION_EXECUTION_ANCHOR_BINDING_MODE_V3
            and isinstance(
                self.x_binding,
                GoldilocksExecutionAnchorPcsBindingProofV3,
            )
            and self.s_binding is None
        )
        squares = tuple(self.x_row_squares)
        input_cells = tuple(self.input_cells)
        runtime_cells = tuple(self.runtime_cells)
        output_cells = tuple(self.output_cells)
        if (
            (not helper_mode and not anchor_mode)
            or not squares
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= _U31_MAX
                for value in squares
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not -128 <= value <= 127
                for value in input_cells
            )
            or any(
                not isinstance(pair, tuple)
                or len(pair) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in pair
                )
                or not -(1 << 47) <= pair[0] < 1 << 47
                or not -128 <= pair[1] <= 127
                for pair in runtime_cells
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < -(1 << 63)
                or value >= 1 << 63
                for value in output_cells
            )
            or (
                self.y_binding is not None
                and not isinstance(
                    self.y_binding,
                    GoldilocksExecutionAnchorPcsBindingProofV3,
                )
            )
        ):
            raise ProofV3Error(
                "projection capture binding proof is malformed")
        object.__setattr__(self, "x_row_squares", squares)
        object.__setattr__(self, "input_cells", input_cells)
        object.__setattr__(self, "runtime_cells", runtime_cells)
        object.__setattr__(self, "output_cells", output_cells)


@dataclass(frozen=True, slots=True)
class GoldilocksProjectionRelationProofV3:
    product: GoldilocksSuccinctProductProofV3
    surrogate_fold: SuccinctEqFoldProofV3
    x_square_product: GoldilocksSuccinctProductProofV3

    def __post_init__(self) -> None:
        if (
            not isinstance(self.product, GoldilocksSuccinctProductProofV3)
            or not isinstance(self.surrogate_fold, SuccinctEqFoldProofV3)
            or not isinstance(
                self.x_square_product,
                GoldilocksSuccinctProductProofV3,
            )
        ):
            raise ProofV3Error("projection relation proof is malformed")


@dataclass(frozen=True, slots=True)
class GoldilocksProjectionCompositionProofV3:
    phase1_groups: tuple[GoldilocksProjectionGroupCommitmentV3, ...]
    phase2_groups: tuple[GoldilocksProjectionGroupCommitmentV3, ...]
    captures: tuple[GoldilocksProjectionCaptureProofV3, ...]
    relations: tuple[GoldilocksProjectionRelationProofV3, ...]
    static_bridge: GoldilocksStaticCatalogBridgeProofV3
    batch_opening: object

    def __post_init__(self) -> None:
        phase1 = tuple(self.phase1_groups)
        phase2 = tuple(self.phase2_groups)
        captures = tuple(self.captures)
        relations = tuple(self.relations)
        for groups in (phase1, phase2):
            tags = tuple(group.group_tag for group in groups)
            if (
                not groups
                or not all(
                    isinstance(
                        group,
                        GoldilocksProjectionGroupCommitmentV3,
                    )
                    for group in groups
                )
                or tags != tuple(sorted(set(tags)))
            ):
                raise ProofV3Error(
                    "projection group commitment inventory is malformed")
        if (
            not captures
            or not all(
                isinstance(item, GoldilocksProjectionCaptureProofV3)
                for item in captures
            )
            or len(relations) != len(captures)
            or not all(
                isinstance(item, GoldilocksProjectionRelationProofV3)
                for item in relations
            )
            or not isinstance(
                self.static_bridge,
                GoldilocksStaticCatalogBridgeProofV3,
            )
        ):
            raise ProofV3Error(
                "projection composition proof inventory is malformed")
        object.__setattr__(self, "phase1_groups", phase1)
        object.__setattr__(self, "phase2_groups", phase2)
        object.__setattr__(self, "captures", captures)
        object.__setattr__(self, "relations", relations)


def _claims_digest(
    claims: tuple[GoldilocksProjectionClaimV3, ...],
) -> bytes:
    if (
        not claims
        or len(claims) > MAX_PROJECTION_COMPOSITION_OPERATIONS_V3
        or len({claim.operation.operation_key for claim in claims})
        != len(claims)
    ):
        raise ProofV3Error("projection claim inventory is malformed")
    chi2_bits = {
        claim.runtime.corridor_chi2_bits
        for claim in claims
        if claim.runtime is not None
    }
    if len(chi2_bits) > 1:
        raise ProofV3Error(
            "projection runtime corridors disagree on the aggregate cap"
        )
    material = bytearray(
        _TRANSCRIPT_DOMAIN
        + b"/claims/"
        + GOLDILOCKS_PROJECTION_COMPOSITION_ABI_V3.encode("ascii")
        + struct.pack("<I", len(claims))
    )
    for index, claim in enumerate(claims):
        material.extend(struct.pack("<I", index))
        material.extend(claim.operation.operation_digest)
        material.extend(claim.operation.operation_root)
        material.extend(claim.operation.registered_catalog_id)
        material.extend(
            struct.pack(
                "<III",
                claim.operation.input_dim,
                claim.operation.padded_input_dim,
                claim.operation.output_dim,
            )
        )
        material.extend(_oracle_record(claim.x_oracle))
        material.extend(_oracle_record(claim.s_oracle))
        mode = (
            PROJECTION_HELPER_ROOT_BINDING_MODE_V3
            if claim.x_anchor is None
            else PROJECTION_EXECUTION_ANCHOR_BINDING_MODE_V3
        )
        material.extend(struct.pack("<H", len(mode)))
        material.extend(mode.encode("ascii"))
        if claim.x_anchor is not None:
            anchor = claim.x_anchor
            record = anchor.commitment.canonical_bytes()
            encoding = anchor.encoding_id.encode("ascii")
            material.extend(struct.pack("<I", len(record)))
            material.extend(record)
            material.extend(
                struct.pack(
                    "<IIB",
                    anchor.source_column_offset,
                    len(anchor.anchor_rows),
                    len(encoding),
                )
            )
            material.extend(encoding)
            material.extend(
                b"".join(
                    struct.pack("<I", row)
                    for row in anchor.anchor_rows
                )
            )
        material.extend(struct.pack("<B", int(claim.runtime is not None)))
        if claim.runtime is not None:
            runtime = claim.runtime
            material.extend(_oracle_record(runtime.y_oracle))
            kind = runtime.corridor_kind.encode("ascii")
            material.extend(
                struct.pack("<B", int(runtime.y_anchor is not None))
            )
            if runtime.y_anchor is not None:
                record = runtime.y_anchor.commitment.canonical_bytes()
                encoding = runtime.y_anchor.encoding_id.encode("ascii")
                material.extend(struct.pack("<I", len(record)))
                material.extend(record)
                material.extend(
                    struct.pack(
                        "<IIB",
                        runtime.y_anchor.source_column_offset,
                        len(runtime.y_anchor.anchor_rows),
                        len(encoding),
                    )
                )
                material.extend(encoding)
                material.extend(
                    b"".join(
                        struct.pack("<I", row)
                        for row in runtime.y_anchor.anchor_rows
                    )
                )
            material.extend(
                struct.pack(
                    "<QQQB",
                    runtime.weight_scale_bits,
                    runtime.corridor_sigma_bits,
                    runtime.corridor_chi2_bits,
                    len(kind),
                )
            )
            material.extend(kind)
            material.extend(
                struct.pack("<I", len(runtime.output_columns))
            )
            material.extend(
                b"".join(
                    struct.pack("<IQ", column, square)
                    for (column, square)
                    in runtime.weight_row_squares
                )
            )
            material.extend(
                struct.pack("<I", len(runtime.input_columns))
            )
            material.extend(
                b"".join(
                    struct.pack("<I", column)
                    for column in runtime.input_columns
                )
            )
            material.extend(
                struct.pack(
                    "<QI",
                    runtime.bias_scale_bits,
                    len(runtime.bias_values),
                )
            )
            material.extend(
                b"".join(
                    struct.pack("<Ib", column, value)
                    for column, value in runtime.bias_values
                )
            )
        material.extend(struct.pack("<Q", claim.weight_scale_bits))
        material.extend(struct.pack("<I", len(claim.selected_rows)))
        material.extend(
            b"".join(
                struct.pack("<I", row) for row in claim.selected_rows
            )
        )
        material.extend(
            struct.pack("<I", len(claim.consumer_input_cells))
        )
        material.extend(
            b"".join(
                struct.pack("<II", row, column)
                for row, column in claim.consumer_input_cells
            )
        )
        material.extend(
            struct.pack("<I", len(claim.consumer_output_cells))
        )
        material.extend(
            b"".join(
                struct.pack("<II", row, column)
                for row, column in claim.consumer_output_cells
            )
        )
    return hashlib.sha256(bytes(material)).digest()


def _phase1_tile(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    claims_digest: bytes,
) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/phase1/"
        + _fixed32(
            validator_binding_digest,
            "projection validator binding",
        )
        + _fixed32(validator_nonce, "projection validator nonce")
        + claims_digest
    ).digest()


def _tags(index: int) -> tuple[str, str, str]:
    return (
        f"projection/{index}/x",
        f"projection/{index}/s",
        f"projection/{index}/z",
    )


def _runtime_y_tag(index: int) -> str:
    return f"projection/{index}/runtime_y"


def _consumer_input_coordinates(
    claim: GoldilocksProjectionClaimV3,
) -> tuple[tuple[int, int], ...]:
    cells = set(claim.consumer_input_cells)
    if claim.runtime is not None:
        cells.update(
            (row, column)
            for row in claim.selected_rows
            for column in claim.runtime.input_columns
        )
    return tuple(sorted(cells))


def _phase1_sizes(
    claims: tuple[GoldilocksProjectionClaimV3, ...],
) -> tuple[tuple[str, int], ...]:
    result = []
    for index, claim in enumerate(claims):
        row_pad = _pow2(len(claim.selected_rows))
        x_tag, s_tag, _z_tag = _tags(index)
        result.extend(
            (
                (x_tag, row_pad * _pow2(claim.operation.input_dim)),
                (s_tag, row_pad * _pow2(claim.operation.output_dim)),
            )
        )
        if claim.runtime is not None:
            result.append(
                (
                    _runtime_y_tag(index),
                    row_pad * _pow2(claim.operation.output_dim),
                )
            )
    return tuple(result)


def _phase2_sizes(
    claims: tuple[GoldilocksProjectionClaimV3, ...],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (
            _tags(index)[2],
            LEAN_PROJECTION_FOLD_COUNT_V3
            * claim.operation.padded_input_dim,
        )
        for index, claim in enumerate(claims)
    )


def _root_records(groups) -> tuple[GoldilocksProjectionGroupCommitmentV3, ...]:
    return tuple(
        GoldilocksProjectionGroupCommitmentV3(
            group_tag=group.tag,
            commitment=group.tree.commitment,
        )
        for group in sorted(groups, key=lambda item: item.tag)
    )


def _root_digest(
    roots: tuple[GoldilocksProjectionGroupCommitmentV3, ...],
) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/roots/"
        + struct.pack("<I", len(roots))
        + b"".join(
            struct.pack("<B", len(root.group_tag.encode("ascii")))
            + root.group_tag.encode("ascii")
            + root.commitment
            for root in roots
        )
    ).digest()


def goldilocks_projection_runtime_binding_v3(
    proof: object,
    claims,
) -> bytes:
    """Bind a downstream transition proof to one projection composition."""

    claims_t = tuple(claims)
    if (
        not isinstance(proof, GoldilocksProjectionCompositionProofV3)
        or len(claims_t) != len(proof.captures)
    ):
        raise ProofV3Error(
            "projection runtime consumer inventory is inconsistent"
        )
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/runtime-consumer/"
        + _claims_digest(claims_t)
        + _root_digest(proof.phase1_groups)
    ).digest()


def goldilocks_projection_input_cells_v3(
    proof: object,
    claims,
    *,
    claim_index: int,
) -> tuple[tuple[int, int, int], ...]:
    """Return canonical ``(row, column, X)`` cells for a verified claim."""

    claims_t = tuple(claims)
    if (
        not isinstance(proof, GoldilocksProjectionCompositionProofV3)
        or isinstance(claim_index, bool)
        or not isinstance(claim_index, int)
        or claim_index < 0
        or claim_index >= len(claims_t)
        or len(claims_t) != len(proof.captures)
    ):
        raise ProofV3Error(
            "projection input cell request is inconsistent"
        )
    claim = claims_t[claim_index]
    capture = proof.captures[claim_index]
    coordinates = _consumer_input_coordinates(claim)
    if len(capture.input_cells) != len(coordinates):
        raise ProofV3Error(
            "projection input cell inventory is incomplete"
        )
    return tuple(
        (row, column, value)
        for (row, column), value in zip(
            coordinates,
            capture.input_cells,
            strict=True,
        )
    )


def goldilocks_projection_output_cells_v3(
    proof: object,
    claims,
    *,
    claim_index: int,
) -> tuple[tuple[int, int, int], ...]:
    """Return canonical ``(row, column, S)`` consumer cells.

    These cells are authenticated by a nonce-derived sparse term folded into
    the projection's existing surrogate sumcheck.  They therefore reuse the
    projection PCS commitment and the selected trace's shared terminal opening
    instead of adding one PCS opening per carried cell.
    """

    claims_t = tuple(claims)
    if (
        not isinstance(proof, GoldilocksProjectionCompositionProofV3)
        or isinstance(claim_index, bool)
        or not isinstance(claim_index, int)
        or claim_index < 0
        or claim_index >= len(claims_t)
        or len(claims_t) != len(proof.captures)
    ):
        raise ProofV3Error(
            "projection output cell request is inconsistent"
        )
    claim = claims_t[claim_index]
    capture = proof.captures[claim_index]
    expected = len(claim.consumer_output_cells)
    if len(capture.output_cells) != expected:
        raise ProofV3Error(
            "projection output cell inventory is incomplete"
        )
    result = []
    slot = 0
    for row, column in claim.consumer_output_cells:
        result.append((row, column, capture.output_cells[slot]))
        slot += 1
    return tuple(result)


def goldilocks_projection_x_row_squares_v3(
    proof: object,
    claims,
    *,
    claim_index: int,
) -> tuple[tuple[int, int], ...]:
    """Return canonical ``(row, sum(X_i^2))`` values for a verified claim."""

    claims_t = tuple(claims)
    if (
        not isinstance(proof, GoldilocksProjectionCompositionProofV3)
        or isinstance(claim_index, bool)
        or not isinstance(claim_index, int)
        or claim_index < 0
        or claim_index >= len(claims_t)
        or len(claims_t) != len(proof.captures)
    ):
        raise ProofV3Error(
            "projection row-square request is inconsistent"
        )
    claim = claims_t[claim_index]
    values = tuple(proof.captures[claim_index].x_row_squares)
    if len(values) != len(claim.selected_rows):
        raise ProofV3Error(
            "projection row-square inventory is incomplete"
        )
    return tuple(zip(claim.selected_rows, values, strict=True))


def goldilocks_projection_runtime_cells_v3(
    proof: object,
    claims,
    *,
    claim_index: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return canonical ``(row, column, S, Y)`` cells for a verified claim."""

    claims_t = tuple(claims)
    if (
        not isinstance(proof, GoldilocksProjectionCompositionProofV3)
        or isinstance(claim_index, bool)
        or not isinstance(claim_index, int)
        or claim_index < 0
        or claim_index >= len(claims_t)
        or len(claims_t) != len(proof.captures)
    ):
        raise ProofV3Error(
            "projection runtime cell request is inconsistent"
        )
    claim = claims_t[claim_index]
    capture = proof.captures[claim_index]
    runtime = claim.runtime
    if (
        runtime is None
        or (
            (runtime.y_anchor is None)
            != (capture.y_binding is None)
        )
        or len(capture.runtime_cells)
        != len(claim.selected_rows) * len(runtime.output_columns)
    ):
        raise ProofV3Error(
            "projection runtime cell inventory is incomplete"
        )
    result = []
    slot = 0
    for row in claim.selected_rows:
        for column in runtime.output_columns:
            s_value, y_value = capture.runtime_cells[slot]
            slot += 1
            result.append((row, column, s_value, y_value))
    return tuple(result)


def _u31_rows(seed: bytes, count: int) -> tuple[int, ...]:
    stream = hashlib.shake_256(seed).digest(count * 4)
    return tuple(
        int.from_bytes(stream[offset : offset + 4], "little") & _U31_MAX
        for offset in range(0, len(stream), 4)
    )


def derive_goldilocks_projection_coefficients_v3(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    claims,
    phase1_groups,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Derive four output folds only after the X/S PCS roots are fixed."""

    claims_t = tuple(claims)
    claims_digest = _claims_digest(claims_t)
    tile = _phase1_tile(
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
        claims_digest=claims_digest,
    )
    roots = tuple(phase1_groups)
    seed = hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/output-folds/"
        + tile
        + _root_digest(roots)
    ).digest()
    result = []
    for index, claim in enumerate(claims_t):
        words = _u31_rows(
            seed + struct.pack("<I", index),
            LEAN_PROJECTION_FOLD_COUNT_V3 * claim.operation.output_dim,
        )
        result.append(
            tuple(
                words[
                    fold * claim.operation.output_dim:
                    (fold + 1) * claim.operation.output_dim
                ]
                for fold in range(LEAN_PROJECTION_FOLD_COUNT_V3)
            )
        )
    return tuple(result)


def _phase2_tile(
    *,
    phase1_tile: bytes,
    phase1_roots,
    coefficients,
) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/phase2/"
        + phase1_tile
        + _root_digest(tuple(phase1_roots))
        + b"".join(
            struct.pack("<I", value)
            for operation in coefficients
            for row in operation
            for value in row
        )
    ).digest()


def _relation_seed(
    *,
    phase2_tile: bytes,
    phase2_roots,
) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/relations/"
        + phase2_tile
        + _root_digest(tuple(phase2_roots))
    ).digest()


def _field_vector(seed: bytes, label: bytes, count: int) -> tuple[int, ...]:
    result = []
    counter = 0
    while len(result) < count:
        block = hashlib.sha256(
            seed
            + struct.pack("<H", len(label))
            + label
            + struct.pack("<Q", counter)
        ).digest()
        counter += 1
        for offset in range(0, 32, 8):
            value = int.from_bytes(block[offset : offset + 8], "little")
            if value < GOLDILOCKS_MODULUS:
                result.append(value)
            if len(result) == count:
                break
    return tuple(result)


def _selected_values(
    committed: EconomicCommittedOracleV3,
    rows: tuple[int, ...],
    *,
    fused,
):
    row_pad = _pow2(len(rows))
    col_pad = _pow2(committed.commitment.col_count)
    if fused is not None and committed.int_rows_cpu is not None:
        import torch

        source = committed.int_rows_cpu
        selected = source[list(rows)].to(dtype=torch.int64, device="cuda")
        padded = torch.zeros(
            (row_pad, col_pad),
            dtype=torch.int64,
            device="cuda",
        )
        padded[: len(rows), : committed.commitment.col_count] = selected
        # Canonical Goldilocks field bytes in signed-int64 storage:
        # p + negative == 2^64 + (negative - (2^32 - 1)).
        return torch.where(
            padded < 0,
            padded - ((1 << 32) - 1),
            padded,
        ).reshape(-1)
    result = []
    for row in rows:
        result.extend(
            signed_to_field_v3(
                committed.signed_value(row, column)
            )
            for column in range(committed.commitment.col_count)
        )
        result.extend(
            (0,) * (col_pad - committed.commitment.col_count)
        )
    result.extend((0,) * (row_pad * col_pad - len(result)))
    return tuple(result)


def _folded_values(rows, *, fused):
    values = tuple(value for row in rows for value in row)
    if fused is None:
        return tuple(signed_to_field_v3(value) for value in values)
    import torch

    tensor = torch.tensor(values, dtype=torch.int64, device="cuda")
    return torch.where(
        tensor < 0,
        tensor - ((1 << 32) - 1),
        tensor,
    )


def _build_folded_weights(
    *,
    claims: tuple[GoldilocksProjectionClaimV3, ...],
    coefficients,
    weight_rows_i8,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    fused,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Build exact W*c rows after the output-fold challenge is known."""

    weights = tuple(weight_rows_i8)
    if len(weights) != len(claims):
        raise ProofV3Error(
            "projection weight witness inventory is inconsistent")
    result = []
    for index, (claim, weight) in enumerate(
        zip(claims, weights, strict=True)
    ):
        operation = claim.operation
        if fused is not None and hasattr(weight, "shape"):
            from verallm.proof_v3.lean_projection_native import (
                _build_lean_projection_fold_group_cuda_v3,
            )

            built = _build_lean_projection_fold_group_cuda_v3(
                statements=(
                    operation.statement(
                        validator_binding_digest=validator_binding_digest
                    ),
                ),
                validator_nonce=validator_nonce,
                input_rows_i8=((),),
                surrogate_outputs_i64=((),),
                weight_rows_i8=weight,
                coefficient_rows=(coefficients[index],),
            )[0]
            result.append(built.folded_weights)
            continue
        try:
            rows = tuple(
                tuple(int(value) for value in row)
                for row in weight
            )
        except (TypeError, ValueError) as exc:
            raise ProofV3Error(
                "projection weight witness is malformed") from exc
        if (
            len(rows) != operation.output_dim
            or any(
                len(row) not in (
                    operation.input_dim,
                    operation.padded_input_dim,
                )
                for row in rows
            )
            or any(
                value < -128 or value > 127
                for row in rows
                for value in row
            )
        ):
            raise ProofV3Error(
                "projection weight witness is not canonical int8 out_in")
        result.append(
            tuple(
                tuple(
                    sum(
                        coefficients[index][fold][output]
                        * rows[output][inner]
                        for output in range(operation.output_dim)
                    )
                    if inner < operation.input_dim else 0
                    for inner in range(operation.padded_input_dim)
                )
                for fold in range(LEAN_PROJECTION_FOLD_COUNT_V3)
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _PublicColumnV3:
    tag: str
    pcs_statement: object
    commitment: bytes
    group_tag: str
    block_point: tuple[int, ...]

    @property
    def tree(self):
        return self


def _public_columns(
    *,
    tile_digest: bytes,
    group_tag_prefix: str,
    sizes: tuple[tuple[str, int], ...],
    root_records,
) -> tuple[
    tuple[_PublicColumnV3, ...],
    dict[str, _PublicColumnV3],
    tuple[VariableColumnGroupPlanV3, ...],
]:
    plans = plan_succinct_variable_column_groups_v3(
        tile_digest=tile_digest,
        group_tag_prefix=group_tag_prefix,
        ordered_sizes=sizes,
        max_group_cells=MAX_PROJECTION_GROUP_CELLS_V3,
    )
    records = tuple(root_records)
    if tuple(record.group_tag for record in records) != tuple(
        sorted(plan.group_tag for plan in plans)
    ):
        raise ProofV3VerificationError(
            "projection packed-group inventory is not exact")
    by_tag = {record.group_tag: record for record in records}
    groups = []
    members = {}
    for plan in plans:
        record = by_tag[plan.group_tag]
        group = _PublicColumnV3(
            tag=plan.group_tag,
            pcs_statement=column_pcs_statement_v3(
                plan.layout_digest,
                plan.group_tag,
                plan.cell_count.bit_length() - 1,
            ),
            commitment=record.commitment,
            group_tag=plan.group_tag,
            block_point=(),
        )
        groups.append(group)
        for member in plan.members:
            members[member.tag] = _PublicColumnV3(
                tag=member.tag,
                pcs_statement=column_pcs_statement_v3(
                    tile_digest,
                    member.tag,
                    member.cell_count.bit_length() - 1,
                ),
                commitment=record.commitment,
                group_tag=plan.group_tag,
                block_point=member.block_point,
            )
    return tuple(groups), members, plans


def _factor_device(components):
    from verallm.proof_v3.native_goldilocks_backend import (
        gl_mul_t,
        to_field_tensor,
    )

    result = to_field_tensor((1,), "cuda")
    for component in components:
        values = to_field_tensor(component, "cuda")
        result = gl_mul_t(
            result.repeat_interleave(len(component)),
            values.repeat(result.numel()),
        )
    return result


def _add_sparse_factor_device(
    factor,
    indices: tuple[int, ...],
    coefficients: tuple[int, ...],
):
    if not indices:
        return factor
    import torch

    from verallm.proof_v3.native_goldilocks_backend import (
        gl_add_t,
        to_field_tensor,
    )

    index_tensor = torch.tensor(
        indices,
        dtype=torch.int64,
        device=factor.device,
    )
    current = factor.index_select(0, index_tensor)
    updated = gl_add_t(
        current,
        to_field_tensor(coefficients, factor.device),
    )
    factor.index_copy_(0, index_tensor, updated)
    return factor


def _relation_material(seed, index, claim):
    row_pad = _pow2(len(claim.selected_rows))
    alpha = _field_vector(
        seed + struct.pack("<I", index),
        b"rows",
        row_pad,
    )
    beta = _field_vector(
        seed + struct.pack("<I", index),
        b"folds",
        LEAN_PROJECTION_FOLD_COUNT_V3,
    )
    return alpha, beta


def _relation_statement(seed, index, row_pad, input_pad):
    return GoldilocksSuccinctProductStatementV3(
        validator_binding_digest=hashlib.sha256(
            _TRANSCRIPT_DOMAIN
            + b"/product/"
            + seed
            + struct.pack("<I", index)
        ).digest(),
        variable_count=int(
            math.log2(
                LEAN_PROJECTION_FOLD_COUNT_V3
                * row_pad
                * input_pad
            )
        ),
        factor_component_sizes=(
            LEAN_PROJECTION_FOLD_COUNT_V3,
            row_pad,
            input_pad,
        ),
    )


def _selected_x_row_squares(
    witness: GoldilocksProjectionWitnessV3,
) -> tuple[int, ...]:
    committed = witness.committed_x
    result = []
    for row in witness.claim.selected_rows:
        if committed.int_rows_cpu is not None:
            source = committed.int_rows_cpu[row]
            if hasattr(source, "tolist"):
                source = source.tolist()
            values = tuple(int(value) for value in source)
        else:
            values = tuple(
                field_to_signed_v3(
                    committed.raw_values[
                        row * committed.col_pad + column
                    ]
                )
                for column in range(witness.claim.operation.input_dim)
            )
        if (
            len(values) != witness.claim.operation.input_dim
            or any(value < -128 or value > 127 for value in values)
        ):
            raise ProofV3Error(
                "projection input row escapes its signed-int8 geometry"
            )
        square = sum(value * value for value in values)
        if square > _U31_MAX:
            raise ProofV3Error(
                "projection input row square exceeds the signed bound"
            )
        result.append(square)
    return tuple(result)


def _boolean_point(index: int, variable_count: int) -> tuple[int, ...]:
    return tuple((index >> bit) & 1 for bit in range(variable_count))


def _projection_runtime_metrics(
    *,
    runtime: GoldilocksProjectionRuntimeClaimV3,
    s_value: int,
    y_value: int,
    x_square: int,
    output_column: int,
    x_scale_bits: int,
    coordinate: str = "",
) -> tuple[float, float]:
    x_scale = bits_to_scale_v3(x_scale_bits)
    w_scale = bits_to_scale_v3(runtime.weight_scale_bits)
    y_scale = bits_to_scale_v3(runtime.y_oracle.scale_bits)
    sigma_cap = bits_to_scale_v3(runtime.corridor_sigma_bits)
    weight_squares = dict(runtime.weight_row_squares)
    bias_values = dict(runtime.bias_values)
    weight_square = weight_squares[output_column]
    bias_scale = (
        bits_to_scale_v3(runtime.bias_scale_bits)
        if runtime.bias_scale_bits
        else 0.0
    )
    bias_value = bias_values.get(output_column, 0) * bias_scale
    lhs = s_value * x_scale * w_scale + bias_value
    rhs = y_value * y_scale
    variance = (
        (x_scale * x_scale / 12.0)
        * (w_scale * w_scale)
        * weight_square
        + (w_scale * w_scale / 12.0)
        * (x_scale * x_scale)
        * x_square
    )
    sigma = math.sqrt(variance)
    relative = (
        CORRIDOR_REL_COEFF_NUM_V3
        / CORRIDOR_REL_COEFF_DEN_V3
        * x_scale
        * w_scale
        * math.sqrt(x_square * weight_square)
    )
    extra = 0.5 * y_scale + 0.5 * bias_scale + relative
    delta = abs(lhs - rhs)
    limit = sigma_cap * sigma + extra
    if delta > limit:
        detail = (
            f" ({coordinate}, delta={delta:.9g}, limit={limit:.9g})"
            if coordinate
            else ""
        )
        raise ProofV3VerificationError(
            "projection runtime output is outside its signed corridor"
            + detail
        )
    return delta, sigma + extra


def _verify_projection_runtime_aggregate(
    metrics: list[tuple[str, float, float]],
    claims: tuple[GoldilocksProjectionClaimV3, ...],
) -> None:
    if not metrics:
        return
    caps = {
        claim.runtime.corridor_chi2_bits
        for claim in claims
        if claim.runtime is not None
    }
    if len(caps) != 1:
        raise ProofV3VerificationError(
            "projection runtime aggregate cap is inconsistent"
        )
    cap = bits_to_scale_v3(next(iter(caps)))
    by_kind: dict[str, list[float]] = {}
    for kind, delta, spread in metrics:
        if spread <= 0.0:
            raise ProofV3VerificationError(
                "projection runtime corridor has no positive spread"
            )
        by_kind.setdefault(kind, []).append((delta / spread) ** 2)
    worst = max(
        sum(values) / len(values) for values in by_kind.values()
    )
    if worst > cap:
        raise ProofV3VerificationError(
            "projection runtime aggregate corridor exceeds its signed cap"
        )


def _square_alpha(
    seed: bytes,
    index: int,
    row_pad: int,
) -> tuple[int, ...]:
    return _field_vector(
        seed + struct.pack("<I", index),
        b"x-row-squares",
        row_pad,
    )


def _square_statement(
    seed: bytes,
    index: int,
    row_pad: int,
    input_pad: int,
    row_squares: tuple[int, ...],
) -> GoldilocksSuccinctProductStatementV3:
    return GoldilocksSuccinctProductStatementV3(
        validator_binding_digest=hashlib.sha256(
            _TRANSCRIPT_DOMAIN
            + b"/x-row-squares/"
            + seed
            + struct.pack("<II", index, len(row_squares))
            + b"".join(
                struct.pack("<Q", value) for value in row_squares
            )
        ).digest(),
        variable_count=int(math.log2(row_pad * input_pad)),
        factor_component_sizes=(row_pad, input_pad),
    )


def _surrogate_binding(seed, index) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/surrogate-fold/"
        + seed
        + struct.pack("<I", index)
    ).digest()


def _consumer_material(
    seed: bytes,
    index: int,
    claim: GoldilocksProjectionClaimV3,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return product-cube indices and fresh coefficients for carried S cells."""

    row_pad = _pow2(len(claim.selected_rows))
    output_pad = _pow2(claim.operation.output_dim)
    row_slots = {
        row: row_slot
        for row_slot, row in enumerate(claim.selected_rows)
    }
    indices = tuple(
        output * row_pad + row_slots[row]
        for row, output in claim.consumer_output_cells
    )
    coefficients = _field_vector(
        seed + struct.pack("<I", index),
        b"consumer-output-cells",
        len(indices),
    )
    return indices, coefficients


def _consumer_claimed_sum(
    values: tuple[int, ...],
    coefficients: tuple[int, ...],
) -> int:
    if len(values) != len(coefficients):
        raise ProofV3Error(
            "projection consumer-output inventory is inconsistent"
        )
    return sum(
        (value % GOLDILOCKS_MODULUS) * coefficient
        for value, coefficient in zip(values, coefficients, strict=True)
    ) % GOLDILOCKS_MODULUS


def _boolean_eq_at_msb_point(
    point: tuple[int, ...],
    index: int,
) -> int:
    result = 1
    width = len(point)
    for axis, challenge in enumerate(point):
        bit = (index >> (width - axis - 1)) & 1
        factor = challenge if bit else 1 - challenge
        result = (
            result * (factor % GOLDILOCKS_MODULUS)
            % GOLDILOCKS_MODULUS
        )
    return result


def _consumer_factor_eval(
    challenges: tuple[int, ...],
    indices: tuple[int, ...],
    coefficients: tuple[int, ...],
) -> int:
    return sum(
        coefficient * _boolean_eq_at_msb_point(challenges, index)
        for index, coefficient in zip(
            indices,
            coefficients,
            strict=True,
        )
    ) % GOLDILOCKS_MODULUS


def prove_goldilocks_projection_composition_v3(
    *,
    validator_binding_digest: bytes,
    capture_base_binding_digest: bytes,
    validator_nonce: bytes,
    witnesses,
    weight_rows_i8,
    fused=None,
    external_collector=None,
    collector_ns: str = "",
) -> GoldilocksProjectionCompositionProofV3:
    """Build the complete selected-row projection composition."""

    witnesses_t = tuple(witnesses)
    claims = tuple(witness.claim for witness in witnesses_t)
    claims_digest = _claims_digest(claims)
    phase1_tile = _phase1_tile(
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
        claims_digest=claims_digest,
    )
    phase1_ordered = []
    for index, witness in enumerate(witnesses_t):
        x_tag, s_tag, _z_tag = _tags(index)
        phase1_ordered.extend(
            (
                (
                    x_tag,
                    _selected_values(
                        witness.committed_x,
                        witness.claim.selected_rows,
                        fused=fused,
                    ),
                ),
                (
                    s_tag,
                    _selected_values(
                        witness.committed_s,
                        witness.claim.selected_rows,
                        fused=fused,
                    ),
                ),
            )
        )
        if witness.claim.runtime is not None:
            phase1_ordered.append(
                (
                    _runtime_y_tag(index),
                    _selected_values(
                        witness.committed_y,
                        witness.claim.selected_rows,
                        fused=fused,
                    ),
                )
            )
    with pcs_coset_profile_v3("chain"):
        phase1_groups, phase1_members, _phase1_plans = (
            commit_succinct_variable_column_groups_v3(
                tile_digest=phase1_tile,
                group_tag_prefix="projection/phase1",
                ordered=tuple(phase1_ordered),
                max_group_cells=MAX_PROJECTION_GROUP_CELLS_V3,
                fused=fused,
            )
        )
    phase1_roots = _root_records(phase1_groups)
    coefficients = derive_goldilocks_projection_coefficients_v3(
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
        claims=claims,
        phase1_groups=phase1_roots,
    )
    folded_weights = _build_folded_weights(
        claims=claims,
        coefficients=coefficients,
        weight_rows_i8=weight_rows_i8,
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
        fused=fused,
    )
    phase2_tile = _phase2_tile(
        phase1_tile=phase1_tile,
        phase1_roots=phase1_roots,
        coefficients=coefficients,
    )
    with pcs_coset_profile_v3("chain"):
        phase2_groups, phase2_members, _phase2_plans = (
            commit_succinct_variable_column_groups_v3(
                tile_digest=phase2_tile,
                group_tag_prefix="projection/phase2",
                ordered=tuple(
                    (
                        _tags(index)[2],
                        _folded_values(
                            folded_weights[index],
                            fused=fused,
                        ),
                    )
                    for index, witness in enumerate(witnesses_t)
                ),
                max_group_cells=MAX_PROJECTION_GROUP_CELLS_V3,
                fused=fused,
            )
        )
    phase2_roots = _root_records(phase2_groups)
    relation_seed = _relation_seed(
        phase2_tile=phase2_tile,
        phase2_roots=phase2_roots,
    )
    if external_collector is not None:
        from verallm.proof_v3.goldilocks_succinct_batch_opening import (
            NamespacedCollectorV3,
        )

        collector = NamespacedCollectorV3(
            external_collector,
            collector_ns,
        )
    else:
        collector = BatchOpeningCollectorV3()
    for group in (*phase1_groups, *phase2_groups):
        collector.register_group(group)
    for member in (*phase1_members.values(), *phase2_members.values()):
        collector.register_column(member.tag, member)

    captures = []
    relations = []
    runtime_metrics: list[tuple[str, float, float]] = []
    for index, witness in enumerate(witnesses_t):
        claim = witness.claim
        x_tag, s_tag, z_tag = _tags(index)
        x_row_squares = _selected_x_row_squares(witness)
        if claim.x_anchor is None:
            capture = GoldilocksProjectionCaptureProofV3(
                binding_mode=PROJECTION_HELPER_ROOT_BINDING_MODE_V3,
                x_binding=prove_goldilocks_capture_pcs_binding_v3(
                    tile_digest=phase1_tile,
                    capture_base_binding_digest=(
                        capture_base_binding_digest
                    ),
                    validator_nonce=validator_nonce,
                    tag=x_tag,
                    committed_oracle=witness.committed_x,
                    pcs_column=phase1_members[x_tag],
                    selected_rows=claim.selected_rows,
                    collector=collector,
                    value_mode=VALUE_MODE_INT8,
                ),
                s_binding=prove_goldilocks_capture_pcs_binding_v3(
                    tile_digest=phase1_tile,
                    capture_base_binding_digest=(
                        capture_base_binding_digest
                    ),
                    validator_nonce=validator_nonce,
                    tag=s_tag,
                    committed_oracle=witness.committed_s,
                    pcs_column=phase1_members[s_tag],
                    selected_rows=claim.selected_rows,
                    collector=collector,
                    value_mode=VALUE_MODE_BOUNDED,
                    bounded_width=bounded_byte_width_v3(
                        claim.operation.input_dim
                    ),
                ),
                y_binding=None,
                x_row_squares=x_row_squares,
                input_cells=(),
                runtime_cells=(),
            )
        else:
            source = witness.x_anchor_witness
            if source is None:
                raise ProofV3Error(
                    "projection execution-anchor witness is missing"
                )
            anchor = claim.x_anchor
            required_input_columns = tuple(
                sorted(
                    {
                        column
                        for _row, column
                        in _consumer_input_coordinates(claim)
                    }
                )
            )
            lane_keys = (
                derive_goldilocks_execution_anchor_pcs_lanes_v3(
                    tile_digest=phase1_tile,
                    validator_nonce=validator_nonce,
                    tag=x_tag,
                    anchor=anchor.commitment,
                    anchor_rows=anchor.anchor_rows,
                    pcs_column=phase1_members[x_tag],
                    source_column_offset=anchor.source_column_offset,
                    active_columns=claim.operation.input_dim,
                    scale_bits=claim.x_oracle.scale_bits,
                    encoding_id=anchor.encoding_id,
                    required_member_columns=required_input_columns,
                )
            )
            lane_openings = (
                build_goldilocks_execution_anchor_lane_openings_v3(
                    commitment=anchor.commitment,
                    row_bytes_by_index=source.row_bytes_by_index,
                    row_tree=source.row_tree,
                    lane_keys=lane_keys,
                )
            )
            capture = GoldilocksProjectionCaptureProofV3(
                binding_mode=PROJECTION_EXECUTION_ANCHOR_BINDING_MODE_V3,
                x_binding=(
                    prove_goldilocks_execution_anchor_pcs_binding_v3(
                        tile_digest=phase1_tile,
                        validator_nonce=validator_nonce,
                        tag=x_tag,
                        anchor=anchor.commitment,
                        anchor_rows=anchor.anchor_rows,
                        pcs_column=phase1_members[x_tag],
                        source_column_offset=(
                            anchor.source_column_offset
                        ),
                        active_columns=claim.operation.input_dim,
                        scale_bits=claim.x_oracle.scale_bits,
                        encoding_id=anchor.encoding_id,
                        lane_openings=lane_openings,
                        collector=collector,
                        required_member_columns=required_input_columns,
                    )
                ),
                s_binding=None,
                y_binding=None,
                x_row_squares=x_row_squares,
                input_cells=(),
                runtime_cells=(),
            )
        input_cells = []
        input_pad = claim.operation.padded_input_dim
        x_column = phase1_members[x_tag]
        for selected_row, input_column in _consumer_input_coordinates(
            claim
        ):
            row_slot = claim.selected_rows.index(selected_row)
            x_value = witness.committed_x.signed_value(
                selected_row,
                input_column,
            )
            member_cell = row_slot * input_pad + input_column
            collector.defer(
                x_tag,
                _boolean_point(
                    member_cell,
                    x_column.pcs_statement.variable_count,
                ),
                x_value % GOLDILOCKS_MODULUS,
            )
            input_cells.append(x_value)
        capture = replace(
            capture,
            input_cells=tuple(input_cells),
        )
        if claim.runtime is not None:
            runtime = claim.runtime
            y_source = witness.y_anchor_witness
            committed_y = witness.committed_y
            if committed_y is None or (
                (runtime.y_anchor is None) != (y_source is None)
            ):
                raise ProofV3Error(
                    "projection runtime-output witness is missing"
                )
            y_tag = _runtime_y_tag(index)
            y_column = phase1_members[y_tag]
            y_anchor = runtime.y_anchor
            y_binding = None
            if y_anchor is not None:
                y_lane_keys = (
                    derive_goldilocks_execution_anchor_pcs_lanes_v3(
                        tile_digest=phase1_tile,
                        validator_nonce=validator_nonce,
                        tag=y_tag,
                        anchor=y_anchor.commitment,
                        anchor_rows=y_anchor.anchor_rows,
                        pcs_column=y_column,
                        source_column_offset=y_anchor.source_column_offset,
                        active_columns=claim.operation.output_dim,
                        scale_bits=runtime.y_oracle.scale_bits,
                        encoding_id=y_anchor.encoding_id,
                        required_member_columns=runtime.output_columns,
                    )
                )
                y_lane_openings = (
                    build_goldilocks_execution_anchor_lane_openings_v3(
                        commitment=y_anchor.commitment,
                        row_bytes_by_index=y_source.row_bytes_by_index,
                        row_tree=y_source.row_tree,
                        lane_keys=y_lane_keys,
                    )
                )
                y_binding = (
                    prove_goldilocks_execution_anchor_pcs_binding_v3(
                        tile_digest=phase1_tile,
                        validator_nonce=validator_nonce,
                        tag=y_tag,
                        anchor=y_anchor.commitment,
                        anchor_rows=y_anchor.anchor_rows,
                        pcs_column=y_column,
                        source_column_offset=y_anchor.source_column_offset,
                        active_columns=claim.operation.output_dim,
                        scale_bits=runtime.y_oracle.scale_bits,
                        encoding_id=y_anchor.encoding_id,
                        lane_openings=y_lane_openings,
                        collector=collector,
                        required_member_columns=runtime.output_columns,
                    )
                )
            output_pad = _pow2(claim.operation.output_dim)
            runtime_cells = []
            for row_slot, selected_row in enumerate(claim.selected_rows):
                for output in runtime.output_columns:
                    s_value = witness.committed_s.signed_value(
                        selected_row,
                        output,
                    )
                    y_value = committed_y.signed_value(
                        selected_row,
                        output,
                    )
                    member_cell = row_slot * output_pad + output
                    point = _boolean_point(
                        member_cell,
                        y_column.pcs_statement.variable_count,
                    )
                    collector.defer(s_tag, point, s_value % GOLDILOCKS_MODULUS)
                    collector.defer(y_tag, point, y_value % GOLDILOCKS_MODULUS)
                    delta, spread = _projection_runtime_metrics(
                        runtime=runtime,
                        s_value=s_value,
                        y_value=y_value,
                        x_square=x_row_squares[row_slot],
                        output_column=output,
                        x_scale_bits=claim.x_oracle.scale_bits,
                        coordinate=(
                            f"{runtime.corridor_kind},"
                            f"row={selected_row},column={output}"
                        ),
                    )
                    runtime_metrics.append(
                        (runtime.corridor_kind, delta, spread)
                    )
                    runtime_cells.append((s_value, y_value))
            capture = replace(
                capture,
                y_binding=y_binding,
                runtime_cells=tuple(runtime_cells),
            )
        output_cells = tuple(
            witness.committed_s.signed_value(row, output)
            for row, output in claim.consumer_output_cells
        )
        capture = replace(
            capture,
            output_cells=output_cells,
        )
        captures.append(capture)
        row_pad = _pow2(len(claim.selected_rows))
        input_pad = claim.operation.padded_input_dim
        output_pad = _pow2(claim.operation.output_dim)
        row_bits = int(math.log2(row_pad))
        input_bits = int(math.log2(input_pad))
        output_bits = int(math.log2(output_pad))
        alpha, beta = _relation_material(relation_seed, index, claim)
        x_column = phase1_members[x_tag]
        s_column = phase1_members[s_tag]
        z_column = phase2_members[z_tag]
        x_source = (
            x_column.device_values
            if x_column.device_values is not None
            else x_column.values
        )
        z_source = (
            z_column.device_values
            if z_column.device_values is not None
            else z_column.values
        )
        if fused is not None:
            x_small = x_column.device_values.reshape(row_pad, input_pad)
            z_small = z_column.device_values.reshape(
                LEAN_PROJECTION_FOLD_COUNT_V3,
                input_pad,
            )
            x_broadcast = (
                x_small.unsqueeze(0)
                .expand(
                    LEAN_PROJECTION_FOLD_COUNT_V3,
                    row_pad,
                    input_pad,
                )
                .reshape(-1)
            )
            z_broadcast = (
                z_small.unsqueeze(1)
                .expand(
                    LEAN_PROJECTION_FOLD_COUNT_V3,
                    row_pad,
                    input_pad,
                )
                .reshape(-1)
            )
            from verallm.proof_v3.native_pcs_backend import (
                fused_prove_goldilocks_succinct_product_v3,
            )

            product = fused_prove_goldilocks_succinct_product_v3(
                fold_extension=fused[0],
                tree_extension=fused[1],
                statement=_relation_statement(
                    relation_seed,
                    index,
                    row_pad,
                    input_pad,
                ),
                a_column=x_column,
                b_column=z_column,
                factor_components=(beta, alpha, (1,) * input_pad),
                validator_nonce=validator_nonce,
                collector=collector,
                a_tag=x_tag,
                b_tag=z_tag,
                a_point_map=tuple(range(input_bits + row_bits)),
                b_point_map=(
                    tuple(range(input_bits))
                    + tuple(
                        range(
                            input_bits + row_bits,
                            input_bits + row_bits + 2,
                        )
                    )
                ),
                a_fold_device=x_broadcast,
                b_fold_device=z_broadcast,
            )
        else:
            x_small = tuple(x_source)
            z_small = tuple(z_source)
            x_broadcast = tuple(
                x_small[row * input_pad + inner]
                for _fold in range(LEAN_PROJECTION_FOLD_COUNT_V3)
                for row in range(row_pad)
                for inner in range(input_pad)
            )
            z_broadcast = tuple(
                z_small[fold * input_pad + inner]
                for fold in range(LEAN_PROJECTION_FOLD_COUNT_V3)
                for _row in range(row_pad)
                for inner in range(input_pad)
            )
            product = prove_goldilocks_succinct_product_v3(
                statement=_relation_statement(
                    relation_seed,
                    index,
                    row_pad,
                    input_pad,
                ),
                a_pcs_statement=x_column.pcs_statement,
                b_pcs_statement=z_column.pcs_statement,
                a_tree=x_column.tree,
                b_tree=z_column.tree,
                a_evaluations=x_broadcast,
                b_evaluations=z_broadcast,
                factor_components=(beta, alpha, (1,) * input_pad),
                validator_nonce=validator_nonce,
                collector=collector,
                a_tag=x_tag,
                b_tag=z_tag,
                a_point_map=tuple(range(input_bits + row_bits)),
                b_point_map=(
                    tuple(range(input_bits))
                    + tuple(
                        range(
                            input_bits + row_bits,
                            input_bits + row_bits + 2,
                        )
                    )
                ),
            )
        square_alpha = _square_alpha(
            relation_seed,
            index,
            row_pad,
        )
        square_statement = _square_statement(
            relation_seed,
            index,
            row_pad,
            input_pad,
            x_row_squares,
        )
        if fused is not None:
            x_square_product = (
                fused_prove_goldilocks_succinct_product_v3(
                    fold_extension=fused[0],
                    tree_extension=fused[1],
                    statement=square_statement,
                    a_column=x_column,
                    b_column=x_column,
                    factor_components=(
                        square_alpha,
                        (1,) * input_pad,
                    ),
                    validator_nonce=validator_nonce,
                    collector=collector,
                    a_tag=x_tag,
                    b_tag=x_tag,
                )
            )
        else:
            x_values = tuple(x_source)
            x_square_product = prove_goldilocks_succinct_product_v3(
                statement=square_statement,
                a_pcs_statement=x_column.pcs_statement,
                b_pcs_statement=x_column.pcs_statement,
                a_tree=x_column.tree,
                b_tree=x_column.tree,
                a_evaluations=x_values,
                b_evaluations=x_values,
                factor_components=(
                    square_alpha,
                    (1,) * input_pad,
                ),
                validator_nonce=validator_nonce,
                collector=collector,
                a_tag=x_tag,
                b_tag=x_tag,
            )
        expected_square_sum = sum(
            coefficient * square
            for coefficient, square in zip(
                square_alpha,
                x_row_squares + (0,) * (
                    row_pad - len(x_row_squares)
                ),
                strict=True,
            )
        ) % GOLDILOCKS_MODULUS
        if x_square_product.claimed_sum != expected_square_sum:
            raise ProofV3Error(
                "projection input row-square relation is not satisfied"
            )
        padded_coefficients = tuple(
            tuple(row) + (0,) * (output_pad - len(row))
            for row in coefficients[index]
        )
        g_values = tuple(
            beta[fold] * padded_coefficients[fold][output]
            % GOLDILOCKS_MODULUS
            for fold in range(LEAN_PROJECTION_FOLD_COUNT_V3)
            for output in range(output_pad)
        )
        consumer_indices, consumer_coefficients = _consumer_material(
            relation_seed,
            index,
            claim,
        )
        consumer_sum = _consumer_claimed_sum(
            output_cells,
            consumer_coefficients,
        )
        if fused is not None:
            s_small = s_column.device_values.reshape(row_pad, output_pad)
            s_broadcast = (
                s_small.transpose(0, 1)
                .contiguous()
                .unsqueeze(0)
                .expand(
                    LEAN_PROJECTION_FOLD_COUNT_V3,
                    output_pad,
                    row_pad,
                )
                .reshape(-1)
            )
            s_factor = _add_sparse_factor_device(
                _factor_device((g_values, alpha)),
                consumer_indices,
                consumer_coefficients,
            )
        else:
            s_values = tuple(s_column.values)
            s_broadcast = tuple(
                s_values[row * output_pad + output]
                for _fold in range(LEAN_PROJECTION_FOLD_COUNT_V3)
                for output in range(output_pad)
                for row in range(row_pad)
            )
            s_factor = None
            host_factor = [
                g * a % GOLDILOCKS_MODULUS
                for g in g_values
                for a in alpha
            ]
            for cell, coefficient in zip(
                consumer_indices,
                consumer_coefficients,
                strict=True,
            ):
                host_factor[cell] = (
                    host_factor[cell] + coefficient
                ) % GOLDILOCKS_MODULUS
        surrogate_fold = prove_succinct_public_fold_v3(
            tile_digest=relation_seed,
            column=s_column,
            factor=(
                ()
                if fused is not None
                else tuple(host_factor)
            ),
            label=f"projection-surrogate/{index}",
            validator_nonce=validator_nonce,
            fused=fused,
            collector=collector,
            structured_binding=_surrogate_binding(relation_seed, index),
            product_values=s_broadcast,
            factor_device=s_factor,
            product_variable_count=int(
                math.log2(
                    LEAN_PROJECTION_FOLD_COUNT_V3
                    * output_pad
                    * row_pad
                )
            ),
            point_map=(
                tuple(range(row_bits, row_bits + output_bits))
                + tuple(range(row_bits))
            ),
        )
        if (
            product.claimed_sum + consumer_sum
        ) % GOLDILOCKS_MODULUS != surrogate_fold.claimed_sum:
            raise ProofV3Error(
                "projection composition relation is not satisfied")
        relations.append(
            GoldilocksProjectionRelationProofV3(
                product=product,
                surrogate_fold=surrogate_fold,
                x_square_product=x_square_product,
            )
        )

    _verify_projection_runtime_aggregate(runtime_metrics, claims)
    static_bridge = prove_goldilocks_static_catalog_bridge_v3(
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
        operations=tuple(claim.operation for claim in claims),
        coefficient_rows=coefficients,
        folded_weights_i64=folded_weights,
        z_columns=tuple(
            phase2_members[_tags(index)[2]]
            for index in range(len(claims))
        ),
        collector=collector,
        fused=fused,
    )
    if external_collector is not None:
        from verallm.proof_v3.goldilocks_succinct_batch_opening import (
            park_column_device_values_v3,
        )

        for group in (*phase1_groups, *phase2_groups):
            park_column_device_values_v3(group)
        batch_opening = None
    else:
        batch_opening = collector.prove_all_batched(
            validator_nonce=validator_nonce,
            fused=fused,
        ) if fused is not None else collector.prove_all(
            validator_nonce=validator_nonce,
        )
    return GoldilocksProjectionCompositionProofV3(
        phase1_groups=phase1_roots,
        phase2_groups=phase2_roots,
        captures=tuple(captures),
        relations=tuple(relations),
        static_bridge=static_bridge,
        batch_opening=batch_opening,
    )


def verify_goldilocks_projection_composition_v3(
    proof: object,
    *,
    validator_binding_digest: bytes,
    capture_base_binding_digest: bytes,
    validator_nonce: bytes,
    claims,
    batched_opening: bool = False,
    external_checker=None,
    checker_ns: str = "",
) -> None | tuple[dict[str, object], dict[str, bytes]]:
    """Verify the complete selected-row projection composition."""

    try:
        if not isinstance(proof, GoldilocksProjectionCompositionProofV3):
            raise ProofV3VerificationError(
                "projection composition proof has a wrong type")
        claims_t = tuple(claims)
        if (
            len(claims_t) != len(proof.captures)
            or len(claims_t) != len(proof.relations)
        ):
            raise ProofV3VerificationError(
                "projection composition inventory is inconsistent")
        claims_digest = _claims_digest(claims_t)
        phase1_tile = _phase1_tile(
            validator_binding_digest=validator_binding_digest,
            validator_nonce=validator_nonce,
            claims_digest=claims_digest,
        )
        with pcs_coset_profile_v3("chain"):
            phase1_groups, phase1_members, _plans1 = _public_columns(
                tile_digest=phase1_tile,
                group_tag_prefix="projection/phase1",
                sizes=_phase1_sizes(claims_t),
                root_records=proof.phase1_groups,
            )
        coefficients = derive_goldilocks_projection_coefficients_v3(
            validator_binding_digest=validator_binding_digest,
            validator_nonce=validator_nonce,
            claims=claims_t,
            phase1_groups=proof.phase1_groups,
        )
        phase2_tile = _phase2_tile(
            phase1_tile=phase1_tile,
            phase1_roots=proof.phase1_groups,
            coefficients=coefficients,
        )
        with pcs_coset_profile_v3("chain"):
            phase2_groups, phase2_members, _plans2 = _public_columns(
                tile_digest=phase2_tile,
                group_tag_prefix="projection/phase2",
                sizes=_phase2_sizes(claims_t),
                root_records=proof.phase2_groups,
            )
        relation_seed = _relation_seed(
            phase2_tile=phase2_tile,
            phase2_roots=proof.phase2_groups,
        )
        if external_checker is not None:
            from verallm.proof_v3.goldilocks_succinct_batch_opening import (
                NamespacedCheckerV3,
            )

            checker = NamespacedCheckerV3(
                external_checker,
                checker_ns,
            )
        else:
            checker = BatchClaimCheckerV3()
        for member in (*phase1_members.values(), *phase2_members.values()):
            checker.alias(
                member.tag,
                member.group_tag,
                member.block_point,
            )
        runtime_metrics: list[tuple[str, float, float]] = []
        for index, (claim, capture, relation) in enumerate(
            zip(
                claims_t,
                proof.captures,
                proof.relations,
                strict=True,
            )
        ):
            x_tag, s_tag, z_tag = _tags(index)
            bound_x = {}
            if claim.x_anchor is None:
                if (
                    capture.binding_mode
                    != PROJECTION_HELPER_ROOT_BINDING_MODE_V3
                    or not isinstance(
                        capture.x_binding,
                        GoldilocksCapturePcsBindingProofV3,
                    )
                    or not isinstance(
                        capture.s_binding,
                        GoldilocksCapturePcsBindingProofV3,
                    )
                ):
                    raise ProofV3VerificationError(
                        "projection helper-root binding mode is inconsistent"
                    )
                verify_goldilocks_capture_pcs_binding_v3(
                    capture.x_binding,
                    tile_digest=phase1_tile,
                    capture_base_binding_digest=capture_base_binding_digest,
                    validator_nonce=validator_nonce,
                    tag=x_tag,
                    oracle=claim.x_oracle,
                    pcs_column=phase1_members[x_tag],
                    selected_rows=claim.selected_rows,
                    checker=checker,
                    expected_mode=VALUE_MODE_INT8,
                )
                verify_goldilocks_capture_pcs_binding_v3(
                    capture.s_binding,
                    tile_digest=phase1_tile,
                    capture_base_binding_digest=capture_base_binding_digest,
                    validator_nonce=validator_nonce,
                    tag=s_tag,
                    oracle=claim.s_oracle,
                    pcs_column=phase1_members[s_tag],
                    selected_rows=claim.selected_rows,
                    checker=checker,
                    expected_mode=VALUE_MODE_BOUNDED,
                    expected_bounded_width=bounded_byte_width_v3(
                        claim.operation.input_dim
                    ),
                )
            else:
                if (
                    capture.binding_mode
                    != PROJECTION_EXECUTION_ANCHOR_BINDING_MODE_V3
                    or not isinstance(
                        capture.x_binding,
                        GoldilocksExecutionAnchorPcsBindingProofV3,
                    )
                    or capture.s_binding is not None
                ):
                    raise ProofV3VerificationError(
                        "projection execution-anchor binding mode is "
                        "inconsistent"
                    )
                anchor = claim.x_anchor
                required_input_columns = tuple(
                    sorted(
                        {
                            column
                            for _row, column
                            in _consumer_input_coordinates(claim)
                        }
                    )
                )
                bound_x = dict(
                    verify_goldilocks_execution_anchor_pcs_binding_v3(
                        capture.x_binding,
                        tile_digest=phase1_tile,
                        validator_nonce=validator_nonce,
                        tag=x_tag,
                        anchor=anchor.commitment,
                        anchor_rows=anchor.anchor_rows,
                        pcs_column=phase1_members[x_tag],
                        source_column_offset=anchor.source_column_offset,
                        active_columns=claim.operation.input_dim,
                        scale_bits=claim.x_oracle.scale_bits,
                        encoding_id=anchor.encoding_id,
                        checker=checker,
                        required_member_columns=required_input_columns,
                    )
                )
            input_coordinates = _consumer_input_coordinates(claim)
            if len(capture.input_cells) != len(input_coordinates):
                raise ProofV3VerificationError(
                    "projection consumer-input cell inventory is "
                    "inconsistent"
                )
            input_pad = claim.operation.padded_input_dim
            x_column = phase1_members[x_tag]
            row_slots = {
                row: slot
                for slot, row in enumerate(claim.selected_rows)
            }
            for (row, input_column), x_value in zip(
                input_coordinates,
                capture.input_cells,
                strict=True,
            ):
                member_cell = (
                    row_slots[row] * input_pad + input_column
                )
                if (
                    claim.x_anchor is not None
                    and bound_x.get(member_cell) != x_value
                ):
                    raise ProofV3VerificationError(
                        "projection consumer-input cell is detached "
                        "from its execution anchor"
                    )
                checker.expect(
                    x_tag,
                    _boolean_point(
                        member_cell,
                        x_column.pcs_statement.variable_count,
                    ),
                    x_value,
                )
            if claim.runtime is None:
                if (
                    capture.y_binding is not None
                    or capture.runtime_cells
                ):
                    raise ProofV3VerificationError(
                        "projection proof carries an unexpected runtime "
                        "output"
                    )
            else:
                runtime = claim.runtime
                if (
                    runtime.y_anchor is None
                    and capture.y_binding is not None
                ) or (
                    runtime.y_anchor is not None
                    and not isinstance(
                        capture.y_binding,
                        GoldilocksExecutionAnchorPcsBindingProofV3,
                    )
                ):
                    raise ProofV3VerificationError(
                        "projection runtime-output binding mode is inconsistent"
                    )
                expected_cells = (
                    len(claim.selected_rows)
                    * len(runtime.output_columns)
                )
                if len(capture.runtime_cells) != expected_cells:
                    raise ProofV3VerificationError(
                        "projection runtime-output cell inventory is "
                        "inconsistent"
                    )
                y_tag = _runtime_y_tag(index)
                y_column = phase1_members[y_tag]
                y_anchor = runtime.y_anchor
                if y_anchor is not None:
                    verify_goldilocks_execution_anchor_pcs_binding_v3(
                        capture.y_binding,
                        tile_digest=phase1_tile,
                        validator_nonce=validator_nonce,
                        tag=y_tag,
                        anchor=y_anchor.commitment,
                        anchor_rows=y_anchor.anchor_rows,
                        pcs_column=y_column,
                        source_column_offset=y_anchor.source_column_offset,
                        active_columns=claim.operation.output_dim,
                        scale_bits=runtime.y_oracle.scale_bits,
                        encoding_id=y_anchor.encoding_id,
                        checker=checker,
                        required_member_columns=runtime.output_columns,
                    )
                output_pad = _pow2(claim.operation.output_dim)
                s_column = phase1_members[s_tag]
                cell_slot = 0
                for row_slot, selected_row in enumerate(
                    claim.selected_rows
                ):
                    for output in runtime.output_columns:
                        s_value, y_value = capture.runtime_cells[cell_slot]
                        cell_slot += 1
                        member_cell = row_slot * output_pad + output
                        checker.expect(
                            s_tag,
                            _boolean_point(
                                member_cell,
                                s_column.pcs_statement.variable_count,
                            ),
                            s_value,
                        )
                        checker.expect(
                            y_tag,
                            _boolean_point(
                                member_cell,
                                y_column.pcs_statement.variable_count,
                            ),
                            y_value,
                        )
                        delta, spread = _projection_runtime_metrics(
                            runtime=runtime,
                            s_value=s_value,
                            y_value=y_value,
                            x_square=capture.x_row_squares[row_slot],
                            output_column=output,
                            x_scale_bits=claim.x_oracle.scale_bits,
                            coordinate=(
                                f"{runtime.corridor_kind},"
                                f"row={selected_row},column={output}"
                            ),
                        )
                        runtime_metrics.append(
                            (runtime.corridor_kind, delta, spread)
                        )
            expected_output_cells = len(claim.consumer_output_cells)
            if len(capture.output_cells) != expected_output_cells:
                raise ProofV3VerificationError(
                    "projection consumer-output cell inventory is "
                    "inconsistent"
                )
            row_pad = _pow2(len(claim.selected_rows))
            input_pad = claim.operation.padded_input_dim
            output_pad = _pow2(claim.operation.output_dim)
            row_bits = int(math.log2(row_pad))
            input_bits = int(math.log2(input_pad))
            output_bits = int(math.log2(output_pad))
            if len(capture.x_row_squares) != len(claim.selected_rows):
                raise ProofV3VerificationError(
                    "projection input row-square inventory is inconsistent"
                )
            alpha, beta = _relation_material(
                relation_seed,
                index,
                claim,
            )
            padded_coefficients = tuple(
                tuple(row) + (0,) * (output_pad - len(row))
                for row in coefficients[index]
            )
            g_values = tuple(
                beta[fold] * padded_coefficients[fold][output]
                % GOLDILOCKS_MODULUS
                for fold in range(LEAN_PROJECTION_FOLD_COUNT_V3)
                for output in range(output_pad)
            )
            consumer_indices, consumer_coefficients = _consumer_material(
                relation_seed,
                index,
                claim,
            )
            consumer_sum = _consumer_claimed_sum(
                capture.output_cells,
                consumer_coefficients,
            )
            verify_goldilocks_succinct_product_v3(
                relation.product,
                statement=_relation_statement(
                    relation_seed,
                    index,
                    row_pad,
                    input_pad,
                ),
                a_pcs_statement=phase1_members[x_tag].pcs_statement,
                b_pcs_statement=phase2_members[z_tag].pcs_statement,
                a_commitment=phase1_members[x_tag].commitment,
                b_commitment=phase2_members[z_tag].commitment,
                factor_components=(beta, alpha, (1,) * input_pad),
                validator_nonce=validator_nonce,
                expected_sum=(
                    relation.surrogate_fold.claimed_sum
                    - consumer_sum
                ) % GOLDILOCKS_MODULUS,
                checker=checker,
                a_tag=x_tag,
                b_tag=z_tag,
                a_point_map=tuple(range(input_bits + row_bits)),
                b_point_map=(
                    tuple(range(input_bits))
                    + tuple(
                        range(
                            input_bits + row_bits,
                            input_bits + row_bits + 2,
                        )
                    )
                ),
            )
            square_alpha = _square_alpha(
                relation_seed,
                index,
                row_pad,
            )
            expected_square_sum = sum(
                coefficient * square
                for coefficient, square in zip(
                    square_alpha,
                    capture.x_row_squares + (0,) * (
                        row_pad - len(capture.x_row_squares)
                    ),
                    strict=True,
                )
            ) % GOLDILOCKS_MODULUS
            verify_goldilocks_succinct_product_v3(
                relation.x_square_product,
                statement=_square_statement(
                    relation_seed,
                    index,
                    row_pad,
                    input_pad,
                    capture.x_row_squares,
                ),
                a_pcs_statement=phase1_members[x_tag].pcs_statement,
                b_pcs_statement=phase1_members[x_tag].pcs_statement,
                a_commitment=phase1_members[x_tag].commitment,
                b_commitment=phase1_members[x_tag].commitment,
                factor_components=(
                    square_alpha,
                    (1,) * input_pad,
                ),
                validator_nonce=validator_nonce,
                expected_sum=expected_square_sum,
                checker=checker,
                a_tag=x_tag,
                b_tag=x_tag,
            )
            verify_succinct_public_fold_v3(
                relation.surrogate_fold,
                tile_digest=relation_seed,
                label=f"projection-surrogate/{index}",
                pcs_statement=phase1_members[s_tag].pcs_statement,
                commitment=phase1_members[s_tag].commitment,
                factor=(),
                validator_nonce=validator_nonce,
                checker=checker,
                tag=s_tag,
                factor_eval=lambda challenges, gv=g_values, av=alpha, split=(
                    2 + output_bits
                ), ci=consumer_indices, cc=consumer_coefficients: (
                    (
                        _mle_eval_msb_local(gv, challenges[:split])
                        * _mle_eval_msb_local(av, challenges[split:])
                    )
                    + _consumer_factor_eval(
                        tuple(challenges),
                        ci,
                        cc,
                    )
                ) % GOLDILOCKS_MODULUS,
                structured_binding=_surrogate_binding(
                    relation_seed,
                    index,
                ),
                product_variable_count=int(
                    math.log2(
                        LEAN_PROJECTION_FOLD_COUNT_V3
                        * output_pad
                        * row_pad
                    )
                ),
                point_map=(
                    tuple(range(row_bits, row_bits + output_bits))
                    + tuple(range(row_bits))
                ),
            )
        _verify_projection_runtime_aggregate(runtime_metrics, claims_t)
        verify_goldilocks_static_catalog_bridge_v3(
            proof.static_bridge,
            validator_binding_digest=validator_binding_digest,
            validator_nonce=validator_nonce,
            operations=tuple(claim.operation for claim in claims_t),
            coefficient_rows=coefficients,
            z_columns=tuple(
                phase2_members[_tags(index)[2]]
                for index in range(len(claims_t))
            ),
            checker=checker,
        )
        statements = {
            group.tag: group.pcs_statement
            for group in (*phase1_groups, *phase2_groups)
        }
        commitments = {
            group.tag: group.commitment
            for group in (*phase1_groups, *phase2_groups)
        }
        if external_checker is not None:
            if proof.batch_opening is not None:
                raise ProofV3VerificationError(
                    "aggregated projection proof carries its own opening"
                )
            return (
                {
                    checker_ns + tag: statement
                    for tag, statement in statements.items()
                },
                {
                    checker_ns + tag: commitment
                    for tag, commitment in commitments.items()
                },
            )
        if proof.batch_opening is None:
            raise ProofV3VerificationError(
                "standalone projection proof lacks its opening"
            )
        if batched_opening:
            checker.verify_all_batched(
                proof.batch_opening,
                statements=statements,
                commitments=commitments,
                validator_nonce=validator_nonce,
            )
        else:
            checker.verify_all(
                proof.batch_opening,
                statements=statements,
                commitments=commitments,
                validator_nonce=validator_nonce,
            )
    except ProofV3VerificationError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "projection composition proof is malformed") from exc
