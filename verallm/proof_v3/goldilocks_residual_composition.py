"""Compact residual transitions over the shared Goldilocks opening.

The economic hard audit checks nonce-selected hidden cells rather than every
hidden coordinate.  This module preserves that signed probabilistic policy
while removing complete residual rows from the wire:

* selected residual-in, mid-residual and residual-out rows are committed under
  packed Goldilocks/BaseFold columns;
* each column is joined to its pre-nonce FP16/BF16 execution anchor, including
  every lane containing a selected residual cell;
* attention/GDN-output and MLP-down values are reused from the already verified
  registered-weight projection composition; and
* the validator applies the existing two residual corridor equations directly
  to those authenticated cells.

Adjacent-layer continuity remains a commitment-root invariant checked by the
execution-anchor inventory.  This module proves per-layer arithmetic and does
not retransmit rows merely to restate equal pre-nonce roots.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.economic_challenge import (
    CORRIDOR_QUANT_COEFF_DEN_V3,
    CORRIDOR_QUANT_COEFF_NUM_V3,
    CORRIDOR_REL_COEFF_DEN_V3,
    CORRIDOR_REL_COEFF_NUM_V3,
)
from verallm.proof_v3.economic_commitment import EconomicCommittedOracleV3
from verallm.proof_v3.economic_wire import (
    EconomicOracleCommitmentV3,
    bits_to_scale_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_execution_anchor_pcs_binding import (
    GoldilocksExecutionAnchorPcsBindingProofV3,
    build_goldilocks_execution_anchor_lane_openings_v3,
    derive_goldilocks_execution_anchor_pcs_lanes_v3,
    prove_goldilocks_execution_anchor_pcs_binding_v3,
    verify_goldilocks_execution_anchor_pcs_binding_v3,
)
from verallm.proof_v3.goldilocks_projection_composition import (
    GoldilocksProjectionAnchorClaimV3,
    GoldilocksProjectionAnchorWitnessV3,
    GoldilocksProjectionClaimV3,
    GoldilocksProjectionCompositionProofV3,
    goldilocks_projection_output_cells_v3,
    goldilocks_projection_runtime_binding_v3,
    goldilocks_projection_runtime_cells_v3,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
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
    VariableColumnGroupPlanV3,
    column_pcs_statement_v3,
    commit_succinct_variable_column_groups_v3,
    pcs_coset_profile_v3,
    plan_succinct_variable_column_groups_v3,
)
from verallm.proof_v3.lean_projection_fold import (
    lean_projection_operation_key_v3,
)


GOLDILOCKS_RESIDUAL_COMPOSITION_ABI_V3: Final = (
    "residual.selected_cells.goldilocks_fri.projection_links.v4"
)
MAX_RESIDUAL_COMPOSITION_LAYERS_V3: Final = 64
MAX_RESIDUAL_GROUP_CELLS_V3: Final = 1 << 24

_TRANSCRIPT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_RESIDUAL_COMPOSITION/V4"
)
_QUANT_COEFF: Final = (
    CORRIDOR_QUANT_COEFF_NUM_V3
    / CORRIDOR_QUANT_COEFF_DEN_V3
)
_REL_COEFF: Final = (
    CORRIDOR_REL_COEFF_NUM_V3
    / CORRIDOR_REL_COEFF_DEN_V3
)
_U31_MAX: Final = (1 << 31) - 1

__all__ = [
    "GOLDILOCKS_RESIDUAL_COMPOSITION_ABI_V3",
    "GoldilocksResidualCaptureProofV3",
    "GoldilocksResidualClaimV3",
    "GoldilocksResidualCompositionProofV3",
    "GoldilocksResidualGroupCommitmentV3",
    "GoldilocksResidualStageClaimV3",
    "GoldilocksResidualStageWitnessV3",
    "GoldilocksResidualWitnessV3",
    "goldilocks_residual_row_squares_v3",
    "goldilocks_residual_runtime_binding_v3",
    "goldilocks_residual_stage_cells_v3",
    "prove_goldilocks_residual_composition_v3",
    "verify_goldilocks_residual_composition_v3",
]


def _fixed32(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    return value


def _pow2(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


def _encoded(value: object, name: str) -> bytes:
    try:
        encoded = value.encode("ascii")
    except (AttributeError, UnicodeEncodeError) as exc:
        raise ProofV3Error(f"{name} is malformed") from exc
    if not encoded or len(encoded) > 255:
        raise ProofV3Error(f"{name} is malformed")
    return struct.pack("<B", len(encoded)) + encoded


def _oracle_record(oracle: EconomicOracleCommitmentV3) -> bytes:
    if not isinstance(oracle, EconomicOracleCommitmentV3):
        raise ProofV3Error("residual oracle is malformed")
    return (
        _encoded(oracle.oracle_id, "residual oracle id")
        + _encoded(oracle.phase, "residual oracle phase")
        + _encoded(oracle.operation, "residual oracle operation")
        + struct.pack(
            "<IIIQ",
            oracle.layer_index,
            oracle.row_count,
            oracle.col_count,
            oracle.scale_bits,
        )
        + _fixed32(oracle.root, "residual oracle root")
    )


def _anchor_record(anchor: GoldilocksProjectionAnchorClaimV3) -> bytes:
    if not isinstance(anchor, GoldilocksProjectionAnchorClaimV3):
        raise ProofV3Error("residual execution anchor is malformed")
    record = anchor.commitment.canonical_bytes()
    return (
        struct.pack("<I", len(record))
        + record
        + struct.pack(
            "<II",
            anchor.source_column_offset,
            len(anchor.anchor_rows),
        )
        + _encoded(anchor.encoding_id, "residual anchor encoding")
        + b"".join(
            struct.pack("<I", row) for row in anchor.anchor_rows
        )
    )


@dataclass(frozen=True, slots=True)
class GoldilocksResidualStageClaimV3:
    oracle: EconomicOracleCommitmentV3
    anchor: GoldilocksProjectionAnchorClaimV3 | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.oracle, EconomicOracleCommitmentV3)
            or (
                self.anchor is not None
                and (
                    not isinstance(
                        self.anchor,
                        GoldilocksProjectionAnchorClaimV3,
                    )
                    or (
                        self.anchor.source_column_offset
                        + self.oracle.col_count
                        > self.anchor.commitment.row_width // 2
                    )
                )
            )
        ):
            raise ProofV3Error(
                "residual stage claim is inconsistent"
            )


@dataclass(frozen=True, slots=True)
class GoldilocksResidualStageWitnessV3:
    committed: EconomicCommittedOracleV3
    anchor: GoldilocksProjectionAnchorWitnessV3 | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.committed, EconomicCommittedOracleV3)
            or (
                self.anchor is not None
                and not isinstance(
                    self.anchor,
                    GoldilocksProjectionAnchorWitnessV3,
                )
            )
        ):
            raise ProofV3Error("residual stage witness is malformed")


@dataclass(frozen=True, slots=True)
class GoldilocksResidualClaimV3:
    layer_index: int
    selected_rows: tuple[int, ...]
    selected_columns: tuple[int, ...]
    residual_in: GoldilocksResidualStageClaimV3
    mid_residual: GoldilocksResidualStageClaimV3
    residual_out: GoldilocksResidualStageClaimV3
    attention_projection_index: int
    attention_projection_role: str
    down_projection_index: int

    def __post_init__(self) -> None:
        rows = tuple(self.selected_rows)
        columns = tuple(self.selected_columns)
        stages = (
            self.residual_in,
            self.mid_residual,
            self.residual_out,
        )
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or not 0 <= self.layer_index < 1 << 32
            or not rows
            or rows != tuple(sorted(set(rows)))
            or not columns
            or columns != tuple(sorted(set(columns)))
            or not all(
                isinstance(stage, GoldilocksResidualStageClaimV3)
                for stage in stages
            )
            or self.attention_projection_role not in {"o", "gdn_o"}
            or isinstance(self.attention_projection_index, bool)
            or not isinstance(self.attention_projection_index, int)
            or self.attention_projection_index < 0
            or isinstance(self.down_projection_index, bool)
            or not isinstance(self.down_projection_index, int)
            or self.down_projection_index < 0
            or self.down_projection_index
            == self.attention_projection_index
        ):
            raise ProofV3Error("residual composition claim is malformed")
        row_counts = {stage.oracle.row_count for stage in stages}
        widths = {stage.oracle.col_count for stage in stages}
        anchors = tuple(
            stage.anchor for stage in stages if stage.anchor is not None
        )
        anchor_rows = {anchor.anchor_rows for anchor in anchors}
        expected_ids = (
            (
                f"l{self.layer_index}.residual_in",
                f"l{self.layer_index}.residual_in",
            ),
            (
                f"l{self.layer_index}.mid_residual",
                f"l{self.layer_index}.residual_after_attention",
            ),
            (
                f"l{self.layer_index}.residual_out",
                f"l{self.layer_index}.residual_out",
            ),
        )
        if (
            len(row_counts) != 1
            or len(widths) != 1
            or len(anchor_rows) > 1
            or any(
                row < 0 or row >= stages[0].oracle.row_count
                for row in rows
            )
            or any(
                column < 0 or column >= stages[0].oracle.col_count
                for column in columns
            )
            or any(
                stage.oracle.layer_index != self.layer_index
                or (
                    stage.anchor is not None
                    and len(stage.anchor.anchor_rows) != len(rows)
                )
                for stage in stages
            )
            or any(
                stage.oracle.oracle_id != oracle_id
                or (
                    stage.anchor is not None
                    and stage.anchor.commitment.stage_id
                    not in (
                        (stage_id,)
                        if oracle_id != f"l{self.layer_index}.residual_in"
                        or self.layer_index == 0
                        else (
                            stage_id,
                            f"l{self.layer_index - 1}.residual_out",
                        )
                    )
                )
                for stage, (oracle_id, stage_id) in zip(
                    stages,
                    expected_ids,
                    strict=True,
                )
            )
        ):
            raise ProofV3Error(
                "residual composition geometry is inconsistent"
            )
        object.__setattr__(self, "selected_rows", rows)
        object.__setattr__(self, "selected_columns", columns)


def _stage_witness_matches(
    stage: GoldilocksResidualStageClaimV3,
    witness: GoldilocksResidualStageWitnessV3,
) -> bool:
    return (
        witness.committed.commitment == stage.oracle
        and (stage.anchor is None) == (witness.anchor is None)
        and (
            stage.anchor is None
            or (
                witness.anchor.row_tree.root
                == stage.anchor.commitment.root
                and witness.anchor.row_tree.num_leaves
                == stage.anchor.commitment.row_count
                and tuple(
                    row
                    for row, _raw in witness.anchor.row_bytes_by_index
                )
                == stage.anchor.anchor_rows
                and all(
                    len(raw) == stage.anchor.commitment.row_width
                    for _row, raw in witness.anchor.row_bytes_by_index
                )
            )
        )
    )


@dataclass(frozen=True, slots=True)
class GoldilocksResidualWitnessV3:
    claim: GoldilocksResidualClaimV3
    residual_in: GoldilocksResidualStageWitnessV3
    mid_residual: GoldilocksResidualStageWitnessV3
    residual_out: GoldilocksResidualStageWitnessV3

    def __post_init__(self) -> None:
        if (
            not isinstance(self.claim, GoldilocksResidualClaimV3)
            or not _stage_witness_matches(
                self.claim.residual_in,
                self.residual_in,
            )
            or not _stage_witness_matches(
                self.claim.mid_residual,
                self.mid_residual,
            )
            or not _stage_witness_matches(
                self.claim.residual_out,
                self.residual_out,
            )
        ):
            raise ProofV3Error(
                "residual composition witness is inconsistent"
            )


@dataclass(frozen=True, slots=True)
class GoldilocksResidualGroupCommitmentV3:
    group_tag: str
    commitment: bytes

    def __post_init__(self) -> None:
        _encoded(self.group_tag, "residual group tag")
        _fixed32(self.commitment, "residual group commitment")


@dataclass(frozen=True, slots=True)
class GoldilocksResidualCaptureProofV3:
    residual_in: GoldilocksExecutionAnchorPcsBindingProofV3 | None
    mid_residual: GoldilocksExecutionAnchorPcsBindingProofV3 | None
    residual_out: GoldilocksExecutionAnchorPcsBindingProofV3 | None
    residual_in_cells: tuple[int, ...]
    mid_residual_cells: tuple[int, ...]
    residual_out_cells: tuple[int, ...]
    residual_in_row_squares: tuple[int, ...]
    mid_residual_row_squares: tuple[int, ...]
    residual_in_square_product: GoldilocksSuccinctProductProofV3
    mid_residual_square_product: GoldilocksSuccinctProductProofV3

    def __post_init__(self) -> None:
        cell_sets = (
            tuple(self.residual_in_cells),
            tuple(self.mid_residual_cells),
            tuple(self.residual_out_cells),
        )
        square_sets = (
            tuple(self.residual_in_row_squares),
            tuple(self.mid_residual_row_squares),
        )
        if (
            not all(
                item is None
                or isinstance(
                    item,
                    GoldilocksExecutionAnchorPcsBindingProofV3,
                )
                for item in (
                    self.residual_in,
                    self.mid_residual,
                    self.residual_out,
                )
            )
            or any(
                not values
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not -128 <= value <= 127
                    for value in values
                )
                for values in cell_sets
            )
            or any(
                not values
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value <= _U31_MAX
                    for value in values
                )
                for values in square_sets
            )
            or not isinstance(
                self.residual_in_square_product,
                GoldilocksSuccinctProductProofV3,
            )
            or not isinstance(
                self.mid_residual_square_product,
                GoldilocksSuccinctProductProofV3,
            )
        ):
            raise ProofV3Error(
                "residual capture proof is malformed"
            )
        object.__setattr__(
            self,
            "residual_in_cells",
            cell_sets[0],
        )
        object.__setattr__(
            self,
            "mid_residual_cells",
            cell_sets[1],
        )
        object.__setattr__(
            self,
            "residual_out_cells",
            cell_sets[2],
        )
        object.__setattr__(
            self,
            "residual_in_row_squares",
            square_sets[0],
        )
        object.__setattr__(
            self,
            "mid_residual_row_squares",
            square_sets[1],
        )


@dataclass(frozen=True, slots=True)
class GoldilocksResidualCompositionProofV3:
    groups: tuple[GoldilocksResidualGroupCommitmentV3, ...]
    captures: tuple[GoldilocksResidualCaptureProofV3, ...]
    batch_opening: object

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        captures = tuple(self.captures)
        if (
            not groups
            or not captures
            or tuple(group.group_tag for group in groups)
            != tuple(
                sorted(set(group.group_tag for group in groups))
            )
            or not all(
                isinstance(
                    group,
                    GoldilocksResidualGroupCommitmentV3,
                )
                for group in groups
            )
            or not all(
                isinstance(
                    capture,
                    GoldilocksResidualCaptureProofV3,
                )
                for capture in captures
            )
        ):
            raise ProofV3Error(
                "residual composition proof inventory is malformed"
            )
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "captures", captures)


def _claims_digest(
    claims: tuple[GoldilocksResidualClaimV3, ...],
) -> bytes:
    if (
        not claims
        or len(claims) > MAX_RESIDUAL_COMPOSITION_LAYERS_V3
        or tuple(claim.layer_index for claim in claims)
        != tuple(sorted(set(claim.layer_index for claim in claims)))
    ):
        raise ProofV3Error("residual claim inventory is malformed")
    material = bytearray(
        _TRANSCRIPT_DOMAIN
        + b"/claims/"
        + struct.pack("<I", len(claims))
    )
    for claim in claims:
        material.extend(
            struct.pack(
                "<IIII",
                claim.layer_index,
                claim.attention_projection_index,
                claim.down_projection_index,
                len(claim.selected_rows),
            )
        )
        material.extend(
            _encoded(
                claim.attention_projection_role,
                "residual attention projection role",
            )
        )
        material.extend(
            b"".join(
                struct.pack("<I", row) for row in claim.selected_rows
            )
        )
        material.extend(
            struct.pack("<I", len(claim.selected_columns))
        )
        material.extend(
            b"".join(
                struct.pack("<I", column)
                for column in claim.selected_columns
            )
        )
        for stage in (
            claim.residual_in,
            claim.mid_residual,
            claim.residual_out,
        ):
            material.extend(_oracle_record(stage.oracle))
            material.extend(struct.pack("<B", int(stage.anchor is not None)))
            if stage.anchor is not None:
                material.extend(_anchor_record(stage.anchor))
    return hashlib.sha256(bytes(material)).digest()


def _tile_digest(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    claims_digest: bytes,
    projection_binding_digest: bytes,
) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/tile/"
        + _fixed32(
            validator_binding_digest,
            "residual validator binding",
        )
        + _fixed32(validator_nonce, "residual validator nonce")
        + _fixed32(claims_digest, "residual claims digest")
        + _fixed32(
            projection_binding_digest,
            "residual projection binding",
        )
    ).digest()


def _tags(index: int) -> tuple[str, str, str]:
    return (
        f"residual/{index}/in",
        f"residual/{index}/mid",
        f"residual/{index}/out",
    )


def _sizes(
    claims: tuple[GoldilocksResidualClaimV3, ...],
) -> tuple[tuple[str, int], ...]:
    result = []
    for index, claim in enumerate(claims):
        cells = (
            _pow2(len(claim.selected_rows))
            * _pow2(claim.residual_in.oracle.col_count)
        )
        result.extend((tag, cells) for tag in _tags(index))
    return tuple(result)


def _root_records(groups) -> tuple[GoldilocksResidualGroupCommitmentV3, ...]:
    return tuple(
        GoldilocksResidualGroupCommitmentV3(
            group_tag=group.tag,
            commitment=group.tree.commitment,
        )
        for group in sorted(groups, key=lambda item: item.tag)
    )


def _root_digest(
    roots: tuple[GoldilocksResidualGroupCommitmentV3, ...],
) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/roots/"
        + struct.pack("<I", len(roots))
        + b"".join(
            _encoded(root.group_tag, "residual group tag")
            + _fixed32(
                root.commitment,
                "residual group commitment",
            )
            for root in roots
        )
    ).digest()


def _relation_seed(
    *,
    tile_digest: bytes,
    roots: tuple[GoldilocksResidualGroupCommitmentV3, ...],
) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/relations/"
        + _fixed32(tile_digest, "residual tile digest")
        + _root_digest(roots)
    ).digest()


def goldilocks_residual_runtime_binding_v3(
    proof: object,
    claims,
    *,
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims,
) -> bytes:
    """Bind a downstream relation to one verified residual composition."""

    claims_t = tuple(claims)
    projection_claims_t = tuple(projection_claims)
    if (
        not isinstance(proof, GoldilocksResidualCompositionProofV3)
        or len(claims_t) != len(proof.captures)
    ):
        raise ProofV3Error(
            "residual runtime consumer inventory is inconsistent"
        )
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/runtime-consumer/"
        + _claims_digest(claims_t)
        + _root_digest(proof.groups)
        + goldilocks_projection_runtime_binding_v3(
            projection_proof,
            projection_claims_t,
        )
    ).digest()


def _consumer_capture(
    proof: object,
    claims,
    claim_index: int,
) -> tuple[
    GoldilocksResidualClaimV3,
    GoldilocksResidualCaptureProofV3,
]:
    claims_t = tuple(claims)
    if (
        not isinstance(proof, GoldilocksResidualCompositionProofV3)
        or isinstance(claim_index, bool)
        or not isinstance(claim_index, int)
        or claim_index < 0
        or claim_index >= len(claims_t)
        or len(claims_t) != len(proof.captures)
    ):
        raise ProofV3Error(
            "residual runtime cell request is inconsistent"
        )
    return claims_t[claim_index], proof.captures[claim_index]


def goldilocks_residual_stage_cells_v3(
    proof: object,
    claims,
    *,
    claim_index: int,
    stage: str,
) -> tuple[tuple[int, int, int], ...]:
    """Return canonical ``(row, column, value)`` cells after verification."""

    claim, capture = _consumer_capture(
        proof,
        claims,
        claim_index,
    )
    try:
        values = {
            "residual_in": capture.residual_in_cells,
            "mid_residual": capture.mid_residual_cells,
            "residual_out": capture.residual_out_cells,
        }[stage]
    except (KeyError, TypeError) as exc:
        raise ProofV3Error(
            "residual runtime stage is unsupported"
        ) from exc
    expected = len(claim.selected_rows) * len(claim.selected_columns)
    if len(values) != expected:
        raise ProofV3Error(
            "residual runtime cell inventory is incomplete"
        )
    result = []
    slot = 0
    for row in claim.selected_rows:
        for column in claim.selected_columns:
            result.append((row, column, values[slot]))
            slot += 1
    return tuple(result)


def goldilocks_residual_row_squares_v3(
    proof: object,
    claims,
    *,
    claim_index: int,
    stage: str,
) -> tuple[tuple[int, int], ...]:
    """Return canonical ``(row, sum(value**2))`` after verification."""

    claim, capture = _consumer_capture(
        proof,
        claims,
        claim_index,
    )
    try:
        values = {
            "residual_in": capture.residual_in_row_squares,
            "mid_residual": capture.mid_residual_row_squares,
        }[stage]
    except (KeyError, TypeError) as exc:
        raise ProofV3Error(
            "residual row-square stage is unsupported"
        ) from exc
    if len(values) != len(claim.selected_rows):
        raise ProofV3Error(
            "residual row-square inventory is incomplete"
        )
    return tuple(zip(claim.selected_rows, values, strict=True))


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
    sizes: tuple[tuple[str, int], ...],
    root_records,
) -> tuple[
    tuple[_PublicColumnV3, ...],
    dict[str, _PublicColumnV3],
    tuple[VariableColumnGroupPlanV3, ...],
]:
    plans = plan_succinct_variable_column_groups_v3(
        tile_digest=tile_digest,
        group_tag_prefix="residual/groups",
        ordered_sizes=sizes,
        max_group_cells=MAX_RESIDUAL_GROUP_CELLS_V3,
    )
    records = tuple(root_records)
    if tuple(record.group_tag for record in records) != tuple(
        sorted(plan.group_tag for plan in plans)
    ):
        raise ProofV3VerificationError(
            "residual packed-group inventory is not exact"
        )
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

        selected = committed.int_rows_cpu[list(rows)].to(
            dtype=torch.int64,
            device="cuda",
        )
        padded = torch.zeros(
            (row_pad, col_pad),
            dtype=torch.int64,
            device="cuda",
        )
        padded[: len(rows), : committed.commitment.col_count] = selected
        return torch.where(
            padded < 0,
            padded - ((1 << 32) - 1),
            padded,
        ).reshape(-1)
    values = []
    for row in rows:
        values.extend(
            committed.signed_value(row, column) % GOLDILOCKS_MODULUS
            for column in range(committed.commitment.col_count)
        )
        values.extend(
            (0,) * (col_pad - committed.commitment.col_count)
        )
    values.extend(
        (0,) * ((row_pad - len(rows)) * col_pad)
    )
    return tuple(values)


def _selected_cells(
    committed: EconomicCommittedOracleV3,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        committed.signed_value(row, column)
        for row in rows
        for column in columns
    )


def _selected_row_squares(
    committed: EconomicCommittedOracleV3,
    rows: tuple[int, ...],
) -> tuple[int, ...]:
    result = []
    for row in rows:
        if committed.int_rows_cpu is not None:
            source = committed.int_rows_cpu[row]
            if hasattr(source, "tolist"):
                source = source.tolist()
            values = tuple(int(value) for value in source)
        else:
            values = tuple(
                committed.signed_value(row, column)
                for column in range(committed.commitment.col_count)
            )
        if (
            len(values) != committed.commitment.col_count
            or any(value < -128 or value > 127 for value in values)
        ):
            raise ProofV3Error(
                "residual row escapes its signed-int8 geometry"
            )
        square = sum(value * value for value in values)
        if square > _U31_MAX:
            raise ProofV3Error(
                "residual row square exceeds the signed bound"
            )
        result.append(square)
    return tuple(result)


def _field_vector(
    seed: bytes,
    label: bytes,
    count: int,
) -> tuple[int, ...]:
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
            value = int.from_bytes(
                block[offset:offset + 8],
                "little",
            )
            if value < GOLDILOCKS_MODULUS:
                result.append(value)
            if len(result) == count:
                break
    return tuple(result)


def _square_alpha(
    seed: bytes,
    tag: str,
    row_pad: int,
) -> tuple[int, ...]:
    return _field_vector(
        seed,
        b"row-squares/" + tag.encode("ascii"),
        row_pad,
    )


def _square_statement(
    *,
    seed: bytes,
    tag: str,
    row_pad: int,
    col_pad: int,
    row_squares: tuple[int, ...],
) -> GoldilocksSuccinctProductStatementV3:
    return GoldilocksSuccinctProductStatementV3(
        validator_binding_digest=hashlib.sha256(
            _TRANSCRIPT_DOMAIN
            + b"/row-squares/"
            + seed
            + _encoded(tag, "residual square tag")
            + struct.pack("<I", len(row_squares))
            + b"".join(
                struct.pack("<Q", value) for value in row_squares
            )
        ).digest(),
        variable_count=int(math.log2(row_pad * col_pad)),
        factor_component_sizes=(row_pad, col_pad),
    )


def _prove_square_product(
    *,
    seed: bytes,
    tag: str,
    column,
    row_squares: tuple[int, ...],
    row_count: int,
    col_count: int,
    validator_nonce: bytes,
    collector,
    fused,
) -> GoldilocksSuccinctProductProofV3:
    row_pad = _pow2(row_count)
    col_pad = _pow2(col_count)
    alpha = _square_alpha(seed, tag, row_pad)
    statement = _square_statement(
        seed=seed,
        tag=tag,
        row_pad=row_pad,
        col_pad=col_pad,
        row_squares=row_squares,
    )
    factors = (alpha, (1,) * col_pad)
    if fused is not None:
        from verallm.proof_v3.native_pcs_backend import (
            fused_prove_goldilocks_succinct_product_v3,
        )

        product = fused_prove_goldilocks_succinct_product_v3(
            fold_extension=fused[0],
            tree_extension=fused[1],
            statement=statement,
            a_column=column,
            b_column=column,
            factor_components=factors,
            validator_nonce=validator_nonce,
            collector=collector,
            a_tag=tag,
            b_tag=tag,
        )
    else:
        values = tuple(column.values)
        product = prove_goldilocks_succinct_product_v3(
            statement=statement,
            a_pcs_statement=column.pcs_statement,
            b_pcs_statement=column.pcs_statement,
            a_tree=column.tree,
            b_tree=column.tree,
            a_evaluations=values,
            b_evaluations=values,
            factor_components=factors,
            validator_nonce=validator_nonce,
            collector=collector,
            a_tag=tag,
            b_tag=tag,
        )
    expected = sum(
        coefficient * square
        for coefficient, square in zip(
            alpha,
            row_squares + (0,) * (row_pad - len(row_squares)),
            strict=True,
        )
    ) % GOLDILOCKS_MODULUS
    if product.claimed_sum != expected:
        raise ProofV3Error(
            "residual row-square relation is not satisfied"
        )
    return product


def _boolean_point(index: int, variable_count: int) -> tuple[int, ...]:
    return tuple((index >> bit) & 1 for bit in range(variable_count))


def _projection_output_values(
    *,
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims: tuple[GoldilocksProjectionClaimV3, ...],
    claim: GoldilocksResidualClaimV3,
) -> tuple[
    tuple[dict[tuple[int, int], int], float],
    tuple[dict[tuple[int, int], int], float],
]:
    indices = (
        (
            claim.attention_projection_index,
            claim.attention_projection_role,
        ),
        (claim.down_projection_index, "down"),
    )
    results = []
    anchor_rows = (
        claim.residual_in.anchor.anchor_rows
        if claim.residual_in.anchor is not None
        else None
    )
    for index, role in indices:
        if index >= len(projection_claims):
            raise ProofV3Error(
                "residual projection reference is outside the inventory"
            )
        projection = projection_claims[index]
        runtime = projection.runtime
        if (
            projection.operation.operation_key
            != lean_projection_operation_key_v3(
                layer_index=claim.layer_index,
                projection=role,
            )
            or projection.selected_rows != claim.selected_rows
            or projection.operation.output_dim
            != claim.residual_in.oracle.col_count
        ):
            raise ProofV3Error(
                "residual projection reference is inconsistent"
            )
        required = {
            (row, column)
            for row in claim.selected_rows
            for column in claim.selected_columns
        }
        if runtime is not None:
            if (
                (
                    anchor_rows is not None
                    and runtime.y_anchor is not None
                    and runtime.y_anchor.anchor_rows != anchor_rows
                )
                or not set(claim.selected_columns).issubset(
                    runtime.output_columns
                )
            ):
                raise ProofV3Error(
                    "residual projection reference is inconsistent"
                )
            cells = goldilocks_projection_runtime_cells_v3(
                projection_proof,
                projection_claims,
                claim_index=index,
            )
            values = {
                (row, column): y_value
                for row, column, _s_value, y_value in cells
                if column in claim.selected_columns
            }
            scale = bits_to_scale_v3(
                runtime.y_oracle.scale_bits
            )
        else:
            if (
                not projection.weight_scale_bits
                or not required.issubset(
                    set(projection.consumer_output_cells)
                )
            ):
                raise ProofV3Error(
                    "derived residual projection output is incomplete"
                )
            cells = goldilocks_projection_output_cells_v3(
                projection_proof,
                projection_claims,
                claim_index=index,
            )
            values = {
                (row, column): s_value
                for row, column, s_value in cells
                if (row, column) in required
            }
            scale = (
                bits_to_scale_v3(projection.x_oracle.scale_bits)
                * bits_to_scale_v3(projection.weight_scale_bits)
            )
        if set(values) != required or not math.isfinite(scale) or scale <= 0.0:
            raise ProofV3Error(
                "residual projection cells are incomplete"
            )
        results.append((values, scale))
    return results[0], results[1]


def _check_residual_relations(
    *,
    claim: GoldilocksResidualClaimV3,
    residual_in: dict[tuple[int, int], int],
    mid_residual: dict[tuple[int, int], int],
    residual_out: dict[tuple[int, int], int],
    attention_output: dict[tuple[int, int], int],
    down_output: dict[tuple[int, int], int],
    attention_scale: float,
    down_scale: float,
) -> None:
    rin_scale = bits_to_scale_v3(claim.residual_in.oracle.scale_bits)
    mid_scale = bits_to_scale_v3(claim.mid_residual.oracle.scale_bits)
    rout_scale = bits_to_scale_v3(claim.residual_out.oracle.scale_bits)
    if (
        not math.isfinite(attention_scale)
        or attention_scale <= 0.0
        or not math.isfinite(down_scale)
        or down_scale <= 0.0
    ):
        raise ProofV3VerificationError(
            "residual projection scale is malformed"
        )
    for row in claim.selected_rows:
        for column in claim.selected_columns:
            key = (row, column)
            mid_value = mid_residual[key] * mid_scale
            composed_mid = (
                residual_in[key] * rin_scale
                + attention_output[key] * attention_scale
            )
            quant_mid = 0.5 * (
                mid_scale + rin_scale + attention_scale
            )
            bound_mid = (
                _QUANT_COEFF * quant_mid
                + _REL_COEFF
                * max(abs(mid_value), abs(composed_mid))
            )
            if abs(mid_value - composed_mid) > bound_mid:
                raise ProofV3VerificationError(
                    "attention residual composition is outside its "
                    "signed corridor"
                )
            out_value = residual_out[key] * rout_scale
            composed_out = (
                mid_residual[key] * mid_scale
                + down_output[key] * down_scale
            )
            quant_out = 0.5 * (
                rout_scale + mid_scale + down_scale
            )
            bound_out = (
                _QUANT_COEFF * quant_out
                + _REL_COEFF
                * max(abs(out_value), abs(composed_out))
            )
            if abs(out_value - composed_out) > bound_out:
                raise ProofV3VerificationError(
                    "MLP residual composition is outside its signed "
                    "corridor"
                )


def _stage_binding(
    *,
    tile_digest: bytes,
    validator_nonce: bytes,
    tag: str,
    stage: GoldilocksResidualStageClaimV3,
    witness: GoldilocksResidualStageWitnessV3,
    selected_rows: tuple[int, ...],
    selected_columns: tuple[int, ...],
    pcs_column,
    collector,
) -> GoldilocksExecutionAnchorPcsBindingProofV3 | None:
    if stage.anchor is None:
        if witness.anchor is not None:
            raise ProofV3Error(
                "derived residual stage carries an execution anchor"
            )
        col_pad = _pow2(stage.oracle.col_count)
        for row_slot, row in enumerate(selected_rows):
            for column in selected_columns:
                cell = row_slot * col_pad + column
                collector.defer(
                    tag,
                    _boolean_point(
                        cell,
                        pcs_column.pcs_statement.variable_count,
                    ),
                    witness.committed.signed_value(row, column)
                    % GOLDILOCKS_MODULUS,
                )
        return None
    if witness.anchor is None:
        raise ProofV3Error(
            "anchored residual stage lacks its execution witness"
        )
    keys = derive_goldilocks_execution_anchor_pcs_lanes_v3(
        tile_digest=tile_digest,
        validator_nonce=validator_nonce,
        tag=tag,
        anchor=stage.anchor.commitment,
        anchor_rows=stage.anchor.anchor_rows,
        pcs_column=pcs_column,
        source_column_offset=stage.anchor.source_column_offset,
        active_columns=stage.oracle.col_count,
        scale_bits=stage.oracle.scale_bits,
        encoding_id=stage.anchor.encoding_id,
        required_member_columns=selected_columns,
    )
    openings = build_goldilocks_execution_anchor_lane_openings_v3(
        commitment=stage.anchor.commitment,
        row_bytes_by_index=witness.anchor.row_bytes_by_index,
        row_tree=witness.anchor.row_tree,
        lane_keys=keys,
    )
    return prove_goldilocks_execution_anchor_pcs_binding_v3(
        tile_digest=tile_digest,
        validator_nonce=validator_nonce,
        tag=tag,
        anchor=stage.anchor.commitment,
        anchor_rows=stage.anchor.anchor_rows,
        pcs_column=pcs_column,
        source_column_offset=stage.anchor.source_column_offset,
        active_columns=stage.oracle.col_count,
        scale_bits=stage.oracle.scale_bits,
        encoding_id=stage.anchor.encoding_id,
        lane_openings=openings,
        collector=collector,
        required_member_columns=selected_columns,
    )


def prove_goldilocks_residual_composition_v3(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    witnesses,
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims,
    fused=None,
    external_collector=None,
    collector_ns: str = "",
) -> GoldilocksResidualCompositionProofV3:
    """Prove selected residual transitions against verified projections."""

    witnesses_t = tuple(witnesses)
    claims = tuple(witness.claim for witness in witnesses_t)
    projection_claims_t = tuple(projection_claims)
    projection_binding = goldilocks_projection_runtime_binding_v3(
        projection_proof,
        projection_claims_t,
    )
    tile = _tile_digest(
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
        claims_digest=_claims_digest(claims),
        projection_binding_digest=projection_binding,
    )
    ordered = []
    for index, witness in enumerate(witnesses_t):
        ordered.extend(
            (
                (
                    _tags(index)[0],
                    _selected_values(
                        witness.residual_in.committed,
                        witness.claim.selected_rows,
                        fused=fused,
                    ),
                ),
                (
                    _tags(index)[1],
                    _selected_values(
                        witness.mid_residual.committed,
                        witness.claim.selected_rows,
                        fused=fused,
                    ),
                ),
                (
                    _tags(index)[2],
                    _selected_values(
                        witness.residual_out.committed,
                        witness.claim.selected_rows,
                        fused=fused,
                    ),
                ),
            )
        )
    with pcs_coset_profile_v3("chain"):
        groups, members, _plans = (
            commit_succinct_variable_column_groups_v3(
                tile_digest=tile,
                group_tag_prefix="residual/groups",
                ordered=tuple(ordered),
                max_group_cells=MAX_RESIDUAL_GROUP_CELLS_V3,
                fused=fused,
            )
        )
    root_records = _root_records(groups)
    relation_seed = _relation_seed(
        tile_digest=tile,
        roots=root_records,
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
    for group in groups:
        collector.register_group(group)
    for member in members.values():
        collector.register_column(member.tag, member)

    captures = []
    for index, witness in enumerate(witnesses_t):
        claim = witness.claim
        (attention, attention_scale), (down, down_scale) = (
            _projection_output_values(
                projection_proof=projection_proof,
                projection_claims=projection_claims_t,
                claim=claim,
            )
        )
        tags = _tags(index)
        stages = (
            (claim.residual_in, witness.residual_in),
            (claim.mid_residual, witness.mid_residual),
            (claim.residual_out, witness.residual_out),
        )
        bindings = tuple(
            _stage_binding(
                tile_digest=tile,
                validator_nonce=validator_nonce,
                tag=tag,
                stage=stage,
                witness=stage_witness,
                selected_rows=claim.selected_rows,
                selected_columns=claim.selected_columns,
                pcs_column=members[tag],
                collector=collector,
            )
            for tag, (stage, stage_witness) in zip(
                tags,
                stages,
                strict=True,
            )
        )
        stage_cells = []
        stage_values = []
        for stage, stage_witness in stages:
            cells = _selected_cells(
                stage_witness.committed,
                claim.selected_rows,
                claim.selected_columns,
            )
            stage_cells.append(cells)
            stage_values.append(
                {
                    (row, column): value
                    for (row, column), value in zip(
                        (
                            (row, column)
                            for row in claim.selected_rows
                            for column in claim.selected_columns
                        ),
                        cells,
                        strict=True,
                    )
                }
            )
        row_squares = tuple(
            _selected_row_squares(
                stage_witness.committed,
                claim.selected_rows,
            )
            for _stage, stage_witness in stages[:2]
        )
        square_products = tuple(
            _prove_square_product(
                seed=relation_seed,
                tag=tag,
                column=members[tag],
                row_squares=squares,
                row_count=len(claim.selected_rows),
                col_count=stage.oracle.col_count,
                validator_nonce=validator_nonce,
                collector=collector,
                fused=fused,
            )
            for tag, (stage, _stage_witness), squares in zip(
                tags[:2],
                stages[:2],
                row_squares,
                strict=True,
            )
        )
        _check_residual_relations(
            claim=claim,
            residual_in=stage_values[0],
            mid_residual=stage_values[1],
            residual_out=stage_values[2],
            attention_output=attention,
            down_output=down,
            attention_scale=attention_scale,
            down_scale=down_scale,
        )
        captures.append(
            GoldilocksResidualCaptureProofV3(
                residual_in=bindings[0],
                mid_residual=bindings[1],
                residual_out=bindings[2],
                residual_in_cells=stage_cells[0],
                mid_residual_cells=stage_cells[1],
                residual_out_cells=stage_cells[2],
                residual_in_row_squares=row_squares[0],
                mid_residual_row_squares=row_squares[1],
                residual_in_square_product=square_products[0],
                mid_residual_square_product=square_products[1],
            )
        )

    if external_collector is not None:
        from verallm.proof_v3.goldilocks_succinct_batch_opening import (
            park_column_device_values_v3,
        )

        for group in groups:
            park_column_device_values_v3(group)
        batch_opening = None
    else:
        batch_opening = (
            collector.prove_all_batched(
                validator_nonce=validator_nonce,
                fused=fused,
            )
            if fused is not None
            else collector.prove_all(
                validator_nonce=validator_nonce,
            )
        )
    return GoldilocksResidualCompositionProofV3(
        groups=root_records,
        captures=tuple(captures),
        batch_opening=batch_opening,
    )


def _bound_stage_values(
    *,
    binding: GoldilocksExecutionAnchorPcsBindingProofV3 | None,
    tile_digest: bytes,
    validator_nonce: bytes,
    tag: str,
    stage: GoldilocksResidualStageClaimV3,
    selected_rows: tuple[int, ...],
    selected_columns: tuple[int, ...],
    supplied_cells: tuple[int, ...],
    pcs_column,
    checker,
) -> dict[tuple[int, int], int]:
    if stage.anchor is None:
        if binding is not None:
            raise ProofV3VerificationError(
                "derived residual stage carries an anchor binding"
            )
        expected = len(selected_rows) * len(selected_columns)
        if len(supplied_cells) != expected:
            raise ProofV3VerificationError(
                "derived residual selected-cell inventory is incomplete"
            )
        col_pad = _pow2(stage.oracle.col_count)
        result = {}
        slot = 0
        for row_slot, row in enumerate(selected_rows):
            for column in selected_columns:
                value = supplied_cells[slot]
                slot += 1
                checker.expect(
                    tag,
                    _boolean_point(
                        row_slot * col_pad + column,
                        pcs_column.pcs_statement.variable_count,
                    ),
                    value % GOLDILOCKS_MODULUS,
                )
                result[(row, column)] = value
        return result
    if binding is None:
        raise ProofV3VerificationError(
            "anchored residual stage lacks its binding"
        )
    bound = dict(
        verify_goldilocks_execution_anchor_pcs_binding_v3(
            binding,
            tile_digest=tile_digest,
            validator_nonce=validator_nonce,
            tag=tag,
            anchor=stage.anchor.commitment,
            anchor_rows=stage.anchor.anchor_rows,
            pcs_column=pcs_column,
            source_column_offset=stage.anchor.source_column_offset,
            active_columns=stage.oracle.col_count,
            scale_bits=stage.oracle.scale_bits,
            encoding_id=stage.anchor.encoding_id,
            checker=checker,
            required_member_columns=selected_columns,
        )
    )
    col_pad = _pow2(stage.oracle.col_count)
    result = {}
    for row_slot, row in enumerate(selected_rows):
        for column in selected_columns:
            try:
                result[(row, column)] = bound[
                    row_slot * col_pad + column
                ]
            except KeyError as exc:
                raise ProofV3VerificationError(
                    "residual required cell is absent from its anchor lane"
                ) from exc
    return result


def _verify_square_product(
    *,
    product: GoldilocksSuccinctProductProofV3,
    seed: bytes,
    tag: str,
    column,
    row_squares: tuple[int, ...],
    row_count: int,
    col_count: int,
    validator_nonce: bytes,
    checker,
) -> None:
    row_pad = _pow2(row_count)
    col_pad = _pow2(col_count)
    alpha = _square_alpha(seed, tag, row_pad)
    expected = sum(
        coefficient * square
        for coefficient, square in zip(
            alpha,
            row_squares + (0,) * (row_pad - len(row_squares)),
            strict=True,
        )
    ) % GOLDILOCKS_MODULUS
    verify_goldilocks_succinct_product_v3(
        product,
        statement=_square_statement(
            seed=seed,
            tag=tag,
            row_pad=row_pad,
            col_pad=col_pad,
            row_squares=row_squares,
        ),
        a_pcs_statement=column.pcs_statement,
        b_pcs_statement=column.pcs_statement,
        a_commitment=column.commitment,
        b_commitment=column.commitment,
        factor_components=(alpha, (1,) * col_pad),
        validator_nonce=validator_nonce,
        expected_sum=expected,
        checker=checker,
        a_tag=tag,
        b_tag=tag,
    )


def verify_goldilocks_residual_composition_v3(
    proof: object,
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    claims,
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims,
    batched_opening: bool = False,
    external_checker=None,
    checker_ns: str = "",
) -> None | tuple[dict[str, object], dict[str, bytes]]:
    """Verify residual arithmetic after projection composition verification."""

    try:
        if not isinstance(proof, GoldilocksResidualCompositionProofV3):
            raise ProofV3VerificationError(
                "residual composition proof has a wrong type"
            )
        claims_t = tuple(claims)
        projection_claims_t = tuple(projection_claims)
        if len(claims_t) != len(proof.captures):
            raise ProofV3VerificationError(
                "residual composition inventory is inconsistent"
            )
        projection_binding = goldilocks_projection_runtime_binding_v3(
            projection_proof,
            projection_claims_t,
        )
        tile = _tile_digest(
            validator_binding_digest=validator_binding_digest,
            validator_nonce=validator_nonce,
            claims_digest=_claims_digest(claims_t),
            projection_binding_digest=projection_binding,
        )
        with pcs_coset_profile_v3("chain"):
            groups, members, _plans = _public_columns(
                tile_digest=tile,
                sizes=_sizes(claims_t),
                root_records=proof.groups,
            )
        relation_seed = _relation_seed(
            tile_digest=tile,
            roots=proof.groups,
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
        for member in members.values():
            checker.alias(
                member.tag,
                member.group_tag,
                member.block_point,
            )
        for index, (claim, capture) in enumerate(
            zip(claims_t, proof.captures, strict=True)
        ):
            (attention, attention_scale), (down, down_scale) = (
                _projection_output_values(
                    projection_proof=projection_proof,
                    projection_claims=projection_claims_t,
                    claim=claim,
                )
            )
            tags = _tags(index)
            stages = (
                claim.residual_in,
                claim.mid_residual,
                claim.residual_out,
            )
            bindings = (
                capture.residual_in,
                capture.mid_residual,
                capture.residual_out,
            )
            proof_cells = (
                capture.residual_in_cells,
                capture.mid_residual_cells,
                capture.residual_out_cells,
            )
            stage_values = tuple(
                _bound_stage_values(
                    binding=binding,
                    tile_digest=tile,
                    validator_nonce=validator_nonce,
                    tag=tag,
                    stage=stage,
                    selected_rows=claim.selected_rows,
                    selected_columns=claim.selected_columns,
                    supplied_cells=supplied,
                    pcs_column=members[tag],
                    checker=checker,
                )
                for tag, stage, binding, supplied in zip(
                    tags,
                    stages,
                    bindings,
                    proof_cells,
                    strict=True,
                )
            )
            expected_cell_count = (
                len(claim.selected_rows)
                * len(claim.selected_columns)
            )
            for values, supplied in zip(
                stage_values,
                proof_cells,
                strict=True,
            ):
                expected = tuple(
                    values[(row, column)]
                    for row in claim.selected_rows
                    for column in claim.selected_columns
                )
                if (
                    len(supplied) != expected_cell_count
                    or supplied != expected
                ):
                    raise ProofV3VerificationError(
                        "residual selected-cell inventory is detached"
                    )
            row_square_sets = (
                capture.residual_in_row_squares,
                capture.mid_residual_row_squares,
            )
            square_products = (
                capture.residual_in_square_product,
                capture.mid_residual_square_product,
            )
            for tag, stage, squares, product in zip(
                tags[:2],
                stages[:2],
                row_square_sets,
                square_products,
                strict=True,
            ):
                if len(squares) != len(claim.selected_rows):
                    raise ProofV3VerificationError(
                        "residual row-square inventory is inconsistent"
                    )
                _verify_square_product(
                    product=product,
                    seed=relation_seed,
                    tag=tag,
                    column=members[tag],
                    row_squares=squares,
                    row_count=len(claim.selected_rows),
                    col_count=stage.oracle.col_count,
                    validator_nonce=validator_nonce,
                    checker=checker,
                )
            _check_residual_relations(
                claim=claim,
                residual_in=stage_values[0],
                mid_residual=stage_values[1],
                residual_out=stage_values[2],
                attention_output=attention,
                down_output=down,
                attention_scale=attention_scale,
                down_scale=down_scale,
            )
        statements = {
            group.tag: group.pcs_statement for group in groups
        }
        commitments = {
            group.tag: group.commitment for group in groups
        }
        if external_checker is not None:
            if proof.batch_opening is not None:
                raise ProofV3VerificationError(
                    "aggregated residual proof carries its own opening"
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
                "standalone residual proof lacks its opening"
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
            "residual composition proof is malformed"
        ) from exc
