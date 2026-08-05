"""Bind the authenticated layer-0 input to the real execution tokens.

The compact selected trace still needs a model-input anchor.  This module
opens nonce-selected rows from either the pre-nonce layer-0 execution anchor,
the already-frozen ``response_stamp_input`` oracle, or the selected
``l0.residual_in`` oracle.  It compares their signed-scale int8 values with
embedding rows authenticated by the signed projection manifest.  It adds no
dynamic PCS column or terminal opening.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.attention_anchor_binding import (
    extract_execution_anchor_range_v3,
)
from verallm.proof_v3.economic_artifacts import EconomicVerifiedArtifactsV3
from verallm.proof_v3.economic_challenge import (
    DECODE_CANDIDATE_ROWS_V3,
    PROMPT_CANDIDATE_ROWS_V3,
)
from verallm.proof_v3.economic_execution_anchor import (
    quantize_execution_anchor_row_v3,
)
from verallm.proof_v3.economic_commitment import (
    EconomicCommittedOracleV3,
    oracle_leaf_index_v3,
    verify_economic_oracle_opening_v3,
)
from verallm.proof_v3.economic_wire import (
    EconomicMerkleOpeningV3,
    EconomicOracleCommitmentV3,
    EconomicWeightRowRevealV3,
    VALUE_MODE_INT8,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.execution_anchor import (
    ExecutionAnchorLaneOpeningV3,
    build_execution_anchor_lane_opening_v3,
    execution_anchor_lane_bytes_v3,
)
from verallm.proof_v3.goldilocks_projection_composition import (
    GoldilocksProjectionAnchorClaimV3,
)
from verallm.proof_v3.lean_execution_anchor import (
    LEAN_KV_PROJECTION_ROWS_PER_LAYER_V3,
)
from zkllm.crypto.merkle import MerkleTree


GOLDILOCKS_BOTTOM_ANCHOR_ABI_V3: Final = (
    "terminal.embedding_to_layer0.full_anchor_rows.v1"
)
MAX_BOTTOM_ANCHOR_ROWS_V3: Final = (
    PROMPT_CANDIDATE_ROWS_V3
    + DECODE_CANDIDATE_ROWS_V3
    + LEAN_KV_PROJECTION_ROWS_PER_LAYER_V3
)

_TRANSCRIPT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_BOTTOM_ANCHOR/V1"
)

__all__ = [
    "GOLDILOCKS_BOTTOM_ANCHOR_ABI_V3",
    "GoldilocksBottomAnchorClaimV3",
    "GoldilocksBottomAnchorProofV3",
    "GoldilocksBottomAnchorWitnessV3",
    "prove_goldilocks_bottom_oracle_anchor_v3",
    "prove_goldilocks_bottom_anchor_v3",
    "verify_goldilocks_bottom_anchor_v3",
]


def _fixed32(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    return value


@dataclass(frozen=True, slots=True)
class GoldilocksBottomAnchorClaimV3:
    layer_index: int
    residual_anchor: GoldilocksProjectionAnchorClaimV3 | None
    row_map: tuple[tuple[int, int], ...]
    residual_oracle: EconomicOracleCommitmentV3 | None = None

    def __post_init__(self) -> None:
        rows = tuple(self.row_map)
        malformed_rows = any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in pair
            )
            for pair in rows
        )
        positions = (
            ()
            if malformed_rows
            else tuple(position for position, _row in rows)
        )
        anchor_rows = (
            ()
            if malformed_rows
            else tuple(row for _position, row in rows)
        )
        oracle_id = (
            None
            if not isinstance(
                self.residual_oracle,
                EconomicOracleCommitmentV3,
            )
            else self.residual_oracle.oracle_id
        )
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index != 0
            or (self.residual_anchor is None)
            == (self.residual_oracle is None)
            or (
                self.residual_anchor is not None
                and (
                    not isinstance(
                        self.residual_anchor,
                        GoldilocksProjectionAnchorClaimV3,
                    )
                    or self.residual_anchor.commitment.stage_id
                    != "l0.residual_in"
                    or self.residual_anchor.source_column_offset != 0
                    or self.residual_anchor.encoding_id
                    not in {"fp16.v1", "bf16.v1"}
                )
            )
            or (
                self.residual_oracle is not None
                and (
                    not isinstance(
                        self.residual_oracle,
                        EconomicOracleCommitmentV3,
                    )
                    or oracle_id
                    not in {"response_stamp_input", "l0.residual_in"}
                    or self.residual_oracle.phase != "global"
                    or self.residual_oracle.operation
                    != oracle_id.removeprefix("l0.")
                    or (
                        oracle_id == "l0.residual_in"
                        and self.residual_oracle.layer_index != 0
                    )
                )
            )
            or not rows
            or len(rows) > MAX_BOTTOM_ANCHOR_ROWS_V3
            or malformed_rows
            or rows != tuple(
                sorted(set(rows), key=lambda pair: pair[1])
            )
            or len(set(positions)) != len(positions)
            or len(set(anchor_rows)) != len(anchor_rows)
            or (
                self.residual_anchor is not None
                and self.residual_anchor.anchor_rows != anchor_rows
            )
            or (
                self.residual_oracle is not None
                and any(
                    row >= self.residual_oracle.row_count
                    for row in anchor_rows
                )
            )
        ):
            raise ProofV3Error("bottom-anchor claim is malformed")
        object.__setattr__(self, "row_map", rows)


@dataclass(frozen=True, slots=True)
class GoldilocksBottomAnchorWitnessV3:
    claim: GoldilocksBottomAnchorClaimV3
    row_bytes_by_index: tuple[tuple[int, bytes], ...]
    row_tree: MerkleTree
    embedding_rows: tuple[EconomicWeightRowRevealV3, ...]

    def __post_init__(self) -> None:
        rows = tuple(self.row_bytes_by_index)
        embeddings = tuple(self.embedding_rows)
        expected_rows = tuple(row for _position, row in self.claim.row_map)
        if (
            not isinstance(self.claim, GoldilocksBottomAnchorClaimV3)
            or self.claim.residual_anchor is None
            or self.claim.residual_oracle is not None
            or not isinstance(self.row_tree, MerkleTree)
            or self.row_tree.root
            != self.claim.residual_anchor.commitment.root
            or self.row_tree.num_leaves
            != self.claim.residual_anchor.commitment.row_count
            or tuple(row for row, _raw in rows) != expected_rows
            or len(embeddings) != len(rows)
            or any(
                not isinstance(raw, bytes)
                or len(raw)
                != self.claim.residual_anchor.commitment.row_width
                for _row, raw in rows
            )
            or not all(
                isinstance(item, EconomicWeightRowRevealV3)
                for item in embeddings
            )
        ):
            raise ProofV3Error("bottom-anchor witness is inconsistent")
        object.__setattr__(self, "row_bytes_by_index", rows)
        object.__setattr__(self, "embedding_rows", embeddings)


@dataclass(frozen=True, slots=True)
class GoldilocksBottomAnchorProofV3:
    binding_digest: bytes
    residual_openings: tuple[
        tuple[ExecutionAnchorLaneOpeningV3, ...],
        ...,
    ]
    embedding_rows: tuple[EconomicWeightRowRevealV3, ...]
    residual_oracle_opening: EconomicMerkleOpeningV3 | None = None

    def __post_init__(self) -> None:
        _fixed32(self.binding_digest, "bottom-anchor binding")
        openings = tuple(tuple(row) for row in self.residual_openings)
        embeddings = tuple(self.embedding_rows)
        if (
            (not openings)
            == (self.residual_oracle_opening is None)
            or (
                openings
                and len(openings) != len(embeddings)
            )
            or any(
                not row
                or not all(
                    isinstance(item, ExecutionAnchorLaneOpeningV3)
                    for item in row
                )
                for row in openings
            )
            or not all(
                isinstance(item, EconomicWeightRowRevealV3)
                for item in embeddings
            )
            or (
                self.residual_oracle_opening is not None
                and (
                    not isinstance(
                        self.residual_oracle_opening,
                        EconomicMerkleOpeningV3,
                    )
                    or not embeddings
                )
            )
        ):
            raise ProofV3Error("bottom-anchor proof is malformed")
        object.__setattr__(self, "residual_openings", openings)
        object.__setattr__(self, "embedding_rows", embeddings)


def _claim_digest(claim: GoldilocksBottomAnchorClaimV3) -> bytes:
    anchor = claim.residual_anchor
    if anchor is not None:
        record = b"\x00" + anchor.commitment.canonical_bytes()
        encoding = anchor.encoding_id.encode("ascii")
    else:
        record = b"\x01" + claim.residual_oracle.canonical_bytes()
        encoding = b"int8.v1"
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/claim/"
        + GOLDILOCKS_BOTTOM_ANCHOR_ABI_V3.encode("ascii")
        + struct.pack("<II", claim.layer_index, len(claim.row_map))
        + b"".join(
            struct.pack("<II", position, row)
            for position, row in claim.row_map
        )
        + struct.pack("<I", len(record))
        + record
        + encoding
    ).digest()


def _lane_indices(
    claim: GoldilocksBottomAnchorClaimV3,
) -> tuple[int, ...]:
    if claim.residual_anchor is None:
        raise ProofV3Error(
            "bottom-anchor oracle mode has no execution-anchor lanes"
        )
    commitment = claim.residual_anchor.commitment
    lane_bytes = execution_anchor_lane_bytes_v3(commitment.stage_id)
    return tuple(
        range((commitment.row_width + lane_bytes - 1) // lane_bytes)
    )


def _raw_row(
    *,
    claim: GoldilocksBottomAnchorClaimV3,
    row: int,
    openings: tuple[ExecutionAnchorLaneOpeningV3, ...],
) -> bytes:
    if claim.residual_anchor is None:
        raise ProofV3VerificationError(
            "bottom-anchor oracle mode has no execution-anchor row"
        )
    lanes = _lane_indices(claim)
    if tuple(
        (opening.row_index, opening.lane_index)
        for opening in openings
    ) != tuple((row, lane) for lane in lanes):
        raise ProofV3VerificationError(
            "bottom-anchor openings do not cover the exact row"
        )
    return extract_execution_anchor_range_v3(
        commitment=claim.residual_anchor.commitment,
        row_index=row,
        byte_start=0,
        byte_length=claim.residual_anchor.commitment.row_width,
        openings={
            (opening.row_index, opening.lane_index): opening
            for opening in openings
        },
    )


def _check(
    *,
    claim: GoldilocksBottomAnchorClaimV3,
    proof_openings,
    embedding_rows,
    expected_token_ids,
    artifacts: EconomicVerifiedArtifactsV3,
    residual_oracle_opening: EconomicMerkleOpeningV3 | None = None,
    capture_base_binding_digest: bytes | None = None,
) -> bytes:
    openings = tuple(tuple(row) for row in proof_openings)
    embeddings = tuple(embedding_rows)
    tokens = tuple(expected_token_ids)
    hidden_dim, vocab = artifacts.dims("embed_tokens")
    scale = artifacts.scale_for("embed_tokens")
    anchor = claim.residual_anchor
    oracle = claim.residual_oracle
    if (
        (
            anchor is not None
            and len(openings) != len(claim.row_map)
        )
        or (
            oracle is not None
            and (
                openings
                or residual_oracle_opening is None
                or oracle.col_count != hidden_dim
                or oracle.scale_bits
                != artifacts.entry("embed_tokens").scale_bits
            )
        )
        or len(embeddings) != len(claim.row_map)
        or len(tokens) != len(claim.row_map)
        or (
            anchor is not None
            and hidden_dim * 2
            != anchor.commitment.row_width
        )
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < vocab
            for token in tokens
        )
    ):
        raise ProofV3VerificationError(
            "bottom-anchor inventory is inconsistent"
        )
    material = bytearray(
        _TRANSCRIPT_DOMAIN + b"/binding/" + _claim_digest(claim)
    )
    oracle_rows = {}
    if oracle is not None:
        try:
            _fixed32(
                capture_base_binding_digest,
                "bottom response-stamp base binding",
            )
            base_binding = capture_base_binding_digest
        except ProofV3Error as exc:
            raise ProofV3VerificationError(str(exc)) from exc
        expected_cells = tuple(
            oracle_leaf_index_v3(row, column, oracle.col_count)
            for _position, row in claim.row_map
            for column in range(oracle.col_count)
        )
        values = verify_economic_oracle_opening_v3(
            oracle=oracle,
            base_binding=base_binding,
            expected_indices=expected_cells,
            opening=residual_oracle_opening,
            expected_mode=VALUE_MODE_INT8,
        )
        oracle_rows = {
            row: tuple(
                values[
                    oracle_leaf_index_v3(
                        row,
                        column,
                        oracle.col_count,
                    )
                ]
                for column in range(oracle.col_count)
            )
            for _position, row in claim.row_map
        }
    for index, (
        (_position, row),
        embedding,
        token,
    ) in enumerate(zip(
        claim.row_map,
        embeddings,
        tokens,
        strict=True,
    )):
        row_openings = openings[index] if anchor is not None else ()
        if embedding.row_index != token:
            raise ProofV3VerificationError(
                "bottom-anchor embedding row is not the execution token"
            )
        if anchor is not None:
            raw = _raw_row(
                claim=claim,
                row=row,
                openings=row_openings,
            )
            residual_i8 = tuple(
                quantize_execution_anchor_row_v3(
                    row_bytes=raw,
                    scale=scale,
                    encoding_id=anchor.encoding_id,
                )
            )
        else:
            residual_i8 = oracle_rows[row]
        embedded_i8 = artifacts.verify_weight_row(
            name="embed_tokens",
            reveal=embedding,
        )
        if residual_i8 != embedded_i8:
            raise ProofV3VerificationError(
                "bottom anchor does not match the authenticated embedding"
            )
        material.extend(struct.pack("<II", row, token))
        material.extend(bytes(value & 0xFF for value in residual_i8))
    return hashlib.sha256(bytes(material)).digest()


def prove_goldilocks_bottom_anchor_v3(
    *,
    witness: GoldilocksBottomAnchorWitnessV3,
    expected_token_ids,
    artifacts: EconomicVerifiedArtifactsV3,
) -> GoldilocksBottomAnchorProofV3:
    """Build and check the exact selected bottom-anchor rows."""

    claim = witness.claim
    lane_indices = _lane_indices(claim)
    raw_by_row = dict(witness.row_bytes_by_index)
    openings = tuple(
        tuple(
            build_execution_anchor_lane_opening_v3(
                commitment=claim.residual_anchor.commitment,
                row_index=row,
                row_bytes=raw_by_row[row],
                row_tree=witness.row_tree,
                lane_index=lane,
            )
            for lane in lane_indices
        )
        for _position, row in claim.row_map
    )
    digest = _check(
        claim=claim,
        proof_openings=openings,
        embedding_rows=witness.embedding_rows,
        expected_token_ids=expected_token_ids,
        artifacts=artifacts,
        residual_oracle_opening=None,
        capture_base_binding_digest=None,
    )
    return GoldilocksBottomAnchorProofV3(
        binding_digest=digest,
        residual_openings=openings,
        embedding_rows=witness.embedding_rows,
    )


def prove_goldilocks_bottom_oracle_anchor_v3(
    *,
    claim: GoldilocksBottomAnchorClaimV3,
    committed_residual: EconomicCommittedOracleV3,
    embedding_rows,
    expected_token_ids,
    artifacts: EconomicVerifiedArtifactsV3,
    capture_base_binding_digest: bytes,
) -> GoldilocksBottomAnchorProofV3:
    """Open authenticated int8 input rows without a second PCS column."""

    if (
        not isinstance(claim, GoldilocksBottomAnchorClaimV3)
        or claim.residual_anchor is not None
        or claim.residual_oracle is None
        or not isinstance(committed_residual, EconomicCommittedOracleV3)
        or committed_residual.commitment != claim.residual_oracle
    ):
        raise ProofV3Error(
            "bottom input-oracle witness is malformed"
        )
    rows = tuple(row for _position, row in claim.row_map)
    _indices, opening = committed_residual.open_rows(
        rows,
        value_mode=VALUE_MODE_INT8,
    )
    embeddings = tuple(embedding_rows)
    digest = _check(
        claim=claim,
        proof_openings=(),
        embedding_rows=embeddings,
        expected_token_ids=expected_token_ids,
        artifacts=artifacts,
        residual_oracle_opening=opening,
        capture_base_binding_digest=capture_base_binding_digest,
    )
    return GoldilocksBottomAnchorProofV3(
        binding_digest=digest,
        residual_openings=(),
        embedding_rows=embeddings,
        residual_oracle_opening=opening,
    )


def verify_goldilocks_bottom_anchor_v3(
    proof: object,
    *,
    claim: GoldilocksBottomAnchorClaimV3,
    expected_token_ids,
    artifacts: EconomicVerifiedArtifactsV3,
    capture_base_binding_digest: bytes | None = None,
) -> None:
    """Verify against validator-owned tokens and signed model artifacts."""

    try:
        if not isinstance(proof, GoldilocksBottomAnchorProofV3):
            raise ProofV3VerificationError(
                "bottom-anchor proof has a wrong type"
            )
        digest = _check(
            claim=claim,
            proof_openings=proof.residual_openings,
            embedding_rows=proof.embedding_rows,
            expected_token_ids=expected_token_ids,
            artifacts=artifacts,
            residual_oracle_opening=proof.residual_oracle_opening,
            capture_base_binding_digest=capture_base_binding_digest,
        )
        if proof.binding_digest != digest:
            raise ProofV3VerificationError(
                "bottom-anchor binding is inconsistent"
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
            "bottom-anchor proof is malformed"
        ) from exc
