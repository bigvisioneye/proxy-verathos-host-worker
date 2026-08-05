"""Bind the final residual row to the LM-head's captured final hidden row.

The last residual is only one hidden-width row, so opening its complete
pre-nonce execution-anchor row is smaller and simpler than committing another
RMSNorm trace. The verifier quantizes that authenticated runtime row under the
signed residual scale, applies the signed final RMSNorm, and checks every cell
against the final-hidden row already authenticated by the terminal path.

The terminal path must be verified first. Production admission exposes both
relations only through the selected-trace coordinator.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Final

import numpy as np

from verallm.proof_v3.attention_anchor_binding import (
    extract_execution_anchor_range_v3,
)
from verallm.proof_v3.economic_challenge import (
    CORRIDOR_QUANT_COEFF_DEN_V3,
    CORRIDOR_QUANT_COEFF_NUM_V3,
    CORRIDOR_REL_COEFF_DEN_V3,
    CORRIDOR_REL_COEFF_NUM_V3,
)
from verallm.proof_v3.economic_execution_anchor import (
    quantize_execution_anchor_row_v3,
)
from verallm.proof_v3.economic_wire import (
    EconomicOracleCommitmentV3,
    bits_to_scale_v3,
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
from verallm.proof_v3.goldilocks_rmsnorm_composition import (
    GoldilocksRmsnormArtifactV3,
)
from verallm.proof_v3.rmsnorm_runtime_semantics import (
    decode_rmsnorm_runtime_semantics_v3,
)
from zkllm.crypto.merkle import MerkleTree


GOLDILOCKS_FINAL_RMSNORM_ABI_V3: Final = (
    "terminal.final_rmsnorm.full_anchor_row.v1"
)

_TRANSCRIPT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_FINAL_RMSNORM/V1"
)
_QUANT_COEFF: Final = (
    CORRIDOR_QUANT_COEFF_NUM_V3
    / CORRIDOR_QUANT_COEFF_DEN_V3
)
_REL_COEFF: Final = (
    CORRIDOR_REL_COEFF_NUM_V3
    / CORRIDOR_REL_COEFF_DEN_V3
)

__all__ = [
    "GOLDILOCKS_FINAL_RMSNORM_ABI_V3",
    "GoldilocksFinalRmsnormClaimV3",
    "GoldilocksFinalRmsnormProofV3",
    "GoldilocksFinalRmsnormWitnessV3",
    "prove_goldilocks_final_rmsnorm_v3",
    "verify_goldilocks_final_rmsnorm_v3",
]


def _fixed32(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    return value


def _encoded(value: object, name: str) -> bytes:
    try:
        encoded = value.encode("ascii")
    except (AttributeError, UnicodeEncodeError) as exc:
        raise ProofV3Error(f"{name} is malformed") from exc
    if not encoded or len(encoded) > 255:
        raise ProofV3Error(f"{name} is malformed")
    return struct.pack("<B", len(encoded)) + encoded


@dataclass(frozen=True, slots=True)
class GoldilocksFinalRmsnormClaimV3:
    layer_index: int
    residual_anchor: GoldilocksProjectionAnchorClaimV3
    residual_scale_bits: int
    final_hidden_oracle: EconomicOracleCommitmentV3
    final_hidden_row: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or not 0 <= self.layer_index < 1 << 32
            or not isinstance(
                self.residual_anchor,
                GoldilocksProjectionAnchorClaimV3,
            )
            or self.residual_anchor.commitment.stage_id
            != f"l{self.layer_index}.residual_out"
            or len(self.residual_anchor.anchor_rows) != 1
            or self.residual_anchor.source_column_offset != 0
            or self.residual_anchor.encoding_id
            not in {"fp16.v1", "bf16.v1"}
            or not isinstance(
                self.final_hidden_oracle,
                EconomicOracleCommitmentV3,
            )
            or self.final_hidden_oracle.oracle_id != "final_hidden"
            or self.final_hidden_oracle.operation != "final_hidden"
            or self.final_hidden_oracle.col_count * 2
            != self.residual_anchor.commitment.row_width
            or isinstance(self.final_hidden_row, bool)
            or not isinstance(self.final_hidden_row, int)
            or not 0 <= self.final_hidden_row
            < self.final_hidden_oracle.row_count
            or isinstance(self.residual_scale_bits, bool)
            or not isinstance(self.residual_scale_bits, int)
            or not 0 <= self.residual_scale_bits < 1 << 64
        ):
            raise ProofV3Error("final RMSNorm claim is malformed")
        try:
            scale = bits_to_scale_v3(self.residual_scale_bits)
        except ProofV3Error as exc:
            raise ProofV3Error("final residual scale is malformed") from exc
        if scale <= 0.0:
            raise ProofV3Error("final residual scale is malformed")


@dataclass(frozen=True, slots=True)
class GoldilocksFinalRmsnormWitnessV3:
    claim: GoldilocksFinalRmsnormClaimV3
    residual_row_bytes: bytes
    residual_row_tree: MerkleTree

    def __post_init__(self) -> None:
        if (
            not isinstance(self.claim, GoldilocksFinalRmsnormClaimV3)
            or not isinstance(self.residual_row_tree, MerkleTree)
            or not isinstance(self.residual_row_bytes, bytes)
            or len(self.residual_row_bytes)
            != self.claim.residual_anchor.commitment.row_width
            or self.residual_row_tree.root
            != self.claim.residual_anchor.commitment.root
            or self.residual_row_tree.num_leaves
            != self.claim.residual_anchor.commitment.row_count
        ):
            raise ProofV3Error("final RMSNorm witness is inconsistent")


@dataclass(frozen=True, slots=True)
class GoldilocksFinalRmsnormProofV3:
    binding_digest: bytes
    residual_openings: tuple[ExecutionAnchorLaneOpeningV3, ...]

    def __post_init__(self) -> None:
        _fixed32(
            self.binding_digest,
            "final RMSNorm binding",
        )
        openings = tuple(self.residual_openings)
        if (
            not openings
            or not all(
                isinstance(item, ExecutionAnchorLaneOpeningV3)
                for item in openings
            )
        ):
            raise ProofV3Error("final RMSNorm proof is malformed")
        object.__setattr__(self, "residual_openings", openings)


def _claim_digest(claim: GoldilocksFinalRmsnormClaimV3) -> bytes:
    anchor = claim.residual_anchor
    commitment = anchor.commitment.canonical_bytes()
    oracle = claim.final_hidden_oracle
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/claim/"
        + struct.pack(
            "<IIIQQ",
            claim.layer_index,
            anchor.anchor_rows[0],
            claim.final_hidden_row,
            claim.residual_scale_bits,
            oracle.scale_bits,
        )
        + struct.pack("<I", len(commitment))
        + commitment
        + _encoded(anchor.encoding_id, "final residual encoding")
        + struct.pack(
            "<III",
            oracle.row_count,
            oracle.col_count,
            oracle.layer_index,
        )
        + oracle.root
    ).digest()


def _artifact_digest(artifact: GoldilocksRmsnormArtifactV3) -> bytes:
    if artifact.norm_key != "final_norm":
        raise ProofV3Error("final RMSNorm artifact has a wrong key")
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/artifact/"
        + _encoded(artifact.norm_key, "final RMSNorm key")
        + _encoded(artifact.semantics_id, "final RMSNorm semantics")
        + struct.pack(
            "<QQI",
            artifact.weight_scale_bits,
            artifact.epsilon_bits,
            len(artifact.weight_i8),
        )
        + bytes(value & 0xFF for value in artifact.weight_i8)
    ).digest()


def _binding_digest(
    *,
    claim: GoldilocksFinalRmsnormClaimV3,
    artifact: GoldilocksRmsnormArtifactV3,
    final_hidden_i8: tuple[int, ...],
) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/binding/"
        + _claim_digest(claim)
        + _artifact_digest(artifact)
        + bytes(value & 0xFF for value in final_hidden_i8)
    ).digest()


def _lane_indices(
    claim: GoldilocksFinalRmsnormClaimV3,
) -> tuple[int, ...]:
    commitment = claim.residual_anchor.commitment
    lane_bytes = execution_anchor_lane_bytes_v3(commitment.stage_id)
    return tuple(
        range(
            (commitment.row_width + lane_bytes - 1)
            // lane_bytes
        )
    )


def _raw_residual_row(
    *,
    claim: GoldilocksFinalRmsnormClaimV3,
    openings: tuple[ExecutionAnchorLaneOpeningV3, ...],
) -> bytes:
    row = claim.residual_anchor.anchor_rows[0]
    lanes = _lane_indices(claim)
    if tuple(
        (opening.row_index, opening.lane_index)
        for opening in openings
    ) != tuple((row, lane) for lane in lanes):
        raise ProofV3VerificationError(
            "final residual openings do not cover the exact row"
        )
    opening_map = {
        (opening.row_index, opening.lane_index): opening
        for opening in openings
    }
    return extract_execution_anchor_range_v3(
        commitment=claim.residual_anchor.commitment,
        row_index=row,
        byte_start=0,
        byte_length=claim.residual_anchor.commitment.row_width,
        openings=opening_map,
    )


def _check_relation(
    *,
    claim: GoldilocksFinalRmsnormClaimV3,
    artifact: GoldilocksRmsnormArtifactV3,
    final_hidden_i8: tuple[int, ...],
    openings: tuple[ExecutionAnchorLaneOpeningV3, ...],
) -> bytes:
    hidden_dim = claim.final_hidden_oracle.col_count
    if (
        len(final_hidden_i8) != hidden_dim
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not -128 <= value <= 127
            for value in final_hidden_i8
        )
        or len(artifact.weight_i8) != hidden_dim
    ):
        raise ProofV3VerificationError(
            "final RMSNorm row geometry is inconsistent"
        )
    raw = _raw_residual_row(claim=claim, openings=openings)
    source_i8 = quantize_execution_anchor_row_v3(
        row_bytes=raw,
        scale=bits_to_scale_v3(claim.residual_scale_bits),
        encoding_id=claim.residual_anchor.encoding_id,
    )
    if len(source_i8) != hidden_dim:
        raise ProofV3VerificationError(
            "final residual quantization has a wrong width"
        )
    source_scale = bits_to_scale_v3(claim.residual_scale_bits)
    hidden_scale = bits_to_scale_v3(
        claim.final_hidden_oracle.scale_bits
    )
    weight_scale = bits_to_scale_v3(artifact.weight_scale_bits)
    gain_offset, epsilon = decode_rmsnorm_runtime_semantics_v3(
        artifact.semantics_id,
        artifact.epsilon_bits,
    )
    source = np.asarray(source_i8, dtype=np.float64) * source_scale
    mean_square = float(np.dot(source, source) / hidden_dim + epsilon)
    if not math.isfinite(mean_square) or mean_square <= 0.0:
        raise ProofV3VerificationError(
            "final residual norm is invalid"
        )
    rms = math.sqrt(mean_square)
    sum_abs = float(np.abs(source).sum())
    for column in range(hidden_dim):
        source_value = float(source[column])
        gain = (
            artifact.weight_i8[column] * weight_scale
            + gain_offset
        )
        predicted = source_value / rms * gain
        got = final_hidden_i8[column] * hidden_scale
        quant = (
            0.5 * hidden_scale
            + (abs(gain) / rms)
            * 0.5
            * source_scale
            * (
                1.0
                + abs(source_value)
                * sum_abs
                / (hidden_dim * mean_square)
            )
            + abs(source_value / rms)
            * 0.5
            * weight_scale
        )
        bound = (
            _QUANT_COEFF * quant
            + _REL_COEFF * abs(predicted)
        )
        if (
            not math.isfinite(predicted)
            or not math.isfinite(got)
            or not math.isfinite(bound)
            or abs(got - predicted) > bound
        ):
            raise ProofV3VerificationError(
                "final hidden is outside the signed RMSNorm corridor"
            )
    return _binding_digest(
        claim=claim,
        artifact=artifact,
        final_hidden_i8=final_hidden_i8,
    )


def prove_goldilocks_final_rmsnorm_v3(
    *,
    witness: GoldilocksFinalRmsnormWitnessV3,
    artifact: GoldilocksRmsnormArtifactV3,
    final_hidden_i8,
) -> GoldilocksFinalRmsnormProofV3:
    """Open the complete final residual row and check the top norm link."""

    hidden = tuple(int(value) for value in final_hidden_i8)
    claim = witness.claim
    row = claim.residual_anchor.anchor_rows[0]
    openings = tuple(
        build_execution_anchor_lane_opening_v3(
            commitment=claim.residual_anchor.commitment,
            row_index=row,
            row_bytes=witness.residual_row_bytes,
            row_tree=witness.residual_row_tree,
            lane_index=lane,
        )
        for lane in _lane_indices(claim)
    )
    return GoldilocksFinalRmsnormProofV3(
        binding_digest=_check_relation(
            claim=claim,
            artifact=artifact,
            final_hidden_i8=hidden,
            openings=openings,
        ),
        residual_openings=openings,
    )


def verify_goldilocks_final_rmsnorm_v3(
    proof: object,
    *,
    claim: GoldilocksFinalRmsnormClaimV3,
    artifact: GoldilocksRmsnormArtifactV3,
    final_hidden_i8,
) -> None:
    """Verify after the terminal path authenticates ``final_hidden_i8``."""

    try:
        if not isinstance(proof, GoldilocksFinalRmsnormProofV3):
            raise ProofV3VerificationError(
                "final RMSNorm proof has a wrong type"
            )
        hidden = tuple(int(value) for value in final_hidden_i8)
        digest = _check_relation(
            claim=claim,
            artifact=artifact,
            final_hidden_i8=hidden,
            openings=proof.residual_openings,
        )
        if proof.binding_digest != digest:
            raise ProofV3VerificationError(
                "final RMSNorm binding is inconsistent"
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
            "final RMSNorm proof is malformed"
        ) from exc
