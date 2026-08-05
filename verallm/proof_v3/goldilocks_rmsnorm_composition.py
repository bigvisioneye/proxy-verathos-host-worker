"""Selected RMSNorm links over authenticated residual and projection cells.

The residual composition already commits each selected source row and proves
its squared int8 norm. The projection composition already authenticates the
nonce-selected input cells consumed by registered operations. This module
joins those facts with the signed RMSNorm weight, scale, gain semantics, and
epsilon. It introduces no dynamic column, PCS commitment, or terminal opening.

Both referenced compositions must be verified first. Production admission
therefore exposes this relation only through the selected-trace coordinator.
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
from verallm.proof_v3.economic_wire import bits_to_scale_v3
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_projection_composition import (
    GoldilocksProjectionClaimV3,
    GoldilocksProjectionCompositionProofV3,
    goldilocks_projection_input_cells_v3,
)
from verallm.proof_v3.goldilocks_residual_composition import (
    GoldilocksResidualClaimV3,
    GoldilocksResidualCompositionProofV3,
    goldilocks_residual_row_squares_v3,
    goldilocks_residual_runtime_binding_v3,
    goldilocks_residual_stage_cells_v3,
)
from verallm.proof_v3.lean_projection_fold import (
    lean_projection_operation_key_v3,
)
from verallm.proof_v3.rmsnorm_runtime_semantics import (
    decode_rmsnorm_runtime_semantics_v3,
)


GOLDILOCKS_RMSNORM_COMPOSITION_ABI_V3: Final = (
    "rmsnorm.selected_cells.residual_projection_links.v1"
)
MAX_RMSNORM_COMPOSITION_LINKS_V3: Final = 16

_TRANSCRIPT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_RMSNORM_COMPOSITION/V1"
)
_QUANT_COEFF: Final = (
    CORRIDOR_QUANT_COEFF_NUM_V3
    / CORRIDOR_QUANT_COEFF_DEN_V3
)
_REL_COEFF: Final = (
    CORRIDOR_REL_COEFF_NUM_V3
    / CORRIDOR_REL_COEFF_DEN_V3
)
_TARGET_STAGE_IDS: Final = {
    "qkv": "attention_qkv_input",
    "gdn_qkvz": "gdn_qkvz_input",
    "gdn_ba": "gdn_ba_input",
    "gate_up": "mlp_gate_up_input",
}

__all__ = [
    "GOLDILOCKS_RMSNORM_COMPOSITION_ABI_V3",
    "GoldilocksRmsnormArtifactV3",
    "GoldilocksRmsnormCompositionProofV3",
    "GoldilocksRmsnormLinkClaimV3",
    "GoldilocksRmsnormTargetV3",
    "prove_goldilocks_rmsnorm_composition_v3",
    "verify_goldilocks_rmsnorm_composition_v3",
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
class GoldilocksRmsnormTargetV3:
    projection_index: int
    projection_role: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.projection_index, bool)
            or not isinstance(self.projection_index, int)
            or self.projection_index < 0
            or self.projection_role not in _TARGET_STAGE_IDS
        ):
            raise ProofV3Error("RMSNorm target is malformed")


@dataclass(frozen=True, slots=True)
class GoldilocksRmsnormLinkClaimV3:
    layer_index: int
    residual_claim_index: int
    source_stage: str
    norm_key: str
    targets: tuple[GoldilocksRmsnormTargetV3, ...]

    def __post_init__(self) -> None:
        targets = tuple(self.targets)
        expected = {
            "residual_in": (
                f"l{self.layer_index}.input_norm",
                {
                    ("qkv",),
                    ("gdn_qkvz", "gdn_ba"),
                },
            ),
            "mid_residual": (
                f"l{self.layer_index}.post_norm",
                {("gate_up",)},
            ),
        }
        try:
            expected_key, role_sets = expected[self.source_stage]
        except (KeyError, TypeError) as exc:
            raise ProofV3Error(
                "RMSNorm source stage is unsupported"
            ) from exc
        if not all(
            isinstance(target, GoldilocksRmsnormTargetV3)
            for target in targets
        ):
            raise ProofV3Error("RMSNorm link claim is malformed")
        roles = tuple(target.projection_role for target in targets)
        indices = tuple(target.projection_index for target in targets)
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or not 0 <= self.layer_index < 1 << 32
            or isinstance(self.residual_claim_index, bool)
            or not isinstance(self.residual_claim_index, int)
            or self.residual_claim_index < 0
            or self.norm_key != expected_key
            or roles not in role_sets
            or len(indices) != len(set(indices))
        ):
            raise ProofV3Error("RMSNorm link claim is malformed")
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True, slots=True)
class GoldilocksRmsnormArtifactV3:
    norm_key: str
    weight_i8: tuple[int, ...]
    weight_scale_bits: int
    semantics_id: str
    epsilon_bits: int

    def __post_init__(self) -> None:
        weights = tuple(self.weight_i8)
        if (
            isinstance(self.weight_scale_bits, bool)
            or not isinstance(self.weight_scale_bits, int)
            or not 0 <= self.weight_scale_bits < 1 << 64
            or isinstance(self.epsilon_bits, bool)
            or not isinstance(self.epsilon_bits, int)
            or not 0 <= self.epsilon_bits < 1 << 64
        ):
            raise ProofV3Error("RMSNorm artifact is malformed")
        try:
            _encoded(self.norm_key, "RMSNorm artifact key")
            _encoded(self.semantics_id, "RMSNorm semantics")
            weight_scale = bits_to_scale_v3(self.weight_scale_bits)
            decode_rmsnorm_runtime_semantics_v3(
                self.semantics_id,
                self.epsilon_bits,
            )
        except (ProofV3Error, ProofV3VerificationError) as exc:
            raise ProofV3Error("RMSNorm artifact is malformed") from exc
        if (
            not weights
            or weight_scale <= 0.0
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not -128 <= value <= 127
                for value in weights
            )
        ):
            raise ProofV3Error("RMSNorm artifact is malformed")
        object.__setattr__(self, "weight_i8", weights)


@dataclass(frozen=True, slots=True)
class GoldilocksRmsnormCompositionProofV3:
    binding_digest: bytes

    def __post_init__(self) -> None:
        _fixed32(self.binding_digest, "RMSNorm composition binding")


def _claims_digest(
    claims: tuple[GoldilocksRmsnormLinkClaimV3, ...],
) -> bytes:
    keys = tuple(
        (
            claim.layer_index,
            {"residual_in": 0, "mid_residual": 1}.get(
                claim.source_stage,
                -1,
            ),
        )
        for claim in claims
    )
    if (
        not claims
        or len(claims) > MAX_RMSNORM_COMPOSITION_LINKS_V3
        or -1 in {stage for _layer, stage in keys}
        or keys != tuple(sorted(set(keys)))
    ):
        raise ProofV3Error("RMSNorm link inventory is malformed")
    material = bytearray(
        _TRANSCRIPT_DOMAIN
        + b"/claims/"
        + struct.pack("<I", len(claims))
    )
    for claim in claims:
        material.extend(
            struct.pack(
                "<II",
                claim.layer_index,
                claim.residual_claim_index,
            )
        )
        material.extend(
            _encoded(claim.source_stage, "RMSNorm source stage")
        )
        material.extend(_encoded(claim.norm_key, "RMSNorm key"))
        material.extend(struct.pack("<I", len(claim.targets)))
        for target in claim.targets:
            material.extend(struct.pack("<I", target.projection_index))
            material.extend(
                _encoded(target.projection_role, "RMSNorm target role")
            )
    return hashlib.sha256(bytes(material)).digest()


def _artifacts_digest(
    artifacts: tuple[GoldilocksRmsnormArtifactV3, ...],
    claims: tuple[GoldilocksRmsnormLinkClaimV3, ...],
) -> bytes:
    if (
        len(artifacts) != len(claims)
        or tuple(artifact.norm_key for artifact in artifacts)
        != tuple(claim.norm_key for claim in claims)
    ):
        raise ProofV3Error("RMSNorm artifact inventory is inconsistent")
    material = bytearray(
        _TRANSCRIPT_DOMAIN
        + b"/artifacts/"
        + struct.pack("<I", len(artifacts))
    )
    for artifact in artifacts:
        material.extend(_encoded(artifact.norm_key, "RMSNorm key"))
        material.extend(
            struct.pack(
                "<QQI",
                artifact.weight_scale_bits,
                artifact.epsilon_bits,
                len(artifact.weight_i8),
            )
        )
        material.extend(_encoded(artifact.semantics_id, "RMSNorm semantics"))
        material.extend(
            bytes(value & 0xFF for value in artifact.weight_i8)
        )
    return hashlib.sha256(bytes(material)).digest()


def _binding_digest(
    *,
    claims: tuple[GoldilocksRmsnormLinkClaimV3, ...],
    artifacts: tuple[GoldilocksRmsnormArtifactV3, ...],
    residual_proof: GoldilocksResidualCompositionProofV3,
    residual_claims: tuple[GoldilocksResidualClaimV3, ...],
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims: tuple[GoldilocksProjectionClaimV3, ...],
) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/binding/"
        + _claims_digest(claims)
        + _artifacts_digest(artifacts, claims)
        + goldilocks_residual_runtime_binding_v3(
            residual_proof,
            residual_claims,
            projection_proof=projection_proof,
            projection_claims=projection_claims,
        )
    ).digest()


def _target_values(
    *,
    target: GoldilocksRmsnormTargetV3,
    claim: GoldilocksRmsnormLinkClaimV3,
    residual: GoldilocksResidualClaimV3,
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims: tuple[GoldilocksProjectionClaimV3, ...],
) -> tuple[dict[tuple[int, int], int], int]:
    if target.projection_index >= len(projection_claims):
        raise ProofV3VerificationError(
            "RMSNorm target is outside the projection inventory"
        )
    projection = projection_claims[target.projection_index]
    runtime = projection.runtime
    expected_stage = (
        f"l{claim.layer_index}."
        f"{_TARGET_STAGE_IDS[target.projection_role]}"
    )
    if (
        projection.operation.operation_key
        != lean_projection_operation_key_v3(
            layer_index=claim.layer_index,
            projection=target.projection_role,
        )
        or projection.operation.input_dim
        != residual.residual_in.oracle.col_count
        or not set(projection.selected_rows).issubset(
            residual.selected_rows
        )
        or (
            projection.x_anchor is not None
            and (
                projection.x_anchor.anchor_rows
                != residual.residual_in.anchor.anchor_rows
                or projection.x_anchor.commitment.stage_id
                != expected_stage
                or projection.x_anchor.source_column_offset != 0
            )
        )
        or (
            runtime is not None
            and runtime.input_columns
            and not set(residual.selected_columns).issubset(
                runtime.input_columns
            )
        )
    ):
        raise ProofV3VerificationError(
            "RMSNorm target projection is inconsistent"
        )
    cells = goldilocks_projection_input_cells_v3(
        projection_proof,
        projection_claims,
        claim_index=target.projection_index,
    )
    values = {
        (row, column): value for row, column, value in cells
    }
    expected = {
        (row, column)
        for row in projection.selected_rows
        for column in residual.selected_columns
    }
    if not expected.issubset(values):
        raise ProofV3VerificationError(
            "RMSNorm target input cells are incomplete"
        )
    return (
        {key: values[key] for key in expected},
        projection.x_oracle.scale_bits,
    )


def _check_link(
    *,
    claim: GoldilocksRmsnormLinkClaimV3,
    artifact: GoldilocksRmsnormArtifactV3,
    residual_proof: GoldilocksResidualCompositionProofV3,
    residual_claims: tuple[GoldilocksResidualClaimV3, ...],
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims: tuple[GoldilocksProjectionClaimV3, ...],
) -> None:
    if claim.residual_claim_index >= len(residual_claims):
        raise ProofV3VerificationError(
            "RMSNorm source is outside the residual inventory"
        )
    residual = residual_claims[claim.residual_claim_index]
    source = {
        "residual_in": residual.residual_in,
        "mid_residual": residual.mid_residual,
    }[claim.source_stage]
    if (
        residual.layer_index != claim.layer_index
        or artifact.norm_key != claim.norm_key
        or len(artifact.weight_i8) != source.oracle.col_count
    ):
        raise ProofV3VerificationError(
            "RMSNorm source geometry is inconsistent"
        )
    source_cells = {
        (row, column): value
        for row, column, value in goldilocks_residual_stage_cells_v3(
            residual_proof,
            residual_claims,
            claim_index=claim.residual_claim_index,
            stage=claim.source_stage,
        )
    }
    row_squares = dict(
        goldilocks_residual_row_squares_v3(
            residual_proof,
            residual_claims,
            claim_index=claim.residual_claim_index,
            stage=claim.source_stage,
        )
    )
    expected_cells = {
        (row, column)
        for row in residual.selected_rows
        for column in residual.selected_columns
    }
    if (
        set(source_cells) != expected_cells
        or set(row_squares) != set(residual.selected_rows)
    ):
        raise ProofV3VerificationError(
            "RMSNorm source cells are incomplete"
        )
    source_scale = bits_to_scale_v3(source.oracle.scale_bits)
    weight_scale = bits_to_scale_v3(artifact.weight_scale_bits)
    gain_offset, epsilon = decode_rmsnorm_runtime_semantics_v3(
        artifact.semantics_id,
        artifact.epsilon_bits,
    )
    hidden_dim = source.oracle.col_count
    for target in claim.targets:
        target_values, target_scale_bits = _target_values(
            target=target,
            claim=claim,
            residual=residual,
            projection_proof=projection_proof,
            projection_claims=projection_claims,
        )
        target_scale = bits_to_scale_v3(target_scale_bits)
        target_rows = tuple(
            sorted({row for row, _column in target_values})
        )
        for row in target_rows:
            square = row_squares[row]
            mean_square = (
                square * source_scale * source_scale / hidden_dim
                + epsilon
            )
            if not math.isfinite(mean_square) or mean_square <= 0.0:
                raise ProofV3VerificationError(
                    "RMSNorm source norm is invalid"
                )
            rms = math.sqrt(mean_square)
            sum_abs_upper = (
                source_scale * math.sqrt(hidden_dim * square)
            )
            for column in residual.selected_columns:
                source_value = source_cells[(row, column)] * source_scale
                gain = (
                    artifact.weight_i8[column] * weight_scale
                    + gain_offset
                )
                predicted = source_value / rms * gain
                got = target_values[(row, column)] * target_scale
                quant = (
                    0.5 * target_scale
                    + (abs(gain) / rms)
                    * 0.5
                    * source_scale
                    * (
                        1.0
                        + abs(source_value)
                        * sum_abs_upper
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
                        f"RMSNorm link {claim.norm_key!r} is outside "
                        "its signed quantization corridor"
                    )


def _check_all(
    *,
    claims: tuple[GoldilocksRmsnormLinkClaimV3, ...],
    artifacts: tuple[GoldilocksRmsnormArtifactV3, ...],
    residual_proof: GoldilocksResidualCompositionProofV3,
    residual_claims: tuple[GoldilocksResidualClaimV3, ...],
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims: tuple[GoldilocksProjectionClaimV3, ...],
) -> bytes:
    digest = _binding_digest(
        claims=claims,
        artifacts=artifacts,
        residual_proof=residual_proof,
        residual_claims=residual_claims,
        projection_proof=projection_proof,
        projection_claims=projection_claims,
    )
    for claim, artifact in zip(claims, artifacts, strict=True):
        _check_link(
            claim=claim,
            artifact=artifact,
            residual_proof=residual_proof,
            residual_claims=residual_claims,
            projection_proof=projection_proof,
            projection_claims=projection_claims,
        )
    return digest


def prove_goldilocks_rmsnorm_composition_v3(
    *,
    claims,
    artifacts,
    residual_proof: GoldilocksResidualCompositionProofV3,
    residual_claims,
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims,
) -> GoldilocksRmsnormCompositionProofV3:
    """Check and bind selected RMSNorm links without another PCS proof."""

    claims_t = tuple(claims)
    artifacts_t = tuple(artifacts)
    return GoldilocksRmsnormCompositionProofV3(
        binding_digest=_check_all(
            claims=claims_t,
            artifacts=artifacts_t,
            residual_proof=residual_proof,
            residual_claims=tuple(residual_claims),
            projection_proof=projection_proof,
            projection_claims=tuple(projection_claims),
        )
    )


def verify_goldilocks_rmsnorm_composition_v3(
    proof: object,
    *,
    claims,
    artifacts,
    residual_proof: GoldilocksResidualCompositionProofV3,
    residual_claims,
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims,
) -> None:
    """Verify links after both referenced compositions were verified."""

    try:
        if not isinstance(proof, GoldilocksRmsnormCompositionProofV3):
            raise ProofV3VerificationError(
                "RMSNorm composition proof has a wrong type"
            )
        digest = _check_all(
            claims=tuple(claims),
            artifacts=tuple(artifacts),
            residual_proof=residual_proof,
            residual_claims=tuple(residual_claims),
            projection_proof=projection_proof,
            projection_claims=tuple(projection_claims),
        )
        if proof.binding_digest != digest:
            raise ProofV3VerificationError(
                "RMSNorm composition binding is inconsistent"
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
            "RMSNorm composition proof is malformed"
        ) from exc
