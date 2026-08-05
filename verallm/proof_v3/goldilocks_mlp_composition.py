"""Selected SwiGLU/MLP links over authenticated projection cells.

The projection composition authenticates selected gate/up runtime outputs and
selected down-projection inputs. This module checks the signed activation
relation between those cells. It adds no dynamic column, PCS root, or opening;
both projection claims must already have been verified by the selected-trace
coordinator.
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
    goldilocks_projection_output_cells_v3,
    goldilocks_projection_runtime_binding_v3,
    goldilocks_projection_runtime_cells_v3,
)
from verallm.proof_v3.lean_projection_fold import (
    lean_projection_operation_key_v3,
)


GOLDILOCKS_MLP_COMPOSITION_ABI_V3: Final = (
    "mlp.silu_gate_up.selected_cells.projection_links.v2"
)
MLP_SILU_ACTIVATION_V3: Final = "silu.v1"
MAX_MLP_COMPOSITION_LAYERS_V3: Final = 4

_TRANSCRIPT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_MLP_COMPOSITION/V2"
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
    "GOLDILOCKS_MLP_COMPOSITION_ABI_V3",
    "MLP_SILU_ACTIVATION_V3",
    "GoldilocksMlpCompositionProofV3",
    "GoldilocksMlpLinkClaimV3",
    "prove_goldilocks_mlp_composition_v3",
    "verify_goldilocks_mlp_composition_v3",
]


def _fixed32(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    return value


@dataclass(frozen=True, slots=True)
class GoldilocksMlpLinkClaimV3:
    layer_index: int
    gate_up_projection_index: int
    down_projection_index: int
    selected_columns: tuple[int, ...]
    activation_id: str = MLP_SILU_ACTIVATION_V3

    def __post_init__(self) -> None:
        columns = tuple(self.selected_columns)
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or not 0 <= self.layer_index < 1 << 32
            or isinstance(self.gate_up_projection_index, bool)
            or not isinstance(self.gate_up_projection_index, int)
            or self.gate_up_projection_index < 0
            or isinstance(self.down_projection_index, bool)
            or not isinstance(self.down_projection_index, int)
            or self.down_projection_index < 0
            or self.gate_up_projection_index
            == self.down_projection_index
            or not columns
            or columns != tuple(sorted(set(columns)))
            or any(
                isinstance(column, bool)
                or not isinstance(column, int)
                or column < 0
                for column in columns
            )
            or self.activation_id != MLP_SILU_ACTIVATION_V3
        ):
            raise ProofV3Error("MLP link claim is malformed")
        object.__setattr__(self, "selected_columns", columns)


@dataclass(frozen=True, slots=True)
class GoldilocksMlpCompositionProofV3:
    binding_digest: bytes

    def __post_init__(self) -> None:
        _fixed32(self.binding_digest, "MLP composition binding")


def _claims_digest(
    claims: tuple[GoldilocksMlpLinkClaimV3, ...],
) -> bytes:
    if (
        not claims
        or len(claims) > MAX_MLP_COMPOSITION_LAYERS_V3
        or tuple(claim.layer_index for claim in claims)
        != tuple(sorted({claim.layer_index for claim in claims}))
    ):
        raise ProofV3Error("MLP link inventory is malformed")
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
                claim.gate_up_projection_index,
                claim.down_projection_index,
                len(claim.selected_columns),
            )
        )
        material.extend(
            b"".join(
                struct.pack("<I", column)
                for column in claim.selected_columns
            )
        )
        material.extend(claim.activation_id.encode("ascii"))
    return hashlib.sha256(bytes(material)).digest()


def _binding_digest(
    *,
    claims: tuple[GoldilocksMlpLinkClaimV3, ...],
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims: tuple[GoldilocksProjectionClaimV3, ...],
) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/binding/"
        + _claims_digest(claims)
        + goldilocks_projection_runtime_binding_v3(
            projection_proof,
            projection_claims,
        )
    ).digest()


def _silu(value: float) -> float:
    if value >= 0.0:
        return value / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return value * exponential / (1.0 + exponential)


def _check_link(
    *,
    claim: GoldilocksMlpLinkClaimV3,
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims: tuple[GoldilocksProjectionClaimV3, ...],
) -> None:
    if (
        claim.gate_up_projection_index >= len(projection_claims)
        or claim.down_projection_index >= len(projection_claims)
    ):
        raise ProofV3VerificationError(
            "MLP link references an absent projection"
        )
    gate = projection_claims[claim.gate_up_projection_index]
    down = projection_claims[claim.down_projection_index]
    gate_runtime = gate.runtime
    if (
        gate.operation.operation_key
        != lean_projection_operation_key_v3(
            layer_index=claim.layer_index,
            projection="gate_up",
        )
        or down.operation.operation_key
        != lean_projection_operation_key_v3(
            layer_index=claim.layer_index,
            projection="down",
        )
        or gate.operation.output_dim % 2
        or down.operation.input_dim != gate.operation.output_dim // 2
        or gate.selected_rows != down.selected_rows
        or (
            gate_runtime is not None
            and gate_runtime.y_anchor is not None
            and (
                gate_runtime.y_anchor.commitment.stage_id
                != f"l{claim.layer_index}.mlp_gate_up_output"
            )
        )
        or (
            down.x_anchor is not None
            and (
                down.x_anchor.commitment.stage_id
                != f"l{claim.layer_index}.mlp_down_input"
                or down.x_anchor.source_column_offset != 0
            )
        )
    ):
        raise ProofV3VerificationError(
            "MLP projection inventory is inconsistent"
        )
    intermediate = down.operation.input_dim
    if any(column >= intermediate for column in claim.selected_columns):
        raise ProofV3VerificationError(
            "MLP selected column exceeds the intermediate dimension"
        )
    required_gate = {
        column for column in claim.selected_columns
    } | {
        intermediate + column for column in claim.selected_columns
    }
    required_gate_cells = {
        (row, column)
        for row in gate.selected_rows
        for column in required_gate
    }
    if gate_runtime is not None:
        if not required_gate.issubset(gate_runtime.output_columns):
            raise ProofV3VerificationError(
                "MLP gate/up output cells are incomplete"
            )
        gate_cells = {
            (row, column): value
            for row, column, _surrogate, value in (
                goldilocks_projection_runtime_cells_v3(
                    projection_proof,
                    projection_claims,
                    claim_index=claim.gate_up_projection_index,
                )
            )
            if column in required_gate
        }
        gate_scale = bits_to_scale_v3(
            gate_runtime.y_oracle.scale_bits
        )
    else:
        if (
            not gate.weight_scale_bits
            or not required_gate_cells.issubset(
                set(gate.consumer_output_cells)
            )
        ):
            raise ProofV3VerificationError(
                "derived MLP gate/up cells are incomplete"
            )
        gate_cells = {
            (row, column): value
            for row, column, value in (
                goldilocks_projection_output_cells_v3(
                    projection_proof,
                    projection_claims,
                    claim_index=claim.gate_up_projection_index,
                )
            )
            if column in required_gate
        }
        gate_scale = (
            bits_to_scale_v3(gate.x_oracle.scale_bits)
            * bits_to_scale_v3(gate.weight_scale_bits)
        )
    down_cells = {
        (row, column): value
        for row, column, value in goldilocks_projection_input_cells_v3(
            projection_proof,
            projection_claims,
            claim_index=claim.down_projection_index,
        )
    }
    expected_down = {
        (row, column)
        for row in down.selected_rows
        for column in claim.selected_columns
    }
    if (
        set(gate_cells) != required_gate_cells
        or set(down_cells) != expected_down
    ):
        raise ProofV3VerificationError(
            "MLP selected-cell inventory is incomplete"
        )
    if (
        down.runtime is not None
        and down.runtime.input_columns
        and down.runtime.input_columns != claim.selected_columns
    ):
        raise ProofV3VerificationError(
            "MLP down-input inventory is inconsistent"
        )
    down_scale = bits_to_scale_v3(down.x_oracle.scale_bits)
    for row in gate.selected_rows:
        for column in claim.selected_columns:
            gate_value = gate_cells[(row, column)] * gate_scale
            up_value = (
                gate_cells[(row, intermediate + column)]
                * gate_scale
            )
            activated = _silu(gate_value)
            predicted = activated * up_value
            got = down_cells[(row, column)] * down_scale
            quant = (
                0.5 * down_scale
                + 0.5
                * gate_scale
                * (
                    1.1 * abs(up_value)
                    + abs(activated)
                    + 0.5 * gate_scale
                )
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
                    "MLP activation link is outside its signed "
                    "quantization corridor"
                )


def _check_all(
    *,
    claims: tuple[GoldilocksMlpLinkClaimV3, ...],
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims: tuple[GoldilocksProjectionClaimV3, ...],
) -> bytes:
    digest = _binding_digest(
        claims=claims,
        projection_proof=projection_proof,
        projection_claims=projection_claims,
    )
    for claim in claims:
        _check_link(
            claim=claim,
            projection_proof=projection_proof,
            projection_claims=projection_claims,
        )
    return digest


def prove_goldilocks_mlp_composition_v3(
    *,
    claims,
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims,
) -> GoldilocksMlpCompositionProofV3:
    """Check and bind all selected MLP links."""

    claims_t = tuple(claims)
    projection_claims_t = tuple(projection_claims)
    return GoldilocksMlpCompositionProofV3(
        binding_digest=_check_all(
            claims=claims_t,
            projection_proof=projection_proof,
            projection_claims=projection_claims_t,
        )
    )


def verify_goldilocks_mlp_composition_v3(
    proof: object,
    *,
    claims,
    projection_proof: GoldilocksProjectionCompositionProofV3,
    projection_claims,
) -> None:
    """Verify after the referenced projection composition."""

    try:
        if not isinstance(proof, GoldilocksMlpCompositionProofV3):
            raise ProofV3VerificationError(
                "MLP composition proof has a wrong type"
            )
        digest = _check_all(
            claims=tuple(claims),
            projection_proof=projection_proof,
            projection_claims=tuple(projection_claims),
        )
        if proof.binding_digest != digest:
            raise ProofV3VerificationError(
                "MLP composition binding is inconsistent"
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
            "MLP composition proof is malformed"
        ) from exc
