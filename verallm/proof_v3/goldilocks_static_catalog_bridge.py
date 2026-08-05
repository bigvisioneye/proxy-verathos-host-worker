"""Bind Goldilocks folded-weight columns to the signed Pallas catalog.

Dynamic execution constraints live in the Goldilocks field.  Registered model
weights remain authenticated by the existing Pallas commitment catalog.  For
each padded input-width group this bridge:

1. derives one bounded random linear functional after every folded-weight
   Goldilocks commitment is fixed;
2. proves that functional against each Goldilocks ``Z = W*c`` column;
3. opens the same functional of the catalog-derived Pallas commitments; and
4. compares both residues through the uniquely bounded signed integer result.

No dynamic Pallas commitment is accepted from the prover.  The expected
Pallas commitment is reconstructed exclusively from validator-owned catalog
IDs and transcript-derived unsigned-31-bit output folds.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
    SuccinctEqFoldProofV3,
    _mle_eval_msb_local,
    prove_succinct_public_fold_v3,
    verify_succinct_public_fold_v3,
)
from verallm.proof_v3.lean_projection_fold import (
    LEAN_PROJECTION_FOLD_COUNT_V3,
    LeanProjectionCatalogOperationV3,
)
from zkllm.crypto.gemm_v2_reference import (
    PALLAS_SCALAR_MODULUS,
    scalar_from_bytes,
)
from zkllm.crypto.pcs_v2 import (
    ENCODING_SIGNED_I64,
    PCSOpeningV2,
    combine_commitments,
    combine_registered_catalog_u31_batch,
    prove_i64_linear_combination_linear,
    verify_linear,
)


GOLDILOCKS_STATIC_CATALOG_BRIDGE_ABI_V3: Final = (
    "projection.static_weights.pallas_catalog_to_goldilocks.v1"
)

_TRANSCRIPT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/STATIC_CATALOG_TO_GOLDILOCKS/V1"
)
_U31_MAX: Final = (1 << 31) - 1

__all__ = [
    "GOLDILOCKS_STATIC_CATALOG_BRIDGE_ABI_V3",
    "GoldilocksStaticCatalogBridgeProofV3",
    "GoldilocksStaticCatalogOperationProofV3",
    "GoldilocksStaticCatalogWidthProofV3",
    "derive_static_catalog_bridge_challenges_v3",
    "prove_goldilocks_static_catalog_bridge_v3",
    "verify_goldilocks_static_catalog_bridge_v3",
]


def _fixed32(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    return value


def _column_commitment(column) -> bytes:
    try:
        result = column.tree.commitment
    except AttributeError:
        try:
            result = column.commitment
        except AttributeError as exc:
            raise ProofV3Error(
                "static bridge Goldilocks column is malformed") from exc
    return _fixed32(result, "static bridge Goldilocks commitment")


def _coefficient_rows(
    operation: LeanProjectionCatalogOperationV3,
    rows,
) -> tuple[tuple[int, ...], ...]:
    try:
        result = tuple(tuple(int(value) for value in row) for row in rows)
    except (TypeError, ValueError) as exc:
        raise ProofV3Error(
            "static bridge output folds are malformed") from exc
    if (
        len(result) != LEAN_PROJECTION_FOLD_COUNT_V3
        or any(len(row) != operation.output_dim for row in result)
        or any(
            value < 0 or value > _U31_MAX
            for row in result
            for value in row
        )
    ):
        raise ProofV3Error("static bridge output folds are malformed")
    return result


def _folded_rows(
    operation: LeanProjectionCatalogOperationV3,
    rows,
) -> tuple[tuple[int, ...], ...]:
    try:
        result = tuple(tuple(int(value) for value in row) for row in rows)
    except (TypeError, ValueError) as exc:
        raise ProofV3Error(
            "static bridge folded weights are malformed") from exc
    if (
        len(result) != LEAN_PROJECTION_FOLD_COUNT_V3
        or any(len(row) != operation.padded_input_dim for row in result)
        or any(
            value < -(1 << 63) or value >= 1 << 63
            for row in result
            for value in row
        )
    ):
        raise ProofV3Error("static bridge folded weights are malformed")
    return result


def _operation_record(
    operation: LeanProjectionCatalogOperationV3,
    coefficient_rows: tuple[tuple[int, ...], ...],
    z_column,
) -> bytes:
    if not isinstance(operation, LeanProjectionCatalogOperationV3):
        raise ProofV3Error("static bridge operation is malformed")
    block_point = tuple(int(bit) for bit in z_column.block_point)
    if any(bit not in (0, 1) for bit in block_point):
        raise ProofV3Error("static bridge packed-column prefix is malformed")
    tag = z_column.tag.encode("ascii")
    group_tag = (z_column.group_tag or z_column.tag).encode("ascii")
    if not tag or len(tag) > 255 or not group_tag or len(group_tag) > 255:
        raise ProofV3Error("static bridge column tag is malformed")
    try:
        statement_digest = z_column.pcs_statement.digest()
    except AttributeError as exc:
        raise ProofV3Error(
            "static bridge column statement is malformed") from exc
    return (
        operation.operation_digest
        + operation.operation_root
        + operation.registered_catalog_id
        + struct.pack(
            "<III",
            operation.input_dim,
            operation.padded_input_dim,
            operation.output_dim,
        )
        + struct.pack("<B", len(tag))
        + tag
        + struct.pack("<B", len(group_tag))
        + group_tag
        + struct.pack("<H", len(block_point))
        + bytes(block_point)
        + _fixed32(statement_digest, "static bridge statement digest")
        + _column_commitment(z_column)
        + b"".join(
            struct.pack("<I", value)
            for row in coefficient_rows
            for value in row
        )
    )


def _width_groups(
    operations: tuple[LeanProjectionCatalogOperationV3, ...],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    by_width: dict[int, list[int]] = {}
    for index, operation in enumerate(operations):
        if not isinstance(operation, LeanProjectionCatalogOperationV3):
            raise ProofV3Error("static bridge operation is malformed")
        by_width.setdefault(operation.padded_input_dim, []).append(index)
    return tuple(
        (width, tuple(indices))
        for width, indices in sorted(by_width.items())
    )


def _challenge_seed(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    width: int,
    operation_indices: tuple[int, ...],
    operations: tuple[LeanProjectionCatalogOperationV3, ...],
    coefficient_rows,
    z_columns,
) -> bytes:
    material = bytearray(
        _TRANSCRIPT_DOMAIN
        + GOLDILOCKS_STATIC_CATALOG_BRIDGE_ABI_V3.encode("ascii")
        + _fixed32(
            validator_binding_digest,
            "static bridge validator binding",
        )
        + _fixed32(validator_nonce, "static bridge validator nonce")
        + struct.pack("<II", width, len(operation_indices))
    )
    for index in operation_indices:
        material.extend(struct.pack("<I", index))
        material.extend(
            _operation_record(
                operations[index],
                coefficient_rows[index],
                z_columns[index],
            )
        )
    return hashlib.sha256(bytes(material)).digest()


def _u31_vector(seed: bytes, label: bytes, count: int) -> tuple[int, ...]:
    if count <= 0:
        raise ProofV3Error("static bridge challenge count is invalid")
    stream = hashlib.shake_256(
        seed + struct.pack("<H", len(label)) + label
    ).digest(count * 4)
    return tuple(
        int.from_bytes(stream[offset : offset + 4], "little") & _U31_MAX
        for offset in range(0, len(stream), 4)
    )


def derive_static_catalog_bridge_challenges_v3(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    operations,
    coefficient_rows,
    z_columns,
) -> tuple[
    tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...], bytes],
    ...,
]:
    """Return canonical ``(width, op_indices, beta, d, seed)`` groups."""

    operations_t = tuple(operations)
    coefficients_t = tuple(
        _coefficient_rows(operation, rows)
        for operation, rows in zip(
            operations_t,
            tuple(coefficient_rows),
            strict=True,
        )
    )
    columns_t = tuple(z_columns)
    if (
        not operations_t
        or len(coefficients_t) != len(operations_t)
        or len(columns_t) != len(operations_t)
    ):
        raise ProofV3Error("static bridge inventory is inconsistent")
    result = []
    for width, indices in _width_groups(operations_t):
        seed = _challenge_seed(
            validator_binding_digest=validator_binding_digest,
            validator_nonce=validator_nonce,
            width=width,
            operation_indices=indices,
            operations=operations_t,
            coefficient_rows=coefficients_t,
            z_columns=columns_t,
        )
        beta = _u31_vector(
            seed,
            b"operation-fold-combination",
            len(indices) * LEAN_PROJECTION_FOLD_COUNT_V3,
        )
        d_values = _u31_vector(seed, b"input-functional", width)
        result.append((width, indices, beta, d_values, seed))
    return tuple(result)


def _expected_fold_commitments(
    operation: LeanProjectionCatalogOperationV3,
    coefficients: tuple[tuple[int, ...], ...],
) -> tuple[bytes, ...]:
    packed = b"".join(
        struct.pack("<I", value)
        for row in coefficients
        for value in row
    )
    from verallm.proof_v3.lean_projection_fold import (
        registered_catalog_operations_v3,
    )

    with registered_catalog_operations_v3((operation,)):
        return combine_registered_catalog_u31_batch(
            operation.registered_catalog_id,
            packed,
            term_count=operation.output_dim,
            fold_count=LEAN_PROJECTION_FOLD_COUNT_V3,
        )


def _gold_factor(
    beta: tuple[int, ...],
    d_values: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        beta_value * d_value % GOLDILOCKS_MODULUS
        for beta_value in beta
        for d_value in d_values
    )


def _gold_factor_device(
    beta: tuple[int, ...],
    d_values: tuple[int, ...],
):
    from verallm.proof_v3.native_goldilocks_backend import (
        gl_mul_t,
        to_field_tensor,
    )

    beta_device = to_field_tensor(beta, "cuda")
    d_device = to_field_tensor(d_values, "cuda")
    return gl_mul_t(
        beta_device.repeat_interleave(len(d_values)),
        d_device.repeat(len(beta)),
    )


def _structured_binding(seed: bytes, operation_index: int) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/gold-functional/"
        + seed
        + struct.pack("<I", operation_index)
    ).digest()


def _outer_digest(seed: bytes) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN + b"/pallas-functional/" + seed
    ).digest()


def _signed_result_bound(
    *,
    operation_indices: tuple[int, ...],
    operations: tuple[LeanProjectionCatalogOperationV3, ...],
    coefficients,
    beta: tuple[int, ...],
    d_values: tuple[int, ...],
) -> int:
    d_sum = sum(d_values)
    result = 0
    position = 0
    for index in operation_indices:
        operation = operations[index]
        for fold in coefficients[index]:
            folded_weight_bound = 127 * sum(fold)
            result += beta[position] * d_sum * folded_weight_bound
            position += 1
    if result >= PALLAS_SCALAR_MODULUS // 2:
        raise ProofV3Error(
            "static bridge integer functional exceeds its unique range")
    return result


def _lift_signed_pallas(value: int, bound: int) -> int:
    if value <= bound:
        return value
    if value >= PALLAS_SCALAR_MODULUS - bound:
        return value - PALLAS_SCALAR_MODULUS
    raise ProofV3VerificationError(
        "static bridge Pallas result exceeds the signed bound")


@dataclass(frozen=True, slots=True)
class GoldilocksStaticCatalogOperationProofV3:
    operation_index: int
    goldilocks_functional: SuccinctEqFoldProofV3

    def __post_init__(self) -> None:
        if (
            isinstance(self.operation_index, bool)
            or not isinstance(self.operation_index, int)
            or self.operation_index < 0
            or not isinstance(
                self.goldilocks_functional,
                SuccinctEqFoldProofV3,
            )
        ):
            raise ProofV3Error("static bridge operation proof is malformed")


@dataclass(frozen=True, slots=True)
class GoldilocksStaticCatalogWidthProofV3:
    padded_input_dim: int
    operations: tuple[GoldilocksStaticCatalogOperationProofV3, ...]
    pallas_functional: PCSOpeningV2

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        indices = tuple(item.operation_index for item in operations)
        if (
            isinstance(self.padded_input_dim, bool)
            or not isinstance(self.padded_input_dim, int)
            or self.padded_input_dim < 2
            or self.padded_input_dim & (self.padded_input_dim - 1)
            or not operations
            or indices != tuple(sorted(set(indices)))
            or not isinstance(self.pallas_functional, PCSOpeningV2)
        ):
            raise ProofV3Error("static bridge width proof is malformed")
        object.__setattr__(self, "operations", operations)


@dataclass(frozen=True, slots=True)
class GoldilocksStaticCatalogBridgeProofV3:
    groups: tuple[GoldilocksStaticCatalogWidthProofV3, ...]

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        if (
            not groups
            or not all(
                isinstance(group, GoldilocksStaticCatalogWidthProofV3)
                for group in groups
            )
            or tuple(group.padded_input_dim for group in groups)
            != tuple(sorted({group.padded_input_dim for group in groups}))
        ):
            raise ProofV3Error("static bridge proof inventory is malformed")
        object.__setattr__(self, "groups", groups)


def prove_goldilocks_static_catalog_bridge_v3(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    operations,
    coefficient_rows,
    folded_weights_i64,
    z_columns,
    collector,
    fused=None,
) -> GoldilocksStaticCatalogBridgeProofV3:
    """Prove the cross-field static-weight bridge for every width group."""

    operations_t = tuple(operations)
    coefficients_t = tuple(
        _coefficient_rows(operation, rows)
        for operation, rows in zip(
            operations_t,
            tuple(coefficient_rows),
            strict=True,
        )
    )
    folded_t = tuple(
        _folded_rows(operation, rows)
        for operation, rows in zip(
            operations_t,
            tuple(folded_weights_i64),
            strict=True,
        )
    )
    columns_t = tuple(z_columns)
    if (
        not operations_t
        or len(coefficients_t) != len(operations_t)
        or len(folded_t) != len(operations_t)
        or len(columns_t) != len(operations_t)
    ):
        raise ProofV3Error("static bridge prover inventory is inconsistent")
    challenges = derive_static_catalog_bridge_challenges_v3(
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
        operations=operations_t,
        coefficient_rows=coefficients_t,
        z_columns=columns_t,
    )
    groups = []
    for width, indices, beta, d_values, seed in challenges:
        operation_proofs = []
        raw = bytearray()
        expected_commitments = []
        beta_position = 0
        gold_sum = 0
        for index in indices:
            operation_beta = beta[
                beta_position:
                beta_position + LEAN_PROJECTION_FOLD_COUNT_V3
            ]
            beta_position += LEAN_PROJECTION_FOLD_COUNT_V3
            factor = (
                ()
                if fused is not None
                else _gold_factor(operation_beta, d_values)
            )
            factor_device = (
                _gold_factor_device(operation_beta, d_values)
                if fused is not None
                else None
            )
            binding = _structured_binding(seed, index)
            gold = prove_succinct_public_fold_v3(
                tile_digest=seed,
                column=columns_t[index],
                factor=factor,
                label=f"static-catalog/{width}/{index}",
                validator_nonce=validator_nonce,
                fused=fused,
                collector=collector,
                structured_binding=binding,
                factor_device=factor_device,
            )
            gold_sum = (
                gold_sum + gold.claimed_sum
            ) % GOLDILOCKS_MODULUS
            operation_proofs.append(
                GoldilocksStaticCatalogOperationProofV3(index, gold)
            )
            for row in folded_t[index]:
                raw.extend(struct.pack(f"<{width}q", *row))
            expected_commitments.extend(
                _expected_fold_commitments(
                    operations_t[index],
                    coefficients_t[index],
                )
            )
        opening = prove_i64_linear_combination_linear(
            bytes(raw),
            vector_length=width,
            combination_coefficients=beta,
            linear_coefficients=d_values,
            outer_digest=_outer_digest(seed),
        )
        expected_commitment = combine_commitments(
            tuple(expected_commitments),
            beta,
        )
        if opening.commitment != expected_commitment:
            raise ProofV3Error(
                "static bridge folded weights disagree with the catalog")
        bound = _signed_result_bound(
            operation_indices=indices,
            operations=operations_t,
            coefficients=coefficients_t,
            beta=beta,
            d_values=d_values,
        )
        exact = _lift_signed_pallas(
            scalar_from_bytes(opening.evaluation),
            bound,
        )
        if gold_sum != exact % GOLDILOCKS_MODULUS:
            raise ProofV3Error(
                "static bridge Goldilocks and Pallas results disagree")
        groups.append(
            GoldilocksStaticCatalogWidthProofV3(
                padded_input_dim=width,
                operations=tuple(operation_proofs),
                pallas_functional=opening,
            )
        )
    return GoldilocksStaticCatalogBridgeProofV3(tuple(groups))


def verify_goldilocks_static_catalog_bridge_v3(
    proof: object,
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    operations,
    coefficient_rows,
    z_columns,
    checker,
) -> None:
    """Verify catalog ownership and the matching Goldilocks residues."""

    try:
        if not isinstance(proof, GoldilocksStaticCatalogBridgeProofV3):
            raise ProofV3VerificationError(
                "static bridge proof has a wrong type")
        operations_t = tuple(operations)
        coefficients_t = tuple(
            _coefficient_rows(operation, rows)
            for operation, rows in zip(
                operations_t,
                tuple(coefficient_rows),
                strict=True,
            )
        )
        columns_t = tuple(z_columns)
        challenges = derive_static_catalog_bridge_challenges_v3(
            validator_binding_digest=validator_binding_digest,
            validator_nonce=validator_nonce,
            operations=operations_t,
            coefficient_rows=coefficients_t,
            z_columns=columns_t,
        )
        expected_inventory = tuple(
            (width, indices)
            for width, indices, _beta, _d, _seed in challenges
        )
        actual_inventory = tuple(
            (
                group.padded_input_dim,
                tuple(item.operation_index for item in group.operations),
            )
            for group in proof.groups
        )
        if actual_inventory != expected_inventory:
            raise ProofV3VerificationError(
                "static bridge proof inventory is not exact")
        for group, (
            width,
            indices,
            beta,
            d_values,
            seed,
        ) in zip(proof.groups, challenges, strict=True):
            expected_commitments = []
            beta_position = 0
            gold_sum = 0
            for operation_proof, index in zip(
                group.operations,
                indices,
                strict=True,
            ):
                operation_beta = beta[
                    beta_position:
                    beta_position + LEAN_PROJECTION_FOLD_COUNT_V3
                ]
                beta_position += LEAN_PROJECTION_FOLD_COUNT_V3
                binding = _structured_binding(seed, index)
                value = verify_succinct_public_fold_v3(
                    operation_proof.goldilocks_functional,
                    tile_digest=seed,
                    label=f"static-catalog/{width}/{index}",
                    pcs_statement=columns_t[index].pcs_statement,
                    commitment=_column_commitment(columns_t[index]),
                    factor=(),
                    validator_nonce=validator_nonce,
                    checker=checker,
                    tag=columns_t[index].tag,
                    factor_eval=lambda challenges_, b=operation_beta, d=(
                        d_values
                    ): (
                        _mle_eval_msb_local(b, challenges_[:2])
                        * _mle_eval_msb_local(d, challenges_[2:])
                        % GOLDILOCKS_MODULUS
                    ),
                    structured_binding=binding,
                )
                gold_sum = (gold_sum + value) % GOLDILOCKS_MODULUS
                expected_commitments.extend(
                    _expected_fold_commitments(
                        operations_t[index],
                        coefficients_t[index],
                    )
                )
            expected_commitment = combine_commitments(
                tuple(expected_commitments),
                beta,
            )
            opening = group.pallas_functional
            if (
                opening.encoding != ENCODING_SIGNED_I64
                or opening.vector_length != width
                or opening.commitment != expected_commitment
                or not verify_linear(
                    opening,
                    d_values,
                    _outer_digest(seed),
                )
            ):
                raise ProofV3VerificationError(
                    "static bridge Pallas opening is invalid")
            bound = _signed_result_bound(
                operation_indices=indices,
                operations=operations_t,
                coefficients=coefficients_t,
                beta=beta,
                d_values=d_values,
            )
            exact = _lift_signed_pallas(
                scalar_from_bytes(opening.evaluation),
                bound,
            )
            if gold_sum != exact % GOLDILOCKS_MODULUS:
                raise ProofV3VerificationError(
                    "static bridge field residues disagree")
    except ProofV3VerificationError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "static bridge proof is malformed") from exc
