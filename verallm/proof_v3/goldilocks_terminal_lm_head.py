"""Bind a shared terminal logits column to the registered LM-head catalog.

The validator remains weightless.  Four output folds are derived only after
the logits PCS root is fixed.  The prover commits ``Z_f = W^T c_f`` in
Goldilocks, while the existing Pallas catalog bridge authenticates those
folded vectors against validator-owned static commitments.  One additional
challenge folds the four identities

    <hidden, Z_f> == <logits, c_f>

into two succinct public-fold claims over the shared logits and Z columns.
Both terminal claims defer into the caller-owned global FRI collector.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Final

from verallm.proof_v3.economic_lm_head_catalog_fold import (
    EconomicLmHeadCatalogBindingV3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.goldilocks_static_catalog_bridge import (
    GoldilocksStaticCatalogBridgeProofV3,
    prove_goldilocks_static_catalog_bridge_v3,
    verify_goldilocks_static_catalog_bridge_v3,
)
from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
    SuccinctEqFoldProofV3,
    _mle_eval_msb_local,
    column_pcs_statement_v3,
    commit_succinct_column_v3,
    prove_succinct_public_fold_v3,
    verify_succinct_public_fold_v3,
)
from verallm.proof_v3.lean_projection_fold import (
    LEAN_PROJECTION_FOLD_COUNT_V3,
    LeanProjectionCatalogOperationV3,
)


GOLDILOCKS_TERMINAL_LM_HEAD_ABI_V3: Final = (
    "terminal.lm_head.pallas_catalog.goldilocks_shared_fri.v1"
)
_TRANSCRIPT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/TERMINAL_LM_HEAD/PALLAS_GOLDILOCKS/V1"
)
_OPERATION_DOMAIN: Final = _TRANSCRIPT_DOMAIN + b"/operation/"
_Z_TAG: Final = "terminal/lm_head/folded_weights"
_U31_MAX: Final = (1 << 31) - 1

__all__ = [
    "GOLDILOCKS_TERMINAL_LM_HEAD_ABI_V3",
    "GoldilocksTerminalLmHeadProofV3",
    "lm_head_catalog_projection_operation_v3",
    "prove_goldilocks_terminal_lm_head_v3",
    "verify_goldilocks_terminal_lm_head_v3",
]


def _fixed32(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    return value


def _pow2(value: int) -> int:
    return 1 << max(1, (value - 1).bit_length())


def _hidden_bytes(hidden) -> bytes:
    values = tuple(int(value) for value in hidden)
    if not values or any(value < -128 or value > 127 for value in values):
        raise ProofV3Error("terminal hidden row is not canonical signed int8")
    return bytes(value & 0xFF for value in values)


def _field_words(seed: bytes, count: int) -> tuple[int, ...]:
    result = []
    counter = 0
    while len(result) < count:
        block = hashlib.sha256(
            seed + struct.pack("<I", counter)
        ).digest()
        counter += 1
        for offset in range(0, 32, 8):
            value = int.from_bytes(block[offset : offset + 8], "little")
            if value < GOLDILOCKS_MODULUS:
                result.append(value)
                if len(result) == count:
                    break
    return tuple(result)


def _catalog_id(
    binding: EconomicLmHeadCatalogBindingV3,
) -> bytes:
    catalog_id = binding.registered_catalog_id
    if catalog_id is None:
        try:
            from zkllm.crypto.pcs_v2 import register_catalog_commitments

            catalog_id, count = register_catalog_commitments(
                binding.column_commitments
            )
        except Exception as exc:
            raise ProofV3Error(
                "LM-head Pallas catalog could not be registered"
            ) from exc
        if count != binding.vocab:
            raise ProofV3Error("LM-head Pallas catalog has a wrong size")
    return _fixed32(catalog_id, "LM-head registered catalog id")


def lm_head_catalog_projection_operation_v3(
    binding: EconomicLmHeadCatalogBindingV3,
) -> LeanProjectionCatalogOperationV3:
    """Adapt the authenticated LM-head catalog to the generic static bridge."""

    if not isinstance(binding, EconomicLmHeadCatalogBindingV3):
        raise ProofV3Error("LM-head catalog binding has a wrong type")
    try:
        from verallm.challenge.v2 import (
            MODEL_LM_HEAD_OPERATION_ID,
            MODEL_OPERATION_LAYER_IDX,
            OperationKeyV2,
        )
    except ImportError as exc:
        raise ProofV3Error("LM-head operation support is unavailable") from exc
    operation_digest = hashlib.sha256(
        _OPERATION_DOMAIN
        + binding.operation_root
        + struct.pack("<II", binding.hidden_dim, binding.vocab)
    ).digest()
    return LeanProjectionCatalogOperationV3(
        operation_key=OperationKeyV2(
            MODEL_OPERATION_LAYER_IDX,
            MODEL_LM_HEAD_OPERATION_ID,
            -1,
        ),
        operation_digest=operation_digest,
        input_dim=binding.hidden_dim,
        padded_input_dim=_pow2(binding.hidden_dim),
        output_dim=binding.vocab,
        operation_root=binding.operation_root,
        registered_catalog_id=_catalog_id(binding),
    )


def _coefficient_seed(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    operation: LeanProjectionCatalogOperationV3,
    logits_column,
    hidden_row,
) -> bytes:
    try:
        logits_statement = logits_column.pcs_statement.digest()
        logits_root = logits_column.tree.commitment
    except AttributeError as exc:
        raise ProofV3Error("terminal logits column is malformed") from exc
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/output-folds/"
        + GOLDILOCKS_TERMINAL_LM_HEAD_ABI_V3.encode("ascii")
        + _fixed32(
            validator_binding_digest,
            "terminal LM-head validator binding",
        )
        + _fixed32(validator_nonce, "terminal LM-head validator nonce")
        + operation.operation_digest
        + operation.operation_root
        + _fixed32(logits_statement, "terminal logits statement digest")
        + _fixed32(logits_root, "terminal logits commitment")
        + hashlib.sha256(_hidden_bytes(hidden_row)).digest()
    ).digest()


def _coefficients(
    seed: bytes,
    vocab: int,
) -> tuple[tuple[int, ...], ...]:
    raw = hashlib.shake_256(seed).digest(
        LEAN_PROJECTION_FOLD_COUNT_V3 * vocab * 4
    )
    try:
        import numpy as np

        words = (
            np.frombuffer(raw, dtype="<u4")
            .reshape(LEAN_PROJECTION_FOLD_COUNT_V3, vocab)
            .__and__(np.uint32(_U31_MAX))
        )
        return tuple(
            tuple(int(value) for value in row)
            for row in words
        )
    except ImportError:
        return tuple(
            tuple(
                int.from_bytes(raw[offset : offset + 4], "little")
                & _U31_MAX
                for offset in range(
                    fold * vocab * 4,
                    (fold + 1) * vocab * 4,
                    4,
                )
            )
            for fold in range(LEAN_PROJECTION_FOLD_COUNT_V3)
        )


def _folded_weights(
    *,
    operation: LeanProjectionCatalogOperationV3,
    coefficients,
    weight_rows_i8,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    fused,
):
    if fused is not None and hasattr(weight_rows_i8, "shape"):
        from verallm.proof_v3.lean_projection_native import (
            _build_lean_projection_fold_group_cuda_v3,
        )

        return _build_lean_projection_fold_group_cuda_v3(
            statements=(
                operation.statement(
                    validator_binding_digest=validator_binding_digest
                ),
            ),
            validator_nonce=validator_nonce,
            input_rows_i8=((),),
            surrogate_outputs_i64=((),),
            weight_rows_i8=weight_rows_i8,
            coefficient_rows=(coefficients,),
        )[0].folded_weights
    try:
        rows = tuple(
            tuple(int(value) for value in row)
            for row in weight_rows_i8
        )
    except (TypeError, ValueError) as exc:
        raise ProofV3Error("LM-head weight witness is malformed") from exc
    if (
        len(rows) != operation.output_dim
        or any(len(row) != operation.input_dim for row in rows)
        or any(
            value < -128 or value > 127
            for row in rows
            for value in row
        )
    ):
        raise ProofV3Error("LM-head weight witness is not canonical int8")
    return tuple(
        tuple(
            sum(
                coefficients[fold][output] * rows[output][inner]
                for output in range(operation.output_dim)
            )
            if inner < operation.input_dim
            else 0
            for inner in range(operation.padded_input_dim)
        )
        for fold in range(LEAN_PROJECTION_FOLD_COUNT_V3)
    )


def _folded_field_values(rows, *, fused):
    values = tuple(int(value) for row in rows for value in row)
    if fused is None:
        return tuple(value % GOLDILOCKS_MODULUS for value in values)
    import torch

    tensor = torch.tensor(values, dtype=torch.int64, device="cuda")
    return torch.where(
        tensor < 0,
        tensor - ((1 << 32) - 1),
        tensor,
    )


def _relation_seed(
    *,
    coefficient_seed: bytes,
    logits_root: bytes,
    z_root: bytes,
) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + b"/relation/"
        + coefficient_seed
        + _fixed32(logits_root, "terminal logits commitment")
        + _fixed32(z_root, "terminal folded-weight commitment")
    ).digest()


def _factor_binding(seed: bytes, label: bytes) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN + b"/factor/" + seed + label
    ).digest()


def _z_factor(beta, hidden, padded_input: int) -> tuple[int, ...]:
    padded_hidden = tuple(
        int(value) % GOLDILOCKS_MODULUS for value in hidden
    ) + (0,) * (padded_input - len(hidden))
    return tuple(
        beta_value * hidden_value % GOLDILOCKS_MODULUS
        for beta_value in beta
        for hidden_value in padded_hidden
    )


def _logits_factor(beta, coefficients, padded_vocab: int) -> tuple[int, ...]:
    return tuple(
        sum(
            beta[fold] * coefficients[fold][output]
            for fold in range(LEAN_PROJECTION_FOLD_COUNT_V3)
        )
        % GOLDILOCKS_MODULUS
        if output < len(coefficients[0])
        else 0
        for output in range(padded_vocab)
    )


@dataclass(frozen=True, slots=True)
class GoldilocksTerminalLmHeadProofV3:
    logits_commitment: bytes
    z_commitment: bytes
    static_bridge: GoldilocksStaticCatalogBridgeProofV3
    z_relation: SuccinctEqFoldProofV3
    logits_relation: SuccinctEqFoldProofV3

    def __post_init__(self) -> None:
        _fixed32(self.logits_commitment, "terminal logits commitment")
        _fixed32(self.z_commitment, "terminal folded-weight commitment")
        if (
            not isinstance(
                self.static_bridge,
                GoldilocksStaticCatalogBridgeProofV3,
            )
            or not isinstance(self.z_relation, SuccinctEqFoldProofV3)
            or not isinstance(self.logits_relation, SuccinctEqFoldProofV3)
        ):
            raise ProofV3Error("terminal LM-head proof is malformed")


def prove_goldilocks_terminal_lm_head_v3(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    binding: EconomicLmHeadCatalogBindingV3,
    hidden_row_i8,
    logits_column,
    weight_rows_i8,
    collector,
    fused=None,
) -> GoldilocksTerminalLmHeadProofV3:
    """Bind one already-committed logits column to the signed LM-head."""

    hidden = tuple(int(value) for value in hidden_row_i8)
    _hidden_bytes(hidden)
    operation = lm_head_catalog_projection_operation_v3(binding)
    if len(hidden) != operation.input_dim or collector is None:
        raise ProofV3Error("terminal LM-head prover geometry is malformed")
    try:
        logits_root = logits_column.tree.commitment
        logits_cells = 1 << logits_column.pcs_statement.variable_count
    except AttributeError as exc:
        raise ProofV3Error("terminal logits column is malformed") from exc
    if logits_cells != _pow2(operation.output_dim):
        raise ProofV3Error(
            "terminal logits column does not match the LM-head vocabulary"
        )
    coefficient_seed = _coefficient_seed(
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
        operation=operation,
        logits_column=logits_column,
        hidden_row=hidden,
    )
    coefficients = _coefficients(coefficient_seed, operation.output_dim)
    folds = _folded_weights(
        operation=operation,
        coefficients=coefficients,
        weight_rows_i8=weight_rows_i8,
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
        fused=fused,
    )
    z_tile = hashlib.sha256(
        _TRANSCRIPT_DOMAIN + b"/z/" + coefficient_seed
    ).digest()
    z_column = commit_succinct_column_v3(
        tile_digest=z_tile,
        tag=_Z_TAG,
        values=_folded_field_values(folds, fused=fused),
        fused=fused,
        canonical_input=True,
    )
    collector.register_column(_Z_TAG, z_column)
    static_bridge = prove_goldilocks_static_catalog_bridge_v3(
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
        operations=(operation,),
        coefficient_rows=(coefficients,),
        folded_weights_i64=(folds,),
        z_columns=(z_column,),
        collector=collector,
        fused=fused,
    )
    relation_seed = _relation_seed(
        coefficient_seed=coefficient_seed,
        logits_root=logits_root,
        z_root=z_column.tree.commitment,
    )
    beta = _field_words(
        relation_seed + b"/fold-combination/",
        LEAN_PROJECTION_FOLD_COUNT_V3,
    )
    z_factor = _z_factor(beta, hidden, operation.padded_input_dim)
    logits_factor = _logits_factor(
        beta,
        coefficients,
        logits_cells,
    )
    if fused is not None:
        from verallm.proof_v3.native_goldilocks_backend import (
            to_field_tensor,
        )

        # Earlier sub-arguments may release a committed column's optional
        # device mirror. Rehydrate the already-committed logits evaluations
        # from their canonical host tuple when needed; this does not recommit
        # or alter the shared root.
        if logits_column.device_values is None:
            if logits_column.values is None:
                raise ProofV3Error(
                    "terminal logits column values are unavailable"
                )
            object.__setattr__(
                logits_column,
                "device_values",
                to_field_tensor(logits_column.values, "cuda"),
            )
        z_factor_device = to_field_tensor(z_factor, "cuda")
        logits_factor_device = to_field_tensor(logits_factor, "cuda")
    else:
        z_factor_device = None
        logits_factor_device = None
    z_relation = prove_succinct_public_fold_v3(
        tile_digest=relation_seed,
        column=z_column,
        factor=z_factor,
        label="terminal-lm-head-z",
        validator_nonce=validator_nonce,
        fused=fused,
        collector=collector,
        structured_binding=_factor_binding(relation_seed, b"z"),
        factor_device=z_factor_device,
    )
    logits_relation = prove_succinct_public_fold_v3(
        tile_digest=relation_seed,
        column=logits_column,
        factor=logits_factor,
        label="terminal-lm-head-logits",
        validator_nonce=validator_nonce,
        fused=fused,
        collector=collector,
        structured_binding=_factor_binding(relation_seed, b"logits"),
        factor_device=logits_factor_device,
    )
    if (
        z_relation.claimed_sum - logits_relation.claimed_sum
    ) % GOLDILOCKS_MODULUS:
        raise ProofV3Error(
            "terminal logits disagree with hidden @ registered LM-head"
        )
    return GoldilocksTerminalLmHeadProofV3(
        logits_commitment=logits_root,
        z_commitment=z_column.tree.commitment,
        static_bridge=static_bridge,
        z_relation=z_relation,
        logits_relation=logits_relation,
    )


def _z_public_column(
    *,
    z_tile: bytes,
    commitment: bytes,
    operation: LeanProjectionCatalogOperationV3,
):
    statement = column_pcs_statement_v3(
        z_tile,
        _Z_TAG,
        (
            LEAN_PROJECTION_FOLD_COUNT_V3
            * operation.padded_input_dim
        ).bit_length()
        - 1,
    )
    return SimpleNamespace(
        tag=_Z_TAG,
        pcs_statement=statement,
        tree=SimpleNamespace(commitment=commitment),
        group_tag=None,
        block_point=(),
    )


def verify_goldilocks_terminal_lm_head_v3(
    proof: object,
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    binding: EconomicLmHeadCatalogBindingV3,
    hidden_row_i8,
    logits_pcs_statement,
    expected_logits_commitment: bytes,
    checker,
) -> tuple[dict[str, object], dict[str, bytes]]:
    """Verify LM-head ownership/relation and collect global opening claims."""

    try:
        if not isinstance(proof, GoldilocksTerminalLmHeadProofV3):
            raise ProofV3VerificationError(
                "terminal LM-head proof has a wrong type"
            )
        hidden = tuple(int(value) for value in hidden_row_i8)
        _hidden_bytes(hidden)
        operation = lm_head_catalog_projection_operation_v3(binding)
        if (
            len(hidden) != operation.input_dim
            or proof.logits_commitment
            != _fixed32(
                expected_logits_commitment,
                "expected terminal logits commitment",
            )
            or checker is None
            or getattr(logits_pcs_statement, "variable_count", -1)
            != _pow2(operation.output_dim).bit_length() - 1
        ):
            raise ProofV3VerificationError(
                "terminal LM-head proof disagrees with signed geometry"
            )
        logits_column = SimpleNamespace(
            tag="terminal/argmax/logits",
            pcs_statement=logits_pcs_statement,
            tree=SimpleNamespace(commitment=proof.logits_commitment),
            group_tag=None,
            block_point=(),
        )
        coefficient_seed = _coefficient_seed(
            validator_binding_digest=validator_binding_digest,
            validator_nonce=validator_nonce,
            operation=operation,
            logits_column=logits_column,
            hidden_row=hidden,
        )
        coefficients = _coefficients(
            coefficient_seed,
            operation.output_dim,
        )
        z_tile = hashlib.sha256(
            _TRANSCRIPT_DOMAIN + b"/z/" + coefficient_seed
        ).digest()
        z_column = _z_public_column(
            z_tile=z_tile,
            commitment=proof.z_commitment,
            operation=operation,
        )
        verify_goldilocks_static_catalog_bridge_v3(
            proof.static_bridge,
            validator_binding_digest=validator_binding_digest,
            validator_nonce=validator_nonce,
            operations=(operation,),
            coefficient_rows=(coefficients,),
            z_columns=(z_column,),
            checker=checker,
        )
        relation_seed = _relation_seed(
            coefficient_seed=coefficient_seed,
            logits_root=proof.logits_commitment,
            z_root=proof.z_commitment,
        )
        beta = _field_words(
            relation_seed + b"/fold-combination/",
            LEAN_PROJECTION_FOLD_COUNT_V3,
        )
        padded_hidden = tuple(
            value % GOLDILOCKS_MODULUS for value in hidden
        ) + (0,) * (operation.padded_input_dim - len(hidden))
        z_value = verify_succinct_public_fold_v3(
            proof.z_relation,
            tile_digest=relation_seed,
            label="terminal-lm-head-z",
            pcs_statement=z_column.pcs_statement,
            commitment=proof.z_commitment,
            factor=(),
            validator_nonce=validator_nonce,
            checker=checker,
            tag=_Z_TAG,
            factor_eval=lambda point: (
                _mle_eval_msb_local(beta, point[:2])
                * _mle_eval_msb_local(padded_hidden, point[2:])
                % GOLDILOCKS_MODULUS
            ),
            structured_binding=_factor_binding(relation_seed, b"z"),
        )
        logits_value = verify_succinct_public_fold_v3(
            proof.logits_relation,
            tile_digest=relation_seed,
            label="terminal-lm-head-logits",
            pcs_statement=logits_pcs_statement,
            commitment=proof.logits_commitment,
            factor=(),
            validator_nonce=validator_nonce,
            checker=checker,
            tag=logits_column.tag,
            factor_eval=lambda point: sum(
                beta[fold]
                * _mle_eval_msb_local(
                    coefficients[fold]
                    + (0,) * (
                        _pow2(operation.output_dim)
                        - operation.output_dim
                    ),
                    point,
                )
                for fold in range(LEAN_PROJECTION_FOLD_COUNT_V3)
            )
            % GOLDILOCKS_MODULUS,
            structured_binding=_factor_binding(
                relation_seed,
                b"logits",
            ),
        )
        if z_value != logits_value:
            raise ProofV3VerificationError(
                "terminal logits are detached from hidden @ registered LM-head"
            )
        return (
            {
                _Z_TAG: z_column.pcs_statement,
                logits_column.tag: logits_pcs_statement,
            },
            {
                _Z_TAG: proof.z_commitment,
                logits_column.tag: proof.logits_commitment,
            },
        )
    except ProofV3VerificationError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "terminal LM-head proof is malformed"
        ) from exc
