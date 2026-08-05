"""Native/GPU folded-weight builder for lean full-row projection checks."""

from __future__ import annotations

from typing import Final

from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.lean_projection_fold import (
    LEAN_PROJECTION_FOLD_COUNT_V3,
    LeanProjectionFoldV3,
    LeanProjectionStatementV3,
    derive_lean_projection_coefficients_v3,
)


LEAN_PROJECTION_NATIVE_FOLD_ABI_V3: Final = (
    "projection_fold.cuda_int8_u7x5_to_i64.v1"
)
_DIGIT_BITS: Final = 7
_DIGIT_COUNT: Final = 5
_INT8_GEMM_MIN_ROWS: Final = 32

__all__ = [
    "LEAN_PROJECTION_NATIVE_FOLD_ABI_V3",
    "build_lean_projection_batch_cuda_v3",
    "build_lean_projection_fold_cuda_v3",
    "build_succinct_projection_batch_cuda_v3",
]


def _build_lean_projection_fold_group_cuda_v3(
    *,
    statements,
    validator_nonce: bytes,
    input_rows_i8,
    surrogate_outputs_i64,
    weight_rows_i8,
    coefficient_rows=None,
    weight_output_columns=None,
    device: object = "cuda",
) -> tuple[LeanProjectionFoldV3, ...]:
    """Build several folds for one operation with one weight upload.

    The 31-bit coefficients are decomposed into five unsigned base-128 digits.
    Five existing exact INT8 GEMMs produce int32 partials, which are recombined
    in int64. Claims for the same operation are stacked into those five GEMMs
    so the static matrix is transferred only once.

    ``weight_rows_i8`` uses the manifest's canonical ``out_in`` orientation:
    ``[output_dim, input_dim]``. It may be a CPU or CUDA torch tensor; only the
    selected operation is resident on the proof device at a time.
    """

    statements = tuple(statements)
    input_rows = tuple(input_rows_i8)
    surrogate_outputs = tuple(surrogate_outputs_i64)
    if (
        not statements
        or len(input_rows) != len(statements)
        or len(surrogate_outputs) != len(statements)
        or not all(
            isinstance(statement, LeanProjectionStatementV3)
            for statement in statements
        )
        or any(statement != statements[0] for statement in statements[1:])
    ):
        raise ProofV3Error(
            "native lean projection fold group is malformed"
        )
    statement = statements[0]
    try:
        import torch
        from zkllm.cuda import HAS_CUDA, zkllm_native
    except (ImportError, OSError) as exc:
        raise ProofV3Error("native lean projection CUDA support is unavailable") from exc
    if (
        not HAS_CUDA
        or zkllm_native is None
        or not hasattr(zkllm_native, "cuda_int8_matmul")
        or not torch.cuda.is_available()
    ):
        raise ProofV3Error("native lean projection CUDA kernel is unavailable")
    sampled_columns = (
        None
        if weight_output_columns is None
        else tuple(int(column) for column in weight_output_columns)
    )
    expected_weight_rows = (
        statement.output_dim
        if sampled_columns is None
        else len(sampled_columns)
    )
    if (
        sampled_columns is not None
        and (
            not sampled_columns
            or sampled_columns != tuple(sorted(set(sampled_columns)))
            or sampled_columns[0] < 0
            or sampled_columns[-1] >= statement.output_dim
        )
    ):
        raise ProofV3Error(
            "native lean projection sampled columns are malformed"
        )
    if (
        not isinstance(weight_rows_i8, torch.Tensor)
        or weight_rows_i8.dtype != torch.int8
        or weight_rows_i8.ndim != 2
        or tuple(weight_rows_i8.shape)
        != (expected_weight_rows, statement.input_dim)
    ):
        raise ProofV3Error(
            "native lean projection weight matrix is not canonical out_in int8"
        )
    if coefficient_rows is None:
        coefficients = tuple(
            derive_lean_projection_coefficients_v3(
                statement=current_statement,
                validator_nonce=validator_nonce,
                input_row_i8=input_row,
                surrogate_output_i64=surrogate_output,
            )
            for current_statement, input_row, surrogate_output in zip(
                statements,
                input_rows,
                surrogate_outputs,
                strict=True,
            )
        )
    else:
        coefficients = tuple(coefficient_rows)
        if (
            len(coefficients) != len(statements)
            or any(
                len(rows) != LEAN_PROJECTION_FOLD_COUNT_V3
                or any(
                    len(row) != statement.output_dim
                    for row in rows
                )
                for rows in coefficients
            )
        ):
            raise ProofV3Error(
                "native lean projection coefficient rows are malformed"
            )

    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ProofV3Error("native lean projection folds require a CUDA device")
    device_index = (
        torch.cuda.current_device()
        if target_device.index is None
        else int(target_device.index)
    )
    target_device = torch.device("cuda", device_index)

    # The exact INT8 GEMM accepts arbitrary dimensions. Pad the output-column
    # dimension only to keep the digit buffer aligned and future tensor-core
    # substitutions possible without changing arithmetic.
    padded_output = max(
        _INT8_GEMM_MIN_ROWS,
        (expected_weight_rows + 7) & ~7,
    )
    weights = torch.zeros(
        (padded_output, statement.padded_input_dim),
        dtype=torch.int8,
        device=target_device,
    )
    weights[
        : expected_weight_rows,
        : statement.input_dim,
    ].copy_(
        weight_rows_i8.to(
            device=target_device,
            dtype=torch.int8,
            non_blocking=bool(weight_rows_i8.is_pinned())
            if weight_rows_i8.device.type == "cpu"
            else False,
        )
    )

    coefficient_rows = len(statements) * LEAN_PROJECTION_FOLD_COUNT_V3
    padded_coefficient_rows = max(
        _INT8_GEMM_MIN_ROWS,
        (coefficient_rows + 7) & ~7,
    )
    result = torch.zeros(
        (padded_coefficient_rows, statement.padded_input_dim),
        dtype=torch.int64,
        device=target_device,
    )
    try:
        import numpy as np

        coefficient_array = np.stack(coefficients).astype(
            np.int64, copy=False
        )
        if sampled_columns is not None:
            coefficient_array = coefficient_array[
                :, :, np.asarray(sampled_columns, dtype=np.int64)
            ]
        coefficient_tensor = torch.from_numpy(coefficient_array).reshape(
            coefficient_rows, expected_weight_rows
        )
    except (ImportError, TypeError, ValueError):
        coefficient_tensor = torch.tensor(
            coefficients,
            dtype=torch.int64,
        ).reshape(coefficient_rows, statement.output_dim)
        if sampled_columns is not None:
            coefficient_tensor = coefficient_tensor.index_select(
                1,
                torch.tensor(sampled_columns, dtype=torch.long),
            )
    mask = (1 << _DIGIT_BITS) - 1
    for digit_index in range(_DIGIT_COUNT):
        digits = torch.zeros(
            (padded_coefficient_rows, padded_output),
            dtype=torch.int8,
            device=target_device,
        )
        host_digits = torch.zeros(
            (padded_coefficient_rows, padded_output),
            dtype=torch.int8,
        )
        host_digits[
            :coefficient_rows,
            :expected_weight_rows,
        ] = (
            (coefficient_tensor >> (_DIGIT_BITS * digit_index)) & mask
        ).to(torch.int8)
        digits.copy_(host_digits, non_blocking=False)
        partial = zkllm_native.cuda_int8_matmul(
            digits,
            weights,
            device_index,
        )[:coefficient_rows]
        result[:coefficient_rows].add_(
            partial.to(dtype=torch.int64)
            * (1 << (_DIGIT_BITS * digit_index))
        )

    rows = result[:coefficient_rows].cpu().reshape(
        len(statements),
        LEAN_PROJECTION_FOLD_COUNT_V3,
        statement.padded_input_dim,
    ).tolist()
    return tuple(
        LeanProjectionFoldV3(
            tuple(tuple(int(value) for value in row) for row in claim_rows)
        )
        for claim_rows in rows
    )


def build_lean_projection_fold_cuda_v3(
    *,
    statement: LeanProjectionStatementV3,
    validator_nonce: bytes,
    input_row_i8,
    surrogate_output_i64,
    weight_rows_i8,
    device: object = "cuda",
) -> LeanProjectionFoldV3:
    """Build one exact fold through the operation-batched CUDA path."""

    return _build_lean_projection_fold_group_cuda_v3(
        statements=(statement,),
        validator_nonce=validator_nonce,
        input_rows_i8=(input_row_i8,),
        surrogate_outputs_i64=(surrogate_output_i64,),
        weight_rows_i8=weight_rows_i8,
        device=device,
    )[0]


def build_lean_projection_batch_cuda_v3(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    claims,
    weight_rows_i8,
    device: object = "cuda",
):
    """Build native folds and the canonical compact batch proof."""

    import os
    import time

    from verallm.proof_v3.lean_projection_batch import (
        LeanProjectionBatchClaimV3,
        _claim_materials_batched,
        build_lean_projection_batch_from_folds_v3,
    )

    claims = tuple(claims)
    weights = tuple(weight_rows_i8)
    if (
        not claims
        or len(weights) != len(claims)
        or not all(isinstance(claim, LeanProjectionBatchClaimV3)
                   for claim in claims)
    ):
        raise ProofV3Error("native lean projection batch inputs are malformed")
    trace = os.environ.get("VERATHOS_ATTN_TRACE") == "1"
    started = time.perf_counter()
    materials = _claim_materials_batched(
        claims=claims,
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
    )
    material_seconds = time.perf_counter() - started
    started = time.perf_counter()
    grouped_indices: dict[tuple[int, bytes], list[int]] = {}
    for index, (claim, weight) in enumerate(
        zip(claims, weights, strict=True)
    ):
        grouped_indices.setdefault(
            (id(weight), claim.operation.operation_digest),
            [],
        ).append(index)
    folds: list[LeanProjectionFoldV3 | None] = [None] * len(claims)
    for indices in grouped_indices.values():
        grouped = _build_lean_projection_fold_group_cuda_v3(
            statements=tuple(
                claims[index].operation.statement(
                    validator_binding_digest=validator_binding_digest
                )
                for index in indices
            ),
            validator_nonce=validator_nonce,
            input_rows_i8=tuple(
                claims[index].input_row_i8 for index in indices
            ),
            surrogate_outputs_i64=tuple(
                claims[index].surrogate_output_i64 for index in indices
            ),
            weight_rows_i8=weights[indices[0]],
            coefficient_rows=tuple(
                materials[index][0] for index in indices
            ),
            device=device,
        )
        for index, fold in zip(indices, grouped, strict=True):
            folds[index] = fold
    if any(fold is None for fold in folds):
        raise ProofV3Error("native lean projection fold batching is incomplete")
    fold_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = build_lean_projection_batch_from_folds_v3(
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
        claims=claims,
        folds=tuple(fold for fold in folds if fold is not None),
        native_aggregate=True,
        precomputed_materials=materials,
    )
    if trace:
        print(
            "[PROOF-V3-LEAN-PROJECTION] "
            f"materials={material_seconds:.3f}s "
            f"cuda_folds={fold_seconds:.3f}s "
            f"aggregate={time.perf_counter() - started:.3f}s",
            flush=True,
        )
    return result


def build_succinct_projection_batch_cuda_v3(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    witnesses,
    weight_rows_i8,
    device: object = "cuda",
):
    """Build sampled-capture succinct projection folds on CUDA."""

    from verallm.proof_v3.succinct_projection_batch import (
        SuccinctProjectionWeightRowsV3,
        SuccinctProjectionWitnessV3,
        build_succinct_projection_batch_from_folds_v3,
        derive_succinct_projection_coefficients_v3,
    )

    witnesses = tuple(witnesses)
    weights = tuple(weight_rows_i8)
    if (
        not witnesses
        or len(weights) != len(witnesses)
        or not all(
            isinstance(witness, SuccinctProjectionWitnessV3)
            for witness in witnesses
        )
        or not all(
            isinstance(weight, SuccinctProjectionWeightRowsV3)
            for weight in weights
        )
    ):
        raise ProofV3Error(
            "native succinct projection batch inputs are malformed"
        )
    import os
    import time

    trace = os.environ.get("VERATHOS_ATTN_TRACE") == "1"
    started = time.perf_counter()
    commitments = tuple(
        witness.claim.surrogate_oracle.root
        for witness in witnesses
    )
    coefficients = tuple(
        derive_succinct_projection_coefficients_v3(
            validator_binding_digest=validator_binding_digest,
            validator_nonce=validator_nonce,
            claim=witness.claim,
            surrogate_commitment=commitment,
        )
        for witness, commitment in zip(
            witnesses,
            commitments,
            strict=True,
        )
    )
    coefficient_seconds = time.perf_counter() - started
    grouped_indices: dict[tuple[int, bytes], list[int]] = {}
    for index, (witness, weight) in enumerate(
        zip(witnesses, weights, strict=True)
    ):
        if (
            weight.output_columns != witness.claim.output_columns
            or weight.input_dim != witness.claim.operation.input_dim
            or weight.output_dim != witness.claim.operation.output_dim
        ):
            weight_columns = set(weight.output_columns)
            claim_columns = set(witness.claim.output_columns)
            raise ProofV3Error(
                "native succinct projection sampled weights do not match "
                "the claim "
                f"(index={index}, "
                f"weight_dims={weight.input_dim}x{weight.output_dim}, "
                "claim_dims="
                f"{witness.claim.operation.input_dim}x"
                f"{witness.claim.operation.output_dim}, "
                f"weight_columns={len(weight.output_columns)}, "
                f"claim_columns={len(witness.claim.output_columns)}, "
                "missing="
                f"{tuple(sorted(claim_columns - weight_columns)[:8])}, "
                "extra="
                f"{tuple(sorted(weight_columns - claim_columns)[:8])})"
            )
        grouped_indices.setdefault(
            (
                id(weight.rows_i8),
                witness.claim.operation.operation_digest,
            ),
            [],
        ).append(index)
    started = time.perf_counter()
    folds: list[LeanProjectionFoldV3 | None] = [None] * len(witnesses)
    for indices in grouped_indices.values():
        representative = indices[0]
        representative_coefficients = coefficients[representative]
        if any(
            coefficients[index].shape != representative_coefficients.shape
            or not (
                coefficients[index] == representative_coefficients
            ).all()
            for index in indices[1:]
        ):
            raise ProofV3Error(
                "operation-shared projection coefficients are inconsistent"
            )
        built = _build_lean_projection_fold_group_cuda_v3(
            statements=(
                witnesses[representative].claim.operation.statement(
                    validator_binding_digest=validator_binding_digest
                ),
            ),
            validator_nonce=validator_nonce,
            input_rows_i8=(
                witnesses[representative].claim.input_row_i8,
            ),
            surrogate_outputs_i64=(
                witnesses[representative].surrogate_output_i64,
            ),
            weight_rows_i8=weights[representative].rows_i8,
            coefficient_rows=(representative_coefficients,),
            weight_output_columns=weights[representative].output_columns,
            device=device,
        )
        if len(built) != 1:
            raise ProofV3Error(
                "operation-shared projection fold build is malformed"
            )
        for index in indices:
            folds[index] = built[0]
    if any(fold is None for fold in folds):
        raise ProofV3Error(
            "native succinct projection fold batching is incomplete"
        )
    fold_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = build_succinct_projection_batch_from_folds_v3(
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
        witnesses=witnesses,
        folds=tuple(fold for fold in folds if fold is not None),
    )
    if trace:
        print(
            "[PROOF-V3-SUCCINCT-PROJECTION] "
            f"coefficients={coefficient_seconds:.3f}s "
            f"cuda_folds={fold_seconds:.3f}s "
            f"batch_proof={time.perf_counter() - started:.3f}s",
            flush=True,
        )
    return result
