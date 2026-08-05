"""PRODUCTION-wire RMSNorm tile: succinct prove + O(q log N) verify.

Reference semantics (goldilocks_rmsnorm_tile_reference) per row:
    sum_sq = sum x^2;  mean_sq, mean_rem = divmod(sum_sq, dim)
    inv = rsqrt_table[mean_sq]
    per cell: 2^16 * y + y_rem == inv * x * w + 2^15,  y_rem in [0, 2^16)

Succinct decomposition (columns x, x_biased, y, y_rem_lo, y_rem_hi and
the PUBLIC weight column w committed with a verifier-recomputable root):
  * sum_sq: succinct product P(x,x; ones); mean/inv are scalar-checked
    by the verifier against ITS OWN rsqrt table (no lookup argument
    needed for a single row scalar);
  * per-cell: 2^16 F(y) + F(y_rem_lo) + 256 F(y_rem_hi)
              - inv * P(x, w; eq) == 2^15   (padding cells satisfy it);
  * ranges: byte LogUps on x_biased (= x + 128), y_rem_lo, y_rem_hi with
    witness commitments SHARED with the tile columns; coupling
    F(x_biased) - F(x) == 128.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.goldilocks_rmsnorm_tile_reference import (
    GoldilocksRmsnormTileStatementV3,
    RMSNORM_SCALE_V3,
)
from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (
    GoldilocksSuccinctLogupStatementV3,
    prove_goldilocks_succinct_logup_v3,
    verify_goldilocks_succinct_logup_v3,
)
from verallm.proof_v3.goldilocks_succinct_product_argument_reference import (
    GoldilocksSuccinctProductStatementV3,
    prove_goldilocks_succinct_product_v3,
    verify_goldilocks_succinct_product_v3,
)
from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
    SuccinctEqFoldProofV3,
    column_pcs_statement_v3,
    commit_succinct_column_v3,
    derive_tile_eq_point_v3,
    prove_succinct_eq_fold_v3,
    verify_succinct_eq_fold_v3,
)

_TILE_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_SUCCINCT_RMSNORM/V1"
_COLUMN_TAGS: Final = ("x", "x_biased", "y", "y_rem_lo", "y_rem_hi")
_HALF: Final = RMSNORM_SCALE_V3 // 2


def _sf(value: int) -> int:
    return value % GOLDILOCKS_MODULUS


@dataclass(frozen=True, slots=True)
class GoldilocksSuccinctRmsnormProofV3:
    column_commitments: tuple[bytes, ...]
    weight_commitment: bytes
    eq_folds: tuple[SuccinctEqFoldProofV3, ...]
    product_xx: object
    product_xw: object
    x_range_logup: object
    rem_lo_logup: object
    rem_hi_logup: object
    mean_sq: int
    mean_rem: int


def _tile_digest(statement, row_index: int) -> bytes:
    return hashlib.sha256(
        _TILE_DOMAIN + statement.digest() + row_index.to_bytes(4, "little")
    ).digest()


def weight_column_values_v3(statement) -> tuple[int, ...]:
    dim = statement.model_dim
    pad = 1 << (dim - 1).bit_length()
    return tuple(
        _sf(statement.weight[k]) if k < dim else 0 for k in range(pad)
    )


def prove_goldilocks_succinct_rmsnorm_v3(
    *,
    statement: GoldilocksRmsnormTileStatementV3,
    x_row,
    row_index: int = 0,
    validator_nonce: bytes,
    fused=None,
):
    dim = statement.model_dim
    pad = 1 << (dim - 1).bit_length()
    variable_count = pad.bit_length() - 1
    tile_digest = _tile_digest(statement, row_index)
    x = [int(v) for v in x_row]
    if len(x) != dim:
        raise ProofV3Error("succinct rmsnorm row shape is wrong")
    sum_sq = sum(v * v for v in x)
    mean_sq, mean_rem = divmod(sum_sq, dim)
    index = statement.rsqrt_index(mean_sq)
    inv = statement.rsqrt_table[index]
    columns = {tag: [] for tag in _COLUMN_TAGS}
    outputs = []
    for k in range(pad):
        xv = x[k] if k < dim else 0
        wv = statement.weight[k] if k < dim else 0
        prod = xv * wv * inv + _HALF
        y_shift, y_rem = divmod(prod + (1 << 40), RMSNORM_SCALE_V3)
        y = y_shift - (1 << 24)
        if k < dim:
            outputs.append(y)
        columns["x"].append(_sf(xv))
        columns["x_biased"].append(xv + 128)
        columns["y"].append(_sf(y))
        columns["y_rem_lo"].append(y_rem & 0xFF)
        columns["y_rem_hi"].append(y_rem >> 8)
    committed = {
        tag: commit_succinct_column_v3(
            tile_digest=tile_digest, tag=tag, values=tuple(columns[tag]),
            fused=fused)
        for tag in _COLUMN_TAGS
    }
    w_column = commit_succinct_column_v3(
        tile_digest=tile_digest, tag="w",
        values=weight_column_values_v3(statement), fused=fused)
    commitments = tuple(
        committed[tag].tree.commitment for tag in _COLUMN_TAGS)
    z_point = derive_tile_eq_point_v3(
        tile_digest, commitments + (w_column.tree.commitment,),
        validator_nonce, variable_count)
    eq_folds = tuple(
        prove_succinct_eq_fold_v3(
            tile_digest=tile_digest, column=committed[tag], z_point=z_point,
            validator_nonce=validator_nonce, fused=fused)
        for tag in _COLUMN_TAGS
    )
    ones_components = tuple((1, 1) for _ in range(variable_count))
    eq_components = tuple(
        ((1 - z) % GOLDILOCKS_MODULUS, z) for z in reversed(z_point))
    product_proofs = {}
    for name, b_column, comps in (
        ("xx", committed["x"], ones_components),
        ("xw", w_column, eq_components),
    ):
        prod_statement = GoldilocksSuccinctProductStatementV3(
            validator_binding_digest=hashlib.sha256(
                tile_digest + b"prod/" + name.encode()).digest(),
            variable_count=variable_count,
            factor_component_sizes=tuple(2 for _ in range(variable_count)),
        )
        if fused is not None:
            from verallm.proof_v3.native_pcs_backend import (
                fused_prove_goldilocks_succinct_product_v3,
            )

            product_proofs[name] = fused_prove_goldilocks_succinct_product_v3(
                fold_extension=fused[0], tree_extension=fused[1],
                statement=prod_statement,
                a_column=committed["x"], b_column=b_column,
                factor_components=comps,
                validator_nonce=validator_nonce)
        else:
            product_proofs[name] = prove_goldilocks_succinct_product_v3(
                statement=prod_statement,
                a_pcs_statement=committed["x"].pcs_statement,
                b_pcs_statement=b_column.pcs_statement,
                a_tree=committed["x"].tree,
                b_tree=b_column.tree,
                a_evaluations=committed["x"].values,
                b_evaluations=b_column.values,
                factor_components=comps,
                validator_nonce=validator_nonce,
            )
    logups = {}
    for name, column_tag in (
        ("xrange", "x_biased"),
        ("remlo", "y_rem_lo"),
        ("remhi", "y_rem_hi"),
    ):
        column = committed[column_tag]
        logup_statement = GoldilocksSuccinctLogupStatementV3(
            validator_binding_digest=hashlib.sha256(
                tile_digest + b"logup/" + name.encode()).digest(),
            table=tuple(range(256)),
            witness_variable_count=variable_count,
            witness_binding_override=(
                column.pcs_statement.validator_binding_digest),
        )
        if fused is not None:
            from verallm.proof_v3.native_pcs_backend import (
                fused_prove_goldilocks_succinct_logup_v3,
            )

            logups[name] = fused_prove_goldilocks_succinct_logup_v3(
                fold_extension=fused[0], tree_extension=fused[1],
                statement=logup_statement, looked_up_values=column.values,
                validator_nonce=validator_nonce, witness_column=column)
        else:
            logups[name] = prove_goldilocks_succinct_logup_v3(
                statement=logup_statement, looked_up_values=column.values,
                validator_nonce=validator_nonce, witness_tree=column.tree)
    proof = GoldilocksSuccinctRmsnormProofV3(
        column_commitments=commitments,
        weight_commitment=w_column.tree.commitment,
        eq_folds=eq_folds,
        product_xx=product_proofs["xx"],
        product_xw=product_proofs["xw"],
        x_range_logup=logups["xrange"],
        rem_lo_logup=logups["remlo"],
        rem_hi_logup=logups["remhi"],
        mean_sq=mean_sq,
        mean_rem=mean_rem,
    )
    return proof, tuple(outputs)


def verify_goldilocks_succinct_rmsnorm_v3(
    proof: object,
    *,
    statement: GoldilocksRmsnormTileStatementV3,
    row_index: int = 0,
    expected_weight_commitment: bytes,
    validator_nonce: bytes,
) -> None:
    try:
        if not isinstance(proof, GoldilocksSuccinctRmsnormProofV3):
            raise ProofV3VerificationError(
                "succinct rmsnorm proof type is wrong")
        dim = statement.model_dim
        pad = 1 << (dim - 1).bit_length()
        variable_count = pad.bit_length() - 1
        tile_digest = _tile_digest(statement, row_index)
        if proof.weight_commitment != expected_weight_commitment:
            raise ProofV3VerificationError(
                "succinct rmsnorm weight commitment is not the model's")
        z_point = derive_tile_eq_point_v3(
            tile_digest,
            tuple(proof.column_commitments) + (proof.weight_commitment,),
            validator_nonce, variable_count)
        folds = {}
        for tag, commitment, fold_proof in zip(
            _COLUMN_TAGS, proof.column_commitments, proof.eq_folds,
            strict=True,
        ):
            folds[tag] = verify_succinct_eq_fold_v3(
                fold_proof, tile_digest=tile_digest, tag=tag,
                pcs_statement=column_pcs_statement_v3(
                    tile_digest, tag, variable_count),
                commitment=commitment, z_point=z_point,
                validator_nonce=validator_nonce)
        ones_components = tuple((1, 1) for _ in range(variable_count))
        eq_components = tuple(
            ((1 - z) % GOLDILOCKS_MODULUS, z) for z in reversed(z_point))
        tags = dict(zip(_COLUMN_TAGS, proof.column_commitments, strict=True))
        for name, proof_obj, b_commitment, b_tag, comps in (
            ("xx", proof.product_xx, tags["x"], "x", ones_components),
            ("xw", proof.product_xw, proof.weight_commitment, "w",
             eq_components),
        ):
            prod_statement = GoldilocksSuccinctProductStatementV3(
                validator_binding_digest=hashlib.sha256(
                    tile_digest + b"prod/" + name.encode()).digest(),
                variable_count=variable_count,
                factor_component_sizes=tuple(
                    2 for _ in range(variable_count)),
            )
            verify_goldilocks_succinct_product_v3(
                proof_obj, statement=prod_statement,
                a_pcs_statement=column_pcs_statement_v3(
                    tile_digest, "x", variable_count),
                b_pcs_statement=column_pcs_statement_v3(
                    tile_digest, b_tag, variable_count),
                a_commitment=tags["x"], b_commitment=b_commitment,
                factor_components=comps, validator_nonce=validator_nonce,
                expected_sum=proof_obj.claimed_sum,
            )
        # scalar checks against the verifier's OWN statement tables
        mean_sq, mean_rem = int(proof.mean_sq), int(proof.mean_rem)
        if not 0 <= mean_rem < dim:
            raise ProofV3VerificationError(
                "succinct rmsnorm mean remainder is out of range")
        if (mean_sq * dim + mean_rem) % GOLDILOCKS_MODULUS != (
            proof.product_xx.claimed_sum % GOLDILOCKS_MODULUS
        ):
            raise ProofV3VerificationError(
                "succinct rmsnorm mean decomposition fails")
        if mean_sq >= len(statement.rsqrt_table) or mean_sq < 0:
            raise ProofV3VerificationError(
                "succinct rmsnorm mean square exceeds the rsqrt table")
        inv = statement.rsqrt_table[mean_sq]
        # int8 bias coupling pins the x range
        if (folds["x_biased"] - folds["x"]) % GOLDILOCKS_MODULUS != 128:
            raise ProofV3VerificationError(
                "succinct rmsnorm int8 bias coupling fails")
        # per-cell Euclidean coupling
        lhs = (
            RMSNORM_SCALE_V3 * folds["y"]
            + folds["y_rem_lo"]
            + 256 * folds["y_rem_hi"]
            - inv * (proof.product_xw.claimed_sum % GOLDILOCKS_MODULUS)
        ) % GOLDILOCKS_MODULUS
        if lhs != _HALF:
            raise ProofV3VerificationError(
                "succinct rmsnorm Euclidean coupling fails")
        for name, column_tag, logup_proof in (
            ("xrange", "x_biased", proof.x_range_logup),
            ("remlo", "y_rem_lo", proof.rem_lo_logup),
            ("remhi", "y_rem_hi", proof.rem_hi_logup),
        ):
            logup_statement = GoldilocksSuccinctLogupStatementV3(
                validator_binding_digest=hashlib.sha256(
                    tile_digest + b"logup/" + name.encode()).digest(),
                table=tuple(range(256)),
                witness_variable_count=variable_count,
                witness_binding_override=column_pcs_statement_v3(
                    tile_digest, column_tag, variable_count
                ).validator_binding_digest,
            )
            verify_goldilocks_succinct_logup_v3(
                logup_proof, statement=logup_statement,
                witness_commitment=tags[column_tag],
                validator_nonce=validator_nonce)
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "succinct rmsnorm proof is malformed") from exc
