"""PRODUCTION-wire per-LAYER succinct RMSNorm: ALL rows in one tile.

The per-row succinct tile costs a full sub-argument set per token; at 32
tokens that is 32x the verifier work of one tile.  This module proves
every row of one RMSNorm application in a single (t, d) cube:

  * row sums of squares via one product argument with a row-eq factor,
    coupled to per-row ``mean_q/mean_rem`` columns by the Euclidean
    identity ``sum_sq == dim * mean_q + mean_rem`` (``mean_rem`` proven
    in ``range(dim)``);
  * ``inv = rsqrt_table[mean_q]`` via one packed LogUp on the row cube
    (the reference clamp is vacuous for byte-range inputs: mean of
    squares of int8 cells is always inside the table);
  * the per-cell normalisation ``2^16 * y + y_rem == x * w * inv`` via
    one product argument whose public factor carries BOTH the eq point
    and the validator-owned weight vector (w needs no commitment at
    all), with ``y_rem`` byte-decomposed;
  * ``inv`` broadcast onto the cell cube pinned by one eq-fold coupling.

Columns: cells x, x_biased, y, y_rem_lo, y_rem_hi, inv_b; rows mean_q,
mean_rem, inv, w_rsqrt.  Padding rows are honest zero-input rows.
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
    _eq_table,
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

_TILE_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_SUCCINCT_RMSNORM_LAYER/V1"
_RPACK: Final = 1 << 32
_CELL_TAGS: Final = ("x", "x_biased", "y", "y_rem_lo", "y_rem_hi", "inv_b")
_ROW_TAGS: Final = ("mean_q", "mean_rem", "inv", "w_rsqrt")
_ALL_TAGS: Final = _CELL_TAGS + _ROW_TAGS
_GROUP_PLAN: Final = (
    ("grp_cells", ("x", "y", "inv_b")),
    ("grp_bytes", ("x_biased", "y_rem_lo", "y_rem_hi")),
    ("grp_rows", ("mean_q", "mean_rem", "inv", "w_rsqrt")),
)


def _block_bits(count: int) -> int:
    return max(1, (count - 1).bit_length())


def _sf(value: int) -> int:
    return value % GOLDILOCKS_MODULUS


def _pow2(n: int) -> int:
    return 1 << max(1, (n - 1).bit_length())


def _canonical_output_rows(value):
    try:
        rows = tuple(value)
    except TypeError as exc:
        raise ProofV3Error(
            "rmsnorm output-row bindings are malformed"
        ) from exc
    result = []
    for item in rows:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ProofV3Error(
                "rmsnorm output-row bindings are malformed"
            )
        row, values = item
        if type(row) is not int:
            raise ProofV3Error(
                "rmsnorm output-row bindings are malformed"
            )
        try:
            values = tuple(values)
        except TypeError as exc:
            raise ProofV3Error(
                "rmsnorm output-row bindings are malformed"
            ) from exc
        if any(type(cell) is not int for cell in values):
            raise ProofV3Error(
                "rmsnorm output-row bindings are malformed"
            )
        result.append((row, values))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class GoldilocksSuccinctRmsnormLayerProofV3:
    column_commitments: tuple[bytes, ...]        # ordered by _ALL_TAGS
    eq_folds: tuple[SuccinctEqFoldProofV3, ...]  # ordered by _ALL_TAGS
    sq_product: object
    y_product: object
    logups: tuple[object, ...]                   # ordered by _logup_plan
    output_row_folds: tuple[SuccinctEqFoldProofV3, ...] = ()
    grouped_logup_aux: bool = False
    batch_openings: tuple[tuple[str, object], ...] | None = None


def _tile_digest(statement, token_count: int) -> bytes:
    return hashlib.sha256(
        _TILE_DOMAIN + statement.digest()
        + token_count.to_bytes(4, "little")
    ).digest()


def _dims(statement, token_count: int):
    d_pad = _pow2(statement.model_dim)
    t_pad = _pow2(token_count)
    ld, lt = d_pad.bit_length() - 1, t_pad.bit_length() - 1
    return d_pad, t_pad, ld, lt


def _tag_vars(statement, token_count: int, tag: str) -> int:
    _d_pad, _t_pad, ld, lt = _dims(statement, token_count)
    if tag.startswith("grp_"):
        members = dict(_GROUP_PLAN)[tag]
        return _tag_vars(statement, token_count, members[0]) + _block_bits(
            len(members))
    return lt if tag in _ROW_TAGS else ld + lt


def _logup_plan(statement):
    packed = tuple(
        index + _RPACK * value
        for index, value in enumerate(statement.rsqrt_table)
    )
    return (
        ("bytes", tuple(range(256)), "grp_bytes"),
        ("mean_rem", tuple(range(statement.model_dim)), "mean_rem"),
        ("rsqrt", packed, "w_rsqrt"),
    )


def _eq_components(z_bits_lsb):
    return tuple(
        ((1 - z) % GOLDILOCKS_MODULUS, z % GOLDILOCKS_MODULUS)
        for z in reversed(z_bits_lsb)
    )


def _product_setups(statement, token_count: int, z_c):
    d_pad, _t_pad, ld, _lt = _dims(statement, token_count)
    z_d, z_t = z_c[:ld], z_c[ld:]
    eq_d = _eq_table(z_d)
    weight_vec = tuple(
        eq_d[k] * _sf(statement.weight[k]) % GOLDILOCKS_MODULUS
        if k < statement.model_dim else 0
        for k in range(d_pad)
    )
    return (
        ("sq", "x", "x",
         _eq_components(z_t) + tuple((1, 1) for _ in range(ld))),
        ("ynorm", "x", "inv_b",
         _eq_components(z_t) + (weight_vec,)),
    )


def _product_statement(tile_digest: bytes, name: str, components):
    return GoldilocksSuccinctProductStatementV3(
        validator_binding_digest=hashlib.sha256(
            tile_digest + b"prod/" + name.encode()).digest(),
        variable_count=sum(
            (len(c) - 1).bit_length() for c in components),
        factor_component_sizes=tuple(len(c) for c in components),
    )


def _build_columns(statement, x_rows, token_count: int):
    """Vectorized (torch int64) column builder; exact integer semantics."""

    import torch

    dim = statement.model_dim
    d_pad, t_pad, _ld, _lt = _dims(statement, token_count)
    if len(x_rows) != token_count or any(len(r) != dim for r in x_rows):
        raise ProofV3Error("rmsnorm layer input shape is wrong")
    x = torch.zeros((t_pad, d_pad), dtype=torch.int64)
    x[:token_count, :dim] = torch.tensor(
        [[int(v) for v in row] for row in x_rows], dtype=torch.int64)
    if not bool(((x >= -128) & (x <= 127)).all()):
        raise ProofV3Error("rmsnorm layer input exceeds int8")
    sum_sq = (x * x).sum(dim=1)                      # padded d cells are 0
    mean_q = sum_sq // dim
    mean_rem = sum_sq - mean_q * dim
    if int(mean_q.max()) >= len(statement.rsqrt_table):
        raise ProofV3Error("rmsnorm layer mean escapes the rsqrt table")
    rsqrt_t = torch.tensor(statement.rsqrt_table, dtype=torch.int64)
    inv = rsqrt_t[mean_q]                            # [t_pad]
    w = torch.zeros(d_pad, dtype=torch.int64)
    w[:dim] = torch.tensor(
        [int(v) for v in statement.weight], dtype=torch.int64)
    prod = x * w.unsqueeze(0) * inv.unsqueeze(1) + RMSNORM_SCALE_V3 // 2
    y_shift = (prod + (1 << 40)) // RMSNORM_SCALE_V3
    y_rem = (prod + (1 << 40)) - y_shift * RMSNORM_SCALE_V3
    y = y_shift - (1 << 24)

    def flat(t):
        return tuple(t.reshape(-1).tolist())

    def enc_field(t):
        # canonical-mod-2^64 encoding: p + x wraps to x - (2^32 - 1)
        t = t.reshape(-1)
        return torch.where(t < 0, t - ((1 << 32) - 1), t).contiguous()

    columns = {
        "x": enc_field(x),
        "x_biased": flat(x + 128),
        "y": enc_field(y),
        "y_rem_lo": flat(y_rem & 0xFF),
        "y_rem_hi": flat(y_rem >> 8),
        "inv_b": enc_field(inv.unsqueeze(1).expand(t_pad, d_pad)),
        "mean_q": flat(mean_q),
        "mean_rem": flat(mean_rem),
        "inv": flat(inv),
        "w_rsqrt": flat(mean_q + _RPACK * inv),
    }
    outputs = tuple(
        tuple(int(v) for v in y[t, :dim].tolist())
        for t in range(token_count)
    )
    return columns, outputs


def prove_goldilocks_succinct_rmsnorm_layer_v3(
    *,
    statement: GoldilocksRmsnormTileStatementV3,
    x_rows,
    validator_nonce: bytes,
    fused=None,
    aggregate: bool = False,
    external_collector=None,
    collector_ns: str = "",
    bound_output_rows=(),
):
    """Prove one full RMSNorm layer; returns (proof, integer y rows)."""

    token_count = len(x_rows)
    tile_digest = _tile_digest(statement, token_count)
    columns, outputs = _build_columns(statement, x_rows, token_count)
    del aggregate  # grouped trees require batched openings: always on
    from verallm.proof_v3.goldilocks_succinct_batch_opening import (
        BatchOpeningCollectorV3,
    )
    from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
        commit_succinct_column_group_v3,
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
    committed = {}
    groups = {}
    for group_tag, member_tags in _GROUP_PLAN:
        group, members = commit_succinct_column_group_v3(
            tile_digest=tile_digest, group_tag=group_tag,
            ordered=tuple(
                (tag, columns[tag]
                 if hasattr(columns[tag], "numel")
                 else tuple(columns[tag]))
                for tag in member_tags),
            fused=fused)
        groups[group_tag] = group
        committed.update(members)
        collector.register_group(group)
        for tag in member_tags:
            collector.register_column(tag, members[tag])
    commitments = tuple(committed[tag].tree.commitment for tag in _ALL_TAGS)
    n_cells = _tag_vars(statement, token_count, "x")
    z_c = derive_tile_eq_point_v3(
        tile_digest, commitments, validator_nonce, n_cells, label=b"zC")
    _d_pad, _t_pad, ld, _lt = _dims(statement, token_count)
    z_t = z_c[ld:]
    if fused is not None and hasattr(fused[0], "round_partials_b"):
        from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
            prove_succinct_eq_folds_batched_v3,
        )

        by_tag = {}
        for group_tag, member_tags in _GROUP_PLAN:
            batch = prove_succinct_eq_folds_batched_v3(
                tile_digest=tile_digest,
                members=tuple(committed[tag] for tag in member_tags),
                group_device=groups[group_tag].device_values,
                z_point=(z_c if member_tags[0] in _CELL_TAGS else z_t),
                validator_nonce=validator_nonce,
                fused=fused, collector=collector)
            by_tag.update(zip(member_tags, batch, strict=True))
        eq_folds = tuple(by_tag[tag] for tag in _ALL_TAGS)
    else:
        eq_folds = tuple(
            prove_succinct_eq_fold_v3(
                tile_digest=tile_digest, column=committed[tag],
                z_point=(z_c if tag in _CELL_TAGS else z_t),
                validator_nonce=validator_nonce, fused=fused,
                collector=collector)
            for tag in _ALL_TAGS
        )
    products = {}
    for name, a_tag, b_tag, components in _product_setups(
        statement, token_count, z_c
    ):
        prod_statement = _product_statement(tile_digest, name, components)
        if fused is not None:
            from verallm.proof_v3.native_pcs_backend import (
                fused_prove_goldilocks_succinct_product_v3,
            )

            products[name] = fused_prove_goldilocks_succinct_product_v3(
                fold_extension=fused[0], tree_extension=fused[1],
                statement=prod_statement,
                a_column=committed[a_tag], b_column=committed[b_tag],
                factor_components=components,
                validator_nonce=validator_nonce,
                collector=collector, a_tag=a_tag, b_tag=b_tag)
        else:
            products[name] = prove_goldilocks_succinct_product_v3(
                statement=prod_statement,
                a_pcs_statement=committed[a_tag].pcs_statement,
                b_pcs_statement=committed[b_tag].pcs_statement,
                a_tree=committed[a_tag].tree,
                b_tree=committed[b_tag].tree,
                a_evaluations=committed[a_tag].values,
                b_evaluations=committed[b_tag].values,
                factor_components=components,
                validator_nonce=validator_nonce,
                collector=collector, a_tag=a_tag, b_tag=b_tag)
    logup_instances = []
    for name, table, column_tag in _logup_plan(statement):
        column = (
            groups[column_tag] if column_tag.startswith("grp_")
            else committed[column_tag])
        logup_statement = GoldilocksSuccinctLogupStatementV3(
            validator_binding_digest=hashlib.sha256(
                tile_digest + b"logup/" + name.encode()).digest(),
            table=table,
            witness_variable_count=_tag_vars(
                statement, token_count, column_tag),
            witness_binding_override=(
                column.pcs_statement.validator_binding_digest),
        )
        logup_instances.append(
            (logup_statement, column, f"logup/{name}", column_tag))
    if fused is not None:
        from verallm.proof_v3.native_pcs_backend import (
            fused_prove_logup_batch_v3,
        )

        logups = list(fused_prove_logup_batch_v3(
            fold_extension=fused[0], tree_extension=fused[1],
            tile_digest=tile_digest, instances=logup_instances,
            validator_nonce=validator_nonce, collector=collector))
    else:
        logups = [
            prove_goldilocks_succinct_logup_v3(
                statement=logup_statement,
                looked_up_values=column.values,
                validator_nonce=validator_nonce,
                witness_tree=column.tree,
                collector=collector, tag_prefix=tag_prefix,
                witness_tag=column_tag)
            for logup_statement, column, tag_prefix, column_tag
            in logup_instances
        ]
    output_rows = _canonical_output_rows(bound_output_rows)
    if (
        tuple(row for row, _values in output_rows)
        != tuple(sorted({row for row, _values in output_rows}))
        or any(
            row < 0
            or row >= token_count
            or len(values) != statement.model_dim
            or tuple(outputs[row]) != values
            for row, values in output_rows
        )
    ):
        raise ProofV3Error(
            "rmsnorm output-row bindings disagree with the computed output"
        )
    commitments_for_point = tuple(
        committed[tag].tree.commitment for tag in _ALL_TAGS
    )
    _d_pad, _t_pad, ld, lt = _dims(statement, token_count)
    output_row_folds = []
    for row, _values in output_rows:
        z_d = derive_tile_eq_point_v3(
            tile_digest,
            commitments_for_point,
            validator_nonce,
            ld,
            label=b"y-row/" + row.to_bytes(4, "little"),
        )
        row_point = tuple((row >> bit) & 1 for bit in range(lt))
        output_row_folds.append(
            prove_succinct_eq_fold_v3(
                tile_digest=tile_digest,
                column=committed["y"],
                z_point=z_d + row_point,
                validator_nonce=validator_nonce,
                fused=fused,
                collector=collector,
            )
        )
    if external_collector is not None:
        from verallm.proof_v3.goldilocks_succinct_batch_opening import (
            park_column_device_values_v3,
        )

        for group in groups.values():
            park_column_device_values_v3(group)
        batch_openings = None
    else:
        batch_openings = tuple(sorted(collector.prove_all(
            validator_nonce=validator_nonce, fused=fused).items()))
    proof = GoldilocksSuccinctRmsnormLayerProofV3(
        column_commitments=commitments,
        eq_folds=eq_folds,
        sq_product=products["sq"],
        y_product=products["ynorm"],
        logups=tuple(logups),
        output_row_folds=tuple(output_row_folds),
        grouped_logup_aux=fused is not None,
        batch_openings=batch_openings,
    )
    return proof, outputs


def verify_goldilocks_succinct_rmsnorm_layer_v3(
    proof: object,
    *,
    statement: GoldilocksRmsnormTileStatementV3,
    token_count: int,
    validator_nonce: bytes,
    external_checker=None,
    checker_ns: str = "",
    expected_output_rows=(),
) -> None | tuple[dict[str, object], dict[str, bytes]]:
    """Succinct CPU verification of one full RMSNorm layer."""

    try:
        if not isinstance(proof, GoldilocksSuccinctRmsnormLayerProofV3):
            raise ProofV3VerificationError(
                "succinct rmsnorm layer proof type is wrong")
        tile_digest = _tile_digest(statement, token_count)
        from verallm.proof_v3.goldilocks_succinct_batch_opening import (
            BatchClaimCheckerV3,
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
        for group_tag, member_tags in _GROUP_PLAN:
            bits = _block_bits(len(member_tags))
            for index, tag in enumerate(member_tags):
                checker.alias(tag, group_tag, tuple(
                    (index >> j) & 1 for j in range(bits)))
        if len(proof.column_commitments) != len(_ALL_TAGS) or len(
            proof.eq_folds
        ) != len(_ALL_TAGS):
            raise ProofV3VerificationError(
                "succinct rmsnorm layer shape is wrong")
        tags = dict(zip(_ALL_TAGS, proof.column_commitments, strict=True))
        for group_tag, member_tags in _GROUP_PLAN:
            if len({tags[tag] for tag in member_tags}) != 1:
                raise ProofV3VerificationError(
                    "succinct rmsnorm layer group roots disagree")
        n_cells = _tag_vars(statement, token_count, "x")
        z_c = derive_tile_eq_point_v3(
            tile_digest, tuple(proof.column_commitments), validator_nonce,
            n_cells, label=b"zC")
        _d_pad, _t_pad, ld, _lt = _dims(statement, token_count)
        z_t = z_c[ld:]
        folds = {}
        for tag, fold_proof in zip(
            _ALL_TAGS, proof.eq_folds, strict=True
        ):
            folds[tag] = verify_succinct_eq_fold_v3(
                fold_proof,
                tile_digest=tile_digest,
                tag=tag,
                pcs_statement=column_pcs_statement_v3(
                    tile_digest, tag,
                    _tag_vars(statement, token_count, tag)),
                commitment=tags[tag],
                z_point=(z_c if tag in _CELL_TAGS else z_t),
                validator_nonce=validator_nonce,
                checker=checker,
            )
        claims = {}
        for (name, a_tag, b_tag, components), prod_proof in zip(
            _product_setups(statement, token_count, z_c),
            (proof.sq_product, proof.y_product),
            strict=True,
        ):
            prod_statement = _product_statement(
                tile_digest, name, components)
            claims[name] = prod_proof.claimed_sum % GOLDILOCKS_MODULUS
            verify_goldilocks_succinct_product_v3(
                prod_proof,
                statement=prod_statement,
                a_pcs_statement=column_pcs_statement_v3(
                    tile_digest, a_tag,
                    _tag_vars(statement, token_count, a_tag)),
                b_pcs_statement=column_pcs_statement_v3(
                    tile_digest, b_tag,
                    _tag_vars(statement, token_count, b_tag)),
                a_commitment=tags[a_tag],
                b_commitment=tags[b_tag],
                factor_components=components,
                validator_nonce=validator_nonce,
                expected_sum=claims[name],
                checker=checker, a_tag=a_tag, b_tag=b_tag,
            )
        p = GOLDILOCKS_MODULUS

        def check(condition: bool, message: str) -> None:
            if not condition:
                raise ProofV3VerificationError(
                    f"succinct rmsnorm layer {message}")

        # row squares: sum_d x^2 == dim * mean_q + mean_rem per row
        check(
            claims["sq"]
            == (statement.model_dim * folds["mean_q"]
                + folds["mean_rem"]) % p,
            "mean coupling fails")
        # rsqrt pack: w_rsqrt == mean_q + 2^32 * inv
        check(
            folds["w_rsqrt"]
            == (folds["mean_q"] + _RPACK * folds["inv"]) % p,
            "rsqrt packing coupling fails")
        # inv broadcast onto the cell cube
        check(folds["inv_b"] == folds["inv"],
              "inv broadcast coupling fails")
        # int8 bias
        check((folds["x_biased"] - folds["x"]) % p == 128,
              "x bias coupling fails")
        # per-cell normalisation: 2^16 y + y_rem == x * w * inv + 2^15
        # (sum eq == 1 absorbs the rounding constant)
        lhs = (
            RMSNORM_SCALE_V3 * folds["y"]
            + folds["y_rem_lo"]
            + 256 * folds["y_rem_hi"]
        ) % p
        check(
            lhs == (claims["ynorm"] + RMSNORM_SCALE_V3 // 2) % p,
            "normalisation coupling fails")
        # LogUps bind to the tile column roots
        if len(proof.logups) != len(_logup_plan(statement)):
            raise ProofV3VerificationError(
                "succinct rmsnorm layer logup set is wrong")
        def _witness_root(column_tag: str) -> bytes:
            if column_tag.startswith("grp_"):
                return tags[dict(_GROUP_PLAN)[column_tag][0]]
            return tags[column_tag]

        if type(proof.grouped_logup_aux) is not bool:
            raise ProofV3VerificationError(
                "succinct rmsnorm grouped-aux marker is malformed")
        grouped_aux = proof.grouped_logup_aux
        aux_statements_grouped: dict = {}
        aux_commitments_grouped: dict = {}
        if grouped_aux:
            from verallm.proof_v3.native_pcs_backend import (
                logup_aux_group_plan_v3,
            )

            plan_list = tuple(zip(
                _logup_plan(statement), proof.logups, strict=True))
            shapes = tuple(
                (f"logup/{name}",
                 _tag_vars(statement, token_count, column_tag),
                 GoldilocksSuccinctLogupStatementV3(
                     validator_binding_digest=hashlib.sha256(
                         tile_digest + b"logup/" + name.encode()
                     ).digest(),
                     table=table,
                     witness_variable_count=_tag_vars(
                         statement, token_count, column_tag),
                 ).table_variable_count)
                for (name, table, column_tag), _lp in plan_list)
            plans, group_meta = logup_aux_group_plan_v3(shapes)
            proof_by_prefix = {
                f"logup/{name}": lp
                for (name, _t, _c), lp in plan_list
            }
            for kind in ("M", "D", "E"):
                for prefix, (group_tag, block_point) in plans[
                    kind
                ].items():
                    local = (
                        f"{prefix}/{kind}" if kind == "M"
                        else f"{prefix}/{kind}0")
                    checker.alias(local, group_tag, block_point)
                    member = proof_by_prefix[prefix]
                    root = (
                        member.multiplicity_commitment
                        if kind == "M"
                        else member.inverse_commitments[
                            0 if kind == "D" else 1])
                    vars_total, _used = group_meta[group_tag]
                    aux_statements_grouped[group_tag] = (
                        column_pcs_statement_v3(
                            tile_digest, group_tag, vars_total))
                    if aux_commitments_grouped.setdefault(
                        group_tag, root
                    ) != root:
                        raise ProofV3VerificationError(
                            "succinct rmsnorm aux group roots disagree")

        for (name, table, column_tag), logup_proof in zip(
            _logup_plan(statement), proof.logups, strict=True
        ):
            logup_statement = GoldilocksSuccinctLogupStatementV3(
                validator_binding_digest=hashlib.sha256(
                    tile_digest + b"logup/" + name.encode()).digest(),
                table=table,
                witness_variable_count=_tag_vars(
                    statement, token_count, column_tag),
                witness_binding_override=column_pcs_statement_v3(
                    tile_digest, column_tag,
                    _tag_vars(statement, token_count, column_tag),
                ).validator_binding_digest,
            )
            verify_goldilocks_succinct_logup_v3(
                logup_proof, statement=logup_statement,
                witness_commitment=_witness_root(column_tag),
                validator_nonce=validator_nonce,
                checker=checker, tag_prefix=f"logup/{name}",
                witness_tag=column_tag)
        output_rows = _canonical_output_rows(expected_output_rows)
        if (
            tuple(row for row, _values in output_rows)
            != tuple(sorted({row for row, _values in output_rows}))
            or len(output_rows) != len(proof.output_row_folds)
            or any(
                row < 0
                or row >= token_count
                or len(values) != statement.model_dim
                for row, values in output_rows
            )
        ):
            raise ProofV3VerificationError(
                "succinct rmsnorm output-row inventory is malformed"
            )
        commitments_for_point = tuple(proof.column_commitments)
        for (row, values), fold_proof in zip(
            output_rows,
            proof.output_row_folds,
            strict=True,
        ):
            z_d = derive_tile_eq_point_v3(
                tile_digest,
                commitments_for_point,
                validator_nonce,
                ld,
                label=b"y-row/" + row.to_bytes(4, "little"),
            )
            row_point = tuple(
                (row >> bit) & 1 for bit in range(_lt)
            )
            claimed = verify_succinct_eq_fold_v3(
                fold_proof,
                tile_digest=tile_digest,
                tag="y",
                pcs_statement=column_pcs_statement_v3(
                    tile_digest,
                    "y",
                    _tag_vars(statement, token_count, "y"),
                ),
                commitment=tags["y"],
                z_point=z_d + row_point,
                validator_nonce=validator_nonce,
                checker=checker,
            )
            expected = sum(
                value % GOLDILOCKS_MODULUS * weight
                for value, weight in zip(
                    values + (0,) * (_d_pad - len(values)),
                    _eq_table(z_d),
                    strict=True,
                )
            ) % GOLDILOCKS_MODULUS
            if claimed != expected:
                raise ProofV3VerificationError(
                    "succinct rmsnorm output row is detached"
                )
        if checker is not None:
            from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (  # noqa: E501
                logup_batch_registry_v3,
            )

            statements = {
                group_tag: column_pcs_statement_v3(
                    tile_digest, group_tag,
                    _tag_vars(statement, token_count, group_tag))
                for group_tag, _members in _GROUP_PLAN
            }
            commitments = {
                group_tag: tags[member_tags[0]]
                for group_tag, member_tags in _GROUP_PLAN
            }
            statements.update(aux_statements_grouped)
            commitments.update(aux_commitments_grouped)
            for (name, table, column_tag), logup_proof in zip(
                _logup_plan(statement), proof.logups, strict=True
            ):
                logup_statement = GoldilocksSuccinctLogupStatementV3(
                    validator_binding_digest=hashlib.sha256(
                        tile_digest + b"logup/" + name.encode()).digest(),
                    table=table,
                    witness_variable_count=_tag_vars(
                        statement, token_count, column_tag),
                    witness_binding_override=column_pcs_statement_v3(
                        tile_digest, column_tag,
                        _tag_vars(statement, token_count, column_tag),
                    ).validator_binding_digest,
                )
                if not grouped_aux:
                    aux_statements, aux_commitments = (
                        logup_batch_registry_v3(
                            logup_proof, logup_statement, f"logup/{name}",
                            witness_tag=column_tag))
                    statements.update(aux_statements)
                    commitments.update(aux_commitments)
            if external_checker is not None:
                if proof.batch_openings is not None:
                    raise ProofV3VerificationError(
                        "aggregated rmsnorm proof carries its own opening")
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
            if proof.batch_openings is None:
                raise ProofV3VerificationError(
                    "standalone rmsnorm proof lacks its opening")
            checker.verify_all(
                dict(proof.batch_openings),
                statements=statements,
                commitments=commitments,
                validator_nonce=validator_nonce)
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "succinct rmsnorm layer proof is malformed") from exc


__all__ = [
    "GoldilocksSuccinctRmsnormLayerProofV3",
    "prove_goldilocks_succinct_rmsnorm_layer_v3",
    "verify_goldilocks_succinct_rmsnorm_layer_v3",
]
