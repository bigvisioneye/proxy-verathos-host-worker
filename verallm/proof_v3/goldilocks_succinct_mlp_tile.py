"""PRODUCTION-wire MLP (SwiGLU) tile: succinct prove + O(q log N) verify.

Same statement semantics as the reference MLP tile
(goldilocks_mlp_tile_reference), but every check the reference verifier
did over full openings becomes a succinct argument:

columns (PCS-committed once, padding cells are the valid zero-input
cell so every relation holds on padding):
    g8, u8, s, h_q, rem16, h8, W_silu, W_clamp

sub-arguments (one shared eq point z, derived post-commit):
  * silu LogUp:  W_silu column member of the packed SiLU table
  * clamp LogUp: W_clamp column member of the packed clamp table
  * range LogUp: rem16 member of range(divisor) (exact Euclidean range)
  * eq-folds of every column + one succinct product (s*u8), coupled by
    LINEARITY of the sum:
      pack-silu :  F(W_silu) - F(g8) - 2^32 F(s) == C_silu
      pack-clamp:  F(W_clamp) - F(h_q) - 2^32 F(h8) == C_clamp
      euclid    :  D*F(h_q) + F(rem16) - P(s,u8) == D//2
    with F(col) = sum eq(z,.) col and P = sum eq(z,.) s*u8; constants
    absorb via sum eq == 1.
  * LogUp witness commitments ARE the tile column commitments
    (witness_binding_override), so no equality bridging is needed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_mlp_tile_reference import (
    GoldilocksMlpTileStatementV3,
    GOLDILOCKS_MLP_SHIFT_V3,
    _SILU_BIAS,
    _CLAMP_BIAS,
    _SPACK,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
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
    commit_succinct_column_v3,
    column_pcs_statement_v3,
    derive_tile_eq_point_v3,
    prove_succinct_eq_fold_v3,
    verify_succinct_eq_fold_v3,
)

_TILE_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_SUCCINCT_MLP/V1"
_COLUMN_TAGS: Final = (
    "g8", "u8", "s", "h_q", "rem16", "h8", "W_silu", "W_clamp",
    "g8_biased", "u8_biased",
)
_GROUP_PLAN: Final = (
    ("grp_cells", ("g8", "u8", "s", "h_q", "rem16", "h8", "W_silu",
                   "W_clamp")),
    ("grp_biased8", ("g8_biased", "u8_biased")),
)


def _block_bits(count: int) -> int:
    return max(1, (count - 1).bit_length())


def _signed_field(value: int) -> int:
    return value % GOLDILOCKS_MODULUS


@dataclass(frozen=True, slots=True)
class GoldilocksSuccinctMlpProofV3:
    column_commitments: tuple[bytes, ...]        # ordered by _COLUMN_TAGS
    eq_folds: tuple[SuccinctEqFoldProofV3, ...]  # ordered by _COLUMN_TAGS
    product_proof: object                        # succinct product s*u8
    silu_logup: object
    clamp_logup: object
    byte_logup: object
    g8_range_logup: object
    u8_range_logup: object
    batch_openings: tuple[tuple[str, object], ...] | None = None


def _tile_digest(statement: GoldilocksMlpTileStatementV3) -> bytes:
    return hashlib.sha256(_TILE_DOMAIN + statement.digest()).digest()


def _build_columns(statement, gate_rows, up_rows):
    """Vectorized (torch int64) column builder; exact integer semantics.

    Padding cells are exact ``emit(0, 0)`` cells (same formulas on zero
    inputs), so every relation holds on padding.
    """

    import torch

    tokens, ff_dim = statement.token_count, statement.ff_dim
    divisor = statement.divisor
    shift = GOLDILOCKS_MLP_SHIFT_V3
    cells = statement.cell_count()
    real = tokens * ff_dim
    g8 = torch.zeros(cells, dtype=torch.int64)
    u8 = torch.zeros(cells, dtype=torch.int64)
    g8[:real] = torch.tensor(
        [int(v) for row in gate_rows for v in row], dtype=torch.int64)
    u8[:real] = torch.tensor(
        [int(v) for row in up_rows for v in row], dtype=torch.int64)
    if not bool(((g8 >= -128) & (g8 <= 127)
                 & (u8 >= -128) & (u8 <= 127)).all()):
        raise ProofV3Error("mlp tile inputs exceed int8")
    silu_t = torch.tensor(statement.silu_table, dtype=torch.int64)
    clamp_t = torch.tensor(statement.clamp_table, dtype=torch.int64)
    s_col = silu_t[g8 + 128]
    numerator = s_col * u8 + divisor * shift + divisor // 2
    q_shift = numerator // divisor
    rem = numerator - q_shift * divisor
    h_q = q_shift - shift
    index = torch.clamp(
        h_q + statement.clamp_offset, 0, len(statement.clamp_table) - 1)
    h8 = clamp_t[index]

    def flat(t):
        return tuple(t.tolist())

    def enc_field(t):
        # canonical-mod-2^64 encoding: p + x wraps to x - (2^32 - 1)
        return torch.where(t < 0, t - ((1 << 32) - 1), t).contiguous()

    # witness/public-fold columns need host values (LogUp counting,
    # CPU rounds); the rest stay device-resident encoded tensors
    columns = {
        "g8": enc_field(g8),
        "u8": enc_field(u8),
        "s": enc_field(s_col),
        "h_q": enc_field(h_q),
        "rem16": flat(rem),
        "h8": enc_field(h8),
        "W_silu": flat((g8 + 128) + _SPACK * (s_col + _SILU_BIAS)),
        "W_clamp": flat(index + _SPACK * (h8 + _CLAMP_BIAS)),
        "g8_biased": flat(g8 + 128),
        "u8_biased": flat(u8 + 128),
    }
    h8_real = h8[:real].reshape(tokens, ff_dim)
    outputs = tuple(
        tuple(int(v) for v in h8_real[t].tolist()) for t in range(tokens)
    )
    return columns, outputs


def _constants(statement):
    # pack-silu constant: 128 + 2^32 * BIAS  (from W = (g+128) + 2^32(s+B))
    c_silu = (128 + _SPACK * _SILU_BIAS) % GOLDILOCKS_MODULUS
    # pack-clamp: W = (h_q + OFF) + 2^32 (h8 + CBIAS)
    c_clamp = (
        statement.clamp_offset + _SPACK * _CLAMP_BIAS
    ) % GOLDILOCKS_MODULUS
    c_euclid = (statement.divisor // 2) % GOLDILOCKS_MODULUS
    return c_silu, c_clamp, c_euclid


def prove_goldilocks_succinct_mlp_v3(
    *,
    statement: GoldilocksMlpTileStatementV3,
    gate_rows,
    up_rows,
    validator_nonce: bytes,
    fused=None,
    aggregate: bool = False,
):
    del aggregate  # grouped trees require batched openings: always on
    tile_digest = _tile_digest(statement)
    columns, outputs = _build_columns(statement, gate_rows, up_rows)
    from verallm.proof_v3.goldilocks_succinct_batch_opening import (
        BatchOpeningCollectorV3,
    )
    from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
        commit_succinct_column_group_v3,
    )

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
    commitments = tuple(committed[tag].tree.commitment for tag in _COLUMN_TAGS)
    variable_count = statement.cell_count().bit_length() - 1
    z_point = derive_tile_eq_point_v3(
        tile_digest, commitments, validator_nonce, variable_count)
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
                z_point=z_point, validator_nonce=validator_nonce,
                fused=fused, collector=collector)
            by_tag.update(zip(member_tags, batch, strict=True))
        eq_folds = tuple(by_tag[tag] for tag in _COLUMN_TAGS)
    else:
        eq_folds = tuple(
            prove_succinct_eq_fold_v3(
                tile_digest=tile_digest, column=committed[tag],
                z_point=z_point,
                validator_nonce=validator_nonce, fused=fused,
                collector=collector)
            for tag in _COLUMN_TAGS
        )
    prod_statement = GoldilocksSuccinctProductStatementV3(
        validator_binding_digest=hashlib.sha256(
            tile_digest + b"prod/s*u").digest(),
        variable_count=variable_count,
        factor_component_sizes=tuple(
            2 for _ in range(variable_count)),
    )
    # tensor components are MSB-first; the eq table is LSB-first
    eq_components = tuple(
        ((1 - z) % GOLDILOCKS_MODULUS, z) for z in reversed(z_point)
    )
    if fused is not None:
        from verallm.proof_v3.native_pcs_backend import (
            fused_prove_goldilocks_succinct_product_v3,
        )

        product_proof = fused_prove_goldilocks_succinct_product_v3(
            fold_extension=fused[0], tree_extension=fused[1],
            statement=prod_statement,
            a_column=committed["s"], b_column=committed["u8"],
            factor_components=eq_components,
            validator_nonce=validator_nonce,
            collector=collector, a_tag="s", b_tag="u8")
    else:
        product_proof = prove_goldilocks_succinct_product_v3(
            statement=prod_statement,
            a_pcs_statement=committed["s"].pcs_statement,
            b_pcs_statement=committed["u8"].pcs_statement,
            a_tree=committed["s"].tree,
            b_tree=committed["u8"].tree,
            a_evaluations=committed["s"].values,
            b_evaluations=committed["u8"].values,
            factor_components=eq_components,
            validator_nonce=validator_nonce,
            collector=collector, a_tag="s", b_tag="u8",
        )
    biased_vars = variable_count + _block_bits(2)
    logup_instances = []
    for name, table, column_tag, wvars in (
        ("silu", statement.silu_logup_statement().table, "W_silu",
         variable_count),
        ("clamp", statement.clamp_logup_statement().table, "W_clamp",
         variable_count),
        ("range", tuple(range(statement.divisor)), "rem16",
         variable_count),
        ("biased8", tuple(range(256)), "grp_biased8", biased_vars),
    ):
        column = (
            groups[column_tag] if column_tag.startswith("grp_")
            else committed[column_tag])
        logup_statement = GoldilocksSuccinctLogupStatementV3(
            validator_binding_digest=hashlib.sha256(
                tile_digest + b"logup/" + name.encode()).digest(),
            table=table,
            witness_variable_count=wvars,
            witness_binding_override=(
                column.pcs_statement.validator_binding_digest),
        )
        logup_instances.append(
            (name, logup_statement, column, column_tag))
    logups = {}
    if fused is not None:
        from verallm.proof_v3.native_pcs_backend import (
            fused_prove_logup_batch_v3,
        )

        proofs = fused_prove_logup_batch_v3(
            fold_extension=fused[0], tree_extension=fused[1],
            tile_digest=tile_digest,
            instances=tuple(
                (st, col, f"logup/{name}", ctag)
                for name, st, col, ctag in logup_instances),
            validator_nonce=validator_nonce, collector=collector)
        for (name, _st, _col, _ct), pf in zip(
            logup_instances, proofs, strict=True
        ):
            logups[name] = pf
    else:
        for name, logup_statement, column, column_tag in logup_instances:
            logups[name] = prove_goldilocks_succinct_logup_v3(
                statement=logup_statement,
                looked_up_values=column.values,
                validator_nonce=validator_nonce,
                witness_tree=column.tree,
                collector=collector, tag_prefix=f"logup/{name}",
                witness_tag=column_tag)
    batch_openings = tuple(sorted(collector.prove_all(
        validator_nonce=validator_nonce, fused=fused).items()))
    proof = GoldilocksSuccinctMlpProofV3(
        column_commitments=commitments,
        eq_folds=eq_folds,
        product_proof=product_proof,
        silu_logup=logups["silu"],
        clamp_logup=logups["clamp"],
        byte_logup=logups["range"],
        g8_range_logup=logups["biased8"],
        u8_range_logup=None,
        batch_openings=batch_openings,
    )
    return proof, outputs


def verify_goldilocks_succinct_mlp_v3(
    proof: object,
    *,
    statement: GoldilocksMlpTileStatementV3,
    validator_nonce: bytes,
) -> None:
    """Succinct CPU verification of the whole SwiGLU tile."""

    try:
        if not isinstance(proof, GoldilocksSuccinctMlpProofV3):
            raise ProofV3VerificationError("succinct mlp proof type is wrong")
        tile_digest = _tile_digest(statement)
        from verallm.proof_v3.goldilocks_succinct_batch_opening import (
            BatchClaimCheckerV3,
        )

        checker = BatchClaimCheckerV3()
        variable_count = statement.cell_count().bit_length() - 1
        for group_tag, member_tags in _GROUP_PLAN:
            bits = _block_bits(len(member_tags))
            for index, tag in enumerate(member_tags):
                checker.alias(tag, group_tag, tuple(
                    (index >> j) & 1 for j in range(bits)))
        if len(proof.column_commitments) != len(_COLUMN_TAGS) or len(
            proof.eq_folds
        ) != len(_COLUMN_TAGS):
            raise ProofV3VerificationError("succinct mlp shape is wrong")
        z_point = derive_tile_eq_point_v3(
            tile_digest, tuple(proof.column_commitments), validator_nonce,
            variable_count)
        folds = {}
        for tag, commitment, fold_proof in zip(
            _COLUMN_TAGS, proof.column_commitments, proof.eq_folds,
            strict=True,
        ):
            folds[tag] = verify_succinct_eq_fold_v3(
                fold_proof,
                tile_digest=tile_digest,
                tag=tag,
                pcs_statement=column_pcs_statement_v3(
                    tile_digest, tag, variable_count),
                commitment=commitment,
                z_point=z_point,
                validator_nonce=validator_nonce,
                checker=checker,
            )
        prod_statement = GoldilocksSuccinctProductStatementV3(
            validator_binding_digest=hashlib.sha256(
                tile_digest + b"prod/s*u").digest(),
            variable_count=variable_count,
            factor_component_sizes=tuple(2 for _ in range(variable_count)),
        )
        eq_components = tuple(
            ((1 - z) % GOLDILOCKS_MODULUS, z) for z in reversed(z_point)
        )
        tags = dict(zip(_COLUMN_TAGS, proof.column_commitments, strict=True))
        for group_tag, member_tags in _GROUP_PLAN:
            if len({tags[tag] for tag in member_tags}) != 1:
                raise ProofV3VerificationError(
                    "succinct mlp group roots disagree")
        product_claim = proof.product_proof.claimed_sum % GOLDILOCKS_MODULUS
        verify_goldilocks_succinct_product_v3(
            proof.product_proof,
            statement=prod_statement,
            a_pcs_statement=column_pcs_statement_v3(
                tile_digest, "s", variable_count),
            b_pcs_statement=column_pcs_statement_v3(
                tile_digest, "u8", variable_count),
            a_commitment=tags["s"],
            b_commitment=tags["u8"],
            factor_components=eq_components,
            validator_nonce=validator_nonce,
            expected_sum=product_claim,
            checker=checker, a_tag="s", b_tag="u8",
        )
        c_silu, c_clamp, c_euclid = _constants(statement)
        # pack-silu: F(W_silu) - F(g8) - 2^32 F(s) == C_silu
        lhs = (
            folds["W_silu"] - folds["g8"] - _SPACK * folds["s"]
        ) % GOLDILOCKS_MODULUS
        if lhs != c_silu:
            raise ProofV3VerificationError(
                "succinct mlp silu packing coupling fails")
        # pack-clamp: F(W_clamp) - F(h_q) - 2^32 F(h8) == C_clamp
        lhs = (
            folds["W_clamp"] - folds["h_q"] - _SPACK * folds["h8"]
        ) % GOLDILOCKS_MODULUS
        if lhs != c_clamp:
            raise ProofV3VerificationError(
                "succinct mlp clamp packing coupling fails")
        # euclid: D F(h_q) + F(rem16) - P(s,u8) == D//2
        lhs = (
            statement.divisor * folds["h_q"]
            + folds["rem16"]
            - product_claim
        ) % GOLDILOCKS_MODULUS
        if lhs != c_euclid:
            raise ProofV3VerificationError(
                "succinct mlp Euclidean coupling fails")
        # LogUps: witness commitment MUST equal the tile column root.
        # int8 bias couplings pin g8/u8 ranges (packing anti-aliasing)
        for biased_tag, base_tag in (("g8_biased", "g8"), ("u8_biased", "u8")):
            if (folds[biased_tag] - folds[base_tag]) % GOLDILOCKS_MODULUS != 128:
                raise ProofV3VerificationError(
                    "succinct mlp int8 bias coupling fails")
        biased_vars = variable_count + _block_bits(2)

        def _witness_root(column_tag: str) -> bytes:
            if column_tag.startswith("grp_"):
                return tags[dict(_GROUP_PLAN)[column_tag][0]]
            return tags[column_tag]

        def _wvars(column_tag: str) -> int:
            return biased_vars if column_tag == "grp_biased8" else (
                variable_count)

        logup_specs = (
            ("silu", statement.silu_logup_statement().table, "W_silu",
             proof.silu_logup),
            ("clamp", statement.clamp_logup_statement().table, "W_clamp",
             proof.clamp_logup),
            ("range", tuple(range(statement.divisor)), "rem16",
             proof.byte_logup),
            ("biased8", tuple(range(256)), "grp_biased8",
             proof.g8_range_logup),
        )
        grouped_aux = checker is not None and any(
            tag.startswith("logup_aux/")
            for tag, _p in proof.batch_openings)
        aux_statements_grouped: dict = {}
        aux_commitments_grouped: dict = {}
        if grouped_aux:
            from verallm.proof_v3.native_pcs_backend import (
                logup_aux_group_plan_v3,
            )

            shapes = tuple(
                (f"logup/{name}", _wvars(column_tag),
                 GoldilocksSuccinctLogupStatementV3(
                     validator_binding_digest=hashlib.sha256(
                         tile_digest + b"logup/" + name.encode()
                     ).digest(),
                     table=table,
                     witness_variable_count=_wvars(column_tag),
                 ).table_variable_count)
                for name, table, column_tag, _lp in logup_specs)
            plans, group_meta = logup_aux_group_plan_v3(shapes)
            proof_by_prefix = {
                f"logup/{name}": lp
                for name, _t, _c, lp in logup_specs
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
                            "succinct mlp aux group roots disagree")
        for name, table, column_tag, logup_proof in logup_specs:
            logup_statement = GoldilocksSuccinctLogupStatementV3(
                validator_binding_digest=hashlib.sha256(
                    tile_digest + b"logup/" + name.encode()).digest(),
                table=table,
                witness_variable_count=_wvars(column_tag),
                witness_binding_override=column_pcs_statement_v3(
                    tile_digest, column_tag, _wvars(column_tag)
                ).validator_binding_digest,
            )
            verify_goldilocks_succinct_logup_v3(
                logup_proof, statement=logup_statement,
                witness_commitment=_witness_root(column_tag),
                validator_nonce=validator_nonce,
                checker=checker, tag_prefix=f"logup/{name}",
                witness_tag=column_tag)
        if checker is not None:
            from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (  # noqa: E501
                logup_batch_registry_v3,
            )

            statements = {}
            commitments = {}
            for group_tag, member_tags in _GROUP_PLAN:
                statements[group_tag] = column_pcs_statement_v3(
                    tile_digest, group_tag,
                    variable_count + _block_bits(len(member_tags)))
                commitments[group_tag] = tags[member_tags[0]]
            statements.update(aux_statements_grouped)
            commitments.update(aux_commitments_grouped)
            for name, table, column_tag, logup_proof in logup_specs:
                logup_statement = GoldilocksSuccinctLogupStatementV3(
                    validator_binding_digest=hashlib.sha256(
                        tile_digest + b"logup/" + name.encode()).digest(),
                    table=table,
                    witness_variable_count=_wvars(column_tag),
                    witness_binding_override=column_pcs_statement_v3(
                        tile_digest, column_tag, _wvars(column_tag)
                    ).validator_binding_digest,
                )
                if not grouped_aux:
                    aux_statements, aux_commitments = (
                        logup_batch_registry_v3(
                            logup_proof, logup_statement, f"logup/{name}",
                            witness_tag=column_tag))
                    statements.update(aux_statements)
                    commitments.update(aux_commitments)
            checker.verify_all(
                dict(proof.batch_openings),
                statements=statements,
                commitments=commitments,
                validator_nonce=validator_nonce)
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "succinct mlp proof is malformed") from exc
