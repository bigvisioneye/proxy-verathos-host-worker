"""Exact single-head causal attention composition reference for proof-v3.

Capstone of the relation verticals: one attention head over one chunk is
proven end to end in exact fixed point by composing the shipped
primitives, with no sampled replay anywhere:

1. **Scores**: the frozen raw-score table ``s[t,k] = sum_d Q[t,d] K[k,d]``
   is proven against the frozen Q/K tables by folded-scalar equality: the
   verifier folds the opened score table with nonce coefficients
   ``v[t] u[k]`` and requires the two-table product sumcheck over the
   broadcast (t,k,d) cube to hit the same scalar.
2. **Causal mask + quantization**: each unmasked ``(raw, quantized)`` pair
   is proven a member of the validator-owned quantization table via LogUp
   (packed pairs).  Masked positions (k > t) never enter a softmax tile at
   all — masking is structural, not a witness.
3. **Softmax**: each query row is an exact fixed-point softmax tile
   (existing module) over its unmasked quantized scores.
4. **Output**: the frozen output table ``o[t,d] = sum_k P[t,k] V[k,d]``
   is proven against the softmax probabilities and frozen V by a second
   folded product sumcheck, same shape as step 1.

RoPE note: rotation is a public per-position linear map; in the native
relation it composes into the step-1 factor and costs no witness.  This
reference proves the head without RoPE; adding it changes only the public
factor derivation, not the protocol shape.  GQA is a layout contract in
the broadcast flattening.

All tables freeze pre-nonce; every challenge (fold coefficients, sumcheck
rounds, LogUp challenges) derives post-freeze.  Reference verification
opens tables in full; the crypto-critical steps (sumchecks, LogUp,
softmax tiles) run the real machinery.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_fold_sumcheck_reference import _challenge
from verallm.proof_v3.goldilocks_linear_relation_reference import (
    _fixed32,
    _integer,
    _u32,
)
from verallm.proof_v3.goldilocks_logup_reference import (
    GoldilocksLogupProofV3,
    GoldilocksLogupStatementV3,
    freeze_goldilocks_logup_witness_v3,
    verify_goldilocks_logup_reference_v3,
)
from verallm.proof_v3.goldilocks_merkle_reference import (
    GoldilocksMerkleTreeReference,
)
from verallm.proof_v3.goldilocks_product_sumcheck_reference import (
    GoldilocksProductSumcheckProofV3,
    commit_goldilocks_product_sumcheck_a_v3,
    commit_goldilocks_product_sumcheck_b_v3,
    prove_goldilocks_product_sumcheck_v3,
    verify_goldilocks_product_sumcheck_v3,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.goldilocks_softmax_tile_reference import (
    GoldilocksSoftmaxTileProofV3,
    GoldilocksSoftmaxTileStatementV3,
    freeze_goldilocks_softmax_tile_v3,
    prove_goldilocks_softmax_tile_v3,
    verify_goldilocks_softmax_tile_v3,
    GOLDILOCKS_SOFTMAX_SCALE_V3,
)


GOLDILOCKS_ATTENTION_HEAD_ABI_V3: Final = "goldilocks.attention_head.reference.v1"
MAX_GOLDILOCKS_ATTENTION_TOKENS_V3: Final = 32
MAX_GOLDILOCKS_ATTENTION_DIM_V3: Final = 64
_QPACK: Final = 1 << 32

_STATEMENT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_ATTENTION/V1/STATEMENT/SHA256"
)
_TABLES_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_ATTENTION/V1/TABLES/SHA256"
_QUANT_BINDING_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_ATTENTION/V1/QUANT/SHA256"
)
_FOLD_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_ATTENTION/V1/FOLD/SHA256"
_ROW_NONCE_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_ATTENTION/V1/ROW/SHA256"


def _signed(value: object, name: str, bits: int) -> int:
    integer = _integer(value, name)
    bound = 1 << (bits - 1)
    if not -bound <= integer < bound:
        raise ProofV3Error(f"{name} must be a signed {bits}-bit integer")
    return integer


@dataclass(frozen=True, slots=True)
class GoldilocksAttentionHeadStatementV3:
    """Validator-owned head statement: shape and both nonlinear tables.

    ``quant_table[raw_offset]`` maps a clamped raw score (offset by
    ``quant_offset`` into ``[0, len)``) to a softmax input score in
    ``[0, 2^16)``.  ``exp_table`` is the signed softmax semantics.
    """

    validator_binding_digest: bytes
    token_count: int
    head_dim: int
    quant_offset: int
    quant_table: tuple[int, ...]
    exp_table: tuple[int, ...]

    def __post_init__(self) -> None:
        binding = _fixed32(
            self.validator_binding_digest,
            "attention validator_binding_digest",
            nonzero=True,
        )
        tokens = _u32(self.token_count, "token_count", positive=True)
        dim = _u32(self.head_dim, "head_dim", positive=True)
        if tokens > MAX_GOLDILOCKS_ATTENTION_TOKENS_V3:
            raise ProofV3Error("attention token count exceeds the CPU cap")
        if dim > MAX_GOLDILOCKS_ATTENTION_DIM_V3 or dim & (dim - 1):
            raise ProofV3Error("attention head dim must be a small power of two")
        offset = _integer(self.quant_offset, "quant_offset")
        if not isinstance(self.quant_table, tuple) or not self.quant_table:
            raise ProofV3Error("attention quant table is malformed")
        quant = tuple(
            _u32(value, f"quant_table[{index}]")
            for index, value in enumerate(self.quant_table)
        )
        if any(value >= 1 << 16 for value in quant):
            raise ProofV3Error("attention quant output exceeds the score window")
        if not isinstance(self.exp_table, tuple) or len(self.exp_table) != 1 << 16:
            raise ProofV3Error("attention exp table must cover every score")
        object.__setattr__(self, "validator_binding_digest", binding)
        object.__setattr__(self, "token_count", tokens)
        object.__setattr__(self, "head_dim", dim)
        object.__setattr__(self, "quant_offset", offset)
        object.__setattr__(self, "quant_table", quant)

    def digest(self) -> bytes:
        return hashlib.sha256(
            _STATEMENT_DOMAIN
            + self.validator_binding_digest
            + struct.pack("<IIq", self.token_count, self.head_dim, self.quant_offset)
            + hashlib.sha256(
                b"".join(v.to_bytes(8, "little") for v in self.quant_table)
            ).digest()
            + hashlib.sha256(
                b"".join(
                    _integer(v, "exp value").to_bytes(8, "little")
                    for v in self.exp_table
                )
            ).digest()
        ).digest()

    def tables_binding_digest(self) -> bytes:
        return hashlib.sha256(_TABLES_DOMAIN + self.digest()).digest()

    def quant_logup_statement(self) -> GoldilocksLogupStatementV3:
        return GoldilocksLogupStatementV3(
            validator_binding_digest=hashlib.sha256(
                _QUANT_BINDING_DOMAIN + self.digest()
            ).digest(),
            table=tuple(
                index + _QPACK * value
                for index, value in enumerate(self.quant_table)
            ),
        )

    def softmax_statement(
        self, *, row_index: int
    ) -> GoldilocksSoftmaxTileStatementV3:
        return GoldilocksSoftmaxTileStatementV3(
            validator_binding_digest=hashlib.sha256(
                _ROW_NONCE_DOMAIN + self.digest() + struct.pack("<I", row_index)
            ).digest(),
            tile_length=row_index + 1,
            exp_table=self.exp_table,
        )

    def quantize(self, raw_score: int) -> int:
        index = raw_score + self.quant_offset
        index = min(max(index, 0), len(self.quant_table) - 1)
        return self.quant_table[index]

    def clamped_index(self, raw_score: int) -> int:
        return min(max(raw_score + self.quant_offset, 0), len(self.quant_table) - 1)


def _pad_pow2(values: list[int]) -> tuple[int, ...]:
    length = 1 << max(1, (len(values) - 1).bit_length())
    return tuple(values) + (0,) * (length - len(values))


def _fold_coefficients(
    *, statement_digest: bytes, roots: tuple[bytes, ...], validator_nonce: bytes,
    label: bytes, count: int,
) -> tuple[int, ...]:
    seed = hashlib.sha256(
        _FOLD_DOMAIN
        + statement_digest
        + b"".join(roots)
        + _fixed32(validator_nonce, "validator_nonce")
        + label
    ).digest()
    return tuple(_challenge(seed, index + 1) for index in range(count))


@dataclass(frozen=True, slots=True)
class GoldilocksAttentionHeadProofV3:
    tables_opening: tuple[tuple[int, ...], ...]
    score_sumcheck: GoldilocksProductSumcheckProofV3
    output_sumcheck: GoldilocksProductSumcheckProofV3
    quant_proof: GoldilocksLogupProofV3
    quant_roots: tuple[bytes, bytes]
    softmax_proofs: tuple[GoldilocksSoftmaxTileProofV3, ...]
    softmax_roots: tuple[tuple[bytes, bytes, bytes, bytes, bytes], ...]


@dataclass(frozen=True, slots=True)
class GoldilocksAttentionHeadWitnessV3:
    statement: GoldilocksAttentionHeadStatementV3
    tables_tree: GoldilocksMerkleTreeReference
    q: tuple[tuple[int, ...], ...]
    k: tuple[tuple[int, ...], ...]
    value_table: tuple[tuple[int, ...], ...]


def freeze_goldilocks_attention_head_v3(
    *,
    statement: GoldilocksAttentionHeadStatementV3,
    q: tuple[tuple[int, ...], ...],
    k: tuple[tuple[int, ...], ...],
    value_table: tuple[tuple[int, ...], ...],
    raw_scores: tuple[tuple[int, ...], ...],
    probs: tuple[tuple[int, ...], ...],
    outputs: tuple[tuple[int, ...], ...],
) -> GoldilocksAttentionHeadWitnessV3:
    """Freeze every head table pre-nonce.  Semantics are not checked."""

    tokens, dim = statement.token_count, statement.head_dim
    for name, table, cols in (
        ("q", q, dim), ("k", k, dim), ("v", value_table, dim),
        ("raw_scores", raw_scores, tokens),
        ("probs", probs, tokens), ("outputs", outputs, dim),
    ):
        if len(table) != tokens or any(len(row) != cols for row in table):
            raise ProofV3Error(f"attention {name} table has an unexpected shape")
    rows: list[tuple[int, ...]] = []
    for t in range(tokens):
        rows.append(
            tuple(_signed(v, "q", 16) % GOLDILOCKS_MODULUS for v in q[t])
            + tuple(_signed(v, "k", 16) % GOLDILOCKS_MODULUS for v in k[t])
            + tuple(_signed(v, "v", 16) % GOLDILOCKS_MODULUS for v in value_table[t])
            + tuple(
                _signed(v, "raw", 48) % GOLDILOCKS_MODULUS for v in raw_scores[t]
            )
            + tuple(_integer(v, "prob") % GOLDILOCKS_MODULUS for v in probs[t])
            + tuple(_integer(v, "out") % GOLDILOCKS_MODULUS for v in outputs[t])
        )
    width = len(rows[0])
    padded = 1 << max(1, (len(rows) - 1).bit_length())
    rows.extend((0,) * width for _ in range(padded - len(rows)))
    tables_tree = GoldilocksMerkleTreeReference.from_rows(
        tuple(rows),
        binding_digest=statement.tables_binding_digest(),
    )
    return GoldilocksAttentionHeadWitnessV3(
        statement=statement,
        tables_tree=tables_tree,
        q=q,
        k=k,
        value_table=value_table,
    )


def _broadcast_qk(
    statement: GoldilocksAttentionHeadStatementV3,
    left: tuple[tuple[int, ...], ...],
    right: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    tokens, dim = statement.token_count, statement.head_dim
    a, b = [], []
    for t in range(tokens):
        for s in range(tokens):
            for d in range(dim):
                a.append(_integer(left[t][d], "left") % GOLDILOCKS_MODULUS)
                b.append(_integer(right[s][d], "right") % GOLDILOCKS_MODULUS)
    return _pad_pow2(a), _pad_pow2(b)


def _qk_factor(
    statement: GoldilocksAttentionHeadStatementV3,
    v_coeff: tuple[int, ...],
    u_coeff: tuple[int, ...],
    length: int,
) -> tuple[int, ...]:
    tokens, dim = statement.token_count, statement.head_dim
    factor = []
    for t in range(tokens):
        for s in range(tokens):
            for _ in range(dim):
                factor.append(v_coeff[t] * u_coeff[s] % GOLDILOCKS_MODULUS)
    factor.extend(0 for _ in range(length - len(factor)))
    return tuple(factor)


def prove_goldilocks_attention_head_v3(
    *,
    witness: GoldilocksAttentionHeadWitnessV3,
    raw_scores: tuple[tuple[int, ...], ...],
    probs: tuple[tuple[int, ...], ...],
    validator_nonce: bytes,
) -> GoldilocksAttentionHeadProofV3:
    statement = witness.statement
    tokens, dim = statement.token_count, statement.head_dim
    statement_digest = statement.digest()
    tables_root = witness.tables_tree.commitment

    # Step 1: score-side product sumcheck over the (t,s,d) cube.
    a_vals, b_vals = _broadcast_qk(statement, witness.q, witness.k)
    a_tree = commit_goldilocks_product_sumcheck_a_v3(
        statement_digest=statement_digest, evaluations=a_vals
    )
    b_tree = commit_goldilocks_product_sumcheck_b_v3(
        statement_digest=statement_digest, evaluations=b_vals
    )
    v_coeff = _fold_coefficients(
        statement_digest=statement_digest,
        roots=(tables_root, a_tree.commitment, b_tree.commitment),
        validator_nonce=validator_nonce,
        label=b"score-v",
        count=tokens,
    )
    u_coeff = _fold_coefficients(
        statement_digest=statement_digest,
        roots=(tables_root, a_tree.commitment, b_tree.commitment),
        validator_nonce=validator_nonce,
        label=b"score-u",
        count=tokens,
    )
    factor = _qk_factor(statement, v_coeff, u_coeff, len(a_vals))
    score_sumcheck = prove_goldilocks_product_sumcheck_v3(
        statement_digest=statement_digest,
        a_tree=a_tree,
        b_tree=b_tree,
        a_evaluations=a_vals,
        b_evaluations=b_vals,
        factor=factor,
        validator_nonce=validator_nonce,
    )

    # Step 2: quantization pair lookup over unmasked positions.
    quant_pairs = [
        statement.clamped_index(raw_scores[t][s])
        + _QPACK * statement.quantize(raw_scores[t][s])
        for t in range(tokens)
        for s in range(t + 1)
    ]
    quant_statement = statement.quant_logup_statement()
    quant_w_tree, quant_m_tree = freeze_goldilocks_logup_witness_v3(
        statement=quant_statement,
        looked_up_values=tuple(quant_pairs),
    )
    quant_proof = GoldilocksLogupProofV3(
        witness_opening=tuple(row[0] for row in quant_w_tree.rows),
        multiplicity_opening=tuple(row[0] for row in quant_m_tree.rows),
    )

    # Step 3: per-row softmax tiles over the quantized unmasked scores.
    softmax_proofs = []
    softmax_roots = []
    for t in range(tokens):
        row_statement = statement.softmax_statement(row_index=t)
        scores_row = tuple(
            statement.quantize(raw_scores[t][s]) for s in range(t + 1)
        )
        exps_row = tuple(statement.exp_table[score] for score in scores_row)
        outputs_row = tuple(probs[t][s] for s in range(t + 1))
        tile = freeze_goldilocks_softmax_tile_v3(
            statement=row_statement,
            scores=scores_row,
            exps=exps_row,
            outputs=outputs_row,
        )
        softmax_proofs.append(prove_goldilocks_softmax_tile_v3(witness=tile))
        softmax_roots.append(
            (
                tile.columns_tree.commitment,
                tile.pair_witness_tree.commitment,
                tile.pair_multiplicity_tree.commitment,
                tile.byte_witness_tree.commitment,
                tile.byte_multiplicity_tree.commitment,
            )
        )

    # Step 4: output-side product sumcheck (P x V over the (t,s,d) cube;
    # masked probabilities are structurally zero).
    p_full = [
        [probs[t][s] if s <= t else 0 for s in range(tokens)]
        for t in range(tokens)
    ]
    pa_vals, pb_vals = [], []
    for t in range(tokens):
        for s in range(tokens):
            for d in range(dim):
                pa_vals.append(_integer(p_full[t][s], "prob") % GOLDILOCKS_MODULUS)
                pb_vals.append(
                    _integer(witness.value_table[s][d], "v") % GOLDILOCKS_MODULUS
                )
    pa_vals, pb_vals = _pad_pow2(pa_vals), _pad_pow2(pb_vals)
    pa_tree = commit_goldilocks_product_sumcheck_a_v3(
        statement_digest=hashlib.sha256(statement_digest + b"PV").digest(),
        evaluations=pa_vals,
    )
    pb_tree = commit_goldilocks_product_sumcheck_b_v3(
        statement_digest=hashlib.sha256(statement_digest + b"PV").digest(),
        evaluations=pb_vals,
    )
    ov_coeff = _fold_coefficients(
        statement_digest=statement_digest,
        roots=(tables_root, pa_tree.commitment, pb_tree.commitment),
        validator_nonce=validator_nonce,
        label=b"out-v",
        count=tokens,
    )
    ow_coeff = _fold_coefficients(
        statement_digest=statement_digest,
        roots=(tables_root, pa_tree.commitment, pb_tree.commitment),
        validator_nonce=validator_nonce,
        label=b"out-w",
        count=dim,
    )
    out_factor = []
    for t in range(tokens):
        for _s in range(tokens):
            for d in range(dim):
                out_factor.append(ov_coeff[t] * ow_coeff[d] % GOLDILOCKS_MODULUS)
    out_factor.extend(0 for _ in range(len(pa_vals) - len(out_factor)))
    output_sumcheck = prove_goldilocks_product_sumcheck_v3(
        statement_digest=hashlib.sha256(statement_digest + b"PV").digest(),
        a_tree=pa_tree,
        b_tree=pb_tree,
        a_evaluations=pa_vals,
        b_evaluations=pb_vals,
        factor=tuple(out_factor),
        validator_nonce=validator_nonce,
    )
    return GoldilocksAttentionHeadProofV3(
        tables_opening=tuple(tuple(row) for row in witness.tables_tree.rows),
        score_sumcheck=score_sumcheck,
        output_sumcheck=output_sumcheck,
        quant_proof=quant_proof,
        quant_roots=(quant_w_tree.commitment, quant_m_tree.commitment),
        softmax_proofs=tuple(softmax_proofs),
        softmax_roots=tuple(softmax_roots),
    )


def verify_goldilocks_attention_head_v3(
    proof: object,
    *,
    statement: GoldilocksAttentionHeadStatementV3,
    tables_root: bytes,
    score_a_root: bytes,
    score_b_root: bytes,
    output_a_root: bytes,
    output_b_root: bytes,
    validator_nonce: bytes,
) -> None:
    """Verify one exact attention head against the frozen roots."""

    try:
        if not isinstance(proof, GoldilocksAttentionHeadProofV3):
            raise ProofV3VerificationError("attention proof type is unexpected")
        tokens, dim = statement.token_count, statement.head_dim
        statement_digest = statement.digest()
        rows = proof.tables_opening
        rebuilt = GoldilocksMerkleTreeReference.from_rows(
            tuple(tuple(row) for row in rows),
            binding_digest=statement.tables_binding_digest(),
        )
        if rebuilt.commitment != tables_root:
            raise ProofV3VerificationError(
                "attention tables opening does not match the frozen root"
            )
        active = rows[:tokens]
        q = [row[:dim] for row in active]
        k = [row[dim : 2 * dim] for row in active]
        value_table = [row[2 * dim : 3 * dim] for row in active]
        raw = [row[3 * dim : 3 * dim + tokens] for row in active]
        probs = [row[3 * dim + tokens : 3 * dim + 2 * tokens] for row in active]
        outputs = [row[3 * dim + 2 * tokens :] for row in active]

        # Step 1: fold the opened raw scores and require the sumcheck to
        # reach the same scalar over the frozen Q/K broadcast tables.
        v_coeff = _fold_coefficients(
            statement_digest=statement_digest,
            roots=(tables_root, score_a_root, score_b_root),
            validator_nonce=validator_nonce,
            label=b"score-v",
            count=tokens,
        )
        u_coeff = _fold_coefficients(
            statement_digest=statement_digest,
            roots=(tables_root, score_a_root, score_b_root),
            validator_nonce=validator_nonce,
            label=b"score-u",
            count=tokens,
        )
        folded_scores = 0
        for t in range(tokens):
            for s in range(tokens):
                folded_scores = (
                    folded_scores
                    + v_coeff[t] * u_coeff[s] % GOLDILOCKS_MODULUS * raw[t][s]
                ) % GOLDILOCKS_MODULUS
        cube = tokens * tokens * dim
        length = 1 << max(1, (cube - 1).bit_length())
        factor = _qk_factor(statement, v_coeff, u_coeff, length)
        verify_goldilocks_product_sumcheck_v3(
            proof.score_sumcheck,
            statement_digest=statement_digest,
            a_commitment=score_a_root,
            b_commitment=score_b_root,
            factor=factor,
            validator_nonce=validator_nonce,
            expected_sum=folded_scores,
        )
        # Bind the sumcheck's opened broadcast tables to the head tables.
        a_open, b_open = proof.score_sumcheck.a_full_opening, (
            proof.score_sumcheck.b_full_opening
        )
        index = 0
        for t in range(tokens):
            for s in range(tokens):
                for d in range(dim):
                    if a_open[index] != q[t][d] or b_open[index] != k[s][d]:
                        raise ProofV3VerificationError(
                            "attention broadcast tables do not match Q/K"
                        )
                    index += 1

        # Step 2: quantization pairs (unmasked positions only).
        expected_pairs = []
        for t in range(tokens):
            for s in range(t + 1):
                raw_signed = raw[t][s]
                if raw_signed >= GOLDILOCKS_MODULUS // 2:
                    raw_signed -= GOLDILOCKS_MODULUS
                expected_pairs.append(
                    statement.clamped_index(raw_signed)
                    + _QPACK * statement.quantize(raw_signed)
                )
        quant_statement = statement.quant_logup_statement()
        opening = proof.quant_proof.witness_opening
        if tuple(opening[: len(expected_pairs)]) != tuple(expected_pairs):
            raise ProofV3VerificationError(
                "attention quantization witness does not match the scores"
            )
        if any(
            value != quant_statement.table[0]
            for value in opening[len(expected_pairs) :]
        ):
            raise ProofV3VerificationError(
                "attention quantization padding is malformed"
            )
        verify_goldilocks_logup_reference_v3(
            proof.quant_proof,
            statement=quant_statement,
            witness_root=proof.quant_roots[0],
            multiplicity_root=proof.quant_roots[1],
            validator_nonce=validator_nonce,
        )

        # Step 3: exact softmax per row; tile columns must match the
        # quantized scores and the frozen probabilities.
        if len(proof.softmax_proofs) != tokens or len(proof.softmax_roots) != tokens:
            raise ProofV3VerificationError("attention softmax row set is wrong")
        for t in range(tokens):
            row_statement = statement.softmax_statement(row_index=t)
            tile_proof = proof.softmax_proofs[t]
            roots = proof.softmax_roots[t]
            for s in range(t + 1):
                raw_signed = raw[t][s]
                if raw_signed >= GOLDILOCKS_MODULUS // 2:
                    raw_signed -= GOLDILOCKS_MODULUS
                row = tile_proof.columns_opening[s]
                if row[0] != statement.quantize(raw_signed):
                    raise ProofV3VerificationError(
                        "attention softmax tile score does not match the head"
                    )
                if row[2] != probs[t][s]:
                    raise ProofV3VerificationError(
                        "attention softmax tile output does not match the head"
                    )
            verify_goldilocks_softmax_tile_v3(
                tile_proof,
                statement=row_statement,
                columns_root=roots[0],
                pair_witness_root=roots[1],
                pair_multiplicity_root=roots[2],
                byte_witness_root=roots[3],
                byte_multiplicity_root=roots[4],
                validator_nonce=validator_nonce,
            )
        # Masked probabilities must be structurally zero.
        for t in range(tokens):
            for s in range(t + 1, tokens):
                if probs[t][s] != 0:
                    raise ProofV3VerificationError(
                        "attention masked probability must be zero"
                    )

        # Step 4: output-side fold.
        ov_coeff = _fold_coefficients(
            statement_digest=statement_digest,
            roots=(tables_root, output_a_root, output_b_root),
            validator_nonce=validator_nonce,
            label=b"out-v",
            count=tokens,
        )
        ow_coeff = _fold_coefficients(
            statement_digest=statement_digest,
            roots=(tables_root, output_a_root, output_b_root),
            validator_nonce=validator_nonce,
            label=b"out-w",
            count=dim,
        )
        folded_outputs = 0
        for t in range(tokens):
            for d in range(dim):
                folded_outputs = (
                    folded_outputs
                    + ov_coeff[t] * ow_coeff[d] % GOLDILOCKS_MODULUS * outputs[t][d]
                ) % GOLDILOCKS_MODULUS
        out_factor = []
        for t in range(tokens):
            for _s in range(tokens):
                for d in range(dim):
                    out_factor.append(
                        ov_coeff[t] * ow_coeff[d] % GOLDILOCKS_MODULUS
                    )
        out_factor.extend(0 for _ in range(length - len(out_factor)))
        pv_digest = hashlib.sha256(statement_digest + b"PV").digest()
        verify_goldilocks_product_sumcheck_v3(
            proof.output_sumcheck,
            statement_digest=pv_digest,
            a_commitment=output_a_root,
            b_commitment=output_b_root,
            factor=tuple(out_factor),
            validator_nonce=validator_nonce,
            expected_sum=folded_outputs,
        )
        pa_open = proof.output_sumcheck.a_full_opening
        pb_open = proof.output_sumcheck.b_full_opening
        index = 0
        for t in range(tokens):
            for s in range(tokens):
                for d in range(dim):
                    expected_p = probs[t][s] if s <= t else 0
                    if pa_open[index] != expected_p or (
                        pb_open[index] != value_table[s][d]
                    ):
                        raise ProofV3VerificationError(
                            "attention output broadcast does not match P/V"
                        )
                    index += 1
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("attention proof is malformed") from exc


__all__ = [
    "GOLDILOCKS_ATTENTION_HEAD_ABI_V3",
    "GoldilocksAttentionHeadProofV3",
    "GoldilocksAttentionHeadStatementV3",
    "GoldilocksAttentionHeadWitnessV3",
    "freeze_goldilocks_attention_head_v3",
    "prove_goldilocks_attention_head_v3",
    "verify_goldilocks_attention_head_v3",
]
