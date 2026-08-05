"""End-to-end tiny-transformer proof reference for proof-v3.

THE E2E vertical: a complete forward pass is proven from public prompt
tokens to the returned token, in exact fixed point, with no sampled
replay.  Composition (one attention block, one decode step):

1. **Embedding**: ``x0[t] = E[token[t]]`` — prompt tokens and the signed
   embedding table are validator-owned, so the verifier checks the frozen
   ``x0`` cells directly.  No lookup argument needed: the index is public.
2. **Q/K/V projections**: three folded-linear checks ``P = x0 @ W_p``
   proven by the single-table fold sumcheck (weights signed, so the
   verifier computes ``W_p @ u`` itself); each committed wide result is
   requantized to int8 by the exact Euclidean + clamp-table step below.
3. **Attention head**: the existing exact head reference over the
   requantized Q/K/V (scores, quantize, softmax tiles, causal mask, PV).
4. **Residual + requantization**: ``r[t,d] = attn_out[t,d] +
   x0[t,d] * SCALE`` checked algebraically; then
   ``shifted = r + OFFSET``, ``q = shifted // SCALE`` with remainder in
   ``[0, SCALE)`` proven by byte limbs, and ``h8 = ClampTable[q]`` proven
   by packed-pair LogUp membership.
5. **LM head**: ``logits[j] = sum_d h8[last,d] * W_lm[d,j]`` proven by a
   fold sumcheck over the committed last hidden row.
6. **Sampler**: greedy argmax proven exactly — for the returned token
   ``j*`` every difference ``logit[j*] - logit[j] - [j < j*]`` is shown
   nonnegative by byte-limb decomposition.  Ties break to the lowest
   index, deterministically.

Every wide/int8 witness column freezes pre-nonce in one chain tree; all
fold coefficients, sumcheck rounds, and LogUp challenges derive
post-freeze.  An MLP block and GDN recurrence use exactly the machinery of
steps 2/4 (folded linear + table lookups) and are deliberately not
duplicated here; layer norm is deferred with the same status as RoPE (a
public-factor/table step, not new protocol).

The returned token is therefore bound to the prompt and the signed
weights by an unbroken exact chain: any single wrong cell anywhere breaks
one of the composed checks.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_attention_head_reference import (
    GoldilocksAttentionHeadProofV3,
    GoldilocksAttentionHeadStatementV3,
    freeze_goldilocks_attention_head_v3,
    prove_goldilocks_attention_head_v3,
    verify_goldilocks_attention_head_v3,
)
from verallm.proof_v3.goldilocks_fold_sumcheck_reference import (
    _challenge,
    commit_goldilocks_fold_sumcheck_x_v3,
    prove_goldilocks_fold_sumcheck_v3,
    verify_goldilocks_fold_sumcheck_v3,
    GoldilocksFoldSumcheckProofV3,
)
from verallm.proof_v3.goldilocks_linear_relation_reference import (
    _fixed32,
    _int8,
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
    commit_goldilocks_product_sumcheck_a_v3,
    commit_goldilocks_product_sumcheck_b_v3,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS


GOLDILOCKS_TINY_TRANSFORMER_ABI_V3: Final = (
    "goldilocks.tiny_transformer_e2e.reference.v1"
)
E2E_SCALE_V3: Final = 1 << 16
_CLAMP_PACK: Final = 1 << 32

_STATEMENT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_E2E/V1/STATEMENT/SHA256"
_CHAIN_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_E2E/V1/CHAIN/SHA256"
_CLAMP_BINDING_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_E2E/V1/CLAMP/SHA256"
_BYTE_BINDING_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_E2E/V1/BYTES/SHA256"
_FOLD_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_E2E/V1/FOLD/SHA256"


def _signed_field(value: int) -> int:
    return value % GOLDILOCKS_MODULUS


def _from_field_signed(value: int) -> int:
    return value - GOLDILOCKS_MODULUS if value >= GOLDILOCKS_MODULUS // 2 else value


def _pad_pow2(values: list[int]) -> tuple[int, ...]:
    length = 1 << max(1, (len(values) - 1).bit_length())
    return tuple(values) + (0,) * (length - len(values))


@dataclass(frozen=True, slots=True)
class GoldilocksTinyTransformerStatementV3:
    """Validator-owned E2E statement: prompt, weights, and all tables."""

    validator_binding_digest: bytes
    prompt_tokens: tuple[int, ...]
    vocab_size: int
    model_dim: int
    embedding: tuple[tuple[int, ...], ...]      # vocab x dim, int8
    w_q: tuple[tuple[int, ...], ...]            # dim x dim, int8
    w_k: tuple[tuple[int, ...], ...]
    w_v: tuple[tuple[int, ...], ...]
    w_lm: tuple[tuple[int, ...], ...]           # dim x vocab, int8
    clamp_table: tuple[int, ...]                # q-index -> int8 (+128)
    clamp_offset: int
    quant_table: tuple[int, ...]                # attention score quant
    quant_offset: int
    exp_table: tuple[int, ...]

    def __post_init__(self) -> None:
        binding = _fixed32(
            self.validator_binding_digest, "e2e binding", nonzero=True
        )
        vocab = _u32(self.vocab_size, "vocab_size", positive=True)
        dim = _u32(self.model_dim, "model_dim", positive=True)
        if dim & (dim - 1) or dim > 16 or vocab > 256:
            raise ProofV3Error("e2e shape exceeds the CPU reference cap")
        tokens = tuple(
            _u32(token, "prompt token") for token in self.prompt_tokens
        )
        if not tokens or len(tokens) > 8 or any(t >= vocab for t in tokens):
            raise ProofV3Error("e2e prompt is malformed")
        for name, matrix, rows, cols in (
            ("embedding", self.embedding, vocab, dim),
            ("w_q", self.w_q, dim, dim),
            ("w_k", self.w_k, dim, dim),
            ("w_v", self.w_v, dim, dim),
            ("w_lm", self.w_lm, dim, vocab),
        ):
            if len(matrix) != rows or any(len(row) != cols for row in matrix):
                raise ProofV3Error(f"e2e {name} has an unexpected shape")
            for row in matrix:
                for value in row:
                    _int8(value, f"e2e {name} value")
        if not isinstance(self.clamp_table, tuple) or not self.clamp_table:
            raise ProofV3Error("e2e clamp table is malformed")
        for value in self.clamp_table:
            if not 0 <= _integer(value, "clamp value") < 256:
                raise ProofV3Error("e2e clamp output must be int8 + 128")
        object.__setattr__(self, "validator_binding_digest", binding)
        object.__setattr__(self, "prompt_tokens", tokens)

    def digest(self) -> bytes:
        h = hashlib.sha256(_STATEMENT_DOMAIN + self.validator_binding_digest)
        h.update(struct.pack("<II", self.vocab_size, self.model_dim))
        h.update(struct.pack(f"<{len(self.prompt_tokens)}I", *self.prompt_tokens))
        for matrix in (self.embedding, self.w_q, self.w_k, self.w_v, self.w_lm):
            for row in matrix:
                h.update(b"".join(struct.pack("<b", value) for value in row))
        h.update(struct.pack("<qq", self.clamp_offset, self.quant_offset))
        for table in (self.clamp_table, self.quant_table):
            h.update(
                hashlib.sha256(
                    b"".join(
                        _integer(v, "table").to_bytes(8, "little") for v in table
                    )
                ).digest()
            )
        h.update(
            hashlib.sha256(
                b"".join(
                    _integer(v, "exp").to_bytes(8, "little")
                    for v in self.exp_table
                )
            ).digest()
        )
        return h.digest()

    def chain_binding_digest(self) -> bytes:
        return hashlib.sha256(_CHAIN_DOMAIN + self.digest()).digest()

    def clamp_logup_statement(self) -> GoldilocksLogupStatementV3:
        return GoldilocksLogupStatementV3(
            validator_binding_digest=hashlib.sha256(
                _CLAMP_BINDING_DOMAIN + self.digest()
            ).digest(),
            table=tuple(
                index + _CLAMP_PACK * value
                for index, value in enumerate(self.clamp_table)
            ),
        )

    def byte_logup_statement(self) -> GoldilocksLogupStatementV3:
        return GoldilocksLogupStatementV3(
            validator_binding_digest=hashlib.sha256(
                _BYTE_BINDING_DOMAIN + self.digest()
            ).digest(),
            table=tuple(range(256)),
        )

    def attention_statement(self) -> GoldilocksAttentionHeadStatementV3:
        return GoldilocksAttentionHeadStatementV3(
            validator_binding_digest=hashlib.sha256(
                b"E2E/ATTN" + self.digest()
            ).digest(),
            token_count=len(self.prompt_tokens),
            head_dim=self.model_dim,
            quant_offset=self.quant_offset,
            quant_table=self.quant_table,
            exp_table=self.exp_table,
        )

    def requantize(self, wide_value: int) -> tuple[int, int, int]:
        """Return (clamp_index, remainder, int8_value) for one wide value."""

        shifted = wide_value + self.clamp_offset
        if shifted < 0:
            raise ProofV3Error("e2e wide value underflows the clamp offset")
        quotient, remainder = divmod(shifted, E2E_SCALE_V3)
        index = min(quotient, len(self.clamp_table) - 1)
        return index, remainder, self.clamp_table[index] - 128


def _fold_coefficients(
    *, seed_parts: tuple[bytes, ...], label: bytes, count: int
) -> tuple[int, ...]:
    seed = hashlib.sha256(_FOLD_DOMAIN + b"".join(seed_parts) + label).digest()
    return tuple(_challenge(seed, index + 1) for index in range(count))


def _projection_check_data(
    *,
    statement: GoldilocksTinyTransformerStatementV3,
    weights: tuple[tuple[int, ...], ...],
    x_rows: list[list[int]],
    label: bytes,
    seed_parts: tuple[bytes, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Flatten X, derive fold coefficients, and build the public factor."""

    tokens = len(x_rows)
    in_dim = len(x_rows[0])
    out_dim = len(weights[0])
    v_coeff = _fold_coefficients(
        seed_parts=seed_parts, label=label + b"/v", count=tokens
    )
    u_coeff = _fold_coefficients(
        seed_parts=seed_parts, label=label + b"/u", count=out_dim
    )
    wu = tuple(
        sum(
            _signed_field(weights[k][j]) * u_coeff[j] for j in range(out_dim)
        )
        % GOLDILOCKS_MODULUS
        for k in range(in_dim)
    )
    flat = [
        _signed_field(x_rows[t][k]) for t in range(tokens) for k in range(in_dim)
    ]
    factor = [
        v_coeff[t] * wu[k] % GOLDILOCKS_MODULUS
        for t in range(tokens)
        for k in range(in_dim)
    ]
    x_padded = _pad_pow2(flat)
    factor_padded = _pad_pow2(factor)
    return x_padded, factor_padded, v_coeff, u_coeff


@dataclass(frozen=True, slots=True)
class GoldilocksTinyTransformerProofV3:
    chain_opening: tuple[tuple[int, ...], ...]
    returned_token: int
    projection_proofs: tuple[GoldilocksFoldSumcheckProofV3, ...]  # q, k, v, lm
    attention_proof: GoldilocksAttentionHeadProofV3
    attention_roots: tuple[bytes, bytes, bytes, bytes, bytes]
    clamp_proof: GoldilocksLogupProofV3
    clamp_roots: tuple[bytes, bytes]
    byte_proof: GoldilocksLogupProofV3
    byte_roots: tuple[bytes, bytes]


def run_and_prove_goldilocks_tiny_transformer_v3(
    *,
    statement: GoldilocksTinyTransformerStatementV3,
    validator_nonce: bytes,
) -> tuple[bytes, GoldilocksTinyTransformerProofV3]:
    """Execute the exact forward pass, freeze the chain, and prove it.

    Returns ``(chain_root, proof)``.  The prover here is honest by
    construction; adversarial tests freeze forged chains through the same
    freezing path and must fail verification.
    """

    tokens = list(statement.prompt_tokens)
    T, D, V = len(tokens), statement.model_dim, statement.vocab_size
    exec_digest = statement.digest()

    # Forward pass, exact integers.
    x0 = [list(statement.embedding[token]) for token in tokens]

    def matmul(x, w):
        return [
            [
                sum(x[t][k] * w[k][j] for k in range(len(w)))
                for j in range(len(w[0]))
            ]
            for t in range(len(x))
        ]

    wide_q, wide_k, wide_v = (
        matmul(x0, statement.w_q),
        matmul(x0, statement.w_k),
        matmul(x0, statement.w_v),
    )

    clamp_pairs: list[int] = []
    byte_values: list[int] = []
    requant_meta: list[tuple[int, int]] = []  # (index, remainder) per cell

    def requant_table(wide):
        out = []
        for row in wide:
            out_row = []
            for value in row:
                index, remainder, int8_value = statement.requantize(value)
                clamp_pairs.append(
                    index + _CLAMP_PACK * (int8_value + 128)
                )
                byte_values.extend(
                    (remainder >> (8 * limb)) & 0xFF for limb in range(2)
                )
                requant_meta.append((index, remainder))
                out_row.append(int8_value)
            out.append(out_row)
        return out

    q8, k8, v8 = requant_table(wide_q), requant_table(wide_k), requant_table(wide_v)

    attn_statement = statement.attention_statement()
    raw = [
        [sum(q8[t][d] * k8[s][d] for d in range(D)) for s in range(T)]
        for t in range(T)
    ]
    probs = []
    for t in range(T):
        exps = [
            statement.exp_table[attn_statement.quantize(raw[t][s])]
            for s in range(t + 1)
        ]
        total = sum(exps)
        row = [exp * E2E_SCALE_V3 // total for exp in exps]
        row.extend(0 for _ in range(T - t - 1))
        probs.append(row)
    attn_out = [
        [sum(probs[t][s] * v8[s][d] for s in range(T)) for d in range(D)]
        for t in range(T)
    ]
    resid = [
        [attn_out[t][d] + x0[t][d] * E2E_SCALE_V3 for d in range(D)]
        for t in range(T)
    ]
    h8 = requant_table(resid)
    logits = [
        sum(h8[T - 1][d] * statement.w_lm[d][j] for d in range(D))
        for j in range(V)
    ]
    best = max(range(V), key=lambda j: (logits[j], -j))
    for j in range(V):
        if j == best:
            continue
        diff = logits[best] - logits[j] - (1 if j < best else 0)
        byte_values.extend((diff >> (8 * limb)) & 0xFF for limb in range(6))

    # Freeze the chain tree: per-token row carries every intermediate.
    rows: list[tuple[int, ...]] = []
    for t in range(T):
        rows.append(
            tuple(_signed_field(v) for v in x0[t])
            + tuple(_signed_field(v) for row in (wide_q, wide_k, wide_v)
                    for v in row[t])
            + tuple(_signed_field(v) for m in (q8, k8, v8) for v in m[t])
            + tuple(_signed_field(v) for v in raw[t])
            + tuple(_signed_field(v) for v in probs[t])
            + tuple(_signed_field(v) for v in attn_out[t])
            + tuple(_signed_field(v) for v in h8[t])
        )
    logits_row = tuple(_signed_field(v) for v in logits) + (best,)
    width = max(len(rows[0]), len(logits_row))
    rows = [row + (0,) * (width - len(row)) for row in rows]
    rows.append(logits_row + (0,) * (width - len(logits_row)))
    padded = 1 << max(1, (len(rows) - 1).bit_length())
    rows.extend(((0,) * width,) for _ in range(0))
    while len(rows) < padded:
        rows.append((0,) * width)
    chain_tree = GoldilocksMerkleTreeReference.from_rows(
        tuple(rows),
        binding_digest=statement.chain_binding_digest(),
    )
    chain_root = chain_tree.commitment

    # Projection proofs (q, k, v against x0; lm against last h8 row).
    projection_proofs = []
    for label, weights, x_rows, y_rows in (
        (b"proj/q", statement.w_q, x0, wide_q),
        (b"proj/k", statement.w_k, x0, wide_k),
        (b"proj/v", statement.w_v, x0, wide_v),
        (b"proj/lm", statement.w_lm, [h8[T - 1]], [logits]),
    ):
        x_padded, factor, _v, _u = _projection_check_data(
            statement=statement,
            weights=weights,
            x_rows=x_rows,
            label=label,
            seed_parts=(exec_digest, chain_root, validator_nonce),
        )
        x_tree = commit_goldilocks_fold_sumcheck_x_v3(
            statement_digest=hashlib.sha256(exec_digest + label).digest(),
            x_evaluations=x_padded,
        )
        projection_proofs.append(
            prove_goldilocks_fold_sumcheck_v3(
                statement_digest=hashlib.sha256(exec_digest + label).digest(),
                x_tree=x_tree,
                x_evaluations=x_padded,
                factor=factor,
                validator_nonce=validator_nonce,
            )
        )

    # Attention sub-proof over the requantized tables.
    attn_witness = freeze_goldilocks_attention_head_v3(
        statement=attn_statement,
        q=tuple(tuple(row) for row in q8),
        k=tuple(tuple(row) for row in k8),
        value_table=tuple(tuple(row) for row in v8),
        raw_scores=tuple(tuple(row) for row in raw),
        probs=tuple(tuple(row) for row in probs),
        outputs=tuple(tuple(row) for row in attn_out),
    )
    attention_proof = prove_goldilocks_attention_head_v3(
        witness=attn_witness,
        raw_scores=tuple(tuple(row) for row in raw),
        probs=tuple(tuple(row) for row in probs),
        validator_nonce=validator_nonce,
    )
    attn_digest = attn_statement.digest()
    pv_digest = hashlib.sha256(attn_digest + b"PV").digest()
    attention_roots = (
        attn_witness.tables_tree.commitment,
        commit_goldilocks_product_sumcheck_a_v3(
            statement_digest=attn_digest,
            evaluations=attention_proof.score_sumcheck.a_full_opening,
        ).commitment,
        commit_goldilocks_product_sumcheck_b_v3(
            statement_digest=attn_digest,
            evaluations=attention_proof.score_sumcheck.b_full_opening,
        ).commitment,
        commit_goldilocks_product_sumcheck_a_v3(
            statement_digest=pv_digest,
            evaluations=attention_proof.output_sumcheck.a_full_opening,
        ).commitment,
        commit_goldilocks_product_sumcheck_b_v3(
            statement_digest=pv_digest,
            evaluations=attention_proof.output_sumcheck.b_full_opening,
        ).commitment,
    )

    clamp_statement = statement.clamp_logup_statement()
    clamp_w, clamp_m = freeze_goldilocks_logup_witness_v3(
        statement=clamp_statement, looked_up_values=tuple(clamp_pairs)
    )
    byte_statement = statement.byte_logup_statement()
    byte_w, byte_m = freeze_goldilocks_logup_witness_v3(
        statement=byte_statement, looked_up_values=tuple(byte_values)
    )
    proof = GoldilocksTinyTransformerProofV3(
        chain_opening=tuple(tuple(row) for row in chain_tree.rows),
        returned_token=best,
        projection_proofs=tuple(projection_proofs),
        attention_proof=attention_proof,
        attention_roots=attention_roots,
        clamp_proof=GoldilocksLogupProofV3(
            witness_opening=tuple(row[0] for row in clamp_w.rows),
            multiplicity_opening=tuple(row[0] for row in clamp_m.rows),
        ),
        clamp_roots=(clamp_w.commitment, clamp_m.commitment),
        byte_proof=GoldilocksLogupProofV3(
            witness_opening=tuple(row[0] for row in byte_w.rows),
            multiplicity_opening=tuple(row[0] for row in byte_m.rows),
        ),
        byte_roots=(byte_w.commitment, byte_m.commitment),
    )
    return chain_root, proof


def verify_goldilocks_tiny_transformer_v3(
    proof: object,
    *,
    statement: GoldilocksTinyTransformerStatementV3,
    chain_root: bytes,
    validator_nonce: bytes,
) -> int:
    """Verify the complete chain and return the proven token.

    Every step recomputes its expectation from the statement the verifier
    owns and the frozen chain opening.  Returns ``returned_token`` only if
    the entire chain verifies; any failure raises.
    """

    try:
        if not isinstance(proof, GoldilocksTinyTransformerProofV3):
            raise ProofV3VerificationError("e2e proof type is unexpected")
        T = len(statement.prompt_tokens)
        D, V = statement.model_dim, statement.vocab_size
        exec_digest = statement.digest()
        rows = proof.chain_opening
        rebuilt = GoldilocksMerkleTreeReference.from_rows(
            tuple(tuple(row) for row in rows),
            binding_digest=statement.chain_binding_digest(),
        )
        if rebuilt.commitment != chain_root:
            raise ProofV3VerificationError(
                "e2e chain opening does not match the frozen root"
            )
        # Column slicing per token row.
        offsets = {}
        cursor = 0
        for name, size in (
            ("x0", D), ("wq", D), ("wk", D), ("wv", D),
            ("q8", D), ("k8", D), ("v8", D),
            ("raw", T), ("probs", T), ("attn", D), ("h8", D),
        ):
            offsets[name] = (cursor, cursor + size)
            cursor += size

        def cells(name, t):
            lo, hi = offsets[name]
            return rows[t][lo:hi]

        # 1. Embedding: verifier owns tokens and E.
        for t, token in enumerate(statement.prompt_tokens):
            expected = tuple(
                _signed_field(v) for v in statement.embedding[token]
            )
            if tuple(cells("x0", t)) != expected:
                raise ProofV3VerificationError(
                    "e2e embedding row does not match the public prompt"
                )
        logits_row = rows[T]
        logits = [
            _from_field_signed(logits_row[j]) for j in range(V)
        ]
        returned = _integer(logits_row[V], "returned token")
        if returned != proof.returned_token or not 0 <= returned < V:
            raise ProofV3VerificationError("e2e returned token binding is wrong")

        # 2. Projection folds: expected scalar from the opened wide tables.
        proj_specs = (
            (b"proj/q", statement.w_q, "x0", "wq", T),
            (b"proj/k", statement.w_k, "x0", "wk", T),
            (b"proj/v", statement.w_v, "x0", "wv", T),
        )
        for index, (label, weights, x_name, y_name, rows_count) in enumerate(
            proj_specs
        ):
            x_rows = [
                [_from_field_signed(v) for v in cells(x_name, t)]
                for t in range(rows_count)
            ]
            x_padded, factor, v_coeff, u_coeff = _projection_check_data(
                statement=statement,
                weights=weights,
                x_rows=x_rows,
                label=label,
                seed_parts=(exec_digest, chain_root, validator_nonce),
            )
            expected = 0
            out_dim = len(weights[0])
            for t in range(rows_count):
                y_cells = cells(y_name, t)
                for j in range(out_dim):
                    expected = (
                        expected
                        + v_coeff[t] * u_coeff[j] % GOLDILOCKS_MODULUS
                        * y_cells[j]
                    ) % GOLDILOCKS_MODULUS
            sub_proof = proof.projection_proofs[index]
            digest = hashlib.sha256(exec_digest + label).digest()
            x_commitment = commit_goldilocks_fold_sumcheck_x_v3(
                statement_digest=digest,
                x_evaluations=tuple(sub_proof.x_full_opening),
            ).commitment
            verify_goldilocks_fold_sumcheck_v3(
                sub_proof,
                statement_digest=digest,
                x_commitment=x_commitment,
                factor=factor,
                validator_nonce=validator_nonce,
                expected_sum=expected,
            )
            if tuple(sub_proof.x_full_opening) != x_padded:
                raise ProofV3VerificationError(
                    "e2e projection input does not match the chain"
                )

        # 3. Requantization: Euclidean + clamp pairs + remainder limbs.
        clamp_pairs_expected: list[int] = []
        byte_expected: list[int] = []
        for wide_name, int8_name, use_resid in (
            ("wq", "q8", False),
            ("wk", "k8", False),
            ("wv", "v8", False),
            ("attn", "h8", True),
        ):
            for t in range(T):
                wide_cells = cells(wide_name, t)
                int8_cells = cells(int8_name, t)
                for d in range(D):
                    wide = _from_field_signed(wide_cells[d])
                    if use_resid:
                        wide = wide + _from_field_signed(
                            cells("x0", t)[d]
                        ) * E2E_SCALE_V3
                    index, remainder, int8_value = statement.requantize(wide)
                    if _from_field_signed(int8_cells[d]) != int8_value:
                        raise ProofV3VerificationError(
                            "e2e requantization output does not match"
                        )
                    clamp_pairs_expected.append(
                        index + _CLAMP_PACK * (int8_value + 128)
                    )
                    byte_expected.extend(
                        (remainder >> (8 * limb)) & 0xFF for limb in range(2)
                    )
        # 6. Argmax differences appended to the byte witness.
        for j in range(V):
            if j == returned:
                continue
            diff = logits[returned] - logits[j] - (1 if j < returned else 0)
            if diff < 0 or diff >= 1 << 48:
                raise ProofV3VerificationError(
                    "e2e returned token is not the greedy argmax"
                )
            byte_expected.extend((diff >> (8 * limb)) & 0xFF for limb in range(6))

        clamp_statement = statement.clamp_logup_statement()
        opening = proof.clamp_proof.witness_opening
        if tuple(opening[: len(clamp_pairs_expected)]) != tuple(
            clamp_pairs_expected
        ) or any(
            v != clamp_statement.table[0]
            for v in opening[len(clamp_pairs_expected) :]
        ):
            raise ProofV3VerificationError(
                "e2e clamp witness does not match the chain"
            )
        verify_goldilocks_logup_reference_v3(
            proof.clamp_proof,
            statement=clamp_statement,
            witness_root=proof.clamp_roots[0],
            multiplicity_root=proof.clamp_roots[1],
            validator_nonce=validator_nonce,
        )
        byte_statement = statement.byte_logup_statement()
        opening = proof.byte_proof.witness_opening
        if tuple(opening[: len(byte_expected)]) != tuple(byte_expected) or any(
            v != byte_statement.table[0]
            for v in opening[len(byte_expected) :]
        ):
            raise ProofV3VerificationError(
                "e2e byte witness does not match the chain"
            )
        verify_goldilocks_logup_reference_v3(
            proof.byte_proof,
            statement=byte_statement,
            witness_root=proof.byte_roots[0],
            multiplicity_root=proof.byte_roots[1],
            validator_nonce=validator_nonce,
        )

        # 4. Attention sub-proof, with its tables bound to the chain.
        attn_statement = statement.attention_statement()
        attn_rows = proof.attention_proof.tables_opening
        for t in range(T):
            expected = (
                tuple(cells("q8", t))
                + tuple(cells("k8", t))
                + tuple(cells("v8", t))
                + tuple(cells("raw", t))
                + tuple(cells("probs", t))
                + tuple(cells("attn", t))
            )
            if tuple(attn_rows[t]) != expected:
                raise ProofV3VerificationError(
                    "e2e attention tables do not match the chain"
                )
        verify_goldilocks_attention_head_v3(
            proof.attention_proof,
            statement=attn_statement,
            tables_root=proof.attention_roots[0],
            score_a_root=proof.attention_roots[1],
            score_b_root=proof.attention_roots[2],
            output_a_root=proof.attention_roots[3],
            output_b_root=proof.attention_roots[4],
            validator_nonce=validator_nonce,
        )

        # 5. LM head fold over the last hidden row.
        h_last = [[_from_field_signed(v) for v in cells("h8", T - 1)]]
        x_padded, factor, v_coeff, u_coeff = _projection_check_data(
            statement=statement,
            weights=statement.w_lm,
            x_rows=h_last,
            label=b"proj/lm",
            seed_parts=(exec_digest, chain_root, validator_nonce),
        )
        expected = 0
        for j in range(V):
            expected = (
                expected
                + v_coeff[0] * u_coeff[j] % GOLDILOCKS_MODULUS * logits_row[j]
            ) % GOLDILOCKS_MODULUS
        sub_proof = proof.projection_proofs[3]
        digest = hashlib.sha256(exec_digest + b"proj/lm").digest()
        x_commitment = commit_goldilocks_fold_sumcheck_x_v3(
            statement_digest=digest,
            x_evaluations=tuple(sub_proof.x_full_opening),
        ).commitment
        verify_goldilocks_fold_sumcheck_v3(
            sub_proof,
            statement_digest=digest,
            x_commitment=x_commitment,
            factor=factor,
            validator_nonce=validator_nonce,
            expected_sum=expected,
        )
        if tuple(sub_proof.x_full_opening) != x_padded:
            raise ProofV3VerificationError(
                "e2e LM head input does not match the chain"
            )
        return returned
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("e2e proof is malformed") from exc


__all__ = [
    "E2E_SCALE_V3",
    "GOLDILOCKS_TINY_TRANSFORMER_ABI_V3",
    "GoldilocksTinyTransformerProofV3",
    "GoldilocksTinyTransformerStatementV3",
    "run_and_prove_goldilocks_tiny_transformer_v3",
    "verify_goldilocks_tiny_transformer_v3",
]
