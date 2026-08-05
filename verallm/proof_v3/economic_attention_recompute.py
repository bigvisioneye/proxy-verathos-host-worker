"""Selected-head exact attention recompute -- REFERENCE/spec cross-check.

IMPORTANT (do not wire this into the economic HARD hot path): this wraps the
REFERENCE attention-head primitive (goldilocks_attention_head_reference), which
opens tables in full and is measured at ~8.8s VERIFY per head (dim=64) -- far
too slow for the per-request budget. The PRODUCTION attention audit already
exists in STACK A (GLOBAL_FOLDED_EXECUTION_PROOF_SYSTEM_V3): the succinct
attention tile (goldilocks_succinct_attention, prove_/verify_goldilocks_succinct_
attention_v3) proves ALL heads of a layer in one (h,t,s) cube tile over real
post-RoPE Q/K and real V, chained in proof_v3_full_width_chain_qual.py; measured
k-sampled audit verify ~1.3s (k=1) / ~2.2s (k=2), already in the 1-2s target.

This module is retained as the model-agnostic SPEC (nonce head selection, the
exact fixed-point head semantics, committed-KV table extraction, and an
adversarial cross-check the succinct tile is validated against). It must NOT be
called on the economic verify hot path; attention verification for full audits
is Stack A's job, and the economic tier stays the cheap projections/corridors/
anchors/lm_head-binding complement.

--- original notes ---
Selected-head exact attention recompute for the economic audit.

The economic top/projection anchors bind the qkv/o/gate_up/down projections and
the K/V cache, but NOT the attention OPERATION itself -- softmax(QK^T/sqrt(d) +
causal mask) V -- which links the committed Q/K/V to the committed attention
output (o_proj input). A substitute model can match every projection corridor
yet run a different internal attention. This module closes that link by
recomputing NONCE-SELECTED heads exactly in fixed point, via the shipped
attention-head primitive (goldilocks_attention_head_reference: QK^T product
sumcheck -> causal mask + quant LogUp -> exact softmax tile -> PV product
sumcheck; RoPE composes into the public score factor).

Selected-head, not all-head: the validator draws k heads per audited layer from
the post-nonce transcript, so a miner cannot predict which head to serve
honestly; cheating on any head risks selection like the projection sampling.

Binding to committed KV: the proof opens the head's Q/K/V/output tables in full;
``opened_head_tables_v3`` exposes them so the caller cross-checks them against
the committed K-cache/V-cache/qkv_s/attn-output oracles at sampled cells (same
mechanism as the projection corridors). Without that cross-check the recompute
proves an internally-consistent head over UNCOMMITTED tables; WITH it the head
is pinned to the request's committed activations.

Calibration: the quant/exp tables (softmax semantics + score quantization) are
model calibration and MUST come from the signed profile (see the attention
calibration fields added in step 3) so they are authenticated, not prover-chosen.
This reference is exact at small (tokens, head_dim); real-vLLM softmax
calibration of the tables is the deployment step.
"""

from __future__ import annotations

import hashlib

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_attention_head_reference import (
    GoldilocksAttentionHeadStatementV3,
    freeze_goldilocks_attention_head_v3,
    prove_goldilocks_attention_head_v3,
    verify_goldilocks_attention_head_v3,
)
from verallm.proof_v3.goldilocks_softmax_tile_reference import (
    GOLDILOCKS_SOFTMAX_SCALE_V3,
)

__all__ = [
    "select_audited_heads_v3",
    "attention_statement_from_signed_manifest_v3",
    "attention_head_tables_v3",
    "build_attention_head_audit_v3",
    "verify_attention_head_audit_v3",
    "opened_head_tables_v3",
]

_HEAD_SELECT_DOMAIN = b"VERATHOS/PROOF_V3/ATTN_HEAD_SELECT/V1"


def _table_digest(table) -> bytes:
    import struct

    return hashlib.sha256(
        b"".join(struct.pack("<I", int(v) & 0xFFFFFFFF) for v in table)
    ).digest()


def attention_statement_from_signed_manifest_v3(
    *, manifest, token_count: int, quant_table, exp_table
):
    """Build the head statement from the SIGNED manifest calibration.

    The head/dim/offset and the quant/exp table DIGESTS come from the signed
    manifest; the runtime tables are checked against the signed digests, so a
    prover cannot substitute the softmax semantics. Raises if the manifest has
    no attention calibration or the tables do not match the signed digests.
    """

    if not getattr(manifest, "attn_num_heads", 0):
        raise ProofV3VerificationError(
            "signed manifest has no attention calibration"
        )
    if _table_digest(quant_table) != manifest.attn_quant_table_digest:
        raise ProofV3VerificationError(
            "attention quant table does not match the signed digest"
        )
    if _table_digest(exp_table) != manifest.attn_exp_table_digest:
        raise ProofV3VerificationError(
            "attention exp table does not match the signed digest"
        )
    return GoldilocksAttentionHeadStatementV3(
        validator_binding_digest=manifest.digest(),
        token_count=token_count,
        head_dim=manifest.attn_head_dim,
        quant_offset=manifest.attn_quant_offset,
        quant_table=tuple(int(v) for v in quant_table),
        exp_table=tuple(int(v) for v in exp_table),
    )


def select_audited_heads_v3(
    *, num_heads: int, validator_nonce: bytes, count: int, layer_index: int = 0
) -> tuple[int, ...]:
    """Draw ``count`` distinct head indices from the post-nonce transcript."""

    if num_heads < 1 or count < 1:
        raise ProofV3Error("head selection needs positive counts")
    count = min(count, num_heads)
    chosen: list[int] = []
    counter = 0
    while len(chosen) < count:
        digest = hashlib.sha256(
            _HEAD_SELECT_DOMAIN
            + validator_nonce
            + layer_index.to_bytes(4, "little")
            + counter.to_bytes(4, "little")
        ).digest()
        for chunk in range(0, 32, 4):
            head = int.from_bytes(digest[chunk:chunk + 4], "little") % num_heads
            if head not in chosen:
                chosen.append(head)
                if len(chosen) == count:
                    break
        counter += 1
    return tuple(sorted(chosen))


def attention_head_tables_v3(statement, q, k, v):
    """Exact fixed-point (raw scores, softmax probs, outputs) for one head.

    Mirrors the head semantics: raw[t][s] = sum_d Q[t,d] K[s,d] (causal s<=t);
    prob[t] = exp_table[quantize(raw)] normalized to the softmax scale;
    out[t][d] = sum_s prob[t][s] V[s,d].
    """

    tokens, dim = statement.token_count, statement.head_dim
    raw = tuple(
        tuple(
            sum(q[t][d] * k[s][d] for d in range(dim)) for s in range(tokens)
        )
        for t in range(tokens)
    )
    probs = []
    for t in range(tokens):
        exps = [statement.exp_table[statement.quantize(raw[t][s])] for s in range(t + 1)]
        total = sum(exps)
        if total <= 0:
            raise ProofV3Error("softmax normaliser is non-positive")
        row = [exp * GOLDILOCKS_SOFTMAX_SCALE_V3 // total for exp in exps]
        row.extend(0 for _ in range(tokens - t - 1))
        probs.append(tuple(row))
    probs = tuple(probs)
    outputs = tuple(
        tuple(
            sum(probs[t][s] * v[s][d] for s in range(tokens))
            for d in range(dim)
        )
        for t in range(tokens)
    )
    return raw, probs, outputs


def build_attention_head_audit_v3(*, statement, q, k, v, validator_nonce: bytes):
    """Prover-side: freeze + prove one exact attention head.

    Returns ``(proof, witness, tables)`` where ``tables`` is
    ``(q, k, v, raw, probs, outputs)`` for the caller to bind to committed KV.
    """

    raw, probs, outputs = attention_head_tables_v3(statement, q, k, v)
    witness = freeze_goldilocks_attention_head_v3(
        statement=statement, q=q, k=k, value_table=v,
        raw_scores=raw, probs=probs, outputs=outputs,
    )
    proof = prove_goldilocks_attention_head_v3(
        witness=witness, raw_scores=raw, probs=probs,
        validator_nonce=validator_nonce,
    )
    return proof, witness, (q, k, v, raw, probs, outputs)


def _sumcheck_root(statement_digest: bytes, evaluations, *, side: str) -> bytes:
    from verallm.proof_v3.goldilocks_product_sumcheck_reference import (
        commit_goldilocks_product_sumcheck_a_v3,
        commit_goldilocks_product_sumcheck_b_v3,
    )

    commit = (
        commit_goldilocks_product_sumcheck_a_v3
        if side == "a"
        else commit_goldilocks_product_sumcheck_b_v3
    )
    return commit(statement_digest=statement_digest, evaluations=evaluations).commitment


def verify_attention_head_audit_v3(*, statement, proof, validator_nonce: bytes) -> None:
    """Validator-side: verify one exact attention head proof.

    Encapsulates the reference root-extraction glue (score roots under the
    statement digest, output roots under the ``PV`` sub-digest) and rebuilds
    the frozen tables root from the proof's full opening.
    """

    from verallm.proof_v3.goldilocks_merkle_reference import (
        GoldilocksMerkleTreeReference,
    )

    statement_digest = statement.digest()
    pv_digest = hashlib.sha256(statement_digest + b"PV").digest()
    tables_root = GoldilocksMerkleTreeReference.from_rows(
        tuple(tuple(row) for row in proof.tables_opening),
        binding_digest=statement.tables_binding_digest(),
    ).commitment
    verify_goldilocks_attention_head_v3(
        proof,
        statement=statement,
        tables_root=tables_root,
        score_a_root=_sumcheck_root(
            statement_digest, proof.score_sumcheck.a_full_opening, side="a"),
        score_b_root=_sumcheck_root(
            statement_digest, proof.score_sumcheck.b_full_opening, side="b"),
        output_a_root=_sumcheck_root(
            pv_digest, proof.output_sumcheck.a_full_opening, side="a"),
        output_b_root=_sumcheck_root(
            pv_digest, proof.output_sumcheck.b_full_opening, side="b"),
        validator_nonce=validator_nonce,
    )


def opened_head_tables_v3(statement, proof):
    """Extract the opened (q, k, value, output) tables from a verified proof.

    The caller binds these to the committed K-cache/V-cache/qkv_s/attn-output
    oracles at sampled cells, pinning the recomputed head to the request's
    committed activations. Layout matches ``freeze``'s table concatenation.
    """

    from verallm.proof_v3.economic_commitment import field_to_signed_v3

    tokens, dim = statement.token_count, statement.head_dim
    rows = [tuple(r) for r in proof.tables_opening]

    def _dec(cells):
        return tuple(field_to_signed_v3(c) for c in cells)

    # freeze packs each token row as q|k|v|raw|probs|outputs (per-token concat)
    q, k, value, outputs = [], [], [], []
    for r in rows:
        q.append(_dec(r[0:dim]))
        k.append(_dec(r[dim:2 * dim]))
        value.append(_dec(r[2 * dim:3 * dim]))
        out_base = 3 * dim + 2 * tokens
        outputs.append(_dec(r[out_base:out_base + dim]))
    return {
        "q": tuple(q), "k": tuple(k),
        "value": tuple(value), "outputs": tuple(outputs),
    }
