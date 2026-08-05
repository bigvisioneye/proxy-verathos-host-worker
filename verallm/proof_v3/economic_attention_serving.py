"""Live-vLLM serving hook: commit the attention section from captured buffers.

The vLLM plugin's RequestActivationTracker already captures, per layer and per
request (CUDA-graph-safe), the qkv-projection output (Q||K||V, pre-RoPE) and the
o-projection input (attn_o = the attention output) via ``capture_at_split``
(kinds CAPTURE_ATTENTION_QKV_OUTPUT / CAPTURE_ATTENTION_OUTPUT_INPUT). This hook
turns those into the pre-nonce attention capture roots WITHOUT any new capture
point: split Q/K/V, apply RoPE (a public per-position rotation), GQA-expand K/V,
quantize int8, emit the streaming chunk claims, and commit via
commit_attention_layer_v3. It runs in the serving loop beside the forward (the
capture is a byproduct); the resulting per-layer root commitment is folded into
capture_chain_digest -> execution_root pre-nonce (attention_layer_commitment_v3).

The commit path (quantize -> emit -> GPU capture-tree commit) is the same one
proven on real Qwen at 30k; this module only adapts the SOURCE from the eager
harness to the live tracker buffers.
"""

from __future__ import annotations

from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.economic_attention_section import (
    attention_layer_commitment_v3,
    commit_attention_layer_v3,
)

__all__ = [
    "split_qkv_capture_v3",
    "quantize_head_tensor_v3",
    "calibrated_attention_statement_v3",
    "commit_attention_layer_from_capture_v3",
]


def calibrated_attention_statement_v3(
    *,
    validator_binding_digest: bytes,
    n_heads: int,
    head_dim: int,
    token_count: int,
    key_count: int,
    query_positions,
    q_scale: float,
    k_scale: float,
    qk_bits: int = 8,
    v_bits: int = 8,
    score_bits: int = 12,
    scale_bits: int = 16,
    limb_bits: int = 16,
    peak_raw: int | None = None,
):
    """Attention statement whose exp table tracks the MODEL's real softmax.

    ``score_bits`` sets the logit resolution: the raw int8-product window
    (~2^21) is bucketed into 2^score_bits table entries, so one bucket spans
    ``window*slope/2^score_bits`` in logit space.  At realistic Qwen scales
    score_bits=8 gives ~2.0-logit buckets (weight error up to e^±1 per key,
    useless for output binding); 12 gives ~0.13 (±6%), 14 gives ~0.03 (±1.6%).
    The table is DERIVED deterministically from (scales, dims) -- the verifier
    rebuilds it from the signed profile + the request's committed scales, so
    table size never crosses the wire.

    The integer pipeline computes ``es = table[(q8.k8 + offset) >> shift]`` and
    ``probs = es * 2^scale_bits / total``.  For the committed surrogate to
    approximate the runtime attention output (the o_proj input the tracker
    captures), the table must encode ``exp(raw * q_scale*k_scale/sqrt(d))`` --
    the model's logit per raw int8-product unit -- NOT a fixed-slope curve.
    The statement (incl. this per-request table) is digest-bound into the
    pre-nonce claims commitment, so calibration cannot be swapped post-nonce.
    Softmax is shift-invariant, so the table is normalized to its max bucket;
    the ``max(1, ...)`` floor on far-negative logits adds at most
    ``key_count/max_exp`` relative mass (<1% at 30k with max_exp=2^22).
    """
    import math

    from verallm.proof_v3.goldilocks_succinct_attention import (
        GoldilocksSuccinctAttentionStatementV3,
    )

    if q_scale <= 0 or k_scale <= 0:
        raise ProofV3Error("attention quant scales must be positive")
    qmax = (1 << (qk_bits - 1)) - 1
    raw_max = head_dim * qmax * qmax
    window = 1
    while window < 2 * raw_max + 1:
        window <<= 1
    shift = max(1, window.bit_length() - 1 - score_bits)
    key_pad = 1
    while key_pad < key_count:
        key_pad <<= 1
    max_exp = min(1 << 22, (1 << (3 * limb_bits)) // max(1, key_pad) - 1)
    slope = (q_scale * k_scale) / math.sqrt(head_dim)
    # Softmax is shift-invariant, so the table normalizes at the PEAK logit.
    # peak_raw = the request's actual maximum q8.k8 product (the miner knows
    # it; it is digest-committed with the statement, so it cannot move
    # post-nonce). Normalizing at the worst-case raw_max instead floors every
    # reachable bucket to es=1 (actual logits sit thousands below the
    # worst-case bound) and the surrogate degenerates to UNIFORM attention --
    # measured honest error 45-95% vs <a few %% with the true peak.
    norm_raw = raw_max if peak_raw is None else min(int(peak_raw), raw_max)
    lmax = norm_raw * slope
    offset = 1 << (shift + score_bits - 1)
    half_bucket = 1 << (shift - 1)
    # exponent clamped at 0: buckets above the peak carry no softmax mass
    # worth resolving (and the window's power-of-two padding is unreachable);
    # clamping also avoids math.exp overflow on that padding.
    table = tuple(
        max(1, min(max_exp, int(round(
            math.exp(min(0.0,
                         ((b << shift) + half_bucket - offset) * slope - lmax))
            * max_exp))))
        for b in range(1 << score_bits)
    )
    return GoldilocksSuccinctAttentionStatementV3(
        validator_binding_digest=validator_binding_digest,
        head_count=n_heads, token_count=token_count, head_dim=head_dim,
        qk_bits=qk_bits, v_bits=v_bits, shift=shift, exp_table=table,
        score_bits=score_bits, scale_bits=scale_bits, limb_bits=limb_bits,
        key_count=key_count, query_positions=tuple(query_positions))


def split_qkv_capture_v3(qkv_output, *, n_heads: int, n_kv: int, head_dim: int):
    """Split a captured [T, (n_heads+2*n_kv)*head_dim] qkv output into
    Q [n_heads,T,hd], K [n_kv,T,hd], V [n_kv,T,hd] (pre-RoPE for Q/K)."""
    import torch

    t = qkv_output if hasattr(qkv_output, "shape") else torch.as_tensor(qkv_output)
    T, width = int(t.shape[0]), int(t.shape[1])
    if width != (n_heads + 2 * n_kv) * head_dim:
        raise ProofV3Error("qkv capture width does not match the head layout")
    q = t[:, : n_heads * head_dim].reshape(T, n_heads, head_dim).permute(1, 0, 2)
    k = t[:, n_heads * head_dim: (n_heads + n_kv) * head_dim].reshape(T, n_kv, head_dim).permute(1, 0, 2)
    v = t[:, (n_heads + n_kv) * head_dim:].reshape(T, n_kv, head_dim).permute(1, 0, 2)
    return q.contiguous(), k.contiguous(), v.contiguous()


def quantize_head_tensor_v3(tensor, *, qmax: int = 127):
    """Symmetric absmax int8 quantization -> int64 tensor + scale (per tensor)."""
    import torch

    t = tensor.to(torch.float64) if hasattr(tensor, "to") else torch.as_tensor(tensor, dtype=torch.float64)
    scale = float(t.abs().max()) / qmax or 1.0
    q = torch.round(t / scale).clamp(-qmax, qmax).to(torch.int64)
    return q, scale


def commit_attention_layer_from_capture_v3(
    *,
    capture_binding: bytes,
    layer: int,
    qkv_output,
    attn_o_input,
    positions,
    n_heads: int,
    n_kv: int,
    head_dim: int,
    rope_fn,
    statement_builder,
    chunk_size: int,
    query_pool,
    ext="auto",
):
    """Serving-loop hook: capture -> RoPE -> GQA -> quantize -> emit -> commit.

    ``rope_fn(q, k, positions)`` applies the model's rotary (public) to the
    pre-RoPE Q/K; ``statement_builder(token_count, key_count)`` builds the
    signed-calibration attention statement (quant/exp tables from the profile);
    ``query_pool`` is the committed query-position pool (the decode rows). Returns
    ``(roots, commitment, trees)`` -- ``commitment`` is folded into
    capture_chain_digest pre-nonce; ``trees`` answer the hard-audit opening.
    """
    import torch

    from verallm.proof_v3.recompute_audit import emit_attention_chunk_claims_auto_v3

    q_pre, k_pre, v = split_qkv_capture_v3(
        qkv_output, n_heads=n_heads, n_kv=n_kv, head_dim=head_dim)
    q_post, k_post = rope_fn(q_pre, k_pre, positions)  # public rotation
    if n_kv < 1 or n_heads % n_kv:
        raise ProofV3Error("n_kv must divide n_heads")
    group = n_heads // n_kv
    # K/V stay NATIVE [n_kv, T, d]: commit + retention use the public
    # h -> h//group mapping. Per-tensor absmax is expansion-invariant, so
    # the quantized values match the old expanded path exactly.
    q8, _qs = quantize_head_tensor_v3(q_post)
    k8, _ks = quantize_head_tensor_v3(k_post)
    v8, _vs = quantize_head_tensor_v3(v)
    T = int(v8.shape[1])
    pool = list(query_pool)
    q8_pool = q8[:, pool, :].contiguous()
    st = statement_builder(len(pool), T)
    # claims are per QUERY head; expand only for this emission (transient)
    idx = torch.arange(n_heads) // group
    claims = emit_attention_chunk_claims_auto_v3(
        statement=st, q_heads=q8_pool, k_heads=k8[idx], v_heads=v8[idx],
        chunk_size=chunk_size)
    # attn_o = the SURROGATE output (sum of the committed int8 partials), over
    # the query pool. This is what composition binds; the REAL captured o_proj
    # input (attn_o_input) connects the surrogate to the chain via the separate
    # o_projection corridor (which tolerates the int8 quantization gap).
    attn_o = torch.tensor(claims.partial_out, dtype=torch.int64).sum(dim=0)  # [H,pool,dim]
    roots, trees = commit_attention_layer_v3(
        capture_binding=capture_binding, layer=layer,
        q=q8_pool, k=k8, v=v8, attn_o=attn_o, statement=st, claims=claims,
        ext=ext, kv_heads=n_kv)
    commitment = attention_layer_commitment_v3(layer=layer, roots=roots)
    return roots, commitment, trees, st, claims
