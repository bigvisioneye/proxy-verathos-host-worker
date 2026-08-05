"""GPU leaf emitter + signed-scale quantization for the reduction audit.

Miner-side, serve-time: quantize the captured Q/K/V with the SIGNED
calibration scales (never runtime absmax -- a miner-chosen scale is a
soundness hole), then emit every (head, candidate-row) tree's leaf
summaries in one bounded pass.  Byte-identical to the pure-python
reference (`compute_leaf_summary_v3`): all products stay inside fp64's
exact-integer window, all rounding is the same floor semantics.

VRAM: scores for one (head-batch, rows, keys) block plus the int8
tensors -- bounded well under the 2GB audit budget at 30k-250k context.
"""

from __future__ import annotations

from typing import Final

from verallm.proof_v3.attention_reduction_audit import (
    ReductionGeometryV3,
    ReductionSummaryV3,
    ReductionTreeV3,
    _div_round,
    build_reduction_tree_from_summaries_v3,
)
from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.scored_attention_reference import (
    SCALE_BITS,
    ScoredHeadParamsV3,
    fixed_exp_table_v3,
)

__all__ = [
    "quantize_signed_qkv_v3",
    "emit_reduction_summaries_v3",
    "emit_reduction_summary_tensors_v3",
    "build_reduction_trees_gpu_v3",
]

_SCORE_MIN: Final = -(1 << 62)
_CLAMP_MAX: Final = (1 << 16) - 1


def _div_round_tensor(numer, denom):
    """Vectorized ``_div_round`` (round-half-away, denom > 0 int64).

    Byte-identical to the scalar ABI: |n| + (d>>1) floor-divided, sign
    restored (sign(0) = 0 keeps zero at zero)."""

    import torch

    half = denom >> 1
    return torch.sign(numer) * torch.div(
        numer.abs() + half, denom, rounding_mode="floor")


def quantize_signed_qkv_v3(*, q_post, k_post, v, params_by_head,
                           n_kv: int, k_qmax: int = 127):
    """SIGNED-scale quantization from the calibration profile.

    ``q_post`` is ``[n_heads, T, d]`` post-RoPE float, ``k_post``/``v``
    are ``[n_kv, T, d]`` native GQA float.  Per the scored ABI the
    per-dim k scales fold into the Q side:

        q13[h,t,d] = clamp(round(q[h,t,d] * k_scale_d / q_scale), +-qmax)
        k8[kv,s,d] = clamp(round(k[kv,s,d] / k_scale_d), +-k_qmax)
        v8[kv,s,d] = clamp(round(v[kv,s,d] / v_scale), +-127)

    All scales are exact rationals (num / 2^e) from the SIGNED profile;
    the miner cannot steer them.  k/v use head 0's scales of each kv
    group (the calibration emits identical k-scales within a group --
    validated here, fail-closed).

    ``k_qmax``: the K clamp bound.  The reduction ABI clamps K at the
    int8 +-127; the SCORED succinct tile consumes K at the profile's
    ``qk_qmax`` (13-bit for S13/S13R) -- pass ``params.qk_qmax``
    there."""

    import torch

    n_heads = int(q_post.shape[0])
    if n_heads % n_kv:
        raise ProofV3Error("n_kv must divide n_heads")
    group = n_heads // n_kv
    d = int(q_post.shape[-1])
    q13 = torch.empty_like(q_post, dtype=torch.int64)
    for head in range(n_heads):
        params: ScoredHeadParamsV3 = params_by_head[head]
        if params.head_dim != d:
            raise ProofV3Error("calibration head_dim mismatch")
        rep = params_by_head[(head // group) * group]
        if (params.k_num, params.k_e) != (rep.k_num, rep.k_e):
            raise ProofV3Error(
                "k scales must be identical within a GQA group")
        if (params.v_num, params.v_e) != (rep.v_num, rep.v_e):
            # v8 is quantized ONCE per native kv head with the group
            # representative's scale, while the corridor divides by each
            # query head's own v scale -- they must agree, fail-closed
            raise ProofV3Error(
                "v scales must be identical within a GQA group")
        ratio = torch.tensor(
            [
                (kn / (1 << ke)) / (params.q_num / (1 << params.q_e))
                for kn, ke in zip(params.k_num, params.k_e)
            ],
            dtype=torch.float64, device=q_post.device)
        q13[head] = torch.clamp(
            torch.round(q_post[head].to(torch.float64) * ratio),
            -params.qk_qmax, params.qk_qmax).to(torch.int64)
    k8 = torch.empty_like(k_post, dtype=torch.int64)
    v8 = torch.empty_like(v, dtype=torch.int64)
    for kv in range(n_kv):
        params = params_by_head[kv * group]
        k_scale = torch.tensor(
            [kn / (1 << ke)
             for kn, ke in zip(params.k_num, params.k_e)],
            dtype=torch.float64, device=k_post.device)
        k8[kv] = torch.clamp(
            torch.round(k_post[kv].to(torch.float64) / k_scale),
            -k_qmax, k_qmax).to(torch.int64)
        v_scale = params.v_num / (1 << params.v_e)
        v8[kv] = torch.clamp(
            torch.round(v[kv].to(torch.float64) / v_scale),
            -127, 127).to(torch.int64)
    return q13, k8, v8


def emit_reduction_summary_tensors_v3(*, params: ScoredHeadParamsV3,
                                      q13_rows, k8, v8, row_positions,
                                      chunk_len: int):
    """Exact leaf-summary TENSORS for ONE head over all candidate rows.

    ``q13_rows``: ``[rows, d]`` int64 (CUDA or CPU); ``k8``/``v8``:
    ``[T, d]`` int64 for this head's NATIVE kv head; returns int64
    device tensors ``(peaks [r, c], masses [r, c], outs [r, c, d])``
    with empty chunks already at the identity (SCORE_MIN, 0, 0).

    Exactness: q13.k8 products are < 2^13 * 2^7 * d <= 2^27 (fp64-exact
    matmul); score = floor((raw*m_num + 2^(m_e-1)) >> m_e) computed in
    int64; weights come from the fixed table (<= 2^22); mass sums <=
    chunk_len * 2^22 and num sums <= chunk_len * 2^22 * 127 stay far
    inside both int64 and fp64's 2^53 window for chunk_len <= 4096; the
    SCALE_BITS shift keeps num << 16 <= 2^53 * 2^16 < 2^63 (int64 ops
    are exact regardless of the fp64 window)."""

    import torch

    if chunk_len > 4096:
        raise ProofV3Error("chunk_len exceeds the exactness window")
    device = k8.device
    q13_rows = q13_rows.to(device)
    rows = int(q13_rows.shape[0])
    keys = int(k8.shape[0])
    d = int(k8.shape[1])
    table = torch.tensor(
        fixed_exp_table_v3(), dtype=torch.int64, device=device)
    positions = torch.tensor(
        [int(p) for p in row_positions], dtype=torch.int64,
        device=device)
    if int(positions.shape[0]) != rows:
        raise ProofV3Error("row positions do not match the q rows")
    # raw scores: fp64-exact matmul, then integer slope in int64
    raw = (
        q13_rows.to(torch.float64) @ k8.to(torch.float64).T
    ).to(torch.int64)                                   # [rows, keys]
    half = 1 << (params.m_e - 1)
    scores = (raw * params.m_num + half) >> params.m_e
    key_index = torch.arange(keys, dtype=torch.int64, device=device)
    visible = key_index.unsqueeze(0) <= positions.unsqueeze(1)
    scores = torch.where(
        visible, scores, torch.full_like(scores, _SCORE_MIN))
    chunk_count = (keys + chunk_len - 1) // chunk_len
    pad = chunk_count * chunk_len - keys
    if pad:
        scores = torch.nn.functional.pad(
            scores, (0, pad), value=_SCORE_MIN)
    scores = scores.view(rows, chunk_count, -1)         # [r, c, L]
    peaks = scores.amax(dim=2)                          # [r, c]
    has_visible = peaks > _SCORE_MIN
    gaps = torch.clamp(
        peaks.unsqueeze(2) - scores, max=_CLAMP_MAX)
    weights = torch.where(
        scores > _SCORE_MIN,
        table[torch.clamp(gaps, min=0)],
        torch.zeros_like(scores))                       # [r, c, L]
    masses = weights.sum(dim=2)                         # [r, c]
    v8_pad = v8
    if pad:
        v8_pad = torch.nn.functional.pad(v8, (0, 0, 0, pad))
    v_chunks = v8_pad.view(chunk_count, -1, d).to(torch.float64)
    # nums[r, c, :] = weights[r, c, :] @ v_chunk[c]  (fp64-exact)
    nums = torch.einsum(
        "rcl,cld->rcd", weights.to(torch.float64), v_chunks
    ).to(torch.int64)
    # SCALE_BITS-scaled bounded softmax output, vectorized _div_round
    outs = _div_round_tensor(
        nums << SCALE_BITS,
        masses.clamp(min=1).unsqueeze(-1))
    outs = torch.where(
        has_visible.unsqueeze(-1), outs, torch.zeros_like(outs))
    peaks = torch.where(
        has_visible, peaks, torch.full_like(peaks, _SCORE_MIN))
    masses = torch.where(
        has_visible, masses, torch.zeros_like(masses))
    return peaks, masses, outs


def emit_reduction_summaries_v3(*, params: ScoredHeadParamsV3,
                                q13_rows, k8, v8, row_positions,
                                chunk_len: int):
    """Exact leaf summaries for ONE head over all candidate rows
    (python objects; see ``emit_reduction_summary_tensors_v3``)."""

    peaks, masses, outs = emit_reduction_summary_tensors_v3(
        params=params, q13_rows=q13_rows, k8=k8, v8=v8,
        row_positions=row_positions, chunk_len=chunk_len)
    rows, chunk_count = int(peaks.shape[0]), int(peaks.shape[1])
    peaks_host = peaks.cpu().tolist()
    masses_host = masses.cpu().tolist()
    outs_host = outs.cpu().tolist()
    return [
        [
            ReductionSummaryV3(
                peak=int(peaks_host[r][c]),
                mass=int(masses_host[r][c]),
                out=tuple(int(x) for x in outs_host[r][c]))
            for c in range(chunk_count)
        ]
        for r in range(rows)
    ]


def _build_trees_reference_path(*, layer: int, params_by_head, q13, k8,
                                v8, candidate_rows, chunk_len: int,
                                n_kv: int, profile_digest: bytes) -> dict:
    """Per-tree python merges (arbitrary-precision) with cached native
    kv-chunk hashes -- the long-context fallback beyond the batched
    int64 exactness window."""

    import torch

    from verallm.proof_v3.attention_reduction_audit import (
        material_hash_v3,
    )

    n_heads = int(q13.shape[0])
    group = n_heads // n_kv
    keys = int(k8.shape[1])
    d = int(k8.shape[2])
    rows_index = torch.tensor(
        [int(r) for r in candidate_rows], dtype=torch.long,
        device=q13.device)
    kv_hash_cache: dict = {}

    def _chunk_hashes(kv: int):
        cached = kv_hash_cache.get(kv)
        if cached is None:
            k_host = k8[kv].to(torch.int64).cpu()
            v_host = v8[kv].to(torch.int64).cpu()
            lens = tuple(
                min(chunk_len, keys - start)
                for start in range(0, keys, chunk_len))
            cached = (
                tuple(material_hash_v3(k_host[s:s + chunk_len])
                      for s in range(0, keys, chunk_len)),
                tuple(material_hash_v3(v_host[s:s + chunk_len])
                      for s in range(0, keys, chunk_len)),
                lens)
            kv_hash_cache[kv] = cached
        return cached

    trees = {}
    for head in range(n_heads):
        kv = head // group
        params = params_by_head[head]
        q_rows = q13[head].index_select(0, rows_index)
        summaries = emit_reduction_summaries_v3(
            params=params, q13_rows=q_rows, k8=k8[kv], v8=v8[kv],
            row_positions=candidate_rows, chunk_len=chunk_len)
        k_hashes, v_hashes, chunk_lens = _chunk_hashes(kv)
        q_rows_host = q_rows.cpu().tolist()
        for row_i, position in enumerate(candidate_rows):
            geometry = ReductionGeometryV3(
                layer=layer, head=head, kv_head=kv,
                row_position=int(position), key_count=keys,
                chunk_len=chunk_len, head_dim=d,
                profile_digest=profile_digest)
            trees[(head, int(position))] = (
                build_reduction_tree_from_summaries_v3(
                    geometry=geometry,
                    q13_row=[int(v) for v in q_rows_host[row_i]],
                    k8_chunks=None, v8_chunks=None,
                    summaries=summaries[row_i],
                    k_hashes=k_hashes, v_hashes=v_hashes,
                    chunk_lens=chunk_lens))
    return trees


def _merge_level_tensors(peaks, masses, outs, table):
    """One batched merge level -- byte-identical to the scalar merge.

    ``peaks``/``masses`` ``[r, n]``, ``outs`` ``[r, n, d]`` int64 with n
    even; returns the parent tensors ``[r, n/2, ...]``."""

    import torch

    pa, pb = peaks[:, 0::2], peaks[:, 1::2]
    ma, mb = masses[:, 0::2], masses[:, 1::2]
    oa, ob = outs[:, 0::2], outs[:, 1::2]
    ident_a = (ma == 0) & (pa == _SCORE_MIN)
    ident_b = (mb == 0) & (pb == _SCORE_MIN)
    peak = torch.maximum(pa, pb)
    ra = (ma * table[torch.clamp(peak - pa, 0, _CLAMP_MAX)]) >> 22
    rb = (mb * table[torch.clamp(peak - pb, 0, _CLAMP_MAX)]) >> 22
    mass = ra + rb
    numer = ra.unsqueeze(-1) * oa + rb.unsqueeze(-1) * ob
    out = _div_round_tensor(numer, mass.clamp(min=1).unsqueeze(-1))
    out = torch.where(
        (mass > 0).unsqueeze(-1), out, torch.zeros_like(out))
    # identity passthrough (a checked first, exactly like the reference)
    parent_peak = torch.where(
        ident_a, pb, torch.where(ident_b, pa, peak))
    parent_mass = torch.where(
        ident_a, mb, torch.where(ident_b, ma, mass))
    parent_out = torch.where(
        ident_a.unsqueeze(-1), ob,
        torch.where(ident_b.unsqueeze(-1), oa, out))
    return parent_peak, parent_mass, parent_out


def build_reduction_trees_gpu_v3(*, layer: int, params_by_head,
                                 q13, k8, v8, candidate_rows,
                                 chunk_len: int, n_kv: int,
                                 profile_digest: bytes,
                                 full_levels_for=None) -> dict:
    """All candidate (head, row) trees for one layer, GPU-summarized
    and GPU-MERGED (batched levels), byte-identical to the pure-python
    builder.

    ``q13``: ``[n_heads, T, d]`` (rows are gathered at
    ``candidate_rows``); ``k8``/``v8``: ``[n_kv, T, d]`` native.

    ``full_levels_for``: iterable of (head, row_position) pairs that
    need PYTHON summaries at every node (answer-time reveals + walks);
    all other trees carry hashes plus the ROOT summary only -- the
    commit path (roots into the layer commitment, integer retention)
    never reads inner summaries, and materializing ~2^17 python
    summary objects per layer dominates commit time.  ``None`` (the
    default) materializes everything (reference behavior)."""

    import hashlib as _hashlib

    import numpy
    import torch

    from verallm.proof_v3.attention_reduction_audit import (
        ReductionSummaryV3 as _Summary,
    )
    from verallm.proof_v3.attention_reduction_audit import (
        _identity_summary,
        _LEAF_DOM,
        _NODE_DOM,
        _u32,
        material_hash_v3,
    )

    n_heads = int(q13.shape[0])
    group = n_heads // n_kv
    keys = int(k8.shape[1])
    d = int(k8.shape[2])
    device = k8.device
    # int64 exactness window for the batched merge.  Every node mass is
    # <= sum of leaf masses <= keys*2^22 (rescale only shrinks); a merge
    # numerator ra*oa + rb*ob has ra,rb >= 0, ra+rb = node_mass, and
    # |out| <= 127*2^16, so |numer| <= node_mass * 127*2^16, and
    # _div_round_tensor adds at most (node_mass>>1).  The largest int64
    # value produced is therefore keys*2^22 * (127*2^16) + keys*2^21.
    # Beyond that (~264k keys) fall back to the arbitrary-precision
    # per-tree python merge path.
    _merge_max = keys * (1 << 22) * (127 << SCALE_BITS) + keys * (1 << 21)
    if _merge_max >= (1 << 63):
        return _build_trees_reference_path(
            layer=layer, params_by_head=params_by_head, q13=q13,
            k8=k8, v8=v8, candidate_rows=candidate_rows,
            chunk_len=chunk_len, n_kv=n_kv,
            profile_digest=profile_digest)
    rows_index = torch.tensor(
        [int(r) for r in candidate_rows], dtype=torch.long,
        device=q13.device)
    table = torch.tensor(
        fixed_exp_table_v3(), dtype=torch.int64, device=device)
    chunk_count = (keys + chunk_len - 1) // chunk_len
    slot_count = 1
    while slot_count < chunk_count:
        slot_count *= 2
    full_set = (
        None if full_levels_for is None
        else {(int(h), int(p)) for h, p in full_levels_for})
    empty_material = material_hash_v3(())
    identity = _identity_summary(d)
    identity_enc = identity.encode()

    # each NATIVE kv chunk is hashed ONCE and shared by every (head, row)
    # tree of its group (byte-identical to hashing the chunk ints per tree)
    kv_hash_cache: dict = {}

    def _chunk_hashes(kv: int):
        cached = kv_hash_cache.get(kv)
        if cached is None:
            k_host = k8[kv].to(torch.int64).cpu()
            v_host = v8[kv].to(torch.int64).cpu()
            lens = tuple(
                min(chunk_len, keys - start)
                for start in range(0, keys, chunk_len))
            cached = (
                tuple(material_hash_v3(k_host[s:s + chunk_len])
                      for s in range(0, keys, chunk_len)),
                tuple(material_hash_v3(v_host[s:s + chunk_len])
                      for s in range(0, keys, chunk_len)),
                lens)
            kv_hash_cache[kv] = cached
        return cached

    def _level_bytes(peaks_np, masses_np, outs_np):
        """[r, n(, d)] numpy int64 -> per-node encode bytes (LE i64,
        peak + mass + out -- exactly ``ReductionSummaryV3.encode``)."""

        stacked = numpy.concatenate(
            [peaks_np[..., None], masses_np[..., None], outs_np],
            axis=-1).astype("<i8")
        return stacked.tobytes(), stacked.shape[-1] * 8

    trees = {}
    for head in range(n_heads):
        kv = head // group
        params = params_by_head[head]
        q_rows = q13[head].index_select(0, rows_index)
        peaks, masses, outs = emit_reduction_summary_tensors_v3(
            params=params, q13_rows=q_rows, k8=k8[kv], v8=v8[kv],
            row_positions=candidate_rows, chunk_len=chunk_len)
        pad = slot_count - chunk_count
        if pad:
            rows_n = int(peaks.shape[0])
            peaks = torch.cat([
                peaks, torch.full((rows_n, pad), _SCORE_MIN,
                                  dtype=torch.int64, device=device)],
                dim=1)
            masses = torch.cat([
                masses, torch.zeros((rows_n, pad), dtype=torch.int64,
                                    device=device)], dim=1)
            outs = torch.cat([
                outs, torch.zeros((rows_n, pad, d), dtype=torch.int64,
                                  device=device)], dim=1)
        # batched merge levels on device, then ONE host transfer each
        level_tensors = [(peaks, masses, outs)]
        while int(level_tensors[-1][0].shape[1]) > 1:
            level_tensors.append(_merge_level_tensors(
                *level_tensors[-1], table))
        levels_host = [
            (p.cpu().numpy(), m.cpu().numpy(), o.cpu().numpy())
            for p, m, o in level_tensors]
        levels_enc = [
            _level_bytes(p, m, o) for p, m, o in levels_host]
        k_hashes, v_hashes, chunk_lens = _chunk_hashes(kv)
        q_rows_host = q_rows.cpu().tolist()
        for row_i, position in enumerate(candidate_rows):
            geometry = ReductionGeometryV3(
                layer=layer, head=head, kv_head=kv,
                row_position=int(position), key_count=keys,
                chunk_len=chunk_len, head_dim=d,
                profile_digest=profile_digest)
            geom = geometry.digest()
            want_full = full_set is None or (
                (head, int(position)) in full_set)
            q_hash = material_hash_v3(
                [[int(v) for v in q_rows_host[row_i]]])
            levels = []
            n_nodes = slot_count
            for level_index in range(len(levels_host)):
                blob, stride = levels_enc[level_index]
                row_base = row_i * n_nodes * stride
                entries = []
                for node in range(n_nodes):
                    enc = blob[row_base + node * stride:
                               row_base + (node + 1) * stride]
                    if level_index == 0:
                        if node < chunk_count:
                            digest = _hashlib.sha256(
                                _LEAF_DOM + geom + _u32(node)
                                + _u32(chunk_lens[node]) + enc
                                + q_hash + k_hashes[node]
                                + v_hashes[node]).digest()
                        else:
                            digest = _hashlib.sha256(
                                _LEAF_DOM + geom + _u32(node)
                                + _u32(0) + identity_enc + q_hash
                                + empty_material
                                + empty_material).digest()
                    else:
                        left = levels[level_index - 1][2 * node]
                        right = levels[level_index - 1][2 * node + 1]
                        digest = _hashlib.sha256(
                            _NODE_DOM + geom + _u32(level_index)
                            + _u32(node) + left[0] + right[0]
                            + enc).digest()
                    is_root = (
                        level_index == len(levels_host) - 1
                        and node == 0)
                    if want_full or is_root:
                        p_np, m_np, o_np = levels_host[level_index]
                        summary = _Summary(
                            peak=int(p_np[row_i, node]),
                            mass=int(m_np[row_i, node]),
                            out=tuple(
                                int(x)
                                for x in o_np[row_i, node]))
                    else:
                        summary = None
                    entries.append((digest, summary))
                levels.append(tuple(entries))
                n_nodes //= 2
            trees[(head, int(position))] = ReductionTreeV3(
                geometry=geometry, levels=tuple(levels))
    return trees
