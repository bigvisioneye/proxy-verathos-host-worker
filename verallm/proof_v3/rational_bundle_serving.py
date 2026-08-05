"""Miner-side serving of the RATIONAL (V2) succinct hard-canary bundle.

The one full-context hard canary's SECURITY ADMISSION: after the
validator reveals the post-commit nonce, the miner derives the SAME
subaudit plans as the reduction diagnostic (derive_reduction_bundle_v3
-- exact challenge equality with the validator's own derivation),
quantizes the retained per-layer captures with the SIGNED calibration
scales (the scored ABI: K clamps at ``qk_qmax``, NOT the reduction
rail's int8 +-127), computes the exact-integer global publics
(row totals / global peaks / first-peak selectors), proves every
canonical contiguous chunk through the V2 rational succinct tile, and
packs one RationalLayerSectionWireV3 per signed-selected layer into
the VRXB v2 envelope.

Everything here is exact int64 arithmetic mirroring the tile's own
witness builder (per-head slope numerators with ONE common exponent,
floor shift with half-up bias, causal visibility incl. the padded-row
convention) -- a publics mismatch is a proof failure, never a wrong
accept.
"""

from __future__ import annotations

from functools import wraps

from verallm.proof_v3.attention_reduction_audit import (
    derive_reduction_bundle_v3,
)
from verallm.proof_v3.attention_reduction_emitter import (
    quantize_signed_qkv_v3,
)
from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
    pcs_query_count_v3,
)
from verallm.proof_v3.goldilocks_succinct_attention import (
    GoldilocksSuccinctAttentionStatementV3,
    prove_goldilocks_succinct_attention_v3,
)
from verallm.proof_v3.rational_bundle_adapter import (
    RationalBundleGeometryV3,
    common_scored_slopes_v3,
    rational_bundle_binding_v3,
)


def _pcs_query_scoped(function):
    @wraps(function)
    def scoped(*args, **kwargs):
        with pcs_query_count_v3(kwargs.get("pcs_query_count", 16)):
            return function(*args, **kwargs)

    return scoped
from verallm.proof_v3.succinct_attention_wire import (
    RationalLayerSectionWireV3,
    encode_rational_attention_proof_v3,
    encode_rational_bundle_wire_v3,
)

__all__ = [
    "answer_capture_kv_bundle_v3",
    "answer_rational_bundle_v3",
    "capture_kv_index_map_v3",
    "prove_capture_kv_rational_section_v3",
    "prove_rational_layer_section_v3",
    "rational_layer_publics_v3",
    "scored_quantize_layer_v3",
]


def scored_quantize_layer_v3(*, q_post, k_post, v, params_by_head,
                             n_kv: int):
    """Scored-ABI signed-scale quantization for ONE layer.

    ``q_post`` is ``[n_heads, T, d]`` post-RoPE float, ``k_post``/``v``
    are ``[n_kv, T, d]`` native GQA float.  Identical to the reduction
    rail's quantization EXCEPT K clamps at the profile's ``qk_qmax``
    (the succinct tile consumes k13, not int8 k8).  Returns
    ``(q13 [n_heads, T, d], k13 [n_kv, T, d], v8 [n_kv, T, d])``."""

    params_by_head = tuple(params_by_head)
    if not params_by_head:
        raise ProofV3Error("scored quantization needs head params")
    k_qmax = int(params_by_head[0].qk_qmax)
    if any(int(p.qk_qmax) != k_qmax for p in params_by_head):
        raise ProofV3Error(
            "scored quantization needs one common qk_qmax")
    return quantize_signed_qkv_v3(
        q_post=q_post, k_post=k_post, v=v,
        params_by_head=params_by_head, n_kv=n_kv, k_qmax=k_qmax)


def rational_layer_publics_v3(*, q13_rows, k13, m_nums, m_e: int,
                              row_positions, key_count: int,
                              exp_table, score_bits: int,
                              with_es: bool = False):
    """Exact-integer global publics for one selected layer's section.

    ``q13_rows[h][t][d]``: the SELECTED heads' q13 at the sampled row
    positions (real rows only); ``k13[h][s][d]``: the same heads' full
    key sequence (already GQA-expanded).  Mirrors the tile's witness
    builder exactly: per-head slope numerator with the COMMON exponent
    (padded heads at slope 0), ``su = floor((raw*m + 2^(m_e-1)) /
    2^m_e)``, causal visibility with padded rows at the LAST key
    position, global peak per row, table lookup on the clamped
    peak-relative score.  Returns ``(totals, peaks_canon, first_peak,
    hp, tp)`` -- publics span the padded hp*tp grid row-major.  With
    ``with_es`` the raw es tensor ``[hp, tp, keys]`` is appended
    (bounds measurement needs the exact numerator es @ v8)."""

    import torch

    q_t = torch.as_tensor(q13_rows, dtype=torch.int64)
    k_t = torch.as_tensor(k13, dtype=torch.int64)
    device = q_t.device
    k_t = k_t.to(device)
    h_real, tq = int(q_t.shape[0]), int(q_t.shape[1])
    keys = int(k_t.shape[1])
    if keys != int(key_count):
        raise ProofV3Error("k13 does not span the key sequence")
    if len(m_nums) != h_real:
        raise ProofV3Error("m_nums must match the selected heads")
    hp = 1 << max(1, (h_real - 1).bit_length())
    tp = 1 << max(1, (tq - 1).bit_length())
    d = int(q_t.shape[2])
    qp = torch.zeros((hp, tp, d), dtype=torch.int64, device=device)
    qp[:h_real, :tq] = q_t
    kp = torch.zeros((hp, keys, d), dtype=torch.int64, device=device)
    kp[:h_real] = k_t
    if device.type == "cuda":
        # CUDA has no int64 matmul; fp64 is EXACT here (products are
        # bounded by qmax^2 * d < 2^52 -- guarded below)
        raw = torch.einsum(
            "htd,hsd->hts", qp.double(), kp.double()).to(torch.int64)
        if not bool((raw.abs() < (1 << 52)).all()):
            raise ProofV3Error(
                "raw score exceeds the fp64-exact window")
    else:
        raw = torch.einsum("htd,hsd->hts", qp, kp)
    m_vec = torch.zeros((hp, 1, 1), dtype=torch.int64, device=device)
    m_vec[:h_real, 0, 0] = torch.tensor(
        [int(m) for m in m_nums], dtype=torch.int64)
    prod = raw * m_vec + (1 << (m_e - 1))
    if not bool((prod.abs() < (1 << 62)).all()):
        raise ProofV3Error(
            "scored slope product overflows the safe window")
    su = prod >> m_e
    positions = tuple(int(p) for p in row_positions)
    pos_pad = positions + (keys - 1,) * (tp - tq)
    vis = (torch.arange(keys, device=device)[None, :]
           <= torch.tensor(pos_pad, dtype=torch.int64,
                           device=device)[:, None]
           )[None].expand_as(su)
    peak = su.masked_fill(~vis, -(1 << 62)).amax(dim=2, keepdim=True)
    smax = (1 << score_bits) - 1
    s_pos = (vis.long() * (peak - su)).clamp(max=smax)
    table_t = torch.tensor(
        tuple(exp_table), dtype=torch.int64, device=device)
    es = table_t[s_pos] * vis.long()
    totals = tuple(int(x) for x in es.sum(dim=2).reshape(-1))
    peaks_canon = tuple(
        int(x) if int(x) >= 0 else int(x) + GOLDILOCKS_MODULUS
        for x in peak.reshape(-1))
    first_peak = ((su == peak.expand_as(su)) & vis).long().argmax(dim=2)
    if with_es:
        return totals, peaks_canon, first_peak, hp, tp, es
    return totals, peaks_canon, first_peak, hp, tp


@_pcs_query_scoped
def prove_rational_layer_section_v3(*, plan, q13, k13, v8, n_kv: int,
                                    candidate_rows, sel_params,
                                    geometry: RationalBundleGeometryV3,
                                    tile_binding: bytes,
                                    validator_nonce: bytes,
                                    key_count: int, chunk_len: int,
                                    fused=None, merged: bool = False,
                                    pcs_query_count: int = 16):
    """Prove EVERY canonical chunk of one selected layer.

    ``plan``: the nonce-derived subaudit (layer / heads /
    row_positions); ``q13 [n_heads, pool, d]`` spans the CANDIDATE
    POOL rows (``candidate_rows`` gives each pool row's absolute
    position -- the runtime captures retain only the pool);
    ``k13``/``v8`` ``[n_kv, T, d]`` native GQA from
    scored_quantize_layer_v3; ``sel_params``: the signed
    ScoredHeadParamsV3 for ``plan.heads`` in plan order.  Returns the
    RationalLayerSectionWireV3.

    ``merged`` (production default): all chunks' non-limb columns
    commit into SHARED per-group layer trees with ONE batch-opening
    set per layer (batch openings are the dominant per-chunk wire
    term) -- the section carries the layer openings and each chunk
    proof holds only deferred claims for the merged groups."""

    n_heads = int(q13.shape[0])
    if n_heads % n_kv:
        raise ProofV3Error("n_kv must divide n_heads")
    group = n_heads // n_kv
    heads = tuple(int(h) for h in plan.heads)
    positions = tuple(int(p) for p in plan.row_positions)
    pool = tuple(int(p) for p in candidate_rows)
    try:
        row_index = [pool.index(p) for p in positions]
    except ValueError:
        raise ProofV3Error(
            "plan row positions must come from the candidate pool")
    m_nums, m_e = common_scored_slopes_v3(sel_params)
    q_rows = q13[list(heads)][:, row_index]
    k_sel = k13[[h // group for h in heads]]
    v_sel = v8[[h // group for h in heads]]
    totals, peaks_canon, first_peak, hp, tp = rational_layer_publics_v3(
        q13_rows=q_rows, k13=k_sel, m_nums=m_nums, m_e=m_e,
        row_positions=positions, key_count=key_count,
        exp_table=geometry.exp_table, score_bits=geometry.score_bits)
    sels = []
    statements = []
    k_chunks = []
    v_chunks = []
    for base in range(0, int(key_count), int(chunk_len)):
        count = min(int(chunk_len), int(key_count) - base)
        sel_count = tuple(
            1 if base <= int(first_peak[h, t]) < base + count else 0
            for h in range(hp) for t in range(tp))
        statements.append(GoldilocksSuccinctAttentionStatementV3(
            validator_binding_digest=tile_binding,
            head_count=len(heads), token_count=len(positions),
            head_dim=geometry.head_dim, qk_bits=geometry.qk_bits,
            v_bits=geometry.v_bits, shift=geometry.shift,
            exp_table=geometry.exp_table,
            score_bits=geometry.score_bits,
            scale_bits=geometry.scale_bits,
            limb_bits=geometry.limb_bits, key_count=count,
            query_positions=positions, chunk_base=base,
            public_totals=totals, public_peaks=peaks_canon,
            public_sel_count=sel_count, scored=1, m_nums=m_nums,
            m_e=m_e, rational=1, pcs_query_count=pcs_query_count))
        k_chunks.append(k_sel[:, base:base + count])
        v_chunks.append(v_sel[:, base:base + count])
        sels.append(sel_count)
    if merged:
        from verallm.proof_v3.goldilocks_scored_attention_layer import (
            prove_scored_attention_layer_merged_v3,
        )

        layer_proof, _outs = prove_scored_attention_layer_merged_v3(
            statements=tuple(statements), q13=q_rows,
            k_chunks=k_chunks, v_chunks=v_chunks,
            validator_nonce=validator_nonce, fused=fused)
        proofs = tuple(
            encode_rational_attention_proof_v3(p)
            for p in layer_proof.chunk_proofs)
        layer_openings = layer_proof.batch_openings
    else:
        proofs = tuple(
            encode_rational_attention_proof_v3(
                prove_goldilocks_succinct_attention_v3(
                    statement=st, q_heads=q_rows, k_heads=k_chunks[i],
                    v_heads=v_chunks[i],
                    validator_nonce=validator_nonce, fused=fused)[0])
            for i, st in enumerate(statements))
        layer_openings = ()
    return RationalLayerSectionWireV3(
        layer=int(plan.layer), key_count=int(key_count),
        chunk_len=int(chunk_len), public_totals=totals,
        public_peaks=peaks_canon, chunk_sel_counts=tuple(sels),
        chunk_proofs=proofs, layer_openings=tuple(layer_openings))


@_pcs_query_scoped
def answer_rational_bundle_v3(*, validator_nonce: bytes,
                              capture_chain_digest: bytes,
                              validator_binding_digest: bytes,
                              calibration, selected_layers,
                              quantized_by_layer, n_kv: int,
                              geometry: RationalBundleGeometryV3,
                              head_count: int, candidate_rows,
                              key_count: int, chunk_len: int,
                              heads_per_layer: int = 2,
                              row_samples: int = 8,
                              fused=None, merged: bool = False,
                              pcs_query_count: int = 16) -> bytes:
    """Answer ONE canary's V2 succinct bundle challenge.

    ``quantized_by_layer[layer] = (q13, k13, v8)`` from
    scored_quantize_layer_v3 over the retained captures -- q13 spans
    the CANDIDATE POOL rows in ``candidate_rows`` order, k13/v8 the
    full key history; ``calibration.heads_for(layer)`` returns the
    signed per-head ``(params, bounds)`` pairs.  Derives the SAME
    plans the validator will derive, proves every subaudit, returns
    the VRXB v2 wire."""

    layers = tuple(int(x) for x in selected_layers)
    tile_binding = rational_bundle_binding_v3(
        validator_binding_digest=validator_binding_digest,
        capture_chain_digest=capture_chain_digest)
    plans = derive_reduction_bundle_v3(
        validator_nonce=validator_nonce,
        capture_chain_digest=capture_chain_digest,
        profile_digest=calibration.digest,
        selected_layers=layers, head_count=head_count,
        candidate_rows=candidate_rows,
        chunk_count=(int(key_count) + int(chunk_len) - 1)
        // int(chunk_len),
        heads_per_layer=heads_per_layer, row_samples=row_samples)
    sections = []
    for plan in plans:
        q13, k13, v8 = quantized_by_layer[int(plan.layer)]
        heads = calibration.heads_for(plan.layer)
        sel_params = [heads[h][0] for h in plan.heads]
        sections.append(prove_rational_layer_section_v3(
            plan=plan, q13=q13, k13=k13, v8=v8, n_kv=n_kv,
            candidate_rows=candidate_rows, sel_params=sel_params,
            geometry=geometry, tile_binding=tile_binding,
            validator_nonce=validator_nonce, key_count=key_count,
            chunk_len=chunk_len, fused=fused, merged=merged,
            pcs_query_count=pcs_query_count))
    return encode_rational_bundle_wire_v3(sections=sections)


def capture_kv_index_map_v3(*, heads, group: int, sp: int, d: int):
    """Tile-cube leaf -> pre-nonce capture leaf, as a callable.

    The tile's k/v columns span the nonce-SELECTED heads' padded cube
    (hp_sel, sp, d); the capture tree spans the NATIVE GQA cube
    (nkv_pad, sp, d) committed pre-nonce.  The map is pure public
    data: tile head slot h -> native kv head ``heads[h] // group``.
    Selected head count MUST be a power of two (no pad head slots --
    pad slots have no capture counterpart)."""

    heads = tuple(int(h) for h in heads)
    hp = 1 << max(0, (len(heads) - 1).bit_length())
    if hp != len(heads):
        raise ProofV3Error(
            "capture-kv sections need a power-of-two head selection")

    def _map(i: int) -> int:
        h, rest = divmod(int(i), sp * d)
        return (heads[h] // group) * sp * d + rest
    return _map


@_pcs_query_scoped
def prove_capture_kv_rational_section_v3(
        *, plan, q13, k13, v8, n_kv: int, candidate_rows, sel_params,
        geometry: RationalBundleGeometryV3, tile_binding: bytes,
        validator_nonce: bytes, key_count: int, capture_trees,
        fused=None, equality_samples: int | None = None,
        batched: bool = False, gated: bool = False,
        anchor_root: bytes | None = None,
        pcs_query_count: int = 16,
        external_collector=None, collector_ns: str = ""):
    """ONE non-chunked capture-KV rational tile for one selected layer.

    The long-context production design: K/V is PCS-committed ONCE for
    the whole key range (fused device trees), equality-linked to the
    PRE-NONCE capture trees, and the tile commits only row-local
    columns.  ``capture_trees = (cap_k, cap_v, cap_ox[, cap_gate])``
    -- objects with ``.commitment`` and ``.open(indices)``: K/V over
    the padded NATIVE (nkv_pad, sp, d) scored-domain cube, the
    captured-row trees over the padded (nh_pad, pool_pad, d) cube in
    candidate-row order (o_x quantized under the signed ox scales;
    fixed-point gate factors when ``gated``, which the caller reads
    from the SIGNED calibration bounds).  The section carries w1
    multiproofs of the plan's audited rows against those trees --
    the attention-row transport.  Returns the
    CaptureKvLayerSectionWireV3."""

    import torch

    if external_collector is not None and not batched:
        raise ProofV3Error(
            "composed capture-kv sections require the chain-coset "
            "batched mode")

    from verallm.proof_v3.capture_kv_binding import (
        KV_EQUALITY_SAMPLE_COUNT_V3,
        commit_capture_kv_pcs_v3,
        derive_anchor_kv_equality_indices_v3,
        derive_kv_equality_indices_v3,
        derive_row_opening_indices_v3,
        prove_kv_equality_v3,
    )
    from verallm.proof_v3.goldilocks_succinct_attention import (
        _tile_digest,
    )
    from verallm.proof_v3.goldilocks_succinct_batch_opening import (
        BatchOpeningCollectorV3,
        NamespacedCollectorV3,
    )
    from verallm.proof_v3.succinct_attention_wire import (
        CaptureKvLayerSectionWireV3,
    )

    n_heads = int(q13.shape[0])
    if n_heads % n_kv:
        raise ProofV3Error("n_kv must divide n_heads")
    group = n_heads // n_kv
    heads = tuple(int(h) for h in plan.heads)
    positions = tuple(int(p) for p in plan.row_positions)
    pool = tuple(int(p) for p in candidate_rows)
    try:
        row_index = [pool.index(p) for p in positions]
    except ValueError:
        raise ProofV3Error(
            "plan row positions must come from the candidate pool")
    m_nums, m_e = common_scored_slopes_v3(sel_params)
    q_rows = q13[list(heads)][:, row_index]
    k_sel = k13[[h // group for h in heads]]
    v_sel = v8[[h // group for h in heads]]
    totals, peaks_canon, first_peak, hp, tp = rational_layer_publics_v3(
        q13_rows=q_rows, k13=k_sel, m_nums=m_nums, m_e=m_e,
        row_positions=positions, key_count=key_count,
        exp_table=geometry.exp_table, score_bits=geometry.score_bits)
    sel_count = tuple(
        1 if 0 <= int(first_peak[h, t]) < int(key_count) else 0
        for h in range(hp) for t in range(tp))
    statement = GoldilocksSuccinctAttentionStatementV3(
        validator_binding_digest=tile_binding,
        head_count=len(heads), token_count=len(positions),
        head_dim=geometry.head_dim, qk_bits=geometry.qk_bits,
        v_bits=geometry.v_bits, shift=geometry.shift,
        exp_table=geometry.exp_table, score_bits=geometry.score_bits,
        scale_bits=geometry.scale_bits, limb_bits=geometry.limb_bits,
        key_count=int(key_count), query_positions=positions,
        chunk_base=0, public_totals=totals, public_peaks=peaks_canon,
        public_sel_count=sel_count, scored=1, m_nums=m_nums, m_e=m_e,
        rational=1, capture_kv=1, pcs_query_count=pcs_query_count)
    td = _tile_digest(statement)
    sp = 1 << max(0, (int(key_count) - 1).bit_length())
    d = geometry.head_dim
    trees = tuple(capture_trees)
    if len(trees) < 3:
        raise ProofV3Error(
            "capture-kv transport requires the pre-nonce captured-row "
            "trees")
    if gated and len(trees) < 4:
        raise ProofV3Error(
            "gated capture-kv transport requires the pre-nonce gate "
            "capture tree")
    if not gated and len(trees) > 3:
        raise ProofV3Error(
            "ungated capture-kv transport must not carry a gate "
            "capture tree")

    def _pad_cube(t):
        # (h_sel, S, d) -> canonical field values on the PADDED cube.
        # Device tensors keep the int64 BIT-PATTERN encoding of the
        # canonical u64 (torch wrap of +P == the enc_field encoding);
        # the CPU path materializes true canonical python ints (int64
        # cannot hold the modulus).
        cube = torch.zeros((hp, sp, d), dtype=torch.int64,
                           device=t.device)
        cube[:len(heads), :int(key_count)] = t
        flat = cube.reshape(-1)
        if flat.is_cuda:
            return torch.where(
                flat < 0, flat + GOLDILOCKS_MODULUS, flat)
        return tuple(
            int(x) + GOLDILOCKS_MODULUS if int(x) < 0 else int(x)
            for x in flat.tolist())

    if external_collector is None:
        collector = BatchOpeningCollectorV3()
    else:
        collector = NamespacedCollectorV3(
            external_collector, collector_ns)
    cols = {}
    for tag, tensor in (("k", k_sel), ("v", v_sel)):
        vals = _pad_cube(tensor)
        cols[tag] = commit_capture_kv_pcs_v3(
            tile_digest=td, tag=tag, values=vals, fused=fused)
        collector.register_column(tag, cols[tag])
    proof, _out = prove_goldilocks_succinct_attention_v3(
        statement=statement, q_heads=q_rows, k_heads=k_sel,
        v_heads=v_sel, validator_nonce=validator_nonce, fused=fused,
        external_collector=collector, collector_ns="",
        precommitted=dict(cols))
    if anchor_root is not None:
        import hashlib

        from verallm.proof_v3.capture_kv_binding import (
            derive_anchor_q_mle_points_v3,
        )

        q_pad = torch.zeros(
            (hp, statement.token_pad(), d),
            dtype=torch.int64,
            device=q_rows.device,
        )
        q_pad[:len(heads), :len(positions)] = q_rows
        q_values = tuple(
            (
                int(value) + GOLDILOCKS_MODULUS
                if int(value) < 0
                else int(value)
            )
            for value in q_pad.reshape(-1).cpu().tolist()
        )
        commitments_digest = hashlib.sha256(
            b"".join(proof.column_commitments)
        ).digest()
        q_points = derive_anchor_q_mle_points_v3(
            tile_digest=td,
            anchor_root=anchor_root,
            attention_commitments_digest=commitments_digest,
            validator_nonce=validator_nonce,
            variable_count=(len(q_values) - 1).bit_length(),
        )

        def _mle(point):
            work = list(q_values)
            for coordinate in point:
                z = int(coordinate) % GOLDILOCKS_MODULUS
                work = [
                    (
                        work[2 * index]
                        + z
                        * (
                            work[2 * index + 1]
                            - work[2 * index]
                        )
                    )
                    % GOLDILOCKS_MODULUS
                    for index in range(len(work) // 2)
                ]
            return work[0]

        for point in q_points:
            collector.defer("q", point, _mle(point))
    index_map = capture_kv_index_map_v3(
        heads=heads, group=group, sp=sp, d=d)
    count = (KV_EQUALITY_SAMPLE_COUNT_V3
             if equality_samples is None else int(equality_samples))
    eq_i, eq_v, eq_o = [], [], []
    for tag, cap in (("k", capture_trees[0]), ("v", capture_trees[1])):
        if anchor_root is None:
            idx = derive_kv_equality_indices_v3(
                tile_digest=td, capture_root=cap.commitment,
                pcs_root=cols[tag].tree.commitment,
                validator_nonce=validator_nonce,
                leaf_count=hp * sp * d, count=count)
        else:
            idx = derive_anchor_kv_equality_indices_v3(
                tile_digest=td,
                anchor_root=anchor_root,
                pcs_root=cols[tag].tree.commitment,
                validator_nonce=validator_nonce,
                layer=int(plan.layer),
                tag=tag,
                leaf_count=hp * sp * d,
                count=count,
            )
        # canonical sorted-unique capture leaves: distinct tile
        # samples may map to ONE capture leaf (heads sharing a group)
        eq = prove_kv_equality_v3(
            tag=tag, capture_tree=cap, pcs_column=cols[tag],
            indices=idx, collector=collector,
            capture_indices=tuple(sorted(
                {index_map(i) for i in idx})))
        eq_i.append(eq.indices)
        eq_v.append(eq.values)
        eq_o.append(eq.capture_opening)
    # attention-row transport: open the plan's audited (head, row)
    # cells against the pre-nonce captured-row trees; the values ride
    # the openings' width-1 rows (no separate value tables)
    row_idx, _row_leaves, _pool_pad = derive_row_opening_indices_v3(
        heads=heads, positions=positions, candidate_rows=pool,
        head_count=n_heads, head_dim=d)
    row_openings = [trees[2].open(row_idx)]
    if gated:
        row_openings.append(trees[3].open(row_idx))
    batched_payload = None
    if external_collector is not None:
        # The selected-trace composer owns the one terminal opening.
        # Every attention relation and capture-K/V equality claim has
        # already been deferred into that collector under collector_ns.
        # Carrying a second local opening would both waste wire and
        # create two independently accepted terminal statements.
        openings = ()
    elif batched:
        # ONE lockstep+RLC set for THIS layer: bounded prover memory
        # (device values release as this set closes)
        openings = ()
        batched_payload = collector.prove_all_batched(
            validator_nonce=validator_nonce, fused=fused)
    else:
        openings = tuple(sorted(collector.prove_all(
            validator_nonce=validator_nonce, fused=fused).items()))
    return CaptureKvLayerSectionWireV3(
        layer=int(plan.layer), key_count=int(key_count),
        public_totals=totals, public_peaks=peaks_canon,
        public_sel_count=sel_count,
        proof=encode_rational_attention_proof_v3(proof),
        kv_roots=(cols["k"].tree.commitment,
                  cols["v"].tree.commitment),
        eq_indices=(eq_i[0], eq_i[1]),
        eq_values=(eq_v[0], eq_v[1]),
        eq_openings=(eq_o[0], eq_o[1]), openings=openings,
        row_openings=tuple(row_openings),
        batched_openings=batched_payload)


@_pcs_query_scoped
def answer_capture_kv_bundle_v3(*, validator_nonce: bytes,
                                capture_chain_digest: bytes,
                                validator_binding_digest: bytes,
                                calibration, selected_layers,
                                quantized_by_layer,
                                capture_trees_by_layer, n_kv: int,
                                geometry: RationalBundleGeometryV3,
                                head_count: int, candidate_rows,
                                key_count: int,
                                heads_per_layer: int = 2,
                                row_samples: int = 8, fused=None,
                                equality_samples: int | None = None,
                                batched: bool = False,
                                anchor_roots_by_layer=None,
                                pcs_query_count: int = 16,
                                ) -> bytes:
    """Answer ONE canary's capture-KV bundle.

    ``batched=False`` emits envelope version 4 (per-column opening
    chains inside each section).  ``batched=True`` emits envelope
    version 5: every section's columns commit on the canonical shift
    chain and open through ONE per-layer batched set (opening-v2
    Part 1; per-layer keeps prover memory bounded -- each layer's
    device values release when its set closes).
    """

    from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
        pcs_coset_profile_v3,
    )
    from verallm.proof_v3.succinct_attention_wire import (
        encode_anchor_capture_kv_bundle_wire_v3,
        encode_capture_kv_bundle_wire_v3,
        encode_capture_kv_bundle_wire_v5,
    )

    layers = tuple(int(x) for x in selected_layers)
    tile_binding = rational_bundle_binding_v3(
        validator_binding_digest=validator_binding_digest,
        capture_chain_digest=capture_chain_digest)
    plans = derive_reduction_bundle_v3(
        validator_nonce=validator_nonce,
        capture_chain_digest=capture_chain_digest,
        profile_digest=calibration.digest,
        selected_layers=layers, head_count=head_count,
        candidate_rows=candidate_rows, chunk_count=1,
        heads_per_layer=heads_per_layer, row_samples=row_samples)
    with pcs_coset_profile_v3("chain" if batched else "v1"):
        sections = []
        for plan in plans:
            q13, k13, v8 = quantized_by_layer[int(plan.layer)]
            heads = calibration.heads_for(plan.layer)
            sel_params = [heads[h][0] for h in plan.heads]
            # gatedness is a property of the SIGNED calibration
            # bounds -- it decides whether the gate capture tree is
            # mandatory (and forbidden when absent)
            gated = any(
                bool(getattr(heads[h][1], "gated", False))
                for h in plan.heads)
            sections.append(prove_capture_kv_rational_section_v3(
                plan=plan, q13=q13, k13=k13, v8=v8, n_kv=n_kv,
                candidate_rows=candidate_rows, sel_params=sel_params,
                geometry=geometry, tile_binding=tile_binding,
                validator_nonce=validator_nonce, key_count=key_count,
                capture_trees=capture_trees_by_layer[int(plan.layer)],
                fused=fused, equality_samples=equality_samples,
                batched=batched, gated=gated,
                pcs_query_count=pcs_query_count,
                anchor_root=(
                    None
                    if anchor_roots_by_layer is None
                    else anchor_roots_by_layer[int(plan.layer)]
                )))
    if anchor_roots_by_layer is not None:
        return encode_anchor_capture_kv_bundle_wire_v3(
            sections=sections,
            batched=batched,
        )
    if not batched:
        return encode_capture_kv_bundle_wire_v3(sections=sections)
    return encode_capture_kv_bundle_wire_v5(sections=sections)
