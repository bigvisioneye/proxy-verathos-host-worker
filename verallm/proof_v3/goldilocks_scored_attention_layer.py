"""Layer-level scored attention: chunks aggregated into ONE opening set.

Wire anatomy (cont19) showed per-chunk ``batch_openings`` at 83% of the
proof; a 30k-context layer is ~59 chunks, so per-chunk openings are the
dominant O(context) wire term.  This module proves a full layer's chunk set
with one SHARED batch-opening collector (per-chunk tags namespaced
``c{i}/``): every chunk defers its terminal claims, the layer finalizes one
batch-opening set, and the verifier replays every chunk's couplings against
the shared openings plus the cross-chunk consistency invariants (identical
public peaks/totals across chunks, selector counts summing to exactly 1 per
row).
"""

from __future__ import annotations

from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.goldilocks_succinct_attention import (
    GoldilocksSuccinctAttentionStatementV3,
    prove_goldilocks_succinct_attention_v3,
    verify_goldilocks_succinct_attention_v3,
)

__all__ = [
    "ScoredLayerProofV3",
    "build_scored_layer_statements_v3",
    "prove_scored_attention_layer_v3",
    "verify_scored_attention_layer_v3",
]


@dataclass(frozen=True, slots=True)
class ScoredLayerProofV3:
    chunk_proofs: tuple
    batch_openings: tuple  # (tag, proof), ONE set for the whole layer
    # merged-tree mode: group-witness LogUps (e.g. the merged limb range)
    # run ONCE against the LAYER group trees: (name, proof) pairs
    layer_logups: tuple = ()


def build_scored_layer_statements_v3(
    *,
    validator_binding_digest: bytes,
    head_count: int,
    head_dim: int,
    exp_table,
    score_bits: int,
    m_nums,
    m_e: int,
    q13,
    k13,
    v8,
    query_positions,
    chunk_size: int,
    qk_bits: int = 13,
    v_bits: int = 8,
    scale_bits: int = 16,
    limb_bits: int = 16,
):
    """Global publics + per-chunk statements from full-layer int tensors.

    ``q13[h][row][d]`` over the query pool, ``k13/v8[h][key][d]`` over the
    FULL key sequence.  Returns ``(statements, k_chunks, v_chunks)``.
    """
    import torch

    q_t = torch.tensor(q13, dtype=torch.int64)
    k_t = torch.tensor(k13, dtype=torch.int64)
    h_real, pool = q_t.shape[0], q_t.shape[1]
    keys = k_t.shape[1]
    hp = 1 << max(0, (h_real - 1).bit_length())
    tp = 1 << max(0, (pool - 1).bit_length())
    qp = torch.zeros((hp, tp, head_dim), dtype=torch.int64)
    qp[:h_real, :pool] = q_t
    kp = torch.zeros((hp, keys, head_dim), dtype=torch.int64)
    kp[:h_real] = k_t
    m_vec = torch.zeros((hp, 1, 1), dtype=torch.int64)
    m_vec[:h_real, 0, 0] = torch.tensor(list(m_nums), dtype=torch.int64)
    raw = torch.einsum("htd,hsd->hts", qp, kp)
    su = (raw * m_vec + (1 << (m_e - 1))) >> m_e
    pos_pad = tuple(query_positions) + (keys - 1,) * (tp - pool)
    vis = (torch.arange(keys)[None, :]
           <= torch.tensor(pos_pad)[:, None])[None].expand_as(su)
    peak = su.masked_fill(~vis, -(1 << 62)).amax(dim=2, keepdim=True)
    peak = torch.where(peak == -(1 << 62), torch.zeros_like(peak), peak)
    smax = (1 << score_bits) - 1
    s_pos = (vis.long() * (peak - su)).clamp(max=smax)
    exp_t = torch.tensor(exp_table, dtype=torch.int64)
    es = exp_t[s_pos] * vis.long()
    totals = tuple(es.sum(dim=2).reshape(-1).tolist())
    peaks_pub = tuple(
        int(x) if int(x) >= 0 else int(x) + GOLDILOCKS_MODULUS
        for x in peak.reshape(-1))
    first_peak = ((su == peak.expand_as(su)) & vis).long().argmax(dim=2)
    statements = []
    k_chunks = []
    v_chunks = []
    for base in range(0, keys, chunk_size):
        count = min(chunk_size, keys - base)
        sel_count = tuple(
            1 if base <= int(first_peak[h, t]) < base + count else 0
            for h in range(hp) for t in range(tp))
        statements.append(GoldilocksSuccinctAttentionStatementV3(
            validator_binding_digest=validator_binding_digest,
            head_count=h_real, token_count=pool, head_dim=head_dim,
            qk_bits=qk_bits, v_bits=v_bits, shift=16,
            exp_table=tuple(exp_table), score_bits=score_bits,
            scale_bits=scale_bits, limb_bits=limb_bits,
            key_count=count, query_positions=tuple(query_positions),
            chunk_base=base, public_totals=totals, scored=1,
            m_nums=tuple(m_nums), m_e=m_e,
            public_peaks=peaks_pub, public_sel_count=sel_count))
        k_chunks.append([rows[base:base + count] for rows in k13])
        v_chunks.append([rows[base:base + count] for rows in v8])
    return tuple(statements), tuple(k_chunks), tuple(v_chunks)


def prove_scored_attention_layer_v3(
    *,
    statements,
    q13,
    k_chunks,
    v_chunks,
    validator_nonce: bytes,
    fused=None,
) -> tuple[ScoredLayerProofV3, tuple]:
    from verallm.proof_v3.goldilocks_succinct_batch_opening import (
        BatchOpeningCollectorV3,
    )

    collector = BatchOpeningCollectorV3()
    proofs = []
    outputs = []
    for i, st in enumerate(statements):
        proof, out = prove_goldilocks_succinct_attention_v3(
            statement=st, q_heads=q13, k_heads=k_chunks[i],
            v_heads=v_chunks[i], validator_nonce=validator_nonce,
            fused=fused, external_collector=collector,
            collector_ns=f"c{i}/")
        proofs.append(proof)
        outputs.append(out)
    openings = tuple(sorted(collector.prove_all(
        validator_nonce=validator_nonce, fused=fused).items()))
    return ScoredLayerProofV3(
        chunk_proofs=tuple(proofs), batch_openings=openings), tuple(outputs)


def verify_scored_attention_layer_v3(
    layer_proof: ScoredLayerProofV3,
    *,
    statements,
    validator_nonce: bytes,
) -> None:
    from verallm.proof_v3.goldilocks_succinct_batch_opening import (
        BatchClaimCheckerV3,
    )

    if len(layer_proof.chunk_proofs) != len(statements):
        raise ProofV3VerificationError("layer chunk count mismatch")
    if not statements:
        raise ProofV3VerificationError("layer needs at least one chunk")
    # cross-chunk consistency: identical global publics, selector counts
    # summing to EXACTLY one per row, contiguous chunk coverage
    base_st = statements[0]
    counts = [0] * len(base_st.public_sel_count)
    expected_base = 0
    for st in statements:
        if (
            st.public_peaks != base_st.public_peaks
            or st.public_totals != base_st.public_totals
            or st.query_positions != base_st.query_positions
            or st.m_nums != base_st.m_nums or st.m_e != base_st.m_e
            or st.pcs_query_count != base_st.pcs_query_count
            or st.chunk_base != expected_base
        ):
            raise ProofV3VerificationError(
                "layer chunk statements are inconsistent")
        expected_base += st.key_count
        for i, c in enumerate(st.public_sel_count):
            counts[i] += c
    hp = 1 << max(0, (base_st.head_count - 1).bit_length())
    tp = 1 << max(0, (base_st.token_count - 1).bit_length())
    for h in range(hp):
        for t in range(tp):
            if counts[h * tp + t] != 1:
                raise ProofV3VerificationError(
                    "layer peak selector counts do not sum to one")
    checker = BatchClaimCheckerV3()
    statements_all: dict = {}
    commitments_all: dict = {}
    for i, (st, proof) in enumerate(
        zip(statements, layer_proof.chunk_proofs, strict=True)
    ):
        registry = verify_goldilocks_succinct_attention_v3(
            proof, statement=st, validator_nonce=validator_nonce,
            external_checker=checker, checker_ns=f"c{i}/",
            grouped_aux_hint=any(
                "logup_aux/" in tag
                for tag, _p in layer_proof.batch_openings))
        if registry is None:
            raise ProofV3VerificationError(
                "aggregated chunk verify returned no registry")
        stmts, comms = registry
        statements_all.update(stmts)
        commitments_all.update(comms)
    checker.verify_all(
        dict(layer_proof.batch_openings),
        statements=statements_all,
        commitments=commitments_all,
        validator_nonce=validator_nonce)


def _layer_digest(statements) -> bytes:
    import hashlib

    return hashlib.sha256(
        b"VERATHOS/PROOF_V3/SCORED_LAYER/V1"
        + b"".join(st.digest() for st in statements)).digest()


def prove_scored_attention_layer_merged_v3(
    *,
    statements,
    q13,
    k_chunks,
    v_chunks,
    validator_nonce: bytes,
    fused=None,
) -> tuple[ScoredLayerProofV3, tuple]:
    """CROSS-CHUNK TREES: every chunk's columns commit into shared
    per-group trees (chunk index in the block bits) -- ONE batch-opening
    proof per tag-group for the whole layer, which is the actual wire win
    (collector centralization alone left per-chunk trees intact)."""
    from verallm.proof_v3.goldilocks_succinct_attention import (
        _build_columns_scored,
        _group_plan,
        _tile_digest,
    )
    from verallm.proof_v3.goldilocks_succinct_batch_opening import (
        BatchOpeningCollectorV3,
    )
    from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
        commit_succinct_column_group_v3,
    )

    import torch

    n = len(statements)
    build_device = (
        "cuda" if fused is not None and torch.cuda.is_available()
        else None)
    cols = []
    for i, st in enumerate(statements):
        c, _out, _tot = _build_columns_scored(
            st, q13, k_chunks[i], v_chunks[i], device=build_device)
        cols.append(c)
    layer_digest = _layer_digest(statements)
    collector = BatchOpeningCollectorV3()
    per_chunk_committed: list[dict] = [dict() for _ in range(n)]
    plan = tuple(
        (g, m) for g, m in _group_plan(statements[0])
        if not _is_limb_group(g, m))
    for group_tag, member_tags in plan:
        ordered = []
        bindings = {}
        for i, st in enumerate(statements):
            td = _tile_digest(st)
            for tag in member_tags:
                nk = f"c{i}/{tag}"
                v = cols[i][tag]
                ordered.append(
                    (nk, v if hasattr(v, "numel") else tuple(v)))
                bindings[nk] = (td, tag)
        group, members = commit_succinct_column_group_v3(
            tile_digest=layer_digest, group_tag=group_tag,
            ordered=tuple(ordered), fused=fused,
            member_bindings=bindings)
        collector.register_group(group)
        import dataclasses as _dc

        for nk, col in members.items():
            collector.register_column(nk, col)
            i = int(nk[1:nk.index("/")])
            local = nk.split("/", 1)[1]
            # sub-provers defer with column.tag through the c{i}/ wrapper:
            # the view must carry the LOCAL tag or keys double-namespace
            per_chunk_committed[i][local] = _dc.replace(col, tag=local)
    proofs = []
    outputs = []
    for i, st in enumerate(statements):
        proof, out = prove_goldilocks_succinct_attention_v3(
            statement=st, q_heads=q13, k_heads=k_chunks[i],
            v_heads=v_chunks[i], validator_nonce=validator_nonce,
            fused=fused, external_collector=collector,
            collector_ns=f"c{i}/",
            precommitted=per_chunk_committed[i])
        proofs.append(proof)
        outputs.append(out)
    openings = tuple(sorted(collector.prove_all(
        validator_nonce=validator_nonce, fused=fused).items()))
    return ScoredLayerProofV3(
        chunk_proofs=tuple(proofs),
        batch_openings=openings), tuple(outputs)


def verify_scored_attention_layer_merged_v3(
    layer_proof: ScoredLayerProofV3,
    *,
    statements,
    validator_nonce: bytes,
) -> None:
    from verallm.proof_v3.goldilocks_succinct_attention import (
        _all_tags,
        _group_plan,
        _tag_member_vars,
    )
    from verallm.proof_v3.goldilocks_succinct_batch_opening import (
        BatchClaimCheckerV3,
    )
    from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
        column_pcs_statement_v3,
    )

    # cross-chunk consistency invariants: identical to the unmerged path
    _layer_invariants(layer_proof, statements)
    n = len(statements)
    layer_digest = _layer_digest(statements)
    checker = BatchClaimCheckerV3()
    statements_all: dict = {}
    commitments_all: dict = {}
    tags0 = dict(zip(
        _all_tags(statements[0]),
        layer_proof.chunk_proofs[0].column_commitments, strict=True))
    plan = tuple(
        (g, m) for g, m in _group_plan(statements[0])
        if not _is_limb_group(g, m))
    merged_set = frozenset(g for g, _m in plan)
    for group_tag, member_tags in plan:
        total_members = len(member_tags) * n
        block_bits = (total_members - 1).bit_length()
        member_vars = _tag_member_vars(statements[0], member_tags[0])
        statements_all[group_tag] = column_pcs_statement_v3(
            layer_digest, group_tag, member_vars + block_bits)
        commitments_all[group_tag] = tags0[member_tags[0]]
        for i in range(n):
            for j, tag in enumerate(member_tags):
                idx = i * len(member_tags) + j
                bp = tuple(
                    (idx >> b) & 1 for b in range(block_bits))
                checker.alias(f"c{i}/{tag}", group_tag, bp)
    hint = any(
        "logup_aux/" in tag for tag, _p in layer_proof.batch_openings)
    for i, (st, proof) in enumerate(
        zip(statements, layer_proof.chunk_proofs, strict=True)
    ):
        registry = verify_goldilocks_succinct_attention_v3(
            proof, statement=st, validator_nonce=validator_nonce,
            external_checker=checker, checker_ns=f"c{i}/",
            grouped_aux_hint=hint, precommitted_layout=merged_set)
        if registry is None:
            raise ProofV3VerificationError(
                "merged chunk verify returned no registry")
        statements_all.update(registry[0])
        commitments_all.update(registry[1])
    checker.verify_all(
        dict(layer_proof.batch_openings),
        statements=statements_all,
        commitments=commitments_all,
        validator_nonce=validator_nonce)


def _layer_invariants(layer_proof, statements) -> None:
    if len(layer_proof.chunk_proofs) != len(statements):
        raise ProofV3VerificationError("layer chunk count mismatch")
    if not statements:
        raise ProofV3VerificationError("layer needs at least one chunk")
    base_st = statements[0]
    counts = [0] * len(base_st.public_sel_count)
    expected_base = 0
    for st in statements:
        if (
            st.public_peaks != base_st.public_peaks
            or st.public_totals != base_st.public_totals
            or st.query_positions != base_st.query_positions
            or st.m_nums != base_st.m_nums or st.m_e != base_st.m_e
            or st.chunk_base != expected_base
        ):
            raise ProofV3VerificationError(
                "layer chunk statements are inconsistent")
        expected_base += st.key_count
        for i, c in enumerate(st.public_sel_count):
            counts[i] += c
    hp = 1 << max(0, (base_st.head_count - 1).bit_length())
    tp = 1 << max(0, (base_st.token_count - 1).bit_length())
    for h in range(hp):
        for t in range(tp):
            if counts[h * tp + t] != 1:
                raise ProofV3VerificationError(
                    "layer peak selector counts do not sum to one")


def _is_limb_group(group_tag: str, member_tags) -> bool:
    """Limb groups keep PER-CHUNK trees: their merged range LogUps
    witness the whole group, and layer-wide group witnesses would push
    the LogUp aux domains past practical bounds."""
    limb = {"rb_l0", "rb_l1", "ov_l0", "ov_l1", "rem_l0", "rem_l1",
            "rem_l2", "comp_l0", "comp_l1", "comp_l2"}
    return set(member_tags) <= limb
