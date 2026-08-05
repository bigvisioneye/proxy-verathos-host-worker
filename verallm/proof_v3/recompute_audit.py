"""Reveal-and-recompute audits over the proof-v3 commitment chain.

The routine audit path of the dual-mechanism design (the succinct ZK
tiles remain the dispute/arbitration path): the miner pre-commits all
activations at serve time (capture roots) and, per audited request,
publishes per-chunk softmax claims; the validator nonce-samples
(layer, rows, chunk) targets, the miner reveals the touched int8
slices with Merkle paths against the SAME roots (weight rows verify
against the on-chain registry roots - the validator holds manifests
only, never model weights), and the validator recomputes the EXACT
integer tile semantics locally.  Bit-exact on any hardware because
every audited operation is integer-only.

This module implements the semantic layer: claim emission and the
recompute checkers.  Merkle path verification of revealed slices uses
the existing reference multiopen verifiers and is composed by the
caller (miner/validator wiring).
"""

from __future__ import annotations

from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError

__all__ = [
    "AttentionChunkClaimsV3",
    "emit_attention_chunk_claims_fast_v3",
    "emit_attention_chunk_claims_v3",
    "verify_attention_chunk_reveal_fast_v3",
    "verify_attention_chunk_reveal_v3",
    "verify_attention_composition_v3",
    "verify_projection_row_reveal_v3",
]


@dataclass(frozen=True, slots=True)
class AttentionChunkClaimsV3:
    """Public per-chunk claims for one attention layer's sampled rows.

    ``chunk_totals[j][h][t]`` / ``partial_out[j][h][t][d]`` follow the
    chunked-softmax decomposition (proof_v3_chunked_softmax_design.md):
    global total = sum_j chunk_totals[j], output row = sum_j
    partial_out[j].  All values are plain non-negative ints.
    """

    chunk_size: int
    key_count: int
    chunk_totals: tuple
    partial_out: tuple


def _exp_lookup(statement, raw: int) -> int:
    window = 1 << (statement.shift + statement.score_bits)
    index = raw + statement.raw_offset()
    if not 0 <= index < window:
        raise ProofV3Error("attention raw score escapes the quant window")
    return statement.exp_table[index >> statement.shift]


def _chunk_ranges(key_count: int, chunk_size: int):
    if chunk_size < 1:
        raise ProofV3Error("audit chunk size must be positive")
    return tuple(
        (base, min(chunk_size, key_count - base))
        for base in range(0, key_count, chunk_size)
    )


def emit_attention_chunk_claims_v3(
    *,
    statement,
    q_heads,
    k_heads,
    v_heads,
    chunk_size: int,
) -> AttentionChunkClaimsV3:
    """Miner-side: exact integer chunk claims for the sampled rows.

    ``statement`` is a rectangular attention statement (key_count = the
    FULL sequence length, query_positions = the sampled rows).  Tables
    are ``[head][row][dim]`` ints exactly as the tile provers take.
    """

    if statement.key_count is None:
        raise ProofV3Error("audit claims need a rectangular statement")
    positions = statement.query_positions
    scale = statement.scale()
    totals_by_chunk = []
    out_by_chunk = []
    ranges = _chunk_ranges(statement.key_count, chunk_size)
    # global totals first (they gate the division in every chunk)
    es_rows: list[list[list[int]]] = []
    totals: list[list[int]] = []
    for h in range(statement.head_count):
        es_rows.append([])
        totals.append([])
        for t, pos in enumerate(positions):
            row = [
                _exp_lookup(statement, sum(
                    q_heads[h][t][j] * k_heads[h][s][j]
                    for j in range(statement.head_dim)))
                if s <= pos else 0
                for s in range(statement.key_count)
            ]
            es_rows[h].append(row)
            totals[h].append(sum(row))
    for base, count in ranges:
        chunk_totals = tuple(
            tuple(
                sum(es_rows[h][t][base:base + count])
                for t in range(len(positions)))
            for h in range(statement.head_count))
        chunk_out = tuple(
            tuple(
                tuple(
                    sum(
                        ((es_rows[h][t][s] * scale) // totals[h][t])
                        * v_heads[h][s][j]
                        for s in range(base, base + count))
                    for j in range(statement.head_dim))
                for t in range(len(positions)))
            for h in range(statement.head_count))
        totals_by_chunk.append(chunk_totals)
        out_by_chunk.append(chunk_out)
    return AttentionChunkClaimsV3(
        chunk_size=chunk_size,
        key_count=statement.key_count,
        chunk_totals=tuple(totals_by_chunk),
        partial_out=tuple(out_by_chunk),
    )


def verify_attention_chunk_reveal_v3(
    *,
    statement,
    claims: AttentionChunkClaimsV3,
    chunk_index: int,
    q_rows,
    k_chunk,
    v_chunk,
) -> None:
    """Validator-side: recompute ONE nonce-sampled chunk exactly.

    ``q_rows`` are the sampled rows' q tables ``[head][row][dim]``;
    ``k_chunk``/``v_chunk`` are the revealed key/value slices for the
    sampled chunk ``[head][chunk_row][dim]`` (already Merkle-verified
    against the capture roots by the caller).  Raises on ANY mismatch
    with the published claims.
    """

    ranges = _chunk_ranges(claims.key_count, claims.chunk_size)
    if not 0 <= chunk_index < len(ranges):
        raise ProofV3VerificationError("audit chunk index is out of range")
    base, count = ranges[chunk_index]
    positions = statement.query_positions
    scale = statement.scale()
    global_totals = [
        [
            sum(claims.chunk_totals[j][h][t] for j in range(len(ranges)))
            for t in range(len(positions))
        ]
        for h in range(statement.head_count)
    ]
    for h in range(statement.head_count):
        for t, pos in enumerate(positions):
            total = global_totals[h][t]
            if total < 1:
                raise ProofV3VerificationError(
                    "audit global total is not positive")
            row_total = 0
            row_out = [0] * statement.head_dim
            for local in range(count):
                s = base + local
                if s > pos:
                    continue
                raw = sum(
                    q_rows[h][t][j] * k_chunk[h][local][j]
                    for j in range(statement.head_dim))
                es = _exp_lookup(statement, raw)
                row_total += es
                prob = (es * scale) // total
                for j in range(statement.head_dim):
                    row_out[j] += prob * v_chunk[h][local][j]
            if row_total != claims.chunk_totals[chunk_index][h][t]:
                raise ProofV3VerificationError(
                    "audit chunk total does not match the claim")
            if tuple(row_out) != tuple(
                claims.partial_out[chunk_index][h][t]
            ):
                raise ProofV3VerificationError(
                    "audit chunk partial output does not match the claim")


def verify_attention_composition_v3(
    *,
    statement,
    claims: AttentionChunkClaimsV3,
    expected_out,
) -> None:
    """Validator-side: the public claims must compose to the committed
    attention output rows (``expected_out[h][t][d]``, Merkle-verified
    against the capture root of the layer's output by the caller)."""

    # composition = sum over ALL chunks of partial_out (O(C)); one vectorized
    # reduction keeps this ms-class at C~1e4 (250k-1M context) instead of a
    # per-cell Python triple loop. Byte-identical integer sum; falls back to
    # the scalar reference when torch is unavailable.
    try:
        import torch

        composed = torch.tensor(
            claims.partial_out, dtype=torch.int64).sum(dim=0)
        if composed.tolist() != [
            [list(cell) for cell in head] for head in expected_out
        ]:
            raise ProofV3VerificationError(
                "audit chunk outputs do not compose to the committed output")
        return
    except ImportError:
        pass
    ranges = _chunk_ranges(claims.key_count, claims.chunk_size)
    for h in range(statement.head_count):
        for t in range(len(statement.query_positions)):
            for j in range(statement.head_dim):
                composed = sum(
                    claims.partial_out[k][h][t][j]
                    for k in range(len(ranges)))
                if composed != expected_out[h][t][j]:
                    raise ProofV3VerificationError(
                        "audit chunk outputs do not compose to the "
                        "committed output")


def verify_projection_row_reveal_v3(
    *,
    weight_row,
    input_row,
    expected_value: int,
) -> None:
    """Validator-side: one projection output cell, exact integer.

    ``weight_row`` must already be Merkle-verified against the model's
    ON-CHAIN registry root (the validator holds only the manifest) and
    ``input_row`` against the capture root."""

    if len(weight_row) != len(input_row):
        raise ProofV3VerificationError("audit projection shapes mismatch")
    value = sum(w * x for w, x in zip(weight_row, input_row, strict=True))
    if value != expected_value:
        raise ProofV3VerificationError(
            "audit projection cell does not match the committed output")


def _es_chunk_tensor(statement, q_rows, k_chunk, base: int):
    """Exact integer es/mask tensors for one chunk.

    The QK matmul runs in float64 through vectorized BLAS: exact for
    integers below 2^53, and the statement's quant window bounds every
    partial sum far under that (guarded at statement construction)."""

    import torch

    def _tensor(table):
        return torch.as_tensor(table, dtype=torch.int64)

    q_t = _tensor(q_rows)
    k_t = _tensor(k_chunk)
    raw = torch.einsum(
        "htd,hsd->hts",
        q_t.to(torch.float64), k_t.to(torch.float64)).to(torch.int64)
    index = raw + statement.raw_offset()
    window = 1 << (statement.shift + statement.score_bits)
    if not bool(((index >= 0) & (index < window)).all()):
        raise ProofV3Error("attention raw score escapes the quant window")
    exp_t = torch.tensor(statement.exp_table, dtype=torch.int64)
    es = exp_t[index >> statement.shift]
    positions = torch.tensor(
        statement.query_positions, dtype=torch.int64)
    count = k_t.shape[1]
    mask = (
        (torch.arange(count, dtype=torch.int64) + base)[None, :]
        <= positions[:, None]
    ).to(torch.int64)
    return es * mask[None, :, :]


def emit_attention_chunk_claims_fast_v3(
    *,
    statement,
    q_heads,
    k_heads,
    v_heads,
    chunk_size: int,
) -> AttentionChunkClaimsV3:
    """Vectorized claim emission, byte-identical to the reference."""

    import torch

    if statement.key_count is None:
        raise ProofV3Error("audit claims need a rectangular statement")
    ranges = _chunk_ranges(statement.key_count, chunk_size)
    scale = statement.scale()
    es_chunks = []
    for base, count in ranges:
        k_c = [rows[base:base + count] for rows in k_heads]
        es_chunks.append(_es_chunk_tensor(statement, q_heads, k_c, base))
    totals = sum(chunk.sum(dim=2) for chunk in es_chunks)  # [h, t]
    totals_by_chunk = []
    out_by_chunk = []
    for (base, count), es_m in zip(ranges, es_chunks, strict=True):
        v_c = torch.tensor(
            [rows[base:base + count] for rows in v_heads],
            dtype=torch.int64)
        probs = (es_m * scale) // totals.unsqueeze(2)
        out = torch.einsum(
            "hts,hsd->htd",
            probs.to(torch.float64), v_c.to(torch.float64)
        ).to(torch.int64)
        totals_by_chunk.append(
            tuple(tuple(row) for row in es_m.sum(dim=2).tolist()))
        out_by_chunk.append(
            tuple(
                tuple(tuple(cell) for cell in head)
                for head in out.tolist()))
    return AttentionChunkClaimsV3(
        chunk_size=chunk_size,
        key_count=statement.key_count,
        chunk_totals=tuple(totals_by_chunk),
        partial_out=tuple(out_by_chunk),
    )


def verify_attention_chunk_reveal_fast_v3(
    *,
    statement,
    claims: AttentionChunkClaimsV3,
    chunk_index: int,
    q_rows,
    k_chunk,
    v_chunk,
    precomputed_totals=None,
) -> None:
    """Vectorized validator recompute, byte-identical to the reference.

    ``precomputed_totals`` optionally supplies the global per-(head,row) total
    (sum over all chunks of ``chunk_totals``) so a caller auditing several chunks
    of the same request computes the O(C) denominator once instead of rebuilding
    it -- via a slow ``torch.tensor(nested_tuple)`` -- for every sampled chunk.
    """

    import torch

    ranges = _chunk_ranges(claims.key_count, claims.chunk_size)
    if not 0 <= chunk_index < len(ranges):
        raise ProofV3VerificationError("audit chunk index is out of range")
    base, _count = ranges[chunk_index]
    n_chunks = len(ranges)
    heads = statement.head_count
    rows = len(statement.query_positions)
    # global total = sum over ALL chunks (O(C)); one vectorized reduction
    # instead of a per-chunk Python loop keeps this ms-class even at C~1e4
    # (250k-1M context). Byte-identical integer sum.
    if precomputed_totals is not None:
        totals = torch.as_tensor(precomputed_totals, dtype=torch.int64)
    else:
        totals = torch.tensor(
            claims.chunk_totals, dtype=torch.int64).sum(dim=0)
    if not bool((totals >= 1).all()):
        raise ProofV3VerificationError(
            "audit global total is not positive")
    es_m = _es_chunk_tensor(statement, q_rows, k_chunk, base)
    if es_m.sum(dim=2).tolist() != [
        list(row) for row in claims.chunk_totals[chunk_index]
    ]:
        raise ProofV3VerificationError(
            "audit chunk total does not match the claim")
    probs = (es_m * statement.scale()) // totals.unsqueeze(2)
    v_t = torch.as_tensor(v_chunk, dtype=torch.int64)
    # probs <= scale (2^16) and |v| < 2^7 summed over one chunk stay
    # far below 2^53: exact in float64 BLAS
    out = torch.einsum(
        "hts,hsd->htd",
        probs.to(torch.float64), v_t.to(torch.float64)).to(torch.int64)
    if out.tolist() != [
        [list(cell) for cell in head]
        for head in claims.partial_out[chunk_index]
    ]:
        raise ProofV3VerificationError(
            "audit chunk partial output does not match the claim")


def emit_attention_chunk_claims_gpu_v3(
    *,
    statement,
    q_heads,
    k_heads,
    v_heads,
    chunk_size: int,
    device: str = "cuda",
) -> AttentionChunkClaimsV3:
    """GPU claim emission, byte-identical to the reference.

    Streams one key chunk at a time (tens of MB of VRAM), so it can run
    beside a serving engine - the v1 pattern (GPU matmul, CPU offload
    under load) with the CPU fallback being
    ``emit_attention_chunk_claims_fast_v3``.  CUDA has no int64 matmul,
    so the two matmuls run in float64, which is EXACT for integers
    below 2^53; the bounds are guarded, everything else stays int64.
    """

    import torch

    if statement.key_count is None:
        raise ProofV3Error("audit claims need a rectangular statement")
    max_raw = statement.head_dim << (2 * statement.qk_bits - 2)
    if max_raw >= 1 << 50:
        raise ProofV3Error("audit score range exceeds exact float64")
    # probs <= scale, |v| < 2^(v_bits-1), summed over one chunk
    if statement.scale() * (1 << (statement.v_bits - 1)) * chunk_size >= (
        1 << 52
    ):
        raise ProofV3Error("audit output range exceeds exact float64")
    ranges = _chunk_ranges(statement.key_count, chunk_size)
    scale = statement.scale()
    window = 1 << (statement.shift + statement.score_bits)
    def _as(table, dtype):
        if hasattr(table, "numel"):
            return table.to(device=device, dtype=dtype)
        return torch.tensor(table, dtype=dtype, device=device)

    q_f = _as(q_heads, torch.float64)
    exp_t = torch.tensor(
        statement.exp_table, dtype=torch.int64, device=device)
    positions = torch.tensor(
        statement.query_positions, dtype=torch.int64, device=device)
    es_chunks = []
    for base, count in ranges:
        k_f = (
            _as(k_heads[:, base:base + count], torch.float64)
            if hasattr(k_heads, "numel")
            else torch.tensor(
                [rows[base:base + count] for rows in k_heads],
                dtype=torch.float64, device=device))
        raw = torch.einsum("htd,hsd->hts", q_f, k_f).to(torch.int64)
        index = raw + statement.raw_offset()
        if not bool(((index >= 0) & (index < window)).all()):
            raise ProofV3Error(
                "attention raw score escapes the quant window")
        es = exp_t[index >> statement.shift]
        mask = (
            (torch.arange(count, dtype=torch.int64, device=device)
             + base)[None, :] <= positions[:, None]
        ).to(torch.int64)
        es_chunks.append(es * mask[None, :, :])
    totals = sum(chunk.sum(dim=2) for chunk in es_chunks)  # [h, t]
    totals_by_chunk = []
    out_by_chunk = []
    for (base, count), es_m in zip(ranges, es_chunks, strict=True):
        v_f = (
            _as(v_heads[:, base:base + count], torch.float64)
            if hasattr(v_heads, "numel")
            else torch.tensor(
                [rows[base:base + count] for rows in v_heads],
                dtype=torch.float64, device=device))
        probs = (es_m * scale) // totals.unsqueeze(2)
        out = torch.einsum(
            "hts,hsd->htd", probs.to(torch.float64), v_f
        ).to(torch.int64)
        totals_by_chunk.append(
            tuple(tuple(row) for row in es_m.sum(dim=2).cpu().tolist()))
        out_by_chunk.append(
            tuple(
                tuple(tuple(cell) for cell in head)
                for head in out.cpu().tolist()))
    return AttentionChunkClaimsV3(
        chunk_size=chunk_size,
        key_count=statement.key_count,
        chunk_totals=tuple(totals_by_chunk),
        partial_out=tuple(out_by_chunk),
    )


def emit_attention_chunk_claims_int8_v3(
    *,
    statement,
    q_heads,
    k_heads,
    v_heads,
    chunk_size: int,
    device: str = "cuda",
) -> AttentionChunkClaimsV3:
    """Tensor-core claim emission via ``torch._int_mm`` (v1's kernel).

    Inputs are int8 tensors ``[head][row][dim]`` (exactly the capture
    layout, so device-resident activations need no transfer at all).
    QK^T runs as int8 x int8 -> int32 (exact); probs x V splits probs
    into three 6-bit limbs so every accumulation stays inside int32.
    Byte-identical to the reference emission.
    """

    import torch

    if statement.key_count is None:
        raise ProofV3Error("audit claims need a rectangular statement")
    if statement.qk_bits > 8 or statement.v_bits > 8:
        raise ProofV3Error("int8 audit emission needs <=8-bit q/k/v")
    if statement.scale_bits > 18:
        raise ProofV3Error("int8 audit emission limb split needs <=2^18")
    ranges = _chunk_ranges(statement.key_count, chunk_size)
    scale = statement.scale()
    window = 1 << (statement.shift + statement.score_bits)

    def _as8(table):
        if hasattr(table, "numel"):
            return table.to(device=device, dtype=torch.int8)
        return torch.tensor(table, dtype=torch.int8, device=device)

    q_8 = _as8(q_heads)
    k_8 = _as8(k_heads)
    v_8 = _as8(v_heads)
    heads = statement.head_count
    exp_t = torch.tensor(
        statement.exp_table, dtype=torch.int64, device=device)
    positions = torch.tensor(
        statement.query_positions, dtype=torch.int64, device=device)
    def _pad8(chunk):
        count = chunk.shape[1]
        short = (-count) % 8
        if short == 0:
            return chunk
        return torch.nn.functional.pad(chunk, (0, 0, 0, short))

    es_chunks = []
    for base, count in ranges:
        k_c = _pad8(k_8[:, base:base + count])
        raw = torch.stack([
            torch._int_mm(q_8[h], k_c[h].t().contiguous())
            for h in range(heads)
        ]).to(torch.int64)[:, :, :count]
        index = raw + statement.raw_offset()
        if not bool(((index >= 0) & (index < window)).all()):
            raise ProofV3Error(
                "attention raw score escapes the quant window")
        es = exp_t[index >> statement.shift]
        mask = (
            (torch.arange(count, dtype=torch.int64, device=device)
             + base)[None, :] <= positions[:, None]
        ).to(torch.int64)
        es_chunks.append(es * mask[None, :, :])
    totals = sum(chunk.sum(dim=2) for chunk in es_chunks)
    totals_by_chunk = []
    out_by_chunk = []
    for (base, count), es_m in zip(ranges, es_chunks, strict=True):
        probs = (es_m * scale) // totals.unsqueeze(2)
        out = None
        v_c = _pad8(v_8[:, base:base + count])
        pad_cols = v_c.shape[1] - count
        for limb in range(3):
            limb_8 = ((probs >> (6 * limb)) & 63).to(torch.int8)
            if pad_cols:
                limb_8 = torch.nn.functional.pad(
                    limb_8, (0, pad_cols))
            part = torch.stack([
                torch._int_mm(limb_8[h], v_c[h].contiguous())
                for h in range(heads)
            ]).to(torch.int64) << (6 * limb)
            out = part if out is None else out + part
        totals_by_chunk.append(
            tuple(tuple(row) for row in es_m.sum(dim=2).cpu().tolist()))
        out_by_chunk.append(
            tuple(
                tuple(tuple(cell) for cell in head)
                for head in out.cpu().tolist()))
    return AttentionChunkClaimsV3(
        chunk_size=chunk_size,
        key_count=statement.key_count,
        chunk_totals=tuple(totals_by_chunk),
        partial_out=tuple(out_by_chunk),
    )


def emit_attention_chunk_claims_auto_v3(
    *,
    statement,
    q_heads,
    k_heads,
    v_heads,
    chunk_size: int,
) -> AttentionChunkClaimsV3:
    """Hardware-adaptive emission; results are byte-identical on every
    backend because all paths are exact integer semantics.

    Order: int8 tensor-core path (any CUDA arch with cuBLASLt int8,
    Turing..Blackwell - no arch-specific kernels), then the exact-fp64
    path, then pure CPU (the v1 GPU-with-CPU-offload pattern)."""

    import torch

    if torch.cuda.is_available():
        if (
            statement.qk_bits <= 8 and statement.v_bits <= 8
            and statement.scale_bits <= 18
            and statement.token_count >= 17
            and hasattr(torch, "_int_mm")
        ):
            try:
                return emit_attention_chunk_claims_int8_v3(
                    statement=statement, q_heads=q_heads,
                    k_heads=k_heads, v_heads=v_heads,
                    chunk_size=chunk_size)
            except (RuntimeError, ProofV3Error):
                pass
        try:
            return emit_attention_chunk_claims_gpu_v3(
                statement=statement, q_heads=q_heads, k_heads=k_heads,
                v_heads=v_heads, chunk_size=chunk_size)
        except RuntimeError:
            pass
    if hasattr(q_heads, "numel"):
        q_heads = q_heads.to(torch.int64).tolist()
        k_heads = k_heads.to(torch.int64).tolist()
        v_heads = v_heads.to(torch.int64).tolist()
    return emit_attention_chunk_claims_fast_v3(
        statement=statement, q_heads=q_heads, k_heads=k_heads,
        v_heads=v_heads, chunk_size=chunk_size)


# ---------------------------------------------------------------------------
# Merkle reveal plumbing: capture-tree slice openings
# ---------------------------------------------------------------------------

def build_capture_slice_reveal_v3(
    *,
    validator_binding_digest: bytes,
    raw_values,
    indices,
):
    """Miner-side: open the sampled leaf indices of one capture tree.

    ``raw_values`` is the SAME padded evaluation vector the serve-time
    ``capture_commit_v3`` committed (the miner re-derives it from its
    stored activations); ``indices`` are the nonce-chosen leaves.
    Returns the multiproof; the values travel inside it.
    """

    import hashlib as _h

    from verallm.proof_v3.goldilocks_capture_commitment_reference import (
        _CAPTURE_DOMAIN,
        _field,
    )
    from verallm.proof_v3.goldilocks_merkle_reference import (
        GoldilocksMerkleTreeReference,
    )

    leaf_binding = _h.sha256(
        _CAPTURE_DOMAIN + b"LEAVES/" + validator_binding_digest).digest()
    tree = GoldilocksMerkleTreeReference.from_rows(
        tuple((_field(v, "capture value"),) for v in raw_values),
        binding_digest=leaf_binding,
    )
    return tree.open(tuple(indices))


def verify_capture_slice_reveal_v3(
    *,
    capture_root: bytes,
    validator_binding_digest: bytes,
    leaf_count: int,
    indices,
    opening,
) -> tuple:
    """Validator-side: authenticate a revealed slice against the
    serve-time capture root; returns the leaf values in index order.

    All expectations are validator-derived (root from the signed
    receipt, binding/leaf_count from the audited layer's profile,
    indices from the nonce) - the opening cannot steer them.
    """

    import hashlib as _h

    from verallm.proof_v3.goldilocks_capture_commitment_reference import (
        _CAPTURE_DOMAIN,
    )
    from verallm.proof_v3.goldilocks_merkle_reference import (
        verify_goldilocks_merkle_multiopening_reference,
    )

    leaf_binding = _h.sha256(
        _CAPTURE_DOMAIN + b"LEAVES/" + validator_binding_digest).digest()
    expected = tuple(sorted(set(int(i) for i in indices)))
    verify_goldilocks_merkle_multiopening_reference(
        capture_root,
        opening,
        expected_binding_digest=leaf_binding,
        expected_leaf_count=leaf_count,
        expected_leaf_width=1,
        expected_indices=expected,
    )
    return tuple(row[0] for row in opening.rows)


# ---------------------------------------------------------------------------
# MLP / RMSNorm recompute checkers (exact reference semantics)
# ---------------------------------------------------------------------------

def verify_mlp_rows_reveal_v3(
    *,
    statement,
    gate_rows,
    up_rows,
    expected_h8_rows,
) -> None:
    """Recompute sampled MLP rows exactly; compare to committed output.

    ``statement`` is a ``GoldilocksMlpTileStatementV3`` whose
    ``token_count`` equals the number of sampled rows.  Inputs and the
    expected output are Merkle-verified against capture roots by the
    caller (weight influence enters via the revealed gate/up rows,
    which chain to registry-root-verified projections)."""

    from verallm.proof_v3.goldilocks_mlp_tile_reference import (
        run_and_freeze_goldilocks_mlp_tile_v3,
    )

    _witness, h8 = run_and_freeze_goldilocks_mlp_tile_v3(
        statement=statement,
        gate_rows=tuple(tuple(r) for r in gate_rows),
        up_rows=tuple(tuple(r) for r in up_rows),
    )
    if [list(r) for r in h8] != [list(r) for r in expected_h8_rows]:
        raise ProofV3VerificationError(
            "audit MLP rows do not match the committed output")


def verify_rmsnorm_row_reveal_v3(
    *,
    statement,
    input_row,
    expected_output_row,
) -> None:
    """Recompute one sampled RMSNorm row exactly (weight vector lives
    in the statement, registry-root-verified by the caller)."""

    from verallm.proof_v3.goldilocks_rmsnorm_tile_reference import (
        run_and_freeze_goldilocks_rmsnorm_tile_v3,
    )

    _witness, outputs = run_and_freeze_goldilocks_rmsnorm_tile_v3(
        statement=statement, x_values=tuple(input_row))
    if list(outputs) != list(expected_output_row):
        raise ProofV3VerificationError(
            "audit RMSNorm row does not match the committed output")


# ---------------------------------------------------------------------------
# Sampling policy: nonce -> unpredictable audit targets
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AuditPlanV3:
    """Deterministic nonce-derived audit targets for one request."""

    layer: int
    query_rows: tuple[int, ...]
    chunk_indices: tuple[int, ...]


def derive_audit_plan_v3(
    *,
    validator_nonce: bytes,
    request_binding_digest: bytes,
    layer_count: int,
    sequence_length: int,
    chunk_size: int,
    row_samples: int = 32,
    chunk_samples: int = 3,
    query_pool: tuple[int, ...] | None = None,
) -> AuditPlanV3:
    """Expand a validator nonce into (layer, rows, chunks) targets.

    Deterministic on both sides; unpredictable before the nonce.  Rows
    and chunks are sampled WITHOUT replacement via rejection over a
    SHA256 counter stream (unbiased).

    ``query_pool`` is the set of committed query positions; rows are sampled
    FROM IT (not over the whole sequence), so at long context the audited rows
    always fall inside what was actually committed."""

    import hashlib as _h

    if layer_count < 1 or sequence_length < 1 or chunk_size < 1:
        raise ProofV3Error("audit plan shape is malformed")
    seed = _h.sha256(
        b"VERATHOS/PROOF_V3/AUDIT_PLAN/V1"
        + validator_nonce
        + request_binding_digest
    ).digest()

    def _stream(domain: bytes):
        counter = 0
        while True:
            block = _h.sha256(
                seed + domain + counter.to_bytes(8, "little")).digest()
            counter += 1
            for off in range(0, 32, 8):
                yield int.from_bytes(block[off:off + 8], "little")

    def _sample(domain: bytes, population: int, count: int):
        count = min(count, population)
        bound = (1 << 64) - ((1 << 64) % population)
        chosen: list[int] = []
        seen = set()
        for value in _stream(domain):
            if value >= bound:
                continue
            index = value % population
            if index in seen:
                continue
            seen.add(index)
            chosen.append(index)
            if len(chosen) == count:
                return tuple(sorted(chosen))
        raise ProofV3Error("audit plan sampling failed")  # pragma: no cover

    layer = _sample(b"layer", layer_count, 1)[0]
    if query_pool:
        pool = tuple(sorted({int(p) for p in query_pool}))
        if any(not 0 <= p < sequence_length for p in pool):
            raise ProofV3Error("query pool position outside the sequence")
        rows = tuple(pool[i] for i in _sample(b"rows", len(pool), row_samples))
    else:
        rows = _sample(b"rows", sequence_length, row_samples)
    chunk_count = (sequence_length + chunk_size - 1) // chunk_size
    chunks = _sample(b"chunks", chunk_count, chunk_samples)
    return AuditPlanV3(
        layer=layer, query_rows=rows, chunk_indices=chunks)


def verify_gdn_steps_reveal_v3(
    *,
    statement,
    q,
    k,
    v,
    gate,
    expected_outputs,
) -> None:
    """Recompute a sampled GDN (linear-attention) segment exactly.

    GDN is a per-step recurrence, so the audit unit is a step RANGE:
    the revealed columns cover ``statement.step_count`` steps starting
    from the committed entry state (state continuity is part of the
    revealed chain; both ends verify against capture roots by the
    caller).  Strictly linear cost, context-independent."""

    from verallm.proof_v3.goldilocks_gdn_tile_reference import (
        run_and_freeze_goldilocks_gdn_tile_v3,
    )

    _witness, outputs = run_and_freeze_goldilocks_gdn_tile_v3(
        statement=statement, q=tuple(q), k=tuple(k), v=tuple(v),
        gate=tuple(gate))
    if list(outputs) != list(expected_outputs):
        raise ProofV3VerificationError(
            "audit GDN steps do not match the committed outputs")


def verify_projection_rows_composed_v3(
    *,
    capture_root_in: bytes,
    binding_in: bytes,
    leaf_count_in: int,
    input_indices,
    input_opening,
    weight_rows,
    expected_values,
) -> None:
    """Projection audit composition: authenticate the input activation
    slice against its capture root, then recompute each sampled output
    cell against the (registry-root-verified) weight rows.

    ``weight_rows`` MUST already be Merkle-verified against the model's
    on-chain registry root by the caller (v1 embedding-binding
    machinery); this function composes the capture side + recompute."""

    revealed = verify_capture_slice_reveal_v3(
        capture_root=capture_root_in,
        validator_binding_digest=binding_in,
        leaf_count=leaf_count_in,
        indices=input_indices,
        opening=input_opening,
    )
    if len(weight_rows) != len(expected_values):
        raise ProofV3VerificationError(
            "audit projection reveal shapes mismatch")
    for weight_row, expected in zip(
        weight_rows, expected_values, strict=True
    ):
        verify_projection_row_reveal_v3(
            weight_row=weight_row,
            input_row=revealed,
            expected_value=expected,
        )


# ---------------------------------------------------------------------------
# Routine decode-tail audit: succinct ZK argmax over the FULL vocab
# (cell sampling cannot catch a 1-of-vocab logit swap; this can, and it
# is context-length independent.  The lm_head fold proof - binding the
# committed logits to the hidden state - composes at wiring via the
# existing succinct fold argument with registry-bound weights.)
# ---------------------------------------------------------------------------

def build_decode_argmax_audit_v3(
    *,
    quantized_logits,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    fused=None,
):
    """Miner-side: succinct proof that the returned token is the argmax
    of the committed logits (ties break toward the lower index).

    Returns ``(statement, proof, winner_index)``; the byte-difference
    witness proves ``logit[j*] - logit[j] - (j < j*) >= 0`` for EVERY
    vocab cell via one packed range LogUp."""

    import numpy as np

    from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (
        GoldilocksSuccinctLogupStatementV3,
        prove_goldilocks_succinct_logup_v3,
    )

    lq = np.asarray(quantized_logits, dtype=np.int64)
    vocab = int(lq.shape[0])
    j_star = int(np.lexsort((np.arange(vocab), -lq))[0])
    diffs = int(lq[j_star]) - lq - (np.arange(vocab) < j_star)
    if int(diffs.min()) < 0:
        raise ProofV3Error("argmax winner is not maximal")
    if int(diffs.max()) >= 1 << 24:
        raise ProofV3Error("argmax differences exceed the byte window")
    bytes_ = np.empty((vocab, 3), dtype=np.int64)
    for limb in range(3):
        bytes_[:, limb] = (diffs >> (8 * limb)) & 0xFF
    byte_values = tuple(bytes_.reshape(-1).tolist())
    statement = GoldilocksSuccinctLogupStatementV3(
        validator_binding_digest=validator_binding_digest,
        table=tuple(range(256)),
        witness_variable_count=(len(byte_values) - 1).bit_length(),
    )
    if fused is not None:
        from verallm.proof_v3.native_pcs_backend import (
            fused_prove_goldilocks_succinct_logup_v3,
        )

        proof = fused_prove_goldilocks_succinct_logup_v3(
            fold_extension=fused[0], tree_extension=fused[1],
            statement=statement, looked_up_values=byte_values,
            validator_nonce=validator_nonce)
    else:
        proof = prove_goldilocks_succinct_logup_v3(
            statement=statement, looked_up_values=byte_values,
            validator_nonce=validator_nonce)
    return statement, proof, j_star


def verify_decode_argmax_audit_v3(
    *,
    statement,
    proof,
    validator_nonce: bytes,
) -> None:
    """Validator-side: check the byte-range LogUp (26ms-class CPU).

    The wiring layer additionally couples the diff bytes to the
    committed logits vector (one eq-fold zero-check against the logits
    capture commitment) and the winner's logit cell to the lm_head row
    (registry reveal) - both ms-class."""

    from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (
        verify_goldilocks_succinct_logup_v3,
    )

    verify_goldilocks_succinct_logup_v3(
        proof, statement=statement,
        witness_commitment=proof.witness_commitment,
        validator_nonce=validator_nonce)


def verify_decode_argmax_reveal_v3(
    *,
    revealed_logits,
    winner_index: int,
    capture_root: bytes,
    validator_binding_digest: bytes,
) -> None:
    """Reveal-based decode check: the FULL committed logits vector.

    Cheaper alternative to the succinct argmax proof: the miner ships
    the padded logits vector (~600KB); the validator rebuilds the
    capture root (so the reveal IS the pre-nonce commitment) and takes
    the max directly.  Zero miner proving cost; combined with the
    lm_head fold proof this closes the targeted logit swap exactly
    like the ZK argmax does.  Ties break toward the lower index."""

    import hashlib as _h
    import struct as _struct

    from verallm.proof_v3.goldilocks_capture_commitment_reference import (
        _CAPTURE_DOMAIN,
        _rebuild_capture_root,
    )

    values = tuple(int(v) for v in revealed_logits)
    rebuilt = None
    try:
        from verallm.proof_v3.c_multiopen import rebuild_root_w1
        from verallm.proof_v3.goldilocks_merkle_reference import (
            _LEAF_DOMAIN,
            _NODE_DOMAIN,
            _ROOT_DOMAIN,
        )

        leaf_binding = _h.sha256(
            _CAPTURE_DOMAIN + b"LEAVES/" + validator_binding_digest
        ).digest()
        header = leaf_binding + _struct.pack("<II", len(values), 1)
        raw_root = rebuild_root_w1(
            _LEAF_DOMAIN + header, _NODE_DOMAIN + header, values)
        if raw_root is not None:
            rebuilt = _h.sha256(
                _ROOT_DOMAIN + header + raw_root).digest()
    except ImportError:
        rebuilt = None
    if rebuilt is None:
        rebuilt = _rebuild_capture_root(
            binding=validator_binding_digest, raw_values=values)
    if rebuilt != capture_root:
        raise ProofV3VerificationError(
            "revealed logits do not match the capture commitment")
    import numpy as np

    from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS

    half = GOLDILOCKS_MODULUS >> 1
    signed = np.array(
        [v - GOLDILOCKS_MODULUS if v > half else v for v in values],
        dtype=np.int64)
    best = int(np.lexsort((np.arange(signed.shape[0]), -signed))[0])
    if best != int(winner_index):
        raise ProofV3VerificationError(
            "returned token is not the argmax of the committed logits")


def compute_projection_factor_v3(weight_int8, u_values):
    """W^T u mod p via exact float64 BLAS (16-bit limb split).

    ``weight_int8``: [rows, cols] int8-range matrix; ``u_values``: one
    field element per row.  Every limb product sum stays far below
    2^53, so the BLAS matvecs are exact; limbs recombine in python
    ints mod p.  Replaces the int64 torch matvec (non-BLAS) that
    dominated the lm_head fold prove."""

    import numpy as np

    from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS

    W = np.asarray(weight_int8, dtype=np.float64)
    if W.shape[0] > 1 << 22:
        raise ProofV3Error("projection factor rows exceed exact float64")
    u = np.asarray(u_values, dtype=np.uint64)
    if u.shape[0] != W.shape[0]:
        raise ProofV3Error("projection factor shapes mismatch")
    parts = []
    for limb in range(4):
        u_limb = ((u >> np.uint64(16 * limb))
                  & np.uint64(0xFFFF)).astype(np.float64)
        parts.append(W.T @ u_limb)
    cols = W.shape[1]
    out = []
    for j in range(cols):
        acc = 0
        for limb in range(4):
            acc += int(parts[limb][j]) << (16 * limb)
        out.append(acc % GOLDILOCKS_MODULUS)
    return tuple(out)
