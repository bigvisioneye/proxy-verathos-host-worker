"""Canonical request-bound attention audit section, verified INLINE.

The attention audit is a request-bound proof the hard adapter verifies ITSELF,
not an external assertion. The miner commits post-RoPE Q, K, V and the attention
output (attn_o = o_proj input) as pre-nonce capture roots; the section carries
the nonce-sampled slice Merkle openings against those roots plus the committed
chunk claims. Given only the committed roots + the pre-nonce claims digest the
adapter:

1. checks the claims digest equals the pre-nonce committed digest;
2. AUTHENTICATES the revealed Q rows / K,V chunks / attn_o rows against the
   capture roots (no ``reveals_merkle_verified`` trust boolean);
3. recomputes the sampled chunks from the AUTHENTICATED q/k/v vs the committed
   claims;
4. checks composition against the AUTHENTICATED (opened) attn_o -- output OPENED,
   not re-summed from the same claims (non-tautological).

PRODUCTION: capture commitment runs on the GPU tree (fused_merkle_levels_w1,
byte-identical to the reference) with NO leaf cap, so each tensor is ONE tree
committed + opened on-device -- field encoding, hashing and path extraction all
GPU. A pure-Python reference tree is the fallback when no CUDA device is present
(small-scale tests). A fabricated attention (substituted V) fails the sampled-
chunk recompute or composition against the opened output.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS as _P
from verallm.proof_v3.recompute_audit import (
    _chunk_ranges,
    verify_attention_chunk_reveal_fast_v3,
    verify_capture_slice_reveal_v3,
)

__all__ = [
    "EconomicAttentionLayerRootsV3",
    "EconomicAttentionSectionV3",
    "attention_tensor_binding_v3",
    "attention_layer_commitment_v3",
    "chunk_claims_digest_bytes_v3",
    "commit_attention_layer_v3",
    "build_attention_section_v3",
    "verify_attention_section_v3",
    "verify_attention_output_bridge_v3",
]

_TENSOR_DOMAIN = b"VERATHOS/PROOF_V3/ECON_ATTN_TENSOR/V1"
_CAPTURE_LEAF_PREFIX = b"VERATHOS/PROOF_V3/GOLDILOCKS_CAPTURE/V1/COMMIT/SHA256LEAVES/"


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return max(2, p)


def attention_tensor_binding_v3(*, capture_binding: bytes, layer: int, tag: str) -> bytes:
    """Per-(request, layer, tensor) capture binding digest."""
    return hashlib.sha256(
        _TENSOR_DOMAIN + capture_binding + layer.to_bytes(4, "little") + tag.encode()
    ).digest()


def _leaf_binding(tensor_binding: bytes) -> bytes:
    return hashlib.sha256(_CAPTURE_LEAF_PREFIX + tensor_binding).digest()


def _flat_index(h: int, s: int, d: int, rows: int, dim: int) -> int:
    return (h * rows + s) * dim + d


def _slice_indices(heads, rows, dim, positions) -> tuple[int, ...]:
    idx = []
    for h in range(heads):
        for s in positions:
            base = (h * rows + s) * dim
            idx.extend(range(base, base + dim))
    return tuple(sorted(idx))


# --------------------------------------------------------------------------
# Capture tree: GPU (production) or reference (CPU fallback), byte-identical.
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _CaptureOpening:
    """Lightweight capture multiproof (no reference-cap validation); the native
    reconstruct_w1 reads indices/rows/siblings directly."""
    indices: tuple[int, ...]
    rows: tuple[tuple[int], ...]
    siblings: tuple


class _GpuCaptureTree:
    """One width-1 capture tree committed + openable entirely on GPU."""

    def __init__(self, ext, field_values_cuda, tensor_binding: bytes):
        from verallm.proof_v3.native_cuda_tree_backend import fused_merkle_levels_w1
        import torch

        n = field_values_cuda.numel()
        pad = _pow2(n)
        if pad != n:
            padded = torch.zeros(pad, dtype=torch.int64, device=field_values_cuda.device)
            padded[:n] = field_values_cuda
            field_values_cuda = padded
        self._binding = _leaf_binding(tensor_binding)
        self.leaf_count = pad
        self._values = field_values_cuda
        self.commitment, self._levels = fused_merkle_levels_w1(
            ext, field_values_cuda, binding_digest=self._binding)

    def open(self, indices):
        from verallm.proof_v3.goldilocks_merkle_reference import (
            GoldilocksMerkleSiblingReference,
            _expected_sibling_coordinates,
        )
        import numpy as np
        import torch

        # sorted-unique via numpy (indices may be a large ndarray or tuple);
        # values stay a 1-D int64 ndarray end-to-end (the verifier's _auth
        # consumes ndarrays directly -- no per-leaf python).
        sel = np.unique(np.asarray(indices, dtype=np.int64))
        selected = tuple(int(i) for i in sel)
        sel_vals = self._values[
            torch.from_numpy(sel).to(self._values.device)].cpu().numpy()
        siblings = tuple(
            GoldilocksMerkleSiblingReference(
                level=level, index=index,
                digest=self._levels[level][index * 32:(index + 1) * 32])
            for level, index in _expected_sibling_coordinates(
                leaf_count=self.leaf_count, indices=selected))
        return _CaptureOpening(
            indices=sel, rows=sel_vals, siblings=siblings)


class _RefCaptureTree:
    """Reference width-1 capture tree (CPU fallback, small scale)."""

    def __init__(self, field_values_list, tensor_binding: bytes):
        from verallm.proof_v3.goldilocks_merkle_reference import (
            GoldilocksMerkleTreeReference,
        )
        n = len(field_values_list)
        pad = _pow2(n)
        vals = list(field_values_list) + [0] * (pad - n)
        self._binding = _leaf_binding(tensor_binding)
        self.leaf_count = pad
        self._tree = GoldilocksMerkleTreeReference.from_rows(
            tuple((v,) for v in vals), binding_digest=self._binding)
        self.commitment = self._tree.commitment

    def open(self, indices):
        return self._tree.open(tuple(sorted({int(i) for i in indices})))


def _make_tree(*, ext, tensor, tensor_binding: bytes, heads: int, rows: int, dim: int):
    """Build a capture tree over a [heads][rows][dim] tensor (GPU or reference).

    Field-encodes (signed->Goldilocks) and flattens with index (h*rows+s)*dim+d.
    """
    if ext is not None:
        import torch

        t = tensor if hasattr(tensor, "to") else torch.tensor(tensor, dtype=torch.int64)
        t = t.to("cuda", torch.int64)
        if tuple(t.shape) != (heads, rows, dim):
            raise ProofV3Error("attention tensor has an unexpected shape")
        fv = torch.where(t < 0, t + _P, t).reshape(-1).contiguous()
        return _GpuCaptureTree(ext, fv, tensor_binding)
    # reference path
    from verallm.proof_v3.economic_commitment import signed_to_field_v3
    tl = tensor.tolist() if hasattr(tensor, "tolist") else tensor
    flat = [0] * (heads * rows * dim)
    for h in range(heads):
        for s in range(rows):
            base = (h * rows + s) * dim
            row = tl[h][s]
            for d in range(dim):
                flat[base + d] = signed_to_field_v3(int(row[d]))
    return _RefCaptureTree(flat, tensor_binding)


def _load_ext():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        from verallm.proof_v3.native_cuda_tree_backend import load_tree_kernels
        return load_tree_kernels()
    except Exception:  # noqa: BLE001
        return None


def _digest_from_arrays(statement, chunk_size, key_count, ct, po) -> bytes:
    import numpy as _np

    h = hashlib.sha256()
    h.update(b"VERATHOS/PROOF_V3/ECON_ATTN_CLAIMS/V2")
    h.update(statement.digest())
    h.update(int(chunk_size).to_bytes(4, "little"))
    h.update(int(key_count).to_bytes(8, "little"))
    h.update(_np.asarray(ct.shape, dtype=_np.int64).tobytes())
    h.update(_np.ascontiguousarray(ct, dtype=_np.int64).tobytes())
    h.update(_np.asarray(po.shape, dtype=_np.int64).tobytes())
    h.update(_np.ascontiguousarray(po, dtype=_np.int64).tobytes())
    return h.digest()


def chunk_claims_digest_bytes_v3(statement, claims) -> bytes:
    """Canonical digest over the committed chunk claims (bound pre-nonce).

    Vectorised: numpy parses the nested claims in C and hashes their raw int64
    bytes (memory-bandwidth bound), so a 250k-1M context digest is sub-second
    instead of the old O(context) per-value Python to_bytes loop. Chunk claims
    are non-negative and bounded well under 2^63, so int64 is exact + canonical.
    """
    import numpy as _np

    ct = _np.asarray(claims.chunk_totals, dtype=_np.int64)
    po = _np.asarray(claims.partial_out, dtype=_np.int64)
    return _digest_from_arrays(statement, claims.chunk_size, claims.key_count, ct, po)


@dataclass(frozen=True, slots=True)
class EconomicAttentionLayerRootsV3:
    """Pre-nonce committed capture roots for one audited layer (one tree each).

    ``kv_heads`` is the NATIVE key/value head count: the k/v trees commit
    the model's own GQA layout, never a physically expanded copy.  Query
    head ``h`` reads kv head ``h // (heads // kv_heads)`` -- the mapping is
    public model config, applied by the verifier.  0 means kv_heads==heads
    (MHA / legacy commits)."""
    q_root: bytes
    k_root: bytes
    v_root: bytes
    o_root: bytes
    claims_digest: bytes
    heads: int
    q_rows: int
    key_count: int
    dim: int
    chunk_size: int
    kv_heads: int = 0


@dataclass(frozen=True, slots=True)
class EconomicAttentionSectionV3:
    """Post-nonce reveal (openings authenticated by the verifier vs the roots)."""
    layer: int
    rows: tuple[int, ...]
    chunk_indices: tuple[int, ...]
    q_opening: object
    o_opening: object
    k_openings: tuple[tuple[int, object], ...]
    v_openings: tuple[tuple[int, object], ...]
    claims: object
    statement: object


def commit_attention_layer_v3(*, capture_binding: bytes, layer: int, q, k, v, attn_o,
                              statement, claims, ext="auto",
                              kv_heads: int | None = None) -> tuple:
    """Miner-side ALWAYS-ON: capture-commit Q/K/V/attn_o + claims digest.

    q/attn_o are [heads][q_rows][dim] over the query pool; k/v are
    [kv_heads][key_count][dim] over the key sequence in the model's NATIVE
    GQA layout (tensors on the GPU path).  ``kv_heads=None`` means k/v are
    per-query-head (MHA / legacy callers).  Returns (roots, trees) --
    trees retained for the hard-audit opening.
    """
    if ext == "auto":
        ext = _load_ext()
    heads = statement.head_count
    dim = statement.head_dim
    q_rows = statement.token_count
    key_count = statement.key_count
    kv = heads if kv_heads is None else int(kv_heads)
    if kv < 1 or heads % kv:
        raise ProofV3Error("kv_heads must divide the query head count")

    def _tree(tag, tensor, n_heads, rows):
        b = attention_tensor_binding_v3(capture_binding=capture_binding, layer=layer, tag=tag)
        return _make_tree(ext=ext, tensor=tensor, tensor_binding=b, heads=n_heads, rows=rows, dim=dim)

    trees = {
        "q": _tree("q", q, heads, q_rows), "o": _tree("o", attn_o, heads, q_rows),
        "k": _tree("k", k, kv, key_count), "v": _tree("v", v, kv, key_count),
    }
    meta = EconomicAttentionLayerRootsV3(
        q_root=trees["q"].commitment, k_root=trees["k"].commitment,
        v_root=trees["v"].commitment, o_root=trees["o"].commitment,
        claims_digest=chunk_claims_digest_bytes_v3(statement, claims),
        heads=heads, q_rows=q_rows, key_count=key_count, dim=dim,
        chunk_size=claims.chunk_size,
        kv_heads=0 if kv == heads else kv)
    return meta, trees


def _claims_with_arrays(claims):
    """Section copy of the claims with numpy int64 arrays instead of nested
    tuples. The verifier's ``np.asarray`` then costs nothing (no-copy on an
    ndarray) -- parsing nested python tuples was the dominant O(context) verify
    cost (~1.6s at 30k, ~13s at 250k). The conversion happens ONCE here on the
    prover side (where slow is fine), and it pins the wire direction: claims
    serialize as flat int64 buffers, deserializing straight into numpy."""
    import numpy as _np

    from verallm.proof_v3.recompute_audit import AttentionChunkClaimsV3

    ct, po = claims.chunk_totals, claims.partial_out
    if isinstance(ct, _np.ndarray) and isinstance(po, _np.ndarray):
        return claims
    return AttentionChunkClaimsV3(
        chunk_size=claims.chunk_size, key_count=claims.key_count,
        chunk_totals=_np.asarray(ct, dtype=_np.int64),
        partial_out=_np.asarray(po, dtype=_np.int64))


def build_attention_section_v3(*, capture_binding: bytes, layer: int, statement, claims,
                               flats, rows, chunk_indices,
                               kv_heads: int | None = None) -> EconomicAttentionSectionV3:
    """Miner-side HARD-AUDIT: open the nonce-sampled slices from the trees."""
    heads, dim = statement.head_count, statement.head_dim
    q_rows = statement.token_count
    key_count = statement.key_count
    kv = heads if not kv_heads else int(kv_heads)
    ranges = _chunk_ranges(key_count, claims.chunk_size)
    k_openings, v_openings = [], []
    for ci in chunk_indices:
        base, cnt = ranges[ci]
        keys = range(base, base + cnt)
        k_openings.append((ci, flats["k"].open(_slice_indices(kv, key_count, dim, keys))))
        v_openings.append((ci, flats["v"].open(_slice_indices(kv, key_count, dim, keys))))
    return EconomicAttentionSectionV3(
        layer=layer, rows=tuple(rows), chunk_indices=tuple(chunk_indices),
        q_opening=flats["q"].open(_slice_indices(heads, q_rows, dim, rows)),
        o_opening=flats["o"].open(_slice_indices(heads, q_rows, dim, rows)),
        k_openings=tuple(k_openings), v_openings=tuple(v_openings),
        claims=_claims_with_arrays(claims), statement=statement)


def _auth(root, binding, heads, total_rows, dim, positions, opening):
    """Authenticate a capture slice against ``root`` via the C multiopen
    (reconstruct_w1) -- NO 2^16 reference cap, so real-context single-tree
    capture roots verify. Byte-identical to the reference reconstruction.
    """
    import hashlib
    import struct

    from verallm.proof_v3.c_multiopen import reconstruct_w1, sibling_coordinates
    from verallm.proof_v3.goldilocks_merkle_reference import (
        _LEAF_DOMAIN, _NODE_DOMAIN, _ROOT_DOMAIN,
    )

    import numpy as _np

    positions = list(positions)
    leaf_count = _pow2(heads * total_rows * dim)
    # expected leaf schedule, fully vectorized (h-major, position, dim-inner --
    # ascending because positions are ascending, so identical to the sorted
    # tuple _slice_indices produced)
    pos_arr = _np.asarray(positions, dtype=_np.int64)
    expected_idx = (
        ((_np.arange(heads, dtype=_np.int64)[:, None] * total_rows
          + pos_arr[None, :]) * dim)[:, :, None]
        + _np.arange(dim, dtype=_np.int64)[None, None, :]
    ).reshape(-1)
    got_idx = _np.asarray(opening.indices, dtype=_np.int64)
    if got_idx.shape != expected_idx.shape or not _np.array_equal(got_idx, expected_idx):
        raise ProofV3VerificationError("capture opening reveals the wrong leaves")
    # canonical sibling schedule (validator-derived; opening cannot steer it)
    coords = sibling_coordinates(leaf_count, expected_idx.tolist())
    if coords is None:
        raise ProofV3VerificationError("capture sibling schedule unavailable")
    if tuple((s.level, s.index) for s in opening.siblings) != tuple(coords):
        raise ProofV3VerificationError("capture opening siblings are non-canonical")
    leaf_binding = _leaf_binding(binding)
    header = leaf_binding + struct.pack("<II", leaf_count, 1)
    # GPU trees store field values (0..P, P~2^64) as their two's-complement
    # int64 bit-pattern (values >= 2^63 come back negative); reinterpret to the
    # canonical u64 field element (v & 2^64-1). Reference trees are canonical.
    # Vectorised end-to-end: openings carry either a 1-D int64 ndarray (GPU
    # tree) or nested (value,) tuples (reference tree).
    rows = opening.rows
    if isinstance(rows, _np.ndarray):
        arr = _np.ascontiguousarray(rows, dtype=_np.int64)
        u = arr.view(_np.uint64)
    else:
        try:
            # reference openings carry canonical u64 field values (may be
            # >= 2^63: python ints, so uint64 is the right target dtype)
            u = _np.fromiter(
                (int(r[0]) for r in rows), dtype=_np.uint64, count=len(rows))
        except (TypeError, ValueError, OverflowError):
            try:
                # negative ints = two's-complement bit patterns
                u = _np.fromiter(
                    (int(r[0]) & ((1 << 64) - 1) for r in rows),
                    dtype=_np.uint64, count=len(rows))
            except (TypeError, ValueError, OverflowError):
                raise ProofV3VerificationError(
                    "capture leaf value is not canonical")
        arr = u.view(_np.int64)
    if bool((u >= _np.uint64(_P)).any()):
        raise ProofV3VerificationError("capture leaf value is not canonical")
    sib_digests = b"".join(bytes(s.digest) for s in opening.siblings)
    raw = reconstruct_w1(
        _LEAF_DOMAIN + header, _NODE_DOMAIN + header, leaf_count,
        expected_idx, u,
        [c[0] for c in coords], [c[1] for c in coords], sib_digests)
    if not isinstance(raw, bytes):
        raise ProofV3VerificationError("capture multiopen reconstruction failed")
    actual = hashlib.sha256(_ROOT_DOMAIN + header + raw).digest()
    if actual != root:
        raise ProofV3VerificationError("capture slice does not match the committed root")
    # vectorized field_to_signed: v - P if v > HALF else v.  For v > HALF the
    # two's-complement bit pattern of (v - P) is (v + 2^64 - P) mod 2^64, so an
    # unsigned add + int64 reinterpret is exact for every canonical v.
    half = _np.uint64(_P >> 1)
    wrap = _np.uint64((1 << 64) - _P)
    signed = _np.where(u > half, (u + wrap).view(_np.int64), arr)
    return signed.reshape(heads, len(positions), dim)


def verify_attention_section_v3(*, capture_binding: bytes, roots: EconomicAttentionLayerRootsV3,
                                section: EconomicAttentionSectionV3,
                                expected_claims_digest: bytes) -> None:
    """Validator-side INLINE verification. Raises on ANY mismatch."""
    import numpy as _np

    from verallm.miner.proof_v3_serving import restrict_statement_to_rows_v3
    from verallm.proof_v3.recompute_audit import AttentionChunkClaimsV3

    st, claims = section.statement, section.claims
    heads, dim = st.head_count, st.head_dim
    q_rows_n, key_count = st.token_count, st.key_count
    # parse the full claims to numpy ONCE (used by the digest, the row
    # restriction and the composition) -- the only O(context) work, vectorized.
    _ct_full = _np.asarray(claims.chunk_totals, dtype=_np.int64)
    _po_full = _np.asarray(claims.partial_out, dtype=_np.int64)
    if _digest_from_arrays(st, claims.chunk_size, claims.key_count, _ct_full, _po_full) != expected_claims_digest:
        raise ProofV3VerificationError("attention chunk claims != pre-nonce committed digest")
    if roots.claims_digest != expected_claims_digest:
        raise ProofV3VerificationError("committed roots carry a different claims digest")
    sub_st = restrict_statement_to_rows_v3(st, section.rows)
    _row_idx = list(section.rows)
    _ct = _ct_full[:, :, _row_idx]
    _po = _po_full[:, :, _row_idx, :]
    # Both chunk_totals[ci] and partial_out[ci] are indexed ONLY at the sampled
    # chunk_indices in the recompute below; the global softmax denominator (the
    # only all-chunks use of chunk_totals) is precomputed in numpy and passed to
    # the chunk verifier. Composition uses the numpy _po directly. So building
    # the full O(C) nested tuples was the dominant verify cost (partial_out alone
    # ~9000x slower than needed at C~1e4). Fill only the sampled chunks; every
    # other slot shares one empty placeholder that is never read.
    _sampled = sorted(set(int(ci) for ci in section.chunk_indices))
    _empty = ()
    _ct_slots = [_empty] * _ct.shape[0]
    _po_slots = [_empty] * _po.shape[0]
    for ci in _sampled:
        _ct_slots[ci] = tuple(tuple(r) for r in _ct[ci].tolist())
        _po_slots[ci] = tuple(
            tuple(tuple(c) for c in r) for r in _po[ci].tolist())
    sub_claims = AttentionChunkClaimsV3(
        chunk_size=claims.chunk_size, key_count=claims.key_count,
        chunk_totals=tuple(_ct_slots),
        partial_out=tuple(_po_slots))

    def _bind(tag):
        return attention_tensor_binding_v3(capture_binding=capture_binding, layer=section.layer, tag=tag)

    q_rows = _auth(roots.q_root, _bind("q"), heads, q_rows_n, dim, section.rows, section.q_opening)
    o_rows = _auth(roots.o_root, _bind("o"), heads, q_rows_n, dim, section.rows, section.o_opening)
    ranges = _chunk_ranges(key_count, claims.chunk_size)
    # k/v are committed in the model's NATIVE GQA layout; authenticate at
    # kv_heads width, then expand per query head via the PUBLIC mapping
    # h -> h // (heads // kv). The expansion is a validator-side numpy
    # fancy-index over the audited chunk only.
    kv = roots.kv_heads or heads
    if kv < 1 or heads % kv:
        raise ProofV3VerificationError("roots kv_heads does not divide heads")
    kv_map = _np.arange(heads, dtype=_np.int64) // (heads // kv)
    k_by_chunk, v_by_chunk = {}, {}
    for ci, op in section.k_openings:
        base, cnt = ranges[ci]
        k_by_chunk[ci] = _auth(
            roots.k_root, _bind("k"), kv, key_count, dim,
            range(base, base + cnt), op)[kv_map]
    for ci, op in section.v_openings:
        base, cnt = ranges[ci]
        v_by_chunk[ci] = _auth(
            roots.v_root, _bind("v"), kv, key_count, dim,
            range(base, base + cnt), op)[kv_map]

    # global softmax denominator (sum over ALL chunks at the sampled rows),
    # computed once in numpy and reused for every sampled chunk instead of
    # rebuilt from the nested-tuple claims per chunk.
    _totals = _ct.sum(axis=0)  # [heads, sampled]
    for ci in section.chunk_indices:
        verify_attention_chunk_reveal_fast_v3(
            statement=sub_st, claims=sub_claims, chunk_index=ci,
            q_rows=q_rows, k_chunk=k_by_chunk[ci], v_chunk=v_by_chunk[ci],
            precomputed_totals=_totals)

    # composition from the numpy partials directly (vectorized O(C) sum), vs the
    # AUTHENTICATED opened output -- non-tautological.
    composed = _po.sum(axis=0)  # [heads, sampled, dim]
    if not _np.array_equal(composed, _np.asarray(o_rows, dtype=_np.int64)):
        raise ProofV3VerificationError(
            "attention chunk partials do not compose to the committed (opened) output")
    # the AUTHENTICATED composed output rows: what the runtime-output
    # bridge binds against the captured o_proj input
    return _np.asarray(o_rows, dtype=_np.int64)


def verify_attention_output_bridge_v3(*, section, o_rows, ox8_rows_by_token,
                                      calibration_heads) -> None:
    """RUNTIME-OUTPUT BINDING (release blocker): the section's authenticated
    composed attention output must match the CAPTURED o_proj input rows
    within the SIGNED per-cell and per-row bridge bounds, per head.

    ``o_rows`` is the ndarray ``verify_attention_section_v3`` returned
    ([heads, sampled, dim] -- authenticated + composition-checked);
    ``ox8_rows_by_token`` maps each audited ABSOLUTE token position to the
    authenticated full o_proj input row (heads*dim wide, int8);
    ``calibration_heads[h] = (ScoredHeadParamsV3, ScoredBridgeBoundsV3)``
    from the digest-verified signed calibration blob.
    """
    from verallm.proof_v3.scored_attention_reference import (
        verify_output_bridge_v3,
    )

    st = section.statement
    heads, dim = st.head_count, st.head_dim
    if len(calibration_heads) < heads:
        raise ProofV3VerificationError(
            "calibration does not cover every audited head")
    tokens = tuple(st.query_positions[r] for r in section.rows)
    if len(tokens) != o_rows.shape[1]:
        raise ProofV3VerificationError(
            "bridge token set does not match the audited rows")
    ox_rows = []
    for token in tokens:
        row = ox8_rows_by_token.get(token)
        if row is None:
            raise ProofV3VerificationError(
                "bridge is missing the o_proj capture at an audited token")
        if len(row) < heads * dim:
            raise ProofV3VerificationError(
                "captured o_proj row is narrower than heads*dim")
        ox_rows.append(row)
    for h in range(heads):
        params, bounds = calibration_heads[h]
        if params.head_dim != dim:
            raise ProofV3VerificationError(
                "calibration head_dim does not match the statement")
        verify_output_bridge_v3(
            params=params,
            surrogate_rows=[o_rows[h][i] for i in range(len(tokens))],
            ox8_rows=[
                row[h * dim:(h + 1) * dim] for row in ox_rows],
            bounds=bounds)


def attention_layer_commitment_v3(*, layer: int, roots: EconomicAttentionLayerRootsV3) -> bytes:
    """Pre-nonce commitment over one audited layer's attention capture roots.

    Folded into the request's capture_chain_digest (-> execution_root -> the
    nonce), so the miner cannot fabricate roots post-nonce: q/k/v/attn_o roots
    and the chunk-claims digest are all fixed before the nonce exists.
    """
    return hashlib.sha256(
        b"VERATHOS/PROOF_V3/ECON_ATTN_ROOTS/V2"
        + int(layer).to_bytes(4, "little")
        + roots.q_root + roots.o_root + roots.k_root + roots.v_root
        + roots.claims_digest
        # geometry is part of the commitment: the k/v roots alone do not
        # pin the native kv width the openings must verify against
        + int(roots.heads).to_bytes(4, "little")
        + int(roots.kv_heads).to_bytes(4, "little")
        + int(roots.q_rows).to_bytes(4, "little")
        + int(roots.key_count).to_bytes(8, "little")
        + int(roots.dim).to_bytes(4, "little")
        + int(roots.chunk_size).to_bytes(4, "little")
    ).digest()
