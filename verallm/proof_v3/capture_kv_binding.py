"""Capture-KV binding: the tile's K/V come from the capture plane.

Design doc: proof_v3_capture_kv_attention_design.md.  The served K/V is
pre-nonce committed by the capture plane (width-1 value trees).  The PCS
product arguments need K/V as codeword-tree commitments, so the prover
PCS-commits each audited layer's K/V ONCE and proves VALUE EQUALITY to the
capture root at nonce-derived sampled indices: the capture side opens w1
paths (C reconstruct), the PCS side answers the same indices as boolean-
point evaluation claims through the SAME deferred batch opening the tile
already uses.  Equality at j random indices binds the two commitments
w.h.p.; every tile product claim then runs against the PCS column.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS

__all__ = [
    "KV_EQUALITY_SAMPLE_COUNT_V3",
    "CaptureKvEqualityV3",
    "capture_kv_commitment_v3",
    "commit_capture_kv_pcs_v3",
    "fold_capture_kv_commitment_v3",
    "derive_anchor_kv_equality_indices_v3",
    "derive_anchor_q_mle_points_v3",
    "derive_kv_equality_indices_v3",
    "derive_row_opening_indices_v3",
    "prove_kv_equality_v3",
    "verify_kv_equality_v3",
    "verify_row_transport_v3",
]

_CKV_COMMIT_DOMAIN = b"VERATHOS/PROOF_V3/CAPTURE_KV_COMMIT/V1"
_CKV_FOLD_DOMAIN = b"VERATHOS/PROOF_V3/CAPTURE_KV_FOLD/V1"


def fold_capture_kv_commitment_v3(*, base_capture_digest: bytes,
                                  commitment: bytes) -> bytes:
    """THE canonical fold of the capture-kv transport commitment into
    the request's capture chain.

    The miner computes ``capture_chain_digest =
    fold(base, commitment)`` BEFORE the nonce; the chain digest feeds
    the execution root frozen in the authenticated commitment
    envelope. The VERIFIER recomputes this exact fold from the proof's
    attention request section and requires equality with the proof's
    ``capture_chain_digest`` -- the commitment is then authenticated
    by the envelope chain (base -> fold -> execution root -> nonce),
    never self-declared."""

    if len(base_capture_digest) != 32:
        raise ProofV3Error("base capture digest must be 32 bytes")
    if len(commitment) != 32:
        raise ProofV3Error("capture-kv commitment must be 32 bytes")
    return hashlib.sha256(
        _CKV_FOLD_DOMAIN + base_capture_digest + commitment).digest()


def capture_kv_commitment_v3(*, roots_by_layer, capture_binding: bytes,
                             candidate_rows, key_count: int) -> bytes:
    """Canonical PRE-NONCE commitment of one request's capture-kv
    transport inputs.

    Binds, in one digest the prover folds into the capture chain
    BEFORE the nonce: every audited layer's ordered capture roots
    (k, v, ox[, gate]), the tree binding, the candidate pool in pool
    order, and the authenticated key count.  The validator recovers
    this digest from the request's authenticated pre-nonce envelope;
    the economic adapter recomputes it from the artifact-supplied
    pieces and rejects on mismatch -- post-nonce fabrication of any
    input (roots, pool, context length, binding) fails closed."""

    if len(capture_binding) != 32:
        raise ProofV3Error("capture binding must be 32 bytes")
    pool = tuple(int(p) for p in candidate_rows)
    parts = [_CKV_COMMIT_DOMAIN, capture_binding,
             int(key_count).to_bytes(8, "little"),
             len(pool).to_bytes(4, "little")]
    parts.extend(int(p).to_bytes(8, "little") for p in pool)
    layers = sorted(int(x) for x in roots_by_layer)
    parts.append(len(layers).to_bytes(4, "little"))
    for layer in layers:
        roots = tuple(roots_by_layer[layer])
        if not 3 <= len(roots) <= 4:
            raise ProofV3Error(
                "capture-kv commitment needs (k, v, ox[, gate]) roots")
        parts.append(int(layer).to_bytes(4, "little"))
        parts.append(len(roots).to_bytes(1, "little"))
        for root in roots:
            if len(root) != 32:
                raise ProofV3Error("capture root must be 32 bytes")
            parts.append(bytes(root))
    return hashlib.sha256(b"".join(parts)).digest()

_EQ_DOMAIN = b"VERATHOS/PROOF_V3/CAPTURE_KV_EQ/V1"
_ANCHOR_EQ_DOMAIN = (
    b"VERATHOS/PROOF_V3/EXECUTION_ANCHOR_KV_EQ/STREAMING_V1"
)

# --- equality sample count: explicit economic bound -----------------------
# Both roots are fixed before the nonce, so an adversary whose PCS column
# differs from the capture plane in a fraction f of cells survives the
# j-sample equality check with probability (1-f)^j.  The check's job is
# the BULK regime (serving different K/V than captured: f near 1); the
# sparse regime (f below _EQ_FMIN) cannot steer the scored tile's
# committed outputs without breaking the o_proj output bridge, which is
# verified separately.  j is chosen so any f >= _EQ_FMIN escapes with
# probability at most 2^-_EQ_LAMBDA per audited tensor:
#   j = ceil(_EQ_LAMBDA * ln 2 / -ln(1 - _EQ_FMIN))
_EQ_LAMBDA = 20
_EQ_FMIN = 0.125
KV_EQUALITY_SAMPLE_COUNT_V3 = math.ceil(
    _EQ_LAMBDA * math.log(2) / -math.log(1.0 - _EQ_FMIN))  # = 104


@dataclass(frozen=True, slots=True)
class CaptureKvEqualityV3:
    """One tensor's equality bundle: sampled indices + the capture-side
    multiproof (PCS-side answers ride the tile's batch openings)."""

    tag: str
    indices: tuple[int, ...]
    values: tuple[int, ...]          # canonical field values at the indices
    capture_opening: object          # w1 multiproof against the capture root


def commit_capture_kv_pcs_v3(*, tile_digest: bytes, tag: str, values,
                             fused=None):
    """PCS-commit one audited layer's K (or V) as a solo column."""
    from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
        commit_succinct_column_v3,
    )

    return commit_succinct_column_v3(
        tile_digest=tile_digest, tag=tag,
        values=values if not hasattr(values, "numel") else values,
        fused=fused, canonical_input=True)


def derive_kv_equality_indices_v3(*, tile_digest: bytes,
                                  capture_root: bytes, pcs_root: bytes,
                                  validator_nonce: bytes, leaf_count: int,
                                  count: int = KV_EQUALITY_SAMPLE_COUNT_V3,
                                  ) -> tuple[int, ...]:
    """Nonce-derived sampled indices binding BOTH roots (neither side can
    steer the sample)."""
    seed = hashlib.sha256(
        _EQ_DOMAIN + tile_digest + capture_root + pcs_root
        + validator_nonce).digest()
    out = set()
    counter = 0
    while len(out) < min(count, leaf_count):
        block = hashlib.sha256(
            seed + counter.to_bytes(4, "little")).digest()
        for off in range(0, 32, 8):
            out.add(int.from_bytes(block[off:off + 8], "little")
                    % leaf_count)
            if len(out) >= min(count, leaf_count):
                break
        counter += 1
    return tuple(sorted(out))


def derive_anchor_kv_equality_indices_v3(
    *,
    tile_digest: bytes,
    anchor_root: bytes,
    pcs_root: bytes,
    validator_nonce: bytes,
    layer: int,
    tag: str,
    leaf_count: int,
    count: int = KV_EQUALITY_SAMPLE_COUNT_V3,
) -> tuple[int, ...]:
    """Fiat--Shamir K/V equality coordinates for an anchor-backed helper.

    ``anchor_root`` and ``validator_nonce`` select the audited runtime
    statement.  ``pcs_root`` is also mandatory: it freezes the post-nonce
    helper polynomial before its equality coordinates are derived.  Omitting
    it would reveal the checked cells before the helper was committed.
    """

    if (
        not isinstance(tile_digest, bytes)
        or len(tile_digest) != 32
        or not isinstance(anchor_root, bytes)
        or len(anchor_root) != 32
        or not isinstance(pcs_root, bytes)
        or len(pcs_root) != 32
        or not isinstance(validator_nonce, bytes)
        or len(validator_nonce) != 32
        or isinstance(layer, bool)
        or not isinstance(layer, int)
        or not 0 <= layer < 1 << 32
        or tag not in {"k", "v"}
        or isinstance(leaf_count, bool)
        or not isinstance(leaf_count, int)
        or leaf_count <= 0
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
    ):
        raise ProofV3Error(
            "anchor-backed K/V equality sampler input is malformed"
        )
    seed = hashlib.sha256(
        _ANCHOR_EQ_DOMAIN
        + tile_digest
        + anchor_root
        + pcs_root
        + validator_nonce
        + int(layer).to_bytes(4, "little")
        + tag.encode("ascii")
    ).digest()
    out = set()
    counter = 0
    while len(out) < min(count, leaf_count):
        block = hashlib.sha256(
            seed + counter.to_bytes(4, "little")
        ).digest()
        for offset in range(0, 32, 8):
            out.add(
                int.from_bytes(block[offset:offset + 8], "little")
                % leaf_count
            )
            if len(out) >= min(count, leaf_count):
                break
        counter += 1
    return tuple(sorted(out))


def derive_anchor_q_mle_points_v3(
    *,
    tile_digest: bytes,
    anchor_root: bytes,
    attention_commitments_digest: bytes,
    validator_nonce: bytes,
    variable_count: int,
    count: int = 2,
) -> tuple[tuple[int, ...], ...]:
    """Full-row Q equality points for ~128-bit aggregate soundness."""

    if (
        any(
            not isinstance(value, bytes) or len(value) != 32
            for value in (
                tile_digest,
                anchor_root,
                attention_commitments_digest,
                validator_nonce,
            )
        )
        or isinstance(variable_count, bool)
        or not isinstance(variable_count, int)
        or not 1 <= variable_count <= 32
        or count != 2
    ):
        raise ProofV3Error(
            "anchor-backed Q equality sampler input is malformed"
        )
    seed = hashlib.sha256(
        b"VERATHOS/PROOF_V3/EXECUTION_ANCHOR_Q_MLE/STREAMING_V1"
        + tile_digest
        + anchor_root
        + attention_commitments_digest
        + validator_nonce
    ).digest()
    points = []
    for point_index in range(count):
        coordinates = []
        counter = 0
        while len(coordinates) < variable_count:
            block = hashlib.sha256(
                seed
                + point_index.to_bytes(4, "little")
                + counter.to_bytes(4, "little")
            ).digest()
            coordinates.extend(
                int.from_bytes(block[offset:offset + 8], "little")
                % GOLDILOCKS_MODULUS
                for offset in range(0, 32, 8)
            )
            counter += 1
        points.append(tuple(coordinates[:variable_count]))
    return tuple(points)


def _boolean_point(index: int, n_vars: int) -> tuple[int, ...]:
    # batch-opening eq convention: each LATER point coordinate becomes the
    # HIGHER index bit (the prefix/tail doublings append the newest
    # variable as the top bit) -> point[j] = index bit j (LSB-first)
    return tuple((index >> j) & 1 for j in range(n_vars))


def prove_kv_equality_v3(*, tag: str, capture_tree, pcs_column, indices,
                         collector,
                         capture_indices=None) -> CaptureKvEqualityV3:
    """Open the capture side + defer the PCS side at the same values.

    ``capture_indices``: when the pre-nonce capture tree uses a
    DIFFERENT (public) leaf layout than the tile's PCS column -- the
    production case: capture commits the NATIVE GQA (n_kv, sp, d)
    cube pre-nonce, the tile column spans the nonce-SELECTED heads --
    pass the mapped capture leaf per sampled tile leaf.  The verifier
    recomputes the same map from public data; equality holds because
    both sides answer identical values at the mapped positions.
    Defaults to ``indices`` (same-layout capture trees)."""

    n_vars = pcs_column.pcs_statement.variable_count
    if pcs_column.values is not None:
        vals = tuple(
            int(pcs_column.values[i]) % GOLDILOCKS_MODULUS
            for i in indices)
    else:
        dev = pcs_column.device_values
        if dev is None:
            # parked column: the stash answers point reads on host
            dev = pcs_column.device_values_host
        vals = tuple(
            int(v) + (1 << 64) if int(v) < 0 else int(v)
            for v in dev[list(indices)].tolist())
    for i, v in zip(indices, vals, strict=True):
        collector.defer(tag, _boolean_point(i, n_vars), v)
    opening = capture_tree.open(tuple(
        capture_indices if capture_indices is not None else indices))
    return CaptureKvEqualityV3(
        tag=tag, indices=tuple(indices), values=vals,
        capture_opening=opening)


def verify_kv_equality_v3(*, equality: CaptureKvEqualityV3,
                          capture_root: bytes, capture_binding: bytes,
                          heads: int, rows: int, dim: int,
                          pcs_statement, expected_indices,
                          checker, capture_indices=None,
                          capture_leaf_count=None,
                          capture_values=None) -> None:
    """Capture-side reconstruct + PCS-side expectations; equality holds
    because BOTH sides answer the identical values at the identical
    nonce-derived (and, for layered captures, publicly MAPPED)
    positions.

    ``capture_indices``/``capture_leaf_count``: the VALIDATOR-computed
    capture-side leaves (canonical sorted-unique -- distinct tile
    samples may map to ONE capture leaf when selected heads share a
    GQA group) + the capture tree's leaf space when the pre-nonce
    capture layout differs from the tile column.  ``capture_values``:
    the expected value per capture leaf, aligned to
    ``capture_indices`` (the caller derives them from the tile-side
    claimed values through the same public map, consistency-checked).
    All default to the tile's own indices/values/leaf space."""
    import numpy as _np

    from verallm.proof_v3.economic_attention_section import _auth

    if tuple(equality.indices) != tuple(expected_indices):
        raise ProofV3VerificationError(
            "capture-kv equality indices are not the derived sample")
    n_vars = pcs_statement.variable_count
    # PCS side: expectations against the tile's shared batch opening
    for i, v in zip(equality.indices, equality.values, strict=True):
        if not 0 <= int(v) < GOLDILOCKS_MODULUS:
            raise ProofV3VerificationError(
                "capture-kv equality value is not canonical")
        checker.expect(
            equality.tag, _boolean_point(i, n_vars), int(v))
    # capture side: full w1 reconstruct at the (mapped) capture leaves.
    want_capture = tuple(
        capture_indices if capture_indices is not None
        else equality.indices)
    want_values = tuple(
        capture_values if capture_values is not None
        else equality.values)
    if len(want_values) != len(want_capture):
        raise ProofV3VerificationError(
            "capture-kv expected values do not align with the leaves")
    opening = equality.capture_opening
    got = _np.asarray(opening.indices, dtype=_np.int64)
    if not _np.array_equal(
            got, _np.asarray(want_capture, dtype=_np.int64)):
        raise ProofV3VerificationError(
            "capture-kv opening reveals the wrong leaves")
    rows_arr = opening.rows
    if isinstance(rows_arr, _np.ndarray):
        vals = [
            int(v) + (1 << 64) if int(v) < 0 else int(v)
            for v in rows_arr.tolist()]
    else:
        vals = [int(r[0]) % (1 << 64) for r in rows_arr]
    if tuple(v % GOLDILOCKS_MODULUS for v in vals) != tuple(
            int(v) for v in want_values):
        raise ProofV3VerificationError(
            "capture-kv opened values disagree with the claimed values")
    _verify_capture_multiproof(
        capture_root=capture_root, capture_binding=capture_binding,
        leaf_count=(int(capture_leaf_count) if capture_leaf_count
                    is not None else 1 << n_vars),
        indices=want_capture, values=vals, opening=opening)


def derive_row_opening_indices_v3(*, heads, positions, candidate_rows,
                                  head_count: int, head_dim: int,
                                  ) -> tuple[tuple[int, ...], int, int]:
    """Canonical capture-cube leaves for the plan's audited attention
    rows.

    The pre-nonce captured-row trees (o_x under the signed ox scales;
    the fixed-point gate factors on gated models) commit the padded
    (nh_pad, pool_pad, head_dim) scored-domain cube over the CANDIDATE
    pool, in candidate-row order.  Both sides derive the same sorted
    scalar-leaf set from public plan data alone: leaf(h, slot, d) =
    (h * pool_pad + slot) * head_dim + d for every audited head and
    row position.  Returns (indices, leaf_count, pool_pad)."""

    pool = tuple(int(p) for p in candidate_rows)
    nh_pad = 1 << max(0, (int(head_count) - 1).bit_length())
    pool_pad = 1 << max(0, (len(pool) - 1).bit_length())
    head_dim = int(head_dim)
    leaf_count = nh_pad * pool_pad * head_dim
    slots = {}
    for position in positions:
        try:
            slots[int(position)] = pool.index(int(position))
        except ValueError:
            raise ProofV3Error(
                "plan row positions must come from the candidate pool")
    bases = sorted(
        (int(h) * pool_pad + slots[int(p)]) * head_dim
        for h in heads for p in positions)
    if len(set(bases)) != len(bases):
        raise ProofV3Error(
            "row transport heads/positions must be distinct")
    indices = tuple(
        base + d for base in bases for d in range(head_dim))
    return indices, leaf_count, pool_pad


def verify_row_transport_v3(*, opening, capture_root: bytes,
                            capture_binding: bytes, expected_indices,
                            leaf_count: int, value_min: int,
                            value_max: int) -> dict[int, int]:
    """Verify ONE captured-row multiproof; return {leaf: signed int}.

    The transported values ride the opening's width-1 rows.  They are
    accepted only if (a) the revealed leaves are EXACTLY the
    plan-derived cells, (b) every decoded value lies inside the
    scheme's signed range, and (c) the w1 multiproof reconstructs the
    PRE-NONCE capture root under the request binding.  This is the
    production replacement for the qualification harness's trusted
    o_x/gate oracles: rows the root does not commit cannot verify."""
    import numpy as _np

    expected = tuple(int(i) for i in expected_indices)
    got = _np.asarray(opening.indices, dtype=_np.int64)
    if not _np.array_equal(got, _np.asarray(expected, dtype=_np.int64)):
        raise ProofV3VerificationError(
            "captured-row opening reveals the wrong leaves")
    rows = opening.rows
    if isinstance(rows, _np.ndarray):
        raw = [
            int(v) + (1 << 64) if int(v) < 0 else int(v)
            for v in rows.tolist()]
    else:
        raw = []
        for row in rows:
            if len(row) != 1:
                raise ProofV3VerificationError(
                    "captured-row opening rows must be width-1")
            raw.append(int(row[0]) % (1 << 64))
    if len(raw) != len(expected):
        raise ProofV3VerificationError(
            "captured-row opening rows do not align with the leaves")
    half_p = GOLDILOCKS_MODULUS >> 1
    out = {}
    for leaf, value in zip(expected, raw, strict=True):
        canonical = value % GOLDILOCKS_MODULUS
        signed = (canonical - GOLDILOCKS_MODULUS
                  if canonical > half_p else canonical)
        if not int(value_min) <= signed <= int(value_max):
            raise ProofV3VerificationError(
                "captured-row value is outside the signed range")
        out[leaf] = signed
    _verify_capture_multiproof(
        capture_root=capture_root, capture_binding=capture_binding,
        leaf_count=int(leaf_count), indices=expected, values=raw,
        opening=opening)
    return out


def _verify_capture_multiproof(*, capture_root, capture_binding,
                               leaf_count, indices, values, opening):
    import hashlib as _h
    import struct

    from verallm.proof_v3.c_multiopen import (
        reconstruct_w1,
        sibling_coordinates,
    )
    from verallm.proof_v3.economic_attention_section import _leaf_binding
    from verallm.proof_v3.goldilocks_merkle_reference import (
        _LEAF_DOMAIN,
        _NODE_DOMAIN,
        _ROOT_DOMAIN,
    )

    coords = sibling_coordinates(leaf_count, list(indices))
    if coords is None:
        raise ProofV3VerificationError(
            "capture-kv sibling schedule unavailable")
    if tuple((s.level, s.index) for s in opening.siblings) != tuple(coords):
        raise ProofV3VerificationError(
            "capture-kv opening siblings are non-canonical")
    leaf_binding = _leaf_binding(capture_binding)
    header = leaf_binding + struct.pack("<II", leaf_count, 1)
    sib = b"".join(bytes(s.digest) for s in opening.siblings)
    raw = reconstruct_w1(
        _LEAF_DOMAIN + header, _NODE_DOMAIN + header, leaf_count,
        list(indices), [v % (1 << 64) for v in values],
        [c[0] for c in coords], [c[1] for c in coords], sib)
    if not isinstance(raw, bytes):
        raise ProofV3VerificationError(
            "capture-kv multiopen reconstruction failed")
    if _h.sha256(_ROOT_DOMAIN + header + raw).digest() != capture_root:
        raise ProofV3VerificationError(
            "capture-kv opening does not match the capture root")
