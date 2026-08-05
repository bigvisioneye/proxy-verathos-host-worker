"""Canonical bounded wire for RATIONAL (V2) succinct attention proofs.

Deterministic binary encoding (economic_wire ``_Writer``/``_Reader``
discipline, no pickle) of ``GoldilocksSuccinctAttentionProofV3`` and
its full sub-proof graph (eq-folds, product arguments, LogUps, batched
PCS openings, Merkle multi-openings).  V2-scoped and fail-closed:

* a rational proof NEVER carries a division product -- encoding one is
  an error and the decoder cannot represent one;
* every count and width is bounds-checked BEFORE any allocation-heavy
  decoding; a reply exceeding any cap fails closed at the reader;
* statements never ride the wire -- the validator constructs its own
  statement from validator-owned data (signed geometry, nonce-derived
  selection, carried public tables) and verification binds the proof
  to it through the transcript digest.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.economic_wire import _Reader, _Writer
from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.goldilocks_merkle_reference import (
    GoldilocksMerkleMultiOpeningReference,
    GoldilocksMerkleSiblingReference,
)
from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
    GoldilocksMultilinearOpeningProofV3,
)
from verallm.proof_v3.goldilocks_succinct_attention import (
    GoldilocksSuccinctAttentionProofV3,
)
from verallm.proof_v3.goldilocks_succinct_batch_opening import (
    GoldilocksBatchOpeningProofV3,
    GoldilocksDeferredOpeningV3,
)
from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (
    GoldilocksSuccinctLogupProofV3,
    GoldilocksSuccinctLogupSubProofV3,
)
from verallm.proof_v3.goldilocks_succinct_product_argument_reference import (
    GoldilocksSuccinctProductProofV3,
)
from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
    SuccinctEqFoldProofV3,
)

__all__ = [
    "MAX_RATIONAL_SECTION_WIRE_BYTES",
    "MAX_RATIONAL_BUNDLE_WIRE_BYTES",
    "RationalLayerSectionWireV3",
    "decode_rational_attention_proof_v3",
    "encode_rational_attention_proof_v3",
    "decode_rational_bundle_wire_v3",
    "encode_rational_bundle_wire_v3",
    "CaptureKvLayerSectionWireV3",
    "decode_capture_kv_bundle_wire_v3",
    "encode_capture_kv_bundle_wire_v3",
    "decode_capture_kv_bundle_wire_v5",
    "encode_capture_kv_bundle_wire_v5",
    "decode_anchor_capture_kv_bundle_wire_v3",
    "encode_anchor_capture_kv_bundle_wire_v3",
    "capture_kv_bundle_wire_version",
]

MAX_RATIONAL_SECTION_WIRE_BYTES: Final = 64 << 20
_MAX_ROUNDS: Final = 64
_MAX_COLUMNS: Final = 64
_MAX_FOLDS: Final = 192
_MAX_LOGUPS: Final = 16
_MAX_SUBPROOFS: Final = 64
_MAX_OPENINGS: Final = 256
# sized for the row transport's scalar-leaf openings: heads_per_layer
# * row_samples * head_dim (2*8*256 = 4096 on hd-256 models) with
# headroom for hardened sampling policies
_MAX_INDICES: Final = 1 << 13
_MAX_SIBLINGS: Final = 1 << 18
_MAX_LEAF_WIDTH: Final = 1 << 12
# attention-row transport: o_x rows always, gate rows on gated models
_MAX_ROW_OPENINGS: Final = 2
_MAX_TAG: Final = 32
_MAX_PUBLIC: Final = 1 << 22


def _w_hash(w: _Writer, value: bytes) -> None:
    if not isinstance(value, (bytes, bytearray)) or len(value) != 32:
        raise ProofV3Error("wire hash must be exactly 32 bytes")
    w.raw(bytes(value))


def _w_rounds(w: _Writer, rounds, width: int) -> None:
    rounds = tuple(rounds)
    if len(rounds) > _MAX_ROUNDS:
        raise ProofV3Error("round count exceeds the wire cap")
    w.pack("<H", len(rounds))
    for poly in rounds:
        poly = tuple(int(x) for x in poly)
        if len(poly) != width:
            raise ProofV3Error("round polynomial width mismatch")
        w.raw(struct.pack(f"<{width}Q", *poly))


def _r_rounds(r: _Reader, width: int):
    (count,) = r.unpack("<H")
    if count > _MAX_ROUNDS:
        raise ProofV3Error("round count exceeds the wire cap")
    return tuple(tuple(r.unpack(f"<{width}Q")) for _ in range(count))


def _w_multiopen(w: _Writer,
                 opening: GoldilocksMerkleMultiOpeningReference) -> None:
    _w_hash(w, opening.binding_digest)
    indices = tuple(int(i) for i in opening.indices)
    rows = tuple(tuple(int(v) for v in row) for row in opening.rows)
    siblings = tuple(opening.siblings)
    if len(indices) > _MAX_INDICES or len(rows) != len(indices):
        raise ProofV3Error("multiopen index/row shape is out of bounds")
    if opening.leaf_width > _MAX_LEAF_WIDTH:
        raise ProofV3Error("multiopen leaf width exceeds the wire cap")
    if len(siblings) > _MAX_SIBLINGS:
        raise ProofV3Error("multiopen sibling count exceeds the cap")
    w.pack("<QHI", int(opening.leaf_count), int(opening.leaf_width),
           len(indices))
    for index in indices:
        w.pack("<Q", index)
    for row in rows:
        if len(row) != opening.leaf_width:
            raise ProofV3Error("multiopen row width mismatch")
        w.raw(struct.pack(f"<{len(row)}Q", *row))
    w.pack("<I", len(siblings))
    for sibling in siblings:
        w.pack("<HQ", int(sibling.level), int(sibling.index))
        _w_hash(w, sibling.digest)


_SIBLING_STRUCT: Final = struct.Struct("<HQ")
_SIBLING_BYTES: Final = _SIBLING_STRUCT.size + 32


def _r_multiopen(r: _Reader) -> GoldilocksMerkleMultiOpeningReference:
    binding = r.read(32)
    leaf_count, leaf_width, n_idx = r.unpack("<QHI")
    if n_idx > _MAX_INDICES or leaf_width > _MAX_LEAF_WIDTH:
        raise ProofV3Error("multiopen shape exceeds the wire caps")
    # bulk reads: the encoding is a fixed-width array per section, so one
    # read + local slicing replaces a reader call per element (the
    # dominant decode cost); byte layout and truncation behavior are
    # unchanged
    indices = r.unpack(f"<{n_idx}Q")
    if leaf_width:
        flat_rows = r.unpack(f"<{n_idx * leaf_width}Q")
        rows = tuple(
            flat_rows[base : base + leaf_width]
            for base in range(0, n_idx * leaf_width, leaf_width))
    else:
        rows = ((),) * n_idx
    (n_sib,) = r.unpack("<I")
    if n_sib > _MAX_SIBLINGS:
        raise ProofV3Error("multiopen sibling count exceeds the cap")
    blob = r.read(_SIBLING_BYTES * n_sib)
    unpack_from = _SIBLING_STRUCT.unpack_from
    head = _SIBLING_STRUCT.size
    # trusted-wire construction: level/index come from the fixed <HQ>
    # struct and the digest is an exact 32-byte slice; the opening is
    # re-verified downstream (coordinate schedule + Merkle root), so the
    # per-sibling __post_init__ validation is redundant on this decode
    # path (millions of siblings per attention-bundle verify)
    from_trusted = GoldilocksMerkleSiblingReference._from_trusted_wire
    siblings = []
    for offset in range(0, len(blob), _SIBLING_BYTES):
        level, index = unpack_from(blob, offset)
        siblings.append(from_trusted(
            level, index, blob[offset + head : offset + _SIBLING_BYTES]))
    return GoldilocksMerkleMultiOpeningReference(
        binding_digest=binding, leaf_count=leaf_count,
        leaf_width=leaf_width, indices=indices, rows=rows,
        siblings=tuple(siblings))


def _w_pcs_opening(w: _Writer, opening) -> None:
    """Kind-aware: 0 = deferred (claimed value only, resolved by the
    tile's batch openings), 1 = full multilinear opening."""

    if isinstance(opening, GoldilocksDeferredOpeningV3):
        w.pack("<B", 0)
        w.pack("<Q", int(opening.claimed_value))
        return
    w.pack("<B", 1)
    w.pack("<Q", int(opening.claimed_value))
    _w_rounds(w, opening.round_polynomials, 3)
    commitments = tuple(opening.layer_commitments)
    if len(commitments) != len(opening.round_polynomials):
        raise ProofV3Error("mlpcs layer/round count mismatch")
    for commitment in commitments:
        _w_hash(w, commitment)
    w.pack("<Q", int(opening.final_value))
    layer_openings = tuple(opening.layer_openings)
    if len(layer_openings) > _MAX_OPENINGS:
        raise ProofV3Error("mlpcs opening count exceeds the wire cap")
    w.pack("<H", len(layer_openings))
    for item in layer_openings:
        _w_multiopen(w, item)


def _r_pcs_opening(r: _Reader):
    (kind,) = r.unpack("<B")
    if kind == 0:
        (claimed,) = r.unpack("<Q")
        return GoldilocksDeferredOpeningV3(claimed_value=claimed)
    if kind != 1:
        raise ProofV3Error("opening kind is malformed")
    (claimed,) = r.unpack("<Q")
    rounds = _r_rounds(r, 3)
    commitments = tuple(r.read(32) for _ in range(len(rounds)))
    (final_value,) = r.unpack("<Q")
    (n_open,) = r.unpack("<H")
    if n_open > _MAX_OPENINGS:
        raise ProofV3Error("mlpcs opening count exceeds the wire cap")
    layer_openings = tuple(_r_multiopen(r) for _ in range(n_open))
    return GoldilocksMultilinearOpeningProofV3(
        claimed_value=claimed, round_polynomials=rounds,
        layer_commitments=commitments, final_value=final_value,
        layer_openings=layer_openings)


def _w_eqfold(w: _Writer, fold: SuccinctEqFoldProofV3) -> None:
    w.pack("<Q", int(fold.claimed_sum))
    _w_rounds(w, fold.round_polynomials, 4)
    _w_pcs_opening(w, fold.opening)


def _r_eqfold(r: _Reader) -> SuccinctEqFoldProofV3:
    (claimed,) = r.unpack("<Q")
    rounds = _r_rounds(r, 4)
    return SuccinctEqFoldProofV3(
        claimed_sum=claimed, round_polynomials=rounds,
        opening=_r_pcs_opening(r))


def _w_product(w: _Writer,
               proof: GoldilocksSuccinctProductProofV3) -> None:
    w.pack("<Q", int(proof.claimed_sum))
    _w_rounds(w, proof.round_polynomials, 4)
    _w_pcs_opening(w, proof.a_opening)
    _w_pcs_opening(w, proof.b_opening)


def _r_product(r: _Reader) -> GoldilocksSuccinctProductProofV3:
    (claimed,) = r.unpack("<Q")
    rounds = _r_rounds(r, 4)
    return GoldilocksSuccinctProductProofV3(
        claimed_sum=claimed, round_polynomials=rounds,
        a_opening=_r_pcs_opening(r), b_opening=_r_pcs_opening(r))


def _w_logup(w: _Writer, proof: GoldilocksSuccinctLogupProofV3) -> None:
    _w_hash(w, proof.witness_commitment)
    _w_hash(w, proof.multiplicity_commitment)
    inverse = tuple(proof.inverse_commitments)
    sums = tuple(int(v) for v in proof.sums)
    subproofs = tuple(proof.subproofs)
    if len(inverse) > _MAX_LOGUPS or len(sums) > _MAX_LOGUPS:
        raise ProofV3Error("logup commitment/sum count exceeds the cap")
    if len(subproofs) > _MAX_SUBPROOFS:
        raise ProofV3Error("logup subproof count exceeds the cap")
    w.pack("<B", len(inverse))
    for commitment in inverse:
        _w_hash(w, commitment)
    w.pack("<B", len(sums))
    for value in sums:
        w.pack("<Q", value)
    w.pack("<B", len(subproofs))
    for sub in subproofs:
        w.pack("<Q", int(sub.claimed_sum))
        _w_rounds(w, sub.round_polynomials, 4)
        openings = tuple(sub.openings)
        if len(openings) > _MAX_LOGUPS:
            raise ProofV3Error(
                "logup subproof opening count exceeds the cap")
        w.pack("<B", len(openings))
        for opening in openings:
            _w_pcs_opening(w, opening)


def _r_logup(r: _Reader) -> GoldilocksSuccinctLogupProofV3:
    witness = r.read(32)
    multiplicity = r.read(32)
    (n_inv,) = r.unpack("<B")
    if n_inv > _MAX_LOGUPS:
        raise ProofV3Error("logup commitment count exceeds the cap")
    inverse = tuple(r.read(32) for _ in range(n_inv))
    (n_sums,) = r.unpack("<B")
    if n_sums > _MAX_LOGUPS:
        raise ProofV3Error("logup sum count exceeds the cap")
    sums = tuple(r.unpack("<Q")[0] for _ in range(n_sums))
    (n_sub,) = r.unpack("<B")
    if n_sub > _MAX_SUBPROOFS:
        raise ProofV3Error("logup subproof count exceeds the cap")
    subproofs = []
    for _ in range(n_sub):
        (claimed,) = r.unpack("<Q")
        rounds = _r_rounds(r, 4)
        (n_open,) = r.unpack("<B")
        if n_open > _MAX_LOGUPS:
            raise ProofV3Error(
                "logup subproof opening count exceeds the cap")
        openings = tuple(_r_pcs_opening(r) for _ in range(n_open))
        subproofs.append(GoldilocksSuccinctLogupSubProofV3(
            claimed_sum=claimed, round_polynomials=rounds,
            openings=openings))
    return GoldilocksSuccinctLogupProofV3(
        witness_commitment=witness,
        multiplicity_commitment=multiplicity,
        inverse_commitments=inverse, sums=sums,
        subproofs=tuple(subproofs))


def _w_batch(w: _Writer, entry) -> None:
    tag, proof = entry
    encoded = str(tag).encode()
    if not 0 < len(encoded) <= _MAX_TAG:
        raise ProofV3Error("batch opening tag is out of bounds")
    w.pack("<B", len(encoded))
    w.raw(encoded)
    w.pack("<Q", 0)  # reserved
    _w_rounds(w, proof.round_polynomials, 4)
    _w_pcs_opening(w, proof.opening)


def _r_batch(r: _Reader):
    (tag_len,) = r.unpack("<B")
    if not 0 < tag_len <= _MAX_TAG:
        raise ProofV3Error("batch opening tag is out of bounds")
    tag = r.read(tag_len).decode()
    r.unpack("<Q")  # reserved
    rounds = _r_rounds(r, 4)
    return (tag, GoldilocksBatchOpeningProofV3(
        round_polynomials=rounds, opening=_r_pcs_opening(r)))


def encode_rational_attention_proof_v3(
        proof: GoldilocksSuccinctAttentionProofV3) -> bytes:
    """Canonical bytes for one rational chunk proof.  Fail-closed: a
    division product (a V1 artifact) cannot be encoded."""

    if proof.division_product is not None:
        raise ProofV3Error(
            "rational proofs never carry a division product")
    w = _Writer()
    commitments = tuple(proof.column_commitments)
    if not 0 < len(commitments) <= _MAX_COLUMNS:
        raise ProofV3Error("column commitment count is out of bounds")
    w.pack("<H", len(commitments))
    for commitment in commitments:
        _w_hash(w, commitment)
    eq_folds = tuple(proof.eq_folds)
    if len(eq_folds) > _MAX_FOLDS:
        raise ProofV3Error("eq-fold count exceeds the wire cap")
    w.pack("<H", len(eq_folds))
    for fold in eq_folds:
        _w_eqfold(w, fold)
    public_folds = tuple(proof.public_folds)
    if len(public_folds) > _MAX_FOLDS:
        raise ProofV3Error("public-fold count exceeds the wire cap")
    w.pack("<H", len(public_folds))
    for fold in public_folds:
        _w_eqfold(w, fold)
    _w_product(w, proof.score_product)
    _w_product(w, proof.pv_product)
    logups = tuple(proof.logups)
    if len(logups) > _MAX_LOGUPS:
        raise ProofV3Error("logup count exceeds the wire cap")
    w.pack("<B", len(logups))
    for logup in logups:
        _w_logup(w, logup)
    scored = tuple(proof.scored_products)
    if len(scored) != 4:
        raise ProofV3Error("scored product set must have four entries")
    for product in scored:
        _w_product(w, product)
    batch = proof.batch_openings
    if batch is None:
        w.pack("<H", 0xFFFF)
    else:
        batch = tuple(batch)
        if len(batch) >= 0xFFFF or len(batch) > _MAX_COLUMNS:
            raise ProofV3Error(
                "batch opening count exceeds the wire cap")
        w.pack("<H", len(batch))
        for entry in batch:
            _w_batch(w, entry)
    for name in ("chunk_totals", "partial_out"):
        values = getattr(proof, name)
        if values is None:
            w.pack("<I", 0xFFFFFFFF)
            continue
        values = tuple(int(v) for v in values)
        if len(values) > _MAX_PUBLIC:
            raise ProofV3Error("public table exceeds the wire cap")
        w.pack("<I", len(values))
        w.raw(struct.pack(f"<{len(values)}Q", *values))
    encoded = w.finish()
    if len(encoded) > MAX_RATIONAL_SECTION_WIRE_BYTES:
        raise ProofV3Error("rational proof wire exceeds the byte budget")
    return encoded


def decode_rational_attention_proof_v3(
        encoded: bytes) -> GoldilocksSuccinctAttentionProofV3:
    """Decode + bounds-check one rational chunk proof.  The decoded
    proof structurally cannot carry a division product."""

    if len(encoded) > MAX_RATIONAL_SECTION_WIRE_BYTES:
        raise ProofV3Error("rational proof wire exceeds the byte budget")
    r = _Reader(encoded, "rational attention proof wire")
    (n_col,) = r.unpack("<H")
    if not 0 < n_col <= _MAX_COLUMNS:
        raise ProofV3Error("column commitment count is out of bounds")
    commitments = tuple(r.read(32) for _ in range(n_col))
    (n_eq,) = r.unpack("<H")
    if n_eq > _MAX_FOLDS:
        raise ProofV3Error("eq-fold count exceeds the wire cap")
    eq_folds = tuple(_r_eqfold(r) for _ in range(n_eq))
    (n_pub,) = r.unpack("<H")
    if n_pub > _MAX_FOLDS:
        raise ProofV3Error("public-fold count exceeds the wire cap")
    public_folds = tuple(_r_eqfold(r) for _ in range(n_pub))
    score_product = _r_product(r)
    pv_product = _r_product(r)
    (n_logup,) = r.unpack("<B")
    if n_logup > _MAX_LOGUPS:
        raise ProofV3Error("logup count exceeds the wire cap")
    logups = tuple(_r_logup(r) for _ in range(n_logup))
    scored_products = tuple(_r_product(r) for _ in range(4))
    (n_batch,) = r.unpack("<H")
    if n_batch == 0xFFFF:
        batch_openings = None
    elif n_batch > _MAX_COLUMNS:
        raise ProofV3Error("batch opening count exceeds the wire cap")
    else:
        batch_openings = tuple(_r_batch(r) for _ in range(n_batch))
    publics = []
    for _name in ("chunk_totals", "partial_out"):
        (count,) = r.unpack("<I")
        if count == 0xFFFFFFFF:
            publics.append(None)
            continue
        if count > _MAX_PUBLIC:
            raise ProofV3Error("public table exceeds the wire cap")
        publics.append(tuple(r.unpack(f"<{count}Q")))
    r.finish()
    return GoldilocksSuccinctAttentionProofV3(
        column_commitments=commitments, eq_folds=eq_folds,
        public_folds=public_folds, score_product=score_product,
        pv_product=pv_product, division_product=None, logups=logups,
        scored_products=scored_products, batch_openings=batch_openings,
        chunk_totals=publics[0], partial_out=publics[1])


# ---------------------------------------------------------------------------
# bundled one-canary envelope: one V2 layer section per signed-policy-
# selected full-attention layer (VRXB version 2)
# ---------------------------------------------------------------------------

_BUNDLE_MAGIC: Final = b"VRXB"
_RATIONAL_BUNDLE_VERSION: Final = 2
_RATIONAL_BUNDLE_VERSION_MERGED: Final = 3
# versions 4/5 were the pre-transport capture-KV envelopes (attention
# o_x/gate rows arrived through validator-side oracles); 6/7 carry the
# rows as wire multiproofs against the pre-nonce captured-row trees.
# Old envelopes reject cleanly with a version error.
_RATIONAL_BUNDLE_VERSION_CAPTURE_KV: Final = 6
_RATIONAL_BUNDLE_VERSION_CAPTURE_KV_BATCHED: Final = 7
# versions 8/9 use the same bounded section encoding but additionally require
# two Q-column MLE claims and anchor-derived K/V equality indices.  The
# version bit is the fail-closed statement marker; old verifiers cannot
# silently accept a bundle without the anchor claims.
_RATIONAL_BUNDLE_VERSION_ANCHOR_CAPTURE_KV: Final = 8
_RATIONAL_BUNDLE_VERSION_ANCHOR_CAPTURE_KV_BATCHED: Final = 9
_MAX_LAYER_OPENINGS: Final = 1 << 10
_MAX_EQ_SAMPLES: Final = 1 << 12
_MAX_SECTIONS: Final = 8
_MAX_SECTION_CHUNKS: Final = 1 << 10
_MAX_SECTION_PUBLIC: Final = 1 << 12
MAX_RATIONAL_BUNDLE_WIRE_BYTES: Final = (
    _MAX_SECTIONS * MAX_RATIONAL_SECTION_WIRE_BYTES + (1 << 20))


@dataclass(frozen=True, slots=True)
class RationalLayerSectionWireV3:
    """ONE selected layer's V2 section as it rides the envelope.

    ``public_totals``/``public_peaks`` are the prover-declared GLOBAL
    row aggregates (hp*tp canonical field ints) the chunk proofs
    jointly authenticate (row-sum folds + exact aggregate equality +
    selector coverage); ``chunk_sel_counts[c]`` marks each row's
    peak-achiever chunk; ``chunk_proofs[c]`` is the canonical encoded
    rational proof for chunk c.  Statements NEVER ride the wire.

    ``layer_openings`` (envelope v3, MERGED sections): the layer's
    ONE shared batch-opening set over the cross-chunk group trees --
    chunk proofs then carry only deferred claims for the merged
    groups.  Empty on v2 (standalone-chunk) sections."""

    layer: int
    key_count: int
    chunk_len: int
    public_totals: tuple[int, ...]
    public_peaks: tuple[int, ...]
    chunk_sel_counts: tuple[tuple[int, ...], ...]
    chunk_proofs: tuple[bytes, ...]
    layer_openings: tuple = ()


def encode_rational_bundle_wire_v3(*, sections) -> bytes:
    """Pack one canary's V2 bundle: layer-ascending distinct sections,
    exact counts, per-section and whole-envelope caps.

    Envelope version is derived from the sections: version 3 when
    EVERY section carries a shared layer batch-opening set (merged
    cross-chunk trees), version 2 when NONE does; mixing is
    malformed."""

    items = tuple(sections)
    if not 0 < len(items) <= _MAX_SECTIONS:
        raise ProofV3Error("bundle section count is out of bounds")
    layers = tuple(int(s.layer) for s in items)
    if tuple(sorted(layers)) != layers or len(set(layers)) != len(layers):
        raise ProofV3Error(
            "bundle section layers must be sorted and distinct")
    merged_flags = tuple(
        bool(getattr(s, "layer_openings", ())) for s in items)
    if any(merged_flags) and not all(merged_flags):
        raise ProofV3Error(
            "bundle sections mix merged and standalone forms")
    version = (_RATIONAL_BUNDLE_VERSION_MERGED if all(merged_flags)
               else _RATIONAL_BUNDLE_VERSION)
    w = _Writer()
    w.raw(_BUNDLE_MAGIC)
    w.pack("<BB", version, len(items))
    for section in items:
        totals = tuple(int(v) for v in section.public_totals)
        peaks = tuple(int(v) for v in section.public_peaks)
        if not 0 < len(totals) <= _MAX_SECTION_PUBLIC or (
                len(peaks) != len(totals)):
            raise ProofV3Error(
                "bundle section public tables are out of bounds")
        proofs = tuple(section.chunk_proofs)
        sels = tuple(tuple(int(v) for v in row)
                     for row in section.chunk_sel_counts)
        if not 0 < len(proofs) <= _MAX_SECTION_CHUNKS or (
                len(sels) != len(proofs)):
            raise ProofV3Error(
                "bundle section chunk shape is out of bounds")
        w.pack("<IQI", int(section.layer), int(section.key_count),
               int(section.chunk_len))
        w.pack("<I", len(totals))
        w.raw(struct.pack(f"<{len(totals)}Q", *totals))
        w.raw(struct.pack(f"<{len(peaks)}Q", *peaks))
        w.pack("<H", len(proofs))
        for sel, proof in zip(sels, proofs, strict=True):
            if len(sel) != len(totals) or any(
                    v not in (0, 1) for v in sel):
                raise ProofV3Error(
                    "bundle section selector table is malformed")
            w.raw(bytes(sel))
            if not 0 < len(proof) <= MAX_RATIONAL_SECTION_WIRE_BYTES:
                raise ProofV3Error(
                    "bundle section proof exceeds the byte budget")
            w.pack("<I", len(proof))
            w.raw(bytes(proof))
        if version == _RATIONAL_BUNDLE_VERSION_MERGED:
            openings = tuple(section.layer_openings)
            if not 0 < len(openings) <= _MAX_LAYER_OPENINGS:
                raise ProofV3Error(
                    "bundle section layer openings are out of bounds")
            w.pack("<H", len(openings))
            for entry in openings:
                _w_batch(w, entry)
    encoded = w.finish()
    if len(encoded) > MAX_RATIONAL_BUNDLE_WIRE_BYTES:
        raise ProofV3Error("bundle wire exceeds the byte budget")
    return encoded


def decode_rational_bundle_wire_v3(encoded: bytes, *,
                                   expected_layers) -> tuple:
    """Decode + bounds-check; the section layer sequence must equal
    the signed selection EXACTLY (count and order) before any proof
    bytes decode."""

    expected = tuple(int(x) for x in expected_layers)
    if not 0 < len(expected) <= _MAX_SECTIONS:
        raise ProofV3Error("expected layer count is out of bounds")
    if len(encoded) > MAX_RATIONAL_BUNDLE_WIRE_BYTES:
        raise ProofV3Error("bundle wire exceeds the byte budget")
    r = _Reader(encoded, "rational bundle wire")
    if r.read(len(_BUNDLE_MAGIC)) != _BUNDLE_MAGIC:
        raise ProofV3Error("bundle wire magic is malformed")
    version, count = r.unpack("<BB")
    if version not in (_RATIONAL_BUNDLE_VERSION,
                       _RATIONAL_BUNDLE_VERSION_MERGED):
        raise ProofV3Error("bundle wire version is not supported")
    if count != len(expected):
        raise ProofV3Error(
            "bundle section count does not match the signed selection")
    sections = []
    for index in range(count):
        layer, key_count, chunk_len = r.unpack("<IQI")
        if layer != expected[index]:
            raise ProofV3Error(
                "bundle section layers do not match the signed "
                "selection")
        (n_pub,) = r.unpack("<I")
        if not 0 < n_pub <= _MAX_SECTION_PUBLIC:
            raise ProofV3Error(
                "bundle section public tables are out of bounds")
        totals = tuple(r.unpack(f"<{n_pub}Q"))
        peaks = tuple(r.unpack(f"<{n_pub}Q"))
        (n_chunks,) = r.unpack("<H")
        if not 0 < n_chunks <= _MAX_SECTION_CHUNKS:
            raise ProofV3Error(
                "bundle section chunk shape is out of bounds")
        sels = []
        proofs = []
        for _ in range(n_chunks):
            sel = tuple(r.read(n_pub))
            if any(v not in (0, 1) for v in sel):
                raise ProofV3Error(
                    "bundle section selector table is malformed")
            (nbytes,) = r.unpack("<I")
            if not 0 < nbytes <= MAX_RATIONAL_SECTION_WIRE_BYTES:
                raise ProofV3Error(
                    "bundle section proof exceeds the byte budget")
            sels.append(sel)
            proofs.append(r.read(nbytes))
        openings = ()
        if version == _RATIONAL_BUNDLE_VERSION_MERGED:
            (n_open,) = r.unpack("<H")
            if not 0 < n_open <= _MAX_LAYER_OPENINGS:
                raise ProofV3Error(
                    "bundle section layer openings are out of bounds")
            openings = tuple(_r_batch(r) for _ in range(n_open))
        sections.append(RationalLayerSectionWireV3(
            layer=layer, key_count=key_count, chunk_len=chunk_len,
            public_totals=totals, public_peaks=peaks,
            chunk_sel_counts=tuple(sels), chunk_proofs=tuple(proofs),
            layer_openings=openings))
    r.finish()
    return tuple(sections)


# ---------------------------------------------------------------------------
# capture-KV one-canary envelope (VRXB version 4): ONE non-chunked
# rational tile per selected layer; K/V PCS-committed once and
# equality-linked to the pre-nonce capture roots (the long-context
# production design -- wire is log in KEYS)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CaptureKvLayerSectionWireV3:
    """ONE selected layer's capture-KV V2 section.

    ``proof`` is the single whole-range rational chunk proof
    (chunk_base 0, key_count = the full sequence).  ``kv_roots`` are
    the miner's per-layer K/V PCS tree roots (bound to the PRE-NONCE
    capture roots by ``eq_k``/``eq_v``; the capture roots themselves
    come from the validator envelope, never the wire).  ``openings``
    is the layer's batch-opening set (k/v claims ride it).
    ``row_openings`` is the attention-row transport: w1 multiproofs of
    the plan's audited o_x rows (and, on gated models, the fixed-point
    gate rows) against the layer's pre-nonce captured-row trees --
    the values ride the width-1 rows of the openings themselves."""

    layer: int
    key_count: int
    public_totals: tuple[int, ...]
    public_peaks: tuple[int, ...]
    public_sel_count: tuple[int, ...]
    proof: bytes
    kv_roots: tuple[bytes, bytes]
    eq_indices: tuple[tuple[int, ...], tuple[int, ...]]
    eq_values: tuple[tuple[int, ...], tuple[int, ...]]
    eq_openings: tuple
    openings: tuple
    row_openings: tuple = ()
    # batched envelope: THIS layer's batched opening set (one
    # lockstep+RLC chain per section keeps prover memory bounded;
    # whole-canary sharing measured OOM at 1M)
    batched_openings: object = None


def _w_equality(w: _Writer, indices, values, opening) -> None:
    indices = tuple(int(i) for i in indices)
    values = tuple(int(v) for v in values)
    if not 0 < len(indices) <= _MAX_EQ_SAMPLES or (
            len(values) != len(indices)):
        raise ProofV3Error("kv equality sample shape is out of bounds")
    w.pack("<H", len(indices))
    w.raw(struct.pack(f"<{len(indices)}Q", *indices))
    w.raw(struct.pack(f"<{len(values)}Q", *values))
    _w_multiopen(w, opening)


def _r_equality(r: _Reader):
    (n,) = r.unpack("<H")
    if not 0 < n <= _MAX_EQ_SAMPLES:
        raise ProofV3Error("kv equality sample shape is out of bounds")
    indices = tuple(r.unpack(f"<{n}Q"))
    values = tuple(r.unpack(f"<{n}Q"))
    return indices, values, _r_multiopen(r)


def _w_capture_section(w: _Writer, section, *, with_openings: bool) -> None:
    totals = tuple(int(v) for v in section.public_totals)
    peaks = tuple(int(v) for v in section.public_peaks)
    sel = tuple(int(v) for v in section.public_sel_count)
    if not 0 < len(totals) <= _MAX_SECTION_PUBLIC or (
            len(peaks) != len(totals) or len(sel) != len(totals)):
        raise ProofV3Error(
            "bundle section public tables are out of bounds")
    if any(v not in (0, 1) for v in sel):
        raise ProofV3Error(
            "bundle section selector table is malformed")
    proof = bytes(section.proof)
    if not 0 < len(proof) <= MAX_RATIONAL_SECTION_WIRE_BYTES:
        raise ProofV3Error(
            "bundle section proof exceeds the byte budget")
    w.pack("<IQ", int(section.layer), int(section.key_count))
    w.pack("<I", len(totals))
    w.raw(struct.pack(f"<{len(totals)}Q", *totals))
    w.raw(struct.pack(f"<{len(peaks)}Q", *peaks))
    w.raw(bytes(sel))
    w.pack("<I", len(proof))
    w.raw(proof)
    for root in section.kv_roots:
        _w_hash(w, root)
    for (indices, values, opening) in zip(
            section.eq_indices, section.eq_values,
            section.eq_openings, strict=True):
        _w_equality(w, indices, values, opening)
    row_openings = tuple(section.row_openings)
    if not 0 < len(row_openings) <= _MAX_ROW_OPENINGS:
        raise ProofV3Error(
            "bundle section row transport is out of bounds")
    w.pack("<B", len(row_openings))
    for opening in row_openings:
        _w_multiopen(w, opening)
    if with_openings:
        opens = tuple(section.openings)
        if not 0 < len(opens) <= _MAX_LAYER_OPENINGS:
            raise ProofV3Error(
                "bundle section layer openings are out of bounds")
        w.pack("<H", len(opens))
        for entry in opens:
            _w_batch(w, entry)
    else:
        if tuple(section.openings):
            raise ProofV3Error(
                "batched bundle sections must not carry per-column "
                "openings")
        if section.batched_openings is None:
            raise ProofV3Error(
                "batched bundle sections must carry a batched set")
        _w_batched_openings(w, section.batched_openings)


def _r_capture_section(r: _Reader, expected_layer: int, *,
                       with_openings: bool) -> CaptureKvLayerSectionWireV3:
    layer, key_count = r.unpack("<IQ")
    if layer != expected_layer:
        raise ProofV3Error(
            "bundle section layers do not match the signed selection")
    (n_pub,) = r.unpack("<I")
    if not 0 < n_pub <= _MAX_SECTION_PUBLIC:
        raise ProofV3Error(
            "bundle section public tables are out of bounds")
    totals = tuple(r.unpack(f"<{n_pub}Q"))
    peaks = tuple(r.unpack(f"<{n_pub}Q"))
    sel = tuple(r.read(n_pub))
    if any(v not in (0, 1) for v in sel):
        raise ProofV3Error(
            "bundle section selector table is malformed")
    (nbytes,) = r.unpack("<I")
    if not 0 < nbytes <= MAX_RATIONAL_SECTION_WIRE_BYTES:
        raise ProofV3Error(
            "bundle section proof exceeds the byte budget")
    proof = r.read(nbytes)
    k_root = r.read(32)
    v_root = r.read(32)
    eq_i = []
    eq_v = []
    eq_o = []
    for _ in range(2):
        indices, values, opening = _r_equality(r)
        eq_i.append(indices)
        eq_v.append(values)
        eq_o.append(opening)
    (n_rows,) = r.unpack("<B")
    if not 0 < n_rows <= _MAX_ROW_OPENINGS:
        raise ProofV3Error(
            "bundle section row transport is out of bounds")
    row_openings = tuple(_r_multiopen(r) for _ in range(n_rows))
    opens: tuple = ()
    batched_openings = None
    if with_openings:
        (n_open,) = r.unpack("<H")
        if not 0 < n_open <= _MAX_LAYER_OPENINGS:
            raise ProofV3Error(
                "bundle section layer openings are out of bounds")
        opens = tuple(_r_batch(r) for _ in range(n_open))
    else:
        batched_openings = _r_batched_openings(r)
    return CaptureKvLayerSectionWireV3(
        layer=layer, key_count=key_count, public_totals=totals,
        public_peaks=peaks, public_sel_count=sel, proof=proof,
        kv_roots=(k_root, v_root),
        eq_indices=(eq_i[0], eq_i[1]),
        eq_values=(eq_v[0], eq_v[1]),
        eq_openings=(eq_o[0], eq_o[1]), openings=opens,
        row_openings=row_openings,
        batched_openings=batched_openings)


def _checked_sections(sections) -> tuple:
    items = tuple(sections)
    if not 0 < len(items) <= _MAX_SECTIONS:
        raise ProofV3Error("bundle section count is out of bounds")
    layers = tuple(int(s.layer) for s in items)
    if tuple(sorted(layers)) != layers or len(set(layers)) != len(layers):
        raise ProofV3Error(
            "bundle section layers must be sorted and distinct")
    return items


def encode_capture_kv_bundle_wire_v3(*, sections) -> bytes:
    """Pack one canary's capture-KV bundle (envelope version 4)."""

    items = _checked_sections(sections)
    w = _Writer()
    w.raw(_BUNDLE_MAGIC)
    w.pack("<BB", _RATIONAL_BUNDLE_VERSION_CAPTURE_KV, len(items))
    for section in items:
        _w_capture_section(w, section, with_openings=True)
    encoded = w.finish()
    if len(encoded) > MAX_RATIONAL_BUNDLE_WIRE_BYTES:
        raise ProofV3Error("bundle wire exceeds the byte budget")
    return encoded


def decode_capture_kv_bundle_wire_v3(encoded: bytes, *,
                                     expected_layers) -> tuple:
    """Decode + bounds-check a version-4 capture-KV bundle; the
    section layer sequence must equal the signed selection EXACTLY."""

    expected = tuple(int(x) for x in expected_layers)
    if not 0 < len(expected) <= _MAX_SECTIONS:
        raise ProofV3Error("expected layer count is out of bounds")
    if len(encoded) > MAX_RATIONAL_BUNDLE_WIRE_BYTES:
        raise ProofV3Error("bundle wire exceeds the byte budget")
    r = _Reader(encoded, "capture-kv bundle wire")
    if r.read(len(_BUNDLE_MAGIC)) != _BUNDLE_MAGIC:
        raise ProofV3Error("bundle wire magic is malformed")
    version, count = r.unpack("<BB")
    if version != _RATIONAL_BUNDLE_VERSION_CAPTURE_KV:
        raise ProofV3Error("bundle wire version is not supported")
    if count != len(expected):
        raise ProofV3Error(
            "bundle section count does not match the signed selection")
    sections = [
        _r_capture_section(r, expected[index], with_openings=True)
        for index in range(count)
    ]
    r.finish()
    return tuple(sections)


# ---------------------------------------------------------------------------
# capture-KV batched-opening envelope (VRXB version 5): sections as in
# version 4 but WITHOUT per-column opening chains; the whole canary
# opens through ONE bundle-level batched set (lockstep sumchecks + one
# RLC'd FRI layer chain on the canonical shift-chain cosets)
# ---------------------------------------------------------------------------


_MAX_BATCH_COLUMNS: Final = 256
_MAX_BATCH_LAYERS: Final = 40


def _w_tag(w: _Writer, tag: str) -> None:
    encoded = str(tag).encode()
    if not 0 < len(encoded) <= _MAX_TAG:
        raise ProofV3Error("batch opening tag is out of bounds")
    w.pack("<B", len(encoded))
    w.raw(encoded)


def _r_tag(r: _Reader) -> str:
    (tag_len,) = r.unpack("<B")
    if not 0 < tag_len <= _MAX_TAG:
        raise ProofV3Error("batch opening tag is out of bounds")
    return r.read(tag_len).decode()


def _w_batched_openings(w: _Writer, payload) -> None:
    claims = payload["claims"]
    batched = payload["batched"]
    tags = tuple(sorted(claims))
    if not 0 < len(tags) <= _MAX_BATCH_COLUMNS:
        raise ProofV3Error("batched claim column count is out of bounds")
    w.pack("<H", len(tags))
    for tag in tags:
        _w_tag(w, tag)
        _w_rounds(w, claims[tag], 4)
    components = tuple(batched.components)
    if not 0 < len(components) <= _MAX_BATCH_COLUMNS:
        raise ProofV3Error("batched component count is out of bounds")
    w.pack("<H", len(components))
    for component in components:
        _w_tag(w, component.tag)
        _w_rounds(w, component.round_polynomials, 3)
        w.pack("<Q", int(component.final_value))
        _w_multiopen(w, component.base_opening)
    roots = tuple(batched.layer_commitments)
    openings = tuple(batched.layer_openings)
    if not 0 < len(roots) <= _MAX_BATCH_LAYERS or (
            len(openings) != len(roots)):
        raise ProofV3Error("batched layer chain is out of bounds")
    w.pack("<H", len(roots))
    for root in roots:
        _w_hash(w, root)
    for opening in openings:
        _w_multiopen(w, opening)


def _r_batched_openings(r: _Reader) -> dict:
    from verallm.proof_v3.goldilocks_batched_pcs_opening import (
        GoldilocksBatchedComponentOpeningV3,
        GoldilocksBatchedOpeningProofV3,
    )

    (n_tags,) = r.unpack("<H")
    if not 0 < n_tags <= _MAX_BATCH_COLUMNS:
        raise ProofV3Error("batched claim column count is out of bounds")
    claims = {}
    for _ in range(n_tags):
        tag = _r_tag(r)
        if tag in claims:
            raise ProofV3Error("batched claim tags collide")
        claims[tag] = _r_rounds(r, 4)
    (n_components,) = r.unpack("<H")
    if not 0 < n_components <= _MAX_BATCH_COLUMNS:
        raise ProofV3Error("batched component count is out of bounds")
    components = []
    for _ in range(n_components):
        tag = _r_tag(r)
        rounds = _r_rounds(r, 3)
        (final_value,) = r.unpack("<Q")
        components.append(GoldilocksBatchedComponentOpeningV3(
            tag=tag, round_polynomials=rounds,
            final_value=final_value, base_opening=_r_multiopen(r)))
    (n_layers,) = r.unpack("<H")
    if not 0 < n_layers <= _MAX_BATCH_LAYERS:
        raise ProofV3Error("batched layer chain is out of bounds")
    roots = tuple(r.read(32) for _ in range(n_layers))
    openings = tuple(_r_multiopen(r) for _ in range(n_layers))
    return {
        "claims": claims,
        "batched": GoldilocksBatchedOpeningProofV3(
            components=tuple(components),
            layer_commitments=roots,
            layer_openings=openings,
        ),
    }


def encode_capture_kv_bundle_wire_v5(*, sections) -> bytes:
    """Pack one canary's batched capture-KV bundle (envelope v5).

    Every section carries its OWN batched opening set: one layer per
    lockstep+RLC chain keeps prover memory bounded (each layer's device
    values release when its set closes; whole-canary sharing measured
    OOM at 1M keys).
    """

    items = _checked_sections(sections)
    w = _Writer()
    w.raw(_BUNDLE_MAGIC)
    w.pack("<BB", _RATIONAL_BUNDLE_VERSION_CAPTURE_KV_BATCHED, len(items))
    for section in items:
        _w_capture_section(w, section, with_openings=False)
    encoded = w.finish()
    if len(encoded) > MAX_RATIONAL_BUNDLE_WIRE_BYTES:
        raise ProofV3Error("bundle wire exceeds the byte budget")
    return encoded


def decode_capture_kv_bundle_wire_v5(encoded: bytes, *,
                                     expected_layers) -> tuple:
    """Decode + bounds-check a version-5 batched capture-KV bundle."""

    expected = tuple(int(x) for x in expected_layers)
    if not 0 < len(expected) <= _MAX_SECTIONS:
        raise ProofV3Error("expected layer count is out of bounds")
    if len(encoded) > MAX_RATIONAL_BUNDLE_WIRE_BYTES:
        raise ProofV3Error("bundle wire exceeds the byte budget")
    r = _Reader(encoded, "capture-kv bundle wire")
    if r.read(len(_BUNDLE_MAGIC)) != _BUNDLE_MAGIC:
        raise ProofV3Error("bundle wire magic is malformed")
    version, count = r.unpack("<BB")
    if version != _RATIONAL_BUNDLE_VERSION_CAPTURE_KV_BATCHED:
        raise ProofV3Error("bundle wire version is not supported")
    if count != len(expected):
        raise ProofV3Error(
            "bundle section count does not match the signed selection")
    sections = tuple(
        _r_capture_section(r, expected[index], with_openings=False)
        for index in range(count)
    )
    r.finish()
    return sections


def encode_anchor_capture_kv_bundle_wire_v3(
    *, sections, batched: bool
) -> bytes:
    """Pack an anchor-backed capture-KV bundle (envelope v8/v9)."""

    items = _checked_sections(sections)
    writer = _Writer()
    writer.raw(_BUNDLE_MAGIC)
    writer.pack(
        "<BB",
        (
            _RATIONAL_BUNDLE_VERSION_ANCHOR_CAPTURE_KV_BATCHED
            if batched
            else _RATIONAL_BUNDLE_VERSION_ANCHOR_CAPTURE_KV
        ),
        len(items),
    )
    for section in items:
        _w_capture_section(writer, section, with_openings=not batched)
    encoded = writer.finish()
    if len(encoded) > MAX_RATIONAL_BUNDLE_WIRE_BYTES:
        raise ProofV3Error("bundle wire exceeds the byte budget")
    return encoded


def decode_anchor_capture_kv_bundle_wire_v3(
    encoded: bytes,
    *,
    expected_layers,
) -> tuple:
    """Decode an anchor-backed bundle and preserve its v8/v9 mode."""

    expected = tuple(int(value) for value in expected_layers)
    if not 0 < len(expected) <= _MAX_SECTIONS:
        raise ProofV3Error("expected layer count is out of bounds")
    if len(encoded) > MAX_RATIONAL_BUNDLE_WIRE_BYTES:
        raise ProofV3Error("bundle wire exceeds the byte budget")
    reader = _Reader(encoded, "anchor capture-kv bundle wire")
    if reader.read(len(_BUNDLE_MAGIC)) != _BUNDLE_MAGIC:
        raise ProofV3Error("bundle wire magic is malformed")
    version, count = reader.unpack("<BB")
    if version not in (
        _RATIONAL_BUNDLE_VERSION_ANCHOR_CAPTURE_KV,
        _RATIONAL_BUNDLE_VERSION_ANCHOR_CAPTURE_KV_BATCHED,
    ):
        raise ProofV3Error(
            "anchor capture-kv bundle wire version is not supported"
        )
    if count != len(expected):
        raise ProofV3Error(
            "bundle section count does not match the signed selection"
        )
    batched = (
        version == _RATIONAL_BUNDLE_VERSION_ANCHOR_CAPTURE_KV_BATCHED
    )
    sections = tuple(
        _r_capture_section(
            reader,
            expected[index],
            with_openings=not batched,
        )
        for index in range(count)
    )
    reader.finish()
    return sections


def capture_kv_bundle_wire_version(encoded: bytes) -> int:
    """Peek the envelope version byte (dispatch v4 vs v5)."""

    prefix = len(_BUNDLE_MAGIC)
    if len(encoded) < prefix + 1 or encoded[:prefix] != _BUNDLE_MAGIC:
        raise ProofV3Error("bundle wire magic is malformed")
    return encoded[prefix]
