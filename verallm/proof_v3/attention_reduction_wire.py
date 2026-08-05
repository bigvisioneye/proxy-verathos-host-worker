"""Canonical bounded wire for the reduction-audit reveal.

Deterministic binary encoding (economic_wire ``_Writer``/``_Reader``
discipline, no pickle) of ``ReductionAuditRevealV3``.  This is the
PRODUCTION succinct transport -- distinct from the raw-claims economic
wire; nothing O(context) rides here.  Captured o_proj rows are NOT on
the wire: the validator binds the corridor against its OWN
authenticated ``l{layer}.attn_o_x`` oracle rows.

Every count and width is bounds-checked BEFORE any allocation-heavy
decoding; a reply exceeding any cap fails closed at the reader."""

from __future__ import annotations

import struct
from typing import Final

from verallm.proof_v3.attention_reduction_audit import (
    ReductionAuditRevealV3,
    ReductionGeometryV3,
    ReductionLeafRevealV3,
    ReductionPathRevealV3,
    ReductionPlanV3,
    ReductionSummaryV3,
)
from verallm.proof_v3.economic_wire import (
    _Reader,
    _Writer,
    decode_int8_row_v3,
    encode_int8_row_v3,
)
from verallm.proof_v3.errors import ProofV3Error

__all__ = [
    "MAX_REDUCTION_WIRE_BYTES",
    "MAX_REDUCTION_BUNDLE_WIRE_BYTES",
    "decode_reduction_reveal_v3",
    "encode_reduction_reveal_v3",
    "decode_reduction_bundle_wire_v3",
    "encode_reduction_bundle_wire_v3",
]

MAX_REDUCTION_WIRE_BYTES: Final = 5 << 20      # the design target
_MAX_PAIRS: Final = 64
_MAX_HEADS: Final = 256
_MAX_DIM: Final = 1 << 10
_MAX_CHUNK_LEN: Final = 1 << 12
_MAX_DEPTH: Final = 32
_MAX_LEAVES: Final = 1 << 10
_MAX_KV_CHUNKS: Final = 1 << 9
_MAX_UNIFORM: Final = 64
_MAX_ROWS: Final = 64


def _write_summary(writer: _Writer, summary: ReductionSummaryV3,
                   dim: int) -> None:
    if len(summary.out) != dim:
        raise ProofV3Error("summary width does not match the geometry")
    writer.raw(struct.pack("<qq", summary.peak, summary.mass))
    writer.raw(struct.pack(f"<{dim}q", *summary.out))


def _read_summary(reader: _Reader, dim: int) -> ReductionSummaryV3:
    peak, mass = reader.unpack("<qq")
    out = reader.unpack(f"<{dim}q")
    return ReductionSummaryV3(peak=peak, mass=mass, out=tuple(out))


def _write_leaf(writer: _Writer, leaf: ReductionLeafRevealV3,
                dim: int) -> None:
    if len(leaf.siblings) != len(leaf.directions):
        raise ProofV3Error("leaf path shape mismatch")
    if len(leaf.siblings) > _MAX_DEPTH:
        raise ProofV3Error("leaf path exceeds the depth cap")
    writer.pack(
        "<HQIIH", leaf.head, leaf.row_position, leaf.chunk_index,
        leaf.chunk_len, len(leaf.siblings))
    _write_summary(writer, leaf.summary, dim)
    for (sib_hash, sib_summary), direction in zip(
            leaf.siblings, leaf.directions, strict=True):
        writer.pack("<B", direction)
        writer.raw(sib_hash)
        _write_summary(writer, sib_summary, dim)


def _read_leaf(reader: _Reader, dim: int) -> ReductionLeafRevealV3:
    head, row_position, chunk_index, chunk_len, depth = reader.unpack(
        "<HQIIH")
    if depth > _MAX_DEPTH or chunk_len > _MAX_CHUNK_LEN:
        raise ProofV3Error("leaf reveal exceeds the wire caps")
    summary = _read_summary(reader, dim)
    siblings = []
    directions = []
    for _ in range(depth):
        (direction,) = reader.unpack("<B")
        if direction not in (0, 1):
            raise ProofV3Error("leaf path direction is malformed")
        sib_hash = reader.read(32)
        sib_summary = _read_summary(reader, dim)
        siblings.append((sib_hash, sib_summary))
        directions.append(direction)
    return ReductionLeafRevealV3(
        head=head, row_position=row_position, chunk_index=chunk_index,
        chunk_len=chunk_len, summary=summary,
        siblings=tuple(siblings), directions=tuple(directions))


def encode_reduction_reveal_v3(*, reveal: ReductionAuditRevealV3,
                               plan: ReductionPlanV3) -> bytes:
    """Pack the reveal (and the plan echo the verifier equality-checks).

    o_rows never ride the wire."""

    pairs = tuple(sorted(reveal.geometry))
    if not pairs or len(pairs) > _MAX_PAIRS:
        raise ProofV3Error("reveal pair count is out of bounds")
    dim = reveal.geometry[pairs[0]].head_dim
    writer = _Writer()
    # -- plan echo ------------------------------------------------------
    writer.pack(
        "<IHHHH", plan.layer, len(plan.heads), len(plan.row_positions),
        plan.mass_draws, len(plan.uniform_chunks))
    writer.raw(struct.pack(f"<{len(plan.heads)}H", *plan.heads))
    writer.raw(struct.pack(
        f"<{len(plan.row_positions)}Q", *plan.row_positions))
    for (head, position), chunks in sorted(plan.uniform_chunks.items()):
        chunks = tuple(chunks)
        writer.pack("<HQH", head, position, len(chunks))
        writer.raw(struct.pack(f"<{len(chunks)}I", *chunks))
    # -- membership -----------------------------------------------------
    writer.raw(reveal.layer_root)
    writer.pack("<H", len(reveal.layer_opening))
    for node in reveal.layer_opening:
        writer.raw(node)
    # -- pairs ----------------------------------------------------------
    writer.pack("<H", len(pairs))
    for pair in pairs:
        geometry = reveal.geometry[pair]
        writer.pack(
            "<IHHQQIH", geometry.layer, geometry.head,
            geometry.kv_head, geometry.row_position,
            geometry.key_count, geometry.chunk_len, geometry.head_dim)
        writer.raw(geometry.profile_digest)
        root_hash, root_summary = reveal.roots[pair]
        writer.raw(root_hash)
        _write_summary(writer, root_summary, dim)
        opening = reveal.pair_openings[pair]
        writer.pack("<H", len(opening))
        for node in opening:
            writer.raw(node)
        q_row = reveal.q_rows[pair]
        if len(q_row) != dim:
            raise ProofV3Error("q row width mismatch")
        writer.raw(struct.pack(f"<{dim}q", *q_row))
    # -- kv chunks (int8-packed) ---------------------------------------
    chunk_keys = tuple(sorted(reveal.kv_chunks))
    if len(chunk_keys) > _MAX_KV_CHUNKS:
        raise ProofV3Error("kv chunk count exceeds the wire cap")
    writer.pack("<H", len(chunk_keys))
    for kv_head, chunk_index in chunk_keys:
        k_rows, v_rows = reveal.kv_chunks[(kv_head, chunk_index)]
        if len(k_rows) != len(v_rows):
            raise ProofV3Error("kv chunk row mismatch")
        writer.pack("<HIH", kv_head, chunk_index, len(k_rows))
        for row in k_rows:
            writer.raw(encode_int8_row_v3(row))
        for row in v_rows:
            writer.raw(encode_int8_row_v3(row))
    # -- sampled leaves + paths ----------------------------------------
    if len(reveal.uniform_leaves) > _MAX_LEAVES:
        raise ProofV3Error("uniform leaf count exceeds the wire cap")
    writer.pack("<H", len(reveal.uniform_leaves))
    for leaf in reveal.uniform_leaves:
        _write_leaf(writer, leaf, dim)
    if len(reveal.mass_paths) > _MAX_LEAVES:
        raise ProofV3Error("mass path count exceeds the wire cap")
    writer.pack("<H", len(reveal.mass_paths))
    for path in reveal.mass_paths:
        if len(path.nodes) > _MAX_DEPTH:
            raise ProofV3Error("mass path exceeds the depth cap")
        writer.pack(
            "<HQH", path.head, path.row_position, len(path.nodes))
        for left_h, left_s, right_h, right_s in path.nodes:
            writer.raw(left_h)
            _write_summary(writer, left_s, dim)
            writer.raw(right_h)
            _write_summary(writer, right_s, dim)
        _write_leaf(writer, path.leaf, dim)
    encoded = writer.finish()
    if len(encoded) > MAX_REDUCTION_WIRE_BYTES:
        raise ProofV3Error("reduction wire exceeds the byte budget")
    return encoded


def decode_reduction_reveal_v3(encoded: bytes) -> tuple[
        ReductionAuditRevealV3, ReductionPlanV3]:
    """Decode + bounds-check; returns (reveal, echoed_plan).

    The verifier still derives its OWN plan and requires exact equality
    -- the echo only lets the reveal's plan_key be reconstructed."""

    if len(encoded) > MAX_REDUCTION_WIRE_BYTES:
        raise ProofV3Error("reduction wire exceeds the byte budget")
    reader = _Reader(encoded, "reduction audit wire")
    layer, n_heads_sel, n_rows, mass_draws, n_uniform = reader.unpack(
        "<IHHHH")
    if (n_heads_sel > _MAX_HEADS or n_rows > _MAX_ROWS
            or n_uniform > _MAX_PAIRS or mass_draws > _MAX_UNIFORM):
        raise ProofV3Error("plan echo exceeds the wire caps")
    heads = reader.unpack(f"<{n_heads_sel}H")
    row_positions = reader.unpack(f"<{n_rows}Q")
    uniform: dict = {}
    for _ in range(n_uniform):
        head, position, count = reader.unpack("<HQH")
        if count > _MAX_UNIFORM:
            raise ProofV3Error("uniform chunk count exceeds the cap")
        uniform[(head, position)] = tuple(reader.unpack(f"<{count}I"))
    plan = ReductionPlanV3(
        layer=layer, heads=tuple(int(h) for h in heads),
        row_positions=tuple(int(p) for p in row_positions),
        uniform_chunks=uniform, mass_draws=mass_draws)
    layer_root = reader.read(32)
    (layer_depth,) = reader.unpack("<H")
    if layer_depth > _MAX_DEPTH:
        raise ProofV3Error("layer opening exceeds the depth cap")
    layer_opening = tuple(reader.read(32) for _ in range(layer_depth))
    (n_pairs,) = reader.unpack("<H")
    if not 0 < n_pairs <= _MAX_PAIRS:
        raise ProofV3Error("pair count is out of bounds")
    geometry: dict = {}
    roots: dict = {}
    pair_openings: dict = {}
    q_rows: dict = {}
    dim = None
    for _ in range(n_pairs):
        (g_layer, g_head, g_kv, g_row, g_keys, g_chunk,
         g_dim) = reader.unpack("<IHHQQIH")
        if g_dim > _MAX_DIM or g_chunk > _MAX_CHUNK_LEN:
            raise ProofV3Error("geometry exceeds the wire caps")
        if dim is None:
            dim = g_dim
        elif dim != g_dim:
            raise ProofV3Error("pair geometries disagree on head_dim")
        profile = reader.read(32)
        pair = (g_head, g_row)
        geometry[pair] = ReductionGeometryV3(
            layer=g_layer, head=g_head, kv_head=g_kv,
            row_position=g_row, key_count=g_keys, chunk_len=g_chunk,
            head_dim=g_dim, profile_digest=profile)
        root_hash = reader.read(32)
        root_summary = _read_summary(reader, dim)
        roots[pair] = (root_hash, root_summary)
        (depth,) = reader.unpack("<H")
        if depth > _MAX_DEPTH:
            raise ProofV3Error("pair opening exceeds the depth cap")
        pair_openings[pair] = tuple(
            reader.read(32) for _ in range(depth))
        q_rows[pair] = tuple(reader.unpack(f"<{dim}q"))
    (n_chunks,) = reader.unpack("<H")
    if n_chunks > _MAX_KV_CHUNKS:
        raise ProofV3Error("kv chunk count exceeds the wire cap")
    kv_chunks: dict = {}
    for _ in range(n_chunks):
        kv_head, chunk_index, rows = reader.unpack("<HIH")
        if rows > _MAX_CHUNK_LEN:
            raise ProofV3Error("kv chunk rows exceed the wire cap")
        k_rows = tuple(
            decode_int8_row_v3(reader.read(dim)) for _ in range(rows))
        v_rows = tuple(
            decode_int8_row_v3(reader.read(dim)) for _ in range(rows))
        kv_chunks[(kv_head, chunk_index)] = (k_rows, v_rows)
    (n_leaves,) = reader.unpack("<H")
    if n_leaves > _MAX_LEAVES:
        raise ProofV3Error("uniform leaf count exceeds the wire cap")
    uniform_leaves = tuple(
        _read_leaf(reader, dim) for _ in range(n_leaves))
    (n_paths,) = reader.unpack("<H")
    if n_paths > _MAX_LEAVES:
        raise ProofV3Error("mass path count exceeds the wire cap")
    mass_paths = []
    for _ in range(n_paths):
        head, position, depth = reader.unpack("<HQH")
        if depth > _MAX_DEPTH:
            raise ProofV3Error("mass path exceeds the depth cap")
        nodes = []
        for _level in range(depth):
            left_h = reader.read(32)
            left_s = _read_summary(reader, dim)
            right_h = reader.read(32)
            right_s = _read_summary(reader, dim)
            nodes.append((left_h, left_s, right_h, right_s))
        leaf = _read_leaf(reader, dim)
        mass_paths.append(ReductionPathRevealV3(
            head=head, row_position=position, nodes=tuple(nodes),
            leaf=leaf))
    reader.finish()
    reveal = ReductionAuditRevealV3(
        plan_key=plan.key(), geometry=geometry, roots=roots,
        layer_root=layer_root, layer_opening=layer_opening,
        pair_openings=pair_openings, q_rows=q_rows,
        kv_chunks=kv_chunks, uniform_leaves=uniform_leaves,
        mass_paths=tuple(mass_paths),
        o_rows={pair: () for pair in geometry})
    return reveal, plan


# ---------------------------------------------------------------------------
# bundled one-canary envelope: N domain-separated layer subaudits in ONE
# proof response (derive_reduction_bundle_v3 ordering)
# ---------------------------------------------------------------------------

_BUNDLE_MAGIC: Final = b"VRXB"
_BUNDLE_VERSION: Final = 1
_MAX_SUBAUDITS: Final = 8
MAX_REDUCTION_BUNDLE_WIRE_BYTES: Final = (
    _MAX_SUBAUDITS * MAX_REDUCTION_WIRE_BYTES + 1024)


def encode_reduction_bundle_wire_v3(*, sections) -> bytes:
    """Pack one canary's bundled response: ``sections`` is the
    canonical sequence of (reveal, plan) per subaudit, layer-ascending
    (derive_reduction_bundle_v3's order).  Exact count, strictly
    increasing distinct layers, per-section wire caps -- all enforced
    here and again at decode."""

    items = tuple(sections)
    if not 0 < len(items) <= _MAX_SUBAUDITS:
        raise ProofV3Error("bundle subaudit count is out of bounds")
    layers = tuple(int(plan.layer) for _reveal, plan in items)
    if tuple(sorted(layers)) != layers or len(set(layers)) != len(layers):
        raise ProofV3Error(
            "bundle subaudit layers must be sorted and distinct")
    out = bytearray(_BUNDLE_MAGIC)
    out += struct.pack("<BB", _BUNDLE_VERSION, len(items))
    for reveal, plan in items:
        encoded = encode_reduction_reveal_v3(reveal=reveal, plan=plan)
        out += struct.pack("<I", len(encoded))
        out += encoded
    if len(out) > MAX_REDUCTION_BUNDLE_WIRE_BYTES:
        raise ProofV3Error("bundle wire exceeds the byte budget")
    return bytes(out)


def decode_reduction_bundle_wire_v3(encoded: bytes, *,
                                    expected_subaudits: int) -> tuple:
    """Decode + bounds-check the bundled envelope; returns the tuple
    of (reveal, echoed_plan) per subaudit in wire order.

    Fail-closed: magic/version, EXACT subaudit count (the verifier
    passes the signed policy's selection size -- an omitted or extra
    subaudit is malformed before any section decodes), per-section
    byte bounds, exact consume, strictly increasing distinct echoed
    layers.  The verifier still derives its OWN bundle and requires
    per-subaudit plan equality."""

    if not 0 < int(expected_subaudits) <= _MAX_SUBAUDITS:
        raise ProofV3Error("expected subaudit count is out of bounds")
    if len(encoded) > MAX_REDUCTION_BUNDLE_WIRE_BYTES:
        raise ProofV3Error("bundle wire exceeds the byte budget")
    reader = _Reader(encoded, "reduction bundle wire")
    if reader.read(len(_BUNDLE_MAGIC)) != _BUNDLE_MAGIC:
        raise ProofV3Error("bundle wire magic is malformed")
    version, count = reader.unpack("<BB")
    if version != _BUNDLE_VERSION:
        raise ProofV3Error("bundle wire version is not supported")
    if count != int(expected_subaudits):
        raise ProofV3Error(
            "bundle subaudit count does not match the signed policy")
    sections = []
    for _ in range(count):
        (nbytes,) = reader.unpack("<I")
        if not 0 < nbytes <= MAX_REDUCTION_WIRE_BYTES:
            raise ProofV3Error("bundle section exceeds the byte budget")
        sections.append(decode_reduction_reveal_v3(reader.read(nbytes)))
    reader.finish()
    layers = tuple(int(plan.layer) for _reveal, plan in sections)
    if tuple(sorted(layers)) != layers or len(set(layers)) != len(layers):
        raise ProofV3Error(
            "bundle subaudit layers must be sorted and distinct")
    return tuple(sections)
