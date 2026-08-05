"""Canonical binary wire for the hard attention audit answer.

Bundles what the validator needs to set on the artifacts before
``verify_economic_recompute_v3``: the post-nonce section (statement,
claims, capture openings), the pre-nonce roots, the root commitment, and
the o_proj bridge opening.  Bounded, deterministic, no pickle -- the
same ``_Writer``/``_Reader`` discipline as ``economic_wire``.

The statement travels WITHOUT its exp table: hard-audit statements must
pin the fixed universal table (``fixed_exp_table_v3``), which is a
protocol constant the decoder reinjects.  Chunked statements never ride
this wire (the section covers the full committed pool).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from verallm.proof_v3.economic_wire import (
    EconomicMerkleOpeningV3,
    EconomicMerkleSiblingV3,
    _Reader,
    _Writer,
)
from verallm.proof_v3.errors import ProofV3Error

__all__ = [
    "MAX_ATTENTION_WIRE_BYTES",
    "AttentionAuditWireV3",
    "apply_attention_audit_wire_v3",
    "build_attention_audit_wire_v3",
    "fetch_attention_audit_wire_v3",
]

MAX_ATTENTION_WIRE_BYTES = 128 << 20
_MAX_POOL = 1 << 12
_MAX_HEADS = 256
_MAX_DIM = 1 << 10
_MAX_CHUNK_OPENINGS = 64
_MAX_CLAIM_CELLS = 1 << 24   # int64 cells across both claim tables
_M64 = (1 << 64) - 1


def _u64_values(rows) -> tuple[int, ...]:
    if hasattr(rows, "tolist"):
        return tuple(int(v) & _M64 for v in rows.tolist())
    return tuple(int(r[0]) & _M64 for r in rows)


@dataclass(frozen=True, slots=True)
class _LegacyOpeningV3:
    """V1 attention-wire opening: indices + u64 values ride the wire.

    The legacy scheme-1 section verifier authenticates the CARRIED
    indices against its own derived slice expectations, so this local
    container keeps the pre-compact encoding; the shared economic
    opening class moved to the compact (index-free) v3 form."""

    binding_digest: bytes
    leaf_count: int
    indices: tuple[int, ...]
    values: tuple[int, ...]
    siblings: tuple[EconomicMerkleSiblingV3, ...]

    def encode(self, writer) -> None:
        writer.raw(self.binding_digest)
        count = len(self.indices)
        writer.pack("<QII", self.leaf_count, count, len(self.siblings))
        writer.raw(struct.pack(f"<{count}Q", *self.indices))
        writer.raw(struct.pack(f"<{count}Q", *self.values))
        for sibling in self.siblings:
            writer.pack("<II", sibling.level, sibling.index)
            writer.raw(sibling.digest)

    @classmethod
    def decode(cls, reader) -> "_LegacyOpeningV3":
        binding_digest = reader.read(32)
        leaf_count, index_count, sibling_count = reader.unpack("<QII")
        if index_count == 0 or index_count > (1 << 22):
            raise ProofV3Error("legacy opening index count is out of range")
        if sibling_count > (1 << 20):
            raise ProofV3Error("legacy opening sibling count is out of range")
        indices = struct.unpack(
            f"<{index_count}Q", reader.read(8 * index_count))
        values = struct.unpack(
            f"<{index_count}Q", reader.read(8 * index_count))
        siblings = []
        for _ in range(sibling_count):
            level, index = reader.unpack("<II")
            siblings.append(EconomicMerkleSiblingV3(
                level=level, index=index, digest=reader.read(32)))
        return cls(
            binding_digest=binding_digest, leaf_count=leaf_count,
            indices=indices, values=values, siblings=tuple(siblings))

    def to_reference(self):
        from verallm.proof_v3.goldilocks_merkle_reference import (
            GoldilocksMerkleMultiOpeningReference,
            GoldilocksMerkleSiblingReference,
        )

        return GoldilocksMerkleMultiOpeningReference(
            binding_digest=self.binding_digest,
            leaf_count=self.leaf_count,
            leaf_width=1,
            indices=self.indices,
            rows=tuple(zip(self.values)),
            siblings=tuple(
                GoldilocksMerkleSiblingReference(
                    level=s.level, index=s.index, digest=s.digest)
                for s in self.siblings),
        )


def _capture_to_wire(opening, *, leaf_count: int,
                     binding: bytes) -> _LegacyOpeningV3:
    return _LegacyOpeningV3(
        binding_digest=binding,
        leaf_count=leaf_count,
        indices=tuple(int(i) for i in opening.indices),
        values=_u64_values(opening.rows),
        siblings=tuple(
            EconomicMerkleSiblingV3(
                level=int(s.level), index=int(s.index),
                digest=bytes(s.digest))
            for s in opening.siblings),
    )


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return max(2, p)


@dataclass(frozen=True, slots=True)
class AttentionAuditWireV3:
    """Decoded (or to-encode) attention audit bundle."""

    layer: int
    rows: tuple[int, ...]
    chunk_indices: tuple[int, ...]
    # statement geometry (exp table = the fixed protocol constant)
    validator_binding_digest: bytes
    head_count: int
    token_count: int
    head_dim: int
    qk_bits: int
    v_bits: int
    shift: int
    score_bits: int
    scale_bits: int
    limb_bits: int
    key_count: int
    query_positions: tuple[int, ...]
    scored: int
    m_nums: tuple[int, ...]
    m_e: int
    capture_kv: int
    # () = the FIXED protocol table (score_bits must be 16; the decoder
    # reinjects the constant -- the 65536-entry table NEVER rides the
    # wire).  Non-empty = small derived table (score_bits <= 12).
    exp_table: tuple[int, ...]
    # claims (flat little-endian int64 buffers)
    claims_chunk_size: int
    chunk_count: int
    chunk_totals: bytes           # [C, H, pool]
    partial_out: bytes            # [C, H, pool, dim]
    # pre-nonce roots + commitment
    q_root: bytes
    k_root: bytes
    v_root: bytes
    o_root: bytes
    claims_digest: bytes
    kv_heads: int
    root_commitment: bytes
    # openings
    q_opening: _LegacyOpeningV3
    o_opening: _LegacyOpeningV3
    k_openings: tuple[tuple[int, _LegacyOpeningV3], ...]
    v_openings: tuple[tuple[int, _LegacyOpeningV3], ...]
    # the bridge feeds the SHARED economic verifier -> compact class
    bridge_opening: EconomicMerkleOpeningV3 | None

    def __post_init__(self) -> None:
        if not 0 < self.head_count <= _MAX_HEADS:
            raise ProofV3Error("attention wire head count out of range")
        if not 0 < self.head_dim <= _MAX_DIM:
            raise ProofV3Error("attention wire head dim out of range")
        if not 0 < self.token_count <= _MAX_POOL:
            raise ProofV3Error("attention wire pool size out of range")
        if len(self.query_positions) != self.token_count:
            raise ProofV3Error("attention wire positions != pool size")
        if self.scored and len(self.m_nums) != self.head_count:
            raise ProofV3Error("attention wire slope count != heads")
        kv = self.kv_heads or self.head_count
        if kv < 1 or self.head_count % kv:
            raise ProofV3Error("attention wire kv_heads does not divide heads")
        cells = self.chunk_count * self.head_count * self.token_count
        if cells * (1 + self.head_dim) > _MAX_CLAIM_CELLS:
            raise ProofV3Error(
                "attention wire claims exceed the cell budget "
                f"({cells * (1 + self.head_dim)} > {_MAX_CLAIM_CELLS})")
        if len(self.chunk_totals) != 8 * cells:
            raise ProofV3Error("attention wire chunk totals buffer size")
        if len(self.partial_out) != 8 * cells * self.head_dim:
            raise ProofV3Error("attention wire partial out buffer size")
        for digest in (self.validator_binding_digest, self.q_root,
                       self.k_root, self.v_root, self.o_root,
                       self.claims_digest, self.root_commitment):
            if not isinstance(digest, bytes) or len(digest) != 32:
                raise ProofV3Error("attention wire digest is malformed")
        if len(self.k_openings) > _MAX_CHUNK_OPENINGS or len(
                self.v_openings) > _MAX_CHUNK_OPENINGS:
            raise ProofV3Error("attention wire chunk opening count")
        if self.exp_table:
            if self.score_bits > 12 or len(self.exp_table) != (
                    1 << self.score_bits):
                raise ProofV3Error(
                    "inline exp tables are capped at 12 score bits")
        elif self.score_bits != 16:
            raise ProofV3Error(
                "fixed-table statements must use 16 score bits")

    # -- statement / claims / roots / section reconstruction -------------

    def statement(self):
        from verallm.proof_v3.goldilocks_succinct_attention import (
            GoldilocksSuccinctAttentionStatementV3,
        )
        from verallm.proof_v3.scored_attention_reference import (
            fixed_exp_table_v3,
        )

        table = self.exp_table if self.exp_table else fixed_exp_table_v3()
        return GoldilocksSuccinctAttentionStatementV3(
            validator_binding_digest=self.validator_binding_digest,
            head_count=self.head_count, token_count=self.token_count,
            head_dim=self.head_dim, qk_bits=self.qk_bits,
            v_bits=self.v_bits, shift=self.shift,
            exp_table=table, score_bits=self.score_bits,
            scale_bits=self.scale_bits, limb_bits=self.limb_bits,
            key_count=self.key_count,
            query_positions=self.query_positions,
            scored=self.scored,
            m_nums=self.m_nums if self.scored else None,
            m_e=self.m_e, capture_kv=self.capture_kv)

    def claims(self):
        from verallm.proof_v3.recompute_audit import AttentionChunkClaimsV3

        shape = (self.chunk_count, self.head_count, self.token_count)
        totals = np.frombuffer(
            self.chunk_totals, dtype="<i8").reshape(shape).copy()
        partial = np.frombuffer(
            self.partial_out, dtype="<i8").reshape(
                shape + (self.head_dim,)).copy()
        return AttentionChunkClaimsV3(
            chunk_size=self.claims_chunk_size, key_count=self.key_count,
            chunk_totals=totals, partial_out=partial)

    def roots(self):
        from verallm.proof_v3.economic_attention_section import (
            EconomicAttentionLayerRootsV3,
        )

        return EconomicAttentionLayerRootsV3(
            q_root=self.q_root, k_root=self.k_root, v_root=self.v_root,
            o_root=self.o_root, claims_digest=self.claims_digest,
            heads=self.head_count, q_rows=self.token_count,
            key_count=self.key_count, dim=self.head_dim,
            chunk_size=self.claims_chunk_size, kv_heads=self.kv_heads)

    def section(self):
        from verallm.proof_v3.economic_attention_section import (
            EconomicAttentionSectionV3,
        )

        return EconomicAttentionSectionV3(
            layer=self.layer, rows=self.rows,
            chunk_indices=self.chunk_indices,
            q_opening=self.q_opening.to_reference(),
            o_opening=self.o_opening.to_reference(),
            k_openings=tuple(
                (ci, op.to_reference()) for ci, op in self.k_openings),
            v_openings=tuple(
                (ci, op.to_reference()) for ci, op in self.v_openings),
            claims=self.claims(), statement=self.statement())

    # -- codec ------------------------------------------------------------

    def canonical_bytes(self) -> bytes:
        writer = _Writer()
        writer.pack("<IHH", self.layer, len(self.rows),
                    len(self.chunk_indices))
        writer.raw(struct.pack(f"<{len(self.rows)}I", *self.rows))
        writer.raw(struct.pack(
            f"<{len(self.chunk_indices)}I", *self.chunk_indices))
        writer.raw(self.validator_binding_digest)
        writer.pack(
            "<HHHBBBBBBQHBQ",
            self.head_count, self.token_count, self.head_dim,
            self.qk_bits, self.v_bits, self.shift, self.score_bits,
            self.scale_bits, self.limb_bits, self.key_count,
            self.kv_heads, self.scored, self.m_e)
        writer.pack("<B", self.capture_kv)
        writer.raw(struct.pack(
            f"<{self.token_count}Q", *self.query_positions))
        writer.pack("<I", len(self.m_nums))
        if self.m_nums:
            writer.raw(struct.pack(f"<{len(self.m_nums)}Q", *self.m_nums))
        writer.pack("<I", len(self.exp_table))
        if self.exp_table:
            writer.raw(struct.pack(
                f"<{len(self.exp_table)}Q", *self.exp_table))
        writer.pack("<IQ", self.claims_chunk_size, self.chunk_count)
        writer.vbytes(self.chunk_totals, "chunk totals",
                      MAX_ATTENTION_WIRE_BYTES)
        writer.vbytes(self.partial_out, "partial out",
                      MAX_ATTENTION_WIRE_BYTES)
        for digest in (self.q_root, self.k_root, self.v_root, self.o_root,
                       self.claims_digest, self.root_commitment):
            writer.raw(digest)
        self.q_opening.encode(writer)
        self.o_opening.encode(writer)
        writer.pack("<II", len(self.k_openings), len(self.v_openings))
        for ci, opening in self.k_openings:
            writer.pack("<I", ci)
            opening.encode(writer)
        for ci, opening in self.v_openings:
            writer.pack("<I", ci)
            opening.encode(writer)
        writer.pack("<B", 1 if self.bridge_opening is not None else 0)
        if self.bridge_opening is not None:
            self.bridge_opening.encode(writer)
        encoded = writer.finish()
        if len(encoded) > MAX_ATTENTION_WIRE_BYTES:
            raise ProofV3Error("attention wire exceeds the byte budget")
        return encoded

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "AttentionAuditWireV3":
        if len(encoded) > MAX_ATTENTION_WIRE_BYTES:
            raise ProofV3Error("attention wire exceeds the byte budget")
        reader = _Reader(encoded, "attention audit wire")
        layer, row_count, chunk_count_sel = reader.unpack("<IHH")
        if not 0 < row_count <= _MAX_POOL:
            raise ProofV3Error("attention wire row count out of range")
        if chunk_count_sel > _MAX_CHUNK_OPENINGS:
            raise ProofV3Error("attention wire sampled chunk count")
        rows = reader.unpack(f"<{row_count}I")
        chunk_indices = reader.unpack(f"<{chunk_count_sel}I")
        binding = reader.read(32)
        (head_count, token_count, head_dim, qk_bits, v_bits, shift,
         score_bits, scale_bits, limb_bits, key_count, kv_heads,
         scored, m_e) = reader.unpack("<HHHBBBBBBQHBQ")
        (capture_kv,) = reader.unpack("<B")
        if not 0 < token_count <= _MAX_POOL:
            raise ProofV3Error("attention wire pool size out of range")
        if not 0 < head_count <= _MAX_HEADS:
            raise ProofV3Error("attention wire head count out of range")
        query_positions = reader.unpack(f"<{token_count}Q")
        (m_count,) = reader.unpack("<I")
        if m_count > _MAX_HEADS:
            raise ProofV3Error("attention wire slope count out of range")
        m_nums = reader.unpack(f"<{m_count}Q") if m_count else ()
        (table_len,) = reader.unpack("<I")
        if table_len > (1 << 12):
            raise ProofV3Error("inline exp table exceeds the wire cap")
        exp_table = (
            reader.unpack(f"<{table_len}Q") if table_len else ())
        claims_chunk_size, chunk_count = reader.unpack("<IQ")
        chunk_totals = reader.vbytes(
            "chunk totals", MAX_ATTENTION_WIRE_BYTES)
        partial_out = reader.vbytes(
            "partial out", MAX_ATTENTION_WIRE_BYTES)
        q_root = reader.read(32)
        k_root = reader.read(32)
        v_root = reader.read(32)
        o_root = reader.read(32)
        claims_digest = reader.read(32)
        root_commitment = reader.read(32)
        q_opening = _LegacyOpeningV3.decode(reader)
        o_opening = _LegacyOpeningV3.decode(reader)
        k_count, v_count = reader.unpack("<II")
        if k_count > _MAX_CHUNK_OPENINGS or v_count > _MAX_CHUNK_OPENINGS:
            raise ProofV3Error("attention wire chunk opening count")
        k_openings = []
        for _ in range(k_count):
            (ci,) = reader.unpack("<I")
            k_openings.append((ci, _LegacyOpeningV3.decode(reader)))
        v_openings = []
        for _ in range(v_count):
            (ci,) = reader.unpack("<I")
            v_openings.append((ci, _LegacyOpeningV3.decode(reader)))
        (has_bridge,) = reader.unpack("<B")
        bridge = EconomicMerkleOpeningV3.decode(reader) if has_bridge else None
        reader.finish()
        return cls(
            layer=layer, rows=tuple(int(r) for r in rows),
            chunk_indices=tuple(int(c) for c in chunk_indices),
            validator_binding_digest=binding, head_count=head_count,
            token_count=token_count, head_dim=head_dim, qk_bits=qk_bits,
            v_bits=v_bits, shift=shift, score_bits=score_bits,
            scale_bits=scale_bits, limb_bits=limb_bits,
            key_count=key_count,
            query_positions=tuple(int(p) for p in query_positions),
            scored=scored, m_nums=tuple(int(m) for m in m_nums), m_e=m_e,
            capture_kv=capture_kv,
            exp_table=tuple(int(v) for v in exp_table),
            claims_chunk_size=claims_chunk_size,
            chunk_count=chunk_count, chunk_totals=chunk_totals,
            partial_out=partial_out, q_root=q_root, k_root=k_root,
            v_root=v_root, o_root=o_root, claims_digest=claims_digest,
            kv_heads=kv_heads, root_commitment=root_commitment,
            q_opening=q_opening, o_opening=o_opening,
            k_openings=tuple(k_openings), v_openings=tuple(v_openings),
            bridge_opening=bridge)


def build_attention_audit_wire_v3(*, capture_binding: bytes, section, roots,
                                  root_commitment: bytes,
                                  bridge_opening=None) -> AttentionAuditWireV3:
    """Miner-side: pack an ``answer_attention_section`` result for the wire."""

    from verallm.proof_v3.economic_attention_section import (
        _leaf_binding,
        attention_tensor_binding_v3,
    )
    from verallm.proof_v3.scored_attention_reference import (
        fixed_exp_table_v3,
    )

    st = section.statement
    if getattr(st, "chunk_base", None) is not None:
        raise ProofV3Error("chunked statements never ride the audit wire")
    if st.score_bits == 16:
        if tuple(st.exp_table) != fixed_exp_table_v3():
            raise ProofV3Error(
                "16-bit statements must pin the fixed exp table")
        wire_table: tuple[int, ...] = ()
    else:
        wire_table = tuple(int(v) for v in st.exp_table)
    claims = section.claims
    totals = np.asarray(claims.chunk_totals, dtype=np.int64)
    partial = np.asarray(claims.partial_out, dtype=np.int64)
    heads, pool, dim = st.head_count, st.token_count, st.head_dim
    kv = roots.kv_heads or heads

    def _bind(tag: str) -> bytes:
        return _leaf_binding(attention_tensor_binding_v3(
            capture_binding=capture_binding, layer=section.layer, tag=tag))

    q_leaves = _pow2(heads * pool * dim)
    kv_leaves = _pow2(kv * st.key_count * dim)
    return AttentionAuditWireV3(
        layer=section.layer, rows=tuple(section.rows),
        chunk_indices=tuple(section.chunk_indices),
        validator_binding_digest=st.validator_binding_digest,
        head_count=heads, token_count=pool, head_dim=dim,
        qk_bits=st.qk_bits, v_bits=st.v_bits, shift=st.shift,
        score_bits=st.score_bits, scale_bits=st.scale_bits,
        limb_bits=st.limb_bits, key_count=st.key_count,
        query_positions=tuple(st.query_positions),
        scored=int(getattr(st, "scored", 0)),
        m_nums=tuple(st.m_nums or ()),
        m_e=int(getattr(st, "m_e", 0)),
        capture_kv=int(getattr(st, "capture_kv", 0)),
        exp_table=wire_table,
        claims_chunk_size=claims.chunk_size,
        chunk_count=int(totals.shape[0]),
        chunk_totals=totals.astype("<i8").tobytes(),
        partial_out=partial.astype("<i8").tobytes(),
        q_root=roots.q_root, k_root=roots.k_root, v_root=roots.v_root,
        o_root=roots.o_root, claims_digest=roots.claims_digest,
        kv_heads=roots.kv_heads, root_commitment=root_commitment,
        q_opening=_capture_to_wire(
            section.q_opening, leaf_count=q_leaves, binding=_bind("q")),
        o_opening=_capture_to_wire(
            section.o_opening, leaf_count=q_leaves, binding=_bind("o")),
        k_openings=tuple(
            (ci, _capture_to_wire(
                op, leaf_count=kv_leaves, binding=_bind("k")))
            for ci, op in section.k_openings),
        v_openings=tuple(
            (ci, _capture_to_wire(
                op, leaf_count=kv_leaves, binding=_bind("v")))
            for ci, op in section.v_openings),
        bridge_opening=bridge_opening)


def apply_attention_audit_wire_v3(artifacts, wire: AttentionAuditWireV3) -> None:
    """Validator-side: install a decoded audit bundle on the artifacts.

    TRUST BOUNDARY: ``attention_root_commitment`` is VALIDATOR-OWNED --
    it must be recovered from the authenticated pre-nonce envelope
    (capture_chain_digest -> execution_root), never taken from the
    miner's wire.  This function therefore refuses to run unless the
    validator already installed its own expected commitment, and the
    wire's echoed copy must MATCH it (a mismatched reply fails closed
    here, before any section verification spends cycles)."""

    expected = getattr(artifacts, "attention_root_commitment", None)
    if expected is None:
        raise ProofV3Error(
            "attention_root_commitment must be validator-derived from the "
            "pre-nonce envelope BEFORE applying a miner audit wire")
    if wire.root_commitment != expected:
        raise ProofV3Error(
            "miner audit wire echoes a different attention root commitment "
            "than the validator derived pre-nonce (stale or cross-request "
            "reply)")
    artifacts.attention_section = wire.section()
    artifacts.attention_roots = wire.roots()
    artifacts.attention_bridge_opening = wire.bridge_opening


def fetch_attention_audit_wire_v3(*, endpoint: str, request_id: str,
                                  validator_nonce: bytes,
                                  row_samples: int = 32,
                                  chunk_samples: int = 3,
                                  timeout: float = 60.0,
                                  verify_tls: bool = False,
                                  ) -> AttentionAuditWireV3:
    """Validator-side transport: POST the miner's attention-audit route
    and decode the canonical wire (decode enforces every bound)."""

    import httpx

    url = endpoint.rstrip("/") + "/proof_v3/attention_audit"
    response = httpx.post(
        url,
        json={
            "request_id": request_id,
            "validator_nonce": validator_nonce.hex(),
            "row_samples": int(row_samples),
            "chunk_samples": int(chunk_samples),
        },
        timeout=timeout, verify=verify_tls)
    if response.status_code != 200:
        raise ProofV3Error(
            f"attention audit fetch failed ({response.status_code}): "
            + response.text[:200])
    payload = response.json()
    try:
        encoded = bytes.fromhex(payload["wire"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProofV3Error("attention audit reply is malformed") from exc
    return AttentionAuditWireV3.from_canonical_bytes(encoded)
