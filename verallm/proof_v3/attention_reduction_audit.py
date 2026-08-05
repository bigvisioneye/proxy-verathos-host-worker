"""Bounded probabilistic authenticated reduction-tree attention audit.

The PRODUCTION hard-canary attention path (design doc:
proof_v3_attention_reduction_audit_design.md).  The generic succinct
PCS stays as the reference tier; nothing here uses PCS/FRI/LogUp.

Pre-nonce, the miner commits -- per audited (layer, head, query row) --
a deterministic binary reduction tree over bounded key chunks: each
leaf carries the chunk's fixed-point (peak, mass, weighted-V numerator)
summary plus hashes binding the captured Q/K/V material; each node
carries the signed-ABI merge of its children.  Post-nonce the validator
derives an exact sampling plan (uniform + authenticated mass-weighted),
the miner reveals only the sampled leaves/paths + native-GQA K/V
chunks + o_proj rows, and the validator recomputes everything it sees
exactly, checks the roots against the envelope-derived commitment, and
binds the root output to the captured o_proj input via the signed
corridor.  Work and wire scale with the sampled chunks and
log(context), never with every chunk.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.scored_attention_reference import (
    SCALE_BITS,
    ScoredHeadParamsV3,
    fixed_exp_table_v3,
)

__all__ = [
    "REDUCTION_AUDIT_ABI_V3",
    "ReductionGeometryV3",
    "ReductionSummaryV3",
    "ReductionTreeV3",
    "ReductionPlanV3",
    "ReductionLeafRevealV3",
    "ReductionPathRevealV3",
    "ReductionAuditRevealV3",
    "attention_reduction_layer_root_v3",
    "attention_reduction_request_root_v3",
    "build_reduction_tree_v3",
    "build_reduction_tree_from_summaries_v3",
    "candidate_pair_order_v3",
    "compute_leaf_summary_v3",
    "derive_reduction_plan_v3",
    "derive_reduction_bundle_v3",
    "effective_child_masses_v3",
    "merge_summaries_v3",
    "build_reduction_reveal_v3",
    "verify_reduction_reveal_v3",
    "reduction_bridge_check_v3",
    "reduction_root_output_v3",
    "reduction_escape_probability_v3",
]

REDUCTION_AUDIT_ABI_V3: Final = "verathos.attention_reduction_audit.v1"
_LEAF_DOM: Final = b"VERATHOS/PROOF_V3/ATTN_REDUCE/LEAF/V1"
_NODE_DOM: Final = b"VERATHOS/PROOF_V3/ATTN_REDUCE/NODE/V1"
_GEOM_DOM: Final = b"VERATHOS/PROOF_V3/ATTN_REDUCE/GEOM/V1"
_PLAN_DOM: Final = b"VERATHOS/PROOF_V3/ATTN_REDUCE/PLAN/V1"
_MAT_DOM: Final = b"VERATHOS/PROOF_V3/ATTN_REDUCE/MAT/V1"
# membership hierarchy: request root -> layer roots -> (head,row) pairs.
# EVERY candidate pair is committed pre-nonce; post-nonce the miner
# opens Merkle membership for exactly the validator-selected ones.
_PAIR_DOM: Final = b"VERATHOS/PROOF_V3/ATTN_REDUCE/PAIR/V1"
_MEMB_LEAF_DOM: Final = b"VERATHOS/PROOF_V3/ATTN_REDUCE/MEMB/LEAF/V1"
_MEMB_NODE_DOM: Final = b"VERATHOS/PROOF_V3/ATTN_REDUCE/MEMB/NODE/V1"
_MEMB_EMPTY: Final = hashlib.sha256(
    b"VERATHOS/PROOF_V3/ATTN_REDUCE/MEMB/EMPTY/V1").digest()
_LAYER_LEAF_DOM: Final = b"VERATHOS/PROOF_V3/ATTN_REDUCE/LAYERLEAF/V1"

_TABLE_ONE: Final = 1 << 22          # T[0]
_SCORE_MIN: Final = -(1 << 62)       # identity peak (no visible key)
_CLAMP_MAX: Final = (1 << 16) - 1    # fixed table index range


def _i64(value: int) -> bytes:
    return int(value).to_bytes(8, "little", signed=True)


def _u32(value: int) -> bytes:
    return int(value).to_bytes(4, "little")


def _hash_ints(domain: bytes, values) -> bytes:
    hasher = hashlib.sha256(domain)
    for value in values:
        hasher.update(_i64(value))
    return hasher.digest()


@dataclass(frozen=True, slots=True)
class ReductionGeometryV3:
    """Everything that pins ONE (layer, head, query-row) tree's shape.

    ``kv_head`` is the NATIVE kv head this query head reads
    (``head // (n_heads // n_kv)``) -- the k/v binding hashes are over
    that native material, shared across query heads."""

    layer: int
    head: int
    kv_head: int
    row_position: int
    key_count: int
    chunk_len: int
    head_dim: int
    profile_digest: bytes

    def digest(self) -> bytes:
        return hashlib.sha256(
            _GEOM_DOM + _u32(self.layer) + _u32(self.head)
            + _u32(self.kv_head) + struct.pack(
                "<QQII", self.row_position, self.key_count,
                self.chunk_len, self.head_dim)
            + self.profile_digest).digest()

    @property
    def chunk_count(self) -> int:
        return (self.key_count + self.chunk_len - 1) // self.chunk_len

    @property
    def slot_count(self) -> int:
        count = max(1, self.chunk_count)
        slot = 1
        while slot < count:
            slot <<= 1
        return slot


@dataclass(frozen=True, slots=True)
class ReductionSummaryV3:
    """(peak, mass, weighted numerator) in signed fixed-point units."""

    peak: int
    mass: int
    out: tuple[int, ...]         # SCALE_BITS-scaled bounded softmax output

    def encode(self) -> bytes:
        return _i64(self.peak) + _i64(self.mass) + b"".join(
            _i64(v) for v in self.out)


def _identity_summary(head_dim: int) -> ReductionSummaryV3:
    return ReductionSummaryV3(
        peak=_SCORE_MIN, mass=0, out=(0,) * head_dim)


def _rescale(value: int, weight: int) -> int:
    # floor division of a SIGNED value by 2^22 after the table weight --
    # python floor semantics are the ABI (exact on both sides)
    return (value * weight) >> 22


def _div_round(numer: int, denom: int) -> int:
    # round-half-away-from-zero integer division; deterministic pure-int
    # ABI (identical on prover and verifier).  denom > 0.
    if numer >= 0:
        return (numer + (denom >> 1)) // denom
    return -((-numer + (denom >> 1)) // denom)


def merge_summaries_v3(a: ReductionSummaryV3,
                       b: ReductionSummaryV3) -> ReductionSummaryV3:
    """Deterministic flash-attention merge (floor rounding, exact ints).

    The node carries the peak-normalized OUTPUT (a mass-weighted average
    of the children's outputs), NOT a raw numerator.  This makes a leaf's
    influence on the root output exactly its effective mass share:
    ``out`` moves by at most (child mass share) x (output range), so the
    mass-weighted sampler's (1-g)^jm bound is real.  A raw-numerator ABI
    let a tiny-mass leaf inject unbounded output influence (review
    finding), which this closes."""

    if a.mass == 0 and a.peak == _SCORE_MIN:
        return b
    if b.mass == 0 and b.peak == _SCORE_MIN:
        return a
    table = fixed_exp_table_v3()
    peak = a.peak if a.peak >= b.peak else b.peak
    m_a = _rescale(a.mass, table[min(_CLAMP_MAX, peak - a.peak)])
    m_b = _rescale(b.mass, table[min(_CLAMP_MAX, peak - b.peak)])
    mass = m_a + m_b
    if mass <= 0:
        # both children rescaled to zero contribution: identity output
        return ReductionSummaryV3(
            peak=peak, mass=mass, out=(0,) * len(a.out))
    out = tuple(
        _div_round(m_a * oa + m_b * ob, mass)
        for oa, ob in zip(a.out, b.out, strict=True))
    return ReductionSummaryV3(peak=peak, mass=mass, out=out)


def effective_child_masses_v3(parent_peak: int, left: ReductionSummaryV3,
                              right: ReductionSummaryV3) -> tuple[int, int]:
    """The children's contributions to the PARENT mass -- exactly the
    merge's floor-rescaled terms.  Mass-weighted walks MUST draw against
    these (raw child masses at different peaks are not comparable, and a
    raw-mass walk would not sample proportionally to root influence)."""

    table = fixed_exp_table_v3()

    def _effective(child: ReductionSummaryV3) -> int:
        if child.mass <= 0 or child.peak == _SCORE_MIN:
            return 0
        return _rescale(
            child.mass, table[min(_CLAMP_MAX, parent_peak - child.peak)])

    return _effective(left), _effective(right)


def _score(q13_row, k8_key, params: ScoredHeadParamsV3) -> int:
    product = 0
    for qv, kv in zip(q13_row, k8_key, strict=True):
        product += int(qv) * int(kv)
    return (product * params.m_num + (1 << (params.m_e - 1))) >> params.m_e


def compute_leaf_summary_v3(*, geometry: ReductionGeometryV3,
                            params: ScoredHeadParamsV3,
                            chunk_index: int, q13_row, k8_chunk,
                            v8_chunk) -> ReductionSummaryV3:
    """Exact leaf summary over one key chunk (validator recomputes this
    byte-for-byte from the revealed material)."""

    table = fixed_exp_table_v3()
    base = chunk_index * geometry.chunk_len
    peak = _SCORE_MIN
    scores = []
    for offset, key_row in enumerate(k8_chunk):
        key = base + offset
        if key > geometry.row_position:
            scores.append(None)
            continue
        s = _score(q13_row, key_row, params)
        scores.append(s)
        if s > peak:
            peak = s
    if peak == _SCORE_MIN:
        return _identity_summary(geometry.head_dim)
    mass = 0
    num = [0] * geometry.head_dim
    for offset, s in enumerate(scores):
        if s is None:
            continue
        weight = table[min(_CLAMP_MAX, peak - s)]
        mass += weight
        v_row = v8_chunk[offset]
        for d in range(geometry.head_dim):
            num[d] += weight * int(v_row[d])
    # the leaf's peak-normalized OUTPUT in SCALE_BITS-scaled v8 units
    # (bounded to +-127*2^SCALE_BITS since it is a convex combination of
    # int8 V); the raw numerator never leaves this function
    out = tuple(
        _div_round(n << SCALE_BITS, mass) for n in num)
    return ReductionSummaryV3(peak=peak, mass=mass, out=out)


def material_hash_v3(rows) -> bytes:
    """Binding hash of one q row / k chunk / v chunk (row-major ints).

    Tensor/ndarray inputs take a vectorized path (little-endian signed
    int64 row-major bytes -- byte-identical to the scalar loop)."""

    if hasattr(rows, "detach"):        # torch tensor
        rows = rows.detach().cpu().numpy()
    if hasattr(rows, "astype"):        # numpy ndarray
        import numpy

        flat = numpy.ascontiguousarray(rows).astype("<i8", copy=False)
        return hashlib.sha256(_MAT_DOM + flat.tobytes()).digest()
    hasher = hashlib.sha256(_MAT_DOM)
    for row in rows:
        if isinstance(row, int):
            hasher.update(_i64(row))
            continue
        for value in row:
            hasher.update(_i64(value))
    return hasher.digest()


def leaf_hash_v3(*, geometry: ReductionGeometryV3, chunk_index: int,
                 chunk_len: int, summary: ReductionSummaryV3,
                 q_hash: bytes, k_hash: bytes, v_hash: bytes) -> bytes:
    return hashlib.sha256(
        _LEAF_DOM + geometry.digest() + _u32(chunk_index)
        + _u32(chunk_len) + summary.encode() + q_hash + k_hash
        + v_hash).digest()


def node_hash_v3(*, geometry: ReductionGeometryV3, level: int, index: int,
                 left: bytes, right: bytes,
                 summary: ReductionSummaryV3) -> bytes:
    return hashlib.sha256(
        _NODE_DOM + geometry.digest() + _u32(level) + _u32(index)
        + left + right + summary.encode()).digest()


@dataclass(frozen=True, slots=True)
class ReductionTreeV3:
    """One (layer, head, row) authenticated reduction tree.

    ``levels[0]`` are the leaf (hash, summary) pairs over the padded
    slot count; ``levels[-1]`` is the single root."""

    geometry: ReductionGeometryV3
    levels: tuple[tuple[tuple[bytes, ReductionSummaryV3], ...], ...]

    @property
    def root_hash(self) -> bytes:
        return self.levels[-1][0][0]

    @property
    def root_summary(self) -> ReductionSummaryV3:
        return self.levels[-1][0][1]


def build_reduction_tree_v3(*, geometry: ReductionGeometryV3,
                            params: ScoredHeadParamsV3, q13_row,
                            k8_chunks, v8_chunks) -> ReductionTreeV3:
    """Miner-side tree build from the captured (already-quantized)
    material -- pure-python reference (the GPU emitter precomputes the
    summaries and calls ``build_reduction_tree_from_summaries_v3``)."""

    if len(k8_chunks) != geometry.chunk_count:
        raise ProofV3Error("reduction chunks do not match the geometry")
    summaries = [
        compute_leaf_summary_v3(
            geometry=geometry, params=params, chunk_index=slot,
            q13_row=q13_row, k8_chunk=k8_chunks[slot],
            v8_chunk=v8_chunks[slot])
        for slot in range(geometry.chunk_count)
    ]
    return build_reduction_tree_from_summaries_v3(
        geometry=geometry, q13_row=q13_row, k8_chunks=k8_chunks,
        v8_chunks=v8_chunks, summaries=summaries)


def build_reduction_tree_from_summaries_v3(
        *, geometry: ReductionGeometryV3, q13_row, k8_chunks, v8_chunks,
        summaries, k_hashes=None, v_hashes=None,
        chunk_lens=None) -> ReductionTreeV3:
    """Tree assembly from precomputed leaf summaries (hashing + merges
    only -- byte-identical whichever emitter produced the summaries).

    ``k_hashes``/``v_hashes``/``chunk_lens`` (all per real chunk) let a
    batch emitter hash each NATIVE kv chunk ONCE and reuse it across
    every (head, row) tree of the group -- the hashes are byte-identical
    to ``material_hash_v3`` of the chunk ints, so the tree is unchanged;
    when provided, ``k8_chunks``/``v8_chunks`` may be None."""

    if k_hashes is not None:
        if (len(k_hashes) != geometry.chunk_count
                or len(v_hashes or ()) != geometry.chunk_count
                or len(chunk_lens or ()) != geometry.chunk_count):
            raise ProofV3Error(
                "precomputed chunk hashes do not match the geometry")
    elif len(k8_chunks) != geometry.chunk_count or len(v8_chunks) != (
            geometry.chunk_count):
        raise ProofV3Error("reduction chunks do not match the geometry")
    if len(summaries) != geometry.chunk_count:
        raise ProofV3Error("reduction chunks do not match the geometry")
    q_hash = material_hash_v3([q13_row])
    leaves = []
    empty = _identity_summary(geometry.head_dim)
    for slot in range(geometry.slot_count):
        if slot < geometry.chunk_count:
            summary = summaries[slot]
            if k_hashes is not None:
                digest = leaf_hash_v3(
                    geometry=geometry, chunk_index=slot,
                    chunk_len=int(chunk_lens[slot]), summary=summary,
                    q_hash=q_hash, k_hash=k_hashes[slot],
                    v_hash=v_hashes[slot])
                leaves.append((digest, summary))
                continue
            k8 = k8_chunks[slot]
            v8 = v8_chunks[slot]
            digest = leaf_hash_v3(
                geometry=geometry, chunk_index=slot, chunk_len=len(k8),
                summary=summary, q_hash=q_hash,
                k_hash=material_hash_v3(k8), v_hash=material_hash_v3(v8))
        else:
            summary = empty
            digest = leaf_hash_v3(
                geometry=geometry, chunk_index=slot, chunk_len=0,
                summary=summary, q_hash=q_hash,
                k_hash=material_hash_v3(()), v_hash=material_hash_v3(()))
        leaves.append((digest, summary))
    levels = [tuple(leaves)]
    level_index = 0
    while len(levels[-1]) > 1:
        level_index += 1
        prev = levels[-1]
        nxt = []
        for index in range(0, len(prev), 2):
            left_hash, left_sum = prev[index]
            right_hash, right_sum = prev[index + 1]
            summary = merge_summaries_v3(left_sum, right_sum)
            nxt.append((node_hash_v3(
                geometry=geometry, level=level_index, index=index // 2,
                left=left_hash, right=right_hash, summary=summary),
                summary))
        levels.append(tuple(nxt))
    return ReductionTreeV3(geometry=geometry, levels=tuple(levels))


# ---------------------------------------------------------------------------
# membership hierarchy (request root -> layer -> (head,row) pair)
# ---------------------------------------------------------------------------


def _memb_leaf(payload: bytes) -> bytes:
    return hashlib.sha256(_MEMB_LEAF_DOM + payload).digest()


def _memb_node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_MEMB_NODE_DOM + left + right).digest()


def _memb_levels(payloads) -> list:
    """All levels of the padded binary membership tree (leaves first)."""

    leaves = [_memb_leaf(p) for p in payloads]
    if not leaves:
        raise ProofV3Error("membership tree needs at least one leaf")
    while len(leaves) & (len(leaves) - 1):
        leaves.append(_MEMB_EMPTY)
    levels = [leaves]
    while len(levels[-1]) > 1:
        prev = levels[-1]
        levels.append([
            _memb_node(prev[i], prev[i + 1])
            for i in range(0, len(prev), 2)])
    return levels


def _memb_path(levels, index: int) -> tuple[bytes, ...]:
    path = []
    for level in levels[:-1]:
        path.append(level[index ^ 1])
        index >>= 1
    return tuple(path)


def _memb_verify(*, root: bytes, payload: bytes, index: int,
                 leaf_count: int, path) -> None:
    padded = 1
    while padded < leaf_count:
        padded <<= 1
    depth = padded.bit_length() - 1
    if len(path) != depth or not 0 <= index < leaf_count:
        raise ProofV3VerificationError(
            "membership opening shape does not match the tree")
    node = _memb_leaf(payload)
    position = index
    for sibling in path:
        if position & 1:
            node = _memb_node(sibling, node)
        else:
            node = _memb_node(node, sibling)
        position >>= 1
    if node != root:
        raise ProofV3VerificationError(
            "membership opening does not fold to the committed root")


def _pair_payload(geometry_digest: bytes, root_hash: bytes,
                  root_summary: "ReductionSummaryV3") -> bytes:
    # bind the root SUMMARY into the pre-nonce commitment: the summary is
    # the surrogate output the corridor consumes, so membership must
    # authenticate it directly (the mass-walk only binds it when the tree
    # has interior nodes -- single-chunk trees have none)
    return (_PAIR_DOM + geometry_digest + root_hash
            + root_summary.encode())


def _layer_payload(layer: int, layer_root: bytes) -> bytes:
    return _LAYER_LEAF_DOM + _u32(layer) + layer_root


def attention_reduction_layer_root_v3(*, pair_payloads) -> bytes:
    """Layer root over EVERY candidate (head,row) pair, canonical order
    (head-major over the committed candidate row pool)."""

    return _memb_levels(pair_payloads)[-1][0]


def attention_reduction_request_root_v3(*, layer_entries) -> bytes:
    """Pre-nonce request attention root over the ordered audited layers:
    ``layer_entries = ((layer, layer_root), ...)``.  THIS is the value
    that rides the authenticated envelope; everything below it is opened
    by Merkle membership post-nonce."""

    return _memb_levels(
        [_layer_payload(layer, root) for layer, root in layer_entries]
    )[-1][0]


# ---------------------------------------------------------------------------
# post-nonce sampling plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReductionPlanV3:
    """Validator-derived exact sampling plan for one audited layer."""

    layer: int
    heads: tuple[int, ...]
    row_positions: tuple[int, ...]
    uniform_chunks: dict
    mass_draws: int

    def key(self):
        return (self.layer, self.heads, self.row_positions,
                tuple(sorted(
                    (pair, tuple(chunks))
                    for pair, chunks in self.uniform_chunks.items())),
                self.mass_draws)

    def canonical_bytes(self) -> bytes:
        """Canonical binary encoding (transcript material -- NEVER repr)."""

        writer = [
            _u32(self.layer), _u32(len(self.heads)),
            b"".join(_u32(h) for h in self.heads),
            _u32(len(self.row_positions)),
            b"".join(
                struct.pack("<Q", p) for p in self.row_positions),
            _u32(self.mass_draws),
        ]
        for (head, position), chunks in sorted(
                self.uniform_chunks.items()):
            writer.append(_u32(head))
            writer.append(struct.pack("<Q", position))
            writer.append(_u32(len(tuple(chunks))))
            writer.append(b"".join(_u32(c) for c in chunks))
        return b"".join(writer)


def _stream(seed: bytes, domain: bytes):
    counter = 0
    while True:
        block = hashlib.sha256(
            seed + domain + counter.to_bytes(8, "little")).digest()
        counter += 1
        for offset in range(0, 32, 8):
            yield int.from_bytes(block[offset:offset + 8], "little")


def _sample_distinct(stream, population: int, count: int):
    chosen: list[int] = []
    seen = set()
    while len(chosen) < min(count, population):
        value = next(stream) % population
        if value not in seen:
            seen.add(value)
            chosen.append(value)
    return tuple(sorted(chosen))


def derive_reduction_plan_v3(*, validator_nonce: bytes,
                             capture_chain_digest: bytes,
                             profile_digest: bytes,
                             discriminative_layers,
                             head_count: int,
                             candidate_rows,
                             chunk_count: int,
                             heads_per_layer: int = 2,
                             row_samples: int = 8,
                             uniform_chunk_samples: int = 3,
                             mass_draws: int = 2) -> ReductionPlanV3:
    """Expand the nonce into the EXACT audit coordinates.  Deterministic
    on both sides; unpredictable pre-nonce; every consumer requires
    exact equality with this derivation (fail-closed)."""

    layers = tuple(sorted(int(x) for x in discriminative_layers))
    if not layers:
        raise ProofV3Error("reduction plan needs a discriminative layer set")
    rows_pool = tuple(int(r) for r in candidate_rows)
    if not rows_pool:
        raise ProofV3Error("reduction plan needs a candidate row pool")
    seed = hashlib.sha256(
        _PLAN_DOM + validator_nonce + capture_chain_digest
        + profile_digest).digest()
    layer = layers[next(_stream(seed, b"layer")) % len(layers)]
    heads = _sample_distinct(
        _stream(seed, b"heads"), head_count,
        min(heads_per_layer, head_count))
    row_indices = _sample_distinct(
        _stream(seed, b"rows"), len(rows_pool),
        min(row_samples, len(rows_pool)))
    row_positions = tuple(rows_pool[i] for i in row_indices)
    uniform: dict = {}
    for head in heads:
        for position in row_positions:
            uniform[(head, position)] = _sample_distinct(
                _stream(seed, b"chunks" + _u32(head)
                        + struct.pack("<Q", position)),
                chunk_count, uniform_chunk_samples)
    return ReductionPlanV3(
        layer=layer, heads=heads, row_positions=row_positions,
        uniform_chunks=uniform, mass_draws=mass_draws)


def derive_reduction_bundle_v3(*, validator_nonce: bytes,
                               capture_chain_digest: bytes,
                               profile_digest: bytes,
                               selected_layers,
                               head_count: int,
                               candidate_rows,
                               chunk_count: int,
                               heads_per_layer: int = 2,
                               row_samples: int = 8,
                               uniform_chunk_samples: int = 3,
                               mass_draws: int = 2
                               ) -> tuple[ReductionPlanV3, ...]:
    """ONE hard canary's BUNDLED reduction audit: one domain-separated
    subaudit per signed-policy-selected layer, all inside a single
    proof response.

    ``selected_layers`` is the canonical output of the signed
    stratified hard-audit selection (derive_hard_audit_selection_v3's
    full-attention layers): it MUST be sorted, distinct and non-empty
    -- fail-closed here, again on the wire, and again at verify.
    Subaudit ``i`` derives its head/row/chunk coordinates through the
    REAL derive_reduction_plan_v3 with the domain-extended nonce
    ``sha256(validator_nonce || b"/SUBAUDIT/" || bytes([i]))``
    restricted to ``selected_layers[i]`` alone, so every selection is
    a deterministic function of the SAME post-commit validator nonce
    with an explicit subaudit-domain index, unpredictable pre-nonce,
    and no two subaudits share a derivation stream.  The epoch escape
    of the bundle is the measured JOINT catch over all subaudits --
    never an exponentiated one-layer rate."""

    layers = tuple(int(x) for x in selected_layers)
    if not layers:
        raise ProofV3Error("reduction bundle needs selected layers")
    if len(layers) > 255:
        raise ProofV3Error("reduction bundle subaudit count is out of bounds")
    if tuple(sorted(layers)) != layers or len(set(layers)) != len(layers):
        raise ProofV3Error(
            "reduction bundle layers must be sorted and distinct")
    plans = []
    for i, layer in enumerate(layers):
        sub_nonce = hashlib.sha256(
            validator_nonce + b"/SUBAUDIT/" + bytes([i])).digest()
        plan = derive_reduction_plan_v3(
            validator_nonce=sub_nonce,
            capture_chain_digest=capture_chain_digest,
            profile_digest=profile_digest,
            discriminative_layers=(layer,),
            head_count=head_count,
            candidate_rows=candidate_rows,
            chunk_count=chunk_count,
            heads_per_layer=heads_per_layer,
            row_samples=row_samples,
            uniform_chunk_samples=uniform_chunk_samples,
            mass_draws=mass_draws)
        if plan.layer != layer:
            raise ProofV3Error("reduction bundle subaudit layer drifted")
        plans.append(plan)
    return tuple(plans)


def _walk_step(seed_stream, parent_peak: int, left: ReductionSummaryV3,
               right: ReductionSummaryV3) -> int:
    """One authenticated mass-weighted descent decision: draw against
    the EFFECTIVE (peak-rescaled) child masses -- identical prover and
    verifier.  Returns 1 to descend right."""

    left_eff, right_eff = effective_child_masses_v3(
        parent_peak, left, right)
    total = left_eff + right_eff
    if total <= 0:
        return 0
    draw = next(seed_stream) % total
    return 0 if draw < left_eff else 1


# ---------------------------------------------------------------------------
# reveal + verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReductionLeafRevealV3:
    """One sampled leaf: the summary + the captured material to recompute
    it, plus sibling (hash, summary) pairs up to the root."""

    head: int
    row_position: int
    chunk_index: int
    chunk_len: int
    summary: ReductionSummaryV3
    siblings: tuple  # ((hash, summary), ...) leaf level upward
    directions: tuple[int, ...]  # 0 = revealed node is LEFT child


@dataclass(frozen=True, slots=True)
class ReductionPathRevealV3:
    """One mass-weighted walk: per-level (left, right) child summaries +
    hashes so the validator can replay the descent decisions."""

    head: int
    row_position: int
    nodes: tuple  # ((left_hash, left_summary, right_hash, right_summary))
    leaf: ReductionLeafRevealV3


@dataclass(frozen=True, slots=True)
class ReductionAuditRevealV3:
    """Everything the miner sends for one audited layer.

    ``layer_root``/``layer_opening`` open the audited layer inside the
    pre-nonce REQUEST root; ``pair_openings[(head,row)]`` open each
    selected pair inside the layer root (every candidate pair was
    committed pre-nonce -- selection membership is what makes the
    post-nonce subset verifiable)."""

    plan_key: tuple
    geometry: dict          # (head, row) -> ReductionGeometryV3
    roots: dict             # (head, row) -> (root_hash, ReductionSummaryV3)
    layer_root: bytes
    layer_opening: tuple[bytes, ...]
    pair_openings: dict     # (head, row) -> tuple[bytes, ...]
    q_rows: dict            # (head, row) -> q13 ints
    kv_chunks: dict         # (kv_head, chunk_index) -> (k8_rows, v8_rows)
    uniform_leaves: tuple[ReductionLeafRevealV3, ...]
    mass_paths: tuple[ReductionPathRevealV3, ...]
    o_rows: dict            # (head, row) -> captured o_proj input slice


def _leaf_reveal(tree: ReductionTreeV3, slot: int) -> tuple:
    siblings = []
    directions = []
    index = slot
    for level in range(len(tree.levels) - 1):
        sibling_index = index ^ 1
        siblings.append(tree.levels[level][sibling_index])
        directions.append(index & 1)
        index >>= 1
    return tuple(siblings), tuple(directions)


def candidate_pair_order_v3(head_count: int, candidate_rows):
    """The canonical committed pair ordering BOTH sides derive: head-major
    over the committed candidate row pool."""

    return tuple(
        (head, int(position))
        for head in range(head_count) for position in candidate_rows)


def build_reduction_reveal_v3(*, plan: ReductionPlanV3, trees: dict,
                              q_rows: dict, kv_chunks_source,
                              o_rows: dict, head_count: int,
                              candidate_rows, audited_layers,
                              layer_roots_by_layer: dict,
                              ) -> ReductionAuditRevealV3:
    """Miner-side reveal assembly.  ``trees[(head, row_position)]`` must
    cover EVERY candidate pair of the audited layer (they were all
    committed pre-nonce); ``layer_roots_by_layer`` carries every audited
    layer's pre-nonce layer root so the request-tree opening can be
    built.  Membership openings prove the selected subset against the
    request root.  K/V chunks are deduplicated across rows and heads."""

    pair_order = candidate_pair_order_v3(head_count, candidate_rows)
    pair_payloads = [
        _pair_payload(
            trees[pair].geometry.digest(), trees[pair].root_hash,
            trees[pair].root_summary)
        for pair in pair_order]
    pair_levels = _memb_levels(pair_payloads)
    layer_root = pair_levels[-1][0]
    layers = tuple(sorted(int(x) for x in audited_layers))
    if layer_roots_by_layer.get(plan.layer) != layer_root:
        raise ProofV3Error(
            "audited layer's trees do not reproduce its pre-nonce root")
    request_levels = _memb_levels([
        _layer_payload(layer, layer_roots_by_layer[layer])
        for layer in layers])
    layer_opening = _memb_path(
        request_levels, layers.index(plan.layer))

    geometry = {}
    roots = {}
    kv_chunks: dict = {}
    uniform_leaves = []
    mass_paths = []
    pair_openings = {}

    def _need_chunk(kv_head: int, chunk_index: int):
        key = (kv_head, chunk_index)
        if key not in kv_chunks:
            kv_chunks[key] = kv_chunks_source(kv_head, chunk_index)

    seed = hashlib.sha256(
        _PLAN_DOM + b"WALK" + plan.canonical_bytes()).digest()
    for head in plan.heads:
        for position in plan.row_positions:
            pair = (head, position)
            tree = trees[pair]
            geometry[pair] = tree.geometry
            roots[pair] = (tree.root_hash, tree.root_summary)
            pair_openings[pair] = _memb_path(
                pair_levels, pair_order.index(pair))
            for chunk_index in plan.uniform_chunks[pair]:
                if chunk_index >= tree.geometry.chunk_count:
                    continue
                siblings, directions = _leaf_reveal(tree, chunk_index)
                _leaf_hash, summary = tree.levels[0][chunk_index]
                uniform_leaves.append(ReductionLeafRevealV3(
                    head=head, row_position=position,
                    chunk_index=chunk_index,
                    chunk_len=min(
                        tree.geometry.chunk_len,
                        tree.geometry.key_count
                        - chunk_index * tree.geometry.chunk_len),
                    summary=summary, siblings=siblings,
                    directions=directions))
                _need_chunk(tree.geometry.kv_head, chunk_index)
            walk_stream = _stream(
                seed, b"walk" + _u32(head)
                + struct.pack("<Q", position))
            for _draw in range(plan.mass_draws):
                nodes = []
                index = 0
                parent_summary = tree.root_summary
                for level in range(len(tree.levels) - 1, 0, -1):
                    left = tree.levels[level - 1][index * 2]
                    right = tree.levels[level - 1][index * 2 + 1]
                    nodes.append((left[0], left[1], right[0], right[1]))
                    go_right = _walk_step(
                        walk_stream, parent_summary.peak,
                        left[1], right[1])
                    index = index * 2 + go_right
                    parent_summary = right[1] if go_right else left[1]
                slot = index
                if slot >= tree.geometry.chunk_count:
                    slot = tree.geometry.chunk_count - 1
                siblings, directions = _leaf_reveal(tree, slot)
                _leaf_hash, summary = tree.levels[0][slot]
                mass_paths.append(ReductionPathRevealV3(
                    head=head, row_position=position, nodes=tuple(nodes),
                    leaf=ReductionLeafRevealV3(
                        head=head, row_position=position,
                        chunk_index=slot,
                        chunk_len=min(
                            tree.geometry.chunk_len,
                            tree.geometry.key_count
                            - slot * tree.geometry.chunk_len),
                        summary=summary, siblings=siblings,
                        directions=directions)))
                _need_chunk(tree.geometry.kv_head, slot)
    return ReductionAuditRevealV3(
        plan_key=plan.key(), geometry=geometry, roots=roots,
        layer_root=layer_root,
        layer_opening=layer_opening,
        pair_openings=pair_openings,
        q_rows={k: tuple(int(v) for v in q) for k, q in q_rows.items()},
        kv_chunks=kv_chunks, uniform_leaves=tuple(uniform_leaves),
        mass_paths=tuple(mass_paths),
        o_rows={k: tuple(int(v) for v in o) for k, o in o_rows.items()})


def _replay_to_root(*, geometry: ReductionGeometryV3,
                    leaf: ReductionLeafRevealV3, q_hash: bytes,
                    k_hash: bytes, v_hash: bytes,
                    recomputed: ReductionSummaryV3) -> tuple:
    """Recompute the leaf hash + fold the sibling path; returns
    (root_hash, root_summary, leaf_hash)."""

    if recomputed != leaf.summary:
        raise ProofV3VerificationError(
            "reduction leaf summary does not match the revealed material")
    leaf_digest = leaf_hash_v3(
        geometry=geometry, chunk_index=leaf.chunk_index,
        chunk_len=leaf.chunk_len, summary=recomputed, q_hash=q_hash,
        k_hash=k_hash, v_hash=v_hash)
    node_hash = leaf_digest
    summary = recomputed
    index = leaf.chunk_index
    for level, ((sib_hash, sib_summary), direction) in enumerate(
            zip(leaf.siblings, leaf.directions, strict=True), start=1):
        if direction == 0:
            left_h, right_h = node_hash, sib_hash
            merged = merge_summaries_v3(summary, sib_summary)
        else:
            left_h, right_h = sib_hash, node_hash
            merged = merge_summaries_v3(sib_summary, summary)
        index >>= 1
        node_hash = node_hash_v3(
            geometry=geometry, level=level, index=index,
            left=left_h, right=right_h, summary=merged)
        summary = merged
    return node_hash, summary, leaf_digest


def verify_reduction_reveal_v3(*, reveal: ReductionAuditRevealV3,
                               plan: ReductionPlanV3,
                               expected_request_root: bytes,
                               audited_layers,
                               head_count: int,
                               candidate_rows,
                               expected_key_count: int,
                               chunk_len: int,
                               n_kv: int,
                               expected_profile_digest: bytes,
                               params_by_head,
                               bridge_check) -> None:
    """Validator-side verification.  Fail-closed everywhere:

    1. the reveal must target EXACTLY the derived plan (no extra,
       missing, duplicate or reordered coordinates anywhere);
    2. EVERY geometry field equals the validator-derived value
       (key_count = the true committed context length, chunk_len /
       head_dim / profile_digest from the signed profile, kv_head =
       head // (head_count // n_kv)) -- geometry is NEVER trusted from
       the miner's self-consistent digest;
    3. the audited layer's root opens by Merkle membership inside the
       PRE-NONCE request root at the validator-derived layer position,
       and every selected (head,row) root+summary opens inside the layer
       root at its validator-derived candidate position;
    4. every revealed leaf recomputes exactly from the revealed
       Q/K/V material (with binding hashes);
    5. every sibling path folds to the committed (head,row) root;
    6. every mass-weighted walk's descent decisions match the committed
       EFFECTIVE (peak-rescaled) child masses, AND the walk's terminal
       leaf is exactly the revealed leaf (identity + hash bound);
    7. ``bridge_check(head, row_position, root_summary, o_row)`` binds
       the root output to the captured o_proj input (signed corridor).

    ``expected_request_root``/``expected_key_count`` come from the
    AUTHENTICATED pre-nonce envelope; ``audited_layers``/``head_count``/
    ``candidate_rows``/``chunk_len``/``n_kv``/``expected_profile_digest``
    come from the signed profile, never from the miner.
    """

    if reveal.plan_key != plan.key():
        raise ProofV3VerificationError(
            "reduction reveal does not target the validator-derived plan")
    if head_count < 1 or n_kv < 1 or head_count % n_kv:
        raise ProofV3VerificationError(
            "invalid head geometry for the reduction audit")
    group = head_count // n_kv
    expected_pairs = tuple(
        (head, position)
        for head in plan.heads for position in plan.row_positions)
    expected_set = tuple(sorted(expected_pairs))
    for name, mapping in (
        ("roots", reveal.roots), ("geometry", reveal.geometry),
        ("q rows", reveal.q_rows), ("o rows", reveal.o_rows),
        ("pair openings", reveal.pair_openings),
    ):
        if tuple(sorted(mapping)) != expected_set:
            raise ProofV3VerificationError(
                f"reduction reveal {name} do not cover exactly the "
                "selected (head, row) set")

    # --- geometry MUST equal the validator-derived shape --------------
    # (the miner-self-consistent digest only proves post-nonce
    # immutability; correctness has to be pinned against owned config)
    for pair in expected_pairs:
        head, position = pair
        geometry = reveal.geometry[pair]
        head_dim = params_by_head[head].head_dim
        expected = ReductionGeometryV3(
            layer=plan.layer, head=head, kv_head=head // group,
            row_position=position, key_count=expected_key_count,
            chunk_len=chunk_len, head_dim=head_dim,
            profile_digest=expected_profile_digest)
        if geometry != expected:
            raise ProofV3VerificationError(
                "reduction geometry does not equal the validator-derived "
                "shape (context length / chunk_len / kv_head / head_dim / "
                "profile all pinned)")

    # --- membership: request root -> layer -> selected pairs ----------
    layers = tuple(sorted(int(x) for x in audited_layers))
    if plan.layer not in layers:
        raise ProofV3VerificationError(
            "plan layer is outside the signed audited layer set")
    _memb_verify(
        root=expected_request_root,
        payload=_layer_payload(plan.layer, reveal.layer_root),
        index=layers.index(plan.layer), leaf_count=len(layers),
        path=reveal.layer_opening)
    pair_order = candidate_pair_order_v3(head_count, candidate_rows)
    for pair in expected_pairs:
        geometry = reveal.geometry[pair]
        if geometry.layer != plan.layer:
            raise ProofV3VerificationError(
                "reduction reveal geometry is cross-layer")
        if (geometry.head, geometry.row_position) != pair:
            raise ProofV3VerificationError(
                "reduction reveal geometry does not match its slot")
        try:
            pair_index = pair_order.index(pair)
        except ValueError:
            raise ProofV3VerificationError(
                "selected pair is outside the committed candidate set")
        _memb_verify(
            root=reveal.layer_root,
            payload=_pair_payload(
                geometry.digest(), reveal.roots[pair][0],
                reveal.roots[pair][1]),
            index=pair_index, leaf_count=len(pair_order),
            path=reveal.pair_openings[pair])

    q_hashes = {
        pair: material_hash_v3([reveal.q_rows[pair]])
        for pair in expected_pairs}
    kv_hashes = {
        key: (material_hash_v3(chunks[0]), material_hash_v3(chunks[1]))
        for key, chunks in reveal.kv_chunks.items()}
    used_kv_chunks: set = set()

    def _check_leaf(leaf: ReductionLeafRevealV3) -> bytes:
        # the leaf's declared (head,row) is its OWN pair; callers that
        # attach a leaf to a specific walk must additionally bind that
        # identity (see the mass-walk terminal check below)
        pair = (leaf.head, leaf.row_position)
        geometry = reveal.geometry[pair]
        # leaf_len is validator-derived from the pinned geometry -- NOT
        # trusted from the leaf (a lying chunk_len would recompute a
        # different set of keys)
        base = leaf.chunk_index * geometry.chunk_len
        expected_len = min(
            geometry.chunk_len, geometry.key_count - base)
        if leaf.chunk_index >= geometry.chunk_count or (
                leaf.chunk_len != expected_len):
            raise ProofV3VerificationError(
                "leaf chunk index/length disagrees with the pinned "
                "geometry")
        chunk_key = (geometry.kv_head, leaf.chunk_index)
        chunks = reveal.kv_chunks.get(chunk_key)
        if chunks is None:
            raise ProofV3VerificationError(
                "reduction reveal is missing a sampled K/V chunk")
        used_kv_chunks.add(chunk_key)
        k8, v8 = chunks
        if len(k8) != leaf.chunk_len or len(v8) != leaf.chunk_len:
            raise ProofV3VerificationError(
                "revealed K/V chunk length disagrees with the leaf")
        recomputed = compute_leaf_summary_v3(
            geometry=geometry, params=params_by_head[leaf.head],
            chunk_index=leaf.chunk_index,
            q13_row=reveal.q_rows[pair], k8_chunk=k8, v8_chunk=v8)
        root_hash, _root_summary, leaf_digest = _replay_to_root(
            geometry=geometry, leaf=leaf, q_hash=q_hashes[pair],
            k_hash=kv_hashes[chunk_key][0],
            v_hash=kv_hashes[chunk_key][1], recomputed=recomputed)
        if root_hash != reveal.roots[pair][0]:
            raise ProofV3VerificationError(
                "reduction leaf path does not fold to the committed root")
        return leaf_digest

    expected_uniform = set()
    for head in plan.heads:
        for position in plan.row_positions:
            geometry = reveal.geometry[(head, position)]
            for chunk_index in plan.uniform_chunks[(head, position)]:
                if chunk_index < geometry.chunk_count:
                    expected_uniform.add((head, position, chunk_index))
    got_uniform = {
        (leaf.head, leaf.row_position, leaf.chunk_index)
        for leaf in reveal.uniform_leaves}
    if len(reveal.uniform_leaves) != len(got_uniform):
        raise ProofV3VerificationError(
            "duplicate uniform leaf reveals")
    if got_uniform != expected_uniform:
        raise ProofV3VerificationError(
            "uniform chunk reveals do not equal the derived plan")
    for leaf in reveal.uniform_leaves:
        _check_leaf(leaf)

    seed = hashlib.sha256(
        _PLAN_DOM + b"WALK" + plan.canonical_bytes()).digest()
    paths_by_pair: dict = {}
    for path in reveal.mass_paths:
        paths_by_pair.setdefault(
            (path.head, path.row_position), []).append(path)
    if not set(paths_by_pair).issubset(set(expected_pairs)):
        raise ProofV3VerificationError(
            "mass-weighted paths for unselected (head, row) pairs")
    for head in plan.heads:
        for position in plan.row_positions:
            pair = (head, position)
            geometry = reveal.geometry[pair]
            paths = paths_by_pair.get(pair, [])
            if len(paths) != plan.mass_draws:
                raise ProofV3VerificationError(
                    "mass-weighted path count does not match the plan")
            walk_stream = _stream(
                seed, b"walk" + _u32(head)
                + struct.pack("<Q", position))
            for path in paths:
                # replay the committed descent: parent hash chain from
                # the root down, checking each decision
                parent_hash = reveal.roots[pair][0]
                parent_summary = reveal.roots[pair][1]
                level = len(path.nodes)
                index = 0
                slot = 0
                for (left_h, left_s, right_h, right_s) in path.nodes:
                    merged = merge_summaries_v3(left_s, right_s)
                    if merged != parent_summary:
                        raise ProofV3VerificationError(
                            "mass path node summary does not merge to "
                            "its parent")
                    if node_hash_v3(
                            geometry=geometry, level=level, index=index,
                            left=left_h, right=right_h,
                            summary=parent_summary) != parent_hash:
                        raise ProofV3VerificationError(
                            "mass path node hash does not match its "
                            "parent")
                    go_right = _walk_step(
                        walk_stream, parent_summary.peak, left_s, right_s)
                    parent_hash = right_h if go_right else left_h
                    parent_summary = right_s if go_right else left_s
                    slot = slot * 2 + go_right
                    index = index * 2 + go_right
                    level -= 1
                expected_slot = slot
                if expected_slot >= geometry.chunk_count:
                    expected_slot = geometry.chunk_count - 1
                # the walk's terminal leaf MUST be the revealed leaf, of
                # THIS pair, at the forced slot, and the descent's
                # terminal hash MUST equal that leaf's recomputed hash
                # (else a compliant leaf from another slot/pair could be
                # substituted for the one the committed masses forced)
                if (path.leaf.head, path.leaf.row_position) != pair:
                    raise ProofV3VerificationError(
                        "mass-walk leaf belongs to a different pair")
                if path.leaf.chunk_index != expected_slot:
                    raise ProofV3VerificationError(
                        "mass-weighted walk revealed the wrong leaf")
                leaf_digest = _check_leaf(path.leaf)
                if leaf_digest != parent_hash:
                    raise ProofV3VerificationError(
                        "mass-walk terminal hash does not equal the "
                        "revealed leaf")

    if set(reveal.kv_chunks) != used_kv_chunks:
        raise ProofV3VerificationError(
            "reduction reveal carries K/V chunks the plan never sampled")

    for pair in expected_pairs:
        head, position = pair
        bridge_check(
            head, position, reveal.roots[pair][1], reveal.o_rows[pair])


def reduction_root_output_v3(summary: ReductionSummaryV3) -> tuple[int, ...]:
    """The root's surrogate attention output: the flash-merged node
    already carries the SCALE_BITS-scaled bounded softmax output, so this
    just returns it.  Same scale as the scored-section surrogate, so the
    SIGNED ScoredBridgeBoundsV3 corridor applies unchanged."""

    if summary.mass <= 0:
        raise ProofV3VerificationError(
            "reduction root has no visible mass to bridge")
    return tuple(int(v) for v in summary.out)


def reduction_bridge_check_v3(*, params: ScoredHeadParamsV3, bounds,
                              summary: ReductionSummaryV3,
                              ox8_row) -> None:
    """Bind ONE (head,row) reduction root to the captured o_proj input
    slice via the signed corridor (verify_output_bridge_v3 semantics)."""

    from verallm.proof_v3.scored_attention_reference import (
        verify_output_bridge_v3,
    )

    verify_output_bridge_v3(
        params=params,
        surrogate_rows=[reduction_root_output_v3(summary)],
        ox8_rows=[tuple(int(v) for v in ox8_row)],
        bounds=bounds)


def reduction_escape_probability_v3(*, tampered_fraction: float,
                                    tampered_mass_share: float,
                                    uniform_samples: int,
                                    mass_draws: int) -> float:
    """Documented escape bound for the signed hard-canary policy:
    (1-f)^j_u * (1-g)^j_m per (head, row).  NOT deterministic
    verifiable inference -- the policy fixes (j_u, j_m) and this is the
    per-pair escape probability the simulator validates empirically."""

    f = min(max(tampered_fraction, 0.0), 1.0)
    g = min(max(tampered_mass_share, 0.0), 1.0)
    return ((1.0 - f) ** uniform_samples) * ((1.0 - g) ** mass_draws)
