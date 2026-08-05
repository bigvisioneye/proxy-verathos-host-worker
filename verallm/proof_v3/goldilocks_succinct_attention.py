"""PRODUCTION-wire succinct attention: ALL heads of one layer, one tile.

The reference attention head proves one head at a time and verifies by
re-opening every table in full.  This module replaces it on the
production wire with a single succinct tile per LAYER:

* every head lives in one head-indexed cube: score cells are (h, t, s),
  broadcast cells are (h, t, s, d) -- one set of sub-arguments amortises
  the verifier across all heads;
* quantization is a pure Euclidean truncation
  ``raw + raw_offset == 2^shift * qs + rem_s`` (the validator-owned
  window covers the whole reachable score range, so there is no clamp
  branch and every verifier table stays <= 2^score_bits);
* ``exp`` is one packed LogUp (``qs + 2^32 * es`` against the packed
  exp table), the causal mask and the softmax row sums are public-factor
  folds, and the softmax division is the softmax-tile Euclidean identity
  ``es_m * SCALE == probs * total + rem`` with ``rem < total`` enforced
  by the complement trick over 16-bit limb range LogUps;
* QK^T and PV are succinct product arguments over the broadcast cube
  whose public factor is the tensor-decomposed eq point; the broadcast
  columns are pinned to their small-cube originals by single eq-fold
  couplings (Schwartz-Zippel at a post-commit point).

Every eq coupling holds per cell because the MLE of the difference
column vanishes at a random post-commit point only if it is identically
zero (w.h.p.).  Padding heads / rows are honest zero-input heads, so
every relation holds on padding and ``sum eq == 1`` absorbs constants.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from functools import wraps
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
    GOLDILOCKS_MULTILINEAR_PCS_QUERY_COUNT_V3,
    MAX_GOLDILOCKS_MULTILINEAR_PCS_QUERY_COUNT_V3,
    pcs_query_count_v3,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (
    GoldilocksSuccinctLogupStatementV3,
    _eq_table,
    prove_goldilocks_succinct_logup_v3,
    verify_goldilocks_succinct_logup_v3,
)
from verallm.proof_v3.goldilocks_succinct_product_argument_reference import (
    GoldilocksSuccinctProductStatementV3,
    prove_goldilocks_succinct_product_v3,
    verify_goldilocks_succinct_product_v3,
)
from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
    SuccinctEqFoldProofV3,
    column_pcs_statement_v3,
    commit_succinct_column_v3,
    derive_tile_eq_point_v3,
    prove_succinct_eq_fold_v3,
    prove_succinct_public_fold_v3,
    verify_succinct_eq_fold_v3,
    verify_succinct_public_fold_v3,
)

GOLDILOCKS_SUCCINCT_ATTENTION_ABI_V3: Final = (
    "goldilocks.succinct_attention.v1"
)
# rational scheme (V2): es@v numerator + public totals, no per-key
# division -- a DISTINCT signed ABI; verifiers accept exactly the
# scheme the manifest signs, never both
GOLDILOCKS_SUCCINCT_ATTENTION_RATIONAL_ABI: Final = (
    "goldilocks.succinct_attention.rational.v2"
)
_TILE_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_SUCCINCT_ATTENTION/V1"
_QPACK: Final = 1 << 32
_LIMB_COUNT: Final = 3  # rem / complement window = 3 * limb_bits bits

_S_TAGS: Final = (
    "raw", "qs", "rem_s", "w_exp", "es", "es_m", "probs", "rem_d", "comp",
    "total_b", "rem_l0", "rem_l1", "rem_l2", "comp_l0", "comp_l1", "comp_l2",
)
_SMALL_TAGS: Final = (
    "q", "q_biased", "k", "k_biased", "v", "v_biased", "out", "total",
)
_B_TAGS: Final = ("q_b", "k_b", "p_b", "v_b")
_ALL_TAGS: Final = _S_TAGS + _SMALL_TAGS + _B_TAGS


def _is_rect(statement) -> bool:
    return statement.key_count is not None


def _is_chunked(statement) -> bool:
    return getattr(statement, "chunk_base", None) is not None


def _signed_public_peaks_tensor_v3(values, *, device=None):
    """Decode canonical Goldilocks peaks before constructing int64 tensors."""

    import torch

    decoded = []
    half = GOLDILOCKS_MODULUS >> 1
    for value in values:
        value = int(value)
        if not 0 <= value < GOLDILOCKS_MODULUS:
            raise ProofV3Error("attention public peak is not canonical")
        decoded.append(
            value - GOLDILOCKS_MODULUS if value > half else value
        )
    return torch.tensor(decoded, dtype=torch.int64, device=device)


_CHUNK_DROPPED: Final = ("total_b", "total", "out")


_CKV_DROPPED: Final = ("k", "k_biased", "v", "v_biased")


def _rational_filter(statement, tags):
    """Drop the division-only columns under the rational scheme."""

    if getattr(statement, "rational", 0):
        return tuple(t for t in tags if t not in _RATIONAL_DROPPED)
    return tuple(tags)


def _all_tags(statement):
    if getattr(statement, "scored", 0) and getattr(
            statement, "capture_kv", 0):
        # capture-KV mode: K/V live in the capture plane (per-layer PCS
        # columns equality-bound to the capture roots); never tile columns
        base_tags = _rational_filter(statement, (
            tag for tag in _S_TAGS_SCORED + _SMALL_TAGS_SCORED
            if tag not in _CKV_DROPPED))
        if _is_chunked(statement):
            return tuple(
                tag for tag in base_tags
                if tag not in _CHUNK_DROPPED and tag != "peak")
        if _is_rect(statement):
            return base_tags
        return base_tags + _B_TAGS
    if getattr(statement, "scored", 0):
        # scored scheme: S-cube columns + small cubes incl. the per-row
        # peak; broadcasts only on square tiles (rect products open the
        # small cubes via point maps); chunked publishes totals/partials
        # as PUBLIC tables (total/total_b/out never committed)
        if _is_chunked(statement):
            # peak is PUBLIC in chunked mode (global normalization)
            return tuple(
                tag for tag in _rational_filter(
                    statement, _S_TAGS_SCORED + _SMALL_TAGS_SCORED)
                if tag not in _CHUNK_DROPPED and tag != "peak")
        if _is_rect(statement):
            return _rational_filter(
                statement, _S_TAGS_SCORED + _SMALL_TAGS_SCORED)
        return _rational_filter(
            statement, _S_TAGS_SCORED + _SMALL_TAGS_SCORED) + _B_TAGS
    # rectangular tiles are broadcast-free: the product arguments open
    # the small q/k/v/probs cubes directly, so no broadcast columns are
    # ever committed.  Chunked tiles additionally publish totals and
    # partial outputs as PUBLIC tables, so total/total_b/out are never
    # committed either (see proof_v3_chunked_softmax_design.md).
    if _is_chunked(statement):
        return tuple(
            tag for tag in _S_TAGS + _SMALL_TAGS
            if tag not in _CHUNK_DROPPED)
    if _is_rect(statement):
        return _S_TAGS + _SMALL_TAGS
    return _ALL_TAGS
# one block tree may not exceed this many variables (2^22 leaves x4
# blowup = the square tile's proven ceiling); larger cubes commit as
# per-column trees
_MAX_GROUP_VARS: Final = 22

# Shared block trees: one commitment per group, one batched opening per
# group.  Pad blocks are all-zero and zero is a valid cell for every
# relation that reads them (range tables contain 0).
_GROUP_PLAN: Final = (
    ("grp_s", ("raw", "qs", "rem_s", "w_exp", "es", "es_m", "probs",
               "rem_d", "comp", "total_b")),
    ("grp_limbs", ("rem_l0", "rem_l1", "rem_l2",
                   "comp_l0", "comp_l1", "comp_l2")),
    ("grp_qkv", ("q", "k", "v", "out")),
    ("grp_biased", ("q_biased", "k_biased", "v_biased")),
    ("grp_bcast", ("q_b", "k_b", "p_b", "v_b")),
)


def _block_bits(member_count: int) -> int:
    return (member_count - 1).bit_length()


_GROUP_PLAN_SCORED: Final = (
    ("grp_s", ("raw", "su", "rem_b", "peak_b", "s_pos", "ovf", "sel",
               "w_exp", "es", "es_m", "probs", "rem_d", "comp", "total_b")),
    ("grp_limbs", ("rb_l0", "ov_l0", "ov_l1", "rem_l0", "rem_l1", "rem_l2",
                   "comp_l0", "comp_l1", "comp_l2")),
    ("solo/rb_l1", ("rb_l1",)),
    ("grp_qkv", ("q", "k", "v", "out")),
    ("grp_biased", ("q_biased", "k_biased", "v_biased")),
    ("grp_bcast", ("q_b", "k_b", "p_b", "v_b")),
)


_GROUP_PLAN_SCORED_CKV: Final = (
    ("grp_s", ("raw", "su", "rem_b", "peak_b", "s_pos", "ovf", "sel",
               "w_exp", "es", "es_m", "probs", "rem_d", "comp", "total_b")),
    ("grp_limbs", ("rb_l0", "ov_l0", "ov_l1", "rem_l0", "rem_l1", "rem_l2",
                   "comp_l0", "comp_l1", "comp_l2")),
    ("solo/rb_l1", ("rb_l1",)),
    ("grp_qkv", ("q", "out")),
    ("grp_biased", ("q_biased",)),
)


def _group_plan(statement):
    """Effective block-tree group plan.

    Square tiles keep the original plan verbatim.  Rectangular tiles
    with distinct query/key pads split any group whose members live on
    differently-sized cubes (block trees need equal member sizes); the
    split subgroups get deterministic ``_<i>`` suffixes.
    """

    plan = []
    dropped = _CHUNK_DROPPED if _is_chunked(statement) else ()
    if getattr(statement, "rational", 0):
        # V2: the division chain's columns do not exist
        dropped = tuple(dropped) + _RATIONAL_DROPPED
    if getattr(statement, "scored", 0) and getattr(
            statement, "capture_kv", 0):
        base_plan = _GROUP_PLAN_SCORED_CKV
    elif getattr(statement, "scored", 0):
        base_plan = _GROUP_PLAN_SCORED
    else:
        base_plan = _GROUP_PLAN
    for group_tag, member_tags in base_plan:
        if group_tag == "grp_bcast" and _is_rect(statement):
            continue
        member_tags = tuple(
            tag for tag in member_tags if tag not in dropped)
        if not member_tags:
            continue
        by_size: dict[int, list[str]] = {}
        for tag in member_tags:
            by_size.setdefault(
                _tag_member_vars(statement, tag), []).append(tag)
        if len(by_size) == 1:
            subgroups = [(group_tag, member_tags)]
        else:
            subgroups = [
                (f"{group_tag}_{index}", tuple(tags_))
                for index, (_size, tags_) in enumerate(
                    sorted(by_size.items(),
                           key=lambda kv: member_tags.index(kv[1][0])))
            ]
        for sub_tag, sub_members in subgroups:
            base = _tag_member_vars(statement, sub_members[0])
            if base + _block_bits(len(sub_members)) <= _MAX_GROUP_VARS:
                plan.append((sub_tag, sub_members))
            elif base >= _MAX_GROUP_VARS:
                # giant cubes (long-context rectangles): per-column trees
                plan.extend(
                    (f"solo/{tag}", (tag,)) for tag in sub_members)
            else:
                # chunk into the largest block trees that fit the cap
                size = 1 << (_MAX_GROUP_VARS - base)
                plan.extend(
                    (f"{sub_tag}/c{index}",
                     tuple(sub_members[start:start + size]))
                    for index, start in enumerate(
                        range(0, len(sub_members), size)))
    return tuple(plan)


def _group_layout(statement):
    """tag -> (group_tag, block_point); group_tag -> variable count."""

    member_map = {}
    group_vars = {}
    for group_tag, member_tags in _group_plan(statement):
        bits = _block_bits(len(member_tags))
        base = _tag_member_vars(statement, member_tags[0])
        group_vars[group_tag] = base + bits
        for index, tag in enumerate(member_tags):
            member_map[tag] = (
                group_tag, tuple((index >> j) & 1 for j in range(bits)))
    return member_map, group_vars


def _pow2_at_least(n: int) -> int:
    return 1 << max(1, (n - 1).bit_length())


@dataclass(frozen=True, slots=True)
class GoldilocksSuccinctAttentionStatementV3:
    """Validator-owned per-layer attention statement.

    ``exp_table[qs]`` is the softmax semantics over quantized scores
    ``qs = (raw + raw_offset) >> shift``; the window
    ``2^(shift + score_bits)`` must cover every reachable raw score
    given the declared q/k bit widths, so quantization is total.
    """

    validator_binding_digest: bytes
    head_count: int
    token_count: int
    head_dim: int
    qk_bits: int
    v_bits: int
    shift: int
    exp_table: tuple[int, ...]
    score_bits: int = 16
    scale_bits: int = 16
    limb_bits: int = 16
    # RECTANGULAR (sampled-row) mode: ``token_count`` query rows sampled
    # out of a ``key_count``-long committed KV sequence; row t attends
    # causally to keys ``s <= query_positions[t]``.  Both None = the
    # square causal tile (wire-identical to the original statement).
    key_count: int | None = None
    query_positions: tuple[int, ...] | None = None
    # KEY-CHUNKED mode (rect only): this tile covers keys
    # [chunk_base, chunk_base + key_count) of a longer committed
    # sequence; positions stay GLOBAL.  ``public_totals`` (h*t_pad,
    # row-major, phase-2 only) carries the GLOBAL softmax totals the
    # division runs against.  See proof_v3_chunked_softmax_design.md.
    chunk_base: int | None = None
    public_totals: tuple[int, ...] | None = None
    # SCORED scheme (scored_attention_reference semantics, the only scheme a
    # qualified hard profile may select -- fb13b99/a8a301c): the exp lookup
    # indexes a committed peak-normalized SCORE (s_pos = min(peak-su, 65535))
    # instead of the raw-product bucket.  ``m_nums[h]/2^m_e`` is head h's
    # exact rational score slope from the SIGNED calibration, renormalized to
    # the common exponent ``m_e`` (renormalization is exact: mantissas shift
    # left).  exp_table MUST be the fixed universal table
    # (fixed_exp_table_v3) with score_bits=16.
    scored: int = 0
    m_nums: tuple[int, ...] | None = None
    m_e: int = 0
    # capture-KV mode: K/V are NOT tile columns; the audit references the
    # capture plane's pre-nonce KV commitments through per-layer PCS
    # commitments equality-bound to the capture roots (design doc
    # proof_v3_capture_kv_attention_design.md)
    capture_kv: int = 0
    # scored CHUNKED mode: the peak normalization is GLOBAL per row, so the
    # per-row peaks are PUBLIC statement data (canonical field ints, h*t_pad
    # row-major) like public_totals; public_sel_count[h*t] in {0,1} marks
    # whether THIS chunk contains the row's peak achiever (the aggregator
    # checks the counts sum to exactly 1 across the row's chunks, mirroring
    # the totals-consistency check).
    public_peaks: tuple[int, ...] | None = None
    public_sel_count: tuple[int, ...] | None = None
    # RATIONAL scheme (SCORED_SCHEME_RATIONAL_V2, scored only): the tile
    # proves the exact integer pair (total = sum es_m, numerator =
    # sum es_m * v) with NO per-key probability rounding -- the probs/
    # rem_d/comp division chain and its limb range LogUps do not exist,
    # the pv product's factor is es_m, and the small ``out`` cube is the
    # NUMERATOR (<= T * 2^29, in-field).  The runtime binding is the
    # OUT-OF-CIRCUIT cross-multiplied bridge
    # (verify_output_bridge_rational_v3) over the public (numerator,
    # total) aggregates.  See project plan: V2 succinct swap.
    rational: int = 0
    # Exact FRI proximity-query budget used by every dynamic PCS column in
    # this tile. The default preserves the shipped 16-query transcript.
    pcs_query_count: int = GOLDILOCKS_MULTILINEAR_PCS_QUERY_COUNT_V3

    def __post_init__(self) -> None:
        if self.rational not in (0, 1):
            raise ProofV3Error(
                "the rational flag must be canonically 0 or 1")
        if self.rational and not self.scored:
            raise ProofV3Error(
                "the rational scheme requires the scored scheme")
        if self.rational and self.key_count is None:
            # production selected-row tiles are rectangular; the square
            # broadcast machinery (p_b over probs) never carries V2
            raise ProofV3Error(
                "the rational scheme requires a rectangular tile")
        if (
            not isinstance(self.pcs_query_count, int)
            or isinstance(self.pcs_query_count, bool)
            or not 1 <= self.pcs_query_count
            <= MAX_GOLDILOCKS_MULTILINEAR_PCS_QUERY_COUNT_V3
        ):
            raise ProofV3Error(
                "attention PCS query count is outside the protocol cap")
        if self.scored:
            # production profiles pin score_bits=16 (the fixed universal
            # table); smaller tables are allowed for CPU-cap tile tests
            if not 8 <= self.score_bits <= 16:
                raise ProofV3Error("scored score_bits out of range")
            if self.chunk_base is not None:
                if (
                    not isinstance(self.public_peaks, tuple)
                    or not isinstance(self.public_sel_count, tuple)
                    or not all(
                        isinstance(v, int) and 0 <= v < GOLDILOCKS_MODULUS
                        for v in self.public_peaks)
                    or not all(v in (0, 1) for v in self.public_sel_count)
                    or len(self.public_peaks) != len(self.public_sel_count)
                ):
                    raise ProofV3Error(
                        "scored chunked statements need public peaks and "
                        "selector counts")
            if (
                not isinstance(self.m_nums, tuple)
                or len(self.m_nums) != self.head_count
                or not all(
                    isinstance(m, int) and 1 <= m < 1 << 34
                    for m in self.m_nums)
            ):
                raise ProofV3Error("scored slope mantissas are malformed")
            if not 8 <= self.m_e <= 40:
                raise ProofV3Error("scored slope exponent is out of range")
        if not (
            isinstance(self.validator_binding_digest, bytes)
            and len(self.validator_binding_digest) == 32
            and any(self.validator_binding_digest)
        ):
            raise ProofV3Error("attention binding digest is malformed")
        for name in ("head_count", "token_count", "head_dim", "qk_bits",
                     "v_bits", "shift", "score_bits", "scale_bits",
                     "limb_bits"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ProofV3Error(f"attention {name} must be an integer")
        if self.head_count < 1 or self.head_count > 128:
            raise ProofV3Error("attention head count is out of range")
        if self.token_count < 1 or self.token_count > 1024:
            raise ProofV3Error("attention token count is out of range")
        if self.head_dim < 2 or self.head_dim & (self.head_dim - 1):
            raise ProofV3Error("attention head dim must be a power of two")
        if not 2 <= self.qk_bits <= 16 or not 2 <= self.v_bits <= 16:
            raise ProofV3Error("attention q/k/v bit widths are out of range")
        if not 1 <= self.shift <= 16:
            raise ProofV3Error("attention shift is out of range")
        if not 4 <= self.score_bits <= 16 or not 4 <= self.scale_bits <= 16:
            raise ProofV3Error("attention score/scale bits are out of range")
        if not 2 <= self.limb_bits <= 16:
            raise ProofV3Error("attention limb bits are out of range")
        window = 1 << (self.shift + self.score_bits)
        max_raw = self.head_dim << (2 * self.qk_bits - 2)
        if 2 * max_raw > window and not self.scored:
            # V1 product-domain bucketing only: the scored scheme indexes
            # the table by the peak-normalized s_pos, never by raw
            raise ProofV3Error(
                "attention quant window does not cover the score range")
        if (
            not isinstance(self.exp_table, tuple)
            or len(self.exp_table) != 1 << self.score_bits
        ):
            raise ProofV3Error("attention exp table must cover every score")
        for value in self.exp_table:
            if not isinstance(value, int) or not 1 <= value < 1 << 31:
                raise ProofV3Error(
                    "attention exp values must be in [1, 2^31)")
        if (self.key_count is None) != (self.query_positions is None):
            raise ProofV3Error(
                "attention rectangular mode needs key_count AND positions")
        if self.key_count is None and self.chunk_base is not None:
            raise ProofV3Error("attention chunking requires rect mode")
        if self.key_count is not None:
            if (
                not isinstance(self.key_count, int)
                or isinstance(self.key_count, bool)
                or not 1 <= self.key_count <= 1 << 20
            ):
                raise ProofV3Error("attention key count is out of range")
            positions = self.query_positions
            if (
                not isinstance(positions, tuple)
                or len(positions) != self.token_count
            ):
                raise ProofV3Error(
                    "attention query positions must cover every row")
            position_bound = (
                self.key_count if self.chunk_base is None else 1 << 31)
            previous = -1
            for value in positions:
                if (
                    not isinstance(value, int) or isinstance(value, bool)
                    or not previous < value < position_bound
                ):
                    raise ProofV3Error(
                        "attention query positions must be strictly "
                        "increasing and inside the key sequence")
                previous = value
            if self.chunk_base is not None:
                if (
                    not isinstance(self.chunk_base, int)
                    or isinstance(self.chunk_base, bool)
                    or not 0 <= self.chunk_base < 1 << 31
                ):
                    raise ProofV3Error(
                        "attention chunk base is out of range")
            elif self.public_totals is not None:
                raise ProofV3Error(
                    "attention public totals require chunked mode")
            if self.public_totals is not None:
                expected = self.head_pad() * self.token_pad()
                if (
                    not isinstance(self.public_totals, tuple)
                    or len(self.public_totals) != expected
                    or not all(
                        isinstance(v, int) and not isinstance(v, bool)
                        and 1 <= v < 1 << (_LIMB_COUNT * self.limb_bits)
                        for v in self.public_totals)
                ):
                    raise ProofV3Error(
                        "attention public totals table is malformed")
        # every per-cell total (hence rem / complement) must fit the
        # 3-limb remainder window
        if self.key_pad() * max(self.exp_table) >= 1 << (
            _LIMB_COUNT * self.limb_bits
        ):
            raise ProofV3Error("attention totals exceed the limb window")

    def head_pad(self) -> int:
        return _pow2_at_least(self.head_count)

    def token_pad(self) -> int:
        return _pow2_at_least(self.token_count)

    def key_pad(self) -> int:
        return _pow2_at_least(
            self.token_count if self.key_count is None else self.key_count)

    def row_positions(self) -> tuple[int, ...]:
        """Causal position of every padded query row.

        Square mode: row t sits at position t (the classic tril mask).
        Rectangular mode: the sampled positions; padding rows attend to
        the whole real key sequence (their inputs are zero, and totals
        stay positive because exp values are >= 1).
        """

        tp = self.token_pad()
        if self.key_count is None:
            return tuple(range(tp))
        pad_position = (self.chunk_base or 0) + self.key_count - 1
        return self.query_positions + (
            (pad_position,) * (tp - self.token_count))

    def raw_offset(self) -> int:
        return 1 << (self.shift + self.score_bits - 1)

    def scale(self) -> int:
        return 1 << self.scale_bits

    def digest(self) -> bytes:
        rect = b""
        if self.scored:
            rect += (
                b"SCORED"
                + struct.pack(
                    "<III", self.scored, self.m_e, self.capture_kv)
                + b"".join(m.to_bytes(8, "little") for m in self.m_nums)
            )
            if self.public_peaks is not None:
                rect += b"SCPK" + hashlib.sha256(
                    b"".join(v.to_bytes(8, "little")
                             for v in self.public_peaks)
                    + b"/"
                    + bytes(self.public_sel_count)
                ).digest()
        if self.rational:
            # the rational scheme is a DISTINCT signed ABI: bind the
            # flag and the ABI string so no V1 transcript can replay
            # against a V2 statement (or vice versa)
            rect += (
                b"RATIONAL"
                + struct.pack("<I", self.rational)
                + GOLDILOCKS_SUCCINCT_ATTENTION_RATIONAL_ABI.encode()
            )
        if self.key_count is not None:
            # APPEND -- a plain assignment here dropped the scored
            # metadata (scored/m_nums/m_e/capture_kv/peaks/selector)
            # from every rectangular scored digest (transcript-binding
            # bug): two statements differing only in signed slopes or
            # public peaks shared one Fiat-Shamir transcript
            rect += (
                b"RECT"
                + struct.pack("<I", self.key_count)
                + hashlib.sha256(
                    b"".join(
                        value.to_bytes(4, "little")
                        for value in self.query_positions)
                ).digest()
            )
            if self.chunk_base is not None:
                totals = b"" if self.public_totals is None else b"".join(
                    value.to_bytes(8, "little")
                    for value in self.public_totals)
                rect += (
                    b"RECT2"
                    + struct.pack("<I", self.chunk_base)
                    + hashlib.sha256(totals).digest()
                )
        if (
            self.pcs_query_count
            != GOLDILOCKS_MULTILINEAR_PCS_QUERY_COUNT_V3
        ):
            rect += (
                b"PCS_QUERY_COUNT"
                + struct.pack("<I", self.pcs_query_count)
            )
        return hashlib.sha256(
            _TILE_DOMAIN
            + self.validator_binding_digest
            + struct.pack(
                "<IIIIIIIII", self.head_count, self.token_count,
                self.head_dim, self.qk_bits, self.v_bits, self.shift,
                self.score_bits, self.scale_bits, self.limb_bits,
            )
            + _exp_table_digest_cached(tuple(self.exp_table))
            + rect
        ).digest()


@dataclass(frozen=True, slots=True)
class GoldilocksSuccinctAttentionProofV3:
    column_commitments: tuple[bytes, ...]         # ordered by _all_tags
    eq_folds: tuple[SuccinctEqFoldProofV3, ...]   # ordered by _fold_plan
    public_folds: tuple[SuccinctEqFoldProofV3, ...]  # es-mask, probs-zero, rowsum
    score_product: object
    pv_product: object
    division_product: object                      # chunked: public fold proof
    logups: tuple[object, ...]                    # ordered by _logup_plan
    # scored scheme: (ovf-spos, sel-sel, sel-peak, sel-su) product proofs
    scored_products: tuple = ()
    batch_openings: tuple[tuple[str, object], ...] | None = None
    # chunked tiles only: PUBLIC per-chunk row totals (h*t_pad,
    # row-major) and partial outputs (h*t_pad*d).  Bound to committed
    # cells by the row-sum fold / PV product terminal checks.
    chunk_totals: tuple[int, ...] | None = None
    partial_out: tuple[int, ...] | None = None


def _tile_digest(statement: GoldilocksSuccinctAttentionStatementV3) -> bytes:
    return hashlib.sha256(_TILE_DOMAIN + statement.digest()).digest()


def _log2(n: int) -> int:
    return n.bit_length() - 1


def _dims(statement):
    hp, tp, sp, d = (
        statement.head_pad(), statement.token_pad(), statement.key_pad(),
        statement.head_dim,
    )
    return hp, tp, sp, d, _log2(hp), _log2(tp), _log2(sp), _log2(d)


def _cube_vars(statement):
    _hp, _tp, _sp, _d, lh, lt, ls, ld = _dims(statement)
    n_s = lh + lt + ls                # (h, t, s)
    n_b = n_s + ld                    # (h, t, s, d)
    n_q = lh + lt + ld                # (h, t, d)
    n_tot = lh + lt                   # (h, t)
    return n_s, n_b, n_q, n_tot


_KV_TAGS: Final = ("k", "k_biased", "v", "v_biased")


def _tag_member_vars(statement, tag: str) -> int:
    _hp, _tp, _sp, _d, lh, lt, ls, ld = _dims(statement)
    if tag in _S_TAGS or tag in _S_TAGS_SCORED:
        return lh + lt + ls
    if tag in _B_TAGS:
        return lh + lt + ls + ld
    if tag in ("total", "peak"):
        return lh + lt
    if tag in _KV_TAGS:
        return lh + ls + ld           # (h, s, d)
    return lh + lt + ld               # (h, t, d)


def _tag_vars(statement, tag: str) -> int:
    if tag.startswith("grp_"):
        _member_map, group_vars = _group_layout(statement)
        return group_vars[tag]
    return _tag_member_vars(statement, tag)


def _fold_plan(statement, z_s, z_b, z_o):
    """Ordered (tag, point-label, point) eq-fold plan, shared by both sides.

    LSB-first slices: S-cube z = [s][t][h]; B-cube z = [d][s][t][h];
    q/out cube z = [d][t][h]; k/v cube z = [d][s][h]; total z = [t][h].
    """

    _hp, _tp, _sp, _d, _lh, lt, ls, ld = _dims(statement)
    z_bq = z_b[:ld] + z_b[ld + ls: ld + ls + lt] + z_b[ld + ls + lt:]
    z_bk = z_b[:ld] + z_b[ld: ld + ls] + z_b[ld + ls + lt:]
    z_bp = z_b[ld:]
    z_st = z_s[ls:]
    chunked = _is_chunked(statement)
    s_tags = _rational_filter(statement, (
        _S_TAGS_SCORED if getattr(statement, "scored", 0) else _S_TAGS))
    plan = [
        (tag, "zS", z_s) for tag in s_tags
        if not (chunked and tag in _CHUNK_DROPPED)]
    if not _is_rect(statement):
        plan += [(tag, "zB", z_b) for tag in _B_TAGS]
    if getattr(statement, "scored", 0) and getattr(
            statement, "capture_kv", 0):
        plan += [("q", "zBq", z_bq), ("q_biased", "zBq", z_bq)]
    else:
        plan += [
            ("q", "zBq", z_bq), ("q_biased", "zBq", z_bq),
            ("k", "zBk", z_bk), ("k_biased", "zBk", z_bk),
            ("v", "zBk", z_bk), ("v_biased", "zBk", z_bk),
        ]
    if not _is_rect(statement) and not getattr(statement, "rational", 0):
        plan += [("probs", "zBp", z_bp)]
    if not chunked:
        plan += [
            ("out", "zO", z_o),
            ("total", "zST", z_st),
        ]
    if getattr(statement, "scored", 0) and not chunked:
        plan += [("peak", "zST", z_st)]
    return plan


from functools import lru_cache


@lru_cache(maxsize=64)
def _packed_exp_cached(exp_table: tuple) -> tuple:
    return tuple(
        index + _QPACK * value for index, value in enumerate(exp_table))


@lru_cache(maxsize=64)
def _exp_table_digest_cached(exp_table: tuple) -> bytes:
    """sha256 of the packed exp table.  The table is a protocol constant
    shared by every statement, so its digest is cached by VALUE -- a
    statement carrying any other table still binds that table's own
    digest."""

    return hashlib.sha256(
        b"".join(v.to_bytes(8, "little") for v in exp_table)).digest()


def _logup_plan(statement):
    """Ordered (name, table, witness-tag) LogUp plan.

    Same-table range proofs merge into ONE instance whose witness is a
    whole block-tree group (zero pad blocks are valid table members).
    """

    packed_exp = _packed_exp_cached(statement.exp_table)
    if getattr(statement, "scored", 0):
        # scored: s_pos indexes the FIXED table; bucketing remainder limbs
        # rb_l0 (16-bit, merged) + rb_l1 (EXACT 2^(m_e-16) range: a loose
        # range would let the prover deflate su by up to that slack); ovf
        # limbs 16+16; division/complement limbs unchanged. sel/su/peak
        # need no LogUp (bound by products + identities).
        head = (
            ("s_pos_range",
             tuple(range(1 << statement.score_bits)), "s_pos"),
            ("exp_pack", packed_exp, "w_exp"),
        )
        if not getattr(statement, "rational", 0):
            # V2 has no per-key probabilities: the probs_range table
            # (2^(scale_bits+1) entries) ceases to exist
            head += (
                ("probs_range",
                 tuple(range(1 << (statement.scale_bits + 1))), "probs"),
            )
        return head + (
            ("rb1_range",
             tuple(range(1 << (statement.m_e - statement.limb_bits))),
             "rb_l1"),
        ) + tuple(
            # limb groups may be SPLIT at large cubes: one range instance
            # per effective group (same dynamic detection as V1)
            (f"limbs_range{i}" if i else "limbs_range",
             tuple(range(1 << statement.limb_bits)),
             members[0] if group_tag.startswith("solo/") else group_tag)
            for i, (group_tag, members) in enumerate(
                (g, m) for g, m in _group_plan(statement)
                if set(m) <= {
                    "rb_l0", "ov_l0", "ov_l1", "rem_l0", "rem_l1",
                    "rem_l2", "comp_l0", "comp_l1", "comp_l2"})
        ) + (
            ("q_range", tuple(range(1 << statement.qk_bits)), "q_biased"),
        ) + (() if getattr(statement, "capture_kv", 0) else (
            ("k_range", tuple(range(1 << statement.qk_bits)), "k_biased"),
            ("v_range", tuple(range(1 << statement.v_bits)), "v_biased"),
        ))
    plan = [
        ("qs_range", tuple(range(1 << statement.score_bits)), "qs"),
        ("rem_s_range", tuple(range(1 << max(1, statement.shift))), "rem_s"),
        ("exp_pack", packed_exp, "w_exp"),
        ("probs_range", tuple(range(1 << (statement.scale_bits + 1))),
         "probs"),
    ]
    limb_tags = {f"rem_l{i}" for i in range(_LIMB_COUNT)} | {
        f"comp_l{i}" for i in range(_LIMB_COUNT)}
    limb_groups = tuple(
        (group_tag, member_tags)
        for group_tag, member_tags in _group_plan(statement)
        if set(member_tags) <= limb_tags)
    limb_table = tuple(range(1 << statement.limb_bits))
    if len(limb_groups) == 1 and limb_groups[0][0] == "grp_limbs":
        plan.append(("limbs_range", limb_table, "grp_limbs"))
    else:
        plan.extend(
            (f"limbs_range{index}",
             limb_table,
             members[0] if group_tag.startswith("solo/") else group_tag)
            for index, (group_tag, members) in enumerate(limb_groups))
    if statement.qk_bits == statement.v_bits:
        biased = {"q_biased", "k_biased", "v_biased"}
        biased_groups = tuple(
            (group_tag, member_tags)
            for group_tag, member_tags in _group_plan(statement)
            if set(member_tags) <= biased)
        table = tuple(range(1 << statement.qk_bits))
        if len(biased_groups) == 1:
            plan.append(("biased_range", table, biased_groups[0][0]))
        else:
            # rectangular split: one range instance per split group;
            # singleton groups witness the member column directly
            plan.extend(
                (f"biased_range{index}",
                 table,
                 members[0] if group_tag.startswith("solo/") else group_tag)
                for index, (group_tag, members) in enumerate(biased_groups))
    else:
        plan.extend((
            ("q_range", tuple(range(1 << statement.qk_bits)), "q_biased"),
            ("k_range", tuple(range(1 << statement.qk_bits)), "k_biased"),
            ("v_range", tuple(range(1 << statement.v_bits)), "v_biased"),
        ))
    return tuple(plan)


def _eq_components(z_bits_lsb):
    """MSB-first tensor components of eq(z, .) for one LSB-first z part."""

    return tuple(
        ((1 - z) % GOLDILOCKS_MODULUS, z % GOLDILOCKS_MODULUS)
        for z in reversed(z_bits_lsb)
    )


def _ones_components(count: int):
    return tuple((1, 1) for _ in range(count))


def _product_setups(statement, z_s, z_o):
    """(name, a-tag, b-tag, factor components) for the three products."""

    _hp, _tp, _sp, _d, lh, lt, ls, ld = _dims(statement)
    z_s_s, z_s_t, z_s_h = z_s[:ls], z_s[ls: ls + lt], z_s[ls + lt:]
    z_o_d, z_o_t, z_o_h = z_o[:ld], z_o[ld: ld + lt], z_o[ld + lt:]
    eq_s_full = (
        _eq_components(z_s_h) + _eq_components(z_s_t) + _eq_components(z_s_s)
    )
    pv_factor = (
        _eq_components(z_o_h) + _eq_components(z_o_t)
        + _ones_components(ls) + _eq_components(z_o_d))
    if not _is_rect(statement):
        base = (
            ("scores", "q_b", "k_b", eq_s_full + _ones_components(ld),
             None, None, None, None),
            ("pv", "p_b", "v_b", pv_factor, None, None, None, None),
            ("division", "probs", "total_b", eq_s_full,
             None, None, None, None),
        )
        if getattr(statement, "scored", 0):
            base += (
                ("ovf-spos", "ovf", "s_pos", eq_s_full,
                 None, None, None, None),
                ("sel-sel", "sel", "sel", eq_s_full,
                 None, None, None, None),
                ("sel-peak", "sel", "peak_b", eq_s_full,
                 None, None, None, None),
                ("sel-su", "sel", "su", eq_s_full,
                 None, None, None, None),
            )
        return base
    # broadcast-free: LSB-first big point layout [d | s | t | h]
    n = ld + ls + lt + lh
    map_q = tuple(range(ld)) + tuple(range(ld + ls, n))          # (d,t,h)
    map_kv = tuple(range(ld + ls)) + tuple(range(ld + ls + lt, n))  # (d,s,h)
    map_probs = tuple(range(ld, n))                              # (s,t,h)
    rational = bool(getattr(statement, "rational", 0))
    # V2: the pv product's weight column is es_m (same S-cube layout as
    # probs), so ``out`` is the EXACT numerator sum es_m * v; there is
    # no division product anywhere
    pv_a = "es_m" if rational else "probs"
    setups = [
        ("scores", "q", "k", eq_s_full + _ones_components(ld),
         map_q, map_kv, "q_b", "k_b"),
        ("pv", pv_a, "v", pv_factor,
         map_probs, map_kv, "p_b", "v_b"),
    ]
    if not _is_chunked(statement) and not rational:
        setups.append(
            ("division", "probs", "total_b", eq_s_full,
             None, None, None, None))
    if getattr(statement, "scored", 0):
        setups += [
            ("ovf-spos", "ovf", "s_pos", eq_s_full,
             None, None, None, None),
            ("sel-sel", "sel", "sel", eq_s_full,
             None, None, None, None),
            ("sel-peak", "sel", "peak_b", eq_s_full,
             None, None, None, None),
            ("sel-su", "sel", "su", eq_s_full,
             None, None, None, None),
        ]
    return tuple(setups)


_STREAM_PRODUCT_MIN_CELLS: Final = 1 << 25
_CUBE_KIND: Final = {"q_b": "td", "k_b": "sd", "p_b": "row", "v_b": "sd"}


def _axis_factor_tables(components, bits_per_axis):
    """Per-axis factor tables from the MSB-first per-bit components --
    the flat factor value at (h,t,s,d) is the product of the four axis
    table entries (same doubling order as the monolithic build)."""

    from verallm.proof_v3.native_goldilocks_backend import (
        gl_mul_t,
        to_field_tensor,
    )

    tables = []
    position = 0
    for bits in bits_per_axis:
        table = to_field_tensor((1,), "cuda")
        for component in components[position:position + bits]:
            comp_t = to_field_tensor(tuple(component), "cuda")
            table = gl_mul_t(
                table.repeat_interleave(len(component)),
                comp_t.repeat(table.numel()))
        tables.append(table)
        position += bits
    return tables


def _make_cube_gen(statement, a_kind, a_enc, b_kind, b_enc, tables):
    """Chunk generator for the streamed product prover: base-cube slices
    of a/b/factor for flat range [lo, hi), all encoded int64 CUDA.
    Ranges are d-aligned (the stream chunk is a power of two >= d)."""

    import torch

    from verallm.proof_v3.native_goldilocks_backend import gl_mul_t

    hp, tp, sp, d, _lh, _lt, _ls, _ld = _dims(statement)
    h_tab, t_tab, s_tab, d_tab = tables

    def _pick(kind, enc, h, t, s, rows):
        if kind == "row":
            return enc.index_select(0, rows).repeat_interleave(d)
        idx = h * tp + t if kind == "td" else h * sp + s
        return enc.view(-1, d).index_select(0, idx).reshape(-1)

    def gen(lo, hi):
        rows = torch.arange(
            lo // d, hi // d, device="cuda", dtype=torch.long)
        h = rows // (tp * sp)
        rem = rows - h * (tp * sp)
        t = rem // sp
        s = rem - t * sp
        a = _pick(a_kind, a_enc, h, t, s, rows)
        b = _pick(b_kind, b_enc, h, t, s, rows)
        f_rows = gl_mul_t(
            gl_mul_t(
                h_tab.index_select(0, h), t_tab.index_select(0, t)),
            s_tab.index_select(0, s))
        f = gl_mul_t(
            f_rows.repeat_interleave(d), d_tab.repeat(rows.numel()))
        return a, b, f

    return gen


def _cube_source(statement, columns, committed, src_tag, own_tag):
    """Compact encoded source for one cube side: the COMMITTED column's
    device values when resident (guaranteed byte-consistency with the
    tree), else the encoded compact bcast source."""

    import torch

    column = committed.get(own_tag)
    device_values = getattr(column, "device_values", None)
    if device_values is not None:
        return device_values
    _axis, base = columns["bcast/src"][src_tag]
    flat = base.to("cuda").reshape(-1)
    return torch.where(flat < 0, flat - ((1 << 32) - 1), flat).contiguous()


def _bcast_tensor(statement, columns, tag, device):
    """Expanded broadcast column as a flat encoded tensor.

    Square tiles committed the broadcast, so the builder's encoded tensor
    is returned as-is.  Rect builders keep only the compact source
    (``bcast/src``): the hp*tp*sp*d expansion happens HERE, on the
    requested device, so it never materializes on host for the fused
    path.  Explicit ``columns[tag]`` entries (overrides) win."""

    import torch

    if tag in columns:
        t = columns[tag]
        return t.to(device) if device == "cuda" else t
    hp, tp, sp, d, *_rest = _dims(statement)
    axis, base = columns["bcast/src"][tag]
    if device == "cuda":
        base = base.to(device)
    flat = base.unsqueeze(axis).expand(hp, tp, sp, d).reshape(-1)
    return torch.where(
        flat < 0, flat - ((1 << 32) - 1), flat).contiguous()


def _product_statement(statement, tile_digest: bytes, name: str, components):
    return GoldilocksSuccinctProductStatementV3(
        validator_binding_digest=hashlib.sha256(
            tile_digest + b"prod/" + name.encode()).digest(),
        variable_count=len(components),
        factor_component_sizes=tuple(len(c) for c in components),
    )


def _signed_bound(value: int, name: str, bits: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProofV3Error(f"attention {name} must be an integer")
    bound = 1 << (bits - 1)
    if not -bound < value < bound:
        raise ProofV3Error(f"attention {name} exceeds its declared width")
    return value


def run_goldilocks_succinct_attention_reference_v3(
    *,
    statement: GoldilocksSuccinctAttentionStatementV3,
    q_heads,
    k_heads,
    v_heads,
):
    """Plain integer execution of the tile semantics (no proof)."""

    columns, outputs, _totals = _build_columns(
        statement, q_heads, k_heads, v_heads)
    del columns
    return outputs


def _build_columns(statement, q_heads, k_heads, v_heads):
    """Vectorized (torch int64) column builder; exact integer semantics."""

    import torch

    hp, tp, sp, d, _lh, _lt, _ls, _ld = _dims(statement)
    h_real, t_real = statement.head_count, statement.token_count
    s_real = (
        statement.token_count if statement.key_count is None
        else statement.key_count)
    for name, table, rows_real in (
        ("q", q_heads, t_real), ("k", k_heads, s_real),
        ("v", v_heads, s_real),
    ):
        if len(table) != h_real or any(
            len(rows) != rows_real or any(len(row) != d for row in rows)
            for rows in table
        ):
            raise ProofV3Error(f"attention {name} table shape is wrong")

    def load(table, name, bits, rows_pad, rows_real):
        bound = 1 << (bits - 1)
        dense = torch.zeros((hp, rows_pad, d), dtype=torch.int64)
        # tensor inputs pass through without a python-list round trip
        block = torch.as_tensor(table, dtype=torch.int64).cpu()
        if not bool((block > -bound).all() and (block < bound).all()):
            raise ProofV3Error(
                f"attention {name} exceeds its declared width")
        dense[:h_real, :rows_real, :] = block
        return dense

    q_t = load(q_heads, "q", statement.qk_bits, tp, t_real)
    k_t = load(k_heads, "k", statement.qk_bits, sp, s_real)
    v_t = load(v_heads, "v", statement.v_bits, sp, s_real)
    q_bias = 1 << (statement.qk_bits - 1)
    v_bias = 1 << (statement.v_bits - 1)
    shift, offset = statement.shift, statement.raw_offset()
    scale = statement.scale()
    window = 1 << (shift + statement.score_bits)
    raw = torch.einsum("htd,hsd->hts", q_t, k_t)
    qidx = raw + offset
    if not bool(((qidx >= 0) & (qidx < window)).all()):
        raise ProofV3Error("attention raw score escapes the quant window")
    qs = qidx >> shift
    rem_s = qidx & ((1 << shift) - 1)
    exp_t = torch.tensor(statement.exp_table, dtype=torch.int64)
    es = exp_t[qs]
    positions = torch.tensor(
        statement.row_positions(), dtype=torch.int64)
    chunk_base = getattr(statement, "chunk_base", None) or 0
    key_index = torch.arange(sp, dtype=torch.int64)
    mask = (
        (key_index + chunk_base).unsqueeze(0) <= positions.unsqueeze(1))
    if _is_chunked(statement):
        # pad key columns must never enter the PUBLIC chunk sums
        mask = mask & (key_index < statement.key_count).unsqueeze(0)
    mask = mask.to(torch.int64)                      # [tp, sp]
    es_m = es * mask.unsqueeze(0)
    chunk_sums = es_m.sum(dim=2, keepdim=True)     # [hp, tp, 1]
    if _is_chunked(statement):
        if statement.public_totals is None:
            raise ProofV3Error(
                "chunked attention prove requires public totals")
        total = torch.tensor(
            statement.public_totals, dtype=torch.int64,
            ).reshape(hp, tp, 1)
        if not bool((total >= chunk_sums).all()):
            raise ProofV3Error(
                "attention public totals are below this chunk's sums")
    else:
        total = chunk_sums
    probs = (es_m * scale) // total
    rem_d = es_m * scale - probs * total
    comp = total.expand_as(rem_d) - 1 - rem_d
    total_b = total.expand_as(rem_d)
    w_exp = qs + _QPACK * es
    out = torch.einsum("hts,hsd->htd", probs, v_t)
    limb_mask = (1 << statement.limb_bits) - 1

    def flat(t):
        return tuple(t.reshape(-1).tolist())

    def flat_field(t):
        # canonicalize signed values: p + x for x < 0
        return tuple(
            v if v >= 0 else v + GOLDILOCKS_MODULUS
            for v in t.reshape(-1).tolist())

    def enc(t):
        # encoded int64 tensor (canonical mod 2^64 wrap) - stays a
        # tensor: fused commits consume it without any host tolist
        return t.reshape(-1).contiguous()

    def enc_field(t):
        # for x < 0 the canonical p + x encodes as x - (2^32 - 1)
        t = t.reshape(-1)
        return torch.where(t < 0, t - ((1 << 32) - 1), t).contiguous()

    columns = {
        "raw": enc_field(raw),
        "qs": flat(qs),
        "rem_s": flat(rem_s),
        "w_exp": flat(w_exp),
        "es": flat(es),
        "es_m": flat(es_m),
        "probs": flat(probs),
        "rem_d": enc(rem_d),
        "comp": enc(comp),
        "total_b": enc(total_b),
        "q": enc_field(q_t),
        "q_biased": flat(q_t + q_bias),
        "k": enc_field(k_t),
        "k_biased": flat(k_t + q_bias),
        "v": enc_field(v_t),
        "v_biased": flat(v_t + v_bias),
        "out": enc_field(out),
        "total": flat(total.reshape(hp, tp)),
    }
    for i in range(_LIMB_COUNT):
        columns[f"rem_l{i}"] = flat(
            (rem_d >> (statement.limb_bits * i)) & limb_mask)
        columns[f"comp_l{i}"] = flat(
            (comp >> (statement.limb_bits * i)) & limb_mask)
    if _is_rect(statement):
        # rect tiles never commit the broadcasts; the product arguments
        # need them only as fold inputs, expanded lazily (on device for
        # the fused path) -- hp*tp*sp*d must NOT materialize on host
        columns["bcast/src"] = {
            "q_b": (2, q_t), "k_b": (1, k_t),
            "p_b": (3, probs), "v_b": (1, v_t)}
    else:
        columns["q_b"] = enc_field(
            q_t.unsqueeze(2).expand(hp, tp, sp, d))
        columns["k_b"] = enc_field(
            k_t.unsqueeze(1).expand(hp, tp, sp, d))
        columns["p_b"] = enc(
            probs.unsqueeze(3).expand(hp, tp, sp, d))
        columns["v_b"] = enc_field(
            v_t.unsqueeze(1).expand(hp, tp, sp, d))
    if _is_chunked(statement):
        for tag in _CHUNK_DROPPED:
            del columns[tag]
        # PUBLIC tables: chunk row sums + full padded partial output
        columns["public/chunk_totals"] = flat(
            chunk_sums.reshape(hp, tp))
        columns["public/partial_out"] = flat_field(out)
    out_host = out[:h_real, :t_real, :d].tolist()
    outputs = tuple(
        tuple(tuple(row) for row in head) for head in out_host)
    totals = flat(chunk_sums.reshape(hp, tp))
    return columns, outputs, totals


# scored-scheme S-cube column set (see the tile design doc): su/rem_b from
# the bucketing identity, peak_b/s_pos/ovf/sel from peak legitimacy + the
# branchless clamp, w_exp packs s_pos against the FIXED table.
_S_TAGS_SCORED: Final = (
    "raw", "su", "rem_b", "rb_l0", "rb_l1", "peak_b", "s_pos",
    "ovf", "ov_l0", "ov_l1", "sel", "w_exp", "es", "es_m", "probs",
    "rem_d", "comp", "total_b",
    "rem_l0", "rem_l1", "rem_l2", "comp_l0", "comp_l1", "comp_l2",
)
_SMALL_TAGS_SCORED: Final = _SMALL_TAGS + ("peak",)

# columns that exist ONLY for the per-key probability division -- the
# rational scheme (V2) removes all of them (and the probs_range LogUp
# table + the division fold that consumed them)
_RATIONAL_DROPPED: Final = (
    "probs", "rem_d", "comp", "total_b",
    "rem_l0", "rem_l1", "rem_l2", "comp_l0", "comp_l1", "comp_l2",
)


def _build_columns_scored(statement, q_heads, k_heads, v_heads,
                          device=None):
    """Scored-scheme column builder: EXACT scored_attention_reference
    semantics, vectorized.  All intermediates < 2^62 (int64-safe); the
    einsum products < 2^53 (fp64-exact).

    ``device``: witness device.  On CUDA the whole build stays
    device-resident (fp64-exact einsums under guarded windows, no
    python-list materialization) and the tensor-aware group commit
    consumes the columns without a host round trip; proof-public
    entries (chunk totals / partial numerators / peaks) are always
    materialized host-side."""

    import torch

    if not statement.scored:
        raise ProofV3Error("scored column builder needs a scored statement")
    on_gpu = device is not None and str(device) != "cpu"
    hp, tp, sp, d, _lh, _lt, _ls, _ld = _dims(statement)
    h_real, t_real = statement.head_count, statement.token_count
    s_real = (
        statement.token_count if statement.key_count is None
        else statement.key_count)

    def load(table, name, bits, rows_pad, rows_real):
        bound = 1 << (bits - 1)
        dense = torch.zeros(
            (hp, rows_pad, d), dtype=torch.int64, device=device)
        # tensor inputs pass through without a python-list round trip
        block = torch.as_tensor(table, dtype=torch.int64).to(device)
        if not bool((block > -bound).all() and (block < bound).all()):
            raise ProofV3Error(
                f"attention {name} exceeds its declared width")
        dense[:h_real, :rows_real, :] = block
        return dense

    def _exact_einsum(spec, a, b, window):
        """int64 einsum; CUDA has no int64 matmul, so the device path
        runs fp64 (exact under ``window``, guarded)."""
        if on_gpu:
            out_t = torch.einsum(
                spec, a.double(), b.double()).to(torch.int64)
            if not bool((out_t.abs() < (1 << window)).all()):
                raise ProofV3Error(
                    "attention einsum exceeds the fp64-exact window")
            return out_t
        return torch.einsum(spec, a, b)

    q_t = load(q_heads, "q", statement.qk_bits, tp, t_real)
    k_t = load(k_heads, "k", statement.qk_bits, sp, s_real)
    v_t = load(v_heads, "v", statement.v_bits, sp, s_real)
    q_bias = 1 << (statement.qk_bits - 1)
    v_bias = 1 << (statement.v_bits - 1)
    scale = statement.scale()
    raw = _exact_einsum("htd,hsd->hts", q_t, k_t, 52)
    m_vec = torch.zeros((hp, 1, 1), dtype=torch.int64, device=device)
    m_vec[:h_real, 0, 0] = torch.tensor(statement.m_nums, dtype=torch.int64)
    half = 1 << (statement.m_e - 1)
    prod = raw * m_vec + half
    if not bool((prod.abs() < (1 << 62)).all()):
        raise ProofV3Error("scored slope product overflows the safe window")
    su = prod >> statement.m_e                      # arithmetic = floor
    rem_b = prod - (su << statement.m_e)            # in [0, 2^m_e)
    positions = torch.tensor(
        statement.row_positions(), dtype=torch.int64, device=device)
    chunk_base = getattr(statement, "chunk_base", None) or 0
    key_index = torch.arange(sp, dtype=torch.int64, device=device)
    mask = (
        (key_index + chunk_base).unsqueeze(0) <= positions.unsqueeze(1))
    if _is_chunked(statement):
        mask = mask & (key_index < statement.key_count).unsqueeze(0)
    mask = mask.to(torch.int64)                     # [tp, sp]
    m3 = mask.unsqueeze(0)                          # [1, tp, sp]
    if _is_chunked(statement):
        # GLOBAL peak: decoded from the PUBLIC statement data
        peak = _signed_public_peaks_tensor_v3(
            statement.public_peaks, device=device).reshape(hp, tp, 1)
        if not bool((m3 * (peak.expand_as(su) - su) >= 0).all()):
            raise ProofV3Error(
                "scored chunk score exceeds the public row peak")
    else:
        neg = -(1 << 62)
        peak = su.masked_fill(m3.expand_as(su) == 0, neg).amax(
            dim=2, keepdim=True)                    # [hp, tp, 1]
        peak = torch.where(peak == neg, torch.zeros_like(peak), peak)
    peak_b = peak.expand_as(su)
    dgap = m3 * (peak_b - su)                       # masked: 0
    smax = (1 << statement.score_bits) - 1
    s_pos = dgap.clamp(max=smax)
    ovf = dgap - s_pos
    if not bool((ovf < (1 << (2 * statement.limb_bits))).all()):
        raise ProofV3Error(
            "scored clamp overflow exceeds its two-limb range window")
    # sel: one-hot at the FIRST visible position achieving the peak
    is_peak = ((su == peak_b) & (m3.expand_as(su) == 1))
    first = torch.cumsum(is_peak.to(torch.int64), dim=2)
    sel = (is_peak & (first == 1)).to(torch.int64)
    if _is_chunked(statement):
        # ties (and padding rows) can place achiever-valued cells in more
        # than one chunk: the selector is only EMITTED in the publicly
        # designated chunk, and that chunk must actually contain one
        want = torch.tensor(
            statement.public_sel_count, dtype=torch.int64,
            device=device).reshape(hp, tp)
        sel = sel * want.unsqueeze(2)
        if not bool((sel.sum(dim=2) == want).all()):
            raise ProofV3Error(
                "scored chunk selector count disagrees with the public "
                "declaration")
    exp_t = torch.tensor(
        statement.exp_table, dtype=torch.int64, device=device)
    es = exp_t[s_pos]
    es_m = es * m3
    chunk_sums = es_m.sum(dim=2, keepdim=True)
    if _is_chunked(statement):
        if statement.public_totals is None:
            raise ProofV3Error(
                "chunked attention prove requires public totals")
        total = torch.tensor(
            statement.public_totals, dtype=torch.int64,
            device=device).reshape(hp, tp, 1)
        if not bool((total >= chunk_sums).all()):
            raise ProofV3Error(
                "attention public totals are below this chunk's sums")
    else:
        total = chunk_sums
    total_safe = torch.where(total == 0, torch.ones_like(total), total)
    rational = bool(getattr(statement, "rational", 0))
    if rational:
        # RATIONAL (V2): out is the EXACT integer numerator sum
        # es_m * v -- no probability rounding, no division chain.
        # es_m <= 2^22 and |v| < 2^(v_bits-1), so a row numerator is
        # < sp * 2^(21 + v_bits): int64-safe and in-field; the fp64
        # einsum stays exact below 2^53 (guarded by the same window
        # as the raw product above)
        probs = rem_d = comp = total_b = None
        out = _exact_einsum("hts,hsd->htd", es_m, v_t, 53)
        if not on_gpu and not bool((out.abs() < (1 << 53)).all()):
            raise ProofV3Error(
                "rational numerator exceeds the fp64 exact window")
    else:
        # ROUNDED division: floor((es_m*2*scale + total) / (2*total))
        num = es_m * (2 * scale) + total_safe
        probs = num // (2 * total_safe)
        rem_d = num - probs * 2 * total_safe
        comp = 2 * total_safe.expand_as(rem_d) - 1 - rem_d
        total_b = total_safe.expand_as(rem_d)
        out = _exact_einsum("hts,hsd->htd", probs * m3, v_t, 53)
    w_exp = s_pos + _QPACK * es
    limb_mask = (1 << statement.limb_bits) - 1

    def flat_host(t):
        return tuple(t.reshape(-1).tolist())

    def flat(t):
        if on_gpu:
            return t.reshape(-1).contiguous()
        return tuple(t.reshape(-1).tolist())

    def enc(t):
        return t.reshape(-1).contiguous()

    def enc_field(t):
        t = t.reshape(-1)
        return torch.where(t < 0, t - ((1 << 32) - 1), t).contiguous()

    columns = {
        "raw": enc_field(raw), "su": enc_field(su), "rem_b": enc(rem_b),
        "rb_l0": flat(rem_b & limb_mask),
        # high limb UNMASKED: its exact 2^(m_e-limb_bits) range table is the
        # soundness bound (a masked split would drop rem_b's high bits)
        "rb_l1": flat(rem_b >> statement.limb_bits),
        "peak_b": enc_field(peak_b), "s_pos": flat(s_pos),
        "ovf": enc(ovf), "ov_l0": flat(ovf & limb_mask),
        "ov_l1": flat((ovf >> statement.limb_bits) & limb_mask),
        "sel": flat(sel), "w_exp": flat(w_exp), "es": flat(es),
        "es_m": flat(es_m),
        "q": enc_field(q_t), "q_biased": flat(q_t + q_bias),
        "out": enc_field(out),
        "total": flat_host(total_safe.reshape(hp, tp)),
        "peak": tuple(
            v if v >= 0 else v + GOLDILOCKS_MODULUS
            for v in peak.reshape(-1).tolist()),
    }
    if not rational:
        columns["probs"] = flat(probs)
        columns["rem_d"] = enc(rem_d)
        columns["comp"] = enc(comp)
        columns["total_b"] = enc(total_b)
        for i in range(_LIMB_COUNT):
            columns[f"rem_l{i}"] = flat(
                (rem_d >> (statement.limb_bits * i)) & limb_mask)
            columns[f"comp_l{i}"] = flat(
                (comp >> (statement.limb_bits * i)) & limb_mask)
    if not getattr(statement, "capture_kv", 0):
        columns["k"] = enc_field(k_t)
        columns["k_biased"] = flat(k_t + q_bias)
        columns["v"] = enc_field(v_t)
        columns["v_biased"] = flat(v_t + v_bias)
    if _is_rect(statement):
        # rect tiles never commit the broadcasts; expand lazily at
        # product-fold time (see _bcast_tensor) -- hp*tp*sp*d must NOT
        # materialize on host.  V2: the pv weight source is es_m
        columns["bcast/src"] = {
            "q_b": (2, q_t), "k_b": (1, k_t),
            "p_b": (3, es_m if rational else probs), "v_b": (1, v_t)}
    else:
        columns["q_b"] = enc_field(
            q_t.unsqueeze(2).expand(hp, tp, sp, d))
        columns["k_b"] = enc_field(
            k_t.unsqueeze(1).expand(hp, tp, sp, d))
        columns["p_b"] = enc(
            probs.unsqueeze(3).expand(hp, tp, sp, d))
        columns["v_b"] = enc_field(
            v_t.unsqueeze(1).expand(hp, tp, sp, d))
    if _is_chunked(statement):
        for tag in _CHUNK_DROPPED + ("peak",):
            # total_b never exists under the rational scheme
            columns.pop(tag, None)
        columns["public/chunk_totals"] = flat_host(
            chunk_sums.reshape(hp, tp))
        columns["public/partial_out"] = tuple(
            v if v >= 0 else v + GOLDILOCKS_MODULUS
            for v in out.reshape(-1).tolist())
    out_host = out[:h_real, :t_real, :d].tolist()
    outputs = tuple(
        tuple(tuple(row) for row in head) for head in out_host)
    totals = flat_host(chunk_sums.reshape(hp, tp))
    return columns, outputs, totals


def _division_factor(statement, z_s):
    """Public factor eq(z_s,.) * T_bcast for the chunked division fold."""

    import numpy as np

    from verallm.proof_v3.goldilocks_numpy import gl_mul_np

    hp, tp, sp, _d, _lh, _lt, _ls, _ld = _dims(statement)
    eq = np.array(_eq_table(z_s), dtype=np.uint64).reshape(hp, tp, sp)
    totals = np.array(
        statement.public_totals, dtype=np.uint64).reshape(hp, tp, 1)
    product = gl_mul_np(
        eq.reshape(-1), np.broadcast_to(totals, (hp, tp, sp)).reshape(-1))
    return tuple(product.tolist())


def _public_mle(values, z_point) -> int:
    """MLE of a public integer table at an LSB-first point (exact)."""

    import numpy as np

    from verallm.proof_v3.goldilocks_numpy import gl_mul_np

    eq = np.array(_eq_table(z_point), dtype=np.uint64)
    table = np.array(
        [v % GOLDILOCKS_MODULUS for v in values], dtype=np.uint64)
    if eq.shape != table.shape:
        raise ProofV3Error("public table size does not match the point")
    terms = gl_mul_np(eq, table).tolist()
    result = 0
    for term in terms:
        result += term
    return result % GOLDILOCKS_MODULUS


def _eq_table_np(point):
    """eq table as a numpy uint64 array DIRECTLY (the reference
    _eq_table detours through a python list -- tolist + re-array cost
    seconds at multi-million-cell cubes; the math is identical)."""

    import numpy as np

    from verallm.proof_v3.goldilocks_numpy import gl_mul_np

    table = np.ones(1, dtype=np.uint64)
    for r in point:
        rr = np.broadcast_to(
            np.uint64(r % GOLDILOCKS_MODULUS), table.shape).copy()
        om = np.broadcast_to(
            np.uint64((1 - r) % GOLDILOCKS_MODULUS),
            table.shape).copy()
        table = np.concatenate(
            [gl_mul_np(table, om), gl_mul_np(table, rr)])
    return table


def _mask_factors(statement, z_s):
    """(eq*mask, eq*(1-mask), eq_tot broadcast) public-fold factors."""

    hp, tp, sp, _d, _lh, _lt, ls, _ld = _dims(statement)
    import numpy as np

    eq = _eq_table_np(z_s).reshape(hp, tp, sp)
    eq_tot = _eq_table_np(z_s[ls:])
    positions = np.array(statement.row_positions(), dtype=np.int64)
    chunk_base = getattr(statement, "chunk_base", None) or 0
    key_index = np.arange(sp, dtype=np.int64)
    lower = (key_index + chunk_base)[None, :] <= positions[:, None]
    if _is_chunked(statement):
        lower = lower & (key_index < statement.key_count)[None, :]
    f_mask = np.where(lower, eq, np.uint64(0))
    f_inv = np.where(lower, np.uint64(0), eq)
    f_rowsum = np.broadcast_to(
        eq_tot.reshape(hp, tp, 1), (hp, tp, sp))
    # ndarray factors: the public-fold prover/verifier consume numpy
    # natively (tuple materialization of hp*tp*sp cells cost ~20s at 262k)
    return (np.ascontiguousarray(f_mask.reshape(-1)),
            np.ascontiguousarray(f_inv.reshape(-1)),
            np.ascontiguousarray(f_rowsum.reshape(-1)))


def _m_factor(statement, z_s):
    """Public bucketing factor: (eq[h,t,s] * M_h) mod p -- validator-built
    from the SIGNED per-head slope mantissas."""

    import numpy as np

    from verallm.proof_v3.goldilocks_numpy import gl_mul_np

    hp, tp, sp, _d, _lh, _lt, _ls, _ld = _dims(statement)
    eq_s = _eq_table_np(z_s).reshape(hp, tp * sp)
    m_pad = np.zeros((hp, 1), dtype=np.uint64)
    m_pad[:statement.head_count, 0] = np.array(
        statement.m_nums, dtype=np.uint64)
    return gl_mul_np(
        eq_s.reshape(-1).copy(),
        np.broadcast_to(m_pad, (hp, tp * sp)).reshape(-1).copy())


def _attention_pcs_query_scope(function):
    """Run every nested dynamic PCS statement under the tile's budget."""

    @wraps(function)
    def scoped(*args, **kwargs):
        statement = kwargs.get("statement")
        if not isinstance(statement, GoldilocksSuccinctAttentionStatementV3):
            raise ProofV3Error(
                "succinct attention call lacks its validator-owned statement")
        with pcs_query_count_v3(statement.pcs_query_count):
            return function(*args, **kwargs)

    return scoped


@_attention_pcs_query_scope
def prove_goldilocks_succinct_attention_v3(
    *,
    statement: GoldilocksSuccinctAttentionStatementV3,
    q_heads,
    k_heads,
    v_heads,
    validator_nonce: bytes,
    fused=None,
    columns_override=None,
    aggregate: bool = False,
    external_collector=None,
    collector_ns: str = "",
    precommitted=None,
    skip_group_logups: bool = False,
):
    """Prove one full attention layer; returns (proof, integer outputs).

    ``q_heads/k_heads/v_heads`` are per-Q-head integer tables
    ``[head][token][dim]`` (the caller replicates shared KV heads).
    ``columns_override`` is for adversarial tests only.
    """

    del aggregate  # grouped trees require batched openings: always on
    import os as _os
    import time as _time
    _trace = _os.environ.get("VERATHOS_ATTN_TRACE") == "1"

    def _mark(label, _t0=[_time.perf_counter()]):
        if _trace:
            import torch as _torch
            peak = ""
            if _torch.cuda.is_available():
                _torch.cuda.synchronize()
                peak = (
                    f" peak={_torch.cuda.max_memory_allocated() / 2**30:.2f}"
                    f"GB now={_torch.cuda.memory_allocated() / 2**30:.2f}GB")
            now = _time.perf_counter()
            print(f"ATTN {label}: +{now - _t0[0]:.2f}s{peak}", flush=True)
            _t0[0] = now
    tile_digest = _tile_digest(statement)
    scored_stmt = bool(getattr(statement, "scored", 0))
    if scored_stmt:
        import torch as _torch2

        build_device = (
            "cuda" if fused is not None
            and _torch2.cuda.is_available() else None)
        columns, outputs, _totals = _build_columns_scored(
            statement, q_heads, k_heads, v_heads,
            device=build_device)
    else:
        columns, outputs, _totals = _build_columns(
            statement, q_heads, k_heads, v_heads)
    _mark("build")
    if columns_override is not None:
        columns.update(columns_override)
    from verallm.proof_v3.goldilocks_succinct_batch_opening import (
        BatchOpeningCollectorV3,
    )
    from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
        commit_succinct_column_group_v3,
    )

    if external_collector is not None:
        from verallm.proof_v3.goldilocks_succinct_batch_opening import (
            NamespacedCollectorV3,
        )

        collector = NamespacedCollectorV3(external_collector, collector_ns)
    else:
        collector = BatchOpeningCollectorV3()

    committed = {}
    groups = {}
    group_plan = _group_plan(statement)
    if precommitted is not None:
        # CROSS-CHUNK TREES: the LAYER prover committed the heavy groups
        # into shared per-group trees (chunk index in the block bits) and
        # registered them on the raw collector; this tile commits only the
        # groups NOT covered (e.g. the per-chunk limb groups, whose merged
        # range LogUps need per-chunk group witnesses). groups stays empty
        # for merged groups -> the fused batched-fold fast path is skipped.
        committed = dict(precommitted)
        group_plan = tuple(
            (g, m) for g, m in group_plan if m[0] not in committed)
    for group_tag, member_tags in group_plan:
        group, members = commit_succinct_column_group_v3(
            tile_digest=tile_digest, group_tag=group_tag,
            ordered=tuple(
                (tag, columns[tag]
                 if hasattr(columns[tag], "numel")
                 else tuple(columns[tag]))
                for tag in member_tags),
            fused=fused)
        groups[group_tag] = group
        committed.update(members)
        collector.register_group(group)
        for tag in member_tags:
            collector.register_column(tag, members[tag])
    if "total" not in committed and not _is_chunked(statement):
        committed["total"] = commit_succinct_column_v3(
            tile_digest=tile_digest, tag="total",
            values=tuple(columns["total"]), fused=fused,
            canonical_input=True)
        collector.register_column("total", committed["total"])
    if ("peak" not in committed and getattr(statement, "scored", 0)
            and not _is_chunked(statement)):
        committed["peak"] = commit_succinct_column_v3(
            tile_digest=tile_digest, tag="peak",
            values=tuple(columns["peak"]), fused=None,
            canonical_input=True)
        collector.register_column("peak", committed["peak"])
    _mark("commit")
    commitments = tuple(
        committed[tag].tree.commitment for tag in _all_tags(statement))
    n_s, n_b, n_q, _n_tot = _cube_vars(statement)
    z_s = derive_tile_eq_point_v3(
        tile_digest, commitments, validator_nonce, n_s, label=b"zS")
    z_b = derive_tile_eq_point_v3(
        tile_digest, commitments, validator_nonce, n_b, label=b"zB")
    z_o = derive_tile_eq_point_v3(
        tile_digest, commitments, validator_nonce, n_q, label=b"zO")
    plan = _fold_plan(statement, z_s, z_b, z_o)
    if fused is not None and hasattr(fused[0], "round_partials_b"):
        from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
            prove_succinct_eq_folds_batched_v3,
        )

        # same-point folds batch per contiguous group: grp_s core,
        # grp_limbs, grp_bcast; the rest fold individually
        plan_map = dict(group_plan)
        by_tag = {}
        for batch_group in ("grp_s", "grp_limbs"):
            members_ = plan_map.get(batch_group)
            if members_ is None:
                continue  # split group (big rect): generic folds cover it
            batch = prove_succinct_eq_folds_batched_v3(
                tile_digest=tile_digest,
                members=tuple(committed[tag] for tag in members_),
                group_device=groups[batch_group].device_values,
                z_point=z_s, validator_nonce=validator_nonce,
                fused=fused, collector=collector)
            by_tag.update(zip(members_, batch, strict=True))
        if "grp_bcast" in groups:
            b_batch = prove_succinct_eq_folds_batched_v3(
                tile_digest=tile_digest,
                members=tuple(committed[tag] for tag in _B_TAGS),
                group_device=groups["grp_bcast"].device_values,
                z_point=z_b, validator_nonce=validator_nonce,
                fused=fused, collector=collector)
            by_tag.update(zip(_B_TAGS, b_batch, strict=True))
        eq_folds = tuple(
            by_tag[tag] if (tag in by_tag and label in ("zS", "zB"))
            else prove_succinct_eq_fold_v3(
                tile_digest=tile_digest, column=committed[tag],
                z_point=point, validator_nonce=validator_nonce,
                fused=fused, collector=collector)
            for tag, label, point in plan
        )
    else:
        eq_folds = tuple(
            prove_succinct_eq_fold_v3(
                tile_digest=tile_digest, column=committed[tag],
                z_point=point, validator_nonce=validator_nonce,
                fused=fused, collector=collector)
            for tag, _label, point in plan
        )
    _mark("eq_folds")
    f_mask, f_inv, f_rowsum = _mask_factors(statement, z_s)
    fold_specs = [
        ("es", "es-mask", f_mask),
        ("es_m", "row-sum", f_rowsum),
    ]
    if not getattr(statement, "rational", 0):
        fold_specs.insert(1, ("probs", "probs-zero", f_inv))
    if getattr(statement, "scored", 0):
        fold_specs += [
            ("raw", "bucket-m", _m_factor(statement, z_s)),
            ("su", "su-mask", f_mask),
            ("peak_b", "peakb-mask", f_mask),
            ("sel", "sel-invmask", f_inv),
            ("sel", "sel-rowsum", f_rowsum),
        ]
    if _is_chunked(statement) and not getattr(statement, "rational", 0):
        fold_specs.append(
            ("probs", "division", _division_factor(statement, z_s)))
    _rational_stmt = bool(getattr(statement, "rational", 0))
    public_folds = tuple(
        prove_succinct_public_fold_v3(
            tile_digest=tile_digest, column=committed[tag], factor=factor,
            label=label, validator_nonce=validator_nonce, fused=fused,
            collector=collector,
            structured_binding=(
                _pubfold_binding_v2(tile_digest, label)
                if _rational_stmt else None))
        for tag, label, factor in fold_specs
    )
    _mark("public_folds")
    def _bcast_host(source_tag):
        return tuple(
            v + (1 << 64) if v < 0 else v
            for v in _bcast_tensor(
                statement, columns, source_tag, "cpu").tolist())

    products = {}
    for (name, a_tag, b_tag, components, a_map, b_map,
         a_src, b_src) in _product_setups(statement, z_s, z_o):
        prod_statement = _product_statement(
            statement, tile_digest, name, components)
        if fused is not None:
            from verallm.proof_v3.native_pcs_backend import (
                fused_prove_goldilocks_succinct_product_v3,
            )

            _hp, _tp, _sp, _d, lh, lt, ls, ld = _dims(statement)
            cube_cells = 1 << len(components)
            stream_kwargs = {}
            a_dev = b_dev = None
            if (a_src is not None
                    and cube_cells >= _STREAM_PRODUCT_MIN_CELLS
                    and "bcast/src" in columns
                    and _d == 1 << ld):
                # OUT-OF-CORE product: never materialize the hp*tp*sp*d
                # broadcast cubes (the 262k VRAM dominator); chunks are
                # generated from the compact sources instead
                tables = _axis_factor_tables(
                    components, (lh, lt, ls, ld))
                a_enc = _cube_source(
                    statement, columns, committed, a_src, a_tag)
                b_enc = _cube_source(
                    statement, columns, committed, b_src, b_tag)
                stream_kwargs = dict(
                    stream_gen=_make_cube_gen(
                        statement, _CUBE_KIND[a_src], a_enc,
                        _CUBE_KIND[b_src], b_enc, tables),
                    stream_cells=cube_cells,
                    # compact prefix: fold on the un-broadcast operands
                    compact_spec=(
                        _CUBE_KIND[a_src], a_enc, _CUBE_KIND[b_src],
                        b_enc, tables,
                        (_hp, _tp, _sp, _d)))
            else:
                a_dev = (
                    None if a_src is None
                    else _bcast_tensor(statement, columns, a_src, "cuda"))
                b_dev = (
                    None if b_src is None
                    else _bcast_tensor(statement, columns, b_src, "cuda"))
            products[name] = fused_prove_goldilocks_succinct_product_v3(
                fold_extension=fused[0], tree_extension=fused[1],
                statement=prod_statement,
                a_column=committed[a_tag], b_column=committed[b_tag],
                factor_components=components,
                validator_nonce=validator_nonce,
                collector=collector, a_tag=a_tag, b_tag=b_tag,
                a_point_map=a_map, b_point_map=b_map,
                a_fold_device=a_dev, b_fold_device=b_dev,
                **stream_kwargs)
        else:
            products[name] = prove_goldilocks_succinct_product_v3(
                statement=prod_statement,
                a_pcs_statement=committed[a_tag].pcs_statement,
                b_pcs_statement=committed[b_tag].pcs_statement,
                a_tree=committed[a_tag].tree,
                b_tree=committed[b_tag].tree,
                a_evaluations=(
                    committed[a_tag].values if a_src is None
                    else _bcast_host(a_src)),
                b_evaluations=(
                    committed[b_tag].values if b_src is None
                    else _bcast_host(b_src)),
                factor_components=components,
                validator_nonce=validator_nonce,
                collector=collector, a_tag=a_tag, b_tag=b_tag,
                a_point_map=a_map, b_point_map=b_map)
    _mark("products")
    # the raw DEVICE column tensors' last readers are the eq/product
    # folds: drop them BEFORE the logup group commits stack their own
    # NTT transients on top.  Host entries stay (the chunked wire reads
    # public/* below); callers holding their own refs keep theirs.
    for _tag in [t for t, v in columns.items()
                 if getattr(v, "is_cuda", False)]:
        del columns[_tag]
    logup_instances = []
    _lg_plan = tuple(
        entry for entry in _logup_plan(statement)
        if not (skip_group_logups and entry[2].startswith("grp_")))
    for name, table, column_tag in _lg_plan:
        column = (
            groups[column_tag] if column_tag.startswith("grp_")
            else committed[column_tag])
        logup_statement = GoldilocksSuccinctLogupStatementV3(
            validator_binding_digest=hashlib.sha256(
                tile_digest + b"logup/" + name.encode()).digest(),
            table=table,
            witness_variable_count=_tag_vars(statement, column_tag),
            witness_binding_override=(
                column.pcs_statement.validator_binding_digest),
        )
        logup_instances.append(
            (logup_statement, column, f"logup/{name}", column_tag))
    if fused is not None:
        from verallm.proof_v3.native_pcs_backend import (
            fused_prove_logup_batch_v3,
        )

        logups = fused_prove_logup_batch_v3(
            fold_extension=fused[0], tree_extension=fused[1],
            tile_digest=tile_digest, instances=logup_instances,
            validator_nonce=validator_nonce, collector=collector)
    else:
        logups = [
            prove_goldilocks_succinct_logup_v3(
                statement=logup_statement,
                looked_up_values=column.values,
                validator_nonce=validator_nonce,
                witness_tree=column.tree,
                collector=collector, tag_prefix=tag_prefix,
                witness_tag=column_tag)
            for logup_statement, column, tag_prefix, column_tag
            in logup_instances
        ]
    _mark("logups")
    if external_collector is not None:
        # cross-tile aggregation: the LAYER prover finalizes one shared
        # batch-opening set; this tile carries none.  Park every
        # committed column's device values until that claims phase --
        # nothing in between reads them, and the stacking is the
        # long-context peak
        from verallm.proof_v3.goldilocks_succinct_batch_opening import (
            park_column_device_values_v3,
        )

        for _column in (*groups.values(), *committed.values()):
            park_column_device_values_v3(_column)
        batch_openings = ()
    else:
        batch_openings = tuple(sorted(collector.prove_all(
            validator_nonce=validator_nonce, fused=fused).items()))
    _mark("openings")
    chunked = _is_chunked(statement)
    scored_prods = (
        tuple(products[n]
              for n in ("ovf-spos", "sel-sel", "sel-peak", "sel-su"))
        if getattr(statement, "scored", 0) else ())
    proof = GoldilocksSuccinctAttentionProofV3(
        column_commitments=commitments,
        eq_folds=eq_folds,
        public_folds=public_folds,
        score_product=products["scores"],
        pv_product=products["pv"],
        division_product=products.get("division"),
        logups=tuple(logups),
        scored_products=scored_prods,
        batch_openings=batch_openings,
        chunk_totals=(
            tuple(columns["public/chunk_totals"]) if chunked else None),
        partial_out=(
            tuple(columns["public/partial_out"]) if chunked else None),
    )
    return proof, outputs


@_attention_pcs_query_scope
def verify_goldilocks_succinct_attention_v3(
    proof: object,
    *,
    statement: GoldilocksSuccinctAttentionStatementV3,
    validator_nonce: bytes,
    external_checker=None,
    checker_ns: str = "",
    grouped_aux_hint: bool | None = None,
    precommitted_layout: bool = False,
    skip_group_logups: bool = False,
    capture_kv_roots: dict | None = None,
) -> None | tuple:
    """Succinct CPU verification of one full attention layer."""

    try:
        if not isinstance(proof, GoldilocksSuccinctAttentionProofV3):
            raise ProofV3VerificationError(
                "succinct attention proof type is wrong")
        tile_digest = _tile_digest(statement)
        tag_list = _all_tags(statement)
        if len(proof.column_commitments) != len(tag_list):
            raise ProofV3VerificationError(
                "succinct attention column set is wrong")
        tags = dict(zip(tag_list, proof.column_commitments, strict=True))
        if getattr(statement, "capture_kv", 0):
            # K/V commitments come from the capture plane (per-layer PCS
            # columns equality-bound to the capture roots) -- fail closed
            if (
                not isinstance(capture_kv_roots, dict)
                or set(capture_kv_roots) != {"k", "v"}
            ):
                raise ProofV3VerificationError(
                    "capture-kv statements require external k/v roots")
            tags.update(capture_kv_roots)
        from verallm.proof_v3.goldilocks_succinct_batch_opening import (
            BatchClaimCheckerV3,
        )

        if external_checker is not None:
            from verallm.proof_v3.goldilocks_succinct_batch_opening import (
                NamespacedCheckerV3,
            )

            checker = NamespacedCheckerV3(external_checker, checker_ns)
        else:
            checker = BatchClaimCheckerV3()
        member_map, group_vars = _group_layout(statement)
        _merged = (
            precommitted_layout
            if isinstance(precommitted_layout, frozenset)
            else (frozenset() if not precommitted_layout else None))
        for tag, (group_tag, block_point) in member_map.items():
            if _merged is None or group_tag in _merged:
                continue
            checker.alias(tag, group_tag, block_point)
        group_plan = _group_plan(statement)
        # every member of one group must report the SAME (group) root
        for group_tag, member_tags in group_plan:
            roots = {tags[tag] for tag in member_tags}
            if len(roots) != 1:
                raise ProofV3VerificationError(
                    "succinct attention group roots disagree")
        n_s, n_b, n_q, _n_tot = _cube_vars(statement)
        z_s = derive_tile_eq_point_v3(
            tile_digest, tuple(proof.column_commitments), validator_nonce,
            n_s, label=b"zS")
        z_b = derive_tile_eq_point_v3(
            tile_digest, tuple(proof.column_commitments), validator_nonce,
            n_b, label=b"zB")
        z_o = derive_tile_eq_point_v3(
            tile_digest, tuple(proof.column_commitments), validator_nonce,
            n_q, label=b"zO")
        plan = _fold_plan(statement, z_s, z_b, z_o)
        if len(proof.eq_folds) != len(plan):
            raise ProofV3VerificationError(
                "succinct attention fold set is wrong")
        folds = {}
        for (tag, label, point), fold_proof in zip(
            plan, proof.eq_folds, strict=True
        ):
            folds[(tag, label)] = verify_succinct_eq_fold_v3(
                fold_proof,
                tile_digest=tile_digest,
                tag=tag,
                pcs_statement=column_pcs_statement_v3(
                    tile_digest, tag, _tag_vars(statement, tag)),
                commitment=tags[tag],
                z_point=point,
                validator_nonce=validator_nonce,
                checker=checker,
            )
        rational = bool(getattr(statement, "rational", 0))
        if rational:
            # closed-form factor evaluation: the cube-sized tables
            # never materialize verifier-side (opening-v2 Part 2)
            f_mask = f_inv = f_rowsum = None
        else:
            f_mask, f_inv, f_rowsum = _mask_factors(statement, z_s)
        fold_specs = [
            ("es", "es-mask", f_mask),
            ("es_m", "row-sum", f_rowsum),
        ]
        if not rational:
            fold_specs.insert(1, ("probs", "probs-zero", f_inv))
        scored = bool(getattr(statement, "scored", 0))
        if scored:
            fold_specs += [
                ("raw", "bucket-m",
                 None if rational else _m_factor(statement, z_s)),
                ("su", "su-mask", f_mask),
                ("peak_b", "peakb-mask", f_mask),
                ("sel", "sel-invmask", f_inv),
                ("sel", "sel-rowsum", f_rowsum),
            ]
        chunked = _is_chunked(statement)
        if chunked:
            hp_, tp_, _sp_, d_, _lh_, _lt_, _ls_, _ld_ = _dims(statement)
            if (
                not isinstance(proof.chunk_totals, tuple)
                or len(proof.chunk_totals) != hp_ * tp_
                or not isinstance(proof.partial_out, tuple)
                or len(proof.partial_out) != hp_ * tp_ * d_
                or not all(
                    isinstance(v, int) and 0 <= v < GOLDILOCKS_MODULUS
                    for v in proof.chunk_totals + proof.partial_out)
            ):
                raise ProofV3VerificationError(
                    "succinct attention public tables are malformed")
            if not rational:
                fold_specs.append(
                    ("probs", "division",
                     _division_factor(statement, z_s)))
        if len(proof.public_folds) != len(fold_specs):
            raise ProofV3VerificationError(
                "succinct attention public-fold set is wrong")
        pub = {}
        for fold_proof, (tag, label, factor) in zip(
            proof.public_folds,
            fold_specs,
            strict=True,
        ):
            pub[label] = verify_succinct_public_fold_v3(
                fold_proof,
                tile_digest=tile_digest,
                label=label,
                pcs_statement=column_pcs_statement_v3(
                    tile_digest, tag, _tag_vars(statement, tag)),
                commitment=tags[tag],
                factor=factor,
                validator_nonce=validator_nonce,
                checker=checker,
                tag=tag,
                factor_eval=(
                    (lambda ch, _lb=label: pubfold_factor_eval_v3(
                        statement, z_s, _lb, ch))
                    if rational else None),
                structured_binding=(
                    _pubfold_binding_v2(tile_digest, label)
                    if rational else None),
            )
        claims = {}
        if rational and proof.division_product is not None:
            raise ProofV3VerificationError(
                "unexpected division product on a rational statement")
        product_proofs = (
            (proof.score_product, proof.pv_product)
            if chunked or rational
            else (proof.score_product, proof.pv_product,
                  proof.division_product))
        if scored:
            if len(proof.scored_products) != 4:
                raise ProofV3VerificationError(
                    "scored attention product set is wrong")
            product_proofs = product_proofs + tuple(proof.scored_products)
        elif proof.scored_products:
            raise ProofV3VerificationError(
                "unexpected scored products on a non-scored statement")
        for (name, a_tag, b_tag, components, a_map, b_map,
             _a_src, _b_src), prod_proof in zip(
            _product_setups(statement, z_s, z_o),
            product_proofs,
            strict=True,
        ):
            prod_statement = _product_statement(
                statement, tile_digest, name, components)
            claims[name] = prod_proof.claimed_sum % GOLDILOCKS_MODULUS
            verify_goldilocks_succinct_product_v3(
                prod_proof,
                statement=prod_statement,
                a_pcs_statement=column_pcs_statement_v3(
                    tile_digest, a_tag, _tag_vars(statement, a_tag)),
                b_pcs_statement=column_pcs_statement_v3(
                    tile_digest, b_tag, _tag_vars(statement, b_tag)),
                a_commitment=tags[a_tag],
                b_commitment=tags[b_tag],
                factor_components=components,
                validator_nonce=validator_nonce,
                expected_sum=claims[name],
                checker=checker, a_tag=a_tag, b_tag=b_tag,
                a_point_map=a_map, b_point_map=b_map,
            )

        # ---- linear couplings (each pins a per-cell relation by SZ) ----
        def check(condition: bool, message: str) -> None:
            if not condition:
                raise ProofV3VerificationError(
                    f"succinct attention {message}")

        p = GOLDILOCKS_MODULUS
        if scored:
            scale = statement.scale()
            half = 1 << (statement.m_e - 1)
            # scores: sum eq*raw == sum q_b*k_b*eq over the broadcast cube
            check(claims["scores"] == folds[("raw", "zS")],
                  "score product coupling fails")
            # bucketing: raw*M + 2^(E-1) == su*2^E + rem_b (sum eq == 1
            # absorbs the constant; M enters via the public bucket factor)
            check(
                (pub["bucket-m"] + half) % p
                == ((1 << statement.m_e) * folds[("su", "zS")]
                    + folds[("rem_b", "zS")]) % p,
                "bucketing coupling fails")
            # rem_b limb recomposition (16-bit + exact high limb)
            check(
                folds[("rem_b", "zS")]
                == (folds[("rb_l0", "zS")]
                    + (1 << statement.limb_bits)
                    * folds[("rb_l1", "zS")]) % p,
                "bucketing remainder limb coupling fails")
            # branchless clamp on VISIBLE cells:
            # mask*(peak_b - su) == s_pos + ovf
            check(
                (pub["peakb-mask"] - pub["su-mask"]) % p
                == (folds[("s_pos", "zS")] + folds[("ovf", "zS")]) % p,
                "clamp split coupling fails")
            check(
                folds[("ovf", "zS")]
                == (folds[("ov_l0", "zS")]
                    + (1 << statement.limb_bits)
                    * folds[("ov_l1", "zS")]) % p,
                "overflow limb coupling fails")
            # saturation: ovf * (smax - s_pos) == 0 per cell
            smax = (1 << statement.score_bits) - 1
            check(
                (smax * folds[("ovf", "zS")]) % p == claims["ovf-spos"],
                "clamp saturation coupling fails")
            # sel booleanity, one-per-row, visibility, peak achievement
            check(folds[("sel", "zS")] == claims["sel-sel"],
                  "peak selector booleanity coupling fails")
            _dimv = _dims(statement)
            _z_st = z_s[_dimv[6]:]
            if chunked:
                check(
                    pub["sel-rowsum"] == _public_mle(
                        tuple(statement.public_sel_count), _z_st),
                    "peak selector row-sum coupling fails")
            else:
                check(pub["sel-rowsum"] == 1,
                      "peak selector row-sum coupling fails")
            check(pub["sel-invmask"] == 0,
                  "peak selector visibility coupling fails")
            check(claims["sel-peak"] == claims["sel-su"],
                  "peak achievement coupling fails")
            # peak broadcast pinned to the committed (or PUBLIC) row peak
            if chunked:
                check(
                    folds[("peak_b", "zS")] == _public_mle(
                        tuple(statement.public_peaks), _z_st),
                    "peak broadcast coupling fails")
            else:
                check(folds[("peak_b", "zS")] == folds[("peak", "zST")],
                      "peak broadcast coupling fails")
            # exp pack against the FIXED table: w_exp == s_pos + 2^32*es
            check(
                folds[("w_exp", "zS")]
                == (folds[("s_pos", "zS")]
                    + _QPACK * folds[("es", "zS")]) % p,
                "exp packing coupling fails")
            check(pub["es-mask"] == folds[("es_m", "zS")],
                  "mask coupling fails")
            if not rational:
                check(pub["probs-zero"] == 0,
                      "masked probability coupling fails")
            if chunked:
                check(pub["row-sum"] == _public_mle(
                    proof.chunk_totals, _z_st),
                    "row-sum coupling fails")
                if not rational:
                    t_mle = _public_mle(statement.public_totals, _z_st)
                    # ROUNDED division against the PUBLIC global totals
                    check(
                        (2 * scale * folds[("es_m", "zS")] + t_mle) % p
                        == (2 * pub["division"]
                            + folds[("rem_d", "zS")]) % p,
                        "division coupling fails")
                    check(
                        (folds[("rem_d", "zS")]
                         + folds[("comp", "zS")] + 1) % p
                        == (2 * t_mle) % p,
                        "remainder bound coupling fails")
            else:
                check(pub["row-sum"] == folds[("total", "zST")],
                      "row-sum coupling fails")
                if not rational:
                    check(
                        folds[("total_b", "zS")]
                        == folds[("total", "zST")],
                        "total broadcast coupling fails")
                    # ROUNDED division: es_m*2*scale + total ==
                    # probs*2*total + rem
                    check(
                        (2 * scale * folds[("es_m", "zS")]
                         + folds[("total_b", "zS")]) % p
                        == (2 * claims["division"]
                            + folds[("rem_d", "zS")]) % p,
                        "division coupling fails")
                    check(
                        (folds[("rem_d", "zS")]
                         + folds[("comp", "zS")] + 1) % p
                        == (2 * folds[("total_b", "zS")]) % p,
                        "remainder bound coupling fails")
            if not rational:
                for base, limb_prefix in (
                    ("rem_d", "rem_l"), ("comp", "comp_l")):
                    recomposed = 0
                    for i in range(_LIMB_COUNT):
                        recomposed = (
                            recomposed
                            + (1 << (statement.limb_bits * i))
                            * folds[(f"{limb_prefix}{i}", "zS")]
                        ) % p
                    check(folds[(base, "zS")] == recomposed,
                          f"{base} limb coupling fails")
            # broadcast + signed-bias couplings (same as V1; rect tiles
            # are broadcast-free -- products open small cubes directly)
            if not _is_rect(statement):
                check(folds[("q_b", "zB")] == folds[("q", "zBq")],
                      "q broadcast coupling fails")
                check(folds[("k_b", "zB")] == folds[("k", "zBk")],
                      "k broadcast coupling fails")
                check(folds[("v_b", "zB")] == folds[("v", "zBk")],
                      "v broadcast coupling fails")
                check(folds[("p_b", "zB")] == folds[("probs", "zBp")],
                      "probs broadcast coupling fails")
            q_bias = 1 << (statement.qk_bits - 1)
            v_bias = 1 << (statement.v_bits - 1)
            check(
                (folds[("q_biased", "zBq")]
                 - folds[("q", "zBq")]) % p == q_bias,
                "q bias coupling fails")
            if not getattr(statement, "capture_kv", 0):
                check(
                    (folds[("k_biased", "zBk")]
                     - folds[("k", "zBk")]) % p == q_bias,
                    "k bias coupling fails")
                check(
                    (folds[("v_biased", "zBk")]
                     - folds[("v", "zBk")]) % p == v_bias,
                    "v bias coupling fails")
            if chunked:
                check(claims["pv"] == _public_mle(proof.partial_out, z_o),
                      "output product coupling fails")
            else:
                check(claims["pv"] == folds[("out", "zO")],
                      "output product coupling fails")
        else:
            # scores: sum eq*raw == sum q_b*k_b*eq over the broadcast cube
            check(claims["scores"] == folds[("raw", "zS")],
                  "score product coupling fails")
            # Euclid quant: raw + offset == 2^shift * qs + rem_s
            check(
                (folds[("raw", "zS")] + statement.raw_offset()) % p
                == ((1 << statement.shift) * folds[("qs", "zS")]
                    + folds[("rem_s", "zS")]) % p,
                "quantization coupling fails")
            # exp pack: w_exp == qs + 2^32 * es
            check(
                folds[("w_exp", "zS")]
                == (folds[("qs", "zS")] + _QPACK * folds[("es", "zS")]) % p,
                "exp packing coupling fails")
            # causal mask: es_m == es * mask
            check(pub["es-mask"] == folds[("es_m", "zS")],
                  "mask coupling fails")
            # masked probabilities are zero
            check(pub["probs-zero"] == 0, "masked probability coupling fails")
            _hpv, _tpv, _spv, _dv, _lhv, _ltv, lsv, _ldv = _dims(statement)
            z_st_pt = z_s[lsv:]
            if chunked:
                # row sums: PUBLIC chunk totals == sum_s es_m
                check(
                    pub["row-sum"] == _public_mle(proof.chunk_totals, z_st_pt),
                    "row-sum coupling fails")
                # softmax division vs the PUBLIC global totals broadcast
                check(
                    statement.scale() * folds[("es_m", "zS")] % p
                    == (pub["division"] + folds[("rem_d", "zS")]) % p,
                    "division coupling fails")
                # remainder bound: rem_d + comp + 1 == T_bcast (public MLE)
                check(
                    (folds[("rem_d", "zS")] + folds[("comp", "zS")] + 1) % p
                    == _public_mle(statement.public_totals, z_st_pt),
                    "remainder bound coupling fails")
            else:
                # row sums: total == sum_s es_m
                check(pub["row-sum"] == folds[("total", "zST")],
                      "row-sum coupling fails")
                # total broadcast onto the score cube
                check(folds[("total_b", "zS")] == folds[("total", "zST")],
                      "total broadcast coupling fails")
                # softmax division: es_m * SCALE == probs * total_b + rem_d
                check(
                    statement.scale() * folds[("es_m", "zS")] % p
                    == (claims["division"] + folds[("rem_d", "zS")]) % p,
                    "division coupling fails")
                # remainder bound: rem_d + comp + 1 == total_b
                check(
                    (folds[("rem_d", "zS")] + folds[("comp", "zS")] + 1) % p
                    == folds[("total_b", "zS")],
                    "remainder bound coupling fails")
            # limb recompositions
            for base, limb_prefix in (("rem_d", "rem_l"), ("comp", "comp_l")):
                recomposed = 0
                for i in range(_LIMB_COUNT):
                    recomposed = (
                        recomposed
                        + (1 << (statement.limb_bits * i))
                        * folds[(f"{limb_prefix}{i}", "zS")]
                    ) % p
                check(folds[(base, "zS")] == recomposed,
                      f"{base} limb coupling fails")
            if not _is_rect(statement):
                # broadcast columns pinned to their small-cube originals
                check(folds[("q_b", "zB")] == folds[("q", "zBq")],
                      "q broadcast coupling fails")
                check(folds[("k_b", "zB")] == folds[("k", "zBk")],
                      "k broadcast coupling fails")
                check(folds[("v_b", "zB")] == folds[("v", "zBk")],
                      "v broadcast coupling fails")
                check(folds[("p_b", "zB")] == folds[("probs", "zBp")],
                      "probs broadcast coupling fails")
            # signed range biases
            q_bias = 1 << (statement.qk_bits - 1)
            v_bias = 1 << (statement.v_bits - 1)
            check(
                (folds[("q_biased", "zBq")] - folds[("q", "zBq")]) % p == q_bias,
                "q bias coupling fails")
            check(
                (folds[("k_biased", "zBk")] - folds[("k", "zBk")]) % p == q_bias,
                "k bias coupling fails")
            check(
                (folds[("v_biased", "zBk")] - folds[("v", "zBk")]) % p == v_bias,
                "v bias coupling fails")
            # output: sum eq*out == sum p_b*v_b*eq over the broadcast cube
            if chunked:
                check(claims["pv"] == _public_mle(proof.partial_out, z_o),
                      "output product coupling fails")
            else:
                check(claims["pv"] == folds[("out", "zO")],
                      "output product coupling fails")

        # ---- LogUps: witness commitment IS the tile column root ----
        _lg_plan_v = tuple(
            entry for entry in _logup_plan(statement)
            if not (skip_group_logups and entry[2].startswith("grp_")))
        if len(proof.logups) != len(_lg_plan_v):
            raise ProofV3VerificationError(
                "succinct attention logup set is wrong")
        def _witness_root(column_tag: str) -> bytes:
            if column_tag.startswith("grp_"):
                members = dict(group_plan)[column_tag]
                return tags[members[0]]
            return tags[column_tag]

        # aux aliases MUST be registered before any logup verification
        # files its deferred claims
        if grouped_aux_hint is not None:
            # aggregated tiles carry no own openings; the LAYER verifier
            # detects grouped aux from the shared opening set
            grouped_aux = grouped_aux_hint
        else:
            grouped_aux = checker is not None and any(
                tag.startswith("logup_aux/")
                for tag, _p in proof.batch_openings)
        aux_statements_grouped: dict = {}
        aux_commitments_grouped: dict = {}
        if grouped_aux:
            from verallm.proof_v3.native_pcs_backend import (
                logup_aux_group_plan_v3,
            )

            plan_list = tuple(zip(
                _lg_plan_v, proof.logups, strict=True))
            shapes = tuple(
                (f"logup/{name}",
                 _tag_vars(statement, column_tag),
                 GoldilocksSuccinctLogupStatementV3(
                     validator_binding_digest=hashlib.sha256(
                         tile_digest + b"logup/" + name.encode()
                     ).digest(),
                     table=table,
                     witness_variable_count=_tag_vars(
                         statement, column_tag),
                 ).table_variable_count)
                for (name, table, column_tag), _lp in plan_list)
            plans, group_meta = logup_aux_group_plan_v3(shapes)
            proof_by_prefix = {
                f"logup/{name}": logup_proof
                for (name, _t, _c), logup_proof in plan_list
            }
            for kind in ("M", "D", "E"):
                for prefix, (group_tag, block_point) in plans[
                    kind
                ].items():
                    local = (
                        f"{prefix}/{kind}" if kind == "M"
                        else f"{prefix}/{kind}0")
                    checker.alias(local, group_tag, block_point)
                    member = proof_by_prefix[prefix]
                    root = (
                        member.multiplicity_commitment
                        if kind == "M"
                        else member.inverse_commitments[
                            0 if kind == "D" else 1])
                    vars_total, _used = group_meta[group_tag]
                    aux_statements_grouped[group_tag] = (
                        column_pcs_statement_v3(
                            tile_digest, group_tag, vars_total))
                    if aux_commitments_grouped.setdefault(
                        group_tag, root
                    ) != root:
                        raise ProofV3VerificationError(
                            "succinct attention aux group roots disagree")

        for (name, table, column_tag), logup_proof in zip(
            _lg_plan_v, proof.logups, strict=True
        ):
            logup_statement = GoldilocksSuccinctLogupStatementV3(
                validator_binding_digest=hashlib.sha256(
                    tile_digest + b"logup/" + name.encode()).digest(),
                table=table,
                witness_variable_count=_tag_vars(statement, column_tag),
                witness_binding_override=column_pcs_statement_v3(
                    tile_digest, column_tag,
                    _tag_vars(statement, column_tag),
                ).validator_binding_digest,
            )
            verify_goldilocks_succinct_logup_v3(
                logup_proof, statement=logup_statement,
                witness_commitment=_witness_root(column_tag),
                validator_nonce=validator_nonce,
                checker=checker, tag_prefix=f"logup/{name}",
                witness_tag=column_tag)
        if checker is not None:
            from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (  # noqa: E501
                logup_batch_registry_v3,
            )

            _mg = (
                precommitted_layout
                if isinstance(precommitted_layout, frozenset)
                else (frozenset() if not precommitted_layout
                      else frozenset(g for g, _m in group_plan)))
            statements = {
                group_tag: column_pcs_statement_v3(
                    tile_digest, group_tag, group_vars[group_tag])
                for group_tag, _members in group_plan
                if group_tag not in _mg
            }
            commitments = {
                group_tag: tags[member_tags[0]]
                for group_tag, member_tags in group_plan
                if group_tag not in _mg
            }
            if getattr(statement, "capture_kv", 0):
                for _ckv in ("k", "v"):
                    statements[_ckv] = column_pcs_statement_v3(
                        tile_digest, _ckv, _tag_vars(statement, _ckv))
                    commitments[_ckv] = tags[_ckv]
            if not precommitted_layout:
                if not chunked:
                    statements["total"] = column_pcs_statement_v3(
                        tile_digest, "total",
                        _tag_vars(statement, "total"))
                    commitments["total"] = tags["total"]
                if scored and not chunked:
                    statements["peak"] = column_pcs_statement_v3(
                        tile_digest, "peak", _tag_vars(statement, "peak"))
                    commitments["peak"] = tags["peak"]
            statements.update(aux_statements_grouped)
            commitments.update(aux_commitments_grouped)
            plan_entries = tuple(zip(
                _lg_plan_v, proof.logups, strict=True))
            for (name, table, column_tag), logup_proof in plan_entries:
                logup_statement = GoldilocksSuccinctLogupStatementV3(
                    validator_binding_digest=hashlib.sha256(
                        tile_digest + b"logup/" + name.encode()).digest(),
                    table=table,
                    witness_variable_count=_tag_vars(statement, column_tag),
                    witness_binding_override=column_pcs_statement_v3(
                        tile_digest, column_tag,
                        _tag_vars(statement, column_tag),
                    ).validator_binding_digest,
                )
                if not grouped_aux:
                    aux_statements, aux_commitments = (
                        logup_batch_registry_v3(
                            logup_proof, logup_statement, f"logup/{name}",
                            witness_tag=column_tag))
                    statements.update(aux_statements)
                    commitments.update(aux_commitments)
            if external_checker is not None:
                # cross-tile aggregation: the LAYER verifier runs one
                # shared verify_all; hand it this tile's registry
                if proof.batch_openings:
                    raise ProofV3VerificationError(
                        "aggregated tile must not carry its own openings")
                return (
                    {checker_ns + k: v for k, v in statements.items()},
                    {checker_ns + k: v for k, v in commitments.items()},
                )
            checker.verify_all(
                dict(proof.batch_openings),
                statements=statements,
                commitments=commitments,
                validator_nonce=validator_nonce)
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "succinct attention proof is malformed") from exc


def verify_rational_chunked_layer_v3(*, sections, validator_nonce: bytes,
                                     expected_key_count: int,
                                     chunk_len: int,
                                     params_by_head,
                                     bounds_by_head,
                                     ox8_rows,
                                     capture_kv_roots=None,
                                     shared_openings=None):
    """PRODUCTION rational (V2) chunk aggregation for ONE layer.

    ``sections`` is the ordered tuple of (statement, proof) chunk
    sections.  Fail-closed enforcement, in order:

    * every statement is a rational scored CHUNKED tile and all
      digest-relevant global fields agree across sections;
    * chunks form the CANONICAL contiguous cover of
      ``[0, expected_key_count)`` at ``chunk_len`` -- any omission,
      duplication, reordering or gap is malformed before any proof
      verifies;
    * the peak-achiever selector counts sum to exactly 1 per row
      across the cover;
    * every chunk proof verifies (verify_goldilocks_succinct_
      attention_v3);
    * every proven chunk total and chunk numerator is summed EXACTLY
      once; the aggregate total must equal the authenticated global
      ``public_totals`` per row;
    * the exact aggregates bind the AUTHENTICATED runtime o_proj
      capture through the cross-multiplied rational bridge
      (verify_output_bridge_rational_v3) with the SIGNED V2 bounds --
      ``ox8_rows[h][t]`` must be validator-owned material, never
      wire-supplied.

    Returns (numerators, totals) for the REAL heads/rows:
    ``numerators[h][t][d]`` exact ints, ``totals[h][t]``."""

    from verallm.proof_v3.scored_attention_reference import (
        verify_output_bridge_rational_v3,
    )

    items = tuple(sections)
    if not items:
        raise ProofV3VerificationError(
            "rational layer needs at least one chunk section")
    if len(items) > 1 << 10:
        raise ProofV3VerificationError(
            "rational layer chunk count is out of bounds")
    if int(chunk_len) < 1 or int(expected_key_count) < 1:
        raise ProofV3VerificationError(
            "invalid rational layer geometry")
    first = items[0][0]
    shared = (
        "validator_binding_digest", "head_count", "token_count",
        "head_dim", "qk_bits", "v_bits", "shift", "exp_table",
        "score_bits", "scale_bits", "limb_bits", "query_positions",
        "public_totals", "public_peaks", "scored", "m_nums", "m_e",
        "capture_kv", "rational", "pcs_query_count",
    )
    for statement, _proof in items:
        if not getattr(statement, "rational", 0) or not statement.scored:
            raise ProofV3VerificationError(
                "rational layer sections must use the rational scored "
                "scheme")
        if getattr(statement, "chunk_base", None) is None:
            raise ProofV3VerificationError(
                "rational layer sections must be chunked tiles")
        for name in shared:
            if getattr(statement, name) != getattr(first, name):
                raise ProofV3VerificationError(
                    "rational layer sections disagree on shared "
                    "statement data")
    expected_bases = tuple(
        range(0, int(expected_key_count), int(chunk_len)))
    got = tuple(
        (int(statement.chunk_base), int(statement.key_count))
        for statement, _proof in items)
    want = tuple(
        (base, min(int(chunk_len), int(expected_key_count) - base))
        for base in expected_bases)
    if got != want:
        raise ProofV3VerificationError(
            "rational layer chunks do not form the canonical "
            "contiguous cover (omitted, duplicated, reordered or "
            "misaligned)")
    hp, tp = first.head_pad(), first.token_pad()
    d = first.head_dim
    half_p = GOLDILOCKS_MODULUS >> 1
    sel_sum = [0] * (hp * tp)
    totals = [0] * (hp * tp)
    numerators = [[0] * d for _ in range(hp * tp)]
    if shared_openings is not None:
        # MERGED layer: cross-chunk group trees, ONE batch-opening
        # set.  All chunk transcripts + claims verify through the
        # layer replay before the aggregation below consumes the
        # proven publics.
        if capture_kv_roots is not None:
            raise ProofV3VerificationError(
                "merged rational layers do not carry capture-kv "
                "binding yet")
        from verallm.proof_v3.goldilocks_scored_attention_layer import (
            ScoredLayerProofV3,
            verify_scored_attention_layer_merged_v3,
        )

        verify_scored_attention_layer_merged_v3(
            ScoredLayerProofV3(
                chunk_proofs=tuple(p for _s, p in items),
                batch_openings=tuple(shared_openings)),
            statements=tuple(s for s, _p in items),
            validator_nonce=validator_nonce)
    for statement, proof in items:
        if shared_openings is None:
            verify_goldilocks_succinct_attention_v3(
                proof, statement=statement,
                validator_nonce=validator_nonce,
                capture_kv_roots=capture_kv_roots)
        for i, v in enumerate(statement.public_sel_count):
            sel_sum[i] += int(v)
        for i, v in enumerate(proof.chunk_totals):
            totals[i] += int(v)
        for i in range(hp * tp):
            row = numerators[i]
            for dd in range(d):
                x = int(proof.partial_out[i * d + dd])
                row[dd] += x - GOLDILOCKS_MODULUS if x > half_p else x
    if any(v != 1 for v in sel_sum):
        raise ProofV3VerificationError(
            "rational layer selector counts do not cover every row "
            "exactly once")
    if tuple(totals) != tuple(int(v) for v in first.public_totals):
        raise ProofV3VerificationError(
            "rational layer aggregate total does not equal the "
            "authenticated global total")
    for row in numerators:
        for x in row:
            if abs(x) >= 1 << 53:
                raise ProofV3VerificationError(
                    "rational layer aggregate numerator is out of "
                    "bounds")
    h_real, t_real = first.head_count, first.token_count
    num_out = []
    tot_out = []
    for h in range(h_real):
        num_rows = [numerators[h * tp + t] for t in range(t_real)]
        tot_rows = [totals[h * tp + t] for t in range(t_real)]
        ox_h = [tuple(int(v) for v in ox8_rows[h][t])
                for t in range(t_real)]
        for row in ox_h:
            if len(row) != d:
                raise ProofV3VerificationError(
                    "authenticated o_x row does not span head_dim")
        verify_output_bridge_rational_v3(
            params=params_by_head[h],
            numerator_rows=num_rows, total_rows=tot_rows,
            ox8_rows=ox_h, bounds=bounds_by_head[h])
        num_out.append(tuple(tuple(row) for row in num_rows))
        tot_out.append(tuple(tot_rows))
    return tuple(num_out), tuple(tot_out)


__all__ = [
    "GOLDILOCKS_SUCCINCT_ATTENTION_ABI_V3",
    "GOLDILOCKS_SUCCINCT_ATTENTION_RATIONAL_ABI",
    "GoldilocksSuccinctAttentionProofV3",
    "GoldilocksSuccinctAttentionStatementV3",
    "prove_goldilocks_succinct_attention_v3",
    "run_goldilocks_succinct_attention_reference_v3",
    "verify_goldilocks_succinct_attention_v3",
    "verify_rational_chunked_layer_v3",
]


# ---------------------------------------------------------------------------
# CLOSED-FORM public-fold factor evaluation (verifier O(polylog)).
#
# The public-fold verifier needs factor_MLE at the terminal challenges
# (MSB-first fold order).  Every rational-tile factor is either an
# eq(z_s,.)-product table with an axis-structured indicator or an
# axis-broadcast eq -- so the evaluation factorizes per bit:
#     eq(z,i) * eq(i,r) = prod_b ( u_b*[i_b=0] + v_b*[i_b=1] ),
#     u_b = (1-z_b)(1-r_b),  v_b = z_b*r_b,   ones_b = u_b + v_b
# and the causal mask [s <= bound_t] evaluates by a weighted binary
# prefix walk.  Bit b (LSB-first flat (h,t,s) layout) maps to
# challenges[n-1-b].
# ---------------------------------------------------------------------------


_PUBFOLD_BINDING_DOMAIN_V2 = (
    b"VERATHOS/PROOF_V3/PUBFOLD_FACTOR_BINDING/V2")


def _pubfold_binding_v2(tile_digest: bytes, label: str) -> bytes:
    """Structured factor binding (rational statements): the factor is
    a pure function of (tile_digest, label, statement publics), all
    already transcript-bound -- hashing the cube-sized table bytes
    added nothing and forced the verifier to materialize it."""

    import hashlib as _h

    return _h.sha256(
        _PUBFOLD_BINDING_DOMAIN_V2 + tile_digest
        + label.encode()).digest()


def _pubfold_weights(z_s, challenges):
    n = len(z_s)
    p = GOLDILOCKS_MODULUS
    u, v = [], []
    for b in range(n):
        z = int(z_s[b]) % p
        r = int(challenges[n - 1 - b]) % p
        u.append((1 - z) % p * ((1 - r) % p) % p)
        v.append(z * r % p)
    ones = [(a + c) % p for a, c in zip(u, v)]
    return u, v, ones


def _prefix_weighted(bound: int, u, v, ones) -> int:
    """sum over s <= bound of prod_b weights(s_b), bits LSB-first."""

    p = GOLDILOCKS_MODULUS
    if bound < 0:
        return 0
    n = len(u)
    below = [1] * (n + 1)
    for k in range(n):
        below[k + 1] = below[k] * ones[k] % p
    value, carry = 0, 1
    for k in range(n - 1, -1, -1):
        if (bound >> k) & 1:
            value = (value + carry * u[k] % p * below[k]) % p
            carry = carry * v[k] % p
        else:
            carry = carry * u[k] % p
    return (value + carry) % p


def _axis_weight(index: int, u, v) -> int:
    p = GOLDILOCKS_MODULUS
    out = 1
    for k, (a, c) in enumerate(zip(u, v)):
        out = out * (c if (index >> k) & 1 else a) % p
    return out


def pubfold_factor_eval_v3(statement, z_s, label: str, challenges):
    """Closed-form factor MLE at the terminal challenges for the
    rational tile's public folds -- EXACTLY mle_eval_msb(table, .) of
    the corresponding _mask_factors/_m_factor table."""

    p = GOLDILOCKS_MODULUS
    hp, tp, sp, _d, lh, lt, ls, _ld = _dims(statement)
    u, v, ones = _pubfold_weights(z_s, challenges)
    s_u, s_v, s_ones = u[:ls], v[:ls], ones[:ls]
    t_u, t_v = u[ls:ls + lt], v[ls:ls + lt]
    h_ones = ones[ls + lt:]
    total = 1
    for w in ones:
        total = total * w % p
    if label in ("es-mask", "su-mask", "peakb-mask", "sel-invmask"):
        h_part = 1
        for w in h_ones:
            h_part = h_part * w % p
        base = getattr(statement, "chunk_base", None) or 0
        positions = statement.row_positions()
        key_count = (
            statement.key_count if statement.key_count is not None
            else statement.token_count)
        acc = 0
        for t in range(tp):
            bound = min(int(positions[t]) - base, int(key_count) - 1)
            acc = (acc + _axis_weight(t, t_u, t_v)
                   * _prefix_weighted(bound, s_u, s_v, s_ones)) % p
        masked = h_part * acc % p
        if label == "sel-invmask":
            return (total - masked) % p
        return masked
    if label in ("row-sum", "sel-rowsum"):
        z_tot = z_s[ls:]
        out = 1
        n_ht = len(z_tot)
        for b in range(n_ht):
            z = int(z_tot[b]) % p
            r = int(challenges[n_ht - 1 - b]) % p
            out = out * (((1 - z) % p) * ((1 - r) % p) + z * r) % p
        return out
    if label == "bucket-m":
        acc = 0
        h_u = u[ls + lt:]
        h_v = v[ls + lt:]
        for h in range(hp):
            m = (int(statement.m_nums[h])
                 if h < statement.head_count else 0)
            acc = (acc + m * _axis_weight(h, h_u, h_v)) % p
        rest = 1
        for w in ones[:ls + lt]:
            rest = rest * w % p
        return acc * rest % p
    raise ProofV3Error(f"no closed form for public fold '{label}'")
