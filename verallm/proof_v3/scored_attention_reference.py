"""Scored-attention reference: integer-exact runtime-output binding.

The calibration study (fb13b99) proved the uniform product-domain exp table
UNSOUND for runtime binding (bucket width 2*sqrt(D)*max|q|*max|k|/2^sb nats,
independent of qk_bits).  This module is the replacement semantics, and every
quantity is a deterministic integer function of committed captures + the
SIGNED profile -- no prover-chosen scale, offset or float anywhere:

* scales: fixed signed calibration.  Per-head q~ scale, per-dim k scales
  (folded into q~ BEFORE quantization, single slope preserved), per-head v
  and ox scales -- all fixed-point rationals derived bit-exactly from the
  signed profile floats (20-bit mantissa, deterministic normalization).
* score oracle: s16 = clamp(round_shift(P*M_h, E_h) - peak_units, -65535, 0)
  with P the exact int13 q~.k product, M_h/2^E_h the exact rational slope in
  LAMBDA units (lambda = 21/2^16 nats).  Causal mask zeroes es.  Per-row
  PEAK CLAIM (peak_units, peak_key): constraint (a) every visible score
  <= peak_units (lying LOW fails at the true max), constraint (b) the score
  at peak_key EQUALS peak_units (lying HIGH fails exactly) -- no offset
  freedom remains.
* softmax: es = FIXED_TABLE[-s16] (universal table over [-21,0], 2^16
  entries, never on the wire), totals = sum es, probs by ROUNDED division
  floor((es*2^17 + total)/(2*total)) (floor-div mass deficit at long context
  measured 23%), out = sum probs*v8.
* bridge: the surrogate output is bound to the RUNTIME o_proj input (ox8,
  quantized from the tracker capture with the signed ox scale) through the
  all-integer residual
      r = surrogate*vs_num*2^(ox_e) - ox8*ox_num*2^(vs_e)*2^SCALE_BITS
  checked against BOTH a per-cell cap and a per-row sum-of-squares cap from
  the signed profile (a global norm alone would let a miner hide a large
  localized deviation).

RATIONAL scheme (V2, emit/verify_*_rational_v3): identical scores/es, but
the claim is the exact integer pair (total = sum es, numerator = es @ v8)
and the output binds through the cross-multiplied residual
    r = num*vs_num*2^(ox_e) - ox8*ox_num*2^(vs_e)*total
with the SAME signed bounds scaled by the total (|r|*2^SB <= cell*total,
sum r^2 * 2^(2*SB) <= row_sq*total^2).  No per-key probability rounding
(the V1 rounded division loses 15-25% of diffuse mass on flat heads at
SCALE_BITS=16), no probs column and no division constraint in the
succinct statement; chunk claims verify locally without the row total.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError

__all__ = [
    "SCORE_MIN", "LAMBDA_RANGE_NATS", "SCALE_BITS", "SCORED_SCHEME_V1",
    "SCORED_SCHEME_RATIONAL_V2",
    "ScoredHeadParamsV3", "ScoredBridgeBoundsV3",
    "fixed_point_from_scale_v3", "fixed_exp_table_v3",
    "score_from_product_v3", "emit_scored_head_v3",
    "verify_scored_rows_v3", "verify_output_bridge_v3",
    "emit_scored_head_rational_v3", "verify_scored_rows_rational_v3",
    "verify_output_bridge_rational_v3",
    "scored_calibration_digest_v3",
]

SCORED_SCHEME_V1: Final = 1

# Rational-accumulator scored scheme: the per-head claim is the exact
# integer pair (total = sum es, numerator[d] = sum es * v8[d]) and the
# output binds through a CROSS-MULTIPLIED bridge -- no per-key
# probability rounding (the SCALE_BITS=16 rounded division loses
# 15-25% of diffuse mass on flat heads) and no probs column/division
# constraint in the succinct statement.
SCORED_SCHEME_RATIONAL_V2: Final = 2


def scored_calibration_digest_v3(layers, *, discriminative=()) -> bytes:
    """Canonical sha256 of the full scored calibration -- the value SIGNED
    into the manifest (``attn_calibration_digest``).

    ``layers[layer] = tuple of (ScoredHeadParamsV3, ScoredBridgeBoundsV3)``
    per head.  ``discriminative`` is the signed set of layers the nonce may
    sample for the hard attention audit (blind local-attention layers, where
    deviation ~ harm ~ 0, are excluded -- cont10 calibration).  Everything is
    little-endian integers, so two verifiers derive the identical digest from
    the identical calibration blob; the blob ships beside the manifest and
    any mismatch fails closed before use."""
    import hashlib
    import struct

    h = hashlib.sha256(b"VERATHOS/PROOF_V3/SCORED_ATTN_CALIBRATION/V1")
    disc = tuple(sorted(int(x) for x in discriminative))
    h.update(struct.pack("<I", len(disc)))
    for layer in disc:
        h.update(struct.pack("<I", layer))
    h.update(struct.pack("<I", len(layers)))
    for layer in sorted(layers):
        heads = layers[layer]
        h.update(struct.pack("<II", int(layer), len(heads)))
        for params, bounds in heads:
            h.update(struct.pack(
                "<IIqqqqqqqq", params.head_dim, params.qk_qmax,
                params.q_num, params.q_e, params.m_num, params.m_e,
                params.v_num, params.v_e, params.ox_num, params.ox_e))
            h.update(struct.pack("<I", len(params.k_num)))
            for kn, ke in zip(params.k_num, params.k_e):
                h.update(struct.pack("<qq", kn, ke))
            h.update(int(bounds.cell_bound).to_bytes(32, "little"))
            h.update(int(bounds.row_sq_bound).to_bytes(32, "little"))
            if getattr(bounds, "gated", False):
                # domain-separate GATED bridge bounds (their units carry
                # the 2^GATE_FIXED_BITS sigmoid factor); ungated blobs
                # keep their exact historical digest bytes
                h.update(b"GATED-BRIDGE/V1")
    return h.digest()

SCORE_MIN: Final = -65535
LAMBDA_RANGE_NATS: Final = 21.0
SCALE_BITS: Final = 16
_MANTISSA_BITS: Final = 20


def fixed_point_from_scale_v3(value: float) -> tuple[int, int]:
    """Deterministic (mantissa, exponent): value ~= num / 2^e, num in
    [2^19, 2^20).  Uses the float's exact binary fraction -- two verifiers
    always derive identical integers from the same signed profile bytes."""
    if not (value > 0.0) or not math.isfinite(value):
        raise ProofV3Error("signed scale must be a positive finite float")
    frac, exp2 = math.frexp(value)          # value = frac * 2^exp2, frac in [0.5,1)
    num = int(round(frac * (1 << _MANTISSA_BITS)))
    e = _MANTISSA_BITS - exp2
    if num == 1 << _MANTISSA_BITS:          # frac rounded up to 1.0
        num >>= 1
        e -= 1
    return num, e


@lru_cache(maxsize=1)
def fixed_exp_table_v3() -> tuple[int, ...]:
    """Universal exp table over [-21, 0] nats: TABLE[i] =
    max(1, round(2^22 * exp(-i * LAMBDA))) for i in [0, 65535].
    A constant of the protocol -- never per-request, never on the wire."""
    lam = LAMBDA_RANGE_NATS / 65536.0
    max_exp = 1 << 22
    return tuple(
        max(1, int(round(max_exp * math.exp(-i * lam))))
        for i in range(65536)
    )


@dataclass(frozen=True, slots=True)
class ScoredHeadParamsV3:
    """One head's signed-profile calibration (all integers, all fixed).

    ``k_num/k_e`` are the per-dim k scales; q~ = q * k_scale_d is quantized
    with ``q_num/q_e``.  ``m_num/m_e`` is the score slope in LAMBDA units per
    raw q13.k13 product unit: qs / (sqrt(head_dim) * LAMBDA), fixed-point.
    All derive deterministically from the signed profile via
    fixed_point_from_scale_v3."""

    head_dim: int
    qk_qmax: int                     # 4095 = 13-bit signed
    q_num: int
    q_e: int
    k_num: tuple[int, ...]           # per-dim mantissas
    k_e: tuple[int, ...]
    m_num: int
    m_e: int
    v_num: int
    v_e: int
    ox_num: int
    ox_e: int

    @classmethod
    def from_signed_scales(cls, *, head_dim: int, q_scale: float,
                           k_scales, v_scale: float, ox_scale: float,
                           qk_bits: int = 13) -> "ScoredHeadParamsV3":
        if len(k_scales) != head_dim:
            raise ProofV3Error("per-dim k scales must cover head_dim")
        qn, qe = fixed_point_from_scale_v3(q_scale)
        kn, ke = zip(*(fixed_point_from_scale_v3(s) for s in k_scales))
        vn, ve = fixed_point_from_scale_v3(v_scale)
        on, oe = fixed_point_from_scale_v3(ox_scale)
        lam = LAMBDA_RANGE_NATS / 65536.0
        mn, me = fixed_point_from_scale_v3(
            q_scale / (math.sqrt(head_dim) * lam))
        return cls(head_dim=head_dim, qk_qmax=(1 << (qk_bits - 1)) - 1,
                   q_num=qn, q_e=qe, k_num=tuple(kn), k_e=tuple(ke),
                   m_num=mn, m_e=me, v_num=vn, v_e=ve, ox_num=on, ox_e=oe)


def score_from_product_v3(product: int, params: ScoredHeadParamsV3) -> int:
    """Deterministic score units (BEFORE peak shift): floor((P*M + 2^(E-1)) / 2^E).

    Unconditional floor-shift (python ``>>`` floors negatives too) so ONE
    sign-free Euclidean identity holds everywhere -- the succinct tile proves
    ``P*M + 2^(E-1) == su*2^E + rem_b`` with ``rem_b in [0, 2^E)``."""
    p = int(product) * params.m_num
    return (p + (1 << (params.m_e - 1))) >> params.m_e


def _clamped(score_units: int, peak_units: int) -> int:
    s = score_units - peak_units
    return SCORE_MIN if s < SCORE_MIN else s     # s > 0 is a VERIFY error


def emit_scored_head_v3(*, params: ScoredHeadParamsV3, q13, k13, v8,
                        positions, key_count: int, chunk_size: int):
    """Miner-side claims for ONE head: exact integer pipeline.

    ``q13[row][d]``/``k13[key][d]`` int13, ``v8[key][d]`` int8, rows over the
    committed ``positions`` pool.  Returns
    (chunk_totals, partial_out, peaks) with peaks[row] = (peak_units,
    peak_key) -- the per-row peak claims the verifier enforces."""
    rows = len(positions)
    n_chunks = (key_count + chunk_size - 1) // chunk_size
    peaks = []
    scores = []                      # [row][key] clamped s16 (visible only)
    for r, pos in enumerate(positions):
        best, best_key = None, -1
        row_scores = {}
        for key in range(pos + 1):   # causal: visible keys only
            p = sum(int(q13[r][d]) * int(k13[key][d])
                    for d in range(params.head_dim))
            s = score_from_product_v3(p, params)
            row_scores[key] = s
            if best is None or s > best:
                best, best_key = s, key
        peaks.append((best, best_key))
        scores.append({k: _clamped(s, best) for k, s in row_scores.items()})
    table = fixed_exp_table_v3()
    es = [{k: table[-s] for k, s in row.items()} for row in scores]
    totals = [sum(row.values()) for row in es]
    chunk_totals = [
        [sum(es[r].get(k, 0)
             for k in range(c * chunk_size,
                            min((c + 1) * chunk_size, key_count)))
         for r in range(rows)]
        for c in range(n_chunks)
    ]
    partial_out = []
    for c in range(n_chunks):
        chunk_rows = []
        for r in range(rows):
            out = [0] * params.head_dim
            for k in range(c * chunk_size,
                           min((c + 1) * chunk_size, key_count)):
                e = es[r].get(k)
                if not e:
                    continue
                prob = (e * (2 << SCALE_BITS) + totals[r]) // (2 * totals[r])
                for d in range(params.head_dim):
                    out[d] += prob * int(v8[k][d])
            chunk_rows.append(tuple(out))
        partial_out.append(tuple(chunk_rows))
    return tuple(tuple(ct) for ct in chunk_totals), tuple(partial_out), tuple(peaks)


def verify_scored_rows_v3(*, params: ScoredHeadParamsV3, q13_rows, k13_chunk,
                          v8_chunk, chunk_index: int, chunk_size: int,
                          positions, peaks, totals, claimed_chunk_totals,
                          claimed_partials) -> None:
    """Validator-side exact recompute of one sampled chunk for sampled rows.

    Enforces: score <= peak for every visible key (lying-LOW peak fails when
    the true-max chunk is sampled); the peak-key equality is enforced by
    verify_peak_equality_v3 on the force-opened peak chunk.  All integers."""
    base = chunk_index * chunk_size
    table = fixed_exp_table_v3()
    for r, pos in enumerate(positions):
        peak_units, _peak_key = peaks[r]
        total = int(totals[r])
        if total < 1:
            raise ProofV3VerificationError("scored total must be positive")
        ct = 0
        out = [0] * params.head_dim
        for j, key in enumerate(range(base, base + len(k13_chunk))):
            if key > pos:
                continue
            p = sum(int(q13_rows[r][d]) * int(k13_chunk[j][d])
                    for d in range(params.head_dim))
            s = score_from_product_v3(p, params)
            if s > peak_units:
                raise ProofV3VerificationError(
                    "scored key exceeds the claimed row peak")
            e = table[-_clamped(s, peak_units)]
            ct += e
            prob = (e * (2 << SCALE_BITS) + total) // (2 * total)
            for d in range(params.head_dim):
                out[d] += prob * int(v8_chunk[j][d])
        if ct != int(claimed_chunk_totals[r]):
            raise ProofV3VerificationError(
                "scored chunk total does not match the claim")
        if tuple(out) != tuple(int(x) for x in claimed_partials[r]):
            raise ProofV3VerificationError(
                "scored chunk partial does not match the claim")


def verify_peak_equality_v3(*, params: ScoredHeadParamsV3, q13_row, k13_peak,
                            pos: int, peak_units: int, peak_key: int) -> None:
    """The claimed peak must EQUAL the score at the claimed peak key (which
    must be visible).  Removes the lying-HIGH freedom exactly."""
    if not 0 <= peak_key <= pos:
        raise ProofV3VerificationError("peak key is not causally visible")
    p = sum(int(q13_row[d]) * int(k13_peak[d]) for d in range(params.head_dim))
    if score_from_product_v3(p, params) != int(peak_units):
        raise ProofV3VerificationError(
            "claimed row peak does not equal the score at the peak key")


def emit_scored_head_rational_v3(*, params: ScoredHeadParamsV3, q13, k13,
                                 v8, positions, key_count: int,
                                 chunk_size: int):
    """Miner-side RATIONAL claims for ONE head (scheme V2): the exact
    integer pipeline of emit_scored_head_v3 up to es, then per chunk
    the pair (chunk_total = sum es, chunk_numerator[d] = sum es*v8[d])
    with NO probability rounding anywhere.  Row output is
    (sum_c chunk_numerator) / (sum_c chunk_total) -- the division is
    deferred to the cross-multiplied output bridge.

    Returns (chunk_totals, chunk_numerators, peaks); chunk_totals and
    peaks are IDENTICAL to emit_scored_head_v3's (same scores, same
    es), so the peak constraints and total accounting carry over
    unchanged.  Limb budget: es <= 2^22, |v8| <= 127, so a row
    numerator over T keys is < T * 2^29 (~2^49 at T = 1M) and a row
    total < T * 2^22 (~2^42) -- the succinct tile carries these at the
    final bridge only."""

    rows = len(positions)
    n_chunks = (key_count + chunk_size - 1) // chunk_size
    peaks = []
    scores = []                      # [row][key] clamped s16 (visible only)
    for r, pos in enumerate(positions):
        best, best_key = None, -1
        row_scores = {}
        for key in range(pos + 1):   # causal: visible keys only
            p = sum(int(q13[r][d]) * int(k13[key][d])
                    for d in range(params.head_dim))
            s = score_from_product_v3(p, params)
            row_scores[key] = s
            if best is None or s > best:
                best, best_key = s, key
        peaks.append((best, best_key))
        scores.append({k: _clamped(s, best) for k, s in row_scores.items()})
    table = fixed_exp_table_v3()
    es = [{k: table[-s] for k, s in row.items()} for row in scores]
    chunk_totals = []
    chunk_numerators = []
    for c in range(n_chunks):
        totals_row = []
        num_row = []
        for r in range(rows):
            ct = 0
            num = [0] * params.head_dim
            for k in range(c * chunk_size,
                           min((c + 1) * chunk_size, key_count)):
                e = es[r].get(k)
                if not e:
                    continue
                ct += e
                for d in range(params.head_dim):
                    num[d] += e * int(v8[k][d])
            totals_row.append(ct)
            num_row.append(tuple(num))
        chunk_totals.append(tuple(totals_row))
        chunk_numerators.append(tuple(num_row))
    return tuple(chunk_totals), tuple(chunk_numerators), tuple(peaks)


def verify_scored_rows_rational_v3(*, params: ScoredHeadParamsV3, q13_rows,
                                   k13_chunk, v8_chunk, chunk_index: int,
                                   chunk_size: int, positions, peaks,
                                   claimed_chunk_totals,
                                   claimed_chunk_numerators) -> None:
    """Validator-side exact recompute of one sampled chunk for sampled
    rows under the RATIONAL scheme.  Unlike the prob path, the chunk
    claim is LOCAL: no row total enters (the total only appears at the
    output bridge), so a sampled chunk verifies from the chunk's own
    K/V plus the row peaks alone.  Enforces score <= peak for every
    visible key exactly as verify_scored_rows_v3 (peak-key equality
    stays with verify_peak_equality_v3)."""

    base = chunk_index * chunk_size
    table = fixed_exp_table_v3()
    for r, pos in enumerate(positions):
        peak_units, _peak_key = peaks[r]
        ct = 0
        num = [0] * params.head_dim
        for j, key in enumerate(range(base, base + len(k13_chunk))):
            if key > pos:
                continue
            p = sum(int(q13_rows[r][d]) * int(k13_chunk[j][d])
                    for d in range(params.head_dim))
            s = score_from_product_v3(p, params)
            if s > peak_units:
                raise ProofV3VerificationError(
                    "scored key exceeds the claimed row peak")
            e = table[-_clamped(s, peak_units)]
            ct += e
            for d in range(params.head_dim):
                num[d] += e * int(v8_chunk[j][d])
        if ct != int(claimed_chunk_totals[r]):
            raise ProofV3VerificationError(
                "scored chunk total does not match the claim")
        if tuple(num) != tuple(
                int(x) for x in claimed_chunk_numerators[r]):
            raise ProofV3VerificationError(
                "scored chunk numerator does not match the claim")


@dataclass(frozen=True, slots=True)
class ScoredBridgeBoundsV3:
    """Signed-profile bridge bounds (integers, in residual units).

    ``gated``: the bounds were measured under the GATED bridge relation
    (sigmoid(attention-output-gate) fixed-point factor).  Gated bounds
    are ~2^GATE_FIXED_BITS larger than ungated residual units, so a
    verifier that dropped the gate factor would accept anything --
    the flag rides the SIGNED calibration digest and the adapter
    refuses a gate-oracle mismatch in either direction."""

    cell_bound: int          # max |residual| for any single cell
    row_sq_bound: int        # max sum of squared residuals per row
    gated: bool = False


def verify_output_bridge_v3(*, params: ScoredHeadParamsV3, surrogate_rows,
                            ox8_rows, bounds: ScoredBridgeBoundsV3) -> None:
    """All-integer runtime-output binding: the score-derived surrogate must
    match the RUNTIME o_proj capture within the signed per-cell AND per-row
    bounds.  residual = surr*vs_num*2^ox_e - ox8*ox_num*2^vs_e*2^SCALE_BITS,
    normalized to the common denominator 2^(vs_e+ox_e+SCALE_BITS); bounds are
    calibrated in the SAME units.  No floats anywhere."""
    for r, (srow, oxrow) in enumerate(zip(surrogate_rows, ox8_rows)):
        acc = 0
        for d in range(params.head_dim):
            res = (int(srow[d]) * params.v_num * (1 << params.ox_e)
                   - int(oxrow[d]) * params.ox_num
                   * (1 << (params.v_e + SCALE_BITS)))
            if abs(res) > bounds.cell_bound:
                raise ProofV3VerificationError(
                    "output bridge cell residual exceeds the signed bound")
            acc += res * res
        if acc > bounds.row_sq_bound:
            raise ProofV3VerificationError(
                "output bridge row residual exceeds the signed bound")


# fixed-point precision of the sigmoid(attention-output-gate) factor in
# the gated bridge relation (Qwen3-Next-family gated attention)
GATE_FIXED_BITS: Final = 16


def verify_output_bridge_rational_v3(*, params: ScoredHeadParamsV3,
                                     numerator_rows, total_rows, ox8_rows,
                                     bounds: ScoredBridgeBoundsV3,
                                     gate_fx_rows=None) -> None:
    """All-integer CROSS-MULTIPLIED runtime-output binding (scheme V2).

    The rational surrogate is numerator/total; instead of dividing,
    the residual is cross-multiplied by the row total:

        res = num[d]*vs_num*2^ox_e - ox8[d]*ox_num*2^vs_e*total

    which equals (total / 2^SCALE_BITS) times the prob-path residual
    evaluated at the UNROUNDED surrogate num*2^SCALE_BITS/total.  The
    SAME signed ScoredBridgeBoundsV3 therefore transfers by scaling
    both checks with the total:

        |res| * 2^SCALE_BITS      <= cell_bound   * total
        sum(res^2) * 2^(2*SB)     <= row_sq_bound * total^2

    all integers (python bigint; the succinct tile carries the limbs
    at this final bridge only).  Bounds calibrated for the prob-path
    bridge are CONSERVATIVE here: the rational residual lacks the
    per-key rounding term, so honest residuals only shrink.

    ``gate_fx_rows``: gated-attention models (Qwen3-Next family) apply
    o_input = attn_out * sigmoid(gate); pass the AUTHENTICATED per-cell
    gate factors as round(sigmoid(g) * 2^GATE_FIXED_BITS) and the
    residual becomes

        res = num[d]*vs_num*2^ox_e*g_fx[d]
              - ox8[d]*ox_num*2^vs_e*2^GATE_FIXED_BITS*total

    with bounds measured under the SAME gated formula.  ``None`` keeps
    the ungated relation byte-identical."""

    two_sb = 1 << SCALE_BITS
    for r, (nrow, oxrow) in enumerate(zip(numerator_rows, ox8_rows)):
        total = int(total_rows[r])
        if total < 1:
            raise ProofV3VerificationError("scored total must be positive")
        gate_row = None if gate_fx_rows is None else gate_fx_rows[r]
        acc = 0
        for d in range(params.head_dim):
            if gate_row is None:
                res = (int(nrow[d]) * params.v_num * (1 << params.ox_e)
                       - int(oxrow[d]) * params.ox_num
                       * (1 << params.v_e) * total)
            else:
                res = (int(nrow[d]) * params.v_num * (1 << params.ox_e)
                       * int(gate_row[d])
                       - int(oxrow[d]) * params.ox_num
                       * (1 << params.v_e)
                       * (1 << GATE_FIXED_BITS) * total)
            cell_lhs = abs(res) * two_sb
            cell_rhs = bounds.cell_bound * total
            if cell_lhs > cell_rhs:
                utilization_ppm = (
                    cell_lhs * 1_000_000 + cell_rhs - 1
                ) // cell_rhs
                zero_residual = (
                    abs(int(oxrow[d]))
                    * params.ox_num
                    * (1 << params.v_e)
                    * total
                )
                if gate_row is not None:
                    zero_residual *= 1 << GATE_FIXED_BITS
                residual_vs_zero_ppm = (
                    cell_lhs * 1_000_000
                    // max(1, zero_residual * two_sb)
                )
                raise ProofV3VerificationError(
                    "output bridge cell residual exceeds the signed bound "
                    f"(row={r}, column={d}, "
                    f"utilization_ppm={utilization_ppm}, "
                    f"residual_vs_zero_ppm={residual_vs_zero_ppm})"
                )
            acc += res * res
        row_lhs = acc * two_sb * two_sb
        row_rhs = bounds.row_sq_bound * total * total
        if row_lhs > row_rhs:
            utilization_ppm = (
                row_lhs * 1_000_000 + row_rhs - 1
            ) // row_rhs
            raise ProofV3VerificationError(
                "output bridge row residual exceeds the signed bound "
                f"(row={r}, utilization_ppm={utilization_ppm})"
            )
