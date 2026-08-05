"""Exact vectorized Goldilocks arithmetic on uint64 numpy arrays.

Pure optimization for verifier hot loops (table MLE folds): identical
results to the big-int reference, just C-speed. The 128-bit products
are assembled from 32-bit limb products; the reduction uses
2^64 == 2^32 - 1 (mod p) with explicit borrow/carry corrections.
"""

from __future__ import annotations

import numpy as np

P = np.uint64(0xFFFFFFFF00000001)
_MASK32 = np.uint64(0xFFFFFFFF)
_EPS = np.uint64(0xFFFFFFFF)  # 2^32 - 1


def _mul_wide(a: np.ndarray, b: np.ndarray):
    """(hi, lo) of the exact 128-bit product, elementwise."""

    al = a & _MASK32
    ah = a >> np.uint64(32)
    bl = b & _MASK32
    bh = b >> np.uint64(32)
    ll = al * bl
    lh = al * bh
    hl = ah * bl
    hh = ah * bh
    mid = (ll >> np.uint64(32)) + (lh & _MASK32) + (hl & _MASK32)
    lo = (ll & _MASK32) | ((mid & _MASK32) << np.uint64(32))
    hi = hh + (lh >> np.uint64(32)) + (hl >> np.uint64(32)) + (
        mid >> np.uint64(32))
    return hi, lo


def _reduce(hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """Canonical (hi * 2^64 + lo) mod p."""

    hi_hi = hi >> np.uint64(32)
    hi_lo = hi & _MASK32
    # t = lo - hi_hi  (mod p), with borrow correction
    borrow = lo < hi_hi
    t = lo - hi_hi
    t[borrow] += P  # lo < hi_hi < 2^32 <= p so t+P stays < 2^64
    # u = hi_lo * (2^32 - 1) < 2^64 exactly
    u = hi_lo * _EPS
    # result = t + u (mod p) with carry correction
    r = t + u
    carry = r < t
    # 2^64 mod p == 2^32 - 1
    r[carry] += _EPS
    r[r >= P] -= P
    return r


def gl_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    hi, lo = _mul_wide(a, b)
    return _reduce(hi, lo)


def gl_sub_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    borrow = a < b
    r = a - b
    r[borrow] += P
    return r


def gl_add_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = a + b
    carry = r < a
    r[carry] += _EPS
    r[r >= P] -= P
    return r


def mle_eval_msb_np(values, point) -> int:
    """MSB-first multilinear fold, byte-identical to _mle_eval_msb."""

    work = np.array(values, dtype=np.uint64)
    for challenge in point:
        half = work.shape[0] // 2
        c = np.uint64(challenge)
        lo = work[:half]
        hi = work[half:]
        diff = gl_sub_np(hi, lo)
        work = gl_add_np(lo, gl_mul_np(
            diff, np.broadcast_to(c, diff.shape).copy()))
    return int(work[0])


def mle_eval_lsb_np(values, point) -> int:
    """LSB-first multilinear fold, byte-identical to the scalar reference."""

    work = np.asarray(values, dtype=np.uint64)
    for challenge in point:
        lo = work[0::2]
        hi = work[1::2]
        diff = gl_sub_np(hi, lo)
        work = gl_add_np(
            lo,
            gl_mul_np(diff, np.uint64(challenge)),
        )
    return int(work[0])


def gl_sum_np(a: np.ndarray) -> int:
    """Exact field sum of canonical u64 values (hi/lo split avoids u64
    accumulator overflow)."""

    lo = int(np.sum(a & np.uint64(0xFFFFFFFF), dtype=np.uint64))
    hi = int(np.sum(a >> np.uint64(32), dtype=np.uint64))
    return (lo + (hi << 32)) % ((1 << 64) - (1 << 32) + 1)


def public_fold_round_np(work_v: np.ndarray, work_f: np.ndarray):
    """One public-fold prover round, byte-identical to the python loop:
    returns (evals4, lo_v, hi_v, diff_v, lo_f, hi_f, diff_f)."""

    half = work_v.shape[0] // 2
    v_lo, v_hi = work_v[:half], work_v[half:]
    f_lo, f_hi = work_f[:half], work_f[half:]
    dv = gl_sub_np(v_hi, v_lo)
    df = gl_sub_np(f_hi, f_lo)
    evals = []
    vv = v_lo.copy()
    ff = f_lo.copy()
    for z in range(4):
        if z:
            vv = gl_add_np(vv, dv)
            ff = gl_add_np(ff, df)
        evals.append(gl_sum_np(gl_mul_np(vv, ff)))
    return tuple(evals), v_lo, v_hi, dv, f_lo, f_hi, df
