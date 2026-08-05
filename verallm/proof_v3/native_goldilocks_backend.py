"""Torch-vectorized native Goldilocks backend for proof-v3 (CPU + CUDA).

First native-backend increment: the prover-side hot loops of the shipped
reference protocol, produced **byte-identical** to the pure-Python
reference so every conformance test can compare proofs directly.

Implemented natively:

* exact Goldilocks vector arithmetic on ``torch.int64`` bit patterns —
  multiplication via 16-bit limb decomposition (no 64-bit wrap-around
  assumptions, portable across CPU and CUDA), addition/subtraction with
  epsilon correction, batched inversion by a vectorized square-and-multiply
  power chain (``x^(p-2)``, 63 squarings over the whole tensor);
* the fold-sumcheck prover (single committed table x public factor) and
  the two-table product-sumcheck prover: all round polynomials computed as
  tensor folds, matching the reference transcript byte for byte;
* batched LogUp rational-sum evaluation (witness and table sides).

Merkle trees remain on the SHA-256 reference path: they must stay
byte-identical to validator-side recomputation, and hashing is not the
asymptotic bottleneck the field loops are.  The CUDA-kernel step beyond
torch (fused NTT, on-GPU hashing) builds on the A40 spike kernels and is
benchmarked separately; this module is the portable correctness bridge
between the reference and that backend.

Import of ``torch`` is deferred so the reference stack keeps working in
torch-free environments.
"""

from __future__ import annotations

import hashlib
from typing import Final

from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.goldilocks_fold_sumcheck_reference import (
    GoldilocksFoldSumcheckProofV3,
    _challenge,
    _transcript_seed,
    factor_digest_v3,
)
from verallm.proof_v3.goldilocks_merkle_reference import (
    GoldilocksMerkleTreeReference,
)
from verallm.proof_v3.goldilocks_product_sumcheck_reference import (
    GoldilocksProductSumcheckProofV3,
    _seed as _product_seed,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS

_P: Final = GOLDILOCKS_MODULUS
_EPS: Final = (1 << 32) - 1
_MASK16: Final = 0xFFFF
_MASK32: Final = 0xFFFFFFFF
_TWO63: Final = 1 << 63
_TWO64: Final = 1 << 64


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ProofV3Error("native backend requires torch") from exc
    return torch


def to_field_tensor(values, device: str = "cpu"):
    """Encode canonical field ints as int64 bit patterns."""

    torch = _torch()
    try:
        import numpy as np

        # uint64 -> int64 bit reinterpretation == value - 2**64 for the
        # upper half: byte-identical to the per-element encoding below.
        arr = np.asarray(values, dtype=np.uint64).view(np.int64)
        return torch.from_numpy(arr).to(device)
    except (ImportError, OverflowError, TypeError, ValueError):
        pass
    encoded = [
        value - _TWO64 if value >= _TWO63 else value for value in values
    ]
    return torch.tensor(encoded, dtype=torch.int64, device=device)


def from_field_tensor(tensor) -> tuple[int, ...]:
    return tuple(
        value + _TWO64 if value < 0 else value for value in tensor.tolist()
    )


def _limbs16(x):
    """Split int64 bit patterns into four unsigned 16-bit limb tensors."""

    l0 = x & _MASK16
    l1 = (x >> 16) & _MASK16
    l2 = (x >> 32) & _MASK16
    l3 = (x >> 48) & _MASK16
    return l0, l1, l2, l3


_EW_EXT = None


def _ew_ext():
    """Fused elementwise-modmul kernel, if the CUDA extension is loaded."""

    global _EW_EXT
    if _EW_EXT is None:
        try:
            from verallm.proof_v3.native_cuda_fold_backend import (
                load_fused_kernels,
            )

            ext = load_fused_kernels()
            _EW_EXT = ext if hasattr(ext, "gl_mul_ew") else False
        except Exception:
            _EW_EXT = False
    return _EW_EXT


def gl_mul_t(a, b):
    """Exact Goldilocks multiplication on int64 bit-pattern tensors."""

    if getattr(a, "is_cuda", False):
        ext = _ew_ext()
        if ext:
            return ext.gl_mul_ew(a, b)
    a0, a1, a2, a3 = _limbs16(a)
    b0, b1, b2, b3 = _limbs16(b)
    # Diagonal sums d_k = sum_{i+j=k} a_i * b_j; each term < 2^32, each
    # diagonal < 2^34 — safely inside int64.
    d0 = a0 * b0
    d1 = a0 * b1 + a1 * b0
    d2 = a0 * b2 + a1 * b1 + a2 * b0
    d3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0
    d4 = a1 * b3 + a2 * b2 + a3 * b1
    d5 = a2 * b3 + a3 * b2
    d6 = a3 * b3
    # Carry-propagate the 16-bit diagonals into 32-bit words.
    s0 = d0
    s1 = d1 + (s0 >> 16)
    s2 = d2 + (s1 >> 16)
    s3 = d3 + (s2 >> 16)
    s4 = d4 + (s3 >> 16)
    s5 = d5 + (s4 >> 16)
    s6 = d6 + (s5 >> 16)
    s7 = s6 >> 16
    w0 = (s0 & _MASK16) | ((s1 & _MASK16) << 16)
    w1 = (s2 & _MASK16) | ((s3 & _MASK16) << 16)
    w2 = (s4 & _MASK16) | ((s5 & _MASK16) << 16)
    w3 = (s6 & _MASK16) | ((s7 & _MASK16) << 16)
    # 128-bit value = (w3:w2:w1:w0); reduce with the Goldilocks epsilon
    # identity 2^64 == 2^32 - 1 (mod p), all in nonnegative int64 pieces.
    # value = lo64 + 2^64 * hi64 where lo64 = (w1:w0), hi64 = (w3:w2).
    # 2^64*hi64 mod p = (2^32-1)*hi64 mod p = (hi64 << 32) - hi64 mod p,
    # and (hi_hi:hi_lo) splits keep every product below 2^63.
    lo_lo, lo_hi = w0, w1              # unsigned 32-bit words
    hi_lo, hi_hi = w2, w3
    torch = _torch()
    # t = lo64 - hi_hi (mod 2^64), minus EPS on 64-bit borrow, word-wise.
    lo_minus = lo_lo - hi_hi
    under = lo_minus < 0
    t_lo = torch.where(under, lo_minus + (1 << 32), lo_minus)
    t_hi = lo_hi - under.to(torch.int64)
    under2 = t_hi < 0
    t_hi = torch.where(under2, t_hi + (1 << 32), t_hi)
    # borrow out of 64 bits => subtract EPS once more (standard trick).
    t_lo2 = t_lo - under2.to(torch.int64) * _EPS
    under3 = t_lo2 < 0
    t_lo = torch.where(under3, t_lo2 + (1 << 32), t_lo2)
    t_hi = t_hi - under3.to(torch.int64)
    under4 = t_hi < 0
    t_hi = torch.where(under4, t_hi + (1 << 32), t_hi)
    # (No further borrow is possible: t >= 0 after at most one EPS fix.)
    # u = hi_lo * EPS, computed in 16-bit limbs (product may exceed 2^63).
    hl_lo = hi_lo & _MASK16
    hl_hi = (hi_lo >> 16) & _MASK16
    e_lo = _EPS & _MASK16
    e_hi = (_EPS >> 16) & _MASK16
    p0 = hl_lo * e_lo
    p1 = hl_lo * e_hi + hl_hi * e_lo
    p2 = hl_hi * e_hi
    q0 = p0
    q1 = p1 + (q0 >> 16)
    q2 = p2 + (q1 >> 16)
    q3 = q2 >> 16
    u_lo = (q0 & _MASK16) | ((q1 & _MASK16) << 16)
    u_hi = (q2 & _MASK16) | ((q3 & _MASK16) << 16)
    # res = t + u (mod 2^64), carry => add EPS.
    r_lo = t_lo + u_lo
    carry = (r_lo >> 32) & 1
    r_lo = r_lo & _MASK32
    r_hi = t_hi + u_hi + carry
    carry2 = (r_hi >> 32) & 1
    r_hi = r_hi & _MASK32
    r_lo = r_lo + carry2 * _EPS
    carry3 = (r_lo >> 32) & 1
    r_lo = r_lo & _MASK32
    r_hi = r_hi + carry3
    # (r_hi cannot re-overflow: r_hi <= 2^32 - 1 + 1.)
    over = r_hi >> 32
    r_hi = r_hi & _MASK32
    r_lo = r_lo + over * _EPS
    r_hi = r_hi + (r_lo >> 32)
    r_lo = r_lo & _MASK32
    # Final conditional subtract of p = 2^64 - 2^32 + 1:
    # value = r_hi * 2^32 + r_lo >= p  iff  r_hi == 2^32-1 and r_lo >= 1?
    # p in words: hi = 0xFFFFFFFF, lo = 0x00000001.
    ge = (r_hi > _MASK32) | (
        (r_hi == _MASK32) & (r_lo >= 1)
    )
    r_lo = torch.where(ge, r_lo - 1, r_lo)
    borrow_f = r_lo < 0
    r_lo = torch.where(borrow_f, r_lo + (1 << 32), r_lo)
    r_hi = torch.where(ge, r_hi - _MASK32 - borrow_f.to(torch.int64), r_hi)
    result = r_lo | (r_hi << 32)
    return result


_BIG_ELEMWISE_CHUNK = 1 << 24


def _chunked_elementwise(impl, a, b):
    """Bound the word-decomposition transients on huge CUDA tensors.

    The pure-torch add/sub bodies allocate ~15 intermediates of input
    size; on multi-GiB codewords that is a multi-GiB VRAM spike.  Same
    math per chunk, byte-identical result.
    """

    torch = _torch()
    out = torch.empty_like(a)
    flat_a = a.view(-1)
    flat_b = b.view(-1)
    flat_out = out.view(-1)
    for start in range(0, flat_a.numel(), _BIG_ELEMWISE_CHUNK):
        stop = start + _BIG_ELEMWISE_CHUNK
        flat_out[start:stop] = impl(
            flat_a[start:stop], flat_b[start:stop])
    return out


def _chunkable(a, b) -> bool:
    return (
        getattr(a, "is_cuda", False)
        and a.numel() > _BIG_ELEMWISE_CHUNK
        and a.shape == b.shape
        and a.is_contiguous()
        and b.is_contiguous()
    )


def gl_scale_t(a, value: int):
    """a * scalar (mod p) without a full-size constant tensor."""

    torch = _torch()
    encoded = value - (1 << 64) if value >= (1 << 63) else value
    if getattr(a, "is_cuda", False) and a.is_contiguous():
        ext = _ew_ext()
        if ext:
            scalar = torch.tensor(
                encoded, dtype=torch.int64, device=a.device)
            return ext.gl_mul_ew(a, scalar)
    return gl_mul_t(a, torch.full_like(a, encoded))


def _ew_native(op_name: str, a, b):
    """Native elementwise dispatch: one kernel instead of ~15 limb ops
    (and none of their full-size intermediates).  Same-shape CUDA
    tensors only; broadcast callers keep the limb path."""

    if not (getattr(a, "is_cuda", False) and getattr(b, "is_cuda", False)):
        return None
    if a.shape != b.shape:
        return None
    ext = _ew_ext()
    if not ext:
        return None
    op = getattr(ext, op_name, None)
    if op is None:
        return None
    return op(a, b)


def gl_add_t(a, b):
    native = _ew_native("gl_add_ew", a, b)
    if native is not None:
        return native
    if _chunkable(a, b):
        return _chunked_elementwise(_gl_add_impl, a, b)
    return _gl_add_impl(a, b)


def _gl_add_impl(a, b):
    torch = _torch()
    a_lo, a_hi = a & _MASK32, (a >> 32) & _MASK32
    b_lo, b_hi = b & _MASK32, (b >> 32) & _MASK32
    lo = a_lo + b_lo
    hi = a_hi + b_hi + (lo >> 32)
    lo = lo & _MASK32
    over = hi >> 32
    hi = hi & _MASK32
    lo = lo + over * _EPS
    hi = hi + (lo >> 32)
    lo = lo & _MASK32
    ge = (hi > _MASK32) | ((hi == _MASK32) & (lo >= 1))
    lo2 = torch.where(ge, lo - 1, lo)
    borrow = lo2 < 0
    lo = torch.where(borrow, lo2 + (1 << 32), lo2)
    hi = torch.where(ge, hi - _MASK32 - borrow.to(torch.int64), hi)
    return lo | (hi << 32)


def gl_sub_t(a, b):
    native = _ew_native("gl_sub_ew", a, b)
    if native is not None:
        return native
    if _chunkable(a, b):
        return _chunked_elementwise(_gl_sub_impl, a, b)
    return _gl_sub_impl(a, b)


def _gl_sub_impl(a, b):
    """d = a - b (mod 2^64); on borrow d -= EPS; canonicalise below p."""

    torch = _torch()
    a_lo, a_hi = a & _MASK32, (a >> 32) & _MASK32
    b_lo, b_hi = b & _MASK32, (b >> 32) & _MASK32
    lo = a_lo - b_lo
    borrow = (lo < 0).to(torch.int64)
    lo = lo + borrow * (1 << 32)
    hi = a_hi - b_hi - borrow
    under = (hi < 0).to(torch.int64)
    hi = hi + under * (1 << 32)
    # The 64-bit wrap already added 2^64 == p + EPS (mod p): remove EPS.
    lo = lo - under * _EPS
    borrow2 = (lo < 0).to(torch.int64)
    lo = lo + borrow2 * (1 << 32)
    hi = hi - borrow2
    under2 = (hi < 0).to(torch.int64)
    hi = hi + under2 * (1 << 32)
    # (a second 64-bit wrap here is plain mod-2^64 semantics, as in the
    # uint64 kernel; no further compensation.)
    ge = (hi == _MASK32) & (lo >= 1)
    lo2 = torch.where(ge, lo - 1, lo)
    borrow3 = (lo2 < 0).to(torch.int64)
    lo = lo2 + borrow3 * (1 << 32)
    hi = torch.where(ge, hi - _MASK32 - borrow3, hi)
    return lo | (hi << 32)


def gl_inv_t(x):
    """Batched inversion via the fixed power chain x^(p-2)."""

    exponent = _P - 2
    torch = _torch()
    result = torch.ones_like(x)
    base = x
    while exponent:
        if exponent & 1:
            result = gl_mul_t(result, base)
        exponent >>= 1
        if exponent:
            base = gl_mul_t(base, base)
    return result


def gl_sum_t(x) -> int:
    """Exact modular sum of a bit-pattern tensor (host-side reduction)."""

    lo = (x & _MASK32).sum().item()
    hi = ((x >> 32) & _MASK32).sum().item()
    return (lo + (hi << 32)) % _P


def _fold_t(values, challenge_tensor):
    half = values.shape[0] // 2
    low, high = values[:half], values[half:]
    return gl_add_t(low, gl_mul_t(challenge_tensor, gl_sub_t(high, low)))


def native_prove_fold_sumcheck_v3(
    *,
    statement_digest: bytes,
    x_tree: GoldilocksMerkleTreeReference,
    x_evaluations: tuple[int, ...],
    factor: tuple[int, ...],
    validator_nonce: bytes,
    device: str = "cpu",
) -> GoldilocksFoldSumcheckProofV3:
    """Byte-identical native fold-sumcheck prover (torch tensors)."""

    torch = _torch()
    if len(x_evaluations) != len(factor):
        raise ProofV3Error("native fold-sumcheck factor length mismatch")
    x = to_field_tensor(x_evaluations, device=device)
    f = to_field_tensor(factor, device=device)
    claimed = gl_sum_t(gl_mul_t(x, f))
    seed = _transcript_seed(
        statement_digest=statement_digest,
        x_commitment=x_tree.commitment,
        validator_nonce=validator_nonce,
        factor_digest=factor_digest_v3(tuple(factor)),
        claimed_sum=claimed,
    )
    transcript = seed
    rounds = []
    while x.shape[0] > 1:
        half = x.shape[0] // 2
        x_lo, x_hi = x[:half], x[half:]
        f_lo, f_hi = f[:half], f[half:]
        g0 = gl_sum_t(gl_mul_t(x_lo, f_lo))
        g1 = gl_sum_t(gl_mul_t(x_hi, f_hi))
        x2 = gl_sub_t(gl_add_t(x_hi, x_hi), x_lo)
        f2 = gl_sub_t(gl_add_t(f_hi, f_hi), f_lo)
        g2 = gl_sum_t(gl_mul_t(x2, f2))
        rounds.append((g0, g1, g2))
        transcript = hashlib.sha256(
            transcript
            + g0.to_bytes(8, "little")
            + g1.to_bytes(8, "little")
            + g2.to_bytes(8, "little")
        ).digest()
        challenge = _challenge(transcript, len(rounds))
        c = to_field_tensor([challenge] * half, device=device)
        x = _fold_t(x, c)
        f = _fold_t(f, c)
    return GoldilocksFoldSumcheckProofV3(
        claimed_sum=claimed,
        round_polynomials=tuple(rounds),
        x_full_opening=tuple(row[0] for row in x_tree.rows),
    )


def native_prove_product_sumcheck_v3(
    *,
    statement_digest: bytes,
    a_tree: GoldilocksMerkleTreeReference,
    b_tree: GoldilocksMerkleTreeReference,
    a_evaluations: tuple[int, ...],
    b_evaluations: tuple[int, ...],
    factor: tuple[int, ...],
    validator_nonce: bytes,
    device: str = "cpu",
) -> GoldilocksProductSumcheckProofV3:
    """Byte-identical native two-table product-sumcheck prover."""

    torch = _torch()
    if not len(a_evaluations) == len(b_evaluations) == len(factor):
        raise ProofV3Error("native product-sumcheck length mismatch")
    a = to_field_tensor(a_evaluations, device=device)
    b = to_field_tensor(b_evaluations, device=device)
    f = to_field_tensor(factor, device=device)
    claimed = gl_sum_t(gl_mul_t(gl_mul_t(a, b), f))
    transcript = _product_seed(
        statement_digest=statement_digest,
        a_commitment=a_tree.commitment,
        b_commitment=b_tree.commitment,
        validator_nonce=validator_nonce,
        factor=tuple(factor),
        claimed_sum=claimed,
    )
    rounds = []
    while a.shape[0] > 1:
        half = a.shape[0] // 2
        parts = [(t[:half], t[half:]) for t in (a, b, f)]
        evaluations = []
        for z in range(4):
            z_t = to_field_tensor([z] * half, device=device)
            term = None
            for low, high in parts:
                folded = gl_add_t(low, gl_mul_t(z_t, gl_sub_t(high, low)))
                term = folded if term is None else gl_mul_t(term, folded)
            evaluations.append(gl_sum_t(term))
        rounds.append(tuple(evaluations))
        transcript = hashlib.sha256(
            transcript
            + b"".join(value.to_bytes(8, "little") for value in evaluations)
        ).digest()
        challenge = _challenge(transcript, len(rounds))
        c = to_field_tensor([challenge] * half, device=device)
        a, b, f = (_fold_t(t, c) for t in (a, b, f))
    return GoldilocksProductSumcheckProofV3(
        claimed_sum=claimed,
        round_polynomials=tuple(rounds),
        a_full_opening=tuple(row[0] for row in a_tree.rows),
        b_full_opening=tuple(row[0] for row in b_tree.rows),
    )


def native_logup_rational_sum(
    values: tuple[int, ...],
    *,
    alpha: int,
    multiplicities: tuple[int, ...] | None = None,
    device: str = "cpu",
) -> int:
    """Batched ``sum m_i / (alpha - v_i)`` with vectorized inversion."""

    torch = _torch()
    v = to_field_tensor(values, device=device)
    alpha_t = to_field_tensor([alpha] * len(values), device=device)
    denominators = gl_sub_t(alpha_t, v)
    if bool((denominators == 0).any()):
        raise ProofV3Error("native logup challenge collides with a value")
    inverses = gl_inv_t(denominators)
    if multiplicities is not None:
        inverses = gl_mul_t(
            inverses, to_field_tensor(multiplicities, device=device)
        )
    return gl_sum_t(inverses)


__all__ = [
    "from_field_tensor",
    "gl_add_t",
    "gl_inv_t",
    "gl_mul_t",
    "gl_sub_t",
    "gl_sum_t",
    "native_logup_rational_sum",
    "native_prove_fold_sumcheck_v3",
    "native_prove_product_sumcheck_v3",
    "to_field_tensor",
]
