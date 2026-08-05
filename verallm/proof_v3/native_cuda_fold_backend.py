"""Fused-CUDA fold-sumcheck prover tier for proof-v3 (A40-class GPUs).

JIT-compiles ``gl_sumcheck_kernels.cu`` (one fused kernel per sumcheck
round: block-reduced g0/g1/g2 partials, plus a fused lerp fold) and runs
the byte-identical fold-sumcheck prover on it.

Measured on the A40 (2026-07-19): the complete round chain over a 2^24
table takes ~11 ms (1.5 Gelem/s effective) with proofs byte-identical to
the reference — over 100x the torch-limb tier and comfortably inside the
decode-idle budget for hard-audited requests.

Import and compilation are strictly opt-in (``load_fused_kernels()``)
because JIT-building CUDA extensions on arbitrary hosts is not safe to do
implicitly; tests gate on ``VERATHOS_NATIVE_CUDA_TEST=1``.
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
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS

_TWO63: Final = 1 << 63
_TWO64: Final = 1 << 64


def load_fused_kernels():
    """Load the precompiled or development-JIT fused CUDA extension."""

    try:
        from verathos_proof_v3_cuda import (
            load_fused_kernels as load_precompiled_fused_kernels,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "verathos_proof_v3_cuda":
            raise ProofV3Error(
                "precompiled proof-v3 CUDA fold runtime is incompatible"
            ) from exc
    except ImportError as exc:
        raise ProofV3Error(
            "precompiled proof-v3 CUDA fold runtime is incompatible"
        ) from exc
    else:
        try:
            return load_precompiled_fused_kernels()
        except (ImportError, OSError, RuntimeError) as exc:
            raise ProofV3Error(
                "precompiled proof-v3 CUDA fold runtime failed to load"
            ) from exc

    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ProofV3Error("fused CUDA tier requires torch") from exc
    if not torch.cuda.is_available():
        raise ProofV3Error("fused CUDA tier requires a CUDA device")
    import os

    source = os.path.join(os.path.dirname(__file__), "gl_sumcheck_kernels.cu")
    return load(
        name="gl_sumcheck_kernels",
        sources=[source],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def _sum_partials(partials) -> tuple[int, int, int]:
    values = [v + _TWO64 if v < 0 else v for v in partials.cpu().tolist()]
    sums = [0, 0, 0]
    for index, value in enumerate(values):
        sums[index % 3] = (sums[index % 3] + value) % GOLDILOCKS_MODULUS
    return sums[0], sums[1], sums[2]


def fused_prove_fold_sumcheck_v3(
    *,
    extension,
    statement_digest: bytes,
    x_tree: GoldilocksMerkleTreeReference,
    x_evaluations: tuple[int, ...],
    factor: tuple[int, ...],
    validator_nonce: bytes,
) -> GoldilocksFoldSumcheckProofV3:
    """Byte-identical fold-sumcheck prover on the fused kernel tier."""

    import torch

    from verallm.proof_v3.native_goldilocks_backend import (
        gl_mul_t,
        gl_sum_t,
        to_field_tensor,
    )

    if len(x_evaluations) != len(factor):
        raise ProofV3Error("fused fold-sumcheck factor length mismatch")
    x = to_field_tensor(x_evaluations, "cuda")
    f = to_field_tensor(factor, "cuda")
    claimed = gl_sum_t(gl_mul_t(x, f))
    transcript = _transcript_seed(
        statement_digest=statement_digest,
        x_commitment=x_tree.commitment,
        validator_nonce=validator_nonce,
        factor_digest=factor_digest_v3(tuple(factor)),
        claimed_sum=claimed,
    )
    rounds: list[tuple[int, int, int]] = []
    while x.shape[0] > 1:
        partials = extension.round_partials(x, f)
        torch.cuda.synchronize()
        g0, g1, g2 = _sum_partials(partials)
        rounds.append((g0, g1, g2))
        transcript = hashlib.sha256(
            transcript
            + g0.to_bytes(8, "little")
            + g1.to_bytes(8, "little")
            + g2.to_bytes(8, "little")
        ).digest()
        challenge = _challenge(transcript, len(rounds))
        encoded = challenge - _TWO64 if challenge >= _TWO63 else challenge
        x = extension.lerp_fold(x, encoded)
        f = extension.lerp_fold(f, encoded)
    return GoldilocksFoldSumcheckProofV3(
        claimed_sum=claimed,
        round_polynomials=tuple(rounds),
        x_full_opening=tuple(row[0] for row in x_tree.rows),
    )


__all__ = ["fused_prove_fold_sumcheck_v3", "load_fused_kernels"]


def fused_prove_product_sumcheck_v3(
    *,
    extension,
    statement_digest: bytes,
    a_tree,
    b_tree,
    a_evaluations: tuple[int, ...],
    b_evaluations: tuple[int, ...],
    factor: tuple[int, ...],
    validator_nonce: bytes,
):
    """Byte-identical two-table product-sumcheck prover, fused tier."""

    import torch

    from verallm.proof_v3.goldilocks_product_sumcheck_reference import (
        GoldilocksProductSumcheckProofV3,
        _seed as _product_seed,
    )
    from verallm.proof_v3.native_goldilocks_backend import (
        gl_mul_t,
        gl_sum_t,
        to_field_tensor,
    )

    a = to_field_tensor(a_evaluations, "cuda")
    b = to_field_tensor(b_evaluations, "cuda")
    f = to_field_tensor(factor, "cuda")
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
        partials = extension.product_round_partials(a, b, f)
        torch.cuda.synchronize()
        values = [v + _TWO64 if v < 0 else v for v in partials.cpu().tolist()]
        sums = [0, 0, 0, 0]
        for index, value in enumerate(values):
            sums[index % 4] = (sums[index % 4] + value) % GOLDILOCKS_MODULUS
        rounds.append(tuple(sums))
        transcript = hashlib.sha256(
            transcript
            + b"".join(value.to_bytes(8, "little") for value in sums)
        ).digest()
        challenge = _challenge(transcript, len(rounds))
        encoded = challenge - _TWO64 if challenge >= _TWO63 else challenge
        a = extension.lerp_fold(a, encoded)
        b = extension.lerp_fold(b, encoded)
        f = extension.lerp_fold(f, encoded)
    return GoldilocksProductSumcheckProofV3(
        claimed_sum=claimed,
        round_polynomials=tuple(rounds),
        a_full_opening=tuple(row[0] for row in a_tree.rows),
        b_full_opening=tuple(row[0] for row in b_tree.rows),
    )


__all__.append("fused_prove_product_sumcheck_v3")


_NTT_STAGE_CACHE: dict = {}
_NTT_COSET_CACHE: dict = {}
# tables above this size are computed per call and FREED: coset shifts
# are statement-derived (unique per request/column), so caching a
# size-n device tensor per shift is a resident-VRAM leak at long
# context (measured 250k: ~10GB of dead 2^24..2^27 tables)
import os as _os

_NTT_CACHE_MAX_N: int = int(_os.environ.get(
    "VERATHOS_NTT_CACHE_MAX_N", str(1 << 20)))
# STAGE tables depend only on (generator, n) -- statement-free and
# reused across every commit/opening of that size within a prove (13+
# rebuilds per layer at 250k) -- so they cache at a higher cap; the
# resident cost is one ladder-slice set per DISTINCT size (~2n * 8B).
# COSET tables stay at the small cap: shifts are statement-derived and
# never repeat.
_NTT_STAGE_CACHE_MAX_N: int = int(_os.environ.get(
    "VERATHOS_NTT_STAGE_CACHE_MAX_N", str(1 << 24)))


def _power_ladder(base, length):
    """base^j for j < length, log-doubled INTO one buffer (exact).

    The cat-based doubling transiently held ~2x the final table -- a
    full extra codeword at 2^27.  Writing each doubling into a
    preallocated buffer caps the transient at half the table (the
    elementwise-mul result before its slice assign).
    """

    import torch

    from verallm.proof_v3.native_goldilocks_backend import (
        gl_mul_t,
        to_field_tensor,
    )

    ladder = torch.empty(length, dtype=torch.int64, device="cuda")
    ladder[:1] = to_field_tensor((1,), "cuda")
    filled = 1
    step = base % GOLDILOCKS_MODULUS
    while filled < length:
        take = min(filled, length - filled)
        step_t = to_field_tensor((step,), "cuda")
        ladder[filled:filled + take] = gl_mul_t(
            ladder[:take], step_t.expand(take))
        filled += take
        step = step * step % GOLDILOCKS_MODULUS
    return ladder


class _LazyStageTables:
    """Above-cache-cap stage twiddles, materialized one stage at a time.

    The eager tuple holds every stage at once -- summed spans ~n
    elements, a full extra codeword at 2^27.  This keeps only the n/2
    ladder resident and yields each stage's contiguous slice on demand
    (the previous slice frees when the loop variable rebinds); the
    final stage IS the ladder (stride 1), no copy.
    """

    __slots__ = ("_ladder", "_n")

    def __init__(self, ladder, n):
        self._ladder = ladder
        self._n = n

    def __iter__(self):
        span = 2
        while span <= self._n:
            stride = self._n // span
            yield span, (
                self._ladder if stride == 1
                else self._ladder[::stride].contiguous())
            span <<= 1


def _ntt_stage_tables(generator, n):
    """Per-stage twiddles depend only on (generator, n): statement-free.

    Stage ``span`` needs ``(g^(n/span))^off`` for ``off < span/2`` -- every
    stage is a strided slice of ONE ladder ``g^j (j < n/2)``, so the whole
    set is built on device by log-doubling instead of an O(n) host loop.
    Above the cache cap the slices are NOT materialized up front: the
    returned object yields them per stage (see _LazyStageTables).
    """

    key = (generator, n)
    cached = _NTT_STAGE_CACHE.get(key)
    if cached is not None:
        return cached
    ladder = _power_ladder(generator, max(n // 2, 1))
    if n > _NTT_STAGE_CACHE_MAX_N:
        return _LazyStageTables(ladder, n)
    stage_tw = []
    span = 2
    while span <= n:
        stage_tw.append((span, ladder[:: n // span].contiguous()))
        span <<= 1
    stage_tw = tuple(stage_tw)
    if len(_NTT_STAGE_CACHE) < 64:
        _NTT_STAGE_CACHE[key] = stage_tw
    return stage_tw


def _ntt_coset_powers(shift, n):
    """shift^j for j < n, built ON DEVICE by log-doubling (exact).

    The coset shift is statement-derived, so a fresh statement must not
    cost an O(N) host loop -- this is log(n) tensor muls instead, into
    one preallocated buffer (see _power_ladder).
    """

    key = (shift, n)
    cached = _NTT_COSET_CACHE.get(key)
    if cached is not None:
        return cached
    powers = _power_ladder(shift, n)
    if n <= _NTT_CACHE_MAX_N and len(_NTT_COSET_CACHE) < 256:
        _NTT_COSET_CACHE[key] = powers
    return powers


def _ntt_tables(shift, generator, n):
    return _ntt_coset_powers(shift, n), _ntt_stage_tables(generator, n)


# big-n coset scales run per chunk from ONE bounded ladder instead of a
# size-n table (a full codeword + its build transient at 2^27)
_NTT_COSET_CHUNK: int = 1 << 23


def _coset_scale_chunked(extension, src, shift):
    """Elementwise shift^j scale WITHOUT the size-n power table.

    Chunk [i0, i1) scales by shift^i0 * ladder[j - i0] -- one
    chunk-size ladder plus a host pow() scalar per chunk, byte-identical
    to coset_scale_table over the full table (exact field arithmetic).
    """

    from verallm.proof_v3.native_goldilocks_backend import gl_scale_t

    n = src.numel()
    chunk = min(n, _NTT_COSET_CHUNK)
    ladder = _power_ladder(shift, chunk)
    base = shift % GOLDILOCKS_MODULUS
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        piece = ladder if stop - start == chunk else ladder[:stop - start]
        if start:
            piece = gl_scale_t(
                piece, pow(base, start, GOLDILOCKS_MODULUS))
        extension.coset_scale_table(src[start:stop], piece)


def _fused_ntt_apply(extension, holder, *, shift, generator, tables):
    """Shared NTT body over a 1-element ``holder`` list (ownership in).

    Popping the input out of ``holder`` means that when no caller frame
    keeps its own name for the tensor, the pre-bitrev buffer's refcount
    hits zero at the ``src = out`` rebind -- it frees BEFORE the
    butterfly stages instead of surviving all of them (a full codeword
    at 2^27)."""

    import torch

    src = holder.pop()
    n = src.numel()
    bits = (n - 1).bit_length()
    if 1 << bits != n:
        raise ProofV3Error("fused NTT size must be a power of two")
    if tables is not None:
        coset_powers, stage_tw = tables
        extension.coset_scale_table(src, coset_powers)
        del coset_powers
    elif n > _NTT_CACHE_MAX_N:
        _coset_scale_chunked(extension, src, shift)
        stage_tw = _ntt_stage_tables(generator, n)
    else:
        coset_powers, stage_tw = _ntt_tables(shift, generator, n)
        extension.coset_scale_table(src, coset_powers)
        del coset_powers
    out = torch.empty_like(src)
    extension.bitrev(src, out, bits)
    src = out
    for span, tw in stage_tw:
        extension.ntt_butterfly(src, tw, span)
    return src


def fused_ntt_goldilocks(extension, coefficients, *, shift, generator,
                         tables=None, mutable: bool = False):
    """GPU NTT byte-identical to ntt_goldilocks_reference (cached tables).

    ``tables``: explicit (coset_powers, stage_tw) device tensors --
    required inside CUDA-graph capture where cache lookups must not
    allocate or transfer.  ``mutable``: the caller owns
    ``coefficients`` as scratch (freshly built padding buffers) -- skip
    the defensive full-size clone (the coset scale mutates in
    place)."""

    from verallm.proof_v3.native_goldilocks_backend import to_field_tensor

    import torch as _torch

    if isinstance(coefficients, _torch.Tensor):
        if mutable and coefficients.is_cuda and (
                coefficients.is_contiguous()):
            a = coefficients
        else:
            a = coefficients.to("cuda").clone()
    else:
        a = to_field_tensor(coefficients, "cuda")
    holder = [a]
    del a
    return _fused_ntt_apply(
        extension, holder, shift=shift, generator=generator, tables=tables)


# -- four-step (Bailey) segmented NTT ---------------------------------
#
# n = N1 * N2; j = j1*N2 + j2, k = k1 + N1*k2.  With w the order-n
# principal root:
#   X[k1 + N1*k2] = sum_j2 [ (sum_j1 x[j1*N2+j2] * S^(N2*j1) * wN1^(j1*k1))
#                            * S^j2 * w^(j2*k1) ] * wN2^(j2*k2)
# Pass A: per column j2, a size-N1 coset NTT (shift S^N2) -- run as
# row-flattened batches through the EXISTING butterfly kernel (span
# blocks never cross row boundaries when rows are N1-aligned).
# Twiddle: w^(j2*k1) * S^j2 via two split power tables (e < n).
# Pass B: per row k1, a size-N2 plain NTT, scattered into the natural-
# order codeword (codeword.view(N2, N1)[k2, k1] = B[k1, k2]).
# The full padded buffer, its size-n coset table, and the n/2 stage
# ladder never materialize; the inter-pass intermediate lives in ONE
# reusable pinned host buffer.  Exact modular arithmetic throughout:
# byte-identical to the monolithic kernel by construction.

_FOUR_STEP_MIN_N: int = int(_os.environ.get(
    "VERATHOS_NTT_FOUR_STEP_MIN_N", str(1 << 26)))
_FOUR_STEP_SEG_ELEMS: int = 1 << 24
_FOUR_STEP_HOST_MID: dict = {}


def _four_step_enabled(n: int) -> bool:
    if _os.environ.get("VERATHOS_NTT_FOUR_STEP", "1") == "0":
        return False
    return n >= _FOUR_STEP_MIN_N


def _four_step_host_mid(n: int):
    import torch

    buf = _FOUR_STEP_HOST_MID.get(n)
    if buf is None:
        buf = torch.empty(n, dtype=torch.int64, pin_memory=True)
        _FOUR_STEP_HOST_MID.clear()
        _FOUR_STEP_HOST_MID[n] = buf
    return buf


def _rows_ntt_inplace(extension, flat, rows: int, size: int, generator):
    """size-``size`` DIT NTT on each of ``rows`` contiguous rows.

    ``flat`` is the (rows*size) contiguous buffer AFTER per-row bitrev;
    the global butterfly kernel is row-safe for span <= size."""

    for span, tw in _ntt_stage_tables(generator, size):
        extension.ntt_butterfly(flat, tw, span)


def fused_ntt_goldilocks_four_step(extension, values_device, n: int, *,
                                   shift, generator):
    """Segment-bounded coset NTT of ``values_device`` zero-padded to n."""

    import torch

    from verallm.proof_v3.native_goldilocks_backend import gl_mul_t
    from verallm.proof_v3.native_pcs_backend import _bitrev_indices

    bits = (n - 1).bit_length()
    if 1 << bits != n:
        raise ProofV3Error("four-step NTT size must be a power of two")
    bits1 = (bits + 1) // 2
    n1, n2 = 1 << bits1, 1 << (bits - bits1)
    values_device = values_device.contiguous()
    filled = values_device.numel() // n2
    if filled * n2 != values_device.numel() or filled > n1 or filled == 0:
        raise ProofV3Error("four-step NTT needs N2 | value count <= n")
    shift = shift % GOLDILOCKS_MODULUS
    generator = generator % GOLDILOCKS_MODULUS
    shift_a = pow(shift, n2, GOLDILOCKS_MODULUS)
    w_n1 = pow(generator, n2, GOLDILOCKS_MODULUS)
    w_n2 = pow(generator, n1, GOLDILOCKS_MODULUS)
    ladder_a = _power_ladder(shift_a, n1)
    shift_j2 = _power_ladder(shift, n2)
    w_lo = _power_ladder(generator, n1)
    w_hi = _power_ladder(pow(generator, n1, GOLDILOCKS_MODULUS), n2)
    rev1 = _bitrev_indices(bits1)
    rev2 = _bitrev_indices(bits - bits1)
    values_mat = values_device.view(filled, n2)
    host_mid = _four_step_host_mid(n).view(n1, n2)

    seg_cols = max(1, min(n2, _FOUR_STEP_SEG_ELEMS // n1))
    k1_idx = torch.arange(n1, dtype=torch.int64, device="cuda")
    for c0 in range(0, n2, seg_cols):
        c1 = min(c0 + seg_cols, n2)
        seg = c1 - c0
        mat = torch.zeros(seg, n1, dtype=torch.int64, device="cuda")
        mat[:, :filled] = values_mat[:, c0:c1].t()
        mat = gl_mul_t(mat, ladder_a.view(1, n1).expand(seg, n1))
        mat = mat.index_select(1, rev1).contiguous()
        _rows_ntt_inplace(extension, mat.view(-1), seg, n1, w_n1)
        exps = torch.arange(
            c0, c1, dtype=torch.int64, device="cuda").view(seg, 1) * k1_idx
        tw = gl_mul_t(
            w_lo.index_select(0, exps.view(-1) & (n1 - 1)),
            w_hi.index_select(0, exps.view(-1) >> bits1),
        ).view(seg, n1)
        del exps
        tw = gl_mul_t(
            tw, shift_j2[c0:c1].view(seg, 1).expand(seg, n1))
        mat = gl_mul_t(mat.view(seg, n1), tw)
        del tw
        host_mid[:, c0:c1].copy_(mat.t().contiguous())
        del mat

    codeword = torch.empty(n, dtype=torch.int64, device="cuda")
    code_mat = codeword.view(n2, n1)
    seg_rows = max(1, min(n1, _FOUR_STEP_SEG_ELEMS // n2))
    for r0 in range(0, n1, seg_rows):
        r1 = min(r0 + seg_rows, n1)
        rows = host_mid[r0:r1, :].to("cuda", non_blocking=True)
        rows = rows.index_select(1, rev2).contiguous()
        _rows_ntt_inplace(extension, rows.view(-1), r1 - r0, n2, w_n2)
        code_mat[:, r0:r1] = rows.t()
        del rows
    return codeword


def fused_ntt_goldilocks_consume(extension, holder, *, shift, generator,
                                 tables=None):
    """fused_ntt_goldilocks(mutable=True) taking SOLE ownership.

    ``holder`` is a 1-element list containing the input tensor and is
    emptied here; the caller must hold no other name for it, so the
    pre-bitrev buffer frees before the butterflies (-1 codeword of
    commit transient at big n)."""

    import torch as _torch

    tensor = holder.pop()
    if not (isinstance(tensor, _torch.Tensor) and tensor.is_cuda
            and tensor.is_contiguous()
            and tensor.dtype == _torch.int64):
        raise ProofV3Error(
            "consume NTT requires a contiguous cuda int64 tensor")
    inner = [tensor]
    del tensor
    return _fused_ntt_apply(
        extension, inner, shift=shift, generator=generator, tables=tables)


__all__.append("fused_ntt_goldilocks")
__all__.append("fused_ntt_goldilocks_consume")
