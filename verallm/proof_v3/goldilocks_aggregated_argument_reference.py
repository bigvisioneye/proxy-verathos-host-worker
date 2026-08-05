"""Random-linear-combination aggregation of succinct fold arguments.

Collapses N same-shape fold arguments (e.g. all of a model's projection
matmuls) into ONE succinct argument: one verify, one wire, instead of N.
This is the structural fix for "verify too long" (1.47 s over 97 args) and
the 5.5 MB wire -- both drop to a single argument's cost.

Construction. Each sub-argument proves ``sum_k W_a[k] * F_a[k] == S_a``
where ``F_a`` is a public tensor factor. Post-nonce the verifier draws one
aggregation coefficient ``c_a`` per argument (from a transcript over all
sub-commitments + nonce). Concatenate the witnesses arg-major into one
multilinear ``W`` over ``a+m`` variables (``2^a >= N`` blocks of ``2^m``),
committed once. The block-structured factor
``F[a*2^m + k] = c_a * F_a[k]`` satisfies
``sum_i W[i] F[i] == sum_a c_a S_a``, so a single fold sumcheck + one PCS
opening proves the whole batch.

Soundness. If any sub-claim ``S_a`` is wrong for the committed block, the
aggregate sum is wrong except with probability ``(N)/p`` over the
coefficients (Schwartz-Zippel), on top of the underlying argument's
soundness. So aggregation does not weaken the guarantee; it only shares
the verifier's fixed costs.

Verifier work: one ``(a+m)``-round sumcheck chain, one PCS opening
(O(q log N)), and ``F~(c)`` evaluated via the block structure in
``O(N*(a+m))`` -- N and m are small, so this is negligible. Never touches
a witness vector.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_fold_sumcheck_reference import _challenge
from verallm.proof_v3.goldilocks_linear_relation_reference import (
    _fixed32,
    _integer,
)
from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
    GoldilocksMultilinearOpeningProofV3,
    GoldilocksMultilinearPcsStatementV3,
    commit_goldilocks_multilinear_v3,
    open_goldilocks_multilinear_v3,
    verify_goldilocks_multilinear_opening_v3,
)
from verallm.proof_v3.goldilocks_reference import (
    GOLDILOCKS_MODULUS,
    goldilocks_inv,
)


GOLDILOCKS_AGGREGATED_ARGUMENT_ABI_V3: Final = (
    "goldilocks.aggregated_argument.reference.v1"
)
_AGG_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_AGG/V1/TRANSCRIPT/SHA256"
_COEFF_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_AGG/V1/COEFF/SHA256"
_ROUND_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_AGG/V1/ROUND/SHA256"


def _field(value: object, name: str) -> int:
    integer = _integer(value, name)
    if not 0 <= integer < GOLDILOCKS_MODULUS:
        raise ProofV3Error(f"{name} must be a canonical Goldilocks element")
    return integer


def _tensor_factor(components: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    table = (1,)
    for component in components:
        table = tuple(
            left * right % GOLDILOCKS_MODULUS
            for left in table
            for right in component
        )
    return table


def _mle_eval_msb(values: tuple[int, ...], point: tuple[int, ...]) -> int:
    working = list(values)
    for challenge in point:
        half = len(working) // 2
        working = [
            (working[i] + challenge * (working[half + i] - working[i]))
            % GOLDILOCKS_MODULUS
            for i in range(half)
        ]
    return working[0]


@dataclass(frozen=True, slots=True)
class GoldilocksAggregatedArgumentProofV3:
    aggregate_claim: int
    outer_rounds: tuple[tuple[int, int, int], ...]
    opening: GoldilocksMultilinearOpeningProofV3


@dataclass(frozen=True, slots=True)
class GoldilocksAggregatedSubClaimV3:
    """One sub-argument: its factor components and claimed folded sum."""

    factor_components: tuple[tuple[int, ...], ...]
    claimed_sum: int
    witness: tuple[int, ...]  # prover-side only; not in the proof


def _agg_seed(
    *,
    validator_binding_digest: bytes,
    contraction_bits: int,
    arg_bits: int,
    commitment: bytes,
    sub_digests: tuple[bytes, ...],
    validator_nonce: bytes,
) -> bytes:
    h = hashlib.sha256(_AGG_DOMAIN + validator_binding_digest)
    h.update(struct.pack("<II", contraction_bits, arg_bits))
    h.update(_fixed32(commitment, "agg commitment"))
    for digest in sub_digests:
        h.update(digest)
    h.update(_fixed32(validator_nonce, "validator_nonce"))
    return h.digest()


def _sub_digest(sub: GoldilocksAggregatedSubClaimV3) -> bytes:
    h = hashlib.sha256(b"agg-sub")
    for component in sub.factor_components:
        h.update(struct.pack("<I", len(component)))
        h.update(b"".join(v.to_bytes(8, "little") for v in component))
    h.update(sub.claimed_sum.to_bytes(8, "little"))
    return h.digest()


def prove_goldilocks_aggregated_argument_v3(
    *,
    validator_binding_digest: bytes,
    subclaims: tuple[GoldilocksAggregatedSubClaimV3, ...],
    validator_nonce: bytes,
) -> tuple[GoldilocksMultilinearPcsStatementV3, bytes, GoldilocksAggregatedArgumentProofV3]:
    """Aggregate N same-shape sub-arguments into one succinct argument."""

    binding = _fixed32(validator_binding_digest, "agg binding", nonzero=True)
    if not subclaims:
        raise ProofV3Error("aggregation needs at least one sub-claim")
    m = len(subclaims[0].witness)
    if m < 2 or m & (m - 1):
        raise ProofV3Error("sub-witness size must be a power of two")
    for sub in subclaims:
        if len(sub.witness) != m:
            raise ProofV3Error("all sub-witnesses must share the same shape")
        if len(_tensor_factor(sub.factor_components)) != m:
            raise ProofV3Error("sub factor does not match the witness shape")
    contraction_bits = m.bit_length() - 1
    n_args = len(subclaims)
    arg_bits = max(0, (n_args - 1).bit_length())
    blocks = 1 << arg_bits
    total_vars = arg_bits + contraction_bits

    # Concatenate witnesses arg-major (arg = high bits), zero-pad blocks.
    concat = []
    for arg in range(blocks):
        if arg < n_args:
            concat.extend(_field(v, "witness") for v in subclaims[arg].witness)
        else:
            concat.extend(0 for _ in range(m))
    concat = tuple(concat)

    statement = GoldilocksMultilinearPcsStatementV3(
        validator_binding_digest=binding, variable_count=total_vars
    )
    commit = commit_goldilocks_multilinear_v3(
        statement=statement, evaluations=concat
    )
    sub_digests = tuple(_sub_digest(s) for s in subclaims)
    seed = _agg_seed(
        validator_binding_digest=binding,
        contraction_bits=contraction_bits,
        arg_bits=arg_bits,
        commitment=commit.commitment,
        sub_digests=sub_digests,
        validator_nonce=validator_nonce,
    )
    coeffs = tuple(
        _challenge(hashlib.sha256(_COEFF_DOMAIN + seed).digest(), arg + 1)
        for arg in range(blocks)
    )
    # Block-structured factor and aggregate claim.
    factor = []
    for arg in range(blocks):
        if arg < n_args:
            f_arg = _tensor_factor(subclaims[arg].factor_components)
            factor.extend(coeffs[arg] * fk % GOLDILOCKS_MODULUS for fk in f_arg)
        else:
            factor.extend(0 for _ in range(m))
    factor = tuple(factor)
    aggregate = 0
    for arg in range(n_args):
        aggregate = (
            aggregate + coeffs[arg] * (subclaims[arg].claimed_sum % GOLDILOCKS_MODULUS)
        ) % GOLDILOCKS_MODULUS

    # Outer fold sumcheck over concat and factor (MSB-first).
    w = list(concat)
    f = list(factor)
    transcript = hashlib.sha256(_ROUND_DOMAIN + seed).digest()
    rounds = []
    challenges = []
    while len(w) > 1:
        half = len(w) // 2
        g0 = g1 = g2 = 0
        for i in range(half):
            wl, wh = w[i], w[half + i]
            fl, fh = f[i], f[half + i]
            g0 = (g0 + wl * fl) % GOLDILOCKS_MODULUS
            g1 = (g1 + wh * fh) % GOLDILOCKS_MODULUS
            g2 = (g2 + (2 * wh - wl) * (2 * fh - fl)) % GOLDILOCKS_MODULUS
        rounds.append((g0, g1, g2))
        transcript = hashlib.sha256(
            transcript
            + g0.to_bytes(8, "little")
            + g1.to_bytes(8, "little")
            + g2.to_bytes(8, "little")
        ).digest()
        ch = _challenge(transcript, len(rounds))
        challenges.append(ch)
        w = [(w[i] + ch * (w[half + i] - w[i])) % GOLDILOCKS_MODULUS for i in range(half)]
        f = [(f[i] + ch * (f[half + i] - f[i])) % GOLDILOCKS_MODULUS for i in range(half)]

    opening = open_goldilocks_multilinear_v3(
        statement=statement,
        tree=commit,
        evaluations=concat,
        point=tuple(reversed(challenges)),
        validator_nonce=validator_nonce,
    )
    proof = GoldilocksAggregatedArgumentProofV3(
        aggregate_claim=aggregate,
        outer_rounds=tuple(rounds),
        opening=opening,
    )
    return statement, commit.commitment, proof


def verify_goldilocks_aggregated_argument_v3(
    proof: object,
    *,
    validator_binding_digest: bytes,
    statement: GoldilocksMultilinearPcsStatementV3,
    commitment: bytes,
    sub_factor_components: tuple[tuple[tuple[int, ...], ...], ...],
    sub_claimed_sums: tuple[int, ...],
    validator_nonce: bytes,
) -> None:
    """Verify the whole batch with ONE argument's cost."""

    try:
        if not isinstance(proof, GoldilocksAggregatedArgumentProofV3):
            raise ProofV3VerificationError("aggregated proof type is unexpected")
        binding = _fixed32(validator_binding_digest, "agg binding", nonzero=True)
        n_args = len(sub_claimed_sums)
        if len(sub_factor_components) != n_args:
            raise ProofV3VerificationError("aggregate sub-shape mismatch")
        m = len(_tensor_factor(sub_factor_components[0]))
        contraction_bits = m.bit_length() - 1
        arg_bits = max(0, (n_args - 1).bit_length())
        blocks = 1 << arg_bits
        total_vars = arg_bits + contraction_bits
        if statement.variable_count != total_vars:
            raise ProofV3VerificationError("aggregate statement shape mismatch")
        if len(proof.outer_rounds) != total_vars:
            raise ProofV3VerificationError("aggregate round count mismatch")

        subclaim_digests = tuple(
            hashlib.sha256(
                b"agg-sub"
                + b"".join(
                    struct.pack("<I", len(c))
                    + b"".join(v.to_bytes(8, "little") for v in c)
                    for c in sub_factor_components[a]
                )
                + (sub_claimed_sums[a] % GOLDILOCKS_MODULUS).to_bytes(8, "little")
            ).digest()
            for a in range(n_args)
        )
        seed = _agg_seed(
            validator_binding_digest=binding,
            contraction_bits=contraction_bits,
            arg_bits=arg_bits,
            commitment=_fixed32(commitment, "agg commitment"),
            sub_digests=subclaim_digests,
            validator_nonce=validator_nonce,
        )
        coeffs = tuple(
            _challenge(hashlib.sha256(_COEFF_DOMAIN + seed).digest(), arg + 1)
            for arg in range(blocks)
        )
        aggregate = 0
        for a in range(n_args):
            aggregate = (
                aggregate + coeffs[a] * (sub_claimed_sums[a] % GOLDILOCKS_MODULUS)
            ) % GOLDILOCKS_MODULUS
        if proof.aggregate_claim != aggregate:
            raise ProofV3VerificationError(
                "aggregate claim does not match the sub-claims"
            )

        # Outer sumcheck chain.
        transcript = hashlib.sha256(_ROUND_DOMAIN + seed).digest()
        running = aggregate
        challenges = []
        inv2 = (GOLDILOCKS_MODULUS + 1) // 2
        for idx, (g0, g1, g2) in enumerate(proof.outer_rounds, 1):
            g0, g1, g2 = _field(g0, "g0"), _field(g1, "g1"), _field(g2, "g2")
            if (g0 + g1) % GOLDILOCKS_MODULUS != running:
                raise ProofV3VerificationError(
                    "aggregate round does not match the running sum"
                )
            transcript = hashlib.sha256(
                transcript
                + g0.to_bytes(8, "little")
                + g1.to_bytes(8, "little")
                + g2.to_bytes(8, "little")
            ).digest()
            ch = _challenge(transcript, idx)
            challenges.append(ch)
            z = ch
            running = (
                g0 * ((z - 1) * (z - 2) % GOLDILOCKS_MODULUS) % GOLDILOCKS_MODULUS * inv2
                - g1 * (z * (z - 2) % GOLDILOCKS_MODULUS)
                + g2 * (z * (z - 1) % GOLDILOCKS_MODULUS) % GOLDILOCKS_MODULUS * inv2
            ) % GOLDILOCKS_MODULUS

        # F~(c) via block structure: high `arg_bits` challenges select the
        # block (eq), low `contraction_bits` evaluate each tensor factor.
        c_arg = tuple(challenges[:arg_bits])
        c_low = tuple(challenges[arg_bits:])
        f_at_point = 0
        for a in range(n_args):
            # eq(arg_bits index a, c_arg): bit j (MSB-first) of a
            eq = 1
            for bit_pos, ch in enumerate(c_arg):
                bit = (a >> (arg_bits - 1 - bit_pos)) & 1
                eq = eq * (ch if bit else (1 - ch) % GOLDILOCKS_MODULUS) % GOLDILOCKS_MODULUS
            f_arg_at = 1
            cursor = 0
            for component in sub_factor_components[a]:
                bits = len(component).bit_length() - 1
                f_arg_at = (
                    f_arg_at * _mle_eval_msb(component, tuple(c_low[cursor:cursor + bits]))
                ) % GOLDILOCKS_MODULUS
                cursor += bits
            f_at_point = (
                f_at_point + eq * coeffs[a] % GOLDILOCKS_MODULUS * f_arg_at
            ) % GOLDILOCKS_MODULUS
        if f_at_point == 0:
            raise ProofV3VerificationError("aggregate factor evaluates to zero")
        expected_w = running * goldilocks_inv(f_at_point) % GOLDILOCKS_MODULUS
        verify_goldilocks_multilinear_opening_v3(
            proof.opening,
            statement=statement,
            commitment=commitment,
            point=tuple(reversed(challenges)),
            expected_value=expected_w,
            validator_nonce=validator_nonce,
        )
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("aggregated proof is malformed") from exc


__all__ = [
    "GOLDILOCKS_AGGREGATED_ARGUMENT_ABI_V3",
    "GoldilocksAggregatedArgumentProofV3",
    "GoldilocksAggregatedSubClaimV3",
    "prove_goldilocks_aggregated_argument_v3",
    "verify_goldilocks_aggregated_argument_v3",
]
