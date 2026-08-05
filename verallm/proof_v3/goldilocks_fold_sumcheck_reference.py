"""Sumcheck reference for the folded X-scan of the global linear relation.

The folded relation still commits an O(T*M) accumulator trace for the
x-scan.  This module removes it: the scalar ``S_x = sum_{t,k} v[t] *
X[t,k] * wu[k]`` is proven by a standard multilinear sumcheck over the
boolean hypercube of ``log2(T*M)`` variables, against a pre-nonce frozen
multilinear commitment of X.  Committed data becomes O(1) (the X
commitment already exists); prover work stays O(T*M) raw field muls with
zero committed trace cells — the production cost profile from the A40
spike (raw muls are ~250x cheaper than committed cells).

Reference-only PCS: the X commitment is the existing Goldilocks Merkle tree
over the padded evaluation vector, and the final sumcheck evaluation
``X(r)`` is checked by the verifier recomputing the multilinear extension
from a full opening of the tree (O(T*M) verifier work).  A production
backend replaces exactly this step with a succinct multilinear PCS
(FRI-based, e.g. Basefold-style) — the sumcheck rounds themselves are
already production-shaped.

The public factor ``F(t,k) = v[t] * wu[k]`` is multilinear in the bit
variables of ``t`` and ``k``?  It is not — ``v`` and ``wu`` are arbitrary
tables — but its multilinear extension over the hypercube is exactly what
the verifier can evaluate at the sumcheck point ``r`` on its own in
O(T + M) time using the tensor split ``F~(r) = v~(r_t) * wu~(r_k)``,
because the index bits of ``t`` and ``k`` are disjoint variable groups and
``F`` factorises.  ``v`` is nonce-derived and ``wu = W @ u`` comes from the
signed weights, so no part of ``F`` is prover-supplied.

Fiat-Shamir: every round polynomial is absorbed into a running SHA-256
transcript seeded by the statement digest, the frozen X commitment, the
validator nonce, and the claimed scalar.  Round challenges are rejection
sampled below the modulus.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_linear_relation_reference import (
    _fixed32,
    _integer,
)
from verallm.proof_v3.goldilocks_merkle_reference import (
    GoldilocksMerkleTreeReference,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS


GOLDILOCKS_FOLD_SUMCHECK_ABI_V3: Final = "goldilocks.fold_sumcheck.reference.v1"
MAX_GOLDILOCKS_FOLD_SUMCHECK_VARIABLES_V3: Final = 20

_TRANSCRIPT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_FOLD_SUMCHECK/V1/TRANSCRIPT/SHA256"
)
_CHALLENGE_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_FOLD_SUMCHECK/V1/CHALLENGE/SHA256"
)
_COMMIT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_FOLD_SUMCHECK/V1/X_COMMIT/SHA256"
)


def _field(value: object, name: str) -> int:
    integer = _integer(value, name)
    if not 0 <= integer < GOLDILOCKS_MODULUS:
        raise ProofV3Error(f"{name} must be a canonical Goldilocks element")
    return integer


def _challenge(seed: bytes, round_index: int) -> int:
    for counter in range(1 << 16):
        candidate = int.from_bytes(
            hashlib.sha256(
                _CHALLENGE_DOMAIN + seed + struct.pack("<II", round_index, counter)
            ).digest()[:8],
            "little",
        )
        if candidate < GOLDILOCKS_MODULUS:
            return candidate
    raise ProofV3Error("unable to derive a fold-sumcheck challenge")


def _mle_fold(values: list[int], challenge: int) -> list[int]:
    """One multilinear variable binding: f(b, rest) -> f(challenge, rest)."""

    half = len(values) // 2
    return [
        (values[i] + challenge * (values[half + i] - values[i]))
        % GOLDILOCKS_MODULUS
        for i in range(half)
    ]


def _mle_eval(values: tuple[int, ...], point: tuple[int, ...]) -> int:
    working = list(values)
    # Variables bound most-significant-bit first, matching the prover's
    # halving order over the top half of the table.
    for challenge in point:
        working = _mle_fold(working, challenge)
    return working[0]


@dataclass(frozen=True, slots=True)
class GoldilocksFoldSumcheckProofV3:
    """Round polynomials (degree <= 2, three evaluations each) + claimed sum."""

    claimed_sum: int
    round_polynomials: tuple[tuple[int, int, int], ...]
    x_full_opening: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "claimed_sum", _field(self.claimed_sum, "claimed_sum")
        )
        if not isinstance(self.round_polynomials, tuple) or not all(
            isinstance(item, tuple)
            and len(item) == 3
            and all(0 <= _integer(value, "round value") < GOLDILOCKS_MODULUS
                    for value in item)
            for item in self.round_polynomials
        ):
            raise ProofV3Error("fold-sumcheck round polynomial set is malformed")
        if len(self.round_polynomials) > MAX_GOLDILOCKS_FOLD_SUMCHECK_VARIABLES_V3:
            raise ProofV3Error("fold-sumcheck variable count is out of range")
        if not isinstance(self.x_full_opening, tuple):
            raise ProofV3Error("fold-sumcheck X opening is malformed")


def commit_goldilocks_fold_sumcheck_x_v3(
    *,
    statement_digest: bytes,
    x_evaluations: tuple[int, ...],
) -> GoldilocksMerkleTreeReference:
    """Freeze the padded X evaluation vector pre-nonce (power-of-two length)."""

    length = len(x_evaluations)
    if length < 2 or length & (length - 1):
        raise ProofV3Error("fold-sumcheck X evaluations must be a power of two")
    if length.bit_length() - 1 > MAX_GOLDILOCKS_FOLD_SUMCHECK_VARIABLES_V3:
        raise ProofV3Error("fold-sumcheck X exceeds the CPU reference cap")
    values = tuple(_field(value, "x evaluation") for value in x_evaluations)
    binding = hashlib.sha256(
        _COMMIT_DOMAIN
        + _fixed32(statement_digest, "statement_digest", nonzero=True)
    ).digest()
    return GoldilocksMerkleTreeReference.from_rows(
        tuple((value,) for value in values),
        binding_digest=binding,
    )


def _transcript_seed(
    *,
    statement_digest: bytes,
    x_commitment: bytes,
    validator_nonce: bytes,
    factor_digest: bytes,
    claimed_sum: int,
) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + _fixed32(statement_digest, "statement_digest", nonzero=True)
        + _fixed32(x_commitment, "x_commitment")
        + _fixed32(validator_nonce, "validator_nonce")
        + _fixed32(factor_digest, "factor_digest", nonzero=True)
        + claimed_sum.to_bytes(8, "little")
    ).digest()


def factor_digest_v3(factor: tuple[int, ...]) -> bytes:
    """Digest of the public factor table F (verifier-derived, never prover)."""

    return hashlib.sha256(
        b"".join(_field(value, "factor value").to_bytes(8, "little")
                 for value in factor)
    ).digest()


def prove_goldilocks_fold_sumcheck_v3(
    *,
    statement_digest: bytes,
    x_tree: GoldilocksMerkleTreeReference,
    x_evaluations: tuple[int, ...],
    factor: tuple[int, ...],
    validator_nonce: bytes,
) -> GoldilocksFoldSumcheckProofV3:
    """Prove ``sum_i X[i] * F[i]`` with zero committed trace cells.

    ``factor`` is the public table ``F`` (v ⊗ wu on the x-index range,
    zero on padding).  Both tables must share the padded power-of-two
    length of the frozen commitment.
    """

    if len(x_evaluations) != len(factor):
        raise ProofV3Error("fold-sumcheck factor length mismatch")
    if tuple((v,) for v in x_evaluations) != x_tree.rows[: len(x_evaluations)] and (
        len(x_tree.rows) != len(x_evaluations)
    ):
        raise ProofV3Error("fold-sumcheck X evaluations do not match the tree")
    x_values = [_field(value, "x evaluation") for value in x_evaluations]
    f_values = [_field(value, "factor value") for value in factor]
    claimed = sum(
        x * f % GOLDILOCKS_MODULUS for x, f in zip(x_values, f_values)
    ) % GOLDILOCKS_MODULUS
    seed = _transcript_seed(
        statement_digest=statement_digest,
        x_commitment=x_tree.commitment,
        validator_nonce=validator_nonce,
        factor_digest=factor_digest_v3(tuple(f_values)),
        claimed_sum=claimed,
    )
    rounds: list[tuple[int, int, int]] = []
    transcript = seed
    while len(x_values) > 1:
        half = len(x_values) // 2
        # Round polynomial g(z) = sum_rest X(z, rest) * F(z, rest), degree 2.
        g0 = g1 = g2 = 0
        for i in range(half):
            x_lo, x_hi = x_values[i], x_values[half + i]
            f_lo, f_hi = f_values[i], f_values[half + i]
            g0 = (g0 + x_lo * f_lo) % GOLDILOCKS_MODULUS
            g1 = (g1 + x_hi * f_hi) % GOLDILOCKS_MODULUS
            # g(2) with linear extensions: x(2) = 2*x_hi - x_lo, same for f.
            x2 = (2 * x_hi - x_lo) % GOLDILOCKS_MODULUS
            f2 = (2 * f_hi - f_lo) % GOLDILOCKS_MODULUS
            g2 = (g2 + x2 * f2) % GOLDILOCKS_MODULUS
        rounds.append((g0, g1, g2))
        transcript = hashlib.sha256(
            transcript
            + g0.to_bytes(8, "little")
            + g1.to_bytes(8, "little")
            + g2.to_bytes(8, "little")
        ).digest()
        challenge = _challenge(transcript, len(rounds))
        x_values = _mle_fold(x_values, challenge)
        f_values = _mle_fold(f_values, challenge)
    return GoldilocksFoldSumcheckProofV3(
        claimed_sum=claimed,
        round_polynomials=tuple(rounds),
        x_full_opening=tuple(
            row[0] for row in x_tree.rows
        ),
    )


def verify_goldilocks_fold_sumcheck_v3(
    proof: object,
    *,
    statement_digest: bytes,
    x_commitment: bytes,
    factor: tuple[int, ...],
    validator_nonce: bytes,
    expected_sum: int,
) -> None:
    """Verify the sumcheck and the final X(r) evaluation.

    ``factor`` is recomputed by the verifier (nonce fold coefficients and
    signed weights); ``expected_sum`` is the claimed folded scalar the outer
    relation requires (the y-scan total).  The reference checks ``X(r)`` by
    rebuilding the multilinear extension from the full opening and
    re-deriving the commitment; a production PCS replaces exactly this step.
    """

    try:
        if not isinstance(proof, GoldilocksFoldSumcheckProofV3):
            raise ProofV3VerificationError("fold-sumcheck proof type is unexpected")
        f_values = tuple(_field(value, "factor value") for value in factor)
        length = len(f_values)
        if length < 2 or length & (length - 1):
            raise ProofV3VerificationError("fold-sumcheck factor length is invalid")
        variables = length.bit_length() - 1
        if len(proof.round_polynomials) != variables:
            raise ProofV3VerificationError("fold-sumcheck round count is wrong")
        if proof.claimed_sum != _field(expected_sum, "expected_sum"):
            raise ProofV3VerificationError(
                "fold-sumcheck claimed sum does not match the outer relation"
            )
        seed = _transcript_seed(
            statement_digest=statement_digest,
            x_commitment=_fixed32(x_commitment, "x_commitment"),
            validator_nonce=validator_nonce,
            factor_digest=factor_digest_v3(f_values),
            claimed_sum=proof.claimed_sum,
        )
        transcript = seed
        running = proof.claimed_sum
        point: list[int] = []
        for round_index, (g0, g1, g2) in enumerate(proof.round_polynomials, 1):
            if (g0 + g1) % GOLDILOCKS_MODULUS != running:
                raise ProofV3VerificationError(
                    "fold-sumcheck round polynomial does not match the running sum"
                )
            transcript = hashlib.sha256(
                transcript
                + g0.to_bytes(8, "little")
                + g1.to_bytes(8, "little")
                + g2.to_bytes(8, "little")
            ).digest()
            challenge = _challenge(transcript, round_index)
            point.append(challenge)
            # Evaluate the degree-2 polynomial from values at 0, 1, 2 via
            # Lagrange: g(z) = g0*(z-1)(z-2)/2 - g1*z(z-2) + g2*z(z-1)/2.
            inv2 = (GOLDILOCKS_MODULUS + 1) // 2
            z = challenge
            running = (
                g0 * ((z - 1) * (z - 2) % GOLDILOCKS_MODULUS) % GOLDILOCKS_MODULUS * inv2
                - g1 * (z * (z - 2) % GOLDILOCKS_MODULUS)
                + g2 * (z * (z - 1) % GOLDILOCKS_MODULUS) % GOLDILOCKS_MODULUS * inv2
            ) % GOLDILOCKS_MODULUS
        # Final check: running == X(r) * F(r).
        f_at_point = _mle_eval(f_values, tuple(point))
        opening = proof.x_full_opening
        if len(opening) != length:
            raise ProofV3VerificationError("fold-sumcheck X opening length is wrong")
        x_values = tuple(_field(value, "x opening value") for value in opening)
        binding = hashlib.sha256(
            _COMMIT_DOMAIN
            + _fixed32(statement_digest, "statement_digest", nonzero=True)
        ).digest()
        rebuilt = GoldilocksMerkleTreeReference.from_rows(
            tuple((value,) for value in x_values),
            binding_digest=binding,
        )
        if rebuilt.commitment != x_commitment:
            raise ProofV3VerificationError(
                "fold-sumcheck X opening does not match the frozen commitment"
            )
        x_at_point = _mle_eval(x_values, tuple(point))
        if running != x_at_point * f_at_point % GOLDILOCKS_MODULUS:
            raise ProofV3VerificationError(
                "fold-sumcheck final evaluation does not match the commitment"
            )
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("fold-sumcheck proof is malformed") from exc


__all__ = [
    "GOLDILOCKS_FOLD_SUMCHECK_ABI_V3",
    "GoldilocksFoldSumcheckProofV3",
    "commit_goldilocks_fold_sumcheck_x_v3",
    "factor_digest_v3",
    "prove_goldilocks_fold_sumcheck_v3",
    "verify_goldilocks_fold_sumcheck_v3",
]
