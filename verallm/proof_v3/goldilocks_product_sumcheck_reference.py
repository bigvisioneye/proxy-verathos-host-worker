"""Two-table product sumcheck reference over Goldilocks for proof-v3.

Generalises the single-table fold sumcheck to ``sum_i A[i] * B[i] * F[i]``
where **both** ``A`` and ``B`` are independently frozen committed tables
and ``F`` is a public factor the verifier evaluates itself.  This is the
shape runtime-by-runtime matrix products need — attention scores
``v^T (Q K^T) u = sum_{t,s,d} v[t] u[s] Q[t,d] K[s,d]`` and the PV product
— where neither operand is validator-signed, so neither side can be folded
away like the weight matrix.

Round polynomials have degree 3 (two committed multilinears times the
public factor's linear extension), sent as evaluations at 0..3.  The final
check needs ``A(r)`` and ``B(r)``; the reference rebuilds both MLEs from
full openings against their frozen roots — production replaces exactly
that with a succinct multilinear PCS, the rounds are already
production-shaped.

For the attention use the index space is the product cube: caller flattens
``(t, s, d)`` into one hypercube and supplies ``A = Q[t,d]`` broadcast,
``B = K[s,d]`` broadcast, ``F = v[t] * u[s]``.  Broadcast layouts are the
caller's contract; this module proves the sum over whatever tables were
frozen.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_fold_sumcheck_reference import (
    _challenge,
    _field,
    _mle_eval,
    _mle_fold,
)
from verallm.proof_v3.goldilocks_linear_relation_reference import _fixed32
from verallm.proof_v3.goldilocks_merkle_reference import (
    GoldilocksMerkleTreeReference,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS


GOLDILOCKS_PRODUCT_SUMCHECK_ABI_V3: Final = (
    "goldilocks.product_sumcheck.reference.v1"
)
MAX_GOLDILOCKS_PRODUCT_SUMCHECK_VARIABLES_V3: Final = 20

_TRANSCRIPT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_PRODUCT_SUMCHECK/V1/TRANSCRIPT/SHA256"
)
_A_COMMIT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_PRODUCT_SUMCHECK/V1/A_COMMIT/SHA256"
)
_B_COMMIT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_PRODUCT_SUMCHECK/V1/B_COMMIT/SHA256"
)


def _commit(
    *,
    statement_digest: bytes,
    evaluations: tuple[int, ...],
    domain: bytes,
) -> GoldilocksMerkleTreeReference:
    length = len(evaluations)
    if length < 2 or length & (length - 1):
        raise ProofV3Error("product-sumcheck table must be a power of two")
    if length.bit_length() - 1 > MAX_GOLDILOCKS_PRODUCT_SUMCHECK_VARIABLES_V3:
        raise ProofV3Error("product-sumcheck table exceeds the CPU cap")
    binding = hashlib.sha256(
        domain + _fixed32(statement_digest, "statement_digest", nonzero=True)
    ).digest()
    return GoldilocksMerkleTreeReference.from_rows(
        tuple((_field(value, "table value"),) for value in evaluations),
        binding_digest=binding,
    )


def commit_goldilocks_product_sumcheck_a_v3(
    *, statement_digest: bytes, evaluations: tuple[int, ...]
) -> GoldilocksMerkleTreeReference:
    return _commit(
        statement_digest=statement_digest,
        evaluations=evaluations,
        domain=_A_COMMIT_DOMAIN,
    )


def commit_goldilocks_product_sumcheck_b_v3(
    *, statement_digest: bytes, evaluations: tuple[int, ...]
) -> GoldilocksMerkleTreeReference:
    return _commit(
        statement_digest=statement_digest,
        evaluations=evaluations,
        domain=_B_COMMIT_DOMAIN,
    )


def _factor_digest(factor: tuple[int, ...]) -> bytes:
    return hashlib.sha256(
        b"".join(_field(value, "factor value").to_bytes(8, "little")
                 for value in factor)
    ).digest()


def _seed(
    *,
    statement_digest: bytes,
    a_commitment: bytes,
    b_commitment: bytes,
    validator_nonce: bytes,
    factor: tuple[int, ...],
    claimed_sum: int,
) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + _fixed32(statement_digest, "statement_digest", nonzero=True)
        + _fixed32(a_commitment, "a_commitment")
        + _fixed32(b_commitment, "b_commitment")
        + _fixed32(validator_nonce, "validator_nonce")
        + _factor_digest(factor)
        + claimed_sum.to_bytes(8, "little")
    ).digest()


@dataclass(frozen=True, slots=True)
class GoldilocksProductSumcheckProofV3:
    """Degree-3 round polynomials (4 evaluations) + both full openings."""

    claimed_sum: int
    round_polynomials: tuple[tuple[int, int, int, int], ...]
    a_full_opening: tuple[int, ...]
    b_full_opening: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "claimed_sum", _field(self.claimed_sum, "claimed_sum")
        )
        if not isinstance(self.round_polynomials, tuple) or not all(
            isinstance(item, tuple) and len(item) == 4
            for item in self.round_polynomials
        ):
            raise ProofV3Error("product-sumcheck round set is malformed")


def prove_goldilocks_product_sumcheck_v3(
    *,
    statement_digest: bytes,
    a_tree: GoldilocksMerkleTreeReference,
    b_tree: GoldilocksMerkleTreeReference,
    a_evaluations: tuple[int, ...],
    b_evaluations: tuple[int, ...],
    factor: tuple[int, ...],
    validator_nonce: bytes,
) -> GoldilocksProductSumcheckProofV3:
    if not len(a_evaluations) == len(b_evaluations) == len(factor):
        raise ProofV3Error("product-sumcheck table lengths mismatch")
    a_values = [_field(v, "a value") for v in a_evaluations]
    b_values = [_field(v, "b value") for v in b_evaluations]
    f_values = [_field(v, "factor value") for v in factor]
    claimed = 0
    for a, b, f in zip(a_values, b_values, f_values, strict=True):
        claimed = (claimed + a * b % GOLDILOCKS_MODULUS * f) % GOLDILOCKS_MODULUS
    transcript = _seed(
        statement_digest=statement_digest,
        a_commitment=a_tree.commitment,
        b_commitment=b_tree.commitment,
        validator_nonce=validator_nonce,
        factor=tuple(f_values),
        claimed_sum=claimed,
    )
    rounds: list[tuple[int, int, int, int]] = []
    while len(a_values) > 1:
        half = len(a_values) // 2
        evaluations = [0, 0, 0, 0]
        for i in range(half):
            lows = (a_values[i], b_values[i], f_values[i])
            highs = (a_values[half + i], b_values[half + i], f_values[half + i])
            for z in range(4):
                term = 1
                for low, high in zip(lows, highs, strict=True):
                    term = term * ((low + z * (high - low)) % GOLDILOCKS_MODULUS)
                    term %= GOLDILOCKS_MODULUS
                evaluations[z] = (evaluations[z] + term) % GOLDILOCKS_MODULUS
        rounds.append(tuple(evaluations))
        transcript = hashlib.sha256(
            transcript
            + b"".join(value.to_bytes(8, "little") for value in evaluations)
        ).digest()
        challenge = _challenge(transcript, len(rounds))
        a_values = _mle_fold(a_values, challenge)
        b_values = _mle_fold(b_values, challenge)
        f_values = _mle_fold(f_values, challenge)
    return GoldilocksProductSumcheckProofV3(
        claimed_sum=claimed,
        round_polynomials=tuple(rounds),
        a_full_opening=tuple(row[0] for row in a_tree.rows),
        b_full_opening=tuple(row[0] for row in b_tree.rows),
    )


def _lagrange_eval_0123(
    evaluations: tuple[int, int, int, int], z: int
) -> int:
    """Evaluate the degree-3 polynomial from values at 0,1,2,3."""

    result = 0
    points = (0, 1, 2, 3)
    for i, value in enumerate(evaluations):
        numerator, denominator = 1, 1
        for j, point in enumerate(points):
            if i == j:
                continue
            numerator = numerator * ((z - point) % GOLDILOCKS_MODULUS)
            numerator %= GOLDILOCKS_MODULUS
            denominator = denominator * ((points[i] - point) % GOLDILOCKS_MODULUS)
            denominator %= GOLDILOCKS_MODULUS
        result = (
            result
            + value * numerator % GOLDILOCKS_MODULUS
            * pow(denominator, GOLDILOCKS_MODULUS - 2, GOLDILOCKS_MODULUS)
        ) % GOLDILOCKS_MODULUS
    return result


def verify_goldilocks_product_sumcheck_v3(
    proof: object,
    *,
    statement_digest: bytes,
    a_commitment: bytes,
    b_commitment: bytes,
    factor: tuple[int, ...],
    validator_nonce: bytes,
    expected_sum: int,
) -> None:
    try:
        if not isinstance(proof, GoldilocksProductSumcheckProofV3):
            raise ProofV3VerificationError(
                "product-sumcheck proof type is unexpected"
            )
        f_values = tuple(_field(v, "factor value") for v in factor)
        length = len(f_values)
        if length < 2 or length & (length - 1):
            raise ProofV3VerificationError(
                "product-sumcheck factor length is invalid"
            )
        variables = length.bit_length() - 1
        if len(proof.round_polynomials) != variables:
            raise ProofV3VerificationError("product-sumcheck round count is wrong")
        if proof.claimed_sum != _field(expected_sum, "expected_sum"):
            raise ProofV3VerificationError(
                "product-sumcheck claimed sum does not match the outer relation"
            )
        transcript = _seed(
            statement_digest=statement_digest,
            a_commitment=_fixed32(a_commitment, "a_commitment"),
            b_commitment=_fixed32(b_commitment, "b_commitment"),
            validator_nonce=validator_nonce,
            factor=f_values,
            claimed_sum=proof.claimed_sum,
        )
        running = proof.claimed_sum
        point: list[int] = []
        for round_index, evaluations in enumerate(proof.round_polynomials, 1):
            values = tuple(_field(v, "round value") for v in evaluations)
            if (values[0] + values[1]) % GOLDILOCKS_MODULUS != running:
                raise ProofV3VerificationError(
                    "product-sumcheck round does not match the running sum"
                )
            transcript = hashlib.sha256(
                transcript
                + b"".join(value.to_bytes(8, "little") for value in values)
            ).digest()
            challenge = _challenge(transcript, round_index)
            point.append(challenge)
            running = _lagrange_eval_0123(values, challenge)
        for name, opening, commitment, domain in (
            ("a", proof.a_full_opening, a_commitment, _A_COMMIT_DOMAIN),
            ("b", proof.b_full_opening, b_commitment, _B_COMMIT_DOMAIN),
        ):
            if len(opening) != length:
                raise ProofV3VerificationError(
                    f"product-sumcheck {name} opening length is wrong"
                )
            rebuilt = _commit(
                statement_digest=statement_digest,
                evaluations=tuple(opening),
                domain=domain,
            )
            if rebuilt.commitment != commitment:
                raise ProofV3VerificationError(
                    f"product-sumcheck {name} opening does not match its root"
                )
        a_at = _mle_eval(
            tuple(_field(v, "a value") for v in proof.a_full_opening),
            tuple(point),
        )
        b_at = _mle_eval(
            tuple(_field(v, "b value") for v in proof.b_full_opening),
            tuple(point),
        )
        f_at = _mle_eval(f_values, tuple(point))
        if running != a_at * b_at % GOLDILOCKS_MODULUS * f_at % GOLDILOCKS_MODULUS:
            raise ProofV3VerificationError(
                "product-sumcheck final evaluation does not match the commitments"
            )
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "product-sumcheck proof is malformed"
        ) from exc


__all__ = [
    "GOLDILOCKS_PRODUCT_SUMCHECK_ABI_V3",
    "GoldilocksProductSumcheckProofV3",
    "commit_goldilocks_product_sumcheck_a_v3",
    "commit_goldilocks_product_sumcheck_b_v3",
    "prove_goldilocks_product_sumcheck_v3",
    "verify_goldilocks_product_sumcheck_v3",
]
