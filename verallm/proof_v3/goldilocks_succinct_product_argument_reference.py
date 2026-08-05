"""Succinct two-column product argument: sum A*B*F with PCS terminals.

Proves ``sum_i A[i] * B[i] * F[i] == S`` where A and B are PCS-committed
columns and F is a TENSOR-STRUCTURED public factor (eq tables, nonce
tensor coefficients). Degree-3 rounds; at the terminal point the
verifier needs A~ and B~ (PCS evaluation openings against the SHARED
column commitments) and evaluates F~ itself from the components in
O(sum |F_j|).

This is the succinct replacement for the full-opening product sumcheck
in the tile verticals; combined with the succinct fold argument it
expresses every per-cell tile relation as an eq-weighted zero-check.
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
from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
    GoldilocksMultilinearOpeningProofV3,
    GoldilocksMultilinearPcsStatementV3,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS

GOLDILOCKS_SUCCINCT_PRODUCT_ABI_V3: Final = (
    "goldilocks.succinct_product.reference.v1"
)
_STATEMENT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_SUCCINCT_PRODUCT/V1/STATEMENT"
)
_TRANSCRIPT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_SUCCINCT_PRODUCT/V1/TRANSCRIPT"
)
_CHALLENGE_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_SUCCINCT_PRODUCT/V1/CHALLENGE"
)


def _field(value: object, name: str) -> int:
    integer = _integer(value, name)
    if not 0 <= integer < GOLDILOCKS_MODULUS:
        raise ProofV3Error(f"{name} must be a canonical Goldilocks element")
    return integer


def _derive(seed: bytes, index: int) -> int:
    for counter in range(1 << 16):
        candidate = int.from_bytes(
            hashlib.sha256(
                _CHALLENGE_DOMAIN + seed + struct.pack("<II", index, counter)
            ).digest()[:8],
            "little",
        )
        if candidate < GOLDILOCKS_MODULUS:
            return candidate
    raise ProofV3Error("unable to derive a succinct-product challenge")


@dataclass(frozen=True, slots=True)
class GoldilocksSuccinctProductStatementV3:
    """Shape + the PCS statements the two columns were committed under."""

    validator_binding_digest: bytes
    variable_count: int
    factor_component_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        binding = _fixed32(
            self.validator_binding_digest, "succinct-product binding",
            nonzero=True,
        )
        variables = _integer(self.variable_count, "variable_count")
        sizes = tuple(
            _integer(s, "factor component size")
            for s in self.factor_component_sizes
        )
        if variables < 1 or any(s < 1 or s & (s - 1) for s in sizes):
            raise ProofV3Error("succinct-product shape is malformed")
        total_bits = sum(s.bit_length() - 1 for s in sizes)
        if total_bits != variables:
            raise ProofV3Error(
                "succinct-product factor components do not span the cube"
            )
        object.__setattr__(self, "validator_binding_digest", binding)
        object.__setattr__(self, "variable_count", variables)
        object.__setattr__(self, "factor_component_sizes", sizes)

    def digest(self) -> bytes:
        return hashlib.sha256(
            _STATEMENT_DOMAIN
            + self.validator_binding_digest
            + struct.pack("<I", self.variable_count)
            + b"".join(
                struct.pack("<I", s) for s in self.factor_component_sizes
            )
        ).digest()


@dataclass(frozen=True, slots=True)
class GoldilocksSuccinctProductProofV3:
    claimed_sum: int
    round_polynomials: tuple[tuple[int, int, int, int], ...]
    a_opening: GoldilocksMultilinearOpeningProofV3
    b_opening: GoldilocksMultilinearOpeningProofV3


def _tensor_factor(components: tuple[tuple[int, ...], ...]) -> list[int]:
    factor = [1]
    for component in components:
        factor = [
            f * value % GOLDILOCKS_MODULUS
            for f in factor
            for value in component
        ]
    return factor


def _mle_eval_msb(values, point) -> int:
    work = list(values)
    for r in point:
        half = len(work) // 2
        work = [
            (work[i] + r * (work[half + i] - work[i])) % GOLDILOCKS_MODULUS
            for i in range(half)
        ]
    return work[0]


def _seed(statement, a_commitment, b_commitment, factor_digest, claimed,
          validator_nonce) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + statement.digest()
        + a_commitment
        + b_commitment
        + factor_digest
        + claimed.to_bytes(8, "little")
        + _fixed32(validator_nonce, "validator_nonce")
    ).digest()


def _factor_digest(components) -> bytes:
    return hashlib.sha256(
        b"".join(
            struct.pack("<I", len(component))
            + b"".join(v.to_bytes(8, "little") for v in component)
            for component in components
        )
    ).digest()


def prove_goldilocks_succinct_product_v3(
    *,
    statement: GoldilocksSuccinctProductStatementV3,
    a_pcs_statement: GoldilocksMultilinearPcsStatementV3,
    b_pcs_statement: GoldilocksMultilinearPcsStatementV3,
    a_tree,
    b_tree,
    a_evaluations: tuple[int, ...],
    b_evaluations: tuple[int, ...],
    factor_components: tuple[tuple[int, ...], ...],
    validator_nonce: bytes,
    open_fn=None,
    collector=None,
    a_tag: str | None = None,
    b_tag: str | None = None,
    a_point_map: tuple[int, ...] | None = None,
    b_point_map: tuple[int, ...] | None = None,
) -> GoldilocksSuccinctProductProofV3:
    """CPU reference prover; ``open_fn`` may supply a fused PCS opener.

    ``a_point_map``/``b_point_map``: broadcast-free mode.  When set, the
    corresponding factor is the broadcast of a SMALLER committed column
    onto the product cube (constant in the unmapped variables), and its
    terminal claim opens the small column at the mapped sub-point
    (LSB-first indices into the terminal point).  This is the classic
    overlapping-subsets product sumcheck: folding a broadcast variable
    leaves the factor unchanged, so the terminal value equals the small
    column's MLE at the mapped coordinates."""

    from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
        open_goldilocks_multilinear_v3,
    )

    components = tuple(
        tuple(_field(v, "factor value") for v in component)
        for component in factor_components
    )
    if tuple(len(c) for c in components) != statement.factor_component_sizes:
        raise ProofV3Error("succinct-product factor shapes are wrong")
    a_values = [_field(v, "a value") for v in a_evaluations]
    b_values = [_field(v, "b value") for v in b_evaluations]
    f_values = _tensor_factor(components)
    if not len(a_values) == len(b_values) == len(f_values):
        raise ProofV3Error("succinct-product column lengths mismatch")
    claimed = 0
    for a, b, f in zip(a_values, b_values, f_values, strict=True):
        claimed = (
            claimed + a * b % GOLDILOCKS_MODULUS * f
        ) % GOLDILOCKS_MODULUS
    transcript = _seed(
        statement, a_tree.commitment, b_tree.commitment,
        _factor_digest(components), claimed, validator_nonce,
    )
    rounds: list[tuple[int, int, int, int]] = []
    challenges: list[int] = []
    while len(a_values) > 1:
        half = len(a_values) // 2
        evals = [0, 0, 0, 0]
        for i in range(half):
            lows = (a_values[i], b_values[i], f_values[i])
            highs = (a_values[half + i], b_values[half + i], f_values[half + i])
            for z in range(4):
                term = 1
                for low, high in zip(lows, highs, strict=True):
                    term = term * ((low + z * (high - low)) % GOLDILOCKS_MODULUS)
                    term %= GOLDILOCKS_MODULUS
                evals[z] = (evals[z] + term) % GOLDILOCKS_MODULUS
        rounds.append(tuple(evals))
        transcript = hashlib.sha256(
            transcript
            + b"".join(value.to_bytes(8, "little") for value in evals)
        ).digest()
        challenge = _derive(transcript, len(rounds))
        challenges.append(challenge)
        fold = lambda values: [
            (values[i] + challenge * (values[half + i] - values[i]))
            % GOLDILOCKS_MODULUS
            for i in range(half)
        ]
        a_values, b_values, f_values = fold(a_values), fold(b_values), fold(
            f_values
        )
    point = tuple(reversed(challenges))
    a_point = (
        point if a_point_map is None
        else tuple(point[i] for i in a_point_map))
    b_point = (
        point if b_point_map is None
        else tuple(point[i] for i in b_point_map))
    if collector is not None:
        return GoldilocksSuccinctProductProofV3(
            claimed_sum=claimed,
            round_polynomials=tuple(rounds),
            a_opening=collector.defer(a_tag, a_point, a_values[0]),
            b_opening=collector.defer(b_tag, b_point, b_values[0]),
        )
    opener = open_fn or (
        lambda pcs_statement, tree, evaluations: open_goldilocks_multilinear_v3(
            statement=pcs_statement, tree=tree, evaluations=evaluations,
            point=point, validator_nonce=validator_nonce,
        )
    )
    if a_point_map is not None or b_point_map is not None:
        raise ProofV3Error(
            "broadcast-free product claims require a deferred collector")
    return GoldilocksSuccinctProductProofV3(
        claimed_sum=claimed,
        round_polynomials=tuple(rounds),
        a_opening=opener(a_pcs_statement, a_tree, tuple(a_evaluations)),
        b_opening=opener(b_pcs_statement, b_tree, tuple(b_evaluations)),
    )


def verify_goldilocks_succinct_product_v3(
    proof: object,
    *,
    statement: GoldilocksSuccinctProductStatementV3,
    a_pcs_statement: GoldilocksMultilinearPcsStatementV3,
    b_pcs_statement: GoldilocksMultilinearPcsStatementV3,
    a_commitment: bytes,
    b_commitment: bytes,
    factor_components: tuple[tuple[int, ...], ...],
    validator_nonce: bytes,
    expected_sum: int,
    checker=None,
    a_tag: str | None = None,
    b_tag: str | None = None,
    a_point_map: tuple[int, ...] | None = None,
    b_point_map: tuple[int, ...] | None = None,
) -> None:
    """O(q log N) verify: rounds + two PCS openings + factor MLE."""

    from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
        verify_goldilocks_multilinear_opening_v3,
    )
    from verallm.proof_v3.goldilocks_reference import goldilocks_inv

    try:
        if not isinstance(proof, GoldilocksSuccinctProductProofV3):
            raise ProofV3VerificationError(
                "succinct-product proof type is wrong"
            )
        components = tuple(
            tuple(_field(v, "factor value") for v in component)
            for component in factor_components
        )
        if tuple(len(c) for c in components) != (
            statement.factor_component_sizes
        ):
            raise ProofV3VerificationError(
                "succinct-product factor shapes are wrong"
            )
        if proof.claimed_sum != _field(expected_sum, "expected_sum"):
            raise ProofV3VerificationError(
                "succinct-product claimed sum does not match the relation"
            )
        n = statement.variable_count
        if len(proof.round_polynomials) != n:
            raise ProofV3VerificationError(
                "succinct-product round count is wrong"
            )
        transcript = _seed(
            statement,
            _fixed32(a_commitment, "a_commitment"),
            _fixed32(b_commitment, "b_commitment"),
            _factor_digest(components),
            proof.claimed_sum,
            validator_nonce,
        )
        running = proof.claimed_sum
        challenges: list[int] = []
        compiled = None
        try:
            from verallm.proof_v3.c_multiopen import replay_rounds4

            compiled = replay_rounds4(
                transcript, running,
                tuple(tuple(int(v) for v in row)
                      for row in proof.round_polynomials),
                _CHALLENGE_DOMAIN, b"", False, 1)
        except ImportError:
            compiled = None
        if isinstance(compiled, tuple):
            challenges_t, running, transcript = compiled
            challenges = list(challenges_t)
        elif isinstance(compiled, int):
            raise ProofV3VerificationError(
                "succinct-product round replay fails")
        inv6 = goldilocks_inv(6)
        inv2 = goldilocks_inv(2)
        for evals in (
            () if compiled is not None and not isinstance(compiled, int)
            else proof.round_polynomials
        ):
            evals = tuple(_field(v, "round evaluation") for v in evals)
            if (evals[0] + evals[1]) % GOLDILOCKS_MODULUS != running:
                raise ProofV3VerificationError(
                    "succinct-product round does not match the running sum"
                )
            transcript = hashlib.sha256(
                transcript
                + b"".join(value.to_bytes(8, "little") for value in evals)
            ).digest()
            challenge = _derive(transcript, len(challenges) + 1)
            challenges.append(challenge)
            z = challenge
            zm1 = (z - 1) % GOLDILOCKS_MODULUS
            zm2 = (z - 2) % GOLDILOCKS_MODULUS
            zm3 = (z - 3) % GOLDILOCKS_MODULUS
            running = (
                evals[0] * (zm1 * zm2 % GOLDILOCKS_MODULUS * zm3
                            % GOLDILOCKS_MODULUS) % GOLDILOCKS_MODULUS
                * (GOLDILOCKS_MODULUS - inv6)
                + evals[1] * (z * zm2 % GOLDILOCKS_MODULUS * zm3
                              % GOLDILOCKS_MODULUS) % GOLDILOCKS_MODULUS
                * inv2
                + evals[2] * (z * zm1 % GOLDILOCKS_MODULUS * zm3
                              % GOLDILOCKS_MODULUS) % GOLDILOCKS_MODULUS
                * (GOLDILOCKS_MODULUS - inv2)
                + evals[3] * (z * zm1 % GOLDILOCKS_MODULUS * zm2
                              % GOLDILOCKS_MODULUS) % GOLDILOCKS_MODULUS
                * inv6
            ) % GOLDILOCKS_MODULUS
        # F~ at the terminal point from the tensor components (MSB slices)
        cursor = 0
        f_at_point = 1
        for component in components:
            bits = len(component).bit_length() - 1
            slice_point = tuple(challenges[cursor: cursor + bits])
            cursor += bits
            f_at_point = (
                f_at_point * _mle_eval_msb(component, slice_point)
            ) % GOLDILOCKS_MODULUS
        expected = (
            proof.a_opening.claimed_value
            * proof.b_opening.claimed_value
            % GOLDILOCKS_MODULUS
            * f_at_point
            % GOLDILOCKS_MODULUS
        )
        if running != expected:
            raise ProofV3VerificationError(
                "succinct-product terminal coupling fails"
            )
        point = tuple(reversed(challenges))
        a_point = (
            point if a_point_map is None
            else tuple(point[i] for i in a_point_map))
        b_point = (
            point if b_point_map is None
            else tuple(point[i] for i in b_point_map))
        if checker is not None:
            checker.expect(a_tag, a_point, proof.a_opening.claimed_value)
            checker.expect(b_tag, b_point, proof.b_opening.claimed_value)
        elif a_point_map is not None or b_point_map is not None:
            raise ProofV3VerificationError(
                "broadcast-free product claims require a deferred checker")
        else:
            verify_goldilocks_multilinear_opening_v3(
                proof.a_opening, statement=a_pcs_statement,
                commitment=a_commitment, point=point,
                expected_value=proof.a_opening.claimed_value,
                validator_nonce=validator_nonce,
            )
            verify_goldilocks_multilinear_opening_v3(
                proof.b_opening, statement=b_pcs_statement,
                commitment=b_commitment, point=point,
                expected_value=proof.b_opening.claimed_value,
                validator_nonce=validator_nonce,
            )
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "succinct-product proof is malformed"
        ) from exc
