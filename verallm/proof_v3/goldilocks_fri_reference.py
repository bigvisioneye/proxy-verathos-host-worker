"""Bounded CPU Goldilocks FRI reference for future proof-v3 conformance.

This module is deliberately unregistered.  It is a small, single-polynomial
reference implementation for testing a future GPU-native dynamic backend; it
is not imported by a miner or validator, has no wire payload, and does not
prove an AIR, RAM relation, model execution, or model-substitution resistance.

The reference fixes one canonical FRI transcript over the existing
Goldilocks-field and Merkle-reference ABIs:

* a statement fixes a coset, strict degree bound, query count, and external
  32-byte binding;
* every FRI layer is committed as one width-one Goldilocks Merkle tree with a
  distinct statement/round/domain binding;
* Fiat--Shamir folding challenges are derived after the corresponding layer
  commitment, and query bases are derived only after all layer commitments;
* each opening uses the exact Merkle multiproof schedule, and the final small
  codeword is opened completely and required to be constant.

It intentionally materializes all vectors and Merkle trees and is capped at
``2**16`` source evaluations.  That makes it suitable for golden vectors and
native prover/verifier conformance only.  A qualified backend must stream its
oracles on device, authenticate an actual composition polynomial, and bind it
to the signed execution relation before this primitive could contribute to a
hard proof.
"""

from __future__ import annotations

import hashlib
import operator
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_merkle_reference import (
    GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES,
    GoldilocksMerkleMultiOpeningReference,
    GoldilocksMerkleTreeReference,
    verify_goldilocks_merkle_multiopening_reference,
)
from verallm.proof_v3.goldilocks_reference import (
    GOLDILOCKS_MODULUS,
    GOLDILOCKS_TWO_ADICITY,
    MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE,
    GoldilocksRadix2DomainReference,
    canonical_goldilocks,
    goldilocks_inv,
    goldilocks_radix2_domain_reference,
)


GOLDILOCKS_FRI_REFERENCE_ABI_V3: Final = "goldilocks.fri.reference.v1"
GOLDILOCKS_FRI_REFERENCE_FORMAT_VERSION_V3: Final = 1
MAX_GOLDILOCKS_FRI_REFERENCE_QUERY_COUNT_V3: Final = 64
MAX_GOLDILOCKS_FRI_REFERENCE_FINAL_DOMAIN_SIZE_V3: Final = 64
MAX_GOLDILOCKS_FRI_REFERENCE_ROUNDS_V3: Final = GOLDILOCKS_TWO_ADICITY

_STATEMENT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_FRI/V1/STATEMENT/SHA256"
_ROUND_BINDING_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_FRI/V1/ROUND_BINDING/SHA256"
)
_BETA_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_FRI/V1/BETA/SHA256"
_QUERY_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_FRI/V1/QUERY/SHA256"
_MAX_REJECTION_ATTEMPTS: Final = 1 << 16


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ProofV3Error(f"{name} must be an integer, not boolean")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ProofV3Error(f"{name} must be an integer") from exc


def _fixed32(value: object, name: str, *, nonzero: bool = False) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ProofV3Error(f"{name} must be bytes")
    result = bytes(value)
    if len(result) != GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    if nonzero and result == bytes(len(result)):
        raise ProofV3Error(f"{name} must not be zero")
    return result


def _power_of_two(
    value: object,
    name: str,
    *,
    maximum: int,
) -> int:
    integer = _integer(value, name)
    if integer < 1 or integer > maximum or integer & (integer - 1):
        raise ProofV3Error(f"{name} must be a power of two within the reference cap")
    return integer


def _field_vector(value: object, *, length: int, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        raise ProofV3Error(f"{name} must be an iterable of canonical field values")
    try:
        entries = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ProofV3Error(
            f"{name} must be an iterable of canonical field values"
        ) from exc
    if len(entries) != length:
        raise ProofV3Error(f"{name} must contain exactly {length} field values")
    return tuple(
        canonical_goldilocks(entry, f"{name}[{index}]")
        for index, entry in enumerate(entries)
    )


@dataclass(frozen=True, slots=True)
class GoldilocksFriStatementReference:
    """Canonical single-polynomial FRI statement for CPU conformance only.

    ``degree_bound`` is strict: the claimed source polynomial has degree less
    than that power-of-two value.  The final codeword therefore has
    ``domain_size // degree_bound`` rows and is opened in full by this bounded
    reference.
    """

    binding_digest: bytes
    domain_size: int
    degree_bound: int
    domain_shift: int = 1
    query_count: int = 16
    abi_id: str = GOLDILOCKS_FRI_REFERENCE_ABI_V3
    format_version: int = GOLDILOCKS_FRI_REFERENCE_FORMAT_VERSION_V3

    def __post_init__(self) -> None:
        if self.abi_id != GOLDILOCKS_FRI_REFERENCE_ABI_V3:
            raise ProofV3Error("Goldilocks FRI reference ABI is unsupported")
        if type(self.format_version) is not int or self.format_version != (
            GOLDILOCKS_FRI_REFERENCE_FORMAT_VERSION_V3
        ):
            raise ProofV3Error("Goldilocks FRI reference format version is unsupported")
        binding = _fixed32(
            self.binding_digest,
            "Goldilocks FRI statement binding_digest",
            nonzero=True,
        )
        domain_size = _power_of_two(
            self.domain_size,
            "Goldilocks FRI domain_size",
            maximum=MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE,
        )
        degree_bound = _power_of_two(
            self.degree_bound,
            "Goldilocks FRI degree_bound",
            maximum=domain_size,
        )
        if degree_bound >= domain_size:
            raise ProofV3Error(
                "Goldilocks FRI degree_bound must leave a nontrivial code rate"
            )
        final_domain_size = domain_size // degree_bound
        if final_domain_size > MAX_GOLDILOCKS_FRI_REFERENCE_FINAL_DOMAIN_SIZE_V3:
            raise ProofV3Error(
                "Goldilocks FRI final domain exceeds the CPU reference cap"
            )
        query_count = _integer(self.query_count, "Goldilocks FRI query_count")
        if (
            query_count < 1
            or query_count > MAX_GOLDILOCKS_FRI_REFERENCE_QUERY_COUNT_V3
            or query_count > domain_size // 2
        ):
            raise ProofV3Error("Goldilocks FRI query_count is outside the reference cap")
        shift = canonical_goldilocks(
            self.domain_shift,
            "Goldilocks FRI domain_shift",
        )
        if shift == 0:
            raise ProofV3Error("Goldilocks FRI domain_shift must be nonzero")
        object.__setattr__(self, "binding_digest", binding)
        object.__setattr__(self, "domain_size", domain_size)
        object.__setattr__(self, "degree_bound", degree_bound)
        object.__setattr__(self, "domain_shift", shift)
        object.__setattr__(self, "query_count", query_count)

    @property
    def round_count(self) -> int:
        """Return the exact number of folds needed for a constant remainder."""

        return self.degree_bound.bit_length() - 1

    @property
    def final_domain_size(self) -> int:
        return self.domain_size // self.degree_bound

    def canonical_bytes(self) -> bytes:
        """Return the sole statement encoding used by the reference transcript."""

        return (
            struct.pack("<HH", self.format_version, len(self.abi_id))
            + self.abi_id.encode("ascii")
            + self.binding_digest
            + struct.pack(
                "<IIQI",
                self.domain_size,
                self.degree_bound,
                self.domain_shift,
                self.query_count,
            )
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_STATEMENT_DOMAIN + self.canonical_bytes()).digest()


def _round_domain(
    statement: GoldilocksFriStatementReference,
    *,
    round_index: int,
) -> GoldilocksRadix2DomainReference:
    if round_index < 0 or round_index > statement.round_count:
        raise ProofV3Error("Goldilocks FRI round index is out of range")
    return goldilocks_radix2_domain_reference(
        size=statement.domain_size >> round_index,
        shift=pow(statement.domain_shift, 1 << round_index, GOLDILOCKS_MODULUS),
    )


def _round_binding(
    statement: GoldilocksFriStatementReference,
    *,
    round_index: int,
) -> bytes:
    domain = _round_domain(statement, round_index=round_index)
    return hashlib.sha256(
        _ROUND_BINDING_DOMAIN
        + statement.digest()
        + struct.pack("<IIQ", round_index, domain.size, domain.shift)
    ).digest()


def _commitment_bytes(value: object, name: str) -> bytes:
    return _fixed32(value, name)


def _derive_beta(
    statement: GoldilocksFriStatementReference,
    *,
    commitments: tuple[bytes, ...],
    round_index: int,
) -> int:
    if len(commitments) != round_index + 1:
        raise ProofV3Error("Goldilocks FRI beta transcript has an unexpected commitment count")
    normalized = tuple(
        _commitment_bytes(item, f"Goldilocks FRI commitment[{index}]")
        for index, item in enumerate(commitments)
    )
    prefix = (
        _BETA_DOMAIN
        + statement.digest()
        + struct.pack("<II", round_index, len(normalized))
        + b"".join(normalized)
    )
    for counter in range(_MAX_REJECTION_ATTEMPTS):
        candidate = int.from_bytes(
            hashlib.sha256(prefix + struct.pack("<I", counter)).digest()[:8],
            "little",
        )
        if 0 < candidate < GOLDILOCKS_MODULUS:
            return candidate
    raise ProofV3Error("unable to derive a canonical Goldilocks FRI beta")


def _derive_query_bases(
    statement: GoldilocksFriStatementReference,
    *,
    commitments: tuple[bytes, ...],
) -> tuple[int, ...]:
    expected_count = statement.round_count + 1
    if len(commitments) != expected_count:
        raise ProofV3Error("Goldilocks FRI query transcript has an unexpected commitment count")
    normalized = tuple(
        _commitment_bytes(item, f"Goldilocks FRI commitment[{index}]")
        for index, item in enumerate(commitments)
    )
    limit = statement.domain_size // 2
    threshold = (1 << 64) - ((1 << 64) % limit)
    prefix = (
        _QUERY_DOMAIN
        + statement.digest()
        + struct.pack("<II", statement.query_count, len(normalized))
        + b"".join(normalized)
    )
    result: set[int] = set()
    for counter in range(_MAX_REJECTION_ATTEMPTS):
        candidate = int.from_bytes(
            hashlib.sha256(prefix + struct.pack("<I", counter)).digest()[:8],
            "little",
        )
        if candidate >= threshold:
            continue
        result.add(candidate % limit)
        if len(result) == statement.query_count:
            return tuple(sorted(result))
    raise ProofV3Error("unable to derive distinct Goldilocks FRI query bases")


def _fold_layer(
    values: tuple[int, ...],
    *,
    domain: GoldilocksRadix2DomainReference,
    beta: int,
) -> tuple[int, ...]:
    if len(values) != domain.size or domain.size < 2:
        raise ProofV3Error("Goldilocks FRI fold has an unexpected source domain")
    challenge = canonical_goldilocks(beta, "Goldilocks FRI beta")
    if challenge == 0:
        raise ProofV3Error("Goldilocks FRI beta must be nonzero")
    points = domain.points()
    half = domain.size // 2
    inverse_two = goldilocks_inv(2)
    result: list[int] = []
    for index in range(half):
        positive = values[index]
        negative = values[index + half]
        even = (positive + negative) * inverse_two % GOLDILOCKS_MODULUS
        odd = (
            (positive - negative)
            * inverse_two
            % GOLDILOCKS_MODULUS
            * goldilocks_inv(points[index])
            % GOLDILOCKS_MODULUS
        )
        result.append((even + challenge * odd) % GOLDILOCKS_MODULUS)
    return tuple(result)


def _pair_indices(
    statement: GoldilocksFriStatementReference,
    *,
    query_bases: tuple[int, ...],
    round_index: int,
) -> tuple[int, ...]:
    if round_index < 0 or round_index >= statement.round_count:
        raise ProofV3Error("Goldilocks FRI query round is out of range")
    half = statement.domain_size >> (round_index + 1)
    bases = tuple(sorted({base % half for base in query_bases}))
    return tuple(sorted((*bases, *(base + half for base in bases))))


@dataclass(frozen=True, slots=True)
class GoldilocksFriProofReference:
    """In-memory FRI proof object for conformance vectors only.

    This is intentionally not a network payload.  The verifier takes its
    statement as a separate validator-owned argument and derives all expected
    bindings, roots, indices, and folding challenges itself.
    """

    commitments: tuple[bytes, ...]
    round_openings: tuple[GoldilocksMerkleMultiOpeningReference, ...]
    final_opening: GoldilocksMerkleMultiOpeningReference
    abi_id: str = GOLDILOCKS_FRI_REFERENCE_ABI_V3
    format_version: int = GOLDILOCKS_FRI_REFERENCE_FORMAT_VERSION_V3

    def __post_init__(self) -> None:
        if self.abi_id != GOLDILOCKS_FRI_REFERENCE_ABI_V3:
            raise ProofV3Error("Goldilocks FRI proof ABI is unsupported")
        if type(self.format_version) is not int or self.format_version != (
            GOLDILOCKS_FRI_REFERENCE_FORMAT_VERSION_V3
        ):
            raise ProofV3Error("Goldilocks FRI proof format version is unsupported")
        if not isinstance(self.commitments, tuple) or not self.commitments:
            raise ProofV3Error("Goldilocks FRI proof commitments must be nonempty")
        if len(self.commitments) > MAX_GOLDILOCKS_FRI_REFERENCE_ROUNDS_V3 + 1:
            raise ProofV3Error("Goldilocks FRI proof has too many commitments")
        commitments = tuple(
            _commitment_bytes(item, f"Goldilocks FRI proof commitment[{index}]")
            for index, item in enumerate(self.commitments)
        )
        if not isinstance(self.round_openings, tuple) or not all(
            isinstance(item, GoldilocksMerkleMultiOpeningReference)
            for item in self.round_openings
        ):
            raise ProofV3Error("Goldilocks FRI proof round openings are malformed")
        if not isinstance(self.final_opening, GoldilocksMerkleMultiOpeningReference):
            raise ProofV3Error("Goldilocks FRI proof final opening is malformed")
        object.__setattr__(self, "commitments", commitments)


def prove_goldilocks_fri_reference(
    evaluations: object,
    *,
    statement: GoldilocksFriStatementReference,
) -> GoldilocksFriProofReference:
    """Build one bounded, deterministic FRI conformance proof.

    The caller may supply arbitrary evaluations.  A non-low-degree vector can
    still be committed and folded, but verification will reject it unless it
    happens to satisfy the Fiat--Shamir FRI checks.  This behavior is useful
    for adversarial golden vectors; it is not a substitute for a real prover
    that first constructs an execution composition polynomial.
    """

    if not isinstance(statement, GoldilocksFriStatementReference):
        raise ProofV3Error("Goldilocks FRI statement has an unexpected type")
    current = _field_vector(
        evaluations,
        length=statement.domain_size,
        name="Goldilocks FRI source evaluations",
    )
    trees: list[GoldilocksMerkleTreeReference] = []
    commitments: list[bytes] = []
    for round_index in range(statement.round_count + 1):
        tree = GoldilocksMerkleTreeReference.from_rows(
            tuple((value,) for value in current),
            binding_digest=_round_binding(statement, round_index=round_index),
        )
        trees.append(tree)
        commitments.append(tree.commitment)
        if round_index == statement.round_count:
            break
        beta = _derive_beta(
            statement,
            commitments=tuple(commitments),
            round_index=round_index,
        )
        current = _fold_layer(
            current,
            domain=_round_domain(statement, round_index=round_index),
            beta=beta,
        )
    normalized_commitments = tuple(commitments)
    query_bases = _derive_query_bases(
        statement,
        commitments=normalized_commitments,
    )
    round_openings = tuple(
        trees[round_index].open(
            _pair_indices(
                statement,
                query_bases=query_bases,
                round_index=round_index,
            )
        )
        for round_index in range(statement.round_count)
    )
    final_opening = trees[-1].open(tuple(range(statement.final_domain_size)))
    return GoldilocksFriProofReference(
        commitments=normalized_commitments,
        round_openings=round_openings,
        final_opening=final_opening,
    )


def _opened_values(
    opening: GoldilocksMerkleMultiOpeningReference,
) -> dict[int, int]:
    if opening.leaf_width != 1:
        raise ProofV3VerificationError("Goldilocks FRI opening has an unexpected width")
    return {
        index: row[0]
        for index, row in zip(opening.indices, opening.rows, strict=True)
    }


def verify_goldilocks_fri_reference(
    proof: object,
    *,
    statement: GoldilocksFriStatementReference,
) -> None:
    """Verify a bounded FRI conformance proof against validator-owned state."""

    if not isinstance(statement, GoldilocksFriStatementReference):
        raise ProofV3VerificationError("Goldilocks FRI statement has an unexpected type")
    if not isinstance(proof, GoldilocksFriProofReference):
        raise ProofV3VerificationError("Goldilocks FRI proof has an unexpected type")
    expected_commitment_count = statement.round_count + 1
    if len(proof.commitments) != expected_commitment_count:
        raise ProofV3VerificationError(
            "Goldilocks FRI proof has an unexpected commitment count"
        )
    if len(proof.round_openings) != statement.round_count:
        raise ProofV3VerificationError(
            "Goldilocks FRI proof has an unexpected round-opening count"
        )
    try:
        query_bases = _derive_query_bases(
            statement,
            commitments=proof.commitments,
        )
        for round_index, opening in enumerate(proof.round_openings):
            domain = _round_domain(statement, round_index=round_index)
            verify_goldilocks_merkle_multiopening_reference(
                proof.commitments[round_index],
                opening,
                expected_binding_digest=_round_binding(
                    statement,
                    round_index=round_index,
                ),
                expected_leaf_count=domain.size,
                expected_leaf_width=1,
                expected_indices=_pair_indices(
                    statement,
                    query_bases=query_bases,
                    round_index=round_index,
                ),
            )
        verify_goldilocks_merkle_multiopening_reference(
            proof.commitments[-1],
            proof.final_opening,
            expected_binding_digest=_round_binding(
                statement,
                round_index=statement.round_count,
            ),
            expected_leaf_count=statement.final_domain_size,
            expected_leaf_width=1,
            expected_indices=tuple(range(statement.final_domain_size)),
        )
    except ProofV3VerificationError:
        raise
    except ProofV3Error as exc:
        raise ProofV3VerificationError("Goldilocks FRI proof is malformed") from exc

    opened_rounds = tuple(_opened_values(opening) for opening in proof.round_openings)
    final_values = _opened_values(proof.final_opening)
    for round_index, source_values in enumerate(opened_rounds):
        beta = _derive_beta(
            statement,
            commitments=proof.commitments[: round_index + 1],
            round_index=round_index,
        )
        domain = _round_domain(statement, round_index=round_index)
        points = domain.points()
        half = domain.size // 2
        inverse_two = goldilocks_inv(2)
        target_values = (
            final_values
            if round_index + 1 == statement.round_count
            else opened_rounds[round_index + 1]
        )
        for base in (index for index in source_values if index < half):
            positive = source_values.get(base)
            negative = source_values.get(base + half)
            if positive is None or negative is None:
                raise ProofV3VerificationError(
                    "Goldilocks FRI opening omits a folded pair"
                )
            expected = (
                (positive + negative) * inverse_two
                + beta
                * (
                    (positive - negative)
                    * inverse_two
                    % GOLDILOCKS_MODULUS
                    * goldilocks_inv(points[base])
                    % GOLDILOCKS_MODULUS
                )
            ) % GOLDILOCKS_MODULUS
            target_index = base % half
            actual = target_values.get(target_index)
            if actual is None or actual != expected:
                raise ProofV3VerificationError(
                    "Goldilocks FRI folding relation does not hold"
                )
    if set(final_values) != set(range(statement.final_domain_size)):
        raise ProofV3VerificationError("Goldilocks FRI final opening is incomplete")
    if len(set(final_values.values())) != 1:
        raise ProofV3VerificationError(
            "Goldilocks FRI final codeword is not constant"
        )


__all__ = [
    "GOLDILOCKS_FRI_REFERENCE_ABI_V3",
    "GOLDILOCKS_FRI_REFERENCE_FORMAT_VERSION_V3",
    "MAX_GOLDILOCKS_FRI_REFERENCE_FINAL_DOMAIN_SIZE_V3",
    "MAX_GOLDILOCKS_FRI_REFERENCE_QUERY_COUNT_V3",
    "MAX_GOLDILOCKS_FRI_REFERENCE_ROUNDS_V3",
    "GoldilocksFriProofReference",
    "GoldilocksFriStatementReference",
    "prove_goldilocks_fri_reference",
    "verify_goldilocks_fri_reference",
]
