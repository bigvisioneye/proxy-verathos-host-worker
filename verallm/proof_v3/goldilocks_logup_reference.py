"""Dual-challenge LogUp lookup reference over Goldilocks for proof-v3.

Membership argument: every element of a frozen witness column ``L`` must
belong to a public table ``T`` (byte ranges, exp/GDN nonlinearity tables).
The LogUp identity is checked for two independently derived post-freeze
challenges::

    sum_i 1 / (alpha - L[i])  ==  sum_j m[j] / (alpha - T[j])

``m`` is the multiplicity vector, committed pre-nonce together with ``L``.
A single Goldilocks challenge gives only ~(|L|+|T|)/2^64 soundness, so two
independent challenges are mandatory (error multiplies: ~2^-88 at 2^20
rows); this is the same doubled-batch posture as the FRI references.

Reference verification opens ``L`` and ``m`` in full and recomputes both
rational sums exactly (zero denominators hard-fail).  The production
backend replaces the full openings with committed inverse columns whose
product constraints ``inv * (alpha - value) == 1`` are batch-verified by
the generalized fold sumcheck (two committed tables, degree-3 rounds) —
the freeze order, challenge derivation, multiplicity binding, and domain
separation fixed here are exactly the ABI that backend must reproduce.

Chronology: commit ``L`` and ``m`` (both Merkle trees, distinct bindings)
before the validator nonce; both challenges derive from the statement, the
two roots, and the nonce.  A miner never chooses table contents: ``T`` is
part of the validator-owned statement.
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
from verallm.proof_v3.goldilocks_reference import (
    GOLDILOCKS_MODULUS,
    goldilocks_inv,
)


GOLDILOCKS_LOGUP_ABI_V3: Final = "goldilocks.logup.reference.v1"
GOLDILOCKS_LOGUP_CHALLENGE_COUNT_V3: Final = 2
MAX_GOLDILOCKS_LOGUP_ROWS_V3: Final = 1 << 20

_STATEMENT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_LOGUP/V1/STATEMENT/SHA256"
_WITNESS_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_LOGUP/V1/WITNESS/SHA256"
_MULTIPLICITY_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_LOGUP/V1/MULTIPLICITY/SHA256"
)
_CHALLENGE_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_LOGUP/V1/CHALLENGE/SHA256"


def _field(value: object, name: str) -> int:
    integer = _integer(value, name)
    if not 0 <= integer < GOLDILOCKS_MODULUS:
        raise ProofV3Error(f"{name} must be a canonical Goldilocks element")
    return integer


@dataclass(frozen=True, slots=True)
class GoldilocksLogupStatementV3:
    """Validator-owned lookup statement: the exact public table."""

    validator_binding_digest: bytes
    table: tuple[int, ...]

    def __post_init__(self) -> None:
        binding = _fixed32(
            self.validator_binding_digest,
            "logup validator_binding_digest",
            nonzero=True,
        )
        if not isinstance(self.table, tuple) or not self.table:
            raise ProofV3Error("logup table must be a non-empty tuple")
        if len(self.table) > MAX_GOLDILOCKS_LOGUP_ROWS_V3:
            raise ProofV3Error("logup table exceeds the CPU reference cap")
        table = tuple(_field(value, "logup table value") for value in self.table)
        if len(set(table)) != len(table):
            raise ProofV3Error("logup table values must be distinct")
        object.__setattr__(self, "validator_binding_digest", binding)
        object.__setattr__(self, "table", table)

    def digest(self) -> bytes:
        return hashlib.sha256(
            _STATEMENT_DOMAIN
            + self.validator_binding_digest
            + struct.pack("<I", len(self.table))
            + b"".join(value.to_bytes(8, "little") for value in self.table)
        ).digest()

    def witness_binding_digest(self) -> bytes:
        return hashlib.sha256(_WITNESS_DOMAIN + self.digest()).digest()

    def multiplicity_binding_digest(self) -> bytes:
        return hashlib.sha256(_MULTIPLICITY_DOMAIN + self.digest()).digest()


def freeze_goldilocks_logup_witness_v3(
    *,
    statement: GoldilocksLogupStatementV3,
    looked_up_values: tuple[int, ...],
) -> tuple[GoldilocksMerkleTreeReference, GoldilocksMerkleTreeReference]:
    """Freeze the witness column and its multiplicities pre-nonce.

    Multiplicities are derived from the witness, not caller-supplied.  The
    helper deliberately does not require membership: a non-member witness
    must fail proof verification, not construction (its multiplicity row
    simply never covers it).
    """

    if not isinstance(statement, GoldilocksLogupStatementV3):
        raise ProofV3Error("logup statement has an unexpected type")
    if not isinstance(looked_up_values, tuple) or not looked_up_values:
        raise ProofV3Error("logup witness must be a non-empty tuple")
    if len(looked_up_values) > MAX_GOLDILOCKS_LOGUP_ROWS_V3:
        raise ProofV3Error("logup witness exceeds the CPU reference cap")
    witness = tuple(_field(value, "logup witness value") for value in looked_up_values)
    # The Merkle reference requires power-of-two leaf counts.  Witness
    # padding repeats table[0] (a legitimate member, counted normally);
    # multiplicity padding is zero and must verify as zero.
    padded_length = 1 << max(1, (len(witness) - 1).bit_length())
    witness = witness + (statement.table[0],) * (padded_length - len(witness))
    positions = {value: index for index, value in enumerate(statement.table)}
    counts = [0] * len(statement.table)
    for value in witness:
        index = positions.get(value)
        if index is not None:
            counts[index] += 1
    counts_length = 1 << max(1, (len(counts) - 1).bit_length())
    counts.extend(0 for _ in range(counts_length - len(counts)))
    witness_tree = GoldilocksMerkleTreeReference.from_rows(
        tuple((value,) for value in witness),
        binding_digest=statement.witness_binding_digest(),
    )
    multiplicity_tree = GoldilocksMerkleTreeReference.from_rows(
        tuple((value,) for value in counts),
        binding_digest=statement.multiplicity_binding_digest(),
    )
    return witness_tree, multiplicity_tree


def _challenges(
    *,
    statement: GoldilocksLogupStatementV3,
    witness_root: bytes,
    multiplicity_root: bytes,
    validator_nonce: bytes,
) -> tuple[int, ...]:
    seed = hashlib.sha256(
        _CHALLENGE_DOMAIN
        + statement.digest()
        + _fixed32(witness_root, "witness_root")
        + _fixed32(multiplicity_root, "multiplicity_root")
        + _fixed32(validator_nonce, "validator_nonce")
    ).digest()
    challenges: list[int] = []
    for challenge_index in range(GOLDILOCKS_LOGUP_CHALLENGE_COUNT_V3):
        for counter in range(1 << 16):
            candidate = int.from_bytes(
                hashlib.sha256(
                    seed + struct.pack("<II", challenge_index, counter)
                ).digest()[:8],
                "little",
            )
            if 0 < candidate < GOLDILOCKS_MODULUS:
                challenges.append(candidate)
                break
        else:
            raise ProofV3Error("unable to derive a logup challenge")
    return tuple(challenges)


@dataclass(frozen=True, slots=True)
class GoldilocksLogupProofV3:
    """Reference proof: full witness and multiplicity openings."""

    witness_opening: tuple[int, ...]
    multiplicity_opening: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, opening in (
            ("witness", self.witness_opening),
            ("multiplicity", self.multiplicity_opening),
        ):
            if not isinstance(opening, tuple) or not opening:
                raise ProofV3Error(f"logup {name} opening is malformed")


def prove_goldilocks_logup_reference_v3(
    *,
    witness_tree: GoldilocksMerkleTreeReference,
    multiplicity_tree: GoldilocksMerkleTreeReference,
) -> GoldilocksLogupProofV3:
    return GoldilocksLogupProofV3(
        witness_opening=tuple(row[0] for row in witness_tree.rows),
        multiplicity_opening=tuple(row[0] for row in multiplicity_tree.rows),
    )


def verify_goldilocks_logup_reference_v3(
    proof: object,
    *,
    statement: GoldilocksLogupStatementV3,
    witness_root: bytes,
    multiplicity_root: bytes,
    validator_nonce: bytes,
) -> None:
    """Verify both LogUp identities against the frozen roots.

    The reference recomputes both trees from the full openings and both
    rational sums for each of the two independent challenges.  A zero
    denominator (challenge collides with a value) is a hard failure, never
    a skip.
    """

    try:
        if not isinstance(proof, GoldilocksLogupProofV3):
            raise ProofV3VerificationError("logup proof type is unexpected")
        if not isinstance(statement, GoldilocksLogupStatementV3):
            raise ProofV3VerificationError("logup statement is malformed")
        witness = tuple(
            _field(value, "logup witness value")
            for value in proof.witness_opening
        )
        counts = tuple(
            _field(value, "logup multiplicity value")
            for value in proof.multiplicity_opening
        )
        table_length = len(statement.table)
        expected_counts = 1 << max(1, (table_length - 1).bit_length())
        if len(counts) != expected_counts:
            raise ProofV3VerificationError("logup multiplicity shape is wrong")
        if any(count != 0 for count in counts[table_length:]):
            raise ProofV3VerificationError(
                "logup multiplicity padding must be zero"
            )
        rebuilt_witness = GoldilocksMerkleTreeReference.from_rows(
            tuple((value,) for value in witness),
            binding_digest=statement.witness_binding_digest(),
        )
        if rebuilt_witness.commitment != witness_root:
            raise ProofV3VerificationError(
                "logup witness opening does not match the frozen root"
            )
        rebuilt_counts = GoldilocksMerkleTreeReference.from_rows(
            tuple((value,) for value in counts),
            binding_digest=statement.multiplicity_binding_digest(),
        )
        if rebuilt_counts.commitment != multiplicity_root:
            raise ProofV3VerificationError(
                "logup multiplicity opening does not match the frozen root"
            )
        # Multiplicities are counts, not arbitrary field elements: bound
        # them by the witness length so wrap-around forgeries are excluded.
        if any(count > len(witness) for count in counts):
            raise ProofV3VerificationError("logup multiplicity is out of range")
        for alpha in _challenges(
            statement=statement,
            witness_root=witness_root,
            multiplicity_root=multiplicity_root,
            validator_nonce=validator_nonce,
        ):
            left = 0
            for value in witness:
                denominator = (alpha - value) % GOLDILOCKS_MODULUS
                left = (left + goldilocks_inv(denominator)) % GOLDILOCKS_MODULUS
            right = 0
            for count, value in zip(
                counts[:table_length], statement.table, strict=True
            ):
                denominator = (alpha - value) % GOLDILOCKS_MODULUS
                right = (
                    right + count * goldilocks_inv(denominator)
                ) % GOLDILOCKS_MODULUS
            if left != right:
                raise ProofV3VerificationError(
                    "logup identity fails: witness is not covered by the table"
                )
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("logup proof is malformed") from exc


__all__ = [
    "GOLDILOCKS_LOGUP_ABI_V3",
    "GoldilocksLogupProofV3",
    "GoldilocksLogupStatementV3",
    "freeze_goldilocks_logup_witness_v3",
    "prove_goldilocks_logup_reference_v3",
    "verify_goldilocks_logup_reference_v3",
]
