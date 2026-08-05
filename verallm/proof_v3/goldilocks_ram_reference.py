"""Field-consistent logical KV/cache RAM reference over Goldilocks.

Port of the LogUp-RAM shape from ``ram_reference.py`` (Pallas, v2) into the
same field as the v3 AIR stack, so cache reads can become in-relation
constraints instead of cross-field bridges.

Model: offline memory checking (Blum-style) over one logical table.  The
frozen access log is a sequence of writes and reads with strictly
increasing counters.  Every access contributes:

* to the read multiset: the tuple it observed ``(address, prev_value,
  prev_counter)``;
* to the write multiset: the tuple it produced ``(address, value,
  counter)``.

Initial zero-writes at counter 0 seed every touched address; a final
audit read of the terminal state closes every address.  RAM consistency
is exactly the multiset equality ``reads + finals == writes + inits``
plus per-access counter monotonicity (``prev_counter < counter``).

Multiset equality is checked with the dual-challenge LogUp identity over
fingerprints ``address + gamma * value + gamma^2 * counter`` — two
independent ``(alpha, gamma)`` pairs derived post-freeze (single-pair
soundness ~(accesses)/2^64 is not enough in a 64-bit field; the pair
doubling matches the FRI/lookup posture).

Reference verification opens the log in full, checks monotonicity and
tuple wellformedness directly, and evaluates both rational identities.
The production backend keeps the identical fingerprint/challenge ABI but
proves the sums with committed inverse columns inside the AIR (and the
generalized fold sumcheck), never opening the log.
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
    _u32,
)
from verallm.proof_v3.goldilocks_merkle_reference import (
    GoldilocksMerkleTreeReference,
)
from verallm.proof_v3.goldilocks_reference import (
    GOLDILOCKS_MODULUS,
    goldilocks_inv,
)


GOLDILOCKS_RAM_ABI_V3: Final = "goldilocks.ram.reference.v1"
GOLDILOCKS_RAM_CHALLENGE_COUNT_V3: Final = 2
MAX_GOLDILOCKS_RAM_ACCESSES_V3: Final = 1 << 18

_STATEMENT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_RAM/V1/STATEMENT/SHA256"
_LOG_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_RAM/V1/ACCESS_LOG/SHA256"
_CHALLENGE_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_RAM/V1/CHALLENGE/SHA256"

_OP_WRITE: Final = 0
_OP_READ: Final = 1


def _field(value: object, name: str) -> int:
    integer = _integer(value, name)
    if not 0 <= integer < GOLDILOCKS_MODULUS:
        raise ProofV3Error(f"{name} must be a canonical Goldilocks element")
    return integer


@dataclass(frozen=True, slots=True)
class GoldilocksRamStatementV3:
    """Validator-owned RAM statement: address space and access budget."""

    validator_binding_digest: bytes
    address_count: int
    max_accesses: int

    def __post_init__(self) -> None:
        binding = _fixed32(
            self.validator_binding_digest,
            "ram validator_binding_digest",
            nonzero=True,
        )
        addresses = _u32(self.address_count, "address_count", positive=True)
        accesses = _u32(self.max_accesses, "max_accesses", positive=True)
        if accesses > MAX_GOLDILOCKS_RAM_ACCESSES_V3:
            raise ProofV3Error("ram access budget exceeds the CPU reference cap")
        object.__setattr__(self, "validator_binding_digest", binding)
        object.__setattr__(self, "address_count", addresses)
        object.__setattr__(self, "max_accesses", accesses)

    def digest(self) -> bytes:
        return hashlib.sha256(
            _STATEMENT_DOMAIN
            + self.validator_binding_digest
            + struct.pack("<II", self.address_count, self.max_accesses)
        ).digest()

    def log_binding_digest(self) -> bytes:
        return hashlib.sha256(_LOG_DOMAIN + self.digest()).digest()


@dataclass(frozen=True, slots=True)
class GoldilocksRamAccessV3:
    """One access row: (op, address, value, prev_value, prev_counter)."""

    op: int
    address: int
    value: int
    prev_value: int
    prev_counter: int

    def __post_init__(self) -> None:
        if self.op not in (_OP_WRITE, _OP_READ):
            raise ProofV3Error("ram access op is unsupported")
        _integer(self.address, "ram address")
        _field(self.value, "ram value")
        _field(self.prev_value, "ram prev_value")
        _integer(self.prev_counter, "ram prev_counter")

    def row(self) -> tuple[int, int, int, int, int]:
        return (
            self.op,
            self.address,
            self.value,
            self.prev_value,
            self.prev_counter,
        )


def freeze_goldilocks_ram_log_v3(
    *,
    statement: GoldilocksRamStatementV3,
    accesses: tuple[GoldilocksRamAccessV3, ...],
) -> GoldilocksMerkleTreeReference:
    """Freeze the padded access log pre-nonce.

    Shapes are validated; RAM *consistency* is deliberately not checked —
    an inconsistent log must fail verification, not construction.
    """

    if not isinstance(statement, GoldilocksRamStatementV3):
        raise ProofV3Error("ram statement has an unexpected type")
    if not isinstance(accesses, tuple) or not accesses:
        raise ProofV3Error("ram access log must be a non-empty tuple")
    if len(accesses) > statement.max_accesses:
        raise ProofV3Error("ram access log exceeds the signed budget")
    rows = []
    for index, access in enumerate(accesses):
        if not isinstance(access, GoldilocksRamAccessV3):
            raise ProofV3Error("ram access has an unexpected type")
        if access.address >= statement.address_count:
            raise ProofV3Error("ram access address is out of the signed space")
        rows.append(access.row())
    padded = 1 << max(1, (len(rows) - 1).bit_length())
    # Padding rows are inert no-op writes of the zero state at address 0
    # counter 0; the verifier skips rows whose op field is outside the
    # real-access range marker (op == 2).
    rows.extend((2, 0, 0, 0, 0) for _ in range(padded - len(rows)))
    return GoldilocksMerkleTreeReference.from_rows(
        tuple(rows),
        binding_digest=statement.log_binding_digest(),
    )


def _challenges(
    *,
    statement: GoldilocksRamStatementV3,
    log_root: bytes,
    validator_nonce: bytes,
) -> tuple[tuple[int, int], ...]:
    seed = hashlib.sha256(
        _CHALLENGE_DOMAIN
        + statement.digest()
        + _fixed32(log_root, "log_root")
        + _fixed32(validator_nonce, "validator_nonce")
    ).digest()
    pairs: list[tuple[int, int]] = []
    for pair_index in range(GOLDILOCKS_RAM_CHALLENGE_COUNT_V3):
        values: list[int] = []
        for part in range(2):
            for counter in range(1 << 16):
                candidate = int.from_bytes(
                    hashlib.sha256(
                        seed + struct.pack("<III", pair_index, part, counter)
                    ).digest()[:8],
                    "little",
                )
                if 0 < candidate < GOLDILOCKS_MODULUS:
                    values.append(candidate)
                    break
            else:
                raise ProofV3Error("unable to derive a ram challenge")
        pairs.append((values[0], values[1]))
    return tuple(pairs)


def _fingerprint(
    *,
    address: int,
    value: int,
    counter: int,
    gamma: int,
) -> int:
    return (
        address + gamma * value + gamma * gamma % GOLDILOCKS_MODULUS * counter
    ) % GOLDILOCKS_MODULUS


@dataclass(frozen=True, slots=True)
class GoldilocksRamProofV3:
    """Reference proof: the full opened access log."""

    log_opening: tuple[tuple[int, int, int, int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.log_opening, tuple) or not self.log_opening:
            raise ProofV3Error("ram log opening is malformed")


def prove_goldilocks_ram_reference_v3(
    *,
    log_tree: GoldilocksMerkleTreeReference,
) -> GoldilocksRamProofV3:
    return GoldilocksRamProofV3(
        log_opening=tuple(tuple(row) for row in log_tree.rows),
    )


def verify_goldilocks_ram_reference_v3(
    proof: object,
    *,
    statement: GoldilocksRamStatementV3,
    log_root: bytes,
    validator_nonce: bytes,
) -> None:
    """Verify RAM consistency of the frozen access log.

    Checks, in order: the opening matches the frozen root; access
    wellformedness (ops, addresses, counter monotonicity per access and
    strictly increasing global counters); reads observe their claimed
    previous tuples only through the multiset identity; and the dual
    LogUp fingerprint identities ``reads + finals == writes + inits``.
    """

    try:
        if not isinstance(proof, GoldilocksRamProofV3):
            raise ProofV3VerificationError("ram proof type is unexpected")
        if not isinstance(statement, GoldilocksRamStatementV3):
            raise ProofV3VerificationError("ram statement is malformed")
        rows = proof.log_opening
        rebuilt = GoldilocksMerkleTreeReference.from_rows(
            tuple(tuple(row) for row in rows),
            binding_digest=statement.log_binding_digest(),
        )
        if rebuilt.commitment != log_root:
            raise ProofV3VerificationError(
                "ram log opening does not match the frozen root"
            )
        touched: dict[int, tuple[int, int]] = {}
        read_tuples: list[tuple[int, int, int]] = []
        write_tuples: list[tuple[int, int, int]] = []
        init_tuples: list[tuple[int, int, int]] = []
        in_padding = False
        for counter, row in enumerate(rows, start=1):
            op, address, value, prev_value, prev_counter = (
                _integer(part, "ram log field") for part in row
            )
            if op == 2:
                in_padding = True
                if (address, value, prev_value, prev_counter) != (0, 0, 0, 0):
                    raise ProofV3VerificationError("ram padding row is malformed")
                continue
            if in_padding:
                raise ProofV3VerificationError(
                    "ram real access appears after padding"
                )
            if op not in (_OP_WRITE, _OP_READ):
                raise ProofV3VerificationError("ram access op is unsupported")
            if not 0 <= address < statement.address_count:
                raise ProofV3VerificationError("ram address is out of range")
            _field(value, "ram value")
            _field(prev_value, "ram prev_value")
            if prev_counter >= counter or prev_counter < 0:
                raise ProofV3VerificationError(
                    "ram access counter is not monotone"
                )
            if address not in touched:
                if prev_counter != 0 or prev_value != 0:
                    raise ProofV3VerificationError(
                        "ram first access must observe the zero init state"
                    )
                init_tuples.append((address, 0, 0))
            read_tuples.append((address, prev_value, prev_counter))
            new_value = value if op == _OP_WRITE else prev_value
            if op == _OP_READ and value != prev_value:
                raise ProofV3VerificationError(
                    "ram read must return its observed previous value"
                )
            write_tuples.append((address, new_value, counter))
            touched[address] = (new_value, counter)
        if not touched:
            raise ProofV3VerificationError("ram log has no real accesses")
        final_tuples = [
            (address, value, counter)
            for address, (value, counter) in sorted(touched.items())
        ]
        for alpha, gamma in _challenges(
            statement=statement,
            log_root=log_root,
            validator_nonce=validator_nonce,
        ):
            def side(tuples: list[tuple[int, int, int]]) -> int:
                total = 0
                for address, value, counter in tuples:
                    denominator = (
                        alpha
                        - _fingerprint(
                            address=address,
                            value=value,
                            counter=counter,
                            gamma=gamma,
                        )
                    ) % GOLDILOCKS_MODULUS
                    total = (
                        total + goldilocks_inv(denominator)
                    ) % GOLDILOCKS_MODULUS
                return total

            left = (side(read_tuples) + side(final_tuples)) % GOLDILOCKS_MODULUS
            right = (side(write_tuples) + side(init_tuples)) % GOLDILOCKS_MODULUS
            if left != right:
                raise ProofV3VerificationError(
                    "ram multiset identity fails: log is not consistent"
                )
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("ram proof is malformed") from exc


__all__ = [
    "GOLDILOCKS_RAM_ABI_V3",
    "GoldilocksRamAccessV3",
    "GoldilocksRamProofV3",
    "GoldilocksRamStatementV3",
    "freeze_goldilocks_ram_log_v3",
    "prove_goldilocks_ram_reference_v3",
    "verify_goldilocks_ram_reference_v3",
]
