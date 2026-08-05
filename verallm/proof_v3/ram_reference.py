"""Bounded Pallas LogUp RAM reference for a future proof-v3 adapter.

This module is deliberately *not* registered with a miner, validator, payload,
or execution profile.  It proves one narrow relation over one power-of-two
Pallas-field table:

* a verifier-owned, pre-nonce source-commitment inventory selects one source
  table;
* that source commitment is exactly the cache-write commitment, under the same
  deterministic Pallas PCS key and the same logical layout; and
* a frozen logical read schedule reads the selected ``(address, value)`` pairs
  from that table.

The source inventory, its digest, the logical schedule, and an external
pre-nonce envelope digest are all part of the precommit transcript.  The
caller must retain and authenticate that precommit before revealing the
validator nonce.  This module cannot turn a local function call into a
network-timing guarantee.

The inverse witnesses are intentionally committed only after the nonce-derived
lookup encoding is known, but before *any* Fiat--Shamir sumcheck challenge is
derived.  Every physical row receives an inverse constraint, including padded
read rows and source rows whose multiplicity is zero.  A zero denominator is a
hard failure; there is no zero-target or inactive-row exception.

This is a compact CPU reference, capped by the existing PCS vector limit.  It
does not establish a Q/K/V projection, RoPE, attention-core recurrence,
residual transition, or returned token.  In particular, it is not evidence of
model-substitution resistance until a qualified adapter binds those relations
to the authenticated execution trace.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from zkllm.crypto.pcs_v2 import (
    COMMITMENT_BYTES,
    ENCODING_PALLAS_SCALAR,
    MAX_VALIDATE_COMMITMENTS,
    MAX_VECTOR_LEN,
    PALLAS_SCALAR_MODULUS,
    PCSNativeError,
    PCSOpeningV2,
    PCSUnavailableError,
    commit,
    encode_scalar,
    prove,
    scalar_to_int,
    validate_commitments,
    verify,
)

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError


PALLAS_LOGUP_RAM_REFERENCE_ABI_V3: Final = "pallas.logup_ram.reference.v1"
PALLAS_LOGUP_RAM_KEY_ABI_V3: Final = "pallas_pedersen.mle.v2"

_STATEMENT_DOMAIN: Final = b"VERATHOS/PROOF_V3/LOGUP_RAM/STATEMENT/SHA256"
_SOURCE_INVENTORY_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/LOGUP_RAM/SOURCE_INVENTORY/SHA256"
)
_PRECOMMIT_DOMAIN: Final = b"VERATHOS/PROOF_V3/LOGUP_RAM/PRECOMMIT/SHA256"
_TRANSCRIPT_INIT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/LOGUP_RAM/TRANSCRIPT_INIT/SHA256"
)
_TRANSCRIPT_ABSORB_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/LOGUP_RAM/TRANSCRIPT_ABSORB/SHA256"
)
_TRANSCRIPT_SCALAR_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/LOGUP_RAM/TRANSCRIPT_SCALAR/SHA256"
)
_OPENING_DOMAIN: Final = b"VERATHOS/PROOF_V3/LOGUP_RAM/OPENING/SHA256"
_MAX_REJECTION_ATTEMPTS: Final = 4096


def _fixed_bytes(value: object, length: int, name: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ProofV3Error(f"{name} must be bytes")
    result = bytes(value)
    if len(result) != length:
        raise ProofV3Error(f"{name} must be exactly {length} bytes")
    return result


def _fixed32(value: object, name: str) -> bytes:
    return _fixed_bytes(value, 32, name)


def _fixed_commitment(value: object, name: str) -> bytes:
    return _fixed_bytes(value, COMMITMENT_BYTES, name)


def _u32(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProofV3Error(f"{name} must be an unsigned 32-bit integer")
    if value < (1 if positive else 0) or value >= 1 << 32:
        qualifier = "positive " if positive else ""
        raise ProofV3Error(f"{name} must be a {qualifier}unsigned 32-bit integer")
    return value


def _field(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProofV3Error(f"{name} must be a canonical Pallas scalar")
    if value < 0 or value >= PALLAS_SCALAR_MODULUS:
        raise ProofV3Error(f"{name} is outside the Pallas scalar field")
    return value


def _vector_length(value: object) -> int:
    length = _u32(value, "RAM vector_length", positive=True)
    if length < 2 or length > MAX_VECTOR_LEN or length & (length - 1):
        raise ProofV3Error(
            "RAM vector_length must be a power of two in 2..=PCS capacity"
        )
    return length


def _field_vector(values: object, length: int, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray, memoryview)):
        raise ProofV3Error(f"{name} must be an iterable of Pallas scalars")
    try:
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ProofV3Error(f"{name} must be an iterable of Pallas scalars") from exc
    if len(result) != length:
        raise ProofV3Error(f"{name} must contain exactly {length} field rows")
    return tuple(_field(value, f"{name}[{index}]") for index, value in enumerate(result))


def _commitment_inventory(value: object) -> tuple[bytes, ...]:
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        raise ProofV3Error("source commitment inventory must be an iterable")
    try:
        inventory = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ProofV3Error("source commitment inventory must be an iterable") from exc
    if not inventory or len(inventory) > MAX_VALIDATE_COMMITMENTS:
        raise ProofV3Error("source commitment inventory has an invalid size")
    return tuple(
        _fixed_commitment(item, f"source commitment {index}")
        for index, item in enumerate(inventory)
    )


def source_inventory_digest_v3(source_commitments: object) -> bytes:
    """Return the canonical digest of the complete ordered source inventory.

    This digest says nothing about the semantic authorization of the inventory.
    The caller must obtain that authorization from the signed model/profile
    artifact, then put this exact digest in :class:`LogUpRamStatementV3` before
    the nonce is revealed.
    """

    inventory = _commitment_inventory(source_commitments)
    return hashlib.sha256(
        _SOURCE_INVENTORY_DOMAIN
        + struct.pack("<I", len(inventory))
        + b"".join(inventory)
    ).digest()


@dataclass(frozen=True, slots=True)
class LogUpRamStatementV3:
    """Verifier-owned immutable RAM statement and logical read schedule.

    ``precommit_envelope_digest`` must be the digest of the externally
    authenticated envelope that was retained before nonce reveal.  The miner
    must not originate this value.  ``source_inventory_digest`` similarly
    belongs to the verifier/signed-profile side; :func:`freeze_logup_ram_precommit_v3`
    checks it against the complete supplied inventory.

    ``read_active`` permits canonical padding.  Its histogram is always
    recomputed from ``read_addresses`` and this mask; no prover-supplied count
    vector exists in the protocol.
    """

    precommit_envelope_digest: bytes
    source_inventory_digest: bytes
    source_inventory_index: int
    vector_length: int
    read_addresses: tuple[int, ...]
    read_active: tuple[int, ...]
    commitment_key_abi_id: str = PALLAS_LOGUP_RAM_KEY_ABI_V3

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "precommit_envelope_digest",
            _fixed32(self.precommit_envelope_digest, "RAM precommit_envelope_digest"),
        )
        object.__setattr__(
            self,
            "source_inventory_digest",
            _fixed32(self.source_inventory_digest, "RAM source_inventory_digest"),
        )
        object.__setattr__(
            self,
            "source_inventory_index",
            _u32(self.source_inventory_index, "RAM source_inventory_index"),
        )
        length = _vector_length(self.vector_length)
        if self.commitment_key_abi_id != PALLAS_LOGUP_RAM_KEY_ABI_V3:
            raise ProofV3Error("RAM commitment key ABI is unsupported")
        try:
            addresses = tuple(self.read_addresses)
            active = tuple(self.read_active)
        except TypeError as exc:
            raise ProofV3Error("RAM read schedule must be iterable") from exc
        if len(addresses) != length or len(active) != length:
            raise ProofV3Error("RAM read schedule must have exactly vector_length rows")
        normalized_addresses: list[int] = []
        normalized_active: list[int] = []
        for index, (address, is_active) in enumerate(zip(addresses, active, strict=True)):
            address = _u32(address, f"RAM read address {index}")
            if address >= length:
                raise ProofV3Error("RAM read address exceeds the source table")
            if isinstance(is_active, bool) or not isinstance(is_active, int):
                raise ProofV3Error("RAM read activity must be a 0/1 integer")
            if is_active not in (0, 1):
                raise ProofV3Error("RAM read activity must be a 0/1 integer")
            normalized_addresses.append(address)
            normalized_active.append(is_active)
        if not any(normalized_active):
            raise ProofV3Error("RAM schedule must contain at least one active read")
        object.__setattr__(self, "vector_length", length)
        object.__setattr__(self, "read_addresses", tuple(normalized_addresses))
        object.__setattr__(self, "read_active", tuple(normalized_active))

    def canonical_bytes(self) -> bytes:
        abi = PALLAS_LOGUP_RAM_REFERENCE_ABI_V3.encode("ascii")
        key_abi = self.commitment_key_abi_id.encode("ascii")
        schedule = b"".join(
            struct.pack("<IB", address, active)
            for address, active in zip(self.read_addresses, self.read_active, strict=True)
        )
        return (
            struct.pack("<B", len(abi))
            + abi
            + struct.pack("<B", len(key_abi))
            + key_abi
            + self.precommit_envelope_digest
            + self.source_inventory_digest
            + struct.pack("<II", self.source_inventory_index, self.vector_length)
            + schedule
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_STATEMENT_DOMAIN + self.canonical_bytes()).digest()

    def read_histogram(self) -> tuple[int, ...]:
        """Return the validator-derived source-table multiplicity vector."""

        histogram = [0] * self.vector_length
        for address, active in zip(self.read_addresses, self.read_active, strict=True):
            histogram[address] += active
        return tuple(histogram)


@dataclass(frozen=True, slots=True)
class FrozenLogUpRamPrecommitV3:
    """The exact commitments retained by the verifier before the nonce.

    ``cache_write_commitment`` is intentionally required to equal the selected
    source commitment byte-for-byte.  That is valid only for the same Pallas
    commitment key and identical field layout; a reshape or quantization bridge
    needs a separately qualified relation and cannot use this shortcut.
    """

    statement_digest: bytes
    source_commitments: tuple[bytes, ...]
    cache_write_commitment: bytes
    cache_read_commitment: bytes

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "statement_digest",
            _fixed32(self.statement_digest, "RAM precommit statement_digest"),
        )
        object.__setattr__(
            self,
            "source_commitments",
            _commitment_inventory(self.source_commitments),
        )
        object.__setattr__(
            self,
            "cache_write_commitment",
            _fixed_commitment(self.cache_write_commitment, "RAM cache_write_commitment"),
        )
        object.__setattr__(
            self,
            "cache_read_commitment",
            _fixed_commitment(self.cache_read_commitment, "RAM cache_read_commitment"),
        )

    @property
    def source_inventory_digest(self) -> bytes:
        return source_inventory_digest_v3(self.source_commitments)

    def canonical_bytes(self) -> bytes:
        return (
            self.statement_digest
            + struct.pack("<I", len(self.source_commitments))
            + b"".join(self.source_commitments)
            + self.cache_write_commitment
            + self.cache_read_commitment
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_PRECOMMIT_DOMAIN + self.canonical_bytes()).digest()


def _validate_commitments(
    commitments: tuple[bytes, ...],
    *,
    error_type: type[ProofV3Error],
) -> None:
    try:
        for start in range(0, len(commitments), MAX_VALIDATE_COMMITMENTS):
            validate_commitments(
                commitments[start : start + MAX_VALIDATE_COMMITMENTS]
            )
    except (PCSNativeError, PCSUnavailableError, ValueError, TypeError) as exc:
        raise error_type("RAM precommit contains a noncanonical Pallas commitment") from exc


def _validate_precommit(
    statement: LogUpRamStatementV3,
    precommit: FrozenLogUpRamPrecommitV3,
    *,
    error_type: type[ProofV3Error],
) -> None:
    if not isinstance(statement, LogUpRamStatementV3):
        raise error_type("RAM statement has an unexpected type")
    if not isinstance(precommit, FrozenLogUpRamPrecommitV3):
        raise error_type("RAM precommit has an unexpected type")
    if precommit.statement_digest != statement.digest():
        raise error_type("RAM precommit belongs to a different statement")
    if precommit.source_inventory_digest != statement.source_inventory_digest:
        raise error_type("RAM source inventory does not match the verifier digest")
    if statement.source_inventory_index >= len(precommit.source_commitments):
        raise error_type("RAM selected source commitment is outside the inventory")
    if (
        precommit.source_commitments[statement.source_inventory_index]
        != precommit.cache_write_commitment
    ):
        raise error_type(
            "RAM source and cache-write commitments are not the same key/layout"
        )
    _validate_commitments(
        precommit.source_commitments
        + (precommit.cache_write_commitment, precommit.cache_read_commitment),
        error_type=error_type,
    )


def freeze_logup_ram_precommit_v3(
    *,
    statement: LogUpRamStatementV3,
    source_commitments: object,
    cache_write_commitment: bytes,
    cache_read_commitment: bytes,
) -> FrozenLogUpRamPrecommitV3:
    """Validate and freeze the complete pre-nonce RAM commitment inventory.

    The returned object must be included in (or committed by) the serving
    protocol's pre-nonce envelope.  Calling this helper after a nonce reveal
    does not satisfy that external timing requirement.
    """

    if not isinstance(statement, LogUpRamStatementV3):
        raise ProofV3Error("RAM statement has an unexpected type")
    precommit = FrozenLogUpRamPrecommitV3(
        statement_digest=statement.digest(),
        source_commitments=_commitment_inventory(source_commitments),
        cache_write_commitment=cache_write_commitment,
        cache_read_commitment=cache_read_commitment,
    )
    _validate_precommit(statement, precommit, error_type=ProofV3Error)
    return precommit


class _Transcript:
    """Small domain-separated Fiat--Shamir transcript local to this reference."""

    def __init__(self) -> None:
        self._state = hashlib.sha256(_TRANSCRIPT_INIT_DOMAIN).digest()

    @property
    def state(self) -> bytes:
        return self._state

    def absorb(self, label: bytes, value: bytes) -> None:
        if len(label) > 255 or len(value) >= 1 << 32:
            raise ProofV3Error("RAM transcript item exceeds its canonical bound")
        self._state = hashlib.sha256(
            _TRANSCRIPT_ABSORB_DOMAIN
            + self._state
            + struct.pack("<B", len(label))
            + label
            + struct.pack("<I", len(value))
            + value
        ).digest()

    def scalar(self, label: bytes, *, nonzero: bool) -> int:
        if len(label) > 255:
            raise ProofV3Error("RAM transcript scalar label is too long")
        for counter in range(_MAX_REJECTION_ATTEMPTS):
            candidate = int.from_bytes(
                hashlib.sha256(
                    _TRANSCRIPT_SCALAR_DOMAIN
                    + self._state
                    + struct.pack("<B", len(label))
                    + label
                    + struct.pack("<I", counter)
                ).digest(),
                "little",
            )
            if candidate < PALLAS_SCALAR_MODULUS and (candidate != 0 or not nonzero):
                encoded = struct.pack("<I", counter) + encode_scalar(candidate)
                self.absorb(b"scalar:" + label, encoded)
                return candidate
        raise ProofV3Error("unable to derive a canonical RAM transcript scalar")


def _nonce(value: object) -> bytes:
    return _fixed32(value, "validator_nonce")


def _start_transcript(
    statement: LogUpRamStatementV3,
    precommit: FrozenLogUpRamPrecommitV3,
    validator_nonce: bytes,
) -> _Transcript:
    transcript = _Transcript()
    transcript.absorb(b"statement", statement.canonical_bytes())
    transcript.absorb(b"statement-digest", statement.digest())
    transcript.absorb(b"precommit", precommit.canonical_bytes())
    transcript.absorb(b"precommit-digest", precommit.digest())
    transcript.absorb(b"validator-nonce", validator_nonce)
    return transcript


def _lookup_challenges(transcript: _Transcript) -> tuple[int, int]:
    beta = transcript.scalar(b"lookup-beta", nonzero=False)
    gamma = transcript.scalar(b"lookup-gamma", nonzero=True)
    return beta, gamma


def _lookup_target(*, beta: int, gamma: int, address: int, value: int) -> int:
    return (beta + address + gamma * value) % PALLAS_SCALAR_MODULUS


def _inverse_table(
    *,
    beta: int,
    gamma: int,
    addresses: tuple[int, ...],
    values: tuple[int, ...],
    label: str,
) -> tuple[int, ...]:
    """Invert every row; activity/multiplicity deliberately never appears here."""

    inverses: list[int] = []
    for index, (address, value) in enumerate(zip(addresses, values, strict=True)):
        target = _lookup_target(
            beta=beta,
            gamma=gamma,
            address=address,
            value=value,
        )
        if target == 0:
            raise ProofV3Error(
                f"RAM {label} row {index} has a zero lookup denominator"
            )
        inverses.append(pow(target, PALLAS_SCALAR_MODULUS - 2, PALLAS_SCALAR_MODULUS))
    return tuple(inverses)


def _mle_basis(point: tuple[int, ...]) -> tuple[int, ...]:
    values = [1]
    modulus = PALLAS_SCALAR_MODULUS
    for coordinate in point:
        next_values: list[int] = []
        for value in values:
            next_values.append(value * (1 - coordinate) % modulus)
            next_values.append(value * coordinate % modulus)
        values = next_values
    return tuple(values)


def _fold(values: tuple[int, ...], coordinate: int) -> tuple[int, ...]:
    if len(values) < 2 or len(values) & (len(values) - 1):
        raise ProofV3Error("RAM MLE fold has an invalid table length")
    half = len(values) // 2
    modulus = PALLAS_SCALAR_MODULUS
    return tuple(
        (values[index] + coordinate * (values[index + half] - values[index]))
        % modulus
        for index in range(half)
    )


def _mle_evaluate(values: tuple[int, ...], point: tuple[int, ...]) -> int:
    current = values
    for coordinate in point:
        current = _fold(current, coordinate)
    if len(current) != 1:
        raise ProofV3Error("RAM MLE point has an unexpected dimension")
    return current[0]


def _interpolate_degree3(evaluations: tuple[int, int, int, int], point: int) -> int:
    """Evaluate a degree-at-most-three polynomial known at 0, 1, 2, and 3."""

    modulus = PALLAS_SCALAR_MODULUS
    e0, e1, e2, e3 = evaluations
    # Integer divisions are exact before reduction.  The constants remain
    # explicit so the transcript format never depends on a generic polynomial
    # library or a host floating-point operation.
    l0 = -(point - 1) * (point - 2) * (point - 3) * pow(6, modulus - 2, modulus)
    l1 = point * (point - 2) * (point - 3) * pow(2, modulus - 2, modulus)
    l2 = -point * (point - 1) * (point - 3) * pow(2, modulus - 2, modulus)
    l3 = point * (point - 1) * (point - 2) * pow(6, modulus - 2, modulus)
    return (e0 * l0 + e1 * l1 + e2 * l2 + e3 * l3) % modulus


@dataclass(frozen=True, slots=True)
class _LogUpTables:
    source: tuple[int, ...]
    read: tuple[int, ...]
    inverse_source: tuple[int, ...]
    inverse_read: tuple[int, ...]
    source_addresses: tuple[int, ...]
    read_addresses: tuple[int, ...]
    source_multiplicity: tuple[int, ...]
    read_active: tuple[int, ...]
    eq_tau_source: tuple[int, ...]
    eq_tau_read: tuple[int, ...]

    def __post_init__(self) -> None:
        length = len(self.source)
        if length < 1 or length & (length - 1):
            raise ProofV3Error("RAM sumcheck table has an invalid length")
        for name, value in (
            ("read", self.read),
            ("inverse_source", self.inverse_source),
            ("inverse_read", self.inverse_read),
            ("source_addresses", self.source_addresses),
            ("read_addresses", self.read_addresses),
            ("source_multiplicity", self.source_multiplicity),
            ("read_active", self.read_active),
            ("eq_tau_source", self.eq_tau_source),
            ("eq_tau_read", self.eq_tau_read),
        ):
            if len(value) != length:
                raise ProofV3Error(f"RAM sumcheck {name} length is inconsistent")

    def fold(self, coordinate: int) -> "_LogUpTables":
        return _LogUpTables(
            source=_fold(self.source, coordinate),
            read=_fold(self.read, coordinate),
            inverse_source=_fold(self.inverse_source, coordinate),
            inverse_read=_fold(self.inverse_read, coordinate),
            source_addresses=_fold(self.source_addresses, coordinate),
            read_addresses=_fold(self.read_addresses, coordinate),
            source_multiplicity=_fold(self.source_multiplicity, coordinate),
            read_active=_fold(self.read_active, coordinate),
            eq_tau_source=_fold(self.eq_tau_source, coordinate),
            eq_tau_read=_fold(self.eq_tau_read, coordinate),
        )


def _row_relation(
    *,
    source: int,
    read: int,
    inverse_source: int,
    inverse_read: int,
    source_address: int,
    read_address: int,
    source_multiplicity: int,
    read_active: int,
    eq_tau_source: int,
    eq_tau_read: int,
    beta: int,
    gamma: int,
    lambda_lookup: int,
    lambda_source_inverse: int,
    lambda_read_inverse: int,
) -> int:
    modulus = PALLAS_SCALAR_MODULUS
    source_target = _lookup_target(
        beta=beta,
        gamma=gamma,
        address=source_address,
        value=source,
    )
    read_target = _lookup_target(
        beta=beta,
        gamma=gamma,
        address=read_address,
        value=read,
    )
    return (
        lambda_lookup
        * (source_multiplicity * inverse_source - read_active * inverse_read)
        + lambda_source_inverse
        * eq_tau_source
        * (source_target * inverse_source - 1)
        + lambda_read_inverse * eq_tau_read * (read_target * inverse_read - 1)
    ) % modulus


def _round_evaluations(
    tables: _LogUpTables,
    *,
    beta: int,
    gamma: int,
    lambda_lookup: int,
    lambda_source_inverse: int,
    lambda_read_inverse: int,
) -> tuple[int, int, int, int]:
    if len(tables.source) < 2:
        raise ProofV3Error("RAM sumcheck has no remaining round")
    half = len(tables.source) // 2
    modulus = PALLAS_SCALAR_MODULUS

    def evaluate_at(point: int) -> int:
        total = 0
        for index in range(half):
            def interpolate(values: tuple[int, ...]) -> int:
                return (
                    values[index]
                    + point * (values[index + half] - values[index])
                ) % modulus

            total += _row_relation(
                source=interpolate(tables.source),
                read=interpolate(tables.read),
                inverse_source=interpolate(tables.inverse_source),
                inverse_read=interpolate(tables.inverse_read),
                source_address=interpolate(tables.source_addresses),
                read_address=interpolate(tables.read_addresses),
                source_multiplicity=interpolate(tables.source_multiplicity),
                read_active=interpolate(tables.read_active),
                eq_tau_source=interpolate(tables.eq_tau_source),
                eq_tau_read=interpolate(tables.eq_tau_read),
                beta=beta,
                gamma=gamma,
                lambda_lookup=lambda_lookup,
                lambda_source_inverse=lambda_source_inverse,
                lambda_read_inverse=lambda_read_inverse,
            )
        return total % modulus

    return tuple(evaluate_at(point) for point in range(4))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class LogUpRamSumcheckRoundV3:
    """One canonical degree-three Pallas sumcheck round."""

    evaluation_at_0: int
    evaluation_at_1: int
    evaluation_at_2: int
    evaluation_at_3: int

    def __post_init__(self) -> None:
        for name in (
            "evaluation_at_0",
            "evaluation_at_1",
            "evaluation_at_2",
            "evaluation_at_3",
        ):
            _field(getattr(self, name), f"RAM sumcheck {name}")

    @property
    def evaluations(self) -> tuple[int, int, int, int]:
        return (
            self.evaluation_at_0,
            self.evaluation_at_1,
            self.evaluation_at_2,
            self.evaluation_at_3,
        )


@dataclass(frozen=True, slots=True)
class PallasLogUpRamProofReferenceV3:
    """One bounded local-reference LogUp proof and four PCS openings."""

    inverse_source_commitment: bytes
    inverse_read_commitment: bytes
    rounds: tuple[LogUpRamSumcheckRoundV3, ...]
    source_opening: PCSOpeningV2
    read_opening: PCSOpeningV2
    inverse_source_opening: PCSOpeningV2
    inverse_read_opening: PCSOpeningV2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "inverse_source_commitment",
            _fixed_commitment(
                self.inverse_source_commitment,
                "RAM inverse_source_commitment",
            ),
        )
        object.__setattr__(
            self,
            "inverse_read_commitment",
            _fixed_commitment(
                self.inverse_read_commitment,
                "RAM inverse_read_commitment",
            ),
        )
        try:
            rounds = tuple(self.rounds)
        except TypeError as exc:
            raise ProofV3Error("RAM sumcheck rounds must be iterable") from exc
        if not all(isinstance(round_, LogUpRamSumcheckRoundV3) for round_ in rounds):
            raise ProofV3Error("RAM sumcheck rounds have an unexpected type")
        object.__setattr__(self, "rounds", rounds)


def _sumcheck_round_bytes(
    round_index: int,
    round_: LogUpRamSumcheckRoundV3,
) -> bytes:
    return struct.pack("<I", round_index) + b"".join(
        encode_scalar(value) for value in round_.evaluations
    )


def _sumcheck_challenge_label(round_index: int) -> bytes:
    return b"sumcheck-r" + struct.pack("<I", round_index)


def _opening_digest(transcript: _Transcript, label: bytes) -> bytes:
    return hashlib.sha256(
        _OPENING_DOMAIN
        + transcript.state
        + struct.pack("<B", len(label))
        + label
    ).digest()


def _replay_sumcheck(
    transcript: _Transcript,
    rounds: tuple[LogUpRamSumcheckRoundV3, ...],
) -> tuple[tuple[int, ...], int]:
    """Replay sumcheck and return the opening point and final claimed value."""

    claim = 0
    point: list[int] = []
    for round_index, round_ in enumerate(rounds):
        if not isinstance(round_, LogUpRamSumcheckRoundV3):
            raise ProofV3VerificationError("RAM sumcheck round has an unexpected type")
        evaluations = tuple(
            _field(value, f"RAM round {round_index} evaluation")
            for value in round_.evaluations
        )
        if (evaluations[0] + evaluations[1]) % PALLAS_SCALAR_MODULUS != claim:
            raise ProofV3VerificationError("RAM sumcheck round does not match its claim")
        transcript.absorb(b"sumcheck-round", _sumcheck_round_bytes(round_index, round_))
        coordinate = transcript.scalar(
            _sumcheck_challenge_label(round_index),
            nonzero=False,
        )
        claim = _interpolate_degree3(evaluations, coordinate)
        point.append(coordinate)
    return tuple(point), claim


def _make_tables(
    *,
    statement: LogUpRamStatementV3,
    source_values: tuple[int, ...],
    read_values: tuple[int, ...],
    inverse_source: tuple[int, ...],
    inverse_read: tuple[int, ...],
    tau_source: tuple[int, ...],
    tau_read: tuple[int, ...],
) -> _LogUpTables:
    return _LogUpTables(
        source=source_values,
        read=read_values,
        inverse_source=inverse_source,
        inverse_read=inverse_read,
        source_addresses=tuple(range(statement.vector_length)),
        read_addresses=statement.read_addresses,
        source_multiplicity=statement.read_histogram(),
        read_active=statement.read_active,
        eq_tau_source=_mle_basis(tau_source),
        eq_tau_read=_mle_basis(tau_read),
    )


def _derive_post_inverse_challenges(
    transcript: _Transcript,
    *,
    inverse_source_commitment: bytes,
    inverse_read_commitment: bytes,
    log_vector_length: int,
) -> tuple[int, int, int, tuple[int, ...], tuple[int, ...]]:
    transcript.absorb(b"inverse-source-commitment", inverse_source_commitment)
    transcript.absorb(b"inverse-read-commitment", inverse_read_commitment)
    lambda_lookup = transcript.scalar(b"lambda-lookup", nonzero=True)
    lambda_source_inverse = transcript.scalar(b"lambda-source-inverse", nonzero=True)
    lambda_read_inverse = transcript.scalar(b"lambda-read-inverse", nonzero=True)
    tau_source = tuple(
        transcript.scalar(
            b"sumcheck-source-equality-point" + struct.pack("<I", index),
            nonzero=False,
        )
        for index in range(log_vector_length)
    )
    tau_read = tuple(
        transcript.scalar(
            b"sumcheck-read-equality-point" + struct.pack("<I", index),
            nonzero=False,
        )
        for index in range(log_vector_length)
    )
    return (
        lambda_lookup,
        lambda_source_inverse,
        lambda_read_inverse,
        tau_source,
        tau_read,
    )


def _proof_openings(
    *,
    values: tuple[int, ...],
    point: tuple[int, ...],
    digest: bytes,
    label: str,
) -> PCSOpeningV2:
    try:
        return prove(values, point, digest, encoding=ENCODING_PALLAS_SCALAR)
    except (PCSNativeError, PCSUnavailableError, ValueError, TypeError) as exc:
        raise ProofV3Error(f"unable to create RAM {label} PCS opening") from exc


def prove_logup_ram_reference_v3(
    *,
    statement: LogUpRamStatementV3,
    precommit: FrozenLogUpRamPrecommitV3,
    validator_nonce: bytes,
    source_values: object,
    read_values: object,
) -> PallasLogUpRamProofReferenceV3:
    """Create a bounded Pallas LogUp proof from precommitted field vectors.

    This API refuses to create a proof if the supplied source/read field rows
    do not exactly reproduce the commitments retained in ``precommit``.  It
    is intentionally not a production serving API.
    """

    _validate_precommit(statement, precommit, error_type=ProofV3Error)
    nonce = _nonce(validator_nonce)
    source = _field_vector(source_values, statement.vector_length, "RAM source_values")
    read = _field_vector(read_values, statement.vector_length, "RAM read_values")
    selected_source = precommit.source_commitments[statement.source_inventory_index]
    try:
        source_commitment = commit(source, encoding=ENCODING_PALLAS_SCALAR)
        read_commitment = commit(read, encoding=ENCODING_PALLAS_SCALAR)
    except (PCSNativeError, PCSUnavailableError, ValueError, TypeError) as exc:
        raise ProofV3Error("unable to check RAM precommitted field vectors") from exc
    if source_commitment != selected_source:
        raise ProofV3Error("RAM source_values do not match the frozen source commitment")
    if read_commitment != precommit.cache_read_commitment:
        raise ProofV3Error("RAM read_values do not match the frozen read commitment")

    transcript = _start_transcript(statement, precommit, nonce)
    beta, gamma = _lookup_challenges(transcript)
    source_addresses = tuple(range(statement.vector_length))
    inverse_source = _inverse_table(
        beta=beta,
        gamma=gamma,
        addresses=source_addresses,
        values=source,
        label="source",
    )
    inverse_read = _inverse_table(
        beta=beta,
        gamma=gamma,
        addresses=statement.read_addresses,
        values=read,
        label="read",
    )
    try:
        inverse_source_commitment = commit(
            inverse_source,
            encoding=ENCODING_PALLAS_SCALAR,
        )
        inverse_read_commitment = commit(
            inverse_read,
            encoding=ENCODING_PALLAS_SCALAR,
        )
    except (PCSNativeError, PCSUnavailableError, ValueError, TypeError) as exc:
        raise ProofV3Error("unable to commit RAM inverse witnesses") from exc
    (
        lambda_lookup,
        lambda_source_inverse,
        lambda_read_inverse,
        tau_source,
        tau_read,
    ) = _derive_post_inverse_challenges(
        transcript,
        inverse_source_commitment=inverse_source_commitment,
        inverse_read_commitment=inverse_read_commitment,
        log_vector_length=statement.vector_length.bit_length() - 1,
    )
    tables = _make_tables(
        statement=statement,
        source_values=source,
        read_values=read,
        inverse_source=inverse_source,
        inverse_read=inverse_read,
        tau_source=tau_source,
        tau_read=tau_read,
    )
    rounds: list[LogUpRamSumcheckRoundV3] = []
    current_tables = tables
    current_claim = 0
    point: list[int] = []
    for round_index in range(statement.vector_length.bit_length() - 1):
        round_ = LogUpRamSumcheckRoundV3(
            *_round_evaluations(
                current_tables,
                beta=beta,
                gamma=gamma,
                lambda_lookup=lambda_lookup,
                lambda_source_inverse=lambda_source_inverse,
                lambda_read_inverse=lambda_read_inverse,
            )
        )
        if (
            round_.evaluation_at_0 + round_.evaluation_at_1
        ) % PALLAS_SCALAR_MODULUS != current_claim:
            if round_index == 0:
                raise ProofV3Error("RAM lookup multiset relation does not hold")
            raise ProofV3Error("RAM prover generated an inconsistent sumcheck round")
        rounds.append(round_)
        transcript.absorb(b"sumcheck-round", _sumcheck_round_bytes(round_index, round_))
        coordinate = transcript.scalar(
            _sumcheck_challenge_label(round_index),
            nonzero=False,
        )
        current_claim = _interpolate_degree3(round_.evaluations, coordinate)
        point.append(coordinate)
        current_tables = current_tables.fold(coordinate)
    rounds_tuple = tuple(rounds)
    point_tuple = tuple(point)
    final_relation = _row_relation(
        source=current_tables.source[0],
        read=current_tables.read[0],
        inverse_source=current_tables.inverse_source[0],
        inverse_read=current_tables.inverse_read[0],
        source_address=current_tables.source_addresses[0],
        read_address=current_tables.read_addresses[0],
        source_multiplicity=current_tables.source_multiplicity[0],
        read_active=current_tables.read_active[0],
        eq_tau_source=current_tables.eq_tau_source[0],
        eq_tau_read=current_tables.eq_tau_read[0],
        beta=beta,
        gamma=gamma,
        lambda_lookup=lambda_lookup,
        lambda_source_inverse=lambda_source_inverse,
        lambda_read_inverse=lambda_read_inverse,
    )
    if current_claim != final_relation:
        raise ProofV3Error("RAM prover sumcheck final relation is inconsistent")
    # ``transcript`` is exactly the one which derived the post-inverse
    # challenges and consumed every round; no fresh/restarted transcript may
    # derive an opening point or opening digest.
    return PallasLogUpRamProofReferenceV3(
        inverse_source_commitment=inverse_source_commitment,
        inverse_read_commitment=inverse_read_commitment,
        rounds=rounds_tuple,
        source_opening=_proof_openings(
            values=source,
            point=point_tuple,
            digest=_opening_digest(transcript, b"source"),
            label="source",
        ),
        read_opening=_proof_openings(
            values=read,
            point=point_tuple,
            digest=_opening_digest(transcript, b"read"),
            label="read",
        ),
        inverse_source_opening=_proof_openings(
            values=inverse_source,
            point=point_tuple,
            digest=_opening_digest(transcript, b"inverse-source"),
            label="inverse source",
        ),
        inverse_read_opening=_proof_openings(
            values=inverse_read,
            point=point_tuple,
            digest=_opening_digest(transcript, b"inverse-read"),
            label="inverse read",
        ),
    )


def _verify_opening(
    *,
    opening: object,
    expected_commitment: bytes,
    vector_length: int,
    point: tuple[int, ...],
    digest: bytes,
    label: str,
) -> int:
    if not isinstance(opening, PCSOpeningV2):
        raise ProofV3VerificationError(f"RAM {label} opening has an unexpected type")
    if opening.commitment != expected_commitment:
        raise ProofV3VerificationError(f"RAM {label} opening commitment is unexpected")
    if (
        opening.vector_length != vector_length
        or opening.padded_length != vector_length
        or opening.encoding != ENCODING_PALLAS_SCALAR
    ):
        raise ProofV3VerificationError(
            f"RAM {label} opening does not have the statement vector length"
        )
    try:
        if not verify(opening, point, digest):
            raise ProofV3VerificationError(f"RAM {label} IPA opening is invalid")
        return scalar_to_int(opening.evaluation)
    except ProofV3VerificationError:
        raise
    except (PCSNativeError, PCSUnavailableError, ValueError, TypeError) as exc:
        raise ProofV3VerificationError(f"RAM {label} opening is malformed") from exc


def verify_logup_ram_reference_v3(
    *,
    statement: LogUpRamStatementV3,
    precommit: FrozenLogUpRamPrecommitV3,
    validator_nonce: bytes,
    proof: PallasLogUpRamProofReferenceV3,
) -> None:
    """Fail closed on any malformed, cross-context, or invalid RAM proof."""

    try:
        _validate_precommit(statement, precommit, error_type=ProofV3VerificationError)
        nonce = _nonce(validator_nonce)
        if not isinstance(proof, PallasLogUpRamProofReferenceV3):
            raise ProofV3VerificationError("RAM proof has an unexpected type")
        expected_rounds = statement.vector_length.bit_length() - 1
        if len(proof.rounds) != expected_rounds:
            raise ProofV3VerificationError("RAM proof has an unexpected round count")
        _validate_commitments(
            (proof.inverse_source_commitment, proof.inverse_read_commitment),
            error_type=ProofV3VerificationError,
        )
        transcript = _start_transcript(statement, precommit, nonce)
        beta, gamma = _lookup_challenges(transcript)
        (
            lambda_lookup,
            lambda_source_inverse,
            lambda_read_inverse,
            tau_source,
            tau_read,
        ) = _derive_post_inverse_challenges(
            transcript,
            inverse_source_commitment=proof.inverse_source_commitment,
            inverse_read_commitment=proof.inverse_read_commitment,
            log_vector_length=expected_rounds,
        )
        point, final_claim = _replay_sumcheck(
            transcript=transcript,
            rounds=proof.rounds,
        )
        selected_source = precommit.source_commitments[statement.source_inventory_index]
        source = _verify_opening(
            opening=proof.source_opening,
            expected_commitment=selected_source,
            vector_length=statement.vector_length,
            point=point,
            digest=_opening_digest(transcript, b"source"),
            label="source",
        )
        read = _verify_opening(
            opening=proof.read_opening,
            expected_commitment=precommit.cache_read_commitment,
            vector_length=statement.vector_length,
            point=point,
            digest=_opening_digest(transcript, b"read"),
            label="read",
        )
        inverse_source = _verify_opening(
            opening=proof.inverse_source_opening,
            expected_commitment=proof.inverse_source_commitment,
            vector_length=statement.vector_length,
            point=point,
            digest=_opening_digest(transcript, b"inverse-source"),
            label="inverse source",
        )
        inverse_read = _verify_opening(
            opening=proof.inverse_read_opening,
            expected_commitment=proof.inverse_read_commitment,
            vector_length=statement.vector_length,
            point=point,
            digest=_opening_digest(transcript, b"inverse-read"),
            label="inverse read",
        )
        final_relation = _row_relation(
            source=source,
            read=read,
            inverse_source=inverse_source,
            inverse_read=inverse_read,
            source_address=_mle_evaluate(
                tuple(range(statement.vector_length)), point
            ),
            read_address=_mle_evaluate(statement.read_addresses, point),
            source_multiplicity=_mle_evaluate(statement.read_histogram(), point),
            read_active=_mle_evaluate(statement.read_active, point),
            eq_tau_source=_mle_evaluate(_mle_basis(tau_source), point),
            eq_tau_read=_mle_evaluate(_mle_basis(tau_read), point),
            beta=beta,
            gamma=gamma,
            lambda_lookup=lambda_lookup,
            lambda_source_inverse=lambda_source_inverse,
            lambda_read_inverse=lambda_read_inverse,
        )
        if final_claim != final_relation:
            raise ProofV3VerificationError("RAM sumcheck final relation is invalid")
    except ProofV3VerificationError:
        raise
    except ProofV3Error as exc:
        raise ProofV3VerificationError("RAM proof is malformed") from exc
    except (PCSNativeError, PCSUnavailableError, ValueError, TypeError) as exc:
        raise ProofV3VerificationError("RAM proof is malformed") from exc


__all__ = [
    "PALLAS_LOGUP_RAM_KEY_ABI_V3",
    "PALLAS_LOGUP_RAM_REFERENCE_ABI_V3",
    "FrozenLogUpRamPrecommitV3",
    "LogUpRamStatementV3",
    "LogUpRamSumcheckRoundV3",
    "PallasLogUpRamProofReferenceV3",
    "freeze_logup_ram_precommit_v3",
    "prove_logup_ram_reference_v3",
    "source_inventory_digest_v3",
    "verify_logup_ram_reference_v3",
]
