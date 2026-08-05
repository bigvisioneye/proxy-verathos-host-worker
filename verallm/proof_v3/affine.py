"""Exact Pallas commitment folds for proof-v3 affine handoffs.

This module verifies one deliberately narrow but real relation: a complete,
validator-derived inventory of equal-length Pallas-field vectors must satisfy
``left == right``.  The verifier derives a distinct nonzero coefficient for
each inventory coordinate only after the envelope is frozen, then checks the
weighted commitment difference is the Pallas identity.

That is a computationally binding global equality check, rather than a Merkle
sample.  It is suitable for exact copies, reshapes, fixed-scale field bridges,
and Q/K/V-to-logical-cache-write handoffs once both sides have the same signed
field encoding.  It does *not* prove a GEMM, floating-point rounding,
normalization, cache read semantics, or the attention recurrence.  In
particular, it must not be registered as a proof-v3 execution adapter or used
to claim model-substitution resistance by itself.
"""

from __future__ import annotations

import hashlib
import struct
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from zkllm.crypto.pcs_v2 import (
    COMMITMENT_BYTES,
    MAX_COMBINE_TERMS,
    MAX_VALIDATE_COMMITMENTS,
    MAX_VECTOR_LEN,
    PALLAS_SCALAR_MODULUS,
    PCSNativeError,
    PCSUnavailableError,
    combine_commitments,
    validate_commitments,
)

from verallm.proof_v3.challenge import (
    FoldedExecutionChallengeV3,
    GlobalRelationCoordinateV3,
    derive_folded_execution_challenge_v3,
    derive_pallas_affine_fold_scalar_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.payload import ProofV3CommitmentEnvelope
from verallm.proof_v3.profile import ExecutionSecurityProfileV3


PALLAS_AFFINE_TRANSITION_FOLD_ABI_V3 = "pallas.affine_transition_fold.v1"
PALLAS_AFFINE_TRANSITION_KEY_ABI_V3 = "pallas_pedersen.mle.v2"
PALLAS_IDENTITY_COMMITMENT_V3 = bytes(COMMITMENT_BYTES)
_FROZEN_AFFINE_INVENTORY_DOMAIN = (
    b"VERATHOS/PROOF_V3/AFFINE_TRANSITION_INVENTORY/SHA256"
)
_PHASE_CODES = {"prefill": 1, "decode": 2}


def _u32(value: int, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProofV3Error(f"{name} must be an unsigned 32-bit integer")
    if value < (1 if positive else 0) or value >= 1 << 32:
        qualifier = "positive " if positive else ""
        raise ProofV3Error(f"{name} must be a {qualifier}unsigned 32-bit integer")
    return value


def _coordinate_key(
    coordinate: GlobalRelationCoordinateV3,
) -> tuple[str, str, int, int, int, int]:
    if not isinstance(coordinate, GlobalRelationCoordinateV3):
        raise ProofV3Error("affine transition coordinate has an unexpected type")
    return (
        coordinate.node_id,
        coordinate.phase,
        coordinate.chunk_index,
        coordinate.logical_token_start,
        coordinate.token_count,
        coordinate.relation_index,
    )


def _fixed_commitment(value: bytes, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != COMMITMENT_BYTES:
        raise ProofV3Error(f"{name} must be exactly {COMMITMENT_BYTES} bytes")
    return value


def _fixed32(value: bytes, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    return value


def _canonical_coordinate_bytes(coordinate: GlobalRelationCoordinateV3) -> bytes:
    (
        node_id,
        phase,
        chunk_index,
        token_start,
        token_count,
        relation_index,
    ) = _coordinate_key(coordinate)
    encoded_node = node_id.encode("ascii")
    return (
        struct.pack("<B", len(encoded_node))
        + encoded_node
        + struct.pack(
            "<BIIIQ",
            _PHASE_CODES[phase],
            chunk_index,
            token_start,
            token_count,
            relation_index,
        )
    )


def _validate_inventory_items(
    expected_items: Sequence["ExpectedAffineTransitionV3"],
    statement_items: Sequence["AffineTransitionCommitmentV3"],
) -> None:
    if not expected_items:
        raise ProofV3VerificationError("affine transition inventory is empty")
    if not all(isinstance(item, ExpectedAffineTransitionV3) for item in expected_items):
        raise ProofV3VerificationError("affine transition inventory is malformed")
    if not all(
        isinstance(item, AffineTransitionCommitmentV3) for item in statement_items
    ):
        raise ProofV3VerificationError("affine transition statements are malformed")

    expected_keys = tuple(item.key for item in expected_items)
    statement_keys = tuple(item.key for item in statement_items)
    if expected_keys != tuple(sorted(expected_keys)) or len(expected_keys) != len(
        set(expected_keys)
    ):
        raise ProofV3VerificationError(
            "verifier affine transition inventory is not canonically ordered"
        )
    if statement_keys != tuple(sorted(statement_keys)) or len(statement_keys) != len(
        set(statement_keys)
    ):
        raise ProofV3VerificationError(
            "affine transition statements are not canonically ordered"
        )
    if statement_keys != expected_keys:
        raise ProofV3VerificationError(
            "affine transition statements do not exactly match the verifier inventory"
        )
    for expected_item, statement in zip(expected_items, statement_items, strict=True):
        if statement.vector_length != expected_item.vector_length:
            raise ProofV3VerificationError(
                "affine transition statement has an unexpected vector length"
            )
        if statement.commitment_key_abi_id != expected_item.commitment_key_abi_id:
            raise ProofV3VerificationError(
                "affine transition statement has an unexpected commitment key ABI"
            )


def _validate_inventory_commitments(
    statements: Sequence["AffineTransitionCommitmentV3"],
) -> None:
    commitments = tuple(
        commitment
        for statement in statements
        for commitment in (statement.left_commitment, statement.right_commitment)
    )
    try:
        for start in range(0, len(commitments), MAX_VALIDATE_COMMITMENTS):
            validate_commitments(commitments[start : start + MAX_VALIDATE_COMMITMENTS])
    except (PCSNativeError, PCSUnavailableError, ValueError, TypeError) as exc:
        raise ProofV3VerificationError(
            "affine transition inventory contains a noncanonical Pallas commitment"
        ) from exc


def _reduce_weighted_commitments(
    terms: Sequence[tuple[bytes, int]],
) -> bytes:
    """Combine an arbitrarily long canonical sequence without changing its sum."""

    if not terms:
        raise ProofV3VerificationError("affine transition fold has no terms")
    current = tuple(terms)
    while len(current) > 1:
        reduced: list[tuple[bytes, int]] = []
        for start in range(0, len(current), MAX_COMBINE_TERMS):
            chunk = current[start : start + MAX_COMBINE_TERMS]
            commitments = tuple(item[0] for item in chunk)
            coefficients = tuple(item[1] for item in chunk)
            try:
                combined = combine_commitments(commitments, coefficients)
            except (PCSNativeError, PCSUnavailableError, ValueError, TypeError) as exc:
                raise ProofV3VerificationError(
                    "affine transition fold contains an invalid Pallas commitment"
                ) from exc
            reduced.append((combined, 1))
        current = tuple(reduced)
    return current[0][0]


@dataclass(frozen=True, slots=True)
class ExpectedAffineTransitionV3:
    """One verifier-owned exact field-equality coordinate.

    The constraint-system artifact must determine the complete sequence of
    these entries.  The miner is only allowed to supply the two commitments at
    each coordinate; it cannot remove a coordinate or select a shorter vector.
    """

    coordinate: GlobalRelationCoordinateV3
    vector_length: int
    commitment_key_abi_id: str = PALLAS_AFFINE_TRANSITION_KEY_ABI_V3

    def __post_init__(self) -> None:
        _coordinate_key(self.coordinate)
        _u32(self.vector_length, "affine transition vector_length", positive=True)
        if self.vector_length > MAX_VECTOR_LEN:
            raise ProofV3Error("affine transition vector_length exceeds PCS capacity")
        if self.commitment_key_abi_id != PALLAS_AFFINE_TRANSITION_KEY_ABI_V3:
            raise ProofV3Error("affine transition commitment key ABI is unsupported")

    @property
    def key(self) -> tuple[str, str, int, int, int, int]:
        return _coordinate_key(self.coordinate)

    @property
    def group_key(self) -> tuple[str, int]:
        return (self.commitment_key_abi_id, self.vector_length)


@dataclass(frozen=True, slots=True)
class AffineTransitionCommitmentV3:
    """The two committed sides of one exact affine handoff.

    ``left_commitment`` and ``right_commitment`` are commitments under the
    same Pallas commitment key and exact logical vector length.  The signed
    relation artifact gives their semantic meaning (for example, QKV output
    and a logical cache-table write); this object never infers that meaning
    from a miner-controlled label.
    """

    coordinate: GlobalRelationCoordinateV3
    vector_length: int
    left_commitment: bytes
    right_commitment: bytes
    commitment_key_abi_id: str = PALLAS_AFFINE_TRANSITION_KEY_ABI_V3

    def __post_init__(self) -> None:
        _coordinate_key(self.coordinate)
        _u32(self.vector_length, "affine transition vector_length", positive=True)
        if self.vector_length > MAX_VECTOR_LEN:
            raise ProofV3Error("affine transition vector_length exceeds PCS capacity")
        if self.commitment_key_abi_id != PALLAS_AFFINE_TRANSITION_KEY_ABI_V3:
            raise ProofV3Error("affine transition commitment key ABI is unsupported")
        _fixed_commitment(self.left_commitment, "affine transition left commitment")
        _fixed_commitment(self.right_commitment, "affine transition right commitment")

    @property
    def key(self) -> tuple[str, str, int, int, int, int]:
        return _coordinate_key(self.coordinate)

    @property
    def group_key(self) -> tuple[str, int]:
        return (self.commitment_key_abi_id, self.vector_length)


@dataclass(frozen=True, slots=True)
class FrozenAffineTransitionInventoryV3:
    """Exact handoff commitments accepted by the validator before nonce reveal.

    A serving session must construct this object only while atomically
    accepting the sealed envelope, before it reveals the validator nonce. The
    inventory carries every precommitted pair the later random fold will use;
    accepting fresh statements after the nonce would let a miner construct a
    cancellation deliberately and is forbidden.

    This object is validator-side lifecycle state. A future native envelope
    must commit its :meth:`digest` through the signed execution-table inventory
    before it can be serialized as part of a production proof.
    """

    commitment_envelope_digest: bytes
    expected: tuple[ExpectedAffineTransitionV3, ...]
    statements: tuple[AffineTransitionCommitmentV3, ...]

    def __post_init__(self) -> None:
        _fixed32(
            self.commitment_envelope_digest,
            "affine transition commitment_envelope_digest",
        )
        expected = tuple(self.expected)
        statements = tuple(self.statements)
        _validate_inventory_items(expected, statements)
        _validate_inventory_commitments(statements)
        object.__setattr__(self, "expected", expected)
        object.__setattr__(self, "statements", statements)

    def canonical_bytes(self) -> bytes:
        records = []
        for expected_item, statement in zip(
            self.expected,
            self.statements,
            strict=True,
        ):
            key_abi = expected_item.commitment_key_abi_id.encode("ascii")
            records.append(
                _canonical_coordinate_bytes(expected_item.coordinate)
                + struct.pack("<IB", expected_item.vector_length, len(key_abi))
                + key_abi
                + statement.left_commitment
                + statement.right_commitment
            )
        return (
            self.commitment_envelope_digest
            + struct.pack("<I", len(records))
            + b"".join(records)
        )

    def digest(self) -> bytes:
        """Return the deterministic precommit inventory identity."""

        return hashlib.sha256(
            _FROZEN_AFFINE_INVENTORY_DOMAIN + self.canonical_bytes()
        ).digest()


def freeze_affine_transition_inventory_v3(
    *,
    envelope: ProofV3CommitmentEnvelope,
    expected: Iterable[ExpectedAffineTransitionV3],
    statements: Iterable[AffineTransitionCommitmentV3],
) -> FrozenAffineTransitionInventoryV3:
    """Freeze exact affine commitments during the pre-nonce envelope phase.

    This pure constructor cannot itself enforce network timing. Production
    wiring must call it from the validator's atomic envelope-accept path and
    retain the resulting object in the one-use nonce session.
    """

    if not isinstance(envelope, ProofV3CommitmentEnvelope):
        raise ProofV3VerificationError(
            "affine transition envelope has an unexpected type"
        )
    return FrozenAffineTransitionInventoryV3(
        commitment_envelope_digest=envelope.digest(),
        expected=tuple(expected),
        statements=tuple(statements),
    )


@dataclass(frozen=True, slots=True)
class AffineTransitionFoldResultV3:
    """Successful exact-coverage verification summary.

    The result intentionally exposes counts only.  It is not a proof receipt
    and must not be serialized as evidence of an execution proof.
    """

    statement_count: int
    commitment_group_count: int

    def __post_init__(self) -> None:
        _u32(self.statement_count, "affine transition statement_count", positive=True)
        _u32(
            self.commitment_group_count,
            "affine transition commitment_group_count",
            positive=True,
        )


def verify_affine_transition_folds_v3(
    *,
    profile: ExecutionSecurityProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    validator_nonce: bytes,
    challenge: FoldedExecutionChallengeV3,
    inventory: FrozenAffineTransitionInventoryV3,
) -> AffineTransitionFoldResultV3:
    """Verify all exact Pallas-field handoffs in a frozen statement inventory.

    This must run only after the validator has accepted the matching envelope
    and revealed its nonce. ``inventory`` is generated from the signed
    constraint-system artifact, validator-derived chunk universe, and sealed
    pre-nonce commitments. A nonzero difference vector can cancel against the
    independently nonce-derived field coefficients only with the usual
    field-level soundness probability, assuming that frozen inventory was not
    replaceable after the nonce was known.

    The function establishes exact equality only.  Callers still need a GEMM
    proof for linear arithmetic, a RAM argument for cache reads, and a
    transition relation for attention/GDN/bridge semantics.
    """

    if not isinstance(profile, ExecutionSecurityProfileV3):
        raise ProofV3VerificationError(
            "affine transition profile has an unexpected type"
        )
    if not isinstance(envelope, ProofV3CommitmentEnvelope):
        raise ProofV3VerificationError(
            "affine transition envelope has an unexpected type"
        )
    if not isinstance(challenge, FoldedExecutionChallengeV3):
        raise ProofV3VerificationError(
            "affine transition challenge has an unexpected type"
        )
    if not isinstance(inventory, FrozenAffineTransitionInventoryV3):
        raise ProofV3VerificationError(
            "affine transition inventory has an unexpected type"
        )
    if inventory.commitment_envelope_digest != envelope.digest():
        raise ProofV3VerificationError(
            "affine transition inventory is bound to a different commitment envelope"
        )
    try:
        expected_challenge = derive_folded_execution_challenge_v3(
            validator_nonce=validator_nonce,
            profile=profile,
            envelope=envelope,
        )
    except ProofV3Error as exc:
        raise ProofV3VerificationError(
            "affine transition validator nonce is malformed"
        ) from exc
    if challenge != expected_challenge:
        raise ProofV3VerificationError(
            "affine transition challenge is not bound to the validator transcript"
        )

    expected_items = inventory.expected
    statement_items = inventory.statements

    grouped_terms: dict[tuple[str, int], list[tuple[bytes, int]]] = defaultdict(list)
    for expected_item, statement in zip(expected_items, statement_items, strict=True):
        try:
            coefficient = derive_pallas_affine_fold_scalar_v3(
                challenge=challenge,
                profile=profile,
                envelope=envelope,
                coordinate=expected_item.coordinate,
            )
        except ProofV3Error as exc:
            raise ProofV3VerificationError(
                "affine transition coordinate is not verifier-derived"
            ) from exc
        grouped_terms[expected_item.group_key].extend(
            (
                (statement.right_commitment, coefficient),
                (
                    statement.left_commitment,
                    (-coefficient) % PALLAS_SCALAR_MODULUS,
                ),
            )
        )

    for group_key in sorted(grouped_terms):
        aggregate = _reduce_weighted_commitments(tuple(grouped_terms[group_key]))
        if aggregate != PALLAS_IDENTITY_COMMITMENT_V3:
            raise ProofV3VerificationError(
                "affine transition fold does not satisfy exact field equality"
            )
    return AffineTransitionFoldResultV3(
        statement_count=len(expected_items),
        commitment_group_count=len(grouped_terms),
    )


__all__ = [
    "PALLAS_AFFINE_TRANSITION_FOLD_ABI_V3",
    "PALLAS_AFFINE_TRANSITION_KEY_ABI_V3",
    "PALLAS_IDENTITY_COMMITMENT_V3",
    "AffineTransitionCommitmentV3",
    "AffineTransitionFoldResultV3",
    "ExpectedAffineTransitionV3",
    "FrozenAffineTransitionInventoryV3",
    "freeze_affine_transition_inventory_v3",
    "verify_affine_transition_folds_v3",
]
