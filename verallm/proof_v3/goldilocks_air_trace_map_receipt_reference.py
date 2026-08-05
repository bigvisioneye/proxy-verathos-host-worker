"""Validator-owned receipt gate for the V3 Goldilocks AIR trace-map reference.

The trace-map and semantic-selection references intentionally expose their
algebraic operations so they can be tested independently.  That is not a safe
integration boundary: a miner must not choose which map root, selected slots,
or nonce authorizes verification.  This unregistered CPU conformance module
owns the complete chronology for one request instead:

* accept the canonical envelope and derive the exact map statement from
  validator-qualified artifacts;
* seal one complete map precommitment in validator-owned state before the
  nonce can be revealed;
* derive the signed semantic slot union from the revealed nonce internally;
  and
* atomically consume the receipt before checking the exact map opening and
  every selected AIR proof.

The inner proof nonce is domain-separated by the map precommitment, semantic
selection, and canonical slot index.  This is defense in depth: map
membership is checked first, but a valid slot proof is also not reusable under
another frozen map or selected-slot set.

This file is deliberately not imported by the V3 package facade, payload,
session, miner, validator, or adapter registry.  It is not a canonical wire
format, distributed receipt store, native trace backend, runtime-tensor
proof, cache-RAM proof, transition adapter, or production-verifiable-inference
claim.
"""

from __future__ import annotations

import hashlib
import secrets
import struct
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

from verallm.proof_v3.accumulator import PostCommitOpeningTicketV3
from verallm.proof_v3.challenge import PostCommitAuditDecisionV3
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_air_reference import (
    GoldilocksAirProofReferenceV3,
    GoldilocksAirTracePrecommitmentReferenceV3,
    verify_goldilocks_air_reference_v3,
)
from verallm.proof_v3.goldilocks_air_trace_map_reference import (
    GoldilocksAirTraceMapOpeningReferenceV3,
    GoldilocksAirTraceMapPrecommitmentReferenceV3,
    GoldilocksAirTraceMapStatementReferenceV3,
    derive_goldilocks_air_trace_map_statement_reference_v3,
    verify_goldilocks_air_trace_map_opening_reference_v3,
)
from verallm.proof_v3.goldilocks_air_trace_selection_reference import (
    GoldilocksAirTraceHardSelectionReferenceV3,
    derive_goldilocks_air_trace_hard_selection_reference_v3,
    verify_goldilocks_air_trace_hard_selection_reference_v3,
)
from verallm.proof_v3.payload import (
    ProofV3CommitmentEnvelope,
    commitment_envelope_from_bytes,
)
from verallm.proof_v3.relation import RuntimeHardAuditPolicyV3
from verallm.proof_v3.request import (
    ObservedExecutionOutputV3,
    ValidatorExecutionRequestContextV3,
    commit_validator_nonce_v3,
)
from verallm.proof_v3.session import (
    DEFAULT_PROOF_ARRIVAL_BUDGET_NS_V3,
    NonceRevealV3,
    ProofV3ChallengeSession,
    QualifiedExecutionProfileV3,
)
from verallm.proof_v3.verifier import (
    EnvelopeBindingV3,
    validate_execution_envelope_against_precommit_v3,
)


GOLDILOCKS_AIR_TRACE_MAP_RECEIPT_REFERENCE_ABI_V3: Final = (
    "goldilocks.air_trace_map_receipt.reference.v1"
)
GOLDILOCKS_AIR_TRACE_MAP_RECEIPT_REFERENCE_FORMAT_VERSION_V3: Final = 1

_RECEIPT_IDENTIFIER_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR_TRACE_MAP_RECEIPT/V1/IDENTIFIER/SHA256"
)
_RECEIPT_DIGEST_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR_TRACE_MAP_RECEIPT/V1/DIGEST/SHA256"
)
_RECEIPT_LOGICAL_KEY_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR_TRACE_MAP_RECEIPT/V1/LOGICAL_KEY/SHA256"
)
_SLOT_PROOF_NONCE_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR_TRACE_MAP_RECEIPT/V1/SLOT_NONCE/SHA256"
)
_ACCEPTED_STATEMENT_FACTORY_TOKEN = object()
_RECEIPT_FACTORY_TOKEN = object()
_POST_REVEAL_FACTORY_TOKEN = object()
_COORDINATOR_FACTORY_TOKEN = object()


def _fixed32(value: object, name: str, *, nonzero: bool = False) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3VerificationError(f"{name} must be exactly 32 bytes")
    if nonzero and value == bytes(32):
        raise ProofV3VerificationError(f"{name} must not be the zero digest")
    return value


def _monotonic_ns(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ProofV3VerificationError(f"{name} must be a monotonic nanosecond value")
    return value


def _arrival_budget_ns(value: object) -> int:
    if type(value) is not int or not 0 < value <= DEFAULT_PROOF_ARRIVAL_BUDGET_NS_V3:
        raise ProofV3VerificationError(
            "proof_arrival_budget_ns must be between one nanosecond and one second"
        )
    return value


def _require_bound_envelope(
    *,
    qualified_profile: QualifiedExecutionProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    envelope_binding: EnvelopeBindingV3,
) -> None:
    """Require the map statement inputs to be validator-accepted exactly once."""

    if not isinstance(qualified_profile, QualifiedExecutionProfileV3):
        raise ProofV3VerificationError(
            "Goldilocks AIR trace-map receipt requires a qualified execution profile"
        )
    qualified_profile.require_qualification_provenance()
    if not isinstance(envelope, ProofV3CommitmentEnvelope):
        raise ProofV3VerificationError(
            "Goldilocks AIR trace-map receipt envelope is malformed"
        )
    if not isinstance(envelope_binding, EnvelopeBindingV3):
        raise ProofV3VerificationError(
            "Goldilocks AIR trace-map receipt envelope binding is malformed"
        )
    profile = qualified_profile.profile
    precommit_context = envelope_binding.precommit_context
    request_binding = envelope_binding.request_binding
    if (
        envelope_binding.profile_digest != profile.digest()
        or request_binding.precommit_context != precommit_context
        or envelope.static_manifest_digest != profile.static_manifest_digest
        or envelope.execution_profile_digest != profile.digest()
        or envelope.precommit_context_digest != precommit_context.digest()
        or envelope.prompt_token_root != precommit_context.prompt_token_root
        or envelope.sampler_config_digest != precommit_context.sampler_config_digest
        or envelope.cache_lease_digest != precommit_context.cache_lease_digest
        or envelope.request_digest != request_binding.request_digest
        or envelope.output_token_root != request_binding.output.output_token_root
        or envelope.output_stream_digest != request_binding.output.output_stream_digest
        or envelope.context_token_count != precommit_context.context_token_count
        or envelope.decode_token_count != request_binding.output.decode_token_count
    ):
        raise ProofV3VerificationError(
            "Goldilocks AIR trace-map receipt envelope is not validator accepted"
        )


def _require_map_precommitment(
    *,
    statement: GoldilocksAirTraceMapStatementReferenceV3,
    precommitment: GoldilocksAirTraceMapPrecommitmentReferenceV3,
) -> GoldilocksAirTraceMapPrecommitmentReferenceV3:
    """Validate and snapshot an untrusted map header for validator retention."""

    if not isinstance(statement, GoldilocksAirTraceMapStatementReferenceV3):
        raise ProofV3VerificationError(
            "Goldilocks AIR trace-map receipt statement is malformed"
        )
    statement.require_factory_provenance()
    if type(precommitment) is not GoldilocksAirTraceMapPrecommitmentReferenceV3:
        raise ProofV3VerificationError(
            "Goldilocks AIR trace-map receipt precommitment is malformed"
        )
    try:
        if type(
            precommitment.statement,
        ) is not GoldilocksAirTraceMapStatementReferenceV3:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map receipt precommitment statement is malformed"
            )
        precommitment.statement.require_factory_provenance()
        if precommitment.statement.digest() != statement.digest():
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map receipt precommitment belongs to another statement"
            )
        # Reconstruct with the validator-derived statement, then retain this
        # independent frozen object.  Retaining the caller's object would let
        # a post-accept object mutation swap its map root before the nonce.
        normalized = GoldilocksAirTraceMapPrecommitmentReferenceV3(
            statement=statement,
            tree_leaf_count=precommitment.tree_leaf_count,
            map_tree_binding_digest=precommitment.map_tree_binding_digest,
            trace_map_commitment=precommitment.trace_map_commitment,
            abi_id=precommitment.abi_id,
            format_version=precommitment.format_version,
        )
    except ProofV3VerificationError:
        raise
    except (AttributeError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "Goldilocks AIR trace-map receipt precommitment is malformed"
        ) from exc
    return normalized


@dataclass(frozen=True, slots=True)
class GoldilocksAirTraceMapProofBundleReferenceV3:
    """One structural final bundle with no miner-authoritative selection data.

    The coordinator alone provides the expected selection, retained map root,
    and validator nonce.  The bundle carries only the resulting map opening
    and one canonical AIR proof per opened slot.
    """

    map_opening: GoldilocksAirTraceMapOpeningReferenceV3
    air_proofs: tuple[GoldilocksAirProofReferenceV3, ...]
    abi_id: str = GOLDILOCKS_AIR_TRACE_MAP_RECEIPT_REFERENCE_ABI_V3
    format_version: int = GOLDILOCKS_AIR_TRACE_MAP_RECEIPT_REFERENCE_FORMAT_VERSION_V3

    def __post_init__(self) -> None:
        if self.abi_id != GOLDILOCKS_AIR_TRACE_MAP_RECEIPT_REFERENCE_ABI_V3:
            raise ProofV3Error("Goldilocks AIR trace-map receipt bundle ABI is unsupported")
        if type(self.format_version) is not int or self.format_version != (
            GOLDILOCKS_AIR_TRACE_MAP_RECEIPT_REFERENCE_FORMAT_VERSION_V3
        ):
            raise ProofV3Error(
                "Goldilocks AIR trace-map receipt bundle format version is unsupported"
            )
        if not isinstance(self.map_opening, GoldilocksAirTraceMapOpeningReferenceV3):
            raise ProofV3Error("Goldilocks AIR trace-map receipt opening is malformed")
        # Require a tuple at the reference boundary so an unbounded iterable
        # cannot be materialized before the receipt is atomically consumed.
        if not isinstance(self.air_proofs, tuple) or not self.air_proofs:
            raise ProofV3Error("Goldilocks AIR trace-map receipt proofs are malformed")
        if len(self.air_proofs) != len(self.map_opening.trace_precommitments):
            raise ProofV3Error(
                "Goldilocks AIR trace-map receipt proof count does not match opening"
            )
        if not all(isinstance(proof, GoldilocksAirProofReferenceV3) for proof in self.air_proofs):
            raise ProofV3Error("Goldilocks AIR trace-map receipt proof is malformed")


@dataclass(frozen=True, slots=True, init=False)
class _GoldilocksAirTraceMapAcceptedStatementCapabilityReferenceV3:
    """Opaque coordinator evidence that the exact map statement was accepted."""

    statement: GoldilocksAirTraceMapStatementReferenceV3
    validator_nonce_commitment: bytes
    nonce_context_digest: bytes
    _factory_token: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ProofV3VerificationError(
            "Goldilocks AIR trace-map accepted statements are issued by the coordinator"
        )

    @classmethod
    def _construct(
        cls,
        *,
        statement: GoldilocksAirTraceMapStatementReferenceV3,
        validator_nonce_commitment: bytes,
        nonce_context_digest: bytes,
        _factory_token: object | None = None,
    ) -> "_GoldilocksAirTraceMapAcceptedStatementCapabilityReferenceV3":
        if _factory_token is not _ACCEPTED_STATEMENT_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map accepted statements are issued by the coordinator"
            )
        result = object.__new__(cls)
        object.__setattr__(result, "statement", statement)
        object.__setattr__(result, "validator_nonce_commitment", validator_nonce_commitment)
        object.__setattr__(result, "nonce_context_digest", nonce_context_digest)
        object.__setattr__(result, "_factory_token", _ACCEPTED_STATEMENT_FACTORY_TOKEN)
        result.require_factory_provenance()
        return result

    def require_factory_provenance(self) -> None:
        if self._factory_token is not _ACCEPTED_STATEMENT_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map accepted statement lacks coordinator provenance"
            )
        if not isinstance(self.statement, GoldilocksAirTraceMapStatementReferenceV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map accepted statement is malformed"
            )
        self.statement.require_factory_provenance()
        _fixed32(
            self.validator_nonce_commitment,
            "Goldilocks AIR trace-map accepted statement nonce commitment",
            nonzero=True,
        )
        _fixed32(
            self.nonce_context_digest,
            "Goldilocks AIR trace-map accepted statement nonce context",
            nonzero=True,
        )


def _accept_validator_map_statement_reference_v3(
    *,
    qualified_profile: QualifiedExecutionProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    envelope_binding: EnvelopeBindingV3,
) -> _GoldilocksAirTraceMapAcceptedStatementCapabilityReferenceV3:
    """Mint a store capability from the coordinator's accepted envelope only."""

    _require_bound_envelope(
        qualified_profile=qualified_profile,
        envelope=envelope,
        envelope_binding=envelope_binding,
    )
    statement = derive_goldilocks_air_trace_map_statement_reference_v3(
        qualified_profile=qualified_profile,
        envelope=envelope,
        envelope_binding=envelope_binding,
    )
    precommit_context = envelope_binding.precommit_context
    return _GoldilocksAirTraceMapAcceptedStatementCapabilityReferenceV3._construct(
        statement=statement,
        validator_nonce_commitment=precommit_context.validator_nonce_commitment,
        nonce_context_digest=precommit_context.nonce_context_digest(),
        _factory_token=_ACCEPTED_STATEMENT_FACTORY_TOKEN,
    )


@dataclass(frozen=True, slots=True, init=False)
class GoldilocksAirTraceMapReceiptReferenceV3:
    """One validator-retained complete trace-map root before nonce disclosure."""

    receipt_id: bytes
    statement: GoldilocksAirTraceMapStatementReferenceV3
    precommitment: GoldilocksAirTraceMapPrecommitmentReferenceV3
    validator_nonce_commitment: bytes
    nonce_context_digest: bytes
    deadline_monotonic_ns: int
    _factory_token: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ProofV3VerificationError(
            "Goldilocks AIR trace-map receipts must be sealed by the validator factory"
        )

    @classmethod
    def _construct(
        cls,
        *,
        receipt_id: bytes,
        statement: GoldilocksAirTraceMapStatementReferenceV3,
        precommitment: GoldilocksAirTraceMapPrecommitmentReferenceV3,
        validator_nonce_commitment: bytes,
        nonce_context_digest: bytes,
        deadline_monotonic_ns: int,
        _factory_token: object | None = None,
    ) -> "GoldilocksAirTraceMapReceiptReferenceV3":
        if _factory_token is not _RECEIPT_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map receipts must be sealed by the validator factory"
            )
        result = object.__new__(cls)
        for name, value in (
            ("receipt_id", receipt_id),
            ("statement", statement),
            ("precommitment", precommitment),
            ("validator_nonce_commitment", validator_nonce_commitment),
            ("nonce_context_digest", nonce_context_digest),
            ("deadline_monotonic_ns", deadline_monotonic_ns),
        ):
            object.__setattr__(result, name, value)
        object.__setattr__(result, "_factory_token", _RECEIPT_FACTORY_TOKEN)
        result.require_factory_provenance()
        return result

    def require_factory_provenance(self) -> None:
        if self._factory_token is not _RECEIPT_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map receipt lacks validator factory provenance"
            )
        _fixed32(self.receipt_id, "Goldilocks AIR trace-map receipt identifier", nonzero=True)
        if not isinstance(self.statement, GoldilocksAirTraceMapStatementReferenceV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map receipt statement is malformed"
            )
        self.statement.require_factory_provenance()
        normalized = _require_map_precommitment(
            statement=self.statement,
            precommitment=self.precommitment,
        )
        if self.precommitment.statement is not self.statement:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map receipt precommitment has another statement object"
            )
        if normalized != self.precommitment:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map receipt precommitment was mutated"
            )
        _fixed32(
            self.validator_nonce_commitment,
            "Goldilocks AIR trace-map receipt nonce commitment",
            nonzero=True,
        )
        _fixed32(
            self.nonce_context_digest,
            "Goldilocks AIR trace-map receipt nonce context",
            nonzero=True,
        )
        _monotonic_ns(
            self.deadline_monotonic_ns,
            "Goldilocks AIR trace-map receipt deadline",
        )

    def canonical_bytes(self) -> bytes:
        return (
            self.receipt_id
            + self.statement.digest()
            + self.precommitment.digest()
            + self.validator_nonce_commitment
            + self.nonce_context_digest
            + struct.pack("<Q", self.deadline_monotonic_ns)
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_RECEIPT_DIGEST_DOMAIN + self.canonical_bytes()).digest()

    def require_matching_opening_ticket(self, *, ticket: PostCommitOpeningTicketV3) -> None:
        """Require the exact post-nonce ticket for this sealed request/map."""

        if not isinstance(ticket, PostCommitOpeningTicketV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map receipt opening ticket is malformed"
            )
        statement = self.statement
        if ticket.proof_challenge_id != statement.proof_challenge_id:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map receipt ticket has an unexpected challenge"
            )
        if ticket.precommit_context_digest != statement.precommit_context_digest:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map receipt ticket has an unexpected precommit"
            )
        if ticket.execution_profile_digest != statement.execution_profile_digest:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map receipt ticket has an unexpected profile"
            )
        if ticket.cache_lease_digest != statement.cache_lease_digest:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map receipt ticket has an unexpected cache lease"
            )
        if ticket.commitment_envelope_digest != statement.commitment_envelope_digest:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map receipt ticket has an unexpected envelope"
            )
        expected_nonce_commitment = commit_validator_nonce_v3(
            validator_nonce=ticket.validator_nonce,
            nonce_context_digest=self.nonce_context_digest,
        )
        if expected_nonce_commitment != self.validator_nonce_commitment:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map receipt ticket nonce is not validator committed"
            )


def _derive_receipt_id(
    *,
    statement: GoldilocksAirTraceMapStatementReferenceV3,
    precommitment: GoldilocksAirTraceMapPrecommitmentReferenceV3,
) -> bytes:
    for counter in range(16):
        identifier = hashlib.sha256(
            _RECEIPT_IDENTIFIER_DOMAIN
            + secrets.token_bytes(32)
            + statement.digest()
            + precommitment.digest()
            + struct.pack("<I", counter)
        ).digest()
        if identifier != bytes(32):
            return identifier
    raise ProofV3VerificationError("unable to mint a Goldilocks AIR trace-map receipt")


def _receipt_logical_key(
    *,
    statement: GoldilocksAirTraceMapStatementReferenceV3,
) -> bytes:
    """Return the one-root-per-request identity for a receipt store.

    The statement already binds the challenge, precommit context, cache lease,
    request, and accepted envelope.  Including the miner-provided map root
    here would allow two different roots to be sealed for one request by a
    future shared/retry store, creating a post-nonce choice.
    """

    return hashlib.sha256(
        _RECEIPT_LOGICAL_KEY_DOMAIN + statement.digest()
    ).digest()


def _make_goldilocks_air_trace_map_receipt_reference_v3(
    *,
    accepted_statement: _GoldilocksAirTraceMapAcceptedStatementCapabilityReferenceV3,
    precommitment: GoldilocksAirTraceMapPrecommitmentReferenceV3,
    deadline_monotonic_ns: int,
) -> GoldilocksAirTraceMapReceiptReferenceV3:
    if not isinstance(
        accepted_statement,
        _GoldilocksAirTraceMapAcceptedStatementCapabilityReferenceV3,
    ):
        raise ProofV3VerificationError(
            "Goldilocks AIR trace-map receipt requires coordinator acceptance"
        )
    accepted_statement.require_factory_provenance()
    statement = accepted_statement.statement
    retained_precommitment = _require_map_precommitment(
        statement=statement,
        precommitment=precommitment,
    )
    return GoldilocksAirTraceMapReceiptReferenceV3._construct(
        receipt_id=_derive_receipt_id(
            statement=statement,
            precommitment=retained_precommitment,
        ),
        statement=statement,
        precommitment=retained_precommitment,
        validator_nonce_commitment=accepted_statement.validator_nonce_commitment,
        nonce_context_digest=accepted_statement.nonce_context_digest,
        deadline_monotonic_ns=_monotonic_ns(
            deadline_monotonic_ns,
            "Goldilocks AIR trace-map receipt deadline",
        ),
        _factory_token=_RECEIPT_FACTORY_TOKEN,
    )


def _derive_slot_proof_nonce(
    *,
    validator_nonce: bytes,
    statement: GoldilocksAirTraceMapStatementReferenceV3,
    precommitment: GoldilocksAirTraceMapPrecommitmentReferenceV3,
    selection: GoldilocksAirTraceHardSelectionReferenceV3,
    slot_index: int,
) -> bytes:
    """Bind a selected slot's inner AIR transcript to its map and selection."""

    nonce = _fixed32(
        validator_nonce,
        "Goldilocks AIR trace-map slot validator nonce",
        nonzero=True,
    )
    if not isinstance(selection, GoldilocksAirTraceHardSelectionReferenceV3):
        raise ProofV3VerificationError("Goldilocks AIR trace-map slot selection is malformed")
    if selection.statement_digest != statement.digest():
        raise ProofV3VerificationError(
            "Goldilocks AIR trace-map slot selection belongs to another statement"
        )
    if type(slot_index) is not int or slot_index < 0 or slot_index >= len(statement.slot_universe):
        raise ProofV3VerificationError("Goldilocks AIR trace-map slot index is malformed")
    if slot_index not in selection.slot_indices:
        raise ProofV3VerificationError("Goldilocks AIR trace-map slot is not selected")
    retained_precommitment = _require_map_precommitment(
        statement=statement,
        precommitment=precommitment,
    )
    result = hashlib.sha256(
        _SLOT_PROOF_NONCE_DOMAIN
        + nonce
        + statement.digest()
        + retained_precommitment.digest()
        + selection.digest()
        + struct.pack("<I", slot_index)
    ).digest()
    # A zero transcript nonce is invalid for the inner AIR type.  The raw
    # validator nonce is already fixed before the map is sealed, so this
    # extraordinarily unlikely rehash cannot be miner-selected.
    if result == bytes(32):
        result = hashlib.sha256(
            _SLOT_PROOF_NONCE_DOMAIN
            + b"\x01"
            + nonce
            + statement.digest()
            + retained_precommitment.digest()
            + selection.digest()
            + struct.pack("<I", slot_index)
        ).digest()
    if result == bytes(32):
        raise ProofV3VerificationError("unable to derive Goldilocks AIR trace-map slot nonce")
    return result


def derive_goldilocks_air_trace_map_slot_proof_nonce_reference_v3(
    *,
    validator_nonce: bytes,
    statement: GoldilocksAirTraceMapStatementReferenceV3,
    precommitment: GoldilocksAirTraceMapPrecommitmentReferenceV3,
    selection: GoldilocksAirTraceHardSelectionReferenceV3,
    slot_index: int,
) -> bytes:
    """Derive one public post-reveal inner AIR nonce for an exact map slot.

    This helper is usable by the isolated reference prover.  It replays the
    semantic selection from the validator nonce before deriving the inner
    transcript, so a caller cannot weaken the selected-slot set.
    """

    try:
        if not isinstance(statement, GoldilocksAirTraceMapStatementReferenceV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map slot statement is malformed"
            )
        statement.require_factory_provenance()
        verified_slots = verify_goldilocks_air_trace_hard_selection_reference_v3(
            selection,
            validator_nonce=validator_nonce,
            statement=statement,
        )
        if slot_index not in verified_slots:
            raise ProofV3VerificationError("Goldilocks AIR trace-map slot is not selected")
        return _derive_slot_proof_nonce(
            validator_nonce=validator_nonce,
            statement=statement,
            precommitment=precommitment,
            selection=selection,
            slot_index=slot_index,
        )
    except ProofV3VerificationError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "Goldilocks AIR trace-map slot nonce inputs are malformed"
        ) from exc


@dataclass(frozen=True, slots=True, init=False)
class _GoldilocksAirTraceMapPostRevealCapabilityReferenceV3:
    """Opaque one-use authority for a retained receipt after a hard reveal."""

    receipt: GoldilocksAirTraceMapReceiptReferenceV3
    ticket: PostCommitOpeningTicketV3
    selection: GoldilocksAirTraceHardSelectionReferenceV3
    _factory_token: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ProofV3VerificationError(
            "Goldilocks AIR trace-map post-reveal capabilities are issued by the coordinator"
        )

    @classmethod
    def _construct(
        cls,
        *,
        receipt: GoldilocksAirTraceMapReceiptReferenceV3,
        reveal: NonceRevealV3,
        selection: GoldilocksAirTraceHardSelectionReferenceV3,
        _factory_token: object | None = None,
    ) -> "_GoldilocksAirTraceMapPostRevealCapabilityReferenceV3":
        if _factory_token is not _POST_REVEAL_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map post-reveal capabilities are issued by the coordinator"
            )
        if not isinstance(reveal, NonceRevealV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map nonce reveal is malformed"
            )
        result = object.__new__(cls)
        object.__setattr__(result, "receipt", receipt)
        object.__setattr__(result, "ticket", reveal.opening_ticket())
        object.__setattr__(result, "selection", selection)
        object.__setattr__(result, "_factory_token", _POST_REVEAL_FACTORY_TOKEN)
        result.require_factory_provenance()
        if selection.transcript_digest != reveal.audit_decision.transcript_digest:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map selection has an unexpected transcript"
            )
        return result

    def require_factory_provenance(self) -> None:
        if self._factory_token is not _POST_REVEAL_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map post-reveal capability lacks coordinator provenance"
            )
        if not isinstance(self.receipt, GoldilocksAirTraceMapReceiptReferenceV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map post-reveal receipt is malformed"
            )
        self.receipt.require_factory_provenance()
        self.receipt.require_matching_opening_ticket(ticket=self.ticket)
        if not isinstance(self.selection, GoldilocksAirTraceHardSelectionReferenceV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map post-reveal selection is malformed"
            )
        if self.selection.statement_digest != self.receipt.statement.digest():
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map post-reveal selection belongs to another statement"
            )
        verify_goldilocks_air_trace_hard_selection_reference_v3(
            self.selection,
            validator_nonce=self.ticket.validator_nonce,
            statement=self.receipt.statement,
        )


class _GoldilocksAirTraceMapReceiptStoreReferenceV3:
    """Bounded one-receipt store owned by one reference coordinator.

    The coordinator creates one store per issued request, so retaining exactly
    one active receipt, one receipt-id tombstone, and one logical-key tombstone
    is sufficient for its lifecycle and cannot accumulate with traffic.  A
    production multi-request store still requires a durable bounded-TTL CAS
    implementation; this class is intentionally not that API.
    """

    __slots__ = ("_active_receipt", "_consumed_receipt_id", "_lock", "_logical_key")

    def __init__(self) -> None:
        self._active_receipt: GoldilocksAirTraceMapReceiptReferenceV3 | None = None
        self._consumed_receipt_id: bytes | None = None
        self._logical_key: bytes | None = None
        self._lock = threading.Lock()

    def seal(
        self,
        *,
        accepted_statement: _GoldilocksAirTraceMapAcceptedStatementCapabilityReferenceV3,
        precommitment: GoldilocksAirTraceMapPrecommitmentReferenceV3,
        deadline_monotonic_ns: int,
    ) -> GoldilocksAirTraceMapReceiptReferenceV3:
        """Atomically retain the exact all-slot map root before nonce disclosure."""

        receipt = _make_goldilocks_air_trace_map_receipt_reference_v3(
            accepted_statement=accepted_statement,
            precommitment=precommitment,
            deadline_monotonic_ns=deadline_monotonic_ns,
        )
        logical_key = _receipt_logical_key(statement=receipt.statement)
        with self._lock:
            if self._active_receipt is not None or self._logical_key is not None:
                raise ProofV3VerificationError(
                    "Goldilocks AIR trace-map precommitment is already sealed"
                )
            self._active_receipt = receipt
            # Keep this fixed-size statement tombstone after consume/discard.
            # A randomized receipt ID is not a one-root-per-statement replay
            # boundary; no unbounded set is needed because this store owns
            # exactly one request lifecycle.
            self._logical_key = logical_key
        return receipt

    def verify_once(
        self,
        *,
        capability: _GoldilocksAirTraceMapPostRevealCapabilityReferenceV3,
        bundle: object,
    ) -> None:
        """Claim the receipt before any final-map or nested-proof parsing."""

        if not isinstance(
            capability,
            _GoldilocksAirTraceMapPostRevealCapabilityReferenceV3,
        ):
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map receipt requires a coordinator post-reveal capability"
            )
        with self._lock:
            try:
                capability.require_factory_provenance()
                receipt = capability.receipt
                ticket = capability.ticket
                selection = capability.selection
                receipt.require_factory_provenance()
                if self._consumed_receipt_id is not None:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR trace-map receipt has already been consumed"
                    )
                stored = self._active_receipt
                if stored is not receipt:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR trace-map receipt was not retained by this validator"
                    )
                # Any malformed bundle is terminal: retrying on a distinction
                # between map and inner AIR errors would reopen the nonce.
                self._consumed_receipt_id = receipt.receipt_id
                self._active_receipt = None
                receipt.require_matching_opening_ticket(ticket=ticket)
            except ProofV3VerificationError:
                raise
            except (AttributeError, TypeError, ValueError) as exc:
                raise ProofV3VerificationError(
                    "Goldilocks AIR trace-map receipt opening is malformed"
                ) from exc
        try:
            if not isinstance(bundle, GoldilocksAirTraceMapProofBundleReferenceV3):
                raise ProofV3VerificationError(
                    "Goldilocks AIR trace-map receipt bundle is malformed"
                )
            # Reconstruct to re-run structural checks if a caller bypassed a
            # frozen dataclass initializer before the receipt was claimed.
            normalized_bundle = GoldilocksAirTraceMapProofBundleReferenceV3(
                map_opening=bundle.map_opening,
                air_proofs=bundle.air_proofs,
                abi_id=bundle.abi_id,
                format_version=bundle.format_version,
            )
            if normalized_bundle != bundle:
                raise ProofV3VerificationError(
                    "Goldilocks AIR trace-map receipt bundle is malformed"
                )
            selected_precommitments = (
                verify_goldilocks_air_trace_map_opening_reference_v3(
                    normalized_bundle.map_opening,
                    statement=receipt.statement,
                    precommitment=receipt.precommitment,
                    expected_slot_indices=selection.slot_indices,
                )
            )
            if len(selected_precommitments) != len(normalized_bundle.air_proofs):
                raise ProofV3VerificationError(
                    "Goldilocks AIR trace-map receipt has an incomplete proof bundle"
                )
            for slot_index, trace_precommitment, proof in zip(
                selection.slot_indices,
                selected_precommitments,
                normalized_bundle.air_proofs,
                strict=True,
            ):
                verify_goldilocks_air_reference_v3(
                    proof,
                    core=receipt.statement.slot_core(slot_index=slot_index),
                    precommitment=trace_precommitment,
                    validator_nonce=_derive_slot_proof_nonce(
                        validator_nonce=ticket.validator_nonce,
                        statement=receipt.statement,
                        precommitment=receipt.precommitment,
                        selection=selection,
                        slot_index=slot_index,
                    ),
                )
        except ProofV3VerificationError:
            raise
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map receipt proof is malformed"
            ) from exc

    def discard(self, *, receipt: GoldilocksAirTraceMapReceiptReferenceV3) -> None:
        """Release a sealed map root while preserving its one-use tombstone."""

        if not isinstance(receipt, GoldilocksAirTraceMapReceiptReferenceV3):
            raise ProofV3VerificationError("Goldilocks AIR trace-map receipt is malformed")
        with self._lock:
            stored = self._active_receipt
            if stored is receipt:
                self._active_receipt = None
                self._consumed_receipt_id = receipt.receipt_id


class GoldilocksAirTraceMapReferenceCoordinatorStateV3(str, Enum):
    """Reference-only state machine; the underlying session is never exposed."""

    AWAITING_PRECOMMIT = "awaiting_precommit"
    PRECOMMIT_ACCEPTED = "precommit_accepted"
    NONCE_REVEALED = "nonce_revealed"
    LIGHT_REVEALED = "light_revealed"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"
    ABORTED = "aborted"


_TERMINAL_COORDINATOR_STATES = frozenset(
    {
        GoldilocksAirTraceMapReferenceCoordinatorStateV3.LIGHT_REVEALED,
        GoldilocksAirTraceMapReferenceCoordinatorStateV3.VERIFIED,
        GoldilocksAirTraceMapReferenceCoordinatorStateV3.FAILED,
        GoldilocksAirTraceMapReferenceCoordinatorStateV3.EXPIRED,
        GoldilocksAirTraceMapReferenceCoordinatorStateV3.ABORTED,
    }
)


@dataclass(slots=True, init=False)
class GoldilocksAirTraceMapReferencePrecommitmentCoordinatorV3:
    """Own an all-map receipt and session so nonce chronology cannot be split.

    The API deliberately does not accept a raw nonce, ticket, statement,
    selected slots, or expected map root during final verification.  All of
    those values are derived or retained by this coordinator.
    """

    _qualified_profile: QualifiedExecutionProfileV3
    _proof_arrival_budget_ns: int
    _accepted_statement: _GoldilocksAirTraceMapAcceptedStatementCapabilityReferenceV3 | None
    _receipt: GoldilocksAirTraceMapReceiptReferenceV3 | None
    _receipt_store: _GoldilocksAirTraceMapReceiptStoreReferenceV3
    _audit_decision: PostCommitAuditDecisionV3 | None
    _post_reveal_capability: _GoldilocksAirTraceMapPostRevealCapabilityReferenceV3 | None
    _reveal: NonceRevealV3 | None
    _reveal_monotonic_ns: int | None
    _session: ProofV3ChallengeSession
    _state: GoldilocksAirTraceMapReferenceCoordinatorStateV3
    _deadline_monotonic_ns: int | None
    _lock: threading.Lock

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ProofV3VerificationError(
            "Goldilocks AIR trace-map coordinators must be issued by the validator"
        )

    @classmethod
    def _construct(
        cls,
        *,
        qualified_profile: QualifiedExecutionProfileV3,
        proof_arrival_budget_ns: int,
        session: ProofV3ChallengeSession,
        _factory_token: object | None = None,
    ) -> "GoldilocksAirTraceMapReferencePrecommitmentCoordinatorV3":
        if _factory_token is not _COORDINATOR_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map coordinators must be issued by the validator"
            )
        if not isinstance(qualified_profile, QualifiedExecutionProfileV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map coordinator has an unexpected profile"
            )
        qualified_profile.require_qualification_provenance()
        if not isinstance(session, ProofV3ChallengeSession):
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map coordinator has an unexpected session"
            )
        result = object.__new__(cls)
        object.__setattr__(result, "_qualified_profile", qualified_profile)
        object.__setattr__(
            result,
            "_proof_arrival_budget_ns",
            _arrival_budget_ns(proof_arrival_budget_ns),
        )
        object.__setattr__(result, "_accepted_statement", None)
        object.__setattr__(result, "_receipt", None)
        object.__setattr__(
            result,
            "_receipt_store",
            _GoldilocksAirTraceMapReceiptStoreReferenceV3(),
        )
        object.__setattr__(result, "_audit_decision", None)
        object.__setattr__(result, "_post_reveal_capability", None)
        object.__setattr__(result, "_reveal", None)
        object.__setattr__(result, "_reveal_monotonic_ns", None)
        object.__setattr__(result, "_session", session)
        object.__setattr__(
            result,
            "_state",
            GoldilocksAirTraceMapReferenceCoordinatorStateV3.AWAITING_PRECOMMIT,
        )
        object.__setattr__(result, "_deadline_monotonic_ns", None)
        object.__setattr__(result, "_lock", threading.Lock())
        return result

    @classmethod
    def issue(
        cls,
        *,
        qualified_profile: QualifiedExecutionProfileV3,
        proof_challenge_id: bytes,
        validator_identity_digest: bytes,
        miner_identity_digest: bytes,
        prompt_token_ids: Sequence[int],
        sampler_config_digest: bytes,
        runtime_policy: RuntimeHardAuditPolicyV3,
        proof_arrival_budget_ns: int = DEFAULT_PROOF_ARRIVAL_BUDGET_NS_V3,
    ) -> tuple[
        "GoldilocksAirTraceMapReferencePrecommitmentCoordinatorV3",
        ValidatorExecutionRequestContextV3,
    ]:
        """Issue one hidden validator-owned session for the map receipt gate."""

        budget = _arrival_budget_ns(proof_arrival_budget_ns)
        session, request_context = ProofV3ChallengeSession.issue(
            qualified_profile=qualified_profile,
            proof_challenge_id=proof_challenge_id,
            validator_identity_digest=validator_identity_digest,
            miner_identity_digest=miner_identity_digest,
            prompt_token_ids=prompt_token_ids,
            sampler_config_digest=sampler_config_digest,
            runtime_policy=runtime_policy,
            proof_arrival_budget_ns=budget,
        )
        return (
            cls._construct(
                qualified_profile=qualified_profile,
                proof_arrival_budget_ns=budget,
                session=session,
                _factory_token=_COORDINATOR_FACTORY_TOKEN,
            ),
            request_context,
        )

    @property
    def state(self) -> GoldilocksAirTraceMapReferenceCoordinatorStateV3:
        with self._lock:
            return self._state

    def _terminate_locked(
        self,
        state: GoldilocksAirTraceMapReferenceCoordinatorStateV3,
    ) -> None:
        if self._receipt is not None:
            self._receipt_store.discard(receipt=self._receipt)
        self._session.fail_closed()
        self._state = state
        self._accepted_statement = None
        self._receipt = None
        self._audit_decision = None
        self._post_reveal_capability = None
        self._reveal = None
        self._reveal_monotonic_ns = None

    def accept_precommit(
        self,
        *,
        encoded_envelope: bytes,
        map_precommitment: GoldilocksAirTraceMapPrecommitmentReferenceV3,
        observed_output: ObservedExecutionOutputV3,
        last_visible_token_monotonic_ns: int,
        received_monotonic_ns: int,
    ) -> None:
        """Accept envelope and seal a complete map root before nonce disclosure."""

        with self._lock:
            if self._state != GoldilocksAirTraceMapReferenceCoordinatorStateV3.AWAITING_PRECOMMIT:
                raise ProofV3VerificationError(
                    "Goldilocks AIR trace-map coordinator does not accept another precommit"
                )
            try:
                last_visible = _monotonic_ns(
                    last_visible_token_monotonic_ns,
                    "last_visible_token_monotonic_ns",
                )
                deadline = last_visible + self._proof_arrival_budget_ns
                if deadline >= 1 << 63:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR trace-map coordinator arrival deadline overflows"
                    )
                received = _monotonic_ns(
                    received_monotonic_ns,
                    "received_monotonic_ns",
                )
                self._session.accept_precommit_bytes(
                    encoded_envelope=encoded_envelope,
                    observed_output=observed_output,
                    last_visible_token_monotonic_ns=last_visible,
                    received_monotonic_ns=received,
                )
                envelope = commitment_envelope_from_bytes(encoded_envelope)
                binding = validate_execution_envelope_against_precommit_v3(
                    profile=self._qualified_profile.profile,
                    envelope=envelope,
                    expected_static_manifest_digest=(
                        self._qualified_profile.expected_static_manifest_digest
                    ),
                    expected_execution_profile_digest=(
                        self._qualified_profile.expected_execution_profile_digest
                    ),
                    precommit_context=self._session.precommit_context,
                    output_binding=observed_output.derive_output_binding(),
                )
                accepted_statement = _accept_validator_map_statement_reference_v3(
                    qualified_profile=self._qualified_profile,
                    envelope=envelope,
                    envelope_binding=binding,
                )
                receipt = self._receipt_store.seal(
                    accepted_statement=accepted_statement,
                    precommitment=map_precommitment,
                    deadline_monotonic_ns=deadline,
                )
            except ProofV3VerificationError:
                self._terminate_locked(
                    GoldilocksAirTraceMapReferenceCoordinatorStateV3.FAILED
                )
                raise
            except Exception as exc:
                self._terminate_locked(
                    GoldilocksAirTraceMapReferenceCoordinatorStateV3.FAILED
                )
                raise ProofV3VerificationError(
                    "Goldilocks AIR trace-map coordinator precommitment is malformed"
                ) from exc
            self._accepted_statement = accepted_statement
            self._receipt = receipt
            self._deadline_monotonic_ns = deadline
            self._state = GoldilocksAirTraceMapReferenceCoordinatorStateV3.PRECOMMIT_ACCEPTED

    def _select_audit_tier_locked(
        self,
        *,
        selected_monotonic_ns: int,
    ) -> PostCommitAuditDecisionV3:
        if self._state != GoldilocksAirTraceMapReferenceCoordinatorStateV3.PRECOMMIT_ACCEPTED:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map coordinator has no sealed map receipt"
            )
        if self._audit_decision is not None:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map coordinator already selected its audit tier"
            )
        if self._receipt is None:
            self._terminate_locked(
                GoldilocksAirTraceMapReferenceCoordinatorStateV3.FAILED
            )
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map coordinator lost its sealed map receipt"
            )
        try:
            decision = self._session.select_audit_tier_once(
                selected_monotonic_ns=_monotonic_ns(
                    selected_monotonic_ns,
                    "selected_monotonic_ns",
                )
            )
        except ProofV3VerificationError:
            self._terminate_locked(
                GoldilocksAirTraceMapReferenceCoordinatorStateV3.FAILED
            )
            raise
        except Exception as exc:
            self._terminate_locked(
                GoldilocksAirTraceMapReferenceCoordinatorStateV3.FAILED
            )
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map audit-tier selection is malformed"
            ) from exc
        if decision.hard_audit_selected:
            self._audit_decision = decision
            return decision
        self._receipt_store.discard(receipt=self._receipt)
        self._accepted_statement = None
        self._receipt = None
        self._post_reveal_capability = None
        self._state = GoldilocksAirTraceMapReferenceCoordinatorStateV3.LIGHT_REVEALED
        return decision

    def select_audit_tier_once(
        self,
        *,
        selected_monotonic_ns: int,
    ) -> PostCommitAuditDecisionV3:
        """Select hard/light locally without disclosing the nonce on light."""

        with self._lock:
            return self._select_audit_tier_locked(
                selected_monotonic_ns=selected_monotonic_ns
            )

    def reveal_nonce_once(self, *, revealed_monotonic_ns: int) -> NonceRevealV3:
        """Reveal once only for a locally selected hard tier."""

        with self._lock:
            if self._state != GoldilocksAirTraceMapReferenceCoordinatorStateV3.PRECOMMIT_ACCEPTED:
                if self._state == GoldilocksAirTraceMapReferenceCoordinatorStateV3.LIGHT_REVEALED:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR trace-map light selection does not reveal a nonce"
                    )
                raise ProofV3VerificationError(
                    "Goldilocks AIR trace-map coordinator has no sealed map receipt"
                )
            if self._audit_decision is None:
                decision = self._select_audit_tier_locked(
                    selected_monotonic_ns=revealed_monotonic_ns
                )
                if not decision.hard_audit_selected:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR trace-map light selection does not reveal a nonce"
                    )
            if self._receipt is None:
                self._terminate_locked(
                    GoldilocksAirTraceMapReferenceCoordinatorStateV3.FAILED
                )
                raise ProofV3VerificationError(
                    "Goldilocks AIR trace-map coordinator lost its sealed map receipt"
                )
            try:
                revealed = _monotonic_ns(revealed_monotonic_ns, "revealed_monotonic_ns")
                reveal = self._session.reveal_nonce_once(revealed_monotonic_ns=revealed)
                selection = derive_goldilocks_air_trace_hard_selection_reference_v3(
                    validator_nonce=reveal.validator_nonce,
                    statement=self._receipt.statement,
                )
                if selection.transcript_digest != reveal.audit_decision.transcript_digest:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR trace-map selection has an unexpected transcript"
                    )
                capability = _GoldilocksAirTraceMapPostRevealCapabilityReferenceV3._construct(
                    receipt=self._receipt,
                    reveal=reveal,
                    selection=selection,
                    _factory_token=_POST_REVEAL_FACTORY_TOKEN,
                )
            except ProofV3VerificationError:
                self._terminate_locked(
                    GoldilocksAirTraceMapReferenceCoordinatorStateV3.FAILED
                )
                raise
            except Exception as exc:
                self._terminate_locked(
                    GoldilocksAirTraceMapReferenceCoordinatorStateV3.FAILED
                )
                raise ProofV3VerificationError(
                    "Goldilocks AIR trace-map coordinator nonce reveal is malformed"
                ) from exc
            self._reveal = reveal
            self._reveal_monotonic_ns = revealed
            self._post_reveal_capability = capability
            self._state = GoldilocksAirTraceMapReferenceCoordinatorStateV3.NONCE_REVEALED
            return reveal

    def verify_proof_once(
        self,
        *,
        bundle: object,
        received_monotonic_ns: int,
    ) -> None:
        """Claim and check one hard map proof; any failure is terminal."""

        with self._lock:
            if self._state != GoldilocksAirTraceMapReferenceCoordinatorStateV3.NONCE_REVEALED:
                if self._state == GoldilocksAirTraceMapReferenceCoordinatorStateV3.LIGHT_REVEALED:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR trace-map light selection does not accept a hard proof"
                    )
                raise ProofV3VerificationError(
                    "Goldilocks AIR trace-map coordinator does not accept another proof"
                )
            try:
                received = _monotonic_ns(received_monotonic_ns, "received_monotonic_ns")
                deadline = self._deadline_monotonic_ns
                if deadline is None or received > deadline:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR trace-map proof arrived after its deadline"
                    )
                if (
                    self._reveal_monotonic_ns is None
                    or received < self._reveal_monotonic_ns
                ):
                    raise ProofV3VerificationError(
                        "Goldilocks AIR trace-map proof arrived before its nonce reveal"
                    )
                capability = self._post_reveal_capability
                if capability is None or self._receipt is None or self._reveal is None:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR trace-map coordinator lost its proof state"
                    )
                self._state = GoldilocksAirTraceMapReferenceCoordinatorStateV3.VERIFYING
            except ProofV3VerificationError:
                self._terminate_locked(
                    GoldilocksAirTraceMapReferenceCoordinatorStateV3.FAILED
                )
                raise
        try:
            self._receipt_store.verify_once(capability=capability, bundle=bundle)
        except ProofV3VerificationError:
            with self._lock:
                if self._state == GoldilocksAirTraceMapReferenceCoordinatorStateV3.VERIFYING:
                    self._terminate_locked(
                        GoldilocksAirTraceMapReferenceCoordinatorStateV3.FAILED
                    )
            raise
        except Exception as exc:
            with self._lock:
                if self._state == GoldilocksAirTraceMapReferenceCoordinatorStateV3.VERIFYING:
                    self._terminate_locked(
                        GoldilocksAirTraceMapReferenceCoordinatorStateV3.FAILED
                    )
            # The outer coordinator is the error-type firewall for the
            # verifier integration: an unexpected parser/backend exception is
            # still a terminal proof failure, never a retryable "not
            # requested" condition for scoring/probation callers.
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map receipt proof is malformed"
            ) from exc
        with self._lock:
            if self._state != GoldilocksAirTraceMapReferenceCoordinatorStateV3.VERIFYING:
                raise ProofV3VerificationError(
                    "Goldilocks AIR trace-map coordinator terminated during verification"
                )
            # The retained session belongs only to the conformance chronology
            # gate.  It must not remain usable after the map verdict.
            self._session.fail_closed()
            self._state = GoldilocksAirTraceMapReferenceCoordinatorStateV3.VERIFIED
            self._accepted_statement = None
            self._receipt = None
            self._audit_decision = None
            self._post_reveal_capability = None
            self._reveal = None
            self._reveal_monotonic_ns = None

    def expire(self, *, now_monotonic_ns: int) -> None:
        """Fail a pending request after its validator-local arrival deadline.

        Once a proof has been admitted and atomically claimed (``VERIFYING``),
        it may finish later without an expiry race.  Production still needs a
        bounded verifier queue and durable ownership record.
        """

        with self._lock:
            if self._state in _TERMINAL_COORDINATOR_STATES or self._state == (
                GoldilocksAirTraceMapReferenceCoordinatorStateV3.VERIFYING
            ):
                return
            now = _monotonic_ns(now_monotonic_ns, "now_monotonic_ns")
            deadline = self._deadline_monotonic_ns
            if deadline is None or now <= deadline:
                return
            self._terminate_locked(GoldilocksAirTraceMapReferenceCoordinatorStateV3.EXPIRED)

    def fail_closed(self) -> None:
        """Dispose retained state after any transport or coordinator failure."""

        with self._lock:
            if self._state not in _TERMINAL_COORDINATOR_STATES:
                self._terminate_locked(GoldilocksAirTraceMapReferenceCoordinatorStateV3.FAILED)

    def abort(self) -> None:
        """Dispose a cancelled request without accepting a later proof."""

        with self._lock:
            if self._state not in _TERMINAL_COORDINATOR_STATES:
                self._terminate_locked(GoldilocksAirTraceMapReferenceCoordinatorStateV3.ABORTED)


__all__ = [
    "GOLDILOCKS_AIR_TRACE_MAP_RECEIPT_REFERENCE_ABI_V3",
    "GOLDILOCKS_AIR_TRACE_MAP_RECEIPT_REFERENCE_FORMAT_VERSION_V3",
    "GoldilocksAirTraceMapProofBundleReferenceV3",
    "GoldilocksAirTraceMapReceiptReferenceV3",
    "GoldilocksAirTraceMapReferenceCoordinatorStateV3",
    "GoldilocksAirTraceMapReferencePrecommitmentCoordinatorV3",
    "derive_goldilocks_air_trace_map_slot_proof_nonce_reference_v3",
]
