"""Validator-owned receipt gate for the unregistered Goldilocks AIR reference.

The bounded CPU AIR/FRI reference deliberately accepts a raw statement core,
precommitment, and nonce so its algebra can be tested in isolation.  That is
not an acceptable production integration boundary: a miner must never select
the statement binding, expected trace root, or nonce used by verification.

This module supplies a reference-only control-plane bridge with the required
chronology:

* it derives the AIR statement from a validator-qualified profile, accepted
  envelope, parsed signed artifacts, and one validator-owned reference
  coordinate;
* it seals the submitted trace precommitment into validator-owned receipt
  state before a nonce is exposed; and
* it consumes that receipt exactly once through an opaque capability minted
  only after the validator-owned session reveals a hard nonce decision.

It is deliberately not imported by the V3 package facade, payload, session,
miner, validator, or adapter registry.  The coordinator below is a bounded
CPU conformance harness, not a qualified dynamic proof backend.  In
particular, it does not make the AIR reference a full execution proof or a
model-substitution-security claim.
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
from verallm.proof_v3.constraint_program import (
    GoldilocksConstraintProgramBundleV3,
)
from verallm.proof_v3.constraint_system import (
    GoldilocksConstraintCoordinateV3,
    expected_goldilocks_constraint_universe_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_air_reference import (
    GoldilocksAirStatementCoreReferenceV3,
    GoldilocksAirTracePrecommitmentReferenceV3,
    verify_goldilocks_air_reference_v3,
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


GOLDILOCKS_AIR_RECEIPT_REFERENCE_ABI_V3: Final = (
    "goldilocks.air_receipt.reference.v1"
)
GOLDILOCKS_AIR_RECEIPT_REFERENCE_FORMAT_VERSION_V3: Final = 1

_STATEMENT_BINDING_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR/V1/VALIDATOR_STATEMENT/SHA256"
)
_STATEMENT_DIGEST_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR/V1/VALIDATOR_STATEMENT_DIGEST/SHA256"
)
_RECEIPT_IDENTIFIER_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR/V1/RECEIPT_IDENTIFIER/SHA256"
)
_RECEIPT_DIGEST_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR/V1/RECEIPT_DIGEST/SHA256"
)
_RECEIPT_LOGICAL_KEY_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR/V1/RECEIPT_LOGICAL_KEY/SHA256"
)
_STATEMENT_FACTORY_TOKEN = object()
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
    """Require the receipt inputs to be the validator's accepted statement."""

    if not isinstance(qualified_profile, QualifiedExecutionProfileV3):
        raise ProofV3VerificationError(
            "Goldilocks AIR receipt requires a qualified execution profile"
        )
    qualified_profile.require_qualification_provenance()
    if not isinstance(envelope, ProofV3CommitmentEnvelope):
        raise ProofV3VerificationError("Goldilocks AIR receipt envelope is malformed")
    if not isinstance(envelope_binding, EnvelopeBindingV3):
        raise ProofV3VerificationError(
            "Goldilocks AIR receipt envelope binding is malformed"
        )
    profile = qualified_profile.profile
    profile_digest = profile.digest()
    if envelope_binding.profile_digest != profile_digest:
        raise ProofV3VerificationError(
            "Goldilocks AIR receipt binding has an unexpected profile"
        )
    precommit_context = envelope_binding.precommit_context
    request_binding = envelope_binding.request_binding
    if request_binding.precommit_context != precommit_context:
        raise ProofV3VerificationError(
            "Goldilocks AIR receipt request binding has an unexpected precommit"
        )
    if (
        envelope.static_manifest_digest != profile.static_manifest_digest
        or envelope.execution_profile_digest != profile_digest
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
            "Goldilocks AIR receipt envelope is not the accepted validator statement"
        )


@dataclass(frozen=True, slots=True, init=False)
class GoldilocksAirStatementBindingReferenceV3:
    """Factory-minted exact local AIR statement for one signed coordinate.

    This object is intentionally a reference-only validator view.  It binds a
    single whole-program trace to one globally derived atomic coordinate; the
    atomic coordinate remains in the binding even though the parsed program
    enforces every atomic constraint for its layout.  A native backend may
    later aggregate same-layout traces only under a separately reviewed ABI.
    """

    core: GoldilocksAirStatementCoreReferenceV3
    coordinate: GoldilocksConstraintCoordinateV3
    proof_challenge_id: bytes
    static_manifest_digest: bytes
    execution_profile_digest: bytes
    precommit_context_digest: bytes
    cache_lease_digest: bytes
    request_digest: bytes
    commitment_envelope_digest: bytes
    constraint_system_digest: bytes
    constraint_program_bundle_digest: bytes
    universe_binding_digest: bytes
    binding_digest: bytes
    _factory_token: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ProofV3VerificationError(
            "Goldilocks AIR statements must be derived from validator artifacts"
        )

    @classmethod
    def _construct(
        cls,
        *,
        core: GoldilocksAirStatementCoreReferenceV3,
        coordinate: GoldilocksConstraintCoordinateV3,
        proof_challenge_id: bytes,
        static_manifest_digest: bytes,
        execution_profile_digest: bytes,
        precommit_context_digest: bytes,
        cache_lease_digest: bytes,
        request_digest: bytes,
        commitment_envelope_digest: bytes,
        constraint_system_digest: bytes,
        constraint_program_bundle_digest: bytes,
        universe_binding_digest: bytes,
        binding_digest: bytes,
        _factory_token: object | None = None,
    ) -> "GoldilocksAirStatementBindingReferenceV3":
        if _factory_token is not _STATEMENT_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "Goldilocks AIR statements must be derived from validator artifacts"
            )
        result = object.__new__(cls)
        object.__setattr__(result, "core", core)
        object.__setattr__(result, "coordinate", coordinate)
        object.__setattr__(result, "proof_challenge_id", proof_challenge_id)
        object.__setattr__(result, "static_manifest_digest", static_manifest_digest)
        object.__setattr__(result, "execution_profile_digest", execution_profile_digest)
        object.__setattr__(result, "precommit_context_digest", precommit_context_digest)
        object.__setattr__(result, "cache_lease_digest", cache_lease_digest)
        object.__setattr__(result, "request_digest", request_digest)
        object.__setattr__(result, "commitment_envelope_digest", commitment_envelope_digest)
        object.__setattr__(result, "constraint_system_digest", constraint_system_digest)
        object.__setattr__(
            result,
            "constraint_program_bundle_digest",
            constraint_program_bundle_digest,
        )
        object.__setattr__(result, "universe_binding_digest", universe_binding_digest)
        object.__setattr__(result, "binding_digest", binding_digest)
        object.__setattr__(result, "_factory_token", _STATEMENT_FACTORY_TOKEN)
        result.require_factory_provenance()
        return result

    def require_factory_provenance(self) -> None:
        """Reject a forged/mutated reference statement before it is sealed."""

        if self._factory_token is not _STATEMENT_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "Goldilocks AIR statement lacks validator factory provenance"
            )
        if not isinstance(self.core, GoldilocksAirStatementCoreReferenceV3):
            raise ProofV3VerificationError("Goldilocks AIR statement core is malformed")
        if not isinstance(self.coordinate, GoldilocksConstraintCoordinateV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR statement coordinate is malformed"
            )
        for value, name in (
            (self.proof_challenge_id, "Goldilocks AIR statement challenge"),
            (self.static_manifest_digest, "Goldilocks AIR statement static manifest"),
            (self.execution_profile_digest, "Goldilocks AIR statement profile"),
            (self.precommit_context_digest, "Goldilocks AIR statement precommit"),
            (self.cache_lease_digest, "Goldilocks AIR statement cache lease"),
            (self.request_digest, "Goldilocks AIR statement request"),
            (self.commitment_envelope_digest, "Goldilocks AIR statement envelope"),
            (self.constraint_system_digest, "Goldilocks AIR statement system"),
            (
                self.constraint_program_bundle_digest,
                "Goldilocks AIR statement program bundle",
            ),
            (self.universe_binding_digest, "Goldilocks AIR statement universe"),
            (self.binding_digest, "Goldilocks AIR statement binding"),
        ):
            _fixed32(value, name, nonzero=True)
        if self.core.validator_binding_digest != self.binding_digest:
            raise ProofV3VerificationError(
                "Goldilocks AIR statement core has an unexpected validator binding"
            )

    def canonical_bytes(self) -> bytes:
        """Return a non-wire identity for receipt and regression binding."""

        return (
            self.proof_challenge_id
            + self.static_manifest_digest
            + self.execution_profile_digest
            + self.precommit_context_digest
            + self.cache_lease_digest
            + self.request_digest
            + self.commitment_envelope_digest
            + self.constraint_system_digest
            + self.constraint_program_bundle_digest
            + self.universe_binding_digest
            + self.coordinate.canonical_bytes()
            + self.core.program_digest
            + struct.pack("<I", self.core.token_count)
            + self.binding_digest
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_STATEMENT_DIGEST_DOMAIN + self.canonical_bytes()).digest()


def derive_goldilocks_air_statement_binding_reference_v3(
    *,
    qualified_profile: QualifiedExecutionProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    envelope_binding: EnvelopeBindingV3,
    coordinate: GoldilocksConstraintCoordinateV3,
) -> GoldilocksAirStatementBindingReferenceV3:
    """Derive one exact pre-nonce AIR statement from validator-owned inputs.

    The artifact system/program inputs are intentionally read only from the
    already-qualified profile registration.  This low-level reference factory
    validates a supplied coordinate against that exact universe; the
    coordinator below derives its coordinate itself and never accepts one from
    a caller.
    """

    try:
        _require_bound_envelope(
            qualified_profile=qualified_profile,
            envelope=envelope,
            envelope_binding=envelope_binding,
        )
        if not isinstance(coordinate, GoldilocksConstraintCoordinateV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt coordinate is malformed"
            )
        profile = qualified_profile.profile
        artifacts = qualified_profile.registration.artifacts
        if artifacts.execution_profile_digest != profile.digest():
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt artifacts have an unexpected profile"
            )
        constraint_system = artifacts.constraint_system
        program_bundle = artifacts.constraint_program_bundle
        if not isinstance(program_bundle, GoldilocksConstraintProgramBundleV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt program bundle is malformed"
            )
        program_bundle.validate_constraint_system(
            constraint_system=constraint_system,
            relation=profile.relation_spec,
        )
        universe = expected_goldilocks_constraint_universe_v3(
            profile=profile,
            envelope=envelope,
            constraint_system=constraint_system,
        )
        layout = universe.validate_coordinate(coordinate)
        layout_index, resolved_layout, atomic_index = (
            constraint_system.layout_for_relation_index(coordinate.relation_index)
        )
        if resolved_layout != layout:
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt coordinate resolves to another layout"
            )
        program = program_bundle.programs[layout_index]
        if program.digest() != layout.constraint_program_digest:
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt program does not match its signed layout"
            )
        if (
            program.atomic_constraints[atomic_index].constraint_id
            != layout.atomic_constraint_ids[atomic_index]
        ):
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt program has an unexpected atomic constraint"
            )
        binding_digest = hashlib.sha256(
            _STATEMENT_BINDING_DOMAIN
            + profile.static_manifest_digest
            + profile.digest()
            + envelope_binding.precommit_context.digest()
            + envelope_binding.request_binding.request_digest
            + envelope.digest()
            + constraint_system.digest()
            + program_bundle.digest()
            + program.digest()
            + universe.binding_digest
            + coordinate.canonical_bytes()
        ).digest()
        core = GoldilocksAirStatementCoreReferenceV3(
            validator_binding_digest=binding_digest,
            program=program,
            token_count=coordinate.token_count,
        )
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "Goldilocks AIR receipt statement inputs are malformed"
        ) from exc
    return GoldilocksAirStatementBindingReferenceV3._construct(
        core=core,
        coordinate=coordinate,
        proof_challenge_id=envelope_binding.precommit_context.proof_challenge_id,
        static_manifest_digest=profile.static_manifest_digest,
        execution_profile_digest=profile.digest(),
        precommit_context_digest=envelope_binding.precommit_context.digest(),
        cache_lease_digest=envelope_binding.precommit_context.cache_lease_digest,
        request_digest=envelope_binding.request_binding.request_digest,
        commitment_envelope_digest=envelope.digest(),
        constraint_system_digest=constraint_system.digest(),
        constraint_program_bundle_digest=program_bundle.digest(),
        universe_binding_digest=universe.binding_digest,
        binding_digest=binding_digest,
        _factory_token=_STATEMENT_FACTORY_TOKEN,
    )


def _validator_owned_reference_coordinate_v3(
    *,
    qualified_profile: QualifiedExecutionProfileV3,
    envelope: ProofV3CommitmentEnvelope,
) -> GoldilocksConstraintCoordinateV3:
    """Return the bounded coordinator's canonical validator-owned coordinate.

    This removes caller authority over the one-coordinate conformance harness.
    It is intentionally *not* the production hard-audit selection mechanism:
    a qualified backend must commit an authenticated map of every selectable
    coordinate before the nonce, then derive and open its exact coordinate
    after the nonce.
    """

    try:
        profile = qualified_profile.profile
        universe = expected_goldilocks_constraint_universe_v3(
            profile=profile,
            envelope=envelope,
            constraint_system=qualified_profile.registration.artifacts.constraint_system,
        )
        if len(universe) == 0:
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt coordinate universe is empty"
            )
        return universe[0]
    except ProofV3VerificationError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise ProofV3VerificationError(
            "Goldilocks AIR receipt coordinate universe is malformed"
        ) from exc


@dataclass(frozen=True, slots=True, init=False)
class _GoldilocksAirAcceptedStatementCapabilityReferenceV3:
    """Opaque validator-only evidence that an exact statement was accepted.

    The receipt store deliberately cannot re-derive an accepted envelope from
    miner/caller input.  The coordinator mints this capability only after its
    owned :class:`ProofV3ChallengeSession` has accepted the envelope, and only
    then may the store retain a trace root.
    """

    statement: GoldilocksAirStatementBindingReferenceV3
    validator_nonce_commitment: bytes
    nonce_context_digest: bytes
    _factory_token: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ProofV3VerificationError(
            "Goldilocks AIR accepted statements are issued by the coordinator"
        )

    @classmethod
    def _construct(
        cls,
        *,
        statement: GoldilocksAirStatementBindingReferenceV3,
        validator_nonce_commitment: bytes,
        nonce_context_digest: bytes,
        _factory_token: object | None = None,
    ) -> "_GoldilocksAirAcceptedStatementCapabilityReferenceV3":
        if _factory_token is not _ACCEPTED_STATEMENT_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "Goldilocks AIR accepted statements are issued by the coordinator"
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
                "Goldilocks AIR accepted statement lacks coordinator provenance"
            )
        if not isinstance(self.statement, GoldilocksAirStatementBindingReferenceV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR accepted statement is malformed"
            )
        self.statement.require_factory_provenance()
        _fixed32(
            self.validator_nonce_commitment,
            "Goldilocks AIR accepted statement nonce commitment",
            nonzero=True,
        )
        _fixed32(
            self.nonce_context_digest,
            "Goldilocks AIR accepted statement nonce context",
            nonzero=True,
        )


def _accept_validator_statement_reference_v3(
    *,
    qualified_profile: QualifiedExecutionProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    envelope_binding: EnvelopeBindingV3,
) -> _GoldilocksAirAcceptedStatementCapabilityReferenceV3:
    """Mint the store capability from the coordinator's accepted envelope."""

    coordinate = _validator_owned_reference_coordinate_v3(
        qualified_profile=qualified_profile,
        envelope=envelope,
    )
    statement = derive_goldilocks_air_statement_binding_reference_v3(
        qualified_profile=qualified_profile,
        envelope=envelope,
        envelope_binding=envelope_binding,
        coordinate=coordinate,
    )
    precommit_context = envelope_binding.precommit_context
    return _GoldilocksAirAcceptedStatementCapabilityReferenceV3._construct(
        statement=statement,
        validator_nonce_commitment=precommit_context.validator_nonce_commitment,
        nonce_context_digest=precommit_context.nonce_context_digest(),
        _factory_token=_ACCEPTED_STATEMENT_FACTORY_TOKEN,
    )


@dataclass(frozen=True, slots=True, init=False)
class GoldilocksAirTraceReceiptReferenceV3:
    """Validator-owned trace-root record retained until one opening attempt."""

    receipt_id: bytes
    statement: GoldilocksAirStatementBindingReferenceV3
    precommitment: GoldilocksAirTracePrecommitmentReferenceV3
    validator_nonce_commitment: bytes
    nonce_context_digest: bytes
    _factory_token: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ProofV3VerificationError(
            "Goldilocks AIR receipts must be sealed by the validator factory"
        )

    @classmethod
    def _construct(
        cls,
        *,
        receipt_id: bytes,
        statement: GoldilocksAirStatementBindingReferenceV3,
        precommitment: GoldilocksAirTracePrecommitmentReferenceV3,
        validator_nonce_commitment: bytes,
        nonce_context_digest: bytes,
        _factory_token: object | None = None,
    ) -> "GoldilocksAirTraceReceiptReferenceV3":
        if _factory_token is not _RECEIPT_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "Goldilocks AIR receipts must be sealed by the validator factory"
            )
        result = object.__new__(cls)
        object.__setattr__(result, "receipt_id", receipt_id)
        object.__setattr__(result, "statement", statement)
        object.__setattr__(result, "precommitment", precommitment)
        object.__setattr__(
            result,
            "validator_nonce_commitment",
            validator_nonce_commitment,
        )
        object.__setattr__(result, "nonce_context_digest", nonce_context_digest)
        object.__setattr__(result, "_factory_token", _RECEIPT_FACTORY_TOKEN)
        result.require_factory_provenance()
        return result

    def require_factory_provenance(self) -> None:
        """Validate the immutable validator-owned receipt identity."""

        if self._factory_token is not _RECEIPT_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt lacks validator factory provenance"
            )
        _fixed32(self.receipt_id, "Goldilocks AIR receipt identifier", nonzero=True)
        if not isinstance(self.statement, GoldilocksAirStatementBindingReferenceV3):
            raise ProofV3VerificationError("Goldilocks AIR receipt statement is malformed")
        self.statement.require_factory_provenance()
        if not isinstance(
            self.precommitment,
            GoldilocksAirTracePrecommitmentReferenceV3,
        ):
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt precommitment is malformed"
            )
        if self.precommitment.core.digest() != self.statement.core.digest():
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt precommitment has an unexpected statement"
            )
        _fixed32(
            self.validator_nonce_commitment,
            "Goldilocks AIR receipt nonce commitment",
            nonzero=True,
        )
        _fixed32(
            self.nonce_context_digest,
            "Goldilocks AIR receipt nonce context",
            nonzero=True,
        )

    def canonical_bytes(self) -> bytes:
        return (
            self.receipt_id
            + self.statement.digest()
            + self.precommitment.digest()
            + self.validator_nonce_commitment
            + self.nonce_context_digest
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_RECEIPT_DIGEST_DOMAIN + self.canonical_bytes()).digest()

    def require_matching_opening_ticket(self, *, ticket: PostCommitOpeningTicketV3) -> None:
        """Require the one post-nonce ticket for exactly this frozen receipt."""

        if not isinstance(ticket, PostCommitOpeningTicketV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt opening ticket is malformed"
            )
        statement = self.statement
        if ticket.proof_challenge_id != statement.proof_challenge_id:
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt ticket has an unexpected challenge"
            )
        if ticket.precommit_context_digest != statement.precommit_context_digest:
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt ticket has an unexpected precommit"
            )
        if ticket.execution_profile_digest != statement.execution_profile_digest:
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt ticket has an unexpected profile"
            )
        if ticket.cache_lease_digest != statement.cache_lease_digest:
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt ticket has an unexpected cache lease"
            )
        if ticket.commitment_envelope_digest != statement.commitment_envelope_digest:
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt ticket has an unexpected envelope"
            )
        expected_nonce_commitment = commit_validator_nonce_v3(
            validator_nonce=ticket.validator_nonce,
            nonce_context_digest=self.nonce_context_digest,
        )
        if expected_nonce_commitment != self.validator_nonce_commitment:
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt ticket nonce is not validator committed"
            )


def _derive_receipt_id(
    *,
    statement: GoldilocksAirStatementBindingReferenceV3,
    precommitment: GoldilocksAirTracePrecommitmentReferenceV3,
) -> bytes:
    """Mint a local validator receipt identifier without exposing a test hook."""

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
    raise ProofV3VerificationError("unable to mint a Goldilocks AIR receipt")


def _make_goldilocks_air_trace_receipt_reference_v3(
    *,
    accepted_statement: _GoldilocksAirAcceptedStatementCapabilityReferenceV3,
    precommitment: GoldilocksAirTracePrecommitmentReferenceV3,
) -> GoldilocksAirTraceReceiptReferenceV3:
    """Seal one trace root for a coordinator-minted accepted statement."""

    if not isinstance(
        accepted_statement,
        _GoldilocksAirAcceptedStatementCapabilityReferenceV3,
    ):
        raise ProofV3VerificationError(
            "Goldilocks AIR receipt requires coordinator acceptance"
        )
    accepted_statement.require_factory_provenance()
    statement = accepted_statement.statement
    if not isinstance(precommitment, GoldilocksAirTracePrecommitmentReferenceV3):
        raise ProofV3VerificationError(
            "Goldilocks AIR receipt precommitment is malformed"
        )
    if precommitment.core.digest() != statement.core.digest():
        raise ProofV3VerificationError(
            "Goldilocks AIR receipt precommitment belongs to a different statement"
        )
    return GoldilocksAirTraceReceiptReferenceV3._construct(
        receipt_id=_derive_receipt_id(
            statement=statement,
            precommitment=precommitment,
        ),
        statement=statement,
        precommitment=precommitment,
        validator_nonce_commitment=accepted_statement.validator_nonce_commitment,
        nonce_context_digest=accepted_statement.nonce_context_digest,
        _factory_token=_RECEIPT_FACTORY_TOKEN,
    )


@dataclass(frozen=True, slots=True, init=False)
class _GoldilocksAirPostRevealCapabilityReferenceV3:
    """Opaque evidence of the coordinator's one hard nonce reveal.

    A raw :class:`NonceRevealV3` or opening ticket is deliberately insufficient
    for the receipt store.  The coordinator alone mints this capability after
    its hidden challenge session has performed the one hard reveal.
    """

    receipt: GoldilocksAirTraceReceiptReferenceV3
    ticket: PostCommitOpeningTicketV3
    _factory_token: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ProofV3VerificationError(
            "Goldilocks AIR post-reveal capabilities are issued by the coordinator"
        )

    @classmethod
    def _construct(
        cls,
        *,
        receipt: GoldilocksAirTraceReceiptReferenceV3,
        reveal: NonceRevealV3,
        _factory_token: object | None = None,
    ) -> "_GoldilocksAirPostRevealCapabilityReferenceV3":
        if _factory_token is not _POST_REVEAL_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "Goldilocks AIR post-reveal capabilities are issued by the coordinator"
            )
        if not isinstance(reveal, NonceRevealV3):
            raise ProofV3VerificationError("Goldilocks AIR nonce reveal is malformed")
        ticket = reveal.opening_ticket()
        if not isinstance(receipt, GoldilocksAirTraceReceiptReferenceV3):
            raise ProofV3VerificationError("Goldilocks AIR receipt is malformed")
        receipt.require_factory_provenance()
        receipt.require_matching_opening_ticket(ticket=ticket)
        result = object.__new__(cls)
        object.__setattr__(result, "receipt", receipt)
        object.__setattr__(result, "ticket", ticket)
        object.__setattr__(result, "_factory_token", _POST_REVEAL_FACTORY_TOKEN)
        result.require_factory_provenance()
        return result

    def require_factory_provenance(self) -> None:
        if self._factory_token is not _POST_REVEAL_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "Goldilocks AIR post-reveal capability lacks coordinator provenance"
            )
        if not isinstance(self.receipt, GoldilocksAirTraceReceiptReferenceV3):
            raise ProofV3VerificationError("Goldilocks AIR post-reveal receipt is malformed")
        self.receipt.require_factory_provenance()
        if not isinstance(self.ticket, PostCommitOpeningTicketV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR post-reveal ticket is malformed"
            )
        self.receipt.require_matching_opening_ticket(ticket=self.ticket)


def _receipt_logical_key(
    *,
    statement: GoldilocksAirStatementBindingReferenceV3,
    precommitment: GoldilocksAirTracePrecommitmentReferenceV3,
) -> bytes:
    """Bind one statement/root pair independently of its random receipt id."""

    return hashlib.sha256(
        _RECEIPT_LOGICAL_KEY_DOMAIN + statement.digest() + precommitment.digest()
    ).digest()


class GoldilocksAirTraceReceiptStoreReferenceV3:
    """One-use in-memory store for validator-retained AIR receipts.

    This is an internal helper for the reference coordinator, not a standalone
    validator API.  It accepts only coordinator-minted post-accept and
    post-reveal capabilities.  It intentionally mirrors the single-worker
    state assumption of :class:`ProofV3ChallengeSession`; a future multi-worker
    native integration needs an equivalent shared compare-and-swap receipt
    store with bounded tombstone retention.
    """

    __slots__ = ("_consumed", "_lock", "_logical_keys", "_receipts")

    def __init__(self) -> None:
        self._receipts: dict[bytes, GoldilocksAirTraceReceiptReferenceV3] = {}
        self._consumed: set[bytes] = set()
        self._logical_keys: set[bytes] = set()
        self._lock = threading.Lock()

    def seal(
        self,
        *,
        accepted_statement: _GoldilocksAirAcceptedStatementCapabilityReferenceV3,
        precommitment: GoldilocksAirTracePrecommitmentReferenceV3,
    ) -> GoldilocksAirTraceReceiptReferenceV3:
        """Atomically retain one expected root before nonce disclosure."""

        receipt = _make_goldilocks_air_trace_receipt_reference_v3(
            accepted_statement=accepted_statement,
            precommitment=precommitment,
        )
        logical_key = _receipt_logical_key(
            statement=receipt.statement,
            precommitment=receipt.precommitment,
        )
        with self._lock:
            if receipt.receipt_id in self._receipts:
                raise ProofV3VerificationError(
                    "Goldilocks AIR receipt identifier already exists"
                )
            if logical_key in self._logical_keys:
                raise ProofV3VerificationError(
                    "Goldilocks AIR trace precommitment is already sealed"
                )
            self._receipts[receipt.receipt_id] = receipt
            # Keep this tombstone after consume/discard.  Randomized receipt
            # identifiers are not a replay boundary for the same statement/root.
            self._logical_keys.add(logical_key)
        return receipt

    def verify_once(
        self,
        *,
        capability: _GoldilocksAirPostRevealCapabilityReferenceV3,
        proof: object,
    ) -> None:
        """Consume a stored receipt before one fail-closed AIR verification."""

        if not isinstance(capability, _GoldilocksAirPostRevealCapabilityReferenceV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt requires a coordinator post-reveal capability"
            )
        with self._lock:
            try:
                capability.require_factory_provenance()
                receipt = capability.receipt
                ticket = capability.ticket
                receipt.require_factory_provenance()
                if receipt.receipt_id in self._consumed:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR receipt has already been consumed"
                    )
                stored = self._receipts.get(receipt.receipt_id)
                if stored is not receipt:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR receipt was not retained by this validator"
                    )
                # Any malformed proof is terminal for this receipt just like a
                # stale opening: allowing a retry would make the one-use nonce
                # boundary depend on an error classification.
                self._consumed.add(receipt.receipt_id)
                del self._receipts[receipt.receipt_id]
                receipt.require_matching_opening_ticket(ticket=ticket)
            except ProofV3VerificationError:
                raise
            except (AttributeError, TypeError, ValueError) as exc:
                raise ProofV3VerificationError(
                    "Goldilocks AIR receipt opening is malformed"
                ) from exc
        try:
            verify_goldilocks_air_reference_v3(
                proof,
                core=receipt.statement.core,
                precommitment=receipt.precommitment,
                validator_nonce=ticket.validator_nonce,
            )
        except ProofV3VerificationError:
            raise
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise ProofV3VerificationError(
                "Goldilocks AIR receipt proof is malformed"
            ) from exc

    def discard(self, *, receipt: GoldilocksAirTraceReceiptReferenceV3) -> None:
        """Release an unopened receipt after a light draw or terminal failure."""

        if not isinstance(receipt, GoldilocksAirTraceReceiptReferenceV3):
            raise ProofV3VerificationError("Goldilocks AIR receipt is malformed")
        with self._lock:
            stored = self._receipts.get(receipt.receipt_id)
            if stored is receipt:
                del self._receipts[receipt.receipt_id]
                self._consumed.add(receipt.receipt_id)


class GoldilocksAirReferenceCoordinatorStateV3(str, Enum):
    """Reference-only lifecycle that never exposes its underlying session."""

    AWAITING_PRECOMMIT = "awaiting_precommit"
    PRECOMMIT_ACCEPTED = "precommit_accepted"
    NONCE_REVEALED = "nonce_revealed"
    LIGHT_REVEALED = "light_revealed"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    FAILED = "failed"


@dataclass(slots=True, init=False)
class GoldilocksAirReferencePrecommitmentCoordinatorV3:
    """Own one session so a receipt is sealed before it can reveal a nonce.

    The production :class:`ProofV3ChallengeSession` intentionally has no
    reference-AIR attachment hook.  Calling an independent receipt store next
    to a caller-owned session would leave a race between envelope acceptance
    and nonce reveal.  This conformance coordinator owns the session and
    exposes only the safe ordering.  It remains unregistered and does not
    replace the production session or its final folded-proof verifier.
    """

    _qualified_profile: QualifiedExecutionProfileV3
    _proof_arrival_budget_ns: int
    _accepted_statement: _GoldilocksAirAcceptedStatementCapabilityReferenceV3 | None
    _receipt: GoldilocksAirTraceReceiptReferenceV3 | None
    _receipt_store: GoldilocksAirTraceReceiptStoreReferenceV3
    _audit_decision: PostCommitAuditDecisionV3 | None
    _post_reveal_capability: _GoldilocksAirPostRevealCapabilityReferenceV3 | None
    _reveal: NonceRevealV3 | None
    _reveal_monotonic_ns: int | None
    _session: ProofV3ChallengeSession
    _state: GoldilocksAirReferenceCoordinatorStateV3
    _deadline_monotonic_ns: int | None
    _lock: threading.Lock

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ProofV3VerificationError(
            "Goldilocks AIR reference coordinators must be issued by the validator"
        )

    @classmethod
    def _construct(
        cls,
        *,
        qualified_profile: QualifiedExecutionProfileV3,
        proof_arrival_budget_ns: int,
        session: ProofV3ChallengeSession,
        _factory_token: object | None = None,
    ) -> "GoldilocksAirReferencePrecommitmentCoordinatorV3":
        if _factory_token is not _COORDINATOR_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "Goldilocks AIR reference coordinators must be issued by the validator"
            )
        if not isinstance(qualified_profile, QualifiedExecutionProfileV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR reference coordinator has an unexpected profile"
            )
        qualified_profile.require_qualification_provenance()
        if not isinstance(session, ProofV3ChallengeSession):
            raise ProofV3VerificationError(
                "Goldilocks AIR reference coordinator has an unexpected session"
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
            GoldilocksAirTraceReceiptStoreReferenceV3(),
        )
        object.__setattr__(result, "_audit_decision", None)
        object.__setattr__(result, "_post_reveal_capability", None)
        object.__setattr__(result, "_reveal", None)
        object.__setattr__(result, "_reveal_monotonic_ns", None)
        object.__setattr__(result, "_session", session)
        object.__setattr__(
            result,
            "_state",
            GoldilocksAirReferenceCoordinatorStateV3.AWAITING_PRECOMMIT,
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
        "GoldilocksAirReferencePrecommitmentCoordinatorV3",
        ValidatorExecutionRequestContextV3,
    ]:
        """Issue a validator-owned V3 session hidden behind the receipt gate."""

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
    def state(self) -> GoldilocksAirReferenceCoordinatorStateV3:
        """Return the outer reference lifecycle without exposing the session."""

        with self._lock:
            return self._state

    def _fail_locked(self) -> None:
        if self._receipt is not None:
            self._receipt_store.discard(receipt=self._receipt)
        self._session.fail_closed()
        self._state = GoldilocksAirReferenceCoordinatorStateV3.FAILED
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
        precommitment: GoldilocksAirTracePrecommitmentReferenceV3,
        observed_output: ObservedExecutionOutputV3,
        last_visible_token_monotonic_ns: int,
        received_monotonic_ns: int,
    ) -> None:
        """Accept envelope and trace root atomically before nonce disclosure."""

        with self._lock:
            if self._state != GoldilocksAirReferenceCoordinatorStateV3.AWAITING_PRECOMMIT:
                raise ProofV3VerificationError(
                    "Goldilocks AIR coordinator does not accept another precommit"
                )
            try:
                last_visible = _monotonic_ns(
                    last_visible_token_monotonic_ns,
                    "last_visible_token_monotonic_ns",
                )
                deadline = last_visible + self._proof_arrival_budget_ns
                if deadline >= 1 << 63:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR coordinator arrival deadline overflows"
                    )
                self._session.accept_precommit_bytes(
                    encoded_envelope=encoded_envelope,
                    observed_output=observed_output,
                    last_visible_token_monotonic_ns=last_visible,
                    received_monotonic_ns=_monotonic_ns(
                        received_monotonic_ns,
                        "received_monotonic_ns",
                    ),
                )
                envelope = commitment_envelope_from_bytes(encoded_envelope)
                profile = self._qualified_profile.profile
                binding = validate_execution_envelope_against_precommit_v3(
                    profile=profile,
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
                accepted_statement = _accept_validator_statement_reference_v3(
                    qualified_profile=self._qualified_profile,
                    envelope=envelope,
                    envelope_binding=binding,
                )
                self._receipt = self._receipt_store.seal(
                    accepted_statement=accepted_statement,
                    precommitment=precommitment,
                )
                self._accepted_statement = accepted_statement
            except ProofV3VerificationError:
                self._fail_locked()
                raise
            except Exception as exc:
                self._fail_locked()
                raise ProofV3VerificationError(
                    "Goldilocks AIR coordinator precommitment is malformed"
                ) from exc
            self._deadline_monotonic_ns = deadline
            self._state = GoldilocksAirReferenceCoordinatorStateV3.PRECOMMIT_ACCEPTED

    def _select_audit_tier_locked(
        self,
        *,
        selected_monotonic_ns: int,
    ) -> PostCommitAuditDecisionV3:
        if self._state != GoldilocksAirReferenceCoordinatorStateV3.PRECOMMIT_ACCEPTED:
            raise ProofV3VerificationError(
                "Goldilocks AIR coordinator has no sealed trace receipt"
            )
        if self._audit_decision is not None:
            raise ProofV3VerificationError(
                "Goldilocks AIR coordinator already selected its audit tier"
            )
        if self._receipt is None:
            self._fail_locked()
            raise ProofV3VerificationError(
                "Goldilocks AIR coordinator lost its sealed trace receipt"
            )
        try:
            decision = self._session.select_audit_tier_once(
                selected_monotonic_ns=_monotonic_ns(
                    selected_monotonic_ns,
                    "selected_monotonic_ns",
                )
            )
        except ProofV3VerificationError:
            self._fail_locked()
            raise
        except Exception as exc:
            self._fail_locked()
            raise ProofV3VerificationError(
                "Goldilocks AIR coordinator audit-tier selection is malformed"
            ) from exc
        if decision.hard_audit_selected:
            self._audit_decision = decision
            return decision
        self._receipt_store.discard(receipt=self._receipt)
        self._accepted_statement = None
        self._receipt = None
        self._post_reveal_capability = None
        self._state = GoldilocksAirReferenceCoordinatorStateV3.LIGHT_REVEALED
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
        """Reveal the session nonce only for a locally selected hard tier."""

        with self._lock:
            if self._state != GoldilocksAirReferenceCoordinatorStateV3.PRECOMMIT_ACCEPTED:
                if self._state == GoldilocksAirReferenceCoordinatorStateV3.LIGHT_REVEALED:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR light selection does not reveal a nonce"
                    )
                raise ProofV3VerificationError(
                    "Goldilocks AIR coordinator has no sealed trace receipt"
                )
            if self._audit_decision is None:
                decision = self._select_audit_tier_locked(
                    selected_monotonic_ns=revealed_monotonic_ns
                )
                if not decision.hard_audit_selected:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR light selection does not reveal a nonce"
                    )
            if self._receipt is None:
                self._fail_locked()
                raise ProofV3VerificationError(
                    "Goldilocks AIR coordinator lost its sealed trace receipt"
                )
            try:
                revealed = _monotonic_ns(revealed_monotonic_ns, "revealed_monotonic_ns")
                reveal = self._session.reveal_nonce_once(
                    revealed_monotonic_ns=revealed
                )
                capability = (
                    _GoldilocksAirPostRevealCapabilityReferenceV3._construct(
                        receipt=self._receipt,
                        reveal=reveal,
                        _factory_token=_POST_REVEAL_FACTORY_TOKEN,
                    )
                )
            except ProofV3VerificationError:
                self._fail_locked()
                raise
            except Exception as exc:
                self._fail_locked()
                raise ProofV3VerificationError(
                    "Goldilocks AIR coordinator nonce reveal is malformed"
                ) from exc
            self._reveal = reveal
            self._reveal_monotonic_ns = revealed
            self._post_reveal_capability = capability
            self._state = GoldilocksAirReferenceCoordinatorStateV3.NONCE_REVEALED
            return reveal

    def verify_proof_once(
        self,
        *,
        proof: object,
        received_monotonic_ns: int,
    ) -> None:
        """Verify one post-nonce reference proof and then discard session secrets."""

        with self._lock:
            if self._state != GoldilocksAirReferenceCoordinatorStateV3.NONCE_REVEALED:
                if self._state == GoldilocksAirReferenceCoordinatorStateV3.LIGHT_REVEALED:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR light selection does not accept a hard proof"
                    )
                raise ProofV3VerificationError(
                    "Goldilocks AIR coordinator does not accept another proof"
                )
            try:
                received = _monotonic_ns(received_monotonic_ns, "received_monotonic_ns")
                deadline = self._deadline_monotonic_ns
                if deadline is None or received > deadline:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR proof arrived after its deadline"
                    )
                if (
                    self._reveal_monotonic_ns is None
                    or received < self._reveal_monotonic_ns
                ):
                    raise ProofV3VerificationError(
                        "Goldilocks AIR proof arrived before its nonce reveal"
                    )
                receipt = self._receipt
                reveal = self._reveal
                capability = self._post_reveal_capability
                if receipt is None or reveal is None or capability is None:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR coordinator lost its proof state"
                    )
                self._state = GoldilocksAirReferenceCoordinatorStateV3.VERIFYING
            except ProofV3VerificationError:
                self._fail_locked()
                raise
        try:
            self._receipt_store.verify_once(
                capability=capability,
                proof=proof,
            )
        except Exception:
            with self._lock:
                if self._state == GoldilocksAirReferenceCoordinatorStateV3.VERIFYING:
                    self._fail_locked()
            raise
        with self._lock:
            if self._state != GoldilocksAirReferenceCoordinatorStateV3.VERIFYING:
                raise ProofV3VerificationError(
                    "Goldilocks AIR coordinator terminated during verification"
                )
            # The hidden session is not a reference-proof verifier. Scrub its
            # nonce/binding state after the outer coordinator records success;
            # its terminal state is deliberately never exposed by this class.
            self._session.fail_closed()
            self._state = GoldilocksAirReferenceCoordinatorStateV3.VERIFIED
            self._accepted_statement = None
            self._receipt = None
            self._audit_decision = None
            self._post_reveal_capability = None
            self._reveal = None
            self._reveal_monotonic_ns = None


__all__ = [
    "GOLDILOCKS_AIR_RECEIPT_REFERENCE_ABI_V3",
    "GOLDILOCKS_AIR_RECEIPT_REFERENCE_FORMAT_VERSION_V3",
    "GoldilocksAirReferenceCoordinatorStateV3",
    "GoldilocksAirReferencePrecommitmentCoordinatorV3",
    "GoldilocksAirStatementBindingReferenceV3",
    "GoldilocksAirTraceReceiptReferenceV3",
    "derive_goldilocks_air_statement_binding_reference_v3",
]
