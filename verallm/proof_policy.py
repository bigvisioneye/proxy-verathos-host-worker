"""Version-aware proof acceptance policy shared by validators and proxies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import struct
from typing import Callable, Optional

from verallm.api.proof_protocol import (
    LEGACY_PROOF_PROTOCOL_VERSION,
    PROOF_PROTOCOL_V2,
)
from verallm.types import InferenceProofBundle, VerificationResult

CURRENT_PROOF_PROTOCOL_VERSION = PROOF_PROTOCOL_V2
_PROOF_PAYLOAD_INVALID_ATTR = "_proof_payload_invalid"
_PARSED_V2_COMMITMENT_ATTR = "_parsed_proof_v2_commitment"
_PARSED_V2_PAYLOAD_ATTR = "_parsed_proof_v2_payload"
_PROOF_DATA_EXCEPTIONS = (
    AttributeError,
    IndexError,
    KeyError,
    OverflowError,
    TypeError,
    ValueError,
    struct.error,
)


class ProofAcceptance(str, Enum):
    """Stable policy outcomes for scoring, receipts, and probation."""

    VERIFIED = "verified"
    LEGACY_COMPATIBILITY = "legacy_compatibility"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ProofPolicyDecision:
    """Result of applying rollout policy to one parsed proof."""

    acceptance: ProofAcceptance
    protocol_version: Optional[int]
    accepted: bool
    proof_verified: bool
    legacy_compatibility_accepted: bool
    probation_required: bool
    reason: str


def evaluate_proof_policy(
    *,
    protocol_version: Optional[int],
    verification_passed: bool,
    legacy_v1_compatibility_active: bool,
) -> ProofPolicyDecision:
    """Apply the v1/v2 protocol rollout matrix.

    Bounded compatibility permits a valid legacy proof to retain normal receipt
    and scoring treatment. Operational maintenance is applied separately by
    validators and proxies and never changes this version decision.
    """

    if type(protocol_version) is not int:
        protocol_version = None
    verification_passed = verification_passed is True

    if protocol_version == CURRENT_PROOF_PROTOCOL_VERSION:
        if verification_passed:
            return ProofPolicyDecision(
                acceptance=ProofAcceptance.VERIFIED,
                protocol_version=protocol_version,
                accepted=True,
                proof_verified=True,
                legacy_compatibility_accepted=False,
                probation_required=False,
                reason="v2 proof verified",
            )
        return ProofPolicyDecision(
            acceptance=ProofAcceptance.REJECTED,
            protocol_version=protocol_version,
            accepted=False,
            proof_verified=False,
            legacy_compatibility_accepted=False,
            probation_required=True,
            reason="v2 proof verification failed",
        )

    if protocol_version == LEGACY_PROOF_PROTOCOL_VERSION:
        if not verification_passed:
            return ProofPolicyDecision(
                acceptance=ProofAcceptance.REJECTED,
                protocol_version=protocol_version,
                accepted=False,
                proof_verified=False,
                legacy_compatibility_accepted=False,
                probation_required=True,
                reason="legacy proof verification failed",
            )
        if legacy_v1_compatibility_active:
            return ProofPolicyDecision(
                acceptance=ProofAcceptance.LEGACY_COMPATIBILITY,
                protocol_version=protocol_version,
                accepted=True,
                proof_verified=True,
                legacy_compatibility_accepted=True,
                probation_required=False,
                reason="legacy proof accepted during bounded compatibility",
            )
        return ProofPolicyDecision(
            acceptance=ProofAcceptance.REJECTED,
            protocol_version=protocol_version,
            accepted=False,
            proof_verified=False,
            legacy_compatibility_accepted=False,
            probation_required=True,
            reason="legacy proof protocol is no longer accepted",
        )

    return ProofPolicyDecision(
        acceptance=ProofAcceptance.REJECTED,
        protocol_version=protocol_version,
        accepted=False,
        proof_verified=False,
        legacy_compatibility_accepted=False,
        probation_required=True,
        reason="missing or unsupported proof protocol version",
    )


def mark_proof_payload_invalid(
    proof_bundle: InferenceProofBundle,
    *,
    protocol_version: Optional[int],
) -> InferenceProofBundle:
    """Mark a returned bundle whose wire payload could not be decoded."""
    if type(protocol_version) is not int:
        protocol_version = None
    proof_bundle.proof_protocol_version = protocol_version
    setattr(proof_bundle, _PROOF_PAYLOAD_INVALID_ATTR, True)
    return proof_bundle


def proof_protocol_version_from_bundle(
    proof_bundle: object,
) -> Optional[int]:
    """Return only a canonical integer protocol version from a proof bundle."""
    version = getattr(proof_bundle, "proof_protocol_version", None)
    return version if type(version) is int else None


def _parse_canonical_v2_payloads(proof_bundle: object) -> bool:
    """Parse and cache both canonical v2 wire envelopes."""
    from verallm.proof_v2.payload import (
        ProofV2CommitmentEnvelope,
        ProofV2Payload,
    )

    try:
        commitment = ProofV2CommitmentEnvelope.from_canonical_bytes(
            getattr(getattr(proof_bundle, "commitment"), "proof_v2_commitment")
        )
        payload = ProofV2Payload.from_canonical_bytes(
            getattr(proof_bundle, "proof_v2_payload")
        )
    except (AttributeError, TypeError, ValueError):
        return False
    if commitment.manifest_digest != payload.manifest_digest:
        return False
    setattr(proof_bundle, _PARSED_V2_COMMITMENT_ATTR, commitment)
    setattr(proof_bundle, _PARSED_V2_PAYLOAD_ATTR, payload)
    return True


def verify_with_proof_policy(
    proof_bundle: InferenceProofBundle,
    verifier: Callable[[], tuple[VerificationResult, dict]],
    *,
    legacy_v1_compatibility_active: bool,
    expected_protocol_version: Optional[int] = None,
) -> tuple[VerificationResult, dict, ProofPolicyDecision]:
    """Run local verification and apply the rollout policy to its result.

    Wire decoding and value-shape errors are returned as failed proof results.
    Exceptions outside that input-error set continue to the caller so validator
    runtime problems retain their existing handling.
    """
    protocol_version = proof_protocol_version_from_bundle(proof_bundle)
    if (
        type(expected_protocol_version) is int
        and protocol_version != expected_protocol_version
    ):
        decision = ProofPolicyDecision(
            acceptance=ProofAcceptance.REJECTED,
            protocol_version=protocol_version,
            accepted=False,
            proof_verified=False,
            legacy_compatibility_accepted=False,
            probation_required=True,
            reason="proof protocol version differs from the requested version",
        )
        result = VerificationResult.failure(decision.reason)
        result.details = {
            "proof_policy": {
                "acceptance": decision.acceptance.value,
                "protocol_version": decision.protocol_version,
                "legacy_compatibility_accepted": False,
                "probation_required": True,
                "reason": decision.reason,
            }
        }
        return result, {}, decision
    payload_invalid = bool(
        getattr(proof_bundle, _PROOF_PAYLOAD_INVALID_ATTR, False)
    )
    if (
        not payload_invalid
        and protocol_version == CURRENT_PROOF_PROTOCOL_VERSION
        and not _parse_canonical_v2_payloads(proof_bundle)
    ):
        payload_invalid = True

    forced_decision: Optional[ProofPolicyDecision] = None
    if protocol_version not in (
        LEGACY_PROOF_PROTOCOL_VERSION,
        CURRENT_PROOF_PROTOCOL_VERSION,
    ):
        raw_result = VerificationResult.failure(
            "missing or unsupported proof protocol version"
        )
        timing: dict = {}
        forced_decision = evaluate_proof_policy(
            protocol_version=protocol_version,
            verification_passed=False,
            legacy_v1_compatibility_active=legacy_v1_compatibility_active,
        )
    elif (
        protocol_version == LEGACY_PROOF_PROTOCOL_VERSION
        and not legacy_v1_compatibility_active
    ):
        # Once compatibility ends, v1 is rejected by version alone. Do not enter the
        # legacy verifier: malformed legacy data must not turn this mandatory
        # rejection into a validator-side exception or a "not tested" receipt.
        raw_result = VerificationResult.failure(
            "legacy proof protocol is no longer accepted"
        )
        timing = {}
        forced_decision = evaluate_proof_policy(
            protocol_version=protocol_version,
            verification_passed=True,
            legacy_v1_compatibility_active=False,
        )
    elif payload_invalid:
        raw_result = VerificationResult.failure("proof payload verification failed")
        timing = {}
    else:
        try:
            raw_result, timing = verifier()
            if not isinstance(raw_result, VerificationResult) or not isinstance(timing, dict):
                raise TypeError("proof verifier returned an invalid result")
        except _PROOF_DATA_EXCEPTIONS:
            payload_invalid = True
            raw_result = VerificationResult.failure("proof payload verification failed")
            timing = {}

    decision = forced_decision or evaluate_proof_policy(
        protocol_version=protocol_version,
        verification_passed=raw_result.passed,
        legacy_v1_compatibility_active=legacy_v1_compatibility_active,
    )

    details = dict(raw_result.details or {})
    details["proof_policy"] = {
        "acceptance": decision.acceptance.value,
        "protocol_version": decision.protocol_version,
        "legacy_compatibility_accepted": decision.legacy_compatibility_accepted,
        "probation_required": decision.probation_required,
    }
    if payload_invalid:
        details["proof_payload_invalid"] = True

    message = raw_result.message
    if forced_decision is not None or (raw_result.passed and not decision.accepted):
        message = decision.reason
    result = VerificationResult(
        passed=decision.accepted,
        message=message,
        details=details,
    )
    return result, timing, decision
