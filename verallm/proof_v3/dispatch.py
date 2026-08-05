"""Proof-system dispatch for proof-v3 execution proofs.

The validator catalog selects one exact signed profile digest; the profile's
``proof_system_id`` (bound into that digest) routes verification:

* ``global_folded_execution_v3`` -> the strong native AIR path
  (:func:`verifier.verify_folded_execution_proof_v3`) -- untouched,
  fail-closed until a genuine qualified adapter is registered;
* ``economic_recompute_v3``      -> the economic recompute HARD audit
  (:func:`economic_registry.verify_economic_execution_proof_v3`);
* anything else                   -> rejected.

Each proof system has its own registry type keyed by the exact signed
profile digest, so a proof can never be verified under a different profile
than the validator selected (no strong->economic downgrade and no
cross-profile adapter confusion).
"""

from __future__ import annotations

from collections.abc import Mapping

from verallm.proof_v3.economic_recompute_adapter import (
    ECONOMIC_RECOMPUTE_PROOF_SYSTEM_V3,
)
from verallm.proof_v3.economic_registry import (
    QualifiedEconomicAdapterV3,
    verify_economic_execution_proof_v3,
)
from verallm.proof_v3.economic_wire import EconomicRecomputeProofV3
from verallm.proof_v3.errors import ProofV3VerificationError
from verallm.proof_v3.payload import FoldedExecutionProofV3, ProofV3CommitmentEnvelope
from verallm.proof_v3.profile import (
    GLOBAL_FOLDED_EXECUTION_PROOF_SYSTEM_V3,
    ExecutionSecurityProfileV3,
)
from verallm.proof_v3.relation import RuntimeHardAuditPolicyV3
from verallm.proof_v3.request import (
    ObservedExecutionOutputV3,
    ValidatorExecutionRequestContextV3,
)
from verallm.proof_v3.verifier import (
    AdapterKeyV3,
    QualifiedExecutionAdapterV3,
    verify_folded_execution_proof_v3,
)

__all__ = ["verify_execution_proof_v3"]


def verify_execution_proof_v3(
    *,
    profile: ExecutionSecurityProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    proof,
    validator_nonce: bytes,
    expected_static_manifest_digest: bytes,
    expected_execution_profile_digest: bytes,
    validator_request_context: ValidatorExecutionRequestContextV3,
    observed_output: ObservedExecutionOutputV3,
    runtime_policy: RuntimeHardAuditPolicyV3,
    global_adapters: Mapping[AdapterKeyV3, QualifiedExecutionAdapterV3]
    | None = None,
    economic_adapters: Mapping[AdapterKeyV3, QualifiedEconomicAdapterV3]
    | None = None,
):
    """Route one execution proof to its exact signed proof system.

    Returns the derived challenge of the selected proof system.  Fails
    closed for an unknown proof system, a proof object of the wrong type
    for the selected system, or an unregistered profile digest.
    """

    if not isinstance(profile, ExecutionSecurityProfileV3):
        raise ProofV3VerificationError("profile has an unexpected type")
    proof_system_id = profile.proof_system_id
    if proof_system_id == GLOBAL_FOLDED_EXECUTION_PROOF_SYSTEM_V3:
        if not isinstance(proof, FoldedExecutionProofV3):
            raise ProofV3VerificationError(
                "global-folded proof system requires a folded execution proof"
            )
        return verify_folded_execution_proof_v3(
            profile=profile,
            envelope=envelope,
            proof=proof,
            validator_nonce=validator_nonce,
            expected_static_manifest_digest=expected_static_manifest_digest,
            expected_execution_profile_digest=expected_execution_profile_digest,
            validator_request_context=validator_request_context,
            observed_output=observed_output,
            runtime_policy=runtime_policy,
            adapters=global_adapters,
        )
    if proof_system_id == ECONOMIC_RECOMPUTE_PROOF_SYSTEM_V3:
        if not isinstance(proof, EconomicRecomputeProofV3):
            raise ProofV3VerificationError(
                "economic proof system requires an economic recompute proof"
            )
        return verify_economic_execution_proof_v3(
            profile=profile,
            envelope=envelope,
            proof=proof,
            validator_nonce=validator_nonce,
            expected_static_manifest_digest=expected_static_manifest_digest,
            expected_execution_profile_digest=expected_execution_profile_digest,
            validator_request_context=validator_request_context,
            observed_output=observed_output,
            runtime_policy=runtime_policy,
            adapters=economic_adapters,
        )
    raise ProofV3VerificationError(
        f"proof system {proof_system_id!r} has no qualified verifier"
    )
