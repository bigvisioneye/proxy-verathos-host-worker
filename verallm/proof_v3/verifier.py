"""Fail-closed verifier dispatch for proof-v3 global execution adapters.

There is intentionally no default adapter.  A syntactically valid opaque
payload is never a verified execution proof.  Integration must provide a
qualified, signed profile and an adapter which verifies all three global
relations: folded linear operations, transition/prefill provenance, and the
final-hidden-to-token path.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Protocol

from verallm.proof_v3.challenge import (
    FoldedExecutionChallengeV3,
    PostCommitAuditDecisionV3,
    derive_folded_execution_challenge_v3,
    derive_postcommit_audit_decision_v3,
)
from verallm.proof_v3.catalog import QualifiedExecutionArtifactsV3
from verallm.proof_v3.errors import ProofV3UnavailableError, ProofV3VerificationError
from verallm.proof_v3.document import (
    SignedExecutionProfileDocumentV3,
    verify_signed_execution_profile_v3,
)
from verallm.proof_v3.payload import FoldedExecutionProofV3, ProofV3CommitmentEnvelope
from verallm.proof_v3.profile import ExecutionSecurityProfileV3
from verallm.proof_v3.relation import RuntimeHardAuditPolicyV3
from verallm.proof_v3.request import (
    ExecutionRequestBindingV3,
    ExecutionOutputBindingV3,
    ObservedExecutionOutputV3,
    PreExecutionRequestContextV3,
    ValidatorExecutionRequestContextV3,
    commit_validator_nonce_v3,
)


class FoldedExecutionAdapterV3(Protocol):
    """A qualified native verifier for one signed execution-profile adapter."""

    def validate_profile(
        self,
        *,
        profile: ExecutionSecurityProfileV3,
        artifacts: QualifiedExecutionArtifactsV3,
    ) -> None:
        """Reject unsupported graph/cache/static ABI before proof dispatch.

        Implementations receive only the exact, validator-owned artifact view
        that has already cross-checked static operations, bindings, and loaded
        capture/circuit/tokenizer/sampler/quantization artifacts.
        """

    def verify(
        self,
        *,
        profile: ExecutionSecurityProfileV3,
        envelope: ProofV3CommitmentEnvelope,
        proof: FoldedExecutionProofV3,
        challenge: FoldedExecutionChallengeV3,
    ) -> None:
        """Raise :class:`ProofV3VerificationError` for any invalid relation.

        A dynamic global adapter must enumerate the verifier-derived chunk
        universe and derive a distinct backend-field coefficient from
        ``challenge.global_relation_seed`` plus every constraint-system-defined
        atomic relation coordinate under its signed backend-specific ABI. It
        must not reduce the Pallas-only
        ``derive_pallas_affine_fold_scalar_v3`` value into its dynamic field,
        nor use the legacy Pallas family-wide alphas as its sole global fold.
        """


AdapterKeyV3 = bytes
MAX_VERIFIER_KEY_BYTES = 64 << 20


@dataclass(frozen=True, slots=True)
class EnvelopeBindingV3:
    """Validator-derived public statement accepted before nonce reveal."""

    profile_digest: bytes
    precommit_context: PreExecutionRequestContextV3
    request_binding: ExecutionRequestBindingV3


@dataclass(frozen=True, slots=True)
class QualifiedExecutionAdapterV3:
    """One native verifier registered for one exact authenticated profile."""

    artifacts: QualifiedExecutionArtifactsV3
    adapter: FoldedExecutionAdapterV3

    def __post_init__(self) -> None:
        if not isinstance(self.artifacts, QualifiedExecutionArtifactsV3):
            raise ProofV3VerificationError("qualified adapter artifacts are malformed")
        if self.adapter is None:
            raise ProofV3VerificationError(
                "qualified adapter implementation is missing"
            )

    def validate_profile(self, *, profile: ExecutionSecurityProfileV3) -> None:
        self.artifacts.validate_profile(profile=profile)
        self.adapter.validate_profile(profile=profile, artifacts=self.artifacts)


def adapter_key_v3(profile: ExecutionSecurityProfileV3) -> AdapterKeyV3:
    """Return the exact signed-profile key required for adapter dispatch.

    A broad architecture/ABI key is unsafe: two signed profiles may differ in
    graph, capture, circuit, quantization, or static bindings. Runtime code
    must register a native adapter only for the exact catalog-selected digest.
    """

    if not isinstance(profile, ExecutionSecurityProfileV3):
        raise ProofV3VerificationError("profile has an unexpected type")
    return profile.digest()


def validate_execution_envelope_binding_v3(
    *,
    profile: ExecutionSecurityProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    validator_nonce: bytes,
    expected_static_manifest_digest: bytes,
    expected_execution_profile_digest: bytes,
    validator_request_context: ValidatorExecutionRequestContextV3,
    observed_output: ObservedExecutionOutputV3,
    capability_requirement=None,
) -> EnvelopeBindingV3:
    """Validate one sealed public statement before revealing the nonce.

    This intentionally consumes no proof bytes and derives no Fiat--Shamir
    challenge. A stateful validator session calls it exactly once while the
    nonce remains private, stores the envelope digest, and only then reveals
    the nonce for post-commitment openings.

    ``capability_requirement`` defaults to the strong
    ``require_hard_execution_capability`` gate; the economic proof system
    passes its own exact capability check so the SAME admission flow is
    shared instead of duplicated (never weakened for global-folded).
    """

    if not isinstance(validator_request_context, ValidatorExecutionRequestContextV3):
        raise ProofV3VerificationError(
            "validator_request_context has an unexpected type"
        )
    if not isinstance(observed_output, ObservedExecutionOutputV3):
        raise ProofV3VerificationError("observed_output has an unexpected type")
    if not isinstance(profile, ExecutionSecurityProfileV3):
        raise ProofV3VerificationError("profile has an unexpected type")
    try:
        validator_context_count = validator_request_context.prompt_token_count()
        observed_decode_count = observed_output.output_token_count()
    except Exception as exc:
        raise ProofV3VerificationError(
            "validator request/output context is malformed"
        ) from exc
    if validator_context_count > profile.max_verified_context_tokens:
        raise ProofV3VerificationError("context length exceeds signed profile limit")
    if observed_decode_count > profile.max_verified_decode_tokens:
        raise ProofV3VerificationError("decode length exceeds signed profile limit")
    try:
        profile_digest = profile.digest()
        precommit_context = validator_request_context.derive_precommit_context(
            static_manifest_digest=expected_static_manifest_digest,
            execution_profile_digest=profile_digest,
        )
        output_binding = observed_output.derive_output_binding()
        expected_nonce_commitment = commit_validator_nonce_v3(
            validator_nonce=validator_nonce,
            nonce_context_digest=precommit_context.nonce_context_digest(),
        )
    except Exception as exc:
        raise ProofV3VerificationError(
            "validator request/output context is malformed"
        ) from exc
    if precommit_context.validator_nonce_commitment != expected_nonce_commitment:
        raise ProofV3VerificationError(
            "validator nonce does not match the pre-request commitment"
        )
    return validate_execution_envelope_against_precommit_v3(
        profile=profile,
        envelope=envelope,
        expected_static_manifest_digest=expected_static_manifest_digest,
        expected_execution_profile_digest=expected_execution_profile_digest,
        precommit_context=precommit_context,
        output_binding=output_binding,
        capability_requirement=capability_requirement,
    )


def validate_execution_envelope_against_precommit_v3(
    *,
    profile: ExecutionSecurityProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    expected_static_manifest_digest: bytes,
    expected_execution_profile_digest: bytes,
    precommit_context: PreExecutionRequestContextV3,
    output_binding: ExecutionOutputBindingV3,
    capability_requirement=None,
    allow_unverified_context: bool = False,
) -> EnvelopeBindingV3:
    """Validate a sealed envelope against fixed-size validator-owned roots.

    This helper deliberately accepts no nonce and no raw prompt/output token
    arrays. The stateful nonce session uses it after it has derived and frozen
    the validator request context, so repeated post-token proof checks never
    retain or re-hash a long prompt or output stream.
    """

    if not isinstance(profile, ExecutionSecurityProfileV3):
        raise ProofV3VerificationError("profile has an unexpected type")
    if not isinstance(envelope, ProofV3CommitmentEnvelope):
        raise ProofV3VerificationError("commitment envelope has an unexpected type")
    if not isinstance(precommit_context, PreExecutionRequestContextV3):
        raise ProofV3VerificationError("precommit context has an unexpected type")
    if not isinstance(output_binding, ExecutionOutputBindingV3):
        raise ProofV3VerificationError("output binding has an unexpected type")
    if (
        not isinstance(expected_static_manifest_digest, bytes)
        or len(expected_static_manifest_digest) != 32
    ):
        raise ProofV3VerificationError(
            "expected_static_manifest_digest must be exactly 32 bytes"
        )
    if (
        not isinstance(expected_execution_profile_digest, bytes)
        or len(expected_execution_profile_digest) != 32
    ):
        raise ProofV3VerificationError(
            "expected_execution_profile_digest must be exactly 32 bytes"
        )
    if capability_requirement is None:
        profile.require_hard_execution_capability()
    else:
        capability_requirement(profile)
    context_is_qualified = (
        precommit_context.context_token_count
        <= profile.max_verified_context_tokens
    )
    if not context_is_qualified and not allow_unverified_context:
        raise ProofV3VerificationError("context length exceeds signed profile limit")
    if output_binding.decode_token_count > profile.max_verified_decode_tokens:
        raise ProofV3VerificationError("decode length exceeds signed profile limit")
    if context_is_qualified:
        try:
            profile.relation_spec.validate_execution_counts(
                context_token_count=precommit_context.context_token_count,
                decode_token_count=output_binding.decode_token_count,
            )
        except Exception as exc:
            raise ProofV3VerificationError(
                "request/output token counts do not satisfy the signed sequence domain"
            ) from exc
    profile_digest = profile.digest()
    if profile_digest != expected_execution_profile_digest:
        raise ProofV3VerificationError(
            "execution profile was not selected by the validator catalog"
        )
    if profile.static_manifest_digest != expected_static_manifest_digest:
        raise ProofV3VerificationError(
            "execution profile has an unexpected static manifest"
        )
    if precommit_context.static_manifest_digest != expected_static_manifest_digest:
        raise ProofV3VerificationError(
            "precommit context has an unexpected static manifest"
        )
    if precommit_context.execution_profile_digest != profile_digest:
        raise ProofV3VerificationError("precommit context has an unexpected profile")
    if envelope.static_manifest_digest != expected_static_manifest_digest:
        raise ProofV3VerificationError(
            "commitment envelope has an unexpected static manifest"
        )
    if envelope.execution_profile_digest != profile_digest:
        raise ProofV3VerificationError("commitment envelope has an unexpected profile")
    request_binding = precommit_context.bind_output(output_binding)
    if envelope.precommit_context_digest != precommit_context.digest():
        raise ProofV3VerificationError(
            "commitment envelope has an unexpected validator precommit context"
        )
    if envelope.prompt_token_root != precommit_context.prompt_token_root:
        raise ProofV3VerificationError(
            "commitment envelope has an unexpected prompt root"
        )
    if envelope.sampler_config_digest != precommit_context.sampler_config_digest:
        raise ProofV3VerificationError(
            "commitment envelope has an unexpected sampler configuration"
        )
    if envelope.cache_lease_digest != precommit_context.cache_lease_digest:
        raise ProofV3VerificationError(
            "commitment envelope has an unexpected validator cache lease"
        )
    if envelope.output_token_root != output_binding.output_token_root:
        raise ProofV3VerificationError(
            "commitment envelope has an unexpected token root"
        )
    if envelope.output_stream_digest != output_binding.output_stream_digest:
        raise ProofV3VerificationError("commitment envelope has an unexpected stream")
    if envelope.request_digest != request_binding.request_digest:
        raise ProofV3VerificationError(
            "commitment envelope has an unexpected validator request"
        )
    if envelope.context_token_count != precommit_context.context_token_count:
        raise ProofV3VerificationError(
            "commitment envelope has an unexpected validator context length"
        )
    if envelope.decode_token_count != output_binding.decode_token_count:
        raise ProofV3VerificationError(
            "commitment envelope has an unexpected validator decode length"
        )
    return EnvelopeBindingV3(
        profile_digest=profile_digest,
        precommit_context=precommit_context,
        request_binding=request_binding,
    )


def verify_folded_execution_proof_v3(
    *,
    profile: ExecutionSecurityProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    proof: FoldedExecutionProofV3,
    validator_nonce: bytes,
    expected_static_manifest_digest: bytes,
    expected_execution_profile_digest: bytes,
    validator_request_context: ValidatorExecutionRequestContextV3,
    observed_output: ObservedExecutionOutputV3,
    runtime_policy: RuntimeHardAuditPolicyV3,
    adapters: Mapping[AdapterKeyV3, QualifiedExecutionAdapterV3] | None = None,
) -> FoldedExecutionChallengeV3:
    """Validate public binding, then delegate only to a qualified verifier.

    The validator selects ``expected_execution_profile_digest`` from its
    authenticated model/quantization/cache catalog and derives request/output
    bindings from its exact pre-request context and observed SSE token stream.
    A miner may not choose a weaker signed profile or an arbitrary request
    context that happens to reference the same static weights.
    With the default empty registry this always fails closed.  Production code
    must not install an adapter until its native proof system, signed profile,
    long-context capture implementation, and adversarial qualification suite
    all exist.
    """

    if not isinstance(proof, FoldedExecutionProofV3):
        raise ProofV3VerificationError("folded execution proof has an unexpected type")
    binding = validate_execution_envelope_binding_v3(
        profile=profile,
        envelope=envelope,
        validator_nonce=validator_nonce,
        expected_static_manifest_digest=expected_static_manifest_digest,
        expected_execution_profile_digest=expected_execution_profile_digest,
        validator_request_context=validator_request_context,
        observed_output=observed_output,
    )
    return _verify_folded_execution_proof_against_binding_v3(
        profile=profile,
        envelope=envelope,
        proof=proof,
        validator_nonce=validator_nonce,
        expected_static_manifest_digest=expected_static_manifest_digest,
        expected_execution_profile_digest=expected_execution_profile_digest,
        binding=binding,
        runtime_policy=runtime_policy,
        adapters=adapters,
    )


def _verify_folded_execution_proof_against_binding_v3(
    *,
    profile: ExecutionSecurityProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    proof: FoldedExecutionProofV3,
    validator_nonce: bytes,
    expected_static_manifest_digest: bytes,
    expected_execution_profile_digest: bytes,
    binding: EnvelopeBindingV3,
    runtime_policy: RuntimeHardAuditPolicyV3,
    audit_decision: PostCommitAuditDecisionV3 | None = None,
    adapters: Mapping[AdapterKeyV3, QualifiedExecutionAdapterV3] | None = None,
    registration_prequalified: bool = False,
) -> FoldedExecutionChallengeV3:
    """Verify one proof against a validator-owned prevalidated statement.

    This is deliberately private. :class:`ProofV3ChallengeSession` uses it
    after atomically accepting the exact envelope before nonce reveal. Public
    callers must use one of the context-validating wrappers above instead.
    """

    if not isinstance(proof, FoldedExecutionProofV3):
        raise ProofV3VerificationError("folded execution proof has an unexpected type")
    if not isinstance(binding, EnvelopeBindingV3):
        raise ProofV3VerificationError(
            "execution envelope binding has an unexpected type"
        )
    try:
        profile.relation_spec.audit_policy.validate_runtime(runtime_policy)
    except ProofV3VerificationError:
        raise
    except Exception as exc:
        raise ProofV3VerificationError(
            "runtime hard-audit policy is malformed"
        ) from exc
    validated_binding = validate_execution_envelope_against_precommit_v3(
        profile=profile,
        envelope=envelope,
        expected_static_manifest_digest=expected_static_manifest_digest,
        expected_execution_profile_digest=expected_execution_profile_digest,
        precommit_context=binding.precommit_context,
        output_binding=binding.request_binding.output,
    )
    if validated_binding != binding:
        raise ProofV3VerificationError("execution envelope binding is inconsistent")
    try:
        expected_audit_decision = derive_postcommit_audit_decision_v3(
            validator_nonce=validator_nonce,
            profile=profile,
            envelope=envelope,
            runtime_policy=runtime_policy,
        )
    except ProofV3VerificationError:
        raise
    except Exception as exc:
        raise ProofV3VerificationError(
            "postcommit hard-audit decision is malformed"
        ) from exc
    if audit_decision is not None:
        if not isinstance(audit_decision, PostCommitAuditDecisionV3):
            raise ProofV3VerificationError(
                "postcommit hard-audit decision has an unexpected type"
            )
        if audit_decision != expected_audit_decision:
            raise ProofV3VerificationError(
                "postcommit hard-audit decision is not bound to the validator transcript"
            )
    if not expected_audit_decision.hard_audit_selected:
        raise ProofV3VerificationError(
            "postcommit audit decision did not select a hard proof"
        )
    profile_digest = validated_binding.profile_digest
    if proof.commitment_envelope_digest != envelope.digest():
        raise ProofV3VerificationError(
            "proof is bound to a different commitment envelope"
        )
    if proof.execution_profile_digest != profile_digest:
        raise ProofV3VerificationError(
            "proof is bound to a different execution profile"
        )
    challenge = derive_folded_execution_challenge_v3(
        validator_nonce=validator_nonce,
        profile=profile,
        envelope=envelope,
    )
    registration = (adapters or {}).get(adapter_key_v3(profile))
    if registration is None:
        raise ProofV3UnavailableError(
            "no qualified proof-v3 adapter is registered for this execution profile"
        )
    if not isinstance(registration, QualifiedExecutionAdapterV3):
        raise ProofV3VerificationError(
            "proof-v3 adapter registration is not backed by qualified artifacts"
        )
    # A serving session obtains its registration exclusively through
    # QualifiedExecutionProfileV3, which authenticates the signed profile and
    # performs this potentially expensive static-catalog qualification once at
    # catalog load time. Rebuilding every Pallas tree for every untrusted proof
    # would turn the proof endpoint into a validator CPU DoS. Public/offline
    # helpers retain the defensive validation below.
    if not registration_prequalified:
        try:
            registration.validate_profile(profile=profile)
        except ProofV3VerificationError:
            raise
        except Exception as exc:
            raise ProofV3VerificationError(
                "proof-v3 adapter rejected the signed relation profile"
            ) from exc
    try:
        registration.adapter.verify(
            profile=profile,
            envelope=envelope,
            proof=proof,
            challenge=challenge,
        )
    except ProofV3VerificationError:
        raise
    except Exception as exc:
        raise ProofV3VerificationError(
            "proof-v3 adapter rejected the folded execution proof"
        ) from exc
    return challenge


def verify_signed_folded_execution_proof_v3(
    *,
    document: SignedExecutionProfileDocumentV3,
    envelope: ProofV3CommitmentEnvelope,
    proof: FoldedExecutionProofV3,
    validator_nonce: bytes,
    expected_static_manifest_digest: bytes,
    expected_execution_profile_digest: bytes,
    validator_request_context: ValidatorExecutionRequestContextV3,
    observed_output: ObservedExecutionOutputV3,
    runtime_policy: RuntimeHardAuditPolicyV3,
    expected_authority_signers: Collection[str | bytes],
    authority_threshold: int,
    verifier_key_bytes: bytes,
    adapters: Mapping[AdapterKeyV3, QualifiedExecutionAdapterV3] | None = None,
) -> FoldedExecutionChallengeV3:
    """Offline signed-statement helper, not a serving lifecycle.

    This authenticates the model/quantization/catalog-selected profile digest,
    authority signatures, and verifier-key artifact before any adapter can
    inspect opaque proof bytes. It cannot establish that the validator nonce
    remained secret until an envelope was atomically accepted, so production
    serving must use :class:`ProofV3ChallengeSession` with a
    :class:`QualifiedExecutionProfileV3`. The lower
    ``verify_folded_execution_proof_v3`` helper is likewise a bound-statement
    primitive for tests and adapter composition.
    """

    if not isinstance(document, SignedExecutionProfileDocumentV3):
        raise ProofV3VerificationError(
            "signed execution profile document has an unexpected type"
        )
    if (
        not isinstance(verifier_key_bytes, bytes)
        or not verifier_key_bytes
        or len(verifier_key_bytes) > MAX_VERIFIER_KEY_BYTES
    ):
        raise ProofV3VerificationError("verifier key artifact length is out of range")
    profile = document.profile
    if hashlib.sha256(verifier_key_bytes).digest() != profile.verifier_key_digest:
        raise ProofV3VerificationError(
            "verifier key artifact does not match the execution profile"
        )
    verify_signed_execution_profile_v3(
        profile,
        document.signatures,
        expected_static_manifest_digest=expected_static_manifest_digest,
        expected_authority_signers=expected_authority_signers,
        authority_threshold=authority_threshold,
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
        adapters=adapters,
    )


__all__ = [
    "AdapterKeyV3",
    "EnvelopeBindingV3",
    "FoldedExecutionAdapterV3",
    "MAX_VERIFIER_KEY_BYTES",
    "QualifiedExecutionAdapterV3",
    "adapter_key_v3",
    "validate_execution_envelope_against_precommit_v3",
    "validate_execution_envelope_binding_v3",
    "verify_folded_execution_proof_v3",
    "verify_signed_folded_execution_proof_v3",
]
