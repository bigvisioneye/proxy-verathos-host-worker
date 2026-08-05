"""Proof-v3 rollout policy: dual-stack acceptance behind SubnetConfig
feature flags (same mechanism as ``tee_enabled``).

Phase 1 (``proof_v3_enabled``): validators accept v3 capture-commit +
audit receipts ALONGSIDE live v1 proofs; miners upgrade at their own
pace. Phase 2 (``proof_v3_mandatory``): v1-only miners stop passing
audits; the v3 capture commitment is required for scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# keccak256("proof_v3_enabled")
PROOF_V3_ENABLED_KEY: Final = bytes.fromhex(
    "71ccf5b9958b26b303ad6974d0f3eb8cd71864f80b34daa962aca6d9882df50c")
# keccak256("proof_v3_mandatory")
PROOF_V3_MANDATORY_KEY: Final = bytes.fromhex(
    "a4b931ce6b8cdc2a83fd2eab6cbfccd42bc73186eb9622c4765a540baac4577d")


@dataclass(frozen=True, slots=True)
class ProofV3RolloutFlags:
    enabled: bool
    mandatory: bool


def read_proof_v3_flags(subnet_config_client) -> ProofV3RolloutFlags:
    """Read both flags with one client (mandatory implies enabled)."""

    try:
        enabled = bool(
            subnet_config_client.get_feature_flag(PROOF_V3_ENABLED_KEY))
        mandatory = bool(
            subnet_config_client.get_feature_flag(PROOF_V3_MANDATORY_KEY))
    except Exception:
        # chain unavailable -> fail toward the live v1 behaviour
        return ProofV3RolloutFlags(enabled=False, mandatory=False)
    return ProofV3RolloutFlags(
        enabled=enabled or mandatory, mandatory=mandatory)


def acceptable_proof_modes(flags: ProofV3RolloutFlags) -> tuple[str, ...]:
    """Which proof stacks a validator scores in the current phase."""

    if flags.mandatory:
        return ("v3",)
    if flags.enabled:
        return ("v1", "v3")
    return ("v1",)


def audit_passes(
    flags: ProofV3RolloutFlags,
    *,
    v1_proof_valid: bool | None,
    v3_receipt_valid: bool | None,
) -> bool:
    """Dual-stack audit decision.

    ``None`` means the miner did not present that stack. During the
    transition either valid stack passes; once mandatory, only v3.
    """

    modes = acceptable_proof_modes(flags)
    if "v3" in modes and v3_receipt_valid:
        return True
    if "v1" in modes and v1_proof_valid:
        return True
    return False
