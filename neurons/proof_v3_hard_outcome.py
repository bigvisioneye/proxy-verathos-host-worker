"""Canonical owner-signed proof-v3 hard-audit failure outcomes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace


HARD_AUDIT_FAILURE_ABI_V3 = "verathos.proof_v3.hard_failure.v1"
HARD_AUDIT_FAILURE_CODES_V3 = frozenset(
    {
        "obligation_missing",
        "post_precommit_failure",
        "late_precommit_busy",
    }
)


@dataclass(frozen=True)
class HardAuditFailureV3:
    source_epoch: int
    miner_address: str
    model_id: str
    model_index: int
    obligation_id: bytes
    failure_code: str
    observed_at: int
    validator_hotkey: bytes
    validator_signature: bytes = b""
    abi_id: str = HARD_AUDIT_FAILURE_ABI_V3

    def validate(self) -> None:
        if self.abi_id != HARD_AUDIT_FAILURE_ABI_V3:
            raise ValueError("hard-audit failure ABI is unsupported")
        if type(self.source_epoch) is not int or self.source_epoch < 0:
            raise ValueError("hard-audit failure epoch is invalid")
        address = str(self.miner_address).strip().lower()
        if (
            len(address) != 42
            or not address.startswith("0x")
        ):
            raise ValueError("hard-audit failure miner address is invalid")
        try:
            bytes.fromhex(address[2:])
        except ValueError as exc:
            raise ValueError(
                "hard-audit failure miner address is invalid"
            ) from exc
        if not self.model_id or len(self.model_id.encode("utf-8")) > 512:
            raise ValueError("hard-audit failure model id is invalid")
        if type(self.model_index) is not int or self.model_index < 0:
            raise ValueError("hard-audit failure model index is invalid")
        if len(self.obligation_id) != 16:
            raise ValueError("hard-audit failure obligation id is invalid")
        if self.failure_code not in HARD_AUDIT_FAILURE_CODES_V3:
            raise ValueError("hard-audit failure code is unsupported")
        if type(self.observed_at) is not int or self.observed_at <= 0:
            raise ValueError("hard-audit failure timestamp is invalid")
        if len(self.validator_hotkey) != 32:
            raise ValueError("hard-audit failure signer is invalid")
        if self.validator_signature and len(self.validator_signature) != 64:
            raise ValueError("hard-audit failure signature is invalid")

    def unsigned_dict(self) -> dict:
        self.validate()
        return {
            "abi_id": self.abi_id,
            "source_epoch": self.source_epoch,
            "miner_address": self.miner_address.lower(),
            "model_id": self.model_id,
            "model_index": self.model_index,
            "obligation_id": self.obligation_id.hex(),
            "failure_code": self.failure_code,
            "observed_at": self.observed_at,
            "validator_hotkey": self.validator_hotkey.hex(),
        }

    def signing_message(self) -> bytes:
        encoded = json.dumps(
            self.unsigned_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return b"VERATHOS/PROOF_V3/HARD_FAILURE/V1\x00" + encoded

    def digest(self) -> bytes:
        return hashlib.sha256(self.signing_message()).digest()

    def to_dict(self) -> dict:
        value = self.unsigned_dict()
        value["validator_signature"] = self.validator_signature.hex()
        return value

    @classmethod
    def from_dict(cls, value: dict) -> "HardAuditFailureV3":
        if not isinstance(value, dict):
            raise ValueError("hard-audit failure must be an object")
        required = {
            "abi_id",
            "source_epoch",
            "miner_address",
            "model_id",
            "model_index",
            "obligation_id",
            "failure_code",
            "observed_at",
            "validator_hotkey",
            "validator_signature",
        }
        if set(value) != required:
            raise ValueError("hard-audit failure fields are invalid")
        try:
            outcome = cls(
                abi_id=str(value["abi_id"]),
                source_epoch=int(value["source_epoch"]),
                miner_address=str(value["miner_address"]).lower(),
                model_id=str(value["model_id"]),
                model_index=int(value["model_index"]),
                obligation_id=bytes.fromhex(str(value["obligation_id"])),
                failure_code=str(value["failure_code"]),
                observed_at=int(value["observed_at"]),
                validator_hotkey=bytes.fromhex(
                    str(value["validator_hotkey"])
                ),
                validator_signature=bytes.fromhex(
                    str(value["validator_signature"])
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("hard-audit failure encoding is invalid") from exc
        outcome.validate()
        if len(outcome.validator_signature) != 64:
            raise ValueError("hard-audit failure signature is missing")
        return outcome


def sign_hard_audit_failure_v3(
    outcome: HardAuditFailureV3,
    validator_seed: bytes,
) -> HardAuditFailureV3:
    """Sign one canonical failure with the designated validator hotkey."""

    if len(validator_seed) < 32:
        raise ValueError("hard-audit failure signing seed is invalid")
    outcome.validate()
    from bittensor_wallet import Keypair

    keypair = Keypair.create_from_seed(validator_seed[:32].hex())
    if bytes(keypair.public_key) != outcome.validator_hotkey:
        raise ValueError("hard-audit failure signing key does not match")
    signature = keypair.sign(outcome.signing_message())
    if not isinstance(signature, bytes):
        signature = bytes.fromhex(
            signature[2:] if signature.startswith("0x") else signature
        )
    signed = replace(outcome, validator_signature=signature)
    signed.validate()
    return signed


def verify_hard_audit_failure_v3(
    outcome: HardAuditFailureV3,
    *,
    expected_validator_hotkey_ss58: str,
) -> bool:
    """Verify canonical encoding and the exact configured owner identity."""

    try:
        outcome.validate()
        if len(outcome.validator_signature) != 64:
            return False
        from bittensor_wallet import Keypair
        from verallm.chain.wallet import ss58_decode

        expected_hotkey = bytes(
            ss58_decode(expected_validator_hotkey_ss58)
        )
        if outcome.validator_hotkey != expected_hotkey:
            return False
        keypair = Keypair(ss58_address=expected_validator_hotkey_ss58)
        return bool(
            keypair.verify(
                outcome.signing_message(),
                outcome.validator_signature,
            )
        )
    except Exception:
        return False


__all__ = [
    "HARD_AUDIT_FAILURE_ABI_V3",
    "HARD_AUDIT_FAILURE_CODES_V3",
    "HardAuditFailureV3",
    "sign_hard_audit_failure_v3",
    "verify_hard_audit_failure_v3",
]
