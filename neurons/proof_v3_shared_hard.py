"""Independent verification of receipt-matched proof-v3 hard bundles."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence

from neurons.receipts import (
    ServiceReceipt,
    encode_receipt_message,
    receipt_to_dict,
)
from verallm.chain.wallet import ss58_decode
from verallm.proof_v3.errors import ProofV3VerificationError
from verallm.proof_v3.hard_bundle import (
    RetainedHardProofBundleV3,
    verify_retained_hard_proof_bundle_v3,
)
from verallm.proof_v3.request import (
    miner_hotkey_identity_digest_v3,
    validator_hotkey_identity_digest_v3,
)

_ORDER_DOMAIN = b"VERATHOS/PROOF_V3/SHARED_HARD_ORDER/V1/SHA256"

__all__ = [
    "SharedHardBundleUnavailable",
    "SharedHardBundlePeerFailure",
    "deterministic_shared_hard_order_v3",
    "shared_hard_receipt_cache_key_v3",
    "verify_indexed_shared_hard_bundle_v3",
    "verify_shared_hard_bundle_v3",
]


class SharedHardBundlePeerFailure(ProofV3VerificationError):
    """A receipt or retained bundle violated a miner protocol obligation."""


class SharedHardBundleUnavailable(SharedHardBundlePeerFailure):
    """The exact receipt-matched bundle has not become fetchable yet."""


def shared_hard_receipt_cache_key_v3(receipt: ServiceReceipt) -> bytes:
    """Key one exact signed receipt for safe prefetch reconciliation."""

    if not isinstance(receipt, ServiceReceipt):
        raise ValueError("shared hard proof receipt has an unexpected type")
    signature = bytes(receipt.validator_signature or b"")
    if len(signature) != 64:
        raise ValueError("shared hard proof receipt signature is malformed")
    return hashlib.sha256(
        b"VERATHOS/PROOF_V3/SHARED_HARD_RECEIPT_CACHE/V1/SHA256"
        + encode_receipt_message(receipt)
        + signature
    ).digest()


def _u64(value: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < 1 << 64
    ):
        raise ValueError(f"{name} must be a uint64")
    return value


def deterministic_shared_hard_order_v3(
    *,
    epoch_number: int,
    epoch_seed: bytes,
    endpoint_keys: Sequence[tuple[str, int, str]],
) -> tuple[tuple[str, int, str], ...]:
    """Order endpoint work identically from one public epoch seed."""

    epoch = _u64(epoch_number, "epoch_number")
    if not isinstance(epoch_seed, bytes) or len(epoch_seed) != 32:
        raise ValueError("epoch_seed must be exactly 32 bytes")
    normalized = []
    seen = set()
    for value in endpoint_keys:
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError("endpoint key is malformed")
        address, model_index, endpoint = value
        address = str(address).strip().lower()
        endpoint = str(endpoint).strip().rstrip("/")
        index = _u64(model_index, "model_index")
        if not address or not endpoint:
            raise ValueError("endpoint key is incomplete")
        key = (address, index, endpoint)
        if key in seen:
            raise ValueError("endpoint key is duplicated")
        seen.add(key)
        encoded_address = address.encode("utf-8")
        encoded_endpoint = endpoint.encode("utf-8")
        if len(encoded_address) > 512 or len(encoded_endpoint) > 4096:
            raise ValueError("endpoint key is too long")
        sort_digest = hashlib.sha256(
            _ORDER_DOMAIN
            + struct.pack("<Q", epoch)
            + epoch_seed
            + struct.pack("<I", len(encoded_address))
            + encoded_address
            + struct.pack("<Q", index)
            + struct.pack("<I", len(encoded_endpoint))
            + encoded_endpoint
        ).digest()
        normalized.append((sort_digest, key))
    normalized.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[1] for item in normalized)


def verify_shared_hard_bundle_v3(
    *,
    encoded_bundle: bytes,
    receipt: ServiceReceipt,
    qualified_release,
    hard_auditor_hotkey_ss58: str,
    miner_hotkey_ss58: str,
    expected_epoch_number: int,
    expected_miner_address: str,
    expected_model_id: str,
    expected_model_index: int,
):
    """Verify one hard-auditor receipt and retained proof independently."""

    if not isinstance(receipt, ServiceReceipt):
        raise SharedHardBundlePeerFailure(
            "shared hard proof receipt has an unexpected type"
        )
    try:
        hard_auditor_key = ss58_decode(hard_auditor_hotkey_ss58)
    except Exception as exc:
        raise ValueError("configured hard-auditor hotkey is malformed") from exc
    if receipt.validator_hotkey != hard_auditor_key:
        raise SharedHardBundlePeerFailure(
            "shared hard proof receipt belongs to another validator"
        )
    if (
        int(getattr(receipt, "receipt_version", 0) or 0) < 4
        or receipt.proof_requested is not True
        or receipt.proof_verified is not True
        or receipt.is_canary is not True
    ):
        raise SharedHardBundlePeerFailure(
            "shared hard proof receipt does not record a verified hard canary"
        )
    expected_address = str(expected_miner_address).strip().lower()
    expected_epoch = _u64(expected_epoch_number, "expected_epoch_number")
    if (
        not expected_address
        or receipt.epoch_number != expected_epoch
        or str(receipt.miner_address).strip().lower() != expected_address
        or receipt.model_id != expected_model_id
        or receipt.model_index != expected_model_index
    ):
        raise SharedHardBundlePeerFailure(
            "shared hard proof receipt belongs to another endpoint"
        )
    capture_chain_digest = bytes(
        getattr(receipt, "capture_chain_digest", b"") or b""
    )
    if len(capture_chain_digest) != 32 or capture_chain_digest == bytes(32):
        raise SharedHardBundlePeerFailure(
            "shared hard proof receipt lacks its capture-chain binding"
        )
    qualified_profile = getattr(qualified_release, "qualified_profile", None)
    if qualified_profile is None:
        raise ValueError("qualified proof-v3 release is unavailable")
    try:
        miner_identity = miner_hotkey_identity_digest_v3(miner_hotkey_ss58)
        validator_identity = validator_hotkey_identity_digest_v3(
            hard_auditor_hotkey_ss58
        )
    except Exception as exc:
        raise ValueError("shared hard proof identity is unavailable") from exc
    try:
        bundle = RetainedHardProofBundleV3.from_canonical_bytes(encoded_bundle)
        if bundle.precommit_context.context_token_count != int(
            receipt.prompt_tokens
        ):
            raise SharedHardBundlePeerFailure(
                "shared hard proof prompt count differs from its receipt"
            )
        if bundle.envelope.decode_token_count != int(receipt.tokens_generated):
            raise SharedHardBundlePeerFailure(
                "shared hard proof output count differs from its receipt"
            )
        return verify_retained_hard_proof_bundle_v3(
            bundle=bundle,
            qualified_profile=qualified_profile,
            expected_validator_identity_digest=validator_identity,
            expected_miner_identity_digest=miner_identity,
            expected_commitment_envelope_digest=receipt.commitment_hash,
            expected_capture_chain_digest=capture_chain_digest,
        )
    except SharedHardBundlePeerFailure:
        raise
    except ProofV3VerificationError as exc:
        raise SharedHardBundlePeerFailure(str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise SharedHardBundlePeerFailure(
            "shared hard proof bundle is malformed"
        ) from exc


def verify_indexed_shared_hard_bundle_v3(
    *,
    client,
    index: dict,
    receipt: ServiceReceipt,
    qualified_release,
    hard_auditor_hotkey_ss58: str,
    miner_hotkey_ss58: str,
    expected_epoch_number: int,
    expected_miner_address: str,
    expected_model_id: str,
    expected_model_index: int,
):
    """Fetch and verify the one index entry matching an exact receipt."""

    if not isinstance(index, dict) or not isinstance(
        index.get("bundles"), list
    ):
        raise SharedHardBundlePeerFailure(
            "shared hard proof index is malformed"
        )
    commitment_hex = receipt.commitment_hash.hex()
    entries = [
        entry
        for entry in index["bundles"]
        if isinstance(entry, dict)
        and entry.get("commitment_hash") == commitment_hex
    ]
    if len(entries) != 1:
        raise SharedHardBundleUnavailable(
            "receipt-matched hard bundle is missing"
        )
    entry = entries[0]
    profile_digest = qualified_release.qualified_profile.profile.digest()
    if (
        entry.get("receipt") != receipt_to_dict(receipt)
        or entry.get("execution_profile_digest") != profile_digest.hex()
    ):
        raise SharedHardBundlePeerFailure(
            "hard-bundle index does not match its receipt"
        )
    encoded = client.fetch_proof_v3_hard_bundle(
        epoch_number=expected_epoch_number,
        commitment_envelope_digest=receipt.commitment_hash,
    )
    if (
        entry.get("byte_length") != len(encoded)
        or entry.get("bundle_hash")
        != hashlib.sha256(encoded).hexdigest()
    ):
        raise SharedHardBundlePeerFailure(
            "hard-bundle index checksum is inconsistent"
        )
    return verify_shared_hard_bundle_v3(
        encoded_bundle=encoded,
        receipt=receipt,
        qualified_release=qualified_release,
        hard_auditor_hotkey_ss58=hard_auditor_hotkey_ss58,
        miner_hotkey_ss58=miner_hotkey_ss58,
        expected_epoch_number=expected_epoch_number,
        expected_miner_address=expected_miner_address,
        expected_model_id=expected_model_id,
        expected_model_index=expected_model_index,
    )
