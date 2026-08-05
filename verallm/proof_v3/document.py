"""Signed, canonical release artifacts for proof-v3 execution profiles.

The artifact is deliberately separate from a static weight manifest: the
static manifest remains immutable and may be shared by multiple qualified
execution profiles (for example, different cache-layout adapters).  The
profile signature binds its exact static-manifest and static-artifact digests,
quantization and cache semantics, verifier-key digest, capability flags, and
size limits.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

from verallm.proof_v3.errors import (
    ProofV3DocumentError,
    ProofV3SignatureError,
    ProofV3VerificationError,
)
from verallm.proof_v3.profile import ExecutionSecurityProfileV3
from verallm.proof_v3.relation import execution_relation_spec_from_dict


MAX_EXECUTION_PROFILE_DOCUMENT_BYTES = 1 << 20
MAX_EXECUTION_PROFILE_SIGNATURES = 32
_SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
_SECP256K1_HALF_ORDER = _SECP256K1_ORDER // 2
_ZERO_ADDRESS = b"\x00" * 20


def _address(value: str | bytes, name: str) -> bytes:
    if isinstance(value, bytes):
        address = value
    elif isinstance(value, str):
        if len(value) != 42 or not value.startswith("0x"):
            raise ProofV3SignatureError(f"{name} must be a 0x-prefixed 20-byte address")
        try:
            address = bytes.fromhex(value[2:])
        except ValueError as exc:
            raise ProofV3SignatureError(f"{name} must be hexadecimal") from exc
    else:
        raise ProofV3SignatureError(f"{name} must be bytes or an address")
    if len(address) != 20:
        raise ProofV3SignatureError(f"{name} must be exactly 20 bytes")
    return address


def _signature(value: str | bytes) -> bytes:
    if isinstance(value, bytes):
        signature = value
    elif isinstance(value, str):
        encoded = value[2:] if value.startswith("0x") else value
        try:
            signature = bytes.fromhex(encoded)
        except ValueError as exc:
            raise ProofV3SignatureError(
                "execution profile signature must be hexadecimal"
            ) from exc
    else:
        raise ProofV3SignatureError(
            "execution profile signature must be bytes or hexadecimal"
        )
    if len(signature) != 65:
        raise ProofV3SignatureError(
            "execution profile signature must be exactly 65 bytes"
        )
    r = int.from_bytes(signature[:32], "big")
    s = int.from_bytes(signature[32:64], "big")
    v = signature[64]
    if not 0 < r < _SECP256K1_ORDER or not 0 < s <= _SECP256K1_HALF_ORDER:
        raise ProofV3SignatureError("execution profile signature is not canonical")
    if v not in (27, 28):
        raise ProofV3SignatureError(
            "execution profile signature recovery id is not canonical"
        )
    return signature


def _expected_authorities(values: Collection[str | bytes]) -> frozenset[bytes]:
    authorities = tuple(_address(value, "expected authority") for value in values)
    if not authorities:
        raise ProofV3SignatureError("expected authority set must not be empty")
    if _ZERO_ADDRESS in authorities:
        raise ProofV3SignatureError(
            "expected authority set must not contain the zero address"
        )
    if len(authorities) != len(set(authorities)):
        raise ProofV3SignatureError(
            "expected authority set must not contain duplicates"
        )
    return frozenset(authorities)


def verify_signed_execution_profile_v3(
    profile: ExecutionSecurityProfileV3,
    signatures: Sequence[str | bytes],
    *,
    expected_static_manifest_digest: bytes,
    expected_authority_signers: Collection[str | bytes],
    authority_threshold: int,
) -> tuple[str, ...]:
    """Verify a threshold of EIP-191 authorities for one exact profile.

    The caller must obtain the expected authority set from the relevant model
    registry owner/multisig.  This function performs no network or vault I/O.
    """

    if not isinstance(profile, ExecutionSecurityProfileV3):
        raise ProofV3VerificationError("execution profile has an unexpected type")
    if (
        not isinstance(expected_static_manifest_digest, bytes)
        or len(expected_static_manifest_digest) != 32
    ):
        raise ProofV3VerificationError(
            "expected_static_manifest_digest must be exactly 32 bytes"
        )
    if profile.static_manifest_digest != expected_static_manifest_digest:
        raise ProofV3VerificationError(
            "execution profile has an unexpected static manifest"
        )
    authorities = _expected_authorities(expected_authority_signers)
    if (
        isinstance(authority_threshold, bool)
        or not isinstance(authority_threshold, int)
        or not 1 <= authority_threshold <= len(authorities)
        or authority_threshold > MAX_EXECUTION_PROFILE_SIGNATURES
    ):
        raise ProofV3SignatureError(
            "authority_threshold must be between 1 and authority count"
        )
    if not isinstance(signatures, Sequence) or isinstance(signatures, (str, bytes)):
        raise ProofV3SignatureError("execution profile signatures must be a sequence")
    if (
        not signatures
        or len(signatures) > len(authorities)
        or len(signatures) > MAX_EXECUTION_PROFILE_SIGNATURES
    ):
        raise ProofV3SignatureError("execution profile signature count is out of range")
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
        from eth_keys.exceptions import BadSignature
    except ImportError as exc:
        raise ProofV3DocumentError(
            "execution profile signature verification requires chain dependencies"
        ) from exc

    signable = encode_defunct(primitive=profile.digest())
    recovered: set[bytes] = set()
    for value in signatures:
        signature = _signature(value)
        try:
            signer = _address(
                Account.recover_message(signable, signature=signature),
                "recovered authority",
            )
        except (BadSignature, ProofV3SignatureError, ValueError, TypeError) as exc:
            raise ProofV3SignatureError(
                "execution profile signature recovery failed"
            ) from exc
        if signer not in authorities:
            raise ProofV3SignatureError(
                "execution profile signature was not produced by an expected authority"
            )
        if signer in recovered:
            raise ProofV3SignatureError(
                "execution profile signatures must have unique signers"
            )
        recovered.add(signer)
    if len(recovered) < authority_threshold:
        raise ProofV3SignatureError("execution profile authority threshold was not met")
    return tuple(f"0x{address.hex()}" for address in sorted(recovered))


def execution_profile_to_dict(profile: ExecutionSecurityProfileV3) -> dict[str, object]:
    """Render exactly the canonical profile fields for a release document."""

    if not isinstance(profile, ExecutionSecurityProfileV3):
        raise ProofV3DocumentError("execution profile has an unexpected type")
    return {
        "adapter_id": profile.adapter_id,
        "adapter_version": profile.adapter_version,
        "max_verified_context_tokens": profile.max_verified_context_tokens,
        "max_verified_decode_tokens": profile.max_verified_decode_tokens,
        "profile_version": profile.profile_version,
        "proof_system_id": profile.proof_system_id,
        "quantization_semantics_id": profile.quantization_semantics_id,
        "relation_spec": profile.relation_spec.to_dict(),
        "static_artifact_digest": profile.static_artifact_digest.hex(),
        "static_manifest_digest": profile.static_manifest_digest.hex(),
        "verifier_key_digest": profile.verifier_key_digest.hex(),
    }


def _object(value: object, keys: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ProofV3DocumentError(f"{name} fields do not match the canonical schema")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ProofV3DocumentError(f"{name} must be an unsigned integer")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ProofV3DocumentError(f"{name} must be a string")
    return value


def _hex(value: object, size: int, name: str) -> bytes:
    if not isinstance(value, str) or len(value) != size * 2 or value.lower() != value:
        raise ProofV3DocumentError(f"{name} must be lowercase hexadecimal")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ProofV3DocumentError(f"{name} must be hexadecimal") from exc
    if len(decoded) != size:
        raise ProofV3DocumentError(f"{name} must be exactly {size} bytes")
    return decoded


def execution_profile_from_dict(value: object) -> ExecutionSecurityProfileV3:
    """Parse an exact JSON object and re-run canonical profile validation."""

    profile = _object(
        value,
        {
            "adapter_id",
            "adapter_version",
            "max_verified_context_tokens",
            "max_verified_decode_tokens",
            "profile_version",
            "proof_system_id",
            "quantization_semantics_id",
            "relation_spec",
            "static_artifact_digest",
            "static_manifest_digest",
            "verifier_key_digest",
        },
        "execution profile",
    )
    try:
        return ExecutionSecurityProfileV3(
            static_manifest_digest=_hex(
                profile["static_manifest_digest"],
                32,
                "static_manifest_digest",
            ),
            static_artifact_digest=_hex(
                profile["static_artifact_digest"],
                32,
                "static_artifact_digest",
            ),
            adapter_id=_text(profile["adapter_id"], "adapter_id"),
            adapter_version=_text(profile["adapter_version"], "adapter_version"),
            proof_system_id=_text(
                profile["proof_system_id"],
                "proof_system_id",
            ),
            verifier_key_digest=_hex(
                profile["verifier_key_digest"],
                32,
                "verifier_key_digest",
            ),
            quantization_semantics_id=_text(
                profile["quantization_semantics_id"],
                "quantization_semantics_id",
            ),
            max_verified_context_tokens=_integer(
                profile["max_verified_context_tokens"],
                "max_verified_context_tokens",
            ),
            max_verified_decode_tokens=_integer(
                profile["max_verified_decode_tokens"],
                "max_verified_decode_tokens",
            ),
            relation_spec=execution_relation_spec_from_dict(profile["relation_spec"]),
            profile_version=_integer(profile["profile_version"], "profile_version"),
        )
    except ProofV3DocumentError:
        raise
    except Exception as exc:
        raise ProofV3DocumentError("execution profile body is not canonical") from exc


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProofV3DocumentError("JSON object has duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ProofV3DocumentError("JSON constants are not allowed")


@dataclass(frozen=True, slots=True)
class SignedExecutionProfileDocumentV3:
    """A content-addressable, authority-signed v3 execution profile artifact."""

    profile: ExecutionSecurityProfileV3
    signatures: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ExecutionSecurityProfileV3):
            raise ProofV3DocumentError("execution profile has an unexpected type")
        signatures = tuple(self.signatures)
        if not signatures or len(signatures) > MAX_EXECUTION_PROFILE_SIGNATURES:
            raise ProofV3DocumentError(
                "execution profile signature count is out of range"
            )
        try:
            canonical = tuple(sorted(_signature(signature) for signature in signatures))
        except ProofV3SignatureError as exc:
            raise ProofV3DocumentError(
                "execution profile signatures are not canonical"
            ) from exc
        if len(canonical) != len(set(canonical)):
            raise ProofV3DocumentError(
                "execution profile signatures must not contain duplicates"
            )
        object.__setattr__(self, "signatures", canonical)

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": execution_profile_to_dict(self.profile),
            "signatures": [signature.hex() for signature in self.signatures],
        }

    def canonical_json_bytes(self) -> bytes:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        if len(encoded) > MAX_EXECUTION_PROFILE_DOCUMENT_BYTES:
            raise ProofV3DocumentError(
                "execution profile document exceeds the protocol limit"
            )
        return encoded

    @classmethod
    def from_dict(cls, value: object) -> "SignedExecutionProfileDocumentV3":
        document = _object(value, {"profile", "signatures"}, "document")
        signatures_value = document["signatures"]
        if not isinstance(signatures_value, list):
            raise ProofV3DocumentError("signatures must be a list")
        try:
            signatures = tuple(_signature(value) for value in signatures_value)
        except ProofV3SignatureError as exc:
            raise ProofV3DocumentError(
                "execution profile signatures are not canonical"
            ) from exc
        return cls(execution_profile_from_dict(document["profile"]), signatures)

    @classmethod
    def from_json_bytes(cls, encoded: bytes) -> "SignedExecutionProfileDocumentV3":
        if (
            not isinstance(encoded, bytes)
            or not encoded
            or len(encoded) > MAX_EXECUTION_PROFILE_DOCUMENT_BYTES
        ):
            raise ProofV3DocumentError(
                "execution profile document length is out of range"
            )
        try:
            value = json.loads(
                encoded.decode("ascii"),
                object_pairs_hook=_unique_pairs,
                parse_constant=_reject_json_constant,
            )
        except ProofV3DocumentError:
            raise
        except (RecursionError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProofV3DocumentError(
                "execution profile document is not valid JSON"
            ) from exc
        result = cls.from_dict(value)
        if result.canonical_json_bytes() != encoded:
            raise ProofV3DocumentError(
                "execution profile document is not canonical JSON"
            )
        return result


def load_signed_execution_profile_document_v3(
    path: str | Path,
) -> SignedExecutionProfileDocumentV3:
    """Load one bounded release artifact without any implicit network access."""

    artifact_path = Path(path).expanduser().resolve()
    try:
        size = artifact_path.stat().st_size
        if size <= 0 or size > MAX_EXECUTION_PROFILE_DOCUMENT_BYTES:
            raise ProofV3DocumentError(
                "execution profile document file size is out of range"
            )
        with artifact_path.open("rb") as handle:
            encoded = handle.read(MAX_EXECUTION_PROFILE_DOCUMENT_BYTES + 1)
        if len(encoded) != size:
            raise ProofV3DocumentError(
                "execution profile document file size changed while loading"
            )
    except ProofV3DocumentError:
        raise
    except OSError as exc:
        raise ProofV3DocumentError(
            f"cannot read execution profile document: {artifact_path}"
        ) from exc
    return SignedExecutionProfileDocumentV3.from_json_bytes(encoded)


__all__ = [
    "MAX_EXECUTION_PROFILE_DOCUMENT_BYTES",
    "MAX_EXECUTION_PROFILE_SIGNATURES",
    "SignedExecutionProfileDocumentV3",
    "execution_profile_from_dict",
    "execution_profile_to_dict",
    "load_signed_execution_profile_document_v3",
    "verify_signed_execution_profile_v3",
]
