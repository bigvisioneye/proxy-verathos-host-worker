"""Wire-level proof protocol negotiation and reveal helpers.

Omitting ``proof_protocol_version`` is the legacy v1 wire contract.  Keeping
that distinction (rather than eagerly serializing ``1``) lets updated clients
talk to miners that predate explicit protocol negotiation.
"""

import hashlib
import hmac
from typing import Optional


LEGACY_PROOF_PROTOCOL_VERSION = 1
PROOF_PROTOCOL_V2 = 2
PROOF_PROTOCOL_V3 = 3
SUPPORTED_PROOF_PROTOCOL_VERSIONS = frozenset(
    (
        LEGACY_PROOF_PROTOCOL_VERSION,
        PROOF_PROTOCOL_V2,
        PROOF_PROTOCOL_V3,
    )
)
VALIDATOR_NONCE_SIZE = 32
PROOF_CHALLENGE_ID_SIZE = 32
PROOF_COMMITMENT_HASH_SIZE = 32
_VALIDATOR_NONCE_COMMITMENT_DOMAIN = (
    b"VERATHOS_PROOF_V2_VALIDATOR_NONCE_COMMITMENT_V1"
)


def validate_proof_protocol_version(version: Optional[int]) -> Optional[int]:
    """Validate an explicitly requested wire version without defaulting it."""
    if version is None:
        return None
    if type(version) is not int or version not in SUPPORTED_PROOF_PROTOCOL_VERSIONS:
        supported = ", ".join(str(v) for v in sorted(SUPPORTED_PROOF_PROTOCOL_VERSIONS))
        raise ValueError(f"unsupported proof_protocol_version; expected one of: {supported}")
    return version


def resolve_proof_protocol_version(version: Optional[int]) -> int:
    """Resolve an omitted wire version to the legacy v1 protocol."""
    validated = validate_proof_protocol_version(version)
    if validated is None:
        return LEGACY_PROOF_PROTOCOL_VERSION
    return validated


def encode_validator_nonce(nonce: bytes) -> str:
    """Return the canonical request encoding for an exact 32-byte nonce."""
    if not isinstance(nonce, bytes) or len(nonce) != VALIDATOR_NONCE_SIZE:
        raise ValueError(f"validator_nonce must be exactly {VALIDATOR_NONCE_SIZE} bytes")
    return nonce.hex()


def decode_validator_nonce(nonce_hex: str) -> bytes:
    """Decode and validate the request's exact 32-byte hexadecimal nonce."""
    if not isinstance(nonce_hex, str) or len(nonce_hex) != VALIDATOR_NONCE_SIZE * 2:
        raise ValueError(
            f"validator_nonce must be exactly {VALIDATOR_NONCE_SIZE * 2} hexadecimal characters"
        )
    try:
        nonce = bytes.fromhex(nonce_hex)
    except ValueError as exc:
        raise ValueError("validator_nonce must be hexadecimal") from exc
    if len(nonce) != VALIDATOR_NONCE_SIZE:
        raise ValueError(f"validator_nonce must decode to exactly {VALIDATOR_NONCE_SIZE} bytes")
    return nonce


def encode_proof_challenge_id(challenge_id: bytes) -> str:
    """Return the canonical encoding for an exact 32-byte challenge id."""
    if (
        not isinstance(challenge_id, bytes)
        or len(challenge_id) != PROOF_CHALLENGE_ID_SIZE
    ):
        raise ValueError(
            f"proof_challenge_id must be exactly {PROOF_CHALLENGE_ID_SIZE} bytes"
        )
    return challenge_id.hex()


def decode_proof_challenge_id(challenge_id_hex: str) -> bytes:
    """Decode and validate a request's exact 32-byte challenge id."""
    if (
        not isinstance(challenge_id_hex, str)
        or len(challenge_id_hex) != PROOF_CHALLENGE_ID_SIZE * 2
    ):
        raise ValueError(
            "proof_challenge_id must be exactly "
            f"{PROOF_CHALLENGE_ID_SIZE * 2} hexadecimal characters"
        )
    try:
        challenge_id = bytes.fromhex(challenge_id_hex)
    except ValueError as exc:
        raise ValueError("proof_challenge_id must be hexadecimal") from exc
    if len(challenge_id) != PROOF_CHALLENGE_ID_SIZE:
        raise ValueError(
            "proof_challenge_id must decode to exactly "
            f"{PROOF_CHALLENGE_ID_SIZE} bytes"
        )
    return challenge_id


def encode_proof_commitment_hash(commitment_hash: bytes) -> str:
    """Return the canonical encoding for an exact 32-byte commitment hash."""
    if (
        not isinstance(commitment_hash, bytes)
        or len(commitment_hash) != PROOF_COMMITMENT_HASH_SIZE
    ):
        raise ValueError(
            "proof commitment hash must be exactly "
            f"{PROOF_COMMITMENT_HASH_SIZE} bytes"
        )
    return commitment_hash.hex()


def decode_proof_commitment_hash(commitment_hash_hex: str) -> bytes:
    """Decode and validate an exact 32-byte commitment hash."""
    if (
        not isinstance(commitment_hash_hex, str)
        or len(commitment_hash_hex) != PROOF_COMMITMENT_HASH_SIZE * 2
    ):
        raise ValueError(
            "proof commitment hash must be exactly "
            f"{PROOF_COMMITMENT_HASH_SIZE * 2} hexadecimal characters"
        )
    try:
        commitment_hash = bytes.fromhex(commitment_hash_hex)
    except ValueError as exc:
        raise ValueError("proof commitment hash must be hexadecimal") from exc
    if len(commitment_hash) != PROOF_COMMITMENT_HASH_SIZE:
        raise ValueError(
            "proof commitment hash must decode to exactly "
            f"{PROOF_COMMITMENT_HASH_SIZE} bytes"
        )
    return commitment_hash


def commit_validator_nonce_v2(
    validator_nonce: bytes,
    proof_challenge_id: bytes,
) -> bytes:
    """Bind a hidden validator nonce to one v2 inference request."""
    if (
        not isinstance(validator_nonce, bytes)
        or len(validator_nonce) != VALIDATOR_NONCE_SIZE
    ):
        raise ValueError(
            f"validator_nonce must be exactly {VALIDATOR_NONCE_SIZE} bytes"
        )
    if (
        not isinstance(proof_challenge_id, bytes)
        or len(proof_challenge_id) != PROOF_CHALLENGE_ID_SIZE
    ):
        raise ValueError(
            f"proof_challenge_id must be exactly {PROOF_CHALLENGE_ID_SIZE} bytes"
        )
    return hashlib.sha256(
        _VALIDATOR_NONCE_COMMITMENT_DOMAIN
        + proof_challenge_id
        + validator_nonce
    ).digest()


def validator_nonce_matches_commitment_v2(
    *,
    validator_nonce: bytes,
    proof_challenge_id: bytes,
    expected_commitment: bytes,
) -> bool:
    """Compare a revealed nonce with its request commitment in constant time."""
    if not isinstance(expected_commitment, bytes) or len(expected_commitment) != 32:
        return False
    try:
        actual = commit_validator_nonce_v2(validator_nonce, proof_challenge_id)
    except ValueError:
        return False
    return hmac.compare_digest(actual, expected_commitment)


def add_proof_protocol_version(request_body: dict, version: Optional[int]) -> dict:
    """Add an explicit negotiated version while preserving omitted-field v1."""
    validated = validate_proof_protocol_version(version)
    if validated is not None:
        request_body["proof_protocol_version"] = validated
    return request_body
