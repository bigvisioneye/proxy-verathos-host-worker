"""Fiat-Shamir batching for proof-v2 same-point PCS openings."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

from zkllm.crypto.gemm_v2_reference import (
    PALLAS_SCALAR_MODULUS,
    scalar_to_bytes,
)


_BATCH_CHALLENGE_DOMAIN = (
    b"VERATHOS/PROOF_V2/PCS_BATCH/XW_CHALLENGE/PALLAS_SCALAR/SHA256"
)
_BATCH_OPENING_DOMAIN = b"VERATHOS/PROOF_V2/PCS_BATCH/XW_OPENING/SHA256"
_CHALLENGE_LIMIT = (1 << 256) - ((1 << 256) % PALLAS_SCALAR_MODULUS)
_MAX_CHALLENGE_ATTEMPTS = 1 << 32


class XWBatchContextError(ValueError):
    """An X/W batch-opening context is malformed."""


@dataclass(frozen=True)
class XWBatchOpeningContextV2:
    """Derived context for one batched X/W opening at the shared K point."""

    batching_scalar: int
    combined_evaluation: int
    opening_outer_digest: bytes


@dataclass(frozen=True)
class SamePointBatchOpeningContextV2:
    """Derived random linear combination for same-point PCS claims."""

    coefficients: tuple[int, ...]
    combined_evaluation: int
    opening_outer_digest: bytes


_SAME_POINT_CHALLENGE_DOMAIN = (
    b"VERATHOS/PROOF_V2/PCS_BATCH/SAME_POINT_CHALLENGE/PALLAS_SCALAR/SHA256"
)
_SAME_POINT_OPENING_DOMAIN = b"VERATHOS/PROOF_V2/PCS_BATCH/SAME_POINT_OPENING/SHA256"
_MAX_SAME_POINT_TERMS = 256


def _fixed_bytes(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise XWBatchContextError(f"{name} must be exactly 32 bytes")
    return value


def _canonical_scalar(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise XWBatchContextError(f"{name} must be an integer")
    if value < 0 or value >= PALLAS_SCALAR_MODULUS:
        raise XWBatchContextError(f"{name} is not a canonical Pallas scalar")
    return value


def _record(fields: tuple[tuple[bytes, bytes], ...]) -> bytes:
    encoded = bytearray()
    for label, value in fields:
        encoded.extend(struct.pack("<H", len(label)))
        encoded.extend(label)
        encoded.extend(struct.pack("<I", len(value)))
        encoded.extend(value)
    return bytes(encoded)


def _batch_statement(
    *,
    statement_digest: object,
    x_commitment: object,
    w_commitment: object,
    x_evaluation: object,
    w_evaluation: object,
) -> bytes:
    return _record(
        (
            (b"protocol_version", struct.pack("<H", 2)),
            (b"statement_digest", _fixed_bytes(statement_digest, "statement digest")),
            (b"x_commitment", _fixed_bytes(x_commitment, "X commitment")),
            (b"w_commitment", _fixed_bytes(w_commitment, "W commitment")),
            (
                b"x_evaluation",
                scalar_to_bytes(_canonical_scalar(x_evaluation, "X evaluation")),
            ),
            (
                b"w_evaluation",
                scalar_to_bytes(_canonical_scalar(w_evaluation, "W evaluation")),
            ),
        )
    )


def derive_xw_batch_opening_context_v2(
    *,
    statement_digest: bytes,
    x_commitment: bytes,
    w_commitment: bytes,
    x_evaluation: int,
    w_evaluation: int,
) -> XWBatchOpeningContextV2:
    """Derive the nonzero X/W batching scalar and opening transcript context.

    The individual commitments and terminal evaluations are fixed before this
    challenge is sampled. The verifier derives the same values and does not
    accept a prover-supplied batching scalar.
    """

    batch_statement = _batch_statement(
        statement_digest=statement_digest,
        x_commitment=x_commitment,
        w_commitment=w_commitment,
        x_evaluation=x_evaluation,
        w_evaluation=w_evaluation,
    )
    batching_scalar = None
    for attempt in range(_MAX_CHALLENGE_ATTEMPTS):
        digest = hashlib.sha256(
            _BATCH_CHALLENGE_DOMAIN
            + batch_statement
            + _record(((b"attempt", struct.pack("<I", attempt)),))
        ).digest()
        candidate = int.from_bytes(digest, "little")
        if candidate < _CHALLENGE_LIMIT:
            scalar = candidate % PALLAS_SCALAR_MODULUS
            if scalar != 0:
                batching_scalar = scalar
                break
    if batching_scalar is None:
        raise RuntimeError(
            "X/W batching challenge rejection sampling did not terminate"
        )

    combined_evaluation = (
        _canonical_scalar(x_evaluation, "X evaluation")
        + batching_scalar * _canonical_scalar(w_evaluation, "W evaluation")
    ) % PALLAS_SCALAR_MODULUS
    opening_outer_digest = hashlib.sha256(
        _BATCH_OPENING_DOMAIN
        + batch_statement
        + _record(((b"batching_scalar", scalar_to_bytes(batching_scalar)),))
    ).digest()
    return XWBatchOpeningContextV2(
        batching_scalar=batching_scalar,
        combined_evaluation=combined_evaluation,
        opening_outer_digest=opening_outer_digest,
    )


def derive_same_point_batch_opening_context_v2(
    *,
    label: bytes,
    statement_digest: bytes,
    term_bindings: tuple[bytes, ...],
    evaluations: tuple[int, ...],
) -> SamePointBatchOpeningContextV2:
    """Bind and combine an ordered set of authenticated same-point claims.

    The first coefficient is fixed to one and every later coefficient is a
    nonzero Fiat--Shamir scalar sampled only after every term identity and
    evaluation claim is fixed. A term binding is a transcript identity, not
    necessarily a curve commitment; the caller separately enforces that the
    resulting aggregate opening commitment equals the authenticated vector
    fold.
    """

    if (
        not isinstance(label, bytes)
        or not label
        or len(label) > 64
        or any(value < 0x21 or value > 0x7E for value in label)
    ):
        raise XWBatchContextError("same-point batch label is not canonical ASCII")
    digest = _fixed_bytes(statement_digest, "statement digest")
    bindings = tuple(term_bindings)
    values = tuple(evaluations)
    if (
        not bindings
        or len(bindings) > _MAX_SAME_POINT_TERMS
        or len(values) != len(bindings)
    ):
        raise XWBatchContextError("same-point batch term count is out of range")
    fields: list[tuple[bytes, bytes]] = [
        (b"protocol_version", struct.pack("<H", 2)),
        (b"label", label),
        (b"statement_digest", digest),
        (b"term_count", struct.pack("<H", len(bindings))),
    ]
    for index, (binding, evaluation) in enumerate(zip(bindings, values)):
        suffix = struct.pack("<H", index)
        fields.append((b"term_binding" + suffix, _fixed_bytes(binding, "term binding")))
        fields.append(
            (
                b"evaluation" + suffix,
                scalar_to_bytes(_canonical_scalar(evaluation, "evaluation")),
            )
        )
    batch_statement = _record(tuple(fields))
    coefficients = [1]
    for index in range(1, len(bindings)):
        coefficient = None
        for attempt in range(_MAX_CHALLENGE_ATTEMPTS):
            challenge = hashlib.sha256(
                _SAME_POINT_CHALLENGE_DOMAIN
                + batch_statement
                + _record(
                    (
                        (b"term_index", struct.pack("<H", index)),
                        (b"attempt", struct.pack("<I", attempt)),
                    )
                )
            ).digest()
            candidate = int.from_bytes(challenge, "little")
            if candidate < _CHALLENGE_LIMIT:
                scalar = candidate % PALLAS_SCALAR_MODULUS
                if scalar != 0:
                    coefficient = scalar
                    break
        if coefficient is None:
            raise RuntimeError(
                "same-point batching challenge rejection sampling did not terminate"
            )
        coefficients.append(coefficient)
    combined_evaluation = (
        sum(coefficient * value for coefficient, value in zip(coefficients, values))
        % PALLAS_SCALAR_MODULUS
    )
    encoded_coefficients = b"".join(scalar_to_bytes(item) for item in coefficients)
    opening_outer_digest = hashlib.sha256(
        _SAME_POINT_OPENING_DOMAIN
        + batch_statement
        + _record(((b"coefficients", encoded_coefficients),))
    ).digest()
    return SamePointBatchOpeningContextV2(
        tuple(coefficients),
        combined_evaluation,
        opening_outer_digest,
    )


__all__ = [
    "XWBatchContextError",
    "XWBatchOpeningContextV2",
    "SamePointBatchOpeningContextV2",
    "derive_same_point_batch_opening_context_v2",
    "derive_xw_batch_opening_context_v2",
]
