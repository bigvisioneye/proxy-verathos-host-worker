"""Compact full-row projection checks for lean proof-v3 corridors.

The lean execution profile reconstructs nonce-selected layer corridors after
serving.  Every reconstructed projection therefore has to bind its *complete*
output row to the authenticated static weights; checking a few output cells is
not sufficient when the intermediate row itself was not committed pre-nonce.

For each projection, four independent unsigned-31-bit output folds turn
``S = X @ W`` into four inner-product claims::

    <X, W @ u_f> = <S, u_f>

The signed catalog authenticates the commitment of every ``W @ u_f``.  Claims
with the same padded input width share one aggregate sumcheck and one Pallas
IPA opening.  Consequently the proof carries no folded weight vectors and its
size is logarithmic in the input width.

This module contains the portable reference prover and production verifier.
The production prover may replace only the folded-vector and sumcheck
construction with a native/GPU implementation; the transcript and wire values
must remain byte-identical.
"""

from __future__ import annotations

import hashlib
import struct
import sys
from array import array
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.lean_projection_fold import (
    LEAN_PROJECTION_FOLD_COUNT_V3,
    LeanProjectionCatalogOperationV3,
    LeanProjectionFoldV3,
    build_lean_projection_fold_reference_v3,
    derive_lean_projection_coefficients_v3,
    registered_catalog_operations_v3,
)
from zkllm.crypto.gemm_v2_reference import (
    PALLAS_SCALAR_MODULUS,
    scalar_from_bytes,
    scalar_to_bytes,
)
from zkllm.crypto.pcs_v2 import (
    MAX_LEAN_PROJECTION_VECTORS,
    PCSOpeningV2,
)


LEAN_PROJECTION_BATCH_ABI_V3: Final = (
    "projection_fold.full_row.u31x4.batch_sumcheck_ipa.aggregate_catalog.v3"
)
MAX_LEAN_PROJECTION_BATCH_CLAIMS_V3: Final = 512
MAX_LEAN_PROJECTION_BATCH_WIRE_BYTES_V3: Final = 1 << 20

_TRANSCRIPT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/LEAN_PROJECTION_BATCH/U31X4/AGGREGATE_CATALOG/V2/SHA256"
)
_CHALLENGE_LIMIT: Final = (
    (1 << 256) - ((1 << 256) % PALLAS_SCALAR_MODULUS)
)
_WIRE_MAGIC: Final = b"VLPB"
_WIRE_VERSION: Final = 2

__all__ = [
    "LEAN_PROJECTION_BATCH_ABI_V3",
    "MAX_LEAN_PROJECTION_BATCH_CLAIMS_V3",
    "MAX_LEAN_PROJECTION_BATCH_WIRE_BYTES_V3",
    "LeanProjectionBatchClaimV3",
    "LeanProjectionBatchGroupProofV3",
    "LeanProjectionBatchProofV3",
    "build_lean_projection_batch_from_folds_v3",
    "build_lean_projection_batch_reference_v3",
    "decode_lean_projection_batch_v3",
    "encode_lean_projection_batch_v3",
    "verify_lean_projection_batch_v3",
]


def _fixed32(value: object, name: str, *, nonzero: bool = False) -> bytes:
    if (
        not isinstance(value, bytes)
        or len(value) != 32
        or (nonzero and not any(value))
    ):
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    return value


def _field(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= PALLAS_SCALAR_MODULUS
    ):
        raise ProofV3Error(f"{name} is not a canonical Pallas scalar")
    return value


def _signed_row(
    values: object,
    *,
    length: int,
    bits: int,
    name: str,
) -> tuple[int, ...]:
    try:
        row = tuple(int(value) for value in values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ProofV3Error(f"{name} is malformed") from exc
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if len(row) != length or any(value < minimum or value > maximum for value in row):
        raise ProofV3Error(f"{name} is not a canonical signed-i{bits} row")
    return row


def _record(label: bytes, value: bytes) -> bytes:
    return (
        struct.pack("<H", len(label))
        + label
        + struct.pack("<Q", len(value))
        + value
    )


class _Transcript:
    def __init__(self, binding: bytes) -> None:
        self.state = hashlib.sha256(
            _TRANSCRIPT_DOMAIN
            + _record(
                b"abi",
                LEAN_PROJECTION_BATCH_ABI_V3.encode("ascii"),
            )
            + _record(b"binding", binding)
        ).digest()

    def absorb(self, label: bytes, value: bytes) -> None:
        self.state = hashlib.sha256(
            self.state + _record(label, value)
        ).digest()

    def scalar(self, label: bytes) -> int:
        for counter in range(1 << 32):
            digest = hashlib.sha256(
                self.state
                + _record(b"challenge_label", label)
                + _record(b"counter", struct.pack("<I", counter))
            ).digest()
            candidate = int.from_bytes(digest, "little")
            if candidate < _CHALLENGE_LIMIT:
                result = candidate % PALLAS_SCALAR_MODULUS
                self.absorb(
                    b"challenge",
                    _record(b"label", label)
                    + _record(b"digest", digest)
                    + _record(b"scalar", scalar_to_bytes(result)),
                )
                return result
        raise ProofV3Error("lean projection challenge derivation did not terminate")


@dataclass(frozen=True, slots=True)
class LeanProjectionBatchClaimV3:
    """One externally-authenticated selected-row projection claim."""

    operation: LeanProjectionCatalogOperationV3
    input_row_i8: tuple[int, ...]
    surrogate_output_i64: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operation, LeanProjectionCatalogOperationV3):
            raise ProofV3Error("lean projection claim operation is malformed")
        object.__setattr__(
            self,
            "input_row_i8",
            _signed_row(
                self.input_row_i8,
                length=self.operation.input_dim,
                bits=8,
                name="lean projection claim input",
            ),
        )
        object.__setattr__(
            self,
            "surrogate_output_i64",
            _signed_row(
                self.surrogate_output_i64,
                length=self.operation.output_dim,
                bits=64,
                name="lean projection claim output",
            ),
        )


@dataclass(frozen=True, slots=True)
class LeanProjectionBatchGroupProofV3:
    """One shared-width aggregate sumcheck and authenticated PCS opening."""

    padded_input_dim: int
    claim_indices: tuple[int, ...]
    rounds: tuple[tuple[int, int, int], ...]
    terminal_fold_evaluations: tuple[int, ...]
    opening: PCSOpeningV2

    def __post_init__(self) -> None:
        width = self.padded_input_dim
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or width & (width - 1)
        ):
            raise ProofV3Error("lean projection batch width is invalid")
        indices = tuple(self.claim_indices)
        if (
            not indices
            or indices != tuple(sorted(set(indices)))
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                for index in indices
            )
        ):
            raise ProofV3Error("lean projection batch claim indices are invalid")
        rounds = tuple(tuple(round_) for round_ in self.rounds)
        if (
            len(rounds) != width.bit_length() - 1
            or any(
                len(round_) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value < PALLAS_SCALAR_MODULUS
                    for value in round_
                )
                for round_ in rounds
            )
        ):
            raise ProofV3Error("lean projection batch rounds are malformed")
        terminal = tuple(self.terminal_fold_evaluations)
        expected_terminal = len(indices) * LEAN_PROJECTION_FOLD_COUNT_V3
        if (
            expected_terminal > MAX_LEAN_PROJECTION_VECTORS
            or len(terminal) != expected_terminal
        ):
            raise ProofV3Error(
                "lean projection batch terminal evaluation count is wrong"
            )
        for value in terminal:
            _field(value, "lean projection terminal evaluation")
        if not isinstance(self.opening, PCSOpeningV2):
            raise ProofV3Error("lean projection batch opening is malformed")
        object.__setattr__(self, "claim_indices", indices)
        object.__setattr__(self, "rounds", rounds)
        object.__setattr__(self, "terminal_fold_evaluations", terminal)


@dataclass(frozen=True, slots=True)
class LeanProjectionBatchProofV3:
    groups: tuple[LeanProjectionBatchGroupProofV3, ...]

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        if (
            not groups
            or not all(isinstance(group, LeanProjectionBatchGroupProofV3)
                       for group in groups)
            or tuple(
                (group.padded_input_dim, group.claim_indices[0])
                for group in groups
            )
            != tuple(
                sorted(
                    (group.padded_input_dim, group.claim_indices[0])
                    for group in groups
                )
            )
        ):
            raise ProofV3Error("lean projection batch groups are malformed")
        claimed = tuple(
            index for group in groups for index in group.claim_indices
        )
        if len(claimed) != len(set(claimed)):
            raise ProofV3Error("lean projection claim is assigned more than once")
        object.__setattr__(self, "groups", groups)


def encode_lean_projection_batch_v3(
    proof: LeanProjectionBatchProofV3,
) -> bytes:
    """Encode the compact proof in one strict, bounded canonical form."""

    if not isinstance(proof, LeanProjectionBatchProofV3):
        raise ProofV3Error("lean projection batch proof has a wrong type")
    encoded = bytearray(
        _WIRE_MAGIC + struct.pack("<HH", _WIRE_VERSION, len(proof.groups))
    )
    for group in proof.groups:
        encoded.extend(
            struct.pack(
                "<IHH",
                group.padded_input_dim,
                len(group.claim_indices),
                len(group.rounds),
            )
        )
        encoded.extend(
            struct.pack(
                f"<{len(group.claim_indices)}H",
                *group.claim_indices,
            )
        )
        for round_ in group.rounds:
            encoded.extend(
                b"".join(scalar_to_bytes(value) for value in round_)
            )
        encoded.extend(
            struct.pack("<H", len(group.terminal_fold_evaluations))
        )
        encoded.extend(
            b"".join(
                scalar_to_bytes(value)
                for value in group.terminal_fold_evaluations
            )
        )
        opening = group.opening
        if len(opening.proof) >= 1 << 16:
            raise ProofV3Error("lean projection IPA proof is too large")
        encoded.extend(opening.commitment)
        encoded.extend(opening.evaluation)
        encoded.extend(
            struct.pack(
                "<IIBH",
                opening.vector_length,
                opening.padded_length,
                opening.encoding,
                len(opening.proof),
            )
        )
        encoded.extend(opening.proof)
    result = bytes(encoded)
    if len(result) > MAX_LEAN_PROJECTION_BATCH_WIRE_BYTES_V3:
        raise ProofV3Error("lean projection batch exceeds the wire bound")
    return result


class _WireReader:
    def __init__(self, encoded: bytes) -> None:
        if (
            not isinstance(encoded, bytes)
            or not encoded
            or len(encoded) > MAX_LEAN_PROJECTION_BATCH_WIRE_BYTES_V3
        ):
            raise ProofV3Error("lean projection batch wire length is invalid")
        self.encoded = encoded
        self.offset = 0

    def take(self, count: int) -> bytes:
        if count < 0 or self.offset + count > len(self.encoded):
            raise ProofV3Error("lean projection batch wire is truncated")
        result = self.encoded[self.offset : self.offset + count]
        self.offset += count
        return result

    def unpack(self, format_: str) -> tuple:
        size = struct.calcsize(format_)
        return struct.unpack(format_, self.take(size))

    def finish(self) -> None:
        if self.offset != len(self.encoded):
            raise ProofV3Error("lean projection batch wire has trailing bytes")


def decode_lean_projection_batch_v3(
    encoded: bytes,
) -> LeanProjectionBatchProofV3:
    """Decode and canonical-check a compact full-row projection proof."""

    reader = _WireReader(encoded)
    if reader.take(4) != _WIRE_MAGIC:
        raise ProofV3Error("lean projection batch wire magic is wrong")
    version, group_count = reader.unpack("<HH")
    if version != _WIRE_VERSION or not 0 < group_count <= 32:
        raise ProofV3Error("lean projection batch wire header is unsupported")
    groups = []
    total_claims = 0
    for _group_index in range(group_count):
        width, claim_count, round_count = reader.unpack("<IHH")
        if (
            not 0 < claim_count <= MAX_LEAN_PROJECTION_BATCH_CLAIMS_V3
            or round_count > 31
        ):
            raise ProofV3Error("lean projection batch wire counts are invalid")
        claim_indices = tuple(
            reader.unpack(f"<{claim_count}H")
        )
        total_claims += claim_count
        if total_claims > MAX_LEAN_PROJECTION_BATCH_CLAIMS_V3:
            raise ProofV3Error("lean projection batch wire has too many claims")
        rounds = tuple(
            tuple(
                scalar_from_bytes(reader.take(32))
                for _ in range(3)
            )
            for _ in range(round_count)
        )
        terminal_count = reader.unpack("<H")[0]
        if terminal_count != claim_count * LEAN_PROJECTION_FOLD_COUNT_V3:
            raise ProofV3Error(
                "lean projection batch terminal count is not canonical"
            )
        terminal = tuple(
            scalar_from_bytes(reader.take(32))
            for _ in range(terminal_count)
        )
        commitment = reader.take(32)
        evaluation = reader.take(32)
        vector_length, padded_length, encoding, proof_length = reader.unpack(
            "<IIBH"
        )
        if proof_length > 4096:
            raise ProofV3Error("lean projection IPA proof exceeds the bound")
        opening = PCSOpeningV2(
            commitment=commitment,
            evaluation=evaluation,
            proof=reader.take(proof_length),
            vector_length=vector_length,
            padded_length=padded_length,
            encoding=encoding,
        )
        groups.append(
            LeanProjectionBatchGroupProofV3(
                padded_input_dim=width,
                claim_indices=claim_indices,
                rounds=rounds,
                terminal_fold_evaluations=terminal,
                opening=opening,
            )
        )
    reader.finish()
    result = LeanProjectionBatchProofV3(tuple(groups))
    if encode_lean_projection_batch_v3(result) != encoded:
        raise ProofV3Error("lean projection batch wire is not canonical")
    return result


def _fold(values: list[int], challenge: int) -> list[int]:
    half = len(values) // 2
    return [
        (
            values[index]
            + challenge * (values[half + index] - values[index])
        )
        % PALLAS_SCALAR_MODULUS
        for index in range(half)
    ]


def _mle(values: tuple[int, ...], point: tuple[int, ...]) -> int:
    working = list(values)
    for challenge in point:
        working = _fold(working, challenge)
    return working[0]


def _claim_material(
    *,
    claim: LeanProjectionBatchClaimV3,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[bytes, ...],
]:
    statement = claim.operation.statement(
        validator_binding_digest=validator_binding_digest
    )
    coefficients = derive_lean_projection_coefficients_v3(
        statement=statement,
        validator_nonce=validator_nonce,
        input_row_i8=claim.input_row_i8,
        surrogate_output_i64=claim.surrogate_output_i64,
    )
    targets = tuple(
        sum(
            int(value) * int(coefficient)
            for value, coefficient in zip(
                claim.surrogate_output_i64,
                fold_coefficients,
                strict=True,
            )
        )
        % PALLAS_SCALAR_MODULUS
        for fold_coefficients in coefficients
    )
    return coefficients, targets, ()


def _claim_materials_batched(
    *,
    claims: tuple[LeanProjectionBatchClaimV3, ...],
    validator_binding_digest: bytes,
    validator_nonce: bytes,
):
    """Derive coefficient rows and exact targets without catalog MSM work."""

    import numpy as np

    from verallm.proof_v3.lean_projection_fold import (
        _derive_lean_projection_coefficient_words_v3,
    )

    def _exact_targets(output, coefficients):
        """Exact signed-i64 · u31 folds without Python per-cell arithmetic."""

        values = np.asarray(output, dtype=np.int64)
        max_abs = max((abs(int(value)) for value in output), default=0)
        if max_abs == 0:
            return (0,) * LEAN_PROJECTION_FOLD_COUNT_V3
        # Each NumPy dot stays below signed-i64. The two 16-bit limbs are
        # accumulated into Python integers only once per chunk.
        chunk = max(1, min(
            len(values),
            ((1 << 62) - 1) // (max_abs * ((1 << 16) - 1)),
        ))
        targets = []
        for row in coefficients:
            low_total = 0
            high_total = 0
            for offset in range(0, len(values), chunk):
                segment = values[offset : offset + chunk]
                coeffs = row[offset : offset + chunk]
                low_total += int(
                    segment @ (coeffs & np.uint32(0xFFFF)).astype(np.int64)
                )
                high_total += int(
                    segment @ (coeffs >> np.uint32(16)).astype(np.int64)
                )
            targets.append(
                (low_total + (high_total << 16)) % PALLAS_SCALAR_MODULUS
            )
        return tuple(targets)

    base = []
    for claim in claims:
        statement = claim.operation.statement(
            validator_binding_digest=validator_binding_digest
        )
        coefficients = _derive_lean_projection_coefficient_words_v3(
            statement=statement,
            validator_nonce=validator_nonce,
            input_row_i8=claim.input_row_i8,
            surrogate_output_i64=claim.surrogate_output_i64,
        )
        targets = _exact_targets(
            claim.surrogate_output_i64,
            coefficients,
        )
        base.append((coefficients, targets))
    return tuple(
        (
            coefficients,
            targets,
            (),
        )
        for coefficients, targets in base
    )


def _claim_materials_native_verifier(
    *,
    claims: tuple[LeanProjectionBatchClaimV3, ...],
    validator_binding_digest: bytes,
    validator_nonce: bytes,
):
    """Derive verifier-only claim material without retaining coefficient rows.

    The prover still needs coefficient rows for its CUDA weight folds and
    therefore continues to use :func:`_claim_materials_batched`. The verifier
    needs only the four exact targets before the terminal aggregate opening.
    """

    import numpy as np

    from zkllm.crypto.pcs_v2 import (
        PCSUnavailableError,
        derive_lean_projection_claim_targets,
    )

    grouped: dict[
        tuple[bytes, int, int],
        list[int],
    ] = {}
    for index, claim in enumerate(claims):
        statement_digest = claim.operation.statement(
            validator_binding_digest=validator_binding_digest
        ).digest()
        grouped.setdefault(
            (
                statement_digest,
                claim.operation.input_dim,
                claim.operation.output_dim,
            ),
            [],
        ).append(index)

    result = [None] * len(claims)
    try:
        for (
            statement_digest,
            input_dim,
            output_dim,
        ), indices in grouped.items():
            x_rows = np.asarray(
                [claims[index].input_row_i8 for index in indices],
                dtype=np.int8,
            )
            surrogate_rows = np.asarray(
                [claims[index].surrogate_output_i64 for index in indices],
                dtype="<i8",
            )
            if (
                x_rows.shape != (len(indices), input_dim)
                or surrogate_rows.shape != (len(indices), output_dim)
            ):
                raise ProofV3VerificationError(
                    "lean projection native material rows are ragged"
                )
            native = derive_lean_projection_claim_targets(
                statement_digest=statement_digest,
                validator_nonce=validator_nonce,
                input_rows_i8=x_rows.tobytes(order="C"),
                surrogate_rows_i64_le=surrogate_rows.tobytes(order="C"),
                claim_count=len(indices),
                input_dim=input_dim,
                output_dim=output_dim,
            )
            for index, targets in zip(indices, native, strict=True):
                result[index] = ((), targets, ())
    except PCSUnavailableError:
        return _claim_materials_batched(
            claims=claims,
            validator_binding_digest=validator_binding_digest,
            validator_nonce=validator_nonce,
        )
    if any(material is None for material in result):
        raise ProofV3VerificationError(
            "native lean projection material derivation is incomplete"
        )
    return tuple(result)


def _aggregate_expected_catalog_commitment(
    *,
    indexed_claims: tuple[tuple[int, LeanProjectionBatchClaimV3], ...],
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    betas: tuple[int, ...],
) -> bytes:
    """Commit the beta-weighted registered folds without per-claim MSMs."""

    import numpy as np

    from zkllm.crypto.pcs_v2 import (
        PCSUnavailableError,
        combine_commitments,
        combine_registered_catalog_u31_batch,
        combine_registered_lean_projection_aggregate,
    )

    if len(betas) != len(indexed_claims) * LEAN_PROJECTION_FOLD_COUNT_V3:
        raise ProofV3VerificationError(
            "lean projection aggregate beta count is inconsistent"
        )
    grouped: dict[
        tuple[bytes, bytes, int, int],
        list[int],
    ] = {}
    for slot, (_claim_index, claim) in enumerate(indexed_claims):
        statement_digest = claim.operation.statement(
            validator_binding_digest=validator_binding_digest
        ).digest()
        grouped.setdefault(
            (
                statement_digest,
                claim.operation.registered_catalog_id,
                claim.operation.input_dim,
                claim.operation.output_dim,
            ),
            [],
        ).append(slot)

    try:
        operation_commitments = []
        for (
            statement_digest,
            catalog_id,
            input_dim,
            output_dim,
        ), slots in grouped.items():
            selected = tuple(indexed_claims[slot][1] for slot in slots)
            x_rows = np.asarray(
                [claim.input_row_i8 for claim in selected],
                dtype=np.int8,
            )
            surrogate_rows = np.asarray(
                [claim.surrogate_output_i64 for claim in selected],
                dtype="<i8",
            )
            if (
                x_rows.shape != (len(slots), input_dim)
                or surrogate_rows.shape != (len(slots), output_dim)
            ):
                raise ProofV3VerificationError(
                    "lean projection aggregate rows are ragged"
                )
            beta_bytes = b"".join(
                scalar_to_bytes(
                    betas[
                        slot * LEAN_PROJECTION_FOLD_COUNT_V3 + fold_index
                    ]
                )
                for slot in slots
                for fold_index in range(LEAN_PROJECTION_FOLD_COUNT_V3)
            )
            with registered_catalog_operations_v3(
                tuple(claim.operation for claim in selected)
            ):
                operation_commitments.append(
                    combine_registered_lean_projection_aggregate(
                        statement_digest=statement_digest,
                        validator_nonce=validator_nonce,
                        catalog_id=catalog_id,
                        input_rows_i8=x_rows.tobytes(order="C"),
                        surrogate_rows_i64_le=surrogate_rows.tobytes(order="C"),
                        beta_bytes=beta_bytes,
                        claim_count=len(slots),
                        input_dim=input_dim,
                        output_dim=output_dim,
                    )
                )
        return combine_commitments(
            tuple(operation_commitments),
            (1,) * len(operation_commitments),
        )
    except PCSUnavailableError:
        claims = tuple(claim for _claim_index, claim in indexed_claims)
        materials = _claim_materials_batched(
            claims=claims,
            validator_binding_digest=validator_binding_digest,
            validator_nonce=validator_nonce,
        )
        fold_commitments = []
        for claim, (coefficients, _targets, _commitments) in zip(
            claims, materials, strict=True
        ):
            with registered_catalog_operations_v3((claim.operation,)):
                fold_commitments.extend(
                    combine_registered_catalog_u31_batch(
                        claim.operation.registered_catalog_id,
                        coefficients.tobytes(order="C"),
                        term_count=claim.operation.output_dim,
                        fold_count=LEAN_PROJECTION_FOLD_COUNT_V3,
                    )
                )
        return combine_commitments(tuple(fold_commitments), betas)


def _group_transcript(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    width: int,
    indexed_claims: tuple[tuple[int, LeanProjectionBatchClaimV3], ...],
    materials: tuple[
        tuple[tuple[tuple[int, ...], ...], tuple[int, ...], tuple[bytes, ...]],
        ...,
    ],
) -> _Transcript:
    transcript = _Transcript(validator_binding_digest)
    transcript.absorb(b"validator_nonce", validator_nonce)
    transcript.absorb(b"padded_input_dim", struct.pack("<I", width))
    transcript.absorb(b"claim_count", struct.pack("<I", len(indexed_claims)))
    for (claim_index, claim), (_coefficients, targets, _commitments) in zip(
        indexed_claims, materials, strict=True
    ):
        statement = claim.operation.statement(
            validator_binding_digest=validator_binding_digest
        )
        transcript.absorb(b"claim_index", struct.pack("<I", claim_index))
        transcript.absorb(b"statement", statement.digest())
        transcript.absorb(
            b"input_row",
            bytes(value & 0xFF for value in claim.input_row_i8),
        )
        transcript.absorb(
            b"surrogate_output",
            struct.pack(
                f"<{len(claim.surrogate_output_i64)}q",
                *claim.surrogate_output_i64,
            ),
        )
        transcript.absorb(
            b"fold_targets",
            b"".join(scalar_to_bytes(value) for value in targets),
        )
    return transcript


def _group_indices(
    claims: tuple[LeanProjectionBatchClaimV3, ...],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    widths: dict[int, list[int]] = {}
    for index, claim in enumerate(claims):
        widths.setdefault(claim.operation.padded_input_dim, []).append(index)
    claims_per_group = (
        MAX_LEAN_PROJECTION_VECTORS // LEAN_PROJECTION_FOLD_COUNT_V3
    )
    return tuple(
        (width, tuple(indices[offset : offset + claims_per_group]))
        for width in sorted(widths)
        for indices in (widths[width],)
        for offset in range(0, len(indices), claims_per_group)
    )


def _build_native_group_proof_v3(
    *,
    width: int,
    claim_indices: tuple[int, ...],
    indexed_claims: tuple[tuple[int, LeanProjectionBatchClaimV3], ...],
    folds: tuple[LeanProjectionFoldV3, ...],
    fold_targets: tuple[tuple[int, ...], ...],
    alphas: tuple[int, ...],
    transcript: _Transcript,
) -> LeanProjectionBatchGroupProofV3:
    if sys.byteorder != "little":
        raise ProofV3Error(
            "native lean projection aggregate requires a little-endian host"
        )
    from zkllm.crypto.pcs_v2 import (
        ENCODING_PALLAS_SCALAR,
        prove_lean_projection_group_deduplicated,
    )

    unique_x_raw = bytearray()
    unique_z_raw = array("q")
    x_indices: dict[bytes, int] = {}
    z_indices: dict[int, int] = {}
    x_vector_map = []
    z_vector_map = []
    targets = []
    for (claim_index, claim), targets_for_claim in zip(
        indexed_claims,
        fold_targets,
        strict=True,
    ):
        padded_x = claim.input_row_i8 + (0,) * (
            width - claim.operation.input_dim
        )
        packed_x = bytes(value & 0xFF for value in padded_x)
        x_index = x_indices.get(packed_x)
        if x_index is None:
            x_index = len(x_indices)
            x_indices[packed_x] = x_index
            unique_x_raw.extend(packed_x)
        fold = folds[claim_index]
        for folded_weight, target in zip(
            fold.folded_weights,
            targets_for_claim,
            strict=True,
        ):
            z_key = id(folded_weight)
            z_index = z_indices.get(z_key)
            if z_index is None:
                z_index = len(z_indices)
                z_indices[z_key] = z_index
                unique_z_raw.extend(folded_weight)
            x_vector_map.append(x_index)
            z_vector_map.append(z_index)
            targets.append(target)
    vector_count = len(targets)
    native = prove_lean_projection_group_deduplicated(
        transcript_state=transcript.state,
        unique_x_values_i8=bytes(unique_x_raw),
        x_vector_map=tuple(x_vector_map),
        unique_z_values_i64_le=unique_z_raw.tobytes(),
        z_vector_map=tuple(z_vector_map),
        alpha_bytes=b"".join(scalar_to_bytes(value) for value in alphas),
        target_bytes=b"".join(
            scalar_to_bytes(value) for value in targets
        ),
        vector_count=vector_count,
        width=width,
    )
    header = struct.Struct("<4sHIIHH")
    if len(native) < header.size:
        raise ProofV3Error("native lean projection proof is truncated")
    (
        magic,
        version,
        native_width,
        native_vectors,
        round_count,
        ipa_proof_len,
    ) = header.unpack_from(native)
    if (
        magic != b"VLPN"
        or version != 1
        or native_width != width
        or native_vectors != vector_count
        or round_count != width.bit_length() - 1
    ):
        raise ProofV3Error("native lean projection proof header is inconsistent")
    cursor = header.size

    def read_scalar() -> int:
        nonlocal cursor
        end = cursor + 32
        if end > len(native):
            raise ProofV3Error("native lean projection proof is truncated")
        value = scalar_from_bytes(native[cursor:end])
        cursor = end
        return value

    rounds = tuple(
        (read_scalar(), read_scalar(), read_scalar())
        for _ in range(round_count)
    )
    terminal = tuple(read_scalar() for _ in range(vector_count))
    commitment_end = cursor + 32
    evaluation_end = commitment_end + 32
    proof_end = evaluation_end + ipa_proof_len
    if proof_end != len(native):
        raise ProofV3Error("native lean projection proof length is inconsistent")
    opening = PCSOpeningV2(
        commitment=native[cursor:commitment_end],
        evaluation=native[commitment_end:evaluation_end],
        proof=native[evaluation_end:proof_end],
        vector_length=width,
        padded_length=width,
        encoding=ENCODING_PALLAS_SCALAR,
    )
    return LeanProjectionBatchGroupProofV3(
        padded_input_dim=width,
        claim_indices=claim_indices,
        rounds=rounds,
        terminal_fold_evaluations=terminal,
        opening=opening,
    )


def build_lean_projection_batch_from_folds_v3(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    claims: tuple[LeanProjectionBatchClaimV3, ...],
    folds: tuple[LeanProjectionFoldV3, ...],
    native_aggregate: bool = False,
    precomputed_materials=None,
) -> LeanProjectionBatchProofV3:
    """Build the compact proof from exact native or reference weight folds."""

    binding = _fixed32(
        validator_binding_digest,
        "lean projection batch validator binding",
        nonzero=True,
    )
    nonce = _fixed32(
        validator_nonce,
        "lean projection batch validator nonce",
        nonzero=True,
    )
    claims = tuple(claims)
    if (
        not claims
        or len(claims) > MAX_LEAN_PROJECTION_BATCH_CLAIMS_V3
        or not all(isinstance(claim, LeanProjectionBatchClaimV3)
                   for claim in claims)
    ):
        raise ProofV3Error("lean projection batch claim set is malformed")
    folds = tuple(folds)
    if (
        len(folds) != len(claims)
        or not all(isinstance(fold, LeanProjectionFoldV3) for fold in folds)
        or any(
            any(
                len(row) != claim.operation.padded_input_dim
                for row in fold.folded_weights
            )
            for claim, fold in zip(claims, folds, strict=True)
        )
    ):
        raise ProofV3Error("lean projection batch folded witness set is wrong")

    groups = []
    for width, claim_indices in _group_indices(claims):
        indexed_claims = tuple((index, claims[index]) for index in claim_indices)
        if precomputed_materials is None:
            materials = tuple(
                _claim_material(
                    claim=claim,
                    validator_binding_digest=binding,
                    validator_nonce=nonce,
                )
                for _index, claim in indexed_claims
            )
        else:
            try:
                materials = tuple(
                    precomputed_materials[index] for index in claim_indices
                )
            except (IndexError, TypeError) as exc:
                raise ProofV3Error(
                    "precomputed lean projection material is incomplete"
                ) from exc
        transcript = _group_transcript(
            validator_binding_digest=binding,
            validator_nonce=nonce,
            width=width,
            indexed_claims=indexed_claims,
            materials=materials,
        )

        targets = [
            target
            for _coefficients, fold_targets, _commitments in materials
            for target in fold_targets
        ]

        alphas = tuple(
            transcript.scalar(b"claim_alpha" + struct.pack("<I", index))
            for index in range(len(targets))
        )
        if native_aggregate:
            groups.append(
                _build_native_group_proof_v3(
                    width=width,
                    claim_indices=claim_indices,
                    indexed_claims=indexed_claims,
                    folds=folds,
                    fold_targets=tuple(
                        fold_targets for _coefficients, fold_targets, _commitments
                        in materials
                    ),
                    alphas=alphas,
                    transcript=transcript,
                )
            )
            continue
        x_vectors: list[list[int]] = []
        z_vectors: list[list[int]] = []
        for (claim_index, claim), (
            _coefficients,
            fold_targets,
            _commitments,
        ) in zip(indexed_claims, materials, strict=True):
            fold = folds[claim_index]
            x = [
                value % PALLAS_SCALAR_MODULUS
                for value in (
                    claim.input_row_i8
                    + (0,) * (width - claim.operation.input_dim)
                )
            ]
            for folded_weight, _target in zip(
                fold.folded_weights,
                fold_targets,
                strict=True,
            ):
                x_vectors.append(list(x))
                z_vectors.append(
                    [value % PALLAS_SCALAR_MODULUS for value in folded_weight]
                )
        running = sum(
            alpha * target
            for alpha, target in zip(alphas, targets, strict=True)
        ) % PALLAS_SCALAR_MODULUS
        rounds = []
        point = []
        while len(x_vectors[0]) > 1:
            half = len(x_vectors[0]) // 2
            g0 = g1 = g2 = 0
            for alpha, x, z in zip(alphas, x_vectors, z_vectors, strict=True):
                for offset in range(half):
                    x0, x1 = x[offset], x[half + offset]
                    z0, z1 = z[offset], z[half + offset]
                    g0 = (g0 + alpha * x0 * z0) % PALLAS_SCALAR_MODULUS
                    g1 = (g1 + alpha * x1 * z1) % PALLAS_SCALAR_MODULUS
                    g2 = (
                        g2
                        + alpha
                        * ((2 * x1 - x0) % PALLAS_SCALAR_MODULUS)
                        * ((2 * z1 - z0) % PALLAS_SCALAR_MODULUS)
                    ) % PALLAS_SCALAR_MODULUS
            if (g0 + g1) % PALLAS_SCALAR_MODULUS != running:
                raise ProofV3Error(
                    "lean projection witness does not satisfy the full-row relation"
                )
            rounds.append((g0, g1, g2))
            transcript.absorb(
                b"sumcheck_round",
                b"".join(scalar_to_bytes(value) for value in (g0, g1, g2)),
            )
            challenge = transcript.scalar(
                b"inner_challenge" + struct.pack("<I", len(rounds) - 1)
            )
            point.append(challenge)
            x_vectors = [_fold(vector, challenge) for vector in x_vectors]
            z_vectors = [_fold(vector, challenge) for vector in z_vectors]
            running = sum(
                alpha
                * sum(
                    x_value * z_value
                    for x_value, z_value in zip(x, z, strict=True)
                )
                for alpha, x, z in zip(
                    alphas, x_vectors, z_vectors, strict=True
                )
            ) % PALLAS_SCALAR_MODULUS

        terminal_z = tuple(vector[0] for vector in z_vectors)
        transcript.absorb(
            b"terminal_fold_evaluations",
            b"".join(scalar_to_bytes(value) for value in terminal_z),
        )
        betas = tuple(
            transcript.scalar(b"opening_beta" + struct.pack("<I", index))
            for index in range(len(z_vectors))
        )

        original_z = [
            tuple(value % PALLAS_SCALAR_MODULUS for value in row)
            for claim_index in claim_indices
            for row in folds[claim_index].folded_weights
        ]
        combined = tuple(
            sum(
                beta * vector[index]
                for beta, vector in zip(betas, original_z, strict=True)
            )
            % PALLAS_SCALAR_MODULUS
            for index in range(width)
        )
        opening_outer_digest = hashlib.sha256(
            _TRANSCRIPT_DOMAIN + b"/OPENING/" + transcript.state
        ).digest()
        from zkllm.crypto.pcs_v2 import prove

        opening = prove(
            combined,
            tuple(point),
            opening_outer_digest,
            encoding="field",
        )
        expected_evaluation = sum(
            beta * value
            for beta, value in zip(betas, terminal_z, strict=True)
        ) % PALLAS_SCALAR_MODULUS
        if scalar_from_bytes(opening.evaluation) != expected_evaluation:
            raise ProofV3Error(
                "lean projection batched opening evaluation is inconsistent"
            )
        groups.append(
            LeanProjectionBatchGroupProofV3(
                padded_input_dim=width,
                claim_indices=claim_indices,
                rounds=tuple(rounds),
                terminal_fold_evaluations=terminal_z,
                opening=opening,
            )
        )
    return LeanProjectionBatchProofV3(tuple(groups))


def build_lean_projection_batch_reference_v3(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    claims: tuple[LeanProjectionBatchClaimV3, ...],
    weight_columns_i8: tuple[tuple[tuple[int, ...], ...], ...],
) -> LeanProjectionBatchProofV3:
    """Build exact folds portably, then use the canonical compact prover."""

    claims = tuple(claims)
    witnesses = tuple(weight_columns_i8)
    if len(witnesses) != len(claims):
        raise ProofV3Error("lean projection batch witness count is wrong")
    folds = tuple(
        build_lean_projection_fold_reference_v3(
            statement=claim.operation.statement(
                validator_binding_digest=validator_binding_digest
            ),
            validator_nonce=validator_nonce,
            input_row_i8=claim.input_row_i8,
            surrogate_output_i64=claim.surrogate_output_i64,
            weight_columns_i8=witness,
        )
        for claim, witness in zip(claims, witnesses, strict=True)
    )
    return build_lean_projection_batch_from_folds_v3(
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
        claims=claims,
        folds=folds,
    )


def verify_lean_projection_batch_v3(
    *,
    proof: LeanProjectionBatchProofV3,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    claims: tuple[LeanProjectionBatchClaimV3, ...],
) -> None:
    """Verify every complete selected-row projection, fail closed."""

    import os
    import time

    profile_timing = (
        os.environ.get("VERATHOS_PROOF_V3_PROFILE", "") == "1"
    )
    verify_started = time.perf_counter()
    try:
        if not isinstance(proof, LeanProjectionBatchProofV3):
            raise ProofV3VerificationError(
                "lean projection batch proof has a wrong type"
            )
        binding = _fixed32(
            validator_binding_digest,
            "lean projection batch validator binding",
            nonzero=True,
        )
        nonce = _fixed32(
            validator_nonce,
            "lean projection batch validator nonce",
            nonzero=True,
        )
        claims = tuple(claims)
        if (
            not claims
            or len(claims) > MAX_LEAN_PROJECTION_BATCH_CLAIMS_V3
            or not all(isinstance(claim, LeanProjectionBatchClaimV3)
                       for claim in claims)
        ):
            raise ProofV3VerificationError(
                "lean projection batch claim set is malformed"
            )
        expected_groups = _group_indices(claims)
        if tuple(
            (group.padded_input_dim, group.claim_indices)
            for group in proof.groups
        ) != expected_groups:
            raise ProofV3VerificationError(
                "lean projection batch groups do not exactly cover the claims"
            )

        from zkllm.crypto.pcs_v2 import (
            ENCODING_PALLAS_SCALAR,
            verify,
        )

        materials_started = time.perf_counter()
        materials_by_claim = _claim_materials_native_verifier(
            claims=claims,
            validator_binding_digest=binding,
            validator_nonce=nonce,
        )
        materials_finished = time.perf_counter()
        groups_started = materials_finished
        group_timings = []
        for group_index, group in enumerate(proof.groups):
            group_started = time.perf_counter()
            indexed_claims = tuple(
                (index, claims[index]) for index in group.claim_indices
            )
            materials = tuple(
                materials_by_claim[index]
                for index, _claim in indexed_claims
            )
            transcript = _group_transcript(
                validator_binding_digest=binding,
                validator_nonce=nonce,
                width=group.padded_input_dim,
                indexed_claims=indexed_claims,
                materials=materials,
            )
            targets = tuple(
                target
                for _coefficients, fold_targets, _commitments in materials
                for target in fold_targets
            )
            alphas = tuple(
                transcript.scalar(b"claim_alpha" + struct.pack("<I", index))
                for index in range(len(targets))
            )
            running = sum(
                alpha * target
                for alpha, target in zip(alphas, targets, strict=True)
            ) % PALLAS_SCALAR_MODULUS
            point = []
            for round_index, (g0, g1, g2) in enumerate(group.rounds):
                if (g0 + g1) % PALLAS_SCALAR_MODULUS != running:
                    raise ProofV3VerificationError(
                        "lean projection sumcheck round does not match"
                    )
                transcript.absorb(
                    b"sumcheck_round",
                    b"".join(
                        scalar_to_bytes(value) for value in (g0, g1, g2)
                    ),
                )
                challenge = transcript.scalar(
                    b"inner_challenge" + struct.pack("<I", round_index)
                )
                point.append(challenge)
                inv2 = (PALLAS_SCALAR_MODULUS + 1) // 2
                running = (
                    g0
                    * ((challenge - 1) * (challenge - 2)
                       % PALLAS_SCALAR_MODULUS)
                    % PALLAS_SCALAR_MODULUS
                    * inv2
                    - g1
                    * (challenge * (challenge - 2)
                       % PALLAS_SCALAR_MODULUS)
                    + g2
                    * (challenge * (challenge - 1)
                       % PALLAS_SCALAR_MODULUS)
                    % PALLAS_SCALAR_MODULUS
                    * inv2
                ) % PALLAS_SCALAR_MODULUS

            relation_started = time.perf_counter()
            try:
                import numpy as np

                from zkllm.crypto.pcs_v2 import (
                    PCSUnavailableError,
                    evaluate_lean_projection_terminal_relation,
                )

                x_matrix = np.zeros(
                    (len(indexed_claims), group.padded_input_dim),
                    dtype=np.int8,
                )
                for row_index, (_claim_index, claim) in enumerate(
                    indexed_claims
                ):
                    x_matrix[
                        row_index, :claim.operation.input_dim
                    ] = claim.input_row_i8
                terminal_relation = (
                    evaluate_lean_projection_terminal_relation(
                        x_matrix.tobytes(order="C"),
                        claim_count=len(indexed_claims),
                        width=group.padded_input_dim,
                        point=point,
                        alphas=alphas,
                        terminal_fold_evaluations=(
                            group.terminal_fold_evaluations
                        ),
                    )
                )
            except PCSUnavailableError:
                x_evaluations = []
                for _claim_index, claim in indexed_claims:
                    x = tuple(
                        value % PALLAS_SCALAR_MODULUS
                        for value in (
                            claim.input_row_i8
                            + (0,) * (
                                group.padded_input_dim
                                - claim.operation.input_dim
                            )
                        )
                    )
                    evaluation = _mle(x, tuple(point))
                    x_evaluations.extend(
                        (evaluation,) * LEAN_PROJECTION_FOLD_COUNT_V3
                    )
                terminal_relation = sum(
                    alpha * x_value * z_value
                    for alpha, x_value, z_value in zip(
                        alphas,
                        x_evaluations,
                        group.terminal_fold_evaluations,
                        strict=True,
                    )
                ) % PALLAS_SCALAR_MODULUS
            if running != terminal_relation:
                raise ProofV3VerificationError(
                    "lean projection sumcheck terminal relation does not match"
                )

            transcript.absorb(
                b"terminal_fold_evaluations",
                b"".join(
                    scalar_to_bytes(value)
                    for value in group.terminal_fold_evaluations
                ),
            )
            betas = tuple(
                transcript.scalar(b"opening_beta" + struct.pack("<I", index))
                for index in range(len(group.terminal_fold_evaluations))
            )
            relation_finished = time.perf_counter()
            expected_commitment = _aggregate_expected_catalog_commitment(
                indexed_claims=indexed_claims,
                validator_binding_digest=binding,
                validator_nonce=nonce,
                betas=betas,
            )
            aggregate_finished = time.perf_counter()
            if group.opening.commitment != expected_commitment:
                raise ProofV3VerificationError(
                    "lean projection opening is not the authenticated catalog "
                    f"fold (group={group_index}, width={group.padded_input_dim}, "
                    f"claims={group.claim_indices[0]}.."
                    f"{group.claim_indices[-1]})"
                )
            if (
                group.opening.vector_length != group.padded_input_dim
                or group.opening.padded_length != group.padded_input_dim
                or group.opening.encoding != ENCODING_PALLAS_SCALAR
            ):
                raise ProofV3VerificationError(
                    "lean projection opening shape or encoding is wrong"
                )
            expected_evaluation = sum(
                beta * value
                for beta, value in zip(
                    betas,
                    group.terminal_fold_evaluations,
                    strict=True,
                )
            ) % PALLAS_SCALAR_MODULUS
            if scalar_from_bytes(group.opening.evaluation) != expected_evaluation:
                raise ProofV3VerificationError(
                    "lean projection opening evaluation does not match"
                )
            opening_outer_digest = hashlib.sha256(
                _TRANSCRIPT_DOMAIN + b"/OPENING/" + transcript.state
            ).digest()
            pcs_started = time.perf_counter()
            if not verify(
                group.opening,
                tuple(point),
                opening_outer_digest,
            ):
                raise ProofV3VerificationError(
                    "lean projection PCS opening is invalid"
                )
            pcs_finished = time.perf_counter()
            group_timings.append(
                (
                    group_index,
                    group.padded_input_dim,
                    len(group.claim_indices),
                    relation_started - group_started,
                    relation_finished - relation_started,
                    aggregate_finished - relation_finished,
                    pcs_finished - pcs_started,
                    pcs_finished - group_started,
                )
            )
        if profile_timing:
            finished = time.perf_counter()
            print(
                "[PROOF-V3-HARD-VERIFY] "
                f"projection_materials="
                f"{materials_finished - materials_started:.3f}s "
                f"projection_groups={finished - groups_started:.3f}s "
                f"projection_total={finished - verify_started:.3f}s "
                f"claims={len(claims)} groups={len(proof.groups)}",
                flush=True,
            )
            print(
                "[PROOF-V3-HARD-VERIFY] projection_group_detail "
                + " ".join(
                    f"g{index}[width={width},claims={claim_count}]="
                    f"prefix:{prefix:.3f}s/"
                    f"relation:{relation:.3f}s/"
                    f"catalog:{catalog:.3f}s/"
                    f"pcs:{pcs:.3f}s/"
                    f"total:{total:.3f}s"
                    for (
                        index,
                        width,
                        claim_count,
                        prefix,
                        relation,
                        catalog,
                        pcs,
                        total,
                    ) in group_timings
                ),
                flush=True,
            )
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, OverflowError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "lean projection batch proof is malformed"
        ) from exc
