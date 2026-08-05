"""Nonce-sampled projection checks for proof-v3 hard audits.

The compact path authenticates the exact validator-derived output coordinates
of ``S = int8(X) @ int8(W)``.  The post-nonce surrogate Merkle root freezes the
selected values before four transcript coefficients are derived.  One shared
Merkle multiproof opens those values, while the registered Pallas catalog
authenticates the corresponding sparse static-weight folds.  No dynamic
activation is committed with Pallas.

This is the deliberate probabilistic projection lane.  The existing
complete-row proof remains an independently selected escalation.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.economic_commitment import (
    EconomicCommittedOracleV3,
    oracle_leaf_index_v3,
    verify_economic_oracle_opening_v3,
)
from verallm.proof_v3.economic_wire import (
    EconomicMerkleOpeningV3,
    EconomicOracleCommitmentV3,
    _Reader,
    _Writer,
    bounded_byte_width_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.lean_projection_batch import (
    LeanProjectionBatchClaimV3,
    LeanProjectionBatchGroupProofV3,
    LeanProjectionBatchProofV3,
    _TRANSCRIPT_DOMAIN as _LEAN_TRANSCRIPT_DOMAIN,
    _Transcript,
    _build_native_group_proof_v3,
    _group_indices,
    _mle,
    decode_lean_projection_batch_v3,
    encode_lean_projection_batch_v3,
)
from verallm.proof_v3.lean_projection_fold import (
    LEAN_PROJECTION_FOLD_COUNT_V3,
    LeanProjectionCatalogOperationV3,
    LeanProjectionFoldV3,
    registered_catalog_operations_v3,
)
from zkllm.crypto.gemm_v2_reference import (
    PALLAS_SCALAR_MODULUS,
    scalar_from_bytes,
    scalar_to_bytes,
)
from zkllm.crypto.pcs_v2 import (
    ENCODING_PALLAS_SCALAR,
    MAX_CATALOG_FOLDS,
    MAX_COMBINE_TERMS,
    MAX_LEAN_PROJECTION_VECTORS,
    PCSOpeningV2,
    PCSNativeError,
    combine_commitments,
    combine_registered_catalog_u31_batch,
    verify,
)


SUCCINCT_PROJECTION_BATCH_ABI_V3: Final = (
    "projection.sampled_output.merkle_catalog_sumcheck.v5"
)
# Four selected Qwen3.6 GDN value heads can require 2,048 exact Q/K/V/Z
# runtime coordinates. The projection statement additionally carries the
# 16 transcript-selected generic output coordinates, so 2,048 was not a
# valid production bound. Keep a fixed decoder/allocation ceiling above the
# qualified union; wider adapters fail closed and require qualification
# rather than silently widening this limit.
MAX_SAMPLED_OUTPUTS_PER_CLAIM_V3: Final = 4096
# Four selected checkpointed GDN corridors can exceed 512 row/projection
# claims once their nonce-selected transition rows are unioned with the
# 32-row decode checkpoint window. The shipped 27B hybrid profile produces
# 550 claims for a representative two-attention/two-GDN selection; 1,024
# bounds every four-layer stride-32 production selection with margin while
# remaining a small, explicit decoder allocation cap.
MAX_SUCCINCT_PROJECTION_CLAIMS_V3: Final = 1024
# This bounds the canonical *uncompressed subsection*. The enclosing hard
# bundle is compressed once after assembly. Production GDN/QKV provenance at
# the qualified four-row policy measures about 1.57 MiB before that outer
# compression, so keep the parser bound at the operator-approved 1--2 MiB
# range while the final network bundle remains gated against the matched v9
# measurement.
MAX_SUCCINCT_PROJECTION_WIRE_BYTES_V3: Final = 2 << 20

_COEFFICIENT_DOMAIN = (
    b"VERATHOS/PROOF_V3/SUCCINCT_PROJECTION/COEFFICIENTS/U31X4/V2"
)
_WIRE_MAGIC = b"VSPB"
_WIRE_VERSION = 5

__all__ = [
    "MAX_SUCCINCT_PROJECTION_WIRE_BYTES_V3",
    "MAX_SAMPLED_OUTPUTS_PER_CLAIM_V3",
    "SUCCINCT_PROJECTION_BATCH_ABI_V3",
    "SuccinctProjectionBatchProofV3",
    "SuccinctProjectionClaimProofV3",
    "SuccinctProjectionClaimV3",
    "SuccinctProjectionDynamicGroupProofV3",
    "SuccinctProjectionWeightRowsV3",
    "SuccinctProjectionWitnessV3",
    "build_succinct_projection_batch_from_folds_v3",
    "build_succinct_projection_batch_reference_v3",
    "decode_succinct_projection_batch_v3",
    "derive_succinct_projection_coefficients_v3",
    "encode_succinct_projection_batch_v3",
    "verify_succinct_projection_batch_v3",
]


def _combine_commitments_bounded_v3(
    commitments: tuple[bytes, ...],
    coefficients: tuple[int, ...],
) -> bytes:
    """Combine a native-vector-sized inventory through bounded MSM calls."""

    if not commitments or len(commitments) != len(coefficients):
        raise ProofV3VerificationError(
            "succinct projection commitment inventory is malformed"
        )
    if len(commitments) <= MAX_COMBINE_TERMS:
        return combine_commitments(commitments, coefficients)
    partials = tuple(
        combine_commitments(
            commitments[offset : offset + MAX_COMBINE_TERMS],
            coefficients[offset : offset + MAX_COMBINE_TERMS],
        )
        for offset in range(0, len(commitments), MAX_COMBINE_TERMS)
    )
    return combine_commitments(partials, (1,) * len(partials))


def _fixed32(value: object, name: str, *, nonzero: bool = False) -> bytes:
    if (
        not isinstance(value, bytes)
        or len(value) != 32
        or (nonzero and not any(value))
    ):
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    return value


def _signed_row(
    values: object,
    *,
    length: int,
    bits: int,
    name: str,
) -> tuple[int, ...]:
    if isinstance(values, tuple) and all(type(value) is int for value in values):
        row = values
    else:
        try:
            row = tuple(int(value) for value in values)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ProofV3Error(f"{name} is malformed") from exc
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if (
        len(row) != length
        or (row and (min(row) < minimum or max(row) > maximum))
    ):
        raise ProofV3Error(f"{name} is not a canonical signed-i{bits} row")
    return row


@dataclass(frozen=True, slots=True)
class SuccinctProjectionClaimV3:
    """Validator-owned statement for one selected projection row."""

    operation: LeanProjectionCatalogOperationV3
    input_row_i8: tuple[int, ...]
    surrogate_oracle: EconomicOracleCommitmentV3
    row_index: int
    output_columns: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operation, LeanProjectionCatalogOperationV3):
            raise ProofV3Error("succinct projection operation is malformed")
        object.__setattr__(
            self,
            "input_row_i8",
            _signed_row(
                self.input_row_i8,
                length=self.operation.input_dim,
                bits=8,
                name="succinct projection input",
            ),
        )
        if not isinstance(self.surrogate_oracle, EconomicOracleCommitmentV3):
            raise ProofV3Error("succinct projection oracle is malformed")
        if (
            self.surrogate_oracle.col_count != self.operation.output_dim
            or not self.surrogate_oracle.operation.endswith("_s")
        ):
            raise ProofV3Error(
                "succinct projection oracle does not match the operation"
            )
        if (
            isinstance(self.row_index, bool)
            or not isinstance(self.row_index, int)
            or not 0 <= self.row_index < self.surrogate_oracle.row_count
        ):
            raise ProofV3Error("succinct projection row index is out of range")
        columns = tuple(self.output_columns)
        if (
            not columns
            or len(columns) > MAX_SAMPLED_OUTPUTS_PER_CLAIM_V3
            or columns != tuple(sorted(set(columns)))
            or any(
                isinstance(column, bool)
                or not isinstance(column, int)
                or column < 0
                or column >= self.operation.output_dim
                for column in columns
            )
        ):
            raise ProofV3Error(
                "succinct projection output columns are malformed"
            )
        object.__setattr__(self, "output_columns", columns)


@dataclass(frozen=True, slots=True)
class SuccinctProjectionWitnessV3:
    claim: SuccinctProjectionClaimV3
    surrogate_output_i64: tuple[int, ...]
    committed_surrogate: EconomicCommittedOracleV3

    def __post_init__(self) -> None:
        if not isinstance(self.claim, SuccinctProjectionClaimV3):
            raise ProofV3Error("succinct projection witness claim is malformed")
        object.__setattr__(
            self,
            "surrogate_output_i64",
            _signed_row(
                self.surrogate_output_i64,
                length=self.claim.operation.output_dim,
                bits=64,
                name="succinct projection surrogate",
            ),
        )
        if (
            not isinstance(
                self.committed_surrogate,
                EconomicCommittedOracleV3,
            )
            or self.committed_surrogate.commitment
            != self.claim.surrogate_oracle
        ):
            raise ProofV3Error(
                "succinct projection committed surrogate is inconsistent"
            )


@dataclass(frozen=True, slots=True)
class SuccinctProjectionWeightRowsV3:
    """Ephemeral canonical rows for the exact sampled output inventory."""

    output_columns: tuple[int, ...]
    rows_i8: object
    input_dim: int
    output_dim: int

    def __post_init__(self) -> None:
        columns = tuple(self.output_columns)
        shape = getattr(self.rows_i8, "shape", None)
        if (
            not columns
            or columns != tuple(sorted(set(columns)))
            or isinstance(self.input_dim, bool)
            or not isinstance(self.input_dim, int)
            or self.input_dim <= 0
            or isinstance(self.output_dim, bool)
            or not isinstance(self.output_dim, int)
            or self.output_dim <= 0
            or any(column < 0 or column >= self.output_dim for column in columns)
            or shape is None
            or tuple(int(value) for value in shape)
            != (len(columns), self.input_dim)
        ):
            raise ProofV3Error(
                "succinct projection sampled weight rows are malformed"
            )
        object.__setattr__(self, "output_columns", columns)


@dataclass(frozen=True, slots=True)
class SuccinctProjectionClaimProofV3:
    surrogate_commitment: bytes
    fold_targets: tuple[int, ...]

    def __post_init__(self) -> None:
        _fixed32(
            self.surrogate_commitment,
            "succinct surrogate commitment",
            nonzero=True,
        )
        targets = tuple(self.fold_targets)
        if (
            len(targets) != LEAN_PROJECTION_FOLD_COUNT_V3
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < PALLAS_SCALAR_MODULUS
                for value in targets
            )
        ):
            raise ProofV3Error("succinct projection fold targets are malformed")
        object.__setattr__(self, "fold_targets", targets)


@dataclass(frozen=True, slots=True)
class SuccinctProjectionDynamicGroupProofV3:
    claim_indices: tuple[int, ...]
    capture_opening: EconomicMerkleOpeningV3

    def __post_init__(self) -> None:
        indices = tuple(self.claim_indices)
        if (
            not indices
            or indices != tuple(sorted(set(indices)))
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= MAX_SUCCINCT_PROJECTION_CLAIMS_V3
                for index in indices
            )
        ):
            raise ProofV3Error(
                "succinct projection dynamic group indices are malformed"
            )
        if not isinstance(self.capture_opening, EconomicMerkleOpeningV3):
            raise ProofV3Error(
                "succinct projection capture opening is malformed"
            )
        object.__setattr__(self, "claim_indices", indices)


@dataclass(frozen=True, slots=True)
class SuccinctProjectionBatchProofV3:
    claims: tuple[SuccinctProjectionClaimProofV3, ...]
    dynamic_groups: tuple[SuccinctProjectionDynamicGroupProofV3, ...]
    groups: LeanProjectionBatchProofV3

    def __post_init__(self) -> None:
        claims = tuple(self.claims)
        if (
            not claims
            or len(claims) > MAX_SUCCINCT_PROJECTION_CLAIMS_V3
            or not all(
                isinstance(claim, SuccinctProjectionClaimProofV3)
                for claim in claims
            )
            or not self.dynamic_groups
            or not all(
                isinstance(group, SuccinctProjectionDynamicGroupProofV3)
                for group in self.dynamic_groups
            )
            or not isinstance(self.groups, LeanProjectionBatchProofV3)
        ):
            raise ProofV3Error("succinct projection batch is malformed")
        object.__setattr__(self, "claims", claims)
        object.__setattr__(
            self,
            "dynamic_groups",
            tuple(self.dynamic_groups),
        )


def _sample_columns(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    claim: SuccinctProjectionClaimV3,
    surrogate_commitment: bytes,
) -> tuple[int, ...]:
    _fixed32(
        validator_binding_digest,
        "succinct projection validator binding",
        nonzero=True,
    )
    _fixed32(
        validator_nonce,
        "succinct projection validator nonce",
        nonzero=True,
    )
    commitment = _fixed32(
        surrogate_commitment,
        "succinct projection surrogate commitment",
        nonzero=True,
    )
    if (
        not isinstance(claim, SuccinctProjectionClaimV3)
        or commitment != claim.surrogate_oracle.root
    ):
        raise ProofV3Error(
            "succinct projection surrogate root is inconsistent"
        )
    return claim.output_columns


def derive_succinct_projection_coefficients_v3(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    claim: SuccinctProjectionClaimV3,
    surrogate_commitment: bytes,
):
    """Derive four operation-shared sparse folds after capture is frozen.

    Every claim for one registered operation opens the same validator-selected
    output-coordinate inventory. Sharing the post-nonce coefficients preserves
    the four-fold cancellation bound while allowing the static folded weight
    witness to be built once per operation instead of once per runtime row.
    """

    binding = _fixed32(
        validator_binding_digest,
        "succinct projection validator binding",
        nonzero=True,
    )
    nonce = _fixed32(
        validator_nonce,
        "succinct projection validator nonce",
        nonzero=True,
    )
    commitment = _fixed32(
        surrogate_commitment,
        "succinct projection surrogate commitment",
        nonzero=True,
    )
    if not isinstance(claim, SuccinctProjectionClaimV3):
        raise ProofV3Error("succinct projection claim is malformed")
    statement = claim.operation.statement(validator_binding_digest=binding)
    seed = (
        _COEFFICIENT_DOMAIN
        + statement.digest()
        + nonce
        + claim.surrogate_oracle.root
        + commitment
        + struct.pack("<I", len(claim.output_columns))
        + b"".join(
            struct.pack("<I", column)
            for column in claim.output_columns
        )
    )
    count = LEAN_PROJECTION_FOLD_COUNT_V3 * len(claim.output_columns)
    raw = hashlib.shake_256(seed).digest(count * 4)
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - production dependency.
        raise ProofV3Error(
            "succinct projection coefficients require NumPy"
        ) from exc
    selected = (
        np.frombuffer(raw, dtype="<u4")
        .reshape(LEAN_PROJECTION_FOLD_COUNT_V3, len(claim.output_columns))
        .__and__(np.uint32((1 << 31) - 1))
        .astype("<u4", copy=False)
    )
    coefficients = np.zeros(
        (
            LEAN_PROJECTION_FOLD_COUNT_V3,
            claim.operation.output_dim,
        ),
        dtype="<u4",
    )
    coefficients[:, np.asarray(claim.output_columns, dtype=np.int64)] = (
        selected
    )
    return coefficients


def _exact_targets(
    sample_values: tuple[int, ...],
    sample_columns: tuple[int, ...],
    coefficients,
) -> tuple[int, ...]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - production dependency.
        raise ProofV3Error(
            "succinct projection target verification requires NumPy"
        ) from exc
    values = np.fromiter(
        sample_values,
        dtype=object,
        count=len(sample_values),
    )
    columns = np.fromiter(
        sample_columns,
        dtype=np.int64,
        count=len(sample_columns),
    )
    # Fold targets live in the Pallas scalar field. Runtime surrogate values
    # times u31 coefficients can exceed signed i64 at production widths, so
    # retain Python's exact integers while moving the loop into NumPy's C
    # object-array iterator.
    selected = coefficients[:, columns].astype(object)
    products = selected @ values
    return tuple(
        int(value) % PALLAS_SCALAR_MODULUS for value in products
    )


def _capture_blob(opening: EconomicMerkleOpeningV3) -> bytes:
    writer = _Writer()
    opening.encode(writer)
    return writer.finish()


def _decode_capture_blob(encoded: bytes) -> EconomicMerkleOpeningV3:
    reader = _Reader(encoded, "succinct projection capture opening")
    opening = EconomicMerkleOpeningV3.decode(reader)
    reader.finish()
    if _capture_blob(opening) != encoded:
        raise ProofV3Error(
            "succinct projection capture opening is not canonical"
        )
    return opening


def _claim_transcript_record(
    *,
    claim_index: int,
    claim: SuccinctProjectionClaimV3,
    proof: SuccinctProjectionClaimProofV3,
    sample_columns: tuple[int, ...],
    sample_values: tuple[int, ...],
    validator_binding_digest: bytes,
) -> bytes:
    statement = claim.operation.statement(
        validator_binding_digest=validator_binding_digest
    )
    oracle = claim.surrogate_oracle.canonical_bytes()
    return (
        struct.pack("<I", claim_index)
        + statement.digest()
        + struct.pack("<I", claim.row_index)
        + struct.pack("<I", len(claim.input_row_i8))
        + struct.pack(
            f"<{len(claim.input_row_i8)}b", *claim.input_row_i8
        )
        + struct.pack("<I", len(oracle))
        + oracle
        + proof.surrogate_commitment
        + struct.pack("<I", len(sample_columns))
        + struct.pack(f"<{len(sample_columns)}I", *sample_columns)
        + struct.pack(f"<{len(sample_values)}q", *sample_values)
        + b"".join(scalar_to_bytes(value) for value in proof.fold_targets)
    )


def _group_transcript(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    width: int,
    claim_indices: tuple[int, ...],
    claims: tuple[SuccinctProjectionClaimV3, ...],
    proofs: tuple[SuccinctProjectionClaimProofV3, ...],
    sample_columns: tuple[tuple[int, ...], ...],
    sample_values: tuple[tuple[int, ...], ...],
) -> _Transcript:
    transcript = _Transcript(validator_binding_digest)
    transcript.absorb(
        b"succinct_projection_abi",
        SUCCINCT_PROJECTION_BATCH_ABI_V3.encode("ascii"),
    )
    transcript.absorb(b"validator_nonce", validator_nonce)
    transcript.absorb(b"padded_input_dim", struct.pack("<I", width))
    transcript.absorb(b"claim_count", struct.pack("<I", len(claim_indices)))
    for claim_index in claim_indices:
        transcript.absorb(
            b"succinct_claim",
            _claim_transcript_record(
                claim_index=claim_index,
                claim=claims[claim_index],
                proof=proofs[claim_index],
                sample_columns=sample_columns[claim_index],
                sample_values=sample_values[claim_index],
                validator_binding_digest=validator_binding_digest,
            ),
        )
    return transcript


def _replay_group(
    *,
    group: LeanProjectionBatchGroupProofV3,
    transcript: _Transcript,
    claims: tuple[SuccinctProjectionClaimV3, ...],
    proofs: tuple[SuccinctProjectionClaimProofV3, ...],
    coefficients,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    targets = tuple(
        target
        for claim_index in group.claim_indices
        for target in proofs[claim_index].fold_targets
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
    inv2 = (PALLAS_SCALAR_MODULUS + 1) // 2
    for round_index, (g0, g1, g2) in enumerate(group.rounds):
        if (g0 + g1) % PALLAS_SCALAR_MODULUS != running:
            raise ProofV3VerificationError(
                "succinct projection sumcheck round does not match"
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
        running = (
            g0
            * ((challenge - 1) * (challenge - 2) % PALLAS_SCALAR_MODULUS)
            % PALLAS_SCALAR_MODULUS
            * inv2
            - g1
            * (challenge * (challenge - 2) % PALLAS_SCALAR_MODULUS)
            + g2
            * (challenge * (challenge - 1) % PALLAS_SCALAR_MODULUS)
            % PALLAS_SCALAR_MODULUS
            * inv2
        ) % PALLAS_SCALAR_MODULUS

    try:
        import numpy as np

        from zkllm.crypto.pcs_v2 import (
            PCSUnavailableError,
            evaluate_lean_projection_terminal_relation,
        )

        x_matrix = np.zeros(
            (len(group.claim_indices), group.padded_input_dim),
            dtype=np.int8,
        )
        for row_index, claim_index in enumerate(group.claim_indices):
            claim = claims[claim_index]
            x_matrix[row_index, : claim.operation.input_dim] = (
                claim.input_row_i8
            )
        terminal_relation = evaluate_lean_projection_terminal_relation(
            x_matrix.tobytes(order="C"),
            claim_count=len(group.claim_indices),
            width=group.padded_input_dim,
            point=point,
            alphas=alphas,
            terminal_fold_evaluations=group.terminal_fold_evaluations,
        )
    except PCSUnavailableError:
        terminal_relation = 0
        terminal_offset = 0
        for claim_index in group.claim_indices:
            claim = claims[claim_index]
            padded_x = claim.input_row_i8 + (0,) * (
                group.padded_input_dim - claim.operation.input_dim
            )
            x_evaluation = _mle(
                tuple(
                    value % PALLAS_SCALAR_MODULUS for value in padded_x
                ),
                tuple(point),
            )
            for _ in range(LEAN_PROJECTION_FOLD_COUNT_V3):
                terminal_relation += (
                    alphas[terminal_offset]
                    * x_evaluation
                    * group.terminal_fold_evaluations[terminal_offset]
                )
                terminal_offset += 1
    if running != terminal_relation % PALLAS_SCALAR_MODULUS:
        raise ProofV3VerificationError(
            "succinct projection terminal relation does not match"
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
    commitment_slots: list[bytes | None] = [
        None
    ] * len(group.terminal_fold_evaluations)
    by_catalog: dict[
        tuple[bytes, int],
        dict[bytes, list[int]],
    ] = {}
    for local_index, claim_index in enumerate(group.claim_indices):
        claim = claims[claim_index]
        coefficient_bytes = coefficients[claim_index].astype(
            "<u4", copy=False
        ).tobytes(order="C")
        by_catalog.setdefault(
            (
                claim.operation.registered_catalog_id,
                claim.operation.output_dim,
            ),
            {},
        ).setdefault(coefficient_bytes, []).append(local_index)
    folds_per_catalog_call = (
        MAX_CATALOG_FOLDS // LEAN_PROJECTION_FOLD_COUNT_V3
    )
    selected_operations = tuple(
        claims[index].operation for index in group.claim_indices
    )
    with registered_catalog_operations_v3(selected_operations):
        for (catalog_id, output_dim), coefficient_groups in by_catalog.items():
            unique_coefficients = tuple(coefficient_groups)
            for offset in range(
                0, len(unique_coefficients), folds_per_catalog_call
            ):
                batch = unique_coefficients[
                    offset : offset + folds_per_catalog_call
                ]
                folded = combine_registered_catalog_u31_batch(
                    catalog_id,
                    b"".join(batch),
                    term_count=output_dim,
                    fold_count=(
                        len(batch) * LEAN_PROJECTION_FOLD_COUNT_V3
                    ),
                )
                for batch_index, coefficient_bytes in enumerate(batch):
                    source = batch_index * LEAN_PROJECTION_FOLD_COUNT_V3
                    claim_folded = folded[
                        source : source + LEAN_PROJECTION_FOLD_COUNT_V3
                    ]
                    for local_index in coefficient_groups[coefficient_bytes]:
                        destination = (
                            local_index * LEAN_PROJECTION_FOLD_COUNT_V3
                        )
                        commitment_slots[
                            destination : destination
                            + LEAN_PROJECTION_FOLD_COUNT_V3
                        ] = claim_folded
    if any(commitment is None for commitment in commitment_slots):
        raise ProofV3VerificationError(
            "succinct projection catalog fold inventory is incomplete"
        )
    aggregate_coefficients: dict[bytes, int] = {}
    for commitment, beta in zip(
        commitment_slots, betas, strict=True
    ):
        if commitment is None:
            raise ProofV3VerificationError(
                "succinct projection catalog fold inventory is incomplete"
            )
        aggregate_coefficients[commitment] = (
            aggregate_coefficients.get(commitment, 0) + beta
        ) % PALLAS_SCALAR_MODULUS
    nonzero = tuple(
        (commitment, coefficient)
        for commitment, coefficient in aggregate_coefficients.items()
        if coefficient
    )
    if not nonzero:
        raise ProofV3VerificationError(
            "succinct projection aggregate commitment is degenerate"
        )
    expected_commitment = _combine_commitments_bounded_v3(
        tuple(commitment for commitment, _coefficient in nonzero),
        tuple(coefficient for _commitment, coefficient in nonzero),
    )
    if group.opening.commitment != expected_commitment:
        raise ProofV3VerificationError(
            "succinct projection opening is not the signed catalog fold"
        )
    if (
        group.opening.vector_length != group.padded_input_dim
        or group.opening.padded_length != group.padded_input_dim
        or group.opening.encoding != ENCODING_PALLAS_SCALAR
    ):
        raise ProofV3VerificationError(
            "succinct projection static opening shape is wrong"
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
            "succinct projection static opening evaluation is wrong"
        )
    outer = hashlib.sha256(
        _LEAN_TRANSCRIPT_DOMAIN + b"/OPENING/" + transcript.state
    ).digest()
    if not verify(group.opening, tuple(point), outer):
        raise ProofV3VerificationError(
            "succinct projection static PCS opening is invalid"
        )
    return tuple(point), betas


def _dynamic_group_indices(
    claims: tuple[SuccinctProjectionClaimV3, ...],
) -> tuple[tuple[int, ...], ...]:
    grouped: dict[tuple[bytes, bytes], list[int]] = {}
    for index, claim in enumerate(claims):
        grouped.setdefault(
            (
                claim.operation.operation_digest,
                claim.surrogate_oracle.root,
            ),
            [],
        ).append(index)
    result = tuple(tuple(indices) for indices in grouped.values())
    if tuple(sorted(index for group in result for index in group)) != tuple(
        range(len(claims))
    ):
        raise ProofV3Error(
            "succinct projection dynamic grouping is incomplete"
        )
    for indices in result:
        first = claims[indices[0]]
        if any(
            claims[index].operation != first.operation
            or claims[index].surrogate_oracle != first.surrogate_oracle
            for index in indices[1:]
        ):
            raise ProofV3Error(
                "succinct projection dynamic group is inconsistent"
            )
    return result


def _relation_group_indices(
    claims: tuple[SuccinctProjectionClaimV3, ...],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Mirror the lean width grouping without allocating fake output rows."""

    widths: dict[int, list[int]] = {}
    for index, claim in enumerate(claims):
        widths.setdefault(
            claim.operation.padded_input_dim,
            [],
        ).append(index)
    claims_per_group = (
        MAX_LEAN_PROJECTION_VECTORS // LEAN_PROJECTION_FOLD_COUNT_V3
    )
    return tuple(
        (width, tuple(indices[offset : offset + claims_per_group]))
        for width in sorted(widths)
        for indices in (widths[width],)
        for offset in range(0, len(indices), claims_per_group)
    )


def _placeholder_claim_proof(
    *,
    surrogate_commitment: bytes,
    targets: tuple[int, ...],
) -> SuccinctProjectionClaimProofV3:
    return SuccinctProjectionClaimProofV3(
        surrogate_commitment=surrogate_commitment,
        fold_targets=targets,
    )


def build_succinct_projection_batch_from_folds_v3(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    witnesses: tuple[SuccinctProjectionWitnessV3, ...],
    folds: tuple[LeanProjectionFoldV3, ...],
) -> SuccinctProjectionBatchProofV3:
    """Build the production-shaped succinct projection section."""

    import os
    import time

    trace = os.environ.get("VERATHOS_ATTN_TRACE") == "1"
    phase_started = time.perf_counter()

    binding = _fixed32(
        validator_binding_digest,
        "succinct projection validator binding",
        nonzero=True,
    )
    nonce = _fixed32(
        validator_nonce,
        "succinct projection validator nonce",
        nonzero=True,
    )
    witnesses = tuple(witnesses)
    if (
        not witnesses
        or len(witnesses) > MAX_SUCCINCT_PROJECTION_CLAIMS_V3
        or not all(
            isinstance(witness, SuccinctProjectionWitnessV3)
            for witness in witnesses
        )
    ):
        raise ProofV3Error("succinct projection witness set is malformed")
    claims = tuple(witness.claim for witness in witnesses)
    folds = tuple(folds)
    if (
        len(folds) != len(claims)
        or any(
            not isinstance(fold, LeanProjectionFoldV3)
            or any(
                len(row) != claim.operation.padded_input_dim
                for row in fold.folded_weights
            )
            for claim, fold in zip(claims, folds, strict=True)
        )
    ):
        raise ProofV3Error("succinct projection folded witness set is wrong")

    dynamic_group_indices = _dynamic_group_indices(claims)
    claim_commitments = tuple(
        claim.surrogate_oracle.root for claim in claims
    )
    if trace:
        print(
            "[PROOF-V3-SUCCINCT-PROJECTION] "
            f"phase=surrogate-root-bindings seconds="
            f"{time.perf_counter() - phase_started:.3f} "
            f"claims={len(witnesses)} groups={len(dynamic_group_indices)}",
            flush=True,
        )
    phase_started = time.perf_counter()
    sample_columns = tuple(
        _sample_columns(
            validator_binding_digest=binding,
            validator_nonce=nonce,
            claim=claim,
            surrogate_commitment=commitment,
        )
        for claim, commitment in zip(
            claims,
            claim_commitments,
            strict=True,
        )
    )
    sample_values = tuple(
        tuple(
            witness.committed_surrogate.signed_value(
                witness.claim.row_index,
                column,
            )
            for column in columns
        )
        for witness, columns in zip(
            witnesses,
            sample_columns,
            strict=True,
        )
    )
    capture_openings = []
    for indices in dynamic_group_indices:
        first = witnesses[indices[0]]
        _opened, opening = first.committed_surrogate.open_cells(
            tuple(
                (
                    witnesses[claim_index].claim.row_index,
                    column,
                )
                for claim_index in indices
                for column in sample_columns[claim_index]
            ),
            value_mode=3,
            bounded_width=bounded_byte_width_v3(
                first.claim.operation.input_dim
            ),
        )
        capture_openings.append(opening)
    if trace:
        print(
            "[PROOF-V3-SUCCINCT-PROJECTION] "
            f"phase=capture-openings seconds="
            f"{time.perf_counter() - phase_started:.3f}",
            flush=True,
        )
    phase_started = time.perf_counter()
    coefficients = tuple(
        derive_succinct_projection_coefficients_v3(
            validator_binding_digest=binding,
            validator_nonce=nonce,
            claim=claim,
            surrogate_commitment=commitment,
        )
        for claim, commitment in zip(
            claims,
            claim_commitments,
            strict=True,
        )
    )
    targets = tuple(
        _exact_targets(values, columns, rows)
        for values, columns, rows in zip(
            sample_values,
            sample_columns,
            coefficients,
            strict=True,
        )
    )
    placeholders = tuple(
        _placeholder_claim_proof(
            surrogate_commitment=commitment,
            targets=claim_targets,
        )
        for commitment, claim_targets in zip(
            claim_commitments,
            targets,
            strict=True,
        )
    )
    lean_claims = tuple(
        LeanProjectionBatchClaimV3(
            operation=claim.operation,
            input_row_i8=claim.input_row_i8,
            surrogate_output_i64=witness.surrogate_output_i64,
        )
        for claim, witness in zip(claims, witnesses, strict=True)
    )
    groups = []
    for width, claim_indices in _group_indices(lean_claims):
        transcript = _group_transcript(
            validator_binding_digest=binding,
            validator_nonce=nonce,
            width=width,
            claim_indices=claim_indices,
            claims=claims,
            proofs=placeholders,
            sample_columns=sample_columns,
            sample_values=sample_values,
        )
        group_targets = tuple(
            target
            for claim_index in claim_indices
            for target in targets[claim_index]
        )
        alphas = tuple(
            transcript.scalar(b"claim_alpha" + struct.pack("<I", index))
            for index in range(len(group_targets))
        )
        try:
            groups.append(
                _build_native_group_proof_v3(
                    width=width,
                    claim_indices=claim_indices,
                    indexed_claims=tuple(
                        (index, lean_claims[index])
                        for index in claim_indices
                    ),
                    folds=folds,
                    fold_targets=tuple(
                        targets[index] for index in claim_indices
                    ),
                    alphas=alphas,
                    transcript=transcript,
                )
            )
        except PCSNativeError as exc:
            raise ProofV3Error(
                f"succinct projection relation is not satisfied: {exc}"
            ) from exc
    group_proof = LeanProjectionBatchProofV3(tuple(groups))
    if trace:
        print(
            "[PROOF-V3-SUCCINCT-PROJECTION] "
            f"phase=static-fold-proofs seconds="
            f"{time.perf_counter() - phase_started:.3f} "
            f"width-groups={len(groups)}",
            flush=True,
        )
    phase_started = time.perf_counter()
    dynamic_groups = []
    for claim_indices, capture_opening in zip(
        dynamic_group_indices,
        capture_openings,
        strict=True,
    ):
        dynamic_groups.append(
            SuccinctProjectionDynamicGroupProofV3(
                claim_indices=claim_indices,
                capture_opening=capture_opening,
            )
        )
    if trace:
        print(
            "[PROOF-V3-SUCCINCT-PROJECTION] "
            f"phase=capture-groups-total seconds="
            f"{time.perf_counter() - phase_started:.3f}",
            flush=True,
        )
    return SuccinctProjectionBatchProofV3(
        placeholders,
        tuple(dynamic_groups),
        group_proof,
    )


def build_succinct_projection_batch_reference_v3(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    witnesses: tuple[SuccinctProjectionWitnessV3, ...],
    weight_columns_i8,
) -> SuccinctProjectionBatchProofV3:
    """Reference folded-weight builder for small conformance fixtures."""

    witnesses = tuple(witnesses)
    columns_by_claim = tuple(weight_columns_i8)
    if len(columns_by_claim) != len(witnesses):
        raise ProofV3Error("succinct projection weight witness count is wrong")
    commitments = tuple(
        witness.claim.surrogate_oracle.root
        for witness in witnesses
    )
    coefficients = tuple(
        derive_succinct_projection_coefficients_v3(
            validator_binding_digest=validator_binding_digest,
            validator_nonce=validator_nonce,
            claim=witness.claim,
            surrogate_commitment=commitment,
        )
        for witness, commitment in zip(
            witnesses,
            commitments,
            strict=True,
        )
    )
    folds = []
    for witness, columns, rows in zip(
        witnesses,
        columns_by_claim,
        coefficients,
        strict=True,
    ):
        claim = witness.claim
        if isinstance(columns, SuccinctProjectionWeightRowsV3):
            if (
                columns.output_columns != claim.output_columns
                or columns.input_dim != claim.operation.input_dim
                or columns.output_dim != claim.operation.output_dim
            ):
                raise ProofV3Error(
                    "succinct projection sampled weights do not match "
                    "the claim"
                )
            raw_columns = (
                columns.rows_i8.tolist()
                if hasattr(columns.rows_i8, "tolist")
                else columns.rows_i8
            )
            selected_weight_rows = {
                output: _signed_row(
                    tuple(int(value) for value in row)
                    + (0,) * (
                        claim.operation.padded_input_dim
                        - claim.operation.input_dim
                    ),
                    length=claim.operation.padded_input_dim,
                    bits=8,
                    name="succinct projection sampled weight row",
                )
                for output, row in zip(
                    columns.output_columns,
                    raw_columns,
                    strict=True,
                )
            }
        else:
            columns = tuple(
                _signed_row(
                    column,
                    length=claim.operation.padded_input_dim,
                    bits=8,
                    name="succinct projection weight column",
                )
                for column in columns
            )
            if len(columns) != claim.operation.output_dim:
                raise ProofV3Error(
                    "succinct projection weight column count is wrong"
                )
            selected_weight_rows = {
                output: columns[output]
                for output in claim.output_columns
            }
        selected_columns = claim.output_columns
        folded = tuple(
            tuple(
                sum(
                    int(rows[fold_index, output])
                    * selected_weight_rows[output][inner]
                    for output in selected_columns
                )
                for inner in range(claim.operation.padded_input_dim)
            )
            for fold_index in range(LEAN_PROJECTION_FOLD_COUNT_V3)
        )
        folds.append(LeanProjectionFoldV3(folded))
    return build_succinct_projection_batch_from_folds_v3(
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
        witnesses=witnesses,
        folds=tuple(folds),
    )


def verify_succinct_projection_batch_v3(
    *,
    proof: SuccinctProjectionBatchProofV3,
    validator_binding_digest: bytes,
    capture_base_binding_digest: bytes,
    validator_nonce: bytes,
    claims: tuple[SuccinctProjectionClaimV3, ...],
) -> None:
    """Verify the nonce-sampled projection section."""

    try:
        binding = _fixed32(
            validator_binding_digest,
            "succinct projection validator binding",
            nonzero=True,
        )
        capture_binding = _fixed32(
            capture_base_binding_digest,
            "succinct projection capture base binding",
            nonzero=True,
        )
        nonce = _fixed32(
            validator_nonce,
            "succinct projection validator nonce",
            nonzero=True,
        )
        if not isinstance(proof, SuccinctProjectionBatchProofV3):
            raise ProofV3VerificationError(
                "succinct projection proof has an unexpected type"
            )
        claims = tuple(claims)
        if (
            len(claims) != len(proof.claims)
            or not claims
            or not all(
                isinstance(claim, SuccinctProjectionClaimV3)
                for claim in claims
            )
        ):
            raise ProofV3VerificationError(
                "succinct projection claim inventory is inconsistent"
            )

        expected_dynamic_groups = _dynamic_group_indices(claims)
        actual_dynamic_groups = tuple(
            group.claim_indices for group in proof.dynamic_groups
        )
        if actual_dynamic_groups != expected_dynamic_groups:
            raise ProofV3VerificationError(
                "succinct projection dynamic group inventory is not exact"
            )

        sample_columns: list[tuple[int, ...] | None] = [None] * len(claims)
        sample_values: list[tuple[int, ...] | None] = [None] * len(claims)
        coefficients: list[object | None] = [None] * len(claims)
        coefficient_cache: dict[
            tuple[
                LeanProjectionCatalogOperationV3,
                bytes,
                bytes,
                tuple[int, ...],
            ],
            object,
        ] = {}
        for claim_index, (claim, claim_proof) in enumerate(
            zip(claims, proof.claims, strict=True)
        ):
            if (
                claim_proof.surrogate_commitment
                != claim.surrogate_oracle.root
            ):
                raise ProofV3VerificationError(
                    "succinct projection surrogate root is inconsistent"
                )
            coefficient_key = (
                claim.operation,
                claim.surrogate_oracle.root,
                claim_proof.surrogate_commitment,
                claim.output_columns,
            )
            shared_coefficients = coefficient_cache.get(coefficient_key)
            if shared_coefficients is None:
                shared_coefficients = derive_succinct_projection_coefficients_v3(
                    validator_binding_digest=binding,
                    validator_nonce=nonce,
                    claim=claim,
                    surrogate_commitment=(
                        claim_proof.surrogate_commitment
                    ),
                )
                coefficient_cache[coefficient_key] = shared_coefficients
            coefficients[claim_index] = shared_coefficients
        for dynamic_group in proof.dynamic_groups:
            group_claims = tuple(
                claims[index] for index in dynamic_group.claim_indices
            )
            group_proofs = tuple(
                proof.claims[index]
                for index in dynamic_group.claim_indices
            )
            first_claim = group_claims[0]
            first_commitment = group_proofs[0].surrogate_commitment
            if (
                any(
                    claim.operation != first_claim.operation
                    or claim.surrogate_oracle != first_claim.surrogate_oracle
                    for claim in group_claims[1:]
                )
                or any(
                    item.surrogate_commitment != first_commitment
                    for item in group_proofs[1:]
                )
            ):
                raise ProofV3VerificationError(
                    "succinct projection dynamic group is inconsistent"
                )
            group_cells: list[int] = []
            for claim_index in dynamic_group.claim_indices:
                claim = claims[claim_index]
                claim_proof = proof.claims[claim_index]
                columns = _sample_columns(
                    validator_binding_digest=binding,
                    validator_nonce=nonce,
                    claim=claim,
                    surrogate_commitment=claim_proof.surrogate_commitment,
                )
                sample_columns[claim_index] = columns
                group_cells.extend(
                    oracle_leaf_index_v3(
                        claim.row_index,
                        column,
                        claim.surrogate_oracle.col_count,
                    )
                    for column in columns
                )
            opened = verify_economic_oracle_opening_v3(
                oracle=first_claim.surrogate_oracle,
                base_binding=capture_binding,
                expected_indices=group_cells,
                opening=dynamic_group.capture_opening,
                expected_mode=3,
                expected_bounded_width=bounded_byte_width_v3(
                    first_claim.operation.input_dim
                ),
            )
            for claim_index in dynamic_group.claim_indices:
                claim = claims[claim_index]
                columns = sample_columns[claim_index]
                if columns is None:
                    raise ProofV3VerificationError(
                        "succinct projection samples are incomplete"
                    )
                cells = tuple(
                    oracle_leaf_index_v3(
                        claim.row_index,
                        column,
                        claim.surrogate_oracle.col_count,
                    )
                    for column in columns
                )
                sample_values[claim_index] = tuple(
                    opened[cell] for cell in cells
                )

        if (
            any(value is None for value in sample_columns)
            or any(value is None for value in sample_values)
            or any(value is None for value in coefficients)
        ):
            raise ProofV3VerificationError(
                "succinct projection dynamic witness is incomplete"
            )
        sample_columns_tuple = tuple(
            value for value in sample_columns if value is not None
        )
        sample_values_tuple = tuple(
            value for value in sample_values if value is not None
        )
        coefficients_tuple = tuple(
            value for value in coefficients if value is not None
        )
        for claim_index, claim_proof in enumerate(proof.claims):
            expected_targets = _exact_targets(
                sample_values_tuple[claim_index],
                sample_columns_tuple[claim_index],
                coefficients_tuple[claim_index],
            )
            if claim_proof.fold_targets != expected_targets:
                raise ProofV3VerificationError(
                    "succinct projection fold target does not match "
                    "the authenticated surrogate cells"
                )
        expected_groups = _relation_group_indices(claims)
        actual_groups = tuple(
            (group.padded_input_dim, group.claim_indices)
            for group in proof.groups.groups
        )
        if actual_groups != expected_groups:
            raise ProofV3VerificationError(
                "succinct projection group inventory is not exact"
            )
        for group in proof.groups.groups:
            transcript = _group_transcript(
                validator_binding_digest=binding,
                validator_nonce=nonce,
                width=group.padded_input_dim,
                claim_indices=group.claim_indices,
                claims=claims,
                proofs=proof.claims,
                sample_columns=sample_columns_tuple,
                sample_values=sample_values_tuple,
            )
            _replay_group(
                group=group,
                transcript=transcript,
                claims=claims,
                proofs=proof.claims,
                coefficients=coefficients_tuple,
            )
    except ProofV3VerificationError:
        raise
    except Exception as exc:
        raise ProofV3VerificationError(
            f"succinct projection verification failed: {exc}"
        ) from exc


def _encode_pcs_opening(writer: _Writer, opening: PCSOpeningV2) -> None:
    if len(opening.proof) >= 1 << 16:
        raise ProofV3Error("succinct projection IPA proof is too large")
    writer.raw(opening.commitment)
    writer.raw(opening.evaluation)
    writer.pack(
        "<IIBH",
        opening.vector_length,
        opening.padded_length,
        opening.encoding,
        len(opening.proof),
    )
    writer.raw(opening.proof)


def _decode_pcs_opening(reader: _Reader) -> PCSOpeningV2:
    commitment = reader.read(32)
    evaluation = reader.read(32)
    vector_length, padded_length, encoding, proof_length = reader.unpack(
        "<IIBH"
    )
    if proof_length == 0 or proof_length > 2048:
        raise ProofV3Error("succinct projection IPA proof length is invalid")
    return PCSOpeningV2(
        commitment=commitment,
        evaluation=evaluation,
        proof=reader.read(proof_length),
        vector_length=vector_length,
        padded_length=padded_length,
        encoding=encoding,
    )


def encode_succinct_projection_batch_v3(
    proof: SuccinctProjectionBatchProofV3,
) -> bytes:
    if not isinstance(proof, SuccinctProjectionBatchProofV3):
        raise ProofV3Error("succinct projection proof has an unexpected type")
    writer = _Writer()
    writer.raw(_WIRE_MAGIC)
    writer.pack("<HI", _WIRE_VERSION, len(proof.claims))
    for claim in proof.claims:
        writer.raw(claim.surrogate_commitment)
        for target in claim.fold_targets:
            writer.raw(scalar_to_bytes(target))
    claim_bytes = len(writer.finish())
    writer.pack("<I", len(proof.dynamic_groups))
    capture_bytes = 0
    for group in proof.dynamic_groups:
        writer.pack("<I", len(group.claim_indices))
        for claim_index in group.claim_indices:
            writer.pack("<I", claim_index)
        capture = _capture_blob(group.capture_opening)
        capture_bytes += len(capture)
        writer.vbytes(
            capture,
            "succinct projection capture opening",
            MAX_SUCCINCT_PROJECTION_WIRE_BYTES_V3,
        )
    group_blob = encode_lean_projection_batch_v3(proof.groups)
    writer.vbytes(
        group_blob,
        "succinct projection group proof",
        MAX_SUCCINCT_PROJECTION_WIRE_BYTES_V3,
    )
    encoded = writer.finish()
    import os
    if os.environ.get("VERATHOS_ATTN_TRACE") == "1":
        print(
            "[PROOF-V3-SUCCINCT-PROJECTION] "
            f"wire_total={len(encoded)} "
            f"claim_records={claim_bytes} "
            f"capture_openings={capture_bytes} "
            f"static_group_proof={len(group_blob)} "
            f"dynamic_groups={len(proof.dynamic_groups)}",
            flush=True,
        )
    if len(encoded) > MAX_SUCCINCT_PROJECTION_WIRE_BYTES_V3:
        raise ProofV3Error(
            "succinct projection proof exceeds the wire gate "
            f"({len(encoded)} > "
            f"{MAX_SUCCINCT_PROJECTION_WIRE_BYTES_V3} bytes)"
        )
    return encoded


def decode_succinct_projection_batch_v3(
    encoded: bytes,
) -> SuccinctProjectionBatchProofV3:
    if (
        not isinstance(encoded, bytes)
        or not 0 < len(encoded) <= MAX_SUCCINCT_PROJECTION_WIRE_BYTES_V3
    ):
        raise ProofV3Error("succinct projection wire size is invalid")
    reader = _Reader(encoded, "succinct projection proof")
    if reader.read(4) != _WIRE_MAGIC:
        raise ProofV3Error("succinct projection wire magic is wrong")
    version, claim_count = reader.unpack("<HI")
    if (
        version != _WIRE_VERSION
        or not 0 < claim_count <= MAX_SUCCINCT_PROJECTION_CLAIMS_V3
    ):
        raise ProofV3Error("succinct projection wire header is invalid")
    claims = []
    for _ in range(claim_count):
        commitment = reader.read(32)
        targets = tuple(
            scalar_from_bytes(reader.read(32))
            for _ in range(LEAN_PROJECTION_FOLD_COUNT_V3)
        )
        claims.append(
            SuccinctProjectionClaimProofV3(
                surrogate_commitment=commitment,
                fold_targets=targets,
            )
        )
    (dynamic_group_count,) = reader.unpack("<I")
    if (
        dynamic_group_count == 0
        or dynamic_group_count > claim_count
    ):
        raise ProofV3Error(
            "succinct projection dynamic group count is invalid"
        )
    dynamic_groups = []
    for _ in range(dynamic_group_count):
        (group_claim_count,) = reader.unpack("<I")
        if (
            group_claim_count == 0
            or group_claim_count > claim_count
        ):
            raise ProofV3Error(
                "succinct projection dynamic group size is invalid"
            )
        claim_indices = tuple(
            reader.unpack("<I")[0] for _ in range(group_claim_count)
        )
        capture = _decode_capture_blob(
            reader.vbytes(
                "succinct projection capture opening",
                MAX_SUCCINCT_PROJECTION_WIRE_BYTES_V3,
            )
        )
        dynamic_groups.append(
            SuccinctProjectionDynamicGroupProofV3(
                claim_indices=claim_indices,
                capture_opening=capture,
            )
        )
    groups = decode_lean_projection_batch_v3(
        reader.vbytes(
            "succinct projection group proof",
            MAX_SUCCINCT_PROJECTION_WIRE_BYTES_V3,
        )
    )
    reader.finish()
    result = SuccinctProjectionBatchProofV3(
        tuple(claims),
        tuple(dynamic_groups),
        groups,
    )
    if encode_succinct_projection_batch_v3(result) != encoded:
        raise ProofV3Error("succinct projection wire is not canonical")
    return result
