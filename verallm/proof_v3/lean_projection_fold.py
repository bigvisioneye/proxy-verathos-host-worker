"""Full-output registered-weight folding for lean proof-v3 transitions.

One selected runtime row is small enough for the hard-audit wire to disclose.
The validator therefore does not need a GEMM sumcheck over that row.  It checks
the complete output with four independent 31-bit random folds:

    <X, sum_j u_j W_j> == sum_j u_j S_j

where ``S`` is the exact integer projection surrogate and every ``W_j`` is a
column committed by the authenticated static catalog.  The prover sends only
the four folded weight vectors.  The validator commits each folded vector and
compares it with the corresponding linear combination of signed catalog
commitments before checking the equations.

The 31-bit coefficient domain gives at most 2^-31 cancellation probability per
fold for any fixed non-zero output discrepancy; four folds give at most
2^-124.  Coefficients are derived after X and S are transcript-bound.
"""

from __future__ import annotations

import hashlib
import struct
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterator

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from zkllm.crypto.gemm_v2_reference import PALLAS_SCALAR_MODULUS

LEAN_PROJECTION_FOLD_ABI_V3: Final = "projection_fold.full_row.u31x4.v1"
LEAN_PROJECTION_FOLD_COUNT_V3: Final = 4
LEAN_PROJECTION_COEFFICIENT_BITS_V3: Final = 31

_TRANSCRIPT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/LEAN_PROJECTION_FOLD/U31X4/V1/SHAKE256"
)

__all__ = [
    "LEAN_PROJECTION_COEFFICIENT_BITS_V3",
    "LEAN_PROJECTION_FOLD_ABI_V3",
    "LEAN_PROJECTION_FOLD_COUNT_V3",
    "LeanProjectionCatalogV3",
    "LeanProjectionCatalogOperationV3",
    "LeanProjectionFoldV3",
    "LeanProjectionStatementV3",
    "build_lean_projection_fold_reference_v3",
    "derive_lean_projection_coefficients_v3",
    "lean_projection_operation_key_v3",
    "verify_lean_projection_fold_v3",
]

_OPERATION_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/LEAN_PROJECTION_OPERATION/V1"
)
_CATALOG_ID_DOMAIN: Final = b"VERATHOS/PCS_V2/CATALOG_POINT_CACHE/V1"


def _catalog_commitment_id_v3(commitments: tuple[bytes, ...]) -> bytes:
    digest = hashlib.sha256()
    digest.update(_CATALOG_ID_DOMAIN)
    digest.update(struct.pack("<Q", len(commitments)))
    for commitment in commitments:
        digest.update(commitment)
    return digest.digest()


def lean_projection_operation_key_v3(
    *,
    layer_index: int,
    projection: str,
):
    """Map the economic manifest name to its signed proof-v2 operation key."""

    try:
        from verallm.challenge.v2 import OperationKeyV2
        from verallm.proof_v2.layout import (
            FULL_OUTPUT_OPERATION_ID,
            FULL_QKV_OPERATION_ID,
            GDN_BA_OPERATION_ID,
            GDN_OUTPUT_OPERATION_ID,
            GDN_QKVZ_OPERATION_ID,
            MLP_DOWN_OPERATION_ID,
            MLP_GATE_UP_OPERATION_ID,
        )
    except ImportError as exc:  # pragma: no cover - production dependency.
        raise ProofV3Error(
            "proof-v2 operation support is unavailable"
        ) from exc
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or not 0 <= layer_index < 1 << 32
    ):
        raise ProofV3Error("lean projection layer is malformed")
    operation_ids = {
        "qkv": FULL_QKV_OPERATION_ID,
        "o": FULL_OUTPUT_OPERATION_ID,
        "gate_up": MLP_GATE_UP_OPERATION_ID,
        "down": MLP_DOWN_OPERATION_ID,
        "gdn_qkvz": GDN_QKVZ_OPERATION_ID,
        "gdn_ba": GDN_BA_OPERATION_ID,
        "gdn_o": GDN_OUTPUT_OPERATION_ID,
    }
    try:
        operation_id = operation_ids[projection]
    except (KeyError, TypeError) as exc:
        raise ProofV3Error(
            "lean projection operation is unsupported"
        ) from exc
    return OperationKeyV2(layer_index, operation_id, -1)


def _fixed32(value, name: str, *, nonzero: bool = False) -> bytes:
    if (
        not isinstance(value, bytes)
        or len(value) != 32
        or (nonzero and not any(value))
    ):
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    return value


def _positive(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProofV3Error(f"{name} must be positive")
    return value


def _power_of_two(value, name: str) -> int:
    result = _positive(value, name)
    if result & (result - 1):
        raise ProofV3Error(f"{name} must be a power of two")
    return result


def _signed_i8_row(values, *, length: int, name: str) -> tuple[int, ...]:
    row = tuple(int(value) for value in values)
    if len(row) != length or any(value < -128 or value > 127 for value in row):
        raise ProofV3Error(f"{name} is not a canonical signed-i8 row")
    return row


def _signed_i64_row(values, *, length: int, name: str) -> tuple[int, ...]:
    row = tuple(int(value) for value in values)
    if len(row) != length or any(
        value < -(1 << 63) or value >= 1 << 63 for value in row
    ):
        raise ProofV3Error(f"{name} is not a canonical signed-i64 row")
    return row


@dataclass(frozen=True, slots=True)
class LeanProjectionStatementV3:
    validator_binding_digest: bytes
    operation_digest: bytes
    input_dim: int
    padded_input_dim: int
    output_dim: int

    def __post_init__(self) -> None:
        _fixed32(
            self.validator_binding_digest,
            "lean projection validator binding",
            nonzero=True,
        )
        _fixed32(
            self.operation_digest,
            "lean projection operation digest",
            nonzero=True,
        )
        input_dim = _positive(self.input_dim, "lean projection input_dim")
        padded = _power_of_two(
            self.padded_input_dim, "lean projection padded_input_dim"
        )
        output = _positive(self.output_dim, "lean projection output_dim")
        if input_dim > padded or output >= 1 << 32 or padded >= 1 << 32:
            raise ProofV3Error("lean projection dimensions are out of range")

    def digest(self) -> bytes:
        return hashlib.sha256(
            _TRANSCRIPT_DOMAIN
            + b"/STATEMENT/"
            + self.validator_binding_digest
            + self.operation_digest
            + struct.pack(
                "<III",
                self.input_dim,
                self.padded_input_dim,
                self.output_dim,
            )
        ).digest()


@dataclass(frozen=True, slots=True)
class LeanProjectionFoldV3:
    folded_weights: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        folds = tuple(tuple(int(value) for value in row) for row in self.folded_weights)
        if len(folds) != LEAN_PROJECTION_FOLD_COUNT_V3:
            raise ProofV3Error("lean projection fold count is wrong")
        if not folds or any(len(row) != len(folds[0]) for row in folds):
            raise ProofV3Error("lean projection folded weights are ragged")
        if any(
            value < -(1 << 63) or value >= 1 << 63
            for row in folds
            for value in row
        ):
            raise ProofV3Error("lean projection folded weight exceeds signed i64")
        object.__setattr__(self, "folded_weights", folds)


@dataclass(frozen=True, slots=True)
class LeanProjectionCatalogOperationV3:
    """One signed operation registered in the native Pallas fold cache."""

    operation_key: object
    operation_digest: bytes
    input_dim: int
    padded_input_dim: int
    output_dim: int
    operation_root: bytes
    registered_catalog_id: bytes
    _source_path: Path | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _source_offset: int | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _inline_commitments: tuple[bytes, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        try:
            from verallm.challenge.v2 import OperationKeyV2
        except ImportError as exc:  # pragma: no cover - proof-v2 is required.
            raise ProofV3Error("proof-v2 operation support is unavailable") from exc
        if not isinstance(self.operation_key, OperationKeyV2):
            raise ProofV3Error("lean projection operation key has a wrong type")
        _fixed32(
            self.operation_digest,
            "lean projection operation digest",
            nonzero=True,
        )
        _positive(self.input_dim, "lean projection operation input_dim")
        _power_of_two(
            self.padded_input_dim,
            "lean projection operation padded_input_dim",
        )
        _positive(self.output_dim, "lean projection operation output_dim")
        if self.input_dim > self.padded_input_dim:
            raise ProofV3Error("lean projection operation padding is too short")
        _fixed32(
            self.operation_root,
            "lean projection operation root",
            nonzero=True,
        )
        _fixed32(
            self.registered_catalog_id,
            "lean projection registered catalog id",
            nonzero=True,
        )
        file_backed = isinstance(self._source_path, Path)
        if file_backed != (self._source_offset is not None) or (
            self._source_offset is not None
            and (
                isinstance(self._source_offset, bool)
                or not isinstance(self._source_offset, int)
                or self._source_offset < 0
            )
        ):
            raise ProofV3Error(
                "lean projection catalog file source is malformed"
            )
        inline = self._inline_commitments
        if file_backed and inline is not None:
            raise ProofV3Error(
                "lean projection catalog operation has ambiguous sources"
            )
        if inline is not None and (
            len(inline) != self.output_dim
            or any(type(item) is not bytes or len(item) != 32 for item in inline)
        ):
            raise ProofV3Error(
                "lean projection inline catalog commitments are malformed"
            )

    def _commitment_bytes(self) -> bytes:
        expected = self.output_dim * 32
        if self._source_path is not None:
            try:
                with self._source_path.open("rb", buffering=0) as handle:
                    handle.seek(self._source_offset or 0)
                    encoded = handle.read(expected)
            except OSError as exc:
                raise ProofV3VerificationError(
                    "authenticated projection catalog source is unavailable"
                ) from exc
            if len(encoded) != expected:
                raise ProofV3VerificationError(
                    "authenticated projection catalog source changed"
                )
            return encoded
        if self._inline_commitments is None:
            raise ProofV3VerificationError(
                "authenticated projection catalog source is unavailable"
            )
        return b"".join(self._inline_commitments)

    def ensure_native_registration(self) -> None:
        """Rehydrate an authenticated operation after native LRU eviction."""

        # Directly constructed test/reference operations may refer to a
        # catalog that their caller registered explicitly. Authenticated
        # release operations always carry one of the sources above.
        if self._source_path is None and self._inline_commitments is None:
            return
        try:
            from zkllm.crypto.pcs_v2 import register_catalog_commitments

            catalog_id, term_count = register_catalog_commitments(
                self._commitment_bytes()
            )
        except ProofV3VerificationError:
            raise
        except Exception as exc:
            raise ProofV3VerificationError(
                "authenticated projection catalog could not be registered"
            ) from exc
        if (
            term_count != self.output_dim
            or catalog_id != self.registered_catalog_id
        ):
            raise ProofV3VerificationError(
                "authenticated projection catalog source does not match"
            )

    def statement(
        self,
        *,
        validator_binding_digest: bytes,
    ) -> LeanProjectionStatementV3:
        return LeanProjectionStatementV3(
            validator_binding_digest=validator_binding_digest,
            operation_digest=self.operation_digest,
            input_dim=self.input_dim,
            padded_input_dim=self.padded_input_dim,
            output_dim=self.output_dim,
        )


@contextmanager
def registered_catalog_operations_v3(
    operations: tuple[LeanProjectionCatalogOperationV3, ...],
) -> Iterator[None]:
    """Keep selected authenticated operations resident for one fold batch."""

    from zkllm.crypto.pcs_v2 import registered_catalog_use

    with registered_catalog_use():
        unique = {item.registered_catalog_id: item for item in operations}
        for operation in unique.values():
            operation.ensure_native_registration()
        yield

@dataclass(frozen=True, slots=True)
class LeanProjectionCatalogV3:
    """Serving-time view of the existing authenticated proof-v2 catalog.

    Construction accepts only the factory-authenticated v3 bridge.  It parses
    the exact canonical catalog bytes once, derives operation identities from
    the signed descriptors rather than miner-supplied data, and registers only
    selected operations in the bounded native cache when they are consumed.
    """

    manifest_digest: bytes
    operations: tuple[LeanProjectionCatalogOperationV3, ...]
    _by_key: dict[object, LeanProjectionCatalogOperationV3] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _fixed32(
            self.manifest_digest,
            "lean projection catalog manifest digest",
            nonzero=True,
        )
        operations = tuple(self.operations)
        if not operations or not all(
            isinstance(item, LeanProjectionCatalogOperationV3)
            for item in operations
        ):
            raise ProofV3Error("lean projection catalog operations are malformed")
        keys = tuple(item.operation_key for item in operations)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ProofV3Error(
                "lean projection catalog operations are not ordered and distinct"
            )
        object.__setattr__(self, "operations", operations)
        object.__setattr__(
            self,
            "_by_key",
            {item.operation_key: item for item in operations},
        )

    @classmethod
    def from_verified_v2_catalog_binding(
        cls,
        *,
        verified_v2_catalog_binding: object,
    ) -> "LeanProjectionCatalogV3":
        """Register the exact signed operation catalogs after qualification."""

        try:
            from verallm.challenge.v2 import (
                MODEL_OPERATION_LAYER_IDX,
                OperationKeyV2,
            )
            from verallm.proof_v2.catalog import WeightCommitmentCatalogV2
            from verallm.proof_v3.catalog import VerifiedV2CatalogBindingV3
        except ImportError as exc:  # pragma: no cover - production dependencies.
            raise ProofV3Error(
                "lean projection catalog support is unavailable"
            ) from exc
        if not isinstance(
            verified_v2_catalog_binding,
            VerifiedV2CatalogBindingV3,
        ):
            raise ProofV3Error(
                "lean projection catalog requires a verified proof-v2 binding"
            )
        try:
            verified_v2_catalog_binding.require_verified_v2_provenance()
            static_manifest = verified_v2_catalog_binding.static_manifest
            catalog = (
                verified_v2_catalog_binding._source_weight_catalog
            )
            if not isinstance(catalog, WeightCommitmentCatalogV2):
                raise ProofV3Error(
                    "lean projection catalog lacks authenticated source bytes"
                )
            descriptors_by_key = {
                OperationKeyV2(
                    (
                        MODEL_OPERATION_LAYER_IDX
                        if descriptor.layer == -1
                        else descriptor.layer
                    ),
                    descriptor.operation_id,
                    (
                        -1
                        if descriptor.expert_id is None
                        else descriptor.expert_id
                    ),
                ): descriptor
                for descriptor in static_manifest.operations
            }
            if (
                catalog.manifest_digest != static_manifest.manifest_digest
                or len(descriptors_by_key) != len(static_manifest.operations)
                or frozenset(descriptors_by_key)
                != frozenset(catalog.operation_keys)
            ):
                raise ProofV3Error(
                    "lean projection catalog does not exactly cover the "
                    "authenticated manifest"
                )
            operations = []
            for catalog_operation in catalog.operations:
                key = catalog_operation.key
                descriptor = descriptors_by_key[key]
                commitments = catalog_operation.column_commitments
                term_count = len(commitments)
                if catalog_operation._source_path is None:
                    from zkllm.crypto.pcs_v2 import (
                        register_catalog_commitments,
                    )

                    catalog_id, registered_count = (
                        register_catalog_commitments(commitments)
                    )
                    if registered_count != term_count:
                        raise ProofV3Error(
                            "lean projection catalog registration is malformed"
                        )
                else:
                    catalog_id = _catalog_commitment_id_v3(commitments)
                if term_count != descriptor.cols:
                    raise ProofV3Error(
                        "lean projection catalog dimensions do not match the "
                        "authenticated manifest"
                    )
                padded_input_dim = 1 << (descriptor.rows - 1).bit_length()
                operation_digest = hashlib.sha256(
                    _OPERATION_DOMAIN
                    + static_manifest.manifest_digest
                    + descriptor.canonical_bytes()
                ).digest()
                operations.append(
                    LeanProjectionCatalogOperationV3(
                        operation_key=key,
                        operation_digest=operation_digest,
                        input_dim=descriptor.rows,
                        padded_input_dim=padded_input_dim,
                        output_dim=descriptor.cols,
                        operation_root=descriptor.commitment,
                        registered_catalog_id=catalog_id,
                        _source_path=catalog_operation._source_path,
                        _source_offset=catalog_operation._source_offset,
                        _inline_commitments=(
                            None
                            if catalog_operation._source_path is not None
                            else commitments
                        ),
                    )
                )
            return cls(static_manifest.manifest_digest, tuple(operations))
        except ProofV3Error:
            raise
        except Exception as exc:
            raise ProofV3Error(
                "authenticated lean projection catalog could not be registered"
            ) from exc

    def operation(self, key: object) -> LeanProjectionCatalogOperationV3:
        try:
            return self._by_key[key]
        except (KeyError, TypeError) as exc:
            raise ProofV3VerificationError(
                "lean projection operation is not in the authenticated catalog"
            ) from exc


def derive_lean_projection_coefficients_v3(
    *,
    statement: LeanProjectionStatementV3,
    validator_nonce: bytes,
    input_row_i8,
    surrogate_output_i64,
) -> tuple[tuple[int, ...], ...]:
    """Derive four output-fold vectors after X and S are frozen."""

    words = _derive_lean_projection_coefficient_words_v3(
        statement=statement,
        validator_nonce=validator_nonce,
        input_row_i8=input_row_i8,
        surrogate_output_i64=surrogate_output_i64,
    )
    return tuple(
        tuple(int(value) for value in row)
        for row in words
    )


def _derive_lean_projection_coefficient_words_v3(
    *,
    statement: LeanProjectionStatementV3,
    validator_nonce: bytes,
    input_row_i8,
    surrogate_output_i64,
):
    """Return the canonical contiguous little-endian u31 coefficient array."""

    if not isinstance(statement, LeanProjectionStatementV3):
        raise ProofV3Error("lean projection statement has a wrong type")
    nonce = _fixed32(
        validator_nonce, "lean projection validator nonce", nonzero=True
    )
    x = _signed_i8_row(
        input_row_i8,
        length=statement.input_dim,
        name="lean projection input",
    )
    output = _signed_i64_row(
        surrogate_output_i64,
        length=statement.output_dim,
        name="lean projection surrogate output",
    )
    transcript = (
        _TRANSCRIPT_DOMAIN
        + statement.digest()
        + nonce
        + bytes(value & 0xFF for value in x)
        + struct.pack(f"<{len(output)}q", *output)
    )
    count = LEAN_PROJECTION_FOLD_COUNT_V3 * statement.output_dim
    raw = hashlib.shake_256(transcript).digest(count * 4)
    mask = (1 << LEAN_PROJECTION_COEFFICIENT_BITS_V3) - 1
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - production dependency.
        raise ProofV3Error(
            "lean projection coefficient derivation requires NumPy"
        ) from exc
    return (
        np.frombuffer(raw, dtype="<u4")
        .reshape(LEAN_PROJECTION_FOLD_COUNT_V3, statement.output_dim)
        .__and__(np.uint32(mask))
        .astype("<u4", copy=False)
    )


def build_lean_projection_fold_reference_v3(
    *,
    statement: LeanProjectionStatementV3,
    validator_nonce: bytes,
    input_row_i8,
    surrogate_output_i64,
    weight_columns_i8,
) -> LeanProjectionFoldV3:
    """Reference prover for tests and qualification.

    Production serving replaces this Python matrix fold with a native/GPU
    implementation while preserving the byte-exact signed-i64 result.
    """

    columns = tuple(
        _signed_i8_row(
            column,
            length=statement.padded_input_dim,
            name=f"lean projection weight column {index}",
        )
        for index, column in enumerate(weight_columns_i8)
    )
    if len(columns) != statement.output_dim:
        raise ProofV3Error("lean projection weight column count is wrong")
    coefficients = derive_lean_projection_coefficients_v3(
        statement=statement,
        validator_nonce=validator_nonce,
        input_row_i8=input_row_i8,
        surrogate_output_i64=surrogate_output_i64,
    )
    folds = []
    for fold_coefficients in coefficients:
        row = tuple(
            sum(
                coefficient * columns[column][inner]
                for column, coefficient in enumerate(fold_coefficients)
            )
            for inner in range(statement.padded_input_dim)
        )
        _signed_i64_row(
            row,
            length=statement.padded_input_dim,
            name="lean projection folded weight",
        )
        folds.append(row)
    return LeanProjectionFoldV3(tuple(folds))


def verify_lean_projection_fold_v3(
    *,
    statement: LeanProjectionStatementV3,
    fold: LeanProjectionFoldV3,
    validator_nonce: bytes,
    input_row_i8,
    surrogate_output_i64,
    catalog_id: bytes,
) -> None:
    """Verify complete selected-row projection against signed commitments."""

    try:
        if not isinstance(statement, LeanProjectionStatementV3):
            raise ProofV3VerificationError(
                "lean projection statement has a wrong type"
            )
        if not isinstance(fold, LeanProjectionFoldV3):
            raise ProofV3VerificationError(
                "lean projection fold has a wrong type"
            )
        catalog = _fixed32(
            catalog_id, "lean projection registered catalog id", nonzero=True
        )
        x = _signed_i8_row(
            input_row_i8,
            length=statement.input_dim,
            name="lean projection input",
        )
        output = _signed_i64_row(
            surrogate_output_i64,
            length=statement.output_dim,
            name="lean projection surrogate output",
        )
        if any(
            len(row) != statement.padded_input_dim
            for row in fold.folded_weights
        ):
            raise ProofV3VerificationError(
                "lean projection folded weight width is wrong"
            )
        coefficients = derive_lean_projection_coefficients_v3(
            statement=statement,
            validator_nonce=validator_nonce,
            input_row_i8=x,
            surrogate_output_i64=output,
        )

        from zkllm.crypto.pcs_v2 import (
            combine_registered_catalog_u63,
            commit,
        )

        x_padded = x + (0,) * (statement.padded_input_dim - statement.input_dim)
        for fold_index, (row, row_coefficients) in enumerate(
            zip(fold.folded_weights, coefficients, strict=True)
        ):
            expected_commitment = combine_registered_catalog_u63(
                catalog,
                row_coefficients,
                term_count=statement.output_dim,
            )
            actual_commitment = commit(row, encoding="i64")
            if actual_commitment != expected_commitment:
                raise ProofV3VerificationError(
                    f"lean projection fold {fold_index} is not a fold of "
                    "the authenticated weights"
                )
            left = sum(
                int(value) * int(weight)
                for value, weight in zip(x_padded, row, strict=True)
            ) % PALLAS_SCALAR_MODULUS
            right = sum(
                int(value) * int(coefficient)
                for value, coefficient in zip(
                    output, row_coefficients, strict=True
                )
            ) % PALLAS_SCALAR_MODULUS
            if left != right:
                raise ProofV3VerificationError(
                    f"lean projection fold {fold_index} does not bind the "
                    "complete output row"
                )
    except ProofV3VerificationError:
        raise
    except (OverflowError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "lean projection fold is malformed"
        ) from exc
