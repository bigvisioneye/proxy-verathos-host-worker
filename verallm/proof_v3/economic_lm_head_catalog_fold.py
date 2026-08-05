"""Weightless full-vocabulary LM-head binding over a signed Pallas catalog."""

from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from zkllm.crypto.pcs_v2 import (
    ENCODING_SIGNED_I64,
    MAX_BATCH_COMMITMENTS,
    MAX_BATCH_I8_BYTES,
    MAX_COMBINE_TERMS,
    combine_commitments,
    combine_registered_catalog_u7_batch,
    commit,
    commit_i8_batch,
    register_catalog_commitments,
)

LM_HEAD_CATALOG_FOLD_ABI_V3: Final = (
    "pallas.catalog_lm_head_fold.19x7.i64.logits_digest.v2"
)
LM_HEAD_CATALOG_FOLD_COUNT_V3: Final = 19
LM_HEAD_CATALOG_COEFFICIENT_BITS_V3: Final = 7
MAX_LM_HEAD_CATALOG_FOLD_BYTES_V3: Final = 1 << 20
MAX_LM_HEAD_CATALOG_COLUMNS_V3: Final = 1 << 20
MAX_LM_HEAD_CATALOG_ARTIFACT_BYTES_V3: Final = 33 << 20

_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/LM_HEAD/CATALOG_FOLD/19X7/I64/LOGITS_DIGEST/V2"
)
_LOGITS_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/LM_HEAD/REVEALED_LOGITS/I64/V1"
)
_CATALOG_MAGIC: Final = b"V3LC"
_CATALOG_VERSION: Final = 1
_CATALOG_HEADER: Final = struct.Struct("<4sHII32s")
_U7_TRANSLATION: Final = bytes(value & 0x7F for value in range(256))
_I64_MIN: Final = -(1 << 63)
_I64_MAX: Final = (1 << 63) - 1


def _fixed32(value: bytes, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    return value


def _checked_i64(value: int, name: str) -> int:
    value = int(value)
    if not _I64_MIN <= value <= _I64_MAX:
        raise ProofV3Error(f"{name} exceeds signed 64-bit arithmetic")
    return value


@dataclass(frozen=True, slots=True)
class EconomicLmHeadCatalogBindingV3:
    """Validator-owned view of one authenticated LM-head catalog operation."""

    operation_root: bytes
    hidden_dim: int
    vocab: int
    column_commitments: tuple[bytes, ...]
    registered_catalog_id: bytes | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _fixed32(self.operation_root, "LM-head catalog operation root")
        if (
            isinstance(self.hidden_dim, bool)
            or not isinstance(self.hidden_dim, int)
            or not 0 < self.hidden_dim < 1 << 24
            or isinstance(self.vocab, bool)
            or not isinstance(self.vocab, int)
            or not 0 < self.vocab <= MAX_LM_HEAD_CATALOG_COLUMNS_V3
        ):
            raise ProofV3Error("LM-head catalog dimensions are malformed")
        commitments = tuple(self.column_commitments)
        if len(commitments) != self.vocab or any(
            not isinstance(value, bytes) or len(value) != 32
            for value in commitments
        ):
            raise ProofV3Error(
                "LM-head catalog commitments do not match the vocabulary"
            )
        try:
            from verallm.challenge.v2 import (
                MODEL_LM_HEAD_OPERATION_ID,
                MODEL_OPERATION_LAYER_IDX,
                OperationKeyV2,
                RegisteredOperationV2,
            )
            from verallm.proof_v2.engine import (
                build_weight_commitment_catalog_tree_v2,
            )
            from verallm.proof_v2.layout import MAX_BLOCK_AXIS

            operation = RegisteredOperationV2(
                key=OperationKeyV2(
                    MODEL_OPERATION_LAYER_IDX,
                    MODEL_LM_HEAD_OPERATION_ID,
                    -1,
                ),
                inner_dim=self.hidden_dim,
                output_dim=self.vocab,
                block_rows=MAX_BLOCK_AXIS,
                block_cols=MAX_BLOCK_AXIS,
                weight_commitment_root=self.operation_root,
            )
            build_weight_commitment_catalog_tree_v2(
                operation,
                commitments,
            )
        except Exception as exc:
            raise ProofV3Error(
                "LM-head catalog commitments do not reconstruct the "
                "authenticated operation root"
            ) from exc
        object.__setattr__(self, "column_commitments", commitments)
        if self.vocab > MAX_COMBINE_TERMS:
            try:
                catalog_id, count = register_catalog_commitments(commitments)
            except Exception as exc:
                raise ProofV3Error(
                    "LM-head catalog commitments could not be registered"
                ) from exc
            if count != self.vocab:
                raise ProofV3Error(
                    "registered LM-head catalog has the wrong vocabulary"
                )
            object.__setattr__(self, "registered_catalog_id", catalog_id)


@dataclass(frozen=True, slots=True)
class EconomicLmHeadCatalogArtifactV3:
    """Content-addressed canonical LM-head commitment artifact."""

    hidden_dim: int
    vocab: int
    operation_root: bytes
    column_commitments: tuple[bytes, ...]
    binding: EconomicLmHeadCatalogBindingV3 = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        binding = EconomicLmHeadCatalogBindingV3(
            operation_root=self.operation_root,
            hidden_dim=self.hidden_dim,
            vocab=self.vocab,
            column_commitments=self.column_commitments,
        )
        object.__setattr__(
            self,
            "column_commitments",
            binding.column_commitments,
        )
        object.__setattr__(self, "binding", binding)

    def canonical_bytes(self) -> bytes:
        encoded = _CATALOG_HEADER.pack(
            _CATALOG_MAGIC,
            _CATALOG_VERSION,
            self.hidden_dim,
            self.vocab,
            self.operation_root,
        ) + b"".join(self.column_commitments)
        if len(encoded) > MAX_LM_HEAD_CATALOG_ARTIFACT_BYTES_V3:
            raise ProofV3Error("LM-head catalog artifact exceeds its byte bound")
        return encoded

    def digest(self) -> bytes:
        return hashlib.sha256(self.canonical_bytes()).digest()

    @classmethod
    def from_canonical_bytes(
        cls,
        encoded: bytes,
    ) -> "EconomicLmHeadCatalogArtifactV3":
        if (
            not isinstance(encoded, bytes)
            or not _CATALOG_HEADER.size < len(encoded)
            <= MAX_LM_HEAD_CATALOG_ARTIFACT_BYTES_V3
        ):
            raise ProofV3Error("LM-head catalog artifact size is invalid")
        magic, version, hidden_dim, vocab, operation_root = (
            _CATALOG_HEADER.unpack_from(encoded)
        )
        if magic != _CATALOG_MAGIC or version != _CATALOG_VERSION:
            raise ProofV3Error("LM-head catalog artifact header is unsupported")
        expected = _CATALOG_HEADER.size + vocab * 32
        if len(encoded) != expected:
            raise ProofV3Error("LM-head catalog artifact length is not canonical")
        commitments = tuple(
            encoded[offset : offset + 32]
            for offset in range(_CATALOG_HEADER.size, len(encoded), 32)
        )
        result = cls(
            hidden_dim=hidden_dim,
            vocab=vocab,
            operation_root=operation_root,
            column_commitments=commitments,
        )
        if result.canonical_bytes() != encoded:
            raise ProofV3Error("LM-head catalog artifact is not canonical")
        return result

    @classmethod
    def load(cls, path) -> "EconomicLmHeadCatalogArtifactV3":
        try:
            encoded = Path(path).read_bytes()
        except OSError as exc:
            raise ProofV3Error("LM-head catalog artifact could not be loaded") from exc
        return cls.from_canonical_bytes(encoded)


def build_lm_head_catalog_artifact_v3(
    *,
    packed_int8_rows: bytes,
    hidden_dim: int,
    vocab: int,
) -> EconomicLmHeadCatalogArtifactV3:
    """Build the one-time compact catalog from canonical ``[vocab, hidden]`` rows."""

    if (
        not isinstance(packed_int8_rows, bytes)
        or isinstance(hidden_dim, bool)
        or not isinstance(hidden_dim, int)
        or not 0 < hidden_dim < 1 << 24
        or isinstance(vocab, bool)
        or not isinstance(vocab, int)
        or not 0 < vocab <= MAX_LM_HEAD_CATALOG_COLUMNS_V3
        or len(packed_int8_rows) != hidden_dim * vocab
    ):
        raise ProofV3Error("LM-head catalog source matrix is malformed")
    try:
        rows_per_batch = min(
            MAX_BATCH_COMMITMENTS,
            MAX_BATCH_I8_BYTES // hidden_dim,
        )
        if rows_per_batch < 1:
            raise ProofV3Error(
                "LM-head catalog rows exceed the native batch byte bound"
            )
        batch_bytes = rows_per_batch * hidden_dim
        commitments = tuple(
            commitment
            for offset in range(0, len(packed_int8_rows), batch_bytes)
            for commitment in commit_i8_batch(
                packed_int8_rows[offset : offset + batch_bytes],
                vector_length=hidden_dim,
            )
        )
        from verallm.challenge.v2 import (
            MODEL_LM_HEAD_OPERATION_ID,
            MODEL_OPERATION_LAYER_IDX,
            OperationKeyV2,
            RegisteredOperationV2,
        )
        from verallm.proof_v2.engine import build_weight_commitment_tree_v2
        from verallm.proof_v2.layout import MAX_BLOCK_AXIS

        operation = RegisteredOperationV2(
            key=OperationKeyV2(
                MODEL_OPERATION_LAYER_IDX,
                MODEL_LM_HEAD_OPERATION_ID,
                -1,
            ),
            inner_dim=hidden_dim,
            output_dim=vocab,
            block_rows=MAX_BLOCK_AXIS,
            block_cols=MAX_BLOCK_AXIS,
            weight_commitment_root=b"",
        )
        operation_root = build_weight_commitment_tree_v2(
            operation,
            commitments,
            expected_root=b"",
        ).root
        return EconomicLmHeadCatalogArtifactV3(
            hidden_dim=hidden_dim,
            vocab=vocab,
            operation_root=operation_root,
            column_commitments=commitments,
        )
    except ProofV3Error:
        raise
    except Exception as exc:
        raise ProofV3Error(
            "LM-head catalog commitments could not be generated"
        ) from exc


def derive_lm_head_catalog_coefficients_v3(
    *,
    selection_seed: bytes,
    envelope_digest: bytes,
    manifest_digest: bytes,
    operation_root: bytes,
    revealed_logits_digest: bytes,
    audited_position: int,
    vocab: int,
) -> bytes:
    """Derive fold coefficients after binding the complete disclosed logits.

    The post-nonce prover first freezes the one-row vector through
    ``revealed_logits_digest``.  Fiat--Shamir coefficients then depend on that
    digest, so the vector cannot be adapted to already-known linear checks.
    """

    for value, name in (
        (selection_seed, "selection seed"),
        (envelope_digest, "envelope digest"),
        (manifest_digest, "manifest digest"),
        (operation_root, "operation root"),
        (revealed_logits_digest, "revealed logits digest"),
    ):
        _fixed32(value, name)
    if (
        isinstance(audited_position, bool)
        or not isinstance(audited_position, int)
        or not 0 <= audited_position < 1 << 32
        or isinstance(vocab, bool)
        or not isinstance(vocab, int)
        or not 0 < vocab < 1 << 24
    ):
        raise ProofV3Error("LM-head fold geometry is malformed")
    seed = hashlib.sha256(
        _DOMAIN
        + selection_seed
        + envelope_digest
        + manifest_digest
        + operation_root
        + revealed_logits_digest
        + struct.pack("<II", audited_position, vocab)
    ).digest()
    return hashlib.shake_256(seed).digest(
        LM_HEAD_CATALOG_FOLD_COUNT_V3 * vocab
    ).translate(_U7_TRANSLATION)


def lm_head_logits_digest_v3(revealed_logits) -> bytes:
    """Canonical digest of one complete signed-i64 vocabulary row."""

    try:
        logits = tuple(int(value) for value in revealed_logits)
    except (TypeError, ValueError) as exc:
        raise ProofV3Error("revealed LM-head logits are malformed") from exc
    if not logits or len(logits) >= 1 << 24 or any(
        not _I64_MIN <= value <= _I64_MAX for value in logits
    ):
        raise ProofV3Error("revealed LM-head logits are malformed")
    digest = hashlib.sha256()
    digest.update(_LOGITS_DOMAIN)
    digest.update(struct.pack("<I", len(logits)))
    for start in range(0, len(logits), 8_192):
        chunk = logits[start : start + 8_192]
        digest.update(struct.pack(f"<{len(chunk)}q", *chunk))
    return digest.digest()


def _coefficient_rows(
    packed: bytes,
    *,
    vocab: int,
) -> tuple[bytes, ...]:
    expected = LM_HEAD_CATALOG_FOLD_COUNT_V3 * vocab
    if (
        not isinstance(packed, bytes)
        or len(packed) != expected
        or max(packed, default=0)
        >= 1 << LM_HEAD_CATALOG_COEFFICIENT_BITS_V3
    ):
        raise ProofV3Error("LM-head coefficient matrix is malformed")
    return tuple(
        packed[offset : offset + vocab]
        for offset in range(0, len(packed), vocab)
    )


def combine_lm_head_catalog_commitments_v3(
    binding: EconomicLmHeadCatalogBindingV3,
    packed_coefficients: bytes,
) -> tuple[bytes, ...]:
    """Compute every expected folded commitment from validator-owned points."""

    rows = _coefficient_rows(packed_coefficients, vocab=binding.vocab)
    if binding.vocab > MAX_COMBINE_TERMS:
        if binding.registered_catalog_id is None:
            raise ProofV3Error("large LM-head catalog is not registered")
        from zkllm.crypto.pcs_v2 import registered_catalog_use

        with registered_catalog_use():
            catalog_id, count = register_catalog_commitments(
                binding.column_commitments
            )
            if (
                count != binding.vocab
                or catalog_id != binding.registered_catalog_id
            ):
                raise ProofV3Error(
                    "LM-head catalog registration does not match"
                )
            return combine_registered_catalog_u7_batch(
                binding.registered_catalog_id,
                packed_coefficients,
                term_count=binding.vocab,
                fold_count=LM_HEAD_CATALOG_FOLD_COUNT_V3,
            )
    return tuple(
        combine_commitments(binding.column_commitments, tuple(row))
        for row in rows
    )


def build_lm_head_catalog_folds_reference_v3(
    *,
    lm_head_rows,
    packed_coefficients: bytes,
) -> tuple[bytes, ...]:
    """Small-model reference builder for exact signed-i64 folded weights."""

    rows = tuple(tuple(int(value) for value in row) for row in lm_head_rows)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ProofV3Error("LM-head rows are ragged or empty")
    if any(not -128 <= value <= 127 for row in rows for value in row):
        raise ProofV3Error("LM-head rows are not signed int8")
    coefficients = _coefficient_rows(
        packed_coefficients,
        vocab=len(rows),
    )
    hidden_dim = len(rows[0])
    packed = []
    for vector in coefficients:
        folded = tuple(
            _checked_i64(
                sum(
                    vector[column] * rows[column][inner]
                    for column in range(len(rows))
                ),
                "LM-head folded weight",
            )
            for inner in range(hidden_dim)
        )
        encoded = struct.pack(f"<{hidden_dim}q", *folded)
        if len(encoded) > MAX_LM_HEAD_CATALOG_FOLD_BYTES_V3:
            raise ProofV3Error("LM-head folded vector exceeds the wire bound")
        packed.append(encoded)
    return tuple(packed)


def build_lm_head_catalog_folds_cuda_v3(
    *,
    lm_head_rows,
    packed_coefficients: bytes,
) -> tuple[bytes, ...]:
    """Production CUDA builder using two overflow-safe int8 GEMM chunks."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production dependency.
        raise ProofV3Error("CUDA LM-head folding requires torch") from exc
    if (
        not isinstance(lm_head_rows, torch.Tensor)
        or lm_head_rows.ndim != 2
        or lm_head_rows.dtype != torch.int8
        or not lm_head_rows.is_cuda
        or not lm_head_rows.is_contiguous()
    ):
        raise ProofV3Error(
            "production LM-head folds require contiguous CUDA int8 rows"
        )
    vocab, hidden_dim = map(int, lm_head_rows.shape)
    _coefficient_rows(packed_coefficients, vocab=vocab)
    coefficients = torch.frombuffer(
        bytearray(packed_coefficients),
        dtype=torch.uint8,
    ).view(LM_HEAD_CATALOG_FOLD_COUNT_V3, vocab)
    # torch._int_mm requires a 32-row left operand on the A100 CUDA 13
    # cublasLt path used by the supported torch 2.11 stack.  Padding only to
    # eight rows (19 -> 24) works on some older stacks but fails there with
    # CUBLAS_STATUS_NOT_SUPPORTED.  Zero rows preserve every fold exactly.
    padding_rows = (-LM_HEAD_CATALOG_FOLD_COUNT_V3) % 32
    if padding_rows:
        coefficients = torch.cat(
            (
                coefficients,
                torch.zeros((padding_rows, vocab), dtype=torch.uint8),
            ),
            dim=0,
        )
    coefficients = coefficients.to(
        device=lm_head_rows.device,
        dtype=torch.int8,
    )
    folded = torch.zeros(
        (int(coefficients.shape[0]), hidden_dim),
        dtype=torch.int64,
        device=lm_head_rows.device,
    )
    # 2^17 * 127 * 127 < 2^31, so every _int_mm accumulator is exact.
    for start in range(0, vocab, 1 << 17):
        stop = min(start + (1 << 17), vocab)
        coefficient_chunk = coefficients[:, start:stop].contiguous()
        weight_chunk = lm_head_rows[start:stop]
        tail_padding = (-int(weight_chunk.shape[0])) % 8
        if tail_padding:
            coefficient_chunk = torch.nn.functional.pad(
                coefficient_chunk, (0, tail_padding)
            )
            weight_chunk = torch.nn.functional.pad(
                weight_chunk, (0, 0, 0, tail_padding)
            )
        folded += torch._int_mm(
            coefficient_chunk,
            weight_chunk,
        ).to(torch.int64)
    folded = (
        folded[:LM_HEAD_CATALOG_FOLD_COUNT_V3]
        .cpu()
        .contiguous()
        .numpy()
        .astype("<i8", copy=False)
    )
    return tuple(row.tobytes() for row in folded)


def build_lm_head_logits_v3(
    *,
    lm_head_rows,
    hidden_row_int8,
    chunk_rows: int = 8_192,
) -> tuple[int, ...]:
    """Compute one exact LM-head row with bounded device transients.

    Production keeps the authenticated int8 weight catalog on CUDA.  This
    routine never transposes or materializes the complete vocabulary matrix:
    it processes bounded row chunks and copies only the resulting int32
    logits to host.  CPU/list inputs retain a small-model reference path.
    """

    if (
        isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or not 8 <= chunk_rows <= 1 << 17
    ):
        raise ProofV3Error("LM-head logit chunk size is malformed")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production dependency.
        raise ProofV3Error("LM-head logit construction requires torch") from exc

    hidden = tuple(int(value) for value in hidden_row_int8)
    if not hidden or any(not -128 <= value <= 127 for value in hidden):
        raise ProofV3Error("LM-head hidden row is not signed int8")
    if isinstance(lm_head_rows, torch.Tensor):
        if (
            lm_head_rows.ndim != 2
            or int(lm_head_rows.shape[1]) != len(hidden)
            or lm_head_rows.dtype != torch.int8
        ):
            raise ProofV3Error("LM-head compute rows are malformed")
        if lm_head_rows.is_cuda:
            if len(hidden) * 128 * 128 > int(torch.iinfo(torch.int32).max):
                raise ProofV3Error(
                    "LM-head dot product exceeds exact int32 accumulation"
                )
            x = torch.as_tensor(
                hidden,
                dtype=torch.int8,
                device=lm_head_rows.device,
            ).reshape(1, -1)
            row_padding = 31
            x = torch.nn.functional.pad(x, (0, 0, 0, row_padding))
            output: list[int] = []
            vocab = int(lm_head_rows.shape[0])
            for start in range(0, vocab, chunk_rows):
                stop = min(start + chunk_rows, vocab)
                weights = lm_head_rows[start:stop]
                tail_padding = (-int(weights.shape[0])) % 8
                if tail_padding:
                    weights = torch.nn.functional.pad(
                        weights,
                        (0, 0, 0, tail_padding),
                    )
                values = torch._int_mm(
                    x,
                    weights.t().contiguous(),
                )[0, : stop - start]
                output.extend(
                    int(value)
                    for value in values.cpu().tolist()
                )
            return tuple(output)
        rows = lm_head_rows.to(dtype=torch.int64)
        x = torch.as_tensor(hidden, dtype=torch.int64)
        return tuple(int(value) for value in (rows @ x).tolist())

    try:
        rows = tuple(
            tuple(int(value) for value in row)
            for row in lm_head_rows
        )
    except (TypeError, ValueError) as exc:
        raise ProofV3Error("LM-head compute rows are malformed") from exc
    if (
        not rows
        or any(len(row) != len(hidden) for row in rows)
        or any(not -128 <= value <= 127 for row in rows for value in row)
    ):
        raise ProofV3Error("LM-head compute rows are malformed")
    return tuple(
        _checked_i64(
            sum(a * b for a, b in zip(row, hidden, strict=True)),
            "LM-head logit",
        )
        for row in rows
    )


def _unpack_folds(
    folded_weights,
    *,
    hidden_dim: int,
) -> tuple[tuple[int, ...], ...]:
    folds = tuple(folded_weights)
    expected_bytes = hidden_dim * 8
    if len(folds) != LM_HEAD_CATALOG_FOLD_COUNT_V3 or any(
        not isinstance(value, bytes)
        or len(value) != expected_bytes
        or len(value) > MAX_LM_HEAD_CATALOG_FOLD_BYTES_V3
        for value in folds
    ):
        raise ProofV3VerificationError(
            "LM-head catalog folds have a wrong shape"
        )
    return tuple(
        struct.unpack(f"<{hidden_dim}q", packed)
        for packed in folds
    )


def verify_lm_head_catalog_folds_v3(
    *,
    binding: EconomicLmHeadCatalogBindingV3,
    folded_weights,
    hidden_row_int8,
    revealed_logits,
    selection_seed: bytes,
    envelope_digest: bytes,
    manifest_digest: bytes,
    audited_position: int,
) -> None:
    """Verify all revealed logits against a signed catalog without weights."""

    if not isinstance(binding, EconomicLmHeadCatalogBindingV3):
        raise ProofV3VerificationError(
            "LM-head catalog binding has an unexpected type"
        )
    hidden = tuple(int(value) for value in hidden_row_int8)
    logits = tuple(int(value) for value in revealed_logits)
    if len(hidden) != binding.hidden_dim or any(
        not -128 <= value <= 127 for value in hidden
    ):
        raise ProofV3VerificationError(
            "audited hidden row does not match the LM-head catalog"
        )
    if len(logits) != binding.vocab or any(
        not _I64_MIN <= value <= _I64_MAX for value in logits
    ):
        raise ProofV3VerificationError(
            "revealed logits do not match the LM-head catalog"
        )
    values = _unpack_folds(
        folded_weights,
        hidden_dim=binding.hidden_dim,
    )
    coefficients = derive_lm_head_catalog_coefficients_v3(
        selection_seed=selection_seed,
        envelope_digest=envelope_digest,
        manifest_digest=manifest_digest,
        operation_root=binding.operation_root,
        revealed_logits_digest=lm_head_logits_digest_v3(logits),
        audited_position=audited_position,
        vocab=binding.vocab,
    )
    try:
        disclosed = tuple(
            commit(vector, encoding=ENCODING_SIGNED_I64)
            for vector in values
        )
        expected = combine_lm_head_catalog_commitments_v3(
            binding,
            coefficients,
        )
    except Exception as exc:
        raise ProofV3VerificationError(
            "LM-head catalog folds are malformed"
        ) from exc
    if len(expected) != len(disclosed) or any(
        not hmac.compare_digest(actual, wanted)
        for actual, wanted in zip(disclosed, expected, strict=True)
    ):
        raise ProofV3VerificationError(
            "LM-head catalog folds do not open the authenticated weights"
        )

    import numpy as np

    coefficient_matrix = np.frombuffer(
        coefficients,
        dtype=np.uint8,
    ).reshape(LM_HEAD_CATALOG_FOLD_COUNT_V3, binding.vocab)
    logits_array = np.asarray(logits, dtype=np.int64)
    values_array = np.asarray(values, dtype=np.int64)
    hidden_array = np.asarray(hidden, dtype=np.int64)

    def _safe_dot(matrix, vector, *, width: int):
        if (
            np.any(matrix == _I64_MIN)
            or np.any(vector == _I64_MIN)
        ):
            return None
        matrix_max = int(np.abs(matrix).max(initial=0))
        vector_max = int(np.abs(vector).max(initial=0))
        if matrix_max * vector_max * width > _I64_MAX:
            return None
        return matrix @ vector

    logits_folds = _safe_dot(
        coefficient_matrix.astype(np.int64, copy=False),
        logits_array,
        width=binding.vocab,
    )
    weight_folds = _safe_dot(
        values_array,
        hidden_array,
        width=binding.hidden_dim,
    )
    if logits_folds is None or weight_folds is None:
        coefficient_rows = _coefficient_rows(
            coefficients,
            vocab=binding.vocab,
        )
        logits_folds = tuple(
            _checked_i64(
                sum(
                    coefficient * logit
                    for coefficient, logit in zip(
                        coefficient_row,
                        logits,
                        strict=True,
                    )
                ),
                "folded logits",
            )
            for coefficient_row in coefficient_rows
        )
        weight_folds = tuple(
            _checked_i64(
                sum(
                    value * hidden_value
                    for value, hidden_value in zip(
                        vector,
                        hidden,
                        strict=True,
                    )
                ),
                "folded hidden @ weight",
            )
            for vector in values
        )

    for fold_index, (logits_fold, weight_fold) in enumerate(
        zip(logits_folds, weight_folds, strict=True)
    ):
        if int(logits_fold) != int(weight_fold):
            raise ProofV3VerificationError(
                f"LM-head catalog fold {fold_index} does not bind the "
                "revealed logits to hidden @ registered weights"
            )


__all__ = [
    "EconomicLmHeadCatalogBindingV3",
    "EconomicLmHeadCatalogArtifactV3",
    "LM_HEAD_CATALOG_COEFFICIENT_BITS_V3",
    "LM_HEAD_CATALOG_FOLD_ABI_V3",
    "LM_HEAD_CATALOG_FOLD_COUNT_V3",
    "MAX_LM_HEAD_CATALOG_FOLD_BYTES_V3",
    "MAX_LM_HEAD_CATALOG_ARTIFACT_BYTES_V3",
    "build_lm_head_catalog_artifact_v3",
    "build_lm_head_catalog_folds_cuda_v3",
    "build_lm_head_catalog_folds_reference_v3",
    "build_lm_head_logits_v3",
    "combine_lm_head_catalog_commitments_v3",
    "derive_lm_head_catalog_coefficients_v3",
    "lm_head_logits_digest_v3",
    "verify_lm_head_catalog_folds_v3",
]
