"""Canonical wire payloads for inference proof protocol v2."""

from __future__ import annotations

import struct
from dataclasses import dataclass

from verallm.challenge.v2 import (
    OperationKeyV2,
    ProofBlockDescriptorV2,
    RuntimeYCommitmentV2,
    XCommitmentV2,
)
from verallm.proof_v2.layout import RUNTIME_Y_COMMITMENT_BLOCK_COLS
from verallm.proof_v2.trace import (
    MAX_TRACE_PROOF_BYTES,
    ExecutionTraceCommitmentV2,
    ExecutionTraceProofV2,
    ProofV2TraceError,
)
from verallm.proof_v2.transition import (
    ProofV2TransitionError,
    TransitionHistoryCommitmentV2,
)
from zkllm.crypto.gemm_v2_batch import GemmV2BatchProof
from zkllm.crypto.gemm_v2_reference import scalar_from_bytes
from zkllm.crypto.pcs_merkle_v2 import (
    PCS_ENCODING_HASHED_FP16_BLOCK,
    PCS_ENCODING_HASHED_QUANTIZED_I8_ROW,
    PcsMerkleV2MultiProof,
    hash_fp16_block_v2,
    hash_quantized_i8_row_v2,
)
from zkllm.crypto.pcs_merkle_v2 import (
    PCS_ENCODING_SIGNED_I8 as MERKLE_ENCODING_SIGNED_I8,
)
from zkllm.crypto.pcs_v2 import (
    ENCODING_PALLAS_SCALAR,
    MAX_VECTOR_LEN,
    PCSOpeningV2,
)

PROTOCOL_VERSION = 2
MAX_COMMITMENT_OPERATIONS = 100_000
MAX_BLOCK_PROOFS = 4_096
MAX_MEMBERSHIP_PROOF_BYTES = 1 << 20
MAX_X_ROWS_BYTES = 2 << 20
MAX_X_SCALES_BYTES = 1 << 20
MAX_RUNTIME_Y_BYTES = 8 << 20
MAX_PROOF_Y_BYTES = 16 << 20
MAX_SUMCHECK_PROOF_BYTES = 1 << 20
MAX_IPA_PROOF_BYTES = 2_048
MAX_COMMITMENT_ENVELOPE_BYTES = 16 << 20
MAX_FINAL_PROOF_BYTES = 32 << 20

_COMMITMENT_MAGIC = b"V2CM"
_PROOF_MAGIC = b"V2PF"


class ProofV2PayloadError(ValueError):
    """A proof-v2 wire payload is malformed or noncanonical."""


def _checked_u32(value: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < (1 << 32)
    ):
        raise ProofV2PayloadError(f"{name} must be an unsigned 32-bit integer")
    return value


def _fixed(value: bytes, size: int, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != size:
        raise ProofV2PayloadError(f"{name} must be exactly {size} bytes")
    return value


def _bounded(
    value: bytes, maximum: int, name: str, *, allow_empty: bool = False
) -> bytes:
    if not isinstance(value, bytes):
        raise ProofV2PayloadError(f"{name} must be bytes")
    if (not value and not allow_empty) or len(value) > maximum:
        raise ProofV2PayloadError(f"{name} length is out of range")
    return value


def _power_of_two(value: int, name: str) -> int:
    value = _checked_u32(value, name)
    if value == 0 or value & (value - 1):
        raise ProofV2PayloadError(f"{name} must be a positive power of two")
    return value


def _encode_key(key: OperationKeyV2) -> bytes:
    if not isinstance(key, OperationKeyV2):
        raise ProofV2PayloadError("operation key has an unexpected type")
    key.canonical_bytes()
    operation_id = key.operation_id.encode("ascii")
    return (
        struct.pack("<IiH", key.layer_idx, key.expert_idx, len(operation_id))
        + operation_id
    )


def _encode_descriptor(descriptor: ProofBlockDescriptorV2) -> bytes:
    _validate_descriptor(descriptor)
    return _encode_key(descriptor.key) + struct.pack(
        "<11I",
        descriptor.block_row,
        descriptor.block_col,
        descriptor.row_offset,
        descriptor.column_offset,
        descriptor.rows,
        descriptor.inner_dim,
        descriptor.padded_inner_dim,
        descriptor.cols,
        descriptor.row_rounds,
        descriptor.inner_rounds,
        descriptor.col_rounds,
    )


def _validate_descriptor(descriptor: ProofBlockDescriptorV2) -> None:
    if not isinstance(descriptor, ProofBlockDescriptorV2):
        raise ProofV2PayloadError("block descriptor has an unexpected type")
    descriptor.key.canonical_bytes()
    for name in ("block_row", "block_col", "row_offset", "column_offset"):
        _checked_u32(getattr(descriptor, name), name)
    rows = _power_of_two(descriptor.rows, "rows")
    inner = _checked_u32(descriptor.inner_dim, "inner_dim")
    padded_inner = _power_of_two(descriptor.padded_inner_dim, "padded_inner_dim")
    columns = _power_of_two(descriptor.cols, "cols")
    if (
        inner == 0
        or inner > padded_inner
        or padded_inner != 1 << (inner - 1).bit_length()
    ):
        raise ProofV2PayloadError("padded_inner_dim is not canonical for inner_dim")
    expected_rounds = (
        rows.bit_length() - 1,
        padded_inner.bit_length() - 1,
        columns.bit_length() - 1,
    )
    actual_rounds = (
        _checked_u32(descriptor.row_rounds, "row_rounds"),
        _checked_u32(descriptor.inner_rounds, "inner_rounds"),
        _checked_u32(descriptor.col_rounds, "col_rounds"),
    )
    if actual_rounds != expected_rounds:
        raise ProofV2PayloadError("block round counts do not match its dimensions")


def _expected_ipa_size(vector_length: int) -> int:
    length = _power_of_two(vector_length, "PCS vector length")
    if length > MAX_VECTOR_LEN:
        raise ProofV2PayloadError("PCS vector length exceeds the protocol limit")
    return 2 + 64 * (length.bit_length() - 1) + 32


def _validate_opening(
    opening: PCSOpeningV2,
    *,
    expected_length: int,
    expected_encoding: int,
    name: str,
) -> None:
    if not isinstance(opening, PCSOpeningV2):
        raise ProofV2PayloadError(f"{name} opening has an unexpected type")
    length = _power_of_two(opening.vector_length, f"{name} vector_length")
    if length != expected_length or opening.padded_length != length:
        raise ProofV2PayloadError(f"{name} opening dimensions do not match the block")
    if isinstance(opening.encoding, bool) or opening.encoding != expected_encoding:
        raise ProofV2PayloadError(f"{name} opening encoding is not supported")
    _fixed(opening.commitment, 32, f"{name} commitment")
    try:
        scalar_from_bytes(opening.evaluation)
    except ValueError as exc:
        raise ProofV2PayloadError(f"{name} evaluation is not canonical") from exc
    proof = _bounded(opening.proof, MAX_IPA_PROOF_BYTES, f"{name} IPA proof")
    if len(proof) != _expected_ipa_size(length):
        raise ProofV2PayloadError(f"{name} IPA proof length is not canonical")


def _encode_opening(opening: PCSOpeningV2) -> bytes:
    return (
        opening.commitment
        + opening.evaluation
        + struct.pack(
            "<IIBH",
            opening.vector_length,
            opening.padded_length,
            opening.encoding,
            len(opening.proof),
        )
        + opening.proof
    )


@dataclass(frozen=True)
class ProofV2CommitmentEnvelope:
    """Pre-challenge arithmetic and causal-execution commitments."""

    manifest_digest: bytes
    x_commitments: tuple[XCommitmentV2, ...]
    runtime_y_commitments: tuple[RuntimeYCommitmentV2, ...]
    execution_trace_commitment: ExecutionTraceCommitmentV2 | None = None
    transition_history_commitment: TransitionHistoryCommitmentV2 | None = None

    def __post_init__(self) -> None:
        _fixed(self.manifest_digest, 32, "manifest_digest")
        if self.execution_trace_commitment is not None and not isinstance(
            self.execution_trace_commitment,
            ExecutionTraceCommitmentV2,
        ):
            raise ProofV2PayloadError(
                "execution trace commitment has an unexpected type"
            )
        if self.transition_history_commitment is not None and not isinstance(
            self.transition_history_commitment,
            TransitionHistoryCommitmentV2,
        ):
            raise ProofV2PayloadError(
                "transition history commitment has an unexpected type"
            )
        commitments = tuple(self.x_commitments)
        if not commitments or len(commitments) > MAX_COMMITMENT_OPERATIONS:
            raise ProofV2PayloadError("X commitment count is out of range")
        if not all(isinstance(item, XCommitmentV2) for item in commitments):
            raise ProofV2PayloadError("X commitments have an unexpected type")
        keys = [item.key for item in commitments]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ProofV2PayloadError(
                "X commitments must have sorted unique operation keys"
            )
        for item in commitments:
            item.canonical_bytes()
            if item.row_count <= 0 or item.inner_dim <= 0:
                raise ProofV2PayloadError("X commitment dimensions must be positive")
        runtime_y = tuple(self.runtime_y_commitments)
        if len(runtime_y) != len(commitments) or not all(
            isinstance(item, RuntimeYCommitmentV2) for item in runtime_y
        ):
            raise ProofV2PayloadError("runtime Y commitment set is not exact")
        y_keys = [item.key for item in runtime_y]
        if y_keys != sorted(y_keys) or len(y_keys) != len(set(y_keys)):
            raise ProofV2PayloadError(
                "runtime Y commitments must have sorted unique operation keys"
            )
        if y_keys != keys:
            raise ProofV2PayloadError(
                "X and runtime Y commitments must cover the same operation keys"
            )
        for x_item, y_item in zip(commitments, runtime_y):
            y_item.canonical_bytes()
            if (
                y_item.row_count != x_item.row_count
                or y_item.output_dim <= 0
                or y_item.block_rows <= 0
                or y_item.block_cols <= 0
            ):
                raise ProofV2PayloadError(
                    "runtime Y commitment dimensions are not canonical"
                )
        object.__setattr__(self, "x_commitments", commitments)
        object.__setattr__(self, "runtime_y_commitments", runtime_y)

    def canonical_bytes(self) -> bytes:
        trace_commitment = (
            b""
            if self.execution_trace_commitment is None
            else self.execution_trace_commitment.canonical_bytes()
        )
        transition_commitment = (
            b""
            if self.transition_history_commitment is None
            else self.transition_history_commitment.canonical_bytes()
        )
        encoded = bytearray(
            struct.pack(
                "<4sHIIII",
                _COMMITMENT_MAGIC,
                PROTOCOL_VERSION,
                len(self.x_commitments),
                len(self.runtime_y_commitments),
                len(trace_commitment),
                len(transition_commitment),
            )
        )
        encoded.extend(self.manifest_digest)
        encoded.extend(trace_commitment)
        encoded.extend(transition_commitment)
        for item in self.x_commitments:
            encoded.extend(_encode_key(item.key))
            encoded.extend(struct.pack("<II", item.row_count, item.inner_dim))
            encoded.extend(item.row_commitment_root)
        for item in self.runtime_y_commitments:
            encoded.extend(_encode_key(item.key))
            encoded.extend(
                struct.pack(
                    "<IIII",
                    item.row_count,
                    item.output_dim,
                    item.block_rows,
                    item.block_cols,
                )
            )
            encoded.extend(item.block_commitment_root)
        if len(encoded) > MAX_COMMITMENT_ENVELOPE_BYTES:
            raise ProofV2PayloadError("commitment envelope exceeds the protocol limit")
        return bytes(encoded)

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "ProofV2CommitmentEnvelope":
        reader = _Reader(encoded, MAX_COMMITMENT_ENVELOPE_BYTES, "commitment envelope")
        (
            magic,
            version,
            count,
            runtime_y_count,
            trace_length,
            transition_length,
        ) = reader.unpack("<4sHIIII")
        if magic != _COMMITMENT_MAGIC or version != PROTOCOL_VERSION:
            raise ProofV2PayloadError("commitment envelope header is not supported")
        if not 0 < count <= MAX_COMMITMENT_OPERATIONS:
            raise ProofV2PayloadError("X commitment count is out of range")
        if runtime_y_count != count:
            raise ProofV2PayloadError("runtime Y commitment count is not exact")
        if trace_length > 512:
            raise ProofV2PayloadError(
                "execution trace commitment length is out of range"
            )
        if transition_length > 512:
            raise ProofV2PayloadError(
                "transition history commitment length is out of range"
            )
        digest = reader.read(32)
        trace_commitment = None
        if trace_length:
            try:
                trace_commitment = ExecutionTraceCommitmentV2.from_canonical_bytes(
                    reader.read(trace_length)
                )
            except ProofV2TraceError as exc:
                raise ProofV2PayloadError(
                    "execution trace commitment is malformed"
                ) from exc
        transition_commitment = None
        if transition_length:
            try:
                transition_commitment = (
                    TransitionHistoryCommitmentV2.from_canonical_bytes(
                        reader.read(transition_length)
                    )
                )
            except ProofV2TransitionError as exc:
                raise ProofV2PayloadError(
                    "transition history commitment is malformed"
                ) from exc
        commitments = []
        for _ in range(count):
            key = reader.read_key()
            row_count, inner_dim = reader.unpack("<II")
            commitments.append(
                XCommitmentV2(key, row_count, inner_dim, reader.read(32))
            )
        runtime_y_commitments = []
        for _ in range(runtime_y_count):
            key = reader.read_key()
            row_count, output_dim, block_rows, block_cols = reader.unpack("<IIII")
            runtime_y_commitments.append(
                RuntimeYCommitmentV2(
                    key,
                    row_count,
                    output_dim,
                    block_rows,
                    block_cols,
                    reader.read(32),
                )
            )
        reader.finish()
        result = cls(
            digest,
            tuple(commitments),
            tuple(runtime_y_commitments),
            trace_commitment,
            transition_commitment,
        )
        if result.canonical_bytes() != encoded:
            raise ProofV2PayloadError("commitment envelope is not canonical")
        return result


@dataclass(frozen=True)
class GemmBlockProofV2:
    """One exact selected block's authenticated data and membership paths."""

    descriptor: ProofBlockDescriptorV2
    x_membership_proof: bytes
    x_rows: bytes
    x_scales_q32: bytes
    runtime_y_membership_proof: bytes
    runtime_y_values: bytes
    proof_y_values: bytes
    w_membership_proof: bytes
    proof_y_commitment: bytes

    def __post_init__(self) -> None:
        _validate_descriptor(self.descriptor)
        x_membership = _bounded(
            self.x_membership_proof,
            MAX_MEMBERSHIP_PROOF_BYTES,
            "X membership proof",
        )
        x_rows = _bounded(self.x_rows, MAX_X_ROWS_BYTES, "X row data")
        expected_x_bytes = self.descriptor.rows * self.descriptor.inner_dim
        if len(x_rows) != expected_x_bytes:
            raise ProofV2PayloadError(
                "X row data length does not match the challenged block"
            )
        x_scales = _bounded(
            self.x_scales_q32,
            MAX_X_SCALES_BYTES,
            "X scale data",
        )
        if len(x_scales) != self.descriptor.rows * 8:
            raise ProofV2PayloadError(
                "X scale data length does not match the challenged block"
            )
        scales = struct.unpack(f"<{self.descriptor.rows}Q", x_scales)
        if any(scale == 0 for scale in scales):
            raise ProofV2PayloadError("X scales must be positive Q32 values")
        runtime_y_membership = _bounded(
            self.runtime_y_membership_proof,
            MAX_MEMBERSHIP_PROOF_BYTES,
            "runtime Y membership proof",
        )
        runtime_y_values = _bounded(
            self.runtime_y_values,
            MAX_RUNTIME_Y_BYTES,
            "runtime Y data",
        )
        expected_runtime_y_values = (
            self.descriptor.rows * RUNTIME_Y_COMMITMENT_BLOCK_COLS
        )
        if len(runtime_y_values) != expected_runtime_y_values * 2:
            raise ProofV2PayloadError(
                "runtime Y data length does not match the commitment segment"
            )
        try:
            import numpy as np

            runtime_values = np.frombuffer(runtime_y_values, dtype="<f2")
            if not np.isfinite(runtime_values).all():
                raise ProofV2PayloadError("runtime Y contains a non-finite value")
        except ImportError as exc:  # pragma: no cover - numpy is a runtime dependency
            raise ProofV2PayloadError(
                "numpy is required for proof-v2 decoding"
            ) from exc
        proof_y_values = _bounded(
            self.proof_y_values,
            MAX_PROOF_Y_BYTES,
            "proof Y data",
        )
        expected_y_values = self.descriptor.rows * self.descriptor.cols
        if len(proof_y_values) != expected_y_values * 8:
            raise ProofV2PayloadError(
                "proof Y data length does not match the challenged block"
            )
        w_membership = _bounded(
            self.w_membership_proof,
            MAX_MEMBERSHIP_PROOF_BYTES,
            "W membership proof",
        )
        parsed_x_membership = _validate_membership_proof(
            x_membership,
            expected_indices=tuple(
                range(
                    self.descriptor.row_offset,
                    self.descriptor.row_offset + self.descriptor.rows,
                )
            ),
            expected_vector_length=self.descriptor.inner_dim,
            expected_encoding=PCS_ENCODING_HASHED_QUANTIZED_I8_ROW,
            name="X",
        )
        for row_offset, opened in enumerate(parsed_x_membership.opened_leaves):
            start = row_offset * self.descriptor.inner_dim
            row = x_rows[start : start + self.descriptor.inner_dim]
            if opened.leaf.commitment != hash_quantized_i8_row_v2(
                row,
                self.descriptor.inner_dim,
                scales[row_offset],
            ):
                raise ProofV2PayloadError(
                    "X row data does not match its authenticated leaf digest"
                )
        parsed_y_membership = _validate_membership_proof(
            runtime_y_membership,
            expected_indices=None,
            expected_vector_length=expected_runtime_y_values,
            expected_encoding=PCS_ENCODING_HASHED_FP16_BLOCK,
            name="runtime Y",
        )
        if len(parsed_y_membership.opened_leaves) != 1:
            raise ProofV2PayloadError("runtime Y must open exactly one block leaf")
        if parsed_y_membership.opened_leaves[0].leaf.commitment != hash_fp16_block_v2(
            runtime_y_values,
            expected_runtime_y_values,
        ):
            raise ProofV2PayloadError(
                "runtime Y data does not match its authenticated leaf digest"
            )
        _validate_membership_proof(
            w_membership,
            expected_indices=tuple(
                range(
                    self.descriptor.column_offset,
                    self.descriptor.column_offset + self.descriptor.cols,
                )
            ),
            expected_vector_length=self.descriptor.inner_dim,
            expected_encoding=MERKLE_ENCODING_SIGNED_I8,
            name="W",
        )
        _fixed(self.proof_y_commitment, 32, "proof Y commitment")

    def canonical_bytes(self) -> bytes:
        encoded = bytearray(_encode_descriptor(self.descriptor))
        for value in (
            self.x_membership_proof,
            self.x_rows,
            self.x_scales_q32,
            self.runtime_y_membership_proof,
            self.runtime_y_values,
            self.proof_y_values,
            self.w_membership_proof,
        ):
            encoded.extend(struct.pack("<I", len(value)))
            encoded.extend(value)
        encoded.extend(self.proof_y_commitment)
        return bytes(encoded)


@dataclass(frozen=True)
class ProofV2Payload:
    """Final single-message proof-v2 payload."""

    manifest_digest: bytes
    block_proofs: tuple[GemmBlockProofV2, ...]
    batch_sumcheck_proof: bytes
    w_opening: PCSOpeningV2
    execution_trace_proof: bytes = b""

    def __post_init__(self) -> None:
        _fixed(self.manifest_digest, 32, "manifest_digest")
        blocks = tuple(self.block_proofs)
        if not blocks or len(blocks) > MAX_BLOCK_PROOFS:
            raise ProofV2PayloadError("block proof count is out of range")
        if not all(isinstance(item, GemmBlockProofV2) for item in blocks):
            raise ProofV2PayloadError("block proofs have an unexpected type")
        descriptors = [item.descriptor.as_challenge() for item in blocks]
        if descriptors != sorted(descriptors) or len(descriptors) != len(
            set(descriptors)
        ):
            raise ProofV2PayloadError(
                "block proofs must have sorted unique descriptors"
            )
        inner = max(block.descriptor.padded_inner_dim for block in blocks)
        sumcheck = _bounded(
            self.batch_sumcheck_proof,
            MAX_SUMCHECK_PROOF_BYTES,
            "batch sumcheck proof",
        )
        try:
            parsed_sumcheck = GemmV2BatchProof.from_canonical_bytes(
                sumcheck,
                expected_blocks=len(blocks),
                expected_rounds=inner.bit_length() - 1,
            )
        except ValueError as exc:
            raise ProofV2PayloadError("batch sumcheck proof is not canonical") from exc
        if parsed_sumcheck.canonical_bytes() != sumcheck:
            raise ProofV2PayloadError("batch sumcheck proof is not canonical")
        _validate_opening(
            self.w_opening,
            expected_length=inner,
            expected_encoding=ENCODING_PALLAS_SCALAR,
            name="batched W",
        )
        if not isinstance(self.execution_trace_proof, bytes):
            raise ProofV2PayloadError("execution trace proof must be bytes")
        if len(self.execution_trace_proof) > MAX_TRACE_PROOF_BYTES:
            raise ProofV2PayloadError(
                "execution trace proof exceeds the protocol limit"
            )
        if self.execution_trace_proof:
            try:
                parsed_trace = ExecutionTraceProofV2.from_canonical_bytes(
                    self.execution_trace_proof
                )
            except ProofV2TraceError as exc:
                raise ProofV2PayloadError(
                    "execution trace proof is not canonical"
                ) from exc
            if parsed_trace.canonical_bytes() != self.execution_trace_proof:
                raise ProofV2PayloadError("execution trace proof is not canonical")
        object.__setattr__(self, "block_proofs", blocks)

    def canonical_bytes(self) -> bytes:
        encoded = bytearray(
            struct.pack("<4sHI", _PROOF_MAGIC, PROTOCOL_VERSION, len(self.block_proofs))
        )
        encoded.extend(self.manifest_digest)
        for block in self.block_proofs:
            block_bytes = block.canonical_bytes()
            encoded.extend(struct.pack("<I", len(block_bytes)))
            encoded.extend(block_bytes)
        encoded.extend(struct.pack("<I", len(self.batch_sumcheck_proof)))
        encoded.extend(self.batch_sumcheck_proof)
        encoded.extend(struct.pack("<I", len(self.execution_trace_proof)))
        encoded.extend(self.execution_trace_proof)
        encoded.extend(_encode_opening(self.w_opening))
        if len(encoded) > MAX_FINAL_PROOF_BYTES:
            raise ProofV2PayloadError("final proof payload exceeds the protocol limit")
        return bytes(encoded)

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "ProofV2Payload":
        reader = _Reader(encoded, MAX_FINAL_PROOF_BYTES, "final proof payload")
        magic, version, count = reader.unpack("<4sHI")
        if magic != _PROOF_MAGIC or version != PROTOCOL_VERSION:
            raise ProofV2PayloadError("final proof payload header is not supported")
        if not 0 < count <= MAX_BLOCK_PROOFS:
            raise ProofV2PayloadError("block proof count is out of range")
        manifest_digest = reader.read(32)
        blocks = []
        for _ in range(count):
            block_length = reader.unpack("<I")[0]
            if block_length == 0 or block_length > reader.remaining:
                raise ProofV2PayloadError("block proof length is out of range")
            blocks.append(_decode_block(reader.read(block_length)))
        sumcheck_length = reader.unpack("<I")[0]
        if sumcheck_length == 0 or sumcheck_length > MAX_SUMCHECK_PROOF_BYTES:
            raise ProofV2PayloadError("batch sumcheck proof length is out of range")
        batch_sumcheck_proof = reader.read(sumcheck_length)
        trace_length = reader.unpack("<I")[0]
        if trace_length > MAX_TRACE_PROOF_BYTES:
            raise ProofV2PayloadError("execution trace proof length is out of range")
        execution_trace_proof = reader.read(trace_length)
        w_opening = _decode_opening(reader, name="batched W")
        reader.finish()
        result = cls(
            manifest_digest,
            tuple(blocks),
            batch_sumcheck_proof,
            w_opening,
            execution_trace_proof,
        )
        if result.canonical_bytes() != encoded:
            raise ProofV2PayloadError("final proof payload is not canonical")
        return result


class _Reader:
    def __init__(self, encoded: bytes, maximum: int, name: str):
        if not isinstance(encoded, bytes) or not encoded or len(encoded) > maximum:
            raise ProofV2PayloadError(f"{name} length is out of range")
        self._encoded = encoded
        self._offset = 0

    @property
    def remaining(self) -> int:
        return len(self._encoded) - self._offset

    def read(self, length: int) -> bytes:
        if length < 0 or length > self.remaining:
            raise ProofV2PayloadError("wire payload is truncated")
        start = self._offset
        self._offset += length
        return self._encoded[start : start + length]

    def unpack(self, format_: str) -> tuple:
        size = struct.calcsize(format_)
        try:
            result = struct.unpack(format_, self.read(size))
        except struct.error as exc:
            raise ProofV2PayloadError("wire payload is malformed") from exc
        return result

    def read_key(self) -> OperationKeyV2:
        layer_idx, expert_idx, length = self.unpack("<IiH")
        if length == 0 or length > 64:
            raise ProofV2PayloadError("operation identifier length is out of range")
        try:
            operation_id = self.read(length).decode("ascii")
        except UnicodeDecodeError as exc:
            raise ProofV2PayloadError("operation identifier is not ASCII") from exc
        key = OperationKeyV2(layer_idx, operation_id, expert_idx)
        try:
            key.canonical_bytes()
        except (TypeError, ValueError) as exc:
            raise ProofV2PayloadError("operation identifier is not canonical") from exc
        return key

    def finish(self) -> None:
        if self.remaining:
            raise ProofV2PayloadError("wire payload contains trailing data")


def _decode_descriptor(reader: _Reader) -> ProofBlockDescriptorV2:
    key = reader.read_key()
    values = reader.unpack("<11I")
    descriptor = ProofBlockDescriptorV2(key, *values)
    _validate_descriptor(descriptor)
    return descriptor


def _decode_opening(reader: _Reader, *, name: str) -> PCSOpeningV2:
    commitment = reader.read(32)
    evaluation = reader.read(32)
    vector_length, padded_length, encoding, proof_length = reader.unpack("<IIBH")
    if proof_length == 0 or proof_length > MAX_IPA_PROOF_BYTES:
        raise ProofV2PayloadError(f"{name} IPA proof length is out of range")
    return PCSOpeningV2(
        commitment=commitment,
        evaluation=evaluation,
        proof=reader.read(proof_length),
        vector_length=vector_length,
        padded_length=padded_length,
        encoding=encoding,
    )


def _validate_membership_proof(
    encoded: bytes,
    *,
    expected_indices: tuple[int, ...] | None,
    expected_vector_length: int,
    expected_encoding: int,
    name: str,
) -> PcsMerkleV2MultiProof:
    try:
        proof = PcsMerkleV2MultiProof.from_canonical_bytes(encoded)
    except ValueError as exc:
        raise ProofV2PayloadError(f"{name} membership proof is not canonical") from exc
    if expected_indices is not None and proof.indices != expected_indices:
        raise ProofV2PayloadError(
            f"{name} membership proof does not open the exact block"
        )
    for opened in proof.opened_leaves:
        if (
            opened.leaf.logical_vector_length != expected_vector_length
            or opened.leaf.encoding_id != expected_encoding
        ):
            raise ProofV2PayloadError(
                f"{name} membership leaf metadata does not match the operation"
            )
    return proof


def _decode_block(encoded: bytes) -> GemmBlockProofV2:
    reader = _Reader(encoded, MAX_FINAL_PROOF_BYTES, "block proof")
    descriptor = _decode_descriptor(reader)
    variable_fields = []
    for name, maximum in (
        ("X membership proof", MAX_MEMBERSHIP_PROOF_BYTES),
        ("X row data", MAX_X_ROWS_BYTES),
        ("X scale data", MAX_X_SCALES_BYTES),
        ("runtime Y membership proof", MAX_MEMBERSHIP_PROOF_BYTES),
        ("runtime Y data", MAX_RUNTIME_Y_BYTES),
        ("proof Y data", MAX_PROOF_Y_BYTES),
        ("W membership proof", MAX_MEMBERSHIP_PROOF_BYTES),
    ):
        length = reader.unpack("<I")[0]
        if length == 0 or length > maximum:
            raise ProofV2PayloadError(f"{name} length is out of range")
        variable_fields.append(reader.read(length))
    proof_y_commitment = reader.read(32)
    reader.finish()
    result = GemmBlockProofV2(
        descriptor=descriptor,
        x_membership_proof=variable_fields[0],
        x_rows=variable_fields[1],
        x_scales_q32=variable_fields[2],
        runtime_y_membership_proof=variable_fields[3],
        runtime_y_values=variable_fields[4],
        proof_y_values=variable_fields[5],
        w_membership_proof=variable_fields[6],
        proof_y_commitment=proof_y_commitment,
    )
    if result.canonical_bytes() != encoded:
        raise ProofV2PayloadError("block proof is not canonical")
    return result


def commitment_envelope_from_bytes(encoded: bytes) -> ProofV2CommitmentEnvelope:
    return ProofV2CommitmentEnvelope.from_canonical_bytes(encoded)


def proof_payload_from_bytes(encoded: bytes) -> ProofV2Payload:
    return ProofV2Payload.from_canonical_bytes(encoded)


__all__ = [
    "GemmBlockProofV2",
    "ProofV2CommitmentEnvelope",
    "ProofV2Payload",
    "ProofV2PayloadError",
    "commitment_envelope_from_bytes",
    "proof_payload_from_bytes",
]
