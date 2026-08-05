"""Canonical static weight commitment catalogs for proof protocol v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Iterable, Mapping

from verallm.challenge.v2 import OperationKeyV2
from zkllm.crypto.pcs_v2 import (
    PCSFormatError,
    validate_commitments,
)


PROTOCOL_VERSION = 2
MAX_CATALOG_OPERATIONS = 100_000
MAX_COLUMNS_PER_OPERATION = 1 << 20
MAX_TOTAL_COLUMN_COMMITMENTS = 8_000_000
MAX_CATALOG_BYTES = 256 << 20

_MAGIC = b"V2WC"
_HEADER = struct.Struct("<4sHII32s")
_COLUMN_COUNT = struct.Struct("<I")


class WeightCommitmentCatalogError(ValueError):
    """A weight commitment catalog is malformed or noncanonical."""


def _bounded_items(value: object, maximum: int, name: str) -> tuple:
    if isinstance(value, (bytes, bytearray, memoryview, str)):
        raise WeightCommitmentCatalogError(f"{name} must be an iterable of records")
    try:
        items = tuple(islice(iter(value), maximum + 1))  # type: ignore[arg-type]
    except TypeError as exc:
        raise WeightCommitmentCatalogError(
            f"{name} must be an iterable of records"
        ) from exc
    if not items or len(items) > maximum:
        raise WeightCommitmentCatalogError(f"{name} count is out of range")
    return items


def _canonical_key_bytes(key: OperationKeyV2) -> bytes:
    if not isinstance(key, OperationKeyV2):
        raise WeightCommitmentCatalogError("operation key has an unexpected type")
    try:
        encoded = key.canonical_bytes()
    except (TypeError, ValueError) as exc:
        raise WeightCommitmentCatalogError("operation key is not canonical") from exc
    if len(encoded) > 4 + 4 + 2 + 64:
        raise WeightCommitmentCatalogError("operation key exceeds the protocol limit")
    return encoded


def _validate_commitment_batch(commitments: tuple[bytes, ...]) -> None:
    try:
        validate_commitments(commitments)
    except PCSFormatError as exc:
        raise WeightCommitmentCatalogError(
            "column commitments contain a non-canonical Pallas commitment"
        ) from exc


def _canonical_commitments(value: object) -> tuple[bytes, ...]:
    commitments = _bounded_items(
        value,
        MAX_COLUMNS_PER_OPERATION,
        "column commitments",
    )
    for index, commitment in enumerate(commitments):
        if type(commitment) is not bytes or len(commitment) != 32:
            raise WeightCommitmentCatalogError(
                f"column commitment {index} must be exactly 32 bytes"
            )
    _validate_commitment_batch(commitments)
    return commitments


@dataclass(frozen=True, slots=True)
class WeightCommitmentOperationV2:
    """Canonical per-column commitments for one registered operation."""

    key: OperationKeyV2
    column_commitments: tuple[bytes, ...]
    _source_path: Path | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _source_offset: int | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _canonical_key_bytes(self.key)
        object.__setattr__(
            self,
            "column_commitments",
            _canonical_commitments(self.column_commitments),
        )

    @property
    def column_count(self) -> int:
        return len(self.column_commitments)

    def canonical_bytes(self) -> bytes:
        return (
            _canonical_key_bytes(self.key)
            + _COLUMN_COUNT.pack(self.column_count)
            + b"".join(self.column_commitments)
        )


@dataclass(frozen=True, slots=True)
class WeightCommitmentCatalogV2:
    """One manifest-bound, exact operation-to-column commitment catalog."""

    manifest_digest: bytes
    operations: tuple[WeightCommitmentOperationV2, ...]
    _lookup: Mapping[OperationKeyV2, tuple[bytes, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.manifest_digest) is not bytes or len(self.manifest_digest) != 32:
            raise WeightCommitmentCatalogError(
                "manifest digest must be exactly 32 bytes"
            )
        operations = _bounded_items(
            self.operations,
            MAX_CATALOG_OPERATIONS,
            "catalog operations",
        )
        if not all(
            isinstance(item, WeightCommitmentOperationV2) for item in operations
        ):
            raise WeightCommitmentCatalogError(
                "catalog operation has an unexpected type"
            )
        keys = [item.key for item in operations]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise WeightCommitmentCatalogError(
                "catalog operations must have sorted unique operation keys"
            )
        total = sum(item.column_count for item in operations)
        if total <= 0 or total > MAX_TOTAL_COLUMN_COMMITMENTS:
            raise WeightCommitmentCatalogError(
                "total column commitment count is out of range"
            )
        encoded_size = _HEADER.size + sum(
            len(_canonical_key_bytes(item.key))
            + _COLUMN_COUNT.size
            + 32 * item.column_count
            for item in operations
        )
        if encoded_size > MAX_CATALOG_BYTES:
            raise WeightCommitmentCatalogError(
                "weight commitment catalog exceeds the protocol limit"
            )
        object.__setattr__(self, "operations", operations)
        object.__setattr__(
            self,
            "_lookup",
            MappingProxyType(
                {item.key: item.column_commitments for item in operations}
            ),
        )

    @property
    def operation_keys(self) -> tuple[OperationKeyV2, ...]:
        return tuple(item.key for item in self.operations)

    @property
    def total_column_commitments(self) -> int:
        return sum(item.column_count for item in self.operations)

    def lookup(self, key: OperationKeyV2) -> tuple[bytes, ...]:
        """Return the exact ordered column commitments for ``key``."""
        if not isinstance(key, OperationKeyV2):
            raise TypeError("key must be OperationKeyV2")
        return self._lookup[key]

    def column_commitment(self, key: OperationKeyV2, column_index: int) -> bytes:
        """Return one operation column commitment by canonical column index."""
        if isinstance(column_index, bool) or not isinstance(column_index, int):
            raise TypeError("column_index must be an integer")
        columns = self.lookup(key)
        if column_index < 0 or column_index >= len(columns):
            raise IndexError("column_index is out of range")
        return columns[column_index]

    def canonical_bytes(self) -> bytes:
        encoded = bytearray(
            _HEADER.pack(
                _MAGIC,
                PROTOCOL_VERSION,
                len(self.operations),
                self.total_column_commitments,
                self.manifest_digest,
            )
        )
        for operation in self.operations:
            encoded.extend(operation.canonical_bytes())
        if len(encoded) > MAX_CATALOG_BYTES:
            raise WeightCommitmentCatalogError(
                "weight commitment catalog exceeds the protocol limit"
            )
        return bytes(encoded)

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "WeightCommitmentCatalogV2":
        return cls._from_canonical_bytes(
            encoded,
            validate_commitment_points=True,
        )

    @classmethod
    def _from_canonical_bytes(
        cls,
        encoded: bytes,
        *,
        validate_commitment_points: bool,
    ) -> "WeightCommitmentCatalogV2":
        reader = _Reader(encoded)
        magic, version, operation_count, total_count, digest = reader.unpack(_HEADER)
        if magic != _MAGIC or version != PROTOCOL_VERSION:
            raise WeightCommitmentCatalogError(
                "weight commitment catalog header is not supported"
            )
        if not 0 < operation_count <= MAX_CATALOG_OPERATIONS:
            raise WeightCommitmentCatalogError(
                "catalog operation count is out of range"
            )
        if not 0 < total_count <= MAX_TOTAL_COLUMN_COMMITMENTS:
            raise WeightCommitmentCatalogError(
                "total column commitment count is out of range"
            )
        operations = []
        parsed_total = 0
        for _ in range(operation_count):
            key = reader.read_key()
            column_count = reader.unpack(_COLUMN_COUNT)[0]
            if not 0 < column_count <= MAX_COLUMNS_PER_OPERATION:
                raise WeightCommitmentCatalogError(
                    "operation column commitment count is out of range"
                )
            parsed_total += column_count
            if parsed_total > total_count:
                raise WeightCommitmentCatalogError(
                    "catalog column commitment count does not match its header"
                )
            source_offset = reader.offset
            raw = reader.read(column_count * 32)
            columns = tuple(
                raw[offset : offset + 32] for offset in range(0, len(raw), 32)
            )
            if validate_commitment_points:
                operation = WeightCommitmentOperationV2(key, columns)
            else:
                # This path is reserved for a catalog whose exact bytes and
                # signed manifest were covered by a persistent deep-validation
                # receipt. Structural decoding above still enforces every
                # canonical length/count/key invariant.
                operation = object.__new__(WeightCommitmentOperationV2)
                object.__setattr__(operation, "key", key)
                object.__setattr__(
                    operation,
                    "column_commitments",
                    columns,
                )
            object.__setattr__(operation, "_source_offset", source_offset)
            operations.append(operation)
        if parsed_total != total_count:
            raise WeightCommitmentCatalogError(
                "catalog column commitment count does not match its header"
            )
        reader.finish()
        result = cls(digest, tuple(operations))
        if result.canonical_bytes() != encoded:
            raise WeightCommitmentCatalogError(
                "weight commitment catalog is not canonical"
            )
        return result

    @classmethod
    def load(cls, path: str | Path) -> "WeightCommitmentCatalogV2":
        """Load and strictly decode a catalog from ``path``."""
        return cls._load(path, validate_commitment_points=True)

    @classmethod
    def _load_prevalidated(
        cls,
        path: str | Path,
    ) -> "WeightCommitmentCatalogV2":
        """Structurally decode bytes covered by a v3 validation receipt."""

        return cls._load(path, validate_commitment_points=False)

    @classmethod
    def _load(
        cls,
        path: str | Path,
        *,
        validate_commitment_points: bool,
    ) -> "WeightCommitmentCatalogV2":
        catalog_path = Path(path)
        size = catalog_path.stat().st_size
        if size <= 0 or size > MAX_CATALOG_BYTES:
            raise WeightCommitmentCatalogError(
                "weight commitment catalog file size is out of range"
            )
        with catalog_path.open("rb") as handle:
            encoded = handle.read(MAX_CATALOG_BYTES + 1)
        if len(encoded) != size or len(encoded) > MAX_CATALOG_BYTES:
            raise WeightCommitmentCatalogError(
                "weight commitment catalog file size changed while loading"
            )
        result = cls._from_canonical_bytes(
            encoded,
            validate_commitment_points=validate_commitment_points,
        )
        source = catalog_path.resolve()
        for operation in result.operations:
            object.__setattr__(operation, "_source_path", source)
        return result


class _Reader:
    def __init__(self, encoded: bytes):
        if (
            type(encoded) is not bytes
            or not encoded
            or len(encoded) > MAX_CATALOG_BYTES
        ):
            raise WeightCommitmentCatalogError(
                "weight commitment catalog length is out of range"
            )
        self._encoded = encoded
        self._offset = 0

    @property
    def remaining(self) -> int:
        return len(self._encoded) - self._offset

    @property
    def offset(self) -> int:
        return self._offset

    def read(self, length: int) -> bytes:
        if length < 0 or length > self.remaining:
            raise WeightCommitmentCatalogError("weight commitment catalog is truncated")
        start = self._offset
        self._offset += length
        return self._encoded[start : self._offset]

    def unpack(self, format_: struct.Struct) -> tuple:
        try:
            return format_.unpack(self.read(format_.size))
        except struct.error as exc:
            raise WeightCommitmentCatalogError(
                "weight commitment catalog is malformed"
            ) from exc

    def read_key(self) -> OperationKeyV2:
        layer_idx, expert_idx = struct.unpack("<Ii", self.read(8))
        operation_length = struct.unpack("<H", self.read(2))[0]
        if operation_length == 0 or operation_length > 64:
            raise WeightCommitmentCatalogError(
                "operation identifier length is out of range"
            )
        raw_identifier = self.read(operation_length)
        try:
            operation_id = raw_identifier.decode("ascii")
        except UnicodeDecodeError as exc:
            raise WeightCommitmentCatalogError(
                "operation identifier is not ASCII"
            ) from exc
        key = OperationKeyV2(layer_idx, operation_id, expert_idx)
        if _canonical_key_bytes(key) != (
            struct.pack("<IiH", layer_idx, expert_idx, operation_length)
            + raw_identifier
        ):
            raise WeightCommitmentCatalogError(
                "operation key encoding is not canonical"
            )
        return key

    def finish(self) -> None:
        if self.remaining:
            raise WeightCommitmentCatalogError(
                "weight commitment catalog contains trailing data"
            )


__all__ = [
    "MAX_CATALOG_BYTES",
    "MAX_CATALOG_OPERATIONS",
    "MAX_COLUMNS_PER_OPERATION",
    "MAX_TOTAL_COLUMN_COMMITMENTS",
    "PROTOCOL_VERSION",
    "WeightCommitmentCatalogError",
    "WeightCommitmentCatalogV2",
    "WeightCommitmentOperationV2",
]
