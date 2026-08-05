"""Bounded raw-static-byte conformance reference for proof-v3.

The signed V3 static artifact already binds a static-table root, a source
binding, a table-layout digest, and an opaque construction digest.  This
unregistered CPU module supplies the *first deliberately narrow*
interpretation of that construction artifact:

* a canonical ``V3SC`` construction file fixes a byte-page SHA-256 tree;
* the source binding authenticates the construction bytes and resulting root;
* one logical leaf is exactly one signed int8 scalar; and
* every active fixed trace cell for one selected AIR program is compared to
  the corresponding authenticated byte.

Nothing broader is inferred.  The reference rejects fp16/bf16/fp8/int4/NF4,
multi-byte leaves, scale/dequantization semantics, and unknown construction
versions.  It is not a static lookup argument, a matrix-product relation, a
transition proof, or a production verifier.  In particular, authenticating a
raw fixed trace column does not prove that the column was used correctly by a
linear or attention relation.

There is deliberately no public hard-audit verifier in this module.  Its
retained-cell helper is private because it accepts caller-supplied program and
precommitment objects.  A future validator integration must instead derive
the exact slot/program from a validator-sealed trace-map receipt and bind that
receipt atomically to the pre-nonce runtime tensor receipt.

The byte tree is intentionally independent from ``goldilocks_merkle_reference``:
that module commits Goldilocks field rows, whereas this one commits raw static
bytes under the signed ``sha256.merkle.binary.v1`` static-root leg.  The small
in-memory tree and retained trace witness exist only for golden-vector tests;
a qualified backend must replace them with streaming byte-to-field lookup/copy
constraints and the applicable arithmetic relation.
"""

from __future__ import annotations

import hashlib
import hmac
import operator
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.constraint_program import (
    GOLDILOCKS_TRACE_SOURCE_BINDING_MODE_EXACT_LAYOUT_V3,
    GoldilocksConstraintProgramV3,
    GoldilocksConstraintTraceReferenceV3,
    verify_goldilocks_constraint_program_reference_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_air_reference import (
    GoldilocksAirTracePrecommitmentReferenceV3,
    build_goldilocks_air_trace_oracle_reference_v3,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.static_artifact import (
    GOLDILOCKS_STATIC_TABLE_CONSTRUCTION_ABI_ID_V3,
    GOLDILOCKS_STATIC_TABLE_HASH_ABI_ID_V3,
    StaticTableDescriptorV3,
    StaticTableSourceBindingV3,
    StaticWeightArtifactV3,
)


GOLDILOCKS_STATIC_BYTE_REFERENCE_ABI_V3: Final = (
    "goldilocks.static_byte.reference.v1"
)
GOLDILOCKS_STATIC_BYTE_REFERENCE_FORMAT_VERSION_V3: Final = 1
GOLDILOCKS_STATIC_BYTE_CONSTRUCTION_ABI_V3: Final = (
    GOLDILOCKS_STATIC_TABLE_CONSTRUCTION_ABI_ID_V3
)
GOLDILOCKS_STATIC_BYTE_CONSTRUCTION_MAGIC_V3: Final = b"V3SC"
GOLDILOCKS_STATIC_BYTE_CONSTRUCTION_FORMAT_VERSION_V3: Final = 1
GOLDILOCKS_STATIC_BYTE_PADDING_RULE_V3: Final = "zero_bytes.v1"
GOLDILOCKS_STATIC_BYTE_LOWERING_SIGNED_INT8_V3: Final = (
    "signed_int8.goldilocks.v1"
)
GOLDILOCKS_STATIC_BYTE_ELEMENT_ENCODING_INT8_V3: Final = "int8.symmetric.v1"
GOLDILOCKS_STATIC_BYTE_SCALE_ENCODING_NONE_V3: Final = "none.v1"

# These are deliberately CPU-reference bounds, not profile limits or a
# qualified backend capacity claim.  They cap an all-pages retained test tree.
MAX_GOLDILOCKS_STATIC_BYTE_REFERENCE_TREE_LEAVES_V3: Final = 1 << 12
MAX_GOLDILOCKS_STATIC_BYTE_REFERENCE_PAGE_BYTES_V3: Final = 1 << 16
MAX_GOLDILOCKS_STATIC_BYTE_REFERENCE_TOTAL_BYTES_V3: Final = 1 << 20
MAX_GOLDILOCKS_STATIC_BYTE_REFERENCE_CONSTRUCTION_BYTES_V3: Final = 1 << 20
MAX_GOLDILOCKS_STATIC_BYTE_REFERENCE_CONSTRUCTION_TABLES_V3: Final = (
    (1 << 16) - 1
)

_CONSTRUCTION_DIGEST_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_STATIC_BYTE/V1/CONSTRUCTION_DIGEST/SHA256"
)
_TREE_BINDING_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_STATIC_BYTE/V1/TREE_BINDING/SHA256"
)
_TREE_LEAF_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_STATIC_BYTE/V1/TREE_LEAF/SHA256"
)
_TREE_NODE_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_STATIC_BYTE/V1/TREE_NODE/SHA256"
)
_TREE_ROOT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_STATIC_BYTE/V1/TREE_ROOT/SHA256"
)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ProofV3Error(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ProofV3Error(f"{name} must be an integer") from exc


def _u16(value: object, name: str, *, positive: bool = False) -> int:
    result = _integer(value, name)
    if result < (1 if positive else 0) or result >= 1 << 16:
        raise ProofV3Error(f"{name} must be an unsigned 16-bit integer")
    return result


def _u32(value: object, name: str, *, positive: bool = False) -> int:
    result = _integer(value, name)
    if result < (1 if positive else 0) or result >= 1 << 32:
        raise ProofV3Error(f"{name} must be an unsigned 32-bit integer")
    return result


def _u64(value: object, name: str, *, positive: bool = False) -> int:
    result = _integer(value, name)
    if result < (1 if positive else 0) or result >= 1 << 64:
        raise ProofV3Error(f"{name} must be an unsigned 64-bit integer")
    return result


def _fixed32(value: object, name: str, *, nonzero: bool = False) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    if nonzero and value == bytes(32):
        raise ProofV3Error(f"{name} must not be zero")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ProofV3Error(f"{name} must be a string")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ProofV3Error(f"{name} must be ASCII") from exc
    if not encoded or len(encoded) > 128:
        raise ProofV3Error(f"{name} is malformed")
    return value


def _encode_identifier(value: str, name: str) -> bytes:
    identifier = _identifier(value, name)
    encoded = identifier.encode("ascii")
    return struct.pack("<H", len(encoded)) + encoded


class _Reader:
    __slots__ = ("_encoded", "_offset")

    def __init__(self, encoded: object) -> None:
        if not isinstance(encoded, bytes) or not encoded:
            raise ProofV3Error("static byte construction must be nonempty bytes")
        if len(encoded) > MAX_GOLDILOCKS_STATIC_BYTE_REFERENCE_CONSTRUCTION_BYTES_V3:
            raise ProofV3Error("static byte construction exceeds the CPU reference cap")
        self._encoded = encoded
        self._offset = 0

    def read(self, length: int, name: str) -> bytes:
        if length < 0 or self._offset + length > len(self._encoded):
            raise ProofV3Error(f"static byte construction is truncated at {name}")
        result = self._encoded[self._offset : self._offset + length]
        self._offset += length
        return result

    def unpack(self, fmt: str, name: str) -> tuple[object, ...]:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size, name))

    def identifier(self, name: str) -> str:
        (length_raw,) = self.unpack("<H", f"{name} length")
        length = int(length_raw)
        if not 0 < length <= 128:
            raise ProofV3Error(f"{name} length is out of range")
        try:
            return _identifier(self.read(length, name).decode("ascii"), name)
        except UnicodeDecodeError as exc:
            raise ProofV3Error(f"{name} must be ASCII") from exc

    def finish(self) -> None:
        if self._offset != len(self._encoded):
            raise ProofV3Error("static byte construction has trailing bytes")


def _materialize_tuple(
    value: object,
    *,
    name: str,
    maximum: int,
    allow_empty: bool = False,
) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        raise ProofV3Error(f"{name} must be an iterable")
    try:
        result = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ProofV3Error(f"{name} must be an iterable") from exc
    if (not result and not allow_empty) or len(result) > maximum:
        raise ProofV3Error(f"{name} has an invalid size")
    return result


def _power_of_two(value: int, name: str) -> int:
    if value < 1 or value & (value - 1):
        raise ProofV3Error(f"{name} must be a positive power of two")
    return value


def _tree_height(leaf_count: int) -> int:
    return leaf_count.bit_length() - 1


def _opening_indices(
    value: object,
    *,
    maximum: int,
    name: str,
) -> tuple[int, ...]:
    entries = _materialize_tuple(value, name=name, maximum=maximum)
    result = tuple(_u64(item, f"{name}[{index}]") for index, item in enumerate(entries))
    if result != tuple(sorted(set(result))):
        raise ProofV3Error(f"{name} must be sorted and distinct")
    if any(index >= maximum for index in result):
        raise ProofV3Error(f"{name} contains an out-of-range page")
    return result


def _expected_sibling_coordinates(
    *,
    leaf_count: int,
    indices: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    current = set(indices)
    result: list[tuple[int, int]] = []
    for level in range(_tree_height(leaf_count)):
        missing = sorted(index ^ 1 for index in current if (index ^ 1) not in current)
        result.extend((level, index) for index in missing)
        current = {index // 2 for index in current}
    return tuple(result)


@dataclass(frozen=True, slots=True)
class GoldilocksStaticByteConstructionTableReferenceV3:
    """One descriptor identity interpreted by a ``V3SC`` construction file."""

    table_id: str
    descriptor_digest: bytes

    def __post_init__(self) -> None:
        _identifier(self.table_id, "static byte construction table_id")
        _fixed32(
            self.descriptor_digest,
            "static byte construction table descriptor digest",
            nonzero=True,
        )

    def canonical_bytes(self) -> bytes:
        return _encode_identifier(self.table_id, "static byte construction table_id") + self.descriptor_digest


@dataclass(frozen=True, slots=True)
class GoldilocksStaticByteConstructionReferenceV3:
    """Strict raw-int8 construction metadata authenticated by a source binding.

    The construction describes only the byte-tree interpretation.  The signed
    source binding supplies its SHA-256 digest, geometry, and committed root;
    the signed static artifact supplies the descriptor objects whose digests
    appear here.  This prevents a prover from choosing a local table layout.
    """

    static_table_layout_digest: bytes
    tables: tuple[GoldilocksStaticByteConstructionTableReferenceV3, ...]
    table_hash_abi_id: str = GOLDILOCKS_STATIC_TABLE_HASH_ABI_ID_V3
    padding_rule_id: str = GOLDILOCKS_STATIC_BYTE_PADDING_RULE_V3
    cell_lowering_id: str = GOLDILOCKS_STATIC_BYTE_LOWERING_SIGNED_INT8_V3
    abi_id: str = GOLDILOCKS_STATIC_BYTE_CONSTRUCTION_ABI_V3
    format_version: int = GOLDILOCKS_STATIC_BYTE_CONSTRUCTION_FORMAT_VERSION_V3

    def __post_init__(self) -> None:
        if self.abi_id != GOLDILOCKS_STATIC_BYTE_CONSTRUCTION_ABI_V3:
            raise ProofV3Error("static byte construction ABI is unsupported")
        if type(self.format_version) is not int or self.format_version != (
            GOLDILOCKS_STATIC_BYTE_CONSTRUCTION_FORMAT_VERSION_V3
        ):
            raise ProofV3Error("static byte construction version is unsupported")
        _fixed32(
            self.static_table_layout_digest,
            "static byte construction layout digest",
            nonzero=True,
        )
        if self.table_hash_abi_id != GOLDILOCKS_STATIC_TABLE_HASH_ABI_ID_V3:
            raise ProofV3Error("static byte construction hash ABI is unsupported")
        if self.padding_rule_id != GOLDILOCKS_STATIC_BYTE_PADDING_RULE_V3:
            raise ProofV3Error("static byte construction padding rule is unsupported")
        if self.cell_lowering_id != GOLDILOCKS_STATIC_BYTE_LOWERING_SIGNED_INT8_V3:
            raise ProofV3Error("static byte construction cell lowering is unsupported")
        raw_tables = _materialize_tuple(
            self.tables,
            name="static byte construction tables",
            maximum=MAX_GOLDILOCKS_STATIC_BYTE_REFERENCE_CONSTRUCTION_TABLES_V3,
        )
        if not all(
            isinstance(item, GoldilocksStaticByteConstructionTableReferenceV3)
            for item in raw_tables
        ):
            raise ProofV3Error("static byte construction table has an unexpected type")
        tables = tuple(raw_tables)
        if tuple(item.table_id for item in tables) != tuple(
            sorted(item.table_id for item in tables)
        ):
            raise ProofV3Error("static byte construction tables are not canonical")
        if len({item.table_id for item in tables}) != len(tables):
            raise ProofV3Error("static byte construction tables are duplicated")
        object.__setattr__(self, "tables", tables)

    def canonical_bytes(self) -> bytes:
        table_bytes = tuple(item.canonical_bytes() for item in self.tables)
        return (
            struct.pack(
                "<4sHH",
                GOLDILOCKS_STATIC_BYTE_CONSTRUCTION_MAGIC_V3,
                self.format_version,
                len(table_bytes),
            )
            + self.static_table_layout_digest
            + _encode_identifier(self.abi_id, "static byte construction ABI")
            + _encode_identifier(
                self.table_hash_abi_id,
                "static byte construction hash ABI",
            )
            + _encode_identifier(
                self.padding_rule_id,
                "static byte construction padding rule",
            )
            + _encode_identifier(
                self.cell_lowering_id,
                "static byte construction cell lowering",
            )
            + b"".join(table_bytes)
        )

    def raw_sha256(self) -> bytes:
        """Return the digest stored verbatim in ``StaticTableSourceBindingV3``."""

        return hashlib.sha256(self.canonical_bytes()).digest()

    def digest(self) -> bytes:
        """Return a domain-separated identity for local reference transcripts."""

        return hashlib.sha256(
            _CONSTRUCTION_DIGEST_DOMAIN + self.canonical_bytes()
        ).digest()

    @classmethod
    def from_canonical_bytes(
        cls,
        encoded: bytes,
    ) -> "GoldilocksStaticByteConstructionReferenceV3":
        reader = _Reader(encoded)
        magic, version, table_count_raw = reader.unpack("<4sHH", "construction header")
        if magic != GOLDILOCKS_STATIC_BYTE_CONSTRUCTION_MAGIC_V3:
            raise ProofV3Error("static byte construction magic is unsupported")
        if version != GOLDILOCKS_STATIC_BYTE_CONSTRUCTION_FORMAT_VERSION_V3:
            raise ProofV3Error("static byte construction version is unsupported")
        table_count = int(table_count_raw)
        if not 0 < table_count <= MAX_GOLDILOCKS_STATIC_BYTE_REFERENCE_CONSTRUCTION_TABLES_V3:
            raise ProofV3Error("static byte construction table count is out of range")
        layout_digest = reader.read(32, "construction layout digest")
        abi_id = reader.identifier("construction ABI")
        table_hash_abi_id = reader.identifier("construction hash ABI")
        padding_rule_id = reader.identifier("construction padding rule")
        cell_lowering_id = reader.identifier("construction cell lowering")
        result = cls(
            static_table_layout_digest=layout_digest,
            tables=tuple(
                GoldilocksStaticByteConstructionTableReferenceV3(
                    table_id=reader.identifier("construction table_id"),
                    descriptor_digest=reader.read(32, "construction descriptor digest"),
                )
                for _ in range(table_count)
            ),
            abi_id=abi_id,
            table_hash_abi_id=table_hash_abi_id,
            padding_rule_id=padding_rule_id,
            cell_lowering_id=cell_lowering_id,
            format_version=version,
        )
        reader.finish()
        if result.canonical_bytes() != encoded:
            raise ProofV3Error("static byte construction is not canonical")
        return result


def make_goldilocks_static_byte_construction_reference_v3(
    *,
    static_table_layout_digest: bytes,
    static_tables: Sequence[StaticTableDescriptorV3],
) -> GoldilocksStaticByteConstructionReferenceV3:
    """Build canonical raw-int8 construction bytes for an isolated qualifier.

    This helper does not make a profile qualified.  A real qualifier must
    publish/sign the returned bytes and the matching static root before any
    validator may use them.
    """

    try:
        tables = tuple(static_tables)
    except TypeError as exc:
        raise ProofV3Error("static byte construction static tables are malformed") from exc
    if not tables or not all(isinstance(item, StaticTableDescriptorV3) for item in tables):
        raise ProofV3Error("static byte construction static tables are malformed")
    return GoldilocksStaticByteConstructionReferenceV3(
        static_table_layout_digest=static_table_layout_digest,
        tables=tuple(
            GoldilocksStaticByteConstructionTableReferenceV3(
                table_id=item.table_id,
                descriptor_digest=item.digest(),
            )
            for item in sorted(tables, key=lambda item: item.table_id)
        ),
    )


def _validated_static_tables(
    *,
    construction: GoldilocksStaticByteConstructionReferenceV3,
    source_binding: StaticTableSourceBindingV3,
    static_artifact: StaticWeightArtifactV3,
) -> dict[str, StaticTableDescriptorV3]:
    """Return the exact raw-int8 descriptor map or fail closed."""

    if not isinstance(construction, GoldilocksStaticByteConstructionReferenceV3):
        raise ProofV3VerificationError("static byte construction is malformed")
    if not isinstance(source_binding, StaticTableSourceBindingV3):
        raise ProofV3VerificationError("static byte source binding is malformed")
    if not isinstance(static_artifact, StaticWeightArtifactV3):
        raise ProofV3VerificationError("static byte static artifact is malformed")
    try:
        static_artifact.validate_source_binding(source_binding)
        if source_binding.static_table_construction_abi_id != (
            GOLDILOCKS_STATIC_TABLE_CONSTRUCTION_ABI_ID_V3
        ):
            raise ProofV3VerificationError("static byte source construction ABI is unsupported")
        if construction.abi_id != source_binding.static_table_construction_abi_id:
            raise ProofV3VerificationError(
                "static byte construction ABI does not match the authenticated source"
            )
        if source_binding.table_hash_abi_id != GOLDILOCKS_STATIC_TABLE_HASH_ABI_ID_V3:
            raise ProofV3VerificationError("static byte source hash ABI is unsupported")
        if source_binding.leaf_byte_width != 1:
            raise ProofV3VerificationError(
                "static byte reference only supports one byte per logical leaf"
            )
        page_width = source_binding.page_leaf_capacity * source_binding.leaf_byte_width
        if (
            page_width > MAX_GOLDILOCKS_STATIC_BYTE_REFERENCE_PAGE_BYTES_V3
            or source_binding.tree_leaf_count
            > MAX_GOLDILOCKS_STATIC_BYTE_REFERENCE_TREE_LEAVES_V3
            or source_binding.tree_leaf_count * page_width
            > MAX_GOLDILOCKS_STATIC_BYTE_REFERENCE_TOTAL_BYTES_V3
        ):
            raise ProofV3VerificationError("static byte table exceeds the CPU reference cap")
        if construction.raw_sha256() != source_binding.static_table_construction_digest:
            raise ProofV3VerificationError(
                "static byte construction does not match the authenticated source"
            )
        if construction.static_table_layout_digest != source_binding.static_table_layout_digest:
            raise ProofV3VerificationError(
                "static byte construction does not match the authenticated layout"
            )
        tables = {item.table_id: item for item in static_artifact.static_tables}
        expected = tuple(
            (item.table_id, item.digest())
            for item in sorted(static_artifact.static_tables, key=lambda item: item.table_id)
        )
        actual = tuple((item.table_id, item.descriptor_digest) for item in construction.tables)
        if actual != expected:
            raise ProofV3VerificationError(
                "static byte construction does not exactly cover authenticated tables"
            )
        for descriptor in tables.values():
            if descriptor.element_encoding_id != GOLDILOCKS_STATIC_BYTE_ELEMENT_ENCODING_INT8_V3:
                raise ProofV3VerificationError(
                    "static byte reference does not support this table encoding"
                )
            if descriptor.scale_encoding_id != GOLDILOCKS_STATIC_BYTE_SCALE_ENCODING_NONE_V3:
                raise ProofV3VerificationError(
                    "static byte reference does not support this table scale encoding"
                )
        return tables
    except ProofV3VerificationError:
        raise
    except (AttributeError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("static byte construction is malformed") from exc


def validate_goldilocks_static_byte_construction_reference_v3(
    *,
    construction: GoldilocksStaticByteConstructionReferenceV3,
    source_binding: StaticTableSourceBindingV3,
    static_artifact: StaticWeightArtifactV3,
) -> None:
    """Require construction bytes to match the signed raw-int8 static source."""

    _validated_static_tables(
        construction=construction,
        source_binding=source_binding,
        static_artifact=static_artifact,
    )


def _tree_binding_digest(
    *,
    construction: GoldilocksStaticByteConstructionReferenceV3,
    source_binding: StaticTableSourceBindingV3,
) -> bytes:
    """Bind all static source fields except its self-referential root.

    The signed source binding authenticates the eventual root; including that
    root in a leaf hash would create a circular construction.  Every other
    geometry/ABI/digest field is included here, together with the exact raw
    construction digest.
    """

    return hashlib.sha256(
        _TREE_BINDING_DOMAIN
        + construction.raw_sha256()
        + source_binding.canonical_model_bytes_digest
        + source_binding.source_file_index_digest
        + source_binding.canonicalization_recipe_digest
        + source_binding.static_table_construction_digest
        + source_binding.static_table_layout_digest
        + source_binding.quantization_binding_digest
        + struct.pack(
            "<QQQII",
            source_binding.logical_leaf_count,
            source_binding.static_page_count,
            source_binding.tree_leaf_count,
            source_binding.page_leaf_capacity,
            source_binding.leaf_byte_width,
        )
        + _encode_identifier(
            source_binding.canonical_model_bytes_abi_id,
            "static byte source canonical model ABI",
        )
        + _encode_identifier(
            source_binding.static_table_construction_abi_id,
            "static byte source construction ABI",
        )
        + _encode_identifier(source_binding.field_id, "static byte source field")
        + _encode_identifier(source_binding.table_hash_abi_id, "static byte source hash")
        + _encode_identifier(
            source_binding.table_encoding_abi_id,
            "static byte source encoding",
        )
        + _encode_identifier(source_binding.table_layout_abi_id, "static byte source layout")
    ).digest()


def _tree_header(
    *,
    binding_digest: bytes,
    tree_leaf_count: int,
    page_byte_width: int,
) -> bytes:
    return binding_digest + struct.pack("<QI", tree_leaf_count, page_byte_width)


def _leaf_hash(
    *,
    binding_digest: bytes,
    tree_leaf_count: int,
    page_byte_width: int,
    index: int,
    page: bytes,
) -> bytes:
    return hashlib.sha256(
        _TREE_LEAF_DOMAIN
        + _tree_header(
            binding_digest=binding_digest,
            tree_leaf_count=tree_leaf_count,
            page_byte_width=page_byte_width,
        )
        + struct.pack("<Q", index)
        + page
    ).digest()


def _parent_hash(
    *,
    binding_digest: bytes,
    tree_leaf_count: int,
    page_byte_width: int,
    level: int,
    index: int,
    left: bytes,
    right: bytes,
) -> bytes:
    return hashlib.sha256(
        _TREE_NODE_DOMAIN
        + _tree_header(
            binding_digest=binding_digest,
            tree_leaf_count=tree_leaf_count,
            page_byte_width=page_byte_width,
        )
        + struct.pack("<IQ", level, index)
        + _fixed32(left, "static byte tree left node")
        + _fixed32(right, "static byte tree right node")
    ).digest()


def _root_hash(
    *,
    binding_digest: bytes,
    tree_leaf_count: int,
    page_byte_width: int,
    raw_root: bytes,
) -> bytes:
    return hashlib.sha256(
        _TREE_ROOT_DOMAIN
        + _tree_header(
            binding_digest=binding_digest,
            tree_leaf_count=tree_leaf_count,
            page_byte_width=page_byte_width,
        )
        + _fixed32(raw_root, "static byte tree raw root")
    ).digest()


def _build_levels(
    *,
    binding_digest: bytes,
    pages: tuple[bytes, ...],
    page_byte_width: int,
) -> tuple[tuple[bytes, ...], ...]:
    leaf_count = len(pages)
    current = tuple(
        _leaf_hash(
            binding_digest=binding_digest,
            tree_leaf_count=leaf_count,
            page_byte_width=page_byte_width,
            index=index,
            page=page,
        )
        for index, page in enumerate(pages)
    )
    levels: list[tuple[bytes, ...]] = [current]
    level = 1
    while len(current) > 1:
        current = tuple(
            _parent_hash(
                binding_digest=binding_digest,
                tree_leaf_count=leaf_count,
                page_byte_width=page_byte_width,
                level=level,
                index=index // 2,
                left=current[index],
                right=current[index + 1],
            )
            for index in range(0, len(current), 2)
        )
        levels.append(current)
        level += 1
    return tuple(levels)


def _zero_tree_node(
    *,
    binding_digest: bytes,
    tree_leaf_count: int,
    page_byte_width: int,
    level: int,
    index: int,
) -> bytes:
    """Return one deterministic all-zero subtree node.

    Raw byte-tree leaves include their absolute index, so zero leaves cannot
    share one digest.  The bounded recursive calculation is used only for
    verifier-side checks of fully tree-padding sibling subtrees.
    """

    if level == 0:
        return _leaf_hash(
            binding_digest=binding_digest,
            tree_leaf_count=tree_leaf_count,
            page_byte_width=page_byte_width,
            index=index,
            page=bytes(page_byte_width),
        )
    return _parent_hash(
        binding_digest=binding_digest,
        tree_leaf_count=tree_leaf_count,
        page_byte_width=page_byte_width,
        level=level,
        index=index,
        left=_zero_tree_node(
            binding_digest=binding_digest,
            tree_leaf_count=tree_leaf_count,
            page_byte_width=page_byte_width,
            level=level - 1,
            index=index * 2,
        ),
        right=_zero_tree_node(
            binding_digest=binding_digest,
            tree_leaf_count=tree_leaf_count,
            page_byte_width=page_byte_width,
            level=level - 1,
            index=index * 2 + 1,
        ),
    )


def _normalized_pages(
    value: object,
    *,
    source_binding: StaticTableSourceBindingV3,
) -> tuple[bytes, ...]:
    page_width = source_binding.page_leaf_capacity * source_binding.leaf_byte_width
    raw_pages = _materialize_tuple(
        value,
        name="static byte pages",
        maximum=source_binding.static_page_count,
    )
    if len(raw_pages) != source_binding.static_page_count:
        raise ProofV3Error("static byte pages do not match the signed page count")
    pages: list[bytes] = []
    for index, page in enumerate(raw_pages):
        if not isinstance(page, bytes) or len(page) != page_width:
            raise ProofV3Error(
                f"static byte page {index} does not match the signed page width"
            )
        pages.append(page)
    return tuple(pages)


def _validate_page_padding(
    *,
    pages: tuple[bytes, ...],
    source_binding: StaticTableSourceBindingV3,
    tables: dict[str, StaticTableDescriptorV3],
) -> None:
    """Require unused leaf slots in every real descriptor page to be zero."""

    capacity = source_binding.page_leaf_capacity
    width = source_binding.leaf_byte_width
    by_page: dict[int, StaticTableDescriptorV3] = {}
    for descriptor in tables.values():
        for page_index in range(descriptor.page_start, descriptor.page_start + descriptor.page_count):
            if page_index in by_page:
                raise ProofV3Error("static byte descriptor pages are overlapping")
            by_page[page_index] = descriptor
    if tuple(sorted(by_page)) != tuple(range(source_binding.static_page_count)):
        raise ProofV3Error("static byte descriptor pages do not cover the static page domain")
    for page_index, page in enumerate(pages):
        descriptor = by_page[page_index]
        local_page = page_index - descriptor.page_start
        used = max(0, min(capacity, descriptor.logical_leaf_count - local_page * capacity))
        if page[used * width :] != bytes((capacity - used) * width):
            raise ProofV3Error("static byte page has nonzero unused leaf padding")


def _validate_opened_page_padding(
    *,
    page_indices: tuple[int, ...],
    pages: tuple[bytes, ...],
    source_binding: StaticTableSourceBindingV3,
    tables: dict[str, StaticTableDescriptorV3],
) -> None:
    """Check zero padding only for the real pages present in one opening."""

    capacity = source_binding.page_leaf_capacity
    width = source_binding.leaf_byte_width
    by_page: dict[int, StaticTableDescriptorV3] = {}
    for descriptor in tables.values():
        for page_index in range(descriptor.page_start, descriptor.page_start + descriptor.page_count):
            by_page[page_index] = descriptor
    for page_index, page in zip(page_indices, pages, strict=True):
        descriptor = by_page.get(page_index)
        if descriptor is None:
            raise ProofV3VerificationError("static byte opening includes a non-static page")
        local_page = page_index - descriptor.page_start
        used = max(0, min(capacity, descriptor.logical_leaf_count - local_page * capacity))
        if page[used * width :] != bytes((capacity - used) * width):
            raise ProofV3VerificationError("static byte opening has nonzero unused leaf padding")


@dataclass(frozen=True, slots=True)
class GoldilocksStaticByteSiblingReferenceV3:
    """One canonical raw node in a static byte-page Merkle multiproof."""

    level: int
    index: int
    digest: bytes

    def __post_init__(self) -> None:
        _u32(self.level, "static byte sibling level")
        _u64(self.index, "static byte sibling index")
        _fixed32(self.digest, "static byte sibling digest")


@dataclass(frozen=True, slots=True)
class GoldilocksStaticByteMultiOpeningReferenceV3:
    """Exact pages and canonical sibling schedule for one static-root opening."""

    binding_digest: bytes
    tree_leaf_count: int
    page_byte_width: int
    page_indices: tuple[int, ...]
    pages: tuple[bytes, ...]
    siblings: tuple[GoldilocksStaticByteSiblingReferenceV3, ...]
    abi_id: str = GOLDILOCKS_STATIC_BYTE_REFERENCE_ABI_V3
    format_version: int = GOLDILOCKS_STATIC_BYTE_REFERENCE_FORMAT_VERSION_V3

    def __post_init__(self) -> None:
        if self.abi_id != GOLDILOCKS_STATIC_BYTE_REFERENCE_ABI_V3:
            raise ProofV3Error("static byte opening ABI is unsupported")
        if type(self.format_version) is not int or self.format_version != (
            GOLDILOCKS_STATIC_BYTE_REFERENCE_FORMAT_VERSION_V3
        ):
            raise ProofV3Error("static byte opening format version is unsupported")
        binding = _fixed32(self.binding_digest, "static byte opening binding digest", nonzero=True)
        tree_leaf_count = _power_of_two(
            _u64(self.tree_leaf_count, "static byte opening tree leaf count", positive=True),
            "static byte opening tree leaf count",
        )
        if tree_leaf_count > MAX_GOLDILOCKS_STATIC_BYTE_REFERENCE_TREE_LEAVES_V3:
            raise ProofV3Error("static byte opening tree exceeds the CPU reference cap")
        page_width = _u32(
            self.page_byte_width,
            "static byte opening page width",
            positive=True,
        )
        if page_width > MAX_GOLDILOCKS_STATIC_BYTE_REFERENCE_PAGE_BYTES_V3:
            raise ProofV3Error("static byte opening page exceeds the CPU reference cap")
        indices = _opening_indices(
            self.page_indices,
            maximum=tree_leaf_count,
            name="static byte opening page indices",
        )
        raw_pages = _materialize_tuple(
            self.pages,
            name="static byte opening pages",
            maximum=len(indices),
        )
        if len(raw_pages) != len(indices) or not all(
            isinstance(page, bytes) and len(page) == page_width for page in raw_pages
        ):
            raise ProofV3Error("static byte opening pages are malformed")
        raw_siblings = _materialize_tuple(
            self.siblings,
            name="static byte opening siblings",
            maximum=tree_leaf_count,
            allow_empty=True,
        )
        if not all(isinstance(item, GoldilocksStaticByteSiblingReferenceV3) for item in raw_siblings):
            raise ProofV3Error("static byte opening sibling is malformed")
        siblings = tuple(raw_siblings)
        expected_coordinates = _expected_sibling_coordinates(
            leaf_count=tree_leaf_count,
            indices=indices,
        )
        if tuple((item.level, item.index) for item in siblings) != expected_coordinates:
            raise ProofV3Error(
                "static byte opening siblings are incomplete, duplicate, or reordered"
            )
        object.__setattr__(self, "binding_digest", binding)
        object.__setattr__(self, "tree_leaf_count", tree_leaf_count)
        object.__setattr__(self, "page_byte_width", page_width)
        object.__setattr__(self, "page_indices", indices)
        object.__setattr__(self, "pages", tuple(raw_pages))
        object.__setattr__(self, "siblings", siblings)


@dataclass(frozen=True, slots=True)
class GoldilocksStaticByteTableOracleReferenceV3:
    """Small retained byte tree used only to construct isolated golden vectors."""

    binding_digest: bytes
    static_page_count: int
    tree_leaf_count: int
    page_byte_width: int
    static_table_root: bytes
    pages: tuple[bytes, ...]
    levels: tuple[tuple[bytes, ...], ...]

    def open(self, page_indices: object) -> GoldilocksStaticByteMultiOpeningReferenceV3:
        indices = _opening_indices(
            page_indices,
            maximum=self.static_page_count,
            name="static byte oracle page indices",
        )
        return GoldilocksStaticByteMultiOpeningReferenceV3(
            binding_digest=self.binding_digest,
            tree_leaf_count=self.tree_leaf_count,
            page_byte_width=self.page_byte_width,
            page_indices=indices,
            pages=tuple(self.pages[index] for index in indices),
            siblings=tuple(
                GoldilocksStaticByteSiblingReferenceV3(
                    level=level,
                    index=index,
                    digest=self.levels[level][index],
                )
                for level, index in _expected_sibling_coordinates(
                    leaf_count=self.tree_leaf_count,
                    indices=indices,
                )
            ),
        )


def _candidate_tree(
    *,
    construction: GoldilocksStaticByteConstructionReferenceV3,
    source_binding: StaticTableSourceBindingV3,
    static_artifact: StaticWeightArtifactV3,
    pages: object,
) -> tuple[bytes, tuple[bytes, ...], tuple[tuple[bytes, ...], ...], bytes]:
    tables = _validated_static_tables(
        construction=construction,
        source_binding=source_binding,
        static_artifact=static_artifact,
    )
    normalized_pages = _normalized_pages(pages, source_binding=source_binding)
    _validate_page_padding(
        pages=normalized_pages,
        source_binding=source_binding,
        tables=tables,
    )
    page_width = source_binding.page_leaf_capacity * source_binding.leaf_byte_width
    padded_pages = normalized_pages + (bytes(page_width),) * (
        source_binding.tree_leaf_count - source_binding.static_page_count
    )
    binding = _tree_binding_digest(
        construction=construction,
        source_binding=source_binding,
    )
    levels = _build_levels(
        binding_digest=binding,
        pages=padded_pages,
        page_byte_width=page_width,
    )
    root = _root_hash(
        binding_digest=binding,
        tree_leaf_count=source_binding.tree_leaf_count,
        page_byte_width=page_width,
        raw_root=levels[-1][0],
    )
    return root, normalized_pages, levels, binding


def compute_goldilocks_static_byte_table_root_reference_v3(
    *,
    construction: GoldilocksStaticByteConstructionReferenceV3,
    source_binding: StaticTableSourceBindingV3,
    static_artifact: StaticWeightArtifactV3,
    pages: object,
) -> bytes:
    """Compute a candidate root from pages under the signed source projection.

    This exists for offline qualification/test-vector generation.  It does
    *not* authenticate a candidate root by itself; a verifier must use
    :func:`build_goldilocks_static_byte_table_oracle_reference_v3` or
    :func:`verify_goldilocks_static_byte_opening_reference_v3`, both of which
    require equality with the source binding's signed ``static_table_root``.
    """

    root, _pages, _levels, _binding = _candidate_tree(
        construction=construction,
        source_binding=source_binding,
        static_artifact=static_artifact,
        pages=pages,
    )
    return root


def build_goldilocks_static_byte_table_oracle_reference_v3(
    *,
    construction: GoldilocksStaticByteConstructionReferenceV3,
    source_binding: StaticTableSourceBindingV3,
    static_artifact: StaticWeightArtifactV3,
    pages: object,
) -> GoldilocksStaticByteTableOracleReferenceV3:
    """Retain a bounded authenticated byte-page tree for golden-vector tests."""

    root, normalized_pages, levels, binding = _candidate_tree(
        construction=construction,
        source_binding=source_binding,
        static_artifact=static_artifact,
        pages=pages,
    )
    if not hmac.compare_digest(root, source_binding.static_table_root):
        raise ProofV3Error("static byte pages do not match the authenticated static root")
    return GoldilocksStaticByteTableOracleReferenceV3(
        binding_digest=binding,
        static_page_count=source_binding.static_page_count,
        tree_leaf_count=source_binding.tree_leaf_count,
        page_byte_width=source_binding.page_leaf_capacity * source_binding.leaf_byte_width,
        static_table_root=root,
        pages=normalized_pages,
        levels=levels,
    )


def verify_goldilocks_static_byte_opening_reference_v3(
    opening: object,
    *,
    construction: GoldilocksStaticByteConstructionReferenceV3,
    source_binding: StaticTableSourceBindingV3,
    static_artifact: StaticWeightArtifactV3,
    expected_page_indices: object,
) -> tuple[bytes, ...]:
    """Verify exact authenticated raw pages derived by the validator.

    The caller must derive every expected page from signed table coordinates;
    opening-selected indices, metadata, and siblings never choose the set.
    """

    try:
        tables = _validated_static_tables(
            construction=construction,
            source_binding=source_binding,
            static_artifact=static_artifact,
        )
        expected = _opening_indices(
            expected_page_indices,
            maximum=source_binding.static_page_count,
            name="expected static byte page indices",
        )
        if not isinstance(opening, GoldilocksStaticByteMultiOpeningReferenceV3):
            raise ProofV3VerificationError("static byte opening is malformed")
        binding = _tree_binding_digest(
            construction=construction,
            source_binding=source_binding,
        )
        page_width = source_binding.page_leaf_capacity * source_binding.leaf_byte_width
        if (
            opening.binding_digest != binding
            or opening.tree_leaf_count != source_binding.tree_leaf_count
            or opening.page_byte_width != page_width
            or opening.page_indices != expected
        ):
            raise ProofV3VerificationError("static byte opening metadata is unexpected")
        expected_coordinates = _expected_sibling_coordinates(
            leaf_count=source_binding.tree_leaf_count,
            indices=expected,
        )
        if tuple((item.level, item.index) for item in opening.siblings) != expected_coordinates:
            raise ProofV3VerificationError(
                "static byte opening siblings are incomplete, duplicate, or reordered"
            )
        for sibling in opening.siblings:
            if sibling.index * (1 << sibling.level) >= source_binding.static_page_count:
                expected_zero = _zero_tree_node(
                    binding_digest=binding,
                    tree_leaf_count=source_binding.tree_leaf_count,
                    page_byte_width=page_width,
                    level=sibling.level,
                    index=sibling.index,
                )
                if not hmac.compare_digest(sibling.digest, expected_zero):
                    raise ProofV3VerificationError(
                        "static byte opening has nonzero tree padding"
                    )
        current = {
            index: _leaf_hash(
                binding_digest=binding,
                tree_leaf_count=source_binding.tree_leaf_count,
                page_byte_width=page_width,
                index=index,
                page=page,
            )
            for index, page in zip(expected, opening.pages, strict=True)
        }
        sibling_offset = 0
        for level in range(_tree_height(source_binding.tree_leaf_count)):
            active = tuple(sorted(current))
            expected_level = tuple(
                coordinate for coordinate in expected_coordinates if coordinate[0] == level
            )
            for _unused, sibling_index in expected_level:
                sibling = opening.siblings[sibling_offset]
                sibling_offset += 1
                current[sibling_index] = sibling.digest
            parents: dict[int, bytes] = {}
            for parent_index in sorted({index // 2 for index in active}):
                left = current.get(parent_index * 2)
                right = current.get(parent_index * 2 + 1)
                if left is None or right is None:
                    raise ProofV3VerificationError("static byte opening is missing a sibling")
                parents[parent_index] = _parent_hash(
                    binding_digest=binding,
                    tree_leaf_count=source_binding.tree_leaf_count,
                    page_byte_width=page_width,
                    level=level + 1,
                    index=parent_index,
                    left=left,
                    right=right,
                )
            current = parents
        if sibling_offset != len(opening.siblings) or set(current) != {0}:
            raise ProofV3VerificationError("static byte opening reconstruction is incomplete")
        root = _root_hash(
            binding_digest=binding,
            tree_leaf_count=source_binding.tree_leaf_count,
            page_byte_width=page_width,
            raw_root=current[0],
        )
        if not hmac.compare_digest(root, source_binding.static_table_root):
            raise ProofV3VerificationError("static byte opening does not match root")
        # Check observable real-page padding as well; hidden pages are bound by
        # the Merkle root and cannot be changed without invalidating it.
        _validate_opened_page_padding(
            page_indices=expected,
            pages=opening.pages,
            source_binding=source_binding,
            tables=tables,
        )
        return opening.pages
    except ProofV3VerificationError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("static byte opening is malformed") from exc


@dataclass(frozen=True, slots=True)
class _GoldilocksStaticTraceCellsWitnessReferenceV3:
    """Test-only retained witness for local raw fixed-cell conformance."""

    base_trace: GoldilocksConstraintTraceReferenceV3
    static_opening: GoldilocksStaticByteMultiOpeningReferenceV3

    def __post_init__(self) -> None:
        if not isinstance(self.base_trace, GoldilocksConstraintTraceReferenceV3):
            raise ProofV3Error("static trace binding base trace is malformed")
        if not isinstance(self.static_opening, GoldilocksStaticByteMultiOpeningReferenceV3):
            raise ProofV3Error("static trace binding opening is malformed")


def _expected_static_pages_for_trace(
    *,
    program: GoldilocksConstraintProgramV3,
    token_count: int,
    static_tables: dict[str, StaticTableDescriptorV3],
    source_binding: StaticTableSourceBindingV3,
) -> tuple[int, ...]:
    if program.source_binding_mode != GOLDILOCKS_TRACE_SOURCE_BINDING_MODE_EXACT_LAYOUT_V3:
        raise ProofV3VerificationError("static trace binding requires exact source bindings")
    if not program.static_column_bindings:
        raise ProofV3VerificationError("static trace binding program has no fixed columns")
    active_rows = token_count * program.layout_binding.rows_per_token
    pages: set[int] = set()
    for binding in program.static_column_bindings:
        descriptor = static_tables.get(binding.table_id)
        if descriptor is None:
            raise ProofV3VerificationError("static trace binding references an unknown table")
        if (
            binding.cell_encoding_id != descriptor.element_encoding_id
            or binding.cell_encoding_id != GOLDILOCKS_STATIC_BYTE_ELEMENT_ENCODING_INT8_V3
        ):
            raise ProofV3VerificationError("static trace binding has an unsupported cell encoding")
        last_leaf = binding.logical_leaf_offset + (
            (active_rows - 1) * binding.trace_row_stride
        )
        if last_leaf >= descriptor.logical_leaf_count:
            raise ProofV3VerificationError("static trace binding exceeds its signed table range")
        for row in range(active_rows):
            local_leaf = binding.logical_leaf_offset + row * binding.trace_row_stride
            page_index = descriptor.page_start + local_leaf // source_binding.page_leaf_capacity
            if page_index >= descriptor.page_start + descriptor.page_count:
                raise ProofV3VerificationError("static trace binding exceeds its signed page range")
            pages.add(page_index)
    return tuple(sorted(pages))


def _decode_signed_int8(value: int) -> int:
    signed = value - 256 if value >= 128 else value
    return signed % GOLDILOCKS_MODULUS


def _verify_static_trace_cells(
    *,
    program: GoldilocksConstraintProgramV3,
    trace: GoldilocksConstraintTraceReferenceV3,
    token_count: int,
    static_tables: dict[str, StaticTableDescriptorV3],
    source_binding: StaticTableSourceBindingV3,
    pages_by_index: dict[int, bytes],
) -> None:
    positions = {
        column.column_id: index for index, column in enumerate(program.trace_columns)
    }
    active_rows = token_count * program.layout_binding.rows_per_token
    for binding in program.static_column_bindings:
        descriptor = static_tables[binding.table_id]
        position = positions.get(binding.column_id)
        if position is None:
            raise ProofV3VerificationError("static trace binding references an unknown column")
        for row in range(active_rows):
            local_leaf = binding.logical_leaf_offset + row * binding.trace_row_stride
            page_index = descriptor.page_start + local_leaf // source_binding.page_leaf_capacity
            slot = local_leaf % source_binding.page_leaf_capacity
            page = pages_by_index.get(page_index)
            if page is None:
                raise ProofV3VerificationError("static trace binding opening omits a page")
            expected = _decode_signed_int8(page[slot])
            if trace.rows[row][position] != expected:
                raise ProofV3VerificationError(
                    "static trace cell does not match its authenticated byte"
                )


def _verify_goldilocks_static_trace_cells_reference_v3(
    witness: object,
    *,
    program: GoldilocksConstraintProgramV3,
    token_count: int,
    validator_binding_digest: bytes,
    trace_precommitment: GoldilocksAirTracePrecommitmentReferenceV3,
    construction: GoldilocksStaticByteConstructionReferenceV3,
    source_binding: StaticTableSourceBindingV3,
    static_artifact: StaticWeightArtifactV3,
) -> None:
    """Test-only local check of raw fixed cells against a signed static root.

    This intentionally accepts caller-supplied ``program`` and
    ``trace_precommitment`` objects.  It is therefore not admissible in a hard
    audit: a future wrapper must derive the exact signed slot/program from a
    validator-sealed trace-map receipt and bind that receipt atomically to a
    pre-nonce runtime tensor receipt.  The primitive also proves neither
    static lookup/range/carry semantics nor model arithmetic.
    """

    try:
        if not isinstance(witness, _GoldilocksStaticTraceCellsWitnessReferenceV3):
            raise ProofV3VerificationError("static trace binding witness is malformed")
        if not isinstance(program, GoldilocksConstraintProgramV3):
            raise ProofV3VerificationError("static trace binding program is malformed")
        if not isinstance(trace_precommitment, GoldilocksAirTracePrecommitmentReferenceV3):
            raise ProofV3VerificationError("static trace binding precommitment is malformed")
        binding_digest = _fixed32(
            validator_binding_digest,
            "static trace binding validator digest",
            nonzero=True,
        )
        checked_token_count = _u64(token_count, "static trace binding token count", positive=True)
        static_tables = _validated_static_tables(
            construction=construction,
            source_binding=source_binding,
            static_artifact=static_artifact,
        )
        verify_goldilocks_constraint_program_reference_v3(
            program=program,
            trace=witness.base_trace,
            token_count=checked_token_count,
        )
        reconstructed = build_goldilocks_air_trace_oracle_reference_v3(
            program=program,
            trace=witness.base_trace,
            token_count=checked_token_count,
            validator_binding_digest=binding_digest,
        )
        if reconstructed.precommitment.digest() != trace_precommitment.digest():
            raise ProofV3VerificationError(
                "static trace binding trace does not match its frozen trace root"
            )
        expected_pages = _expected_static_pages_for_trace(
            program=program,
            token_count=checked_token_count,
            static_tables=static_tables,
            source_binding=source_binding,
        )
        pages = verify_goldilocks_static_byte_opening_reference_v3(
            witness.static_opening,
            construction=construction,
            source_binding=source_binding,
            static_artifact=static_artifact,
            expected_page_indices=expected_pages,
        )
        _verify_static_trace_cells(
            program=program,
            trace=witness.base_trace,
            token_count=checked_token_count,
            static_tables=static_tables,
            source_binding=source_binding,
            pages_by_index=dict(zip(expected_pages, pages, strict=True)),
        )
    except ProofV3VerificationError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("static trace binding witness is malformed") from exc


__all__ = [
    "GOLDILOCKS_STATIC_BYTE_CONSTRUCTION_FORMAT_VERSION_V3",
    "GOLDILOCKS_STATIC_BYTE_CONSTRUCTION_MAGIC_V3",
    "GOLDILOCKS_STATIC_BYTE_CONSTRUCTION_ABI_V3",
    "GOLDILOCKS_STATIC_BYTE_ELEMENT_ENCODING_INT8_V3",
    "GOLDILOCKS_STATIC_BYTE_LOWERING_SIGNED_INT8_V3",
    "GOLDILOCKS_STATIC_BYTE_PADDING_RULE_V3",
    "GOLDILOCKS_STATIC_BYTE_REFERENCE_ABI_V3",
    "GOLDILOCKS_STATIC_BYTE_REFERENCE_FORMAT_VERSION_V3",
    "GOLDILOCKS_STATIC_BYTE_SCALE_ENCODING_NONE_V3",
    "GoldilocksStaticByteConstructionReferenceV3",
    "GoldilocksStaticByteConstructionTableReferenceV3",
    "GoldilocksStaticByteMultiOpeningReferenceV3",
    "GoldilocksStaticByteSiblingReferenceV3",
    "GoldilocksStaticByteTableOracleReferenceV3",
    "build_goldilocks_static_byte_table_oracle_reference_v3",
    "compute_goldilocks_static_byte_table_root_reference_v3",
    "make_goldilocks_static_byte_construction_reference_v3",
    "validate_goldilocks_static_byte_construction_reference_v3",
    "verify_goldilocks_static_byte_opening_reference_v3",
]
