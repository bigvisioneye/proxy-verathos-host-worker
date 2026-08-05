"""Canonical dual-root static-weight artifact for proof-v3 hard adapters.

Proof-v2 already has an authority-signed static manifest and Pallas column
catalog.  A proof-v3 dynamic AIR/FRI backend must authenticate the *same*
model through a separate fixed-table commitment without pretending that a
Pallas point, a Mersenne value, or a byte hash can be cast into Goldilocks.

This immutable artifact joins those independent commitments through their
common, authority-qualified canonical source representation:

* the verified v2 manifest/model-spec identity and source file hash;
* the exact canonical Pallas catalog bytes and PCS parameters;
* a canonical model-byte recipe and source-file index;
* the exact dynamic backend, field, Merkle, page-layout, and kernel ABI; and
* a complete, non-overlapping catalog of static-table descriptors.

The artifact is a statement, not a proof of construction.  An authority-signed
V3 execution profile binds its digest; a qualified loader independently checks
the v2 leg and loaded dynamic artifacts before registering a native adapter.
The validator never has to load model weights.  No runtime path imports this
module yet, so it cannot accidentally make an unqualified proof acceptable.
"""

from __future__ import annotations

import hashlib
import re
import struct
import unicodedata
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError


STATIC_WEIGHT_ARTIFACT_FORMAT_VERSION_V3 = 2
STATIC_WEIGHT_ARTIFACT_MAGIC_V3 = b"V3SA"
STATIC_WEIGHT_ARTIFACT_DIGEST_DOMAIN_V3 = (
    b"VERATHOS/PROOF_V3/STATIC_WEIGHT_ARTIFACT/SHA256"
)
STATIC_TABLE_SOURCE_BINDING_MAGIC_V3 = b"V3SB"
STATIC_TABLE_SOURCE_BINDING_FORMAT_VERSION_V3 = 1
STATIC_TABLE_SOURCE_BINDING_DIGEST_DOMAIN_V3 = (
    b"VERATHOS/PROOF_V3/STATIC_TABLE_SOURCE_BINDING/SHA256"
)
STATIC_TABLE_DESCRIPTOR_DIGEST_DOMAIN_V3 = (
    b"VERATHOS/PROOF_V3/STATIC_TABLE_DESCRIPTOR/SHA256"
)
STATIC_TABLE_LAYOUT_MAGIC_V3 = b"V3SL"
STATIC_TABLE_LAYOUT_FORMAT_VERSION_V3 = 1
QUANTIZATION_BINDING_MAGIC_V3 = b"V3QB"
QUANTIZATION_BINDING_FORMAT_VERSION_V3 = 1

# These exact ABI values name the initial dynamic field/table family.  A
# different backend is possible only through an explicitly signed artifact and
# separately qualified native adapter; callers must never infer equivalence
# from a broadly similar field or hash label.
GOLDILOCKS_STATIC_FIELD_ID_V3 = "goldilocks64.v1"
GOLDILOCKS_STATIC_TABLE_HASH_ABI_ID_V3 = "sha256.merkle.binary.v1"
GOLDILOCKS_STATIC_TABLE_ENCODING_ABI_ID_V3 = "quantized_weight_blocks.v1"
GOLDILOCKS_STATIC_TABLE_LAYOUT_ABI_ID_V3 = "static_table.layout.v1"
CANONICAL_MODEL_BYTES_ABI_ID_V3 = "model_bytes.canonical.v1"
QUANTIZATION_BINDING_ABI_ID_V3 = "model.quantization_binding.v1"
# The dynamic proof system and its static tables must share the same field
# family.  Version one intentionally supports one reviewed Goldilocks backend
# only; accepting a syntactically valid but algebraically unrelated backend
# would reintroduce an unauthenticated cross-field handoff.
GOLDILOCKS_DYNAMIC_BACKEND_ABI_ID_V3 = "goldilocks.stark.fri.v1"
GOLDILOCKS_STATIC_TABLE_CONSTRUCTION_ABI_ID_V3 = (
    "goldilocks.static_table_construction.v1"
)
PALLAS_COLUMN_CATALOG_ABI_ID_V3 = "pallas.column_catalog.v2"
# These are the only V2 static PCS parameters whose column roots the V3
# bridge knows how to reconstruct. Treating the suite/generator as descriptive
# metadata would allow a signed-but-different PCS claim to be paired with the
# fixed V2 verifier implementation.
V2_PALLAS_PCS_SUITE_V3 = "pallas-pedersen-ipa-v1"
V2_PALLAS_GENERATOR_VERSION_V3 = "verathos-pcs-v2-pallas-pedersen-gens-v1"

MAX_STATIC_WEIGHT_ARTIFACT_BYTES_V3 = 4 << 20
MAX_STATIC_TABLE_DESCRIPTORS_V3 = 65_535
# ``page_leaf_capacity`` is encoded as a nonzero u32. Its largest canonical
# power-of-two value is therefore 2**31; a larger advertised cap would be
# unreachable on the authenticated wire format.
MAX_STATIC_TABLE_LEAVES_V3 = 1 << 31
MAX_STATIC_TABLE_PAGE_COUNT_V3 = 1 << 32
MAX_STATIC_TABLE_LEAF_BYTES_V3 = 1 << 20
MAX_STATIC_TABLE_PAGES_PER_DESCRIPTOR_V3 = 1 << 32
MAX_STATIC_TABLE_SOURCE_TENSOR_IDS_V3 = 32
MAX_IDENTIFIER_BYTES_V3 = 128
MAX_MODEL_ID_BYTES_V3 = 4096
MAX_QUANTIZATION_BINDING_BYTES_V3 = 4 << 10
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,127}$")


def _fixed32(value: object, name: str, *, nonzero: bool = False) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    if nonzero and value == bytes(32):
        raise ProofV3Error(f"{name} must not be the zero digest")
    return value


def _address(value: object, name: str) -> bytes:
    if type(value) is not bytes or len(value) != 20 or value == bytes(20):
        raise ProofV3Error(f"{name} must be a nonzero 20-byte address")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ProofV3Error(f"{name} must be a string")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ProofV3Error(f"{name} must be ASCII") from exc
    if (
        not encoded
        or len(encoded) > MAX_IDENTIFIER_BYTES_V3
        or _IDENTIFIER_RE.fullmatch(value) is None
    ):
        raise ProofV3Error(f"{name} is not a canonical identifier")
    return value


def _model_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ProofV3Error("model_id must be a non-empty string")
    if value != unicodedata.normalize("NFC", value) or "\x00" in value:
        raise ProofV3Error("model_id must use canonical NFC text without NUL")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ProofV3Error("model_id must be valid UTF-8 text") from exc
    if len(encoded) > MAX_MODEL_ID_BYTES_V3:
        raise ProofV3Error("model_id exceeds the protocol limit")
    return value


def _model_quant_mode(value: object, name: str = "model_quant_mode") -> str:
    """Validate the exact V2 ModelSpec ``quant_mode`` text representation.

    V2 intentionally treats ``quant_mode`` as canonical NFC text rather than
    an enum: model-qualified loaders may need names such as ``gptq_int4`` or
    a future vendor-specific scheme.  The V3 binding mirrors that byte-level
    contract instead of guessing a lossy mapping from the semantics ID.
    """

    if not isinstance(value, str) or not value:
        raise ProofV3Error(f"{name} must be a non-empty string")
    if value != unicodedata.normalize("NFC", value) or "\x00" in value:
        raise ProofV3Error(f"{name} must use canonical NFC text without NUL")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ProofV3Error(f"{name} must be valid UTF-8 text") from exc
    if len(encoded) > MAX_IDENTIFIER_BYTES_V3:
        raise ProofV3Error(f"{name} exceeds the protocol limit")
    return value


def _u32(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise ProofV3Error(f"{name} must be an unsigned 32-bit integer")
    minimum = 1 if positive else 0
    if not minimum <= value < 1 << 32:
        qualifier = "positive " if positive else ""
        raise ProofV3Error(f"{name} must be a {qualifier}unsigned 32-bit integer")
    return value


def _u64(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise ProofV3Error(f"{name} must be an unsigned 64-bit integer")
    minimum = 1 if positive else 0
    if not minimum <= value < 1 << 64:
        qualifier = "positive " if positive else ""
        raise ProofV3Error(f"{name} must be a {qualifier}unsigned 64-bit integer")
    return value


def _power_of_two(value: int, name: str, maximum: int) -> int:
    if value > maximum or value & (value - 1):
        raise ProofV3Error(f"{name} must be a supported power of two")
    return value


def _encode_identifier(value: str, name: str) -> bytes:
    encoded = _identifier(value, name).encode("ascii")
    return struct.pack("<B", len(encoded)) + encoded


def _encode_model_id(value: str) -> bytes:
    encoded = _model_id(value).encode("utf-8")
    return struct.pack("<H", len(encoded)) + encoded


def _encode_model_quant_mode(value: str) -> bytes:
    encoded = _model_quant_mode(value).encode("utf-8")
    return struct.pack("<B", len(encoded)) + encoded


def _trusted_v2_catalog_binding_fields_v3(
    binding: object,
) -> tuple[object, bytes]:
    """Return the authenticated V2 view behind a catalog bridge.

    ``StaticWeightArtifactV3`` is a signed, serializable statement, so its
    low-level constructor and byte parser intentionally remain available for
    decoding an already signed artifact and for explicit test fixtures.  They
    are *not* the registration boundary.  Production qualification must enter
    through :meth:`StaticWeightArtifactV3.from_verified_v2_catalog_binding`
    (or derive the same context through
    :meth:`StaticWeightArtifactContextV3.from_verified_v2_catalog_binding`).

    Keep the import local: ``catalog`` imports this module to define the V3
    bridge, while this helper is only called after both modules have loaded.
    The bridge factory is responsible for verifying the original signed V2
    manifest and rebuilding every catalog tree.  At this boundary we still
    verify the immutable catalog bytes against the bridge's authenticated
    metadata, so a hand-constructed or corrupted bridge cannot silently alter
    the V2 fields copied into a V3 artifact.
    """

    try:
        from verallm.proof_v3.catalog import (
            AuthenticatedStaticManifestV3,
            VerifiedV2CatalogBindingV3,
        )
    except Exception as exc:  # pragma: no cover - package wiring failure.
        raise ProofV3VerificationError(
            "verified proof-v2 catalog bridge support is unavailable"
        ) from exc
    if not isinstance(binding, VerifiedV2CatalogBindingV3):
        raise ProofV3VerificationError(
            "static-weight qualification requires a verified proof-v2 catalog binding"
        )
    binding.require_verified_v2_provenance()
    manifest = binding.static_manifest
    catalog_bytes = binding.catalog_bytes
    if not isinstance(manifest, AuthenticatedStaticManifestV3):
        raise ProofV3VerificationError(
            "verified proof-v2 catalog binding has an unexpected manifest type"
        )
    if type(catalog_bytes) is not bytes or not catalog_bytes:
        raise ProofV3VerificationError(
            "verified proof-v2 catalog binding has malformed catalog bytes"
        )
    if (
        len(catalog_bytes) != manifest.pallas_catalog_size
        or hashlib.sha256(catalog_bytes).digest() != manifest.pallas_catalog_sha256
    ):
        raise ProofV3VerificationError(
            "verified proof-v2 catalog binding does not match its authenticated catalog"
        )
    return manifest, catalog_bytes


class _Reader:
    def __init__(self, encoded: bytes) -> None:
        if (
            type(encoded) is not bytes
            or not encoded
            or len(encoded) > MAX_STATIC_WEIGHT_ARTIFACT_BYTES_V3
        ):
            raise ProofV3Error("static-weight artifact byte length is out of range")
        self._encoded = encoded
        self._offset = 0

    def read(self, size: int) -> bytes:
        if size < 0 or self._offset + size > len(self._encoded):
            raise ProofV3Error("static-weight artifact is truncated")
        result = self._encoded[self._offset : self._offset + size]
        self._offset += size
        return result

    def unpack(self, format_string: str) -> tuple[object, ...]:
        return struct.unpack(format_string, self.read(struct.calcsize(format_string)))

    def identifier(self, name: str) -> str:
        size = self.unpack("<B")[0]
        if type(size) is not int or not 0 < size <= MAX_IDENTIFIER_BYTES_V3:
            raise ProofV3Error(f"{name} length is out of range")
        try:
            return _identifier(self.read(size).decode("ascii"), name)
        except UnicodeDecodeError as exc:
            raise ProofV3Error(f"{name} must be ASCII") from exc

    def model_id(self) -> str:
        size = self.unpack("<H")[0]
        if type(size) is not int or not 0 < size <= MAX_MODEL_ID_BYTES_V3:
            raise ProofV3Error("model_id length is out of range")
        try:
            return _model_id(self.read(size).decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ProofV3Error("model_id must be UTF-8") from exc

    def model_quant_mode(self) -> str:
        size = self.unpack("<B")[0]
        if type(size) is not int or not 0 < size <= MAX_IDENTIFIER_BYTES_V3:
            raise ProofV3Error("model_quant_mode length is out of range")
        try:
            return _model_quant_mode(self.read(size).decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ProofV3Error("model_quant_mode must be UTF-8") from exc

    def bytes(self, name: str, maximum: int) -> bytes:
        size = self.unpack("<I")[0]
        if type(size) is not int or not 0 < size <= maximum:
            raise ProofV3Error(f"{name} length is out of range")
        return self.read(size)

    def finish(self) -> None:
        if self._offset != len(self._encoded):
            raise ProofV3Error("static-weight artifact has trailing bytes")


@dataclass(frozen=True, slots=True)
class StaticTableDescriptorV3:
    """One exact source-to-page range in the dynamic static-table catalog.

    ``source_tensor_ids`` are canonical names in the signed source-file
    canonicalization recipe.  The descriptor digest is what a future relation
    profile references for a particular linear/static lookup; raw table IDs
    alone are intentionally insufficient.
    """

    table_id: str
    source_tensor_ids: tuple[str, ...]
    logical_leaf_start: int
    logical_leaf_count: int
    page_start: int
    page_count: int
    element_encoding_id: str
    scale_encoding_id: str

    def __post_init__(self) -> None:
        _identifier(self.table_id, "static table_id")
        try:
            sources = tuple(self.source_tensor_ids)
        except TypeError as exc:
            raise ProofV3Error(
                "static table source_tensor_ids must be an iterable"
            ) from exc
        if (
            not sources
            or len(sources) > MAX_STATIC_TABLE_SOURCE_TENSOR_IDS_V3
            or sources != tuple(sorted(set(sources)))
        ):
            raise ProofV3Error(
                "static table source_tensor_ids must be sorted and distinct"
            )
        for index, source in enumerate(sources):
            _identifier(source, f"static table source_tensor_ids[{index}]")
        start = _u64(self.logical_leaf_start, "static table logical_leaf_start")
        count = _u64(
            self.logical_leaf_count,
            "static table logical_leaf_count",
            positive=True,
        )
        if start + count >= 1 << 64:
            raise ProofV3Error("static table logical leaf range overflows")
        page_start = _u64(self.page_start, "static table page_start")
        page_count = _u64(self.page_count, "static table page_count", positive=True)
        if page_count > MAX_STATIC_TABLE_PAGES_PER_DESCRIPTOR_V3:
            raise ProofV3Error("static table page_count exceeds the protocol limit")
        if page_start + page_count >= 1 << 64:
            raise ProofV3Error("static table page range overflows")
        _identifier(self.element_encoding_id, "static table element_encoding_id")
        _identifier(self.scale_encoding_id, "static table scale_encoding_id")
        object.__setattr__(self, "source_tensor_ids", sources)

    def canonical_bytes(self) -> bytes:
        return (
            _encode_identifier(self.table_id, "static table_id")
            + struct.pack(
                "<QQQQH",
                self.logical_leaf_start,
                self.logical_leaf_count,
                self.page_start,
                self.page_count,
                len(self.source_tensor_ids),
            )
            + b"".join(
                _encode_identifier(source, "static table source_tensor_id")
                for source in self.source_tensor_ids
            )
            + _encode_identifier(
                self.element_encoding_id,
                "static table element_encoding_id",
            )
            + _encode_identifier(
                self.scale_encoding_id,
                "static table scale_encoding_id",
            )
        )

    def digest(self) -> bytes:
        return hashlib.sha256(
            STATIC_TABLE_DESCRIPTOR_DIGEST_DOMAIN_V3 + self.canonical_bytes()
        ).digest()

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "StaticTableDescriptorV3":
        reader = _Reader(encoded)
        table_id = reader.identifier("static table_id")
        (
            logical_leaf_start,
            logical_leaf_count,
            page_start,
            page_count,
            source_count,
        ) = reader.unpack("<QQQQH")
        if (
            type(source_count) is not int
            or not 0 < source_count <= MAX_STATIC_TABLE_SOURCE_TENSOR_IDS_V3
        ):
            raise ProofV3Error("static table source_tensor_ids count is out of range")
        result = cls(
            table_id=table_id,
            source_tensor_ids=tuple(
                reader.identifier("static table source_tensor_id")
                for _ in range(source_count)
            ),
            logical_leaf_start=logical_leaf_start,
            logical_leaf_count=logical_leaf_count,
            page_start=page_start,
            page_count=page_count,
            element_encoding_id=reader.identifier("static table element_encoding_id"),
            scale_encoding_id=reader.identifier("static table scale_encoding_id"),
        )
        reader.finish()
        if result.canonical_bytes() != encoded:
            raise ProofV3Error("static table descriptor is not canonical")
        return result


def _canonical_static_tables(
    value: object,
    *,
    name: str,
) -> tuple[StaticTableDescriptorV3, ...]:
    if isinstance(value, (bytes, bytearray, memoryview, str)):
        raise ProofV3Error(f"{name} must be a tuple of descriptors")
    try:
        tables = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ProofV3Error(f"{name} must be a tuple of descriptors") from exc
    if not tables or len(tables) > MAX_STATIC_TABLE_DESCRIPTORS_V3:
        raise ProofV3Error(f"{name} count is out of range")
    if not all(isinstance(item, StaticTableDescriptorV3) for item in tables):
        raise ProofV3Error(f"{name} have an unexpected type")
    table_ids = tuple(item.table_id for item in tables)
    if table_ids != tuple(sorted(table_ids)) or len(set(table_ids)) != len(table_ids):
        raise ProofV3Error(f"{name} must have sorted unique table ids")
    return tables


def canonical_static_table_layout_bytes_v3(
    tables: tuple[StaticTableDescriptorV3, ...],
) -> bytes:
    """Serialize one exact content-addressed static-table layout file.

    The offline qualifier emits this alongside the dynamic static root.  A
    validator parses it independently, hashes the exact bytes, and compares
    both the digest and parsed descriptors against the signed artifact.
    """

    canonical = _canonical_static_tables(tables, name="static table descriptors")
    encoded = struct.pack(
        "<4sHI",
        STATIC_TABLE_LAYOUT_MAGIC_V3,
        STATIC_TABLE_LAYOUT_FORMAT_VERSION_V3,
        len(canonical),
    ) + b"".join(
        struct.pack("<I", len(item.canonical_bytes())) + item.canonical_bytes()
        for item in canonical
    )
    if len(encoded) > MAX_STATIC_WEIGHT_ARTIFACT_BYTES_V3:
        raise ProofV3Error("static table layout exceeds the protocol limit")
    return encoded


def static_table_layout_from_canonical_bytes_v3(
    encoded: bytes,
) -> tuple[StaticTableDescriptorV3, ...]:
    """Strictly parse the content-addressed static-table layout file."""

    reader = _Reader(encoded)
    magic, version, count = reader.unpack("<4sHI")
    if magic != STATIC_TABLE_LAYOUT_MAGIC_V3:
        raise ProofV3Error("static table layout header is not supported")
    if version != STATIC_TABLE_LAYOUT_FORMAT_VERSION_V3:
        raise ProofV3Error("static table layout version is not supported")
    if type(count) is not int or not 0 < count <= MAX_STATIC_TABLE_DESCRIPTORS_V3:
        raise ProofV3Error("static table layout descriptor count is out of range")
    tables = tuple(
        StaticTableDescriptorV3.from_canonical_bytes(
            reader.bytes(
                "static table layout descriptor",
                MAX_STATIC_WEIGHT_ARTIFACT_BYTES_V3,
            )
        )
        for _ in range(count)
    )
    reader.finish()
    canonical = _canonical_static_tables(tables, name="static table layout descriptors")
    if canonical_static_table_layout_bytes_v3(canonical) != encoded:
        raise ProofV3Error("static table layout is not canonical")
    return canonical


@dataclass(frozen=True, slots=True)
class QuantizationBindingArtifactV3:
    """Canonical model-specific quantization binding for one V3 relation.

    The relation deliberately binds the **raw** SHA-256 of these exact bytes,
    not a domain-separated object digest.  That preserves the existing loaded
    artifact ABI while making the previously opaque bytes parseable and
    cross-checkable.  The artifact names the approved semantics and exact V2
    ``ModelSpec.quant_mode``; it does not attempt to infer a universal mapping
    between them.  That pairing is model-qualified and authority-signed.

    This declaration is a binding statement, not a proof that a runtime
    performed the stated quantization.  The dynamic relation and static-table
    construction still enforce the actual encodings and scales.
    """

    static_manifest_digest: bytes
    model_spec_identity_digest: bytes
    quantization_semantics_id: str
    model_quant_mode: str
    quantization_binding_abi_id: str = QUANTIZATION_BINDING_ABI_ID_V3
    binding_version: int = QUANTIZATION_BINDING_FORMAT_VERSION_V3

    def __post_init__(self) -> None:
        if (
            _u32(
                self.binding_version,
                "quantization binding version",
                positive=True,
            )
            != QUANTIZATION_BINDING_FORMAT_VERSION_V3
        ):
            raise ProofV3Error("quantization binding version is not supported")
        if (
            _identifier(
                self.quantization_binding_abi_id,
                "quantization binding ABI",
            )
            != QUANTIZATION_BINDING_ABI_ID_V3
        ):
            raise ProofV3Error("quantization binding ABI is not supported")
        _fixed32(
            self.static_manifest_digest,
            "quantization binding static_manifest_digest",
            nonzero=True,
        )
        _fixed32(
            self.model_spec_identity_digest,
            "quantization binding model_spec_identity_digest",
            nonzero=True,
        )
        _identifier(
            self.quantization_semantics_id,
            "quantization binding quantization_semantics_id",
        )
        _model_quant_mode(
            self.model_quant_mode,
            "quantization binding model_quant_mode",
        )

    def canonical_bytes(self) -> bytes:
        return (
            struct.pack(
                "<4sH",
                QUANTIZATION_BINDING_MAGIC_V3,
                self.binding_version,
            )
            + self.static_manifest_digest
            + self.model_spec_identity_digest
            + _encode_identifier(
                self.quantization_binding_abi_id,
                "quantization_binding_abi_id",
            )
            + _encode_identifier(
                self.quantization_semantics_id,
                "quantization_semantics_id",
            )
            + _encode_model_quant_mode(self.model_quant_mode)
        )

    def raw_sha256(self) -> bytes:
        """Return the raw SHA-256 which the relation binds exactly."""

        return hashlib.sha256(self.canonical_bytes()).digest()

    def validate_authenticated_context(
        self,
        *,
        relation_quantization_binding_digest: bytes,
        profile_quantization_semantics_id: str,
        static_manifest_digest: bytes,
        model_spec_identity_digest: bytes,
        model_quant_mode: str,
    ) -> None:
        """Cross-check this binding against profile and verified V2 metadata."""

        try:
            expected_relation_digest = _fixed32(
                relation_quantization_binding_digest,
                "relation quantization_binding_digest",
                nonzero=True,
            )
            expected_semantics = _identifier(
                profile_quantization_semantics_id,
                "profile quantization_semantics_id",
            )
            expected_manifest_digest = _fixed32(
                static_manifest_digest,
                "authenticated static_manifest_digest",
                nonzero=True,
            )
            expected_model_spec_digest = _fixed32(
                model_spec_identity_digest,
                "authenticated model_spec_identity_digest",
                nonzero=True,
            )
            expected_quant_mode = _model_quant_mode(
                model_quant_mode,
                "authenticated model_quant_mode",
            )
        except ProofV3Error as exc:
            raise ProofV3VerificationError(
                "quantization binding authenticated context is malformed"
            ) from exc

        comparisons = (
            (
                self.raw_sha256(),
                expected_relation_digest,
                "relation quantization binding digest",
            ),
            (
                self.quantization_semantics_id,
                expected_semantics,
                "profile quantization semantics",
            ),
            (
                self.static_manifest_digest,
                expected_manifest_digest,
                "authenticated static manifest",
            ),
            (
                self.model_spec_identity_digest,
                expected_model_spec_digest,
                "authenticated model-spec identity",
            ),
            (
                self.model_quant_mode,
                expected_quant_mode,
                "authenticated model quant mode",
            ),
        )
        for actual, expected, name in comparisons:
            if actual != expected:
                raise ProofV3VerificationError(
                    f"quantization binding does not match the expected {name}"
                )

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "QuantizationBindingArtifactV3":
        """Strictly parse a relation-bound quantization binding artifact."""

        if (
            type(encoded) is not bytes
            or not encoded
            or len(encoded) > MAX_QUANTIZATION_BINDING_BYTES_V3
        ):
            raise ProofV3Error("quantization binding byte length is out of range")
        reader = _Reader(encoded)
        magic, version = reader.unpack("<4sH")
        if magic != QUANTIZATION_BINDING_MAGIC_V3:
            raise ProofV3Error("quantization binding header is not supported")
        if version != QUANTIZATION_BINDING_FORMAT_VERSION_V3:
            raise ProofV3Error("quantization binding version is not supported")
        result = cls(
            static_manifest_digest=reader.read(32),
            model_spec_identity_digest=reader.read(32),
            quantization_binding_abi_id=reader.identifier(
                "quantization_binding_abi_id"
            ),
            quantization_semantics_id=reader.identifier("quantization_semantics_id"),
            model_quant_mode=reader.model_quant_mode(),
            binding_version=version,
        )
        reader.finish()
        if result.canonical_bytes() != encoded:
            raise ProofV3Error("quantization binding is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class StaticTableSourceBindingV3:
    """Canonical qualified declaration for one dynamic static-table root.

    A validator intentionally does not load model weights.  It *does* load the
    compact, content-addressed recipe/index/construction artifacts used by the
    offline qualifier.  This declaration binds their digests and every field
    that determines how those source bytes become the static Goldilocks table
    root.  The enclosing static-weight artifact binds this declaration's
    domain-separated digest.
    """

    canonical_model_bytes_digest: bytes
    source_file_index_digest: bytes
    canonicalization_recipe_digest: bytes
    static_table_root: bytes
    static_table_construction_digest: bytes
    static_table_layout_digest: bytes
    quantization_binding_digest: bytes
    logical_leaf_count: int
    static_page_count: int
    tree_leaf_count: int
    page_leaf_capacity: int
    leaf_byte_width: int
    canonical_model_bytes_abi_id: str = CANONICAL_MODEL_BYTES_ABI_ID_V3
    static_table_construction_abi_id: str = (
        GOLDILOCKS_STATIC_TABLE_CONSTRUCTION_ABI_ID_V3
    )
    field_id: str = GOLDILOCKS_STATIC_FIELD_ID_V3
    table_hash_abi_id: str = GOLDILOCKS_STATIC_TABLE_HASH_ABI_ID_V3
    table_encoding_abi_id: str = GOLDILOCKS_STATIC_TABLE_ENCODING_ABI_ID_V3
    table_layout_abi_id: str = GOLDILOCKS_STATIC_TABLE_LAYOUT_ABI_ID_V3

    def __post_init__(self) -> None:
        if self.canonical_model_bytes_abi_id != CANONICAL_MODEL_BYTES_ABI_ID_V3:
            raise ProofV3Error("canonical model-bytes ABI is not supported")
        if (
            self.static_table_construction_abi_id
            != GOLDILOCKS_STATIC_TABLE_CONSTRUCTION_ABI_ID_V3
        ):
            raise ProofV3Error("static-table construction ABI is not supported")
        if self.field_id != GOLDILOCKS_STATIC_FIELD_ID_V3:
            raise ProofV3Error("static-table source field is not supported")
        if self.table_hash_abi_id != GOLDILOCKS_STATIC_TABLE_HASH_ABI_ID_V3:
            raise ProofV3Error("static-table source hash ABI is not supported")
        if self.table_encoding_abi_id != GOLDILOCKS_STATIC_TABLE_ENCODING_ABI_ID_V3:
            raise ProofV3Error("static-table source encoding ABI is not supported")
        if self.table_layout_abi_id != GOLDILOCKS_STATIC_TABLE_LAYOUT_ABI_ID_V3:
            raise ProofV3Error("static-table source layout ABI is not supported")
        for value, name in (
            (self.canonical_model_bytes_digest, "canonical_model_bytes_digest"),
            (self.source_file_index_digest, "source_file_index_digest"),
            (self.canonicalization_recipe_digest, "canonicalization_recipe_digest"),
            (self.static_table_root, "static_table_root"),
            (
                self.static_table_construction_digest,
                "static_table_construction_digest",
            ),
            (self.static_table_layout_digest, "static_table_layout_digest"),
            (self.quantization_binding_digest, "quantization_binding_digest"),
        ):
            _fixed32(value, name, nonzero=True)
        logical = _u64(self.logical_leaf_count, "logical_leaf_count", positive=True)
        pages = _u64(self.static_page_count, "static_page_count", positive=True)
        if pages > MAX_STATIC_TABLE_PAGE_COUNT_V3:
            raise ProofV3Error("static_page_count exceeds the protocol limit")
        tree = _power_of_two(
            _u64(self.tree_leaf_count, "tree_leaf_count", positive=True),
            "tree_leaf_count",
            MAX_STATIC_TABLE_PAGE_COUNT_V3,
        )
        if pages > tree:
            raise ProofV3Error("static_page_count exceeds tree_leaf_count")
        capacity = _power_of_two(
            _u32(self.page_leaf_capacity, "page_leaf_capacity", positive=True),
            "page_leaf_capacity",
            MAX_STATIC_TABLE_LEAVES_V3,
        )
        if logical > pages * capacity:
            raise ProofV3Error("static pages cannot hold the logical leaf count")
        width = _u32(self.leaf_byte_width, "leaf_byte_width", positive=True)
        if width > MAX_STATIC_TABLE_LEAF_BYTES_V3:
            raise ProofV3Error("leaf_byte_width exceeds the protocol limit")

    def canonical_bytes(self) -> bytes:
        return (
            struct.pack(
                "<4sHQQQII",
                STATIC_TABLE_SOURCE_BINDING_MAGIC_V3,
                STATIC_TABLE_SOURCE_BINDING_FORMAT_VERSION_V3,
                self.logical_leaf_count,
                self.static_page_count,
                self.tree_leaf_count,
                self.page_leaf_capacity,
                self.leaf_byte_width,
            )
            + self.canonical_model_bytes_digest
            + self.source_file_index_digest
            + self.canonicalization_recipe_digest
            + self.static_table_root
            + self.static_table_construction_digest
            + self.static_table_layout_digest
            + self.quantization_binding_digest
            + _encode_identifier(
                self.canonical_model_bytes_abi_id,
                "canonical_model_bytes_abi_id",
            )
            + _encode_identifier(
                self.static_table_construction_abi_id,
                "static_table_construction_abi_id",
            )
            + _encode_identifier(self.field_id, "field_id")
            + _encode_identifier(self.table_hash_abi_id, "table_hash_abi_id")
            + _encode_identifier(
                self.table_encoding_abi_id,
                "table_encoding_abi_id",
            )
            + _encode_identifier(self.table_layout_abi_id, "table_layout_abi_id")
        )

    def digest(self) -> bytes:
        return hashlib.sha256(
            STATIC_TABLE_SOURCE_BINDING_DIGEST_DOMAIN_V3 + self.canonical_bytes()
        ).digest()

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "StaticTableSourceBindingV3":
        reader = _Reader(encoded)
        (
            magic,
            version,
            logical_leaf_count,
            static_page_count,
            tree_leaf_count,
            page_leaf_capacity,
            leaf_byte_width,
        ) = reader.unpack("<4sHQQQII")
        if magic != STATIC_TABLE_SOURCE_BINDING_MAGIC_V3:
            raise ProofV3Error("static-table source binding header is not supported")
        if version != STATIC_TABLE_SOURCE_BINDING_FORMAT_VERSION_V3:
            raise ProofV3Error("static-table source binding version is not supported")
        fixed = tuple(reader.read(32) for _ in range(7))
        result = cls(
            canonical_model_bytes_digest=fixed[0],
            source_file_index_digest=fixed[1],
            canonicalization_recipe_digest=fixed[2],
            static_table_root=fixed[3],
            static_table_construction_digest=fixed[4],
            static_table_layout_digest=fixed[5],
            quantization_binding_digest=fixed[6],
            logical_leaf_count=logical_leaf_count,
            static_page_count=static_page_count,
            tree_leaf_count=tree_leaf_count,
            page_leaf_capacity=page_leaf_capacity,
            leaf_byte_width=leaf_byte_width,
            canonical_model_bytes_abi_id=reader.identifier(
                "canonical_model_bytes_abi_id"
            ),
            static_table_construction_abi_id=reader.identifier(
                "static_table_construction_abi_id"
            ),
            field_id=reader.identifier("field_id"),
            table_hash_abi_id=reader.identifier("table_hash_abi_id"),
            table_encoding_abi_id=reader.identifier("table_encoding_abi_id"),
            table_layout_abi_id=reader.identifier("table_layout_abi_id"),
        )
        reader.finish()
        if result.canonical_bytes() != encoded:
            raise ProofV3Error("static-table source binding is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class StaticWeightArtifactContextV3:
    """Validator-owned v2 context used to validate a V3 static artifact.

    Construction happens only after the v2 manifest, ModelSpec, and Pallas
    catalog have been authenticated through their existing path.  It contains
    hashes/metadata only; no model tensor is retained by the validator.
    """

    chain_id: int
    netuid: int
    registry_address: bytes
    model_id: str
    static_manifest_digest: bytes
    model_spec_identity_digest: bytes
    model_weight_merkle_root: bytes
    source_weight_file_hash: bytes
    canonical_model_bytes_abi_id: str
    canonical_model_bytes_digest: bytes
    source_file_index_digest: bytes
    canonicalization_recipe_digest: bytes
    pallas_catalog_abi_id: str
    pallas_catalog_sha256: bytes
    pallas_catalog_size: int
    pallas_pcs_suite: str
    pallas_generator_version: str
    quantization_binding_digest: bytes
    kernel_abi_digest: bytes
    dynamic_backend_abi_id: str
    dynamic_backend_digest: bytes
    static_table_root: bytes
    static_table_construction_abi_id: str
    static_table_construction_digest: bytes
    static_table_layout_digest: bytes
    static_tables: tuple[StaticTableDescriptorV3, ...]

    def __post_init__(self) -> None:
        _u32(self.chain_id, "chain_id", positive=True)
        _u32(self.netuid, "netuid")
        _address(self.registry_address, "registry_address")
        _model_id(self.model_id)
        for value, name in (
            (self.static_manifest_digest, "static_manifest_digest"),
            (self.model_spec_identity_digest, "model_spec_identity_digest"),
            (self.model_weight_merkle_root, "model_weight_merkle_root"),
            (self.source_weight_file_hash, "source_weight_file_hash"),
            (self.canonical_model_bytes_digest, "canonical_model_bytes_digest"),
            (self.source_file_index_digest, "source_file_index_digest"),
            (self.canonicalization_recipe_digest, "canonicalization_recipe_digest"),
            (self.pallas_catalog_sha256, "pallas_catalog_sha256"),
            (self.quantization_binding_digest, "quantization_binding_digest"),
            (self.kernel_abi_digest, "kernel_abi_digest"),
            (self.dynamic_backend_digest, "dynamic_backend_digest"),
            (self.static_table_root, "static_table_root"),
            (
                self.static_table_construction_digest,
                "static_table_construction_digest",
            ),
            (self.static_table_layout_digest, "static_table_layout_digest"),
        ):
            _fixed32(value, name, nonzero=True)
        _identifier(self.pallas_catalog_abi_id, "pallas_catalog_abi_id")
        if self.canonical_model_bytes_abi_id != CANONICAL_MODEL_BYTES_ABI_ID_V3:
            raise ProofV3Error("canonical model-bytes ABI is not supported")
        _u64(self.pallas_catalog_size, "pallas_catalog_size", positive=True)
        _identifier(self.pallas_pcs_suite, "pallas_pcs_suite")
        _identifier(self.pallas_generator_version, "pallas_generator_version")
        if self.pallas_pcs_suite != V2_PALLAS_PCS_SUITE_V3:
            raise ProofV3Error("Pallas PCS suite is not supported")
        if self.pallas_generator_version != V2_PALLAS_GENERATOR_VERSION_V3:
            raise ProofV3Error("Pallas generator version is not supported")
        if self.dynamic_backend_abi_id != GOLDILOCKS_DYNAMIC_BACKEND_ABI_ID_V3:
            raise ProofV3Error("dynamic backend ABI is not supported")
        if (
            self.static_table_construction_abi_id
            != GOLDILOCKS_STATIC_TABLE_CONSTRUCTION_ABI_ID_V3
        ):
            raise ProofV3Error("static-table construction ABI is not supported")
        tables = _canonical_static_tables(
            self.static_tables,
            name="static table descriptors",
        )
        object.__setattr__(self, "static_tables", tables)

    @classmethod
    def from_verified_v2_catalog_binding(
        cls,
        *,
        verified_v2_catalog_binding: object,
        canonical_model_bytes_digest: bytes,
        source_file_index_digest: bytes,
        canonicalization_recipe_digest: bytes,
        quantization_binding_digest: bytes,
        kernel_abi_digest: bytes,
        dynamic_backend_digest: bytes,
        static_table_root: bytes,
        static_table_construction_digest: bytes,
        static_table_layout_digest: bytes,
        static_tables: tuple[StaticTableDescriptorV3, ...],
        canonical_model_bytes_abi_id: str = CANONICAL_MODEL_BYTES_ABI_ID_V3,
        dynamic_backend_abi_id: str = GOLDILOCKS_DYNAMIC_BACKEND_ABI_ID_V3,
        static_table_construction_abi_id: str = (
            GOLDILOCKS_STATIC_TABLE_CONSTRUCTION_ABI_ID_V3
        ),
    ) -> "StaticWeightArtifactContextV3":
        """Build a validator context from the trusted V2 bridge.

        In particular, callers cannot supply a second manifest digest, model
        identity, Pallas catalog digest/size, or PCS parameters alongside the
        dynamic artifacts.  Those values are derived exactly once from
        ``VerifiedV2CatalogBindingV3``.  This is the production construction
        path; direct construction remains only for canonical parsing and
        deliberately explicit unit fixtures.
        """

        manifest, _ = _trusted_v2_catalog_binding_fields_v3(verified_v2_catalog_binding)
        return cls(
            chain_id=manifest.chain_id,
            netuid=manifest.netuid,
            registry_address=manifest.registry_address,
            model_id=manifest.model_id,
            static_manifest_digest=manifest.manifest_digest,
            model_spec_identity_digest=manifest.model_spec_identity_digest,
            model_weight_merkle_root=manifest.model_weight_merkle_root,
            source_weight_file_hash=manifest.source_weight_file_hash,
            canonical_model_bytes_abi_id=canonical_model_bytes_abi_id,
            canonical_model_bytes_digest=canonical_model_bytes_digest,
            source_file_index_digest=source_file_index_digest,
            canonicalization_recipe_digest=canonicalization_recipe_digest,
            pallas_catalog_abi_id=manifest.pallas_catalog_abi_id,
            pallas_catalog_sha256=manifest.pallas_catalog_sha256,
            pallas_catalog_size=manifest.pallas_catalog_size,
            pallas_pcs_suite=manifest.pallas_pcs_suite,
            pallas_generator_version=manifest.pallas_generator_version,
            quantization_binding_digest=quantization_binding_digest,
            kernel_abi_digest=kernel_abi_digest,
            dynamic_backend_abi_id=dynamic_backend_abi_id,
            dynamic_backend_digest=dynamic_backend_digest,
            static_table_root=static_table_root,
            static_table_construction_abi_id=static_table_construction_abi_id,
            static_table_construction_digest=static_table_construction_digest,
            static_table_layout_digest=static_table_layout_digest,
            static_tables=static_tables,
        )


@dataclass(frozen=True, slots=True)
class StaticWeightArtifactV3:
    """One authority-profile-bound static table artifact for one exact model.

    The descriptor catalog covers every non-padding logical fixed-table leaf
    exactly once and maps it to a non-overlapping static-table page range.
    The tree itself is padded to ``tree_leaf_count`` canonical pages, keeping
    Merkle/FRI domain size independent from the logical number of source data
    leaves.
    """

    chain_id: int
    netuid: int
    registry_address: bytes
    model_id: str
    static_manifest_digest: bytes
    model_spec_identity_digest: bytes
    model_weight_merkle_root: bytes
    source_weight_file_hash: bytes
    canonical_model_bytes_digest: bytes
    source_file_index_digest: bytes
    canonicalization_recipe_digest: bytes
    pallas_catalog_abi_id: str
    pallas_catalog_sha256: bytes
    pallas_catalog_size: int
    pallas_pcs_suite: str
    pallas_generator_version: str
    dynamic_backend_abi_id: str
    dynamic_backend_digest: bytes
    static_table_root: bytes
    static_source_binding_digest: bytes
    static_table_construction_digest: bytes
    static_table_layout_digest: bytes
    quantization_binding_digest: bytes
    kernel_abi_digest: bytes
    logical_leaf_count: int
    static_page_count: int
    tree_leaf_count: int
    page_leaf_capacity: int
    leaf_byte_width: int
    static_tables: tuple[StaticTableDescriptorV3, ...]
    canonical_model_bytes_abi_id: str = CANONICAL_MODEL_BYTES_ABI_ID_V3
    static_table_construction_abi_id: str = (
        GOLDILOCKS_STATIC_TABLE_CONSTRUCTION_ABI_ID_V3
    )
    field_id: str = GOLDILOCKS_STATIC_FIELD_ID_V3
    table_hash_abi_id: str = GOLDILOCKS_STATIC_TABLE_HASH_ABI_ID_V3
    table_encoding_abi_id: str = GOLDILOCKS_STATIC_TABLE_ENCODING_ABI_ID_V3
    table_layout_abi_id: str = GOLDILOCKS_STATIC_TABLE_LAYOUT_ABI_ID_V3
    artifact_version: int = STATIC_WEIGHT_ARTIFACT_FORMAT_VERSION_V3

    def __post_init__(self) -> None:
        _u32(self.chain_id, "chain_id", positive=True)
        _u32(self.netuid, "netuid")
        _address(self.registry_address, "registry_address")
        _model_id(self.model_id)
        for value, name in (
            (self.static_manifest_digest, "static_manifest_digest"),
            (self.model_spec_identity_digest, "model_spec_identity_digest"),
            (self.model_weight_merkle_root, "model_weight_merkle_root"),
            (self.source_weight_file_hash, "source_weight_file_hash"),
            (self.canonical_model_bytes_digest, "canonical_model_bytes_digest"),
            (self.source_file_index_digest, "source_file_index_digest"),
            (self.canonicalization_recipe_digest, "canonicalization_recipe_digest"),
            (self.pallas_catalog_sha256, "pallas_catalog_sha256"),
            (self.dynamic_backend_digest, "dynamic_backend_digest"),
            (self.static_table_root, "static_table_root"),
            (self.static_source_binding_digest, "static_source_binding_digest"),
            (
                self.static_table_construction_digest,
                "static_table_construction_digest",
            ),
            (self.static_table_layout_digest, "static_table_layout_digest"),
            (self.quantization_binding_digest, "quantization_binding_digest"),
            (self.kernel_abi_digest, "kernel_abi_digest"),
        ):
            _fixed32(value, name, nonzero=True)
        if self.canonical_model_bytes_abi_id != CANONICAL_MODEL_BYTES_ABI_ID_V3:
            raise ProofV3Error("canonical model-bytes ABI is not supported")
        if (
            self.static_table_construction_abi_id
            != GOLDILOCKS_STATIC_TABLE_CONSTRUCTION_ABI_ID_V3
        ):
            raise ProofV3Error("static-table construction ABI is not supported")
        if self.pallas_catalog_abi_id != PALLAS_COLUMN_CATALOG_ABI_ID_V3:
            raise ProofV3Error("Pallas catalog ABI is not supported")
        _u64(self.pallas_catalog_size, "pallas_catalog_size", positive=True)
        _identifier(self.pallas_pcs_suite, "pallas_pcs_suite")
        _identifier(self.pallas_generator_version, "pallas_generator_version")
        if self.pallas_pcs_suite != V2_PALLAS_PCS_SUITE_V3:
            raise ProofV3Error("Pallas PCS suite is not supported")
        if self.pallas_generator_version != V2_PALLAS_GENERATOR_VERSION_V3:
            raise ProofV3Error("Pallas generator version is not supported")
        if self.dynamic_backend_abi_id != GOLDILOCKS_DYNAMIC_BACKEND_ABI_ID_V3:
            raise ProofV3Error("dynamic backend ABI is not supported")
        if self.field_id != GOLDILOCKS_STATIC_FIELD_ID_V3:
            raise ProofV3Error("static-weight field is not supported")
        if self.table_hash_abi_id != GOLDILOCKS_STATIC_TABLE_HASH_ABI_ID_V3:
            raise ProofV3Error("static-table hash ABI is not supported")
        if self.table_encoding_abi_id != GOLDILOCKS_STATIC_TABLE_ENCODING_ABI_ID_V3:
            raise ProofV3Error("static-table encoding ABI is not supported")
        if self.table_layout_abi_id != GOLDILOCKS_STATIC_TABLE_LAYOUT_ABI_ID_V3:
            raise ProofV3Error("static-table layout ABI is not supported")
        logical = _u64(self.logical_leaf_count, "logical_leaf_count", positive=True)
        pages = _u64(self.static_page_count, "static_page_count", positive=True)
        if pages > MAX_STATIC_TABLE_PAGE_COUNT_V3:
            raise ProofV3Error("static_page_count exceeds the protocol limit")
        tree = _power_of_two(
            _u64(self.tree_leaf_count, "tree_leaf_count", positive=True),
            "tree_leaf_count",
            MAX_STATIC_TABLE_PAGE_COUNT_V3,
        )
        if pages > tree:
            raise ProofV3Error("static_page_count exceeds tree_leaf_count")
        capacity = _power_of_two(
            _u32(self.page_leaf_capacity, "page_leaf_capacity", positive=True),
            "page_leaf_capacity",
            MAX_STATIC_TABLE_LEAVES_V3,
        )
        if logical > pages * capacity:
            raise ProofV3Error("static pages cannot hold the logical leaf count")
        width = _u32(self.leaf_byte_width, "leaf_byte_width", positive=True)
        if width > MAX_STATIC_TABLE_LEAF_BYTES_V3:
            raise ProofV3Error("leaf_byte_width exceeds the protocol limit")
        tables = _canonical_static_tables(
            self.static_tables,
            name="static table descriptors",
        )
        expected_leaf_start = 0
        used_pages: list[tuple[int, int]] = []
        for descriptor in tables:
            if descriptor.logical_leaf_start != expected_leaf_start:
                raise ProofV3Error(
                    "static table descriptors must cover logical leaves exactly once"
                )
            expected_leaf_start += descriptor.logical_leaf_count
            if descriptor.page_start + descriptor.page_count > pages:
                raise ProofV3Error(
                    "static table descriptor page range is out of bounds"
                )
            if descriptor.logical_leaf_count > descriptor.page_count * capacity:
                raise ProofV3Error(
                    "static table descriptor pages cannot hold its logical leaves"
                )
            used_pages.append(
                (descriptor.page_start, descriptor.page_start + descriptor.page_count)
            )
        if expected_leaf_start != logical:
            raise ProofV3Error(
                "static table descriptors do not cover the logical leaf domain"
            )
        ordered_pages = sorted(used_pages)
        if ordered_pages[0][0] != 0:
            raise ProofV3Error(
                "static table descriptor page ranges must cover static pages exactly once"
            )
        for (_, previous_end), (current_start, _) in zip(
            ordered_pages,
            ordered_pages[1:],
        ):
            if current_start != previous_end:
                if current_start < previous_end:
                    raise ProofV3Error("static table descriptor page ranges overlap")
                raise ProofV3Error(
                    "static table descriptor page ranges must cover static pages exactly once"
                )
        if ordered_pages[-1][1] != pages:
            raise ProofV3Error(
                "static table descriptor page ranges must cover static pages exactly once"
            )
        if self.artifact_version != STATIC_WEIGHT_ARTIFACT_FORMAT_VERSION_V3:
            raise ProofV3Error("static-weight artifact version is not supported")
        object.__setattr__(self, "static_tables", tables)
        if len(self.canonical_bytes()) > MAX_STATIC_WEIGHT_ARTIFACT_BYTES_V3:
            raise ProofV3Error("static-weight artifact exceeds the protocol size limit")

    @classmethod
    def from_verified_v2_catalog_binding(
        cls,
        *,
        verified_v2_catalog_binding: object,
        canonical_model_bytes_digest: bytes,
        source_file_index_digest: bytes,
        canonicalization_recipe_digest: bytes,
        dynamic_backend_digest: bytes,
        static_table_root: bytes,
        static_source_binding_digest: bytes,
        static_table_construction_digest: bytes,
        static_table_layout_digest: bytes,
        quantization_binding_digest: bytes,
        kernel_abi_digest: bytes,
        logical_leaf_count: int,
        static_page_count: int,
        tree_leaf_count: int,
        page_leaf_capacity: int,
        leaf_byte_width: int,
        static_tables: tuple[StaticTableDescriptorV3, ...],
        canonical_model_bytes_abi_id: str = CANONICAL_MODEL_BYTES_ABI_ID_V3,
        static_table_construction_abi_id: str = (
            GOLDILOCKS_STATIC_TABLE_CONSTRUCTION_ABI_ID_V3
        ),
        dynamic_backend_abi_id: str = GOLDILOCKS_DYNAMIC_BACKEND_ABI_ID_V3,
        field_id: str = GOLDILOCKS_STATIC_FIELD_ID_V3,
        table_hash_abi_id: str = GOLDILOCKS_STATIC_TABLE_HASH_ABI_ID_V3,
        table_encoding_abi_id: str = GOLDILOCKS_STATIC_TABLE_ENCODING_ABI_ID_V3,
        table_layout_abi_id: str = GOLDILOCKS_STATIC_TABLE_LAYOUT_ABI_ID_V3,
    ) -> "StaticWeightArtifactV3":
        """Construct a V3 static artifact from the trusted V2 catalog bridge.

        This is the only intended construction path for a newly qualified,
        authority-signed artifact.  Every V2 identity and PCS/catalog field is
        copied from ``VerifiedV2CatalogBindingV3`` after its exact canonical
        catalog bytes are checked.  Dynamic-table artifact claims remain
        explicit inputs because they are qualified by the V3 backend, not by
        the V2 manifest.

        ``StaticWeightArtifactV3(...)`` remains deliberately available for
        strict deserialization and small, explicit test fixtures.  Production
        catalog registration must use this factory (or validate an imported
        artifact with :meth:`validate_verified_v2_catalog_binding`) instead of
        accepting separately supplied V2 metadata.
        """

        manifest, _ = _trusted_v2_catalog_binding_fields_v3(verified_v2_catalog_binding)
        artifact = cls(
            chain_id=manifest.chain_id,
            netuid=manifest.netuid,
            registry_address=manifest.registry_address,
            model_id=manifest.model_id,
            static_manifest_digest=manifest.manifest_digest,
            model_spec_identity_digest=manifest.model_spec_identity_digest,
            model_weight_merkle_root=manifest.model_weight_merkle_root,
            source_weight_file_hash=manifest.source_weight_file_hash,
            canonical_model_bytes_digest=canonical_model_bytes_digest,
            source_file_index_digest=source_file_index_digest,
            canonicalization_recipe_digest=canonicalization_recipe_digest,
            pallas_catalog_abi_id=manifest.pallas_catalog_abi_id,
            pallas_catalog_sha256=manifest.pallas_catalog_sha256,
            pallas_catalog_size=manifest.pallas_catalog_size,
            pallas_pcs_suite=manifest.pallas_pcs_suite,
            pallas_generator_version=manifest.pallas_generator_version,
            dynamic_backend_abi_id=dynamic_backend_abi_id,
            dynamic_backend_digest=dynamic_backend_digest,
            static_table_root=static_table_root,
            static_source_binding_digest=static_source_binding_digest,
            static_table_construction_digest=static_table_construction_digest,
            static_table_layout_digest=static_table_layout_digest,
            quantization_binding_digest=quantization_binding_digest,
            kernel_abi_digest=kernel_abi_digest,
            logical_leaf_count=logical_leaf_count,
            static_page_count=static_page_count,
            tree_leaf_count=tree_leaf_count,
            page_leaf_capacity=page_leaf_capacity,
            leaf_byte_width=leaf_byte_width,
            static_tables=static_tables,
            canonical_model_bytes_abi_id=canonical_model_bytes_abi_id,
            static_table_construction_abi_id=static_table_construction_abi_id,
            field_id=field_id,
            table_hash_abi_id=table_hash_abi_id,
            table_encoding_abi_id=table_encoding_abi_id,
            table_layout_abi_id=table_layout_abi_id,
        )
        artifact.validate_verified_v2_catalog_binding(
            verified_v2_catalog_binding=verified_v2_catalog_binding
        )
        return artifact

    def validate_verified_v2_catalog_binding(
        self,
        *,
        verified_v2_catalog_binding: object,
    ) -> None:
        """Fail closed unless this artifact's V2 leg is the trusted bridge.

        Validators call this when loading an authority-signed artifact from
        bytes.  It intentionally validates only the V2/static PCS leg; the
        caller separately validates the dynamic source-binding/layout/kernel
        artifacts before accepting a native adapter.
        """

        manifest, _ = _trusted_v2_catalog_binding_fields_v3(verified_v2_catalog_binding)
        comparisons = (
            (manifest.chain_id, self.chain_id, "chain_id"),
            (manifest.netuid, self.netuid, "netuid"),
            (manifest.registry_address, self.registry_address, "registry address"),
            (manifest.model_id, self.model_id, "model_id"),
            (
                manifest.manifest_digest,
                self.static_manifest_digest,
                "static manifest",
            ),
            (
                manifest.model_spec_identity_digest,
                self.model_spec_identity_digest,
                "model-spec identity",
            ),
            (
                manifest.model_weight_merkle_root,
                self.model_weight_merkle_root,
                "model weight Merkle root",
            ),
            (
                manifest.source_weight_file_hash,
                self.source_weight_file_hash,
                "source weight file hash",
            ),
            (
                manifest.pallas_catalog_abi_id,
                self.pallas_catalog_abi_id,
                "Pallas catalog ABI",
            ),
            (
                manifest.pallas_catalog_sha256,
                self.pallas_catalog_sha256,
                "Pallas catalog",
            ),
            (
                manifest.pallas_catalog_size,
                self.pallas_catalog_size,
                "Pallas catalog size",
            ),
            (
                manifest.pallas_pcs_suite,
                self.pallas_pcs_suite,
                "Pallas PCS suite",
            ),
            (
                manifest.pallas_generator_version,
                self.pallas_generator_version,
                "Pallas generator version",
            ),
        )
        for expected, actual, name in comparisons:
            if expected != actual:
                raise ProofV3VerificationError(
                    "static-weight artifact does not match the verified proof-v2 "
                    f"{name}"
                )

    def canonical_bytes(self) -> bytes:
        descriptor_bytes = tuple(item.canonical_bytes() for item in self.static_tables)
        return (
            struct.pack(
                "<4sHII20sQQQQIIH",
                STATIC_WEIGHT_ARTIFACT_MAGIC_V3,
                self.artifact_version,
                self.chain_id,
                self.netuid,
                self.registry_address,
                self.pallas_catalog_size,
                self.logical_leaf_count,
                self.static_page_count,
                self.tree_leaf_count,
                self.page_leaf_capacity,
                self.leaf_byte_width,
                len(descriptor_bytes),
            )
            + self.static_manifest_digest
            + self.model_spec_identity_digest
            + self.model_weight_merkle_root
            + self.source_weight_file_hash
            + self.canonical_model_bytes_digest
            + self.source_file_index_digest
            + self.canonicalization_recipe_digest
            + self.pallas_catalog_sha256
            + self.dynamic_backend_digest
            + self.static_table_root
            + self.static_source_binding_digest
            + self.static_table_construction_digest
            + self.static_table_layout_digest
            + self.quantization_binding_digest
            + self.kernel_abi_digest
            + _encode_model_id(self.model_id)
            + _encode_identifier(
                self.canonical_model_bytes_abi_id,
                "canonical_model_bytes_abi_id",
            )
            + _encode_identifier(
                self.static_table_construction_abi_id,
                "static_table_construction_abi_id",
            )
            + _encode_identifier(self.pallas_catalog_abi_id, "pallas_catalog_abi_id")
            + _encode_identifier(self.pallas_pcs_suite, "pallas_pcs_suite")
            + _encode_identifier(
                self.pallas_generator_version,
                "pallas_generator_version",
            )
            + _encode_identifier(self.dynamic_backend_abi_id, "dynamic_backend_abi_id")
            + _encode_identifier(self.field_id, "field_id")
            + _encode_identifier(self.table_hash_abi_id, "table_hash_abi_id")
            + _encode_identifier(self.table_encoding_abi_id, "table_encoding_abi_id")
            + _encode_identifier(self.table_layout_abi_id, "table_layout_abi_id")
            + b"".join(
                struct.pack("<I", len(encoded)) + encoded
                for encoded in descriptor_bytes
            )
        )

    def digest(self) -> bytes:
        """Return the exact digest an authority-signed V3 profile binds."""

        return hashlib.sha256(
            STATIC_WEIGHT_ARTIFACT_DIGEST_DOMAIN_V3 + self.canonical_bytes()
        ).digest()

    def validate_context(self, context: StaticWeightArtifactContextV3) -> None:
        """Check the independently authenticated v2/static-loader context.

        The qualified loader separately parses a content-addressed static-table
        source binding before constructing ``context``.  No cross-field cast
        is used or implied.
        """

        if not isinstance(context, StaticWeightArtifactContextV3):
            raise ProofV3VerificationError(
                "static-weight artifact context has an unexpected type"
            )
        comparisons = (
            (context.chain_id, self.chain_id, "chain_id"),
            (context.netuid, self.netuid, "netuid"),
            (context.registry_address, self.registry_address, "registry address"),
            (context.model_id, self.model_id, "model_id"),
            (
                context.static_manifest_digest,
                self.static_manifest_digest,
                "static manifest",
            ),
            (
                context.model_spec_identity_digest,
                self.model_spec_identity_digest,
                "model-spec identity",
            ),
            (
                context.model_weight_merkle_root,
                self.model_weight_merkle_root,
                "model weight Merkle root",
            ),
            (
                context.source_weight_file_hash,
                self.source_weight_file_hash,
                "source weight file hash",
            ),
            (
                context.canonical_model_bytes_abi_id,
                self.canonical_model_bytes_abi_id,
                "canonical model-bytes ABI",
            ),
            (
                context.canonical_model_bytes_digest,
                self.canonical_model_bytes_digest,
                "canonical model bytes",
            ),
            (
                context.source_file_index_digest,
                self.source_file_index_digest,
                "source-file index",
            ),
            (
                context.canonicalization_recipe_digest,
                self.canonicalization_recipe_digest,
                "canonicalization recipe",
            ),
            (
                context.pallas_catalog_abi_id,
                self.pallas_catalog_abi_id,
                "Pallas catalog ABI",
            ),
            (
                context.pallas_catalog_sha256,
                self.pallas_catalog_sha256,
                "Pallas catalog",
            ),
            (
                context.pallas_catalog_size,
                self.pallas_catalog_size,
                "Pallas catalog size",
            ),
            (
                context.pallas_pcs_suite,
                self.pallas_pcs_suite,
                "Pallas PCS suite",
            ),
            (
                context.pallas_generator_version,
                self.pallas_generator_version,
                "Pallas generator version",
            ),
            (
                context.quantization_binding_digest,
                self.quantization_binding_digest,
                "quantization binding",
            ),
            (context.kernel_abi_digest, self.kernel_abi_digest, "kernel ABI"),
            (
                context.dynamic_backend_abi_id,
                self.dynamic_backend_abi_id,
                "dynamic backend ABI",
            ),
            (
                context.dynamic_backend_digest,
                self.dynamic_backend_digest,
                "dynamic backend",
            ),
            (
                context.static_table_root,
                self.static_table_root,
                "static-table root",
            ),
            (
                context.static_table_construction_abi_id,
                self.static_table_construction_abi_id,
                "static-table construction ABI",
            ),
            (
                context.static_table_construction_digest,
                self.static_table_construction_digest,
                "static-table construction",
            ),
            (
                context.static_table_layout_digest,
                self.static_table_layout_digest,
                "static-table layout",
            ),
            (context.static_tables, self.static_tables, "static table descriptors"),
        )
        for expected, actual, name in comparisons:
            if expected != actual:
                raise ProofV3VerificationError(
                    f"static-weight artifact does not match the expected {name}"
                )

    def validate_source_binding(self, source: StaticTableSourceBindingV3) -> None:
        """Require the loaded source-binding declaration to match this artifact."""

        if not isinstance(source, StaticTableSourceBindingV3):
            raise ProofV3VerificationError(
                "static-table source binding has an unexpected type"
            )
        if source.digest() != self.static_source_binding_digest:
            raise ProofV3VerificationError(
                "static-table source binding does not match the signed artifact"
            )
        comparisons = (
            (
                source.canonical_model_bytes_abi_id,
                self.canonical_model_bytes_abi_id,
                "canonical model-bytes ABI",
            ),
            (
                source.canonical_model_bytes_digest,
                self.canonical_model_bytes_digest,
                "canonical model bytes",
            ),
            (
                source.source_file_index_digest,
                self.source_file_index_digest,
                "source-file index",
            ),
            (
                source.canonicalization_recipe_digest,
                self.canonicalization_recipe_digest,
                "canonicalization recipe",
            ),
            (source.static_table_root, self.static_table_root, "static-table root"),
            (
                source.static_table_construction_abi_id,
                self.static_table_construction_abi_id,
                "static-table construction ABI",
            ),
            (
                source.static_table_construction_digest,
                self.static_table_construction_digest,
                "static-table construction",
            ),
            (
                source.static_table_layout_digest,
                self.static_table_layout_digest,
                "static-table layout",
            ),
            (
                source.quantization_binding_digest,
                self.quantization_binding_digest,
                "quantization binding",
            ),
            (source.field_id, self.field_id, "static-table field"),
            (source.table_hash_abi_id, self.table_hash_abi_id, "static-table hash ABI"),
            (
                source.table_encoding_abi_id,
                self.table_encoding_abi_id,
                "static-table encoding ABI",
            ),
            (
                source.table_layout_abi_id,
                self.table_layout_abi_id,
                "static-table layout ABI",
            ),
            (
                source.logical_leaf_count,
                self.logical_leaf_count,
                "static-table logical leaf count",
            ),
            (
                source.static_page_count,
                self.static_page_count,
                "static-table page count",
            ),
            (
                source.tree_leaf_count,
                self.tree_leaf_count,
                "static-table tree leaf count",
            ),
            (
                source.page_leaf_capacity,
                self.page_leaf_capacity,
                "static-table page capacity",
            ),
            (
                source.leaf_byte_width,
                self.leaf_byte_width,
                "static-table leaf width",
            ),
        )
        for expected, actual, name in comparisons:
            if expected != actual:
                raise ProofV3VerificationError(
                    f"static-table source binding does not match the expected {name}"
                )

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "StaticWeightArtifactV3":
        reader = _Reader(encoded)
        (
            magic,
            version,
            chain_id,
            netuid,
            registry_address,
            pallas_catalog_size,
            logical_leaf_count,
            static_page_count,
            tree_leaf_count,
            page_leaf_capacity,
            leaf_byte_width,
            descriptor_count,
        ) = reader.unpack("<4sHII20sQQQQIIH")
        if magic != STATIC_WEIGHT_ARTIFACT_MAGIC_V3:
            raise ProofV3Error("static-weight artifact header is not supported")
        if version != STATIC_WEIGHT_ARTIFACT_FORMAT_VERSION_V3:
            raise ProofV3Error("static-weight artifact version is not supported")
        if (
            type(descriptor_count) is not int
            or not 0 < descriptor_count <= MAX_STATIC_TABLE_DESCRIPTORS_V3
        ):
            raise ProofV3Error("static table descriptor count is out of range")
        fixed = tuple(reader.read(32) for _ in range(15))
        model_id = reader.model_id()
        canonical_model_bytes_abi_id = reader.identifier("canonical_model_bytes_abi_id")
        static_table_construction_abi_id = reader.identifier(
            "static_table_construction_abi_id"
        )
        pallas_catalog_abi_id = reader.identifier("pallas_catalog_abi_id")
        pallas_pcs_suite = reader.identifier("pallas_pcs_suite")
        pallas_generator_version = reader.identifier("pallas_generator_version")
        dynamic_backend_abi_id = reader.identifier("dynamic_backend_abi_id")
        field_id = reader.identifier("field_id")
        table_hash_abi_id = reader.identifier("table_hash_abi_id")
        table_encoding_abi_id = reader.identifier("table_encoding_abi_id")
        table_layout_abi_id = reader.identifier("table_layout_abi_id")
        descriptors = tuple(
            StaticTableDescriptorV3.from_canonical_bytes(
                reader.bytes(
                    "static table descriptor",
                    MAX_STATIC_WEIGHT_ARTIFACT_BYTES_V3,
                )
            )
            for _ in range(descriptor_count)
        )
        reader.finish()
        final = cls(
            chain_id=chain_id,
            netuid=netuid,
            registry_address=registry_address,
            model_id=model_id,
            static_manifest_digest=fixed[0],
            model_spec_identity_digest=fixed[1],
            model_weight_merkle_root=fixed[2],
            source_weight_file_hash=fixed[3],
            canonical_model_bytes_digest=fixed[4],
            source_file_index_digest=fixed[5],
            canonicalization_recipe_digest=fixed[6],
            pallas_catalog_abi_id=pallas_catalog_abi_id,
            pallas_catalog_sha256=fixed[7],
            pallas_catalog_size=pallas_catalog_size,
            pallas_pcs_suite=pallas_pcs_suite,
            pallas_generator_version=pallas_generator_version,
            dynamic_backend_abi_id=dynamic_backend_abi_id,
            dynamic_backend_digest=fixed[8],
            static_table_root=fixed[9],
            static_source_binding_digest=fixed[10],
            static_table_construction_digest=fixed[11],
            static_table_layout_digest=fixed[12],
            quantization_binding_digest=fixed[13],
            kernel_abi_digest=fixed[14],
            logical_leaf_count=logical_leaf_count,
            static_page_count=static_page_count,
            tree_leaf_count=tree_leaf_count,
            page_leaf_capacity=page_leaf_capacity,
            leaf_byte_width=leaf_byte_width,
            static_tables=descriptors,
            canonical_model_bytes_abi_id=canonical_model_bytes_abi_id,
            static_table_construction_abi_id=static_table_construction_abi_id,
            field_id=field_id,
            table_hash_abi_id=table_hash_abi_id,
            table_encoding_abi_id=table_encoding_abi_id,
            table_layout_abi_id=table_layout_abi_id,
            artifact_version=version,
        )
        if final.canonical_bytes() != encoded:
            raise ProofV3Error("static-weight artifact is not canonical")
        return final


__all__ = [
    "CANONICAL_MODEL_BYTES_ABI_ID_V3",
    "GOLDILOCKS_DYNAMIC_BACKEND_ABI_ID_V3",
    "GOLDILOCKS_STATIC_FIELD_ID_V3",
    "GOLDILOCKS_STATIC_TABLE_CONSTRUCTION_ABI_ID_V3",
    "GOLDILOCKS_STATIC_TABLE_ENCODING_ABI_ID_V3",
    "GOLDILOCKS_STATIC_TABLE_HASH_ABI_ID_V3",
    "GOLDILOCKS_STATIC_TABLE_LAYOUT_ABI_ID_V3",
    "MAX_STATIC_TABLE_DESCRIPTORS_V3",
    "MAX_STATIC_WEIGHT_ARTIFACT_BYTES_V3",
    "MAX_QUANTIZATION_BINDING_BYTES_V3",
    "PALLAS_COLUMN_CATALOG_ABI_ID_V3",
    "QUANTIZATION_BINDING_ABI_ID_V3",
    "QUANTIZATION_BINDING_FORMAT_VERSION_V3",
    "V2_PALLAS_GENERATOR_VERSION_V3",
    "V2_PALLAS_PCS_SUITE_V3",
    "STATIC_TABLE_LAYOUT_FORMAT_VERSION_V3",
    "STATIC_TABLE_SOURCE_BINDING_DIGEST_DOMAIN_V3",
    "STATIC_TABLE_SOURCE_BINDING_FORMAT_VERSION_V3",
    "STATIC_TABLE_DESCRIPTOR_DIGEST_DOMAIN_V3",
    "STATIC_WEIGHT_ARTIFACT_DIGEST_DOMAIN_V3",
    "STATIC_WEIGHT_ARTIFACT_FORMAT_VERSION_V3",
    "QuantizationBindingArtifactV3",
    "StaticTableDescriptorV3",
    "StaticTableSourceBindingV3",
    "StaticWeightArtifactContextV3",
    "StaticWeightArtifactV3",
    "canonical_static_table_layout_bytes_v3",
    "static_table_layout_from_canonical_bytes_v3",
]
