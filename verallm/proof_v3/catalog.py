"""Validator-owned authenticated artifact views for qualified proof-v3 paths.

The profile is a signed description, not a substitute for the bytes loaded by
the validator.  This module joins that description to a catalog entry built
from an already authenticated static manifest and exact local verifier/circuit/
capture/tokenizer/sampler/quantization artifacts.  It performs no network,
signature, or model-registry I/O itself.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from verallm.proof_v3.constraint_system import (
    GoldilocksExecutionConstraintSystemV3,
)
from verallm.proof_v3.constraint_program import (
    GoldilocksConstraintProgramBundleV3,
)
from verallm.proof_v3.errors import ProofV3VerificationError
from verallm.proof_v3.profile import ExecutionSecurityProfileV3
from verallm.proof_v3.relation import (
    StaticParameterBindingV3,
    StaticTableBindingV3,
    validate_registered_operations_against_manifest_v3,
)
from verallm.proof_v3.static_artifact import (
    GOLDILOCKS_DYNAMIC_BACKEND_ABI_ID_V3,
    PALLAS_COLUMN_CATALOG_ABI_ID_V3,
    QuantizationBindingArtifactV3,
    StaticTableSourceBindingV3,
    StaticWeightArtifactContextV3,
    StaticWeightArtifactV3,
    V2_PALLAS_GENERATOR_VERSION_V3,
    V2_PALLAS_PCS_SUITE_V3,
    static_table_layout_from_canonical_bytes_v3,
)


MAX_QUALIFIED_ARTIFACT_BYTES_V3 = 64 << 20
MAX_PALLAS_CATALOG_BYTES_V3 = 256 << 20
MODEL_SPEC_IDENTITY_DIGEST_DOMAIN_V3 = b"VERATHOS/PROOF_V3/MODEL_SPEC_IDENTITY/SHA256"
_V2_CATALOG_BINDING_FACTORY_TOKEN = object()
_V3_PROJECTION_MANIFEST_LOADER_TOKEN = object()


def _fixed32(value: bytes, name: str, *, nonzero: bool = False) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3VerificationError(f"{name} must be exactly 32 bytes")
    if nonzero and value == bytes(32):
        raise ProofV3VerificationError(f"{name} must not be the zero digest")
    return value


def _artifact_bytes(value: bytes, name: str) -> bytes:
    if (
        not isinstance(value, bytes)
        or not value
        or len(value) > MAX_QUALIFIED_ARTIFACT_BYTES_V3
    ):
        raise ProofV3VerificationError(f"{name} byte length is out of range")
    return value


def _pallas_catalog_bytes(value: bytes) -> bytes:
    if (
        not isinstance(value, bytes)
        or not value
        or len(value) > MAX_PALLAS_CATALOG_BYTES_V3
    ):
        raise ProofV3VerificationError("Pallas catalog byte length is out of range")
    return value


def _ascii_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProofV3VerificationError(f"{name} is malformed")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ProofV3VerificationError(f"{name} is malformed") from exc
    if len(encoded) > 128:
        raise ProofV3VerificationError(f"{name} is malformed")
    return value


def _model_quant_mode(value: object, name: str) -> str:
    """Validate the exact NFC ``ModelSpec.quant_mode`` text value.

    This deliberately mirrors the V2 model-spec representation instead of
    treating a model-qualified quantization mode as a generic ABI identifier.
    """

    if not isinstance(value, str) or not value:
        raise ProofV3VerificationError(f"{name} is malformed")
    if value != unicodedata.normalize("NFC", value) or "\x00" in value:
        raise ProofV3VerificationError(f"{name} is malformed")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ProofV3VerificationError(f"{name} is malformed") from exc
    if len(encoded) > 128:
        raise ProofV3VerificationError(f"{name} is malformed")
    return value


def model_spec_identity_digest_v3(canonical_model_spec_bytes: bytes) -> bytes:
    """Domain-separate the exact V2 ModelSpec identity for V3 artifacts."""

    encoded = _artifact_bytes(
        canonical_model_spec_bytes,
        "canonical model-spec identity",
    )
    return hashlib.sha256(MODEL_SPEC_IDENTITY_DIGEST_DOMAIN_V3 + encoded).digest()


@dataclass(frozen=True, slots=True)
class AuthenticatedStaticManifestV3:
    """Static-manifest data already authenticated by the validator catalog.

    The surrounding catalog loader must authenticate ``manifest_digest`` to the
    on-chain model specification before constructing this object.  The exact
    operation descriptors and static bindings are then cross-checked against
    the signed execution profile before proof parsing.
    """

    manifest_digest: bytes
    chain_id: int
    netuid: int
    registry_address: bytes
    model_id: str
    model_spec_identity_digest: bytes
    model_weight_merkle_root: bytes
    source_weight_file_hash: bytes
    pallas_catalog_abi_id: str
    pallas_catalog_sha256: bytes
    pallas_catalog_size: int
    pallas_pcs_suite: str
    pallas_generator_version: str
    quantization_semantics_id: str
    model_quant_mode: str
    operations: tuple[object, ...]
    static_bindings: tuple[StaticParameterBindingV3, ...]

    def __post_init__(self) -> None:
        _fixed32(self.manifest_digest, "manifest_digest", nonzero=True)
        if (
            isinstance(self.chain_id, bool)
            or not isinstance(self.chain_id, int)
            or self.chain_id <= 0
            or self.chain_id >= 1 << 32
        ):
            raise ProofV3VerificationError(
                "authenticated manifest chain_id is malformed"
            )
        if (
            isinstance(self.netuid, bool)
            or not isinstance(self.netuid, int)
            or not 0 <= self.netuid < 1 << 32
        ):
            raise ProofV3VerificationError("authenticated manifest netuid is malformed")
        if (
            type(self.registry_address) is not bytes
            or len(self.registry_address) != 20
            or self.registry_address == bytes(20)
        ):
            raise ProofV3VerificationError(
                "authenticated manifest registry address is malformed"
            )
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ProofV3VerificationError(
                "authenticated manifest model_id is malformed"
            )
        try:
            model_id_bytes = self.model_id.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ProofV3VerificationError(
                "authenticated manifest model_id is malformed"
            ) from exc
        if len(model_id_bytes) > 4_096 or "\x00" in self.model_id:
            raise ProofV3VerificationError(
                "authenticated manifest model_id is malformed"
            )
        for value, name in (
            (self.model_spec_identity_digest, "model_spec_identity_digest"),
            (self.model_weight_merkle_root, "model_weight_merkle_root"),
            (self.pallas_catalog_sha256, "pallas_catalog_sha256"),
        ):
            _fixed32(value, name, nonzero=True)
        # Older registered models legitimately carry an all-zero optional
        # source-file hash. The exact on-chain ModelSpec identity and nonzero
        # registered Merkle/catalog roots remain mandatory.
        _fixed32(self.source_weight_file_hash, "source_weight_file_hash")
        if self.pallas_catalog_abi_id != PALLAS_COLUMN_CATALOG_ABI_ID_V3:
            raise ProofV3VerificationError(
                "authenticated manifest Pallas catalog ABI is unsupported"
            )
        if (
            isinstance(self.pallas_catalog_size, bool)
            or not isinstance(self.pallas_catalog_size, int)
            or not 0 < self.pallas_catalog_size <= MAX_PALLAS_CATALOG_BYTES_V3
        ):
            raise ProofV3VerificationError(
                "authenticated manifest Pallas catalog size is malformed"
            )
        _ascii_identifier(
            self.pallas_pcs_suite,
            "authenticated manifest pallas_pcs_suite",
        )
        _ascii_identifier(
            self.pallas_generator_version,
            "authenticated manifest pallas_generator_version",
        )
        if self.pallas_pcs_suite != V2_PALLAS_PCS_SUITE_V3:
            raise ProofV3VerificationError(
                "authenticated manifest Pallas PCS suite is unsupported"
            )
        if self.pallas_generator_version != V2_PALLAS_GENERATOR_VERSION_V3:
            raise ProofV3VerificationError(
                "authenticated manifest Pallas generator version is unsupported"
            )
        _ascii_identifier(self.quantization_semantics_id, "quantization_semantics_id")
        _model_quant_mode(self.model_quant_mode, "model_quant_mode")
        operations = tuple(self.operations)
        if not operations:
            raise ProofV3VerificationError("authenticated manifest has no operations")
        bindings = tuple(self.static_bindings)
        if not bindings or not all(
            isinstance(binding, StaticParameterBindingV3) for binding in bindings
        ):
            raise ProofV3VerificationError(
                "authenticated manifest bindings are malformed"
            )
        keys = tuple(binding.binding_id for binding in bindings)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ProofV3VerificationError(
                "authenticated manifest bindings are not canonically ordered"
            )
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "static_bindings", bindings)


def _verify_v2_catalog_bytes_against_manifest_v3(
    *,
    manifest: object,
    catalog_bytes: bytes,
) -> None:
    """Parse and rebuild every V2 Pallas tree from canonical source bytes.

    A digest and byte length only identify a file; they do not establish that
    its per-operation columns reconstruct every signed root. The factory calls
    this once before minting an immutable bridge. Subsequent serving/session
    paths use the bridge's exact source identities and never repeat this
    potentially hours-long catalog qualification work.
    """

    try:
        from verallm.proof_v2.catalog import WeightCommitmentCatalogV2
        from verallm.proof_v2.document import PCS_GENERATOR_VERSION, PCS_SUITE
        from verallm.proof_v2.engine import build_weight_commitment_catalog_tree_v2
        from verallm.proof_v2.layout import registered_all_operations_from_manifest
        from verallm.proof_v2.manifest import StaticWeightCommitmentManifest
    except ImportError as exc:  # pragma: no cover - V2 is a package dependency.
        raise ProofV3VerificationError(
            "proof-v2 catalog support is unavailable"
        ) from exc
    if not isinstance(manifest, StaticWeightCommitmentManifest):
        raise ProofV3VerificationError(
            "v3 catalog binding has an invalid loader-verified proof-v2 manifest"
        )
    encoded = _pallas_catalog_bytes(catalog_bytes)
    if (
        manifest.pcs_suite != PCS_SUITE
        or manifest.pcs_generator_version != PCS_GENERATOR_VERSION
    ):
        raise ProofV3VerificationError(
            "loader-verified proof-v2 manifest has an unsupported PCS configuration"
        )
    try:
        catalog = WeightCommitmentCatalogV2.from_canonical_bytes(encoded)
        if catalog.canonical_bytes() != encoded:
            raise ProofV3VerificationError(
                "v3 catalog binding has a noncanonical loader-verified catalog"
            )
        if catalog.manifest_digest != manifest.digest():
            raise ProofV3VerificationError(
                "loader-verified catalog does not match the proof-v2 manifest"
            )
        registered_operations = registered_all_operations_from_manifest(manifest)
        if catalog.operation_keys != tuple(
            operation.key for operation in registered_operations
        ):
            raise ProofV3VerificationError(
                "loader-verified catalog operations do not exactly cover the manifest"
            )
        for operation in registered_operations:
            build_weight_commitment_catalog_tree_v2(
                operation,
                catalog.lookup(operation.key),
            )
    except ProofV3VerificationError:
        raise
    except Exception as exc:
        raise ProofV3VerificationError(
            "loader-verified catalog does not authenticate every signed operation"
        ) from exc


@dataclass(frozen=True, slots=True, init=False)
class VerifiedProjectionManifestV3:
    """Chain-authenticated static projection manifest for a v3 profile.

    The document deliberately uses the existing canonical Pallas manifest
    format, but it is not admitted as a proof-v2 execution profile. V3 owns
    the runtime transition relation; this source authenticates only the exact
    registered model identity, operation inventory and static PCS roots.
    """

    document: object
    recovered_signers: tuple[str, ...]
    source_path: Path
    _loader_token: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ProofV3VerificationError(
            "verified projection manifests must be created by the v3 loader"
        )

    @property
    def manifest(self):
        return self.document.manifest

    @classmethod
    def _from_verified_loader(
        cls,
        *,
        document: object,
        recovered_signers: tuple[str, ...],
        source_path: Path,
        _factory_token: object | None = None,
    ) -> "VerifiedProjectionManifestV3":
        if _factory_token is not _V3_PROJECTION_MANIFEST_LOADER_TOKEN:
            raise ProofV3VerificationError(
                "verified projection manifests must be created by the v3 loader"
            )
        result = object.__new__(cls)
        object.__setattr__(result, "document", document)
        object.__setattr__(result, "recovered_signers", recovered_signers)
        object.__setattr__(result, "source_path", source_path)
        object.__setattr__(
            result,
            "_loader_token",
            _V3_PROJECTION_MANIFEST_LOADER_TOKEN,
        )
        result.require_loader_provenance()
        return result

    def require_loader_provenance(self) -> None:
        if self._loader_token is not _V3_PROJECTION_MANIFEST_LOADER_TOKEN:
            raise ProofV3VerificationError(
                "projection manifest lacks v3 loader provenance"
            )
        try:
            from verallm.proof_v2.document import SignedManifestDocument
        except ImportError as exc:  # pragma: no cover
            raise ProofV3VerificationError(
                "static projection manifest support is unavailable"
            ) from exc
        if (
            not isinstance(self.document, SignedManifestDocument)
            or not isinstance(self.source_path, Path)
            or not self.recovered_signers
        ):
            raise ProofV3VerificationError(
                "projection manifest lacks v3 loader provenance"
            )


def load_verified_projection_manifest_v3(
    path: str | Path,
    *,
    chain_config,
    model_registry_client,
    expected_model_id: str | None = None,
) -> VerifiedProjectionManifestV3:
    """Authenticate a static projection manifest against live chain state."""

    source = Path(path).expanduser().resolve()
    try:
        from verallm.proof_v2.document import (
            load_signed_manifest_document,
            verify_manifest_document,
        )
        from verallm.proof_v2.layout import (
            validate_qwen_hybrid_execution_manifest_profile,
        )

        document = load_signed_manifest_document(source)
        model_id = expected_model_id or document.manifest.model_spec.model_id
        if document.manifest.model_spec.model_id != model_id:
            raise ProofV3VerificationError(
                "projection manifest model_id does not match the requested model"
            )
        if not hasattr(model_registry_client, "get_on_chain_model_spec"):
            raise ProofV3VerificationError(
                "projection manifests require an exact on-chain ModelSpec reader"
            )
        on_chain_spec = model_registry_client.get_on_chain_model_spec(model_id)
        if on_chain_spec is None:
            raise ProofV3VerificationError(
                f"projection manifest model is not registered: {model_id}"
            )
        authority = model_registry_client.get_manifest_authority()
        recovered = verify_manifest_document(
            document,
            expected_chain_id=chain_config.chain_id,
            expected_netuid=chain_config.netuid,
            expected_registry_address=chain_config.model_registry_address,
            expected_model_spec=on_chain_spec,
            expected_authority_signers=authority.signers,
            authority_threshold=authority.threshold,
        )
        # This validates the canonical complete Qwen projection inventory and
        # dimensions. It intentionally does not require proof-v2 transition
        # eligibility: the signed v3 profile supplies that relation.
        validate_qwen_hybrid_execution_manifest_profile(document.manifest)
    except ProofV3VerificationError:
        raise
    except Exception as exc:
        raise ProofV3VerificationError(
            f"projection manifest authentication failed: {source}"
        ) from exc
    return VerifiedProjectionManifestV3._from_verified_loader(
        document=document,
        recovered_signers=recovered,
        source_path=source,
        _factory_token=_V3_PROJECTION_MANIFEST_LOADER_TOKEN,
    )


@dataclass(frozen=True, slots=True, init=False)
class VerifiedV2CatalogBindingV3:
    """A V3 static view derived only from an authenticated V2 manifest/catalog.

    The factory deliberately accepts the in-memory objects produced by the V2
    loader rather than catalog metadata claims.  It reconstructs every V2
    static Pallas tree against the signed operation root before deriving the
    V3 view.  This is an offline qualification input only; it does not load
    model weights and does not enable a V3 serving adapter.
    """

    static_manifest: AuthenticatedStaticManifestV3
    catalog_bytes: bytes
    _source_verified_manifest: object
    _source_weight_catalog: object
    _source_catalog_bytes: bytes
    _source_catalog_digest: bytes
    _factory_token: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ProofV3VerificationError(
            "v3 catalog bindings must be created through from_verified_v2"
        )

    @classmethod
    def _construct_from_verified_v2(
        cls,
        *,
        static_manifest: AuthenticatedStaticManifestV3,
        catalog_bytes: bytes,
        source_verified_manifest: object,
        source_weight_catalog: object | None = None,
        source_catalog_bytes: bytes,
        source_catalog_digest: bytes,
        _factory_token: object | None = None,
    ) -> "VerifiedV2CatalogBindingV3":
        """Mint the opaque bridge after the V2 source has been reconstructed."""

        if _factory_token is not _V2_CATALOG_BINDING_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "v3 catalog bindings must be created through from_verified_v2"
            )
        result = object.__new__(cls)
        object.__setattr__(result, "static_manifest", static_manifest)
        object.__setattr__(result, "catalog_bytes", catalog_bytes)
        object.__setattr__(
            result, "_source_verified_manifest", source_verified_manifest
        )
        object.__setattr__(
            result,
            "_source_weight_catalog",
            source_weight_catalog,
        )
        object.__setattr__(result, "_source_catalog_bytes", source_catalog_bytes)
        object.__setattr__(result, "_source_catalog_digest", source_catalog_digest)
        object.__setattr__(result, "_factory_token", _V2_CATALOG_BINDING_FACTORY_TOKEN)
        result.require_verified_v2_provenance()
        return result

    def require_verified_v2_provenance(self) -> None:
        """Require a factory-minted, reconstructed V2 source record.

        This is a trusted-validator API boundary. It prevents a normal caller
        from treating a hand-constructed dataclass or unauthenticated byte
        string as an authenticated catalog; it is not a defense against code
        already able to subvert the validator process itself. Deep Pallas-tree
        reconstruction happened before this immutable record was minted; this
        method deliberately remains constant-cost for serving.
        """

        if self._factory_token is not _V2_CATALOG_BINDING_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "v3 catalog binding requires factory-authenticated proof-v2 provenance"
            )
        if not isinstance(self.static_manifest, AuthenticatedStaticManifestV3):
            raise ProofV3VerificationError("v2 static manifest has an unexpected type")
        _pallas_catalog_bytes(self.catalog_bytes)

        try:
            from verallm.proof_v2.runtime import VerifiedProofV2Manifest
        except ImportError as exc:  # pragma: no cover - V2 is a dependency.
            raise ProofV3VerificationError(
                "proof-v2 catalog provenance support is unavailable"
            ) from exc
        source_manifest = self._source_verified_manifest
        source_weight_catalog = self._source_weight_catalog
        source_catalog_bytes = self._source_catalog_bytes
        if isinstance(source_manifest, VerifiedProofV2Manifest):
            require_source_provenance = source_manifest.require_loader_provenance
        elif isinstance(source_manifest, VerifiedProjectionManifestV3):
            require_source_provenance = source_manifest.require_loader_provenance
        else:
            raise ProofV3VerificationError(
                "v3 catalog binding requires loader-authenticated projection provenance"
            )
        try:
            from verallm.proof_v2.catalog import WeightCommitmentCatalogV2
        except ImportError as exc:  # pragma: no cover
            raise ProofV3VerificationError(
                "proof-v2 catalog provenance support is unavailable"
            ) from exc
        if type(source_catalog_bytes) is not bytes:
            raise ProofV3VerificationError(
                "v3 catalog binding requires loader-authenticated proof-v2 provenance"
            )
        if not isinstance(source_weight_catalog, WeightCommitmentCatalogV2):
            raise ProofV3VerificationError(
                "v3 catalog binding lacks its loader-authenticated catalog"
            )
        try:
            require_source_provenance()
        except Exception as exc:
            raise ProofV3VerificationError(
                "v3 catalog binding requires loader-authenticated projection provenance"
            ) from exc
        if self.catalog_bytes is not source_catalog_bytes:
            raise ProofV3VerificationError(
                "v3 catalog binding does not match its loader-verified catalog"
            )
        source_catalog_digest = self._source_catalog_digest
        if (
            type(source_catalog_digest) is not bytes
            or len(source_catalog_digest) != 32
            or source_catalog_digest != self.static_manifest.pallas_catalog_sha256
        ):
            raise ProofV3VerificationError(
                "v3 catalog binding does not match its loader-verified catalog"
            )

        manifest = source_manifest.manifest
        spec = manifest.model_spec
        static_manifest = self.static_manifest
        expected_fields = (
            (static_manifest.manifest_digest, manifest.digest()),
            (static_manifest.chain_id, manifest.chain_id),
            (static_manifest.netuid, manifest.netuid),
            (static_manifest.registry_address, manifest.registry_address),
            (static_manifest.model_id, spec.model_id),
            (
                static_manifest.model_spec_identity_digest,
                model_spec_identity_digest_v3(spec.canonical_bytes()),
            ),
            (static_manifest.model_weight_merkle_root, spec.weight_merkle_root),
            (static_manifest.source_weight_file_hash, spec.weight_file_hash),
            (static_manifest.pallas_catalog_sha256, source_catalog_digest),
            (static_manifest.pallas_catalog_size, len(source_catalog_bytes)),
            (static_manifest.pallas_pcs_suite, manifest.pcs_suite),
            (
                static_manifest.pallas_generator_version,
                manifest.pcs_generator_version,
            ),
            (static_manifest.model_quant_mode, spec.quant_mode),
            (static_manifest.operations, manifest.operations),
        )
        if any(actual != expected for actual, expected in expected_fields):
            raise ProofV3VerificationError(
                "v3 catalog binding does not match its loader-verified proof-v2 manifest"
            )

    def deep_revalidate_v2_catalog(self) -> None:
        """Rebuild every signed V2 tree for offline diagnostics only.

        Normal qualification calls :meth:`from_verified_v2`, which performs
        this exact work once before minting the bridge. Calling this method on
        a hot serving path would be a validator CPU denial-of-service risk.
        """

        self.require_verified_v2_provenance()
        _verify_v2_catalog_bytes_against_manifest_v3(
            manifest=self._source_verified_manifest.manifest,
            catalog_bytes=self._source_catalog_bytes,
        )

    @classmethod
    def from_verified_v2(
        cls,
        *,
        verified_manifest: object,
        weight_catalog: object,
        quantization_semantics_id: str,
        static_bindings: tuple[StaticParameterBindingV3, ...],
    ) -> "VerifiedV2CatalogBindingV3":
        """Derive a fail-closed V3 catalog binding from the verified V2 leg."""

        try:
            from verallm.proof_v2.catalog import WeightCommitmentCatalogV2
            from verallm.proof_v2.runtime import VerifiedProofV2Manifest
        except ImportError as exc:  # pragma: no cover - V2 is a package dependency.
            raise ProofV3VerificationError(
                "proof-v2 catalog support is unavailable"
            ) from exc
        if not isinstance(verified_manifest, VerifiedProofV2Manifest):
            raise ProofV3VerificationError(
                "v3 catalog binding requires a verified proof-v2 manifest"
            )
        try:
            verified_manifest.require_loader_provenance()
        except Exception as exc:
            raise ProofV3VerificationError(
                "v3 catalog binding requires loader-authenticated proof-v2 provenance"
            ) from exc
        if not isinstance(weight_catalog, WeightCommitmentCatalogV2):
            raise ProofV3VerificationError(
                "v3 catalog binding requires a canonical proof-v2 catalog"
            )
        return cls._from_authenticated_source(
            verified_manifest=verified_manifest,
            weight_catalog=weight_catalog,
            quantization_semantics_id=quantization_semantics_id,
            static_bindings=static_bindings,
        )

    @classmethod
    def from_verified_projection_manifest(
        cls,
        *,
        verified_manifest: object,
        weight_catalog: object,
        quantization_semantics_id: str,
        static_bindings: tuple[StaticParameterBindingV3, ...],
        validation_context=None,
    ) -> "VerifiedV2CatalogBindingV3":
        """Bind a v3-authenticated projection manifest to its Pallas catalog."""

        try:
            from verallm.proof_v2.catalog import WeightCommitmentCatalogV2
        except ImportError as exc:  # pragma: no cover
            raise ProofV3VerificationError(
                "Pallas projection catalog support is unavailable"
            ) from exc
        if not isinstance(verified_manifest, VerifiedProjectionManifestV3):
            raise ProofV3VerificationError(
                "v3 catalog binding requires a verified projection manifest"
            )
        verified_manifest.require_loader_provenance()
        if not isinstance(weight_catalog, WeightCommitmentCatalogV2):
            raise ProofV3VerificationError(
                "v3 catalog binding requires a canonical Pallas catalog"
            )
        return cls._from_authenticated_source(
            verified_manifest=verified_manifest,
            weight_catalog=weight_catalog,
            quantization_semantics_id=quantization_semantics_id,
            static_bindings=static_bindings,
            validation_context=validation_context,
        )

    @classmethod
    def _from_authenticated_source(
        cls,
        *,
        verified_manifest: object,
        weight_catalog: object,
        quantization_semantics_id: str,
        static_bindings: tuple[StaticParameterBindingV3, ...],
        validation_context=None,
    ) -> "VerifiedV2CatalogBindingV3":
        manifest = verified_manifest.manifest
        manifest_digest = manifest.digest()
        catalog_bytes = weight_catalog.canonical_bytes()
        if validation_context is None:
            _verify_v2_catalog_bytes_against_manifest_v3(
                manifest=manifest,
                catalog_bytes=catalog_bytes,
            )
        else:
            from verallm.proof_v3.catalog_validation_cache import (
                CatalogValidationContextV3,
            )

            if not isinstance(
                validation_context,
                CatalogValidationContextV3,
            ):
                raise ProofV3VerificationError(
                    "projection catalog validation context is malformed"
                )
            validation_context.require_exact_source(
                verified_manifest=verified_manifest,
                catalog_bytes=catalog_bytes,
            )
            if not validation_context.cache_hit:
                _verify_v2_catalog_bytes_against_manifest_v3(
                    manifest=manifest,
                    catalog_bytes=catalog_bytes,
                )
                validation_context.publish_after_deep_validation()
        spec = manifest.model_spec
        return cls._construct_from_verified_v2(
            static_manifest=AuthenticatedStaticManifestV3(
                manifest_digest=manifest_digest,
                chain_id=manifest.chain_id,
                netuid=manifest.netuid,
                registry_address=manifest.registry_address,
                model_id=spec.model_id,
                model_spec_identity_digest=model_spec_identity_digest_v3(
                    spec.canonical_bytes()
                ),
                model_weight_merkle_root=spec.weight_merkle_root,
                source_weight_file_hash=spec.weight_file_hash,
                pallas_catalog_abi_id=PALLAS_COLUMN_CATALOG_ABI_ID_V3,
                pallas_catalog_sha256=hashlib.sha256(catalog_bytes).digest(),
                pallas_catalog_size=len(catalog_bytes),
                pallas_pcs_suite=manifest.pcs_suite,
                pallas_generator_version=manifest.pcs_generator_version,
                quantization_semantics_id=quantization_semantics_id,
                model_quant_mode=spec.quant_mode,
                operations=manifest.operations,
                static_bindings=static_bindings,
            ),
            catalog_bytes=catalog_bytes,
            source_verified_manifest=verified_manifest,
            source_weight_catalog=weight_catalog,
            source_catalog_bytes=catalog_bytes,
            source_catalog_digest=hashlib.sha256(catalog_bytes).digest(),
            _factory_token=_V2_CATALOG_BINDING_FACTORY_TOKEN,
        )


@dataclass(frozen=True, slots=True)
class QualifiedExecutionArtifactsV3:
    """Exact loaded artifacts required before a profile can dispatch natively."""

    execution_profile_digest: bytes
    verified_v2_catalog_binding: VerifiedV2CatalogBindingV3
    static_manifest: AuthenticatedStaticManifestV3
    static_artifact: StaticWeightArtifactV3
    pallas_catalog_bytes: bytes
    verifier_key_bytes: bytes
    capture_abi_bytes: bytes
    constraint_system_bytes: bytes
    constraint_program_bundle_bytes: bytes
    tokenizer_binding_bytes: bytes
    sampler_abi_bytes: bytes
    quantization_binding_bytes: bytes
    recursive_accumulator_bytes: bytes
    static_table_layout_bytes: bytes
    kernel_abi_bytes: bytes
    dynamic_backend_abi_id: str
    dynamic_backend_bytes: bytes
    static_source_binding_bytes: bytes
    canonicalization_recipe_bytes: bytes
    source_file_index_bytes: bytes
    static_table_construction_bytes: bytes
    _pallas_catalog_digest: bytes = field(init=False, repr=False, compare=False)
    _constraint_system: GoldilocksExecutionConstraintSystemV3 = field(
        init=False,
        repr=False,
        compare=False,
    )
    _constraint_program_bundle: GoldilocksConstraintProgramBundleV3 = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _fixed32(
            self.execution_profile_digest,
            "execution_profile_digest",
            nonzero=True,
        )
        if not isinstance(self.verified_v2_catalog_binding, VerifiedV2CatalogBindingV3):
            raise ProofV3VerificationError(
                "qualified artifacts require a verified proof-v2 catalog binding"
            )
        self.verified_v2_catalog_binding.require_verified_v2_provenance()
        if not isinstance(self.static_manifest, AuthenticatedStaticManifestV3):
            raise ProofV3VerificationError("static_manifest has an unexpected type")
        if not isinstance(self.static_artifact, StaticWeightArtifactV3):
            raise ProofV3VerificationError("static_artifact has an unexpected type")
        _pallas_catalog_bytes(self.pallas_catalog_bytes)
        catalog_digest = hashlib.sha256(self.pallas_catalog_bytes).digest()
        object.__setattr__(self, "_pallas_catalog_digest", catalog_digest)
        if self.static_manifest != self.verified_v2_catalog_binding.static_manifest:
            raise ProofV3VerificationError(
                "qualified artifacts static manifest is not the verified proof-v2 binding"
            )
        if (
            len(self.pallas_catalog_bytes)
            != self.verified_v2_catalog_binding.static_manifest.pallas_catalog_size
            or catalog_digest
            != self.verified_v2_catalog_binding.static_manifest.pallas_catalog_sha256
        ):
            raise ProofV3VerificationError(
                "qualified artifacts Pallas catalog is not the verified proof-v2 binding"
            )
        self.static_artifact.validate_verified_v2_catalog_binding(
            verified_v2_catalog_binding=self.verified_v2_catalog_binding
        )
        for value, name in (
            (self.verifier_key_bytes, "verifier_key_bytes"),
            (self.capture_abi_bytes, "capture_abi_bytes"),
            (self.constraint_system_bytes, "constraint_system_bytes"),
            (self.constraint_program_bundle_bytes, "constraint_program_bundle_bytes"),
            (self.tokenizer_binding_bytes, "tokenizer_binding_bytes"),
            (self.sampler_abi_bytes, "sampler_abi_bytes"),
            (self.quantization_binding_bytes, "quantization_binding_bytes"),
            (self.recursive_accumulator_bytes, "recursive_accumulator_bytes"),
            (self.static_table_layout_bytes, "static_table_layout_bytes"),
            (self.kernel_abi_bytes, "kernel_abi_bytes"),
            (self.dynamic_backend_bytes, "dynamic_backend_bytes"),
            (self.static_source_binding_bytes, "static source binding"),
            (self.canonicalization_recipe_bytes, "canonicalization recipe"),
            (self.source_file_index_bytes, "source-file index"),
            (self.static_table_construction_bytes, "static-table construction"),
        ):
            _artifact_bytes(value, name)
        try:
            constraint_system = GoldilocksExecutionConstraintSystemV3.from_canonical_bytes(
                self.constraint_system_bytes
            )
        except ProofV3VerificationError:
            raise
        except Exception as exc:
            raise ProofV3VerificationError(
                "qualified constraint-system artifact is malformed"
            ) from exc
        object.__setattr__(self, "_constraint_system", constraint_system)
        try:
            constraint_program_bundle = (
                GoldilocksConstraintProgramBundleV3.from_canonical_bytes(
                    self.constraint_program_bundle_bytes
                )
            )
        except ProofV3VerificationError:
            raise
        except Exception as exc:
            raise ProofV3VerificationError(
                "qualified constraint-program bundle artifact is malformed"
            ) from exc
        object.__setattr__(
            self,
            "_constraint_program_bundle",
            constraint_program_bundle,
        )
        if self.dynamic_backend_abi_id != GOLDILOCKS_DYNAMIC_BACKEND_ABI_ID_V3:
            raise ProofV3VerificationError("dynamic_backend_abi_id is unsupported")

    @property
    def constraint_system(self) -> GoldilocksExecutionConstraintSystemV3:
        """Return the once-parsed immutable dynamic constraint-system contract."""

        return self._constraint_system

    @property
    def constraint_program_bundle(self) -> GoldilocksConstraintProgramBundleV3:
        """Return the once-parsed program bundle bound to the signed system."""

        return self._constraint_program_bundle

    @classmethod
    def from_verified_v2_catalog_binding(
        cls,
        *,
        execution_profile_digest: bytes,
        verified_v2_catalog_binding: VerifiedV2CatalogBindingV3,
        static_artifact: StaticWeightArtifactV3,
        verifier_key_bytes: bytes,
        capture_abi_bytes: bytes,
        constraint_system_bytes: bytes,
        constraint_program_bundle_bytes: bytes,
        tokenizer_binding_bytes: bytes,
        sampler_abi_bytes: bytes,
        quantization_binding_bytes: bytes,
        recursive_accumulator_bytes: bytes,
        static_table_layout_bytes: bytes,
        kernel_abi_bytes: bytes,
        dynamic_backend_abi_id: str,
        dynamic_backend_bytes: bytes,
        static_source_binding_bytes: bytes,
        canonicalization_recipe_bytes: bytes,
        source_file_index_bytes: bytes,
        static_table_construction_bytes: bytes,
    ) -> "QualifiedExecutionArtifactsV3":
        """Build the registration input from one verified V2 catalog leg.

        This is the only intended production construction path.  It derives
        both the static manifest and Pallas bytes from the same verified V2
        binding, then lets :meth:`__post_init__` cross-check every imported
        V3 static artifact against that immutable source.
        """

        if not isinstance(verified_v2_catalog_binding, VerifiedV2CatalogBindingV3):
            raise ProofV3VerificationError(
                "qualified artifacts require a verified proof-v2 catalog binding"
            )
        return cls(
            execution_profile_digest=execution_profile_digest,
            verified_v2_catalog_binding=verified_v2_catalog_binding,
            static_manifest=verified_v2_catalog_binding.static_manifest,
            static_artifact=static_artifact,
            pallas_catalog_bytes=verified_v2_catalog_binding.catalog_bytes,
            verifier_key_bytes=verifier_key_bytes,
            capture_abi_bytes=capture_abi_bytes,
            constraint_system_bytes=constraint_system_bytes,
            constraint_program_bundle_bytes=constraint_program_bundle_bytes,
            tokenizer_binding_bytes=tokenizer_binding_bytes,
            sampler_abi_bytes=sampler_abi_bytes,
            quantization_binding_bytes=quantization_binding_bytes,
            recursive_accumulator_bytes=recursive_accumulator_bytes,
            static_table_layout_bytes=static_table_layout_bytes,
            kernel_abi_bytes=kernel_abi_bytes,
            dynamic_backend_abi_id=dynamic_backend_abi_id,
            dynamic_backend_bytes=dynamic_backend_bytes,
            static_source_binding_bytes=static_source_binding_bytes,
            canonicalization_recipe_bytes=canonicalization_recipe_bytes,
            source_file_index_bytes=source_file_index_bytes,
            static_table_construction_bytes=static_table_construction_bytes,
        )

    @staticmethod
    def _require_digest(*, artifact: bytes, expected_digest: bytes, name: str) -> None:
        expected = _fixed32(expected_digest, f"{name}_digest", nonzero=True)
        if hashlib.sha256(artifact).digest() != expected:
            raise ProofV3VerificationError(
                f"loaded {name} artifact does not match the signed profile"
            )

    def _validate_pallas_catalog(self) -> None:
        """Join loaded catalog bytes to the once-qualified immutable bridge.

        :meth:`VerifiedV2CatalogBindingV3.from_verified_v2` already parses the
        canonical V2 catalog and rebuilds every static Pallas tree before it
        mints its bridge.  Repeating that reconstruction whenever a signed
        profile is checked would turn catalog qualification into a validator
        CPU denial-of-service vector.  Serving-time validation therefore only
        rechecks the immutable bridge provenance and exact authenticated bytes.
        """

        # The artifact constructor hashes catalog bytes once. Recheck that
        # cached digest against the immutable bridge rather than rehashing a
        # potentially large local catalog during every profile/session check.
        catalog_digest = self._pallas_catalog_digest
        if (
            len(self.pallas_catalog_bytes) != self.static_manifest.pallas_catalog_size
            or type(catalog_digest) is not bytes
            or len(catalog_digest) != 32
            or catalog_digest != self.static_manifest.pallas_catalog_sha256
        ):
            raise ProofV3VerificationError(
                "loaded Pallas catalog does not match authenticated catalog metadata"
            )
        try:
            binding = self.verified_v2_catalog_binding
            binding.require_verified_v2_provenance()
            if (
                self.static_manifest != binding.static_manifest
                or catalog_digest != binding.static_manifest.pallas_catalog_sha256
            ):
                raise ProofV3VerificationError(
                    "loaded Pallas catalog is not the verified proof-v2 binding"
                )
            if (
                self.static_manifest.pallas_pcs_suite != V2_PALLAS_PCS_SUITE_V3
                or self.static_manifest.pallas_generator_version
                != V2_PALLAS_GENERATOR_VERSION_V3
                or self.static_artifact.pallas_pcs_suite != V2_PALLAS_PCS_SUITE_V3
                or self.static_artifact.pallas_generator_version
                != V2_PALLAS_GENERATOR_VERSION_V3
            ):
                raise ProofV3VerificationError(
                    "loaded Pallas catalog has an unsupported PCS configuration"
                )
        except ProofV3VerificationError:
            raise
        except Exception as exc:
            raise ProofV3VerificationError(
                "loaded Pallas catalog is malformed or does not authenticate static operations"
            ) from exc

    @staticmethod
    def _validate_static_table_relation_bindings(
        *,
        relation: object,
        static_tables: tuple[object, ...],
    ) -> None:
        """Join each loaded dynamic table to its exact signed relation source.

        The static-table artifact and the execution relation are separately
        authenticated documents.  Accepting both independently would let a
        future adapter substitute an unrelated table for an operation or
        parameter family.  The relation records the expected descriptor,
        geometry, and encoding for every table, while this loader proves that
        those records exactly cover the parsed artifact layout.
        """

        table_bindings = getattr(relation, "static_table_bindings", None)
        if (
            not isinstance(table_bindings, tuple)
            or not table_bindings
            or not all(
                isinstance(item, StaticTableBindingV3) for item in table_bindings
            )
        ):
            raise ProofV3VerificationError(
                "signed relation static table bindings are malformed"
            )
        try:
            by_id = {item.table_id: item for item in static_tables}
        except AttributeError as exc:
            raise ProofV3VerificationError(
                "loaded static table layout is malformed"
            ) from exc
        if len(by_id) != len(static_tables):
            raise ProofV3VerificationError(
                "loaded static table layout has duplicate table ids"
            )
        binding_table_ids = {item.table_id for item in table_bindings}
        if binding_table_ids != set(by_id) or len(binding_table_ids) != len(
            table_bindings
        ):
            raise ProofV3VerificationError(
                "signed relation table bindings do not exactly cover loaded static tables"
            )
        for binding in table_bindings:
            descriptor = by_id[binding.table_id]
            if (
                descriptor.digest() != binding.static_table_descriptor_digest
                or descriptor.logical_leaf_start != binding.logical_leaf_start
                or descriptor.logical_leaf_count != binding.logical_leaf_count
                or descriptor.page_start != binding.page_start
                or descriptor.page_count != binding.page_count
                or descriptor.element_encoding_id != binding.element_encoding_id
                or descriptor.scale_encoding_id != binding.scale_encoding_id
            ):
                raise ProofV3VerificationError(
                    "loaded static table does not match its signed relation binding"
                )

    def validate_profile(self, *, profile: ExecutionSecurityProfileV3) -> None:
        """Cross-check every loaded artifact before native proof dispatch."""

        if not isinstance(profile, ExecutionSecurityProfileV3):
            raise ProofV3VerificationError("profile has an unexpected type")
        if profile.digest() != self.execution_profile_digest:
            raise ProofV3VerificationError(
                "qualified artifacts are not registered for this exact profile"
            )
        if profile.static_manifest_digest != self.static_manifest.manifest_digest:
            raise ProofV3VerificationError(
                "qualified artifacts have an unexpected static manifest"
            )
        if profile.static_artifact_digest != self.static_artifact.digest():
            raise ProofV3VerificationError(
                "qualified artifacts have an unexpected static-weight artifact"
            )
        if (
            profile.quantization_semantics_id
            != self.static_manifest.quantization_semantics_id
        ):
            raise ProofV3VerificationError(
                "qualified artifacts have unexpected quantization semantics"
            )
        self._validate_pallas_catalog()
        self._require_digest(
            artifact=self.verifier_key_bytes,
            expected_digest=profile.verifier_key_digest,
            name="verifier key",
        )
        relation = profile.relation_spec
        self._require_digest(
            artifact=self.capture_abi_bytes,
            expected_digest=relation.capture_abi_digest,
            name="capture ABI",
        )
        try:
            from verallm.proof_v3.sidecar import NativeSidecarCaptureAbiV3

            capture_abi = NativeSidecarCaptureAbiV3.from_canonical_bytes(
                self.capture_abi_bytes
            )
            capture_abi.validate_profile(profile=profile)
        except ProofV3VerificationError:
            raise
        except Exception as exc:
            raise ProofV3VerificationError(
                "qualified native sidecar capture ABI validation failed"
            ) from exc
        self._require_digest(
            artifact=self.constraint_system_bytes,
            expected_digest=relation.constraint_system_digest,
            name="constraint system",
        )
        try:
            self._constraint_system.validate_relation(relation=relation)
        except ProofV3VerificationError:
            raise
        except Exception as exc:
            raise ProofV3VerificationError(
                "qualified constraint-system artifact does not match the signed relation"
            ) from exc
        try:
            self._constraint_program_bundle.validate_constraint_system(
                constraint_system=self._constraint_system,
                relation=relation,
            )
        except ProofV3VerificationError:
            raise
        except Exception as exc:
            raise ProofV3VerificationError(
                "qualified constraint-program bundle does not match the signed relation"
            ) from exc
        self._require_digest(
            artifact=self.tokenizer_binding_bytes,
            expected_digest=relation.tokenizer_binding_digest,
            name="tokenizer binding",
        )
        self._require_digest(
            artifact=self.sampler_abi_bytes,
            expected_digest=relation.sampler_abi_digest,
            name="sampler ABI",
        )
        self._require_digest(
            artifact=self.quantization_binding_bytes,
            expected_digest=relation.quantization_binding_digest,
            name="quantization binding",
        )
        try:
            quantization_binding = QuantizationBindingArtifactV3.from_canonical_bytes(
                self.quantization_binding_bytes
            )
            quantization_binding.validate_authenticated_context(
                relation_quantization_binding_digest=(
                    relation.quantization_binding_digest
                ),
                profile_quantization_semantics_id=profile.quantization_semantics_id,
                static_manifest_digest=self.static_manifest.manifest_digest,
                model_spec_identity_digest=(
                    self.static_manifest.model_spec_identity_digest
                ),
                model_quant_mode=self.static_manifest.model_quant_mode,
            )
        except ProofV3VerificationError:
            raise
        except Exception as exc:
            raise ProofV3VerificationError(
                "qualified quantization binding validation failed"
            ) from exc
        self._require_digest(
            artifact=self.recursive_accumulator_bytes,
            expected_digest=relation.recursive_accumulator_digest,
            name="recursive accumulator",
        )
        self._require_digest(
            artifact=self.static_table_layout_bytes,
            expected_digest=self.static_artifact.static_table_layout_digest,
            name="static table layout",
        )
        self._require_digest(
            artifact=self.kernel_abi_bytes,
            expected_digest=self.static_artifact.kernel_abi_digest,
            name="kernel ABI",
        )
        self._require_digest(
            artifact=self.dynamic_backend_bytes,
            expected_digest=self.static_artifact.dynamic_backend_digest,
            name="dynamic backend",
        )
        try:
            source_binding = StaticTableSourceBindingV3.from_canonical_bytes(
                self.static_source_binding_bytes
            )
            self.static_artifact.validate_source_binding(source_binding)
            self._require_digest(
                artifact=self.canonicalization_recipe_bytes,
                expected_digest=source_binding.canonicalization_recipe_digest,
                name="canonicalization recipe",
            )
            self._require_digest(
                artifact=self.source_file_index_bytes,
                expected_digest=source_binding.source_file_index_digest,
                name="source-file index",
            )
            self._require_digest(
                artifact=self.static_table_construction_bytes,
                expected_digest=source_binding.static_table_construction_digest,
                name="static-table construction",
            )
            static_tables = static_table_layout_from_canonical_bytes_v3(
                self.static_table_layout_bytes
            )
            static_context = StaticWeightArtifactContextV3(
                chain_id=self.static_manifest.chain_id,
                netuid=self.static_manifest.netuid,
                registry_address=self.static_manifest.registry_address,
                model_id=self.static_manifest.model_id,
                static_manifest_digest=self.static_manifest.manifest_digest,
                model_spec_identity_digest=self.static_manifest.model_spec_identity_digest,
                model_weight_merkle_root=self.static_manifest.model_weight_merkle_root,
                source_weight_file_hash=self.static_manifest.source_weight_file_hash,
                canonical_model_bytes_abi_id=source_binding.canonical_model_bytes_abi_id,
                canonical_model_bytes_digest=source_binding.canonical_model_bytes_digest,
                source_file_index_digest=source_binding.source_file_index_digest,
                canonicalization_recipe_digest=source_binding.canonicalization_recipe_digest,
                pallas_catalog_abi_id=self.static_manifest.pallas_catalog_abi_id,
                pallas_catalog_sha256=self.static_manifest.pallas_catalog_sha256,
                pallas_catalog_size=self.static_manifest.pallas_catalog_size,
                pallas_pcs_suite=self.static_manifest.pallas_pcs_suite,
                pallas_generator_version=self.static_manifest.pallas_generator_version,
                quantization_binding_digest=relation.quantization_binding_digest,
                kernel_abi_digest=hashlib.sha256(self.kernel_abi_bytes).digest(),
                dynamic_backend_abi_id=self.dynamic_backend_abi_id,
                dynamic_backend_digest=hashlib.sha256(
                    self.dynamic_backend_bytes
                ).digest(),
                static_table_root=source_binding.static_table_root,
                static_table_construction_abi_id=(
                    source_binding.static_table_construction_abi_id
                ),
                static_table_construction_digest=(
                    source_binding.static_table_construction_digest
                ),
                static_table_layout_digest=hashlib.sha256(
                    self.static_table_layout_bytes
                ).digest(),
                static_tables=static_tables,
            )
            self.static_artifact.validate_context(static_context)
            self._validate_static_table_relation_bindings(
                relation=relation,
                static_tables=static_tables,
            )
        except ProofV3VerificationError:
            raise
        except Exception as exc:
            raise ProofV3VerificationError(
                "qualified static-weight artifact validation failed"
            ) from exc
        validate_registered_operations_against_manifest_v3(
            relation.registered_operations,
            manifest_digest=self.static_manifest.manifest_digest,
            expected_manifest_digest=profile.static_manifest_digest,
            manifest_operations=self.static_manifest.operations,
        )
        expected_bindings = tuple(relation.static_bindings)
        if self.static_manifest.static_bindings != expected_bindings:
            raise ProofV3VerificationError(
                "signed relation bindings do not exactly cover the static manifest"
            )


__all__ = [
    "AuthenticatedStaticManifestV3",
    "MAX_QUALIFIED_ARTIFACT_BYTES_V3",
    "MODEL_SPEC_IDENTITY_DIGEST_DOMAIN_V3",
    "QualifiedExecutionArtifactsV3",
    "VerifiedProjectionManifestV3",
    "VerifiedV2CatalogBindingV3",
    "load_verified_projection_manifest_v3",
    "model_spec_identity_digest_v3",
]
