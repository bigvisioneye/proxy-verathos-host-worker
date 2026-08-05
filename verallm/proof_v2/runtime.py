"""Runtime loading for chain-authenticated proof-v2 manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from verallm.proof_v2.document import (
    SignedManifestDocument,
    load_signed_manifest_document,
    verify_manifest_document,
)
from verallm.proof_v2.manifest import StaticWeightCommitmentManifest
from verallm.proof_v2.layout import (
    ProofV2LayoutError,
    require_independent_execution_transitions_v2,
    validate_qwen_hybrid_execution_manifest_profile,
)


BUNDLED_ARTIFACT_DIRECTORY = "proof_v2_artifacts"
BUNDLED_MANIFEST_SUFFIX = ".manifest.json"
BUNDLED_CATALOG_SUFFIX = ".catalog.bin"


class ProofV2RuntimeManifestError(RuntimeError):
    """A configured runtime manifest is missing or is not chain-authenticated."""


_VERIFIED_MANIFEST_LOADER_TOKEN = object()


@dataclass(frozen=True, slots=True)
class _VerifiedManifestLoaderProvenance:
    """Private immutable record minted by the completed runtime loader.

    The object identity binds the returned result to the exact parsed document,
    so a normal ``dataclasses.replace`` cannot retarget a verified result while
    retaining its provenance marker. This is a trusted-process API guard, not
    a cryptographic defense against code that can already alter validator
    process memory.
    """

    document: SignedManifestDocument
    manifest_digest: bytes
    _token: object


@dataclass(frozen=True)
class VerifiedProofV2Manifest:
    document: SignedManifestDocument
    recovered_signers: tuple[str, ...]
    source_path: Path
    _provenance: _VerifiedManifestLoaderProvenance | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def manifest(self) -> StaticWeightCommitmentManifest:
        return self.document.manifest

    @classmethod
    def _from_verified_loader(
        cls,
        *,
        document: SignedManifestDocument,
        recovered_signers: tuple[str, ...],
        source_path: Path,
        _factory_token: object | None = None,
    ) -> "VerifiedProofV2Manifest":
        """Create the opaque result of a completed V2 loader verification.

        The leading underscore and private factory token are intentional:
        callers that need a verified manifest must use
        :func:`load_verified_proof_v2_manifest`, which owns the chain,
        authority, and signature checks. The provenance fields prevent another
        layer from mistaking a hand-constructed dataclass for that loader
        result.
        """

        if _factory_token is not _VERIFIED_MANIFEST_LOADER_TOKEN:
            raise ProofV2RuntimeManifestError(
                "verified proof-v2 manifests must be created by the runtime loader"
            )
        result = cls(
            document=document,
            recovered_signers=recovered_signers,
            source_path=source_path,
        )
        object.__setattr__(
            result,
            "_provenance",
            _VerifiedManifestLoaderProvenance(
                document=document,
                manifest_digest=document.manifest.digest(),
                _token=_VERIFIED_MANIFEST_LOADER_TOKEN,
            ),
        )
        return result

    def require_loader_provenance(self) -> None:
        """Reject records not minted by the chain-authenticated V2 loader."""

        provenance = self._provenance
        if (
            not isinstance(provenance, _VerifiedManifestLoaderProvenance)
            or provenance._token is not _VERIFIED_MANIFEST_LOADER_TOKEN
            or provenance.document is not self.document
        ):
            raise ProofV2RuntimeManifestError(
                "proof-v2 manifest did not originate from the verified loader"
            )
        if (
            type(provenance.manifest_digest) is not bytes
            or len(provenance.manifest_digest) != 32
            or provenance.manifest_digest != self.document.manifest.digest()
        ):
            raise ProofV2RuntimeManifestError(
                "proof-v2 verified manifest provenance no longer matches its document"
            )


def bundled_proof_v2_artifact_directory(
    project_root: str | Path | None = None,
    *,
    chain_id: int | None = None,
    netuid: int | None = None,
) -> Path:
    """Return the release-tree directory reserved for authenticated artifacts."""

    if (chain_id is None) != (netuid is None):
        raise ProofV2RuntimeManifestError(
            "bundled proof-v2 artifact context requires chain_id and netuid together"
        )
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    directory = root / BUNDLED_ARTIFACT_DIRECTORY
    if chain_id is not None and netuid is not None:
        if (
            type(chain_id) is not int
            or chain_id <= 0
            or type(netuid) is not int
            or netuid < 0
        ):
            raise ProofV2RuntimeManifestError(
                "bundled proof-v2 artifact context is invalid"
            )
        directory /= f"{chain_id}-{netuid}"
    return directory


def bundled_proof_v2_manifest_paths(
    project_root: str | Path | None = None,
    *,
    chain_id: int | None = None,
    netuid: int | None = None,
) -> tuple[Path, ...]:
    """Discover deterministic release-bundled manifest documents.

    Discovery is only a convenience for clean auto-updated checkouts. Every
    returned document is still authenticated against live chain state by the
    normal runtime loader before it can affect verification.
    """

    directory = bundled_proof_v2_artifact_directory(
        project_root,
        chain_id=chain_id,
        netuid=netuid,
    )
    if not directory.is_dir():
        return ()
    try:
        candidates = tuple(
            sorted(
                path.resolve()
                for path in directory.iterdir()
                if path.name.endswith(BUNDLED_MANIFEST_SUFFIX) and path.is_file()
            )
        )
    except OSError as exc:
        raise ProofV2RuntimeManifestError(
            f"cannot inspect bundled proof-v2 artifacts: {directory}"
        ) from exc
    return candidates


def bundled_proof_v2_catalog_path(
    manifest_digest: bytes,
    project_root: str | Path | None = None,
    *,
    chain_id: int | None = None,
    netuid: int | None = None,
) -> Path:
    """Derive the exact bundled catalog filename for one verified manifest."""

    if type(manifest_digest) is not bytes or len(manifest_digest) != 32:
        raise ProofV2RuntimeManifestError(
            "proof-v2 manifest digest must be exactly 32 bytes"
        )
    return bundled_proof_v2_artifact_directory(
        project_root,
        chain_id=chain_id,
        netuid=netuid,
    ) / (manifest_digest.hex() + BUNDLED_CATALOG_SUFFIX)


def load_verified_proof_v2_manifest(
    path: str | Path,
    *,
    chain_config,
    model_registry_client,
    expected_model_id: str | None = None,
) -> VerifiedProofV2Manifest:
    """Load one document and verify it against current exact chain state."""

    source = Path(path).expanduser().resolve()
    try:
        document = load_signed_manifest_document(source)
        model_id = expected_model_id or document.manifest.model_spec.model_id
        if document.manifest.model_spec.model_id != model_id:
            raise ProofV2RuntimeManifestError(
                "proof-v2 manifest model_id does not match the requested model"
            )
        if not hasattr(model_registry_client, "get_on_chain_model_spec"):
            raise ProofV2RuntimeManifestError(
                "proof-v2 manifests require an exact on-chain ModelSpec reader"
            )
        on_chain_spec = model_registry_client.get_on_chain_model_spec(model_id)
        if on_chain_spec is None:
            raise ProofV2RuntimeManifestError(
                f"proof-v2 manifest model is not registered: {model_id}"
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
        try:
            validate_qwen_hybrid_execution_manifest_profile(document.manifest)
            require_independent_execution_transitions_v2(document.manifest)
        except ProofV2LayoutError as exc:
            raise ProofV2RuntimeManifestError(
                "proof-v2 manifest is outside the independently verifiable runtime profile"
            ) from exc
    except ProofV2RuntimeManifestError:
        raise
    except Exception as exc:
        raise ProofV2RuntimeManifestError(
            f"proof-v2 manifest authentication failed: {source}"
        ) from exc
    return VerifiedProofV2Manifest._from_verified_loader(
        document=document,
        recovered_signers=recovered,
        source_path=source,
        _factory_token=_VERIFIED_MANIFEST_LOADER_TOKEN,
    )


def load_verified_proof_v2_manifests(
    paths: Iterable[str | Path],
    *,
    chain_config,
    model_registry_client,
) -> Mapping[str, VerifiedProofV2Manifest]:
    """Load an exact unique model-to-manifest mapping."""

    verified = {}
    for path in tuple(paths):
        item = load_verified_proof_v2_manifest(
            path,
            chain_config=chain_config,
            model_registry_client=model_registry_client,
        )
        model_id = item.manifest.model_spec.model_id
        if model_id in verified:
            raise ProofV2RuntimeManifestError(
                f"duplicate proof-v2 manifest for model: {model_id}"
            )
        verified[model_id] = item
    if not verified:
        raise ProofV2RuntimeManifestError("no proof-v2 manifests were configured")
    return verified


__all__ = [
    "BUNDLED_ARTIFACT_DIRECTORY",
    "BUNDLED_CATALOG_SUFFIX",
    "BUNDLED_MANIFEST_SUFFIX",
    "ProofV2RuntimeManifestError",
    "VerifiedProofV2Manifest",
    "bundled_proof_v2_artifact_directory",
    "bundled_proof_v2_catalog_path",
    "bundled_proof_v2_manifest_paths",
    "load_verified_proof_v2_manifest",
    "load_verified_proof_v2_manifests",
]
