"""Content-addressed remote storage for proof-v2 static artifacts.

The remote index is discovery metadata, not a trust anchor.  Every downloaded
manifest is still authenticated against the current ModelRegistry state and
manifest authority, while the miner validates every catalog commitment against
the signed manifest.  A compromised artifact host can therefore deny service
but cannot authorize different model weights or proof parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence
import unicodedata
from urllib.parse import urlsplit, urlunsplit

from verallm.proof_v2.catalog import (
    MAX_CATALOG_BYTES,
    WeightCommitmentCatalogError,
    WeightCommitmentCatalogV2,
)
from verallm.proof_v2.document import MAX_MANIFEST_DOCUMENT_BYTES
from verallm.proof_v2.runtime import (
    ProofV2RuntimeManifestError,
    VerifiedProofV2Manifest,
    load_verified_proof_v2_manifest,
)


ARTIFACT_INDEX_SCHEMA = "verathos-proof-v2-artifact-index-v1"
ARTIFACT_INDEX_FILENAME = "index.json"
ARTIFACT_BASE_URLS_ENV = "VERATHOS_PROOF_V2_ARTIFACT_BASE_URLS"
ARTIFACT_CACHE_DIR_ENV = "VERATHOS_PROOF_V2_ARTIFACT_CACHE_DIR"
MAX_ARTIFACT_INDEX_BYTES = 8 << 20
MAX_ARTIFACT_INDEX_MODELS = 10_000
DEFAULT_ARTIFACT_TIMEOUT_SECONDS = 300.0

_ADDRESS_RE = re.compile(r"^[0-9a-f]{40}$")


class ProofV2ArtifactStoreError(RuntimeError):
    """Remote proof-v2 artifact discovery or validation failed."""


def _canonical_hex(value: object, size: int, name: str) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) != size * 2
        or value.lower() != value
    ):
        raise ProofV2ArtifactStoreError(
            f"{name} must be canonical lowercase hexadecimal"
        )
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ProofV2ArtifactStoreError(f"{name} must be hexadecimal") from exc
    if len(decoded) != size:
        raise ProofV2ArtifactStoreError(f"{name} must be exactly {size} bytes")
    return decoded


def _positive_size(value: object, maximum: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > maximum
    ):
        raise ProofV2ArtifactStoreError(f"{name} is out of range")
    return value


def _canonical_model_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ProofV2ArtifactStoreError(
            "artifact model_id must be a non-empty string"
        )
    if value != unicodedata.normalize("NFC", value) or "\x00" in value:
        raise ProofV2ArtifactStoreError("artifact model_id is not canonical")
    if len(value.encode("utf-8")) > 4096:
        raise ProofV2ArtifactStoreError("artifact model_id is too long")
    return value


def _strict_object(value: object, fields: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ProofV2ArtifactStoreError(
            f"{name} fields do not match the canonical schema"
        )
    return value


def _unique_object_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProofV2ArtifactStoreError(
                f"artifact index contains duplicate field: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ProofV2ArtifactStoreError(
        f"artifact index contains a non-finite number: {value}"
    )


@dataclass(frozen=True, slots=True)
class ProofV2ArtifactIndexEntry:
    """Content-addressed manifest and catalog metadata for one exact model."""

    model_id: str
    manifest_digest: bytes
    manifest_sha256: bytes
    manifest_size: int
    catalog_sha256: bytes
    catalog_size: int

    def __post_init__(self) -> None:
        _canonical_model_id(self.model_id)
        for name, value in (
            ("manifest_digest", self.manifest_digest),
            ("manifest_sha256", self.manifest_sha256),
            ("catalog_sha256", self.catalog_sha256),
        ):
            if type(value) is not bytes or len(value) != 32:
                raise ProofV2ArtifactStoreError(f"{name} must be exactly 32 bytes")
        _positive_size(
            self.manifest_size,
            MAX_MANIFEST_DOCUMENT_BYTES,
            "manifest_size",
        )
        _positive_size(self.catalog_size, MAX_CATALOG_BYTES, "catalog_size")

    @property
    def manifest_filename(self) -> str:
        return f"{self.manifest_sha256.hex()}.manifest.json"

    @property
    def catalog_filename(self) -> str:
        return f"{self.catalog_sha256.hex()}.catalog.bin"

    def to_dict(self) -> dict[str, object]:
        return {
            "catalog_sha256": self.catalog_sha256.hex(),
            "catalog_size": self.catalog_size,
            "manifest_digest": self.manifest_digest.hex(),
            "manifest_sha256": self.manifest_sha256.hex(),
            "manifest_size": self.manifest_size,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ProofV2ArtifactIndexEntry":
        item = _strict_object(
            value,
            {
                "catalog_sha256",
                "catalog_size",
                "manifest_digest",
                "manifest_sha256",
                "manifest_size",
                "model_id",
            },
            "artifact index entry",
        )
        return cls(
            model_id=_canonical_model_id(item["model_id"]),
            manifest_digest=_canonical_hex(
                item["manifest_digest"], 32, "manifest_digest"
            ),
            manifest_sha256=_canonical_hex(
                item["manifest_sha256"], 32, "manifest_sha256"
            ),
            manifest_size=_positive_size(
                item["manifest_size"],
                MAX_MANIFEST_DOCUMENT_BYTES,
                "manifest_size",
            ),
            catalog_sha256=_canonical_hex(
                item["catalog_sha256"], 32, "catalog_sha256"
            ),
            catalog_size=_positive_size(
                item["catalog_size"], MAX_CATALOG_BYTES, "catalog_size"
            ),
        )


@dataclass(frozen=True, slots=True)
class ProofV2ArtifactIndex:
    """Canonical network-scoped index for content-addressed proof artifacts."""

    chain_id: int
    netuid: int
    registry_address: bytes
    models: tuple[ProofV2ArtifactIndexEntry, ...]
    schema: str = ARTIFACT_INDEX_SCHEMA
    _by_model: Mapping[str, ProofV2ArtifactIndexEntry] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.schema != ARTIFACT_INDEX_SCHEMA:
            raise ProofV2ArtifactStoreError("artifact index schema is unsupported")
        if (
            isinstance(self.chain_id, bool)
            or not isinstance(self.chain_id, int)
            or self.chain_id <= 0
        ):
            raise ProofV2ArtifactStoreError("artifact index chain_id is invalid")
        if (
            isinstance(self.netuid, bool)
            or not isinstance(self.netuid, int)
            or self.netuid < 0
        ):
            raise ProofV2ArtifactStoreError("artifact index netuid is invalid")
        if (
            type(self.registry_address) is not bytes
            or len(self.registry_address) != 20
            or self.registry_address == b"\x00" * 20
        ):
            raise ProofV2ArtifactStoreError(
                "artifact index registry_address is invalid"
            )
        models = tuple(self.models)
        if not models or len(models) > MAX_ARTIFACT_INDEX_MODELS:
            raise ProofV2ArtifactStoreError(
                "artifact index model count is out of range"
            )
        if not all(isinstance(item, ProofV2ArtifactIndexEntry) for item in models):
            raise ProofV2ArtifactStoreError(
                "artifact index contains an unexpected model entry"
            )
        model_ids = [item.model_id for item in models]
        if model_ids != sorted(model_ids) or len(model_ids) != len(set(model_ids)):
            raise ProofV2ArtifactStoreError(
                "artifact index models must be sorted and unique"
            )
        object.__setattr__(self, "models", models)
        object.__setattr__(
            self,
            "_by_model",
            MappingProxyType({item.model_id: item for item in models}),
        )

    @property
    def by_model(self) -> Mapping[str, ProofV2ArtifactIndexEntry]:
        return self._by_model

    def to_dict(self) -> dict[str, object]:
        return {
            "chain_id": self.chain_id,
            "models": [item.to_dict() for item in self.models],
            "netuid": self.netuid,
            "registry_address": self.registry_address.hex(),
            "schema": self.schema,
        }

    def canonical_json_bytes(self) -> bytes:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        if len(encoded) > MAX_ARTIFACT_INDEX_BYTES:
            raise ProofV2ArtifactStoreError("artifact index exceeds the size limit")
        return encoded

    def validate_context(self, chain_config) -> None:
        try:
            expected_address = bytes.fromhex(
                str(chain_config.model_registry_address).removeprefix("0x")
            )
        except ValueError as exc:
            raise ProofV2ArtifactStoreError(
                "chain model registry address is invalid"
            ) from exc
        if (
            self.chain_id != chain_config.chain_id
            or self.netuid != chain_config.netuid
            or self.registry_address != expected_address
        ):
            raise ProofV2ArtifactStoreError(
                "artifact index does not match the configured chain context"
            )

    @classmethod
    def from_json_bytes(cls, encoded: bytes) -> "ProofV2ArtifactIndex":
        if (
            type(encoded) is not bytes
            or not encoded
            or len(encoded) > MAX_ARTIFACT_INDEX_BYTES
        ):
            raise ProofV2ArtifactStoreError(
                "artifact index length is out of range"
            )
        try:
            value = json.loads(
                encoded.decode("ascii"),
                object_pairs_hook=_unique_object_pairs,
                parse_constant=_reject_json_constant,
            )
        except ProofV2ArtifactStoreError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProofV2ArtifactStoreError(
                "artifact index is not valid JSON"
            ) from exc
        document = _strict_object(
            value,
            {"chain_id", "models", "netuid", "registry_address", "schema"},
            "artifact index",
        )
        raw_models = document["models"]
        if not isinstance(raw_models, list):
            raise ProofV2ArtifactStoreError("artifact index models must be a list")
        if not raw_models or len(raw_models) > MAX_ARTIFACT_INDEX_MODELS:
            raise ProofV2ArtifactStoreError(
                "artifact index model count is out of range"
            )
        registry_address = document["registry_address"]
        if (
            not isinstance(registry_address, str)
            or _ADDRESS_RE.fullmatch(registry_address) is None
        ):
            raise ProofV2ArtifactStoreError(
                "artifact index registry_address is not canonical"
            )
        result = cls(
            chain_id=document["chain_id"],
            netuid=document["netuid"],
            registry_address=bytes.fromhex(registry_address),
            models=tuple(
                ProofV2ArtifactIndexEntry.from_dict(item) for item in raw_models
            ),
            schema=document["schema"],
        )
        if result.canonical_json_bytes() != encoded:
            raise ProofV2ArtifactStoreError(
                "artifact index is not canonical JSON"
            )
        return result


@dataclass(frozen=True)
class ResolvedProofV2Artifacts:
    """One authenticated manifest plus its downloaded canonical catalog."""

    verified_manifest: VerifiedProofV2Manifest
    weight_catalog: WeightCommitmentCatalogV2
    manifest_path: Path
    catalog_path: Path
    index_source_url: str


@dataclass(frozen=True)
class RemoteProofV2ManifestResolution:
    """Best-effort multi-model manifest resolution for validators and proxies."""

    manifests: Mapping[str, VerifiedProofV2Manifest]
    missing_model_ids: tuple[str, ...]
    failures: Mapping[str, str]
    index_source_url: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "manifests",
            MappingProxyType(dict(self.manifests)),
        )
        object.__setattr__(
            self,
            "failures",
            MappingProxyType(dict(self.failures)),
        )


def _normalize_base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProofV2ArtifactStoreError("artifact base URL must be non-empty")
    parsed = urlsplit(value.strip())
    if parsed.scheme not in ("https", "http") or not parsed.hostname:
        raise ProofV2ArtifactStoreError(
            "artifact base URL must use HTTP or HTTPS"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ProofV2ArtifactStoreError(
            "artifact base URL must not contain credentials"
        )
    if parsed.query or parsed.fragment:
        raise ProofV2ArtifactStoreError(
            "artifact base URL must not contain a query or fragment"
        )
    if parsed.scheme == "http" and parsed.hostname not in (
        "localhost",
        "127.0.0.1",
        "::1",
    ):
        raise ProofV2ArtifactStoreError(
            "non-loopback artifact base URLs must use HTTPS"
        )
    normalized_path = parsed.path.rstrip("/")
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            normalized_path,
            "",
            "",
        )
    )


def normalize_proof_v2_artifact_base_urls(
    values: Iterable[str],
) -> tuple[str, ...]:
    """Validate, canonicalize, and de-duplicate artifact source URLs."""

    normalized = []
    seen = set()
    for value in tuple(values):
        source = _normalize_base_url(value)
        if source not in seen:
            normalized.append(source)
            seen.add(source)
    if not normalized:
        raise ProofV2ArtifactStoreError("no proof-v2 artifact base URLs configured")
    return tuple(normalized)


def configured_proof_v2_artifact_base_urls(
    cli_values: Sequence[str] | None,
    *,
    default_values: Sequence[str] = (),
) -> tuple[str, ...]:
    """Resolve CLI, environment, then release chain-config source URLs."""

    if cli_values is not None:
        values = tuple(item for item in cli_values if item)
        return normalize_proof_v2_artifact_base_urls(values) if values else ()
    raw = os.environ.get(ARTIFACT_BASE_URLS_ENV, "")
    values = tuple(
        item.strip()
        for line in raw.splitlines()
        for item in line.split(",")
        if item.strip()
    )
    if values:
        return normalize_proof_v2_artifact_base_urls(values)
    defaults = tuple(item for item in default_values if item)
    return normalize_proof_v2_artifact_base_urls(defaults) if defaults else ()


def proof_v2_artifact_cache_directory(
    configured_path: str | Path | None,
    *,
    chain_id: int,
    netuid: int,
) -> Path:
    """Return the network-scoped local cache directory."""

    raw = str(configured_path or os.environ.get(ARTIFACT_CACHE_DIR_ENV, "")).strip()
    if raw:
        root = Path(raw).expanduser()
    else:
        data_dir = os.environ.get("VERALLM_DATA_DIR", "").strip()
        root = (
            Path(data_dir).expanduser()
            if data_dir
            else Path.home() / ".verathos"
        ) / "proof_v2_artifacts"
    return root.absolute() / f"{chain_id}-{netuid}"


def _url(base_url: str, filename: str) -> str:
    return f"{base_url}/{filename}"


def _sha256_file(path: Path, *, maximum: int) -> tuple[bytes, int]:
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum:
                    raise ProofV2ArtifactStoreError(
                        "cached artifact exceeds the size limit"
                    )
                digest.update(chunk)
    except ProofV2ArtifactStoreError:
        raise
    except OSError as exc:
        raise ProofV2ArtifactStoreError(
            f"cannot read cached proof-v2 artifact: {path}"
        ) from exc
    return digest.digest(), total


def _atomic_write(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _read_bounded_file(path: Path, *, maximum: int, name: str) -> bytes:
    try:
        size = path.stat().st_size
        if size <= 0 or size > maximum:
            raise ProofV2ArtifactStoreError(f"{name} size is out of range")
        with path.open("rb") as handle:
            encoded = handle.read(maximum + 1)
        if len(encoded) != size or len(encoded) > maximum:
            raise ProofV2ArtifactStoreError(
                f"{name} size changed while loading"
            )
        return encoded
    except ProofV2ArtifactStoreError:
        raise
    except OSError as exc:
        raise ProofV2ArtifactStoreError(f"cannot read {name}") from exc


def _fetch_small_bytes(url: str, *, maximum: int, timeout_seconds: float) -> bytes:
    try:
        import httpx

        timeout = httpx.Timeout(timeout_seconds, connect=min(10.0, timeout_seconds))
        with httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            headers={"Accept-Encoding": "identity"},
        ) as client:
            with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise ProofV2ArtifactStoreError(
                        f"artifact request returned HTTP {response.status_code}: {url}"
                    )
                content_encoding = response.headers.get(
                    "content-encoding", "identity"
                ).lower()
                if content_encoding not in ("", "identity"):
                    raise ProofV2ArtifactStoreError(
                        "artifact server returned an encoded response"
                    )
                raw_length = response.headers.get("content-length")
                content_length = None
                if raw_length is not None:
                    try:
                        content_length = int(raw_length)
                    except ValueError as exc:
                        raise ProofV2ArtifactStoreError(
                            "artifact Content-Length is invalid"
                        ) from exc
                    if content_length <= 0 or content_length > maximum:
                        raise ProofV2ArtifactStoreError(
                            "artifact Content-Length is out of range"
                        )
                encoded = bytearray()
                for chunk in response.iter_raw():
                    encoded.extend(chunk)
                    if len(encoded) > maximum:
                        raise ProofV2ArtifactStoreError(
                            "artifact response exceeds the size limit"
                        )
                if (
                    content_length is not None
                    and len(encoded) != content_length
                ):
                    raise ProofV2ArtifactStoreError(
                        "artifact response does not match Content-Length"
                    )
    except ProofV2ArtifactStoreError:
        raise
    except Exception as exc:
        raise ProofV2ArtifactStoreError(
            f"artifact request failed: {url}"
        ) from exc
    if not encoded:
        raise ProofV2ArtifactStoreError("artifact response is empty")
    return bytes(encoded)


def _download_object(
    url: str,
    destination: Path,
    *,
    expected_sha256: bytes,
    expected_size: int,
    maximum: int,
    timeout_seconds: float,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    total = 0
    try:
        import httpx

        timeout = httpx.Timeout(timeout_seconds, connect=min(10.0, timeout_seconds))
        with httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            headers={"Accept-Encoding": "identity"},
        ) as client:
            with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise ProofV2ArtifactStoreError(
                        f"artifact request returned HTTP {response.status_code}: {url}"
                    )
                content_encoding = response.headers.get(
                    "content-encoding", "identity"
                ).lower()
                if content_encoding not in ("", "identity"):
                    raise ProofV2ArtifactStoreError(
                        "artifact server returned an encoded response"
                    )
                raw_length = response.headers.get("content-length")
                if raw_length is not None:
                    try:
                        content_length = int(raw_length)
                    except ValueError as exc:
                        raise ProofV2ArtifactStoreError(
                            "artifact Content-Length is invalid"
                        ) from exc
                    if content_length != expected_size:
                        raise ProofV2ArtifactStoreError(
                            "artifact Content-Length does not match the index"
                        )
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb", closefd=True) as handle:
                    descriptor = -1
                    for chunk in response.iter_raw():
                        total += len(chunk)
                        if total > expected_size or total > maximum:
                            raise ProofV2ArtifactStoreError(
                                "artifact response exceeds the indexed size"
                            )
                        digest.update(chunk)
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
        if total != expected_size:
            raise ProofV2ArtifactStoreError(
                "artifact response size does not match the index"
            )
        if digest.digest() != expected_sha256:
            raise ProofV2ArtifactStoreError(
                "artifact SHA-256 does not match the index"
            )
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
        return destination
    except ProofV2ArtifactStoreError:
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise ProofV2ArtifactStoreError(
            f"artifact request failed: {url}"
        ) from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _cached_or_downloaded_object(
    *,
    base_urls: tuple[str, ...],
    cache_directory: Path,
    filename: str,
    expected_sha256: bytes,
    expected_size: int,
    maximum: int,
    timeout_seconds: float,
) -> Path:
    destination = (
        cache_directory
        / "objects"
        / expected_sha256.hex()[:2]
        / filename
    )
    if destination.is_file():
        try:
            digest, size = _sha256_file(destination, maximum=maximum)
            if digest == expected_sha256 and size == expected_size:
                return destination
        except ProofV2ArtifactStoreError:
            pass

    failures = []
    for base_url in base_urls:
        url = _url(base_url, filename)
        try:
            return _download_object(
                url,
                destination,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
                maximum=maximum,
                timeout_seconds=timeout_seconds,
            )
        except ProofV2ArtifactStoreError as exc:
            failures.append(str(exc))
    raise ProofV2ArtifactStoreError(
        "proof-v2 artifact is unavailable from all configured sources: "
        + "; ".join(failures)
    )


def _load_remote_index(
    base_urls: tuple[str, ...],
    *,
    cache_directory: Path,
    chain_config,
    timeout_seconds: float,
) -> tuple[ProofV2ArtifactIndex, str]:
    failures = []
    for base_url in base_urls:
        index_url = _url(base_url, ARTIFACT_INDEX_FILENAME)
        cache_path = (
            cache_directory
            / "indexes"
            / f"{hashlib.sha256(base_url.encode('utf-8')).hexdigest()}.json"
        )
        try:
            encoded = _fetch_small_bytes(
                index_url,
                maximum=MAX_ARTIFACT_INDEX_BYTES,
                timeout_seconds=timeout_seconds,
            )
            index = ProofV2ArtifactIndex.from_json_bytes(encoded)
            index.validate_context(chain_config)
            _atomic_write(cache_path, encoded)
            return index, index_url
        except ProofV2ArtifactStoreError as remote_exc:
            try:
                encoded = _read_bounded_file(
                    cache_path,
                    maximum=MAX_ARTIFACT_INDEX_BYTES,
                    name="cached proof-v2 artifact index",
                )
                index = ProofV2ArtifactIndex.from_json_bytes(encoded)
                index.validate_context(chain_config)
                return index, f"{index_url} (cached)"
            except Exception:
                failures.append(str(remote_exc))
    raise ProofV2ArtifactStoreError(
        "no valid proof-v2 artifact index is available: " + "; ".join(failures)
    )


def resolve_remote_proof_v2_manifests(
    base_urls: Iterable[str],
    *,
    chain_config,
    model_registry_client,
    cache_directory: str | Path | None = None,
    model_ids: Iterable[str] | None = None,
    timeout_seconds: float = DEFAULT_ARTIFACT_TIMEOUT_SECONDS,
) -> RemoteProofV2ManifestResolution:
    """Resolve and authenticate available manifests for registered models.

    Failures are isolated per model.  Callers must leave v2 verification
    unconfigured for missing or failed models, which preserves fail-closed
    proof handling without taking unrelated models offline.
    """

    sources = normalize_proof_v2_artifact_base_urls(base_urls)
    cache = proof_v2_artifact_cache_directory(
        cache_directory,
        chain_id=chain_config.chain_id,
        netuid=chain_config.netuid,
    )
    index, source_url = _load_remote_index(
        sources,
        cache_directory=cache,
        chain_config=chain_config,
        timeout_seconds=timeout_seconds,
    )
    if model_ids is None:
        try:
            requested = tuple(model_registry_client.get_model_list())
        except Exception as exc:
            raise ProofV2ArtifactStoreError(
                "cannot read the registered model list for artifact resolution"
            ) from exc
    else:
        requested = tuple(model_ids)
    try:
        requested = tuple(_canonical_model_id(model_id) for model_id in requested)
    except ProofV2ArtifactStoreError as exc:
        raise ProofV2ArtifactStoreError(
            "requested proof-v2 model IDs are not canonical"
        ) from exc
    requested = tuple(sorted(set(requested)))

    manifests: dict[str, VerifiedProofV2Manifest] = {}
    missing = []
    failures = {}
    for model_id in requested:
        entry = index.by_model.get(model_id)
        if entry is None:
            missing.append(model_id)
            continue
        try:
            manifest_path = _cached_or_downloaded_object(
                base_urls=sources,
                cache_directory=cache,
                filename=entry.manifest_filename,
                expected_sha256=entry.manifest_sha256,
                expected_size=entry.manifest_size,
                maximum=MAX_MANIFEST_DOCUMENT_BYTES,
                timeout_seconds=timeout_seconds,
            )
            verified = load_verified_proof_v2_manifest(
                manifest_path,
                chain_config=chain_config,
                model_registry_client=model_registry_client,
                expected_model_id=model_id,
            )
            if verified.manifest.digest() != entry.manifest_digest:
                raise ProofV2ArtifactStoreError(
                    "manifest digest does not match the artifact index"
                )
            manifests[model_id] = verified
        except (ProofV2ArtifactStoreError, ProofV2RuntimeManifestError) as exc:
            failures[model_id] = str(exc)
    return RemoteProofV2ManifestResolution(
        manifests=manifests,
        missing_model_ids=tuple(missing),
        failures=failures,
        index_source_url=source_url,
    )


def resolve_remote_proof_v2_artifacts(
    model_id: str,
    base_urls: Iterable[str],
    *,
    chain_config,
    model_registry_client,
    cache_directory: str | Path | None = None,
    timeout_seconds: float = DEFAULT_ARTIFACT_TIMEOUT_SECONDS,
) -> ResolvedProofV2Artifacts:
    """Resolve one miner model's authenticated manifest and exact catalog."""

    sources = normalize_proof_v2_artifact_base_urls(base_urls)
    cache = proof_v2_artifact_cache_directory(
        cache_directory,
        chain_id=chain_config.chain_id,
        netuid=chain_config.netuid,
    )
    index, source_url = _load_remote_index(
        sources,
        cache_directory=cache,
        chain_config=chain_config,
        timeout_seconds=timeout_seconds,
    )
    entry = index.by_model.get(model_id)
    if entry is None:
        raise ProofV2ArtifactStoreError(
            f"no remote proof-v2 artifacts are published for model: {model_id}"
        )
    manifest_path = _cached_or_downloaded_object(
        base_urls=sources,
        cache_directory=cache,
        filename=entry.manifest_filename,
        expected_sha256=entry.manifest_sha256,
        expected_size=entry.manifest_size,
        maximum=MAX_MANIFEST_DOCUMENT_BYTES,
        timeout_seconds=timeout_seconds,
    )
    verified = load_verified_proof_v2_manifest(
        manifest_path,
        chain_config=chain_config,
        model_registry_client=model_registry_client,
        expected_model_id=model_id,
    )
    if verified.manifest.digest() != entry.manifest_digest:
        raise ProofV2ArtifactStoreError(
            "manifest digest does not match the artifact index"
        )
    catalog_path = _cached_or_downloaded_object(
        base_urls=sources,
        cache_directory=cache,
        filename=entry.catalog_filename,
        expected_sha256=entry.catalog_sha256,
        expected_size=entry.catalog_size,
        maximum=MAX_CATALOG_BYTES,
        timeout_seconds=timeout_seconds,
    )
    try:
        catalog = WeightCommitmentCatalogV2.load(catalog_path)
    except (OSError, WeightCommitmentCatalogError) as exc:
        raise ProofV2ArtifactStoreError(
            "downloaded proof-v2 catalog is malformed"
        ) from exc
    if catalog.manifest_digest != verified.manifest.digest():
        raise ProofV2ArtifactStoreError(
            "downloaded proof-v2 catalog does not match the manifest"
        )
    return ResolvedProofV2Artifacts(
        verified_manifest=verified,
        weight_catalog=catalog,
        manifest_path=manifest_path,
        catalog_path=catalog_path,
        index_source_url=source_url,
    )


def artifact_index_entry_from_files(
    manifest_path: str | Path,
    catalog_path: str | Path,
) -> ProofV2ArtifactIndexEntry:
    """Build one publishing-index entry from canonical local artifacts."""

    from verallm.proof_v2.document import SignedManifestDocument

    manifest_source = Path(manifest_path).expanduser().resolve()
    catalog_source = Path(catalog_path).expanduser().resolve()
    manifest_bytes = _read_bounded_file(
        manifest_source,
        maximum=MAX_MANIFEST_DOCUMENT_BYTES,
        name="proof-v2 release manifest",
    )
    catalog_bytes = _read_bounded_file(
        catalog_source,
        maximum=MAX_CATALOG_BYTES,
        name="proof-v2 release catalog",
    )
    document = SignedManifestDocument.from_json_bytes(manifest_bytes)
    try:
        catalog = WeightCommitmentCatalogV2.from_canonical_bytes(catalog_bytes)
    except WeightCommitmentCatalogError as exc:
        raise ProofV2ArtifactStoreError(
            "proof-v2 release catalog is malformed"
        ) from exc
    manifest_digest = document.manifest.digest()
    if catalog.manifest_digest != manifest_digest:
        raise ProofV2ArtifactStoreError(
            "proof-v2 release catalog does not match the manifest"
        )
    return ProofV2ArtifactIndexEntry(
        model_id=document.manifest.model_spec.model_id,
        manifest_digest=manifest_digest,
        manifest_sha256=hashlib.sha256(manifest_bytes).digest(),
        manifest_size=len(manifest_bytes),
        catalog_sha256=hashlib.sha256(catalog_bytes).digest(),
        catalog_size=len(catalog_bytes),
    )


__all__ = [
    "ARTIFACT_BASE_URLS_ENV",
    "ARTIFACT_CACHE_DIR_ENV",
    "ARTIFACT_INDEX_FILENAME",
    "ARTIFACT_INDEX_SCHEMA",
    "DEFAULT_ARTIFACT_TIMEOUT_SECONDS",
    "MAX_ARTIFACT_INDEX_BYTES",
    "MAX_ARTIFACT_INDEX_MODELS",
    "ProofV2ArtifactIndex",
    "ProofV2ArtifactIndexEntry",
    "ProofV2ArtifactStoreError",
    "RemoteProofV2ManifestResolution",
    "ResolvedProofV2Artifacts",
    "artifact_index_entry_from_files",
    "configured_proof_v2_artifact_base_urls",
    "normalize_proof_v2_artifact_base_urls",
    "proof_v2_artifact_cache_directory",
    "resolve_remote_proof_v2_artifacts",
    "resolve_remote_proof_v2_manifests",
]
