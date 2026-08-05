"""Fail-closed native GPU-sidecar control contract for proof-v3.

This module deliberately validates only scheduler/control-plane facts and
fixed-size chunk receipts.  It never accepts activations, K/V pages, or proof
state on the host, and it does not establish any execution relation by itself.
The eventual native adapter must constrain every root emitted here through the
signed global PCS/AIR/RAM relation before a validator can accept a proof.

The purpose of the contract is narrower but important: it freezes the exact
GPU-resident capture, scheduler-span, retained-witness, and lease lifecycle so
the current CPU v2 tracker cannot accidentally become a proof-v3 backend.

The signed capture ABI embeds a canonical binder qualification (its ABI ID,
version, and code identity digest).  Runtime sessions can only name that
content-addressed qualification; they cannot independently assert a broad
``runtime mode`` under a different binder.  This is deliberately not remote
attestation or an execution proof: the eventual native relation must still
bind the observed runtime facts to the retained witness.
"""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass

from verallm.proof_v3.accumulator import (
    ExecutionAccumulatorCommitmentV3,
    ExecutionChunkCommitmentV3,
    PostCommitOpeningTicketV3,
    RootOnlyExecutionTraceAccumulatorV3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3UnavailableError
from verallm.proof_v3.profile import ExecutionSecurityProfileV3
from verallm.proof_v3.request import (
    ExecutionOutputBindingV3,
    PreExecutionRequestContextV3,
)


NATIVE_GPU_SIDECAR_CAPTURE_ABI_V3 = "native_gpu_sidecar.capture.v2"
NATIVE_GPU_SIDECAR_WITNESS_ABI_V3 = "gpu.retained_witness.v1"
NATIVE_GPU_SIDECAR_DEVICE_KIND_V3 = "cuda"
NATIVE_GPU_SIDECAR_BINDER_ABI_V3 = "native_gpu_sidecar.binder.v1"
NATIVE_GPU_SIDECAR_RUNTIME_OBSERVATION_ABI_V3 = (
    "native_gpu_sidecar.runtime_observation.v1"
)
MAX_SIDECAR_BATCH_SPANS_V3 = 65_535
MAX_SIDECAR_PACKED_ROWS_V3 = 1 << 24
MAX_SIDECAR_CAPTURE_ABI_BYTES_V3 = 1 << 20
MAX_SIDECAR_BINDER_ABI_BYTES_V3 = 1 << 12
MAX_SIDECAR_TRANSITION_ADAPTERS_V3 = 1_024

_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,127}$")
_CAPTURE_ABI_MAGIC_V3 = b"VTH-SIDECAR-ABI\x00"
_CAPTURE_ABI_FORMAT_VERSION_V3 = 2
_BINDER_ABI_MAGIC_V3 = b"VTH-SIDECAR-BINDER\x00"
_BINDER_ABI_FORMAT_VERSION_V3 = 1
_BINDER_ABI_DIGEST_DOMAIN = (
    b"VERATHOS/PROOF_V3/NATIVE_SIDECAR/BINDER_ABI/SHA256"
)
_BINDER_IDENTITY_DIGEST_DOMAIN = (
    b"VERATHOS/PROOF_V3/NATIVE_SIDECAR/BINDER_IDENTITY/SHA256"
)
_RUNTIME_OBSERVATION_DIGEST_DOMAIN = (
    b"VERATHOS/PROOF_V3/NATIVE_SIDECAR/RUNTIME_OBSERVATION/SHA256"
)
_GENERATION_DIGEST_DOMAIN = b"VERATHOS/PROOF_V3/NATIVE_SIDECAR/GENERATION/SHA256"
_BATCH_SPAN_DIGEST_DOMAIN = b"VERATHOS/PROOF_V3/NATIVE_SIDECAR/SPAN/SHA256"
_BATCH_LAYOUT_DIGEST_DOMAIN = b"VERATHOS/PROOF_V3/NATIVE_SIDECAR/LAYOUT/SHA256"
_SCHEDULER_COVERAGE_INIT_DOMAIN = (
    b"VERATHOS/PROOF_V3/NATIVE_SIDECAR/SCHEDULER_COVERAGE/INIT/SHA256"
)
_SCHEDULER_COVERAGE_EVENT_DOMAIN = (
    b"VERATHOS/PROOF_V3/NATIVE_SIDECAR/SCHEDULER_COVERAGE/EVENT/SHA256"
)
_SCHEDULER_COVERAGE_SEAL_DOMAIN = (
    b"VERATHOS/PROOF_V3/NATIVE_SIDECAR/SCHEDULER_COVERAGE/SEAL/SHA256"
)
_PHASE_CODES = {"prefill": 1, "decode": 2}


def _fixed32(value: bytes, name: str, *, nonzero: bool = False) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    if nonzero and value == bytes(32):
        raise ProofV3Error(f"{name} must not be the zero digest")
    return value


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ProofV3Error(f"{name} is not a canonical identifier")
    return value


def _u32(value: int, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProofV3Error(f"{name} must be an unsigned 32-bit integer")
    if value < (1 if positive else 0) or value >= 1 << 32:
        qualifier = "positive " if positive else ""
        raise ProofV3Error(f"{name} must be a {qualifier}unsigned 32-bit integer")
    return value


def _u64(value: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < 1 << 64
    ):
        raise ProofV3Error(f"{name} must be an unsigned 64-bit integer")
    return value


def _bool(value: bool, name: str) -> bool:
    if type(value) is not bool:
        raise ProofV3Error(f"{name} must be a boolean")
    return value


def _encoded_identifier(value: str, name: str) -> bytes:
    encoded = _identifier(value, name).encode("ascii")
    return struct.pack("<B", len(encoded)) + encoded


class _CaptureAbiReaderV3:
    """Strict parser for one bounded canonical sidecar ABI artifact.

    Capture-ABI and nested binder-ABI artifacts both use this parser.  They
    are authenticated by the outer signed capture artifact, so parsing must
    reject every alternate encoding, including trailing bytes and noncanonical
    identifier spellings.
    """

    __slots__ = ("_artifact_label", "_offset", "_payload")

    def __init__(
        self,
        payload: bytes,
        *,
        artifact_label: str = "capture ABI",
        maximum_bytes: int = MAX_SIDECAR_CAPTURE_ABI_BYTES_V3,
    ) -> None:
        if not isinstance(payload, bytes):
            raise ProofV3Error(f"{artifact_label} artifact must be bytes")
        if not payload or len(payload) > maximum_bytes:
            raise ProofV3Error(f"{artifact_label} artifact length is out of range")
        self._artifact_label = artifact_label
        self._offset = 0
        self._payload = payload

    def read_exact(self, length: int, name: str) -> bytes:
        end = self._offset + length
        if end > len(self._payload):
            raise ProofV3Error(
                f"{self._artifact_label} artifact is truncated at {name}"
            )
        value = self._payload[self._offset : end]
        self._offset = end
        return value

    def read_u8(self, name: str) -> int:
        return self.read_exact(1, name)[0]

    def read_u16(self, name: str) -> int:
        return struct.unpack("<H", self.read_exact(2, name))[0]

    def read_u32(self, name: str) -> int:
        return struct.unpack("<I", self.read_exact(4, name))[0]

    def read_identifier(self, name: str) -> str:
        length = self.read_u8(f"{name} length")
        if not length:
            raise ProofV3Error(f"{self._artifact_label} {name} is empty")
        try:
            value = self.read_exact(length, name).decode("ascii")
        except UnicodeDecodeError as exc:
            raise ProofV3Error(f"{self._artifact_label} {name} is not ASCII") from exc
        _identifier(value, name)
        if _encoded_identifier(value, name) != struct.pack("<B", length) + value.encode(
            "ascii"
        ):
            # This branch is defensive: ``_identifier`` and ASCII encoding above
            # already make the current representation unique.
            raise ProofV3Error(f"{self._artifact_label} {name} is not canonical")
        return value

    def finish(self) -> None:
        if self._offset != len(self._payload):
            raise ProofV3Error(f"{self._artifact_label} artifact has trailing bytes")


@dataclass(frozen=True, slots=True)
class NativeSidecarBinderAbiV3:
    """Canonical authority-qualified identity of one native sidecar binder.

    ``binder_code_digest`` identifies the exact reviewed binder source/binary
    build selected during qualification.  Its semantics are fixed by the
    signed profile only after this canonical artifact is embedded in the
    capture ABI.  It is therefore an ABI identity, not a claim that a remote
    process has loaded that binary.
    """

    binder_abi_id: str
    binder_version: str
    binder_code_digest: bytes

    def __post_init__(self) -> None:
        if self.binder_abi_id != NATIVE_GPU_SIDECAR_BINDER_ABI_V3:
            raise ProofV3Error("native sidecar binder ABI is not supported")
        _identifier(self.binder_abi_id, "binder_abi_id")
        _identifier(self.binder_version, "binder_version")
        _fixed32(self.binder_code_digest, "binder_code_digest", nonzero=True)

    def canonical_bytes(self) -> bytes:
        canonical = (
            _BINDER_ABI_MAGIC_V3
            + struct.pack("<H", _BINDER_ABI_FORMAT_VERSION_V3)
            + _encoded_identifier(self.binder_abi_id, "binder_abi_id")
            + _encoded_identifier(self.binder_version, "binder_version")
            + self.binder_code_digest
        )
        if len(canonical) > MAX_SIDECAR_BINDER_ABI_BYTES_V3:
            raise ProofV3Error("canonical binder ABI exceeds the protocol limit")
        return canonical

    @classmethod
    def from_canonical_bytes(cls, artifact: bytes) -> "NativeSidecarBinderAbiV3":
        """Parse the exact canonical binder ABI embedded in a capture ABI."""

        reader = _CaptureAbiReaderV3(
            artifact,
            artifact_label="binder ABI",
            maximum_bytes=MAX_SIDECAR_BINDER_ABI_BYTES_V3,
        )
        if (
            reader.read_exact(len(_BINDER_ABI_MAGIC_V3), "magic")
            != _BINDER_ABI_MAGIC_V3
        ):
            raise ProofV3Error("binder ABI artifact has an unexpected magic")
        if reader.read_u16("format version") != _BINDER_ABI_FORMAT_VERSION_V3:
            raise ProofV3Error("binder ABI artifact has an unsupported format version")
        parsed = cls(
            binder_abi_id=reader.read_identifier("binder_abi_id"),
            binder_version=reader.read_identifier("binder_version"),
            binder_code_digest=reader.read_exact(32, "binder_code_digest"),
        )
        reader.finish()
        if parsed.canonical_bytes() != artifact:
            raise ProofV3Error("binder ABI artifact is not canonically encoded")
        return parsed

    def digest(self) -> bytes:
        """Return the signed binder identity/version digest for this ABI."""

        return hashlib.sha256(
            _BINDER_ABI_DIGEST_DOMAIN + self.canonical_bytes()
        ).digest()


@dataclass(frozen=True, slots=True)
class NativeSidecarCaptureAbiV3:
    """Exact device-side capture ABI authenticated by one signed profile.

    The ABI is intentionally narrow.  A profile qualifies one exact runtime
    family, scheduler layout, attention kernel, sampler, quantization encoding,
    and set of graph adapters.  It is not a broad "vLLM-compatible" flag.
    """

    runner_abi_id: str
    scheduler_abi_id: str
    attention_backend_abi_id: str
    cache_layout_abi_id: str
    sampler_abi_id: str
    quantization_semantics_id: str
    transition_adapter_ids: tuple[str, ...]
    binder_abi: NativeSidecarBinderAbiV3
    runtime_observation_abi_id: str
    max_batch_spans: int
    max_packed_rows: int
    cuda_graph_padding_abi_id: str | None = None
    capture_abi_id: str = NATIVE_GPU_SIDECAR_CAPTURE_ABI_V3
    witness_residency_abi_id: str = NATIVE_GPU_SIDECAR_WITNESS_ABI_V3

    def __post_init__(self) -> None:
        if self.capture_abi_id != NATIVE_GPU_SIDECAR_CAPTURE_ABI_V3:
            raise ProofV3Error("native sidecar capture ABI is not supported")
        if self.witness_residency_abi_id != NATIVE_GPU_SIDECAR_WITNESS_ABI_V3:
            raise ProofV3Error("native sidecar witness residency ABI is not supported")
        for value, name in (
            (self.runner_abi_id, "runner_abi_id"),
            (self.scheduler_abi_id, "scheduler_abi_id"),
            (self.attention_backend_abi_id, "attention_backend_abi_id"),
            (self.cache_layout_abi_id, "cache_layout_abi_id"),
            (self.sampler_abi_id, "sampler_abi_id"),
            (self.quantization_semantics_id, "quantization_semantics_id"),
            (self.runtime_observation_abi_id, "runtime_observation_abi_id"),
        ):
            _identifier(value, name)
        if not isinstance(self.binder_abi, NativeSidecarBinderAbiV3):
            raise ProofV3Error("native sidecar binder ABI has an unexpected type")
        if (
            self.runtime_observation_abi_id
            != NATIVE_GPU_SIDECAR_RUNTIME_OBSERVATION_ABI_V3
        ):
            raise ProofV3Error(
                "native sidecar runtime observation ABI is not supported"
            )
        if self.cuda_graph_padding_abi_id is not None:
            _identifier(self.cuda_graph_padding_abi_id, "cuda_graph_padding_abi_id")
        adapters = tuple(self.transition_adapter_ids)
        if not adapters:
            raise ProofV3Error("native sidecar ABI requires graph adapters")
        if len(adapters) > MAX_SIDECAR_TRANSITION_ADAPTERS_V3:
            raise ProofV3Error("native sidecar ABI has too many graph adapters")
        if adapters != tuple(sorted(set(adapters))):
            raise ProofV3Error(
                "native sidecar graph adapters must be sorted and distinct"
            )
        for index, adapter_id in enumerate(adapters):
            _identifier(adapter_id, f"transition_adapter_ids[{index}]")
        max_spans = _u32(self.max_batch_spans, "max_batch_spans", positive=True)
        if max_spans > MAX_SIDECAR_BATCH_SPANS_V3:
            raise ProofV3Error("max_batch_spans exceeds the protocol limit")
        max_rows = _u32(self.max_packed_rows, "max_packed_rows", positive=True)
        if max_rows > MAX_SIDECAR_PACKED_ROWS_V3:
            raise ProofV3Error("max_packed_rows exceeds the protocol limit")
        object.__setattr__(self, "transition_adapter_ids", adapters)

    def canonical_bytes(self) -> bytes:
        binder_abi_bytes = self.binder_abi.canonical_bytes()
        padding = (
            b"\x00"
            if self.cuda_graph_padding_abi_id is None
            else b"\x01"
            + _encoded_identifier(
                self.cuda_graph_padding_abi_id,
                "cuda_graph_padding_abi_id",
            )
        )
        canonical = (
            _CAPTURE_ABI_MAGIC_V3
            + struct.pack("<H", _CAPTURE_ABI_FORMAT_VERSION_V3)
            + _encoded_identifier(self.capture_abi_id, "capture_abi_id")
            + _encoded_identifier(
                self.witness_residency_abi_id,
                "witness_residency_abi_id",
            )
            + _encoded_identifier(self.runner_abi_id, "runner_abi_id")
            + _encoded_identifier(self.scheduler_abi_id, "scheduler_abi_id")
            + _encoded_identifier(
                self.attention_backend_abi_id,
                "attention_backend_abi_id",
            )
            + _encoded_identifier(self.cache_layout_abi_id, "cache_layout_abi_id")
            + _encoded_identifier(self.sampler_abi_id, "sampler_abi_id")
            + _encoded_identifier(
                self.quantization_semantics_id,
                "quantization_semantics_id",
            )
            + struct.pack("<H", len(binder_abi_bytes))
            + binder_abi_bytes
            + self.binder_abi.digest()
            + _encoded_identifier(
                self.runtime_observation_abi_id,
                "runtime_observation_abi_id",
            )
            + struct.pack(
                "<IIH",
                self.max_batch_spans,
                self.max_packed_rows,
                len(self.transition_adapter_ids),
            )
            + b"".join(
                _encoded_identifier(adapter_id, "transition_adapter_id")
                for adapter_id in self.transition_adapter_ids
            )
            + padding
        )
        if len(canonical) > MAX_SIDECAR_CAPTURE_ABI_BYTES_V3:
            raise ProofV3Error("canonical capture ABI exceeds the protocol limit")
        return canonical

    @classmethod
    def from_canonical_bytes(cls, artifact: bytes) -> "NativeSidecarCaptureAbiV3":
        """Parse the exact signed capture-ABI artifact without normalization.

        The resulting object round-trips byte-for-byte: callers must compare
        ``parsed.canonical_bytes()`` with the authenticated catalog artifact
        before treating it as a qualified capture ABI.
        """

        reader = _CaptureAbiReaderV3(artifact)
        if (
            reader.read_exact(len(_CAPTURE_ABI_MAGIC_V3), "magic")
            != _CAPTURE_ABI_MAGIC_V3
        ):
            raise ProofV3Error("capture ABI artifact has an unexpected magic")
        if reader.read_u16("format version") != _CAPTURE_ABI_FORMAT_VERSION_V3:
            raise ProofV3Error("capture ABI artifact has an unsupported format version")
        capture_abi_id = reader.read_identifier("capture_abi_id")
        witness_residency_abi_id = reader.read_identifier("witness_residency_abi_id")
        runner_abi_id = reader.read_identifier("runner_abi_id")
        scheduler_abi_id = reader.read_identifier("scheduler_abi_id")
        attention_backend_abi_id = reader.read_identifier("attention_backend_abi_id")
        cache_layout_abi_id = reader.read_identifier("cache_layout_abi_id")
        sampler_abi_id = reader.read_identifier("sampler_abi_id")
        quantization_semantics_id = reader.read_identifier("quantization_semantics_id")
        binder_abi_size = reader.read_u16("binder ABI byte length")
        if not binder_abi_size or binder_abi_size > MAX_SIDECAR_BINDER_ABI_BYTES_V3:
            raise ProofV3Error("capture ABI binder artifact length is out of range")
        binder_abi = NativeSidecarBinderAbiV3.from_canonical_bytes(
            reader.read_exact(binder_abi_size, "binder ABI")
        )
        if reader.read_exact(32, "binder ABI digest") != binder_abi.digest():
            raise ProofV3Error(
                "capture ABI binder identity digest does not match its qualified ABI"
            )
        runtime_observation_abi_id = reader.read_identifier(
            "runtime_observation_abi_id"
        )
        max_batch_spans = reader.read_u32("max_batch_spans")
        max_packed_rows = reader.read_u32("max_packed_rows")
        adapter_count = reader.read_u16("transition_adapter_ids count")
        if not adapter_count or adapter_count > MAX_SIDECAR_TRANSITION_ADAPTERS_V3:
            raise ProofV3Error("capture ABI graph-adapter count is out of range")
        transition_adapter_ids = tuple(
            reader.read_identifier(f"transition_adapter_ids[{index}]")
            for index in range(adapter_count)
        )
        has_padding = reader.read_u8("cuda_graph_padding_abi_id presence")
        if has_padding not in {0, 1}:
            raise ProofV3Error("capture ABI padding presence flag is not canonical")
        cuda_graph_padding_abi_id = (
            None
            if has_padding == 0
            else reader.read_identifier("cuda_graph_padding_abi_id")
        )
        reader.finish()
        parsed = cls(
            capture_abi_id=capture_abi_id,
            witness_residency_abi_id=witness_residency_abi_id,
            runner_abi_id=runner_abi_id,
            scheduler_abi_id=scheduler_abi_id,
            attention_backend_abi_id=attention_backend_abi_id,
            cache_layout_abi_id=cache_layout_abi_id,
            sampler_abi_id=sampler_abi_id,
            quantization_semantics_id=quantization_semantics_id,
            transition_adapter_ids=transition_adapter_ids,
            binder_abi=binder_abi,
            runtime_observation_abi_id=runtime_observation_abi_id,
            max_batch_spans=max_batch_spans,
            max_packed_rows=max_packed_rows,
            cuda_graph_padding_abi_id=cuda_graph_padding_abi_id,
        )
        if parsed.canonical_bytes() != artifact:
            raise ProofV3Error("capture ABI artifact is not canonically encoded")
        return parsed

    def artifact_digest(self) -> bytes:
        """Return the raw SHA-256 digest used by the signed artifact catalog."""

        return hashlib.sha256(self.canonical_bytes()).digest()

    def digest(self) -> bytes:
        """Compatibility spelling for :meth:`artifact_digest`.

        Earlier scaffold code domain-separated this digest, which could never
        equal the catalog's raw-artifact digest.  V3 signs the canonical bytes
        directly; new callers should use :meth:`artifact_digest` explicitly.
        """

        return self.artifact_digest()

    @property
    def binder_abi_digest(self) -> bytes:
        """Return the signed identity/version digest of the qualified binder."""

        return self.binder_abi.digest()

    def validate_profile(self, *, profile: ExecutionSecurityProfileV3) -> None:
        """Require exact profile/runtime compatibility before capture begins."""

        if not isinstance(profile, ExecutionSecurityProfileV3):
            raise ProofV3Error("execution profile has an unexpected type")
        profile.require_hard_execution_capability()
        relation = profile.relation_spec
        if relation.capture_abi_digest != self.digest():
            raise ProofV3Error("signed profile does not bind this native sidecar ABI")
        if self.sampler_abi_id != relation.sampler_abi_id:
            raise ProofV3Error("native sidecar sampler ABI differs from the profile")
        if self.quantization_semantics_id != profile.quantization_semantics_id:
            raise ProofV3Error(
                "native sidecar quantization semantics differ from the profile"
            )
        expected_adapters = tuple(
            sorted({node.transition_adapter_id for node in relation.nodes})
        )
        if self.transition_adapter_ids != expected_adapters:
            raise ProofV3Error(
                "native sidecar graph adapters do not exactly match the profile"
            )
        if self.max_packed_rows < max(
            relation.prefill_chunk_tokens,
            relation.decode_chunk_tokens,
        ):
            raise ProofV3Error(
                "native sidecar packed-row bound is smaller than a signed chunk"
            )


@dataclass(frozen=True, slots=True)
class NativeSidecarBinderIdentityV3:
    """Identity of the local binder that observed runtime control facts.

    The runtime does not claim an ABI ID or source digest directly.  It may
    only name the content-addressed, authority-qualified binder ABI embedded
    in the signed capture artifact, alongside a per-instance handle.  This is
    still not remote attestation: a native adapter must eventually constrain
    that handle and the observed facts in its authenticated witness.
    """

    qualified_binder_abi_digest: bytes
    binder_instance_digest: bytes

    def __post_init__(self) -> None:
        _fixed32(
            self.qualified_binder_abi_digest,
            "qualified_binder_abi_digest",
            nonzero=True,
        )
        _fixed32(
            self.binder_instance_digest,
            "binder_instance_digest",
            nonzero=True,
        )

    def canonical_bytes(self) -> bytes:
        return self.qualified_binder_abi_digest + self.binder_instance_digest

    def digest(self) -> bytes:
        return hashlib.sha256(
            _BINDER_IDENTITY_DIGEST_DOMAIN + self.canonical_bytes()
        ).digest()

    def validate_capture_abi(self, *, capture_abi: NativeSidecarCaptureAbiV3) -> None:
        if not isinstance(capture_abi, NativeSidecarCaptureAbiV3):
            raise ProofV3Error("native sidecar capture ABI has an unexpected type")
        if self.qualified_binder_abi_digest != capture_abi.binder_abi_digest:
            raise ProofV3Error(
                "runtime binder qualification differs from the signed sidecar ABI"
            )


@dataclass(frozen=True, slots=True)
class NativeSidecarRuntimeObservationV3:
    """Explicit observed runtime facts; there are intentionally no defaults.

    Constructing this record requires the binder to state every feature that
    affects capture.  It is not a self-authenticating attestation: the future
    native witness relation still has to constrain the observations.  Requiring
    explicit values here merely removes the unsafe path where omitted fields
    silently meant the easiest qualified execution mode.
    """

    observation_abi_id: str
    runner_abi_id: str
    scheduler_abi_id: str
    attention_backend_abi_id: str
    cache_layout_abi_id: str
    sampler_abi_id: str
    quantization_semantics_id: str
    device_rank: int
    device_kind: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    cuda_graph_padding_abi_id: str | None
    uses_host_tensor_export: bool
    uses_host_proof_state: bool
    uses_prefix_cache_sharing: bool
    uses_sliding_window: bool
    uses_speculative_decode: bool
    uses_ubatch_slices: bool
    uses_cascade_attention: bool
    uses_kv_offload_or_transfer: bool
    uses_lora_or_adapter: bool
    uses_moe: bool
    uses_multimodal_or_prompt_embeds: bool
    uses_noncanonical_sampling: bool

    def __post_init__(self) -> None:
        for value, name in (
            (self.observation_abi_id, "observation_abi_id"),
            (self.runner_abi_id, "runner_abi_id"),
            (self.scheduler_abi_id, "scheduler_abi_id"),
            (self.attention_backend_abi_id, "attention_backend_abi_id"),
            (self.cache_layout_abi_id, "cache_layout_abi_id"),
            (self.sampler_abi_id, "sampler_abi_id"),
            (self.quantization_semantics_id, "quantization_semantics_id"),
        ):
            _identifier(value, name)
        if self.observation_abi_id != NATIVE_GPU_SIDECAR_RUNTIME_OBSERVATION_ABI_V3:
            raise ProofV3Error(
                "native sidecar runtime observation ABI is not supported"
            )
        _u32(self.device_rank, "device_rank")
        if self.device_kind != NATIVE_GPU_SIDECAR_DEVICE_KIND_V3:
            raise ProofV3Error("proof-v3 sidecar requires a CUDA device")
        for value, name in (
            (self.tensor_parallel_size, "tensor_parallel_size"),
            (self.pipeline_parallel_size, "pipeline_parallel_size"),
            (self.data_parallel_size, "data_parallel_size"),
        ):
            _u32(value, name, positive=True)
        if self.cuda_graph_padding_abi_id is not None:
            _identifier(self.cuda_graph_padding_abi_id, "cuda_graph_padding_abi_id")
        for value, name in (
            (self.uses_host_tensor_export, "uses_host_tensor_export"),
            (self.uses_host_proof_state, "uses_host_proof_state"),
            (self.uses_prefix_cache_sharing, "uses_prefix_cache_sharing"),
            (self.uses_sliding_window, "uses_sliding_window"),
            (self.uses_speculative_decode, "uses_speculative_decode"),
            (self.uses_ubatch_slices, "uses_ubatch_slices"),
            (self.uses_cascade_attention, "uses_cascade_attention"),
            (self.uses_kv_offload_or_transfer, "uses_kv_offload_or_transfer"),
            (self.uses_lora_or_adapter, "uses_lora_or_adapter"),
            (self.uses_moe, "uses_moe"),
            (
                self.uses_multimodal_or_prompt_embeds,
                "uses_multimodal_or_prompt_embeds",
            ),
            (self.uses_noncanonical_sampling, "uses_noncanonical_sampling"),
        ):
            _bool(value, name)

    def canonical_bytes(self) -> bytes:
        padding = (
            b"\x00"
            if self.cuda_graph_padding_abi_id is None
            else b"\x01"
            + _encoded_identifier(
                self.cuda_graph_padding_abi_id,
                "cuda_graph_padding_abi_id",
            )
        )
        feature_flags = (
            (1 if self.uses_host_tensor_export else 0) << 0
            | (1 if self.uses_host_proof_state else 0) << 1
            | (1 if self.uses_prefix_cache_sharing else 0) << 2
            | (1 if self.uses_sliding_window else 0) << 3
            | (1 if self.uses_speculative_decode else 0) << 4
            | (1 if self.uses_ubatch_slices else 0) << 5
            | (1 if self.uses_cascade_attention else 0) << 6
            | (1 if self.uses_kv_offload_or_transfer else 0) << 7
            | (1 if self.uses_lora_or_adapter else 0) << 8
            | (1 if self.uses_moe else 0) << 9
            | (1 if self.uses_multimodal_or_prompt_embeds else 0) << 10
            | (1 if self.uses_noncanonical_sampling else 0) << 11
        )
        return (
            _encoded_identifier(self.observation_abi_id, "observation_abi_id")
            + _encoded_identifier(self.runner_abi_id, "runner_abi_id")
            + _encoded_identifier(self.scheduler_abi_id, "scheduler_abi_id")
            + _encoded_identifier(
                self.attention_backend_abi_id,
                "attention_backend_abi_id",
            )
            + _encoded_identifier(self.cache_layout_abi_id, "cache_layout_abi_id")
            + _encoded_identifier(self.sampler_abi_id, "sampler_abi_id")
            + _encoded_identifier(
                self.quantization_semantics_id,
                "quantization_semantics_id",
            )
            + struct.pack(
                "<IBIII",
                self.device_rank,
                1 if self.device_kind == NATIVE_GPU_SIDECAR_DEVICE_KIND_V3 else 0,
                self.tensor_parallel_size,
                self.pipeline_parallel_size,
                self.data_parallel_size,
            )
            + padding
            + struct.pack("<H", feature_flags)
        )

    def digest(self) -> bytes:
        return hashlib.sha256(
            _RUNTIME_OBSERVATION_DIGEST_DOMAIN + self.canonical_bytes()
        ).digest()

    def validate_capture_abi(self, *, capture_abi: NativeSidecarCaptureAbiV3) -> None:
        """Reject every runtime mode not covered by the initial sidecar ABI."""

        if not isinstance(capture_abi, NativeSidecarCaptureAbiV3):
            raise ProofV3Error("native sidecar capture ABI has an unexpected type")
        if self.observation_abi_id != capture_abi.runtime_observation_abi_id:
            raise ProofV3Error("runtime observation ABI differs from the sidecar ABI")
        if self.runner_abi_id != capture_abi.runner_abi_id:
            raise ProofV3Error("runtime runner differs from the sidecar ABI")
        if self.scheduler_abi_id != capture_abi.scheduler_abi_id:
            raise ProofV3Error("runtime scheduler differs from the sidecar ABI")
        if self.attention_backend_abi_id != capture_abi.attention_backend_abi_id:
            raise ProofV3Error("runtime attention backend differs from the sidecar ABI")
        if self.cache_layout_abi_id != capture_abi.cache_layout_abi_id:
            raise ProofV3Error("runtime cache layout differs from the sidecar ABI")
        if self.sampler_abi_id != capture_abi.sampler_abi_id:
            raise ProofV3Error("runtime sampler differs from the sidecar ABI")
        if self.quantization_semantics_id != capture_abi.quantization_semantics_id:
            raise ProofV3Error("runtime quantization differs from the sidecar ABI")
        if self.cuda_graph_padding_abi_id != capture_abi.cuda_graph_padding_abi_id:
            raise ProofV3Error(
                "runtime CUDA-graph padding differs from the sidecar ABI"
            )
        if (
            self.tensor_parallel_size != 1
            or self.pipeline_parallel_size != 1
            or self.data_parallel_size != 1
        ):
            raise ProofV3Error("parallel proof-v3 sidecar execution is not qualified")
        if any(
            (
                self.uses_host_tensor_export,
                self.uses_host_proof_state,
                self.uses_prefix_cache_sharing,
                self.uses_sliding_window,
                self.uses_speculative_decode,
                self.uses_ubatch_slices,
                self.uses_cascade_attention,
                self.uses_kv_offload_or_transfer,
                self.uses_lora_or_adapter,
                self.uses_moe,
                self.uses_multimodal_or_prompt_embeds,
                self.uses_noncanonical_sampling,
            )
        ):
            raise ProofV3Error(
                "runtime mode is not qualified for proof-v3 sidecar capture"
            )


# Keep the first scaffold's public spelling as a compatibility alias.  The
# canonical type name makes clear that values are observed control-plane facts,
# not an optional mode request.
NativeSidecarRuntimeModeV3 = NativeSidecarRuntimeObservationV3


@dataclass(frozen=True, slots=True)
class NativeSidecarRequestGenerationV3:
    """Opaque per-runner request-generation handle, never a vLLM request ID."""

    generation_digest: bytes
    precommit_context_digest: bytes
    execution_profile_digest: bytes
    cache_lease_digest: bytes
    capture_abi_digest: bytes
    binder_identity_digest: bytes
    runtime_observation_digest: bytes
    device_rank: int

    def __post_init__(self) -> None:
        _fixed32(self.generation_digest, "generation_digest", nonzero=True)
        _fixed32(
            self.precommit_context_digest,
            "precommit_context_digest",
            nonzero=True,
        )
        _fixed32(
            self.execution_profile_digest,
            "execution_profile_digest",
            nonzero=True,
        )
        _fixed32(self.cache_lease_digest, "cache_lease_digest", nonzero=True)
        _fixed32(self.capture_abi_digest, "capture_abi_digest", nonzero=True)
        _fixed32(
            self.binder_identity_digest,
            "binder_identity_digest",
            nonzero=True,
        )
        _fixed32(
            self.runtime_observation_digest,
            "runtime_observation_digest",
            nonzero=True,
        )
        _u32(self.device_rank, "device_rank")

    @classmethod
    def derive(
        cls,
        *,
        device_handle_digest: bytes,
        precommit_context: PreExecutionRequestContextV3,
        capture_abi: NativeSidecarCaptureAbiV3,
        binder_identity: NativeSidecarBinderIdentityV3,
        runtime_observation: NativeSidecarRuntimeObservationV3,
    ) -> "NativeSidecarRequestGenerationV3":
        """Bind a local device handle to the qualified capture control plane."""

        if not isinstance(precommit_context, PreExecutionRequestContextV3):
            raise ProofV3Error("precommit context has an unexpected type")
        if not isinstance(capture_abi, NativeSidecarCaptureAbiV3):
            raise ProofV3Error("capture ABI has an unexpected type")
        if not isinstance(binder_identity, NativeSidecarBinderIdentityV3):
            raise ProofV3Error("binder identity has an unexpected type")
        if not isinstance(runtime_observation, NativeSidecarRuntimeObservationV3):
            raise ProofV3Error("runtime observation has an unexpected type")
        binder_identity.validate_capture_abi(capture_abi=capture_abi)
        runtime_observation.validate_capture_abi(capture_abi=capture_abi)
        handle = _fixed32(device_handle_digest, "device_handle_digest", nonzero=True)
        rank = _u32(runtime_observation.device_rank, "device_rank")
        capture_abi_digest = capture_abi.artifact_digest()
        binder_identity_digest = binder_identity.digest()
        runtime_observation_digest = runtime_observation.digest()
        return cls(
            generation_digest=hashlib.sha256(
                _GENERATION_DIGEST_DOMAIN
                + handle
                + precommit_context.digest()
                + precommit_context.execution_profile_digest
                + precommit_context.cache_lease_digest
                + capture_abi_digest
                + binder_identity_digest
                + runtime_observation_digest
                + struct.pack("<I", rank)
            ).digest(),
            precommit_context_digest=precommit_context.digest(),
            execution_profile_digest=precommit_context.execution_profile_digest,
            cache_lease_digest=precommit_context.cache_lease_digest,
            capture_abi_digest=capture_abi_digest,
            binder_identity_digest=binder_identity_digest,
            runtime_observation_digest=runtime_observation_digest,
            device_rank=rank,
        )


@dataclass(frozen=True, slots=True)
class NativeSidecarBatchSpanV3:
    """One exact scheduler-owned packed-row span for a tracked request.

    Positions are required to equal logical sequence coordinates in the first
    qualified text-only ABI.  Models needing position offsets, virtual tokens,
    or another layout need a new signed capture ABI and adapter qualification.
    """

    generation_digest: bytes
    phase: str
    logical_token_start: int
    token_count: int
    packed_row_start: int
    packed_row_count: int
    sequence_length_before: int
    sequence_length_after: int
    position_start: int

    def __post_init__(self) -> None:
        _fixed32(self.generation_digest, "batch span generation_digest", nonzero=True)
        if self.phase not in _PHASE_CODES:
            raise ProofV3Error("batch span phase is not supported")
        for value, name, positive in (
            (self.logical_token_start, "batch span logical_token_start", False),
            (self.token_count, "batch span token_count", True),
            (self.packed_row_start, "batch span packed_row_start", False),
            (self.packed_row_count, "batch span packed_row_count", True),
            (self.sequence_length_before, "batch span sequence_length_before", False),
            (self.sequence_length_after, "batch span sequence_length_after", True),
            (self.position_start, "batch span position_start", False),
        ):
            _u32(value, name, positive=positive)
        if self.packed_row_count != self.token_count:
            raise ProofV3Error("batch span packed rows must exactly match token rows")
        if self.packed_row_start + self.packed_row_count > 1 << 32:
            raise ProofV3Error("batch span packed-row range overflows")
        logical_end = self.logical_token_start + self.token_count
        if logical_end > 1 << 32:
            raise ProofV3Error("batch span logical-token range overflows")
        if self.sequence_length_before != self.logical_token_start:
            raise ProofV3Error(
                "batch span sequence length does not match logical start"
            )
        if self.sequence_length_after != logical_end:
            raise ProofV3Error("batch span sequence length does not match logical end")
        if self.position_start != self.logical_token_start:
            raise ProofV3Error(
                "batch span position does not match the logical coordinate"
            )

    def canonical_bytes(self) -> bytes:
        return self.generation_digest + struct.pack(
            "<B7I",
            _PHASE_CODES[self.phase],
            self.logical_token_start,
            self.token_count,
            self.packed_row_start,
            self.packed_row_count,
            self.sequence_length_before,
            self.sequence_length_after,
            self.position_start,
        )

    def digest(self) -> bytes:
        return hashlib.sha256(
            _BATCH_SPAN_DIGEST_DOMAIN + self.canonical_bytes()
        ).digest()


@dataclass(frozen=True, slots=True)
class NativeSidecarBatchLayoutV3:
    """Canonical tracked portion of one scheduler batch.

    Untracked ordinary requests may occupy other packed rows.  Every v3-tracked
    request appears exactly once, and spans may neither overlap nor alias a
    generation handle.  A per-request sidecar checks its own exact span before
    accepting a chunk receipt.
    """

    batch_sequence: int
    device_rank: int
    packed_row_count: int
    spans: tuple[NativeSidecarBatchSpanV3, ...]

    def __post_init__(self) -> None:
        _u64(self.batch_sequence, "batch_sequence")
        _u32(self.device_rank, "batch device_rank")
        packed_rows = _u32(
            self.packed_row_count,
            "batch packed_row_count",
            positive=True,
        )
        if packed_rows > MAX_SIDECAR_PACKED_ROWS_V3:
            raise ProofV3Error("batch packed_row_count exceeds the protocol limit")
        spans = tuple(self.spans)
        if not spans or not all(
            isinstance(span, NativeSidecarBatchSpanV3) for span in spans
        ):
            raise ProofV3Error("native sidecar batch requires canonical spans")
        if len(spans) > MAX_SIDECAR_BATCH_SPANS_V3:
            raise ProofV3Error("native sidecar batch has too many spans")
        keys = tuple((span.packed_row_start, span.generation_digest) for span in spans)
        if keys != tuple(sorted(keys)):
            raise ProofV3Error("native sidecar batch spans must be canonically ordered")
        generations = tuple(span.generation_digest for span in spans)
        if len(set(generations)) != len(generations):
            raise ProofV3Error(
                "native sidecar batch cannot duplicate a request generation"
            )
        previous_end = 0
        for span in spans:
            if span.packed_row_start < previous_end:
                raise ProofV3Error("native sidecar batch spans overlap")
            if span.packed_row_start + span.packed_row_count > packed_rows:
                raise ProofV3Error("native sidecar span exceeds the packed batch rows")
            previous_end = span.packed_row_start + span.packed_row_count
        object.__setattr__(self, "spans", spans)

    def span_for(
        self,
        *,
        generation: NativeSidecarRequestGenerationV3,
    ) -> NativeSidecarBatchSpanV3:
        """Return the unique scheduler span for one validator-bound request."""

        if not isinstance(generation, NativeSidecarRequestGenerationV3):
            raise ProofV3Error("request generation has an unexpected type")
        matches = tuple(
            span
            for span in self.spans
            if span.generation_digest == generation.generation_digest
        )
        if len(matches) != 1:
            raise ProofV3Error(
                "scheduler batch does not contain exactly one request span"
            )
        return matches[0]

    def canonical_bytes(self) -> bytes:
        return struct.pack(
            "<QIIH",
            self.batch_sequence,
            self.device_rank,
            self.packed_row_count,
            len(self.spans),
        ) + b"".join(span.digest() for span in self.spans)

    def digest(self) -> bytes:
        """Commit packed-row capacity and every V3-tracked request span.

        Rows belonging to ordinary untracked requests are represented only by
        gaps within ``packed_row_count``; they have no individually committed
        span in this per-request control ABI.  A qualified native binder must
        derive both the capacity and the tracked span from the original
        scheduler layout before it captures the corresponding device rows.
        """

        return hashlib.sha256(
            _BATCH_LAYOUT_DIGEST_DOMAIN + self.canonical_bytes()
        ).digest()


class NativeSidecarCaptureSessionV3:
    """Control-plane validator for native sidecar receipts.

    This session is intentionally *not* a prover and has no post-nonce witness
    openings.  It uses the root-only ledger only to enforce bounded, ordered
    receipt mechanics.  Passing this class is never proof-v3 verification.
    """

    __slots__ = (
        "_binder_identity",
        "_capture_abi",
        "_captured_end",
        "_flushed_end",
        "_generation",
        "_last_batch_sequence",
        "_ledger",
        "_precommit_context",
        "_profile",
        "_runtime_mode",
        "_scheduler_coverage_chain",
        "_scheduler_coverage_span_count",
        "_scheduler_coverage_start",
        "_state",
    )

    def __init__(
        self,
        *,
        profile: ExecutionSecurityProfileV3,
        precommit_context: PreExecutionRequestContextV3,
        capture_abi: NativeSidecarCaptureAbiV3,
        binder_identity: NativeSidecarBinderIdentityV3,
        runtime_mode: NativeSidecarRuntimeObservationV3,
        device_handle_digest: bytes,
        cache_epoch: int,
    ) -> None:
        if not isinstance(profile, ExecutionSecurityProfileV3):
            raise ProofV3Error("execution profile has an unexpected type")
        if not isinstance(precommit_context, PreExecutionRequestContextV3):
            raise ProofV3Error("precommit context has an unexpected type")
        if not isinstance(capture_abi, NativeSidecarCaptureAbiV3):
            raise ProofV3Error("native sidecar capture ABI has an unexpected type")
        if not isinstance(binder_identity, NativeSidecarBinderIdentityV3):
            raise ProofV3Error("native sidecar binder identity has an unexpected type")
        if not isinstance(runtime_mode, NativeSidecarRuntimeObservationV3):
            raise ProofV3Error(
                "native sidecar runtime observation has an unexpected type"
            )
        capture_abi.validate_profile(profile=profile)
        binder_identity.validate_capture_abi(capture_abi=capture_abi)
        runtime_mode.validate_capture_abi(capture_abi=capture_abi)
        if precommit_context.execution_profile_digest != profile.digest():
            raise ProofV3Error("precommit context has an unexpected execution profile")
        if precommit_context.static_manifest_digest != profile.static_manifest_digest:
            raise ProofV3Error("precommit context has an unexpected static manifest")
        generation = NativeSidecarRequestGenerationV3.derive(
            device_handle_digest=device_handle_digest,
            precommit_context=precommit_context,
            capture_abi=capture_abi,
            binder_identity=binder_identity,
            runtime_observation=runtime_mode,
        )
        ledger = RootOnlyExecutionTraceAccumulatorV3()
        ledger.begin(
            profile=profile,
            precommit_context=precommit_context,
            cache_epoch=cache_epoch,
        )
        self._binder_identity = binder_identity
        self._capture_abi = capture_abi
        self._captured_end = 0
        self._flushed_end = 0
        self._generation = generation
        self._last_batch_sequence: int | None = None
        self._ledger = ledger
        self._precommit_context = precommit_context
        self._profile = profile
        self._runtime_mode = runtime_mode
        self._scheduler_coverage_chain: bytes | None = None
        self._scheduler_coverage_span_count = 0
        self._scheduler_coverage_start: int | None = None
        self._state = "prefill"

    @property
    def generation(self) -> NativeSidecarRequestGenerationV3:
        """Return the opaque handle that scheduler spans must bind."""

        return self._generation

    @property
    def state(self) -> str:
        """Return the capture lifecycle without exposing witness material."""

        return self._state

    def _append_scheduler_coverage(
        self,
        *,
        layout: NativeSidecarBatchLayoutV3,
        span: NativeSidecarBatchSpanV3,
    ) -> None:
        """Accumulate observed scheduler facts until one chunk flushes them.

        A chunk must flush the entire pending span interval.  This deliberately
        avoids a control-plane-only claim that a fraction of a scheduler span
        was captured; a future native relation can use the same exact boundary.
        """

        if self._scheduler_coverage_chain is None:
            if span.logical_token_start != self._flushed_end:
                raise ProofV3Error(
                    "native sidecar scheduler coverage does not begin at the flush boundary"
                )
            self._scheduler_coverage_start = span.logical_token_start
            self._scheduler_coverage_chain = hashlib.sha256(
                _SCHEDULER_COVERAGE_INIT_DOMAIN
                + self._generation.generation_digest
                + self._generation.capture_abi_digest
                + self._generation.binder_identity_digest
                + self._generation.runtime_observation_digest
                + struct.pack("<I", span.logical_token_start)
            ).digest()
        assert self._scheduler_coverage_chain is not None
        self._scheduler_coverage_chain = hashlib.sha256(
            _SCHEDULER_COVERAGE_EVENT_DOMAIN
            + self._scheduler_coverage_chain
            + layout.digest()
            + span.digest()
        ).digest()
        self._scheduler_coverage_span_count += 1

    def _pending_scheduler_coverage_digest(self, *, logical_token_end: int) -> bytes:
        if (
            self._scheduler_coverage_chain is None
            or self._scheduler_coverage_start is None
            or not self._scheduler_coverage_span_count
        ):
            raise ProofV3Error("native sidecar chunk has no scheduler coverage")
        return hashlib.sha256(
            _SCHEDULER_COVERAGE_SEAL_DOMAIN
            + self._scheduler_coverage_chain
            + struct.pack(
                "<III",
                self._scheduler_coverage_start,
                logical_token_end,
                self._scheduler_coverage_span_count,
            )
        ).digest()

    def _clear_scheduler_coverage(self) -> None:
        self._scheduler_coverage_chain = None
        self._scheduler_coverage_span_count = 0
        self._scheduler_coverage_start = None

    @property
    def pending_scheduler_coverage_digest(self) -> bytes:
        """Return the receipt binding for the currently observed span interval.

        The native sidecar, not the host ledger, is expected to compute the
        same value when producing the chunk.  Exposing the deterministic value
        is useful for adapter conformance tests; it is not an attestation or a
        substitute for a native execution witness.
        """

        if self._captured_end == self._flushed_end:
            raise ProofV3Error("native sidecar has no pending scheduler coverage")
        return self._pending_scheduler_coverage_digest(
            logical_token_end=self._captured_end
        )

    def record_batch_layout(self, *, layout: NativeSidecarBatchLayoutV3) -> None:
        """Bind one original scheduler span before accepting its chunk receipt."""

        if self._state not in {"prefill", "decode"}:
            raise ProofV3Error("native sidecar cannot record a batch after sealing")
        if not isinstance(layout, NativeSidecarBatchLayoutV3):
            raise ProofV3Error("native sidecar batch layout has an unexpected type")
        if layout.device_rank != self._runtime_mode.device_rank:
            raise ProofV3Error("native sidecar batch belongs to another device rank")
        if len(layout.spans) > self._capture_abi.max_batch_spans:
            raise ProofV3Error("native sidecar batch exceeds its signed span limit")
        if layout.packed_row_count > self._capture_abi.max_packed_rows:
            raise ProofV3Error("native sidecar batch exceeds its signed row limit")
        if (
            self._last_batch_sequence is not None
            and layout.batch_sequence <= self._last_batch_sequence
        ):
            raise ProofV3Error("native sidecar batch sequence is stale or reordered")
        span = layout.span_for(generation=self._generation)
        if span.phase != self._state:
            raise ProofV3Error("native sidecar span phase does not match its lifecycle")
        if span.logical_token_start != self._captured_end:
            raise ProofV3Error(
                "native sidecar spans must cover logical tokens contiguously"
            )
        logical_end = span.logical_token_start + span.token_count
        if self._state == "prefill":
            if logical_end > self._precommit_context.context_token_count:
                raise ProofV3Error(
                    "native sidecar prefill span exceeds validator prompt"
                )
        elif logical_end > (
            self._precommit_context.context_token_count
            + self._profile.max_verified_decode_tokens
        ):
            raise ProofV3Error(
                "native sidecar decode span exceeds signed profile limit"
            )
        self._append_scheduler_coverage(layout=layout, span=span)
        self._captured_end = logical_end
        self._last_batch_sequence = layout.batch_sequence

    def accept_chunk(self, *, chunk: ExecutionChunkCommitmentV3) -> None:
        """Accept a sidecar-produced chunk only after covered scheduler spans."""

        if self._state not in {"prefill", "decode"}:
            raise ProofV3Error("native sidecar cannot accept a chunk after sealing")
        if not isinstance(chunk, ExecutionChunkCommitmentV3):
            raise ProofV3Error("native sidecar chunk has an unexpected type")
        if chunk.phase != self._state:
            raise ProofV3Error(
                "native sidecar chunk phase does not match its lifecycle"
            )
        if chunk.cache_lease_digest != self._generation.cache_lease_digest:
            raise ProofV3Error("native sidecar chunk has an unexpected cache lease")
        if chunk.logical_token_start != self._flushed_end:
            raise ProofV3Error(
                "native sidecar chunks must flush logical tokens contiguously"
            )
        logical_end = chunk.logical_token_start + chunk.token_count
        if logical_end != self._captured_end:
            raise ProofV3Error(
                "native sidecar chunk must exactly cover pending scheduler spans"
            )
        if chunk.sidecar_generation_digest != self._generation.generation_digest:
            raise ProofV3Error(
                "native sidecar chunk has an unexpected request generation"
            )
        if chunk.scheduler_coverage_digest != self._pending_scheduler_coverage_digest(
            logical_token_end=logical_end
        ):
            raise ProofV3Error(
                "native sidecar chunk has an unexpected scheduler coverage digest"
            )
        self._ledger.accumulate_chunk(chunk=chunk)
        self._flushed_end = logical_end
        self._clear_scheduler_coverage()

    def seal_prefill(self) -> bytes:
        """Seal the fully captured validator prompt before any decode span."""

        if self._state != "prefill":
            raise ProofV3Error("native sidecar prefill is not active")
        context_count = self._precommit_context.context_token_count
        if self._captured_end != context_count or self._flushed_end != context_count:
            raise ProofV3Error(
                "native sidecar prefill does not cover the validator prompt"
            )
        root = self._ledger.seal_prefill(context_token_count=context_count)
        self._state = "decode"
        return root

    def seal_decode(
        self,
        *,
        output_binding: ExecutionOutputBindingV3,
    ) -> ExecutionAccumulatorCommitmentV3:
        """Seal exact observed output coverage before the validator nonce."""

        if self._state != "decode":
            raise ProofV3Error("native sidecar decode is not active")
        if not isinstance(output_binding, ExecutionOutputBindingV3):
            raise ProofV3Error("output binding has an unexpected type")
        expected_end = (
            self._precommit_context.context_token_count
            + output_binding.decode_token_count
        )
        if self._captured_end != expected_end or self._flushed_end != expected_end:
            raise ProofV3Error(
                "native sidecar decode does not cover the observed output"
            )
        sealed = self._ledger.seal_decode(
            decode_token_count=output_binding.decode_token_count,
            output_binding=output_binding,
        )
        self._state = "sealed"
        return sealed

    def freeze_precommit(self) -> None:
        """Forbid mutation before the validator's inline nonce reveal."""

        if self._state != "sealed":
            raise ProofV3Error("native sidecar must seal before freezing")
        self._ledger.freeze_precommit()
        self._state = "frozen"

    def open_postnonce(self, *, ticket: PostCommitOpeningTicketV3) -> bytes:
        """Fail closed: this control ledger has no retained native witnesses."""

        if self._state != "frozen":
            raise ProofV3Error("native sidecar must freeze before opening")
        try:
            return self._ledger.open_postnonce(ticket=ticket)
        except ProofV3UnavailableError:
            raise ProofV3UnavailableError(
                "sidecar control ledger has no native retained witness openings"
            ) from None

    def release(self) -> None:
        """Release control-plane state; native GPU state needs the same lifecycle."""

        if self._state == "released":
            return
        self._ledger.release()
        self._state = "released"


__all__ = [
    "MAX_SIDECAR_BATCH_SPANS_V3",
    "MAX_SIDECAR_BINDER_ABI_BYTES_V3",
    "MAX_SIDECAR_CAPTURE_ABI_BYTES_V3",
    "MAX_SIDECAR_PACKED_ROWS_V3",
    "MAX_SIDECAR_TRANSITION_ADAPTERS_V3",
    "NATIVE_GPU_SIDECAR_BINDER_ABI_V3",
    "NATIVE_GPU_SIDECAR_CAPTURE_ABI_V3",
    "NATIVE_GPU_SIDECAR_DEVICE_KIND_V3",
    "NATIVE_GPU_SIDECAR_RUNTIME_OBSERVATION_ABI_V3",
    "NATIVE_GPU_SIDECAR_WITNESS_ABI_V3",
    "NativeSidecarBatchLayoutV3",
    "NativeSidecarBatchSpanV3",
    "NativeSidecarBinderAbiV3",
    "NativeSidecarBinderIdentityV3",
    "NativeSidecarCaptureAbiV3",
    "NativeSidecarCaptureSessionV3",
    "NativeSidecarRequestGenerationV3",
    "NativeSidecarRuntimeModeV3",
    "NativeSidecarRuntimeObservationV3",
]
