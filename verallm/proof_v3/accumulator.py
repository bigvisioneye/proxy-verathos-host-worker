"""Runtime-capture interface required before proof-v3 can be enabled.

This is deliberately an interface, not a CPU trace collector.  Long-context
proofs cannot retain raw prefix activations or raw K/V tensors on the host.
An implementation must use bounded GPU-side authenticated state plus a
retention lease that can serve nonce-selected openings after the final token.
"""

from __future__ import annotations

import hashlib
import re
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error, ProofV3UnavailableError
from verallm.proof_v3.payload import ProofV3CommitmentEnvelope
from verallm.proof_v3.profile import ExecutionSecurityProfileV3
from verallm.proof_v3.relation import RECURSIVE_CHUNK_ACCUMULATOR_ABI_V3
from verallm.proof_v3.request import (
    ExecutionOutputBindingV3,
    PreExecutionRequestContextV3,
)


MAX_ROOT_ONLY_EXECUTION_RECORDS = 1 << 24
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,127}$")
_ROOT_CONTEXT_DOMAIN = b"VERATHOS/PROOF_V3/ROOT_ONLY/CONTEXT/SHA256"
_PREFILL_INIT_DOMAIN = b"VERATHOS/PROOF_V3/ROOT_ONLY/PREFILL_INIT/SHA256"
_PREFILL_EVENT_DOMAIN = b"VERATHOS/PROOF_V3/ROOT_ONLY/PREFILL_EVENT/SHA256"
_PREFILL_SEAL_DOMAIN = b"VERATHOS/PROOF_V3/ROOT_ONLY/PREFILL_SEAL/SHA256"
_DECODE_INIT_DOMAIN = b"VERATHOS/PROOF_V3/ROOT_ONLY/DECODE_INIT/SHA256"
_DECODE_EVENT_DOMAIN = b"VERATHOS/PROOF_V3/ROOT_ONLY/DECODE_EVENT/SHA256"
_DECODE_SEAL_DOMAIN = b"VERATHOS/PROOF_V3/ROOT_ONLY/DECODE_SEAL/SHA256"
_EXECUTION_ROOT_DOMAIN = b"VERATHOS/PROOF_V3/ROOT_ONLY/EXECUTION/SHA256"
_CACHE_TABLE_ROOT_DOMAIN = b"VERATHOS/PROOF_V3/ROOT_ONLY/CACHE_TABLES/SHA256"
_CACHE_TABLE_INIT_DOMAIN = b"VERATHOS/PROOF_V3/ROOT_ONLY/CACHE_TABLE_INIT/SHA256"
_CACHE_TABLE_EVENT_DOMAIN = b"VERATHOS/PROOF_V3/ROOT_ONLY/CACHE_TABLE_EVENT/SHA256"
_CHUNK_RECORD_DOMAIN = b"VERATHOS/PROOF_V3/ROOT_ONLY/CHUNK/SHA256"
_CHUNK_CACHE_TABLE_ROOT_DOMAIN = (
    b"VERATHOS/PROOF_V3/ROOT_ONLY/CHUNK_CACHE_TABLES/SHA256"
)
_POSTCOMMIT_OPENING_TICKET_DOMAIN = (
    b"VERATHOS/PROOF_V3/POSTCOMMIT_OPENING_TICKET/SHA256"
)
_STAGE_RECORD_DOMAIN = b"VERATHOS/PROOF_V3/ROOT_ONLY/STAGE/SHA256"
_CACHE_PAGE_RECORD_DOMAIN = b"VERATHOS/PROOF_V3/ROOT_ONLY/CACHE_PAGE/SHA256"
_CACHE_KIND_CODES = {"key": 1, "value": 2, "state": 3}
_CACHE_ACCESS_CODES = {"write": 1, "read": 2, "snapshot": 3}
_CHUNK_PHASE_CODES = {"prefill": 1, "decode": 2}


def _fixed32(value: bytes, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
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


def _identifier(value: str, name: str) -> bytes:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ProofV3Error(f"{name} is not a canonical identifier")
    return value.encode("ascii")


def _hash(domain: bytes, *parts: bytes) -> bytes:
    return hashlib.sha256(domain + b"".join(parts)).digest()


@dataclass(frozen=True, slots=True)
class LogicalKVPageAddressV3:
    """One request-relative logical cache-table page, never a physical pointer.

    ``cache_lease_digest`` is issued by the validator before execution and is
    absorbed into the request precommit. It namespaces every logical cache
    access, so a cache witness from another request cannot be replayed under a
    miner-selected epoch. ``cache_epoch`` remains diagnostic coordination
    metadata only; it is not a freshness boundary. A qualified AIR/cache
    relation must additionally establish complete writes, exact latest-read
    semantics, eviction, and release for this logical address.
    """

    layer_index: int
    cache_lease_digest: bytes
    cache_table_id: str
    cache_kind: str
    access_kind: str
    page_start: int
    page_token_count: int
    sequence_length: int
    cache_version: int
    head_index: int
    channel_start: int
    channel_count: int

    def __post_init__(self) -> None:
        _u32(self.layer_index, "layer_index")
        lease = _fixed32(self.cache_lease_digest, "cache_lease_digest")
        if lease == bytes(32):
            raise ProofV3Error("cache_lease_digest must not be the zero digest")
        _identifier(self.cache_table_id, "cache_table_id")
        if (
            not isinstance(self.cache_kind, str)
            or self.cache_kind not in _CACHE_KIND_CODES
        ):
            raise ProofV3Error("cache_kind must be 'key', 'value', or 'state'")
        if (
            not isinstance(self.access_kind, str)
            or self.access_kind not in _CACHE_ACCESS_CODES
        ):
            raise ProofV3Error("cache access_kind is not supported")
        page_start = _u32(self.page_start, "page_start")
        page_token_count = _u32(
            self.page_token_count,
            "page_token_count",
            positive=True,
        )
        sequence_length = _u32(
            self.sequence_length,
            "sequence_length",
            positive=True,
        )
        _u64(self.cache_version, "cache_version")
        _u32(self.head_index, "head_index")
        channel_start = _u32(self.channel_start, "channel_start")
        channel_count = _u32(self.channel_count, "channel_count", positive=True)
        if page_start + page_token_count > 1 << 32:
            raise ProofV3Error("logical cache page token range overflows")
        if page_start + page_token_count > sequence_length:
            raise ProofV3Error("logical cache page exceeds its sequence length")
        if channel_start + channel_count > 1 << 32:
            raise ProofV3Error("logical cache page channel range overflows")

    def canonical_bytes(self) -> bytes:
        table_id = _identifier(self.cache_table_id, "cache_table_id")
        return (
            self.cache_lease_digest
            + struct.pack("<B", len(table_id))
            + table_id
            + struct.pack(
                "<BB7IQ",
                _CACHE_KIND_CODES[self.cache_kind],
                _CACHE_ACCESS_CODES[self.access_kind],
                self.layer_index,
                self.page_start,
                self.page_token_count,
                self.sequence_length,
                self.head_index,
                self.channel_start,
                self.channel_count,
                self.cache_version,
            )
        )


# New runtime code should use this generic name. The historical alias remains
# for compatibility with the first v3 scaffold and denotes the same logical
# table-page ABI.
LogicalCachePageAddressV3 = LogicalKVPageAddressV3


@dataclass(frozen=True, slots=True)
class PostCommitOpeningTicketV3:
    """One-use validator reveal bound to a frozen retained-trace handle.

    A native GPU sidecar must atomically consume this ticket against its local
    request-generation handle. The ticket deliberately contains no physical
    cache pointer, allocator block ID, or miner-selected freshness value.
    """

    proof_challenge_id: bytes
    precommit_context_digest: bytes
    execution_profile_digest: bytes
    cache_lease_digest: bytes
    commitment_envelope_digest: bytes
    validator_nonce: bytes

    def __post_init__(self) -> None:
        _fixed32(self.proof_challenge_id, "proof_challenge_id")
        _fixed32(self.precommit_context_digest, "precommit_context_digest")
        _fixed32(self.execution_profile_digest, "execution_profile_digest")
        lease = _fixed32(self.cache_lease_digest, "cache_lease_digest")
        _fixed32(self.commitment_envelope_digest, "commitment_envelope_digest")
        nonce = _fixed32(self.validator_nonce, "validator_nonce")
        if lease == bytes(32):
            raise ProofV3Error("cache_lease_digest must not be the zero digest")
        if nonce == bytes(32):
            raise ProofV3Error("validator_nonce must not be the zero digest")

    def digest(self) -> bytes:
        """Return the canonical sidecar-consumption identity for this reveal."""

        return _hash(
            _POSTCOMMIT_OPENING_TICKET_DOMAIN,
            self.proof_challenge_id,
            self.precommit_context_digest,
            self.execution_profile_digest,
            self.cache_lease_digest,
            self.commitment_envelope_digest,
            self.validator_nonce,
        )


@dataclass(frozen=True, slots=True)
class ExecutionChunkCommitmentV3:
    """One bounded GPU-sidecar chunk emitted before nonce disclosure.

    This is a fixed-size trace commitment, not a proof. A qualified native
    adapter must establish every signed relation behind these roots, including
    complete tile coverage and cache RAM semantics. The host-side lifecycle can
    nevertheless reject gaps, reordering, phase changes, and cross-chunk state
    discontinuities without retaining an activation or K/V prefix.
    """

    phase: str
    chunk_index: int
    logical_token_start: int
    token_count: int
    cache_lease_digest: bytes
    entry_state_root: bytes
    exit_state_root: bytes
    linear_table_root: bytes
    bridge_table_root: bytes
    transition_table_root: bytes
    cache_table_root: bytes
    sidecar_generation_digest: bytes | None = None
    scheduler_coverage_digest: bytes | None = None

    def __post_init__(self) -> None:
        if self.phase not in _CHUNK_PHASE_CODES:
            raise ProofV3Error("execution chunk phase is not supported")
        _u32(self.chunk_index, "execution chunk_index")
        _u32(self.logical_token_start, "execution chunk logical_token_start")
        _u32(self.token_count, "execution chunk token_count", positive=True)
        if self.logical_token_start + self.token_count > 1 << 32:
            raise ProofV3Error("execution chunk logical token range overflows")
        if _fixed32(
            self.cache_lease_digest, "execution chunk cache_lease_digest"
        ) == bytes(32):
            raise ProofV3Error(
                "execution chunk cache_lease_digest must not be the zero digest"
            )
        for value, name in (
            (self.entry_state_root, "execution chunk entry_state_root"),
            (self.exit_state_root, "execution chunk exit_state_root"),
            (self.linear_table_root, "execution chunk linear_table_root"),
            (self.bridge_table_root, "execution chunk bridge_table_root"),
            (self.transition_table_root, "execution chunk transition_table_root"),
            (self.cache_table_root, "execution chunk cache_table_root"),
        ):
            if _fixed32(value, name) == bytes(32):
                raise ProofV3Error(f"{name} must not be the zero digest")
        if (self.sidecar_generation_digest is None) != (
            self.scheduler_coverage_digest is None
        ):
            raise ProofV3Error(
                "execution chunk must include both generation and scheduler coverage digests"
            )
        if self.sidecar_generation_digest is not None:
            _fixed32(
                self.sidecar_generation_digest,
                "execution chunk sidecar_generation_digest",
            )
            _fixed32(
                self.scheduler_coverage_digest,
                "execution chunk scheduler_coverage_digest",
            )
            if self.sidecar_generation_digest == bytes(32):
                raise ProofV3Error(
                    "execution chunk sidecar_generation_digest must not be the zero digest"
                )
            if self.scheduler_coverage_digest == bytes(32):
                raise ProofV3Error(
                    "execution chunk scheduler_coverage_digest must not be the zero digest"
                )

    @property
    def has_sidecar_scheduler_binding(self) -> bool:
        """Whether this receipt includes the native sidecar control binding.

        The root-only scaffold retains compatibility with old unbound receipts
        for isolated plumbing tests, but a native sidecar session requires this
        property to be true for every accepted chunk.
        """

        return self.sidecar_generation_digest is not None

    def canonical_bytes(self) -> bytes:
        if self.sidecar_generation_digest is None:
            sidecar_binding = b"\x00"
        else:
            # ``__post_init__`` rejects a partial binding pair.
            assert self.scheduler_coverage_digest is not None
            sidecar_binding = (
                b"\x01"
                + self.sidecar_generation_digest
                + self.scheduler_coverage_digest
            )
        return (
            struct.pack(
                "<BIII",
                _CHUNK_PHASE_CODES[self.phase],
                self.chunk_index,
                self.logical_token_start,
                self.token_count,
            )
            + self.cache_lease_digest
            + self.entry_state_root
            + self.exit_state_root
            + self.linear_table_root
            + self.bridge_table_root
            + self.transition_table_root
            + self.cache_table_root
            + sidecar_binding
        )

    def digest(self) -> bytes:
        """Return the canonical root-only record identity for this chunk."""

        return _hash(_CHUNK_RECORD_DOMAIN, self.canonical_bytes())


@dataclass(frozen=True, slots=True)
class ExecutionAccumulatorCommitmentV3:
    """Sealed roots emitted before the validator reveals the hard-audit nonce."""

    static_manifest_digest: bytes
    execution_profile_digest: bytes
    precommit_context_digest: bytes
    prompt_token_root: bytes
    sampler_config_digest: bytes
    cache_lease_digest: bytes
    request_digest: bytes
    output_token_root: bytes
    output_stream_digest: bytes
    execution_root: bytes
    prefill_root: bytes
    cache_table_root: bytes
    context_token_count: int
    decode_token_count: int
    cache_epoch: int

    def __post_init__(self) -> None:
        # Reuse the strict envelope validation rather than duplicating bounds.
        ProofV3CommitmentEnvelope(
            static_manifest_digest=self.static_manifest_digest,
            execution_profile_digest=self.execution_profile_digest,
            precommit_context_digest=self.precommit_context_digest,
            prompt_token_root=self.prompt_token_root,
            sampler_config_digest=self.sampler_config_digest,
            cache_lease_digest=self.cache_lease_digest,
            request_digest=self.request_digest,
            output_token_root=self.output_token_root,
            output_stream_digest=self.output_stream_digest,
            execution_root=self.execution_root,
            prefill_root=self.prefill_root,
            cache_table_root=self.cache_table_root,
            context_token_count=self.context_token_count,
            decode_token_count=self.decode_token_count,
            cache_epoch=self.cache_epoch,
        )

    def bind(
        self,
        *,
        profile: ExecutionSecurityProfileV3,
    ) -> ProofV3CommitmentEnvelope:
        """Confirm the originating profile and return its fixed envelope.

        A sealed root must never be rewrapped under another profile or request.
        Both digests therefore originate at :meth:`begin` and are immutable here.
        """

        if not isinstance(profile, ExecutionSecurityProfileV3):
            raise ProofV3Error("profile has an unexpected type")
        if profile.static_manifest_digest != self.static_manifest_digest:
            raise ProofV3Error("sealed accumulator has an unexpected static manifest")
        if profile.digest() != self.execution_profile_digest:
            raise ProofV3Error("sealed accumulator has an unexpected execution profile")
        return ProofV3CommitmentEnvelope(
            static_manifest_digest=self.static_manifest_digest,
            execution_profile_digest=self.execution_profile_digest,
            precommit_context_digest=self.precommit_context_digest,
            prompt_token_root=self.prompt_token_root,
            sampler_config_digest=self.sampler_config_digest,
            cache_lease_digest=self.cache_lease_digest,
            request_digest=self.request_digest,
            output_token_root=self.output_token_root,
            output_stream_digest=self.output_stream_digest,
            execution_root=self.execution_root,
            prefill_root=self.prefill_root,
            cache_table_root=self.cache_table_root,
            context_token_count=self.context_token_count,
            decode_token_count=self.decode_token_count,
            cache_epoch=self.cache_epoch,
        )


class ExecutionTraceAccumulatorV3(ABC):
    """Required bounded runtime interface for a global execution proof.

    Every baseline root must be accumulated during serving, before the nonce.
    Only selected openings and the proof serialization may happen afterwards.
    """

    @abstractmethod
    def begin(
        self,
        *,
        profile: ExecutionSecurityProfileV3,
        precommit_context: PreExecutionRequestContextV3,
        cache_epoch: int,
    ) -> None:
        """Start one request-bound trace before prefill begins."""

    @abstractmethod
    def accumulate_stage(
        self,
        *,
        stage_id: str,
        token_index: int,
        stage_root: bytes,
    ) -> None:
        """Accumulate one request-relative logical-stage commitment.

        A qualified adapter must additionally enforce its signed graph's exact
        stage ordering and coverage; this interface only gives it a bounded,
        unambiguous logical position.
        """

    @abstractmethod
    def accumulate_chunk(self, *, chunk: ExecutionChunkCommitmentV3) -> None:
        """Absorb one contiguous native sidecar chunk before nonce reveal.

        Implementations must keep raw tensor/proof state native and bounded;
        only this fixed-size receipt may cross into the control plane.
        """

    @abstractmethod
    def accumulate_kv_pages(
        self,
        *,
        address: LogicalKVPageAddressV3,
        page_root: bytes,
    ) -> None:
        """Accumulate one logical paged-KV commitment without host copies."""

    @abstractmethod
    def seal_prefill(self, *, context_token_count: int) -> bytes:
        """Seal the captured prompt/prefill root.

        Only a qualified adapter may call this a complete provenance proof.
        """

    @abstractmethod
    def seal_decode(
        self,
        *,
        decode_token_count: int,
        output_binding: ExecutionOutputBindingV3,
    ) -> ExecutionAccumulatorCommitmentV3:
        """Seal captured roots before the validator nonce is revealed."""

    @abstractmethod
    def freeze_precommit(self) -> None:
        """Forbid post-nonce mutation of any committed witness root."""

    @abstractmethod
    def open_postnonce(self, *, ticket: PostCommitOpeningTicketV3) -> bytes:
        """Consume one validator-bound ticket to produce retained openings."""

    @abstractmethod
    def release(self) -> None:
        """Release the request's state/cache retention lease."""


class RootOnlyExecutionTraceAccumulatorV3(ExecutionTraceAccumulatorV3):
    """Bounded ordered-root plumbing, deliberately without witness openings.

    This reference object retains only fixed-size hashes and counters; it never
    retains raw activations or cache pages. It establishes the exact lifecycle
    expected from a future GPU sidecar, but cannot produce an AIR/PCS opening
    after the nonce, does not establish graph coverage, and therefore cannot be
    used as a proof-v3 adapter.
    """

    def __init__(self) -> None:
        self._state = "new"
        self._context: bytes | None = None
        self._prefill_chain: bytes | None = None
        self._decode_chain: bytes | None = None
        self._prefill_root: bytes | None = None
        self._sealed: ExecutionAccumulatorCommitmentV3 | None = None
        self._prefill_record_count = 0
        self._decode_record_count = 0
        self._cache_epoch = 0
        self._context_token_count = 0
        self._max_verified_context_tokens = 0
        self._max_verified_decode_tokens = 0
        self._precommit_context: PreExecutionRequestContextV3 | None = None
        self._prefill_max_logical_token_exclusive = 0
        self._decode_max_logical_token_exclusive = 0
        self._cache_table_bindings: dict[str, tuple[int, str]] = {}
        self._cache_page_token_count = 0
        self._cache_max_tokens = 0
        self._cache_latest_versions: dict[tuple[int, str], int] = {}
        self._cache_final_lengths: dict[tuple[int, str], int] = {}
        self._cache_written_tables: set[tuple[int, str]] = set()
        self._cache_table_chains: dict[tuple[int, str], bytes] = {}
        self._profile: ExecutionSecurityProfileV3 | None = None
        self._capture_mode: str | None = None
        self._next_chunk_index = 0
        self._next_chunk_token_start = 0
        self._initial_execution_state_anchor: bytes | None = None
        self._last_chunk_exit_state_root: bytes | None = None
        self._last_chunk_cache_table_root: bytes | None = None
        self._chunk_sidecar_binding_mode: bool | None = None
        self._chunk_sidecar_generation_digest: bytes | None = None

    def _require_state(self, *expected: str) -> None:
        if self._state not in expected:
            joined = ", ".join(expected)
            raise ProofV3Error(
                f"root-only accumulator is in {self._state!r} state, expected {joined}"
            )

    def _require_capture_mode(self, mode: str) -> None:
        if self._capture_mode is None:
            self._capture_mode = mode
            return
        if self._capture_mode != mode:
            raise ProofV3Error(
                "root-only accumulator cannot mix chunk receipts with per-record capture"
            )

    def _append(self, record_digest: bytes) -> None:
        _fixed32(record_digest, "execution record digest")
        if self._state == "prefill":
            if self._prefill_record_count >= MAX_ROOT_ONLY_EXECUTION_RECORDS:
                raise ProofV3Error("prefill root-only record count is out of range")
            assert self._prefill_chain is not None
            self._prefill_chain = _hash(
                _PREFILL_EVENT_DOMAIN,
                self._prefill_chain,
                record_digest,
            )
            self._prefill_record_count += 1
            return
        if self._state == "decode":
            if self._decode_record_count >= MAX_ROOT_ONLY_EXECUTION_RECORDS:
                raise ProofV3Error("decode root-only record count is out of range")
            assert self._decode_chain is not None
            self._decode_chain = _hash(
                _DECODE_EVENT_DOMAIN,
                self._decode_chain,
                record_digest,
            )
            self._decode_record_count += 1
            return
        raise ProofV3Error("execution records must be appended before sealing")

    def begin(
        self,
        *,
        profile: ExecutionSecurityProfileV3,
        precommit_context: PreExecutionRequestContextV3,
        cache_epoch: int,
    ) -> None:
        self._require_state("new")
        if not isinstance(profile, ExecutionSecurityProfileV3):
            raise ProofV3Error("profile has an unexpected type")
        if not isinstance(precommit_context, PreExecutionRequestContextV3):
            raise ProofV3Error("precommit context has an unexpected type")
        profile.require_hard_execution_capability()
        if (
            profile.relation_spec.recursive_accumulator_abi_id
            != RECURSIVE_CHUNK_ACCUMULATOR_ABI_V3
        ):
            raise ProofV3Error("root-only accumulator has an unsupported chunk ABI")
        epoch = _u64(cache_epoch, "cache_epoch")
        profile_digest = profile.digest()
        if precommit_context.static_manifest_digest != profile.static_manifest_digest:
            raise ProofV3Error("precommit context has an unexpected static manifest")
        if precommit_context.execution_profile_digest != profile_digest:
            raise ProofV3Error("precommit context has an unexpected execution profile")
        if precommit_context.context_token_count > profile.max_verified_context_tokens:
            raise ProofV3Error("context length exceeds signed profile limit")
        context = _hash(
            _ROOT_CONTEXT_DOMAIN,
            profile_digest,
            precommit_context.digest(),
            struct.pack("<Q", epoch),
        )
        self._context = context
        self._prefill_chain = _hash(_PREFILL_INIT_DOMAIN, context)
        self._cache_epoch = epoch
        self._max_verified_context_tokens = profile.max_verified_context_tokens
        self._max_verified_decode_tokens = profile.max_verified_decode_tokens
        self._precommit_context = precommit_context
        self._initial_execution_state_anchor = (
            precommit_context.initial_execution_state_anchor()
        )
        self._profile = profile
        bindings: dict[str, tuple[int, str]] = {}
        for family in profile.relation_spec.cache.layer_caches:
            if family.cache_kind == "attention_kv":
                assert family.key_tensor_id is not None
                assert family.value_tensor_id is not None
                bindings[family.key_tensor_id] = (family.layer_index, "key")
                bindings[family.value_tensor_id] = (family.layer_index, "value")
            else:
                for table_id in family.state_tensor_ids:
                    bindings[table_id] = (family.layer_index, "state")
        if not bindings:
            raise ProofV3Error("signed cache relation has no logical tables")
        self._cache_table_bindings = bindings
        self._cache_page_token_count = profile.relation_spec.cache.page_token_count
        self._cache_max_tokens = profile.relation_spec.cache.max_cache_tokens
        self._cache_latest_versions = {}
        self._cache_final_lengths = {}
        self._cache_written_tables = set()
        self._cache_table_chains = {
            (layer_index, table_id): _hash(
                _CACHE_TABLE_INIT_DOMAIN,
                context,
                struct.pack("<I", layer_index),
                _identifier(table_id, "cache_table_id"),
            )
            for table_id, (layer_index, _cache_kind) in bindings.items()
        }
        self._state = "prefill"

    def accumulate_stage(
        self,
        *,
        stage_id: str,
        token_index: int,
        stage_root: bytes,
    ) -> None:
        self._require_state("prefill", "decode")
        self._require_capture_mode("records")
        stage = _identifier(stage_id, "stage_id")
        token = _u32(token_index, "token_index")
        root = _fixed32(stage_root, "stage_root")
        logical_end = token + 1
        if self._state == "prefill":
            if logical_end > self._max_verified_context_tokens:
                raise ProofV3Error("prefill stage index exceeds signed profile limit")
            self._prefill_max_logical_token_exclusive = max(
                self._prefill_max_logical_token_exclusive,
                logical_end,
            )
        else:
            if token < self._context_token_count:
                raise ProofV3Error("decode stage index precedes the prompt boundary")
            if logical_end > (
                self._max_verified_context_tokens + self._max_verified_decode_tokens
            ):
                raise ProofV3Error("decode stage index exceeds signed profile limit")
            self._decode_max_logical_token_exclusive = max(
                self._decode_max_logical_token_exclusive,
                logical_end,
            )
        record = _hash(
            _STAGE_RECORD_DOMAIN,
            struct.pack("<I", token),
            struct.pack("<B", len(stage)),
            stage,
            root,
        )
        self._append(record)

    def accumulate_chunk(self, *, chunk: ExecutionChunkCommitmentV3) -> None:
        """Absorb one contiguous fixed-size sidecar receipt.

        The reference ledger verifies only public lifecycle/coverage mechanics.
        It does not inspect tensors, derive cache page roots, or establish the
        computation represented by any root.
        """

        self._require_state("prefill", "decode")
        self._require_capture_mode("chunks")
        if not isinstance(chunk, ExecutionChunkCommitmentV3):
            raise ProofV3Error("execution chunk has an unexpected type")
        has_sidecar_binding = chunk.has_sidecar_scheduler_binding
        if self._chunk_sidecar_binding_mode is None:
            self._chunk_sidecar_binding_mode = has_sidecar_binding
        elif self._chunk_sidecar_binding_mode != has_sidecar_binding:
            raise ProofV3Error(
                "execution chunks cannot mix sidecar-bound and unbound receipts"
            )
        if has_sidecar_binding:
            assert chunk.sidecar_generation_digest is not None
            if self._chunk_sidecar_generation_digest is None:
                self._chunk_sidecar_generation_digest = chunk.sidecar_generation_digest
            elif (
                self._chunk_sidecar_generation_digest != chunk.sidecar_generation_digest
            ):
                raise ProofV3Error(
                    "execution chunks cannot mix sidecar request generations"
                )
        if chunk.phase != self._state:
            raise ProofV3Error("execution chunk phase does not match the lifecycle")
        if chunk.chunk_index != self._next_chunk_index:
            raise ProofV3Error("execution chunks must have contiguous indices")
        if chunk.logical_token_start != self._next_chunk_token_start:
            raise ProofV3Error("execution chunks must have contiguous token ranges")
        assert self._precommit_context is not None
        if chunk.cache_lease_digest != self._precommit_context.cache_lease_digest:
            raise ProofV3Error("execution chunk has an unexpected cache lease")
        assert self._profile is not None
        maximum = (
            self._profile.relation_spec.prefill_chunk_tokens
            if chunk.phase == "prefill"
            else self._profile.relation_spec.decode_chunk_tokens
        )
        if chunk.token_count > maximum:
            raise ProofV3Error("execution chunk exceeds the signed chunk limit")
        logical_end = chunk.logical_token_start + chunk.token_count
        if chunk.phase == "prefill":
            if logical_end > self._precommit_context.context_token_count:
                raise ProofV3Error(
                    "prefill chunk exceeds the validator prompt boundary"
                )
            self._prefill_max_logical_token_exclusive = logical_end
        else:
            if chunk.logical_token_start < self._context_token_count:
                raise ProofV3Error("decode chunk precedes the prompt boundary")
            if logical_end > (
                self._context_token_count + self._max_verified_decode_tokens
            ):
                raise ProofV3Error("decode chunk exceeds the signed profile limit")
            self._decode_max_logical_token_exclusive = logical_end
        if self._last_chunk_exit_state_root is None:
            assert self._initial_execution_state_anchor is not None
            if chunk.entry_state_root != self._initial_execution_state_anchor:
                raise ProofV3Error(
                    "first prefill chunk entry state is not anchored to validator context"
                )
        elif chunk.entry_state_root != self._last_chunk_exit_state_root:
            raise ProofV3Error("execution chunk state roots are discontinuous")
        self._append(chunk.digest())
        self._next_chunk_index += 1
        self._next_chunk_token_start = logical_end
        self._last_chunk_exit_state_root = chunk.exit_state_root
        self._last_chunk_cache_table_root = chunk.cache_table_root

    def accumulate_kv_pages(
        self,
        *,
        address: LogicalKVPageAddressV3,
        page_root: bytes,
    ) -> None:
        self._require_state("prefill", "decode")
        self._require_capture_mode("records")
        if not isinstance(address, LogicalKVPageAddressV3):
            raise ProofV3Error("logical K/V page address has an unexpected type")
        assert self._precommit_context is not None
        if address.cache_lease_digest != self._precommit_context.cache_lease_digest:
            raise ProofV3Error("logical K/V page has an unexpected cache lease")
        expected_table = self._cache_table_bindings.get(address.cache_table_id)
        if expected_table is None:
            raise ProofV3Error("logical cache page has an undeclared cache table")
        if expected_table != (address.layer_index, address.cache_kind):
            raise ProofV3Error("logical cache page does not match its signed table")
        if address.page_token_count > self._cache_page_token_count:
            raise ProofV3Error("logical cache page exceeds the signed page size")
        if address.page_start % self._cache_page_token_count:
            raise ProofV3Error("logical cache page start is not canonically aligned")
        if address.sequence_length > self._cache_max_tokens:
            raise ProofV3Error("logical cache page exceeds the signed cache capacity")
        root = _fixed32(page_root, "page_root")
        version_key = (address.layer_index, address.cache_table_id)
        prior_version = self._cache_latest_versions.get(version_key)
        if prior_version is not None and address.cache_version < prior_version:
            raise ProofV3Error("logical cache page version regresses")
        if (
            address.access_kind in {"read", "snapshot"}
            and version_key not in self._cache_written_tables
        ):
            raise ProofV3Error("logical cache read has no prior signed table write")
        self._cache_latest_versions[version_key] = max(
            address.cache_version,
            -1 if prior_version is None else prior_version,
        )
        if address.access_kind == "write":
            self._cache_written_tables.add(version_key)
            self._cache_final_lengths[version_key] = max(
                address.sequence_length,
                self._cache_final_lengths.get(version_key, 0),
            )
        self._cache_table_chains[version_key] = _hash(
            _CACHE_TABLE_EVENT_DOMAIN,
            self._cache_table_chains[version_key],
            address.canonical_bytes(),
            root,
        )
        logical_end = address.sequence_length
        if self._state == "prefill":
            if logical_end > self._max_verified_context_tokens:
                raise ProofV3Error("prefill K/V range exceeds signed profile limit")
            self._prefill_max_logical_token_exclusive = max(
                self._prefill_max_logical_token_exclusive,
                logical_end,
            )
        else:
            if logical_end > (
                self._max_verified_context_tokens + self._max_verified_decode_tokens
            ):
                raise ProofV3Error("decode K/V range exceeds signed profile limit")
            self._decode_max_logical_token_exclusive = max(
                self._decode_max_logical_token_exclusive,
                logical_end,
            )
        record = _hash(
            _CACHE_PAGE_RECORD_DOMAIN,
            address.canonical_bytes(),
            root,
        )
        self._append(record)

    def seal_prefill(self, *, context_token_count: int) -> bytes:
        self._require_state("prefill")
        context_count = _u32(
            context_token_count,
            "context_token_count",
            positive=True,
        )
        if context_count > self._max_verified_context_tokens:
            raise ProofV3Error("context length exceeds signed profile limit")
        assert self._precommit_context is not None
        if context_count != self._precommit_context.context_token_count:
            raise ProofV3Error(
                "context length differs from validator precommit context"
            )
        if self._capture_mode != "chunks":
            raise ProofV3UnavailableError(
                "per-record capture has no signed complete coverage rule; bounded chunks are required"
            )
        if self._capture_mode == "chunks" and (
            self._next_chunk_token_start != context_count
            or self._prefill_max_logical_token_exclusive != context_count
        ):
            raise ProofV3Error(
                "prefill chunks do not cover the validator prompt contiguously"
            )
        if self._prefill_max_logical_token_exclusive > context_count:
            raise ProofV3Error("prefill record index exceeds declared context length")
        if self._prefill_record_count == 0:
            raise ProofV3Error("prefill root cannot be sealed without records")
        assert self._context is not None
        assert self._prefill_chain is not None
        self._prefill_root = _hash(
            _PREFILL_SEAL_DOMAIN,
            self._context,
            struct.pack("<IQ", context_count, self._prefill_record_count),
            self._prefill_chain,
        )
        self._context_token_count = context_count
        if self._capture_mode == "chunks":
            # Decode coordinates continue the same logical sequence. This also
            # gives an empty observed decode a canonical covered endpoint.
            self._decode_max_logical_token_exclusive = context_count
        self._decode_chain = _hash(
            _DECODE_INIT_DOMAIN,
            self._context,
            self._prefill_root,
        )
        self._state = "decode"
        return self._prefill_root

    def _seal_cache_table_root(self) -> bytes:
        """Commit the exact signed table inventory and latest observed state.

        This root is an inventory/lifecycle commitment only. It does not prove
        RAM correctness, cache coverage, or transition provenance; a native
        AIR/RAM adapter must establish those claims before accepting a proof.
        """

        assert self._context is not None
        if self._capture_mode == "chunks":
            if self._last_chunk_cache_table_root is None:
                raise ProofV3Error("chunk trace has no final cache-table root")
            return _hash(
                _CHUNK_CACHE_TABLE_ROOT_DOMAIN,
                self._context,
                struct.pack("<I", self._next_chunk_index),
                self._last_chunk_cache_table_root,
            )
        entries = bytearray()
        for table_id, (layer_index, cache_kind) in sorted(
            self._cache_table_bindings.items()
        ):
            encoded_table_id = _identifier(table_id, "cache_table_id")
            key = (layer_index, table_id)
            entries.extend(struct.pack("<B", len(encoded_table_id)))
            entries.extend(encoded_table_id)
            entries.extend(
                struct.pack(
                    "<IBIQB",
                    layer_index,
                    _CACHE_KIND_CODES[cache_kind],
                    self._cache_final_lengths.get(key, 0),
                    self._cache_latest_versions.get(key, 0),
                    1 if key in self._cache_written_tables else 0,
                )
            )
            entries.extend(self._cache_table_chains[key])
        return _hash(
            _CACHE_TABLE_ROOT_DOMAIN,
            self._context,
            struct.pack("<I", len(self._cache_table_bindings)),
            bytes(entries),
        )

    def seal_decode(
        self,
        *,
        decode_token_count: int,
        output_binding: ExecutionOutputBindingV3,
    ) -> ExecutionAccumulatorCommitmentV3:
        self._require_state("decode")
        decode_count = _u32(
            decode_token_count,
            "decode_token_count",
        )
        if decode_count > self._max_verified_decode_tokens:
            raise ProofV3Error("decode length exceeds signed profile limit")
        if not isinstance(output_binding, ExecutionOutputBindingV3):
            raise ProofV3Error("output binding has an unexpected type")
        if decode_count != output_binding.decode_token_count:
            raise ProofV3Error("decode length differs from output token binding")
        if self._capture_mode != "chunks":
            raise ProofV3UnavailableError(
                "per-record capture has no signed complete coverage rule; bounded chunks are required"
            )
        if self._capture_mode == "chunks" and (
            self._next_chunk_token_start != self._context_token_count + decode_count
            or self._decode_max_logical_token_exclusive
            != self._context_token_count + decode_count
        ):
            raise ProofV3Error(
                "decode chunks do not cover the observed output contiguously"
            )
        if self._decode_max_logical_token_exclusive > (
            self._context_token_count + decode_count
        ):
            raise ProofV3Error("decode record index exceeds declared sequence length")
        if decode_count == 0:
            if self._decode_record_count != 0 or (
                self._decode_max_logical_token_exclusive > self._context_token_count
            ):
                raise ProofV3Error(
                    "empty decode output must not contain decode-stage records"
                )
        elif self._decode_record_count == 0:
            raise ProofV3Error("decode root cannot be sealed without records")
        assert self._context is not None
        assert self._prefill_root is not None
        assert self._decode_chain is not None
        assert self._precommit_context is not None
        request_binding = self._precommit_context.bind_output(output_binding)
        decode_root = _hash(
            _DECODE_SEAL_DOMAIN,
            self._context,
            struct.pack("<IQ", decode_count, self._decode_record_count),
            self._decode_chain,
        )
        execution_root = _hash(
            _EXECUTION_ROOT_DOMAIN,
            self._context,
            self._prefill_root,
            decode_root,
            struct.pack("<QQ", self._prefill_record_count, self._decode_record_count),
        )
        cache_table_root = self._seal_cache_table_root()
        self._sealed = ExecutionAccumulatorCommitmentV3(
            static_manifest_digest=self._precommit_context.static_manifest_digest,
            execution_profile_digest=self._precommit_context.execution_profile_digest,
            precommit_context_digest=self._precommit_context.digest(),
            prompt_token_root=self._precommit_context.prompt_token_root,
            sampler_config_digest=self._precommit_context.sampler_config_digest,
            cache_lease_digest=self._precommit_context.cache_lease_digest,
            request_digest=request_binding.request_digest,
            output_token_root=output_binding.output_token_root,
            output_stream_digest=output_binding.output_stream_digest,
            execution_root=execution_root,
            prefill_root=self._prefill_root,
            cache_table_root=cache_table_root,
            context_token_count=self._context_token_count,
            decode_token_count=decode_count,
            cache_epoch=self._cache_epoch,
        )
        self._state = "sealed"
        return self._sealed

    def freeze_precommit(self) -> None:
        self._require_state("sealed")
        self._state = "frozen"

    def open_postnonce(self, *, ticket: PostCommitOpeningTicketV3) -> bytes:
        self._require_state("frozen")
        if not isinstance(ticket, PostCommitOpeningTicketV3):
            raise ProofV3Error("postcommit opening ticket has an unexpected type")
        assert self._sealed is not None
        assert self._precommit_context is not None
        if ticket.proof_challenge_id != self._precommit_context.proof_challenge_id:
            raise ProofV3Error("postcommit opening ticket has an unexpected challenge")
        if ticket.precommit_context_digest != self._sealed.precommit_context_digest:
            raise ProofV3Error("postcommit opening ticket has an unexpected precommit")
        if ticket.execution_profile_digest != self._sealed.execution_profile_digest:
            raise ProofV3Error("postcommit opening ticket has an unexpected profile")
        if ticket.cache_lease_digest != self._sealed.cache_lease_digest:
            raise ProofV3Error(
                "postcommit opening ticket has an unexpected cache lease"
            )
        assert self._profile is not None
        if (
            ticket.commitment_envelope_digest
            != self._sealed.bind(profile=self._profile).digest()
        ):
            raise ProofV3Error("postcommit opening ticket has an unexpected envelope")
        raise ProofV3UnavailableError(
            "root-only accumulator has no retained post-nonce witness openings"
        )

    def release(self) -> None:
        if self._state == "released":
            return
        self._context = None
        self._prefill_chain = None
        self._decode_chain = None
        self._prefill_root = None
        self._sealed = None
        self._prefill_record_count = 0
        self._decode_record_count = 0
        self._cache_epoch = 0
        self._context_token_count = 0
        self._max_verified_context_tokens = 0
        self._max_verified_decode_tokens = 0
        self._precommit_context = None
        self._prefill_max_logical_token_exclusive = 0
        self._decode_max_logical_token_exclusive = 0
        self._cache_table_bindings = {}
        self._cache_page_token_count = 0
        self._cache_max_tokens = 0
        self._cache_latest_versions = {}
        self._cache_final_lengths = {}
        self._cache_written_tables = set()
        self._cache_table_chains = {}
        self._profile = None
        self._capture_mode = None
        self._next_chunk_index = 0
        self._next_chunk_token_start = 0
        self._initial_execution_state_anchor = None
        self._last_chunk_exit_state_root = None
        self._last_chunk_cache_table_root = None
        self._chunk_sidecar_binding_mode = None
        self._chunk_sidecar_generation_digest = None
        self._state = "released"


class UnavailableExecutionTraceAccumulatorV3(ExecutionTraceAccumulatorV3):
    """Explicit fail-closed implementation used until a native adapter exists."""

    @staticmethod
    def _raise() -> None:
        raise ProofV3UnavailableError(
            "proof-v3 runtime execution accumulator is not implemented"
        )

    def begin(
        self,
        *,
        profile: ExecutionSecurityProfileV3,
        precommit_context: PreExecutionRequestContextV3,
        cache_epoch: int,
    ) -> None:
        self._raise()

    def accumulate_stage(
        self,
        *,
        stage_id: str,
        token_index: int,
        stage_root: bytes,
    ) -> None:
        self._raise()

    def accumulate_chunk(self, *, chunk: ExecutionChunkCommitmentV3) -> None:
        self._raise()

    def accumulate_kv_pages(
        self,
        *,
        address: LogicalKVPageAddressV3,
        page_root: bytes,
    ) -> None:
        self._raise()

    def seal_prefill(self, *, context_token_count: int) -> bytes:
        self._raise()

    def seal_decode(
        self,
        *,
        decode_token_count: int,
        output_binding: ExecutionOutputBindingV3,
    ) -> ExecutionAccumulatorCommitmentV3:
        self._raise()

    def freeze_precommit(self) -> None:
        self._raise()

    def open_postnonce(self, *, ticket: PostCommitOpeningTicketV3) -> bytes:
        self._raise()

    def release(self) -> None:
        self._raise()


__all__ = [
    "ExecutionChunkCommitmentV3",
    "ExecutionAccumulatorCommitmentV3",
    "ExecutionTraceAccumulatorV3",
    "LogicalCachePageAddressV3",
    "LogicalKVPageAddressV3",
    "MAX_ROOT_ONLY_EXECUTION_RECORDS",
    "RootOnlyExecutionTraceAccumulatorV3",
    "PostCommitOpeningTicketV3",
    "UnavailableExecutionTraceAccumulatorV3",
]
