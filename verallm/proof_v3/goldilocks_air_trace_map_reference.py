"""Authenticated pre-nonce trace-map reference for V3 Goldilocks AIR.

The one-coordinate AIR receipt harness proves only that one validator-owned
root was retained before nonce release.  A predictable coordinate still lets a
prover prepare only that trace.  This separate, unregistered CPU reference
adds the missing map shape:

* derive the full lazy trace-slot universe from the signed profile and the
  validator-accepted envelope;
* require one frozen trace-LDE precommitment for every exact
  ``(layout, phase, chunk)`` slot before the nonce; and
* select exact slots from the nonce without incorporating the miner-controlled
  map root, then verify Merkle openings for those already committed entries.

Slots deliberately aggregate all atomic AIR constraints for one parsed
program/trace.  This avoids duplicating the same trace root for every atomic
constraint while still proving the complete program when that slot is opened.
The reference tree retains every row only for conformance tests.  A qualified
backend must build the same canonical map incrementally/streamingly and bind
its root through a validator-owned session receipt before nonce release.
The map root is intentionally a receipt-sidecar commitment, not an
``execution_root`` field in the envelope: this reference's root-independent
slot statement is already bound to the accepted envelope, so putting the map
root back into that envelope would create a digest cycle.

This module is not imported by a payload, miner, validator, adapter registry,
or public package facade.  It is not a runtime-tensor, cache-RAM, attention,
GDN, token, or model-substitution proof on its own.
"""

from __future__ import annotations

import hashlib
import operator
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Final

from verallm.proof_v3.constraint_program import GoldilocksConstraintProgramBundleV3
from verallm.proof_v3.constraint_system import (
    ExpectedGoldilocksConstraintUniverseV3,
    GoldilocksConstraintCoordinateV3,
    GoldilocksExecutionConstraintSystemV3,
    expected_goldilocks_constraint_universe_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_air_reference import (
    GoldilocksAirStatementCoreReferenceV3,
    GoldilocksAirTraceOracleReferenceV3,
    GoldilocksAirTracePrecommitmentReferenceV3,
)
from verallm.proof_v3.goldilocks_merkle_reference import (
    GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES,
    MAX_GOLDILOCKS_MERKLE_REFERENCE_LEAF_COUNT,
    GoldilocksMerkleMultiOpeningReference,
    GoldilocksMerkleTreeReference,
    verify_goldilocks_merkle_multiopening_reference,
)
from verallm.proof_v3.payload import ProofV3CommitmentEnvelope
from verallm.proof_v3.session import QualifiedExecutionProfileV3
from verallm.proof_v3.verifier import EnvelopeBindingV3


GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_ABI_V3: Final = (
    "goldilocks.air_trace_map.reference.v1"
)
GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_FORMAT_VERSION_V3: Final = 1
GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_LEAF_WIDTH_V3: Final = 8
GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_QUERY_COUNT_V3: Final = 2
MAX_GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_REJECTION_ATTEMPTS_V3: Final = 1 << 16

_PHASE_CODES: Final = {"prefill": 1, "decode": 2}
_STATEMENT_BINDING_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR_TRACE_MAP/V1/STATEMENT/SHA256"
)
_STATEMENT_DIGEST_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR_TRACE_MAP/V1/STATEMENT_DIGEST/SHA256"
)
_SLOT_UNIVERSE_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR_TRACE_MAP/V1/SLOT_UNIVERSE/SHA256"
)
_SLOT_CORE_BINDING_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR_TRACE_MAP/V1/SLOT_CORE/SHA256"
)
_TREE_BINDING_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR_TRACE_MAP/V1/TREE_BINDING/SHA256"
)
_PRECOMMITMENT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR_TRACE_MAP/V1/PRECOMMITMENT/SHA256"
)
_SELECTION_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR_TRACE_MAP/V1/POSTCOMMIT_SELECTION/SHA256"
)
_STATEMENT_FACTORY_TOKEN = object()


def _fixed32(value: object, name: str, *, nonzero: bool = False) -> bytes:
    if not isinstance(value, bytes) or len(value) != GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    if nonzero and value == bytes(GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES):
        raise ProofV3Error(f"{name} must not be zero")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ProofV3Error(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ProofV3Error(f"{name} must be an integer") from exc


def _u32(value: object, name: str, *, positive: bool = False) -> int:
    result = _integer(value, name)
    if result < (1 if positive else 0) or result >= 1 << 32:
        raise ProofV3Error(f"{name} must be an unsigned 32-bit integer")
    return result


def _next_power_of_two(value: int, *, name: str) -> int:
    if value < 1:
        raise ProofV3Error(f"{name} must be positive")
    return 1 << (value - 1).bit_length()


def _digest_row(value: object, name: str) -> tuple[int, ...]:
    """Encode arbitrary digest bytes as eight canonical 32-bit field limbs."""

    digest = _fixed32(value, name)
    return tuple(struct.unpack("<8I", digest))


def _row_digest(value: object, name: str) -> bytes:
    if (
        not isinstance(value, tuple)
        or len(value) != GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_LEAF_WIDTH_V3
    ):
        raise ProofV3VerificationError(
            f"{name} must contain exactly {GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_LEAF_WIDTH_V3} limbs"
        )
    limbs: list[int] = []
    for index, limb in enumerate(value):
        integer = _integer(limb, f"{name}[{index}]")
        if integer < 0 or integer >= 1 << 32:
            raise ProofV3VerificationError(f"{name}[{index}] is not a 32-bit limb")
        limbs.append(integer)
    return struct.pack("<8I", *limbs)


def _require_bound_envelope(
    *,
    qualified_profile: QualifiedExecutionProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    envelope_binding: EnvelopeBindingV3,
) -> None:
    """Check the typed inputs a validator coordinator must have accepted."""

    if not isinstance(qualified_profile, QualifiedExecutionProfileV3):
        raise ProofV3VerificationError(
            "Goldilocks AIR trace map requires a qualified execution profile"
        )
    qualified_profile.require_qualification_provenance()
    if not isinstance(envelope, ProofV3CommitmentEnvelope):
        raise ProofV3VerificationError("Goldilocks AIR trace map envelope is malformed")
    if not isinstance(envelope_binding, EnvelopeBindingV3):
        raise ProofV3VerificationError(
            "Goldilocks AIR trace map envelope binding is malformed"
        )
    profile = qualified_profile.profile
    precommit_context = envelope_binding.precommit_context
    request_binding = envelope_binding.request_binding
    if (
        envelope_binding.profile_digest != profile.digest()
        or request_binding.precommit_context != precommit_context
        or envelope.static_manifest_digest != profile.static_manifest_digest
        or envelope.execution_profile_digest != profile.digest()
        or envelope.precommit_context_digest != precommit_context.digest()
        or envelope.prompt_token_root != precommit_context.prompt_token_root
        or envelope.sampler_config_digest != precommit_context.sampler_config_digest
        or envelope.cache_lease_digest != precommit_context.cache_lease_digest
        or envelope.request_digest != request_binding.request_digest
        or envelope.output_token_root != request_binding.output.output_token_root
        or envelope.output_stream_digest != request_binding.output.output_stream_digest
        or envelope.context_token_count != precommit_context.context_token_count
        or envelope.decode_token_count != request_binding.output.decode_token_count
    ):
        raise ProofV3VerificationError(
            "Goldilocks AIR trace map envelope is not validator accepted"
        )


@dataclass(frozen=True, slots=True)
class GoldilocksAirTraceSlotReferenceV3:
    """One complete parsed AIR trace: layout, phase, and execution chunk."""

    layout_index: int
    node_id: str
    phase: str
    chunk_index: int
    logical_token_start: int
    token_count: int

    def __post_init__(self) -> None:
        _u32(self.layout_index, "Goldilocks AIR trace slot layout_index")
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ProofV3Error("Goldilocks AIR trace slot node_id is malformed")
        if self.phase not in _PHASE_CODES:
            raise ProofV3Error("Goldilocks AIR trace slot phase is unsupported")
        _u32(self.chunk_index, "Goldilocks AIR trace slot chunk_index")
        start = _u32(
            self.logical_token_start,
            "Goldilocks AIR trace slot logical_token_start",
        )
        count = _u32(
            self.token_count,
            "Goldilocks AIR trace slot token_count",
            positive=True,
        )
        if start + count > 1 << 32:
            raise ProofV3Error("Goldilocks AIR trace slot token range overflows")

    def canonical_bytes(self) -> bytes:
        try:
            node_id = self.node_id.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ProofV3Error("Goldilocks AIR trace slot node_id is not ASCII") from exc
        if not 1 <= len(node_id) <= 128:
            raise ProofV3Error("Goldilocks AIR trace slot node_id is malformed")
        return (
            struct.pack("<IB", self.layout_index, len(node_id))
            + node_id
            + struct.pack(
                "<BIII",
                _PHASE_CODES[self.phase],
                self.chunk_index,
                self.logical_token_start,
                self.token_count,
            )
        )


@dataclass(frozen=True, slots=True)
class GoldilocksAirTraceSlotUniverseReferenceV3(
    Sequence[GoldilocksAirTraceSlotReferenceV3]
):
    """Lazy exact slot universe with one entry per layout/chunk trace.

    The full constraint universe has one coordinate per atomic constraint.  A
    slot instead covers the whole program for one layout and chunk, so the
    same trace root is not duplicated for every atomic constraint in that
    program.  Its index order is signed-layout order inside canonical prefill
    then decode chunk order.
    """

    profile: object
    envelope: ProofV3CommitmentEnvelope
    constraint_system: GoldilocksExecutionConstraintSystemV3
    _constraint_universe: ExpectedGoldilocksConstraintUniverseV3 = field(
        init=False,
        repr=False,
        compare=False,
    )
    _prefill_layout_indices: tuple[int, ...] = field(init=False, repr=False)
    _decode_layout_indices: tuple[int, ...] = field(init=False, repr=False)
    _prefill_chunk_count: int = field(init=False, repr=False)
    _decode_chunk_count: int = field(init=False, repr=False)
    _length: int = field(init=False, repr=False)
    _binding_digest: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from verallm.proof_v3.profile import ExecutionSecurityProfileV3

        if not isinstance(self.profile, ExecutionSecurityProfileV3):
            raise ProofV3Error("Goldilocks AIR trace slot universe profile is malformed")
        if not isinstance(self.envelope, ProofV3CommitmentEnvelope):
            raise ProofV3Error("Goldilocks AIR trace slot universe envelope is malformed")
        if not isinstance(self.constraint_system, GoldilocksExecutionConstraintSystemV3):
            raise ProofV3Error("Goldilocks AIR trace slot universe system is malformed")
        constraint_universe = expected_goldilocks_constraint_universe_v3(
            profile=self.profile,
            envelope=self.envelope,
            constraint_system=self.constraint_system,
        )
        relation = self.profile.relation_spec
        prefill_layouts = tuple(
            index
            for index, layout in enumerate(self.constraint_system.layouts)
            if "prefill" in layout.phases
        )
        decode_layouts = tuple(
            index
            for index, layout in enumerate(self.constraint_system.layouts)
            if "decode" in layout.phases
        )
        prefill_count = (
            self.envelope.context_token_count + relation.prefill_chunk_tokens - 1
        ) // relation.prefill_chunk_tokens
        decode_count = (
            self.envelope.decode_token_count + relation.decode_chunk_tokens - 1
        ) // relation.decode_chunk_tokens
        if prefill_count and not prefill_layouts:
            raise ProofV3Error("Goldilocks AIR trace slots lack prefill layouts")
        if decode_count and not decode_layouts:
            raise ProofV3Error("Goldilocks AIR trace slots lack decode layouts")
        length = prefill_count * len(prefill_layouts) + decode_count * len(decode_layouts)
        if length <= 0 or length >= 1 << 32:
            raise ProofV3Error("Goldilocks AIR trace slot universe is out of range")
        layout_bytes = b"".join(
            struct.pack("<I", index) for index in prefill_layouts + decode_layouts
        )
        binding = hashlib.sha256(
            _SLOT_UNIVERSE_DOMAIN
            + constraint_universe.binding_digest
            + self.constraint_system.digest()
            + struct.pack(
                "<IIIIII",
                prefill_count,
                decode_count,
                len(prefill_layouts),
                len(decode_layouts),
                length,
                len(layout_bytes),
            )
            + layout_bytes
        ).digest()
        object.__setattr__(self, "_constraint_universe", constraint_universe)
        object.__setattr__(self, "_prefill_layout_indices", prefill_layouts)
        object.__setattr__(self, "_decode_layout_indices", decode_layouts)
        object.__setattr__(self, "_prefill_chunk_count", prefill_count)
        object.__setattr__(self, "_decode_chunk_count", decode_count)
        object.__setattr__(self, "_length", length)
        object.__setattr__(self, "_binding_digest", binding)

    @property
    def binding_digest(self) -> bytes:
        return self._binding_digest

    @property
    def constraint_universe_binding_digest(self) -> bytes:
        return self._constraint_universe.binding_digest

    def __len__(self) -> int:
        return self._length

    def _slot_for(
        self,
        *,
        phase: str,
        local_chunk_index: int,
        layout_index: int,
    ) -> GoldilocksAirTraceSlotReferenceV3:
        relation = self.profile.relation_spec
        if phase == "prefill":
            if local_chunk_index >= self._prefill_chunk_count:
                raise ProofV3Error("Goldilocks AIR trace slot chunk is out of range")
            chunk_index = local_chunk_index
            start = local_chunk_index * relation.prefill_chunk_tokens
            token_count = min(
                relation.prefill_chunk_tokens,
                self.envelope.context_token_count - start,
            )
        else:
            if local_chunk_index >= self._decode_chunk_count:
                raise ProofV3Error("Goldilocks AIR trace slot chunk is out of range")
            chunk_index = self._prefill_chunk_count + local_chunk_index
            start = self.envelope.context_token_count + (
                local_chunk_index * relation.decode_chunk_tokens
            )
            token_count = min(
                relation.decode_chunk_tokens,
                self.envelope.decode_token_count
                - local_chunk_index * relation.decode_chunk_tokens,
            )
        layout = self.constraint_system.layouts[layout_index]
        return GoldilocksAirTraceSlotReferenceV3(
            layout_index=layout_index,
            node_id=layout.node_id,
            phase=phase,
            chunk_index=chunk_index,
            logical_token_start=start,
            token_count=token_count,
        )

    def __getitem__(self, index: int) -> GoldilocksAirTraceSlotReferenceV3:
        if type(index) is not int:
            raise TypeError("Goldilocks AIR trace slot index must be an integer")
        if index < 0 or index >= self._length:
            raise IndexError("Goldilocks AIR trace slot index is out of range")
        prefill_length = self._prefill_chunk_count * len(self._prefill_layout_indices)
        if index < prefill_length:
            chunk, offset = divmod(index, len(self._prefill_layout_indices))
            return self._slot_for(
                phase="prefill",
                local_chunk_index=chunk,
                layout_index=self._prefill_layout_indices[offset],
            )
        decode_index = index - prefill_length
        chunk, offset = divmod(decode_index, len(self._decode_layout_indices))
        return self._slot_for(
            phase="decode",
            local_chunk_index=chunk,
            layout_index=self._decode_layout_indices[offset],
        )

    def __iter__(self) -> Iterator[GoldilocksAirTraceSlotReferenceV3]:
        for index in range(self._length):
            yield self[index]

    def validate_slot(self, slot: GoldilocksAirTraceSlotReferenceV3) -> None:
        if not isinstance(slot, GoldilocksAirTraceSlotReferenceV3):
            raise ProofV3Error("Goldilocks AIR trace slot is malformed")
        if slot.layout_index < 0 or slot.layout_index >= len(self.constraint_system.layouts):
            raise ProofV3Error("Goldilocks AIR trace slot layout is out of range")
        layout = self.constraint_system.layouts[slot.layout_index]
        if layout.node_id != slot.node_id or slot.phase not in layout.phases:
            raise ProofV3Error("Goldilocks AIR trace slot does not match its layout")
        if slot.phase == "prefill":
            local_chunk_index = slot.chunk_index
            if local_chunk_index >= self._prefill_chunk_count:
                raise ProofV3Error("Goldilocks AIR trace slot chunk is out of range")
        else:
            local_chunk_index = slot.chunk_index - self._prefill_chunk_count
            if local_chunk_index < 0 or local_chunk_index >= self._decode_chunk_count:
                raise ProofV3Error("Goldilocks AIR trace slot chunk is out of range")
        expected = self._slot_for(
            phase=slot.phase,
            local_chunk_index=local_chunk_index,
            layout_index=slot.layout_index,
        )
        if slot != expected:
            raise ProofV3Error("Goldilocks AIR trace slot is not verifier derived")

    def index_of(self, slot: GoldilocksAirTraceSlotReferenceV3) -> int:
        """Return the canonical index for one already validated slot."""

        self.validate_slot(slot)
        if slot.phase == "prefill":
            layout_indices = self._prefill_layout_indices
            local_chunk_index = slot.chunk_index
            base = 0
        else:
            layout_indices = self._decode_layout_indices
            local_chunk_index = slot.chunk_index - self._prefill_chunk_count
            base = self._prefill_chunk_count * len(self._prefill_layout_indices)
        try:
            layout_offset = layout_indices.index(slot.layout_index)
        except ValueError as exc:
            raise ProofV3Error("Goldilocks AIR trace slot is not in this universe") from exc
        return base + local_chunk_index * len(layout_indices) + layout_offset

    def slot_indices_for_node_range(
        self,
        *,
        node_id: object,
        logical_token_start: object,
        token_count: object,
    ) -> tuple[int, ...]:
        """Return every exact slot for one signed node over a token window.

        This is deliberately range-to-chunk expansion, not a per-token
        materialization.  It gives a post-nonce semantic selector the complete
        set of prefill and/or decode chunk traces that overlap its
        validator-derived global token window.  A node which lacks one
        required phase fails closed instead of silently narrowing coverage.
        """

        if not isinstance(node_id, str) or not node_id:
            raise ProofV3Error("Goldilocks AIR trace-map node_id is malformed")
        start = _u32(
            logical_token_start,
            "Goldilocks AIR trace-map logical_token_start",
        )
        count = _u32(
            token_count,
            "Goldilocks AIR trace-map token_count",
            positive=True,
        )
        total_tokens = self.envelope.context_token_count + self.envelope.decode_token_count
        if start + count > total_tokens:
            raise ProofV3Error(
                "Goldilocks AIR trace-map token range exceeds the request"
            )
        matching_layouts = tuple(
            index
            for index, layout in enumerate(self.constraint_system.layouts)
            if layout.node_id == node_id
        )
        if len(matching_layouts) != 1:
            raise ProofV3Error(
                "Goldilocks AIR trace-map node does not resolve to one signed layout"
            )
        layout_index = matching_layouts[0]
        layout = self.constraint_system.layouts[layout_index]
        relation = self.profile.relation_spec
        end = start + count
        indices: list[int] = []
        if start < self.envelope.context_token_count:
            if "prefill" not in layout.phases:
                raise ProofV3Error(
                    "Goldilocks AIR trace-map node lacks required prefill coverage"
                )
            prefill_end = min(end, self.envelope.context_token_count)
            first_chunk = start // relation.prefill_chunk_tokens
            last_chunk = (prefill_end - 1) // relation.prefill_chunk_tokens
            for local_chunk_index in range(first_chunk, last_chunk + 1):
                indices.append(
                    self.index_of(
                        self._slot_for(
                            phase="prefill",
                            local_chunk_index=local_chunk_index,
                            layout_index=layout_index,
                        )
                    )
                )
        if end > self.envelope.context_token_count:
            if "decode" not in layout.phases:
                raise ProofV3Error(
                    "Goldilocks AIR trace-map node lacks required decode coverage"
                )
            decode_start = max(start, self.envelope.context_token_count)
            first_chunk = (
                decode_start - self.envelope.context_token_count
            ) // relation.decode_chunk_tokens
            last_chunk = (
                end - 1 - self.envelope.context_token_count
            ) // relation.decode_chunk_tokens
            for local_chunk_index in range(first_chunk, last_chunk + 1):
                indices.append(
                    self.index_of(
                        self._slot_for(
                            phase="decode",
                            local_chunk_index=local_chunk_index,
                            layout_index=layout_index,
                        )
                    )
                )
        if not indices:
            raise ProofV3Error("Goldilocks AIR trace-map token range has no slots")
        return tuple(indices)

    def atomic_coordinate(
        self,
        *,
        slot: GoldilocksAirTraceSlotReferenceV3,
        atomic_index: int,
    ) -> GoldilocksConstraintCoordinateV3:
        """Recover one existing atomic coordinate without duplicating a trace."""

        self.validate_slot(slot)
        index = _u32(
            atomic_index,
            "Goldilocks AIR trace-slot atomic_index",
        )
        layout = self.constraint_system.layouts[slot.layout_index]
        if index >= len(layout.atomic_constraint_ids):
            raise ProofV3Error("Goldilocks AIR trace-slot atomic index is out of range")
        return GoldilocksConstraintCoordinateV3(
            node_id=slot.node_id,
            phase=slot.phase,
            chunk_index=slot.chunk_index,
            logical_token_start=slot.logical_token_start,
            token_count=slot.token_count,
            relation_index=self.constraint_system.relation_index_for(
                layout_index=slot.layout_index,
                atomic_constraint_index=index,
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class GoldilocksAirTraceMapStatementReferenceV3:
    """Typed validator statement shared by all trace slots in one request."""

    slot_universe: GoldilocksAirTraceSlotUniverseReferenceV3
    constraint_system: GoldilocksExecutionConstraintSystemV3
    program_bundle: GoldilocksConstraintProgramBundleV3
    proof_challenge_id: bytes
    static_manifest_digest: bytes
    execution_profile_digest: bytes
    precommit_context_digest: bytes
    cache_lease_digest: bytes
    request_digest: bytes
    commitment_envelope_digest: bytes
    binding_digest: bytes
    _factory_token: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ProofV3VerificationError(
            "Goldilocks AIR trace-map statements must use the validator factory"
        )

    @classmethod
    def _construct(
        cls,
        *,
        slot_universe: GoldilocksAirTraceSlotUniverseReferenceV3,
        constraint_system: GoldilocksExecutionConstraintSystemV3,
        program_bundle: GoldilocksConstraintProgramBundleV3,
        proof_challenge_id: bytes,
        static_manifest_digest: bytes,
        execution_profile_digest: bytes,
        precommit_context_digest: bytes,
        cache_lease_digest: bytes,
        request_digest: bytes,
        commitment_envelope_digest: bytes,
        binding_digest: bytes,
        _factory_token: object | None = None,
    ) -> "GoldilocksAirTraceMapStatementReferenceV3":
        if _factory_token is not _STATEMENT_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map statements must use the validator factory"
            )
        result = object.__new__(cls)
        for name, value in (
            ("slot_universe", slot_universe),
            ("constraint_system", constraint_system),
            ("program_bundle", program_bundle),
            ("proof_challenge_id", proof_challenge_id),
            ("static_manifest_digest", static_manifest_digest),
            ("execution_profile_digest", execution_profile_digest),
            ("precommit_context_digest", precommit_context_digest),
            ("cache_lease_digest", cache_lease_digest),
            ("request_digest", request_digest),
            ("commitment_envelope_digest", commitment_envelope_digest),
            ("binding_digest", binding_digest),
        ):
            object.__setattr__(result, name, value)
        object.__setattr__(result, "_factory_token", _STATEMENT_FACTORY_TOKEN)
        result.require_factory_provenance()
        return result

    def require_factory_provenance(self) -> None:
        if self._factory_token is not _STATEMENT_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map statement lacks validator provenance"
            )
        if not isinstance(self.slot_universe, GoldilocksAirTraceSlotUniverseReferenceV3):
            raise ProofV3VerificationError("Goldilocks AIR trace-map slots are malformed")
        if self.slot_universe.constraint_system is not self.constraint_system:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map slots have another constraint system"
            )
        if not isinstance(self.program_bundle, GoldilocksConstraintProgramBundleV3):
            raise ProofV3VerificationError("Goldilocks AIR trace-map programs are malformed")
        for value, name in (
            (self.proof_challenge_id, "Goldilocks AIR trace-map challenge"),
            (self.static_manifest_digest, "Goldilocks AIR trace-map static manifest"),
            (self.execution_profile_digest, "Goldilocks AIR trace-map profile"),
            (self.precommit_context_digest, "Goldilocks AIR trace-map precommit"),
            (self.cache_lease_digest, "Goldilocks AIR trace-map cache lease"),
            (self.request_digest, "Goldilocks AIR trace-map request"),
            (self.commitment_envelope_digest, "Goldilocks AIR trace-map envelope"),
            (self.binding_digest, "Goldilocks AIR trace-map binding"),
        ):
            _fixed32(value, name, nonzero=True)

    def canonical_bytes(self) -> bytes:
        return (
            self.proof_challenge_id
            + self.static_manifest_digest
            + self.execution_profile_digest
            + self.precommit_context_digest
            + self.cache_lease_digest
            + self.request_digest
            + self.commitment_envelope_digest
            + self.constraint_system.digest()
            + self.program_bundle.digest()
            + self.slot_universe.constraint_universe_binding_digest
            + self.slot_universe.binding_digest
            + struct.pack("<I", len(self.slot_universe))
            + self.binding_digest
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_STATEMENT_DIGEST_DOMAIN + self.canonical_bytes()).digest()

    def slot_core(self, *, slot_index: int) -> GoldilocksAirStatementCoreReferenceV3:
        """Derive the unique all-constraints AIR core for one exact slot."""

        try:
            slot = self.slot_universe[slot_index]
            self.slot_universe.validate_slot(slot)
            layout = self.constraint_system.layouts[slot.layout_index]
            program = self.program_bundle.programs[slot.layout_index]
            if program.digest() != layout.constraint_program_digest:
                raise ProofV3VerificationError(
                    "Goldilocks AIR trace-map program does not match its layout"
                )
            core_binding = hashlib.sha256(
                _SLOT_CORE_BINDING_DOMAIN
                + self.binding_digest
                + self.slot_universe.binding_digest
                + struct.pack("<I", slot_index)
                + slot.canonical_bytes()
                + program.digest()
            ).digest()
            return GoldilocksAirStatementCoreReferenceV3(
                validator_binding_digest=core_binding,
                program=program,
                token_count=slot.token_count,
            )
        except ProofV3VerificationError:
            raise
        except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map slot statement is malformed"
            ) from exc


def derive_goldilocks_air_trace_map_statement_reference_v3(
    *,
    qualified_profile: QualifiedExecutionProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    envelope_binding: EnvelopeBindingV3,
) -> GoldilocksAirTraceMapStatementReferenceV3:
    """Derive the complete trace-map statement from accepted validator data."""

    try:
        _require_bound_envelope(
            qualified_profile=qualified_profile,
            envelope=envelope,
            envelope_binding=envelope_binding,
        )
        profile = qualified_profile.profile
        artifacts = qualified_profile.registration.artifacts
        constraint_system = artifacts.constraint_system
        program_bundle = artifacts.constraint_program_bundle
        if artifacts.execution_profile_digest != profile.digest():
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map artifacts have an unexpected profile"
            )
        if not isinstance(program_bundle, GoldilocksConstraintProgramBundleV3):
            raise ProofV3VerificationError("Goldilocks AIR trace-map programs are malformed")
        program_bundle.validate_constraint_system(
            constraint_system=constraint_system,
            relation=profile.relation_spec,
        )
        slots = GoldilocksAirTraceSlotUniverseReferenceV3(
            profile=profile,
            envelope=envelope,
            constraint_system=constraint_system,
        )
        binding = hashlib.sha256(
            _STATEMENT_BINDING_DOMAIN
            + profile.static_manifest_digest
            + profile.digest()
            + envelope_binding.precommit_context.digest()
            + envelope_binding.request_binding.request_digest
            + envelope.digest()
            + constraint_system.digest()
            + program_bundle.digest()
            + slots.constraint_universe_binding_digest
            + slots.binding_digest
        ).digest()
    except ProofV3VerificationError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "Goldilocks AIR trace-map statement inputs are malformed"
        ) from exc
    return GoldilocksAirTraceMapStatementReferenceV3._construct(
        slot_universe=slots,
        constraint_system=constraint_system,
        program_bundle=program_bundle,
        proof_challenge_id=envelope_binding.precommit_context.proof_challenge_id,
        static_manifest_digest=profile.static_manifest_digest,
        execution_profile_digest=profile.digest(),
        precommit_context_digest=envelope_binding.precommit_context.digest(),
        cache_lease_digest=envelope_binding.precommit_context.cache_lease_digest,
        request_digest=envelope_binding.request_binding.request_digest,
        commitment_envelope_digest=envelope.digest(),
        binding_digest=binding,
        _factory_token=_STATEMENT_FACTORY_TOKEN,
    )


def _reference_tree_leaf_count(slot_count: int) -> int:
    leaf_count = _next_power_of_two(slot_count, name="Goldilocks AIR trace-map slot_count")
    if leaf_count > MAX_GOLDILOCKS_MERKLE_REFERENCE_LEAF_COUNT:
        raise ProofV3Error(
            "Goldilocks AIR trace map exceeds the CPU Merkle reference cap"
        )
    return leaf_count


def _tree_binding_digest(
    *,
    statement: GoldilocksAirTraceMapStatementReferenceV3,
    tree_leaf_count: int,
) -> bytes:
    return hashlib.sha256(
        _TREE_BINDING_DOMAIN
        + statement.digest()
        + statement.slot_universe.binding_digest
        + struct.pack("<II", len(statement.slot_universe), tree_leaf_count)
    ).digest()


@dataclass(frozen=True, slots=True)
class GoldilocksAirTraceMapPrecommitmentReferenceV3:
    """Frozen root for the complete pre-nonce trace-slot map."""

    statement: GoldilocksAirTraceMapStatementReferenceV3
    tree_leaf_count: int
    map_tree_binding_digest: bytes
    trace_map_commitment: bytes
    abi_id: str = GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_ABI_V3
    format_version: int = GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_FORMAT_VERSION_V3

    def __post_init__(self) -> None:
        if self.abi_id != GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_ABI_V3:
            raise ProofV3Error("Goldilocks AIR trace-map ABI is unsupported")
        if type(self.format_version) is not int or self.format_version != (
            GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_FORMAT_VERSION_V3
        ):
            raise ProofV3Error("Goldilocks AIR trace-map format version is unsupported")
        if not isinstance(self.statement, GoldilocksAirTraceMapStatementReferenceV3):
            raise ProofV3Error("Goldilocks AIR trace-map statement is malformed")
        self.statement.require_factory_provenance()
        expected_leaf_count = _reference_tree_leaf_count(len(self.statement.slot_universe))
        if self.tree_leaf_count != expected_leaf_count:
            raise ProofV3Error("Goldilocks AIR trace-map tree size is unexpected")
        expected_binding = _tree_binding_digest(
            statement=self.statement,
            tree_leaf_count=expected_leaf_count,
        )
        if self.map_tree_binding_digest != expected_binding:
            raise ProofV3Error("Goldilocks AIR trace-map tree binding is unexpected")
        _fixed32(
            self.trace_map_commitment,
            "Goldilocks AIR trace-map commitment",
            nonzero=True,
        )

    @property
    def slot_count(self) -> int:
        return len(self.statement.slot_universe)

    def canonical_bytes(self) -> bytes:
        abi = self.abi_id.encode("ascii")
        return (
            struct.pack("<HH", self.format_version, len(abi))
            + abi
            + self.statement.digest()
            + self.map_tree_binding_digest
            + struct.pack("<II", self.slot_count, self.tree_leaf_count)
            + self.trace_map_commitment
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_PRECOMMITMENT_DOMAIN + self.canonical_bytes()).digest()


@dataclass(frozen=True, slots=True)
class GoldilocksAirTraceMapOpeningReferenceV3:
    """One canonical Merkle opening plus selected trace precommitments."""

    map_opening: GoldilocksMerkleMultiOpeningReference
    trace_precommitments: tuple[GoldilocksAirTracePrecommitmentReferenceV3, ...]
    abi_id: str = GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_ABI_V3
    format_version: int = GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_FORMAT_VERSION_V3

    def __post_init__(self) -> None:
        if self.abi_id != GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_ABI_V3:
            raise ProofV3Error("Goldilocks AIR trace-map opening ABI is unsupported")
        if type(self.format_version) is not int or self.format_version != (
            GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_FORMAT_VERSION_V3
        ):
            raise ProofV3Error(
                "Goldilocks AIR trace-map opening format version is unsupported"
            )
        if not isinstance(self.map_opening, GoldilocksMerkleMultiOpeningReference):
            raise ProofV3Error("Goldilocks AIR trace-map Merkle opening is malformed")
        entries = tuple(self.trace_precommitments)
        if len(entries) != len(self.map_opening.indices) or not entries:
            raise ProofV3Error("Goldilocks AIR trace-map opening entry count is malformed")
        if not all(
            isinstance(item, GoldilocksAirTracePrecommitmentReferenceV3)
            for item in entries
        ):
            raise ProofV3Error("Goldilocks AIR trace-map opening entries are malformed")
        object.__setattr__(self, "trace_precommitments", entries)


@dataclass(frozen=True, slots=True)
class GoldilocksAirTraceMapOracleReferenceV3:
    """Retained complete map for small CPU conformance proofs only."""

    precommitment: GoldilocksAirTraceMapPrecommitmentReferenceV3
    trace_oracles: tuple[GoldilocksAirTraceOracleReferenceV3, ...]
    map_tree: GoldilocksMerkleTreeReference

    def __post_init__(self) -> None:
        if not isinstance(
            self.precommitment,
            GoldilocksAirTraceMapPrecommitmentReferenceV3,
        ):
            raise ProofV3Error("Goldilocks AIR trace-map precommitment is malformed")
        oracles = tuple(self.trace_oracles)
        if len(oracles) != self.precommitment.slot_count or not all(
            isinstance(item, GoldilocksAirTraceOracleReferenceV3) for item in oracles
        ):
            raise ProofV3Error("Goldilocks AIR trace-map oracle set is malformed")
        if not isinstance(self.map_tree, GoldilocksMerkleTreeReference):
            raise ProofV3Error("Goldilocks AIR trace-map tree is malformed")
        if (
            self.map_tree.commitment != self.precommitment.trace_map_commitment
            or self.map_tree.binding_digest != self.precommitment.map_tree_binding_digest
            or self.map_tree.leaf_count != self.precommitment.tree_leaf_count
            or self.map_tree.leaf_width != GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_LEAF_WIDTH_V3
        ):
            raise ProofV3Error("Goldilocks AIR trace-map tree does not match its root")
        statement = self.precommitment.statement
        for index, oracle in enumerate(oracles):
            expected_core = statement.slot_core(slot_index=index)
            if oracle.precommitment.core.digest() != expected_core.digest():
                raise ProofV3Error(
                    "Goldilocks AIR trace-map oracle belongs to another slot"
                )
        object.__setattr__(self, "trace_oracles", oracles)

    def open(self, slot_indices: object) -> GoldilocksAirTraceMapOpeningReferenceV3:
        indices = _slot_indices(
            slot_indices,
            slot_count=self.precommitment.slot_count,
            name="Goldilocks AIR trace-map opening indices",
        )
        return GoldilocksAirTraceMapOpeningReferenceV3(
            map_opening=self.map_tree.open(indices),
            trace_precommitments=tuple(
                self.trace_oracles[index].precommitment for index in indices
            ),
        )


def _slot_indices(
    value: object,
    *,
    slot_count: int,
    name: str,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        raise ProofV3Error(f"{name} must be an iterable")
    try:
        raw = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ProofV3Error(f"{name} must be an iterable") from exc
    if not raw:
        raise ProofV3Error(f"{name} must not be empty")
    indices = tuple(_integer(item, f"{name}[{index}]") for index, item in enumerate(raw))
    if indices != tuple(sorted(set(indices))):
        raise ProofV3Error(f"{name} must be sorted and distinct")
    if any(index < 0 or index >= slot_count for index in indices):
        raise ProofV3Error(f"{name} contains an out-of-range slot")
    return indices


def build_goldilocks_air_trace_map_oracle_reference_v3(
    *,
    statement: GoldilocksAirTraceMapStatementReferenceV3,
    trace_oracles: Sequence[GoldilocksAirTraceOracleReferenceV3],
) -> GoldilocksAirTraceMapOracleReferenceV3:
    """Commit every exact slot root in canonical order before nonce reveal.

    The CPU reference receives retained slot oracles so tests can construct
    openings.  A native prover must stream the same leaves into a Merkle
    frontier and retain/reconstruct witness material independently.
    """

    if not isinstance(statement, GoldilocksAirTraceMapStatementReferenceV3):
        raise ProofV3Error("Goldilocks AIR trace-map statement is malformed")
    statement.require_factory_provenance()
    try:
        oracles = tuple(trace_oracles)
    except TypeError as exc:
        raise ProofV3Error("Goldilocks AIR trace-map oracles are malformed") from exc
    if len(oracles) != len(statement.slot_universe):
        raise ProofV3Error("Goldilocks AIR trace-map omits or adds a slot")
    rows: list[tuple[int, ...]] = []
    for index, oracle in enumerate(oracles):
        if not isinstance(oracle, GoldilocksAirTraceOracleReferenceV3):
            raise ProofV3Error("Goldilocks AIR trace-map oracle is malformed")
        expected_core = statement.slot_core(slot_index=index)
        if oracle.precommitment.core.digest() != expected_core.digest():
            raise ProofV3Error("Goldilocks AIR trace-map oracle belongs to another slot")
        rows.append(
            _digest_row(
                oracle.precommitment.digest(),
                "Goldilocks AIR trace-map entry digest",
            )
        )
    tree_leaf_count = _reference_tree_leaf_count(len(rows))
    rows.extend(
        (0,) * GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_LEAF_WIDTH_V3
        for _ in range(tree_leaf_count - len(rows))
    )
    tree_binding = _tree_binding_digest(
        statement=statement,
        tree_leaf_count=tree_leaf_count,
    )
    tree = GoldilocksMerkleTreeReference.from_rows(rows, binding_digest=tree_binding)
    precommitment = GoldilocksAirTraceMapPrecommitmentReferenceV3(
        statement=statement,
        tree_leaf_count=tree_leaf_count,
        map_tree_binding_digest=tree_binding,
        trace_map_commitment=tree.commitment,
    )
    return GoldilocksAirTraceMapOracleReferenceV3(
        precommitment=precommitment,
        trace_oracles=oracles,
        map_tree=tree,
    )


def derive_goldilocks_air_trace_map_selection_reference_v3(
    *,
    validator_nonce: bytes,
    statement: GoldilocksAirTraceMapStatementReferenceV3,
) -> tuple[int, ...]:
    """Select two distinct slots after their map was frozen.

    The map root is deliberately absent from this sampler.  The hidden nonce
    already makes the exact selection unavailable before the map is sealed;
    using a miner-chosen map root as challenge entropy would only expand the
    prover-controlled transcript surface.  A qualified backend will replace
    this fixed reference count with its signed stratified policy.
    """

    try:
        nonce = _fixed32(
            validator_nonce,
            "Goldilocks AIR trace-map validator nonce",
            nonzero=True,
        )
        if not isinstance(statement, GoldilocksAirTraceMapStatementReferenceV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map statement is malformed"
            )
        statement.require_factory_provenance()
        slot_count = len(statement.slot_universe)
        query_count = GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_QUERY_COUNT_V3
        if slot_count < query_count:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map has too few slots for its query count"
            )
        seed = hashlib.sha256(
            _SELECTION_DOMAIN
            + nonce
            + statement.digest()
            + statement.slot_universe.binding_digest
            + struct.pack("<II", slot_count, query_count)
        ).digest()
        ceiling = (1 << 256) - ((1 << 256) % slot_count)
        selected: set[int] = set()
        counter = 0
        while len(selected) < query_count:
            if counter >= MAX_GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_REJECTION_ATTEMPTS_V3:
                raise ProofV3VerificationError(
                    "Goldilocks AIR trace-map selection exhausted rejection sampling"
                )
            candidate = int.from_bytes(
                hashlib.sha256(
                    _SELECTION_DOMAIN + seed + struct.pack("<I", counter)
                ).digest(),
                "big",
            )
            counter += 1
            if candidate < ceiling:
                selected.add(candidate % slot_count)
        return tuple(sorted(selected))
    except ProofV3VerificationError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "Goldilocks AIR trace-map selection is malformed"
        ) from exc


def verify_goldilocks_air_trace_map_opening_reference_v3(
    opening: object,
    *,
    statement: GoldilocksAirTraceMapStatementReferenceV3,
    precommitment: GoldilocksAirTraceMapPrecommitmentReferenceV3,
    expected_slot_indices: object,
) -> tuple[GoldilocksAirTracePrecommitmentReferenceV3, ...]:
    """Verify exact selected roots from a previously frozen trace map.

    This function only authenticates the selected slot entries.  The caller
    must next run the AIR/FRI verifier against every returned precommitment
    using the same post-nonce validator nonce.
    """

    try:
        if not isinstance(statement, GoldilocksAirTraceMapStatementReferenceV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map statement is malformed"
            )
        statement.require_factory_provenance()
        if not isinstance(precommitment, GoldilocksAirTraceMapPrecommitmentReferenceV3):
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map precommitment is malformed"
            )
        if precommitment.statement.digest() != statement.digest():
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map belongs to another statement"
            )
        indices = _slot_indices(
            expected_slot_indices,
            slot_count=precommitment.slot_count,
            name="expected Goldilocks AIR trace-map slot indices",
        )
        if not isinstance(opening, GoldilocksAirTraceMapOpeningReferenceV3):
            raise ProofV3VerificationError("Goldilocks AIR trace-map opening is malformed")
        if opening.map_opening.indices != indices:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace-map opening has unexpected slot indices"
            )
        verify_goldilocks_merkle_multiopening_reference(
            precommitment.trace_map_commitment,
            opening.map_opening,
            expected_binding_digest=precommitment.map_tree_binding_digest,
            expected_leaf_count=precommitment.tree_leaf_count,
            expected_leaf_width=GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_LEAF_WIDTH_V3,
            expected_indices=indices,
        )
        selected: list[GoldilocksAirTracePrecommitmentReferenceV3] = []
        for index, row, trace_precommitment in zip(
            indices,
            opening.map_opening.rows,
            opening.trace_precommitments,
            strict=True,
        ):
            if not isinstance(trace_precommitment, GoldilocksAirTracePrecommitmentReferenceV3):
                raise ProofV3VerificationError(
                    "Goldilocks AIR trace-map entry is malformed"
                )
            expected_core = statement.slot_core(slot_index=index)
            if trace_precommitment.core.digest() != expected_core.digest():
                raise ProofV3VerificationError(
                    "Goldilocks AIR trace-map entry belongs to another slot"
                )
            if (
                _row_digest(row, "Goldilocks AIR trace-map entry row")
                != trace_precommitment.digest()
            ):
                raise ProofV3VerificationError(
                    "Goldilocks AIR trace-map entry digest does not match its leaf"
                )
            selected.append(trace_precommitment)
        return tuple(selected)
    except ProofV3VerificationError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("Goldilocks AIR trace-map opening is malformed") from exc


__all__ = [
    "GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_ABI_V3",
    "GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_FORMAT_VERSION_V3",
    "GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_LEAF_WIDTH_V3",
    "GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_QUERY_COUNT_V3",
    "GoldilocksAirTraceMapOpeningReferenceV3",
    "GoldilocksAirTraceMapOracleReferenceV3",
    "GoldilocksAirTraceMapPrecommitmentReferenceV3",
    "GoldilocksAirTraceMapStatementReferenceV3",
    "GoldilocksAirTraceSlotReferenceV3",
    "GoldilocksAirTraceSlotUniverseReferenceV3",
    "build_goldilocks_air_trace_map_oracle_reference_v3",
    "derive_goldilocks_air_trace_map_selection_reference_v3",
    "derive_goldilocks_air_trace_map_statement_reference_v3",
    "verify_goldilocks_air_trace_map_opening_reference_v3",
]
