"""Bounded runtime-state tensor binding reference for proof-v3.

This unregistered CPU conformance module closes one narrow gap in the frozen
AIR trace reference: a base trace is reconstructed from a selected AIR slot
and compared cell-for-cell to canonical, validator-derived runtime-state
segments.  The segment roots are committed in a second flat map whose exact
universe is derived from the validator-owned trace-map statement.

The module deliberately supports only ``runtime_state`` tensors with the
exact source-coordinate ABI.  It rejects request/input roots, final
hidden/logit/token roots, and cache tensors; those require their own encoding,
sampler, or logical-RAM relations.  It also leaves static table columns out of
scope.  Thus this is a bounded golden-vector relation, not an execution proof,
model-substitution claim, cache proof, or production wire/backend.

The full base-to-LDE reconstruction is required because an AIR precommitment
commits to a disjoint-coset LDE, not directly to base trace rows.  A future
receipt coordinator must atomically retain this tensor-map root together with
the trace-map root before nonce reveal.  This module deliberately does not
create that network chronology by itself.
"""

from __future__ import annotations

import hashlib
import operator
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Final

from verallm.proof_v3.constraint_program import (
    GoldilocksConstraintProgramV3,
    GoldilocksConstraintTraceReferenceV3,
    GoldilocksRuntimeTraceColumnBindingV3,
    GOLDILOCKS_RUNTIME_TRACE_FIELD_ENCODING_V3,
    GOLDILOCKS_RUNTIME_TOKEN_AXIS_CONTEXT_V3,
    GOLDILOCKS_RUNTIME_TOKEN_AXIS_DECODE_V3,
    GOLDILOCKS_RUNTIME_TOKEN_AXIS_SEQUENCE_V3,
    verify_goldilocks_constraint_program_reference_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_air_reference import (
    GoldilocksAirTraceOracleReferenceV3,
    GoldilocksAirTracePrecommitmentReferenceV3,
    build_goldilocks_air_trace_oracle_reference_v3,
)
from verallm.proof_v3.goldilocks_air_trace_map_reference import (
    GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_LEAF_WIDTH_V3,
    GoldilocksAirTraceMapOpeningReferenceV3,
    GoldilocksAirTraceMapPrecommitmentReferenceV3,
    GoldilocksAirTraceMapStatementReferenceV3,
    verify_goldilocks_air_trace_map_opening_reference_v3,
)
from verallm.proof_v3.goldilocks_merkle_reference import (
    GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES,
    MAX_GOLDILOCKS_MERKLE_REFERENCE_LEAF_COUNT,
    GoldilocksMerkleMultiOpeningReference,
    GoldilocksMerkleTreeReference,
    verify_goldilocks_merkle_multiopening_reference,
)
from verallm.proof_v3.goldilocks_reference import (
    MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE,
    canonical_goldilocks,
)


GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_ABI_V3: Final = (
    "goldilocks.runtime_tensor_binding.reference.v1"
)
GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_FORMAT_VERSION_V3: Final = 1
GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_LEAF_WIDTH_V3: Final = (
    GOLDILOCKS_AIR_TRACE_MAP_REFERENCE_LEAF_WIDTH_V3
)

_PHASE_CODES: Final = {"prefill": 1, "decode": 2}
_UNIVERSE_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_RUNTIME_TENSOR/V1/UNIVERSE/SHA256"
)
_STATEMENT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_RUNTIME_TENSOR/V1/STATEMENT/SHA256"
)
_STATEMENT_DIGEST_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_RUNTIME_TENSOR/V1/STATEMENT_DIGEST/SHA256"
)
_SEGMENT_TREE_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_RUNTIME_TENSOR/V1/SEGMENT_TREE/SHA256"
)
_SEGMENT_PRECOMMITMENT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_RUNTIME_TENSOR/V1/SEGMENT_PRECOMMIT/SHA256"
)
_MAP_TREE_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_RUNTIME_TENSOR/V1/MAP_TREE/SHA256"
)
_MAP_PRECOMMITMENT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_RUNTIME_TENSOR/V1/MAP_PRECOMMIT/SHA256"
)
_PAIR_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_RUNTIME_TENSOR/V1/PRECOMMITMENT_PAIR/SHA256"
)
_PAIR_DIGEST_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_RUNTIME_TENSOR/V1/PRECOMMITMENT_PAIR_DIGEST/SHA256"
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


def _u64(value: object, name: str, *, positive: bool = False) -> int:
    result = _integer(value, name)
    if result < (1 if positive else 0) or result >= 1 << 64:
        raise ProofV3Error(f"{name} must be an unsigned 64-bit integer")
    return result


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


def _next_power_of_two(value: int, *, name: str) -> int:
    if value < 1:
        raise ProofV3Error(f"{name} must be positive")
    return 1 << (value - 1).bit_length()


def _digest_row(value: object, name: str) -> tuple[int, ...]:
    digest = _fixed32(value, name)
    return tuple(struct.unpack("<8I", digest))


def _row_digest(value: object, name: str) -> bytes:
    if (
        not isinstance(value, tuple)
        or len(value) != GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_LEAF_WIDTH_V3
    ):
        raise ProofV3VerificationError(
            f"{name} must contain exactly "
            f"{GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_LEAF_WIDTH_V3} limbs"
        )
    limbs: list[int] = []
    for index, limb in enumerate(value):
        integer = _integer(limb, f"{name}[{index}]")
        if integer < 0 or integer >= 1 << 32:
            raise ProofV3VerificationError(f"{name}[{index}] is not a 32-bit limb")
        limbs.append(integer)
    return struct.pack("<8I", *limbs)


def _segment_tree_leaf_count(element_count: int) -> int:
    leaf_count = _next_power_of_two(
        element_count,
        name="runtime tensor segment element_count",
    )
    if leaf_count > MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE:
        raise ProofV3Error("runtime tensor segment exceeds the CPU reference cap")
    return leaf_count


def _map_tree_leaf_count(segment_count: int) -> int:
    leaf_count = _next_power_of_two(
        segment_count,
        name="runtime tensor segment count",
    )
    if leaf_count > MAX_GOLDILOCKS_MERKLE_REFERENCE_LEAF_COUNT:
        raise ProofV3Error("runtime tensor map exceeds the CPU reference cap")
    return leaf_count


def _segment_indices(
    value: object,
    *,
    segment_count: int,
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
    if any(index < 0 or index >= segment_count for index in indices):
        raise ProofV3Error(f"{name} contains an out-of-range segment")
    return indices


@dataclass(frozen=True, slots=True)
class GoldilocksRuntimeTensorSegmentReferenceV3:
    """One canonical runtime-state tensor segment for a request chunk."""

    tensor_id: str
    source_encoding_id: str
    source_layout_id: str
    token_axis_rule_id: str
    field_encoding_id: str
    phase: str
    chunk_index: int
    logical_token_start: int
    source_token_start: int
    token_count: int
    elements_per_token: int
    element_count: int

    def __post_init__(self) -> None:
        for value, name in (
            (self.tensor_id, "runtime tensor segment tensor_id"),
            (self.source_encoding_id, "runtime tensor segment source_encoding_id"),
            (self.source_layout_id, "runtime tensor segment source_layout_id"),
            (self.token_axis_rule_id, "runtime tensor segment token_axis_rule_id"),
        ):
            _identifier(value, name)
        if self.token_axis_rule_id not in {
            GOLDILOCKS_RUNTIME_TOKEN_AXIS_CONTEXT_V3,
            GOLDILOCKS_RUNTIME_TOKEN_AXIS_DECODE_V3,
            GOLDILOCKS_RUNTIME_TOKEN_AXIS_SEQUENCE_V3,
        }:
            raise ProofV3Error("runtime tensor segment token axis is unsupported")
        if self.phase not in _PHASE_CODES:
            raise ProofV3Error("runtime tensor segment phase is unsupported")
        if self.field_encoding_id != GOLDILOCKS_RUNTIME_TRACE_FIELD_ENCODING_V3:
            raise ProofV3Error("runtime tensor segment field encoding is unsupported")
        if (
            self.token_axis_rule_id == GOLDILOCKS_RUNTIME_TOKEN_AXIS_CONTEXT_V3
            and self.phase != "prefill"
        ) or (
            self.token_axis_rule_id == GOLDILOCKS_RUNTIME_TOKEN_AXIS_DECODE_V3
            and self.phase != "decode"
        ):
            raise ProofV3Error("runtime tensor segment token axis is incompatible with phase")
        _u32(self.chunk_index, "runtime tensor segment chunk_index")
        start = _u32(
            self.logical_token_start,
            "runtime tensor segment logical_token_start",
        )
        source_start = _u32(
            self.source_token_start,
            "runtime tensor segment source_token_start",
        )
        count = _u32(self.token_count, "runtime tensor segment token_count", positive=True)
        width = _u32(
            self.elements_per_token,
            "runtime tensor segment elements_per_token",
            positive=True,
        )
        element_count = _u64(
            self.element_count,
            "runtime tensor segment element_count",
            positive=True,
        )
        if element_count != count * width:
            raise ProofV3Error("runtime tensor segment has an unexpected element count")
        if start + count > 1 << 32:
            raise ProofV3Error("runtime tensor segment token range overflows")
        if source_start + count > 1 << 32:
            raise ProofV3Error("runtime tensor segment source range overflows")

    def canonical_bytes(self) -> bytes:
        fields = (
            self.tensor_id,
            self.source_encoding_id,
            self.source_layout_id,
            self.token_axis_rule_id,
            self.field_encoding_id,
        )
        encoded = b"".join(
            struct.pack("<B", len(field.encode("ascii"))) + field.encode("ascii")
            for field in fields
        )
        return encoded + struct.pack(
            "<BIIIIIQ",
            _PHASE_CODES[self.phase],
            self.chunk_index,
            self.logical_token_start,
            self.source_token_start,
            self.token_count,
            self.elements_per_token,
            self.element_count,
        )


@dataclass(frozen=True, slots=True)
class _RuntimeStateTemplateReferenceV3:
    tensor_id: str
    source_encoding_id: str
    source_layout_id: str
    token_axis_rule_id: str
    field_encoding_id: str
    elements_per_token: int

    def canonical_bytes(self) -> bytes:
        fields = (
            self.tensor_id,
            self.source_encoding_id,
            self.source_layout_id,
            self.token_axis_rule_id,
            self.field_encoding_id,
        )
        return b"".join(
            struct.pack("<B", len(field.encode("ascii"))) + field.encode("ascii")
            for field in fields
        ) + struct.pack("<I", self.elements_per_token)


@dataclass(frozen=True, slots=True)
class GoldilocksRuntimeTensorSegmentUniverseReferenceV3(
    Sequence[GoldilocksRuntimeTensorSegmentReferenceV3]
):
    """Lazy global home for every eligible runtime-state tensor segment."""

    trace_map_statement: GoldilocksAirTraceMapStatementReferenceV3
    _prefill_templates: tuple[_RuntimeStateTemplateReferenceV3, ...] = field(
        init=False, repr=False
    )
    _decode_templates: tuple[_RuntimeStateTemplateReferenceV3, ...] = field(
        init=False, repr=False
    )
    _prefill_chunk_count: int = field(init=False, repr=False)
    _decode_chunk_count: int = field(init=False, repr=False)
    _length: int = field(init=False, repr=False)
    _binding_digest: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        statement = self.trace_map_statement
        if not isinstance(statement, GoldilocksAirTraceMapStatementReferenceV3):
            raise ProofV3Error("runtime tensor universe statement is malformed")
        statement.require_factory_provenance()
        relation = statement.slot_universe.profile.relation_spec
        tensors = {tensor.tensor_id: tensor for tensor in relation.tensors}

        def templates_for_phase(phase: str) -> tuple[_RuntimeStateTemplateReferenceV3, ...]:
            by_tensor: dict[str, _RuntimeStateTemplateReferenceV3] = {}
            for layout_index, layout in enumerate(statement.constraint_system.layouts):
                if phase not in layout.phases:
                    continue
                program = statement.program_bundle.programs[layout_index]
                if not program.has_exact_source_bindings:
                    raise ProofV3Error(
                        "runtime tensor segments require exact source-bound programs"
                    )
                for binding in program.runtime_column_bindings:
                    tensor = tensors.get(binding.tensor_id)
                    if tensor is None:
                        raise ProofV3Error("runtime tensor segment references an unknown tensor")
                    if tensor.commitment_role != "runtime_state":
                        continue
                    if (
                        binding.token_axis_rule_id
                        == GOLDILOCKS_RUNTIME_TOKEN_AXIS_CONTEXT_V3
                        and phase != "prefill"
                    ) or (
                        binding.token_axis_rule_id
                        == GOLDILOCKS_RUNTIME_TOKEN_AXIS_DECODE_V3
                        and phase != "decode"
                    ):
                        raise ProofV3Error(
                            "runtime tensor source axis is incompatible with trace phase"
                        )
                    template = _RuntimeStateTemplateReferenceV3(
                        tensor_id=binding.tensor_id,
                        source_encoding_id=binding.source_encoding_id,
                        source_layout_id=binding.source_layout_id,
                        token_axis_rule_id=binding.token_axis_rule_id,
                        field_encoding_id=binding.field_encoding_id,
                        elements_per_token=binding.elements_per_token,
                    )
                    existing = by_tensor.get(binding.tensor_id)
                    if existing is not None and existing != template:
                        raise ProofV3Error(
                            "runtime tensor has inconsistent signed source coordinates"
                        )
                    by_tensor[binding.tensor_id] = template
            return tuple(by_tensor[tensor_id] for tensor_id in sorted(by_tensor))

        prefill_templates = templates_for_phase("prefill")
        decode_templates = templates_for_phase("decode")
        prefill_chunk_count = (
            statement.slot_universe.envelope.context_token_count
            + relation.prefill_chunk_tokens
            - 1
        ) // relation.prefill_chunk_tokens
        decode_chunk_count = (
            statement.slot_universe.envelope.decode_token_count
            + relation.decode_chunk_tokens
            - 1
        ) // relation.decode_chunk_tokens
        if prefill_chunk_count and not prefill_templates:
            raise ProofV3Error("runtime tensor universe lacks prefill templates")
        if decode_chunk_count and not decode_templates:
            raise ProofV3Error("runtime tensor universe lacks decode templates")
        length = prefill_chunk_count * len(prefill_templates) + decode_chunk_count * len(
            decode_templates
        )
        if length <= 0 or length >= 1 << 32:
            raise ProofV3Error("runtime tensor universe is out of range")
        template_bytes = b"".join(
            struct.pack("<BI", _PHASE_CODES[phase], len(templates))
            + b"".join(template.canonical_bytes() for template in templates)
            for phase, templates in (
                ("prefill", prefill_templates),
                ("decode", decode_templates),
            )
        )
        binding_digest = hashlib.sha256(
            _UNIVERSE_DOMAIN
            + statement.digest()
            + statement.slot_universe.binding_digest
            + struct.pack(
                "<IIIII",
                prefill_chunk_count,
                decode_chunk_count,
                len(prefill_templates),
                len(decode_templates),
                length,
            )
            + template_bytes
        ).digest()
        object.__setattr__(self, "_prefill_templates", prefill_templates)
        object.__setattr__(self, "_decode_templates", decode_templates)
        object.__setattr__(self, "_prefill_chunk_count", prefill_chunk_count)
        object.__setattr__(self, "_decode_chunk_count", decode_chunk_count)
        object.__setattr__(self, "_length", length)
        object.__setattr__(self, "_binding_digest", binding_digest)

    @property
    def binding_digest(self) -> bytes:
        return self._binding_digest

    def __len__(self) -> int:
        return self._length

    def _templates_for_phase(
        self, phase: str
    ) -> tuple[_RuntimeStateTemplateReferenceV3, ...]:
        if phase == "prefill":
            return self._prefill_templates
        if phase == "decode":
            return self._decode_templates
        raise ProofV3Error("runtime tensor segment phase is unsupported")

    def _segment_for(
        self,
        *,
        phase: str,
        local_chunk_index: int,
        template: _RuntimeStateTemplateReferenceV3,
    ) -> GoldilocksRuntimeTensorSegmentReferenceV3:
        statement = self.trace_map_statement
        relation = statement.slot_universe.profile.relation_spec
        if phase == "prefill":
            if local_chunk_index >= self._prefill_chunk_count:
                raise ProofV3Error("runtime tensor segment chunk is out of range")
            chunk_index = local_chunk_index
            token_start = local_chunk_index * relation.prefill_chunk_tokens
            token_count = min(
                relation.prefill_chunk_tokens,
                statement.slot_universe.envelope.context_token_count - token_start,
            )
        else:
            if local_chunk_index >= self._decode_chunk_count:
                raise ProofV3Error("runtime tensor segment chunk is out of range")
            chunk_index = self._prefill_chunk_count + local_chunk_index
            token_start = statement.slot_universe.envelope.context_token_count + (
                local_chunk_index * relation.decode_chunk_tokens
            )
            token_count = min(
                relation.decode_chunk_tokens,
                statement.slot_universe.envelope.decode_token_count
                - local_chunk_index * relation.decode_chunk_tokens,
            )
        return GoldilocksRuntimeTensorSegmentReferenceV3(
            tensor_id=template.tensor_id,
            source_encoding_id=template.source_encoding_id,
            source_layout_id=template.source_layout_id,
            token_axis_rule_id=template.token_axis_rule_id,
            field_encoding_id=template.field_encoding_id,
            phase=phase,
            chunk_index=chunk_index,
            logical_token_start=token_start,
            source_token_start=(
                token_start - statement.slot_universe.envelope.context_token_count
                if template.token_axis_rule_id
                == GOLDILOCKS_RUNTIME_TOKEN_AXIS_DECODE_V3
                else token_start
            ),
            token_count=token_count,
            elements_per_token=template.elements_per_token,
            element_count=token_count * template.elements_per_token,
        )

    def __getitem__(self, index: int) -> GoldilocksRuntimeTensorSegmentReferenceV3:
        if type(index) is not int:
            raise TypeError("runtime tensor segment index must be an integer")
        if index < 0 or index >= self._length:
            raise IndexError("runtime tensor segment index is out of range")
        prefill_length = self._prefill_chunk_count * len(self._prefill_templates)
        if index < prefill_length:
            local_chunk, template_index = divmod(index, len(self._prefill_templates))
            return self._segment_for(
                phase="prefill",
                local_chunk_index=local_chunk,
                template=self._prefill_templates[template_index],
            )
        decode_index = index - prefill_length
        local_chunk, template_index = divmod(decode_index, len(self._decode_templates))
        return self._segment_for(
            phase="decode",
            local_chunk_index=local_chunk,
            template=self._decode_templates[template_index],
        )

    def __iter__(self) -> Iterator[GoldilocksRuntimeTensorSegmentReferenceV3]:
        for index in range(self._length):
            yield self[index]

    def _local_chunk_index(self, *, phase: str, chunk_index: int) -> int:
        if phase == "prefill":
            return chunk_index
        return chunk_index - self._prefill_chunk_count

    def segment_for_slot_binding(
        self,
        *,
        slot_index: object,
        binding: GoldilocksRuntimeTraceColumnBindingV3,
    ) -> GoldilocksRuntimeTensorSegmentReferenceV3:
        index = _u32(slot_index, "runtime tensor slot_index")
        statement = self.trace_map_statement
        if index >= len(statement.slot_universe):
            raise ProofV3Error("runtime tensor slot is out of range")
        if not isinstance(binding, GoldilocksRuntimeTraceColumnBindingV3):
            raise ProofV3Error("runtime tensor binding is malformed")
        slot = statement.slot_universe[index]
        program = statement.program_bundle.programs[slot.layout_index]
        if binding not in program.runtime_column_bindings:
            raise ProofV3Error("runtime tensor binding does not belong to this slot")
        templates = self._templates_for_phase(slot.phase)
        try:
            template = next(
                item for item in templates if item.tensor_id == binding.tensor_id
            )
        except StopIteration as exc:
            raise ProofV3Error("runtime tensor binding has no canonical segment") from exc
        expected = _RuntimeStateTemplateReferenceV3(
            tensor_id=binding.tensor_id,
            source_encoding_id=binding.source_encoding_id,
            source_layout_id=binding.source_layout_id,
            token_axis_rule_id=binding.token_axis_rule_id,
            field_encoding_id=binding.field_encoding_id,
            elements_per_token=binding.elements_per_token,
        )
        if template != expected:
            raise ProofV3Error("runtime tensor binding does not match its segment")
        return self._segment_for(
            phase=slot.phase,
            local_chunk_index=self._local_chunk_index(
                phase=slot.phase,
                chunk_index=slot.chunk_index,
            ),
            template=template,
        )

    def index_of(self, segment: GoldilocksRuntimeTensorSegmentReferenceV3) -> int:
        if not isinstance(segment, GoldilocksRuntimeTensorSegmentReferenceV3):
            raise ProofV3Error("runtime tensor segment is malformed")
        templates = self._templates_for_phase(segment.phase)
        local_chunk = self._local_chunk_index(
            phase=segment.phase,
            chunk_index=segment.chunk_index,
        )
        if local_chunk < 0:
            raise ProofV3Error("runtime tensor segment chunk is out of range")
        try:
            template_index = tuple(template.tensor_id for template in templates).index(
                segment.tensor_id
            )
        except ValueError as exc:
            raise ProofV3Error("runtime tensor segment tensor is out of range") from exc
        base = (
            0
            if segment.phase == "prefill"
            else self._prefill_chunk_count * len(self._prefill_templates)
        )
        index = base + local_chunk * len(templates) + template_index
        if index < 0 or index >= self._length or self[index] != segment:
            raise ProofV3Error("runtime tensor segment is not validator derived")
        return index

    def require_partial_runtime_state_slot(self, slot_index: object) -> tuple[int, ...]:
        """Return runtime-state segment indices for an intentionally partial check.

        It authenticates runtime-state columns only.  It does *not* establish
        a complete AIR slot: fixed/static columns still need a byte-table
        lookup relation, and cache/prompt/output/sampler roles need their own
        relations.  A final execution verifier must call
        :meth:`require_complete_slot` instead.
        """

        index = _u32(slot_index, "runtime tensor slot_index")
        statement = self.trace_map_statement
        if index >= len(statement.slot_universe):
            raise ProofV3Error("runtime tensor slot is out of range")
        slot = statement.slot_universe[index]
        program = statement.program_bundle.programs[slot.layout_index]
        if not program.has_exact_source_bindings or not program.runtime_column_bindings:
            raise ProofV3Error("runtime tensor slot lacks exact runtime bindings")
        tensors = {
            tensor.tensor_id: tensor
            for tensor in statement.slot_universe.profile.relation_spec.tensors
        }
        indices: set[int] = set()
        for binding in program.runtime_column_bindings:
            tensor = tensors.get(binding.tensor_id)
            if tensor is None or tensor.commitment_role != "runtime_state":
                raise ProofV3Error(
                    "runtime tensor reference does not support this slot source role"
                )
            indices.add(
                self.index_of(
                    self.segment_for_slot_binding(slot_index=index, binding=binding)
                )
            )
        return tuple(sorted(indices))

    def require_complete_slot(self, slot_index: object) -> tuple[int, ...]:
        """Fail closed unless this module can cover every non-auxiliary source.

        No current qualified fixture can pass this guard because static-table
        lookup has not landed.  Keeping the guard explicit prevents a future
        coordinator from accidentally presenting the partial runtime relation
        as a complete linear or transition witness.
        """

        indices = self.require_partial_runtime_state_slot(slot_index)
        slot = self.trace_map_statement.slot_universe[_u32(slot_index, "runtime tensor slot_index")]
        program = self.trace_map_statement.program_bundle.programs[slot.layout_index]
        if program.static_column_bindings:
            raise ProofV3Error(
                "runtime tensor reference cannot complete a slot with static columns"
            )
        return indices


@dataclass(frozen=True, slots=True, init=False)
class GoldilocksRuntimeTensorMapStatementReferenceV3:
    """Validator-owned statement for the second runtime-state commitment map."""

    trace_map_statement: GoldilocksAirTraceMapStatementReferenceV3
    segment_universe: GoldilocksRuntimeTensorSegmentUniverseReferenceV3
    binding_digest: bytes
    _factory_token: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ProofV3VerificationError(
            "runtime tensor map statements must use the validator factory"
        )

    @classmethod
    def _construct(
        cls,
        *,
        trace_map_statement: GoldilocksAirTraceMapStatementReferenceV3,
        segment_universe: GoldilocksRuntimeTensorSegmentUniverseReferenceV3,
        binding_digest: bytes,
        _factory_token: object | None = None,
    ) -> "GoldilocksRuntimeTensorMapStatementReferenceV3":
        if _factory_token is not _STATEMENT_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "runtime tensor map statements must use the validator factory"
            )
        result = object.__new__(cls)
        object.__setattr__(result, "trace_map_statement", trace_map_statement)
        object.__setattr__(result, "segment_universe", segment_universe)
        object.__setattr__(result, "binding_digest", binding_digest)
        object.__setattr__(result, "_factory_token", _STATEMENT_FACTORY_TOKEN)
        result.require_factory_provenance()
        return result

    def require_factory_provenance(self) -> None:
        if self._factory_token is not _STATEMENT_FACTORY_TOKEN:
            raise ProofV3VerificationError(
                "runtime tensor map statement lacks validator provenance"
            )
        if not isinstance(self.trace_map_statement, GoldilocksAirTraceMapStatementReferenceV3):
            raise ProofV3VerificationError("runtime tensor map trace statement is malformed")
        self.trace_map_statement.require_factory_provenance()
        if not isinstance(
            self.segment_universe,
            GoldilocksRuntimeTensorSegmentUniverseReferenceV3,
        ) or self.segment_universe.trace_map_statement is not self.trace_map_statement:
            raise ProofV3VerificationError("runtime tensor map segment universe is malformed")
        _fixed32(self.binding_digest, "runtime tensor map binding", nonzero=True)

    def canonical_bytes(self) -> bytes:
        return (
            self.trace_map_statement.digest()
            + self.segment_universe.binding_digest
            + struct.pack("<I", len(self.segment_universe))
            + self.binding_digest
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_STATEMENT_DIGEST_DOMAIN + self.canonical_bytes()).digest()


def derive_goldilocks_runtime_tensor_map_statement_reference_v3(
    *,
    trace_map_statement: GoldilocksAirTraceMapStatementReferenceV3,
) -> GoldilocksRuntimeTensorMapStatementReferenceV3:
    """Derive the tensor-map statement only from accepted trace-map state."""

    try:
        if not isinstance(trace_map_statement, GoldilocksAirTraceMapStatementReferenceV3):
            raise ProofV3VerificationError("runtime tensor map trace statement is malformed")
        trace_map_statement.require_factory_provenance()
        universe = GoldilocksRuntimeTensorSegmentUniverseReferenceV3(
            trace_map_statement=trace_map_statement
        )
        binding = hashlib.sha256(
            _STATEMENT_DOMAIN
            + trace_map_statement.digest()
            + trace_map_statement.binding_digest
            + universe.binding_digest
            + struct.pack("<I", len(universe))
        ).digest()
        return GoldilocksRuntimeTensorMapStatementReferenceV3._construct(
            trace_map_statement=trace_map_statement,
            segment_universe=universe,
            binding_digest=binding,
            _factory_token=_STATEMENT_FACTORY_TOKEN,
        )
    except ProofV3VerificationError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("runtime tensor map statement is malformed") from exc


def _canonical_segment_values(
    value: object,
    *,
    expected_count: int,
    name: str,
) -> tuple[int, ...]:
    """Materialize exactly one bounded canonical field vector."""

    if isinstance(value, (str, bytes, bytearray, memoryview)):
        raise ProofV3Error(f"{name} must be an iterable")
    try:
        iterator = iter(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ProofV3Error(f"{name} must be an iterable") from exc
    result: list[int] = []
    for index, item in enumerate(iterator):
        if index >= expected_count:
            raise ProofV3Error(f"{name} has too many elements")
        result.append(canonical_goldilocks(item, f"{name}[{index}]"))
    if len(result) != expected_count:
        raise ProofV3Error(f"{name} has an unexpected element count")
    return tuple(result)


def _segment_tree_binding_digest(
    *,
    statement_digest: bytes,
    universe_binding_digest: bytes,
    segment_index: int,
    segment: GoldilocksRuntimeTensorSegmentReferenceV3,
    tree_leaf_count: int,
) -> bytes:
    return hashlib.sha256(
        _SEGMENT_TREE_DOMAIN
        + statement_digest
        + universe_binding_digest
        + struct.pack("<II", segment_index, tree_leaf_count)
        + segment.canonical_bytes()
    ).digest()


def _map_tree_binding_digest(
    *,
    statement: GoldilocksRuntimeTensorMapStatementReferenceV3,
    tree_leaf_count: int,
) -> bytes:
    return hashlib.sha256(
        _MAP_TREE_DOMAIN
        + statement.digest()
        + statement.trace_map_statement.digest()
        + statement.segment_universe.binding_digest
        + struct.pack("<II", len(statement.segment_universe), tree_leaf_count)
    ).digest()


@dataclass(frozen=True, slots=True)
class GoldilocksRuntimeTensorSegmentPrecommitmentReferenceV3:
    """Frozen canonical field-vector root for one runtime-state segment.

    The values are already field-lowered test vectors.  This record does not
    decode fp16/int8 bytes or establish that those values came from a runtime;
    that encoding bridge is intentionally a separate required relation.
    """

    statement_digest: bytes
    universe_binding_digest: bytes
    segment_index: int
    segment: GoldilocksRuntimeTensorSegmentReferenceV3
    tree_leaf_count: int
    segment_tree_binding_digest: bytes
    segment_commitment: bytes
    abi_id: str = GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_ABI_V3
    format_version: int = GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_FORMAT_VERSION_V3

    def __post_init__(self) -> None:
        if self.abi_id != GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_ABI_V3:
            raise ProofV3Error("runtime tensor segment ABI is unsupported")
        if type(self.format_version) is not int or self.format_version != (
            GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_FORMAT_VERSION_V3
        ):
            raise ProofV3Error("runtime tensor segment format version is unsupported")
        statement_digest = _fixed32(
            self.statement_digest,
            "runtime tensor segment statement",
            nonzero=True,
        )
        universe_digest = _fixed32(
            self.universe_binding_digest,
            "runtime tensor segment universe",
            nonzero=True,
        )
        segment_index = _u32(
            self.segment_index,
            "runtime tensor segment index",
        )
        if not isinstance(self.segment, GoldilocksRuntimeTensorSegmentReferenceV3):
            raise ProofV3Error("runtime tensor segment descriptor is malformed")
        expected_leaf_count = _segment_tree_leaf_count(self.segment.element_count)
        if self.tree_leaf_count != expected_leaf_count:
            raise ProofV3Error("runtime tensor segment tree size is unexpected")
        expected_binding = _segment_tree_binding_digest(
            statement_digest=statement_digest,
            universe_binding_digest=universe_digest,
            segment_index=segment_index,
            segment=self.segment,
            tree_leaf_count=expected_leaf_count,
        )
        if self.segment_tree_binding_digest != expected_binding:
            raise ProofV3Error("runtime tensor segment tree binding is unexpected")
        commitment = _fixed32(
            self.segment_commitment,
            "runtime tensor segment commitment",
            nonzero=True,
        )
        object.__setattr__(self, "statement_digest", statement_digest)
        object.__setattr__(self, "universe_binding_digest", universe_digest)
        object.__setattr__(self, "segment_index", segment_index)
        object.__setattr__(self, "segment_commitment", commitment)

    def canonical_bytes(self) -> bytes:
        abi = self.abi_id.encode("ascii")
        return (
            struct.pack("<HH", self.format_version, len(abi))
            + abi
            + self.statement_digest
            + self.universe_binding_digest
            + struct.pack("<II", self.segment_index, self.tree_leaf_count)
            + self.segment.canonical_bytes()
            + self.segment_tree_binding_digest
            + self.segment_commitment
        )

    def digest(self) -> bytes:
        return hashlib.sha256(
            _SEGMENT_PRECOMMITMENT_DOMAIN + self.canonical_bytes()
        ).digest()


@dataclass(frozen=True, slots=True)
class GoldilocksRuntimeTensorSegmentOracleReferenceV3:
    """Small retained runtime-state segment oracle for CPU conformance only."""

    precommitment: GoldilocksRuntimeTensorSegmentPrecommitmentReferenceV3
    values: tuple[int, ...]
    segment_tree: GoldilocksMerkleTreeReference

    def __post_init__(self) -> None:
        if not isinstance(
            self.precommitment,
            GoldilocksRuntimeTensorSegmentPrecommitmentReferenceV3,
        ):
            raise ProofV3Error("runtime tensor segment precommitment is malformed")
        values = _canonical_segment_values(
            self.values,
            expected_count=self.precommitment.segment.element_count,
            name="runtime tensor segment values",
        )
        if not isinstance(self.segment_tree, GoldilocksMerkleTreeReference):
            raise ProofV3Error("runtime tensor segment tree is malformed")
        if (
            self.segment_tree.commitment != self.precommitment.segment_commitment
            or self.segment_tree.binding_digest
            != self.precommitment.segment_tree_binding_digest
            or self.segment_tree.leaf_count != self.precommitment.tree_leaf_count
            or self.segment_tree.leaf_width != 1
        ):
            raise ProofV3Error("runtime tensor segment tree does not match its root")
        padded_rows = tuple((item,) for item in values) + tuple(
            (0,)
            for _ in range(self.precommitment.tree_leaf_count - len(values))
        )
        if self.segment_tree.rows != padded_rows:
            raise ProofV3Error("runtime tensor segment tree values are unexpected")
        object.__setattr__(self, "values", values)


def build_goldilocks_runtime_tensor_segment_oracle_reference_v3(
    *,
    statement: GoldilocksRuntimeTensorMapStatementReferenceV3,
    segment_index: object,
    values: object,
) -> GoldilocksRuntimeTensorSegmentOracleReferenceV3:
    """Commit one validator-derived field segment in canonical order.

    The retained full vector is intentional for this CPU golden checker only;
    a qualified backend will stream its vector commitment and later use an
    all-row copy/lookup argument instead of sending these values.
    """

    if not isinstance(statement, GoldilocksRuntimeTensorMapStatementReferenceV3):
        raise ProofV3Error("runtime tensor map statement is malformed")
    statement.require_factory_provenance()
    index = _u32(segment_index, "runtime tensor segment index")
    if index >= len(statement.segment_universe):
        raise ProofV3Error("runtime tensor segment index is out of range")
    segment = statement.segment_universe[index]
    # Reject an unmaterializable CPU segment before consuming an arbitrary
    # caller iterable.  The native backend will stream/tile this shape.
    tree_leaf_count = _segment_tree_leaf_count(segment.element_count)
    normalized_values = _canonical_segment_values(
        values,
        expected_count=segment.element_count,
        name="runtime tensor segment values",
    )
    tree_binding = _segment_tree_binding_digest(
        statement_digest=statement.digest(),
        universe_binding_digest=statement.segment_universe.binding_digest,
        segment_index=index,
        segment=segment,
        tree_leaf_count=tree_leaf_count,
    )
    tree = GoldilocksMerkleTreeReference.from_rows(
        tuple((item,) for item in normalized_values)
        + tuple((0,) for _ in range(tree_leaf_count - len(normalized_values))),
        binding_digest=tree_binding,
    )
    precommitment = GoldilocksRuntimeTensorSegmentPrecommitmentReferenceV3(
        statement_digest=statement.digest(),
        universe_binding_digest=statement.segment_universe.binding_digest,
        segment_index=index,
        segment=segment,
        tree_leaf_count=tree_leaf_count,
        segment_tree_binding_digest=tree_binding,
        segment_commitment=tree.commitment,
    )
    return GoldilocksRuntimeTensorSegmentOracleReferenceV3(
        precommitment=precommitment,
        values=normalized_values,
        segment_tree=tree,
    )


@dataclass(frozen=True, slots=True)
class GoldilocksRuntimeTensorMapPrecommitmentReferenceV3:
    """Frozen root for every canonical runtime-state segment in one request."""

    statement: GoldilocksRuntimeTensorMapStatementReferenceV3
    tree_leaf_count: int
    map_tree_binding_digest: bytes
    runtime_tensor_map_commitment: bytes
    abi_id: str = GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_ABI_V3
    format_version: int = GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_FORMAT_VERSION_V3

    def __post_init__(self) -> None:
        if self.abi_id != GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_ABI_V3:
            raise ProofV3Error("runtime tensor map ABI is unsupported")
        if type(self.format_version) is not int or self.format_version != (
            GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_FORMAT_VERSION_V3
        ):
            raise ProofV3Error("runtime tensor map format version is unsupported")
        if not isinstance(self.statement, GoldilocksRuntimeTensorMapStatementReferenceV3):
            raise ProofV3Error("runtime tensor map statement is malformed")
        self.statement.require_factory_provenance()
        expected_leaf_count = _map_tree_leaf_count(len(self.statement.segment_universe))
        if self.tree_leaf_count != expected_leaf_count:
            raise ProofV3Error("runtime tensor map tree size is unexpected")
        expected_binding = _map_tree_binding_digest(
            statement=self.statement,
            tree_leaf_count=expected_leaf_count,
        )
        if self.map_tree_binding_digest != expected_binding:
            raise ProofV3Error("runtime tensor map tree binding is unexpected")
        root = _fixed32(
            self.runtime_tensor_map_commitment,
            "runtime tensor map commitment",
            nonzero=True,
        )
        object.__setattr__(self, "runtime_tensor_map_commitment", root)

    @property
    def segment_count(self) -> int:
        return len(self.statement.segment_universe)

    def canonical_bytes(self) -> bytes:
        abi = self.abi_id.encode("ascii")
        return (
            struct.pack("<HH", self.format_version, len(abi))
            + abi
            + self.statement.digest()
            + self.map_tree_binding_digest
            + struct.pack("<II", self.segment_count, self.tree_leaf_count)
            + self.runtime_tensor_map_commitment
        )

    def digest(self) -> bytes:
        return hashlib.sha256(
            _MAP_PRECOMMITMENT_DOMAIN + self.canonical_bytes()
        ).digest()


@dataclass(frozen=True, slots=True)
class GoldilocksRuntimeTensorMapOpeningReferenceV3:
    """Exact map opening for the required runtime-state segments."""

    map_opening: GoldilocksMerkleMultiOpeningReference
    segment_precommitments: tuple[GoldilocksRuntimeTensorSegmentPrecommitmentReferenceV3, ...]
    abi_id: str = GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_ABI_V3
    format_version: int = GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_FORMAT_VERSION_V3

    def __post_init__(self) -> None:
        if self.abi_id != GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_ABI_V3:
            raise ProofV3Error("runtime tensor map opening ABI is unsupported")
        if type(self.format_version) is not int or self.format_version != (
            GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_FORMAT_VERSION_V3
        ):
            raise ProofV3Error("runtime tensor map opening format version is unsupported")
        if not isinstance(self.map_opening, GoldilocksMerkleMultiOpeningReference):
            raise ProofV3Error("runtime tensor map Merkle opening is malformed")
        entries = tuple(self.segment_precommitments)
        if len(entries) != len(self.map_opening.indices) or not entries:
            raise ProofV3Error("runtime tensor map opening entry count is malformed")
        if not all(
            isinstance(item, GoldilocksRuntimeTensorSegmentPrecommitmentReferenceV3)
            for item in entries
        ):
            raise ProofV3Error("runtime tensor map opening entries are malformed")
        object.__setattr__(self, "segment_precommitments", entries)


@dataclass(frozen=True, slots=True)
class GoldilocksRuntimeTensorMapOracleReferenceV3:
    """Retained complete segment map for bounded CPU conformance only."""

    precommitment: GoldilocksRuntimeTensorMapPrecommitmentReferenceV3
    segment_oracles: tuple[GoldilocksRuntimeTensorSegmentOracleReferenceV3, ...]
    map_tree: GoldilocksMerkleTreeReference

    def __post_init__(self) -> None:
        if not isinstance(
            self.precommitment,
            GoldilocksRuntimeTensorMapPrecommitmentReferenceV3,
        ):
            raise ProofV3Error("runtime tensor map precommitment is malformed")
        oracles = tuple(self.segment_oracles)
        if len(oracles) != self.precommitment.segment_count or not all(
            isinstance(item, GoldilocksRuntimeTensorSegmentOracleReferenceV3)
            for item in oracles
        ):
            raise ProofV3Error("runtime tensor map oracle set is malformed")
        if not isinstance(self.map_tree, GoldilocksMerkleTreeReference):
            raise ProofV3Error("runtime tensor map tree is malformed")
        if (
            self.map_tree.commitment != self.precommitment.runtime_tensor_map_commitment
            or self.map_tree.binding_digest != self.precommitment.map_tree_binding_digest
            or self.map_tree.leaf_count != self.precommitment.tree_leaf_count
            or self.map_tree.leaf_width
            != GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_LEAF_WIDTH_V3
        ):
            raise ProofV3Error("runtime tensor map tree does not match its root")
        statement = self.precommitment.statement
        for index, oracle in enumerate(oracles):
            expected_segment = statement.segment_universe[index]
            precommitment = oracle.precommitment
            if (
                precommitment.statement_digest != statement.digest()
                or precommitment.universe_binding_digest
                != statement.segment_universe.binding_digest
                or precommitment.segment_index != index
                or precommitment.segment != expected_segment
            ):
                raise ProofV3Error("runtime tensor map oracle belongs to another segment")
            if self.map_tree.rows[index] != _digest_row(
                precommitment.digest(),
                "runtime tensor map entry digest",
            ):
                raise ProofV3Error("runtime tensor map tree has another segment entry")
        if any(
            row != (0,) * GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_LEAF_WIDTH_V3
            for row in self.map_tree.rows[self.precommitment.segment_count :]
        ):
            raise ProofV3Error("runtime tensor map tree has noncanonical padding")
        object.__setattr__(self, "segment_oracles", oracles)

    def open(self, segment_indices: object) -> GoldilocksRuntimeTensorMapOpeningReferenceV3:
        indices = _segment_indices(
            segment_indices,
            segment_count=self.precommitment.segment_count,
            name="runtime tensor map opening indices",
        )
        return GoldilocksRuntimeTensorMapOpeningReferenceV3(
            map_opening=self.map_tree.open(indices),
            segment_precommitments=tuple(
                self.segment_oracles[index].precommitment for index in indices
            ),
        )


def build_goldilocks_runtime_tensor_map_oracle_reference_v3(
    *,
    statement: GoldilocksRuntimeTensorMapStatementReferenceV3,
    segment_oracles: Sequence[GoldilocksRuntimeTensorSegmentOracleReferenceV3],
) -> GoldilocksRuntimeTensorMapOracleReferenceV3:
    """Commit every validator-derived segment root before nonce reveal."""

    if not isinstance(statement, GoldilocksRuntimeTensorMapStatementReferenceV3):
        raise ProofV3Error("runtime tensor map statement is malformed")
    statement.require_factory_provenance()
    try:
        oracles = tuple(segment_oracles)
    except TypeError as exc:
        raise ProofV3Error("runtime tensor map oracles are malformed") from exc
    if len(oracles) != len(statement.segment_universe):
        raise ProofV3Error("runtime tensor map omits or adds a segment")
    rows: list[tuple[int, ...]] = []
    for index, oracle in enumerate(oracles):
        if not isinstance(oracle, GoldilocksRuntimeTensorSegmentOracleReferenceV3):
            raise ProofV3Error("runtime tensor map oracle is malformed")
        expected_segment = statement.segment_universe[index]
        precommitment = oracle.precommitment
        if (
            precommitment.statement_digest != statement.digest()
            or precommitment.universe_binding_digest
            != statement.segment_universe.binding_digest
            or precommitment.segment_index != index
            or precommitment.segment != expected_segment
        ):
            raise ProofV3Error("runtime tensor map oracle belongs to another segment")
        rows.append(
            _digest_row(
                precommitment.digest(),
                "runtime tensor map entry digest",
            )
        )
    tree_leaf_count = _map_tree_leaf_count(len(rows))
    rows.extend(
        (0,) * GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_LEAF_WIDTH_V3
        for _ in range(tree_leaf_count - len(rows))
    )
    tree_binding = _map_tree_binding_digest(
        statement=statement,
        tree_leaf_count=tree_leaf_count,
    )
    tree = GoldilocksMerkleTreeReference.from_rows(rows, binding_digest=tree_binding)
    precommitment = GoldilocksRuntimeTensorMapPrecommitmentReferenceV3(
        statement=statement,
        tree_leaf_count=tree_leaf_count,
        map_tree_binding_digest=tree_binding,
        runtime_tensor_map_commitment=tree.commitment,
    )
    return GoldilocksRuntimeTensorMapOracleReferenceV3(
        precommitment=precommitment,
        segment_oracles=oracles,
        map_tree=tree,
    )


def verify_goldilocks_runtime_tensor_map_opening_reference_v3(
    opening: object,
    *,
    statement: GoldilocksRuntimeTensorMapStatementReferenceV3,
    precommitment: GoldilocksRuntimeTensorMapPrecommitmentReferenceV3,
    expected_segment_indices: object,
) -> tuple[GoldilocksRuntimeTensorSegmentPrecommitmentReferenceV3, ...]:
    """Verify exact selected segment roots from a frozen runtime-state map."""

    try:
        if not isinstance(statement, GoldilocksRuntimeTensorMapStatementReferenceV3):
            raise ProofV3VerificationError("runtime tensor map statement is malformed")
        statement.require_factory_provenance()
        if not isinstance(
            precommitment,
            GoldilocksRuntimeTensorMapPrecommitmentReferenceV3,
        ):
            raise ProofV3VerificationError("runtime tensor map precommitment is malformed")
        if precommitment.statement.digest() != statement.digest():
            raise ProofV3VerificationError("runtime tensor map belongs to another statement")
        indices = _segment_indices(
            expected_segment_indices,
            segment_count=precommitment.segment_count,
            name="expected runtime tensor map segment indices",
        )
        if not isinstance(opening, GoldilocksRuntimeTensorMapOpeningReferenceV3):
            raise ProofV3VerificationError("runtime tensor map opening is malformed")
        if opening.map_opening.indices != indices:
            raise ProofV3VerificationError(
                "runtime tensor map opening has unexpected segment indices"
            )
        verify_goldilocks_merkle_multiopening_reference(
            precommitment.runtime_tensor_map_commitment,
            opening.map_opening,
            expected_binding_digest=precommitment.map_tree_binding_digest,
            expected_leaf_count=precommitment.tree_leaf_count,
            expected_leaf_width=GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_LEAF_WIDTH_V3,
            expected_indices=indices,
        )
        selected: list[GoldilocksRuntimeTensorSegmentPrecommitmentReferenceV3] = []
        for index, row, segment_precommitment in zip(
            indices,
            opening.map_opening.rows,
            opening.segment_precommitments,
            strict=True,
        ):
            if not isinstance(
                segment_precommitment,
                GoldilocksRuntimeTensorSegmentPrecommitmentReferenceV3,
            ):
                raise ProofV3VerificationError("runtime tensor map entry is malformed")
            if (
                segment_precommitment.statement_digest != statement.digest()
                or segment_precommitment.universe_binding_digest
                != statement.segment_universe.binding_digest
                or segment_precommitment.segment_index != index
                or segment_precommitment.segment
                != statement.segment_universe[index]
            ):
                raise ProofV3VerificationError(
                    "runtime tensor map entry belongs to another segment"
                )
            if (
                _row_digest(row, "runtime tensor map entry row")
                != segment_precommitment.digest()
            ):
                raise ProofV3VerificationError(
                    "runtime tensor map entry digest does not match its leaf"
                )
            selected.append(segment_precommitment)
        return tuple(selected)
    except ProofV3VerificationError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("runtime tensor map opening is malformed") from exc


@dataclass(frozen=True, slots=True)
class GoldilocksRuntimeTensorBindingPrecommitmentPairReferenceV3:
    """One typed pair of trace-map and runtime-state-map roots.

    The pair removes an API footgun where callers could independently carry
    roots from different statements.  A validator-owned coordinator must
    still retain this exact pair atomically before nonce release; this
    in-memory type deliberately does not create network chronology itself.
    """

    trace_map_precommitment: GoldilocksAirTraceMapPrecommitmentReferenceV3
    runtime_tensor_map_precommitment: GoldilocksRuntimeTensorMapPrecommitmentReferenceV3
    binding_digest: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        trace_precommitment = self.trace_map_precommitment
        tensor_precommitment = self.runtime_tensor_map_precommitment
        if not isinstance(trace_precommitment, GoldilocksAirTraceMapPrecommitmentReferenceV3):
            raise ProofV3Error("runtime tensor binding trace precommitment is malformed")
        if not isinstance(
            tensor_precommitment,
            GoldilocksRuntimeTensorMapPrecommitmentReferenceV3,
        ):
            raise ProofV3Error("runtime tensor binding tensor precommitment is malformed")
        trace_statement = trace_precommitment.statement
        tensor_statement = tensor_precommitment.statement
        if tensor_statement.trace_map_statement.digest() != trace_statement.digest():
            raise ProofV3Error("runtime tensor binding maps belong to different statements")
        binding = hashlib.sha256(
            _PAIR_DOMAIN
            + trace_statement.digest()
            + trace_precommitment.digest()
            + tensor_precommitment.digest()
        ).digest()
        object.__setattr__(self, "binding_digest", binding)

    def canonical_bytes(self) -> bytes:
        return (
            self.trace_map_precommitment.digest()
            + self.runtime_tensor_map_precommitment.digest()
            + self.binding_digest
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_PAIR_DIGEST_DOMAIN + self.canonical_bytes()).digest()


@dataclass(frozen=True, slots=True)
class GoldilocksRuntimeStateTraceBindingWitnessReferenceV3:
    """Full CPU witness for a partial runtime-state trace conformance check.

    This is intentionally not a network proof.  It carries complete base
    trace rows and complete selected field segments so tests can establish
    the exact coordinate relation before a Goldilocks all-row copy/lookup
    argument replaces this retained-witness form.
    """

    slot_index: int
    trace_map_opening: GoldilocksAirTraceMapOpeningReferenceV3
    runtime_tensor_map_opening: GoldilocksRuntimeTensorMapOpeningReferenceV3
    base_trace: GoldilocksConstraintTraceReferenceV3
    segment_oracles: tuple[GoldilocksRuntimeTensorSegmentOracleReferenceV3, ...]

    def __post_init__(self) -> None:
        slot_index = _u32(self.slot_index, "runtime state witness slot_index")
        if not isinstance(self.trace_map_opening, GoldilocksAirTraceMapOpeningReferenceV3):
            raise ProofV3Error("runtime state witness trace-map opening is malformed")
        if not isinstance(
            self.runtime_tensor_map_opening,
            GoldilocksRuntimeTensorMapOpeningReferenceV3,
        ):
            raise ProofV3Error("runtime state witness tensor-map opening is malformed")
        if not isinstance(self.base_trace, GoldilocksConstraintTraceReferenceV3):
            raise ProofV3Error("runtime state witness base trace is malformed")
        oracles = tuple(self.segment_oracles)
        if not oracles or not all(
            isinstance(item, GoldilocksRuntimeTensorSegmentOracleReferenceV3)
            for item in oracles
        ):
            raise ProofV3Error("runtime state witness segment oracles are malformed")
        object.__setattr__(self, "slot_index", slot_index)
        object.__setattr__(self, "segment_oracles", oracles)


def _verify_runtime_state_trace_cells(
    *,
    statement: GoldilocksRuntimeTensorMapStatementReferenceV3,
    slot_index: int,
    core_program: GoldilocksConstraintProgramV3,
    base_trace: GoldilocksConstraintTraceReferenceV3,
    segment_oracles_by_index: dict[int, GoldilocksRuntimeTensorSegmentOracleReferenceV3],
) -> None:
    """Check every active runtime-state column against its canonical segment."""

    trace_statement = statement.trace_map_statement
    slot = trace_statement.slot_universe[slot_index]
    layout = trace_statement.constraint_system.layouts[slot.layout_index]
    tensors = {
        tensor.tensor_id: tensor
        for tensor in trace_statement.slot_universe.profile.relation_spec.tensors
    }
    column_positions = {
        column.column_id: position
        for position, column in enumerate(core_program.trace_columns)
    }
    for binding in core_program.runtime_column_bindings:
        tensor = tensors.get(binding.tensor_id)
        if tensor is None or tensor.commitment_role != "runtime_state":
            raise ProofV3VerificationError(
                "runtime state reference does not support this source role"
            )
        segment = statement.segment_universe.segment_for_slot_binding(
            slot_index=slot_index,
            binding=binding,
        )
        segment_index = statement.segment_universe.index_of(segment)
        segment_oracle = segment_oracles_by_index.get(segment_index)
        if segment_oracle is None:
            raise ProofV3VerificationError("runtime state witness omits a segment")
        if segment_oracle.precommitment.segment != segment:
            raise ProofV3VerificationError("runtime state witness has another segment")
        column_position = column_positions.get(binding.column_id)
        if column_position is None:
            raise ProofV3VerificationError("runtime state binding has an unknown column")
        last_subrow_element = (
            binding.element_offset
            + (layout.rows_per_token - 1) * binding.trace_row_stride
        )
        if last_subrow_element >= binding.elements_per_token:
            raise ProofV3VerificationError("runtime state binding exceeds its tensor row")
        for trace_row in range(slot.token_count * layout.rows_per_token):
            token_offset, subrow = divmod(trace_row, layout.rows_per_token)
            element_index = (
                token_offset * binding.elements_per_token
                + binding.element_offset
                + subrow * binding.trace_row_stride
            )
            if element_index >= len(segment_oracle.values):
                raise ProofV3VerificationError("runtime state binding exceeds its segment")
            if base_trace.rows[trace_row][column_position] != segment_oracle.values[
                element_index
            ]:
                raise ProofV3VerificationError(
                    "runtime state trace cell does not match its committed segment"
                )


def verify_goldilocks_runtime_state_trace_binding_reference_v3(
    witness: object,
    *,
    precommitment_pair: GoldilocksRuntimeTensorBindingPrecommitmentPairReferenceV3,
) -> None:
    """Verify a bounded *partial* runtime-state trace binding.

    This verifies a full base trace against a frozen trace-map entry and every
    exact runtime-state source column against frozen field segments.  It is
    deliberately insufficient for a complete slot or model-execution claim:
    it neither proves static lookup, cache RAM, raw quantized/fp16 lowering,
    prompt provenance, nor final token semantics.  It remains unregistered
    until a native all-row copy/lookup relation and those missing relations
    replace this CPU golden checker.
    """

    try:
        if not isinstance(
            precommitment_pair,
            GoldilocksRuntimeTensorBindingPrecommitmentPairReferenceV3,
        ):
            raise ProofV3VerificationError(
                "runtime state binding precommitment pair is malformed"
            )
        if not isinstance(witness, GoldilocksRuntimeStateTraceBindingWitnessReferenceV3):
            raise ProofV3VerificationError("runtime state binding witness is malformed")
        trace_precommitment = precommitment_pair.trace_map_precommitment
        tensor_precommitment = precommitment_pair.runtime_tensor_map_precommitment
        trace_statement = trace_precommitment.statement
        statement = tensor_precommitment.statement
        statement.require_factory_provenance()
        if statement.trace_map_statement.digest() != trace_statement.digest():
            raise ProofV3VerificationError(
                "runtime state binding maps belong to different statements"
            )
        slot_index = witness.slot_index
        if slot_index >= len(trace_statement.slot_universe):
            raise ProofV3VerificationError("runtime state binding slot is out of range")
        selected_trace_precommitments = verify_goldilocks_air_trace_map_opening_reference_v3(
            witness.trace_map_opening,
            statement=trace_statement,
            precommitment=trace_precommitment,
            expected_slot_indices=(slot_index,),
        )
        required_segment_indices = statement.segment_universe.require_partial_runtime_state_slot(
            slot_index
        )
        selected_segment_precommitments = (
            verify_goldilocks_runtime_tensor_map_opening_reference_v3(
                witness.runtime_tensor_map_opening,
                statement=statement,
                precommitment=tensor_precommitment,
                expected_segment_indices=required_segment_indices,
            )
        )
        if len(witness.segment_oracles) != len(required_segment_indices):
            raise ProofV3VerificationError(
                "runtime state witness has an unexpected segment-oracle count"
            )
        segment_oracles_by_index: dict[
            int, GoldilocksRuntimeTensorSegmentOracleReferenceV3
        ] = {}
        for segment_index, expected_precommitment, segment_oracle in zip(
            required_segment_indices,
            selected_segment_precommitments,
            witness.segment_oracles,
            strict=True,
        ):
            actual_precommitment = segment_oracle.precommitment
            if (
                actual_precommitment.digest() != expected_precommitment.digest()
                or actual_precommitment.segment_index != segment_index
            ):
                raise ProofV3VerificationError(
                    "runtime state witness segment does not match its map opening"
                )
            segment_oracles_by_index[segment_index] = segment_oracle
        core = trace_statement.slot_core(slot_index=slot_index)
        if witness.base_trace.constraint_program_digest != core.program.digest():
            raise ProofV3VerificationError("runtime state witness trace has another program")
        verify_goldilocks_constraint_program_reference_v3(
            program=core.program,
            trace=witness.base_trace,
            token_count=core.token_count,
        )
        reconstructed_oracle = build_goldilocks_air_trace_oracle_reference_v3(
            program=core.program,
            trace=witness.base_trace,
            token_count=core.token_count,
            validator_binding_digest=core.validator_binding_digest,
        )
        if (
            reconstructed_oracle.precommitment.digest()
            != selected_trace_precommitments[0].digest()
        ):
            raise ProofV3VerificationError(
                "runtime state witness trace does not match its frozen trace root"
            )
        _verify_runtime_state_trace_cells(
            statement=statement,
            slot_index=slot_index,
            core_program=core.program,
            base_trace=witness.base_trace,
            segment_oracles_by_index=segment_oracles_by_index,
        )
    except ProofV3VerificationError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("runtime state binding witness is malformed") from exc


__all__ = [
    "GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_ABI_V3",
    "GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_FORMAT_VERSION_V3",
    "GOLDILOCKS_RUNTIME_TENSOR_BINDING_REFERENCE_LEAF_WIDTH_V3",
    "GoldilocksRuntimeStateTraceBindingWitnessReferenceV3",
    "GoldilocksRuntimeTensorBindingPrecommitmentPairReferenceV3",
    "GoldilocksRuntimeTensorMapOpeningReferenceV3",
    "GoldilocksRuntimeTensorMapOracleReferenceV3",
    "GoldilocksRuntimeTensorMapPrecommitmentReferenceV3",
    "GoldilocksRuntimeTensorMapStatementReferenceV3",
    "GoldilocksRuntimeTensorSegmentOracleReferenceV3",
    "GoldilocksRuntimeTensorSegmentPrecommitmentReferenceV3",
    "GoldilocksRuntimeTensorSegmentReferenceV3",
    "GoldilocksRuntimeTensorSegmentUniverseReferenceV3",
    "build_goldilocks_runtime_tensor_map_oracle_reference_v3",
    "build_goldilocks_runtime_tensor_segment_oracle_reference_v3",
    "derive_goldilocks_runtime_tensor_map_statement_reference_v3",
    "verify_goldilocks_runtime_state_trace_binding_reference_v3",
    "verify_goldilocks_runtime_tensor_map_opening_reference_v3",
]
