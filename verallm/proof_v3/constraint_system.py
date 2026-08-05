"""Canonical Goldilocks constraint-system topology for future V3 AIR proofs.

This module defines the validator-owned statement that a future native
Goldilocks AIR/RAM backend must consume.  It is deliberately only a parsed,
signed *constraint-system contract*: it is not a STARK/FRI proof, does not
evaluate a model, and is not wired into a miner or validator runtime.

The contract removes unsafe degrees of freedom from the future proof wire
format. A prover may not choose relation labels, graph nodes, chunks,
constraint-template indices, or an opaque program for a signed layout. The
authenticated artifact fixes ordered templates and a content-addressed parsed
program bundle; the verifier derives the complete lazy universe from the
signed profile and validator-bound request/output counts. A template denotes a
constraint polynomial over an AIR trace domain, never one individual token
row. Consequently a 1M-token request scales with chunks and templates rather
than materialising one coordinate per token or tensor element.

It intentionally does not claim that declaring a topology proves any
transition.  Qualification still requires a native witness backend, exact
adapter semantics, a RAM relation, a polynomial commitment/FRI verifier, and
adversarial end-to-end evidence.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import re
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.goldilocks_reference import (
    GOLDILOCKS_MODULUS,
    GOLDILOCKS_TWO_ADICITY,
)
from verallm.proof_v3.payload import ProofV3CommitmentEnvelope
from verallm.proof_v3.profile import ExecutionSecurityProfileV3
from verallm.proof_v3.relation import (
    ExecutionRelationNodeV3,
    ExecutionRelationSpecV3,
)
from verallm.proof_v3.static_artifact import (
    GOLDILOCKS_DYNAMIC_BACKEND_ABI_ID_V3,
    GOLDILOCKS_STATIC_FIELD_ID_V3,
)


GOLDILOCKS_EXECUTION_CONSTRAINT_SYSTEM_ABI_V3 = (
    "goldilocks.execution_constraints.v2"
)
GOLDILOCKS_EXECUTION_CONSTRAINT_SYSTEM_FORMAT_VERSION_V3 = 2
GOLDILOCKS_TRACE_DOMAIN_RULE_V3 = "radix2.next_power_of_two.v1"
GOLDILOCKS_TRACE_PADDING_RULE_V3 = "zero_pad.inactive_rows.v1"

MAX_GOLDILOCKS_CONSTRAINT_SYSTEM_BYTES_V3 = 16 << 20
MAX_GOLDILOCKS_CONSTRAINT_LAYOUTS_V3 = 65_535
MAX_GOLDILOCKS_ATOMIC_CONSTRAINTS_PER_LAYOUT_V3 = 65_535
MAX_GOLDILOCKS_ROWS_PER_TOKEN_V3 = 1 << 20
MAX_GOLDILOCKS_TRACE_DOMAIN_SIZE_V3 = 1 << GOLDILOCKS_TWO_ADICITY
# A rate-one LDE leaves no low-degree redundancy. This is only a structural
# floor; a qualified backend may require a larger reviewed blowup.
MIN_GOLDILOCKS_LDE_BLOWUP_V3 = 2
# This is a protocol cap, not ``sys.maxsize``: validators on different Python
# builds must derive the same accepted coordinate universe.
MAX_GOLDILOCKS_CONSTRAINT_UNIVERSE_V3 = (1 << 31) - 1

_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,127}$")
_HEX_32_RE = re.compile(r"^[0-9a-f]{64}$")
_PHASE_CODES = {"prefill": 1, "decode": 2}
_PHASE_ORDER = ("prefill", "decode")
_RELATION_BINDING_DOMAIN_V3 = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_CONSTRAINT_SYSTEM/RELATION/SHA256"
)
_COEFFICIENT_DOMAIN_V3 = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_CONSTRAINT_SYSTEM/COEFFICIENT/SHA256"
)
_UNIVERSE_BINDING_DOMAIN_V3 = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_CONSTRAINT_SYSTEM/UNIVERSE_BINDING/SHA256"
)
_MAX_REJECTION_ATTEMPTS = 4096
_CONSTRAINT_TRANSCRIPT_FACTORY_TOKEN_V3 = object()


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ProofV3Error(f"{name} is not a canonical identifier")
    return value


def _fixed32(value: object, name: str, *, nonzero: bool = False) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    if nonzero and value == bytes(32):
        raise ProofV3Error(f"{name} must not be the zero digest")
    return value


def _u32(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise ProofV3Error(f"{name} must be an unsigned 32-bit integer")
    if value < (1 if positive else 0) or value >= 1 << 32:
        qualifier = "positive " if positive else ""
        raise ProofV3Error(f"{name} must be a {qualifier}unsigned 32-bit integer")
    return value


def _u64(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise ProofV3Error(f"{name} must be an unsigned 64-bit integer")
    if value < (1 if positive else 0) or value >= 1 << 64:
        qualifier = "positive " if positive else ""
        raise ProofV3Error(f"{name} must be a {qualifier}unsigned 64-bit integer")
    return value


def _power_of_two(value: object, name: str, *, maximum: int) -> int:
    integer = _u64(value, name, positive=True)
    if integer > maximum or integer & (integer - 1):
        raise ProofV3Error(f"{name} must be a power of two within the protocol limit")
    return integer


def _bounded_tuple(
    value: object,
    *,
    name: str,
    maximum: int,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ProofV3Error(f"{name} must be a tuple")
    if len(value) > maximum or (not allow_empty and not value):
        raise ProofV3Error(f"{name} has an invalid length")
    result = tuple(_identifier(item, f"{name}[{index}]") for index, item in enumerate(value))
    return result


def _sorted_distinct(value: tuple[str, ...], name: str) -> tuple[str, ...]:
    if value != tuple(sorted(value)) or len(value) != len(set(value)):
        raise ProofV3Error(f"{name} must be sorted and distinct")
    return value


def _next_power_of_two(value: int, name: str) -> int:
    if value <= 0:
        raise ProofV3Error(f"{name} must be positive")
    if value > MAX_GOLDILOCKS_TRACE_DOMAIN_SIZE_V3:
        raise ProofV3Error(f"{name} exceeds the Goldilocks trace-domain limit")
    return 1 << (value - 1).bit_length()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProofV3Error("constraint system contains duplicate JSON object keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ProofV3Error(f"constraint system contains unsupported JSON constant {value}")


def _object(value: object, keys: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ProofV3Error(f"{name} fields do not match the canonical schema")
    return value


def _list(value: object, name: str, maximum: int, *, allow_empty: bool = False) -> list[object]:
    if not isinstance(value, list) or len(value) > maximum or (not allow_empty and not value):
        raise ProofV3Error(f"{name} must be a bounded list")
    return value


def _json_identifier(value: object, name: str) -> str:
    return _identifier(value, name)


def _json_u32(value: object, name: str, *, positive: bool = False) -> int:
    return _u32(value, name, positive=positive)


def _json_u64(value: object, name: str, *, positive: bool = False) -> int:
    return _u64(value, name, positive=positive)


def _json_digest(value: object, name: str) -> bytes:
    if not isinstance(value, str) or _HEX_32_RE.fullmatch(value) is None:
        raise ProofV3Error(f"{name} must be a lowercase 32-byte hexadecimal digest")
    return bytes.fromhex(value)


def _json_identifier_tuple(
    value: object,
    *,
    name: str,
    maximum: int,
    sorted_distinct: bool,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    entries = _list(value, name, maximum, allow_empty=allow_empty)
    result = tuple(_json_identifier(item, f"{name}[{index}]") for index, item in enumerate(entries))
    if sorted_distinct:
        _sorted_distinct(result, name)
    return result


def _canonical_json_bytes(value: object, *, name: str, maximum: int) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    if not encoded or len(encoded) > maximum:
        raise ProofV3Error(f"{name} exceeds the protocol size limit")
    return encoded


def _relation_projection_dict(relation: ExecutionRelationSpecV3) -> dict[str, object]:
    """Return the constraint-system-relevant relation projection.

    The relation carries the raw SHA-256 of this artifact, so including that
    field would make a self-hash.  The projection also intentionally excludes
    separately authenticated capture/tokenizer/quantization/recursive artifact
    digests.  Those legs have their own exact catalog checks and should not
    make a graph-topology artifact silently stand in for them.
    """

    if not isinstance(relation, ExecutionRelationSpecV3):
        raise ProofV3Error("constraint-system relation has an unexpected type")
    source = relation.to_dict()
    keys = (
        "cache",
        "decode_chunk_tokens",
        "dimensions",
        "final_token_tensor_id",
        "nodes",
        "prefill_chunk_tokens",
        "registered_operations",
        "relation_abi_id",
        "relation_spec_version",
        "request_tensor_id",
        "sequence_domain",
        "static_bindings",
        "static_table_bindings",
        "tensors",
        "tolerances",
    )
    return {key: source[key] for key in keys}


def constraint_system_relation_projection_bytes_v3(
    relation: ExecutionRelationSpecV3,
) -> bytes:
    """Canonical relation projection bound by a constraint-system artifact."""

    return _canonical_json_bytes(
        _relation_projection_dict(relation),
        name="constraint-system relation projection",
        maximum=MAX_GOLDILOCKS_CONSTRAINT_SYSTEM_BYTES_V3,
    )


def constraint_system_relation_binding_digest_v3(
    relation: ExecutionRelationSpecV3,
) -> bytes:
    """Domain-separated digest of the non-circular relation projection."""

    return hashlib.sha256(
        _RELATION_BINDING_DOMAIN_V3
        + constraint_system_relation_projection_bytes_v3(relation)
    ).digest()


def expected_static_table_ids_for_relation_node_v3(
    *,
    relation: ExecutionRelationSpecV3,
    node: ExecutionRelationNodeV3,
) -> tuple[str, ...]:
    """Derive the exact static-table set consumed by one signed graph node."""

    if not isinstance(relation, ExecutionRelationSpecV3):
        raise ProofV3Error("constraint-system relation has an unexpected type")
    if not isinstance(node, ExecutionRelationNodeV3):
        raise ProofV3Error("constraint-system node has an unexpected type")
    expected = _expected_static_table_ids_by_node_v3(relation)
    nodes_by_id = {item.node_id: item for item in relation.nodes}
    if nodes_by_id.get(node.node_id) != node or node.node_id not in expected:
        raise ProofV3Error("constraint-system node is not part of the signed graph")
    return expected[node.node_id]


def _expected_static_table_ids_by_node_v3(
    relation: ExecutionRelationSpecV3,
) -> dict[str, tuple[str, ...]]:
    """Build the exact table map once instead of rescanning it per node."""

    if not isinstance(relation, ExecutionRelationSpecV3):
        raise ProofV3Error("constraint-system relation has an unexpected type")
    operation_tables = {
        item.operation_descriptor_digest: item.table_id
        for item in relation.static_table_bindings
        if item.subject_kind == "operation"
    }
    static_binding_tables = {
        item.static_binding_id: item.table_id
        for item in relation.static_table_bindings
        if item.subject_kind == "static_parameter"
    }
    result: dict[str, tuple[str, ...]] = {}
    for node in relation.nodes:
        if node.operation_reference is not None:
            table_id = operation_tables.get(node.operation_reference.descriptor_digest)
            if table_id is None:
                raise ProofV3Error(
                    "signed relation does not define an exact operation static table"
                )
            result[node.node_id] = (table_id,)
            continue
        table_ids: list[str] = []
        for binding_id in node.static_binding_ids:
            table_id = static_binding_tables.get(binding_id)
            if table_id is None:
                raise ProofV3Error(
                    "signed relation does not define an exact static binding table"
                )
            table_ids.append(table_id)
        if not table_ids or len(table_ids) != len(set(table_ids)):
            raise ProofV3Error(
                "signed relation does not define one exact static table per node binding"
            )
        result[node.node_id] = tuple(sorted(table_ids))
    return result


def expected_constraint_layout_phases_for_relation_node_v3(
    *,
    relation: ExecutionRelationSpecV3,
    node: ExecutionRelationNodeV3,
) -> tuple[str, ...]:
    """Derive the mandatory phase set from signed node tensor time domains.

    The current relation ABI has no separate free-form phase field.  A node is
    eligible only in phases where *every* declared input and output tensor is
    defined: context-indexed tensors are prefill-only, decode-indexed tensors
    are decode-only, and sequence-indexed tensors support both.  Taking a
    union of input phases would incorrectly create, for example, a prefill
    trace that writes a decode-only final-hidden tensor.  A constraint-system
    artifact may not broaden or narrow this signed data-flow intersection.
    """

    if not isinstance(relation, ExecutionRelationSpecV3):
        raise ProofV3Error("constraint-system relation has an unexpected type")
    if not isinstance(node, ExecutionRelationNodeV3):
        raise ProofV3Error("constraint-system node has an unexpected type")
    expected = _expected_constraint_layout_phases_by_node_v3(relation)
    nodes_by_id = {item.node_id: item for item in relation.nodes}
    if nodes_by_id.get(node.node_id) != node or node.node_id not in expected:
        raise ProofV3Error("constraint-system node is not part of the signed graph")
    return expected[node.node_id]


def _expected_constraint_layout_phases_by_node_v3(
    relation: ExecutionRelationSpecV3,
) -> dict[str, tuple[str, ...]]:
    """Build the tensor-domain phase plan once for all signed graph nodes."""

    if not isinstance(relation, ExecutionRelationSpecV3):
        raise ProofV3Error("constraint-system relation has an unexpected type")
    tensors = {tensor.tensor_id: tensor for tensor in relation.tensors}
    context_dimension = relation.sequence_domain.context_dimension_id
    decode_dimension = relation.sequence_domain.decode_dimension_id
    sequence_dimension = relation.sequence_domain.sequence_dimension_id
    result: dict[str, tuple[str, ...]] = {}
    for node in relation.nodes:
        phases: set[str] | None = None
        for tensor_id in node.input_tensor_ids + node.output_tensor_ids:
            tensor = tensors.get(tensor_id)
            if tensor is None:
                raise ProofV3Error(
                    "constraint-system node references an unknown tensor"
                )
            tensor_phases: set[str] = set()
            if context_dimension in tensor.shape or sequence_dimension in tensor.shape:
                tensor_phases.add("prefill")
            if decode_dimension in tensor.shape or sequence_dimension in tensor.shape:
                tensor_phases.add("decode")
            if not tensor_phases:
                raise ProofV3Error(
                    "constraint-system node tensor has no signed token-phase coverage"
                )
            phases = (
                tensor_phases
                if phases is None
                else phases.intersection(tensor_phases)
            )
        assert phases is not None
        node_phases = tuple(phase for phase in _PHASE_ORDER if phase in phases)
        if not node_phases:
            raise ProofV3Error(
                "constraint-system node has no shared signed token-phase coverage"
            )
        result[node.node_id] = node_phases
    return result


@dataclass(frozen=True, slots=True)
class GoldilocksConstraintLayoutV3:
    """One ordered graph-node plan for a Goldilocks AIR trace.

    ``atomic_constraint_ids`` identify canonical constraint-polynomial
    templates.  They are not rows: each template applies to every active row
    in its verifier-derived chunk trace.  A qualified native adapter binds the
    program digest and gives these identifiers their exact arithmetic meaning.
    """

    node_id: str
    relation_id: str
    transition_adapter_id: str
    layer_index: int
    phases: tuple[str, ...]
    runtime_tensor_ids: tuple[str, ...]
    static_binding_ids: tuple[str, ...]
    static_table_ids: tuple[str, ...]
    atomic_constraint_ids: tuple[str, ...]
    constraint_program_digest: bytes
    rows_per_token: int
    minimum_trace_rows: int
    lde_blowup: int
    max_constraint_degree: int
    trace_domain_rule_id: str = GOLDILOCKS_TRACE_DOMAIN_RULE_V3
    padding_rule_id: str = GOLDILOCKS_TRACE_PADDING_RULE_V3

    def __post_init__(self) -> None:
        _identifier(self.node_id, "constraint layout node_id")
        _identifier(self.relation_id, "constraint layout relation_id")
        _identifier(
            self.transition_adapter_id,
            "constraint layout transition_adapter_id",
        )
        if type(self.layer_index) is not int or not -(1 << 31) <= self.layer_index < 1 << 31:
            raise ProofV3Error("constraint layout layer_index is malformed")
        phases = _bounded_tuple(
            self.phases,
            name="constraint layout phases",
            maximum=len(_PHASE_ORDER),
        )
        if phases != tuple(phase for phase in _PHASE_ORDER if phase in phases):
            raise ProofV3Error("constraint layout phases are not canonically ordered")
        if any(phase not in _PHASE_CODES for phase in phases):
            raise ProofV3Error("constraint layout phase is unsupported")
        runtime_tensors = _bounded_tuple(
            self.runtime_tensor_ids,
            name="constraint layout runtime_tensor_ids",
            maximum=64,
        )
        if len(runtime_tensors) != len(set(runtime_tensors)):
            raise ProofV3Error("constraint layout runtime tensors contain duplicates")
        static_bindings = _bounded_tuple(
            self.static_binding_ids,
            name="constraint layout static_binding_ids",
            maximum=32,
            allow_empty=True,
        )
        if len(static_bindings) != len(set(static_bindings)):
            raise ProofV3Error("constraint layout static bindings contain duplicates")
        static_tables = _sorted_distinct(
            _bounded_tuple(
                self.static_table_ids,
                name="constraint layout static_table_ids",
                maximum=32,
                allow_empty=True,
            ),
            "constraint layout static_table_ids",
        )
        atomics = _sorted_distinct(
            _bounded_tuple(
                self.atomic_constraint_ids,
                name="constraint layout atomic_constraint_ids",
                maximum=MAX_GOLDILOCKS_ATOMIC_CONSTRAINTS_PER_LAYOUT_V3,
            ),
            "constraint layout atomic_constraint_ids",
        )
        _fixed32(
            self.constraint_program_digest,
            "constraint layout constraint_program_digest",
            nonzero=True,
        )
        rows_per_token = _u32(
            self.rows_per_token,
            "constraint layout rows_per_token",
            positive=True,
        )
        if rows_per_token > MAX_GOLDILOCKS_ROWS_PER_TOKEN_V3:
            raise ProofV3Error("constraint layout rows_per_token exceeds the limit")
        _power_of_two(
            self.minimum_trace_rows,
            "constraint layout minimum_trace_rows",
            maximum=MAX_GOLDILOCKS_TRACE_DOMAIN_SIZE_V3,
        )
        lde_blowup = _power_of_two(
            self.lde_blowup,
            "constraint layout lde_blowup",
            maximum=MAX_GOLDILOCKS_TRACE_DOMAIN_SIZE_V3,
        )
        if lde_blowup < MIN_GOLDILOCKS_LDE_BLOWUP_V3:
            raise ProofV3Error("constraint layout LDE blowup is below the minimum")
        _u32(
            self.max_constraint_degree,
            "constraint layout max_constraint_degree",
            positive=True,
        )
        if self.trace_domain_rule_id != GOLDILOCKS_TRACE_DOMAIN_RULE_V3:
            raise ProofV3Error("constraint layout trace-domain rule is unsupported")
        if self.padding_rule_id != GOLDILOCKS_TRACE_PADDING_RULE_V3:
            raise ProofV3Error("constraint layout padding rule is unsupported")
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "runtime_tensor_ids", runtime_tensors)
        object.__setattr__(self, "static_binding_ids", static_bindings)
        object.__setattr__(self, "static_table_ids", static_tables)
        object.__setattr__(self, "atomic_constraint_ids", atomics)

    def to_dict(self) -> dict[str, object]:
        return {
            "atomic_constraint_ids": list(self.atomic_constraint_ids),
            "constraint_program_digest": self.constraint_program_digest.hex(),
            "layer_index": self.layer_index,
            "lde_blowup": self.lde_blowup,
            "max_constraint_degree": self.max_constraint_degree,
            "minimum_trace_rows": self.minimum_trace_rows,
            "node_id": self.node_id,
            "padding_rule_id": self.padding_rule_id,
            "phases": list(self.phases),
            "relation_id": self.relation_id,
            "rows_per_token": self.rows_per_token,
            "runtime_tensor_ids": list(self.runtime_tensor_ids),
            "static_binding_ids": list(self.static_binding_ids),
            "static_table_ids": list(self.static_table_ids),
            "trace_domain_rule_id": self.trace_domain_rule_id,
            "transition_adapter_id": self.transition_adapter_id,
        }

    def trace_domain_size(self, *, token_count: int) -> int:
        """Return the exact radix-2 trace size for a validator-derived chunk."""

        count = _u32(token_count, "constraint trace token_count", positive=True)
        active_rows = count * self.rows_per_token
        if active_rows > MAX_GOLDILOCKS_TRACE_DOMAIN_SIZE_V3:
            raise ProofV3Error("constraint trace active rows exceed Goldilocks two-adicity")
        physical_rows = max(
            self.minimum_trace_rows,
            _next_power_of_two(active_rows, "constraint trace active rows"),
        )
        if physical_rows * self.lde_blowup > MAX_GOLDILOCKS_TRACE_DOMAIN_SIZE_V3:
            raise ProofV3Error("constraint trace LDE exceeds Goldilocks two-adicity")
        return physical_rows


def _layout_from_dict(value: object) -> GoldilocksConstraintLayoutV3:
    item = _object(
        value,
        {
            "atomic_constraint_ids",
            "constraint_program_digest",
            "layer_index",
            "lde_blowup",
            "max_constraint_degree",
            "minimum_trace_rows",
            "node_id",
            "padding_rule_id",
            "phases",
            "relation_id",
            "rows_per_token",
            "runtime_tensor_ids",
            "static_binding_ids",
            "static_table_ids",
            "trace_domain_rule_id",
            "transition_adapter_id",
        },
        "constraint layout",
    )
    layer_index = item["layer_index"]
    if type(layer_index) is not int or not -(1 << 31) <= layer_index < 1 << 31:
        raise ProofV3Error("constraint layout layer_index is malformed")
    return GoldilocksConstraintLayoutV3(
        node_id=_json_identifier(item["node_id"], "constraint layout node_id"),
        relation_id=_json_identifier(
            item["relation_id"], "constraint layout relation_id"
        ),
        transition_adapter_id=_json_identifier(
            item["transition_adapter_id"],
            "constraint layout transition_adapter_id",
        ),
        layer_index=layer_index,
        phases=_json_identifier_tuple(
            item["phases"],
            name="constraint layout phases",
            maximum=len(_PHASE_ORDER),
            sorted_distinct=False,
        ),
        runtime_tensor_ids=_json_identifier_tuple(
            item["runtime_tensor_ids"],
            name="constraint layout runtime_tensor_ids",
            maximum=64,
            sorted_distinct=False,
        ),
        static_binding_ids=_json_identifier_tuple(
            item["static_binding_ids"],
            name="constraint layout static_binding_ids",
            maximum=32,
            sorted_distinct=False,
            allow_empty=True,
        ),
        static_table_ids=_json_identifier_tuple(
            item["static_table_ids"],
            name="constraint layout static_table_ids",
            maximum=32,
            sorted_distinct=True,
            allow_empty=True,
        ),
        atomic_constraint_ids=_json_identifier_tuple(
            item["atomic_constraint_ids"],
            name="constraint layout atomic_constraint_ids",
            maximum=MAX_GOLDILOCKS_ATOMIC_CONSTRAINTS_PER_LAYOUT_V3,
            sorted_distinct=True,
        ),
        constraint_program_digest=_json_digest(
            item["constraint_program_digest"],
            "constraint layout constraint_program_digest",
        ),
        rows_per_token=_json_u32(
            item["rows_per_token"],
            "constraint layout rows_per_token",
            positive=True,
        ),
        minimum_trace_rows=_json_u64(
            item["minimum_trace_rows"],
            "constraint layout minimum_trace_rows",
            positive=True,
        ),
        lde_blowup=_json_u64(
            item["lde_blowup"],
            "constraint layout lde_blowup",
            positive=True,
        ),
        max_constraint_degree=_json_u32(
            item["max_constraint_degree"],
            "constraint layout max_constraint_degree",
            positive=True,
        ),
        trace_domain_rule_id=_json_identifier(
            item["trace_domain_rule_id"],
            "constraint layout trace_domain_rule_id",
        ),
        padding_rule_id=_json_identifier(
            item["padding_rule_id"],
            "constraint layout padding_rule_id",
        ),
    )


@dataclass(frozen=True, slots=True)
class GoldilocksExecutionConstraintSystemV3:
    """Canonical, signed topology for the dynamic Goldilocks proof backend."""

    relation_binding_digest: bytes
    constraint_program_bundle_digest: bytes
    layouts: tuple[GoldilocksConstraintLayoutV3, ...]
    constraint_system_abi_id: str = GOLDILOCKS_EXECUTION_CONSTRAINT_SYSTEM_ABI_V3
    field_id: str = GOLDILOCKS_STATIC_FIELD_ID_V3
    dynamic_backend_abi_id: str = GOLDILOCKS_DYNAMIC_BACKEND_ABI_ID_V3
    format_version: int = GOLDILOCKS_EXECUTION_CONSTRAINT_SYSTEM_FORMAT_VERSION_V3
    _relation_index_offsets: tuple[int, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _raw_digest: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.constraint_system_abi_id != GOLDILOCKS_EXECUTION_CONSTRAINT_SYSTEM_ABI_V3:
            raise ProofV3Error("constraint-system ABI is unsupported")
        if self.field_id != GOLDILOCKS_STATIC_FIELD_ID_V3:
            raise ProofV3Error("constraint-system field is unsupported")
        if self.dynamic_backend_abi_id != GOLDILOCKS_DYNAMIC_BACKEND_ABI_ID_V3:
            raise ProofV3Error("constraint-system dynamic backend ABI is unsupported")
        _u32(
            self.format_version,
            "constraint-system format_version",
            positive=True,
        )
        if self.format_version != GOLDILOCKS_EXECUTION_CONSTRAINT_SYSTEM_FORMAT_VERSION_V3:
            raise ProofV3Error("constraint-system format version is unsupported")
        _fixed32(
            self.relation_binding_digest,
            "constraint-system relation_binding_digest",
            nonzero=True,
        )
        _fixed32(
            self.constraint_program_bundle_digest,
            "constraint-system constraint_program_bundle_digest",
            nonzero=True,
        )
        if not isinstance(self.layouts, tuple) or not self.layouts:
            raise ProofV3Error("constraint-system layouts must be a nonempty tuple")
        if len(self.layouts) > MAX_GOLDILOCKS_CONSTRAINT_LAYOUTS_V3 or not all(
            isinstance(layout, GoldilocksConstraintLayoutV3) for layout in self.layouts
        ):
            raise ProofV3Error("constraint-system layouts are malformed")
        offsets: list[int] = []
        current = 0
        for layout in self.layouts:
            offsets.append(current)
            current += len(layout.atomic_constraint_ids)
            if current >= 1 << 64:
                raise ProofV3Error("constraint-system atomic relation index overflows")
        object.__setattr__(self, "_relation_index_offsets", tuple(offsets))
        object.__setattr__(
            self,
            "_raw_digest",
            hashlib.sha256(self.canonical_bytes()).digest(),
        )

    @property
    def atomic_constraint_count(self) -> int:
        """Number of globally indexed atomic constraint templates."""

        last = self.layouts[-1]
        return self._relation_index_offsets[-1] + len(last.atomic_constraint_ids)

    def relation_index_for(
        self,
        *,
        layout_index: int,
        atomic_constraint_index: int,
    ) -> int:
        layout_position = _u32(
            layout_index,
            "constraint layout_index",
        )
        if layout_position >= len(self.layouts):
            raise ProofV3Error("constraint layout_index is out of range")
        atomic_position = _u32(
            atomic_constraint_index,
            "constraint atomic_constraint_index",
        )
        if atomic_position >= len(self.layouts[layout_position].atomic_constraint_ids):
            raise ProofV3Error("constraint atomic_constraint_index is out of range")
        return self._relation_index_offsets[layout_position] + atomic_position

    def layout_for_relation_index(
        self,
        relation_index: int,
    ) -> tuple[int, GoldilocksConstraintLayoutV3, int]:
        """Resolve a dense global atomic index without prover-supplied labels."""

        index = _u64(relation_index, "constraint relation_index")
        if index >= self.atomic_constraint_count:
            raise ProofV3Error("constraint relation_index is out of range")
        layout_index = bisect.bisect_right(self._relation_index_offsets, index) - 1
        if layout_index < 0:
            raise ProofV3Error("constraint relation_index is out of range")
        layout = self.layouts[layout_index]
        atomic_index = index - self._relation_index_offsets[layout_index]
        if atomic_index >= len(layout.atomic_constraint_ids):
            raise ProofV3Error("constraint relation_index is out of range")
        return layout_index, layout, atomic_index

    def to_dict(self) -> dict[str, object]:
        return {
            "constraint_program_bundle_digest": self.constraint_program_bundle_digest.hex(),
            "constraint_system_abi_id": self.constraint_system_abi_id,
            "dynamic_backend_abi_id": self.dynamic_backend_abi_id,
            "field_id": self.field_id,
            "format_version": self.format_version,
            "layouts": [layout.to_dict() for layout in self.layouts],
            "relation_binding_digest": self.relation_binding_digest.hex(),
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(
            self.to_dict(),
            name="constraint system",
            maximum=MAX_GOLDILOCKS_CONSTRAINT_SYSTEM_BYTES_V3,
        )

    def digest(self) -> bytes:
        """Return the raw SHA-256 digest stored in the signed relation."""

        return self._raw_digest

    @classmethod
    def from_canonical_bytes(
        cls,
        encoded: bytes,
    ) -> "GoldilocksExecutionConstraintSystemV3":
        if (
            type(encoded) is not bytes
            or not encoded
            or len(encoded) > MAX_GOLDILOCKS_CONSTRAINT_SYSTEM_BYTES_V3
        ):
            raise ProofV3Error("constraint-system byte length is out of range")
        try:
            value = json.loads(
                encoded.decode("ascii"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except ProofV3Error:
            raise
        except (RecursionError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProofV3Error("constraint-system is not canonical JSON") from exc
        item = _object(
            value,
            {
                "constraint_system_abi_id",
                "constraint_program_bundle_digest",
                "dynamic_backend_abi_id",
                "field_id",
                "format_version",
                "layouts",
                "relation_binding_digest",
            },
            "constraint system",
        )
        result = cls(
            relation_binding_digest=_json_digest(
                item["relation_binding_digest"],
                "constraint-system relation_binding_digest",
            ),
            constraint_program_bundle_digest=_json_digest(
                item["constraint_program_bundle_digest"],
                "constraint-system constraint_program_bundle_digest",
            ),
            layouts=tuple(
                _layout_from_dict(entry)
                for entry in _list(
                    item["layouts"],
                    "constraint-system layouts",
                    MAX_GOLDILOCKS_CONSTRAINT_LAYOUTS_V3,
                )
            ),
            constraint_system_abi_id=_json_identifier(
                item["constraint_system_abi_id"],
                "constraint-system ABI",
            ),
            field_id=_json_identifier(item["field_id"], "constraint-system field"),
            dynamic_backend_abi_id=_json_identifier(
                item["dynamic_backend_abi_id"],
                "constraint-system dynamic backend ABI",
            ),
            format_version=_json_u32(
                item["format_version"],
                "constraint-system format_version",
                positive=True,
            ),
        )
        if result.canonical_bytes() != encoded:
            raise ProofV3Error("constraint-system is not canonically encoded")
        return result

    def validate_relation(self, *, relation: ExecutionRelationSpecV3) -> None:
        """Require exact topology/table coverage of the signed relation graph."""

        if not isinstance(relation, ExecutionRelationSpecV3):
            raise ProofV3Error("constraint-system relation has an unexpected type")
        if self.relation_binding_digest != constraint_system_relation_binding_digest_v3(
            relation
        ):
            raise ProofV3Error(
                "constraint-system relation projection does not match the signed relation"
            )
        if len(self.layouts) != len(relation.nodes):
            raise ProofV3Error("constraint-system layouts do not exactly cover graph nodes")
        expected_table_ids = {binding.table_id for binding in relation.static_table_bindings}
        expected_tables_by_node = _expected_static_table_ids_by_node_v3(relation)
        expected_phases_by_node = _expected_constraint_layout_phases_by_node_v3(
            relation
        )
        covered_table_ids: set[str] = set()
        for layout_index, (layout, node) in enumerate(zip(self.layouts, relation.nodes)):
            if (
                layout.node_id != node.node_id
                or layout.relation_id != node.relation_id
                or layout.transition_adapter_id != node.transition_adapter_id
                or layout.layer_index != node.layer_index
            ):
                raise ProofV3Error(
                    "constraint-system layout does not match signed graph node order"
                )
            if layout.runtime_tensor_ids != (
                node.input_tensor_ids + node.output_tensor_ids
            ):
                raise ProofV3Error(
                    "constraint-system layout does not match signed runtime tensor order"
                )
            if layout.static_binding_ids != node.static_binding_ids:
                raise ProofV3Error(
                    "constraint-system layout does not match signed static bindings"
                )
            expected_for_node = expected_tables_by_node[node.node_id]
            if layout.static_table_ids != expected_for_node:
                raise ProofV3Error(
                    "constraint-system layout does not match signed static table bindings"
                )
            covered_table_ids.update(layout.static_table_ids)
            required_phases = expected_phases_by_node[node.node_id]
            if layout.phases != required_phases:
                raise ProofV3Error(
                    "constraint-system layout does not match signed input phase coverage"
                )
            for phase in layout.phases:
                chunk_capacity = (
                    relation.prefill_chunk_tokens
                    if phase == "prefill"
                    else relation.decode_chunk_tokens
                )
                try:
                    trace_domain_size = layout.trace_domain_size(
                        token_count=chunk_capacity
                    )
                except ProofV3Error as exc:
                    raise ProofV3Error(
                        "constraint-system layout cannot represent its signed chunk domain"
                    ) from exc
                if layout.max_constraint_degree >= trace_domain_size * layout.lde_blowup:
                    raise ProofV3Error(
                        "constraint-system degree bound exceeds its signed LDE domain"
                    )
            # A final partial chunk can contain one active token. The signed
            # minimum physical domain must still support the declared degree;
            # otherwise a profile could hide an unconstrained short suffix.
            minimum_trace_domain_size = layout.trace_domain_size(token_count=1)
            if (
                layout.max_constraint_degree
                >= minimum_trace_domain_size * layout.lde_blowup
            ):
                raise ProofV3Error(
                    "constraint-system degree bound exceeds its final-chunk LDE domain"
                )
            if self._relation_index_offsets[layout_index] >= self.atomic_constraint_count:
                raise ProofV3Error("constraint-system atomic relation index is malformed")
        if covered_table_ids != expected_table_ids:
            raise ProofV3Error(
                "constraint-system layouts do not exactly cover signed static tables"
            )


@dataclass(frozen=True, slots=True)
class GoldilocksConstraintCoordinateV3:
    """One verifier-derived dynamic constraint template over one execution chunk."""

    node_id: str
    phase: str
    chunk_index: int
    logical_token_start: int
    token_count: int
    relation_index: int

    def __post_init__(self) -> None:
        _identifier(self.node_id, "Goldilocks constraint node_id")
        if self.phase not in _PHASE_CODES:
            raise ProofV3Error("Goldilocks constraint phase is unsupported")
        _u32(self.chunk_index, "Goldilocks constraint chunk_index")
        start = _u32(
            self.logical_token_start,
            "Goldilocks constraint logical_token_start",
        )
        count = _u32(
            self.token_count,
            "Goldilocks constraint token_count",
            positive=True,
        )
        if start + count > 1 << 32:
            raise ProofV3Error("Goldilocks constraint token range overflows")
        _u64(self.relation_index, "Goldilocks constraint relation_index")

    def canonical_bytes(self) -> bytes:
        node_id = _identifier(self.node_id, "Goldilocks constraint node_id").encode(
            "ascii"
        )
        return (
            struct.pack("<B", len(node_id))
            + node_id
            + struct.pack(
                "<BIIIQ",
                _PHASE_CODES[self.phase],
                self.chunk_index,
                self.logical_token_start,
                self.token_count,
                self.relation_index,
            )
        )


@dataclass(frozen=True, slots=True)
class ExpectedGoldilocksConstraintUniverseV3(Sequence[GoldilocksConstraintCoordinateV3]):
    """Indexable, non-materialised complete dynamic constraint universe.

    The object stores only profile/system metadata and small layout prefix
    sums.  ``universe[index]`` is deterministic; iterating it never builds a
    tuple of all rows or coordinates.
    """

    profile: ExecutionSecurityProfileV3
    envelope: ProofV3CommitmentEnvelope
    constraint_system: GoldilocksExecutionConstraintSystemV3
    _prefill_chunk_count: int = field(init=False, repr=False, compare=False)
    _decode_chunk_count: int = field(init=False, repr=False, compare=False)
    _prefill_layout_indices: tuple[int, ...] = field(
        init=False, repr=False, compare=False
    )
    _decode_layout_indices: tuple[int, ...] = field(
        init=False, repr=False, compare=False
    )
    _prefill_offsets: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _decode_offsets: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _prefill_per_chunk: int = field(init=False, repr=False, compare=False)
    _decode_per_chunk: int = field(init=False, repr=False, compare=False)
    _constraint_system_digest: bytes = field(
        init=False,
        repr=False,
        compare=False,
    )
    _binding_digest: bytes = field(init=False, repr=False, compare=False)
    _length: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ExecutionSecurityProfileV3):
            raise ProofV3Error("constraint universe profile has an unexpected type")
        if not isinstance(self.envelope, ProofV3CommitmentEnvelope):
            raise ProofV3Error("constraint universe envelope has an unexpected type")
        if not isinstance(
            self.constraint_system,
            GoldilocksExecutionConstraintSystemV3,
        ):
            raise ProofV3Error("constraint universe system has an unexpected type")
        self.profile.require_hard_execution_capability()
        relation = self.profile.relation_spec
        if self.envelope.execution_profile_digest != self.profile.digest():
            raise ProofV3Error("constraint universe envelope has an unexpected profile")
        if self.envelope.static_manifest_digest != self.profile.static_manifest_digest:
            raise ProofV3Error(
                "constraint universe envelope has an unexpected static manifest"
            )
        constraint_system_digest = self.constraint_system.digest()
        if constraint_system_digest != relation.constraint_system_digest:
            raise ProofV3Error("constraint universe system does not match signed relation")
        self.constraint_system.validate_relation(relation=relation)
        relation.validate_execution_counts(
            context_token_count=self.envelope.context_token_count,
            decode_token_count=self.envelope.decode_token_count,
        )
        prefill_count = (
            self.envelope.context_token_count + relation.prefill_chunk_tokens - 1
        ) // relation.prefill_chunk_tokens
        decode_count = (
            self.envelope.decode_token_count + relation.decode_chunk_tokens - 1
        ) // relation.decode_chunk_tokens
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
        if prefill_count and not prefill_layouts:
            raise ProofV3Error("constraint universe has no prefill constraint templates")
        if decode_count and not decode_layouts:
            raise ProofV3Error("constraint universe has no decode constraint templates")
        prefill_offsets, prefill_per_chunk = self._phase_offsets(prefill_layouts)
        decode_offsets, decode_per_chunk = self._phase_offsets(decode_layouts)
        length = prefill_count * prefill_per_chunk + decode_count * decode_per_chunk
        if length <= 0 or length > MAX_GOLDILOCKS_CONSTRAINT_UNIVERSE_V3:
            raise ProofV3Error("constraint universe exceeds the indexable protocol limit")
        object.__setattr__(self, "_prefill_chunk_count", prefill_count)
        object.__setattr__(self, "_decode_chunk_count", decode_count)
        object.__setattr__(self, "_prefill_layout_indices", prefill_layouts)
        object.__setattr__(self, "_decode_layout_indices", decode_layouts)
        object.__setattr__(self, "_prefill_offsets", prefill_offsets)
        object.__setattr__(self, "_decode_offsets", decode_offsets)
        object.__setattr__(self, "_prefill_per_chunk", prefill_per_chunk)
        object.__setattr__(self, "_decode_per_chunk", decode_per_chunk)
        object.__setattr__(self, "_constraint_system_digest", constraint_system_digest)
        object.__setattr__(self, "_length", length)
        object.__setattr__(
            self,
            "_binding_digest",
            hashlib.sha256(
                _UNIVERSE_BINDING_DOMAIN_V3
                + self.profile.digest()
                + self.envelope.digest()
                + constraint_system_digest
                + struct.pack("<IIQ", prefill_count, decode_count, length)
            ).digest(),
        )

    def _phase_offsets(self, layout_indices: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
        offsets: list[int] = []
        current = 0
        for layout_index in layout_indices:
            offsets.append(current)
            current += len(self.constraint_system.layouts[layout_index].atomic_constraint_ids)
        return tuple(offsets), current

    @property
    def prefill_chunk_count(self) -> int:
        return self._prefill_chunk_count

    @property
    def decode_chunk_count(self) -> int:
        return self._decode_chunk_count

    @property
    def constraint_system_digest(self) -> bytes:
        """Return the cached raw system digest bound during universe creation."""

        return self._constraint_system_digest

    @property
    def binding_digest(self) -> bytes:
        """Return the canonical native/wire binding for this exact universe."""

        return self._binding_digest

    def __len__(self) -> int:
        return self._length

    def _chunk_for(
        self,
        *,
        phase: str,
        local_chunk_index: int,
    ) -> tuple[int, int, int]:
        relation = self.profile.relation_spec
        if phase == "prefill":
            if local_chunk_index >= self._prefill_chunk_count:
                raise ProofV3Error("constraint chunk index is out of range")
            start = local_chunk_index * relation.prefill_chunk_tokens
            count = min(
                relation.prefill_chunk_tokens,
                self.envelope.context_token_count - start,
            )
            return local_chunk_index, start, count
        if local_chunk_index >= self._decode_chunk_count:
            raise ProofV3Error("constraint chunk index is out of range")
        start = self.envelope.context_token_count + (
            local_chunk_index * relation.decode_chunk_tokens
        )
        count = min(
            relation.decode_chunk_tokens,
            self.envelope.decode_token_count
            - local_chunk_index * relation.decode_chunk_tokens,
        )
        return self._prefill_chunk_count + local_chunk_index, start, count

    def _coordinate_for_phase_offset(
        self,
        *,
        phase: str,
        chunk_index: int,
        logical_token_start: int,
        token_count: int,
        phase_offset: int,
    ) -> GoldilocksConstraintCoordinateV3:
        if phase == "prefill":
            layout_indices = self._prefill_layout_indices
            offsets = self._prefill_offsets
            per_chunk = self._prefill_per_chunk
        else:
            layout_indices = self._decode_layout_indices
            offsets = self._decode_offsets
            per_chunk = self._decode_per_chunk
        if phase_offset < 0 or phase_offset >= per_chunk:
            raise ProofV3Error("constraint phase offset is out of range")
        position = bisect.bisect_right(offsets, phase_offset) - 1
        if position < 0:
            raise ProofV3Error("constraint phase offset is out of range")
        layout_index = layout_indices[position]
        layout = self.constraint_system.layouts[layout_index]
        atomic_index = phase_offset - offsets[position]
        if atomic_index >= len(layout.atomic_constraint_ids):
            raise ProofV3Error("constraint phase offset is out of range")
        return GoldilocksConstraintCoordinateV3(
            node_id=layout.node_id,
            phase=phase,
            chunk_index=chunk_index,
            logical_token_start=logical_token_start,
            token_count=token_count,
            relation_index=self.constraint_system.relation_index_for(
                layout_index=layout_index,
                atomic_constraint_index=atomic_index,
            ),
        )

    def __getitem__(self, index: int) -> GoldilocksConstraintCoordinateV3:
        if type(index) is not int:
            raise TypeError("constraint universe index must be an integer")
        if index < 0 or index >= self._length:
            raise IndexError("constraint universe index is out of range")
        prefill_length = self._prefill_chunk_count * self._prefill_per_chunk
        if index < prefill_length:
            local_chunk_index, phase_offset = divmod(index, self._prefill_per_chunk)
            chunk_index, start, count = self._chunk_for(
                phase="prefill",
                local_chunk_index=local_chunk_index,
            )
            return self._coordinate_for_phase_offset(
                phase="prefill",
                chunk_index=chunk_index,
                logical_token_start=start,
                token_count=count,
                phase_offset=phase_offset,
            )
        decode_index = index - prefill_length
        local_chunk_index, phase_offset = divmod(decode_index, self._decode_per_chunk)
        chunk_index, start, count = self._chunk_for(
            phase="decode",
            local_chunk_index=local_chunk_index,
        )
        return self._coordinate_for_phase_offset(
            phase="decode",
            chunk_index=chunk_index,
            logical_token_start=start,
            token_count=count,
            phase_offset=phase_offset,
        )

    def __iter__(self) -> Iterator[GoldilocksConstraintCoordinateV3]:
        for index in range(self._length):
            yield self[index]

    def validate_coordinate(
        self,
        coordinate: GoldilocksConstraintCoordinateV3,
    ) -> GoldilocksConstraintLayoutV3:
        """Reject every coordinate not generated by this exact universe."""

        if not isinstance(coordinate, GoldilocksConstraintCoordinateV3):
            raise ProofV3Error("constraint coordinate has an unexpected type")
        _, layout, _ = self.constraint_system.layout_for_relation_index(
            coordinate.relation_index
        )
        if layout.node_id != coordinate.node_id or coordinate.phase not in layout.phases:
            raise ProofV3Error("constraint coordinate does not match its signed template")
        if coordinate.phase == "prefill":
            local_index = coordinate.chunk_index
            max_chunks = self._prefill_chunk_count
        else:
            local_index = coordinate.chunk_index - self._prefill_chunk_count
            max_chunks = self._decode_chunk_count
        if local_index < 0 or local_index >= max_chunks:
            raise ProofV3Error("constraint coordinate chunk index is out of range")
        expected_chunk_index, start, count = self._chunk_for(
            phase=coordinate.phase,
            local_chunk_index=local_index,
        )
        if (
            coordinate.chunk_index != expected_chunk_index
            or coordinate.logical_token_start != start
            or coordinate.token_count != count
        ):
            raise ProofV3Error(
                "constraint coordinate does not match the verifier-derived chunk"
            )
        return layout


def expected_goldilocks_constraint_universe_v3(
    *,
    profile: ExecutionSecurityProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    constraint_system: GoldilocksExecutionConstraintSystemV3,
) -> ExpectedGoldilocksConstraintUniverseV3:
    """Build the lazy full dynamic constraint universe for one request."""

    return ExpectedGoldilocksConstraintUniverseV3(
        profile=profile,
        envelope=envelope,
        constraint_system=constraint_system,
    )


def goldilocks_constraint_universe_binding_digest_v3(
    universe: ExpectedGoldilocksConstraintUniverseV3,
) -> bytes:
    """Return the serializable binding a future native proof header must carry."""

    if not isinstance(universe, ExpectedGoldilocksConstraintUniverseV3):
        raise ProofV3Error("constraint transcript universe has an unexpected type")
    return universe.binding_digest


@dataclass(frozen=True, slots=True, init=False)
class GoldilocksConstraintTranscriptV3:
    """One factory-bound Goldilocks transcript for one exact constraint universe.

    A future native adapter must not accept a miner-supplied challenge seed.
    This wrapper is minted only by replaying the validator nonce against the
    exact profile/envelope/universe; coefficient derivation then reuses that
    validated result without rebuilding the full universe per atomic term.
    """

    universe_binding_digest: bytes
    challenge: object
    _factory_token: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ProofV3Error(
            "Goldilocks constraint transcripts must be bound from a validator nonce"
        )

    @classmethod
    def _construct(
        cls,
        *,
        universe: ExpectedGoldilocksConstraintUniverseV3,
        challenge: object,
        _factory_token: object | None = None,
    ) -> "GoldilocksConstraintTranscriptV3":
        if _factory_token is not _CONSTRAINT_TRANSCRIPT_FACTORY_TOKEN_V3:
            raise ProofV3Error(
                "Goldilocks constraint transcripts must be bound from a validator nonce"
            )
        # Keep the challenge import local to avoid making the base relation
        # schema depend on the future dynamic proof contract at module import.
        from verallm.proof_v3.challenge import FoldedExecutionChallengeV3

        if not isinstance(universe, ExpectedGoldilocksConstraintUniverseV3):
            raise ProofV3Error("constraint transcript universe has an unexpected type")
        if not isinstance(challenge, FoldedExecutionChallengeV3):
            raise ProofV3Error("constraint transcript challenge has an unexpected type")
        result = object.__new__(cls)
        object.__setattr__(
            result,
            "universe_binding_digest",
            goldilocks_constraint_universe_binding_digest_v3(universe),
        )
        object.__setattr__(result, "challenge", challenge)
        object.__setattr__(
            result,
            "_factory_token",
            _CONSTRAINT_TRANSCRIPT_FACTORY_TOKEN_V3,
        )
        return result

    def require_universe(self, *, universe: ExpectedGoldilocksConstraintUniverseV3) -> None:
        """Reject a stale transcript even when both requests have equal counts."""

        if self._factory_token is not _CONSTRAINT_TRANSCRIPT_FACTORY_TOKEN_V3:
            raise ProofV3Error(
                "constraint transcript lacks validator-bound factory provenance"
            )
        _fixed32(
            self.universe_binding_digest,
            "constraint transcript universe_binding_digest",
            nonzero=True,
        )
        if not isinstance(universe, ExpectedGoldilocksConstraintUniverseV3):
            raise ProofV3Error("constraint coefficient universe has an unexpected type")
        if (
            self.universe_binding_digest
            != goldilocks_constraint_universe_binding_digest_v3(universe)
        ):
            raise ProofV3Error(
                "constraint transcript is bound to a different profile or envelope"
            )


def bind_goldilocks_constraint_transcript_v3(
    *,
    validator_nonce: bytes,
    universe: ExpectedGoldilocksConstraintUniverseV3,
) -> GoldilocksConstraintTranscriptV3:
    """Replay the validator transcript once for one exact dynamic universe."""

    if not isinstance(universe, ExpectedGoldilocksConstraintUniverseV3):
        raise ProofV3Error("constraint transcript universe has an unexpected type")
    from verallm.proof_v3.challenge import derive_folded_execution_challenge_v3

    challenge = derive_folded_execution_challenge_v3(
        validator_nonce=validator_nonce,
        profile=universe.profile,
        envelope=universe.envelope,
    )
    return GoldilocksConstraintTranscriptV3._construct(
        universe=universe,
        challenge=challenge,
        _factory_token=_CONSTRAINT_TRANSCRIPT_FACTORY_TOKEN_V3,
    )


def derive_goldilocks_constraint_coefficient_v3(
    *,
    transcript: GoldilocksConstraintTranscriptV3,
    universe: ExpectedGoldilocksConstraintUniverseV3,
    coordinate: GoldilocksConstraintCoordinateV3,
) -> int:
    """Derive one nonzero Goldilocks coefficient for a signed AIR template.

    A 64-bit rejection word is intentional.  Comparing a full SHA-256 integer
    against a 64-bit field modulus would almost never terminate.  The digest
    still binds the full challenge seed, canonical system digest, and exact
    coordinate before its first eight little-endian bytes are interpreted.
    """

    if not isinstance(transcript, GoldilocksConstraintTranscriptV3):
        raise ProofV3Error("constraint coefficient transcript has an unexpected type")
    if not isinstance(universe, ExpectedGoldilocksConstraintUniverseV3):
        raise ProofV3Error("constraint coefficient universe has an unexpected type")
    transcript.require_universe(universe=universe)
    challenge = transcript.challenge
    # The factory has already replayed the full nonce transcript. Keep this
    # field-level assertion as a defensive guard against trusted-process
    # mutation before a native adapter consumes the wrapper.
    if challenge.hard_audit_selection.sequence_token_count != (
        universe.envelope.context_token_count + universe.envelope.decode_token_count
    ):
        raise ProofV3Error(
            "constraint coefficient challenge has an unexpected sequence domain"
        )
    universe.validate_coordinate(coordinate)
    encoded_coordinate = coordinate.canonical_bytes()
    for counter in range(_MAX_REJECTION_ATTEMPTS):
        digest = hashlib.sha256(
            _COEFFICIENT_DOMAIN_V3
            + challenge.global_relation_seed
            + transcript.universe_binding_digest
            + universe.constraint_system_digest
            + encoded_coordinate
            + struct.pack("<I", counter)
        ).digest()
        candidate = int.from_bytes(digest[:8], "little")
        if 0 < candidate < GOLDILOCKS_MODULUS:
            return candidate
    raise ProofV3Error("unable to derive a canonical Goldilocks constraint coefficient")


__all__ = [
    "GOLDILOCKS_EXECUTION_CONSTRAINT_SYSTEM_ABI_V3",
    "GOLDILOCKS_EXECUTION_CONSTRAINT_SYSTEM_FORMAT_VERSION_V3",
    "GOLDILOCKS_TRACE_DOMAIN_RULE_V3",
    "GOLDILOCKS_TRACE_PADDING_RULE_V3",
    "MAX_GOLDILOCKS_CONSTRAINT_SYSTEM_BYTES_V3",
    "MIN_GOLDILOCKS_LDE_BLOWUP_V3",
    "GoldilocksConstraintLayoutV3",
    "GoldilocksConstraintCoordinateV3",
    "GoldilocksConstraintTranscriptV3",
    "GoldilocksExecutionConstraintSystemV3",
    "ExpectedGoldilocksConstraintUniverseV3",
    "constraint_system_relation_projection_bytes_v3",
    "constraint_system_relation_binding_digest_v3",
    "bind_goldilocks_constraint_transcript_v3",
    "goldilocks_constraint_universe_binding_digest_v3",
    "expected_static_table_ids_for_relation_node_v3",
    "expected_constraint_layout_phases_for_relation_node_v3",
    "expected_goldilocks_constraint_universe_v3",
    "derive_goldilocks_constraint_coefficient_v3",
]
