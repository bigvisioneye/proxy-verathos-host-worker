"""Canonical field-polynomial AIR program artifacts for proof-v3.

The earlier constraint-system artifact fixed only a layout plus an opaque
``constraint_program_digest``.  This module gives that digest a strict,
content-addressed meaning: one parsed field-polynomial program, bound to one
exact signed graph layout.  The bundle is loaded locally by a validator and is
transitively authenticated through the constraint-system digest stored in the
signed execution relation.

This is deliberately a narrow first compiler contract.  It accepts only a
finite Goldilocks polynomial IR (constants, current/next trace cells, add,
multiply, and negate), explicit active/boundary scopes, a fixed inactive-row
padding contract, and an optional exact affine source-coordinate map for
runtime/fixed columns.  It rejects opaque bytecode, callbacks, arbitrary CUDA
or Python opcodes, and unspecified degree bounds.  The source map is only an
authenticated coordinate ABI: range decomposition, static-table lookup,
logical RAM, and the field relation tying those coordinates to runtime/static
commitments remain unimplemented.  No program in this module is a native
adapter or a hard-proof qualification.

The module is still useful before those larger relations land: it prevents a
future native backend from treating a signed hash label as permission to run
unreviewed arithmetic, and it provides one canonical input language for a
reference/native AIR compiler.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.constraint_system import (
    GOLDILOCKS_TRACE_PADDING_RULE_V3,
    MAX_GOLDILOCKS_ATOMIC_CONSTRAINTS_PER_LAYOUT_V3,
    GoldilocksConstraintLayoutV3,
    GoldilocksExecutionConstraintSystemV3,
    constraint_system_relation_binding_digest_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_reference import (
    GOLDILOCKS_MODULUS,
    MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE,
    canonical_goldilocks,
)
from verallm.proof_v3.relation import ExecutionRelationSpecV3


GOLDILOCKS_CONSTRAINT_PROGRAM_ABI_V3: Final = "goldilocks.air_program.v4"
GOLDILOCKS_CONSTRAINT_PROGRAM_BUNDLE_ABI_V3: Final = (
    "goldilocks.air_program_bundle.v4"
)
GOLDILOCKS_CONSTRAINT_PROGRAM_FORMAT_VERSION_V3: Final = 4
GOLDILOCKS_CONSTRAINT_PROGRAM_BUNDLE_FORMAT_VERSION_V3: Final = 4
GOLDILOCKS_FIELD_POLYNOMIAL_IR_V3: Final = "goldilocks.field_poly.v1"
GOLDILOCKS_ACTIVE_ROW_RULE_V3: Final = "token_count_times_rows_per_token.v1"
GOLDILOCKS_STRUCTURAL_PADDING_CONSTRAINT_PREFIX_V3: Final = "sys.padding."
GOLDILOCKS_TRACE_SOURCE_BINDING_MODE_UNBOUND_REFERENCE_V3: Final = (
    "unbound_reference.v1"
)
GOLDILOCKS_TRACE_SOURCE_BINDING_MODE_EXACT_LAYOUT_V3: Final = (
    "exact_layout_sources.v1"
)
GOLDILOCKS_RUNTIME_TRACE_FIELD_ENCODING_V3: Final = "goldilocks.canonical_u64.v1"
GOLDILOCKS_RUNTIME_TOKEN_AXIS_CONTEXT_V3: Final = "context_tokens.v1"
GOLDILOCKS_RUNTIME_TOKEN_AXIS_DECODE_V3: Final = "decode_tokens.v1"
GOLDILOCKS_RUNTIME_TOKEN_AXIS_SEQUENCE_V3: Final = "sequence_tokens.v1"

MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_BYTES_V3: Final = 1 << 20
MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_BUNDLE_BYTES_V3: Final = 64 << 20
MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_COLUMNS_V3: Final = 4_096
MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_EXPRESSION_NODES_V3: Final = 4_096
MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_EXPRESSION_DEPTH_V3: Final = 32
MAX_GOLDILOCKS_CONSTRAINT_PROGRAMS_V3: Final = 65_535

_IDENTIFIER_RE: Final = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,127}$")
_HEX_32_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_PHASE_ORDER: Final = ("prefill", "decode")
_COLUMN_ROLES: Final = frozenset(
    {"auxiliary", "fixed", "public", "runtime", "selector"}
)
_ROW_ACCESS: Final = frozenset({"current", "next", "both"})
_CONSTRAINT_SCOPES: Final = frozenset(
    {
        "active_rows",
        "first_active_row",
        "last_active_row",
        "padding_rows",
        "transition_rows",
    }
)
_EXPRESSION_OPS: Final = frozenset({"add", "cell", "const", "mul", "neg"})
_TRACE_SOURCE_BINDING_MODES: Final = frozenset(
    {
        GOLDILOCKS_TRACE_SOURCE_BINDING_MODE_UNBOUND_REFERENCE_V3,
        GOLDILOCKS_TRACE_SOURCE_BINDING_MODE_EXACT_LAYOUT_V3,
    }
)
_RUNTIME_TOKEN_AXIS_RULE_IDS: Final = frozenset(
    {
        GOLDILOCKS_RUNTIME_TOKEN_AXIS_CONTEXT_V3,
        GOLDILOCKS_RUNTIME_TOKEN_AXIS_DECODE_V3,
        GOLDILOCKS_RUNTIME_TOKEN_AXIS_SEQUENCE_V3,
    }
)
_PROGRAM_DIGEST_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR_PROGRAM/V1/SHA256"
)
_BUNDLE_DIGEST_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR_PROGRAM_BUNDLE/V1/SHA256"
)


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


def _i32(value: object, name: str) -> int:
    if type(value) is not int or not -(1 << 31) <= value < 1 << 31:
        raise ProofV3Error(f"{name} must be a signed 32-bit integer")
    return value


def _power_of_two(value: object, name: str) -> int:
    integer = _u64(value, name, positive=True)
    if integer & (integer - 1):
        raise ProofV3Error(f"{name} must be a power of two")
    return integer


def _identifier_tuple(
    value: object,
    *,
    name: str,
    maximum: int,
    allow_empty: bool = False,
    sorted_distinct: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, tuple) or len(value) > maximum or (
        not allow_empty and not value
    ):
        raise ProofV3Error(f"{name} has an invalid length")
    result = tuple(_identifier(item, f"{name}[{index}]") for index, item in enumerate(value))
    if sorted_distinct and (
        result != tuple(sorted(result)) or len(result) != len(set(result))
    ):
        raise ProofV3Error(f"{name} must be sorted and distinct")
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


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProofV3Error("constraint program contains duplicate JSON object keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ProofV3Error(f"constraint program contains unsupported JSON constant {value}")


def _object(value: object, keys: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ProofV3Error(f"{name} fields do not match the canonical schema")
    return value


def _list(
    value: object,
    name: str,
    maximum: int,
    *,
    allow_empty: bool = False,
) -> list[object]:
    if not isinstance(value, list) or len(value) > maximum or (
        not allow_empty and not value
    ):
        raise ProofV3Error(f"{name} must be a bounded list")
    return value


def _json_identifier(value: object, name: str) -> str:
    return _identifier(value, name)


def _json_digest(value: object, name: str) -> bytes:
    if not isinstance(value, str) or _HEX_32_RE.fullmatch(value) is None:
        raise ProofV3Error(f"{name} must be a lowercase 32-byte hexadecimal digest")
    return bytes.fromhex(value)


def _json_identifier_tuple(
    value: object,
    *,
    name: str,
    maximum: int,
    allow_empty: bool = False,
    sorted_distinct: bool = False,
) -> tuple[str, ...]:
    entries = _list(value, name, maximum, allow_empty=allow_empty)
    return _identifier_tuple(
        tuple(_json_identifier(item, f"{name}[{index}]") for index, item in enumerate(entries)),
        name=name,
        maximum=maximum,
        allow_empty=allow_empty,
        sorted_distinct=sorted_distinct,
    )


@dataclass(frozen=True, slots=True)
class GoldilocksConstraintProgramLayoutBindingV3:
    """Full program-side mirror of one constraint-system layout.

    Duplicating the layout avoids a retargeted program whose digest happens to
    be valid but whose compiler inputs point at a different graph node.  The
    program digest itself is intentionally excluded to avoid a self-reference.
    """

    layout_index: int
    node_id: str
    relation_id: str
    transition_adapter_id: str
    layer_index: int
    phases: tuple[str, ...]
    runtime_tensor_ids: tuple[str, ...]
    static_binding_ids: tuple[str, ...]
    static_table_ids: tuple[str, ...]
    atomic_constraint_ids: tuple[str, ...]
    rows_per_token: int
    minimum_trace_rows: int
    lde_blowup: int
    max_constraint_degree: int
    trace_domain_rule_id: str
    padding_rule_id: str

    def __post_init__(self) -> None:
        _u32(self.layout_index, "program layout_index")
        for value, name in (
            (self.node_id, "program node_id"),
            (self.relation_id, "program relation_id"),
            (self.transition_adapter_id, "program transition_adapter_id"),
            (self.trace_domain_rule_id, "program trace_domain_rule_id"),
            (self.padding_rule_id, "program padding_rule_id"),
        ):
            _identifier(value, name)
        _i32(self.layer_index, "program layer_index")
        phases = _identifier_tuple(
            self.phases,
            name="program phases",
            maximum=len(_PHASE_ORDER),
        )
        if phases != tuple(phase for phase in _PHASE_ORDER if phase in phases):
            raise ProofV3Error("program phases are not canonically ordered")
        if any(phase not in _PHASE_ORDER for phase in phases):
            raise ProofV3Error("program phase is unsupported")
        runtime_tensors = _identifier_tuple(
            self.runtime_tensor_ids,
            name="program runtime_tensor_ids",
            maximum=64,
        )
        if len(runtime_tensors) != len(set(runtime_tensors)):
            raise ProofV3Error("program runtime_tensor_ids contain duplicates")
        static_bindings = _identifier_tuple(
            self.static_binding_ids,
            name="program static_binding_ids",
            maximum=32,
            allow_empty=True,
        )
        if len(static_bindings) != len(set(static_bindings)):
            raise ProofV3Error("program static_binding_ids contain duplicates")
        static_tables = _identifier_tuple(
            self.static_table_ids,
            name="program static_table_ids",
            maximum=32,
            allow_empty=True,
            sorted_distinct=True,
        )
        atomics = _identifier_tuple(
            self.atomic_constraint_ids,
            name="program atomic_constraint_ids",
            maximum=MAX_GOLDILOCKS_ATOMIC_CONSTRAINTS_PER_LAYOUT_V3,
            sorted_distinct=True,
        )
        _u32(self.rows_per_token, "program rows_per_token", positive=True)
        _power_of_two(self.minimum_trace_rows, "program minimum_trace_rows")
        _power_of_two(self.lde_blowup, "program lde_blowup")
        _u32(
            self.max_constraint_degree,
            "program max_constraint_degree",
            positive=True,
        )
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "runtime_tensor_ids", runtime_tensors)
        object.__setattr__(self, "static_binding_ids", static_bindings)
        object.__setattr__(self, "static_table_ids", static_tables)
        object.__setattr__(self, "atomic_constraint_ids", atomics)

    @classmethod
    def from_layout(
        cls,
        *,
        layout_index: int,
        layout: GoldilocksConstraintLayoutV3,
    ) -> "GoldilocksConstraintProgramLayoutBindingV3":
        if not isinstance(layout, GoldilocksConstraintLayoutV3):
            raise ProofV3Error("constraint program layout has an unexpected type")
        return cls(
            layout_index=layout_index,
            node_id=layout.node_id,
            relation_id=layout.relation_id,
            transition_adapter_id=layout.transition_adapter_id,
            layer_index=layout.layer_index,
            phases=layout.phases,
            runtime_tensor_ids=layout.runtime_tensor_ids,
            static_binding_ids=layout.static_binding_ids,
            static_table_ids=layout.static_table_ids,
            atomic_constraint_ids=layout.atomic_constraint_ids,
            rows_per_token=layout.rows_per_token,
            minimum_trace_rows=layout.minimum_trace_rows,
            lde_blowup=layout.lde_blowup,
            max_constraint_degree=layout.max_constraint_degree,
            trace_domain_rule_id=layout.trace_domain_rule_id,
            padding_rule_id=layout.padding_rule_id,
        )

    def matches_layout(
        self,
        *,
        layout_index: int,
        layout: GoldilocksConstraintLayoutV3,
    ) -> bool:
        return self == self.from_layout(layout_index=layout_index, layout=layout)

    def to_dict(self) -> dict[str, object]:
        return {
            "atomic_constraint_ids": list(self.atomic_constraint_ids),
            "layer_index": self.layer_index,
            "layout_index": self.layout_index,
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


def _layout_binding_from_dict(
    value: object,
) -> GoldilocksConstraintProgramLayoutBindingV3:
    item = _object(
        value,
        {
            "atomic_constraint_ids",
            "layer_index",
            "layout_index",
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
        "constraint program layout binding",
    )
    layer_index = item["layer_index"]
    if type(layer_index) is not int:
        raise ProofV3Error("program layout layer_index is malformed")
    return GoldilocksConstraintProgramLayoutBindingV3(
        layout_index=_u32(item["layout_index"], "program layout_index"),
        node_id=_json_identifier(item["node_id"], "program node_id"),
        relation_id=_json_identifier(item["relation_id"], "program relation_id"),
        transition_adapter_id=_json_identifier(
            item["transition_adapter_id"], "program transition_adapter_id"
        ),
        layer_index=_i32(layer_index, "program layer_index"),
        phases=_json_identifier_tuple(
            item["phases"],
            name="program phases",
            maximum=len(_PHASE_ORDER),
        ),
        runtime_tensor_ids=_json_identifier_tuple(
            item["runtime_tensor_ids"],
            name="program runtime_tensor_ids",
            maximum=64,
        ),
        static_binding_ids=_json_identifier_tuple(
            item["static_binding_ids"],
            name="program static_binding_ids",
            maximum=32,
            allow_empty=True,
        ),
        static_table_ids=_json_identifier_tuple(
            item["static_table_ids"],
            name="program static_table_ids",
            maximum=32,
            allow_empty=True,
            sorted_distinct=True,
        ),
        atomic_constraint_ids=_json_identifier_tuple(
            item["atomic_constraint_ids"],
            name="program atomic_constraint_ids",
            maximum=MAX_GOLDILOCKS_ATOMIC_CONSTRAINTS_PER_LAYOUT_V3,
            sorted_distinct=True,
        ),
        rows_per_token=_u32(
            item["rows_per_token"], "program rows_per_token", positive=True
        ),
        minimum_trace_rows=_power_of_two(
            item["minimum_trace_rows"], "program minimum_trace_rows"
        ),
        lde_blowup=_power_of_two(item["lde_blowup"], "program lde_blowup"),
        max_constraint_degree=_u32(
            item["max_constraint_degree"],
            "program max_constraint_degree",
            positive=True,
        ),
        trace_domain_rule_id=_json_identifier(
            item["trace_domain_rule_id"], "program trace_domain_rule_id"
        ),
        padding_rule_id=_json_identifier(
            item["padding_rule_id"], "program padding_rule_id"
        ),
    )


@dataclass(frozen=True, slots=True)
class GoldilocksTraceColumnV3:
    """One named scalar trace column in the finite field-polynomial IR."""

    column_id: str
    column_role: str
    source_id: str | None = None
    row_access: str = "current"

    def __post_init__(self) -> None:
        _identifier(self.column_id, "program trace column_id")
        if self.column_role not in _COLUMN_ROLES:
            raise ProofV3Error("program trace column role is unsupported")
        if self.row_access not in _ROW_ACCESS:
            raise ProofV3Error("program trace column row_access is unsupported")
        if self.source_id is not None:
            _identifier(self.source_id, "program trace column source_id")
        if self.column_role == "auxiliary" and self.source_id is not None:
            raise ProofV3Error("auxiliary program trace columns must not name a source")
        if self.column_role != "auxiliary" and self.source_id is None:
            raise ProofV3Error("non-auxiliary program trace columns require a source")
        if self.column_role in {"fixed", "public", "selector"} and self.row_access != "current":
            raise ProofV3Error(
                "fixed, public, and selector trace columns may only use current rows"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "column_id": self.column_id,
            "column_role": self.column_role,
            "row_access": self.row_access,
            "source_id": self.source_id,
        }


def _trace_column_from_dict(value: object) -> GoldilocksTraceColumnV3:
    item = _object(
        value,
        {"column_id", "column_role", "row_access", "source_id"},
        "constraint program trace column",
    )
    source_id = item["source_id"]
    if source_id is not None:
        source_id = _json_identifier(source_id, "program trace column source_id")
    return GoldilocksTraceColumnV3(
        column_id=_json_identifier(item["column_id"], "program trace column_id"),
        column_role=_json_identifier(
            item["column_role"], "program trace column_role"
        ),
        source_id=source_id,
        row_access=_json_identifier(item["row_access"], "program trace row_access"),
    )


@dataclass(frozen=True, slots=True)
class GoldilocksRuntimeTraceColumnBindingV3:
    """Canonical affine source coordinate for one runtime trace column.

    A runtime column does not become an execution witness merely because its
    ``source_id`` names a tensor.  This binding fixes the exact logical source
    coordinate for every trace row so a later field-consistent tensor relation
    can prove the cell equality.  The parsed-program artifact authenticates
    the mapping; the mapping itself is deliberately not a Merkle or lookup
    proof.
    """

    column_id: str
    tensor_id: str
    source_encoding_id: str
    source_layout_id: str
    token_axis_rule_id: str
    elements_per_token: int
    element_offset: int
    trace_row_stride: int
    field_encoding_id: str = GOLDILOCKS_RUNTIME_TRACE_FIELD_ENCODING_V3

    def __post_init__(self) -> None:
        _identifier(self.column_id, "runtime trace binding column_id")
        _identifier(self.tensor_id, "runtime trace binding tensor_id")
        _identifier(
            self.source_encoding_id,
            "runtime trace binding source_encoding_id",
        )
        _identifier(self.source_layout_id, "runtime trace binding source_layout_id")
        if self.token_axis_rule_id not in _RUNTIME_TOKEN_AXIS_RULE_IDS:
            raise ProofV3Error("runtime trace binding token axis is unsupported")
        _u32(
            self.elements_per_token,
            "runtime trace binding elements_per_token",
            positive=True,
        )
        _u64(
            self.element_offset,
            "runtime trace binding element_offset",
        )
        _u32(
            self.trace_row_stride,
            "runtime trace binding trace_row_stride",
            positive=True,
        )
        if self.field_encoding_id != GOLDILOCKS_RUNTIME_TRACE_FIELD_ENCODING_V3:
            raise ProofV3Error("runtime trace binding field encoding is unsupported")

    def to_dict(self) -> dict[str, object]:
        return {
            "column_id": self.column_id,
            "element_offset": self.element_offset,
            "elements_per_token": self.elements_per_token,
            "field_encoding_id": self.field_encoding_id,
            "source_encoding_id": self.source_encoding_id,
            "source_layout_id": self.source_layout_id,
            "tensor_id": self.tensor_id,
            "token_axis_rule_id": self.token_axis_rule_id,
            "trace_row_stride": self.trace_row_stride,
        }


def _runtime_trace_column_binding_from_dict(
    value: object,
) -> GoldilocksRuntimeTraceColumnBindingV3:
    item = _object(
        value,
        {
            "column_id",
            "element_offset",
            "elements_per_token",
            "field_encoding_id",
            "source_encoding_id",
            "source_layout_id",
            "tensor_id",
            "token_axis_rule_id",
            "trace_row_stride",
        },
        "runtime trace column binding",
    )
    return GoldilocksRuntimeTraceColumnBindingV3(
        column_id=_json_identifier(
            item["column_id"], "runtime trace binding column_id"
        ),
        tensor_id=_json_identifier(
            item["tensor_id"], "runtime trace binding tensor_id"
        ),
        source_encoding_id=_json_identifier(
            item["source_encoding_id"],
            "runtime trace binding source_encoding_id",
        ),
        source_layout_id=_json_identifier(
            item["source_layout_id"],
            "runtime trace binding source_layout_id",
        ),
        token_axis_rule_id=_json_identifier(
            item["token_axis_rule_id"],
            "runtime trace binding token_axis_rule_id",
        ),
        elements_per_token=_u32(
            item["elements_per_token"],
            "runtime trace binding elements_per_token",
            positive=True,
        ),
        element_offset=_u64(
            item["element_offset"], "runtime trace binding element_offset"
        ),
        trace_row_stride=_u32(
            item["trace_row_stride"],
            "runtime trace binding trace_row_stride",
            positive=True,
        ),
        field_encoding_id=_json_identifier(
            item["field_encoding_id"], "runtime trace binding field_encoding_id"
        ),
    )


@dataclass(frozen=True, slots=True)
class GoldilocksStaticTraceColumnBindingV3:
    """Canonical affine source coordinate for one fixed trace column.

    The table coordinate is authenticated by the parsed program, but this
    object is not a static-table opening or a cross-field equality proof.  A
    zero stride denotes an explicitly repeated fixed source; static byte-table
    construction, lookup/broadcast semantics, and decoding remain a separate
    required relation before a native proof can use this mapping.
    """

    column_id: str
    table_id: str
    logical_leaf_offset: int
    trace_row_stride: int
    cell_encoding_id: str

    def __post_init__(self) -> None:
        _identifier(self.column_id, "static trace binding column_id")
        _identifier(self.table_id, "static trace binding table_id")
        _u64(
            self.logical_leaf_offset,
            "static trace binding logical_leaf_offset",
        )
        _u32(self.trace_row_stride, "static trace binding trace_row_stride")
        _identifier(self.cell_encoding_id, "static trace binding cell_encoding_id")

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_encoding_id": self.cell_encoding_id,
            "column_id": self.column_id,
            "logical_leaf_offset": self.logical_leaf_offset,
            "table_id": self.table_id,
            "trace_row_stride": self.trace_row_stride,
        }


def _static_trace_column_binding_from_dict(
    value: object,
) -> GoldilocksStaticTraceColumnBindingV3:
    item = _object(
        value,
        {
            "cell_encoding_id",
            "column_id",
            "logical_leaf_offset",
            "table_id",
            "trace_row_stride",
        },
        "static trace column binding",
    )
    return GoldilocksStaticTraceColumnBindingV3(
        column_id=_json_identifier(
            item["column_id"], "static trace binding column_id"
        ),
        table_id=_json_identifier(item["table_id"], "static trace binding table_id"),
        logical_leaf_offset=_u64(
            item["logical_leaf_offset"], "static trace binding logical_leaf_offset"
        ),
        trace_row_stride=_u32(
            item["trace_row_stride"],
            "static trace binding trace_row_stride",
        ),
        cell_encoding_id=_json_identifier(
            item["cell_encoding_id"], "static trace binding cell_encoding_id"
        ),
    )


def _runtime_tensor_binding_metadata_v3(
    *,
    relation: ExecutionRelationSpecV3,
    tensor_id: str,
) -> tuple[str, str, str, int]:
    """Resolve the signed source ABI needed by an exact runtime binding.

    The first field-copy conformance relation only supports tensors with one
    explicit token axis and a statically resolved per-token footprint.  A
    profile with a dynamic/ambiguous shape must add a qualified lowerer rather
    than rely on an implicit flattening convention here.
    """

    if not isinstance(relation, ExecutionRelationSpecV3):
        raise ProofV3Error("runtime trace binding relation is malformed")
    tensors = {tensor.tensor_id: tensor for tensor in relation.tensors}
    dimensions = {dimension.dimension_id: dimension for dimension in relation.dimensions}
    tensor = tensors.get(tensor_id)
    if tensor is None:
        raise ProofV3Error("runtime trace binding references an unknown tensor")
    axes = (
        (
            relation.sequence_domain.context_dimension_id,
            GOLDILOCKS_RUNTIME_TOKEN_AXIS_CONTEXT_V3,
        ),
        (
            relation.sequence_domain.decode_dimension_id,
            GOLDILOCKS_RUNTIME_TOKEN_AXIS_DECODE_V3,
        ),
        (
            relation.sequence_domain.sequence_dimension_id,
            GOLDILOCKS_RUNTIME_TOKEN_AXIS_SEQUENCE_V3,
        ),
    )
    matching_axes = tuple(
        (dimension_id, rule_id)
        for dimension_id, rule_id in axes
        if dimension_id in tensor.shape
    )
    if len(matching_axes) != 1:
        raise ProofV3Error(
            "runtime trace binding tensor must have exactly one signed token axis"
        )
    token_dimension_id, token_axis_rule_id = matching_axes[0]
    elements_per_token = 1
    for dimension in tensor.shape:
        if dimension == token_dimension_id:
            continue
        if type(dimension) is int:
            size = dimension
        else:
            declared = dimensions.get(dimension)
            size = None if declared is None else declared.exact_value
        if size is None or size <= 0:
            raise ProofV3Error(
                "runtime trace binding tensor has no resolved per-token shape"
            )
        elements_per_token *= size
        if elements_per_token >= 1 << 32:
            raise ProofV3Error(
                "runtime trace binding tensor elements_per_token overflows"
            )
    return (
        tensor.encoding_id,
        tensor.layout_id,
        token_axis_rule_id,
        elements_per_token,
    )


@dataclass(frozen=True, slots=True)
class GoldilocksPolynomialExpressionV3:
    """One recursively bounded expression in the finite field-polynomial IR."""

    op: str
    value: int | None = None
    column_id: str | None = None
    row_offset: int | None = None
    arguments: tuple["GoldilocksPolynomialExpressionV3", ...] = ()

    def __post_init__(self) -> None:
        _validate_expression_shape(self, depth=0, nodes=[0])

    @property
    def degree(self) -> int:
        if self.op == "const":
            return 0
        if self.op == "cell":
            return 1
        if self.op == "neg":
            return self.arguments[0].degree
        if self.op == "add":
            return max(argument.degree for argument in self.arguments)
        assert self.op == "mul"
        return sum(argument.degree for argument in self.arguments)

    @property
    def uses_next_row(self) -> bool:
        if self.op == "cell":
            return self.row_offset == 1
        return any(argument.uses_next_row for argument in self.arguments)

    def to_dict(self) -> dict[str, object]:
        if self.op == "const":
            return {"op": self.op, "value": self.value}
        if self.op == "cell":
            return {
                "column_id": self.column_id,
                "op": self.op,
                "row_offset": self.row_offset,
            }
        return {
            "arguments": [argument.to_dict() for argument in self.arguments],
            "op": self.op,
        }

    def _validate_references(
        self,
        *,
        columns: Mapping[str, GoldilocksTraceColumnV3],
    ) -> None:
        if self.op == "cell":
            assert self.column_id is not None
            assert self.row_offset is not None
            column = columns.get(self.column_id)
            if column is None:
                raise ProofV3Error("program expression references an unknown trace column")
            if self.row_offset == 1 and column.row_access not in {"next", "both"}:
                raise ProofV3Error("program expression uses an unauthorized next-row cell")
        for argument in self.arguments:
            argument._validate_references(columns=columns)

    def _evaluate(
        self,
        *,
        current_row: tuple[int, ...],
        next_row: tuple[int, ...] | None,
        column_positions: Mapping[str, int],
    ) -> int:
        if self.op == "const":
            assert self.value is not None
            return self.value
        if self.op == "cell":
            assert self.column_id is not None
            assert self.row_offset is not None
            position = column_positions[self.column_id]
            if self.row_offset == 0:
                return current_row[position]
            if next_row is None:
                raise ProofV3Error("program expression requires an unavailable next row")
            return next_row[position]
        if self.op == "neg":
            return (-self.arguments[0]._evaluate(
                current_row=current_row,
                next_row=next_row,
                column_positions=column_positions,
            )) % GOLDILOCKS_MODULUS
        values = tuple(
            argument._evaluate(
                current_row=current_row,
                next_row=next_row,
                column_positions=column_positions,
            )
            for argument in self.arguments
        )
        if self.op == "add":
            return (values[0] + values[1]) % GOLDILOCKS_MODULUS
        assert self.op == "mul"
        return values[0] * values[1] % GOLDILOCKS_MODULUS


def _validate_expression_shape(
    expression: GoldilocksPolynomialExpressionV3,
    *,
    depth: int,
    nodes: list[int],
) -> None:
    if not isinstance(expression, GoldilocksPolynomialExpressionV3):
        raise ProofV3Error("program expression has an unexpected type")
    if depth > MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_EXPRESSION_DEPTH_V3:
        raise ProofV3Error("program expression exceeds the nesting limit")
    nodes[0] += 1
    if nodes[0] > MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_EXPRESSION_NODES_V3:
        raise ProofV3Error("program expression exceeds the node limit")
    if expression.op not in _EXPRESSION_OPS:
        raise ProofV3Error("program expression opcode is unsupported")
    arguments = expression.arguments
    if not isinstance(arguments, tuple):
        raise ProofV3Error("program expression arguments must be a tuple")
    if expression.op == "const":
        if (
            type(expression.value) is not int
            or not 0 <= expression.value < GOLDILOCKS_MODULUS
            or expression.column_id is not None
            or expression.row_offset is not None
            or arguments
        ):
            raise ProofV3Error("program constant expression is malformed")
        return
    if expression.op == "cell":
        if (
            expression.value is not None
            or expression.column_id is None
            or type(expression.row_offset) is not int
            or expression.row_offset not in {0, 1}
            or arguments
        ):
            raise ProofV3Error("program cell expression is malformed")
        _identifier(expression.column_id, "program expression column_id")
        return
    expected_count = 1 if expression.op == "neg" else 2
    if (
        expression.value is not None
        or expression.column_id is not None
        or expression.row_offset is not None
        or len(arguments) != expected_count
    ):
        raise ProofV3Error("program polynomial expression is malformed")
    for argument in arguments:
        _validate_expression_shape(argument, depth=depth + 1, nodes=nodes)


def _expression_from_dict(
    value: object,
    *,
    depth: int,
    nodes: list[int],
) -> GoldilocksPolynomialExpressionV3:
    if depth > MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_EXPRESSION_DEPTH_V3:
        raise ProofV3Error("program expression exceeds the nesting limit")
    if not isinstance(value, Mapping):
        raise ProofV3Error("program expression must be an object")
    op = value.get("op")
    if not isinstance(op, str) or op not in _EXPRESSION_OPS:
        raise ProofV3Error("program expression opcode is unsupported")
    if op == "const":
        item = _object(value, {"op", "value"}, "program constant expression")
        candidate = item["value"]
        if type(candidate) is not int or not 0 <= candidate < GOLDILOCKS_MODULUS:
            raise ProofV3Error("program constant is outside Goldilocks")
        expression = GoldilocksPolynomialExpressionV3(op="const", value=candidate)
    elif op == "cell":
        item = _object(value, {"column_id", "op", "row_offset"}, "program cell expression")
        expression = GoldilocksPolynomialExpressionV3(
            op="cell",
            column_id=_json_identifier(item["column_id"], "program expression column_id"),
            row_offset=_u32(item["row_offset"], "program expression row_offset"),
        )
    else:
        item = _object(value, {"arguments", "op"}, "program polynomial expression")
        arguments = tuple(
            _expression_from_dict(entry, depth=depth + 1, nodes=nodes)
            for entry in _list(
                item["arguments"],
                "program expression arguments",
                2,
            )
        )
        expression = GoldilocksPolynomialExpressionV3(op=op, arguments=arguments)
    nodes[0] += 1
    if nodes[0] > MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_EXPRESSION_NODES_V3:
        raise ProofV3Error("program expression exceeds the node limit")
    return expression


@dataclass(frozen=True, slots=True)
class GoldilocksAtomicConstraintV3:
    """One exact zero-polynomial requirement over a signed trace scope."""

    constraint_id: str
    expression: GoldilocksPolynomialExpressionV3
    scope: str = "active_rows"

    def __post_init__(self) -> None:
        _identifier(self.constraint_id, "program atomic constraint_id")
        if not isinstance(self.expression, GoldilocksPolynomialExpressionV3):
            raise ProofV3Error("program atomic expression has an unexpected type")
        if self.scope not in _CONSTRAINT_SCOPES:
            raise ProofV3Error("program atomic constraint scope is unsupported")
        if self.expression.uses_next_row and self.scope != "transition_rows":
            raise ProofV3Error(
                "program next-row expressions require transition_rows scope"
            )

    @property
    def degree(self) -> int:
        return self.expression.degree

    def to_dict(self) -> dict[str, object]:
        return {
            "constraint_id": self.constraint_id,
            "expression": self.expression.to_dict(),
            "scope": self.scope,
        }


def _atomic_constraint_from_dict(value: object) -> GoldilocksAtomicConstraintV3:
    item = _object(
        value,
        {"constraint_id", "expression", "scope"},
        "program atomic constraint",
    )
    return GoldilocksAtomicConstraintV3(
        constraint_id=_json_identifier(
            item["constraint_id"], "program atomic constraint_id"
        ),
        expression=_expression_from_dict(item["expression"], depth=0, nodes=[0]),
        scope=_json_identifier(item["scope"], "program atomic constraint scope"),
    )


@dataclass(frozen=True, slots=True)
class GoldilocksPaddingContractV3:
    """The required inactive-row behavior for one field-polynomial trace."""

    active_row_rule_id: str
    active_selector_column_id: str
    zero_column_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.active_row_rule_id != GOLDILOCKS_ACTIVE_ROW_RULE_V3:
            raise ProofV3Error("program active-row rule is unsupported")
        _identifier(
            self.active_selector_column_id,
            "program active_selector_column_id",
        )
        zero_columns = _identifier_tuple(
            self.zero_column_ids,
            name="program padding zero_column_ids",
            maximum=MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_COLUMNS_V3,
            allow_empty=True,
            sorted_distinct=True,
        )
        object.__setattr__(self, "zero_column_ids", zero_columns)

    def to_dict(self) -> dict[str, object]:
        return {
            "active_row_rule_id": self.active_row_rule_id,
            "active_selector_column_id": self.active_selector_column_id,
            "zero_column_ids": list(self.zero_column_ids),
        }


def _padding_contract_from_dict(value: object) -> GoldilocksPaddingContractV3:
    item = _object(
        value,
        {"active_row_rule_id", "active_selector_column_id", "zero_column_ids"},
        "program padding contract",
    )
    return GoldilocksPaddingContractV3(
        active_row_rule_id=_json_identifier(
            item["active_row_rule_id"], "program active_row_rule_id"
        ),
        active_selector_column_id=_json_identifier(
            item["active_selector_column_id"],
            "program active_selector_column_id",
        ),
        zero_column_ids=_json_identifier_tuple(
            item["zero_column_ids"],
            name="program padding zero_column_ids",
            maximum=MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_COLUMNS_V3,
            allow_empty=True,
            sorted_distinct=True,
        ),
    )


def _structural_padding_constraints_v3(
    padding_contract: GoldilocksPaddingContractV3,
) -> tuple[GoldilocksAtomicConstraintV3, ...]:
    """Return the canonical AIR terms implied by the signed padding contract.

    Padding is not a verifier-side convenience check.  Every qualified AIR
    compiler must lower these exact terms alongside the user-declared program
    constraints.  The compact numeric suffixes keep IDs within the canonical
    identifier limit even when a program has maximum-length column IDs.
    """

    active = GoldilocksPolynomialExpressionV3(
        op="cell",
        column_id=padding_contract.active_selector_column_id,
        row_offset=0,
    )
    one = GoldilocksPolynomialExpressionV3(op="const", value=1)
    active_minus_one = GoldilocksPolynomialExpressionV3(
        op="add",
        arguments=(
            active,
            GoldilocksPolynomialExpressionV3(op="neg", arguments=(one,)),
        ),
    )
    result: list[GoldilocksAtomicConstraintV3] = [
        GoldilocksAtomicConstraintV3(
            constraint_id="sys.padding.active.one",
            expression=active_minus_one,
            scope="active_rows",
        ),
        GoldilocksAtomicConstraintV3(
            constraint_id="sys.padding.active.zero",
            expression=active,
            scope="padding_rows",
        ),
    ]
    for index, column_id in enumerate(padding_contract.zero_column_ids):
        result.append(
            GoldilocksAtomicConstraintV3(
                constraint_id=f"sys.padding.zero.{index:04d}",
                expression=GoldilocksPolynomialExpressionV3(
                    op="cell",
                    column_id=column_id,
                    row_offset=0,
                ),
                scope="padding_rows",
            )
        )
    return tuple(sorted(result, key=lambda constraint: constraint.constraint_id))


@dataclass(frozen=True, slots=True)
class GoldilocksConstraintProgramV3:
    """One parsed field-polynomial program bound to one signed layout."""

    layout_binding: GoldilocksConstraintProgramLayoutBindingV3
    trace_columns: tuple[GoldilocksTraceColumnV3, ...]
    atomic_constraints: tuple[GoldilocksAtomicConstraintV3, ...]
    padding_contract: GoldilocksPaddingContractV3
    runtime_column_bindings: tuple[GoldilocksRuntimeTraceColumnBindingV3, ...] = ()
    static_column_bindings: tuple[GoldilocksStaticTraceColumnBindingV3, ...] = ()
    source_binding_mode: str = GOLDILOCKS_TRACE_SOURCE_BINDING_MODE_UNBOUND_REFERENCE_V3
    program_abi_id: str = GOLDILOCKS_CONSTRAINT_PROGRAM_ABI_V3
    ir_abi_id: str = GOLDILOCKS_FIELD_POLYNOMIAL_IR_V3
    format_version: int = GOLDILOCKS_CONSTRAINT_PROGRAM_FORMAT_VERSION_V3

    def __post_init__(self) -> None:
        if self.program_abi_id != GOLDILOCKS_CONSTRAINT_PROGRAM_ABI_V3:
            raise ProofV3Error("constraint program ABI is unsupported")
        if self.ir_abi_id != GOLDILOCKS_FIELD_POLYNOMIAL_IR_V3:
            raise ProofV3Error("constraint program IR ABI is unsupported")
        if type(self.format_version) is not int or self.format_version != (
            GOLDILOCKS_CONSTRAINT_PROGRAM_FORMAT_VERSION_V3
        ):
            raise ProofV3Error("constraint program format version is unsupported")
        if not isinstance(
            self.layout_binding,
            GoldilocksConstraintProgramLayoutBindingV3,
        ):
            raise ProofV3Error("constraint program layout binding is malformed")
        if not isinstance(self.trace_columns, tuple) or not self.trace_columns:
            raise ProofV3Error("constraint program trace columns must be nonempty")
        if len(self.trace_columns) > MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_COLUMNS_V3 or not all(
            isinstance(column, GoldilocksTraceColumnV3)
            for column in self.trace_columns
        ):
            raise ProofV3Error("constraint program trace columns are malformed")
        column_ids = tuple(column.column_id for column in self.trace_columns)
        if column_ids != tuple(sorted(column_ids)) or len(column_ids) != len(set(column_ids)):
            raise ProofV3Error("constraint program trace columns are not canonical")
        if not isinstance(self.atomic_constraints, tuple) or not self.atomic_constraints:
            raise ProofV3Error("constraint program atomic constraints must be nonempty")
        if (
            len(self.atomic_constraints)
            > MAX_GOLDILOCKS_ATOMIC_CONSTRAINTS_PER_LAYOUT_V3
            or not all(
                isinstance(constraint, GoldilocksAtomicConstraintV3)
                for constraint in self.atomic_constraints
            )
        ):
            raise ProofV3Error("constraint program atomic constraints are malformed")
        constraint_ids = tuple(
            constraint.constraint_id for constraint in self.atomic_constraints
        )
        if constraint_ids != tuple(sorted(constraint_ids)) or len(constraint_ids) != len(
            set(constraint_ids)
        ):
            raise ProofV3Error("constraint program atomic constraints are not canonical")
        if any(
            constraint_id.startswith(GOLDILOCKS_STRUCTURAL_PADDING_CONSTRAINT_PREFIX_V3)
            for constraint_id in constraint_ids
        ):
            raise ProofV3Error(
                "constraint program reserves the structural padding constraint namespace"
            )
        if not isinstance(self.padding_contract, GoldilocksPaddingContractV3):
            raise ProofV3Error("constraint program padding contract is malformed")
        columns = {column.column_id: column for column in self.trace_columns}
        active = columns.get(self.padding_contract.active_selector_column_id)
        if (
            active is None
            or active.column_role != "selector"
            or active.source_id != "active_rows"
        ):
            raise ProofV3Error(
                "constraint program padding contract has no active-row selector"
            )
        zeroable = tuple(
            sorted(
                column.column_id
                for column in self.trace_columns
                if column.column_role in {"auxiliary", "runtime"}
            )
        )
        if self.padding_contract.zero_column_ids != zeroable:
            raise ProofV3Error(
                "constraint program padding does not zero every witness column"
            )
        if self.source_binding_mode not in _TRACE_SOURCE_BINDING_MODES:
            raise ProofV3Error("constraint program source binding mode is unsupported")
        if not isinstance(self.runtime_column_bindings, tuple) or not all(
            isinstance(binding, GoldilocksRuntimeTraceColumnBindingV3)
            for binding in self.runtime_column_bindings
        ):
            raise ProofV3Error("constraint program runtime column bindings are malformed")
        if not isinstance(self.static_column_bindings, tuple) or not all(
            isinstance(binding, GoldilocksStaticTraceColumnBindingV3)
            for binding in self.static_column_bindings
        ):
            raise ProofV3Error("constraint program static column bindings are malformed")
        runtime_binding_ids = tuple(
            binding.column_id for binding in self.runtime_column_bindings
        )
        static_binding_ids = tuple(
            binding.column_id for binding in self.static_column_bindings
        )
        if (
            runtime_binding_ids != tuple(sorted(runtime_binding_ids))
            or len(runtime_binding_ids) != len(set(runtime_binding_ids))
        ):
            raise ProofV3Error(
                "constraint program runtime column bindings are not canonical"
            )
        if (
            static_binding_ids != tuple(sorted(static_binding_ids))
            or len(static_binding_ids) != len(set(static_binding_ids))
        ):
            raise ProofV3Error(
                "constraint program static column bindings are not canonical"
            )
        runtime_columns = tuple(
            column
            for column in self.trace_columns
            if column.column_role == "runtime"
        )
        fixed_columns = tuple(
            column
            for column in self.trace_columns
            if column.column_role == "fixed"
        )
        if self.source_binding_mode == (
            GOLDILOCKS_TRACE_SOURCE_BINDING_MODE_UNBOUND_REFERENCE_V3
        ):
            if (
                self.runtime_column_bindings
                or self.static_column_bindings
                or runtime_columns
                or fixed_columns
            ):
                raise ProofV3Error(
                    "unbound reference programs may not declare runtime or fixed columns"
                )
        else:
            if runtime_binding_ids != tuple(column.column_id for column in runtime_columns):
                raise ProofV3Error(
                    "constraint program runtime bindings do not exactly cover runtime columns"
                )
            if static_binding_ids != tuple(column.column_id for column in fixed_columns):
                raise ProofV3Error(
                    "constraint program static bindings do not exactly cover fixed columns"
                )
            for binding in self.runtime_column_bindings:
                column = columns.get(binding.column_id)
                if (
                    column is None
                    or column.column_role != "runtime"
                    or column.source_id != binding.tensor_id
                ):
                    raise ProofV3Error(
                        "constraint program runtime binding does not match its trace column"
                    )
            for binding in self.static_column_bindings:
                column = columns.get(binding.column_id)
                if (
                    column is None
                    or column.column_role != "fixed"
                    or column.source_id != binding.table_id
                ):
                    raise ProofV3Error(
                        "constraint program static binding does not match its trace column"
                    )
            if {
                binding.tensor_id for binding in self.runtime_column_bindings
            } != set(self.layout_binding.runtime_tensor_ids):
                raise ProofV3Error(
                    "constraint program runtime bindings do not exactly cover layout tensors"
                )
            if {
                binding.table_id for binding in self.static_column_bindings
            } != set(self.layout_binding.static_table_ids):
                raise ProofV3Error(
                    "constraint program static bindings do not exactly cover layout tables"
                )
        for constraint in self.atomic_constraints:
            constraint.expression._validate_references(columns=columns)
        if self.max_constraint_degree >= 1 << 32:
            raise ProofV3Error("constraint program degree exceeds uint32")

    @property
    def max_constraint_degree(self) -> int:
        return max(constraint.degree for constraint in self.atomic_constraints)

    @property
    def structural_padding_constraints(self) -> tuple[GoldilocksAtomicConstraintV3, ...]:
        """Return exact selector and zero-padding AIR terms for this program.

        The terms are derived solely from the signed padding contract, not
        supplied by a prover or native backend.  They deliberately remain
        separate from ``atomic_constraints`` so the signed layout's existing
        user-constraint coordinate ABI stays stable while this unregistered
        compiler reference gains explicit algebraic padding semantics.
        """

        return _structural_padding_constraints_v3(self.padding_contract)

    @property
    def air_constraints(self) -> tuple[GoldilocksAtomicConstraintV3, ...]:
        """Return every canonical user and structural AIR constraint by ID."""

        return tuple(
            sorted(
                (*self.atomic_constraints, *self.structural_padding_constraints),
                key=lambda constraint: constraint.constraint_id,
            )
        )

    @property
    def max_air_constraint_degree(self) -> int:
        """Return the degree bound including canonical padding constraints."""

        return max(constraint.degree for constraint in self.air_constraints)

    @property
    def has_exact_source_bindings(self) -> bool:
        """Whether every signed runtime/static source has a cell coordinate."""

        return self.source_binding_mode == (
            GOLDILOCKS_TRACE_SOURCE_BINDING_MODE_EXACT_LAYOUT_V3
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "atomic_constraints": [
                constraint.to_dict() for constraint in self.atomic_constraints
            ],
            "format_version": self.format_version,
            "ir_abi_id": self.ir_abi_id,
            "layout_binding": self.layout_binding.to_dict(),
            "padding_contract": self.padding_contract.to_dict(),
            "program_abi_id": self.program_abi_id,
            "runtime_column_bindings": [
                binding.to_dict() for binding in self.runtime_column_bindings
            ],
            "source_binding_mode": self.source_binding_mode,
            "static_column_bindings": [
                binding.to_dict() for binding in self.static_column_bindings
            ],
            "trace_columns": [column.to_dict() for column in self.trace_columns],
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(
            self.to_dict(),
            name="constraint program",
            maximum=MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_BYTES_V3,
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_PROGRAM_DIGEST_DOMAIN + self.canonical_bytes()).digest()

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "GoldilocksConstraintProgramV3":
        if (
            type(encoded) is not bytes
            or not encoded
            or len(encoded) > MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_BYTES_V3
        ):
            raise ProofV3Error("constraint program byte length is out of range")
        try:
            value = json.loads(
                encoded.decode("ascii"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except ProofV3Error:
            raise
        except (RecursionError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProofV3Error("constraint program is not canonical JSON") from exc
        item = _object(
            value,
            {
                "atomic_constraints",
                "format_version",
                "ir_abi_id",
                "layout_binding",
                "padding_contract",
                "program_abi_id",
                "runtime_column_bindings",
                "source_binding_mode",
                "static_column_bindings",
                "trace_columns",
            },
            "constraint program",
        )
        result = cls(
            layout_binding=_layout_binding_from_dict(item["layout_binding"]),
            trace_columns=tuple(
                _trace_column_from_dict(entry)
                for entry in _list(
                    item["trace_columns"],
                    "constraint program trace_columns",
                    MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_COLUMNS_V3,
                )
            ),
            atomic_constraints=tuple(
                _atomic_constraint_from_dict(entry)
                for entry in _list(
                    item["atomic_constraints"],
                    "constraint program atomic_constraints",
                    MAX_GOLDILOCKS_ATOMIC_CONSTRAINTS_PER_LAYOUT_V3,
                )
            ),
            padding_contract=_padding_contract_from_dict(item["padding_contract"]),
            runtime_column_bindings=tuple(
                _runtime_trace_column_binding_from_dict(entry)
                for entry in _list(
                    item["runtime_column_bindings"],
                    "constraint program runtime column bindings",
                    MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_COLUMNS_V3,
                    allow_empty=True,
                )
            ),
            static_column_bindings=tuple(
                _static_trace_column_binding_from_dict(entry)
                for entry in _list(
                    item["static_column_bindings"],
                    "constraint program static column bindings",
                    MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_COLUMNS_V3,
                    allow_empty=True,
                )
            ),
            source_binding_mode=_json_identifier(
                item["source_binding_mode"], "constraint program source binding mode"
            ),
            program_abi_id=_json_identifier(
                item["program_abi_id"], "constraint program ABI"
            ),
            ir_abi_id=_json_identifier(item["ir_abi_id"], "constraint program IR ABI"),
            format_version=_u32(
                item["format_version"], "constraint program format_version", positive=True
            ),
        )
        if result.canonical_bytes() != encoded:
            raise ProofV3Error("constraint program is not canonically encoded")
        return result

    def validate_layout(
        self,
        *,
        layout_index: int,
        layout: GoldilocksConstraintLayoutV3,
    ) -> None:
        """Require exact layout identity, user-atom coverage, and degree.

        Structural padding terms are authenticated through this program's
        padding contract and derived identically by every compiler; they are
        intentionally not miner-selectable layout coordinates.
        """

        if not isinstance(layout, GoldilocksConstraintLayoutV3):
            raise ProofV3Error("constraint program layout has an unexpected type")
        if not self.layout_binding.matches_layout(
            layout_index=layout_index,
            layout=layout,
        ):
            raise ProofV3Error("constraint program does not match its signed layout")
        if tuple(
            constraint.constraint_id for constraint in self.atomic_constraints
        ) != layout.atomic_constraint_ids:
            raise ProofV3Error(
                "constraint program does not exactly cover signed atomic constraints"
            )
        if self.max_constraint_degree != layout.max_constraint_degree:
            raise ProofV3Error(
                "constraint program degree does not match the signed layout"
            )
        if layout.padding_rule_id != GOLDILOCKS_TRACE_PADDING_RULE_V3:
            raise ProofV3Error("constraint program layout has an unsupported padding rule")
        runtime_sources = {
            column.source_id
            for column in self.trace_columns
            if column.column_role == "runtime"
        }
        expected_runtime_sources = set(layout.runtime_tensor_ids)
        if self.has_exact_source_bindings:
            runtime_sources_match = runtime_sources == expected_runtime_sources
        else:
            runtime_sources_match = runtime_sources.issubset(expected_runtime_sources)
        if not runtime_sources_match:
            raise ProofV3Error(
                "constraint program runtime column does not match its signed layout"
            )
        fixed_sources = {
            column.source_id
            for column in self.trace_columns
            if column.column_role == "fixed"
        }
        expected_fixed_sources = set(layout.static_table_ids)
        if self.has_exact_source_bindings:
            fixed_sources_match = fixed_sources == expected_fixed_sources
        else:
            fixed_sources_match = fixed_sources.issubset(expected_fixed_sources)
        if not fixed_sources_match:
            raise ProofV3Error(
                "constraint program fixed column does not match its signed static table"
            )


@dataclass(frozen=True, slots=True)
class GoldilocksConstraintProgramBundleV3:
    """One ordered parsed program for every signed constraint-system layout."""

    relation_binding_digest: bytes
    programs: tuple[GoldilocksConstraintProgramV3, ...]
    bundle_abi_id: str = GOLDILOCKS_CONSTRAINT_PROGRAM_BUNDLE_ABI_V3
    format_version: int = GOLDILOCKS_CONSTRAINT_PROGRAM_BUNDLE_FORMAT_VERSION_V3

    def __post_init__(self) -> None:
        if self.bundle_abi_id != GOLDILOCKS_CONSTRAINT_PROGRAM_BUNDLE_ABI_V3:
            raise ProofV3Error("constraint program bundle ABI is unsupported")
        if type(self.format_version) is not int or self.format_version != (
            GOLDILOCKS_CONSTRAINT_PROGRAM_BUNDLE_FORMAT_VERSION_V3
        ):
            raise ProofV3Error("constraint program bundle format version is unsupported")
        _fixed32(
            self.relation_binding_digest,
            "constraint program bundle relation_binding_digest",
            nonzero=True,
        )
        if not isinstance(self.programs, tuple) or not self.programs:
            raise ProofV3Error("constraint program bundle programs must be nonempty")
        if len(self.programs) > MAX_GOLDILOCKS_CONSTRAINT_PROGRAMS_V3 or not all(
            isinstance(program, GoldilocksConstraintProgramV3)
            for program in self.programs
        ):
            raise ProofV3Error("constraint program bundle programs are malformed")
        layout_indices = tuple(
            program.layout_binding.layout_index for program in self.programs
        )
        if layout_indices != tuple(range(len(self.programs))):
            raise ProofV3Error(
                "constraint program bundle layouts are incomplete or reordered"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_abi_id": self.bundle_abi_id,
            "format_version": self.format_version,
            "programs": [program.to_dict() for program in self.programs],
            "relation_binding_digest": self.relation_binding_digest.hex(),
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(
            self.to_dict(),
            name="constraint program bundle",
            maximum=MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_BUNDLE_BYTES_V3,
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_BUNDLE_DIGEST_DOMAIN + self.canonical_bytes()).digest()

    @classmethod
    def from_canonical_bytes(
        cls,
        encoded: bytes,
    ) -> "GoldilocksConstraintProgramBundleV3":
        if (
            type(encoded) is not bytes
            or not encoded
            or len(encoded) > MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_BUNDLE_BYTES_V3
        ):
            raise ProofV3Error("constraint program bundle byte length is out of range")
        try:
            value = json.loads(
                encoded.decode("ascii"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except ProofV3Error:
            raise
        except (RecursionError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProofV3Error("constraint program bundle is not canonical JSON") from exc
        item = _object(
            value,
            {"bundle_abi_id", "format_version", "programs", "relation_binding_digest"},
            "constraint program bundle",
        )
        result = cls(
            relation_binding_digest=_json_digest(
                item["relation_binding_digest"],
                "constraint program bundle relation_binding_digest",
            ),
            programs=tuple(
                GoldilocksConstraintProgramV3.from_canonical_bytes(
                    _canonical_json_bytes(
                        entry,
                        name="constraint program bundle program",
                        maximum=MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_BYTES_V3,
                    )
                )
                for entry in _list(
                    item["programs"],
                    "constraint program bundle programs",
                    MAX_GOLDILOCKS_CONSTRAINT_PROGRAMS_V3,
                )
            ),
            bundle_abi_id=_json_identifier(
                item["bundle_abi_id"], "constraint program bundle ABI"
            ),
            format_version=_u32(
                item["format_version"],
                "constraint program bundle format_version",
                positive=True,
            ),
        )
        if result.canonical_bytes() != encoded:
            raise ProofV3Error("constraint program bundle is not canonically encoded")
        return result

    def validate_constraint_system(
        self,
        *,
        constraint_system: GoldilocksExecutionConstraintSystemV3,
        relation: ExecutionRelationSpecV3,
    ) -> None:
        """Require exact program/layout coverage for one signed relation."""

        if not isinstance(constraint_system, GoldilocksExecutionConstraintSystemV3):
            raise ProofV3Error("constraint program system has an unexpected type")
        if not isinstance(relation, ExecutionRelationSpecV3):
            raise ProofV3Error("constraint program relation has an unexpected type")
        expected_relation_binding = constraint_system_relation_binding_digest_v3(relation)
        if (
            self.relation_binding_digest != expected_relation_binding
            or constraint_system.relation_binding_digest != expected_relation_binding
        ):
            raise ProofV3Error(
                "constraint program bundle does not match the signed relation"
            )
        if self.digest() != constraint_system.constraint_program_bundle_digest:
            raise ProofV3Error(
                "constraint program bundle does not match the signed constraint system"
            )
        if len(self.programs) != len(constraint_system.layouts):
            raise ProofV3Error(
                "constraint program bundle does not exactly cover signed layouts"
            )
        for index, (program, layout) in enumerate(
            zip(self.programs, constraint_system.layouts, strict=True)
        ):
            program.validate_layout(layout_index=index, layout=layout)
            if program.digest() != layout.constraint_program_digest:
                raise ProofV3Error(
                    "constraint program digest does not match the signed layout"
                )
            if not program.has_exact_source_bindings:
                continue
            static_tables = {
                binding.table_id: binding for binding in relation.static_table_bindings
            }
            for binding in program.runtime_column_bindings:
                (
                    expected_encoding,
                    expected_layout,
                    expected_axis,
                    expected_elements_per_token,
                ) = _runtime_tensor_binding_metadata_v3(
                    relation=relation,
                    tensor_id=binding.tensor_id,
                )
                if (
                    binding.source_encoding_id != expected_encoding
                    or binding.source_layout_id != expected_layout
                    or binding.token_axis_rule_id != expected_axis
                    or binding.elements_per_token != expected_elements_per_token
                ):
                    raise ProofV3Error(
                        "constraint program runtime binding does not match its signed tensor ABI"
                    )
                if (
                    binding.token_axis_rule_id
                    == GOLDILOCKS_RUNTIME_TOKEN_AXIS_CONTEXT_V3
                    and "decode" in layout.phases
                ) or (
                    binding.token_axis_rule_id
                    == GOLDILOCKS_RUNTIME_TOKEN_AXIS_DECODE_V3
                    and "prefill" in layout.phases
                ):
                    raise ProofV3Error(
                        "constraint program runtime binding has an incompatible phase"
                    )
                last_subrow_element = (
                    binding.element_offset
                    + (layout.rows_per_token - 1) * binding.trace_row_stride
                )
                if last_subrow_element >= binding.elements_per_token:
                    raise ProofV3Error(
                        "constraint program runtime binding exceeds its tensor row"
                    )
            for binding in program.static_column_bindings:
                static_table = static_tables.get(binding.table_id)
                if static_table is None:
                    raise ProofV3Error(
                        "constraint program static binding references an unknown table"
                    )
                if (
                    binding.cell_encoding_id != static_table.element_encoding_id
                    or binding.logical_leaf_offset >= static_table.logical_leaf_count
                ):
                    raise ProofV3Error(
                        "constraint program static binding does not match its signed table ABI"
                    )


@dataclass(frozen=True, slots=True)
class GoldilocksConstraintTraceReferenceV3:
    """A bounded in-memory trace for field-polynomial compiler conformance.

    This is not a commitment, payload, runtime capture, or execution proof.
    It exists solely so the parsed finite-field IR has a deterministic,
    fail-closed semantic reference before a native GPU trace backend consumes
    the same program artifact. A trace explicitly names the exact program
    digest it was constructed for, has a radix-2 physical row count, and uses
    canonical Goldilocks values only.
    """

    constraint_program_digest: bytes
    rows: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        digest = _fixed32(
            self.constraint_program_digest,
            "reference trace constraint_program_digest",
            nonzero=True,
        )
        if not isinstance(self.rows, tuple) or not self.rows:
            raise ProofV3Error("reference trace rows must be a nonempty tuple")
        row_count = _power_of_two(len(self.rows), "reference trace row count")
        if row_count > MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE:
            raise ProofV3Error("reference trace exceeds the CPU reference cap")
        if not all(isinstance(row, tuple) and row for row in self.rows):
            raise ProofV3Error("reference trace rows must be nonempty tuples")
        width = len(self.rows[0])
        normalized_rows: list[tuple[int, ...]] = []
        for row_index, row in enumerate(self.rows):
            if len(row) != width:
                raise ProofV3Error("reference trace rows have inconsistent widths")
            normalized_rows.append(
                tuple(
                    canonical_goldilocks(
                        value,
                        f"reference trace row[{row_index}][{column_index}]",
                    )
                    for column_index, value in enumerate(row)
                )
            )
        object.__setattr__(self, "constraint_program_digest", digest)
        object.__setattr__(self, "rows", tuple(normalized_rows))


def _reference_scope_rows(
    *,
    scope: str,
    active_row_count: int,
    physical_row_count: int,
) -> range:
    if scope == "active_rows":
        return range(active_row_count)
    if scope == "first_active_row":
        return range(1)
    if scope == "last_active_row":
        return range(active_row_count - 1, active_row_count)
    if scope == "padding_rows":
        return range(active_row_count, physical_row_count)
    assert scope == "transition_rows"
    return range(active_row_count - 1)


def verify_goldilocks_constraint_program_reference_v3(
    *,
    program: GoldilocksConstraintProgramV3,
    trace: GoldilocksConstraintTraceReferenceV3,
    token_count: int,
) -> None:
    """Evaluate a tiny parsed field-polynomial trace fail-closed.

    The reference verifies only the exact polynomial IR, its active-row
    formula, selector, and padding contract. It neither authenticates a trace
    nor lowers model tensors, encodings, static lookups, RAM, attention, GDN,
    or a prompt-to-token execution. Production code must never use it as an
    execution-proof verifier.
    """

    try:
        if not isinstance(program, GoldilocksConstraintProgramV3):
            raise ProofV3VerificationError(
                "reference trace program has an unexpected type"
            )
        if not isinstance(trace, GoldilocksConstraintTraceReferenceV3):
            raise ProofV3VerificationError("reference trace has an unexpected type")
        if trace.constraint_program_digest != program.digest():
            raise ProofV3VerificationError(
                "reference trace belongs to a different program"
            )
        checked_token_count = _u64(
            token_count,
            "reference trace token_count",
            positive=True,
        )
        active_row_count = checked_token_count * program.layout_binding.rows_per_token
        if active_row_count > len(trace.rows):
            raise ProofV3VerificationError(
                "reference trace does not contain the required active rows"
            )
        if len(trace.rows) < program.layout_binding.minimum_trace_rows:
            raise ProofV3VerificationError(
                "reference trace does not meet the signed minimum domain"
            )
        expected_width = len(program.trace_columns)
        if any(len(row) != expected_width for row in trace.rows):
            raise ProofV3VerificationError(
                "reference trace column width does not match the program"
            )
        columns = {
            column.column_id: index
            for index, column in enumerate(program.trace_columns)
        }
        for constraint in program.air_constraints:
            for row_index in _reference_scope_rows(
                scope=constraint.scope,
                active_row_count=active_row_count,
                physical_row_count=len(trace.rows),
            ):
                value = constraint.expression._evaluate(
                    current_row=trace.rows[row_index],
                    next_row=(
                        trace.rows[row_index + 1]
                        if row_index + 1 < active_row_count
                        else None
                    ),
                    column_positions=columns,
                )
                if value != 0:
                    raise ProofV3VerificationError(
                        "reference trace violates constraint "
                        f"{constraint.constraint_id} at row {row_index}"
                    )
    except ProofV3VerificationError:
        raise
    except ProofV3Error as exc:
        raise ProofV3VerificationError(
            "reference trace/program data is malformed"
        ) from exc


__all__ = [
    "GOLDILOCKS_ACTIVE_ROW_RULE_V3",
    "GOLDILOCKS_CONSTRAINT_PROGRAM_ABI_V3",
    "GOLDILOCKS_CONSTRAINT_PROGRAM_BUNDLE_ABI_V3",
    "GOLDILOCKS_CONSTRAINT_PROGRAM_BUNDLE_FORMAT_VERSION_V3",
    "GOLDILOCKS_CONSTRAINT_PROGRAM_FORMAT_VERSION_V3",
    "GOLDILOCKS_FIELD_POLYNOMIAL_IR_V3",
    "GOLDILOCKS_RUNTIME_TRACE_FIELD_ENCODING_V3",
    "GOLDILOCKS_RUNTIME_TOKEN_AXIS_CONTEXT_V3",
    "GOLDILOCKS_RUNTIME_TOKEN_AXIS_DECODE_V3",
    "GOLDILOCKS_RUNTIME_TOKEN_AXIS_SEQUENCE_V3",
    "GOLDILOCKS_STRUCTURAL_PADDING_CONSTRAINT_PREFIX_V3",
    "GOLDILOCKS_TRACE_SOURCE_BINDING_MODE_EXACT_LAYOUT_V3",
    "GOLDILOCKS_TRACE_SOURCE_BINDING_MODE_UNBOUND_REFERENCE_V3",
    "MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_BUNDLE_BYTES_V3",
    "MAX_GOLDILOCKS_CONSTRAINT_PROGRAM_BYTES_V3",
    "GoldilocksAtomicConstraintV3",
    "GoldilocksConstraintProgramBundleV3",
    "GoldilocksConstraintProgramLayoutBindingV3",
    "GoldilocksConstraintProgramV3",
    "GoldilocksConstraintTraceReferenceV3",
    "GoldilocksPaddingContractV3",
    "GoldilocksPolynomialExpressionV3",
    "GoldilocksRuntimeTraceColumnBindingV3",
    "GoldilocksStaticTraceColumnBindingV3",
    "GoldilocksTraceColumnV3",
    "verify_goldilocks_constraint_program_reference_v3",
]
