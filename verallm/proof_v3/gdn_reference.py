"""Exact two-row fixed-point GDN reference for future V3 transition work.

This module is deliberately unregistered.  It is not exported from
``verallm.proof_v3``, has no payload format, and is not imported by miner,
validator, profile, sidecar, or adapter runtime code.  It models one tiny,
fully opened Gated-Delta-Net-shaped recurrence with signed Q16.16 integers and
finite static lookup tables.  It is a CPU golden-vector/reference relation,
not a Qwen kernel implementation, a production proof backend, or evidence of
model-substitution resistance.

The narrow scope is intentional:

* ``nk = nv = dk = dv = 1``, a two-tap causal convolution, and exactly two
  contiguous decode rows;
* every fixed-point multiplication uses round-to-nearest, ties-to-even; there
  is no float, tolerance, saturation, or implicit reduction;
* all nonlinear operations are exact finite lookup-table semantics;
* roots are deterministic Goldilocks-row Merkle commitments bound to a frozen
  envelope/chunk/static context, but field arithmetic is never used for the
  integer recurrence itself.

The entry state is only externally authenticated in this reference.  It does
not establish prefill provenance, complete linear/RAM coverage, or a global
prompt-to-token proof.  Keep this module unregistered until those relations are
joined by a qualified native backend.  Its static parameter/lookup digest is
also toy-local data: it has no authenticated equality relation to the envelope
static manifest, registered model weights, residual path, or output token.
"""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.accumulator import ExecutionChunkCommitmentV3
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_merkle_reference import GoldilocksMerkleTreeReference
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.payload import ProofV3CommitmentEnvelope


GDN_FIXED_REFERENCE_ABI_V3: Final = "gdn.reference.fixed.v1"
GDN_FIXED_ENCODING_ABI_V3: Final = "signed.q16.16.rne-even.v1"
GDN_FIXED_FRACTION_BITS_V3: Final = 16
GDN_FIXED_SCALE_V3: Final = 1 << GDN_FIXED_FRACTION_BITS_V3
GDN_FIXED_ROWS_V3: Final = 2
GDN_FIXED_CONV_KERNEL_SIZE_V3: Final = 2
GDN_FIXED_LOOKUP_INDEX_FRACTION_BITS_V3: Final = 8
GDN_FIXED_LOOKUP_INDEX_MIN_V3: Final = -2048
GDN_FIXED_LOOKUP_INDEX_MAX_V3: Final = 2048
GDN_FIXED_LOOKUP_ENTRY_COUNT_V3: Final = (
    GDN_FIXED_LOOKUP_INDEX_MAX_V3 - GDN_FIXED_LOOKUP_INDEX_MIN_V3 + 1
)
GDN_FIXED_LOOKUP_TABLE_IDS_V3: Final = (
    "decay",
    "exp",
    "invsqrt",
    "sigmoid",
    "silu",
    "softplus",
)
GDN_FIXED_MIN_VALUE_V3: Final = -(1 << 30)
GDN_FIXED_MAX_VALUE_V3: Final = (1 << 30) - 1
_GDN_FIXED_DECODE_PHASE_CODE: Final = 1

_IDENTIFIER_RE: Final = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,127}$")
_ROOT_CONTEXT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GDN_FIXED/ROOT_CONTEXT/SHA256"
_ROOT_ROLE_DOMAIN: Final = b"VERATHOS/PROOF_V3/GDN_FIXED/ROOT_ROLE/SHA256"
_ROOT_INVENTORY_DOMAIN: Final = b"VERATHOS/PROOF_V3/GDN_FIXED/ROOTS/SHA256"
_STATEMENT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GDN_FIXED/STATEMENT/SHA256"
_STATIC_DOMAIN: Final = b"VERATHOS/PROOF_V3/GDN_FIXED/STATIC/SHA256"
_PRECOMMIT_FACTORY_TOKEN = object()


def _fixed32(value: object, name: str, *, nonzero: bool = False) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    if nonzero and value == bytes(32):
        raise ProofV3Error(f"{name} must not be zero")
    return value


def _u32(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProofV3Error(f"{name} must be an unsigned 32-bit integer")
    if value < (1 if positive else 0) or value >= 1 << 32:
        qualifier = "positive " if positive else ""
        raise ProofV3Error(f"{name} must be a {qualifier}unsigned 32-bit integer")
    return value


def _u64(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= 1 << 64
    ):
        raise ProofV3Error(f"{name} must be an unsigned 64-bit integer")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ProofV3Error(f"{name} is not a canonical identifier")
    return value


def _fixed(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProofV3Error(f"{name} must be a signed fixed-point integer")
    if not GDN_FIXED_MIN_VALUE_V3 <= value <= GDN_FIXED_MAX_VALUE_V3:
        raise ProofV3Error(f"{name} is outside the fixed-point reference range")
    return value


def _materialize_exact(
    value: object,
    *,
    length: int,
    name: str,
) -> tuple[object, ...]:
    """Bound one public iterable to its exact fixed ABI length.

    Public reference inputs must not materialize an arbitrary generator before
    discovering that it has too many rows or columns.  Consume at most one
    surplus item, then fail closed.
    """

    if isinstance(value, (str, bytes, bytearray, memoryview)):
        raise ProofV3Error(f"{name} must be an iterable")
    try:
        iterator = iter(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ProofV3Error(f"{name} must be an iterable") from exc
    items: list[object] = []
    for item in iterator:
        if len(items) >= length:
            raise ProofV3Error(f"{name} must contain exactly {length} values")
        items.append(item)
    if len(items) != length:
        raise ProofV3Error(f"{name} must contain exactly {length} values")
    return tuple(items)


def _fixed_vector(value: object, *, length: int, name: str) -> tuple[int, ...]:
    items = _materialize_exact(value, length=length, name=name)
    return tuple(_fixed(item, f"{name}[{index}]") for index, item in enumerate(items))


def _fixed_rows(
    value: object,
    *,
    row_count: int,
    width: int,
    name: str,
) -> tuple[tuple[int, ...], ...]:
    rows = _materialize_exact(value, length=row_count, name=name)
    return tuple(
        _fixed_vector(row, length=width, name=f"{name}[{index}]")
        for index, row in enumerate(rows)
    )


def _fixed_bytes(values: tuple[int, ...]) -> bytes:
    return struct.pack("<" + "i" * len(values), *values)


def fixed_rne_divide_v3(numerator: object, divisor: object) -> int:
    """Divide signed integers with exact round-to-nearest, ties-to-even."""

    if isinstance(numerator, bool) or not isinstance(numerator, int):
        raise ProofV3Error("fixed-point numerator must be an integer")
    if isinstance(divisor, bool) or not isinstance(divisor, int) or divisor <= 0:
        raise ProofV3Error("fixed-point divisor must be a positive integer")
    sign = -1 if numerator < 0 else 1
    quotient, remainder = divmod(abs(numerator), divisor)
    doubled = remainder * 2
    if doubled > divisor or (doubled == divisor and quotient & 1):
        quotient += 1
    return _fixed(sign * quotient, "fixed-point rounded quotient")


def fixed_mul_q16_v3(left: object, right: object) -> int:
    """Multiply two Q16.16 values with the sole reference rounding rule."""

    return fixed_rne_divide_v3(
        _fixed(left, "fixed-point multiplication left")
        * _fixed(right, "fixed-point multiplication right"),
        GDN_FIXED_SCALE_V3,
    )


def _fixed_add(left: object, right: object, name: str) -> int:
    return _fixed(
        _fixed(left, f"{name} left") + _fixed(right, f"{name} right"),
        name,
    )


def _fixed_sub(left: object, right: object, name: str) -> int:
    return _fixed(
        _fixed(left, f"{name} left") - _fixed(right, f"{name} right"),
        name,
    )


def _fixed_neg(value: object, name: str) -> int:
    return _fixed(-_fixed(value, name), name)


def _fixed_conv2(
    *,
    previous: object,
    current: object,
    previous_weight: object,
    current_weight: object,
) -> int:
    """Apply a two-tap Q16.16 convolution with one final RNE-even rounding."""

    numerator = _fixed(previous, "GDN convolution previous") * _fixed(
        previous_weight, "GDN convolution previous weight"
    ) + _fixed(current, "GDN convolution current") * _fixed(
        current_weight, "GDN convolution current weight"
    )
    return fixed_rne_divide_v3(numerator, GDN_FIXED_SCALE_V3)


def _lookup_index(value: object) -> int:
    return fixed_rne_divide_v3(
        _fixed(value, "GDN lookup input"),
        1 << (GDN_FIXED_FRACTION_BITS_V3 - GDN_FIXED_LOOKUP_INDEX_FRACTION_BITS_V3),
    )


def gdn_fixed_to_goldilocks_v3(value: object) -> int:
    """Inject one bounded signed fixed value into Goldilocks without reduction."""

    fixed = _fixed(value, "GDN fixed Goldilocks value")
    return fixed if fixed >= 0 else GOLDILOCKS_MODULUS + fixed


@dataclass(frozen=True, slots=True)
class FixedLookupTableV3:
    """One complete signed-Q8.8-indexed finite lookup table.

    The output vector covers every key in the fixed ABI interval in increasing
    order.  Lookup is exact and fails outside the interval; no clipping or
    fallback value exists.
    """

    table_id: str
    outputs: tuple[int, ...]
    input_min: int = GDN_FIXED_LOOKUP_INDEX_MIN_V3
    input_max: int = GDN_FIXED_LOOKUP_INDEX_MAX_V3

    def __post_init__(self) -> None:
        _identifier(self.table_id, "GDN lookup table_id")
        input_min = _fixed(self.input_min, "GDN lookup table input_min")
        input_max = _fixed(self.input_max, "GDN lookup table input_max")
        if (
            input_min != GDN_FIXED_LOOKUP_INDEX_MIN_V3
            or input_max != GDN_FIXED_LOOKUP_INDEX_MAX_V3
        ):
            raise ProofV3Error("GDN lookup table range is not the fixed ABI range")
        object.__setattr__(self, "input_min", input_min)
        object.__setattr__(self, "input_max", input_max)
        outputs = _fixed_vector(
            self.outputs,
            length=GDN_FIXED_LOOKUP_ENTRY_COUNT_V3,
            name=f"GDN lookup table {self.table_id} outputs",
        )
        object.__setattr__(self, "outputs", outputs)

    def lookup_q16(self, value: object) -> tuple[int, int]:
        """Return the exact Q8.8 index and table output for one Q16.16 value."""

        index = _lookup_index(value)
        if not self.input_min <= index <= self.input_max:
            raise ProofV3Error(
                f"GDN lookup table {self.table_id} has no entry for index {index}"
            )
        return index, self.outputs[index - self.input_min]

    def canonical_bytes(self) -> bytes:
        table_id = self.table_id.encode("ascii")
        return (
            struct.pack(
                "<BhhI",
                len(table_id),
                self.input_min,
                self.input_max,
                len(self.outputs),
            )
            + table_id
            + _fixed_bytes(self.outputs)
        )


@dataclass(frozen=True, slots=True)
class GDNFixedStaticParametersV3:
    """Static exact parameters and every nonlinear lookup-table definition."""

    qkvz_weights: tuple[int, int, int, int]
    ba_weights: tuple[int, int]
    conv_weights: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    a_log: int
    dt_bias: int
    q_norm_epsilon: int
    output_norm_epsilon: int
    norm_weight: int
    output_weight: int
    lookup_tables: tuple[FixedLookupTableV3, ...]
    fraction_bits: int = GDN_FIXED_FRACTION_BITS_V3
    encoding_abi_id: str = GDN_FIXED_ENCODING_ABI_V3

    def __post_init__(self) -> None:
        fraction_bits = _u32(self.fraction_bits, "GDN fixed fraction_bits")
        if fraction_bits != GDN_FIXED_FRACTION_BITS_V3:
            raise ProofV3Error("GDN fixed fraction_bits is unsupported")
        object.__setattr__(self, "fraction_bits", fraction_bits)
        if self.encoding_abi_id != GDN_FIXED_ENCODING_ABI_V3:
            raise ProofV3Error("GDN fixed encoding ABI is unsupported")
        object.__setattr__(
            self,
            "qkvz_weights",
            _fixed_vector(self.qkvz_weights, length=4, name="GDN QKVZ weights"),
        )
        object.__setattr__(
            self,
            "ba_weights",
            _fixed_vector(self.ba_weights, length=2, name="GDN BA weights"),
        )
        conv_rows = _materialize_exact(
            self.conv_weights,
            length=3,
            name="GDN convolution weights",
        )
        object.__setattr__(
            self,
            "conv_weights",
            tuple(
                _fixed_vector(row, length=2, name=f"GDN convolution weights[{index}]")
                for index, row in enumerate(conv_rows)
            ),
        )
        for name in (
            "a_log",
            "dt_bias",
            "q_norm_epsilon",
            "output_norm_epsilon",
            "norm_weight",
            "output_weight",
        ):
            object.__setattr__(self, name, _fixed(getattr(self, name), f"GDN {name}"))
        if self.q_norm_epsilon <= 0 or self.output_norm_epsilon <= 0:
            raise ProofV3Error("GDN fixed normalization epsilon must be positive")
        tables = _materialize_exact(
            self.lookup_tables,
            length=len(GDN_FIXED_LOOKUP_TABLE_IDS_V3),
            name="GDN lookup tables",
        )
        if not all(isinstance(table, FixedLookupTableV3) for table in tables):
            raise ProofV3Error("GDN lookup tables have an unexpected type")
        table_ids = tuple(table.table_id for table in tables)
        if table_ids != GDN_FIXED_LOOKUP_TABLE_IDS_V3:
            raise ProofV3Error("GDN lookup tables are incomplete or noncanonical")
        object.__setattr__(self, "lookup_tables", tables)

    def lookup_q16(self, table_id: str, value: object) -> tuple[int, int]:
        for table in self.lookup_tables:
            if table.table_id == table_id:
                return table.lookup_q16(value)
        raise ProofV3Error("GDN fixed lookup table is unavailable")

    def canonical_bytes(self) -> bytes:
        abi = self.encoding_abi_id.encode("ascii")
        fixed_values = (
            self.qkvz_weights
            + self.ba_weights
            + tuple(value for row in self.conv_weights for value in row)
            + (
                self.a_log,
                self.dt_bias,
                self.q_norm_epsilon,
                self.output_norm_epsilon,
                self.norm_weight,
                self.output_weight,
            )
        )
        return (
            struct.pack("<BIB", len(abi), self.fraction_bits, len(self.lookup_tables))
            + abi
            + _fixed_bytes(fixed_values)
            + b"".join(table.canonical_bytes() for table in self.lookup_tables)
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_STATIC_DOMAIN + self.canonical_bytes()).digest()


@dataclass(frozen=True, slots=True)
class GDNFixedStateV3:
    """The complete scalar GDN state before or after one reference row."""

    convolution_history: tuple[int, int, int]
    recurrent_state: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "convolution_history",
            _fixed_vector(
                self.convolution_history,
                length=3,
                name="GDN convolution history",
            ),
        )
        object.__setattr__(
            self,
            "recurrent_state",
            _fixed(self.recurrent_state, "GDN recurrent state"),
        )

    def as_row(self) -> tuple[int, int, int, int]:
        return (*self.convolution_history, self.recurrent_state)


@dataclass(frozen=True, slots=True)
class GDNFixedChunkIdentityV3:
    """The root-independent identity of exactly two contiguous decode rows."""

    phase: str
    chunk_index: int
    logical_token_start: int
    token_count: int
    cache_lease_digest: bytes
    sidecar_generation_digest: bytes
    scheduler_coverage_digest: bytes

    def __post_init__(self) -> None:
        if self.phase != "decode":
            raise ProofV3Error("GDN fixed reference supports decode chunks only")
        chunk_index = _u32(self.chunk_index, "GDN fixed chunk_index")
        logical_token_start = _u32(
            self.logical_token_start,
            "GDN fixed logical_token_start",
        )
        token_count = _u32(self.token_count, "GDN fixed token_count", positive=True)
        if token_count != GDN_FIXED_ROWS_V3:
            raise ProofV3Error("GDN fixed reference requires exactly two chunk rows")
        if logical_token_start + token_count - 1 > GDN_FIXED_MAX_VALUE_V3:
            raise ProofV3Error(
                "GDN fixed logical token range is outside the reference range"
            )
        object.__setattr__(self, "chunk_index", chunk_index)
        object.__setattr__(self, "logical_token_start", logical_token_start)
        object.__setattr__(self, "token_count", token_count)
        object.__setattr__(
            self,
            "cache_lease_digest",
            _fixed32(
                self.cache_lease_digest,
                "GDN fixed cache_lease_digest",
                nonzero=True,
            ),
        )
        object.__setattr__(
            self,
            "sidecar_generation_digest",
            _fixed32(
                self.sidecar_generation_digest,
                "GDN fixed sidecar_generation_digest",
                nonzero=True,
            ),
        )
        object.__setattr__(
            self,
            "scheduler_coverage_digest",
            _fixed32(
                self.scheduler_coverage_digest,
                "GDN fixed scheduler_coverage_digest",
                nonzero=True,
            ),
        )

    @classmethod
    def from_execution_chunk(
        cls,
        chunk: object,
    ) -> "GDNFixedChunkIdentityV3":
        if not isinstance(chunk, ExecutionChunkCommitmentV3):
            raise ProofV3Error("GDN fixed execution chunk has an unexpected type")
        if (
            chunk.sidecar_generation_digest is None
            or chunk.scheduler_coverage_digest is None
        ):
            raise ProofV3Error("GDN fixed execution chunk must be sidecar-bound")
        return cls(
            phase=chunk.phase,
            chunk_index=chunk.chunk_index,
            logical_token_start=chunk.logical_token_start,
            token_count=chunk.token_count,
            cache_lease_digest=chunk.cache_lease_digest,
            sidecar_generation_digest=chunk.sidecar_generation_digest,
            scheduler_coverage_digest=chunk.scheduler_coverage_digest,
        )

    def canonical_bytes(self) -> bytes:
        # The reference currently admits only decode chunks, but retain the
        # phase discriminator in the canonical form so a later ABI extension
        # cannot reinterpret a decode identity as another chunk phase.
        return (
            struct.pack(
                "<BIII",
                _GDN_FIXED_DECODE_PHASE_CODE,
                self.chunk_index,
                self.logical_token_start,
                self.token_count,
            )
            + self.cache_lease_digest
            + self.sidecar_generation_digest
            + self.scheduler_coverage_digest
        )


@dataclass(frozen=True, slots=True)
class GDNFixedRowWitnessV3:
    """Every exact arithmetic value for one scalar GDN recurrence row."""

    logical_token_index: int
    linear_input: int
    qkvz: tuple[int, int, int, int]
    ba: tuple[int, int]
    convolution_before: tuple[int, int, int]
    convolution_pre_activation: tuple[int, int, int]
    convolution_post_activation: tuple[int, int, int]
    convolution_after: tuple[int, int, int]
    q_invsqrt_lookup_index: int
    q_invsqrt: int
    q_normalized: int
    k_invsqrt_lookup_index: int
    k_invsqrt: int
    k_normalized: int
    softplus_lookup_index: int
    softplus: int
    a_log_exp_lookup_index: int
    a_log_exp: int
    decay_lookup_index: int
    decay: int
    beta_lookup_index: int
    beta: int
    recurrent_before: int
    recurrent_decayed: int
    recurrent_dot: int
    recurrent_delta: int
    recurrent_update: int
    recurrent_after: int
    core_output: int
    output_invsqrt_lookup_index: int
    output_invsqrt: int
    output_normalized: int
    z_silu_lookup_index: int
    z_silu: int
    output_projection_input: int
    output_projection_output: int

    def __post_init__(self) -> None:
        token_index = _u64(self.logical_token_index, "GDN row logical_token_index")
        _fixed(token_index, "GDN row logical_token_index")
        object.__setattr__(self, "logical_token_index", token_index)
        for name, length in (
            ("qkvz", 4),
            ("ba", 2),
            ("convolution_before", 3),
            ("convolution_pre_activation", 3),
            ("convolution_post_activation", 3),
            ("convolution_after", 3),
        ):
            object.__setattr__(
                self,
                name,
                _fixed_vector(
                    getattr(self, name), length=length, name=f"GDN row {name}"
                ),
            )
        for name in (
            "linear_input",
            "q_invsqrt",
            "q_normalized",
            "k_invsqrt",
            "k_normalized",
            "softplus",
            "a_log_exp",
            "decay",
            "beta",
            "recurrent_before",
            "recurrent_decayed",
            "recurrent_dot",
            "recurrent_delta",
            "recurrent_update",
            "recurrent_after",
            "core_output",
            "output_invsqrt",
            "output_normalized",
            "z_silu",
            "output_projection_input",
            "output_projection_output",
        ):
            object.__setattr__(
                self, name, _fixed(getattr(self, name), f"GDN row {name}")
            )
        for name in (
            "q_invsqrt_lookup_index",
            "k_invsqrt_lookup_index",
            "softplus_lookup_index",
            "a_log_exp_lookup_index",
            "decay_lookup_index",
            "beta_lookup_index",
            "output_invsqrt_lookup_index",
            "z_silu_lookup_index",
        ):
            index = _fixed(getattr(self, name), f"GDN row {name}")
            if (
                not GDN_FIXED_LOOKUP_INDEX_MIN_V3
                <= index
                <= GDN_FIXED_LOOKUP_INDEX_MAX_V3
            ):
                raise ProofV3Error(f"GDN row {name} is outside the lookup ABI range")
            object.__setattr__(self, name, index)

    def trace_row(self) -> tuple[int, ...]:
        """Return the exact fixed-width transition-table row committed pre-nonce."""

        return (
            self.logical_token_index,
            self.linear_input,
            *self.qkvz,
            *self.ba,
            *self.convolution_before,
            *self.convolution_pre_activation,
            *self.convolution_post_activation,
            *self.convolution_after,
            self.q_invsqrt_lookup_index,
            self.q_invsqrt,
            self.q_normalized,
            self.k_invsqrt_lookup_index,
            self.k_invsqrt,
            self.k_normalized,
            self.softplus_lookup_index,
            self.softplus,
            self.a_log_exp_lookup_index,
            self.a_log_exp,
            self.decay_lookup_index,
            self.decay,
            self.beta_lookup_index,
            self.beta,
            self.recurrent_before,
            self.recurrent_decayed,
            self.recurrent_dot,
            self.recurrent_delta,
            self.recurrent_update,
            self.recurrent_after,
            self.core_output,
            self.output_invsqrt_lookup_index,
            self.output_invsqrt,
            self.output_normalized,
            self.z_silu_lookup_index,
            self.z_silu,
            self.output_projection_input,
            self.output_projection_output,
        )


@dataclass(frozen=True, slots=True)
class GDNFixedExecutionTraceV3:
    """The deterministic two-row reference execution and its final state."""

    rows: tuple[GDNFixedRowWitnessV3, GDNFixedRowWitnessV3]
    exit_state: GDNFixedStateV3

    def __post_init__(self) -> None:
        rows = _materialize_exact(
            self.rows,
            length=GDN_FIXED_ROWS_V3,
            name="GDN fixed execution trace",
        )
        if len(rows) != GDN_FIXED_ROWS_V3 or not all(
            isinstance(row, GDNFixedRowWitnessV3) for row in rows
        ):
            raise ProofV3Error("GDN fixed execution trace must contain two typed rows")
        if not isinstance(self.exit_state, GDNFixedStateV3):
            raise ProofV3Error("GDN fixed execution trace exit state is malformed")
        if rows[1].logical_token_index != rows[0].logical_token_index + 1:
            raise ProofV3Error("GDN fixed execution rows are not contiguous")
        if (
            rows[1].convolution_before != rows[0].convolution_after
            or rows[1].recurrent_before != rows[0].recurrent_after
        ):
            raise ProofV3Error("GDN fixed execution rows have discontinuous state")
        if self.exit_state.as_row() != (
            *rows[-1].convolution_after,
            rows[-1].recurrent_after,
        ):
            raise ProofV3Error(
                "GDN fixed execution exit state does not match final row"
            )
        object.__setattr__(self, "rows", (rows[0], rows[1]))


@dataclass(frozen=True, slots=True)
class GDNFixedTableRootsV3:
    """All deterministic roots frozen for one exact two-row reference trace."""

    entry_state_root: bytes
    linear_input_root: bytes
    qkvz_root: bytes
    ba_root: bytes
    transition_trace_root: bytes
    output_projection_input_root: bytes
    output_projection_output_root: bytes
    exit_state_root: bytes
    cache_state_root: bytes
    transition_table_root: bytes

    def __post_init__(self) -> None:
        for name in (
            "entry_state_root",
            "linear_input_root",
            "qkvz_root",
            "ba_root",
            "transition_trace_root",
            "output_projection_input_root",
            "output_projection_output_root",
            "exit_state_root",
            "cache_state_root",
            "transition_table_root",
        ):
            object.__setattr__(
                self, name, _fixed32(getattr(self, name), name, nonzero=True)
            )

    def canonical_bytes(self) -> bytes:
        return b"".join(
            getattr(self, name)
            for name in (
                "entry_state_root",
                "linear_input_root",
                "qkvz_root",
                "ba_root",
                "transition_trace_root",
                "output_projection_input_root",
                "output_projection_output_root",
                "exit_state_root",
                "cache_state_root",
                "transition_table_root",
            )
        )


@dataclass(frozen=True, slots=True)
class GDNFixedStatementV3:
    """Verifier-owned pre-nonce statement for one exact GDN reference trace."""

    commitment_envelope_digest: bytes
    execution_chunk_digest: bytes
    chunk_identity: GDNFixedChunkIdentityV3
    static_parameter_digest: bytes
    layer_index: int
    transition_node_id: str
    table_roots: GDNFixedTableRootsV3
    abi_id: str = GDN_FIXED_REFERENCE_ABI_V3

    def __post_init__(self) -> None:
        if self.abi_id != GDN_FIXED_REFERENCE_ABI_V3:
            raise ProofV3Error("GDN fixed statement ABI is unsupported")
        object.__setattr__(
            self,
            "commitment_envelope_digest",
            _fixed32(
                self.commitment_envelope_digest,
                "GDN fixed commitment_envelope_digest",
                nonzero=True,
            ),
        )
        object.__setattr__(
            self,
            "execution_chunk_digest",
            _fixed32(
                self.execution_chunk_digest,
                "GDN fixed execution_chunk_digest",
                nonzero=True,
            ),
        )
        if not isinstance(self.chunk_identity, GDNFixedChunkIdentityV3):
            raise ProofV3Error("GDN fixed statement chunk identity is malformed")
        object.__setattr__(
            self,
            "static_parameter_digest",
            _fixed32(
                self.static_parameter_digest,
                "GDN fixed static_parameter_digest",
                nonzero=True,
            ),
        )
        object.__setattr__(
            self, "layer_index", _u32(self.layer_index, "GDN layer_index")
        )
        object.__setattr__(
            self,
            "transition_node_id",
            _identifier(self.transition_node_id, "GDN transition_node_id"),
        )
        if not isinstance(self.table_roots, GDNFixedTableRootsV3):
            raise ProofV3Error("GDN fixed statement table roots are malformed")

    def canonical_bytes(self) -> bytes:
        abi = self.abi_id.encode("ascii")
        node_id = self.transition_node_id.encode("ascii")
        return (
            struct.pack("<B", len(abi))
            + abi
            + self.commitment_envelope_digest
            + self.execution_chunk_digest
            + self.chunk_identity.canonical_bytes()
            + self.static_parameter_digest
            + struct.pack("<IB", self.layer_index, len(node_id))
            + node_id
            + self.table_roots.canonical_bytes()
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_STATEMENT_DOMAIN + self.canonical_bytes()).digest()


def _root_context(
    *,
    envelope: ProofV3CommitmentEnvelope,
    chunk_identity: GDNFixedChunkIdentityV3,
    static_parameters: GDNFixedStaticParametersV3,
    layer_index: int,
    transition_node_id: str,
) -> bytes:
    if not isinstance(envelope, ProofV3CommitmentEnvelope):
        raise ProofV3Error("GDN fixed envelope has an unexpected type")
    if envelope.cache_lease_digest != chunk_identity.cache_lease_digest:
        raise ProofV3Error("GDN fixed chunk cache lease does not match the envelope")
    _require_gdn_fixed_decode_range(
        envelope=envelope,
        chunk_identity=chunk_identity,
    )
    node_id = _identifier(transition_node_id, "GDN transition_node_id").encode("ascii")
    abi_id = GDN_FIXED_REFERENCE_ABI_V3.encode("ascii")
    return hashlib.sha256(
        _ROOT_CONTEXT_DOMAIN
        + struct.pack("<B", len(abi_id))
        + abi_id
        + envelope.digest()
        + chunk_identity.canonical_bytes()
        + static_parameters.digest()
        + struct.pack("<IB", _u32(layer_index, "GDN layer_index"), len(node_id))
        + node_id
    ).digest()


def _require_gdn_fixed_decode_range(
    *,
    envelope: ProofV3CommitmentEnvelope,
    chunk_identity: GDNFixedChunkIdentityV3,
) -> None:
    """Require this exact decode slice to fit the observed envelope range.

    This reference does not establish profile-derived complete decode coverage,
    but it must not authenticate a decode trace outside the validator-observed
    output interval or under an envelope that observed no decode tokens.
    """

    decode_start = envelope.context_token_count
    decode_end = decode_start + envelope.decode_token_count
    chunk_start = chunk_identity.logical_token_start
    chunk_end = chunk_start + chunk_identity.token_count
    if chunk_start < decode_start or chunk_end > decode_end:
        raise ProofV3Error(
            "GDN fixed decode chunk is outside the commitment envelope range"
        )


def _role_binding(root_context: bytes, role: bytes) -> bytes:
    return hashlib.sha256(
        _ROOT_ROLE_DOMAIN + root_context + struct.pack("<B", len(role)) + role
    ).digest()


def _merkle_root(
    *,
    root_context: bytes,
    role: bytes,
    rows: tuple[tuple[int, ...], ...],
) -> bytes:
    field_rows = tuple(
        tuple(gdn_fixed_to_goldilocks_v3(value) for value in row) for row in rows
    )
    return GoldilocksMerkleTreeReference.from_rows(
        field_rows,
        binding_digest=_role_binding(root_context, role),
    ).commitment


def _trace_tables(
    rows: tuple[GDNFixedRowWitnessV3, ...],
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    return (
        tuple((row.linear_input,) for row in rows),
        tuple(row.qkvz for row in rows),
        tuple(row.ba for row in rows),
        tuple((row.output_projection_input,) for row in rows),
        tuple((row.output_projection_output,) for row in rows),
    )


def derive_gdn_fixed_table_roots_v3(
    *,
    envelope: ProofV3CommitmentEnvelope,
    chunk_identity: GDNFixedChunkIdentityV3,
    static_parameters: GDNFixedStaticParametersV3,
    layer_index: int,
    transition_node_id: str,
    entry_state: GDNFixedStateV3,
    exit_state: GDNFixedStateV3,
    linear_inputs: object,
    qkvz_rows: object,
    ba_rows: object,
    trace_rows: object,
    output_projection_inputs: object,
    output_projection_outputs: object,
) -> GDNFixedTableRootsV3:
    """Derive all role-bound roots without accepting an opaque root claim."""

    if not isinstance(chunk_identity, GDNFixedChunkIdentityV3):
        raise ProofV3Error("GDN fixed chunk identity has an unexpected type")
    if not isinstance(static_parameters, GDNFixedStaticParametersV3):
        raise ProofV3Error("GDN fixed static parameters have an unexpected type")
    if not isinstance(entry_state, GDNFixedStateV3) or not isinstance(
        exit_state, GDNFixedStateV3
    ):
        raise ProofV3Error("GDN fixed state roots require typed states")
    linear = _fixed_rows(
        linear_inputs,
        row_count=GDN_FIXED_ROWS_V3,
        width=1,
        name="GDN fixed linear inputs",
    )
    qkvz = _fixed_rows(
        qkvz_rows,
        row_count=GDN_FIXED_ROWS_V3,
        width=4,
        name="GDN fixed QKVZ rows",
    )
    ba = _fixed_rows(
        ba_rows,
        row_count=GDN_FIXED_ROWS_V3,
        width=2,
        name="GDN fixed BA rows",
    )
    outputs_in = _fixed_rows(
        output_projection_inputs,
        row_count=GDN_FIXED_ROWS_V3,
        width=1,
        name="GDN fixed output-projection inputs",
    )
    outputs_out = _fixed_rows(
        output_projection_outputs,
        row_count=GDN_FIXED_ROWS_V3,
        width=1,
        name="GDN fixed output-projection outputs",
    )
    trace = _materialize_exact(
        trace_rows,
        length=GDN_FIXED_ROWS_V3,
        name="GDN fixed transition trace",
    )
    if len(trace) != GDN_FIXED_ROWS_V3 or not all(
        isinstance(row, GDNFixedRowWitnessV3) for row in trace
    ):
        raise ProofV3Error("GDN fixed transition trace must contain two typed rows")
    execution_trace = GDNFixedExecutionTraceV3(
        rows=(trace[0], trace[1]),
        exit_state=exit_state,
    )
    if (
        execution_trace.rows[0].logical_token_index
        != chunk_identity.logical_token_start
        or execution_trace.rows[0].convolution_before != entry_state.convolution_history
        or execution_trace.rows[0].recurrent_before != entry_state.recurrent_state
    ):
        raise ProofV3Error("GDN fixed transition trace does not start at entry state")
    context = _root_context(
        envelope=envelope,
        chunk_identity=chunk_identity,
        static_parameters=static_parameters,
        layer_index=layer_index,
        transition_node_id=transition_node_id,
    )
    entry_root = _merkle_root(
        root_context=context,
        role=b"entry-state",
        rows=(entry_state.as_row(),),
    )
    linear_root = _merkle_root(
        root_context=context,
        role=b"linear-input",
        rows=linear,
    )
    qkvz_root = _merkle_root(root_context=context, role=b"qkvz-y", rows=qkvz)
    ba_root = _merkle_root(root_context=context, role=b"ba-y", rows=ba)
    trace_root = _merkle_root(
        root_context=context,
        role=b"transition-trace",
        rows=tuple(row.trace_row() for row in execution_trace.rows),
    )
    output_input_root = _merkle_root(
        root_context=context,
        role=b"out-projection-input",
        rows=outputs_in,
    )
    output_output_root = _merkle_root(
        root_context=context,
        role=b"out-projection-output",
        rows=outputs_out,
    )
    exit_root = _merkle_root(
        root_context=context,
        role=b"exit-state",
        rows=(exit_state.as_row(),),
    )
    cache_root = _merkle_root(
        root_context=context,
        role=b"cache-state-pair",
        rows=(entry_state.as_row(), exit_state.as_row()),
    )
    transition_root = hashlib.sha256(
        _ROOT_INVENTORY_DOMAIN
        + context
        + entry_root
        + linear_root
        + qkvz_root
        + ba_root
        + trace_root
        + output_input_root
        + output_output_root
        + exit_root
        + cache_root
    ).digest()
    return GDNFixedTableRootsV3(
        entry_state_root=entry_root,
        linear_input_root=linear_root,
        qkvz_root=qkvz_root,
        ba_root=ba_root,
        transition_trace_root=trace_root,
        output_projection_input_root=output_input_root,
        output_projection_output_root=output_output_root,
        exit_state_root=exit_root,
        cache_state_root=cache_root,
        transition_table_root=transition_root,
    )


def _chunk_matches_gdn_fixed_roots(
    *,
    chunk: ExecutionChunkCommitmentV3,
    roots: GDNFixedTableRootsV3,
) -> bool:
    """Return whether the generic receipt has this reference ABI's root map.

    The generic chunk object deliberately leaves root-role semantics to a
    qualified adapter.  This isolated reference defines all six mappings
    explicitly instead of accepting opaque generic slots: its linear slot is
    the two-row linear-input table and its bridge slot is the output-projection
    output table.
    """

    return (
        chunk.entry_state_root == roots.entry_state_root
        and chunk.exit_state_root == roots.exit_state_root
        and chunk.linear_table_root == roots.linear_input_root
        and chunk.bridge_table_root == roots.output_projection_output_root
        and chunk.transition_table_root == roots.transition_table_root
        and chunk.cache_table_root == roots.cache_state_root
    )


@dataclass(frozen=True, slots=True, init=False)
class FrozenGDNFixedPrecommitV3:
    """Verifier-retained, fully rooted reference data frozen before a nonce."""

    statement: GDNFixedStatementV3
    envelope: ProofV3CommitmentEnvelope
    chunk: ExecutionChunkCommitmentV3
    static_parameters: GDNFixedStaticParametersV3
    entry_state: GDNFixedStateV3
    exit_state: GDNFixedStateV3
    linear_inputs: tuple[tuple[int, ...], tuple[int, ...]]
    qkvz_rows: tuple[tuple[int, ...], tuple[int, ...]]
    ba_rows: tuple[tuple[int, ...], tuple[int, ...]]
    output_projection_inputs: tuple[tuple[int, ...], tuple[int, ...]]
    output_projection_outputs: tuple[tuple[int, ...], tuple[int, ...]]
    _factory_token: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ProofV3Error("GDN fixed precommits must be frozen through the factory")

    @classmethod
    def _construct(
        cls,
        *,
        statement: GDNFixedStatementV3,
        envelope: ProofV3CommitmentEnvelope,
        chunk: ExecutionChunkCommitmentV3,
        static_parameters: GDNFixedStaticParametersV3,
        entry_state: GDNFixedStateV3,
        exit_state: GDNFixedStateV3,
        linear_inputs: tuple[tuple[int, ...], tuple[int, ...]],
        qkvz_rows: tuple[tuple[int, ...], tuple[int, ...]],
        ba_rows: tuple[tuple[int, ...], tuple[int, ...]],
        output_projection_inputs: tuple[tuple[int, ...], tuple[int, ...]],
        output_projection_outputs: tuple[tuple[int, ...], tuple[int, ...]],
        _factory_token: object | None = None,
    ) -> "FrozenGDNFixedPrecommitV3":
        if _factory_token is not _PRECOMMIT_FACTORY_TOKEN:
            raise ProofV3Error(
                "GDN fixed precommits must be frozen through the factory"
            )
        result = object.__new__(cls)
        for name, value in (
            ("statement", statement),
            ("envelope", envelope),
            ("chunk", chunk),
            ("static_parameters", static_parameters),
            ("entry_state", entry_state),
            ("exit_state", exit_state),
            ("linear_inputs", linear_inputs),
            ("qkvz_rows", qkvz_rows),
            ("ba_rows", ba_rows),
            ("output_projection_inputs", output_projection_inputs),
            ("output_projection_outputs", output_projection_outputs),
            ("_factory_token", _PRECOMMIT_FACTORY_TOKEN),
        ):
            object.__setattr__(result, name, value)
        result.require_precommit_provenance()
        return result

    def require_precommit_provenance(self) -> None:
        """Recompute every retained root and reject a detached frozen record."""

        if getattr(self, "_factory_token", None) is not _PRECOMMIT_FACTORY_TOKEN:
            raise ProofV3Error("GDN fixed precommit lacks factory provenance")
        if not isinstance(self.statement, GDNFixedStatementV3):
            raise ProofV3Error("GDN fixed precommit statement is malformed")
        if not isinstance(self.envelope, ProofV3CommitmentEnvelope) or not isinstance(
            self.chunk, ExecutionChunkCommitmentV3
        ):
            raise ProofV3Error("GDN fixed precommit context is malformed")
        if not isinstance(self.static_parameters, GDNFixedStaticParametersV3):
            raise ProofV3Error("GDN fixed precommit static parameters are malformed")
        if not isinstance(self.entry_state, GDNFixedStateV3) or not isinstance(
            self.exit_state, GDNFixedStateV3
        ):
            raise ProofV3Error("GDN fixed precommit states are malformed")
        identity = GDNFixedChunkIdentityV3.from_execution_chunk(self.chunk)
        statement = self.statement
        if (
            statement.commitment_envelope_digest != self.envelope.digest()
            or statement.execution_chunk_digest != self.chunk.digest()
            or statement.chunk_identity != identity
            or statement.static_parameter_digest != self.static_parameters.digest()
        ):
            raise ProofV3Error("GDN fixed precommit does not match its statement")
        roots = derive_gdn_fixed_table_roots_v3(
            envelope=self.envelope,
            chunk_identity=identity,
            static_parameters=self.static_parameters,
            layer_index=statement.layer_index,
            transition_node_id=statement.transition_node_id,
            entry_state=self.entry_state,
            exit_state=self.exit_state,
            linear_inputs=self.linear_inputs,
            qkvz_rows=self.qkvz_rows,
            ba_rows=self.ba_rows,
            trace_rows=_trace_rows_from_captured(
                static_parameters=self.static_parameters,
                entry_state=self.entry_state,
                linear_inputs=self.linear_inputs,
                row_start=identity.logical_token_start,
            ).rows,
            output_projection_inputs=self.output_projection_inputs,
            output_projection_outputs=self.output_projection_outputs,
        )
        if roots != statement.table_roots:
            raise ProofV3Error("GDN fixed precommit roots do not match retained tables")
        if not _chunk_matches_gdn_fixed_roots(chunk=self.chunk, roots=roots):
            raise ProofV3Error(
                "GDN fixed execution chunk has opaque or mismatched roots"
            )


def _execute_row(
    *,
    static_parameters: GDNFixedStaticParametersV3,
    state_before: GDNFixedStateV3,
    linear_input: int,
    logical_token_index: int,
) -> tuple[GDNFixedRowWitnessV3, GDNFixedStateV3]:
    qkvz = tuple(
        fixed_mul_q16_v3(linear_input, weight)
        for weight in static_parameters.qkvz_weights
    )
    ba = tuple(
        fixed_mul_q16_v3(linear_input, weight)
        for weight in static_parameters.ba_weights
    )
    convolution_pre = tuple(
        _fixed_conv2(
            previous=state_before.convolution_history[channel],
            current=qkvz[channel],
            previous_weight=static_parameters.conv_weights[channel][0],
            current_weight=static_parameters.conv_weights[channel][1],
        )
        for channel in range(3)
    )
    convolution_post_values = tuple(
        static_parameters.lookup_q16("silu", value) for value in convolution_pre
    )
    convolution_post = tuple(value for _index, value in convolution_post_values)
    q_norm_argument = _fixed_add(
        fixed_mul_q16_v3(convolution_post[0], convolution_post[0]),
        static_parameters.q_norm_epsilon,
        "GDN q normalization argument",
    )
    q_norm_index, q_invsqrt = static_parameters.lookup_q16("invsqrt", q_norm_argument)
    q_normalized = fixed_mul_q16_v3(convolution_post[0], q_invsqrt)
    k_norm_argument = _fixed_add(
        fixed_mul_q16_v3(convolution_post[1], convolution_post[1]),
        static_parameters.q_norm_epsilon,
        "GDN k normalization argument",
    )
    k_norm_index, k_invsqrt = static_parameters.lookup_q16("invsqrt", k_norm_argument)
    k_normalized = fixed_mul_q16_v3(convolution_post[1], k_invsqrt)
    softplus_argument = _fixed_add(
        ba[1], static_parameters.dt_bias, "GDN softplus argument"
    )
    softplus_index, softplus = static_parameters.lookup_q16(
        "softplus", softplus_argument
    )
    a_log_exp_index, a_log_exp = static_parameters.lookup_q16(
        "exp", static_parameters.a_log
    )
    decay_argument = _fixed_neg(
        fixed_mul_q16_v3(a_log_exp, softplus), "GDN decay argument"
    )
    decay_index, decay = static_parameters.lookup_q16("decay", decay_argument)
    beta_index, beta = static_parameters.lookup_q16("sigmoid", ba[0])
    recurrent_decayed = fixed_mul_q16_v3(state_before.recurrent_state, decay)
    recurrent_dot = fixed_mul_q16_v3(recurrent_decayed, k_normalized)
    recurrent_delta = _fixed_sub(
        convolution_post[2], recurrent_dot, "GDN recurrent delta"
    )
    recurrent_update = fixed_mul_q16_v3(
        fixed_mul_q16_v3(beta, recurrent_delta), k_normalized
    )
    recurrent_after = _fixed_add(
        recurrent_decayed, recurrent_update, "GDN recurrent state after"
    )
    core_output = fixed_mul_q16_v3(recurrent_after, q_normalized)
    output_norm_argument = _fixed_add(
        fixed_mul_q16_v3(core_output, core_output),
        static_parameters.output_norm_epsilon,
        "GDN output normalization argument",
    )
    output_norm_index, output_invsqrt = static_parameters.lookup_q16(
        "invsqrt", output_norm_argument
    )
    output_normalized = fixed_mul_q16_v3(core_output, output_invsqrt)
    z_silu_index, z_silu = static_parameters.lookup_q16("silu", qkvz[3])
    output_projection_input = fixed_mul_q16_v3(
        fixed_mul_q16_v3(output_normalized, static_parameters.norm_weight), z_silu
    )
    output_projection_output = fixed_mul_q16_v3(
        output_projection_input, static_parameters.output_weight
    )
    state_after = GDNFixedStateV3(
        convolution_history=(qkvz[0], qkvz[1], qkvz[2]),
        recurrent_state=recurrent_after,
    )
    return (
        GDNFixedRowWitnessV3(
            logical_token_index=logical_token_index,
            linear_input=linear_input,
            qkvz=qkvz,
            ba=ba,
            convolution_before=state_before.convolution_history,
            convolution_pre_activation=convolution_pre,
            convolution_post_activation=convolution_post,
            convolution_after=state_after.convolution_history,
            q_invsqrt_lookup_index=q_norm_index,
            q_invsqrt=q_invsqrt,
            q_normalized=q_normalized,
            k_invsqrt_lookup_index=k_norm_index,
            k_invsqrt=k_invsqrt,
            k_normalized=k_normalized,
            softplus_lookup_index=softplus_index,
            softplus=softplus,
            a_log_exp_lookup_index=a_log_exp_index,
            a_log_exp=a_log_exp,
            decay_lookup_index=decay_index,
            decay=decay,
            beta_lookup_index=beta_index,
            beta=beta,
            recurrent_before=state_before.recurrent_state,
            recurrent_decayed=recurrent_decayed,
            recurrent_dot=recurrent_dot,
            recurrent_delta=recurrent_delta,
            recurrent_update=recurrent_update,
            recurrent_after=recurrent_after,
            core_output=core_output,
            output_invsqrt_lookup_index=output_norm_index,
            output_invsqrt=output_invsqrt,
            output_normalized=output_normalized,
            z_silu_lookup_index=z_silu_index,
            z_silu=z_silu,
            output_projection_input=output_projection_input,
            output_projection_output=output_projection_output,
        ),
        state_after,
    )


def _trace_rows_from_captured(
    *,
    static_parameters: GDNFixedStaticParametersV3,
    entry_state: GDNFixedStateV3,
    linear_inputs: object,
    row_start: int,
) -> GDNFixedExecutionTraceV3:
    inputs = _fixed_rows(
        linear_inputs,
        row_count=GDN_FIXED_ROWS_V3,
        width=1,
        name="GDN fixed linear inputs",
    )
    start = _u32(row_start, "GDN fixed row_start")
    state = entry_state
    rows: list[GDNFixedRowWitnessV3] = []
    for offset, input_row in enumerate(inputs):
        row, state = _execute_row(
            static_parameters=static_parameters,
            state_before=state,
            linear_input=input_row[0],
            logical_token_index=start + offset,
        )
        rows.append(row)
    return GDNFixedExecutionTraceV3(rows=(rows[0], rows[1]), exit_state=state)


def execute_gdn_fixed_reference_v3(
    *,
    static_parameters: GDNFixedStaticParametersV3,
    entry_state: GDNFixedStateV3,
    linear_inputs: object,
    row_start: int,
) -> GDNFixedExecutionTraceV3:
    """Execute the exact reference recurrence without any runtime integration."""

    if not isinstance(static_parameters, GDNFixedStaticParametersV3):
        raise ProofV3Error("GDN fixed static parameters have an unexpected type")
    if not isinstance(entry_state, GDNFixedStateV3):
        raise ProofV3Error("GDN fixed entry state has an unexpected type")
    return _trace_rows_from_captured(
        static_parameters=static_parameters,
        entry_state=entry_state,
        linear_inputs=linear_inputs,
        row_start=row_start,
    )


def freeze_gdn_fixed_precommit_v3(
    *,
    envelope: ProofV3CommitmentEnvelope,
    chunk: ExecutionChunkCommitmentV3,
    static_parameters: GDNFixedStaticParametersV3,
    layer_index: int,
    transition_node_id: str,
    entry_state: GDNFixedStateV3,
    exit_state: GDNFixedStateV3,
    linear_inputs: object,
    qkvz_rows: object,
    ba_rows: object,
    output_projection_inputs: object,
    output_projection_outputs: object,
) -> FrozenGDNFixedPrecommitV3:
    """Freeze one fully captured two-row reference statement before a nonce.

    This helper does not accept caller-provided roots.  It derives all table
    roots from retained canonical values, then requires the supplied generic
    chunk receipt to contain the matching entry/exit/transition/cache roots.
    """

    if not isinstance(envelope, ProofV3CommitmentEnvelope):
        raise ProofV3Error("GDN fixed envelope has an unexpected type")
    if not isinstance(chunk, ExecutionChunkCommitmentV3):
        raise ProofV3Error("GDN fixed execution chunk has an unexpected type")
    if not isinstance(static_parameters, GDNFixedStaticParametersV3):
        raise ProofV3Error("GDN fixed static parameters have an unexpected type")
    if not isinstance(entry_state, GDNFixedStateV3) or not isinstance(
        exit_state, GDNFixedStateV3
    ):
        raise ProofV3Error("GDN fixed states have an unexpected type")
    identity = GDNFixedChunkIdentityV3.from_execution_chunk(chunk)
    if envelope.cache_lease_digest != identity.cache_lease_digest:
        raise ProofV3Error("GDN fixed chunk cache lease does not match the envelope")
    linear = _fixed_rows(
        linear_inputs,
        row_count=GDN_FIXED_ROWS_V3,
        width=1,
        name="GDN fixed linear inputs",
    )
    qkvz = _fixed_rows(
        qkvz_rows,
        row_count=GDN_FIXED_ROWS_V3,
        width=4,
        name="GDN fixed QKVZ rows",
    )
    ba = _fixed_rows(
        ba_rows,
        row_count=GDN_FIXED_ROWS_V3,
        width=2,
        name="GDN fixed BA rows",
    )
    output_inputs = _fixed_rows(
        output_projection_inputs,
        row_count=GDN_FIXED_ROWS_V3,
        width=1,
        name="GDN fixed output-projection inputs",
    )
    output_outputs = _fixed_rows(
        output_projection_outputs,
        row_count=GDN_FIXED_ROWS_V3,
        width=1,
        name="GDN fixed output-projection outputs",
    )
    trace = _trace_rows_from_captured(
        static_parameters=static_parameters,
        entry_state=entry_state,
        linear_inputs=linear,
        row_start=identity.logical_token_start,
    )
    roots = derive_gdn_fixed_table_roots_v3(
        envelope=envelope,
        chunk_identity=identity,
        static_parameters=static_parameters,
        layer_index=layer_index,
        transition_node_id=transition_node_id,
        entry_state=entry_state,
        exit_state=exit_state,
        linear_inputs=linear,
        qkvz_rows=qkvz,
        ba_rows=ba,
        trace_rows=trace.rows,
        output_projection_inputs=output_inputs,
        output_projection_outputs=output_outputs,
    )
    if not _chunk_matches_gdn_fixed_roots(chunk=chunk, roots=roots):
        raise ProofV3Error("GDN fixed execution chunk has opaque or mismatched roots")
    statement = GDNFixedStatementV3(
        commitment_envelope_digest=envelope.digest(),
        execution_chunk_digest=chunk.digest(),
        chunk_identity=identity,
        static_parameter_digest=static_parameters.digest(),
        layer_index=layer_index,
        transition_node_id=transition_node_id,
        table_roots=roots,
    )
    return FrozenGDNFixedPrecommitV3._construct(
        statement=statement,
        envelope=envelope,
        chunk=chunk,
        static_parameters=static_parameters,
        entry_state=entry_state,
        exit_state=exit_state,
        linear_inputs=(linear[0], linear[1]),
        qkvz_rows=(qkvz[0], qkvz[1]),
        ba_rows=(ba[0], ba[1]),
        output_projection_inputs=(output_inputs[0], output_inputs[1]),
        output_projection_outputs=(output_outputs[0], output_outputs[1]),
        _factory_token=_PRECOMMIT_FACTORY_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class GDNFixedWitnessV3:
    """Complete opened reference witness, never a production wire payload."""

    statement_digest: bytes
    table_roots: GDNFixedTableRootsV3
    rows: tuple[GDNFixedRowWitnessV3, GDNFixedRowWitnessV3]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "statement_digest",
            _fixed32(
                self.statement_digest,
                "GDN fixed witness statement_digest",
                nonzero=True,
            ),
        )
        if not isinstance(self.table_roots, GDNFixedTableRootsV3):
            raise ProofV3Error("GDN fixed witness table roots are malformed")
        rows = _materialize_exact(
            self.rows,
            length=GDN_FIXED_ROWS_V3,
            name="GDN fixed witness rows",
        )
        if len(rows) != GDN_FIXED_ROWS_V3 or not all(
            isinstance(row, GDNFixedRowWitnessV3) for row in rows
        ):
            raise ProofV3Error("GDN fixed witness must contain exactly two typed rows")
        if rows[1].logical_token_index != rows[0].logical_token_index + 1:
            raise ProofV3Error("GDN fixed witness rows are not contiguous")
        object.__setattr__(self, "rows", (rows[0], rows[1]))


def build_gdn_fixed_reference_witness_v3(
    *,
    precommit: FrozenGDNFixedPrecommitV3,
) -> GDNFixedWitnessV3:
    """Build an honest complete witness from one already-frozen reference trace."""

    if not isinstance(precommit, FrozenGDNFixedPrecommitV3):
        raise ProofV3Error("GDN fixed witness builder requires a frozen precommit")
    precommit.require_precommit_provenance()
    trace = _trace_rows_from_captured(
        static_parameters=precommit.static_parameters,
        entry_state=precommit.entry_state,
        linear_inputs=precommit.linear_inputs,
        row_start=precommit.statement.chunk_identity.logical_token_start,
    )
    qkvz, ba, output_inputs, output_outputs = _trace_tables(trace.rows)[1:]
    if (
        qkvz != precommit.qkvz_rows
        or ba != precommit.ba_rows
        or output_inputs != precommit.output_projection_inputs
        or output_outputs != precommit.output_projection_outputs
        or trace.exit_state != precommit.exit_state
    ):
        raise ProofV3Error(
            "GDN fixed captured runtime tables do not satisfy the relation"
        )
    return GDNFixedWitnessV3(
        statement_digest=precommit.statement.digest(),
        table_roots=precommit.statement.table_roots,
        rows=trace.rows,
    )


def _verify_gdn_fixed_reference_unwrapped_v3(
    *,
    precommit: FrozenGDNFixedPrecommitV3,
    witness: GDNFixedWitnessV3,
) -> None:
    if not isinstance(precommit, FrozenGDNFixedPrecommitV3):
        raise ProofV3VerificationError("GDN fixed precommit has an unexpected type")
    if not isinstance(witness, GDNFixedWitnessV3):
        raise ProofV3VerificationError("GDN fixed witness has an unexpected type")
    try:
        precommit.require_precommit_provenance()
    except ProofV3Error as exc:
        raise ProofV3VerificationError("GDN fixed precommit is malformed") from exc
    statement = precommit.statement
    if (
        witness.statement_digest != statement.digest()
        or witness.table_roots != statement.table_roots
    ):
        raise ProofV3VerificationError(
            "GDN fixed witness is stale or bound to another statement"
        )
    expected_start = statement.chunk_identity.logical_token_start
    if tuple(row.logical_token_index for row in witness.rows) != (
        expected_start,
        expected_start + 1,
    ):
        raise ProofV3VerificationError(
            "GDN fixed witness rows are missing or reordered"
        )
    try:
        (
            linear_inputs,
            qkvz_rows,
            ba_rows,
            output_inputs,
            output_outputs,
        ) = _trace_tables(witness.rows)
        claimed_exit = GDNFixedStateV3(
            convolution_history=witness.rows[-1].convolution_after,
            recurrent_state=witness.rows[-1].recurrent_after,
        )
        witness_roots = derive_gdn_fixed_table_roots_v3(
            envelope=precommit.envelope,
            chunk_identity=statement.chunk_identity,
            static_parameters=precommit.static_parameters,
            layer_index=statement.layer_index,
            transition_node_id=statement.transition_node_id,
            entry_state=precommit.entry_state,
            exit_state=claimed_exit,
            linear_inputs=linear_inputs,
            qkvz_rows=qkvz_rows,
            ba_rows=ba_rows,
            trace_rows=witness.rows,
            output_projection_inputs=output_inputs,
            output_projection_outputs=output_outputs,
        )
    except ProofV3Error as exc:
        raise ProofV3VerificationError("GDN fixed witness is malformed") from exc
    if witness_roots != statement.table_roots:
        raise ProofV3VerificationError(
            "GDN fixed witness tables do not match frozen roots"
        )
    try:
        expected_trace = _trace_rows_from_captured(
            static_parameters=precommit.static_parameters,
            entry_state=precommit.entry_state,
            linear_inputs=linear_inputs,
            row_start=expected_start,
        )
    except ProofV3Error as exc:
        raise ProofV3VerificationError(
            "GDN fixed witness arithmetic is malformed"
        ) from exc
    if (
        witness.rows != expected_trace.rows
        or expected_trace.exit_state != precommit.exit_state
        or linear_inputs != precommit.linear_inputs
        or qkvz_rows != precommit.qkvz_rows
        or ba_rows != precommit.ba_rows
        or output_inputs != precommit.output_projection_inputs
        or output_outputs != precommit.output_projection_outputs
    ):
        raise ProofV3VerificationError(
            "GDN fixed witness does not satisfy the exact recurrence"
        )


def verify_gdn_fixed_reference_v3(
    *,
    precommit: FrozenGDNFixedPrecommitV3,
    witness: GDNFixedWitnessV3,
) -> None:
    """Fail closed unless every opened row satisfies the exact frozen relation."""

    if not isinstance(precommit, FrozenGDNFixedPrecommitV3):
        raise ProofV3VerificationError("GDN fixed precommit has an unexpected type")
    if not isinstance(witness, GDNFixedWitnessV3):
        raise ProofV3VerificationError("GDN fixed witness has an unexpected type")
    try:
        _verify_gdn_fixed_reference_unwrapped_v3(
            precommit=precommit,
            witness=witness,
        )
    except ProofV3VerificationError:
        raise
    except Exception as exc:
        # This boundary accepts typed Python objects, not a wire parser yet.
        # A forged slotted instance can still carry an iterator/property that
        # raises an arbitrary ordinary exception.  Normalize every such
        # malformed-object failure to the verifier's sole failure type rather
        # than letting an upstream caller treat it as "not requested."
        raise ProofV3VerificationError("GDN fixed proof data is malformed") from exc


__all__ = [
    "FixedLookupTableV3",
    "FrozenGDNFixedPrecommitV3",
    "GDN_FIXED_CONV_KERNEL_SIZE_V3",
    "GDN_FIXED_ENCODING_ABI_V3",
    "GDN_FIXED_FRACTION_BITS_V3",
    "GDN_FIXED_LOOKUP_ENTRY_COUNT_V3",
    "GDN_FIXED_LOOKUP_INDEX_MAX_V3",
    "GDN_FIXED_LOOKUP_INDEX_MIN_V3",
    "GDN_FIXED_LOOKUP_TABLE_IDS_V3",
    "GDN_FIXED_MAX_VALUE_V3",
    "GDN_FIXED_MIN_VALUE_V3",
    "GDN_FIXED_REFERENCE_ABI_V3",
    "GDN_FIXED_ROWS_V3",
    "GDN_FIXED_SCALE_V3",
    "GDNFixedChunkIdentityV3",
    "GDNFixedExecutionTraceV3",
    "GDNFixedRowWitnessV3",
    "GDNFixedStateV3",
    "GDNFixedStatementV3",
    "GDNFixedStaticParametersV3",
    "GDNFixedTableRootsV3",
    "GDNFixedWitnessV3",
    "build_gdn_fixed_reference_witness_v3",
    "derive_gdn_fixed_table_roots_v3",
    "execute_gdn_fixed_reference_v3",
    "fixed_mul_q16_v3",
    "fixed_rne_divide_v3",
    "freeze_gdn_fixed_precommit_v3",
    "gdn_fixed_to_goldilocks_v3",
    "verify_gdn_fixed_reference_v3",
]
