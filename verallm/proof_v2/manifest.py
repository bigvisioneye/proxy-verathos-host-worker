"""Authenticated static weight commitment manifests for proof protocol v2."""

from __future__ import annotations

import hashlib
import re
import struct
import unicodedata
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Collection, Iterable, Sequence

from verallm.challenge.v2 import MAX_BLOCKS_PER_OPERATION

if TYPE_CHECKING:
    from verallm.chain.types import OnChainModelSpec


PROTOCOL_VERSION = 2
MODEL_LEVEL_LAYER = -1
MAX_MANIFEST_OPERATIONS = 100_000
MAX_EXECUTION_PROFILE_LAYERS = 16_384
WEIGHT_SCALE_BLOCK_COLS = 16
# Protocol safety ceilings.  A model whose quantized proof equation cannot
# track its captured runtime output within these bounds is not eligible for
# this proof profile; an authority signature cannot opt a validator into a
# vacuous runtime-output comparison.
MAX_RUNTIME_ABS_TOLERANCE_Q32 = (1 << 32) // 8
MAX_RUNTIME_REL_TOLERANCE_BPS = 3_500
_DIGEST_DOMAIN = b"VERATHOS/PROOF_V2/STATIC_WEIGHT_COMMITMENT_MANIFEST/SHA256"
_LAYER_BRIDGE_PARAMETER_DOMAIN = (
    b"VERATHOS/PROOF_V2/QWEN_LAYER_BRIDGE_PARAMETERS/SHA256"
)
_LAYER_TRANSITION_PARAMETER_DOMAIN = (
    b"VERATHOS/PROOF_V2/QWEN_LAYER_TRANSITION_PARAMETERS/SHA256"
)
_MODEL_EXECUTION_PARAMETER_DOMAIN = (
    b"VERATHOS/PROOF_V2/QWEN_MODEL_EXECUTION_PARAMETERS/SHA256"
)
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,127}$")
_ZERO_ADDRESS = b"\x00" * 20
_SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
_SECP256K1_HALF_ORDER = _SECP256K1_ORDER // 2


class ManifestFormatError(ValueError):
    """The manifest is not in its canonical form."""


class ManifestContextError(ValueError):
    """The manifest does not match the verifier's expected context."""


class ManifestSignatureError(ValueError):
    """The manifest authority signatures are not acceptable."""


def _frame(value: bytes) -> bytes:
    if len(value) >= 1 << 32:
        raise ManifestFormatError("manifest field exceeds the framing limit")
    return struct.pack(">I", len(value)) + value


def _record(fields: Iterable[tuple[str, bytes]]) -> bytes:
    encoded = bytearray()
    for name, value in fields:
        encoded.extend(_frame(name.encode("ascii")))
        encoded.extend(_frame(value))
    return bytes(encoded)


def _sequence(values: Iterable[bytes]) -> bytes:
    items = tuple(values)
    if len(items) >= 1 << 32:
        raise ManifestFormatError("manifest sequence exceeds the framing limit")
    return struct.pack(">I", len(items)) + b"".join(_frame(item) for item in items)


def _uint(value: int, width: int, field_name: str, *, positive: bool = False) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestFormatError(f"{field_name} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum or value >= 1 << (width * 8):
        qualifier = "positive " if positive else ""
        raise ManifestFormatError(f"{field_name} must be a {qualifier}uint{width * 8}")
    return value.to_bytes(width, "big")


def _layer_int(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestFormatError("operation layer must be an integer")
    if value < MODEL_LEVEL_LAYER or value >= 1 << 63:
        raise ManifestFormatError("operation layer is outside the canonical range")
    return value.to_bytes(8, "big", signed=True)


def _text(value: str, field_name: str, *, max_bytes: int = 4096) -> bytes:
    if not isinstance(value, str) or not value:
        raise ManifestFormatError(f"{field_name} must be a non-empty string")
    if value != unicodedata.normalize("NFC", value):
        raise ManifestFormatError(f"{field_name} must use NFC text")
    if "\x00" in value:
        raise ManifestFormatError(f"{field_name} must not contain NUL")
    encoded = value.encode("utf-8")
    if len(encoded) > max_bytes:
        raise ManifestFormatError(f"{field_name} is too long")
    return encoded


def _identifier(value: str, field_name: str) -> bytes:
    encoded = _text(value, field_name, max_bytes=128)
    if _IDENTIFIER_RE.fullmatch(value) is None:
        raise ManifestFormatError(
            f"{field_name} must contain only lowercase letters, digits, '.', '_', ':', '/', or '-'"
        )
    return encoded


def _fixed_bytes(value: bytes, size: int, field_name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != size:
        raise ManifestFormatError(f"{field_name} must be exactly {size} bytes")
    return value


def _address_bytes(value: str | bytes, field_name: str) -> bytes:
    if isinstance(value, bytes):
        address = value
    elif isinstance(value, str):
        if len(value) != 42 or not value.startswith("0x"):
            raise ManifestFormatError(
                f"{field_name} must be a 0x-prefixed 20-byte address"
            )
        try:
            address = bytes.fromhex(value[2:])
        except ValueError as exc:
            raise ManifestFormatError(f"{field_name} must be hexadecimal") from exc
    else:
        raise ManifestFormatError(
            f"{field_name} must be bytes or a hexadecimal address"
        )
    if len(address) != 20:
        raise ManifestFormatError(f"{field_name} must be exactly 20 bytes")
    return address


def _root(value: bytes, field_name: str) -> bytes:
    return _fixed_bytes(value, 32, field_name)


def _parameter_bytes(value: bytes, field_name: str) -> bytes:
    if not isinstance(value, bytes) or not value or len(value) >= 1 << 32:
        raise ManifestFormatError(f"{field_name} must be non-empty bytes")
    return value


def qwen_layer_bridge_parameter_root_v2(
    *,
    layer: int,
    attention_profile: str,
    input_norm_weight_f16: bytes,
    post_attention_norm_weight_f16: bytes,
    norm_epsilon_q32: int,
) -> bytes:
    """Commit the exact signed parameters used by both layer RMSNorm bridges."""

    encoded = _record(
        (
            ("layer", _layer_int(layer)),
            (
                "attention_profile",
                _identifier(attention_profile, "attention_profile"),
            ),
            (
                "input_norm_weight_f16",
                _parameter_bytes(
                    input_norm_weight_f16,
                    "input_norm_weight_f16",
                ),
            ),
            (
                "post_attention_norm_weight_f16",
                _parameter_bytes(
                    post_attention_norm_weight_f16,
                    "post_attention_norm_weight_f16",
                ),
            ),
            (
                "norm_epsilon_q32",
                _uint(norm_epsilon_q32, 8, "norm_epsilon_q32", positive=True),
            ),
        )
    )
    return hashlib.sha256(
        _frame(_LAYER_BRIDGE_PARAMETER_DOMAIN) + _frame(encoded)
    ).digest()


def qwen_layer_transition_parameter_root_v2(
    *,
    layer: int,
    transition_profile: str,
    transition_parameters: bytes,
) -> bytes:
    """Commit a verifier-supported nonlinear transition profile and inputs."""

    encoded = _record(
        (
            ("layer", _layer_int(layer)),
            (
                "transition_profile",
                _identifier(transition_profile, "transition_profile"),
            ),
            (
                "transition_parameters",
                _parameter_bytes(
                    transition_parameters,
                    "transition_parameters",
                ),
            ),
        )
    )
    return hashlib.sha256(
        _frame(_LAYER_TRANSITION_PARAMETER_DOMAIN) + _frame(encoded)
    ).digest()


def qwen_model_execution_parameter_root_v2(
    *,
    embedding_scale_q32: int,
    final_norm_weight_f16: bytes,
    final_norm_epsilon_q32: int,
    audit_policy: "ExecutionAuditPolicyV2 | None" = None,
) -> bytes:
    """Commit the embedding dequantization scale and final RMSNorm parameters."""

    fields = [
        (
            "embedding_scale_q32",
            _uint(
                embedding_scale_q32,
                8,
                "embedding_scale_q32",
                positive=True,
            ),
        ),
        (
            "final_norm_weight_f16",
            _parameter_bytes(
                final_norm_weight_f16,
                "final_norm_weight_f16",
            ),
        ),
        (
            "final_norm_epsilon_q32",
            _uint(
                final_norm_epsilon_q32,
                8,
                "final_norm_epsilon_q32",
                positive=True,
            ),
        ),
    ]
    if audit_policy is not None:
        if not isinstance(audit_policy, ExecutionAuditPolicyV2):
            raise ManifestFormatError("audit_policy must be an ExecutionAuditPolicyV2")
        fields.append(("audit_policy", audit_policy.canonical_bytes()))
    encoded = _record(fields)
    return hashlib.sha256(
        _frame(_MODEL_EXECUTION_PARAMETER_DOMAIN) + _frame(encoded)
    ).digest()


@dataclass(frozen=True)
class ModelSpecIdentity:
    """Immutable copy of every field in the on-chain ModelSpec."""

    model_id: str
    weight_merkle_root: bytes
    layer_roots: tuple[bytes, ...]
    num_layers: int
    hidden_dim: int
    intermediate_dim: int
    num_heads: int
    head_dim: int
    vocab_size: int
    quant_mode: str
    merkle_chunk_size: int
    activation: str
    norm_type: str
    attention_type: str
    num_experts: int
    expert_w_num_cols: int
    lm_head_root: bytes
    embedding_root: bytes
    weight_file_hash: bytes
    tokenizer_hash: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_roots", tuple(self.layer_roots))
        self.canonical_bytes()

    @classmethod
    def from_on_chain(cls, spec: "OnChainModelSpec") -> "ModelSpecIdentity":
        """Copy an OnChainModelSpec without retaining mutable list fields."""
        try:
            source_fields = {field.name for field in fields(spec)}
        except TypeError as exc:
            raise ManifestFormatError(
                "model spec must be an OnChainModelSpec dataclass"
            ) from exc
        identity_fields = {field.name for field in fields(cls)}
        if source_fields != identity_fields:
            raise ManifestFormatError(
                "on-chain ModelSpec fields do not match the manifest schema"
            )
        return cls(
            model_id=spec.model_id,
            weight_merkle_root=spec.weight_merkle_root,
            layer_roots=tuple(spec.layer_roots),
            num_layers=spec.num_layers,
            hidden_dim=spec.hidden_dim,
            intermediate_dim=spec.intermediate_dim,
            num_heads=spec.num_heads,
            head_dim=spec.head_dim,
            vocab_size=spec.vocab_size,
            quant_mode=spec.quant_mode,
            merkle_chunk_size=spec.merkle_chunk_size,
            activation=spec.activation,
            norm_type=spec.norm_type,
            attention_type=spec.attention_type,
            num_experts=spec.num_experts,
            expert_w_num_cols=spec.expert_w_num_cols,
            lm_head_root=spec.lm_head_root,
            embedding_root=spec.embedding_root,
            weight_file_hash=spec.weight_file_hash,
            tokenizer_hash=spec.tokenizer_hash,
        )

    def canonical_bytes(self) -> bytes:
        model_id = _text(self.model_id, "model_id")
        weight_root = _root(self.weight_merkle_root, "weight_merkle_root")
        num_layers = _uint(self.num_layers, 8, "num_layers", positive=True)
        layer_roots = tuple(
            _root(root, f"layer_roots[{index}]")
            for index, root in enumerate(self.layer_roots)
        )
        if len(layer_roots) != self.num_layers:
            raise ManifestFormatError(
                "layer_roots must contain exactly num_layers entries"
            )

        return _record(
            (
                ("model_id", model_id),
                ("weight_merkle_root", weight_root),
                ("layer_roots", _sequence(layer_roots)),
                ("num_layers", num_layers),
                ("hidden_dim", _uint(self.hidden_dim, 8, "hidden_dim", positive=True)),
                (
                    "intermediate_dim",
                    _uint(self.intermediate_dim, 8, "intermediate_dim", positive=True),
                ),
                ("num_heads", _uint(self.num_heads, 8, "num_heads", positive=True)),
                ("head_dim", _uint(self.head_dim, 8, "head_dim", positive=True)),
                ("vocab_size", _uint(self.vocab_size, 8, "vocab_size", positive=True)),
                ("quant_mode", _text(self.quant_mode, "quant_mode", max_bytes=128)),
                (
                    "merkle_chunk_size",
                    _uint(
                        self.merkle_chunk_size, 8, "merkle_chunk_size", positive=True
                    ),
                ),
                ("activation", _text(self.activation, "activation", max_bytes=128)),
                ("norm_type", _text(self.norm_type, "norm_type", max_bytes=128)),
                (
                    "attention_type",
                    _text(self.attention_type, "attention_type", max_bytes=128),
                ),
                ("num_experts", _uint(self.num_experts, 8, "num_experts")),
                (
                    "expert_w_num_cols",
                    _uint(self.expert_w_num_cols, 8, "expert_w_num_cols"),
                ),
                ("lm_head_root", _root(self.lm_head_root, "lm_head_root")),
                ("embedding_root", _root(self.embedding_root, "embedding_root")),
                ("weight_file_hash", _root(self.weight_file_hash, "weight_file_hash")),
                ("tokenizer_hash", _root(self.tokenizer_hash, "tokenizer_hash")),
            )
        )


@dataclass(frozen=True)
class OperationDescriptor:
    """PCS commitment and matrix shape for one exact model operation."""

    layer: int
    operation_id: str
    expert_id: int | None
    rows: int
    cols: int
    commitment: bytes
    # Runtime-output binding parameters.  The proof matrix contains signed
    # int8 weights W_q and the miner commits a per-row X scale.  Validators
    # compare the selected exact X_q*W_q result with the pre-challenge runtime
    # output using these manifest-authenticated fixed-point parameters.
    # Values use unsigned Q32 (integer / 2**32); relative tolerance is in bps.
    weight_scale_q32: int = 1 << 32
    # New manifests quantize each consecutive 16-column output block with its
    # own signed Q32 scale.  The empty tuple retains canonical compatibility
    # with already signed scalar-scale manifests.
    weight_block_scales_q32: tuple[int, ...] = ()
    min_x_scale_q32: int = 1 << 24
    runtime_abs_tolerance_q32: int = 0
    runtime_rel_tolerance_bps: int = 0

    def __post_init__(self) -> None:
        self.canonical_bytes()

    def sort_key(
        self,
    ) -> tuple[int, bytes, int, int, int, bytes, int, tuple[int, ...], int, int, int]:
        expert_sort = -1 if self.expert_id is None else self.expert_id
        return (
            self.layer,
            _identifier(self.operation_id, "operation_id"),
            expert_sort,
            self.rows,
            self.cols,
            self.commitment,
            self.weight_scale_q32,
            self.weight_block_scales_q32,
            self.min_x_scale_q32,
            self.runtime_abs_tolerance_q32,
            self.runtime_rel_tolerance_bps,
        )

    def identity_key(self) -> tuple[int, str, int | None]:
        return self.layer, self.operation_id, self.expert_id

    def canonical_bytes(self) -> bytes:
        layer = _layer_int(self.layer)
        operation_id = _identifier(self.operation_id, "operation_id")
        if self.expert_id is None:
            expert_id = b"\x00"
        else:
            expert_id = b"\x01" + _uint(self.expert_id, 8, "expert_id")
        rows = _uint(self.rows, 8, "rows", positive=True)
        cols = _uint(self.cols, 8, "cols", positive=True)
        commitment = _fixed_bytes(self.commitment, 32, "operation commitment root")
        weight_scale_q32 = _uint(
            self.weight_scale_q32,
            8,
            "weight_scale_q32",
            positive=True,
        )
        if not isinstance(self.weight_block_scales_q32, tuple):
            raise ManifestFormatError("weight_block_scales_q32 must be a tuple")
        weight_block_scales_q32 = tuple(
            _uint(
                scale,
                8,
                f"weight_block_scales_q32[{index}]",
                positive=True,
            )
            for index, scale in enumerate(self.weight_block_scales_q32)
        )
        if (
            weight_block_scales_q32
            and len(weight_block_scales_q32)
            != (self.cols + WEIGHT_SCALE_BLOCK_COLS - 1) // WEIGHT_SCALE_BLOCK_COLS
        ):
            raise ManifestFormatError(
                "weight_block_scales_q32 does not cover every output block"
            )
        min_x_scale_q32 = _uint(
            self.min_x_scale_q32,
            8,
            "min_x_scale_q32",
            positive=True,
        )
        runtime_abs_tolerance_q32 = _uint(
            self.runtime_abs_tolerance_q32,
            8,
            "runtime_abs_tolerance_q32",
        )
        runtime_rel_tolerance_bps = _uint(
            self.runtime_rel_tolerance_bps,
            4,
            "runtime_rel_tolerance_bps",
        )
        if self.runtime_rel_tolerance_bps > 10_000:
            raise ManifestFormatError("runtime_rel_tolerance_bps must not exceed 10000")
        fields = [
            ("layer", layer),
            ("operation_id", operation_id),
            ("expert_id", expert_id),
            ("rows", rows),
            ("cols", cols),
            ("commitment", commitment),
            ("weight_scale_q32", weight_scale_q32),
        ]
        if weight_block_scales_q32:
            fields.append(
                (
                    "weight_block_scales_q32",
                    struct.pack(">I", len(weight_block_scales_q32))
                    + b"".join(weight_block_scales_q32),
                )
            )
        fields.extend(
            (
                ("min_x_scale_q32", min_x_scale_q32),
                ("runtime_abs_tolerance_q32", runtime_abs_tolerance_q32),
                ("runtime_rel_tolerance_bps", runtime_rel_tolerance_bps),
            )
        )
        return _record(fields)


@dataclass(frozen=True)
class LayerExecutionDescriptor:
    """Authority-authenticated execution profile for one transformer layer.

    ``bridge_parameter_root`` commits the non-PCS parameters used between the
    registered linear operations (normalization, activation, convolution, and
    recurrent-gate parameters).  Validators only accept profile identifiers
    whose transition rules they implement; a signature cannot make an unknown
    profile executable.
    """

    layer: int
    attention_profile: str
    bridge_parameter_root: bytes
    input_norm_weight_f16: bytes = b""
    post_attention_norm_weight_f16: bytes = b""
    norm_epsilon_q32: int = 0
    transition_profile: str | None = None
    transition_parameter_root: bytes = b""
    transition_parameters: bytes = b""

    def __post_init__(self) -> None:
        self.canonical_bytes()

    def canonical_bytes(self) -> bytes:
        layer = _layer_int(self.layer)
        if self.layer < 0:
            raise ManifestFormatError("execution-profile layers must be non-negative")
        attention_profile = _identifier(
            self.attention_profile,
            "attention_profile",
        )
        bridge_parameter_root = _root(
            self.bridge_parameter_root,
            "bridge_parameter_root",
        )
        if bridge_parameter_root == bytes(32):
            raise ManifestFormatError("bridge_parameter_root must not be the zero root")
        has_parameters = bool(
            self.input_norm_weight_f16
            or self.post_attention_norm_weight_f16
            or self.norm_epsilon_q32
        )
        if has_parameters:
            input_norm_weight_f16 = _parameter_bytes(
                self.input_norm_weight_f16,
                "input_norm_weight_f16",
            )
            post_attention_norm_weight_f16 = _parameter_bytes(
                self.post_attention_norm_weight_f16,
                "post_attention_norm_weight_f16",
            )
            norm_epsilon_q32 = _uint(
                self.norm_epsilon_q32,
                8,
                "norm_epsilon_q32",
                positive=True,
            )
            expected_root = qwen_layer_bridge_parameter_root_v2(
                layer=self.layer,
                attention_profile=self.attention_profile,
                input_norm_weight_f16=input_norm_weight_f16,
                post_attention_norm_weight_f16=post_attention_norm_weight_f16,
                norm_epsilon_q32=self.norm_epsilon_q32,
            )
            if bridge_parameter_root != expected_root:
                raise ManifestFormatError(
                    "bridge_parameter_root does not match its signed parameters"
                )
        else:
            input_norm_weight_f16 = b""
            post_attention_norm_weight_f16 = b""
            norm_epsilon_q32 = b""
        transition_fields = (
            self.transition_profile,
            self.transition_parameter_root,
            self.transition_parameters,
        )
        if all(value in (None, b"") for value in transition_fields):
            transition_profile = b"\x00"
            transition_parameter_root = b""
            transition_parameters = b""
        elif (
            not isinstance(self.transition_profile, str)
            or not self.transition_parameter_root
            or not self.transition_parameters
        ):
            raise ManifestFormatError(
                "layer transition descriptor fields must be present together"
            )
        else:
            encoded_transition_profile = _identifier(
                self.transition_profile,
                "transition_profile",
            )
            transition_profile = b"\x01" + encoded_transition_profile
            transition_parameters = _parameter_bytes(
                self.transition_parameters,
                "transition_parameters",
            )
            transition_parameter_root = _root(
                self.transition_parameter_root,
                "transition_parameter_root",
            )
            expected_transition_root = qwen_layer_transition_parameter_root_v2(
                layer=self.layer,
                transition_profile=self.transition_profile,
                transition_parameters=transition_parameters,
            )
            if transition_parameter_root != expected_transition_root:
                raise ManifestFormatError(
                    "transition_parameter_root does not match its signed parameters"
                )
        return _record(
            (
                ("layer", layer),
                ("attention_profile", attention_profile),
                ("bridge_parameter_root", bridge_parameter_root),
                ("input_norm_weight_f16", input_norm_weight_f16),
                (
                    "post_attention_norm_weight_f16",
                    post_attention_norm_weight_f16,
                ),
                ("norm_epsilon_q32", norm_epsilon_q32),
                ("transition_profile", transition_profile),
                (
                    "transition_parameter_root",
                    transition_parameter_root,
                ),
                ("transition_parameters", transition_parameters),
            )
        )


@dataclass(frozen=True)
class ExecutionAuditPolicyV2:
    """Authority-signed hard-audit coverage policy for one model profile.

    The policy is deliberately model metadata rather than validator process
    configuration: lowering its coverage requires a new authenticated manifest
    and qualification run, not a local validator flag.
    """

    # This rate governs the post-commitment execution audit only.  It is
    # deliberately distinct from request-level decode sampling so a caller
    # cannot identify a hard-audit candidate from its request metadata.
    hard_audit_bps: int
    hard_layer_count: int
    min_full_attention_layers: int
    min_gdn_layers: int
    full_attention_heads_per_layer: int
    hard_blocks_per_operation: int

    def __post_init__(self) -> None:
        self.canonical_bytes()

    def canonical_bytes(self) -> bytes:
        if (
            isinstance(self.hard_audit_bps, bool)
            or not isinstance(self.hard_audit_bps, int)
            or not 1 <= self.hard_audit_bps <= 10_000
        ):
            raise ManifestFormatError(
                "hard_audit_bps must be an integer in 1..10000"
            )
        positive_values = (
            ("hard_layer_count", self.hard_layer_count),
            ("hard_blocks_per_operation", self.hard_blocks_per_operation),
            ("full_attention_heads_per_layer", self.full_attention_heads_per_layer),
        )
        for name, value in positive_values:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ManifestFormatError(f"{name} must be a positive integer")
        if self.hard_blocks_per_operation > MAX_BLOCKS_PER_OPERATION:
            raise ManifestFormatError(
                "hard_blocks_per_operation exceeds the protocol maximum"
            )
        optional_coverage_values = (
            ("min_full_attention_layers", self.min_full_attention_layers),
            ("min_gdn_layers", self.min_gdn_layers),
        )
        for name, value in optional_coverage_values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ManifestFormatError(f"{name} must be a non-negative integer")
        if self.hard_layer_count < (
            self.min_full_attention_layers + self.min_gdn_layers
        ):
            raise ManifestFormatError(
                "hard_layer_count is smaller than required transition coverage"
            )
        values = (
            (("hard_audit_bps", self.hard_audit_bps),)
            + positive_values[:2]
            + optional_coverage_values
            + positive_values[2:]
        )
        return _record((name, _uint(value, 4, name)) for name, value in values)


@dataclass(frozen=True)
class ModelExecutionDescriptor:
    """Authority-signed model-boundary parameters for the causal profile."""

    embedding_scale_q32: int
    final_norm_weight_f16: bytes
    final_norm_epsilon_q32: int
    parameter_root: bytes
    audit_policy: ExecutionAuditPolicyV2 | None = None

    def __post_init__(self) -> None:
        self.canonical_bytes()

    def canonical_bytes(self) -> bytes:
        embedding_scale_q32 = _uint(
            self.embedding_scale_q32,
            8,
            "embedding_scale_q32",
            positive=True,
        )
        final_norm_weight_f16 = _parameter_bytes(
            self.final_norm_weight_f16,
            "final_norm_weight_f16",
        )
        final_norm_epsilon_q32 = _uint(
            self.final_norm_epsilon_q32,
            8,
            "final_norm_epsilon_q32",
            positive=True,
        )
        parameter_root = _root(self.parameter_root, "model execution parameter root")
        audit_policy = self.audit_policy
        if audit_policy is not None and not isinstance(
            audit_policy,
            ExecutionAuditPolicyV2,
        ):
            raise ManifestFormatError("audit_policy must be an ExecutionAuditPolicyV2")
        expected_root = qwen_model_execution_parameter_root_v2(
            embedding_scale_q32=self.embedding_scale_q32,
            final_norm_weight_f16=final_norm_weight_f16,
            final_norm_epsilon_q32=self.final_norm_epsilon_q32,
            audit_policy=audit_policy,
        )
        if parameter_root != expected_root:
            raise ManifestFormatError(
                "model execution parameter root does not match its signed parameters"
            )
        fields = [
            ("embedding_scale_q32", embedding_scale_q32),
            ("final_norm_weight_f16", final_norm_weight_f16),
            ("final_norm_epsilon_q32", final_norm_epsilon_q32),
            ("parameter_root", parameter_root),
        ]
        if audit_policy is not None:
            fields.append(("audit_policy", audit_policy.canonical_bytes()))
        return _record(fields)


def _canonical_layer_execution(
    layers: Sequence[LayerExecutionDescriptor],
    *,
    num_layers: int,
) -> tuple[LayerExecutionDescriptor, ...]:
    canonical = tuple(layers)
    if len(canonical) > MAX_EXECUTION_PROFILE_LAYERS:
        raise ManifestFormatError("execution-profile layer count exceeds the limit")
    for descriptor in canonical:
        if not isinstance(descriptor, LayerExecutionDescriptor):
            raise ManifestFormatError(
                "layer_execution must contain LayerExecutionDescriptor values"
            )
        descriptor.canonical_bytes()
    if canonical:
        if len(canonical) != num_layers:
            raise ManifestFormatError(
                "layer_execution must contain exactly num_layers entries"
            )
        if tuple(descriptor.layer for descriptor in canonical) != tuple(
            range(num_layers)
        ):
            raise ManifestFormatError(
                "layer_execution must be ordered and cover every layer exactly once"
            )
    return canonical


def _canonical_operations(
    operations: Sequence[OperationDescriptor],
    *,
    num_layers: int,
) -> tuple[OperationDescriptor, ...]:
    canonical = tuple(operations)
    if not canonical:
        raise ManifestFormatError("operations must not be empty")
    if len(canonical) > MAX_MANIFEST_OPERATIONS:
        raise ManifestFormatError("operation count exceeds the manifest limit")
    for operation in canonical:
        if not isinstance(operation, OperationDescriptor):
            raise ManifestFormatError(
                "operations must contain OperationDescriptor values"
            )
        if operation.layer >= num_layers:
            raise ManifestFormatError(
                "operation layer must be -1 or less than num_layers"
            )
        operation.canonical_bytes()
    identities = [operation.identity_key() for operation in canonical]
    if len(set(identities)) != len(identities):
        raise ManifestFormatError("operation identities must be unique")
    expected_order = tuple(sorted(canonical, key=OperationDescriptor.sort_key))
    if canonical != expected_order:
        raise ManifestFormatError("operations must be in canonical order")
    return canonical


@dataclass(frozen=True)
class StaticWeightCommitmentManifest:
    """Canonical protocol v2 manifest signed by configured authorities."""

    chain_id: int
    netuid: int
    registry_address: bytes
    model_spec: ModelSpecIdentity
    pcs_suite: str
    pcs_generator_version: str
    operations: tuple[OperationDescriptor, ...]
    # Empty fields identify the pre-causal-binding manifest schema.  Runtime
    # proof-v2 release paths must reject that schema; retaining an explicit
    # representation here allows deterministic migration and negative tests.
    execution_profile: str | None = None
    layer_execution: tuple[LayerExecutionDescriptor, ...] = ()
    model_execution: ModelExecutionDescriptor | None = None
    protocol_version: int = PROTOCOL_VERSION
    _canonical_cache: bytes = field(init=False, repr=False, compare=False)
    _digest_cache: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model_spec, ModelSpecIdentity):
            raise ManifestFormatError("model_spec must be a ModelSpecIdentity")
        registry_address = _address_bytes(self.registry_address, "registry_address")
        if registry_address == _ZERO_ADDRESS:
            raise ManifestFormatError("registry_address must not be the zero address")
        object.__setattr__(self, "registry_address", registry_address)
        object.__setattr__(
            self,
            "operations",
            _canonical_operations(
                self.operations, num_layers=self.model_spec.num_layers
            ),
        )
        if self.execution_profile is None:
            if self.layer_execution or self.model_execution is not None:
                raise ManifestFormatError(
                    "execution descriptors require an execution_profile"
                )
        else:
            _identifier(self.execution_profile, "execution_profile")
            if not self.layer_execution:
                raise ManifestFormatError(
                    "execution_profile requires exact per-layer descriptors"
                )
            if self.model_execution is not None and not isinstance(
                self.model_execution,
                ModelExecutionDescriptor,
            ):
                raise ManifestFormatError(
                    "model_execution must be a ModelExecutionDescriptor"
                )
        object.__setattr__(
            self,
            "layer_execution",
            _canonical_layer_execution(
                self.layer_execution,
                num_layers=self.model_spec.num_layers,
            ),
        )
        canonical = self._build_canonical_bytes()
        object.__setattr__(self, "_canonical_cache", canonical)
        object.__setattr__(
            self,
            "_digest_cache",
            hashlib.sha256(_frame(_DIGEST_DOMAIN) + _frame(canonical)).digest(),
        )

    def _build_canonical_bytes(self) -> bytes:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ManifestFormatError(
                "static weight manifests require proof protocol version 2"
            )
        chain_id = _uint(self.chain_id, 32, "chain_id", positive=True)
        netuid = _uint(self.netuid, 32, "netuid")
        registry_address = _address_bytes(self.registry_address, "registry_address")
        if registry_address == _ZERO_ADDRESS:
            raise ManifestFormatError("registry_address must not be the zero address")
        pcs_suite = _identifier(self.pcs_suite, "pcs_suite")
        generator_version = _identifier(
            self.pcs_generator_version, "pcs_generator_version"
        )
        operations = _canonical_operations(
            self.operations, num_layers=self.model_spec.num_layers
        )
        if self.execution_profile is None:
            execution_profile = b"\x00"
        else:
            execution_profile = b"\x01" + _identifier(
                self.execution_profile,
                "execution_profile",
            )
        layer_execution = _canonical_layer_execution(
            self.layer_execution,
            num_layers=self.model_spec.num_layers,
        )
        if self.model_execution is None:
            model_execution = b"\x00"
        else:
            model_execution = b"\x01" + self.model_execution.canonical_bytes()
        return _record(
            (
                (
                    "protocol_version",
                    _uint(self.protocol_version, 2, "protocol_version"),
                ),
                ("chain_id", chain_id),
                ("netuid", netuid),
                ("registry_address", registry_address),
                ("model_spec", self.model_spec.canonical_bytes()),
                ("pcs_suite", pcs_suite),
                ("pcs_generator_version", generator_version),
                ("execution_profile", execution_profile),
                (
                    "layer_execution",
                    _sequence(
                        descriptor.canonical_bytes() for descriptor in layer_execution
                    ),
                ),
                ("model_execution", model_execution),
                (
                    "operations",
                    _sequence(operation.canonical_bytes() for operation in operations),
                ),
            )
        )

    def canonical_bytes(self) -> bytes:
        """Return the immutable canonical encoding validated at construction."""

        return self._canonical_cache

    def digest(self) -> bytes:
        """Return the SHA-256 digest used as the EIP-191 message body."""
        return self._digest_cache


def _coerce_model_spec(
    expected: ModelSpecIdentity | "OnChainModelSpec",
) -> ModelSpecIdentity:
    if isinstance(expected, ModelSpecIdentity):
        return expected
    try:
        return ModelSpecIdentity.from_on_chain(expected)
    except (AttributeError, TypeError) as exc:
        raise ManifestContextError(
            "expected_model_spec is not an on-chain ModelSpec"
        ) from exc


def _signature_bytes(value: str | bytes) -> bytes:
    if isinstance(value, bytes):
        signature = value
    elif isinstance(value, str):
        encoded = value[2:] if value.startswith("0x") else value
        try:
            signature = bytes.fromhex(encoded)
        except ValueError as exc:
            raise ManifestSignatureError(
                "authority signature must be hexadecimal"
            ) from exc
    else:
        raise ManifestSignatureError("authority signature must be bytes or hexadecimal")
    if len(signature) != 65:
        raise ManifestSignatureError("authority signature must be exactly 65 bytes")
    r = int.from_bytes(signature[:32], "big")
    s = int.from_bytes(signature[32:64], "big")
    v = signature[64]
    if not 0 < r < _SECP256K1_ORDER or not 0 < s <= _SECP256K1_HALF_ORDER:
        raise ManifestSignatureError("authority signature is not canonical")
    if v not in (27, 28):
        raise ManifestSignatureError("authority signature recovery id is not canonical")
    return signature


def _expected_authorities(values: Collection[str | bytes]) -> frozenset[bytes]:
    authorities = tuple(_address_bytes(value, "expected authority") for value in values)
    if not authorities:
        raise ManifestSignatureError("expected authority set must not be empty")
    if _ZERO_ADDRESS in authorities:
        raise ManifestSignatureError(
            "expected authority set must not contain the zero address"
        )
    if len(set(authorities)) != len(authorities):
        raise ManifestSignatureError(
            "expected authority set must not contain duplicates"
        )
    return frozenset(authorities)


def verify_signed_manifest(
    manifest: StaticWeightCommitmentManifest,
    signatures: Sequence[str | bytes],
    *,
    expected_chain_id: int,
    expected_netuid: int,
    expected_registry_address: str | bytes,
    expected_model_spec: ModelSpecIdentity | "OnChainModelSpec",
    expected_pcs_suite: str,
    expected_pcs_generator_version: str,
    expected_operations: Sequence[OperationDescriptor],
    expected_authority_signers: Collection[str | bytes],
    authority_threshold: int,
) -> tuple[str, ...]:
    """Verify exact manifest context and a threshold of EIP-191 authorities.

    The expected context is required at every call so a valid signed manifest
    cannot be accepted for another chain, subnet, registry, model, PCS suite,
    generator version, or operation set.
    """
    if not isinstance(manifest, StaticWeightCommitmentManifest):
        raise ManifestContextError("manifest has an unexpected type")

    expected_registry = _address_bytes(
        expected_registry_address, "expected_registry_address"
    )
    expected_spec = _coerce_model_spec(expected_model_spec)
    expected_ops = _canonical_operations(
        expected_operations, num_layers=expected_spec.num_layers
    )
    expected_fields = (
        ("chain_id", manifest.chain_id, expected_chain_id),
        ("netuid", manifest.netuid, expected_netuid),
        ("registry_address", manifest.registry_address, expected_registry),
        ("model_spec", manifest.model_spec, expected_spec),
        ("pcs_suite", manifest.pcs_suite, expected_pcs_suite),
        (
            "pcs_generator_version",
            manifest.pcs_generator_version,
            expected_pcs_generator_version,
        ),
        ("operations", manifest.operations, expected_ops),
    )
    for field_name, actual, expected in expected_fields:
        if actual != expected:
            raise ManifestContextError(
                f"manifest {field_name} does not match the expected value"
            )

    authorities = _expected_authorities(expected_authority_signers)
    if (
        isinstance(authority_threshold, bool)
        or not isinstance(authority_threshold, int)
        or authority_threshold < 1
        or authority_threshold > len(authorities)
    ):
        raise ManifestSignatureError(
            "authority_threshold must be between 1 and the authority count"
        )
    if len(signatures) > len(authorities):
        raise ManifestSignatureError(
            "authority signature count exceeds the authority count"
        )

    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
        from eth_keys.exceptions import BadSignature
    except ImportError as exc:
        raise RuntimeError(
            "manifest signature verification requires the chain optional dependency"
        ) from exc

    signable = encode_defunct(primitive=manifest.digest())
    recovered: set[bytes] = set()
    for signature_value in signatures:
        signature = _signature_bytes(signature_value)
        try:
            signer = _address_bytes(
                Account.recover_message(signable, signature=signature),
                "recovered authority",
            )
        except (BadSignature, ManifestFormatError, ValueError, TypeError) as exc:
            raise ManifestSignatureError("authority signature recovery failed") from exc
        if signer not in authorities:
            raise ManifestSignatureError(
                "signature was not produced by an expected authority"
            )
        if signer in recovered:
            raise ManifestSignatureError(
                "authority signatures must have unique signers"
            )
        recovered.add(signer)

    if len(recovered) < authority_threshold:
        raise ManifestSignatureError("authority signature threshold was not met")
    return tuple(f"0x{address.hex()}" for address in sorted(recovered))
