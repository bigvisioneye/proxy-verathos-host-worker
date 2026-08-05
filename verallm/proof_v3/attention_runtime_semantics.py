"""Signed runtime semantics for anchor-backed full-attention verification.

The projection manifest authenticates the digest of this small artifact.
Validators load it without model weights and use it to interpret raw QKV
execution-anchor bytes.  Unknown layouts, normalization rules, RoPE rules or
cache mappings fail closed rather than falling back to local model code.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError

ATTENTION_RUNTIME_SEMANTICS_VERSION_V3 = 1
ATTENTION_RUNTIME_SEMANTICS_ABI_V3 = (
    "attention.runtime_semantics.anchor_kv.v1"
)
QKV_CONTIGUOUS_LAYOUT_V3 = "qkv.contiguous.v1"
Q_GATE_INTERLEAVED_LAYOUT_V3 = "q_gate.interleaved.v1"
NO_QK_NORM_V3 = "none"
GEMMA_RMS_NORM_V3 = "gemma_rms.v1"
NEOX_ROPE_V3 = "neox.v1"
LOGICAL_PAGED_KV_V3 = "paged_kv.logical.v1"

_DOMAIN = b"VERATHOS/PROOF_V3/ATTENTION_RUNTIME_SEMANTICS/V1"
_LAYOUTS = frozenset(
    {QKV_CONTIGUOUS_LAYOUT_V3, Q_GATE_INTERLEAVED_LAYOUT_V3}
)
_NORMS = frozenset({NO_QK_NORM_V3, GEMMA_RMS_NORM_V3})

__all__ = [
    "ATTENTION_RUNTIME_SEMANTICS_ABI_V3",
    "ATTENTION_RUNTIME_SEMANTICS_VERSION_V3",
    "AttentionRuntimeSemanticsV3",
    "AttentionNormBindingV3",
    "GEMMA_RMS_NORM_V3",
    "LOGICAL_PAGED_KV_V3",
    "NEOX_ROPE_V3",
    "NO_QK_NORM_V3",
    "QKV_CONTIGUOUS_LAYOUT_V3",
    "Q_GATE_INTERLEAVED_LAYOUT_V3",
    "dump_attention_runtime_semantics_v3",
    "load_attention_runtime_semantics_v3",
]


def _float_bits(value: float, name: str, *, positive: bool) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0)
        or (not positive and float(value) < 0)
    ):
        raise ProofV3Error(f"attention runtime {name} is malformed")
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


@dataclass(frozen=True, slots=True)
class AttentionNormBindingV3:
    layer_index: int
    q_weight_bytes: bytes
    k_weight_bytes: bytes

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or not 0 <= self.layer_index < 1 << 32
            or not isinstance(self.q_weight_bytes, bytes)
            or not isinstance(self.k_weight_bytes, bytes)
            or not self.q_weight_bytes
            or len(self.q_weight_bytes) != len(self.k_weight_bytes)
            or len(self.q_weight_bytes) & 1
            or len(self.q_weight_bytes) >= 1 << 16
        ):
            raise ProofV3Error(
                "attention runtime norm binding is malformed"
            )


@dataclass(frozen=True, slots=True)
class AttentionRuntimeSemanticsV3:
    adapter_id: str
    qkv_layout_id: str
    q_norm_id: str
    k_norm_id: str
    q_norm_epsilon: float
    k_norm_epsilon: float
    rope_id: str
    rope_theta: float
    rotary_dimension: int
    cache_layout_id: str = LOGICAL_PAGED_KV_V3
    norm_encoding_id: str = "bf16.v1"
    norm_bindings: tuple[AttentionNormBindingV3, ...] = ()
    integer_tolerance: int = 0
    version: int = ATTENTION_RUNTIME_SEMANTICS_VERSION_V3

    def __post_init__(self) -> None:
        if (
            self.version != ATTENTION_RUNTIME_SEMANTICS_VERSION_V3
            or not isinstance(self.adapter_id, str)
            or not self.adapter_id
            or len(self.adapter_id.encode("ascii", "strict")) > 96
            or self.qkv_layout_id not in _LAYOUTS
            or self.q_norm_id not in _NORMS
            or self.k_norm_id not in _NORMS
            or self.rope_id != NEOX_ROPE_V3
            or self.cache_layout_id != LOGICAL_PAGED_KV_V3
            or self.norm_encoding_id not in {"fp16.v1", "bf16.v1"}
            or isinstance(self.rotary_dimension, bool)
            or not isinstance(self.rotary_dimension, int)
            or self.rotary_dimension < 2
            or self.rotary_dimension & 1
            or self.rotary_dimension >= 1 << 16
            or isinstance(self.integer_tolerance, bool)
            or not isinstance(self.integer_tolerance, int)
            or not 0 <= self.integer_tolerance <= 2
        ):
            raise ProofV3Error("attention runtime semantics are malformed")
        _float_bits(self.rope_theta, "rope_theta", positive=True)
        _float_bits(self.q_norm_epsilon, "q_norm_epsilon", positive=False)
        _float_bits(self.k_norm_epsilon, "k_norm_epsilon", positive=False)
        if (
            (self.q_norm_id == NO_QK_NORM_V3)
            != (float(self.q_norm_epsilon) == 0.0)
            or (self.k_norm_id == NO_QK_NORM_V3)
            != (float(self.k_norm_epsilon) == 0.0)
        ):
            raise ProofV3Error(
                "attention runtime norm rule and epsilon disagree"
            )
        if (
            self.qkv_layout_id == Q_GATE_INTERLEAVED_LAYOUT_V3
            and (
                self.q_norm_id != GEMMA_RMS_NORM_V3
                or self.k_norm_id != GEMMA_RMS_NORM_V3
            )
        ):
            raise ProofV3Error(
                "gated QKV layout requires explicit Q/K normalization"
            )
        bindings = tuple(self.norm_bindings)
        if any(not isinstance(item, AttentionNormBindingV3)
               for item in bindings):
            raise ProofV3Error(
                "attention runtime norm bindings have an unexpected type"
            )
        if tuple(item.layer_index for item in bindings) != tuple(
            sorted({item.layer_index for item in bindings})
        ):
            raise ProofV3Error(
                "attention runtime norm bindings must be ordered and distinct"
            )
        if (
            self.q_norm_id == NO_QK_NORM_V3
            and bindings
            or self.q_norm_id != NO_QK_NORM_V3
            and not bindings
        ):
            raise ProofV3Error(
                "attention runtime norm bindings disagree with the norm rule"
            )
        object.__setattr__(self, "norm_bindings", bindings)

    @property
    def gated(self) -> bool:
        return self.qkv_layout_id == Q_GATE_INTERLEAVED_LAYOUT_V3

    def canonical_bytes(self) -> bytes:
        fields = (
            ATTENTION_RUNTIME_SEMANTICS_ABI_V3,
            self.adapter_id,
            self.qkv_layout_id,
            self.q_norm_id,
            self.k_norm_id,
            self.rope_id,
            self.cache_layout_id,
            self.norm_encoding_id,
        )
        encoded = []
        for value in fields:
            raw = value.encode("ascii", "strict")
            encoded.append(struct.pack("<H", len(raw)) + raw)
        norm_payload = [
            struct.pack("<I", len(self.norm_bindings))
        ]
        for binding in self.norm_bindings:
            norm_payload.extend(
                (
                    struct.pack(
                        "<II",
                        binding.layer_index,
                        len(binding.q_weight_bytes),
                    ),
                    binding.q_weight_bytes,
                    binding.k_weight_bytes,
                )
            )
        return (
            _DOMAIN
            + struct.pack(
                "<IIQQQI",
                self.version,
                self.rotary_dimension,
                _float_bits(
                    self.q_norm_epsilon,
                    "q_norm_epsilon",
                    positive=False,
                ),
                _float_bits(
                    self.k_norm_epsilon,
                    "k_norm_epsilon",
                    positive=False,
                ),
                _float_bits(self.rope_theta, "rope_theta", positive=True),
                self.integer_tolerance,
            )
            + b"".join(encoded)
            + b"".join(norm_payload)
        )

    def digest(self) -> bytes:
        return hashlib.sha256(self.canonical_bytes()).digest()


def dump_attention_runtime_semantics_v3(
    semantics: AttentionRuntimeSemanticsV3,
) -> dict:
    if not isinstance(semantics, AttentionRuntimeSemanticsV3):
        raise ProofV3Error(
            "attention runtime semantics have an unexpected type"
        )
    return {
        "adapter_id": semantics.adapter_id,
        "cache_layout_id": semantics.cache_layout_id,
        "digest": semantics.digest().hex(),
        "integer_tolerance": semantics.integer_tolerance,
        "k_norm_epsilon": semantics.k_norm_epsilon,
        "k_norm_id": semantics.k_norm_id,
        "q_norm_epsilon": semantics.q_norm_epsilon,
        "q_norm_id": semantics.q_norm_id,
        "qkv_layout_id": semantics.qkv_layout_id,
        "norm_encoding_id": semantics.norm_encoding_id,
        "norm_bindings": [
            {
                "k_weight_bytes": item.k_weight_bytes.hex(),
                "layer_index": item.layer_index,
                "q_weight_bytes": item.q_weight_bytes.hex(),
            }
            for item in semantics.norm_bindings
        ],
        "rope_id": semantics.rope_id,
        "rope_theta": semantics.rope_theta,
        "rotary_dimension": semantics.rotary_dimension,
        "version": semantics.version,
    }


def load_attention_runtime_semantics_v3(
    source,
) -> AttentionRuntimeSemanticsV3:
    if isinstance(source, (str, bytes)):
        try:
            text = source.decode() if isinstance(source, bytes) else source
            obj = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            with open(source, "rb") as handle:
                obj = json.load(handle)
    else:
        obj = source
    if not isinstance(obj, dict):
        raise ProofV3Error(
            "attention runtime semantics must be a JSON object"
        )
    try:
        result = AttentionRuntimeSemanticsV3(
            adapter_id=str(obj["adapter_id"]),
            qkv_layout_id=str(obj["qkv_layout_id"]),
            q_norm_id=str(obj["q_norm_id"]),
            k_norm_id=str(obj["k_norm_id"]),
            q_norm_epsilon=float(obj["q_norm_epsilon"]),
            k_norm_epsilon=float(obj["k_norm_epsilon"]),
            rope_id=str(obj["rope_id"]),
            rope_theta=float(obj["rope_theta"]),
            rotary_dimension=int(obj["rotary_dimension"]),
            cache_layout_id=str(obj["cache_layout_id"]),
            norm_encoding_id=str(obj.get("norm_encoding_id", "bf16.v1")),
            norm_bindings=tuple(
                AttentionNormBindingV3(
                    layer_index=int(item["layer_index"]),
                    q_weight_bytes=bytes.fromhex(item["q_weight_bytes"]),
                    k_weight_bytes=bytes.fromhex(item["k_weight_bytes"]),
                )
                for item in obj.get("norm_bindings", ())
            ),
            integer_tolerance=int(obj.get("integer_tolerance", 0)),
            version=int(obj["version"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProofV3Error(
            f"attention runtime semantics are malformed: {exc}"
        ) from exc
    declared = obj.get("digest")
    if declared is not None:
        try:
            digest = bytes.fromhex(str(declared))
        except ValueError as exc:
            raise ProofV3Error(
                "attention runtime semantics digest is malformed"
            ) from exc
        if digest != result.digest():
            raise ProofV3VerificationError(
                "attention runtime semantics digest does not match contents"
            )
    return result
