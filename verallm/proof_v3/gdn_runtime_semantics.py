"""Authenticated runtime semantics for Qwen GDN execution anchors.

The artifact is small and model-qualified.  It binds each GDN layer to the
existing canonical Qwen recurrence parameters from proof-v2, while proof-v3
binds the corresponding runtime tensors and state boundary into its request
envelope.  Validators never load model weights from this artifact.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass

from verallm.proof_v2.transition import (
    GDNTransitionParametersV2,
    ProofV2TransitionError,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError

GDN_RUNTIME_SEMANTICS_VERSION_V3 = 2
GDN_RUNTIME_SEMANTICS_ABI_V3 = "gdn.runtime_semantics.qwen.v2"
GDN_RUNTIME_CHECKPOINT_SEMANTICS_VERSION_V3 = 4
GDN_RUNTIME_CHECKPOINT_SEMANTICS_ABI_V3 = (
    "gdn.runtime_semantics.qwen.decode_checkpoints.v4"
)
GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3 = 5
GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_ABI_V3 = (
    "gdn.runtime_semantics.qwen.decode_and_prefix_checkpoints.v5"
)
_DOMAIN_V2 = b"VERATHOS/PROOF_V3/GDN_RUNTIME_SEMANTICS/V2"
_DOMAIN_V4 = b"VERATHOS/PROOF_V3/GDN_RUNTIME_SEMANTICS/V4"
_DOMAIN_V5 = b"VERATHOS/PROOF_V3/GDN_RUNTIME_SEMANTICS/V5"
_RUNTIME_ENCODINGS = frozenset(("fp16.v1", "bf16.v1"))
_STATE_ENCODING_BYTES = {
    "fp16.v1": 2,
    "bf16.v1": 2,
    "fp32.v1": 4,
}

__all__ = [
    "GDN_RUNTIME_SEMANTICS_ABI_V3",
    "GDN_RUNTIME_SEMANTICS_VERSION_V3",
    "GDN_RUNTIME_CHECKPOINT_SEMANTICS_ABI_V3",
    "GDN_RUNTIME_CHECKPOINT_SEMANTICS_VERSION_V3",
    "GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_ABI_V3",
    "GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3",
    "GdnLayerRuntimeSemanticsV3",
    "GdnRuntimeSemanticsV3",
    "dump_gdn_runtime_semantics_v3",
    "load_gdn_runtime_semantics_v3",
]


def _adapter(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("ascii", "strict")) > 96
    ):
        raise ProofV3Error("GDN runtime adapter id is malformed")
    return value


def _encoding(value: object, *, state: bool) -> str:
    choices = _STATE_ENCODING_BYTES if state else _RUNTIME_ENCODINGS
    if not isinstance(value, str) or value not in choices:
        raise ProofV3Error("GDN runtime encoding is not qualified")
    return value


def _q24(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 16 * (1 << 24)
    ):
        raise ProofV3Error(f"GDN {name} is out of range")
    return value


@dataclass(frozen=True, slots=True)
class GdnLayerRuntimeSemanticsV3:
    layer_index: int
    transition_parameters: bytes
    runtime_encoding_id: str
    conv_state_encoding_id: str
    recurrent_state_encoding_id: str
    output_atol_q24: int
    output_rtol_q24: int
    conv_state_atol_q24: int
    recurrent_state_atol_q24: int
    max_decode_replay_rows: int
    decode_checkpoint_stride: int = 0
    prefix_cache_output_atol_q24: int = 0
    prefix_cache_output_rtol_q24: int = 0
    prefix_cache_conv_state_atol_q24: int = 0
    prefix_cache_recurrent_state_atol_q24: int = 0
    max_prefix_cache_replay_rows: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or not 0 <= self.layer_index < 1 << 32
            or not isinstance(self.transition_parameters, bytes)
            or not self.transition_parameters
            or len(self.transition_parameters) >= 1 << 24
        ):
            raise ProofV3Error("GDN layer runtime semantics are malformed")
        try:
            parameters = GDNTransitionParametersV2.from_canonical_bytes(
                self.transition_parameters
            )
        except ProofV2TransitionError as exc:
            raise ProofV3Error(
                "GDN transition parameters are malformed"
            ) from exc
        object.__setattr__(
            self,
            "runtime_encoding_id",
            _encoding(self.runtime_encoding_id, state=False),
        )
        object.__setattr__(
            self,
            "conv_state_encoding_id",
            _encoding(self.conv_state_encoding_id, state=True),
        )
        object.__setattr__(
            self,
            "recurrent_state_encoding_id",
            _encoding(self.recurrent_state_encoding_id, state=True),
        )
        _q24(self.output_atol_q24, "output absolute tolerance")
        _q24(self.output_rtol_q24, "output relative tolerance")
        _q24(
            self.conv_state_atol_q24,
            "convolution-state absolute tolerance",
        )
        _q24(
            self.recurrent_state_atol_q24,
            "recurrent-state absolute tolerance",
        )
        if (
            isinstance(self.max_decode_replay_rows, bool)
            or not isinstance(self.max_decode_replay_rows, int)
            or not 1 <= self.max_decode_replay_rows <= 4096
        ):
            raise ProofV3Error(
                "GDN maximum decode replay rows are out of range"
            )
        if (
            isinstance(self.decode_checkpoint_stride, bool)
            or not isinstance(self.decode_checkpoint_stride, int)
            or not 0 <= self.decode_checkpoint_stride <= 4096
        ):
            raise ProofV3Error(
                "GDN decode checkpoint stride is out of range"
            )
        _q24(
            self.prefix_cache_output_atol_q24,
            "prefix-cache output absolute tolerance",
        )
        _q24(
            self.prefix_cache_output_rtol_q24,
            "prefix-cache output relative tolerance",
        )
        _q24(
            self.prefix_cache_conv_state_atol_q24,
            "prefix-cache convolution-state absolute tolerance",
        )
        _q24(
            self.prefix_cache_recurrent_state_atol_q24,
            "prefix-cache recurrent-state absolute tolerance",
        )
        if (
            isinstance(self.max_prefix_cache_replay_rows, bool)
            or not isinstance(self.max_prefix_cache_replay_rows, int)
            or not 0 <= self.max_prefix_cache_replay_rows <= 4096
        ):
            raise ProofV3Error(
                "GDN maximum prefix-cache replay rows are out of range"
            )
        transition_encodings = {
            "f16": "fp16.v1",
            "bf16": "bf16.v1",
            "f32": "fp32.v1",
        }
        try:
            expected_conv = transition_encodings[
                parameters.runtime_dtype
            ]
            expected_recurrent = transition_encodings[
                parameters.recurrent_state_dtype
            ]
        except KeyError as exc:  # pragma: no cover - canonical parser guards
            raise ProofV3Error(
                "GDN transition parameter encoding is unsupported"
            ) from exc
        # The v2 parameter blob carries the historical reference dtypes. V3
        # may qualify BF16 runtime/cache execution explicitly, but an FP16 or
        # FP32 claim must still agree with the embedded canonical parameters.
        if (
            self.conv_state_encoding_id != "bf16.v1"
            and self.conv_state_encoding_id != expected_conv
        ) or (
            self.recurrent_state_encoding_id != "bf16.v1"
            and self.recurrent_state_encoding_id != expected_recurrent
        ):
            raise ProofV3Error(
                "GDN state encoding disagrees with transition parameters"
            )

    def parameters(self) -> GDNTransitionParametersV2:
        try:
            return GDNTransitionParametersV2.from_canonical_bytes(
                self.transition_parameters
            )
        except ProofV2TransitionError as exc:  # pragma: no cover - frozen
            raise ProofV3VerificationError(
                "authenticated GDN transition parameters are malformed"
            ) from exc

    @property
    def conv_state_bytes(self) -> int:
        parameters = self.parameters()
        return (
            (parameters.conv_kernel_size - 1)
            * (
                2 * parameters.num_key_heads * parameters.key_head_dim
                + parameters.num_value_heads * parameters.value_head_dim
            )
            * _STATE_ENCODING_BYTES[self.conv_state_encoding_id]
        )

    @property
    def recurrent_state_bytes(self) -> int:
        parameters = self.parameters()
        element_bytes = _STATE_ENCODING_BYTES[
            self.recurrent_state_encoding_id
        ]
        return (
            parameters.num_value_heads
            * parameters.value_head_dim
            * parameters.key_head_dim
            * element_bytes
        )


@dataclass(frozen=True, slots=True)
class GdnRuntimeSemanticsV3:
    adapter_id: str
    layers: tuple[GdnLayerRuntimeSemanticsV3, ...]
    version: int = GDN_RUNTIME_SEMANTICS_VERSION_V3
    max_hard_audit_decode_tokens: int = 0

    def __post_init__(self) -> None:
        if self.version not in (
            GDN_RUNTIME_SEMANTICS_VERSION_V3,
            GDN_RUNTIME_CHECKPOINT_SEMANTICS_VERSION_V3,
            GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3,
        ):
            raise ProofV3Error("GDN runtime semantics version is unsupported")
        object.__setattr__(self, "adapter_id", _adapter(self.adapter_id))
        layers = tuple(self.layers)
        if (
            not layers
            or not all(
                isinstance(item, GdnLayerRuntimeSemanticsV3)
                for item in layers
            )
            or tuple(item.layer_index for item in layers)
            != tuple(sorted({item.layer_index for item in layers}))
        ):
            raise ProofV3Error(
                "GDN runtime layer semantics must be ordered and distinct"
            )
        object.__setattr__(self, "layers", layers)
        checkpoint_strides = {
            item.decode_checkpoint_stride for item in layers
        }
        replay_bounds = {
            item.max_decode_replay_rows for item in layers
        }
        if (
            self.version == GDN_RUNTIME_SEMANTICS_VERSION_V3
            and (
                checkpoint_strides != {0}
                or self.max_hard_audit_decode_tokens != 0
                or any(
                    item.conv_state_atol_q24 != item.output_atol_q24
                    or item.recurrent_state_atol_q24
                    != item.output_atol_q24
                    for item in layers
                )
                or any(
                    item.max_prefix_cache_replay_rows
                    or item.prefix_cache_output_atol_q24
                    or item.prefix_cache_output_rtol_q24
                    or item.prefix_cache_conv_state_atol_q24
                    or item.prefix_cache_recurrent_state_atol_q24
                    for item in layers
                )
            )
        ):
            raise ProofV3Error(
                "legacy GDN semantics cannot declare checkpoint audit state"
            )
        if (
            self.version in (
                GDN_RUNTIME_CHECKPOINT_SEMANTICS_VERSION_V3,
                GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3,
            )
            and (
                len(checkpoint_strides) != 1
                or next(iter(checkpoint_strides)) <= 0
                or replay_bounds != checkpoint_strides
            )
        ):
            raise ProofV3Error(
                "checkpointed GDN semantics require one exact replay stride"
            )
        if (
            self.version in (
                GDN_RUNTIME_CHECKPOINT_SEMANTICS_VERSION_V3,
                GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3,
            )
            and (
                isinstance(self.max_hard_audit_decode_tokens, bool)
                or not isinstance(
                    self.max_hard_audit_decode_tokens, int
                )
                or not 2 <= self.max_hard_audit_decode_tokens < 1 << 32
            )
        ):
            raise ProofV3Error(
                "checkpointed GDN semantics require signed audit reach"
            )
        if self.version == GDN_RUNTIME_CHECKPOINT_SEMANTICS_VERSION_V3 and any(
            item.max_prefix_cache_replay_rows
            or item.prefix_cache_output_atol_q24
            or item.prefix_cache_output_rtol_q24
            or item.prefix_cache_conv_state_atol_q24
            or item.prefix_cache_recurrent_state_atol_q24
            for item in layers
        ):
            raise ProofV3Error(
                "decode-only GDN semantics cannot declare prefix-cache bounds"
            )
        if (
            self.version == GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3
            and any(
                item.max_prefix_cache_replay_rows <= 0
                or item.prefix_cache_output_atol_q24 <= 0
                or item.prefix_cache_conv_state_atol_q24 <= 0
                or item.prefix_cache_recurrent_state_atol_q24 <= 0
                for item in layers
            )
        ):
            raise ProofV3Error(
                "prefix-cache GDN semantics require separate qualified bounds"
            )

    @property
    def decode_checkpoint_stride(self) -> int:
        if self.version not in (
            GDN_RUNTIME_CHECKPOINT_SEMANTICS_VERSION_V3,
            GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3,
        ):
            return 0
        return self.layers[0].decode_checkpoint_stride

    def layer_for(self, layer_index: int) -> GdnLayerRuntimeSemanticsV3:
        for item in self.layers:
            if item.layer_index == int(layer_index):
                return item
        raise ProofV3VerificationError(
            f"authenticated GDN semantics have no layer {int(layer_index)}"
        )

    def canonical_bytes(self) -> bytes:
        adapter = self.adapter_id.encode("ascii", "strict")
        checkpointed = self.version in (
            GDN_RUNTIME_CHECKPOINT_SEMANTICS_VERSION_V3,
            GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3,
        )
        prefix_cache = (
            self.version == GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3
        )
        parts = [
            (
                _DOMAIN_V5
                if prefix_cache
                else (_DOMAIN_V4 if checkpointed else _DOMAIN_V2)
            ),
            (
                struct.pack(
                    "<IIIH",
                    self.version,
                    len(self.layers),
                    self.max_hard_audit_decode_tokens,
                    len(adapter),
                )
                if checkpointed
                else struct.pack(
                    "<IIH", self.version, len(self.layers), len(adapter)
                )
            ),
            adapter,
        ]
        for item in self.layers:
            runtime = item.runtime_encoding_id.encode("ascii")
            conv = item.conv_state_encoding_id.encode("ascii")
            recurrent = item.recurrent_state_encoding_id.encode("ascii")
            header = (
                struct.pack(
                        "<IIHHHIII",
                        item.layer_index,
                        len(item.transition_parameters),
                        len(runtime),
                        len(conv),
                        len(recurrent),
                        item.output_atol_q24,
                        item.output_rtol_q24,
                        item.max_decode_replay_rows,
                    )
                if self.version == GDN_RUNTIME_SEMANTICS_VERSION_V3
                else struct.pack(
                    "<IIHHHIIIIII",
                    item.layer_index,
                    len(item.transition_parameters),
                    len(runtime),
                    len(conv),
                    len(recurrent),
                    item.output_atol_q24,
                    item.output_rtol_q24,
                    item.conv_state_atol_q24,
                    item.recurrent_state_atol_q24,
                    item.max_decode_replay_rows,
                    item.decode_checkpoint_stride,
                )
            )
            if prefix_cache:
                header += struct.pack(
                    "<IIIII",
                    item.prefix_cache_output_atol_q24,
                    item.prefix_cache_output_rtol_q24,
                    item.prefix_cache_conv_state_atol_q24,
                    item.prefix_cache_recurrent_state_atol_q24,
                    item.max_prefix_cache_replay_rows,
                )
            parts.extend(
                (
                    header,
                    item.transition_parameters,
                    runtime,
                    conv,
                    recurrent,
                )
            )
        return b"".join(parts)

    def digest(self) -> bytes:
        return hashlib.sha256(self.canonical_bytes()).digest()


def dump_gdn_runtime_semantics_v3(
    semantics: GdnRuntimeSemanticsV3,
) -> dict:
    if not isinstance(semantics, GdnRuntimeSemanticsV3):
        raise ProofV3Error(
            "GDN runtime semantics have an unexpected type"
        )
    return {
        "adapter_id": semantics.adapter_id,
        "digest": semantics.digest().hex(),
        "layers": [
            {
                "layer_index": item.layer_index,
                "transition_parameters": item.transition_parameters.hex(),
                "runtime_encoding_id": item.runtime_encoding_id,
                "conv_state_encoding_id": item.conv_state_encoding_id,
                "recurrent_state_encoding_id":
                    item.recurrent_state_encoding_id,
                "output_atol_q24": item.output_atol_q24,
                "output_rtol_q24": item.output_rtol_q24,
                **(
                    {
                        "conv_state_atol_q24":
                            item.conv_state_atol_q24,
                        "recurrent_state_atol_q24":
                            item.recurrent_state_atol_q24,
                    }
                    if semantics.version in (
                        GDN_RUNTIME_CHECKPOINT_SEMANTICS_VERSION_V3,
                        GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3,
                    )
                    else {}
                ),
                "max_decode_replay_rows": item.max_decode_replay_rows,
                **(
                    {
                        "decode_checkpoint_stride":
                            item.decode_checkpoint_stride
                    }
                    if semantics.version in (
                        GDN_RUNTIME_CHECKPOINT_SEMANTICS_VERSION_V3,
                        GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3,
                    )
                    else {}
                ),
                **(
                    {
                        "prefix_cache_output_atol_q24":
                            item.prefix_cache_output_atol_q24,
                        "prefix_cache_output_rtol_q24":
                            item.prefix_cache_output_rtol_q24,
                        "prefix_cache_conv_state_atol_q24":
                            item.prefix_cache_conv_state_atol_q24,
                        "prefix_cache_recurrent_state_atol_q24":
                            item.prefix_cache_recurrent_state_atol_q24,
                        "max_prefix_cache_replay_rows":
                            item.max_prefix_cache_replay_rows,
                    }
                    if semantics.version
                    == GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3
                    else {}
                ),
            }
            for item in semantics.layers
        ],
        "version": semantics.version,
        **(
            {
                "max_hard_audit_decode_tokens":
                    semantics.max_hard_audit_decode_tokens
            }
            if semantics.version in (
                GDN_RUNTIME_CHECKPOINT_SEMANTICS_VERSION_V3,
                GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3,
            )
            else {}
        ),
    }


def load_gdn_runtime_semantics_v3(source) -> GdnRuntimeSemanticsV3:
    if isinstance(source, dict):
        value = source
    elif isinstance(source, bytes):
        value = json.loads(source.decode("utf-8"))
    elif isinstance(source, str):
        stripped = source.lstrip()
        if stripped.startswith("{"):
            value = json.loads(source)
        else:
            with open(source, "rb") as handle:
                value = json.load(handle)
    else:
        raise ProofV3Error("GDN runtime semantics source is unsupported")
    if not isinstance(value, dict):
        raise ProofV3Error("GDN runtime semantics object is malformed")
    try:
        version = int(value["version"])
        expected_keys = {
            "adapter_id",
            "digest",
            "layers",
            "version",
        }
        if version in (
            GDN_RUNTIME_CHECKPOINT_SEMANTICS_VERSION_V3,
            GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3,
        ):
            expected_keys.add("max_hard_audit_decode_tokens")
        if set(value) != expected_keys:
            raise ProofV3Error(
                "GDN runtime semantics object is malformed"
            )
        checkpointed = version in (
            GDN_RUNTIME_CHECKPOINT_SEMANTICS_VERSION_V3,
            GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3,
        )
        prefix_cache = (
            version == GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3
        )
        expected_layer_keys = {
            "layer_index",
            "transition_parameters",
            "runtime_encoding_id",
            "conv_state_encoding_id",
            "recurrent_state_encoding_id",
            "output_atol_q24",
            "output_rtol_q24",
            "max_decode_replay_rows",
        }
        if checkpointed:
            expected_layer_keys.update(
                {
                    "conv_state_atol_q24",
                    "recurrent_state_atol_q24",
                    "decode_checkpoint_stride",
                }
            )
        if prefix_cache:
            expected_layer_keys.update(
                {
                    "prefix_cache_output_atol_q24",
                    "prefix_cache_output_rtol_q24",
                    "prefix_cache_conv_state_atol_q24",
                    "prefix_cache_recurrent_state_atol_q24",
                    "max_prefix_cache_replay_rows",
                }
            )
        if (
            not isinstance(value["layers"], list)
            or any(
                not isinstance(item, dict)
                or set(item) != expected_layer_keys
                for item in value["layers"]
            )
        ):
            raise ProofV3Error(
                "GDN runtime layer semantics object is malformed"
            )
        semantics = GdnRuntimeSemanticsV3(
            adapter_id=value["adapter_id"],
            layers=tuple(
                GdnLayerRuntimeSemanticsV3(
                    layer_index=int(item["layer_index"]),
                    transition_parameters=bytes.fromhex(
                        item["transition_parameters"]
                    ),
                    runtime_encoding_id=item["runtime_encoding_id"],
                    conv_state_encoding_id=item["conv_state_encoding_id"],
                    recurrent_state_encoding_id=(
                        item["recurrent_state_encoding_id"]
                    ),
                    output_atol_q24=int(item["output_atol_q24"]),
                    output_rtol_q24=int(item["output_rtol_q24"]),
                    conv_state_atol_q24=(
                        int(item["conv_state_atol_q24"])
                        if checkpointed
                        else int(item["output_atol_q24"])
                    ),
                    recurrent_state_atol_q24=(
                        int(item["recurrent_state_atol_q24"])
                        if checkpointed
                        else int(item["output_atol_q24"])
                    ),
                    max_decode_replay_rows=int(
                        item["max_decode_replay_rows"]
                    ),
                    decode_checkpoint_stride=(
                        int(item["decode_checkpoint_stride"])
                        if checkpointed
                        else 0
                    ),
                    prefix_cache_output_atol_q24=(
                        int(item["prefix_cache_output_atol_q24"])
                        if prefix_cache else 0
                    ),
                    prefix_cache_output_rtol_q24=(
                        int(item["prefix_cache_output_rtol_q24"])
                        if prefix_cache else 0
                    ),
                    prefix_cache_conv_state_atol_q24=(
                        int(item["prefix_cache_conv_state_atol_q24"])
                        if prefix_cache else 0
                    ),
                    prefix_cache_recurrent_state_atol_q24=(
                        int(item["prefix_cache_recurrent_state_atol_q24"])
                        if prefix_cache else 0
                    ),
                    max_prefix_cache_replay_rows=(
                        int(item["max_prefix_cache_replay_rows"])
                        if prefix_cache else 0
                    ),
                )
                for item in value["layers"]
            ),
            version=version,
            max_hard_audit_decode_tokens=(
                int(value["max_hard_audit_decode_tokens"])
                if checkpointed
                else 0
            ),
        )
        declared = bytes.fromhex(value["digest"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProofV3Error("GDN runtime semantics object is malformed") from exc
    if declared != semantics.digest():
        raise ProofV3Error("GDN runtime semantics digest does not match")
    return semantics
