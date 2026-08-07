"""Build the signed attention-runtime artifact from a qualified model.

This module is release/qualification tooling.  The resulting small artifact is
public and weightless-validator consumable; model weights are inspected only
while the model is being qualified.
"""

from __future__ import annotations

from verallm.proof_v3.attention_runtime_semantics import (
    ATTENTION_RUNTIME_SEMANTICS_VERSION_V3,
    AttentionNormBindingV3,
    AttentionRuntimeSemanticsV3,
    GEMMA_RMS_NORM_V3,
    LOGICAL_PAGED_KV_V3,
    NEOX_ROPE_V3,
    NO_QK_NORM_V3,
    QKV_CONTIGUOUS_LAYOUT_V3,
    Q_GATE_INTERLEAVED_LAYOUT_V3,
)
from verallm.proof_v3.capture_extraction import capture_model_semantics_v3
from verallm.proof_v3.errors import ProofV3Error

__all__ = ["qualify_attention_runtime_semantics_v3"]


def _unwrap_attention(layer):
    import torch.nn as nn

    owner = layer
    while (
        isinstance(getattr(owner, "original", None), nn.Module)
        and not hasattr(owner, "self_attn")
    ):
        owner = owner.original
    attention = getattr(owner, "self_attn", None)
    while (
        attention is not None
        and isinstance(getattr(attention, "original", None), nn.Module)
        and not hasattr(attention, "q_norm")
    ):
        attention = attention.original
    if attention is None:
        raise ProofV3Error(
            "qualified full-attention layer has no attention module"
        )
    return attention


def _norm_bytes(module, *, head_dim: int):
    import torch

    weight = getattr(module, "weight", None)
    if (
        not isinstance(weight, torch.Tensor)
        or weight.ndim != 1
        or int(weight.numel()) != int(head_dim)
        or weight.dtype not in (torch.float16, torch.bfloat16)
        or not bool(torch.isfinite(weight).all())
    ):
        raise ProofV3Error(
            "qualified attention normalization weights are unsupported"
        )
    encoding = (
        "fp16.v1" if weight.dtype == torch.float16 else "bf16.v1"
    )
    raw = (
        weight.detach()
        .cpu()
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    epsilon = float(
        getattr(
            module,
            "variance_epsilon",
            getattr(module, "eps", 1e-6),
        )
    )
    return raw, encoding, epsilon


def _qualified_rope_table(*, layers, selected, rotary_dimension, row_count):
    """Freeze the exact runtime BF16/FP16 coefficient table.

    vLLM creates this table on the accelerator. Rare float32 sin/cos ULP
    differences are observable after fixed-point K quantization, so rebuilding
    it with a validator's libm is not an exact execution binding.
    """

    import torch

    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 1
    ):
        raise ProofV3Error(
            "qualified attention RoPE row count is malformed"
        )
    reference = None
    encoding = None
    for layer_index in selected:
        attention = _unwrap_attention(layers[layer_index])
        rotary = getattr(attention, "rotary_emb", None)
        table = getattr(rotary, "cos_sin_cache", None)
        if (
            not isinstance(table, torch.Tensor)
            or table.ndim != 2
            or int(table.shape[0]) < row_count
            or int(table.shape[1]) != int(rotary_dimension)
            or table.dtype not in (torch.float16, torch.bfloat16)
            or not bool(torch.isfinite(table[:row_count]).all())
        ):
            raise ProofV3Error(
                f"qualified attention layer {layer_index} has no exact "
                "runtime RoPE table"
            )
        current_encoding = (
            "fp16.v1" if table.dtype == torch.float16 else "bf16.v1"
        )
        # Qualification may shard layers across devices. Compare the exact
        # materialized bytes on the host rather than invoking a cross-device
        # equality kernel.
        current = table[:row_count].detach().cpu().contiguous()
        if reference is None:
            reference = current
            encoding = current_encoding
        elif (
            encoding != current_encoding
            or not bool(torch.equal(reference, current))
        ):
            raise ProofV3Error(
                "qualified full-attention layers have different runtime "
                "RoPE tables"
            )
    if reference is None or encoding is None:
        raise ProofV3Error("qualified attention RoPE inventory is empty")
    raw = bytes(
        reference.view(torch.uint8).numpy().tobytes()
    )
    if len(raw) != row_count * int(rotary_dimension) * 2:
        raise ProofV3Error(
            "qualified attention RoPE table has a wrong byte length"
        )
    return raw, encoding


def qualify_attention_runtime_semantics_v3(
    *,
    config,
    layers,
    full_attention_layers,
    rope_position_count: int,
    adapter_id: str = "qwen.paged_attention.v1",
    integer_tolerance: int = 0,
) -> AttentionRuntimeSemanticsV3:
    """Derive the canonical runtime adapter artifact during qualification."""

    semantics = capture_model_semantics_v3(config)
    selected = tuple(int(layer) for layer in full_attention_layers)
    if (
        not selected
        or selected != tuple(sorted(set(selected)))
        or any(layer < 0 or layer >= len(layers) for layer in selected)
    ):
        raise ProofV3Error(
            "qualified full-attention layer inventory is malformed"
        )
    rope_bytes, rope_encoding = _qualified_rope_table(
        layers=layers,
        selected=selected,
        rotary_dimension=semantics.rot_dim,
        row_count=rope_position_count,
    )
    if not semantics.gated:
        return AttentionRuntimeSemanticsV3(
            adapter_id=adapter_id,
            qkv_layout_id=QKV_CONTIGUOUS_LAYOUT_V3,
            q_norm_id=NO_QK_NORM_V3,
            k_norm_id=NO_QK_NORM_V3,
            q_norm_epsilon=0.0,
            k_norm_epsilon=0.0,
            rope_id=NEOX_ROPE_V3,
            rope_theta=semantics.theta,
            rotary_dimension=semantics.rot_dim,
            cache_layout_id=LOGICAL_PAGED_KV_V3,
            norm_encoding_id=rope_encoding,
            integer_tolerance=integer_tolerance,
            rope_coefficient_row_count=rope_position_count,
            rope_coefficient_encoding_id=rope_encoding,
            rope_coefficient_bytes=rope_bytes,
            version=ATTENTION_RUNTIME_SEMANTICS_VERSION_V3,
        )

    bindings = []
    encoding = None
    q_epsilon = None
    k_epsilon = None
    for layer_index in selected:
        attention = _unwrap_attention(layers[layer_index])
        q_norm = getattr(attention, "q_norm", None)
        k_norm = getattr(attention, "k_norm", None)
        if q_norm is None or k_norm is None:
            raise ProofV3Error(
                f"gated attention layer {layer_index} has no explicit Q/K "
                "normalization"
            )
        q_raw, q_encoding, q_eps = _norm_bytes(
            q_norm, head_dim=semantics.hd
        )
        k_raw, k_encoding, k_eps = _norm_bytes(
            k_norm, head_dim=semantics.hd
        )
        if q_encoding != k_encoding:
            raise ProofV3Error(
                "qualified Q/K normalization encodings disagree"
            )
        if encoding is None:
            encoding = q_encoding
            q_epsilon = q_eps
            k_epsilon = k_eps
        elif (
            encoding != q_encoding
            or q_epsilon != q_eps
            or k_epsilon != k_eps
        ):
            raise ProofV3Error(
                "qualified gated-attention normalization ABI varies by layer"
            )
        bindings.append(
            AttentionNormBindingV3(
                layer_index=layer_index,
                q_weight_bytes=q_raw,
                k_weight_bytes=k_raw,
            )
        )
    return AttentionRuntimeSemanticsV3(
        adapter_id=adapter_id,
        qkv_layout_id=Q_GATE_INTERLEAVED_LAYOUT_V3,
        q_norm_id=GEMMA_RMS_NORM_V3,
        k_norm_id=GEMMA_RMS_NORM_V3,
        q_norm_epsilon=float(q_epsilon),
        k_norm_epsilon=float(k_epsilon),
        rope_id=NEOX_ROPE_V3,
        rope_theta=semantics.theta,
        rotary_dimension=semantics.rot_dim,
        cache_layout_id=LOGICAL_PAGED_KV_V3,
        norm_encoding_id=str(encoding),
        norm_bindings=tuple(bindings),
        integer_tolerance=integer_tolerance,
        rope_coefficient_row_count=rope_position_count,
        rope_coefficient_encoding_id=rope_encoding,
        rope_coefficient_bytes=rope_bytes,
        version=ATTENTION_RUNTIME_SEMANTICS_VERSION_V3,
    )
