"""Build the signed attention-runtime artifact from a qualified model.

This module is release/qualification tooling.  The resulting small artifact is
public and weightless-validator consumable; model weights are inspected only
while the model is being qualified.
"""

from __future__ import annotations

from verallm.proof_v3.attention_runtime_semantics import (
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


def qualify_attention_runtime_semantics_v3(
    *,
    config,
    layers,
    full_attention_layers,
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
            norm_encoding_id="bf16.v1",
            integer_tolerance=integer_tolerance,
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
    )
