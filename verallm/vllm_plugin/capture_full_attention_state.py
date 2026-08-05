"""Capture logical Qwen full-attention K/V state at the prompt boundary.

vLLM stores full-attention K/V in paged physical blocks. The proof protocol
uses a request-local logical ``[token, kv_head, head_dim]`` history instead,
so the wrapper resolves the scheduler-owned block table before the request can
be challenged. Unsupported cache layouts are deliberately not inferred.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _metadata_for_layer(context, layer_name: str):
    """Return one unambiguous full-attention metadata record."""

    metadata = getattr(context, "attn_metadata", None)
    if isinstance(metadata, dict):
        return metadata.get(layer_name)
    # Dual-batch overlap has a distinct request-to-metadata mapping ABI. It
    # must be qualified explicitly instead of guessing an ordering here.
    return None


def _tensor_metadata_value(metadata, *names: str) -> torch.Tensor:
    for name in names:
        value = getattr(metadata, name, None)
        if isinstance(value, torch.Tensor):
            return value
    raise RuntimeError(f"proof-v2 full-attention metadata has no {names[0]}")


def _logical_full_attention_kv_history(
    cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve one logical request history from vLLM's FP16 paged cache."""

    if (
        not isinstance(cache, torch.Tensor)
        or cache.ndim != 5
        or int(cache.shape[0]) != 2
        or cache.dtype != torch.float16
        or not cache.is_floating_point()
    ):
        raise RuntimeError("proof-v2 full-attention cache ABI is unsupported")
    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or sequence_length <= 0
    ):
        raise RuntimeError("proof-v2 full-attention sequence length is invalid")
    if not isinstance(block_table, torch.Tensor) or block_table.ndim != 1:
        raise RuntimeError("proof-v2 full-attention block table is invalid")
    block_size = int(cache.shape[2])
    required_blocks = (sequence_length + block_size - 1) // block_size
    if required_blocks > int(block_table.numel()):
        raise RuntimeError("proof-v2 full-attention block table is truncated")
    physical_blocks = (
        block_table[:required_blocks]
        .detach()
        .to(
            device=cache.device,
            dtype=torch.long,
        )
    )
    if (
        physical_blocks.numel() != required_blocks
        or torch.any(physical_blocks < 0)
        or torch.any(physical_blocks >= int(cache.shape[1]))
        or int(torch.unique(physical_blocks).numel()) != required_blocks
    ):
        raise RuntimeError("proof-v2 full-attention block table is noncanonical")
    positions = torch.arange(sequence_length, device=cache.device)
    block_indices = physical_blocks.index_select(0, positions // block_size)
    offsets = positions % block_size
    keys = cache[0, block_indices, offsets]
    values = cache[1, block_indices, offsets]
    if (
        keys.ndim != 3
        or values.shape != keys.shape
        or keys.dtype != torch.float16
        or not torch.isfinite(keys).all()
        or not torch.isfinite(values).all()
    ):
        raise RuntimeError("proof-v2 full-attention cache values are invalid")
    return keys.contiguous(), values.contiguous()


class CaptureFullAttentionStateWrapper(nn.Module):
    """Transparent post-prefill logical K/V capture for one Qwen layer."""

    def __init__(self, original: nn.Module, layer_idx: int):
        super().__init__()
        self.original = original
        self._layer_idx = layer_idx

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original, name)

    def forward(self, *args, **kwargs):
        result = self.original(*args, **kwargs)
        self._capture_after_prefill()
        return result

    def _capture_after_prefill(self) -> None:
        try:
            from vllm.forward_context import get_forward_context
            from verallm.vllm_plugin.ops import get_active_tracker

            tracker = get_active_tracker()
            if tracker is None or not tracker._current_req_ids:
                return
            if not tracker.has_full_attention_transition_capture():
                return
            attention = getattr(self.original, "attn", None)
            layer_name = getattr(attention, "layer_name", None)
            cache = getattr(attention, "kv_cache", None)
            if not isinstance(layer_name, str) or not layer_name:
                raise RuntimeError("proof-v2 full-attention layer name is unavailable")
            context = get_forward_context()
            metadata = _metadata_for_layer(context, layer_name)
            if metadata is None:
                raise RuntimeError("proof-v2 full-attention metadata is unavailable")
            prefill_indices = tracker.full_attention_transition_prefill_indices()
            if not prefill_indices:
                return
            request_count = len(tracker._current_req_ids)
            if (
                tuple(sorted(set(prefill_indices))) != tuple(prefill_indices)
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < request_count
                    for index in prefill_indices
                )
            ):
                raise RuntimeError("proof-v2 full-attention prefill indices are invalid")
            sequence_lengths = _tensor_metadata_value(
                metadata,
                "seq_lens_cpu",
                "seq_lens",
            )
            block_table = _tensor_metadata_value(
                metadata,
                "block_table_tensor",
                "block_table",
            )
            if (
                sequence_lengths.ndim != 1
                or int(sequence_lengths.numel()) != request_count
                or block_table.ndim != 2
                or int(block_table.shape[0]) != request_count
            ):
                raise RuntimeError(
                    "proof-v2 full-attention request mapping is unsupported"
                )
            for request_index in prefill_indices:
                sequence_length = int(sequence_lengths[request_index].item())
                keys, values = _logical_full_attention_kv_history(
                    cache,
                    block_table=block_table[request_index],
                    sequence_length=sequence_length,
                )
                tracker.capture_full_attention_prompt_boundary(
                    layer_idx=self._layer_idx,
                    request_index=request_index,
                    keys=keys,
                    values=values,
                )
        except Exception as exc:
            # Failure is retained as missing boundary material. The hard proof
            # cannot then construct its exact opening and fails closed.
            logger.warning(
                "proof-v2 full-attention boundary capture skipped for layer %d: %s",
                self._layer_idx,
                exc,
            )


def wrap_qwen_full_attention_state_modules(layers) -> int:
    """Wrap structurally compatible Qwen full-attention modules once."""

    count = 0
    for layer_idx, layer in enumerate(layers):
        owner = layer
        while isinstance(getattr(owner, "original", None), nn.Module):
            owner = owner.original
        original = getattr(owner, "self_attn", None)
        attention = getattr(original, "attn", None)
        if original is None or isinstance(original, CaptureFullAttentionStateWrapper):
            continue
        if not (
            isinstance(attention, nn.Module)
            and hasattr(attention, "kv_cache")
            and hasattr(attention, "layer_name")
            and hasattr(original, "qkv_proj")
            and hasattr(original, "o_proj")
        ):
            continue
        setattr(
            owner,
            "self_attn",
            CaptureFullAttentionStateWrapper(original, layer_idx),
        )
        count += 1
    return count


__all__ = [
    "CaptureFullAttentionStateWrapper",
    "_logical_full_attention_kv_history",
    "wrap_qwen_full_attention_state_modules",
]
