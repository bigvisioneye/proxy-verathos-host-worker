"""Capture the live Qwen GDN cache at the prompt/decode boundary.

The wrapper is installed before vLLM graph construction but is inert unless a
request explicitly negotiated the supported GDN transition profile.  It never
alters the model result: it snapshots only the post-prefill cache entries that
vLLM already maintains for the request.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _gdn_cache_pair(module: nn.Module, context) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two live GDN cache tensors across supported vLLM ABIs.

    vLLM 0.19 stores ``(conv, recurrent)`` directly on the module.  Older
    releases keep that pair inside a virtual-engine container.  Both layouts
    are read-only here and anything else is intentionally rejected.
    """

    cache = module.kv_cache
    if (
        isinstance(cache, tuple)
        and len(cache) == 2
        and all(isinstance(item, torch.Tensor) for item in cache)
    ):
        return cache
    virtual_engine = getattr(context, "virtual_engine", None)
    if virtual_engine is None:
        raise RuntimeError("proof-v2 GDN cache has no virtual-engine selector")
    try:
        pair = cache[virtual_engine]
    except (IndexError, KeyError, TypeError) as exc:
        raise RuntimeError("proof-v2 GDN virtual-engine cache is unavailable") from exc
    if (
        not isinstance(pair, tuple)
        or len(pair) != 2
        or not all(isinstance(item, torch.Tensor) for item in pair)
    ):
        raise RuntimeError("proof-v2 GDN cache layout is unsupported")
    return pair


class CaptureGDNStateWrapper(nn.Module):
    """Transparent post-prefill state capture around one Qwen GDN module."""

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
        """Capture only a metadata layout that we can identify exactly."""

        try:
            from vllm.forward_context import get_forward_context
            from verallm.vllm_plugin.ops import get_active_tracker

            tracker = get_active_tracker()
            if tracker is None or not tracker._current_req_ids:
                return
            if not tracker.has_gdn_transition_capture():
                return
            context = get_forward_context()
            metadata = context.attn_metadata
            if not isinstance(metadata, dict):
                return
            metadata = metadata.get(self.original.prefix)
            if metadata is None:
                return
            # The transition witness currently supports only the ordinary
            # non-speculative cache.  Speculative layouts use multiple cache
            # positions per sequence and need their own authenticated ABI.
            if metadata.spec_sequence_masks is not None:
                return
            state_indices = metadata.non_spec_state_indices_tensor
            if state_indices is None or state_indices.ndim != 1:
                return
            request_count = len(tracker._current_req_ids)
            if int(state_indices.numel()) != request_count:
                return
            prefill_indices = tracker.gdn_transition_prefill_indices()
            if (
                not isinstance(prefill_indices, tuple)
                or not prefill_indices
                or tuple(sorted(set(prefill_indices))) != prefill_indices
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < request_count
                    for index in prefill_indices
                )
            ):
                return
            selection = torch.tensor(
                prefill_indices,
                dtype=torch.long,
                device=state_indices.device,
            )
            slots = state_indices.index_select(0, selection)
            cache = _gdn_cache_pair(self.original, context)
            # vLLM stores this cache in its canonical runtime order
            # ``[slot, kernel - 1, conv_width]``.  The core temporarily
            # transposes it only for the causal-convolution kernel; the
            # verifier replay and signed ABI use the stored order.
            conv_cache = cache[0]
            recurrent_cache = cache[1]
            conv_states = conv_cache.index_select(0, slots)
            recurrent_states = recurrent_cache.index_select(0, slots)
            tracker.capture_gdn_prompt_boundary(
                layer_idx=self._layer_idx,
                request_indices=prefill_indices,
                conv_states=conv_states,
                recurrent_states=recurrent_states,
            )
        except Exception as exc:
            # A capture error must not silently become a valid transition
            # proof: the miner lacks the opening and the hard verifier rejects
            # it.  Keep serving behavior intact while preserving the detail in
            # logs for the unsupported runtime profile.
            logger.warning(
                "proof-v2 GDN boundary capture skipped for layer %d: %s",
                self._layer_idx,
                exc,
            )


def wrap_qwen_gdn_state_modules(layers) -> int:
    """Wrap structurally compatible Qwen GDN modules once per loaded model."""

    count = 0
    for layer_idx, layer in enumerate(layers):
        # The dense residual wrapper is installed first.  Replace the module
        # on the underlying decoder, not on that transparent outer wrapper.
        owner = layer
        while isinstance(getattr(owner, "original", None), nn.Module):
            owner = owner.original
        original = getattr(owner, "linear_attn", None)
        if original is None or isinstance(original, CaptureGDNStateWrapper):
            continue
        if not (
            hasattr(original, "kv_cache")
            and hasattr(original, "prefix")
            and hasattr(original, "in_proj_qkvz")
            and hasattr(original, "in_proj_ba")
        ):
            continue
        setattr(owner, "linear_attn", CaptureGDNStateWrapper(original, layer_idx))
        count += 1
    return count


__all__ = [
    "CaptureGDNStateWrapper",
    "_gdn_cache_pair",
    "wrap_qwen_gdn_state_modules",
]
