"""Nonce-selected GDN recurrence windows over authenticated cache blocks."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from verallm.proof_v3.economic_challenge import EconomicChallengeV3
from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.execution_anchor import execution_anchor_lane_bytes_v3
from verallm.proof_v3.gdn_runtime_semantics import GdnRuntimeSemanticsV3
from verallm.proof_v3.prefix_cache import PrefixCacheCommitmentV3

PREFIX_CACHE_GDN_WINDOW_ABI_V3 = "prefix_cache.gdn_block_window.v1"
_WINDOW_DOMAIN = b"VERATHOS/PROOF_V3/PREFIX_CACHE/GDN_WINDOW/V1"

__all__ = [
    "PREFIX_CACHE_GDN_WINDOW_ABI_V3",
    "PrefixCacheGdnWindowV3",
    "derive_prefix_cache_gdn_windows_v3",
    "select_prefix_cache_gdn_layer_v3",
    "required_prefix_cache_gdn_lane_keys_v3",
]


def select_prefix_cache_gdn_layer_v3(
    *, selection_seed: bytes, candidate_layers
) -> tuple[int, ...]:
    """Select one GDN layer from the already-audited hard corridor."""

    layers = tuple(int(layer) for layer in candidate_layers)
    if (
        not isinstance(selection_seed, bytes)
        or len(selection_seed) != 32
        or selection_seed == bytes(32)
        or not layers
        or layers != tuple(sorted(set(layers)))
        or any(layer < 0 or layer >= 1 << 32 for layer in layers)
    ):
        raise ProofV3Error("prefix-cache GDN layer selection is malformed")
    digest = hashlib.sha256(
        _WINDOW_DOMAIN
        + b"/LAYER"
        + selection_seed
        + b"".join(struct.pack("<I", layer) for layer in layers)
    ).digest()
    return (layers[int.from_bytes(digest[:8], "little") % len(layers)],)


@dataclass(frozen=True, slots=True)
class PrefixCacheGdnWindowV3:
    """One full logical cache block selected after the hard nonce."""

    layer_index: int
    block_index: int
    block_token_count: int
    cached_token_count: int

    def __post_init__(self) -> None:
        if (
            type(self.layer_index) is not int
            or not 0 <= self.layer_index < 1 << 32
            or type(self.block_index) is not int
            or self.block_index < 0
            or type(self.block_token_count) is not int
            or self.block_token_count < 1
            or type(self.cached_token_count) is not int
            or self.cached_token_count < self.block_token_count
            or self.cached_token_count % self.block_token_count
            or self.block_index >= self.cached_token_count // self.block_token_count
        ):
            raise ProofV3Error("prefix-cache GDN window is malformed")

    @property
    def start_position(self) -> int:
        return self.block_index * self.block_token_count

    @property
    def end_position(self) -> int:
        return self.start_position + self.block_token_count

    @property
    def sequence_positions(self) -> tuple[int, ...]:
        return tuple(range(self.start_position, self.end_position))

    @property
    def start_state_block(self) -> int | None:
        return None if self.block_index == 0 else self.block_index - 1

    @property
    def end_state_block(self) -> int:
        return self.block_index


def derive_prefix_cache_gdn_windows_v3(
    *,
    validator_nonce: bytes,
    commitment: PrefixCacheCommitmentV3,
    selected_gdn_layers,
) -> tuple[PrefixCacheGdnWindowV3, ...]:
    """Select one complete cached recurrence block for every GDN layer."""

    if (
        not isinstance(validator_nonce, bytes)
        or len(validator_nonce) != 32
        or validator_nonce == bytes(32)
        or not isinstance(commitment, PrefixCacheCommitmentV3)
        or commitment.cached_token_count % commitment.block_token_count
    ):
        raise ProofV3Error("prefix-cache GDN window source is malformed")
    layers = tuple(int(layer) for layer in selected_gdn_layers)
    if (
        not layers
        or layers != tuple(sorted(set(layers)))
        or any(layer < 0 or layer >= 1 << 32 for layer in layers)
    ):
        raise ProofV3Error("prefix-cache GDN layer selection is malformed")
    block_count = commitment.cached_token_count // commitment.block_token_count
    result = []
    for layer in layers:
        digest = hashlib.sha256(
            _WINDOW_DOMAIN
            + validator_nonce
            + commitment.digest()
            + struct.pack("<I", layer)
        ).digest()
        result.append(PrefixCacheGdnWindowV3(
            layer_index=layer,
            block_index=int.from_bytes(digest[:8], "little") % block_count,
            block_token_count=commitment.block_token_count,
            cached_token_count=commitment.cached_token_count,
        ))
    return tuple(result)


def required_prefix_cache_gdn_lane_keys_v3(
    *,
    windows,
    challenge: EconomicChallengeV3,
    semantics: GdnRuntimeSemanticsV3,
) -> tuple[tuple[int, str, int, int], ...]:
    """Open exact start/end state lanes consumed by selected-head replay."""

    plans = tuple(windows)
    if (
        not isinstance(challenge, EconomicChallengeV3)
        or not isinstance(semantics, GdnRuntimeSemanticsV3)
        or not plans
        or not all(isinstance(plan, PrefixCacheGdnWindowV3) for plan in plans)
        or tuple(plan.layer_index for plan in plans)
        != tuple(sorted({plan.layer_index for plan in plans}))
    ):
        raise ProofV3Error("prefix-cache GDN lane source is malformed")
    keys: set[tuple[int, str, int, int]] = set()
    for plan in plans:
        signed = semantics.layer_for(plan.layer_index)
        parameters = signed.parameters().replay_parameters()
        heads = challenge.gdn_value_heads_for(
            layer_index=plan.layer_index,
            num_key_heads=parameters.num_key_heads,
            num_value_heads=parameters.num_value_heads,
        )
        recurrent_element_bytes = signed.recurrent_state_bytes // (
            parameters.num_value_heads
            * parameters.value_head_dim
            * parameters.key_head_dim
        )
        recurrent_head_bytes = (
            parameters.value_head_dim
            * parameters.key_head_dim
            * recurrent_element_bytes
        )
        state_blocks = tuple(
            block
            for block in (plan.start_state_block, plan.end_state_block)
            if block is not None
        )
        conv_stage = f"l{plan.layer_index}.gdn_conv_boundary"
        recurrent_stage = f"l{plan.layer_index}.gdn_recurrent_boundary"
        conv_lane_bytes = execution_anchor_lane_bytes_v3(conv_stage)
        recurrent_lane_bytes = execution_anchor_lane_bytes_v3(recurrent_stage)
        for block in state_blocks:
            keys.update(
                (block, conv_stage, 0, lane)
                for lane in range(
                    (signed.conv_state_bytes + conv_lane_bytes - 1)
                    // conv_lane_bytes
                )
            )
            for head in heads:
                start = head * recurrent_head_bytes
                keys.update(
                    (block, recurrent_stage, 0, lane)
                    for lane in range(
                        start // recurrent_lane_bytes,
                        (start + recurrent_head_bytes - 1)
                        // recurrent_lane_bytes
                        + 1,
                    )
                )
    return tuple(sorted(keys))
