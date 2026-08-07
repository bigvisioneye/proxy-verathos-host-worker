"""Interpret signed raw QKV execution anchors for attention proofs."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

from verallm.proof_v3.attention_runtime_semantics import (
    ATTENTION_RUNTIME_SEMANTICS_VERSION_V3,
    AttentionRuntimeSemanticsV3,
    GEMMA_RMS_NORM_V3,
    NO_QK_NORM_V3,
    QKV_CONTIGUOUS_LAYOUT_V3,
    Q_GATE_INTERLEAVED_LAYOUT_V3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.execution_anchor import (
    ExecutionAnchorCommitmentV3,
    ExecutionAnchorLaneOpeningV3,
    execution_anchor_lane_bytes_v3,
    verify_execution_anchor_lane_v3,
)

__all__ = [
    "AttentionAnchorGeometryV3",
    "RuntimeAttentionQuantizedRowV3",
    "attention_anchor_geometry_v3",
    "attention_anchor_head_byte_range_v3",
    "canonical_attention_qkv_output_columns_v3",
    "decode_runtime_values_v3",
    "extract_execution_anchor_range_v3",
    "required_attention_anchor_lane_keys_v3",
    "required_attention_anchor_lane_keys_for_sections_v3",
    "required_prefix_cache_attention_lane_keys_v3",
    "required_prefix_cache_attention_lane_keys_for_sections_v3",
    "derive_prefix_cache_projection_heads_v3",
    "required_prefix_cache_projection_heads_for_sections_v3",
    "required_execution_anchor_lanes_v3",
    "runtime_attention_q_head_quantized_v3",
    "runtime_attention_q13_pool_v3",
    "runtime_attention_quantized_row_v3",
    "runtime_kv_head_quantized_v3",
]


def canonical_attention_qkv_output_columns_v3(
    *,
    plans,
    geometries_by_layer: Mapping[int, "AttentionAnchorGeometryV3"],
    prefix_projection_heads=(),
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Derive one canonical QKV column inventory for weights and claims."""

    plans = tuple(plans)
    prefix_heads_by_layer: dict[int, set[int]] = {}
    try:
        for layer, kv_head, _position in tuple(prefix_projection_heads):
            prefix_heads_by_layer.setdefault(int(layer), set()).add(
                int(kv_head)
            )
    except (TypeError, ValueError) as exc:
        raise ProofV3Error(
            "attention prefix-cache projection heads are malformed"
        ) from exc

    result = []
    seen_layers = set()
    for plan in plans:
        layer = int(plan.layer)
        if layer in seen_layers:
            raise ProofV3Error("attention QKV plan layers are duplicated")
        seen_layers.add(layer)
        geometry = geometries_by_layer.get(layer)
        if not isinstance(geometry, AttentionAnchorGeometryV3):
            raise ProofV3Error("attention QKV geometry is incomplete")
        query_stride = geometry.head_dim * (2 if geometry.gated else 1)
        columns = {
            column
            for head in plan.heads
            for column in range(
                int(head) * query_stride,
                (int(head) + 1) * query_stride,
            )
        }
        for kv_head in prefix_heads_by_layer.get(layer, ()):
            if kv_head < 0 or kv_head >= geometry.kv_heads:
                raise ProofV3Error(
                    "attention prefix-cache projection head is out of range"
                )
            for base in (
                geometry.q_block_width,
                geometry.v_block_offset,
            ):
                columns.update(range(
                    base + kv_head * geometry.head_dim,
                    base + (kv_head + 1) * geometry.head_dim,
                ))
        if (
            not columns
            or min(columns) < 0
            or max(columns) >= geometry.qkv_width
        ):
            raise ProofV3Error(
                "attention QKV output columns are malformed"
            )
        result.append((layer, tuple(sorted(columns))))
    if set(prefix_heads_by_layer) - seen_layers:
        raise ProofV3Error(
            "attention prefix-cache projection heads reference an "
            "unselected layer"
        )
    return tuple(result)


def required_attention_anchor_lane_keys_v3(
    *,
    bundle_wire: bytes,
    plans,
    commitment_indices_by_layer: Mapping[int, int],
    geometries_by_layer: Mapping[int, "AttentionAnchorGeometryV3"],
    key_count: int,
    cached_token_count: int = 0,
) -> tuple[tuple[int, int, int], ...]:
    """Derive the exact raw-QKV lane openings required by one bundle.

    The succinct tile chooses K/V equality cells from the nonce-bound
    transcript.  This maps those cells back to the graph-integrated raw QKV
    anchor rows and 2-KiB lanes.  The miner and verifier therefore use the
    same canonical coordinate set; no payload-supplied row or lane is trusted.
    """

    if (
        not isinstance(bundle_wire, bytes)
        or not bundle_wire
        or type(key_count) is not int
        or key_count < 1
    ):
        raise ProofV3Error(
            "attention anchor lane-key source is malformed"
        )
    from verallm.proof_v3.succinct_attention_wire import (
        decode_anchor_capture_kv_bundle_wire_v3,
    )

    plans = tuple(plans)
    layers = tuple(int(plan.layer) for plan in plans)
    sections = decode_anchor_capture_kv_bundle_wire_v3(
        bundle_wire,
        expected_layers=layers,
    )
    return required_attention_anchor_lane_keys_for_sections_v3(
        sections=sections,
        plans=plans,
        commitment_indices_by_layer=commitment_indices_by_layer,
        geometries_by_layer=geometries_by_layer,
        key_count=key_count,
        cached_token_count=cached_token_count,
    )


def required_attention_anchor_lane_keys_for_sections_v3(
    *,
    sections,
    plans,
    commitment_indices_by_layer: Mapping[int, int],
    geometries_by_layer: Mapping[int, "AttentionAnchorGeometryV3"],
    key_count: int,
    cached_token_count: int = 0,
) -> tuple[tuple[int, int, int], ...]:
    """Derive exact raw K/V anchor lanes from decoded succinct sections."""

    from verallm.proof_v3.rational_bundle_serving import (
        capture_kv_index_map_v3,
    )
    from verallm.proof_v3.succinct_attention_wire import (
        CaptureKvLayerSectionWireV3,
    )

    plans = tuple(plans)
    sections = tuple(sections)
    layers = tuple(int(plan.layer) for plan in plans)
    if (
        type(key_count) is not int
        or key_count < 1
        or type(cached_token_count) is not int
        or not 0 <= cached_token_count <= key_count
        or not layers
        or layers != tuple(sorted(set(layers)))
        or len(sections) != len(plans)
        or not all(
            isinstance(section, CaptureKvLayerSectionWireV3)
            for section in sections
        )
        or tuple(int(section.layer) for section in sections) != layers
    ):
        raise ProofV3Error(
            "attention anchor section plan is malformed"
        )
    sp = 1 << max(0, (key_count - 1).bit_length())
    keys: set[tuple[int, int, int]] = set()
    for plan, section in zip(plans, sections, strict=True):
        layer = int(plan.layer)
        try:
            commitment_index = int(commitment_indices_by_layer[layer])
            geometry = geometries_by_layer[layer]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProofV3Error(
                f"attention anchor metadata is missing layer {layer}"
            ) from exc
        if not isinstance(geometry, AttentionAnchorGeometryV3):
            raise ProofV3Error(
                f"attention anchor geometry is malformed at layer {layer}"
            )
        index_map = capture_kv_index_map_v3(
            heads=plan.heads,
            group=geometry.group,
            sp=sp,
            d=geometry.head_dim,
        )
        for tag, indices in zip(
            ("k", "v"),
            section.eq_indices,
            strict=True,
        ):
            for tile_leaf in indices:
                native_leaf = index_map(tile_leaf)
                kv_head, remainder = divmod(
                    int(native_leaf),
                    sp * geometry.head_dim,
                )
                position, _coordinate = divmod(
                    remainder,
                    geometry.head_dim,
                )
                if (
                    kv_head >= geometry.kv_heads
                    or position >= key_count
                    or position < cached_token_count
                ):
                    continue
                byte_start, byte_length = (
                    attention_anchor_head_byte_range_v3(
                        geometry=geometry,
                        tag=tag,
                        head=kv_head,
                    )
                )
                keys.update(
                    (
                        commitment_index,
                        position - cached_token_count,
                        lane,
                    )
                    for lane in required_execution_anchor_lanes_v3(
                        byte_start=byte_start,
                        byte_length=byte_length,
                    )
                )
    return tuple(sorted(keys))


def required_prefix_cache_attention_lane_keys_v3(
    *,
    bundle_wire: bytes,
    plans,
    geometries_by_layer: Mapping[int, "AttentionAnchorGeometryV3"],
    key_count: int,
    cached_token_count: int,
    block_token_count: int,
) -> tuple[tuple[int, str, int, int], ...]:
    """Map succinct K/V equality samples into logical cached-page lanes."""

    from verallm.proof_v3.succinct_attention_wire import (
        decode_anchor_capture_kv_bundle_wire_v3,
    )

    plans = tuple(plans)
    sections = decode_anchor_capture_kv_bundle_wire_v3(
        bundle_wire,
        expected_layers=tuple(int(plan.layer) for plan in plans),
    )
    return required_prefix_cache_attention_lane_keys_for_sections_v3(
        sections=sections,
        plans=plans,
        geometries_by_layer=geometries_by_layer,
        key_count=key_count,
        cached_token_count=cached_token_count,
        block_token_count=block_token_count,
    )


def required_prefix_cache_attention_lane_keys_for_sections_v3(
    *,
    sections,
    plans,
    geometries_by_layer: Mapping[int, "AttentionAnchorGeometryV3"],
    key_count: int,
    cached_token_count: int,
    block_token_count: int,
) -> tuple[tuple[int, str, int, int], ...]:
    """Map decoded attention equality samples to hierarchical page openings."""

    from verallm.proof_v3.rational_bundle_serving import (
        capture_kv_index_map_v3,
    )
    from verallm.proof_v3.succinct_attention_wire import (
        CaptureKvLayerSectionWireV3,
    )

    plans = tuple(plans)
    sections = tuple(sections)
    cached = int(cached_token_count)
    block_width = int(block_token_count)
    if (
        type(key_count) is not int
        or key_count < 1
        or type(cached_token_count) is not int
        or not 0 < cached <= key_count
        or type(block_token_count) is not int
        or block_width < 1
        or len(sections) != len(plans)
        or not all(
            isinstance(section, CaptureKvLayerSectionWireV3)
            for section in sections
        )
    ):
        raise ProofV3Error("prefix-cache attention lane plan is malformed")
    sp = 1 << max(0, (key_count - 1).bit_length())
    keys: set[tuple[int, str, int, int]] = set()
    for plan, section in zip(plans, sections, strict=True):
        layer = int(plan.layer)
        geometry = geometries_by_layer.get(layer)
        if (
            not isinstance(geometry, AttentionAnchorGeometryV3)
            or int(section.layer) != layer
        ):
            raise ProofV3Error(
                "prefix-cache attention geometry is incomplete"
            )
        index_map = capture_kv_index_map_v3(
            heads=plan.heads,
            group=geometry.group,
            sp=sp,
            d=geometry.head_dim,
        )
        for tag, indices in zip(("k", "v"), section.eq_indices, strict=True):
            stage_id = f"l{layer}.attention_{tag}_cache"
            lane_bytes = execution_anchor_lane_bytes_v3(stage_id)
            for tile_leaf in indices:
                native_leaf = index_map(tile_leaf)
                kv_head, remainder = divmod(
                    int(native_leaf), sp * geometry.head_dim
                )
                position, _coordinate = divmod(
                    remainder, geometry.head_dim
                )
                if (
                    kv_head >= geometry.kv_heads
                    or position >= cached
                ):
                    continue
                byte_start = kv_head * geometry.head_dim * 2
                keys.update(
                    (
                        position // block_width,
                        stage_id,
                        position % block_width,
                        lane,
                    )
                    for lane in required_execution_anchor_lanes_v3(
                        byte_start=byte_start,
                        byte_length=geometry.head_dim * 2,
                        lane_bytes=lane_bytes,
                    )
                )
    return tuple(sorted(keys))


def required_prefix_cache_projection_heads_for_sections_v3(
    *,
    sections,
    plans,
    geometries_by_layer: Mapping[int, "AttentionAnchorGeometryV3"],
    key_count: int,
    cached_token_count: int,
) -> tuple[tuple[int, int, int], ...]:
    """Return exact cached ``(layer, kv_head, position)`` equality heads.

    The succinct attention transcript samples individual K/V leaves.  A
    registered QKV projection cannot be joined to a post-RoPE cache value at
    one coordinate: K normalization and RoPE consume the complete logical
    head.  Expand each cached equality sample to its complete head while
    retaining the transcript-derived layer, head and position.
    """

    from verallm.proof_v3.rational_bundle_serving import (
        capture_kv_index_map_v3,
    )
    from verallm.proof_v3.succinct_attention_wire import (
        CaptureKvLayerSectionWireV3,
    )

    plans = tuple(plans)
    sections = tuple(sections)
    cached = int(cached_token_count)
    if (
        type(key_count) is not int
        or key_count < 1
        or type(cached_token_count) is not int
        or not 0 < cached <= key_count
        or len(sections) != len(plans)
        or not all(
            isinstance(section, CaptureKvLayerSectionWireV3)
            for section in sections
        )
    ):
        raise ProofV3Error("prefix-cache projection-head plan is malformed")
    sp = 1 << max(0, (key_count - 1).bit_length())
    result: set[tuple[int, int, int]] = set()
    for plan, section in zip(plans, sections, strict=True):
        layer = int(plan.layer)
        geometry = geometries_by_layer.get(layer)
        if (
            not isinstance(geometry, AttentionAnchorGeometryV3)
            or int(section.layer) != layer
        ):
            raise ProofV3Error(
                "prefix-cache projection-head geometry is incomplete"
            )
        index_map = capture_kv_index_map_v3(
            heads=plan.heads,
            group=geometry.group,
            sp=sp,
            d=geometry.head_dim,
        )
        for indices in section.eq_indices:
            for tile_leaf in indices:
                native_leaf = index_map(tile_leaf)
                kv_head, remainder = divmod(
                    int(native_leaf), sp * geometry.head_dim
                )
                position, _coordinate = divmod(
                    remainder, geometry.head_dim
                )
                if kv_head < geometry.kv_heads and position < cached:
                    result.add((layer, kv_head, position))
    return tuple(sorted(result))


def derive_prefix_cache_projection_heads_v3(
    *,
    plans,
    positions_by_layer,
    geometries_by_layer: Mapping[int, "AttentionAnchorGeometryV3"],
    cached_token_count: int,
) -> tuple[tuple[int, int, int], ...]:
    """Derive pre-replay complete KV heads from the nonce-bound plan."""

    plans = tuple(plans)
    try:
        positions = {
            int(layer): tuple(int(position) for position in rows)
            for layer, rows in tuple(positions_by_layer)
        }
    except (TypeError, ValueError) as exc:
        raise ProofV3Error(
            "prefix-cache projection-head positions are malformed"
        ) from exc
    if (
        type(cached_token_count) is not int
        or cached_token_count <= 0
        or set(positions) != {int(plan.layer) for plan in plans}
        or any(
            rows != tuple(sorted(set(rows))) or any(row < 0 for row in rows)
            for rows in positions.values()
        )
    ):
        raise ProofV3Error(
            "prefix-cache projection-head positions are malformed"
        )
    result = set()
    for plan in plans:
        layer = int(plan.layer)
        geometry = geometries_by_layer.get(layer)
        if not isinstance(geometry, AttentionAnchorGeometryV3):
            raise ProofV3Error(
                "prefix-cache projection-head geometry is incomplete"
            )
        kv_heads = tuple(
            sorted({int(head) // geometry.group for head in plan.heads})
        )
        if not kv_heads or kv_heads[-1] >= geometry.kv_heads:
            raise ProofV3Error(
                "prefix-cache projection-head selection is malformed"
            )
        result.update(
            (layer, kv_head, position)
            for kv_head in kv_heads
            for position in positions[layer]
            if position < cached_token_count
        )
    if not result:
        raise ProofV3Error("prefix-cache projection-head selection is empty")
    return tuple(sorted(result))


def decode_runtime_values_v3(raw: bytes, encoding_id: str):
    try:
        import numpy as np

        if encoding_id == "fp16.v1":
            values = np.frombuffer(raw, dtype="<f2").astype(np.float64)
        elif encoding_id == "bf16.v1":
            words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            values = (words << 16).view("<f4").astype(np.float64)
        else:
            raise ProofV3VerificationError(
                "attention anchor runtime encoding is not qualified"
            )
        if not bool(np.isfinite(values).all()):
            raise ProofV3VerificationError(
                "attention anchor contains a non-finite value"
            )
        return values
    except (ValueError, TypeError) as exc:
        raise ProofV3VerificationError(
            "attention anchor runtime values are malformed"
        ) from exc


@dataclass(frozen=True, slots=True)
class AttentionAnchorGeometryV3:
    query_heads: int
    kv_heads: int
    head_dim: int
    qkv_width: int
    q_block_width: int
    k_block_offset: int
    v_block_offset: int
    gated: bool

    @property
    def group(self) -> int:
        return self.query_heads // self.kv_heads


def attention_anchor_geometry_v3(
    *,
    qkv_width: int,
    o_input_width: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    semantics: AttentionRuntimeSemanticsV3,
) -> AttentionAnchorGeometryV3:
    if (
        not isinstance(semantics, AttentionRuntimeSemanticsV3)
        or query_heads <= 0
        or kv_heads <= 0
        or query_heads % kv_heads
        or head_dim <= 0
        or o_input_width != query_heads * head_dim
        or semantics.rotary_dimension > head_dim
    ):
        raise ProofV3VerificationError(
            "signed attention anchor geometry is inconsistent"
        )
    gated = semantics.qkv_layout_id == Q_GATE_INTERLEAVED_LAYOUT_V3
    q_block = query_heads * head_dim * (2 if gated else 1)
    expected = q_block + 2 * kv_heads * head_dim
    if qkv_width != expected:
        raise ProofV3VerificationError(
            "QKV execution-anchor width disagrees with signed semantics"
        )
    return AttentionAnchorGeometryV3(
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        qkv_width=qkv_width,
        q_block_width=q_block,
        k_block_offset=q_block,
        v_block_offset=q_block + kv_heads * head_dim,
        gated=gated,
    )


def _norm_weights(semantics, layer: int, which: str, head_dim: int):
    if (
        semantics.q_norm_id if which == "q" else semantics.k_norm_id
    ) == NO_QK_NORM_V3:
        return None
    binding = next(
        (
            item
            for item in semantics.norm_bindings
            if item.layer_index == int(layer)
        ),
        None,
    )
    if binding is None:
        raise ProofV3VerificationError(
            f"signed attention runtime artifact has no layer {layer} "
            f"{which}_norm weights"
        )
    raw = (
        binding.q_weight_bytes
        if which == "q"
        else binding.k_weight_bytes
    )
    values = decode_runtime_values_v3(raw, semantics.norm_encoding_id)
    if len(values) != head_dim:
        raise ProofV3VerificationError(
            f"signed attention {which}_norm width is inconsistent"
        )
    return values


def _normalize(values, *, norm_id: str, weights, epsilon: float):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if norm_id == NO_QK_NORM_V3:
        return values
    if norm_id != GEMMA_RMS_NORM_V3 or weights is None:
        raise ProofV3VerificationError(
            "signed attention normalization rule is unsupported"
        )
    return (
        values
        / math.sqrt(float(np.mean(values * values)) + float(epsilon))
        * (1.0 + weights)
    )


def _canonical_f32(value: float) -> float:
    """Round one scalar to IEEE-754 binary32 without a NumPy ufunc."""

    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


@lru_cache(maxsize=16_384)
def _canonical_rope_coefficients(
    rotary_dimension: int,
    rope_theta: float,
    position: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Canonical scalar-f32 RoPE coefficients.

    NumPy's vector ``power/sin/cos`` implementations can differ by one ULP
    across otherwise supported NumPy releases.  An exact PCS claim cannot
    tolerate that drift.  This routine specifies every binary32 boundary and
    uses scalar libm calls, then the caller applies the signed runtime
    FP16/BF16 boundary.
    """

    rot = int(rotary_dimension)
    if rot <= 0 or rot % 2 or int(position) < 0:
        raise ProofV3VerificationError(
            "attention RoPE coefficient geometry is malformed"
        )
    base = _canonical_f32(rope_theta)
    pos = _canonical_f32(position)
    cosines: list[float] = []
    sines: list[float] = []
    for index in range(0, rot, 2):
        exponent = _canonical_f32(index / rot)
        inverse = _canonical_f32(1.0 / math.pow(base, exponent))
        frequency = _canonical_f32(pos * inverse)
        cosines.append(_canonical_f32(math.cos(frequency)))
        sines.append(_canonical_f32(math.sin(frequency)))
    return tuple(cosines), tuple(sines)


def _runtime_rope_coefficients(semantics, position: int):
    """Return the signed runtime coefficient row for one position.

    Version-2 artifacts carry the exact FP16/BF16 table materialized by the
    qualified runtime. Version 1 remains readable for already signed legacy
    artifacts, but new qualification never rebuilds GPU coefficients through
    the verifier host's libm.
    """

    if semantics.version != ATTENTION_RUNTIME_SEMANTICS_VERSION_V3:
        return _canonical_rope_coefficients(
            semantics.rotary_dimension,
            float(semantics.rope_theta),
            int(position),
        )
    if not 0 <= int(position) < semantics.rope_coefficient_row_count:
        raise ProofV3VerificationError(
            "attention position exceeds the signed runtime RoPE table"
        )
    width = int(semantics.rotary_dimension)
    start = int(position) * width * 2
    raw = semantics.rope_coefficient_bytes[start:start + width * 2]
    values = decode_runtime_values_v3(
        raw, semantics.rope_coefficient_encoding_id
    )
    if len(values) != width:
        raise ProofV3VerificationError(
            "signed runtime RoPE coefficient row is truncated"
        )
    half = width // 2
    return tuple(values[:half]), tuple(values[half:])


def _rope(values, *, position: int, semantics, encoding_id: str):
    import numpy as np

    rot = semantics.rotary_dimension
    # vLLM constructs its RoPE cache in float32, casts that cache to the
    # query/key dtype, and writes the in-place result back at the same
    # precision.  Replaying an ideal float64 RoPE is measurably different at
    # long positions and does not authenticate the paged-KV values actually
    # consumed by the serving kernel.
    cos_f32, sin_f32 = _runtime_rope_coefficients(
        semantics, int(position)
    )
    cos = _round_runtime_precision(
        np.asarray(cos_f32, dtype=np.float32), encoding_id
    )
    sin = _round_runtime_precision(
        np.asarray(sin_f32, dtype=np.float32), encoding_id
    )
    first = values[: rot // 2]
    second = values[rot // 2:rot]
    rotated = np.concatenate(
        (first * cos - second * sin, second * cos + first * sin)
    )
    rotated = _round_runtime_precision(rotated, encoding_id)
    if rot == len(values):
        return rotated
    return np.concatenate(
        (rotated, _round_runtime_precision(values[rot:], encoding_id))
    )


def _round_clip(values, scale, low: int, high: int):
    import numpy as np

    return tuple(
        int(value)
        for value in np.clip(
            np.rint(values / scale), low, high
        ).astype(np.int64).tolist()
    )


def _round_runtime_precision(values, encoding_id: str):
    """Reproduce the runtime activation/cache precision boundary.

    vLLM writes post-RoPE K back to the FP16/BF16 paged cache before the
    attention kernel consumes it.  The weightless verifier reconstructs RoPE
    from the authenticated pre-RoPE QKV row, so it must apply that same cast
    before the signed fixed-point quantization.
    """

    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if encoding_id == "fp16.v1":
        return values.astype("<f2").astype(np.float64)
    if encoding_id == "bf16.v1":
        words = values.astype("<f4").view("<u4")
        # IEEE round-to-nearest-even from float32 to bfloat16.
        bias = np.uint32(0x7FFF) + ((words >> 16) & np.uint32(1))
        bf16 = ((words + bias) >> 16).astype("<u2")
        return (bf16.astype("<u4") << 16).view("<f4").astype(np.float64)
    raise ProofV3VerificationError(
        "attention runtime precision boundary is not qualified"
    )


def _params_for_kv(params_by_head, kv_head: int, geometry):
    slot = int(kv_head) * geometry.group
    try:
        params = params_by_head[slot]
    except IndexError as exc:
        raise ProofV3VerificationError(
            "signed attention calibration does not cover the KV head"
        ) from exc
    for other in params_by_head[
        slot:slot + geometry.group
    ]:
        if (
            other.k_num,
            other.k_e,
            other.v_num,
            other.v_e,
        ) != (
            params.k_num,
            params.k_e,
            params.v_num,
            params.v_e,
        ):
            raise ProofV3VerificationError(
                "signed attention GQA scales disagree within a KV group"
            )
    return params


@dataclass(frozen=True, slots=True)
class RuntimeAttentionQuantizedRowV3:
    q13_by_head: tuple[tuple[int, ...], ...]
    k13_by_head: tuple[tuple[int, ...], ...]
    v8_by_head: tuple[tuple[int, ...], ...]
    gate_by_head: tuple[tuple[float, ...], ...] | None


def runtime_attention_q13_pool_v3(
    *,
    rows_by_position: Mapping[int, bytes],
    candidate_positions,
    required_positions,
    layer: int,
    geometry: AttentionAnchorGeometryV3,
    semantics: AttentionRuntimeSemanticsV3,
    params_by_head,
    encoding_id: str,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Build the proof-side Q cube from the authenticated raw-QKV rows.

    Only nonce-selected ``required_positions`` are materialized; other
    candidate slots stay zero and are never consumed by the selected-row
    attention statement. Using the same row interpreter on prover and verifier
    prevents backend/Python floating-point replay drift from changing an exact
    PCS claim.
    """

    candidates = tuple(int(position) for position in candidate_positions)
    required = tuple(int(position) for position in required_positions)
    if (
        not candidates
        or len(candidates) != len(set(candidates))
        or not required
        or len(required) != len(set(required))
        or any(position not in candidates for position in required)
    ):
        raise ProofV3Error("attention Q pool coordinates are malformed")
    params = tuple(params_by_head)
    if len(params) != geometry.query_heads:
        raise ProofV3Error("attention Q pool has the wrong signed head count")
    slots = {position: index for index, position in enumerate(candidates)}
    result = [
        [[0] * geometry.head_dim for _position in candidates]
        for _head in range(geometry.query_heads)
    ]
    for position in required:
        try:
            row_bytes = rows_by_position[position]
        except KeyError as exc:
            raise ProofV3Error(
                "attention Q pool is missing a nonce-selected raw anchor row"
            ) from exc
        quantized = runtime_attention_quantized_row_v3(
            row_bytes=row_bytes,
            layer=int(layer),
            position=position,
            geometry=geometry,
            semantics=semantics,
            params_by_head=params,
            encoding_id=encoding_id,
        )
        slot = slots[position]
        for head, row in enumerate(quantized.q13_by_head):
            result[head][slot] = list(row)
    return tuple(
        tuple(tuple(int(value) for value in row) for row in head_rows)
        for head_rows in result
    )


def runtime_attention_quantized_row_v3(
    *,
    row_bytes: bytes,
    layer: int,
    position: int,
    geometry: AttentionAnchorGeometryV3,
    semantics: AttentionRuntimeSemanticsV3,
    params_by_head,
    encoding_id: str,
) -> RuntimeAttentionQuantizedRowV3:
    """Derive signed Q/K/V integers from one authenticated raw QKV row."""

    import numpy as np

    row = decode_runtime_values_v3(row_bytes, encoding_id)
    if len(row) != geometry.qkv_width:
        raise ProofV3VerificationError(
            "raw QKV anchor row has the wrong signed width"
        )
    params_by_head = tuple(params_by_head)
    if len(params_by_head) != geometry.query_heads:
        raise ProofV3VerificationError(
            "signed attention calibration has the wrong head count"
        )
    q_weights = _norm_weights(
        semantics, layer, "q", geometry.head_dim
    )
    k_weights = _norm_weights(
        semantics, layer, "k", geometry.head_dim
    )
    q_rows = []
    gates = []
    for head, params in enumerate(params_by_head):
        if geometry.gated:
            base = head * 2 * geometry.head_dim
            q_raw = row[base:base + geometry.head_dim]
            gate_raw = row[
                base + geometry.head_dim:base + 2 * geometry.head_dim
            ]
            gates.append(
                tuple(
                    float(value)
                    for value in (1.0 / (1.0 + np.exp(-gate_raw))).tolist()
                )
            )
        else:
            base = head * geometry.head_dim
            q_raw = row[base:base + geometry.head_dim]
        q_post = _rope(
            _normalize(
                q_raw,
                norm_id=semantics.q_norm_id,
                weights=q_weights,
                epsilon=semantics.q_norm_epsilon,
            ),
            position=position,
            semantics=semantics,
            encoding_id=encoding_id,
        )
        k_scale = np.asarray(
            [
                numerator / (1 << exponent)
                for numerator, exponent in zip(
                    params.k_num, params.k_e, strict=True
                )
            ],
            dtype=np.float64,
        )
        q_scale = params.q_num / (1 << params.q_e)
        q_rows.append(
            _round_clip(
                q_post * k_scale,
                q_scale,
                -params.qk_qmax,
                params.qk_qmax,
            )
        )
    k_rows, v_rows = [], []
    for kv_head in range(geometry.kv_heads):
        params = _params_for_kv(params_by_head, kv_head, geometry)
        start = geometry.k_block_offset + kv_head * geometry.head_dim
        k_raw = row[start:start + geometry.head_dim]
        k_post = _rope(
            _normalize(
                k_raw,
                norm_id=semantics.k_norm_id,
                weights=k_weights,
                epsilon=semantics.k_norm_epsilon,
            ),
            position=position,
            semantics=semantics,
            encoding_id=encoding_id,
        )
        k_scale = np.asarray(
            [
                numerator / (1 << exponent)
                for numerator, exponent in zip(
                    params.k_num, params.k_e, strict=True
                )
            ],
            dtype=np.float64,
        )
        k_rows.append(
            _round_clip(
                k_post,
                k_scale,
                -params.qk_qmax,
                params.qk_qmax,
            )
        )
        start = geometry.v_block_offset + kv_head * geometry.head_dim
        v_scale = params.v_num / (1 << params.v_e)
        v_rows.append(
            _round_clip(
                row[start:start + geometry.head_dim],
                v_scale,
                -127,
                127,
            )
        )
    return RuntimeAttentionQuantizedRowV3(
        q13_by_head=tuple(q_rows),
        k13_by_head=tuple(k_rows),
        v8_by_head=tuple(v_rows),
        gate_by_head=tuple(gates) if geometry.gated else None,
    )


def runtime_attention_q_head_quantized_v3(
    *,
    raw_head_bytes: bytes,
    layer: int,
    position: int,
    head: int,
    geometry: AttentionAnchorGeometryV3,
    semantics: AttentionRuntimeSemanticsV3,
    params_by_head,
    encoding_id: str,
) -> tuple[tuple[int, ...], tuple[float, ...] | None]:
    """Derive one selected Q head without transporting the full QKV row."""

    import numpy as np

    params_by_head = tuple(params_by_head)
    head = int(head)
    expected = geometry.head_dim * (2 if geometry.gated else 1)
    values = decode_runtime_values_v3(raw_head_bytes, encoding_id)
    if (
        len(values) != expected
        or len(params_by_head) != geometry.query_heads
        or not 0 <= head < geometry.query_heads
    ):
        raise ProofV3VerificationError(
            "attention anchor Q head geometry is inconsistent"
        )
    q_raw = values[: geometry.head_dim]
    gate = None
    if geometry.gated:
        gate_raw = values[geometry.head_dim :]
        gate = tuple(
            float(value)
            for value in (1.0 / (1.0 + np.exp(-gate_raw))).tolist()
        )
    q_post = _rope(
        _normalize(
            q_raw,
            norm_id=semantics.q_norm_id,
            weights=_norm_weights(
                semantics,
                int(layer),
                "q",
                geometry.head_dim,
            ),
            epsilon=semantics.q_norm_epsilon,
        ),
        position=int(position),
        semantics=semantics,
        encoding_id=encoding_id,
    )
    params = params_by_head[head]
    k_scale = np.asarray(
        [
            numerator / (1 << exponent)
            for numerator, exponent in zip(
                params.k_num,
                params.k_e,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    q_scale = params.q_num / (1 << params.q_e)
    q13 = _round_clip(
        q_post * k_scale,
        q_scale,
        -params.qk_qmax,
        params.qk_qmax,
    )
    return q13, gate


def attention_anchor_head_byte_range_v3(
    *,
    geometry: AttentionAnchorGeometryV3,
    tag: str,
    head: int,
) -> tuple[int, int]:
    if not 0 <= int(head) < geometry.kv_heads or tag not in {"k", "v"}:
        raise ProofV3Error("attention anchor KV coordinate is malformed")
    base = (
        geometry.k_block_offset if tag == "k"
        else geometry.v_block_offset
    )
    start = (base + int(head) * geometry.head_dim) * 2
    return start, geometry.head_dim * 2


def required_execution_anchor_lanes_v3(
    *, byte_start: int, byte_length: int, lane_bytes: int = 2048
) -> tuple[int, ...]:
    if (
        isinstance(byte_start, bool)
        or isinstance(byte_length, bool)
        or not isinstance(byte_start, int)
        or not isinstance(byte_length, int)
        or byte_start < 0
        or byte_length <= 0
        or type(lane_bytes) is not int
        or lane_bytes not in (256, 2048)
    ):
        raise ProofV3Error("execution anchor byte range is malformed")
    return tuple(
        range(
            byte_start // lane_bytes,
            (byte_start + byte_length - 1) // lane_bytes + 1,
        )
    )


def extract_execution_anchor_range_v3(
    *,
    commitment: ExecutionAnchorCommitmentV3,
    row_index: int,
    byte_start: int,
    byte_length: int,
    openings: Mapping[tuple[int, int], ExecutionAnchorLaneOpeningV3],
) -> bytes:
    if byte_start + byte_length > commitment.row_width:
        raise ProofV3VerificationError(
            "attention anchor byte range exceeds the signed row width"
        )
    lane_bytes = execution_anchor_lane_bytes_v3(commitment.stage_id)
    lanes = required_execution_anchor_lanes_v3(
        byte_start=byte_start,
        byte_length=byte_length,
        lane_bytes=lane_bytes,
    )
    verified = {}
    for lane in lanes:
        opening = openings.get((int(row_index), lane))
        if opening is None:
            raise ProofV3VerificationError(
                "attention anchor lane opening is missing"
            )
        verified[lane] = verify_execution_anchor_lane_v3(
            commitment=commitment,
            opening=opening,
        )
    first = lanes[0] * lane_bytes
    blob = b"".join(verified[lane] for lane in lanes)
    offset = byte_start - first
    result = blob[offset:offset + byte_length]
    if len(result) != byte_length:
        raise ProofV3VerificationError(
            "attention anchor lane reconstruction is truncated"
        )
    return result


def runtime_kv_head_quantized_v3(
    *,
    tag: str,
    raw_head_bytes: bytes,
    layer: int,
    position: int,
    kv_head: int,
    geometry: AttentionAnchorGeometryV3,
    semantics: AttentionRuntimeSemanticsV3,
    params_by_head,
    encoding_id: str,
) -> tuple[int, ...]:
    values = decode_runtime_values_v3(raw_head_bytes, encoding_id)
    if len(values) != geometry.head_dim:
        raise ProofV3VerificationError(
            "attention anchor KV head has the wrong width"
        )
    params = _params_for_kv(
        tuple(params_by_head), int(kv_head), geometry
    )
    if tag == "k":
        values = _rope(
            _normalize(
                values,
                norm_id=semantics.k_norm_id,
                weights=_norm_weights(
                    semantics, layer, "k", geometry.head_dim
                ),
                epsilon=semantics.k_norm_epsilon,
            ),
            position=position,
            semantics=semantics,
            encoding_id=encoding_id,
        )
        import numpy as np

        scale = np.asarray(
            [
                numerator / (1 << exponent)
                for numerator, exponent in zip(
                    params.k_num, params.k_e, strict=True
                )
            ],
            dtype=np.float64,
        )
        return _round_clip(
            values,
            scale,
            -params.qk_qmax,
            params.qk_qmax,
        )
    if tag == "v":
        return _round_clip(
            values,
            params.v_num / (1 << params.v_e),
            -127,
            127,
        )
    raise ProofV3VerificationError(
        "attention anchor KV tag is unsupported"
    )


def runtime_paged_cache_kv_head_quantized_v3(
    *,
    tag: str,
    raw_head_bytes: bytes,
    kv_head: int,
    geometry: AttentionAnchorGeometryV3,
    params_by_head,
    encoding_id: str,
) -> tuple[int, ...]:
    """Quantize vLLM paged-cache K/V without reapplying QK transforms.

    Unlike the raw QKV projection anchor, cache K is already normalized and
    RoPE-rotated. Applying those transforms again changes an honest cache row.
    """

    values = decode_runtime_values_v3(raw_head_bytes, encoding_id)
    if len(values) != geometry.head_dim:
        raise ProofV3VerificationError(
            "attention paged-cache KV head has the wrong width"
        )
    params = _params_for_kv(
        tuple(params_by_head), int(kv_head), geometry
    )
    if tag == "k":
        import numpy as np

        scale = np.asarray(
            [
                numerator / (1 << exponent)
                for numerator, exponent in zip(
                    params.k_num, params.k_e, strict=True
                )
            ],
            dtype=np.float64,
        )
        return _round_clip(
            values,
            scale,
            -params.qk_qmax,
            params.qk_qmax,
        )
    if tag == "v":
        return _round_clip(
            values,
            params.v_num / (1 << params.v_e),
            -127,
            127,
        )
    raise ProofV3VerificationError(
        "attention paged-cache KV tag is unsupported"
    )
