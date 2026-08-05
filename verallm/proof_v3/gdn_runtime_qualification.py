"""Build authenticated GDN runtime semantics from a qualified vLLM model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction

from verallm.proof_v2.transition import GDNTransitionParametersV2
from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.gdn_runtime_semantics import (
    GDN_RUNTIME_CHECKPOINT_SEMANTICS_VERSION_V3,
    GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3,
    GDN_RUNTIME_SEMANTICS_VERSION_V3,
    GdnLayerRuntimeSemanticsV3,
    GdnRuntimeSemanticsV3,
)

__all__ = [
    "GdnLayerToleranceMeasurementsV3",
    "GdnLayerToleranceReportV3",
    "GdnLayerStateToleranceReportV3",
    "freeze_gdn_checkpoint_state_tolerances_q24_v3",
    "freeze_gdn_output_tolerances_q24_v3",
    "attach_prefix_cache_gdn_qualification_v3",
    "maximum_gdn_zero_output_escape_probability_v3",
    "minimum_joint_gdn_zero_output_level_v3",
    "qualify_gdn_runtime_semantics_v3",
    "validate_gdn_runtime_semantics_v3",
]


def attach_prefix_cache_gdn_qualification_v3(
    *,
    semantics: GdnRuntimeSemanticsV3,
    output_tolerances_q24: Mapping[int, tuple[int, int]],
    checkpoint_state_tolerances_q24: Mapping[int, tuple[int, int]],
    page_token_count: int,
) -> GdnRuntimeSemanticsV3:
    """Attach independently qualified cache-page bounds to decode semantics.

    The existing short decode corridor remains unchanged.  Cache-page output
    and end-state bounds come from a separate full-page qualification run and
    are authenticated under the v5 artifact domain.
    """

    if (
        not isinstance(semantics, GdnRuntimeSemanticsV3)
        or semantics.version
        not in (
            GDN_RUNTIME_CHECKPOINT_SEMANTICS_VERSION_V3,
            GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3,
        )
        or not isinstance(output_tolerances_q24, Mapping)
        or not isinstance(checkpoint_state_tolerances_q24, Mapping)
        or isinstance(page_token_count, bool)
        or not isinstance(page_token_count, int)
        or not 1 <= page_token_count <= _MAX_DECODE_REPLAY_ROWS
    ):
        raise ProofV3Error("prefix-cache GDN qualification is malformed")
    layers = {item.layer_index for item in semantics.layers}
    if (
        set(output_tolerances_q24) != layers
        or set(checkpoint_state_tolerances_q24) != layers
    ):
        raise ProofV3Error(
            "prefix-cache GDN qualification does not cover the exact layer "
            "inventory"
        )

    qualified = []
    for item in semantics.layers:
        output = output_tolerances_q24[item.layer_index]
        state = checkpoint_state_tolerances_q24[item.layer_index]
        if (
            not isinstance(output, tuple)
            or len(output) != 2
            or not isinstance(state, tuple)
            or len(state) != 2
            or any(type(value) is not int for value in (*output, *state))
            or output[0] <= 0
            or output[1] < 0
            or state[0] <= 0
            or state[1] <= 0
        ):
            raise ProofV3Error(
                "prefix-cache GDN qualification tolerance is malformed"
            )
        qualified.append(replace(
            item,
            prefix_cache_output_atol_q24=output[0],
            prefix_cache_output_rtol_q24=output[1],
            prefix_cache_conv_state_atol_q24=state[0],
            prefix_cache_recurrent_state_atol_q24=state[1],
            max_prefix_cache_replay_rows=page_token_count,
        ))
    return replace(
        semantics,
        layers=tuple(qualified),
        version=GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3,
    )

_Q24 = 1 << 24
_MAX_DECODE_REPLAY_ROWS = 4096


def _elementary_symmetric_sum(values, degree: int):
    """Return the exact sum of products over all distinct subsets."""

    accumulators = [0] * (degree + 1)
    accumulators[0] = 1
    for value in values:
        for width in range(degree, 0, -1):
            accumulators[width] += (
                accumulators[width - 1] * value
            )
    return accumulators[degree]


def _finite_samples(values, name: str) -> tuple[float, ...]:
    try:
        samples = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ProofV3Error(f"GDN {name} are malformed") from exc
    if (
        not samples
        or any(not math.isfinite(value) or value < 0.0 for value in samples)
    ):
        raise ProofV3Error(f"GDN {name} are malformed")
    return samples


@dataclass(frozen=True, slots=True)
class GdnLayerToleranceMeasurementsV3:
    """One layer's graph-integrated calibration and held-out measurements."""

    calibration_absolute_errors: tuple[float, ...]
    heldout_absolute_errors: tuple[float, ...]
    zero_output_error_levels: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "calibration_absolute_errors",
            _finite_samples(
                self.calibration_absolute_errors,
                "calibration absolute errors",
            ),
        )
        object.__setattr__(
            self,
            "heldout_absolute_errors",
            _finite_samples(
                self.heldout_absolute_errors,
                "held-out absolute errors",
            ),
        )
        zero = _finite_samples(
            self.zero_output_error_levels,
            "zero-output error levels",
        )
        object.__setattr__(self, "zero_output_error_levels", zero)


@dataclass(frozen=True, slots=True)
class GdnLayerToleranceReportV3:
    layer_index: int
    calibration_maximum_absolute_error: float
    heldout_maximum_absolute_error: float
    signed_absolute_tolerance: float
    minimum_zero_output_headroom: float


@dataclass(frozen=True, slots=True)
class GdnLayerStateToleranceReportV3:
    layer_index: int
    maximum_conv_state_absolute_error: float
    maximum_recurrent_state_absolute_error: float
    signed_conv_state_absolute_tolerance: float
    signed_recurrent_state_absolute_tolerance: float


def freeze_gdn_checkpoint_state_tolerances_q24_v3(
    *,
    conv_state_absolute_errors: Mapping[int, tuple[float, ...]],
    recurrent_state_absolute_errors: Mapping[int, tuple[float, ...]],
    error_margin: float = 2.0,
) -> tuple[
    dict[int, tuple[int, int]],
    tuple[GdnLayerStateToleranceReportV3, ...],
]:
    """Freeze independent absolute corridors for authenticated GDN states.

    Output tensors, convolution cache state, and recurrent cache state are
    distinct numerical domains.  A signed output corridor must therefore
    never be reused as a checkpoint-state corridor.
    """

    if (
        not isinstance(conv_state_absolute_errors, Mapping)
        or not isinstance(recurrent_state_absolute_errors, Mapping)
        or not conv_state_absolute_errors
        or set(conv_state_absolute_errors)
        != set(recurrent_state_absolute_errors)
        or not math.isfinite(error_margin)
        or error_margin < 1.0
    ):
        raise ProofV3Error(
            "GDN checkpoint-state tolerance-freeze policy is malformed"
        )
    tolerances: dict[int, tuple[int, int]] = {}
    reports = []
    for raw_layer in sorted(conv_state_absolute_errors):
        if (
            isinstance(raw_layer, bool)
            or not isinstance(raw_layer, int)
            or raw_layer < 0
        ):
            raise ProofV3Error(
                "GDN checkpoint-state tolerance layer is malformed"
            )
        conv_max = max(
            _finite_samples(
                conv_state_absolute_errors[raw_layer],
                "convolution-state absolute errors",
            )
        )
        recurrent_max = max(
            _finite_samples(
                recurrent_state_absolute_errors[raw_layer],
                "recurrent-state absolute errors",
            )
        )
        conv_q24 = max(
            1,
            int(math.ceil(conv_max * error_margin * _Q24)),
        )
        recurrent_q24 = max(
            1,
            int(math.ceil(recurrent_max * error_margin * _Q24)),
        )
        if conv_q24 > 16 * _Q24 or recurrent_q24 > 16 * _Q24:
            raise ProofV3Error(
                f"GDN layer {raw_layer} checkpoint-state tolerance "
                "exceeds the protocol range"
            )
        tolerances[raw_layer] = (conv_q24, recurrent_q24)
        reports.append(
            GdnLayerStateToleranceReportV3(
                layer_index=raw_layer,
                maximum_conv_state_absolute_error=conv_max,
                maximum_recurrent_state_absolute_error=recurrent_max,
                signed_conv_state_absolute_tolerance=(
                    conv_q24 / float(_Q24)
                ),
                signed_recurrent_state_absolute_tolerance=(
                    recurrent_q24 / float(_Q24)
                ),
            )
        )
    return tolerances, tuple(reports)


def minimum_joint_gdn_zero_output_level_v3(
    per_value_head_levels,
    *,
    num_key_heads: int,
    num_value_heads: int,
    selected_key_head_groups: int,
) -> float:
    """Return the weakest all-zero signal for the signed GDN head sampler.

    The hard challenge first selects distinct key-head groups and then one
    value head from each selected group.  Verification compares every selected
    head, so an all-zero substituted output is rejected when at least one head
    in every feasible selection exceeds the signed tolerance.  The weakest
    feasible selection is therefore the ``k``-th smallest group minimum, where
    ``k`` is the number of distinct selected groups.
    """

    try:
        levels = tuple(float(value) for value in per_value_head_levels)
    except (TypeError, ValueError) as exc:
        raise ProofV3Error("GDN per-head zero-output levels are malformed") from exc
    if (
        isinstance(num_key_heads, bool)
        or not isinstance(num_key_heads, int)
        or num_key_heads <= 0
        or isinstance(num_value_heads, bool)
        or not isinstance(num_value_heads, int)
        or num_value_heads <= 0
        or num_value_heads % num_key_heads
        or len(levels) != num_value_heads
        or any(not math.isfinite(value) or value < 0.0 for value in levels)
        or isinstance(selected_key_head_groups, bool)
        or not isinstance(selected_key_head_groups, int)
        or not 1 <= selected_key_head_groups <= num_key_heads
    ):
        raise ProofV3Error("GDN per-head zero-output geometry is malformed")
    group_size = num_value_heads // num_key_heads
    group_minima = sorted(
        min(levels[start : start + group_size])
        for start in range(0, num_value_heads, group_size)
    )
    return group_minima[selected_key_head_groups - 1]


def freeze_gdn_output_tolerances_q24_v3(
    measurements: Mapping[int, GdnLayerToleranceMeasurementsV3],
    *,
    error_margin: float = 2.0,
    minimum_zero_output_headroom: float | None = 4.0,
    envelope_scope: str = "calibration",
) -> tuple[
    dict[int, tuple[int, int]],
    tuple[GdnLayerToleranceReportV3, ...],
]:
    """Freeze non-vacuous absolute corridors from qualified runtime traces.

    The runtime/verifier relation is deterministic for a fixed qualified
    backend. We therefore use a Q24 absolute tolerance and zero relative
    tolerance. ``calibration`` preserves the split-derived envelope and
    requires every held-out trace to fit it. ``representative_corpus`` freezes
    the same fixed margin over the maximum of both signed corpus splits; this
    is useful when the complete corpus is the qualification population and a
    separate post-freeze runtime battery supplies the independent smoke.

    A corridor close enough to admit a zero runtime output fails closed when
    ``minimum_zero_output_headroom`` is set. A sampler-aware qualification can
    defer that decision by passing ``None`` and applying the exact
    post-commitment escape bound separately.
    """

    if (
        not isinstance(measurements, Mapping)
        or not measurements
        or not math.isfinite(error_margin)
        or error_margin < 1.0
        or envelope_scope not in {
            "calibration",
            "representative_corpus",
        }
        or (
            minimum_zero_output_headroom is not None
            and (
                not math.isfinite(minimum_zero_output_headroom)
                or minimum_zero_output_headroom <= 1.0
            )
        )
    ):
        raise ProofV3Error("GDN tolerance-freeze policy is malformed")
    tolerances: dict[int, tuple[int, int]] = {}
    reports = []
    for raw_layer in sorted(measurements):
        if (
            isinstance(raw_layer, bool)
            or not isinstance(raw_layer, int)
            or raw_layer < 0
        ):
            raise ProofV3Error("GDN tolerance layer index is malformed")
        layer = raw_layer
        item = measurements[layer]
        if not isinstance(item, GdnLayerToleranceMeasurementsV3):
            raise ProofV3Error(
                "GDN tolerance measurement has an unexpected type"
        )
        calibration_max = max(item.calibration_absolute_errors)
        heldout_max = max(item.heldout_absolute_errors)
        envelope_max = (
            calibration_max
            if envelope_scope == "calibration"
            else max(calibration_max, heldout_max)
        )
        tolerance_q24 = max(
            1,
            int(math.ceil(envelope_max * error_margin * _Q24)),
        )
        if tolerance_q24 > 16 * _Q24:
            raise ProofV3Error(
                f"GDN layer {layer} tolerance exceeds the protocol range"
            )
        tolerance = tolerance_q24 / float(_Q24)
        if envelope_scope == "calibration" and heldout_max > tolerance:
            raise ProofV3Error(
                f"GDN layer {layer} held-out replay exceeds the frozen "
                "tolerance"
            )
        zero_headroom = min(item.zero_output_error_levels) / tolerance
        if (
            minimum_zero_output_headroom is not None
            and zero_headroom < minimum_zero_output_headroom
        ):
            raise ProofV3Error(
                f"GDN layer {layer} tolerance is vacuous against a zero "
                "runtime output: "
                f"calibration_max={calibration_max:.9g}, "
                f"heldout_max={heldout_max:.9g}, "
                f"signed_tolerance={tolerance:.9g}, "
                f"minimum_zero_level="
                f"{min(item.zero_output_error_levels):.9g}, "
                f"headroom={zero_headroom:.9g}, "
                f"required_headroom={minimum_zero_output_headroom:.9g}"
            )
        tolerances[layer] = (tolerance_q24, 0)
        reports.append(
            GdnLayerToleranceReportV3(
                layer_index=layer,
                calibration_maximum_absolute_error=calibration_max,
                heldout_maximum_absolute_error=heldout_max,
                signed_absolute_tolerance=tolerance,
                minimum_zero_output_headroom=zero_headroom,
            )
        )
    return tolerances, tuple(reports)


def maximum_gdn_zero_output_escape_probability_v3(
    *,
    zero_output_levels_by_layer: Mapping[
        int,
        tuple[tuple[float, ...], ...],
    ],
    output_tolerances_q24: Mapping[int, tuple[int, int]],
    head_geometry_by_layer: Mapping[int, tuple[int, int]],
    selected_layer_count: int,
    selected_key_head_groups: int,
    safety_headroom: float,
) -> tuple[Fraction, int]:
    """Bound robust all-zero escape over the exact signed GDN sampler.

    Qualification takes the worst prompt/window sample. Within that sample,
    the post-commitment challenge selects distinct GDN layers uniformly,
    distinct key-head groups uniformly in each layer, and one value head
    uniformly in each selected group. A selection counts as a robust catch
    only when at least one selected value head meets ``safety_headroom``
    times the signed honest tolerance.
    """

    if (
        not isinstance(zero_output_levels_by_layer, Mapping)
        or not isinstance(output_tolerances_q24, Mapping)
        or not isinstance(head_geometry_by_layer, Mapping)
        or not zero_output_levels_by_layer
    ):
        raise ProofV3Error("GDN zero-output escape policy is malformed")
    raw_layers = tuple(zero_output_levels_by_layer)
    if any(
        isinstance(layer, bool)
        or not isinstance(layer, int)
        or layer < 0
        for layer in raw_layers
    ):
        raise ProofV3Error("GDN zero-output escape policy is malformed")
    layers = tuple(sorted(raw_layers))
    if (
        set(layers) != set(output_tolerances_q24)
        or set(layers) != set(head_geometry_by_layer)
        or isinstance(selected_layer_count, bool)
        or not isinstance(selected_layer_count, int)
        or not 1 <= selected_layer_count <= len(layers)
        or isinstance(selected_key_head_groups, bool)
        or not isinstance(selected_key_head_groups, int)
        or selected_key_head_groups <= 0
        or not math.isfinite(safety_headroom)
        or safety_headroom <= 1.0
    ):
        raise ProofV3Error("GDN zero-output escape policy is malformed")
    sample_counts = {
        len(zero_output_levels_by_layer[layer]) for layer in layers
    }
    if len(sample_counts) != 1 or not next(iter(sample_counts)):
        raise ProofV3Error(
            "GDN zero-output escape samples are not aligned"
        )
    sample_count = next(iter(sample_counts))
    layer_escape_by_sample: dict[int, tuple[Fraction, ...]] = {}
    for layer in layers:
        geometry = head_geometry_by_layer[layer]
        tolerance = output_tolerances_q24[layer]
        if (
            not isinstance(geometry, tuple)
            or len(geometry) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in geometry
            )
            or geometry[0] <= 0
            or geometry[1] <= 0
            or geometry[1] % geometry[0]
            or selected_key_head_groups > geometry[0]
            or not isinstance(tolerance, tuple)
            or len(tolerance) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in tolerance
            )
            or tolerance[0] <= 0
            or tolerance[1] < 0
        ):
            raise ProofV3Error(
                "GDN zero-output escape geometry is malformed"
            )
        num_key_heads, num_value_heads = geometry
        group_size = num_value_heads // num_key_heads
        denominator = (
            math.comb(num_key_heads, selected_key_head_groups)
            * group_size**selected_key_head_groups
        )
        signed_atol = tolerance[0] / float(_Q24)
        signed_rtol = tolerance[1] / float(_Q24)
        robust_denominator = 1.0 - safety_headroom * signed_rtol
        robust_threshold = (
            safety_headroom * signed_atol / robust_denominator
            if robust_denominator > 0.0
            else math.inf
        )
        probabilities = []
        for raw_levels in zero_output_levels_by_layer[layer]:
            levels = _finite_samples(
                raw_levels,
                "per-head zero-output levels",
            )
            if len(levels) != num_value_heads:
                raise ProofV3Error(
                    "GDN zero-output escape head inventory changed"
                )
            weak_counts = tuple(
                sum(
                    levels[index] < robust_threshold
                    for index in range(start, start + group_size)
                )
                for start in range(0, num_value_heads, group_size)
            )
            numerator = _elementary_symmetric_sum(
                weak_counts,
                selected_key_head_groups,
            )
            probabilities.append(Fraction(numerator, denominator))
        layer_escape_by_sample[layer] = tuple(probabilities)

    worst_probability = Fraction(0, 1)
    worst_sample = 0
    layer_denominator = math.comb(len(layers), selected_layer_count)
    for sample_index in range(sample_count):
        probability = _elementary_symmetric_sum(
            (
                layer_escape_by_sample[layer][sample_index]
                for layer in layers
            ),
            selected_layer_count,
        ) / layer_denominator
        if probability > worst_probability:
            worst_probability = probability
            worst_sample = sample_index
    return worst_probability, worst_sample


def _unwrap_gdn(layer):
    import torch.nn as nn

    owner = layer
    while isinstance(getattr(owner, "original", None), nn.Module):
        owner = owner.original
    module = getattr(owner, "linear_attn", None)
    while (
        isinstance(getattr(module, "original", None), nn.Module)
    ):
        module = module.original
    if module is None:
        raise ProofV3Error("qualified GDN layer has no linear-attention module")
    return module


def _encoding(dtype, *, runtime: bool) -> str:
    import torch

    choices = {
        torch.float16: "fp16.v1",
        torch.bfloat16: "bf16.v1",
    }
    if not runtime:
        choices[torch.float32] = "fp32.v1"
    try:
        return choices[dtype]
    except KeyError as exc:
        raise ProofV3Error("qualified GDN runtime dtype is unsupported") from exc


def _transition_dtype(dtype) -> str:
    import torch

    choices = {
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.float32: "f32",
    }
    try:
        return choices[dtype]
    except KeyError as exc:
        raise ProofV3Error(
            "qualified GDN transition dtype is unsupported"
        ) from exc


def _exact_runtime_16bit_bytes(
    tensor,
    *,
    shape,
    name: str,
    runtime_dtype: str,
) -> bytes:
    import torch

    value = tensor.detach().cpu().contiguous()
    if tuple(value.shape) != tuple(shape) or not bool(torch.isfinite(value).all()):
        raise ProofV3Error(f"qualified GDN {name} tensor is malformed")
    torch_dtype = {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(runtime_dtype)
    if torch_dtype is None:
        raise ProofV3Error(
            "qualified GDN 16-bit parameter encoding is unsupported"
        )
    rounded = value.to(torch_dtype)
    if not bool(torch.equal(rounded.float(), value.float())):
        label = "FP16" if runtime_dtype == "f16" else "BF16"
        raise ProofV3Error(
            f"qualified GDN {name} is not exactly representable in "
            f"{label}"
        )
    if runtime_dtype == "f16":
        return rounded.numpy().astype("<f2", copy=False).tobytes()
    return (
        rounded.view(torch.int16)
        .numpy()
        .astype("<u2", copy=False)
        .tobytes()
    )


def _f32_bytes(tensor, *, shape, name: str) -> bytes:
    import torch

    value = tensor.detach().to(torch.float32).cpu().contiguous()
    if tuple(value.shape) != tuple(shape) or not bool(torch.isfinite(value).all()):
        raise ProofV3Error(f"qualified GDN {name} tensor is malformed")
    return value.numpy().astype("<f4", copy=False).tobytes()


def qualify_gdn_runtime_semantics_v3(
    *,
    layers,
    layer_kinds,
    output_tolerances_q24: Mapping[int, tuple[int, int]],
    checkpoint_state_tolerances_q24: (
        Mapping[int, tuple[int, int]] | None
    ) = None,
    max_decode_replay_rows: int,
    decode_checkpoint_stride: int = 0,
    max_hard_audit_decode_tokens: int = 0,
    adapter_id: str = "qwen.gdn.v1",
) -> GdnRuntimeSemanticsV3:
    """Freeze exact per-layer GDN parameters and qualified error corridors.

    ``output_tolerances_q24`` is produced by an offline qualification corpus;
    it is deliberately mandatory rather than guessed from the model family.
    """

    layers = tuple(layers)
    kinds = tuple(str(kind) for kind in layer_kinds)
    if (
        not layers
        or len(layers) != len(kinds)
        or any(kind not in {"full_attention", "gdn"} for kind in kinds)
        or not isinstance(output_tolerances_q24, Mapping)
        or (
            checkpoint_state_tolerances_q24 is not None
            and not isinstance(checkpoint_state_tolerances_q24, Mapping)
        )
        or isinstance(max_decode_replay_rows, bool)
        or not isinstance(max_decode_replay_rows, int)
        or not 1 <= max_decode_replay_rows <= _MAX_DECODE_REPLAY_ROWS
        or isinstance(decode_checkpoint_stride, bool)
        or not isinstance(decode_checkpoint_stride, int)
        or not 0 <= decode_checkpoint_stride <= _MAX_DECODE_REPLAY_ROWS
        or (
            decode_checkpoint_stride
            and decode_checkpoint_stride != max_decode_replay_rows
        )
        or (
            decode_checkpoint_stride
            and (
                isinstance(max_hard_audit_decode_tokens, bool)
                or not isinstance(max_hard_audit_decode_tokens, int)
                or not 2 <= max_hard_audit_decode_tokens < 1 << 32
            )
        )
        or (
            not decode_checkpoint_stride
            and max_hard_audit_decode_tokens != 0
        )
    ):
        raise ProofV3Error("GDN runtime qualification inputs are malformed")
    gdn_layers = tuple(
        index for index, kind in enumerate(kinds) if kind == "gdn"
    )
    if not gdn_layers or set(output_tolerances_q24) != set(gdn_layers):
        raise ProofV3Error(
            "GDN output tolerances do not cover the exact GDN layer inventory"
        )
    if decode_checkpoint_stride:
        if (
            checkpoint_state_tolerances_q24 is None
            or set(checkpoint_state_tolerances_q24) != set(gdn_layers)
        ):
            raise ProofV3Error(
                "GDN checkpoint-state tolerances do not cover the exact "
                "GDN layer inventory"
            )
    elif checkpoint_state_tolerances_q24 is not None:
        raise ProofV3Error(
            "legacy GDN qualification cannot declare checkpoint-state "
            "tolerances"
        )

    qualified = []
    for layer_index in gdn_layers:
        module = _unwrap_gdn(layers[layer_index])
        try:
            tp_size = int(module.tp_size)
            num_key_heads = int(module.num_k_heads)
            num_value_heads = int(module.num_v_heads)
            key_head_dim = int(module.head_k_dim)
            value_head_dim = int(module.head_v_dim)
            kernel = int(module.conv_kernel_size)
            epsilon = float(module.layer_norm_epsilon)
            conv_weight = module.conv1d.weight
            a_log = module.A_log
            dt_bias = module.dt_bias
            norm_weight = module.norm.weight
            state_dtypes = tuple(module.get_state_dtype())
        except (AttributeError, TypeError, ValueError) as exc:
            raise ProofV3Error(
                f"qualified GDN layer {layer_index} has unsupported metadata"
            ) from exc
        if (
            tp_size != 1
            or min(
                num_key_heads,
                num_value_heads,
                key_head_dim,
                value_head_dim,
                kernel,
            ) <= 0
            or len(state_dtypes) != 2
            or not epsilon > 0
        ):
            raise ProofV3Error(
                f"qualified GDN layer {layer_index} geometry is unsupported"
            )
        runtime_dtype = getattr(
            getattr(module, "in_proj_qkvz", None),
            "params_dtype",
            conv_weight.dtype,
        )
        transition_runtime_dtype = _transition_dtype(runtime_dtype)
        if transition_runtime_dtype not in {"f16", "bf16"}:
            raise ProofV3Error(
                f"qualified GDN layer {layer_index} runtime dtype is "
                "unsupported"
            )
        conv_dim = (
            2 * num_key_heads * key_head_dim
            + num_value_heads * value_head_dim
        )
        if tuple(conv_weight.shape) == (conv_dim, 1, kernel):
            conv_weight = conv_weight[:, 0, :]
        transition = GDNTransitionParametersV2(
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            conv_kernel_size=kernel,
            rms_epsilon_q32=max(
                1, int(round(epsilon * float(1 << 32)))
            ),
            conv_weight_f16=_exact_runtime_16bit_bytes(
                conv_weight,
                shape=(conv_dim, kernel),
                name="convolution weight",
                runtime_dtype=transition_runtime_dtype,
            ),
            a_log_f32=_f32_bytes(
                a_log,
                shape=(num_value_heads,),
                name="A_log",
            ),
            dt_bias_f16=_exact_runtime_16bit_bytes(
                dt_bias,
                shape=(num_value_heads,),
                name="dt_bias",
                runtime_dtype=transition_runtime_dtype,
            ),
            norm_weight_f16=_exact_runtime_16bit_bytes(
                norm_weight,
                shape=(value_head_dim,),
                name="normalization weight",
                runtime_dtype=transition_runtime_dtype,
            ),
            runtime_dtype=transition_runtime_dtype,
            recurrent_state_dtype=_transition_dtype(state_dtypes[1]),
        )
        tolerance = output_tolerances_q24[layer_index]
        state_tolerances = (
            checkpoint_state_tolerances_q24[layer_index]
            if checkpoint_state_tolerances_q24 is not None
            else (tolerance[0], tolerance[0])
        )
        if (
            not isinstance(tolerance, tuple)
            or len(tolerance) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in tolerance
            )
            or not isinstance(state_tolerances, tuple)
            or len(state_tolerances) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in state_tolerances
            )
        ):
            raise ProofV3Error(
                f"qualified GDN layer {layer_index} tolerance is malformed"
            )
        qualified.append(
            GdnLayerRuntimeSemanticsV3(
                layer_index=layer_index,
                transition_parameters=transition.canonical_bytes(),
                runtime_encoding_id=_encoding(runtime_dtype, runtime=True),
                conv_state_encoding_id=_encoding(
                    state_dtypes[0], runtime=False
                ),
                recurrent_state_encoding_id=_encoding(
                    state_dtypes[1], runtime=False
                ),
                output_atol_q24=tolerance[0],
                output_rtol_q24=tolerance[1],
                conv_state_atol_q24=state_tolerances[0],
                recurrent_state_atol_q24=state_tolerances[1],
                max_decode_replay_rows=max_decode_replay_rows,
                decode_checkpoint_stride=decode_checkpoint_stride,
            )
        )
    return GdnRuntimeSemanticsV3(
        adapter_id=adapter_id,
        layers=tuple(qualified),
        version=(
            GDN_RUNTIME_CHECKPOINT_SEMANTICS_VERSION_V3
            if decode_checkpoint_stride
            else GDN_RUNTIME_SEMANTICS_VERSION_V3
        ),
        max_hard_audit_decode_tokens=max_hard_audit_decode_tokens,
    )


def validate_gdn_runtime_semantics_v3(
    *,
    layers,
    layer_kinds,
    semantics: GdnRuntimeSemanticsV3,
) -> None:
    """Require a released GDN artifact to describe this exact runtime.

    Qualification-derived tolerances remain unchanged, while all static
    recurrence parameters, encodings, layer indices, adapter identity, and
    decode-row bounds are re-derived from the loaded model.
    """

    if not isinstance(semantics, GdnRuntimeSemanticsV3):
        raise ProofV3Error(
            "released GDN runtime semantics have an unexpected type"
        )
    row_bounds = {
        item.max_decode_replay_rows for item in semantics.layers
    }
    if len(row_bounds) != 1:
        raise ProofV3Error(
            "released GDN runtime semantics vary the decode-row bound"
        )
    rebuilt = qualify_gdn_runtime_semantics_v3(
        layers=layers,
        layer_kinds=layer_kinds,
        output_tolerances_q24={
            item.layer_index: (
                item.output_atol_q24,
                item.output_rtol_q24,
            )
            for item in semantics.layers
        },
        checkpoint_state_tolerances_q24=(
            {
                item.layer_index: (
                    item.conv_state_atol_q24,
                    item.recurrent_state_atol_q24,
                )
                for item in semantics.layers
            }
            if semantics.decode_checkpoint_stride
            else None
        ),
        max_decode_replay_rows=next(iter(row_bounds)),
        decode_checkpoint_stride=semantics.decode_checkpoint_stride,
        max_hard_audit_decode_tokens=(
            semantics.max_hard_audit_decode_tokens
        ),
        adapter_id=semantics.adapter_id,
    )
    if semantics.version == GDN_RUNTIME_PREFIX_CACHE_SEMANTICS_VERSION_V3:
        prefix_bounds = {
            item.max_prefix_cache_replay_rows for item in semantics.layers
        }
        if len(prefix_bounds) != 1:
            raise ProofV3Error(
                "released GDN runtime semantics vary the prefix-cache row "
                "bound"
            )
        rebuilt = attach_prefix_cache_gdn_qualification_v3(
            semantics=rebuilt,
            output_tolerances_q24={
                item.layer_index: (
                    item.prefix_cache_output_atol_q24,
                    item.prefix_cache_output_rtol_q24,
                )
                for item in semantics.layers
            },
            checkpoint_state_tolerances_q24={
                item.layer_index: (
                    item.prefix_cache_conv_state_atol_q24,
                    item.prefix_cache_recurrent_state_atol_q24,
                )
                for item in semantics.layers
            },
            page_token_count=next(iter(prefix_bounds)),
        )
    if rebuilt != semantics:
        raise ProofV3Error(
            "released GDN runtime semantics do not match the loaded model"
        )
