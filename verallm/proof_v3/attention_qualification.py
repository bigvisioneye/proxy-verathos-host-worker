"""Fail-closed evidence for scored-attention runtime qualification.

The runtime calibration artifact contains the signed arithmetic parameters and
bounds.  This companion report records that those bounds were derived from a
calibration split and then admitted an independent held-out split over every
registered full-attention layer/head and each qualified context band.  It is a
release-builder gate; miners and validators continue to consume the unchanged
runtime calibration ABI.
"""

from __future__ import annotations

import hashlib
import json
import math

from verallm.proof_v3.errors import ProofV3Error


ATTENTION_RUNTIME_QUALIFICATION_ABI_V3 = (
    "verathos.proof_v3.attention_runtime_qualification.v2"
)
ATTENTION_QUALIFICATION_PROMPT_STYLE_COUNT_V3 = 8
MINIMUM_ATTENTION_CALIBRATION_PLANS_V3 = 8
MINIMUM_ATTENTION_HELDOUT_PLANS_V3 = 8
MINIMUM_ATTENTION_CONTEXT_SAMPLES_PER_BAND_V3 = 2
MINIMUM_ATTENTION_POLICY_NONCE_SAMPLES_V3 = 4096
MAXIMUM_ZEROED_OUTPUT_ESCAPE_UPPER_BOUND_V3 = 0.02


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ProofV3Error(f"attention qualification {name} is malformed")
    return value


def _positive_finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ProofV3Error(f"attention qualification {name} is malformed")
    return float(value)


def _nonnegative_finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ProofV3Error(f"attention qualification {name} is malformed")
    return float(value)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProofV3Error(f"attention qualification {name} is malformed")
    return value


def _wilson_upper(successes: int, trials: int, z: float = 1.645) -> float:
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    centre = rate + z * z / (2.0 * trials)
    radius = z * (
        rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)
    ) ** 0.5
    return min(1.0, (centre + radius) / denominator)


def validate_attention_runtime_qualification_v3(
    value: object,
    *,
    model_id: str,
    calibration_set,
    runtime_identity: object | None = None,
    minimum_attention_subaudits: int | None = None,
) -> None:
    """Validate one complete calibration/held-out qualification report."""

    required = {
        "abi",
        "model_id",
        "calibration_set_digest",
        "runtime_fingerprint_sha256",
        "backend_id",
        "minimum_calibration_plans",
        "minimum_heldout_plans",
        "minimum_context_samples_per_band",
        "minimum_attention_subaudits",
        "policy_nonce_samples",
        "maximum_zeroed_output_escape_upper_bound",
        "inventory",
        "bands",
        "calibration_case_count",
        "heldout_case_count",
        "heldout_gate_passed",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ProofV3Error("attention qualification report is malformed")
    if (
        value["abi"] != ATTENTION_RUNTIME_QUALIFICATION_ABI_V3
        or value["model_id"] != model_id
        or value["calibration_set_digest"] != calibration_set.digest.hex()
        or value["heldout_gate_passed"] is not True
    ):
        raise ProofV3Error(
            "attention qualification does not match the runtime calibration"
        )
    runtime_fingerprint = _sha256(
        value["runtime_fingerprint_sha256"], "runtime fingerprint"
    )
    if not isinstance(value["backend_id"], str) or not value["backend_id"]:
        raise ProofV3Error("attention qualification backend is malformed")
    if runtime_identity is not None:
        required_identity = {
            "model_id",
            "model_path_identity",
            "model_config_class",
            "torch_version",
            "torch_cuda_version",
            "vllm_version",
            "gpu_name",
            "gpu_capability",
            "capture_adapter",
        }
        if (
            not isinstance(runtime_identity, dict)
            or set(runtime_identity) != required_identity
            or runtime_identity.get("model_id") != model_id
            or runtime_identity.get("capture_adapter")
            != "request_activation_tracker.v1"
        ):
            raise ProofV3Error(
                "attention qualification runtime identity is malformed"
            )
        for name in (
            "model_path_identity",
            "model_config_class",
            "torch_version",
            "torch_cuda_version",
            "vllm_version",
            "gpu_name",
        ):
            if not isinstance(runtime_identity.get(name), str) or not (
                runtime_identity[name]
            ):
                raise ProofV3Error(
                    "attention qualification runtime identity is malformed"
                )
        capability = runtime_identity.get("gpu_capability")
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 0
                for item in capability
            )
        ):
            raise ProofV3Error(
                "attention qualification runtime identity is malformed"
            )
        expected_fingerprint = hashlib.sha256(
            json.dumps(
                runtime_identity,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()
        expected_backend = (
            f"vllm-{runtime_identity['vllm_version']}/"
            f"cuda-{runtime_identity['torch_cuda_version']}/"
            f"sm{capability[0]}{capability[1]}/"
            "request-activation-tracker"
        )
        if (
            runtime_fingerprint != expected_fingerprint
            or value["backend_id"] != expected_backend
        ):
            raise ProofV3Error(
                "attention qualification runtime identity is unauthenticated"
            )

    minimum_calibration = _positive_int(
        value["minimum_calibration_plans"], "calibration-plan minimum"
    )
    minimum_heldout = _positive_int(
        value["minimum_heldout_plans"], "held-out-plan minimum"
    )
    minimum_contexts = _positive_int(
        value["minimum_context_samples_per_band"],
        "context-sample minimum",
    )
    minimum_subaudits = _positive_int(
        value["minimum_attention_subaudits"],
        "attention-subaudit minimum",
    )
    policy_nonce_samples = _positive_int(
        value["policy_nonce_samples"], "policy nonce-sample count"
    )
    maximum_escape = _positive_finite(
        value["maximum_zeroed_output_escape_upper_bound"],
        "zeroed-output escape bound",
    )
    if (
        minimum_calibration < MINIMUM_ATTENTION_CALIBRATION_PLANS_V3
        or minimum_heldout < MINIMUM_ATTENTION_HELDOUT_PLANS_V3
        or minimum_contexts
        < MINIMUM_ATTENTION_CONTEXT_SAMPLES_PER_BAND_V3
        or policy_nonce_samples < MINIMUM_ATTENTION_POLICY_NONCE_SAMPLES_V3
        or maximum_escape > MAXIMUM_ZEROED_OUTPUT_ESCAPE_UPPER_BOUND_V3
    ):
        raise ProofV3Error("attention qualification coverage is too weak")
    if (
        minimum_attention_subaudits is not None
        and minimum_subaudits != minimum_attention_subaudits
    ):
        raise ProofV3Error(
            "attention qualification does not match the signed layer policy"
        )

    expected_inventory = tuple(
        (int(layer), len(heads))
        for layer, heads in sorted(calibration_set.bands[0].calibration.layers.items())
    )
    raw_inventory = value["inventory"]
    if not isinstance(raw_inventory, list):
        raise ProofV3Error("attention qualification inventory is malformed")
    inventory = []
    for item in raw_inventory:
        if not isinstance(item, dict) or set(item) != {"layer", "head_count"}:
            raise ProofV3Error("attention qualification inventory is malformed")
        layer = item["layer"]
        heads = item["head_count"]
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer < 0
            or isinstance(heads, bool)
            or not isinstance(heads, int)
            or heads < 1
        ):
            raise ProofV3Error("attention qualification inventory is malformed")
        inventory.append((layer, heads))
    if tuple(inventory) != expected_inventory:
        raise ProofV3Error(
            "attention qualification does not cover the exact layer/head inventory"
        )
    for band in calibration_set.bands[1:]:
        if tuple(
            (int(layer), len(heads))
            for layer, heads in sorted(band.calibration.layers.items())
        ) != expected_inventory:
            raise ProofV3Error("attention calibration bands disagree on inventory")

    raw_bands = value["bands"]
    if not isinstance(raw_bands, list) or len(raw_bands) != len(
        calibration_set.bands
    ):
        raise ProofV3Error("attention qualification band inventory is malformed")
    total_calibration = 0
    total_heldout = 0
    for raw, calibrated in zip(raw_bands, calibration_set.bands):
        required_band = {
            "lo",
            "hi",
            "context_lengths",
            "calibration_cases",
            "heldout_cases",
            "calibration_maximum_cell_utilization",
            "calibration_maximum_row_utilization",
            "heldout_maximum_cell_utilization",
            "heldout_maximum_row_utilization",
            "maximum_zeroed_output_escape_upper_bound",
        }
        if not isinstance(raw, dict) or set(raw) != required_band:
            raise ProofV3Error("attention qualification band is malformed")
        if raw["lo"] != calibrated.lo or raw["hi"] != calibrated.hi:
            raise ProofV3Error("attention qualification band range is wrong")
        context_lengths = raw["context_lengths"]
        if (
            not isinstance(context_lengths, list)
            or len(context_lengths) < minimum_contexts
            or tuple(context_lengths) != tuple(sorted(set(context_lengths)))
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or not calibrated.lo <= item <= calibrated.hi
                for item in context_lengths
            )
            or max(context_lengths) < max(calibrated.lo, calibrated.hi - 1)
        ):
            raise ProofV3Error(
                "attention qualification context coverage is incomplete"
            )

        split_cases = {}
        split_extrema = {}
        for split in ("calibration_cases", "heldout_cases"):
            cases = raw[split]
            if not isinstance(cases, list) or not cases:
                raise ProofV3Error("attention qualification cases are malformed")
            seen_ids = set()
            seen_plans = set()
            seen_contexts = set()
            plans_by_context = {context: set() for context in context_lengths}
            styles_by_context = {context: set() for context in context_lengths}
            maximum_cell = 0.0
            maximum_row = 0.0
            maximum_case_escape = 0.0
            for case in cases:
                if not isinstance(case, dict) or set(case) != {
                    "id",
                    "context_tokens",
                    "prompt_style",
                    "selection_plan_sha256",
                    "complete_inventory",
                    "maximum_cell_utilization",
                    "maximum_row_utilization",
                    "zeroed_output_escape_count",
                    "zeroed_output_escape_upper_bound",
                }:
                    raise ProofV3Error(
                        "attention qualification case is malformed"
                    )
                case_id = case["id"]
                context = case["context_tokens"]
                prompt_style = case["prompt_style"]
                plan = _sha256(
                    case["selection_plan_sha256"], "selection plan"
                )
                if (
                    not isinstance(case_id, str)
                    or not case_id
                    or case_id in seen_ids
                    or isinstance(context, bool)
                    or not isinstance(context, int)
                    or context not in context_lengths
                    or isinstance(prompt_style, bool)
                    or not isinstance(prompt_style, int)
                    or not 0 <= prompt_style < (
                        ATTENTION_QUALIFICATION_PROMPT_STYLE_COUNT_V3
                    )
                    or case["complete_inventory"] is not True
                ):
                    raise ProofV3Error(
                        "attention qualification case is malformed"
                    )
                seen_ids.add(case_id)
                seen_plans.add(plan)
                seen_contexts.add(context)
                plans_by_context[context].add(plan)
                styles_by_context[context].add(prompt_style)
                cell = _nonnegative_finite(
                    case["maximum_cell_utilization"], "cell utilization"
                )
                row = _nonnegative_finite(
                    case["maximum_row_utilization"], "row utilization"
                )
                escape_count = case["zeroed_output_escape_count"]
                if (
                    isinstance(escape_count, bool)
                    or not isinstance(escape_count, int)
                    or not 0 <= escape_count <= policy_nonce_samples
                ):
                    raise ProofV3Error(
                        "attention qualification escape count is malformed"
                    )
                case_escape = _nonnegative_finite(
                    case["zeroed_output_escape_upper_bound"],
                    "zeroed-output escape bound",
                )
                if not math.isclose(
                    case_escape,
                    _wilson_upper(escape_count, policy_nonce_samples),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise ProofV3Error(
                        "attention qualification escape bound is inconsistent"
                    )
                if (
                    cell > 1.0
                    or row > 1.0
                    or case_escape > maximum_escape
                ):
                    raise ProofV3Error(
                        "attention qualification case does not satisfy frozen bounds"
                    )
                maximum_cell = max(maximum_cell, cell)
                maximum_row = max(maximum_row, row)
                maximum_case_escape = max(maximum_case_escape, case_escape)
            required_plans = (
                minimum_calibration
                if split == "calibration_cases"
                else minimum_heldout
            )
            if (
                len(seen_plans) < required_plans
                or len(seen_plans) != len(cases)
                or seen_contexts != set(context_lengths)
                or any(
                    len(plans) < required_plans
                    for plans in plans_by_context.values()
                )
                or any(
                    styles != set(
                        range(ATTENTION_QUALIFICATION_PROMPT_STYLE_COUNT_V3)
                    )
                    for styles in styles_by_context.values()
                )
            ):
                raise ProofV3Error(
                    "attention qualification plan/context coverage is incomplete"
                )
            split_cases[split] = len(cases)
            split_extrema[split] = (
                maximum_cell,
                maximum_row,
                maximum_case_escape,
            )

        cal_cell = _nonnegative_finite(
            raw["calibration_maximum_cell_utilization"],
            "calibration cell utilization",
        )
        cal_row = _nonnegative_finite(
            raw["calibration_maximum_row_utilization"],
            "calibration row utilization",
        )
        held_cell = _nonnegative_finite(
            raw["heldout_maximum_cell_utilization"],
            "held-out cell utilization",
        )
        held_row = _nonnegative_finite(
            raw["heldout_maximum_row_utilization"],
            "held-out row utilization",
        )
        band_escape = _nonnegative_finite(
            raw["maximum_zeroed_output_escape_upper_bound"],
            "zeroed-output escape bound",
        )
        if (
            max(cal_cell, cal_row, held_cell, held_row) > 1.0
            or band_escape > maximum_escape
            or (cal_cell, cal_row)
            != split_extrema["calibration_cases"][:2]
            or (held_cell, held_row)
            != split_extrema["heldout_cases"][:2]
            or band_escape
            != max(
                split_extrema["calibration_cases"][2],
                split_extrema["heldout_cases"][2],
            )
        ):
            raise ProofV3Error("attention qualification band did not pass")
        total_calibration += split_cases["calibration_cases"]
        total_heldout += split_cases["heldout_cases"]

    if (
        value["calibration_case_count"] != total_calibration
        or value["heldout_case_count"] != total_heldout
    ):
        raise ProofV3Error("attention qualification case counts are inconsistent")


__all__ = [
    "ATTENTION_RUNTIME_QUALIFICATION_ABI_V3",
    "ATTENTION_QUALIFICATION_PROMPT_STYLE_COUNT_V3",
    "MINIMUM_ATTENTION_CALIBRATION_PLANS_V3",
    "MINIMUM_ATTENTION_CONTEXT_SAMPLES_PER_BAND_V3",
    "MINIMUM_ATTENTION_HELDOUT_PLANS_V3",
    "MINIMUM_ATTENTION_POLICY_NONCE_SAMPLES_V3",
    "MAXIMUM_ZEROED_OUTPUT_ESCAPE_UPPER_BOUND_V3",
    "validate_attention_runtime_qualification_v3",
]
