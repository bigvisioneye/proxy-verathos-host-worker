"""Release gate for raw-QKV to real paged-K/V runtime equality."""

from __future__ import annotations

from verallm.proof_v3.attention_runtime_semantics import (
    ATTENTION_RUNTIME_SEMANTICS_ULP_VERSION_V3,
    AttentionRuntimeSemanticsV3,
)
from verallm.proof_v3.errors import ProofV3Error


ATTENTION_ANCHOR_QUALIFICATION_ABI_V3 = (
    "verathos.proof_v3.attention_anchor_qualification.v2"
)


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ProofV3Error(
            f"attention anchor qualification {name} is malformed"
        )
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProofV3Error(
            f"attention anchor qualification {name} is malformed"
        )
    return value


def validate_attention_anchor_qualification_v3(
    value: object,
    *,
    model_id: str,
    runtime_semantics: AttentionRuntimeSemanticsV3,
    attention_qualification: object,
) -> None:
    """Require exhaustive real-capture evidence for signed ULP semantics."""

    required = {
        "abi",
        "model_id",
        "runtime_semantics_digest",
        "runtime_fingerprint_sha256",
        "runtime_ulp_tolerance",
        "inventory",
        "bands",
        "case_count",
        "gate_passed",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ProofV3Error(
            "attention anchor qualification report is malformed"
        )
    if (
        not isinstance(runtime_semantics, AttentionRuntimeSemanticsV3)
        or runtime_semantics.version
        != ATTENTION_RUNTIME_SEMANTICS_ULP_VERSION_V3
        or runtime_semantics.runtime_ulp_tolerance < 1
    ):
        raise ProofV3Error(
            "attention anchor qualification requires signed ULP semantics"
        )
    if (
        value["abi"] != ATTENTION_ANCHOR_QUALIFICATION_ABI_V3
        or value["model_id"] != model_id
        or value["runtime_semantics_digest"]
        != runtime_semantics.digest().hex()
        or value["runtime_ulp_tolerance"]
        != runtime_semantics.runtime_ulp_tolerance
        or value["gate_passed"] is not True
    ):
        raise ProofV3Error(
            "attention anchor qualification does not match signed semantics"
        )
    if not isinstance(attention_qualification, dict):
        raise ProofV3Error(
            "attention anchor qualification has no corridor qualification"
        )
    fingerprint = _digest(
        value["runtime_fingerprint_sha256"], "runtime fingerprint"
    )
    if fingerprint != attention_qualification.get(
        "runtime_fingerprint_sha256"
    ):
        raise ProofV3Error(
            "attention anchor and corridor qualifications used different runtimes"
        )

    expected_inventory = attention_qualification.get("inventory")
    raw_inventory = value["inventory"]
    if (
        not isinstance(expected_inventory, list)
        or not isinstance(raw_inventory, list)
        or len(raw_inventory) != len(expected_inventory)
    ):
        raise ProofV3Error(
            "attention anchor qualification inventory is malformed"
        )
    expected_heads = []
    inventory = []
    for expected, item in zip(
        expected_inventory, raw_inventory, strict=True
    ):
        if (
            not isinstance(expected, dict)
            or set(expected) != {"layer", "head_count"}
            or not isinstance(item, dict)
            or set(item)
            != {"layer", "query_head_count", "kv_head_count"}
        ):
            raise ProofV3Error(
                "attention anchor qualification inventory is malformed"
            )
        layer = item["layer"]
        query_heads = item["query_head_count"]
        kv_heads = item["kv_head_count"]
        if (
            layer != expected["layer"]
            or query_heads != expected["head_count"]
            or isinstance(kv_heads, bool)
            or not isinstance(kv_heads, int)
            or kv_heads < 1
            or query_heads % kv_heads
        ):
            raise ProofV3Error(
                "attention anchor qualification inventory is malformed"
            )
        inventory.append((layer, query_heads, kv_heads))
        expected_heads.append(kv_heads)
    if len(inventory) != len(expected_inventory) or not inventory:
        raise ProofV3Error(
            "attention anchor qualification inventory is incomplete"
        )

    source_bands = attention_qualification.get("bands")
    bands = value["bands"]
    if (
        not isinstance(source_bands, list)
        or not isinstance(bands, list)
        or len(bands) != len(source_bands)
    ):
        raise ProofV3Error(
            "attention anchor qualification bands are malformed"
        )
    total_cases = 0
    rows_per_position = sum(expected_heads)
    for source, band in zip(source_bands, bands, strict=True):
        if (
            not isinstance(source, dict)
            or not isinstance(band, dict)
            or set(band) != {"lo", "hi", "cases"}
            or band["lo"] != source.get("lo")
            or band["hi"] != source.get("hi")
        ):
            raise ProofV3Error(
                "attention anchor qualification band is malformed"
            )
        expected_cases = [
            (split, case)
            for split, field in (
                ("calibration", "calibration_cases"),
                ("heldout", "heldout_cases"),
            )
            for case in source.get(field, ())
        ]
        cases = band["cases"]
        if not isinstance(cases, list) or len(cases) != len(expected_cases):
            raise ProofV3Error(
                "attention anchor qualification case coverage is incomplete"
            )
        for (expected_split, expected), case in zip(
            expected_cases, cases, strict=True
        ):
            required_case = {
                "id",
                "split",
                "context_tokens",
                "prompt_style",
                "complete_inventory",
                "position_count",
                "head_rows_compared",
                "maximum_k_ulp",
                "maximum_v_ulp",
                "out_of_envelope_count",
                "evidence_sha256",
            }
            if (
                not isinstance(expected, dict)
                or not isinstance(case, dict)
                or set(case) != required_case
                or case["id"] != expected.get("id")
                or case["split"] != expected_split
                or case["context_tokens"]
                != expected.get("context_tokens")
                or case["prompt_style"] != expected.get("prompt_style")
                or case["complete_inventory"] is not True
            ):
                raise ProofV3Error(
                    "attention anchor qualification case is malformed"
                )
            positions = _positive_int(
                case["position_count"], "position count"
            )
            if case["head_rows_compared"] != positions * rows_per_position:
                raise ProofV3Error(
                    "attention anchor qualification row count is inconsistent"
                )
            maximum_k_ulp = case["maximum_k_ulp"]
            maximum_v_ulp = case["maximum_v_ulp"]
            out_of_envelope = case["out_of_envelope_count"]
            # maximum_k_ulp is diagnostic only. The signed allowance applies
            # before RoPE and its exact propagated interval is represented by
            # out_of_envelope_count. Near-zero RoPE cancellation can make the
            # final-K ULP distance arbitrarily large without escaping that
            # narrow signed interval.
            if (
                isinstance(maximum_k_ulp, bool)
                or not isinstance(maximum_k_ulp, int)
                or maximum_k_ulp < 0
                or maximum_v_ulp != 0
                or out_of_envelope != 0
            ):
                raise ProofV3Error(
                    "attention anchor qualification runtime envelope failed"
                )
            _digest(case["evidence_sha256"], "case evidence")
            total_cases += 1
    if value["case_count"] != total_cases:
        raise ProofV3Error(
            "attention anchor qualification case count is inconsistent"
        )


__all__ = [
    "ATTENTION_ANCHOR_QUALIFICATION_ABI_V3",
    "validate_attention_anchor_qualification_v3",
]
