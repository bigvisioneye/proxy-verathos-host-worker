"""Canonical projection-corridor negative qualification records.

The signed corridor bounds admit honest runtime quantization error.  A release
qualification must therefore show that the same frozen bounds reject the
model-independent negative classes exercised by the hard verifier.  This
module validates the small, content-addressed result record; it does not run
the qualification cases or trust miner-provided evidence.
"""

from __future__ import annotations

from collections.abc import Iterable

from verallm.proof_v3.errors import ProofV3Error

PROJECTION_CORRIDOR_QUALIFICATION_ABI_V2 = (
    "verathos.proof_v3.projection_corridor_qualification.v2"
)
PROJECTION_CORRIDOR_NEGATIVE_GATE_ABI_V3 = (
    "verathos.proof_v3.projection_corridor_negative_gate.v1"
)

_COMMON_NEGATIVE_CASES_V3 = frozenset(
    (
        "registered_weight_mismatch",
        "detached_runtime_trace",
        "zero_runtime_state",
        "wrong_terminal_output",
    )
)
_FULL_ATTENTION_NEGATIVE_CASE_V3 = (
    "self_consistent_full_attention_substitute"
)
_GDN_NEGATIVE_CASE_V3 = "self_consistent_gdn_substitute"
_EVIDENCE_CLASSES_V3 = frozenset(
    (
        "real_model_wire",
        "qualified_relation_fixture",
        "retained_bundle_replay",
    )
)
_CASE_EVIDENCE_CLASSES_V3 = {
    "registered_weight_mismatch": frozenset(("real_model_wire",)),
    "detached_runtime_trace": frozenset(
        ("real_model_wire", "retained_bundle_replay")
    ),
    "zero_runtime_state": frozenset(
        ("real_model_wire", "retained_bundle_replay")
    ),
    "wrong_terminal_output": frozenset(
        ("real_model_wire", "retained_bundle_replay")
    ),
    _FULL_ATTENTION_NEGATIVE_CASE_V3: frozenset(
        ("real_model_wire", "qualified_relation_fixture")
    ),
    _GDN_NEGATIVE_CASE_V3: frozenset(
        ("real_model_wire", "qualified_relation_fixture")
    ),
}


def required_projection_corridor_negative_cases_v3(
    manifest_entry_names: Iterable[str],
) -> tuple[str, ...]:
    """Return the fail-closed negative inventory for one signed manifest."""

    names = tuple(str(name) for name in manifest_entry_names)
    required = set(_COMMON_NEGATIVE_CASES_V3)
    if any(name.rsplit(".", 1)[-1] == "qkv" for name in names):
        required.add(_FULL_ATTENTION_NEGATIVE_CASE_V3)
    if any(name.rsplit(".", 1)[-1] == "gdn_ba" for name in names):
        required.add(_GDN_NEGATIVE_CASE_V3)
    return tuple(sorted(required))


def validate_projection_corridor_negative_gate_v3(
    value,
    *,
    model_id: str,
    manifest_digest: str,
    manifest_entry_names: Iterable[str],
) -> tuple[dict, ...]:
    """Validate a completed, model-bound negative qualification result."""

    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "abi",
            "model_id",
            "manifest_digest",
            "case_count",
            "cases",
            "passed",
        }
        or value.get("abi") != PROJECTION_CORRIDOR_NEGATIVE_GATE_ABI_V3
        or value.get("model_id") != model_id
        or value.get("manifest_digest") != manifest_digest
        or value.get("passed") is not True
    ):
        raise ProofV3Error(
            "projection-corridor negative qualification is malformed"
        )
    cases = value.get("cases")
    case_count = value.get("case_count")
    if (
        isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or not isinstance(cases, list)
        or case_count != len(cases)
        or case_count < 1
    ):
        raise ProofV3Error(
            "projection-corridor negative qualification inventory is malformed"
        )

    parsed = []
    for item in cases:
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "case_id",
                "evidence_class",
                "trial_count",
                "rejected_count",
                "accepted_count",
                "evidence_sha256",
            }
        ):
            raise ProofV3Error(
                "projection-corridor negative qualification case is malformed"
            )
        case_id = item.get("case_id")
        evidence_class = item.get("evidence_class")
        trial_count = item.get("trial_count")
        rejected_count = item.get("rejected_count")
        accepted_count = item.get("accepted_count")
        digest = item.get("evidence_sha256")
        if (
            not isinstance(case_id, str)
            or not case_id
            or evidence_class not in _EVIDENCE_CLASSES_V3
            or evidence_class
            not in _CASE_EVIDENCE_CLASSES_V3.get(case_id, ())
            or isinstance(trial_count, bool)
            or not isinstance(trial_count, int)
            or trial_count < 1
            or isinstance(rejected_count, bool)
            or not isinstance(rejected_count, int)
            or isinstance(accepted_count, bool)
            or not isinstance(accepted_count, int)
            or rejected_count != trial_count
            or accepted_count != 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or digest.lower() != digest
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ProofV3Error(
                "projection-corridor negative qualification case did not pass"
            )
        parsed.append(dict(item))

    ids = tuple(item["case_id"] for item in parsed)
    if len(set(ids)) != len(ids) or ids != tuple(sorted(ids)):
        raise ProofV3Error(
            "projection-corridor negative qualification cases are not canonical"
        )
    required = required_projection_corridor_negative_cases_v3(
        manifest_entry_names
    )
    if ids != required:
        raise ProofV3Error(
            "projection-corridor negative qualification is incomplete"
        )
    return tuple(parsed)
