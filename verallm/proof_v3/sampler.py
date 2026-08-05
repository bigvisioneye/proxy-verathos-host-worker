"""Canonical sampling-policy binding for proof-v3 requests.

Every v3 request binds the exact resolved vLLM sampling configuration before
execution.  Light-only organic requests may use stochastic sampling because
they claim request/output binding, not a decoding proof.  A request that can
select the hard execution tier is additionally required to use the qualified
greedy sampler relation.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping

from verallm.proof_v3.errors import ProofV3Error

ECONOMIC_GREEDY_SAMPLER_ABI_V3 = "sampler.greedy.v1"
_SAMPLER_DOMAIN = b"VERATHOS/PROOF_V3/SAMPLER/GREEDY/V1/SHA256"
_SAMPLER_KEYS = frozenset(
    (
        "max_tokens",
        "temperature",
        "presence_penalty",
        "top_k",
        "top_p",
        "min_p",
    )
)

__all__ = [
    "ECONOMIC_GREEDY_SAMPLER_ABI_V3",
    "economic_sampler_config_digest_v3",
    "validate_economic_hard_sampler_config_v3",
    "validate_economic_sampler_config_v3",
]


def _finite_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProofV3Error(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ProofV3Error(f"{name} must be a finite number")
    # Canonicalize negative zero before binary serialization.
    return 0.0 if result == 0.0 else result


def validate_economic_sampler_config_v3(
    *,
    do_sample: bool,
    resolved_sampling_params: Mapping[str, object],
    max_decode_tokens: int,
) -> tuple[int, float, float, int, float, float]:
    """Validate and return one exact sampler tuple bound by a v3 request."""

    if type(do_sample) is not bool:
        raise ProofV3Error("proof-v3 do_sample must be boolean")
    if not isinstance(resolved_sampling_params, Mapping):
        raise ProofV3Error("proof-v3 sampling configuration is malformed")
    if frozenset(resolved_sampling_params) != _SAMPLER_KEYS:
        raise ProofV3Error(
            "proof-v3 sampling configuration has an unexpected inventory"
        )
    if (
        isinstance(max_decode_tokens, bool)
        or not isinstance(max_decode_tokens, int)
        or not 0 < max_decode_tokens < 1 << 32
    ):
        raise ProofV3Error("proof-v3 maximum decode bound is malformed")

    max_tokens = resolved_sampling_params["max_tokens"]
    top_k = resolved_sampling_params["top_k"]
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 0 < max_tokens <= max_decode_tokens
    ):
        raise ProofV3Error(
            "proof-v3 requested decode length exceeds the signed profile"
        )
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k < -1
        or top_k >= 1 << 31
    ):
        raise ProofV3Error("proof-v3 top_k is malformed")

    temperature = _finite_float(
        resolved_sampling_params["temperature"],
        "proof-v3 temperature",
    )
    presence_penalty = _finite_float(
        resolved_sampling_params["presence_penalty"],
        "proof-v3 presence_penalty",
    )
    top_p = _finite_float(
        resolved_sampling_params["top_p"],
        "proof-v3 top_p",
    )
    min_p = _finite_float(
        resolved_sampling_params["min_p"],
        "proof-v3 min_p",
    )
    if not 0.0 <= temperature <= 100.0:
        raise ProofV3Error("proof-v3 temperature is out of range")
    if not -2.0 <= presence_penalty <= 2.0:
        raise ProofV3Error("proof-v3 presence penalty is out of range")
    if not 0.0 <= top_p <= 1.0 or not 0.0 <= min_p <= 1.0:
        raise ProofV3Error("proof-v3 probability cutoff is malformed")

    return (
        max_tokens,
        temperature,
        presence_penalty,
        top_k,
        top_p,
        min_p,
    )


def validate_economic_hard_sampler_config_v3(
    *,
    do_sample: bool,
    resolved_sampling_params: Mapping[str, object],
    max_decode_tokens: int,
) -> tuple[int, float, float, int, float, float]:
    """Require the exact greedy sampler relation qualified by hard v3."""

    result = validate_economic_sampler_config_v3(
        do_sample=do_sample,
        resolved_sampling_params=resolved_sampling_params,
        max_decode_tokens=max_decode_tokens,
    )
    _, temperature, presence_penalty, *_ = result
    if do_sample or temperature != 0.0 or presence_penalty != 0.0:
        raise ProofV3Error(
            "proof-v3 hard auditing requires greedy decoding with zero "
            "presence penalty"
        )
    return result


def economic_sampler_config_digest_v3(
    *,
    do_sample: bool,
    resolved_sampling_params: Mapping[str, object],
    max_decode_tokens: int,
) -> bytes:
    """Hash the canonical resolved sampler configuration."""

    (
        max_tokens,
        temperature,
        presence_penalty,
        top_k,
        top_p,
        min_p,
    ) = validate_economic_sampler_config_v3(
        do_sample=do_sample,
        resolved_sampling_params=resolved_sampling_params,
        max_decode_tokens=max_decode_tokens,
    )
    return hashlib.sha256(
        _SAMPLER_DOMAIN
        + struct.pack(
            "<?Iddiddd",
            do_sample,
            max_tokens,
            temperature,
            presence_penalty,
            top_k,
            top_p,
            min_p,
            0.0,
        )
    ).digest()
