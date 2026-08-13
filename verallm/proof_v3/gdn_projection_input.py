"""Canonical GDN input-RMSNorm projection rows for compact hard audits."""

from __future__ import annotations

import math

from verallm.proof_v3.errors import ProofV3VerificationError

GDN_CANONICAL_NORM_INPUT_ABI_V3 = (
    "gdn.projection_input.authenticated_norm_source.canonical_f64.v1"
)


def canonical_gdn_projection_inputs_v3(
    *,
    source_rows_by_token,
    norm_row,
    norm_scale: float,
    norm_gain_offset: float,
    epsilon: float,
):
    """Return canonical ``(scale_bits, scale, int8 rows)``.

    Both prover and verifier consume the same exact FP16/BF16 residual source
    and registered int8 norm-weight center in binary64. This keeps appended
    decode rows inside the scale domain without trusting a backend-specific
    fused RMSNorm intermediate.
    """

    import numpy as np

    if (
        not source_rows_by_token
        or not math.isfinite(norm_scale)
        or norm_scale <= 0.0
        or not math.isfinite(norm_gain_offset)
        or not math.isfinite(epsilon)
        or epsilon <= 0.0
    ):
        raise ProofV3VerificationError(
            "canonical GDN RMSNorm projection input is malformed"
        )
    weights = np.asarray(norm_row, dtype=np.float64)
    if weights.ndim != 1 or not len(weights):
        raise ProofV3VerificationError(
            "canonical GDN RMSNorm weight row is malformed"
        )
    gains = weights * float(norm_scale) + float(norm_gain_offset)
    normalized = {}
    for token, source_values in source_rows_by_token.items():
        source = np.asarray(source_values, dtype=np.float64)
        if source.shape != weights.shape or not bool(np.isfinite(source).all()):
            raise ProofV3VerificationError(
                "canonical GDN RMSNorm source row is malformed"
            )
        denominator = math.sqrt(float(np.square(source).mean()) + epsilon)
        normalized[int(token)] = source * gains / denominator

    from verallm.proof_v3.economic_execution_anchor import (
        _absmax_scale_v3,
        _quantize_row_v3,
    )
    from verallm.proof_v3.economic_wire import (
        bits_to_scale_v3,
        scale_to_bits_v3,
    )

    scale_bits = scale_to_bits_v3(_absmax_scale_v3(normalized.values()))
    scale = bits_to_scale_v3(scale_bits)
    return (
        scale_bits,
        scale,
        {
            token: _quantize_row_v3(row, scale)
            for token, row in normalized.items()
        },
    )
