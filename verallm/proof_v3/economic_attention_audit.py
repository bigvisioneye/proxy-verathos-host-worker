"""Economic-tier dispatch of the sampled-chunk attention audit.

Reuses the built, tested chunk-audit orchestration (audit_flow.run_recompute_
audit_v3 over recompute_audit.verify_attention_chunk_reveal_v3 +
verify_attention_composition_v3) -- NOT the reference full-head recompute
(economic_attention_recompute.py, ~8.8s/head, spec only). The validator draws
nonce-selected chunks, recomputes them exactly, and requires composition to the
committed o_proj input; the exact recompute is bounded by the sampled chunks,
independent of context length (see docs/internal/proof_v3_economic_attention_
chunk_audit.md).

This module is the fail-closed economic dispatch + the probabilistic detection
bound. It is gated by the SIGNED manifest policy ``attn_audit_required`` so a
model that mandates attention auditing rejects any request whose attention audit
did not pass.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import comb

from verallm.proof_v3.audit_flow import AuditRevealRequestV3, run_recompute_audit_v3
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError

__all__ = [
    "EconomicAttentionVerdictV3",
    "attention_detection_probability_v3",
    "economic_attention_request_binding_v3",
    "dispatch_economic_attention_audit_v3",
    "establish_economic_attention_audit_v3",
]

_VERDICT_DOMAIN = b"VERATHOS/PROOF_V3/ECON_ATTN_VERDICT/V1"


@dataclass(frozen=True, slots=True)
class EconomicAttentionVerdictV3:
    """A REQUEST-BOUND attention-audit result.

    ``request_binding`` ties the verdict to one exact (envelope, challenge)
    request, so a True verdict from a prior request cannot be trusted for a
    later, unaudited one -- the hard adapter recomputes the binding from the
    request it is verifying and rejects any mismatch.
    """

    request_binding: bytes
    passed: bool


def economic_attention_request_binding_v3(*, envelope, challenge) -> bytes:
    """Deterministic per-request binding over the envelope + nonce-bound seed.

    ``challenge.selection_seed`` is derived solely from the validator nonce and
    the pre-nonce envelope, so it is unique per request and unforgeable before
    the nonce -- exactly what the verdict must be bound to.
    """

    return hashlib.sha256(
        _VERDICT_DOMAIN + envelope.digest() + bytes(challenge.selection_seed)
    ).digest()


def attention_detection_probability_v3(
    *, chunk_count: int, false_chunks: int, samples: int, canaries: int = 1
) -> float:
    """Exact miss probability for the sampled-chunk audit.

    With ``false_chunks`` of ``chunk_count`` corrupted and ``samples`` distinct
    chunks drawn without replacement, one canary misses with hypergeometric
    probability ``C(C-b, m) / C(C, m)``; ``canaries`` independent draws compound
    multiplicatively. Returns the probability the cheat is NEVER caught.
    """

    C = int(chunk_count)
    b = int(false_chunks)
    m = min(int(samples), C)
    if C < 1 or b < 0 or b > C or m < 0 or canaries < 1:
        raise ProofV3Error("detection-probability inputs are out of range")
    if b == 0:
        return 1.0
    if m >= C - b + 1 and b > 0:
        # once samples exceed the good-chunk count a hit is guaranteed per draw
        per_canary = 0.0 if m > C - b else comb(C - b, m) / comb(C, m)
    else:
        per_canary = comb(C - b, m) / comb(C, m)
    return per_canary ** int(canaries)


def dispatch_economic_attention_audit_v3(
    *,
    reveal,
    committed_attention_out,
    reveals_merkle_verified: bool,
    weight_rows_registry_verified: bool = True,
    argmax_ok: bool = True,
) -> None:
    """Run the sampled-chunk attention audit for one request; raise on failure.

    ``reveal`` is the coordinator's AttentionAuditRevealV3 (statement restricted
    to the audited rows, committed claims, and the nonce-sampled q_rows/k_chunks/
    v_chunks -- already Merkle-verified against the capture roots by the caller,
    which sets ``reveals_merkle_verified``). ``committed_attention_out`` is the
    committed attention output (o_proj input) rows for the audited layer.
    """

    outcome = run_recompute_audit_v3(
        statement=reveal.statement,
        claims=reveal.claims,
        request=AuditRevealRequestV3(
            layer=reveal.plan.layer,
            query_rows=tuple(range(reveal.statement.token_count)),
            chunk_indices=reveal.plan.chunk_indices,
        ),
        revealed_q_rows=reveal.revealed_q_rows,
        revealed_k_chunks=reveal.revealed_k_chunks,
        revealed_v_chunks=reveal.revealed_v_chunks,
        committed_attention_out=committed_attention_out,
        reveals_merkle_verified=reveals_merkle_verified,
        weight_rows_registry_verified=weight_rows_registry_verified,
        argmax_ok=argmax_ok,
    )
    if not outcome.passed:
        raise ProofV3VerificationError(
            "economic attention audit failed: "
            + "; ".join(outcome.failure_reasons)
        )


def establish_economic_attention_audit_v3(
    *,
    artifacts,
    envelope,
    challenge,
    reveal,
    committed_attention_out,
    reveals_merkle_verified: bool,
    weight_rows_registry_verified: bool = True,
    argmax_ok: bool = True,
) -> EconomicAttentionVerdictV3:
    """Validator glue: run the audit and record a REQUEST-BOUND verdict.

    Binds the verdict to (envelope, challenge) so the hard adapter can confirm
    it belongs to the request it is verifying (not a stale True from an earlier
    request). On failure records a bound ``passed=False`` verdict and re-raises.
    Call this before verify_economic_recompute_v3 when attn_audit_required.
    """

    binding = economic_attention_request_binding_v3(
        envelope=envelope, challenge=challenge)
    try:
        dispatch_economic_attention_audit_v3(
            reveal=reveal,
            committed_attention_out=committed_attention_out,
            reveals_merkle_verified=reveals_merkle_verified,
            weight_rows_registry_verified=weight_rows_registry_verified,
            argmax_ok=argmax_ok,
        )
    except ProofV3VerificationError:
        artifacts.attention_audit_verdict = EconomicAttentionVerdictV3(
            request_binding=binding, passed=False)
        raise
    verdict = EconomicAttentionVerdictV3(request_binding=binding, passed=True)
    artifacts.attention_audit_verdict = verdict
    return verdict
