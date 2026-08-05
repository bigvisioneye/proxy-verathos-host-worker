"""Bind the capture-trace commitment to request identity + observed output.

Closes detached-trace and stale/cross-request-cache attacks: a miner that
serves request A but commits request B's (honest) trace, or replays a cached
trace, has bound its capture roots to the WRONG request.  The validator
re-derives the request digest from what it actually sent (prompt) and
observed (the SSE token stream + text), using the Stack-A request machinery,
and checks the receipt-signed binding.  A mismatched request digest ->
rejected, without the validator holding any weights.

The signed value is  H(domain + request_digest + capture_chain_digest)  where
request_digest folds prompt_token_root + output_token_root +
output_stream_digest (verallm.proof_v3.request).
"""

from __future__ import annotations

import hashlib

from verallm.proof_v3.errors import ProofV3VerificationError
from verallm.proof_v3.request import (
    ExecutionOutputBindingV3,
    PreExecutionRequestContextV3,
    derive_execution_request_digest_v3,
    token_sequence_digest_v3,
)

_BIND_DOMAIN = b"VERATHOS/PROOF_V3/REQUEST_BOUND_CAPTURE/V1"

__all__ = [
    "request_bound_capture_digest_v3",
    "verify_request_bound_capture_v3",
]


def request_bound_capture_digest_v3(
    *, request_digest: bytes, capture_chain_digest: bytes
) -> bytes:
    """The value the receipt signs: capture trace bound to the request."""

    if len(request_digest) != 32:
        raise ProofV3VerificationError("request_digest must be 32 bytes")
    if not capture_chain_digest:
        raise ProofV3VerificationError("capture_chain_digest is empty")
    return hashlib.sha256(
        _BIND_DOMAIN + request_digest + capture_chain_digest).digest()


def verify_request_bound_capture_v3(
    *,
    signed_bound_digest: bytes,
    capture_chain_digest: bytes,
    precommit_context: PreExecutionRequestContextV3,
    observed_output_token_ids,
    observed_text_utf8: bytes,
    finish_reason: str,
) -> bytes:
    """Validator-side: re-derive the request digest from OBSERVED prompt+output
    and check the receipt-signed binding.

    ``precommit_context`` is the validator's own pre-nonce context (its
    prompt_token_root, identities, nonce commitment...).  The output binding
    is derived from what the validator OBSERVED on the wire.  A detached or
    stale trace was bound to a different request -> the re-derived digest
    differs -> raise.  Returns the request_digest on success.
    """

    output = ExecutionOutputBindingV3.from_observed_output(
        output_token_ids=observed_output_token_ids,
        emitted_text_utf8=observed_text_utf8,
        finish_reason=finish_reason,
    )
    request_digest = derive_execution_request_digest_v3(
        precommit_context=precommit_context, output=output)
    expected = request_bound_capture_digest_v3(
        request_digest=request_digest, capture_chain_digest=capture_chain_digest)
    if expected != signed_bound_digest:
        raise ProofV3VerificationError(
            "capture trace is not bound to THIS request/output "
            "(detached trace or stale/cross-request cache)")
    return request_digest
