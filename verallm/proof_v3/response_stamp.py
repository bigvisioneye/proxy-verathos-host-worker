"""Per-response model-binding stamp (the "every response has a proof").

Two anchors, one at each end of the audited chain, attached to EVERY
response (~15KB wire, ~3ms miner CPU, ~0.5ms proxy CPU):

* INPUT anchor: one response-hash-derived prompt position's embedding
  row (Merkle path vs the ON-CHAIN registry root) must equal the
  committed layer-0 input row (path vs the capture root).
* OUTPUT anchor: the returned token's lm_head row (registry path)
  dotted with the committed final hidden row (capture path) must equal
  the committed winner logit (capture path).

A naively-substituted model fails BOTH on every response.  Adaptive
evasion requires computing the anchored cells under the real weights
per response and still faces the retroactive nonce audits over the
chain between the anchors (see
docs/internal/proof_v3_recompute_audit_security.md).

Weight-row verification against the registry root reuses the live v1
embedding-binding machinery; this module takes the verified rows as
inputs and owns index derivation plus the integer checks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from verallm.proof_v3.economic_challenge import (
    economic_attention_candidate_positions_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError

_STAMP_DOMAIN = b"VERATHOS/PROOF_V3/RESPONSE_STAMP/V2/BOUNDED_POOL"

__all__ = [
    "ResponseStampPlanV3",
    "derive_response_stamp_plan_v3",
    "verify_response_stamp_input_anchor_v3",
    "verify_response_stamp_output_anchor_v3",
]


@dataclass(frozen=True, slots=True)
class ResponseStampPlanV3:
    """Deterministic post-generation anchor positions for one response."""

    prompt_position: int
    winner_token_id: int


def derive_response_stamp_plan_v3(
    *,
    response_digest: bytes,
    prompt_length: int,
    winner_token_id: int,
) -> ResponseStampPlanV3:
    """Anchor positions fixed by the response content itself.

    Computable by miner and proxy alike, but only AFTER generation -
    a faker cannot know at prompt time which prompt position anchors.
    """

    if prompt_length < 1:
        raise ProofV3Error("stamp needs a non-empty prompt")
    seed = hashlib.sha256(_STAMP_DOMAIN + response_digest).digest()
    candidates = economic_attention_candidate_positions_v3(
        context_token_count=prompt_length
    )
    position = candidates[int.from_bytes(seed[:8], "little") % len(candidates)]
    return ResponseStampPlanV3(
        prompt_position=position,
        winner_token_id=int(winner_token_id),
    )


def verify_response_stamp_input_anchor_v3(
    *,
    embedding_row,
    committed_input_row,
) -> None:
    """INPUT anchor: registry embedding row == committed layer-0 row.

    ``embedding_row`` must already be Merkle-verified against the
    model's on-chain registry root; ``committed_input_row`` against the
    request's capture root (both by the caller/proxy)."""

    if len(embedding_row) != len(committed_input_row):
        raise ProofV3VerificationError("stamp input anchor shape mismatch")
    for expected, committed in zip(
        embedding_row, committed_input_row, strict=True
    ):
        if int(expected) != int(committed):
            raise ProofV3VerificationError(
                "stamp input anchor does not match the registered "
                "embedding table")


def verify_response_stamp_output_anchor_v3(
    *,
    lm_head_row,
    committed_hidden_row,
    committed_winner_logit: int,
) -> None:
    """OUTPUT anchor: lm_head[winner] . hidden == committed logit.

    ``lm_head_row`` registry-verified, ``committed_hidden_row`` and
    ``committed_winner_logit`` capture-verified by the caller."""

    if len(lm_head_row) != len(committed_hidden_row):
        raise ProofV3VerificationError("stamp output anchor shape mismatch")
    value = 0
    for weight, activation in zip(
        lm_head_row, committed_hidden_row, strict=True
    ):
        value += int(weight) * int(activation)
    if value != int(committed_winner_logit):
        raise ProofV3VerificationError(
            "stamp output anchor does not match the committed logit")
