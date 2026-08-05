"""Full-chain binding: the anti-substitution core (v1's actual hole).

v1 was substitutable because the committed computation was never forced to
be the real model on the real prompt: only gate_proj was pinned on-chain,
the layer-0 input binding was DISABLED, and the decode hidden rows were
not bound to the layer chain -- so a miner could substitute the model and
commit a fabricated ``final_hidden`` that argmaxes to the returned token.

This module verifies the binding that makes that impossible.  For every
audited request the miner commits the residual stream at each layer
boundary; the validator checks:

* BOTTOM ANCHOR   residual[0][t] == embedding_table[token[t]]  (revealed
  against the manifest embedding root) -- the input IS the real prompt.
* PER-OP RECOMPUTE  each layer's committed delta == the op applied to the
  committed residual, with weights revealed against the manifest.
* CHAINING (linchpin)  residual[L+1] == residual[L] + delta[L] -- a single
  connected forward, so no disconnected per-op-consistent fake trace.
* TOP ANCHOR   argmax(lm_head @ residual[N]) == returned token, lm_head
  revealed against the manifest.

Drop ANY one and there is a passing fake (that was v1).  With all four the
only way to satisfy the checks is to run the real model on the real prompt.

The layer op here is abstract (``op_apply``): in the real model it is the
attention/MLP tile recompute (RMSNorm + softmax + SiLU + projections).
This module owns the binding + chaining logic; the per-op checkers plug in
as ``op_apply``.  The adversarial test exercises substitution, fabricated
hidden, and a broken chain link.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from verallm.proof_v3.errors import ProofV3VerificationError

__all__ = [
    "ChainStepV3",
    "verify_chain_binding_v3",
]


@dataclass(frozen=True, slots=True)
class ChainStepV3:
    """One committed layer step of the residual chain.

    ``residual_in`` / ``residual_out`` are committed rows ``[token][dim]``;
    ``delta`` is the committed layer contribution; ``recompute`` returns the
    op applied to ``residual_in`` under the manifest-verified weights (the
    real per-op checker) -- the validator compares it to ``delta``.
    """

    layer: int
    residual_in: tuple
    delta: tuple
    residual_out: tuple
    recompute: Callable[[tuple], tuple]


def _rows_equal(a, b) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb) or any(int(x) != int(y) for x, y in zip(ra, rb)):
            return False
    return True


def verify_chain_binding_v3(
    *,
    prompt_tokens: tuple[int, ...],
    committed_residual0: tuple,
    embedding_row_for: Callable[[int], tuple],
    steps: tuple[ChainStepV3, ...],
    committed_final_residual: tuple,
    returned_token: int,
    lm_head_argmax: Callable[[tuple], int],
) -> None:
    """Verify the full chain binds the committed trace to the real model.

    ``embedding_row_for(token_id)`` returns the manifest-embedding-root-
    verified embedding row (caller Merkle-checks it); ``lm_head_argmax``
    returns argmax(lm_head @ final_hidden) with lm_head manifest-verified.
    Raises ``ProofV3VerificationError`` on the first broken binding.
    """

    # -- BOTTOM ANCHOR: layer-0 residual == real prompt embeddings --------
    if len(committed_residual0) != len(prompt_tokens):
        raise ProofV3VerificationError(
            "bottom anchor: residual[0] row count != prompt length")
    for t, token in enumerate(prompt_tokens):
        expected = embedding_row_for(token)
        if not _rows_equal((committed_residual0[t],), (expected,)):
            raise ProofV3VerificationError(
                f"bottom anchor: residual[0][{t}] != embedding[token {token}] "
                "(input is not the real prompt)")

    # -- CHAIN: recompute each op + enforce connection --------------------
    prev_out = committed_residual0
    for step in steps:
        if not _rows_equal(step.residual_in, prev_out):
            raise ProofV3VerificationError(
                f"chaining: layer {step.layer} input != previous layer output "
                "(disconnected trace)")
        # per-op recompute under manifest-verified weights
        recomputed = step.recompute(step.residual_in)
        if not _rows_equal(recomputed, step.delta):
            raise ProofV3VerificationError(
                f"recompute: layer {step.layer} delta != op(residual_in) "
                "(wrong weights or faked compute)")
        # residual add: residual_out == residual_in + delta
        added = tuple(
            tuple(int(a) + int(d) for a, d in zip(ri, dl))
            for ri, dl in zip(step.residual_in, step.delta)
        )
        if not _rows_equal(added, step.residual_out):
            raise ProofV3VerificationError(
                f"chaining: layer {step.layer} residual_out != in + delta")
        prev_out = step.residual_out

    # -- final residual is the chained output, not a fabrication ----------
    if not _rows_equal(committed_final_residual, prev_out):
        raise ProofV3VerificationError(
            "final hidden != last chained residual (fabricated hidden)")

    # -- TOP ANCHOR: argmax(lm_head @ final) == returned token ------------
    winner = lm_head_argmax(committed_final_residual)
    if int(winner) != int(returned_token):
        raise ProofV3VerificationError(
            "top anchor: argmax(lm_head @ final_hidden) != returned token")
