"""Capture-authenticated chain audit: the wiring that closes the hole.

The projection endpoint's hole (test_proof_v3_projection_audit_hole) is that
X is unauthenticated and Y is computed, so ``X=0, Y=0`` with the public
registered W passes.  Closing it needs TWO things joined, both of which
already exist as separate pieces:

* CAPTURE AUTHENTICATION -- the revealed rows must open against the
  per-request capture roots the miner committed PRE-NONCE
  (``capture_commit_v3`` / ``verify_capture_slice_reveal_v3``).  A miner
  cannot reveal values it did not commit before seeing the nonce.
* CHAIN BINDING -- the committed residual stream must anchor at the bottom
  to the real prompt embedding (manifest embedding root), chain layer to
  layer, and anchor at the top to the returned token
  (``verify_chain_binding_v3``).

Capture authentication alone does NOT stop ``X=0`` (a miner can commit
zeros).  The BOTTOM ANCHOR does: ``residual[0]`` must equal the embedding
of the real prompt tokens, so a zero (or any fabricated) trace is rejected.
This module joins the two so the endpoint audit is sound.

The per-op ``recompute`` callbacks are the exact tile checkers
(``recompute_audit``) over capture-authenticated rows; this module owns the
capture-authentication + anchoring glue, not the op semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from verallm.proof_v3.chain_binding import ChainStepV3, verify_chain_binding_v3
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_capture_commitment_reference import (
    capture_commit_v3,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.recompute_audit import (
    build_capture_slice_reveal_v3,
    verify_capture_slice_reveal_v3,
)

_HALF = GOLDILOCKS_MODULUS >> 1


def _to_field(v: int) -> int:
    v = int(v)
    return v + GOLDILOCKS_MODULUS if v < 0 else v


def _from_field(v: int) -> int:
    v = int(v)
    return v - GOLDILOCKS_MODULUS if v > _HALF else v

__all__ = [
    "ResidualCaptureV3",
    "commit_residual_capture_v3",
    "open_residual_rows_v3",
    "verify_residual_rows_authenticated_v3",
    "verify_capture_authenticated_chain_v3",
]


@dataclass(frozen=True, slots=True)
class ResidualCaptureV3:
    """A committed residual-stream tensor for one boundary of one request."""

    binding: bytes            # per-request+boundary binding digest
    width: int                # hidden dim
    token_count: int
    root: bytes               # capture Merkle root (pre-nonce)
    raw_values: tuple         # padded flat eval vector the root committed


def _flatten_rows(rows, width: int):
    t_pad = 1 << max(0, (len(rows) - 1).bit_length())
    w_pad = 1 << max(0, (width - 1).bit_length())
    flat = []
    for r in range(t_pad):
        row = rows[r] if r < len(rows) else ()
        for c in range(w_pad):
            flat.append(_to_field(row[c]) if c < len(row) else 0)
    return tuple(flat), t_pad, w_pad


def commit_residual_capture_v3(*, binding: bytes, rows, width: int) -> ResidualCaptureV3:
    """Commit one residual-stream tensor pre-nonce (int rows ``[t][dim]``)."""

    if not rows:
        raise ProofV3Error("residual capture needs rows")
    flat, t_pad, w_pad = _flatten_rows(rows, width)
    commitment = capture_commit_v3(
        validator_binding_digest=binding, raw_values=flat)
    return ResidualCaptureV3(
        binding=binding, width=w_pad, token_count=len(rows),
        root=commitment.root, raw_values=flat)


def open_residual_rows_v3(capture: ResidualCaptureV3, token_indices):
    """Miner-side: open the leaves of the requested token rows."""

    leaves = []
    for t in token_indices:
        base = t * capture.width
        leaves.extend(range(base, base + capture.width))
    opening = build_capture_slice_reveal_v3(
        validator_binding_digest=capture.binding,
        raw_values=capture.raw_values,
        indices=leaves,
    )
    return tuple(sorted(set(leaves))), opening


def verify_residual_rows_authenticated_v3(
    *, root: bytes, binding: bytes, width: int, leaf_count: int,
    token_indices, indices, opening,
) -> dict:
    """Validator-side: authenticate revealed rows against the capture root.

    Returns ``{token_index: row_tuple}`` reconstructed from the opened
    leaves.  The values come from the opening but are pinned to the
    pre-nonce ``root`` -- a miner cannot substitute them post-nonce.
    """

    values = verify_capture_slice_reveal_v3(
        capture_root=root, validator_binding_digest=binding,
        leaf_count=leaf_count, indices=indices, opening=opening)
    # map sorted leaf index -> value
    sorted_idx = tuple(sorted(set(int(i) for i in indices)))
    by_leaf = dict(zip(sorted_idx, values, strict=True))
    rows = {}
    for t in token_indices:
        base = t * width
        rows[t] = tuple(_from_field(by_leaf[base + c]) for c in range(width))
    return rows


def verify_capture_authenticated_chain_v3(
    *,
    prompt_tokens: tuple[int, ...],
    residual0_capture: ResidualCaptureV3,
    residual0_reveal,                      # (token_indices, indices, opening)
    embedding_row_for: Callable[[int], tuple],
    steps: tuple[ChainStepV3, ...],
    committed_final_residual: tuple,
    returned_token: int,
    lm_head_argmax: Callable[[tuple], int],
) -> None:
    """Authenticate residual[0] against its capture root, then bind the chain.

    This is the sound endpoint audit: the ``X=0/Y=0`` attack commits a zero
    residual[0]; it authenticates fine against the (miner-chosen) capture
    root, but FAILS the bottom anchor because ``0 != embedding(prompt)``.
    ``steps``/``final`` carry the rest of the chain (their rows are likewise
    capture-authenticated by the caller before being passed in).
    """

    token_indices, indices, opening = residual0_reveal
    leaf_count = residual0_capture.width * (
        1 << max(0, (residual0_capture.token_count - 1).bit_length()))
    authed = verify_residual_rows_authenticated_v3(
        root=residual0_capture.root, binding=residual0_capture.binding,
        width=residual0_capture.width, leaf_count=leaf_count,
        token_indices=token_indices, indices=indices, opening=opening)

    # the authenticated residual[0] rows are what the chain binds at the bottom
    residual0 = tuple(
        authed[t][:len(embedding_row_for(prompt_tokens[t]))]
        for t in range(len(prompt_tokens))
    )
    verify_chain_binding_v3(
        prompt_tokens=prompt_tokens,
        committed_residual0=residual0,
        embedding_row_for=embedding_row_for,
        steps=steps,
        committed_final_residual=committed_final_residual,
        returned_token=returned_token,
        lm_head_argmax=lm_head_argmax,
    )
