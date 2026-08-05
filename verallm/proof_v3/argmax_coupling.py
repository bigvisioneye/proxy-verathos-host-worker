"""Config-B argmax <-> committed-logits coupling (sampled-reveal).

The succinct argmax LogUp (``build_decode_argmax_audit_v3``) proves that
EVERY vocab diff ``logit[j*] - logit[j] - (j < j*)`` is a non-negative
24-bit value -- i.e. ``j*`` is the max of WHATEVER logit vector the diffs
were built from.  It does NOT, by itself, bind those diffs to the logits
the miner actually committed at serve time.  Without that binding a miner
could prove argmax over a fabricated vector while serving another.

This module supplies the missing binding the cheap way (the user's
ruling, 2026-07-20), matching the rest of the v3 recompute design:

* the winner cell ``logit[j*]`` is already bound to ``lm_head[j*] . hidden``
  by the response stamp (registry-verified lm_head row);
* here we nonce-sample ``k`` OTHER vocab cells, the miner reveals those
  committed logit cells (Merkle-opened against the logits capture root by
  the caller) plus the diff values at those positions (opened against the
  LogUp witness commitment by the caller), and we check the exact linear
  relation at each sampled cell.

Binds ``k`` unpredictable cells per audit; the full-vocab ``>= 0`` is
still proven by the LogUp, and sampling compounds across an epoch, so
sustained fabrication is caught within a handful of audits (see
docs/internal/proof_v3_recompute_audit_security.md, argmax-tail row).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError

_COUPLING_DOMAIN = b"VERATHOS/PROOF_V3/ARGMAX_COUPLING/V1"

__all__ = [
    "ArgmaxCouplingPlanV3",
    "derive_argmax_coupling_positions_v3",
    "verify_argmax_logit_coupling_v3",
]


@dataclass(frozen=True, slots=True)
class ArgmaxCouplingPlanV3:
    """Which vocab cells the miner must open to bind the argmax proof."""

    winner_index: int
    sampled_positions: tuple[int, ...]


def derive_argmax_coupling_positions_v3(
    *,
    validator_nonce: bytes,
    capture_chain_digest: bytes,
    winner_index: int,
    vocab_size: int,
    samples: int = 24,
) -> ArgmaxCouplingPlanV3:
    """Nonce + signed digest -> the vocab cells to reveal.

    Deterministic on both sides, unpredictable before the nonce, bound to
    the SIGNED ``capture_chain_digest`` so the audited request is fixed.
    The winner index itself is never sampled (its cell is bound by the
    response stamp); everything else is fair game, without replacement.
    """

    if vocab_size < 2:
        raise ProofV3Error("argmax coupling needs vocab >= 2")
    if not 0 <= winner_index < vocab_size:
        raise ProofV3Error("winner index is out of vocab range")
    if not validator_nonce or not capture_chain_digest:
        raise ProofV3Error("argmax coupling needs nonce + signed digest")

    population = vocab_size - 1  # exclude the winner cell
    count = min(samples, population)
    seed = hashlib.sha256(
        _COUPLING_DOMAIN
        + validator_nonce
        + capture_chain_digest
        + winner_index.to_bytes(8, "little")
    ).digest()

    bound = (1 << 64) - ((1 << 64) % population)
    chosen: list[int] = []
    seen: set[int] = set()
    counter = 0
    while len(chosen) < count:
        block = hashlib.sha256(
            seed + counter.to_bytes(8, "little")).digest()
        counter += 1
        for off in range(0, 32, 8):
            value = int.from_bytes(block[off:off + 8], "little")
            if value >= bound:
                continue
            index = value % population
            # map the reduced population back over the vocab, skipping j*
            if index >= winner_index:
                index += 1
            if index in seen:
                continue
            seen.add(index)
            chosen.append(index)
            if len(chosen) == count:
                break
    return ArgmaxCouplingPlanV3(
        winner_index=int(winner_index),
        sampled_positions=tuple(sorted(chosen)),
    )


def verify_argmax_logit_coupling_v3(
    *,
    plan: ArgmaxCouplingPlanV3,
    committed_winner_logit: int,
    revealed_logit_cells: dict[int, int],
    revealed_diffs: dict[int, int],
) -> None:
    """Check the exact argmax relation at every sampled cell.

    Inputs the caller must have Merkle-verified FIRST:

    * ``committed_winner_logit`` -- ``logit[j*]``, bound to the response
      stamp's ``lm_head[j*] . hidden`` output anchor (registry-verified).
    * ``revealed_logit_cells[j]`` -- committed ``logit[j]``, opened
      against the logits capture root.
    * ``revealed_diffs[j]`` -- the LogUp witness diff at ``j``, opened
      against the argmax witness commitment.

    Raises on the first mismatch.  The relation is exactly the one the
    diffs were built from:
    ``diff[j] == logit[j*] - logit[j] - (1 if j < j* else 0)``.
    """

    j_star = plan.winner_index
    if not plan.sampled_positions:
        raise ProofV3VerificationError("argmax coupling has no sampled cells")
    for j in plan.sampled_positions:
        if j == j_star:
            raise ProofV3VerificationError(
                "argmax coupling sampled the winner cell")
        if j not in revealed_logit_cells:
            raise ProofV3VerificationError(
                f"miner did not reveal committed logit cell {j}")
        if j not in revealed_diffs:
            raise ProofV3VerificationError(
                f"miner did not reveal argmax diff at cell {j}")
        expected = (
            int(committed_winner_logit)
            - int(revealed_logit_cells[j])
            - (1 if j < j_star else 0)
        )
        if int(revealed_diffs[j]) != expected:
            raise ProofV3VerificationError(
                "argmax diff does not match the committed logits "
                f"at cell {j} (proof is over a different logit vector)")
        if int(revealed_diffs[j]) < 0:
            raise ProofV3VerificationError(
                f"argmax diff at cell {j} is negative (winner not maximal)")
