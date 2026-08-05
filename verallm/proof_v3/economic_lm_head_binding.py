"""Full-vocabulary LM-head binding for the economic top anchor.

The reveal-path top anchor takes the argmax over the COMMITTED logits and
recomputes only the SAMPLED lm_head cells (plus the winner) against
``final_hidden @ lm_head``. Every UNSAMPLED committed logit is therefore
unbound to the model: a miner can commit the true argmax token's logit
artificially low at an unsampled cell so a wrong token appears to win. With
``k`` sampled cells over vocabulary ``V`` the suppressed cell evades sampling
with probability ``(V - k) / V`` (~0.9998 at k=32, V=150k), so sampling does
not close the gap.

This module binds EVERY committed logit to ``final_hidden @ lm_head`` in one
nonce-folded linear relation (``goldilocks_folded_linear_relation_reference``):
for the single audited decode position,

    L[j] == sum_k hidden[k] * lm_head[j, k]   for ALL j in [0, V)

is the relation ``X[1, d] @ W[d, V] == Y[1, V]`` with ``X = hidden``,
``W[k, j] = lm_head[j, k]`` (the registered signed weights the verifier owns)
and ``Y = L`` (the committed logits). The folded identity
``v^T X (W u) == v^T Y u`` fails if ANY single ``Y[j]`` is wrong (Schwartz-
Zippel, error ~ (1 + V) / p), so a suppressed true-winner cell is caught
regardless of which cell it is. Forged weights fail because the verifier folds
``W @ u`` from the weights it owns, not from anything the prover sends.

SCALE. The reference folded-linear statement materialises the full ``d * V``
weight tuple and hashes it into the statement digest -- fine at audit/test
scale, but ``d * V`` for a production LM head (~136M int8) is native-backend
territory (field-native weight commitment + FRI-query overlap), exactly like the
succinct argmax (see docs/internal/proof_v3_succinct_argmax_path_b.md). This
module is the SOUND ARGUMENT the native backend scales; it is wired behind the
signed-manifest ``lm_head_binding`` policy so the reveal path stays the default
until the native weight commitment lands.
"""

from __future__ import annotations

from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_folded_linear_relation_reference import (
    GoldilocksFoldedLinearPrecommitmentV3,
    GoldilocksFoldedLinearProofV3,
    GoldilocksFoldedLinearStatementV3,
    freeze_goldilocks_folded_linear_sources_v3,
    prove_goldilocks_folded_linear_reference_v3,
    verify_goldilocks_folded_linear_reference_v3,
)

__all__ = [
    "EconomicLmHeadBindingV3",
    "build_lm_head_binding_v3",
    "lm_head_binding_statement_v3",
    "verify_lm_head_binding_v3",
]


@dataclass(frozen=True, slots=True)
class EconomicLmHeadBindingV3:
    """The prover-side artifacts of one full-vocab lm_head binding."""

    precommitment: GoldilocksFoldedLinearPrecommitmentV3
    proof: GoldilocksFoldedLinearProofV3


def _weights_from_rows(lm_head_rows, *, in_dim: int, vocab: int) -> tuple[int, ...]:
    """Flatten registered lm_head rows into the folded ``W[k, j]`` layout.

    ``lm_head_rows[j]`` is the signed int8 weight row for output token ``j``
    (length ``in_dim``); the folded statement wants ``weights[k * vocab + j] =
    lm_head[j][k]``.
    """

    if len(lm_head_rows) != vocab:
        raise ProofV3Error("lm_head row count does not match the vocabulary")
    for row in lm_head_rows:
        if len(row) != in_dim:
            raise ProofV3Error("lm_head row width does not match the hidden dim")
    return tuple(
        lm_head_rows[j][k] for k in range(in_dim) for j in range(vocab)
    )


def lm_head_binding_statement_v3(
    *,
    validator_binding_digest: bytes,
    hidden_dim: int,
    vocab: int,
    lm_head_rows,
) -> GoldilocksFoldedLinearStatementV3:
    """Build the validator-owned folded statement over the registered weights.

    The caller MUST pass the authenticated registered lm_head rows (checked
    against the signed manifest root); the folded relation trusts the weights
    in the statement as verifier-owned, so the authentication is the binding
    to the registered model.
    """

    return GoldilocksFoldedLinearStatementV3(
        validator_binding_digest=validator_binding_digest,
        token_count=1,
        contraction_length=hidden_dim,
        output_features=vocab,
        weights=_weights_from_rows(
            lm_head_rows, in_dim=hidden_dim, vocab=vocab
        ),
    )


def build_lm_head_binding_v3(
    *,
    statement: GoldilocksFoldedLinearStatementV3,
    hidden_row,
    committed_logits,
    validator_nonce: bytes,
) -> EconomicLmHeadBindingV3:
    """Prove ``committed_logits == hidden_row @ lm_head`` for ALL vocab cells.

    ``hidden_row`` is the audited final-hidden int8 row (length ``hidden_dim``),
    already bound to the residual chain by the final-norm link; ``committed_logits``
    are the full committed logits the argmax runs over. Sources are frozen
    pre-nonce; the folded proof is post-nonce.
    """

    source_base, source_tree = freeze_goldilocks_folded_linear_sources_v3(
        statement=statement,
        x_values=(tuple(int(v) for v in hidden_row),),
        y_values=(tuple(int(v) for v in committed_logits),),
    )
    precommitment, proof = prove_goldilocks_folded_linear_reference_v3(
        statement=statement,
        source_base=source_base,
        source_tree=source_tree,
        validator_nonce=validator_nonce,
    )
    return EconomicLmHeadBindingV3(precommitment=precommitment, proof=proof)


def verify_lm_head_binding_v3(
    *,
    statement: GoldilocksFoldedLinearStatementV3,
    binding: EconomicLmHeadBindingV3,
    validator_nonce: bytes,
) -> None:
    """Validator-side: the committed logits ARE ``hidden @ registered_lm_head``.

    Raises ProofV3VerificationError on any mismatch, including a single
    suppressed/forged committed logit or forged weights.
    """

    if not isinstance(binding, EconomicLmHeadBindingV3):
        raise ProofV3VerificationError("lm_head binding has an unexpected type")
    verify_goldilocks_folded_linear_reference_v3(
        binding.proof,
        statement=statement,
        precommitment=binding.precommitment,
        validator_nonce=validator_nonce,
    )


# ---------------------------------------------------------------------------
# PRODUCTION full-vocab path: tensor-native authenticate + direct recompute.
#
# The folded-relation argument above is the succinct/reference form; at a real
# LM head (V x d ~ 136M int8) the reference cannot hold the weight tuple. This
# path binds EVERY committed logit at production scale WITHOUT a Python weight
# list, NTT or FRI: the validator loads the registered lm_head, re-derives the
# FlatWeightMerkle root ENTIRELY in torch int8 (byte-identical to the manifest
# reduction) to authenticate it, then recomputes logits = int8(lm_head) @
# int8(hidden) as one torch matmul and requires every committed cell to match.
# O(V*d) tensor work (~sub-second, GPU-trivial); the weight tree is static per
# model so the authentication caches. This is the real-model closure of the
# suppression gap; it needs the validator to hold the registered weights (which
# it authenticates against the signed manifest root), unlike the slice-only
# reveal path.
# ---------------------------------------------------------------------------


def _int8_reduce_tensor(weight):
    """torch int8 reduction byte-identical to ``int8_reduce_matrix``.

    absmax/127 symmetric, float64 division, round-half-to-even, clamp to
    [-128, 127] -- torch.round and Python round both round half to even, and
    float64 matches the Python-float reference, so the FlatWeightMerkle root
    over this tensor equals the manifest root exactly.
    """

    import torch

    if not isinstance(weight, torch.Tensor):
        weight = torch.as_tensor(weight)
    w64 = weight.detach().to(torch.float64)
    absmax = float(w64.abs().max()) if w64.numel() else 0.0
    if absmax < 1e-8:
        absmax = 1e-8
    scale = absmax / 127.0
    q = torch.round(w64 / scale).clamp(-128, 127).to(torch.int8)
    return q, scale


def authenticate_lm_head_v3(*, lm_head_weight, expected_root: bytes, chunk_size: int):
    """Authenticate registered lm_head weights against the signed manifest root.

    Returns the int8 weight tensor ``[V, d]`` on success. The weight tree is
    STATIC per model, so the validator authenticates once and caches the
    returned tensor; per-proof verification then reuses it via
    ``verify_full_vocab_lm_head_recompute_v3(int8_lm_head=...)`` at ms cost.
    """

    from zkllm.crypto.merkle import FlatWeightMerkle

    q, _scale = _int8_reduce_tensor(lm_head_weight)
    if q.dim() != 2:
        raise ProofV3VerificationError("lm_head weight is not a 2-D matrix")
    root = FlatWeightMerkle(q, chunk_size=chunk_size).root
    if root != expected_root:
        raise ProofV3VerificationError(
            "lm_head weights do not reproduce the registered manifest root"
        )
    return q


def verify_full_vocab_lm_head_recompute_v3(
    *,
    lm_head_weight=None,
    int8_lm_head=None,
    hidden_row_int8,
    committed_logits,
    expected_root: bytes | None = None,
    chunk_size: int | None = None,
) -> None:
    """Bind ALL committed logits to the registered lm_head at production scale.

    Pass either ``int8_lm_head`` (a pre-authenticated int8 tensor from
    ``authenticate_lm_head_v3`` -- the per-proof fast path, no root rebuild) or
    ``lm_head_weight`` + ``expected_root`` + ``chunk_size`` (authenticate then
    recompute in one call).

    ``hidden_row_int8`` is the audited final-hidden int8 row (length d, bound to
    the residual chain by the final-norm link); ``committed_logits`` the full
    committed logits vector (length V) the argmax runs over.

    Raises ProofV3VerificationError if the loaded weights do not reproduce the
    manifest root (wrong/forged weights) or if ANY committed logit differs from
    ``int8(lm_head) @ int8(hidden)`` (suppression / fabricated logits).
    """

    import torch  # noqa: F401

    if int8_lm_head is not None:
        q = int8_lm_head
        if getattr(q, "dim", None) is None or q.dim() != 2:
            raise ProofV3VerificationError("int8_lm_head is not a 2-D tensor")
    else:
        if lm_head_weight is None or expected_root is None or chunk_size is None:
            raise ProofV3VerificationError(
                "verify_full_vocab_lm_head_recompute_v3 needs int8_lm_head or "
                "lm_head_weight+expected_root+chunk_size"
            )
        q = authenticate_lm_head_v3(
            lm_head_weight=lm_head_weight,
            expected_root=expected_root,
            chunk_size=chunk_size,
        )
    vocab, hidden_dim = int(q.shape[0]), int(q.shape[1])
    hidden = tuple(int(v) for v in hidden_row_int8)
    committed = tuple(int(v) for v in committed_logits)
    if len(hidden) != hidden_dim:
        raise ProofV3VerificationError(
            "audited hidden row width does not match the lm_head hidden dim"
        )
    if len(committed) != vocab:
        raise ProofV3VerificationError(
            "committed logits count does not match the lm_head vocabulary"
        )
    if any(not -128 <= v <= 127 for v in hidden):
        raise ProofV3VerificationError("audited hidden row is not int8")
    # int8*int8 over d (~1.4e7 at d=896) is far below float64's 2^53 exact-
    # integer range, and CUDA implements float64 matmul (not integer matmul),
    # so the fp64 GEMM is bit-exact here and runs on GPU or CPU alike.
    h = torch.tensor(hidden, dtype=torch.float64, device=q.device)
    recomputed = torch.round(q.to(torch.float64) @ h).to(torch.int64)
    expected = torch.tensor(committed, dtype=torch.int64, device=q.device)
    mismatch = recomputed != expected
    if bool(mismatch.any()):
        cell = int(torch.nonzero(mismatch, as_tuple=False)[0][0])
        raise ProofV3VerificationError(
            f"committed logit {cell} != registered lm_head @ hidden -- a "
            "committed logit is not the model output (suppression / fabricated "
            "logits)"
        )


__all__.append("authenticate_lm_head_v3")
__all__.append("verify_full_vocab_lm_head_recompute_v3")
