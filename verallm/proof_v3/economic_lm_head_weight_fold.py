"""Weight-only succinct lm_head binding (validator folds revealed logits).

Because the HARD reveal path already exposes ALL logits, the validator does not
need a succinct opening of the logits at all -- it folds the REVEALED logits
itself. The only object that needs a commitment is the STATIC registered
lm_head weight. This removes the capture<->PCS bridge and its native FRI-overlap
requirement (see docs/internal/proof_v3_succinct_argmax_path_b.md, "REFINED").

Relation bound: for the audited decode position, ``L[j] == sum_k X[k] * W[k,j]``
for ALL j in [0, V), where ``X`` is the audited final-hidden int8 row, ``W[k,j]
= lm_head[j][k]`` the registered signed weights, ``L`` the committed logits.

Setup (owner, once per model, SIGNED): commit ``W`` as a multilinear PCS
``C_W`` over ``a = ceil_log2(d)`` weight-row vars (LSB) followed by
``b = ceil_log2(V)`` output vars, evaluations laid out ``idx = j*d_pad + k``.
``C_W`` (its commitment bytes) goes in the signed manifest.

Per proof (validator has revealed logits ``L`` and audited hidden ``X``):
1. ``z in F^b`` derived from ``C_W`` + validator nonce (post-commit, un-grindable).
2. ``L~(z) = sum_j eq(z,j) L[j]``  -- the validator computes this from the
   REVEALED logits (an MLE eval, no opening, no bridge).
3. Miner sends ``wu[k] = W~(bits(k), z)`` for k in [0, d) plus ONE batch opening
   proving each ``wu[k]`` is ``C_W`` evaluated at ``(bits(k), z)``.
4. Validator checks ``sum_k X[k] * wu[k] == L~(z) (mod p)`` and verifies the
   batch opening against the signed ``C_W``. The points ``(bits(k), z)`` are
   reconstructed by the validator (not trusted from the prover).

Soundness: the opening pins ``wu`` to the true committed weights; if ``L != X @
W`` the step-4 equality fails except with probability ``b / |F|`` over the
nonce-derived ``z`` (Schwartz-Zippel). A fabricated/suppressed logit at ANY cell
is caught. The validator never holds ``W`` and never bridges logits.

Scale: the one native step is the owner-side RS-NTT commit over ``d_pad*V_pad``
(~2^28 for a real LM head); static, amortised over every proof. This reference
is exact at any (d, V); tests exercise small shapes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from verallm.proof_v3.economic_commitment import signed_to_field_v3
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
    GoldilocksMultilinearPcsStatementV3,
    commit_goldilocks_multilinear_v3,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.goldilocks_succinct_batch_opening import (
    GoldilocksBatchOpeningClaimV3,
    prove_goldilocks_batch_opening_v3,
    verify_goldilocks_batch_opening_v3,
)

__all__ = [
    "EconomicLmHeadWeightCommitmentV3",
    "EconomicLmHeadWeightFoldV3",
    "commit_lm_head_weights_v3",
    "derive_lm_head_fold_point_v3",
    "build_lm_head_weight_fold_v3",
    "verify_lm_head_weight_fold_v3",
]

_FOLD_DOMAIN = b"VERATHOS/PROOF_V3/LM_HEAD_WEIGHT_FOLD/V1"


def _ceil_log2(n: int) -> int:
    if n < 1:
        raise ProofV3Error("dimension must be positive")
    return (n - 1).bit_length() if n > 1 else 1


def _mle_eval_lsb(values, point) -> int:
    """MLE of ``values`` at ``point`` (LSB-first: point[0] folds bit 0).

    Identical convention to the batch-opening / PCS reference so a claim value
    computed here matches the committed evaluations exactly.
    """

    from verallm.proof_v3.goldilocks_numpy import mle_eval_lsb_np

    if len(values) >= 1024:
        return mle_eval_lsb_np(values, point)
    work = [v % GOLDILOCKS_MODULUS for v in values]
    for z in point:
        z %= GOLDILOCKS_MODULUS
        work = [
            (work[2 * i] + z * (work[2 * i + 1] - work[2 * i]))
            % GOLDILOCKS_MODULUS
            for i in range(len(work) // 2)
        ]
    return work[0] % GOLDILOCKS_MODULUS


def _point_for_row(k: int, a_bits: int, z_point: tuple[int, ...]) -> tuple[int, ...]:
    """The MLE point ``(bits(k) LSB-first, z)`` selecting weight row ``k``."""

    return tuple((k >> i) & 1 for i in range(a_bits)) + tuple(z_point)


@dataclass(frozen=True, slots=True)
class EconomicLmHeadWeightCommitmentV3:
    """The static signed lm_head PCS commitment + its shape."""

    statement: GoldilocksMultilinearPcsStatementV3
    commitment: bytes
    a_bits: int          # ceil_log2(hidden_dim)
    b_bits: int          # ceil_log2(vocab)
    hidden_dim: int
    vocab: int


@dataclass(frozen=True, slots=True)
class EconomicLmHeadWeightFoldV3:
    """Per-proof weight-fold reveal: wu values + one batch opening."""

    wu: tuple[int, ...]                      # length hidden_dim (field elems)
    proof: object                            # GoldilocksBatchOpeningProofV3


def commit_lm_head_weights_v3(
    *, lm_head_rows, validator_binding_digest: bytes
):
    """Owner-side static commitment. Returns (commitment_meta, evaluations).

    ``lm_head_rows[j]`` is the signed int8 weight row for output token ``j``
    (length ``d``). ``evaluations`` is retained by the prover to answer folds.
    """

    vocab = len(lm_head_rows)
    if vocab < 1:
        raise ProofV3Error("lm_head has no rows")
    hidden_dim = len(lm_head_rows[0])
    if hidden_dim < 1 or any(len(r) != hidden_dim for r in lm_head_rows):
        raise ProofV3Error("lm_head rows are ragged or empty")
    a_bits = _ceil_log2(hidden_dim)
    b_bits = _ceil_log2(vocab)
    d_pad = 1 << a_bits
    v_pad = 1 << b_bits
    evaluations = [0] * (d_pad * v_pad)
    for j in range(vocab):
        row = lm_head_rows[j]
        base = j * d_pad
        for k in range(hidden_dim):
            evaluations[base + k] = signed_to_field_v3(int(row[k]))
    statement = GoldilocksMultilinearPcsStatementV3(
        validator_binding_digest=validator_binding_digest,
        variable_count=a_bits + b_bits,
    )
    tree = commit_goldilocks_multilinear_v3(
        statement=statement, evaluations=tuple(evaluations)
    )
    meta = EconomicLmHeadWeightCommitmentV3(
        statement=statement,
        commitment=tree.commitment,
        a_bits=a_bits,
        b_bits=b_bits,
        hidden_dim=hidden_dim,
        vocab=vocab,
    )
    return meta, tuple(evaluations)


def derive_lm_head_fold_point_v3(
    *, commitment: bytes, validator_nonce: bytes, b_bits: int
) -> tuple[int, ...]:
    """Nonce-derived eq point ``z in F^b`` (post-commit, un-grindable)."""

    out = []
    counter = 0
    while len(out) < b_bits:
        digest = hashlib.sha256(
            _FOLD_DOMAIN
            + commitment
            + validator_nonce
            + counter.to_bytes(4, "little")
        ).digest()
        for chunk in range(0, 32, 8):
            value = int.from_bytes(digest[chunk:chunk + 8], "little")
            if value < GOLDILOCKS_MODULUS:
                out.append(value)
                if len(out) == b_bits:
                    break
        counter += 1
    return tuple(out)


def build_lm_head_weight_fold_v3(
    *,
    meta: EconomicLmHeadWeightCommitmentV3,
    evaluations: tuple[int, ...],
    z_point: tuple[int, ...],
    validator_nonce: bytes,
) -> EconomicLmHeadWeightFoldV3:
    """Prover-side: fold the committed weights by ``z`` over the output vars."""

    if len(z_point) != meta.b_bits:
        raise ProofV3Error("fold point arity does not match the commitment")
    claims = []
    wu = []
    for k in range(meta.hidden_dim):
        point = _point_for_row(k, meta.a_bits, z_point)
        value = _mle_eval_lsb(evaluations, point)
        wu.append(value)
        claims.append(GoldilocksBatchOpeningClaimV3(point=point, value=value))
    tree = commit_goldilocks_multilinear_v3(
        statement=meta.statement, evaluations=evaluations
    )
    proof = prove_goldilocks_batch_opening_v3(
        pcs_statement=meta.statement,
        tree=tree,
        values=evaluations,
        claims=tuple(claims),
        validator_nonce=validator_nonce,
    )
    return EconomicLmHeadWeightFoldV3(wu=tuple(wu), proof=proof)


def verify_lm_head_weight_fold_v3(
    *,
    meta: EconomicLmHeadWeightCommitmentV3,
    fold: EconomicLmHeadWeightFoldV3,
    hidden_row_int8,
    revealed_logits,
    validator_nonce: bytes,
) -> None:
    """Validator-side: bind every revealed logit to the committed lm_head.

    Reconstructs ``z`` and the ``(bits(k), z)`` points itself, verifies the
    batch opening against the SIGNED commitment, and checks
    ``sum_k X[k] wu[k] == sum_j eq(z,j) L[j]``. Raises on any mismatch.
    """

    if not isinstance(fold, EconomicLmHeadWeightFoldV3):
        raise ProofV3VerificationError("lm_head weight fold has a wrong type")
    hidden = tuple(int(v) for v in hidden_row_int8)
    logits = tuple(int(v) for v in revealed_logits)
    if len(hidden) != meta.hidden_dim:
        raise ProofV3VerificationError(
            "audited hidden width does not match the lm_head commitment"
        )
    if len(logits) != meta.vocab:
        raise ProofV3VerificationError(
            "revealed logits count does not match the lm_head commitment"
        )
    if len(fold.wu) != meta.hidden_dim:
        raise ProofV3VerificationError("weight fold has the wrong wu length")
    if any(not -128 <= v <= 127 for v in hidden):
        raise ProofV3VerificationError("audited hidden row is not int8")

    z_point = derive_lm_head_fold_point_v3(
        commitment=meta.commitment,
        validator_nonce=validator_nonce,
        b_bits=meta.b_bits,
    )
    # Reconstruct the claims from z + k (do NOT trust prover-sent points).
    claims = tuple(
        GoldilocksBatchOpeningClaimV3(
            point=_point_for_row(k, meta.a_bits, z_point),
            value=fold.wu[k] % GOLDILOCKS_MODULUS,
        )
        for k in range(meta.hidden_dim)
    )
    verify_goldilocks_batch_opening_v3(
        fold.proof,
        pcs_statement=meta.statement,
        commitment=meta.commitment,
        claims=claims,
        validator_nonce=validator_nonce,
    )
    # Fold the revealed logits directly (padded to the committed vocab width).
    v_pad = 1 << meta.b_bits
    logits_field = [signed_to_field_v3(v) for v in logits] + [0] * (
        v_pad - meta.vocab
    )
    lhs = _mle_eval_lsb(logits_field, z_point)
    rhs = 0
    for k in range(meta.hidden_dim):
        rhs = (rhs + signed_to_field_v3(hidden[k]) * fold.wu[k]) % GOLDILOCKS_MODULUS
    if lhs != rhs:
        raise ProofV3VerificationError(
            "revealed logits are not hidden @ registered lm_head "
            "(fold mismatch -- fabricated/suppressed logits)"
        )
