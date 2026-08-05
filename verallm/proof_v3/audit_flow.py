"""Validator-side retroactive audit orchestration for proof-v3.

Ties the three already-tested pieces together into the flow a validator
runs against ONE served request:

1. ``derive_audit_plan_v3`` expands a fresh validator nonce (bound to the
   receipt's signed ``capture_chain_digest``) into (layer, rows, chunks)
   targets the miner could not predict at serve time.
2. The miner reveals the sampled slices; the validator Merkle-verifies
   them against the SIGNED capture roots (activations) and the ON-CHAIN
   registry root (weights) BEFORE any recompute -- that verification is
   the caller's job and its boolean result is passed in here.
3. This module runs the exact-integer recompute checkers over the
   verified material and folds every per-check verdict into one
   ``AuditOutcomeV3``, then gates it through the dual-stack rollout
   policy (``audit_passes``).

Pure orchestration: no torch, no chain, no transport.  Every heavy check
lives in ``recompute_audit`` / ``response_stamp`` and is called here.  The
config-B tail (succinct ZK argmax) stays routine -- ``argmax_ok`` is a
required input, not optional -- so the 1-of-vocab logit swap is closed
(see docs/internal/proof_v3_recompute_audit_security.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.recompute_audit import (
    AttentionChunkClaimsV3,
    AuditPlanV3,
    derive_audit_plan_v3,
    verify_attention_chunk_reveal_v3,
    verify_attention_composition_v3,
)
from verallm.proof_v3.rollout import ProofV3RolloutFlags, audit_passes

__all__ = [
    "AuditRevealRequestV3",
    "AuditCheckResultV3",
    "AuditOutcomeV3",
    "plan_audit_reveal_request_v3",
    "run_recompute_audit_v3",
    "gate_audit_outcome_v3",
]


@dataclass(frozen=True, slots=True)
class AuditRevealRequestV3:
    """What the validator asks the miner to open for one audit.

    Derived deterministically from the plan; the miner reconstructs the
    same targets from the same (nonce, digest) and cannot know them
    before the nonce is issued.
    """

    layer: int
    query_rows: tuple[int, ...]
    chunk_indices: tuple[int, ...]

    @classmethod
    def from_plan(cls, plan: AuditPlanV3) -> "AuditRevealRequestV3":
        return cls(
            layer=plan.layer,
            query_rows=plan.query_rows,
            chunk_indices=plan.chunk_indices,
        )


@dataclass(frozen=True, slots=True)
class AuditCheckResultV3:
    """One named sub-check verdict."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class AuditOutcomeV3:
    """Folded verdict over every sub-check of one audited request."""

    passed: bool
    checks: tuple[AuditCheckResultV3, ...] = field(default_factory=tuple)

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        return tuple(
            f"{c.name}: {c.detail}" for c in self.checks if not c.passed
        )


def plan_audit_reveal_request_v3(
    *,
    validator_nonce: bytes,
    capture_chain_digest: bytes,
    layer_count: int,
    sequence_length: int,
    chunk_size: int,
    row_samples: int = 32,
    chunk_samples: int = 3,
) -> AuditRevealRequestV3:
    """Nonce + signed digest -> the reveal targets for one request.

    ``capture_chain_digest`` is the value the receipt v3 signed
    pre-nonce; binding the plan to it means a miner cannot swap which
    request is being audited after seeing the nonce.
    """

    if not validator_nonce:
        raise ProofV3Error("audit reveal request needs a validator nonce")
    if not capture_chain_digest:
        raise ProofV3Error("audit reveal request needs the signed digest")
    plan = derive_audit_plan_v3(
        validator_nonce=validator_nonce,
        request_binding_digest=capture_chain_digest,
        layer_count=layer_count,
        sequence_length=sequence_length,
        chunk_size=chunk_size,
        row_samples=row_samples,
        chunk_samples=chunk_samples,
    )
    return AuditRevealRequestV3.from_plan(plan)


def run_recompute_audit_v3(
    *,
    statement,
    claims: AttentionChunkClaimsV3,
    request: AuditRevealRequestV3,
    revealed_q_rows,
    revealed_k_chunks: dict[int, object],
    revealed_v_chunks: dict[int, object],
    committed_attention_out,
    reveals_merkle_verified: bool,
    weight_rows_registry_verified: bool,
    argmax_ok: bool,
) -> AuditOutcomeV3:
    """Run every sub-check over Merkle-verified reveal material.

    Boolean gate inputs the caller must have established first:

    * ``reveals_merkle_verified`` -- the revealed activation slices open
      against the receipt's SIGNED capture roots.
    * ``weight_rows_registry_verified`` -- any revealed weight rows open
      against the model's ON-CHAIN registry root (validator holds only
      the manifest, never the weights -- the hard requirement).
    * ``argmax_ok`` -- the routine succinct ZK argmax tail verified
      (config B).  Required, not optional.

    Any False gate fails the audit before recompute even runs.  Recompute
    then re-derives the sampled chunks bit-exactly and compares to the
    published integer claims; a single mismatch fails the audit.
    """

    checks: list[AuditCheckResultV3] = []

    def _record(name: str, passed: bool, detail: str = "") -> None:
        checks.append(AuditCheckResultV3(name=name, passed=passed, detail=detail))

    _record(
        "reveals_merkle_verified",
        bool(reveals_merkle_verified),
        "" if reveals_merkle_verified
        else "revealed slices did not open against the signed capture roots",
    )
    _record(
        "weight_rows_registry_verified",
        bool(weight_rows_registry_verified),
        "" if weight_rows_registry_verified
        else "revealed weight rows did not open against the registry root",
    )
    _record(
        "argmax_tail",
        bool(argmax_ok),
        "" if argmax_ok else "succinct argmax tail did not verify",
    )

    # Only recompute once the material is proven to be the committed
    # material; recomputing over unverified reveals proves nothing.
    if reveals_merkle_verified:
        for chunk_index in request.chunk_indices:
            k_chunk = revealed_k_chunks.get(chunk_index)
            v_chunk = revealed_v_chunks.get(chunk_index)
            if k_chunk is None or v_chunk is None:
                _record(
                    f"chunk_reveal[{chunk_index}]",
                    False,
                    "miner did not reveal the sampled chunk",
                )
                continue
            try:
                # vectorized recompute (byte-identical); falls back to the
                # scalar reference when torch is unavailable. Bounded by the
                # sampled chunk regardless of context length.
                try:
                    from verallm.proof_v3.recompute_audit import (
                        verify_attention_chunk_reveal_fast_v3 as _reveal_check,
                    )

                    _reveal_check(
                        statement=statement, claims=claims,
                        chunk_index=chunk_index, q_rows=revealed_q_rows,
                        k_chunk=k_chunk, v_chunk=v_chunk)
                except (ImportError, RuntimeError):
                    verify_attention_chunk_reveal_v3(
                        statement=statement,
                        claims=claims,
                        chunk_index=chunk_index,
                        q_rows=revealed_q_rows,
                        k_chunk=k_chunk,
                        v_chunk=v_chunk,
                    )
                _record(f"chunk_reveal[{chunk_index}]", True)
            except ProofV3VerificationError as exc:
                _record(f"chunk_reveal[{chunk_index}]", False, str(exc))

        try:
            verify_attention_composition_v3(
                statement=statement,
                claims=claims,
                expected_out=committed_attention_out,
            )
            _record("composition", True)
        except ProofV3VerificationError as exc:
            _record("composition", False, str(exc))

    passed = all(c.passed for c in checks) and bool(checks)
    return AuditOutcomeV3(passed=passed, checks=tuple(checks))


def gate_audit_outcome_v3(
    *,
    flags: ProofV3RolloutFlags,
    outcome: AuditOutcomeV3 | None,
    v1_proof_valid: bool | None,
) -> bool:
    """Fold a v3 audit outcome through the dual-stack rollout policy.

    ``outcome=None`` means the miner presented no v3 stack; during the
    transition a valid v1 proof still passes, once mandatory it does not.
    """

    v3_valid = None if outcome is None else outcome.passed
    return audit_passes(
        flags,
        v1_proof_valid=v1_proof_valid,
        v3_receipt_valid=v3_valid,
    )
