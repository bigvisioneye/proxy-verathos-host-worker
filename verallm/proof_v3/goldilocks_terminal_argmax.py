"""Succinct full-vocabulary argmax over a shared Goldilocks opening.

The observed token is fixed by the request envelope.  For every active
vocabulary cell ``j`` this argument proves

    winner_logit - logits[j] - (j < winner) >= 0

by decomposing the difference into bounded byte limbs.  All limbs are checked
against the public byte table with LogUp.  One random multilinear evaluation
then binds the limbs to the committed logits, while a boolean-point evaluation
binds ``winner_logit`` to the observed token's exact cell.  Padding cells are
set to ``winner_logit`` with zero limbs, so they cannot affect the statement.

The logits column is intentionally supplied by the caller: the terminal
LM-head relation must separately bind that same column to
``final_hidden @ registered_lm_head``.  Both arguments defer their terminal
claims into the caller-owned batch-opening collector, allowing one global
BaseFold/FRI opening for the complete selected trace.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (
    GoldilocksSuccinctLogupStatementV3,
    logup_batch_registry_v3,
    prove_goldilocks_succinct_logup_v3,
    verify_goldilocks_succinct_logup_v3,
)
from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
    SuccinctEqFoldProofV3,
    column_pcs_statement_v3,
    commit_succinct_column_group_v3,
    commit_succinct_column_v3,
    prove_succinct_eq_fold_v3,
    verify_succinct_eq_fold_v3,
)


GOLDILOCKS_TERMINAL_ARGMAX_ABI_V3: Final = (
    "terminal.argmax.full_vocab.byte_logup.shared_fri.v1"
)
MAX_TERMINAL_ARGMAX_LIMBS_V3: Final = 8

_TRANSCRIPT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/TERMINAL_ARGMAX/FULL_VOCAB/BYTE_LOGUP/V1"
)
_LOGITS_TAG: Final = "terminal/argmax/logits"
_LIMB_GROUP_TAG: Final = "terminal/argmax/limbs"
_LOGUP_TAG: Final = "terminal/argmax/range"

__all__ = [
    "GOLDILOCKS_TERMINAL_ARGMAX_ABI_V3",
    "GoldilocksTerminalArgmaxProofV3",
    "MAX_TERMINAL_ARGMAX_LIMBS_V3",
    "prove_goldilocks_terminal_argmax_v3",
    "terminal_argmax_batch_registry_v3",
    "verify_goldilocks_terminal_argmax_v3",
]


def _fixed32(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    return value


def _pow2(value: int) -> int:
    return 1 << max(1, (value - 1).bit_length())


def _field_challenges(seed: bytes, count: int) -> tuple[int, ...]:
    result = []
    counter = 0
    while len(result) < count:
        block = hashlib.sha256(
            seed + struct.pack("<I", counter)
        ).digest()
        counter += 1
        for offset in range(0, 32, 8):
            value = int.from_bytes(block[offset : offset + 8], "little")
            if value < GOLDILOCKS_MODULUS:
                result.append(value)
                if len(result) == count:
                    break
    return tuple(result)


def _limb_tag(index: int) -> str:
    return f"{_LIMB_GROUP_TAG}/{index}"


def _limb_count(hidden_dim: int) -> int:
    # Each exact int8 LM-head logit has magnitude at most d*127^2.
    # The strict lower-index tie rule adds one to the full logit span.
    maximum = 2 * hidden_dim * 127 * 127 + 1
    count = max(1, (maximum.bit_length() + 7) // 8)
    if count > MAX_TERMINAL_ARGMAX_LIMBS_V3:
        raise ProofV3Error(
            "terminal argmax difference exceeds the supported exact range"
        )
    return count


def _canonical_logits(logits) -> tuple[int, ...]:
    """Materialize one logits vector without per-cell device synchronizations."""

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a runtime dependency.
        torch = None
    if torch is not None and isinstance(logits, torch.Tensor):
        if logits.ndim != 1:
            raise ProofV3Error("terminal logits tensor is not one-dimensional")
        try:
            logits = logits.detach().to(device="cpu").tolist()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ProofV3Error("terminal logits are malformed") from exc
    try:
        return tuple(int(value) for value in logits)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ProofV3Error("terminal logits are malformed") from exc


def _statement_tile(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    vocab: int,
    hidden_dim: int,
    observed_token: int,
    limb_count: int,
) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + GOLDILOCKS_TERMINAL_ARGMAX_ABI_V3.encode("ascii")
        + _fixed32(
            validator_binding_digest,
            "terminal argmax validator binding",
        )
        + _fixed32(validator_nonce, "terminal argmax validator nonce")
        + struct.pack(
            "<IIIB",
            vocab,
            hidden_dim,
            observed_token,
            limb_count,
        )
    ).digest()


def _relation_point(
    *,
    tile_digest: bytes,
    logits_commitment: bytes,
    limb_commitment: bytes,
    variable_count: int,
) -> tuple[int, ...]:
    return _field_challenges(
        hashlib.sha256(
            _TRANSCRIPT_DOMAIN
            + b"/relation-point/"
            + tile_digest
            + _fixed32(logits_commitment, "terminal logits commitment")
            + _fixed32(limb_commitment, "terminal limb commitment")
        ).digest(),
        variable_count,
    )


def _winner_point(
    observed_token: int,
    variable_count: int,
) -> tuple[int, ...]:
    return tuple(
        (observed_token >> bit) & 1
        for bit in range(variable_count)
    )


def _public_mle(values: tuple[int, ...], point: tuple[int, ...]) -> int:
    weights = [1]
    for coordinate in point:
        one_minus = (1 - coordinate) % GOLDILOCKS_MODULUS
        weights = [
            weight * one_minus % GOLDILOCKS_MODULUS
            for weight in weights
        ] + [
            weight * coordinate % GOLDILOCKS_MODULUS
            for weight in weights
        ]
    if len(weights) != len(values):
        raise ProofV3Error("terminal public MLE arity is malformed")
    return sum(
        value * weight
        for value, weight in zip(values, weights, strict=True)
    ) % GOLDILOCKS_MODULUS


@dataclass(frozen=True, slots=True)
class GoldilocksTerminalArgmaxProofV3:
    """Full-vocabulary argmax sub-proof without its global PCS opening."""

    vocab: int
    hidden_dim: int
    observed_token: int
    limb_count: int
    winner_logit: int
    logits_commitment: bytes
    limb_commitment: bytes
    grouped_logup_aux: bool
    range_proof: object
    logits_relation: SuccinctEqFoldProofV3
    limb_relations: tuple[SuccinctEqFoldProofV3, ...]
    winner_relation: SuccinctEqFoldProofV3

    def __post_init__(self) -> None:
        if (
            isinstance(self.vocab, bool)
            or not isinstance(self.vocab, int)
            or not 1 < self.vocab < 1 << 24
            or isinstance(self.hidden_dim, bool)
            or not isinstance(self.hidden_dim, int)
            or not 0 < self.hidden_dim < 1 << 24
            or isinstance(self.observed_token, bool)
            or not isinstance(self.observed_token, int)
            or not 0 <= self.observed_token < self.vocab
            or self.limb_count != _limb_count(self.hidden_dim)
            or not -(1 << 63) <= self.winner_logit < 1 << 63
            or type(self.grouped_logup_aux) is not bool
        ):
            raise ProofV3Error("terminal argmax proof geometry is malformed")
        _fixed32(self.logits_commitment, "terminal logits commitment")
        _fixed32(self.limb_commitment, "terminal limb commitment")
        limbs = tuple(self.limb_relations)
        if (
            len(limbs) != self.limb_count
            or not isinstance(self.logits_relation, SuccinctEqFoldProofV3)
            or not isinstance(self.winner_relation, SuccinctEqFoldProofV3)
            or not all(
                isinstance(item, SuccinctEqFoldProofV3)
                for item in limbs
            )
        ):
            raise ProofV3Error("terminal argmax proof inventory is malformed")
        object.__setattr__(self, "limb_relations", limbs)


def _commit_columns(
    *,
    tile_digest: bytes,
    logits: tuple[int, ...],
    observed_token: int,
    limb_count: int,
    fused,
):
    padded = _pow2(len(logits))
    winner = logits[observed_token]
    padded_logits = logits + (winner,) * (padded - len(logits))
    differences = tuple(
        winner - value - int(index < observed_token)
        for index, value in enumerate(logits)
    )
    if min(differences, default=0) < 0:
        raise ProofV3Error(
            "observed token is not the tie-stable full-vocabulary argmax"
        )
    if max(differences, default=0) >= 1 << (8 * limb_count):
        raise ProofV3Error(
            "terminal argmax difference exceeds the signed geometry bound"
        )
    padded_differences = differences + (0,) * (padded - len(logits))
    field_logits = tuple(value % GOLDILOCKS_MODULUS for value in padded_logits)
    limb_values = tuple(
        tuple(
            (value >> (8 * limb)) & 0xFF
            for value in padded_differences
        )
        for limb in range(limb_count)
    )
    logits_column = commit_succinct_column_v3(
        tile_digest=tile_digest,
        tag=_LOGITS_TAG,
        values=field_logits,
        fused=fused,
    )
    limb_group, limb_columns = commit_succinct_column_group_v3(
        tile_digest=tile_digest,
        group_tag=_LIMB_GROUP_TAG,
        ordered=tuple(
            (_limb_tag(index), values)
            for index, values in enumerate(limb_values)
        ),
        fused=fused,
    )
    return logits_column, limb_group, limb_columns, limb_values


def prove_goldilocks_terminal_argmax_v3(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    logits,
    hidden_dim: int,
    observed_token: int,
    collector,
    fused=None,
) -> GoldilocksTerminalArgmaxProofV3:
    """Prove the observed token is the full-vocabulary stable argmax.

    ``collector`` is mandatory.  The caller must later emit exactly one
    global opening covering this argument and the LM-head/execution relations.
    """

    logits_t = _canonical_logits(logits)
    if (
        not logits_t
        or len(logits_t) >= 1 << 24
        or any(not -(1 << 63) <= value < 1 << 63 for value in logits_t)
        or isinstance(hidden_dim, bool)
        or not isinstance(hidden_dim, int)
        or hidden_dim <= 0
        or isinstance(observed_token, bool)
        or not isinstance(observed_token, int)
        or not 0 <= observed_token < len(logits_t)
        or collector is None
    ):
        raise ProofV3Error("terminal argmax prover inputs are malformed")
    limb_count = _limb_count(hidden_dim)
    tile = _statement_tile(
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
        vocab=len(logits_t),
        hidden_dim=hidden_dim,
        observed_token=observed_token,
        limb_count=limb_count,
    )
    logits_column, limb_group, limb_columns, limb_values = _commit_columns(
        tile_digest=tile,
        logits=logits_t,
        observed_token=observed_token,
        limb_count=limb_count,
        fused=fused,
    )
    collector.register_column(_LOGITS_TAG, logits_column)
    collector.register_group(limb_group)
    for tag, column in limb_columns.items():
        collector.register_column(tag, column)

    variable_count = logits_column.pcs_statement.variable_count
    point = _relation_point(
        tile_digest=tile,
        logits_commitment=logits_column.tree.commitment,
        limb_commitment=limb_group.tree.commitment,
        variable_count=variable_count,
    )
    logits_relation = prove_succinct_eq_fold_v3(
        tile_digest=tile,
        column=logits_column,
        z_point=point,
        validator_nonce=validator_nonce,
        fused=fused,
        collector=collector,
    )
    limb_relations = tuple(
        prove_succinct_eq_fold_v3(
            tile_digest=tile,
            column=limb_columns[_limb_tag(index)],
            z_point=point,
            validator_nonce=validator_nonce,
            fused=fused,
            collector=collector,
        )
        for index in range(limb_count)
    )
    winner_relation = prove_succinct_eq_fold_v3(
        tile_digest=tile,
        column=logits_column,
        z_point=_winner_point(observed_token, variable_count),
        validator_nonce=validator_nonce,
        fused=fused,
        collector=collector,
    )

    logup_statement = GoldilocksSuccinctLogupStatementV3(
        validator_binding_digest=tile,
        table=tuple(range(256)),
        witness_variable_count=(
            limb_group.pcs_statement.variable_count
        ),
        witness_binding_override=(
            limb_group.pcs_statement.validator_binding_digest
        ),
    )
    flat_limbs = tuple(
        value
        for row in limb_values
        for value in row
    )
    if fused is None:
        range_proof = prove_goldilocks_succinct_logup_v3(
            statement=logup_statement,
            looked_up_values=flat_limbs,
            validator_nonce=validator_nonce,
            witness_tree=limb_group.tree,
            collector=collector,
            tag_prefix=_LOGUP_TAG,
            witness_tag=_LIMB_GROUP_TAG,
        )
    else:
        from verallm.proof_v3.native_pcs_backend import (
            fused_prove_logup_batch_v3,
        )

        range_proof = fused_prove_logup_batch_v3(
            fold_extension=fused[0],
            tree_extension=fused[1],
            tile_digest=tile,
            instances=(
                (
                    logup_statement,
                    limb_group,
                    _LOGUP_TAG,
                    _LIMB_GROUP_TAG,
                ),
            ),
            validator_nonce=validator_nonce,
            collector=collector,
        )[0]
    return GoldilocksTerminalArgmaxProofV3(
        vocab=len(logits_t),
        hidden_dim=hidden_dim,
        observed_token=observed_token,
        limb_count=limb_count,
        winner_logit=logits_t[observed_token],
        logits_commitment=logits_column.tree.commitment,
        limb_commitment=limb_group.tree.commitment,
        grouped_logup_aux=fused is not None,
        range_proof=range_proof,
        logits_relation=logits_relation,
        limb_relations=limb_relations,
        winner_relation=winner_relation,
    )


def _verifier_columns(
    proof: GoldilocksTerminalArgmaxProofV3,
    *,
    tile_digest: bytes,
):
    vocab_pad = _pow2(proof.vocab)
    variable_count = vocab_pad.bit_length() - 1
    block_bits = (proof.limb_count - 1).bit_length()
    logits_statement = column_pcs_statement_v3(
        tile_digest,
        _LOGITS_TAG,
        variable_count,
    )
    limb_group_statement = column_pcs_statement_v3(
        tile_digest,
        _LIMB_GROUP_TAG,
        variable_count + block_bits,
    )
    logits_column = SimpleNamespace(
        tag=_LOGITS_TAG,
        pcs_statement=logits_statement,
        tree=SimpleNamespace(commitment=proof.logits_commitment),
    )
    limb_columns = {}
    for index in range(proof.limb_count):
        limb_columns[_limb_tag(index)] = SimpleNamespace(
            tag=_limb_tag(index),
            pcs_statement=column_pcs_statement_v3(
                tile_digest,
                _limb_tag(index),
                variable_count,
            ),
            tree=SimpleNamespace(commitment=proof.limb_commitment),
            group_tag=_LIMB_GROUP_TAG,
            block_point=tuple(
                (index >> bit) & 1 for bit in range(block_bits)
            ),
        )
    return logits_column, limb_group_statement, limb_columns


def terminal_argmax_batch_registry_v3(
    proof: GoldilocksTerminalArgmaxProofV3,
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
) -> tuple[dict[str, object], dict[str, bytes]]:
    """Return exact global-opening statements and commitments."""

    tile = _statement_tile(
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
        vocab=proof.vocab,
        hidden_dim=proof.hidden_dim,
        observed_token=proof.observed_token,
        limb_count=proof.limb_count,
    )
    logits, limb_group_statement, _limbs = _verifier_columns(
        proof,
        tile_digest=tile,
    )
    statements = {
        _LOGITS_TAG: logits.pcs_statement,
        _LIMB_GROUP_TAG: limb_group_statement,
    }
    commitments = {
        _LOGITS_TAG: proof.logits_commitment,
        _LIMB_GROUP_TAG: proof.limb_commitment,
    }
    logup_statement = GoldilocksSuccinctLogupStatementV3(
        validator_binding_digest=tile,
        table=tuple(range(256)),
        witness_variable_count=limb_group_statement.variable_count,
        witness_binding_override=(
            limb_group_statement.validator_binding_digest
        ),
    )
    if proof.grouped_logup_aux:
        from verallm.proof_v3.native_pcs_backend import (
            logup_aux_group_plan_v3,
        )

        plans, group_meta = logup_aux_group_plan_v3(
            (
                (
                    _LOGUP_TAG,
                    logup_statement.witness_variable_count,
                    logup_statement.table_variable_count,
                ),
            )
        )
        roots = {
            "M": proof.range_proof.multiplicity_commitment,
            "D": proof.range_proof.inverse_commitments[0],
            "E": proof.range_proof.inverse_commitments[1],
        }
        for kind in ("M", "D", "E"):
            group_tag, _block_point = plans[kind][_LOGUP_TAG]
            variables, _used = group_meta[group_tag]
            statements[group_tag] = column_pcs_statement_v3(
                tile,
                group_tag,
                variables,
            )
            commitments[group_tag] = roots[kind]
    else:
        aux_statements, aux_commitments = logup_batch_registry_v3(
            proof.range_proof,
            logup_statement,
            _LOGUP_TAG,
            _LIMB_GROUP_TAG,
        )
        statements.update(aux_statements)
        commitments.update(aux_commitments)
    return statements, commitments


def verify_goldilocks_terminal_argmax_v3(
    proof: object,
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    expected_vocab: int,
    expected_hidden_dim: int,
    expected_observed_token: int,
    checker,
) -> tuple[dict[str, object], dict[str, bytes]]:
    """Verify the full-vocabulary argmax and collect global PCS claims."""

    try:
        if not isinstance(proof, GoldilocksTerminalArgmaxProofV3):
            raise ProofV3VerificationError(
                "terminal argmax proof has a wrong type"
            )
        if (
            proof.vocab != expected_vocab
            or proof.hidden_dim != expected_hidden_dim
            or proof.observed_token != expected_observed_token
            or proof.limb_count != _limb_count(expected_hidden_dim)
            or checker is None
        ):
            raise ProofV3VerificationError(
                "terminal argmax proof disagrees with signed geometry"
            )
        tile = _statement_tile(
            validator_binding_digest=validator_binding_digest,
            validator_nonce=validator_nonce,
            vocab=proof.vocab,
            hidden_dim=proof.hidden_dim,
            observed_token=proof.observed_token,
            limb_count=proof.limb_count,
        )
        logits, limb_group_statement, limbs = _verifier_columns(
            proof,
            tile_digest=tile,
        )
        for tag, column in limbs.items():
            checker.alias(tag, _LIMB_GROUP_TAG, column.block_point)
        point = _relation_point(
            tile_digest=tile,
            logits_commitment=proof.logits_commitment,
            limb_commitment=proof.limb_commitment,
            variable_count=logits.pcs_statement.variable_count,
        )
        logits_value = verify_succinct_eq_fold_v3(
            proof.logits_relation,
            tile_digest=tile,
            tag=_LOGITS_TAG,
            pcs_statement=logits.pcs_statement,
            commitment=proof.logits_commitment,
            z_point=point,
            validator_nonce=validator_nonce,
            checker=checker,
        )
        limb_values = tuple(
            verify_succinct_eq_fold_v3(
                relation,
                tile_digest=tile,
                tag=_limb_tag(index),
                pcs_statement=limbs[_limb_tag(index)].pcs_statement,
                commitment=proof.limb_commitment,
                z_point=point,
                validator_nonce=validator_nonce,
                checker=checker,
            )
            for index, relation in enumerate(proof.limb_relations)
        )
        winner_value = verify_succinct_eq_fold_v3(
            proof.winner_relation,
            tile_digest=tile,
            tag=_LOGITS_TAG,
            pcs_statement=logits.pcs_statement,
            commitment=proof.logits_commitment,
            z_point=_winner_point(
                proof.observed_token,
                logits.pcs_statement.variable_count,
            ),
            validator_nonce=validator_nonce,
            checker=checker,
        )
        if winner_value != proof.winner_logit % GOLDILOCKS_MODULUS:
            raise ProofV3VerificationError(
                "terminal argmax winner logit is detached from the logits"
            )
        tie = tuple(
            1 if index < proof.observed_token else 0
            for index in range(_pow2(proof.vocab))
        )
        tie_value = _public_mle(tie, point)
        decomposed = sum(
            value * (1 << (8 * index))
            for index, value in enumerate(limb_values)
        ) % GOLDILOCKS_MODULUS
        expected = (
            proof.winner_logit - logits_value - tie_value
        ) % GOLDILOCKS_MODULUS
        if decomposed != expected:
            raise ProofV3VerificationError(
                "terminal argmax limbs are detached from the logits"
            )
        logup_statement = GoldilocksSuccinctLogupStatementV3(
            validator_binding_digest=tile,
            table=tuple(range(256)),
            witness_variable_count=limb_group_statement.variable_count,
            witness_binding_override=(
                limb_group_statement.validator_binding_digest
            ),
        )
        if proof.grouped_logup_aux:
            from verallm.proof_v3.native_pcs_backend import (
                logup_aux_group_plan_v3,
            )

            plans, _group_meta = logup_aux_group_plan_v3(
                (
                    (
                        _LOGUP_TAG,
                        logup_statement.witness_variable_count,
                        logup_statement.table_variable_count,
                    ),
                )
            )
            for kind, local in (
                ("M", f"{_LOGUP_TAG}/M"),
                ("D", f"{_LOGUP_TAG}/D0"),
                ("E", f"{_LOGUP_TAG}/E0"),
            ):
                group_tag, block_point = plans[kind][_LOGUP_TAG]
                checker.alias(local, group_tag, block_point)
        verify_goldilocks_succinct_logup_v3(
            proof.range_proof,
            statement=logup_statement,
            witness_commitment=proof.limb_commitment,
            validator_nonce=validator_nonce,
            checker=checker,
            tag_prefix=_LOGUP_TAG,
            witness_tag=_LIMB_GROUP_TAG,
        )
        return terminal_argmax_batch_registry_v3(
            proof,
            validator_binding_digest=validator_binding_digest,
            validator_nonce=validator_nonce,
        )
    except ProofV3VerificationError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "terminal argmax proof is malformed"
        ) from exc
