"""Complete succinct final-hidden to returned-token terminal path.

This coordinator is intentionally the only production entry point for the
terminal pair.  It commits the logits once, proves those exact values are the
registered LM-head evaluation of the committed final hidden row, and proves
the validator-observed token is their stable full-vocabulary argmax.  Every
terminal claim is deferred into the caller-owned global Goldilocks opening.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.economic_commitment import (
    EconomicCommittedOracleV3,
    oracle_leaf_index_v3,
    verify_economic_oracle_opening_v3,
)
from verallm.proof_v3.economic_lm_head_catalog_fold import (
    EconomicLmHeadCatalogBindingV3,
)
from verallm.proof_v3.economic_wire import (
    EconomicMerkleOpeningV3,
    EconomicOracleCommitmentV3,
    VALUE_MODE_INT8,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_terminal_argmax import (
    GoldilocksTerminalArgmaxProofV3,
    prove_goldilocks_terminal_argmax_v3,
    verify_goldilocks_terminal_argmax_v3,
)
from verallm.proof_v3.goldilocks_terminal_lm_head import (
    GoldilocksTerminalLmHeadProofV3,
    prove_goldilocks_terminal_lm_head_v3,
    verify_goldilocks_terminal_lm_head_v3,
)


GOLDILOCKS_TERMINAL_PATH_ABI_V3: Final = (
    "terminal.capture_final_hidden_lm_head_argmax.shared_fri.v2"
)
_LOGITS_TAG: Final = "terminal/argmax/logits"

__all__ = [
    "GOLDILOCKS_TERMINAL_PATH_ABI_V3",
    "GoldilocksTerminalPathProofV3",
    "prove_goldilocks_terminal_path_v3",
    "verify_goldilocks_terminal_path_v3",
]


@dataclass(frozen=True, slots=True)
class GoldilocksTerminalPathProofV3:
    final_hidden_opening: EconomicMerkleOpeningV3
    argmax: GoldilocksTerminalArgmaxProofV3
    lm_head: GoldilocksTerminalLmHeadProofV3

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.final_hidden_opening,
                EconomicMerkleOpeningV3,
            )
            or not isinstance(self.argmax, GoldilocksTerminalArgmaxProofV3)
            or not isinstance(
                self.lm_head,
                GoldilocksTerminalLmHeadProofV3,
            )
            or self.argmax.logits_commitment
            != self.lm_head.logits_commitment
        ):
            raise ProofV3Error("terminal path inventory is malformed")


def _ensure_fresh_terminal_namespace(collector) -> None:
    if collector is None:
        raise ProofV3Error("terminal path needs a global opening collector")
    for name in ("columns", "aliases", "claims"):
        values = getattr(collector, name, None)
        if not isinstance(values, dict):
            raise ProofV3Error("terminal path collector is malformed")
        if any(
            tag == _LOGITS_TAG or tag.startswith("terminal/")
            for tag in values
        ):
            raise ProofV3Error("terminal path collector namespace is occupied")


def prove_goldilocks_terminal_path_v3(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    binding: EconomicLmHeadCatalogBindingV3,
    committed_final_hidden: EconomicCommittedOracleV3,
    final_hidden_row: int,
    final_hidden_i8,
    logits,
    observed_token: int,
    weight_rows_i8,
    collector,
    fused=None,
) -> GoldilocksTerminalPathProofV3:
    """Prove one complete terminal path into the global opening set."""

    _ensure_fresh_terminal_namespace(collector)
    hidden = tuple(int(value) for value in final_hidden_i8)
    if (
        not isinstance(
            committed_final_hidden,
            EconomicCommittedOracleV3,
        )
        or isinstance(final_hidden_row, bool)
        or not isinstance(final_hidden_row, int)
        or not 0 <= final_hidden_row
        < committed_final_hidden.commitment.row_count
        or committed_final_hidden.commitment.oracle_id != "final_hidden"
        or committed_final_hidden.commitment.operation != "final_hidden"
        or committed_final_hidden.commitment.col_count != binding.hidden_dim
        or len(hidden) != binding.hidden_dim
        or any(
            committed_final_hidden.signed_value(
                final_hidden_row,
                column,
            )
            != value
            for column, value in enumerate(hidden)
        )
    ):
        raise ProofV3Error(
            "terminal final hidden disagrees with its capture commitment"
        )
    _leaves, final_hidden_opening = committed_final_hidden.open_cells(
        (
            (final_hidden_row, column)
            for column in range(binding.hidden_dim)
        ),
        value_mode=VALUE_MODE_INT8,
    )
    argmax = prove_goldilocks_terminal_argmax_v3(
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
        logits=logits,
        hidden_dim=binding.hidden_dim,
        observed_token=observed_token,
        collector=collector,
        fused=fused,
    )
    try:
        logits_column = collector.columns[_LOGITS_TAG]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ProofV3Error(
            "terminal argmax did not register its logits column"
        ) from exc
    lm_head = prove_goldilocks_terminal_lm_head_v3(
        validator_binding_digest=validator_binding_digest,
        validator_nonce=validator_nonce,
        binding=binding,
        hidden_row_i8=final_hidden_i8,
        logits_column=logits_column,
        weight_rows_i8=weight_rows_i8,
        collector=collector,
        fused=fused,
    )
    return GoldilocksTerminalPathProofV3(
        final_hidden_opening=final_hidden_opening,
        argmax=argmax,
        lm_head=lm_head,
    )


def _merge_registry(
    base: tuple[dict[str, object], dict[str, bytes]],
    extra: tuple[dict[str, object], dict[str, bytes]],
) -> tuple[dict[str, object], dict[str, bytes]]:
    statements = dict(base[0])
    commitments = dict(base[1])
    for tag, statement in extra[0].items():
        commitment = extra[1].get(tag)
        if tag in statements:
            if (
                statements[tag] != statement
                or commitments.get(tag) != commitment
            ):
                raise ProofV3VerificationError(
                    "terminal path duplicate registry entry disagrees"
                )
        else:
            if commitment is None:
                raise ProofV3VerificationError(
                    "terminal path registry has no commitment"
                )
            statements[tag] = statement
            commitments[tag] = commitment
    if set(extra[0]) != set(extra[1]):
        raise ProofV3VerificationError(
            "terminal path registry inventory is inconsistent"
        )
    return statements, commitments


def verify_goldilocks_terminal_path_v3(
    proof: object,
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    binding: EconomicLmHeadCatalogBindingV3,
    capture_base_binding_digest: bytes,
    final_hidden_oracle: EconomicOracleCommitmentV3,
    expected_final_hidden_row: int,
    final_hidden_i8,
    expected_observed_token: int,
    checker,
) -> tuple[dict[str, object], dict[str, bytes]]:
    """Verify the terminal pair and return its global-opening registry."""

    try:
        if not isinstance(proof, GoldilocksTerminalPathProofV3):
            raise ProofV3VerificationError(
                "terminal path proof has a wrong type"
            )
        hidden = tuple(int(value) for value in final_hidden_i8)
        if (
            not isinstance(
                final_hidden_oracle,
                EconomicOracleCommitmentV3,
            )
            or isinstance(expected_final_hidden_row, bool)
            or not isinstance(expected_final_hidden_row, int)
            or not 0 <= expected_final_hidden_row
            < final_hidden_oracle.row_count
            or final_hidden_oracle.oracle_id != "final_hidden"
            or final_hidden_oracle.operation != "final_hidden"
            or final_hidden_oracle.col_count != binding.hidden_dim
            or len(hidden) != binding.hidden_dim
        ):
            raise ProofV3VerificationError(
                "terminal final-hidden capture geometry is malformed"
            )
        expected_cells = tuple(
            oracle_leaf_index_v3(
                expected_final_hidden_row,
                column,
                final_hidden_oracle.col_count,
            )
            for column in range(final_hidden_oracle.col_count)
        )
        opened = verify_economic_oracle_opening_v3(
            oracle=final_hidden_oracle,
            base_binding=capture_base_binding_digest,
            expected_indices=expected_cells,
            opening=proof.final_hidden_opening,
            expected_mode=VALUE_MODE_INT8,
        )
        if tuple(opened[index] for index in expected_cells) != hidden:
            raise ProofV3VerificationError(
                "terminal final hidden is detached from its capture root"
            )
        argmax_registry = verify_goldilocks_terminal_argmax_v3(
            proof.argmax,
            validator_binding_digest=validator_binding_digest,
            validator_nonce=validator_nonce,
            expected_vocab=binding.vocab,
            expected_hidden_dim=binding.hidden_dim,
            expected_observed_token=expected_observed_token,
            checker=checker,
        )
        lm_head_registry = verify_goldilocks_terminal_lm_head_v3(
            proof.lm_head,
            validator_binding_digest=validator_binding_digest,
            validator_nonce=validator_nonce,
            binding=binding,
            hidden_row_i8=final_hidden_i8,
            logits_pcs_statement=argmax_registry[0][_LOGITS_TAG],
            expected_logits_commitment=argmax_registry[1][_LOGITS_TAG],
            checker=checker,
        )
        return _merge_registry(argmax_registry, lm_head_registry)
    except ProofV3VerificationError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "terminal path proof is malformed"
        ) from exc
