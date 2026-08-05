"""Exact multi-step GDN recurrence tile reference over Goldilocks.

Transition vertical for Gated-DeltaNet-class layers (Qwen3.5/3.6): an
S-step gated recurrence is proven exactly in fixed point, generalising the
two-row toy in ``gdn_reference`` to arbitrary bounded step counts using
the shipped primitive set (packed-pair LogUp for the gate nonlinearity,
Euclidean identities with byte-limb range proofs for every fixed-point
rescale).

Recurrence (scalar channel; the vector case is this channel-parallel):

    decay[t] = DecayTable[gate[t]]                      (0 <= decay <= 2^16)
    carried  = floor(decay[t] * state[t-1] / 2^16)      (exact, remainder)
    state[t] = carried + k[t] * v[t]
    y[t]     = floor(q[t] * state[t] / 2^16)            (exact, remainder)

with ``state[-1] = 0``.  All inputs int8; ``decay <= 2^16`` makes
``carried <= state[t-1]``, so ``|state[t]| <= S * 127^2`` — every identity
is an exact integer identity in the field.

The gate nonlinearity is signed table semantics: each ``(gate + 128,
decay)`` pair must be a member of the validator-owned packed decay table
(dual-challenge LogUp).  Both Euclidean remainders per step are proven in
``[0, 2^16)`` by two byte limbs each (byte-table LogUp).

Chronology and verification posture match the other tiles: columns freeze
pre-nonce in one tree, challenges derive post-freeze, the reference opens
in full while production keeps the identical column/lookup ABI in the AIR.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_linear_relation_reference import (
    _fixed32,
    _int8,
    _integer,
    _u32,
)
from verallm.proof_v3.goldilocks_logup_reference import (
    GoldilocksLogupProofV3,
    GoldilocksLogupStatementV3,
    freeze_goldilocks_logup_witness_v3,
    verify_goldilocks_logup_reference_v3,
)
from verallm.proof_v3.goldilocks_merkle_reference import (
    GoldilocksMerkleTreeReference,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS


GOLDILOCKS_GDN_TILE_ABI_V3: Final = "goldilocks.gdn_tile.reference.v1"
GDN_SCALE_V3: Final = 1 << 16
MAX_GOLDILOCKS_GDN_STEPS_V3: Final = 1 << 10
_GPACK: Final = 1 << 32

_STATEMENT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_GDN/V1/STATEMENT/SHA256"
_COLUMNS_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_GDN/V1/COLUMNS/SHA256"
_DECAY_BINDING_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_GDN/V1/DECAY/SHA256"
_BYTE_BINDING_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_GDN/V1/BYTES/SHA256"


def _signed_field(value: int) -> int:
    return value % GOLDILOCKS_MODULUS


def _from_field_signed(value: int) -> int:
    return value - GOLDILOCKS_MODULUS if value >= GOLDILOCKS_MODULUS // 2 else value


@dataclass(frozen=True, slots=True)
class GoldilocksGdnTileStatementV3:
    """Validator-owned recurrence statement: step count and decay table."""

    validator_binding_digest: bytes
    step_count: int
    decay_table: tuple[int, ...]  # indexed by gate + 128, values in [0, 2^16]

    def __post_init__(self) -> None:
        binding = _fixed32(
            self.validator_binding_digest, "gdn binding", nonzero=True
        )
        steps = _u32(self.step_count, "step_count", positive=True)
        if steps > MAX_GOLDILOCKS_GDN_STEPS_V3:
            raise ProofV3Error("gdn step count exceeds the CPU reference cap")
        if not isinstance(self.decay_table, tuple) or len(self.decay_table) != 256:
            raise ProofV3Error("gdn decay table must cover every int8 gate")
        table = tuple(
            _integer(value, f"decay_table[{index}]")
            for index, value in enumerate(self.decay_table)
        )
        if any(not 0 <= value <= GDN_SCALE_V3 for value in table):
            raise ProofV3Error("gdn decay value is out of the unit window")
        object.__setattr__(self, "validator_binding_digest", binding)
        object.__setattr__(self, "step_count", steps)
        object.__setattr__(self, "decay_table", table)

    def digest(self) -> bytes:
        return hashlib.sha256(
            _STATEMENT_DOMAIN
            + self.validator_binding_digest
            + struct.pack("<I", self.step_count)
            + b"".join(value.to_bytes(4, "little") for value in self.decay_table)
        ).digest()

    def columns_binding_digest(self) -> bytes:
        return hashlib.sha256(_COLUMNS_DOMAIN + self.digest()).digest()

    def decay_logup_statement(self) -> GoldilocksLogupStatementV3:
        return GoldilocksLogupStatementV3(
            validator_binding_digest=hashlib.sha256(
                _DECAY_BINDING_DOMAIN + self.digest()
            ).digest(),
            table=tuple(
                index + _GPACK * value
                for index, value in enumerate(self.decay_table)
            ),
        )

    def byte_logup_statement(self) -> GoldilocksLogupStatementV3:
        return GoldilocksLogupStatementV3(
            validator_binding_digest=hashlib.sha256(
                _BYTE_BINDING_DOMAIN + self.digest()
            ).digest(),
            table=tuple(range(256)),
        )


@dataclass(frozen=True, slots=True)
class GoldilocksGdnTileProofV3:
    columns_opening: tuple[tuple[int, ...], ...]
    decay_proof: GoldilocksLogupProofV3
    decay_roots: tuple[bytes, bytes]
    byte_proof: GoldilocksLogupProofV3
    byte_roots: tuple[bytes, bytes]


@dataclass(frozen=True, slots=True)
class GoldilocksGdnTileWitnessV3:
    statement: GoldilocksGdnTileStatementV3
    columns_tree: GoldilocksMerkleTreeReference
    decay_witness_tree: GoldilocksMerkleTreeReference
    decay_multiplicity_tree: GoldilocksMerkleTreeReference
    byte_witness_tree: GoldilocksMerkleTreeReference
    byte_multiplicity_tree: GoldilocksMerkleTreeReference


def run_and_freeze_goldilocks_gdn_tile_v3(
    *,
    statement: GoldilocksGdnTileStatementV3,
    q: tuple[int, ...],
    k: tuple[int, ...],
    v: tuple[int, ...],
    gate: tuple[int, ...],
    outputs: tuple[int, ...] | None = None,
) -> tuple[GoldilocksGdnTileWitnessV3, tuple[int, ...]]:
    """Execute the exact recurrence, freeze all columns, return outputs.

    Passing forged ``outputs`` freezes them as-is (with recomputed honest
    remainders where possible) so adversarial tests can exercise the
    verifier; the recurrence itself is never silently repaired.
    """

    steps = statement.step_count
    for name, column in (("q", q), ("k", k), ("v", v), ("gate", gate)):
        if not isinstance(column, tuple) or len(column) != steps:
            raise ProofV3Error(f"gdn {name} column has an unexpected shape")
        for value in column:
            _int8(value, f"gdn {name} value")
    rows: list[tuple[int, ...]] = []
    decay_pairs: list[int] = []
    byte_values: list[int] = []
    state = 0
    honest_outputs: list[int] = []
    for t in range(steps):
        decay = statement.decay_table[gate[t] + 128]
        decay_pairs.append((gate[t] + 128) + _GPACK * decay)
        product = decay * state
        shifted = product + (1 << 40)  # keep the dividend nonnegative
        carried_shift, remainder_a = divmod(shifted, GDN_SCALE_V3)
        carried = carried_shift - (1 << 24)
        new_state = carried + k[t] * v[t]
        y_product = q[t] * new_state + (1 << 40)
        y_shift, remainder_b = divmod(y_product, GDN_SCALE_V3)
        y_value = y_shift - (1 << 24)
        honest_outputs.append(y_value)
        out_value = y_value if outputs is None else outputs[t]
        byte_values.extend((remainder_a >> (8 * i)) & 0xFF for i in range(2))
        byte_values.extend((remainder_b >> (8 * i)) & 0xFF for i in range(2))
        rows.append(
            (
                _signed_field(q[t]),
                _signed_field(k[t]),
                _signed_field(v[t]),
                _signed_field(gate[t]),
                decay,
                _signed_field(state),
                _signed_field(carried),
                remainder_a,
                _signed_field(new_state),
                _signed_field(out_value),
                remainder_b,
            )
        )
        state = new_state
    width = len(rows[0])
    padded = 1 << max(1, (len(rows) - 1).bit_length())
    while len(rows) < padded:
        rows.append((0,) * width)
    columns_tree = GoldilocksMerkleTreeReference.from_rows(
        tuple(rows), binding_digest=statement.columns_binding_digest()
    )
    decay_w, decay_m = freeze_goldilocks_logup_witness_v3(
        statement=statement.decay_logup_statement(),
        looked_up_values=tuple(decay_pairs),
    )
    byte_w, byte_m = freeze_goldilocks_logup_witness_v3(
        statement=statement.byte_logup_statement(),
        looked_up_values=tuple(byte_values),
    )
    witness = GoldilocksGdnTileWitnessV3(
        statement=statement,
        columns_tree=columns_tree,
        decay_witness_tree=decay_w,
        decay_multiplicity_tree=decay_m,
        byte_witness_tree=byte_w,
        byte_multiplicity_tree=byte_m,
    )
    return witness, tuple(honest_outputs)


def prove_goldilocks_gdn_tile_v3(
    *, witness: GoldilocksGdnTileWitnessV3
) -> GoldilocksGdnTileProofV3:
    return GoldilocksGdnTileProofV3(
        columns_opening=tuple(tuple(row) for row in witness.columns_tree.rows),
        decay_proof=GoldilocksLogupProofV3(
            witness_opening=tuple(
                row[0] for row in witness.decay_witness_tree.rows
            ),
            multiplicity_opening=tuple(
                row[0] for row in witness.decay_multiplicity_tree.rows
            ),
        ),
        decay_roots=(
            witness.decay_witness_tree.commitment,
            witness.decay_multiplicity_tree.commitment,
        ),
        byte_proof=GoldilocksLogupProofV3(
            witness_opening=tuple(row[0] for row in witness.byte_witness_tree.rows),
            multiplicity_opening=tuple(
                row[0] for row in witness.byte_multiplicity_tree.rows
            ),
        ),
        byte_roots=(
            witness.byte_witness_tree.commitment,
            witness.byte_multiplicity_tree.commitment,
        ),
    )


def verify_goldilocks_gdn_tile_v3(
    proof: object,
    *,
    statement: GoldilocksGdnTileStatementV3,
    columns_root: bytes,
    validator_nonce: bytes,
) -> tuple[int, ...]:
    """Verify the exact recurrence; return the proven output column."""

    try:
        if not isinstance(proof, GoldilocksGdnTileProofV3):
            raise ProofV3VerificationError("gdn proof type is unexpected")
        steps = statement.step_count
        rows = proof.columns_opening
        rebuilt = GoldilocksMerkleTreeReference.from_rows(
            tuple(tuple(row) for row in rows),
            binding_digest=statement.columns_binding_digest(),
        )
        if rebuilt.commitment != columns_root:
            raise ProofV3VerificationError(
                "gdn columns opening does not match the frozen root"
            )
        if any(any(v != 0 for v in row) for row in rows[steps:]):
            raise ProofV3VerificationError("gdn padding rows must be zero")
        expected_pairs: list[int] = []
        expected_bytes: list[int] = []
        prev_state = 0
        outputs: list[int] = []
        for t in range(steps):
            (
                q_cell, k_cell, v_cell, gate_cell, decay,
                state_cell, carried_cell, remainder_a,
                new_state_cell, out_cell, remainder_b,
            ) = (_integer(value, "gdn cell") for value in rows[t])
            q_val = _from_field_signed(q_cell)
            k_val = _from_field_signed(k_cell)
            v_val = _from_field_signed(v_cell)
            gate_val = _from_field_signed(gate_cell)
            state_val = _from_field_signed(state_cell)
            carried = _from_field_signed(carried_cell)
            new_state = _from_field_signed(new_state_cell)
            out_val = _from_field_signed(out_cell)
            for name, value in (("q", q_val), ("k", k_val), ("v", v_val),
                                ("gate", gate_val)):
                _int8(value, f"gdn {name}")
            if state_val != prev_state:
                raise ProofV3VerificationError("gdn state chain is broken")
            # decay via table membership (checked below); the pair binds
            # gate -> decay so an inconsistent decay cell cannot pass.
            expected_pairs.append((gate_val + 128) + _GPACK * decay)
            # Euclidean rescale of the carried term, exactly.
            if (carried + (1 << 24)) * GDN_SCALE_V3 + remainder_a != (
                decay * state_val + (1 << 40)
            ):
                raise ProofV3VerificationError("gdn carry identity fails")
            if not 0 <= remainder_a < GDN_SCALE_V3:
                raise ProofV3VerificationError("gdn carry remainder is out of range")
            if new_state != carried + k_val * v_val:
                raise ProofV3VerificationError("gdn state update identity fails")
            if (out_val + (1 << 24)) * GDN_SCALE_V3 + remainder_b != (
                q_val * new_state + (1 << 40)
            ):
                raise ProofV3VerificationError("gdn output identity fails")
            if not 0 <= remainder_b < GDN_SCALE_V3:
                raise ProofV3VerificationError("gdn output remainder is out of range")
            expected_bytes.extend((remainder_a >> (8 * i)) & 0xFF for i in range(2))
            expected_bytes.extend((remainder_b >> (8 * i)) & 0xFF for i in range(2))
            prev_state = new_state
            outputs.append(out_val)
        decay_statement = statement.decay_logup_statement()
        opening = proof.decay_proof.witness_opening
        if tuple(opening[: len(expected_pairs)]) != tuple(expected_pairs) or any(
            value != decay_statement.table[0]
            for value in opening[len(expected_pairs) :]
        ):
            raise ProofV3VerificationError(
                "gdn decay witness does not match the columns"
            )
        verify_goldilocks_logup_reference_v3(
            proof.decay_proof,
            statement=decay_statement,
            witness_root=proof.decay_roots[0],
            multiplicity_root=proof.decay_roots[1],
            validator_nonce=validator_nonce,
        )
        byte_statement = statement.byte_logup_statement()
        opening = proof.byte_proof.witness_opening
        if tuple(opening[: len(expected_bytes)]) != tuple(expected_bytes) or any(
            value != byte_statement.table[0]
            for value in opening[len(expected_bytes) :]
        ):
            raise ProofV3VerificationError(
                "gdn byte witness does not match the columns"
            )
        verify_goldilocks_logup_reference_v3(
            proof.byte_proof,
            statement=byte_statement,
            witness_root=proof.byte_roots[0],
            multiplicity_root=proof.byte_roots[1],
            validator_nonce=validator_nonce,
        )
        return tuple(outputs)
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("gdn proof is malformed") from exc


__all__ = [
    "GDN_SCALE_V3",
    "GOLDILOCKS_GDN_TILE_ABI_V3",
    "GoldilocksGdnTileProofV3",
    "GoldilocksGdnTileStatementV3",
    "GoldilocksGdnTileWitnessV3",
    "prove_goldilocks_gdn_tile_v3",
    "run_and_freeze_goldilocks_gdn_tile_v3",
    "verify_goldilocks_gdn_tile_v3",
]
