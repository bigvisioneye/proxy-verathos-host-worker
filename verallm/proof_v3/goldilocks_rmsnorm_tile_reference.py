"""Exact fixed-point RMSNorm tile reference over Goldilocks for proof-v3.

The remaining transition primitive for a full transformer block:
``y[d] = x[d] * weight[d] * rsqrt(mean_sq + eps)`` proven exactly in fixed
point, no float, using the shipped LogUp + Euclidean-range machinery.

Per row (one token, model dim D), all int8 inputs, scale 2^16:

* ``sum_sq = sum_d x[d]^2`` recomputed by the verifier from the opened x
  column (a folded scan in the native path; direct here);
* ``mean_sq = sum_sq // D`` with remainder in ``[0, D)`` proven by byte
  limbs (the divisor D is public);
* ``inv = RsqrtTable[mean_sq_index]`` proven a member of the signed packed
  rsqrt table via dual-challenge LogUp (index clamped into the table
  domain; the table IS the signed rsqrt semantics, eps folded in);
* ``y[d] = (x[d] * weight[d] * inv + HALF) // 2^16`` (two chained
  Euclidean rescales) with both remainders byte-range-proven.

Chronology matches every tile: columns freeze pre-nonce, LogUp challenges
derive post-freeze; reference opens in full, native keeps the same ABI.
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


GOLDILOCKS_RMSNORM_TILE_ABI_V3: Final = "goldilocks.rmsnorm_tile.reference.v1"
RMSNORM_SCALE_V3: Final = 1 << 16
MAX_GOLDILOCKS_RMSNORM_DIM_V3: Final = 4096
_RPACK: Final = 1 << 32
_LIMB_COUNT: Final = 4  # 32-bit range windows in 8-bit limbs

_STATEMENT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_RMSNORM/V1/STATEMENT/SHA256"
_COLUMNS_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_RMSNORM/V1/COLUMNS/SHA256"
_RSQRT_BINDING_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_RMSNORM/V1/RSQRT/SHA256"
_BYTE_BINDING_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_RMSNORM/V1/BYTES/SHA256"


def _signed_field(value: int) -> int:
    return value % GOLDILOCKS_MODULUS


def _from_field_signed(value: int) -> int:
    return value - GOLDILOCKS_MODULUS if value >= GOLDILOCKS_MODULUS // 2 else value


@dataclass(frozen=True, slots=True)
class GoldilocksRmsnormTileStatementV3:
    """Validator-owned RMSNorm statement: dim, weight, rsqrt table."""

    validator_binding_digest: bytes
    model_dim: int
    weight: tuple[int, ...]
    rsqrt_table: tuple[int, ...]

    def __post_init__(self) -> None:
        binding = _fixed32(
            self.validator_binding_digest, "rmsnorm binding", nonzero=True
        )
        dim = _u32(self.model_dim, "model_dim", positive=True)
        if dim > MAX_GOLDILOCKS_RMSNORM_DIM_V3:
            raise ProofV3Error("rmsnorm dim exceeds the CPU reference cap")
        if not isinstance(self.weight, tuple) or len(self.weight) != dim:
            raise ProofV3Error("rmsnorm weight shape does not match the dim")
        weight = tuple(_int8(w, f"rmsnorm weight[{i}]") for i, w in enumerate(self.weight))
        if not isinstance(self.rsqrt_table, tuple) or not self.rsqrt_table:
            raise ProofV3Error("rmsnorm rsqrt table is malformed")
        table = tuple(
            _integer(v, f"rsqrt_table[{i}]") for i, v in enumerate(self.rsqrt_table)
        )
        if any(not 0 <= v < _RPACK for v in table):
            raise ProofV3Error("rmsnorm rsqrt value is out of range")
        object.__setattr__(self, "validator_binding_digest", binding)
        object.__setattr__(self, "model_dim", dim)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "rsqrt_table", table)

    def digest(self) -> bytes:
        return hashlib.sha256(
            _STATEMENT_DOMAIN
            + self.validator_binding_digest
            + struct.pack("<I", self.model_dim)
            + b"".join(struct.pack("<b", w) for w in self.weight)
            + hashlib.sha256(
                b"".join(v.to_bytes(8, "little") for v in self.rsqrt_table)
            ).digest()
        ).digest()

    def columns_binding_digest(self) -> bytes:
        return hashlib.sha256(_COLUMNS_DOMAIN + self.digest()).digest()

    def rsqrt_logup_statement(self) -> GoldilocksLogupStatementV3:
        return GoldilocksLogupStatementV3(
            validator_binding_digest=hashlib.sha256(
                _RSQRT_BINDING_DOMAIN + self.digest()
            ).digest(),
            table=tuple(
                index + _RPACK * value
                for index, value in enumerate(self.rsqrt_table)
            ),
        )

    def byte_logup_statement(self) -> GoldilocksLogupStatementV3:
        return GoldilocksLogupStatementV3(
            validator_binding_digest=hashlib.sha256(
                _BYTE_BINDING_DOMAIN + self.digest()
            ).digest(),
            table=tuple(range(256)),
        )

    def rsqrt_index(self, mean_sq: int) -> int:
        return min(max(mean_sq, 0), len(self.rsqrt_table) - 1)


@dataclass(frozen=True, slots=True)
class GoldilocksRmsnormTileProofV3:
    columns_opening: tuple[tuple[int, ...], ...]
    mean_row: tuple[int, ...]
    rsqrt_proof: GoldilocksLogupProofV3
    rsqrt_roots: tuple[bytes, bytes]
    byte_proof: GoldilocksLogupProofV3
    byte_roots: tuple[bytes, bytes]


@dataclass(frozen=True, slots=True)
class GoldilocksRmsnormTileWitnessV3:
    statement: GoldilocksRmsnormTileStatementV3
    columns_tree: GoldilocksMerkleTreeReference
    mean_row: tuple[int, ...]
    rsqrt_witness_tree: GoldilocksMerkleTreeReference
    rsqrt_multiplicity_tree: GoldilocksMerkleTreeReference
    byte_witness_tree: GoldilocksMerkleTreeReference
    byte_multiplicity_tree: GoldilocksMerkleTreeReference


def _limbs(value: int) -> tuple[int, ...]:
    if not 0 <= value < 1 << (8 * _LIMB_COUNT):
        raise ProofV3Error("rmsnorm range witness exceeds the limb window")
    return tuple((value >> (8 * i)) & 0xFF for i in range(_LIMB_COUNT))


def run_and_freeze_goldilocks_rmsnorm_tile_v3(
    *,
    statement: GoldilocksRmsnormTileStatementV3,
    x_row: tuple[int, ...],
    outputs: tuple[int, ...] | None = None,
) -> tuple[GoldilocksRmsnormTileWitnessV3, tuple[int, ...]]:
    """Execute exact RMSNorm on one row, freeze columns, return outputs."""

    dim = statement.model_dim
    if not isinstance(x_row, tuple) or len(x_row) != dim:
        raise ProofV3Error("rmsnorm x row has an unexpected shape")
    x = tuple(_int8(v, "rmsnorm x") for v in x_row)
    sum_sq = sum(v * v for v in x)
    mean_sq, mean_rem = divmod(sum_sq, dim)
    index = statement.rsqrt_index(mean_sq)
    inv = statement.rsqrt_table[index]
    half = RMSNORM_SCALE_V3 // 2
    rows: list[tuple[int, ...]] = []
    rsqrt_pairs = [index + _RPACK * inv]
    byte_values: list[int] = list(_limbs(mean_rem))
    honest: list[int] = []
    for d in range(dim):
        prod = x[d] * statement.weight[d] * inv + half
        y_shift, y_rem = divmod(prod + (1 << 40), RMSNORM_SCALE_V3)
        y_value = y_shift - (1 << 24)
        honest.append(y_value)
        out_value = y_value if outputs is None else outputs[d]
        byte_values.extend(_limbs(y_rem))
        rows.append(
            (
                _signed_field(x[d]),
                _signed_field(statement.weight[d]),
                _signed_field(out_value),
                y_rem,
            )
        )
    width = len(rows[0])
    padded = 1 << max(1, (len(rows) - 1).bit_length())
    while len(rows) < padded:
        rows.append((0,) * width)
    columns_tree = GoldilocksMerkleTreeReference.from_rows(
        tuple(rows), binding_digest=statement.columns_binding_digest()
    )
    mean_row = (
        _signed_field(sum_sq % GOLDILOCKS_MODULUS),
        _signed_field(mean_sq),
        mean_rem,
        index,
        _signed_field(inv),
    )
    rsqrt_w, rsqrt_m = freeze_goldilocks_logup_witness_v3(
        statement=statement.rsqrt_logup_statement(),
        looked_up_values=tuple(rsqrt_pairs),
    )
    byte_w, byte_m = freeze_goldilocks_logup_witness_v3(
        statement=statement.byte_logup_statement(),
        looked_up_values=tuple(byte_values),
    )
    witness = GoldilocksRmsnormTileWitnessV3(
        statement=statement,
        columns_tree=columns_tree,
        mean_row=mean_row,
        rsqrt_witness_tree=rsqrt_w,
        rsqrt_multiplicity_tree=rsqrt_m,
        byte_witness_tree=byte_w,
        byte_multiplicity_tree=byte_m,
    )
    return witness, tuple(honest)


def prove_goldilocks_rmsnorm_tile_v3(
    *, witness: GoldilocksRmsnormTileWitnessV3
) -> GoldilocksRmsnormTileProofV3:
    return GoldilocksRmsnormTileProofV3(
        columns_opening=tuple(tuple(row) for row in witness.columns_tree.rows),
        mean_row=witness.mean_row,
        rsqrt_proof=GoldilocksLogupProofV3(
            witness_opening=tuple(r[0] for r in witness.rsqrt_witness_tree.rows),
            multiplicity_opening=tuple(
                r[0] for r in witness.rsqrt_multiplicity_tree.rows
            ),
        ),
        rsqrt_roots=(
            witness.rsqrt_witness_tree.commitment,
            witness.rsqrt_multiplicity_tree.commitment,
        ),
        byte_proof=GoldilocksLogupProofV3(
            witness_opening=tuple(r[0] for r in witness.byte_witness_tree.rows),
            multiplicity_opening=tuple(
                r[0] for r in witness.byte_multiplicity_tree.rows
            ),
        ),
        byte_roots=(
            witness.byte_witness_tree.commitment,
            witness.byte_multiplicity_tree.commitment,
        ),
    )


def verify_goldilocks_rmsnorm_tile_v3(
    proof: object,
    *,
    statement: GoldilocksRmsnormTileStatementV3,
    columns_root: bytes,
    validator_nonce: bytes,
) -> tuple[int, ...]:
    """Verify exact RMSNorm of one row; return the proven output column."""

    try:
        if not isinstance(proof, GoldilocksRmsnormTileProofV3):
            raise ProofV3VerificationError("rmsnorm proof type is unexpected")
        dim = statement.model_dim
        rows = proof.columns_opening
        rebuilt = GoldilocksMerkleTreeReference.from_rows(
            tuple(tuple(row) for row in rows),
            binding_digest=statement.columns_binding_digest(),
        )
        if rebuilt.commitment != columns_root:
            raise ProofV3VerificationError(
                "rmsnorm columns opening does not match the frozen root"
            )
        active = rows[:dim]
        if any(any(v != 0 for v in row) for row in rows[dim:]):
            raise ProofV3VerificationError("rmsnorm padding rows must be zero")
        # Bind the x/weight columns and recompute sum of squares.
        sum_sq = 0
        for d in range(dim):
            x_cell, w_cell, _out, _rem = (
                _integer(v, "rmsnorm cell") for v in active[d]
            )
            x_val = _from_field_signed(x_cell)
            if _from_field_signed(w_cell) != statement.weight[d]:
                raise ProofV3VerificationError(
                    "rmsnorm weight column does not match the signed weight"
                )
            sum_sq += x_val * x_val
        (
            sum_sq_cell, mean_sq_cell, mean_rem, index, inv_cell
        ) = (_integer(v, "rmsnorm mean cell") for v in proof.mean_row)
        if _from_field_signed(sum_sq_cell) % GOLDILOCKS_MODULUS != (
            sum_sq % GOLDILOCKS_MODULUS
        ):
            raise ProofV3VerificationError("rmsnorm sum-of-squares mismatch")
        mean_sq = _from_field_signed(mean_sq_cell)
        if mean_sq * dim + mean_rem != sum_sq or not 0 <= mean_rem < dim:
            raise ProofV3VerificationError("rmsnorm mean identity fails")
        if index != statement.rsqrt_index(mean_sq):
            raise ProofV3VerificationError("rmsnorm rsqrt index is not canonical")
        inv = _from_field_signed(inv_cell)
        # rsqrt membership binds index -> inv.
        expected_rsqrt_pairs = [index + _RPACK * inv]
        rsqrt_statement = statement.rsqrt_logup_statement()
        opening = proof.rsqrt_proof.witness_opening
        if tuple(opening[:1]) != tuple(expected_rsqrt_pairs) or any(
            v != rsqrt_statement.table[0] for v in opening[1:]
        ):
            raise ProofV3VerificationError(
                "rmsnorm rsqrt witness does not match the mean row"
            )
        verify_goldilocks_logup_reference_v3(
            proof.rsqrt_proof,
            statement=rsqrt_statement,
            witness_root=proof.rsqrt_roots[0],
            multiplicity_root=proof.rsqrt_roots[1],
            validator_nonce=validator_nonce,
        )
        # Output identity + byte-range witnesses.
        half = RMSNORM_SCALE_V3 // 2
        expected_bytes: list[int] = list(_limbs(mean_rem))
        outputs: list[int] = []
        for d in range(dim):
            x_val = _from_field_signed(active[d][0])
            out_val = _from_field_signed(active[d][2])
            y_rem = _integer(active[d][3], "rmsnorm y remainder")
            prod = x_val * statement.weight[d] * inv + half
            if (out_val + (1 << 24)) * RMSNORM_SCALE_V3 + y_rem != prod + (1 << 40):
                raise ProofV3VerificationError("rmsnorm output identity fails")
            if not 0 <= y_rem < RMSNORM_SCALE_V3:
                raise ProofV3VerificationError("rmsnorm y remainder out of range")
            expected_bytes.extend(_limbs(y_rem))
            outputs.append(out_val)
        byte_statement = statement.byte_logup_statement()
        opening = proof.byte_proof.witness_opening
        if tuple(opening[: len(expected_bytes)]) != tuple(expected_bytes) or any(
            v != byte_statement.table[0] for v in opening[len(expected_bytes) :]
        ):
            raise ProofV3VerificationError(
                "rmsnorm byte witness does not match the columns"
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
        raise ProofV3VerificationError("rmsnorm proof is malformed") from exc


__all__ = [
    "GOLDILOCKS_RMSNORM_TILE_ABI_V3",
    "RMSNORM_SCALE_V3",
    "GoldilocksRmsnormTileProofV3",
    "GoldilocksRmsnormTileStatementV3",
    "GoldilocksRmsnormTileWitnessV3",
    "prove_goldilocks_rmsnorm_tile_v3",
    "run_and_freeze_goldilocks_rmsnorm_tile_v3",
    "verify_goldilocks_rmsnorm_tile_v3",
]
