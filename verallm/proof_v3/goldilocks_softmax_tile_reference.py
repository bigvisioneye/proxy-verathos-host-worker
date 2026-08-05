"""Exact fixed-point softmax tile reference over Goldilocks for proof-v3.

First nonlinear transition vertical: one attention-score row (a "tile") is
proven to be softmax-normalised **exactly**, with no float, tolerance, or
implicit rounding.  All nonlinearity is table semantics, which is why the
LogUp argument is the substrate:

* ``exp`` step: each ``(score, exp)`` pair is packed into one field element
  ``score + 2^32 * exp`` and proven a member of the validator-owned packed
  exp table via the dual-challenge LogUp reference.  The table IS the
  signed nonlinear semantics — there is no "close enough" exp.
* sum step: ``S = sum_i exp_i`` recomputed by the verifier from the frozen
  column (production: folded scan / sumcheck).
* normalisation: ``out_i = floor(exp_i * SCALE / S)`` proven exactly via
  the Euclidean identity ``out_i * S + rem_i == exp_i * SCALE`` with
  ``0 <= rem_i < S``.  The inequality is proven by byte-decomposing both
  ``rem_i`` and ``S - 1 - rem_i`` (limbs proven in the byte table via
  LogUp); the recombination is checked algebraically.

Chronology: all witness columns (scores, exps, outputs, remainders, limbs)
are frozen in one Merkle tree pre-nonce; LogUp challenges derive post-
freeze.  Reference verification opens the tree in full; the production
backend keeps the identical column/lookup ABI inside the AIR with
committed inverse columns.

Scale contract: scores in ``[0, MAX_SCORE]``, exps in ``[1, 2^32)``,
``SCALE = 2^16``.  Sums are bounded by ``tile * 2^32 << p``, so every
identity here is an exact integer identity in the field.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_linear_relation_reference import (
    _fixed32,
    _integer,
    _u32,
)
from verallm.proof_v3.goldilocks_logup_reference import (
    GoldilocksLogupStatementV3,
    freeze_goldilocks_logup_witness_v3,
    prove_goldilocks_logup_reference_v3,
    verify_goldilocks_logup_reference_v3,
)
from verallm.proof_v3.goldilocks_merkle_reference import (
    GoldilocksMerkleTreeReference,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS


GOLDILOCKS_SOFTMAX_TILE_ABI_V3: Final = "goldilocks.softmax_tile.reference.v1"
GOLDILOCKS_SOFTMAX_SCALE_V3: Final = 1 << 16
MAX_GOLDILOCKS_SOFTMAX_TILE_V3: Final = 1 << 12
MAX_GOLDILOCKS_SOFTMAX_SCORE_V3: Final = (1 << 16) - 1
_PACK_SHIFT: Final = 1 << 32
_LIMB_COUNT: Final = 6  # 48-bit range windows in 8-bit limbs

_STATEMENT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_SOFTMAX/V1/STATEMENT/SHA256"
_COLUMNS_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_SOFTMAX/V1/COLUMNS/SHA256"
_PAIR_BINDING_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_SOFTMAX/V1/PAIRS/SHA256"
_BYTE_BINDING_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_SOFTMAX/V1/BYTES/SHA256"


@dataclass(frozen=True, slots=True)
class GoldilocksSoftmaxTileStatementV3:
    """Validator-owned tile statement: length and the signed exp table."""

    validator_binding_digest: bytes
    tile_length: int
    exp_table: tuple[int, ...]

    def __post_init__(self) -> None:
        binding = _fixed32(
            self.validator_binding_digest,
            "softmax validator_binding_digest",
            nonzero=True,
        )
        tile = _u32(self.tile_length, "tile_length", positive=True)
        if tile > MAX_GOLDILOCKS_SOFTMAX_TILE_V3:
            raise ProofV3Error("softmax tile exceeds the CPU reference cap")
        if not isinstance(self.exp_table, tuple) or len(self.exp_table) != (
            MAX_GOLDILOCKS_SOFTMAX_SCORE_V3 + 1
        ):
            raise ProofV3Error("softmax exp table must cover every score")
        table = []
        for score, value in enumerate(self.exp_table):
            integer = _integer(value, f"exp_table[{score}]")
            if not 1 <= integer < _PACK_SHIFT:
                raise ProofV3Error("softmax exp table value is out of range")
            table.append(integer)
        object.__setattr__(self, "validator_binding_digest", binding)
        object.__setattr__(self, "tile_length", tile)
        object.__setattr__(self, "exp_table", tuple(table))

    def digest(self) -> bytes:
        return hashlib.sha256(
            _STATEMENT_DOMAIN
            + self.validator_binding_digest
            + struct.pack("<I", self.tile_length)
            + hashlib.sha256(
                b"".join(value.to_bytes(8, "little") for value in self.exp_table)
            ).digest()
        ).digest()

    def columns_binding_digest(self) -> bytes:
        return hashlib.sha256(_COLUMNS_DOMAIN + self.digest()).digest()

    def pair_logup_statement(self) -> GoldilocksLogupStatementV3:
        return GoldilocksLogupStatementV3(
            validator_binding_digest=hashlib.sha256(
                _PAIR_BINDING_DOMAIN + self.digest()
            ).digest(),
            table=tuple(
                score + _PACK_SHIFT * value
                for score, value in enumerate(self.exp_table)
            ),
        )

    def byte_logup_statement(self) -> GoldilocksLogupStatementV3:
        return GoldilocksLogupStatementV3(
            validator_binding_digest=hashlib.sha256(
                _BYTE_BINDING_DOMAIN + self.digest()
            ).digest(),
            table=tuple(range(256)),
        )


def _limbs(value: int) -> tuple[int, ...]:
    if not 0 <= value < 1 << (8 * _LIMB_COUNT):
        raise ProofV3Error("softmax range witness exceeds the limb window")
    return tuple((value >> (8 * index)) & 0xFF for index in range(_LIMB_COUNT))


@dataclass(frozen=True, slots=True)
class GoldilocksSoftmaxTileWitnessV3:
    """Frozen columns plus the two LogUp sub-witnesses."""

    columns_tree: GoldilocksMerkleTreeReference
    pair_witness_tree: GoldilocksMerkleTreeReference
    pair_multiplicity_tree: GoldilocksMerkleTreeReference
    byte_witness_tree: GoldilocksMerkleTreeReference
    byte_multiplicity_tree: GoldilocksMerkleTreeReference


def freeze_goldilocks_softmax_tile_v3(
    *,
    statement: GoldilocksSoftmaxTileStatementV3,
    scores: tuple[int, ...],
    exps: tuple[int, ...],
    outputs: tuple[int, ...],
) -> GoldilocksSoftmaxTileWitnessV3:
    """Freeze the tile columns pre-nonce.

    Shapes and packing windows are validated; the softmax *semantics*
    (table membership, Euclidean identity, ranges) are deliberately not
    checked — a wrong tile must fail verification, not construction.
    """

    tile = statement.tile_length
    for name, column in (("scores", scores), ("exps", exps), ("outputs", outputs)):
        if not isinstance(column, tuple) or len(column) != tile:
            raise ProofV3Error(f"softmax {name} column has an unexpected shape")
    scores = tuple(_u32(v, "score") for v in scores)
    if any(v > MAX_GOLDILOCKS_SOFTMAX_SCORE_V3 for v in scores):
        raise ProofV3Error("softmax score exceeds the signed window")
    exps = tuple(_integer(v, "exp") for v in exps)
    if any(not 0 <= v < _PACK_SHIFT for v in exps):
        raise ProofV3Error("softmax exp exceeds the packing window")
    outputs = tuple(_integer(v, "output") for v in outputs)
    total = sum(exps)
    rows: list[tuple[int, ...]] = []
    pair_values: list[int] = []
    byte_values: list[int] = []
    for score, exp, out in zip(scores, exps, outputs, strict=True):
        product = exp * GOLDILOCKS_SOFTMAX_SCALE_V3
        remainder = product - out * total
        if not 0 <= remainder < 1 << (8 * _LIMB_COUNT):
            remainder_window = 0  # frozen as-is; verification rejects
            remainder = remainder % (1 << (8 * _LIMB_COUNT))
        complement = (total - 1 - remainder) % (1 << (8 * _LIMB_COUNT))
        remainder_limbs = _limbs(remainder)
        complement_limbs = _limbs(complement)
        rows.append(
            (score, exp, out, remainder, complement)
            + remainder_limbs
            + complement_limbs
        )
        pair_values.append(score + _PACK_SHIFT * exp)
        byte_values.extend(remainder_limbs)
        byte_values.extend(complement_limbs)
    padded = 1 << max(1, (len(rows) - 1).bit_length())
    width = len(rows[0])
    rows.extend((0,) * width for _ in range(padded - len(rows)))
    columns_tree = GoldilocksMerkleTreeReference.from_rows(
        tuple(rows),
        binding_digest=statement.columns_binding_digest(),
    )
    pair_witness_tree, pair_multiplicity_tree = freeze_goldilocks_logup_witness_v3(
        statement=statement.pair_logup_statement(),
        looked_up_values=tuple(pair_values),
    )
    byte_witness_tree, byte_multiplicity_tree = freeze_goldilocks_logup_witness_v3(
        statement=statement.byte_logup_statement(),
        looked_up_values=tuple(byte_values),
    )
    return GoldilocksSoftmaxTileWitnessV3(
        columns_tree=columns_tree,
        pair_witness_tree=pair_witness_tree,
        pair_multiplicity_tree=pair_multiplicity_tree,
        byte_witness_tree=byte_witness_tree,
        byte_multiplicity_tree=byte_multiplicity_tree,
    )


@dataclass(frozen=True, slots=True)
class GoldilocksSoftmaxTileProofV3:
    columns_opening: tuple[tuple[int, ...], ...]
    pair_witness_opening: tuple[int, ...]
    pair_multiplicity_opening: tuple[int, ...]
    byte_witness_opening: tuple[int, ...]
    byte_multiplicity_opening: tuple[int, ...]


def prove_goldilocks_softmax_tile_v3(
    *,
    witness: GoldilocksSoftmaxTileWitnessV3,
) -> GoldilocksSoftmaxTileProofV3:
    return GoldilocksSoftmaxTileProofV3(
        columns_opening=tuple(tuple(row) for row in witness.columns_tree.rows),
        pair_witness_opening=tuple(
            row[0] for row in witness.pair_witness_tree.rows
        ),
        pair_multiplicity_opening=tuple(
            row[0] for row in witness.pair_multiplicity_tree.rows
        ),
        byte_witness_opening=tuple(
            row[0] for row in witness.byte_witness_tree.rows
        ),
        byte_multiplicity_opening=tuple(
            row[0] for row in witness.byte_multiplicity_tree.rows
        ),
    )


def verify_goldilocks_softmax_tile_v3(
    proof: object,
    *,
    statement: GoldilocksSoftmaxTileStatementV3,
    columns_root: bytes,
    pair_witness_root: bytes,
    pair_multiplicity_root: bytes,
    byte_witness_root: bytes,
    byte_multiplicity_root: bytes,
    validator_nonce: bytes,
) -> None:
    """Verify one exact softmax tile against the frozen roots."""

    try:
        if not isinstance(proof, GoldilocksSoftmaxTileProofV3):
            raise ProofV3VerificationError("softmax proof type is unexpected")
        tile = statement.tile_length
        rows = proof.columns_opening
        rebuilt = GoldilocksMerkleTreeReference.from_rows(
            tuple(tuple(row) for row in rows),
            binding_digest=statement.columns_binding_digest(),
        )
        if rebuilt.commitment != columns_root:
            raise ProofV3VerificationError(
                "softmax columns opening does not match the frozen root"
            )
        active = rows[:tile]
        if any(any(v != 0 for v in row) for row in rows[tile:]):
            raise ProofV3VerificationError("softmax padding rows must be zero")
        width = 5 + 2 * _LIMB_COUNT
        if any(len(row) != width for row in rows):
            raise ProofV3VerificationError("softmax column width is unexpected")
        total = sum(_integer(row[1], "exp") for row in active)
        if total <= 0 or total >= 1 << (8 * _LIMB_COUNT):
            raise ProofV3VerificationError("softmax exp sum is out of range")
        expected_pairs = []
        expected_bytes = []
        for row in active:
            score, exp, out, remainder, complement = (
                _integer(v, "softmax cell") for v in row[:5]
            )
            remainder_limbs = tuple(
                _integer(v, "limb") for v in row[5 : 5 + _LIMB_COUNT]
            )
            complement_limbs = tuple(
                _integer(v, "limb") for v in row[5 + _LIMB_COUNT :]
            )
            # Euclidean identity: out * S + rem == exp * SCALE, exactly.
            if out * total + remainder != exp * GOLDILOCKS_SOFTMAX_SCALE_V3:
                raise ProofV3VerificationError(
                    "softmax normalisation identity fails"
                )
            # rem < S proven via rem + (S - 1 - rem) == S - 1 with both
            # sides byte-decomposed (limbs membership-checked below).
            if remainder + complement != total - 1:
                raise ProofV3VerificationError(
                    "softmax remainder range identity fails"
                )
            if sum(
                limb << (8 * index) for index, limb in enumerate(remainder_limbs)
            ) != remainder or sum(
                limb << (8 * index) for index, limb in enumerate(complement_limbs)
            ) != complement:
                raise ProofV3VerificationError(
                    "softmax limb recombination fails"
                )
            expected_pairs.append(score + _PACK_SHIFT * exp)
            expected_bytes.extend(remainder_limbs)
            expected_bytes.extend(complement_limbs)
        # The LogUp witnesses must be exactly the columns' derived values
        # (up to the documented table[0] padding), then membership holds.
        pair_statement = statement.pair_logup_statement()
        byte_statement = statement.byte_logup_statement()
        for opening, expected, logup_statement in (
            (proof.pair_witness_opening, expected_pairs, pair_statement),
            (proof.byte_witness_opening, expected_bytes, byte_statement),
        ):
            if tuple(opening[: len(expected)]) != tuple(expected):
                raise ProofV3VerificationError(
                    "softmax lookup witness does not match the frozen columns"
                )
            if any(
                value != logup_statement.table[0]
                for value in opening[len(expected) :]
            ):
                raise ProofV3VerificationError(
                    "softmax lookup witness padding is malformed"
                )
        from verallm.proof_v3.goldilocks_logup_reference import (
            GoldilocksLogupProofV3,
        )

        verify_goldilocks_logup_reference_v3(
            GoldilocksLogupProofV3(
                witness_opening=proof.pair_witness_opening,
                multiplicity_opening=proof.pair_multiplicity_opening,
            ),
            statement=pair_statement,
            witness_root=pair_witness_root,
            multiplicity_root=pair_multiplicity_root,
            validator_nonce=validator_nonce,
        )
        verify_goldilocks_logup_reference_v3(
            GoldilocksLogupProofV3(
                witness_opening=proof.byte_witness_opening,
                multiplicity_opening=proof.byte_multiplicity_opening,
            ),
            statement=byte_statement,
            witness_root=byte_witness_root,
            multiplicity_root=byte_multiplicity_root,
            validator_nonce=validator_nonce,
        )
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("softmax proof is malformed") from exc


__all__ = [
    "GOLDILOCKS_SOFTMAX_SCALE_V3",
    "GOLDILOCKS_SOFTMAX_TILE_ABI_V3",
    "GoldilocksSoftmaxTileProofV3",
    "GoldilocksSoftmaxTileStatementV3",
    "GoldilocksSoftmaxTileWitnessV3",
    "freeze_goldilocks_softmax_tile_v3",
    "prove_goldilocks_softmax_tile_v3",
    "verify_goldilocks_softmax_tile_v3",
]
