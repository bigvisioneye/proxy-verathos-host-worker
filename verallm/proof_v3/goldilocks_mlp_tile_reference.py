"""Exact MLP (gate/SiLU/up/down bridge) tile reference for proof-v3.

Closes the last internal-arithmetic vertical: the SwiGLU nonlinearity
``h = SiLU(gate) * up`` is proven exactly in fixed point, composing the
shipped primitives with no sampled replay:

1. **SiLU**: every ``(g8, s)`` pair is proven a member of the
   validator-owned quantized-SiLU table via packed-pair LogUp.  The table
   maps the full int8 gate domain to a fixed-point SiLU output, so the
   nonlinearity itself is table-exact.
2. **Elementwise product**: the wide product ``wide[t,j] = s[t,j] *
   u8[t,j]`` is tied to the committed ``s``/``u8`` tables by a folded
   two-table product sumcheck: the verifier folds with nonce-derived
   tensor coefficients ``v[t] w[j]`` and requires the degree-3 sumcheck
   over the committed tables to hit the scalar recomputed from the opened
   wide column.
3. **Requantization**: ``h_q = round(wide / D)`` for the public integer
   divisor ``D`` is proven by exact Euclidean division with the remainder
   in ``[0, D)`` shown by byte limbs (LogUp), and the final int8 output
   ``h8 = Clamp[h_q]`` is proven a member of the validator-owned clamp
   table via packed-pair LogUp.

The gate/up/down *projections* around this tile are folded-linear checks
(the succinct fold argument) and are deliberately not duplicated here —
this tile is exactly the internal nonlinearity that the projections
cannot express.

All tables freeze pre-nonce in one tile tree; every challenge (fold
coefficients, sumcheck rounds, LogUp challenges) derives post-freeze.
Reference verification opens tables in full; the crypto-critical steps
(product sumcheck, three LogUps) run the real machinery.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_fold_sumcheck_reference import _challenge
from verallm.proof_v3.goldilocks_linear_relation_reference import (
    _fixed32,
    _integer,
    _u32,
)
from verallm.proof_v3.goldilocks_logup_reference import (
    GoldilocksLogupProofV3,
    GoldilocksLogupStatementV3,
    freeze_goldilocks_logup_witness_v3,
    prove_goldilocks_logup_reference_v3,
    verify_goldilocks_logup_reference_v3,
)
from verallm.proof_v3.goldilocks_merkle_reference import (
    GoldilocksMerkleTreeReference,
)
from verallm.proof_v3.goldilocks_product_sumcheck_reference import (
    GoldilocksProductSumcheckProofV3,
    commit_goldilocks_product_sumcheck_a_v3,
    commit_goldilocks_product_sumcheck_b_v3,
    prove_goldilocks_product_sumcheck_v3,
    verify_goldilocks_product_sumcheck_v3,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS


GOLDILOCKS_MLP_TILE_ABI_V3: Final = "goldilocks.mlp_tile.reference.v1"
MAX_GOLDILOCKS_MLP_TOKENS_V3: Final = 32
MAX_GOLDILOCKS_MLP_FF_DIM_V3: Final = 8192
GOLDILOCKS_MLP_SHIFT_V3: Final = 1 << 20
_SPACK: Final = 1 << 32
_SILU_BIAS: Final = 1 << 16
_CLAMP_BIAS: Final = 1 << 8

_STATEMENT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MLP_TILE/V1/STATEMENT"
_TABLES_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MLP_TILE/V1/TABLES"
_SILU_BINDING_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MLP_TILE/V1/SILU"
_CLAMP_BINDING_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MLP_TILE/V1/CLAMP"
_BYTE_BINDING_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MLP_TILE/V1/BYTE"
_FOLD_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MLP_TILE/V1/FOLD"


def _signed_field(value: int) -> int:
    return value % GOLDILOCKS_MODULUS


def _from_field_signed(value: int) -> int:
    return value - GOLDILOCKS_MODULUS if value > GOLDILOCKS_MODULUS // 2 else value


def _int8(value: object, name: str) -> int:
    number = _integer(value, name)
    if not -128 <= number <= 127:
        raise ProofV3Error(f"{name} is not int8")
    return number


def _limbs(value: int, count: int) -> tuple[int, ...]:
    if not 0 <= value < 1 << (8 * count):
        raise ProofV3Error("mlp remainder exceeds the limb window")
    return tuple((value >> (8 * i)) & 0xFF for i in range(count))


@dataclass(frozen=True, slots=True)
class GoldilocksMlpTileStatementV3:
    """Validator-owned MLP-tile statement: shape, SiLU and clamp tables."""

    validator_binding_digest: bytes
    token_count: int
    ff_dim: int
    silu_table: tuple[int, ...]
    divisor: int
    clamp_offset: int
    clamp_table: tuple[int, ...]

    def __post_init__(self) -> None:
        binding = _fixed32(
            self.validator_binding_digest, "mlp validator_binding_digest",
            nonzero=True,
        )
        tokens = _u32(self.token_count, "token_count", positive=True)
        ff_dim = _u32(self.ff_dim, "ff_dim", positive=True)
        if tokens > MAX_GOLDILOCKS_MLP_TOKENS_V3:
            raise ProofV3Error("mlp token count exceeds the CPU reference cap")
        if ff_dim > MAX_GOLDILOCKS_MLP_FF_DIM_V3:
            raise ProofV3Error("mlp ff dim exceeds the CPU reference cap")
        if not isinstance(self.silu_table, tuple) or len(self.silu_table) != 256:
            raise ProofV3Error("mlp silu table must cover the int8 gate domain")
        silu = tuple(_integer(v, "silu value") for v in self.silu_table)
        if any(abs(v) >= _SILU_BIAS for v in silu):
            raise ProofV3Error("mlp silu value exceeds the fixed-point window")
        divisor = _integer(self.divisor, "divisor")
        if not 1 <= divisor < 1 << 32:
            raise ProofV3Error("mlp divisor is out of range")
        offset = _u32(self.clamp_offset, "clamp_offset")
        if not isinstance(self.clamp_table, tuple) or not self.clamp_table:
            raise ProofV3Error("mlp clamp table is malformed")
        clamp = tuple(_int8(v, "clamp value") for v in self.clamp_table)
        object.__setattr__(self, "validator_binding_digest", binding)
        object.__setattr__(self, "token_count", tokens)
        object.__setattr__(self, "ff_dim", ff_dim)
        object.__setattr__(self, "silu_table", silu)
        object.__setattr__(self, "divisor", divisor)
        object.__setattr__(self, "clamp_offset", offset)
        object.__setattr__(self, "clamp_table", clamp)

    def digest(self) -> bytes:
        return hashlib.sha256(
            _STATEMENT_DOMAIN
            + self.validator_binding_digest
            + struct.pack(
                "<IIQI",
                self.token_count,
                self.ff_dim,
                self.divisor,
                self.clamp_offset,
            )
            + hashlib.sha256(
                b"".join(
                    _signed_field(v).to_bytes(8, "little")
                    for v in self.silu_table
                )
            ).digest()
            + hashlib.sha256(
                b"".join(
                    _signed_field(v).to_bytes(8, "little")
                    for v in self.clamp_table
                )
            ).digest()
        ).digest()

    def tables_binding_digest(self) -> bytes:
        return hashlib.sha256(_TABLES_DOMAIN + self.digest()).digest()

    def silu_logup_statement(self) -> GoldilocksLogupStatementV3:
        return GoldilocksLogupStatementV3(
            validator_binding_digest=hashlib.sha256(
                _SILU_BINDING_DOMAIN + self.digest()
            ).digest(),
            table=tuple(
                index + _SPACK * (value + _SILU_BIAS)
                for index, value in enumerate(self.silu_table)
            ),
        )

    def clamp_logup_statement(self) -> GoldilocksLogupStatementV3:
        return GoldilocksLogupStatementV3(
            validator_binding_digest=hashlib.sha256(
                _CLAMP_BINDING_DOMAIN + self.digest()
            ).digest(),
            table=tuple(
                index + _SPACK * (value + _CLAMP_BIAS)
                for index, value in enumerate(self.clamp_table)
            ),
        )

    def byte_logup_statement(self) -> GoldilocksLogupStatementV3:
        return GoldilocksLogupStatementV3(
            validator_binding_digest=hashlib.sha256(
                _BYTE_BINDING_DOMAIN + self.digest()
            ).digest(),
            table=tuple(range(256)),
        )

    def clamp_index(self, h_q: int) -> int:
        index = h_q + self.clamp_offset
        return min(max(index, 0), len(self.clamp_table) - 1)

    def remainder_limbs(self) -> int:
        """Byte limbs per remainder: 2 keeps the LogUp witness under the
        CPU reference Merkle cap for the small divisors this tile uses."""
        return 2 if self.divisor <= 1 << 16 else 4

    def cell_count(self) -> int:
        length = self.token_count * self.ff_dim
        padded = 1 << max(1, (length - 1).bit_length())
        return padded


@dataclass(frozen=True, slots=True)
class GoldilocksMlpTileProofV3:
    cells_opening: tuple[tuple[int, ...], ...]
    product_sumcheck: GoldilocksProductSumcheckProofV3
    silu_logup: GoldilocksLogupProofV3
    silu_roots: tuple[bytes, bytes]
    clamp_logup: GoldilocksLogupProofV3
    clamp_roots: tuple[bytes, bytes]
    byte_logup: GoldilocksLogupProofV3
    byte_roots: tuple[bytes, bytes]


@dataclass(frozen=True, slots=True)
class GoldilocksMlpTileWitnessV3:
    statement: GoldilocksMlpTileStatementV3
    cells_tree: GoldilocksMerkleTreeReference
    silu_witness_tree: GoldilocksMerkleTreeReference
    silu_multiplicity_tree: GoldilocksMerkleTreeReference
    clamp_witness_tree: GoldilocksMerkleTreeReference
    clamp_multiplicity_tree: GoldilocksMerkleTreeReference
    byte_witness_tree: GoldilocksMerkleTreeReference
    byte_multiplicity_tree: GoldilocksMerkleTreeReference
    s_values: tuple[int, ...]
    u_values: tuple[int, ...]


def run_and_freeze_goldilocks_mlp_tile_v3(
    *,
    statement: GoldilocksMlpTileStatementV3,
    gate_rows: tuple[tuple[int, ...], ...],
    up_rows: tuple[tuple[int, ...], ...],
) -> tuple[GoldilocksMlpTileWitnessV3, tuple[tuple[int, ...], ...]]:
    """Execute the exact SwiGLU pipeline, freeze every column, return h8."""

    tokens, ff_dim = statement.token_count, statement.ff_dim
    for name, table in (("gate", gate_rows), ("up", up_rows)):
        if len(table) != tokens or any(len(row) != ff_dim for row in table):
            raise ProofV3Error(f"mlp {name} table has an unexpected shape")
    divisor = statement.divisor
    shift = GOLDILOCKS_MLP_SHIFT_V3
    rows: list[tuple[int, ...]] = []
    s_flat: list[int] = []
    u_flat: list[int] = []
    silu_pairs: list[int] = []
    clamp_pairs: list[int] = []
    byte_values: list[int] = []
    outputs: list[tuple[int, ...]] = []
    for t in range(tokens):
        out_row: list[int] = []
        for j in range(ff_dim):
            g8 = _int8(gate_rows[t][j], "mlp gate")
            u8 = _int8(up_rows[t][j], "mlp up")
            s = statement.silu_table[g8 + 128]
            wide = s * u8
            q_shift, rem = divmod(wide + divisor * shift + divisor // 2, divisor)
            h_q = q_shift - shift
            index = statement.clamp_index(h_q)
            h8 = statement.clamp_table[index]
            out_row.append(h8)
            s_flat.append(_signed_field(s))
            u_flat.append(_signed_field(u8))
            silu_pairs.append((g8 + 128) + _SPACK * (s + _SILU_BIAS))
            clamp_pairs.append(index + _SPACK * (h8 + _CLAMP_BIAS))
            byte_values.extend(_limbs(rem, statement.remainder_limbs()))
            rows.append(
                (
                    _signed_field(g8),
                    _signed_field(u8),
                    _signed_field(s),
                    _signed_field(h_q),
                    rem,
                    _signed_field(h8),
                )
            )
        outputs.append(tuple(out_row))
    width = len(rows[0])
    padded = statement.cell_count()
    while len(rows) < padded:
        rows.append((0,) * width)
        s_flat.append(0)
        u_flat.append(0)
    cells_tree = GoldilocksMerkleTreeReference.from_rows(
        tuple(rows), binding_digest=statement.tables_binding_digest()
    )
    silu_w, silu_m = freeze_goldilocks_logup_witness_v3(
        statement=statement.silu_logup_statement(),
        looked_up_values=tuple(silu_pairs),
    )
    clamp_w, clamp_m = freeze_goldilocks_logup_witness_v3(
        statement=statement.clamp_logup_statement(),
        looked_up_values=tuple(clamp_pairs),
    )
    byte_w, byte_m = freeze_goldilocks_logup_witness_v3(
        statement=statement.byte_logup_statement(),
        looked_up_values=tuple(byte_values),
    )
    witness = GoldilocksMlpTileWitnessV3(
        statement=statement,
        cells_tree=cells_tree,
        silu_witness_tree=silu_w,
        silu_multiplicity_tree=silu_m,
        clamp_witness_tree=clamp_w,
        clamp_multiplicity_tree=clamp_m,
        byte_witness_tree=byte_w,
        byte_multiplicity_tree=byte_m,
        s_values=tuple(s_flat),
        u_values=tuple(u_flat),
    )
    return witness, tuple(outputs)


def _fold_coefficients(
    statement: GoldilocksMlpTileStatementV3,
    *,
    cells_root: bytes,
    validator_nonce: bytes,
) -> tuple[int, ...]:
    """Tensor fold coefficients v[t] w[j], nonce-derived post-freeze."""

    seed = hashlib.sha256(
        _FOLD_DOMAIN + statement.digest() + cells_root + validator_nonce
    ).digest()
    tokens, ff_dim = statement.token_count, statement.ff_dim
    v = tuple(_challenge(seed, 1 + t) for t in range(tokens))
    w = tuple(_challenge(seed, 1 + tokens + j) for j in range(ff_dim))
    factor = [
        v[t] * w[j] % GOLDILOCKS_MODULUS
        for t in range(tokens)
        for j in range(ff_dim)
    ]
    factor.extend(0 for _ in range(statement.cell_count() - len(factor)))
    return tuple(factor)


def prove_goldilocks_mlp_tile_v3(
    *,
    witness: GoldilocksMlpTileWitnessV3,
    validator_nonce: bytes,
) -> GoldilocksMlpTileProofV3:
    statement = witness.statement
    factor = _fold_coefficients(
        statement,
        cells_root=witness.cells_tree.commitment,
        validator_nonce=validator_nonce,
    )
    statement_digest = hashlib.sha256(
        statement.digest() + witness.cells_tree.commitment
    ).digest()
    a_tree = commit_goldilocks_product_sumcheck_a_v3(
        statement_digest=statement_digest, evaluations=witness.s_values
    )
    b_tree = commit_goldilocks_product_sumcheck_b_v3(
        statement_digest=statement_digest, evaluations=witness.u_values
    )
    product = prove_goldilocks_product_sumcheck_v3(
        statement_digest=statement_digest,
        a_tree=a_tree,
        b_tree=b_tree,
        a_evaluations=witness.s_values,
        b_evaluations=witness.u_values,
        factor=factor,
        validator_nonce=validator_nonce,
    )
    silu = prove_goldilocks_logup_reference_v3(
        witness_tree=witness.silu_witness_tree,
        multiplicity_tree=witness.silu_multiplicity_tree,
    )
    clamp = prove_goldilocks_logup_reference_v3(
        witness_tree=witness.clamp_witness_tree,
        multiplicity_tree=witness.clamp_multiplicity_tree,
    )
    byte = prove_goldilocks_logup_reference_v3(
        witness_tree=witness.byte_witness_tree,
        multiplicity_tree=witness.byte_multiplicity_tree,
    )
    return GoldilocksMlpTileProofV3(
        cells_opening=tuple(row for row in witness.cells_tree.rows),
        product_sumcheck=product,
        silu_logup=silu,
        silu_roots=(
            witness.silu_witness_tree.commitment,
            witness.silu_multiplicity_tree.commitment,
        ),
        clamp_logup=clamp,
        clamp_roots=(
            witness.clamp_witness_tree.commitment,
            witness.clamp_multiplicity_tree.commitment,
        ),
        byte_logup=byte,
        byte_roots=(
            witness.byte_witness_tree.commitment,
            witness.byte_multiplicity_tree.commitment,
        ),
    )


def verify_goldilocks_mlp_tile_v3(
    proof: object,
    *,
    statement: GoldilocksMlpTileStatementV3,
    cells_root: bytes,
    validator_nonce: bytes,
    expected_h8_rows: tuple[tuple[int, ...], ...] | None = None,
) -> None:
    try:
        if not isinstance(proof, GoldilocksMlpTileProofV3):
            raise ProofV3VerificationError("mlp tile proof type is unexpected")
        tokens, ff_dim = statement.token_count, statement.ff_dim
        padded = statement.cell_count()
        opening = proof.cells_opening
        if len(opening) != padded or any(len(row) != 6 for row in opening):
            raise ProofV3VerificationError("mlp cells opening shape is wrong")
        rebuilt = GoldilocksMerkleTreeReference.from_rows(
            tuple(tuple(row) for row in opening),
            binding_digest=statement.tables_binding_digest(),
        )
        if rebuilt.commitment != _fixed32(cells_root, "cells_root"):
            raise ProofV3VerificationError(
                "mlp cells opening does not match the committed root"
            )
        divisor = statement.divisor
        shift = GOLDILOCKS_MLP_SHIFT_V3
        silu_pairs: list[int] = []
        clamp_pairs: list[int] = []
        byte_values: list[int] = []
        s_flat: list[int] = []
        u_flat: list[int] = []
        for cell in range(padded):
            g8_f, u8_f, s_f, h_q_f, rem, h8_f = opening[cell]
            in_range = cell < tokens * ff_dim
            g8 = _from_field_signed(g8_f)
            u8 = _from_field_signed(u8_f)
            s = _from_field_signed(s_f)
            h_q = _from_field_signed(h_q_f)
            h8 = _from_field_signed(h8_f)
            if not in_range:
                if any(value != 0 for value in opening[cell]):
                    raise ProofV3VerificationError("mlp padding cell is nonzero")
                s_flat.append(0)
                u_flat.append(0)
                continue
            if not -128 <= g8 <= 127 or not -128 <= u8 <= 127:
                raise ProofV3VerificationError("mlp opened cell is not int8")
            if not 0 <= rem < divisor:
                raise ProofV3VerificationError("mlp remainder is out of range")
            wide = s * u8
            if (h_q + shift) * divisor + rem != wide + divisor * shift + divisor // 2:
                raise ProofV3VerificationError(
                    "mlp Euclidean requantization does not hold"
                )
            index = statement.clamp_index(h_q)
            if h8 != statement.clamp_table[index]:
                raise ProofV3VerificationError("mlp clamp output is wrong")
            silu_pairs.append((g8 + 128) + _SPACK * (s + _SILU_BIAS))
            clamp_pairs.append(index + _SPACK * (h8 + _CLAMP_BIAS))
            byte_values.extend(_limbs(rem, statement.remainder_limbs()))
            s_flat.append(_signed_field(s))
            u_flat.append(_signed_field(u8))
            if expected_h8_rows is not None:
                t, j = divmod(cell, ff_dim)
                if expected_h8_rows[t][j] != h8:
                    raise ProofV3VerificationError(
                        "mlp opened h8 does not match the bound output"
                    )
        factor = _fold_coefficients(
            statement, cells_root=rebuilt.commitment,
            validator_nonce=validator_nonce,
        )
        expected = 0
        for cell in range(padded):
            wide_f = s_flat[cell] * u_flat[cell] % GOLDILOCKS_MODULUS
            expected = (expected + factor[cell] * wide_f) % GOLDILOCKS_MODULUS
        statement_digest = hashlib.sha256(
            statement.digest() + rebuilt.commitment
        ).digest()
        a_tree = commit_goldilocks_product_sumcheck_a_v3(
            statement_digest=statement_digest, evaluations=tuple(s_flat)
        )
        b_tree = commit_goldilocks_product_sumcheck_b_v3(
            statement_digest=statement_digest, evaluations=tuple(u_flat)
        )
        verify_goldilocks_product_sumcheck_v3(
            proof.product_sumcheck,
            statement_digest=statement_digest,
            a_commitment=a_tree.commitment,
            b_commitment=b_tree.commitment,
            factor=factor,
            validator_nonce=validator_nonce,
            expected_sum=expected,
        )
        for logup_statement, pairs, logup_proof, roots in (
            (
                statement.silu_logup_statement(),
                tuple(silu_pairs),
                proof.silu_logup,
                proof.silu_roots,
            ),
            (
                statement.clamp_logup_statement(),
                tuple(clamp_pairs),
                proof.clamp_logup,
                proof.clamp_roots,
            ),
            (
                statement.byte_logup_statement(),
                tuple(byte_values),
                proof.byte_logup,
                proof.byte_roots,
            ),
        ):
            witness_tree, multiplicity_tree = freeze_goldilocks_logup_witness_v3(
                statement=logup_statement, looked_up_values=pairs
            )
            if (
                witness_tree.commitment != _fixed32(roots[0], "logup root")
                or multiplicity_tree.commitment != _fixed32(roots[1], "logup root")
            ):
                raise ProofV3VerificationError(
                    "mlp logup roots do not match the opened cells"
                )
            verify_goldilocks_logup_reference_v3(
                logup_proof,
                statement=logup_statement,
                witness_root=roots[0],
                multiplicity_root=roots[1],
                validator_nonce=validator_nonce,
            )
    except ProofV3VerificationError:
        raise
    except (ProofV3Error, ValueError, TypeError, IndexError) as error:
        raise ProofV3VerificationError(f"mlp tile proof is malformed: {error}")


def build_goldilocks_mlp_silu_table_v3(
    *,
    gate_scale: float,
    silu_scale: float,
) -> tuple[int, ...]:
    """Quantized SiLU over the int8 gate domain at the given scales."""

    import math

    table = []
    for g8 in range(-128, 128):
        real = g8 * gate_scale
        silu = real / (1.0 + math.exp(-real))
        value = int(round(silu / silu_scale))
        value = min(max(value, -(_SILU_BIAS - 1)), _SILU_BIAS - 1)
        table.append(value)
    return tuple(table)


def build_goldilocks_mlp_clamp_table_v3(
    *,
    window: int,
) -> tuple[int, tuple[int, ...]]:
    """Saturating int8 clamp table over ``[-window, window]``."""

    offset = window
    table = tuple(
        min(max(index - offset, -127), 127) for index in range(2 * window + 1)
    )
    return offset, table
