"""Succinct folded-linear argument + wire format for proof-v3.

Composes the outer fold sumcheck with the BaseFold multilinear PCS so the
verifier never reads a full opening, and freezes the first compact wire
encoding of the combined proof.

Protocol (proves ``sum_i X[i] * F[i] == S`` succinctly):

1. ``X`` is committed with the multilinear PCS (RS codeword root).
2. The outer sumcheck runs exactly as in the fold-sumcheck reference
   (MSB-first halving), reducing the claim to ``X~(c) * F~(c) == running``
   at the challenge vector ``c``.
3. The public factor is supplied in **tensor-product form**
   ``F = F_1 (x) F_2 (x) ...`` (e.g. ``v (x) (W @ u)``), so the verifier
   evaluates ``F~(c)`` as the product of small per-component MLE
   evaluations — O(T + M) work instead of O(T * M).
4. The prover then opens ``X~`` at ``c`` with the succinct PCS.  Variable
   order bridge: the outer sumcheck binds MSB-first while the PCS indexes
   LSB-first, so the PCS point is ``reversed(c)``; the opened value must
   equal ``running * F~(c)^-1``.

Verifier cost: O(log N) round scalars + O(sum |F_j|) factor evaluation +
O(queries * log N) Merkle cells.  No step reads the witness vector.

Wire format ``V3SF``: length-prefixed little-endian sections in transcript
order (claimed sum, outer rounds, PCS rounds, layer roots, final value,
per-layer Merkle openings).  The parser is strict: unknown magic, trailing
bytes, or any out-of-range field is a hard failure.
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
)
from verallm.proof_v3.goldilocks_merkle_reference import (
    GoldilocksMerkleMultiOpeningReference,
    GoldilocksMerkleTreeReference,
)
from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
    GoldilocksMultilinearOpeningProofV3,
    GoldilocksMultilinearPcsStatementV3,
    commit_goldilocks_multilinear_v3,
    open_goldilocks_multilinear_v3,
    verify_goldilocks_multilinear_opening_v3,
)
from verallm.proof_v3.goldilocks_reference import (
    GOLDILOCKS_MODULUS,
    goldilocks_inv,
)


GOLDILOCKS_SUCCINCT_FOLD_ABI_V3: Final = "goldilocks.succinct_fold.reference.v1"
_WIRE_MAGIC: Final = b"V3SF"
_WIRE_VERSION: Final = 1

_TRANSCRIPT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_SUCCINCT_FOLD/V1/TRANSCRIPT/SHA256"
)


def _field(value: object, name: str) -> int:
    integer = _integer(value, name)
    if not 0 <= integer < GOLDILOCKS_MODULUS:
        raise ProofV3Error(f"{name} must be a canonical Goldilocks element")
    return integer


def _mle_eval_msb(values: tuple[int, ...], point: tuple[int, ...]) -> int:
    working = list(values)
    for challenge in point:
        half = len(working) // 2
        working = [
            (working[i] + challenge * (working[half + i] - working[i]))
            % GOLDILOCKS_MODULUS
            for i in range(half)
        ]
    return working[0]


def _tensor_factor(components: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Materialize F = F_1 (x) F_2 (x) ... (component 0 = most significant)."""

    table = (1,)
    for component in components:
        table = tuple(
            left * right % GOLDILOCKS_MODULUS
            for left in table
            for right in component
        )
    return table


@dataclass(frozen=True, slots=True)
class GoldilocksSuccinctFoldStatementV3:
    """Public statement: binding, PCS parameters, factor component sizes."""

    validator_binding_digest: bytes
    variable_count: int
    factor_component_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        binding = _fixed32(
            self.validator_binding_digest, "succinct-fold binding", nonzero=True
        )
        variables = _integer(self.variable_count, "variable_count")
        sizes = tuple(
            _integer(size, "factor component size")
            for size in self.factor_component_sizes
        )
        if not sizes or any(
            size < 1 or size & (size - 1) for size in sizes
        ):
            raise ProofV3Error(
                "succinct-fold factor components must be powers of two"
            )
        total_bits = sum(size.bit_length() - 1 for size in sizes)
        if total_bits != variables:
            raise ProofV3Error(
                "succinct-fold factor components do not span the cube"
            )
        object.__setattr__(self, "validator_binding_digest", binding)
        object.__setattr__(self, "variable_count", variables)
        object.__setattr__(self, "factor_component_sizes", sizes)

    def digest(self) -> bytes:
        return hashlib.sha256(
            _TRANSCRIPT_DOMAIN
            + self.validator_binding_digest
            + struct.pack("<I", self.variable_count)
            + struct.pack(
                f"<{len(self.factor_component_sizes)}I",
                *self.factor_component_sizes,
            )
        ).digest()

    def pcs_statement(self) -> GoldilocksMultilinearPcsStatementV3:
        return GoldilocksMultilinearPcsStatementV3(
            validator_binding_digest=self.digest(),
            variable_count=self.variable_count,
        )


@dataclass(frozen=True, slots=True)
class GoldilocksSuccinctFoldProofV3:
    claimed_sum: int
    outer_rounds: tuple[tuple[int, int, int], ...]
    opening: GoldilocksMultilinearOpeningProofV3


def commit_goldilocks_succinct_fold_witness_v3(
    *,
    statement: GoldilocksSuccinctFoldStatementV3,
    x_evaluations: tuple[int, ...],
) -> GoldilocksMerkleTreeReference:
    return commit_goldilocks_multilinear_v3(
        statement=statement.pcs_statement(),
        evaluations=x_evaluations,
    )


def _outer_seed(
    statement: GoldilocksSuccinctFoldStatementV3,
    commitment: bytes,
    factor_digest: bytes,
    claimed: int,
    validator_nonce: bytes,
) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + statement.digest()
        + _fixed32(commitment, "succinct-fold commitment")
        + factor_digest
        + claimed.to_bytes(8, "little")
        + _fixed32(validator_nonce, "validator_nonce")
    ).digest()


def _factor_digest(components: tuple[tuple[int, ...], ...]) -> bytes:
    h = hashlib.sha256()
    for component in components:
        h.update(struct.pack("<I", len(component)))
        h.update(
            b"".join(value.to_bytes(8, "little") for value in component)
        )
    return h.digest()


def prove_goldilocks_succinct_fold_v3(
    *,
    statement: GoldilocksSuccinctFoldStatementV3,
    tree: GoldilocksMerkleTreeReference,
    x_evaluations: tuple[int, ...],
    factor_components: tuple[tuple[int, ...], ...],
    validator_nonce: bytes,
) -> GoldilocksSuccinctFoldProofV3:
    if tuple(len(c) for c in factor_components) != (
        statement.factor_component_sizes
    ):
        raise ProofV3Error("succinct-fold factor component shapes are wrong")
    components = tuple(
        tuple(_field(v, "factor value") for v in component)
        for component in factor_components
    )
    x_values = [
        _field(v, "x evaluation") for v in x_evaluations
    ]
    if len(x_values) != 1 << statement.variable_count:
        raise ProofV3Error("succinct-fold witness size is wrong")
    f_values = list(_tensor_factor(components))
    claimed = 0
    for x, f in zip(x_values, f_values, strict=True):
        claimed = (claimed + x * f) % GOLDILOCKS_MODULUS
    transcript = _outer_seed(
        statement,
        tree.commitment,
        _factor_digest(components),
        claimed,
        validator_nonce,
    )
    rounds: list[tuple[int, int, int]] = []
    challenges: list[int] = []
    while len(x_values) > 1:
        half = len(x_values) // 2
        g0 = g1 = g2 = 0
        for i in range(half):
            x_lo, x_hi = x_values[i], x_values[half + i]
            f_lo, f_hi = f_values[i], f_values[half + i]
            g0 = (g0 + x_lo * f_lo) % GOLDILOCKS_MODULUS
            g1 = (g1 + x_hi * f_hi) % GOLDILOCKS_MODULUS
            g2 = (
                g2
                + (2 * x_hi - x_lo) * (2 * f_hi - f_lo)
            ) % GOLDILOCKS_MODULUS
        rounds.append((g0, g1, g2))
        transcript = hashlib.sha256(
            transcript
            + g0.to_bytes(8, "little")
            + g1.to_bytes(8, "little")
            + g2.to_bytes(8, "little")
        ).digest()
        challenge = _challenge(transcript, len(rounds))
        challenges.append(challenge)
        x_values = [
            (x_values[i] + challenge * (x_values[half + i] - x_values[i]))
            % GOLDILOCKS_MODULUS
            for i in range(half)
        ]
        f_values = [
            (f_values[i] + challenge * (f_values[half + i] - f_values[i]))
            % GOLDILOCKS_MODULUS
            for i in range(half)
        ]
    # MSB-first challenge j binds index bit (n-1-j); PCS points are
    # LSB-first, so the opening point is the reversed challenge vector.
    pcs_point = tuple(reversed(challenges))
    opening = open_goldilocks_multilinear_v3(
        statement=statement.pcs_statement(),
        tree=tree,
        evaluations=tuple(
            _field(v, "x evaluation") for v in x_evaluations
        ),
        point=pcs_point,
        validator_nonce=validator_nonce,
    )
    return GoldilocksSuccinctFoldProofV3(
        claimed_sum=claimed,
        outer_rounds=tuple(rounds),
        opening=opening,
    )


def verify_goldilocks_succinct_fold_v3(
    proof: object,
    *,
    statement: GoldilocksSuccinctFoldStatementV3,
    commitment: bytes,
    factor_components: tuple[tuple[int, ...], ...],
    validator_nonce: bytes,
    expected_sum: int,
) -> None:
    """Succinctly verify sum X*F == expected_sum; never reads the witness."""

    try:
        if not isinstance(proof, GoldilocksSuccinctFoldProofV3):
            raise ProofV3VerificationError("succinct-fold proof type is wrong")
        if tuple(len(c) for c in factor_components) != (
            statement.factor_component_sizes
        ):
            raise ProofV3VerificationError(
                "succinct-fold factor component shapes are wrong"
            )
        components = tuple(
            tuple(_field(v, "factor value") for v in component)
            for component in factor_components
        )
        n = statement.variable_count
        if len(proof.outer_rounds) != n:
            raise ProofV3VerificationError("succinct-fold round count is wrong")
        if proof.claimed_sum != _field(expected_sum, "expected_sum"):
            raise ProofV3VerificationError(
                "succinct-fold claimed sum does not match the outer relation"
            )
        transcript = _outer_seed(
            statement,
            _fixed32(commitment, "succinct-fold commitment"),
            _factor_digest(components),
            proof.claimed_sum,
            validator_nonce,
        )
        running = proof.claimed_sum
        challenges: list[int] = []
        inv2 = (GOLDILOCKS_MODULUS + 1) // 2
        for round_index, (g0, g1, g2) in enumerate(proof.outer_rounds, 1):
            g0, g1, g2 = _field(g0, "g0"), _field(g1, "g1"), _field(g2, "g2")
            if (g0 + g1) % GOLDILOCKS_MODULUS != running:
                raise ProofV3VerificationError(
                    "succinct-fold round does not match the running sum"
                )
            transcript = hashlib.sha256(
                transcript
                + g0.to_bytes(8, "little")
                + g1.to_bytes(8, "little")
                + g2.to_bytes(8, "little")
            ).digest()
            challenge = _challenge(transcript, round_index)
            challenges.append(challenge)
            z = challenge
            running = (
                g0 * ((z - 1) * (z - 2) % GOLDILOCKS_MODULUS)
                % GOLDILOCKS_MODULUS
                * inv2
                - g1 * (z * (z - 2) % GOLDILOCKS_MODULUS)
                + g2 * (z * (z - 1) % GOLDILOCKS_MODULUS)
                % GOLDILOCKS_MODULUS
                * inv2
            ) % GOLDILOCKS_MODULUS
        # F~(c): per-component MLE evaluations over the matching challenge
        # slices (component 0 owns the most significant variables).
        cursor = 0
        f_at_point = 1
        for component in components:
            bits = len(component).bit_length() - 1
            slice_point = tuple(challenges[cursor : cursor + bits])
            cursor += bits
            f_at_point = (
                f_at_point * _mle_eval_msb(component, slice_point)
            ) % GOLDILOCKS_MODULUS
        if f_at_point == 0:
            raise ProofV3VerificationError(
                "succinct-fold factor evaluates to zero at the challenge"
            )
        expected_x = running * goldilocks_inv(f_at_point) % GOLDILOCKS_MODULUS
        verify_goldilocks_multilinear_opening_v3(
            proof.opening,
            statement=statement.pcs_statement(),
            commitment=commitment,
            point=tuple(reversed(challenges)),
            expected_value=expected_x,
            validator_nonce=validator_nonce,
        )
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "succinct-fold proof is malformed"
        ) from exc


# ── wire format ──────────────────────────────────────────────────────


def _encode_opening(opening: GoldilocksMerkleMultiOpeningReference) -> bytes:
    parts = [opening.binding_digest]
    parts.append(
        struct.pack(
            "<QII",
            opening.leaf_count,
            opening.leaf_width,
            len(opening.indices),
        )
    )
    parts.append(struct.pack(f"<{len(opening.indices)}Q", *opening.indices))
    for row in opening.rows:
        parts.append(b"".join(value.to_bytes(8, "little") for value in row))
    parts.append(struct.pack("<I", len(opening.siblings)))
    for sibling in opening.siblings:
        parts.append(struct.pack("<II", sibling.level, sibling.index))
        parts.append(sibling.digest)
    return b"".join(parts)


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def take(self, count: int) -> bytes:
        if self.offset + count > len(self.data):
            raise ProofV3Error("succinct-fold wire payload is truncated")
        chunk = self.data[self.offset : self.offset + count]
        self.offset += count
        return chunk

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def field(self) -> int:
        value = self.u64()
        if value >= GOLDILOCKS_MODULUS:
            raise ProofV3Error("succinct-fold wire field is not canonical")
        return value


def _decode_opening(
    reader: _Reader,
) -> GoldilocksMerkleMultiOpeningReference:
    from verallm.proof_v3.goldilocks_merkle_reference import (
        GoldilocksMerkleSiblingReference,
    )

    binding_digest = reader.take(32)
    leaf_count = reader.u64()
    width = reader.u32()
    count = reader.u32()
    if count > 1 << 16 or width > 1 << 12:
        raise ProofV3Error("succinct-fold wire opening is out of range")
    indices = tuple(reader.u64() for _ in range(count))
    rows = tuple(
        tuple(reader.field() for _ in range(width)) for _ in range(count)
    )
    sibling_count = reader.u32()
    if sibling_count > 1 << 20:
        raise ProofV3Error("succinct-fold wire sibling count is out of range")
    siblings = tuple(
        GoldilocksMerkleSiblingReference(
            level=reader.u32(),
            index=reader.u32(),
            digest=reader.take(32),
        )
        for _ in range(sibling_count)
    )
    return GoldilocksMerkleMultiOpeningReference(
        binding_digest=binding_digest,
        leaf_count=leaf_count,
        leaf_width=width,
        indices=indices,
        rows=rows,
        siblings=siblings,
    )


def encode_goldilocks_succinct_fold_proof_v3(
    proof: GoldilocksSuccinctFoldProofV3,
) -> bytes:
    parts = [_WIRE_MAGIC, struct.pack("<H", _WIRE_VERSION)]
    parts.append(proof.claimed_sum.to_bytes(8, "little"))
    parts.append(struct.pack("<I", len(proof.outer_rounds)))
    for g0, g1, g2 in proof.outer_rounds:
        parts.append(
            g0.to_bytes(8, "little")
            + g1.to_bytes(8, "little")
            + g2.to_bytes(8, "little")
        )
    opening = proof.opening
    parts.append(opening.claimed_value.to_bytes(8, "little"))
    parts.append(struct.pack("<I", len(opening.round_polynomials)))
    for g0, g1, g2 in opening.round_polynomials:
        parts.append(
            g0.to_bytes(8, "little")
            + g1.to_bytes(8, "little")
            + g2.to_bytes(8, "little")
        )
    parts.append(struct.pack("<I", len(opening.layer_commitments)))
    for root in opening.layer_commitments:
        parts.append(root)
    parts.append(opening.final_value.to_bytes(8, "little"))
    parts.append(struct.pack("<I", len(opening.layer_openings)))
    for layer_opening in opening.layer_openings:
        encoded = _encode_opening(layer_opening)
        parts.append(struct.pack("<I", len(encoded)))
        parts.append(encoded)
    return b"".join(parts)


def decode_goldilocks_succinct_fold_proof_v3(
    payload: bytes,
) -> GoldilocksSuccinctFoldProofV3:
    reader = _Reader(bytes(payload))
    if reader.take(4) != _WIRE_MAGIC:
        raise ProofV3Error("succinct-fold wire magic is unknown")
    if struct.unpack("<H", reader.take(2))[0] != _WIRE_VERSION:
        raise ProofV3Error("succinct-fold wire version is unsupported")
    claimed = reader.field()
    outer_count = reader.u32()
    if outer_count > 64:
        raise ProofV3Error("succinct-fold wire round count is out of range")
    outer_rounds = tuple(
        (reader.field(), reader.field(), reader.field())
        for _ in range(outer_count)
    )
    opening_claimed = reader.field()
    pcs_count = reader.u32()
    if pcs_count > 64:
        raise ProofV3Error("succinct-fold wire PCS round count is out of range")
    pcs_rounds = tuple(
        (reader.field(), reader.field(), reader.field())
        for _ in range(pcs_count)
    )
    layer_count = reader.u32()
    if layer_count > 64:
        raise ProofV3Error("succinct-fold wire layer count is out of range")
    layer_roots = tuple(reader.take(32) for _ in range(layer_count))
    final_value = reader.field()
    opening_count = reader.u32()
    if opening_count > 65:
        raise ProofV3Error("succinct-fold wire opening count is out of range")
    layer_openings = []
    for _ in range(opening_count):
        length = reader.u32()
        sub = _Reader(reader.take(length))
        layer_openings.append(_decode_opening(sub))
        if sub.offset != len(sub.data):
            raise ProofV3Error("succinct-fold wire opening has trailing bytes")
    if reader.offset != len(reader.data):
        raise ProofV3Error("succinct-fold wire payload has trailing bytes")
    return GoldilocksSuccinctFoldProofV3(
        claimed_sum=claimed,
        outer_rounds=outer_rounds,
        opening=GoldilocksMultilinearOpeningProofV3(
            claimed_value=opening_claimed,
            round_polynomials=pcs_rounds,
            layer_commitments=layer_roots,
            final_value=final_value,
            layer_openings=tuple(layer_openings),
        ),
    )


__all__ = [
    "GOLDILOCKS_SUCCINCT_FOLD_ABI_V3",
    "GoldilocksSuccinctFoldProofV3",
    "GoldilocksSuccinctFoldStatementV3",
    "commit_goldilocks_succinct_fold_witness_v3",
    "decode_goldilocks_succinct_fold_proof_v3",
    "encode_goldilocks_succinct_fold_proof_v3",
    "prove_goldilocks_succinct_fold_v3",
    "verify_goldilocks_succinct_fold_v3",
]
