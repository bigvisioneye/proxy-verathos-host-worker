"""Succinct weightless projection relation for proof-v3 selected traces.

This module proves a batch of selected runtime rows satisfies ``X @ W = Y``
without disclosing X, Y, or W and without loading W on the validator.

The static matrix W has a stable Goldilocks BaseFold/FRI commitment qualified
and authenticated by the signed per-model artifact.  Pallas remains the
independent static catalog leg; no Pallas scalar is cast into Goldilocks.
Dynamic X/Y commitments are request-bound and are created only after the
validator nonce.  Binding those dynamic commitments to pre-nonce execution
anchors is a separate, mandatory serving-layer step.

For one operation, nonce-derived multilinear points ``v`` and ``u`` fold the
token and output axes.  The prover commits the helper

    z[i] = MLE_j(W[j, i], u)

and proves

    sum[t, i] eq(v, t) * X[t, i] * z[i] = MLE(Y, (u, v)).

The product sumcheck opens X and z at its terminal point.  The same z value
must open the authenticated static W polynomial at ``(r_i, u)``.  X, z, W,
and Y terminal claims share the existing batched Goldilocks PCS opening.
Wrong rows, wrong weights, omitted relations, and cross-request replay fail
closed.  The construction is field-native throughout.

This is the projection arithmetic component only.  It is not a registered
hard-audit adapter until the caller also verifies capture-to-PCS bindings and
the surrounding residual/nonlinear/attention/GDN/token relations.
"""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from typing import Final, Mapping

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
    GoldilocksMultilinearPcsStatementV3,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.goldilocks_succinct_batch_opening import (
    BatchClaimCheckerV3,
    BatchOpeningCollectorV3,
)
from verallm.proof_v3.goldilocks_succinct_product_argument_reference import (
    GoldilocksSuccinctProductProofV3,
    GoldilocksSuccinctProductStatementV3,
    prove_goldilocks_succinct_product_v3,
    verify_goldilocks_succinct_product_v3,
)
from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
    SuccinctColumnV3,
    column_pcs_statement_v3,
    commit_succinct_column_v3,
    pcs_coset_profile_v3,
)


GOLDILOCKS_PROJECTION_RELATION_ABI_V3: Final = (
    "goldilocks.projection_relation.selected_trace.v1"
)
GOLDILOCKS_STATIC_PROJECTION_PCS_ABI_V3: Final = (
    "goldilocks.static_projection_pcs.basefold_fri.v1"
)
MAX_GOLDILOCKS_PROJECTION_RELATIONS_V3: Final = 512
MAX_GOLDILOCKS_PROJECTION_AXIS_V3: Final = 1 << 20

_STATIC_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_STATIC_PROJECTION/V1"
)
_RELATION_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_PROJECTION_RELATION/V1"
)
_POINT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_PROJECTION_RELATION/V1/POINT"
)
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,127}$")


def _fixed32(value: object, name: str, *, nonzero: bool = False) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    if nonzero and value == bytes(32):
        raise ProofV3Error(f"{name} must not be zero")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ProofV3Error(f"{name} is malformed")
    return value


def _positive_axis(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= MAX_GOLDILOCKS_PROJECTION_AXIS_V3
    ):
        raise ProofV3Error(f"{name} is out of range")
    return value


def _power_of_two_at_least(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _field(value: int) -> int:
    return int(value) % GOLDILOCKS_MODULUS


def _field_tuple(values) -> tuple[int, ...]:
    return tuple(_field(value) for value in values)


def _mle_lsb(values, point) -> int:
    work = [_field(value) for value in values]
    if len(work) != 1 << len(point):
        raise ProofV3Error("projection MLE shape does not match its point")
    for coordinate in point:
        coordinate = _field(coordinate)
        work = [
            (
                work[2 * index]
                + coordinate * (work[2 * index + 1] - work[2 * index])
            )
            % GOLDILOCKS_MODULUS
            for index in range(len(work) // 2)
        ]
    return work[0]


def _eq_vector_lsb(point: tuple[int, ...]) -> tuple[int, ...]:
    """Return ``eq(point, boolean_index)`` in natural integer order."""

    result = []
    for index in range(1 << len(point)):
        value = 1
        for bit, coordinate in enumerate(point):
            coordinate = _field(coordinate)
            value = (
                value
                * (
                    coordinate
                    if (index >> bit) & 1
                    else (1 - coordinate) % GOLDILOCKS_MODULUS
                )
            ) % GOLDILOCKS_MODULUS
        result.append(value)
    return tuple(result)


def _derive_point(seed: bytes, label: bytes, variables: int) -> tuple[int, ...]:
    coordinates = []
    counter = 0
    while len(coordinates) < variables:
        block = hashlib.sha256(
            _POINT_DOMAIN
            + seed
            + struct.pack("<H", len(label))
            + label
            + struct.pack("<I", counter)
        ).digest()
        for offset in range(0, 32, 8):
            candidate = int.from_bytes(block[offset : offset + 8], "little")
            if candidate < GOLDILOCKS_MODULUS:
                coordinates.append(candidate)
                if len(coordinates) == variables:
                    break
        counter += 1
    return tuple(coordinates)


@dataclass(frozen=True, slots=True)
class GoldilocksStaticProjectionStatementV3:
    """Signed statement for one exact static int8 projection matrix."""

    static_artifact_digest: bytes
    operation_id: str
    input_dim: int
    output_dim: int
    weight_scale_bits: int
    weight_encoding_id: str = "int8.row_major.v1"
    pcs_abi_id: str = GOLDILOCKS_STATIC_PROJECTION_PCS_ABI_V3

    def __post_init__(self) -> None:
        _fixed32(
            self.static_artifact_digest,
            "static projection artifact digest",
            nonzero=True,
        )
        _identifier(self.operation_id, "static projection operation id")
        _positive_axis(self.input_dim, "static projection input dimension")
        _positive_axis(self.output_dim, "static projection output dimension")
        if (
            isinstance(self.weight_scale_bits, bool)
            or not isinstance(self.weight_scale_bits, int)
            or not 0 <= self.weight_scale_bits < 1 << 64
        ):
            raise ProofV3Error("static projection scale bits are malformed")
        if self.weight_encoding_id != "int8.row_major.v1":
            raise ProofV3Error("static projection weight encoding is unsupported")
        if self.pcs_abi_id != GOLDILOCKS_STATIC_PROJECTION_PCS_ABI_V3:
            raise ProofV3Error("static projection PCS ABI is unsupported")

    @property
    def padded_input_dim(self) -> int:
        return _power_of_two_at_least(self.input_dim)

    @property
    def padded_output_dim(self) -> int:
        return _power_of_two_at_least(self.output_dim)

    @property
    def variable_count(self) -> int:
        return (
            self.padded_input_dim.bit_length()
            + self.padded_output_dim.bit_length()
            - 2
        )

    def digest(self) -> bytes:
        operation = self.operation_id.encode("ascii")
        encoding = self.weight_encoding_id.encode("ascii")
        pcs = self.pcs_abi_id.encode("ascii")
        return hashlib.sha256(
            _STATIC_DOMAIN
            + self.static_artifact_digest
            + struct.pack(
                "<HIIQHH",
                len(operation),
                self.input_dim,
                self.output_dim,
                self.weight_scale_bits,
                len(encoding),
                len(pcs),
            )
            + operation
            + encoding
            + pcs
        ).digest()

    def column_tag(self) -> str:
        return f"static_w/{self.digest().hex()}"

    def pcs_statement(self) -> GoldilocksMultilinearPcsStatementV3:
        tile_digest = hashlib.sha256(_STATIC_DOMAIN + self.digest()).digest()
        with pcs_coset_profile_v3("chain"):
            return column_pcs_statement_v3(
                tile_digest,
                self.column_tag(),
                self.variable_count,
            )


@dataclass(frozen=True, slots=True)
class GoldilocksStaticProjectionCommitmentV3:
    statement: GoldilocksStaticProjectionStatementV3
    commitment: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.statement, GoldilocksStaticProjectionStatementV3):
            raise ProofV3Error("static projection statement has a wrong type")
        _fixed32(
            self.commitment,
            "static projection commitment",
            nonzero=True,
        )


@dataclass(frozen=True, slots=True)
class GoldilocksProjectionRelationV3:
    """Validator-derived selected-row relation for one registered operation."""

    operation_id: str
    token_count: int
    input_dim: int
    output_dim: int

    def __post_init__(self) -> None:
        _identifier(self.operation_id, "projection relation operation id")
        _positive_axis(self.token_count, "projection relation token count")
        _positive_axis(self.input_dim, "projection relation input dimension")
        _positive_axis(self.output_dim, "projection relation output dimension")

    @property
    def padded_token_count(self) -> int:
        return _power_of_two_at_least(self.token_count)

    @property
    def padded_input_dim(self) -> int:
        return _power_of_two_at_least(self.input_dim)

    @property
    def padded_output_dim(self) -> int:
        return _power_of_two_at_least(self.output_dim)

    def canonical_bytes(self) -> bytes:
        operation = self.operation_id.encode("ascii")
        return (
            struct.pack(
                "<HIII",
                len(operation),
                self.token_count,
                self.input_dim,
                self.output_dim,
            )
            + operation
        )


@dataclass(frozen=True, slots=True)
class GoldilocksProjectionWitnessV3:
    relation: GoldilocksProjectionRelationV3
    input_rows_i8: tuple[tuple[int, ...], ...]
    output_rows_i64: tuple[tuple[int, ...], ...]
    static_weights_i8: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.relation, GoldilocksProjectionRelationV3):
            raise ProofV3Error("projection witness relation has a wrong type")
        if (
            not isinstance(self.input_rows_i8, tuple)
            or len(self.input_rows_i8) != self.relation.token_count
            or any(
                not isinstance(row, tuple)
                or len(row) != self.relation.input_dim
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not -128 <= value <= 127
                    for value in row
                )
                for row in self.input_rows_i8
            )
        ):
            raise ProofV3Error("projection input witness is malformed")
        if (
            not isinstance(self.output_rows_i64, tuple)
            or len(self.output_rows_i64) != self.relation.token_count
            or any(
                not isinstance(row, tuple)
                or len(row) != self.relation.output_dim
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not -(1 << 63) <= value < 1 << 63
                    for value in row
                )
                for row in self.output_rows_i64
            )
        ):
            raise ProofV3Error("projection output witness is malformed")
        if (
            not isinstance(self.static_weights_i8, tuple)
            or len(self.static_weights_i8) != self.relation.output_dim
            or any(
                not isinstance(row, tuple)
                or len(row) != self.relation.input_dim
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not -128 <= value <= 127
                    for value in row
                )
                for row in self.static_weights_i8
            )
        ):
            raise ProofV3Error("projection weight witness is malformed")


@dataclass(frozen=True, slots=True)
class GoldilocksProjectionRelationProofV3:
    x_commitment: bytes
    y_commitment: bytes
    z_commitment: bytes
    folded_output: int
    product_proof: GoldilocksSuccinctProductProofV3

    def __post_init__(self) -> None:
        for value, name in (
            (self.x_commitment, "projection X commitment"),
            (self.y_commitment, "projection Y commitment"),
            (self.z_commitment, "projection z commitment"),
        ):
            _fixed32(value, name, nonzero=True)
        if (
            isinstance(self.folded_output, bool)
            or not isinstance(self.folded_output, int)
            or not 0 <= self.folded_output < GOLDILOCKS_MODULUS
        ):
            raise ProofV3Error("projection folded output is not canonical")
        if not isinstance(
            self.product_proof, GoldilocksSuccinctProductProofV3
        ):
            raise ProofV3Error("projection product proof has a wrong type")


@dataclass(frozen=True, slots=True)
class GoldilocksProjectionBatchProofV3:
    relation_proofs: tuple[GoldilocksProjectionRelationProofV3, ...]
    opening_payload: object
    batched_opening: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.relation_proofs, tuple)
            or not self.relation_proofs
            or len(self.relation_proofs)
            > MAX_GOLDILOCKS_PROJECTION_RELATIONS_V3
            or any(
                not isinstance(item, GoldilocksProjectionRelationProofV3)
                for item in self.relation_proofs
            )
        ):
            raise ProofV3Error("projection relation proof set is malformed")
        if not isinstance(self.batched_opening, bool):
            raise ProofV3Error("projection opening mode is malformed")


def _static_weight_values(
    statement: GoldilocksStaticProjectionStatementV3,
    weights: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    if (
        not isinstance(weights, tuple)
        or len(weights) != statement.output_dim
        or any(
            not isinstance(row, tuple)
            or len(row) != statement.input_dim
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not -128 <= value <= 127
                for value in row
            )
            for row in weights
        )
    ):
        raise ProofV3Error("static projection weights are malformed")
    result = []
    for output in range(statement.padded_output_dim):
        row = weights[output] if output < statement.output_dim else ()
        result.extend(
            (
                _field(row[index])
                if output < statement.output_dim
                and index < statement.input_dim
                else 0
            )
            for index in range(statement.padded_input_dim)
        )
    return tuple(result)


def commit_goldilocks_static_projection_v3(
    *,
    statement: GoldilocksStaticProjectionStatementV3,
    weights_i8: tuple[tuple[int, ...], ...],
    fused=None,
) -> SuccinctColumnV3:
    """Build the qualifier/miner cache for one signed static W commitment."""

    values = _static_weight_values(statement, weights_i8)
    tile_digest = hashlib.sha256(_STATIC_DOMAIN + statement.digest()).digest()
    with pcs_coset_profile_v3("chain"):
        column = commit_succinct_column_v3(
            tile_digest=tile_digest,
            tag=statement.column_tag(),
            values=values,
            fused=fused,
            canonical_input=True,
        )
    if column.pcs_statement.digest() != statement.pcs_statement().digest():
        raise ProofV3Error("static projection PCS statement drifted")
    return column


def static_projection_commitment_v3(
    *,
    statement: GoldilocksStaticProjectionStatementV3,
    column: SuccinctColumnV3,
) -> GoldilocksStaticProjectionCommitmentV3:
    if (
        not isinstance(column, SuccinctColumnV3)
        or column.pcs_statement.digest() != statement.pcs_statement().digest()
    ):
        raise ProofV3Error("static projection column does not match its statement")
    return GoldilocksStaticProjectionCommitmentV3(
        statement=statement,
        commitment=column.tree.commitment,
    )


def _batch_digest(
    *,
    validator_binding_digest: bytes,
    relations: tuple[GoldilocksProjectionRelationV3, ...],
    static_commitments: tuple[GoldilocksStaticProjectionCommitmentV3, ...],
) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(_RELATION_DOMAIN)
    hasher.update(
        _fixed32(
            validator_binding_digest,
            "projection validator binding",
            nonzero=True,
        )
    )
    hasher.update(struct.pack("<I", len(relations)))
    for relation, static in zip(relations, static_commitments, strict=True):
        if relation.operation_id != static.statement.operation_id or (
            relation.input_dim,
            relation.output_dim,
        ) != (
            static.statement.input_dim,
            static.statement.output_dim,
        ):
            raise ProofV3Error(
                "projection relation does not match its signed static operation"
            )
        encoded = relation.canonical_bytes()
        hasher.update(struct.pack("<I", len(encoded)))
        hasher.update(encoded)
        hasher.update(static.statement.digest())
        hasher.update(static.commitment)
    return hasher.digest()


def _dynamic_tags(batch_digest: bytes, index: int) -> tuple[str, str, str]:
    prefix = f"projection/{batch_digest.hex()}/{index}"
    return f"{prefix}/x", f"{prefix}/y", f"{prefix}/z"


def _dynamic_tile_digest(batch_digest: bytes, index: int) -> bytes:
    return hashlib.sha256(
        _RELATION_DOMAIN + batch_digest + struct.pack("<I", index)
    ).digest()


def _pad_rows(
    rows,
    *,
    row_count: int,
    row_width: int,
) -> tuple[int, ...]:
    result = []
    for row_index in range(row_count):
        row = rows[row_index] if row_index < len(rows) else ()
        result.extend(
            _field(row[column]) if column < len(row) else 0
            for column in range(row_width)
        )
    return tuple(result)


def _fold_static_weights_at_output_point(
    *,
    relation: GoldilocksProjectionRelationV3,
    weights,
    output_point: tuple[int, ...],
) -> tuple[int, ...]:
    output_coefficients = _eq_vector_lsb(output_point)
    folded = []
    for input_index in range(relation.padded_input_dim):
        value = 0
        for output_index, coefficient in enumerate(output_coefficients):
            weight = (
                weights[output_index][input_index]
                if output_index < relation.output_dim
                and input_index < relation.input_dim
                else 0
            )
            value = (value + coefficient * weight) % GOLDILOCKS_MODULUS
        folded.append(value)
    return tuple(folded)


def _relation_points(
    *,
    batch_digest: bytes,
    relation_index: int,
    relation: GoldilocksProjectionRelationV3,
    validator_nonce: bytes,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    seed = hashlib.sha256(
        _POINT_DOMAIN
        + batch_digest
        + struct.pack("<I", relation_index)
        + relation.canonical_bytes()
        + _fixed32(validator_nonce, "projection validator nonce", nonzero=True)
    ).digest()
    output_point = _derive_point(
        seed,
        b"output",
        relation.padded_output_dim.bit_length() - 1,
    )
    token_point = _derive_point(
        seed,
        b"token",
        relation.padded_token_count.bit_length() - 1,
    )
    return output_point, token_point


def prove_goldilocks_projection_batch_v3(
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    witnesses: tuple[GoldilocksProjectionWitnessV3, ...],
    static_columns: Mapping[str, SuccinctColumnV3],
    static_commitments: Mapping[str, GoldilocksStaticProjectionCommitmentV3],
    fused=None,
    batched_opening: bool = True,
) -> GoldilocksProjectionBatchProofV3:
    """Prove an exact, ordered selected projection inventory.

    ``batched_opening=False`` exists for bounded CPU conformance tests only.
    A registered production profile must require the batched opening mode.
    """

    witnesses = tuple(witnesses)
    if (
        not witnesses
        or len(witnesses) > MAX_GOLDILOCKS_PROJECTION_RELATIONS_V3
        or len({item.relation.operation_id for item in witnesses})
        != len(witnesses)
    ):
        raise ProofV3Error("projection witness inventory is empty or duplicated")
    relations = tuple(item.relation for item in witnesses)
    ordered_static = tuple(
        static_commitments[item.operation_id] for item in relations
    )
    batch_digest = _batch_digest(
        validator_binding_digest=validator_binding_digest,
        relations=relations,
        static_commitments=ordered_static,
    )
    collector = BatchOpeningCollectorV3()
    relation_proofs = []
    for relation_index, (witness, static_public) in enumerate(
        zip(witnesses, ordered_static, strict=True)
    ):
        relation = witness.relation
        try:
            static_column = static_columns[relation.operation_id]
        except KeyError as exc:
            raise ProofV3Error(
                f"static projection column is missing {relation.operation_id!r}"
            ) from exc
        if (
            static_column.tree.commitment != static_public.commitment
            or static_column.pcs_statement.digest()
            != static_public.statement.pcs_statement().digest()
        ):
            raise ProofV3Error(
                "static projection prover column is not the signed commitment"
            )
        if witness.static_weights_i8 != tuple(
            tuple(
                (
                    value - GOLDILOCKS_MODULUS
                    if value > GOLDILOCKS_MODULUS // 2
                    else value
                )
                for value in static_column.values[
                    output * relation.padded_input_dim :
                    output * relation.padded_input_dim + relation.input_dim
                ]
            )
            for output in range(relation.output_dim)
        ):
            raise ProofV3Error(
                "projection witness weights do not match the cached static column"
            )

        x_tag, y_tag, z_tag = _dynamic_tags(batch_digest, relation_index)
        tile_digest = _dynamic_tile_digest(batch_digest, relation_index)
        x_values = _pad_rows(
            witness.input_rows_i8,
            row_count=relation.padded_token_count,
            row_width=relation.padded_input_dim,
        )
        y_values = _pad_rows(
            witness.output_rows_i64,
            row_count=relation.padded_token_count,
            row_width=relation.padded_output_dim,
        )
        output_point, token_point = _relation_points(
            batch_digest=batch_digest,
            relation_index=relation_index,
            relation=relation,
            validator_nonce=validator_nonce,
        )
        z_values = _fold_static_weights_at_output_point(
            relation=relation,
            weights=witness.static_weights_i8,
            output_point=output_point,
        )
        with pcs_coset_profile_v3("chain"):
            x_column = commit_succinct_column_v3(
                tile_digest=tile_digest,
                tag=x_tag,
                values=x_values,
                fused=fused,
                canonical_input=True,
            )
            y_column = commit_succinct_column_v3(
                tile_digest=tile_digest,
                tag=y_tag,
                values=y_values,
                fused=fused,
                canonical_input=True,
            )
            z_column = commit_succinct_column_v3(
                tile_digest=tile_digest,
                tag=z_tag,
                values=z_values,
                fused=fused,
                canonical_input=True,
            )
        for tag, column in (
            (x_tag, x_column),
            (y_tag, y_column),
            (z_tag, z_column),
            (static_public.statement.column_tag(), static_column),
        ):
            collector.register_column(tag, column)

        token_coefficients = _eq_vector_lsb(token_point)
        broadcast_z = tuple(
            value
            for _token in range(relation.padded_token_count)
            for value in z_values
        )
        product_statement = GoldilocksSuccinctProductStatementV3(
            validator_binding_digest=hashlib.sha256(
                _RELATION_DOMAIN
                + batch_digest
                + struct.pack("<I", relation_index)
            ).digest(),
            variable_count=(
                relation.padded_token_count.bit_length()
                + relation.padded_input_dim.bit_length()
                - 2
            ),
            factor_component_sizes=(
                relation.padded_token_count,
                relation.padded_input_dim,
            ),
        )
        product = prove_goldilocks_succinct_product_v3(
            statement=product_statement,
            a_pcs_statement=x_column.pcs_statement,
            b_pcs_statement=z_column.pcs_statement,
            a_tree=x_column.tree,
            b_tree=z_column.tree,
            a_evaluations=x_values,
            b_evaluations=broadcast_z,
            factor_components=(
                token_coefficients,
                (1,) * relation.padded_input_dim,
            ),
            validator_nonce=validator_nonce,
            collector=collector,
            a_tag=x_tag,
            b_tag=z_tag,
            b_point_map=tuple(
                range(relation.padded_input_dim.bit_length() - 1)
            ),
        )
        z_claim = collector.claims[z_tag][-1]
        collector.defer(
            static_public.statement.column_tag(),
            z_claim.point + output_point,
            z_claim.value,
        )
        y_point = output_point + token_point
        folded_output = _mle_lsb(y_values, y_point)
        collector.defer(y_tag, y_point, folded_output)
        relation_proofs.append(
            GoldilocksProjectionRelationProofV3(
                x_commitment=x_column.tree.commitment,
                y_commitment=y_column.tree.commitment,
                z_commitment=z_column.tree.commitment,
                folded_output=folded_output,
                product_proof=product,
            )
        )

    opening_payload = (
        collector.prove_all_batched(
            validator_nonce=validator_nonce,
            fused=fused,
        )
        if batched_opening
        else collector.prove_all(
            validator_nonce=validator_nonce,
            fused=fused,
        )
    )
    return GoldilocksProjectionBatchProofV3(
        relation_proofs=tuple(relation_proofs),
        opening_payload=opening_payload,
        batched_opening=batched_opening,
    )


def verify_goldilocks_projection_batch_v3(
    proof: object,
    *,
    validator_binding_digest: bytes,
    validator_nonce: bytes,
    relations: tuple[GoldilocksProjectionRelationV3, ...],
    static_commitments: Mapping[str, GoldilocksStaticProjectionCommitmentV3],
    require_batched_opening: bool = True,
) -> None:
    """Verify the exact validator-derived selected projection inventory."""

    try:
        if not isinstance(proof, GoldilocksProjectionBatchProofV3):
            raise ProofV3VerificationError(
                "projection batch proof has a wrong type"
            )
        relations = tuple(relations)
        if (
            not relations
            or len(relations) != len(proof.relation_proofs)
            or len(relations) > MAX_GOLDILOCKS_PROJECTION_RELATIONS_V3
            or len({item.operation_id for item in relations}) != len(relations)
        ):
            raise ProofV3VerificationError(
                "projection relation inventory is incomplete or duplicated"
            )
        if require_batched_opening and not proof.batched_opening:
            raise ProofV3VerificationError(
                "production projection relation requires a batched PCS opening"
            )
        ordered_static = tuple(
            static_commitments[item.operation_id] for item in relations
        )
        batch_digest = _batch_digest(
            validator_binding_digest=validator_binding_digest,
            relations=relations,
            static_commitments=ordered_static,
        )
        checker = BatchClaimCheckerV3()
        statements: dict[str, GoldilocksMultilinearPcsStatementV3] = {}
        commitments: dict[str, bytes] = {}
        for relation_index, (relation, relation_proof, static_public) in enumerate(
            zip(
                relations,
                proof.relation_proofs,
                ordered_static,
                strict=True,
            )
        ):
            x_tag, y_tag, z_tag = _dynamic_tags(batch_digest, relation_index)
            tile_digest = _dynamic_tile_digest(batch_digest, relation_index)
            with pcs_coset_profile_v3("chain"):
                x_statement = column_pcs_statement_v3(
                    tile_digest,
                    x_tag,
                    (
                        relation.padded_token_count
                        * relation.padded_input_dim
                    ).bit_length()
                    - 1,
                )
                y_statement = column_pcs_statement_v3(
                    tile_digest,
                    y_tag,
                    (
                        relation.padded_token_count
                        * relation.padded_output_dim
                    ).bit_length()
                    - 1,
                )
                z_statement = column_pcs_statement_v3(
                    tile_digest,
                    z_tag,
                    relation.padded_input_dim.bit_length() - 1,
                )
            static_tag = static_public.statement.column_tag()
            for tag, statement, commitment in (
                (x_tag, x_statement, relation_proof.x_commitment),
                (y_tag, y_statement, relation_proof.y_commitment),
                (z_tag, z_statement, relation_proof.z_commitment),
                (
                    static_tag,
                    static_public.statement.pcs_statement(),
                    static_public.commitment,
                ),
            ):
                if tag in statements and (
                    statements[tag].digest() != statement.digest()
                    or commitments[tag] != commitment
                ):
                    raise ProofV3VerificationError(
                        "projection PCS tag is inconsistently reused"
                    )
                statements[tag] = statement
                commitments[tag] = commitment

            output_point, token_point = _relation_points(
                batch_digest=batch_digest,
                relation_index=relation_index,
                relation=relation,
                validator_nonce=validator_nonce,
            )
            token_coefficients = _eq_vector_lsb(token_point)
            product_statement = GoldilocksSuccinctProductStatementV3(
                validator_binding_digest=hashlib.sha256(
                    _RELATION_DOMAIN
                    + batch_digest
                    + struct.pack("<I", relation_index)
                ).digest(),
                variable_count=(
                    relation.padded_token_count.bit_length()
                    + relation.padded_input_dim.bit_length()
                    - 2
                ),
                factor_component_sizes=(
                    relation.padded_token_count,
                    relation.padded_input_dim,
                ),
            )
            verify_goldilocks_succinct_product_v3(
                relation_proof.product_proof,
                statement=product_statement,
                a_pcs_statement=x_statement,
                b_pcs_statement=z_statement,
                a_commitment=relation_proof.x_commitment,
                b_commitment=relation_proof.z_commitment,
                factor_components=(
                    token_coefficients,
                    (1,) * relation.padded_input_dim,
                ),
                validator_nonce=validator_nonce,
                expected_sum=relation_proof.folded_output,
                checker=checker,
                a_tag=x_tag,
                b_tag=z_tag,
                b_point_map=tuple(
                    range(relation.padded_input_dim.bit_length() - 1)
                ),
            )
            try:
                z_claim = checker.claims[z_tag][-1]
            except (KeyError, IndexError) as exc:
                raise ProofV3VerificationError(
                    "projection product omitted its helper opening"
                ) from exc
            checker.expect(
                static_tag,
                z_claim.point + output_point,
                z_claim.value,
            )
            checker.expect(
                y_tag,
                output_point + token_point,
                relation_proof.folded_output,
            )

        if proof.batched_opening:
            checker.verify_all_batched(
                proof.opening_payload,
                statements=statements,
                commitments=commitments,
                validator_nonce=validator_nonce,
            )
        else:
            checker.verify_all(
                proof.opening_payload,
                statements=statements,
                commitments=commitments,
                validator_nonce=validator_nonce,
            )
    except ProofV3VerificationError:
        raise
    except (KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "projection batch proof is malformed"
        ) from exc


__all__ = [
    "GOLDILOCKS_PROJECTION_RELATION_ABI_V3",
    "GOLDILOCKS_STATIC_PROJECTION_PCS_ABI_V3",
    "GoldilocksProjectionBatchProofV3",
    "GoldilocksProjectionRelationProofV3",
    "GoldilocksProjectionRelationV3",
    "GoldilocksProjectionWitnessV3",
    "GoldilocksStaticProjectionCommitmentV3",
    "GoldilocksStaticProjectionStatementV3",
    "commit_goldilocks_static_projection_v3",
    "prove_goldilocks_projection_batch_v3",
    "static_projection_commitment_v3",
    "verify_goldilocks_projection_batch_v3",
]
