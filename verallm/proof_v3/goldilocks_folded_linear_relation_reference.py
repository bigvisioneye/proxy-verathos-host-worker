"""Nonce-folded global linear-relation reference for proof-v3.

The all-row reference (`goldilocks_linear_relation_reference`) commits one
trace row per multiply: O(T*M*N) rows, unusable at model scale (see the A40
spike: ~11 ns per committed cell).  This module implements the scalable
shape: after the runtime sources are frozen, the validator nonce derives
per-index independent fold coefficients ``v[t]`` and ``u[j]``, and the
relation ``X[T,M] @ W[M,N] == Y[T,N]`` is checked as the folded scalar
identity ``v^T @ X @ (W @ u) == v^T @ Y @ u``.

Committed rows collapse to the two source scans, O(T*M + T*N):

* an x-scan segment accumulates ``sum_{t,k} v[t] * X[t,k] * wu[k]`` where
  ``wu = W @ u`` is recomputed by the verifier from the signed weights it
  owns (no proof needed for the weight side at all);
* a y-scan segment accumulates ``sum_{t,j} v[t] * u[j] * Y[t,j]``;
* both segments must end at the same claimed scalar ``S``, enforced by
  boundary constraints whose public value the verifier substitutes itself.

Coefficient columns and segment selectors are public: the verifier evaluates
their LDEs from the nonce and its own weights, so only ``val`` (the source
scan) and ``acc`` are committed.  The ``val`` column is cross-checked
cell-for-cell against an independently frozen source oracle (stand-in for
the capture chain's runtime commitments) at every query index.

If ``X @ W != Y`` then the folded identity fails except with probability
about ``(T + N) / p`` over the nonce-derived coefficients (Schwartz-Zippel),
independently per batch.

Production notes: the remaining O(T*M) committed x-scan is eliminated by a
sumcheck against an algebraic (field-static / multilinear) commitment of X,
which is also why the static catalog must be field-native rather than a
SHA-byte tree — SHA trees cannot answer folded openings.  This reference
keeps the accumulator-trace form so the whole protocol stays inside the
existing Goldilocks Merkle/FRI substrate.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_fri_reference import (
    GoldilocksFriProofReference,
    GoldilocksFriStatementReference,
    prove_goldilocks_fri_reference,
    verify_goldilocks_fri_reference,
)
from verallm.proof_v3.goldilocks_linear_relation_reference import (
    _derive_nonzero_field,
    _field_encode_signed,
    _fixed32,
    _int8,
    _integer,
    _next_power_of_two,
    _u32,
)
from verallm.proof_v3.goldilocks_merkle_reference import (
    GoldilocksMerkleMultiOpeningReference,
    GoldilocksMerkleTreeReference,
    verify_goldilocks_merkle_multiopening_reference,
)
from verallm.proof_v3.goldilocks_reference import (
    GOLDILOCKS_MODULUS,
    MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE,
    goldilocks_inv,
    goldilocks_radix2_domain_reference,
    lde_goldilocks_reference,
)


GOLDILOCKS_FOLDED_LINEAR_ABI_V3: Final = "goldilocks.folded_linear.reference.v1"
GOLDILOCKS_FOLDED_LINEAR_LDE_BLOWUP_V3: Final = 4
GOLDILOCKS_FOLDED_LINEAR_QUERY_COUNT_V3: Final = 16
GOLDILOCKS_FOLDED_LINEAR_BATCH_COUNT_V3: Final = 2

_TRACE_COLUMN_IDS_V3: Final = ("val", "acc")
_CONSTRAINT_IDS_V3: Final = (
    "segment_first",
    "segment_transition",
    "x_scan_total",
    "y_scan_total",
    "padding_val",
    "padding_acc",
)

_STATEMENT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_FOLDED/V1/STATEMENT/SHA256"
_SHIFT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_FOLDED/V1/LDE_SHIFT/SHA256"
_TRACE_TREE_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_FOLDED/V1/TRACE_TREE/SHA256"
_SOURCE_ORACLE_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_FOLDED/V1/SOURCE_ORACLE/SHA256"
)
_PRECOMMITMENT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_FOLDED/V1/PRECOMMITMENT/SHA256"
)
_FOLD_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_FOLDED/V1/FOLD/SHA256"
_POSTCLAIM_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_FOLDED/V1/POSTCLAIM/SHA256"
_TRACE_FRI_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_FOLDED/V1/TRACE_FRI/SHA256"
_COMPOSITION_FRI_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_FOLDED/V1/COMPOSITION_FRI/SHA256"
)


@dataclass(frozen=True, slots=True)
class GoldilocksFoldedLinearStatementV3:
    """Validator-owned folded-relation statement with the signed weights."""

    validator_binding_digest: bytes
    token_count: int
    contraction_length: int
    output_features: int
    weights: tuple[int, ...]
    active_row_count: int = field(init=False)
    x_scan_rows: int = field(init=False)
    trace_domain_size: int = field(init=False)
    lde_domain_size: int = field(init=False)
    lde_shift: int = field(init=False)
    query_count: int = field(init=False)
    composition_degree_bound: int = field(init=False)

    def __post_init__(self) -> None:
        binding = _fixed32(
            self.validator_binding_digest,
            "folded-linear validator_binding_digest",
            nonzero=True,
        )
        tokens = _u32(self.token_count, "token_count", positive=True)
        contraction = _u32(self.contraction_length, "contraction_length", positive=True)
        outputs = _u32(self.output_features, "output_features", positive=True)
        if not isinstance(self.weights, tuple) or len(self.weights) != (
            contraction * outputs
        ):
            raise ProofV3Error("folded-linear weight count does not match the shape")
        weights = tuple(
            _int8(value, f"weights[{index}]")
            for index, value in enumerate(self.weights)
        )
        x_scan_rows = tokens * contraction
        active = x_scan_rows + tokens * outputs
        domain = _next_power_of_two(active, name="folded-linear active rows")
        if domain == active:
            domain *= 2
        lde = domain * GOLDILOCKS_FOLDED_LINEAR_LDE_BLOWUP_V3
        if lde > MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE:
            raise ProofV3Error("folded-linear trace exceeds the CPU reference cap")
        query_count = min(GOLDILOCKS_FOLDED_LINEAR_QUERY_COUNT_V3, lde // 2)
        maximum_quotient_degree = max(0, 3 * (domain - 1) - domain)
        bound = 1
        while bound <= maximum_quotient_degree:
            bound <<= 1
        composition_degree_bound = max(bound, max(1, lde // 64))
        if composition_degree_bound >= lde:
            raise ProofV3Error("folded-linear quotient does not fit the LDE domain")
        object.__setattr__(self, "validator_binding_digest", binding)
        object.__setattr__(self, "token_count", tokens)
        object.__setattr__(self, "contraction_length", contraction)
        object.__setattr__(self, "output_features", outputs)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "active_row_count", active)
        object.__setattr__(self, "x_scan_rows", x_scan_rows)
        object.__setattr__(self, "trace_domain_size", domain)
        object.__setattr__(self, "lde_domain_size", lde)
        shift_seed = hashlib.sha256(
            _SHIFT_DOMAIN
            + binding
            + struct.pack("<IIIQ", tokens, contraction, outputs, lde)
            + b"".join(struct.pack("<b", weight) for weight in weights)
        ).digest()
        for counter in range(1 << 16):
            candidate = int.from_bytes(
                hashlib.sha256(shift_seed + struct.pack("<I", counter)).digest()[:8],
                "little",
            )
            if (
                0 < candidate < GOLDILOCKS_MODULUS
                and pow(candidate, lde, GOLDILOCKS_MODULUS) != 1
            ):
                object.__setattr__(self, "lde_shift", candidate)
                break
        else:
            raise ProofV3Error("unable to derive a folded-linear LDE coset shift")
        object.__setattr__(self, "query_count", query_count)
        object.__setattr__(
            self,
            "composition_degree_bound",
            composition_degree_bound,
        )

    def digest(self) -> bytes:
        return hashlib.sha256(
            _STATEMENT_DOMAIN
            + self.validator_binding_digest
            + struct.pack(
                "<IIIQQQ",
                self.token_count,
                self.contraction_length,
                self.output_features,
                self.trace_domain_size,
                self.lde_domain_size,
                self.lde_shift,
            )
            + b"".join(struct.pack("<b", weight) for weight in self.weights)
        ).digest()

    def trace_tree_binding_digest(self) -> bytes:
        return hashlib.sha256(_TRACE_TREE_DOMAIN + self.digest()).digest()

    def source_oracle_binding_digest(self) -> bytes:
        return hashlib.sha256(_SOURCE_ORACLE_DOMAIN + self.digest()).digest()


@dataclass(frozen=True, slots=True)
class GoldilocksFoldedLinearPrecommitmentV3:
    """Frozen trace root and source-scan oracle root for one statement."""

    statement_digest: bytes
    trace_lde_commitment: bytes
    source_oracle_commitment: bytes

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "statement_digest",
            _fixed32(self.statement_digest, "statement_digest", nonzero=True),
        )
        object.__setattr__(
            self,
            "trace_lde_commitment",
            _fixed32(self.trace_lde_commitment, "trace_lde_commitment"),
        )
        object.__setattr__(
            self,
            "source_oracle_commitment",
            _fixed32(self.source_oracle_commitment, "source_oracle_commitment"),
        )

    def digest(self) -> bytes:
        return hashlib.sha256(
            _PRECOMMITMENT_DOMAIN
            + self.statement_digest
            + self.trace_lde_commitment
            + self.source_oracle_commitment
        ).digest()


def _fold_coefficients(
    *,
    statement: GoldilocksFoldedLinearStatementV3,
    precommitment: GoldilocksFoldedLinearPrecommitmentV3,
    validator_nonce: bytes,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Derive per-index independent v[t] and u[j] strictly post-freeze."""

    seed = hashlib.sha256(
        _FOLD_DOMAIN
        + precommitment.digest()
        + _fixed32(validator_nonce, "folded-linear validator_nonce")
    ).digest()
    v = tuple(
        _derive_nonzero_field(
            transcript_digest=seed,
            label=b"fold-token",
            batch_index=0,
            coordinate_index=index,
            coordinate_id="v",
        )
        for index in range(statement.token_count)
    )
    u = tuple(
        _derive_nonzero_field(
            transcript_digest=seed,
            label=b"fold-output",
            batch_index=0,
            coordinate_index=index,
            coordinate_id="u",
        )
        for index in range(statement.output_features)
    )
    return v, u


def _public_columns(
    *,
    statement: GoldilocksFoldedLinearStatementV3,
    v: tuple[int, ...],
    u: tuple[int, ...],
) -> dict[str, tuple[int, ...]]:
    """Verifier-computable coefficient and selector base columns."""

    contraction = statement.contraction_length
    outputs = statement.output_features
    wu = tuple(
        sum(
            _field_encode_signed(statement.weights[k * outputs + j]) * u[j]
            for j in range(outputs)
        )
        % GOLDILOCKS_MODULUS
        for k in range(contraction)
    )
    size = statement.trace_domain_size
    x_rows = statement.x_scan_rows
    active = statement.active_row_count
    coeff, seg_first, seg_cont = [], [], []
    x_last, y_last, padding = [], [], []
    for row in range(size):
        if row < x_rows:
            token, step = divmod(row, contraction)
            coeff.append(v[token] * wu[step] % GOLDILOCKS_MODULUS)
        elif row < active:
            token, feature = divmod(row - x_rows, outputs)
            coeff.append(v[token] * u[feature] % GOLDILOCKS_MODULUS)
        else:
            coeff.append(0)
        seg_first.append(1 if row in (0, x_rows) else 0)
        seg_cont.append(
            1 if (row + 1 < active and row + 1 != x_rows) and row < active else 0
        )
        x_last.append(1 if row == x_rows - 1 else 0)
        y_last.append(1 if row == active - 1 else 0)
        padding.append(0 if row < active else 1)
    return {
        name: lde_goldilocks_reference(
            tuple(base),
            target_size=statement.lde_domain_size,
            source_shift=1,
            target_shift=statement.lde_shift,
        )
        for name, base in (
            ("coeff", coeff),
            ("seg_first", seg_first),
            ("seg_cont", seg_cont),
            ("x_last", x_last),
            ("y_last", y_last),
            ("padding", padding),
        )
    }


def _constraint_values(
    *,
    row: tuple[int, ...],
    next_row: tuple[int, ...],
    public: dict[str, int],
    claimed_total: int,
) -> tuple[int, ...]:
    val, acc = row
    val_next, acc_next = next_row
    modulus = GOLDILOCKS_MODULUS
    return (
        public["seg_first"] * (acc - val * public["coeff"]) % modulus,
        public["seg_cont"]
        * (acc_next - acc - val_next * public["coeff_next"])
        % modulus,
        public["x_last"] * (acc - claimed_total) % modulus,
        public["y_last"] * (acc - claimed_total) % modulus,
        public["padding"] * val % modulus,
        public["padding"] * acc % modulus,
    )


@dataclass(frozen=True, slots=True)
class GoldilocksFoldedLinearOracleV3:
    statement: GoldilocksFoldedLinearStatementV3
    precommitment: GoldilocksFoldedLinearPrecommitmentV3
    trace_tree: GoldilocksMerkleTreeReference
    source_tree: GoldilocksMerkleTreeReference
    source_base: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.precommitment.statement_digest != self.statement.digest():
            raise ProofV3Error("folded-linear oracle statement digest mismatch")
        if self.trace_tree.commitment != self.precommitment.trace_lde_commitment:
            raise ProofV3Error("folded-linear trace root mismatch")
        if self.source_tree.commitment != (
            self.precommitment.source_oracle_commitment
        ):
            raise ProofV3Error("folded-linear source oracle root mismatch")


def freeze_goldilocks_folded_linear_sources_v3(
    *,
    statement: GoldilocksFoldedLinearStatementV3,
    x_values: tuple[tuple[int, ...], ...],
    y_values: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], GoldilocksMerkleTreeReference]:
    """Freeze the concatenated source scan (X rows then Y rows) pre-nonce.

    Shapes and int8/int32 encodings are validated; ``X @ W == Y`` is
    deliberately not checked here.
    """

    tokens = statement.token_count
    contraction = statement.contraction_length
    outputs = statement.output_features
    if len(x_values) != tokens or any(
        len(row) != contraction for row in x_values
    ):
        raise ProofV3Error("folded-linear x sources have an unexpected shape")
    if len(y_values) != tokens or any(len(row) != outputs for row in y_values):
        raise ProofV3Error("folded-linear y sources have an unexpected shape")
    bound = contraction * (1 << 14)
    base: list[int] = []
    for row in x_values:
        base.extend(_field_encode_signed(_int8(value, "x source")) for value in row)
    for row in y_values:
        for value in row:
            integer = _integer(value, "y source")
            if integer < -bound or integer > bound:
                raise ProofV3Error("y source is out of the accumulator range")
            base.append(_field_encode_signed(integer))
    base.extend(0 for _ in range(statement.trace_domain_size - len(base)))
    source_lde = lde_goldilocks_reference(
        tuple(base),
        target_size=statement.lde_domain_size,
        source_shift=1,
        target_shift=statement.lde_shift,
    )
    tree = GoldilocksMerkleTreeReference.from_rows(
        tuple((value,) for value in source_lde),
        binding_digest=statement.source_oracle_binding_digest(),
    )
    return tuple(base), tree


def prove_goldilocks_folded_linear_reference_v3(
    *,
    statement: GoldilocksFoldedLinearStatementV3,
    source_base: tuple[int, ...],
    source_tree: GoldilocksMerkleTreeReference,
    validator_nonce: bytes,
) -> tuple[GoldilocksFoldedLinearPrecommitmentV3, "GoldilocksFoldedLinearProofV3"]:
    """Post-nonce prover: build the accumulator trace and the folded proof.

    Unlike the per-MAC reference, the accumulator trace itself is a
    post-nonce object (it depends on the fold coefficients).  Only the
    source scan is frozen pre-nonce; the trace tree is committed inside the
    proof and every binding challenge is derived after that commitment.
    """

    interim = GoldilocksFoldedLinearPrecommitmentV3(
        statement_digest=statement.digest(),
        trace_lde_commitment=bytes(32),
        source_oracle_commitment=source_tree.commitment,
    )
    v, u = _fold_coefficients(
        statement=statement,
        precommitment=interim,
        validator_nonce=validator_nonce,
    )
    public = _public_columns(statement=statement, v=v, u=u)
    contraction = statement.contraction_length
    outputs = statement.output_features
    x_rows = statement.x_scan_rows
    acc_base: list[int] = []
    acc = 0
    for row in range(statement.trace_domain_size):
        if row >= statement.active_row_count:
            acc_base.append(0)
            continue
        # Coefficient recomputed directly from the base definition:
        if row < x_rows:
            token, step = divmod(row, contraction)
            wu_step = sum(
                _field_encode_signed(statement.weights[step * outputs + j]) * u[j]
                for j in range(outputs)
            ) % GOLDILOCKS_MODULUS
            coefficient = v[token] * wu_step % GOLDILOCKS_MODULUS
        else:
            token, feature = divmod(row - x_rows, outputs)
            coefficient = v[token] * u[feature] % GOLDILOCKS_MODULUS
        term = source_base[row] * coefficient % GOLDILOCKS_MODULUS
        acc = term if row in (0, x_rows) else (acc + term) % GOLDILOCKS_MODULUS
        acc_base.append(acc)
    claimed_total = acc_base[x_rows - 1]
    lde_columns = tuple(
        lde_goldilocks_reference(
            tuple(column),
            target_size=statement.lde_domain_size,
            source_shift=1,
            target_shift=statement.lde_shift,
        )
        for column in (source_base, tuple(acc_base))
    )
    lde_rows = tuple(
        tuple(column[row] for column in lde_columns)
        for row in range(statement.lde_domain_size)
    )
    trace_tree = GoldilocksMerkleTreeReference.from_rows(
        lde_rows,
        binding_digest=statement.trace_tree_binding_digest(),
    )
    precommitment = GoldilocksFoldedLinearPrecommitmentV3(
        statement_digest=statement.digest(),
        trace_lde_commitment=trace_tree.commitment,
        source_oracle_commitment=source_tree.commitment,
    )
    transcript = _postclaim_digest(
        precommitment=precommitment,
        validator_nonce=validator_nonce,
        claimed_total=claimed_total,
    )
    trace_statements, composition_statements = _fri_statements(
        statement=statement,
        transcript=transcript,
    )
    trace_proofs = tuple(
        prove_goldilocks_fri_reference(
            tuple(
                sum(
                    coefficient * value
                    for coefficient, value in zip(
                        _batch_coefficients(
                            transcript, b"trace-column", index, _TRACE_COLUMN_IDS_V3
                        ),
                        row,
                        strict=True,
                    )
                )
                % GOLDILOCKS_MODULUS
                for row in trace_tree.rows
            ),
            statement=fri_statement,
        )
        for index, fri_statement in enumerate(trace_statements)
    )
    domain = goldilocks_radix2_domain_reference(
        size=statement.lde_domain_size,
        shift=statement.lde_shift,
    )
    denominators = tuple(
        goldilocks_inv(
            (pow(point, statement.trace_domain_size, GOLDILOCKS_MODULUS) - 1)
            % GOLDILOCKS_MODULUS
        )
        for point in domain.points()
    )
    blowup = GOLDILOCKS_FOLDED_LINEAR_LDE_BLOWUP_V3
    composition_proofs = []
    for index, fri_statement in enumerate(composition_statements):
        coefficients = _batch_coefficients(
            transcript, b"folded-constraint", index, _CONSTRAINT_IDS_V3
        )
        values = []
        for row_index, row in enumerate(trace_tree.rows):
            next_index = (row_index + blowup) % statement.lde_domain_size
            constraint_values = _constraint_values(
                row=row,
                next_row=trace_tree.rows[next_index],
                public={
                    "coeff": public["coeff"][row_index],
                    "coeff_next": public["coeff"][next_index],
                    "seg_first": public["seg_first"][row_index],
                    "seg_cont": public["seg_cont"][row_index],
                    "x_last": public["x_last"][row_index],
                    "y_last": public["y_last"][row_index],
                    "padding": public["padding"][row_index],
                },
                claimed_total=claimed_total,
            )
            total = 0
            for coefficient, value in zip(coefficients, constraint_values, strict=True):
                total = (total + coefficient * value) % GOLDILOCKS_MODULUS
            values.append(total * denominators[row_index] % GOLDILOCKS_MODULUS)
        composition_proofs.append(
            prove_goldilocks_fri_reference(tuple(values), statement=fri_statement)
        )
    indices = _consistency_indices(
        statement=statement,
        trace_proofs=trace_proofs,
        trace_statements=trace_statements,
        composition_proofs=tuple(composition_proofs),
        composition_statements=composition_statements,
    )
    proof = GoldilocksFoldedLinearProofV3(
        claimed_total=claimed_total,
        trace_batch_fri_proofs=trace_proofs,
        composition_fri_proofs=tuple(composition_proofs),
        trace_consistency_opening=trace_tree.open(indices),
        source_oracle_opening=source_tree.open(indices),
    )
    return precommitment, proof


def _postclaim_digest(
    *,
    precommitment: GoldilocksFoldedLinearPrecommitmentV3,
    validator_nonce: bytes,
    claimed_total: int,
) -> bytes:
    if not 0 <= claimed_total < GOLDILOCKS_MODULUS:
        raise ProofV3Error("folded-linear claimed total is not canonical")
    return hashlib.sha256(
        _POSTCLAIM_DOMAIN
        + precommitment.digest()
        + _fixed32(validator_nonce, "folded-linear validator_nonce")
        + claimed_total.to_bytes(8, "little")
    ).digest()


def _batch_coefficients(
    transcript: bytes,
    label: bytes,
    batch_index: int,
    coordinate_ids: tuple[str, ...],
) -> tuple[int, ...]:
    return tuple(
        _derive_nonzero_field(
            transcript_digest=transcript,
            label=label,
            batch_index=batch_index,
            coordinate_index=index,
            coordinate_id=coordinate_id,
        )
        for index, coordinate_id in enumerate(coordinate_ids)
    )


def _fri_statements(
    *,
    statement: GoldilocksFoldedLinearStatementV3,
    transcript: bytes,
) -> tuple[
    tuple[GoldilocksFriStatementReference, ...],
    tuple[GoldilocksFriStatementReference, ...],
]:
    trace_statements = []
    composition_statements = []
    for index in range(GOLDILOCKS_FOLDED_LINEAR_BATCH_COUNT_V3):
        trace_statements.append(
            GoldilocksFriStatementReference(
                binding_digest=hashlib.sha256(
                    _TRACE_FRI_DOMAIN + transcript + struct.pack("<I", index)
                ).digest(),
                domain_size=statement.lde_domain_size,
                degree_bound=statement.trace_domain_size,
                domain_shift=statement.lde_shift,
                query_count=statement.query_count,
            )
        )
        composition_statements.append(
            GoldilocksFriStatementReference(
                binding_digest=hashlib.sha256(
                    _COMPOSITION_FRI_DOMAIN + transcript + struct.pack("<I", index)
                ).digest(),
                domain_size=statement.lde_domain_size,
                degree_bound=statement.composition_degree_bound,
                domain_shift=statement.lde_shift,
                query_count=statement.query_count,
            )
        )
    return tuple(trace_statements), tuple(composition_statements)


def _scalar_source_opening(
    proof: GoldilocksFriProofReference,
    *,
    statement: GoldilocksFriStatementReference,
) -> GoldilocksMerkleMultiOpeningReference:
    if statement.round_count:
        return proof.round_openings[0]
    return proof.final_opening


def _consistency_indices(
    *,
    statement: GoldilocksFoldedLinearStatementV3,
    trace_proofs: tuple[GoldilocksFriProofReference, ...],
    trace_statements: tuple[GoldilocksFriStatementReference, ...],
    composition_proofs: tuple[GoldilocksFriProofReference, ...],
    composition_statements: tuple[GoldilocksFriStatementReference, ...],
) -> tuple[int, ...]:
    indices: set[int] = set()
    for proof, fri_statement in zip(trace_proofs, trace_statements, strict=True):
        indices.update(_scalar_source_opening(proof, statement=fri_statement).indices)
    for proof, fri_statement in zip(
        composition_proofs, composition_statements, strict=True
    ):
        source = _scalar_source_opening(proof, statement=fri_statement).indices
        indices.update(source)
        indices.update(
            (index + GOLDILOCKS_FOLDED_LINEAR_LDE_BLOWUP_V3)
            % statement.lde_domain_size
            for index in source
        )
    if not indices:
        raise ProofV3Error("folded-linear consistency opening has no indices")
    return tuple(sorted(indices))


@dataclass(frozen=True, slots=True)
class GoldilocksFoldedLinearProofV3:
    claimed_total: int
    trace_batch_fri_proofs: tuple[GoldilocksFriProofReference, ...]
    composition_fri_proofs: tuple[GoldilocksFriProofReference, ...]
    trace_consistency_opening: GoldilocksMerkleMultiOpeningReference
    source_oracle_opening: GoldilocksMerkleMultiOpeningReference

    def __post_init__(self) -> None:
        total = _integer(self.claimed_total, "claimed_total")
        if not 0 <= total < GOLDILOCKS_MODULUS:
            raise ProofV3Error("folded-linear claimed total is not canonical")
        for name, proofs in (
            ("trace", self.trace_batch_fri_proofs),
            ("composition", self.composition_fri_proofs),
        ):
            if (
                not isinstance(proofs, tuple)
                or len(proofs) != GOLDILOCKS_FOLDED_LINEAR_BATCH_COUNT_V3
                or not all(
                    isinstance(item, GoldilocksFriProofReference) for item in proofs
                )
            ):
                raise ProofV3Error(f"folded-linear {name} FRI proof set is malformed")


def verify_goldilocks_folded_linear_reference_v3(
    proof: object,
    *,
    statement: GoldilocksFoldedLinearStatementV3,
    precommitment: GoldilocksFoldedLinearPrecommitmentV3,
    validator_nonce: bytes,
) -> None:
    """Verify one folded global linear-relation proof.

    The verifier recomputes every fold coefficient, the folded weight vector
    ``W @ u`` from the signed weights it owns, all public selector and
    coefficient LDE values, and both scan boundary constraints against the
    prover's claimed total.  Both segments ending at the same total is the
    folded identity ``v^T X (W u) == v^T Y u``.
    """

    try:
        if not isinstance(statement, GoldilocksFoldedLinearStatementV3):
            raise ProofV3VerificationError("folded-linear statement is malformed")
        if not isinstance(precommitment, GoldilocksFoldedLinearPrecommitmentV3):
            raise ProofV3VerificationError("folded-linear precommitment is malformed")
        if precommitment.statement_digest != statement.digest():
            raise ProofV3VerificationError(
                "folded-linear precommitment belongs to a different statement"
            )
        if not isinstance(proof, GoldilocksFoldedLinearProofV3):
            raise ProofV3VerificationError("folded-linear proof type is unexpected")
        interim = GoldilocksFoldedLinearPrecommitmentV3(
            statement_digest=precommitment.statement_digest,
            trace_lde_commitment=bytes(32),
            source_oracle_commitment=precommitment.source_oracle_commitment,
        )
        v, u = _fold_coefficients(
            statement=statement,
            precommitment=interim,
            validator_nonce=validator_nonce,
        )
        public = _public_columns(statement=statement, v=v, u=u)
        transcript = _postclaim_digest(
            precommitment=precommitment,
            validator_nonce=validator_nonce,
            claimed_total=proof.claimed_total,
        )
        trace_statements, composition_statements = _fri_statements(
            statement=statement,
            transcript=transcript,
        )
        for fri_proof, fri_statement in zip(
            proof.trace_batch_fri_proofs, trace_statements, strict=True
        ):
            verify_goldilocks_fri_reference(fri_proof, statement=fri_statement)
        for fri_proof, fri_statement in zip(
            proof.composition_fri_proofs, composition_statements, strict=True
        ):
            verify_goldilocks_fri_reference(fri_proof, statement=fri_statement)
        indices = _consistency_indices(
            statement=statement,
            trace_proofs=proof.trace_batch_fri_proofs,
            trace_statements=trace_statements,
            composition_proofs=proof.composition_fri_proofs,
            composition_statements=composition_statements,
        )
        verify_goldilocks_merkle_multiopening_reference(
            precommitment.trace_lde_commitment,
            proof.trace_consistency_opening,
            expected_binding_digest=statement.trace_tree_binding_digest(),
            expected_leaf_count=statement.lde_domain_size,
            expected_leaf_width=len(_TRACE_COLUMN_IDS_V3),
            expected_indices=indices,
        )
        verify_goldilocks_merkle_multiopening_reference(
            precommitment.source_oracle_commitment,
            proof.source_oracle_opening,
            expected_binding_digest=statement.source_oracle_binding_digest(),
            expected_leaf_count=statement.lde_domain_size,
            expected_leaf_width=1,
            expected_indices=indices,
        )
        trace_rows = {
            index: row
            for index, row in zip(
                proof.trace_consistency_opening.indices,
                proof.trace_consistency_opening.rows,
                strict=True,
            )
        }
        source_cells = {
            index: row[0]
            for index, row in zip(
                proof.source_oracle_opening.indices,
                proof.source_oracle_opening.rows,
                strict=True,
            )
        }
        for index, row in trace_rows.items():
            if row[0] != source_cells[index]:
                raise ProofV3VerificationError(
                    "folded-linear trace is not bound to the frozen source oracle"
                )
        for batch_index, (fri_proof, fri_statement) in enumerate(
            zip(proof.trace_batch_fri_proofs, trace_statements, strict=True)
        ):
            opened = _scalar_source_opening(fri_proof, statement=fri_statement)
            coefficients = _batch_coefficients(
                transcript, b"trace-column", batch_index, _TRACE_COLUMN_IDS_V3
            )
            for index, value_row in zip(opened.indices, opened.rows, strict=True):
                row = trace_rows.get(index)
                if row is None:
                    raise ProofV3VerificationError(
                        "folded-linear trace opening omits a trace-batch row"
                    )
                expected = sum(
                    coefficient * element
                    for coefficient, element in zip(coefficients, row, strict=True)
                ) % GOLDILOCKS_MODULUS
                if value_row[0] != expected:
                    raise ProofV3VerificationError(
                        "folded-linear trace FRI is not bound to the frozen trace"
                    )
        domain = goldilocks_radix2_domain_reference(
            size=statement.lde_domain_size,
            shift=statement.lde_shift,
        )
        points = domain.points()
        blowup = GOLDILOCKS_FOLDED_LINEAR_LDE_BLOWUP_V3
        for batch_index, (fri_proof, fri_statement) in enumerate(
            zip(proof.composition_fri_proofs, composition_statements, strict=True)
        ):
            opened = _scalar_source_opening(fri_proof, statement=fri_statement)
            coefficients = _batch_coefficients(
                transcript, b"folded-constraint", batch_index, _CONSTRAINT_IDS_V3
            )
            for index, value_row in zip(opened.indices, opened.rows, strict=True):
                row = trace_rows.get(index)
                next_index = (index + blowup) % statement.lde_domain_size
                next_row = trace_rows.get(next_index)
                if row is None or next_row is None:
                    raise ProofV3VerificationError(
                        "folded-linear trace opening omits a composition row"
                    )
                constraint_values = _constraint_values(
                    row=row,
                    next_row=next_row,
                    public={
                        "coeff": public["coeff"][index],
                        "coeff_next": public["coeff"][next_index],
                        "seg_first": public["seg_first"][index],
                        "seg_cont": public["seg_cont"][index],
                        "x_last": public["x_last"][index],
                        "y_last": public["y_last"][index],
                        "padding": public["padding"][index],
                    },
                    claimed_total=proof.claimed_total,
                )
                total = 0
                for coefficient, value in zip(
                    coefficients, constraint_values, strict=True
                ):
                    total = (total + coefficient * value) % GOLDILOCKS_MODULUS
                denominator = goldilocks_inv(
                    (
                        pow(points[index], statement.trace_domain_size, GOLDILOCKS_MODULUS)
                        - 1
                    )
                    % GOLDILOCKS_MODULUS
                )
                if value_row[0] != total * denominator % GOLDILOCKS_MODULUS:
                    raise ProofV3VerificationError(
                        "folded-linear composition is not bound to the frozen trace"
                    )
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("folded-linear proof is malformed") from exc


__all__ = [
    "GOLDILOCKS_FOLDED_LINEAR_ABI_V3",
    "GoldilocksFoldedLinearPrecommitmentV3",
    "GoldilocksFoldedLinearProofV3",
    "GoldilocksFoldedLinearStatementV3",
    "freeze_goldilocks_folded_linear_sources_v3",
    "prove_goldilocks_folded_linear_reference_v3",
    "verify_goldilocks_folded_linear_reference_v3",
]
