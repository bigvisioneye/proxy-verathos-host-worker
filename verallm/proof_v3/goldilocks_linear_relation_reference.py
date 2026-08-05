"""All-row linear-relation reference prover/verifier for proof-v3.

This module is deliberately unregistered.  It is the first CPU reference in
the tree that proves a complete execution relation over *every* row of one
linear operation: frozen runtime activations ``X``, validator-signed weights
``W``, and frozen outputs ``Y`` must satisfy ``X @ W == Y`` over the whole
statement, not over sampled tiles.

Protocol shape (all primitives reused from the existing references):

* the prover lays out one accumulation segment of ``K`` rows per
  ``(token, output_feature)`` pair, with trace columns ``x``, ``w``, ``y``
  and a running accumulator ``acc``;
* the trace LDE tree plus two independent width-1 runtime oracles (the
  ``x`` and ``y`` layouts, stand-ins for the accumulator chain's frozen
  runtime-state segment commitments) are frozen before the validator nonce;
* after the nonce, independent trace batches and constraint-composition
  batches are low-degree tested with the existing FRI reference, and exact
  Merkle openings tie every queried oracle cell back to the frozen trees;
* the verifier recomputes the expected weight column itself from the signed
  weights it owns and compares it against the opened trace cells at every
  query index, and cross-checks the ``x``/``y`` trace cells against the
  independently frozen runtime oracles at the same indices.

Two committed degree-bounded columns that agree at every uniformly queried
LDE index are identical except with probability at most
``(1 / lde_blowup) ** query_count``; this is the same binding posture as the
existing trace-batch consistency check.

Integer semantics: inputs are validated as signed int8 (the byte authority
for runtime captures and static weights), so every product is bounded by
``2**14`` and every accumulator by ``K * 2**14 << p / 2``.  The Goldilocks
field identity proven here is therefore an exact integer identity; no range
decomposition is required for the linear shell.

This reference is not a production backend: it retains full trees in memory,
recomputes selector LDEs naively, and binds the runtime oracles to the
statement rather than to the accumulator chain.  Composing those oracles
with the frozen trace-map/tensor-binding chain, and replacing the in-memory
trees with native streaming oracles, is the native backend's job.
"""

from __future__ import annotations

import hashlib
import operator
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
from verallm.proof_v3.goldilocks_merkle_reference import (
    GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES,
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


GOLDILOCKS_LINEAR_RELATION_ABI_V3: Final = "goldilocks.linear_relation.reference.v1"
GOLDILOCKS_LINEAR_RELATION_FORMAT_VERSION_V3: Final = 1
GOLDILOCKS_LINEAR_RELATION_LDE_BLOWUP_V3: Final = 4
GOLDILOCKS_LINEAR_RELATION_QUERY_COUNT_V3: Final = 16
GOLDILOCKS_LINEAR_RELATION_TRACE_BATCH_COUNT_V3: Final = 2
GOLDILOCKS_LINEAR_RELATION_COMPOSITION_BATCH_COUNT_V3: Final = 2
MAX_GOLDILOCKS_LINEAR_RELATION_REJECTION_ATTEMPTS_V3: Final = 1 << 16

_TRACE_COLUMN_IDS_V3: Final = ("x", "w", "y", "acc")
_CONSTRAINT_IDS_V3: Final = (
    "segment_first",
    "segment_transition",
    "segment_last",
    "padding_x",
    "padding_y",
    "padding_acc",
    "padding_w",
)

_STATEMENT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_LINEAR/V1/STATEMENT/SHA256"
)
_SHIFT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_LINEAR/V1/LDE_SHIFT/SHA256"
_TRACE_TREE_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_LINEAR/V1/TRACE_TREE/SHA256"
)
_X_ORACLE_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_LINEAR/V1/X_ORACLE/SHA256"
_Y_ORACLE_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_LINEAR/V1/Y_ORACLE/SHA256"
_PRECOMMITMENT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_LINEAR/V1/PRECOMMITMENT/SHA256"
)
_POSTCOMMIT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_LINEAR/V1/POSTCOMMIT/SHA256"
)
_FIELD_CHALLENGE_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_LINEAR/V1/FIELD_CHALLENGE/SHA256"
)
_TRACE_FRI_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_LINEAR/V1/TRACE_FRI/SHA256"
_COMPOSITION_FRI_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_LINEAR/V1/COMPOSITION_FRI/SHA256"
)


_ZERO32: Final = bytes(GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES)


def _fixed32(value: object, name: str, *, nonzero: bool = False) -> bytes:
    # fast path: exact 32-byte ``bytes`` is returned without a defensive
    # copy (millions of digests per verify are already immutable bytes)
    if type(value) is bytes:
        if len(value) != GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES:
            raise ProofV3Error(f"{name} must be exactly 32 bytes")
        if nonzero and value == _ZERO32:
            raise ProofV3Error(f"{name} must not be zero")
        return value
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ProofV3Error(f"{name} must be bytes")
    result = bytes(value)
    if len(result) != GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    if nonzero and result == bytes(len(result)):
        raise ProofV3Error(f"{name} must not be zero")
    return result


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ProofV3Error(f"{name} must be an integer, not boolean")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ProofV3Error(f"{name} must be an integer") from exc


def _u32(value: object, name: str, *, positive: bool = False) -> int:
    integer = _integer(value, name)
    if integer < (1 if positive else 0) or integer >= 1 << 32:
        qualifier = "positive " if positive else ""
        raise ProofV3Error(f"{name} must be a {qualifier}unsigned 32-bit integer")
    return integer


def _int8(value: object, name: str) -> int:
    integer = _integer(value, name)
    if integer < -128 or integer > 127:
        raise ProofV3Error(f"{name} must be a signed 8-bit integer")
    return integer


def _field_encode_signed(value: int) -> int:
    return value % GOLDILOCKS_MODULUS


def _next_power_of_two(value: int, *, name: str) -> int:
    if value < 1:
        raise ProofV3Error(f"{name} must be positive")
    return 1 << (value - 1).bit_length()


def _derive_lde_shift(*, statement_digest: bytes, lde_domain_size: int) -> int:
    prefix = _SHIFT_DOMAIN + statement_digest + struct.pack("<Q", lde_domain_size)
    for counter in range(MAX_GOLDILOCKS_LINEAR_RELATION_REJECTION_ATTEMPTS_V3):
        candidate = int.from_bytes(
            hashlib.sha256(prefix + struct.pack("<I", counter)).digest()[:8],
            "little",
        )
        if (
            0 < candidate < GOLDILOCKS_MODULUS
            and pow(candidate, lde_domain_size, GOLDILOCKS_MODULUS) != 1
        ):
            return candidate
    raise ProofV3Error("unable to derive a safe linear-relation LDE coset shift")


def _derive_nonzero_field(
    *,
    transcript_digest: bytes,
    label: bytes,
    batch_index: int,
    coordinate_index: int,
    coordinate_id: str,
) -> int:
    identifier = coordinate_id.encode("ascii")
    prefix = (
        _FIELD_CHALLENGE_DOMAIN
        + _fixed32(transcript_digest, "linear-relation transcript digest", nonzero=True)
        + struct.pack("<BII", len(label), batch_index, coordinate_index)
        + label
        + struct.pack("<B", len(identifier))
        + identifier
    )
    for counter in range(MAX_GOLDILOCKS_LINEAR_RELATION_REJECTION_ATTEMPTS_V3):
        candidate = int.from_bytes(
            hashlib.sha256(prefix + struct.pack("<I", counter)).digest()[:8],
            "little",
        )
        if 0 < candidate < GOLDILOCKS_MODULUS:
            return candidate
    raise ProofV3Error("unable to derive a linear-relation challenge")


@dataclass(frozen=True, slots=True)
class GoldilocksLinearRelationStatementV3:
    """Validator-owned statement: shape, signed weights, and request binding.

    ``validator_binding_digest`` is opaque here; a real verifier derives it
    from the signed profile, layout coordinate, and sealed request context.
    ``weights`` is the exact signed int8 weight matrix in
    ``(contraction, output)`` row-major order — the verifier owns these bytes
    through the signed static artifact and never learns them from the prover.
    """

    validator_binding_digest: bytes
    token_count: int
    contraction_length: int
    output_features: int
    weights: tuple[int, ...]
    abi_id: str = GOLDILOCKS_LINEAR_RELATION_ABI_V3
    format_version: int = GOLDILOCKS_LINEAR_RELATION_FORMAT_VERSION_V3
    active_row_count: int = field(init=False)
    trace_domain_size: int = field(init=False)
    lde_domain_size: int = field(init=False)
    lde_shift: int = field(init=False)
    query_count: int = field(init=False)
    composition_degree_bound: int = field(init=False)

    def __post_init__(self) -> None:
        if self.abi_id != GOLDILOCKS_LINEAR_RELATION_ABI_V3:
            raise ProofV3Error("linear-relation ABI is unsupported")
        if type(self.format_version) is not int or self.format_version != (
            GOLDILOCKS_LINEAR_RELATION_FORMAT_VERSION_V3
        ):
            raise ProofV3Error("linear-relation format version is unsupported")
        binding = _fixed32(
            self.validator_binding_digest,
            "linear-relation validator_binding_digest",
            nonzero=True,
        )
        token_count = _u32(self.token_count, "token_count", positive=True)
        contraction = _u32(self.contraction_length, "contraction_length", positive=True)
        outputs = _u32(self.output_features, "output_features", positive=True)
        if not isinstance(self.weights, tuple):
            raise ProofV3Error("linear-relation weights must be a tuple")
        if len(self.weights) != contraction * outputs:
            raise ProofV3Error("linear-relation weight count does not match the shape")
        weights = tuple(
            _int8(value, f"weights[{index}]")
            for index, value in enumerate(self.weights)
        )
        active_row_count = token_count * outputs * contraction
        trace_domain_size = _next_power_of_two(
            active_row_count,
            name="linear-relation active row count",
        )
        # The transition constraint reads next-row cells, so the last active
        # row must have an in-domain successor after padding.
        if trace_domain_size == active_row_count:
            trace_domain_size *= 2
        lde_domain_size = trace_domain_size * GOLDILOCKS_LINEAR_RELATION_LDE_BLOWUP_V3
        if lde_domain_size > MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE:
            raise ProofV3Error("linear-relation trace exceeds the CPU reference cap")
        query_count = min(
            GOLDILOCKS_LINEAR_RELATION_QUERY_COUNT_V3,
            lde_domain_size // 2,
        )
        if query_count < 1:
            raise ProofV3Error("linear-relation LDE domain is too small for FRI")
        # Every constraint is selector (degree < N) times an expression of
        # degree at most 2 in the trace columns (degree < N each), so the
        # numerator degree is at most 3 * (N - 1); dividing by X**N - 1
        # bounds the quotient below.  Kept as a power of two and floored for
        # the full final codeword opening of the standalone FRI reference.
        maximum_quotient_degree = max(0, 3 * (trace_domain_size - 1) - trace_domain_size)
        bound = 1
        while bound <= maximum_quotient_degree:
            bound <<= 1
        composition_degree_bound = max(bound, max(1, lde_domain_size // 64))
        if composition_degree_bound >= lde_domain_size:
            raise ProofV3Error(
                "linear-relation quotient degree does not fit the LDE domain"
            )
        object.__setattr__(self, "validator_binding_digest", binding)
        object.__setattr__(self, "token_count", token_count)
        object.__setattr__(self, "contraction_length", contraction)
        object.__setattr__(self, "output_features", outputs)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "active_row_count", active_row_count)
        object.__setattr__(self, "trace_domain_size", trace_domain_size)
        object.__setattr__(self, "lde_domain_size", lde_domain_size)
        object.__setattr__(
            self,
            "lde_shift",
            _derive_lde_shift(
                statement_digest=self._pre_shift_digest(binding=binding),
                lde_domain_size=lde_domain_size,
            ),
        )
        object.__setattr__(self, "query_count", query_count)
        object.__setattr__(
            self,
            "composition_degree_bound",
            composition_degree_bound,
        )

    def _pre_shift_digest(self, *, binding: bytes) -> bytes:
        payload = (
            binding
            + struct.pack(
                "<III",
                self.token_count,
                self.contraction_length,
                self.output_features,
            )
            + b"".join(
                struct.pack("<b", weight) for weight in self.weights
            )
        )
        return hashlib.sha256(_STATEMENT_DOMAIN + payload).digest()

    def canonical_bytes(self) -> bytes:
        abi = self.abi_id.encode("ascii")
        return (
            struct.pack("<HH", self.format_version, len(abi))
            + abi
            + self.validator_binding_digest
            + struct.pack(
                "<IIIQQQI",
                self.token_count,
                self.contraction_length,
                self.output_features,
                self.trace_domain_size,
                self.lde_domain_size,
                self.lde_shift,
                self.query_count,
            )
            + b"".join(struct.pack("<b", weight) for weight in self.weights)
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_STATEMENT_DOMAIN + self.canonical_bytes()).digest()

    def trace_tree_binding_digest(self) -> bytes:
        return hashlib.sha256(_TRACE_TREE_DOMAIN + self.digest()).digest()

    def x_oracle_binding_digest(self) -> bytes:
        return hashlib.sha256(_X_ORACLE_DOMAIN + self.digest()).digest()

    def y_oracle_binding_digest(self) -> bytes:
        return hashlib.sha256(_Y_ORACLE_DOMAIN + self.digest()).digest()

    def segment_coordinates(self, row_index: int) -> tuple[int, int, int]:
        """Return ``(token, output_feature, contraction_step)`` for one row."""

        contraction_step = row_index % self.contraction_length
        segment = row_index // self.contraction_length
        return (
            segment // self.output_features,
            segment % self.output_features,
            contraction_step,
        )


def _selector_bases(
    statement: GoldilocksLinearRelationStatementV3,
) -> dict[str, tuple[int, ...]]:
    contraction = statement.contraction_length
    active_rows = statement.active_row_count
    size = statement.trace_domain_size
    seg_first = tuple(
        1 if row < active_rows and row % contraction == 0 else 0
        for row in range(size)
    )
    seg_last = tuple(
        1 if row < active_rows and row % contraction == contraction - 1 else 0
        for row in range(size)
    )
    seg_cont = tuple(
        1
        if row + 1 < active_rows and row % contraction != contraction - 1
        else 0
        for row in range(size)
    )
    padding = tuple(0 if row < active_rows else 1 for row in range(size))
    return {
        "seg_first": seg_first,
        "seg_cont": seg_cont,
        "seg_last": seg_last,
        "padding": padding,
    }


def _selector_ldes(
    statement: GoldilocksLinearRelationStatementV3,
) -> dict[str, tuple[int, ...]]:
    return {
        name: lde_goldilocks_reference(
            base,
            target_size=statement.lde_domain_size,
            source_shift=1,
            target_shift=statement.lde_shift,
        )
        for name, base in _selector_bases(statement).items()
    }


def _expected_weight_column_lde(
    statement: GoldilocksLinearRelationStatementV3,
) -> tuple[int, ...]:
    """Recompute the exact weight-column LDE from the signed weights."""

    outputs = statement.output_features
    base: list[int] = []
    for row in range(statement.trace_domain_size):
        if row < statement.active_row_count:
            _token, feature, step = statement.segment_coordinates(row)
            base.append(
                _field_encode_signed(statement.weights[step * outputs + feature])
            )
        else:
            base.append(0)
    return lde_goldilocks_reference(
        tuple(base),
        target_size=statement.lde_domain_size,
        source_shift=1,
        target_shift=statement.lde_shift,
    )


def _constraint_values(
    *,
    statement: GoldilocksLinearRelationStatementV3,
    row: tuple[int, ...],
    next_row: tuple[int, ...],
    selectors: dict[str, int],
) -> tuple[int, ...]:
    """Evaluate every atomic constraint numerator at one (LDE) position."""

    x_value, w_value, y_value, acc_value = row
    x_next, w_next, _y_next, acc_next = next_row
    modulus = GOLDILOCKS_MODULUS
    return (
        selectors["seg_first"] * (acc_value - x_value * w_value) % modulus,
        selectors["seg_cont"]
        * (acc_next - acc_value - x_next * w_next)
        % modulus,
        selectors["seg_last"] * (acc_value - y_value) % modulus,
        selectors["padding"] * x_value % modulus,
        selectors["padding"] * y_value % modulus,
        selectors["padding"] * acc_value % modulus,
        selectors["padding"] * w_value % modulus,
    )


@dataclass(frozen=True, slots=True)
class GoldilocksLinearRelationPrecommitmentV3:
    """Frozen trace and runtime-oracle roots for one statement."""

    statement_digest: bytes
    trace_lde_commitment: bytes
    x_oracle_commitment: bytes
    y_oracle_commitment: bytes

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
            "x_oracle_commitment",
            _fixed32(self.x_oracle_commitment, "x_oracle_commitment"),
        )
        object.__setattr__(
            self,
            "y_oracle_commitment",
            _fixed32(self.y_oracle_commitment, "y_oracle_commitment"),
        )

    def digest(self) -> bytes:
        return hashlib.sha256(
            _PRECOMMITMENT_DOMAIN
            + self.statement_digest
            + self.trace_lde_commitment
            + self.x_oracle_commitment
            + self.y_oracle_commitment
        ).digest()


@dataclass(frozen=True, slots=True)
class GoldilocksLinearRelationOracleV3:
    """In-memory frozen trees retained only for the CPU reference prover."""

    statement: GoldilocksLinearRelationStatementV3
    precommitment: GoldilocksLinearRelationPrecommitmentV3
    trace_tree: GoldilocksMerkleTreeReference
    x_tree: GoldilocksMerkleTreeReference
    y_tree: GoldilocksMerkleTreeReference

    def __post_init__(self) -> None:
        if not isinstance(self.statement, GoldilocksLinearRelationStatementV3):
            raise ProofV3Error("linear-relation oracle statement is malformed")
        if self.precommitment.statement_digest != self.statement.digest():
            raise ProofV3Error("linear-relation oracle statement digest mismatch")
        if self.trace_tree.commitment != self.precommitment.trace_lde_commitment:
            raise ProofV3Error("linear-relation trace tree root mismatch")
        if self.x_tree.commitment != self.precommitment.x_oracle_commitment:
            raise ProofV3Error("linear-relation x oracle root mismatch")
        if self.y_tree.commitment != self.precommitment.y_oracle_commitment:
            raise ProofV3Error("linear-relation y oracle root mismatch")


def build_goldilocks_linear_relation_witness_v3(
    *,
    statement: GoldilocksLinearRelationStatementV3,
    x_values: tuple[tuple[int, ...], ...],
    y_values: tuple[tuple[int, ...], ...],
) -> GoldilocksLinearRelationOracleV3:
    """LDE-extend and freeze the trace and runtime oracles for one witness.

    The helper validates shapes and int8/int32 encodings but deliberately
    does not check ``X @ W == Y``: an invalid witness must be rejected by
    proof verification, never silently repaired by prover-side tooling.
    """

    if not isinstance(statement, GoldilocksLinearRelationStatementV3):
        raise ProofV3Error("linear-relation statement has an unexpected type")
    tokens = statement.token_count
    contraction = statement.contraction_length
    outputs = statement.output_features
    if len(x_values) != tokens or any(
        len(row) != contraction for row in x_values
    ):
        raise ProofV3Error("linear-relation x witness has an unexpected shape")
    if len(y_values) != tokens or any(len(row) != outputs for row in y_values):
        raise ProofV3Error("linear-relation y witness has an unexpected shape")
    x_checked = tuple(
        tuple(_int8(value, "x witness value") for value in row) for row in x_values
    )
    bound = contraction * (1 << 14)
    y_checked: list[tuple[int, ...]] = []
    for row in y_values:
        checked_row = []
        for value in row:
            integer = _integer(value, "y witness value")
            if integer < -bound or integer > bound:
                raise ProofV3Error("y witness value is out of the accumulator range")
            checked_row.append(integer)
        y_checked.append(tuple(checked_row))

    base_rows: list[tuple[int, int, int, int]] = []
    acc = 0
    for row_index in range(statement.trace_domain_size):
        if row_index >= statement.active_row_count:
            base_rows.append((0, 0, 0, 0))
            continue
        token, feature, step = statement.segment_coordinates(row_index)
        x_cell = _field_encode_signed(x_checked[token][step])
        w_cell = _field_encode_signed(statement.weights[step * outputs + feature])
        y_cell = _field_encode_signed(y_checked[token][feature])
        product = x_cell * w_cell % GOLDILOCKS_MODULUS
        acc = product if step == 0 else (acc + product) % GOLDILOCKS_MODULUS
        base_rows.append((x_cell, w_cell, y_cell, acc))

    lde_columns = tuple(
        lde_goldilocks_reference(
            tuple(row[column_index] for row in base_rows),
            target_size=statement.lde_domain_size,
            source_shift=1,
            target_shift=statement.lde_shift,
        )
        for column_index in range(len(_TRACE_COLUMN_IDS_V3))
    )
    lde_rows = tuple(
        tuple(column[row_index] for column in lde_columns)
        for row_index in range(statement.lde_domain_size)
    )
    trace_tree = GoldilocksMerkleTreeReference.from_rows(
        lde_rows,
        binding_digest=statement.trace_tree_binding_digest(),
    )
    x_tree = GoldilocksMerkleTreeReference.from_rows(
        tuple((value,) for value in lde_columns[0]),
        binding_digest=statement.x_oracle_binding_digest(),
    )
    y_tree = GoldilocksMerkleTreeReference.from_rows(
        tuple((value,) for value in lde_columns[2]),
        binding_digest=statement.y_oracle_binding_digest(),
    )
    precommitment = GoldilocksLinearRelationPrecommitmentV3(
        statement_digest=statement.digest(),
        trace_lde_commitment=trace_tree.commitment,
        x_oracle_commitment=x_tree.commitment,
        y_oracle_commitment=y_tree.commitment,
    )
    return GoldilocksLinearRelationOracleV3(
        statement=statement,
        precommitment=precommitment,
        trace_tree=trace_tree,
        x_tree=x_tree,
        y_tree=y_tree,
    )


@dataclass(frozen=True, slots=True)
class GoldilocksLinearRelationTranscriptV3:
    """Post-nonce transcript over one frozen precommitment."""

    precommitment: GoldilocksLinearRelationPrecommitmentV3
    validator_nonce: bytes
    digest_value: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        nonce = _fixed32(self.validator_nonce, "linear-relation validator_nonce")
        digest = hashlib.sha256(
            _POSTCOMMIT_DOMAIN + self.precommitment.digest() + nonce
        ).digest()
        object.__setattr__(self, "validator_nonce", nonce)
        object.__setattr__(self, "digest_value", digest)

    def trace_batch_coefficients(self, *, batch_index: int) -> tuple[int, ...]:
        if (
            batch_index < 0
            or batch_index >= GOLDILOCKS_LINEAR_RELATION_TRACE_BATCH_COUNT_V3
        ):
            raise ProofV3Error("linear-relation trace batch index is out of range")
        return tuple(
            _derive_nonzero_field(
                transcript_digest=self.digest_value,
                label=b"trace-column",
                batch_index=batch_index,
                coordinate_index=index,
                coordinate_id=column_id,
            )
            for index, column_id in enumerate(_TRACE_COLUMN_IDS_V3)
        )

    def composition_coefficients(self, *, batch_index: int) -> tuple[int, ...]:
        if (
            batch_index < 0
            or batch_index >= GOLDILOCKS_LINEAR_RELATION_COMPOSITION_BATCH_COUNT_V3
        ):
            raise ProofV3Error(
                "linear-relation composition batch index is out of range"
            )
        return tuple(
            _derive_nonzero_field(
                transcript_digest=self.digest_value,
                label=b"linear-constraint",
                batch_index=batch_index,
                coordinate_index=index,
                coordinate_id=constraint_id,
            )
            for index, constraint_id in enumerate(_CONSTRAINT_IDS_V3)
        )

    def trace_fri_statement(
        self,
        *,
        statement: GoldilocksLinearRelationStatementV3,
        batch_index: int,
    ) -> GoldilocksFriStatementReference:
        coefficients = self.trace_batch_coefficients(batch_index=batch_index)
        binding = hashlib.sha256(
            _TRACE_FRI_DOMAIN
            + self.digest_value
            + struct.pack("<I", batch_index)
            + b"".join(value.to_bytes(8, "little") for value in coefficients)
        ).digest()
        return GoldilocksFriStatementReference(
            binding_digest=binding,
            domain_size=statement.lde_domain_size,
            degree_bound=statement.trace_domain_size,
            domain_shift=statement.lde_shift,
            query_count=statement.query_count,
        )

    def composition_fri_statement(
        self,
        *,
        statement: GoldilocksLinearRelationStatementV3,
        batch_index: int,
    ) -> GoldilocksFriStatementReference:
        coefficients = self.composition_coefficients(batch_index=batch_index)
        binding = hashlib.sha256(
            _COMPOSITION_FRI_DOMAIN
            + self.digest_value
            + struct.pack("<I", batch_index)
            + b"".join(value.to_bytes(8, "little") for value in coefficients)
        ).digest()
        return GoldilocksFriStatementReference(
            binding_digest=binding,
            domain_size=statement.lde_domain_size,
            degree_bound=statement.composition_degree_bound,
            domain_shift=statement.lde_shift,
            query_count=statement.query_count,
        )


@dataclass(frozen=True, slots=True)
class GoldilocksLinearRelationProofV3:
    """Post-nonce proof over one frozen linear-relation precommitment."""

    trace_batch_fri_proofs: tuple[GoldilocksFriProofReference, ...]
    composition_fri_proofs: tuple[GoldilocksFriProofReference, ...]
    trace_consistency_opening: GoldilocksMerkleMultiOpeningReference
    x_oracle_opening: GoldilocksMerkleMultiOpeningReference
    y_oracle_opening: GoldilocksMerkleMultiOpeningReference
    abi_id: str = GOLDILOCKS_LINEAR_RELATION_ABI_V3
    format_version: int = GOLDILOCKS_LINEAR_RELATION_FORMAT_VERSION_V3

    def __post_init__(self) -> None:
        if self.abi_id != GOLDILOCKS_LINEAR_RELATION_ABI_V3:
            raise ProofV3Error("linear-relation proof ABI is unsupported")
        if type(self.format_version) is not int or self.format_version != (
            GOLDILOCKS_LINEAR_RELATION_FORMAT_VERSION_V3
        ):
            raise ProofV3Error("linear-relation proof format version is unsupported")
        if (
            not isinstance(self.trace_batch_fri_proofs, tuple)
            or len(self.trace_batch_fri_proofs)
            != GOLDILOCKS_LINEAR_RELATION_TRACE_BATCH_COUNT_V3
            or not all(
                isinstance(item, GoldilocksFriProofReference)
                for item in self.trace_batch_fri_proofs
            )
        ):
            raise ProofV3Error("linear-relation trace FRI proof set is malformed")
        if (
            not isinstance(self.composition_fri_proofs, tuple)
            or len(self.composition_fri_proofs)
            != GOLDILOCKS_LINEAR_RELATION_COMPOSITION_BATCH_COUNT_V3
            or not all(
                isinstance(item, GoldilocksFriProofReference)
                for item in self.composition_fri_proofs
            )
        ):
            raise ProofV3Error(
                "linear-relation composition FRI proof set is malformed"
            )
        for name, opening in (
            ("trace", self.trace_consistency_opening),
            ("x oracle", self.x_oracle_opening),
            ("y oracle", self.y_oracle_opening),
        ):
            if not isinstance(opening, GoldilocksMerkleMultiOpeningReference):
                raise ProofV3Error(f"linear-relation {name} opening is malformed")


def _scalar_source_opening(
    proof: GoldilocksFriProofReference,
    *,
    statement: GoldilocksFriStatementReference,
) -> GoldilocksMerkleMultiOpeningReference:
    if statement.round_count:
        return proof.round_openings[0]
    return proof.final_opening


def _scalar_opened_values(
    opening: GoldilocksMerkleMultiOpeningReference,
) -> dict[int, int]:
    if opening.leaf_width != 1:
        raise ProofV3VerificationError(
            "linear-relation FRI opening has unexpected width"
        )
    return {
        index: row[0]
        for index, row in zip(opening.indices, opening.rows, strict=True)
    }


def _consistency_indices(
    *,
    statement: GoldilocksLinearRelationStatementV3,
    trace_proofs: tuple[GoldilocksFriProofReference, ...],
    trace_statements: tuple[GoldilocksFriStatementReference, ...],
    composition_proofs: tuple[GoldilocksFriProofReference, ...],
    composition_statements: tuple[GoldilocksFriStatementReference, ...],
) -> tuple[int, ...]:
    indices: set[int] = set()
    for proof, fri_statement in zip(trace_proofs, trace_statements, strict=True):
        indices.update(
            _scalar_source_opening(proof, statement=fri_statement).indices
        )
    for proof, fri_statement in zip(
        composition_proofs,
        composition_statements,
        strict=True,
    ):
        source_indices = _scalar_source_opening(
            proof,
            statement=fri_statement,
        ).indices
        indices.update(source_indices)
        indices.update(
            (index + GOLDILOCKS_LINEAR_RELATION_LDE_BLOWUP_V3)
            % statement.lde_domain_size
            for index in source_indices
        )
    if not indices:
        raise ProofV3Error("linear-relation consistency opening has no indices")
    return tuple(sorted(indices))


def _composition_evaluations(
    *,
    statement: GoldilocksLinearRelationStatementV3,
    trace_rows: tuple[tuple[int, ...], ...],
    coefficients: tuple[int, ...],
) -> tuple[int, ...]:
    selectors = _selector_ldes(statement)
    domain = goldilocks_radix2_domain_reference(
        size=statement.lde_domain_size,
        shift=statement.lde_shift,
    )
    denominator_inverses = tuple(
        goldilocks_inv(
            (pow(point, statement.trace_domain_size, GOLDILOCKS_MODULUS) - 1)
            % GOLDILOCKS_MODULUS
        )
        for point in domain.points()
    )
    result: list[int] = []
    for row_index, row in enumerate(trace_rows):
        next_row = trace_rows[
            (row_index + GOLDILOCKS_LINEAR_RELATION_LDE_BLOWUP_V3)
            % statement.lde_domain_size
        ]
        values = _constraint_values(
            statement=statement,
            row=row,
            next_row=next_row,
            selectors={
                name: values[row_index] for name, values in selectors.items()
            },
        )
        total = 0
        for coefficient, value in zip(coefficients, values, strict=True):
            total = (total + coefficient * value) % GOLDILOCKS_MODULUS
        result.append(total * denominator_inverses[row_index] % GOLDILOCKS_MODULUS)
    return tuple(result)


def prove_goldilocks_linear_relation_reference_v3(
    *,
    oracle: GoldilocksLinearRelationOracleV3,
    validator_nonce: bytes,
) -> GoldilocksLinearRelationProofV3:
    """Build one post-nonce all-row linear-relation proof."""

    if not isinstance(oracle, GoldilocksLinearRelationOracleV3):
        raise ProofV3Error("linear-relation oracle has an unexpected type")
    statement = oracle.statement
    transcript = GoldilocksLinearRelationTranscriptV3(
        precommitment=oracle.precommitment,
        validator_nonce=validator_nonce,
    )
    trace_statements = tuple(
        transcript.trace_fri_statement(statement=statement, batch_index=index)
        for index in range(GOLDILOCKS_LINEAR_RELATION_TRACE_BATCH_COUNT_V3)
    )
    trace_proofs = tuple(
        prove_goldilocks_fri_reference(
            tuple(
                sum(
                    coefficient * value
                    for coefficient, value in zip(
                        transcript.trace_batch_coefficients(batch_index=index),
                        row,
                        strict=True,
                    )
                )
                % GOLDILOCKS_MODULUS
                for row in oracle.trace_tree.rows
            ),
            statement=fri_statement,
        )
        for index, fri_statement in enumerate(trace_statements)
    )
    composition_statements = tuple(
        transcript.composition_fri_statement(statement=statement, batch_index=index)
        for index in range(GOLDILOCKS_LINEAR_RELATION_COMPOSITION_BATCH_COUNT_V3)
    )
    composition_proofs = tuple(
        prove_goldilocks_fri_reference(
            _composition_evaluations(
                statement=statement,
                trace_rows=oracle.trace_tree.rows,
                coefficients=transcript.composition_coefficients(batch_index=index),
            ),
            statement=fri_statement,
        )
        for index, fri_statement in enumerate(composition_statements)
    )
    indices = _consistency_indices(
        statement=statement,
        trace_proofs=trace_proofs,
        trace_statements=trace_statements,
        composition_proofs=composition_proofs,
        composition_statements=composition_statements,
    )
    return GoldilocksLinearRelationProofV3(
        trace_batch_fri_proofs=trace_proofs,
        composition_fri_proofs=composition_proofs,
        trace_consistency_opening=oracle.trace_tree.open(indices),
        x_oracle_opening=oracle.x_tree.open(indices),
        y_oracle_opening=oracle.y_tree.open(indices),
    )


def verify_goldilocks_linear_relation_reference_v3(
    proof: object,
    *,
    statement: GoldilocksLinearRelationStatementV3,
    precommitment: GoldilocksLinearRelationPrecommitmentV3,
    validator_nonce: bytes,
) -> None:
    """Verify one all-row linear-relation proof against the frozen roots.

    Every expectation is recomputed from the verifier-owned statement (which
    includes the signed weights), the frozen precommitment, and the nonce.
    Invalid or malformed data is always a proof failure.
    """

    try:
        if not isinstance(statement, GoldilocksLinearRelationStatementV3):
            raise ProofV3VerificationError("linear-relation statement is malformed")
        if not isinstance(precommitment, GoldilocksLinearRelationPrecommitmentV3):
            raise ProofV3VerificationError(
                "linear-relation precommitment is malformed"
            )
        if precommitment.statement_digest != statement.digest():
            raise ProofV3VerificationError(
                "linear-relation precommitment belongs to a different statement"
            )
        if not isinstance(proof, GoldilocksLinearRelationProofV3):
            raise ProofV3VerificationError(
                "linear-relation proof has an unexpected type"
            )
        transcript = GoldilocksLinearRelationTranscriptV3(
            precommitment=precommitment,
            validator_nonce=validator_nonce,
        )
        trace_statements = tuple(
            transcript.trace_fri_statement(statement=statement, batch_index=index)
            for index in range(GOLDILOCKS_LINEAR_RELATION_TRACE_BATCH_COUNT_V3)
        )
        composition_statements = tuple(
            transcript.composition_fri_statement(
                statement=statement,
                batch_index=index,
            )
            for index in range(
                GOLDILOCKS_LINEAR_RELATION_COMPOSITION_BATCH_COUNT_V3
            )
        )
        for fri_proof, fri_statement in zip(
            proof.trace_batch_fri_proofs,
            trace_statements,
            strict=True,
        ):
            verify_goldilocks_fri_reference(fri_proof, statement=fri_statement)
        for fri_proof, fri_statement in zip(
            proof.composition_fri_proofs,
            composition_statements,
            strict=True,
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
            precommitment.x_oracle_commitment,
            proof.x_oracle_opening,
            expected_binding_digest=statement.x_oracle_binding_digest(),
            expected_leaf_count=statement.lde_domain_size,
            expected_leaf_width=1,
            expected_indices=indices,
        )
        verify_goldilocks_merkle_multiopening_reference(
            precommitment.y_oracle_commitment,
            proof.y_oracle_opening,
            expected_binding_digest=statement.y_oracle_binding_digest(),
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
        x_cells = _scalar_opened_values(proof.x_oracle_opening)
        y_cells = _scalar_opened_values(proof.y_oracle_opening)
        expected_weights = _expected_weight_column_lde(statement)
        for index, row in trace_rows.items():
            if row[1] != expected_weights[index]:
                raise ProofV3VerificationError(
                    "linear-relation trace does not carry the signed weights"
                )
            if row[0] != x_cells[index]:
                raise ProofV3VerificationError(
                    "linear-relation trace is not bound to the frozen x oracle"
                )
            if row[2] != y_cells[index]:
                raise ProofV3VerificationError(
                    "linear-relation trace is not bound to the frozen y oracle"
                )
        for batch_index, (fri_proof, fri_statement) in enumerate(
            zip(proof.trace_batch_fri_proofs, trace_statements, strict=True)
        ):
            opened = _scalar_opened_values(
                _scalar_source_opening(fri_proof, statement=fri_statement)
            )
            coefficients = transcript.trace_batch_coefficients(
                batch_index=batch_index
            )
            for index, value in opened.items():
                row = trace_rows.get(index)
                if row is None:
                    raise ProofV3VerificationError(
                        "linear-relation trace opening omits a trace-batch row"
                    )
                expected = sum(
                    coefficient * element
                    for coefficient, element in zip(coefficients, row, strict=True)
                ) % GOLDILOCKS_MODULUS
                if value != expected:
                    raise ProofV3VerificationError(
                        "linear-relation trace FRI is not bound to the frozen trace"
                    )
        selectors = _selector_ldes(statement)
        domain = goldilocks_radix2_domain_reference(
            size=statement.lde_domain_size,
            shift=statement.lde_shift,
        )
        points = domain.points()
        for batch_index, (fri_proof, fri_statement) in enumerate(
            zip(
                proof.composition_fri_proofs,
                composition_statements,
                strict=True,
            )
        ):
            opened = _scalar_opened_values(
                _scalar_source_opening(fri_proof, statement=fri_statement)
            )
            coefficients = transcript.composition_coefficients(
                batch_index=batch_index
            )
            for index, value in opened.items():
                row = trace_rows.get(index)
                next_row = trace_rows.get(
                    (index + GOLDILOCKS_LINEAR_RELATION_LDE_BLOWUP_V3)
                    % statement.lde_domain_size
                )
                if row is None or next_row is None:
                    raise ProofV3VerificationError(
                        "linear-relation trace opening omits a composition row"
                    )
                constraint_values = _constraint_values(
                    statement=statement,
                    row=row,
                    next_row=next_row,
                    selectors={
                        name: values[index]
                        for name, values in selectors.items()
                    },
                )
                total = 0
                for coefficient, constraint_value in zip(
                    coefficients,
                    constraint_values,
                    strict=True,
                ):
                    total = (
                        total + coefficient * constraint_value
                    ) % GOLDILOCKS_MODULUS
                denominator_inverse = goldilocks_inv(
                    (
                        pow(
                            points[index],
                            statement.trace_domain_size,
                            GOLDILOCKS_MODULUS,
                        )
                        - 1
                    )
                    % GOLDILOCKS_MODULUS
                )
                if value != total * denominator_inverse % GOLDILOCKS_MODULUS:
                    raise ProofV3VerificationError(
                        "linear-relation composition is not bound to the frozen trace"
                    )
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "linear-relation proof is malformed"
        ) from exc


__all__ = [
    "GOLDILOCKS_LINEAR_RELATION_ABI_V3",
    "GOLDILOCKS_LINEAR_RELATION_FORMAT_VERSION_V3",
    "GOLDILOCKS_LINEAR_RELATION_LDE_BLOWUP_V3",
    "GOLDILOCKS_LINEAR_RELATION_QUERY_COUNT_V3",
    "GoldilocksLinearRelationOracleV3",
    "GoldilocksLinearRelationPrecommitmentV3",
    "GoldilocksLinearRelationProofV3",
    "GoldilocksLinearRelationStatementV3",
    "GoldilocksLinearRelationTranscriptV3",
    "build_goldilocks_linear_relation_witness_v3",
    "prove_goldilocks_linear_relation_reference_v3",
    "verify_goldilocks_linear_relation_reference_v3",
]
