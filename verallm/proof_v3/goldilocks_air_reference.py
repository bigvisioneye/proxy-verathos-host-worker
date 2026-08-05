"""Bounded two-oracle Goldilocks AIR/FRI reference for proof-v3.

This module is deliberately unregistered.  It is a CPU conformance reference
for a future native dynamic proof backend; no miner, validator, payload, or
profile eligibility path imports it.  In particular, it does not prove that a
trace came from vLLM, authenticates model execution, implements range/lookup
or RAM arguments, or establishes model-substitution resistance.

It does establish the first useful cryptographic vertical over the parsed
field-polynomial program ABI:

* a width-row trace LDE tree is frozen under a nonce-independent binding;
* after the validator nonce, independent trace batches are low-degree tested;
* independently batched scoped AIR quotients are low-degree tested; and
* exact Merkle openings tie every queried post-nonce oracle back to the
  frozen trace tree, including the canonical ``next``-row rotation.

The caller is responsible for the protocol chronology: the
``GoldilocksAirTracePrecommitmentReferenceV3`` must be sealed into a
validator-authenticated precommitment before the nonce is released.  This
small in-memory reference cannot enforce transport timing by itself.

Goldilocks is a 64-bit field.  The fixed pair of independently domain-separated
trace and composition batches is useful for regression coverage, but this
module makes no production soundness-bit claim.  A qualified production
backend needs a dedicated soundness review, native streaming oracles, actual
runtime witness binding, numeric range/carry/lookup constraints, cache RAM,
and recursive all-chunk aggregation.
"""

from __future__ import annotations

import hashlib
import operator
import struct
from dataclasses import dataclass, field
from typing import Final

from verallm.proof_v3.constraint_program import (
    GoldilocksConstraintProgramV3,
    GoldilocksConstraintTraceReferenceV3,
)
from verallm.proof_v3.constraint_system import (
    GOLDILOCKS_TRACE_DOMAIN_RULE_V3,
    GOLDILOCKS_TRACE_PADDING_RULE_V3,
)
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


GOLDILOCKS_AIR_REFERENCE_ABI_V3: Final = "goldilocks.air.reference.v1"
GOLDILOCKS_AIR_REFERENCE_FORMAT_VERSION_V3: Final = 1
GOLDILOCKS_AIR_REFERENCE_TRACE_BATCH_COUNT_V3: Final = 2
GOLDILOCKS_AIR_REFERENCE_COMPOSITION_BATCH_COUNT_V3: Final = 2
GOLDILOCKS_AIR_REFERENCE_QUERY_COUNT_V3: Final = 16
MAX_GOLDILOCKS_AIR_REFERENCE_REJECTION_ATTEMPTS_V3: Final = 1 << 16

_CORE_DIGEST_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR/V1/CORE_DIGEST/SHA256"
_SHIFT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR/V1/LDE_SHIFT/SHA256"
_TRACE_TREE_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR/V1/TRACE_TREE/SHA256"
_PRECOMMITMENT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR/V1/PRECOMMITMENT/SHA256"
)
_POSTCOMMIT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR/V1/POSTCOMMIT/SHA256"
_FIELD_CHALLENGE_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR/V1/FIELD_CHALLENGE/SHA256"
)
_TRACE_FRI_BINDING_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR/V1/TRACE_FRI/SHA256"
)
_COMPOSITION_FRI_BINDING_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR/V1/COMPOSITION_FRI/SHA256"
)


def _fixed32(value: object, name: str, *, nonzero: bool = False) -> bytes:
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


def _power_of_two_strictly_above(value: int, *, name: str) -> int:
    if value < 0:
        raise ProofV3Error(f"{name} must not be negative")
    result = 1
    while result <= value:
        result <<= 1
    return result


def _next_power_of_two(value: int, *, name: str) -> int:
    if value < 1:
        raise ProofV3Error(f"{name} must be positive")
    return 1 << (value - 1).bit_length()


def _derive_lde_shift(
    *,
    validator_binding_digest: bytes,
    program_digest: bytes,
    token_count: int,
    trace_domain_size: int,
    lde_domain_size: int,
    trace_width: int,
) -> int:
    """Derive a coset disjoint from the entire LDE root-of-unity subgroup."""

    prefix = (
        _SHIFT_DOMAIN
        + validator_binding_digest
        + program_digest
        + struct.pack(
            "<IQQI",
            token_count,
            trace_domain_size,
            lde_domain_size,
            trace_width,
        )
    )
    for counter in range(MAX_GOLDILOCKS_AIR_REFERENCE_REJECTION_ATTEMPTS_V3):
        candidate = int.from_bytes(
            hashlib.sha256(prefix + struct.pack("<I", counter)).digest()[:8],
            "little",
        )
        if (
            0 < candidate < GOLDILOCKS_MODULUS
            and pow(candidate, lde_domain_size, GOLDILOCKS_MODULUS) != 1
        ):
            return candidate
    raise ProofV3Error("unable to derive a safe Goldilocks AIR LDE coset shift")


def _derive_nonzero_field(
    *,
    transcript_digest: bytes,
    label: bytes,
    batch_index: int,
    coordinate_index: int,
    coordinate_id: str,
) -> int:
    if not isinstance(label, bytes) or not label:
        raise ProofV3Error("Goldilocks AIR challenge label is malformed")
    identifier = coordinate_id.encode("ascii")
    if len(identifier) > 255:
        raise ProofV3Error("Goldilocks AIR challenge identifier is too long")
    prefix = (
        _FIELD_CHALLENGE_DOMAIN
        + _fixed32(transcript_digest, "Goldilocks AIR transcript digest", nonzero=True)
        + struct.pack("<BII", len(label), batch_index, coordinate_index)
        + label
        + struct.pack("<B", len(identifier))
        + identifier
    )
    for counter in range(MAX_GOLDILOCKS_AIR_REFERENCE_REJECTION_ATTEMPTS_V3):
        candidate = int.from_bytes(
            hashlib.sha256(prefix + struct.pack("<I", counter)).digest()[:8],
            "little",
        )
        if 0 < candidate < GOLDILOCKS_MODULUS:
            return candidate
    raise ProofV3Error("unable to derive a canonical Goldilocks AIR challenge")


def _scope_rows(
    *,
    scope: str,
    active_row_count: int,
    trace_domain_size: int,
) -> range:
    if scope == "active_rows":
        return range(active_row_count)
    if scope == "first_active_row":
        return range(1)
    if scope == "last_active_row":
        return range(active_row_count - 1, active_row_count)
    if scope == "padding_rows":
        return range(active_row_count, trace_domain_size)
    if scope == "transition_rows":
        return range(active_row_count - 1)
    raise ProofV3Error("Goldilocks AIR constraint scope is unsupported")


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
        raise ProofV3VerificationError("Goldilocks AIR FRI opening has unexpected width")
    return {
        index: row[0]
        for index, row in zip(opening.indices, opening.rows, strict=True)
    }


@dataclass(frozen=True, slots=True)
class GoldilocksAirStatementCoreReferenceV3:
    """Validator-owned statement data derived before trace precommitment.

    ``validator_binding_digest`` is intentionally opaque to this generic
    reference.  A real verifier must derive it from the signed profile,
    relation/system/program bundle, exact chunk coordinate, and sealed request
    context; a miner must never select it.
    """

    validator_binding_digest: bytes
    program: GoldilocksConstraintProgramV3
    token_count: int
    abi_id: str = GOLDILOCKS_AIR_REFERENCE_ABI_V3
    format_version: int = GOLDILOCKS_AIR_REFERENCE_FORMAT_VERSION_V3
    program_digest: bytes = field(init=False, repr=False)
    active_row_count: int = field(init=False)
    trace_domain_size: int = field(init=False)
    lde_domain_size: int = field(init=False)
    lde_blowup: int = field(init=False)
    lde_shift: int = field(init=False)
    trace_width: int = field(init=False)
    trace_degree_bound: int = field(init=False)
    composition_degree_bound: int = field(init=False)
    query_count: int = field(init=False)

    def __post_init__(self) -> None:
        if self.abi_id != GOLDILOCKS_AIR_REFERENCE_ABI_V3:
            raise ProofV3Error("Goldilocks AIR reference ABI is unsupported")
        if type(self.format_version) is not int or self.format_version != (
            GOLDILOCKS_AIR_REFERENCE_FORMAT_VERSION_V3
        ):
            raise ProofV3Error("Goldilocks AIR reference format version is unsupported")
        binding = _fixed32(
            self.validator_binding_digest,
            "Goldilocks AIR validator_binding_digest",
            nonzero=True,
        )
        if not isinstance(self.program, GoldilocksConstraintProgramV3):
            raise ProofV3Error("Goldilocks AIR program has an unexpected type")
        if tuple(
            constraint.constraint_id for constraint in self.program.atomic_constraints
        ) != self.program.layout_binding.atomic_constraint_ids:
            raise ProofV3Error(
                "Goldilocks AIR program does not exactly cover its layout constraints"
            )
        if self.program.max_constraint_degree != (
            self.program.layout_binding.max_constraint_degree
        ):
            raise ProofV3Error("Goldilocks AIR program degree does not match its layout")
        if (
            self.program.layout_binding.trace_domain_rule_id
            != GOLDILOCKS_TRACE_DOMAIN_RULE_V3
            or self.program.layout_binding.padding_rule_id
            != GOLDILOCKS_TRACE_PADDING_RULE_V3
        ):
            raise ProofV3Error("Goldilocks AIR program has an unsupported layout rule")
        token_count = _u32(self.token_count, "Goldilocks AIR token_count", positive=True)
        active_row_count = token_count * self.program.layout_binding.rows_per_token
        trace_domain_size = max(
            self.program.layout_binding.minimum_trace_rows,
            _next_power_of_two(
                active_row_count,
                name="Goldilocks AIR active row count",
            ),
        )
        if trace_domain_size > MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE:
            raise ProofV3Error("Goldilocks AIR trace exceeds the CPU reference cap")
        lde_blowup = self.program.layout_binding.lde_blowup
        lde_domain_size = trace_domain_size * lde_blowup
        if lde_domain_size > MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE:
            raise ProofV3Error("Goldilocks AIR LDE exceeds the CPU reference cap")
        if lde_blowup > 64:
            raise ProofV3Error("Goldilocks AIR trace FRI final domain exceeds the CPU cap")
        trace_width = len(self.program.trace_columns)
        if trace_width < 1:
            raise ProofV3Error("Goldilocks AIR trace width must be positive")
        if self.program.max_air_constraint_degree > (
            self.program.layout_binding.max_constraint_degree
        ):
            raise ProofV3Error(
                "Goldilocks AIR structural constraint degree exceeds the signed layout"
            )
        # A degree-e expression over trace columns of degree < N has degree at
        # most e*(N-1).  Multiplying by a scope selector of degree < N and
        # dividing by X**N-1 gives the conservative quotient degree below.
        maximum_quotient_degree = max(
            0,
            (self.program.max_air_constraint_degree + 1)
            * (trace_domain_size - 1)
            - trace_domain_size,
        )
        required_composition_bound = _power_of_two_strictly_above(
            maximum_quotient_degree,
            name="Goldilocks AIR quotient degree",
        )
        # The standalone FRI reference opens its final codeword in full.  A
        # larger, still-safe bound keeps that reference-only opening bounded.
        reference_floor = max(1, lde_domain_size // 64)
        composition_degree_bound = max(required_composition_bound, reference_floor)
        if composition_degree_bound >= lde_domain_size:
            raise ProofV3Error(
                "Goldilocks AIR quotient degree does not fit the signed LDE domain"
            )
        query_count = min(
            GOLDILOCKS_AIR_REFERENCE_QUERY_COUNT_V3,
            lde_domain_size // 2,
        )
        if query_count < 1:
            raise ProofV3Error("Goldilocks AIR LDE domain is too small for FRI")
        program_digest = self.program.digest()
        lde_shift = _derive_lde_shift(
            validator_binding_digest=binding,
            program_digest=program_digest,
            token_count=token_count,
            trace_domain_size=trace_domain_size,
            lde_domain_size=lde_domain_size,
            trace_width=trace_width,
        )
        object.__setattr__(self, "validator_binding_digest", binding)
        object.__setattr__(self, "token_count", token_count)
        object.__setattr__(self, "program_digest", program_digest)
        object.__setattr__(self, "active_row_count", active_row_count)
        object.__setattr__(self, "trace_domain_size", trace_domain_size)
        object.__setattr__(self, "lde_domain_size", lde_domain_size)
        object.__setattr__(self, "lde_blowup", lde_blowup)
        object.__setattr__(self, "lde_shift", lde_shift)
        object.__setattr__(self, "trace_width", trace_width)
        object.__setattr__(self, "trace_degree_bound", trace_domain_size)
        object.__setattr__(
            self,
            "composition_degree_bound",
            composition_degree_bound,
        )
        object.__setattr__(self, "query_count", query_count)

    def canonical_bytes(self) -> bytes:
        abi = self.abi_id.encode("ascii")
        return (
            struct.pack("<HH", self.format_version, len(abi))
            + abi
            + self.validator_binding_digest
            + self.program_digest
            + struct.pack(
                "<IQQQQQII",
                self.token_count,
                self.active_row_count,
                self.trace_domain_size,
                self.lde_domain_size,
                self.lde_blowup,
                self.lde_shift,
                self.trace_width,
                self.query_count,
            )
            + struct.pack(
                "<QQ",
                self.trace_degree_bound,
                self.composition_degree_bound,
            )
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_CORE_DIGEST_DOMAIN + self.canonical_bytes()).digest()

    def trace_tree_binding_digest(self) -> bytes:
        """Return the nonce-independent Merkle-tree binding for LDE rows."""

        return hashlib.sha256(_TRACE_TREE_DOMAIN + self.digest()).digest()


@dataclass(frozen=True, slots=True)
class GoldilocksAirTracePrecommitmentReferenceV3:
    """Frozen trace-LDE root and its complete validator-derived statement."""

    core: GoldilocksAirStatementCoreReferenceV3
    trace_lde_commitment: bytes
    abi_id: str = GOLDILOCKS_AIR_REFERENCE_ABI_V3
    format_version: int = GOLDILOCKS_AIR_REFERENCE_FORMAT_VERSION_V3

    def __post_init__(self) -> None:
        if self.abi_id != GOLDILOCKS_AIR_REFERENCE_ABI_V3:
            raise ProofV3Error("Goldilocks AIR precommitment ABI is unsupported")
        if type(self.format_version) is not int or self.format_version != (
            GOLDILOCKS_AIR_REFERENCE_FORMAT_VERSION_V3
        ):
            raise ProofV3Error(
                "Goldilocks AIR precommitment format version is unsupported"
            )
        if not isinstance(self.core, GoldilocksAirStatementCoreReferenceV3):
            raise ProofV3Error("Goldilocks AIR precommitment core is malformed")
        root = _fixed32(
            self.trace_lde_commitment,
            "Goldilocks AIR trace_lde_commitment",
        )
        object.__setattr__(self, "trace_lde_commitment", root)

    def canonical_bytes(self) -> bytes:
        abi = self.abi_id.encode("ascii")
        return (
            struct.pack("<HH", self.format_version, len(abi))
            + abi
            + self.core.digest()
            + self.trace_lde_commitment
        )

    def digest(self) -> bytes:
        return hashlib.sha256(
            _PRECOMMITMENT_DOMAIN + self.canonical_bytes()
        ).digest()


@dataclass(frozen=True, slots=True)
class GoldilocksAirTraceOracleReferenceV3:
    """In-memory precommitted LDE tree retained only for CPU conformance."""

    precommitment: GoldilocksAirTracePrecommitmentReferenceV3
    trace_tree: GoldilocksMerkleTreeReference

    def __post_init__(self) -> None:
        if not isinstance(
            self.precommitment,
            GoldilocksAirTracePrecommitmentReferenceV3,
        ):
            raise ProofV3Error("Goldilocks AIR trace precommitment is malformed")
        if not isinstance(self.trace_tree, GoldilocksMerkleTreeReference):
            raise ProofV3Error("Goldilocks AIR trace tree is malformed")
        core = self.precommitment.core
        if self.trace_tree.commitment != self.precommitment.trace_lde_commitment:
            raise ProofV3Error("Goldilocks AIR trace tree does not match its precommitment")
        if self.trace_tree.binding_digest != core.trace_tree_binding_digest():
            raise ProofV3Error("Goldilocks AIR trace tree binding is unexpected")
        if (
            self.trace_tree.leaf_count != core.lde_domain_size
            or self.trace_tree.leaf_width != core.trace_width
        ):
            raise ProofV3Error("Goldilocks AIR trace tree shape is unexpected")


@dataclass(frozen=True, slots=True)
class GoldilocksAirTranscriptReferenceV3:
    """Post-nonce transcript over one previously frozen AIR trace root."""

    precommitment: GoldilocksAirTracePrecommitmentReferenceV3
    validator_nonce: bytes
    digest_value: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.precommitment,
            GoldilocksAirTracePrecommitmentReferenceV3,
        ):
            raise ProofV3Error("Goldilocks AIR transcript precommitment is malformed")
        nonce = _fixed32(self.validator_nonce, "Goldilocks AIR validator_nonce")
        digest = hashlib.sha256(
            _POSTCOMMIT_DOMAIN + self.precommitment.digest() + nonce
        ).digest()
        object.__setattr__(self, "validator_nonce", nonce)
        object.__setattr__(self, "digest_value", digest)

    @property
    def core(self) -> GoldilocksAirStatementCoreReferenceV3:
        return self.precommitment.core

    def trace_batch_coefficients(self, *, batch_index: int) -> tuple[int, ...]:
        if batch_index < 0 or batch_index >= GOLDILOCKS_AIR_REFERENCE_TRACE_BATCH_COUNT_V3:
            raise ProofV3Error("Goldilocks AIR trace batch index is out of range")
        return tuple(
            _derive_nonzero_field(
                transcript_digest=self.digest_value,
                label=b"trace-column",
                batch_index=batch_index,
                coordinate_index=index,
                coordinate_id=column.column_id,
            )
            for index, column in enumerate(self.core.program.trace_columns)
        )

    def composition_coefficients(self, *, batch_index: int) -> tuple[int, ...]:
        if (
            batch_index < 0
            or batch_index >= GOLDILOCKS_AIR_REFERENCE_COMPOSITION_BATCH_COUNT_V3
        ):
            raise ProofV3Error("Goldilocks AIR composition batch index is out of range")
        return tuple(
            _derive_nonzero_field(
                transcript_digest=self.digest_value,
                label=b"air-constraint",
                batch_index=batch_index,
                coordinate_index=index,
                coordinate_id=constraint.constraint_id,
            )
            for index, constraint in enumerate(self.core.program.air_constraints)
        )

    def trace_fri_statement(self, *, batch_index: int) -> GoldilocksFriStatementReference:
        coefficients = self.trace_batch_coefficients(batch_index=batch_index)
        binding = hashlib.sha256(
            _TRACE_FRI_BINDING_DOMAIN
            + self.digest_value
            + struct.pack("<I", batch_index)
            + b"".join(
                coefficient.to_bytes(8, "little") for coefficient in coefficients
            )
        ).digest()
        return GoldilocksFriStatementReference(
            binding_digest=binding,
            domain_size=self.core.lde_domain_size,
            degree_bound=self.core.trace_degree_bound,
            domain_shift=self.core.lde_shift,
            query_count=self.core.query_count,
        )

    def composition_fri_statement(
        self,
        *,
        batch_index: int,
    ) -> GoldilocksFriStatementReference:
        coefficients = self.composition_coefficients(batch_index=batch_index)
        binding = hashlib.sha256(
            _COMPOSITION_FRI_BINDING_DOMAIN
            + self.digest_value
            + struct.pack("<I", batch_index)
            + b"".join(
                coefficient.to_bytes(8, "little") for coefficient in coefficients
            )
        ).digest()
        return GoldilocksFriStatementReference(
            binding_digest=binding,
            domain_size=self.core.lde_domain_size,
            degree_bound=self.core.composition_degree_bound,
            domain_shift=self.core.lde_shift,
            query_count=self.core.query_count,
        )


@dataclass(frozen=True, slots=True)
class GoldilocksAirProofReferenceV3:
    """In-memory post-nonce proof for one frozen AIR trace LDE tree."""

    trace_batch_fri_proofs: tuple[GoldilocksFriProofReference, ...]
    composition_fri_proofs: tuple[GoldilocksFriProofReference, ...]
    trace_consistency_opening: GoldilocksMerkleMultiOpeningReference
    abi_id: str = GOLDILOCKS_AIR_REFERENCE_ABI_V3
    format_version: int = GOLDILOCKS_AIR_REFERENCE_FORMAT_VERSION_V3

    def __post_init__(self) -> None:
        if self.abi_id != GOLDILOCKS_AIR_REFERENCE_ABI_V3:
            raise ProofV3Error("Goldilocks AIR proof ABI is unsupported")
        if type(self.format_version) is not int or self.format_version != (
            GOLDILOCKS_AIR_REFERENCE_FORMAT_VERSION_V3
        ):
            raise ProofV3Error("Goldilocks AIR proof format version is unsupported")
        if (
            not isinstance(self.trace_batch_fri_proofs, tuple)
            or len(self.trace_batch_fri_proofs)
            != GOLDILOCKS_AIR_REFERENCE_TRACE_BATCH_COUNT_V3
            or not all(
                isinstance(item, GoldilocksFriProofReference)
                for item in self.trace_batch_fri_proofs
            )
        ):
            raise ProofV3Error("Goldilocks AIR trace FRI proof set is malformed")
        if (
            not isinstance(self.composition_fri_proofs, tuple)
            or len(self.composition_fri_proofs)
            != GOLDILOCKS_AIR_REFERENCE_COMPOSITION_BATCH_COUNT_V3
            or not all(
                isinstance(item, GoldilocksFriProofReference)
                for item in self.composition_fri_proofs
            )
        ):
            raise ProofV3Error("Goldilocks AIR composition FRI proof set is malformed")
        if not isinstance(
            self.trace_consistency_opening,
            GoldilocksMerkleMultiOpeningReference,
        ):
            raise ProofV3Error("Goldilocks AIR trace consistency opening is malformed")


def make_goldilocks_air_trace_precommitment_reference_v3(
    *,
    core: GoldilocksAirStatementCoreReferenceV3,
    trace_lde_commitment: bytes,
) -> GoldilocksAirTracePrecommitmentReferenceV3:
    """Build the typed record that must be frozen before nonce reveal."""

    return GoldilocksAirTracePrecommitmentReferenceV3(
        core=core,
        trace_lde_commitment=trace_lde_commitment,
    )


def build_goldilocks_air_trace_oracle_reference_v3(
    *,
    program: GoldilocksConstraintProgramV3,
    trace: GoldilocksConstraintTraceReferenceV3,
    token_count: int,
    validator_binding_digest: bytes,
) -> GoldilocksAirTraceOracleReferenceV3:
    """LDE-extend and precommit a base trace without checking its semantics.

    The returned oracle deliberately permits an invalid base trace: proof
    verification, rather than a prover-side construction helper, must reject
    invalid AIR witnesses.  The helper does require exact program, row count,
    width, field encoding, and nonce-independent tree binding.
    """

    core = GoldilocksAirStatementCoreReferenceV3(
        validator_binding_digest=validator_binding_digest,
        program=program,
        token_count=token_count,
    )
    if not isinstance(trace, GoldilocksConstraintTraceReferenceV3):
        raise ProofV3Error("Goldilocks AIR base trace has an unexpected type")
    if trace.constraint_program_digest != core.program_digest:
        raise ProofV3Error("Goldilocks AIR base trace belongs to a different program")
    if len(trace.rows) != core.trace_domain_size:
        raise ProofV3Error("Goldilocks AIR base trace has an unexpected row count")
    if any(len(row) != core.trace_width for row in trace.rows):
        raise ProofV3Error("Goldilocks AIR base trace has an unexpected row width")
    lde_columns = tuple(
        lde_goldilocks_reference(
            tuple(row[column_index] for row in trace.rows),
            target_size=core.lde_domain_size,
            source_shift=1,
            target_shift=core.lde_shift,
        )
        for column_index in range(core.trace_width)
    )
    lde_rows = tuple(
        tuple(column[row_index] for column in lde_columns)
        for row_index in range(core.lde_domain_size)
    )
    tree = GoldilocksMerkleTreeReference.from_rows(
        lde_rows,
        binding_digest=core.trace_tree_binding_digest(),
    )
    precommitment = make_goldilocks_air_trace_precommitment_reference_v3(
        core=core,
        trace_lde_commitment=tree.commitment,
    )
    return GoldilocksAirTraceOracleReferenceV3(
        precommitment=precommitment,
        trace_tree=tree,
    )


def _trace_batch_evaluations(
    *,
    trace_tree: GoldilocksMerkleTreeReference,
    coefficients: tuple[int, ...],
) -> tuple[int, ...]:
    if len(coefficients) != trace_tree.leaf_width:
        raise ProofV3Error("Goldilocks AIR trace batch has an unexpected width")
    return tuple(
        sum(
            coefficient * value
            for coefficient, value in zip(coefficients, row, strict=True)
        )
        % GOLDILOCKS_MODULUS
        for row in trace_tree.rows
    )


def _selector_lde_values(
    *,
    core: GoldilocksAirStatementCoreReferenceV3,
    scope: str,
) -> tuple[int, ...]:
    selected = set(
        _scope_rows(
            scope=scope,
            active_row_count=core.active_row_count,
            trace_domain_size=core.trace_domain_size,
        )
    )
    base_values = tuple(
        1 if row_index in selected else 0
        for row_index in range(core.trace_domain_size)
    )
    return lde_goldilocks_reference(
        base_values,
        target_size=core.lde_domain_size,
        source_shift=1,
        target_shift=core.lde_shift,
    )


def _composition_evaluations(
    *,
    core: GoldilocksAirStatementCoreReferenceV3,
    trace_rows: tuple[tuple[int, ...], ...],
    coefficients: tuple[int, ...],
) -> tuple[int, ...]:
    constraints = core.program.air_constraints
    if len(coefficients) != len(constraints):
        raise ProofV3Error("Goldilocks AIR composition coefficient count is unexpected")
    if len(trace_rows) != core.lde_domain_size or any(
        len(row) != core.trace_width for row in trace_rows
    ):
        raise ProofV3Error("Goldilocks AIR composition trace shape is unexpected")
    column_positions = {
        column.column_id: index
        for index, column in enumerate(core.program.trace_columns)
    }
    selectors = {
        constraint.scope: _selector_lde_values(core=core, scope=constraint.scope)
        for constraint in constraints
    }
    domain = goldilocks_radix2_domain_reference(
        size=core.lde_domain_size,
        shift=core.lde_shift,
    )
    denominator_inverses = tuple(
        goldilocks_inv(
            (pow(point, core.trace_domain_size, GOLDILOCKS_MODULUS) - 1)
            % GOLDILOCKS_MODULUS
        )
        for point in domain.points()
    )
    result: list[int] = []
    for row_index, row in enumerate(trace_rows):
        next_row = trace_rows[(row_index + core.lde_blowup) % core.lde_domain_size]
        total = 0
        for coefficient, constraint in zip(coefficients, constraints, strict=True):
            expression_value = constraint.expression._evaluate(
                current_row=row,
                next_row=next_row,
                column_positions=column_positions,
            )
            total = (
                total
                + coefficient
                * selectors[constraint.scope][row_index]
                * expression_value
                * denominator_inverses[row_index]
            ) % GOLDILOCKS_MODULUS
        result.append(total)
    return tuple(result)


def _composition_values_at_indices(
    *,
    core: GoldilocksAirStatementCoreReferenceV3,
    trace_rows: dict[int, tuple[int, ...]],
    coefficients: tuple[int, ...],
    indices: tuple[int, ...],
) -> dict[int, int]:
    """Recompute only verifier-opened composition positions from trace rows."""

    constraints = core.program.air_constraints
    if len(coefficients) != len(constraints):
        raise ProofV3VerificationError(
            "Goldilocks AIR composition coefficient count is unexpected"
        )
    if indices != tuple(sorted(set(indices))):
        raise ProofV3VerificationError("Goldilocks AIR composition indices are malformed")
    column_positions = {
        column.column_id: index
        for index, column in enumerate(core.program.trace_columns)
    }
    selectors = {
        constraint.scope: _selector_lde_values(core=core, scope=constraint.scope)
        for constraint in constraints
    }
    domain = goldilocks_radix2_domain_reference(
        size=core.lde_domain_size,
        shift=core.lde_shift,
    )
    points = domain.points()
    result: dict[int, int] = {}
    for row_index in indices:
        row = trace_rows.get(row_index)
        next_row = trace_rows.get(
            (row_index + core.lde_blowup) % core.lde_domain_size
        )
        if (
            row is None
            or next_row is None
            or len(row) != core.trace_width
            or len(next_row) != core.trace_width
        ):
            raise ProofV3VerificationError(
                "Goldilocks AIR trace opening omits a composition row"
            )
        denominator_inverse = goldilocks_inv(
            (pow(points[row_index], core.trace_domain_size, GOLDILOCKS_MODULUS) - 1)
            % GOLDILOCKS_MODULUS
        )
        total = 0
        for coefficient, constraint in zip(coefficients, constraints, strict=True):
            expression_value = constraint.expression._evaluate(
                current_row=row,
                next_row=next_row,
                column_positions=column_positions,
            )
            total = (
                total
                + coefficient
                * selectors[constraint.scope][row_index]
                * expression_value
                * denominator_inverse
            ) % GOLDILOCKS_MODULUS
        result[row_index] = total
    return result


def _trace_consistency_indices(
    *,
    core: GoldilocksAirStatementCoreReferenceV3,
    trace_proofs: tuple[GoldilocksFriProofReference, ...],
    trace_statements: tuple[GoldilocksFriStatementReference, ...],
    composition_proofs: tuple[GoldilocksFriProofReference, ...],
    composition_statements: tuple[GoldilocksFriStatementReference, ...],
) -> tuple[int, ...]:
    indices: set[int] = set()
    for proof, statement in zip(trace_proofs, trace_statements, strict=True):
        indices.update(_scalar_source_opening(proof, statement=statement).indices)
    for proof, statement in zip(
        composition_proofs,
        composition_statements,
        strict=True,
    ):
        source_indices = _scalar_source_opening(proof, statement=statement).indices
        indices.update(source_indices)
        indices.update(
            (index + core.lde_blowup) % core.lde_domain_size
            for index in source_indices
        )
    if not indices:
        raise ProofV3Error("Goldilocks AIR consistency opening has no query indices")
    return tuple(sorted(indices))


def prove_goldilocks_air_reference_v3(
    *,
    trace_oracle: GoldilocksAirTraceOracleReferenceV3,
    validator_nonce: bytes,
) -> GoldilocksAirProofReferenceV3:
    """Build one bounded post-nonce AIR proof over a frozen trace tree."""

    if not isinstance(trace_oracle, GoldilocksAirTraceOracleReferenceV3):
        raise ProofV3Error("Goldilocks AIR trace oracle has an unexpected type")
    transcript = GoldilocksAirTranscriptReferenceV3(
        precommitment=trace_oracle.precommitment,
        validator_nonce=validator_nonce,
    )
    trace_statements = tuple(
        transcript.trace_fri_statement(batch_index=batch_index)
        for batch_index in range(GOLDILOCKS_AIR_REFERENCE_TRACE_BATCH_COUNT_V3)
    )
    trace_proofs = tuple(
        prove_goldilocks_fri_reference(
            _trace_batch_evaluations(
                trace_tree=trace_oracle.trace_tree,
                coefficients=transcript.trace_batch_coefficients(
                    batch_index=batch_index
                ),
            ),
            statement=statement,
        )
        for batch_index, statement in enumerate(trace_statements)
    )
    composition_statements = tuple(
        transcript.composition_fri_statement(batch_index=batch_index)
        for batch_index in range(
            GOLDILOCKS_AIR_REFERENCE_COMPOSITION_BATCH_COUNT_V3
        )
    )
    composition_proofs = tuple(
        prove_goldilocks_fri_reference(
            _composition_evaluations(
                core=transcript.core,
                trace_rows=trace_oracle.trace_tree.rows,
                coefficients=transcript.composition_coefficients(
                    batch_index=batch_index
                ),
            ),
            statement=statement,
        )
        for batch_index, statement in enumerate(composition_statements)
    )
    indices = _trace_consistency_indices(
        core=transcript.core,
        trace_proofs=trace_proofs,
        trace_statements=trace_statements,
        composition_proofs=composition_proofs,
        composition_statements=composition_statements,
    )
    return GoldilocksAirProofReferenceV3(
        trace_batch_fri_proofs=trace_proofs,
        composition_fri_proofs=composition_proofs,
        trace_consistency_opening=trace_oracle.trace_tree.open(indices),
    )


def _require_precommitment_matches_core(
    *,
    core: GoldilocksAirStatementCoreReferenceV3,
    precommitment: GoldilocksAirTracePrecommitmentReferenceV3,
) -> None:
    if not isinstance(core, GoldilocksAirStatementCoreReferenceV3):
        raise ProofV3VerificationError("Goldilocks AIR verifier core is malformed")
    if not isinstance(precommitment, GoldilocksAirTracePrecommitmentReferenceV3):
        raise ProofV3VerificationError("Goldilocks AIR precommitment is malformed")
    if precommitment.core.digest() != core.digest():
        raise ProofV3VerificationError(
            "Goldilocks AIR precommitment belongs to a different statement"
        )


def verify_goldilocks_air_reference_v3(
    proof: object,
    *,
    core: GoldilocksAirStatementCoreReferenceV3,
    precommitment: GoldilocksAirTracePrecommitmentReferenceV3,
    validator_nonce: bytes,
) -> None:
    """Verify the full frozen-trace, batch-LDT, and AIR-composition reference.

    Every expectation is recomputed from the verifier-owned core,
    precommitment, and nonce.  Invalid or malformed data is always a proof
    failure; this reference has no "not requested" or optional-success path.
    """

    try:
        _require_precommitment_matches_core(
            core=core,
            precommitment=precommitment,
        )
        if not isinstance(proof, GoldilocksAirProofReferenceV3):
            raise ProofV3VerificationError("Goldilocks AIR proof has an unexpected type")
        transcript = GoldilocksAirTranscriptReferenceV3(
            precommitment=precommitment,
            validator_nonce=validator_nonce,
        )
        trace_statements = tuple(
            transcript.trace_fri_statement(batch_index=batch_index)
            for batch_index in range(
                GOLDILOCKS_AIR_REFERENCE_TRACE_BATCH_COUNT_V3
            )
        )
        composition_statements = tuple(
            transcript.composition_fri_statement(batch_index=batch_index)
            for batch_index in range(
                GOLDILOCKS_AIR_REFERENCE_COMPOSITION_BATCH_COUNT_V3
            )
        )
        for fri_proof, statement in zip(
            proof.trace_batch_fri_proofs,
            trace_statements,
            strict=True,
        ):
            verify_goldilocks_fri_reference(fri_proof, statement=statement)
        for fri_proof, statement in zip(
            proof.composition_fri_proofs,
            composition_statements,
            strict=True,
        ):
            verify_goldilocks_fri_reference(fri_proof, statement=statement)
        indices = _trace_consistency_indices(
            core=core,
            trace_proofs=proof.trace_batch_fri_proofs,
            trace_statements=trace_statements,
            composition_proofs=proof.composition_fri_proofs,
            composition_statements=composition_statements,
        )
        verify_goldilocks_merkle_multiopening_reference(
            precommitment.trace_lde_commitment,
            proof.trace_consistency_opening,
            expected_binding_digest=core.trace_tree_binding_digest(),
            expected_leaf_count=core.lde_domain_size,
            expected_leaf_width=core.trace_width,
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
        for batch_index, (fri_proof, statement) in enumerate(
            zip(proof.trace_batch_fri_proofs, trace_statements, strict=True)
        ):
            opened_values = _scalar_opened_values(
                _scalar_source_opening(fri_proof, statement=statement)
            )
            coefficients = transcript.trace_batch_coefficients(batch_index=batch_index)
            for index, value in opened_values.items():
                row = trace_rows.get(index)
                if row is None:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR trace opening omits a trace-batch row"
                    )
                expected = sum(
                    coefficient * element
                    for coefficient, element in zip(coefficients, row, strict=True)
                ) % GOLDILOCKS_MODULUS
                if value != expected:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR trace FRI is not bound to the frozen trace"
                    )
        for batch_index, (fri_proof, statement) in enumerate(
            zip(proof.composition_fri_proofs, composition_statements, strict=True)
        ):
            opened_values = _scalar_opened_values(
                _scalar_source_opening(fri_proof, statement=statement)
            )
            composition = _composition_values_at_indices(
                core=core,
                trace_rows=trace_rows,
                coefficients=transcript.composition_coefficients(
                    batch_index=batch_index
                ),
                indices=tuple(sorted(opened_values)),
            )
            for index, value in opened_values.items():
                if value != composition[index]:
                    raise ProofV3VerificationError(
                        "Goldilocks AIR composition is not bound to the frozen trace"
                    )
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("Goldilocks AIR proof is malformed") from exc


__all__ = [
    "GOLDILOCKS_AIR_REFERENCE_ABI_V3",
    "GOLDILOCKS_AIR_REFERENCE_COMPOSITION_BATCH_COUNT_V3",
    "GOLDILOCKS_AIR_REFERENCE_FORMAT_VERSION_V3",
    "GOLDILOCKS_AIR_REFERENCE_QUERY_COUNT_V3",
    "GOLDILOCKS_AIR_REFERENCE_TRACE_BATCH_COUNT_V3",
    "GoldilocksAirProofReferenceV3",
    "GoldilocksAirStatementCoreReferenceV3",
    "GoldilocksAirTraceOracleReferenceV3",
    "GoldilocksAirTracePrecommitmentReferenceV3",
    "GoldilocksAirTranscriptReferenceV3",
    "build_goldilocks_air_trace_oracle_reference_v3",
    "make_goldilocks_air_trace_precommitment_reference_v3",
    "prove_goldilocks_air_reference_v3",
    "verify_goldilocks_air_reference_v3",
]
