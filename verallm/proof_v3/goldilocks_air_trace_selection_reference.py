"""Semantic hard-audit expansion for the unregistered V3 AIR trace map.

The trace-map reference commits every signed ``(layout, phase, chunk)``
trace before the validator nonce.  Selecting arbitrary uniform slots after
that commitment demonstrates chronology, but it does not enforce the signed
hard-audit policy.  This CPU-only reference expands the existing
post-commitment :class:`HardAuditSelectionV3` into the exact trace slots a
selected layer needs:

* every required registered linear operation;
* the signed full-attention or GDN transition node; and
* the signed residual bridge node,

over every prefill/decode chunk intersected by the selected query window.
The range expansion includes both sides of a crossed chunk or phase boundary.
It preserves the nonce-selected full-attention head indices for a qualified
attention adapter to consume; generic AIR layout slots intentionally do not
pretend that a head-specific witness is already implemented.

This module is deliberately unregistered.  It does not authenticate cache-RAM
pages, lower an attention/GDN witness, bind runtime tensors, or verify an AIR
proof.  A map-aware validator receipt coordinator must still seal the map
sidecar before revealing the nonce, and a qualified native adapter must prove
the returned slot traces and their cache/transition relations.
"""

from __future__ import annotations

import hashlib
import operator
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.challenge import (
    HardAuditLayerSelectionV3,
    HardAuditSelectionV3,
    derive_folded_execution_challenge_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_air_trace_map_reference import (
    GoldilocksAirTraceMapStatementReferenceV3,
)
from verallm.proof_v3.relation import LayerAuditPlanV3


GOLDILOCKS_AIR_TRACE_SELECTION_REFERENCE_ABI_V3: Final = (
    "goldilocks.air_trace_selection.reference.v1"
)
GOLDILOCKS_AIR_TRACE_SELECTION_REFERENCE_FORMAT_VERSION_V3: Final = 1

_TRANSITION_CODES: Final = {"full_attention": 1, "gdn": 2}
_SELECTION_BINDING_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR_TRACE_SELECTION/V1/BINDING/SHA256"
)
_SELECTION_DIGEST_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_AIR_TRACE_SELECTION/V1/DIGEST/SHA256"
)


def _fixed32(value: object, name: str, *, nonzero: bool = False) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    if nonzero and value == bytes(32):
        raise ProofV3Error(f"{name} must not be zero")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ProofV3Error(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ProofV3Error(f"{name} must be an integer") from exc


def _u32(value: object, name: str, *, positive: bool = False) -> int:
    result = _integer(value, name)
    if result < (1 if positive else 0) or result >= 1 << 32:
        raise ProofV3Error(f"{name} must be an unsigned 32-bit integer")
    return result


def _identifier(value: object, name: str) -> bytes:
    if not isinstance(value, str):
        raise ProofV3Error(f"{name} is malformed")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ProofV3Error(f"{name} is not ASCII") from exc
    if not 1 <= len(encoded) <= 128:
        raise ProofV3Error(f"{name} is malformed")
    return encoded


def _sorted_distinct_indices(value: object, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        raise ProofV3Error(f"{name} must be an iterable")
    try:
        raw = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ProofV3Error(f"{name} must be an iterable") from exc
    if not raw:
        raise ProofV3Error(f"{name} must not be empty")
    indices = tuple(_u32(item, f"{name}[{index}]") for index, item in enumerate(raw))
    if indices != tuple(sorted(set(indices))):
        raise ProofV3Error(f"{name} must be sorted and distinct")
    return indices


def _selection_bytes(selection: HardAuditSelectionV3) -> bytes:
    """Serialize the existing typed selection only for local binding checks."""

    if not isinstance(selection, HardAuditSelectionV3):
        raise ProofV3Error("Goldilocks AIR trace selection is malformed")
    abi = _identifier(
        selection.selection_abi_id,
        "Goldilocks AIR trace selection ABI",
    )
    entries = bytearray(
        struct.pack(
            "<BII",
            len(abi),
            selection.sequence_token_count,
            len(selection.layers),
        )
        + abi
    )
    for layer in selection.layers:
        if not isinstance(layer, HardAuditLayerSelectionV3):
            raise ProofV3Error("Goldilocks AIR trace selection layer is malformed")
        heads = tuple(layer.attention_head_indices)
        entries.extend(
            struct.pack(
                "<IBIII",
                layer.layer_index,
                _TRANSITION_CODES[layer.transition_kind],
                layer.query_row_offset,
                layer.query_row_count,
                len(heads),
            )
        )
        entries.extend(b"".join(struct.pack("<I", head) for head in heads))
    return bytes(entries)


@dataclass(frozen=True, slots=True)
class GoldilocksAirTraceLayerCoverageReferenceV3:
    """Exact semantic node and trace-slot coverage for one selected layer."""

    layer_index: int
    transition_kind: str
    query_row_offset: int
    query_row_count: int
    attention_head_indices: tuple[int, ...]
    node_ids: tuple[str, ...]
    slot_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _u32(self.layer_index, "Goldilocks AIR trace coverage layer_index")
        if self.transition_kind not in _TRANSITION_CODES:
            raise ProofV3Error("Goldilocks AIR trace coverage transition is unsupported")
        _u32(
            self.query_row_offset,
            "Goldilocks AIR trace coverage query_row_offset",
        )
        _u32(
            self.query_row_count,
            "Goldilocks AIR trace coverage query_row_count",
            positive=True,
        )
        heads = tuple(
            _u32(head, f"Goldilocks AIR trace coverage head[{index}]")
            for index, head in enumerate(self.attention_head_indices)
        )
        if heads != tuple(sorted(set(heads))):
            raise ProofV3Error(
                "Goldilocks AIR trace coverage heads must be sorted and distinct"
            )
        if self.transition_kind == "full_attention" and not heads:
            raise ProofV3Error("full-attention trace coverage requires selected heads")
        if self.transition_kind == "gdn" and heads:
            raise ProofV3Error("GDN trace coverage must not contain selected heads")
        node_ids = tuple(
            _identifier(node_id, f"Goldilocks AIR trace coverage node[{index}]").decode(
                "ascii"
            )
            for index, node_id in enumerate(self.node_ids)
        )
        if not node_ids or node_ids != tuple(sorted(set(node_ids))):
            raise ProofV3Error(
                "Goldilocks AIR trace coverage nodes must be sorted and distinct"
            )
        slots = _sorted_distinct_indices(
            self.slot_indices,
            "Goldilocks AIR trace coverage slots",
        )
        object.__setattr__(self, "attention_head_indices", heads)
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "slot_indices", slots)

    def canonical_bytes(self) -> bytes:
        encoded_nodes = bytearray()
        for node_id in self.node_ids:
            encoded = _identifier(node_id, "Goldilocks AIR trace coverage node")
            encoded_nodes.extend(struct.pack("<B", len(encoded)))
            encoded_nodes.extend(encoded)
        return (
            struct.pack(
                "<IBIIII",
                self.layer_index,
                _TRANSITION_CODES[self.transition_kind],
                self.query_row_offset,
                self.query_row_count,
                len(self.attention_head_indices),
                len(self.node_ids),
            )
            + b"".join(
                struct.pack("<I", head) for head in self.attention_head_indices
            )
            + bytes(encoded_nodes)
            + struct.pack("<I", len(self.slot_indices))
            + b"".join(struct.pack("<I", index) for index in self.slot_indices)
        )


def _selection_binding_material(
    *,
    abi_id: str,
    format_version: int,
    statement_digest: bytes,
    transcript_digest: bytes,
    hard_audit_selection: HardAuditSelectionV3,
    layer_coverages: tuple[GoldilocksAirTraceLayerCoverageReferenceV3, ...],
    slot_count: int,
    slot_indices: tuple[int, ...],
) -> bytes:
    abi = _identifier(abi_id, "Goldilocks AIR trace selection ABI")
    return (
        struct.pack("<HB", format_version, len(abi))
        + abi
        + statement_digest
        + transcript_digest
        + _selection_bytes(hard_audit_selection)
        + struct.pack("<I", len(layer_coverages))
        + b"".join(item.canonical_bytes() for item in layer_coverages)
        + struct.pack("<II", slot_count, len(slot_indices))
        + b"".join(struct.pack("<I", index) for index in slot_indices)
    )


@dataclass(frozen=True, slots=True)
class GoldilocksAirTraceHardSelectionReferenceV3:
    """Verifier-rederived semantic selection and its trace-map slot union."""

    statement_digest: bytes
    transcript_digest: bytes
    hard_audit_selection: HardAuditSelectionV3
    layer_coverages: tuple[GoldilocksAirTraceLayerCoverageReferenceV3, ...]
    slot_count: int
    slot_indices: tuple[int, ...]
    binding_digest: bytes
    abi_id: str = GOLDILOCKS_AIR_TRACE_SELECTION_REFERENCE_ABI_V3
    format_version: int = GOLDILOCKS_AIR_TRACE_SELECTION_REFERENCE_FORMAT_VERSION_V3

    def __post_init__(self) -> None:
        if self.abi_id != GOLDILOCKS_AIR_TRACE_SELECTION_REFERENCE_ABI_V3:
            raise ProofV3Error("Goldilocks AIR trace selection ABI is unsupported")
        if type(self.format_version) is not int or self.format_version != (
            GOLDILOCKS_AIR_TRACE_SELECTION_REFERENCE_FORMAT_VERSION_V3
        ):
            raise ProofV3Error(
                "Goldilocks AIR trace selection format version is unsupported"
            )
        _fixed32(
            self.statement_digest,
            "Goldilocks AIR trace selection statement digest",
            nonzero=True,
        )
        _fixed32(
            self.transcript_digest,
            "Goldilocks AIR trace selection transcript digest",
        )
        _fixed32(
            self.binding_digest,
            "Goldilocks AIR trace selection binding digest",
        )
        if not isinstance(self.hard_audit_selection, HardAuditSelectionV3):
            raise ProofV3Error("Goldilocks AIR trace hard selection is malformed")
        coverages = tuple(self.layer_coverages)
        if not coverages or not all(
            isinstance(item, GoldilocksAirTraceLayerCoverageReferenceV3)
            for item in coverages
        ):
            raise ProofV3Error("Goldilocks AIR trace layer coverages are malformed")
        selected_layers = self.hard_audit_selection.layers
        if len(coverages) != len(selected_layers):
            raise ProofV3Error(
                "Goldilocks AIR trace layer coverage count does not match selection"
            )
        for coverage, selected in zip(coverages, selected_layers, strict=True):
            if (
                coverage.layer_index != selected.layer_index
                or coverage.transition_kind != selected.transition_kind
                or coverage.query_row_offset != selected.query_row_offset
                or coverage.query_row_count != selected.query_row_count
                or coverage.attention_head_indices != selected.attention_head_indices
            ):
                raise ProofV3Error(
                    "Goldilocks AIR trace layer coverage does not match selection"
                )
        slot_count = _u32(
            self.slot_count,
            "Goldilocks AIR trace selection slot_count",
            positive=True,
        )
        slots = _sorted_distinct_indices(
            self.slot_indices,
            "Goldilocks AIR trace selected slots",
        )
        if any(index >= slot_count for index in slots):
            raise ProofV3Error("Goldilocks AIR trace selected slot is out of range")
        if slots != tuple(
            sorted(
                {
                    index
                    for coverage in coverages
                    for index in coverage.slot_indices
                }
            )
        ):
            raise ProofV3Error(
                "Goldilocks AIR trace selected slots do not match layer coverage"
            )
        object.__setattr__(self, "layer_coverages", coverages)
        object.__setattr__(self, "slot_count", slot_count)
        object.__setattr__(self, "slot_indices", slots)
        expected_binding = self._expected_binding_digest()
        if self.binding_digest != expected_binding:
            raise ProofV3Error("Goldilocks AIR trace selection binding is unexpected")

    def _binding_material(self) -> bytes:
        return _selection_binding_material(
            abi_id=self.abi_id,
            format_version=self.format_version,
            statement_digest=self.statement_digest,
            transcript_digest=self.transcript_digest,
            hard_audit_selection=self.hard_audit_selection,
            layer_coverages=self.layer_coverages,
            slot_count=self.slot_count,
            slot_indices=self.slot_indices,
        )

    def _expected_binding_digest(self) -> bytes:
        return hashlib.sha256(_SELECTION_BINDING_DOMAIN + self._binding_material()).digest()

    def canonical_bytes(self) -> bytes:
        return self._binding_material() + self.binding_digest

    def digest(self) -> bytes:
        return hashlib.sha256(_SELECTION_DIGEST_DOMAIN + self.canonical_bytes()).digest()


def _require_statement(
    statement: object,
) -> GoldilocksAirTraceMapStatementReferenceV3:
    if not isinstance(statement, GoldilocksAirTraceMapStatementReferenceV3):
        raise ProofV3VerificationError("Goldilocks AIR trace-map statement is malformed")
    statement.require_factory_provenance()
    return statement


def _validate_selection_against_signed_policy(
    *,
    statement: GoldilocksAirTraceMapStatementReferenceV3,
    hard_audit_selection: HardAuditSelectionV3,
) -> dict[int, LayerAuditPlanV3]:
    """Reject any typed selection weaker than the signed hard policy."""

    if not isinstance(hard_audit_selection, HardAuditSelectionV3):
        raise ProofV3VerificationError("Goldilocks AIR hard selection is malformed")
    profile = statement.slot_universe.profile
    envelope = statement.slot_universe.envelope
    relation = profile.relation_spec
    policy = relation.audit_policy
    if hard_audit_selection.selection_abi_id != policy.selection_abi_id:
        raise ProofV3VerificationError("Goldilocks AIR hard selection ABI is unexpected")
    expected_sequence_count = envelope.context_token_count + envelope.decode_token_count
    if hard_audit_selection.sequence_token_count != expected_sequence_count:
        raise ProofV3VerificationError(
            "Goldilocks AIR hard selection sequence domain is unexpected"
        )
    selected_layers = hard_audit_selection.layers
    if len(selected_layers) != policy.selected_layer_count:
        raise ProofV3VerificationError(
            "Goldilocks AIR hard selection layer count is weaker than policy"
        )
    plans = {plan.layer_index: plan for plan in relation.layer_audits}
    if len(plans) != len(relation.layer_audits):
        raise ProofV3VerificationError("Goldilocks AIR signed layer plans are malformed")
    full_attention_count = 0
    gdn_count = 0
    expected_rows = min(policy.transition_query_rows, expected_sequence_count)
    for selected in selected_layers:
        plan = plans.get(selected.layer_index)
        if plan is None:
            raise ProofV3VerificationError("Goldilocks AIR selection chose unknown layer")
        expected_kind = "full_attention" if plan.is_full_attention else "gdn"
        if selected.transition_kind != expected_kind:
            raise ProofV3VerificationError(
                "Goldilocks AIR selection transition differs from signed layer plan"
            )
        if selected.query_row_count != expected_rows:
            raise ProofV3VerificationError(
                "Goldilocks AIR selection query coverage differs from signed policy"
            )
        if selected.query_row_offset + selected.query_row_count > expected_sequence_count:
            raise ProofV3VerificationError(
                "Goldilocks AIR selection query range exceeds sequence domain"
            )
        if plan.is_full_attention:
            full_attention_count += 1
            if len(selected.attention_head_indices) != policy.full_attention_heads_per_layer:
                raise ProofV3VerificationError(
                    "Goldilocks AIR selection has fewer full-attention heads than policy"
                )
            if any(
                head >= plan.attention_query_head_count
                for head in selected.attention_head_indices
            ):
                raise ProofV3VerificationError(
                    "Goldilocks AIR selection head is outside signed geometry"
                )
        else:
            gdn_count += 1
            if selected.attention_head_indices:
                raise ProofV3VerificationError(
                    "Goldilocks AIR GDN selection contains attention heads"
                )
    if full_attention_count < policy.minimum_full_attention_layers:
        raise ProofV3VerificationError(
            "Goldilocks AIR selection lacks required full-attention coverage"
        )
    if gdn_count < policy.minimum_gdn_layers:
        raise ProofV3VerificationError("Goldilocks AIR selection lacks required GDN coverage")
    return plans


def _node_ids_for_selected_layer(
    *,
    statement: GoldilocksAirTraceMapStatementReferenceV3,
    layer_index: int,
    plans: dict[int, LayerAuditPlanV3],
) -> tuple[str, ...]:
    """Resolve the selected layer's complete signed local operation set."""

    profile = statement.slot_universe.profile
    relation = profile.relation_spec
    plan = plans.get(layer_index)
    if plan is None:
        raise ProofV3VerificationError("Goldilocks AIR selection chose unknown layer")
    node_ids = {plan.transition_node_id, plan.bridge_node_id}
    relation_nodes = tuple(relation.nodes)
    for operation in plan.required_operation_references:
        matches = tuple(
            node.node_id
            for node in relation_nodes
            if node.operation_reference == operation
        )
        if len(matches) != 1:
            raise ProofV3VerificationError(
                "Goldilocks AIR selected operation lacks one signed graph node"
            )
        node_ids.add(matches[0])
    if len(node_ids) != 2 + len(plan.required_operation_references):
        raise ProofV3VerificationError(
            "Goldilocks AIR selected layer has overlapping signed graph nodes"
        )
    return tuple(sorted(node_ids))


def expand_goldilocks_air_trace_hard_selection_reference_v3(
    *,
    statement: GoldilocksAirTraceMapStatementReferenceV3,
    transcript_digest: bytes,
    hard_audit_selection: HardAuditSelectionV3,
) -> GoldilocksAirTraceHardSelectionReferenceV3:
    """Expand a typed selection into all exact precommitted map slots.

    This lower-level helper validates the signed policy but cannot prove that
    its ``hard_audit_selection`` came from a validator nonce.  Production
    coordinators must use :func:`derive_goldilocks_air_trace_hard_selection_reference_v3`
    or independently rederive the same challenge before accepting an opening.
    """

    try:
        statement = _require_statement(statement)
        transcript = _fixed32(
            transcript_digest,
            "Goldilocks AIR trace selection transcript digest",
        )
        plans = _validate_selection_against_signed_policy(
            statement=statement,
            hard_audit_selection=hard_audit_selection,
        )
        coverages: list[GoldilocksAirTraceLayerCoverageReferenceV3] = []
        for selected in hard_audit_selection.layers:
            node_ids = _node_ids_for_selected_layer(
                statement=statement,
                layer_index=selected.layer_index,
                plans=plans,
            )
            slot_indices = tuple(
                sorted(
                    {
                        slot_index
                        for node_id in node_ids
                        for slot_index in statement.slot_universe.slot_indices_for_node_range(
                            node_id=node_id,
                            logical_token_start=selected.query_row_offset,
                            token_count=selected.query_row_count,
                        )
                    }
                )
            )
            coverages.append(
                GoldilocksAirTraceLayerCoverageReferenceV3(
                    layer_index=selected.layer_index,
                    transition_kind=selected.transition_kind,
                    query_row_offset=selected.query_row_offset,
                    query_row_count=selected.query_row_count,
                    attention_head_indices=selected.attention_head_indices,
                    node_ids=node_ids,
                    slot_indices=slot_indices,
                )
            )
        all_slot_indices = tuple(
            sorted(
                {
                    slot_index
                    for coverage in coverages
                    for slot_index in coverage.slot_indices
                }
            )
        )
        binding = hashlib.sha256(
            _SELECTION_BINDING_DOMAIN
            + _selection_binding_material(
                abi_id=GOLDILOCKS_AIR_TRACE_SELECTION_REFERENCE_ABI_V3,
                format_version=GOLDILOCKS_AIR_TRACE_SELECTION_REFERENCE_FORMAT_VERSION_V3,
                statement_digest=statement.digest(),
                transcript_digest=transcript,
                hard_audit_selection=hard_audit_selection,
                layer_coverages=tuple(coverages),
                slot_count=len(statement.slot_universe),
                slot_indices=all_slot_indices,
            )
        ).digest()
        return GoldilocksAirTraceHardSelectionReferenceV3(
            statement_digest=statement.digest(),
            transcript_digest=transcript,
            hard_audit_selection=hard_audit_selection,
            layer_coverages=tuple(coverages),
            slot_count=len(statement.slot_universe),
            slot_indices=all_slot_indices,
            binding_digest=binding,
        )
    except ProofV3VerificationError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "Goldilocks AIR semantic trace selection is malformed"
        ) from exc


def derive_goldilocks_air_trace_hard_selection_reference_v3(
    *,
    validator_nonce: bytes,
    statement: GoldilocksAirTraceMapStatementReferenceV3,
) -> GoldilocksAirTraceHardSelectionReferenceV3:
    """Replay the nonce-bound signed hard selection and expand its slots."""

    try:
        statement = _require_statement(statement)
        nonce = _fixed32(
            validator_nonce,
            "Goldilocks AIR trace selection validator nonce",
            nonzero=True,
        )
        challenge = derive_folded_execution_challenge_v3(
            validator_nonce=nonce,
            profile=statement.slot_universe.profile,
            envelope=statement.slot_universe.envelope,
        )
        return expand_goldilocks_air_trace_hard_selection_reference_v3(
            statement=statement,
            transcript_digest=challenge.transcript_digest,
            hard_audit_selection=challenge.hard_audit_selection,
        )
    except ProofV3VerificationError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "Goldilocks AIR semantic trace selection is malformed"
        ) from exc


def verify_goldilocks_air_trace_hard_selection_reference_v3(
    selection: object,
    *,
    validator_nonce: bytes,
    statement: GoldilocksAirTraceMapStatementReferenceV3,
) -> tuple[int, ...]:
    """Fail closed unless a supplied selection equals the validator replay."""

    try:
        if not isinstance(selection, GoldilocksAirTraceHardSelectionReferenceV3):
            raise ProofV3VerificationError("Goldilocks AIR trace selection is malformed")
        expected = derive_goldilocks_air_trace_hard_selection_reference_v3(
            validator_nonce=validator_nonce,
            statement=statement,
        )
        if selection != expected:
            raise ProofV3VerificationError(
                "Goldilocks AIR trace selection differs from validator replay"
            )
        return selection.slot_indices
    except ProofV3VerificationError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "Goldilocks AIR semantic trace selection is malformed"
        ) from exc


__all__ = [
    "GOLDILOCKS_AIR_TRACE_SELECTION_REFERENCE_ABI_V3",
    "GOLDILOCKS_AIR_TRACE_SELECTION_REFERENCE_FORMAT_VERSION_V3",
    "GoldilocksAirTraceHardSelectionReferenceV3",
    "GoldilocksAirTraceLayerCoverageReferenceV3",
    "derive_goldilocks_air_trace_hard_selection_reference_v3",
    "expand_goldilocks_air_trace_hard_selection_reference_v3",
    "verify_goldilocks_air_trace_hard_selection_reference_v3",
]
