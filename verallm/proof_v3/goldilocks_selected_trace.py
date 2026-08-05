"""One fail-closed verifier for the complete compact selected trace.

Dynamic projection, residual, attention and terminal relations share one
Goldilocks BaseFold/FRI terminal opening. Static projection and LM-head
weights remain authenticated by their registered Pallas catalogs. Reuse-only
relations (GDN, RMSNorm and SwiGLU) run only after the shared opening has
authenticated their source cells.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Final

from verallm.proof_v3.economic_artifacts import EconomicVerifiedArtifactsV3
from verallm.proof_v3.economic_commitment import oracle_leaf_width_v3
from verallm.proof_v3.economic_challenge import (
    CORRIDOR_REL_COEFF_DEN_V3,
    CORRIDOR_REL_COEFF_NUM_V3,
)
from verallm.proof_v3.economic_lm_head_catalog_fold import (
    EconomicLmHeadCatalogBindingV3,
)
from verallm.proof_v3.economic_gdn_replay import (
    economic_gdn_runtime_columns_v3,
)
from verallm.proof_v3.economic_wire import (
    EconomicExecutionAnchorLaneRevealV3,
    EconomicOracleCommitmentV3,
    EconomicWeightRowRevealV3,
    VALUE_MODE_INT8,
    bits_to_scale_v3,
    scale_to_bits_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.execution_anchor import (
    ExecutionAnchorCommitmentV3,
    execution_anchor_lane_bytes_v3,
)
from verallm.proof_v3.gdn_runtime_semantics import GdnRuntimeSemanticsV3
from verallm.proof_v3.attention_anchor_binding import (
    AttentionAnchorGeometryV3,
    attention_anchor_geometry_v3,
    attention_anchor_head_byte_range_v3,
    decode_runtime_values_v3,
    extract_execution_anchor_range_v3,
    required_execution_anchor_lanes_v3,
    runtime_attention_q_head_quantized_v3,
    runtime_kv_head_quantized_v3,
)
from verallm.proof_v3.attention_runtime_semantics import (
    AttentionRuntimeSemanticsV3,
)
from verallm.proof_v3.goldilocks_bottom_anchor import (
    GoldilocksBottomAnchorClaimV3,
    GoldilocksBottomAnchorProofV3,
    verify_goldilocks_bottom_anchor_v3,
)
from verallm.proof_v3.goldilocks_final_rmsnorm import (
    GoldilocksFinalRmsnormClaimV3,
    GoldilocksFinalRmsnormProofV3,
    verify_goldilocks_final_rmsnorm_v3,
)
from verallm.proof_v3.goldilocks_gdn_composition import (
    GoldilocksGdnClaimV3,
    GoldilocksGdnCompositionProofV3,
    verify_goldilocks_gdn_composition_v3,
)
from verallm.proof_v3.goldilocks_mlp_composition import (
    GoldilocksMlpCompositionProofV3,
    GoldilocksMlpLinkClaimV3,
    verify_goldilocks_mlp_composition_v3,
)
from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
    MAX_GOLDILOCKS_MULTILINEAR_PCS_QUERY_COUNT_V3,
    pcs_query_count_v3,
)
from verallm.proof_v3.goldilocks_projection_composition import (
    GoldilocksProjectionClaimV3,
    GoldilocksProjectionCompositionProofV3,
    goldilocks_projection_input_cells_v3,
    goldilocks_projection_output_cells_v3,
    goldilocks_projection_x_row_squares_v3,
    verify_goldilocks_projection_composition_v3,
)
from verallm.proof_v3.goldilocks_residual_composition import (
    GoldilocksResidualClaimV3,
    GoldilocksResidualCompositionProofV3,
    verify_goldilocks_residual_composition_v3,
)
from verallm.proof_v3.goldilocks_rmsnorm_composition import (
    GoldilocksRmsnormArtifactV3,
    GoldilocksRmsnormCompositionProofV3,
    GoldilocksRmsnormLinkClaimV3,
    verify_goldilocks_rmsnorm_composition_v3,
)
from verallm.proof_v3.goldilocks_succinct_batch_opening import (
    BatchClaimCheckerV3,
    BatchOpeningCollectorV3,
)
from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
    pcs_coset_profile_v3,
)
from verallm.proof_v3.goldilocks_terminal_path import (
    GoldilocksTerminalPathProofV3,
    verify_goldilocks_terminal_path_v3,
)
from verallm.proof_v3.lean_projection_fold import (
    lean_projection_operation_key_v3,
)
from verallm.proof_v3.rational_bundle_adapter import (
    RationalBundleGeometryV3,
    apply_capture_kv_sections_v3,
)
from verallm.proof_v3.scored_calibration import ScoredCalibrationV3
from verallm.proof_v3.scored_attention_reference import GATE_FIXED_BITS
from verallm.proof_v3.succinct_attention_wire import (
    CaptureKvLayerSectionWireV3,
)


GOLDILOCKS_SELECTED_TRACE_ABI_V3: Final = (
    "execution.selected_trace.shared_goldilocks_opening.pallas_static.v5"
)
FULL_ATTENTION_LAYER_KIND_V3: Final = "full_attention"
GDN_LAYER_KIND_V3: Final = "gdn"
MAX_SELECTED_TRACE_LAYERS_V3: Final = 4
_DEFAULT_CORRIDOR_SIGMA_BITS: Final = scale_to_bits_v3(8.0)
_DEFAULT_CORRIDOR_CHI2_BITS: Final = scale_to_bits_v3(0.2)

__all__ = [
    "FULL_ATTENTION_LAYER_KIND_V3",
    "GDN_LAYER_KIND_V3",
    "GOLDILOCKS_SELECTED_TRACE_ABI_V3",
    "GoldilocksSelectedTraceAttentionContextV3",
    "GoldilocksSelectedTraceContextV3",
    "GoldilocksSelectedTraceProofV3",
    "GoldilocksSelectedTraceResultV3",
    "finalize_goldilocks_selected_trace_v3",
    "goldilocks_selected_trace_final_hidden_v3",
    "verify_goldilocks_selected_trace_v3",
]


def _fixed32(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    return value


@dataclass(frozen=True, slots=True)
class GoldilocksSelectedTraceAttentionContextV3:
    selected_layers: tuple[int, ...]
    calibration: ScoredCalibrationV3
    geometry: RationalBundleGeometryV3
    head_count: int
    kv_head_count: int
    candidate_rows: tuple[int, ...]
    key_count: int
    capture_roots_by_layer: object
    capture_binding: bytes
    execution_anchor_commitments: tuple[ExecutionAnchorCommitmentV3, ...]
    kv_commitment_indices_by_layer: object
    qkv_row_positions_by_layer: object
    runtime_semantics: AttentionRuntimeSemanticsV3
    anchor_encoding_id: str
    qkv_biases_by_layer: object = None
    projection_row_positions_by_layer: object = None
    heads_per_layer: int = 2
    row_samples: int = 8
    pcs_query_count: int = 16

    def __post_init__(self) -> None:
        layers = tuple(self.selected_layers)
        rows = tuple(self.candidate_rows)
        if not isinstance(self.capture_roots_by_layer, Mapping):
            raise ProofV3Error(
                "selected-trace capture roots are malformed"
            )
        if any(
            isinstance(layer, bool) or not isinstance(layer, int)
            for layer in self.capture_roots_by_layer
        ):
            raise ProofV3Error(
                "selected-trace capture-root layer is malformed"
            )
        capture_roots = {
            layer: tuple(roots)
            for layer, roots in self.capture_roots_by_layer.items()
        }
        if (
            set(capture_roots) != set(layers)
            or any(
                len(roots) not in (3, 4)
                or any(
                    not isinstance(root, bytes) or len(root) != 32
                    for root in roots
                )
                for roots in capture_roots.values()
            )
        ):
            raise ProofV3Error(
                "selected-trace capture-root inventory is malformed"
            )
        commitments = tuple(self.execution_anchor_commitments)
        if (
            not commitments
            or not all(
                isinstance(item, ExecutionAnchorCommitmentV3)
                for item in commitments
            )
            or len({item.stage_id for item in commitments})
            != len(commitments)
        ):
            raise ProofV3Error(
                "selected-trace execution-anchor inventory is malformed"
            )
        if not isinstance(self.kv_commitment_indices_by_layer, Mapping):
            raise ProofV3Error(
                "selected-trace K/V anchor indices are malformed"
            )
        if any(
            isinstance(layer, bool) or not isinstance(layer, int)
            for layer in self.kv_commitment_indices_by_layer
        ):
            raise ProofV3Error(
                "selected-trace K/V anchor layer is malformed"
            )
        kv_indices = {
            layer: index
            for layer, index in self.kv_commitment_indices_by_layer.items()
        }
        if not isinstance(self.qkv_row_positions_by_layer, Mapping):
            raise ProofV3Error(
                "selected-trace QKV row positions are malformed"
            )
        qkv_positions = {
            layer: tuple(positions)
            for layer, positions
            in self.qkv_row_positions_by_layer.items()
        }
        try:
            projection_positions = (
                dict(qkv_positions)
                if self.projection_row_positions_by_layer is None
                else {
                    int(layer): tuple(int(position) for position in positions)
                    for layer, positions
                    in self.projection_row_positions_by_layer.items()
                }
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ProofV3Error(
                "selected-trace projection row positions are malformed"
            ) from exc
        try:
            qkv_biases = (
                {
                    layer: ((), 0)
                    for layer in layers
                }
                if self.qkv_biases_by_layer is None
                else {
                    int(layer): (tuple(values), int(scale_bits))
                    for layer, (values, scale_bits)
                    in self.qkv_biases_by_layer.items()
                }
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ProofV3Error(
                "selected-trace QKV bias inventory is malformed"
            ) from exc
        if (
            not layers
            or layers != tuple(sorted(set(layers)))
            or not isinstance(self.calibration, ScoredCalibrationV3)
            or not isinstance(self.geometry, RationalBundleGeometryV3)
            or isinstance(self.head_count, bool)
            or not isinstance(self.head_count, int)
            or self.head_count <= 0
            or isinstance(self.kv_head_count, bool)
            or not isinstance(self.kv_head_count, int)
            or self.kv_head_count <= 0
            or self.head_count % self.kv_head_count
            or not rows
            or rows != tuple(sorted(set(rows)))
            or isinstance(self.key_count, bool)
            or not isinstance(self.key_count, int)
            or self.key_count <= 0
            or not isinstance(
                self.runtime_semantics,
                AttentionRuntimeSemanticsV3,
            )
            or self.anchor_encoding_id not in {"fp16.v1", "bf16.v1"}
            or self.heads_per_layer <= 0
            or self.row_samples <= 0
            or self.pcs_query_count <= 0
            or set(kv_indices) != set(layers)
            or set(qkv_positions) != set(layers)
            or set(projection_positions) != set(layers)
            or set(qkv_biases) != set(layers)
            or any(
                bool(values) != bool(scale_bits)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not -128 <= value <= 127
                    for value in values
                )
                for values, scale_bits in qkv_biases.values()
            )
            or any(
                not positions
                or len(positions) != len(set(positions))
                or any(
                    isinstance(position, bool)
                    or not isinstance(position, int)
                    or not 0 <= position < self.key_count
                    for position in positions
                )
                for positions in qkv_positions.values()
            )
            or any(
                not positions
                or len(positions) != len(set(positions))
                or any(position < 0 for position in positions)
                for positions in projection_positions.values()
            )
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(commitments)
                for index in kv_indices.values()
            )
            or any(
                commitments[index].stage_id
                != f"l{layer}.attention_kv_output"
                for layer, index in kv_indices.items()
            )
        ):
            raise ProofV3Error(
                "selected-trace attention context is malformed"
            )
        _fixed32(self.capture_binding, "attention capture binding")
        object.__setattr__(self, "selected_layers", layers)
        object.__setattr__(self, "candidate_rows", rows)
        object.__setattr__(
            self,
            "execution_anchor_commitments",
            commitments,
        )
        object.__setattr__(
            self,
            "capture_roots_by_layer",
            MappingProxyType(capture_roots),
        )
        object.__setattr__(
            self,
            "kv_commitment_indices_by_layer",
            MappingProxyType(kv_indices),
        )
        object.__setattr__(
            self,
            "qkv_row_positions_by_layer",
            MappingProxyType(qkv_positions),
        )
        object.__setattr__(
            self,
            "projection_row_positions_by_layer",
            MappingProxyType(projection_positions),
        )
        object.__setattr__(
            self,
            "qkv_biases_by_layer",
            MappingProxyType(qkv_biases),
        )


@dataclass(frozen=True, slots=True)
class GoldilocksSelectedTraceContextV3:
    validator_binding_digest: bytes
    capture_base_binding_digest: bytes
    capture_chain_digest: bytes
    validator_nonce: bytes
    selected_layer_kinds: tuple[tuple[int, str], ...]
    projection_claims: tuple[GoldilocksProjectionClaimV3, ...]
    residual_claims: tuple[GoldilocksResidualClaimV3, ...]
    attention: GoldilocksSelectedTraceAttentionContextV3 | None
    gdn_claims: tuple[GoldilocksGdnClaimV3, ...]
    gdn_semantics: GdnRuntimeSemanticsV3 | None
    rmsnorm_claims: tuple[GoldilocksRmsnormLinkClaimV3, ...]
    rmsnorm_artifacts: tuple[GoldilocksRmsnormArtifactV3, ...]
    mlp_claims: tuple[GoldilocksMlpLinkClaimV3, ...]
    bottom_claim: GoldilocksBottomAnchorClaimV3
    bottom_token_ids: tuple[int, ...]
    signed_artifacts: EconomicVerifiedArtifactsV3
    terminal_binding: EconomicLmHeadCatalogBindingV3
    final_hidden_oracle: EconomicOracleCommitmentV3
    final_hidden_row: int
    observed_token: int
    final_rmsnorm_claim: GoldilocksFinalRmsnormClaimV3
    final_rmsnorm_artifact: GoldilocksRmsnormArtifactV3
    pcs_query_count: int = 16

    def __post_init__(self) -> None:
        for value, name in (
            (self.validator_binding_digest, "selected-trace binding"),
            (self.capture_base_binding_digest, "capture base binding"),
            (self.capture_chain_digest, "capture chain digest"),
            (self.validator_nonce, "selected-trace nonce"),
        ):
            _fixed32(value, name)
        layer_kinds = tuple(self.selected_layer_kinds)
        layers = tuple(layer for layer, _kind in layer_kinds)
        if (
            not layer_kinds
            or len(layer_kinds) > MAX_SELECTED_TRACE_LAYERS_V3
            or layers != tuple(sorted(set(layers)))
            or any(
                kind not in {
                    FULL_ATTENTION_LAYER_KIND_V3,
                    GDN_LAYER_KIND_V3,
                }
                for _layer, kind in layer_kinds
            )
            or not isinstance(
                self.signed_artifacts,
                EconomicVerifiedArtifactsV3,
            )
            or not isinstance(
                self.terminal_binding,
                EconomicLmHeadCatalogBindingV3,
            )
            or not isinstance(
                self.final_hidden_oracle,
                EconomicOracleCommitmentV3,
            )
            or isinstance(self.final_hidden_row, bool)
            or not isinstance(self.final_hidden_row, int)
            or self.final_hidden_row < 0
            or isinstance(self.observed_token, bool)
            or not isinstance(self.observed_token, int)
            or self.observed_token < 0
            or isinstance(self.pcs_query_count, bool)
            or not isinstance(self.pcs_query_count, int)
            or not 1
            <= self.pcs_query_count
            <= MAX_GOLDILOCKS_MULTILINEAR_PCS_QUERY_COUNT_V3
            or (
                self.attention is not None
                and self.attention.pcs_query_count
                != self.pcs_query_count
            )
        ):
            raise ProofV3Error("selected-trace context is malformed")
        object.__setattr__(self, "selected_layer_kinds", layer_kinds)
        for name in (
            "projection_claims",
            "residual_claims",
            "gdn_claims",
            "rmsnorm_claims",
            "rmsnorm_artifacts",
            "mlp_claims",
            "bottom_token_ids",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class GoldilocksSelectedTraceProofV3:
    bottom: GoldilocksBottomAnchorProofV3
    projection: GoldilocksProjectionCompositionProofV3
    residual: GoldilocksResidualCompositionProofV3
    attention_sections: tuple[CaptureKvLayerSectionWireV3, ...]
    attention_anchor_lane_reveals: tuple[
        EconomicExecutionAnchorLaneRevealV3,
        ...,
    ]
    attention_query_heads: tuple[tuple[int, int, int, bytes], ...]
    gdn: GoldilocksGdnCompositionProofV3 | None
    rmsnorm: GoldilocksRmsnormCompositionProofV3
    mlp: GoldilocksMlpCompositionProofV3
    terminal: GoldilocksTerminalPathProofV3
    final_rmsnorm: GoldilocksFinalRmsnormProofV3
    terminal_opening: object
    attention_capture_roots_by_layer: tuple[
        tuple[int, tuple[bytes, ...]],
        ...,
    ] = ()
    rmsnorm_weight_rows: tuple[
        tuple[str, EconomicWeightRowRevealV3],
        ...,
    ] = ()
    projection_bias_rows: tuple[
        tuple[str, EconomicWeightRowRevealV3],
        ...,
    ] = ()

    def __post_init__(self) -> None:
        sections = tuple(self.attention_sections)
        lane_reveals = tuple(self.attention_anchor_lane_reveals)
        query_heads = tuple(self.attention_query_heads)
        capture_roots = tuple(
            (layer, tuple(roots))
            for layer, roots in self.attention_capture_roots_by_layer
        )
        rmsnorm_rows = tuple(self.rmsnorm_weight_rows)
        bias_rows = tuple(self.projection_bias_rows)
        layers = tuple(int(section.layer) for section in sections)
        capture_layers = tuple(layer for layer, _roots in capture_roots)
        lane_keys = tuple(
            (
                reveal.commitment_index,
                reveal.opening.row_index,
                reveal.opening.lane_index,
            )
            for reveal in lane_reveals
        )
        query_keys = tuple(
            (layer, position, head)
            for layer, position, head, _raw in query_heads
        )
        if (
            not isinstance(self.bottom, GoldilocksBottomAnchorProofV3)
            or not isinstance(
                self.projection,
                GoldilocksProjectionCompositionProofV3,
            )
            or not isinstance(
                self.residual,
                GoldilocksResidualCompositionProofV3,
            )
            or not all(
                isinstance(section, CaptureKvLayerSectionWireV3)
                for section in sections
            )
            or layers != tuple(sorted(set(layers)))
            or not all(
                isinstance(
                    reveal,
                    EconomicExecutionAnchorLaneRevealV3,
                )
                for reveal in lane_reveals
            )
            or lane_keys != tuple(sorted(set(lane_keys)))
            or query_keys != tuple(sorted(set(query_keys)))
            or any(
                isinstance(layer, bool)
                or not isinstance(layer, int)
                or layer < 0
                for layer in capture_layers
            )
            or capture_layers != tuple(sorted(set(capture_layers)))
            or capture_layers != layers
            or any(
                len(roots) not in (3, 4)
                or any(
                    not isinstance(root, bytes) or len(root) != 32
                    for root in roots
                )
                for _layer, roots in capture_roots
            )
            or any(
                not isinstance(name, str)
                or not name
                or len(name) > 64
                or not isinstance(row, EconomicWeightRowRevealV3)
                for name, row in rmsnorm_rows
            )
            or tuple(name for name, _row in rmsnorm_rows)
            != tuple(sorted({name for name, _row in rmsnorm_rows}))
            or any(
                not isinstance(name, str)
                or not name
                or len(name) > 64
                or not isinstance(row, EconomicWeightRowRevealV3)
                for name, row in bias_rows
            )
            or tuple(name for name, _row in bias_rows)
            != tuple(sorted({name for name, _row in bias_rows}))
            or any(
                isinstance(layer, bool)
                or not isinstance(layer, int)
                or layer < 0
                or isinstance(position, bool)
                or not isinstance(position, int)
                or position < 0
                or isinstance(head, bool)
                or not isinstance(head, int)
                or head < 0
                or not isinstance(raw, bytes)
                or not raw
                or len(raw) > 1 << 20
                for layer, position, head, raw in query_heads
            )
            or (
                self.gdn is not None
                and not isinstance(
                    self.gdn,
                    GoldilocksGdnCompositionProofV3,
                )
            )
            or not isinstance(
                self.rmsnorm,
                GoldilocksRmsnormCompositionProofV3,
            )
            or not isinstance(self.mlp, GoldilocksMlpCompositionProofV3)
            or not isinstance(self.terminal, GoldilocksTerminalPathProofV3)
            or not isinstance(
                self.final_rmsnorm,
                GoldilocksFinalRmsnormProofV3,
            )
        ):
            raise ProofV3Error("selected-trace proof is malformed")
        _validate_shared_opening_shape(self.terminal_opening)
        _require_no_local_openings(
            self.projection,
            self.residual,
            sections,
        )
        object.__setattr__(self, "attention_sections", sections)
        object.__setattr__(
            self,
            "attention_anchor_lane_reveals",
            lane_reveals,
        )
        object.__setattr__(
            self,
            "attention_query_heads",
            query_heads,
        )
        object.__setattr__(
            self,
            "attention_capture_roots_by_layer",
            capture_roots,
        )
        object.__setattr__(
            self,
            "rmsnorm_weight_rows",
            rmsnorm_rows,
        )
        object.__setattr__(
            self,
            "projection_bias_rows",
            bias_rows,
        )


@dataclass(frozen=True, slots=True)
class GoldilocksSelectedTraceResultV3:
    attention_plans: tuple[object, ...]
    final_hidden_i8: tuple[int, ...]


def _validate_shared_opening_shape(payload: object) -> None:
    try:
        if (
            not isinstance(payload, dict)
            or set(payload) != {"claims", "batched"}
            or not isinstance(payload["claims"], dict)
            or not payload["claims"]
            or any(
                not isinstance(tag, str) or not tag
                for tag in payload["claims"]
            )
            or payload["batched"] is None
        ):
            raise ProofV3Error(
                "selected trace has a malformed shared opening"
            )
    except (KeyError, TypeError) as exc:
        raise ProofV3Error(
            "selected trace has a malformed shared opening"
        ) from exc


def _require_no_local_openings(projection, residual, sections) -> None:
    if projection.batch_opening is not None or residual.batch_opening is not None:
        raise ProofV3Error(
            "selected trace carries a duplicate component opening"
        )
    for section in sections:
        if tuple(section.openings) or section.batched_openings is not None:
            raise ProofV3Error(
                "selected attention section carries a local opening"
            )


def finalize_goldilocks_selected_trace_v3(
    *,
    bottom: GoldilocksBottomAnchorProofV3,
    projection: GoldilocksProjectionCompositionProofV3,
    residual: GoldilocksResidualCompositionProofV3,
    attention_sections=(),
    attention_anchor_lane_reveals=(),
    attention_query_heads=(),
    gdn: GoldilocksGdnCompositionProofV3 | None,
    rmsnorm: GoldilocksRmsnormCompositionProofV3,
    mlp: GoldilocksMlpCompositionProofV3,
    terminal: GoldilocksTerminalPathProofV3,
    final_rmsnorm: GoldilocksFinalRmsnormProofV3,
    collector: BatchOpeningCollectorV3,
    validator_nonce: bytes,
    fused=None,
    attention_capture_roots_by_layer=(),
    rmsnorm_weight_rows=(),
    projection_bias_rows=(),
) -> GoldilocksSelectedTraceProofV3:
    """Close the one caller-owned terminal opening after all components."""

    sections = tuple(attention_sections)
    _require_no_local_openings(projection, residual, sections)
    if not isinstance(collector, BatchOpeningCollectorV3):
        raise ProofV3Error("selected trace has no opening collector")
    expected_prefixes = {"projection/", "residual/", "terminal/"}
    if sections:
        expected_prefixes.add("attention/")
    tags = tuple(collector.claims)
    if any(not any(tag.startswith(prefix) for tag in tags) for prefix in expected_prefixes):
        raise ProofV3Error(
            "selected-trace collector is missing a component namespace"
        )
    with pcs_coset_profile_v3("chain"):
        terminal_opening = collector.prove_all_batched(
            validator_nonce=_fixed32(
                validator_nonce,
                "selected-trace nonce",
            ),
            fused=fused,
        )
    return GoldilocksSelectedTraceProofV3(
        bottom=bottom,
        projection=projection,
        residual=residual,
        attention_sections=sections,
        attention_anchor_lane_reveals=tuple(
            attention_anchor_lane_reveals
        ),
        attention_query_heads=tuple(attention_query_heads),
        gdn=gdn,
        rmsnorm=rmsnorm,
        mlp=mlp,
        terminal=terminal,
        final_rmsnorm=final_rmsnorm,
        terminal_opening=terminal_opening,
        attention_capture_roots_by_layer=tuple(
            attention_capture_roots_by_layer
        ),
        rmsnorm_weight_rows=tuple(rmsnorm_weight_rows),
        projection_bias_rows=tuple(projection_bias_rows),
    )


def goldilocks_selected_trace_final_hidden_v3(
    proof: GoldilocksTerminalPathProofV3,
    oracle: EconomicOracleCommitmentV3,
) -> tuple[int, ...]:
    """Recover the one transported final-hidden row without duplicating it."""

    if (
        not isinstance(proof, GoldilocksTerminalPathProofV3)
        or not isinstance(oracle, EconomicOracleCommitmentV3)
        or proof.final_hidden_opening.value_mode != VALUE_MODE_INT8
        or proof.final_hidden_opening.values is None
    ):
        raise ProofV3VerificationError(
            "selected trace has no canonical final-hidden row"
        )
    width = oracle_leaf_width_v3(oracle.col_count)
    col_pad = 1 << max(0, (oracle.col_count - 1).bit_length())
    values = tuple(proof.final_hidden_opening.values)
    if (
        col_pad % width
        or len(values) != col_pad
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not -128 <= value <= 127
            for value in values
        )
    ):
        raise ProofV3VerificationError(
            "selected trace final-hidden transport is malformed"
        )
    return values[: oracle.col_count]


def _merge_registry(target, extra) -> None:
    statements, commitments = target
    extra_statements, extra_commitments = extra
    if set(extra_statements) != set(extra_commitments):
        raise ProofV3VerificationError(
            "selected-trace component registry is inconsistent"
        )
    for tag, statement in extra_statements.items():
        if tag in statements or tag in commitments:
            raise ProofV3VerificationError(
                "selected-trace component namespaces collide"
            )
        statements[tag] = statement
        commitments[tag] = extra_commitments[tag]


def _canonical_gdn_projection_rows_v3(row_map) -> tuple[int, ...]:
    """Return GDN compact rows in the projection claim's canonical order."""

    return tuple(sorted(row for _position, row in row_map))


def _validate_exact_inventory(
    proof: GoldilocksSelectedTraceProofV3,
    context: GoldilocksSelectedTraceContextV3,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    layer_kinds = dict(context.selected_layer_kinds)
    selected_layers = tuple(layer_kinds)
    attention_layers = tuple(
        layer
        for layer, kind in context.selected_layer_kinds
        if kind == FULL_ATTENTION_LAYER_KIND_V3
    )
    gdn_layers = tuple(
        layer
        for layer, kind in context.selected_layer_kinds
        if kind == GDN_LAYER_KIND_V3
    )
    if tuple(claim.layer_index for claim in context.residual_claims) != selected_layers:
        raise ProofV3VerificationError(
            "selected residual inventory is not exact"
        )
    expected_projection_inventory = tuple(
        (
            layer,
            role,
            lean_projection_operation_key_v3(
                layer_index=layer,
                projection=role,
            ),
        )
        for layer, kind in context.selected_layer_kinds
        for role in (
            ("qkv", "o", "gate_up", "down")
            if kind == FULL_ATTENTION_LAYER_KIND_V3
            else ("gdn_qkvz", "gdn_ba", "gdn_o", "gate_up", "down")
        )
    )
    actual_operations = tuple(
        claim.operation.operation_key for claim in context.projection_claims
    )
    expected_operations = tuple(
        operation
        for _layer, _role, operation in expected_projection_inventory
    )
    if actual_operations != expected_operations:
        raise ProofV3VerificationError(
            "selected projection inventory is not exact"
        )
    catalog = context.signed_artifacts.lean_projection_catalog
    if catalog is None or any(
        claim.operation != catalog.operation(claim.operation.operation_key)
        for claim in context.projection_claims
    ):
        raise ProofV3VerificationError(
            "selected projection inventory is not the authenticated catalog"
        )
    projection_indices = {
        (layer, role): index
        for index, (layer, role, _operation) in enumerate(
            expected_projection_inventory
        )
    }
    residual_by_layer = {
        claim.layer_index: claim for claim in context.residual_claims
    }
    mlp_by_layer = {
        claim.layer_index: claim for claim in context.mlp_claims
    }
    gdn_by_layer = {
        claim.layer_index: claim for claim in context.gdn_claims
    }
    for claim, (layer, role, _operation) in zip(
        context.projection_claims,
        expected_projection_inventory,
        strict=True,
    ):
        entry_name = f"l{layer}.{role}"
        entry = context.signed_artifacts.entry(entry_name)
        bias_name = f"{entry_name}_bias"
        has_bias = context.signed_artifacts.has_entry(bias_name)
        scale_bits = (
            claim.weight_scale_bits
            or (
                claim.runtime.weight_scale_bits
                if claim.runtime is not None
                else 0
            )
        )
        if (
            not entry.scale_bits
            or scale_bits != entry.scale_bits
        ):
            raise ProofV3VerificationError(
                "selected projection arithmetic is not the signed profile"
            )
        runtime_required = role != "qkv"
        runtime = claim.runtime
        if (
            claim.x_anchor is not None
            or runtime_required != (runtime is not None)
        ):
            raise ProofV3VerificationError(
                "selected projection runtime-output mode is not exact"
            )
        if runtime is not None:
            y_prefix = "attn_o" if role == "o" else role
            expected_columns = tuple(
                sorted(
                    {
                        column
                        for _row, column
                        in claim.consumer_output_cells
                    }
                )
            )
            expected_bias_scale = (
                context.signed_artifacts.entry(bias_name).scale_bits
                if has_bias
                else 0
            )
            if (
                runtime.y_oracle.oracle_id
                != f"l{layer}.{y_prefix}_y"
                or runtime.y_oracle.layer_index != layer
                or runtime.y_oracle.col_count != claim.operation.output_dim
                or runtime.y_anchor is not None
                or runtime.output_columns != expected_columns
                or runtime.weight_scale_bits != entry.scale_bits
                or runtime.weight_row_squares
                != tuple(
                    (
                        column,
                        context.signed_artifacts.weight_row_sq(
                            entry_name,
                            column,
                        ),
                    )
                    for column in expected_columns
                )
                or runtime.corridor_sigma_bits
                != (
                    context.signed_artifacts.manifest.corridor_sigma_bits
                    or _DEFAULT_CORRIDOR_SIGMA_BITS
                )
                or runtime.corridor_chi2_bits
                != (
                    context.signed_artifacts.manifest.corridor_chi2_bits
                    or _DEFAULT_CORRIDOR_CHI2_BITS
                )
                or runtime.corridor_kind != f"y_{role}"
                or runtime.input_columns
                or bool(runtime.bias_values) != has_bias
                or runtime.bias_scale_bits != expected_bias_scale
                or (
                    has_bias
                    and tuple(
                        column
                        for column, _value in runtime.bias_values
                    )
                    != expected_columns
                )
            ):
                raise ProofV3VerificationError(
                    "selected projection runtime corridor is not the signed "
                    "statement"
                )
        if layer_kinds[layer] == FULL_ATTENTION_LAYER_KIND_V3:
            try:
                residual = residual_by_layer[layer]
                mlp = mlp_by_layer[layer]
            except KeyError as exc:
                raise ProofV3VerificationError(
                    "selected full-attention reuse inventory is incomplete"
                ) from exc
            input_columns = {
                "qkv": residual.selected_columns,
                "o": (),
                "gate_up": residual.selected_columns,
                "down": mlp.selected_columns,
            }[role]
            expected_inputs = {
                (row, column)
                for row in claim.selected_rows
                for column in input_columns
            }
            if (
                role == "o"
                and not claim.consumer_input_cells
            ) or (
                role != "o"
                and set(claim.consumer_input_cells) != expected_inputs
            ):
                raise ProofV3VerificationError(
                    "selected full-attention projection input consumers "
                    "are not exact"
                )
            if role in {"o", "down"}:
                expected_outputs = {
                    (row, column)
                    for row in claim.selected_rows
                    for column in residual.selected_columns
                }
                if set(claim.consumer_output_cells) != expected_outputs:
                    raise ProofV3VerificationError(
                        "selected residual projection output consumers "
                        "are not exact"
                    )
            elif role == "gate_up":
                intermediate = claim.operation.output_dim // 2
                expected_outputs = {
                    (row, column)
                    for row in claim.selected_rows
                    for column in (
                        *mlp.selected_columns,
                        *(
                            intermediate + item
                            for item in mlp.selected_columns
                        ),
                    )
                }
                if set(claim.consumer_output_cells) != expected_outputs:
                    raise ProofV3VerificationError(
                        "selected MLP gate/up output consumers are not exact"
                    )
        else:
            try:
                residual = residual_by_layer[layer]
                mlp = mlp_by_layer[layer]
                gdn = gdn_by_layer[layer]
                signed_gdn = context.gdn_semantics.layer_for(layer)
            except (AttributeError, KeyError, ProofV3Error) as exc:
                raise ProofV3VerificationError(
                    "selected GDN reuse inventory is incomplete"
                ) from exc
            parameters = signed_gdn.parameters().replay_parameters()
            qkvz_columns, ba_columns, gdn_o_columns = (
                economic_gdn_runtime_columns_v3(
                    parameters=parameters,
                    selected_value_heads=gdn.selected_value_heads,
                )
            )
            # Projection claims use canonical compact-row order.  GDN replay
            # keeps the same row identifiers in absolute-position order, so
            # compare the canonical row inventory rather than its replay
            # traversal order.
            gdn_rows = _canonical_gdn_projection_rows_v3(gdn.row_map)
            if (
                claim.selected_rows != gdn_rows
            ):
                raise ProofV3VerificationError(
                    "selected GDN projection has the wrong row map"
                )
            input_columns = {
                "gdn_qkvz": residual.selected_columns,
                "gdn_ba": residual.selected_columns,
                "gdn_o": gdn_o_columns,
                "gate_up": residual.selected_columns,
                "down": mlp.selected_columns,
            }[role]
            expected_inputs = {
                (row, column)
                for row in claim.selected_rows
                for column in input_columns
            }
            if set(claim.consumer_input_cells) != expected_inputs:
                raise ProofV3VerificationError(
                    "selected GDN projection input consumers are not exact"
                )
            output_columns = {
                "gdn_qkvz": qkvz_columns,
                "gdn_ba": ba_columns,
                "gdn_o": residual.selected_columns,
                "gate_up": (
                    *mlp.selected_columns,
                    *(
                        claim.operation.output_dim // 2 + item
                        for item in mlp.selected_columns
                    ),
                ),
                "down": residual.selected_columns,
            }[role]
            expected_outputs = {
                (row, column)
                for row in claim.selected_rows
                for column in output_columns
            }
            if set(claim.consumer_output_cells) != expected_outputs:
                raise ProofV3VerificationError(
                    "selected GDN projection output consumers are not exact"
                )
    for layer in attention_layers:
        residual = residual_by_layer[layer]
        mlp = mlp_by_layer[layer]
        if (
            residual.attention_projection_index
            != projection_indices[(layer, "o")]
            or residual.down_projection_index
            != projection_indices[(layer, "down")]
            or mlp.gate_up_projection_index
            != projection_indices[(layer, "gate_up")]
            or mlp.down_projection_index
            != projection_indices[(layer, "down")]
        ):
            raise ProofV3VerificationError(
                "selected full-attention projection links are not exact"
            )
    for layer in gdn_layers:
        residual = residual_by_layer[layer]
        mlp = mlp_by_layer[layer]
        gdn = gdn_by_layer[layer]
        if (
            residual.attention_projection_index
            != projection_indices[(layer, "gdn_o")]
            or residual.down_projection_index
            != projection_indices[(layer, "down")]
            or gdn.qkvz_projection_index
            != projection_indices[(layer, "gdn_qkvz")]
            or gdn.ba_projection_index
            != projection_indices[(layer, "gdn_ba")]
            or gdn.gdn_o_projection_index
            != projection_indices[(layer, "gdn_o")]
            or mlp.gate_up_projection_index
            != projection_indices[(layer, "gate_up")]
            or mlp.down_projection_index
            != projection_indices[(layer, "down")]
        ):
            raise ProofV3VerificationError(
                "selected GDN projection links are not exact"
            )
    if context.attention is not None:
        qkv_layers = set()
        for claim in context.projection_claims:
            layer = claim.operation.operation_key.layer_idx
            if (
                layer not in attention_layers
                or claim.operation.operation_key
                != lean_projection_operation_key_v3(
                    layer_index=layer,
                    projection="qkv",
                )
            ):
                continue
            if (
                not claim.consumer_output_cells
            ):
                raise ProofV3VerificationError(
                    "selected attention QKV projection has no consumer binding"
                )
            qkv_layers.add(layer)
        if qkv_layers != set(attention_layers):
            raise ProofV3VerificationError(
                "selected attention QKV consumer inventory is incomplete"
            )
    for residual in context.residual_claims:
        expected_role = (
            "o"
            if layer_kinds[residual.layer_index]
            == FULL_ATTENTION_LAYER_KIND_V3
            else "gdn_o"
        )
        if residual.attention_projection_role != expected_role:
            raise ProofV3VerificationError(
                "selected residual role disagrees with the signed layer kind"
            )
        layer = residual.layer_index
        residual_in_anchor = residual.residual_in.anchor
        residual_out_anchor = residual.residual_out.anchor
        if (
            (
                layer == 0
                and (
                    residual_in_anchor is not None
                    or context.bottom_claim.residual_oracle
                    != residual.residual_in.oracle
                    or {
                        row for _position, row
                        in context.bottom_claim.row_map
                    }
                    != set(residual.selected_rows)
                )
            )
            or (
                layer != 0
                and (
                    residual_in_anchor is None
                    or residual_in_anchor.commitment.stage_id
                    != f"l{layer - 1}.residual_out"
                )
            )
            or residual.mid_residual.anchor is not None
            or residual_out_anchor is None
            or residual_out_anchor.commitment.stage_id
            != f"l{layer}.residual_out"
        ):
            raise ProofV3VerificationError(
                "selected residual anchors do not match the lean "
                "checkpoint topology"
            )
    if (
        (context.attention is None) != (not attention_layers)
        or (
            context.attention is not None
            and context.attention.selected_layers != attention_layers
        )
        or tuple(int(section.layer) for section in proof.attention_sections)
        != attention_layers
    ):
        raise ProofV3VerificationError(
            "selected attention inventory is not exact"
        )
    if not attention_layers and (
        proof.attention_anchor_lane_reveals
        or proof.attention_query_heads
    ):
        raise ProofV3VerificationError(
            "selected trace carries attention values without attention"
        )
    if (
        tuple(claim.layer_index for claim in context.gdn_claims) != gdn_layers
        or (proof.gdn is None) != (not gdn_layers)
        or (context.gdn_semantics is None) != (not gdn_layers)
    ):
        raise ProofV3VerificationError(
            "selected GDN inventory is not exact"
        )
    rmsnorm_keys = tuple(
        (claim.layer_index, claim.source_stage)
        for claim in context.rmsnorm_claims
    )
    expected_rmsnorm = tuple(
        (layer, stage)
        for layer in selected_layers
        for stage in ("residual_in", "mid_residual")
    )
    if (
        rmsnorm_keys != expected_rmsnorm
        or len(context.rmsnorm_artifacts)
        != len(context.rmsnorm_claims)
        or tuple(claim.layer_index for claim in context.mlp_claims)
        != selected_layers
        or len(context.bottom_token_ids)
        != len(context.bottom_claim.row_map)
        or context.final_rmsnorm_claim.final_hidden_oracle
        != context.final_hidden_oracle
        or context.final_rmsnorm_claim.final_hidden_row
        != context.final_hidden_row
    ):
        raise ProofV3VerificationError(
            "selected reuse/terminal inventory is not exact"
        )
    return attention_layers, gdn_layers


class _AuthenticatedAttentionAnchorViewV3:
    """Bind selected Q to projection S and K/V to the lean suffix anchor."""

    def __init__(
        self,
        *,
        attention: GoldilocksSelectedTraceAttentionContextV3,
        projection_proof: GoldilocksProjectionCompositionProofV3,
        projection_claims: tuple[GoldilocksProjectionClaimV3, ...],
        signed_artifacts: EconomicVerifiedArtifactsV3,
        lane_reveals: tuple[
            EconomicExecutionAnchorLaneRevealV3,
            ...,
        ],
        query_heads: tuple[tuple[int, int, int, bytes], ...],
    ) -> None:
        self._attention = attention
        self._commitments = attention.execution_anchor_commitments
        self._indices = dict(
            attention.kv_commitment_indices_by_layer
        )
        self._provided_lanes = {
            (
                reveal.commitment_index,
                reveal.opening.row_index,
                reveal.opening.lane_index,
            )
            for reveal in lane_reveals
        }
        self._used_lanes: set[tuple[int, int, int]] = set()
        self._openings = {}
        for reveal in lane_reveals:
            self._openings.setdefault(
                reveal.commitment_index,
                {},
            )[
                (
                    reveal.opening.row_index,
                    reveal.opening.lane_index,
                )
            ] = reveal.opening
        self._geometry = {}
        self._kv_geometry = {}
        self._params = {}
        self._query_cache = {}
        self._kv_cache = {}
        self._query_heads = {
            (int(layer), int(head), int(position)): raw
            for layer, position, head, raw in query_heads
        }
        self._provided_queries = set(self._query_heads)
        self._used_queries: set[tuple[int, int, int]] = set()
        self._projection_s = {}
        self._projection_x_sq = {}
        self._provided_projection_s = set()
        self._used_projection_s: set[tuple[int, int, int]] = set()
        self._projection_params = {}
        claims_by_key = {
            claim.operation.operation_key: (index, claim)
            for index, claim in enumerate(projection_claims)
        }
        for layer in attention.selected_layers:
            try:
                qkv_index, qkv = claims_by_key[
                    lean_projection_operation_key_v3(
                        layer_index=layer,
                        projection="qkv",
                    )
                ]
                _o_index, o_projection = claims_by_key[
                    lean_projection_operation_key_v3(
                        layer_index=layer,
                        projection="o",
                    )
                ]
            except KeyError as exc:
                raise ProofV3VerificationError(
                    "selected attention projection geometry is incomplete"
                ) from exc
            self._geometry[layer] = attention_anchor_geometry_v3(
                qkv_width=qkv.operation.output_dim,
                o_input_width=o_projection.operation.input_dim,
                query_heads=attention.head_count,
                kv_heads=attention.kv_head_count,
                head_dim=attention.geometry.head_dim,
                semantics=attention.runtime_semantics,
            )
            geometry = self._geometry[layer]
            self._kv_geometry[layer] = AttentionAnchorGeometryV3(
                query_heads=geometry.query_heads,
                kv_heads=geometry.kv_heads,
                head_dim=geometry.head_dim,
                qkv_width=2 * geometry.kv_heads * geometry.head_dim,
                q_block_width=0,
                k_block_offset=0,
                v_block_offset=geometry.kv_heads * geometry.head_dim,
                gated=geometry.gated,
            )
            self._params[layer] = tuple(
                params
                for params, _bounds in attention.calibration.heads_for(
                    layer
                )
            )
            try:
                position_by_row = dict(
                    zip(
                        qkv.selected_rows,
                        attention.qkv_row_positions_by_layer[layer],
                        strict=True,
                    )
                )
            except (KeyError, ValueError) as exc:
                raise ProofV3VerificationError(
                    "selected attention QKV row map is inconsistent"
                ) from exc
            for row, column, value in goldilocks_projection_output_cells_v3(
                projection_proof,
                projection_claims,
                claim_index=qkv_index,
            ):
                key = (layer, position_by_row[row], column)
                self._projection_s[key] = value
                self._provided_projection_s.add(key)
            for row, square in goldilocks_projection_x_row_squares_v3(
                projection_proof,
                projection_claims,
                claim_index=qkv_index,
            ):
                self._projection_x_sq[
                    (layer, position_by_row[row])
                ] = square
            entry_name = f"l{layer}.qkv"
            entry = signed_artifacts.entry(entry_name)
            sigma_bits = (
                signed_artifacts.manifest.corridor_sigma_bits
                or _DEFAULT_CORRIDOR_SIGMA_BITS
            )
            bias_values, bias_scale_bits = (
                attention.qkv_biases_by_layer[layer]
            )
            bias_name = f"{entry_name}_bias"
            has_bias = signed_artifacts.has_entry(bias_name)
            if (
                not entry.scale_bits
                or not sigma_bits
                or bool(bias_values) != has_bias
                or (
                    has_bias
                    and (
                        len(bias_values) != qkv.operation.output_dim
                        or bias_scale_bits
                        != signed_artifacts.entry(bias_name).scale_bits
                    )
                )
                or (not has_bias and bias_scale_bits)
            ):
                raise ProofV3VerificationError(
                    "selected attention projection corridor is not qualified"
                )
            self._projection_params[layer] = (
                qkv,
                signed_artifacts,
                bits_to_scale_v3(entry.scale_bits),
                bits_to_scale_v3(sigma_bits),
                tuple(bias_values),
                (
                    bits_to_scale_v3(bias_scale_bits)
                    if bias_scale_bits
                    else 0.0
                ),
            )

    @property
    def roots_by_layer(self):
        return {
            layer: self._commitments[index].root
            for layer, index in self._indices.items()
        }

    def _extract_kv(
        self,
        *,
        layer: int,
        position: int,
        byte_start: int,
        byte_length: int,
    ) -> bytes:
        try:
            commitment_index = self._indices[int(layer)]
            commitment = self._commitments[commitment_index]
            openings = self._openings.get(commitment_index, {})
        except (IndexError, KeyError, TypeError) as exc:
            raise ProofV3VerificationError(
                "selected attention anchor coordinate is malformed"
            ) from exc
        lane_bytes = execution_anchor_lane_bytes_v3(
            commitment.stage_id
        )
        lanes = required_execution_anchor_lanes_v3(
            byte_start=byte_start,
            byte_length=byte_length,
            lane_bytes=lane_bytes,
        )
        self._used_lanes.update(
            (commitment_index, int(position), lane)
            for lane in lanes
        )
        return extract_execution_anchor_range_v3(
            commitment=commitment,
            row_index=int(position),
            byte_start=byte_start,
            byte_length=byte_length,
            openings=openings,
        )

    def _bind_projection_values(
        self,
        *,
        layer: int,
        position: int,
        output_start: int,
        raw: bytes,
    ) -> None:
        try:
            values = decode_runtime_values_v3(
                raw,
                self._attention.anchor_encoding_id,
            )
            (
                claim,
                artifacts,
                weight_scale,
                sigma_cap,
                bias_values,
                bias_scale,
            ) = (
                self._projection_params[layer]
            )
            x_square = self._projection_x_sq[(layer, position)]
            x_scale = bits_to_scale_v3(claim.x_oracle.scale_bits)
        except (KeyError, ProofV3Error) as exc:
            raise ProofV3VerificationError(
                "selected attention runtime value has no projection source"
            ) from exc
        relative_step = (
            2.0 ** -7
            if self._attention.anchor_encoding_id == "bf16.v1"
            else 2.0 ** -10
        )
        for offset, captured in enumerate(values):
            output = output_start + offset
            key = (layer, position, output)
            try:
                surrogate = self._projection_s[key]
                weight_square = artifacts.weight_row_sq(
                    f"l{layer}.qkv",
                    output,
                )
            except (KeyError, ProofV3Error) as exc:
                raise ProofV3VerificationError(
                    "selected attention runtime value is outside the "
                    "projection consumer set"
                ) from exc
            variance = (
                (x_scale * x_scale / 12.0)
                * (weight_scale * weight_scale)
                * weight_square
                + (weight_scale * weight_scale / 12.0)
                * (x_scale * x_scale)
                * x_square
            )
            sigma = math.sqrt(variance)
            relative = (
                CORRIDOR_REL_COEFF_NUM_V3
                / CORRIDOR_REL_COEFF_DEN_V3
                * x_scale
                * weight_scale
                * math.sqrt(x_square * weight_square)
            )
            output_floor = max(
                abs(float(captured)) * relative_step,
                2.0 ** -24,
            )
            bias_value = (
                bias_values[output] * bias_scale
                if bias_values
                else 0.0
            )
            predicted = (
                surrogate * x_scale * weight_scale
                + bias_value
            )
            if abs(predicted - float(captured)) > (
                sigma_cap * sigma
                + relative
                + output_floor
                + 0.5 * bias_scale
            ):
                raise ProofV3VerificationError(
                    "selected attention runtime value is outside its signed "
                    "projection corridor"
                )
            self._used_projection_s.add(key)

    def _query(self, layer: int, head: int, position: int):
        key = (int(layer), int(head), int(position))
        if key in self._query_cache:
            return self._query_cache[key]
        if (
            key[0] not in self._geometry
            or key[2] not in self._attention.candidate_rows
        ):
            raise ProofV3VerificationError(
                "selected attention Q coordinate is outside the challenge"
            )
        geometry = self._geometry[key[0]]
        if not 0 <= key[1] < geometry.query_heads:
            raise ProofV3VerificationError(
                "selected attention Q head is outside the signed geometry"
            )
        try:
            raw = self._query_heads[key]
        except KeyError as exc:
            raise ProofV3VerificationError(
                "selected attention Q head has no carried runtime value"
            ) from exc
        head_width = geometry.head_dim * (2 if geometry.gated else 1)
        if len(raw) != head_width * 2:
            raise ProofV3VerificationError(
                "selected attention Q head has the wrong encoded width"
            )
        self._used_queries.add(key)
        self._bind_projection_values(
            layer=key[0],
            position=key[2],
            output_start=key[1] * head_width,
            raw=raw,
        )
        self._query_cache[key] = runtime_attention_q_head_quantized_v3(
            raw_head_bytes=raw,
            layer=key[0],
            position=key[2],
            head=key[1],
            geometry=geometry,
            semantics=self._attention.runtime_semantics,
            params_by_head=self._params[key[0]],
            encoding_id=self._attention.anchor_encoding_id,
        )
        return self._query_cache[key]

    def q13_head_row(self, layer: int, head: int, position: int):
        return self._query(layer, head, position)[0]

    def gate_fx_head_row(self, layer: int, head: int, position: int):
        gate = self._query(layer, head, position)[1]
        if gate is None:
            raise ProofV3VerificationError(
                "ungated attention was asked for gate values"
            )
        import numpy as np

        return tuple(
            int(value)
            for value in np.rint(
                np.asarray(gate, dtype=np.float64)
                * (1 << GATE_FIXED_BITS)
            ).tolist()
        )

    def kv_value(
        self,
        layer: int,
        tag: str,
        native_leaf: int,
        sp: int,
        dim: int,
    ) -> int:
        layer = int(layer)
        geometry = self._geometry.get(layer)
        if geometry is None or int(dim) != geometry.head_dim:
            raise ProofV3VerificationError(
                "selected attention K/V geometry is malformed"
            )
        kv_head, remainder = divmod(
            int(native_leaf),
            int(sp) * int(dim),
        )
        position, coordinate = divmod(remainder, int(dim))
        if kv_head >= geometry.kv_heads or position >= self._attention.key_count:
            return 0
        key = (layer, str(tag), kv_head, position)
        if key not in self._kv_cache:
            start, length = attention_anchor_head_byte_range_v3(
                geometry=self._kv_geometry[layer],
                tag=str(tag),
                head=kv_head,
            )
            raw = self._extract_kv(
                layer=layer,
                position=position,
                byte_start=start,
                byte_length=length,
            )
            self._kv_cache[key] = runtime_kv_head_quantized_v3(
                tag=str(tag),
                raw_head_bytes=raw,
                layer=layer,
                position=position,
                kv_head=kv_head,
                geometry=geometry,
                semantics=self._attention.runtime_semantics,
                params_by_head=self._params[layer],
                encoding_id=self._attention.anchor_encoding_id,
            )
        return int(self._kv_cache[key][coordinate])

    def bind_kv_projection_cells(self) -> None:
        """Bind the independent nonce-selected K/V corridor cells to S."""

        for layer, position, output in sorted(
            self._provided_projection_s - self._used_projection_s
        ):
            geometry = self._geometry[layer]
            if geometry.k_block_offset <= output < geometry.v_block_offset:
                suffix_column = output - geometry.k_block_offset
            elif (
                geometry.v_block_offset
                <= output
                < geometry.v_block_offset
                + geometry.kv_heads * geometry.head_dim
            ):
                suffix_column = (
                    geometry.kv_heads * geometry.head_dim
                    + output
                    - geometry.v_block_offset
                )
            else:
                raise ProofV3VerificationError(
                    "selected attention consumer cell is neither Q nor K/V"
                )
            raw = self._extract_kv(
                layer=layer,
                position=position,
                byte_start=suffix_column * 2,
                byte_length=2,
            )
            self._bind_projection_values(
                layer=layer,
                position=position,
                output_start=output,
                raw=raw,
            )

    def require_exact_inventory(self) -> None:
        if (
            self._provided_lanes != self._used_lanes
            or self._provided_queries != self._used_queries
            or self._provided_projection_s != self._used_projection_s
        ):
            raise ProofV3VerificationError(
                "selected attention runtime inventory is not exact"
            )


def _economic_ox_callback(
    *,
    proof: GoldilocksProjectionCompositionProofV3,
    claims: tuple[GoldilocksProjectionClaimV3, ...],
    attention: GoldilocksSelectedTraceAttentionContextV3,
):
    by_layer = {}
    provided = set()
    used = set()
    for index, claim in enumerate(claims):
        for layer in attention.selected_layers:
            if claim.operation.operation_key != lean_projection_operation_key_v3(
                layer_index=layer,
                projection="o",
            ):
                continue
            if (
                claim.operation.input_dim
                != attention.head_count * attention.geometry.head_dim
            ):
                raise ProofV3VerificationError(
                    "attention o-projection does not authenticate complete "
                    "selected input rows"
                )
            cells = {
                (row, column): value
                for row, column, value in goldilocks_projection_input_cells_v3(
                    proof,
                    claims,
                    claim_index=index,
                )
            }
            provided.update(
                (layer, row, column) for row, column in cells
            )
            by_layer[layer] = (
                claim,
                cells,
                bits_to_scale_v3(claim.x_oracle.scale_bits),
            )
    if set(by_layer) != set(attention.selected_layers):
        raise ProofV3VerificationError(
            "attention o-projection inventory is incomplete"
        )

    def _callback(layer: int, head: int, position: int):
        try:
            claim, cells, economic_scale = by_layer[int(layer)]
            row = dict(
                zip(
                    attention.projection_row_positions_by_layer[int(layer)],
                    claim.selected_rows,
                    strict=True,
                )
            )[int(position)]
            params = attention.calibration.heads_for(int(layer))[int(head)][0]
        except (IndexError, KeyError, TypeError) as exc:
            raise ProofV3VerificationError(
                "attention output bridge requested an unauthenticated row"
            ) from exc
        start = int(head) * attention.geometry.head_dim
        try:
            selected = tuple(
                cells[(row, column)]
                for column in range(
                    start,
                    start + attention.geometry.head_dim,
                )
            )
        except KeyError as exc:
            raise ProofV3VerificationError(
                "attention output bridge requested an unauthenticated cell"
            ) from exc
        used.update(
            (int(layer), row, column)
            for column in range(
                start,
                start + attention.geometry.head_dim,
            )
        )
        scale_num, scale_den = economic_scale.as_integer_ratio()
        denominator = scale_den * int(params.ox_num)
        multiplier = scale_num * (1 << int(params.ox_e))
        result = []
        for value in selected:
            numerator = int(value) * multiplier
            sign = -1 if numerator < 0 else 1
            quotient, remainder = divmod(abs(numerator), denominator)
            doubled = remainder << 1
            if doubled > denominator or (
                doubled == denominator and quotient & 1
            ):
                quotient += 1
            result.append(max(-127, min(127, sign * quotient)))
        return tuple(result)

    def _require_exact_inventory() -> None:
        if provided != used:
            raise ProofV3VerificationError(
                "selected attention o-projection input inventory is not exact"
            )

    return _callback, _require_exact_inventory


def verify_goldilocks_selected_trace_v3(
    proof: object,
    *,
    context: GoldilocksSelectedTraceContextV3,
) -> GoldilocksSelectedTraceResultV3:
    """Verify the complete selected trace in its signed PCS profile."""

    if not isinstance(context, GoldilocksSelectedTraceContextV3):
        raise ProofV3VerificationError(
            "selected-trace context has a wrong type"
        )
    with (
        pcs_query_count_v3(context.pcs_query_count),
        pcs_coset_profile_v3("chain"),
    ):
        return _verify_goldilocks_selected_trace_v3_impl(
            proof,
            context=context,
        )


def _verify_goldilocks_selected_trace_v3_impl(
    proof: object,
    *,
    context: GoldilocksSelectedTraceContextV3,
) -> GoldilocksSelectedTraceResultV3:
    """Verify the complete selected trace and its one shared opening."""

    try:
        if not isinstance(proof, GoldilocksSelectedTraceProofV3):
            raise ProofV3VerificationError(
                "selected-trace proof has a wrong type"
            )
        if not isinstance(context, GoldilocksSelectedTraceContextV3):
            raise ProofV3VerificationError(
                "selected-trace context has a wrong type"
            )
        attention_layers, gdn_layers = _validate_exact_inventory(
            proof,
            context,
        )
        checker = BatchClaimCheckerV3()
        registry = ({}, {})
        projection_registry = verify_goldilocks_projection_composition_v3(
            proof.projection,
            validator_binding_digest=context.validator_binding_digest,
            capture_base_binding_digest=(
                context.capture_base_binding_digest
            ),
            validator_nonce=context.validator_nonce,
            claims=context.projection_claims,
            external_checker=checker,
            checker_ns="projection/",
        )
        _merge_registry(registry, projection_registry)
        residual_registry = verify_goldilocks_residual_composition_v3(
            proof.residual,
            validator_binding_digest=context.validator_binding_digest,
            validator_nonce=context.validator_nonce,
            claims=context.residual_claims,
            projection_proof=proof.projection,
            projection_claims=context.projection_claims,
            external_checker=checker,
            checker_ns="residual/",
        )
        _merge_registry(registry, residual_registry)
        plans = ()
        if attention_layers:
            attention = context.attention
            anchor_view = _AuthenticatedAttentionAnchorViewV3(
                attention=attention,
                projection_proof=proof.projection,
                projection_claims=context.projection_claims,
                signed_artifacts=context.signed_artifacts,
                lane_reveals=proof.attention_anchor_lane_reveals,
                query_heads=proof.attention_query_heads,
            )
            grouped_aux = {
                layer: any(
                    tag.startswith(
                        f"attention/layer/{layer}/logup_aux/"
                    )
                    for tag in proof.terminal_opening["claims"]
                )
                for layer in attention_layers
            }
            ox_callback, require_exact_ox = _economic_ox_callback(
                proof=proof.projection,
                claims=context.projection_claims,
                attention=attention,
            )
            plans, attention_registry = apply_capture_kv_sections_v3(
                sections=proof.attention_sections,
                batched=True,
                anchor_backed=True,
                validator_nonce=context.validator_nonce,
                capture_chain_digest=context.capture_chain_digest,
                validator_binding_digest=context.validator_binding_digest,
                selected_layers=attention_layers,
                calibration=attention.calibration,
                geometry=attention.geometry,
                head_count=attention.head_count,
                n_kv=attention.kv_head_count,
                candidate_rows=attention.candidate_rows,
                key_count=attention.key_count,
                capture_roots_by_layer=attention.capture_roots_by_layer,
                capture_binding=attention.capture_binding,
                economic_ox8_head_row=ox_callback,
                anchor_roots_by_layer=anchor_view.roots_by_layer,
                anchor_kv_value=anchor_view.kv_value,
                anchor_q13_head_row=anchor_view.q13_head_row,
                anchor_gate_fx_head_row=(
                    anchor_view.gate_fx_head_row
                    if attention.runtime_semantics.gated
                    else None
                ),
                anchor_integer_tolerance=(
                    attention.runtime_semantics.integer_tolerance
                ),
                heads_per_layer=attention.heads_per_layer,
                row_samples=attention.row_samples,
                pcs_query_count=attention.pcs_query_count,
                external_checker=checker,
                checker_ns="attention/",
                grouped_aux_by_layer=grouped_aux,
            )
            _merge_registry(registry, attention_registry)
            require_exact_ox()
            anchor_view.bind_kv_projection_cells()
            anchor_view.require_exact_inventory()
        final_hidden = goldilocks_selected_trace_final_hidden_v3(
            proof.terminal,
            context.final_hidden_oracle,
        )
        terminal_registry = verify_goldilocks_terminal_path_v3(
            proof.terminal,
            validator_binding_digest=context.validator_binding_digest,
            validator_nonce=context.validator_nonce,
            binding=context.terminal_binding,
            capture_base_binding_digest=(
                context.capture_base_binding_digest
            ),
            final_hidden_oracle=context.final_hidden_oracle,
            expected_final_hidden_row=context.final_hidden_row,
            final_hidden_i8=final_hidden,
            expected_observed_token=context.observed_token,
            checker=checker,
        )
        _merge_registry(registry, terminal_registry)
        with pcs_coset_profile_v3("chain"):
            checker.verify_all_batched(
                proof.terminal_opening,
                statements=registry[0],
                commitments=registry[1],
                validator_nonce=context.validator_nonce,
            )
        verify_goldilocks_bottom_anchor_v3(
            proof.bottom,
            claim=context.bottom_claim,
            expected_token_ids=context.bottom_token_ids,
            artifacts=context.signed_artifacts,
            capture_base_binding_digest=(
                context.capture_base_binding_digest
            ),
        )
        if gdn_layers:
            verify_goldilocks_gdn_composition_v3(
                proof.gdn,
                claims=context.gdn_claims,
                projection_proof=proof.projection,
                projection_claims=context.projection_claims,
                semantics=context.gdn_semantics,
            )
        verify_goldilocks_rmsnorm_composition_v3(
            proof.rmsnorm,
            claims=context.rmsnorm_claims,
            artifacts=context.rmsnorm_artifacts,
            residual_proof=proof.residual,
            residual_claims=context.residual_claims,
            projection_proof=proof.projection,
            projection_claims=context.projection_claims,
        )
        verify_goldilocks_mlp_composition_v3(
            proof.mlp,
            claims=context.mlp_claims,
            projection_proof=proof.projection,
            projection_claims=context.projection_claims,
        )
        verify_goldilocks_final_rmsnorm_v3(
            proof.final_rmsnorm,
            claim=context.final_rmsnorm_claim,
            artifact=context.final_rmsnorm_artifact,
            final_hidden_i8=final_hidden,
        )
        return GoldilocksSelectedTraceResultV3(
            attention_plans=tuple(plans),
            final_hidden_i8=final_hidden,
        )
    except ProofV3VerificationError:
        raise
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        ProofV3Error,
    ) as exc:
        raise ProofV3VerificationError(
            "selected-trace proof is malformed"
        ) from exc
