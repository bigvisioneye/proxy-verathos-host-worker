"""Validator-owned statement reconstruction for the selected-trace proof."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from verallm.proof_v3.attention_anchor_binding import (
    attention_anchor_geometry_v3,
)
from verallm.proof_v3.economic_challenge import (
    EconomicChallengeV3,
    economic_attention_candidate_positions_v3,
)
from verallm.proof_v3.economic_gdn_replay import (
    economic_gdn_runtime_columns_v3,
)
from verallm.proof_v3.economic_lm_head_catalog_fold import (
    EconomicLmHeadCatalogBindingV3,
)
from verallm.proof_v3.economic_wire import (
    EconomicOracleCommitmentV3,
    scale_to_bits_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.execution_anchor import ExecutionAnchorCommitmentV3
from verallm.proof_v3.goldilocks_bottom_anchor import (
    GoldilocksBottomAnchorClaimV3,
)
from verallm.proof_v3.goldilocks_final_rmsnorm import (
    GoldilocksFinalRmsnormClaimV3,
)
from verallm.proof_v3.goldilocks_gdn_composition import (
    GoldilocksGdnClaimV3,
)
from verallm.proof_v3.goldilocks_mlp_composition import (
    GoldilocksMlpLinkClaimV3,
)
from verallm.proof_v3.goldilocks_projection_composition import (
    GoldilocksProjectionAnchorClaimV3,
    GoldilocksProjectionClaimV3,
    GoldilocksProjectionRuntimeClaimV3,
)
from verallm.proof_v3.goldilocks_residual_composition import (
    GoldilocksResidualClaimV3,
    GoldilocksResidualStageClaimV3,
)
from verallm.proof_v3.goldilocks_rmsnorm_composition import (
    GoldilocksRmsnormArtifactV3,
    GoldilocksRmsnormLinkClaimV3,
    GoldilocksRmsnormTargetV3,
)
from verallm.proof_v3.goldilocks_selected_trace import (
    FULL_ATTENTION_LAYER_KIND_V3,
    GDN_LAYER_KIND_V3,
    GoldilocksSelectedTraceAttentionContextV3,
    GoldilocksSelectedTraceContextV3,
)
from verallm.proof_v3.lean_projection_fold import (
    lean_projection_operation_key_v3,
)
from verallm.proof_v3.rational_bundle_adapter import (
    release_rational_geometry_v3,
)
from verallm.proof_v3.request import (
    execution_input_token_id_at_position_v3,
)


_ATTENTION_BINDING_DOMAIN = (
    b"VERATHOS/PROOF_V3/ANCHOR_ATTN_HELPER_BINDING/V1"
)
_DEFAULT_CORRIDOR_SIGMA_BITS = scale_to_bits_v3(8.0)
_DEFAULT_CORRIDOR_CHI2_BITS = scale_to_bits_v3(0.2)

__all__ = [
    "build_goldilocks_selected_trace_context_v3",
    "goldilocks_selected_trace_attention_binding_v3",
]


def goldilocks_selected_trace_attention_binding_v3(
    *,
    envelope_digest: bytes,
    capture_chain_digest: bytes,
) -> bytes:
    if (
        not isinstance(envelope_digest, bytes)
        or len(envelope_digest) != 32
        or not isinstance(capture_chain_digest, bytes)
        or len(capture_chain_digest) != 32
    ):
        raise ProofV3Error(
            "selected-trace attention binding inputs are malformed"
        )
    return hashlib.sha256(
        _ATTENTION_BINDING_DOMAIN
        + envelope_digest
        + capture_chain_digest
    ).digest()


def _exact_map(records, *, key, name: str):
    values = tuple(records)
    result = {}
    for value in values:
        item_key = key(value)
        if item_key in result:
            raise ProofV3VerificationError(
                f"selected-trace {name} inventory contains a duplicate"
            )
        result[item_key] = value
    return result


def _anchor_claim(
    *,
    commitment: ExecutionAnchorCommitmentV3,
    positions,
    encoding_id: str,
) -> GoldilocksProjectionAnchorClaimV3:
    return GoldilocksProjectionAnchorClaimV3(
        commitment=commitment,
        anchor_rows=tuple(int(position) for position in positions),
        source_column_offset=0,
        encoding_id=encoding_id,
    )


def _norm_artifact(*, name: str, reveal, artifacts):
    entry = artifacts.entry(name)
    if entry.out_dim != 1:
        raise ProofV3VerificationError(
            f"selected-trace norm {name!r} is not a vector row"
        )
    return GoldilocksRmsnormArtifactV3(
        norm_key=name,
        weight_i8=artifacts.verify_weight_row(
            name=name,
            reveal=reveal,
        ),
        weight_scale_bits=entry.scale_bits,
        semantics_id=artifacts.manifest.rms_norm_semantics_id,
        epsilon_bits=artifacts.manifest.rms_norm_epsilon_bits,
    )


def build_goldilocks_selected_trace_context_v3(
    *,
    attention_capture_roots_by_layer,
    rmsnorm_weight_rows,
    projection_bias_rows=(),
    envelope_digest: bytes,
    capture_base_binding_digest: bytes,
    capture_chain_digest: bytes,
    challenge: EconomicChallengeV3,
    layer_universe,
    layer_kinds: Mapping[int, str],
    row_layouts: Mapping[int, Sequence[tuple[int, int]]],
    attention_metadata,
    attention_kv_positions_by_layer: Mapping[int, Sequence[int]],
    artifacts,
    oracles,
    execution_anchors,
    anchor_encoding_id: str,
    prompt_token_ids,
    observed_output_token_ids,
) -> GoldilocksSelectedTraceContextV3:
    """Rebuild the complete proof statement without trusting miner indices."""

    if (
        not isinstance(challenge, EconomicChallengeV3)
        or anchor_encoding_id not in {"fp16.v1", "bf16.v1"}
    ):
        raise ProofV3VerificationError(
            "selected-trace context inputs are malformed"
        )
    layers = tuple(int(layer) for layer in layer_universe)
    kinds = {int(layer): str(kind) for layer, kind in layer_kinds.items()}
    selected = tuple(int(layer) for layer in challenge.selected_layer_indices)
    if (
        not layers
        or layers != tuple(sorted(set(layers)))
        or set(kinds) != set(layers)
        or selected != tuple(sorted(set(selected)))
        or any(layer not in layers for layer in selected)
        or any(
            kinds[layer] not in {
                FULL_ATTENTION_LAYER_KIND_V3,
                GDN_LAYER_KIND_V3,
            }
            for layer in layers
        )
    ):
        raise ProofV3VerificationError(
            "selected-trace layer inventory is malformed"
        )
    oracle_by_id = _exact_map(
        oracles,
        key=lambda item: item.oracle_id,
        name="oracle",
    )
    if not all(
        isinstance(item, EconomicOracleCommitmentV3)
        for item in oracle_by_id.values()
    ):
        raise ProofV3VerificationError(
            "selected-trace oracle inventory is malformed"
        )
    anchor_by_stage = _exact_map(
        execution_anchors,
        key=lambda item: item.stage_id,
        name="anchor",
    )
    if not all(
        isinstance(item, ExecutionAnchorCommitmentV3)
        for item in anchor_by_stage.values()
    ):
        raise ProofV3VerificationError(
            "selected-trace anchor inventory is malformed"
        )
    anchor_index_by_stage = {
        item.stage_id: index
        for index, item in enumerate(tuple(execution_anchors))
    }
    layouts = {
        int(layer): tuple((int(position), int(row)) for position, row in layout)
        for layer, layout in row_layouts.items()
    }
    if set(layouts) != set(selected):
        raise ProofV3VerificationError(
            "selected-trace row layouts do not cover the exact selection"
        )
    for layer, layout in layouts.items():
        rows = tuple(row for _position, row in layout)
        positions = tuple(position for position, _row in layout)
        if (
            not layout
            or rows != tuple(sorted(set(rows)))
            or len(set(positions)) != len(positions)
            or any(
                not 0 <= position < challenge.sequence_token_count
                for position in positions
            )
        ):
            raise ProofV3VerificationError(
                f"selected-trace row layout is malformed at layer {layer}"
            )

    catalog = artifacts.lean_projection_catalog
    terminal_binding = artifacts.lm_head_catalog_binding
    if catalog is None or not isinstance(
        terminal_binding,
        EconomicLmHeadCatalogBindingV3,
    ):
        raise ProofV3VerificationError(
            "selected-trace static catalogs are not authenticated"
        )

    attention_plans = {}
    attention_calibration = None
    attention_semantics = None
    attention_policy = None
    if attention_metadata is not None:
        try:
            attention_plans = _exact_map(
                attention_metadata["plans"],
                key=lambda plan: int(plan.layer),
                name="attention plan",
            )
            attention_calibration = attention_metadata["calibration"]
            attention_semantics = attention_metadata["semantics"]
            attention_policy = attention_metadata["calibration_set"].policy
        except (KeyError, TypeError, AttributeError) as exc:
            raise ProofV3VerificationError(
                "selected-trace attention metadata is malformed"
            ) from exc
    expected_attention_layers = tuple(
        layer for layer in selected if kinds[layer] == FULL_ATTENTION_LAYER_KIND_V3
    )
    if tuple(sorted(attention_plans)) != expected_attention_layers:
        raise ProofV3VerificationError(
            "selected-trace attention plans are not exact"
        )
    try:
        attention_kv_positions = {
            int(layer): tuple(int(position) for position in positions)
            for layer, positions
            in attention_kv_positions_by_layer.items()
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProofV3VerificationError(
            "selected-trace K/V row plan is malformed"
        ) from exc
    if set(attention_kv_positions) != set(expected_attention_layers):
        raise ProofV3VerificationError(
            "selected-trace K/V row plan is not exact"
        )
    # Geometry is carried in the authenticated attention metadata rather than
    # inferred from miner roots. All selectable attention layers must share it.
    if attention_plans:
        try:
            nh = int(attention_metadata["nh"])
            n_kv = int(attention_metadata["n_kv"])
            head_dim = int(attention_metadata["hd"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProofV3VerificationError(
                "selected-trace attention geometry is malformed"
            ) from exc
    else:
        nh = n_kv = head_dim = 0

    roles_by_kind = {
        FULL_ATTENTION_LAYER_KIND_V3: ("qkv", "o", "gate_up", "down"),
        GDN_LAYER_KIND_V3: (
            "gdn_qkvz",
            "gdn_ba",
            "gdn_o",
            "gate_up",
            "down",
        ),
    }
    expected_bias_names = tuple(
        sorted(
            f"l{layer}.{role}_bias"
            for layer in selected
            for role in roles_by_kind[kinds[layer]]
            if artifacts.has_entry(f"l{layer}.{role}_bias")
        )
    )
    bias_reveals = _exact_map(
        projection_bias_rows,
        key=lambda item: item[0],
        name="projection bias row",
    )
    if tuple(sorted(bias_reveals)) != expected_bias_names:
        raise ProofV3VerificationError(
            "selected-trace projection bias rows are not exact"
        )
    projection_biases = {}
    for bias_name in expected_bias_names:
        reveal = bias_reveals[bias_name][1]
        values = artifacts.verify_weight_row(
            name=bias_name,
            reveal=reveal,
        )
        entry = artifacts.entry(bias_name)
        if entry.out_dim != 1 or not entry.scale_bits:
            raise ProofV3VerificationError(
                "selected-trace projection bias artifact is malformed"
            )
        projection_biases[bias_name.removesuffix("_bias")] = (
            values,
            entry.scale_bits,
        )

    projection_claims = []
    projection_index = {}
    residual_columns = {}
    mlp_columns = {}
    gdn_columns = {}
    for layer in selected:
        layout = layouts[layer]
        rows = tuple(row for _position, row in layout)
        position_to_row = {position: row for position, row in layout}
        hidden = artifacts.dims(f"l{layer}.gate_up")[0]
        residual_columns[layer] = challenge.residual_cols_for(
            layer_index=layer,
            hidden_dim=hidden,
        )
        intermediate = artifacts.dims(f"l{layer}.down")[0]
        mlp_columns[layer] = challenge.mlp_cols_for(
            layer_index=layer,
            inter_dim=intermediate,
        )
        if kinds[layer] == FULL_ATTENTION_LAYER_KIND_V3:
            roles = ("qkv", "o", "gate_up", "down")
            plan = attention_plans[layer]
            qkv_in, qkv_out = artifacts.dims(f"l{layer}.qkv")
            o_in, _o_out = artifacts.dims(f"l{layer}.o")
            geometry = attention_anchor_geometry_v3(
                qkv_width=qkv_out,
                o_input_width=o_in,
                query_heads=nh,
                kv_heads=n_kv,
                head_dim=head_dim,
                semantics=attention_semantics,
            )
            try:
                plan_heads = tuple(int(head) for head in plan.heads)
                plan_positions = tuple(
                    int(position) for position in plan.row_positions
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise ProofV3VerificationError(
                    "selected-trace attention plan is malformed"
                ) from exc
            if (
                not plan_heads
                or plan_heads != tuple(sorted(set(plan_heads)))
                or any(not 0 <= head < nh for head in plan_heads)
                or not plan_positions
                or plan_positions != tuple(sorted(set(plan_positions)))
                or any(
                    position not in position_to_row
                    for position in plan_positions
                )
            ):
                raise ProofV3VerificationError(
                    "selected-trace attention plan is malformed"
                )
            native_heads = {
                head // geometry.group for head in plan_heads
            }
            query_cells = {
                (position_to_row[position], column)
                for position in plan_positions
                for head in plan_heads
                for column in range(
                    head * head_dim * (2 if geometry.gated else 1),
                    (head + 1) * head_dim * (2 if geometry.gated else 1),
                )
            }
            kv_positions = attention_kv_positions[layer]
            if (
                not kv_positions
                or kv_positions != tuple(sorted(set(kv_positions)))
                or any(
                    position < 0
                    or position >= challenge.context_token_count
                    or position not in position_to_row
                    for position in kv_positions
                )
            ):
                raise ProofV3VerificationError(
                    "selected-trace K/V row plan is malformed"
                )
            kv_cells = {
                (position_to_row[position], offset + native * head_dim + dim)
                for position in kv_positions
                for offset in (
                    geometry.k_block_offset,
                    geometry.v_block_offset,
                )
                for native in native_heads
                for dim in range(head_dim)
            }
            qkv_output_cells = tuple(sorted(query_cells | kv_cells))
            o_input_cells = tuple(
                sorted(
                    {
                        (position_to_row[position], column)
                        for position in plan_positions
                        for head in plan_heads
                        for column in range(
                            head * head_dim,
                            (head + 1) * head_dim,
                        )
                    }
                )
            )
            if qkv_in != hidden:
                raise ProofV3VerificationError(
                    "selected-trace QKV input width is inconsistent"
                )
        else:
            roles = ("gdn_qkvz", "gdn_ba", "gdn_o", "gate_up", "down")
            signed = artifacts.gdn_runtime_semantics.layer_for(layer)
            parameters = signed.parameters().replay_parameters()
            selected_heads = challenge.gdn_value_heads_for(
                layer_index=layer,
                num_key_heads=parameters.num_key_heads,
                num_value_heads=parameters.num_value_heads,
            )
            gdn_columns[layer] = (
                selected_heads,
                *economic_gdn_runtime_columns_v3(
                    parameters=parameters,
                    selected_value_heads=selected_heads,
                ),
            )
        for role in roles:
            operation = catalog.operation(
                lean_projection_operation_key_v3(
                    layer_index=layer,
                    projection=role,
                )
            )
            oracle_prefix = "attn_o" if role == "o" else role
            x_oracle = oracle_by_id[f"l{layer}.{oracle_prefix}_x"]
            s_oracle = oracle_by_id[f"l{layer}.{oracle_prefix}_s"]
            input_columns = {
                "qkv": residual_columns[layer],
                "o": (),
                "gdn_qkvz": residual_columns[layer],
                "gdn_ba": residual_columns[layer],
                "gdn_o": gdn_columns.get(layer, ((), (), (), ()))[3],
                "gate_up": residual_columns[layer],
                "down": mlp_columns[layer],
            }[role]
            if role == "qkv":
                output_cells = qkv_output_cells
            elif role in {"o", "gdn_o", "down"}:
                output_cells = tuple(
                    (row, column)
                    for row in rows
                    for column in residual_columns[layer]
                )
            elif role == "gate_up":
                half = operation.output_dim // 2
                output_cells = tuple(
                    (row, column)
                    for row in rows
                    for column in (
                        *mlp_columns[layer],
                        *(half + value for value in mlp_columns[layer]),
                    )
                )
            elif role == "gdn_qkvz":
                output_cells = tuple(
                    (row, column)
                    for row in rows
                    for column in gdn_columns[layer][1]
                )
            elif role == "gdn_ba":
                output_cells = tuple(
                    (row, column)
                    for row in rows
                    for column in gdn_columns[layer][2]
                )
            else:
                raise ProofV3VerificationError(
                    "selected-trace projection role is unsupported"
                )
            claim_rows = (
                tuple(
                    sorted(
                        {
                            row
                            for row, _column in output_cells
                        }
                    )
                )
                if role == "qkv"
                else rows
            )
            if role == "o":
                input_cells = o_input_cells
            else:
                input_cells = tuple(
                    (row, column)
                    for row in claim_rows
                    for column in input_columns
                )
            projection_index[(layer, role)] = len(projection_claims)
            output_columns = tuple(
                sorted({column for _row, column in output_cells})
            )
            runtime = None
            if role != "qkv":
                y_prefix = "attn_o" if role == "o" else role
                y_oracle = oracle_by_id[f"l{layer}.{y_prefix}_y"]
                entry_name = f"l{layer}.{role}"
                entry = artifacts.entry(entry_name)
                bias_values, bias_scale_bits = projection_biases.get(
                    entry_name,
                    ((), 0),
                )
                runtime = GoldilocksProjectionRuntimeClaimV3(
                    y_oracle=y_oracle,
                    y_anchor=None,
                    output_columns=output_columns,
                    weight_scale_bits=entry.scale_bits,
                    weight_row_squares=tuple(
                        (
                            column,
                            artifacts.weight_row_sq(
                                entry_name,
                                column,
                            ),
                        )
                        for column in output_columns
                    ),
                    corridor_sigma_bits=(
                        artifacts.manifest.corridor_sigma_bits
                        or _DEFAULT_CORRIDOR_SIGMA_BITS
                    ),
                    corridor_chi2_bits=(
                        artifacts.manifest.corridor_chi2_bits
                        or _DEFAULT_CORRIDOR_CHI2_BITS
                    ),
                    corridor_kind=f"y_{role}",
                    bias_values=(
                        tuple(
                            (column, bias_values[column])
                            for column in output_columns
                        )
                        if bias_values
                        else ()
                    ),
                    bias_scale_bits=bias_scale_bits,
                )
            projection_claims.append(
                GoldilocksProjectionClaimV3(
                    operation=operation,
                    x_oracle=x_oracle,
                    s_oracle=s_oracle,
                    selected_rows=claim_rows,
                    runtime=runtime,
                    weight_scale_bits=artifacts.entry(
                        f"l{layer}.{role}"
                    ).scale_bits,
                    consumer_input_cells=tuple(sorted(input_cells)),
                    consumer_output_cells=tuple(sorted(output_cells)),
                )
            )

    residual_claims = []
    for layer in selected:
        layout = layouts[layer]
        positions = tuple(position for position, _row in layout)
        rows = tuple(row for _position, row in layout)
        residual_in_anchor = (
            None
            if layer == 0
            else _anchor_claim(
                commitment=anchor_by_stage[f"l{layer - 1}.residual_out"],
                positions=positions,
                encoding_id=anchor_encoding_id,
            )
        )
        residual_claims.append(
            GoldilocksResidualClaimV3(
                layer_index=layer,
                selected_rows=rows,
                selected_columns=residual_columns[layer],
                residual_in=GoldilocksResidualStageClaimV3(
                    oracle=oracle_by_id[f"l{layer}.residual_in"],
                    anchor=residual_in_anchor,
                ),
                mid_residual=GoldilocksResidualStageClaimV3(
                    oracle=oracle_by_id[f"l{layer}.mid_residual"],
                    anchor=None,
                ),
                residual_out=GoldilocksResidualStageClaimV3(
                    oracle=oracle_by_id[f"l{layer}.residual_out"],
                    anchor=_anchor_claim(
                        commitment=anchor_by_stage[
                            f"l{layer}.residual_out"
                        ],
                        positions=positions,
                        encoding_id=anchor_encoding_id,
                    ),
                ),
                attention_projection_index=projection_index[
                    (
                        layer,
                        "o"
                        if kinds[layer] == FULL_ATTENTION_LAYER_KIND_V3
                        else "gdn_o",
                    )
                ],
                attention_projection_role=(
                    "o"
                    if kinds[layer] == FULL_ATTENTION_LAYER_KIND_V3
                    else "gdn_o"
                ),
                down_projection_index=projection_index[(layer, "down")],
            )
        )

    norm_reveals = _exact_map(
        rmsnorm_weight_rows,
        key=lambda item: item[0],
        name="RMSNorm row",
    )
    expected_norms = tuple(
        name
        for layer in selected
        for name in (
            f"l{layer}.input_norm",
            f"l{layer}.post_norm",
        )
    ) + ("final_norm",)
    if tuple(sorted(norm_reveals)) != tuple(sorted(expected_norms)):
        raise ProofV3VerificationError(
            "selected-trace RMSNorm rows are not exact"
        )
    rmsnorm_claims = []
    rmsnorm_artifacts = []
    mlp_claims = []
    for residual_index, layer in enumerate(selected):
        input_roles = (
            ("qkv",)
            if kinds[layer] == FULL_ATTENTION_LAYER_KIND_V3
            else ("gdn_qkvz", "gdn_ba")
        )
        input_claim = GoldilocksRmsnormLinkClaimV3(
            layer_index=layer,
            residual_claim_index=residual_index,
            source_stage="residual_in",
            norm_key=f"l{layer}.input_norm",
            targets=tuple(
                GoldilocksRmsnormTargetV3(
                    projection_index[(layer, role)],
                    role,
                )
                for role in input_roles
            ),
        )
        post_claim = GoldilocksRmsnormLinkClaimV3(
            layer_index=layer,
            residual_claim_index=residual_index,
            source_stage="mid_residual",
            norm_key=f"l{layer}.post_norm",
            targets=(
                GoldilocksRmsnormTargetV3(
                    projection_index[(layer, "gate_up")],
                    "gate_up",
                ),
            ),
        )
        rmsnorm_claims.extend((input_claim, post_claim))
        rmsnorm_artifacts.extend(
            (
                _norm_artifact(
                    name=input_claim.norm_key,
                    reveal=norm_reveals[input_claim.norm_key][1],
                    artifacts=artifacts,
                ),
                _norm_artifact(
                    name=post_claim.norm_key,
                    reveal=norm_reveals[post_claim.norm_key][1],
                    artifacts=artifacts,
                ),
            )
        )
        mlp_claims.append(
            GoldilocksMlpLinkClaimV3(
                layer_index=layer,
                gate_up_projection_index=projection_index[
                    (layer, "gate_up")
                ],
                down_projection_index=projection_index[(layer, "down")],
                selected_columns=mlp_columns[layer],
            )
        )

    gdn_claims = []
    for layer in selected:
        if kinds[layer] != GDN_LAYER_KIND_V3:
            continue
        selected_heads, _qkvz, _ba, _gdn_o = gdn_columns[layer]
        signed = artifacts.gdn_runtime_semantics.layer_for(layer)
        from verallm.proof_v3.gdn_decode_corridor import (
            derive_gdn_decode_corridor_for_challenge_v3,
        )

        checkpoint_plan = derive_gdn_decode_corridor_for_challenge_v3(
            challenge=challenge,
            semantics=signed,
        )
        if checkpoint_plan is None:
            conv_stage = f"l{layer}.gdn_conv_prompt_boundary"
            recurrent_stage = (
                f"l{layer}.gdn_recurrent_prompt_boundary"
            )
            start_state_row = 0
            end_state_row = None
        else:
            if tuple(
                position
                for position, _row in sorted(
                    layouts[layer],
                    key=lambda pair: pair[0],
                )
            ) != checkpoint_plan.sequence_positions:
                raise ProofV3VerificationError(
                    "selected-trace GDN rows do not match the nonce-selected "
                    "decode checkpoint window"
                )
            conv_stage = f"l{layer}.gdn_conv_decode_checkpoints"
            recurrent_stage = (
                f"l{layer}.gdn_recurrent_decode_checkpoints"
            )
            start_state_row = checkpoint_plan.start_checkpoint_row
            end_state_row = checkpoint_plan.end_checkpoint_row
        gdn_claims.append(
            GoldilocksGdnClaimV3(
                layer_index=layer,
                selected_value_heads=selected_heads,
                # Projection claims use canonical compact-row order, while
                # recurrent replay must consume rows in absolute token order.
                row_map=tuple(
                    sorted(layouts[layer], key=lambda pair: pair[0])
                ),
                conv_state_anchor=anchor_by_stage[conv_stage],
                recurrent_state_anchor=anchor_by_stage[recurrent_stage],
                qkvz_projection_index=projection_index[
                    (layer, "gdn_qkvz")
                ],
                ba_projection_index=projection_index[(layer, "gdn_ba")],
                gdn_o_projection_index=projection_index[(layer, "gdn_o")],
                start_state_row=start_state_row,
                end_state_row=end_state_row,
            )
        )

    if 0 in selected:
        bottom_map = layouts[0]
        bottom_oracle = oracle_by_id["l0.residual_in"]
    else:
        stamp_positions = economic_attention_candidate_positions_v3(
            context_token_count=challenge.context_token_count
        )
        bottom_map = tuple(
            (stamp_positions[row], row)
            for row in challenge.bottom_anchor_rows
        )
        bottom_oracle = oracle_by_id["response_stamp_input"]
    bottom_claim = GoldilocksBottomAnchorClaimV3(
        layer_index=0,
        residual_anchor=None,
        row_map=bottom_map,
        residual_oracle=bottom_oracle,
    )
    bottom_tokens = tuple(
        execution_input_token_id_at_position_v3(
            prompt_token_ids=prompt_token_ids,
            observed_output_token_ids=observed_output_token_ids,
            sequence_position=position,
        )
        for position, _row in bottom_map
    )

    attention_context = None
    if attention_plans:
        roots_by_layer = dict(attention_capture_roots_by_layer)
        kv_indices = {
            layer: anchor_index_by_stage[
                f"l{layer}.attention_kv_output"
            ]
            for layer in expected_attention_layers
        }
        attention_context = GoldilocksSelectedTraceAttentionContextV3(
            selected_layers=expected_attention_layers,
            calibration=attention_calibration,
            geometry=release_rational_geometry_v3(head_dim),
            head_count=nh,
            kv_head_count=n_kv,
            candidate_rows=tuple(
                challenge.attention_candidate_positions
            ),
            key_count=challenge.context_token_count,
            capture_roots_by_layer=roots_by_layer,
            capture_binding=goldilocks_selected_trace_attention_binding_v3(
                envelope_digest=envelope_digest,
                capture_chain_digest=capture_chain_digest,
            ),
            execution_anchor_commitments=tuple(execution_anchors),
            kv_commitment_indices_by_layer=kv_indices,
            qkv_row_positions_by_layer={
                layer: tuple(
                    {
                        row: position
                        for position, row in layouts[layer]
                    }[row]
                    for row in projection_claims[
                        projection_index[(layer, "qkv")]
                    ].selected_rows
                )
                for layer in expected_attention_layers
            },
            projection_row_positions_by_layer={
                layer: tuple(
                    position
                    for position, _row in layouts[layer]
                )
                for layer in expected_attention_layers
            },
            runtime_semantics=attention_semantics,
            anchor_encoding_id=anchor_encoding_id,
            qkv_biases_by_layer={
                layer: projection_biases.get(
                    f"l{layer}.qkv",
                    ((), 0),
                )
                for layer in expected_attention_layers
            },
            heads_per_layer=int(attention_policy.heads_per_layer),
            row_samples=int(attention_policy.row_samples),
            pcs_query_count=challenge.pcs_query_count,
        )

    audited_position = tuple(challenge.audited_decode_positions)
    if len(audited_position) != 1:
        raise ProofV3VerificationError(
            "selected-trace terminal selection is not singular"
        )
    final_hidden_row = audited_position[0]
    final_hidden_oracle = oracle_by_id["final_hidden"]
    last_layer = layers[-1]
    final_position = (
        challenge.context_token_count - 1 + final_hidden_row
    )
    final_claim = GoldilocksFinalRmsnormClaimV3(
        layer_index=last_layer,
        residual_anchor=_anchor_claim(
            commitment=anchor_by_stage[f"l{last_layer}.residual_out"],
            positions=(final_position,),
            encoding_id=anchor_encoding_id,
        ),
        residual_scale_bits=oracle_by_id[
            f"l{last_layer}.residual_out"
        ].scale_bits,
        final_hidden_oracle=final_hidden_oracle,
        final_hidden_row=final_hidden_row,
    )
    return GoldilocksSelectedTraceContextV3(
        validator_binding_digest=envelope_digest,
        capture_base_binding_digest=capture_base_binding_digest,
        capture_chain_digest=capture_chain_digest,
        validator_nonce=challenge.selection_seed,
        selected_layer_kinds=tuple(
            (layer, kinds[layer]) for layer in selected
        ),
        projection_claims=tuple(projection_claims),
        residual_claims=tuple(residual_claims),
        attention=attention_context,
        gdn_claims=tuple(gdn_claims),
        gdn_semantics=(
            artifacts.gdn_runtime_semantics if gdn_claims else None
        ),
        rmsnorm_claims=tuple(rmsnorm_claims),
        rmsnorm_artifacts=tuple(rmsnorm_artifacts),
        mlp_claims=tuple(mlp_claims),
        bottom_claim=bottom_claim,
        bottom_token_ids=bottom_tokens,
        signed_artifacts=artifacts,
        terminal_binding=terminal_binding,
        final_hidden_oracle=final_hidden_oracle,
        final_hidden_row=final_hidden_row,
        observed_token=int(observed_output_token_ids[final_hidden_row]),
        final_rmsnorm_claim=final_claim,
        final_rmsnorm_artifact=_norm_artifact(
            name="final_norm",
            reveal=norm_reveals["final_norm"][1],
            artifacts=artifacts,
        ),
        pcs_query_count=challenge.pcs_query_count,
    )
