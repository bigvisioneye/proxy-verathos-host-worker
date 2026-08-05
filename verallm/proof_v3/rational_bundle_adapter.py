"""Validator-side adapter for the RATIONAL (V2) succinct hard-canary
bundle.

The bundle is the SECURITY ADMISSION for the one full-context hard
canary: one V2 succinct layer section per signed-policy-selected
full-attention layer, all selections derived from the single
post-commit validator nonce with explicit subaudit-domain indices
(derive_reduction_bundle_v3 -- the SAME derivation the reduction
diagnostic uses, so challenge equality is exact).  The validator
constructs every statement itself from validator-owned data (signed
geometry + calibration, nonce-derived rows/heads, the section's
carried public tables) and verifies through
verify_rational_chunked_layer_v3: canonical contiguous chunk cover,
exact-once sums, aggregate == carried global total, and the
cross-multiplied rational output bridge against the validator's OWN
authenticated o_proj oracle rows.  Wire-supplied o rows never exist.
"""

from __future__ import annotations

import hashlib
import os as _os
from dataclasses import dataclass
from functools import wraps

from verallm.proof_v3.attention_reduction_audit import (
    derive_reduction_bundle_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
    pcs_query_count_v3,
)
from verallm.proof_v3.goldilocks_succinct_attention import (
    GoldilocksSuccinctAttentionStatementV3,
    verify_rational_chunked_layer_v3,
)
from verallm.proof_v3.succinct_attention_wire import (
    decode_rational_attention_proof_v3,
    decode_rational_bundle_wire_v3,
)

__all__ = [
    "RationalBundleGeometryV3",
    "apply_capture_kv_bundle_wire_v3",
    "apply_capture_kv_sections_v3",
    "apply_rational_bundle_wire_v3",
    "common_scored_slopes_v3",
    "rational_bundle_binding_v3",
]


def _pcs_query_scoped(function):
    @wraps(function)
    def scoped(*args, **kwargs):
        with pcs_query_count_v3(kwargs.get("pcs_query_count", 16)):
            return function(*args, **kwargs)

    return scoped

_BINDING_DOMAIN = b"VERATHOS/PROOF_V3/RATIONAL_BUNDLE_BINDING/V1"


def rational_bundle_binding_v3(*, validator_binding_digest: bytes,
                               capture_chain_digest: bytes) -> bytes:
    """The tile statements' binding digest: request digest folded with
    the pre-nonce capture chain digest.  BOTH sides derive it, so a
    proof from another request (stale/cross-request replay) fails the
    statement transcript STRUCTURALLY -- even when the nonce-derived
    coordinate selection happens to coincide (e.g. saturated row
    sampling), the statement digest cannot."""

    if len(validator_binding_digest) != 32 or len(
            capture_chain_digest) != 32:
        raise ProofV3Error(
            "bundle binding digests must be exactly 32 bytes")
    return hashlib.sha256(
        _BINDING_DOMAIN + validator_binding_digest
        + capture_chain_digest).digest()


def common_scored_slopes_v3(params_list):
    """Renormalize per-head (m_num, m_e) to ONE common exponent.

    Exact: mantissas shift left (value = m_num / 2^m_e is preserved
    bit-for-bit).  Returns (m_nums, m_e)."""

    params_list = tuple(params_list)
    if not params_list:
        raise ProofV3Error("common slopes need at least one head")
    m_e = max(int(p.m_e) for p in params_list)
    m_nums = tuple(
        int(p.m_num) << (m_e - int(p.m_e)) for p in params_list)
    return m_nums, m_e


def release_rational_geometry_v3(head_dim: int) -> "RationalBundleGeometryV3":
    """The ONE released SCORED_SCHEME_RATIONAL_V2 geometry.

    Every field except ``head_dim`` is a scheme constant fixed by the
    qualified release configuration (S13R): the signed ``attn_scheme``
    pins the scheme, the scheme pins these numbers.  Both the
    qualification harness and the economic adapter construct geometry
    through this factory so they can never drift."""

    from verallm.proof_v3.scored_attention_reference import (
        fixed_exp_table_v3,
    )

    return RationalBundleGeometryV3(
        head_dim=int(head_dim), qk_bits=13, v_bits=8, shift=16,
        exp_table=fixed_exp_table_v3(), score_bits=16, scale_bits=16,
        limb_bits=16)


@dataclass(frozen=True, slots=True)
class RationalBundleGeometryV3:
    """Validator-owned tile geometry for the V2 bundle statements.

    Production pins qk_bits=13 / score_bits=16 / the fixed universal
    exp table through the signed policy; tests may supply smaller
    CPU-cap geometries."""

    head_dim: int
    qk_bits: int
    v_bits: int
    shift: int
    exp_table: tuple
    score_bits: int
    scale_bits: int
    limb_bits: int


@_pcs_query_scoped
def apply_rational_bundle_wire_v3(*, wire: bytes, validator_nonce: bytes,
                                  capture_chain_digest: bytes,
                                  validator_binding_digest: bytes,
                                  selected_layers,
                                  calibration,
                                  geometry: RationalBundleGeometryV3,
                                  head_count: int,
                                  candidate_rows,
                                  key_count: int, chunk_len: int,
                                  oracle_ox8_row,
                                  heads_per_layer: int = 2,
                                  row_samples: int = 8,
                                  capture_kv_roots=None,
                                  pcs_query_count: int = 16):
    """Decode + verify ONE canary's V2 succinct bundle.  Fail-closed:
    every subaudit must verify; any failure fails the canary.

    ``selected_layers``: the signed stratified selection (sorted,
    distinct).  ``calibration.heads_for(layer)`` returns the signed
    per-head ``(ScoredHeadParamsV3, ScoredBridgeBoundsV3)`` pairs --
    the bounds MUST be the V2-frozen bridge bounds, never assumed from
    V1.  ``oracle_ox8_row(layer, position)`` returns the validator's
    AUTHENTICATED signed-scale int8 o_proj input row
    ``[head_count * head_dim]``.  ``candidate_rows``/``key_count``/
    ``chunk_len`` come from the authenticated pre-nonce envelope and
    the signed policy.  Returns the derived plan tuple."""

    layers = tuple(int(x) for x in selected_layers)
    tile_binding = rational_bundle_binding_v3(
        validator_binding_digest=validator_binding_digest,
        capture_chain_digest=capture_chain_digest)
    plans = derive_reduction_bundle_v3(
        validator_nonce=validator_nonce,
        capture_chain_digest=capture_chain_digest,
        profile_digest=calibration.digest,
        selected_layers=layers,
        head_count=head_count,
        candidate_rows=candidate_rows,
        chunk_count=(int(key_count) + int(chunk_len) - 1)
        // int(chunk_len),
        heads_per_layer=heads_per_layer, row_samples=row_samples)
    sections = decode_rational_bundle_wire_v3(
        wire, expected_layers=layers)
    for plan, section in zip(plans, sections, strict=True):
        if int(section.key_count) != int(key_count) or (
                int(section.chunk_len) != int(chunk_len)):
            raise ProofV3VerificationError(
                "bundle section geometry does not match the "
                "authenticated envelope")
        heads = calibration.heads_for(plan.layer)
        sel_params = [heads[h][0] for h in plan.heads]
        sel_bounds = [heads[h][1] for h in plan.heads]
        for params in sel_params:
            if params.head_dim != geometry.head_dim:
                raise ProofV3VerificationError(
                    "signed calibration head_dim does not match the "
                    "bundle geometry")
        m_nums, m_e = common_scored_slopes_v3(sel_params)
        positions = tuple(int(p) for p in plan.row_positions)
        n_chunks = (int(key_count) + int(chunk_len) - 1) // int(
            chunk_len)
        if len(section.chunk_proofs) != n_chunks:
            raise ProofV3VerificationError(
                "bundle section chunk count does not match the "
                "authenticated envelope")
        chunk_sections = []
        for index in range(n_chunks):
            base = index * int(chunk_len)
            count = min(int(chunk_len), int(key_count) - base)
            statement = GoldilocksSuccinctAttentionStatementV3(
                validator_binding_digest=tile_binding,
                head_count=len(plan.heads),
                token_count=len(positions),
                head_dim=geometry.head_dim,
                qk_bits=geometry.qk_bits, v_bits=geometry.v_bits,
                shift=geometry.shift, exp_table=geometry.exp_table,
                score_bits=geometry.score_bits,
                scale_bits=geometry.scale_bits,
                limb_bits=geometry.limb_bits,
                key_count=count, query_positions=positions,
                chunk_base=base,
                public_totals=section.public_totals,
                public_peaks=section.public_peaks,
                public_sel_count=section.chunk_sel_counts[index],
                scored=1, m_nums=m_nums, m_e=m_e, rational=1,
                capture_kv=1 if capture_kv_roots is not None else 0,
                pcs_query_count=pcs_query_count)
            proof = decode_rational_attention_proof_v3(
                section.chunk_proofs[index])
            chunk_sections.append((statement, proof))
        # -- AUTHENTICATED o_x rows (never from the wire) --
        dim = geometry.head_dim
        full_rows = {
            position: tuple(
                int(v) for v in oracle_ox8_row(plan.layer, position))
            for position in positions}
        for position, full in full_rows.items():
            if len(full) != head_count * dim:
                raise ProofV3VerificationError(
                    "authenticated o_x oracle row does not span "
                    "head_count * head_dim")
        ox8_rows = [
            [full_rows[position][head * dim:(head + 1) * dim]
             for position in positions]
            for head in plan.heads]
        verify_rational_chunked_layer_v3(
            sections=chunk_sections, validator_nonce=validator_nonce,
            expected_key_count=int(key_count),
            chunk_len=int(chunk_len),
            params_by_head={i: p for i, p in enumerate(sel_params)},
            bounds_by_head={i: b for i, b in enumerate(sel_bounds)},
            ox8_rows=ox8_rows, capture_kv_roots=capture_kv_roots,
            shared_openings=(tuple(section.layer_openings) or None))
    return plans


@_pcs_query_scoped
def apply_capture_kv_sections_v3(
        *, sections, batched: bool, anchor_backed: bool,
        validator_nonce: bytes,
        capture_chain_digest: bytes,
        validator_binding_digest: bytes,
        selected_layers, calibration,
        geometry: RationalBundleGeometryV3,
        head_count: int, n_kv: int,
        candidate_rows, key_count: int,
        capture_roots_by_layer,
        capture_binding: bytes,
        economic_ox8_head_row=None,
        anchor_roots_by_layer=None,
        anchor_kv_value=None,
        anchor_q13_head_row=None,
        anchor_gate_fx_head_row=None,
        anchor_integer_tolerance: int = 0,
        heads_per_layer: int = 2,
        row_samples: int = 8,
        pcs_query_count: int = 16,
        external_checker=None,
        checker_ns: str = "",
        grouped_aux_by_layer=None):
    """Verify decoded capture-KV sections.

    The long-context admission: one non-chunked rational tile per
    signed-selected layer; the miner's per-layer K/V PCS roots ride
    the wire but are only accepted because the kv-equality argument
    binds them to the PRE-NONCE capture roots from the
    validator-recovered envelope, in the padded NATIVE (nkv_pad, sp,
    d) scored-domain layout under ``capture_binding``.  Everything
    else follows the v2/v3 adapter: validator-constructed statements,
    exact challenge equality, aggregation invariants, and the
    cross-multiplied rational bridge.

    ATTENTION-ROW TRANSPORT: ``capture_roots_by_layer[layer] =
    (k_root, v_root, ox_root[, gate_root])``.  The bridge's o_x rows
    (and, on gated models, the fixed-point gate rows) arrive as wire
    multiproofs against the pre-nonce captured-row trees -- the
    padded (nh_pad, pool_pad, head_dim) cube over the candidate pool
    in candidate-row order.  Rows the pre-nonce roots do not commit
    cannot verify; no validator-side oracle exists.  Gatedness is
    read from the SIGNED calibration bounds, never from transport
    presence: a gated calibration without the gate root/opening, or a
    gate root/opening against ungated bounds, all fail closed.

    When ``economic_ox8_head_row`` is supplied, the same rational
    numerator/total is additionally bridged to an authenticated economic
    transition oracle.  The callback returns one signed-scale int8 head
    row for ``(layer, head, absolute_position)``.  The combined economic
    adapter requires this second bridge.

    ``external_checker`` is the selected-trace composition mode.  In
    that mode every section MUST omit its local terminal opening and
    all deferred claims are namespaced into the caller-owned checker.
    The function returns ``(plans, (statements, commitments))`` so the
    caller can verify one terminal opening shared with projection and
    the remaining selected execution relations.  Standalone mode
    retains the historical return value (the derived plans)."""

    from verallm.proof_v3.attention_reduction_audit import (
        derive_reduction_bundle_v3 as _derive,
    )
    from verallm.proof_v3.capture_kv_binding import (
        CaptureKvEqualityV3,
        derive_anchor_kv_equality_indices_v3,
        derive_anchor_q_mle_points_v3,
        derive_kv_equality_indices_v3,
        derive_row_opening_indices_v3,
        verify_kv_equality_v3,
        verify_row_transport_v3,
    )
    from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
    from verallm.proof_v3.goldilocks_succinct_attention import (
        _tile_digest,
        verify_goldilocks_succinct_attention_v3,
    )
    from verallm.proof_v3.goldilocks_succinct_batch_opening import (
        BatchClaimCheckerV3,
        NamespacedCheckerV3,
    )
    from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
        pcs_coset_profile_v3,
    )
    from verallm.proof_v3.rational_bundle_serving import (
        capture_kv_index_map_v3,
    )
    from verallm.proof_v3.scored_attention_reference import (
        GATE_FIXED_BITS,
        verify_output_bridge_rational_v3,
    )
    if head_count % int(n_kv):
        raise ProofV3VerificationError("n_kv must divide head_count")
    group = head_count // int(n_kv)
    layers = tuple(int(x) for x in selected_layers)
    tile_binding = rational_bundle_binding_v3(
        validator_binding_digest=validator_binding_digest,
        capture_chain_digest=capture_chain_digest)
    plans = _derive(
        validator_nonce=validator_nonce,
        capture_chain_digest=capture_chain_digest,
        profile_digest=calibration.digest, selected_layers=layers,
        head_count=head_count, candidate_rows=candidate_rows,
        chunk_count=1, heads_per_layer=heads_per_layer,
        row_samples=row_samples)
    if anchor_backed != (anchor_kv_value is not None):
        raise ProofV3VerificationError(
            "attention bundle anchor mode does not match the signed "
            "selection ABI"
        )
    sections = tuple(sections)
    if len(sections) != len(layers) or tuple(
            int(section.layer) for section in sections) != layers:
        raise ProofV3VerificationError(
            "capture-kv section set does not match the signed selection")
    external = external_checker is not None
    if external:
        if not batched:
            raise ProofV3VerificationError(
                "composed attention verification requires the "
                "chain-coset batched mode")
        if (
            not isinstance(grouped_aux_by_layer, dict)
            or set(grouped_aux_by_layer) != set(layers)
            or any(
                not isinstance(layer, int) or isinstance(layer, bool)
                for layer in grouped_aux_by_layer
            )
            or any(
                not isinstance(value, bool)
                for value in grouped_aux_by_layer.values()
            )
        ):
            raise ProofV3VerificationError(
                "external attention verification needs one grouped-aux "
                "decision for every selected layer")
    elif grouped_aux_by_layer is not None:
        raise ProofV3VerificationError(
            "standalone attention verification cannot take external "
            "grouped-aux decisions")
    all_statements = {}
    all_commitments = {}
    dim = geometry.head_dim
    half_p = GOLDILOCKS_MODULUS >> 1

    def _verify_section(plan, section):
        if int(section.key_count) != int(key_count):
            raise ProofV3VerificationError(
                "bundle section geometry does not match the "
                "authenticated envelope")
        heads = calibration.heads_for(plan.layer)
        sel_params = [heads[h][0] for h in plan.heads]
        sel_bounds = [heads[h][1] for h in plan.heads]
        for params in sel_params:
            if params.head_dim != dim:
                raise ProofV3VerificationError(
                    "signed calibration head_dim does not match the "
                    "bundle geometry")
        m_nums, m_e = common_scored_slopes_v3(sel_params)
        positions = tuple(int(p) for p in plan.row_positions)
        statement = GoldilocksSuccinctAttentionStatementV3(
            validator_binding_digest=tile_binding,
            head_count=len(plan.heads), token_count=len(positions),
            head_dim=dim, qk_bits=geometry.qk_bits,
            v_bits=geometry.v_bits, shift=geometry.shift,
            exp_table=geometry.exp_table,
            score_bits=geometry.score_bits,
            scale_bits=geometry.scale_bits,
            limb_bits=geometry.limb_bits, key_count=int(key_count),
            query_positions=positions, chunk_base=0,
            public_totals=section.public_totals,
            public_peaks=section.public_peaks,
            public_sel_count=section.public_sel_count,
            scored=1, m_nums=m_nums, m_e=m_e, rational=1,
            capture_kv=1, pcs_query_count=pcs_query_count)
        proof = decode_rational_attention_proof_v3(section.proof)
        td = _tile_digest(statement)
        k_root, v_root = section.kv_roots
        section_ns = (
            f"{checker_ns}layer/{int(plan.layer)}/"
            if external else ""
        )
        checker = (
            NamespacedCheckerV3(external_checker, section_ns)
            if external else BatchClaimCheckerV3()
        )
        if external:
            if tuple(section.openings) or section.batched_openings is not None:
                raise ProofV3VerificationError(
                    "composed attention section must not carry a local "
                    "terminal opening")
            aux_hint = bool(
                grouped_aux_by_layer[int(plan.layer)])
        elif batched:
            aux_hint = any(
                "logup_aux/" in tag
                for tag in section.batched_openings["claims"])
        else:
            aux_hint = any(
                "logup_aux/" in tag for tag, _p in section.openings)
        registry = verify_goldilocks_succinct_attention_v3(
            proof, statement=statement,
            validator_nonce=validator_nonce,
            external_checker=checker, checker_ns="",
            grouped_aux_hint=aux_hint,
            capture_kv_roots={"k": k_root, "v": v_root})
        if registry is None:
            raise ProofV3VerificationError(
                "capture-kv section verify returned no registry")
        stmts, comms = registry
        hp = statement.head_pad()
        tp = statement.token_pad()
        sp = 1 << max(0, (int(key_count) - 1).bit_length())
        env_roots = capture_roots_by_layer[int(plan.layer)]
        if anchor_backed:
            if anchor_q13_head_row is None:
                raise ProofV3VerificationError(
                    "anchor-backed attention bundle has no Q runtime binding"
                )
            import hashlib

            q_values = [0] * (hp * tp * dim)
            for head_slot, head in enumerate(plan.heads):
                for row_slot, position in enumerate(positions):
                    row = tuple(
                        int(value)
                        for value in anchor_q13_head_row(
                            int(plan.layer),
                            int(head),
                            int(position),
                        )
                    )
                    if len(row) != dim:
                        raise ProofV3VerificationError(
                            "anchor-backed Q row has the wrong head dimension"
                        )
                    base = (head_slot * tp + row_slot) * dim
                    q_values[base:base + dim] = [
                        value % GOLDILOCKS_MODULUS for value in row
                    ]
            commitments_digest = hashlib.sha256(
                b"".join(proof.column_commitments)
            ).digest()
            q_points = derive_anchor_q_mle_points_v3(
                tile_digest=td,
                anchor_root=anchor_roots_by_layer[int(plan.layer)],
                attention_commitments_digest=commitments_digest,
                validator_nonce=validator_nonce,
                variable_count=(len(q_values) - 1).bit_length(),
            )

            def _mle(point):
                work = list(q_values)
                for coordinate in point:
                    z = int(coordinate) % GOLDILOCKS_MODULUS
                    work = [
                        (
                            work[2 * index]
                            + z
                            * (
                                work[2 * index + 1]
                                - work[2 * index]
                            )
                        )
                        % GOLDILOCKS_MODULUS
                        for index in range(len(work) // 2)
                    ]
                return work[0]

            for point in q_points:
                checker.expect("q", point, _mle(point))
        index_map = capture_kv_index_map_v3(
            heads=plan.heads, group=group, sp=sp, d=dim)
        nkv_pad = 1 << max(0, (int(n_kv) - 1).bit_length())
        for slot, (tag, pcs_root, env_root) in enumerate(
                (("k", k_root, env_roots[0]),
                 ("v", v_root, env_roots[1]))):
            if anchor_kv_value is None:
                idx = derive_kv_equality_indices_v3(
                    tile_digest=td, capture_root=env_root,
                    pcs_root=pcs_root, validator_nonce=validator_nonce,
                    leaf_count=hp * sp * dim,
                    count=len(section.eq_indices[slot]))
            else:
                if anchor_roots_by_layer is None:
                    raise ProofV3VerificationError(
                        "anchor-backed K/V verification has no pre-nonce roots"
                    )
                try:
                    anchor_root = anchor_roots_by_layer[
                        int(plan.layer)
                    ]
                except (KeyError, TypeError) as exc:
                    raise ProofV3VerificationError(
                        "anchor-backed K/V verification has no layer root"
                    ) from exc
                idx = derive_anchor_kv_equality_indices_v3(
                    tile_digest=td,
                    anchor_root=anchor_root,
                    pcs_root=pcs_root,
                    validator_nonce=validator_nonce,
                    layer=int(plan.layer),
                    tag=tag,
                    leaf_count=hp * sp * dim,
                    count=len(section.eq_indices[slot]),
                )
            # canonical capture leaves + expected values through the
            # PUBLIC map: every tile sample mapping to one capture
            # leaf must claim ONE consistent value
            by_leaf: dict = {}
            for i_tile, val in zip(
                    idx, section.eq_values[slot], strict=False):
                leaf = index_map(i_tile)
                if by_leaf.setdefault(leaf, int(val)) != int(val):
                    raise ProofV3VerificationError(
                        "capture-kv samples claim inconsistent "
                        "values for one capture leaf")
            order = tuple(sorted(by_leaf))
            verify_kv_equality_v3(
                equality=CaptureKvEqualityV3(
                    tag=tag, indices=section.eq_indices[slot],
                    values=section.eq_values[slot],
                    capture_opening=section.eq_openings[slot]),
                capture_root=env_root,
                capture_binding=capture_binding, heads=hp, rows=sp,
                dim=dim, pcs_statement=stmts[tag],
                expected_indices=idx, checker=checker,
                capture_indices=order,
                capture_values=tuple(by_leaf[j] for j in order),
                capture_leaf_count=nkv_pad * sp * dim)
            if anchor_kv_value is not None:
                tolerance = int(anchor_integer_tolerance)
                if not 0 <= tolerance <= 2:
                    raise ProofV3VerificationError(
                        "anchor-backed K/V integer tolerance is invalid"
                    )
                for native_leaf in order:
                    claimed = int(by_leaf[native_leaf])
                    if claimed > half_p:
                        claimed -= GOLDILOCKS_MODULUS
                    expected = int(
                        anchor_kv_value(
                            int(plan.layer),
                            tag,
                            int(native_leaf),
                            int(sp),
                            int(dim),
                        )
                    )
                    if abs(claimed - expected) > tolerance:
                        raise ProofV3VerificationError(
                            "attention K/V PCS value is detached from the "
                            "pre-nonce raw QKV execution anchor"
                        )
        if external:
            for tag, statement_entry in stmts.items():
                full_tag = section_ns + tag
                if full_tag in all_statements:
                    raise ProofV3VerificationError(
                        "composed attention statement tags collide")
                all_statements[full_tag] = statement_entry
                all_commitments[full_tag] = comms[tag]
        elif batched:
            checker.verify_all_batched(
                section.batched_openings, statements=stmts,
                commitments=comms, validator_nonce=validator_nonce)
        else:
            checker.verify_all(
                dict(section.openings), statements=stmts,
                commitments=comms, validator_nonce=validator_nonce)
        # -- aggregation invariants (single whole-range chunk) --
        if any(int(v) != 1 for v in section.public_sel_count):
            raise ProofV3VerificationError(
                "capture-kv selector counts do not cover every row "
                "exactly once")
        if tuple(int(v) for v in proof.chunk_totals) != tuple(
                int(v) for v in section.public_totals):
            raise ProofV3VerificationError(
                "capture-kv proven totals disagree with the "
                "authenticated global totals")
        numerators = [
            [0] * dim for _ in range(hp * tp)]
        for i in range(hp * tp):
            for dd in range(dim):
                x = int(proof.partial_out[i * dim + dd])
                numerators[i][dd] = (
                    x - GOLDILOCKS_MODULUS if x > half_p else x)
        # -- attention-row transport + the rational bridge --
        # gatedness is a property of the SIGNED bounds (they were
        # measured under the gated relation, whose units differ by
        # 2^GATE_FIXED_BITS) -- neither side can choose: a gated
        # calibration without the gate root/opening would silently
        # accept anything, and a gate root/opening against ungated
        # bounds is a configuration fault.  All fail closed.
        section_gated = any(
            bool(getattr(b, "gated", False)) for b in sel_bounds)
        if len(env_roots) < 3:
            raise ProofV3VerificationError(
                "capture envelope does not carry the attention-row "
                "capture roots")
        if section_gated and len(env_roots) < 4:
            raise ProofV3VerificationError(
                "gated calibration requires the pre-nonce gate "
                "capture root")
        if not section_gated and len(env_roots) > 3:
            raise ProofV3VerificationError(
                "gate capture root supplied for an ungated "
                "calibration")
        if len(section.row_openings) != (2 if section_gated else 1):
            raise ProofV3VerificationError(
                "bundle section row transport does not match the "
                "signed gatedness")
        row_idx, row_leaves, pool_pad = derive_row_opening_indices_v3(
            heads=plan.heads, positions=positions,
            candidate_rows=candidate_rows, head_count=head_count,
            head_dim=dim)
        ox_vals = verify_row_transport_v3(
            opening=section.row_openings[0],
            capture_root=env_roots[2],
            capture_binding=capture_binding,
            expected_indices=row_idx, leaf_count=row_leaves,
            value_min=-127, value_max=127)
        gate_vals = None
        if section_gated:
            gate_vals = verify_row_transport_v3(
                opening=section.row_openings[1],
                capture_root=env_roots[3],
                capture_binding=capture_binding,
                expected_indices=row_idx, leaf_count=row_leaves,
                value_min=0, value_max=1 << GATE_FIXED_BITS)
        pool = tuple(int(p) for p in candidate_rows)
        pool_slot = {p: pool.index(p) for p in positions}

        def _transport_row(vals, head, position):
            base = (int(head) * pool_pad
                    + pool_slot[int(position)]) * dim
            return tuple(vals[base + dd] for dd in range(dim))

        for slot, head in enumerate(plan.heads):
            params = sel_params[slot]
            bounds = sel_bounds[slot]
            n_rows = [
                numerators[slot * tp + t] for t in range(
                    len(positions))]
            t_rows = [
                int(section.public_totals[slot * tp + t])
                for t in range(len(positions))]
            ox_rows = [
                _transport_row(ox_vals, head, position)
                for position in positions]
            gate_rows = None
            if bool(getattr(bounds, "gated", False)):
                gate_rows = [
                    _transport_row(gate_vals, head, position)
                    for position in positions]
                if anchor_gate_fx_head_row is None and anchor_backed:
                    raise ProofV3VerificationError(
                        "gated anchor-backed attention has no runtime gate "
                        "binding"
                    )
                if anchor_gate_fx_head_row is not None:
                    anchored_gate_rows = [
                        tuple(
                            int(value)
                            for value in anchor_gate_fx_head_row(
                                int(plan.layer),
                                int(head),
                                int(position),
                            )
                        )
                        for position in positions
                    ]
                    if any(len(row) != dim for row in anchored_gate_rows):
                        raise ProofV3VerificationError(
                            "anchor-backed gate row has the wrong head "
                            "dimension"
                        )
                    if any(
                        abs(int(actual) - int(expected))
                        > int(anchor_integer_tolerance)
                        for actual_row, expected_row in zip(
                            gate_rows, anchored_gate_rows, strict=True
                        )
                        for actual, expected in zip(
                            actual_row, expected_row, strict=True
                        )
                    ):
                        raise ProofV3VerificationError(
                            "attention gate values are detached from the "
                            "pre-nonce raw QKV execution anchor"
                        )
            if _os.environ.get("VERATHOS_BRIDGE_DEBUG") == "1":
                print(
                    f"BRIDGE layer={plan.layer} head={head} "
                    f"sum|n|={sum(abs(v) for row in n_rows for v in row)} "
                    f"sum|t|={sum(abs(v) for v in t_rows)} "
                    f"sum|ox|={sum(abs(v) for row in ox_rows for v in row)} "
                    f"gated={gate_rows is not None} "
                    f"bounds={bounds}", flush=True)
            try:
                verify_output_bridge_rational_v3(
                    params=params, numerator_rows=n_rows,
                    total_rows=t_rows, ox8_rows=ox_rows, bounds=bounds,
                    gate_fx_rows=gate_rows)
            except ProofV3VerificationError as exc:
                raise ProofV3VerificationError(
                    f"attention output bridge failed at layer "
                    f"{plan.layer}, head {head}: {exc}"
                ) from exc
            if economic_ox8_head_row is not None:
                economic_rows = [
                    tuple(int(value) for value in economic_ox8_head_row(
                        int(plan.layer), int(head), int(position)))
                    for position in positions
                ]
                if any(len(row) != dim for row in economic_rows):
                    raise ProofV3VerificationError(
                        "economic o_x bridge row does not span head_dim")
                try:
                    verify_output_bridge_rational_v3(
                        params=params,
                        numerator_rows=n_rows,
                        total_rows=t_rows,
                        ox8_rows=economic_rows,
                        bounds=bounds,
                        gate_fx_rows=gate_rows,
                    )
                except ProofV3VerificationError as exc:
                    raise ProofV3VerificationError(
                        f"economic attention output bridge failed at layer "
                        f"{plan.layer}, head {head}: {exc}"
                    ) from exc

    with pcs_coset_profile_v3("chain" if batched else "v1"):
        for plan, section in zip(plans, sections, strict=True):
            _verify_section(plan, section)
    if external:
        return plans, (all_statements, all_commitments)
    return plans


def apply_capture_kv_bundle_wire_v3(
        *, wire: bytes,
        validator_nonce: bytes,
        capture_chain_digest: bytes,
        validator_binding_digest: bytes,
        selected_layers, calibration,
        geometry: RationalBundleGeometryV3,
        head_count: int, n_kv: int,
        candidate_rows, key_count: int,
        capture_roots_by_layer,
        capture_binding: bytes,
        economic_ox8_head_row=None,
        anchor_roots_by_layer=None,
        anchor_kv_value=None,
        anchor_q13_head_row=None,
        anchor_gate_fx_head_row=None,
        anchor_integer_tolerance: int = 0,
        heads_per_layer: int = 2,
        row_samples: int = 8,
        pcs_query_count: int = 16):
    """Decode and verify one standalone capture-KV bundle.

    The wire codecs retain their existing per-layer terminal-opening
    formats.  The selected-trace composition calls
    :func:`apply_capture_kv_sections_v3` directly after its parent
    envelope has decoded sections that intentionally omit those local
    openings.
    """

    from verallm.proof_v3.succinct_attention_wire import (
        _RATIONAL_BUNDLE_VERSION_ANCHOR_CAPTURE_KV,
        _RATIONAL_BUNDLE_VERSION_ANCHOR_CAPTURE_KV_BATCHED,
        _RATIONAL_BUNDLE_VERSION_CAPTURE_KV_BATCHED,
        capture_kv_bundle_wire_version,
        decode_anchor_capture_kv_bundle_wire_v3,
        decode_capture_kv_bundle_wire_v3,
        decode_capture_kv_bundle_wire_v5,
    )

    layers = tuple(int(layer) for layer in selected_layers)
    version = capture_kv_bundle_wire_version(wire)
    anchor_backed = version in (
        _RATIONAL_BUNDLE_VERSION_ANCHOR_CAPTURE_KV,
        _RATIONAL_BUNDLE_VERSION_ANCHOR_CAPTURE_KV_BATCHED,
    )
    batched = version in (
        _RATIONAL_BUNDLE_VERSION_CAPTURE_KV_BATCHED,
        _RATIONAL_BUNDLE_VERSION_ANCHOR_CAPTURE_KV_BATCHED,
    )
    if anchor_backed:
        sections = decode_anchor_capture_kv_bundle_wire_v3(
            wire, expected_layers=layers)
    elif batched:
        sections = decode_capture_kv_bundle_wire_v5(
            wire, expected_layers=layers)
    else:
        sections = decode_capture_kv_bundle_wire_v3(
            wire, expected_layers=layers)
    return apply_capture_kv_sections_v3(
        sections=sections,
        batched=batched,
        anchor_backed=anchor_backed,
        validator_nonce=validator_nonce,
        capture_chain_digest=capture_chain_digest,
        validator_binding_digest=validator_binding_digest,
        selected_layers=layers,
        calibration=calibration,
        geometry=geometry,
        head_count=head_count,
        n_kv=n_kv,
        candidate_rows=candidate_rows,
        key_count=key_count,
        capture_roots_by_layer=capture_roots_by_layer,
        capture_binding=capture_binding,
        economic_ox8_head_row=economic_ox8_head_row,
        anchor_roots_by_layer=anchor_roots_by_layer,
        anchor_kv_value=anchor_kv_value,
        anchor_q13_head_row=anchor_q13_head_row,
        anchor_gate_fx_head_row=anchor_gate_fx_head_row,
        anchor_integer_tolerance=anchor_integer_tolerance,
        heads_per_layer=heads_per_layer,
        row_samples=row_samples,
        pcs_query_count=pcs_query_count,
    )
