"""Validator-side adapter for the reduction-tree attention audit.

Transport fetch + fail-closed apply for the PRODUCTION hard-canary
attention path.  The canonical wire NEVER carries o_proj rows: this
adapter substitutes the validator's OWN authenticated ``l{layer}.
attn_o_x`` oracle rows (signed-scale int8) into the corridor check, so
the o-side of the bridge is validator-owned material end to end -- a
miner wire can only match the audit, never define any of its inputs.
"""

from __future__ import annotations

import dataclasses

from verallm.proof_v3.attention_reduction_audit import (
    derive_reduction_bundle_v3,
    derive_reduction_plan_v3,
    reduction_bridge_check_v3,
    verify_reduction_reveal_v3,
)
from verallm.proof_v3.attention_reduction_wire import (
    decode_reduction_bundle_wire_v3,
    decode_reduction_reveal_v3,
)
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError

__all__ = [
    "fetch_reduction_audit_wire_v3",
    "apply_reduction_audit_wire_v3",
    "fetch_reduction_bundle_wire_v3",
    "apply_reduction_bundle_wire_v3",
]


def fetch_reduction_audit_wire_v3(*, endpoint: str, request_id: str,
                                  validator_nonce: bytes,
                                  heads_per_layer: int = 2,
                                  row_samples: int = 8,
                                  uniform_chunk_samples: int = 3,
                                  mass_draws: int = 2,
                                  timeout: float = 60.0,
                                  verify_tls: bool = False) -> bytes:
    """POST the miner's reduction-audit route; returns the raw wire.

    Decoding + verification happen in ``apply_reduction_audit_wire_v3``
    (decode enforces every wire bound before any allocation)."""

    import httpx

    url = endpoint.rstrip("/") + "/proof_v3/reduction_audit"
    response = httpx.post(
        url,
        json={
            "request_id": request_id,
            "validator_nonce": validator_nonce.hex(),
            "heads_per_layer": int(heads_per_layer),
            "row_samples": int(row_samples),
            "uniform_chunk_samples": int(uniform_chunk_samples),
            "mass_draws": int(mass_draws),
        },
        timeout=timeout, verify=verify_tls)
    if response.status_code != 200:
        raise ProofV3Error(
            f"reduction audit fetch failed ({response.status_code}): "
            + response.text[:200])
    payload = response.json()
    try:
        return bytes.fromhex(payload["wire"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProofV3Error("reduction audit reply is malformed") from exc


def apply_reduction_audit_wire_v3(*, wire: bytes, validator_nonce: bytes,
                                  capture_chain_digest: bytes,
                                  expected_request_root: bytes,
                                  calibration,
                                  audited_layers,
                                  head_count: int, n_kv: int,
                                  candidate_rows,
                                  key_count: int, chunk_len: int,
                                  oracle_ox8_row,
                                  heads_per_layer: int = 2,
                                  row_samples: int = 8,
                                  uniform_chunk_samples: int = 3,
                                  mass_draws: int = 2):
    """Decode + verify one reduction-audit wire.  Fail-closed.

    Every verification input is VALIDATOR-OWNED: ``expected_request_
    root``/``capture_chain_digest``/``key_count`` from the authenticated
    pre-nonce envelope, ``calibration`` digest-checked against the
    signed manifest, ``audited_layers``/``candidate_rows``/``chunk_len``
    /``n_kv`` from the signed policy.  ``oracle_ox8_row(layer,
    row_position)`` must return the validator's AUTHENTICATED
    signed-scale int8 o_proj input row ``[head_count * head_dim]`` for
    that captured row -- wire-supplied o rows never exist in this path.
    Returns the derived plan on success; raises on ANY mismatch."""

    if head_count < 1 or n_kv < 1 or head_count % n_kv:
        raise ProofV3VerificationError(
            "invalid head geometry for the reduction audit")
    if int(chunk_len) < 1 or int(key_count) < 1:
        raise ProofV3VerificationError(
            "invalid audited context geometry")
    chunk_count = (int(key_count) + int(chunk_len) - 1) // int(chunk_len)
    plan = derive_reduction_plan_v3(
        validator_nonce=validator_nonce,
        capture_chain_digest=capture_chain_digest,
        profile_digest=calibration.digest,
        discriminative_layers=audited_layers,
        head_count=head_count,
        candidate_rows=candidate_rows,
        chunk_count=chunk_count,
        heads_per_layer=heads_per_layer, row_samples=row_samples,
        uniform_chunk_samples=uniform_chunk_samples,
        mass_draws=mass_draws)
    reveal, _echo = decode_reduction_reveal_v3(wire)
    _verify_one_reveal_v3(
        reveal=reveal, plan=plan,
        expected_request_root=expected_request_root,
        calibration=calibration, audited_layers=audited_layers,
        head_count=head_count, n_kv=n_kv,
        candidate_rows=candidate_rows, key_count=key_count,
        chunk_len=chunk_len, oracle_ox8_row=oracle_ox8_row)
    return plan


def _verify_one_reveal_v3(*, reveal, plan, expected_request_root,
                          calibration, audited_layers, head_count: int,
                          n_kv: int, candidate_rows, key_count: int,
                          chunk_len: int, oracle_ox8_row) -> None:
    """The shared fail-closed verification of ONE reveal against ONE
    validator-derived plan: authenticated o_x substitution + full
    reveal verification + signed bridge.  Used by the single-plan
    apply and by every subaudit of the bundled apply."""

    heads = calibration.heads_for(plan.layer)
    params_by_head = {h: heads[h][0] for h in range(head_count)}
    dims = {params_by_head[h].head_dim for h in range(head_count)}
    if len(dims) != 1:
        raise ProofV3VerificationError(
            "the o_x oracle row layout needs a uniform head_dim")
    (dim,) = dims
    # -- substitute the AUTHENTICATED o_proj rows (never from the wire) --
    full_rows = {
        int(position): tuple(
            int(v) for v in oracle_ox8_row(plan.layer, int(position)))
        for position in plan.row_positions}
    for position, full in full_rows.items():
        if len(full) != head_count * dim:
            raise ProofV3VerificationError(
                "authenticated o_x oracle row does not span "
                "head_count * head_dim")
    o_rows = {
        (head, position): full[head * dim:(head + 1) * dim]
        for head in plan.heads for position, full in full_rows.items()}
    reveal = dataclasses.replace(reveal, o_rows=o_rows)

    def _bridge(head, _position, summary, o_row):
        reduction_bridge_check_v3(
            params=heads[head][0], bounds=heads[head][1],
            summary=summary, ox8_row=o_row)

    verify_reduction_reveal_v3(
        reveal=reveal, plan=plan,
        expected_request_root=expected_request_root,
        audited_layers=audited_layers, head_count=head_count,
        candidate_rows=candidate_rows,
        expected_key_count=int(key_count), chunk_len=int(chunk_len),
        n_kv=int(n_kv),
        expected_profile_digest=calibration.digest,
        params_by_head=params_by_head,
        bridge_check=_bridge)


def fetch_reduction_bundle_wire_v3(*, endpoint: str, request_id: str,
                                   validator_nonce: bytes,
                                   heads_per_layer: int = 2,
                                   row_samples: int = 8,
                                   uniform_chunk_samples: int = 3,
                                   mass_draws: int = 2,
                                   timeout: float = 60.0,
                                   verify_tls: bool = False) -> bytes:
    """POST the miner's bundled reduction-audit route; raw wire back.

    Decoding + verification happen in apply_reduction_bundle_wire_v3."""

    import httpx

    url = endpoint.rstrip("/") + "/proof_v3/reduction_bundle"
    response = httpx.post(
        url,
        json={
            "request_id": request_id,
            "validator_nonce": validator_nonce.hex(),
            "heads_per_layer": int(heads_per_layer),
            "row_samples": int(row_samples),
            "uniform_chunk_samples": int(uniform_chunk_samples),
            "mass_draws": int(mass_draws),
        },
        timeout=timeout, verify=verify_tls)
    if response.status_code != 200:
        raise ProofV3Error(
            f"reduction bundle fetch failed ({response.status_code}): "
            + response.text[:200])
    payload = response.json()
    try:
        return bytes.fromhex(payload["wire"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProofV3Error("reduction bundle reply is malformed") from exc


def apply_reduction_bundle_wire_v3(*, wire: bytes, validator_nonce: bytes,
                                   capture_chain_digest: bytes,
                                   expected_request_root: bytes,
                                   calibration,
                                   audited_layers,
                                   head_count: int, n_kv: int,
                                   candidate_rows,
                                   key_count: int, chunk_len: int,
                                   oracle_ox8_row,
                                   selected_layers=None,
                                   heads_per_layer: int = 2,
                                   row_samples: int = 8,
                                   uniform_chunk_samples: int = 3,
                                   mass_draws: int = 2):
    """Decode + verify ONE canary's bundled reduction wire: one
    domain-separated subaudit per selected layer, ALL of which must
    verify (any failure fails the whole canary).  Fail-closed.

    ``audited_layers`` is the signed committed layer set (the layer-
    root membership domain of every reveal); ``selected_layers`` is
    the signed policy's bundle selection (canonical sorted+distinct,
    default = all audited layers).  The validator derives its OWN
    bundle from the nonce and requires exact per-subaudit plan
    equality; the wire's count must equal the selection size before
    any section decodes.  Returns the derived plan tuple."""

    if head_count < 1 or n_kv < 1 or head_count % n_kv:
        raise ProofV3VerificationError(
            "invalid head geometry for the reduction audit")
    if int(chunk_len) < 1 or int(key_count) < 1:
        raise ProofV3VerificationError(
            "invalid audited context geometry")
    chunk_count = (int(key_count) + int(chunk_len) - 1) // int(chunk_len)
    layers = tuple(audited_layers if selected_layers is None
                   else selected_layers)
    plans = derive_reduction_bundle_v3(
        validator_nonce=validator_nonce,
        capture_chain_digest=capture_chain_digest,
        profile_digest=calibration.digest,
        selected_layers=layers,
        head_count=head_count,
        candidate_rows=candidate_rows,
        chunk_count=chunk_count,
        heads_per_layer=heads_per_layer, row_samples=row_samples,
        uniform_chunk_samples=uniform_chunk_samples,
        mass_draws=mass_draws)
    sections = decode_reduction_bundle_wire_v3(
        wire, expected_subaudits=len(plans))
    for plan, (reveal, _echo) in zip(plans, sections, strict=True):
        _verify_one_reveal_v3(
            reveal=reveal, plan=plan,
            expected_request_root=expected_request_root,
            calibration=calibration, audited_layers=audited_layers,
            head_count=head_count, n_kv=n_kv,
            candidate_rows=candidate_rows, key_count=key_count,
            chunk_len=chunk_len, oracle_ox8_row=oracle_ox8_row)
    return plans
