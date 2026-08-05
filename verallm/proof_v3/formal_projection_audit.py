"""GATE.5-6 formal: commitment-authenticated exact projection audit.

Ties the pieces together with real pre-nonce commitments + reveal openings:

* the served activation X is committed into a per-request CAPTURE root
  (``capture_commit_v3``) BEFORE any nonce; the reveal opens the sampled X
  rows against it (``verify_capture_slice_reveal_v3``);
* the served weight W is committed as the exact int8 surrogate output
  ``S = int8(X) . int8(W)`` (the miner cannot synthesize this from a fake X:
  X is capture-authenticated and, in the full chain, anchored to the prompt);
* the trusted per-projection weight root comes from the SIGNED manifest;
* the validator recomputes ``int8(X) . int8(W_manifest)`` at nonce-sampled
  output cells and requires EXACT integer equality with the committed S.

A miner that served a weight != the registered one commits S from the served
weight but can only reveal the registered weight (to satisfy the manifest
root), so the exact recompute mismatches -- even for a subtle 0.1-sigma
substitution (which shifts several int8 units).  Approximate/dequant checks
miss that; exact integer catches it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.recompute_audit import (
    build_capture_slice_reveal_v3, verify_capture_slice_reveal_v3)
from verallm.miner.proof_v3_projection_audit import int8_reduce_matrix

__all__ = [
    "CommittedProjectionV3",
    "commit_projection_v3",
    "derive_cells_v3",
    "verify_committed_projection_audit_v3",
]

_HALF = GOLDILOCKS_MODULUS >> 1


def _f(v):
    v = int(v)
    return v + GOLDILOCKS_MODULUS if v < 0 else v


def _int8_matmul_row(x_row, w_rows, outs):
    return {o: sum(int(x_row[i]) * int(w_rows[o][i]) for i in range(len(x_row)))
            for o in outs}


@dataclass(frozen=True, slots=True)
class CommittedProjectionV3:
    """Pre-nonce commitments for one projection of one served request."""

    name: str
    binding: bytes
    x_root: bytes                # capture root over int8(X) (row-major)
    int8_X: tuple                # [token][in] served activation, int8
    int8_W_served: tuple         # [out][in] served weight, int8
    surrogate: dict              # (token,out) -> exact int8(X).int8(W_served)
    x_scale: float
    w_scale: float
    in_dim: int
    out_dim: int
    token_count: int


def commit_projection_v3(*, name: str, binding: bytes, captured_X, served_W) -> CommittedProjectionV3:
    """Miner-side pre-nonce commit: capture root over X + exact surrogate S."""

    from verallm.proof_v3.goldilocks_capture_commitment_reference import capture_commit_v3

    int8_X, xscale = int8_reduce_matrix(captured_X)
    int8_W, wscale = int8_reduce_matrix(served_W)
    if not int8_X or not int8_W:
        raise ProofV3Error("empty projection")
    in_dim, out_dim, tok = len(int8_X[0]), len(int8_W), len(int8_X)
    w_pad = 1 << max(0, (in_dim - 1).bit_length())
    flat = []
    for t in range(1 << max(0, (tok - 1).bit_length())):
        row = int8_X[t] if t < tok else []
        for c in range(w_pad):
            flat.append(_f(row[c]) if c < len(row) else 0)
    x_root = capture_commit_v3(validator_binding_digest=binding, raw_values=tuple(flat)).root
    surrogate = {}
    for t in range(tok):
        for o, v in _int8_matmul_row(int8_X[t], int8_W, range(out_dim)).items():
            surrogate[(t, o)] = v
    return CommittedProjectionV3(
        name=name, binding=binding, x_root=x_root,
        int8_X=tuple(tuple(r) for r in int8_X),
        int8_W_served=tuple(tuple(r) for r in int8_W),
        surrogate=surrogate, x_scale=xscale, w_scale=wscale,
        in_dim=in_dim, out_dim=out_dim, token_count=tok)


def derive_cells_v3(*, validator_nonce: bytes, x_root: bytes,
                    token_count: int, out_dim: int, token_samples: int = 1,
                    out_samples: int = 16):
    """Nonce -> sampled (token, output) audit cells (bound to the X root)."""

    seed = hashlib.sha256(b"VERATHOS/PROOF_V3/CELLS/V1" + validator_nonce + x_root).digest()

    def _sample(domain, pop, k):
        k = min(k, pop); chosen, seen, c = [], set(), 0
        bound = (1 << 64) - ((1 << 64) % pop)
        while len(chosen) < k:
            blk = hashlib.sha256(seed + domain + c.to_bytes(8, "little")).digest(); c += 1
            for off in range(0, 32, 8):
                v = int.from_bytes(blk[off:off + 8], "little")
                if v >= bound:
                    continue
                idx = v % pop
                if idx not in seen:
                    seen.add(idx); chosen.append(idx)
                if len(chosen) == k:
                    break
        return tuple(sorted(chosen))

    return _sample(b"tok", token_count, token_samples), _sample(b"out", out_dim, out_samples)


def verify_committed_projection_audit_v3(
    *,
    committed: CommittedProjectionV3,
    tokens, outs,
    revealed_x_rows: dict,          # token -> int8 X row (miner-revealed)
    x_opening_indices, x_opening,   # capture-root multiproof over the X leaves
    manifest_weight_rows: dict,     # out -> int8 W row (from the SIGNED manifest)
) -> None:
    """Validator: authenticate X vs capture root, then EXACT recompute
    int8(X).int8(W_manifest) == committed surrogate at every sampled cell.

    ``manifest_weight_rows`` are the registry/manifest-authenticated weight
    rows (opened vs the signed manifest root by the caller).  The committed
    surrogate was produced from the SERVED weight; if served != registered
    the exact integer recompute mismatches.
    """

    # 1. authenticate the revealed X rows against the pre-nonce capture root
    w_pad = 1 << max(0, (committed.in_dim - 1).bit_length())
    leaf_count = w_pad * (1 << max(0, (committed.token_count - 1).bit_length()))
    values = verify_capture_slice_reveal_v3(
        capture_root=committed.x_root, validator_binding_digest=committed.binding,
        leaf_count=leaf_count, indices=x_opening_indices, opening=x_opening)
    sorted_idx = tuple(sorted(set(int(i) for i in x_opening_indices)))
    by_leaf = dict(zip(sorted_idx, values, strict=True))
    for t in tokens:
        base = t * w_pad
        row = revealed_x_rows.get(t)
        if row is None:
            raise ProofV3VerificationError(f"X row {t} not revealed")
        for i in range(committed.in_dim):
            fv = int(by_leaf[base + i])
            signed = fv - GOLDILOCKS_MODULUS if fv > _HALF else fv
            if signed != int(row[i]):
                raise ProofV3VerificationError(
                    f"X row {t}[{i}] does not open against the capture root")

    # 2. EXACT recompute vs the committed surrogate, using MANIFEST weights
    for t in tokens:
        x_row = revealed_x_rows[t]
        for o in outs:
            w_row = manifest_weight_rows.get(o)
            if w_row is None:
                raise ProofV3VerificationError(f"manifest weight row {o} missing")
            got = sum(int(x_row[i]) * int(w_row[i]) for i in range(committed.in_dim))
            expected = committed.surrogate.get((t, o))
            if expected is None:
                raise ProofV3VerificationError(f"no committed surrogate at ({t},{o})")
            if got != int(expected):
                raise ProofV3VerificationError(
                    f"exact int8 recompute != committed surrogate at ({t},{o}) "
                    "-- served weight != registered weight (substitution)")


def open_x_rows_v3(committed: CommittedProjectionV3, tokens):
    """Miner-side: multiproof over the X leaves of the sampled token rows."""

    w_pad = 1 << max(0, (committed.in_dim - 1).bit_length())
    flat = []
    for t in range(1 << max(0, (committed.token_count - 1).bit_length())):
        row = committed.int8_X[t] if t < committed.token_count else ()
        for c in range(w_pad):
            flat.append(_f(row[c]) if c < len(row) else 0)
    leaves = []
    for t in tokens:
        base = t * w_pad
        leaves.extend(range(base, base + committed.in_dim))
    opening = build_capture_slice_reveal_v3(
        validator_binding_digest=committed.binding, raw_values=tuple(flat), indices=leaves)
    return tuple(sorted(set(leaves))), opening
