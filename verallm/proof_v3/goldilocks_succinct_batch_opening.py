"""Batch opening: one tree, many points, ONE real PCS opening.

Succinct tiles end dozens of sub-arguments (eq-folds, products, LogUps)
in PCS openings against the same column commitments; verifying each
opening separately is the dominant validator cost.  This argument
collapses ALL openings of one committed multilinear ``f`` at points
``p_1..p_k`` (claimed values ``y_1..y_k``) into one sumcheck:

    sum_x f(x) * G(x) == sum_i c_i * y_i,   G(x) = sum_i c_i * eq(p_i, x)

with aggregation coefficients ``c_i`` drawn post-claims from the
transcript.  The verifier replays the degree-2 rounds, evaluates
``G(r) = sum_i c_i * eq(r, p_i)`` in O(k * n) closed form, and checks a
single real PCS opening of ``f`` at the terminal point.  If any claimed
``y_i`` is wrong, the aggregate target is wrong except with probability
``k / p`` over the coefficients (Schwartz-Zippel), on top of the
sumcheck / PCS soundness.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_linear_relation_reference import _fixed32
from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
    GoldilocksMultilinearOpeningProofV3,
    GoldilocksMultilinearPcsStatementV3,
    open_goldilocks_multilinear_v3,
    verify_goldilocks_multilinear_opening_v3,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (
    _derive,
    _eq_eval,
    _eq_table,
    _lagrange_0123,
)

GOLDILOCKS_BATCH_OPENING_ABI_V3: Final = "goldilocks.batch_opening.v1"
_BATCH_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_BATCH_OPENING/V1"


@dataclass(frozen=True, slots=True)
class GoldilocksBatchOpeningClaimV3:
    """One deferred opening: MLE(f)(point) == value (point LSB-first)."""

    point: tuple[int, ...]
    value: int


@dataclass(frozen=True, slots=True)
class GoldilocksBatchOpeningProofV3:
    round_polynomials: tuple[tuple[int, int, int, int], ...]
    opening: GoldilocksMultilinearOpeningProofV3


def _seed(
    pcs_statement: GoldilocksMultilinearPcsStatementV3,
    commitment: bytes,
    claims,
    validator_nonce: bytes,
) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(_BATCH_DOMAIN)
    hasher.update(pcs_statement.validator_binding_digest)
    hasher.update(pcs_statement.variable_count.to_bytes(4, "little"))
    hasher.update(_fixed32(commitment, "commitment"))
    for claim in claims:
        for coordinate in claim.point:
            hasher.update(
                (coordinate % GOLDILOCKS_MODULUS).to_bytes(8, "little"))
        hasher.update((claim.value % GOLDILOCKS_MODULUS).to_bytes(8, "little"))
    hasher.update(_fixed32(validator_nonce, "validator_nonce"))
    return hasher.digest()


def _validate_claims(pcs_statement, claims):
    claims = tuple(claims)
    if not claims:
        raise ProofV3Error("batch opening needs at least one claim")
    for claim in claims:
        if not isinstance(claim, GoldilocksBatchOpeningClaimV3) or len(
            claim.point
        ) != pcs_statement.variable_count:
            raise ProofV3Error("batch opening claim shape is wrong")
    return claims


_CLAIMS_STREAM_MIN_VARS = 24


def _boolean_claim_index(point, n_vars: int) -> int | None:
    """Return the cube index for a Boolean LSB-first point."""

    if len(point) != n_vars:
        return None
    index = 0
    for bit, coordinate in enumerate(point):
        if coordinate not in (0, 1):
            return None
        index |= int(coordinate) << bit
    return index


def _claims_g(
    torch,
    gl_add_t,
    gl_mul_t,
    to_field_tensor,
    claims,
    coefficients,
    n_vars: int,
):
    """Build ``sum_i c_i * eq(p_i, ·)`` without expanding Boolean points.

    Every capture-cell and execution-anchor claim is made at a Boolean
    hypercube point.  Its equality polynomial is a Kronecker delta over the
    evaluation cube, so materializing one full equality table per claim is
    unnecessary.  Accumulating those coefficients directly into their cube
    cells is algebraically identical and changes neither the transcript nor
    the proof ABI.  The uncommon non-Boolean terminal claims retain the
    general equality-table path.
    """

    sparse: dict[int, int] = {}
    general_claims = []
    general_coefficients = []
    for claim, coefficient in zip(claims, coefficients, strict=True):
        index = _boolean_claim_index(claim.point, n_vars)
        if index is None:
            general_claims.append(claim)
            general_coefficients.append(coefficient)
            continue
        sparse[index] = (
            sparse.get(index, 0) + coefficient
        ) % GOLDILOCKS_MODULUS

    g = None
    if sparse:
        items = tuple(
            (index, value)
            for index, value in sparse.items()
            if value
        )
        g = torch.zeros(
            1 << n_vars,
            dtype=torch.int64,
            device="cuda",
        )
        if items:
            indices = torch.tensor(
                tuple(index for index, _value in items),
                dtype=torch.int64,
                device="cuda",
            )
            values = to_field_tensor(
                tuple(value for _index, value in items),
                "cuda",
            )
            g[indices] = values

    if general_claims:
        general = (
            _stream_claims_g(
                tuple(general_claims),
                tuple(general_coefficients),
                n_vars,
            )
            if n_vars >= _CLAIMS_STREAM_MIN_VARS
            else _doubling_claims_g(
                torch,
                gl_add_t,
                gl_mul_t,
                to_field_tensor,
                tuple(general_claims),
                tuple(general_coefficients),
                n_vars,
            )
        )
        g = general if g is None else gl_add_t(g, general)

    if g is None:
        # A non-empty claim set can reach this only if duplicate Boolean
        # coefficients cancel exactly.  Keep the zero polynomial here; the
        # existing terminal-degeneracy check remains authoritative.
        g = torch.zeros(
            1 << n_vars,
            dtype=torch.int64,
            device="cuda",
        )
    return g


def _doubling_claims_g(torch, gl_add_t, gl_mul_t, to_field_tensor,
                       claims, coefficients, n_vars: int):
    """Combined eq table by claim-batched doubling (small cubes)."""

    prefix_vars = min(12, n_vars)
    # <= 2^25 cells live: the doubling cat holds 2x the chunk, and the
    # big trees run beside vLLM -- transient eq peaks must stay small
    chunk = max(1, (1 << 25) // (1 << n_vars))
    g = None
    for base in range(0, len(claims), chunk):
        part = claims[base: base + chunk]
        part_coeff = coefficients[base: base + chunk]
        rows = []
        tails = []
        for coefficient, claim in zip(part_coeff, part, strict=True):
            prefix = [coefficient]
            for z in claim.point[:prefix_vars]:
                z_c = z % GOLDILOCKS_MODULUS
                one_minus = (1 - z) % GOLDILOCKS_MODULUS
                prefix = [
                    v * f % GOLDILOCKS_MODULUS
                    for f in (one_minus, z_c) for v in prefix
                ]
            rows.append(prefix)
            tails.append(tuple(
                z % GOLDILOCKS_MODULUS
                for z in claim.point[prefix_vars:]))
        eq_dev = to_field_tensor(
            tuple(v for row in rows for v in row), "cuda"
        ).view(len(part), -1)
        n_tail = n_vars - prefix_vars
        if n_tail:
            cols = to_field_tensor(
                tuple((1 - t[j]) % GOLDILOCKS_MODULUS
                      for j in range(n_tail) for t in tails)
                + tuple(t[j] for j in range(n_tail) for t in tails),
                "cuda").view(2, n_tail, len(part))
        for j in range(n_tail):
            one_col = cols[0, j].view(-1, 1)
            z_col = cols[1, j].view(-1, 1)
            eq_dev = torch.cat(
                (gl_mul_t(eq_dev, one_col.expand_as(eq_dev)),
                 gl_mul_t(eq_dev, z_col.expand_as(eq_dev))), dim=1)
        part_sum = eq_dev[0]
        for row_index in range(1, eq_dev.shape[0]):
            part_sum = gl_add_t(part_sum, eq_dev[row_index])
        part_sum = part_sum.contiguous()
        g = part_sum if g is None else gl_add_t(g, part_sum)
        del eq_dev
    return g


def _stream_claims_g(claims, coefficients, n_vars: int):
    """Combined eq table streamed over cube chunks (big cubes).

    eq(point)[chunk c, offset m] factors as
    eq_low(point[:low])[m] * w_high(point[low:], c): one bounded
    eq_low table per claim plus a host scalar per (claim, chunk), so
    the transients never exceed the chunk size -- the claim-batched
    doubling above peaks at multiple full-cube tensors on 2^25 cubes.
    Byte-identical g (pinned by test_stream_claims_g_matches_doubling).
    """

    import torch

    from verallm.proof_v3.native_goldilocks_backend import (
        gl_add_t,
        gl_scale_t,
        to_field_tensor,
    )

    low_vars = _CLAIMS_STREAM_MIN_VARS - 1
    low_vars = min(low_vars, n_vars - 1)
    length = 1 << low_vars
    prefix_vars = min(12, low_vars)
    g = torch.empty(1 << n_vars, dtype=torch.int64, device="cuda")
    for claim_index, (claim, coefficient) in enumerate(
        zip(claims, coefficients, strict=True)
    ):
        prefix = [1]
        for z in claim.point[:prefix_vars]:
            z_c = z % GOLDILOCKS_MODULUS
            one_minus = (1 - z) % GOLDILOCKS_MODULUS
            prefix = [
                v * f % GOLDILOCKS_MODULUS
                for f in (one_minus, z_c) for v in prefix
            ]
        table = to_field_tensor(tuple(prefix), "cuda")
        for z in claim.point[prefix_vars:low_vars]:
            z_c = z % GOLDILOCKS_MODULUS
            one_minus = (1 - z) % GOLDILOCKS_MODULUS
            table = torch.cat((
                gl_scale_t(table, one_minus),
                gl_scale_t(table, z_c),
            ))
        high = tuple(
            z % GOLDILOCKS_MODULUS for z in claim.point[low_vars:]
        )
        for c in range(1 << (n_vars - low_vars)):
            weight = coefficient % GOLDILOCKS_MODULUS
            for bit, z in enumerate(high):
                factor = z if (c >> bit) & 1 else (
                    (1 - z) % GOLDILOCKS_MODULUS)
                weight = weight * factor % GOLDILOCKS_MODULUS
            term = gl_scale_t(table, weight)
            destination = g[c * length:(c + 1) * length]
            if claim_index == 0:
                destination.copy_(term)
            else:
                combined = gl_add_t(destination, term)
                destination.copy_(combined)
                del combined
            del term
        del table
    return g


def _claims_phase(
    *,
    pcs_statement: GoldilocksMultilinearPcsStatementV3,
    tree,
    values: tuple[int, ...],
    claims,
    validator_nonce: bytes,
    fused=None,
    values_device=None,
):
    """Claims-aggregation sumcheck: returns (rounds, point, claimed).

    ``claimed`` is the fully-folded terminal value f(point) -- the same
    field element the terminal opening recomputes as sum(values * eq).
    """

    claims = _validate_claims(pcs_statement, claims)
    transcript = _seed(pcs_statement, tree.commitment, claims, validator_nonce)
    coefficients = tuple(
        _derive(transcript, b"batchcoef", i) for i in range(len(claims))
    )
    rounds: list[tuple[int, int, int, int]] = []
    challenges: list[int] = []
    if fused is not None and values_device is not None:
        import torch

        from verallm.proof_v3.native_goldilocks_backend import (
            gl_add_t,
            gl_mul_t,
            to_field_tensor,
        )
        from verallm.proof_v3.native_cuda_fold_backend import _sum_partials
        from verallm.proof_v3.native_pcs_backend import _encode_challenge

        # Build ALL k eq tables at once: (k, cells) rows, one pair of
        # field muls per variable for the whole batch (launch-bound
        # otherwise).  CPU prefix keeps the tiny early doublings off the
        # GPU; memory is chunk-capped for the big cubes.
        n_vars = pcs_statement.variable_count
        g = _claims_g(
            torch,
            gl_add_t,
            gl_mul_t,
            to_field_tensor,
            claims,
            coefficients,
            n_vars,
        )
        a = values_device
        if hasattr(fused[0], "fs_round"):
            from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (  # noqa: E501
                _CHALLENGE_DOMAIN,
            )

            n_rounds = a.numel().bit_length() - 1
            t_buf = torch.tensor(
                list(transcript), dtype=torch.uint8, device="cuda")
            label = b"batchopen"
            dom_label = torch.tensor(
                list(_CHALLENGE_DOMAIN + label), dtype=torch.uint8,
                device="cuda")
            rounds_buf = torch.zeros(
                (n_rounds, 4), dtype=torch.int64, device="cuda")
            chal_buf = torch.zeros(
                n_rounds, dtype=torch.int64, device="cuda")
            for r in range(n_rounds):
                partials = fused[0].round_partials(a, g)
                fused[0].fs_round(
                    partials, 3, t_buf, dom_label,
                    len(_CHALLENGE_DOMAIN), len(label), r + 1,
                    rounds_buf[r], chal_buf[r:r + 1])
                a = fused[0].lerp_fold_ptr(a, chal_buf[r:r + 1])
                g = fused[0].lerp_fold_ptr(g, chal_buf[r:r + 1])
            torch.cuda.synchronize()

            def _dec(v):
                return v + (1 << 64) if v < 0 else v

            rounds = [
                tuple(_dec(v) for v in row)
                for row in rounds_buf.cpu().tolist()
            ]
            challenges = [_dec(v) for v in chal_buf.cpu().tolist()]
        else:
            while a.numel() > 1:
                partials = fused[0].round_partials(a, g)
                torch.cuda.synchronize()
                g0, g1, g2 = _sum_partials(partials)
                g3 = (3 * g2 - 3 * g1 + g0) % GOLDILOCKS_MODULUS
                evals = (g0, g1, g2, g3)
                rounds.append(evals)
                transcript = hashlib.sha256(
                    transcript
                    + b"".join(v.to_bytes(8, "little") for v in evals)
                ).digest()
                challenge = _derive(transcript, b"batchopen", len(rounds))
                challenges.append(challenge)
                encoded = _encode_challenge(challenge)
                a = fused[0].lerp_fold(a, encoded)
                g = fused[0].lerp_fold(g, encoded)
    else:
        n_cells = 1 << pcs_statement.variable_count
        g_table = [0] * n_cells
        for coefficient, claim in zip(coefficients, claims, strict=True):
            for index, eq_value in enumerate(_eq_table(claim.point)):
                g_table[index] = (
                    g_table[index] + coefficient * eq_value
                ) % GOLDILOCKS_MODULUS
        work_v = [v % GOLDILOCKS_MODULUS for v in values]
        work_g = g_table
        while len(work_v) > 1:
            half = len(work_v) // 2
            evals4 = [0, 0, 0, 0]
            for i in range(half):
                v_lo, v_hi = work_v[i], work_v[half + i]
                g_lo, g_hi = work_g[i], work_g[half + i]
                for z in range(4):
                    vv = (v_lo + z * (v_hi - v_lo)) % GOLDILOCKS_MODULUS
                    gg = (g_lo + z * (g_hi - g_lo)) % GOLDILOCKS_MODULUS
                    evals4[z] = (evals4[z] + vv * gg) % GOLDILOCKS_MODULUS
            evals = tuple(evals4)
            rounds.append(evals)
            transcript = hashlib.sha256(
                transcript
                + b"".join(v.to_bytes(8, "little") for v in evals)
            ).digest()
            challenge = _derive(transcript, b"batchopen", len(rounds))
            challenges.append(challenge)
            work_v = [
                (work_v[i] + challenge * (work_v[half + i] - work_v[i]))
                % GOLDILOCKS_MODULUS for i in range(half)
            ]
            work_g = [
                (work_g[i] + challenge * (work_g[half + i] - work_g[i]))
                % GOLDILOCKS_MODULUS for i in range(half)
            ]
    point = tuple(reversed(challenges))
    if fused is not None and values_device is not None:
        claimed_raw = int(a.cpu()[0].item())
        claimed = claimed_raw + (1 << 64) if claimed_raw < 0 else claimed_raw
    else:
        claimed = work_v[0] % GOLDILOCKS_MODULUS
    return tuple(rounds), point, claimed


def prove_goldilocks_batch_claims_v3(
    *,
    pcs_statement: GoldilocksMultilinearPcsStatementV3,
    tree,
    values: tuple[int, ...],
    claims,
    validator_nonce: bytes,
    fused=None,
    values_device=None,
):
    """Claims aggregation only (no terminal opening).

    The batched opening path (opening-v2 Part 1) collects
    (rounds, point, claimed) per column and opens every column through
    ONE shared fold chain instead of per-column terminal openings.
    """

    return _claims_phase(
        pcs_statement=pcs_statement, tree=tree, values=values,
        claims=claims, validator_nonce=validator_nonce, fused=fused,
        values_device=values_device)


def prove_goldilocks_batch_opening_v3(
    *,
    pcs_statement: GoldilocksMultilinearPcsStatementV3,
    tree,
    values: tuple[int, ...],
    claims,
    validator_nonce: bytes,
    fused=None,
    values_device=None,
) -> GoldilocksBatchOpeningProofV3:
    rounds, point, _claimed = _claims_phase(
        pcs_statement=pcs_statement, tree=tree, values=values,
        claims=claims, validator_nonce=validator_nonce, fused=fused,
        values_device=values_device)
    if fused is not None and values_device is not None:
        from verallm.proof_v3.native_pcs_backend import (
            fused_open_goldilocks_multilinear_v3,
        )

        opening = fused_open_goldilocks_multilinear_v3(
            fold_extension=fused[0], tree_extension=fused[1],
            statement=pcs_statement, tree=tree,
            evaluations_device=values_device, point=point,
            validator_nonce=validator_nonce)
    else:
        opening = open_goldilocks_multilinear_v3(
            statement=pcs_statement, tree=tree, evaluations=tuple(values),
            point=point, validator_nonce=validator_nonce)
    return GoldilocksBatchOpeningProofV3(
        round_polynomials=tuple(rounds), opening=opening)


def _replay_claims(
    round_polynomials,
    *,
    pcs_statement: GoldilocksMultilinearPcsStatementV3,
    commitment: bytes,
    claims,
    validator_nonce: bytes,
):
    """Replay the claims-aggregation rounds; return (point, claimed).

    ``claimed`` is derived from the terminal relation
    ``running == claimed * g(point)``.  A zero eq-combination is
    rejected fail-closed (a negligible-probability event that would
    otherwise leave the terminal value unpinned).
    """

    claims = _validate_claims(pcs_statement, claims)
    n = pcs_statement.variable_count
    if len(round_polynomials) != n:
        raise ProofV3VerificationError("batch opening arity is wrong")
    transcript = _seed(
        pcs_statement, _fixed32(commitment, "commitment"), claims,
        validator_nonce)
    coefficients = tuple(
        _derive(transcript, b"batchcoef", i) for i in range(len(claims))
    )
    running = 0
    for coefficient, claim in zip(coefficients, claims, strict=True):
        running = (
            running + coefficient * (claim.value % GOLDILOCKS_MODULUS)
        ) % GOLDILOCKS_MODULUS
    challenges: list[int] = []
    compiled = None
    try:
        from verallm.proof_v3.c_multiopen import replay_rounds4
        from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (  # noqa: E501
            _CHALLENGE_DOMAIN as _LOGUP_DOMAIN,
        )

        compiled = replay_rounds4(
            transcript, running,
            tuple(tuple(int(v) for v in row)
                  for row in round_polynomials),
            _LOGUP_DOMAIN, b"batchopen", False, 1)
    except ImportError:
        compiled = None
    if isinstance(compiled, tuple):
        challenges_t, running, transcript = compiled
        challenges = list(challenges_t)
    elif isinstance(compiled, int):
        raise ProofV3VerificationError(
            "batch opening round replay fails")
    else:
        for evals in round_polynomials:
            evals = tuple(v % GOLDILOCKS_MODULUS for v in evals)
            if (evals[0] + evals[1]) % GOLDILOCKS_MODULUS != running:
                raise ProofV3VerificationError(
                    "batch opening round does not match the running sum")
            transcript = hashlib.sha256(
                transcript
                + b"".join(v.to_bytes(8, "little") for v in evals)
            ).digest()
            challenge = _derive(
                transcript, b"batchopen", len(challenges) + 1)
            challenges.append(challenge)
            running = _lagrange_0123(evals, challenge)
    point = tuple(reversed(challenges))
    g_at_point = 0
    for coefficient, claim in zip(coefficients, claims, strict=True):
        g_at_point = (
            g_at_point
            + coefficient * _eq_eval(point, tuple(
                z % GOLDILOCKS_MODULUS for z in claim.point))
        ) % GOLDILOCKS_MODULUS
    if g_at_point == 0:
        raise ProofV3VerificationError(
            "batch opening eq combination degenerates")
    claimed = (
        running
        * pow(g_at_point, GOLDILOCKS_MODULUS - 2, GOLDILOCKS_MODULUS)
    ) % GOLDILOCKS_MODULUS
    return point, claimed


def verify_goldilocks_batch_opening_v3(
    proof: object,
    *,
    pcs_statement: GoldilocksMultilinearPcsStatementV3,
    commitment: bytes,
    claims,
    validator_nonce: bytes,
) -> None:
    try:
        if not isinstance(proof, GoldilocksBatchOpeningProofV3):
            raise ProofV3VerificationError(
                "batch opening proof type is wrong")
        point, claimed = _replay_claims(
            proof.round_polynomials, pcs_statement=pcs_statement,
            commitment=commitment, claims=claims,
            validator_nonce=validator_nonce)
        if proof.opening.claimed_value % GOLDILOCKS_MODULUS != claimed:
            raise ProofV3VerificationError(
                "batch opening terminal coupling fails")
        verify_goldilocks_multilinear_opening_v3(
            proof.opening, statement=pcs_statement, commitment=commitment,
            point=point, expected_value=proof.opening.claimed_value,
            validator_nonce=validator_nonce)
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "batch opening proof is malformed") from exc


@dataclass(frozen=True, slots=True)
class GoldilocksDeferredOpeningV3:
    """Stand-in for a PCS opening whose verification is batched later."""

    claimed_value: int


def park_column_device_values_v3(column) -> None:
    """Stash a committed column's device values on pinned host.

    Every committed column otherwise keeps its full value tensor
    resident from commit until the claims phase, stacking across every
    committed column at long context.  The claims phase
    reads the stash back per column (``_resume_device_values``) and
    frees it, so at most one parked column rides the PCIe bus at a
    time.

    ``VERATHOS_PROOF_V3_PARK=0`` disables parking entirely for
    deployments whose host memory limit is tighter than their spare
    VRAM."""

    import os as _os

    if _os.environ.get("VERATHOS_PROOF_V3_PARK", "1") == "0":
        return
    dev = getattr(column, "device_values", None)
    if dev is None or not getattr(dev, "is_cuda", False):
        return
    import torch

    host = torch.empty(dev.shape, dtype=dev.dtype, pin_memory=True)
    host.copy_(dev)
    object.__setattr__(column, "device_values_host", host)
    object.__setattr__(column, "device_values", None)


def _resume_device_values(column):
    """Device values of a column, re-uploading a parked stash.

    The stash is cleared and ``device_values`` re-pinned on the column,
    so late readers behave as if the park never happened; the claims
    paths null the attribute right after handing the tensor to its
    single long-term owner."""

    dev = getattr(column, "device_values", None)
    if dev is not None:
        return dev
    host = getattr(column, "device_values_host", None)
    if host is None:
        return None
    dev = host.to("cuda", non_blocking=True)
    object.__setattr__(column, "device_values", dev)
    object.__setattr__(column, "device_values_host", None)
    return dev


class BatchOpeningCollectorV3:
    """Prover-side accumulator: (column tag) -> deferred claims.

    Sub-argument provers register terminal claims instead of opening;
    the tile then emits ONE batch-opening proof per column tree.
    Columns registered with a ``group_tag`` alias their claims onto the
    shared group tree at the block-extended point.
    """

    def __init__(self) -> None:
        self.claims: dict[str, list[GoldilocksBatchOpeningClaimV3]] = {}
        self.columns: dict[str, object] = {}
        self.aliases: dict[str, tuple[str, tuple[int, ...]]] = {}

    @staticmethod
    def park_all(columns) -> None:
        """Park every column's device values (helper for provers)."""

        for column in columns:
            park_column_device_values_v3(column)

    def register_column(self, tag: str, column) -> None:
        group_tag = getattr(column, "group_tag", None)
        if group_tag is not None:
            self.aliases[tag] = (group_tag, tuple(column.block_point))
        else:
            self.columns[tag] = column

    def register_group(self, group_column) -> None:
        self.columns[group_column.tag] = group_column

    def defer(self, tag: str, point, value: int) -> GoldilocksDeferredOpeningV3:
        alias = self.aliases.get(tag)
        if alias is not None:
            group_tag, block_point = alias
            tag = group_tag
            point = tuple(point) + block_point
        self.claims.setdefault(tag, []).append(
            GoldilocksBatchOpeningClaimV3(
                point=tuple(int(z) % GOLDILOCKS_MODULUS for z in point),
                value=value % GOLDILOCKS_MODULUS,
            ))
        return GoldilocksDeferredOpeningV3(claimed_value=value)

    def prove_all(self, *, validator_nonce: bytes, fused=None):
        import os as _os
        import time as _time

        _trace = _os.environ.get("VERATHOS_ATTN_TRACE") == "1"
        proofs = {}
        for tag in sorted(self.claims):
            column = self.columns[tag]
            _t0 = _time.perf_counter()
            proofs[tag] = prove_goldilocks_batch_opening_v3(
                pcs_statement=column.pcs_statement, tree=column.tree,
                values=column.values, claims=tuple(self.claims[tag]),
                validator_nonce=validator_nonce, fused=fused,
                values_device=_resume_device_values(column))
            if _trace:
                print(
                    f"OPEN {tag}: vars="
                    f"{column.pcs_statement.variable_count} "
                    f"claims={len(self.claims[tag])} "
                    f"+{_time.perf_counter() - _t0:.2f}s", flush=True)
            if getattr(column, "device_values", None) is not None:
                # a proven column's device values are never read again:
                # release them so the next tree's opening workspace
                # reuses the VRAM instead of stacking on it (the
                # long-context peak is exactly this stacking)
                object.__setattr__(column, "device_values", None)
        return proofs

    def prove_all_batched(self, *, validator_nonce: bytes, fused=None):
        """One RLC'd fold chain for every column (opening-v2 Part 1).

        Per column: the claims-aggregation sumcheck runs as in
        ``prove_all``; the terminal openings then run TOGETHER through
        ``prove_goldilocks_batched_opening_v3``.  Requires every column
        statement on the chain coset profile.  Returns
        ``{"claims": {tag: rounds}, "batched": proof}``.
        """

        import os as _os
        import time as _time

        from verallm.proof_v3.goldilocks_batched_pcs_opening import (
            BatchedOpeningComponentV3,
            prove_goldilocks_batched_opening_v3,
        )

        _trace = _os.environ.get("VERATHOS_ATTN_TRACE") == "1"
        aggregated = {}
        components = []
        for tag in sorted(self.claims):
            column = self.columns[tag]
            _t0 = _time.perf_counter()
            stash = getattr(column, "device_values_host", None)
            device_values = getattr(column, "device_values", None)
            transient_upload = device_values is None and stash is not None
            if transient_upload:
                # parked column: upload for the claims sumcheck only;
                # the component carries the STASH and re-uploads at its
                # join round (pre-join device residency was the peak)
                device_values = stash.to("cuda", non_blocking=True)
            rounds, point, claimed = prove_goldilocks_batch_claims_v3(
                pcs_statement=column.pcs_statement, tree=column.tree,
                values=column.values, claims=tuple(self.claims[tag]),
                validator_nonce=validator_nonce, fused=fused,
                values_device=device_values)
            if _trace:
                print(
                    f"CLAIMS {tag}: vars="
                    f"{column.pcs_statement.variable_count} "
                    f"claims={len(self.claims[tag])} "
                    f"+{_time.perf_counter() - _t0:.2f}s", flush=True)
            aggregated[tag] = rounds
            if device_values is None:
                # small host-committed columns (reference trees) join the
                # batch too: upload their values once
                from verallm.proof_v3.native_goldilocks_backend import (
                    to_field_tensor,
                )

                device_values = to_field_tensor(
                    tuple(column.values), "cuda")
            if transient_upload:
                components.append(BatchedOpeningComponentV3(
                    tag=tag, statement=column.pcs_statement,
                    tree=column.tree, values_device=None,
                    point=point, claimed_value=claimed,
                    values_host=stash))
                object.__setattr__(column, "device_values_host", None)
                del device_values
            else:
                components.append(BatchedOpeningComponentV3(
                    tag=tag, statement=column.pcs_statement,
                    tree=column.tree,
                    values_device=device_values,
                    point=point, claimed_value=claimed))
                if getattr(column, "device_values", None) is not None:
                    # the component carries the only reference from here
                    # on; the batched prover releases it at its join
                    object.__setattr__(column, "device_values", None)
        _t0 = _time.perf_counter()
        batched = prove_goldilocks_batched_opening_v3(
            components=tuple(components), validator_nonce=validator_nonce,
            fused=fused)
        if _trace:
            print(
                f"BATCHED-OPEN: columns={len(components)} "
                f"+{_time.perf_counter() - _t0:.2f}s", flush=True)
        return {"claims": aggregated, "batched": batched}


class BatchClaimCheckerV3:
    """Verifier-side accumulator: collects the claims the sub-argument
    verifiers relied on, then checks one batch opening per column."""

    def __init__(self) -> None:
        self.claims: dict[str, list[GoldilocksBatchOpeningClaimV3]] = {}
        self.aliases: dict[str, tuple[str, tuple[int, ...]]] = {}

    def alias(self, tag: str, group_tag: str,
              block_point: tuple[int, ...]) -> None:
        self.aliases[tag] = (group_tag, tuple(block_point))

    def expect(self, tag: str, point, value: int) -> None:
        aliased = self.aliases.get(tag)
        if aliased is not None:
            group_tag, block_point = aliased
            tag = group_tag
            point = tuple(point) + block_point
        self.claims.setdefault(tag, []).append(
            GoldilocksBatchOpeningClaimV3(
                point=tuple(int(z) % GOLDILOCKS_MODULUS for z in point),
                value=value % GOLDILOCKS_MODULUS,
            ))

    def verify_all(
        self,
        proofs: dict,
        *,
        statements: dict,
        commitments: dict,
        validator_nonce: bytes,
    ) -> None:
        if sorted(proofs) != sorted(self.claims):
            raise ProofV3VerificationError(
                "batch opening column set does not match the claims")
        for tag in sorted(self.claims):
            verify_goldilocks_batch_opening_v3(
                proofs[tag], pcs_statement=statements[tag],
                commitment=commitments[tag],
                claims=tuple(self.claims[tag]),
                validator_nonce=validator_nonce)

    def verify_all_batched(
        self,
        payload,
        *,
        statements: dict,
        commitments: dict,
        validator_nonce: bytes,
    ) -> None:
        """Verifier twin of ``prove_all_batched``."""

        from verallm.proof_v3.goldilocks_batched_pcs_opening import (
            BatchedComponentStatementV3,
            verify_goldilocks_batched_opening_v3,
        )

        try:
            aggregated = payload["claims"]
            batched = payload["batched"]
        except (KeyError, TypeError) as exc:
            raise ProofV3VerificationError(
                "batched opening payload is malformed") from exc
        if sorted(aggregated) != sorted(self.claims):
            raise ProofV3VerificationError(
                "batch opening column set does not match the claims")
        publics = []
        for tag in sorted(self.claims):
            try:
                point, claimed = _replay_claims(
                    aggregated[tag], pcs_statement=statements[tag],
                    commitment=commitments[tag],
                    claims=tuple(self.claims[tag]),
                    validator_nonce=validator_nonce)
            except ProofV3VerificationError as exc:
                raise ProofV3VerificationError(
                    f"batch opening column {tag!r} fails: {exc}"
                ) from exc
            publics.append(BatchedComponentStatementV3(
                tag=tag, statement=statements[tag],
                commitment=commitments[tag], point=point,
                claimed_value=claimed))
        verify_goldilocks_batched_opening_v3(
            batched, components=tuple(publics),
            validator_nonce=validator_nonce)


__all__ = [
    "GOLDILOCKS_BATCH_OPENING_ABI_V3",
    "BatchClaimCheckerV3",
    "BatchOpeningCollectorV3",
    "GoldilocksBatchOpeningClaimV3",
    "GoldilocksBatchOpeningProofV3",
    "GoldilocksDeferredOpeningV3",
    "prove_goldilocks_batch_claims_v3",
    "prove_goldilocks_batch_opening_v3",
    "verify_goldilocks_batch_opening_v3",
]


class _NsDict:
    """Dict proxy that namespaces keys (and, for aliases, the group-tag
    value component) -- backends touch collector.columns/.aliases/.claims
    directly, so the namespacing must hold for raw dict access too."""

    def __init__(self, inner: dict, ns: str, ns_value0: bool = False):
        self._d = inner
        self._ns = ns
        self._v0 = ns_value0

    def _k(self, key):
        return self._ns + key

    def __contains__(self, key):
        return self._k(key) in self._d

    def __getitem__(self, key):
        return self._d[self._k(key)]

    def __setitem__(self, key, value):
        if self._v0 and isinstance(value, tuple) and value and isinstance(
                value[0], str):
            value = (self._ns + value[0],) + tuple(value[1:])
        self._d[self._k(key)] = value

    def get(self, key, default=None):
        return self._d.get(self._k(key), default)

    def setdefault(self, key, default):
        return self._d.setdefault(self._k(key), default)


class NamespacedCollectorV3:
    """Cross-tile view of one shared collector: every tag is prefixed with
    the tile namespace, so N same-shaped tiles (e.g. the chunks of one
    layer) defer their opening claims into ONE batch-opening set."""

    def __init__(self, inner: BatchOpeningCollectorV3, ns: str) -> None:
        self._inner = inner
        self._ns = ns
        self.columns = _NsDict(inner.columns, ns)
        self.aliases = _NsDict(inner.aliases, ns, ns_value0=True)
        self.claims = _NsDict(inner.claims, ns)

    def register_column(self, tag: str, column) -> None:
        group_tag = getattr(column, "group_tag", None)
        if group_tag is not None:
            self._inner.aliases[self._ns + tag] = (
                self._ns + group_tag, tuple(column.block_point))
        else:
            self._inner.columns[self._ns + tag] = column

    def register_group(self, group_column) -> None:
        self._inner.columns[self._ns + group_column.tag] = group_column

    def defer(self, tag: str, point, value: int):
        return self._inner.defer(self._ns + tag, point, value)


class NamespacedCheckerV3:
    """Verifier twin of NamespacedCollectorV3."""

    def __init__(self, inner: BatchClaimCheckerV3, ns: str) -> None:
        self._inner = inner
        self._ns = ns

    def alias(self, tag: str, group_tag: str, block_point) -> None:
        self._inner.alias(self._ns + tag, self._ns + group_tag, block_point)

    def expect(self, tag: str, point, value: int) -> None:
        self._inner.expect(self._ns + tag, point, value)
