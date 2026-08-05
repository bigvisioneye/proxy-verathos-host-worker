"""Succinct LogUp argument for proof-v3 (O(q log N) verification).

The verify-side replacement for the reference LogUp: instead of the
verifier re-scanning full openings and both rational sums, every column
is PCS-committed and the two LogUp identities become sumchecks whose
terminal values are supplied by PCS evaluation openings.

Protocol (dual challenge, per the reference semantics):

1. **Freeze** (pre-nonce): PCS-commit the witness column ``W`` (looked-up
   values) and the multiplicity column ``M`` over the padded table.
2. ``beta_1, beta_2`` derive from (statement, W root, M root, nonce).
3. Prover commits the inverse columns ``D_c[i] = 1/(beta_c + w_i)`` and
   ``E_c[t] = m_t/(beta_c + tab_t)`` (post-challenge commit round), then
   eq-points ``z_c``/``z'_c`` derive from the transcript.
4. Per challenge ``c`` four sumcheck sub-arguments, every terminal
   column value opened against its ONE shared PCS commitment (a column
   reused across sub-arguments must never be re-committed, or the
   sub-arguments could be run over different columns):

   * ``dsum``: sum_i D_c[i] == S_c              (open D)
   * ``esum``: sum_t E_c[t] == S_c              (open E; same S_c)
   * ``dwf`` : sum_i eq(z_c,i) * D_c[i]*(beta_c + w_i) == 1
               (degree-3 rounds; open D and W; eq is verifier-computed)
   * ``etf`` : sum_t eq(z'_c,t) * (E_c[t]*(beta_c + tab_t) - m_t) == 0
               (open E and M; the public table's MLE ``tab~(point)`` is
               evaluated by the verifier directly -- O(|table|) native
               multiplies, the only non-logarithmic verifier term)

The witness/table sums coupling ``S_c`` equal on both sides for both
independent challenges is exactly the reference LogUp acceptance.
"""

from __future__ import annotations

from functools import lru_cache

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_linear_relation_reference import (
    _fixed32,
    _integer,
)
from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
    GoldilocksMultilinearOpeningProofV3,
    GoldilocksMultilinearPcsStatementV3,
    commit_goldilocks_multilinear_v3,
    open_goldilocks_multilinear_v3,
    verify_goldilocks_multilinear_opening_v3,
)
from verallm.proof_v3.goldilocks_reference import (
    GOLDILOCKS_MODULUS,
    goldilocks_inv,
)

GOLDILOCKS_SUCCINCT_LOGUP_ABI_V3: Final = "goldilocks.succinct_logup.reference.v1"
_STATEMENT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_SUCCINCT_LOGUP/V1/STATEMENT"
)
_TRANSCRIPT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_SUCCINCT_LOGUP/V1/TRANSCRIPT"
)
_CHALLENGE_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_SUCCINCT_LOGUP/V1/CHALLENGE"
)
_COLUMN_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_SUCCINCT_LOGUP/V1/COLUMN"
# One beta challenge: the rational-sum identity fails for a forged
# multiset only on a beta-collision, probability <= (witness + table
# cells) / p ~= 2^21 / 2^64 = 2^-43 -- far below the wire's query-bound
# union soundness, so a second challenge bought nothing but 2x cost.
_CHALLENGE_COUNT: Final = 1


@lru_cache(maxsize=256)
def _table_digest_cached(table: tuple[int, ...]) -> bytes:
    return hashlib.sha256(
        b"".join(v.to_bytes(8, "little") for v in table)
    ).digest()


def _field(value: object, name: str) -> int:
    integer = _integer(value, name)
    if not 0 <= integer < GOLDILOCKS_MODULUS:
        raise ProofV3Error(f"{name} must be a canonical Goldilocks element")
    return integer


def _derive(seed: bytes, label: bytes, index: int) -> int:
    for counter in range(1 << 16):
        candidate = int.from_bytes(
            hashlib.sha256(
                _CHALLENGE_DOMAIN + seed + label + struct.pack("<II", index, counter)
            ).digest()[:8],
            "little",
        )
        if candidate < GOLDILOCKS_MODULUS:
            return candidate
    raise ProofV3Error("unable to derive a succinct-logup challenge")


def _pad_pow2_len(length: int) -> int:
    return 1 << max(1, (length - 1).bit_length())


@dataclass(frozen=True, slots=True)
class GoldilocksSuccinctLogupStatementV3:
    """Validator-owned statement: the public table and witness arity.

    ``witness_binding_override`` makes the witness column's PCS statement
    equal an EXTERNAL column statement (a tile column), so one shared
    commitment serves both the tile's eq-folds and this LogUp -- the
    binding that prevents a prover from running the membership argument
    over a different column.
    """

    validator_binding_digest: bytes
    table: tuple[int, ...]
    witness_variable_count: int
    witness_binding_override: bytes | None = None

    def __post_init__(self) -> None:
        binding = _fixed32(
            self.validator_binding_digest, "succinct-logup binding", nonzero=True
        )
        if not isinstance(self.table, tuple) or len(self.table) < 1:
            raise ProofV3Error("succinct-logup table is malformed")
        # C-speed range validation: min/max reject out-of-field values,
        # TypeError (mixed/bool-free int check) surfaces as malformed
        try:
            if min(self.table) < 0 or max(self.table) >= GOLDILOCKS_MODULUS:
                raise ProofV3Error(
                    "table value must be a canonical Goldilocks element")
            # tables are constructed by the verifier itself; spot-check
            # types instead of an O(N) scan (min/max already rejected
            # unorderable garbage)
            for v in self.table[:4] + self.table[-4:]:
                if not isinstance(v, int) or isinstance(v, bool):
                    raise ProofV3Error("table value must be an integer")
        except TypeError as exc:
            raise ProofV3Error("succinct-logup table is malformed") from exc
        table = self.table
        variables = _integer(self.witness_variable_count, "witness_variable_count")
        if variables < 1:
            raise ProofV3Error("succinct-logup witness arity is malformed")
        if self.witness_binding_override is not None:
            _fixed32(self.witness_binding_override, "witness binding override")
        object.__setattr__(self, "validator_binding_digest", binding)
        object.__setattr__(self, "table", table)
        object.__setattr__(self, "witness_variable_count", variables)

    @property
    def witness_size(self) -> int:
        return 1 << self.witness_variable_count

    @property
    def table_size(self) -> int:
        return _pad_pow2_len(len(self.table))

    @property
    def table_variable_count(self) -> int:
        return self.table_size.bit_length() - 1

    def padded_table(self) -> tuple[int, ...]:
        return self.table + (0,) * (self.table_size - len(self.table))

    def digest(self) -> bytes:
        return hashlib.sha256(
            _STATEMENT_DOMAIN
            + self.validator_binding_digest
            + struct.pack("<II", self.witness_variable_count, len(self.table))
            + _table_digest_cached(self.table)
            + (self.witness_binding_override or bytes(32))
        ).digest()

    def column_pcs_statement(
        self, column_tag: str
    ) -> GoldilocksMultilinearPcsStatementV3:
        # Import lazily because the zero-check toolkit consumes this
        # LogUp module.  LogUp auxiliary columns must inherit the enclosing
        # protocol PCS scope just like ordinary tile columns; otherwise a
        # reference LogUp silently injects v1 columns into a shared
        # shift-chain batch.
        from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
            current_pcs_coset_profile_v3,
        )

        variables = (
            self.witness_variable_count
            if column_tag in ("W", "D0", "D1")
            else self.table_variable_count
        )
        coset_profile = current_pcs_coset_profile_v3()
        if column_tag == "W" and self.witness_binding_override is not None:
            return GoldilocksMultilinearPcsStatementV3(
                validator_binding_digest=self.witness_binding_override,
                variable_count=variables,
                coset_profile=coset_profile,
            )
        return GoldilocksMultilinearPcsStatementV3(
            validator_binding_digest=hashlib.sha256(
                _COLUMN_DOMAIN + self.digest() + column_tag.encode()
            ).digest(),
            variable_count=variables,
            coset_profile=coset_profile,
        )


@dataclass(frozen=True, slots=True)
class GoldilocksSuccinctLogupSubProofV3:
    """One sub-argument: degree-3 rounds + terminal column openings."""

    claimed_sum: int
    round_polynomials: tuple[tuple[int, int, int, int], ...]
    openings: tuple[GoldilocksMultilinearOpeningProofV3, ...]


@dataclass(frozen=True, slots=True)
class GoldilocksSuccinctLogupProofV3:
    witness_commitment: bytes
    multiplicity_commitment: bytes
    inverse_commitments: tuple[bytes, ...]      # D0, E0, D1, E1
    sums: tuple[int, ...]                       # S_0, S_1
    subproofs: tuple[GoldilocksSuccinctLogupSubProofV3, ...]  # 4 per c


def _seed_transcript(statement, w_root, m_root, validator_nonce) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + statement.digest()
        + w_root
        + m_root
        + _fixed32(validator_nonce, "validator_nonce")
    ).digest()


def _eq_table(point: tuple[int, ...]) -> list[int]:
    if len(point) >= 10:
        try:
            import numpy as np

            from verallm.proof_v3.goldilocks_numpy import gl_mul_np
        except ImportError:
            np = None
        if np is not None:
            table = np.ones(1, dtype=np.uint64)
            for r in point:
                rr = np.broadcast_to(
                    np.uint64(r % GOLDILOCKS_MODULUS), table.shape).copy()
                om = np.broadcast_to(
                    np.uint64((1 - r) % GOLDILOCKS_MODULUS),
                    table.shape).copy()
                table = np.concatenate(
                    [gl_mul_np(table, om), gl_mul_np(table, rr)])
            return table.tolist()
    table = [1]
    for r in point:
        one_minus = (1 - r) % GOLDILOCKS_MODULUS
        table = [w * one_minus % GOLDILOCKS_MODULUS for w in table] + [
            w * r % GOLDILOCKS_MODULUS for w in table
        ]
    return table


def _eq_eval(point_a: tuple[int, ...], point_b: tuple[int, ...]) -> int:
    result = 1
    for a, b in zip(point_a, point_b, strict=True):
        term = ((1 - a) * (1 - b) + a * b) % GOLDILOCKS_MODULUS
        result = result * term % GOLDILOCKS_MODULUS
    return result


def _mle_eval_msb(values: tuple[int, ...], point: tuple[int, ...]) -> int:
    work = list(values)
    for r in point:
        half = len(work) // 2
        work = [
            (work[i] + r * (work[half + i] - work[i])) % GOLDILOCKS_MODULUS
            for i in range(half)
        ]
    return work[0]


def _lagrange_0123(evals: tuple[int, int, int, int], z: int) -> int:
    inv6 = goldilocks_inv(6)
    inv2 = goldilocks_inv(2)
    e0, e1, e2, e3 = evals
    zm1, zm2, zm3 = (z - 1) % GOLDILOCKS_MODULUS, (z - 2) % GOLDILOCKS_MODULUS, (
        z - 3
    ) % GOLDILOCKS_MODULUS
    term0 = e0 * (zm1 * zm2 % GOLDILOCKS_MODULUS * zm3 % GOLDILOCKS_MODULUS)
    term0 = term0 % GOLDILOCKS_MODULUS * (GOLDILOCKS_MODULUS - inv6) % GOLDILOCKS_MODULUS
    term1 = e1 * (z * zm2 % GOLDILOCKS_MODULUS * zm3 % GOLDILOCKS_MODULUS)
    term1 = term1 % GOLDILOCKS_MODULUS * inv2 % GOLDILOCKS_MODULUS
    term2 = e2 * (z * zm1 % GOLDILOCKS_MODULUS * zm3 % GOLDILOCKS_MODULUS)
    term2 = term2 % GOLDILOCKS_MODULUS * (GOLDILOCKS_MODULUS - inv2) % GOLDILOCKS_MODULUS
    term3 = e3 * (z * zm1 % GOLDILOCKS_MODULUS * zm2 % GOLDILOCKS_MODULUS)
    term3 = term3 % GOLDILOCKS_MODULUS * inv6 % GOLDILOCKS_MODULUS
    return (term0 + term1 + term2 + term3) % GOLDILOCKS_MODULUS


def _table_mle_msb(table: tuple[int, ...], challenges) -> int:
    """Public-table MLE, MSB-folded in round order.

    For range tables (``table[i] == i``) the identity MLE has the exact
    closed form ``sum_j challenges[j] * 2^(n-1-j)``, which removes the
    only O(|table|) verifier term; arbitrary tables fall back to the
    generic fold.  The result is byte-identical either way.
    """

    n = len(challenges)
    if table == tuple(range(len(table))):
        value = 0
        for j, challenge in enumerate(challenges):
            value = (value + (challenge << (n - 1 - j))) % GOLDILOCKS_MODULUS
        return value
    if len(table) >= 2048:
        try:
            from verallm.proof_v3.goldilocks_numpy import mle_eval_msb_np

            return mle_eval_msb_np(table, tuple(challenges))
        except ImportError:
            pass
    return _mle_eval_msb(table, tuple(challenges))


def _sumcheck_rounds(
    columns: list[list[int]],
    combine,
    transcript: bytes,
    tag: bytes,
) -> tuple[list[tuple[int, int, int, int]], list[int], bytes]:
    """Degree-3 rounds over MSB half-split pairs of the given columns.

    ``combine(*cell_values) -> field`` is the per-cell polynomial; rounds
    send its evaluations at z in {0,1,2,3}.
    """

    rounds: list[tuple[int, int, int, int]] = []
    challenges: list[int] = []
    work = [list(column) for column in columns]
    while len(work[0]) > 1:
        half = len(work[0]) // 2
        evals = [0, 0, 0, 0]
        for i in range(half):
            lows = [column[i] for column in work]
            highs = [column[half + i] for column in work]
            for z in range(4):
                cell = [
                    (low + z * (high - low)) % GOLDILOCKS_MODULUS
                    for low, high in zip(lows, highs, strict=True)
                ]
                evals[z] = (evals[z] + combine(*cell)) % GOLDILOCKS_MODULUS
        rounds.append(tuple(evals))
        transcript = hashlib.sha256(
            transcript
            + tag
            + b"".join(value.to_bytes(8, "little") for value in evals)
        ).digest()
        challenge = _derive(transcript, tag, len(rounds))
        challenges.append(challenge)
        work = [
            [
                (column[i] + challenge * (column[half + i] - column[i]))
                % GOLDILOCKS_MODULUS
                for i in range(half)
            ]
            for column in work
        ]
    return rounds, challenges, transcript


def _replay_rounds(
    round_polynomials, claimed, transcript, tag
):
    try:
        from verallm.proof_v3.c_multiopen import replay_rounds4

        result = replay_rounds4(
            transcript, claimed % GOLDILOCKS_MODULUS,
            tuple(tuple(int(v) for v in row)
                  for row in round_polynomials),
            _CHALLENGE_DOMAIN, tag, True, 1)
        if isinstance(result, tuple):
            challenges, running, t_out = result
            return running, list(challenges), t_out
        if isinstance(result, int):
            raise ProofV3VerificationError(
                "succinct-logup round replay fails")
    except ImportError:
        pass
    running = claimed % GOLDILOCKS_MODULUS
    challenges: list[int] = []
    for evals in round_polynomials:
        evals = tuple(_field(v, "round evaluation") for v in evals)
        if (evals[0] + evals[1]) % GOLDILOCKS_MODULUS != running:
            raise ProofV3VerificationError(
                "succinct-logup round does not match the running sum"
            )
        transcript = hashlib.sha256(
            transcript + tag
            + b"".join(v.to_bytes(8, "little") for v in evals)
        ).digest()
        challenge = _derive(transcript, tag, len(challenges) + 1)
        challenges.append(challenge)
        running = _lagrange_0123(evals, challenge)
    return running, challenges, transcript


def logup_batch_tag_v3(
    column_tag: str, challenge_index: int,
    tag_prefix: str | None, witness_tag: str | None,
) -> str:
    """Deterministic batch-opening tag for one LogUp column tree."""

    if column_tag == "W" and witness_tag is not None:
        return witness_tag
    if column_tag in ("W", "M"):
        return f"{tag_prefix}/{column_tag}"
    return f"{tag_prefix}/{column_tag}{challenge_index}"


def logup_batch_registry_v3(
    proof, statement, tag_prefix: str, witness_tag: str | None = None,
):
    """(statements, commitments) for the aux trees a deferring verifier
    must check batch openings against."""

    statements = {}
    commitments = {}
    if witness_tag is None:
        statements[f"{tag_prefix}/W"] = statement.column_pcs_statement("W")
        commitments[f"{tag_prefix}/W"] = proof.witness_commitment
    statements[f"{tag_prefix}/M"] = statement.column_pcs_statement("M")
    commitments[f"{tag_prefix}/M"] = proof.multiplicity_commitment
    for c in range(_CHALLENGE_COUNT):
        statements[f"{tag_prefix}/D{c}"] = statement.column_pcs_statement(
            f"D{c}")
        commitments[f"{tag_prefix}/D{c}"] = proof.inverse_commitments[2 * c]
        statements[f"{tag_prefix}/E{c}"] = statement.column_pcs_statement(
            f"E{c}")
        commitments[f"{tag_prefix}/E{c}"] = proof.inverse_commitments[
            2 * c + 1]
    return statements, commitments


def prove_goldilocks_succinct_logup_v3(
    *,
    statement: GoldilocksSuccinctLogupStatementV3,
    looked_up_values: tuple[int, ...],
    validator_nonce: bytes,
    witness_tree=None,
    collector=None,
    tag_prefix: str | None = None,
    witness_tag: str | None = None,
) -> GoldilocksSuccinctLogupProofV3:
    """``witness_tree``: pre-committed witness column tree (shared with a
    tile; the statement's witness_binding_override must match).

    ``collector``: defers every PCS opening into per-tree batch-opening
    claims (aux D/E/M columns register under ``tag_prefix``; witness
    claims go to ``witness_tag`` when the witness is a shared tile
    column, else to ``tag_prefix + "/W"``)."""

    table = statement.padded_table()
    witness = tuple(_field(v, "witness value") for v in looked_up_values)
    if len(witness) > statement.witness_size:
        raise ProofV3Error("succinct-logup witness exceeds the arity")
    if len(witness) < statement.witness_size:
        # pad with the first table entry (multiplicity-accounted)
        witness = witness + (statement.table[0],) * (
            statement.witness_size - len(witness)
        )
    counts: dict[int, int] = {}
    table_index = {value: index for index, value in enumerate(statement.table)}
    for value in witness:
        if value not in table_index:
            raise ProofV3Error("succinct-logup witness value is not in the table")
        counts[table_index[value]] = counts.get(table_index[value], 0) + 1
    multiplicities = tuple(
        counts.get(index, 0) for index in range(statement.table_size)
    )
    w_tree = witness_tree or commit_goldilocks_multilinear_v3(
        statement=statement.column_pcs_statement("W"), evaluations=witness
    )
    m_tree = commit_goldilocks_multilinear_v3(
        statement=statement.column_pcs_statement("M"), evaluations=multiplicities
    )
    transcript = _seed_transcript(
        statement, w_tree.commitment, m_tree.commitment, validator_nonce
    )
    betas = tuple(
        _derive(transcript, b"beta", c) for c in range(_CHALLENGE_COUNT)
    )
    inverse_columns = []
    inverse_trees = []
    for c, beta in enumerate(betas):
        d_column = tuple(
            goldilocks_inv((beta + value) % GOLDILOCKS_MODULUS)
            for value in witness
        )
        e_column = tuple(
            multiplicities[t]
            * goldilocks_inv((beta + table[t]) % GOLDILOCKS_MODULUS)
            % GOLDILOCKS_MODULUS
            for t in range(statement.table_size)
        )
        d_tree = commit_goldilocks_multilinear_v3(
            statement=statement.column_pcs_statement(f"D{c}"),
            evaluations=d_column,
        )
        e_tree = commit_goldilocks_multilinear_v3(
            statement=statement.column_pcs_statement(f"E{c}"),
            evaluations=e_column,
        )
        inverse_columns.append((d_column, e_column))
        inverse_trees.append((d_tree, e_tree))
        transcript = hashlib.sha256(
            transcript + d_tree.commitment + e_tree.commitment
        ).digest()
    sums = []
    subproofs = []
    for c, beta in enumerate(betas):
        d_column, e_column = inverse_columns[c]
        d_tree, e_tree = inverse_trees[c]
        s_c = sum(d_column) % GOLDILOCKS_MODULUS
        sums.append(s_c)
        z_w = tuple(
            _derive(transcript, b"zw", c * 64 + j)
            for j in range(statement.witness_variable_count)
        )
        z_t = tuple(
            _derive(transcript, b"zt", c * 64 + j)
            for j in range(statement.table_variable_count)
        )
        eq_w = _eq_table(z_w)
        eq_t = _eq_table(z_t)
        bw = [
            (beta + value) % GOLDILOCKS_MODULUS for value in witness
        ]
        bt = [(beta + value) % GOLDILOCKS_MODULUS for value in table]
        specs = (
            (b"dsum", [list(d_column)], lambda d: d, s_c,
             (("D", d_tree, d_column),)),
            (b"esum", [list(e_column)], lambda e: e, s_c,
             (("E", e_tree, e_column),)),
            (b"dwf", [list(d_column), bw, list(eq_w)],
             lambda d, b, q: d * b % GOLDILOCKS_MODULUS * q % GOLDILOCKS_MODULUS,
             1,
             (("D", d_tree, d_column), ("W", w_tree, witness))),
            (b"etf", [list(e_column), bt, list(multiplicities), list(eq_t)],
             lambda e, b, m, q: (e * b - m) % GOLDILOCKS_MODULUS * q
             % GOLDILOCKS_MODULUS,
             0,
             (("E", e_tree, e_column), ("M", m_tree, multiplicities))),
        )
        for tag, columns, combine, claimed, opening_specs in specs:
            rounds, challenges, transcript = _sumcheck_rounds(
                columns, combine, transcript, tag
            )
            point = tuple(reversed(challenges))
            if collector is not None:
                openings = []
                for column_tag, tree, values in opening_specs:
                    batch_tag = logup_batch_tag_v3(
                        column_tag, c, tag_prefix, witness_tag)
                    if batch_tag not in collector.columns:
                        import types as _types

                        collector.register_column(
                            batch_tag, _types.SimpleNamespace(
                                pcs_statement=statement.column_pcs_statement(
                                    column_tag if column_tag in ("W", "M")
                                    else f"{column_tag}{c}"),
                                tree=tree,
                                values=tuple(values),
                                device_values=None))
                    openings.append(collector.defer(
                        batch_tag, point,
                        _mle_eval_msb(tuple(values), tuple(challenges))))
                openings = tuple(openings)
            else:
                openings = tuple(
                    open_goldilocks_multilinear_v3(
                        statement=statement.column_pcs_statement(
                            column_tag if column_tag in ("W", "M")
                            else f"{column_tag}{c}"
                        ),
                        tree=tree,
                        evaluations=tuple(values),
                        point=point,
                        validator_nonce=validator_nonce,
                    )
                    for column_tag, tree, values in opening_specs
                )
            subproofs.append(
                GoldilocksSuccinctLogupSubProofV3(
                    claimed_sum=claimed,
                    round_polynomials=tuple(rounds),
                    openings=openings,
                )
            )
    return GoldilocksSuccinctLogupProofV3(
        witness_commitment=w_tree.commitment,
        multiplicity_commitment=m_tree.commitment,
        inverse_commitments=tuple(
            root
            for d_tree, e_tree in inverse_trees
            for root in (d_tree.commitment, e_tree.commitment)
        ),
        sums=tuple(sums),
        subproofs=tuple(subproofs),
    )


def verify_goldilocks_succinct_logup_v3(
    proof: object,
    *,
    statement: GoldilocksSuccinctLogupStatementV3,
    witness_commitment: bytes,
    validator_nonce: bytes,
    checker=None,
    tag_prefix: str | None = None,
    witness_tag: str | None = None,
) -> None:
    """O(q log N) verification (plus one O(|table|) public-table MLE).

    ``checker``: defers every PCS opening into per-tree batch claims
    (the caller must verify them via ``logup_batch_registry_v3``)."""

    try:
        if not isinstance(proof, GoldilocksSuccinctLogupProofV3):
            raise ProofV3VerificationError("succinct-logup proof type is wrong")
        if proof.witness_commitment != _fixed32(
            witness_commitment, "witness_commitment"
        ):
            raise ProofV3VerificationError(
                "succinct-logup witness commitment is not the frozen one"
            )
        if len(proof.subproofs) != 4 * _CHALLENGE_COUNT or len(
            proof.sums
        ) != _CHALLENGE_COUNT or len(proof.inverse_commitments) != 2 * (
            _CHALLENGE_COUNT
        ):
            raise ProofV3VerificationError("succinct-logup shape is wrong")
        transcript = _seed_transcript(
            statement,
            proof.witness_commitment,
            _fixed32(proof.multiplicity_commitment, "multiplicity commitment"),
            validator_nonce,
        )
        betas = tuple(
            _derive(transcript, b"beta", c) for c in range(_CHALLENGE_COUNT)
        )
        for c in range(_CHALLENGE_COUNT):
            transcript = hashlib.sha256(
                transcript
                + _fixed32(proof.inverse_commitments[2 * c], "D commitment")
                + _fixed32(proof.inverse_commitments[2 * c + 1], "E commitment")
            ).digest()
        table = statement.padded_table()
        subproof_index = 0
        for c, beta in enumerate(betas):
            s_c = _field(proof.sums[c], "logup sum")
            d_root = proof.inverse_commitments[2 * c]
            e_root = proof.inverse_commitments[2 * c + 1]
            z_w = tuple(
                _derive(transcript, b"zw", c * 64 + j)
                for j in range(statement.witness_variable_count)
            )
            z_t = tuple(
                _derive(transcript, b"zt", c * 64 + j)
                for j in range(statement.table_variable_count)
            )
            specs = (
                (b"dsum", s_c, (("D", d_root),), None),
                (b"esum", s_c, (("E", e_root),), None),
                (b"dwf", 1, (("D", d_root), ("W", proof.witness_commitment)),
                 ("w", z_w, beta)),
                (b"etf", 0,
                 (("E", e_root), ("M", proof.multiplicity_commitment)),
                 ("t", z_t, beta)),
            )
            for tag, claimed, opening_specs, terminal_spec in specs:
                subproof = proof.subproofs[subproof_index]
                subproof_index += 1
                if _field(subproof.claimed_sum, "claimed") != claimed % (
                    GOLDILOCKS_MODULUS
                ):
                    raise ProofV3VerificationError(
                        "succinct-logup sub-claim does not match the coupling"
                    )
                running, challenges, transcript = _replay_rounds(
                    subproof.round_polynomials,
                    claimed % GOLDILOCKS_MODULUS,
                    transcript,
                    tag,
                )
                point = tuple(reversed(challenges))
                if len(subproof.openings) != len(opening_specs):
                    raise ProofV3VerificationError(
                        "succinct-logup opening count is wrong"
                    )
                opened: dict[str, int] = {}
                for opening, (column_tag, root) in zip(
                    subproof.openings, opening_specs, strict=True
                ):
                    if checker is not None:
                        checker.expect(
                            logup_batch_tag_v3(
                                column_tag, c, tag_prefix, witness_tag),
                            point, opening.claimed_value)
                    else:
                        pcs_statement = statement.column_pcs_statement(
                            column_tag if column_tag in ("W", "M")
                            else f"{column_tag}{c}"
                        )
                        verify_goldilocks_multilinear_opening_v3(
                            opening,
                            statement=pcs_statement,
                            commitment=root,
                            point=point,
                            expected_value=opening.claimed_value,
                            validator_nonce=validator_nonce,
                        )
                    opened[column_tag] = opening.claimed_value
                # terminal coupling
                if tag == b"dsum":
                    expected = opened["D"]
                elif tag == b"esum":
                    expected = opened["E"]
                elif tag == b"dwf":
                    _kind, z_point, beta_c = terminal_spec
                    # eq's index bit j pairs with the LSB-oriented point.
                    eq_value = _eq_eval(point, tuple(z_point))
                    expected = (
                        opened["D"]
                        * ((beta_c + opened["W"]) % GOLDILOCKS_MODULUS)
                        % GOLDILOCKS_MODULUS
                        * eq_value
                        % GOLDILOCKS_MODULUS
                    )
                else:  # etf
                    _kind, z_point, beta_c = terminal_spec
                    eq_value = _eq_eval(point, tuple(z_point))
                    # the public table's MLE, MSB-folded in round order --
                    # the only O(|table|) verifier term.
                    table_at_point = _table_mle_msb(
                        table, tuple(challenges))
                    expected = (
                        (
                            opened["E"]
                            * ((beta_c + table_at_point) % GOLDILOCKS_MODULUS)
                            - opened["M"]
                        )
                        % GOLDILOCKS_MODULUS
                        * eq_value
                        % GOLDILOCKS_MODULUS
                    )
                if running != expected:
                    raise ProofV3VerificationError(
                        "succinct-logup terminal coupling fails"
                    )
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "succinct-logup proof is malformed"
        ) from exc
