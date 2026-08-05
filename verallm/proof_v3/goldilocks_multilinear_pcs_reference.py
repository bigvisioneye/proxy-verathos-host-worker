"""BaseFold-style multilinear PCS reference over Goldilocks for proof-v3.

The last protocol piece for succinct verification: every sumcheck in the
relation layer currently ends with a *full opening* of the committed
table, making verification O(N).  This module supplies the succinct
replacement: a commitment to a multilinear polynomial (its evaluation
vector over the boolean hypercube) that can be opened at one point with
an O(queries * log N) proof.

Construction (BaseFold):

* **Commit**: identify the evaluation vector ``v`` with the univariate
  coefficient vector of ``P(X) = sum_i v[i] X^i`` (bit ``x_1`` = least
  significant index bit) and commit the Reed-Solomon codeword of ``P``
  over a disjoint coset (blowup 4) as one Merkle tree.
* **Open at r with claimed y = f(r)**: run the sumcheck for
  ``sum_i v[i] * eq_i(r) == y`` binding variables LSB-first.  In round
  ``j`` the prover sends the degree-2 round polynomial AND the Merkle
  root of the codeword folded with the *sumcheck challenge itself* (the
  standard FRI fold, beta = c_j, halves the domain and binds ``x_j`` to
  ``c_j`` in coefficient space).  After all rounds the final codeword
  must be constant and equal to the fully-bound value ``v(c)``.
* **Verify**: check the sumcheck chain, derive every challenge from the
  transcript, and spot-check each fold layer at nonce-derived query
  positions against its parent openings (the usual FRI consistency).
  The final acceptance couples both interleaved arguments:
  ``running == final_value * prod_j eq(c_j, r_j)``.

The verifier never touches the evaluation vector: its work is the round
chain (O(log N) scalars), the eq product (O(log N)), and Merkle path
checks (O(queries * log N)).  Fold soundness per query is 1/blowup as in
the FRI reference; the statement pins the query count.

Reference-bounded (in-memory trees, cap 2^16 codeword) like every module
in this tree; the native backend streams the same transcript on device.
"""

from __future__ import annotations

from contextvars import ContextVar
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
from verallm.proof_v3.goldilocks_merkle_reference import (
    GoldilocksMerkleMultiOpeningReference,
    GoldilocksMerkleTreeReference,
    verify_goldilocks_merkle_multiopening_reference,
)
from verallm.proof_v3.goldilocks_reference import (
    GOLDILOCKS_MODULUS,
    MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE,
    goldilocks_inv,
    goldilocks_principal_root_of_unity,
    goldilocks_radix2_domain_reference,
    ntt_goldilocks_reference,
)


GOLDILOCKS_MULTILINEAR_PCS_ABI_V3: Final = "goldilocks.multilinear_pcs.reference.v1"
GOLDILOCKS_MULTILINEAR_PCS_BLOWUP_V3: Final = 4
GOLDILOCKS_MULTILINEAR_PCS_QUERY_COUNT_V3: Final = 16
MAX_GOLDILOCKS_MULTILINEAR_PCS_QUERY_COUNT_V3: Final = 64

_PCS_QUERY_COUNT_V3: ContextVar[int] = ContextVar(
    "verathos_proof_v3_pcs_query_count",
    default=GOLDILOCKS_MULTILINEAR_PCS_QUERY_COUNT_V3,
)


class pcs_query_count_v3:
    """Pin one transcript-bound PCS query budget for a proof scope."""

    def __init__(self, query_count: int) -> None:
        self._query_count = query_count
        self._token = None

    def __enter__(self):
        query_count = self._query_count
        if (
            not isinstance(query_count, int)
            or isinstance(query_count, bool)
            or not 1 <= query_count <= MAX_GOLDILOCKS_MULTILINEAR_PCS_QUERY_COUNT_V3
        ):
            raise ProofV3Error("mlpcs query count is outside the protocol cap")
        self._token = _PCS_QUERY_COUNT_V3.set(query_count)
        return self

    def __exit__(self, *_exc) -> None:
        if self._token is not None:
            _PCS_QUERY_COUNT_V3.reset(self._token)


def current_pcs_query_count_v3() -> int:
    return _PCS_QUERY_COUNT_V3.get()

_STATEMENT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MLPCS/V1/STATEMENT/SHA256"
_LAYER_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MLPCS/V1/LAYER/SHA256"
_SHIFT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MLPCS/V1/SHIFT/SHA256"
_TRANSCRIPT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_MLPCS/V1/TRANSCRIPT/SHA256"
)
_CHALLENGE_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MLPCS/V1/CHALLENGE/SHA256"
_QUERY_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MLPCS/V1/QUERY/SHA256"

_BASE_SHIFT_CACHE: dict[bytes, int] = {}


def _field(value: object, name: str) -> int:
    integer = _integer(value, name)
    if not 0 <= integer < GOLDILOCKS_MODULUS:
        raise ProofV3Error(f"{name} must be a canonical Goldilocks element")
    return integer


def _derive_field(seed: bytes, label: bytes, index: int) -> int:
    for counter in range(1 << 16):
        candidate = int.from_bytes(
            hashlib.sha256(
                _CHALLENGE_DOMAIN + seed + label + struct.pack("<II", index, counter)
            ).digest()[:8],
            "little",
        )
        if candidate < GOLDILOCKS_MODULUS:
            return candidate
    raise ProofV3Error("unable to derive a multilinear PCS challenge")


_COSET_PROFILE_V1: Final = "v1"
_COSET_PROFILE_CHAIN: Final = "chain"
_CHAIN_MARKER_BASE: Final = b"COSET-CHAIN/V2"


@lru_cache(maxsize=1)
def _chain_cap_variables() -> int:
    from verallm.proof_v3.goldilocks_reference import (
        MAX_GOLDILOCKS_NATIVE_DOMAIN_SIZE,
    )

    return MAX_GOLDILOCKS_NATIVE_DOMAIN_SIZE.bit_length() - 3


@lru_cache(maxsize=1)
def _chain_marker() -> bytes:
    """Digest marker for the chain profile, INCLUDING the size cap.

    Every chain shift is S^(2^(cap - n)) for one master S, so two
    implementations with different caps derive different cosets for the
    same column size.  Folding the cap into the marker (and the shift
    seed) makes any cap change fail closed instead of producing
    incompatible proofs under one digest."""

    return _CHAIN_MARKER_BASE + struct.pack("<I", _chain_cap_variables())


@lru_cache(maxsize=1)
def _chain_master_shift() -> int:
    """One master shift S; size-n columns use S^(2^(cap-n)).

    Successive sizes then share the exact fold-domain chain
    (s(n)^2 == s(n-1)), so same-round layers of DIFFERENT columns live on
    the SAME coset -- the precondition for random-linear-combining their
    FRI layers.  Validity is one condition for every size at once:
    s(n)^(4*2^n) == S^(4*2^cap) != 1.
    """

    cap = _chain_cap_variables()
    order = GOLDILOCKS_MULTILINEAR_PCS_BLOWUP_V3 << cap
    seed = hashlib.sha256(_SHIFT_DOMAIN + _chain_marker()).digest()
    for counter in range(1 << 16):
        candidate = int.from_bytes(
            hashlib.sha256(seed + struct.pack("<I", counter)).digest()[:8],
            "little",
        )
        if (
            0 < candidate < GOLDILOCKS_MODULUS
            and pow(candidate, order, GOLDILOCKS_MODULUS) != 1
        ):
            return candidate
    raise ProofV3Error("unable to derive the mlpcs chain master shift")


@lru_cache(maxsize=64)
def _chain_base_shift(variable_count: int) -> int:
    return pow(
        _chain_master_shift(),
        1 << (_chain_cap_variables() - variable_count),
        GOLDILOCKS_MODULUS,
    )


@dataclass(frozen=True, slots=True)
class GoldilocksMultilinearPcsStatementV3:
    """Public parameters: variable count, blowup, coset, query budget.

    ``coset_profile`` selects the commitment coset: ``"v1"`` derives the
    shift from the statement digest (per-column cosets, the shipped
    form); ``"chain"`` uses the canonical per-size shift chain so that
    columns of every size share fold domains (batched-FRI opening).  The
    profile is part of the digest, so the two forms fail closed against
    each other.
    """

    validator_binding_digest: bytes
    variable_count: int
    coset_profile: str = _COSET_PROFILE_V1
    query_count: int | None = None

    def __post_init__(self) -> None:
        binding = _fixed32(
            self.validator_binding_digest, "mlpcs binding", nonzero=True
        )
        variables = _integer(self.variable_count, "variable_count")
        query_count = (
            current_pcs_query_count_v3()
            if self.query_count is None
            else _integer(self.query_count, "query_count")
        )
        if self.coset_profile not in (
            _COSET_PROFILE_V1, _COSET_PROFILE_CHAIN
        ):
            raise ProofV3Error("mlpcs coset profile is unknown")
        from verallm.proof_v3.goldilocks_reference import (
            MAX_GOLDILOCKS_NATIVE_DOMAIN_SIZE,
        )

        if variables < 1 or (1 << variables) * (
            GOLDILOCKS_MULTILINEAR_PCS_BLOWUP_V3
        ) > MAX_GOLDILOCKS_NATIVE_DOMAIN_SIZE:
            raise ProofV3Error("mlpcs variable count exceeds the native cap")
        if (
            query_count < 1
            or query_count > MAX_GOLDILOCKS_MULTILINEAR_PCS_QUERY_COUNT_V3
        ):
            raise ProofV3Error("mlpcs query count is outside the protocol cap")
        object.__setattr__(self, "validator_binding_digest", binding)
        object.__setattr__(self, "variable_count", variables)
        object.__setattr__(self, "query_count", query_count)

    @property
    def leaf_count(self) -> int:
        return 1 << self.variable_count

    @property
    def codeword_size(self) -> int:
        return self.leaf_count * GOLDILOCKS_MULTILINEAR_PCS_BLOWUP_V3

    def digest(self) -> bytes:
        marker = (
            _chain_marker()
            if self.coset_profile == _COSET_PROFILE_CHAIN
            else b""
        )
        return hashlib.sha256(
            _STATEMENT_DOMAIN
            + self.validator_binding_digest
            + struct.pack(
                "<III",
                self.variable_count,
                GOLDILOCKS_MULTILINEAR_PCS_BLOWUP_V3,
                self.query_count,
            )
            + marker
        ).digest()

    def layer_binding_digest(self, layer_index: int) -> bytes:
        return hashlib.sha256(
            _LAYER_DOMAIN + self.digest() + struct.pack("<I", layer_index)
        ).digest()

    def _base_shift(self) -> int:
        if self.coset_profile == _COSET_PROFILE_CHAIN:
            return _chain_base_shift(self.variable_count)
        key = self.digest()
        cached = _BASE_SHIFT_CACHE.get(key)
        if cached is not None:
            return cached
        seed = hashlib.sha256(_SHIFT_DOMAIN + key).digest()
        for counter in range(1 << 16):
            candidate = int.from_bytes(
                hashlib.sha256(seed + struct.pack("<I", counter)).digest()[:8],
                "little",
            )
            if (
                0 < candidate < GOLDILOCKS_MODULUS
                and pow(candidate, self.codeword_size, GOLDILOCKS_MODULUS) != 1
            ):
                if len(_BASE_SHIFT_CACHE) < 4096:
                    _BASE_SHIFT_CACHE[key] = candidate
                return candidate
        raise ProofV3Error("unable to derive an mlpcs coset shift")

    def domain_shift(self, layer_index: int) -> int:
        """Layer coset shift: the base shift squared per fold.

        The fold maps a codeword on ``s * <g>`` to one on ``s^2 * <g^2>``,
        so layer shifts MUST be successive squares of the base shift for
        the queried fold equation to line up index-for-index.
        """

        return _shift_pow(self._base_shift(), layer_index)

    def domain_point(self, layer_index: int, position: int) -> int:
        """One LDE-domain point in O(log size), never the full O(size) list."""

        size = self.codeword_size >> layer_index
        return (
            _shift_pow(self._base_shift(), layer_index)
            * _gen_pow(size, position)
        ) % GOLDILOCKS_MODULUS


@lru_cache(maxsize=1 << 18)
def _gen_pow(size: int, position: int) -> int:
    return pow(
        goldilocks_principal_root_of_unity(size), position,
        GOLDILOCKS_MODULUS)


@lru_cache(maxsize=4096)
def _shift_pow(base_shift: int, layer_index: int) -> int:
    return pow(base_shift, 1 << layer_index, GOLDILOCKS_MODULUS)


def _encode(
    statement: GoldilocksMultilinearPcsStatementV3,
    coefficients: tuple[int, ...],
    layer_index: int,
) -> tuple[int, ...]:
    """Reed-Solomon codeword of the coefficient vector on the layer coset."""

    size = statement.codeword_size >> layer_index
    padded = coefficients + (0,) * (size - len(coefficients))
    domain = goldilocks_radix2_domain_reference(
        size=size, shift=statement.domain_shift(layer_index)
    )
    return ntt_goldilocks_reference(padded, domain=domain)


def commit_goldilocks_multilinear_v3(
    *,
    statement: GoldilocksMultilinearPcsStatementV3,
    evaluations: tuple[int, ...],
) -> GoldilocksMerkleTreeReference:
    """Commit one multilinear evaluation vector (index bit 0 = LSB var)."""

    if len(evaluations) != statement.leaf_count:
        raise ProofV3Error("mlpcs evaluation count does not match the statement")
    values = tuple(_field(value, "mlpcs evaluation") for value in evaluations)
    codeword = _encode(statement, values, 0)
    return GoldilocksMerkleTreeReference.from_rows(
        tuple((value,) for value in codeword),
        binding_digest=statement.layer_binding_digest(0),
    )


def _eq_weights(point: tuple[int, ...]) -> tuple[int, ...]:
    """eq table over the hypercube; index bit j corresponds to point[j]."""

    n = len(point)
    table = []
    for index in range(1 << n):
        w = 1
        for j in range(n):
            bit = (index >> j) & 1
            w = w * (point[j] if bit else (1 - point[j]) % GOLDILOCKS_MODULUS)
            w %= GOLDILOCKS_MODULUS
        table.append(w)
    return tuple(table)


def _fold_pairs(values: list[int], challenge: int) -> list[int]:
    """Bind the LSB variable: pairs (v[2i], v[2i+1]) -> lerp by challenge."""

    return [
        (values[2 * i] + challenge * (values[2 * i + 1] - values[2 * i]))
        % GOLDILOCKS_MODULUS
        for i in range(len(values) // 2)
    ]


@dataclass(frozen=True, slots=True)
class GoldilocksMultilinearOpeningProofV3:
    """Succinct opening: rounds, layer roots, final value, query openings."""

    claimed_value: int
    round_polynomials: tuple[tuple[int, int, int], ...]
    layer_commitments: tuple[bytes, ...]
    final_value: int
    layer_openings: tuple[GoldilocksMerkleMultiOpeningReference, ...]

    def __post_init__(self) -> None:
        _field(self.claimed_value, "claimed_value")
        _field(self.final_value, "final_value")
        if len(self.layer_commitments) != len(self.round_polynomials):
            raise ProofV3Error("mlpcs layer/round count mismatch")


def _layer_query_indices(
    *, positions: list[int], size: int, is_final: bool
) -> tuple[int, ...]:
    """Opened indices for one layer: fold pairs, or the fold targets only."""

    indices: set[int] = set()
    for position in positions:
        if is_final:
            indices.add(position % size)
        else:
            folded = position % (size // 2)
            indices.add(folded)
            indices.add(folded + size // 2)
    return tuple(sorted(indices))


def _transcript_seed(
    statement: GoldilocksMultilinearPcsStatementV3,
    commitment: bytes,
    point: tuple[int, ...],
    claimed_value: int,
    validator_nonce: bytes,
) -> bytes:
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + statement.digest()
        + _fixed32(commitment, "mlpcs commitment")
        + b"".join(value.to_bytes(8, "little") for value in point)
        + claimed_value.to_bytes(8, "little")
        + _fixed32(validator_nonce, "validator_nonce")
    ).digest()


def open_goldilocks_multilinear_v3(
    *,
    statement: GoldilocksMultilinearPcsStatementV3,
    tree: GoldilocksMerkleTreeReference,
    evaluations: tuple[int, ...],
    point: tuple[int, ...],
    validator_nonce: bytes,
) -> GoldilocksMultilinearOpeningProofV3:
    """Prove f(point) with the interleaved sumcheck/FRI transcript."""

    n = statement.variable_count
    if len(point) != n:
        raise ProofV3Error("mlpcs point arity does not match the statement")
    point = tuple(_field(value, "mlpcs point") for value in point)
    values = [_field(value, "mlpcs evaluation") for value in evaluations]
    if len(values) != statement.leaf_count:
        raise ProofV3Error("mlpcs evaluation count does not match the statement")
    weights = list(_eq_weights(point))
    claimed = 0
    for value, weight in zip(values, weights, strict=True):
        claimed = (claimed + value * weight) % GOLDILOCKS_MODULUS
    transcript = _transcript_seed(
        statement, tree.commitment, point, claimed, validator_nonce
    )
    rounds: list[tuple[int, int, int]] = []
    layer_trees: list[GoldilocksMerkleTreeReference] = [tree]
    layer_roots: list[bytes] = []
    for round_index in range(n):
        half = len(values) // 2
        g0 = g1 = g2 = 0
        for i in range(half):
            v_lo, v_hi = values[2 * i], values[2 * i + 1]
            w_lo, w_hi = weights[2 * i], weights[2 * i + 1]
            g0 = (g0 + v_lo * w_lo) % GOLDILOCKS_MODULUS
            g1 = (g1 + v_hi * w_hi) % GOLDILOCKS_MODULUS
            v2 = (2 * v_hi - v_lo) % GOLDILOCKS_MODULUS
            w2 = (2 * w_hi - w_lo) % GOLDILOCKS_MODULUS
            g2 = (g2 + v2 * w2) % GOLDILOCKS_MODULUS
        rounds.append((g0, g1, g2))
        transcript = hashlib.sha256(
            transcript
            + g0.to_bytes(8, "little")
            + g1.to_bytes(8, "little")
            + g2.to_bytes(8, "little")
        ).digest()
        challenge = _derive_field(transcript, b"fold", round_index)
        values = _fold_pairs(values, challenge)
        weights = _fold_pairs(weights, challenge)
        folded_codeword = _encode(statement, tuple(values), round_index + 1)
        folded_tree = GoldilocksMerkleTreeReference.from_rows(
            tuple((value,) for value in folded_codeword),
            binding_digest=statement.layer_binding_digest(round_index + 1),
        )
        layer_trees.append(folded_tree)
        layer_roots.append(folded_tree.commitment)
        transcript = hashlib.sha256(transcript + folded_tree.commitment).digest()
    final_value = values[0]
    # Query phase: nonce-derived positions in the base codeword; each layer
    # is opened at the (folded) position pair needed by the fold equation.
    query_seed = hashlib.sha256(transcript + b"queries").digest()
    base_size = statement.codeword_size
    positions: list[int] = []
    for query_index in range(statement.query_count):
        positions.append(
            int.from_bytes(
                hashlib.sha256(
                    _QUERY_DOMAIN + query_seed + struct.pack("<I", query_index)
                ).digest()[:8],
                "little",
            )
            % (base_size // 2)
        )
    layer_openings: list[GoldilocksMerkleMultiOpeningReference] = []
    for layer_index, layer_tree in enumerate(layer_trees):
        size = base_size >> layer_index
        layer_openings.append(
            layer_tree.open(
                _layer_query_indices(
                    positions=positions,
                    size=size,
                    is_final=layer_index == n,
                )
            )
        )
    return GoldilocksMultilinearOpeningProofV3(
        claimed_value=claimed,
        round_polynomials=tuple(rounds),
        layer_commitments=tuple(layer_roots),
        final_value=final_value,
        layer_openings=tuple(layer_openings),
    )


def _try_compiled_opening_verify(proof, statement, commitment, point,
                                 transcript_seed):
    """Full C verification; True on success, None to fall back."""

    try:
        from verallm.proof_v3.c_multiopen import (
            verify_opening_full as _c_verify,
        )
    except ImportError:
        return None
    n = statement.variable_count
    if len(proof.layer_openings) != n + 1 or len(
        proof.layer_commitments
    ) != n:
        raise ProofV3VerificationError("mlpcs layer opening count is wrong")
    try:
        rounds = tuple(
            (int(g0), int(g1), int(g2))
            for g0, g1, g2 in proof.round_polynomials)
    except (TypeError, ValueError) as exc:
        raise ProofV3VerificationError("mlpcs proof is malformed") from exc
    roots = _fixed32(commitment, "mlpcs commitment") + b"".join(
        _fixed32(root, "mlpcs layer commitment")
        for root in proof.layer_commitments)
    base_size = statement.codeword_size
    layer_shift = tuple(
        statement.domain_shift(i) for i in range(n))
    layer_gen = tuple(
        goldilocks_principal_root_of_unity(base_size >> i)
        for i in range(n))
    layers = []
    for layer_index, opening in enumerate(proof.layer_openings):
        size = base_size >> layer_index
        binding = statement.layer_binding_digest(layer_index)
        header = binding + struct.pack("<II", size, 1)
        if not isinstance(
            opening, GoldilocksMerkleMultiOpeningReference
        ) or opening.binding_digest != binding or (
            opening.leaf_count != size or opening.leaf_width != 1
        ):
            raise ProofV3VerificationError(
                "mlpcs layer opening metadata is unexpected")
        try:
            indices = tuple(int(i) for i in opening.indices)
            values = tuple(int(row[0]) for row in opening.rows)
            slv = tuple(int(node.level) for node in opening.siblings)
            six = tuple(int(node.index) for node in opening.siblings)
            sdig = b"".join(
                _fixed32(node.digest, "mlpcs sibling digest")
                for node in opening.siblings)
        except (TypeError, ValueError, IndexError) as exc:
            raise ProofV3VerificationError(
                "mlpcs proof is malformed") from exc
        from verallm.proof_v3.goldilocks_merkle_reference import (
            _LEAF_DOMAIN as _ML,
            _NODE_DOMAIN as _MN,
            _ROOT_DOMAIN as _MR,
        )

        layers.append((
            indices, values, slv, six, sdig,
            _ML + header, _MN + header, _MR + header))
    result = _c_verify(
        transcript_seed=transcript_seed,
        claimed=proof.claimed_value,
        final_value=_field(proof.final_value, "mlpcs final value"),
        point=point,
        rounds=rounds,
        roots=roots,
        dom=_CHALLENGE_DOMAIN,
        qdom=_QUERY_DOMAIN,
        base_size=base_size,
        layer_shift=layer_shift,
        layer_gen=layer_gen,
        n_queries=statement.query_count,
        layers=layers,
    )
    if result is None:
        return None
    if result != 0:
        raise ProofV3VerificationError(
            "mlpcs compiled verification failed"
        )
    return True


def verify_goldilocks_multilinear_opening_v3(
    proof: object,
    *,
    statement: GoldilocksMultilinearPcsStatementV3,
    commitment: bytes,
    point: tuple[int, ...],
    expected_value: int,
    validator_nonce: bytes,
) -> None:
    """Succinctly verify one multilinear opening (no full openings)."""

    try:
        if not isinstance(proof, GoldilocksMultilinearOpeningProofV3):
            raise ProofV3VerificationError("mlpcs proof type is unexpected")
        n = statement.variable_count
        point = tuple(_field(value, "mlpcs point") for value in point)
        if len(point) != n or len(proof.round_polynomials) != n:
            raise ProofV3VerificationError("mlpcs round arity is wrong")
        if proof.claimed_value != _field(expected_value, "expected_value"):
            raise ProofV3VerificationError(
                "mlpcs claimed value does not match the outer relation"
            )
        transcript = _transcript_seed(
            statement,
            _fixed32(commitment, "mlpcs commitment"),
            point,
            proof.claimed_value,
            validator_nonce,
        )
        compiled = _try_compiled_opening_verify(
            proof, statement, commitment, point, transcript)
        if compiled is True:
            return
        running = proof.claimed_value
        challenges: list[int] = []
        for round_index, (g0, g1, g2) in enumerate(proof.round_polynomials):
            g0, g1, g2 = (
                _field(g0, "g0"), _field(g1, "g1"), _field(g2, "g2")
            )
            if (g0 + g1) % GOLDILOCKS_MODULUS != running:
                raise ProofV3VerificationError(
                    "mlpcs round polynomial does not match the running sum"
                )
            transcript = hashlib.sha256(
                transcript
                + g0.to_bytes(8, "little")
                + g1.to_bytes(8, "little")
                + g2.to_bytes(8, "little")
            ).digest()
            challenge = _derive_field(transcript, b"fold", round_index)
            challenges.append(challenge)
            inv2 = (GOLDILOCKS_MODULUS + 1) // 2
            z = challenge
            running = (
                g0 * ((z - 1) * (z - 2) % GOLDILOCKS_MODULUS)
                % GOLDILOCKS_MODULUS
                * inv2
                - g1 * (z * (z - 2) % GOLDILOCKS_MODULUS)
                + g2 * (z * (z - 1) % GOLDILOCKS_MODULUS)
                % GOLDILOCKS_MODULUS
                * inv2
            ) % GOLDILOCKS_MODULUS
            transcript = hashlib.sha256(
                transcript + proof.layer_commitments[round_index]
            ).digest()
        # Final coupling: running == final_value * prod eq(c_j, r_j).
        eq_product = 1
        for challenge, r in zip(challenges, point, strict=True):
            term = (
                (1 - challenge) * (1 - r) + challenge * r
            ) % GOLDILOCKS_MODULUS
            eq_product = eq_product * term % GOLDILOCKS_MODULUS
        if running != proof.final_value * eq_product % GOLDILOCKS_MODULUS:
            raise ProofV3VerificationError(
                "mlpcs final coupling fails: codeword does not bind the sumcheck"
            )
        # Query phase: recompute positions, check every fold layer.
        query_seed = hashlib.sha256(transcript + b"queries").digest()
        base_size = statement.codeword_size
        positions = [
            int.from_bytes(
                hashlib.sha256(
                    _QUERY_DOMAIN + query_seed + struct.pack("<I", query_index)
                ).digest()[:8],
                "little",
            )
            % (base_size // 2)
            for query_index in range(statement.query_count)
        ]
        if len(proof.layer_openings) != n + 1:
            raise ProofV3VerificationError("mlpcs layer opening count is wrong")
        roots = (commitment,) + tuple(proof.layer_commitments)
        opened: list[dict[int, int]] = []
        for layer_index, opening in enumerate(proof.layer_openings):
            size = base_size >> layer_index
            verify_goldilocks_merkle_multiopening_reference(
                roots[layer_index],
                opening,
                expected_binding_digest=statement.layer_binding_digest(
                    layer_index
                ),
                expected_leaf_count=size,
                expected_leaf_width=1,
                expected_indices=_layer_query_indices(
                    positions=positions,
                    size=size,
                    is_final=layer_index == n,
                ),
            )
            opened.append(
                {
                    index: row[0]
                    for index, row in zip(
                        opening.indices, opening.rows, strict=True
                    )
                }
            )
        inv2 = (GOLDILOCKS_MODULUS + 1) // 2
        for layer_index in range(n):
            size = base_size >> layer_index
            challenge = challenges[layer_index]
            for position in positions:
                folded_position = position % (size // 2)
                value_pos = opened[layer_index][folded_position]
                value_neg = opened[layer_index][folded_position + size // 2]
                # O(log size) point, not the O(size) full domain.
                x = statement.domain_point(layer_index, folded_position)
                even = (value_pos + value_neg) % GOLDILOCKS_MODULUS * inv2
                odd = (
                    (value_pos - value_neg)
                    % GOLDILOCKS_MODULUS
                    * inv2
                    % GOLDILOCKS_MODULUS
                    * goldilocks_inv(x)
                )
                # Lerp fold (matching the sumcheck binding): the child
                # polynomial is (1-c) * P_even + c * P_odd.
                expected_fold = (
                    (1 - challenge) * even + challenge * odd
                ) % GOLDILOCKS_MODULUS
                # The fold lands at child index p mod size_{i+1}; that index
                # is always inside the child's opened pair (or the final
                # layer's direct target).
                child = opened[layer_index + 1][position % (size // 2)]
                if expected_fold != child:
                    raise ProofV3VerificationError(
                        "mlpcs fold consistency fails at a queried position"
                    )
        # The final layer's opened values must all equal final_value (the
        # codeword of a constant polynomial is constant).
        for value in opened[n].values():
            if value != proof.final_value:
                raise ProofV3VerificationError(
                    "mlpcs final codeword is not the claimed constant"
                )
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("mlpcs proof is malformed") from exc


__all__ = [
    "GOLDILOCKS_MULTILINEAR_PCS_ABI_V3",
    "GOLDILOCKS_MULTILINEAR_PCS_BLOWUP_V3",
    "GOLDILOCKS_MULTILINEAR_PCS_QUERY_COUNT_V3",
    "MAX_GOLDILOCKS_MULTILINEAR_PCS_QUERY_COUNT_V3",
    "GoldilocksMultilinearOpeningProofV3",
    "GoldilocksMultilinearPcsStatementV3",
    "commit_goldilocks_multilinear_v3",
    "current_pcs_query_count_v3",
    "open_goldilocks_multilinear_v3",
    "pcs_query_count_v3",
    "verify_goldilocks_multilinear_opening_v3",
]
