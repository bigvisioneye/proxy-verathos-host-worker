"""Bounded CPU Merkle reference for canonical Goldilocks field rows.

This module is deliberately unregistered.  It is a small pure-Python golden
vector/reference implementation for a future proof-v3 dynamic STARK backend;
it is not imported by a miner or validator and has no payload, FRI, GPU, or
release integration.  In particular, a Merkle root and opening alone do not
prove a low-degree relation, an execution transition, or model-substitution
resistance.

Every leaf is a fixed-width row of *canonical* Goldilocks field elements.  The
only accepted field encoding is eight-byte little-endian canonical encoding.
The tree commits to a caller-supplied 32-byte binding digest, tree shape, row
width, leaf index, and row values with distinct SHA-256 domains for leaves,
internal nodes, and the final commitment.  Multiproofs use one exact,
canonical sibling-coordinate schedule: duplicate, omitted, surplus, or
reordered nodes fail closed before root verification.

The deliberately modest limits below bound this CPU reference only.  They are
not protocol parameters and are not a constraint on a qualified GPU backend.
"""

from __future__ import annotations

import hashlib
import hmac
import operator
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.goldilocks_reference import (
    MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE,
    canonical_goldilocks,
    goldilocks_to_bytes,
)


GOLDILOCKS_MERKLE_REFERENCE_ABI_V3: Final = "goldilocks.merkle.reference.v1"
GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES: Final = 32
MAX_GOLDILOCKS_MERKLE_REFERENCE_LEAF_COUNT: Final = (
    MAX_GOLDILOCKS_REFERENCE_DOMAIN_SIZE
)
MAX_GOLDILOCKS_MERKLE_REFERENCE_LEAF_WIDTH: Final = 1 << 12
MAX_GOLDILOCKS_MERKLE_REFERENCE_FIELD_VALUES: Final = 1 << 21

_LEAF_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MERKLE/V1/LEAF/SHA256"
_NODE_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MERKLE/V1/NODE/SHA256"
_ROOT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MERKLE/V1/ROOT/SHA256"


def _integer(value: object, name: str) -> int:
    # fast path: a plain int (``type is int`` excludes bool, whose type is
    # bool) needs no coercion -- the dominant case on the verify hot loop
    if type(value) is int:
        return value
    if isinstance(value, bool):
        raise ProofV3Error(f"{name} must be an integer, not boolean")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ProofV3Error(f"{name} must be an integer") from exc


def _fixed_bytes(value: object, length: int, name: str) -> bytes:
    # fast path: exact ``bytes`` is immutable and used read-only, so return
    # it without the defensive ``bytes(value)`` copy (4.6M sibling digests
    # per attention-bundle verify each copied 32 bytes otherwise)
    if type(value) is bytes:
        if len(value) != length:
            raise ProofV3Error(f"{name} must be exactly {length} bytes")
        return value
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ProofV3Error(f"{name} must be bytes")
    encoded = bytes(value)
    if len(encoded) != length:
        raise ProofV3Error(f"{name} must be exactly {length} bytes")
    return encoded


def _binding_digest(value: object, name: str = "Merkle binding_digest") -> bytes:
    return _fixed_bytes(value, GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES, name)


def _leaf_count(value: object, name: str = "Merkle leaf_count") -> int:
    # Metadata bound: opening/multiproof VALIDATION accepts native-tier
    # tree sizes; CPU tree MATERIALIZATION stays bounded separately by
    # MAX_GOLDILOCKS_MERKLE_REFERENCE_LEAF_COUNT in _field_rows.
    from verallm.proof_v3.goldilocks_reference import (
        MAX_GOLDILOCKS_NATIVE_DOMAIN_SIZE,
    )

    count = _integer(value, name)
    if (
        count < 1
        or count > MAX_GOLDILOCKS_NATIVE_DOMAIN_SIZE
        or count & (count - 1)
    ):
        raise ProofV3Error(
            "Merkle leaf_count must be a power of two within the CPU reference cap"
        )
    return count


def _leaf_width(value: object, name: str = "Merkle leaf_width") -> int:
    width = _integer(value, name)
    if width < 1 or width > MAX_GOLDILOCKS_MERKLE_REFERENCE_LEAF_WIDTH:
        raise ProofV3Error("Merkle leaf_width is outside the CPU reference cap")
    return width


def _materialize_iterable(value: object, name: str, maximum: int) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        raise ProofV3Error(f"{name} must be an iterable")
    try:
        iterator = iter(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ProofV3Error(f"{name} must be an iterable") from exc
    result: list[object] = []
    for item in iterator:
        if len(result) >= maximum:
            raise ProofV3Error(f"{name} exceeds the CPU reference cap")
        result.append(item)
    return tuple(result)


def _field_row(value: object, *, name: str, width: int | None) -> tuple[int, ...]:
    items = _materialize_iterable(
        value,
        name,
        MAX_GOLDILOCKS_MERKLE_REFERENCE_LEAF_WIDTH,
    )
    if not items:
        raise ProofV3Error(f"{name} must not be empty")
    if width is not None and len(items) != width:
        raise ProofV3Error(f"{name} must contain exactly {width} field elements")
    return tuple(
        canonical_goldilocks(item, f"{name}[{index}]")
        for index, item in enumerate(items)
    )


def _field_rows(
    value: object,
    *,
    name: str,
    expected_count: int | None,
    width: int | None,
) -> tuple[tuple[int, ...], ...]:
    # Fast path: an exact tuple of equal-width tuples of plain canonical
    # ints is ACCEPTED by the reference loop below with the identical
    # normalized result, so it can be returned as-is.  Anything else falls
    # through to the reference validation -- the acceptance set is
    # unchanged.
    if (
        type(value) is tuple
        and value
        and len(value) <= MAX_GOLDILOCKS_MERKLE_REFERENCE_LEAF_COUNT
        and (expected_count is None or len(value) == expected_count)
        and type(value[0]) is tuple
        and value[0]
    ):
        fast_width = len(value[0]) if width is None else width
        if (
            fast_width == 1
            and len(value) <= MAX_GOLDILOCKS_MERKLE_REFERENCE_FIELD_VALUES
        ):
            # width-1 bulk subpath: whole-collection C-level passes with the
            # IDENTICAL acceptance set (tuples of one plain canonical int;
            # bool is not int by type identity, so it still falls through)
            if (
                set(map(type, value)) == {tuple}
                and set(map(len, value)) == {1}
            ):
                flat = [row[0] for row in value]
                if (
                    set(map(type, flat)) == {int}
                    and min(flat) >= 0
                    and max(flat) < GOLDILOCKS_MODULUS
                ):
                    return value
        elif (
            1 <= fast_width <= MAX_GOLDILOCKS_MERKLE_REFERENCE_LEAF_WIDTH
            and len(value) * fast_width
            <= MAX_GOLDILOCKS_MERKLE_REFERENCE_FIELD_VALUES
            and all(
                type(row) is tuple
                and len(row) == fast_width
                and all(
                    type(item) is int and 0 <= item < GOLDILOCKS_MODULUS
                    for item in row
                )
                for row in value
            )
        ):
            return value
    # Rows count is bounded by the TOTAL field-value cap (every row
    # carries at least one value; the exact total is enforced in the
    # loop below) -- the CPU tree-materialization domain bound is the
    # wrong cap here: multi-token economic openings legitimately
    # reveal more width-1 leaves than one CPU-reference tree holds.
    rows = _materialize_iterable(
        value,
        name,
        MAX_GOLDILOCKS_MERKLE_REFERENCE_FIELD_VALUES,
    )
    if not rows:
        raise ProofV3Error(f"{name} must not be empty")
    if expected_count is not None and len(rows) != expected_count:
        raise ProofV3Error(f"{name} must contain exactly {expected_count} rows")
    normalized: list[tuple[int, ...]] = []
    expected_width = width
    total_values = 0
    for index, row in enumerate(rows):
        normalized_row = _field_row(
            row,
            name=f"{name}[{index}]",
            width=expected_width,
        )
        if expected_width is None:
            expected_width = len(normalized_row)
        total_values += len(normalized_row)
        if total_values > MAX_GOLDILOCKS_MERKLE_REFERENCE_FIELD_VALUES:
            raise ProofV3Error(f"{name} exceeds the CPU field-value reference cap")
        normalized.append(normalized_row)
    assert expected_width is not None
    _leaf_width(expected_width)
    return tuple(normalized)


def encode_goldilocks_merkle_leaf_reference(values: object) -> bytes:
    """Return the sole canonical byte encoding of one Goldilocks leaf row.

    The enclosing tree binds the corresponding fixed row width and leaf index,
    so this helper intentionally contains exactly the ordered 8-byte
    little-endian field encodings and no alternative packed representation.
    """

    row = _field_row(values, name="Merkle leaf", width=None)
    return b"".join(goldilocks_to_bytes(value) for value in row)


def _tree_header(binding_digest: bytes, leaf_count: int, leaf_width: int) -> bytes:
    return binding_digest + struct.pack("<II", leaf_count, leaf_width)


def _leaf_hash(
    *,
    binding_digest: bytes,
    leaf_count: int,
    leaf_width: int,
    index: int,
    row: tuple[int, ...],
) -> bytes:
    return hashlib.sha256(
        _LEAF_DOMAIN
        + _tree_header(binding_digest, leaf_count, leaf_width)
        + struct.pack("<I", index)
        + encode_goldilocks_merkle_leaf_reference(row)
    ).digest()


def _parent_hash(
    *,
    binding_digest: bytes,
    leaf_count: int,
    leaf_width: int,
    parent_level: int,
    parent_index: int,
    left: bytes,
    right: bytes,
) -> bytes:
    return hashlib.sha256(
        _NODE_DOMAIN
        + _tree_header(binding_digest, leaf_count, leaf_width)
        + struct.pack("<II", parent_level, parent_index)
        + _fixed_bytes(left, GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES, "left node")
        + _fixed_bytes(right, GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES, "right node")
    ).digest()


def _commitment_hash(
    *,
    binding_digest: bytes,
    leaf_count: int,
    leaf_width: int,
    raw_root: bytes,
) -> bytes:
    return hashlib.sha256(
        _ROOT_DOMAIN
        + _tree_header(binding_digest, leaf_count, leaf_width)
        + _fixed_bytes(
            raw_root,
            GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES,
            "Merkle raw root",
        )
    ).digest()


def _height(leaf_count: int) -> int:
    return leaf_count.bit_length() - 1


def _build_levels(
    *,
    binding_digest: bytes,
    rows: tuple[tuple[int, ...], ...],
    leaf_width: int,
) -> tuple[tuple[bytes, ...], ...]:
    leaf_count = len(rows)
    current = tuple(
        _leaf_hash(
            binding_digest=binding_digest,
            leaf_count=leaf_count,
            leaf_width=leaf_width,
            index=index,
            row=row,
        )
        for index, row in enumerate(rows)
    )
    levels: list[tuple[bytes, ...]] = [current]
    parent_level = 1
    while len(current) > 1:
        current = tuple(
            _parent_hash(
                binding_digest=binding_digest,
                leaf_count=leaf_count,
                leaf_width=leaf_width,
                parent_level=parent_level,
                parent_index=index // 2,
                left=current[index],
                right=current[index + 1],
            )
            for index in range(0, len(current), 2)
        )
        levels.append(current)
        parent_level += 1
    return tuple(levels)


def _indices(value: object, *, leaf_count: int, name: str) -> tuple[int, ...]:
    # Fast path: an exact tuple of plain ints, strictly increasing within
    # [0, leaf_count), is accepted by the reference loop below unchanged.
    if type(value) is tuple and 0 < len(value) <= leaf_count:
        previous = -1
        for item in value:
            if type(item) is not int or item <= previous or item >= leaf_count:
                break
            previous = item
        else:
            return value
    entries = _materialize_iterable(value, name, leaf_count)
    if not entries:
        raise ProofV3Error(f"{name} must not be empty")
    normalized = tuple(
        _integer(item, f"{name}[{index}]")
        for index, item in enumerate(entries)
    )
    for index, entry in enumerate(normalized):
        if entry < 0 or entry >= leaf_count:
            raise ProofV3Error(f"{name}[{index}] is outside the Merkle tree")
        if index and normalized[index - 1] >= entry:
            raise ProofV3Error(
                f"{name} must be strictly increasing without duplicates or reordering"
            )
    return normalized


def _expected_sibling_coordinates(
    *,
    leaf_count: int,
    indices: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    """Derive the only accepted ``(level, index)`` proof-node order."""

    try:
        from verallm.proof_v3.c_multiopen import (
            sibling_coordinates as _c_coords,
        )

        compiled = _c_coords(leaf_count, indices)
        if compiled is not None:
            return compiled
    except ImportError:
        pass
    current = set(indices)
    coordinates: list[tuple[int, int]] = []
    for level in range(_height(leaf_count)):
        missing = sorted(index ^ 1 for index in current if (index ^ 1) not in current)
        coordinates.extend((level, index) for index in missing)
        current = {index // 2 for index in current}
    return tuple(coordinates)


@dataclass(frozen=True, slots=True)
class GoldilocksMerkleSiblingReference:
    """One raw Merkle node in the canonical multiproof coordinate schedule."""

    level: int
    index: int
    digest: bytes

    def __post_init__(self) -> None:
        level = _integer(self.level, "Merkle sibling level")
        index = _integer(self.index, "Merkle sibling index")
        if level < 0 or level >= 1 << 32:
            raise ProofV3Error("Merkle sibling level is outside uint32")
        if index < 0 or index >= 1 << 32:
            raise ProofV3Error("Merkle sibling index is outside uint32")
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "index", index)
        object.__setattr__(
            self,
            "digest",
            _fixed_bytes(
                self.digest,
                GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES,
                "Merkle sibling digest",
            ),
        )

    @classmethod
    def _from_trusted_wire(
        cls, level: int, index: int, digest: bytes
    ) -> "GoldilocksMerkleSiblingReference":
        """Construct without per-field validation for decode hot paths.

        The caller guarantees ``level``/``index`` are plain ints from a
        fixed-width struct unpack and ``digest`` is an exact 32-byte
        slice.  Every trusted-wire opening is subsequently checked by
        :func:`verify_goldilocks_merkle_multiopening_reference` (exact
        sibling-coordinate schedule) or the compiled opening verifier
        (coordinate re-derivation + Merkle root reconstruction), so a
        malformed level/index/digest fails closed there -- the
        construction-time validation this skips is pure redundancy on
        that path.  NEVER use this to admit an unverified opening.
        """

        obj = object.__new__(cls)
        object.__setattr__(obj, "level", level)
        object.__setattr__(obj, "index", index)
        object.__setattr__(obj, "digest", digest)
        return obj


@dataclass(frozen=True, slots=True)
class GoldilocksMerkleMultiOpeningReference:
    """Canonical rows and exact sibling schedule for one multiproof.

    The opening itself validates its own canonical layout, but verification
    still requires independent verifier-owned binding, tree metadata, and
    exact selected index expectations through
    :func:`verify_goldilocks_merkle_multiopening_reference`.
    """

    binding_digest: bytes
    leaf_count: int
    leaf_width: int
    indices: tuple[int, ...]
    rows: tuple[tuple[int, ...], ...]
    siblings: tuple[GoldilocksMerkleSiblingReference, ...]
    abi_id: str = GOLDILOCKS_MERKLE_REFERENCE_ABI_V3

    def __post_init__(self) -> None:
        if self.abi_id != GOLDILOCKS_MERKLE_REFERENCE_ABI_V3:
            raise ProofV3Error("Merkle multiproof ABI is unsupported")
        binding_digest = _binding_digest(self.binding_digest)
        leaf_count = _leaf_count(self.leaf_count)
        leaf_width = _leaf_width(self.leaf_width)
        indices = _indices(
            self.indices,
            leaf_count=leaf_count,
            name="Merkle multiproof indices",
        )
        rows = _field_rows(
            self.rows,
            name="Merkle multiproof rows",
            expected_count=len(indices),
            width=leaf_width,
        )
        raw_siblings = _materialize_iterable(
            self.siblings,
            "Merkle multiproof siblings",
            MAX_GOLDILOCKS_MERKLE_REFERENCE_LEAF_COUNT,
        )
        if not all(
            isinstance(node, GoldilocksMerkleSiblingReference)
            for node in raw_siblings
        ):
            raise ProofV3Error("Merkle multiproof siblings have an unexpected type")
        siblings = tuple(raw_siblings)
        expected_coordinates = _expected_sibling_coordinates(
            leaf_count=leaf_count,
            indices=indices,
        )
        coordinates = tuple((node.level, node.index) for node in siblings)
        if coordinates != expected_coordinates:
            raise ProofV3Error(
                "Merkle multiproof sibling coordinates are incomplete, duplicate, "
                "or reordered"
            )
        object.__setattr__(self, "binding_digest", binding_digest)
        object.__setattr__(self, "leaf_count", leaf_count)
        object.__setattr__(self, "leaf_width", leaf_width)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "siblings", siblings)


def _normalized_levels(
    value: object,
    *,
    leaf_count: int,
) -> tuple[tuple[bytes, ...], ...]:
    expected_levels = _height(leaf_count) + 1
    raw_levels = _materialize_iterable(value, "Merkle tree levels", expected_levels)
    if len(raw_levels) != expected_levels:
        raise ProofV3Error("Merkle tree has an unexpected number of levels")
    levels: list[tuple[bytes, ...]] = []
    for level, raw_nodes in enumerate(raw_levels):
        expected_nodes = leaf_count >> level
        nodes = _materialize_iterable(
            raw_nodes,
            f"Merkle tree level {level}",
            expected_nodes,
        )
        if len(nodes) != expected_nodes:
            raise ProofV3Error(
                f"Merkle tree level {level} has an unexpected node count"
            )
        levels.append(
            tuple(
                _fixed_bytes(
                    node,
                    GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES,
                    f"Merkle tree level {level} node {index}",
                )
                for index, node in enumerate(nodes)
            )
        )
    return tuple(levels)


@dataclass(frozen=True, slots=True)
class GoldilocksMerkleTreeReference:
    """A small immutable Goldilocks-row tree retained only for golden vectors.

    Construct it with :meth:`from_rows`.  Its retained rows and internal nodes
    intentionally make it unsuitable for real long-context proving; a future
    GPU backend must stream/build its own authenticated oracle instead.
    """

    binding_digest: bytes
    leaf_count: int
    leaf_width: int
    commitment: bytes
    rows: tuple[tuple[int, ...], ...]
    levels: tuple[tuple[bytes, ...], ...]

    def __post_init__(self) -> None:
        binding_digest = _binding_digest(self.binding_digest)
        leaf_count = _leaf_count(self.leaf_count)
        leaf_width = _leaf_width(self.leaf_width)
        rows = _field_rows(
            self.rows,
            name="Merkle tree rows",
            expected_count=leaf_count,
            width=leaf_width,
        )
        levels = _normalized_levels(self.levels, leaf_count=leaf_count)
        expected_levels = _build_levels(
            binding_digest=binding_digest,
            rows=rows,
            leaf_width=leaf_width,
        )
        if len(levels) != len(expected_levels) or any(
            len(actual) != len(expected)
            or any(
                not hmac.compare_digest(actual_node, expected_node)
                for actual_node, expected_node in zip(actual, expected, strict=True)
            )
            for actual, expected in zip(levels, expected_levels, strict=True)
        ):
            raise ProofV3Error("Merkle tree levels do not match canonical rows")
        commitment = _fixed_bytes(
            self.commitment,
            GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES,
            "Merkle commitment",
        )
        expected_commitment = _commitment_hash(
            binding_digest=binding_digest,
            leaf_count=leaf_count,
            leaf_width=leaf_width,
            raw_root=levels[-1][0],
        )
        if not hmac.compare_digest(commitment, expected_commitment):
            raise ProofV3Error("Merkle commitment does not match canonical rows")
        object.__setattr__(self, "binding_digest", binding_digest)
        object.__setattr__(self, "leaf_count", leaf_count)
        object.__setattr__(self, "leaf_width", leaf_width)
        object.__setattr__(self, "commitment", commitment)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "levels", levels)

    @classmethod
    def from_rows(
        cls,
        rows: object,
        *,
        binding_digest: object,
    ) -> "GoldilocksMerkleTreeReference":
        """Commit an exact power-of-two table of equal-width field rows."""

        normalized_rows = _field_rows(
            rows,
            name="Merkle tree rows",
            expected_count=None,
            width=None,
        )
        leaf_count = _leaf_count(len(normalized_rows))
        leaf_width = _leaf_width(len(normalized_rows[0]))
        binding = _binding_digest(binding_digest)
        levels = _build_levels(
            binding_digest=binding,
            rows=normalized_rows,
            leaf_width=leaf_width,
        )
        commitment = _commitment_hash(
            binding_digest=binding,
            leaf_count=leaf_count,
            leaf_width=leaf_width,
            raw_root=levels[-1][0],
        )
        return cls(
            binding_digest=binding,
            leaf_count=leaf_count,
            leaf_width=leaf_width,
            commitment=commitment,
            rows=normalized_rows,
            levels=levels,
        )

    def open(self, indices: object) -> GoldilocksMerkleMultiOpeningReference:
        """Return rows and the only valid canonical multiproof for ``indices``."""

        selected = _indices(
            indices,
            leaf_count=self.leaf_count,
            name="Merkle opening indices",
        )
        siblings = tuple(
            GoldilocksMerkleSiblingReference(
                level=level,
                index=index,
                digest=self.levels[level][index],
            )
            for level, index in _expected_sibling_coordinates(
                leaf_count=self.leaf_count,
                indices=selected,
            )
        )
        return GoldilocksMerkleMultiOpeningReference(
            binding_digest=self.binding_digest,
            leaf_count=self.leaf_count,
            leaf_width=self.leaf_width,
            indices=selected,
            rows=tuple(self.rows[index] for index in selected),
            siblings=siblings,
        )


def verify_goldilocks_merkle_multiopening_reference(
    commitment: object,
    opening: object,
    *,
    expected_binding_digest: object,
    expected_leaf_count: object,
    expected_leaf_width: object,
    expected_indices: object,
) -> None:
    """Fail closed unless an opening matches verifier-owned exact expectations.

    ``expected_*`` inputs are deliberately required rather than read from the
    prover's opening.  A caller must derive them from the authenticated profile
    and post-commitment challenge before invoking this reference verifier.
    """

    try:
        expected_binding = _binding_digest(
            expected_binding_digest,
            "expected Merkle binding_digest",
        )
        leaf_count = _leaf_count(expected_leaf_count, "expected Merkle leaf_count")
        leaf_width = _leaf_width(expected_leaf_width, "expected Merkle leaf_width")
        indices = _indices(
            expected_indices,
            leaf_count=leaf_count,
            name="expected Merkle indices",
        )
        root = _fixed_bytes(
            commitment,
            GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES,
            "expected Merkle commitment",
        )
    except ProofV3Error as exc:
        raise ProofV3VerificationError("Merkle verifier expectation is malformed") from exc

    if not isinstance(opening, GoldilocksMerkleMultiOpeningReference):
        raise ProofV3VerificationError("Merkle opening has an unexpected type")
    if opening.binding_digest != expected_binding:
        raise ProofV3VerificationError("Merkle opening binding_digest is unexpected")
    if opening.leaf_count != leaf_count or opening.leaf_width != leaf_width:
        raise ProofV3VerificationError("Merkle opening tree metadata is unexpected")
    if opening.indices != indices:
        raise ProofV3VerificationError("Merkle opening indices are unexpected")

    expected_coordinates = _expected_sibling_coordinates(
        leaf_count=leaf_count,
        indices=indices,
    )
    actual_coordinates = tuple((node.level, node.index) for node in opening.siblings)
    if actual_coordinates != expected_coordinates:
        raise ProofV3VerificationError(
            "Merkle opening sibling coordinates are incomplete, duplicate, or "
            "reordered"
        )

    # hot path: identical hash bytes, with the constant header/prefixes
    # hoisted and sibling digests validated ONCE at ingestion
    header = _tree_header(expected_binding, leaf_count, leaf_width)
    leaf_prefix = _LEAF_DOMAIN + header
    node_prefix = _NODE_DOMAIN + header
    sha = hashlib.sha256
    pack_i = struct.Struct("<I").pack
    pack_ii = struct.Struct("<II").pack
    sibling_digests = tuple(
        _fixed_bytes(
            node.digest, GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES,
            "sibling digest")
        for node in opening.siblings
    )
    if leaf_width == 1:
        # width-1 fast path; compiled reconstruction when available
        row_values = []
        for row in opening.rows:
            if len(row) != 1:
                raise ProofV3VerificationError(
                    "Merkle opening row width is unexpected")
            value = row[0]
            if not isinstance(value, int) or isinstance(value, bool) or (
                not 0 <= value < GOLDILOCKS_MODULUS
            ):
                raise ProofV3VerificationError(
                    "Merkle opening row value is not canonical")
            row_values.append(value)
        try:
            from verallm.proof_v3.c_multiopen import reconstruct_w1
        except ImportError:
            reconstruct_w1 = None
        if reconstruct_w1 is not None:
            result = reconstruct_w1(
                leaf_prefix, node_prefix, leaf_count,
                indices, row_values,
                [c[0] for c in expected_coordinates],
                [c[1] for c in expected_coordinates],
                b"".join(sibling_digests))
            if isinstance(result, bytes):
                actual_root = _commitment_hash(
                    binding_digest=expected_binding,
                    leaf_count=leaf_count,
                    leaf_width=leaf_width,
                    raw_root=result,
                )
                if not hmac.compare_digest(root, actual_root):
                    raise ProofV3VerificationError(
                        "Merkle opening does not match commitment")
                return
            if isinstance(result, int):
                raise ProofV3VerificationError(
                    "Merkle opening reconstruction is incomplete")
        current = {}
        for index, row in zip(indices, opening.rows, strict=True):
            if len(row) != 1:
                raise ProofV3VerificationError(
                    "Merkle opening row width is unexpected")
            value = row[0]
            if not isinstance(value, int) or isinstance(value, bool) or (
                not 0 <= value < GOLDILOCKS_MODULUS
            ):
                raise ProofV3VerificationError(
                    "Merkle opening row value is not canonical")
            current[index] = sha(
                leaf_prefix + pack_i(index)
                + value.to_bytes(8, "little")
            ).digest()
    else:
        # width-N compiled reconstruction (chunk-leaf oracles); rows were
        # normalized canonical at opening construction
        try:
            from verallm.proof_v3.c_multiopen import reconstruct_wn
        except ImportError:
            reconstruct_wn = None
        if reconstruct_wn is not None:
            flat: list[int] = []
            for row in opening.rows:
                flat.extend(row)
            result = reconstruct_wn(
                leaf_prefix, node_prefix, leaf_count,
                indices, flat, leaf_width,
                [c[0] for c in expected_coordinates],
                [c[1] for c in expected_coordinates],
                b"".join(sibling_digests))
            if isinstance(result, bytes):
                actual_root = _commitment_hash(
                    binding_digest=expected_binding,
                    leaf_count=leaf_count,
                    leaf_width=leaf_width,
                    raw_root=result,
                )
                if not hmac.compare_digest(root, actual_root):
                    raise ProofV3VerificationError(
                        "Merkle opening does not match commitment")
                return
            if isinstance(result, int):
                raise ProofV3VerificationError(
                    "Merkle opening reconstruction is incomplete")
        current = {
            index: sha(
                leaf_prefix + pack_i(index)
                + encode_goldilocks_merkle_leaf_reference(row)
            ).digest()
            for index, row in zip(indices, opening.rows, strict=True)
        }
    by_level: dict[int, list[int]] = {}
    for level, sibling_index in expected_coordinates:
        by_level.setdefault(level, []).append(sibling_index)
    sibling_offset = 0
    for level in range(_height(leaf_count)):
        active_indices = tuple(sorted(current))
        for sibling_index in by_level.get(level, ()):
            current[sibling_index] = sibling_digests[sibling_offset]
            sibling_offset += 1
        parents: dict[int, bytes] = {}
        get = current.get
        for parent_index in sorted({index // 2 for index in active_indices}):
            left = get(parent_index * 2)
            right = get(parent_index * 2 + 1)
            if left is None or right is None:
                raise ProofV3VerificationError("Merkle opening is missing a sibling node")
            parents[parent_index] = sha(
                node_prefix + pack_ii(level + 1, parent_index)
                + left + right
            ).digest()
        current = parents
    if sibling_offset != len(opening.siblings) or set(current) != {0}:
        raise ProofV3VerificationError("Merkle opening reconstruction is incomplete")
    actual_root = _commitment_hash(
        binding_digest=expected_binding,
        leaf_count=leaf_count,
        leaf_width=leaf_width,
        raw_root=current[0],
    )
    if not hmac.compare_digest(root, actual_root):
        raise ProofV3VerificationError("Merkle opening does not match commitment")


__all__ = [
    "GOLDILOCKS_MERKLE_REFERENCE_ABI_V3",
    "GOLDILOCKS_MERKLE_REFERENCE_DIGEST_BYTES",
    "MAX_GOLDILOCKS_MERKLE_REFERENCE_FIELD_VALUES",
    "MAX_GOLDILOCKS_MERKLE_REFERENCE_LEAF_COUNT",
    "MAX_GOLDILOCKS_MERKLE_REFERENCE_LEAF_WIDTH",
    "GoldilocksMerkleMultiOpeningReference",
    "GoldilocksMerkleSiblingReference",
    "GoldilocksMerkleTreeReference",
    "encode_goldilocks_merkle_leaf_reference",
    "verify_goldilocks_merkle_multiopening_reference",
]
