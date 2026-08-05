"""Compact wire codec for proof-v3 transport.

Pickle carries heavy per-object overhead on the deep opening
structures that dominate proof bytes. This codec struct-encodes the
PCS openings (rounds, layer roots, per-layer indices/values/siblings)
and pickles the remaining scaffolding around them; decode restores
byte-identical objects.
"""

from __future__ import annotations

import pickle
import struct
from typing import Final

from verallm.proof_v3.goldilocks_merkle_reference import (
    GoldilocksMerkleMultiOpeningReference,
    GoldilocksMerkleSiblingReference,
)
from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
    GoldilocksMultilinearOpeningProofV3,
)

_MAGIC: Final = b"VW3\x01"


class _EncodedOpening:
    __slots__ = ("data",)

    def __init__(self, data: bytes) -> None:
        self.data = data


def _pack_opening(op: GoldilocksMultilinearOpeningProofV3) -> bytes:
    parts = [struct.pack(
        "<QQH", op.claimed_value, op.final_value,
        len(op.round_polynomials))]
    for g0, g1, g2 in op.round_polynomials:
        parts.append(struct.pack("<QQQ", g0, g1, g2))
    parts.append(struct.pack("<H", len(op.layer_commitments)))
    parts.extend(op.layer_commitments)
    parts.append(struct.pack("<H", len(op.layer_openings)))
    for lo in op.layer_openings:
        parts.append(struct.pack(
            "<32sQIHH", lo.binding_digest, lo.leaf_count,
            len(lo.indices), lo.leaf_width, len(lo.siblings)))
        for index in lo.indices:
            parts.append(struct.pack("<I", index))
        for row in lo.rows:
            parts.append(struct.pack("<Q", row[0]))
        for node in lo.siblings:
            parts.append(struct.pack("<BI", node.level, node.index))
            parts.append(node.digest)
    return b"".join(parts)


def _unpack_opening(data: bytes) -> GoldilocksMultilinearOpeningProofV3:
    off = 0
    claimed, final_value, n_rounds = struct.unpack_from("<QQH", data, off)
    off += 18
    rounds = []
    for _ in range(n_rounds):
        rounds.append(struct.unpack_from("<QQQ", data, off))
        off += 24
    (n_roots,) = struct.unpack_from("<H", data, off)
    off += 2
    roots = []
    for _ in range(n_roots):
        roots.append(data[off:off + 32])
        off += 32
    (n_layers,) = struct.unpack_from("<H", data, off)
    off += 2
    layers = []
    for _ in range(n_layers):
        binding, leaf_count, n_idx, width, n_sib = struct.unpack_from(
            "<32sQIHH", data, off)
        off += 48
        indices = []
        for _ in range(n_idx):
            indices.append(struct.unpack_from("<I", data, off)[0])
            off += 4
        rows = []
        for _ in range(n_idx):
            rows.append((struct.unpack_from("<Q", data, off)[0],))
            off += 8
        siblings = []
        for _ in range(n_sib):
            level, index = struct.unpack_from("<BI", data, off)
            off += 5
            digest = data[off:off + 32]
            off += 32
            siblings.append(GoldilocksMerkleSiblingReference(
                level=level, index=index, digest=digest))
        layers.append(GoldilocksMerkleMultiOpeningReference(
            binding_digest=binding, leaf_count=leaf_count,
            leaf_width=width, indices=tuple(indices), rows=tuple(rows),
            siblings=tuple(siblings)))
    return GoldilocksMultilinearOpeningProofV3(
        claimed_value=claimed,
        round_polynomials=tuple(tuple(r) for r in rounds),
        layer_commitments=tuple(roots),
        final_value=final_value,
        layer_openings=tuple(layers),
    )


def _transform(obj, encode: bool):
    if encode and isinstance(obj, GoldilocksMultilinearOpeningProofV3):
        return _EncodedOpening(_pack_opening(obj))
    if not encode and isinstance(obj, _EncodedOpening):
        return _unpack_opening(obj.data)
    if isinstance(obj, tuple):
        return tuple(_transform(v, encode) for v in obj)
    if isinstance(obj, list):
        return [_transform(v, encode) for v in obj]
    if isinstance(obj, dict):
        return {k: _transform(v, encode) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        import dataclasses

        changes = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            new_value = _transform(value, encode)
            if new_value is not value:
                changes[field.name] = new_value
        return dataclasses.replace(obj, **changes) if changes else obj
    return obj


def compact_dumps(proof) -> bytes:
    return _MAGIC + pickle.dumps(
        _transform(proof, encode=True), protocol=5)


def compact_loads(data: bytes):
    if not data.startswith(_MAGIC):
        raise ValueError("not a proof-v3 compact wire payload")
    return _transform(pickle.loads(data[len(_MAGIC):]), encode=False)
