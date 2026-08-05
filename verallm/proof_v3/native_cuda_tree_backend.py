"""Fused GPU SHA-256 Merkle tree hasher for proof-v3 (byte-exact roots).

Reproduces goldilocks_merkle_reference commitments on GPU: all leaves hash
in parallel, then each tree level's parents hash in parallel. Byte-exact so
the produced root equals the reference root and validator recomputation.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Final

from verallm.proof_v3.errors import ProofV3Error

_LEAF_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MERKLE/V1/LEAF/SHA256"
_NODE_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MERKLE/V1/NODE/SHA256"
_ROOT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MERKLE/V1/ROOT/SHA256"


def load_tree_kernels():
    try:
        from verathos_proof_v3_cuda import (
            load_tree_kernels as load_precompiled_tree_kernels,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "verathos_proof_v3_cuda":
            raise ProofV3Error(
                "precompiled proof-v3 CUDA tree runtime is incompatible"
            ) from exc
    except ImportError as exc:
        raise ProofV3Error(
            "precompiled proof-v3 CUDA tree runtime is incompatible"
        ) from exc
    else:
        try:
            return load_precompiled_tree_kernels()
        except (ImportError, OSError, RuntimeError) as exc:
            raise ProofV3Error(
                "precompiled proof-v3 CUDA tree runtime failed to load"
            ) from exc

    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError as exc:  # pragma: no cover
        raise ProofV3Error("fused tree tier requires torch") from exc
    if not torch.cuda.is_available():
        raise ProofV3Error("fused tree tier requires a CUDA device")
    import os

    source = os.path.join(os.path.dirname(__file__), "gl_sha256_tree.cu")
    # _v2: width-N chunk leaves (512B sha buffer + leaf_hash_wn_base);
    # the new build name sidesteps any stale cached build artifacts
    return load(name="gl_sha256_tree_v2", sources=[source],
                extra_cuda_cflags=["-O3"], verbose=False)


def _header(binding_digest: bytes, leaf_count: int, leaf_width: int) -> bytes:
    return binding_digest + struct.pack("<II", leaf_count, leaf_width)


def fused_merkle_root_wn(extension, values, *, binding_digest: bytes,
                         leaf_width: int) -> bytes:
    """Byte-exact width-N Merkle commitment on GPU.

    ``values``: CUDA int64, ``leaf_count * leaf_width`` elements in
    row-major leaf order. Byte-identical to the CPU reference tree at
    the same ``leaf_width``."""

    import torch

    width = int(leaf_width)
    total = values.numel()
    if width < 1 or total % width:
        raise ProofV3Error("fused tree values do not tile the leaf width")
    n = total // width
    if n < 1 or n & (n - 1):
        raise ProofV3Error("fused tree leaf count must be a power of two")
    header = _header(binding_digest, n, width)
    leaf_prefix = torch.tensor(
        list(_LEAF_DOMAIN + header), dtype=torch.uint8, device="cuda"
    )
    node_prefix = torch.tensor(
        list(_NODE_DOMAIN + header), dtype=torch.uint8, device="cuda"
    )
    level = extension.leaf_hash_wn_base(
        leaf_prefix, values.contiguous(), 0, width)
    count = n
    plevel = 1
    while count > 1:
        parents = count // 2
        children = level.view(parents, 64)
        level = extension.node_hash(node_prefix, plevel, children.reshape(-1))
        count = parents
        plevel += 1
    raw_root = bytes(level.cpu().tolist())
    return hashlib.sha256(_ROOT_DOMAIN + header + raw_root).digest()


def fused_merkle_sibling_paths_wn(
    extension, values, *, binding_digest: bytes, leaf_width: int,
    leaf_indices,
) -> tuple[bytes, dict[tuple[int, int], bytes]]:
    """Sparse multiproof extraction: device-resident levels, host
    transfer ONLY of the sibling digests the opening needs.

    Returns ``(commitment, {(level, index): digest})`` -- the
    recomputed tree commitment (for the caller's corruption
    self-check against the committed root) plus exactly the canonical
    sibling frontier of ``leaf_indices`` (level 0 = leaves). The full
    tree levels never leave the GPU -- the pre-chunk-leaf design
    transferred every level to host per opened oracle, which measured
    as the dominant prover cost."""

    import torch

    from verallm.proof_v3.c_multiopen import sibling_coordinates

    width = int(leaf_width)
    total = values.numel()
    if width < 1 or total % width:
        raise ProofV3Error("fused tree values do not tile the leaf width")
    n = total // width
    if n < 1 or n & (n - 1):
        raise ProofV3Error("fused tree leaf count must be a power of two")
    indices = sorted({int(i) for i in leaf_indices})
    if not indices or indices[-1] >= n:
        raise ProofV3Error("sibling extraction indices are out of range")
    coords = sibling_coordinates(n, indices)
    if coords is None:
        raise ProofV3Error("sibling schedule unavailable")
    by_level: dict[int, list[int]] = {}
    for level, index in coords:
        by_level.setdefault(int(level), []).append(int(index))
    header = _header(binding_digest, n, width)
    leaf_prefix = torch.tensor(
        list(_LEAF_DOMAIN + header), dtype=torch.uint8, device="cuda"
    )
    node_prefix = torch.tensor(
        list(_NODE_DOMAIN + header), dtype=torch.uint8, device="cuda"
    )
    out: dict[tuple[int, int], bytes] = {}
    level_bytes = extension.leaf_hash_wn_base(
        leaf_prefix, values.contiguous(), 0, width)
    count = n
    level_no = 0
    plevel = 1
    while True:
        wanted = by_level.get(level_no)
        if wanted:
            digest_view = level_bytes.view(count, 32)
            gathered = digest_view[
                torch.tensor(wanted, dtype=torch.long, device="cuda")
            ].cpu()
            for slot, index in enumerate(wanted):
                out[(level_no, index)] = bytes(
                    gathered[slot].tolist())
        if count == 1:
            break
        parents = count // 2
        children = level_bytes.view(parents, 64)
        level_bytes = extension.node_hash(
            node_prefix, plevel, children.reshape(-1))
        count = parents
        level_no += 1
        plevel += 1
    raw_root = bytes(level_bytes.cpu().tolist())
    commitment = hashlib.sha256(_ROOT_DOMAIN + header + raw_root).digest()
    return commitment, out


def _segmented_leaf_digests_wn(
    extension, source_rows, *, leaf_prefix, row_pad: int, col_pad: int,
    leaf_width: int, negative_offset: int, segment_rows: int,
):
    """Width-N leaf digests of the padded signed->field grid WITHOUT
    materializing it: row blocks stream through pad + field map + leaf
    hashing; only the 32B/leaf digest vector stays device-resident.
    ``source_rows`` is a 2-D signed integer tensor (CPU or CUDA)."""

    import torch

    width = int(leaf_width)
    chunks_per_row = col_pad // width
    leaf_count = row_pad * chunks_per_row
    digests = torch.empty(
        leaf_count * 32, dtype=torch.uint8, device="cuda")
    src_rows = int(source_rows.shape[0])
    src_cols = int(source_rows.shape[1])
    for start in range(0, row_pad, segment_rows):
        stop = min(row_pad, start + segment_rows)
        block = torch.zeros(
            (stop - start, col_pad), dtype=torch.int64, device="cuda")
        avail = min(stop, src_rows) - start
        if avail > 0:
            block[:avail, :src_cols] = source_rows[start:start + avail].to(
                device="cuda", dtype=torch.int64)
        field = torch.where(
            block < 0, block + negative_offset, block).reshape(-1)
        segment = extension.leaf_hash_wn_base(
            leaf_prefix, field, start * chunks_per_row, width)
        digests[start * chunks_per_row * 32:stop * chunks_per_row * 32] = (
            segment)
        del block, field, segment
    return digests, leaf_count


def fused_merkle_root_wn_streamed(
    extension, source_rows, *, binding_digest: bytes, leaf_width: int,
    row_pad: int, col_pad: int, negative_offset: int,
    segment_rows: int = 2048,
) -> bytes:
    """Byte-exact width-N root from a signed source tensor, streamed:
    the padded field grid never materializes (bounded device transients
    at any context/width)."""

    import torch

    width = int(leaf_width)
    n = row_pad * col_pad // width
    if n < 1 or n & (n - 1):
        raise ProofV3Error("fused tree leaf count must be a power of two")
    header = _header(binding_digest, n, width)
    leaf_prefix = torch.tensor(
        list(_LEAF_DOMAIN + header), dtype=torch.uint8, device="cuda")
    node_prefix = torch.tensor(
        list(_NODE_DOMAIN + header), dtype=torch.uint8, device="cuda")
    level, count = _segmented_leaf_digests_wn(
        extension, source_rows, leaf_prefix=leaf_prefix, row_pad=row_pad,
        col_pad=col_pad, leaf_width=width,
        negative_offset=negative_offset, segment_rows=segment_rows)
    plevel = 1
    while count > 1:
        parents = count // 2
        children = level.view(parents, 64)
        level = extension.node_hash(node_prefix, plevel, children.reshape(-1))
        count = parents
        plevel += 1
    raw_root = bytes(level.cpu().tolist())
    return hashlib.sha256(_ROOT_DOMAIN + header + raw_root).digest()


def fused_merkle_sibling_paths_wn_streamed(
    extension, source_rows, *, binding_digest: bytes, leaf_width: int,
    row_pad: int, col_pad: int, negative_offset: int, leaf_indices,
    segment_rows: int = 2048,
) -> tuple[bytes, dict[tuple[int, int], bytes]]:
    """Streamed variant of :func:`fused_merkle_sibling_paths_wn`: leaf
    digests build segment-by-segment from the signed source tensor, so
    the padded field grid never materializes at audit time either."""

    import torch

    from verallm.proof_v3.c_multiopen import sibling_coordinates

    width = int(leaf_width)
    n = row_pad * col_pad // width
    if n < 1 or n & (n - 1):
        raise ProofV3Error("fused tree leaf count must be a power of two")
    indices = sorted({int(i) for i in leaf_indices})
    if not indices or indices[-1] >= n:
        raise ProofV3Error("sibling extraction indices are out of range")
    coords = sibling_coordinates(n, indices)
    if coords is None:
        raise ProofV3Error("sibling schedule unavailable")
    by_level: dict[int, list[int]] = {}
    for level_no, index in coords:
        by_level.setdefault(int(level_no), []).append(int(index))
    header = _header(binding_digest, n, width)
    leaf_prefix = torch.tensor(
        list(_LEAF_DOMAIN + header), dtype=torch.uint8, device="cuda")
    node_prefix = torch.tensor(
        list(_NODE_DOMAIN + header), dtype=torch.uint8, device="cuda")
    level_bytes, count = _segmented_leaf_digests_wn(
        extension, source_rows, leaf_prefix=leaf_prefix, row_pad=row_pad,
        col_pad=col_pad, leaf_width=width,
        negative_offset=negative_offset, segment_rows=segment_rows)
    out: dict[tuple[int, int], bytes] = {}
    level_no = 0
    plevel = 1
    while True:
        wanted = by_level.get(level_no)
        if wanted:
            digest_view = level_bytes.view(count, 32)
            gathered = digest_view[
                torch.tensor(wanted, dtype=torch.long, device="cuda")
            ].cpu()
            for slot, index in enumerate(wanted):
                out[(level_no, index)] = bytes(gathered[slot].tolist())
        if count == 1:
            break
        parents = count // 2
        children = level_bytes.view(parents, 64)
        level_bytes = extension.node_hash(
            node_prefix, plevel, children.reshape(-1))
        count = parents
        level_no += 1
        plevel += 1
    raw_root = bytes(level_bytes.cpu().tolist())
    commitment = hashlib.sha256(_ROOT_DOMAIN + header + raw_root).digest()
    return commitment, out


def fused_merkle_root_w1(extension, values, *, binding_digest: bytes) -> bytes:
    """Byte-exact width-1 Merkle commitment on GPU. `values` = CUDA int64."""

    import torch

    n = values.numel()
    if n < 1 or n & (n - 1):
        raise ProofV3Error("fused tree leaf count must be a power of two")
    header = _header(binding_digest, n, 1)
    leaf_prefix = torch.tensor(
        list(_LEAF_DOMAIN + header), dtype=torch.uint8, device="cuda"
    )
    node_prefix = torch.tensor(
        list(_NODE_DOMAIN + header), dtype=torch.uint8, device="cuda"
    )
    level = extension.leaf_hash_w1(leaf_prefix, values.contiguous())  # n*32
    count = n
    plevel = 1
    while count > 1:
        parents = count // 2
        children = level.view(parents, 64)  # left||right per parent
        level = extension.node_hash(node_prefix, plevel, children.reshape(-1))
        count = parents
        plevel += 1
    raw_root = bytes(level.cpu().tolist())
    # commitment hash is a single host SHA (cheap, once per tree)
    return hashlib.sha256(
        _ROOT_DOMAIN + header + raw_root
    ).digest()


def fused_merkle_levels_w1(
    extension, values, *, binding_digest: bytes,
    segment_leaves: int = 1 << 22,
) -> tuple[bytes, tuple[bytes, ...]]:
    """Byte-exact width-1 tree with EVERY level retained (GPU hashed).

    Returns ``(commitment, levels)`` where ``levels[i]`` is the
    concatenated 32-byte digests of level ``i`` (level 0 = leaves, last
    level = single raw root) -- exactly the reference tree's node
    layout, so authentication paths can be sliced out without any host
    hashing.
    """

    n = values.numel()
    if n < 1 or n & (n - 1):
        raise ProofV3Error("fused tree leaf count must be a power of two")
    header = _header(binding_digest, n, 1)
    import torch

    leaf_prefix = torch.tensor(
        list(_LEAF_DOMAIN + header), dtype=torch.uint8, device="cuda"
    )
    node_prefix = torch.tensor(
        list(_NODE_DOMAIN + header), dtype=torch.uint8, device="cuda"
    )
    seg = segment_leaves
    if n > seg and hasattr(extension, "leaf_hash_w1_base"):
        # SEGMENT-STREAMED build (byte-identical): the always-on serving
        # commit runs beside vLLM, so transient device levels are capped
        # at one segment's chain instead of 2x32xN bytes.  The *_base
        # kernels bind the GLOBAL leaf/node indices per segment.
        seg_bits = seg.bit_length() - 1
        segments = n // seg
        host_levels = [
            bytearray(32 * (n >> lvl)) for lvl in range(n.bit_length())]
        sub_roots = torch.empty(
            32 * segments, dtype=torch.uint8, device="cuda")
        for s in range(segments):
            level = extension.leaf_hash_w1_base(
                leaf_prefix, values[s * seg:(s + 1) * seg].contiguous(),
                s * seg)
            host_levels[0][32 * s * seg:32 * (s + 1) * seg] = (
                level.cpu().numpy().tobytes())
            count, plevel = seg, 1
            while count > 1:
                parents = count // 2
                children = level.view(parents, 64)
                level = extension.node_hash_base(
                    node_prefix, plevel, children.reshape(-1),
                    s * (seg >> plevel))
                width = seg >> plevel
                host_levels[plevel][32 * s * width:32 * (s + 1) * width] = (
                    level.cpu().numpy().tobytes())
                count, plevel = parents, plevel + 1
            sub_roots[32 * s:32 * (s + 1)].copy_(level)
        level = sub_roots
        count, plevel = segments, seg_bits + 1
        while count > 1:
            parents = count // 2
            children = level.view(parents, 64)
            level = extension.node_hash(
                node_prefix, plevel, children.reshape(-1))
            host_levels[plevel][:32 * parents] = (
                level.cpu().numpy().tobytes())
            count, plevel = parents, plevel + 1
        levels = tuple(bytes(piece) for piece in host_levels)
        commitment = hashlib.sha256(
            _ROOT_DOMAIN + header + levels[-1]).digest()
        return commitment, levels
    level = extension.leaf_hash_w1(leaf_prefix, values.contiguous())
    device_levels = [level]
    count = n
    plevel = 1
    while count > 1:
        parents = count // 2
        children = level.view(parents, 64)
        level = extension.node_hash(node_prefix, plevel, children.reshape(-1))
        device_levels.append(level)
        count = parents
        plevel += 1
    levels = tuple(
        tensor.cpu().numpy().tobytes() for tensor in device_levels
    )
    commitment = hashlib.sha256(
        _ROOT_DOMAIN + header + levels[-1]
    ).digest()
    return commitment, levels


__all__ = [
    "fused_merkle_levels_w1",
    "fused_merkle_root_w1",
    "load_tree_kernels",
]


def fused_commit_multilinear(fold_extension, tree_extension, evaluations_device,
                             *, statement):
    """Byte-exact fused multilinear commit: RS-encode (NTT) + Merkle root.

    Fully device-resident (no host round-trips): the dominant prover cost.
    A40: ~0.5 ms vs ~865 ms pure-Python at 2^13 leaves (~1557x), byte-
    identical to commit_goldilocks_multilinear_v3.

    ``evaluations_device`` is a CUDA int64 tensor of the raw eval vector;
    ``statement`` is a GoldilocksMultilinearPcsStatementV3.
    """

    import torch

    from verallm.proof_v3.goldilocks_reference import (
        goldilocks_principal_root_of_unity,
    )
    from verallm.proof_v3.native_cuda_fold_backend import fused_ntt_goldilocks

    n = evaluations_device.numel()
    if n != statement.leaf_count:
        raise ProofV3Error("fused commit eval count does not match the statement")
    codeword_size = statement.codeword_size
    shift = statement.domain_shift(0)
    generator = goldilocks_principal_root_of_unity(codeword_size)
    padded = torch.zeros(codeword_size, dtype=torch.int64, device="cuda")
    padded[:n] = evaluations_device
    codeword = fused_ntt_goldilocks(
        fold_extension, padded, shift=shift, generator=generator
    )
    return fused_merkle_root_w1(
        tree_extension, codeword,
        binding_digest=statement.layer_binding_digest(0),
    )


__all__.append("fused_commit_multilinear")
