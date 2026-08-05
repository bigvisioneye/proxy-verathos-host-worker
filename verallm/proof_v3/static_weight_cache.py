"""Persistent, signed-root-keyed proof-v3 static weight material.

The authority-signed manifest authenticates the Merkle root, dimensions and
chunk geometry.  A miner can therefore build the canonical int8 surrogate and
tree once, persist them, and memory-map the surrogate on later starts.  Cache
contents are never trusted by a validator: a stale/corrupt row still fails its
Merkle opening against the signed root.
"""

from __future__ import annotations

import json
import hashlib
import os
import stat
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

from verallm.proof_v3.errors import ProofV3Error

STATIC_WEIGHT_CACHE_ABI_V3 = "verathos.proof_v3.static_weight_cache.v1"
COMPACT_STATIC_WEIGHT_CACHE_ABI_V3 = (
    "verathos.proof_v3.static_weight_cache.sparse_paths.v1"
)
_COMPACT_LOCAL_TREE_HEIGHT = 8
_COMPACT_LOCAL_CACHE_SIZE = 64


def default_static_weight_cache_dir_v3() -> Path:
    configured = os.environ.get("VERATHOS_PROOF_V3_WEIGHT_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    data_dir = Path(
        os.environ.get("VERALLM_DATA_DIR", "~/.verathos")
    ).expanduser()
    return data_dir / "proof_v3_weight_trees"


def _entry_dir(cache_dir: Path, expected_root: bytes) -> Path:
    if not isinstance(expected_root, bytes) or len(expected_root) != 32:
        raise ProofV3Error("static weight cache root must be 32 bytes")
    try:
        mode = cache_dir.lstat().st_mode
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ProofV3Error(
            "static weight cache root is inaccessible"
        ) from exc
    else:
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ProofV3Error(
                "static weight cache root must be a regular directory"
            )
    return cache_dir / expected_root.hex()


def _read_exact_bytearray(path: Path, expected_size: int) -> bytearray:
    if expected_size < 1 or path.stat().st_size != expected_size:
        raise ProofV3Error("static weight cache file size is invalid")
    output = bytearray(expected_size)
    with path.open("rb", buffering=0) as handle:
        view = memoryview(output)
        offset = 0
        while offset < expected_size:
            count = handle.readinto(view[offset:])
            if not count:
                raise ProofV3Error("static weight cache file is truncated")
            offset += count
    return output


def _write_all(handle, data) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        written = handle.write(view[offset:])
        if not written:
            raise OSError("static weight cache write was truncated")
        offset += written


def _require_regular_file(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode)


def _require_regular_directory(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ProofV3Error(
            "static weight cache entry is inaccessible"
        ) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ProofV3Error(
            "static weight cache entry must be a regular directory"
        )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _SparseCachedMerkleTree:
    """Merkle paths backed by leaves plus persisted upper tree levels."""

    __slots__ = (
        "num_leaves",
        "_leaves",
        "_upper",
        "_upper_sizes",
        "_upper_offsets",
        "_split_level",
        "_local_cache",
        "_local_cache_lock",
        "_level_sizes",
        "level_offsets",
    )

    def __init__(
        self,
        *,
        num_leaves: int,
        leaves: bytearray,
        upper: bytearray,
        upper_sizes,
        split_level: int,
    ) -> None:
        if (
            num_leaves < 1
            or len(leaves) != num_leaves * 32
            or split_level < 0
            or split_level > _COMPACT_LOCAL_TREE_HEIGHT
        ):
            raise ProofV3Error("compact static Merkle cache is malformed")
        sizes = tuple(int(value) for value in upper_sizes)
        expected_first = (
            num_leaves + (1 << split_level) - 1
        ) >> split_level
        if (
            not sizes
            or sizes[0] != expected_first
            or sizes[-1] != 1
            or any(value < 1 for value in sizes)
            or any(
                sizes[index + 1] != (sizes[index] + 1) // 2
                for index in range(len(sizes) - 1)
            )
            or len(upper) != sum(sizes) * 32
        ):
            raise ProofV3Error("compact static Merkle cache is malformed")
        offsets = []
        offset = 0
        for size in sizes:
            offsets.append(offset)
            offset += size * 32
        from zkllm.crypto.merkle import hash_node

        for level in range(len(sizes) - 1):
            child_size = sizes[level]
            child_offset = offsets[level]
            parent_offset = offsets[level + 1]
            for parent_index in range(sizes[level + 1]):
                left_index = parent_index * 2
                right_index = min(left_index + 1, child_size - 1)
                left_start = child_offset + left_index * 32
                right_start = child_offset + right_index * 32
                parent_start = parent_offset + parent_index * 32
                if hash_node(
                    bytes(upper[left_start : left_start + 32]),
                    bytes(upper[right_start : right_start + 32]),
                ) != bytes(upper[parent_start : parent_start + 32]):
                    raise ProofV3Error(
                        "compact static Merkle upper tree is inconsistent"
                    )
        self.num_leaves = num_leaves
        self._leaves = leaves
        self._upper = upper
        self._upper_sizes = sizes
        self._upper_offsets = tuple(offsets)
        self._split_level = split_level
        self._local_cache = OrderedDict()
        self._local_cache_lock = threading.RLock()
        sizes_all = [num_leaves]
        while sizes_all[-1] > 1:
            sizes_all.append((sizes_all[-1] + 1) // 2)
        level_offsets = []
        offset = 0
        for size in sizes_all:
            level_offsets.append(offset)
            offset += size
        level_offsets.append(offset)
        self._level_sizes = tuple(sizes_all)
        self.level_offsets = tuple(level_offsets)

    @property
    def root(self) -> bytes:
        return self._upper_hash(len(self._upper_sizes) - 1, 0)

    @property
    def num_levels(self) -> int:
        return self._split_level + len(self._upper_sizes)

    def _leaf_hash(self, index: int) -> bytes:
        if index < 0 or index >= self.num_leaves:
            raise IndexError("Merkle leaf index out of range")
        offset = index * 32
        return bytes(self._leaves[offset : offset + 32])

    def _upper_hash(self, level: int, index: int) -> bytes:
        if (
            level < 0
            or level >= len(self._upper_sizes)
            or index < 0
            or index >= self._upper_sizes[level]
        ):
            raise IndexError("Merkle upper-node index out of range")
        offset = self._upper_offsets[level] + index * 32
        return bytes(self._upper[offset : offset + 32])

    def get_leaf(self, index: int) -> bytes:
        return self._leaf_hash(index)

    def _get_hash(self, flat_index: int) -> bytes:
        if flat_index < 0 or flat_index >= self.level_offsets[-1]:
            raise IndexError("Merkle node index out of range")
        level = 0
        while flat_index >= self.level_offsets[level + 1]:
            level += 1
        index = flat_index - self.level_offsets[level]
        if level >= self._split_level:
            return self._upper_hash(level - self._split_level, index)
        segment_width = 1 << (self._split_level - level)
        segment = index // segment_width
        local_index = index % segment_width
        values = self._local_levels(segment)[level]
        if local_index >= len(values):
            raise IndexError("Merkle local node index out of range")
        return values[local_index]

    def _local_levels(self, segment: int):
        with self._local_cache_lock:
            cached = self._local_cache.get(segment)
            if cached is not None:
                self._local_cache.move_to_end(segment)
                return cached
            from zkllm.crypto.merkle import hash_node

            width = 1 << self._split_level
            start = segment * width
            end = min(start + width, self.num_leaves)
            if start >= end:
                raise IndexError("Merkle segment index out of range")
            levels = [
                tuple(self._leaf_hash(index) for index in range(start, end))
            ]
            for _level in range(self._split_level):
                current = levels[-1]
                levels.append(
                    tuple(
                        hash_node(
                            current[index],
                            (
                                current[index + 1]
                                if index + 1 < len(current)
                                else current[index]
                            ),
                        )
                        for index in range(0, len(current), 2)
                    )
                )
            if (
                len(levels[-1]) != 1
                or levels[-1][0] != self._upper_hash(0, segment)
            ):
                raise ProofV3Error(
                    "compact static Merkle leaves disagree with signed tree"
                )
            result = tuple(levels)
            self._local_cache[segment] = result
            while len(self._local_cache) > _COMPACT_LOCAL_CACHE_SIZE:
                self._local_cache.popitem(last=False)
            return result

    def get_path(self, leaf_index: int):
        from zkllm.types import MerklePath

        if leaf_index < 0 or leaf_index >= self.num_leaves:
            raise IndexError("Merkle leaf index out of range")
        segment = leaf_index >> self._split_level
        local_index = leaf_index - (
            segment << self._split_level
        )
        levels = self._local_levels(segment)
        siblings = []
        index = local_index
        for level in range(self._split_level):
            values = levels[level]
            if index % 2 == 0:
                sibling = index + 1
                is_left = False
            else:
                sibling = index - 1
                is_left = True
            if sibling >= len(values):
                sibling = index
            siblings.append((values[sibling], is_left))
            index //= 2

        index = segment
        for level in range(len(self._upper_sizes) - 1):
            size = self._upper_sizes[level]
            if index % 2 == 0:
                sibling = index + 1
                is_left = False
            else:
                sibling = index - 1
                is_left = True
            if sibling >= size:
                sibling = index
            siblings.append((self._upper_hash(level, sibling), is_left))
            index //= 2
        return MerklePath(leaf_index=leaf_index, siblings=siblings)


def load_compact_static_weight_tree_v3(
    *,
    cache_dir,
    expected_root: bytes,
    out_dim: int,
    in_dim: int,
    chunk_size: int,
    int8_values,
):
    """Load sparse path material for runtime-derived canonical int8 rows.

    The compact serving cache never stores model values.  It persists only
    leaf hashes plus the tiny upper tree. Small lower subtrees are rebuilt
    lazily for opened paths, while canonical rows come from the loaded model.
    """

    import torch
    from zkllm.crypto.merkle import FlatWeightMerkle

    values = torch.as_tensor(int8_values)
    if (
        values.dtype != torch.int8
        or values.dim() != 2
        or tuple(int(item) for item in values.shape)
        != (int(out_dim), int(in_dim))
    ):
        raise ProofV3Error(
            "compact static weight cache rows are malformed"
        )
    directory = _entry_dir(Path(cache_dir), expected_root)
    try:
        directory.lstat()
    except FileNotFoundError:
        return None
    _require_regular_directory(directory)
    metadata_path = directory / "compact_metadata.json"
    leaves_path = directory / "leaf_hashes.bin"
    upper_path = directory / "upper_hashes.bin"
    if not _require_regular_file(metadata_path):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        total_elements = int(out_dim) * int(in_dim)
        num_chunks = (
            total_elements + int(chunk_size) - 1
        ) // int(chunk_size)
        level_sizes = [num_chunks]
        while level_sizes[-1] > 1:
            level_sizes.append((level_sizes[-1] + 1) // 2)
        expected_split = min(
            _COMPACT_LOCAL_TREE_HEIGHT,
            len(level_sizes) - 1,
        )
        expected_upper_sizes = tuple(level_sizes[expected_split:])
        expected_upper_bytes = sum(expected_upper_sizes) * 32
        expected = {
            "abi": COMPACT_STATIC_WEIGHT_CACHE_ABI_V3,
            "root": expected_root.hex(),
            "out_dim": int(out_dim),
            "in_dim": int(in_dim),
            "chunk_size": int(chunk_size),
            "total_elements": total_elements,
            "num_chunks": num_chunks,
            "bytes_per_element": 1,
            "dtype_code": "<i1",
            "leaf_hash_bytes": num_chunks * 32,
            "split_level": expected_split,
            "upper_hash_bytes": expected_upper_bytes,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            return None
        if tuple(metadata.get("upper_level_sizes", ())) != (
            expected_upper_sizes
        ):
            return None
        if (
            not _require_regular_file(leaves_path)
            or leaves_path.stat().st_size != expected["leaf_hash_bytes"]
            or not _require_regular_file(upper_path)
        ):
            return None
        packed_leaves = _read_exact_bytearray(
            leaves_path,
            expected["leaf_hash_bytes"],
        )
        if upper_path.stat().st_size != expected_upper_bytes:
            return None
        packed_upper = _read_exact_bytearray(
            upper_path,
            expected_upper_bytes,
        )
        if (
            hashlib.sha256(packed_leaves).hexdigest()
            != metadata.get("leaf_hash_sha256")
            or hashlib.sha256(packed_upper).hexdigest()
            != metadata.get("upper_hash_sha256")
        ):
            return None
        sparse = _SparseCachedMerkleTree(
            num_leaves=num_chunks,
            leaves=packed_leaves,
            upper=packed_upper,
            upper_sizes=expected_upper_sizes,
            split_level=expected_split,
        )
        tree = FlatWeightMerkle.__new__(FlatWeightMerkle)
        tree.num_rows = int(out_dim)
        tree.num_cols = int(in_dim)
        tree.chunk_size = int(chunk_size)
        tree.total_elements = total_elements
        tree.num_chunks = num_chunks
        tree._bytes_per_element = 1
        tree._dtype_code = "<i1"
        tree._bytes_per_chunk = int(chunk_size)
        tree._raw_bytes = None
        tree._tree = sparse
        if tree.root != expected_root:
            return None
        return tree
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        ProofV3Error,
        json.JSONDecodeError,
    ):
        return None


def save_compact_static_weight_tree_v3(
    *,
    cache_dir,
    expected_root: bytes,
    tree,
    int8_values,
) -> Path:
    """Atomically persist sparse canonical Merkle authentication material.

    Runtime values remain sourced from the loaded model. Leaves plus upper
    levels avoid duplicating the model surrogate or full internal tree while
    bounding each lazy path reconstruction to one small local subtree.
    """

    import torch

    values = torch.as_tensor(int8_values)
    if values.dtype != torch.int8 or values.dim() != 2:
        raise ProofV3Error("compact static cache values must be 2-D int8")
    if tree.root != expected_root:
        raise ProofV3Error(
            "compact static cache tree root is not signed root"
        )
    if (
        int(tree.num_rows) != int(values.shape[0])
        or int(tree.num_cols) != int(values.shape[1])
        or int(tree._bytes_per_element) != 1
        or str(tree._dtype_code) != "<i1"
    ):
        raise ProofV3Error(
            "compact static cache tree dimensions mismatch"
        )

    num_chunks = int(tree.num_chunks)
    leaf_hash_bytes = num_chunks * 32
    try:
        level_offsets = tuple(int(value) for value in tree._tree.level_offsets)
        full_tree = memoryview(tree._tree._data)
    except AttributeError as exc:
        raise ProofV3Error(
            "compact static cache requires a complete source tree"
        ) from exc
    packed_leaves = full_tree[:leaf_hash_bytes]
    if len(packed_leaves) != leaf_hash_bytes:
        raise ProofV3Error("compact static cache leaf tree is truncated")
    total_levels = len(level_offsets) - 1
    split_level = min(
        _COMPACT_LOCAL_TREE_HEIGHT,
        total_levels - 1,
    )
    upper_start = level_offsets[split_level] * 32
    upper_end = level_offsets[-1] * 32
    packed_upper = full_tree[upper_start:upper_end]
    upper_level_sizes = tuple(
        level_offsets[index + 1] - level_offsets[index]
        for index in range(split_level, total_levels)
    )
    if (
        not upper_level_sizes
        or upper_level_sizes[-1] != 1
        or len(packed_upper) != sum(upper_level_sizes) * 32
    ):
        raise ProofV3Error("compact static cache upper tree is malformed")

    directory = _entry_dir(Path(cache_dir), expected_root)
    directory.mkdir(parents=True, exist_ok=True)
    _require_regular_directory(Path(cache_dir))
    _require_regular_directory(directory)
    leaves_path = directory / "leaf_hashes.bin"
    upper_path = directory / "upper_hashes.bin"
    metadata_path = directory / "compact_metadata.json"
    temporary_paths: list[Path] = []
    try:
        def temporary_path(suffix: str) -> Path:
            fd, raw_path = tempfile.mkstemp(
                dir=directory,
                prefix=".building-compact-",
                suffix=suffix,
            )
            os.close(fd)
            path = Path(raw_path)
            temporary_paths.append(path)
            return path

        leaves_tmp = temporary_path(".leaves")
        with leaves_tmp.open("wb", buffering=0) as handle:
            _write_all(handle, packed_leaves)
            os.fsync(handle.fileno())

        upper_tmp = temporary_path(".upper")
        with upper_tmp.open("wb", buffering=0) as handle:
            _write_all(handle, packed_upper)
            os.fsync(handle.fileno())

        metadata = {
            "abi": COMPACT_STATIC_WEIGHT_CACHE_ABI_V3,
            "root": expected_root.hex(),
            "out_dim": int(values.shape[0]),
            "in_dim": int(values.shape[1]),
            "chunk_size": int(tree.chunk_size),
            "total_elements": int(tree.total_elements),
            "num_chunks": num_chunks,
            "bytes_per_element": 1,
            "dtype_code": "<i1",
            "leaf_hash_bytes": leaf_hash_bytes,
            "leaf_hash_sha256": hashlib.sha256(packed_leaves).hexdigest(),
            "split_level": split_level,
            "upper_level_sizes": upper_level_sizes,
            "upper_hash_bytes": len(packed_upper),
            "upper_hash_sha256": hashlib.sha256(packed_upper).hexdigest(),
        }
        metadata_tmp = temporary_path(".json")
        metadata_tmp.write_text(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        with metadata_tmp.open("rb") as handle:
            os.fsync(handle.fileno())

        os.replace(leaves_tmp, leaves_path)
        os.replace(upper_tmp, upper_path)
        os.replace(metadata_tmp, metadata_path)
        _fsync_directory(directory)
        return directory
    finally:
        for path in temporary_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def discard_static_weight_material_v3(
    *,
    cache_dir,
    expected_root: bytes,
    include_compact: bool,
) -> int:
    """Delete one authenticated-root-keyed derived cache entry.

    Only known derived files under the exact root directory are removed.
    Symlinks and metadata claiming a different root fail closed.
    """

    directory = _entry_dir(Path(cache_dir), expected_root)
    if not directory.exists():
        return 0
    if directory.is_symlink() or not directory.is_dir():
        raise ProofV3Error(
            "static weight cache entry is not a regular directory"
        )
    for metadata_name in ("metadata.json", "compact_metadata.json"):
        metadata_path = directory / metadata_name
        if not metadata_path.exists():
            continue
        if not _require_regular_file(metadata_path):
            raise ProofV3Error(
                "static weight cache metadata is not a regular file"
            )
        try:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProofV3Error(
                "static weight cache metadata is malformed"
            ) from exc
        if metadata.get("root") != expected_root.hex():
            raise ProofV3Error(
                "static weight cache metadata root is inconsistent"
            )

    names = ["metadata.json", "tree.bin", "values.i8"]
    if include_compact:
        names.extend(
            (
                "compact_metadata.json",
                "leaf_hashes.bin",
                "upper_hashes.bin",
            )
        )
    removed = 0
    # Remove completion markers first. A crash can leave only inert derived
    # payloads, never a cache entry that appears complete.
    ordered = [
        name for name in names if name.endswith("metadata.json")
    ] + [
        name for name in names if not name.endswith("metadata.json")
    ]
    for name in ordered:
        path = directory / name
        if not path.exists():
            continue
        if not _require_regular_file(path):
            raise ProofV3Error(
                "static weight cache payload is not a regular file"
            )
        removed += path.stat().st_size
        path.unlink()
    for path in directory.glob(".building-*"):
        if _require_regular_file(path):
            removed += path.stat().st_size
            path.unlink()
    try:
        directory.rmdir()
    except OSError:
        pass
    return removed


def load_static_weight_material_v3(
    *,
    cache_dir,
    expected_root: bytes,
    out_dim: int,
    in_dim: int,
    chunk_size: int,
):
    """Return ``(tree, int8[out,in])`` on an exact cache hit, else ``None``."""

    import torch
    from zkllm.crypto.merkle import FlatWeightMerkle

    cache_dir = Path(cache_dir)
    directory = _entry_dir(cache_dir, expected_root)
    try:
        directory.lstat()
    except FileNotFoundError:
        return None
    _require_regular_directory(directory)
    metadata_path = directory / "metadata.json"
    tree_path = directory / "tree.bin"
    values_path = directory / "values.i8"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "abi": STATIC_WEIGHT_CACHE_ABI_V3,
            "root": expected_root.hex(),
            "out_dim": int(out_dim),
            "in_dim": int(in_dim),
            "chunk_size": int(chunk_size),
            "dtype_code": "<i1",
            "bytes_per_element": 1,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            return None
        total_elements = int(out_dim) * int(in_dim)
        if (
            int(metadata.get("total_elements", -1)) != total_elements
            or int(metadata.get("values_bytes", -1)) != total_elements
            or not tree_path.is_file()
            or not values_path.is_file()
            or values_path.stat().st_size != total_elements
        ):
            return None
        tree_size = int(metadata.get("tree_bytes", -1))
        tree_data = _read_exact_bytearray(tree_path, tree_size)
        cache_data = {
            "num_rows": int(out_dim),
            "num_cols": int(in_dim),
            "chunk_size": int(chunk_size),
            "total_elements": total_elements,
            "num_chunks": int(metadata["num_chunks"]),
            "bytes_per_element": 1,
            "dtype_code": "<i1",
            "bytes_per_chunk": int(chunk_size),
            "tree_data": tree_data,
            "tree_num_leaves": int(metadata["tree_num_leaves"]),
            "tree_level_offsets": [
                int(value) for value in metadata["tree_level_offsets"]
            ],
        }
        tree = FlatWeightMerkle.from_cached(cache_data)
        if tree.root != expected_root:
            return None
        values = torch.from_file(
            str(values_path),
            shared=False,
            size=total_elements,
            dtype=torch.int8,
        ).reshape(int(out_dim), int(in_dim))
        return tree, values
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def save_static_weight_material_v3(
    *,
    cache_dir,
    expected_root: bytes,
    tree,
    int8_values,
) -> Path:
    """Atomically publish one canonical int8 surrogate and compact tree."""

    import torch

    values = int8_values.detach()
    if values.dtype != torch.int8 or values.dim() != 2:
        raise ProofV3Error("static weight cache values must be 2-D int8")
    values = values.to(device="cpu").contiguous()
    if tree.root != expected_root:
        raise ProofV3Error("static weight cache tree root is not signed root")
    if (
        int(tree.num_rows) != int(values.shape[0])
        or int(tree.num_cols) != int(values.shape[1])
    ):
        raise ProofV3Error("static weight cache tree dimensions mismatch")

    directory = _entry_dir(Path(cache_dir), expected_root)
    directory.mkdir(parents=True, exist_ok=True)
    _require_regular_directory(Path(cache_dir))
    _require_regular_directory(directory)
    tree_path = directory / "tree.bin"
    values_path = directory / "values.i8"
    metadata_path = directory / "metadata.json"
    temporary_paths: list[Path] = []
    try:
        def temporary_path(suffix: str) -> Path:
            fd, raw_path = tempfile.mkstemp(
                dir=directory, prefix=".building-", suffix=suffix
            )
            os.close(fd)
            path = Path(raw_path)
            temporary_paths.append(path)
            return path

        tree_tmp = temporary_path(".tree")
        with tree_tmp.open("wb", buffering=0) as handle:
            handle.write(tree._tree._data)
            os.fsync(handle.fileno())

        values_tmp = temporary_path(".i8")
        with values_tmp.open("wb", buffering=0) as handle:
            values.numpy().tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())

        cache_data = tree.get_cache_data()
        metadata = {
            "abi": STATIC_WEIGHT_CACHE_ABI_V3,
            "root": expected_root.hex(),
            "out_dim": int(values.shape[0]),
            "in_dim": int(values.shape[1]),
            "chunk_size": int(tree.chunk_size),
            "total_elements": int(tree.total_elements),
            "num_chunks": int(tree.num_chunks),
            "bytes_per_element": int(tree._bytes_per_element),
            "dtype_code": str(tree._dtype_code),
            "tree_num_leaves": int(cache_data["tree_num_leaves"]),
            "tree_level_offsets": [
                int(value) for value in cache_data["tree_level_offsets"]
            ],
            "tree_bytes": len(tree._tree._data),
            "values_bytes": int(values.numel()),
        }
        metadata_tmp = temporary_path(".json")
        metadata_tmp.write_text(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        with metadata_tmp.open("rb") as handle:
            os.fsync(handle.fileno())

        os.replace(tree_tmp, tree_path)
        os.replace(values_tmp, values_path)
        # Metadata is the completion marker and is always published last.
        os.replace(metadata_tmp, metadata_path)
        _fsync_directory(directory)
        return directory
    finally:
        for path in temporary_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
