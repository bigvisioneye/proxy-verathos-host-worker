"""ctypes loader for the compiled Merkle multi-opening hot loop.

Builds c_multiopen.c with the system compiler on first use (cached in
the user cache dir); byte-identical to the Python reconstruction.
Falls back silently when no compiler is available.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile

_LIB = None


def load() -> object | None:
    global _LIB
    if _LIB is not None:
        return _LIB or None
    src = os.path.join(os.path.dirname(__file__), "c_multiopen.c")
    try:
        with open(src, "rb") as fh:
            tag = hashlib.sha256(fh.read()).hexdigest()[:16]
        cache_dir = os.path.join(
            tempfile.gettempdir(), f"verathos_c_multiopen_{os.getuid()}")
        os.makedirs(cache_dir, exist_ok=True)
        so_path = os.path.join(cache_dir, f"c_multiopen_{tag}.so")
        if not os.path.exists(so_path):
            accelerated = subprocess.run(
                [
                    "cc",
                    "-O3",
                    "-shared",
                    "-fPIC",
                    "-DVERATHOS_OPENSSL_SHA256=1",
                    "-o",
                    so_path,
                    src,
                    "-lcrypto",
                ],
                check=False,
                capture_output=True,
            )
            if accelerated.returncode != 0:
                subprocess.run(
                    ["cc", "-O3", "-shared", "-fPIC", "-o", so_path, src],
                    check=True,
                    capture_output=True,
                )
        lib = ctypes.CDLL(so_path)
        lib.reconstruct_multiopen_w1.restype = ctypes.c_int
        lib.reconstruct_multiopen_wn.restype = ctypes.c_int
        lib.sibling_coordinates.restype = ctypes.c_int
        lib.verify_opening_full.restype = ctypes.c_int
        lib.replay_rounds4.restype = ctypes.c_int
        lib.rebuild_root_w1.restype = ctypes.c_int
        lib.rebuild_root_w1.argtypes = [
            ctypes.c_char_p, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64,
            ctypes.c_char_p, ctypes.c_char_p,
        ]
        lib.reconstruct_multiopen_w1.argtypes = [
            ctypes.c_char_p, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64), ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        lib.reconstruct_multiopen_wn.argtypes = [
            ctypes.c_char_p, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64), ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        _LIB = lib
        return lib
    except Exception:
        _LIB = False
        return None


def reconstruct_wn(leaf_prefix: bytes, node_prefix: bytes, leaf_count: int,
                   indices, values, leaf_width: int, sib_levels, sib_indices,
                   sib_digests: bytes):
    """Width-N multiproof root: raw bytes or None (fallback to Python).

    ``values`` is FLAT row-major (``leaf_width`` per opened leaf)."""

    lib = load()
    if lib is None:
        return None
    import array

    width = int(leaf_width)
    n_leaves = len(indices)
    if len(values) != n_leaves * width:
        return None
    n_sib = len(sib_levels)
    scratch = n_leaves + n_sib + 4

    def _i64(seq):
        dtype = getattr(seq, "dtype", None)
        if (
            dtype is not None
            and dtype.kind == "i"
            and dtype.itemsize == 8
        ):
            import numpy as np

            contiguous = np.ascontiguousarray(seq)
            return (ctypes.c_int64 * max(1, len(contiguous))).from_buffer(
                contiguous
            )
        return (ctypes.c_int64 * max(1, len(seq))).from_buffer(
            array.array("q", seq or [0])
        )

    def _u64(seq):
        dtype = getattr(seq, "dtype", None)
        if (
            dtype is not None
            and dtype.kind == "u"
            and dtype.itemsize == 8
        ):
            import numpy as np

            contiguous = np.ascontiguousarray(seq)
            return (ctypes.c_uint64 * max(1, len(contiguous))).from_buffer(
                contiguous
            )
        return (ctypes.c_uint64 * max(1, len(seq))).from_buffer(
            array.array("Q", seq or [0])
        )

    idx_arr = _i64(indices)
    val_arr = _u64(values)
    lev_arr = _i64(sib_levels)
    sidx_arr = _i64(sib_indices)
    work_idx = (ctypes.c_int64 * scratch)()
    work_dig = ctypes.create_string_buffer(32 * scratch)
    root = ctypes.create_string_buffer(32)
    rc = lib.reconstruct_multiopen_wn(
        leaf_prefix, len(leaf_prefix), node_prefix, len(node_prefix),
        leaf_count, idx_arr, val_arr, n_leaves, width,
        lev_arr, sidx_arr, sib_digests, n_sib,
        work_idx, work_dig, root)
    if rc != 0:
        return rc  # error code -> caller raises like the Python path
    return root.raw


def reconstruct_w1(leaf_prefix: bytes, node_prefix: bytes, leaf_count: int,
                   indices, values, sib_levels, sib_indices,
                   sib_digests: bytes):
    """Returns the raw root bytes or None (fallback to Python)."""

    lib = load()
    if lib is None:
        return None
    import array

    n_leaves = len(indices)
    n_sib = len(sib_levels)
    scratch = n_leaves + n_sib + 4
    # array.array packs at C speed; from_buffer avoids per-element ctypes.
    # numpy arrays go through the buffer protocol ZERO-COPY (dtype-checked),
    # skipping the per-element python iteration entirely.
    def _i64_buf(seq):
        dt = getattr(seq, "dtype", None)
        if dt is not None and dt.kind == "i" and dt.itemsize == 8:
            import numpy as _np

            return (ctypes.c_int64 * len(seq)).from_buffer(
                _np.ascontiguousarray(seq))
        return (ctypes.c_int64 * len(seq)).from_buffer(array.array("q", seq))

    def _u64_buf(seq):
        dt = getattr(seq, "dtype", None)
        if dt is not None and dt.kind == "u" and dt.itemsize == 8:
            import numpy as _np

            return (ctypes.c_uint64 * len(seq)).from_buffer(
                _np.ascontiguousarray(seq))
        return (ctypes.c_uint64 * len(seq)).from_buffer(array.array("Q", seq))

    idx_arr = _i64_buf(indices)
    val_arr = _u64_buf(values)
    lev_arr = (ctypes.c_int64 * max(1, n_sib)).from_buffer(
        array.array("q", sib_levels or [0]))
    sidx_arr = (ctypes.c_int64 * max(1, n_sib)).from_buffer(
        array.array("q", sib_indices or [0]))
    work_idx = (ctypes.c_int64 * scratch)()
    work_dig = ctypes.create_string_buffer(32 * scratch)
    root = ctypes.create_string_buffer(32)
    rc = lib.reconstruct_multiopen_w1(
        leaf_prefix, len(leaf_prefix), node_prefix, len(node_prefix),
        leaf_count, idx_arr, val_arr, n_leaves,
        lev_arr, sidx_arr, sib_digests, n_sib,
        work_idx, work_dig, root)
    if rc != 0:
        return rc  # error code -> caller raises like the Python path
    return root.raw


def sibling_coordinates(leaf_count: int, indices):
    """C-computed (level, index) schedule or None (Python fallback)."""

    lib = load()
    if lib is None:
        return None
    import array

    k = len(indices)
    # array.array packs the index vector at C speed; from_buffer avoids the
    # per-element ctypes splat (matches reconstruct_w1). idx_buf must outlive
    # the call since idx_arr aliases its storage.
    idx_buf = array.array("q", indices)
    idx_arr = (ctypes.c_int64 * k).from_buffer(idx_buf)
    cap = k * 64 + 8
    lv = (ctypes.c_int64 * cap)()
    ix = (ctypes.c_int64 * cap)()
    sc = (ctypes.c_int64 * (k + 4))()
    count = lib.sibling_coordinates(
        ctypes.c_int64(leaf_count), idx_arr, k, lv, ix, sc)
    if count < 0 or count > cap:
        return None
    # ctypes-array slicing returns plain Python lists at C speed; zip pairs
    # them without a per-element genexpr.
    return tuple(zip(lv[:count], ix[:count]))


def verify_opening_full(*, transcript_seed: bytes, claimed: int,
                        final_value: int, point, rounds, roots: bytes,
                        dom: bytes, qdom: bytes, base_size: int,
                        layer_shift, layer_gen, n_queries: int,
                        layers):
    """Full compiled opening verification. layers: per-layer
    (indices, values, sib_levels, sib_indices, sib_digests,
    leaf_prefix, node_prefix, root_prefix). Returns 0 or error code,
    None when the library is unavailable."""

    lib = load()
    if lib is None:
        return None
    import array

    n = len(rounds)
    lay_off = [0]
    sib_off = [0]
    all_idx: list[int] = []
    all_val: list[int] = []
    all_lv: list[int] = []
    all_ix: list[int] = []
    dig_parts = []
    leaf_parts = []
    node_parts = []
    root_parts = []
    for (indices, values, slv, six, sdig, lp, np_, rp) in layers:
        all_idx.extend(indices)
        all_val.extend(values)
        all_lv.extend(slv)
        all_ix.extend(six)
        dig_parts.append(sdig)
        leaf_parts.append(lp)
        node_parts.append(np_)
        root_parts.append(rp)
        lay_off.append(len(all_idx))
        sib_off.append(len(all_lv))
    flat_rounds = array.array("Q", [v for row in rounds for v in row])
    scratch_n = len(all_idx) + len(all_lv) + 16
    return lib.verify_opening_full(
        transcript_seed,
        ctypes.c_uint64(claimed), ctypes.c_uint64(final_value),
        n,
        (ctypes.c_uint64 * n).from_buffer(array.array("Q", point)),
        (ctypes.c_uint64 * (3 * n)).from_buffer(flat_rounds),
        roots, dom, len(dom), qdom, len(qdom),
        ctypes.c_int64(base_size),
        (ctypes.c_uint64 * n).from_buffer(array.array("Q", layer_shift)),
        (ctypes.c_uint64 * n).from_buffer(array.array("Q", layer_gen)),
        n_queries,
        (ctypes.c_int64 * (n + 2)).from_buffer(array.array("q", lay_off)),
        (ctypes.c_int64 * max(1, len(all_idx))).from_buffer(
            array.array("q", all_idx or [0])),
        (ctypes.c_uint64 * max(1, len(all_val))).from_buffer(
            array.array("Q", all_val or [0])),
        (ctypes.c_int64 * (n + 2)).from_buffer(array.array("q", sib_off)),
        (ctypes.c_int64 * max(1, len(all_lv))).from_buffer(
            array.array("q", all_lv or [0])),
        (ctypes.c_int64 * max(1, len(all_ix))).from_buffer(
            array.array("q", all_ix or [0])),
        b"".join(dig_parts) or b"\x00" * 32,
        b"".join(leaf_parts), b"".join(node_parts), b"".join(root_parts),
        (ctypes.c_int64 * scratch_n)(),
        ctypes.create_string_buffer(32 * scratch_n),
        (ctypes.c_int64 * 12288)())


def replay_rounds4(transcript_seed: bytes, claimed: int, rounds,
                   dom: bytes, label: bytes, absorb_tag: bool,
                   index_base: int):
    """C sumcheck replay; (challenges, running, transcript) or None."""

    lib = load()
    if lib is None:
        return None
    import array

    n = len(rounds)
    flat = array.array("Q", [v % (1 << 64) for row in rounds for v in row])
    chal = (ctypes.c_uint64 * max(1, n))()
    running = ctypes.c_uint64(0)
    t_out = ctypes.create_string_buffer(32)
    rc = lib.replay_rounds4(
        transcript_seed, ctypes.c_uint64(claimed),
        (ctypes.c_uint64 * (4 * n)).from_buffer(flat), n,
        dom, len(dom), label, len(label),
        1 if absorb_tag else 0, index_base,
        chal, ctypes.byref(running), t_out)
    if rc != 0:
        return rc
    return (tuple(chal[i] for i in range(n)), running.value, t_out.raw)


def rebuild_root_w1(leaf_prefix: bytes, node_prefix: bytes, values):
    """C batch root rebuild; returns the RAW root or None if no lib."""

    import numpy as np

    lib = load()
    if lib is None:
        return None
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.uint64))
    n = arr.shape[0]
    scratch = ctypes.create_string_buffer(int(n) * 32)
    out = ctypes.create_string_buffer(32)
    rc = lib.rebuild_root_w1(
        leaf_prefix, len(leaf_prefix), node_prefix, len(node_prefix),
        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        ctypes.c_int64(n), scratch, out)
    if rc != 0:
        return None
    return out.raw
