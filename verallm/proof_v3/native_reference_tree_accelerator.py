"""Fused-GPU acceleration behind the reference tree/sumcheck APIs.

The reference tile proofs (LogUp, product sumchecks, softmax/attention/
MLP/RMSNorm tiles) spend nearly all their time building width-1 SHA-256
Merkle trees in Python (multiplicity/witness tables, often 2^15..2^17
leaves, rebuilt on BOTH prove and verify). The fused CUDA hasher
(gl_sha256_tree.cu) is byte-exact for exactly this shape.

``install_fused_reference_acceleration()`` monkeypatches, at runtime:

* ``GoldilocksMerkleTreeReference.from_rows``: width-1 power-of-two
  tables at or above a size threshold are committed on the GPU and
  returned as a duck-typed shim exposing the attributes the tile path
  consumes (``commitment``, ``rows``, shape and binding fields). Any
  consumer that needs authentication paths (``open``/``levels`` -- the
  PCS/FRI modules) transparently falls back: the shim materialises the
  full Python reference tree on first access and delegates, so
  correctness is preserved everywhere and only tile-path trees stay on
  the fast path.
* the ``prove_goldilocks_product_sumcheck_v3`` reference entrypoint in
  its CONSUMER modules (attention head, MLP tile): swapped for the
  byte-identical fused prover.

Roots and transcripts are byte-identical by construction (the fused
kernels are conformance-tested against the Python reference), so proofs
produced with acceleration verify against unaccelerated verifiers and
vice versa.
"""

from __future__ import annotations

import logging
from typing import Final

from verallm.proof_v3.errors import ProofV3Error

logger = logging.getLogger(__name__)

GOLDILOCKS_FUSED_TREE_MIN_LEAVES_V3: Final = 2048

_STATE: dict = {"installed": False, "tree_ext": None, "fold_ext": None}


class _FusedTreeShim:
    """Duck-type of GoldilocksMerkleTreeReference with a GPU-built root.

    Tile-path consumers read ``commitment`` and ``rows`` only. ``open``
    and ``levels`` (PCS/FRI) lazily materialise the full Python
    reference tree and delegate, preserving correctness at reference
    speed for those callers.
    """

    __slots__ = (
        "binding_digest",
        "leaf_count",
        "leaf_width",
        "commitment",
        "rows",
        "_materialized",
    )

    def __init__(self, *, binding_digest, leaf_count, commitment, rows):
        self.binding_digest = binding_digest
        self.leaf_count = leaf_count
        self.leaf_width = 1
        self.commitment = commitment
        self.rows = rows
        self._materialized = None

    def _reference(self):
        if self._materialized is None:
            from verallm.proof_v3.goldilocks_merkle_reference import (
                GoldilocksMerkleTreeReference,
            )

            self._materialized = _STATE["original_from_rows"](
                GoldilocksMerkleTreeReference,
                self.rows,
                binding_digest=self.binding_digest,
            )
            if self._materialized.commitment != self.commitment:
                raise ProofV3Error(
                    "fused tree root does not match the reference rebuild"
                )
        return self._materialized

    def open(self, indices):
        return self._reference().open(indices)

    @property
    def levels(self):
        return self._reference().levels


def _fused_from_rows(cls, rows, *, binding_digest):
    tree_ext = _STATE["tree_ext"]
    original = _STATE["original_from_rows"]
    try:
        rows_tuple = tuple(
            row if isinstance(row, tuple) else tuple(row) for row in rows
        )
    except TypeError:
        return original(cls, rows, binding_digest=binding_digest)
    n = len(rows_tuple)
    if (
        tree_ext is None
        or n < GOLDILOCKS_FUSED_TREE_MIN_LEAVES_V3
        or n & (n - 1)
        or not rows_tuple
        or any(len(row) != 1 for row in rows_tuple)
        or not isinstance(binding_digest, bytes)
        or len(binding_digest) != 32
    ):
        return original(cls, rows, binding_digest=binding_digest)
    import numpy
    import torch

    from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
    from verallm.proof_v3.native_cuda_tree_backend import fused_merkle_root_w1

    values = numpy.empty(n, dtype=numpy.uint64)
    for index, row in enumerate(rows_tuple):
        value = row[0]
        if not isinstance(value, int) or not 0 <= value < GOLDILOCKS_MODULUS:
            return original(cls, rows, binding_digest=binding_digest)
        values[index] = value
    device_values = torch.from_numpy(values.view(numpy.int64)).cuda()
    commitment = fused_merkle_root_w1(
        tree_ext, device_values, binding_digest=binding_digest
    )
    return _FusedTreeShim(
        binding_digest=binding_digest,
        leaf_count=n,
        commitment=commitment,
        rows=rows_tuple,
    )


def install_fused_reference_acceleration() -> bool:
    """Install the fused tree + product-sumcheck fast paths. Idempotent.

    Returns True when the CUDA tiers loaded and the patches are active;
    False (with everything left untouched) when no CUDA device or the
    kernels fail to build.
    """

    if _STATE["installed"]:
        return True
    try:
        from verallm.proof_v3.native_cuda_tree_backend import load_tree_kernels
        from verallm.proof_v3.native_cuda_fold_backend import load_fused_kernels

        _STATE["tree_ext"] = load_tree_kernels()
        _STATE["fold_ext"] = load_fused_kernels()
    except (ProofV3Error, RuntimeError, OSError) as error:
        logger.info("fused reference acceleration unavailable: %s", error)
        return False

    from verallm.proof_v3.goldilocks_merkle_reference import (
        GoldilocksMerkleTreeReference,
    )

    _STATE["original_from_rows"] = GoldilocksMerkleTreeReference.from_rows.__func__
    GoldilocksMerkleTreeReference.from_rows = classmethod(_fused_from_rows)

    # Product sumcheck: patch the CONSUMER modules (they bound the
    # reference symbol at import time).
    from verallm.proof_v3.native_cuda_fold_backend import (
        fused_prove_product_sumcheck_v3,
    )
    import verallm.proof_v3.goldilocks_attention_head_reference as attention_mod
    import verallm.proof_v3.goldilocks_mlp_tile_reference as mlp_mod

    def _fused_product_prove(**kwargs):
        return fused_prove_product_sumcheck_v3(
            extension=_STATE["fold_ext"], **kwargs
        )

    attention_mod.prove_goldilocks_product_sumcheck_v3 = _fused_product_prove
    mlp_mod.prove_goldilocks_product_sumcheck_v3 = _fused_product_prove

    _STATE["installed"] = True
    logger.info("fused reference acceleration installed (trees + product sumcheck)")
    return True
