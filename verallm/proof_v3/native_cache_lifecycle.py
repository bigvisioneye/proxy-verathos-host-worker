"""Bound the process-lifetime CUDA footprint of post-nonce proof caches."""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error

__all__ = [
    "ReleasedProofCudaCachesV3",
    "release_transient_proof_cuda_caches_v3",
]


_TRANSIENT_CACHE_TARGETS = (
    (
        "verallm.proof_v3.native_cuda_fold_backend",
        (
            "_NTT_STAGE_CACHE",
            "_NTT_COSET_CACHE",
            "_FOUR_STEP_HOST_MID",
        ),
    ),
    (
        "verallm.proof_v3.native_pcs_backend",
        (
            "_BITREV_CACHE",
            "_U8_TENSOR_CACHE",
            "_EQFOLD_GRAPHS",
            "_OPEN_GRAPHS",
            "_LOGUP_GRAPHS",
            "_FS_ONES_CACHE",
            "_DEVICE_TABLE_CACHE",
        ),
    ),
    (
        "verallm.proof_v3.goldilocks_succinct_zero_check_toolkit",
        ("_BATCH_FOLD_GRAPHS",),
    ),
)


@dataclass(frozen=True, slots=True)
class ReleasedProofCudaCachesV3:
    """Measured allocator state around one transient proof-cache release."""

    entries: int
    allocated_before: int
    allocated_after: int
    reserved_before: int
    reserved_after: int

    @property
    def allocated_released(self) -> int:
        return max(0, self.allocated_before - self.allocated_after)

    @property
    def reserved_released(self) -> int:
        return max(0, self.reserved_before - self.reserved_after)


def _cuda_allocator_state() -> tuple[int, int]:
    import torch

    if not torch.cuda.is_available():
        return 0, 0
    return (
        int(torch.cuda.memory_allocated()),
        int(torch.cuda.memory_reserved()),
    )


def release_transient_proof_cuda_caches_v3() -> ReleasedProofCudaCachesV3:
    """Release proof-only graphs/tables before ordinary serving resumes.

    Hard proofs execute behind the exclusive admission gate. Their CUDA graph,
    NTT and table caches speed repeated proof calls, but retaining them after
    the gate reopens steals the headroom vLLM needs for ordinary inference.
    Only modules already imported by the completed proof are inspected; this
    function never initializes a proof backend merely to clean it.
    """

    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    allocated_before, reserved_before = _cuda_allocator_state()
    entries = 0
    for module_name, cache_names in _TRANSIENT_CACHE_TARGETS:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for cache_name in cache_names:
            cache = getattr(module, cache_name, None)
            if cache is None:
                continue
            if not isinstance(cache, dict):
                raise ProofV3Error(
                    "proof-v3 transient CUDA cache has an unexpected type"
                )
            entries += len(cache)
            cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    allocated_after, reserved_after = _cuda_allocator_state()
    return ReleasedProofCudaCachesV3(
        entries=entries,
        allocated_before=allocated_before,
        allocated_after=allocated_after,
        reserved_before=reserved_before,
        reserved_after=reserved_after,
    )
