"""
Version detection for vLLM CUDA graph support.

Checks whether the installed vLLM supports:
- CustomOp.register_oot() for out-of-tree layer replacement
- CompilationConfig.splitting_ops for piecewise CUDA graphs
"""

import functools
import inspect
import logging

logger = logging.getLogger(__name__)


_PATCHED_VLLM_MIXED_BATCH_RNG = False


def patch_vllm_mixed_batch_rng() -> bool:
    """Keep mixed greedy/random batches off PyTorch's global CUDA RNG.

    vLLM's native sampler first draws a random candidate for every row and
    later replaces greedy rows with their argmax result.  When only the random
    rows have per-request generators, upstream fills the complete candidate
    tensor through the process-global generator before overwriting the seeded
    rows.  A failed CUDA-graph capture can leave that global generator in a
    captured state, making an otherwise valid later mixed batch fail with
    ``Offset increment outside graph capture encountered unexpectedly``.

    Verathos gives every stochastic request a private seed.  Therefore, in a
    partially seeded batch the unseeded rows are greedy and their random
    candidates are discarded by vLLM's final ``torch.where``.  Initialize only
    those unused candidates to one and keep vLLM's exact generator-driven draw
    for every stochastic row.  Pure greedy batches never enter this function;
    fully seeded random batches retain upstream behavior. An unseeded
    stochastic batch fails closed.

    This is installed immediately after model construction and CUDA-graph
    warmup, before serving starts. That preserves vLLM's discarded synthetic
    sampler warmup while protecting all real requests. It is deliberately
    fail-closed on an incompatible sampler ABI instead of silently claiming
    protection on an unknown vLLM implementation.
    """

    global _PATCHED_VLLM_MIXED_BATCH_RNG
    if _PATCHED_VLLM_MIXED_BATCH_RNG:
        return True

    try:
        from vllm.v1.sample.ops import topk_topp_sampler as sampler_module
    except ImportError:
        logger.error(
            "vLLM native top-k/top-p sampler is unavailable; cannot install "
            "mixed-batch CUDA RNG isolation"
        )
        return False

    original = getattr(sampler_module, "random_sample", None)
    if not callable(original):
        logger.error(
            "vLLM random_sample ABI is unavailable; cannot install "
            "mixed-batch CUDA RNG isolation"
        )
        return False
    try:
        parameter_names = tuple(inspect.signature(original).parameters)
    except (TypeError, ValueError):
        parameter_names = ()
    if parameter_names != ("probs", "generators"):
        logger.error(
            "vLLM random_sample ABI is unsupported; expected "
            "(probs, generators), got %s",
            parameter_names,
        )
        return False
    if getattr(original, "_verathos_mixed_batch_rng_v1", False):
        _PATCHED_VLLM_MIXED_BATCH_RNG = True
        return True

    @functools.wraps(original)
    def random_sample(probs, generators):
        row_count = int(probs.shape[0])
        generator_count = len(generators)
        if generator_count == 0:
            # This helper is never reached for an all-greedy batch.  An empty
            # map here therefore means an unseeded stochastic request escaped
            # the serving boundary and would use the unsafe global generator.
            raise RuntimeError(
                "vLLM stochastic sampling requires a private per-request seed"
            )
        if generator_count == row_count:
            return original(probs, generators)

        # Every stochastic Verathos request has a private generator.  Missing
        # entries are consequently greedy rows, whose candidate is ignored by
        # vLLM after this helper returns.  Avoid the process-global RNG without
        # adding one generator/kernel launch per greedy request.
        import torch

        q = torch.ones_like(probs)
        for index, generator in generators.items():
            q[index].exponential_(generator=generator)
        return probs.div_(q).argmax(dim=-1).view(-1)

    random_sample._verathos_mixed_batch_rng_v1 = True
    random_sample._verathos_upstream_random_sample = original
    sampler_module.random_sample = random_sample
    _PATCHED_VLLM_MIXED_BATCH_RNG = True
    logger.info(
        "verallm.compat: installed mixed-batch CUDA RNG isolation for vLLM"
    )
    return True


def has_customop_oot() -> bool:
    """Check if vLLM supports CustomOp.register_oot."""
    try:
        from vllm.model_executor.custom_op import CustomOp, op_registry_oot  # noqa: F401
        return hasattr(CustomOp, 'register_oot')
    except ImportError:
        return False


def has_splitting_ops() -> bool:
    """Check if CompilationConfig supports splitting_ops."""
    try:
        from vllm.config import CompilationConfig  # noqa: F401
        return hasattr(CompilationConfig, 'splitting_ops')
    except ImportError:
        try:
            from vllm.config.compilation import CompilationConfig  # noqa: F401
            return hasattr(CompilationConfig, 'splitting_ops')
        except ImportError:
            return False


_PATCHED_TORCHBIND_BUF_BYTES = False


def patch_inductor_torchbind_buf_bytes() -> bool:
    """Make torch._inductor.ir.TorchBindObject.get_buf_bytes tolerant.

    On torch 2.11 + vLLM 0.20 + Qwen3.6 (GDN hybrid attention), the
    inductor "count_bytes" pass walks the FX graph and calls
    ``TorchBindObject.get_buf_bytes()`` on every script-object buffer.
    The method asserts that the underlying ``real_script_obj`` has
    ``__obj_flatten__`` so it can sum the flattened tensor bytes.  Some
    vLLM-internal script objects (KV cache, attention metadata wrappers
    introduced for Blackwell FP8 paths) do not define that method, which
    trips an AssertionError before any inference happens:

        torch._inductor.exc.InductorError: AssertionError
        File "torch/_inductor/ir.py", line 9580 in get_buf_bytes
            assert hasattr(real_script_obj, "__obj_flatten__")

    The bytes returned by ``get_buf_bytes`` are only used as a heuristic
    for kernel-fusion scheduling — returning 0 (treating the buffer as
    opaque) is safe and matches the existing ``is_opaque_type`` branch
    just above.  This patch keeps the original behaviour for buffers
    that DO have ``__obj_flatten__`` so Qwen3.5-9B and other already-
    working models are unaffected.

    Returns True if the patch was applied (or already in place),
    False if torch doesn't expose ``TorchBindObject`` (older / different
    torch versions).  Idempotent.
    """
    global _PATCHED_TORCHBIND_BUF_BYTES
    if _PATCHED_TORCHBIND_BUF_BYTES:
        return True

    try:
        from torch._inductor import ir as _ir
    except ImportError:
        logger.debug("torch._inductor.ir not available, skip TorchBindObject patch")
        return False

    cls = getattr(_ir, "TorchBindObject", None)
    if cls is None or not hasattr(cls, "get_buf_bytes"):
        logger.debug("TorchBindObject.get_buf_bytes not present, skip patch")
        return False

    original = cls.get_buf_bytes

    def get_buf_bytes(self) -> int:
        real_script_obj = self.get_real_obj()
        # Preserve the upstream opaque-type fast-path.
        try:
            from torch._inductor.utils import is_opaque_type as _is_opaque
            if _is_opaque(real_script_obj):
                return 0
        except Exception:
            pass
        if not hasattr(real_script_obj, "__obj_flatten__"):
            # Treat unknown script objects as opaque (zero bytes for the
            # scheduler heuristic) instead of asserting.  This matches the
            # spirit of the original opaque-type branch.
            logger.debug(
                "verallm.compat: TorchBindObject of type %s lacks "
                "__obj_flatten__; treating as opaque (0 bytes)",
                type(real_script_obj).__name__,
            )
            return 0
        return original(self)

    cls.get_buf_bytes = get_buf_bytes
    _PATCHED_TORCHBIND_BUF_BYTES = True
    logger.info(
        "verallm.compat: patched torch._inductor.ir.TorchBindObject.get_buf_bytes "
        "for backwards-tolerant __obj_flatten__ handling"
    )
    return True


def can_use_cuda_graphs() -> bool:
    """Check if vLLM has both CustomOp OOT and splitting_ops support.

    Both are required for Phase 4 (piecewise CUDA graphs with
    activation capture at split points).
    """
    oot = has_customop_oot()
    split = has_splitting_ops()
    if oot and split:
        logger.info("vLLM supports CUDA graph capture (CustomOp OOT + splitting_ops)")
        return True
    if not oot:
        logger.info("vLLM lacks CustomOp.register_oot — falling back to hooks")
    if not split:
        logger.info("vLLM lacks CompilationConfig.splitting_ops — falling back to hooks")
    return False
