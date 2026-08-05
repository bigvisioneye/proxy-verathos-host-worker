"""Qualified decoder RMSNorm semantics for economic proof-v3.

The projection manifest authenticates the stored normalization weight. This
small signed ABI tells the verifier whether that parameter is the effective
gain itself or whether the runtime applies the Gemma-style ``1 + weight``
transform, and pins the model's epsilon.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError

RMSNORM_WEIGHT_GAIN_V3 = "rmsnorm.weight_gain.v1"
RMSNORM_OFFSET_ONE_GAIN_V3 = "rmsnorm.offset_one_gain.v1"
DEFAULT_RMSNORM_EPSILON_V3 = 1e-6
DEFAULT_RMSNORM_EPSILON_BITS_V3 = struct.unpack(
    "<Q", struct.pack("<d", DEFAULT_RMSNORM_EPSILON_V3)
)[0]

_SUPPORTED_SEMANTICS = frozenset(
    (RMSNORM_WEIGHT_GAIN_V3, RMSNORM_OFFSET_ONE_GAIN_V3)
)

__all__ = [
    "DEFAULT_RMSNORM_EPSILON_BITS_V3",
    "DEFAULT_RMSNORM_EPSILON_V3",
    "RMSNORM_OFFSET_ONE_GAIN_V3",
    "RMSNORM_WEIGHT_GAIN_V3",
    "decode_rmsnorm_runtime_semantics_v3",
    "qualify_rmsnorm_runtime_semantics_v3",
    "resolve_final_rmsnorm_module_v3",
    "validate_rmsnorm_runtime_semantics_v3",
]


def _epsilon_from_bits(value: int) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value < 1 << 64
    ):
        raise ProofV3Error("RMSNorm epsilon bits are malformed")
    epsilon = struct.unpack("<d", struct.pack("<Q", value))[0]
    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise ProofV3Error("RMSNorm epsilon is outside the qualified range")
    return epsilon


def validate_rmsnorm_runtime_semantics_v3(
    semantics_id: str,
    epsilon_bits: int,
) -> None:
    if semantics_id not in _SUPPORTED_SEMANTICS:
        raise ProofV3Error("RMSNorm runtime semantics are unsupported")
    _epsilon_from_bits(epsilon_bits)


def decode_rmsnorm_runtime_semantics_v3(
    semantics_id: str,
    epsilon_bits: int,
) -> tuple[float, float]:
    """Return ``(stored_weight_offset, epsilon)`` for verification."""

    try:
        validate_rmsnorm_runtime_semantics_v3(semantics_id, epsilon_bits)
    except ProofV3Error as exc:
        raise ProofV3VerificationError(str(exc)) from exc
    offset = 1.0 if semantics_id == RMSNORM_OFFSET_ONE_GAIN_V3 else 0.0
    return offset, _epsilon_from_bits(epsilon_bits)


def _unwrap(module):
    try:
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - qualification dependency.
        raise ProofV3Error("RMSNorm qualification requires torch") from exc
    owner = module
    seen: set[int] = set()
    while (
        isinstance(getattr(owner, "original", None), nn.Module)
        and id(owner) not in seen
    ):
        seen.add(id(owner))
        owner = owner.original
    return owner


def _module_semantics(module) -> tuple[str, float]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - qualification dependency.
        raise ProofV3Error("RMSNorm qualification requires torch") from exc

    owner = _unwrap(module)
    weight = getattr(owner, "weight", None)
    epsilon = getattr(
        owner,
        "variance_epsilon",
        getattr(owner, "eps", None),
    )
    if (
        not isinstance(weight, torch.Tensor)
        or weight.ndim != 1
        or weight.numel() < 1
        or not weight.is_floating_point()
        or not bool(torch.isfinite(weight).all())
        or epsilon is None
    ):
        raise ProofV3Error("qualified decoder RMSNorm module is malformed")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise ProofV3Error("qualified decoder RMSNorm epsilon is unsupported")

    class_names = {cls.__name__ for cls in type(owner).__mro__}
    if "GemmaRMSNorm" in class_names:
        semantics_id = RMSNORM_OFFSET_ONE_GAIN_V3
    elif "RMSNorm" in class_names or any(
        name.endswith("RMSNorm") for name in class_names
    ):
        semantics_id = RMSNORM_WEIGHT_GAIN_V3
    else:
        raise ProofV3Error(
            "qualified decoder normalization is not a supported RMSNorm"
        )
    return semantics_id, epsilon


def qualify_rmsnorm_runtime_semantics_v3(
    modules: Iterable[object],
) -> tuple[str, int]:
    """Qualify one uniform decoder/final RMSNorm ABI from loaded modules."""

    qualified = tuple(_module_semantics(module) for module in modules)
    if not qualified:
        raise ProofV3Error("RMSNorm qualification needs at least one module")
    semantics = {item[0] for item in qualified}
    epsilon_bits = {
        struct.unpack("<Q", struct.pack("<d", item[1]))[0]
        for item in qualified
    }
    if len(semantics) != 1 or len(epsilon_bits) != 1:
        raise ProofV3Error(
            "decoder/final RMSNorm semantics vary within the qualified model"
        )
    semantics_id = next(iter(semantics))
    bits = next(iter(epsilon_bits))
    validate_rmsnorm_runtime_semantics_v3(semantics_id, bits)
    return semantics_id, bits


def resolve_final_rmsnorm_module_v3(
    model,
    *,
    decoder_layers: Iterable[object],
    hidden_dim: int,
):
    """Resolve the unique text-stack final RMSNorm by runtime structure.

    Nested multimodal models can expose a vision ``norm`` at a shallower path
    than the language model's final norm. Names and path depth are therefore
    not trustworthy. Qualification excludes every decoder-layer module and
    requires exactly one remaining supported RMSNorm whose parameter width
    matches the signed text hidden dimension.
    """

    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - qualification dependency.
        raise ProofV3Error("RMSNorm qualification requires torch") from exc
    if (
        not isinstance(model, nn.Module)
        or isinstance(hidden_dim, bool)
        or not isinstance(hidden_dim, int)
        or hidden_dim < 1
    ):
        raise ProofV3Error("final RMSNorm qualification inputs are malformed")
    layers = tuple(decoder_layers)
    if not layers or any(not isinstance(layer, nn.Module) for layer in layers):
        raise ProofV3Error("decoder layer inventory is malformed")
    decoder_module_ids = {
        id(module)
        for layer in layers
        for module in layer.modules()
    }
    candidates = []
    for module in model.modules():
        if id(module) in decoder_module_ids:
            continue
        owner = _unwrap(module)
        weight = getattr(owner, "weight", None)
        if (
            not isinstance(weight, torch.Tensor)
            or weight.ndim != 1
            or weight.numel() != hidden_dim
        ):
            continue
        try:
            _module_semantics(owner)
        except ProofV3Error:
            continue
        if all(id(owner) != id(existing) for existing in candidates):
            candidates.append(owner)
    if len(candidates) != 1:
        raise ProofV3Error(
            "qualified model does not expose one unique text final RMSNorm"
        )
    return candidates[0]
