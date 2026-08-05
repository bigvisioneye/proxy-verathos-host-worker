"""Model-semantics extraction for tracker-captured attention material.

ONE source of truth for turning a ``RequestActivationTracker``
reduction entry (raw qkv-output rows, paged-cache K/V, o_proj input
rows) into the float tensors the scored pipeline consumes -- shared by
the qualification E2E and the calibration freeze generator so the
per-architecture details can never drift between them.

Covers both attention families:

* plain (Qwen2-style): q is the first ``nh*hd`` slice, full-dim neox
  rope, no q_norm, no output gate;
* gated (Qwen3-Next family): the fused qkv output packs
  ``[q_h | gate_h]`` PER HEAD (2*hd each), q_norm is GemmaRMSNorm
  (scale ``1 + w``) applied before rope, only the first
  ``hd * partial_rotary_factor`` dims rotate (theta lives in
  ``rope_parameters``; the text-only degenerate of mrope equals
  standard neox rope on the rotary slice), and
  ``o_input = attn_out * sigmoid(gate)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error

__all__ = [
    "CaptureModelSemanticsV3",
    "capture_model_semantics_v3",
    "layer_q_norm_v3",
    "rope_rows_v3",
    "layer_float_tensors_v3",
]


@dataclass(frozen=True, slots=True)
class CaptureModelSemanticsV3:
    """Attention geometry + transform parameters of one text model."""

    nh: int
    nkv: int
    hd: int
    theta: float
    rot_dim: int
    gated: bool

    @property
    def group(self) -> int:
        return self.nh // self.nkv

    @property
    def gated_qkv_width(self) -> int:
        return self.nh * self.hd * 2 + 2 * self.nkv * self.hd

    @property
    def plain_qkv_width(self) -> int:
        return self.nh * self.hd + 2 * self.nkv * self.hd


def capture_model_semantics_v3(config) -> CaptureModelSemanticsV3:
    """Derive the semantics from a HF config (text_config unwrapped)."""

    cfg = getattr(config, "text_config", None) or config
    nh = int(cfg.num_attention_heads)
    nkv = int(cfg.num_key_value_heads)
    hd = int(getattr(cfg, "head_dim", None)
             or int(cfg.hidden_size) // nh)
    rope_params = getattr(cfg, "rope_parameters", None) or {}
    if not isinstance(rope_params, dict):
        rope_params = getattr(rope_params, "__dict__", {}) or {}
    theta = float(rope_params.get("rope_theta")
                  or getattr(cfg, "rope_theta", None) or 1e6)
    partial = float(rope_params.get("partial_rotary_factor")
                    or getattr(cfg, "partial_rotary_factor", None)
                    or 1.0)
    rot_dim = hd if partial >= 1.0 else max(2, int(hd * partial))
    gated = bool(getattr(cfg, "attn_output_gate", False))
    return CaptureModelSemanticsV3(
        nh=nh, nkv=nkv, hd=hd, theta=theta, rot_dim=rot_dim,
        gated=gated)


def layer_q_norm_v3(layers, layer_idx: int):
    """(weight float64, eps) of the layer's q_norm, or None.

    Walks transparent capture wrappers (``.original``) down to the
    attention module that owns the norm."""

    import torch
    import torch.nn as nn

    owner = layers[layer_idx]
    while isinstance(getattr(owner, "original", None),
                     nn.Module) and not hasattr(owner, "self_attn"):
        owner = owner.original
    attn = getattr(owner, "self_attn", None)
    while (attn is not None and not hasattr(attn, "q_norm")
           and isinstance(getattr(attn, "original", None), nn.Module)):
        attn = attn.original
    norm = getattr(attn, "q_norm", None)
    if norm is None or not hasattr(norm, "weight"):
        return None
    eps = float(getattr(norm, "variance_epsilon",
                        getattr(norm, "eps", 1e-6)))
    return norm.weight.detach().cpu().to(torch.float64), eps


def rope_rows_v3(
    x,
    positions,
    semantics: CaptureModelSemanticsV3,
    *,
    runtime_dtype=None,
):
    """Replay vLLM's Neox RoPE/cache precision boundary.

    vLLM constructs the trigonometric cache in float32, casts it to the
    query/key runtime dtype, and writes the result back at that dtype.  The
    returned float64 tensor contains those exact runtime-representable values;
    it is not an idealized float64 RoPE reference.
    """

    import torch

    if runtime_dtype is None:
        runtime_dtype = x.dtype
    if runtime_dtype not in (torch.float16, torch.bfloat16):
        raise ProofV3Error(
            "attention runtime RoPE dtype is unsupported"
        )
    rot = semantics.rot_dim
    inv = 1.0 / (semantics.theta ** (
        torch.arange(0, rot, 2, dtype=torch.float32) / rot))
    pos = torch.tensor(
        [int(p) for p in positions],
        dtype=torch.float32,
    )
    freqs = torch.outer(pos, inv)
    cos = freqs.cos().to(runtime_dtype).to(torch.float32)
    sin = freqs.sin().to(runtime_dtype).to(torch.float32)
    x = x.to(runtime_dtype).to(torch.float32)
    x1 = x[..., :rot // 2]
    x2 = x[..., rot // 2:rot]
    rotated = torch.cat(
        [x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    if rot == x.shape[-1]:
        result = rotated
    else:
        result = torch.cat([rotated, x[..., rot:]], dim=-1)
    return result.to(runtime_dtype).to(torch.float64)


def layer_float_tensors_v3(entry, semantics: CaptureModelSemanticsV3,
                           q_norm=None):
    """(q_post [nh,pool,hd], k_post [nkv,T,hd], v [nkv,T,hd],
    o_rows [pool, nh*hd], positions, gate_sig [pool,nh,hd] or None).

    ``q_norm``: (weight float64, eps) from :func:`layer_q_norm_v3`;
    applied Gemma-style (scale ``1 + w``) before rope on gated models.
    """

    import torch

    nh, hd = semantics.nh, semantics.hd
    positions = tuple(int(p) for p in entry["positions"])
    rows = entry["q_rows"]
    width = int(rows.shape[1])
    gate_sig = None
    # the SIGNED semantics decide the qkv layout; the captured width
    # only validates it -- a mismatch is a capture-wiring failure and
    # MUST fail closed, never fall back to the other slicing
    if semantics.gated:
        if width != semantics.gated_qkv_width:
            raise ProofV3Error(
                f"gated qkv capture is {width} wide but the layout "
                f"needs {semantics.gated_qkv_width} (mis-wired "
                "capture; refusing to extract)")
        if q_norm is None:
            raise ProofV3Error(
                "gated attention extraction requires the layer q_norm "
                "(none was resolved; refusing to extract)")
        q_gate = rows[:, :nh * hd * 2].view(len(positions), nh, 2 * hd)
        q = q_gate[..., :hd].permute(1, 0, 2)
        gate_sig = torch.sigmoid(
            q_gate[..., hd:].cpu().to(torch.float64))
    else:
        if width != semantics.plain_qkv_width:
            raise ProofV3Error(
                f"qkv capture is {width} wide but the ungated layout "
                f"needs {semantics.plain_qkv_width} (mis-wired "
                "capture; refusing to extract)")
        q = (rows[:, :nh * hd].view(len(positions), nh, hd)
             .permute(1, 0, 2))
    runtime_dtype = rows.dtype
    qf = q.cpu().to(torch.float64)
    if gate_sig is not None:
        weight, eps = q_norm
        qf = qf * torch.rsqrt(
            (qf * qf).mean(-1, keepdim=True) + eps) * (1.0 + weight)
        # The graph-integrated q_norm writes its output at activation dtype
        # before the rotary kernel consumes it.
        qf = qf.to(runtime_dtype).to(torch.float64)
    q_post = rope_rows_v3(
        qf,
        positions,
        semantics,
        runtime_dtype=runtime_dtype,
    )
    k_post = entry["keys"].cpu().permute(1, 0, 2).to(torch.float64)
    v = entry["values"].cpu().permute(1, 0, 2).to(torch.float64)
    o = entry["o_rows"][:, :nh * hd].cpu().to(torch.float64)
    return q_post, k_post, v, o, positions, gate_sig
