"""Serving-path capture bridge: vLLM activations -> proof-v3 commitments.

Bridges the PRODUCTION capture plane (the CUDA-graph-safe activation
capture already shipped for the serving engine: buffer-mode
``CaptureLinearWrapper`` copies inside the graph, ``verallm::buffer_copy``
static-grid Triton kernels on Ampere, per-request demux in
``RequestActivationTracker``) to the proof-v3 commitment plane:

* **Always-on (every request)**: each captured oracle tensor is quantized
  on-device and capture-committed (cheap Merkle root, no NTT) using the
  fused GPU SHA-256 tree when available -- byte-identical to the CPU
  reference ``capture_commit_v3``.  This is the undetectable pre-nonce
  commitment: the miner cannot know at commit time whether the request
  will be audited.
* **Hard audit (sampled)**: the SAME device tensor feeds the succinct
  fold argument (fused NTT + tree commit + sumcheck), and the resulting
  witness is bound to the pre-posted capture root, so the proof provably
  covers exactly the activation the serving engine produced.

CUDA-graph note: nothing in this module runs inside the graph.  The
capture copies inside the graph are the shipped buffer-mode wrappers;
this bridge consumes the post-step readout tensors, so serving keeps
full CUDA-graph acceleration (enforce_eager stays False).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.goldilocks_capture_commitment_reference import (
    GoldilocksCaptureCommitmentV3,
    capture_commit_v3,
)

_CAPTURE_BINDING_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/SERVING_CAPTURE_BRIDGE/V1/ORACLE"
)


@dataclass(frozen=True, slots=True)
class ServingOracleCaptureV3:
    """One committed serving oracle: identity, quantization, root."""

    oracle_name: str
    request_binding_digest: bytes
    token_count: int
    width: int
    quant_scale: float
    variable_count: int
    root: bytes


def oracle_binding_digest_v3(
    *, request_binding_digest: bytes, oracle_name: str
) -> bytes:
    if not isinstance(oracle_name, str) or not oracle_name:
        raise ProofV3Error("serving oracle name is malformed")
    if (
        not isinstance(request_binding_digest, bytes)
        or len(request_binding_digest) != 32
    ):
        raise ProofV3Error("serving request binding digest is malformed")
    return hashlib.sha256(
        _CAPTURE_BINDING_DOMAIN
        + request_binding_digest
        + oracle_name.encode()
    ).digest()


def quantize_serving_oracle_int8_v3(tensor):
    """Quantize a [tokens, width] device tensor to int8 field values.

    Returns ``(field_values_int64_device, scale, t_pad, w_pad)`` where the
    field tensor is the zero-padded power-of-two multilinear evaluation
    vector (row-major), still on the capture device.  The quantization is
    the deterministic per-tensor absmax/127 contract used across the
    qualification chain.
    """

    import torch

    if tensor.dim() != 2:
        raise ProofV3Error("serving oracle tensor must be [tokens, width]")
    tokens, width = tensor.shape
    if tokens < 1 or width < 1:
        raise ProofV3Error("serving oracle tensor is empty")
    work = tensor.detach().to(torch.float32)
    scale = float(work.abs().max()) / 127.0 or 1.0
    quantized = torch.round(work / scale).to(torch.int64)
    t_pad = 1 << max(0, (tokens - 1).bit_length())
    w_pad = 1 << (width - 1).bit_length()
    padded = torch.zeros(
        (t_pad, w_pad), dtype=torch.int64, device=work.device
    )
    padded[:tokens, :width] = quantized
    flat = padded.reshape(-1)
    flat = torch.where(flat < 0, flat + GOLDILOCKS_MODULUS, flat)
    return flat, scale, t_pad, w_pad


def capture_commit_serving_oracle_v3(
    *,
    request_binding_digest: bytes,
    oracle_name: str,
    tensor,
    tree_extension=None,
) -> ServingOracleCaptureV3:
    """Capture-commit one serving oracle tensor (GPU tree if available).

    ``tree_extension`` is the loaded fused SHA-256 tree module; when None
    (or the tensor is on CPU) the byte-identical CPU reference path runs.
    The root is identical either way -- conformance is asserted in the
    serving qualification.
    """

    tokens, width = tensor.shape
    flat, scale, t_pad, w_pad = quantize_serving_oracle_int8_v3(tensor)
    binding = oracle_binding_digest_v3(
        request_binding_digest=request_binding_digest,
        oracle_name=oracle_name,
    )
    leaf_binding = hashlib.sha256(
        b"VERATHOS/PROOF_V3/GOLDILOCKS_CAPTURE/V1/COMMIT/SHA256"
        + b"LEAVES/"
        + binding
    ).digest()
    if tree_extension is not None and flat.is_cuda:
        from verallm.proof_v3.native_cuda_tree_backend import (
            fused_merkle_root_w1,
        )

        root = fused_merkle_root_w1(
            tree_extension, flat, binding_digest=leaf_binding
        )
    else:
        commitment = capture_commit_v3(
            validator_binding_digest=binding,
            raw_values=tuple(int(v) for v in flat.tolist()),
        )
        root = commitment.root
    return ServingOracleCaptureV3(
        oracle_name=oracle_name,
        request_binding_digest=request_binding_digest,
        token_count=tokens,
        width=width,
        quant_scale=scale,
        variable_count=(t_pad * w_pad).bit_length() - 1,
        root=root,
    )


def capture_commit_serving_request_v3(
    *,
    request_binding_digest: bytes,
    oracles: dict,
    tree_extension=None,
) -> dict[str, ServingOracleCaptureV3]:
    """Commit every captured oracle of one served request.

    ``oracles`` maps oracle name -> [tokens, width] tensor (the tracker's
    finalized activations).  Returns name -> capture record.  This is the
    ALWAYS-ON cost: O(N) hashing per oracle, no NTT, no proof.
    """

    return {
        name: capture_commit_serving_oracle_v3(
            request_binding_digest=request_binding_digest,
            oracle_name=name,
            tensor=tensor,
            tree_extension=tree_extension,
        )
        for name, tensor in sorted(oracles.items())
    }


def capture_chain_digest_v3(
    *,
    captures: dict,
    chunk_claims_digest: bytes = b"",
    attention_commitments: tuple = (),
) -> bytes:
    """Canonical digest of one request's full capture-root set.

    This is the value signed into receipt v3 (``capture_chain_digest``)
    at response time: name-ordered oracle identities, shapes, quant
    scales and roots, plus (optionally) the digest of the per-chunk
    softmax claims.  Everything a retroactive audit reveals against is
    pinned by this one signed value BEFORE any nonce exists.
    """

    parts = [b"VERATHOS/PROOF_V3/CAPTURE_CHAIN/V1"]
    for name in sorted(captures):
        oracle = captures[name]
        if not isinstance(oracle, ServingOracleCaptureV3):
            raise ProofV3Error("capture chain entry is malformed")
        encoded_name = oracle.oracle_name.encode()
        parts.append(struct.pack("<I", len(encoded_name)))
        parts.append(encoded_name)
        parts.append(oracle.request_binding_digest)
        parts.append(struct.pack(
            "<IIId",
            oracle.token_count,
            oracle.width,
            oracle.variable_count,
            oracle.quant_scale,
        ))
        parts.append(oracle.root)
    parts.append(struct.pack("<I", len(chunk_claims_digest)))
    parts.append(chunk_claims_digest)
    # per-audited-layer attention capture-root commitments (q/k/v/attn_o roots +
    # claims digest), so the attention section's roots are pinned pre-nonce too.
    parts.append(struct.pack("<I", len(attention_commitments)))
    for commitment in attention_commitments:
        parts.append(bytes(commitment))
    return hashlib.sha256(b"".join(parts)).digest()
