"""Signed scored-attention calibration blob: load + digest verification.

The calibration (per-layer per-head ``ScoredHeadParamsV3`` + the bridge
bounds, plus the discriminative layer set) ships beside the signed
manifest as a JSON blob; only its sha256 (``attn_calibration_digest``,
computed by ``scored_calibration_digest_v3``) is inside the authority
signature.  The validator loads the blob, recomputes the digest, and
fails closed on any mismatch before a single parameter is used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.scored_attention_reference import (
    ScoredBridgeBoundsV3,
    ScoredHeadParamsV3,
    scored_calibration_digest_v3,
)

__all__ = [
    "ScoredCalibrationV3",
    "dump_scored_calibration_v3",
    "load_scored_calibration_v3",
]


@dataclass(frozen=True, slots=True)
class ScoredCalibrationV3:
    """Digest-verified calibration: what the hard attention audit reads."""

    layers: dict
    discriminative: tuple[int, ...]
    digest: bytes

    def heads_for(self, layer: int):
        heads = self.layers.get(int(layer))
        if heads is None:
            raise ProofV3Error(
                f"calibration has no entry for layer {layer}")
        return heads


def _head_from_obj(obj: dict) -> tuple[ScoredHeadParamsV3, ScoredBridgeBoundsV3]:
    k_num = tuple(int(v) for v in obj["k_num"])
    k_e = tuple(int(v) for v in obj["k_e"])
    if len(k_num) != len(k_e) or not k_num:
        raise ProofV3Error("calibration head has malformed k scales")
    params = ScoredHeadParamsV3(
        head_dim=int(obj.get("head_dim", len(k_num))),
        qk_qmax=int(obj.get("qk_qmax", 4095)),
        q_num=int(obj["q_num"]), q_e=int(obj["q_e"]),
        k_num=k_num, k_e=k_e,
        m_num=int(obj["m_num"]), m_e=int(obj["m_e"]),
        v_num=int(obj["v_num"]), v_e=int(obj["v_e"]),
        ox_num=int(obj["ox_num"]), ox_e=int(obj["ox_e"]),
    )
    gated = obj.get("gated", False)
    if not isinstance(gated, bool):
        # strict: the flag selects the bridge RELATION the signed
        # bounds were measured under; coercing truthy values would
        # let a malformed blob flip it silently
        raise ProofV3Error("calibration head 'gated' must be a boolean")
    bounds = ScoredBridgeBoundsV3(
        cell_bound=int(obj["cell_bound"]),
        row_sq_bound=int(obj["row_sq_bound"]),
        gated=gated,
    )
    return params, bounds


def load_scored_calibration_v3(source) -> ScoredCalibrationV3:
    """Parse a calibration blob (dict, JSON string, or file path) and
    verify its self-declared digest.  Callers must additionally compare
    ``.digest`` against the manifest's ``attn_calibration_digest``."""

    if isinstance(source, (str, bytes)) and not str(source).lstrip().startswith("{"):
        with open(source, "rb") as handle:
            obj = json.load(handle)
    elif isinstance(source, (str, bytes)):
        obj = json.loads(source)
    else:
        obj = source
    if not isinstance(obj, dict):
        raise ProofV3Error("calibration blob must be a JSON object")
    try:
        discriminative = tuple(sorted(int(x) for x in obj.get("discriminative", ())))
        layers = {
            int(layer): tuple(_head_from_obj(head) for head in heads)
            for layer, heads in obj["layers"].items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ProofV3Error(f"calibration blob is malformed: {exc}") from exc
    digest = scored_calibration_digest_v3(layers, discriminative=discriminative)
    declared = obj.get("digest")
    if declared is not None and bytes.fromhex(declared) != digest:
        raise ProofV3Error(
            "calibration blob digest does not match its contents")
    return ScoredCalibrationV3(
        layers=layers, discriminative=discriminative, digest=digest)


def dump_scored_calibration_v3(calibration: ScoredCalibrationV3) -> dict:
    """Canonical JSON-ready dict; round-trips through
    ``load_scored_calibration_v3`` to the same digest."""

    return {
        "discriminative": [int(x) for x in calibration.discriminative],
        "layers": {
            str(layer): [
                {
                    "head_dim": params.head_dim,
                    "qk_qmax": params.qk_qmax,
                    "q_num": params.q_num, "q_e": params.q_e,
                    "k_num": [int(v) for v in params.k_num],
                    "k_e": [int(v) for v in params.k_e],
                    "m_num": params.m_num, "m_e": params.m_e,
                    "v_num": params.v_num, "v_e": params.v_e,
                    "ox_num": params.ox_num, "ox_e": params.ox_e,
                    "cell_bound": bounds.cell_bound,
                    "row_sq_bound": bounds.row_sq_bound,
                    # emitted only when set: ungated blobs keep their
                    # exact historical JSON shape and digest
                    **({"gated": True} if getattr(
                        bounds, "gated", False) else {}),
                }
                for params, bounds in heads
            ]
            for layer, heads in calibration.layers.items()
        },
        "digest": calibration.digest.hex(),
    }
