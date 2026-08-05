"""Model-derived sizing for graph-integrated proof capture staging."""

from __future__ import annotations

import re

from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.projection_manifest import ProjectionManifestV3
from verallm.proof_v3.runtime_architecture import (
    manifest_projection_suffixes_v3,
)

CAPTURE_STAGING_BUDGET_BYTES_V3 = 5 << 29  # 2.5 GiB
MAX_CAPTURE_STAGING_ROWS_V3 = 2048

_LAYER_ENTRY = re.compile(r"^l([0-9]+)\.([a-z0-9_]+)$")

__all__ = [
    "CAPTURE_STAGING_BUDGET_BYTES_V3",
    "MAX_CAPTURE_STAGING_ROWS_V3",
    "dense_capture_bytes_per_row_v3",
    "recommended_dense_capture_staging_rows_v3",
]


def dense_capture_bytes_per_row_v3(
    manifest: ProjectionManifestV3,
) -> int:
    """Return exact fp16/bf16 buffer bytes allocated per scheduler row.

    Each registered dense projection owns one input and one output capture
    buffer. Each decoder layer additionally owns residual input/output
    buffers. ``gate_up`` is mandatory in the qualified dense shell and
    provides the residual hidden width without model-name assumptions.
    """

    if not isinstance(manifest, ProjectionManifestV3):
        raise ProofV3Error("capture staging manifest is malformed")
    suffixes = manifest_projection_suffixes_v3()
    projections: dict[int, dict[str, object]] = {}
    for entry in manifest.entries:
        match = _LAYER_ENTRY.fullmatch(entry.name)
        if match is None or match.group(2) not in suffixes:
            continue
        layer = int(match.group(1))
        suffix = match.group(2)
        layer_entries = projections.setdefault(layer, {})
        if suffix in layer_entries:
            raise ProofV3Error(
                "capture staging manifest has duplicate layer projections"
            )
        layer_entries[suffix] = entry
    layers = tuple(sorted(projections))
    if not layers or layers != tuple(range(len(layers))):
        raise ProofV3Error(
            "capture staging manifest layer inventory is not contiguous"
        )

    elements_per_row = 0
    for layer in layers:
        layer_entries = projections[layer]
        gate_up = layer_entries.get("gate_up")
        if gate_up is None:
            raise ProofV3Error(
                f"capture staging manifest lacks l{layer}.gate_up"
            )
        hidden = int(gate_up.in_dim)
        elements_per_row += 2 * hidden
        elements_per_row += sum(
            int(entry.in_dim) + int(entry.out_dim)
            for entry in layer_entries.values()
        )
    # Qualified runtime activation encodings are fp16.v1 or bf16.v1.
    return elements_per_row * 2


def recommended_dense_capture_staging_rows_v3(
    manifest: ProjectionManifestV3,
    *,
    budget_bytes: int = CAPTURE_STAGING_BUDGET_BYTES_V3,
    maximum_rows: int = MAX_CAPTURE_STAGING_ROWS_V3,
) -> int:
    """Choose a deterministic power-of-two scheduler chunk within budget."""

    if (
        isinstance(budget_bytes, bool)
        or not isinstance(budget_bytes, int)
        or budget_bytes <= 0
        or isinstance(maximum_rows, bool)
        or not isinstance(maximum_rows, int)
        or maximum_rows <= 0
    ):
        raise ProofV3Error("capture staging budget is malformed")
    bytes_per_row = dense_capture_bytes_per_row_v3(manifest)
    affordable = min(maximum_rows, budget_bytes // bytes_per_row)
    if affordable < 1:
        raise ProofV3Error(
            "capture staging budget cannot hold one scheduler row"
        )
    return 1 << (affordable.bit_length() - 1)
