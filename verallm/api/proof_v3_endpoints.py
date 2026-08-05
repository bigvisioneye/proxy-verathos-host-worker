"""Self-contained proof-v3 serving endpoints + coordinator (flag-gated).

Kept OUT of server.py's hot path: server.py only (1) constructs one
``ProofV3Endpoints`` when the ``proof_v3_enabled`` rollout flag is on,
(2) calls ``commit_request`` once per served request with the captured
projection-input activations, (3) stamps the returned digest into the
receipt, and (4) mounts ``register_routes``.  Everything is inert when the
flag is off, so the live v1/v2 path is untouched.

Weight Merkle trees are the STATIC registry roots -- built once per
projection and cached here -- so per-request work is only int8-quantizing
the captured inputs plus the exact surrogate matmul (dtype-agnostic:
int4/fp8/nvfp4/bf16 all reduce to the same int8 surrogate; see
verallm/miner/proof_v3_projection_audit.py).
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass

from pydantic import BaseModel

from verallm.miner.proof_v3_projection_audit import (
    ProjectionAuditMaterialV3,
    answer_projection_audit_v3,
    build_projection_material_v3,
    derive_projection_audit_plan_v3,
    int8_reduce_rows,
    verify_projection_audit_v3,
)
from verallm.proof_v3.errors import ProofV3Error

_DIGEST_DOMAIN = b"VERATHOS/PROOF_V3/PROJECTION_CHAIN/V1"

__all__ = [
    "ProofV3RequestMaterials",
    "ProofV3Endpoints",
    "ProofV3AuditBody",
    "register_routes",
]


class ProofV3AuditBody(BaseModel):
    request_id: str
    validator_nonce: str  # hex
    projection: str


class ProofV3AttentionAuditBody(BaseModel):
    request_id: str
    validator_nonce: str  # hex
    row_samples: int = 32
    chunk_samples: int = 3


class ProofV3ReductionAuditBody(BaseModel):
    request_id: str
    validator_nonce: str  # hex
    heads_per_layer: int = 2
    row_samples: int = 8
    uniform_chunk_samples: int = 3
    mass_draws: int = 2


@dataclass(frozen=True, slots=True)
class ProofV3RequestMaterials:
    """All projection materials + signed digest for one served request."""

    projections: dict[str, ProjectionAuditMaterialV3]
    capture_chain_digest: bytes


class ProofV3Endpoints:
    """Per-miner proof-v3 coordinator.  Thread-safe, memory-bounded."""

    def __init__(self, *, max_records: int = 256, chunk_size: int = 128) -> None:
        self._materials: dict[str, ProofV3RequestMaterials] = {}
        self._order: list[str] = []
        self._max_records = max(1, int(max_records))
        self._chunk_size = int(chunk_size)
        # Static per-projection weight material (root, int8 rows, tree, dims),
        # keyed by name -> built once, reused every request.
        self._weight_cache: dict = {}
        self._lock = threading.Lock()

    # -- commit (serve time) --------------------------------------------
    def commit_request(
        self,
        *,
        request_id: str,
        captured_inputs: dict,
        weights: dict,
    ) -> bytes:
        """Build + retain projection materials, return the signed digest.

        ``captured_inputs[name]`` is the real activation ``[token][in_dim]``
        captured at projection ``name``'s input; ``weights[name]`` is that
        projection's weight ``[out_dim][in_dim]`` (any dtype).  The digest
        is what receipt v3 signs pre-nonce.
        """

        if not request_id:
            raise ProofV3Error("proof-v3 commit needs a request id")
        from verallm.miner.proof_v3_projection_audit import (
            projection_weight_root_v3,
        )

        projections: dict[str, ProjectionAuditMaterialV3] = {}
        for name in sorted(captured_inputs):
            if name not in weights:
                continue
            # Weight tree is STATIC per model -> build once, cache, reuse; the
            # per-request cost is then only int8(input) + surrogate matmul.
            wm = self._weight_cache.get(name)
            if wm is None:
                wm = projection_weight_root_v3(weights[name], self._chunk_size)
                self._weight_cache[name] = wm
            material = build_projection_material_v3(
                name=name,
                captured_input=captured_inputs[name],
                weight=weights[name],
                chunk_size=self._chunk_size,
                weight_material=wm,
            )
            projections[name] = material
        if not projections:
            raise ProofV3Error("no projection materials to commit")
        digest = self._chain_digest(projections)
        record = ProofV3RequestMaterials(
            projections=projections, capture_chain_digest=digest)
        with self._lock:
            self._store(request_id, record)
        return digest

    def _chain_digest(self, projections: dict) -> bytes:
        parts = [_DIGEST_DOMAIN]
        for name in sorted(projections):
            material = projections[name]
            encoded = name.encode()
            parts.append(len(encoded).to_bytes(4, "little"))
            parts.append(encoded)
            parts.append(material.weight_root)
            parts.append(int(material.out_dim).to_bytes(4, "little"))
            parts.append(int(material.in_dim).to_bytes(4, "little"))
            for row in material.int8_input:
                for val in row:
                    parts.append((int(val) & 0xFF).to_bytes(1, "little"))
            for row in material.output:
                for val in row:
                    parts.append(int(val).to_bytes(16, "little", signed=True))
        return hashlib.sha256(b"".join(parts)).digest()

    def _store(self, request_id: str, record: ProofV3RequestMaterials) -> None:
        if request_id in self._materials:
            self._order.remove(request_id)
        self._materials[request_id] = record
        self._order.append(request_id)
        while len(self._order) > self._max_records:
            self._materials.pop(self._order.pop(0), None)

    def get_digest(self, request_id: str) -> bytes | None:
        with self._lock:
            record = self._materials.get(request_id)
        return None if record is None else record.capture_chain_digest

    # -- audit (post-nonce) ---------------------------------------------
    def answer_audit(
        self,
        *,
        request_id: str,
        validator_nonce: bytes,
        projection: str,
        token_samples: int = 2,
        out_samples: int = 8,
    ):
        with self._lock:
            record = self._materials.get(request_id)
        if record is None:
            raise ProofV3Error("no retained materials for this request")
        material = record.projections.get(projection)
        if material is None:
            raise ProofV3Error("unknown projection for this request")
        tokens, outs = derive_projection_audit_plan_v3(
            validator_nonce=validator_nonce,
            capture_chain_digest=record.capture_chain_digest,
            token_count=len(material.int8_input),
            out_dim=material.out_dim,
            token_samples=token_samples,
            out_samples=out_samples,
        )
        reveal = answer_projection_audit_v3(
            material=material, tokens=tokens, out_indices=outs)
        return record.capture_chain_digest, reveal, material

    def projection_names(self, request_id: str) -> tuple[str, ...]:
        with self._lock:
            record = self._materials.get(request_id)
        return () if record is None else tuple(sorted(record.projections))


def register_routes(app, get_endpoints, get_attention=None,
                    get_reduction=None) -> None:
    """Mount ``/proof_v3/audit`` (+ ``/proof_v3/attention_audit``,
    ``/proof_v3/reduction_audit``) on the miner FastAPI app.

    ``get_endpoints()`` returns the live ``ProofV3Endpoints`` or None when
    the feature is disabled (route then 501s -- inert).
    ``get_attention()`` returns ``(ProofV3ServingCoordinator, oracle_set_by
    request_id callable | None)`` for the hard attention audit, or None.
    ``get_reduction()`` returns the live ``ReductionAuditCoordinatorV3``
    (production hard-canary attention path) or None.
    """

    from fastapi.responses import JSONResponse

    @app.post("/proof_v3/reduction_audit")
    async def proof_v3_reduction_audit(body: ProofV3ReductionAuditBody):  # noqa: ANN001
        coordinator = get_reduction() if get_reduction is not None else None
        if coordinator is None:
            return JSONResponse(
                status_code=501,
                content={"error": "proof-v3 reduction audit is not enabled"})
        try:
            nonce = bytes.fromhex(body.validator_nonce)
        except ValueError:
            return JSONResponse(
                status_code=400, content={"error": "invalid nonce hex"})
        # miner-side self-protection: bound the answer work regardless of
        # what the request asks for (the wire caps would reject more anyway)
        if not (1 <= int(body.heads_per_layer) <= 8
                and 1 <= int(body.row_samples) <= 64
                and 1 <= int(body.uniform_chunk_samples) <= 64
                and 1 <= int(body.mass_draws) <= 64):
            return JSONResponse(
                status_code=400,
                content={"error": "sampling parameters are out of bounds"})
        try:
            wire = coordinator.answer_wire(
                request_id=body.request_id, validator_nonce=nonce,
                heads_per_layer=int(body.heads_per_layer),
                row_samples=int(body.row_samples),
                uniform_chunk_samples=int(body.uniform_chunk_samples),
                mass_draws=int(body.mass_draws))
        except ProofV3Error as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        return {"wire": wire.hex()}

    @app.post("/proof_v3/reduction_bundle")
    async def proof_v3_reduction_bundle(body: ProofV3ReductionAuditBody):  # noqa: ANN001
        """ONE canary's bundled response: a domain-separated subaudit
        per retained signed layer, single wire envelope."""

        coordinator = get_reduction() if get_reduction is not None else None
        if coordinator is None:
            return JSONResponse(
                status_code=501,
                content={"error": "proof-v3 reduction audit is not enabled"})
        try:
            nonce = bytes.fromhex(body.validator_nonce)
        except ValueError:
            return JSONResponse(
                status_code=400, content={"error": "invalid nonce hex"})
        if not (1 <= int(body.heads_per_layer) <= 8
                and 1 <= int(body.row_samples) <= 64
                and 1 <= int(body.uniform_chunk_samples) <= 64
                and 1 <= int(body.mass_draws) <= 64):
            return JSONResponse(
                status_code=400,
                content={"error": "sampling parameters are out of bounds"})
        try:
            wire = coordinator.answer_bundle_wire(
                request_id=body.request_id, validator_nonce=nonce,
                heads_per_layer=int(body.heads_per_layer),
                row_samples=int(body.row_samples),
                uniform_chunk_samples=int(body.uniform_chunk_samples),
                mass_draws=int(body.mass_draws))
        except ProofV3Error as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        return {"wire": wire.hex()}

    @app.post("/proof_v3/attention_audit")
    async def proof_v3_attention_audit(body: ProofV3AttentionAuditBody):  # noqa: ANN001
        provider = get_attention() if get_attention is not None else None
        if provider is None:
            return JSONResponse(
                status_code=501,
                content={"error": "proof-v3 attention audit is not enabled"})
        coordinator, oracle_lookup = provider
        try:
            nonce = bytes.fromhex(body.validator_nonce)
        except ValueError:
            return JSONResponse(
                status_code=400, content={"error": "invalid nonce hex"})
        oracle_set = (
            oracle_lookup(body.request_id) if oracle_lookup else None)
        try:
            wire = coordinator.answer_attention_audit_wire(
                request_id=body.request_id, validator_nonce=nonce,
                row_samples=int(body.row_samples),
                chunk_samples=int(body.chunk_samples),
                oracle_set=oracle_set)
        except ProofV3Error as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        return {"wire": wire.hex()}

    @app.post("/proof_v3/audit")
    async def proof_v3_audit(body: ProofV3AuditBody):  # noqa: ANN001
        endpoints = get_endpoints()
        if endpoints is None:
            return JSONResponse(
                status_code=501,
                content={"error": "proof-v3 is not enabled on this miner"})
        try:
            nonce = bytes.fromhex(body.validator_nonce)
        except ValueError:
            return JSONResponse(
                status_code=400, content={"error": "invalid nonce hex"})
        try:
            digest, reveal, material = endpoints.answer_audit(
                request_id=body.request_id,
                validator_nonce=nonce,
                projection=body.projection,
            )
        except ProofV3Error as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        return {
            "capture_chain_digest": digest.hex(),
            "projection": reveal.name,
            "in_dim": reveal.in_dim,
            "chunk_size": reveal.chunk_size,
            "tokens": list(reveal.tokens),
            "out_indices": list(reveal.out_indices),
            "input_rows": {
                str(t): list(row) for t, row in reveal.input_rows.items()},
            "weight_rows": {
                str(o): {
                    "row": list(row),
                    "chunks": {
                        str(ci): {"data": data.hex(),
                                  "path": _encode_path(path)}
                        for ci, (data, path) in openings.items()},
                }
                for o, (row, openings) in reveal.weight_rows.items()},
            "expected_cells": {
                f"{t},{o}": v for (t, o), v in reveal.expected_cells.items()},
            "weight_root": reveal.weight_root.hex(),
        }


def _encode_path(path) -> list:
    """MerklePath -> JSON-safe [(sibling_hex, is_left), ...]."""

    return [[sib.hex(), bool(is_left)] for sib, is_left in path.siblings]
