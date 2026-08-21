"""FastAPI service for remote capacity-audit workloads (side 3).

Endpoints (bearer via X-Verathos-Audit-Key):
  POST /capacity-audit/v1/prepare              -> {job_id, phase}   (launch + warm)
  POST /capacity-audit/v1/jobs/{id}/start      -> {ok}              (deliver B_start seed)
  GET  /capacity-audit/v1/jobs/{id}            -> job status
  POST /capacity-audit/v1/jobs/{id}/challenge  -> {ok}              (B_proof seed)
  DELETE /capacity-audit/v1/jobs/{id}          -> {ok}
  GET  /health
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from neurons.capacity_audit_balancer import CapacityAuditBalancerClient
from services.audit_worker.runner import AuditJobRunner
from verallm.api.proxy_auth import verify_audit_worker_request

logger = logging.getLogger(__name__)

app = FastAPI(title="Verathos Capacity-Audit Worker", version="0.1.0")
runner = AuditJobRunner()

_worker_id = ""
_worker_endpoint = ""
_worker_key = ""
_gpu_class = ""
_balancer: Optional[CapacityAuditBalancerClient] = None
_hb_stop = threading.Event()
_hb_thread: Optional[threading.Thread] = None


class PrepareBody(BaseModel):
    lease_id: str
    audit_id: str = ""
    workload_spec: dict[str, Any] = Field(default_factory=dict)
    challenge_timeout_s: float = 120.0
    start_timeout_s: float = 120.0
    gpu_class: str = ""
    job_id: str = ""


class StartBody(BaseModel):
    proof_seed: str
    audit_id: str = ""
    b_start: int = 0


class ChallengeBody(BaseModel):
    challenge_seed: str


def _detect_gpu_class() -> str:
    override = str(os.environ.get("VERATHOS_AUDIT_GPU_CLASS", "") or "").strip()
    if override:
        return override
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            return str(torch.cuda.get_device_properties(0).name)
    except Exception:
        pass
    return "unknown"


def _detect_hardware() -> dict[str, Any]:
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            return {}
        props = torch.cuda.get_device_properties(0)
        cc = torch.cuda.get_device_capability(0)
        from verallm.registry.gpu import detect_vram_gb  # noqa: PLC0415
        uuids = []
        for i in range(torch.cuda.device_count()):
            try:
                uuids.append(str(torch.cuda.get_device_properties(i).uuid))
            except Exception:
                pass
        return {
            "gpu_name": props.name,
            "gpu_count": torch.cuda.device_count(),
            "vram_gb": int(detect_vram_gb()),
            "compute_capability": f"{cc[0]}.{cc[1]}",
            "gpu_uuids": uuids,
        }
    except Exception:
        return {}


def _require_key(request: Request) -> None:
    if not verify_audit_worker_request(request, _worker_key):
        raise HTTPException(status_code=401, detail="invalid audit worker key")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "capacity_audit_worker",
        "worker_id": _worker_id,
        "gpu_class": _gpu_class or _detect_gpu_class(),
        "active_jobs": runner.active_job_count(),
        "hardware": _detect_hardware() or None,
    }


@app.post("/capacity-audit/v1/prepare")
async def prepare(body: PrepareBody, request: Request):
    _require_key(request)
    if runner.active_job_count() >= 1:
        raise HTTPException(status_code=503, detail="audit worker busy")
    try:
        job_id = runner.prepare(body.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"job_id": job_id, "phase": "preparing"}


@app.post("/capacity-audit/v1/jobs/{job_id}/start")
async def start(job_id: str, body: StartBody, request: Request):
    _require_key(request)
    try:
        runner.start(job_id, proof_seed=body.proof_seed, audit_id=body.audit_id, b_start=body.b_start)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    return {"ok": True}


@app.get("/capacity-audit/v1/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    _require_key(request)
    record = runner.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")
    with record.lock:
        return {
            "job_id": record.job_id,
            "audit_id": record.audit_id,
            "lease_id": record.lease_id,
            "phase": record.phase,
            "pass0_root": record.pass0_root,
            "final_timing": record.final_timing,
            "final_summary": record.final_summary,
            "error": record.error,
        }


@app.post("/capacity-audit/v1/jobs/{job_id}/challenge")
async def challenge(job_id: str, body: ChallengeBody, request: Request):
    _require_key(request)
    try:
        runner.submit_challenge(job_id, body.challenge_seed)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.delete("/capacity-audit/v1/jobs/{job_id}")
async def cancel(job_id: str, request: Request):
    _require_key(request)
    runner.cancel(job_id)
    return {"ok": True}


def _heartbeat_loop() -> None:
    if _balancer is None or not _worker_id:
        return
    while not _hb_stop.wait(15.0):
        try:
            _balancer.heartbeat_worker({
                "worker_id": _worker_id,
                "endpoint": _worker_endpoint,
                "worker_key": _worker_key,
                "gpu_class": _gpu_class,
                "active_jobs": runner.active_job_count(),
            })
        except Exception as exc:
            logger.debug("audit worker heartbeat failed: %s", exc)


def _register_with_balancer() -> None:
    global _balancer
    url = str(os.environ.get("CAPACITY_AUDIT_BALANCER_URL", "") or "").strip()
    if not url:
        logger.info("audit worker: CAPACITY_AUDIT_BALANCER_URL not set — running standalone")
        return
    api_key = str(os.environ.get("CAPACITY_AUDIT_BALANCER_API_KEY", "") or "").strip()
    _balancer = CapacityAuditBalancerClient(url, api_key=api_key)
    try:
        _balancer.register_worker({
            "worker_id": _worker_id,
            "endpoint": _worker_endpoint,
            "worker_key": _worker_key,
            "gpu_class": _gpu_class,
            "lease_ttl_s": 60,
        })
    except Exception:
        logger.warning("audit worker: initial balancer register failed; heartbeats will retry")


def configure_worker(*, worker_id: str, endpoint: str, worker_key: str, gpu_class: str = "") -> None:
    global _worker_id, _worker_endpoint, _worker_key, _gpu_class, _hb_thread
    _worker_id = worker_id
    _worker_endpoint = endpoint.rstrip("/")
    _worker_key = worker_key
    _gpu_class = gpu_class or _detect_gpu_class()
    _register_with_balancer()
    if _balancer is not None:
        _hb_stop.clear()
        _hb_thread = threading.Thread(target=_heartbeat_loop, name="audit-worker-heartbeat", daemon=True)
        _hb_thread.start()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verathos capacity-audit worker")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8095)
    p.add_argument("--worker-id", default=None)
    p.add_argument("--public-endpoint", default=None, help="URL the proxy/balancer use to reach this worker")
    p.add_argument("--worker-key", default=None, help="Shared secret (or VERATHOS_AUDIT_WORKER_KEY)")
    p.add_argument("--gpu-class", default=None, help="Exact calibrated GPU class string for balancer routing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    worker_key = str(args.worker_key or os.environ.get("VERATHOS_AUDIT_WORKER_KEY", "") or "").strip()
    if not worker_key:
        raise SystemExit("audit worker requires --worker-key or VERATHOS_AUDIT_WORKER_KEY")
    worker_id = str(args.worker_id or os.environ.get("VERATHOS_AUDIT_WORKER_ID", "") or "").strip() \
        or f"audit-worker-{args.host}:{args.port}"
    endpoint = str(args.public_endpoint or os.environ.get("VERATHOS_AUDIT_WORKER_ENDPOINT", "")
                   or f"http://{args.host}:{args.port}").strip()
    configure_worker(worker_id=worker_id, endpoint=endpoint, worker_key=worker_key, gpu_class=str(args.gpu_class or ""))
    logger.info("audit worker ready: worker_id=%s endpoint=%s gpu_class=%s",
                worker_id, endpoint, str(args.gpu_class or _gpu_class or ""))
    uvicorn.run(app, host=args.host, port=args.port, access_log=False, log_level="info")


if __name__ == "__main__":
    main()
