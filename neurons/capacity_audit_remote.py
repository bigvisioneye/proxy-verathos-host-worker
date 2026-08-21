"""HTTP client for a remote capacity-audit GPU worker (side 3).

Mirrors the local subprocess lifecycle in neurons/capacity_audit_miner.py, but
over HTTP so the timed benchmark runs on a leased GPU worker instead of the
proxy's own GPU. The HOT-START is preserved: `prepare` warms the worker's GPU
during the B_select->B_start lead window; `start` (with the B_start-derived
seed) begins the timed measurement. This is the piece the earlier remote
implementation lacked, which made every run cold and fail the class threshold.

Worker phases (from GET /jobs/{id}): preparing -> ready -> pass0_ready ->
final_ready -> proof_ready (or failed).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from verallm.api.proxy_auth import AUDIT_WORKER_HEADER


def audit_worker_verify_ssl() -> bool:
    return str(os.environ.get("PROXY_UPSTREAM_VERIFY_SSL", "1")).strip().lower() not in {
        "0", "false", "no", "off",
    }


@dataclass
class RemoteAuditJobStatus:
    job_id: str
    phase: str
    pass0_root: str = ""
    final_timing: dict = field(default_factory=dict)
    final_summary: dict = field(default_factory=dict)
    error: str = ""


class RemoteAuditClient:
    def __init__(self, endpoint: str, worker_key: str, *, timeout_s: float = 30.0):
        self.endpoint = str(endpoint or "").rstrip("/")
        self.worker_key = str(worker_key or "").strip()
        self.timeout_s = max(1.0, float(timeout_s))
        self.verify_ssl = audit_worker_verify_ssl()

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            AUDIT_WORKER_HEADER: self.worker_key,
        }

    # ── lifecycle ────────────────────────────────────────────────

    def prepare(self, payload: dict[str, Any]) -> str:
        """Launch + warm the benchmark. Returns job_id. Seed is NOT sent yet."""
        resp = httpx.post(
            f"{self.endpoint}/capacity-audit/v1/prepare",
            headers=self._headers(), json=payload,
            timeout=self.timeout_s, verify=self.verify_ssl,
        )
        resp.raise_for_status()
        job_id = str((resp.json() or {}).get("job_id") or "").strip()
        if not job_id:
            raise RuntimeError(f"audit worker /prepare missing job_id: {resp.text[:200]}")
        return job_id

    def start(self, job_id: str, *, proof_seed: str, audit_id: str, b_start: int) -> None:
        """Release the warmed benchmark into the timed run (delivers the seed)."""
        resp = httpx.post(
            f"{self.endpoint}/capacity-audit/v1/jobs/{job_id}/start",
            headers=self._headers(),
            json={"proof_seed": proof_seed, "audit_id": audit_id, "b_start": int(b_start)},
            timeout=self.timeout_s, verify=self.verify_ssl,
        )
        resp.raise_for_status()

    def submit_challenge(self, job_id: str, challenge_seed: str) -> None:
        resp = httpx.post(
            f"{self.endpoint}/capacity-audit/v1/jobs/{job_id}/challenge",
            headers=self._headers(), json={"challenge_seed": challenge_seed},
            timeout=self.timeout_s, verify=self.verify_ssl,
        )
        resp.raise_for_status()

    def get_job(self, job_id: str) -> RemoteAuditJobStatus:
        resp = httpx.get(
            f"{self.endpoint}/capacity-audit/v1/jobs/{job_id}",
            headers=self._headers(), timeout=self.timeout_s, verify=self.verify_ssl,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        return RemoteAuditJobStatus(
            job_id=str(data.get("job_id") or job_id),
            phase=str(data.get("phase") or ""),
            pass0_root=str(data.get("pass0_root") or ""),
            final_timing=data.get("final_timing") if isinstance(data.get("final_timing"), dict) else {},
            final_summary=data.get("final_summary") if isinstance(data.get("final_summary"), dict) else {},
            error=str(data.get("error") or ""),
        )

    def cancel(self, job_id: str) -> None:
        try:
            httpx.delete(
                f"{self.endpoint}/capacity-audit/v1/jobs/{job_id}",
                headers=self._headers(), timeout=self.timeout_s, verify=self.verify_ssl,
            )
        except Exception:
            pass

    # ── polling helpers ──────────────────────────────────────────

    _PHASE_ORDER = {
        "preparing": 0, "ready": 1, "pass0_ready": 2,
        "final_ready": 3, "proof_ready": 4,
    }

    def wait_for_phase(self, job_id: str, phase: str, *, timeout_s: float,
                       poll_s: float = 0.1) -> RemoteAuditJobStatus:
        """Wait until the job reaches (or passes) `phase`. Raises on failure/timeout."""
        want = self._PHASE_ORDER.get(phase, 99)
        deadline = time.time() + max(1.0, float(timeout_s))
        last = RemoteAuditJobStatus(job_id=job_id, phase="")
        while time.time() < deadline:
            try:
                last = self.get_job(job_id)
            except Exception:
                time.sleep(max(0.05, poll_s))
                continue
            if last.phase == "failed":
                raise RuntimeError(last.error or "remote audit job failed")
            if self._PHASE_ORDER.get(last.phase, -1) >= want:
                return last
            time.sleep(max(0.05, poll_s))
        raise TimeoutError(f"remote audit job {job_id} timed out waiting for {phase} (phase={last.phase})")
