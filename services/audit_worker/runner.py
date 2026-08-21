"""Subprocess runner for hot-capacity audit workloads on a remote GPU worker.

Preserves the miner's HOT-START: `prepare()` launches bench_combined.py with
--start-file/--ready-file so it compiles kernels and warms the GPU, then blocks;
`start()` writes the start-file (with the B_start-derived seed) to begin the
timed run. This mirrors neurons/capacity_audit_miner.py exactly so the timing
matches what the validator expects.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Live miners emit v2 (proof_protocol_version 2). v1 kept for older validators.
# Source of truth: neurons/capacity_audit_combined.py COMBINED_PROOF_FORMAT*.
_COMBINED_PROOF_FORMATS = frozenset({
    "hot_capacity_combined_proof_v1",
    "hot_capacity_combined_proof_v2",
})

# Mirror the miner: always request the strict arithmetic (v2) payload from the
# workload.  bench_combined defaults to v1 when the flag is absent, and
# validators reject v1 outside the bounded compatibility window with
# `legacy_capacity_proof_protocol_not_accepted`.  The guarded import keeps this
# module light on hosts whose venv lacks the zkllm chain the constant pulls in.
try:  # pragma: no cover - trivial constant import
    from neurons.capacity_audit_combined import (
        CURRENT_COMBINED_PROOF_PROTOCOL_VERSION as _PROOF_PROTOCOL_VERSION,
    )
except Exception:  # noqa: BLE001
    _PROOF_PROTOCOL_VERSION = 2


def _root_hex(raw: object) -> str:
    if isinstance(raw, str):
        text = raw.strip()
        body = text[2:] if text.startswith("0x") else text
        if len(body) == 64:
            try:
                bytes.fromhex(body)
                return body
            except ValueError:
                pass
    if isinstance(raw, list) and raw:
        try:
            return bytes(raw).hex()
        except Exception:
            pass
    return str(raw or "").strip()


def _proof_summary_ready(final_summary: object) -> bool:
    if not isinstance(final_summary, dict):
        return False
    proof_payload = final_summary.get("proof_payload")
    return isinstance(proof_payload, dict) and str(proof_payload.get("format") or "") in _COMBINED_PROOF_FORMATS


def _workspace_script() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "hot_capacity_workspace" / "bench_combined.py"


def _workspace_command(script: Path) -> list[str]:
    if script.exists():
        return [sys.executable, str(script)]
    return [sys.executable, "-c", "from hot_capacity_workspace.bench_combined import main; main()"]


def _workspace_env(script_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    if script_dir.exists():
        cur = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{script_dir}:{cur}" if cur else str(script_dir)
    try:
        import torch  # noqa: PLC0415
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.exists():
            cur = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{torch_lib}:{cur}" if cur else str(torch_lib)
    except Exception:
        pass
    env.setdefault("MAX_JOBS", "2")
    return env


@dataclass
class AuditJobRecord:
    job_id: str
    lease_id: str
    audit_id: str
    out_dir: Path
    challenge_file: Path
    start_file: Path
    ready_file: Path
    proc: Optional[subprocess.Popen] = None
    phase: str = "preparing"          # preparing -> ready -> pass0_ready -> final_ready -> proof_ready | failed
    started: bool = False
    error: str = ""
    pass0_root: str = ""
    final_timing: dict = field(default_factory=dict)
    final_summary: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    monitor: Optional[threading.Thread] = None


class AuditJobRunner:
    def __init__(self) -> None:
        self._jobs: dict[str, AuditJobRecord] = {}
        self._lock = threading.Lock()

    def active_job_count(self) -> int:
        with self._lock:
            return sum(
                1 for r in self._jobs.values()
                if r.proc is not None and r.proc.poll() is None
            )

    def get_job(self, job_id: str) -> Optional[AuditJobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def _build_cmd(self, record: AuditJobRecord, payload: dict[str, Any]) -> list[str]:
        script = _workspace_script()
        challenge_timeout_s = float(payload.get("challenge_timeout_s") or 120.0)
        start_timeout_s = float(payload.get("start_timeout_s") or 120.0)
        cmd = [
            *_workspace_command(script),
            "--child",
            "--out-dir", str(record.out_dir),
            "--lease-id", record.lease_id,
            "--gpu-index", "0",
            "--challenge-file", str(record.challenge_file),
            "--challenge-timeout-s", str(challenge_timeout_s),
            # HOT-START: warm up, then block on the start-file (seed delivered at /start).
            "--start-file", str(record.start_file),
            "--start-timeout-s", str(max(1.0, start_timeout_s)),
            "--ready-file", str(record.ready_file),
        ]
        spec = payload.get("workload_spec") if isinstance(payload.get("workload_spec"), dict) else {}
        for key, value in spec.items():
            if key in {"workload_version", "pass_count", "proof_protocol_version"}:
                continue
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        # This worker version always emits the strict arithmetic payload; see
        # the module-level note on _PROOF_PROTOCOL_VERSION.
        cmd.extend(["--proof-protocol-version", str(_PROOF_PROTOCOL_VERSION)])
        return cmd

    def prepare(self, payload: dict[str, Any]) -> str:
        """Launch + warm the benchmark; it blocks on the start-file until start()."""
        lease_id = str(payload.get("lease_id") or "").strip()
        if not lease_id:
            raise ValueError("lease_id is required")
        job_id = str(payload.get("job_id") or uuid.uuid4())
        out_dir = Path(tempfile.mkdtemp(prefix="verathos_audit_worker_"))
        record = AuditJobRecord(
            job_id=job_id,
            lease_id=lease_id,
            audit_id=str(payload.get("audit_id") or ""),
            out_dir=out_dir,
            challenge_file=out_dir / f"{lease_id}_challenge.txt",
            start_file=out_dir / f"{lease_id}_start.json",
            ready_file=out_dir / f"{lease_id}_ready.json",
        )
        cmd = self._build_cmd(record, payload)
        env = _workspace_env(_workspace_script().parent)
        try:
            record.proc = subprocess.Popen(
                cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
        except Exception as exc:
            raise RuntimeError(f"failed to launch audit workload: {exc}") from exc
        record.monitor = threading.Thread(
            target=self._monitor, args=(record,), name=f"audit-job-{job_id[:8]}", daemon=True,
        )
        record.monitor.start()
        with self._lock:
            self._jobs[job_id] = record
        logger.info(
            "audit job prepared (warming): job_id=%s lease=%s audit_id=%s out_dir=%s",
            job_id[:12], lease_id[:12], record.audit_id[:12], out_dir,
        )
        return job_id

    def start(self, job_id: str, *, proof_seed: str, audit_id: str, b_start: int) -> None:
        """Deliver the B_start seed via the start-file → timed run begins."""
        record = self.get_job(job_id)
        if record is None:
            raise KeyError(job_id)
        payload = {"seed_hex": str(proof_seed or ""), "audit_id": audit_id, "B_start": int(b_start), "t": time.time()}
        tmp = record.start_file.with_suffix(record.start_file.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True))
        os.replace(tmp, record.start_file)
        with record.lock:
            record.started = True
        logger.info("audit job started (timed run): job_id=%s lease=%s", job_id[:12], record.lease_id[:12])

    def submit_challenge(self, job_id: str, challenge_seed: str) -> None:
        record = self.get_job(job_id)
        if record is None:
            raise KeyError(job_id)
        seed = str(challenge_seed or "").strip()
        if not seed:
            raise ValueError("challenge_seed required")
        tmp = record.challenge_file.with_suffix(record.challenge_file.suffix + ".tmp")
        tmp.write_text(seed)
        os.replace(tmp, record.challenge_file)
        logger.info("audit job challenge submitted: job_id=%s len=%d", job_id[:12], len(seed))

    def cancel(self, job_id: str) -> None:
        record = self.get_job(job_id)
        if record is None:
            return
        proc = record.proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
            except Exception:
                pass
        with record.lock:
            if record.phase not in {"proof_ready", "final_ready"}:
                record.phase = "cancelled"
        logger.info("audit job cancelled: job_id=%s", job_id[:12])

    def _monitor(self, record: AuditJobRecord) -> None:
        proc = record.proc
        if proc is None:
            with record.lock:
                record.phase, record.error = "failed", "process not started"
            return
        lease = record.lease_id
        pass0_path = record.out_dir / f"{lease}_pass0.json"
        final_path = record.out_dir / f"{lease}_final_timing.json"
        summary_path = record.out_dir / f"{lease}_final.json"

        while proc.poll() is None:
            with record.lock:
                if record.phase == "preparing" and record.ready_file.exists():
                    record.phase = "ready"
                if not record.pass0_root and pass0_path.exists():
                    try:
                        root = _root_hex(json.loads(pass0_path.read_text()).get("root"))
                        if root:
                            record.pass0_root = root
                            if self._phase_rank(record.phase) < self._phase_rank("pass0_ready"):
                                record.phase = "pass0_ready"
                    except Exception:
                        pass
                if not record.final_timing and final_path.exists():
                    try:
                        data = json.loads(final_path.read_text())
                        if isinstance(data, dict):
                            record.final_timing = data
                            if self._phase_rank(record.phase) < self._phase_rank("final_ready"):
                                record.phase = "final_ready"
                    except Exception:
                        pass
                if not _proof_summary_ready(record.final_summary) and summary_path.exists():
                    try:
                        data = json.loads(summary_path.read_text())
                        if _proof_summary_ready(data):
                            record.final_summary = data
                            record.phase = "proof_ready"
                    except Exception:
                        pass
                if _proof_summary_ready(record.final_summary):
                    break
            time.sleep(0.02)

        rc = proc.poll()
        with record.lock:
            # final read after exit
            if summary_path.exists() and not _proof_summary_ready(record.final_summary):
                try:
                    data = json.loads(summary_path.read_text())
                    if _proof_summary_ready(data):
                        record.final_summary, record.phase = data, "proof_ready"
                except Exception:
                    pass
            if not _proof_summary_ready(record.final_summary) and record.phase not in {"proof_ready", "cancelled"}:
                stderr = ""
                try:
                    _, stderr = proc.communicate(timeout=1)
                except Exception:
                    pass
                record.phase = "failed"
                record.error = f"workload exited rc={rc} stderr_tail={str(stderr)[-400:]}"
                logger.warning("audit job failed: job_id=%s rc=%s err=%s",
                               record.job_id[:12], rc, record.error[:200])

    @staticmethod
    def _phase_rank(phase: str) -> int:
        return {"preparing": 0, "ready": 1, "pass0_ready": 2, "final_ready": 3, "proof_ready": 4}.get(phase, -1)
