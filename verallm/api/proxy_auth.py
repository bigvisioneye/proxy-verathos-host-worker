"""Shared auth helpers for proxy ↔ inference GPU traffic."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import Request

PROXY_LLM_HEADER = "X-Verathos-Proxy-Key"
AUDIT_WORKER_HEADER = "X-Verathos-Audit-Key"


def proxy_llm_key_from_env() -> str:
    return str(os.environ.get("VERATHOS_PROXY_LLM_KEY", "") or "").strip()


def verify_proxy_llm_request(request: Request, expected_key: Optional[str] = None) -> bool:
    """Return True when the request carries the configured proxy LLM key."""
    key = str(expected_key or proxy_llm_key_from_env() or "").strip()
    if not key:
        return True
    provided = str(request.headers.get(PROXY_LLM_HEADER, "") or "").strip()
    return bool(provided) and provided == key


def audit_worker_key_from_env() -> str:
    return str(os.environ.get("VERATHOS_AUDIT_WORKER_KEY", "") or "").strip()


def verify_audit_worker_request(request: Request, expected_key: Optional[str] = None) -> bool:
    """Return True when the request carries the configured audit-worker key.

    Unlike the proxy-LLM check this is FAIL-CLOSED: an unset key rejects every
    request. The audit worker runs arbitrary GPU work on behalf of a miner, so
    an unconfigured worker must serve nobody rather than everybody.
    """
    key = str(expected_key or audit_worker_key_from_env() or "").strip()
    if not key:
        return False
    provided = str(request.headers.get(AUDIT_WORKER_HEADER, "") or "").strip()
    return bool(provided) and provided == key
