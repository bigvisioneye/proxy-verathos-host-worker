"""Proxy inference forwarding via Balancer 1."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator, Optional
from urllib.parse import urlencode

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from verallm.api.proxy_auth import PROXY_LLM_HEADER

logger = logging.getLogger(__name__)


class ProxyInferenceState:
    def __init__(self) -> None:
        self.enabled: bool = False
        self.balancer_base: str = ""
        self.balancer_api_key: str = ""
        self.proxy_llm_key: str = ""
        self.model_id: str = ""
        self.quant: str = ""
        self.max_context_len: int = 0
        self.slot_id: str = ""
        self.verify_upstream_ssl: bool = True
        self.mirror_health_url: str = ""
        self.advertised_gpu_uuids: list[str] = []
        self.advertised_hardware: bool = False


proxy_state = ProxyInferenceState()


def _mirror_health_fields() -> tuple[str, ...]:
    return (
        "moe",
        "batch_mode",
        "capture_backend",
        "max_model_len",
        "max_context",
        "active_requests",
        "max_requests",
        "kv_pool_tokens",
        "kv_used_tokens",
        "kv_free_tokens",
        "kv_utilization_pct",
        "can_accept_max_context",
        "proof_pending",
        "proof_max_pending",
    )


def _fetch_upstream_health(url: str) -> Optional[dict]:
    try:
        resp = httpx.get(
            f"{url.rstrip('/')}/health",
            timeout=1.0,
            verify=proxy_state.verify_upstream_ssl,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("proxy health mirror failed for %s: %s", url, exc)
    return None


def _parse_advertised_gpu_uuids(args=None) -> list[str]:
    raw = ""
    if args is not None:
        raw = str(getattr(args, "advertised_gpu_uuids", None) or "").strip()
    if not raw:
        raw = str(os.environ.get("VERATHOS_ADVERTISED_GPU_UUIDS", "") or "").strip()
    if not raw:
        return []
    return [u.strip() for u in raw.split(",") if u.strip()]


def advertised_hardware_dict(args=None) -> dict[str, object]:
    """Return configured audit-GPU hardware for proxy /health (CLI or env)."""
    gpu_name = str(
        getattr(args, "advertised_gpu_name", "")
        or os.environ.get("VERATHOS_ADVERTISED_GPU_NAME", "")
        or ""
    ).strip()
    vram_gb = getattr(args, "advertised_vram_gb", None) if args is not None else None
    if vram_gb is None:
        raw = os.environ.get("VERATHOS_ADVERTISED_VRAM_GB", "")
        vram_gb = int(raw) if str(raw).strip().isdigit() else 0
    else:
        vram_gb = int(vram_gb or 0)

    gpu_count = getattr(args, "advertised_gpu_count", None) if args is not None else None
    if gpu_count is None:
        raw = os.environ.get("VERATHOS_ADVERTISED_GPU_COUNT", "1")
        gpu_count = int(raw) if str(raw).strip().isdigit() else 1
    else:
        gpu_count = int(gpu_count or 1)

    compute_capability = str(
        (
            getattr(args, "advertised_compute_capability", "")
            if args is not None
            else ""
        )
        or os.environ.get("VERATHOS_ADVERTISED_COMPUTE_CAPABILITY", "")
        or ""
    ).strip()

    uuids = _parse_advertised_gpu_uuids(args)
    if not gpu_name or vram_gb <= 0:
        return {}
    return {
        "gpu_name": gpu_name,
        "gpu_count": max(1, int(gpu_count)),
        "vram_gb": int(vram_gb),
        "compute_capability": compute_capability,
        "gpu_uuids": uuids,
    }


def has_advertised_hardware(result: Optional[dict] = None) -> bool:
    if proxy_state.advertised_hardware:
        return True
    if isinstance(result, dict):
        hw = result.get("hardware")
        if isinstance(hw, dict):
            return bool(str(hw.get("gpu_name") or "").strip()) and int(hw.get("vram_gb") or 0) > 0
    return bool(advertised_hardware_dict())


def _configured_proxy_gpu_uuids(result: Optional[dict] = None) -> list[str]:
    if isinstance(result, dict):
        hw = result.get("hardware")
        if isinstance(hw, dict):
            existing = hw.get("gpu_uuids")
            if isinstance(existing, list) and existing:
                return list(existing)
    if proxy_state.advertised_gpu_uuids:
        return list(proxy_state.advertised_gpu_uuids)
    return _parse_advertised_gpu_uuids()


def merge_upstream_health(result: dict) -> dict:
    """Mirror inference load/KV stats from upstream; keep audit GPU in hardware."""
    upstream: Optional[dict] = None

    # Optional pin for debugging; normal proxy miners use balancer /pick only.
    explicit = str(proxy_state.mirror_health_url or "").strip()
    if explicit:
        upstream = _fetch_upstream_health(explicit)

    if upstream is None and proxy_state.balancer_base:
        try:
            pick = _pick_upstream(log_pick=False)
            endpoint = str(pick.get("endpoint") or "").rstrip("/")
            if endpoint:
                upstream = _fetch_upstream_health(endpoint)
        except Exception as exc:
            logger.debug("proxy health mirror via balancer pick failed: %s", exc)

    if not upstream:
        return result

    for key in ("model", *_mirror_health_fields()):
        if key in upstream:
            result[key] = upstream[key]

    # Validators use /health hardware for capacity-audit cohorts and model gate.
    # When audit GPU is advertised (remote capacity audit), never replace it with
    # the inference tier reached via Balancer 1.
    if has_advertised_hardware(result):
        hw = dict(result.get("hardware") or {})
        proxy_uuids = _configured_proxy_gpu_uuids(result)
        if proxy_uuids:
            hw["gpu_uuids"] = proxy_uuids
        if hw:
            result["hardware"] = hw
        return result

    if isinstance(upstream.get("hardware"), dict) and upstream["hardware"]:
        hw = dict(upstream["hardware"])
        proxy_uuids = _configured_proxy_gpu_uuids(result)
        if proxy_uuids:
            hw["gpu_uuids"] = proxy_uuids
        result["hardware"] = hw
    return result


def configure_proxy_from_args(args) -> None:
    balancer = str(getattr(args, "proxy_balancer", "") or os.environ.get("PROXY_BALANCER_URL", "") or "").strip()
    if not balancer and not getattr(args, "proxy_mode", False):
        return
    proxy_state.enabled = True
    proxy_state.balancer_base = balancer.rstrip("/")
    proxy_state.balancer_api_key = str(
        getattr(args, "proxy_balancer_key", "")
        or os.environ.get("PROXY_BALANCER_API_KEY", "")
        or ""
    ).strip()
    proxy_state.proxy_llm_key = str(
        getattr(args, "proxy_llm_key", "")
        or os.environ.get("PROXY_LLM_KEY", "")
        or ""
    ).strip()
    proxy_state.model_id = str(
        getattr(args, "model_id", "")
        or getattr(args, "model", "")
        or ""
    )
    proxy_state.quant = str(getattr(args, "quant", "") or "auto")
    proxy_state.max_context_len = int(getattr(args, "max_model_len", 0) or 0)
    proxy_state.slot_id = str(
        getattr(args, "proxy_slot_id", "")
        or os.environ.get("PROXY_SLOT_ID", "")
        or ""
    ).strip()
    proxy_state.verify_upstream_ssl = str(
        os.environ.get("PROXY_UPSTREAM_VERIFY_SSL", "1")
    ).strip().lower() not in {"0", "false", "no"}
    proxy_state.mirror_health_url = str(
        os.environ.get("PROXY_MIRROR_HEALTH_URL", "") or ""
    ).strip()
    proxy_state.advertised_gpu_uuids = _parse_advertised_gpu_uuids(args)


def apply_advertised_hardware(state, args) -> None:
    """Populate /health hardware from CLI/env on proxy nodes without CUDA."""
    hw = advertised_hardware_dict(args)
    if not hw:
        proxy_state.advertised_hardware = False
        return
    state.gpu_name = str(hw["gpu_name"])
    state.vram_gb = int(hw["vram_gb"])
    state.gpu_count = int(hw["gpu_count"])
    if hw.get("compute_capability"):
        state.compute_capability = str(hw["compute_capability"])
    uuids = hw.get("gpu_uuids") or []
    if isinstance(uuids, list) and uuids:
        state.gpu_uuids = list(uuids)
        proxy_state.advertised_gpu_uuids = list(uuids)
    proxy_state.advertised_hardware = True


def _balancer_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if proxy_state.balancer_api_key:
        headers["Authorization"] = f"Bearer {proxy_state.balancer_api_key}"
    return headers


def _validator_hotkey(request: Optional[Request]) -> str:
    if request is None:
        return ""
    from_request = str(
        getattr(getattr(request, "state", None), "validator_hotkey", "") or ""
    ).strip()
    if from_request:
        return from_request
    return str(request.headers.get("X-Validator-Hotkey", "") or "").strip()


def _pick_upstream(*, log_pick: bool = True) -> dict[str, Any]:
    if not proxy_state.balancer_base:
        raise RuntimeError("proxy balancer URL not configured")
    params = {
        "model_id": proxy_state.model_id,
        "quant": proxy_state.quant,
    }
    if proxy_state.max_context_len > 0:
        params["max_context_len"] = str(proxy_state.max_context_len)
    if proxy_state.slot_id:
        params["slot_id"] = proxy_state.slot_id
    url = f"{proxy_state.balancer_base}/pick?{urlencode(params)}"
    resp = httpx.get(url, headers=_balancer_headers(), timeout=5.0)
    resp.raise_for_status()
    data = resp.json() or {}
    endpoint = str(data.get("endpoint") or "").rstrip("/")
    if not endpoint:
        raise RuntimeError(f"balancer /pick missing endpoint: {data}")
    if log_pick:
        logger.info(
            "proxy balancer pick ok: upstream=%s worker_id=%s",
            endpoint,
            str(data.get("worker_id") or data.get("slot_id") or ""),
        )
    return data


def _upstream_headers(pick: dict[str, Any], request: Optional[Request]) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = str(pick.get("api_key") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if proxy_state.proxy_llm_key:
        headers[PROXY_LLM_HEADER] = proxy_state.proxy_llm_key
    if request is not None:
        for name in (
            "X-Request-Id",
            "X-Validator-Hotkey",
            "X-Validator-Signature",
            "X-Validator-Timestamp",
        ):
            value = request.headers.get(name)
            if value:
                headers[name] = value
    return headers


async def _stream_upstream_response(resp: httpx.Response) -> AsyncIterator[bytes]:
    async for chunk in resp.aiter_bytes():
        yield chunk


async def proxy_json_post(
    path: str,
    body: dict[str, Any],
    request: Optional[Request] = None,
) -> Any:
    pick = _pick_upstream()
    endpoint = str(pick["endpoint"]).rstrip("/")
    url = f"{endpoint}{path}"
    validator_hotkey = _validator_hotkey(request)
    hotkey_suffix = f" validator={validator_hotkey[:12]}..." if validator_hotkey else ""
    logger.info("proxy forwarding %s -> %s%s", path, url, hotkey_suffix)
    client = httpx.AsyncClient(verify=proxy_state.verify_upstream_ssl, timeout=None)
    req = client.build_request(
        "POST",
        url,
        headers=_upstream_headers(pick, request),
        json=body,
    )
    resp = await client.send(req, stream=True)
    logger.info(
        "proxy upstream response: path=%s status=%s upstream=%s%s",
        path,
        resp.status_code,
        endpoint,
        hotkey_suffix,
    )
    if resp.status_code >= 400:
        try:
            raw = await resp.aread()
            try:
                content = json.loads(raw.decode()) if raw else {"error": resp.reason_phrase}
            except json.JSONDecodeError:
                content = {"error": raw.decode(errors="replace")}
            return JSONResponse(status_code=resp.status_code, content=content)
        finally:
            await resp.aclose()
            await client.aclose()

    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        async def sse_iter():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            except httpx.StreamClosed:
                logger.debug(
                    "proxy upstream SSE closed: path=%s upstream=%s%s",
                    path,
                    endpoint,
                    hotkey_suffix,
                )
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(
            sse_iter(),
            status_code=resp.status_code,
            media_type=content_type,
            headers={
                k: v
                for k, v in resp.headers.items()
                if k.lower() in {"cache-control", "x-accel-buffering"}
            },
        )

    try:
        raw = await resp.aread()
        if raw:
            try:
                return json.loads(raw.decode())
            except json.JSONDecodeError:
                return JSONResponse(status_code=resp.status_code, content={"raw": raw.decode(errors="replace")})
        return JSONResponse(status_code=resp.status_code, content={})
    finally:
        await resp.aclose()
        await client.aclose()


def proxy_startup_minimal(state, args) -> None:
    """Initialize proxy-only server state without loading vLLM."""
    configure_proxy_from_args(args)
    apply_advertised_hardware(state, args)
    state.model_name = str(getattr(args, "model_id", "") or getattr(args, "model", "") or "")
    if getattr(args, "evm_address", None):
        state.evm_address = args.evm_address
    if getattr(args, "evm_private_key", None):
        state.evm_private_key = args.evm_private_key
    state.capacity_audit_state_file = str(getattr(args, "capacity_audit_state_file", "") or "")
