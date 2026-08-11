"""Proxy inference forwarding via Balancer 1."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from typing import Any, AsyncIterator, Optional
from urllib.parse import urlencode

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from verallm.api.proxy_auth import PROXY_LLM_HEADER

logger = logging.getLogger(__name__)

# Carries this server's hard-auditor decision to the inference upstream.
# The upstream is not a registered miner, so it has no validator allowlist and
# no proof-v3 hard-auditor policy of its own; without this it refuses every
# hard opening with 403.
PROXY_HARD_AUDITOR_HEADER = "X-Validator-Hard-Auditor"

# Seconds to reuse a mirrored upstream /health snapshot. Validators poll the
# proxy's /health often; without this, every poll cost a balancer /pick plus an
# upstream /health, which burned the upstream's public rate-limit budget.
HEALTH_CACHE_TTL_S = float(os.environ.get("PROXY_HEALTH_CACHE_TTL_S", "10") or 10)

# Upstream /health timeout. The proxy and the inference pool can sit in far-apart
# regions (~300ms RTT observed), so a new connection costs ~2 RTT before any
# response. The old 1s budget timed out constantly and mirroring never worked.
# This call is async and cached, so a generous timeout costs nothing.
HEALTH_FETCH_TIMEOUT_S = float(os.environ.get("PROXY_HEALTH_FETCH_TIMEOUT_S", "5") or 5)

# Pooled async client for the short auxiliary calls (balancer /pick, upstream
# /health). These used to be synchronous httpx.get() calls issued from inside
# async request handlers, which blocked the event loop — stalling every in-flight
# SSE relay and truncating them.
_aux_client: Optional[httpx.AsyncClient] = None

_health_cache_at: float = 0.0
_health_cache_data: Optional[dict] = None
_health_cache_lock: Optional[asyncio.Lock] = None

# --- Local load balancer -------------------------------------------------
# The central balancer /pick sits on one VPS, so every forwarded request paid a
# WAN round trip just to learn which upstream to use (measured: 46ms from EU,
# 590ms from Vietnam). httpx's keepalive_expiry is 5s but canaries arrive ~215s
# apart, so 98% of those picks also paid a cold TCP handshake.
#
# With the local balancer the proxy keeps its own view: the endpoint registry is
# refreshed from the central monitor in the background (off the request path),
# per-upstream load comes from each upstream's own /health, and selection happens
# in-process. Set PROXY_LOCAL_BALANCER=0 to fall back to the central /pick.
LOCAL_LB_ENABLED = str(os.environ.get("PROXY_LOCAL_BALANCER", "1")).strip() not in ("0", "false", "no", "")
# How often to re-read the upstream list from the central monitor.
REGISTRY_REFRESH_S = float(os.environ.get("PROXY_LB_REGISTRY_S", "60") or 60)
# How often to re-probe each upstream's /health for load + RTT.
HEALTH_PROBE_S = float(os.environ.get("PROXY_LB_PROBE_S", "5") or 5)
# An upstream whose last successful probe is older than this is not selectable.
STALE_AFTER_S = float(os.environ.get("PROXY_LB_STALE_S", "45") or 45)
# How long a validator stays pinned to the upstream that served its /chat, so the
# stateful proof-v3 handshake (chat -> challenge -> retention) reaches the same
# server that holds the precommit. Without this, a second LLM upstream makes the
# hard challenge round-robin to a server with no pending challenge -> 409 -> the
# proof fails -> probation. Sized to the validator's 300s hard-canary budget.
AFFINITY_TTL_S = float(os.environ.get("PROXY_AFFINITY_TTL_S", "300") or 300)
# RTT weighting: ms of latency treated as equivalent to one queued request. Keeps
# a near-but-busy server from always losing to a far-but-idle one.
RTT_MS_PER_SLOT = float(os.environ.get("PROXY_LB_RTT_PER_SLOT", "120") or 120)

_registry: dict[str, dict] = {}
_registry_at: float = 0.0
_lb_tasks_started: bool = False
_last_pick_api_key: str = ""
_lb_rr: int = 0

_upstream_client: Optional[httpx.AsyncClient] = None


def _aux() -> httpx.AsyncClient:
    global _aux_client
    if _aux_client is None or _aux_client.is_closed:
        _aux_client = httpx.AsyncClient(
            verify=proxy_state.verify_upstream_ssl,
            timeout=10.0,
            limits=httpx.Limits(
                max_keepalive_connections=8,
                max_connections=32,
                keepalive_expiry=300.0,
            ),
        )
    return _aux_client


def _upstream() -> httpx.AsyncClient:
    """Pooled client for forwarded inference requests.

    This used to be constructed per request, which meant a fresh TCP handshake to
    the inference GPU on every call (27ms same-continent, ~200ms cross-ocean).
    The pool is process-lived — callers must close the *response*, never this
    client.
    """
    global _upstream_client
    if _upstream_client is None or _upstream_client.is_closed:
        _upstream_client = httpx.AsyncClient(
            verify=proxy_state.verify_upstream_ssl,
            timeout=None,
            limits=httpx.Limits(
                max_keepalive_connections=32,
                max_connections=256,
                keepalive_expiry=300.0,
            ),
        )
    return _upstream_client


def _health_lock() -> asyncio.Lock:
    global _health_cache_lock
    if _health_cache_lock is None:
        _health_cache_lock = asyncio.Lock()
    return _health_cache_lock


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
        # A proxy proves nothing itself: the upstream inference server holds the
        # authenticated release and answers the hard proof. Advertising the
        # locally-computed set would report v1 (no local runtime), which the
        # owner allowlist now intersects to empty -- so validators would see a
        # miner that supports no protocol at all.
        "proof_protocol_versions",
    )


async def _fetch_upstream_health(url: str) -> Optional[dict]:
    """Fetch upstream /health, authenticated as the proxy.

    /health is a *public* endpoint on the inference server and is rate-limited
    per client IP (60/min). Behind Docker NAT every proxy shares a single source
    IP, so unauthenticated mirroring trips that limit and gets 429s. Sending the
    proxy key takes the trusted-proxy path and skips the public limiter.
    """
    try:
        headers: dict[str, str] = {}
        if proxy_state.proxy_llm_key:
            headers[PROXY_LLM_HEADER] = proxy_state.proxy_llm_key
        resp = await _aux().get(
            f"{url.rstrip('/')}/health",
            timeout=HEALTH_FETCH_TIMEOUT_S,
            headers=headers,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data if isinstance(data, dict) else {}
        if resp.status_code == 429:
            logger.warning("proxy health mirror rate-limited by upstream %s (429)", url)
    except Exception as exc:
        logger.debug("proxy health mirror failed for %s: %s", url, exc)
    return None


async def _upstream_health_cached() -> Optional[dict]:
    """Return a recent upstream /health snapshot, refreshing at most every TTL.

    Both hits and misses are cached so a failing upstream can't be hammered.
    """
    global _health_cache_at, _health_cache_data

    now = time.monotonic()
    if now - _health_cache_at < HEALTH_CACHE_TTL_S:
        return _health_cache_data

    async with _health_lock():
        now = time.monotonic()
        if now - _health_cache_at < HEALTH_CACHE_TTL_S:
            return _health_cache_data

        data: Optional[dict] = None
        # Optional pin for debugging; normal proxy miners use balancer /pick only.
        explicit = str(proxy_state.mirror_health_url or "").strip()
        if explicit:
            data = await _fetch_upstream_health(explicit)

        if data is None and proxy_state.balancer_base:
            try:
                pick = await _pick_upstream(log_pick=False)
                endpoint = str(pick.get("endpoint") or "").rstrip("/")
                if endpoint:
                    data = await _fetch_upstream_health(endpoint)
            except Exception as exc:
                logger.debug("proxy health mirror via balancer pick failed: %s", exc)

        _health_cache_data = data
        _health_cache_at = time.monotonic()
        return data


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


async def merge_upstream_health(result: dict) -> dict:
    """Mirror inference load/KV stats from upstream; keep audit GPU in hardware."""
    upstream = await _upstream_health_cached()

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


async def _refresh_registry() -> None:
    """Re-read the upstream list from the central monitor.

    The monitor stays the source of truth for *which* GPUs exist; only the
    per-request selection moves local. Runs off the request path, so its latency
    never reaches a validator.
    """
    global _registry, _registry_at
    if not proxy_state.balancer_base:
        return
    url = f"{proxy_state.balancer_base}/gpus-dashboard"
    resp = await _aux().get(url, headers=_balancer_headers(), timeout=10.0)
    resp.raise_for_status()
    data = resp.json() or {}
    rows = data.get("rows") or data.get("gpus") or []
    seen: set[str] = set()
    for row in rows:
        endpoint = str(row.get("endpoint") or "").rstrip("/")
        if not endpoint:
            continue
        seen.add(endpoint)
        entry = _registry.setdefault(endpoint, {})
        entry["endpoint"] = endpoint
        entry.setdefault("rtt_ms", 0.0)
        entry.setdefault("active", 0)
        entry.setdefault("ok", False)
        entry.setdefault("last_ok", 0.0)
    for endpoint in list(_registry):
        if endpoint not in seen:
            _registry.pop(endpoint, None)
    _registry_at = time.monotonic()


async def _probe_upstream(endpoint: str) -> None:
    """Probe one upstream's /health for load, model identity and RTT."""
    entry = _registry.get(endpoint)
    if entry is None:
        return
    started = time.monotonic()
    try:
        # Send the proxy key: /health is public but rate-limited per source IP
        # (60/min), and behind Docker NAT every proxy shares one IP. The keyed
        # path is treated as a trusted proxy and skips the public limiter.
        headers: dict[str, str] = {}
        if proxy_state.proxy_llm_key:
            headers[PROXY_LLM_HEADER] = proxy_state.proxy_llm_key
        resp = await _aux().get(
            f"{endpoint}/health", timeout=HEALTH_FETCH_TIMEOUT_S, headers=headers,
        )
        resp.raise_for_status()
        body = resp.json() or {}
    except Exception:
        entry["ok"] = False
        return
    elapsed_ms = (time.monotonic() - started) * 1000.0
    prev = float(entry.get("rtt_ms") or 0.0)
    # EWMA so one slow probe doesn't swing routing.
    entry["rtt_ms"] = elapsed_ms if prev <= 0 else (0.7 * prev + 0.3 * elapsed_ms)
    entry["model"] = str(body.get("model") or "")
    entry["active"] = int(body.get("active_requests") or 0)
    entry["max_requests"] = int(body.get("max_requests") or 0)
    entry["can_accept"] = bool(body.get("can_accept_max_context", True))
    entry["ok"] = str(body.get("status") or "") == "ok"
    if entry["ok"]:
        entry["last_ok"] = time.monotonic()


async def _lb_loop() -> None:
    """Background maintenance: registry refresh + health probes."""
    last_registry = 0.0
    while True:
        try:
            now = time.monotonic()
            if now - last_registry >= REGISTRY_REFRESH_S or not _registry:
                try:
                    await _refresh_registry()
                    last_registry = now
                except Exception as exc:
                    logger.debug("proxy lb registry refresh failed: %s", exc)
            if _registry:
                await asyncio.gather(
                    *(_probe_upstream(ep) for ep in list(_registry)),
                    return_exceptions=True,
                )
        except Exception as exc:  # never let the loop die
            logger.debug("proxy lb loop error: %s", exc)
        await asyncio.sleep(HEALTH_PROBE_S)


def _ensure_lb_started() -> None:
    global _lb_tasks_started
    if _lb_tasks_started or not LOCAL_LB_ENABLED or not proxy_state.balancer_base:
        return
    try:
        asyncio.get_running_loop().create_task(_lb_loop())
        _lb_tasks_started = True
        logger.info(
            "proxy local balancer enabled: registry=%.0fs probe=%.0fs rtt_per_slot=%.0fms",
            REGISTRY_REFRESH_S, HEALTH_PROBE_S, RTT_MS_PER_SLOT,
        )
    except RuntimeError:
        pass  # no loop yet; retried on the next request


def _lb_candidates() -> list[dict]:
    now = time.monotonic()
    want = (proxy_state.model_id or "").strip()
    out = []
    for entry in _registry.values():
        if not entry.get("ok") or now - float(entry.get("last_ok") or 0.0) > STALE_AFTER_S:
            continue
        # Match on the upstream's self-reported model, not the monitor's
        # normalized label — the monitor shortens the id and drops the org.
        if want and entry.get("model") and entry["model"] != want:
            continue
        out.append(entry)
    return out


# validator hotkey -> (upstream endpoint, monotonic expiry). One canary from one
# validator runs its handshake sequentially (chat, then challenge, then
# retention) and canaries are minutes apart, so pinning per validator keeps each
# handshake on one server without needing to parse the commitment digest.
_affinity: dict[str, tuple[str, float]] = {}


def _affinity_set(validator_hotkey: str, endpoint: str) -> None:
    """Pin a validator to the upstream that just served its /chat precommit."""
    if not validator_hotkey or not endpoint:
        return
    _affinity[validator_hotkey] = (endpoint, time.monotonic() + AFFINITY_TTL_S)
    # Bound memory: there are ~7 validators, but drop anything expired on write.
    if len(_affinity) > 64:
        now = time.monotonic()
        for k in [k for k, (_, exp) in _affinity.items() if exp <= now]:
            _affinity.pop(k, None)


def _affinity_get(validator_hotkey: str) -> Optional[str]:
    """Return the pinned upstream for this validator, or None.

    Only returns a pin that is unexpired AND still a live upstream candidate; a
    pin to a server that has gone stale/away falls through to a normal pick,
    which is no worse than having no affinity at all.
    """
    if not validator_hotkey:
        return None
    rec = _affinity.get(validator_hotkey)
    if rec is None:
        return None
    endpoint, expiry = rec
    if time.monotonic() >= expiry:
        _affinity.pop(validator_hotkey, None)
        return None
    live = {str(c.get("endpoint") or "").rstrip("/") for c in _lb_candidates()}
    # When the local balancer has no health view yet (LOCAL_LB disabled or not
    # warmed), trust the pin rather than lose affinity.
    if live and endpoint.rstrip("/") not in live:
        return None
    return endpoint


def _local_pick() -> Optional[dict[str, Any]]:
    """Select an upstream in-process. Returns None to fall back to central /pick.

    Uses power-of-two-choices rather than global least-loaded: every proxy sees
    the same health data, so a strict argmin would make all of them stampede the
    same GPU. Sampling two and taking the better one decorrelates the fleet
    without needing any coordination, and costs no network round trip.
    """
    global _lb_rr
    candidates = _lb_candidates()
    if not candidates:
        return None

    def cost(entry: dict) -> float:
        active = float(entry.get("active") or 0)
        rtt = float(entry.get("rtt_ms") or 0.0)
        return active + (rtt / RTT_MS_PER_SLOT if RTT_MS_PER_SLOT > 0 else 0.0)

    if len(candidates) == 1:
        chosen = candidates[0]
    else:
        # Sample two *independently at random*, then take the cheaper. The
        # power-of-two-choices bound depends on the samples being independent
        # across proxies — a deterministic rotation correlates them and measured
        # 12/24 proxies landing on one GPU in a simultaneous burst.
        _lb_rr += 1
        a, b = random.sample(candidates, 2)
        chosen = a if cost(a) <= cost(b) else b

    # Optimistically count our own in-flight request so back-to-back picks on
    # this proxy spread out before the next health probe lands.
    chosen["active"] = int(chosen.get("active") or 0) + 1
    return {
        "endpoint": chosen["endpoint"],
        "api_key": _last_pick_api_key or proxy_state.proxy_llm_key,
        "worker_id": "",
        "_local": True,
        "_rtt_ms": round(float(chosen.get("rtt_ms") or 0.0), 1),
    }


async def _pick_upstream(*, log_pick: bool = True) -> dict[str, Any]:
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
    resp = await _aux().get(url, headers=_balancer_headers(), timeout=5.0)
    resp.raise_for_status()
    data = resp.json() or {}
    endpoint = str(data.get("endpoint") or "").rstrip("/")
    if not endpoint:
        raise RuntimeError(f"balancer /pick missing endpoint: {data}")
    global _last_pick_api_key
    api_key = str(data.get("api_key") or "").strip()
    if api_key:
        _last_pick_api_key = api_key
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
        # This server verified the validator's sr25519 signature against its own
        # freshly-written allowlist and evaluated the hard-auditor policy while
        # doing so. The upstream cannot repeat either check, so carry the result
        # rather than let it fail closed on a policy file it never receives.
        if bool(getattr(request.state, "proof_v3_hard_auditor_authorized", False)):
            headers[PROXY_HARD_AUDITOR_HEADER] = "1"
    return headers


async def _stream_upstream_response(resp: httpx.Response) -> AsyncIterator[bytes]:
    async for chunk in resp.aiter_bytes():
        yield chunk


# Upstream failures we translate into clean HTTP status codes instead of
# letting them escape into the ASGI layer. httpx.StreamError (e.g. StreamClosed)
# subclasses RuntimeError — NOT HTTPError — so it must be listed explicitly.
# RemoteProtocolError ("incomplete chunked read") and ConnectError are HTTPError.
_UPSTREAM_ERRORS = (httpx.HTTPError, httpx.StreamError)


async def proxy_json_post(
    path: str,
    body: dict[str, Any],
    request: Optional[Request] = None,
) -> Any:
    _ensure_lb_started()
    pick = _local_pick() if LOCAL_LB_ENABLED else None
    if pick is not None:
        logger.info(
            "proxy local pick: upstream=%s rtt=%sms", pick["endpoint"], pick.get("_rtt_ms"),
        )
    else:
        try:
            pick = await _pick_upstream()
        except Exception as exc:
            logger.warning("proxy pick failed: path=%s err=%s", path, exc)
            return JSONResponse(
                status_code=503,
                content={"error": "no upstream available", "detail": str(exc)},
            )
    endpoint = str(pick["endpoint"]).rstrip("/")
    url = f"{endpoint}{path}"
    validator_hotkey = _validator_hotkey(request)
    # /chat and /inference create the proof-v3 precommit on this upstream. Pin the
    # validator here so its follow-up hard challenge and retention hold reach the
    # same server (see proxy_raw_post). Safe to set on every forwarded call; the
    # latest precommit for a validator is the one its next challenge refers to.
    _affinity_set(validator_hotkey, endpoint)
    hotkey_suffix = f" validator={validator_hotkey[:12]}..." if validator_hotkey else ""
    logger.info("proxy forwarding %s -> %s%s", path, url, hotkey_suffix)
    client = _upstream()
    try:
        req = client.build_request(
            "POST",
            url,
            headers=_upstream_headers(pick, request),
            json=body,
        )
        resp = await client.send(req, stream=True)
    except _UPSTREAM_ERRORS as exc:
        logger.warning(
            "proxy upstream connect failed: path=%s upstream=%s%s err=%s",
            path, url, hotkey_suffix, exc,
        )
        return JSONResponse(
            status_code=502,
            content={"error": "upstream connect failed", "detail": str(exc)},
        )
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
        except _UPSTREAM_ERRORS as exc:
            logger.warning(
                "proxy upstream error-body read failed: path=%s upstream=%s%s err=%s",
                path, endpoint, hotkey_suffix, exc,
            )
            return JSONResponse(status_code=502, content={"error": "upstream read failed", "detail": str(exc)})
        finally:
            await resp.aclose()

    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        async def sse_iter():
            # A 200 from upstream only means the HEADERS were fine. The validator
            # scores the *stream*: it needs the token events AND the terminal
            # precommit/done frames. A stream that ends cleanly but short looks
            # identical to success from the status code alone, which is why a
            # proxy could log 200 for every request while the validator recorded
            # "Inference request failed." for every one. Count what we actually
            # relay so the two views can be compared.
            n_bytes = 0
            n_chunks = 0
            n_events = 0
            saw_token = False
            saw_precommit = False
            saw_done = False
            saw_error = False
            t0 = time.monotonic()
            try:
                async for chunk in resp.aiter_bytes():
                    n_bytes += len(chunk)
                    n_chunks += 1
                    n_events += chunk.count(b"event:")
                    if b"event: token" in chunk:
                        saw_token = True
                    if b"precommit" in chunk:
                        saw_precommit = True
                    if b"event: done" in chunk:
                        saw_done = True
                    if b"event: error" in chunk:
                        saw_error = True
                    yield chunk
            except _UPSTREAM_ERRORS as exc:
                # Upstream inference GPU dropped the stream mid-body (crash/OOM,
                # timeout, or connection reset). SSE headers are already sent, so
                # we cannot change the status — stop cleanly; the validator sees a
                # truncated stream and fails that request, which is correct.
                logger.warning(
                    "proxy upstream SSE ended early: path=%s upstream=%s%s err=%s "
                    "bytes=%d chunks=%d events=%d token=%s precommit=%s done=%s",
                    path, endpoint, hotkey_suffix, exc,
                    n_bytes, n_chunks, n_events, saw_token, saw_precommit, saw_done,
                )
            finally:
                complete = saw_done and saw_precommit and not saw_error
                logger.info(
                    "proxy SSE relayed: path=%s upstream=%s%s bytes=%d chunks=%d "
                    "events=%d token=%s precommit=%s done=%s error=%s "
                    "elapsed=%.2fs COMPLETE=%s",
                    path, endpoint, hotkey_suffix, n_bytes, n_chunks, n_events,
                    saw_token, saw_precommit, saw_done, saw_error,
                    time.monotonic() - t0, complete,
                )
                await resp.aclose()

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
    except _UPSTREAM_ERRORS as exc:
        logger.warning(
            "proxy upstream read failed: path=%s upstream=%s%s err=%s",
            path, endpoint, hotkey_suffix, exc,
        )
        return JSONResponse(status_code=502, content={"error": "upstream read failed", "detail": str(exc)})
    finally:
        await resp.aclose()


async def proxy_raw_post(
    path: str,
    body: dict[str, Any],
    request: Optional[Request] = None,
) -> Response:
    """Forward a POST upstream and relay the response bytes verbatim.

    ``proxy_json_post`` re-decodes the upstream body as JSON. That is fine for
    ``/chat`` but destroys a proof-v3 hard opening, which returns raw proof
    bytes: they would come back through ``decode(errors="replace")`` with every
    non-UTF-8 byte silently replaced, so the validator would reject a corrupted
    proof rather than see an error. Status, content-type and body are passed
    through unchanged here.

    These calls (/proof/v3/challenge, /proof/v3/retention) refer to a precommit
    created during /chat on one specific upstream, so they MUST reach that same
    server. Prefer the validator's pinned upstream; only fall back to a fresh
    pick when there is no live pin.
    """
    _ensure_lb_started()
    validator_hotkey = _validator_hotkey(request)
    pinned = _affinity_get(validator_hotkey)
    if pinned is not None:
        pick = {
            "endpoint": pinned,
            "api_key": _last_pick_api_key or proxy_state.proxy_llm_key,
            "_affinity": True,
        }
    else:
        pick = _local_pick() if LOCAL_LB_ENABLED else None
        if pick is None:
            try:
                pick = await _pick_upstream()
            except Exception as exc:
                logger.warning("proxy raw pick failed: path=%s err=%s", path, exc)
                return JSONResponse(
                    status_code=503,
                    content={"error": "no upstream available", "detail": str(exc)},
                )
    endpoint = str(pick["endpoint"]).rstrip("/")
    url = f"{endpoint}{path}"
    client = _upstream()
    # The upstream closes idle keep-alive sockets after ~5s while this pool holds
    # them for 300s, so a proof call arriving a few seconds after the previous one
    # can be written to a socket the peer has already closed. That surfaces as
    # "peer closed connection without sending complete message body" and the
    # request never reaches the upstream at all -- confirmed by its access log,
    # which shows the retention but no challenge. Retry once on a fresh
    # connection; the failed attempt was never processed, so this cannot double
    # up a hard opening.
    # A hard opening is ~10MB of proof bytes. On a high-RTT proxy->upstream link
    # that transfer is dropped mid-body ("peer closed connection without sending
    # complete message body", observed truncating at 0.7/1.0/3.5MB of 10.0MB on
    # uid 243 at ~870ms RTT, while uid 78 at ~450ms relayed all 10,070,571 bytes).
    # The upstream re-runs answer_hard_reveal on each attempt and keeps the
    # challenge pending until it is delivered, so retrying is safe; each attempt
    # costs a replay, so keep the count modest and stay inside the validator's
    # 300s hard-canary budget.
    attempts = max(1, int(os.environ.get("PROXY_RAW_POST_ATTEMPTS", "3") or 3))
    resp = None
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            resp = await client.post(
                url,
                headers=_upstream_headers(pick, request),
                json=body,
            )
            break
        except _UPSTREAM_ERRORS as exc:
            last_exc = exc
            logger.warning(
                "proxy raw upstream connect failed (attempt %d/%d): path=%s upstream=%s err=%s",
                attempt, attempts, path, url, exc,
            )
            if attempt < attempts:
                await asyncio.sleep(0.5 * attempt)
    if resp is None:
        return JSONResponse(
            status_code=502,
            content={"error": "upstream connect failed", "detail": str(last_exc)},
        )
    logger.info(
        "proxy raw relayed: path=%s upstream=%s status=%s bytes=%d validator=%s",
        path, endpoint, resp.status_code, len(resp.content),
        (validator_hotkey or "<none>")[:12],
    )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
        headers={
            k: v for k, v in resp.headers.items() if k.lower() == "cache-control"
        },
    )


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
