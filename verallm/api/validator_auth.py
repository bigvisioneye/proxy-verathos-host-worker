"""Validator authentication middleware for the miner server.

Only allows requests signed by validators registered on the subnet.
The validator list is read from a JSON file (written by neurons/miner.py
which queries the metagraph periodically).

Exempt endpoints: /health, /model_spec (always public, rate-limited per IP).

Package boundary: this module does NOT import bittensor. The metagraph
lookup is done in neurons/miner.py which writes the allowlist file.
Signature verification uses bittensor-wallet (Sr25519).
"""

from __future__ import annotations

import collections
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Set

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# ss58 addresses on this network are base58, 47-48 chars. This is a shape
# check, not an authenticity check -- the proxy already verified the signature.
# It exists so a blank or garbage header can never reach the proof context and
# get bound into a proof.
_SS58_ALPHABET = set(
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
)


def _plausible_ss58(value: str) -> bool:
    return (
        type(value) is str
        and 46 <= len(value) <= 50
        and not (set(value) - _SS58_ALPHABET)
    )

# Historical fallback when a miner has no configured data directory.
DEFAULT_VALIDATORS_PATH = "/tmp/verathos_validators.json"


def resolve_validators_path(validators_path: Optional[str] = None) -> Path:
    """Resolve the per-miner validator policy file.

    Multiple miners may share one host.  Their wrapper and serving subprocess
    must not exchange validator policy through the historical process-global
    ``/tmp`` path.  ``VERALLM_DATA_DIR`` is already unique per installed miner
    and owns its other persistent protocol state, so use it unless an explicit
    policy path was configured.
    """

    explicit = str(validators_path or "").strip()
    if not explicit:
        explicit = os.environ.get("VERATHOS_VALIDATORS_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser()

    data_dir = os.environ.get("VERALLM_DATA_DIR", "").strip()
    if data_dir:
        return Path(data_dir).expanduser() / "validators.json"
    return Path(DEFAULT_VALIDATORS_PATH)


# How often to re-read the file (seconds)
FILE_RELOAD_INTERVAL = 60
HARD_AUDITOR_FILE_MAX_AGE_SECONDS = 900

# Endpoints that don't require validator auth
PUBLIC_ENDPOINTS = {"/health", "/model_spec", "/docs", "/openapi.json", "/identity/challenge", "/tee/info"}
PROOF_V3_HARD_CHALLENGE_PATH = "/proof/v3/challenge"

# Rate limiting for public endpoints (per IP).
# 60/min allows 20 validator proxies health-checking every 10s (= 120/min)
# since each proxy has its own IP. Random scanners get blocked quickly.
PUBLIC_RATE_LIMIT = 60       # max requests per window
PUBLIC_RATE_WINDOW = 60.0    # window in seconds
_MAX_TRACKED_IPS = 10_000    # cap to prevent memory exhaustion


class _PublicRateLimiter:
    """Lightweight in-memory sliding-window rate limiter keyed by IP."""

    __slots__ = ("_limit", "_window", "_hits")

    def __init__(self, limit: int = PUBLIC_RATE_LIMIT, window: float = PUBLIC_RATE_WINDOW):
        self._limit = limit
        self._window = window
        # IP → deque of timestamps
        self._hits: dict[str, collections.deque] = {}

    def is_allowed(self, ip: str) -> bool:
        now = time.monotonic()
        bucket = self._hits.get(ip)
        if bucket is None:
            if len(self._hits) >= _MAX_TRACKED_IPS:
                # Evict oldest bucket to cap memory
                oldest_ip = next(iter(self._hits))
                del self._hits[oldest_ip]
            bucket = collections.deque()
            self._hits[ip] = bucket

        # Expire old entries
        cutoff = now - self._window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= self._limit:
            return False

        bucket.append(now)
        return True


class ValidatorAuthMiddleware(BaseHTTPMiddleware):
    """Require a valid validator signature on all non-public endpoints.

    Reads the allowed validator SS58 hotkeys from a JSON file on disk.
    The file is re-read every FILE_RELOAD_INTERVAL seconds.

    When no validators file exists, blocks all non-public requests (deny by
    default).
    """

    def __init__(self, app, validators_path: Optional[str] = None):
        super().__init__(app)
        self._validators_path = resolve_validators_path(validators_path)
        self._allowed_ss58: Set[str] = set()  # SS58-encoded hotkey addresses
        self._proof_v3_hard_auditor_ss58 = ""
        self._last_load: float = 0.0
        self._no_file_warned: bool = False  # Avoid spamming the missing-file warning
        self._public_limiter = _PublicRateLimiter()
        self._load_validators()

    def _load_validators(self) -> None:
        """Load validator SS58 hotkeys from the JSON file."""
        now = time.time()
        if now - self._last_load < FILE_RELOAD_INTERVAL:
            return
        self._last_load = now

        if not self._validators_path.exists():
            # Ordinary access may retain the last-good validator set during a
            # transient refresh failure, but hard-proof authority must not
            # survive a missing policy file.
            self._proof_v3_hard_auditor_ss58 = ""
            if not self._no_file_warned:
                logger.warning(
                    "Validators file not found at %s — non-public requests will be "
                    "blocked until the file is created.",
                    self._validators_path,
                )
                self._no_file_warned = True
            return

        try:
            data = json.loads(self._validators_path.read_text())
            validators = data.get("validators", [])
            new_ss58 = {v["hotkey_ss58"] for v in validators if v.get("hotkey_ss58")}
            hard_policy = data.get("proof_v3_hard_auditor", {})
            hard_hotkey = ""
            updated_at = data.get("updated_at")
            hard_policy_fresh = (
                isinstance(updated_at, (int, float))
                and not isinstance(updated_at, bool)
                and 0.0
                <= now - float(updated_at)
                <= HARD_AUDITOR_FILE_MAX_AGE_SECONDS
            )
            if (
                hard_policy_fresh
                and isinstance(hard_policy, dict)
                and hard_policy.get("enabled") is True
            ):
                candidate = hard_policy.get("validator_hotkey_ss58", "")
                if isinstance(candidate, str) and candidate in new_ss58:
                    hard_hotkey = candidate

            if new_ss58 != self._allowed_ss58:
                logger.info(
                    "Loaded %d validator hotkeys from %s",
                    len(new_ss58), self._validators_path,
                )
            self._allowed_ss58 = new_ss58
            self._proof_v3_hard_auditor_ss58 = hard_hotkey
            self._no_file_warned = False
        except Exception as e:
            # Ordinary validator access may retain its last-good set, but hard
            # proof authority always fails closed on a malformed policy file.
            self._proof_v3_hard_auditor_ss58 = ""
            logger.warning("Failed to load validators file: %s", e)

    async def dispatch(self, request: Request, call_next):
        # Dev/test bypass: skip validator auth entirely.  Only honored when
        # VERATHOS_NO_VALIDATOR_AUTH=1 is explicitly set in the server's
        # environment.  Default is OFF — production miners NEVER set this
        # env var, so production behaviour is unchanged.  Used by
        # scripts/dry_run_new_model.sh so a fresh-pod test can exercise
        # /v1/chat/completions + proof generation without needing a signed
        # validator request.  Real validator auth is always exercised by
        # canaries from registered validators on the subnet.
        if os.environ.get("VERATHOS_NO_VALIDATOR_AUTH") == "1":
            return await call_next(request)

        # Trusted proxy forwarding from a VPS miner (inference GPU fleet).
        try:
            from verallm.api.proxy_auth import proxy_llm_key_from_env, verify_proxy_llm_request

            if proxy_llm_key_from_env() and verify_proxy_llm_request(request):
                client_ip = request.client.host if request.client else "unknown"
                # The proxy miner already ran this same middleware against its
                # own validator allowlist and verified the sr25519 signature
                # over (method, path, body) before forwarding. This server
                # cannot repeat that check -- it holds no allowlist, and the
                # proxy re-serialises the body, so the signed bytes are gone.
                # It can only carry the identity the proxy authenticated.
                #
                # Without this, request.state.validator_hotkey stays unset and
                # every proof-v3 precommit dies on
                # "network hotkey must be non-empty text" (409), which is
                # exactly what made proxy miners unable to serve at all.
                forwarded = str(
                    request.headers.get("x-validator-hotkey", "") or ""
                ).strip()
                if _plausible_ss58(forwarded):
                    request.state.validator_hotkey = forwarded
                    request.state.proof_v3_hard_auditor_authorized = bool(
                        forwarded == self._proof_v3_hard_auditor_ss58
                    )
                else:
                    # Refuse rather than prove under an unknown identity: a
                    # bound proof carrying a blank or malformed validator is
                    # worse than a clean rejection.
                    if forwarded:
                        logger.warning(
                            "Proxy-forwarded %s carried an implausible "
                            "validator hotkey; refusing",
                            request.url.path,
                        )
                    request.state.validator_hotkey = ""
                    request.state.proof_v3_hard_auditor_authorized = False
                logger.info(
                    "Accepted proxy-forwarded %s from %s (validator=%s)",
                    request.url.path,
                    client_ip,
                    forwarded or "<none>",
                )
                return await call_next(request)
        except Exception:
            pass

        # Reload validators periodically
        self._load_validators()

        # Public endpoints — rate-limited per IP
        if request.url.path in PUBLIC_ENDPOINTS:
            client_ip = request.client.host if request.client else "unknown"
            if not self._public_limiter.is_allowed(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded"},
                    headers={"Retry-After": str(int(PUBLIC_RATE_WINDOW))},
                )
            return await call_next(request)

        # No validators file yet — deny all non-public requests
        if not self._allowed_ss58:
            return JSONResponse(
                status_code=503,
                content={"error": "Miner is starting up — validator allowlist not yet loaded"},
            )

        # Extract auth headers
        hotkey_ss58 = request.headers.get("x-validator-hotkey", "")
        signature_hex = request.headers.get("x-validator-signature", "")
        timestamp_str = request.headers.get("x-validator-timestamp", "")

        if not hotkey_ss58 or not signature_hex or not timestamp_str:
            return JSONResponse(
                status_code=401,
                content={"error": "Missing validator auth headers (X-Validator-Hotkey, X-Validator-Signature, X-Validator-Timestamp)"},
            )

        # Check if hotkey is in the allowed set
        if hotkey_ss58 not in self._allowed_ss58:
            return JSONResponse(
                status_code=403,
                content={"error": "Hotkey is not a registered validator on this subnet"},
            )

        # Verify signature
        # Import here to avoid circular imports at module level
        from neurons.request_signing import verify_request

        body = await request.body()
        ok, reason = verify_request(
            method=request.method,
            path=request.url.path,
            body=body,
            hotkey_ss58=hotkey_ss58,
            signature_hex=signature_hex,
            timestamp_str=timestamp_str,
        )

        if not ok:
            return JSONResponse(
                status_code=401,
                content={"error": f"Validator signature verification failed: {reason}"},
            )

        # Store validated hotkey on request state for downstream logging
        request.state.validator_hotkey = hotkey_ss58
        request.state.proof_v3_hard_auditor_authorized = bool(
            hotkey_ss58 == self._proof_v3_hard_auditor_ss58
        )
        if (
            request.url.path == PROOF_V3_HARD_CHALLENGE_PATH
            and not request.state.proof_v3_hard_auditor_authorized
        ):
            return JSONResponse(
                status_code=403,
                content={
                    "error": (
                        "Validator is not authorized for proof-v3 hard openings"
                    )
                },
            )

        return await call_next(request)
