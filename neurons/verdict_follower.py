"""Fetch and validate the owner validator's current verdict snapshot."""

from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from neurons.verdict_records import (
    VerdictSnapshotV1,
    verify_verdict_snapshot_v1,
)


class VerdictSnapshotUnavailable(RuntimeError):
    """The owner feed could not supply an acceptable current snapshot."""


def verdict_snapshot_url(owner_url: str, *, epoch: int | None = None) -> str:
    value = str(owner_url or "").strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("owner verdict URL must be an absolute HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise ValueError("owner verdict URL cannot include query or fragment data")
    if parsed.path.rstrip("/").endswith("/v1/verdicts/current"):
        base = value
    else:
        base = value + "/v1/verdicts/current"
    if epoch is None:
        return base
    if type(epoch) is not int or epoch < 0:
        raise ValueError("verdict snapshot epoch must be a non-negative integer")
    return f"{base}?epoch={epoch}"


def owner_has_validator_permit(owner_hotkey_ss58: str, metagraph: object) -> bool:
    if metagraph is None:
        return False
    try:
        hotkeys = list(getattr(metagraph, "hotkeys"))
        uid = hotkeys.index(owner_hotkey_ss58)
        permits = getattr(metagraph, "validator_permit")
        return bool(permits[uid])
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class AcceptedVerdictSnapshot:
    snapshot: VerdictSnapshotV1
    raw: bytes


class VerdictSnapshotFollower:
    """In-memory anti-replay state for the simplified follower feed."""

    def __init__(self) -> None:
        self._last_version: tuple[int, int] | None = None
        self._last_raw: bytes = b""
        self._current: AcceptedVerdictSnapshot | None = None

    @property
    def current(self) -> AcceptedVerdictSnapshot | None:
        return self._current

    def clear_current(self) -> None:
        """Neutralize application without weakening the anti-replay floor."""

        self._current = None

    def accept_raw(
        self,
        raw: bytes,
        *,
        expected_owner_hotkey_ss58: str,
        metagraph: object,
        closing_epoch: int,
        epoch_seconds: float,
        now: int | None = None,
    ) -> VerdictSnapshotV1:
        try:
            snapshot = VerdictSnapshotV1.from_bytes(raw)
        except ValueError as exc:
            raise VerdictSnapshotUnavailable(
                f"snapshot encoding rejected: {exc}"
            ) from exc
        if not owner_has_validator_permit(
            expected_owner_hotkey_ss58,
            metagraph,
        ):
            raise VerdictSnapshotUnavailable(
                "configured owner lacks a validator permit in the local metagraph"
            )
        if not verify_verdict_snapshot_v1(
            snapshot,
            expected_auditor_hotkey_ss58=expected_owner_hotkey_ss58,
        ):
            raise VerdictSnapshotUnavailable(
                "snapshot signature or configured owner identity is invalid"
            )

        epoch = int(closing_epoch)
        if snapshot.epoch_number not in (epoch, max(0, epoch - 1)):
            raise VerdictSnapshotUnavailable(
                "snapshot epoch is outside the closing/current decision window"
            )
        now_s = int(time.time()) if now is None else int(now)
        max_age = max(1.0, 2.0 * float(epoch_seconds))
        if snapshot.generated_at < int(now_s - max_age):
            raise VerdictSnapshotUnavailable("snapshot is stale")
        if snapshot.generated_at > now_s + 300:
            raise VerdictSnapshotUnavailable("snapshot timestamp is in the future")

        version = (snapshot.epoch_number, snapshot.generated_at)
        if self._last_version is not None and version < self._last_version:
            raise VerdictSnapshotUnavailable(
                "snapshot is older than the last accepted owner decision"
            )
        if (
            self._last_version is not None
            and version == self._last_version
            and self._last_raw
            and raw != self._last_raw
        ):
            raise VerdictSnapshotUnavailable(
                "snapshot conflicts with the last accepted owner decision"
            )
        self._last_version = version
        self._last_raw = raw
        self._current = AcceptedVerdictSnapshot(snapshot=snapshot, raw=raw)
        return snapshot

    def fetch(
        self,
        owner_url: str,
        *,
        expected_owner_hotkey_ss58: str,
        metagraph: object,
        closing_epoch: int,
        epoch_seconds: float,
        timeout: float = 10.0,
        now: int | None = None,
    ) -> VerdictSnapshotV1:
        url = verdict_snapshot_url(owner_url, epoch=int(closing_epoch))
        try:
            response = httpx.get(
                url,
                timeout=max(1.0, float(timeout)),
                follow_redirects=False,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise VerdictSnapshotUnavailable(
                f"owner feed request failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(payload, dict) or set(payload) != {"snapshot"}:
            raise VerdictSnapshotUnavailable("owner feed response shape is invalid")
        encoded = payload.get("snapshot")
        if not isinstance(encoded, str) or len(encoded) > 64 * 1024 * 1024:
            raise VerdictSnapshotUnavailable("owner feed snapshot is invalid")
        try:
            raw = bytes.fromhex(encoded)
        except ValueError as exc:
            raise VerdictSnapshotUnavailable(
                "owner feed snapshot hex is invalid"
            ) from exc
        return self.accept_raw(
            raw,
            expected_owner_hotkey_ss58=expected_owner_hotkey_ss58,
            metagraph=metagraph,
            closing_epoch=closing_epoch,
            epoch_seconds=epoch_seconds,
            now=now,
        )


__all__ = [
    "AcceptedVerdictSnapshot",
    "VerdictSnapshotFollower",
    "VerdictSnapshotUnavailable",
    "owner_has_validator_permit",
    "verdict_snapshot_url",
]
