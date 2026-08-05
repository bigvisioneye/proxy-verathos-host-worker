"""Canonical owner-signed Gleipnir verdict snapshots."""

from __future__ import annotations

import struct
from dataclasses import dataclass, replace
from typing import Iterable


VERDICT_SNAPSHOT_VERSION = 1
VERDICT_SNAPSHOT_DOMAIN = b"VERATHOS/GLEIPNIR/VERDICT_SNAPSHOT/V1"
_SIGNATURE_BYTES = 64
_MAX_ENTRIES = 100_000
_MAX_STRING_BYTES = 4_096


def _encode_string(value: str, *, field: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    encoded = value.encode("utf-8")
    if not encoded or len(encoded) > _MAX_STRING_BYTES:
        raise ValueError(f"{field} length is invalid")
    return struct.pack(">I", len(encoded)) + encoded


@dataclass(frozen=True, order=True)
class VerdictSnapshotEntryV1:
    miner_address: str
    model_index: int
    miner_hotkey_ss58: str
    model_id: str
    hard_verdict: int
    hard_source_epoch: int
    capacity_gated: bool
    probation: bool

    @property
    def key(self) -> tuple[str, int]:
        return self.miner_address, self.model_index

    def validate(self) -> None:
        if self.miner_address != self.miner_address.lower():
            raise ValueError("miner_address must be lowercase")
        _encode_string(self.miner_address, field="miner_address")
        if type(self.model_index) is not int or self.model_index < 0:
            raise ValueError("model_index is invalid")
        _encode_string(self.miner_hotkey_ss58, field="miner_hotkey_ss58")
        _encode_string(self.model_id, field="model_id")
        if type(self.hard_verdict) is not int or self.hard_verdict not in (-1, 0, 1):
            raise ValueError("hard_verdict is invalid")
        if type(self.hard_source_epoch) is not int:
            raise ValueError("hard_source_epoch is invalid")
        if self.hard_verdict == -1:
            if self.hard_source_epoch != -1:
                raise ValueError("none hard verdict requires source epoch -1")
        elif self.hard_source_epoch < 0:
            raise ValueError("hard verdict requires a non-negative source epoch")
        if type(self.capacity_gated) is not bool:
            raise ValueError("capacity_gated is invalid")
        if type(self.probation) is not bool:
            raise ValueError("probation is invalid")

    def encode(self) -> bytes:
        self.validate()
        return b"".join(
            (
                _encode_string(self.miner_address, field="miner_address"),
                struct.pack(">q", self.model_index),
                _encode_string(
                    self.miner_hotkey_ss58,
                    field="miner_hotkey_ss58",
                ),
                _encode_string(self.model_id, field="model_id"),
                struct.pack(">b", self.hard_verdict),
                struct.pack(">q", self.hard_source_epoch),
                struct.pack(">?", self.capacity_gated),
                struct.pack(">?", self.probation),
            )
        )


@dataclass(frozen=True)
class VerdictSnapshotV1:
    auditor_hotkey: bytes
    epoch_number: int
    generated_at: int
    entries: tuple[VerdictSnapshotEntryV1, ...]
    signature: bytes = b""
    snapshot_version: int = VERDICT_SNAPSHOT_VERSION

    def validate(self, *, require_signature: bool = False) -> None:
        if self.snapshot_version != VERDICT_SNAPSHOT_VERSION:
            raise ValueError("verdict snapshot version is unsupported")
        if len(self.auditor_hotkey) != 32:
            raise ValueError("verdict snapshot auditor hotkey is invalid")
        if type(self.epoch_number) is not int or self.epoch_number < 0:
            raise ValueError("verdict snapshot epoch is invalid")
        if type(self.generated_at) is not int or self.generated_at <= 0:
            raise ValueError("verdict snapshot timestamp is invalid")
        if not isinstance(self.entries, tuple) or len(self.entries) > _MAX_ENTRIES:
            raise ValueError("verdict snapshot entry count is invalid")
        prior_key: tuple[str, int] | None = None
        for entry in self.entries:
            if not isinstance(entry, VerdictSnapshotEntryV1):
                raise ValueError("verdict snapshot entry is invalid")
            entry.validate()
            if prior_key is not None and entry.key <= prior_key:
                raise ValueError("verdict snapshot entries are not strictly sorted")
            prior_key = entry.key
        if self.signature and len(self.signature) != _SIGNATURE_BYTES:
            raise ValueError("verdict snapshot signature is invalid")
        if require_signature and len(self.signature) != _SIGNATURE_BYTES:
            raise ValueError("verdict snapshot signature is missing")

    def unsigned_bytes(self) -> bytes:
        self.validate()
        return b"".join(
            (
                struct.pack(">I", self.snapshot_version),
                self.auditor_hotkey,
                struct.pack(">q", self.epoch_number),
                struct.pack(">q", self.generated_at),
                struct.pack(">I", len(self.entries)),
                *(entry.encode() for entry in self.entries),
            )
        )

    def signing_message(self) -> bytes:
        return VERDICT_SNAPSHOT_DOMAIN + self.unsigned_bytes()

    def to_bytes(self) -> bytes:
        self.validate(require_signature=True)
        return self.unsigned_bytes() + self.signature

    @classmethod
    def from_bytes(cls, raw: bytes) -> "VerdictSnapshotV1":
        if not isinstance(raw, bytes) or len(raw) < 4 + 32 + 8 + 8 + 4 + 64:
            raise ValueError("verdict snapshot bytes are truncated")
        cursor = _Cursor(raw)
        version = cursor.u32()
        auditor_hotkey = cursor.take(32)
        epoch_number = cursor.i64()
        generated_at = cursor.i64()
        entry_count = cursor.u32()
        if entry_count > _MAX_ENTRIES:
            raise ValueError("verdict snapshot entry count is invalid")
        entries: list[VerdictSnapshotEntryV1] = []
        for _ in range(entry_count):
            entries.append(
                VerdictSnapshotEntryV1(
                    miner_address=cursor.string("miner_address"),
                    model_index=cursor.i64(),
                    miner_hotkey_ss58=cursor.string("miner_hotkey_ss58"),
                    model_id=cursor.string("model_id"),
                    hard_verdict=cursor.i8(),
                    hard_source_epoch=cursor.i64(),
                    capacity_gated=cursor.boolean(),
                    probation=cursor.boolean(),
                )
            )
        signature = cursor.take(_SIGNATURE_BYTES)
        if cursor.remaining:
            raise ValueError("verdict snapshot bytes have trailing data")
        snapshot = cls(
            snapshot_version=version,
            auditor_hotkey=auditor_hotkey,
            epoch_number=epoch_number,
            generated_at=generated_at,
            entries=tuple(entries),
            signature=signature,
        )
        snapshot.validate(require_signature=True)
        return snapshot


class _Cursor:
    def __init__(self, raw: bytes):
        self.raw = raw
        self.offset = 0

    @property
    def remaining(self) -> int:
        return len(self.raw) - self.offset

    def take(self, size: int) -> bytes:
        if size < 0 or self.remaining < size:
            raise ValueError("verdict snapshot bytes are truncated")
        value = self.raw[self.offset : self.offset + size]
        self.offset += size
        return value

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(size))[0]

    def u32(self) -> int:
        return int(self.unpack(">I"))

    def i64(self) -> int:
        return int(self.unpack(">q"))

    def i8(self) -> int:
        return int(self.unpack(">b"))

    def boolean(self) -> bool:
        raw = self.take(1)
        if raw not in (b"\x00", b"\x01"):
            raise ValueError("verdict snapshot boolean is invalid")
        return raw == b"\x01"

    def string(self, field: str) -> str:
        size = self.u32()
        if size <= 0 or size > _MAX_STRING_BYTES:
            raise ValueError(f"{field} length is invalid")
        try:
            return self.take(size).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{field} encoding is invalid") from exc


def canonical_verdict_entries_v1(
    entries: Iterable[VerdictSnapshotEntryV1],
) -> tuple[VerdictSnapshotEntryV1, ...]:
    values = tuple(sorted(entries, key=lambda item: item.key))
    for entry in values:
        entry.validate()
    if len({entry.key for entry in values}) != len(values):
        raise ValueError("verdict snapshot contains duplicate endpoint entries")
    return values


def sign_verdict_snapshot_v1(
    snapshot: VerdictSnapshotV1,
    validator_seed: bytes,
) -> VerdictSnapshotV1:
    if len(validator_seed) < 32:
        raise ValueError("verdict snapshot signing seed is invalid")
    snapshot.validate()
    from bittensor_wallet import Keypair

    keypair = Keypair.create_from_seed(validator_seed[:32].hex())
    if bytes(keypair.public_key) != snapshot.auditor_hotkey:
        raise ValueError("verdict snapshot signing key does not match auditor")
    signature = keypair.sign(snapshot.signing_message())
    if not isinstance(signature, bytes):
        signature = bytes.fromhex(
            signature[2:] if signature.startswith("0x") else signature
        )
    signed = replace(snapshot, signature=signature)
    signed.validate(require_signature=True)
    return signed


def verify_verdict_snapshot_v1(
    snapshot: VerdictSnapshotV1,
    *,
    expected_auditor_hotkey_ss58: str,
) -> bool:
    try:
        snapshot.validate(require_signature=True)
        from bittensor_wallet import Keypair
        from verallm.chain.wallet import ss58_decode

        expected_key = bytes(ss58_decode(expected_auditor_hotkey_ss58))
        if snapshot.auditor_hotkey != expected_key:
            return False
        keypair = Keypair(ss58_address=expected_auditor_hotkey_ss58)
        return bool(
            keypair.verify(snapshot.signing_message(), snapshot.signature)
        )
    except Exception:
        return False


__all__ = [
    "VERDICT_SNAPSHOT_DOMAIN",
    "VERDICT_SNAPSHOT_VERSION",
    "VerdictSnapshotEntryV1",
    "VerdictSnapshotV1",
    "canonical_verdict_entries_v1",
    "sign_verdict_snapshot_v1",
    "verify_verdict_snapshot_v1",
]
