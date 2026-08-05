"""Signed per-model calibration SET: context bands as protocol data.

A deployed model ships ONE authenticated calibration set.  The set
carries a version, the audit sampling policy (pool size, chunk length,
heads/rows/uniform/mass sampling counts, attention semantics tag) and an
ordered list of contiguous, non-overlapping context bands, each with
exactly ONE frozen ``ScoredCalibrationV3`` (signed scales + corridor
bounds + discriminative layer set).  The manifest authenticates the
canonical SET digest; each band's calibration digest is bound into the
reduction geometry and transcript, so miner and validator provably use
the same selected band.

Selection is deterministic: the validator picks exactly one band from
its OWN authenticated ``key_count``.  Out-of-domain contexts, gaps,
overlaps, duplicate or unsorted bands, mismatched discriminative sets
and unsupported versions all fail closed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.scored_calibration import (
    ScoredCalibrationV3,
    dump_scored_calibration_v3,
    load_scored_calibration_v3,
)

__all__ = [
    "CalibrationBandV3",
    "ReductionSamplingPolicyV3",
    "ScoredCalibrationSetV3",
    "dump_scored_calibration_set_v3",
    "load_scored_calibration_set_v3",
    "scored_calibration_set_digest_v3",
    "select_signed_calibration_v3",
]

CALIBRATION_SET_VERSION = 1

_SET_DOM = b"verathos.proof_v3.calibration_set.v1"


def _u32(value: int) -> bytes:
    if not 0 <= int(value) < (1 << 32):
        raise ProofV3Error("calibration set field out of u32 range")
    return int(value).to_bytes(4, "little")


def _u64(value: int) -> bytes:
    if not 0 <= int(value) < (1 << 64):
        raise ProofV3Error("calibration set field out of u64 range")
    return int(value).to_bytes(8, "little")


@dataclass(frozen=True, slots=True)
class ReductionSamplingPolicyV3:
    """The audit's sampling policy -- digest-bound protocol data, never
    a per-request miner choice."""

    pool: int
    chunk_len: int
    heads_per_layer: int
    row_samples: int
    uniform_chunk_samples: int
    mass_draws: int
    attention: str          # attention-semantics tag, e.g. "causal-flash-v3"

    def validate(self) -> None:
        for name in ("pool", "chunk_len", "heads_per_layer",
                     "row_samples", "uniform_chunk_samples",
                     "mass_draws"):
            if int(getattr(self, name)) < 1:
                raise ProofV3Error(
                    f"sampling policy field {name} must be >= 1")
        if not self.attention or not isinstance(self.attention, str):
            raise ProofV3Error(
                "sampling policy needs an attention semantics tag")


@dataclass(frozen=True, slots=True)
class CalibrationBandV3:
    """One inclusive context range [lo, hi] with its ONE calibration."""

    lo: int
    hi: int
    calibration: ScoredCalibrationV3


@dataclass(frozen=True, slots=True)
class ScoredCalibrationSetV3:
    version: int
    policy: ReductionSamplingPolicyV3
    bands: tuple[CalibrationBandV3, ...]
    digest: bytes

    def band_for(self, key_count: int) -> CalibrationBandV3:
        """Exactly one band from the AUTHENTICATED key count; contexts
        outside the calibrated domain fail closed."""

        kc = int(key_count)
        if kc < self.bands[0].lo or kc > self.bands[-1].hi:
            raise ProofV3Error(
                f"key count {kc} is outside the calibrated context "
                f"domain [{self.bands[0].lo}, {self.bands[-1].hi}]")
        for band in self.bands:
            if band.lo <= kc <= band.hi:
                return band
        raise ProofV3Error(
            f"no calibration band covers key count {kc}")


def scored_calibration_set_digest_v3(
        *, version: int, policy: ReductionSamplingPolicyV3,
        bands) -> bytes:
    """Canonical set digest: version, policy and per-band
    (lo, hi, band calibration digest) in band order."""

    attention = policy.attention.encode("utf-8")
    payload = [
        _SET_DOM, _u32(version),
        _u32(policy.pool), _u32(policy.chunk_len),
        _u32(policy.heads_per_layer), _u32(policy.row_samples),
        _u32(policy.uniform_chunk_samples), _u32(policy.mass_draws),
        _u32(len(attention)), attention,
        _u32(len(bands)),
    ]
    for band in bands:
        payload.append(_u64(band.lo))
        payload.append(_u64(band.hi))
        if len(band.calibration.digest) != 32:
            raise ProofV3Error("band calibration digest must be 32 bytes")
        payload.append(band.calibration.digest)
    return hashlib.sha256(b"".join(payload)).digest()


def _validate_bands(bands: tuple[CalibrationBandV3, ...]) -> None:
    if not bands:
        raise ProofV3Error("calibration set needs at least one band")
    discriminative = bands[0].calibration.discriminative
    for i, band in enumerate(bands):
        if band.lo > band.hi:
            raise ProofV3Error(
                f"band {i} range [{band.lo}, {band.hi}] is inverted")
        if band.lo < 1:
            raise ProofV3Error(f"band {i} starts below context 1")
        if band.calibration.discriminative != discriminative:
            raise ProofV3Error(
                "bands disagree on the discriminative layer set")
    for i in range(1, len(bands)):
        prev, cur = bands[i - 1], bands[i]
        if cur.lo <= prev.hi:
            raise ProofV3Error(
                f"bands {i - 1} and {i} overlap or are unsorted "
                f"([{prev.lo},{prev.hi}] then [{cur.lo},{cur.hi}])")
        if cur.lo != prev.hi + 1:
            raise ProofV3Error(
                f"gap between bands {i - 1} and {i} "
                f"(context {prev.hi + 1}..{cur.lo - 1} uncovered)")


def load_scored_calibration_set_v3(source) -> ScoredCalibrationSetV3:
    """Parse a calibration-set blob (dict, JSON string, or file path),
    validate every protocol rule and verify the canonical digest.
    Callers must additionally compare ``.digest`` against the
    manifest's authenticated set digest."""

    if isinstance(source, (str, bytes)) and not str(
            source).lstrip().startswith("{"):
        with open(source, "rb") as handle:
            obj = json.load(handle)
    elif isinstance(source, (str, bytes)):
        obj = json.loads(source)
    else:
        obj = source
    if not isinstance(obj, dict):
        raise ProofV3Error("calibration set blob must be a JSON object")
    version = int(obj.get("version", 0))
    if version != CALIBRATION_SET_VERSION:
        raise ProofV3Error(
            f"unsupported calibration set version {version}")
    policy_obj = obj.get("policy")
    if not isinstance(policy_obj, dict):
        raise ProofV3Error("calibration set needs a policy object")
    try:
        policy = ReductionSamplingPolicyV3(
            pool=int(policy_obj["pool"]),
            chunk_len=int(policy_obj["chunk_len"]),
            heads_per_layer=int(policy_obj["heads_per_layer"]),
            row_samples=int(policy_obj["row_samples"]),
            uniform_chunk_samples=int(
                policy_obj["uniform_chunk_samples"]),
            mass_draws=int(policy_obj["mass_draws"]),
            attention=str(policy_obj["attention"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProofV3Error(
            f"calibration set policy is malformed: {exc}") from exc
    policy.validate()
    bands_obj = obj.get("bands")
    if not isinstance(bands_obj, list) or not bands_obj:
        raise ProofV3Error("calibration set needs a bands list")
    bands = []
    for i, band_obj in enumerate(bands_obj):
        if not isinstance(band_obj, dict):
            raise ProofV3Error(f"band {i} must be an object")
        try:
            lo = int(band_obj["lo"])
            hi = int(band_obj["hi"])
            calibration = load_scored_calibration_v3(
                band_obj["calibration"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProofV3Error(
                f"band {i} is malformed: {exc}") from exc
        bands.append(CalibrationBandV3(
            lo=lo, hi=hi, calibration=calibration))
    bands = tuple(bands)
    _validate_bands(bands)
    digest = scored_calibration_set_digest_v3(
        version=version, policy=policy, bands=bands)
    declared = obj.get("digest")
    if declared is not None and bytes.fromhex(declared) != digest:
        raise ProofV3Error(
            "calibration set digest does not match its contents")
    return ScoredCalibrationSetV3(
        version=version, policy=policy, bands=bands, digest=digest)


def select_signed_calibration_v3(manifest, calibration_set, key_count):
    """Bind a calibration set to a signed manifest, then select a band.

    Fail-closed contract for the multi-band release path: the manifest's
    SIGNED ``attn_calibration_set_digest`` must equal the loaded set's
    canonical digest, and exactly one of the manifest's single/set
    calibration digests may be present.  Returns the
    ``ScoredCalibrationV3`` for the request's authenticated ``key_count``
    (the adapter then binds THAT band's per-calibration digest into the
    reduction geometry and transcript).
    """

    single = getattr(manifest, "attn_calibration_digest", b"") or b""
    signed = getattr(manifest, "attn_calibration_set_digest", b"") or b""
    if not signed:
        raise ProofV3Error(
            "manifest pins no calibration SET digest")
    if single:
        raise ProofV3Error(
            "manifest pins BOTH a single and a set calibration digest")
    if calibration_set.digest != signed:
        raise ProofV3Error(
            "calibration set digest does not match the signed manifest")
    return calibration_set.band_for(int(key_count)).calibration


def dump_scored_calibration_set_v3(
        calibration_set: ScoredCalibrationSetV3) -> dict:
    """Canonical JSON-ready dict; round-trips through the loader to the
    same digest."""

    return {
        "version": calibration_set.version,
        "policy": {
            "pool": calibration_set.policy.pool,
            "chunk_len": calibration_set.policy.chunk_len,
            "heads_per_layer": calibration_set.policy.heads_per_layer,
            "row_samples": calibration_set.policy.row_samples,
            "uniform_chunk_samples":
                calibration_set.policy.uniform_chunk_samples,
            "mass_draws": calibration_set.policy.mass_draws,
            "attention": calibration_set.policy.attention,
        },
        "bands": [
            {
                "lo": band.lo,
                "hi": band.hi,
                "calibration": dump_scored_calibration_v3(
                    band.calibration),
            }
            for band in calibration_set.bands
        ],
        "digest": calibration_set.digest.hex(),
    }
