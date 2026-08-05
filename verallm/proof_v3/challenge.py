"""Post-commitment Fiat-Shamir challenges for proof-v3 folded execution."""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from collections.abc import Sequence

from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.payload import ProofV3CommitmentEnvelope
from verallm.proof_v3.profile import ExecutionSecurityProfileV3
from verallm.proof_v3.relation import (
    POSTCOMMIT_AUDIT_TIER_ABI_V3,
    STRATIFIED_LAYER_HEAD_SELECTION_ABI_V3,
    LayerAuditPlanV3,
    RuntimeHardAuditPolicyV3,
)
from zkllm.crypto.pcs_v2 import PALLAS_SCALAR_MODULUS


_TRANSCRIPT_DOMAIN = b"VERATHOS/PROOF_V3/FOLDED_EXECUTION/TRANSCRIPT/SHA256"
_PALLAS_AUXILIARY_SCALAR_DOMAIN = (
    b"VERATHOS/PROOF_V3/FOLDED_EXECUTION/PALLAS_AUXILIARY_SCALAR/SHA256"
)
_SELECTION_DOMAIN = b"VERATHOS/PROOF_V3/STRATIFIED_LAYER_HEADS/SHA256"
_TIER_SEED_DOMAIN = b"VERATHOS/PROOF_V3/AUDIT_TIER/SEED/SHA256"
_TIER_DRAW_DOMAIN = b"VERATHOS/PROOF_V3/AUDIT_TIER/DRAW/SHA256"
_GLOBAL_RELATION_SEED_DOMAIN = b"VERATHOS/PROOF_V3/GLOBAL_RELATION/SEED/SHA256"
_PALLAS_AFFINE_FOLD_SCALAR_DOMAIN = (
    b"VERATHOS/PROOF_V3/PALLAS_AFFINE_FOLD/SCALAR/SHA256"
)
_LABELS = (
    b"linear-alpha",
    b"bridge-alpha",
    b"attention-alpha",
    b"prefill-alpha",
    b"final-token-alpha",
)
_MAX_REJECTION_ATTEMPTS = 4096
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,127}$")
_PHASE_CODES = {"prefill": 1, "decode": 2}
_REQUEST_KIND_CODES = {"organic": 1, "canary": 2}


def _nonce(value: bytes) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error("validator_nonce must be exactly 32 bytes")
    return value


def _identifier(value: str, name: str) -> bytes:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ProofV3Error(f"{name} is not a canonical identifier")
    return value.encode("ascii")


def _u32(value: int, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProofV3Error(f"{name} must be an unsigned 32-bit integer")
    if value < (1 if positive else 0) or value >= 1 << 32:
        qualifier = "positive " if positive else ""
        raise ProofV3Error(f"{name} must be a {qualifier}unsigned 32-bit integer")
    return value


def _u64(value: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < 1 << 64
    ):
        raise ProofV3Error(f"{name} must be an unsigned 64-bit integer")
    return value


def _nonzero_pallas_auxiliary_scalar(transcript_digest: bytes, label: bytes) -> int:
    for counter in range(_MAX_REJECTION_ATTEMPTS):
        candidate = int.from_bytes(
            hashlib.sha256(
                _PALLAS_AUXILIARY_SCALAR_DOMAIN
                + transcript_digest
                + struct.pack("<B", len(label))
                + label
                + struct.pack("<I", counter)
            ).digest(),
            "little",
        )
        if 0 < candidate < PALLAS_SCALAR_MODULUS:
            return candidate
    raise ProofV3Error("unable to derive a canonical nonzero Pallas challenge")


def _execution_transcript_digest_v3(
    *,
    validator_nonce: bytes,
    profile: ExecutionSecurityProfileV3,
    envelope: ProofV3CommitmentEnvelope,
) -> bytes:
    """Derive the one transcript root shared by all postcommit V3 choices."""

    validator_nonce = _nonce(validator_nonce)
    if not isinstance(profile, ExecutionSecurityProfileV3):
        raise ProofV3Error("profile has an unexpected type")
    if not isinstance(envelope, ProofV3CommitmentEnvelope):
        raise ProofV3Error("envelope has an unexpected type")
    profile_digest = profile.digest()
    if envelope.execution_profile_digest != profile_digest:
        raise ProofV3Error(
            "commitment envelope is bound to a different execution profile"
        )
    if envelope.static_manifest_digest != profile.static_manifest_digest:
        raise ProofV3Error(
            "commitment envelope is bound to a different static manifest"
        )
    return hashlib.sha256(
        _TRANSCRIPT_DOMAIN
        + validator_nonce
        + profile_digest
        + envelope.digest()
        + envelope.canonical_bytes()
    ).digest()


def _audit_tier_draw_from_precommit_context_v3(
    *,
    validator_nonce: bytes,
    profile: ExecutionSecurityProfileV3,
    precommit_context_digest: bytes,
    runtime_policy: RuntimeHardAuditPolicyV3,
    effective_hard_bps: int,
) -> int:
    """Derive the tier draw from validator-fixed precommit material."""

    validator_nonce = _nonce(validator_nonce)
    if not isinstance(profile, ExecutionSecurityProfileV3):
        raise ProofV3Error("audit-tier profile has an unexpected type")
    if (
        not isinstance(precommit_context_digest, bytes)
        or len(precommit_context_digest) != 32
    ):
        raise ProofV3Error(
            "audit-tier precommit context digest must be exactly 32 bytes"
        )
    if not isinstance(runtime_policy, RuntimeHardAuditPolicyV3):
        raise ProofV3Error("audit-tier runtime policy has an unexpected type")
    rate = _u32(effective_hard_bps, "audit-tier effective_hard_bps")
    if rate > 10_000:
        raise ProofV3Error("audit-tier effective_hard_bps must not exceed 10000")
    tier_abi = _identifier(
        runtime_policy.tier_selection_abi_id,
        "audit-tier selection ABI",
    )
    if runtime_policy.request_kind not in _REQUEST_KIND_CODES:
        raise ProofV3Error("audit-tier request kind is unsupported")
    tier_seed = hashlib.sha256(
        _TIER_SEED_DOMAIN
        + validator_nonce
        + profile.digest()
        + precommit_context_digest
    ).digest()
    ceiling = (1 << 256) - ((1 << 256) % 10_000)
    for counter in range(_MAX_REJECTION_ATTEMPTS):
        candidate = int.from_bytes(
            hashlib.sha256(
                _TIER_DRAW_DOMAIN
                + struct.pack("<B", len(tier_abi))
                + tier_abi
                + tier_seed
                + struct.pack(
                    "<BH",
                    _REQUEST_KIND_CODES[runtime_policy.request_kind],
                    rate,
                )
                + struct.pack("<I", counter)
            ).digest(),
            "big",
        )
        if candidate < ceiling:
            return candidate % 10_000
    raise ProofV3Error("unable to derive an unbiased postcommit audit-tier draw")


def _audit_tier_draw_basis_point_v3(
    *,
    validator_nonce: bytes,
    profile: ExecutionSecurityProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    runtime_policy: RuntimeHardAuditPolicyV3,
    effective_hard_bps: int,
) -> int:
    """Derive one unbiased 0..9999 draw independent of miner output shape.

    The nonce and validator-bound precommit context make this unknown until
    postcommit while avoiding a miner-controlled output length/root in the
    draw.  The final decision still records the complete envelope transcript
    for replay resistance and exact session binding.
    """

    if not isinstance(envelope, ProofV3CommitmentEnvelope):
        raise ProofV3Error("audit-tier envelope has an unexpected type")
    return _audit_tier_draw_from_precommit_context_v3(
        validator_nonce=validator_nonce,
        profile=profile,
        precommit_context_digest=envelope.precommit_context_digest,
        runtime_policy=runtime_policy,
        effective_hard_bps=effective_hard_bps,
    )


class _SelectionSamplerV3:
    """Unbiased deterministic sampler whose seed includes the hidden nonce."""

    def __init__(self, seed: bytes) -> None:
        if not isinstance(seed, bytes) or len(seed) != 32:
            raise ProofV3Error("hard-audit selection seed must be exactly 32 bytes")
        self._seed = seed
        self._counter = 0

    def _word(self) -> int:
        if self._counter >= 1 << 64:
            raise ProofV3Error("hard-audit selection counter overflowed")
        result = int.from_bytes(
            hashlib.sha256(
                _SELECTION_DOMAIN + self._seed + struct.pack("<Q", self._counter)
            ).digest(),
            "little",
        )
        self._counter += 1
        return result

    def below(self, bound: int) -> int:
        if isinstance(bound, bool) or not isinstance(bound, int) or bound <= 0:
            raise ProofV3Error("hard-audit selection bound must be positive")
        ceiling = (1 << 256) - ((1 << 256) % bound)
        while True:
            candidate = self._word()
            if candidate < ceiling:
                return candidate % bound

    def choose(self, population: Sequence[int], count: int) -> tuple[int, ...]:
        candidates = list(population)
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or count > len(candidates)
        ):
            raise ProofV3Error("hard-audit sample count is invalid")
        selected: list[int] = []
        for _ in range(count):
            selected.append(candidates.pop(self.below(len(candidates))))
        return tuple(selected)


@dataclass(frozen=True, slots=True)
class HardAuditLayerSelectionV3:
    """One nonce-selected layer, token window, and optional attention heads."""

    layer_index: int
    transition_kind: str
    query_row_offset: int
    query_row_count: int
    attention_head_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for value, name, positive in (
            (self.layer_index, "hard-audit layer_index", False),
            (self.query_row_offset, "hard-audit query_row_offset", False),
            (self.query_row_count, "hard-audit query_row_count", True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (1 if positive else 0)
                or value >= 1 << 32
            ):
                raise ProofV3Error(f"{name} must be an unsigned 32-bit integer")
        if self.transition_kind not in {"full_attention", "gdn"}:
            raise ProofV3Error("hard-audit transition kind is not supported")
        heads = tuple(self.attention_head_indices)
        if heads != tuple(sorted(set(heads))):
            raise ProofV3Error("hard-audit attention heads must be sorted and distinct")
        if self.transition_kind == "full_attention" and not heads:
            raise ProofV3Error("full-attention layer selections require query heads")
        if self.transition_kind == "gdn" and heads:
            raise ProofV3Error("GDN layer selections must not include query heads")
        for head in heads:
            if (
                isinstance(head, bool)
                or not isinstance(head, int)
                or not 0 <= head < 1 << 32
            ):
                raise ProofV3Error(
                    "hard-audit attention head is not an unsigned integer"
                )
        object.__setattr__(self, "attention_head_indices", heads)


@dataclass(frozen=True, slots=True)
class HardAuditSelectionV3:
    """Exact miner-independent hard transition set passed to the native adapter."""

    selection_abi_id: str
    sequence_token_count: int
    layers: tuple[HardAuditLayerSelectionV3, ...]

    def __post_init__(self) -> None:
        if self.selection_abi_id != STRATIFIED_LAYER_HEAD_SELECTION_ABI_V3:
            raise ProofV3Error("hard-audit selection ABI is not supported")
        if (
            isinstance(self.sequence_token_count, bool)
            or not isinstance(self.sequence_token_count, int)
            or not 0 < self.sequence_token_count < 1 << 32
        ):
            raise ProofV3Error("hard-audit sequence token count is invalid")
        layers = tuple(self.layers)
        if not layers or not all(
            isinstance(item, HardAuditLayerSelectionV3) for item in layers
        ):
            raise ProofV3Error("hard-audit selection layers are malformed")
        layer_indices = tuple(item.layer_index for item in layers)
        if layer_indices != tuple(sorted(layer_indices)) or len(layer_indices) != len(
            set(layer_indices)
        ):
            raise ProofV3Error("hard-audit selected layers must be sorted and distinct")
        for item in layers:
            if item.query_row_offset + item.query_row_count > self.sequence_token_count:
                raise ProofV3Error("hard-audit query range exceeds the sequence domain")
        object.__setattr__(self, "layers", layers)


@dataclass(frozen=True, slots=True)
class PostCommitAuditDecisionV3:
    """One validator-derived hard/light result after the envelope is frozen.

    The decision is intentionally separate from the selected hard-layer set.
    Every eligible request receives an indistinguishable precommit context;
    only this nonce-bound draw decides whether the retained trace must answer
    the hard opening.  A caller cannot assert the result before inference.
    """

    tier_selection_abi_id: str
    request_kind: str
    effective_hard_bps: int
    draw_basis_point: int
    hard_audit_selected: bool
    transcript_digest: bytes
    commitment_envelope_digest: bytes

    def __post_init__(self) -> None:
        if self.tier_selection_abi_id != POSTCOMMIT_AUDIT_TIER_ABI_V3:
            raise ProofV3Error("postcommit audit-tier ABI is not supported")
        if self.request_kind not in _REQUEST_KIND_CODES:
            raise ProofV3Error("postcommit audit-tier request kind is unsupported")
        rate = _u32(
            self.effective_hard_bps,
            "postcommit audit-tier effective_hard_bps",
        )
        draw = _u32(
            self.draw_basis_point,
            "postcommit audit-tier draw_basis_point",
        )
        if rate > 10_000:
            raise ProofV3Error(
                "postcommit audit-tier effective_hard_bps must not exceed 10000"
            )
        if draw >= 10_000:
            raise ProofV3Error(
                "postcommit audit-tier draw_basis_point must be below 10000"
            )
        if type(self.hard_audit_selected) is not bool:
            raise ProofV3Error(
                "postcommit audit-tier hard_audit_selected must be a boolean"
            )
        if self.hard_audit_selected != (draw < rate):
            raise ProofV3Error(
                "postcommit audit-tier result does not match its draw and rate"
            )
        if self.request_kind == "canary" and (
            rate != 10_000 or not self.hard_audit_selected
        ):
            raise ProofV3Error(
                "postcommit canary audit tier must select the hard proof"
            )
        if not isinstance(self.transcript_digest, bytes) or len(self.transcript_digest) != 32:
            raise ProofV3Error(
                "postcommit audit-tier transcript_digest must be exactly 32 bytes"
            )
        if (
            not isinstance(self.commitment_envelope_digest, bytes)
            or len(self.commitment_envelope_digest) != 32
        ):
            raise ProofV3Error(
                "postcommit audit-tier commitment_envelope_digest must be exactly 32 bytes"
            )

    def canonical_bytes(self) -> bytes:
        """Return the sole compact representation sent with a nonce reveal."""

        abi = _identifier(
            self.tier_selection_abi_id,
            "postcommit audit-tier ABI",
        )
        return (
            struct.pack("<B", len(abi))
            + abi
            + struct.pack(
                "<BHHB",
                _REQUEST_KIND_CODES[self.request_kind],
                self.effective_hard_bps,
                self.draw_basis_point,
                1 if self.hard_audit_selected else 0,
            )
            + self.transcript_digest
            + self.commitment_envelope_digest
        )

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "PostCommitAuditDecisionV3":
        """Parse one bounded, canonical decision from a nonce-reveal wire frame."""

        if not isinstance(encoded, bytes) or not encoded:
            raise ProofV3Error("postcommit audit-tier decision must be nonempty bytes")
        # one ABI-length byte, an ABI of at most 128 bytes, five fixed policy
        # bytes, and two 32-byte transcript bindings.
        if not 1 + 1 + 2 + 2 + 1 + 64 <= len(encoded) <= 1 + 128 + 6 + 64:
            raise ProofV3Error("postcommit audit-tier decision length is invalid")
        abi_length = encoded[0]
        offset = 1
        if not 1 <= abi_length <= 128 or offset + abi_length + 6 + 64 != len(encoded):
            raise ProofV3Error("postcommit audit-tier decision framing is invalid")
        try:
            abi = encoded[offset : offset + abi_length].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ProofV3Error("postcommit audit-tier ABI is not ASCII") from exc
        offset += abi_length
        request_kind_code, effective_hard_bps, draw_basis_point, hard_bit = (
            struct.unpack_from("<BHHB", encoded, offset)
        )
        offset += 6
        request_kind = next(
            (
                name
                for name, code in _REQUEST_KIND_CODES.items()
                if code == request_kind_code
            ),
            None,
        )
        if request_kind is None:
            raise ProofV3Error("postcommit audit-tier request kind is unsupported")
        if hard_bit not in (0, 1):
            raise ProofV3Error("postcommit audit-tier hard result is not canonical")
        result = cls(
            tier_selection_abi_id=abi,
            request_kind=request_kind,
            effective_hard_bps=effective_hard_bps,
            draw_basis_point=draw_basis_point,
            hard_audit_selected=bool(hard_bit),
            transcript_digest=encoded[offset : offset + 32],
            commitment_envelope_digest=encoded[offset + 32 : offset + 64],
        )
        if result.canonical_bytes() != encoded:
            raise ProofV3Error("postcommit audit-tier decision is not canonical")
        return result


def derive_postcommit_audit_decision_v3(
    *,
    validator_nonce: bytes,
    profile: ExecutionSecurityProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    runtime_policy: RuntimeHardAuditPolicyV3,
) -> PostCommitAuditDecisionV3:
    """Derive the hidden hard/light tier after a matching envelope is frozen.

    This must be called only after the validator has accepted the envelope
    while retaining ``validator_nonce`` privately.  The effective runtime
    policy is validator-owned configuration and may be stricter than the
    signed minimum, never weaker.  The result is transcript-bound to the
    nonce, exact profile, and complete envelope, so neither output length nor
    a replayed root can suppress an unfavorable selection after commitment.

    This is a deterministic transcript helper, not an envelope-admission API:
    production callers must first validate the envelope against the
    validator-owned precommit context and observed output. The session and
    public verifier do so before invoking it.
    """

    if not isinstance(profile, ExecutionSecurityProfileV3):
        raise ProofV3Error("profile has an unexpected type")
    if not isinstance(envelope, ProofV3CommitmentEnvelope):
        raise ProofV3Error("envelope has an unexpected type")
    if not isinstance(runtime_policy, RuntimeHardAuditPolicyV3):
        raise ProofV3Error("runtime hard-audit policy has an unexpected type")
    profile.relation_spec.audit_policy.validate_runtime(runtime_policy)
    transcript_digest = _execution_transcript_digest_v3(
        validator_nonce=validator_nonce,
        profile=profile,
        envelope=envelope,
    )
    effective_hard_bps = (
        runtime_policy.effective_canary_hard_bps
        if runtime_policy.request_kind == "canary"
        else runtime_policy.effective_organic_hard_bps
    )
    draw = _audit_tier_draw_basis_point_v3(
        validator_nonce=validator_nonce,
        profile=profile,
        envelope=envelope,
        runtime_policy=runtime_policy,
        effective_hard_bps=effective_hard_bps,
    )
    return PostCommitAuditDecisionV3(
        tier_selection_abi_id=runtime_policy.tier_selection_abi_id,
        request_kind=runtime_policy.request_kind,
        effective_hard_bps=effective_hard_bps,
        draw_basis_point=draw,
        hard_audit_selected=draw < effective_hard_bps,
        transcript_digest=transcript_digest,
        commitment_envelope_digest=envelope.digest(),
    )


def derive_hard_audit_selection_v3(
    *,
    transcript_digest: bytes,
    profile: ExecutionSecurityProfileV3,
    envelope: ProofV3CommitmentEnvelope,
) -> HardAuditSelectionV3:
    """Select all hard-audit indices only after the envelope is frozen.

    The selection is part of the verifier-generated challenge, never a miner
    payload.  It uses a validator nonce through ``transcript_digest`` and is
    therefore unavailable while the miner commits all runtime roots.
    """

    if not isinstance(transcript_digest, bytes) or len(transcript_digest) != 32:
        raise ProofV3Error("hard-audit transcript_digest must be exactly 32 bytes")
    if not isinstance(profile, ExecutionSecurityProfileV3):
        raise ProofV3Error("profile has an unexpected type")
    if not isinstance(envelope, ProofV3CommitmentEnvelope):
        raise ProofV3Error("envelope has an unexpected type")
    relation = profile.relation_spec
    policy = relation.audit_policy
    if policy.selection_abi_id != STRATIFIED_LAYER_HEAD_SELECTION_ABI_V3:
        raise ProofV3Error("signed hard-audit selection ABI is not supported")
    sequence_token_count = envelope.context_token_count + envelope.decode_token_count
    if not 0 < sequence_token_count < 1 << 32:
        raise ProofV3Error("hard-audit sequence token count is invalid")
    plans = tuple(relation.layer_audits)
    if policy.selected_layer_count > len(plans):
        raise ProofV3Error("signed hard-audit layer count exceeds the relation graph")
    full_attention = tuple(plan for plan in plans if plan.is_full_attention)
    gdn = tuple(plan for plan in plans if not plan.is_full_attention)
    if policy.minimum_full_attention_layers > len(full_attention):
        raise ProofV3Error("signed hard-audit full-attention coverage is impossible")
    if policy.minimum_gdn_layers > len(gdn):
        raise ProofV3Error("signed hard-audit GDN coverage is impossible")

    seed = hashlib.sha256(
        _SELECTION_DOMAIN
        + transcript_digest
        + profile.digest()
        + envelope.digest()
        + struct.pack(
            "<IIIII",
            policy.selected_layer_count,
            policy.minimum_full_attention_layers,
            policy.minimum_gdn_layers,
            policy.full_attention_heads_per_layer,
            policy.transition_query_rows,
        )
    ).digest()
    sampler = _SelectionSamplerV3(seed)
    by_layer = {plan.layer_index: plan for plan in plans}
    selected_indices = list(
        sampler.choose(
            tuple(plan.layer_index for plan in full_attention),
            policy.minimum_full_attention_layers,
        )
    )
    selected_indices.extend(
        sampler.choose(
            tuple(plan.layer_index for plan in gdn),
            policy.minimum_gdn_layers,
        )
    )
    chosen = set(selected_indices)
    remaining = tuple(
        plan.layer_index for plan in plans if plan.layer_index not in chosen
    )
    selected_indices.extend(
        sampler.choose(
            remaining,
            policy.selected_layer_count - len(selected_indices),
        )
    )
    if (
        len(selected_indices) != policy.selected_layer_count
        or len(set(selected_indices)) != policy.selected_layer_count
    ):
        raise ProofV3Error("hard-audit selection did not produce distinct layers")

    rows = min(policy.transition_query_rows, sequence_token_count)
    selections: list[HardAuditLayerSelectionV3] = []
    for layer_index in sorted(selected_indices):
        plan: LayerAuditPlanV3 = by_layer[layer_index]
        row_offset = sampler.below(sequence_token_count - rows + 1)
        if plan.is_full_attention:
            heads = tuple(
                sorted(
                    sampler.choose(
                        tuple(range(plan.attention_query_head_count)),
                        policy.full_attention_heads_per_layer,
                    )
                )
            )
            selections.append(
                HardAuditLayerSelectionV3(
                    layer_index=layer_index,
                    transition_kind="full_attention",
                    query_row_offset=row_offset,
                    query_row_count=rows,
                    attention_head_indices=heads,
                )
            )
        else:
            selections.append(
                HardAuditLayerSelectionV3(
                    layer_index=layer_index,
                    transition_kind="gdn",
                    query_row_offset=row_offset,
                    query_row_count=rows,
                )
            )
    return HardAuditSelectionV3(
        selection_abi_id=policy.selection_abi_id,
        sequence_token_count=sequence_token_count,
        layers=tuple(selections),
    )


@dataclass(frozen=True, slots=True)
class ExpectedExecutionChunkV3:
    """One verifier-derived prefill/decode chunk that cannot be miner chosen.

    The native adapter must establish its global relation for every such chunk.
    A chunk receipt from the miner is allowed to carry roots for this exact
    coordinate only; it must never define the coverage universe itself.
    """

    phase: str
    chunk_index: int
    logical_token_start: int
    token_count: int

    def __post_init__(self) -> None:
        if self.phase not in _PHASE_CODES:
            raise ProofV3Error("expected execution chunk phase is not supported")
        _u32(self.chunk_index, "expected execution chunk_index")
        start = _u32(
            self.logical_token_start,
            "expected execution logical_token_start",
        )
        count = _u32(
            self.token_count,
            "expected execution token_count",
            positive=True,
        )
        if start + count > 1 << 32:
            raise ProofV3Error("expected execution chunk token range overflows")

    def canonical_bytes(self) -> bytes:
        return struct.pack(
            "<BIII",
            _PHASE_CODES[self.phase],
            self.chunk_index,
            self.logical_token_start,
            self.token_count,
        )


def expected_execution_chunks_v3(
    *,
    profile: ExecutionSecurityProfileV3,
    envelope: ProofV3CommitmentEnvelope,
) -> tuple[ExpectedExecutionChunkV3, ...]:
    """Derive the complete pre-nonce chunk universe from validator-bound counts.

    Fixed signed chunk sizes make the full coverage set independent of miner
    payload data.  The final partial chunk is also deterministic.  This helper
    intentionally returns only fixed-size coordinates, never a tensor root,
    cache page, or activation.
    """

    if not isinstance(profile, ExecutionSecurityProfileV3):
        raise ProofV3Error("profile has an unexpected type")
    if not isinstance(envelope, ProofV3CommitmentEnvelope):
        raise ProofV3Error("envelope has an unexpected type")
    profile.require_hard_execution_capability()
    if envelope.execution_profile_digest != profile.digest():
        raise ProofV3Error("execution chunk universe has an unexpected profile")
    if envelope.static_manifest_digest != profile.static_manifest_digest:
        raise ProofV3Error("execution chunk universe has an unexpected static manifest")
    relation = profile.relation_spec
    try:
        relation.validate_execution_counts(
            context_token_count=envelope.context_token_count,
            decode_token_count=envelope.decode_token_count,
        )
    except ProofV3Error:
        raise
    except Exception as exc:
        raise ProofV3Error("execution chunk universe has invalid token counts") from exc

    chunks: list[ExpectedExecutionChunkV3] = []
    chunk_index = 0
    for phase, logical_start, token_count, chunk_size in (
        (
            "prefill",
            0,
            envelope.context_token_count,
            relation.prefill_chunk_tokens,
        ),
        (
            "decode",
            envelope.context_token_count,
            envelope.decode_token_count,
            relation.decode_chunk_tokens,
        ),
    ):
        for offset in range(0, token_count, chunk_size):
            count = min(chunk_size, token_count - offset)
            chunks.append(
                ExpectedExecutionChunkV3(
                    phase=phase,
                    chunk_index=chunk_index,
                    logical_token_start=logical_start + offset,
                    token_count=count,
                )
            )
            chunk_index += 1
    return tuple(chunks)


@dataclass(frozen=True, slots=True)
class GlobalRelationCoordinateV3:
    """One atomically indexed global relation selected after precommitment.

    ``relation_index`` is a constraint-system-defined atomic constraint/table
    coordinate.  Its exact mapping is authenticated by the signed constraint
    system artifact; it must not be a prover-defined label.  The native
    adapter derives one field coefficient per such coordinate rather than
    folding an entire relation family under a single scalar.
    """

    node_id: str
    phase: str
    chunk_index: int
    logical_token_start: int
    token_count: int
    relation_index: int

    def __post_init__(self) -> None:
        _identifier(self.node_id, "global relation node_id")
        if self.phase not in _PHASE_CODES:
            raise ProofV3Error("global relation phase is not supported")
        start = _u32(
            self.logical_token_start,
            "global relation logical_token_start",
        )
        count = _u32(
            self.token_count,
            "global relation token_count",
            positive=True,
        )
        if start + count > 1 << 32:
            raise ProofV3Error("global relation token range overflows")
        _u32(self.chunk_index, "global relation chunk_index")
        _u64(self.relation_index, "global relation relation_index")

    def canonical_bytes(self) -> bytes:
        node_id = _identifier(self.node_id, "global relation node_id")
        return (
            struct.pack("<B", len(node_id))
            + node_id
            + struct.pack(
                "<BIIIQ",
                _PHASE_CODES[self.phase],
                self.chunk_index,
                self.logical_token_start,
                self.token_count,
                self.relation_index,
            )
        )


def _validate_global_relation_coordinate(
    *,
    profile: ExecutionSecurityProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    coordinate: GlobalRelationCoordinateV3,
) -> None:
    if not isinstance(coordinate, GlobalRelationCoordinateV3):
        raise ProofV3Error("global relation coordinate has an unexpected type")
    node_ids = {node.node_id for node in profile.relation_spec.nodes}
    if coordinate.node_id not in node_ids:
        raise ProofV3Error("global relation coordinate has an undeclared graph node")
    expected_chunks = expected_execution_chunks_v3(
        profile=profile,
        envelope=envelope,
    )
    if coordinate.chunk_index >= len(expected_chunks):
        raise ProofV3Error("global relation coordinate has an out-of-range chunk")
    expected = expected_chunks[coordinate.chunk_index]
    if (
        coordinate.phase != expected.phase
        or coordinate.logical_token_start != expected.logical_token_start
        or coordinate.token_count != expected.token_count
    ):
        raise ProofV3Error(
            "global relation coordinate does not match the verifier-derived chunk"
        )


def derive_pallas_affine_fold_scalar_v3(
    *,
    challenge: "FoldedExecutionChallengeV3",
    profile: ExecutionSecurityProfileV3,
    envelope: ProofV3CommitmentEnvelope,
    coordinate: GlobalRelationCoordinateV3,
) -> int:
    """Derive one Pallas scalar for the bounded affine-transition primitive.

    This helper is deliberately *not* a generic global-execution coefficient
    interface.  A future dynamic AIR/RAM backend must derive coefficients in
    its signed backend field from ``challenge.global_relation_seed`` and the
    canonical coordinate under its own domain/ABI. In particular, it must not
    reduce this Pallas scalar into Goldilocks or another dynamic field.

    The five historical category alphas on
    :class:`FoldedExecutionChallengeV3` are Pallas auxiliary compatibility
    values and are not sound as the sole folds for all rows/tables in a global
    execution relation: unrelated residuals could otherwise cancel under one
    scalar.
    """

    if not isinstance(challenge, FoldedExecutionChallengeV3):
        raise ProofV3Error("folded execution challenge has an unexpected type")
    _validate_global_relation_coordinate(
        profile=profile,
        envelope=envelope,
        coordinate=coordinate,
    )
    encoded = coordinate.canonical_bytes()
    for counter in range(_MAX_REJECTION_ATTEMPTS):
        candidate = int.from_bytes(
            hashlib.sha256(
                _PALLAS_AFFINE_FOLD_SCALAR_DOMAIN
                + challenge.global_relation_seed
                + encoded
                + struct.pack("<I", counter)
            ).digest(),
            "little",
        )
        if 0 < candidate < PALLAS_SCALAR_MODULUS:
            return candidate
    raise ProofV3Error("unable to derive a canonical Pallas affine-fold scalar")


@dataclass(frozen=True, slots=True)
class FoldedExecutionChallengeV3:
    """Field-neutral global seed plus Pallas auxiliary audit coefficients."""

    transcript_digest: bytes
    global_relation_seed: bytes
    linear_alpha: int
    bridge_alpha: int
    attention_alpha: int
    prefill_alpha: int
    final_token_alpha: int
    hard_audit_selection: HardAuditSelectionV3

    def __post_init__(self) -> None:
        if (
            not isinstance(self.transcript_digest, bytes)
            or len(self.transcript_digest) != 32
        ):
            raise ProofV3Error("transcript_digest must be exactly 32 bytes")
        if (
            not isinstance(self.global_relation_seed, bytes)
            or len(self.global_relation_seed) != 32
        ):
            raise ProofV3Error("global_relation_seed must be exactly 32 bytes")
        for name in (
            "linear_alpha",
            "bridge_alpha",
            "attention_alpha",
            "prefill_alpha",
            "final_token_alpha",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 < value < PALLAS_SCALAR_MODULUS
            ):
                raise ProofV3Error(f"{name} must be a nonzero Pallas scalar")
        if not isinstance(self.hard_audit_selection, HardAuditSelectionV3):
            raise ProofV3Error("hard_audit_selection has an unexpected type")


def derive_folded_execution_challenge_v3(
    *,
    validator_nonce: bytes,
    profile: ExecutionSecurityProfileV3,
    envelope: ProofV3CommitmentEnvelope,
) -> FoldedExecutionChallengeV3:
    """Derive every hard-audit coefficient after all trace roots are frozen."""

    transcript_digest = _execution_transcript_digest_v3(
        validator_nonce=validator_nonce,
        profile=profile,
        envelope=envelope,
    )
    coefficients = tuple(
        _nonzero_pallas_auxiliary_scalar(transcript_digest, label) for label in _LABELS
    )
    global_relation_seed = hashlib.sha256(
        _GLOBAL_RELATION_SEED_DOMAIN
        + transcript_digest
        + profile.relation_spec.constraint_system_digest
        + profile.relation_spec.recursive_accumulator_digest
        + envelope.execution_root
        + envelope.cache_table_root
    ).digest()
    selection = derive_hard_audit_selection_v3(
        transcript_digest=transcript_digest,
        profile=profile,
        envelope=envelope,
    )
    return FoldedExecutionChallengeV3(
        transcript_digest=transcript_digest,
        global_relation_seed=global_relation_seed,
        linear_alpha=coefficients[0],
        bridge_alpha=coefficients[1],
        attention_alpha=coefficients[2],
        prefill_alpha=coefficients[3],
        final_token_alpha=coefficients[4],
        hard_audit_selection=selection,
    )


__all__ = [
    "FoldedExecutionChallengeV3",
    "ExpectedExecutionChunkV3",
    "GlobalRelationCoordinateV3",
    "HardAuditLayerSelectionV3",
    "HardAuditSelectionV3",
    "PostCommitAuditDecisionV3",
    "derive_postcommit_audit_decision_v3",
    "derive_pallas_affine_fold_scalar_v3",
    "expected_execution_chunks_v3",
    "derive_hard_audit_selection_v3",
    "derive_folded_execution_challenge_v3",
]
