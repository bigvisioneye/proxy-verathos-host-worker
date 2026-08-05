"""Canonical validator-owned request and output binding for proof-v3.

The serving trace begins before output tokens exist.  V3 therefore separates a
validator-issued pre-request context from the final observed output binding:

* the miner binds the pre-request context while it captures prefill/decode;
* after the last visible token, it seals roots together with output roots; and
* the validator compares both public contexts before revealing its nonce.

No function here reads transport state. Runtime integration must construct the
input context from its pinned tokenizer/template and the output context from
the observed canonical token stream, never from miner-provided text or IDs.
"""

from __future__ import annotations

import hashlib
import struct
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error


MAX_EXECUTION_REQUEST_TOKENS_V3 = 1 << 24
MAX_OUTPUT_STREAM_BYTES_V3 = 16 << 20
MAX_FINISH_REASON_BYTES_V3 = 256
PREEXECUTION_CONTEXT_FORMAT_VERSION_V3 = 1
PREEXECUTION_CONTEXT_BYTES_V3 = 4 + 2 + 9 * 32 + 4
_TOKEN_SEQUENCE_DOMAIN = b"VERATHOS/PROOF_V3/TOKEN_SEQUENCE/SHA256"
_PRE_NONCE_CONTEXT_DOMAIN = b"VERATHOS/PROOF_V3/PRE_NONCE_CONTEXT/SHA256"
_NONCE_COMMITMENT_DOMAIN = b"VERATHOS/PROOF_V3/NONCE_COMMITMENT/SHA256"
_PRECOMMIT_CONTEXT_DOMAIN = b"VERATHOS/PROOF_V3/PRECOMMIT_CONTEXT/SHA256"
_INITIAL_EXECUTION_STATE_DOMAIN = b"VERATHOS/PROOF_V3/INITIAL_EXECUTION_STATE/SHA256"
_OUTPUT_STREAM_DOMAIN = b"VERATHOS/PROOF_V3/OUTPUT_STREAM/SHA256"
_REQUEST_DIGEST_DOMAIN = b"VERATHOS/PROOF_V3/REQUEST_CONTEXT/SHA256"
_NETWORK_HOTKEY_IDENTITY_DOMAIN = (
    b"VERATHOS/PROOF_V3/NETWORK_HOTKEY_IDENTITY/V1/SHA256"
)
_PREEXECUTION_CONTEXT_MAGIC_V3 = b"V3PC"


def _fixed32(value: bytes, name: str, *, nonzero: bool = False) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ProofV3Error(f"{name} must be exactly 32 bytes")
    if nonzero and value == bytes(32):
        raise ProofV3Error(f"{name} must not be the zero digest")
    return value


def _positive_u32(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProofV3Error(f"{name} must be a positive unsigned 32-bit integer")
    if not 0 < value < 1 << 32:
        raise ProofV3Error(f"{name} must be a positive unsigned 32-bit integer")
    return value


def _u32(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProofV3Error(f"{name} must be an unsigned 32-bit integer")
    if not 0 <= value < 1 << 32:
        raise ProofV3Error(f"{name} must be an unsigned 32-bit integer")
    return value


def execution_input_token_id_at_position_v3(
    *,
    prompt_token_ids: Sequence[int],
    observed_output_token_ids: Sequence[int],
    sequence_position: int,
) -> int:
    """Return the token embedded at one captured execution position.

    Prefill positions consume the prompt token at the same position. Decode
    position ``context`` consumes output token zero: the token returned by the
    preceding decode step. The final returned token has no captured successor
    row and is therefore outside the valid execution-position range.
    """

    context_count = len(prompt_token_ids)
    decode_count = len(observed_output_token_ids)
    if context_count <= 0 or decode_count <= 0:
        raise ProofV3Error(
            "execution input-token mapping requires prompt and output tokens"
        )
    if (
        isinstance(sequence_position, bool)
        or not isinstance(sequence_position, int)
        or sequence_position < 0
        or sequence_position >= context_count + decode_count - 1
    ):
        raise ProofV3Error("execution input-token position is out of range")

    if sequence_position < context_count:
        token = prompt_token_ids[sequence_position]
    else:
        token = observed_output_token_ids[sequence_position - context_count]
    return _u32(token, "execution input token id")


def _frame(value: bytes, name: str, maximum: int) -> bytes:
    if not isinstance(value, bytes) or len(value) > maximum:
        raise ProofV3Error(f"{name} byte length is out of range")
    return struct.pack("<I", len(value)) + value


def _finish_reason_bytes(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ProofV3Error("finish_reason must be a non-empty string")
    if value != unicodedata.normalize("NFC", value) or "\x00" in value:
        raise ProofV3Error("finish_reason is not canonical text")
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_FINISH_REASON_BYTES_V3:
        raise ProofV3Error("finish_reason exceeds the protocol limit")
    return encoded


def _network_hotkey_identity_digest_v3(*, role: bytes, hotkey_ss58: str) -> bytes:
    if role not in (b"validator", b"miner"):
        raise ProofV3Error("proof-v3 network identity role is unsupported")
    if not isinstance(hotkey_ss58, str) or not hotkey_ss58:
        raise ProofV3Error("proof-v3 network hotkey must be non-empty text")
    if hotkey_ss58 != hotkey_ss58.strip() or "\x00" in hotkey_ss58:
        raise ProofV3Error("proof-v3 network hotkey is not canonical text")
    try:
        encoded = hotkey_ss58.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ProofV3Error("proof-v3 network hotkey must be ASCII") from exc
    if len(encoded) > 128:
        raise ProofV3Error("proof-v3 network hotkey exceeds 128 bytes")
    return hashlib.sha256(
        _NETWORK_HOTKEY_IDENTITY_DOMAIN
        + struct.pack("<B", len(role))
        + role
        + struct.pack("<H", len(encoded))
        + encoded
    ).digest()


def validator_hotkey_identity_digest_v3(hotkey_ss58: str) -> bytes:
    """Bind an authenticated validator SS58 hotkey to a request context."""

    return _network_hotkey_identity_digest_v3(
        role=b"validator",
        hotkey_ss58=hotkey_ss58,
    )


def miner_hotkey_identity_digest_v3(hotkey_ss58: str) -> bytes:
    """Bind the selected miner SS58 hotkey to a request context."""

    return _network_hotkey_identity_digest_v3(
        role=b"miner",
        hotkey_ss58=hotkey_ss58,
    )


def _token_sequence_count(token_ids: Sequence[int], name: str) -> int:
    """Validate a token sequence container and return its bounded length."""

    if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
        raise ProofV3Error(f"{name} must be a token-id sequence")
    count = len(token_ids)
    if count > MAX_EXECUTION_REQUEST_TOKENS_V3:
        raise ProofV3Error(f"{name} exceeds the proof-v3 token limit")
    return count


def canonical_tokenizer_token_ids_v3(
    value: object,
    *,
    name: str = "prompt_token_ids",
) -> tuple[int, ...]:
    """Normalize one unbatched tokenizer result to canonical uint32 IDs.

    ``transformers`` may return either a plain sequence or a ``BatchEncoding``
    mapping for a single chat.  Only the mapping's exact ``input_ids`` value is
    relevant to execution; nested/batched results remain invalid.
    """

    if isinstance(value, Mapping):
        value = value.get("input_ids")
    count = _token_sequence_count(value, name)  # type: ignore[arg-type]
    if count == 0:
        raise ProofV3Error(f"{name} must not be empty")
    result: list[int] = []
    for index, token_id in enumerate(value):  # type: ignore[union-attr]
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ProofV3Error(
                f"{name}[{index}] must be an unsigned token id"
            )
        if not 0 <= token_id < 1 << 32:
            raise ProofV3Error(
                f"{name}[{index}] is outside the uint32 range"
            )
        result.append(token_id)
    return tuple(result)


def token_sequence_digest_v3(
    token_ids: Sequence[int],
    *,
    name: str,
) -> bytes:
    """Hash one canonical token-id sequence without retaining another copy."""

    count = _token_sequence_count(token_ids, name)
    digest = hashlib.sha256()
    digest.update(_TOKEN_SEQUENCE_DOMAIN)
    digest.update(struct.pack("<I", count))
    for index, token_id in enumerate(token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ProofV3Error(f"{name}[{index}] must be an unsigned token id")
        if not 0 <= token_id < 1 << 32:
            raise ProofV3Error(f"{name}[{index}] is outside the uint32 range")
        digest.update(struct.pack("<I", token_id))
    return digest.digest()


def _pre_nonce_context_digest_from_root(
    *,
    proof_challenge_id: bytes,
    validator_identity_digest: bytes,
    miner_identity_digest: bytes,
    static_manifest_digest: bytes,
    execution_profile_digest: bytes,
    prompt_token_root: bytes,
    context_token_count: int,
    cache_lease_digest: bytes,
    sampler_config_digest: bytes,
) -> bytes:
    return hashlib.sha256(
        _PRE_NONCE_CONTEXT_DOMAIN
        + _fixed32(proof_challenge_id, "proof_challenge_id")
        + _fixed32(validator_identity_digest, "validator_identity_digest")
        + _fixed32(miner_identity_digest, "miner_identity_digest")
        + _fixed32(static_manifest_digest, "static_manifest_digest")
        + _fixed32(execution_profile_digest, "execution_profile_digest")
        + _fixed32(prompt_token_root, "prompt_token_root")
        + struct.pack(
            "<I",
            _positive_u32(context_token_count, "context_token_count"),
        )
        + _fixed32(cache_lease_digest, "cache_lease_digest", nonzero=True)
        + _fixed32(sampler_config_digest, "sampler_config_digest")
    ).digest()


def derive_pre_nonce_context_digest_v3(
    *,
    proof_challenge_id: bytes,
    validator_identity_digest: bytes,
    miner_identity_digest: bytes,
    static_manifest_digest: bytes,
    execution_profile_digest: bytes,
    prompt_token_ids: Sequence[int],
    cache_lease_digest: bytes,
    sampler_config_digest: bytes,
) -> bytes:
    """Derive the context that a validator commits its hidden nonce to."""

    context_count = _positive_u32(
        _token_sequence_count(prompt_token_ids, "prompt_token_ids"),
        "context_token_count",
    )
    return _pre_nonce_context_digest_from_root(
        proof_challenge_id=proof_challenge_id,
        validator_identity_digest=validator_identity_digest,
        miner_identity_digest=miner_identity_digest,
        static_manifest_digest=static_manifest_digest,
        execution_profile_digest=execution_profile_digest,
        prompt_token_root=token_sequence_digest_v3(
            prompt_token_ids,
            name="prompt_token_ids",
        ),
        context_token_count=context_count,
        cache_lease_digest=cache_lease_digest,
        sampler_config_digest=sampler_config_digest,
    )


def output_stream_digest_v3(
    *,
    emitted_text_utf8: bytes,
    finish_reason: str,
) -> bytes:
    """Bind exactly the streamed UTF-8 bytes and terminal finish reason."""

    return hashlib.sha256(
        _OUTPUT_STREAM_DOMAIN
        + _frame(
            emitted_text_utf8,
            "emitted_text_utf8",
            MAX_OUTPUT_STREAM_BYTES_V3,
        )
        + _frame(
            _finish_reason_bytes(finish_reason),
            "finish_reason",
            MAX_FINISH_REASON_BYTES_V3,
        )
    ).digest()


@dataclass(frozen=True, slots=True)
class PreExecutionRequestContextV3:
    """All validator-issued public context known before model execution."""

    proof_challenge_id: bytes
    validator_identity_digest: bytes
    miner_identity_digest: bytes
    validator_nonce_commitment: bytes
    static_manifest_digest: bytes
    execution_profile_digest: bytes
    prompt_token_root: bytes
    context_token_count: int
    cache_lease_digest: bytes
    sampler_config_digest: bytes

    def __post_init__(self) -> None:
        _fixed32(self.proof_challenge_id, "proof_challenge_id")
        _fixed32(self.validator_identity_digest, "validator_identity_digest")
        _fixed32(self.miner_identity_digest, "miner_identity_digest")
        _fixed32(self.validator_nonce_commitment, "validator_nonce_commitment")
        _fixed32(self.static_manifest_digest, "static_manifest_digest")
        _fixed32(self.execution_profile_digest, "execution_profile_digest")
        _fixed32(self.prompt_token_root, "prompt_token_root")
        _positive_u32(self.context_token_count, "context_token_count")
        _fixed32(self.cache_lease_digest, "cache_lease_digest", nonzero=True)
        _fixed32(self.sampler_config_digest, "sampler_config_digest")

    def nonce_context_digest(self) -> bytes:
        """Return the request context that a hidden nonce commits to."""

        return _pre_nonce_context_digest_from_root(
            proof_challenge_id=self.proof_challenge_id,
            validator_identity_digest=self.validator_identity_digest,
            miner_identity_digest=self.miner_identity_digest,
            static_manifest_digest=self.static_manifest_digest,
            execution_profile_digest=self.execution_profile_digest,
            prompt_token_root=self.prompt_token_root,
            context_token_count=self.context_token_count,
            cache_lease_digest=self.cache_lease_digest,
            sampler_config_digest=self.sampler_config_digest,
        )

    def digest(self) -> bytes:
        """Return the immutable precommit context absorbed by the trace."""

        return hashlib.sha256(
            _PRECOMMIT_CONTEXT_DOMAIN
            + self.nonce_context_digest()
            + self.validator_nonce_commitment
        ).digest()

    def canonical_bytes(self) -> bytes:
        """Return the fixed-size initial request frame sent to the miner."""

        return (
            _PREEXECUTION_CONTEXT_MAGIC_V3
            + struct.pack("<H", PREEXECUTION_CONTEXT_FORMAT_VERSION_V3)
            + self.proof_challenge_id
            + self.validator_identity_digest
            + self.miner_identity_digest
            + self.validator_nonce_commitment
            + self.static_manifest_digest
            + self.execution_profile_digest
            + self.prompt_token_root
            + struct.pack("<I", self.context_token_count)
            + self.cache_lease_digest
            + self.sampler_config_digest
        )

    @classmethod
    def from_canonical_bytes(
        cls,
        encoded: bytes,
    ) -> "PreExecutionRequestContextV3":
        """Parse exactly one bounded initial request frame."""

        if (
            not isinstance(encoded, bytes)
            or len(encoded) != PREEXECUTION_CONTEXT_BYTES_V3
        ):
            raise ProofV3Error(
                "proof-v3 preexecution context byte length is invalid"
            )
        if encoded[:4] != _PREEXECUTION_CONTEXT_MAGIC_V3:
            raise ProofV3Error(
                "proof-v3 preexecution context header is unsupported"
            )
        version = struct.unpack_from("<H", encoded, 4)[0]
        if version != PREEXECUTION_CONTEXT_FORMAT_VERSION_V3:
            raise ProofV3Error(
                "proof-v3 preexecution context version is unsupported"
            )
        offset = 6

        def read_digest() -> bytes:
            nonlocal offset
            result = encoded[offset : offset + 32]
            offset += 32
            return result

        result = cls(
            proof_challenge_id=read_digest(),
            validator_identity_digest=read_digest(),
            miner_identity_digest=read_digest(),
            validator_nonce_commitment=read_digest(),
            static_manifest_digest=read_digest(),
            execution_profile_digest=read_digest(),
            prompt_token_root=read_digest(),
            context_token_count=struct.unpack_from("<I", encoded, offset)[0],
            cache_lease_digest=encoded[offset + 4 : offset + 36],
            sampler_config_digest=encoded[offset + 36 : offset + 68],
        )
        if result.canonical_bytes() != encoded:
            raise ProofV3Error(
                "proof-v3 preexecution context is not canonical"
            )
        return result

    def initial_execution_state_anchor(self) -> bytes:
        """Return the validator-bound synthetic entry for the first trace chunk.

        This is a public input to the native execution relation, not a claim
        that the hash itself is an embedding activation. A qualified AIR must
        prove the prompt/embedding transition from this anchor into the first
        runtime state; the root-only lifecycle uses it only to prevent an
        unanchored or cross-request chunk-zero receipt.
        """

        return hashlib.sha256(_INITIAL_EXECUTION_STATE_DOMAIN + self.digest()).digest()

    def bind_output(
        self,
        output: "ExecutionOutputBindingV3",
    ) -> "ExecutionRequestBindingV3":
        """Join an observed output binding to this pre-request context."""

        if not isinstance(output, ExecutionOutputBindingV3):
            raise ProofV3Error("output binding has an unexpected type")
        return ExecutionRequestBindingV3(
            precommit_context=self,
            output=output,
            request_digest=derive_execution_request_digest_v3(
                precommit_context=self,
                output=output,
            ),
        )


def commit_validator_nonce_v3(
    *,
    validator_nonce: bytes,
    nonce_context_digest: bytes,
) -> bytes:
    """Commit the unrevealed nonce to one validator-issued request context."""

    nonce = _fixed32(validator_nonce, "validator_nonce")
    context = _fixed32(nonce_context_digest, "nonce_context_digest")
    return hashlib.sha256(_NONCE_COMMITMENT_DOMAIN + nonce + context).digest()


@dataclass(frozen=True, slots=True)
class ExecutionOutputBindingV3:
    """Observed output token/text commitment sealed after the final token."""

    output_token_root: bytes
    output_stream_digest: bytes
    decode_token_count: int

    def __post_init__(self) -> None:
        _fixed32(self.output_token_root, "output_token_root")
        _fixed32(self.output_stream_digest, "output_stream_digest")
        _u32(self.decode_token_count, "decode_token_count")

    @classmethod
    def from_observed_output(
        cls,
        *,
        output_token_ids: Sequence[int],
        emitted_text_utf8: bytes,
        finish_reason: str,
    ) -> "ExecutionOutputBindingV3":
        """Derive one output binding from the observed SSE result."""

        _token_sequence_count(output_token_ids, "output_token_ids")
        return cls(
            output_token_root=token_sequence_digest_v3(
                output_token_ids,
                name="output_token_ids",
            ),
            output_stream_digest=output_stream_digest_v3(
                emitted_text_utf8=emitted_text_utf8,
                finish_reason=finish_reason,
            ),
            decode_token_count=_u32(
                len(output_token_ids),
                "decode_token_count",
            ),
        )


@dataclass(frozen=True, slots=True)
class ExecutionRequestBindingV3:
    """Complete public request/output statement carried by an envelope."""

    precommit_context: PreExecutionRequestContextV3
    output: ExecutionOutputBindingV3
    request_digest: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.precommit_context, PreExecutionRequestContextV3):
            raise ProofV3Error("precommit context has an unexpected type")
        if not isinstance(self.output, ExecutionOutputBindingV3):
            raise ProofV3Error("output binding has an unexpected type")
        _fixed32(self.request_digest, "request_digest")
        expected = derive_execution_request_digest_v3(
            precommit_context=self.precommit_context,
            output=self.output,
        )
        if self.request_digest != expected:
            raise ProofV3Error("request digest does not match its public context")


def derive_execution_request_digest_v3(
    *,
    precommit_context: PreExecutionRequestContextV3,
    output: ExecutionOutputBindingV3,
) -> bytes:
    """Hash the final public request/output statement without raw tensors."""

    if not isinstance(precommit_context, PreExecutionRequestContextV3):
        raise ProofV3Error("precommit context has an unexpected type")
    if not isinstance(output, ExecutionOutputBindingV3):
        raise ProofV3Error("output binding has an unexpected type")
    return hashlib.sha256(
        _REQUEST_DIGEST_DOMAIN
        + precommit_context.digest()
        + output.output_token_root
        + output.output_stream_digest
        + struct.pack("<I", output.decode_token_count)
    ).digest()


@dataclass(frozen=True, slots=True)
class ValidatorExecutionRequestContextV3:
    """Validator-owned input used to derive a pre-request context.

    Prompt IDs must come from the validator's pinned tokenizer/template. The
    caller must bind ``validator_nonce_commitment`` to
    :meth:`nonce_context_digest` before sending this context to the miner.
    """

    proof_challenge_id: bytes
    validator_identity_digest: bytes
    miner_identity_digest: bytes
    validator_nonce_commitment: bytes
    prompt_token_ids: Sequence[int]
    cache_lease_digest: bytes
    sampler_config_digest: bytes

    def __post_init__(self) -> None:
        _fixed32(self.proof_challenge_id, "proof_challenge_id")
        _fixed32(self.validator_identity_digest, "validator_identity_digest")
        _fixed32(self.miner_identity_digest, "miner_identity_digest")
        _fixed32(self.validator_nonce_commitment, "validator_nonce_commitment")
        _fixed32(self.cache_lease_digest, "cache_lease_digest", nonzero=True)
        _fixed32(self.sampler_config_digest, "sampler_config_digest")

    def prompt_token_count(self) -> int:
        """Validate and return the exact canonical prompt length."""

        return _positive_u32(
            _token_sequence_count(self.prompt_token_ids, "prompt_token_ids"),
            "context_token_count",
        )

    def nonce_context_digest(
        self,
        *,
        static_manifest_digest: bytes,
        execution_profile_digest: bytes,
    ) -> bytes:
        """Return the context to bind when committing the hidden nonce."""

        self.prompt_token_count()
        return derive_pre_nonce_context_digest_v3(
            proof_challenge_id=self.proof_challenge_id,
            validator_identity_digest=self.validator_identity_digest,
            miner_identity_digest=self.miner_identity_digest,
            static_manifest_digest=static_manifest_digest,
            execution_profile_digest=execution_profile_digest,
            prompt_token_ids=self.prompt_token_ids,
            cache_lease_digest=self.cache_lease_digest,
            sampler_config_digest=self.sampler_config_digest,
        )

    def derive_precommit_context(
        self,
        *,
        static_manifest_digest: bytes,
        execution_profile_digest: bytes,
    ) -> PreExecutionRequestContextV3:
        """Derive prefill-visible roots/counts without copying token IDs."""

        context_count = self.prompt_token_count()
        static_manifest = _fixed32(static_manifest_digest, "static_manifest_digest")
        profile = _fixed32(execution_profile_digest, "execution_profile_digest")
        prompt_root = token_sequence_digest_v3(
            self.prompt_token_ids,
            name="prompt_token_ids",
        )
        result = PreExecutionRequestContextV3(
            proof_challenge_id=self.proof_challenge_id,
            validator_identity_digest=self.validator_identity_digest,
            miner_identity_digest=self.miner_identity_digest,
            validator_nonce_commitment=self.validator_nonce_commitment,
            static_manifest_digest=static_manifest,
            execution_profile_digest=profile,
            prompt_token_root=prompt_root,
            context_token_count=context_count,
            cache_lease_digest=self.cache_lease_digest,
            sampler_config_digest=self.sampler_config_digest,
        )
        if result.nonce_context_digest() != self.nonce_context_digest(
            static_manifest_digest=static_manifest,
            execution_profile_digest=profile,
        ):
            raise ProofV3Error(
                "validator pre-request context derivation is inconsistent"
            )
        return result


@dataclass(frozen=True, slots=True)
class ObservedExecutionOutputV3:
    """Validator-observed canonical output used at final envelope checking."""

    output_token_ids: Sequence[int]
    emitted_text_utf8: bytes
    finish_reason: str

    def output_token_count(self) -> int:
        """Validate and return the exact canonical returned-token count."""

        return _u32(
            _token_sequence_count(self.output_token_ids, "output_token_ids"),
            "decode_token_count",
        )

    def derive_output_binding(self) -> ExecutionOutputBindingV3:
        """Return exact public output roots/count from the observed stream."""

        self.output_token_count()
        return ExecutionOutputBindingV3.from_observed_output(
            output_token_ids=self.output_token_ids,
            emitted_text_utf8=self.emitted_text_utf8,
            finish_reason=self.finish_reason,
        )


__all__ = [
    "MAX_EXECUTION_REQUEST_TOKENS_V3",
    "PREEXECUTION_CONTEXT_BYTES_V3",
    "PREEXECUTION_CONTEXT_FORMAT_VERSION_V3",
    "ExecutionOutputBindingV3",
    "ExecutionRequestBindingV3",
    "ObservedExecutionOutputV3",
    "PreExecutionRequestContextV3",
    "ValidatorExecutionRequestContextV3",
    "commit_validator_nonce_v3",
    "canonical_tokenizer_token_ids_v3",
    "derive_pre_nonce_context_digest_v3",
    "derive_execution_request_digest_v3",
    "miner_hotkey_identity_digest_v3",
    "output_stream_digest_v3",
    "token_sequence_digest_v3",
    "validator_hotkey_identity_digest_v3",
]
