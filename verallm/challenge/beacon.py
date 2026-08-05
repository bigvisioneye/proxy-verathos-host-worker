"""
Beacon-based challenge derivation.

Deterministically derives which layers/operations to verify
from a random beacon and the inference commitment.
"""

from typing import List, Tuple, Optional
import hashlib
import struct
import random

from verallm.types import (
    InferenceCommitment,
    SamplingChallenge,
    EmbeddingChallenge,
    ChallengeSet,
    LayerChallenge,
    GEMMChallenge,
)
from verallm.sampling import clamp_sampling_bps, HIGH_ASSURANCE_BPS
from verallm.config import get_config

# Re-export pure functions from zkllm
from zkllm.challenge.beacon import (  # noqa: F401
    derive_beacon,
    derive_beacon_from_nonce,
    compute_detection_probability,
)


def derive_challenges(
    beacon: bytes,
    commitment: InferenceCommitment,
    k_layers: Optional[int] = None,
    k_gemms_per_layer: int = 2,
    k_blocks_per_gemm: int = 4,
) -> ChallengeSet:
    """
    Derive deterministic challenges from beacon and commitment.

    This is reproducible by both prover and verifier.

    Args:
        beacon: Random 32-byte beacon
        commitment: The inference commitment
        k_layers: Number of layers to challenge (default from config)
        k_gemms_per_layer: Number of GEMMs per layer to verify
        k_blocks_per_gemm: Number of blocks per GEMM

    Returns:
        ChallengeSet with deterministic challenges
    """
    config = get_config()
    k_layers = k_layers or config.k_layers

    # Derive seed from beacon + commitment
    seed_material = (
        b"VERILLM_CHALLENGE_V1"
        + beacon
        + commitment.model_commitment
        + commitment.input_commitment
        + commitment.output_commitment
    )
    for lc in commitment.layer_commitments:
        seed_material += lc

    seed = hashlib.sha256(seed_material).digest()

    # Use seed for deterministic sampling
    rng = random.Random(int.from_bytes(seed[:8], "little"))

    num_layers = len(commitment.layer_commitments)
    if num_layers == 0:
        return ChallengeSet(beacon=beacon, layer_challenges=[])

    # Sample k_layers unique layer indices
    k_actual = min(k_layers, num_layers)
    layer_indices = rng.sample(range(num_layers), k_actual)

    # For each layer, derive GEMM and block challenges
    layer_challenges: List[LayerChallenge] = []

    for layer_idx in layer_indices:
        # Derive layer-specific seed
        layer_seed = hashlib.sha256(seed + struct.pack("<Q", layer_idx)).digest()
        layer_rng = random.Random(int.from_bytes(layer_seed[:8], "little"))

        # Typical transformer has 6 major GEMMs per layer
        num_gemms = 6
        gemm_indices = layer_rng.sample(range(num_gemms), min(k_gemms_per_layer, num_gemms))

        gemm_challenges: List[GEMMChallenge] = []

        for gemm_idx in gemm_indices:
            gemm_seed = hashlib.sha256(
                layer_seed + struct.pack("<Q", gemm_idx)
            ).digest()
            gemm_rng = random.Random(int.from_bytes(gemm_seed[:8], "little"))

            max_blocks = 16
            num_blocks = min(k_blocks_per_gemm, max_blocks * max_blocks)

            block_indices: List[Tuple[int, int]] = []
            all_blocks = [(i, j) for i in range(max_blocks) for j in range(max_blocks)]

            if len(all_blocks) <= num_blocks:
                block_indices = all_blocks
            else:
                block_indices = gemm_rng.sample(all_blocks, num_blocks)

            gemm_challenges.append(
                GEMMChallenge(
                    gemm_idx=gemm_idx,
                    block_indices=block_indices,
                )
            )

        layer_challenges.append(
            LayerChallenge(
                layer_idx=layer_idx,
                gemm_challenges=gemm_challenges,
            )
        )

    return ChallengeSet(
        beacon=beacon,
        layer_challenges=layer_challenges,
    )


def derive_embedding_challenge(
    beacon: bytes,
    commitment: InferenceCommitment,
    num_input_tokens: int,
    k_positions: int = 5,
    include_last_position: bool = False,
) -> Optional[EmbeddingChallenge]:
    """Derive which input token positions to verify for embedding binding.

    Selects k random positions from the input sequence. The miner must
    provide Merkle inclusion proofs for the corresponding embedding rows.

    Args:
        beacon: Random 32-byte beacon
        commitment: The inference commitment
        num_input_tokens: Length of the input token sequence
        k_positions: Number of positions to challenge (default 5)

    Returns:
        EmbeddingChallenge, or None if num_input_tokens is 0
    """
    if num_input_tokens <= 0:
        return None

    seed = hashlib.sha256(
        b"VERILLM_EMBEDDING_CHALLENGE_V1"
        + beacon
        + commitment.input_commitment
        + struct.pack("<I", num_input_tokens)
    ).digest()
    rng = random.Random(int.from_bytes(seed[:8], "little"))

    k_actual = min(k_positions, num_input_tokens)
    required = {num_input_tokens - 1} if include_last_position else set()
    candidates = [
        position for position in range(num_input_tokens) if position not in required
    ]
    random_count = k_actual - len(required)
    positions = sorted(required | set(rng.sample(candidates, random_count)))

    return EmbeddingChallenge(token_positions=positions)


def should_challenge_sampling(beacon: bytes, sampling_verification_bps: int) -> bool:
    """Fiat-Shamir gate for decode-integrity sampling verification."""
    bps = clamp_sampling_bps(sampling_verification_bps)
    if bps <= 0:
        return False
    if bps >= 10_000:
        return True
    h = hashlib.sha256(b"VERILLM_SAMPLING_GATE_V1" + beacon).digest()
    draw = int.from_bytes(h[:2], "little")  # [0, 65535]
    threshold = (bps * 65536) // 10_000
    return draw < threshold


def _compute_k_positions(num_output_tokens: int) -> int:
    """Scale challenged decode positions with output length.

    One verified position is sufficient to catch a wrong lm_head — the
    challenged position is Fiat-Shamir derived and unpredictable.  Extra
    positions only add redundancy, so we scale conservatively to keep
    proof overhead low (each position costs one full lm_head GEMM).
    """
    if num_output_tokens <= 1024:
        return 1
    elif num_output_tokens <= 4096:
        return 2
    else:
        return 3


def decode_challenge_mandated(
    beacon: bytes,
    commitment: InferenceCommitment,
) -> bool:
    """Return the legacy request-level decode-sampling gate.

    This remains useful for lightweight sampling checks.  It must not be used
    for the execution hard audit: a caller can observe the request's sampling
    rate before the miner freezes its commitment.
    """
    is_greedy = (
        not commitment.do_sample
        and commitment.temperature_milli == 0
    )
    is_sampled_with_seed = bool(
        commitment.do_sample
        and commitment.sampling_seed_commitment
    )
    if not is_greedy and not is_sampled_with_seed:
        return False
    # The vLLM cache capture starts after trace row 0 (the final prompt
    # token), so an independently replayed transition needs at least one
    # generated suffix row.  A one-token response still receives its normal
    # light/sampling checks but cannot be mislabeled as a hard transition
    # audit.
    if int(commitment.output_token_count or 0) <= 1:
        return False
    return should_challenge_sampling(
        beacon,
        commitment.sampling_verification_bps,
    )


def _hard_audit_decode_eligible(commitment: InferenceCommitment) -> bool:
    """Return whether the captured decode suffix can carry a hard audit.

    The current transition witness begins at the final prompt token and needs
    at least one generated successor.  This is intentionally independent of
    the caller-visible lightweight sampling rate.
    """

    is_greedy = (
        not commitment.do_sample
        and commitment.temperature_milli == 0
    )
    is_sampled_with_seed = bool(
        commitment.do_sample
        and commitment.sampling_seed_commitment
    )
    return bool(
        (is_greedy or is_sampled_with_seed)
        and int(commitment.output_token_count or 0) > 1
    )


def hard_audit_selected(beacon: bytes, hard_audit_bps: int) -> bool:
    """Return the pure post-commitment execution-audit draw.

    This intentionally depends only on the validator-derived beacon and the
    authority-signed policy rate.  A verifier can therefore decide the
    obligation as soon as it receives the miner's frozen precommitment hash,
    before it reveals the nonce.  Do not fold miner-controlled output length
    or capture availability into this function: doing so would let a miner
    suppress an unfavorable draw after learning the nonce.
    """
    if (
        isinstance(hard_audit_bps, bool)
        or not isinstance(hard_audit_bps, int)
        or not 1 <= hard_audit_bps <= 10_000
    ):
        raise ValueError("hard-audit policy rate is invalid")
    if hard_audit_bps == 10_000:
        return True
    digest = hashlib.sha256(
        b"VERILLM_EXECUTION_HARD_AUDIT_GATE_V1" + beacon
    ).digest()
    draw = int.from_bytes(digest, "big")
    return draw < (hard_audit_bps * (1 << 256)) // 10_000


def hard_audit_mandated(
    beacon: bytes,
    commitment: InferenceCommitment,
    hard_audit_bps: int,
) -> bool:
    """Return whether an eligible response must carry a hard audit.

    ``hard_audit_selected`` is the security-relevant post-commitment draw.
    This compatibility helper additionally checks whether the currently
    implemented decode transition witness can represent the response.  Callers
    that enforce post-nonce delivery must use the pure selection function so a
    miner cannot turn a selected audit into an apparent light response by
    manipulating eligibility.
    """

    return bool(
        hard_audit_selected(beacon, hard_audit_bps)
        and _hard_audit_decode_eligible(commitment)
    )


def hard_audit_required(
    beacon: bytes,
    commitment: InferenceCommitment,
    hard_audit_bps: int,
) -> bool:
    """Return the selected hard-audit obligation or fail if no witness exists.

    Production proof-v2 paths must use this function rather than the
    compatibility ``hard_audit_mandated`` helper.  The draw is fixed from the
    frozen transcript, so an output shape that the current trace cannot
    represent is a proof failure, never a reason to silently downgrade a
    selected response to the light tier.
    """

    if not hard_audit_selected(beacon, hard_audit_bps):
        return False
    if not _hard_audit_decode_eligible(commitment):
        raise ValueError(
            "selected proof-v2 hard audit has no complete decode transition witness"
        )
    return True


def derive_hard_audit_sampling_challenge(
    beacon: bytes,
    commitment: InferenceCommitment,
    vocab_size: int,
    k_positions: int = 2,
) -> Optional[SamplingChallenge]:
    """Derive the high-assurance LM-head check required by a hard audit.

    Unlike :func:`derive_sampling_challenge`, this does not consult the
    request-level sampling rate.  The execution-audit gate has already been
    selected from the signed manifest and the post-commitment nonce.
    """

    if not _hard_audit_decode_eligible(commitment):
        return None
    if (
        not isinstance(commitment.decode_hidden_row_root, bytes)
        or len(commitment.decode_hidden_row_root) != 32
        or not isinstance(commitment.decode_logits_row_root, bytes)
        or len(commitment.decode_logits_row_root) != 32
    ):
        return None
    if commitment.do_sample and (
        not isinstance(commitment.sampling_seed_commitment, bytes)
        or len(commitment.sampling_seed_commitment) != 32
    ):
        return None
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        return None
    if isinstance(k_positions, bool) or not isinstance(k_positions, int) or k_positions <= 0:
        return None

    num_steps = int(commitment.output_token_count)
    seed = hashlib.sha256(
        b"VERILLM_HARD_AUDIT_SAMPLING_CHALLENGE_V1"
        + beacon
        + commitment.decode_hidden_row_root
        + commitment.decode_logits_row_root
        + commitment.output_commitment
        + struct.pack("<I", vocab_size)
    ).digest()
    rng = random.Random(int.from_bytes(seed[:8], "little"))
    k_actual = min(k_positions, num_steps)
    return SamplingChallenge(
        decode_positions=sorted(rng.sample(range(num_steps), k_actual)),
        lm_head_block_indices=[(0, 0)],
        # The execution audit always requires the full canonical sampling
        # witness, even when the public request uses a light sample rate.
        high_assurance=True,
    )


def validate_proof_v2_decode_commitment(
    commitment: InferenceCommitment,
) -> None:
    """Require the canonical decode commitments needed by proof v2."""

    count = commitment.output_token_count
    bps = commitment.sampling_verification_bps
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("proof-v2 output_token_count is invalid")
    if (
        isinstance(bps, bool)
        or not isinstance(bps, int)
        or not 0 <= bps <= 10_000
    ):
        raise ValueError("proof-v2 sampling_verification_bps is invalid")
    if type(commitment.do_sample) is not bool:
        raise ValueError("proof-v2 do_sample is invalid")
    if (
        isinstance(commitment.temperature_milli, bool)
        or not isinstance(commitment.temperature_milli, int)
        or not 0 <= commitment.temperature_milli <= 65_535
    ):
        raise ValueError("proof-v2 temperature is invalid")
    if (
        isinstance(commitment.presence_penalty_milli, bool)
        or not isinstance(commitment.presence_penalty_milli, int)
        or not -2_000 <= commitment.presence_penalty_milli <= 2_000
    ):
        raise ValueError("proof-v2 presence penalty is invalid")
    if bps == 0:
        return
    if count == 0:
        raise ValueError("proof-v2 decode policy requires output tokens")
    if not isinstance(commitment.decode_hidden_row_root, bytes) or len(
        commitment.decode_hidden_row_root
    ) != 32:
        raise ValueError("proof-v2 hidden-row commitment is missing")
    if not isinstance(commitment.decode_logits_row_root, bytes) or len(
        commitment.decode_logits_row_root
    ) != 32:
        raise ValueError("proof-v2 logits-row commitment is missing")
    if commitment.do_sample:
        if not isinstance(commitment.sampling_seed_commitment, bytes) or len(
            commitment.sampling_seed_commitment
        ) != 32:
            raise ValueError("proof-v2 sampling seed commitment is missing")
    else:
        if commitment.temperature_milli != 0:
            raise ValueError("proof-v2 greedy decode temperature is invalid")
        if commitment.sampling_seed_commitment:
            raise ValueError("proof-v2 greedy decode has a sampling seed")


def derive_sampling_challenge(
    beacon: bytes,
    commitment: InferenceCommitment,
    vocab_size: int,
    k_positions: Optional[int] = None,
) -> Optional[SamplingChallenge]:
    """Derive decode positions to verify sampling correctness.

    For temperature=0 (greedy): argmax verification against proved logits.
    For do_sample=True (temperature > 0): canonical sampler replay
    verification when ``sampling_seed_commitment`` is present.

    Args:
        k_positions: Override number of challenged positions.
            None = auto-scale with output length.
    """
    # Allow sampling challenge for do_sample=True when seed is committed.
    is_greedy = (not commitment.do_sample) and (commitment.temperature_milli == 0)
    is_sampled_with_seed = (
        commitment.do_sample
        and commitment.sampling_seed_commitment
    )
    if not is_greedy and not is_sampled_with_seed:
        return None
    if not commitment.decode_hidden_row_root:
        return None
    num_steps = int(commitment.output_token_count or 0)
    if num_steps <= 0:
        return None
    if not should_challenge_sampling(beacon, commitment.sampling_verification_bps):
        return None

    if k_positions is None:
        k_positions = _compute_k_positions(num_steps)

    seed = hashlib.sha256(
        b"VERILLM_SAMPLING_CHALLENGE_V1"
        + beacon
        + commitment.decode_hidden_row_root
        + commitment.output_commitment
        + struct.pack("<I", int(vocab_size))
    ).digest()
    rng = random.Random(int.from_bytes(seed[:8], "little"))

    k_actual = min(max(1, int(k_positions)), num_steps)
    decode_positions = sorted(rng.sample(range(num_steps), k_actual))

    # One-row lm_head proofs use a single Y block spanning vocab.
    block_indices = [(0, 0)]

    # High-assurance is determined solely by the validator-requested bps.
    # The verifier enforces that decode_logits_row_root is present when
    # high_assurance=True — a miner cannot downgrade by omitting it.
    bps = clamp_sampling_bps(commitment.sampling_verification_bps)
    high_assurance = bps >= HIGH_ASSURANCE_BPS

    return SamplingChallenge(
        decode_positions=decode_positions,
        lm_head_block_indices=block_indices,
        high_assurance=high_assurance,
    )
