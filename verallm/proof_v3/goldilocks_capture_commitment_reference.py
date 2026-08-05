"""Cheap always-on capture commitment for proof-v3.

This module separates the two cost tiers that make verified inference
economical:

* **Capture + commit (cheap, ALWAYS-ON).** On every request the miner
  freezes the raw activation/trace values and commits a plain width-1
  Merkle root over them: O(N) hashing, zero field NTT, GPU-streamable.
  Crucially this happens *before* the validator nonce is revealed, so the
  miner cannot tell whether a proof will be demanded and must do identical
  work on every request -- the commitment is *undetectable* (behaviourally
  indistinguishable between audited and non-audited requests).

* **Proof (expensive, PROBABILISTIC).** Only the fraction of requests that
  the validator hard-audits (post-nonce) trigger the full succinct
  argument, whose Reed-Solomon LDE + codeword Merkle tree carry the NTT
  cost. That cost is paid at the hard-audit rate, not per request.

Binding. The capture commitment is the Merkle root over the exact
multilinear evaluation vector the succinct argument proves over. When a
proof is demanded, the prover builds the RS-codeword PCS commitment from
the *same* frozen values; :func:`verify_capture_binds_pcs_v3` checks that
the PCS commitment is the RS encoding of the captured values, so a miner
that committed one trace cannot later prove a different one. (In the
reference this re-derives the codeword from the captured values, O(N); the
native backend replaces that with the systematic FRI-query overlap
opening, documented in the plan doc -- the ABI here is the binding
contract that backend must satisfy.)

The point of this module is the cost model, not new soundness: it
quantifies "commit is cheap, proof is rare" with a measured ratio.
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_linear_relation_reference import (
    _fixed32,
    _integer,
)
from verallm.proof_v3.goldilocks_merkle_reference import (
    GoldilocksMerkleTreeReference,
)
from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
    GoldilocksMultilinearPcsStatementV3,
    commit_goldilocks_multilinear_v3,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS


GOLDILOCKS_CAPTURE_COMMITMENT_ABI_V3: Final = (
    "goldilocks.capture_commitment.reference.v1"
)
_CAPTURE_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_CAPTURE/V1/COMMIT/SHA256"


def _field(value: object, name: str) -> int:
    integer = _integer(value, name)
    if not 0 <= integer < GOLDILOCKS_MODULUS:
        raise ProofV3Error(f"{name} must be a canonical Goldilocks element")
    return integer


@dataclass(frozen=True, slots=True)
class GoldilocksCaptureCommitmentV3:
    """The cheap always-on commitment: a raw-value Merkle root + shape."""

    validator_binding_digest: bytes
    variable_count: int
    root: bytes

    def __post_init__(self) -> None:
        _fixed32(self.validator_binding_digest, "capture binding", nonzero=True)
        variables = _integer(self.variable_count, "variable_count")
        if variables < 1:
            raise ProofV3Error("capture variable count must be positive")
        _fixed32(self.root, "capture root")

    def digest(self) -> bytes:
        return hashlib.sha256(
            _CAPTURE_DOMAIN
            + self.validator_binding_digest
            + struct.pack("<I", self.variable_count)
            + self.root
        ).digest()

    def capture_binding_digest(self) -> bytes:
        return hashlib.sha256(
            _CAPTURE_DOMAIN + b"LEAVES/" + self.validator_binding_digest
        ).digest()


def capture_commit_v3(
    *,
    validator_binding_digest: bytes,
    raw_values: tuple[int, ...],
) -> GoldilocksCaptureCommitmentV3:
    """Commit raw captured values with O(N) hashing and NO field NTT.

    ``raw_values`` is the multilinear evaluation vector (one leaf per value,
    power-of-two length). This is the artifact posted on every request.
    """

    binding = _fixed32(
        validator_binding_digest, "capture binding", nonzero=True
    )
    n = len(raw_values)
    if n < 2 or n & (n - 1):
        raise ProofV3Error("capture values must be a power-of-two count")
    variables = n.bit_length() - 1
    values = tuple((v,) for v in (_field(x, "capture value") for x in raw_values))
    leaf_binding = hashlib.sha256(
        _CAPTURE_DOMAIN + b"LEAVES/" + binding
    ).digest()
    tree = GoldilocksMerkleTreeReference.from_rows(values, binding_digest=leaf_binding)
    return GoldilocksCaptureCommitmentV3(
        validator_binding_digest=binding,
        variable_count=variables,
        root=tree.commitment,
    )


def _rebuild_capture_root(
    *, binding: bytes, raw_values: tuple[int, ...]
) -> bytes:
    leaf_binding = hashlib.sha256(_CAPTURE_DOMAIN + b"LEAVES/" + binding).digest()
    tree = GoldilocksMerkleTreeReference.from_rows(
        tuple((_field(x, "capture value"),) for x in raw_values),
        binding_digest=leaf_binding,
    )
    return tree.commitment


def verify_capture_binds_pcs_v3(
    *,
    capture: GoldilocksCaptureCommitmentV3,
    pcs_statement: GoldilocksMultilinearPcsStatementV3,
    pcs_commitment: bytes,
    raw_values: tuple[int, ...],
) -> None:
    """Check the PCS commitment is the RS encoding of the captured values.

    Two independent checks:
    1. the captured values reproduce the pre-posted capture root (so the
       revealed witness is exactly what was committed before the nonce);
    2. the RS-codeword PCS commitment is the encoding of those same values.

    A miner that posts one capture root cannot prove a different trace: the
    revealed ``raw_values`` must satisfy BOTH, and the two roots pin the
    same vector.
    """

    try:
        if not isinstance(capture, GoldilocksCaptureCommitmentV3):
            raise ProofV3VerificationError("capture commitment is malformed")
        if pcs_statement.variable_count != capture.variable_count:
            raise ProofV3VerificationError(
                "capture and PCS statements have different shapes"
            )
        expected_leaves = 1 << capture.variable_count
        if len(raw_values) != expected_leaves:
            raise ProofV3VerificationError("capture value count is wrong")
        rebuilt = _rebuild_capture_root(
            binding=capture.validator_binding_digest, raw_values=tuple(raw_values)
        )
        if rebuilt != capture.root:
            raise ProofV3VerificationError(
                "revealed values do not match the posted capture commitment"
            )
        rebuilt_pcs = commit_goldilocks_multilinear_v3(
            statement=pcs_statement, evaluations=tuple(raw_values)
        )
        if rebuilt_pcs.commitment != _fixed32(pcs_commitment, "pcs commitment"):
            raise ProofV3VerificationError(
                "PCS commitment is not the RS encoding of the captured values"
            )
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("capture binding is malformed") from exc


def measure_commit_cost_ratio_v3(
    *,
    validator_binding_digest: bytes,
    raw_values: tuple[int, ...],
) -> dict:
    """Measure cheap-capture vs full-PCS commit cost on the same values.

    Returns wall-clock for the always-on capture commit and the on-demand
    PCS commit (RS NTT + codeword tree), plus their ratio -- the factor by
    which the expensive tier is avoided on non-audited requests.
    """

    binding = _fixed32(validator_binding_digest, "capture binding", nonzero=True)
    n = len(raw_values)
    variables = n.bit_length() - 1

    t0 = time.perf_counter()
    capture = capture_commit_v3(
        validator_binding_digest=binding, raw_values=raw_values
    )
    capture_ms = (time.perf_counter() - t0) * 1e3

    statement = GoldilocksMultilinearPcsStatementV3(
        validator_binding_digest=binding, variable_count=variables
    )
    t0 = time.perf_counter()
    commit_goldilocks_multilinear_v3(statement=statement, evaluations=raw_values)
    pcs_ms = (time.perf_counter() - t0) * 1e3

    return {
        "leaves": n,
        "capture_ms": capture_ms,
        "pcs_commit_ms": pcs_ms,
        "ratio": (pcs_ms / capture_ms) if capture_ms else float("inf"),
        "capture_root": capture.root.hex(),
    }


__all__ = [
    "GOLDILOCKS_CAPTURE_COMMITMENT_ABI_V3",
    "GoldilocksCaptureCommitmentV3",
    "capture_commit_v3",
    "measure_commit_cost_ratio_v3",
    "verify_capture_binds_pcs_v3",
]
