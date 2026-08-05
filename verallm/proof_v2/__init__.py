"""Protocol v2 proof metadata."""

from verallm.proof_v2.manifest import (
    ManifestContextError,
    ManifestFormatError,
    ManifestSignatureError,
    ModelSpecIdentity,
    OperationDescriptor,
    StaticWeightCommitmentManifest,
    verify_signed_manifest,
)
from verallm.proof_v2.payload import (
    GemmBlockProofV2,
    ProofV2CommitmentEnvelope,
    ProofV2Payload,
    ProofV2PayloadError,
    commitment_envelope_from_bytes,
    proof_payload_from_bytes,
)
from verallm.proof_v2.pcs_batch import (
    XWBatchContextError,
    XWBatchOpeningContextV2,
    derive_xw_batch_opening_context_v2,
)

__all__ = [
    "ManifestContextError",
    "ManifestFormatError",
    "ManifestSignatureError",
    "ModelSpecIdentity",
    "OperationDescriptor",
    "StaticWeightCommitmentManifest",
    "GemmBlockProofV2",
    "ProofV2CommitmentEnvelope",
    "ProofV2Payload",
    "ProofV2PayloadError",
    "XWBatchContextError",
    "XWBatchOpeningContextV2",
    "commitment_envelope_from_bytes",
    "derive_xw_batch_opening_context_v2",
    "proof_payload_from_bytes",
    "verify_signed_manifest",
]
