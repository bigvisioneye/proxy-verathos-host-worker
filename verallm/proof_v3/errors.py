"""Errors shared by the fail-closed proof-v3 execution-audit boundary."""

from __future__ import annotations


class ProofV3Error(ValueError):
    """Proof-v3 data is malformed, unsupported, or contextually invalid."""


class ProofV3UnavailableError(ProofV3Error):
    """A proof-v3 capability has not been implemented or qualified yet."""


class ProofV3VerificationError(ProofV3Error):
    """A proof-v3 statement does not match the verifier's expected context."""


class ProofV3SignatureError(ProofV3Error):
    """Execution-profile authority signatures are malformed or insufficient."""


class ProofV3DocumentError(ProofV3Error):
    """A signed execution-profile release artifact is malformed."""
