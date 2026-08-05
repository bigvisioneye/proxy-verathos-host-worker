"""HTTP transport for the economic proof-v3 hard nonce reveal."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import logging

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, root_validator, validator

from verallm.miner.economic_proof_v3_serving import (
    EconomicProofV3ServingCoordinator,
)
from verallm.proof_v3.errors import ProofV3VerificationError
from verallm.proof_v3.request import validator_hotkey_identity_digest_v3
from verallm.proof_v3.session import (
    MAX_NONCE_REVEAL_HOLD_BUDGET_NS_V3,
    MAX_NONCE_REVEAL_BYTES_V3,
    NonceRevealV3,
)

ECONOMIC_PROOF_V3_CHALLENGE_PATH = "/proof/v3/challenge"
ECONOMIC_PROOF_V3_RETENTION_PATH = "/proof/v3/retention"
ECONOMIC_PROOF_V3_MEDIA_TYPE = "application/vnd.verathos.proof-v3+octet-stream"
logger = logging.getLogger(__name__)

__all__ = [
    "ECONOMIC_PROOF_V3_CHALLENGE_PATH",
    "ECONOMIC_PROOF_V3_MEDIA_TYPE",
    "ECONOMIC_PROOF_V3_RETENTION_PATH",
    "EconomicProofV3NonceRevealBody",
    "EconomicProofV3RetentionBody",
    "register_economic_proof_v3_routes",
]


class EconomicProofV3NonceRevealBody(BaseModel):
    """One bounded canonical reveal encoded as lowercase hexadecimal."""

    nonce_reveal: str

    @validator("nonce_reveal")
    def _validate_nonce_reveal(cls, value: str) -> str:
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 2 * MAX_NONCE_REVEAL_BYTES_V3
            or len(value) % 2
        ):
            raise ValueError("proof-v3 nonce reveal hex length is invalid")
        try:
            decoded = bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError("proof-v3 nonce reveal must be hexadecimal") from exc
        if decoded.hex() != value:
            raise ValueError(
                "proof-v3 nonce reveal must use canonical lowercase hexadecimal"
            )
        NonceRevealV3.from_canonical_bytes(decoded)
        return value

    @property
    def canonical_bytes(self) -> bytes:
        return bytes.fromhex(self.nonce_reveal)


class EconomicProofV3RetentionBody(BaseModel):
    """One post-precommit hold or light-release operation."""

    action: str
    proof_challenge_id: str
    commitment_envelope_digest: str
    hold_budget_ns: int = 0

    @validator("action")
    def _validate_action(cls, value: str) -> str:
        if value not in ("hold", "release"):
            raise ValueError("proof-v3 retention action is unsupported")
        return value

    @validator("proof_challenge_id", "commitment_envelope_digest")
    def _validate_fixed_digest(cls, value: str) -> str:
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("proof-v3 retention digest must be 32-byte hex")
        try:
            decoded = bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError(
                "proof-v3 retention digest must be hexadecimal"
            ) from exc
        if decoded == bytes(32) or decoded.hex() != value:
            raise ValueError(
                "proof-v3 retention digest must be canonical nonzero hex"
            )
        return value

    @validator("hold_budget_ns", pre=True)
    def _validate_hold_budget_type(cls, value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("proof-v3 retention hold budget must be an integer")
        return value

    @root_validator(skip_on_failure=True)
    def _validate_action_budget(cls, values):
        action = values.get("action")
        hold_budget_ns = values.get("hold_budget_ns")
        if action == "hold" and (
            not isinstance(hold_budget_ns, int)
            or not 0 < hold_budget_ns <= MAX_NONCE_REVEAL_HOLD_BUDGET_NS_V3
        ):
            raise ValueError("proof-v3 retention hold budget is out of range")
        if action == "release" and hold_budget_ns != 0:
            raise ValueError("proof-v3 light release must not carry a hold budget")
        return values


def register_economic_proof_v3_routes(
    app,
    *,
    get_coordinator: Callable[
        [], EconomicProofV3ServingCoordinator | None
    ],
    retain_completed_bundle: Callable[[bytes], object] | None = None,
) -> None:
    """Mount the authenticated hard-opening route on a FastAPI app.

    ``ValidatorAuthMiddleware`` must run before this handler and set the
    authenticated SS58 hotkey on ``request.state.validator_hotkey``. Proof
    generation runs off the event-loop thread because a hard canary may take
    tens of seconds.
    """

    if not callable(get_coordinator):
        raise TypeError("get_coordinator must be callable")

    @app.post(ECONOMIC_PROOF_V3_CHALLENGE_PATH)
    async def economic_proof_v3_challenge(
        body: EconomicProofV3NonceRevealBody,
        request: Request,
    ):
        coordinator = get_coordinator()
        if coordinator is None:
            return JSONResponse(
                status_code=503,
                content={"error": "proof-v3 hard openings are not enabled"},
            )
        hotkey_ss58 = getattr(request.state, "validator_hotkey", "")
        if not hotkey_ss58:
            return JSONResponse(
                status_code=401,
                content={"error": "authenticated validator identity is missing"},
            )
        if not bool(
            getattr(
                request.state,
                "proof_v3_hard_auditor_authorized",
                False,
            )
        ):
            return JSONResponse(
                status_code=403,
                content={
                    "error": (
                        "Validator is not authorized for proof-v3 hard openings"
                    )
                },
            )
        try:
            identity_digest = validator_hotkey_identity_digest_v3(hotkey_ss58)
            reveal = NonceRevealV3.from_canonical_bytes(body.canonical_bytes)
            encoded_proof = await asyncio.to_thread(
                coordinator.answer_hard_reveal,
                encoded_reveal=body.canonical_bytes,
                authenticated_validator_identity_digest=identity_digest,
            )
            if retain_completed_bundle is not None:
                encoded_bundle = await asyncio.to_thread(
                    coordinator.export_completed_bundle,
                    commitment_envelope_digest=(
                        reveal.commitment_envelope_digest
                    ),
                )
                await asyncio.to_thread(
                    retain_completed_bundle,
                    encoded_bundle,
                )
        except ProofV3VerificationError as exc:
            return JSONResponse(
                status_code=409,
                content={"error": str(exc)},
            )
        except Exception:
            logger.exception("proof-v3 hard opening failed")
            return JSONResponse(
                status_code=500,
                content={"error": "proof-v3 hard opening failed"},
            )
        return Response(
            content=encoded_proof,
            media_type=ECONOMIC_PROOF_V3_MEDIA_TYPE,
            headers={"Cache-Control": "no-store"},
        )

    @app.post(ECONOMIC_PROOF_V3_RETENTION_PATH)
    async def economic_proof_v3_retention(
        body: EconomicProofV3RetentionBody,
        request: Request,
    ):
        coordinator = get_coordinator()
        if coordinator is None:
            return JSONResponse(
                status_code=503,
                content={"error": "proof-v3 retention is not enabled"},
            )
        hotkey_ss58 = getattr(request.state, "validator_hotkey", "")
        if not hotkey_ss58:
            return JSONResponse(
                status_code=401,
                content={"error": "authenticated validator identity is missing"},
            )
        if body.action == "hold" and not bool(
            getattr(
                request.state,
                "proof_v3_hard_auditor_authorized",
                False,
            )
        ):
            return JSONResponse(
                status_code=403,
                content={
                    "error": (
                        "Validator is not authorized for proof-v3 "
                        "precommit holds"
                    )
                },
            )
        try:
            identity_digest = validator_hotkey_identity_digest_v3(hotkey_ss58)
            common = {
                "proof_challenge_id": bytes.fromhex(body.proof_challenge_id),
                "commitment_envelope_digest": bytes.fromhex(
                    body.commitment_envelope_digest
                ),
                "authenticated_validator_identity_digest": identity_digest,
            }
            if body.action == "hold":
                await asyncio.to_thread(
                    coordinator.extend_precommit_hold,
                    hold_budget_ns=body.hold_budget_ns,
                    **common,
                )
            else:
                await asyncio.to_thread(
                    coordinator.release_light_precommit,
                    **common,
                )
        except ProofV3VerificationError as exc:
            return JSONResponse(
                status_code=409,
                content={"error": str(exc)},
            )
        except Exception:
            logger.exception("proof-v3 retention operation failed")
            return JSONResponse(
                status_code=500,
                content={"error": "proof-v3 retention operation failed"},
            )
        return JSONResponse(
            status_code=200,
            content={"status": body.action},
            headers={"Cache-Control": "no-store"},
        )
