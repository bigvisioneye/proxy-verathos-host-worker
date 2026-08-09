"""Persistent neutral-first-strike state for proof-v3 hard audits."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Tuple


_STATE_VERSION = 1


@dataclass
class HardProofStrikeState:
    failure_epochs: set[int] = field(default_factory=set)
    clean_epochs: set[int] = field(default_factory=set)
    penalty_required: bool = False
    endpoint: str = ""


class HardProofStrikeTracker:
    """Count distinct failed source epochs and bounded clean recovery epochs.

    The tracker decides only whether a hard failure is operationally neutral
    or should enter the existing penalty path.  It never changes the proof
    verdict and it does not replace ``ProbationTracker``.
    """

    def __init__(
        self,
        *,
        failure_epochs_for_penalty: int = 1,
        clean_hard_audit_epochs_for_reset: int = 3,
    ) -> None:
        self._states: Dict[Tuple[str, int], HardProofStrikeState] = {}
        self.configure(
            failure_epochs_for_penalty=failure_epochs_for_penalty,
            clean_hard_audit_epochs_for_reset=(
                clean_hard_audit_epochs_for_reset
            ),
        )

    @staticmethod
    def _key(key: Tuple[str, int]) -> Tuple[str, int]:
        return str(key[0]).lower(), int(key[1])

    def configure(
        self,
        *,
        failure_epochs_for_penalty: int,
        clean_hard_audit_epochs_for_reset: int,
    ) -> None:
        failures = int(failure_epochs_for_penalty)
        clean = int(clean_hard_audit_epochs_for_reset)
        if not 1 <= failures <= 10:
            raise ValueError("failure epoch threshold is outside [1, 10]")
        if not 1 <= clean <= 100:
            raise ValueError("clean epoch threshold is outside [1, 100]")
        self.failure_epochs_for_penalty = failures
        self.clean_hard_audit_epochs_for_reset = clean
        # A tightened live policy applies to already-pending evidence.  It may
        # require a penalty now, but a relaxed policy never erases evidence.
        for state in self._states.values():
            if len(state.failure_epochs) >= failures:
                state.penalty_required = True

    def record_failure(
        self,
        key: Tuple[str, int],
        source_epoch: int,
        *,
        endpoint: str = "",
        already_on_probation: bool = False,
    ) -> bool:
        """Record one distinct failed epoch and return penalty applicability."""

        epoch = int(source_epoch)
        if epoch < 0:
            raise ValueError("source epoch must be non-negative")
        canonical = self._key(key)
        state = self._states.setdefault(canonical, HardProofStrikeState())
        if endpoint:
            state.endpoint = str(endpoint).rstrip("/")
        if already_on_probation:
            state.penalty_required = True
            state.failure_epochs.add(epoch)
            state.clean_epochs.clear()
            return True
        if epoch in state.failure_epochs:
            return bool(state.penalty_required)
        state.failure_epochs.add(epoch)
        state.clean_epochs.clear()
        if len(state.failure_epochs) >= self.failure_epochs_for_penalty:
            state.penalty_required = True
        return bool(state.penalty_required)

    def record_clean_pass(
        self,
        key: Tuple[str, int],
        source_epoch: int,
    ) -> bool:
        """Record one clean hard-audit epoch; return True when a strike clears."""

        canonical = self._key(key)
        state = self._states.get(canonical)
        if state is None or state.penalty_required or not state.failure_epochs:
            return False
        epoch = int(source_epoch)
        if epoch < 0:
            raise ValueError("source epoch must be non-negative")
        if epoch <= max(state.failure_epochs):
            return False
        state.clean_epochs.add(epoch)
        if (
            len(state.clean_epochs)
            < self.clean_hard_audit_epochs_for_reset
        ):
            return False
        del self._states[canonical]
        return True

    def penalty_required(self, key: Tuple[str, int]) -> bool:
        state = self._states.get(self._key(key))
        return bool(state is not None and state.penalty_required)

    def pending(self, key: Tuple[str, int]) -> bool:
        state = self._states.get(self._key(key))
        return bool(
            state is not None
            and state.failure_epochs
            and not state.penalty_required
        )

    def clear(self, key: Tuple[str, int]) -> bool:
        return self._states.pop(self._key(key), None) is not None

    def clear_all(self) -> int:
        """Clear all pending and penalty-bearing hard-proof strike state."""

        count = len(self._states)
        self._states.clear()
        return count

    def migrate_index(
        self,
        address: str,
        new_index: int,
        *,
        new_endpoint: str = "",
    ) -> bool:
        new_key = self._key((address, new_index))
        if new_key in self._states:
            return False
        candidates = [
            key
            for key, state in self._states.items()
            if key[0] == new_key[0]
            and key[1] != new_key[1]
            and (
                not state.endpoint
                or not new_endpoint
                or state.endpoint == str(new_endpoint).rstrip("/")
            )
        ]
        if len(candidates) != 1:
            return False
        old_key = candidates[0]
        state = self._states.pop(old_key)
        if new_endpoint:
            state.endpoint = str(new_endpoint).rstrip("/")
        self._states[new_key] = state
        return True

    def to_json(self) -> str:
        entries = {}
        for (address, model_index), state in sorted(self._states.items()):
            entries[f"{address}|{model_index}"] = {
                "failure_epochs": sorted(state.failure_epochs),
                "clean_epochs": sorted(state.clean_epochs),
                "penalty_required": bool(state.penalty_required),
                "endpoint": state.endpoint,
            }
        return json.dumps(
            {"version": _STATE_VERSION, "entries": entries},
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(
        cls,
        raw: str,
        *,
        failure_epochs_for_penalty: int,
        clean_hard_audit_epochs_for_reset: int,
    ) -> "HardProofStrikeTracker":
        tracker = cls(
            failure_epochs_for_penalty=failure_epochs_for_penalty,
            clean_hard_audit_epochs_for_reset=(
                clean_hard_audit_epochs_for_reset
            ),
        )
        if not raw:
            return tracker
        value = json.loads(raw)
        if (
            not isinstance(value, dict)
            or value.get("version") != _STATE_VERSION
            or not isinstance(value.get("entries"), dict)
        ):
            raise ValueError("hard-proof strike state is malformed")
        for encoded, item in value["entries"].items():
            if (
                not isinstance(encoded, str)
                or "|" not in encoded
                or not isinstance(item, dict)
                or set(item)
                != {
                    "failure_epochs",
                    "clean_epochs",
                    "penalty_required",
                    "endpoint",
                }
            ):
                raise ValueError("hard-proof strike entry is malformed")
            address, raw_index = encoded.rsplit("|", 1)
            model_index = int(raw_index)
            failures = item["failure_epochs"]
            clean = item["clean_epochs"]
            if (
                not address
                or model_index < 0
                or not isinstance(failures, list)
                or not failures
                or not isinstance(clean, list)
                or type(item["penalty_required"]) is not bool
                or not isinstance(item["endpoint"], str)
                or any(type(epoch) is not int or epoch < 0 for epoch in failures)
                or any(type(epoch) is not int or epoch < 0 for epoch in clean)
                or len(set(failures)) != len(failures)
                or len(set(clean)) != len(clean)
            ):
                raise ValueError("hard-proof strike entry is malformed")
            if set(failures).intersection(clean):
                raise ValueError("hard-proof strike epochs overlap")
            state = HardProofStrikeState(
                failure_epochs=set(failures),
                clean_epochs=set(clean),
                penalty_required=bool(item["penalty_required"]),
                endpoint=str(item["endpoint"]),
            )
            if (
                len(state.failure_epochs)
                >= tracker.failure_epochs_for_penalty
            ):
                state.penalty_required = True
            tracker._states[tracker._key((address, model_index))] = state
        return tracker


__all__ = ["HardProofStrikeState", "HardProofStrikeTracker"]
