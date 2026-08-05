"""Bounded scheduler geometry committed by proof-v3 serving.

The trace contains scheduling shape only. Request identifiers and token values
never enter its canonical representation.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error

SCHEDULER_GEOMETRY_ABI_V3 = "vllm.scheduler_geometry.v2"
# The signed release profile permits 8,192 visible decode tokens.  Decode
# ordinarily advances one token per target-bearing scheduler step, with
# additional steps for prefill and a possible discarded stop row.
MAX_SCHEDULER_GEOMETRY_STEPS_V3 = 16_384
MAX_SCHEDULER_GEOMETRY_GAP_STEPS_V3 = 4096
MAX_SCHEDULER_GEOMETRY_SLOTS_V3 = 256
MAX_SCHEDULER_GEOMETRY_COHORTS_V3 = 4096

_TRACE_DOMAIN = b"VERATHOS/PROOF_V3/SCHEDULER_GEOMETRY/V2"
_PREFIX_TRACE_DOMAIN = b"VERATHOS/PROOF_V3/SCHEDULER_GEOMETRY/V3/PREFIX"
_DIGEST_DOMAIN = b"VERATHOS/PROOF_V3/SCHEDULER_GEOMETRY_DIGEST/V2"

__all__ = [
    "MAX_SCHEDULER_GEOMETRY_COHORTS_V3",
    "MAX_SCHEDULER_GEOMETRY_GAP_STEPS_V3",
    "MAX_SCHEDULER_GEOMETRY_SLOTS_V3",
    "MAX_SCHEDULER_GEOMETRY_STEPS_V3",
    "SCHEDULER_GEOMETRY_ABI_V3",
    "SchedulerGeometrySlotV3",
    "SchedulerGeometryStepV3",
    "SchedulerGeometryTraceBuilderV3",
    "SchedulerGeometryTraceV3",
    "prefix_cache_full_recompute_geometry_v3",
    "scheduler_geometry_replay_equivalent_v3",
]


def _u32(value: int, name: str, *, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < int(positive)
        or value >= 1 << 32
    ):
        raise ProofV3Error(f"{name} is out of range")
    return value


@dataclass(frozen=True, slots=True)
class SchedulerGeometrySlotV3:
    """One scheduled request slot before an engine step executes."""

    cohort_id: int
    prompt_tokens: int
    computed_tokens: int
    scheduled_tokens: int
    is_target: bool

    def __post_init__(self) -> None:
        _u32(self.cohort_id, "scheduler cohort id")
        if self.cohort_id >= MAX_SCHEDULER_GEOMETRY_COHORTS_V3:
            raise ProofV3Error("scheduler cohort id exceeds the bound")
        _u32(self.prompt_tokens, "scheduler prompt token count", positive=True)
        _u32(self.computed_tokens, "scheduler computed token count")
        _u32(self.scheduled_tokens, "scheduler scheduled token count", positive=True)
        if not isinstance(self.is_target, bool):
            raise ProofV3Error("scheduler target marker is malformed")

    @property
    def is_prefill(self) -> bool:
        return self.computed_tokens < self.prompt_tokens

    def canonical_bytes(self) -> bytes:
        return struct.pack(
            "<IIIIB",
            self.cohort_id,
            self.prompt_tokens,
            self.computed_tokens,
            self.scheduled_tokens,
            int(self.is_target),
        )


@dataclass(frozen=True, slots=True)
class SchedulerGeometryStepV3:
    """One target-bearing vLLM scheduler step."""

    gap_steps_before: int
    total_scheduled_tokens: int
    slots: tuple[SchedulerGeometrySlotV3, ...]

    def __post_init__(self) -> None:
        _u32(self.gap_steps_before, "scheduler gap count")
        _u32(
            self.total_scheduled_tokens,
            "scheduler total token count",
            positive=True,
        )
        if (
            not isinstance(self.slots, tuple)
            or not self.slots
            or len(self.slots) > MAX_SCHEDULER_GEOMETRY_SLOTS_V3
            or any(
                not isinstance(slot, SchedulerGeometrySlotV3)
                for slot in self.slots
            )
        ):
            raise ProofV3Error("scheduler slot inventory is malformed")
        if sum(slot.scheduled_tokens for slot in self.slots) != (
            self.total_scheduled_tokens
        ):
            raise ProofV3Error("scheduler slot token counts do not sum")
        if sum(slot.is_target for slot in self.slots) != 1:
            raise ProofV3Error(
                "scheduler step must contain exactly one target slot"
            )
        cohorts = tuple(slot.cohort_id for slot in self.slots)
        if len(cohorts) != len(set(cohorts)):
            raise ProofV3Error("scheduler step repeats a cohort")

    @property
    def target_index(self) -> int:
        return next(
            index
            for index, slot in enumerate(self.slots)
            if slot.is_target
        )

    @property
    def target_query_start(self) -> int:
        return sum(
            slot.scheduled_tokens
            for slot in self.slots[: self.target_index]
        )

    def canonical_bytes(self) -> bytes:
        return (
            struct.pack(
                "<IIH",
                self.gap_steps_before,
                self.total_scheduled_tokens,
                len(self.slots),
            )
            + b"".join(slot.canonical_bytes() for slot in self.slots)
        )


@dataclass(frozen=True, slots=True)
class SchedulerGeometryTraceV3:
    """Canonical target-local geometry for one served request.

    vLLM may execute one final decode row whose sampled stop token is not
    exposed in ``output_token_ids``.  The execution-anchor stream already
    defers one row so that visible-token commitments exclude this tail.  The
    scheduler transcript still records the real step because hard replay must
    reproduce its execution geometry.
    """

    context_token_count: int
    sequence_token_count: int
    steps: tuple[SchedulerGeometryStepV3, ...]
    initial_computed_tokens: int = 0

    def __post_init__(self) -> None:
        _u32(
            self.context_token_count,
            "scheduler context token count",
            positive=True,
        )
        _u32(
            self.sequence_token_count,
            "scheduler sequence token count",
            positive=True,
        )
        initial = _u32(
            self.initial_computed_tokens,
            "scheduler initial computed token count",
        )
        if initial >= self.context_token_count:
            raise ProofV3Error(
                "scheduler initial computed tokens must precede prompt end"
            )
        if (
            not isinstance(self.steps, tuple)
            or not self.steps
            or len(self.steps) > MAX_SCHEDULER_GEOMETRY_STEPS_V3
            or any(
                not isinstance(step, SchedulerGeometryStepV3)
                for step in self.steps
            )
        ):
            raise ProofV3Error("scheduler geometry steps are malformed")
        target_slots = tuple(
            step.slots[step.target_index] for step in self.steps
        )
        if self.steps[0].gap_steps_before:
            raise ProofV3Error(
                "scheduler geometry begins with an impossible target gap"
            )
        if sum(step.gap_steps_before for step in self.steps) > (
            MAX_SCHEDULER_GEOMETRY_GAP_STEPS_V3
        ):
            raise ProofV3Error(
                "scheduler target gap count exceeds the bound"
            )
        if any(
            slot.cohort_id != 0 or not slot.is_target
            for slot in target_slots
        ):
            raise ProofV3Error("scheduler target cohort is not canonical")
        if any(
            slot.prompt_tokens != self.context_token_count
            for slot in target_slots
        ):
            raise ProofV3Error(
                "scheduler target prompt count changed across steps"
            )
        if target_slots[0].computed_tokens != initial:
            raise ProofV3Error(
                "scheduler target did not begin at the committed cache boundary"
            )
        expected_computed = initial
        for _step, slot in zip(self.steps, target_slots, strict=True):
            if slot.computed_tokens != expected_computed:
                raise ProofV3Error(
                    "scheduler target chronology is not contiguous"
                )
            expected_computed += slot.scheduled_tokens
        trailing_stop_row = (
            expected_computed == self.sequence_token_count + 1
            and target_slots[-1].computed_tokens
            == self.sequence_token_count
            and target_slots[-1].scheduled_tokens == 1
        )
        if (
            expected_computed != self.sequence_token_count
            and not trailing_stop_row
        ):
            raise ProofV3Error(
                "scheduler target rows do not cover the committed sequence "
                f"(observed={expected_computed}, "
                f"committed={self.sequence_token_count})"
            )
        cohorts = {
            slot.cohort_id
            for step in self.steps
            for slot in step.slots
        }
        if cohorts != set(range(max(cohorts) + 1)):
            raise ProofV3Error(
                "scheduler cohort inventory is not canonical"
            )
        prompt_counts: dict[int, int] = {}
        next_computed: dict[int, int] = {}
        last_seen_gap_total: dict[int, int] = {}
        gap_total = 0
        for step in self.steps:
            gap_total += step.gap_steps_before
            for slot in step.slots:
                previous_prompt = prompt_counts.setdefault(
                    slot.cohort_id,
                    slot.prompt_tokens,
                )
                if previous_prompt != slot.prompt_tokens:
                    raise ProofV3Error(
                        "scheduler cohort prompt geometry changed"
                    )
                expected = next_computed.get(slot.cohort_id)
                if expected is not None:
                    if slot.computed_tokens < expected:
                        raise ProofV3Error(
                            "scheduler cohort chronology moved backwards"
                        )
                    if (
                        slot.computed_tokens != expected
                        and last_seen_gap_total.get(
                            slot.cohort_id,
                            gap_total,
                        )
                        == gap_total
                    ):
                        raise ProofV3Error(
                            "scheduler cohort chronology is not contiguous"
                        )
                next_computed[slot.cohort_id] = (
                    slot.computed_tokens + slot.scheduled_tokens
                )
                last_seen_gap_total[slot.cohort_id] = gap_total

    @property
    def cohort_count(self) -> int:
        return 1 + max(
            slot.cohort_id
            for step in self.steps
            for slot in step.slots
        )

    def canonical_bytes(self) -> bytes:
        if self.initial_computed_tokens == 0:
            prefix = _TRACE_DOMAIN + struct.pack(
                "<IIH",
                self.context_token_count,
                self.sequence_token_count,
                len(self.steps),
            )
        else:
            prefix = _PREFIX_TRACE_DOMAIN + struct.pack(
                "<IIIH",
                self.context_token_count,
                self.sequence_token_count,
                self.initial_computed_tokens,
                len(self.steps),
            )
        return prefix + b"".join(
            step.canonical_bytes() for step in self.steps
        )

    def digest(self) -> bytes:
        return hashlib.sha256(
            _DIGEST_DOMAIN + self.canonical_bytes()
        ).digest()


class SchedulerGeometryTraceBuilderV3:
    """Mutable request-local builder used only on the serving hot path."""

    def __init__(self, target_request_id: str) -> None:
        if not isinstance(target_request_id, str) or not target_request_id:
            raise ProofV3Error("scheduler target request id is malformed")
        self._target_request_id = target_request_id
        self._cohorts = {target_request_id: 0}
        self._steps: list[SchedulerGeometryStepV3] = []
        self._started = False
        self._gap_steps = 0

    def observe_absent_step(self) -> None:
        if self._started:
            self._gap_steps += 1
            if self._gap_steps >= 1 << 32:
                raise ProofV3Error("scheduler geometry gap count overflowed")

    def observe_step(
        self,
        *,
        request_ids: tuple[str, ...],
        prompt_tokens: tuple[int, ...],
        computed_tokens: tuple[int, ...],
        scheduled_tokens: tuple[int, ...],
        total_scheduled_tokens: int,
    ) -> None:
        length = len(request_ids)
        if (
            not request_ids
            or len(prompt_tokens) != length
            or len(computed_tokens) != length
            or len(scheduled_tokens) != length
        ):
            raise ProofV3Error("scheduler geometry vectors disagree")
        try:
            target_index = request_ids.index(self._target_request_id)
        except ValueError:
            self.observe_absent_step()
            return
        if scheduled_tokens[target_index] <= 0:
            self.observe_absent_step()
            return

        slots = []
        for index, request_id in enumerate(request_ids):
            scheduled = int(scheduled_tokens[index])
            if scheduled <= 0:
                continue
            cohort = self._cohorts.get(request_id)
            if cohort is None:
                cohort = len(self._cohorts)
                if cohort >= MAX_SCHEDULER_GEOMETRY_COHORTS_V3:
                    raise ProofV3Error(
                        "scheduler geometry cohort bound exceeded"
                    )
                self._cohorts[request_id] = cohort
            slots.append(
                SchedulerGeometrySlotV3(
                    cohort_id=cohort,
                    prompt_tokens=int(prompt_tokens[index]),
                    computed_tokens=int(computed_tokens[index]),
                    scheduled_tokens=scheduled,
                    is_target=index == target_index,
                )
            )
        self._steps.append(
            SchedulerGeometryStepV3(
                gap_steps_before=self._gap_steps,
                total_scheduled_tokens=int(total_scheduled_tokens),
                slots=tuple(slots),
            )
        )
        if len(self._steps) > MAX_SCHEDULER_GEOMETRY_STEPS_V3:
            raise ProofV3Error("scheduler geometry step bound exceeded")
        self._started = True
        self._gap_steps = 0

    def finalize(
        self,
        *,
        context_token_count: int,
        sequence_token_count: int,
    ) -> SchedulerGeometryTraceV3:
        initial = 0
        if self._steps:
            first = self._steps[0]
            initial = first.slots[first.target_index].computed_tokens
        return SchedulerGeometryTraceV3(
            context_token_count=int(context_token_count),
            sequence_token_count=int(sequence_token_count),
            steps=tuple(self._steps),
            initial_computed_tokens=int(initial),
        )


def prefix_cache_full_recompute_geometry_v3(
    trace: SchedulerGeometryTraceV3,
    *,
    prefill_token_budget: int,
) -> SchedulerGeometryTraceV3:
    """Build the cache-disabled geometry used by hard equality replay."""

    if not isinstance(trace, SchedulerGeometryTraceV3):
        raise ProofV3Error(
            "prefix-cache replay geometry has an unexpected type"
        )
    budget = _u32(
        prefill_token_budget,
        "prefix-cache replay prefill token budget",
        positive=True,
    )
    initial = trace.initial_computed_tokens
    if initial == 0:
        return trace
    trace_steps = trace.steps
    target_slots = tuple(
        step.slots[step.target_index] for step in trace_steps
    )
    observed_end = initial + sum(
        slot.scheduled_tokens for slot in target_slots
    )
    if (
        observed_end == trace.sequence_token_count + 1
        and target_slots[-1].computed_tokens == trace.sequence_token_count
        and target_slots[-1].scheduled_tokens == 1
    ):
        # vLLM's asynchronous serving loop can execute one row after the last
        # visible token before the finished output retires the request.  The
        # execution-anchor contract deliberately excludes that row.  A
        # partitioned full-prompt replay retires synchronously at max_tokens,
        # so reproducing this non-response row is neither possible nor part of
        # the authenticated computation.
        trace_steps = trace_steps[:-1]
    prefix_steps = []
    computed = 0
    while computed < initial:
        scheduled = min(budget, initial - computed)
        prefix_steps.append(SchedulerGeometryStepV3(
            gap_steps_before=0,
            total_scheduled_tokens=scheduled,
            slots=(SchedulerGeometrySlotV3(
                cohort_id=0,
                prompt_tokens=trace.context_token_count,
                computed_tokens=computed,
                scheduled_tokens=scheduled,
                is_target=True,
            ),),
        ))
        computed += scheduled
    if len(prefix_steps) + len(trace_steps) > MAX_SCHEDULER_GEOMETRY_STEPS_V3:
        raise ProofV3Error(
            "prefix-cache full recompute exceeds the scheduler step bound"
        )
    return SchedulerGeometryTraceV3(
        context_token_count=trace.context_token_count,
        sequence_token_count=trace.sequence_token_count,
        steps=tuple(prefix_steps) + trace_steps,
        initial_computed_tokens=0,
    )


def scheduler_geometry_replay_equivalent_v3(
    expected: SchedulerGeometryTraceV3,
    replayed: SchedulerGeometryTraceV3,
) -> bool:
    """Compare execution shape while ignoring vLLM's storage-slot order.

    vLLM may compact/reorder its persistent input batch independently from the
    scheduler request order. The target computation remains byte-identical
    when the target chronology, total scheduled tokens, and multiset of
    companion shapes are identical. Runtime values are authenticated
    separately against the original pre-nonce execution-anchor roots.
    """

    if (
        not isinstance(expected, SchedulerGeometryTraceV3)
        or not isinstance(replayed, SchedulerGeometryTraceV3)
        or expected.context_token_count != replayed.context_token_count
        or expected.sequence_token_count != replayed.sequence_token_count
        or expected.initial_computed_tokens
        != replayed.initial_computed_tokens
        or len(expected.steps) != len(replayed.steps)
    ):
        return False
    for expected_step, replayed_step in zip(
        expected.steps,
        replayed.steps,
        strict=True,
    ):
        if (
            expected_step.gap_steps_before
            != replayed_step.gap_steps_before
            or expected_step.total_scheduled_tokens
            != replayed_step.total_scheduled_tokens
        ):
            return False
        expected_target = expected_step.slots[
            expected_step.target_index
        ]
        replayed_target = replayed_step.slots[
            replayed_step.target_index
        ]
        if (
            expected_target.prompt_tokens,
            expected_target.computed_tokens,
            expected_target.scheduled_tokens,
        ) != (
            replayed_target.prompt_tokens,
            replayed_target.computed_tokens,
            replayed_target.scheduled_tokens,
        ):
            return False

        def companion_shapes(step):
            return tuple(
                sorted(
                    (
                        slot.prompt_tokens,
                        slot.computed_tokens,
                        slot.scheduled_tokens,
                    )
                    for slot in step.slots
                    if not slot.is_target
                )
            )

        if companion_shapes(expected_step) != companion_shapes(
            replayed_step
        ):
            return False
    return True
