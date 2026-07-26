# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Strict MaxRL effective-group collection primitives.

These helpers are intentionally independent from TransferQueue and Ray so the
binary-success contract and candidate-wave policy can be tested on CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import torch


class GracefulRoundCancellation(Exception):
    """Stop a strict MaxRL run before updating an incomplete effective batch."""


class GroupOutcome(str, Enum):
    ALL_WRONG = "all_wrong"
    EFFECTIVE = "effective"
    ALL_CORRECT = "all_correct"


@dataclass(frozen=True)
class ClassifiedGroup:
    uid: str
    row_indices: tuple[int, ...]
    success_count: int
    outcome: GroupOutcome


@dataclass(frozen=True)
class GroupSelection:
    accepted: tuple[ClassifiedGroup, ...]
    rejected: tuple[ClassifiedGroup, ...]
    surplus: tuple[ClassifiedGroup, ...]
    ordered: tuple[ClassifiedGroup, ...]

    @property
    def accepted_row_indices(self) -> tuple[int, ...]:
        return tuple(index for group in self.accepted for index in group.row_indices)

    @property
    def discarded_row_indices(self) -> tuple[int, ...]:
        groups = (*self.rejected, *self.surplus)
        return tuple(index for group in groups for index in group.row_indices)

    @property
    def completed(self) -> tuple[ClassifiedGroup, ...]:
        return self.ordered


@dataclass
class EffectiveRoundState:
    """Round-local state; it is discarded unless the optimizer round publishes."""

    target_groups: int
    responses_per_prompt: int
    accepted: list[ClassifiedGroup]
    rejected: list[ClassifiedGroup]
    surplus: list[ClassifiedGroup]
    candidate_order: list[str]
    in_flight_groups: int = 0
    refill_waves: int = 0

    @classmethod
    def create(cls, target_groups: int, responses_per_prompt: int) -> EffectiveRoundState:
        return cls(target_groups, responses_per_prompt, [], [], [], [])

    def submit_wave(self, group_count: int) -> None:
        if group_count <= 0:
            raise ValueError("candidate wave must contain at least one group")
        self.in_flight_groups += group_count
        self.refill_waves += 1

    def complete_wave(self, selection: GroupSelection) -> None:
        completed = selection.completed
        if len(completed) > self.in_flight_groups:
            raise ValueError("completed groups exceed the round's in-flight candidates")
        self.in_flight_groups -= len(completed)
        self.candidate_order.extend(group.uid for group in completed)
        self.accepted.extend(selection.accepted)
        self.rejected.extend(selection.rejected)
        self.surplus.extend(selection.surplus)

    @property
    def accepted_groups(self) -> int:
        return len(self.accepted)

    @property
    def candidate_groups(self) -> int:
        return len(self.candidate_order)


@dataclass(frozen=True)
class CandidateDatasetPosition:
    completed_passes: int
    cursor: int

    def advance(self, count: int, *, dataset_size: int) -> CandidateDatasetPosition:
        if count <= 0 or dataset_size <= 0:
            raise ValueError("count and dataset_size must be positive")
        absolute = self.cursor + count
        passes, cursor = divmod(absolute, dataset_size)
        return CandidateDatasetPosition(self.completed_passes + passes, cursor)


@dataclass(frozen=True)
class EffectiveSamplingCheckpoint:
    position: CandidateDatasetPosition
    acceptance_rate: float
    policy_version: int
    optimizer_step: int
    metric_totals: dict[str, float]

    @classmethod
    def from_dict(
        cls,
        payload: dict,
        *,
        dataset_size: int,
        expected_optimizer_step: int,
    ) -> EffectiveSamplingCheckpoint:
        if payload.get("schema_version") != 1:
            raise ValueError(
                f"unsupported strict MaxRL effective-sampling checkpoint schema: {payload.get('schema_version')!r}"
            )
        state = cls(
            position=CandidateDatasetPosition(
                int(payload["candidate_dataset_passes_completed"]),
                int(payload["candidate_prompts_in_current_pass"]),
            ),
            acceptance_rate=float(payload["acceptance_rate"]),
            policy_version=int(payload["policy_version"]),
            optimizer_step=int(payload["optimizer_step"]),
            metric_totals={str(key): float(value) for key, value in payload.get("metric_totals", {}).items()},
        )
        if state.position.completed_passes < 0 or not 0 <= state.position.cursor < dataset_size:
            raise ValueError("strict MaxRL checkpoint has an invalid candidate dataset position")
        if not 0 < state.acceptance_rate <= 1:
            raise ValueError("strict MaxRL checkpoint has an invalid acceptance rate")
        if state.optimizer_step != expected_optimizer_step or state.policy_version != expected_optimizer_step:
            raise ValueError("strict MaxRL checkpoint policy/optimizer identity does not match its folder")
        if any(not math.isfinite(value) or value < 0 for value in state.metric_totals.values()):
            raise ValueError("strict MaxRL checkpoint has invalid cumulative metrics")
        return state


def binary_success_from_rewards(token_level_rewards: torch.Tensor) -> torch.Tensor:
    """Return the exact binary outcome used by the MaxRL estimator."""

    if token_level_rewards.ndim < 2:
        raise ValueError(
            f"token_level_rewards must contain a token dimension; got shape={tuple(token_level_rewards.shape)}"
        )
    return token_level_rewards.sum(dim=-1).gt(0)


def effective_training_should_stop(
    *,
    global_step: int,
    configured_total_training_steps: int | None,
    candidate_dataset_passes_completed: int,
    target_candidate_dataset_passes: int,
    graceful_stop_requested: bool,
) -> bool:
    """Resolve the stop boundary for strict MaxRL training.

    Production runs are pass-bounded when ``total_training_steps`` is unset.
    An explicit step limit remains available to internal smoke and correctness
    runs without becoming a required user-facing configuration knob.
    """

    if graceful_stop_requested:
        return True
    if configured_total_training_steps is not None:
        return global_step >= configured_total_training_steps
    return target_candidate_dataset_passes > 0 and candidate_dataset_passes_completed >= target_candidate_dataset_passes


def classify_complete_groups(
    uids: Sequence[str],
    binary_success: torch.Tensor,
    *,
    responses_per_prompt: int,
    target_groups: int,
) -> GroupSelection:
    """Classify complete prompt groups and deterministically cap accepted groups."""

    if responses_per_prompt < 2:
        raise ValueError("responses_per_prompt must be at least 2")
    if target_groups <= 0:
        raise ValueError("target_groups must be positive")
    if binary_success.ndim != 1 or len(uids) != binary_success.numel():
        raise ValueError("uids and binary_success must be aligned one-dimensional rows")

    grouped_indices: dict[str, list[int]] = {}
    for row_index, uid in enumerate(uids):
        grouped_indices.setdefault(str(uid), []).append(row_index)

    classified: list[ClassifiedGroup] = []
    for uid, row_indices in grouped_indices.items():
        if len(row_indices) != responses_per_prompt:
            raise ValueError(
                f"incomplete MaxRL group {uid!r}: expected {responses_per_prompt} responses, got {len(row_indices)}"
            )
        success_count = int(binary_success[row_indices].sum().item())
        if success_count == 0:
            outcome = GroupOutcome.ALL_WRONG
        elif success_count == responses_per_prompt:
            outcome = GroupOutcome.ALL_CORRECT
        else:
            outcome = GroupOutcome.EFFECTIVE
        classified.append(
            ClassifiedGroup(
                uid=uid,
                row_indices=tuple(row_indices),
                success_count=success_count,
                outcome=outcome,
            )
        )

    effective = [group for group in classified if group.outcome is GroupOutcome.EFFECTIVE]
    accepted = effective[:target_groups]
    surplus = effective[target_groups:]
    rejected = [group for group in classified if group.outcome is not GroupOutcome.EFFECTIVE]
    return GroupSelection(tuple(accepted), tuple(rejected), tuple(surplus), tuple(classified))


@dataclass
class CandidateWavePlanner:
    """Plan bounded candidate waves from the observed effective-group rate."""

    target_groups: int
    group_quantum: int
    max_candidate_groups: int
    max_wave_groups: int
    acceptance_rate: float = 0.5
    smoothing: float = 0.25
    acceptance_floor: float = 0.05
    headroom: float = 1.10
    observed_candidates: int = 0

    def __post_init__(self) -> None:
        if self.target_groups <= 0 or self.group_quantum <= 0:
            raise ValueError("target_groups and group_quantum must be positive")
        if self.max_candidate_groups < self.target_groups:
            raise ValueError("max_candidate_groups must cover the target")
        if self.max_wave_groups < self.group_quantum:
            raise ValueError("max_wave_groups must cover one group quantum")

    def plan(self, accepted_groups: int) -> int:
        deficit = self.target_groups - accepted_groups
        if deficit <= 0:
            return 0
        remaining = self.max_candidate_groups - self.observed_candidates
        if remaining <= 0:
            return 0

        rate = max(self.acceptance_rate, self.acceptance_floor)
        estimated = math.ceil(deficit / rate * self.headroom)
        rounded = math.ceil(estimated / self.group_quantum) * self.group_quantum
        return min(max(rounded, self.group_quantum), self.max_wave_groups, remaining)

    def observe(self, candidate_groups: int, effective_groups: int) -> None:
        if candidate_groups <= 0 or not 0 <= effective_groups <= candidate_groups:
            raise ValueError("invalid candidate/effective observation")
        observed_rate = effective_groups / candidate_groups
        self.acceptance_rate = (1.0 - self.smoothing) * self.acceptance_rate + self.smoothing * observed_rate
        self.observed_candidates += candidate_groups
