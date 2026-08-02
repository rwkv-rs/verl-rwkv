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

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verl.experimental.agent_loop.campaign_fields import (
    CAMPAIGN_PROBLEM_ID_FIELD,
    CAMPAIGN_ROLLOUT_INDEX_FIELD,
    CAMPAIGN_SAMPLING_SEED_FIELD,
    deterministic_campaign_seed,
)
from verl.experimental.agent_loop.shard_artifact import (
    RolloutCampaignArtifact,
    RolloutResponseIdentity,
)
from verl.protocol import DataProto
from verl.trainer.ppo.v1.policy_identity import IDENTITY_TAG_KEYS, canonical_digest
from verl.utils.ngram_repetition import ConsecutiveRepetitionDetector


@dataclass(frozen=True)
class CampaignPolicyIdentity:
    """Frozen model and sampling identity shared by every campaign response.

    ``policy_lineage`` is the committed publication's weight digest. It is
    stored under the campaign schema's lineage name and sent to rollout servers
    as ``weight_digest`` so the strict on-policy request check remains active.
    """

    sampling: Mapping[str, Any]
    model_revision: str
    policy_lineage: str
    policy_version: int
    source_lineage: str
    runtime_identity: str
    base_seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.sampling, Mapping) or not self.sampling:
            raise ValueError("campaign sampling must be a non-empty mapping")
        canonical_sampling = json.loads(json.dumps(self.sampling, sort_keys=True, separators=(",", ":")))
        for name in ("model_revision", "policy_lineage", "source_lineage", "runtime_identity"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"campaign {name} must be a non-empty string")
        if isinstance(self.policy_version, bool) or not isinstance(self.policy_version, int) or self.policy_version < 0:
            raise ValueError("campaign policy_version must be a non-negative integer")
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int):
            raise ValueError("campaign base_seed must be an integer")
        object.__setattr__(self, "sampling", canonical_sampling)


@dataclass(frozen=True)
class CampaignBatchResult:
    generated: int
    skipped_completed: int
    recorded: int
    remaining: int
    manifest_path: Path | None


def prepare_campaign_batch(
    prompts: DataProto,
    *,
    campaign_id: str,
    rollouts_per_problem: int,
    base_seed: int,
    problem_id_field: str = "index",
) -> DataProto:
    """Expand unique prompts into deterministic iid response identities."""

    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("campaign_id must be a non-empty string")
    if isinstance(rollouts_per_problem, bool) or not isinstance(rollouts_per_problem, int) or rollouts_per_problem <= 0:
        raise ValueError("rollouts_per_problem must be a positive integer")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise ValueError("base_seed must be an integer")
    if problem_id_field not in prompts.non_tensor_batch:
        raise ValueError(f"campaign prompt batch is missing {problem_id_field!r}")
    problem_ids = tuple(str(value) for value in prompts.non_tensor_batch[problem_id_field])
    if any(not problem_id for problem_id in problem_ids):
        raise ValueError("campaign problem ids must be non-empty strings")
    if len(problem_ids) != len(set(problem_ids)):
        raise ValueError("campaign prompt batch must contain unique problem ids before expansion")

    expanded = prompts.repeat(rollouts_per_problem, interleave=True)
    expanded_problem_ids = np.repeat(np.asarray(problem_ids, dtype=object), rollouts_per_problem)
    rollout_indices = np.tile(np.arange(rollouts_per_problem, dtype=np.int64), len(problem_ids))
    seeds = np.asarray(
        [
            deterministic_campaign_seed(campaign_id, problem_id, int(rollout_index), base_seed)
            for problem_id, rollout_index in zip(expanded_problem_ids, rollout_indices, strict=True)
        ],
        dtype=np.int64,
    )
    expanded.non_tensor_batch[CAMPAIGN_PROBLEM_ID_FIELD] = expanded_problem_ids
    expanded.non_tensor_batch[CAMPAIGN_ROLLOUT_INDEX_FIELD] = rollout_indices
    expanded.non_tensor_batch[CAMPAIGN_SAMPLING_SEED_FIELD] = seeds
    return expanded


class RolloutCampaignRuntime:
    """Run missing campaign pairs through the real agent-loop generation boundary."""

    def __init__(
        self,
        artifact: RolloutCampaignArtifact,
        generate_sequences: Callable[[DataProto], DataProto | Awaitable[DataProto]],
        tokenizer: Any,
        policy: CampaignPolicyIdentity,
        *,
        eos_token_ids: Sequence[int] | None = None,
        stop_token_ids: Sequence[int] = (),
    ) -> None:
        self.artifact = artifact
        self.generate_sequences = generate_sequences
        self.tokenizer = tokenizer
        self.policy = policy
        self.eos_token_ids = tuple(_normalize_token_ids(eos_token_ids, getattr(tokenizer, "eos_token_id", None)))
        self.stop_token_ids = tuple(_normalize_token_ids(stop_token_ids, None))
        parameters = artifact.contract["parameters"]
        expected = {
            "base_seed": policy.base_seed,
            "sampling": dict(policy.sampling),
            "model_revision": policy.model_revision,
            "policy_lineage": policy.policy_lineage,
            "policy_version": policy.policy_version,
            "source_lineage": policy.source_lineage,
            "runtime_identity": policy.runtime_identity,
        }
        mismatched = [name for name, value in expected.items() if parameters.get(name) != value]
        if mismatched:
            raise ValueError(f"campaign artifact parameters mismatch runtime fields: {', '.join(mismatched)}")
        # A resumed runtime pays one bounded directory scan. Subsequent batches
        # update this counter from atomic record creation instead of rescanning
        # a multi-million-response campaign after every generation call.
        self._remaining = artifact.remaining_count()

    async def run_batch(self, prompts: DataProto) -> CampaignBatchResult:
        """Skip completed pairs, generate missing responses, and atomically record them."""

        pairs, seeds = self._validate_batch_identity(prompts)
        pending_indices = [
            index
            for index, pair in enumerate(pairs)
            if not self.artifact.has_completion(
                *pair,
                expected_identity=self._response_identity(pair, seeds[index]),
            )
        ]
        skipped = len(pairs) - len(pending_indices)
        if not pending_indices:
            if self._remaining == 0 and not self.artifact.is_promoted:
                self.artifact.finalize()
            manifest = self.artifact.final_path if self._remaining == 0 else None
            return CampaignBatchResult(0, skipped, 0, self._remaining, manifest)

        pending_prompts = prompts.select_idxs(pending_indices)
        self._attach_behavior_policy_identity(pending_prompts)
        generated = self.generate_sequences(pending_prompts)
        if inspect.isawaitable(generated):
            generated = await generated
        if not isinstance(generated, DataProto) or len(generated) != len(pending_indices):
            actual = None if not isinstance(generated, DataProto) else len(generated)
            raise RuntimeError(
                f"campaign generation cardinality mismatch: expected {len(pending_indices)}, got {actual}"
            )

        recorded = 0
        for output_index, input_index in enumerate(pending_indices):
            problem_id, rollout_index = pairs[input_index]
            response_token_ids = self._response_token_ids(generated, output_index)
            response = self.tokenizer.decode(response_token_ids, skip_special_tokens=True)
            repetition_value = _non_tensor_value(generated, "repetition_truncated", output_index, None)
            repetition_truncated = (
                ConsecutiveRepetitionDetector().observe(response_token_ids) is not None
                if repetition_value is None
                else bool(repetition_value)
            )
            finish_reason = _non_tensor_value(generated, "finish_reason", output_index, None)
            if finish_reason is None:
                finish_reason = _non_tensor_value(generated, "stop_reason", output_index, None)
            backend_stop_reason = _non_tensor_value(generated, "backend_stop_reason", output_index, None)
            identity = self._response_identity((problem_id, rollout_index), seeds[input_index])
            recorded += int(
                self.artifact.write_completion(
                    identity,
                    response,
                    reward=self._reward(generated, output_index),
                    finish_reason=finish_reason,
                    backend_stop_reason=backend_stop_reason,
                    repetition_truncated=repetition_truncated,
                    response_token_ids=response_token_ids,
                    eos_token_ids=self.eos_token_ids,
                    stop_token_ids=self.stop_token_ids,
                )
            )

        self._remaining -= recorded
        if self._remaining < 0:
            raise RuntimeError("campaign runtime recorded more pairs than the artifact contract")
        manifest = None
        if self._remaining == 0:
            self.artifact.finalize()
            manifest = self.artifact.final_path
        return CampaignBatchResult(
            len(pending_indices),
            skipped,
            recorded,
            self._remaining,
            manifest,
        )

    def _validate_batch_identity(self, prompts: DataProto) -> tuple[list[tuple[str, int]], list[int]]:
        required = (CAMPAIGN_PROBLEM_ID_FIELD, CAMPAIGN_ROLLOUT_INDEX_FIELD, CAMPAIGN_SAMPLING_SEED_FIELD)
        missing = [field for field in required if field not in prompts.non_tensor_batch]
        if missing:
            raise ValueError(f"campaign batch is missing identity fields: {', '.join(missing)}")
        problem_ids = prompts.non_tensor_batch[CAMPAIGN_PROBLEM_ID_FIELD]
        rollout_indices = prompts.non_tensor_batch[CAMPAIGN_ROLLOUT_INDEX_FIELD]
        seeds = prompts.non_tensor_batch[CAMPAIGN_SAMPLING_SEED_FIELD]
        pairs: list[tuple[str, int]] = []
        normalized_seeds: list[int] = []
        for problem_id, rollout_index, seed in zip(problem_ids, rollout_indices, seeds, strict=True):
            if not isinstance(problem_id, str) or not problem_id:
                raise ValueError("campaign problem ids must be non-empty strings")
            if isinstance(rollout_index, bool) or not isinstance(rollout_index, int | np.integer):
                raise ValueError("campaign rollout indices must be integers")
            if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
                raise ValueError("campaign sampling seeds must be integers")
            pair = (problem_id, int(rollout_index))
            expected_seed = deterministic_campaign_seed(self.artifact.campaign_id, *pair, self.policy.base_seed)
            if int(seed) != expected_seed:
                raise ValueError(f"campaign pair {pair!r} has a non-deterministic sampling seed")
            pairs.append(pair)
            normalized_seeds.append(expected_seed)
        if len(pairs) != len(set(pairs)):
            raise ValueError("campaign batch contains duplicate problem-rollout pairs")
        unknown = [pair for pair in pairs if not self.artifact.contains_pair(*pair)]
        if unknown:
            raise ValueError(f"campaign batch contains pairs outside the artifact contract: {unknown}")
        return pairs, normalized_seeds

    def _response_identity(self, pair: tuple[str, int], seed: int) -> RolloutResponseIdentity:
        problem_id, rollout_index = pair
        return RolloutResponseIdentity(
            prompt_id=problem_id,
            sample_index=rollout_index,
            seed=seed,
            sampling=self.policy.sampling,
            model_revision=self.policy.model_revision,
            policy_lineage=self.policy.policy_lineage,
            policy_version=self.policy.policy_version,
            source_lineage=self.policy.source_lineage,
            runtime_identity=self.policy.runtime_identity,
        )

    def _attach_behavior_policy_identity(self, prompts: DataProto) -> None:
        identity = {
            "policy_version": self.policy.policy_version,
            "weight_digest": self.policy.policy_lineage,
            "sampling_config_digest": canonical_digest(self.policy.sampling),
            "runtime_identity": self.policy.runtime_identity,
        }
        for key in IDENTITY_TAG_KEYS:
            expected = identity[key]
            existing = prompts.non_tensor_batch.get(key)
            if existing is not None and any(value != expected for value in existing):
                raise ValueError(f"campaign batch {key} conflicts with the committed policy identity")
            prompts.non_tensor_batch[key] = np.asarray([expected] * len(prompts), dtype=object)

    @staticmethod
    def _response_token_ids(output: DataProto, index: int) -> list[int]:
        required = {"prompts", "responses", "attention_mask"}
        if output.batch is None or not required.issubset(output.batch.keys()):
            raise RuntimeError("campaign generation output is missing prompts, responses, or attention_mask")
        prompt_width = output.batch["prompts"].shape[1]
        response_attention = output.batch["attention_mask"][index, prompt_width:].bool()
        return output.batch["responses"][index][response_attention].tolist()

    @staticmethod
    def _reward(output: DataProto, index: int) -> float:
        if output.batch is not None and "rm_scores" in output.batch:
            return float(output.batch["rm_scores"][index].sum().item())
        reward = _non_tensor_value(output, "reward_score", index, None)
        if isinstance(reward, bool) or not isinstance(reward, int | float | np.number):
            raise RuntimeError("campaign generation output is missing a numeric reward")
        return float(reward)


def _normalize_token_ids(values: Sequence[int] | int | None, fallback: Sequence[int] | int | None) -> list[int]:
    selected = fallback if values is None else values
    if selected is None:
        return []
    if isinstance(selected, int):
        selected = [selected]
    normalized = list(selected)
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in normalized):
        raise ValueError("campaign stop token ids must contain integers")
    return normalized


def _non_tensor_value(output: DataProto, name: str, index: int, default: Any) -> Any:
    values = output.non_tensor_batch.get(name)
    if values is None:
        return default
    value = values[index]
    return value.item() if isinstance(value, np.generic) else value


__all__ = [
    "CAMPAIGN_PROBLEM_ID_FIELD",
    "CAMPAIGN_ROLLOUT_INDEX_FIELD",
    "CAMPAIGN_SAMPLING_SEED_FIELD",
    "CampaignBatchResult",
    "CampaignPolicyIdentity",
    "RolloutCampaignRuntime",
    "prepare_campaign_batch",
]
