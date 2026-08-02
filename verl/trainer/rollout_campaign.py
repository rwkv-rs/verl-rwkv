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

"""Frozen DAPO-Math rollout campaign planning and AgentLoop execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verl.trainer.maxrl import G1I_MODEL_ASSET

DAPO_MATH_REPOSITORY = "open-r1/DAPO-Math-17k-Processed"
DAPO_MATH_REVISION = "31dd309567e3da778038cc87d868b6097a3ccf68"
DAPO_MATH_FILENAME = "dapo-math-17k-processed.parquet"
DAPO_MATH_SHA256 = "94ee2f742411118f8fe79fbcb4cdec661f7ac7bbd83808a55c6437623fbfed3a"
DAPO_MATH_ROW_COUNT = 17_398
DAPO_MATH_PILOT_PROBLEM_COUNT = 174
DAPO_MATH_PILOT_PROBLEM_IDS_SHA256 = "901dcbea975ea4c5d6d59bff6c0862e0ed5e766ba6116d41875904d7b46a27fa"
DAPO_MATH_ROLLOUTS_PER_PROBLEM = 256
DAPO_MATH_PILOT_EXPECTED_RECORDS = DAPO_MATH_PILOT_PROBLEM_COUNT * DAPO_MATH_ROLLOUTS_PER_PROBLEM
DAPO_MATH_CAMPAIGN_ID = "dapo-math-17k-g1i-preview-1pct"
DAPO_MATH_CONFIG_NAME = "dapo_math_17k_pilot"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class RolloutCampaignConfigError(ValueError):
    """Raised before Ray or vLLM starts when the frozen campaign drifts."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(config: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(config, Mapping) and not hasattr(config, "get"):
        raise RolloutCampaignConfigError(f"{name} must be a mapping")
    return config


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RolloutCampaignConfigError(f"{name} must be {expected!r}; got {actual!r}")


def deterministic_pilot_problem_ids(
    problem_ids: Sequence[str],
    *,
    source_revision: str = DAPO_MATH_REVISION,
    count: int = DAPO_MATH_PILOT_PROBLEM_COUNT,
) -> tuple[str, ...]:
    """Select a stable iid-problem pilot independently of parquet row order."""

    normalized = tuple(str(problem_id) for problem_id in problem_ids)
    if any(not problem_id for problem_id in normalized):
        raise RolloutCampaignConfigError("dataset problem ids must be non-empty strings")
    if len(normalized) != len(set(normalized)):
        raise RolloutCampaignConfigError("dataset problem ids must be unique")
    if not isinstance(count, int) or isinstance(count, bool) or not 0 < count <= len(normalized):
        raise RolloutCampaignConfigError("pilot problem count must be within the dataset")
    return tuple(
        sorted(
            normalized,
            key=lambda problem_id: (
                hashlib.sha256(f"{source_revision}\0{problem_id}".encode()).digest(),
                problem_id,
            ),
        )[:count]
    )


def problem_ids_digest(problem_ids: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_bytes(list(problem_ids))).hexdigest()


def resolved_campaign_sampling(rollout_config: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror the base sampling identity consumed by AgentLoopWorker.

    The real runtime compares this result with
    ``build_agent_loop_sampling_params`` before it publishes the checkpoint
    identity to vLLM. Per-response seeds are added after this base digest.
    """

    return {
        "temperature": rollout_config.get("temperature"),
        "top_p": rollout_config.get("top_p"),
        "top_k": rollout_config.get("top_k"),
        "presence_penalty": rollout_config.get("presence_penalty", 0.0),
        "frequency_penalty": rollout_config.get("frequency_penalty", 0.0),
        "repetition_penalty": rollout_config.get("repetition_penalty", 1.0),
        "penalty_decay": rollout_config.get("penalty_decay", 0.996),
        "logprobs": rollout_config.get("calculate_log_probs", False),
    }


def _normalize_messages(value: Any, problem_id: str) -> tuple[dict[str, Any], ...]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list) or not value:
        raise RolloutCampaignConfigError(f"problem {problem_id} source_prompt must be a non-empty chat list")
    messages = []
    for message in value:
        if not isinstance(message, Mapping) or not isinstance(message.get("role"), str):
            raise RolloutCampaignConfigError(f"problem {problem_id} has an invalid source_prompt message")
        if not isinstance(message.get("content"), str):
            raise RolloutCampaignConfigError(f"problem {problem_id} has a non-text source_prompt message")
        messages.append(dict(message))
    return tuple(messages)


@dataclass(frozen=True)
class CampaignPrompt:
    problem_id: str
    raw_prompt: tuple[dict[str, Any], ...]
    data_source: str
    reward_model: Mapping[str, Any]
    extra_info: Mapping[str, Any]


@dataclass(frozen=True)
class DapoMathPilotPlan:
    campaign_id: str
    run_id: str
    dataset_path: Path
    artifact_root: Path
    problem_ids: tuple[str, ...]
    prompts: tuple[CampaignPrompt, ...]
    base_seed: int
    batch_size: int
    sampling: Mapping[str, Any]
    model_revision: str
    policy_lineage: str
    policy_version: int
    source_lineage: str
    runtime_identity: str
    recovery_command: str

    @property
    def expected_records(self) -> int:
        return len(self.problem_ids) * DAPO_MATH_ROLLOUTS_PER_PROBLEM

    @property
    def artifact_path(self) -> Path:
        return self.artifact_root / self.campaign_id

    @property
    def partial_artifact_path(self) -> Path:
        return self.artifact_root / f".{self.campaign_id}.campaign"

    @property
    def run_state_path(self) -> Path:
        return self.artifact_root / "runs" / f"{self.run_id}.json"

    @property
    def plan_digest(self) -> str:
        return _canonical_digest(self.contract_material())

    def contract_material(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "dataset": {
                "repository": DAPO_MATH_REPOSITORY,
                "revision": DAPO_MATH_REVISION,
                "filename": DAPO_MATH_FILENAME,
                "sha256": DAPO_MATH_SHA256,
                "row_count": DAPO_MATH_ROW_COUNT,
            },
            "selection": {
                "algorithm": "sha256(source_revision + NUL + problem_id)",
                "problem_count": len(self.problem_ids),
                "problem_ids_sha256": problem_ids_digest(self.problem_ids),
            },
            "rollouts_per_problem": DAPO_MATH_ROLLOUTS_PER_PROBLEM,
            "expected_records": self.expected_records,
            "base_seed": self.base_seed,
            "sampling": dict(self.sampling),
            "model_revision": self.model_revision,
            "policy_lineage": self.policy_lineage,
            "policy_version": self.policy_version,
            "source_lineage": self.source_lineage,
            "runtime_identity": self.runtime_identity,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": "dry-run-plan",
            "run_id": self.run_id,
            "plan_digest": self.plan_digest,
            **self.contract_material(),
            "batch_size": self.batch_size,
            "batch_count": (self.expected_records + self.batch_size - 1) // self.batch_size,
            "artifact_path": str(self.artifact_path),
            "partial_artifact_path": str(self.partial_artifact_path),
            "run_state_path": str(self.run_state_path),
            "recovery_command": self.recovery_command,
        }


def _validate_frozen_config(config: Mapping[str, Any]) -> None:
    campaign = _mapping(config.get("rollout_campaign"), "rollout_campaign")
    model = _mapping(config.get("actor_rollout_ref").get("model"), "actor_rollout_ref.model")
    rollout = _mapping(config.get("actor_rollout_ref").get("rollout"), "actor_rollout_ref.rollout")
    trainer = _mapping(config.get("trainer"), "trainer")

    frozen_campaign = {
        "campaign_id": DAPO_MATH_CAMPAIGN_ID,
        "source_repository": DAPO_MATH_REPOSITORY,
        "source_revision": DAPO_MATH_REVISION,
        "dataset_filename": DAPO_MATH_FILENAME,
        "dataset_sha256": DAPO_MATH_SHA256,
        "dataset_rows": DAPO_MATH_ROW_COUNT,
        "pilot_problem_count": DAPO_MATH_PILOT_PROBLEM_COUNT,
        "pilot_problem_ids_sha256": DAPO_MATH_PILOT_PROBLEM_IDS_SHA256,
        "rollouts_per_problem": DAPO_MATH_ROLLOUTS_PER_PROBLEM,
    }
    for key, expected in frozen_campaign.items():
        _require_equal(f"rollout_campaign.{key}", campaign.get(key), expected)
    for key, expected in G1I_MODEL_ASSET.items():
        _require_equal(f"actor_rollout_ref.model.{key}", model.get(key), expected)
    _require_equal("actor_rollout_ref.rollout.name", rollout.get("name"), "vllm")
    _require_equal("actor_rollout_ref.rollout.mode", rollout.get("mode"), "async")
    _require_equal("actor_rollout_ref.rollout.n", rollout.get("n"), 1)
    _require_equal("actor_rollout_ref.rollout.tensor_model_parallel_size", rollout.get("tensor_model_parallel_size"), 1)
    _require_equal("actor_rollout_ref.rollout.data_parallel_size", rollout.get("data_parallel_size"), 1)
    _require_equal(
        "actor_rollout_ref.rollout.pipeline_model_parallel_size", rollout.get("pipeline_model_parallel_size"), 1
    )
    _require_equal("actor_rollout_ref.rollout.nnodes", rollout.get("nnodes"), 1)
    _require_equal("actor_rollout_ref.rollout.n_gpus_per_node", rollout.get("n_gpus_per_node"), 8)
    _require_equal("trainer.nnodes", trainer.get("nnodes"), 1)
    _require_equal("trainer.n_gpus_per_node", trainer.get("n_gpus_per_node"), 8)
    if rollout.get("calculate_log_probs") is not False:
        raise RolloutCampaignConfigError("campaign generation must disable rollout logprobs")
    batch_size = campaign.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise RolloutCampaignConfigError("rollout_campaign.batch_size must be a positive integer")
    base_seed = campaign.get("base_seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise RolloutCampaignConfigError("rollout_campaign.base_seed must be an integer")


def load_dapo_math_pilot_plan(config: Mapping[str, Any]) -> DapoMathPilotPlan:
    """Validate the immutable source and materialize the exact 174-problem plan."""

    _validate_frozen_config(config)
    campaign = config.get("rollout_campaign")
    model = config.get("actor_rollout_ref").get("model")
    rollout = config.get("actor_rollout_ref").get("rollout")
    dataset_path = Path(str(campaign.get("dataset_path"))).expanduser().resolve()
    if not dataset_path.is_file():
        raise RolloutCampaignConfigError(f"frozen DAPO-Math parquet not found: {dataset_path}")
    actual_sha256 = _sha256_file(dataset_path)
    _require_equal("DAPO-Math parquet sha256", actual_sha256, DAPO_MATH_SHA256)

    import pandas as pd

    frame = pd.read_parquet(
        dataset_path,
        columns=["source_prompt", "data_source", "reward_model", "extra_info"],
    )
    _require_equal("DAPO-Math parquet row count", len(frame), DAPO_MATH_ROW_COUNT)
    problem_ids = []
    rows_by_problem_id = {}
    for row in frame.itertuples(index=False):
        extra_info = dict(row.extra_info)
        problem_id = str(extra_info.get("index", ""))
        problem_ids.append(problem_id)
        rows_by_problem_id[problem_id] = row
    if len(rows_by_problem_id) != DAPO_MATH_ROW_COUNT:
        raise RolloutCampaignConfigError("frozen DAPO-Math extra_info.index values must be unique")
    selected_ids = deterministic_pilot_problem_ids(problem_ids)
    _require_equal(
        "DAPO-Math pilot problem-id digest",
        problem_ids_digest(selected_ids),
        DAPO_MATH_PILOT_PROBLEM_IDS_SHA256,
    )
    prompts = []
    for problem_id in selected_ids:
        row = rows_by_problem_id[problem_id]
        reward_model = dict(row.reward_model)
        if not isinstance(reward_model.get("ground_truth"), str):
            raise RolloutCampaignConfigError(f"problem {problem_id} has no string ground truth")
        prompts.append(
            CampaignPrompt(
                problem_id=problem_id,
                raw_prompt=_normalize_messages(row.source_prompt, problem_id),
                data_source=str(row.data_source),
                reward_model=reward_model,
                extra_info=dict(row.extra_info),
            )
        )

    sampling = resolved_campaign_sampling(rollout)
    model_revision = f"{model.get('repository')}@{model.get('revision')}:{model.get('filename')}"
    policy_lineage = f"sha256:{model.get('sha256')}"
    runtime_identity = f"vllm-rwkv:{model_revision}:{policy_lineage}"
    artifact_root = Path(str(campaign.get("artifact_root"))).expanduser().resolve()
    recovery_command = f"python -m verl.trainer.main_generation_server --config-name={DAPO_MATH_CONFIG_NAME}"
    provisional_run_id = (
        f"{DAPO_MATH_CAMPAIGN_ID}-{_canonical_digest({'model': model_revision, 'dataset': actual_sha256})[:12]}"
    )
    run_id = os.getenv("HELICOPTER_RUN_ID", provisional_run_id)
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise RolloutCampaignConfigError(f"HELICOPTER_RUN_ID is not path-safe: {run_id!r}")
    plan = DapoMathPilotPlan(
        campaign_id=DAPO_MATH_CAMPAIGN_ID,
        run_id=run_id,
        dataset_path=dataset_path,
        artifact_root=artifact_root,
        problem_ids=selected_ids,
        prompts=tuple(prompts),
        base_seed=int(campaign.get("base_seed")),
        batch_size=int(campaign.get("batch_size")),
        sampling=sampling,
        model_revision=model_revision,
        policy_lineage=policy_lineage,
        policy_version=0,
        source_lineage=f"checkpoint:{policy_lineage}",
        runtime_identity=runtime_identity,
        recovery_command=recovery_command,
    )
    _require_equal("DAPO-Math pilot expected records", plan.expected_records, DAPO_MATH_PILOT_EXPECTED_RECORDS)
    return plan


class CampaignRunJournal:
    """Atomically updated watchdog state outside the immutable campaign tree."""

    def __init__(self, plan: DapoMathPilotPlan) -> None:
        self.plan = plan
        self.path = plan.run_state_path
        self.started_at = time.time()

    def update(self, status: str, *, completed: int, remaining: int, error: str | None = None) -> None:
        payload = {
            "schema_version": 1,
            "run_id": self.plan.run_id,
            "campaign_id": self.plan.campaign_id,
            "plan_digest": self.plan.plan_digest,
            "status": status,
            "completed": completed,
            "remaining": remaining,
            "expected_records": self.plan.expected_records,
            "artifact_path": str(self.plan.artifact_path),
            "partial_artifact_path": str(self.plan.partial_artifact_path),
            "recovery_command": self.plan.recovery_command,
            "started_at_unix": self.started_at,
            "updated_at_unix": time.time(),
            "error": error,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as stream:
                stream.write(_canonical_bytes(payload))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)


def prepare_dapo_math_pilot_batch(plan: DapoMathPilotPlan):
    """Build and validate the exact 174 x 256 AgentLoop campaign batch."""

    from verl.experimental.agent_loop.campaign_runtime import (
        CAMPAIGN_PROBLEM_ID_FIELD,
        CAMPAIGN_ROLLOUT_INDEX_FIELD,
        CAMPAIGN_SAMPLING_SEED_FIELD,
        prepare_campaign_batch,
    )
    from verl.protocol import DataProto

    prompt_batch = DataProto.from_dict(
        non_tensors={
            "index": np.asarray(plan.problem_ids, dtype=object),
            "raw_prompt": np.asarray([list(prompt.raw_prompt) for prompt in plan.prompts], dtype=object),
            "data_source": np.asarray([prompt.data_source for prompt in plan.prompts], dtype=object),
            "reward_model": np.asarray([dict(prompt.reward_model) for prompt in plan.prompts], dtype=object),
            "extra_info": np.asarray([dict(prompt.extra_info) for prompt in plan.prompts], dtype=object),
        }
    )
    prepared = prepare_campaign_batch(
        prompt_batch,
        campaign_id=plan.campaign_id,
        rollouts_per_problem=DAPO_MATH_ROLLOUTS_PER_PROBLEM,
        base_seed=plan.base_seed,
    )
    if len(prepared) != plan.expected_records:
        raise RuntimeError(
            f"campaign expansion cardinality mismatch: expected={plan.expected_records} actual={len(prepared)}"
        )
    identities = set(
        zip(
            prepared.non_tensor_batch[CAMPAIGN_PROBLEM_ID_FIELD].tolist(),
            prepared.non_tensor_batch[CAMPAIGN_ROLLOUT_INDEX_FIELD].tolist(),
            prepared.non_tensor_batch[CAMPAIGN_SAMPLING_SEED_FIELD].tolist(),
            strict=True,
        )
    )
    if len(identities) != plan.expected_records:
        raise RuntimeError("campaign expansion contains duplicate problem-rollout-seed identities")
    return prepared


async def run_campaign_batches(
    runtime: Any,
    prepared: Any,
    *,
    batch_size: int,
    progress: Any | None = None,
) -> Path:
    """Run stable slices through ``RolloutCampaignRuntime.run_batch``."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("campaign batch_size must be a positive integer")
    remaining = runtime.artifact.remaining_count()
    if progress is not None:
        progress(remaining)
    manifest_path = runtime.artifact.final_path if runtime.artifact.is_promoted else None
    for start in range(0, len(prepared), batch_size):
        indices = list(range(start, min(start + batch_size, len(prepared))))
        result = await runtime.run_batch(prepared.select_idxs(indices))
        remaining = result.remaining
        manifest_path = result.manifest_path or manifest_path
        if progress is not None:
            progress(remaining)
    if remaining != 0 or not runtime.artifact.is_promoted or manifest_path is None:
        raise RuntimeError(f"campaign ended without immutable promotion: remaining={remaining}")
    return manifest_path


async def run_dapo_math_pilot(config: Any, plan: DapoMathPilotPlan) -> Path:
    """Generate the frozen pilot through standalone vLLM and AgentLoopManager."""

    from verl.experimental.agent_loop import AgentLoopManager
    from verl.experimental.agent_loop.agent_loop import build_agent_loop_sampling_params
    from verl.experimental.agent_loop.campaign_runtime import CampaignPolicyIdentity, RolloutCampaignRuntime
    from verl.experimental.agent_loop.shard_artifact import RolloutCampaignArtifact
    from verl.experimental.reward_loop import RewardLoopManager
    from verl.trainer.ppo.v1.policy_identity import BehaviorPolicyIdentity, canonical_digest
    from verl.utils import omega_conf_to_dataclass
    from verl.workers.rollout.llm_server import LLMServerManager

    model_config = omega_conf_to_dataclass(config.actor_rollout_ref.model)
    actual_sampling = build_agent_loop_sampling_params(
        omega_conf_to_dataclass(config.actor_rollout_ref.rollout),
        validate=False,
    )
    if actual_sampling != dict(plan.sampling):
        raise RuntimeError(
            "campaign sampling identity changed between dry-run and AgentLoop: "
            f"planned={dict(plan.sampling)} actual={actual_sampling}"
        )
    behavior_identity = BehaviorPolicyIdentity(
        policy_version=plan.policy_version,
        weight_digest=plan.policy_lineage,
        sampling_config_digest=canonical_digest(plan.sampling),
        runtime_identity=plan.runtime_identity,
    )
    llm_server_manager = await LLMServerManager.create(config=config)
    await llm_server_manager.publish_loaded_policy_identity(behavior_identity.as_dict())
    reward_loop_manager = RewardLoopManager(config=config)
    agent_loop_manager = await AgentLoopManager.create(
        config=config,
        llm_client=llm_server_manager.get_client(),
        reward_loop_worker_handles=reward_loop_manager.reward_loop_worker_handles,
    )

    artifact = RolloutCampaignArtifact(
        plan.artifact_root,
        plan.campaign_id,
        plan.problem_ids,
        rollouts_per_problem=DAPO_MATH_ROLLOUTS_PER_PROBLEM,
        dataset_fingerprint=f"sha256:{DAPO_MATH_SHA256}",
        source_revision=DAPO_MATH_REVISION,
        parameters={
            "base_seed": plan.base_seed,
            "sampling": dict(plan.sampling),
            "model_revision": plan.model_revision,
            "policy_lineage": plan.policy_lineage,
            "policy_version": plan.policy_version,
            "source_lineage": plan.source_lineage,
            "runtime_identity": plan.runtime_identity,
        },
    )
    prepared = prepare_dapo_math_pilot_batch(plan)
    prepared.non_tensor_batch["priority"] = np.arange(len(prepared), dtype=np.int64)
    runtime = RolloutCampaignRuntime(
        artifact,
        agent_loop_manager.generate_sequences,
        model_config.tokenizer,
        CampaignPolicyIdentity(
            sampling=plan.sampling,
            model_revision=plan.model_revision,
            policy_lineage=plan.policy_lineage,
            policy_version=plan.policy_version,
            source_lineage=plan.source_lineage,
            runtime_identity=plan.runtime_identity,
            base_seed=plan.base_seed,
        ),
    )
    journal = CampaignRunJournal(plan)
    remaining = artifact.remaining_count()

    def record_progress(current_remaining: int) -> None:
        nonlocal remaining
        remaining = current_remaining
        journal.update(
            "running",
            completed=plan.expected_records - remaining,
            remaining=remaining,
        )

    try:
        manifest_path = await run_campaign_batches(
            runtime,
            prepared,
            batch_size=plan.batch_size,
            progress=record_progress,
        )
        journal.update("completed", completed=plan.expected_records, remaining=0)
        return manifest_path
    except BaseException as exc:
        journal.update(
            "failed",
            completed=plan.expected_records - remaining,
            remaining=remaining,
            error=repr(exc),
        )
        raise


def print_campaign_plan(plan: DapoMathPilotPlan) -> None:
    json.dump(plan.as_dict(), sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


__all__ = [
    "CampaignRunJournal",
    "DAPO_MATH_CAMPAIGN_ID",
    "DAPO_MATH_PILOT_EXPECTED_RECORDS",
    "DAPO_MATH_PILOT_PROBLEM_COUNT",
    "DAPO_MATH_PILOT_PROBLEM_IDS_SHA256",
    "DAPO_MATH_REVISION",
    "DAPO_MATH_ROLLOUTS_PER_PROBLEM",
    "DAPO_MATH_ROW_COUNT",
    "DAPO_MATH_SHA256",
    "DapoMathPilotPlan",
    "RolloutCampaignConfigError",
    "deterministic_pilot_problem_ids",
    "load_dapo_math_pilot_plan",
    "prepare_dapo_math_pilot_batch",
    "print_campaign_plan",
    "problem_ids_digest",
    "resolved_campaign_sampling",
    "run_campaign_batches",
    "run_dapo_math_pilot",
]
