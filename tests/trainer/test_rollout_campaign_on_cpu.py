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

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from verl.experimental.agent_loop.campaign_runtime import (
    CAMPAIGN_PROBLEM_ID_FIELD,
    CAMPAIGN_ROLLOUT_INDEX_FIELD,
    CampaignPolicyIdentity,
    RolloutCampaignRuntime,
    prepare_campaign_batch,
)
from verl.experimental.agent_loop.shard_artifact import RolloutCampaignArtifact
from verl.protocol import DataProto
from verl.trainer.rollout_campaign import (
    DAPO_MATH_PILOT_EXPECTED_RECORDS,
    DAPO_MATH_PILOT_PROBLEM_IDS_SHA256,
    DAPO_MATH_REVISION,
    deterministic_pilot_problem_ids,
    run_campaign_batches,
)
from verl.workers.rollout.llm_server import LLMServerManager

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_SEED = 20260801
SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": -1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "penalty_decay": 0.996,
    "logprobs": False,
}


def _make_tree_writable(path: Path) -> None:
    if not path.exists():
        return
    for root, directories, filenames in os.walk(path):
        for name in directories:
            (Path(root) / name).chmod(stat.S_IRWXU)
        for name in filenames:
            (Path(root) / name).chmod(stat.S_IRUSR | stat.S_IWUSR)
    path.chmod(stat.S_IRWXU)


class _Tokenizer:
    eos_token_id = 0

    @staticmethod
    def decode(token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return f"<think>work</think><answer>{token_ids[0]}</answer>"


class _AgentLoopManager:
    def __init__(self) -> None:
        self.calls = []

    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        pairs = list(
            zip(
                prompts.non_tensor_batch[CAMPAIGN_PROBLEM_ID_FIELD].tolist(),
                prompts.non_tensor_batch[CAMPAIGN_ROLLOUT_INDEX_FIELD].tolist(),
                strict=True,
            )
        )
        self.calls.append(pairs)
        size = len(prompts)
        return DataProto.from_dict(
            tensors={
                "prompts": torch.ones((size, 2), dtype=torch.long),
                "responses": torch.tensor([[101, 0, 0]] * size, dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 0]] * size, dtype=torch.long),
                "rm_scores": torch.tensor([[0.0, 1.0, 0.0]] * size),
            },
            non_tensors={
                "finish_reason": np.asarray(["stop"] * size, dtype=object),
                "backend_stop_reason": np.asarray([0] * size, dtype=object),
                "repetition_truncated": np.asarray([False] * size, dtype=object),
            },
        )


def _artifact(root: Path) -> RolloutCampaignArtifact:
    return RolloutCampaignArtifact(
        root,
        "campaign-entry-test",
        ["problem-a", "problem-b"],
        rollouts_per_problem=2,
        dataset_fingerprint="sha256:dataset",
        source_revision=DAPO_MATH_REVISION,
        parameters={
            "base_seed": BASE_SEED,
            "sampling": SAMPLING,
            "model_revision": "model@revision",
            "policy_lineage": "sha256:weights",
            "policy_version": 0,
            "source_lineage": "checkpoint:sha256:weights",
            "runtime_identity": "vllm-rwkv:model@revision",
        },
    )


def _policy() -> CampaignPolicyIdentity:
    return CampaignPolicyIdentity(
        sampling=SAMPLING,
        model_revision="model@revision",
        policy_lineage="sha256:weights",
        policy_version=0,
        source_lineage="checkpoint:sha256:weights",
        runtime_identity="vllm-rwkv:model@revision",
        base_seed=BASE_SEED,
    )


def _prepared_prompts() -> DataProto:
    prompts = DataProto.from_dict(
        non_tensors={
            "index": np.asarray(["problem-a", "problem-b"], dtype=object),
            "raw_prompt": np.asarray(
                [
                    [{"role": "user", "content": "A"}],
                    [{"role": "user", "content": "B"}],
                ],
                dtype=object,
            ),
        }
    )
    return prepare_campaign_batch(
        prompts,
        campaign_id="campaign-entry-test",
        rollouts_per_problem=2,
        base_seed=BASE_SEED,
    )


def test_deterministic_pilot_selection_is_independent_of_input_order():
    problem_ids = [f"problem-{index:05d}" for index in range(1000)]
    forward = deterministic_pilot_problem_ids(problem_ids, count=174)
    reverse = deterministic_pilot_problem_ids(list(reversed(problem_ids)), count=174)

    assert forward == reverse
    assert len(forward) == len(set(forward)) == 174


@pytest.mark.asyncio
async def test_campaign_entry_calls_manager_generate_resumes_and_promotes(tmp_path):
    prepared = _prepared_prompts()
    promoted = tmp_path / "campaign-entry-test"
    try:
        first_manager = _AgentLoopManager()
        first_runtime = RolloutCampaignRuntime(
            _artifact(tmp_path),
            first_manager.generate_sequences,
            _Tokenizer(),
            _policy(),
        )
        partial = await first_runtime.run_batch(prepared.select_idxs([0, 1]))
        assert partial.remaining == 2

        resumed_manager = _AgentLoopManager()
        resumed_runtime = RolloutCampaignRuntime(
            _artifact(tmp_path),
            resumed_manager.generate_sequences,
            _Tokenizer(),
            _policy(),
        )
        manifest_path = await run_campaign_batches(resumed_runtime, prepared, batch_size=1)

        assert manifest_path == promoted / "manifest.json"
        assert resumed_manager.calls == [[("problem-b", 0)], [("problem-b", 1)]]
        manifest = json.loads(manifest_path.read_bytes())
        assert manifest["counts"]["total"] == 4
        assert manifest["contract"]["expected_pair_count"] == 4
    finally:
        _make_tree_writable(promoted)


class _PublicationReplica:
    def __init__(self, replica_rank: int, *, fail: bool = False) -> None:
        self.replica_rank = replica_rank
        self.fail = fail
        self.servers = [object()]
        self.rollbacks = 0

    async def publish_loaded_policy_identity(self, policy_identity):
        if self.fail:
            raise RuntimeError(f"replica {self.replica_rank} failed")
        return [
            {
                "replica_rank": self.replica_rank,
                "node_rank": 0,
                "policy_identity": policy_identity,
            }
        ]

    async def rollback_loaded_policy_identity(self, policy_identity):
        self.rollbacks += 1


@pytest.mark.asyncio
async def test_standalone_policy_publication_requires_all_replica_ack_and_rolls_back():
    identity = {
        "policy_version": 0,
        "weight_digest": "sha256:weights",
        "sampling_config_digest": "sha256:sampling",
        "runtime_identity": "vllm-rwkv:runtime",
    }
    replicas = [_PublicationReplica(0), _PublicationReplica(1, fail=True)]
    manager = LLMServerManager.__new__(LLMServerManager)
    manager.rollout_replicas = replicas

    with pytest.raises(RuntimeError, match="all replica identities were rolled back"):
        await manager.publish_loaded_policy_identity(identity)

    assert [replica.rollbacks for replica in replicas] == [1, 1]


def test_dapo_math_174_by_256_cli_dry_run_is_read_only(tmp_path):
    datasets_root = Path(os.environ.get("DATASETS_PATH", "/home/caizus/Datasets"))
    dataset_path = datasets_root / "DAPO" / "dapo-math-17k-processed.parquet"
    if not dataset_path.is_file():
        pytest.skip(f"frozen DAPO-Math parquet is not available: {dataset_path}")
    artifact_root = tmp_path / "artifacts"
    environment = {
        **os.environ,
        "DATASETS_PATH": str(datasets_root),
        "WEIGHT_PATH": os.environ.get("WEIGHT_PATH", "/home/caizus/Weights"),
        "ROLLOUT_CAMPAIGN_ARTIFACT_ROOT": str(artifact_root),
        "PYTHONPATH": str(REPO_ROOT),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "verl.trainer.main_generation_server",
            "--config-name=dapo_math_17k_pilot",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    contract = json.loads(result.stdout)

    assert contract["mode"] == "dry-run-plan"
    assert contract["dataset"]["row_count"] == 17_398
    assert contract["selection"] == {
        "algorithm": "sha256(source_revision + NUL + problem_id)",
        "problem_count": 174,
        "problem_ids_sha256": DAPO_MATH_PILOT_PROBLEM_IDS_SHA256,
    }
    assert contract["rollouts_per_problem"] == 256
    assert contract["expected_records"] == DAPO_MATH_PILOT_EXPECTED_RECORDS == 44_544
    assert contract["batch_count"] == 87
    assert contract["recovery_command"].endswith("--config-name=dapo_math_17k_pilot")
    assert not artifact_root.exists(), "dry-run must not create campaign or watchdog artifacts"
