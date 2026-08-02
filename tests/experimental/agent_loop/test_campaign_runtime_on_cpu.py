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

import numpy as np
import pytest
import torch

from verl.experimental.agent_loop.campaign_runtime import (
    CAMPAIGN_PROBLEM_ID_FIELD,
    CAMPAIGN_ROLLOUT_INDEX_FIELD,
    CAMPAIGN_SAMPLING_SEED_FIELD,
    CampaignPolicyIdentity,
    RolloutCampaignRuntime,
    prepare_campaign_batch,
)
from verl.experimental.agent_loop.shard_artifact import RolloutCampaignArtifact
from verl.protocol import DataProto
from verl.trainer.ppo.v1.policy_identity import IDENTITY_TAG_KEYS, canonical_digest

CAMPAIGN_ID = "dapo-math-17k-pilot"
BASE_SEED = 20260801
SAMPLING = {"temperature": 1.0, "top_p": 0.95}
MODEL_REVISION = "rwkv7-g1i@sha256:weights"


class _Tokenizer:
    eos_token_id = 0

    @staticmethod
    def decode(token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return f"<think>work</think><answer>{token_ids[0]}</answer>"


class _GenerateSequences:
    def __init__(self):
        self.calls = []
        self.policy_identities = []

    async def __call__(self, prompts):
        problem_ids = prompts.non_tensor_batch[CAMPAIGN_PROBLEM_ID_FIELD].tolist()
        rollout_indices = prompts.non_tensor_batch[CAMPAIGN_ROLLOUT_INDEX_FIELD].tolist()
        seeds = prompts.non_tensor_batch[CAMPAIGN_SAMPLING_SEED_FIELD].tolist()
        self.calls.append(list(zip(problem_ids, rollout_indices, seeds, strict=True)))
        self.policy_identities.append({key: set(prompts.non_tensor_batch[key].tolist()) for key in IDENTITY_TAG_KEYS})
        batch_size = len(prompts)
        first_tokens = torch.arange(101, 101 + batch_size, dtype=torch.long)
        responses = torch.stack(
            (first_tokens, torch.zeros(batch_size, dtype=torch.long), torch.zeros(batch_size, dtype=torch.long)),
            dim=1,
        )
        return DataProto.from_dict(
            tensors={
                "prompts": torch.full((batch_size, 2), 7, dtype=torch.long),
                "responses": responses,
                # The final zero in each response is padding; the preceding zero
                # is an EOS token and must remain in the persisted token IDs.
                "attention_mask": torch.tensor([[1, 1, 1, 1, 0]] * batch_size),
                "rm_scores": torch.tensor([[0.0, 1.0, 0.0]] * batch_size),
            },
            non_tensors={
                "finish_reason": np.asarray(["stop"] * batch_size, dtype=object),
                "backend_stop_reason": np.asarray([0] * batch_size, dtype=object),
                "repetition_truncated": np.asarray([False] * batch_size, dtype=object),
            },
        )


class _NeverGenerate:
    async def __call__(self, prompts):
        raise AssertionError(f"completed campaign unexpectedly generated {len(prompts)} responses")


def _artifact(root):
    return RolloutCampaignArtifact(
        root,
        CAMPAIGN_ID,
        ["problem-a", "problem-b"],
        rollouts_per_problem=2,
        dataset_fingerprint="sha256:dapo-pilot",
        source_revision="verl-rwkv@abc123",
        parameters={
            "base_seed": BASE_SEED,
            "sampling": SAMPLING,
            "model_revision": MODEL_REVISION,
            "policy_lineage": "policy-0:lineage-digest",
            "policy_version": 0,
            "source_lineage": "checkpoint:sha256:source",
            "runtime_identity": "vllm-rwkv:sha256:runtime",
        },
    )


def _policy(**overrides):
    values = {
        "sampling": SAMPLING,
        "model_revision": MODEL_REVISION,
        "policy_lineage": "policy-0:lineage-digest",
        "policy_version": 0,
        "source_lineage": "checkpoint:sha256:source",
        "runtime_identity": "vllm-rwkv:sha256:runtime",
        "base_seed": BASE_SEED,
    }
    values.update(overrides)
    return CampaignPolicyIdentity(**values)


def _prompts():
    return DataProto.from_dict(
        non_tensors={
            "index": np.asarray(["problem-a", "problem-b"], dtype=object),
            "raw_prompt": np.asarray(["A", "B"], dtype=object),
        }
    )


def _make_tree_writable(path):
    if not path.exists():
        return
    for root, directories, filenames in os.walk(path):
        for name in directories:
            (path.__class__(root) / name).chmod(stat.S_IRWXU)
        for name in filenames:
            (path.__class__(root) / name).chmod(stat.S_IRUSR | stat.S_IWUSR)
    path.chmod(stat.S_IRWXU)


@pytest.mark.asyncio
async def test_campaign_runtime_resumes_before_generation_and_promotes_immutable_artifact(tmp_path):
    prepared = prepare_campaign_batch(
        _prompts(),
        campaign_id=CAMPAIGN_ID,
        rollouts_per_problem=2,
        base_seed=BASE_SEED,
    )
    assert list(
        zip(
            prepared.non_tensor_batch[CAMPAIGN_PROBLEM_ID_FIELD],
            prepared.non_tensor_batch[CAMPAIGN_ROLLOUT_INDEX_FIELD],
            strict=True,
        )
    ) == [("problem-a", 0), ("problem-a", 1), ("problem-b", 0), ("problem-b", 1)]
    seeds = prepared.non_tensor_batch[CAMPAIGN_SAMPLING_SEED_FIELD]
    assert len(set(seeds.tolist())) == 4
    assert np.array_equal(
        seeds,
        prepare_campaign_batch(
            _prompts(),
            campaign_id=CAMPAIGN_ID,
            rollouts_per_problem=2,
            base_seed=BASE_SEED,
        ).non_tensor_batch[CAMPAIGN_SAMPLING_SEED_FIELD],
    )

    first_generate = _GenerateSequences()
    first_runtime = RolloutCampaignRuntime(_artifact(tmp_path), first_generate, _Tokenizer(), _policy())
    first = await first_runtime.run_batch(prepared.select_idxs([0, 1]))
    assert (first.generated, first.skipped_completed, first.recorded, first.remaining) == (2, 0, 2, 2)
    assert first.manifest_path is None

    resumed_generate = _GenerateSequences()
    resumed_runtime = RolloutCampaignRuntime(_artifact(tmp_path), resumed_generate, _Tokenizer(), _policy())
    completed = await resumed_runtime.run_batch(prepared)
    assert (completed.generated, completed.skipped_completed, completed.recorded, completed.remaining) == (2, 2, 2, 0)
    assert completed.manifest_path == tmp_path / CAMPAIGN_ID / "manifest.json"
    assert [[pair[:2] for pair in call] for call in resumed_generate.calls] == [[("problem-b", 0), ("problem-b", 1)]]
    assert resumed_generate.policy_identities == [
        {
            "policy_version": {0},
            "weight_digest": {"policy-0:lineage-digest"},
            "sampling_config_digest": {canonical_digest(SAMPLING)},
            "runtime_identity": {"vllm-rwkv:sha256:runtime"},
        }
    ]

    promoted = tmp_path / CAMPAIGN_ID
    try:
        records = [json.loads(path.read_bytes()) for path in sorted(promoted.glob("records/*/*.json"))]
        assert len(records) == 4
        assert {record["identity"]["seed"] for record in records} == set(seeds.tolist())
        assert len({record["identity"]["sampling_digest"] for record in records}) == 1
        assert {record["identity"]["model_revision"] for record in records} == {MODEL_REVISION}
        assert {record["identity"]["policy_lineage"] for record in records} == {"policy-0:lineage-digest"}
        assert {record["reward"] for record in records} == {1.0}
        assert all(record["finish_source"]["response_token_ids"][-1] == 0 for record in records)
        assert all(record["finish"]["format_valid"] for record in records)
        assert all(record["finish"]["ended_by_eos"] for record in records)

        manifest = json.loads(completed.manifest_path.read_bytes())
        assert manifest["counts"]["total"] == 4
        assert manifest["counts"]["ended_by_eos"] == 4
        assert len(manifest["records_sha256"]) == 64
        assert all(
            path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0
            for path in [promoted, promoted / "records", completed.manifest_path]
        )

        final_runtime = RolloutCampaignRuntime(_artifact(tmp_path), _NeverGenerate(), _Tokenizer(), _policy())
        replay = await final_runtime.run_batch(prepared)
        assert (replay.generated, replay.skipped_completed, replay.recorded, replay.remaining) == (0, 4, 0, 0)
        assert replay.manifest_path == completed.manifest_path
    finally:
        _make_tree_writable(promoted)


@pytest.mark.asyncio
async def test_campaign_runtime_rejects_changed_seed_before_generation(tmp_path):
    prepared = prepare_campaign_batch(
        _prompts(),
        campaign_id=CAMPAIGN_ID,
        rollouts_per_problem=2,
        base_seed=BASE_SEED,
    )
    prepared.non_tensor_batch[CAMPAIGN_SAMPLING_SEED_FIELD][0] += 1
    generator = _GenerateSequences()
    runtime = RolloutCampaignRuntime(_artifact(tmp_path), generator, _Tokenizer(), _policy())

    with pytest.raises(ValueError, match="non-deterministic sampling seed"):
        await runtime.run_batch(prepared)

    assert generator.calls == []


@pytest.mark.asyncio
async def test_campaign_runtime_resumes_complete_partial_campaign_without_regeneration(tmp_path, monkeypatch):
    prepared = prepare_campaign_batch(
        _prompts(),
        campaign_id=CAMPAIGN_ID,
        rollouts_per_problem=2,
        base_seed=BASE_SEED,
    )
    artifact = _artifact(tmp_path)
    runtime = RolloutCampaignRuntime(artifact, _GenerateSequences(), _Tokenizer(), _policy())

    def crash_before_promotion():
        raise RuntimeError("simulated crash before promotion")

    monkeypatch.setattr(artifact, "finalize", crash_before_promotion)
    with pytest.raises(RuntimeError, match="simulated crash"):
        await runtime.run_batch(prepared)
    assert artifact.remaining_count() == 0

    resumed = RolloutCampaignRuntime(_artifact(tmp_path), _NeverGenerate(), _Tokenizer(), _policy())
    result = await resumed.run_batch(prepared)
    try:
        assert (result.generated, result.skipped_completed, result.recorded, result.remaining) == (0, 4, 0, 0)
        assert result.manifest_path == tmp_path / CAMPAIGN_ID / "manifest.json"
    finally:
        _make_tree_writable(tmp_path / CAMPAIGN_ID)


def test_campaign_runtime_rejects_changed_policy_lineage_on_resume(tmp_path):
    with pytest.raises(ValueError, match="policy_lineage"):
        RolloutCampaignRuntime(
            _artifact(tmp_path),
            _NeverGenerate(),
            _Tokenizer(),
            _policy(policy_lineage="policy-0:different-lineage"),
        )


def test_prepare_campaign_batch_rejects_duplicate_problem_ids():
    prompts = DataProto.from_dict(non_tensors={"index": np.asarray(["same", "same"], dtype=object)})

    with pytest.raises(ValueError, match="unique problem ids"):
        prepare_campaign_batch(
            prompts,
            campaign_id=CAMPAIGN_ID,
            rollouts_per_problem=2,
            base_seed=BASE_SEED,
        )
