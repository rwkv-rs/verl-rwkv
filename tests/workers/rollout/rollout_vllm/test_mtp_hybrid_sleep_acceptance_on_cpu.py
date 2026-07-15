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

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ray")
pytest.importorskip("vllm")

from verl.trainer.ppo.ray_trainer import compute_spec_decode_metrics
from verl.workers.rollout.vllm_rollout import vllm_async_server


class _FakeMtpEngine:
    """Minimal model of the MTP failure mode seen in hybrid sleep."""

    def __init__(self):
        self.mtp_drafter_available = True
        self.sleep_levels_that_discard_mtp_drafter = {2}

    async def sleep(self, level: int):
        if level in self.sleep_levels_that_discard_mtp_drafter:
            self.mtp_drafter_available = False

    async def reset_encoder_cache(self):
        pass

    async def wake_up(self, tags):
        self.wake_tags = tags

    async def reset_prefix_cache(self, **kwargs):
        self.reset_prefix_cache_kwargs = kwargs

    def sync_actor_weights(self):
        pass

    def generate_spec_decode_stats(self):
        num_draft_tokens = 3
        num_accepted_tokens = num_draft_tokens if self.mtp_drafter_available else 0
        num_verify_steps = 1
        return num_draft_tokens, num_accepted_tokens, num_verify_steps


def _make_colocated_server(*, lora_rank: int = 0, lora: dict | None = None):
    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.node_rank = 0
    server.rollout_mode = vllm_async_server.RolloutMode.COLOCATED
    server.config = SimpleNamespace(
        free_cache_engine=True,
        mtp=SimpleNamespace(enable=False, enable_rollout=False),
    )
    server.model_config = SimpleNamespace(lora_rank=lora_rank, lora=lora or {})
    server.engine = _FakeMtpEngine()
    server.weight_update_state = "active"
    server.weight_update_failure = None
    server.behavior_policy_identity = {"policy_version": 1}
    return server


def test_colocated_full_weight_sleep_discards_old_inference_weights(monkeypatch):
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)
    server = _make_colocated_server()

    asyncio.run(server.sleep())

    assert not server.engine.mtp_drafter_available
    assert server.weight_update_state == "sleeping"
    assert server.behavior_policy_identity is None


def test_colocated_adapter_sleep_keeps_base_weights(monkeypatch):
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)
    server = _make_colocated_server(lora_rank=8, lora={"rank": 8, "merge": False})

    asyncio.run(server.sleep())

    assert server.engine.mtp_drafter_available
    assert server.weight_update_state == "sleeping"


def test_colocated_wake_respects_two_phase_tags():
    server = _make_colocated_server()

    asyncio.run(server.wake_up(tags=["weights"]))

    assert server.engine.wake_tags == ["weights"]


def test_mtp_hybrid_sleep_keeps_drafter_available_for_nonzero_acceptance(monkeypatch):
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)

    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = SimpleNamespace(mtp=SimpleNamespace(enable=True, enable_rollout=True))
    server.model_config = SimpleNamespace(lora_rank=0, lora={})
    server.engine = _FakeMtpEngine()

    asyncio.run(server._sleep_hybrid())
    server.engine.sync_actor_weights()
    drafts, accepts, verifies = server.engine.generate_spec_decode_stats()
    metrics = compute_spec_decode_metrics(
        spec_drafts=np.array([drafts]),
        spec_accepts=np.array([accepts]),
        spec_verifies=np.array([verifies]),
    )

    assert metrics["rollout/spec_accept_rate"] > 0.0
    assert metrics["rollout/spec_accept_length"] > 1.0
