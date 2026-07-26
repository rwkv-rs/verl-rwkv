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

import pytest
from fastapi import FastAPI

from verl.workers.rollout.llm_server import resolve_rollout_topology
from verl.workers.rollout.utils import run_uvicorn
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


def test_single_gpu_topology_resolves_to_eight_independent_replicas():
    assert resolve_rollout_topology(
        world_size=8,
        tensor_parallel_size=1,
        data_parallel_size=1,
        pipeline_parallel_size=1,
    ) == (1, 8)


def test_topology_rejects_unused_gpu_remainder():
    with pytest.raises(ValueError, match="all GPUs"):
        resolve_rollout_topology(
            world_size=8,
            tensor_parallel_size=3,
            data_parallel_size=1,
            pipeline_parallel_size=1,
        )


def test_duplicate_endpoint_gate_is_represented_by_unique_addresses():
    endpoints = [f"127.0.0.1:{port}" for port in range(30000, 30008)]
    assert len(set(endpoints)) == 8


@pytest.mark.asyncio
async def test_concurrent_single_gpu_servers_receive_distinct_auto_ports():
    app_a, app_b = FastAPI(), FastAPI()
    port_a, task_a = await run_uvicorn(app_a, object(), "127.0.0.1")
    port_b, task_b = await run_uvicorn(app_b, object(), "127.0.0.1")
    try:
        assert port_a > 0
        assert port_b > 0
        assert port_a != port_b
    finally:
        task_a.cancel()
        task_b.cancel()
        await asyncio.gather(task_a, task_b, return_exceptions=True)


@pytest.mark.asyncio
async def test_vllm_server_rejects_request_for_a_different_behavior_policy_before_generation():
    server = vLLMHttpServer.__new__(vLLMHttpServer)
    server.weight_update_state = "active"
    server.behavior_policy_identity = {
        "policy_version": 4,
        "weight_digest": "publication-4",
        "sampling_config_digest": "sampling",
        "runtime_identity": "runtime",
    }
    stale = {**server.behavior_policy_identity, "policy_version": 3}

    with pytest.raises(RuntimeError, match="does not match"):
        await server.generate(
            prompt_ids=[1],
            sampling_params={"temperature": 1.0},
            request_id="stale-request",
            expected_policy_identity=stale,
        )


@pytest.mark.asyncio
async def test_vllm_server_poisoned_weight_update_clears_identity_and_rejects_generation():
    server = vLLMHttpServer.__new__(vLLMHttpServer)
    server.weight_update_state = "active"
    server.weight_update_failure = None
    server.behavior_policy_identity = {"policy_version": 4}

    await server.begin_weight_update()
    assert server.weight_update_state == "updating"
    assert server.behavior_policy_identity is None

    await server.poison_weight_update("partial IPC failure")
    assert server.weight_update_state == "poisoned"
    assert server.behavior_policy_identity is None
    with pytest.raises(RuntimeError, match="publication state.*poisoned"):
        await server.generate(
            prompt_ids=[1],
            sampling_params={"temperature": 1.0},
            request_id="poisoned-request",
            expected_policy_identity={"policy_version": 4},
        )


@pytest.mark.asyncio
async def test_vllm_server_publishes_identity_only_after_successful_update():
    server = vLLMHttpServer.__new__(vLLMHttpServer)
    server.weight_update_state = "active"
    server.weight_update_failure = None
    server.behavior_policy_identity = {"policy_version": 4}

    await server.begin_weight_update()
    await server.stage_behavior_policy_identity({"policy_version": 5})

    assert server.weight_update_state == "weights_ready"
    assert server.behavior_policy_identity == {"policy_version": 5}

    with pytest.raises(RuntimeError, match="publication state.*weights_ready"):
        await server.generate(
            prompt_ids=[1],
            sampling_params={"temperature": 1.0},
            request_id="not-yet-active",
            expected_policy_identity={"policy_version": 5},
        )

    await server.activate_weight_update()
    assert server.weight_update_state == "active"
