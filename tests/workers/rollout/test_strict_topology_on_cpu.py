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

import pytest
from fastapi import FastAPI

from verl.workers.rollout import llm_server
from verl.workers.rollout.llm_server import (
    LLMServerManager,
    resolve_rollout_topology,
    validate_strict_rollout_capacity,
)
from verl.workers.rollout.utils import run_uvicorn
from verl.workers.rollout.vllm_rollout import vllm_async_server
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


def _runtime_deployments(max_num_seqs: int = 32) -> list[dict]:
    return [
        {
            "replica_rank": replica_rank,
            "node_rank": 0,
            "capacity": {
                "capacity_source": "vllm.scheduler_config",
                "max_num_seqs": max_num_seqs,
                "max_num_batched_tokens": 128,
            },
        }
        for replica_rank in range(8)
    ]


def test_strict_capacity_gate_accepts_verified_runtime_capacity():
    validate_strict_rollout_capacity(
        _runtime_deployments(),
        expected_replicas=8,
        expected_max_num_seqs=32,
        expected_max_num_batched_tokens=128,
    )


def test_strict_capacity_gate_rejects_runtime_mismatch():
    with pytest.raises(RuntimeError, match="runtime capacity"):
        validate_strict_rollout_capacity(
            _runtime_deployments(max_num_seqs=16),
            expected_replicas=8,
            expected_max_num_seqs=32,
            expected_max_num_batched_tokens=128,
        )


def test_strict_capacity_gate_rejects_unverified_config_input_metadata():
    deployments = _runtime_deployments()
    deployments[0]["capacity"]["capacity_source"] = "rollout_config"

    with pytest.raises(RuntimeError, match="runtime capacity"):
        validate_strict_rollout_capacity(
            deployments,
            expected_replicas=8,
            expected_max_num_seqs=32,
            expected_max_num_batched_tokens=128,
        )


def test_vllm_server_runtime_metadata_attests_active_policy_and_capacity(monkeypatch):
    context = SimpleNamespace(
        get_node_id=lambda: "node-1",
        get_actor_id=lambda: "actor-1",
        get_actor_name=lambda: "vllm-server-1",
    )
    monkeypatch.setattr(vllm_async_server.ray, "get_runtime_context", lambda: context)
    monkeypatch.setenv("VLLM_RWKV7_WKV_MODE", "fp32io16")
    server = vLLMHttpServer.__new__(vLLMHttpServer)
    server.replica_rank = 1
    server.node_rank = 0
    server.cuda_visible_devices = "1"
    server._server_address = "10.0.0.2"
    server._server_port = 18001
    server._master_port = 19001
    server._dp_rpc_port = 19002
    server._dp_master_port = 19003
    server.behavior_policy_identity = {
        "policy_version": 7,
        "weight_digest": "publication-7",
        "sampling_config_digest": "sampling",
        "runtime_identity": "runtime",
    }
    server.weight_update_state = "active"
    server._runtime_capacity = {
        "capacity_source": "vllm.scheduler_config",
        "capacity_mode": "recurrent-state-no-kv-cache",
        "kv_cache_applicable": False,
        "max_num_seqs": 960,
        "max_num_batched_tokens": 65536,
        "max_model_len": 10240,
        "gpu_memory_utilization": 0.8,
    }

    metadata = server.get_runtime_metadata()

    assert metadata["http_endpoint"] == "10.0.0.2:18001"
    assert metadata["behavior_policy_identity"] == server.behavior_policy_identity
    assert metadata["behavior_policy_identity"] is not server.behavior_policy_identity
    assert metadata["weight_update_state"] == "active"
    assert metadata["wkv_mode"] == "fp32io16"
    assert metadata["capacity"] == server._runtime_capacity
    assert metadata["capacity"] is not server._runtime_capacity


def test_server_manager_reads_fresh_actor_metadata_and_checks_endpoints(monkeypatch):
    metadata = [
        {"http_endpoint": "10.0.0.1:18000", "weight_update_state": "active"},
        {"http_endpoint": "10.0.0.2:18001", "weight_update_state": "active"},
    ]

    class RemoteMethod:
        def __init__(self, value):
            self.value = value

        def remote(self):
            return self.value

    manager = LLMServerManager.__new__(LLMServerManager)
    manager.server_addresses = ["10.0.0.1:18000", "10.0.0.2:18001"]
    manager.server_handles = [SimpleNamespace(get_runtime_metadata=RemoteMethod(value)) for value in metadata]
    monkeypatch.setattr(llm_server.ray, "get", lambda values: values)

    snapshot = manager.get_runtime_metadata_snapshot()

    assert snapshot == metadata
    assert snapshot is not metadata
    metadata[1]["http_endpoint"] = "10.0.0.2:19001"
    with pytest.raises(RuntimeError, match="endpoints do not match"):
        manager.get_runtime_metadata_snapshot()


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
