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

import os

import ray

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


def test_strict_runtime_contract_crosses_ray_boundary_by_allowlist(monkeypatch):
    forwarded = {
        "REMOTE_RUN_LOG_DIR": "/workspace/.helicopter-dev/runs/test",
        "VERL_FILE_LOGGER_PATH": "/workspace/.helicopter-dev/runs/test/metrics.jsonl",
        "VERL_POLICY_IDENTITY_LOG_PATH": "/workspace/.helicopter-dev/runs/test/policy_identity.jsonl",
        "VERL_POLICY_PUBLICATION_STATE_PATH": "/workspace/.helicopter-dev/runs/test/policy_publication.json",
        "HELICOPTER_RUN_ID": "run-1",
        "HELICOPTER_MODEL_REPOSITORY": "BlinkDL/temp-latest-training-models",
        "HELICOPTER_MODEL_REVISION": "d5db8cdf837726ef65a22724c86fa2b6ca95d3d8",
        "HELICOPTER_MODEL_FILENAME": "rwkv7-g1i_preview5445-1.5b-20260729-ctx16384.pth",
        "HELICOPTER_CHECKPOINT_SHA256": "a" * 64,
        "VLLM_RWKV7_WKV_MODE": "fp32io16",
    }
    for key, value in forwarded.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.setenv("VLLM_UNRELATED_DOTENV_VALUE", "must-not-forward")

    runtime_env = get_ppo_ray_runtime_env()["env_vars"]

    assert {key: runtime_env[key] for key in forwarded} == forwarded
    assert "VLLM_USE_V1" not in runtime_env
    assert "VLLM_UNRELATED_DOTENV_VALUE" not in runtime_env


def test_strict_runtime_contract_is_visible_inside_ray_actor(monkeypatch):
    forwarded = {
        "REMOTE_RUN_LOG_DIR": "/workspace/.helicopter-dev/runs/test",
        "VERL_FILE_LOGGER_PATH": "/workspace/.helicopter-dev/runs/test/metrics.jsonl",
        "VERL_POLICY_IDENTITY_LOG_PATH": "/workspace/.helicopter-dev/runs/test/policy_identity.jsonl",
        "VERL_POLICY_PUBLICATION_STATE_PATH": "/workspace/.helicopter-dev/runs/test/policy_publication.json",
        "HELICOPTER_RUN_ID": "run-actor",
        "HELICOPTER_MODEL_REPOSITORY": "BlinkDL/temp-latest-training-models",
        "HELICOPTER_MODEL_REVISION": "d5db8cdf837726ef65a22724c86fa2b6ca95d3d8",
        "HELICOPTER_MODEL_FILENAME": "rwkv7-g1i_preview5445-1.5b-20260729-ctx16384.pth",
        "HELICOPTER_CHECKPOINT_SHA256": "b" * 64,
        "VLLM_RWKV7_WKV_MODE": "fp32io16",
    }
    for key, value in forwarded.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("VLLM_USE_V1", raising=False)

    @ray.remote
    def capture_environment():
        return {key: os.environ.get(key) for key in (*forwarded, "VLLM_USE_V1")}

    ray.init(num_cpus=1, include_dashboard=False, runtime_env=get_ppo_ray_runtime_env())
    try:
        observed = ray.get(capture_environment.remote())
    finally:
        ray.shutdown()

    assert {key: observed[key] for key in forwarded} == forwarded
    assert observed["VLLM_USE_V1"] is None
