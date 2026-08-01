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

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_base import PPOTrainer


BEHAVIOR_POLICY_IDENTITY = {
    "policy_version": 7,
    "weight_digest": "publication-7",
    "sampling_config_digest": "sampling",
    "runtime_identity": "runtime",
}


def _runtime_metadata(
    address: str,
    port: int,
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int,
) -> dict[str, object]:
    return {
        "http_address": address,
        "http_port": port,
        "http_endpoint": f"{address}:{port}",
        "behavior_policy_identity": dict(BEHAVIOR_POLICY_IDENTITY),
        "weight_update_state": "active",
        "wkv_mode": "fp32io16",
        "vllm_version": "0.23.1.dev0",
        "capacity": {
            "capacity_source": "vllm.scheduler_config",
            "capacity_mode": "recurrent-state-no-kv-cache",
            "kv_cache_applicable": False,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_model_len": 10240,
            "gpu_memory_utilization": 0.8,
        },
    }


def test_external_evaluation_uses_checkpoint_command_and_result(tmp_path, monkeypatch):
    weight_root = tmp_path / "weights"
    checkpoint_root = weight_root / "maxrl" / "run"
    checkpoint_file = checkpoint_root / "global_step_7" / "actor" / "rwkv_lm.pth"
    events = []

    def save_checkpoint():
        checkpoint_file.parent.mkdir(parents=True)
        checkpoint_file.write_bytes(b"weights")
        (checkpoint_file.parent / "config.json").write_text("{}", encoding="utf-8")
        events.append("save")

    def run(command, *, cwd, env, check):
        assert command == ["helicopter", "eval", "--config", "maxrl.toml"]
        assert cwd == "/product/helicopter"
        assert check is True
        assert env["MAXRL_EVAL_WEIGHT"] == "maxrl/run/global_step_7/actor/rwkv_lm.pth"
        assert env["MAXRL_EVAL_STEP"] == "7"
        result_path = checkpoint_file.parent / "lighteval_metrics.json"
        assert env["MAXRL_EVAL_RESULT_PATH"] == str(result_path)
        pool_path = Path(env["HELICOPTER_VLLM_POOL_MANIFEST"])
        assert pool_path.is_file()
        assert pool_path.stat().st_mode & 0o777 == 0o600
        assert json.loads(pool_path.read_text(encoding="utf-8")) == {
            "schema_version": 3,
            "global_step": 7,
            "checkpoint_sha256": hashlib.sha256(b"weights").hexdigest(),
            "checkpoint_display_name": "rwkv_lm.pth",
            "policy_checkpoint_path": (
                "maxrl/run/global_step_7/actor/rwkv_lm.pth"
            ),
            "behavior_policy_identity": BEHAVIOR_POLICY_IDENTITY,
            "wkv_mode": "fp32io16",
            "vllm_version": "0.23.1.dev0",
            "max_model_len": 10240,
            "aggregate_scheduler_capacity": {
                "max_num_seqs": 160,
                "max_num_batched_tokens": 12288,
            },
            "replicas": [
                {
                    "base_url": "http://10.0.0.1:18000",
                    "behavior_policy_identity": BEHAVIOR_POLICY_IDENTITY,
                    "weight_update_state": "active",
                    "wkv_mode": "fp32io16",
                    "capacity_source": "vllm.scheduler_config",
                    "capacity_mode": "recurrent-state-no-kv-cache",
                    "kv_cache_applicable": False,
                    "max_num_seqs": 64,
                    "max_num_batched_tokens": 4096,
                    "max_model_len": 10240,
                    "gpu_memory_utilization": 0.8,
                },
                {
                    "base_url": "http://10.0.0.2:18001",
                    "behavior_policy_identity": BEHAVIOR_POLICY_IDENTITY,
                    "weight_update_state": "active",
                    "wkv_mode": "fp32io16",
                    "capacity_source": "vllm.scheduler_config",
                    "capacity_mode": "recurrent-state-no-kv-cache",
                    "kv_cache_applicable": False,
                    "max_num_seqs": 96,
                    "max_num_batched_tokens": 8192,
                    "max_model_len": 10240,
                    "gpu_memory_utilization": 0.8,
                },
            ],
        }
        result_path.write_text(
            json.dumps({"metrics": {"aime24/pass@1": 0.5, "math_500/pass@1": 0.25}}),
            encoding="utf-8",
        )
        events.append("command")

    monkeypatch.setenv("WEIGHT_PATH", str(weight_root))
    monkeypatch.setenv("VLLM_RWKV7_WKV_MODE", "fp32io16")
    monkeypatch.setenv("HELICOPTER_PRODUCT_ROOT", "/product/helicopter")
    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_base.subprocess.run", run)
    runtime_metadata = [
        _runtime_metadata(
            "10.0.0.1",
            18000,
            max_num_seqs=64,
            max_num_batched_tokens=4096,
        ),
        _runtime_metadata(
            "10.0.0.2",
            18001,
            max_num_seqs=96,
            max_num_batched_tokens=8192,
        ),
    ]
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "trainer": {"default_local_dir": str(checkpoint_root)},
                "data": {
                    "max_prompt_length": 2048,
                    "max_response_length": 8192,
                },
                "actor_rollout_ref": {
                    "rollout": {
                        "max_model_len": 10240,
                        "max_num_seqs": 64,
                    }
                },
            }
        ),
        global_steps=7,
        policy_round=SimpleNamespace(
            published=SimpleNamespace(as_dict=lambda: dict(BEHAVIOR_POLICY_IDENTITY))
        ),
        llm_server_manager=SimpleNamespace(
            get_addresses=lambda: ["10.0.0.1:18000", "10.0.0.2:18001"],
            get_runtime_metadata_snapshot=lambda: runtime_metadata,
        ),
        _save_checkpoint=save_checkpoint,
    )

    metrics = PPOTrainer._validate_external(
        trainer,
        OmegaConf.create({"command": ["helicopter", "eval", "--config", "maxrl.toml"]}),
    )

    assert metrics == {
        "val-core/aime24/pass@1": 0.5,
        "val-core/math_500/pass@1": 0.25,
    }
    assert events == ["save", "command"]
    assert not list(checkpoint_file.parent.glob(".vllm-eval-pool-*.json"))


def test_external_evaluation_rejects_stale_replica_policy_before_command(tmp_path, monkeypatch):
    weight_root = tmp_path / "weights"
    checkpoint_root = weight_root / "maxrl" / "run"
    checkpoint_file = checkpoint_root / "global_step_7" / "actor" / "rwkv_lm.pth"
    checkpoint_file.parent.mkdir(parents=True)
    checkpoint_file.write_bytes(b"weights")
    (checkpoint_file.parent / "config.json").write_text("{}", encoding="utf-8")
    stale = _runtime_metadata(
        "10.0.0.1",
        18000,
        max_num_seqs=64,
        max_num_batched_tokens=4096,
    )
    stale["behavior_policy_identity"] = {
        **BEHAVIOR_POLICY_IDENTITY,
        "policy_version": 6,
    }
    monkeypatch.setenv("WEIGHT_PATH", str(weight_root))
    monkeypatch.setenv("VLLM_RWKV7_WKV_MODE", "fp32io16")
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "trainer": {"default_local_dir": str(checkpoint_root)},
                "data": {"max_prompt_length": 2048, "max_response_length": 8192},
                "actor_rollout_ref": {"rollout": {}},
            }
        ),
        global_steps=7,
        policy_round=SimpleNamespace(
            published=SimpleNamespace(as_dict=lambda: dict(BEHAVIOR_POLICY_IDENTITY))
        ),
        llm_server_manager=SimpleNamespace(
            get_addresses=lambda: ["10.0.0.1:18000"],
            get_runtime_metadata_snapshot=lambda: [stale],
        ),
        _save_checkpoint=lambda: None,
    )

    with pytest.raises(RuntimeError, match="does not match the published policy"):
        PPOTrainer._validate_external(
            trainer,
            OmegaConf.create({"command": ["helicopter", "eval"]}),
        )


def test_external_evaluation_rejects_missing_published_policy_with_contract_error(tmp_path, monkeypatch):
    weight_root = tmp_path / "weights"
    checkpoint_root = weight_root / "maxrl" / "run"
    checkpoint_file = checkpoint_root / "global_step_7" / "actor" / "rwkv_lm.pth"
    checkpoint_file.parent.mkdir(parents=True)
    checkpoint_file.write_bytes(b"weights")
    (checkpoint_file.parent / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("WEIGHT_PATH", str(weight_root))
    monkeypatch.setenv("VLLM_RWKV7_WKV_MODE", "fp32io16")
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "trainer": {"default_local_dir": str(checkpoint_root)},
                "data": {"max_prompt_length": 2048, "max_response_length": 8192},
                "actor_rollout_ref": {"rollout": {}},
            }
        ),
        global_steps=7,
        llm_server_manager=SimpleNamespace(get_addresses=lambda: ["10.0.0.1:18000"]),
        _save_checkpoint=lambda: None,
    )

    with pytest.raises(RuntimeError, match="authoritative published behavior-policy identity"):
        PPOTrainer._validate_external(
            trainer,
            OmegaConf.create({"command": ["helicopter", "eval"]}),
        )
