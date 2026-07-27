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

import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_base import PPOTrainer


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
            "schema_version": 1,
            "global_step": 7,
            "wkv_mode": "fp32io16",
            "vllm_version": "0.23.1.dev0",
            "max_model_len": 10240,
            "replicas": [
                {
                    "base_url": "http://10.0.0.1:18000",
                    "max_concurrency": 64,
                },
                {
                    "base_url": "http://10.0.0.2:18001",
                    "max_concurrency": 64,
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
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.23.1.dev0")
    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_base.subprocess.run", run)
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
        llm_server_manager=SimpleNamespace(get_addresses=lambda: ["10.0.0.1:18000", "10.0.0.2:18001"]),
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
