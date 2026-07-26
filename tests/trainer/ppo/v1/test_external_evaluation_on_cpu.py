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

import json
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
        events.append("save")

    class CheckpointManager:
        def sleep_replicas(self):
            events.append("sleep")

        def wake_up_replicas(self):
            events.append("wake")

    def run(command, *, env, check):
        assert command == ["helicopter", "eval", "--config", "maxrl.toml"]
        assert check is True
        assert env["MAXRL_EVAL_WEIGHT"] == "maxrl/run/global_step_7/actor/rwkv_lm.pth"
        assert env["MAXRL_EVAL_STEP"] == "7"
        result_path = checkpoint_file.parent / "lighteval_metrics.json"
        assert env["MAXRL_EVAL_RESULT_PATH"] == str(result_path)
        result_path.write_text(
            json.dumps({"metrics": {"aime24/pass@1": 0.5, "math_500/pass@1": 0.25}}),
            encoding="utf-8",
        )
        events.append("command")

    monkeypatch.setenv("WEIGHT_PATH", str(weight_root))
    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_base.subprocess.run", run)
    trainer = SimpleNamespace(
        config=OmegaConf.create({"trainer": {"default_local_dir": str(checkpoint_root)}}),
        global_steps=7,
        checkpoint_manager=CheckpointManager(),
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
    assert events == ["save", "sleep", "command", "wake"]
