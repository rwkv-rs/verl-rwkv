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

from copy import deepcopy
from pathlib import Path

import pytest
import tomllib
from hydra import compose, initialize_config_dir

from verl.trainer.maxrl import (
    MaxRLConfigError,
    build_overrides,
    context_tokens_from_checkpoint,
)

ROOT = Path(__file__).resolve().parents[2]
ENV = {
    "WEIGHT_PATH": "/weights",
    "DATASETS_PATH": "/datasets",
    "RWKV_LM_PATH": "/src/rwkv-lm",
}
CONFIG_TOML = """
[experiment]
name = "maxrl-dapo-math-17k"
project = "helicopter-math"
seed = 42
candidate_dataset_passes = 10

[model]
name = "g1h-7.2b"
checkpoint = "/weights/rwkv7/pth/rwkv7-g1h-7.2b-20260710-ctx10240.pth"
prompt_mode = "open_think"
prompt_template = "\\nBot✿"

[data.train]
files = ["/datasets/DAPO/dapo-math-17k-processed.parquet"]
prompt_field = "source_prompt"

[algorithm]
name = "maxrl"
prompts_per_step = 32
responses_per_prompt = 16
ppo_clip = 0.2
dual_clip = 3.0
entropy_coefficient = 0.0
kl_coefficient = 0.0

[reward]
manager = "dapo"
scorer = "math_verify"

[optimizer]
learning_rate = 1e-6
warmup_steps = 0
weight_decay = 0.01
gradient_norm_limit = 0.3

[generation.train]
temperature = 1.0
top_k = -1
top_p = 0.95

[execution]
nodes = 1
gpus_per_node = 8
wkv_mode = "fp32io16"
context_mode = "state_passing"
state_chunk_tokens = 2048

[execution.rollout]
replicas = 8
tensor_parallel_size_per_replica = 1
pipeline_parallel_size_per_replica = 1
max_concurrent_sequences_per_replica = 64
generation_token_budget_per_replica = 8192
weight_update_bucket_mib = 64

[evaluation]
before_training = true
every_optimizer_steps = 50
command = [
  "helicopter",
  "eval",
  "--config",
  "configs/eval/maxrl_math.toml",
  "--env-file",
  ".env.remote",
]

[checkpoint]
every_optimizer_steps = 50
directory = "/weights/maxrl/maxrl-dapo-math-17k"

[logging]
backends = ["console", "file", "wandb"]
"""


def config() -> dict:
    return tomllib.loads(CONFIG_TOML)


def resolved(overrides: list[str]) -> dict[str, str]:
    result = {}
    for override in overrides:
        if "=" in override:
            key, value = override.split("=", 1)
            result[key.lstrip("+")] = value
    return result


def test_compiles_strict_maxrl_contract() -> None:
    overrides, child_env = build_overrides(config(), env=ENV)
    values = resolved(overrides)

    assert values["algorithm.adv_estimator"] == "maxrl"
    assert values["trainer.v1.trainer_mode"] == "sync"
    assert values["actor_rollout_ref.actor.ppo_mini_batch_size"] == "32"
    assert values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == "1"
    assert values["actor_rollout_ref.rollout.n"] == "16"
    assert values["actor_rollout_ref.rollout.max_model_len"] == "10240"
    assert values["actor_rollout_ref.rollout.ignore_eos"] == "False"
    assert values["actor_rollout_ref.rollout.top_p"] == "0.95"
    assert values["data.val_files"] == "null"
    assert values["trainer.val_before_train"] == "True"
    assert values["trainer.test_freq"] == "50"
    assert values["trainer.default_local_dir"] == "/weights/maxrl/maxrl-dapo-math-17k"
    assert values["trainer.external_evaluation.command"] == (
        '["helicopter","eval","--config","configs/eval/maxrl_math.toml","--env-file",".env.remote"]'
    )
    assert not any(key.startswith("actor_rollout_ref.rollout.val_kwargs.") for key in values)
    assert values["data.max_prompt_length"] == "null"
    assert values["data.max_response_length"] == "null"
    assert values["data.train_files"] == "['/datasets/DAPO/dapo-math-17k-processed.parquet']"
    assert values["data.train_prompt_key"] == "source_prompt"
    assert "data.model_context_length" not in values
    assert child_env["VLLM_RWKV7_WKV_MODE"] == "fp32io16"


def test_user_override_cannot_disable_strict_eos_or_sync_mode() -> None:
    with pytest.raises(MaxRLConfigError, match="MaxRL override"):
        build_overrides(
            config(),
            env=ENV,
            extra_overrides=["actor_rollout_ref.rollout.ignore_eos=True"],
        )
    with pytest.raises(MaxRLConfigError, match="MaxRL override"):
        build_overrides(
            config(),
            env=ENV,
            extra_overrides=["trainer.v1.trainer_mode=async"],
        )


@pytest.mark.parametrize(
    "override",
    [
        "data.max_prompt_length=512",
        "data.max_response_length=4096",
        "actor_rollout_ref.rollout.max_model_len=8192",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.actor.ppo_mini_batch_size=16",
        "actor_rollout_ref.model.path=/weights/other-ctx10240.pth",
        "data.train_files=[/datasets/raw.parquet]",
        "data={train_batch_size:1}",
        "~actor_rollout_ref.rollout",
        "trainer.total_epochs=1",
    ],
)
def test_user_override_cannot_replace_derived_training_contract(override: str) -> None:
    with pytest.raises(MaxRLConfigError, match="MaxRL override"):
        build_overrides(config(), env=ENV, extra_overrides=[override])


def test_user_override_accepts_documented_operational_fields() -> None:
    overrides, _ = build_overrides(
        config(),
        env=ENV,
        extra_overrides=[
            "trainer.resume_mode=auto",
            "trainer.save_freq=10",
            "actor_rollout_ref.rollout.max_num_seqs=128",
        ],
    )

    values = resolved(overrides)
    assert values["trainer.resume_mode"] == "auto"
    assert values["trainer.save_freq"] == "10"
    assert values["actor_rollout_ref.rollout.max_num_seqs"] == "128"


def test_compiler_output_composes_with_real_hydra_schema() -> None:
    overrides, _ = build_overrides(config(), env=ENV)
    config_dir = str((ROOT / "verl/trainer/config").resolve())

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        composed = compose(config_name="ppo_trainer", overrides=overrides)

    assert composed.data.train_files == ["/datasets/DAPO/dapo-math-17k-processed.parquet"]
    assert composed.data.train_prompt_key == "source_prompt"
    assert composed.actor_rollout_ref.rollout.n == 16
    assert composed.actor_rollout_ref.rollout.max_model_len == 10240
    assert composed.actor_rollout_ref.rollout.response_length == 10240
    assert composed.data.val_files is None
    assert composed.trainer.val_before_train is True
    assert composed.trainer.test_freq == 50
    assert list(composed.trainer.external_evaluation.command) == [
        "helicopter",
        "eval",
        "--config",
        "configs/eval/maxrl_math.toml",
        "--env-file",
        ".env.remote",
    ]


def test_removed_user_knobs_are_rejected() -> None:
    modified = deepcopy(config())
    modified["generation"]["train"]["stop_on_eos"] = False
    with pytest.raises(MaxRLConfigError, match="removed field"):
        build_overrides(modified, env=ENV)

    for section, value in (
        ("data.validation", {"suites": []}),
        ("generation.validation", {"temperature": 0.96}),
    ):
        modified = deepcopy(config())
        target = modified
        parts = section.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
        with pytest.raises(MaxRLConfigError, match="removed table"):
            build_overrides(modified, env=ENV)


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ("rwkv7-g1h-ctx10240.pth", 10240),
        ("/weights/rwkv7-g1g-ctx8192-test.pth", 8192),
    ],
)
def test_context_is_derived_from_checkpoint(checkpoint: str, expected: int) -> None:
    assert context_tokens_from_checkpoint(checkpoint) == expected


def test_context_requires_exactly_one_suffix() -> None:
    with pytest.raises(MaxRLConfigError, match="exactly one"):
        context_tokens_from_checkpoint("rwkv7-g1h.pth")
