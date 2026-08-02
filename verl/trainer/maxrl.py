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

"""RWKV MaxRL's public configuration and training entry point."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping

import tomllib
from hydra.core.override_parser.types import Quote, QuotedString

CONTEXT_SUFFIX_RE = re.compile(r"(?:^|[-_.])ctx(?P<tokens>[1-9]\d*)(?=[-_.]|$)")
ENV_REFERENCE_RE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}")
REQUIRED_SECTIONS = {
    "experiment",
    "model",
    "data",
    "algorithm",
    "reward",
    "optimizer",
    "generation",
    "execution",
    "evaluation",
    "checkpoint",
    "logging",
}
REMOVED_TABLES = (
    "data.validation",
    "generation.validation",
)
REMOVED_FIELDS = (
    ("model", "context_tokens"),
    ("data.train", "max_prompt_tokens"),
    ("generation.train", "max_response_tokens"),
    ("generation.train", "stop_on_eos"),
    ("execution", "dynamic_microbatching"),
    ("execution", "train_token_budget_per_gpu"),
)
OPERATIONAL_OVERRIDE_KEYS = frozenset(
    {
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes",
        "trainer.default_local_dir",
        "trainer.experiment_name",
        "trainer.logger",
        "trainer.project_name",
        "trainer.resume_from_path",
        "trainer.resume_mode",
        "trainer.save_freq",
    }
)
G1I_MODEL_ASSET = {
    "repository": "BlinkDL/temp-latest-training-models",
    "revision": "d5db8cdf837726ef65a22724c86fa2b6ca95d3d8",
    "filename": "rwkv7-g1i_preview5445-1.5b-20260729-ctx16384.pth",
    "sha256": "22fe129988f6e98480b344075597259a13ae4201c1d8dedf987246772e613586",
}


class MaxRLConfigError(ValueError):
    """The user-facing MaxRL experiment file is invalid."""


def _table(config: Mapping[str, Any], path: str) -> dict[str, Any]:
    value: Any = config
    for component in path.split("."):
        if not isinstance(value, Mapping):
            raise MaxRLConfigError(f"MaxRL config requires [{path}]")
        value = value.get(component)
    if not isinstance(value, dict):
        raise MaxRLConfigError(f"MaxRL config requires [{path}]")
    return value


def _required(section: Mapping[str, Any], key: str, *, section_name: str) -> Any:
    value = section.get(key)
    if value is None or value == "":
        raise MaxRLConfigError(f"MaxRL config requires {section_name}.{key}")
    return value


def _expand_string(value: str, env: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in env:
            raise MaxRLConfigError(f"environment variable is not set: {name}")
        return env[name]

    return ENV_REFERENCE_RE.sub(replace, value)


def _expand(value: Any, env: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _expand_string(value, env)
    if isinstance(value, list):
        return [_expand(item, env) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, env) for key, item in value.items()}
    return value


def read_config(path: Path, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Read one complete MaxRL TOML without loading or merging a runtime profile."""

    if not path.is_file():
        raise MaxRLConfigError(f"MaxRL config file not found: {path}")
    with path.open("rb") as handle:
        config = _expand(tomllib.load(handle), os.environ if env is None else env)
    missing = sorted(REQUIRED_SECTIONS.difference(config))
    if missing:
        raise MaxRLConfigError("MaxRL config is missing sections: " + ", ".join(missing))
    algorithm = _table(config, "algorithm")
    if _required(algorithm, "name", section_name="algorithm") != "maxrl":
        raise MaxRLConfigError("verl.trainer.maxrl requires algorithm.name='maxrl'")
    _validate_removed_fields(config)
    return config


def _validate_removed_fields(config: Mapping[str, Any]) -> None:
    for section_path in REMOVED_TABLES:
        value: Any = config
        for component in section_path.split("."):
            if not isinstance(value, Mapping) or component not in value:
                break
            value = value[component]
        else:
            raise MaxRLConfigError(f"MaxRL config contains removed table: {section_path}")
    for section_path, key in REMOVED_FIELDS:
        if key in _table(config, section_path):
            raise MaxRLConfigError(f"MaxRL config contains removed field: {section_path}.{key}")


def context_tokens_from_checkpoint(checkpoint: str) -> int:
    """Derive the only model context limit from the checkpoint's ``ctxN`` suffix."""

    filename = checkpoint.replace("\\", "/").rsplit("/", 1)[-1]
    matches = [int(match.group("tokens")) for match in CONTEXT_SUFFIX_RE.finditer(filename)]
    if len(matches) != 1:
        raise MaxRLConfigError(
            f"model.checkpoint filename must contain exactly one context suffix such as 'ctx10240'; got {filename!r}"
        )
    if matches[0] < 2:
        raise MaxRLConfigError("model context must leave room for prompt and response tokens")
    return matches[0]


def validate_g1i_model_asset(model: Mapping[str, Any]) -> None:
    """Bind the formal MaxRL rollout to the root-published g1i asset contract."""

    for key, expected in G1I_MODEL_ASSET.items():
        actual = _required(model, key, section_name="model")
        if actual != expected:
            raise MaxRLConfigError(f"model.{key} must match the published g1i asset: {expected!r}")
    legacy_checkpoint = str(_required(model, "legacy_checkpoint", section_name="model"))
    legacy_filename = legacy_checkpoint.replace("\\", "/").rsplit("/", 1)[-1]
    if legacy_filename != G1I_MODEL_ASSET["filename"]:
        raise MaxRLConfigError(
            "model.legacy_checkpoint filename does not match model.filename: "
            f"expected={G1I_MODEL_ASSET['filename']!r} actual={legacy_filename!r}"
        )
    checkpoint = str(_required(model, "checkpoint", section_name="model"))
    if checkpoint.lower().endswith(".pth"):
        raise MaxRLConfigError(
            "model.checkpoint must be a converted Hugging Face artifact directory; convert model.legacy_checkpoint "
            "once with `python -m transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf`"
        )


def _hydra(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _quoted(value: Any) -> str:
    return QuotedString(text=str(value), quote=Quote.double).with_quotes()


def _files(value: Any, *, name: str) -> str:
    if not isinstance(value, list) or not value:
        raise MaxRLConfigError(f"{name} must be a non-empty list")
    return "[" + ",".join(repr(str(path)) for path in value) + "]"


def _positive_int(value: Any, *, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise MaxRLConfigError(f"{name} must be an integer") from exc
    if result <= 0:
        raise MaxRLConfigError(f"{name} must be positive")
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise MaxRLConfigError(f"{name} must be an integer") from exc
    if result < 0:
        raise MaxRLConfigError(f"{name} must be non-negative")
    return result


def _reward_path(scorer: str, verl_root: Path) -> Path:
    aliases = {
        "math_verify": "math_verify_reward.py",
        "math_dapo": "math_dapo_reward.py",
        "dapo": "math_dapo_reward.py",
    }
    if scorer in aliases:
        return verl_root / "examples" / "rwkv_trainer" / aliases[scorer]
    candidate = Path(scorer)
    if candidate.suffix == ".py":
        return candidate if candidate.is_absolute() else verl_root / candidate
    raise MaxRLConfigError(f"unsupported reward.scorer: {scorer!r}")


def _resolved_map(overrides: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for override in overrides:
        if "=" in override:
            key, value = override.split("=", 1)
            result[key.lstrip("+")] = value
    return result


def _validate_extra_overrides(overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise MaxRLConfigError(f"MaxRL override must use KEY=VALUE syntax: {override!r}")
        raw_key, _ = override.split("=", 1)
        key = raw_key.lstrip("+")
        if raw_key.startswith("~") or key not in OPERATIONAL_OVERRIDE_KEYS:
            allowed = ", ".join(sorted(OPERATIONAL_OVERRIDE_KEYS))
            raise MaxRLConfigError(
                f"MaxRL override may only change operational fields; got {raw_key!r}. Allowed keys: {allowed}"
            )


def _validate_resolved(
    overrides: list[str],
    *,
    checkpoint_path: str,
    context_tokens: int,
    prompts_per_step: int,
    responses_per_prompt: int,
    candidate_dataset_passes: int,
    max_optimizer_steps: int | None,
    validation_before_training: bool,
    validation_interval: int,
) -> None:
    resolved = _resolved_map(overrides)
    required = {
        "algorithm.adv_estimator": "maxrl",
        "data.max_prompt_length": "null",
        "data.max_response_length": "null",
        "data.val_files": "null",
        "data.filter_overlong_prompts": "False",
        "data.truncation": "error",
        "data.train_batch_size": str(prompts_per_step),
        "actor_rollout_ref.model.path": checkpoint_path,
        "actor_rollout_ref.model.use_remove_padding": "False",
        "actor_rollout_ref.model.enable_gradient_checkpointing": "False",
        "trainer.v1.trainer_mode": "sync",
        "trainer.total_epochs": str(candidate_dataset_passes),
        "actor_rollout_ref.hybrid_engine": "True",
        "actor_rollout_ref.actor.ppo_epochs": "1",
        "actor_rollout_ref.actor.ppo_mini_batch_size": str(prompts_per_step),
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "1",
        "actor_rollout_ref.actor.use_dynamic_bsz": "False",
        "actor_rollout_ref.actor.use_torch_compile": "False",
        "actor_rollout_ref.actor.checkpoint.save_contents": "[model,optimizer,extra,hf_model]",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": "1",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": "False",
        "actor_rollout_ref.ref.use_torch_compile": "False",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": "1",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": "False",
        "actor_rollout_ref.rollout.prompt_length": str(context_tokens),
        "actor_rollout_ref.rollout.response_length": str(context_tokens),
        "actor_rollout_ref.rollout.max_model_len": str(context_tokens),
        "actor_rollout_ref.rollout.n": str(responses_per_prompt),
        "actor_rollout_ref.rollout.ignore_eos": "False",
        "actor_rollout_ref.rollout.top_p": "0.95",
        "actor_rollout_ref.rollout.checkpoint_engine.backend": "naive",
        "trainer.val_before_train": _hydra(validation_before_training),
        "trainer.test_freq": str(validation_interval),
        "algorithm.rollout_correction.rollout_is": "token",
        "algorithm.rollout_correction.rollout_is_threshold": "2.0",
        "algorithm.rollout_correction.rollout_is_batch_normalize": "False",
        "algorithm.rollout_correction.rollout_rs": "null",
        "algorithm.rollout_correction.bypass_mode": "False",
    }
    for key, expected in required.items():
        if resolved.get(key) != expected:
            raise MaxRLConfigError(f"strict MaxRL requires {key}={expected}")
    if max_optimizer_steps is not None and resolved.get("trainer.total_training_steps") != str(max_optimizer_steps):
        raise MaxRLConfigError(f"strict MaxRL requires trainer.total_training_steps={max_optimizer_steps}")
    if resolved.get("data.train_batch_size") != resolved.get("actor_rollout_ref.actor.ppo_mini_batch_size"):
        raise MaxRLConfigError("strict MaxRL requires one global mini-batch per optimizer step")


def build_overrides(
    config: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    extra_overrides: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Compile the public MaxRL TOML into the canonical Verl Hydra contract."""

    _validate_removed_fields(config)
    environment = dict(os.environ if env is None else env)
    experiment = _table(config, "experiment")
    model = _table(config, "model")
    train = _table(config, "data.train")
    algorithm = _table(config, "algorithm")
    reward = _table(config, "reward")
    optimizer = _table(config, "optimizer")
    train_generation = _table(config, "generation.train")
    execution = _table(config, "execution")
    rollout = _table(config, "execution.rollout")
    evaluation = _table(config, "evaluation")
    checkpoint = _table(config, "checkpoint")
    logging = _table(config, "logging")

    if _required(algorithm, "name", section_name="algorithm") != "maxrl":
        raise MaxRLConfigError("verl.trainer.maxrl requires algorithm.name='maxrl'")
    validate_g1i_model_asset(model)
    if "optimizer_steps" in experiment:
        raise MaxRLConfigError("MaxRL experiment.optimizer_steps was removed; use candidate_dataset_passes")
    passes = _nonnegative_int(
        _required(experiment, "candidate_dataset_passes", section_name="experiment"),
        name="experiment.candidate_dataset_passes",
    )
    max_optimizer_steps = experiment.get("max_optimizer_steps")
    if max_optimizer_steps is not None:
        max_optimizer_steps = _positive_int(
            max_optimizer_steps,
            name="experiment.max_optimizer_steps",
        )
    checkpoint_path = str(_required(model, "checkpoint", section_name="model"))
    legacy_checkpoint_path = str(_required(model, "legacy_checkpoint", section_name="model"))
    context_tokens = context_tokens_from_checkpoint(legacy_checkpoint_path)
    prompts_per_step = _positive_int(
        _required(algorithm, "prompts_per_step", section_name="algorithm"),
        name="algorithm.prompts_per_step",
    )
    responses_per_prompt = _positive_int(
        _required(algorithm, "responses_per_prompt", section_name="algorithm"),
        name="algorithm.responses_per_prompt",
    )
    nodes = _positive_int(_required(execution, "nodes", section_name="execution"), name="execution.nodes")
    gpus = _positive_int(
        _required(execution, "gpus_per_node", section_name="execution"),
        name="execution.gpus_per_node",
    )
    replicas = _positive_int(
        _required(rollout, "replicas", section_name="execution.rollout"),
        name="execution.rollout.replicas",
    )
    tensor_parallel = _positive_int(
        _required(
            rollout,
            "tensor_parallel_size_per_replica",
            section_name="execution.rollout",
        ),
        name="execution.rollout.tensor_parallel_size_per_replica",
    )
    if replicas * tensor_parallel != nodes * gpus:
        raise MaxRLConfigError("rollout replicas must consume every configured GPU")
    if _required(execution, "context_mode", section_name="execution") != "state_passing":
        raise MaxRLConfigError("strict MaxRL requires execution.context_mode='state_passing'")

    verl_root = Path(__file__).resolve().parents[2]
    train_prompt_key = str(train.get("prompt_field", "prompt")).strip()
    scorer = str(_required(reward, "scorer", section_name="reward"))
    kl_coefficient = algorithm.get("kl_coefficient", 0.0)
    train_temperature = _required(train_generation, "temperature", section_name="generation.train")
    gradient_norm_limit = _required(optimizer, "gradient_norm_limit", section_name="optimizer")
    wkv_mode = str(_required(execution, "wkv_mode", section_name="execution"))
    if wkv_mode not in {"fp16", "fp32io16"}:
        raise MaxRLConfigError("execution.wkv_mode must be fp16 or fp32io16")
    validation_before_training = bool(_required(evaluation, "before_training", section_name="evaluation"))
    validation_interval = _positive_int(
        _required(evaluation, "every_optimizer_steps", section_name="evaluation"),
        name="evaluation.every_optimizer_steps",
    )
    validation_command = _required(evaluation, "command", section_name="evaluation")
    if (
        not isinstance(validation_command, list)
        or not validation_command
        or any(not isinstance(item, str) or not item.strip() for item in validation_command)
    ):
        raise MaxRLConfigError("evaluation.command must be a non-empty array of command arguments")
    checkpoint_directory = str(_required(checkpoint, "directory", section_name="checkpoint"))

    overrides = [
        "algorithm.adv_estimator=maxrl",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files={_files(_required(train, 'files', section_name='data.train'), name='data.train.files')}",
        "data.val_files=null",
        f"+data.train_prompt_key={train_prompt_key}",
        f"data.train_batch_size={prompts_per_step}",
        f"data.seed={_required(experiment, 'seed', section_name='experiment')}",
        "data.max_prompt_length=null",
        "data.max_response_length=null",
        "data.filter_overlong_prompts=False",
        "data.truncation=error",
        f"actor_rollout_ref.rollout.prompt_length={context_tokens}",
        f"actor_rollout_ref.rollout.response_length={context_tokens}",
        f"reward.custom_reward_function.path={_reward_path(scorer, verl_root)}",
        "reward.custom_reward_function.name=compute_score",
        f"reward.reward_manager.name={_required(reward, 'manager', section_name='reward')}",
        "model@actor_rollout_ref.model=hf_model",
        f"actor_rollout_ref.model.path={checkpoint_path}",
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.model.enable_gradient_checkpointing=False",
        "actor@actor_rollout_ref.actor=dp_actor",
        "actor_rollout_ref.actor.use_torch_compile=False",
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]",
        f"actor_rollout_ref.actor.optim.lr={_required(optimizer, 'learning_rate', section_name='optimizer')}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={prompts_per_step}",
        "actor_rollout_ref.actor.ppo_epochs=1",
        f"actor_rollout_ref.actor.data_loader_seed={_required(experiment, 'seed', section_name='experiment')}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_dynamic_bsz=False",
        f"actor_rollout_ref.actor.use_kl_loss={_hydra(float(kl_coefficient) != 0.0)}",
        f"actor_rollout_ref.actor.kl_loss_coef={kl_coefficient}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps={optimizer.get('warmup_steps', 0)}",
        f"actor_rollout_ref.actor.optim.weight_decay={optimizer.get('weight_decay', 0.0)}",
        f"actor_rollout_ref.actor.entropy_coeff={algorithm.get('entropy_coefficient', 0.0)}",
        f"actor_rollout_ref.actor.optim.clip_grad={gradient_norm_limit}",
        f"actor_rollout_ref.actor.clip_ratio_low={_required(algorithm, 'ppo_clip', section_name='algorithm')}",
        f"actor_rollout_ref.actor.clip_ratio_high={_required(algorithm, 'ppo_clip', section_name='algorithm')}",
        f"actor_rollout_ref.actor.clip_ratio_c={algorithm.get('dual_clip', 3.0)}",
        "ref@actor_rollout_ref.ref=dp_ref",
        "actor_rollout_ref.ref.use_torch_compile=False",
        "actor_rollout_ref.ref.fsdp_config.use_torch_compile=False",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.load_format=auto",
        f"actor_rollout_ref.rollout.max_model_len={context_tokens}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={tensor_parallel}",
        f"actor_rollout_ref.rollout.n={responses_per_prompt}",
        f"actor_rollout_ref.rollout.seed={_required(experiment, 'seed', section_name='experiment')}",
        f"actor_rollout_ref.rollout.temperature={train_temperature}",
        f"actor_rollout_ref.rollout.top_k={_required(train_generation, 'top_k', section_name='generation.train')}",
        f"actor_rollout_ref.rollout.top_p={_required(train_generation, 'top_p', section_name='generation.train')}",
        "actor_rollout_ref.rollout.ignore_eos=False",
        "actor_rollout_ref.rollout.enable_prefix_caching=False",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.tokenizer_mode=rwkv",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=uni",
        '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V2_MODEL_RUNNER="1"',
        '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL="INFO"',
        '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_RWKV7_STRICT_STREAMING_WEIGHT_UPDATE="1"',
        "actor_rollout_ref.hybrid_engine=True",
        "trainer.v1.trainer_mode=sync",
        "actor_rollout_ref.rollout.checkpoint_engine.backend=naive",
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="
        f"{_required(rollout, 'weight_update_bucket_mib', section_name='execution.rollout')}",
        "algorithm.rollout_correction.rollout_is=token",
        "algorithm.rollout_correction.rollout_is_threshold=2.0",
        "algorithm.rollout_correction.rollout_is_batch_normalize=False",
        "algorithm.rollout_correction.rollout_rs=null",
        "algorithm.rollout_correction.bypass_mode=False",
        "data.dataloader_num_workers=0",
        "actor_rollout_ref.rollout.dtype=float16",
        "actor_rollout_ref.rollout.disable_log_stats=False",
    ]
    prompt_mode = model.get("prompt_mode")
    prompt_template = model.get("prompt_template")
    if prompt_mode:
        overrides.extend(
            [
                f"+data.apply_chat_template_kwargs.rwkv_generation_prompt={prompt_mode}",
            ]
        )
    if prompt_template:
        quoted_template = _quoted(prompt_template)
        overrides.extend(
            [
                f"+data.apply_chat_template_kwargs.rwkv_prompt_template={quoted_template}",
                f"actor_rollout_ref.rollout.rwkv_prompt_template={quoted_template}",
            ]
        )
    chunk_tokens = _positive_int(
        _required(execution, "state_chunk_tokens", section_name="execution"),
        name="execution.state_chunk_tokens",
    )
    if chunk_tokens % 16 or chunk_tokens >= context_tokens:
        raise MaxRLConfigError("state_chunk_tokens must be divisible by 16 and smaller than context")
    overrides.extend(
        [
            "actor_rollout_ref.rollout.max_num_seqs="
            f"{_required(rollout, 'max_concurrent_sequences_per_replica', section_name='execution.rollout')}",
            "actor_rollout_ref.rollout.max_num_batched_tokens="
            f"{_required(rollout, 'generation_token_budget_per_replica', section_name='execution.rollout')}",
            "actor_rollout_ref.rollout.data_parallel_size=1",
            "actor_rollout_ref.rollout.pipeline_model_parallel_size="
            f"{_required(rollout, 'pipeline_parallel_size_per_replica', section_name='execution.rollout')}",
            "critic.enable=False",
            f"trainer.logger={_hydra(_required(logging, 'backends', section_name='logging'))}",
            f"trainer.project_name={_required(experiment, 'project', section_name='experiment')}",
            f"trainer.experiment_name={_required(experiment, 'name', section_name='experiment')}",
            f"trainer.nnodes={nodes}",
            f"trainer.n_gpus_per_node={gpus}",
            f"trainer.default_local_dir={checkpoint_directory}",
            f"trainer.save_freq={_required(checkpoint, 'every_optimizer_steps', section_name='checkpoint')}",
            f"trainer.test_freq={validation_interval}",
            f"trainer.val_before_train={_hydra(validation_before_training)}",
            f"+trainer.external_evaluation.command={_hydra(validation_command)}",
            f"trainer.total_epochs={passes}",
        ]
    )
    if max_optimizer_steps is not None:
        overrides.append(f"trainer.total_training_steps={max_optimizer_steps}")
    if extra_overrides:
        _validate_extra_overrides(extra_overrides)
        overrides.extend(extra_overrides)
    _validate_resolved(
        overrides,
        checkpoint_path=checkpoint_path,
        context_tokens=context_tokens,
        prompts_per_step=prompts_per_step,
        responses_per_prompt=responses_per_prompt,
        candidate_dataset_passes=passes,
        max_optimizer_steps=max_optimizer_steps,
        validation_before_training=validation_before_training,
        validation_interval=validation_interval,
    )

    child_env = environment.copy()
    child_env["RWKV_MODEL_PATH"] = checkpoint_path
    child_env["VLLM_RWKV7_WKV_MODE"] = wkv_mode
    child_env["HELICOPTER_MODEL_REPOSITORY"] = G1I_MODEL_ASSET["repository"]
    child_env["HELICOPTER_MODEL_REVISION"] = G1I_MODEL_ASSET["revision"]
    child_env["HELICOPTER_MODEL_FILENAME"] = G1I_MODEL_ASSET["filename"]
    child_env["HELICOPTER_CHECKPOINT_SHA256"] = G1I_MODEL_ASSET["sha256"]
    return overrides, child_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m verl.trainer.maxrl")
    parser.add_argument("--config", type=Path, required=True, help="complete MaxRL TOML experiment")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="operational Hydra override from the documented allowlist",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and print the Verl command")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = read_config(args.config.resolve())
        overrides, child_env = build_overrides(config, extra_overrides=args.override)
    except MaxRLConfigError as exc:
        raise SystemExit(str(exc)) from exc
    command = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
    if args.dry_run:
        print(shlex.join(command))
        return 0
    os.execvpe(command[0], command, child_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
