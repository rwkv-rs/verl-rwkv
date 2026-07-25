# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import warnings
from enum import Enum

from omegaconf import DictConfig, OmegaConf, open_dict

from verl.single_controller.base import Worker
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo.core_algos import AdvantageEstimator

WorkerType = type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    Env = 7
    TeacherModel = 8

    def __str__(self):
        return self._get_role_string()

    def _get_role_string(self):
        role_mapping = {
            Role.Actor: "actor",
            Role.Rollout: "rollout",
            Role.ActorRollout: "actor_rollout",
            Role.Critic: "critic",
            Role.RefPolicy: "ref",
            Role.RewardModel: "rm",
            Role.ActorRolloutRef: "actor_rollout_ref",
            Role.TeacherModel: "teacher",
        }
        return role_mapping.get(self, self.name.lower())

    @classmethod
    def from_string(cls, name: str):
        string_mapping = {
            "actor": cls.Actor,
            "rollout": cls.Rollout,
            "actor_rollout": cls.ActorRollout,
            "critic": cls.Critic,
            "ref": cls.RefPolicy,
            "rm": cls.RewardModel,
            "actor_rollout_ref": cls.ActorRolloutRef,
        }
        role = string_mapping.get(name.lower())
        if role is None:
            raise ValueError(f"No Role found for string: {name}")
        return role


def need_reference_policy(
    config: DictConfig,
) -> bool:
    """Given the config, do we need ref policy."""
    return config.algorithm.get("use_kl_in_reward", False) or config.actor_rollout_ref.actor.use_kl_loss


def need_teacher_policy(
    config: DictConfig,
) -> bool:
    """Given the config, do we need distillation policy."""
    return is_distillation_enabled(config.get("distillation"))


def need_reward_model(
    config: DictConfig,
) -> bool:
    """Given the config, do we need reward model."""
    return config.reward.reward_model.enable


def need_critic(config: DictConfig) -> bool:
    """Given a config, do we need critic."""
    if config.critic.enable is not None:
        return bool(config.critic.enable)
    elif config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return True
    else:
        warnings.warn(
            "Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True",
            stacklevel=2,
        )
        return False


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """

    from verl.utils.dataset.rl_dataset import get_dataset_class

    # Get the dataset class
    dataset_cls = get_dataset_class(data_config)
    prompt_key_config = "train_prompt_key" if is_train else "val_prompt_key"
    prompt_key = data_config.get(
        prompt_key_config,
        data_config.get("prompt_key", "prompt"),
    )
    dataset_config = OmegaConf.create(OmegaConf.to_container(data_config, resolve=False))
    with open_dict(dataset_config):
        dataset_config.prompt_key = prompt_key
    if not is_train and data_config.get("val_apply_chat_template_kwargs", None):
        train_kwargs = dict(dataset_config.get("apply_chat_template_kwargs", {}) or {})
        train_kwargs.update(dict(dataset_config.get("val_apply_chat_template_kwargs", {}) or {}))
        with open_dict(dataset_config):
            dataset_config.apply_chat_template_kwargs = train_kwargs

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=dataset_config,
        max_samples=max_samples,
    )

    return dataset


def resolve_automatic_sequence_lengths(config: DictConfig, *datasets) -> tuple[int, int] | None:
    """Resolve prompt/response lengths from the selected, templated datasets.

    The CLI supplies only the model context derived from the checkpoint name.
    Dataset construction measures every selected prompt after applying its train
    or validation chat template. This function publishes the shared maximum
    before any actor or rollout worker starts.
    """

    if not config.data.get("derive_sequence_lengths", False):
        return None

    context_length = int(config.data.model_context_length)
    measured_lengths = [
        int(dataset.max_templated_prompt_length)
        for dataset in datasets
        if dataset is not None and hasattr(dataset, "max_templated_prompt_length")
    ]
    if len(measured_lengths) != len([dataset for dataset in datasets if dataset is not None]):
        raise RuntimeError(
            "automatic sequence-length derivation requires every dataset to expose max_templated_prompt_length"
        )
    if not measured_lengths:
        raise RuntimeError("automatic sequence-length derivation requires at least one dataset")

    max_prompt_length = max(measured_lengths)
    max_response_length = context_length - max_prompt_length
    if max_response_length <= 0:
        raise RuntimeError(
            "the longest templated dataset prompt does not leave room for a response: "
            f"context={context_length}, max_prompt={max_prompt_length}"
        )

    with open_dict(config):
        config.data.max_prompt_length = max_prompt_length
        config.data.max_response_length = max_response_length
        config.actor_rollout_ref.rollout.prompt_length = max_prompt_length
        config.actor_rollout_ref.rollout.response_length = max_response_length

    print(
        "resolved automatic sequence lengths: "
        f"context={context_length}, max_prompt={max_prompt_length}, "
        f"max_response={max_response_length}"
    )
    return max_prompt_length, max_response_length


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import SequentialSampler

    # torch.utils.data.RandomSampler could not recover properly
    from torchdata.stateful_dataloader.sampler import RandomSampler

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler
