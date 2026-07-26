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

from typing import Any, Optional

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import DictConfigWrap, build_agent_loop_sampling_params
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.replica import TokenOutput


class _FakeServerManager:
    def __init__(self, token_ids: list[int]):
        self.token_ids = token_ids

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        del request_id, prompt_ids, sampling_params, image_data, video_data, audio_data, mm_processor_kwargs, kwargs
        return TokenOutput(
            token_ids=self.token_ids,
            log_probs=[-0.5] * len(self.token_ids),
            extra_fields={"source": "fake_server"},
        )


class _FakeTokenizer:
    def __init__(self):
        self.applied_kwargs: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict]] = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> list[int]:
        del messages, tools, add_generation_prompt, tokenize, return_dict
        self.applied_kwargs.append(dict(kwargs))
        return [101, 102]


def _make_loop(
    token_ids: list[int],
    *,
    tokenizer: _FakeTokenizer | None = None,
    data_overrides: dict[str, Any] | None = None,
) -> SingleTurnAgentLoop:
    data_config = {
        "tool_config_path": None,
        "apply_chat_template_kwargs": {},
        "continuous_token": {"enable": False, "model_family": "auto"},
    }
    if data_overrides:
        data_config.update(data_overrides)
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"prompt_length": 16, "response_length": 128, "multi_turn": {"tool_config_path": None}},
                "model": {},
            },
            "data": data_config,
        }
    )

    return SingleTurnAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=_FakeServerManager(token_ids),
        tokenizer=tokenizer or _FakeTokenizer(),
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
    )


@pytest.mark.asyncio
async def test_single_turn_truncates_when_16gram_repeats_three_times():
    ngram = list(range(16))
    generated = ngram * 6 + [999]
    loop = _make_loop(generated)

    output = await loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "repeat"}])

    assert output.response_ids == generated[:48]
    assert output.response_mask == [1] * 48
    assert output.response_logprobs == [-0.5] * 48
    assert output.extra_fields["source"] == "fake_server"
    assert output.extra_fields["repetition_truncated"] is True
    assert output.extra_fields["repetition_ngram_size"] == 64
    assert output.extra_fields["repetition_ngram_count_threshold"] == 3
    assert output.extra_fields["repetition_truncation_length"] == 48
    assert output.extra_fields["repetition_matched_rule"] == {
        "ngram_size": 16,
        "min_count": 3,
    }
    assert output.extra_fields["original_response_length"] == len(generated)


@pytest.mark.asyncio
async def test_single_turn_truncates_severe_short_ngram_repetition():
    ngram = [11, 12, 13, 14, 15, 16, 17, 18]
    generated = ngram * 10 + [999]
    loop = _make_loop(generated)

    output = await loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "repeat"}])

    assert output.response_ids == generated[:24]
    assert output.response_mask == [1] * 24
    assert output.extra_fields["repetition_truncated"] is True
    assert output.extra_fields["repetition_truncation_length"] == 24
    assert output.extra_fields["repetition_detection_rules"] == [
        {
            "mode": "consecutive",
            "min_pattern_size": 4,
            "max_pattern_size": 64,
            "min_count": 3,
        }
    ]


@pytest.mark.asyncio
async def test_single_turn_keeps_response_below_default_repetition_thresholds():
    ngram = list(range(16))
    generated = ngram * 2 + [999]
    loop = _make_loop(generated)

    output = await loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "repeat"}])

    assert output.response_ids == generated
    assert output.response_mask == [1] * len(generated)
    assert output.response_logprobs == [-0.5] * len(generated)
    assert output.extra_fields["repetition_truncated"] is False
    assert output.extra_fields["original_response_length"] == len(generated)


@pytest.mark.asyncio
async def test_single_turn_keeps_nonadjacent_repeated_math_expressions():
    expression = [101, 102, 103, 104]
    generated = [token_id for step in range(20) for token_id in (*expression, 1000 + step)]
    loop = _make_loop(generated)

    output = await loop.run(
        sampling_params={},
        raw_prompt=[{"role": "user", "content": "enumerate k"}],
    )

    assert output.response_ids == generated
    assert output.extra_fields["repetition_truncated"] is False


@pytest.mark.asyncio
async def test_single_turn_skips_repetition_truncation_when_detection_disabled():
    ngram = list(range(16))
    generated = ngram * 6 + [999]
    loop = _make_loop(generated)

    output = await loop.run(
        sampling_params={"repetition_detection": None},
        raw_prompt=[{"role": "user", "content": "repeat"}],
    )

    assert output.response_ids == generated
    assert output.response_mask == [1] * len(generated)
    assert output.response_logprobs == [-0.5] * len(generated)
    assert output.extra_fields["repetition_truncated"] is False
    assert output.extra_fields["original_response_length"] == len(generated)


@pytest.mark.asyncio
async def test_single_turn_uses_validation_chat_template_kwargs_for_validation_prompt():
    tokenizer = _FakeTokenizer()
    loop = _make_loop(
        [999],
        tokenizer=tokenizer,
        data_overrides={
            "apply_chat_template_kwargs": {
                "rwkv_generation_prompt": "open_think",
                "shared_key": "shared_value",
            },
            "val_apply_chat_template_kwargs": {
                "rwkv_generation_prompt": "fake_think",
            },
        },
    )

    await loop.run(
        sampling_params={"repetition_detection": None},
        raw_prompt=[{"role": "user", "content": "validate"}],
        __validate__=True,
    )
    assert tokenizer.applied_kwargs[-1] == {
        "rwkv_generation_prompt": "fake_think",
        "shared_key": "shared_value",
    }

    await loop.run(
        sampling_params={"repetition_detection": None},
        raw_prompt=[{"role": "user", "content": "train"}],
    )
    assert tokenizer.applied_kwargs[-1] == {
        "rwkv_generation_prompt": "open_think",
        "shared_key": "shared_value",
    }


def test_agent_loop_validation_sampling_params_include_penalties():
    config = OmegaConf.create(
        {
            "temperature": 1.0,
            "top_p": 0.8,
            "top_k": 32,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "penalty_decay": 0.996,
            "calculate_log_probs": True,
            "val_kwargs": {
                "temperature": 0.25,
                "top_p": 0.35,
                "top_k": 40,
                "presence_penalty": 0.65,
                "frequency_penalty": 0.1,
                "repetition_penalty": 0.25,
                "penalty_decay": 0.99,
            },
        }
    )

    sampling_params = build_agent_loop_sampling_params(config, validate=True)

    assert sampling_params["temperature"] == 0.25
    assert sampling_params["top_p"] == 0.35
    assert sampling_params["top_k"] == 40
    assert sampling_params["presence_penalty"] == 0.65
    assert sampling_params["frequency_penalty"] == 0.1
    assert sampling_params["repetition_penalty"] == 0.25
    assert sampling_params["penalty_decay"] == 0.99
    assert sampling_params["logprobs"] is None
    assert sampling_params["repetition_detection"] is None


def test_agent_loop_training_sampling_params_keep_rollout_logprobs():
    config = OmegaConf.create(
        {
            "temperature": 1.0,
            "top_p": 0.8,
            "top_k": 32,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "penalty_decay": 0.996,
            "calculate_log_probs": True,
            "val_kwargs": {},
        }
    )

    sampling_params = build_agent_loop_sampling_params(config, validate=False)

    assert sampling_params["logprobs"] is True
    assert sampling_params["frequency_penalty"] == 0.0
    assert "repetition_detection" not in sampling_params
