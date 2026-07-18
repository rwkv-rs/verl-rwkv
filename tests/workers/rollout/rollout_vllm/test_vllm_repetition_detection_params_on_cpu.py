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

from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("ray")
pytest.importorskip("vllm")

from vllm.sampling_params import RequestOutputKind
from vllm.tokenizers.rwkv_defaults import (
    RWKV_DEFAULT_STOP_TOKEN_IDS,
    RWKV_DEFAULT_STOPS,
)

from verl.utils.ngram_repetition import (
    DEFAULT_REPETITION_MAX_COUNT,
    DEFAULT_REPETITION_NGRAM_SIZE,
    DEFAULT_REPETITION_RULES,
    vllm_repetition_detection_config,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    _apply_rwkv_default_stop_params,
    _rollout_output_kind,
)


@dataclass
class _HFConfig:
    model_type: str = "rwkv7"


@dataclass
class _ModelConfig:
    tokenizer_mode: str = "rwkv"
    hf_config: _HFConfig = field(default_factory=_HFConfig)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def test_default_repetition_detection_matches_rollout_truncation_rule():
    params = vllm_repetition_detection_config()

    assert params["max_pattern_size"] == DEFAULT_REPETITION_NGRAM_SIZE
    assert params["min_pattern_size"] == DEFAULT_REPETITION_NGRAM_SIZE
    assert params["min_count"] == DEFAULT_REPETITION_MAX_COUNT + 1
    assert params["mode"] == "occurrence"
    assert params["occurrence_rules"] == list(DEFAULT_REPETITION_RULES)


@pytest.mark.parametrize(
    ("needs_generation_logprobs", "needs_prompt_logprobs", "expected"),
    (
        (False, False, RequestOutputKind.DELTA),
        (True, False, RequestOutputKind.CUMULATIVE),
        (False, True, RequestOutputKind.CUMULATIVE),
    ),
)
def test_rollout_output_kind_uses_delta_only_when_logprobs_are_not_required(
    needs_generation_logprobs,
    needs_prompt_logprobs,
    expected,
):
    assert (
        _rollout_output_kind(
            needs_generation_logprobs=needs_generation_logprobs,
            needs_prompt_logprobs=needs_prompt_logprobs,
        )
        == expected
    )


def test_rollout_server_applies_rwkv_default_stop_params():
    sampling_params: dict[str, Any] = {}

    _apply_rwkv_default_stop_params(sampling_params, _ModelConfig())

    assert sampling_params["stop"] == list(RWKV_DEFAULT_STOPS)
    assert sampling_params["stop_token_ids"] == list(RWKV_DEFAULT_STOP_TOKEN_IDS)


def test_rollout_server_keeps_explicit_rwkv_stop_params():
    sampling_params: dict[str, Any] = {"stop": [], "stop_token_ids": [123]}

    _apply_rwkv_default_stop_params(sampling_params, _ModelConfig())

    assert sampling_params == {"stop": [], "stop_token_ids": [123]}


def test_rollout_server_does_not_apply_rwkv_stop_params_to_other_models():
    sampling_params: dict[str, Any] = {}

    _apply_rwkv_default_stop_params(
        sampling_params,
        _ModelConfig(tokenizer_mode="auto", hf_config=_HFConfig(model_type="llama")),
    )

    assert sampling_params == {}
