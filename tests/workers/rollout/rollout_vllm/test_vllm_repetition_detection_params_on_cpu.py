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

from vllm.sampling_params import RepetitionDetectionParams, RequestOutputKind
from vllm.tokenizers.rwkv_defaults import (
    RWKV_BOS_EOS_TOKEN_ID,
    RWKV_PROMPT_TEMPLATE_ASSISTANT,
)
from vllm.v1.core.sched.utils import check_sequence_repetition

from verl.utils.ngram_repetition import (
    DEFAULT_CONSECUTIVE_MAX_PATTERN_SIZE,
    DEFAULT_CONSECUTIVE_MIN_COUNT,
    DEFAULT_CONSECUTIVE_MIN_PATTERN_SIZE,
    ConsecutiveRepetitionDetector,
    vllm_repetition_detection_config,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    _apply_rwkv_prompt_template_stops,
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

    assert params["max_pattern_size"] == DEFAULT_CONSECUTIVE_MAX_PATTERN_SIZE
    assert params["min_pattern_size"] == DEFAULT_CONSECUTIVE_MIN_PATTERN_SIZE
    assert params["min_count"] == DEFAULT_CONSECUTIVE_MIN_COUNT
    assert params["mode"] == "consecutive"
    assert "occurrence_rules" not in params


@pytest.mark.parametrize(
    ("token_ids", "detected"),
    [
        pytest.param(
            [*range(40), *range(40), *range(40)],
            True,
            id="three-consecutive-blocks",
        ),
        pytest.param(
            [token_id for step in range(32) for token_id in (101, 102, 103, 104, 105, 106, 1000 + step)],
            False,
            id="nonadjacent-math-expression",
        ),
    ],
)
def test_verl_and_vllm_repetition_boundaries_match(token_ids, detected):
    verl_detector = ConsecutiveRepetitionDetector()
    vllm_params = RepetitionDetectionParams(**vllm_repetition_detection_config())

    assert (verl_detector.observe(token_ids) is not None) is detected
    assert check_sequence_repetition(token_ids, vllm_params) is detected


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


def test_rollout_server_merges_rwkv_template_and_user_stops():
    sampling_params: dict[str, Any] = {
        "stop": ["END"],
        "stop_token_ids": [123],
        "ignore_eos": True,
    }

    _apply_rwkv_prompt_template_stops(
        sampling_params,
        _ModelConfig(),
        prompt_template=RWKV_PROMPT_TEMPLATE_ASSISTANT,
    )

    assert sampling_params == {
        "stop": ["\nUser:", "END"],
        "stop_token_ids": [RWKV_BOS_EOS_TOKEN_ID, 123],
        "ignore_eos": False,
    }


def test_rollout_server_does_not_apply_rwkv_stop_params_to_other_models():
    sampling_params: dict[str, Any] = {}

    _apply_rwkv_prompt_template_stops(
        sampling_params,
        _ModelConfig(tokenizer_mode="auto", hf_config=_HFConfig(model_type="llama")),
    )

    assert sampling_params == {}
