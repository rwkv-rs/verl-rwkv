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

import torch

from verl.experimental.agent_loop.agent_loop import (
    _prompt_with_single_prefix_token,
    _right_pad_prompt_batch,
)


def test_prompt_with_single_prefix_token_prepends_one_attended_eos():
    prompt = _prompt_with_single_prefix_token([11, 12, 13], prefix_token_id=0)

    assert prompt == [0, 11, 12, 13]


def test_prompt_with_single_prefix_token_does_not_duplicate_existing_prefix():
    prompt = _prompt_with_single_prefix_token([0, 11, 12], prefix_token_id=0)

    assert prompt == [0, 11, 12]


def test_prompt_with_single_prefix_token_uses_prompt_budget_for_prefix():
    prompt = _prompt_with_single_prefix_token([11, 12, 13, 14], prefix_token_id=0, max_length=4)

    assert prompt == [0, 12, 13, 14]


def test_right_pad_prompt_batch_keeps_only_real_prefix_zero_attended():
    prompt_ids, attention_mask = _right_pad_prompt_batch(
        [
            torch.tensor([[0, 11, 12, 13]], dtype=torch.long),
            torch.tensor([[0, 21]], dtype=torch.long),
        ],
        pad_token_id=0,
    )

    assert prompt_ids.tolist() == [[0, 11, 12, 13], [0, 21, 0, 0]]
    assert attention_mask.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert prompt_ids[0][attention_mask[0].bool()].tolist().count(0) == 1
    assert prompt_ids[1][attention_mask[1].bool()].tolist().count(0) == 1


def test_right_pad_prompt_batch_supports_fixed_width_for_worker_concat():
    prompt_ids, attention_mask = _right_pad_prompt_batch(
        [torch.tensor([[0, 11]], dtype=torch.long)],
        pad_token_id=0,
        max_length=5,
    )

    assert prompt_ids.tolist() == [[0, 11, 0, 0, 0]]
    assert attention_mask.tolist() == [[1, 1, 0, 0, 0]]
    assert prompt_ids[0][attention_mask[0].bool()].tolist().count(0) == 1
