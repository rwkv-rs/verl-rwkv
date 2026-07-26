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

import pytest

from verl.utils.request_budget import resolve_request_max_tokens


def test_response_budget_is_independent_for_each_tokenized_prompt():
    assert resolve_request_max_tokens(max_model_len=10240, prompt_length=1464) == 8776
    assert resolve_request_max_tokens(max_model_len=10240, prompt_length=848) == 9392


def test_explicit_request_cap_cannot_exceed_remaining_context():
    assert (
        resolve_request_max_tokens(
            max_model_len=10240,
            prompt_length=1464,
            requested_max_tokens=4096,
        )
        == 4096
    )
    assert (
        resolve_request_max_tokens(
            max_model_len=10240,
            prompt_length=9000,
            requested_max_tokens=4096,
        )
        == 1240
    )


def test_prompt_must_leave_at_least_one_response_token():
    with pytest.raises(ValueError, match="leaves no room"):
        resolve_request_max_tokens(max_model_len=10240, prompt_length=10240)
