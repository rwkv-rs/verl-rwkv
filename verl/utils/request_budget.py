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


def resolve_request_max_tokens(
    *,
    max_model_len: int,
    prompt_length: int,
    requested_max_tokens: int | None = None,
) -> int:
    """Return this request's response budget from its tokenized prompt length."""

    remaining_context = max_model_len - prompt_length
    if remaining_context < 1:
        raise ValueError(
            f"Prompt length ({prompt_length}) leaves no room to generate within the "
            f"model's maximum context length ({max_model_len}); need at least 1 token of headroom."
        )
    if requested_max_tokens is None:
        return remaining_context
    return max(1, min(int(requested_max_tokens), remaining_context))
