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

from vllm import SamplingParams

from verl.workers.rollout.vllm_rollout.vllm_async_server import effective_sampling_digest


def test_effective_sampling_digest_excludes_request_capacity_but_captures_policy_defaults():
    clamped_short = SamplingParams(max_tokens=4, repetition_penalty=1.0, ignore_eos=False)
    clamped_long = SamplingParams(max_tokens=8, repetition_penalty=1.0, ignore_eos=False)
    changed_default = SamplingParams(max_tokens=4, repetition_penalty=1.1, ignore_eos=False)

    short_digest = effective_sampling_digest(clamped_short)

    assert short_digest == effective_sampling_digest(clamped_long)
    assert short_digest != effective_sampling_digest(changed_default)
