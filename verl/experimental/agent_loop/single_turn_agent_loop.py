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
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.trainer.ppo.v1.policy_identity import IDENTITY_TAG_KEYS, canonical_digest
from verl.utils.ngram_repetition import ConsecutiveRepetitionDetector, repetition_extra_fields
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        # priority may arrive as np.int64 from non_tensor_batch; normalize to Python int.
        priority = int(priority)
        validate = bool(kwargs.pop("__validate__", False))
        messages = list(kwargs["raw_prompt"])
        identity_values = {key: kwargs.get(key) for key in IDENTITY_TAG_KEYS}
        present_identity_keys = {key for key, value in identity_values.items() if value is not None}
        if present_identity_keys and present_identity_keys != set(IDENTITY_TAG_KEYS):
            missing = sorted(set(IDENTITY_TAG_KEYS) - present_identity_keys)
            raise RuntimeError(f"rollout request has partial behavior-policy identity: missing {missing}")
        expected_policy_identity = identity_values if present_identity_keys else None
        expected_sampling_digest = canonical_digest(sampling_params)

        # 1. extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # 2. apply chat template and tokenize
        use_continuous_token = self.enable_continuous_token and not multi_modal_data
        if use_continuous_token:
            prompt_ids = await self.ct_build_initial_tokens(messages)
        else:
            prompt_ids = await self.apply_chat_template(
                messages,
                images=images,
                videos=videos,
                audios=audios,
                mm_processor_kwargs=mm_processor_kwargs,
                validate=validate,
            )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            request_id = f"det-{priority}" if getattr(self.rollout_config, "full_determinism", False) else uuid4().hex
            token_output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                audio_data=audios,
                video_data=videos,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
                expected_policy_identity=expected_policy_identity,
                expected_sampling_digest=expected_sampling_digest,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = token_output.num_preempted if token_output.num_preempted is not None else -1

        if use_continuous_token:
            merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                prompt_ids,
                token_output.token_ids,
                [],
                [] if token_output.log_probs else None,
                assistant_logprobs=token_output.log_probs if token_output.log_probs else None,
            )
            response_ids = merge_result.token_ids[-len(response_mask) :] if response_mask else []
            prompt_ids = merge_result.token_ids[: len(merge_result.token_ids) - len(response_mask)]
        else:
            response_ids = token_output.token_ids
            response_mask = [1] * len(token_output.token_ids)
            response_logprobs = token_output.log_probs

        extra_fields = dict(token_output.extra_fields)
        original_response_length = int(extra_fields.get("original_response_length") or len(response_ids))
        repetition_detection_enabled = sampling_params.get("repetition_detection", True) is not None
        truncation_length = None
        repetition_detector = None
        if repetition_detection_enabled:
            repetition_detector = ConsecutiveRepetitionDetector()
            truncation_length = repetition_detector.observe(response_ids)
        if truncation_length is not None:
            assert repetition_detector is not None
            response_ids = response_ids[:truncation_length]
            response_mask = response_mask[:truncation_length]
            response_logprobs = response_logprobs[:truncation_length] if response_logprobs else None
            extra_fields.update(
                repetition_extra_fields(
                    truncated=True,
                    truncation_length=truncation_length,
                    original_response_length=original_response_length,
                    matched_rule=repetition_detector.matched_rule,
                    matched_reason=repetition_detector.matched_reason,
                    matched_text_stats=repetition_detector.matched_text_stats,
                )
            )
        elif "repetition_truncated" not in extra_fields:
            extra_fields.update(
                repetition_extra_fields(
                    truncated=False,
                    truncation_length=None,
                    original_response_length=original_response_length,
                )
            )

        routed_experts = token_output.routed_experts
        if truncation_length is not None and routed_experts is not None:
            routed_experts = routed_experts[: len(prompt_ids) + len(response_ids)]

        extra_fields["stop_reason"] = token_output.stop_reason
        extra_fields["response_token_count"] = min(len(response_ids), self.response_length)

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=(
                routed_experts[: len(prompt_ids) + self.response_length] if routed_experts is not None else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output
