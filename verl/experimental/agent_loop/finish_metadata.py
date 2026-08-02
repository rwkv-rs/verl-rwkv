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

import re
from typing import Any

STRICT_COT_FORMAT_VERSION = "strict-cot-v1"
ROLLOUT_FINISH_METADATA_KEYS = (
    "format_valid",
    "format_parser_version",
    "ended_by_eos",
    "repetition_truncated",
    "context_exhausted",
    "other_failure",
)

_BLOCK_CONTENT = r"(?:(?!</?(?:think|answer)>)[\s\S])+"
_STRICT_COT_PATTERN = re.compile(
    rf"\A<think>(?P<think>{_BLOCK_CONTENT})</think><answer>(?P<answer>{_BLOCK_CONTENT})</answer>\Z"
)
_EOS_BACKEND_REASONS = {None, 0, "eos", "eos_token", "eos_token_id"}
_LENGTH_FINISH_REASONS = {"length", "max_length", "max_tokens", "max_length_truncated"}


def parse_strict_cot(text: str) -> dict[str, Any]:
    """Parse exactly one non-empty think block followed by one non-empty answer block."""
    match = _STRICT_COT_PATTERN.fullmatch(text)
    if match is None:
        return {
            "format_valid": False,
            "format_parser_version": STRICT_COT_FORMAT_VERSION,
            "think": None,
            "answer": None,
        }
    return {
        "format_valid": True,
        "format_parser_version": STRICT_COT_FORMAT_VERSION,
        "think": match.group("think"),
        "answer": match.group("answer"),
    }


def classify_rollout_finish(
    *,
    finish_reason: str | None,
    backend_stop_reason: Any,
    repetition_truncated: bool,
) -> dict[str, bool]:
    """Return one auditable terminal category for a completed rollout attempt."""
    normalized_finish_reason = finish_reason.lower() if isinstance(finish_reason, str) else finish_reason
    normalized_backend_reason = (
        backend_stop_reason.lower() if isinstance(backend_stop_reason, str) else backend_stop_reason
    )

    ended_by_eos = (
        not repetition_truncated
        and normalized_finish_reason in {"stop", "eos", "eos_token"}
        and normalized_backend_reason in _EOS_BACKEND_REASONS
    )
    context_exhausted = not repetition_truncated and normalized_finish_reason in _LENGTH_FINISH_REASONS
    other_failure = not (repetition_truncated or ended_by_eos or context_exhausted)
    return {
        "ended_by_eos": ended_by_eos,
        "repetition_truncated": repetition_truncated,
        "context_exhausted": context_exhausted,
        "other_failure": other_failure,
    }


def build_rollout_finish_metadata(
    text: str,
    *,
    finish_reason: str | None,
    backend_stop_reason: Any,
    repetition_truncated: bool,
) -> dict[str, Any]:
    """Build the stable format and terminal metadata stored with a trajectory."""
    parsed = parse_strict_cot(text)
    return {
        "format_valid": parsed["format_valid"],
        "format_parser_version": parsed["format_parser_version"],
        **classify_rollout_finish(
            finish_reason=finish_reason,
            backend_stop_reason=backend_stop_reason,
            repetition_truncated=repetition_truncated,
        ),
    }
