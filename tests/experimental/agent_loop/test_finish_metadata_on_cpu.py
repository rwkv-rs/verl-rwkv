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

from verl.experimental.agent_loop.finish_metadata import (
    STRICT_COT_FORMAT_VERSION,
    build_rollout_finish_metadata,
    classify_rollout_finish,
    parse_strict_cot,
)


@pytest.mark.parametrize(
    ("text", "valid"),
    [
        ("<think>reasoning</think><answer>42</answer>", True),
        ("<think>line 1\nline 2</think><answer>\n42\n</answer>", True),
        ("prefix<think>x</think><answer>y</answer>", False),
        ("<think>x</think>\n<answer>y</answer>", False),
        ("<think></think><answer>y</answer>", False),
        ("<think>x</think><answer></answer>", False),
        ("<think>x</think><answer>y</answer>suffix", False),
        ("<think>x</think><answer>y</answer><answer>z</answer>", False),
    ],
)
def test_parse_strict_cot_decision_table(text: str, valid: bool):
    parsed = parse_strict_cot(text)

    assert parsed["format_valid"] is valid
    assert parsed["format_parser_version"] == STRICT_COT_FORMAT_VERSION


@pytest.mark.parametrize(
    ("finish_reason", "backend_stop_reason", "repetition_truncated", "expected_category"),
    [
        ("stop", None, False, "ended_by_eos"),
        ("stop", 0, False, "ended_by_eos"),
        ("stop", "eos", False, "ended_by_eos"),
        ("stop", 11, False, "other_failure"),
        ("stop", "\nUser:", False, "other_failure"),
        ("completed", None, False, "other_failure"),
        ("length", None, False, "context_exhausted"),
        ("max_length_truncated", None, False, "context_exhausted"),
        ("abort", None, False, "other_failure"),
        (None, None, False, "other_failure"),
        ("length", None, True, "repetition_truncated"),
        ("stop", None, True, "repetition_truncated"),
    ],
)
def test_classify_rollout_finish_is_exhaustive_and_disjoint(
    finish_reason: str | None,
    backend_stop_reason: int | str | None,
    repetition_truncated: bool,
    expected_category: str,
):
    categories = classify_rollout_finish(
        finish_reason=finish_reason,
        backend_stop_reason=backend_stop_reason,
        repetition_truncated=repetition_truncated,
    )

    assert categories[expected_category] is True
    assert sum(categories.values()) == 1


def test_build_rollout_finish_metadata_keeps_format_and_finish_independent():
    metadata = build_rollout_finish_metadata(
        "<think>reasoning</think><answer>42</answer>",
        finish_reason="length",
        backend_stop_reason=None,
        repetition_truncated=False,
    )

    assert metadata == {
        "format_valid": True,
        "format_parser_version": STRICT_COT_FORMAT_VERSION,
        "ended_by_eos": False,
        "repetition_truncated": False,
        "context_exhausted": True,
        "other_failure": False,
    }
