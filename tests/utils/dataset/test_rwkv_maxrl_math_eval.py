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

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "examples/data_preprocess/rwkv_maxrl_math_eval.py"
SPEC = importlib.util.spec_from_file_location("rwkv_maxrl_math_eval", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


def test_convert_record_builds_verl_rule_reward_schema() -> None:
    converted = prepare.convert_record(
        {"problem": "What is 20 + 5?", "answer": 25},
        7,
        prepare.SOURCES["aime25"],
    )

    assert converted["data_source"] == "aime25"
    assert converted["prompt"] == [
        {
            "role": "user",
            "content": f"What is 20 + 5? {prepare.INSTRUCTION}",
        }
    ]
    assert converted["reward_model"] == {"style": "rule", "ground_truth": "25"}
    assert converted["extra_info"]["index"] == 7


def test_convert_record_removes_existing_box_from_ground_truth() -> None:
    converted = prepare.convert_record(
        {"problem": "AIME problem", "solution": r"\boxed{204}"},
        0,
        prepare.SOURCES["aime24"],
    )

    assert converted["reward_model"]["ground_truth"] == "204"


def test_convert_record_rejects_empty_fields() -> None:
    source = prepare.SOURCES["aime25"]
    with pytest.raises(RuntimeError, match="empty question or answer"):
        prepare.convert_record({"problem": "", "answer": "1"}, 0, source)
