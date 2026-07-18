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

import sys
import types


def test_math_verify_score_uses_non_strict_verification(monkeypatch):
    from verl.utils.reward_score.math_verify import _verify_in_subprocess

    calls = []

    grader = types.ModuleType("math_verify.grader")

    def verify(gold, pred, *, strict):
        calls.append((gold, pred, strict))
        return True

    grader.verify = verify

    parser = types.ModuleType("math_verify.parser")

    class ExprExtractionConfig:
        pass

    class LatexExtractionConfig:
        pass

    def parse(text, targets=None):
        return ["gold"] if "boxed" in text else ["pred"]

    parser.ExprExtractionConfig = ExprExtractionConfig
    parser.LatexExtractionConfig = LatexExtractionConfig
    parser.parse = parse

    monkeypatch.setitem(sys.modules, "math_verify.grader", grader)
    monkeypatch.setitem(sys.modules, "math_verify.parser", parser)

    assert _verify_in_subprocess("\\boxed{1}", "1") == 1.0
    assert calls == [("gold", "pred", False)]
