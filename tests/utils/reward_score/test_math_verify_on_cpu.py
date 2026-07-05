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
