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
