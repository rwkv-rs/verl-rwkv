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
