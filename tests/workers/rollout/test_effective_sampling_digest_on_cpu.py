from vllm import SamplingParams

from verl.workers.rollout.vllm_rollout.vllm_async_server import effective_sampling_digest


def test_effective_sampling_digest_captures_context_clamp_and_normalized_defaults():
    clamped_short = SamplingParams(max_tokens=4, repetition_penalty=1.0, ignore_eos=False)
    clamped_long = SamplingParams(max_tokens=8, repetition_penalty=1.0, ignore_eos=False)
    changed_default = SamplingParams(max_tokens=4, repetition_penalty=1.1, ignore_eos=False)

    short_digest = effective_sampling_digest(clamped_short)

    assert short_digest != effective_sampling_digest(clamped_long)
    assert short_digest != effective_sampling_digest(changed_default)
