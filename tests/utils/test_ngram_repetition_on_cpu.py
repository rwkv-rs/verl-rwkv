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

from types import SimpleNamespace

import pytest

from verl.utils.ngram_repetition import (
    DEFAULT_CONSECUTIVE_MAX_PATTERN_SIZE,
    DEFAULT_CONSECUTIVE_MIN_COUNT,
    DEFAULT_CONSECUTIVE_MIN_PATTERN_SIZE,
    DEFAULT_REPETITION_MAX_COUNT,
    DEFAULT_REPETITION_NGRAM_SIZE,
    DEFAULT_REPETITION_RULES,
    ConsecutiveRepetitionDetector,
    NGramRepetitionDetector,
    consume_token_stream,
    consume_until_repetition,
)


def test_consecutive_detector_ignores_nonadjacent_math_expressions():
    expression = [101, 102, 103, 104, 105, 106]
    generated: list[int] = []
    for step in range(32):
        generated.extend([*expression, 1000 + step])

    detector = ConsecutiveRepetitionDetector()

    assert detector.observe(generated) is None
    assert detector.matched_reason is None


def test_consecutive_detector_stops_at_third_complete_tail_block():
    prefix = [900, 901, 902]
    repeated_block = list(range(40))
    generated = [*prefix, *repeated_block, *repeated_block, *repeated_block, 999]
    detector = ConsecutiveRepetitionDetector()

    truncation_length = detector.observe(generated)

    assert truncation_length == len(prefix) + len(repeated_block) * 3
    assert detector.matched_rule == (len(repeated_block), DEFAULT_CONSECUTIVE_MIN_COUNT)
    assert detector.matched_reason == "consecutive_ngram"


def test_consecutive_detector_honors_default_pattern_boundaries():
    assert DEFAULT_CONSECUTIVE_MIN_PATTERN_SIZE == 4
    assert DEFAULT_CONSECUTIVE_MAX_PATTERN_SIZE == 64
    assert ConsecutiveRepetitionDetector().observe([1, 2, 3] * 3) is None
    assert ConsecutiveRepetitionDetector().observe([1, 2, 3, 4] * 3) == 12
    assert ConsecutiveRepetitionDetector().observe(list(range(64)) * 3) == 192
    assert ConsecutiveRepetitionDetector().observe(list(range(65)) * 3) is None


def test_ngram_repetition_detector_fires_on_the_sixth_16gram():
    ngram = list(range(DEFAULT_REPETITION_NGRAM_SIZE))
    detector = NGramRepetitionDetector(rules=[(DEFAULT_REPETITION_NGRAM_SIZE, DEFAULT_REPETITION_MAX_COUNT + 1)])

    assert detector.observe(ngram * DEFAULT_REPETITION_MAX_COUNT) is None
    truncation_length = detector.observe(ngram * (DEFAULT_REPETITION_MAX_COUNT + 1) + [999])

    assert truncation_length == DEFAULT_REPETITION_NGRAM_SIZE * (DEFAULT_REPETITION_MAX_COUNT + 1)


def test_ngram_repetition_detector_uses_default_multi_rules_to_stop_earlier():
    detector = NGramRepetitionDetector()
    ngram = [101, 102, 103, 104, 105, 106, 107, 108]

    assert detector.observe(ngram * 5) is None
    truncation_length = detector.observe(ngram * 9)

    assert truncation_length == 44
    assert detector.matched_rule == (12, dict(DEFAULT_REPETITION_RULES)[12])


def test_ngram_repetition_detector_default_rules_fire_on_the_20th_4gram():
    detector = NGramRepetitionDetector()

    def separated_4grams(count: int) -> list[int]:
        token_ids: list[int] = []
        for i in range(count):
            token_ids.extend([101, 102, 103, 104, 1000 + i])
        return token_ids

    assert detector.observe(separated_4grams(19)) is None
    truncation_length = detector.observe(separated_4grams(20))

    assert truncation_length == 99
    assert detector.matched_rule == (4, dict(DEFAULT_REPETITION_RULES)[4])


def test_custom_ngram_repetition_detector_keeps_single_rule_behavior():
    detector = NGramRepetitionDetector(ngram_size=4, max_count=2)
    ngram = [101, 102, 103, 104]

    assert detector.observe(ngram * 2) is None
    assert detector.observe(ngram * 3) == len(ngram) * 3


def test_default_detector_does_not_fall_back_on_length_only():
    detector = NGramRepetitionDetector(rules=[(16, 1000)])

    assert detector.observe(list(range(5000))) is None
    assert detector.matched_reason is None


def test_text_detector_truncates_low_zstd_ratio_repetition():
    text = ("the same phrase repeats here " * 40).encode("utf-8")
    token_ids = list(text)
    detector = NGramRepetitionDetector(
        rules=[(16, 1000)],
        token_to_bytes=lambda token_id: bytes([token_id]),
        text_check_every_tokens=16,
    )

    truncation_length = detector.observe(token_ids)

    assert truncation_length is not None
    assert truncation_length < len(token_ids)
    assert detector.matched_reason == "zstd_low_ratio"
    assert detector.matched_text_stats["zstd_ratio"] < 0.18


def test_text_detector_truncates_low_zstd_ratio_recent_window():
    text = (
        "This prefix contains enough normal mathematical prose before the model starts looping. " * 20
        + "gcd(a, ar) gcd(a, ar) gcd(a, ar) " * 80
    ).encode("utf-8")
    token_ids = list(text)
    detector = NGramRepetitionDetector(
        rules=[(16, 1000)],
        token_to_bytes=lambda token_id: bytes([token_id]),
        text_check_every_tokens=16,
        zstd_min_ratio=0.0,
        zstd_window_min_total_bytes=512,
        zstd_window_bytes=256,
        zstd_window_min_ratio=0.55,
        text_ngram_min_bytes=100_000,
    )

    truncation_length = detector.observe(token_ids)

    assert truncation_length is not None
    assert truncation_length < len(token_ids)
    assert detector.matched_reason == "zstd_window_low_ratio"
    assert detector.matched_text_stats["zstd_window_ratio"] < 0.55


def test_text_detector_does_not_truncate_high_zstd_ratio_by_default():
    text = "".join(chr(0x4E00 + ((i * 7919) % 0x5000)) for i in range(360)).encode("utf-8")
    token_ids = list(text)
    detector = NGramRepetitionDetector(
        rules=[(16, 1000)],
        token_to_bytes=lambda token_id: bytes([token_id]),
        text_check_every_tokens=16,
    )

    assert detector.observe(token_ids) is None
    assert detector.matched_reason is None

    old_high_ratio_detector = NGramRepetitionDetector(
        rules=[(16, 1000)],
        token_to_bytes=lambda token_id: bytes([token_id]),
        text_check_every_tokens=16,
        zstd_max_ratio=0.92,
    )
    assert old_high_ratio_detector.observe(token_ids) == 256
    assert old_high_ratio_detector.matched_reason == "zstd_high_ratio"


def test_text_detector_treats_mathematical_styled_letters_as_allowed_scripts():
    text = ("We use styled math letters such as 𝒙, 𝔸, ℝ, 𝛼, 𝜷, and 𝝅 in a normal derivation. " * 12).encode("utf-8")
    token_ids = list(text)
    detector = NGramRepetitionDetector(
        rules=[(16, 1000)],
        token_to_bytes=lambda token_id: bytes([token_id]),
        text_check_every_tokens=16,
        zstd_min_ratio=0.0,
        script_min_chars=1,
        script_min_suspicious_count=3,
        script_min_kinds=1,
        script_min_ratio=0.0,
    )

    assert detector.observe(token_ids) is None
    assert detector.matched_reason is None


def test_text_detector_truncates_multilingual_script_soup():
    text = ("We start by solving the equation carefully. " * 12 + "한жا日 " * 18).encode("utf-8")
    token_ids = list(text)
    detector = NGramRepetitionDetector(
        rules=[(16, 1000)],
        token_to_bytes=lambda token_id: bytes([token_id]),
        text_check_every_tokens=16,
        zstd_min_ratio=0.0,
        zstd_max_ratio=1.0,
    )

    truncation_length = detector.observe(token_ids)

    assert truncation_length is not None
    assert truncation_length < len(token_ids)
    assert detector.matched_reason == "script_mix"
    assert detector.matched_text_stats["script_counts"]["Hangul"] >= 3


def test_text_detector_truncates_late_multilingual_script_soup():
    text = (
        " ".join(f"prefix{index}" for index in range(4000)) + " " + ("한" * 4 + "ж" * 4 + "ا" * 4 + "ก" * 4 + " ") * 3
    ).encode("utf-8")
    token_ids = list(text)
    detector = NGramRepetitionDetector(
        rules=[(16, 1000)],
        token_to_bytes=lambda token_id: bytes([token_id]),
        text_check_every_tokens=16,
        zstd_min_ratio=0.0,
        zstd_max_ratio=1.0,
        zstd_window_min_ratio=0.0,
        script_min_chars=32,
        script_window_chars=128,
        text_ngram_min_bytes=100_000,
    )

    truncation_length = detector.observe(token_ids)

    assert truncation_length is not None
    assert truncation_length < len(token_ids)
    assert detector.matched_reason == "script_window_mix"
    assert detector.matched_text_stats["script_window_suspicious_counts"]["Hangul"] >= 3


def test_text_detector_truncates_dense_single_suspicious_script_window():
    text = (" ".join(f"prefix{index}" for index in range(400)) + " " + "ж" * 20).encode("utf-8")
    token_ids = list(text)
    detector = NGramRepetitionDetector(
        rules=[(16, 1000)],
        token_to_bytes=lambda token_id: bytes([token_id]),
        text_check_every_tokens=16,
        zstd_min_ratio=0.0,
        zstd_window_min_ratio=0.0,
        script_window_chars=256,
        text_ngram_min_bytes=100_000,
    )

    truncation_length = detector.observe(token_ids)

    assert truncation_length is not None
    assert truncation_length < len(token_ids)
    assert detector.matched_reason == "script_window_mix"
    assert detector.matched_text_stats["script_window_dense_suspicious_scripts"][0]["script"] == "Cyrillic"


def test_text_detector_truncates_long_text_ngram_repetition():
    text = (" ".join(f"prefix{index}" for index in range(300)) + "wait let us check the same case again " * 14).encode(
        "utf-8"
    )
    token_ids = list(text)
    detector = NGramRepetitionDetector(
        rules=[(16, 1000)],
        token_to_bytes=lambda token_id: bytes([token_id]),
        text_check_every_tokens=16,
        zstd_min_ratio=0.0,
        zstd_window_min_ratio=0.0,
        text_ngram_min_bytes=512,
        text_ngram_min_count=10,
        text_ngram_window_chars=1024,
    )

    truncation_length = detector.observe(token_ids)

    assert truncation_length is not None
    assert truncation_length < len(token_ids)
    assert detector.matched_reason == "text_ngram_repetition"
    assert detector.matched_text_stats["text_ngram_max_count"] >= 10
    assert detector.matched_text_stats["text_ngram_match"] == "waitletuscheckthesame"


def test_text_detector_records_reasoning_markers_without_truncating():
    text = (
        " ".join(f"prefix{index}" for index in range(300))
        + " "
        + "Wait, maybe this is wrong. Alternatively, let's start over. " * 8
    ).encode("utf-8")
    token_ids = list(text)
    detector = NGramRepetitionDetector(
        rules=[(16, 1000)],
        token_to_bytes=lambda token_id: bytes([token_id]),
        text_check_every_tokens=16,
        zstd_min_ratio=0.0,
        zstd_window_min_ratio=0.0,
        text_ngram_min_bytes=100_000,
        reasoning_marker_min_count=24,
        reasoning_marker_window_chars=1024,
    )

    assert detector.observe(token_ids) is None
    assert detector.matched_reason is None
    assert detector.last_text_stats["reasoning_marker_count"] >= 24


def test_text_detector_records_unclosed_think_without_truncating():
    text = (">" + " ".join(f"prefix{index}" for index in range(800))).encode("utf-8")
    token_ids = list(text)
    detector = NGramRepetitionDetector(
        rules=[(16, 1000)],
        token_to_bytes=lambda token_id: bytes([token_id]),
        text_check_every_tokens=16,
        zstd_min_ratio=0.0,
        zstd_window_min_ratio=0.0,
        text_ngram_min_bytes=100_000,
    )

    assert detector.observe(token_ids) is None
    assert detector.matched_reason is None
    assert detector.last_text_stats["starts_in_think"] is True
    assert detector.last_text_stats["seen_think_close"] is False


def test_text_detector_does_not_truncate_closed_think_by_format_rule():
    text = (
        ">"
        + " ".join(f"prefix{index}" for index in range(20))
        + "</think> final "
        + " ".join(f"suffix{index}" for index in range(300))
    ).encode("utf-8")
    token_ids = list(text)
    detector = NGramRepetitionDetector(
        rules=[(16, 1000)],
        token_to_bytes=lambda token_id: bytes([token_id]),
        text_check_every_tokens=16,
        zstd_min_ratio=0.0,
        zstd_window_min_ratio=0.0,
        text_ngram_min_bytes=100_000,
    )

    assert detector.observe(token_ids) is None
    assert detector.matched_reason is None


@pytest.mark.asyncio
async def test_consume_until_repetition_aborts_before_consuming_later_outputs():
    ngram = list(range(DEFAULT_REPETITION_NGRAM_SIZE))
    yielded_lengths: list[int] = []
    aborted: list[str] = []

    async def stream():
        for token_ids in (
            ngram * DEFAULT_REPETITION_MAX_COUNT,
            ngram * (DEFAULT_REPETITION_MAX_COUNT + 1) + [900],
            ngram * 20,
        ):
            yielded_lengths.append(len(token_ids))
            yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=token_ids)])

    async def abort_request():
        aborted.append("request-1")

    final_output, truncation_length, observed_token_ids = await consume_until_repetition(
        stream(),
        get_token_ids=lambda output: output.outputs[0].token_ids if output.outputs else [],
        abort_request=abort_request,
        detector=NGramRepetitionDetector(rules=[(DEFAULT_REPETITION_NGRAM_SIZE, DEFAULT_REPETITION_MAX_COUNT + 1)]),
        cumulative=True,
    )

    assert truncation_length == DEFAULT_REPETITION_NGRAM_SIZE * (DEFAULT_REPETITION_MAX_COUNT + 1)
    assert len(final_output.outputs[0].token_ids) == truncation_length + 1
    assert yielded_lengths == [
        DEFAULT_REPETITION_NGRAM_SIZE * DEFAULT_REPETITION_MAX_COUNT,
        DEFAULT_REPETITION_NGRAM_SIZE * (DEFAULT_REPETITION_MAX_COUNT + 1) + 1,
    ]
    assert aborted == ["request-1"]
    assert observed_token_ids == final_output.outputs[0].token_ids


@pytest.mark.asyncio
async def test_consume_until_repetition_detects_delta_token_stream():
    ngram = list(range(DEFAULT_REPETITION_NGRAM_SIZE))
    token_ids = ngram * (DEFAULT_REPETITION_MAX_COUNT + 1) + [900]
    yielded_count = 0
    aborted: list[str] = []

    async def stream():
        nonlocal yielded_count
        for token_id in token_ids:
            yielded_count += 1
            yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[token_id])])

    async def abort_request():
        aborted.append("request-1")

    final_output, truncation_length, observed_token_ids = await consume_until_repetition(
        stream(),
        get_token_ids=lambda output: output.outputs[0].token_ids if output.outputs else [],
        abort_request=abort_request,
        detector=NGramRepetitionDetector(rules=[(DEFAULT_REPETITION_NGRAM_SIZE, DEFAULT_REPETITION_MAX_COUNT + 1)]),
    )

    assert truncation_length == DEFAULT_REPETITION_NGRAM_SIZE * (DEFAULT_REPETITION_MAX_COUNT + 1)
    assert final_output.outputs[0].token_ids == [token_ids[truncation_length - 1]]
    assert observed_token_ids == token_ids[:truncation_length]
    assert yielded_count == truncation_length
    assert aborted == ["request-1"]


@pytest.mark.asyncio
async def test_consume_token_stream_merges_delta_chunks():
    async def stream():
        for token_ids in ([10], [20, 30], [40]):
            yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=token_ids)])

    final_output, observed_token_ids = await consume_token_stream(
        stream(),
        get_token_ids=lambda output: output.outputs[0].token_ids if output.outputs else [],
    )

    assert final_output.outputs[0].token_ids == [40]
    assert observed_token_ids == [10, 20, 30, 40]


@pytest.mark.asyncio
async def test_consume_token_stream_merges_only_new_cumulative_suffix():
    async def stream():
        for token_ids in ([10], [10, 20, 30], [10, 20, 30, 40]):
            yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=token_ids)])

    final_output, observed_token_ids = await consume_token_stream(
        stream(),
        get_token_ids=lambda output: output.outputs[0].token_ids if output.outputs else [],
        cumulative=True,
    )

    assert final_output.outputs[0].token_ids == [10, 20, 30, 40]
    assert observed_token_ids == [10, 20, 30, 40]
